"""
core/analytics.py - ROI-level analytics, one chart per PNG file, generated
once per multi-ROI run from a list of PopulationRecord objects.

Design note - cell-type / experiment agnosticism:
  Everything in _GENERIC_METRICS and every function that only reads
  fov_id/cell_count/tracking stats works for ANY cell type or experiment,
  HemaChip or not - CellFlow's core promise is that the engine doesn't
  assume a specific biology. _CLINICAL_METRICS (adhesion_rate,
  spreading_rate, endpoint_value, endpoint_variability) are HemaChip-
  specific in the sense that they're currently only populated by
  plugins/hemachip/plugin.py - but this module never imports or assumes
  anything about HemaChip directly. It checks for these fields via
  hasattr() and simply charts whatever is actually present on the
  PopulationRecord objects it's given. A future plugin for a different
  cell type/assay could attach its own metrics the same way and this
  module would chart them automatically, no changes needed here.
  Spatial/gradient heatmaps require grid_row/grid_col, which is a
  general "this experiment has a spatial layout" concept, not inherently
  HemaChip-specific either - any plugin that sets grid position gets
  heatmaps for free.

Each chart is its own figure, saved as its own PNG file - no multi-panel
composite figures. Chart type is chosen to fit what the metric actually
represents:

  Sequential / trend metrics (cell count, track count) -> line plot
  Percentage / proportion metrics (survival rate)        -> pie chart
  Per-ROI distributions (velocity, lifespan, clinical)   -> box plot
  Spatial layout (grid_row/grid_col present)             -> heatmap,
    with gradient source edges labelled when the caller identifies
    which edges gradient sources were applied to (generic mechanism -
    the source-edge labels are passed in, not hardcoded to any specific
    chemical or cell type)
  Two+ conditions present (e.g. two chips)               -> box+scatter
    comparison, one file per metric

All charts use the matplotlib Agg backend, safe for headless Colab
execution. This module never raises on missing data - it degrades to
whatever charts the available data supports and logs what was skipped.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── Metric catalogue ────────────────────────────────────────────────────────
# (attribute_name, display_label, unit_suffix, chart_kind)
# chart_kind: "line" | "pie" | "box"
# These are all CELL-TYPE AGNOSTIC - present on every PopulationRecord
# regardless of experiment or plugin.
_GENERIC_METRICS = [
    ("cell_count_mean",        "Mean Cell Count",           "",        "line"),
    ("n_tracks_total",         "Total Tracks",              "",        "line"),
    ("pct_tracks_survived",    "Tracks Surviving to End",   "%",       "pie"),
    ("mean_lifespan_min",      "Track Lifespan",            "min",     "box"),
    ("mean_velocity_um_min",   "Cell Velocity",             "\u00b5m/min", "box"),
    ("mean_confinement_ratio", "Confinement Ratio",         "",        "box"),
]

# Present only when a plugin attaches them dynamically (e.g. HemaChip).
# Checked via hasattr, never assumed - see module docstring.
_CLINICAL_METRICS = [
    ("adhesion_rate",         "Adhesion Rate",         "cells/min", "box"),
    ("spreading_rate",        "Spreading Rate",        "1/min",     "box"),
    ("endpoint_value",        "Endpoint Cell Count",   "cells",     "box"),
    ("endpoint_variability",  "Endpoint Variability",  "",          "box"),
]

# Present only when core/viability.py classification was run with a real
# threshold - see module docstring for the same agnosticism note.
_VIABILITY_METRICS = [
    ("pct_viable", "Viable Cell Percentage", "%", "pie"),
]

_DARK_BG = "#0f0f1a"
_PANEL_BG = "#1a1a2e"
_GRID_COLOR = "#333344"
_TEXT_COLOR = "#e8e8e8"
_ACCENT = "#4FC3F7"
_ACCENT2 = "#FF7043"
_PIE_COLORS = ["#4FC3F7", "#37474F"]


def generate_roi_dashboard(populations, output_dir, title_prefix="", gradient_edges=None):
    """
    Generate the full ROI-level analytics set. Writes one PNG per chart
    to output_dir/analytics/.

    Args:
        populations    : list[PopulationRecord], one per processed ROI
        output_dir     : pipeline output_dir - charts go in output_dir/analytics/
        title_prefix   : optional string prepended to chart titles
        gradient_edges : optional dict describing which chip edges a
                         gradient source was applied to, for labelling
                         spatial heatmaps. Generic mechanism, not tied to
                         any specific chemical/factor - e.g.:
                           {"top": "Source A", "right": "Source B",
                            "bottom": "Source C", "left": "Source D"}
                         Pass None (default) for no edge labels - the
                         heatmap is still generated, just unlabelled.

    Returns:
        list[Path] of files written (empty if matplotlib unavailable or
        no populations given)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("  matplotlib not installed - skipping analytics dashboard")
        return []

    if not populations:
        log.warning("  No population data - skipping analytics dashboard")
        return []

    analytics_dir = Path(output_dir) / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    written = []

    metrics = _available_metrics(populations)
    if not metrics:
        log.warning("  No chartable metrics found on population records")
    else:
        for attr, label, unit, kind in metrics:
            path = _chart_for_metric(populations, attr, label, unit, kind, analytics_dir, title_prefix)
            if path:
                written.append(path)

    has_grid = _has_spatial_layout(populations)
    if has_grid:
        written.extend(_spatial_heatmaps(populations, analytics_dir, title_prefix, gradient_edges))

        conditions = sorted(set(
            getattr(p, "condition", "") for p in populations if getattr(p, "condition", "")
        ))
        if len(conditions) >= 2:
            written.extend(_condition_comparisons(populations, conditions, metrics, analytics_dir, title_prefix))
        elif len(conditions) == 1:
            log.info(
                f"  Only one chip condition present ({conditions[0]}) - "
                f"skipping gradient-vs-homogeneous comparison (needs 2+ chips)"
            )
    else:
        log.info("  No grid position data - skipping spatial heatmaps (plugin did not set grid_row/grid_col)")

    written = [w for w in written if w is not None]
    log.info(f"  Analytics dashboard: {len(written)} chart(s) written to {analytics_dir}")
    return written


# ── Metric availability ─────────────────────────────────────────────────────

def _available_metrics(populations):
    available = [m for m in _GENERIC_METRICS if all(hasattr(p, m[0]) for p in populations)]
    available += [m for m in _CLINICAL_METRICS if all(hasattr(p, m[0]) for p in populations)]
    available += [m for m in _VIABILITY_METRICS
                  if all(hasattr(p, m[0]) for p in populations)
                  and any(getattr(p, m[0]) is not None for p in populations)]
    return available


def _has_spatial_layout(populations):
    return len(populations) > 1 and all(hasattr(p, "grid_row") for p in populations) and \
           any((p.grid_row != 0 or p.grid_col != 0) for p in populations)


# ── Per-metric chart dispatch ───────────────────────────────────────────────

def _chart_for_metric(populations, attr, label, unit, kind, out_dir, title_prefix):
    if kind == "line":
        return _line_chart(populations, attr, label, unit, out_dir, title_prefix)
    if kind == "pie":
        return _pie_chart(populations, attr, label, unit, out_dir, title_prefix)
    if kind == "box":
        return _box_chart(populations, attr, label, unit, out_dir, title_prefix)
    log.warning(f"  Unknown chart kind '{kind}' for metric {attr} - skipped")
    return None


def _line_chart(populations, attr, label, unit, out_dir, title_prefix):
    """Sequential/trend metrics (cell count, track count) across ROIs,
    sorted by fov_id - shows the shape of variation across the field
    layout, e.g. a density gradient."""
    import matplotlib.pyplot as plt

    pops = sorted(populations, key=lambda p: p.fov_id)
    fov_ids = [p.fov_id for p in pops]
    values = [getattr(p, attr, 0.0) for p in pops]
    n = len(fov_ids)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 5))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    ax.plot(range(n), values, "-o", color=_ACCENT, markersize=5, linewidth=1.6)

    if n >= 4:
        mu, sigma = np.mean(values), np.std(values)
        if sigma > 1e-9:
            for i, v in enumerate(values):
                if abs(v - mu) > 2 * sigma:
                    ax.plot(i, v, "o", color=_ACCENT2, markersize=8, zorder=3)

    ax.set_xticks(range(n))
    ax.set_xticklabels(fov_ids, color=_TEXT_COLOR, fontsize=8,
                       rotation=45 if n > 8 else 0, ha="right" if n > 8 else "center")
    ax.set_ylabel(f"{label}{f' ({unit})' if unit else ''}", color=_TEXT_COLOR, fontsize=10)
    title = f"{title_prefix} - {label}" if title_prefix else label
    ax.set_title(f"{title} across {n} ROIs", color=_TEXT_COLOR, fontsize=13)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_GRID_COLOR)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = out_dir / f"{attr}_trend.png"
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return out_path


def _pie_chart(populations, attr, label, unit, out_dir, title_prefix):
    """Proportion metrics (survival rate, viable percentage) - averaged
    across all ROIs and shown as a single donut, since a per-ROI pie
    grid would be visually noisy at scale (10-225 ROIs)."""
    import matplotlib.pyplot as plt

    values = [getattr(p, attr, None) for p in populations]
    values = [v for v in values if v is not None]
    if not values:
        return None
    mean_val = float(np.mean(values))
    mean_val = max(0.0, min(100.0, mean_val))

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(_DARK_BG)

    ax.pie(
        [mean_val, 100 - mean_val],
        colors=_PIE_COLORS,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor=_DARK_BG, linewidth=2),
    )
    ax.text(0, 0.05, f"{mean_val:.1f}%", ha="center", va="center",
           color=_TEXT_COLOR, fontsize=26, fontweight="bold")
    ax.text(0, -0.15, label, ha="center", va="center", color=_TEXT_COLOR, fontsize=11)

    title = f"{title_prefix} - {label}" if title_prefix else label
    ax.set_title(f"{title} (mean across {len(values)} ROIs)", color=_TEXT_COLOR, fontsize=12, pad=20)

    plt.tight_layout()
    out_path = out_dir / f"{attr}_proportion.png"
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return out_path


def _box_chart(populations, attr, label, unit, out_dir, title_prefix):
    """Distribution metrics - single box+scatter showing spread across
    all ROIs, rather than one bar per ROI, so the overall distribution
    shape (median, spread, outliers) is visible at a glance."""
    import matplotlib.pyplot as plt

    values = [getattr(p, attr, 0.0) for p in populations]
    n = len(values)

    fig, ax = plt.subplots(figsize=(5, 6))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    bp = ax.boxplot(
        [values], positions=[0], widths=0.5, patch_artist=True,
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=_GRID_COLOR), capprops=dict(color=_GRID_COLOR),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=_ACCENT2, markeredgecolor="none"),
    )
    bp["boxes"][0].set_facecolor(_ACCENT)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][0].set_edgecolor("none")

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, size=n)
    ax.scatter(jitter, values, color="white", s=14, alpha=0.55, zorder=3, linewidths=0)

    ax.set_xticks([0])
    ax.set_xticklabels([f"n={n} ROIs"], color=_TEXT_COLOR, fontsize=9)
    ax.set_ylabel(f"{label}{f' ({unit})' if unit else ''}", color=_TEXT_COLOR, fontsize=10)
    title = f"{title_prefix} - {label}" if title_prefix else label
    ax.set_title(title, color=_TEXT_COLOR, fontsize=13)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_GRID_COLOR)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = out_dir / f"{attr}_distribution.png"
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return out_path


# ── Spatial heatmaps ─────────────────────────────────────────────────────────

def _spatial_heatmaps(populations, out_dir, title_prefix, gradient_edges):
    """One heatmap PNG per metric per chip condition, shaped like the
    actual chip grid. gradient_edges (if given) annotates which edges a
    gradient source was applied to - a generic labelling mechanism, not
    tied to any specific chemical/factor."""
    import matplotlib.pyplot as plt

    written = []
    conditions = sorted(set(getattr(p, "condition", "") for p in populations if getattr(p, "condition", "")))
    if not conditions:
        conditions = [""]

    metrics = _available_metrics(populations)
    # NOTE: previously excluded pie-kind metrics (pct_tracks_survived,
    # pct_viable) from spatial heatmaps entirely - but "is survival/
    # viability spatially non-uniform across the chip" is exactly the
    # kind of question a heatmap directly answers, and a heatmap of a
    # 0-100% metric needs no special handling versus any other numeric
    # metric. Found during codebase audit alongside the same gap in
    # condition comparisons (see _condition_comparisons above).
    spatial_metrics = metrics
    if not spatial_metrics:
        return written

    for condition in conditions:
        group = [p for p in populations if getattr(p, "condition", "") == condition] if condition else populations
        if len(group) < 2:
            continue

        max_row = max(p.grid_row for p in group) + 1
        max_col = max(p.grid_col for p in group) + 1
        if max_row * max_col < 2:
            continue

        for attr, label, unit, kind in spatial_metrics:
            grid = np.full((max_row, max_col), np.nan)
            for p in group:
                if 0 <= p.grid_row < max_row and 0 <= p.grid_col < max_col:
                    grid[p.grid_row, p.grid_col] = getattr(p, attr, np.nan)

            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            fig.patch.set_facecolor(_DARK_BG)
            ax.set_facecolor(_PANEL_BG)

            im = ax.imshow(grid, cmap="magma", aspect="equal", interpolation="nearest")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR, labelsize=9)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_COLOR)

            cond_label = condition or "chip"
            title = f"{title_prefix} - {label} ({cond_label})" if title_prefix else f"{label} ({cond_label})"
            ax.set_title(title, color=_TEXT_COLOR, fontsize=13)
            ax.set_xlabel("Column", color=_TEXT_COLOR, fontsize=9)
            ax.set_ylabel("Row", color=_TEXT_COLOR, fontsize=9)
            ax.tick_params(colors=_TEXT_COLOR, labelsize=8)

            if gradient_edges:
                _annotate_gradient_edges(ax, gradient_edges)

            plt.tight_layout()
            safe_label = (condition or "all").replace(" ", "_")
            out_path = out_dir / f"heatmap_{attr}_{safe_label}.png"
            plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_DARK_BG)
            plt.close(fig)
            written.append(out_path)

    return written


def _annotate_gradient_edges(ax, gradient_edges):
    """
    Labels the edges of a spatial heatmap with gradient source names.
    Generic mechanism: gradient_edges is a dict with any subset of keys
    "top", "bottom", "left", "right" mapping to a display string - the
    caller decides what those sources are, this function only places
    the labels. Not specific to any chemical, factor, or cell type.
    """
    style = dict(color=_ACCENT2, fontsize=10, fontweight="bold")
    if "top" in gradient_edges:
        ax.annotate(f"\u2191 {gradient_edges['top']}", xy=(0.5, 1.04), xycoords="axes fraction",
                    ha="center", va="bottom", **style)
    if "bottom" in gradient_edges:
        ax.annotate(f"\u2193 {gradient_edges['bottom']}", xy=(0.5, -0.10), xycoords="axes fraction",
                    ha="center", va="top", **style)
    if "left" in gradient_edges:
        ax.annotate(f"\u2190 {gradient_edges['left']}", xy=(-0.16, 0.5), xycoords="axes fraction",
                    ha="right", va="center", rotation=90, **style)
    if "right" in gradient_edges:
        ax.annotate(f"{gradient_edges['right']} \u2192", xy=(1.12, 0.5), xycoords="axes fraction",
                    ha="left", va="center", rotation=90, **style)


# ── Condition comparison (one file per metric) ─────────────────────────────

def _condition_comparisons(populations, conditions, metrics, out_dir, title_prefix):
    import matplotlib.pyplot as plt

    groups = {c: [p for p in populations if getattr(p, "condition", "") == c] for c in conditions}
    groups = {c: g for c, g in groups.items() if g}
    if len(groups) < 2:
        return []

    condition_colors = _distinct_condition_colors(len(groups))
    written = []

    for attr, label, unit, kind in metrics:
        # NOTE: pie-kind metrics (pct_tracks_survived, pct_viable) were
        # previously skipped entirely here with no comparison chart ever
        # generated for them - even though "does survival/viability rate
        # differ between chip conditions" is exactly the kind of question
        # this feature exists to answer, and a box plot compares 0-100%
        # distributions across groups just as well as any other metric.
        # This was a real feature gap found during codebase audit, not a
        # deliberate design choice - box plots need no special handling
        # for percentage-scaled data.

        fig, ax = plt.subplots(figsize=(6, 5.5))
        fig.patch.set_facecolor(_DARK_BG)
        ax.set_facecolor(_PANEL_BG)

        positions, data, labels = [], [], []
        for i, (cond, group) in enumerate(groups.items()):
            values = [getattr(p, attr, 0.0) for p in group]
            data.append(values)
            labels.append(f"{cond}\n(n={len(values)})")
            positions.append(i)

        bp = ax.boxplot(
            data, positions=positions, widths=0.5, patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5),
            whiskerprops=dict(color=_GRID_COLOR), capprops=dict(color=_GRID_COLOR),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=_ACCENT, markeredgecolor="none"),
        )
        for patch, color in zip(bp["boxes"], condition_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("none")

        rng = np.random.default_rng(0)
        for i, values in enumerate(data):
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values,
                      color="white", s=12, alpha=0.5, zorder=3, linewidths=0)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, color=_TEXT_COLOR, fontsize=10)
        ax.set_ylabel(f"{label}{f' ({unit})' if unit else ''}", color=_TEXT_COLOR, fontsize=10)
        title = f"{title_prefix} - {label} by Condition" if title_prefix else f"{label} by Condition"
        ax.set_title(title, color=_TEXT_COLOR, fontsize=12)
        ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(_GRID_COLOR)
        ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out_path = out_dir / f"comparison_{attr}.png"
        plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        written.append(out_path)

    return written


def _distinct_condition_colors(n):
    palette = ["#4FC3F7", "#FF7043", "#66BB6A", "#AB47BC", "#FFCA28", "#EC407A"]
    if n <= len(palette):
        return palette[:n]
    import colorsys
    return [colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.7, 0.9) for i in range(n)]
