"""core/visualizer.py - QC overlays and track visualisations. Matplotlib Agg backend."""
from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)


def generate_all_overlays(fov, frame_spots, tracks, population, config, masks_dir=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        log.warning("  matplotlib not installed - skipping visualisations")
        return
    overlay_dir = config.output_dir / "qc_overlays" / fov.fov_id
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if config.output.save_overlays:
        _seg_overlays(fov, frame_spots, overlay_dir, masks_dir)
    if config.output.save_tracks_viz and tracks:
        _track_overlay(fov, frame_spots, tracks, overlay_dir)


def generate_qc_summary_grid(fov_results, output_dir, max_fovs=25):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fov_ids = sorted(fov_results.keys())[:max_fovs]
    n = len(fov_ids)
    if n == 0:
        return
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4.5))
    fig.patch.set_facecolor("#0f0f1a")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    counts = []
    for fov_id in fov_ids:
        frame_spots, population = fov_results.get(fov_id, ([], None))
        if population:
            counts.append(population.cell_count_mean)
        elif frame_spots:
            first_frame = next((f for f in frame_spots if f), [])
            counts.append(len(first_frame))
        else:
            counts.append(0)
    median_count = float(np.median(counts)) if counts else 1.0
    for ax, fov_id, count in zip(axes_flat, fov_ids, counts):
        ax.set_facecolor("#1a1a2e")
        if count == 0:
            color, flag = "#FFD700", " [ZERO]"
        elif median_count > 0 and count > median_count * 3:
            color, flag = "#FF6B6B", " [HIGH]"
        else:
            color, flag = "#90EE90", ""
        ax.set_title(f"{fov_id}: {count:.0f} cells{flag}", color=color, fontsize=8, fontweight="bold")
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle("QC Summary - green=normal  red=high  yellow=zero cells", color="white", fontsize=10)
    plt.tight_layout()
    out_path = output_dir / "qc_summary_grid.png"
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    log.info(f"  QC summary grid: {out_path.name}")


def _seg_overlays(fov, frame_spots, overlay_dir, masks_dir):
    indices = sorted({0, len(fov.frames) // 2, len(fov.frames) - 1})
    indices = [i for i in indices if i < len(fov.frames)]
    for idx in indices:
        if idx >= len(frame_spots):
            continue
        frame = fov.frames[idx]
        spots = frame_spots[idx]
        img = _load_raw(frame.path)
        if img is None:
            continue
        mask = _load_mask(frame, fov, masks_dir)
        _single_overlay(img, mask, spots, frame, fov, overlay_dir)


def _single_overlay(img, mask, spots, frame, fov, overlay_dir):
    import matplotlib.pyplot as plt
    norm = _normalise(img)
    n_cells = len([s for s in spots if s.segmentation_ok and s.spot_id > 0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0f0f1a")
    axes[0].imshow(norm, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original", color="white", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(norm, cmap="gray", vmin=0, vmax=255)
    if mask is not None:
        axes[1].imshow(_outline_overlay(mask))
    axes[1].set_title(f"Cellpose outlines | {n_cells} cells detected", color="white", fontsize=11)
    axes[1].axis("off")
    fig.suptitle(
        f"{fov.fov_id} | Frame {frame.frame_index:04d} | t={frame.elapsed_min:.0f} min | {n_cells} cells",
        color="white", fontsize=9,
    )
    plt.tight_layout()
    out_path = overlay_dir / f"frame{frame.frame_index:04d}_overlay.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)


def _track_overlay(fov, frame_spots, tracks, overlay_dir):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    if not fov.frames:
        return
    last_frame = fov.frames[-1]
    img = _load_raw(last_frame.path)
    if img is None:
        return
    norm = _normalise(img)

    from collections import defaultdict
    track_spots = defaultdict(list)
    for frame_s in frame_spots:
        for s in frame_s:
            if s.track_id >= 0:
                track_spots[s.track_id].append((s.frame_index, s.centroid_x_px, s.centroid_y_px))

    n_tracks = len(track_spots)
    colors = _distinct_colors(n_tracks)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#0f0f1a")
    # Dim the raw background (multiply toward black) so colored tracks read
    # clearly against it - full-brightness background competes with track
    # colors, especially the darker end of the HSV sweep in _distinct_colors.
    dimmed = (norm.astype(np.float32) * 0.45).astype(np.uint8)
    ax.imshow(dimmed, cmap="gray", vmin=0, vmax=255)

    # At high track counts, thinner lines and lower peak alpha keep individual
    # trajectories distinguishable rather than blending into a solid mass.
    linewidth = 1.3 if n_tracks <= 50 else 0.8
    max_alpha = 0.9 if n_tracks <= 50 else 0.7
    marker_size = 5 if n_tracks <= 50 else 3

    all_lifespans = []
    all_displacements = []

    for i, (tid, pts) in enumerate(track_spots.items()):
        pts.sort(key=lambda p: p[0])
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        color = colors[i]

        if len(pts) >= 2:
            # Direction-of-travel: fade each segment's alpha from faint (early
            # in the track) to full (most recent), so the eye can read which
            # way a cell moved without needing to check start/end markers.
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_seg = len(segments)
            seg_alphas = np.linspace(max_alpha * 0.25, max_alpha, n_seg)
            seg_colors = [(*color, a) for a in seg_alphas]
            lc = LineCollection(segments, colors=seg_colors, linewidths=linewidth)
            ax.add_collection(lc)
        if pts:
            ax.plot(xs[0], ys[0], "o", color=color, markersize=marker_size, alpha=max_alpha)
            ax.plot(xs[-1], ys[-1], "s", color=color, markersize=marker_size, alpha=max_alpha)

        if tracks:
            match = next((t for t in tracks if t.track_id == tid), None)
            if match:
                all_lifespans.append(match.lifespan_min)
                all_displacements.append(match.total_displacement_um)

    ax.set_xlim(0, norm.shape[1])
    ax.set_ylim(norm.shape[0], 0)
    ax.set_title(f"{fov.fov_id} - {n_tracks} tracks | circle=start  square=end  faint\u2192bold=time",
                color="white", fontsize=10)
    ax.axis("off")

    if all_lifespans:
        stats_text = (
            f"mean lifespan: {np.mean(all_lifespans):.0f} min\n"
            f"mean displacement: {np.mean(all_displacements):.0f} \u00b5m"
        )
        ax.text(
            0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, color="white", verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f0f1a", edgecolor="#555555", alpha=0.85),
        )

    out_path = overlay_dir / "tracks_overlay.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    log.info(f"  Track overlay: {fov.fov_id}")


def _distinct_colors(n):
    """
    Generate n visually distinct colors by sweeping hue through HSV space,
    with alternating saturation/value bands so consecutive track indices
    (which are often spatially/temporally close) don't land on near-identical
    hues. Far more distinguishable at high track counts than cycling a
    fixed 20-color palette (e.g. tab20), which repeats every 20 tracks and
    makes dense fields look like a single blended color.
    """
    import colorsys
    if n <= 0:
        return []
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = 0.0
    for i in range(n):
        h = (h + golden_ratio_conjugate) % 1.0
        # Alternate saturation/value slightly so hue-adjacent tracks
        # (from the golden-ratio sequence landing close by chance) still
        # separate visually.
        s = 0.65 + 0.35 * ((i // 3) % 2)
        v = 0.85 + 0.15 * ((i // 5) % 2)
        colors.append(colorsys.hsv_to_rgb(h, s, v))
    return colors


def _load_raw(path):
    try:
        import tifffile
        img = tifffile.imread(str(path))
        if img.ndim == 3:
            img = img[0] if img.shape[0] <= 4 else img[:, :, 0]
        return img
    except Exception:
        return None


def _load_mask(frame, fov, masks_dir):
    if masks_dir is None:
        return None
    mask_path = masks_dir / fov.fov_id / f"{fov.fov_id}_frame{frame.frame_index:06d}_mask.tif"
    try:
        import tifffile
        return tifffile.imread(str(mask_path)) if mask_path.exists() else None
    except Exception:
        return None


def _normalise(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - p1) / (p99 - p1), 0, 1) if p99 > p1 else np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def _outline_overlay(mask):
    from scipy.ndimage import binary_dilation
    padded = np.pad(mask, 1, mode="edge")
    c = padded[1:-1, 1:-1]
    t = padded[0:-2, 1:-1]
    b = padded[2:, 1:-1]
    l = padded[1:-1, 0:-2]
    r = padded[1:-1, 2:]
    thin = (c > 0) & ((c != t) | (c != b) | (c != l) | (c != r))
    thick = binary_dilation(thin, structure=np.ones((3, 3))) & (c > 0)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    labels = np.unique(c[thick])
    labels = labels[labels != 0]
    # NOTE: previously cycled through only 3 hardcoded colors via
    # label % 3, so any two cells whose labels happened to differ by a
    # multiple of 3 (common with more than 3 cells in a frame - your real
    # data typically has 10-150+) received the identical outline color
    # and became visually indistinguishable in this QC overlay. Reusing
    # _distinct_colors() (already used correctly for the track overlay)
    # gives each label in this frame a genuinely distinct color instead.
    colors = _distinct_colors(len(labels))
    for i, label in enumerate(labels):
        cell_pts = thick & (c == label)
        overlay[cell_pts] = (*colors[i], 0.9)
    return overlay
