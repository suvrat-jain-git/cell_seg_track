"""
core/spot_loader.py - reloads a previous run's spots.csv back into SpotRecord
objects, grouped by FOV and frame, so tracking (and everything downstream of
it) can be re-run with different parameters without re-running Cellpose.

Segmentation is by far the most expensive step (>99% of total runtime in
every real run so far). Once spots.csv exists, sweeping tracking parameters
(--track_min_elapsed_min, --max_dist_px, --tracker_method, etc.) should cost
seconds, not another 30-60+ minutes per attempt.

This module intentionally reconstructs only what tracking and downstream
steps need: SpotRecord objects (grouped by fov_id then frame_index) and
minimal FieldOfView objects (fov_id, pixel_size_um, time_interval_min,
frame count/duration). Grid/chip metadata (grid_row, grid_col, condition
etc.) is NOT reconstructed here - if the HemaChip plugin's enrich_fovs()
needs to run again, that happens the same way it always does, by re-scanning
the original input_dir/EPF, which is cheap and doesn't depend on spots.csv.
"""
from __future__ import annotations
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import SpotRecord, FrameInfo, FieldOfView

log = logging.getLogger(__name__)


def load_spots_csv(spots_csv_path) -> list:
    """
    Load a spots.csv file back into a list of SpotRecord objects.

    Args:
        spots_csv_path: path to a spots.csv written by a previous run
                        (export_all() or export_fov_tracks())

    Returns:
        list[SpotRecord], track_id reset to -1 on every record regardless
        of what was saved, since the whole point of reloading is to
        re-track with (potentially) different parameters - keeping stale
        track_ids around would be misleading if not overwritten correctly
        downstream.

    Raises:
        FileNotFoundError if the file doesn't exist.
        ValueError if the file is missing required columns.
    """
    path = Path(spots_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"spots.csv not found: {path}")

    df = pd.read_csv(path)

    required = {"fov_id", "frame_index", "elapsed_min", "spot_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns for re-tracking: {missing}"
        )

    spots = []
    for row in df.itertuples(index=False):
        r = row._asdict()
        # is_viable is written to CSV as True/False/empty (None). pandas
        # reads an all-empty column as NaN (float), and a mixed True/NaN
        # column also comes back as an object/float mix - normalise all
        # of these back to a real Python True/False/None rather than
        # accidentally storing a numpy bool or NaN.
        raw_viable = r.get("is_viable", None)
        if pd.isna(raw_viable):
            is_viable = None
        else:
            is_viable = bool(raw_viable)

        spots.append(SpotRecord(
            fov_id         = r["fov_id"],
            frame_index    = int(r["frame_index"]),
            elapsed_min    = float(r["elapsed_min"]),
            spot_id        = int(r["spot_id"]),
            track_id       = -1,  # always reset - re-tracking will reassign
            centroid_x_px  = float(r.get("centroid_x_px", 0.0)),
            centroid_y_px  = float(r.get("centroid_y_px", 0.0)),
            centroid_x_um  = float(r.get("centroid_x_um", 0.0)),
            centroid_y_um  = float(r.get("centroid_y_um", 0.0)),
            area_px        = float(r.get("area_px", 0.0)),
            area_um2       = float(r.get("area_um2", 0.0)),
            perimeter_px   = float(r.get("perimeter_px", 0.0)),
            circularity    = float(r.get("circularity", 0.0)),
            eccentricity   = float(r.get("eccentricity", 0.0)),
            mean_intensity = float(r.get("mean_intensity", 0.0)),
            solidity       = float(r.get("solidity", 0.0)),
            # NOTE: previously missing entirely - equivalent_diameter_um and
            # is_viable are real SpotRecord fields written to spots.csv by
            # export_all(), but load_spots_csv() silently dropped both on
            # every reload, reconstructing with class defaults (0.0, None)
            # instead of the saved values. equivalent_diameter_um is always
            # computed during segmentation (not conditional on viability
            # being active), so this was unconditionally discarding real
            # per-cell diameter data on every retrack.py invocation.
            equivalent_diameter_um = float(r.get("equivalent_diameter_um", 0.0)),
            is_viable      = is_viable,
            segmentation_ok= bool(r.get("segmentation_ok", True)),
            error_msg      = str(r.get("error_msg", "")) if pd.notna(r.get("error_msg", "")) else "",
        ))

    log.info(f"  Loaded {len(spots)} spots from {path.name}")
    return spots


def group_by_fov_and_frame(spots: list) -> dict:
    """
    Group a flat spot list into the structure track_fov() expects:
    dict[fov_id -> list[list[SpotRecord]]] where the inner list is
    indexed by position (0-based, sorted by frame_index - matching how
    the live pipeline builds frame_spots from FieldOfView.frames).

    Returns:
        dict[fov_id -> (frame_spots, frame_index_map)]
        frame_spots      : list[list[SpotRecord]], one entry per unique
                            frame_index present for that FOV, sorted
        frame_index_map   : list[int], frame_index value at each position
                            in frame_spots (needed to reconstruct FrameInfo)
    """
    by_fov = defaultdict(lambda: defaultdict(list))
    for s in spots:
        by_fov[s.fov_id][s.frame_index].append(s)

    result = {}
    for fov_id, by_frame in by_fov.items():
        frame_indices = sorted(by_frame.keys())
        frame_spots = [by_frame[fi] for fi in frame_indices]
        result[fov_id] = (frame_spots, frame_indices)

    return result


def rebuild_fov(
    fov_id: str,
    frame_indices: list,
    pixel_size_um: float,
    time_interval_min: float,
    original_path=None,
) -> FieldOfView:
    """
    Rebuild a minimal FieldOfView for a previously-segmented FOV, sufficient
    for track_fov() / compute_population() / plugin.compute_fov_features().

    FrameInfo.path is set to a placeholder if original_path is not given -
    this is fine for re-tracking since no image files are re-read, but
    downstream visualisation (QC overlays that draw on the raw image) will
    need the real path. Pass original_path when overlays are also wanted
    on a re-track-only run.
    """
    frames = [
        FrameInfo(
            path=(Path(original_path) / f"frame{fi:06d}.tif") if original_path else Path(f"frame{fi:06d}.tif"),
            frame_index=fi,
            elapsed_min=fi * time_interval_min,
            fov_id=fov_id,
        )
        for fi in frame_indices
    ]
    return FieldOfView(
        path=Path(original_path) if original_path else Path("."),
        fov_id=fov_id,
        frames=frames,
        pixel_size_um=pixel_size_um,
        time_interval_min=time_interval_min,
    )


def load_run_for_retracking(prior_output_dir, pixel_size_um=None, time_interval_min=None) -> dict:
    """
    High-level convenience function: load a full prior run's spots.csv and
    rebuild everything needed to call track_fov() again per FOV.

    Args:
        prior_output_dir : output_dir of a previous pipeline run
        pixel_size_um    : override; if None, read from that run's
                            run_config.json
        time_interval_min: override; if None, read from that run's
                            run_config.json

    Returns:
        dict[fov_id -> (FieldOfView, list[list[SpotRecord]])]
        ready to pass directly into core.tracker.track_fov()
    """
    prior_output_dir = Path(prior_output_dir)
    spots_path = prior_output_dir / "spots.csv"
    spots = load_spots_csv(spots_path)
    grouped = group_by_fov_and_frame(spots)

    if pixel_size_um is None or time_interval_min is None:
        config_path = prior_output_dir / "run_config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            pixel_size_um = pixel_size_um or cfg.get("imaging", {}).get("pixel_size_um") or 0.377
            time_interval_min = time_interval_min or cfg.get("imaging", {}).get("time_interval_min") or 5.0
        else:
            pixel_size_um = pixel_size_um or 0.377
            time_interval_min = time_interval_min or 5.0
            log.warning(
                f"  No run_config.json found in {prior_output_dir} - "
                f"using defaults pixel_size_um={pixel_size_um}, "
                f"time_interval_min={time_interval_min}"
            )

    result = {}
    for fov_id, (frame_spots, frame_indices) in grouped.items():
        fov = rebuild_fov(fov_id, frame_indices, pixel_size_um, time_interval_min)
        result[fov_id] = (fov, frame_spots)

    log.info(f"  Rebuilt {len(result)} FOV(s) for re-tracking from {prior_output_dir.name}")
    return result
