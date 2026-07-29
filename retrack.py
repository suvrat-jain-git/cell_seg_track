"""
retrack.py - re-run tracking (and everything downstream of it) from a prior
pipeline run's spots.csv, without re-running Cellpose segmentation.

Segmentation is >99% of total pipeline runtime in every real run so far.
Once you have spots.csv from ANY completed run, sweeping tracking
parameters - --track_min_elapsed_min, --max_dist_px, --tracker_method,
--gap_dist_px, --min_track_frames - should take seconds, not another
30-60+ minutes per attempt.

Usage:
  python retrack.py \\
    --prior_output_dir /path/to/previous/run \\
    --output_dir /path/to/new/results \\
    --track_min_elapsed_min 30 \\
    --max_dist_px 20

  # Sweep multiple cutoffs in one go:
  python retrack.py \\
    --prior_output_dir /path/to/previous/run \\
    --output_dir /path/to/sweep_30 --track_min_elapsed_min 30
  python retrack.py \\
    --prior_output_dir /path/to/previous/run \\
    --output_dir /path/to/sweep_15 --track_min_elapsed_min 15

Limitations:
  - QC overlay images that draw on the raw microscope image (segmentation
    overlays) are NOT regenerated, since raw images aren't re-read. Pass
    --raw_input_dir if you want the track trajectory overlay (which only
    needs the LAST frame's raw image) to render - segmentation overlays
    specifically still require a full pipeline.py run.
  - Only works from a spots.csv that has centroid_x_px/centroid_y_px and
    the other fields the pipeline normally writes. A spots.csv edited or
    produced by anything else may be missing columns and will fail loudly
    rather than silently producing wrong tracks.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from config import (
    PipelineConfig, TrackingConfig, ImagingConfig, ExperimentType,
)


def setup_logging(level, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "retrack.log"
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w", encoding="utf-8")]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    return logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="CellFlow retrack - re-run tracking without re-segmenting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prior_output_dir", required=True,
                   help="output_dir of a previous pipeline.py run containing spots.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--raw_input_dir", default=None,
                   help="Optional: original session/ROI folder, enables track "
                        "trajectory overlay generation (needs the last raw frame)")

    p.add_argument("--pixel_size_um", type=float, default=None,
                   help="Override; default reads from prior run's run_config.json")
    p.add_argument("--time_interval_min", type=float, default=None)

    p.add_argument("--max_dist_px", type=float, default=50.0)
    p.add_argument("--max_dist_um", type=float, default=None)
    p.add_argument("--gap_dist_px", type=float, default=80.0)
    p.add_argument("--gap_frames", type=int, default=2)
    p.add_argument("--min_track_frames", type=int, default=3)
    p.add_argument("--tracker_method", choices=["lap", "nearest"], default="lap")
    p.add_argument("--viability_diameter_um", type=float, default=None,
                   help="Diameter threshold (um) for viability classification. "
                        "Re-classifying via retrack.py is cheap - useful for "
                        "sweeping candidate thresholds without re-segmenting.")
    p.add_argument("--track_min_elapsed_min", type=float, default=None,
                   help="Exclude frames before this elapsed time (minutes) from "
                        "tracking. Cell counts from the ORIGINAL run are unaffected "
                        "since segmentation is not re-run.")

    p.add_argument("--experiment", choices=["standard", "hemachip"], default="standard",
                   help="If hemachip, re-runs the plugin's compute_fov_features "
                        "(clinical features) using the new tracking result. Grid/chip "
                        "metadata (enrich_fovs) requires --raw_input_dir to re-scan "
                        "the EPF, since spots.csv alone doesn't carry it.")

    p.add_argument("--save_tracks_viz", action="store_true")
    p.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    return p.parse_args()


def main():
    a = parse_args()
    output_dir = Path(a.output_dir)
    log = setup_logging(a.log_level, output_dir)

    log.info("=" * 65)
    log.info("  CellFlow Retrack - tracking-only re-run (no segmentation)")
    log.info("=" * 65)
    log.info(f"  Prior run      : {a.prior_output_dir}")
    log.info(f"  Output dir     : {output_dir}")
    log.info(f"  Tracker method : {a.tracker_method}")
    log.info(f"  Track min elapsed : {a.track_min_elapsed_min} min")
    log.info("=" * 65)

    t_start = time.perf_counter()

    from core.spot_loader import load_run_for_retracking
    from core.tracker import track_fov
    from core.track_features import compute_population
    from core.exporter import export_fov_tracks
    from core.viability import classify_viability, summarise_viability

    fov_data = load_run_for_retracking(
        a.prior_output_dir,
        pixel_size_um=a.pixel_size_um,
        time_interval_min=a.time_interval_min,
    )
    if not fov_data:
        log.error("  No FOVs loaded from prior run - check --prior_output_dir")
        sys.exit(1)

    any_fov = next(iter(fov_data.values()))[0]
    pixel_size_um = any_fov.pixel_size_um
    time_interval_min = any_fov.time_interval_min

    max_dist_px = a.max_dist_px
    if a.max_dist_um is not None:
        max_dist_px = a.max_dist_um / pixel_size_um

    tracking_cfg = TrackingConfig(
        max_distance_px=max_dist_px, gap_closing_max_dist_px=a.gap_dist_px,
        gap_closing_max_frames=a.gap_frames, min_track_length_frames=a.min_track_frames,
        tracker_method=a.tracker_method, min_elapsed_min=a.track_min_elapsed_min,
    )

    plugin = None
    if a.experiment == "hemachip":
        try:
            fake_config = PipelineConfig(
                input_dir=a.raw_input_dir or ".", output_dir=output_dir,
                experiment_type=ExperimentType.HEMACHIP,
                imaging=ImagingConfig(pixel_size_um=pixel_size_um, time_interval_min=time_interval_min),
                tracking=tracking_cfg,
            )
            from plugins.hemachip.plugin import HemaChipPlugin
            plugin = HemaChipPlugin(fake_config)
            log.info("  Plugin: HemaChip loaded (clinical features will be recomputed)")
            if not a.raw_input_dir:
                log.warning(
                    "  --raw_input_dir not given - grid_row/grid_col/condition will "
                    "NOT be populated (clinical feature VALUES are unaffected, only "
                    "spatial metadata)"
                )
        except Exception as e:
            log.warning(f"  Could not load HemaChip plugin: {e}")

    all_spots, all_tracks, all_populations = [], [], []

    for i, (fov_id, (fov, frame_spots)) in enumerate(fov_data.items()):
        log.info(f"  [{i+1}/{len(fov_data)}] {fov_id} ({fov.n_frames} frames, {sum(len(f) for f in frame_spots)} spots)")

        spots_with_ids, track_records = track_fov(
            frame_spots=frame_spots, tracking_cfg=tracking_cfg,
            pixel_size_um=pixel_size_um, n_frames_total=fov.n_frames,
        )
        classify_viability(spots_with_ids, diameter_threshold_um=a.viability_diameter_um)
        viability_summary = summarise_viability(spots_with_ids)
        pop = compute_population(track_records, spots_with_ids, fov)
        pop.pct_viable = viability_summary["pct_viable"]
        pop.n_viable = viability_summary["n_viable"]
        pop.n_nonviable = viability_summary["n_nonviable"]

        if plugin:
            try:
                pop = plugin.compute_fov_features(spots_with_ids, track_records, pop, fov)
            except Exception as e:
                log.warning(f"    Plugin feature computation failed for {fov_id}: {e}")

        export_fov_tracks(track_records, spots_with_ids, fov_id, output_dir)

        all_spots.extend(spots_with_ids)
        all_tracks.extend(track_records)
        all_populations.append(pop)

        n_tracked = sum(1 for s in spots_with_ids if s.track_id >= 0)
        log.info(f"    {len(track_records)} tracks | {n_tracked} spots tracked")

    if a.save_tracks_viz and a.raw_input_dir:
        log.info("")
        log.info("Generating track overlays...")
        from core.visualizer import _track_overlay
        for fov_id, (fov, _) in fov_data.items():
            raw_fov_dir = Path(a.raw_input_dir) / fov_id
            if not raw_fov_dir.is_dir():
                raw_fov_dir = Path(a.raw_input_dir)
            last_frame_files = sorted(raw_fov_dir.glob("*.tif")) + sorted(raw_fov_dir.glob("*.tiff"))
            if not last_frame_files:
                log.warning(f"    No raw frames found for {fov_id} in {raw_fov_dir} - skipping overlay")
                continue
            fov.frames[-1].path = last_frame_files[-1]
            fov_spots = [s for s in all_spots if s.fov_id == fov_id]
            from collections import defaultdict
            by_frame = defaultdict(list)
            for s in fov_spots:
                by_frame[s.frame_index].append(s)
            frame_spots_for_viz = [by_frame.get(f.frame_index, []) for f in fov.frames]
            overlay_dir = output_dir / "qc_overlays" / fov_id
            overlay_dir.mkdir(parents=True, exist_ok=True)
            try:
                _track_overlay(fov, frame_spots_for_viz, [], overlay_dir)
            except Exception as e:
                log.warning(f"    Overlay failed for {fov_id}: {e}")
    elif a.save_tracks_viz and not a.raw_input_dir:
        log.warning("  --save_tracks_viz requires --raw_input_dir - skipping overlay generation")

    log.info("")
    log.info("Exporting results...")

    class _MinimalConfig:
        pass
    export_cfg = _MinimalConfig()
    export_cfg.output_dir = output_dir
    export_cfg.to_json = lambda path: None  # no meaningful PipelineConfig here to serialise

    written = {}
    import pandas as pd
    if all_spots:
        df = pd.DataFrame([s.to_dict() for s in all_spots])
        tmp = output_dir / "spots.tmp.csv"
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(output_dir / "spots.csv")
        written["spots.csv"] = len(df)
        log.info(f"  spots.csv            {len(df):>8} rows")
    if all_tracks:
        df = pd.DataFrame([t.to_dict() for t in all_tracks])
        tmp = output_dir / "tracks.tmp.csv"
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(output_dir / "tracks.csv")
        written["tracks.csv"] = len(df)
        log.info(f"  tracks.csv           {len(df):>8} rows")
    if all_populations:
        df = pd.DataFrame([p.to_dict() for p in all_populations])
        tmp = output_dir / "population.tmp.csv"
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(output_dir / "population.csv")
        written["population.csv"] = len(df)
        log.info(f"  population.csv       {len(df):>8} rows")

    if plugin:
        try:
            plugin.export(all_spots, all_tracks, all_populations, export_cfg)
        except Exception as e:
            log.warning(f"  Plugin export failed: {e}")

    if len(all_populations) >= 2:
        try:
            from core.analytics import generate_roi_dashboard
            generate_roi_dashboard(all_populations, output_dir, title_prefix=output_dir.name)
        except Exception as e:
            log.warning(f"  Analytics dashboard generation failed (non-fatal): {e}")

    total_time = time.perf_counter() - t_start
    log.info("")
    log.info("=" * 65)
    log.info("  RETRACK COMPLETE")
    log.info("=" * 65)
    log.info(f"  FOVs processed : {len(all_populations)}")
    log.info(f"  Total tracks   : {len(all_tracks)}")
    log.info(f"  Total time     : {total_time:.2f}s")
    log.info(f"  Output         : {output_dir}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
