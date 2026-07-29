"""
CellFlow Pipeline v1.0 - startup-grade single-cell tracking platform.

Usage:
  python pipeline.py --input_dir /path/to/ROI --output_dir /path/to/results \
    --pixel_size_um 0.3769 --time_interval_min 5.0 --model cyto3 \
    --frame_mode all --save_masks --save_overlays --save_tracks_viz
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from config import (
    PipelineConfig, SegmentationConfig, TrackingConfig,
    ImagingConfig, OutputConfig, FrameMode, ExperimentType, LogLevel
)


def setup_logging(level, output_dir, resume=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    # On a resumed run, APPEND to the existing log rather than overwriting
    # it - a resumed run is, by definition, continuing after an earlier
    # interruption, and destroying the log of what happened before that
    # interruption is exactly the wrong behaviour for the one scenario
    # --resume exists to support (long unattended runs where you need to
    # know what happened before things stopped).
    file_mode = "a" if resume else "w"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    log = logging.getLogger(__name__)
    if resume:
        log.info("")
        log.info("=" * 65)
        log.info("  RESUMING - appending to existing pipeline.log")
        log.info("=" * 65)
    return log


def parse_args():
    p = argparse.ArgumentParser(description="CellFlow - Single Cell Tracking Pipeline",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pixel_size_um", type=float, default=None)
    p.add_argument("--time_interval_min", type=float, default=None)
    p.add_argument("--expected_width_px", type=int, default=1600)
    p.add_argument("--expected_height_px", type=int, default=1600)
    p.add_argument("--model", default="cyto3")
    p.add_argument("--diameter", type=float, default=None)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--flow_threshold", type=float, default=0.4)
    p.add_argument("--cellprob_threshold", type=float, default=0.0)
    p.add_argument("--min_diam_um", type=float, default=4.0)
    p.add_argument("--max_diam_um", type=float, default=30.0)
    p.add_argument("--viability_diameter_um", type=float, default=None,
                   help="Diameter threshold (um) below which cells are classified "
                        "non-viable (putative shrunken/fragmented/apoptotic). Not "
                        "set by default - see core/viability.py for why this is not "
                        "auto-populated from the source HSEC paper's own 9.5um "
                        "value. Validate against your own images before setting.")
    p.add_argument("--tracker_method", choices=["lap", "nearest"], default="lap",
                   help="lap=laptrack LAP algorithm (default, robust). "
                        "nearest=published HSEC methodology (simple radius match, "
                        "no gap closing) - use for direct comparability with "
                        "published HSEC results.")
    p.add_argument("--max_dist_px", type=float, default=50.0,
                   help="Max tracking distance in pixels. Ignored if --max_dist_um is set.")
    p.add_argument("--max_dist_um", type=float, default=None,
                   help="Max tracking distance in microns (converted to pixels using "
                        "--pixel_size_um). Published HSEC methodology uses 150um.")
    p.add_argument("--gap_dist_px", type=float, default=80.0)
    p.add_argument("--gap_frames", type=int, default=2)
    p.add_argument("--min_track_frames", type=int, default=3)
    p.add_argument("--detect_division", action="store_true")
    p.add_argument("--track_min_elapsed_min", type=float, default=None,
                   help="Exclude frames before this elapsed time (minutes) from "
                        "TRACKING only - segmentation and cell counts still include "
                        "them. Use this to skip the suspension phase, where cells "
                        "move with flow and are not meaningfully trackable at any "
                        "distance threshold. e.g. 120 to track only adhered cells.")
    p.add_argument("--frame_mode", choices=["all", "first", "median", "clinical"], default="all")
    p.add_argument("--experiment", choices=["standard", "hemachip"], default="standard")
    p.add_argument("--save_masks", action="store_true")
    p.add_argument("--save_overlays", action="store_true")
    p.add_argument("--save_tracks_viz", action="store_true")
    p.add_argument("--n_workers", type=int, default=1)
    p.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fov_pattern", default="ROI-*")

    a = p.parse_args()

    max_dist_px = a.max_dist_px
    if a.max_dist_um is not None:
        if not a.pixel_size_um:
            p.error("--max_dist_um requires --pixel_size_um to convert to pixels")
        max_dist_px = a.max_dist_um / a.pixel_size_um

    # run_id must be DETERMINISTIC (derived from output_dir), not random.
    # JobManager looks for a checkpoint file named run_{run_id}.json - if
    # run_id were randomised per invocation (as it previously was), running
    # the same command twice against the same --output_dir would generate a
    # different run_id each time, so the checkpoint from the first run could
    # never be found by the second - --resume would silently do nothing.
    # Hashing output_dir's absolute path gives the same run_id every time
    # the same output location is used, which is what --resume actually needs.
    import hashlib
    run_id = hashlib.md5(str(Path(a.output_dir).resolve()).encode()).hexdigest()[:8]

    return PipelineConfig(
        input_dir=Path(a.input_dir), output_dir=Path(a.output_dir),
        frame_mode=FrameMode(a.frame_mode), experiment_type=ExperimentType(a.experiment),
        n_workers=a.n_workers, log_level=LogLevel(a.log_level), run_id=run_id,
        segmentation=SegmentationConfig(
            model_name=a.model, diameter=a.diameter, use_gpu=a.gpu,
            flow_threshold=a.flow_threshold, cellprob_threshold=a.cellprob_threshold,
            min_diameter_um=a.min_diam_um, max_diameter_um=a.max_diam_um,
        ),
        tracking=TrackingConfig(
            max_distance_px=max_dist_px, gap_closing_max_dist_px=a.gap_dist_px,
            gap_closing_max_frames=a.gap_frames, min_track_length_frames=a.min_track_frames,
            detect_division=a.detect_division, tracker_method=a.tracker_method,
            min_elapsed_min=a.track_min_elapsed_min,
        ),
        imaging=ImagingConfig(
            pixel_size_um=a.pixel_size_um, time_interval_min=a.time_interval_min,
            expected_width_px=a.expected_width_px, expected_height_px=a.expected_height_px,
        ),
        output=OutputConfig(save_masks=a.save_masks, save_overlays=a.save_overlays, save_tracks_viz=a.save_tracks_viz),
    ), a.fov_pattern, a.resume, a.viability_diameter_um


def print_banner(config, log):
    log.info("=" * 65)
    log.info("  CellFlow  Single Cell Tracking Platform  v1.0")
    log.info("=" * 65)
    log.info(f"  Input dir      : {config.input_dir}")
    log.info(f"  Output dir     : {config.output_dir}")
    log.info(f"  Experiment     : {config.experiment_type.value}")
    log.info(f"  Frame mode     : {config.frame_mode.value}")
    log.info(f"  Model          : {config.segmentation.model_name}")
    log.info(f"  Diameter       : {config.segmentation.diameter or 'auto'} px")
    log.info(f"  GPU            : {config.segmentation.use_gpu}")
    log.info(f"  Pixel size     : {config.imaging.pixel_size_um or 'auto'} um/px")
    log.info(f"  Time interval  : {config.imaging.time_interval_min or 'auto'} min")
    log.info(f"  Size filter    : {config.segmentation.min_diameter_um}-{config.segmentation.max_diameter_um} um")
    log.info(f"  Track max dist : {config.tracking.max_distance_px} px")
    log.info(f"  Gap closing    : {config.tracking.gap_closing_max_frames} frames")
    log.info(f"  Min track len  : {config.tracking.min_track_length_frames} frames")
    log.info(f"  Workers        : {config.n_workers}")
    log.info("=" * 65)


def main():
    config, fov_pattern, resume, viability_threshold = parse_args()
    log = setup_logging(config.log_level.value, config.output_dir, resume=resume)
    timing = {}

    print_banner(config, log)
    pipeline_start = time.perf_counter()

    plugin = _load_plugin(config, log)

    t0 = time.perf_counter()
    log.info("")
    log.info("STEP 1/5 - Discovering fields of view...")
    from core.input_handler import load_fovs_from_batch, load_fov

    tifs_in_dir = list(config.input_dir.glob("*.tif")) + list(config.input_dir.glob("*.tiff"))
    if tifs_in_dir:
        log.info("  Single FOV mode detected")
        fov = load_fov(config.input_dir, config.input_dir.name, config)
        fovs = [fov] if fov else []
    else:
        log.info("  Batch mode detected")
        fovs = load_fovs_from_batch(config.input_dir, config, fov_pattern)

    if plugin:
        fovs = plugin.enrich_fovs(fovs, config.input_dir)

    total_frames = sum(f.n_frames for f in fovs)
    log.info(f"  FOVs found     : {len(fovs)}")
    log.info(f"  Total frames   : {total_frames}")
    timing["discover"] = time.perf_counter() - t0
    log.info(f"  [OK] Discovery complete ({timing['discover']:.2f}s)")

    if not fovs:
        log.error("  No FOVs found. Check --input_dir and --fov_pattern.")
        sys.exit(1)

    t0 = time.perf_counter()
    log.info("")
    log.info("STEP 2+3/5 - Segmentation + Tracking...")
    from core.segmentor import segment_fov, load_model
    from core.tracker import track_fov
    from core.track_features import compute_population
    from core.exporter import export_fov_tracks
    from core.viability import classify_viability, summarise_viability
    from job_manager import JobManager

    job = JobManager(config)

    if job.completed_fovs and not resume:
        log.error(
            f"  Found an existing checkpoint for this output_dir with "
            f"{len(job.completed_fovs)} FOV(s) already complete, but --resume "
            f"was not passed. Refusing to proceed to avoid ambiguity between "
            f"'start fresh' and 'continue'. Either:\n"
            f"    - pass --resume to continue from the checkpoint, or\n"
            f"    - use a different --output_dir to start a fresh run, or\n"
            f"    - delete {config.checkpoint_dir} to discard the old checkpoint"
        )
        sys.exit(1)
    elif job.completed_fovs and resume:
        log.info(f"  --resume: continuing from checkpoint ({len(job.completed_fovs)} FOV(s) already complete)")

    if sys.platform == "win32" and config.n_workers > 1:
        log.info("  Windows: forcing single-process mode")
        config.n_workers = 1

    model = load_model(config.segmentation)
    all_spots, all_tracks, all_populations = [], [], []

    for i, fov in enumerate(fovs):
        log.info(f"  [{i+1}/{len(fovs)}] {fov.fov_id} ({fov.n_frames} frames)")
        if job.is_completed(fov.fov_id):
            log.info("    Skipping - already complete (resume mode)")
            spots, tracks, pop = job.load_fov_results(fov.fov_id)
            if spots:
                all_spots.extend(spots)
            if tracks:
                all_tracks.extend(tracks)
            if pop:
                all_populations.append(pop)
            continue
        try:
            frame_spots = segment_fov(fov, config, model)
            spots_with_ids, track_records = track_fov(
                frame_spots=frame_spots, tracking_cfg=config.tracking,
                pixel_size_um=fov.pixel_size_um, n_frames_total=fov.n_frames,
            )
            classify_viability(spots_with_ids, diameter_threshold_um=viability_threshold)
            viability_summary = summarise_viability(spots_with_ids)
            pop = compute_population(track_records, spots_with_ids, fov)
            pop.pct_viable = viability_summary["pct_viable"]
            pop.n_viable = viability_summary["n_viable"]
            pop.n_nonviable = viability_summary["n_nonviable"]
            if plugin:
                pop = plugin.compute_fov_features(spots_with_ids, track_records, pop, fov)
            export_fov_tracks(track_records, spots_with_ids, fov.fov_id, config.output_dir)
            all_spots.extend(spots_with_ids)
            all_tracks.extend(track_records)
            all_populations.append(pop)
            job.mark_complete(fov.fov_id, spots_with_ids, track_records, pop)
            n_cells = sum(1 for s in spots_with_ids if s.segmentation_ok)
            log.info(f"    [OK] {n_cells} detections | {len(track_records)} tracks")
        except Exception as e:
            log.error(f"    [FAIL] {fov.fov_id}: {e}")
            job.mark_failed(fov.fov_id, str(e))
            import traceback
            log.debug(traceback.format_exc())

    timing["segment_track"] = time.perf_counter() - t0
    log.info(f"  [OK] Segment+Track complete ({timing['segment_track']:.1f}s)")

    if config.output.save_overlays or config.output.save_tracks_viz:
        t0 = time.perf_counter()
        log.info("")
        log.info("STEP 4/5 - Generating QC visualisations...")
        from core.visualizer import generate_all_overlays, generate_qc_summary_grid
        masks_dir = config.output_dir / "masks"
        spots_by_fov = {}
        for s in all_spots:
            spots_by_fov.setdefault(s.fov_id, []).append(s)
        qc_summary_data = {}
        for fov in fovs[:config.output.overlay_max_fovs]:
            fov_spots = spots_by_fov.get(fov.fov_id, [])
            fov_tracks = [t for t in all_tracks if t.fov_id == fov.fov_id]
            fov_pop = next((p for p in all_populations if p.fov_id == fov.fov_id), None)
            from collections import defaultdict
            by_frame = defaultdict(list)
            for s in fov_spots:
                by_frame[s.frame_index].append(s)
            frame_spots = [by_frame.get(f.frame_index, []) for f in fov.frames]
            generate_all_overlays(
                fov=fov, frame_spots=frame_spots, tracks=fov_tracks, population=fov_pop,
                config=config, masks_dir=masks_dir if config.output.save_masks else None,
            )
            qc_summary_data[fov.fov_id] = (frame_spots, fov_pop)

        # An at-a-glance grid of every processed FOV's cell count, flagging
        # zero-cell and unusually-high-density ROIs - only meaningful for a
        # batch run with more than one FOV.
        if len(qc_summary_data) > 1:
            generate_qc_summary_grid(qc_summary_data, config.output_dir)

        timing["visualise"] = time.perf_counter() - t0
        log.info(f"  [OK] Visualisation complete ({timing['visualise']:.1f}s)")

    t0 = time.perf_counter()
    log.info("")
    log.info("STEP 5/5 - Exporting results...")
    from core.exporter import export_all
    export_all(all_spots, all_tracks, all_populations, config, timing)
    if plugin:
        plugin.export(all_spots, all_tracks, all_populations, config)

    if len(all_populations) >= 2:
        try:
            from core.analytics import generate_roi_dashboard
            generate_roi_dashboard(all_populations, config.output_dir, title_prefix=config.input_dir.name)
        except Exception as e:
            log.warning(f"  Analytics dashboard generation failed (non-fatal): {e}")
    else:
        log.info("  Skipping analytics dashboard - needs 2+ ROIs for a meaningful comparison")

    timing["export"] = time.perf_counter() - t0
    log.info(f"  [OK] Export complete ({timing['export']:.2f}s)")

    total_time = time.perf_counter() - pipeline_start
    timing["total"] = total_time
    n_ok = sum(1 for s in all_spots if s.segmentation_ok)
    n_fail = sum(1 for s in all_spots if not s.segmentation_ok)

    log.info("")
    log.info("=" * 65)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 65)
    log.info(f"  FOVs processed   : {len(all_populations)} / {len(fovs)}")
    log.info(f"  Detections OK    : {n_ok}")
    log.info(f"  Detections failed: {n_fail}")
    log.info(f"  Total tracks     : {len(all_tracks)}")
    log.info(f"  Output           : {config.output_dir}")
    log.info("")
    log.info("  Timing breakdown:")
    for step, secs in timing.items():
        if step == "total":
            continue
        pct = secs / total_time * 100 if total_time > 0 else 0
        bar = "#" * int(pct / 5)
        log.info(f"    {step:<20} {secs:>8.2f}s  {pct:>5.1f}%  {bar}")
    log.info(f"    {'TOTAL':<20} {total_time:>8.2f}s")
    log.info("=" * 65)


def _load_plugin(config, log):
    try:
        if config.experiment_type == ExperimentType.HEMACHIP:
            from plugins.hemachip.plugin import HemaChipPlugin
            log.info("  Plugin: HemaChip loaded")
            return HemaChipPlugin(config)
        else:
            from plugins.standard.plugin import StandardPlugin
            log.info("  Plugin: Standard loaded")
            return StandardPlugin(config)
    except ImportError as e:
        log.warning(f"  Plugin not available: {e}")
        return None


if __name__ == "__main__":
    main()
