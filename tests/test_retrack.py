"""tests/test_retrack.py - tests for re-tracking from a prior run's spots.csv,
without re-running segmentation. Verifies both correctness and the actual
speed claim (should be seconds, not minutes)."""
import sys
import time
import tempfile
import logging
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def make_prior_run(tmp_path, n_cells=50, n_frames=40, seed=1):
    """Build a realistic fake prior run's spots.csv + run_config.json on disk."""
    from config import SpotRecord
    rng = np.random.default_rng(seed)
    positions = rng.uniform(20, 780, size=(n_cells, 2))

    spots = []
    for f in range(n_frames):
        for cid in range(n_cells):
            spots.append(SpotRecord(
                fov_id="ROI-1", frame_index=f, elapsed_min=f * 5.0, spot_id=cid + 1,
                centroid_x_px=float(positions[cid, 0]), centroid_y_px=float(positions[cid, 1]),
                area_px=120.0, area_um2=17.0, perimeter_px=42.0,
                circularity=0.95, eccentricity=0.2, mean_intensity=150.0,
                segmentation_ok=True,
            ))
        positions += rng.normal(0, 1.0, size=(n_cells, 2))
        positions = np.clip(positions, 10, 790)

    prior_dir = tmp_path / "prior_run"
    prior_dir.mkdir()
    df = pd.DataFrame([s.to_dict() for s in spots])
    df.to_csv(prior_dir / "spots.csv", index=False)

    import json
    config_json = {
        "input_dir": "/fake", "output_dir": str(prior_dir),
        "frame_mode": "all", "experiment_type": "standard",
        "n_workers": 1, "log_level": "INFO", "run_id": "test", "pipeline_version": "1.0.0",
        "segmentation": {"model_name": "cyto3", "diameter": 31.9, "use_gpu": False, "flow_threshold": 0.4,
                         "cellprob_threshold": 0.3, "min_diameter_um": 4.0, "max_diameter_um": 30.0, "batch_size": 8},
        "tracking": {"max_distance_px": 50.0, "gap_closing_max_dist_px": 80.0, "gap_closing_max_frames": 2,
                    "min_track_length_frames": 3, "detect_division": False, "division_max_dist_px": 40.0,
                    "tracker_method": "lap", "min_elapsed_min": None},
        "imaging": {"pixel_size_um": 0.3769, "time_interval_min": 5.0, "expected_width_px": 800,
                   "expected_height_px": 800, "is_timelapse": True},
        "output": {"save_masks": True, "save_overlays": True, "save_tracks_viz": True,
                  "overlay_max_fovs": 25, "compress_masks": False},
    }
    with open(prior_dir / "run_config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    return prior_dir, spots


def test_load_spots_csv():
    log.info("TEST 1: load_spots_csv round-trip")
    from core.spot_loader import load_spots_csv

    with tempfile.TemporaryDirectory() as tmp:
        prior_dir, original_spots = make_prior_run(Path(tmp), n_cells=20, n_frames=10)
        loaded = load_spots_csv(prior_dir / "spots.csv")

        assert len(loaded) == len(original_spots), f"Expected {len(original_spots)}, got {len(loaded)}"
        assert all(s.track_id == -1 for s in loaded), "All reloaded spots must have track_id reset to -1"
        assert all(s.segmentation_ok for s in loaded)

        # Spot-check one record's fields survived the round trip correctly
        orig = original_spots[5]
        match = [s for s in loaded if s.fov_id == orig.fov_id
                and s.frame_index == orig.frame_index and s.spot_id == orig.spot_id]
        assert len(match) == 1
        assert abs(match[0].centroid_x_px - orig.centroid_x_px) < 0.01
        assert abs(match[0].area_um2 - orig.area_um2) < 0.01

        log.info(f"  [OK] {len(loaded)} spots round-tripped correctly, track_id reset")
    log.info("  load_spots_csv PASSED")
    return True


def test_load_spots_csv_preserves_viability_fields():
    log.info("TEST 1b: load_spots_csv preserves equivalent_diameter_um and is_viable (regression test)")
    # Regression test for a real bug found during codebase audit:
    # load_spots_csv() silently dropped equivalent_diameter_um and
    # is_viable on every reload, reconstructing SpotRecords with the
    # class defaults (0.0, None) instead of the real values actually
    # present in spots.csv. equivalent_diameter_um is always computed
    # during segmentation (not conditional on viability classification
    # being active), so this discarded real per-cell diameter data on
    # EVERY retrack.py invocation, not just ones involving viability.
    # The original test above didn't catch this because make_prior_run()
    # never set these two fields to anything other than their defaults,
    # so a silently-dropped default looked identical to a correctly
    # round-tripped default.
    from config import SpotRecord
    from core.spot_loader import load_spots_csv

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        spots = [
            SpotRecord(fov_id="ROI-1", frame_index=0, elapsed_min=0.0, spot_id=1,
                      area_um2=113.1, equivalent_diameter_um=12.0, is_viable=True),
            SpotRecord(fov_id="ROI-1", frame_index=0, elapsed_min=0.0, spot_id=2,
                      area_um2=50.0, equivalent_diameter_um=8.0, is_viable=False),
            SpotRecord(fov_id="ROI-1", frame_index=0, elapsed_min=0.0, spot_id=3,
                      area_um2=100.0, equivalent_diameter_um=11.0, is_viable=None),
        ]
        df = pd.DataFrame([s.to_dict() for s in spots])
        csv_path = tp / "spots.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_spots_csv(csv_path)
        by_id = {s.spot_id: s for s in loaded}

        assert by_id[1].equivalent_diameter_um == 12.0, (
            f"BUG REGRESSION: expected 12.0, got {by_id[1].equivalent_diameter_um} - "
            f"equivalent_diameter_um is being dropped on reload"
        )
        assert by_id[1].is_viable is True, f"Expected True, got {by_id[1].is_viable!r}"
        assert by_id[2].equivalent_diameter_um == 8.0
        assert by_id[2].is_viable is False, f"Expected False, got {by_id[2].is_viable!r}"
        assert by_id[3].equivalent_diameter_um == 11.0
        assert by_id[3].is_viable is None, (
            f"Expected None (viability inactive), got {by_id[3].is_viable!r} - "
            f"None must not be silently coerced to False"
        )

        log.info("  [OK] equivalent_diameter_um and is_viable (True/False/None) all survive round-trip correctly")
    log.info("  Viability field preservation PASSED")
    return True


def test_load_spots_csv_missing_file():
    log.info("TEST 2: load_spots_csv error handling")
    from core.spot_loader import load_spots_csv
    try:
        load_spots_csv("/nonexistent/spots.csv")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    log.info("  [OK] raises FileNotFoundError for missing file")
    log.info("  Error handling PASSED")
    return True


def test_load_spots_csv_missing_columns():
    log.info("TEST 3: load_spots_csv rejects malformed CSV")
    from core.spot_loader import load_spots_csv
    with tempfile.TemporaryDirectory() as tmp:
        bad_csv = Path(tmp) / "bad_spots.csv"
        pd.DataFrame({"some_column": [1, 2, 3]}).to_csv(bad_csv, index=False)
        try:
            load_spots_csv(bad_csv)
            assert False, "Should have raised ValueError for missing required columns"
        except ValueError:
            pass
    log.info("  [OK] raises ValueError for missing required columns")
    log.info("  Malformed CSV handling PASSED")
    return True


def test_group_by_fov_and_frame():
    log.info("TEST 4: group_by_fov_and_frame")
    from core.spot_loader import load_spots_csv, group_by_fov_and_frame

    with tempfile.TemporaryDirectory() as tmp:
        prior_dir, original_spots = make_prior_run(Path(tmp), n_cells=15, n_frames=8)
        spots = load_spots_csv(prior_dir / "spots.csv")
        grouped = group_by_fov_and_frame(spots)

        assert "ROI-1" in grouped
        frame_spots, frame_indices = grouped["ROI-1"]
        assert len(frame_spots) == 8, f"Expected 8 frames, got {len(frame_spots)}"
        assert frame_indices == sorted(frame_indices), "Frame indices must be sorted"
        assert all(len(f) == 15 for f in frame_spots), "Each frame should have 15 cells"

        log.info(f"  [OK] grouped into {len(frame_spots)} frames, {len(frame_spots[0])} cells/frame")
    log.info("  group_by_fov_and_frame PASSED")
    return True


def test_load_run_for_retracking():
    log.info("TEST 5: load_run_for_retracking end-to-end")
    from core.spot_loader import load_run_for_retracking

    with tempfile.TemporaryDirectory() as tmp:
        prior_dir, original_spots = make_prior_run(Path(tmp), n_cells=10, n_frames=6)
        fov_data = load_run_for_retracking(prior_dir)

        assert "ROI-1" in fov_data
        fov, frame_spots = fov_data["ROI-1"]
        assert fov.pixel_size_um == 0.3769, "Should read pixel_size_um from run_config.json"
        assert fov.time_interval_min == 5.0
        assert fov.n_frames == 6
        assert len(frame_spots) == 6

        log.info(f"  [OK] pixel_size_um={fov.pixel_size_um}, n_frames={fov.n_frames}")
    log.info("  load_run_for_retracking PASSED")
    return True


def test_retrack_matches_live_tracking():
    log.info("TEST 6: retrack produces IDENTICAL results to live tracking")
    # This is the critical correctness test: re-tracking from a reloaded
    # spots.csv must produce the exact same tracks as tracking the same
    # spots live, never having gone through a CSV round-trip. If these
    # diverge, the reload path is silently corrupting something.
    from core.spot_loader import load_run_for_retracking
    from core.tracker import track_fov
    from config import TrackingConfig, SpotRecord
    from collections import defaultdict

    with tempfile.TemporaryDirectory() as tmp:
        prior_dir, original_spots = make_prior_run(Path(tmp), n_cells=30, n_frames=20)

        # Live tracking path: track the original in-memory spots directly
        by_frame = defaultdict(list)
        for s in original_spots:
            by_frame[s.frame_index].append(s)
        live_frame_spots = [by_frame[i] for i in range(20)]
        cfg = TrackingConfig(max_distance_px=30.0, min_track_length_frames=3)
        live_tracked, live_tracks = track_fov(live_frame_spots, cfg, 0.3769, 20)

        # Reload path: load from spots.csv, then track
        fov_data = load_run_for_retracking(prior_dir)
        fov, reloaded_frame_spots = fov_data["ROI-1"]
        reloaded_tracked, reloaded_tracks = track_fov(reloaded_frame_spots, cfg, fov.pixel_size_um, fov.n_frames)

        assert len(live_tracks) == len(reloaded_tracks), (
            f"Track count mismatch: live={len(live_tracks)}, reloaded={len(reloaded_tracks)}"
        )

        live_displacements = sorted(round(t.total_displacement_px, 2) for t in live_tracks)
        reloaded_displacements = sorted(round(t.total_displacement_px, 2) for t in reloaded_tracks)
        assert live_displacements == reloaded_displacements, (
            "Track displacements differ between live and reloaded tracking - "
            "the CSV round-trip is corrupting something"
        )

        log.info(f"  [OK] {len(live_tracks)} tracks, identical displacements in live vs reloaded path")
    log.info("  retrack correctness PASSED")
    return True


def test_retrack_is_fast():
    log.info("TEST 7: retrack completes in seconds, not minutes (realistic scale)")
    from core.spot_loader import load_run_for_retracking
    from core.tracker import track_fov
    from core.track_features import compute_population

    with tempfile.TemporaryDirectory() as tmp:
        # Realistic scale: ~140 cells x 97 frames, matching real CB007 data
        prior_dir, _ = make_prior_run(Path(tmp), n_cells=140, n_frames=97, seed=42)

        t0 = time.perf_counter()
        fov_data = load_run_for_retracking(prior_dir)
        fov, frame_spots = fov_data["ROI-1"]

        from config import TrackingConfig
        cfg = TrackingConfig(max_distance_px=20.0, min_track_length_frames=3, min_elapsed_min=60.0)
        tracked, tracks = track_fov(frame_spots, cfg, fov.pixel_size_um, fov.n_frames)
        pop = compute_population(tracks, tracked, fov)
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0, (
            f"Retrack took {elapsed:.1f}s for 140 cells x 97 frames - expected well under "
            f"30s (a real segmentation run at this scale takes 30-60+ minutes)"
        )
        assert len(tracks) > 0

        log.info(f"  [OK] {len(tracks)} tracks from 140 cells x 97 frames in {elapsed:.2f}s "
                 f"(vs 30-60+ minutes for a full segmentation run)")
    log.info("  Speed PASSED")
    return True


def run_all():
    tests = [
        test_load_spots_csv,
        test_load_spots_csv_preserves_viability_fields,
        test_load_spots_csv_missing_file,
        test_load_spots_csv_missing_columns,
        test_group_by_fov_and_frame,
        test_load_run_for_retracking,
        test_retrack_matches_live_tracking,
        test_retrack_is_fast,
    ]
    passed = failed = 0
    for t in tests:
        log.info("")
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            log.error(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1
    log.info("")
    log.info("=" * 50)
    log.info(f"  Results: {passed} passed, {failed} failed")
    log.info("=" * 50)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_all()
