"""tests/test_core.py - CellFlow core engine test suite. Uses synthetic data only."""
import sys
import tempfile
import logging
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def make_config(input_dir, output_dir):
    from config import PipelineConfig, ImagingConfig
    return PipelineConfig(input_dir=input_dir, output_dir=output_dir,
                          imaging=ImagingConfig(pixel_size_um=0.377, time_interval_min=5.0))


def make_fov(fov_dir, n_frames=10):
    import tifffile
    fov_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        tifffile.imwrite(str(fov_dir / f"test_{i:06d}.tif"), np.random.randint(100, 200, (64, 64), dtype=np.uint16))
    return fov_dir


def make_mask(n_cells=5, size=64):
    mask = np.zeros((size, size), dtype=np.uint16)
    spacing = size // (n_cells + 1)
    for i in range(n_cells):
        cx = (i + 1) * spacing
        cy = size // 2
        r = 6
        rr, cc = np.ogrid[-r:r + 1, -r:r + 1]
        circle = rr ** 2 + cc ** 2 <= r ** 2
        rs = slice(max(0, cy - r), min(size, cy + r + 1))
        cs = slice(max(0, cx - r), min(size, cx + r + 1))
        ch = circle[:rs.stop - rs.start, :cs.stop - cs.start]
        mask[rs, cs][ch] = i + 1
    return mask


def make_spots(fov_id, n_frames=5, n_cells=5):
    from config import SpotRecord
    spots = []
    for f in range(n_frames):
        for c in range(n_cells):
            spots.append(SpotRecord(
                fov_id=fov_id, frame_index=f, elapsed_min=f * 5.0, spot_id=c + 1,
                centroid_x_px=50.0 + c * 20 + f * 0.5, centroid_y_px=30.0 + c * 10,
                area_px=120.0, area_um2=17.0, perimeter_px=42.0,
                circularity=0.85, eccentricity=0.2, mean_intensity=150.0,
            ))
    return spots


def test_input_handler():
    log.info("TEST 1: Input handler")
    from core.input_handler import load_fov, load_fovs_from_batch
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        fov_dir = tp / "ROI-001"
        make_fov(fov_dir, n_frames=20)
        config = make_config(fov_dir, tp / "out")
        fov = load_fov(fov_dir, "ROI-001", config)
        assert fov is not None
        assert fov.fov_id == "ROI-001"
        assert fov.n_frames == 20
        assert not fov.is_empty
        batch = tp / "batch"
        for i in range(3):
            make_fov(batch / f"ROI-{i+1:03d}", n_frames=5)
        config2 = make_config(batch, tp / "out2")
        fovs = load_fovs_from_batch(batch, config2)
        assert len(fovs) == 3
        log.info(f"  [OK] single FOV: {fov.n_frames} frames | batch: {len(fovs)} FOVs")
    log.info("  Input handler PASSED")
    return True


def test_frame_selection():
    log.info("TEST 2: Frame selection modes")
    from core.input_handler import load_fov
    from config import FrameMode, ImagingConfig, PipelineConfig
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        fd = tp / "ROI-001"
        make_fov(fd, n_frames=97)
        for mode, lo, hi in [(FrameMode.FIRST, 1, 1), (FrameMode.MEDIAN, 1, 1),
                              (FrameMode.CLINICAL, 15, 20), (FrameMode.ALL, 97, 97)]:
            c = PipelineConfig(input_dir=fd, output_dir=tp / "out", frame_mode=mode,
                               imaging=ImagingConfig(pixel_size_um=0.377, time_interval_min=5.0))
            fov = load_fov(fd, "ROI-001", c)
            n = fov.n_frames
            assert lo <= n <= hi, f"Mode {mode.value}: expected {lo}-{hi}, got {n}"
            log.info(f"  [OK] {mode.value:<10} -> {n} frames")
    log.info("  Frame selection PASSED")
    return True


def test_spot_features_performance():
    log.info("TEST 3b: Spot feature extraction performance at realistic scale")
    # Regression test for a real bug: extract_spot_features() originally ran
    # perimeter/eccentricity/solidity on a FULL-FRAME boolean array per cell
    # (e.g. 1600x1600) regardless of actual cell size (~15-20px). At realistic
    # HSEC density (140 cells/frame), this cost ~64s/ROI for perimeter alone -
    # over 4 hours wasted across a 225-ROI chip. Fixed by cropping each cell
    # to its bounding box (via scipy.ndimage.find_objects) before running any
    # per-cell geometry.
    #
    # NOTE: this test previously asserted an ABSOLUTE wall-clock ceiling
    # (elapsed < 1.0s), calibrated against one specific machine. That is
    # inherently fragile - a slower CPU or a numpy/scipy build without an
    # optimised BLAS backend can legitimately take several seconds for the
    # same, correctly-cropped code, which produced a false failure report
    # (not a real regression) on at least one reviewer's machine. Fixed to
    # compare the CURRENT (cropped) implementation's time against a
    # deliberately-reproduced OLD (full-frame-array) implementation's time,
    # measured on whatever machine the test actually runs on - this tests
    # the thing that actually matters (cropping is meaningfully faster than
    # not cropping) rather than an arbitrary absolute number tied to one
    # environment's hardware and library versions.
    import time
    from core.spot_features import extract_spot_features, _perim, _ecc, _sol, _centroid
    from config import FieldOfView, FrameInfo, SegmentationConfig

    size = 1600
    n_cells = 140
    mask = np.zeros((size, size), dtype=np.uint16)
    rng = np.random.default_rng(7)
    for i in range(n_cells):
        cy, cx = rng.integers(20, size - 20, 2)
        r = 8
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        circle = xx**2 + yy**2 <= r**2
        mask[cy-r:cy+r+1, cx-r:cx+r+1][circle] = i + 1
    img = rng.integers(50, 200, (size, size)).astype(np.uint16)

    fov = FieldOfView(path=Path("/fake"), fov_id="TEST", frames=[], pixel_size_um=0.3769)
    frame = FrameInfo(path=Path("/fake/f.tif"), frame_index=0, elapsed_min=0.0, fov_id="TEST")
    seg = SegmentationConfig(min_diameter_um=2.0, max_diameter_um=30.0)

    # Current (cropped) implementation - the real code path.
    t0 = time.perf_counter()
    spots = extract_spot_features(mask, img, frame, fov, seg)
    cropped_elapsed = time.perf_counter() - t0

    assert len(spots) == n_cells, f"Expected {n_cells} cells, got {len(spots)}"

    # Deliberately-reproduced OLD (full-frame-array) approach, run on this
    # SAME machine right now, so the comparison is apples-to-apples
    # regardless of hardware speed. This mirrors exactly what the
    # pre-fix code did: run perimeter/eccentricity/solidity on a full-
    # size boolean array per cell instead of a cropped bounding box.
    labels = np.unique(mask)
    labels = labels[labels > 0]
    t0 = time.perf_counter()
    for label in labels:
        m = (mask == label)
        area = float(np.sum(m))
        _centroid(m)
        perim = _perim(m)
        _ecc(m)
        _sol(m, area)
    uncropped_elapsed = time.perf_counter() - t0

    log.info(f"  Cropped (current): {cropped_elapsed*1000:.0f}ms | "
             f"Uncropped (old approach, same machine): {uncropped_elapsed*1000:.0f}ms")

    # NOTE: a bare "cropped_elapsed < uncropped_elapsed" check is too weak -
    # verified directly by temporarily reintroducing the actual old bug and
    # confirming this test still reported a "pass" (cropped and uncropped
    # times came out within noise of each other, ~1.0x, since they were
    # running the same code). Requiring a meaningful margin (at least 2x)
    # means measurement noise alone cannot produce a false pass; the real
    # fix consistently measures over 10x faster on every machine tested.
    min_expected_speedup = 2.0
    actual_speedup = uncropped_elapsed / cropped_elapsed if cropped_elapsed > 0 else float("inf")
    assert actual_speedup >= min_expected_speedup, (
        f"BUG REGRESSION: cropped implementation is only {actual_speedup:.1f}x faster than "
        f"the uncropped full-frame-array approach (expected at least {min_expected_speedup}x) "
        f"on this same machine - extract_spot_features() may have regressed to operating "
        f"on full-frame arrays instead of cropped bounding boxes. "
        f"(cropped={cropped_elapsed*1000:.0f}ms, uncropped={uncropped_elapsed*1000:.0f}ms)"
    )

    log.info(f"  [OK] {n_cells} cells at {size}x{size}: cropped={cropped_elapsed*1000:.0f}ms, "
             f"uncropped={uncropped_elapsed*1000:.0f}ms ({actual_speedup:.1f}x faster, measured on this machine)")
    log.info("  Spot features performance PASSED")
    return True


def test_spot_features():
    log.info("TEST 3: Spot feature extraction")
    from core.spot_features import extract_spot_features
    from config import FieldOfView, FrameInfo, SegmentationConfig
    mask = make_mask(n_cells=5, size=64)
    img = np.random.randint(100, 200, (64, 64), dtype=np.uint16)
    fov = FieldOfView(path=Path("/fake"), fov_id="TEST", frames=[], pixel_size_um=0.377)
    frame = FrameInfo(path=Path("/fake/f.tif"), frame_index=0, elapsed_min=0.0, fov_id="TEST")
    seg = SegmentationConfig(min_diameter_um=2.0, max_diameter_um=30.0)
    spots = extract_spot_features(mask, img, frame, fov, seg)
    assert len(spots) == 5, f"Expected 5, got {len(spots)}"
    for s in spots:
        assert s.area_um2 > 0
        assert 0 <= s.circularity <= 1
        assert s.segmentation_ok
    log.info(f"  [OK] 5 cells | circ={np.mean([s.circularity for s in spots]):.3f} | area={np.mean([s.area_um2 for s in spots]):.1f}um2")
    log.info("  Spot features PASSED")
    return True


def test_tracker():
    log.info("TEST 4: Tracker")
    from core.tracker import track_fov
    from config import TrackingConfig
    from collections import defaultdict
    spots = make_spots("ROI-001", n_frames=5, n_cells=5)
    by_frame = defaultdict(list)
    for s in spots:
        by_frame[s.frame_index].append(s)
    frame_spots = [by_frame[i] for i in range(5)]
    cfg = TrackingConfig(max_distance_px=30.0, gap_closing_max_frames=1, min_track_length_frames=3)
    tracked, tracks = track_fov(frame_spots, cfg, 0.377, 5)
    assert len(tracked) > 0
    assigned = [s for s in tracked if s.track_id >= 0]
    assert len(assigned) > 0, "Some spots should have track IDs"
    assert len(tracks) > 0, "Should have track records"
    for t in tracks:
        assert t.lifespan_frames >= cfg.min_track_length_frames
        assert t.confinement_ratio >= 0
    log.info(f"  [OK] {len(tracks)} tracks | lifespan={np.mean([t.lifespan_min for t in tracks]):.1f}min")
    log.info("  Tracker PASSED")
    return True


def test_tracker_min_elapsed_min():
    log.info("TEST 4b: Tracker min_elapsed_min cutoff (regression test)")
    # Regression test for a real bug: laptrack's predict() returns graph nodes
    # keyed by POSITION in the coords list passed to it, not by the original
    # frame values. When min_elapsed_min excludes leading frames, the coords
    # list no longer starts at frame 0, so failing to remap position->real
    # frame silently produced zero tracks and zero tracked spots, even though
    # spots existed and were well within range of each other.
    from core.tracker import track_fov
    from config import TrackingConfig
    from collections import defaultdict

    spots = make_spots("ROI-001", n_frames=20, n_cells=5)
    by_frame = defaultdict(list)
    for s in spots:
        by_frame[s.frame_index].append(s)
    frame_spots = [by_frame[i] for i in range(20)]

    cfg_all = TrackingConfig(max_distance_px=30.0, min_track_length_frames=3)
    tracked_all, tracks_all = track_fov(frame_spots, cfg_all, 0.377, 20)
    assert len(tracked_all) == len(spots), "No filter: spot count must be preserved"
    assert len(tracks_all) > 0, "No filter: should produce tracks"

    cfg_cut = TrackingConfig(max_distance_px=30.0, min_track_length_frames=3, min_elapsed_min=50.0)
    tracked_cut, tracks_cut = track_fov(frame_spots, cfg_cut, 0.377, 20)

    assert len(tracked_cut) == len(spots), (
        f"Spot count must be preserved even when tracking is restricted "
        f"(pre-cutoff spots pass through untracked): got {len(tracked_cut)}, expected {len(spots)}"
    )
    n_tracked = len([s for s in tracked_cut if s.track_id >= 0])
    n_untracked = len([s for s in tracked_cut if s.track_id < 0])
    assert n_tracked > 0, "BUG REGRESSION: cutoff produced zero tracked spots"
    assert n_untracked == 10 * 5, f"Expected 50 untracked (frames 0-9 x 5 cells), got {n_untracked}"
    assert len(tracks_cut) > 0, "BUG REGRESSION: cutoff produced zero tracks"

    for s in tracked_cut:
        if s.elapsed_min < 50.0:
            assert s.track_id == -1, f"Spot at t={s.elapsed_min} should be untracked (before cutoff)"
    for t in tracks_cut:
        assert t.first_elapsed_min >= 50.0, f"Track must not start before cutoff: {t.first_elapsed_min}"

    log.info(f"  [OK] no filter: {len(tracks_all)} tracks, {len(tracked_all)} spots preserved")
    log.info(f"  [OK] min_elapsed_min=50: {len(tracks_cut)} tracks, {n_tracked} tracked / {n_untracked} untracked, "
             f"{len(tracked_cut)} total spots preserved")
    log.info("  Tracker min_elapsed_min PASSED")
    return True


def test_population():
    log.info("TEST 5: Population features")
    from core.track_features import compute_population
    from core.tracker import track_fov
    from config import TrackingConfig, FieldOfView
    from collections import defaultdict
    spots = make_spots("ROI-001", n_frames=10, n_cells=8)
    by_frame = defaultdict(list)
    for s in spots:
        by_frame[s.frame_index].append(s)
    frame_spots = [by_frame[i] for i in range(10)]
    cfg = TrackingConfig(min_track_length_frames=3)
    tracked, tracks = track_fov(frame_spots, cfg, 0.377, 10)
    fov = FieldOfView(path=Path("/fake"), fov_id="ROI-001", frames=[None] * 10, pixel_size_um=0.377, time_interval_min=5.0)
    pop = compute_population(tracks, tracked, fov)
    assert pop.fov_id == "ROI-001"
    assert pop.n_tracks_total >= 0
    assert pop.mean_lifespan_min >= 0
    assert 0 <= pop.pct_tracks_survived <= 100
    log.info(f"  [OK] {pop.n_tracks_total} tracks | survival={pop.pct_tracks_survived:.1f}%")
    log.info("  Population PASSED")
    return True


def test_exporter():
    log.info("TEST 6: Exporter")
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        out = tp / "results"
        config = make_config(tp, out)
        spots = make_spots("ROI-001", n_frames=5, n_cells=5)
        from core.exporter import export_all
        written = export_all(spots, [], [], config, {"test": 1.0})
        assert (out / "spots.csv").exists()
        assert (out / "run_summary.json").exists()
        assert (out / "run_config.json").exists()
        df = pd.read_csv(out / "spots.csv")
        assert len(df) == 25, f"Expected 25 rows, got {len(df)}"
        assert "track_id" in df.columns
        assert "circularity" in df.columns
        log.info(f"  [OK] spots.csv: {len(df)} rows | all files present")
    log.info("  Exporter PASSED")
    return True


def test_config_serialisation():
    log.info("TEST 7: Config serialisation")
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        config = make_config(tp, tp / "out")
        json_path = tp / "config.json"
        config.to_json(json_path)
        assert json_path.exists()
        from config import PipelineConfig
        loaded = PipelineConfig.from_json(json_path)
        assert loaded.frame_mode == config.frame_mode
        assert loaded.segmentation.model_name == config.segmentation.model_name
        assert loaded.tracking.max_distance_px == config.tracking.max_distance_px
        log.info("  [OK] serialised and deserialised correctly")
    log.info("  Config PASSED")
    return True


def test_resume_checkpoint_determinism():
    log.info("TEST 8: Resume - deterministic run_id finds prior checkpoint")
    # Regression test for a real bug: run_id was previously a random UUID
    # generated fresh on every pipeline.py invocation, which meant a second
    # run against the SAME output_dir could never find the first run's
    # checkpoint file (named run_{run_id}.json) - --resume silently did
    # nothing, and a Colab disconnect would mean starting over from scratch
    # at whatever ROI it died on. Fixed by deriving run_id deterministically
    # from output_dir's absolute path (see pipeline.py parse_args()).
    import hashlib
    from job_manager import JobManager
    from config import PipelineConfig, ImagingConfig

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        input_dir = tp / "input"
        input_dir.mkdir()
        output_dir = tp / "output"
        run_id = hashlib.md5(str(output_dir.resolve()).encode()).hexdigest()[:8]

        config1 = PipelineConfig(input_dir=input_dir, output_dir=output_dir, run_id=run_id,
                                 imaging=ImagingConfig(pixel_size_um=0.377, time_interval_min=5.0))
        job1 = JobManager(config1)
        assert not job1.is_completed("ROI-1")
        job1.mark_complete("ROI-1", spots=[1, 2, 3], tracks=[], pop=None)

        # Fresh JobManager instance, same output_dir -> must find the same run_id
        # and thus the same checkpoint, exactly as a second pipeline.py
        # invocation against the same --output_dir would.
        config2 = PipelineConfig(input_dir=input_dir, output_dir=output_dir, run_id=run_id,
                                 imaging=ImagingConfig(pixel_size_um=0.377, time_interval_min=5.0))
        job2 = JobManager(config2)

        assert job2.is_completed("ROI-1"), (
            "BUG REGRESSION: second JobManager instance did not find first "
            "instance's checkpoint - resume is broken"
        )
        assert not job2.is_completed("ROI-2")

        spots, tracks, pop = job2.load_fov_results("ROI-1")
        assert spots == [1, 2, 3], f"Checkpoint data corrupted on reload: {spots}"

        log.info("  [OK] second JobManager instance correctly resumed from first instance's checkpoint")
    log.info("  Resume determinism PASSED")
    return True


def test_checkpoint_survives_dotted_filenames():
    log.info("TEST 9: Checkpoint save/load survives fov_id containing a literal dot")
    # Regression test for a real bug found during codebase audit:
    # job_manager.py used Path.with_suffix(".tmp.pkl") / (".tmp.json") to
    # build temp filenames before an atomic rename. with_suffix() REPLACES
    # the existing suffix rather than appending one - this happened to
    # produce the intended result for filenames with exactly one "." (e.g.
    # "ROI-1_spots.pkl" -> "ROI-1_spots.tmp.pkl"), but fov_id is taken
    # directly from the ROI folder name on disk with NO sanitisation (see
    # core/input_handler.py: fov_id = d.name). A folder legitimately named
    # something like "ROI-1.backup" would have silently produced the wrong
    # temp file path. Fixed by using string concatenation instead of
    # with_suffix() for temp filenames (job_manager.py, core/exporter.py,
    # plugins/hemachip/plugin.py).
    from job_manager import _save_pickle, _load_pickle

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        tricky_path = tp / "ROI-1.backup_spots.pkl"
        _save_pickle({"test": "data"}, tricky_path)
        result = _load_pickle(tricky_path)

        assert result == {"test": "data"}, f"Round-trip failed for dotted filename: {result}"
        assert tricky_path.exists(), "Final file does not exist at the expected path"
        assert not (tp / "ROI-1.tmp").exists(), (
            "BUG REGRESSION: with_suffix() produced a stray, incorrectly-named "
            "temp file instead of appending .tmp to the full filename"
        )

        log.info("  [OK] dotted filename round-trips correctly, no stray temp files left behind")
    log.info("  Dotted filename checkpoint PASSED")
    return True


def run_all():
    tests = [test_input_handler, test_frame_selection, test_spot_features_performance, test_spot_features,
             test_tracker, test_tracker_min_elapsed_min, test_population, test_exporter,
             test_config_serialisation, test_resume_checkpoint_determinism,
             test_checkpoint_survives_dotted_filenames]
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
