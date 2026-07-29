"""tests/test_viability.py - tests for core/viability.py. Verifies the
classification is genuinely inactive by default (not silently populated
with a borrowed threshold), and correct once a threshold is supplied."""
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def make_spots(areas_um2, fov_id="ROI-1"):
    from config import SpotRecord
    return [
        SpotRecord(fov_id=fov_id, frame_index=0, elapsed_min=0.0, spot_id=i + 1,
                  area_um2=a, segmentation_ok=True)
        for i, a in enumerate(areas_um2)
    ]


def test_equivalent_diameter_calculation():
    log.info("TEST 1: equivalent_diameter_um calculation correctness")
    from core.viability import equivalent_diameter_um
    import numpy as np

    # A circle of diameter 10um has area = pi*(5)^2 = 78.54 um2
    area = 78.53981633974483
    d = equivalent_diameter_um(area)
    assert abs(d - 10.0) < 1e-6, f"Expected 10.0um diameter, got {d}"

    assert equivalent_diameter_um(0.0) == 0.0
    assert equivalent_diameter_um(-5.0) == 0.0, "Negative area should return 0, not crash or go complex"

    log.info(f"  [OK] 78.54um2 area -> {d:.4f}um diameter (expected 10.0)")
    log.info("  Equivalent diameter PASSED")
    return True


def test_inactive_by_default():
    log.info("TEST 2: viability classification is INACTIVE by default")
    from core.viability import classify_viability

    spots = make_spots([10.0, 50.0, 200.0])
    result = classify_viability(spots, diameter_threshold_um=None)

    assert all(s.is_viable is None for s in result), (
        "BUG: is_viable should stay None for every spot when no threshold "
        "is given - classification must not silently activate with a "
        "borrowed/default value"
    )
    assert all(s.equivalent_diameter_um > 0 for s in result), (
        "equivalent_diameter_um should still be computed even when "
        "viability classification itself is inactive"
    )

    log.info("  [OK] is_viable is None on all spots with no threshold configured")
    log.info("  [OK] equivalent_diameter_um still populated regardless")
    log.info("  Inactive-by-default PASSED")
    return True


def test_inactive_warning_fires_every_call_not_just_first():
    log.info("TEST 2b: inactive-viability warning fires on every FOV, not just the first (regression test)")
    # Regression test for a real bug found during codebase audit: the
    # "viability classification inactive" log message was previously
    # gated by a bare module-level global (_WARNED_INACTIVE) that only
    # allowed it to fire ONCE PER PYTHON PROCESS, not once per FOV or
    # once per pipeline run as intended. In a real multi-ROI pipeline run
    # (pipeline.py calls classify_viability once per FOV in its main
    # loop), this meant the warning appeared for the first ROI only and
    # silently vanished for every other ROI in the run - e.g. ROIs 2-225
    # of a full chip would give no indication that viability was inactive
    # for them too. Fixed to log on every call where classification is
    # inactive, which is one short line per FOV, not spam.
    import logging
    import io
    from core.viability import classify_viability
    from config import SpotRecord

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    viability_logger = logging.getLogger("core.viability")
    original_level = viability_logger.level
    viability_logger.addHandler(handler)
    viability_logger.setLevel(logging.INFO)

    try:
        spots1 = [SpotRecord(fov_id="ROI-1", frame_index=0, elapsed_min=0.0, spot_id=1, area_um2=100.0)]
        spots2 = [SpotRecord(fov_id="ROI-2", frame_index=0, elapsed_min=0.0, spot_id=1, area_um2=100.0)]

        classify_viability(spots1, diameter_threshold_um=None)
        classify_viability(spots2, diameter_threshold_um=None)

        log_output = stream.getvalue()
        occurrences = log_output.count("INACTIVE")
        assert occurrences == 2, (
            f"BUG REGRESSION: expected the inactive-viability warning to fire "
            f"on both calls (2 occurrences), got {occurrences} - a module-level "
            f"global may be suppressing repeat occurrences again"
        )
    finally:
        viability_logger.removeHandler(handler)
        viability_logger.setLevel(original_level)

    log.info("  [OK] warning fired on both the first AND second call in the same process")
    log.info("  Per-call warning PASSED")
    return True


def test_classification_correctness():
    log.info("TEST 3: classification correctness once threshold is set")
    from core.viability import classify_viability, equivalent_diameter_um

    # Areas chosen so their equivalent diameters straddle a threshold of 10um:
    # area for d=8um -> pi*16 = 50.27 (below threshold, non-viable)
    # area for d=12um -> pi*36 = 113.1 (above threshold, viable)
    area_below = 3.14159265 * 4**2   # d=8um
    area_above = 3.14159265 * 6**2   # d=12um
    spots = make_spots([area_below, area_above])

    result = classify_viability(spots, diameter_threshold_um=10.0)

    assert result[0].is_viable is False, f"d={result[0].equivalent_diameter_um:.2f}um should be non-viable (< 10um)"
    assert result[1].is_viable is True, f"d={result[1].equivalent_diameter_um:.2f}um should be viable (>= 10um)"

    log.info(f"  [OK] d={result[0].equivalent_diameter_um:.2f}um classified non-viable")
    log.info(f"  [OK] d={result[1].equivalent_diameter_um:.2f}um classified viable")
    log.info("  Classification correctness PASSED")
    return True


def test_summarise_no_threshold():
    log.info("TEST 4: summarise_viability with no threshold set returns None, not 0/100")
    from core.viability import classify_viability, summarise_viability

    spots = make_spots([10.0, 50.0, 200.0])
    classify_viability(spots, diameter_threshold_um=None)
    summary = summarise_viability(spots)

    assert summary["pct_viable"] is None, (
        "BUG: pct_viable should be None (not 0.0 or 100.0) when no "
        "classification was ever performed - a numeric 0 or 100 would "
        "look like a real measurement rather than 'not computed'"
    )
    assert summary["n_unclassified"] == 3

    log.info(f"  [OK] summary: {summary}")
    log.info("  Summarise (no threshold) PASSED")
    return True


def test_summarise_with_threshold():
    log.info("TEST 5: summarise_viability with a real threshold")
    from core.viability import classify_viability, summarise_viability

    # 3 clearly small (non-viable), 7 clearly large (viable)
    small_area = 3.14159265 * 2**2   # d=4um
    large_area = 3.14159265 * 10**2  # d=20um
    spots = make_spots([small_area] * 3 + [large_area] * 7)

    classify_viability(spots, diameter_threshold_um=10.0)
    summary = summarise_viability(spots)

    assert summary["n_viable"] == 7
    assert summary["n_nonviable"] == 3
    assert summary["n_unclassified"] == 0
    assert abs(summary["pct_viable"] - 70.0) < 0.01

    log.info(f"  [OK] {summary}")
    log.info("  Summarise (with threshold) PASSED")
    return True


def test_summarise_excludes_failed_segmentation():
    log.info("TEST 6: summarise_viability ignores failed/invalid detections")
    from config import SpotRecord
    from core.viability import classify_viability, summarise_viability

    good_area = 3.14159265 * 10**2
    spots = [
        SpotRecord(fov_id="ROI-1", frame_index=0, elapsed_min=0.0, spot_id=1,
                  area_um2=good_area, segmentation_ok=True),
        SpotRecord(fov_id="ROI-1", frame_index=0, elapsed_min=0.0, spot_id=-1,
                  area_um2=0.0, segmentation_ok=False, error_msg="failed"),
    ]
    classify_viability(spots, diameter_threshold_um=10.0)
    summary = summarise_viability(spots)

    assert summary["n_viable"] + summary["n_nonviable"] == 1, (
        "Failed-segmentation spot should not be counted in viability totals"
    )
    log.info(f"  [OK] {summary} - failed spot correctly excluded")
    log.info("  Exclusion of failed spots PASSED")
    return True


def test_suggest_threshold():
    log.info("TEST 7: suggest_threshold_from_distribution returns a sane starting point")
    from core.viability import classify_viability, suggest_threshold_from_distribution
    import numpy as np

    rng = np.random.default_rng(0)
    # Simulate mostly-normal cells (d~12um) with a small tail of shrunken ones (d~5um)
    normal_d = rng.normal(12, 1.5, 90)
    shrunken_d = rng.normal(5, 0.5, 10)
    all_d = np.concatenate([normal_d, shrunken_d])
    areas = 3.14159265 * (all_d / 2) ** 2
    spots = make_spots(list(areas))

    suggestion = suggest_threshold_from_distribution(spots, percentile=10.0)

    assert 3.0 < suggestion < 10.0, (
        f"Suggested threshold {suggestion:.2f}um should fall somewhere between "
        f"the shrunken and normal populations for this synthetic bimodal case"
    )
    log.info(f"  [OK] suggested threshold: {suggestion:.2f}um (10th percentile of distribution)")
    log.info("  Suggest threshold PASSED")
    return True


def test_pipeline_integration_inactive():
    log.info("TEST 8: full pipeline path with viability inactive (default)")
    # Confirms the wiring in pipeline.py/retrack.py doesn't accidentally
    # activate classification when no threshold is passed through.
    from core.viability import classify_viability
    from core.tracker import track_fov
    from config import TrackingConfig, SpotRecord
    from collections import defaultdict

    spots = []
    for f in range(5):
        for c in range(5):
            spots.append(SpotRecord(
                fov_id="ROI-1", frame_index=f, elapsed_min=f * 5.0, spot_id=c + 1,
                centroid_x_px=50.0 + c * 20, centroid_y_px=30.0,
                area_um2=100.0, circularity=0.9, segmentation_ok=True,
            ))
    by_frame = defaultdict(list)
    for s in spots:
        by_frame[s.frame_index].append(s)
    frame_spots = [by_frame[i] for i in range(5)]

    cfg = TrackingConfig(max_distance_px=30.0, min_track_length_frames=3)
    tracked, tracks = track_fov(frame_spots, cfg, 0.377, 5)
    classify_viability(tracked, diameter_threshold_um=None)

    assert all(s.is_viable is None for s in tracked)
    assert all(s.equivalent_diameter_um > 0 for s in tracked)

    log.info("  [OK] full track_fov -> classify_viability path stays inactive by default")
    log.info("  Pipeline integration PASSED")
    return True


def run_all():
    tests = [
        test_equivalent_diameter_calculation,
        test_inactive_by_default,
        test_inactive_warning_fires_every_call_not_just_first,
        test_classification_correctness,
        test_summarise_no_threshold,
        test_summarise_with_threshold,
        test_summarise_excludes_failed_segmentation,
        test_suggest_threshold,
        test_pipeline_integration_inactive,
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
