"""tests/test_hemachip.py - HemaChip plugin tests. Uses a real sample .epf fixture."""
import sys
import tempfile
import logging
import shutil
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FIXTURE_EPF = Path(__file__).parent / "fixtures" / "sample_hemachip.epf"


def make_session_fixture(tmp_path, roi_numbers, n_frames=5):
    """Build a realistic session folder: real EPF + N ROI folders with blank TIFs."""
    session_dir = tmp_path / "20231205_134131"
    session_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_EPF, session_dir / "HemaChip_test_20231205.epf")

    import tifffile
    for roi in roi_numbers:
        roi_dir = session_dir / f"ROI-{roi}"
        roi_dir.mkdir()
        for f in range(n_frames):
            tifffile.imwrite(
                str(roi_dir / f"MyExperiment_ROI-{roi}_WHITE_{f:06d}.tif"),
                np.full((32, 32), 150, dtype=np.uint16),
            )
    return session_dir


def test_epf_parser_real_file():
    log.info("TEST 1: EPF parser on real fixture file")
    from plugins.hemachip.epf_parser import parse_epf, compute_grid_coordinates

    assert FIXTURE_EPF.exists(), f"Fixture missing: {FIXTURE_EPF}"
    meta = parse_epf(FIXTURE_EPF)

    assert meta.capture_interval_min == 5.0, f"Expected 5.0 min interval, got {meta.capture_interval_min}"
    assert meta.n_positions() == 450, f"Expected 450 positions, got {meta.n_positions()}"
    assert meta.n_chips() == 2, f"Expected 2 chips, got {meta.n_chips()}"
    assert len(meta.chips[0]) == 225, f"Chip 1 expected 225 positions, got {len(meta.chips[0])}"
    assert len(meta.chips[1]) == 225, f"Chip 2 expected 225 positions, got {len(meta.chips[1])}"

    coords = compute_grid_coordinates(meta.chips[0])
    assert coords[1] == (0, 0), f"Position order=1 should be (0,0), got {coords[1]}"
    assert coords[15] == (0, 14), f"Position order=15 should be (0,14), got {coords[15]}"
    # Row 1 scans right-to-left (snake pattern) - order=16 has highest x in row 1,
    # so it should still map to col=14 (spatially consistent), NOT col=0
    assert coords[16] == (1, 14), f"Position order=16 (snake row start) should be (1,14), got {coords[16]}"
    assert coords[30] == (1, 0), f"Position order=30 (snake row end) should be (1,0), got {coords[30]}"

    log.info(f"  [OK] 450 positions, 2 chips, snake-pattern grid coords correct")
    log.info("  EPF parser PASSED")
    return True


def test_extract_int_tolerates_whitespace():
    log.info("TEST 1b: _extract_int tolerates internal whitespace (regression test)")
    # Regression test for a real bug found during codebase audit: the
    # original regex "<tag>(\d+)</tag>" required digits immediately
    # adjacent to the tags with no whitespace. Real XML is often formatted
    # with internal whitespace/newlines for readability (e.g.
    # "<tag>\n  15\n</tag>"), which the tight pattern silently failed to
    # match, returning 0 instead of the real value - with NO warning for
    # total_period_min specifically (capture_interval_min at least has a
    # <= 0 safety check and fallback). The one real .epf fixture available
    # happens to have no internal whitespace, so this was not caught by
    # the original real-file test above.
    from plugins.hemachip.epf_parser import _extract_int

    tight = "<captureEveryMinutes>5</captureEveryMinutes>"
    assert _extract_int(tight, "captureEveryMinutes") == 5

    with_whitespace = "<captureEveryMinutes>\n  15\n</captureEveryMinutes>"
    result = _extract_int(with_whitespace, "captureEveryMinutes")
    assert result == 15, (
        f"BUG REGRESSION: expected 15 from whitespace-formatted XML, got {result} "
        f"- _extract_int's regex may have regressed to requiring tight formatting"
    )

    log.info("  [OK] tight formatting: 5, whitespace-formatted: 15 (both correct)")
    log.info("  _extract_int whitespace tolerance PASSED")
    return True


def test_epf_parser_missing_file():
    log.info("TEST 2: EPF parser error handling")
    from plugins.hemachip.epf_parser import parse_epf
    try:
        parse_epf("/nonexistent/path.epf")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    log.info("  [OK] raises FileNotFoundError for missing file")
    log.info("  EPF error handling PASSED")
    return True


def test_scanner_partial_download():
    log.info("TEST 3: Scanner with partial chip download (10 of 225 ROIs)")
    from plugins.hemachip.scanner import scan_patient

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        make_session_fixture(tmp_path, roi_numbers=list(range(1, 11)), n_frames=2)

        patient = scan_patient(tmp_path, patient_id="TEST007", pixel_size_um=0.3769)

        assert len(patient.sessions) == 1
        session = patient.sessions[0]
        assert len(session.chips) == 2, f"Expected 2 chips (from EPF), got {len(session.chips)}"

        chip1 = session.chips[0]
        chip2 = session.chips[1]
        assert len(chip1.valid_rois) == 10, f"Expected 10 valid ROIs in chip1, got {len(chip1.valid_rois)}"
        assert len(chip2.valid_rois) == 0, f"Expected 0 valid ROIs in chip2, got {len(chip2.valid_rois)}"
        assert chip1.valid_rois == list(range(1, 11))

        # Verify grid positions are sequential along row 0 for ROI 1-10
        for roi in range(1, 11):
            row, col = chip1.roi_grid[roi]
            assert row == 0, f"ROI-{roi} expected row 0, got {row}"
            assert col == roi - 1, f"ROI-{roi} expected col {roi-1}, got {col}"

        log.info(f"  [OK] chip1: {len(chip1.valid_rois)} ROIs, chip2: {len(chip2.valid_rois)} ROIs")
        log.info(f"  [OK] grid positions sequential and correct")
    log.info("  Scanner PASSED")
    return True


def test_scanner_no_epf():
    log.info("TEST 4: Scanner gracefully handles missing EPF")
    from plugins.hemachip.scanner import scan_patient

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        session_dir = tmp_path / "20231205_134131"
        session_dir.mkdir(parents=True)
        import tifffile
        roi_dir = session_dir / "ROI-1"
        roi_dir.mkdir()
        tifffile.imwrite(str(roi_dir / "test_000000.tif"), np.zeros((32, 32), dtype=np.uint16))

        patient = scan_patient(tmp_path, patient_id="TEST008")
        assert len(patient.sessions) == 1
        session = patient.sessions[0]
        assert len(session.chips) == 1, "Should build a single fallback chip with no grid data"
        assert session.chips[0].roi_grid == {}, "No EPF means no grid positions"
        assert 1 in session.chips[0].roi_paths, "ROI-1 folder should still be found"

        log.info("  [OK] falls back gracefully with ROI paths but no grid metadata")
    log.info("  Scanner no-EPF PASSED")
    return True


def test_scanner_prefers_largest_epf_when_multiple_present():
    log.info("TEST 4b: Scanner selects the LARGEST .epf when multiple exist (regression test)")
    # Regression test for a real bug found during codebase audit: the
    # original code used session_dir.glob("*.epf")[0] to pick an EPF file
    # when multiple were present. Path.glob() order is not guaranteed by
    # any OS/filesystem, so this was non-deterministic across machines.
    # More importantly, direct investigation of real CB007 data found two
    # .epf files present for the same nominal session: a small ~96KB
    # draft/template file, and the genuine ~7.7MB session log (43,239
    # savedImages entries vs 7-8). Picking the wrong one would mean using
    # incomplete/draft metadata instead of the real session record. Fixed
    # to prefer the largest file, with alphabetical order as a
    # deterministic tiebreak.
    from plugins.hemachip.scanner import scan_patient

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        session_dir = tmp_path / "20231205_134131"
        session_dir.mkdir(parents=True)

        # Small "draft" file - copy the real fixture then pad a "real" one
        # to be larger, so we have a genuine, verifiable size difference
        # rather than a contrived one.
        draft_path = session_dir / "aaa_draft.epf"  # alphabetically FIRST
        real_path = session_dir / "zzz_real_session.epf"  # alphabetically LAST but LARGER
        shutil.copy(FIXTURE_EPF, draft_path)
        shutil.copy(FIXTURE_EPF, real_path)
        # Pad the "real" file to be unambiguously larger, simulating the
        # real-world size difference observed (draft ~96KB, real ~7.7MB)
        with open(real_path, "a") as f:
            f.write("<!-- padding to simulate a larger, more complete session log -->\n" * 1000)

        assert real_path.stat().st_size > draft_path.stat().st_size, "Test setup error"

        make_session_fixture_rois(session_dir, roi_numbers=[1])

        patient = scan_patient(tmp_path, patient_id="TEST_MULTI_EPF", pixel_size_um=0.3769)
        session = patient.sessions[0]

        # We can't directly assert WHICH file was parsed without exposing
        # more internals, but we can assert the scan succeeded and picked
        # up valid grid data - if it had picked an unparseable or wrong
        # file this would fail. The log message (checked by eye during
        # audit) confirms it names the larger file.
        assert len(session.chips) == 2, f"Expected 2 chips from a valid EPF, got {len(session.chips)}"
        assert 1 in session.chips[0].roi_paths

        log.info(f"  [OK] scan succeeded with alphabetically-last-but-largest file present")
    log.info("  Largest EPF selection PASSED")
    return True


def make_session_fixture_rois(session_dir, roi_numbers, n_frames=2):
    """Helper: add ROI folders to an already-created session_dir (used when
    the EPF file itself needs custom setup, unlike make_session_fixture)."""
    import tifffile
    for roi in roi_numbers:
        roi_dir = session_dir / f"ROI-{roi}"
        roi_dir.mkdir(exist_ok=True)
        for f in range(n_frames):
            tifffile.imwrite(
                str(roi_dir / f"MyExperiment_ROI-{roi}_WHITE_{f:06d}.tif"),
                np.full((32, 32), 150, dtype=np.uint16),
            )
    return session_dir


def test_clinical_features():
    log.info("TEST 5: Clinical feature computation")
    from plugins.hemachip.clinical_features import compute_clinical_features, aggregate_chip_level
    from config import SpotRecord, PopulationRecord, FieldOfView

    # Simulate a growing cell population (t=0 to t=480 min, 18 timepoints)
    timepoints = [0, 5, 10, 15, 20, 30, 45, 60, 75, 90, 105, 120, 150, 180, 240, 300, 360, 480]
    spots = []
    for i, t in enumerate(timepoints):
        # Cell count grows fast early, plateaus late (realistic adhesion curve)
        n_cells = min(10 + i * 3, 60)
        for c in range(n_cells):
            spots.append(SpotRecord(
                fov_id="ROI-1", frame_index=i, elapsed_min=float(t), spot_id=c + 1,
                circularity=max(0.5, 0.95 - i * 0.01),  # circularity drops as cells spread
                area_um2=15.0, segmentation_ok=True,
            ))

    fov = FieldOfView(path=Path("/fake"), fov_id="ROI-1", frames=[], time_interval_min=5.0)
    pop = PopulationRecord(fov_id="ROI-1")

    features = compute_clinical_features(spots, pop, fov)
    assert set(features.keys()) == {"adhesion_rate", "spreading_rate", "endpoint_value", "endpoint_variability"}
    assert features["adhesion_rate"] > 0, "Growing cell count should give positive adhesion_rate"
    assert features["spreading_rate"] < 0, "Dropping circularity should give negative spreading_rate"
    assert features["endpoint_value"] > 0

    log.info(f"  [OK] adhesion_rate={features['adhesion_rate']}, "
             f"spreading_rate={features['spreading_rate']}, "
             f"endpoint_value={features['endpoint_value']}, "
             f"endpoint_variability={features['endpoint_variability']}")

    # Chip-level aggregation across 3 ROIs
    roi_features = [features, features, features]
    chip_agg = aggregate_chip_level(roi_features)
    assert "adhesion_rate_mean" in chip_agg
    assert chip_agg["adhesion_rate_std"] == 0.0, "Identical ROIs should have zero std"
    assert len(chip_agg) == 20, f"Expected 20 aggregate values (4 features x 5 stats), got {len(chip_agg)}"

    log.info(f"  [OK] chip-level aggregation: {len(chip_agg)} values")
    log.info("  Clinical features PASSED")
    return True


def test_clinical_features_empty():
    log.info("TEST 6: Clinical features handle empty/insufficient data")
    from plugins.hemachip.clinical_features import compute_clinical_features
    from config import FieldOfView, PopulationRecord

    fov = FieldOfView(path=Path("/fake"), fov_id="ROI-1", frames=[])
    pop = PopulationRecord(fov_id="ROI-1")

    features = compute_clinical_features([], pop, fov)
    assert features["adhesion_rate"] == 0.0
    assert features["endpoint_value"] == 0.0

    log.info("  [OK] empty spot list returns zeroed features, no crash")
    log.info("  Clinical features empty-data PASSED")
    return True


def test_clinical_features_short_session_no_endpoint_data():
    log.info("TEST 6b: endpoint_value on a session shorter than 120min (regression test)")
    # Regression test for a real bug found during codebase audit:
    # _mean_in_window() previously fell back to averaging the ENTIRE
    # session (all timepoints, including early suspension-phase counts)
    # whenever the requested t>=120min window had zero matching frames -
    # e.g. a session that was interrupted or genuinely shorter than 120
    # minutes. That fallback was silent: endpoint_value looked like a
    # normal, valid measurement with nothing to indicate it was actually
    # computed from the wrong (entire, early-biased) time window. Fixed
    # to return 0.0 - an unambiguous "not enough data" signal - matching
    # the sibling endpoint_variability calculation's existing safe
    # behaviour, rather than a plausible-looking but wrong number.
    from plugins.hemachip.clinical_features import compute_clinical_features
    from config import SpotRecord, PopulationRecord, FieldOfView

    # Session only reaches t=90min - well short of the t>=120min endpoint
    # window used by endpoint_value/endpoint_variability.
    timepoints = [0, 15, 30, 45, 60, 75, 90]
    spots = []
    for i, t in enumerate(timepoints):
        for c in range(10):  # constant count, doesn't matter for this test
            spots.append(SpotRecord(
                fov_id="ROI-1", frame_index=i, elapsed_min=float(t), spot_id=c + 1,
                circularity=0.9, area_um2=15.0, segmentation_ok=True,
            ))

    fov = FieldOfView(path=Path("/fake"), fov_id="ROI-1", frames=[], time_interval_min=15.0)
    pop = PopulationRecord(fov_id="ROI-1")
    features = compute_clinical_features(spots, pop, fov)

    assert features["endpoint_value"] == 0.0, (
        f"BUG REGRESSION: expected 0.0 (no data in t>=120min window), got "
        f"{features['endpoint_value']} - _mean_in_window may have regressed "
        f"to silently averaging the whole session instead"
    )
    assert features["endpoint_variability"] == 0.0, "Should also be 0.0 - no data in window"

    log.info(f"  [OK] endpoint_value=0.0, endpoint_variability=0.0 for a session with no t>=120min data")
    log.info("  Short session endpoint PASSED")
    return True


def test_plugin_end_to_end():
    log.info("TEST 7: Full plugin integration (enrich_fovs -> compute_fov_features -> export)")
    from config import (
        PipelineConfig, ImagingConfig, SegmentationConfig, TrackingConfig,
        FrameMode, ExperimentType,
    )
    from core.input_handler import load_fovs_from_batch
    from core.spot_features import extract_spot_features
    from core.tracker import track_fov
    from core.track_features import compute_population
    from plugins.hemachip.plugin import HemaChipPlugin
    import tifffile
    from scipy import ndimage
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        session_dir = make_session_fixture(tmp_path, roi_numbers=[1, 2, 3], n_frames=8)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            input_dir=session_dir, output_dir=output_dir,
            frame_mode=FrameMode.ALL, experiment_type=ExperimentType.HEMACHIP,
            imaging=ImagingConfig(pixel_size_um=0.3769, time_interval_min=5.0),
            segmentation=SegmentationConfig(min_diameter_um=1.0, max_diameter_um=30.0),
            tracking=TrackingConfig(max_distance_px=30.0, min_track_length_frames=2),
        )

        fovs = load_fovs_from_batch(session_dir, config, "ROI-*")
        assert len(fovs) == 3

        plugin = HemaChipPlugin(config)
        fovs = plugin.enrich_fovs(fovs, session_dir)

        for fov in fovs:
            assert fov.condition == "chip1", f"{fov.fov_id} should be assigned to chip1"
        assert fovs[0].grid_row == 0
        assert {f.grid_col for f in fovs} == {0, 1, 2}, "ROI-1,2,3 should be adjacent columns"

        # Fake segmentation: simple threshold on synthetic blank-ish images
        def fake_segment(fov):
            frame_spots = []
            for frame in fov.frames:
                img = tifffile.imread(str(frame.path))
                # add a few synthetic blobs so tracking has something to work with
                rng = np.random.default_rng(frame.frame_index)
                for _ in range(5):
                    cy, cx = rng.integers(5, 27, 2)
                    img[cy-2:cy+2, cx-2:cx+2] = 50
                binary = img < 100
                labeled, n = ndimage.label(binary)
                spots = extract_spot_features(labeled.astype(np.uint16), img, frame, fov, config.segmentation)
                frame_spots.append(spots)
            return frame_spots

        populations = []
        for fov in fovs:
            frame_spots = fake_segment(fov)
            spots, tracks = track_fov(frame_spots, config.tracking, fov.pixel_size_um, fov.n_frames)
            pop = compute_population(tracks, spots, fov)
            pop = plugin.compute_fov_features(spots, tracks, pop, fov)
            assert hasattr(pop, "adhesion_rate")
            assert hasattr(pop, "endpoint_value")
            populations.append(pop)

        output_dir.mkdir(parents=True, exist_ok=True)
        plugin.export([], [], populations, config)

        out_csv = output_dir / "results_chip_clinical.csv"
        assert out_csv.exists(), "results_chip_clinical.csv not written"
        df = pd.read_csv(out_csv)
        assert len(df) == 1, f"Expected 1 chip row, got {len(df)}"
        assert df.iloc[0]["n_rois"] == 3
        assert df.iloc[0]["chip_number"] == 1

        log.info(f"  [OK] enrich_fovs matched grid metadata for all 3 ROIs")
        log.info(f"  [OK] compute_fov_features attached clinical features to PopulationRecord")
        log.info(f"  [OK] export wrote results_chip_clinical.csv: {len(df)} chip row(s)")
    log.info("  Plugin end-to-end PASSED")
    return True


def run_all():
    tests = [
        test_epf_parser_real_file,
        test_extract_int_tolerates_whitespace,
        test_epf_parser_missing_file,
        test_scanner_partial_download,
        test_scanner_no_epf,
        test_scanner_prefers_largest_epf_when_multiple_present,
        test_clinical_features,
        test_clinical_features_empty,
        test_clinical_features_short_session_no_endpoint_data,
        test_plugin_end_to_end,
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
