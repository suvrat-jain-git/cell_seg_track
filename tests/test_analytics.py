"""tests/test_analytics.py - tests for core/analytics.py, the ROI-level
dashboard generator. Covers generic mode, spatial/gradient mode, and
graceful degradation when HemaChip metadata is absent."""
import sys
import tempfile
import logging
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def make_generic_populations(n=5, seed=0):
    from config import PopulationRecord
    rng = np.random.default_rng(seed)
    return [
        PopulationRecord(
            fov_id=f"ROI-{i+1}", cell_count_mean=rng.uniform(20, 150),
            n_tracks_total=int(rng.uniform(10, 200)),
            pct_tracks_survived=rng.uniform(5, 40),
            mean_lifespan_min=rng.uniform(30, 90),
            mean_velocity_um_min=rng.uniform(0.5, 3),
            mean_confinement_ratio=rng.uniform(0.3, 0.6),
        )
        for i in range(n)
    ]


def make_gradient_populations(seed=0):
    """6 ROIs on chip1 with a real density gradient, 6 on chip2 flat -
    mirrors a real HemaChip gradient-vs-homogeneous comparison."""
    from config import PopulationRecord
    rng = np.random.default_rng(seed)
    populations = []
    for col in range(6):
        density = 150 - col * 20 + rng.normal(0, 8)
        pop = PopulationRecord(
            fov_id=f"ROI-{col+1}", grid_row=0, grid_col=col, condition="chip1",
            cell_count_mean=max(5, density),
            n_tracks_total=int(max(5, density) * 5),
            pct_tracks_survived=rng.uniform(10, 25),
            mean_lifespan_min=rng.uniform(40, 90),
            mean_velocity_um_min=rng.uniform(1.0, 2.2),
            mean_confinement_ratio=rng.uniform(0.4, 0.55),
        )
        pop.adhesion_rate = max(0.01, density / 1000)
        pop.spreading_rate = rng.normal(0, 0.0005)
        pop.endpoint_value = max(5, density)
        pop.endpoint_variability = rng.uniform(0.03, 0.45)
        populations.append(pop)
    for col in range(6):
        density = 60 + rng.normal(0, 6)
        pop = PopulationRecord(
            fov_id=f"ROI-{col+7}", grid_row=0, grid_col=col, condition="chip2",
            cell_count_mean=max(5, density),
            n_tracks_total=int(max(5, density) * 5),
            pct_tracks_survived=rng.uniform(15, 30),
            mean_lifespan_min=rng.uniform(50, 80),
            mean_velocity_um_min=rng.uniform(1.0, 1.8),
            mean_confinement_ratio=rng.uniform(0.42, 0.52),
        )
        pop.adhesion_rate = max(0.01, density / 1200)
        pop.spreading_rate = rng.normal(0, 0.0003)
        pop.endpoint_value = max(5, density)
        pop.endpoint_variability = rng.uniform(0.1, 0.3)
        populations.append(pop)
    return populations


def test_generic_mode():
    log.info("TEST 1: Generic mode (no grid metadata) - one file per metric")
    from core.analytics import generate_roi_dashboard

    with tempfile.TemporaryDirectory() as tmp:
        populations = make_generic_populations(n=5)
        written = generate_roi_dashboard(populations, Path(tmp))

        # 6 generic metrics defined: cell_count_mean, n_tracks_total (line),
        # pct_tracks_survived (pie), mean_lifespan_min, mean_velocity_um_min,
        # mean_confinement_ratio (box) = 6 separate files, no spatial/
        # comparison charts since there's no grid data.
        assert len(written) == 6, f"Expected 6 chart files, got {len(written)}: {[w.name for w in written]}"
        names = {w.name for w in written}
        assert "cell_count_mean_trend.png" in names
        assert "n_tracks_total_trend.png" in names
        assert "pct_tracks_survived_proportion.png" in names
        assert "mean_lifespan_min_distribution.png" in names
        for w in written:
            assert w.exists() and w.stat().st_size > 0
        log.info(f"  [OK] {len(written)} separate chart files: {sorted(names)}")
    log.info("  Generic mode PASSED")
    return True


def test_gradient_mode_full():
    log.info("TEST 2: Gradient mode (grid + 2 chip conditions), separate files")
    from core.analytics import generate_roi_dashboard

    with tempfile.TemporaryDirectory() as tmp:
        populations = make_gradient_populations()
        written = generate_roi_dashboard(
            populations, Path(tmp), title_prefix="TEST",
            gradient_edges={"top": "Factor A", "bottom": "Factor B"},
        )

        names = {w.name for w in written}
        # Per-metric charts, spatial heatmaps (one file per metric per
        # condition), and per-metric condition comparisons should all be
        # SEPARATE files, never combined into one multi-panel figure.
        assert "cell_count_mean_trend.png" in names
        assert any(n.startswith("heatmap_cell_count_mean_chip1") for n in names)
        assert any(n.startswith("heatmap_cell_count_mean_chip2") for n in names)
        assert any(n.startswith("comparison_adhesion_rate") for n in names)
        assert any(n.startswith("comparison_endpoint_value") for n in names)
        for w in written:
            assert w.exists() and w.stat().st_size > 0, f"{w} is missing or empty"

        log.info(f"  [OK] {len(written)} separate chart files (spot-checked heatmap + comparison naming)")
    log.info("  Gradient mode PASSED")
    return True


def test_percentage_metrics_included_in_heatmaps_and_comparisons():
    log.info("TEST 2b: percentage metrics (pct_tracks_survived) appear in heatmaps and comparisons (regression test)")
    # Regression test for a real bug found during codebase audit:
    # percentage/proportion metrics (pct_tracks_survived, pct_viable) were
    # completely excluded from BOTH spatial heatmaps and condition
    # comparison charts, with no technical reason - box plots and
    # heatmaps both handle 0-100% data just as well as any other numeric
    # metric. This meant "does survival rate differ between chip
    # conditions" or "is survival spatially uniform across the chip" -
    # both directly relevant questions this feature exists to answer -
    # could never be visualised, only the single averaged donut chart.
    from core.analytics import generate_roi_dashboard
    from config import PopulationRecord
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(3)
        populations = []
        for i in range(5):
            populations.append(PopulationRecord(
                fov_id=f"ROI-{i+1}", grid_row=0, grid_col=i, condition="chip1",
                cell_count_mean=100.0, pct_tracks_survived=rng.uniform(10, 20),
            ))
        for i in range(5):
            populations.append(PopulationRecord(
                fov_id=f"ROI-{i+6}", grid_row=0, grid_col=i, condition="chip2",
                cell_count_mean=50.0, pct_tracks_survived=rng.uniform(30, 40),
            ))

        written = generate_roi_dashboard(populations, Path(tmp), title_prefix="test")
        names = {w.name for w in written}

        assert "comparison_pct_tracks_survived.png" in names, (
            "BUG REGRESSION: pct_tracks_survived missing from condition comparisons - "
            "percentage metrics are being excluded again"
        )
        assert any("heatmap_pct_tracks_survived" in n for n in names), (
            "BUG REGRESSION: pct_tracks_survived missing from spatial heatmaps - "
            "percentage metrics are being excluded again"
        )
        # The averaged donut chart should STILL exist too - this fix adds
        # heatmap/comparison views, it doesn't replace the donut.
        assert "pct_tracks_survived_proportion.png" in names, (
            "The single averaged donut chart should still be generated alongside "
            "the new heatmap/comparison views, not replaced by them"
        )

        log.info("  [OK] pct_tracks_survived appears in donut, heatmaps, AND comparison chart")
    log.info("  Percentage metric inclusion PASSED")
    return True


def test_gradient_edges_optional():
    log.info("TEST 2b: gradient_edges is optional - works fine without it")
    from core.analytics import generate_roi_dashboard

    with tempfile.TemporaryDirectory() as tmp:
        populations = make_gradient_populations()
        written = generate_roi_dashboard(populations, Path(tmp))  # no gradient_edges
        assert len(written) > 0
        for w in written:
            assert w.exists() and w.stat().st_size > 0
        log.info(f"  [OK] {len(written)} charts generated with no gradient_edges given")
    log.info("  gradient_edges optional PASSED")
    return True


def test_single_chip_no_comparison():
    log.info("TEST 3: Single chip condition - no comparison chart")
    from core.analytics import generate_roi_dashboard
    from config import PopulationRecord
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(2)
        populations = [
            PopulationRecord(fov_id=f"ROI-{i+1}", grid_row=0, grid_col=i, condition="chip1",
                            cell_count_mean=rng.uniform(20, 150))
            for i in range(5)
        ]
        written = generate_roi_dashboard(populations, Path(tmp))
        names = {w.name for w in written}
        assert not any(n.startswith("comparison_") for n in names), (
            "Should not generate any comparison chart with only 1 chip condition present"
        )
        assert any(n.startswith("heatmap_cell_count_mean_chip1") for n in names)
        log.info(f"  [OK] correctly skipped comparison charts with only 1 condition")
    log.info("  Single chip PASSED")
    return True


def test_empty_populations():
    log.info("TEST 4: Empty population list")
    from core.analytics import generate_roi_dashboard
    with tempfile.TemporaryDirectory() as tmp:
        written = generate_roi_dashboard([], Path(tmp))
        assert written == [], "Should return empty list, not raise, for empty input"
        log.info("  [OK] empty input handled gracefully, no crash")
    log.info("  Empty populations PASSED")
    return True


def test_single_roi_no_spatial():
    log.info("TEST 5: Single ROI - spatial charts need 2+ ROIs")
    from core.analytics import generate_roi_dashboard
    from config import PopulationRecord

    with tempfile.TemporaryDirectory() as tmp:
        populations = [PopulationRecord(fov_id="ROI-1", grid_row=0, grid_col=0,
                                        condition="chip1", cell_count_mean=100)]
        written = generate_roi_dashboard(populations, Path(tmp))
        names = {w.name for w in written}
        assert not any(n.startswith("heatmap_") for n in names), (
            "Should not attempt a spatial heatmap with only 1 ROI"
        )
        assert "cell_count_mean_trend.png" in names
        log.info("  [OK] single ROI correctly skips spatial heatmap, still produces per-ROI charts")
    log.info("  Single ROI PASSED")
    return True


def test_missing_matplotlib_graceful():
    log.info("TEST 6: Missing matplotlib handled gracefully")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    from core.analytics import generate_roi_dashboard
    populations = make_generic_populations(n=3)

    builtins.__import__ = fake_import
    try:
        with tempfile.TemporaryDirectory() as tmp:
            written = generate_roi_dashboard(populations, Path(tmp))
            assert written == [], "Should return empty list, not raise, when matplotlib is unavailable"
    finally:
        builtins.__import__ = real_import

    log.info("  [OK] missing matplotlib returns empty list instead of raising")
    log.info("  Missing matplotlib PASSED")
    return True


def run_all():
    tests = [
        test_generic_mode,
        test_gradient_mode_full,
        test_percentage_metrics_included_in_heatmaps_and_comparisons,
        test_gradient_edges_optional,
        test_single_chip_no_comparison,
        test_empty_populations,
        test_single_roi_no_spatial,
        test_missing_matplotlib_graceful,
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
