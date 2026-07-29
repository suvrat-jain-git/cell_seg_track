"""tests/test_visualizer.py - tests for core/visualizer.py. This file had
no test coverage at all before a codebase audit found and fixed a real
color-collision bug in _outline_overlay(); this suite covers that fix plus
basic sanity checks on the other visualisation entry points."""
import sys
import tempfile
import logging
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def make_mask_with_n_cells(n, size=60):
    """Build a synthetic label mask with n small square cells in a grid."""
    mask = np.zeros((size, size), dtype=np.uint16)
    per_row = max(1, int(np.ceil(np.sqrt(n))))
    for i in range(n):
        row, col = i // per_row, i % per_row
        y = 5 + row * (size // per_row)
        x = 5 + col * (size // per_row)
        y2, x2 = min(y + 6, size), min(x + 6, size)
        mask[y:y2, x:x2] = i + 1
    return mask


def test_outline_overlay_distinct_colors_beyond_three_cells():
    log.info("TEST 1: _outline_overlay gives distinct colors to >3 cells (regression test)")
    # Regression test for a real bug found during codebase audit:
    # _outline_overlay() previously cycled through only 3 hardcoded colors
    # via `label % 3`, so any two cells whose labels differed by a
    # multiple of 3 received the IDENTICAL outline color and became
    # visually indistinguishable in the segmentation QC overlay. Real
    # HSEC frames typically have 10-150+ cells, so this collision was
    # near-guaranteed on every real frame, not an edge case. Fixed to
    # reuse _distinct_colors() (the same HSV-sweep function already used
    # correctly for the track overlay).
    from core.visualizer import _outline_overlay

    mask = make_mask_with_n_cells(6)
    overlay = _outline_overlay(mask)

    nonzero = overlay[overlay[..., 3] > 0]
    assert len(nonzero) > 0, "Overlay should have some outlined pixels for 6 cells"

    colors_seen = set(tuple(np.round(row[:3], 3)) for row in nonzero)
    assert len(colors_seen) == 6, (
        f"BUG REGRESSION: expected 6 distinct colors for 6 cells, got "
        f"{len(colors_seen)} - _outline_overlay may have regressed to a "
        f"small fixed color cycle that collides once cell count exceeds it"
    )

    log.info(f"  [OK] 6 cells -> {len(colors_seen)} distinct outline colors")
    log.info("  Outline overlay distinct colors PASSED")
    return True


def test_outline_overlay_many_cells():
    log.info("TEST 2: _outline_overlay scales to realistic cell counts (~140)")
    from core.visualizer import _outline_overlay

    mask = make_mask_with_n_cells(140, size=400)
    overlay = _outline_overlay(mask)

    nonzero = overlay[overlay[..., 3] > 0]
    assert len(nonzero) > 0
    colors_seen = set(tuple(np.round(row[:3], 2)) for row in nonzero)
    # Some cells at 400x400/140 density may be too close/overlapping to
    # all render distinctly at this coarse synthetic layout, but we
    # should see a large number of genuinely different colors, not a
    # small repeating cycle.
    assert len(colors_seen) > 20, (
        f"Expected many distinct colors at realistic cell density, only "
        f"got {len(colors_seen)} - possible regression to a small color cycle"
    )

    log.info(f"  [OK] ~140 cells -> {len(colors_seen)} distinct outline colors observed")
    log.info("  Outline overlay realistic scale PASSED")
    return True


def test_outline_overlay_empty_mask():
    log.info("TEST 3: _outline_overlay handles an empty mask (no cells)")
    from core.visualizer import _outline_overlay

    mask = np.zeros((60, 60), dtype=np.uint16)
    overlay = _outline_overlay(mask)

    assert overlay.shape == (60, 60, 4)
    assert not (overlay[..., 3] > 0).any(), "Empty mask should produce no visible outline pixels"

    log.info("  [OK] empty mask produces a valid, fully-transparent overlay, no crash")
    log.info("  Outline overlay empty mask PASSED")
    return True


def test_distinct_colors_helper():
    log.info("TEST 4: _distinct_colors produces genuinely distinct RGB tuples")
    from core.visualizer import _distinct_colors

    colors = _distinct_colors(50)
    assert len(colors) == 50
    unique = set(tuple(round(c, 4) for c in col) for col in colors)
    assert len(unique) == 50, f"Expected 50 unique colors, got {len(unique)} distinct values"

    assert _distinct_colors(0) == []

    log.info("  [OK] 50 requested colors are all genuinely distinct; n=0 returns empty list")
    log.info("  _distinct_colors helper PASSED")
    return True


def test_qc_summary_grid_single_fov_no_crash():
    log.info("TEST 5: generate_qc_summary_grid with exactly 1 FOV (subplot-shape edge case)")
    # Verifies the same class of bug found and fixed in core/analytics.py
    # (plt.subplots returning a bare Axes instead of an array when the
    # grid is 1x1) does NOT affect this function - checked directly
    # rather than assumed, since the guard condition here (`n > 1`) looks
    # superficially similar to the one that was buggy elsewhere.
    from core.visualizer import generate_qc_summary_grid

    with tempfile.TemporaryDirectory() as tmp:
        fov_results = {"ROI-1": ([], None)}
        generate_qc_summary_grid(fov_results, Path(tmp))
        out_path = Path(tmp) / "qc_summary_grid.png"
        assert out_path.exists(), "Should produce a QC summary grid image even for a single FOV"

    log.info("  [OK] single-FOV case does not crash, produces output")
    log.info("  QC summary grid single FOV PASSED")
    return True


def run_all():
    tests = [
        test_outline_overlay_distinct_colors_beyond_three_cells,
        test_outline_overlay_many_cells,
        test_outline_overlay_empty_mask,
        test_distinct_colors_helper,
        test_qc_summary_grid_single_fov_no_crash,
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
