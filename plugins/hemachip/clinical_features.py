"""
plugins/hemachip/clinical_features.py - HemaChip clinical biomarker features.

Computes ROI-level temporal biomarkers from the core pipeline's per-frame
spot data, plus tracking-derived population features already computed by
core/track_features.py. This module adds the HemaChip-specific interpretation
layer on top - it does not duplicate core aggregation logic.

The 4 core clinical features:

  adhesion_rate         - slope of cell_count over t=0-60 min (cells/min).
                           Proxy for how fast HSECs adhere to the surface.
  spreading_rate        - slope of mean_circularity over t=30-120 min
                           (1/min, negative = spreading/flattening).
  endpoint_value         - mean cell_count over t=120min to end of session.
  endpoint_variability    - CV (std/mean) of cell_count over the same window.

Naming note (changed from plateau_value / stability_index):
  The HSEC source publication (Allenby group) reports that global cell
  density plateaus only after approximately 48-72 HOURS of culture, with
  gradient cultures continuing to change until roughly 48h and non-gradient
  conditions still expanding at 72h. An 8-hour imaging session - which is
  what a single CellFlow session currently covers - cannot capture a true
  steady state; direct inspection of real CB007 data confirmed most ROIs
  are still climbing at the t=480min session end, not plateaued. Calling
  these features "plateau_value"/"stability_index" would misrepresent an
  in-progress growth snapshot as steady-state biology. "endpoint_value"/
  "endpoint_variability" describe exactly what is measured - the state of
  the culture at the end of whatever session was imaged - without implying
  equilibrium has been reached. If/when multi-day imaging sessions are
  available, a genuine plateau-detection feature (e.g. based on where the
  growth curve's slope actually flattens, not a fixed clock time) would be
  a more accurate addition, kept separate from this endpoint measurement.

These require frame-level cell_count and circularity at specific elapsed
times, which is exactly what frame_mode=clinical's 18 timepoints are chosen
to support (see core/input_handler.py CLINICAL_TIMEPOINTS_MIN) - but this
function works from whatever frames were actually processed, using whatever
overlap exists with the target time windows. It does not require clinical
frame_mode specifically; frame_mode=all works too, just with denser input.
"""
from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)


def compute_clinical_features(spots, population, fov) -> dict:
    """
    Compute the 4 HemaChip clinical features for one ROI/FOV.

    Args:
        spots      : flat list[SpotRecord] for this FOV, all processed frames
        population : PopulationRecord already computed by core/track_features.py
                     (used for tracking-derived context, not re-derived here)
        fov        : FieldOfView (for time_interval_min context)

    Returns:
        dict with keys: adhesion_rate, spreading_rate, endpoint_value,
        endpoint_variability. Values are 0.0 if insufficient data in a
        given time window (not None - keeps downstream CSV columns numeric).
    """
    valid = [s for s in spots if s.segmentation_ok and s.spot_id > 0]
    if not valid:
        return _empty_result()

    by_frame = {}
    for s in valid:
        by_frame.setdefault(s.frame_index, []).append(s)

    frames = sorted(by_frame.keys())
    times = np.array([by_frame[f][0].elapsed_min for f in frames])
    counts = np.array([len(by_frame[f]) for f in frames], dtype=float)
    mean_circ = np.array([
        float(np.mean([s.circularity for s in by_frame[f]])) for f in frames
    ])

    adhesion_rate = _slope_in_window(times, counts, 0, 60)
    spreading_rate = _slope_in_window(times, mean_circ, 30, 120)
    endpoint_value = _mean_in_window(times, counts, 120, None)
    endpoint_variability = _cv_in_window(times, counts, 120, None)

    return {
        "adhesion_rate": round(adhesion_rate, 5),
        "spreading_rate": round(spreading_rate, 5),
        "endpoint_value": round(endpoint_value, 3),
        "endpoint_variability": round(endpoint_variability, 4),
    }


def _empty_result():
    return {
        "adhesion_rate": 0.0,
        "spreading_rate": 0.0,
        "endpoint_value": 0.0,
        "endpoint_variability": 0.0,
    }


def _window_mask(times, lo, hi):
    if hi is None:
        return times >= lo
    return (times >= lo) & (times <= hi)


def _slope_in_window(times, values, lo, hi):
    mask = _window_mask(times, lo, hi)
    if mask.sum() < 2:
        return 0.0
    x = times[mask]
    y = values[mask]
    x_var = np.var(x)
    if x_var < 1e-10:
        return 0.0
    return float(np.cov(x, y)[0, 1] / x_var)


def _mean_in_window(times, values, lo, hi):
    # NOTE: previously fell back to np.mean(values) - i.e. the mean of the
    # ENTIRE session, including early suspension-phase data - whenever the
    # requested window had zero matching frames (e.g. a session shorter
    # than 120min, or an ROI with sparse/failed frames near the endpoint
    # window boundary). That fallback was silent: the returned value looked
    # identical to a genuine endpoint measurement, with nothing to indicate
    # it was actually a full-session average instead. Fixed to return 0.0
    # in that case, matching the already-safe behaviour of the sibling
    # _cv_in_window() function, so a caller/reviewer sees an obviously
    # "no data" value rather than a plausible-looking but wrong one.
    mask = _window_mask(times, lo, hi)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(values[mask]))


def _cv_in_window(times, values, lo, hi):
    mask = _window_mask(times, lo, hi)
    if mask.sum() < 2:
        return 0.0
    vals = values[mask]
    mu = np.mean(vals)
    if mu <= 0:
        return 0.0
    return float(np.std(vals) / mu)


def aggregate_chip_level(roi_features: list) -> dict:
    """
    Collapse a list of per-ROI clinical feature dicts (as produced by
    compute_clinical_features, one per ROI in a chip) into chip-level
    statistics: mean/std/min/max/median of each of the 4 features across
    all ROIs.

    Args:
        roi_features: list[dict], each dict having the 4 clinical feature keys

    Returns:
        dict with keys like "adhesion_rate_mean", "adhesion_rate_std", etc.
        20 values total (4 features x 5 statistics).
    """
    if not roi_features:
        return {}

    keys = ["adhesion_rate", "spreading_rate", "endpoint_value", "endpoint_variability"]
    result = {}
    for key in keys:
        values = np.array([r.get(key, 0.0) for r in roi_features], dtype=float)
        result[f"{key}_mean"] = round(float(np.mean(values)), 5)
        result[f"{key}_std"] = round(float(np.std(values)), 5)
        result[f"{key}_min"] = round(float(np.min(values)), 5)
        result[f"{key}_max"] = round(float(np.max(values)), 5)
        result[f"{key}_median"] = round(float(np.median(values)), 5)
    return result
