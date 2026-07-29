"""
core/viability.py - diameter-based cell viability classification.

Based on a technique described in the HSEC/HemaChip source publication:
cells below a diameter threshold are classified as putative non-viable
(shrunken/fragmented/apoptotic), cells at or above it as viable.

Deliberately NOT auto-populated with the source paper's own threshold
(9.5um). The paper's own text flags its diameter calibration as likely
~2x larger than literature values for the same cell type, making its
raw 9.5um threshold uninterpretable without knowing the true size of
that calibration error - which the paper itself does not pin down
precisely (only "roughly double"). Borrowing an unvalidated, possibly-
miscalibrated threshold from a different dataset would produce a
viable_pct number that looks authoritative in a CSV while resting on
an unchecked assumption.

This module implements the classification logic completely - it is
INACTIVE by default (threshold=None) and stays that way until the
caller supplies a diameter_threshold_um value, ideally chosen by
inspecting this dataset's own images rather than imported from
elsewhere.

Usage:
    from core.viability import classify_viability, summarise_viability

    spots = classify_viability(spots, diameter_threshold_um=None)  # inactive
    spots = classify_viability(spots, diameter_threshold_um=8.0)   # active
    summary = summarise_viability(spots)
"""
from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)


def equivalent_diameter_um(area_um2):
    """
    Diameter of a circle with the same area as the given area - the
    standard morphology "equivalent diameter" measure. This is what a
    viability diameter threshold should be compared against, not raw
    area, since diameter is the unit the threshold is defined in.
    """
    if area_um2 <= 0:
        return 0.0
    return float(2.0 * np.sqrt(area_um2 / np.pi))


def classify_viability(spots, diameter_threshold_um=None):
    """
    Classify each spot as viable/non-viable by equivalent diameter.

    Args:
        spots                 : list[SpotRecord]
        diameter_threshold_um : cells with equivalent_diameter_um below
                                 this value are classified non-viable.
                                 If None (default), no classification is
                                 performed - every spot's is_viable stays
                                 None and equivalent_diameter_um is still
                                 computed and populated (useful on its
                                 own regardless of viability).

    Returns:
        The same list, mutated in place (equivalent_diameter_um always
        set; is_viable set only if diameter_threshold_um is given).
    """
    # NOTE: previously used a bare module-level global (_WARNED_INACTIVE)
    # to log this message only once ever per Python PROCESS - meaning in
    # a real multi-ROI pipeline run, this warning appeared for the FIRST
    # ROI only and silently vanished for every other ROI in the run (e.g.
    # ROIs 2-225 of a full chip), even though it's relevant context for
    # every one of them. A bare module global also meant test runs or any
    # other code calling this function earlier in the same process could
    # silently suppress the warning with no relationship to the current
    # call. Fixed to log at INFO level on every call where classification
    # is inactive - this fires once per FOV (not once per spot), so for a
    # 225-ROI run that's 225 short log lines, not spam.
    if diameter_threshold_um is None:
        log.info(
            "  Viability classification INACTIVE for this FOV (no "
            "diameter_threshold_um configured). equivalent_diameter_um is "
            "still computed; is_viable will be None on every spot until a "
            "validated threshold is supplied. See core/viability.py for why "
            "this does not default to the source paper's own 9.5um value."
        )

    for s in spots:
        s.equivalent_diameter_um = equivalent_diameter_um(s.area_um2)
        if diameter_threshold_um is not None:
            s.is_viable = s.equivalent_diameter_um >= diameter_threshold_um
        else:
            s.is_viable = None

    return spots


def summarise_viability(spots):
    """
    Compute viable/non-viable counts and percentage from a list of
    classified SpotRecords (i.e. after classify_viability() has been
    called with a real threshold).

    Returns a dict with keys: n_viable, n_nonviable, n_unclassified,
    pct_viable. If no spots have been classified (is_viable is None on
    all of them, i.e. no threshold was ever set), pct_viable is None
    rather than a misleading 0.0 or 100.0.
    """
    valid = [s for s in spots if s.segmentation_ok and s.spot_id > 0]
    classified = [s for s in valid if s.is_viable is not None]
    unclassified = len(valid) - len(classified)

    if not classified:
        return {
            "n_viable": 0, "n_nonviable": 0,
            "n_unclassified": unclassified, "pct_viable": None,
        }

    n_viable = sum(1 for s in classified if s.is_viable)
    n_nonviable = len(classified) - n_viable
    pct_viable = round(100.0 * n_viable / len(classified), 2)

    return {
        "n_viable": n_viable, "n_nonviable": n_nonviable,
        "n_unclassified": unclassified, "pct_viable": pct_viable,
    }


def suggest_threshold_from_distribution(spots, percentile=10.0):
    """
    Suggests a candidate diameter threshold from this dataset's OWN size
    distribution, as a starting point for visual validation - not a
    substitute for it. Returns the given low percentile of equivalent
    diameters across all segmented cells (default: 10th percentile).

    This is deliberately framed as a suggestion to check against real
    images, not an automatic threshold - the actual biological cutoff
    between "small but viable" and "shrunken/non-viable" cannot be
    determined from the size distribution's shape alone.
    """
    valid = [s for s in spots if s.segmentation_ok and s.spot_id > 0]
    if not valid:
        return 0.0
    diameters = [equivalent_diameter_um(s.area_um2) for s in valid]
    suggestion = float(np.percentile(diameters, percentile))
    log.info(
        f"  Suggested viability threshold candidate: {suggestion:.2f} um "
        f"({percentile:.0f}th percentile of {len(diameters)} cells' equivalent "
        f"diameter). This is a STARTING POINT for visual validation against "
        f"real images, not a validated threshold - inspect late-timepoint "
        f"frames near this value before using it."
    )
    return suggestion
