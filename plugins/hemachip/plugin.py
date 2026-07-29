"""
plugins/hemachip/plugin.py - HemaChip experiment plugin.

Wires epf_parser.py + scanner.py + clinical_features.py into the core
pipeline via the BasePlugin interface (base_plugin.py).

Hook usage:
  enrich_fovs()         - called once, after core FOV discovery. Re-scans
                           input_dir with scanner.scan_patient() to get
                           grid/chip metadata, then matches each already-
                           discovered FieldOfView (by fov_id == "ROI-<N>")
                           to its scanner record and fills in grid_row,
                           grid_col, x_mm, y_mm, condition.

  compute_fov_features() - called once per FOV, after core population
                           features are computed. Attaches the 4 clinical
                           features as extra attributes on the
                           PopulationRecord (population.adhesion_rate etc.)
                           so they flow straight into population.csv
                           without any change to core/exporter.py.

  export()               - called once at the end. Writes chip-level and
                           session-level aggregate CSVs on top of the core
                           spots/tracks/population.csv outputs.

Design note: core PopulationRecord is a plain Python object (not a
dataclass with a fixed __slots__), so attaching extra attributes at
runtime is safe and they serialise automatically via to_dict()'s use of
self.__dict__.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path

import pandas as pd

from plugins.base_plugin import BasePlugin
from plugins.hemachip.scanner import scan_patient
from plugins.hemachip.clinical_features import (
    compute_clinical_features, aggregate_chip_level,
)

log = logging.getLogger(__name__)

_ROI_ID_RE = re.compile(r"^ROI-(\d+)$", re.IGNORECASE)


class HemaChipPlugin(BasePlugin):

    def __init__(self, config):
        super().__init__(config)
        self._roi_to_chip = {}     # roi_number -> chip_number
        self._roi_clinical = {}    # fov_id -> clinical features dict (for export)
        self._patient = None

    # ── enrich_fovs ────────────────────────────────────────────────────────

    def enrich_fovs(self, fovs, input_dir):
        """
        Re-scan input_dir for EPF/grid metadata and attach it to each
        FieldOfView. If input_dir doesn't look like a HemaChip session
        folder (no .epf, no session-timestamp naming), FOVs pass through
        unchanged with a warning - this keeps the plugin non-fatal on
        partial or non-standard inputs.
        """
        session_dir = self._resolve_session_dir(input_dir)
        if session_dir is None:
            log.warning(
                "  HemaChip plugin: could not resolve a session folder from "
                f"{input_dir} - FOVs will not have grid/chip metadata"
            )
            return fovs

        try:
            patient = scan_patient(
                patient_dir=session_dir.parent,
                pixel_size_um=self.config.imaging.pixel_size_um,
            )
        except Exception as e:
            log.warning(f"  HemaChip plugin: scan failed ({e}) - FOVs unchanged")
            return fovs

        self._patient = patient
        session = next(
            (s for s in patient.sessions if s.session_dir == session_dir), None
        )
        if session is None:
            log.warning(
                f"  HemaChip plugin: session {session_dir.name} not found in scan result"
            )
            return fovs

        for chip in session.chips:
            for roi_num in chip.valid_rois:
                self._roi_to_chip[roi_num] = chip.chip_number

        n_matched = 0
        for fov in fovs:
            roi_num = _parse_roi_number(fov.fov_id)
            if roi_num is None:
                continue
            for chip in session.chips:
                if roi_num in chip.roi_grid:
                    row, col = chip.roi_grid[roi_num]
                    x_mm, y_mm = chip.roi_xy_mm[roi_num]
                    fov.grid_row = row
                    fov.grid_col = col
                    fov.x_mm = x_mm
                    fov.y_mm = y_mm
                    fov.condition = f"chip{chip.chip_number}"
                    n_matched += 1
                    break

        log.info(
            f"  HemaChip plugin: matched {n_matched}/{len(fovs)} FOVs to grid metadata"
        )
        return fovs

    # ── compute_fov_features ──────────────────────────────────────────────

    def compute_fov_features(self, spots, tracks, pop, fov):
        """
        Attach the 4 clinical features to the PopulationRecord as extra
        attributes, and stash them keyed by fov_id for the chip-level
        aggregation done in export().
        """
        features = compute_clinical_features(spots, pop, fov)
        pop.adhesion_rate = features["adhesion_rate"]
        pop.spreading_rate = features["spreading_rate"]
        pop.endpoint_value = features["endpoint_value"]
        pop.endpoint_variability = features["endpoint_variability"]

        self._roi_clinical[fov.fov_id] = features
        return pop

    # ── export ─────────────────────────────────────────────────────────────

    def export(self, spots, tracks, populations, config):
        """
        Write chip-level clinical feature aggregates
        (results_chip_clinical.csv) on top of core outputs. One row per
        chip, with mean/std/min/max/median of the 4 clinical features
        across that chip's ROIs.
        """
        if not self._roi_clinical:
            log.info("  HemaChip plugin: no clinical features computed - skipping export")
            return

        by_chip = {}
        for pop in populations:
            roi_num = _parse_roi_number(pop.fov_id)
            if roi_num is None or roi_num not in self._roi_to_chip:
                continue
            chip_num = self._roi_to_chip[roi_num]
            features = self._roi_clinical.get(pop.fov_id)
            if features is None:
                continue
            by_chip.setdefault(chip_num, []).append(features)

        if not by_chip:
            log.info(
                "  HemaChip plugin: no ROI-to-chip mapping available - "
                "skipping chip-level export (grid metadata may be missing)"
            )
            return

        rows = []
        for chip_num, roi_features_list in sorted(by_chip.items()):
            agg = aggregate_chip_level(roi_features_list)
            agg["chip_number"] = chip_num
            agg["n_rois"] = len(roi_features_list)
            rows.append(agg)

        df = pd.DataFrame(rows)
        cols = ["chip_number", "n_rois"] + [c for c in df.columns if c not in ("chip_number", "n_rois")]
        df = df[cols]

        out_path = config.output_dir / "results_chip_clinical.csv"
        # String concatenation, not Path.with_suffix() - see
        # job_manager.py's _save_state() for why with_suffix() is fragile.
        tmp = out_path.parent / (out_path.name + ".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(out_path)
        log.info(f"  results_chip_clinical.csv    {len(df):>8} rows")

    # ── internal ───────────────────────────────────────────────────────────

    def _resolve_session_dir(self, input_dir):
        """
        input_dir may itself be a session folder (contains ROI-N subfolders
        and/or an .epf directly), or a single ROI folder (single-FOV mode).
        Walk up to find the session-level folder in either case.
        """
        input_dir = Path(input_dir)
        if list(input_dir.glob("*.epf")) or list(input_dir.glob("ROI-*")):
            return input_dir
        if _ROI_ID_RE.match(input_dir.name):
            parent = input_dir.parent
            if list(parent.glob("*.epf")):
                return parent
        return None


def _parse_roi_number(fov_id):
    m = _ROI_ID_RE.match(fov_id)
    return int(m.group(1)) if m else None
