"""
plugins/hemachip/scanner.py - discovers HemaChip patient folder structure on disk
and merges it with EPF stage-position metadata.

Directory convention observed from real data:

  <patient_dir>/
    <session_timestamp>/            e.g. 20231205_134131
      HemaChip_..._<date>.epf       one EPF file per session
      ROI-1/
        MyExperiment_ROI-1_WHITE_000000.tif
        MyExperiment_ROI-1_WHITE_000001.tif
        ...
      ROI-2/
      ...

Key facts confirmed against real data, not assumed:
  - ROI folders on disk are the source of truth for which ROIs actually
    have data. The .epf's roiPositions list describes the FULL planned
    grid (up to 450 positions = 2 chips x 225), but only some ROI folders
    may exist on disk (partial downloads). We scan disk first, then look up
    each existing ROI's spatial metadata from the EPF.
  - ROI numbering in folder names (ROI-1, ROI-2, ...) corresponds directly
    to the EPF position `order` field, 1-indexed matching folder number.
  - Chip assignment for a given ROI number comes from the EPF chip split
    (see epf_parser._split_into_chips), not from any fixed range assumption -
    chip boundaries are detected from real position data per EPF file, since
    a different device or protocol could in principle produce a different
    grid size.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path

from plugins.hemachip.epf_parser import parse_epf, compute_grid_coordinates

log = logging.getLogger(__name__)

_SESSION_DIR_RE = re.compile(r"^\d{8}_\d{6}$")  # e.g. 20231205_134131
_ROI_DIR_RE = re.compile(r"^ROI-(\d+)$", re.IGNORECASE)


class ChipInfo:
    """One physical chip within a session, with its ROI positions."""

    def __init__(self, chip_number, condition=""):
        self.chip_number = chip_number
        self.condition = condition   # e.g. "gradient" / "homogeneous" - set by caller
        self.roi_paths = {}          # roi_number -> Path
        self.roi_grid = {}           # roi_number -> (row, col)
        self.roi_xy_mm = {}          # roi_number -> (x_mm, y_mm)

    @property
    def valid_rois(self):
        """ROI numbers that exist on disk AND have EPF grid metadata."""
        return sorted(set(self.roi_paths) & set(self.roi_grid))

    def __repr__(self):
        return f"ChipInfo(chip={self.chip_number}, rois_on_disk={len(self.roi_paths)})"


class SessionInfo:
    """One imaging session (one .epf run, potentially multiple chips)."""

    def __init__(self, session_id, session_dir, epf_meta):
        self.session_id = session_id
        self.session_dir = session_dir
        self.epf_meta = epf_meta
        self.capture_interval_minutes = epf_meta.capture_interval_min if epf_meta else 5.0
        self.pixel_size_um = None    # not present in EPF - set by caller/config
        self.chips = []              # list[ChipInfo]

    def __repr__(self):
        return f"SessionInfo({self.session_id}, {len(self.chips)} chips)"


class PatientInfo:
    """Top-level patient record - one or more sessions."""

    def __init__(self, patient_id, patient_dir):
        self.patient_id = patient_id
        self.patient_dir = Path(patient_dir)
        self.sessions = []           # list[SessionInfo]

    def total_rois(self):
        return sum(len(chip.valid_rois) for s in self.sessions for chip in s.chips)

    def __repr__(self):
        return f"PatientInfo({self.patient_id}, {len(self.sessions)} sessions, {self.total_rois()} ROIs on disk)"


def scan_patient(patient_dir, patient_id=None, pixel_size_um=None) -> PatientInfo:
    """
    Scan a patient's top-level folder for session subfolders, and within
    each session, discover which ROI folders exist on disk and merge them
    with EPF grid/chip metadata.

    Args:
        patient_dir    : path to the patient's top-level folder
        patient_id     : optional explicit patient ID (default: folder name)
        pixel_size_um  : physical pixel size, not available in EPF, must be
                         supplied here or set later per-session

    Returns:
        PatientInfo with sessions -> chips -> ROI paths and grid positions populated.
    """
    patient_dir = Path(patient_dir)
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    pid = patient_id or patient_dir.name
    patient = PatientInfo(patient_id=pid, patient_dir=patient_dir)

    session_dirs = sorted(
        d for d in patient_dir.iterdir()
        if d.is_dir() and _SESSION_DIR_RE.match(d.name)
    )
    if not session_dirs:
        log.warning(
            f"  No session folders (YYYYMMDD_HHMMSS format) found in {patient_dir}"
        )
        return patient

    for session_dir in session_dirs:
        session = _scan_session(session_dir, pixel_size_um)
        if session:
            patient.sessions.append(session)

    log.info(
        f"  Patient {pid}: {len(patient.sessions)} session(s), "
        f"{patient.total_rois()} ROI(s) on disk total"
    )
    return patient


def _scan_session(session_dir, pixel_size_um) -> SessionInfo:
    log.info(f"  Scanning session: {session_dir.name}")

    # Path.glob() does not guarantee any particular ordering - it reflects
    # filesystem enumeration order, which can vary between machines/OSes.
    # When multiple .epf files exist for one session, direct investigation
    # of real CB007 data found that the genuine, complete session log is
    # reliably the LARGEST file (one real session log had 43,239
    # <string> savedImages entries and was ~7.7MB, while a same-named
    # draft/template file for the same nominal date had only 7-8 entries
    # and was ~96KB - a >80x size difference). Sorting by file size
    # descending, with alphabetical order as a deterministic tiebreak,
    # is both reproducible across machines and matches the one real
    # signal we've confirmed distinguishes a genuine session log from a
    # draft/template file of the same name pattern.
    epf_files = sorted(
        session_dir.glob("*.epf"),
        key=lambda f: (-f.stat().st_size, f.name),
    )
    epf_meta = None
    if epf_files:
        if len(epf_files) > 1:
            log.warning(
                f"    Multiple .epf files in {session_dir.name} - using the "
                f"largest ({epf_files[0].name}, {epf_files[0].stat().st_size:,} bytes), "
                f"since a genuine complete session log is reliably much larger "
                f"than a draft/template file. Other files found: "
                f"{[f.name for f in epf_files[1:]]}"
            )
        try:
            epf_meta = parse_epf(epf_files[0])
        except Exception as e:
            log.warning(f"    Failed to parse {epf_files[0].name}: {e}")
    else:
        log.warning(
            f"    No .epf file in {session_dir.name} - "
            f"chip/grid metadata will be unavailable for this session"
        )

    session = SessionInfo(
        session_id=session_dir.name, session_dir=session_dir, epf_meta=epf_meta,
    )
    session.pixel_size_um = pixel_size_um

    # Discover ROI folders actually present on disk
    roi_dirs = {}
    for d in session_dir.iterdir():
        if not d.is_dir():
            continue
        m = _ROI_DIR_RE.match(d.name)
        if m:
            roi_dirs[int(m.group(1))] = d

    if not roi_dirs:
        log.warning(f"    No ROI-N folders found in {session_dir.name}")
        return session

    if epf_meta is None:
        # No EPF metadata available - build a single chip with disk paths only,
        # no grid positions. Downstream code should handle roi_grid being empty.
        chip = ChipInfo(chip_number=1)
        chip.roi_paths = roi_dirs
        session.chips = [chip]
        log.info(
            f"    {len(roi_dirs)} ROI folder(s) found, no EPF - "
            f"grid positions unavailable"
        )
        return session

    session.chips = _build_chips_from_epf(epf_meta, roi_dirs)
    for chip in session.chips:
        log.info(
            f"    Chip {chip.chip_number}: "
            f"{len(chip.valid_rois)}/{len(chip.roi_paths)} ROI(s) with grid data "
            f"({len(roi_dirs)} total ROI folders on disk for this session)"
        )
    return session


def _build_chips_from_epf(epf_meta, roi_dirs) -> list:
    """
    Assign each disk ROI folder to its chip and grid position using the
    EPF's chip split and position order. ROI-N on disk maps to EPF position
    with order == N.
    """
    chips = []

    for chip_idx, chip_positions in enumerate(epf_meta.chips, start=1):
        chip = ChipInfo(chip_number=chip_idx)
        grid_coords = compute_grid_coordinates(chip_positions)

        for pos in chip_positions:
            roi_number = pos.order  # ROI-N folder <-> EPF order N, 1-indexed globally
            if roi_number in roi_dirs:
                chip.roi_paths[roi_number] = roi_dirs[roi_number]
                chip.roi_grid[roi_number] = grid_coords[pos.order]
                chip.roi_xy_mm[roi_number] = (pos.x_mm, pos.y_mm)

        chips.append(chip)

    return chips
