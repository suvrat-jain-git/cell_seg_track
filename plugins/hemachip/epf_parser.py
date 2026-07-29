"""
plugins/hemachip/epf_parser.py - parses HemaChip .epf protocol files.

.epf is Lumaview 720/600-Series' XML protocol export. It describes the
imaging protocol (capture interval, duration) and the full list of stage
positions visited (roiPositions), NOT actual pixel size or image dimensions
-- those come from the imaging config the user supplies (--pixel_size_um)
or from image metadata directly.

Verified against a real CB007 .epf file. Key structural facts confirmed
by direct inspection, not assumed:

  - <roiPositions> contains one <XYZPosition> per stage position, each with
    x/y/z in millimeters, a 1-based <order>, and an alphabetic <id> (a, b, c...
    continuing beyond z into aa, ab, ...).
  - Positions are laid out in a BOUSTROPHEDON (snake) scan: row 0 goes
    left-to-right, row 1 goes right-to-left, alternating. Reconstructing
    grid_col naively from x-sort-order without accounting for this produces
    a silently wrong grid.
  - Multiple chips on one slide show up as one flat list of positions with
    a large discontinuous jump in y between the last row of one chip and the
    first row of the next (observed: ~14mm gap vs ~0.6mm row spacing).
    Chip boundaries are detected from this gap, not assumed to be exactly
    at position 225.
  - <savedImages> is a short recent-file log, not a full manifest of ROIs -
    it is NOT used for ROI discovery. Actual ROI folders on disk are the
    source of truth (see scanner.py).
  - Pixel size is NOT present in the .epf file. It must come from the
    session's imaging config (ImagingConfig.pixel_size_um) or be supplied
    directly.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_POSITION_RE = re.compile(
    r"<XYZPosition>\s*"
    r"<x>([\d.\-]+)</x>\s*"
    r"<y>([\d.\-]+)</y>\s*"
    r"<z>([\d.\-]+)</z>\s*"
    r"<order>(\d+)</order>\s*"
    r"<id>(\w+)</id>\s*"
    r"</XYZPosition>",
    re.MULTILINE,
)

_SIMPLE_FIELD_RE = {
    "protocolName": re.compile(r"<protocolName>(.*?)</protocolName>"),
    "captureEveryHours": re.compile(r"<captureEveryHours>(\d+)</captureEveryHours>"),
    "captureEveryMinutes": re.compile(r"<captureEveryMinutes>(\d+)</captureEveryMinutes>"),
    "captureEverySeconds": re.compile(r"<captureEverySeconds>(\d+)</captureEverySeconds>"),
    "totalPeriodHours": re.compile(r"<totalPeriodHours>(\d+)</totalPeriodHours>"),
    "totalPeriodMinutes": re.compile(r"<totalPeriodMinutes>(\d+)</totalPeriodMinutes>"),
    "totalPeriodSeconds": re.compile(r"<totalPeriodSeconds>(\d+)</totalPeriodSeconds>"),
}

# Row spacing in mm below which two consecutive rows are considered part of
# the SAME chip. A gap larger than this marks a new chip. Determined from
# real data: within-chip row spacing was ~0.6mm, between-chip gap was ~14mm.
# 3mm is a safe midpoint threshold with wide margin either side.
_CHIP_BOUNDARY_GAP_MM = 3.0


class EPFPosition:
    """One stage position from the .epf roiPositions list."""
    __slots__ = ("x_mm", "y_mm", "z_mm", "order", "id")

    def __init__(self, x_mm, y_mm, z_mm, order, id_):
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.z_mm = z_mm
        self.order = order
        self.id = id_


class EPFMetadata:
    """Parsed contents of one .epf file."""

    def __init__(self):
        self.protocol_name = ""
        self.capture_interval_min = 5.0
        self.total_period_min = 0.0
        self.positions = []          # list[EPFPosition], in scan order
        self.chips = []              # list[list[EPFPosition]] - grouped by chip
        self.source_path = None

    def n_chips(self):
        return len(self.chips)

    def n_positions(self):
        return len(self.positions)


def parse_epf(epf_path) -> EPFMetadata:
    """
    Parse a HemaChip .epf file.

    Args:
        epf_path: path to the .epf XML file

    Returns:
        EPFMetadata with capture timing and chip/position layout.

    Raises:
        FileNotFoundError if the file doesn't exist.
        ValueError if the file has no roiPositions (not a valid protocol export).
    """
    epf_path = Path(epf_path)
    if not epf_path.exists():
        raise FileNotFoundError(f"EPF file not found: {epf_path}")

    content = epf_path.read_text(encoding="utf-8", errors="replace")

    meta = EPFMetadata()
    meta.source_path = epf_path

    m = _SIMPLE_FIELD_RE["protocolName"].search(content)
    meta.protocol_name = m.group(1) if m else epf_path.stem

    hrs = _extract_int(content, "captureEveryHours")
    mins = _extract_int(content, "captureEveryMinutes")
    secs = _extract_int(content, "captureEverySeconds")
    meta.capture_interval_min = hrs * 60 + mins + secs / 60.0
    if meta.capture_interval_min <= 0:
        log.warning("  EPF capture interval computed as 0 - defaulting to 5.0 min")
        meta.capture_interval_min = 5.0

    p_hrs = _extract_int(content, "totalPeriodHours")
    p_mins = _extract_int(content, "totalPeriodMinutes")
    p_secs = _extract_int(content, "totalPeriodSeconds")
    meta.total_period_min = p_hrs * 60 + p_mins + p_secs / 60.0
    if meta.total_period_min <= 0:
        log.warning(
            "  EPF total period computed as 0 - the totalPeriodHours/Minutes/"
            "Seconds tags may be missing or unexpectedly formatted in this file"
        )

    meta.positions = _parse_positions(content)
    if not meta.positions:
        raise ValueError(
            f"No <roiPositions> found in {epf_path.name} - "
            f"not a valid HemaChip protocol export"
        )

    meta.chips = _split_into_chips(meta.positions)

    log.info(
        f"  EPF parsed: {meta.protocol_name} | "
        f"{meta.capture_interval_min:.1f} min interval | "
        f"{len(meta.positions)} positions across {len(meta.chips)} chip(s)"
    )
    return meta


def _extract_int(content, tag):
    # \s* around the digits tolerates XML formatted with internal
    # whitespace/newlines (e.g. "<tag>\n  15\n</tag>"), which the original
    # tight pattern "<tag>(\d+)</tag>" silently failed to match, returning
    # 0 instead of the real value with no warning. The one real .epf file
    # available for testing happens to have no internal whitespace, so
    # this specific gap was not observed in practice, but nothing
    # guarantees every future .epf export is formatted the same way.
    m = re.search(rf"<{tag}>\s*(\d+)\s*</{tag}>", content)
    return int(m.group(1)) if m else 0


def _parse_positions(content):
    positions = []
    for x, y, z, order, id_ in _POSITION_RE.findall(content):
        positions.append(EPFPosition(
            x_mm=float(x), y_mm=float(y), z_mm=float(z),
            order=int(order), id_=id_,
        ))
    positions.sort(key=lambda p: p.order)
    return positions


def _split_into_chips(positions):
    """
    Split a flat position list into per-chip groups by detecting large
    jumps in y between consecutive positions. A jump larger than
    _CHIP_BOUNDARY_GAP_MM marks the start of a new chip.
    """
    if not positions:
        return []

    chips = [[positions[0]]]
    for prev, curr in zip(positions, positions[1:]):
        if abs(curr.y_mm - prev.y_mm) > _CHIP_BOUNDARY_GAP_MM:
            chips.append([])
        chips[-1].append(curr)
    return chips


def compute_grid_coordinates(chip_positions):
    """
    Reconstruct (row, col) grid coordinates for one chip's positions,
    correctly accounting for the boustrophedon (snake) scan pattern.

    Row 0 scans left-to-right (ascending x). Row 1 scans right-to-left
    (descending x), and so on, alternating. Naively sorting each row's
    x-values ascending would silently misassign columns on every odd row.

    Args:
        chip_positions: list[EPFPosition] belonging to a single chip,
                        in scan order (i.e. as read from the .epf, not resorted)

    Returns:
        dict[EPFPosition.order -> (row, col)]
    """
    if not chip_positions:
        return {}

    # Group into rows by unique y value (rounded to avoid float noise)
    y_values = []
    seen_y = set()
    for p in chip_positions:
        ry = round(p.y_mm, 2)
        if ry not in seen_y:
            seen_y.add(ry)
            y_values.append(ry)
    # y_values are in scan order already (first row visited = index 0)
    row_of_y = {y: i for i, y in enumerate(y_values)}

    coords = {}
    rows = {}
    for p in chip_positions:
        row = row_of_y[round(p.y_mm, 2)]
        rows.setdefault(row, []).append(p)

    for row, pts in rows.items():
        # Preserve scan order within the row (already left-to-right or
        # right-to-left as visited) rather than re-sorting by x, which
        # would break the snake pattern's physical left-right meaning.
        # Column index is assigned by physical x position, ascending,
        # regardless of scan direction, so grid_col is spatially consistent
        # across rows.
        pts_by_x = sorted(pts, key=lambda p: p.x_mm)
        for col, p in enumerate(pts_by_x):
            coords[p.order] = (row, col)

    return coords
