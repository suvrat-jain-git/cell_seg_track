"""core/spot_features.py - per-cell morphological features from Cellpose masks."""
from __future__ import annotations
import logging
import numpy as np
from config import SpotRecord

log = logging.getLogger(__name__)


def extract_spot_features(masks, image, frame, fov, seg_config):
    labels = np.unique(masks)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return []
    px = fov.pixel_size_um
    min_a = _d2a(seg_config.min_diameter_um, px)
    max_a = _d2a(seg_config.max_diameter_um, px)

    # Performance: crop each cell to its bounding box ONCE via scipy's
    # find_objects (single pass over the full mask array, O(pixels) total),
    # then run all subsequent per-cell geometry (perimeter, eccentricity,
    # solidity, centroid) on that small crop instead of the full-frame
    # array. Previously each of those ran independently on a full-frame
    # boolean array per cell (e.g. 1600x1600) regardless of actual cell
    # size (~15-20px) - roughly 7x slower at realistic HSEC density
    # (measured: ~64s/ROI -> ~9s/ROI for perimeter alone at 140 cells x
    # 97 frames), which becomes hours of wasted compute at 225-ROI scale.
    from scipy import ndimage
    bboxes = ndimage.find_objects(masks)

    spots = []
    for label in labels:
        sl = bboxes[int(label) - 1]
        if sl is None:
            continue
        crop = (masks[sl] == label)
        area = float(np.sum(crop))
        if area < min_a or area > max_a:
            continue

        # Offsets to convert crop-local coordinates back to full-frame
        # pixel coordinates for centroid/tracking purposes.
        row_offset = sl[0].start
        col_offset = sl[1].start

        cy_local, cx_local = _centroid(crop)
        cy, cx = cy_local + row_offset, cx_local + col_offset

        perim = _perim(crop)
        circ = _circ(area, perim)
        ecc = _ecc(crop)
        sol = _sol(crop, area)

        if image is not None:
            inten = float(np.mean(image[sl][crop]))
        else:
            inten = 0.0

        spots.append(SpotRecord(
            fov_id=fov.fov_id, frame_index=frame.frame_index,
            elapsed_min=frame.elapsed_min, spot_id=int(label),
            centroid_x_px=cx, centroid_y_px=cy,
            centroid_x_um=cx * px, centroid_y_um=cy * px,
            area_px=area, area_um2=round(area * (px ** 2), 3),
            perimeter_px=round(perim, 2), circularity=round(circ, 4),
            eccentricity=round(ecc, 4), solidity=round(sol, 4),
            mean_intensity=round(inten, 2), segmentation_ok=True,
        ))
    return spots


def _centroid(m):
    c = np.argwhere(m)
    return (0.0, 0.0) if len(c) == 0 else (float(c[:, 0].mean()), float(c[:, 1].mean()))


def _perim(m):
    p = np.pad(m, 1, constant_values=False)
    c = p[1:-1, 1:-1]
    t = p[0:-2, 1:-1]
    b = p[2:, 1:-1]
    l = p[1:-1, 0:-2]
    r = p[1:-1, 2:]
    return float(np.sum(c & (~t | ~b | ~l | ~r)))


def _circ(area, perim):
    return 0.0 if perim <= 0 else float(min(1.0, (4 * np.pi * area) / (perim ** 2)))


def _ecc(m):
    try:
        c = np.argwhere(m).astype(float)
        if len(c) < 5:
            return 0.0
        cy, cx = c.mean(axis=0)
        dy = c[:, 0] - cy
        dx = c[:, 1] - cx
        m20 = float(np.mean(dx ** 2))
        m02 = float(np.mean(dy ** 2))
        m11 = float(np.mean(dx * dy))
        tmp = np.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2)
        l1 = (m20 + m02 + tmp) / 2
        l2 = max((m20 + m02 - tmp) / 2, 0)
        return 0.0 if l1 <= 0 else float(np.sqrt(1 - l2 / l1))
    except Exception:
        return 0.0


def _sol(m, area):
    try:
        c = np.argwhere(m)
        if len(c) == 0:
            return 0.0
        rm, cm = c.min(axis=0)
        rx, cx2 = c.max(axis=0)
        bbox = float((rx - rm + 1) * (cx2 - cm + 1))
        return float(min(1.0, area / bbox)) if bbox > 0 else 0.0
    except Exception:
        return 0.0


def _d2a(d_um, px):
    return np.pi * ((d_um / 2 / px) ** 2)
