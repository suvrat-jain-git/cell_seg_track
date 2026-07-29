from __future__ import annotations
import logging
import re
import numpy as np
from config import FieldOfView, FrameInfo, FrameMode

log = logging.getLogger(__name__)

CLINICAL_TIMEPOINTS_MIN = [0, 5, 10, 15, 20, 30, 45, 60, 75, 90, 105, 120, 150, 180, 240, 300, 360, 480]

FRAME_INDEX_PATTERNS = [
    re.compile(r"_(\d{6})\.tif$", re.IGNORECASE),
    re.compile(r"_(\d{4})\.tif$", re.IGNORECASE),
    re.compile(r"_t(\d+)\.tif$", re.IGNORECASE),
    re.compile(r"(\d+)\.tif$", re.IGNORECASE),
]

def load_fov(fov_path, fov_id, config):
    if not fov_path.is_dir():
        return None
    all_frames = _discover(fov_path, config.imaging)
    if not all_frames:
        n_tifs = len([f for f in fov_path.iterdir() if f.is_file() and f.suffix.lower() in (".tif", ".tiff")])
        if n_tifs > 0:
            log.warning(
                f"  {fov_id}: {n_tifs} TIF file(s) found but none matched a known "
                f"frame-index naming pattern - this ROI will be SKIPPED. Check "
                f"filenames against FRAME_INDEX_PATTERNS in input_handler.py."
            )
        return None
    selected = _select(all_frames, config.frame_mode, config.imaging.time_interval_min or 5.0)
    for f in selected:
        f.fov_id = fov_id
    return FieldOfView(
        path=fov_path, fov_id=fov_id, frames=selected,
        pixel_size_um=config.imaging.pixel_size_um or 0.377,
        time_interval_min=config.imaging.time_interval_min or 5.0,
    )

def load_fovs_from_batch(batch_dir, config, fov_pattern="ROI-*"):
    fov_dirs = sorted(batch_dir.glob(fov_pattern))
    fovs = []
    for d in fov_dirs:
        if not d.is_dir():
            continue
        fov = load_fov(d, d.name, config)
        if fov and not fov.is_empty:
            fovs.append(fov)
    log.info(f"  Loaded {len(fovs)} FOVs from {batch_dir.name}")
    return fovs

def load_image(frame):
    try:
        import tifffile
        img = tifffile.imread(str(frame.path))
        if img.ndim == 3:
            img = img[0] if img.shape[0] <= 4 else img[:, :, 0]
        if img.dtype in [np.float32, np.float64]:
            vmax = img.max()
            img = (img / vmax * 65535).astype(np.uint16) if vmax > 0 else np.zeros_like(img, dtype=np.uint16)
        return img
    except Exception:
        return None

def validate_fov(fov, config):
    issues = []
    checked = 0
    for frame in fov.frames[:5]:
        img = load_image(frame)
        if img is None:
            issues.append(f"unreadable_{frame.frame_index}")
            continue
        checked += 1
        h, w = img.shape[:2]
        if config.imaging.expected_width_px and w != config.imaging.expected_width_px:
            issues.append(f"wrong_w_{w}")
        if config.imaging.expected_height_px and h != config.imaging.expected_height_px:
            issues.append(f"wrong_h_{h}")
        if img.mean() < 2.0:
            issues.append(f"black_{frame.frame_index}")
    return {"is_valid": len(issues) == 0, "issues": list(set(issues)), "sample_size": checked}

def _discover(fov_path, imaging):
    tifs = [f for f in fov_path.iterdir() if f.is_file() and f.suffix.lower() in (".tif", ".tiff")]
    interval = imaging.time_interval_min or 5.0
    frames = []
    for t in tifs:
        idx = _parse(t.name)
        if idx is None:
            continue
        frames.append(FrameInfo(path=t, frame_index=idx, elapsed_min=round(idx * interval, 3)))
    frames.sort(key=lambda f: f.frame_index)
    return frames

def _parse(filename):
    for p in FRAME_INDEX_PATTERNS:
        m = p.search(filename)
        if m:
            return int(m.group(1))
    return None

def _select(frames, mode, interval_min):
    if not frames:
        return []
    if mode == FrameMode.FIRST:
        return [frames[0]]
    if mode == FrameMode.MEDIAN:
        return [frames[len(frames) // 2]]
    if mode == FrameMode.CLINICAL:
        return _clinical(frames, interval_min)
    return frames

def _clinical(frames, interval_min):
    if not frames or interval_min <= 0:
        return frames
    max_e = frames[-1].frame_index * interval_min
    selected = {}
    for t in CLINICAL_TIMEPOINTS_MIN:
        if t > max_e + interval_min:
            continue
        best = min(frames, key=lambda f: abs(f.frame_index - t / interval_min))
        selected[best.frame_index] = best
    return sorted(selected.values(), key=lambda f: f.frame_index)
