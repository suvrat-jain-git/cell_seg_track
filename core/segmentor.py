from __future__ import annotations
import logging
import sys
import time
from pathlib import Path
import numpy as np
from core.input_handler import load_image
from core.spot_features import extract_spot_features

log = logging.getLogger(__name__)

def segment_fov(fov, config, model=None):
    if model is None:
        model = load_model(config.segmentation)
    results = []
    total = len(fov.frames)
    t_start = time.perf_counter()
    for i, frame in enumerate(fov.frames):
        spots = _segment_frame(frame, fov, model, config)
        results.append(spots)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - t_start
            per_frame = elapsed / (i + 1)
            remaining = per_frame * (total - i - 1)
            log.info(
                f"    [{fov.fov_id}] frame {frame.frame_index:04d} "
                f"| {i+1}/{total} | {elapsed:.0f}s elapsed | "
                f"~{remaining:.0f}s remaining | {len(spots)} cells"
            )
    return results


def segment_fovs_parallel(fovs, config):
    force_single = config.segmentation.use_gpu or sys.platform == "win32"
    if force_single or config.n_workers <= 1:
        if sys.platform == "win32" and config.n_workers > 1:
            log.info("  Windows detected - using single process mode")
        model = load_model(config.segmentation)
        results = {}
        for fov in fovs:
            results[fov.fov_id] = segment_fov(fov, config, model)
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed
    log.info(f"  Parallel mode: {config.n_workers} workers")
    chunks = _split(fovs, config.n_workers)
    results = {}
    with ProcessPoolExecutor(max_workers=config.n_workers) as ex:
        futures = {
            ex.submit(_worker_chunk, chunk, config.segmentation, config.output, str(config.output_dir)): i
            for i, chunk in enumerate(chunks)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results.update(fut.result())
                log.info(f"  Worker {idx + 1} complete")
            except Exception as e:
                log.error(f"  Worker {idx + 1} failed: {e}")
    return results


def load_model(seg_config):
    try:
        import cellpose
        from cellpose import models
        version = getattr(cellpose, "__version__", "unknown")
        log.info(f"  Cellpose {version} | model={seg_config.model_name} | GPU={seg_config.use_gpu}")
        if hasattr(models, "CellposeModel"):
            model = models.CellposeModel(gpu=seg_config.use_gpu, pretrained_model=seg_config.model_name)
            model._cellflow_api = 3
        elif hasattr(models, "Cellpose"):
            model = models.Cellpose(model_type=seg_config.model_name, gpu=seg_config.use_gpu)
            model._cellflow_api = 2
        else:
            raise RuntimeError("No valid Cellpose API found")
        log.info("  Model loaded successfully")
        return model
    except ImportError:
        raise ImportError("Cellpose not installed. Run: pip install cellpose==3.0.11")


def _segment_frame(frame, fov, model, config):
    try:
        img = load_image(frame)
        if img is None:
            return [_err(frame, fov, "Failed to load image")]
        diameter = config.segmentation.diameter
        if diameter is None:
            # Fallback priority:
            # 1. HSEC-validated diameter from data provider (cyto3, 93.86% accuracy
            #    against 4,647 manually-scored cells) - use as-is when pixel size
            #    matches their imaging setup (0.3769 um/px, 20x objective)
            # 2. Generic 10um-cell estimate for other pixel sizes / cell types
            if abs(fov.pixel_size_um - 0.3769) < 0.01:
                diameter = 31.9
            else:
                diameter = max(5.0, min(60.0, 10.0 / fov.pixel_size_um))
        masks = _run_cellpose(model, img, diameter, config.segmentation)
        if masks is None:
            return [_err(frame, fov, "Cellpose returned None")]
        if config.output.save_masks:
            _save_mask(masks, frame, fov, config.output_dir)
        return extract_spot_features(masks, img, frame, fov, config.segmentation)
    except Exception as e:
        log.warning(f"    [{fov.fov_id}] frame {frame.frame_index:04d} segmentation failed: {e}")
        return [_err(frame, fov, str(e))]


def _run_cellpose(model, img, diameter, seg_cfg):
    try:
        api = getattr(model, "_cellflow_api", 3)
        kwargs = dict(diameter=diameter, channels=[0, 0], do_3D=False,
                     flow_threshold=seg_cfg.flow_threshold, cellprob_threshold=seg_cfg.cellprob_threshold)
        if api >= 3:
            masks, _, _ = model.eval(img, **kwargs)
        else:
            masks, _, _, _ = model.eval(img, **kwargs)
        return masks
    except Exception as e:
        log.warning(f"    Cellpose eval error: {e}")
        return None


def _save_mask(masks, frame, fov, output_dir):
    try:
        import tifffile
        mask_dir = output_dir / "masks" / fov.fov_id
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{fov.fov_id}_frame{frame.frame_index:06d}_mask.tif"
        tifffile.imwrite(str(mask_path), masks.astype(np.uint16))
    except Exception as e:
        log.warning(f"    Could not save mask for frame {frame.frame_index}: {e}")


def _err(frame, fov, msg):
    from config import SpotRecord
    return SpotRecord(fov_id=fov.fov_id, frame_index=frame.frame_index, elapsed_min=frame.elapsed_min,
                       spot_id=-1, segmentation_ok=False, error_msg=msg)


def _worker_chunk(fovs, seg_config, out_config, output_dir):
    # NOTE: this minimal _Cfg only carries what segment_fov() currently touches
    # (config.segmentation, config.output, config.output_dir). If segment_fov()
    # or anything it calls is extended to read other PipelineConfig fields
    # (imaging, tracking, frame_mode, n_workers), this worker-side config
    # object must be extended too, or it will raise AttributeError only
    # inside worker processes - a failure mode that's easy to miss since
    # parallel mode is not currently exercised on Windows (forced single-
    # process) and wasn't hit by any test so far.
    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING)

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.segmentation = seg_config
    cfg.output = out_config
    cfg.output_dir = Path(output_dir)
    model = load_model(seg_config)
    return {fov.fov_id: segment_fov(fov, cfg, model) for fov in fovs}


def _split(items, n):
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
