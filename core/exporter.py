"""core/exporter.py - writes all pipeline outputs to disk, atomic writes."""
from __future__ import annotations
import json
import logging
from datetime import datetime
import pandas as pd

log = logging.getLogger(__name__)


def export_all(spots, tracks, populations, config, timing):
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_locks(output_dir, ["spots.csv", "tracks.csv", "population.csv"])

    written = {}

    if spots:
        df = pd.DataFrame([s.to_dict() for s in spots])
        n = _write_csv(df, output_dir / "spots.csv")
        log.info(f"  spots.csv            {n:>8} rows")
        written["spots.csv"] = n

    if tracks:
        df = pd.DataFrame([t.to_dict() for t in tracks])
        n = _write_csv(df, output_dir / "tracks.csv")
        log.info(f"  tracks.csv           {n:>8} rows")
        written["tracks.csv"] = n

    if populations:
        df = pd.DataFrame([p.to_dict() for p in populations])
        n = _write_csv(df, output_dir / "population.csv")
        log.info(f"  population.csv       {n:>8} rows")
        written["population.csv"] = n

    config.to_json(output_dir / "run_config.json")
    log.info("  run_config.json")

    _write_summary(spots, tracks, populations, timing, written, output_dir)
    log.info("  run_summary.json")

    return written


def export_fov_tracks(tracks, spots, fov_id, output_dir):
    fov_dir = output_dir / "per_fov" / fov_id
    fov_dir.mkdir(parents=True, exist_ok=True)
    if spots:
        df = pd.DataFrame([s.to_dict() for s in spots if s.fov_id == fov_id])
        _write_csv(df, fov_dir / "spots.csv")
    if tracks:
        df = pd.DataFrame([t.to_dict() for t in tracks if t.fov_id == fov_id])
        _write_csv(df, fov_dir / "tracks.csv")


def _write_csv(df, path):
    if df.empty:
        return 0
    # String concatenation, not Path.with_suffix() - see job_manager.py's
    # _save_state() for why with_suffix() is fragile here.
    tmp = path.parent / (path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    return len(df)


def _write_summary(spots, tracks, populations, timing, written, output_dir):
    total_time = sum(timing.values())
    n_ok = sum(1 for s in spots if s.segmentation_ok)
    n_fail = sum(1 for s in spots if not s.segmentation_ok)
    errors = [
        {"fov_id": s.fov_id, "frame": s.frame_index, "error": s.error_msg}
        for s in spots if not s.segmentation_ok
    ][:50]
    timing_pct = {
        step: {"seconds": round(secs, 2), "pct": round(secs / total_time * 100, 1) if total_time > 0 else 0}
        for step, secs in timing.items()
    }
    summary = {
        "pipeline_version": "1.0.0",
        "run_timestamp": datetime.now().isoformat(),
        "total_time_secs": round(total_time, 2),
        "timing": timing_pct,
        "spots": {"total": len(spots), "ok": n_ok, "failed": n_fail},
        "tracks": {"total": len(tracks)},
        "fovs": {"total": len(populations)},
        "output_files": written,
        "errors": errors,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _check_locks(output_dir, filenames):
    locked = []
    for name in filenames:
        path = output_dir / name
        if not path.exists():
            continue
        try:
            with open(path, "a"):
                pass
        except PermissionError:
            locked.append(name)
    if locked:
        raise PermissionError(
            "Cannot write output files - open in another program (e.g. Excel). "
            "Close these files and rerun:\n  " + "\n  ".join(locked)
        )
