"""job_manager.py - progress tracking, checkpointing, resume capability for long Colab runs."""
from __future__ import annotations
import json
import logging
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self, config):
        self.config = config
        self.run_id = config.run_id or str(uuid.uuid4())[:8]
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.state_dir = self.ckpt_dir / "fov_states"
        self.state_file = self.ckpt_dir / f"run_{self.run_id}.json"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.completed_fovs = set()
        self.failed_fovs = set()
        self.timing = {}
        self.start_time = time.perf_counter()
        self._load_state()

    def is_completed(self, fov_id):
        return fov_id in self.completed_fovs

    def mark_complete(self, fov_id, spots, tracks, pop):
        self._save_fov_state(fov_id, spots, tracks, pop)
        self.completed_fovs.add(fov_id)
        self._save_state()
        log.debug(f"  Checkpoint saved: {fov_id}")

    def mark_failed(self, fov_id, error):
        self.failed_fovs.add(fov_id)
        log.warning(f"  FOV failed: {fov_id} - {error}")
        self._save_state()

    def load_fov_results(self, fov_id):
        spots = _load_pickle(self.state_dir / f"{fov_id}_spots.pkl") or []
        tracks = _load_pickle(self.state_dir / f"{fov_id}_tracks.pkl") or []
        pop = _load_pickle(self.state_dir / f"{fov_id}_pop.pkl")
        return spots, tracks, pop

    def record_timing(self, step, elapsed):
        self.timing[step] = round(elapsed, 2)
        self._save_state()

    def summary(self):
        total = time.perf_counter() - self.start_time
        return {
            "run_id": self.run_id,
            "completed_fovs": len(self.completed_fovs),
            "failed_fovs": len(self.failed_fovs),
            "total_time_secs": round(total, 2),
            "timing": self.timing,
        }

    def _save_state(self):
        state = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "completed_fovs": list(self.completed_fovs),
            "failed_fovs": list(self.failed_fovs),
            "timing": self.timing,
        }
        # NOTE: use string concatenation, not Path.with_suffix(), for the
        # temp filename. with_suffix() REPLACES the existing suffix rather
        # than appending - it happens to produce the intended result for
        # filenames with exactly one "." (e.g. "run_abc123.json" ->
        # "run_abc123.tmp.json"), but would silently corrupt the filename
        # if run_id or fov_id ever contained a literal "." (e.g. a ROI
        # folder named "ROI-1.backup" would produce the wrong temp path
        # and could cause a checkpoint write to silently target the wrong
        # file). run_id is an md5 hex digest so this is not currently
        # reachable, but fov_id is taken directly from the ROI folder name
        # on disk with no sanitisation - safer to not rely on that
        # assumption holding forever.
        tmp = self.state_file.parent / (self.state_file.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp.replace(self.state_file)

    def _load_state(self):
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)
            self.completed_fovs = set(state.get("completed_fovs", []))
            self.failed_fovs = set(state.get("failed_fovs", []))
            self.timing = state.get("timing", {})
            n = len(self.completed_fovs)
            if n > 0:
                log.info(f"  Resuming run {self.run_id}: {n} FOVs already complete")
        except Exception as e:
            log.warning(f"  Could not load checkpoint: {e}")

    def _save_fov_state(self, fov_id, spots, tracks, pop):
        _save_pickle(spots, self.state_dir / f"{fov_id}_spots.pkl")
        _save_pickle(tracks, self.state_dir / f"{fov_id}_tracks.pkl")
        _save_pickle(pop, self.state_dir / f"{fov_id}_pop.pkl")


def _save_pickle(obj, path):
    # See _save_state() above for why string concatenation is used here
    # instead of Path.with_suffix().
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def _load_pickle(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
