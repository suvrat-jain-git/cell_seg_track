"""core/track_features.py - population statistics from all tracks in one FOV."""
from __future__ import annotations
import logging
from collections import Counter
import numpy as np
from config import PopulationRecord

log = logging.getLogger(__name__)


def compute_population(tracks, spots, fov):
    pop = PopulationRecord(
        fov_id=fov.fov_id, grid_row=fov.grid_row, grid_col=fov.grid_col,
        x_mm=fov.x_mm, y_mm=fov.y_mm, condition=fov.condition,
        n_frames_total=fov.n_frames,
        n_frames_processed=len(set(s.frame_index for s in spots if s.segmentation_ok)),
    )
    _dynamics(pop, spots, fov.time_interval_min)
    if not tracks:
        return pop
    ls = [t.lifespan_min for t in tracks]
    vs = [t.mean_velocity_um_min for t in tracks]
    ds = [t.total_displacement_um for t in tracks]
    cs = [t.confinement_ratio for t in tracks]
    ar = [t.mean_area_um2 for t in tracks]
    ci = [t.mean_circularity for t in tracks]
    ac = [t.area_change_um2 for t in tracks]
    ns = sum(1 for t in tracks if t.survived_to_end)
    pop.n_tracks_total = len(tracks)
    pop.n_tracks_survived = ns
    pop.pct_tracks_survived = round(ns / len(tracks) * 100, 2)
    pop.mean_lifespan_min = round(float(np.mean(ls)), 2)
    pop.std_lifespan_min = round(float(np.std(ls)), 2)
    pop.median_lifespan_min = round(float(np.median(ls)), 2)
    pop.mean_velocity_um_min = round(float(np.mean(vs)), 4)
    pop.mean_displacement_um = round(float(np.mean(ds)), 2)
    pop.mean_confinement_ratio = round(float(np.mean(cs)), 4)
    pop.mean_area_um2 = round(float(np.mean(ar)), 2)
    pop.mean_circularity = round(float(np.mean(ci)), 4)
    pop.mean_area_change_um2 = round(float(np.mean(ac)), 2)
    return pop


def _dynamics(pop, spots, interval_min):
    valid = [s for s in spots if s.segmentation_ok and s.spot_id > 0]
    if not valid:
        return
    by = Counter(s.frame_index for s in valid)
    frames = sorted(by.keys())
    counts = [by[f] for f in frames]
    pop.cell_count_first = counts[0]
    pop.cell_count_last = counts[-1]
    pop.cell_count_max = max(counts)
    pop.cell_count_mean = round(float(np.mean(counts)), 2)
    if len(frames) < 2:
        return
    # Use elapsed_min directly from spots rather than recomputing
    # frame_index * interval_min - these are equal by construction today
    # (input_handler.py sets both from the same source), but reading the
    # field that's actually meant to carry this value avoids relying on
    # that being true forever, and works correctly even if frame_index
    # semantics ever change (e.g. becoming a positional index).
    elapsed_by_frame = {s.frame_index: s.elapsed_min for s in valid}
    times = [elapsed_by_frame[f] for f in frames]
    early = [(t, c) for t, c in zip(times, counts) if t <= 60]
    if len(early) >= 2:
        et, ec = zip(*early)
        pop.arrival_rate_per_min = round(_slope(np.array(et), np.array(ec)), 4)
    total = times[-1] - times[0]
    cutoff = times[0] + total * 0.4
    late = [(t, c) for t, c in zip(times, counts) if t >= cutoff]
    if len(late) >= 2:
        lt2, lc = zip(*late)
        sl = _slope(np.array(lt2), np.array(lc))
        pop.departure_rate_per_min = round(abs(min(sl, 0.0)), 4)
    pop.net_accumulation_rate = round(pop.arrival_rate_per_min - pop.departure_rate_per_min, 4)


def _slope(x, y):
    if len(x) < 2:
        return 0.0
    xv = np.var(x)
    return 0.0 if xv < 1e-10 else float(np.cov(x, y)[0, 1] / xv)
