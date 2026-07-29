"""core/tracker.py - LAP single cell tracking via laptrack (same algorithm as TrackMate)."""
from __future__ import annotations
import logging
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def track_fov(frame_spots, tracking_cfg, pixel_size_um=0.377, n_frames_total=0):
    all_valid = [[s for s in f if s.segmentation_ok and s.spot_id > 0] for f in frame_spots]
    if not any(all_valid):
        log.warning("  No valid spots")
        return [], []

    cutoff = tracking_cfg.min_elapsed_min
    if cutoff is not None:
        n_before = sum(len(f) for f in all_valid)
        trackable = [
            [s for s in f if s.elapsed_min >= cutoff] for f in all_valid
        ]
        excluded = [
            [s for s in f if s.elapsed_min < cutoff] for f in all_valid
        ]
        n_after = sum(len(f) for f in trackable)
        log.info(
            f"  min_elapsed_min={cutoff} - tracking {n_after}/{n_before} spots "
            f"(excluded frames before t={cutoff}min from tracking, kept for counts)"
        )
    else:
        trackable = all_valid
        excluded = [[] for _ in all_valid]

    valid = trackable
    df = _build_df(valid)
    if df.empty:
        log.warning("  No spots remain after min_elapsed_min filter - returning untracked spots only")
        flat = [s for f in all_valid for s in f]
        for s in flat:
            s.track_id = -1
        return flat, []

    log.info(f"  Tracking {len(df)} spots across {len(valid)} frames "
             f"(method={tracking_cfg.tracker_method})")

    if tracking_cfg.tracker_method == "nearest":
        # Published HSEC methodology: nearest centroid within radius, no gap
        # closing. tracking_cfg.max_distance_px should be set from a physical
        # radius (e.g. 150um / pixel_size_um) by the caller for direct
        # comparability with the published results.
        track_df = _centroid(df, tracking_cfg)
    else:
        try:
            track_df = _lap(df, tracking_cfg)
        except Exception as e:
            log.warning(f"  laptrack failed ({e}) - nearest-centroid fallback")
            track_df = _centroid(df, tracking_cfg)
    spots = _assign(valid, track_df)

    # Re-attach excluded (pre-cutoff) spots as untracked so they still
    # contribute to frame-level cell counts / population stats downstream.
    for f in excluded:
        for s in f:
            s.track_id = -1
        spots.extend(f)

    lengths = Counter(s.track_id for s in spots if s.track_id >= 0)
    valid_ids = {t for t, n in lengths.items() if n >= tracking_cfg.min_track_length_frames}
    for s in spots:
        if s.track_id >= 0 and s.track_id not in valid_ids:
            s.track_id = -1
    log.info(f"  {len(valid_ids)} tracks (>= {tracking_cfg.min_track_length_frames} frames)")
    last = max((s.frame_index for s in spots if s.segmentation_ok), default=0)
    if n_frames_total > 0:
        last = max(last, n_frames_total - 1)
    tracks = _make_tracks(spots, valid_ids, pixel_size_um, last)
    return spots, tracks


def _lap(df, cfg):
    import laptrack
    import networkx as nx
    frames = sorted(df["frame"].unique())
    coords = [df[df["frame"] == f][["cx", "cy"]].values for f in frames]
    # IMPORTANT: metric="euclidean" means cutoff is compared directly against
    # real (unsquared) distances from scipy.cdist. laptrack's own docs warn
    # cutoff should only be squared when metric="sqeuclidean" (the library
    # default). Squaring the cutoff here while using "euclidean" previously
    # inflated the real enforced radius by a factor of max_distance_px itself
    # (e.g. max_distance_px=20 silently enforced an effective ~400px radius),
    # which was the root cause of physically implausible long-range track
    # links seen in real HSEC data. Do not square these values.
    lt = laptrack.LapTrack(
        track_dist_metric="euclidean", track_cost_cutoff=cfg.max_distance_px,
        gap_closing_dist_metric="euclidean", gap_closing_cost_cutoff=cfg.gap_closing_max_dist_px,
        gap_closing_max_frame_count=cfg.gap_closing_max_frames,
    )
    G = lt.predict(coords)
    # laptrack's graph nodes are (position_in_coords_list, spot_index_within_frame),
    # NOT the original df["frame"] values - `frames` may be non-contiguous (e.g.
    # after a min_elapsed_min cutoff removes leading frames). Must remap position
    # back to the real frame value via `frames[position]` or downstream code
    # (_assign) will look up the wrong frame and silently produce zero matches.
    rows = []
    for tid, comp in enumerate(nx.weakly_connected_components(G)):
        for node in comp:
            pos, si = node
            real_frame = frames[pos]
            rows.append({"frame": int(real_frame), "spot_index": int(si), "track_id": tid})
    return pd.DataFrame(rows)


def _centroid(df, cfg):
    frames = sorted(df["frame"].unique())
    next_id = 0
    id_map = {}
    f0 = df[df["frame"] == frames[0]]
    for i in range(len(f0)):
        id_map[(frames[0], i)] = next_id
        next_id += 1
    for i in range(1, len(frames)):
        pf, cf = frames[i - 1], frames[i]
        prev = df[df["frame"] == pf]
        curr = df[df["frame"] == cf]
        if prev.empty or curr.empty:
            for j in range(len(curr)):
                id_map[(cf, j)] = next_id
                next_id += 1
            continue
        pc = prev[["cx", "cy"]].values
        cc = curr[["cx", "cy"]].values
        used = set()
        c2p = {}
        for ci, c in enumerate(cc):
            dists = np.sqrt(np.sum((pc - c) ** 2, axis=1))
            pi = int(np.argmin(dists))
            if dists[pi] <= cfg.max_distance_px and pi not in used:
                c2p[ci] = pi
                used.add(pi)
        for ci in range(len(cc)):
            if ci in c2p:
                id_map[(cf, ci)] = id_map.get((pf, c2p[ci]), next_id)
            else:
                id_map[(cf, ci)] = next_id
                next_id += 1
    return pd.DataFrame([{"frame": f, "spot_index": si, "track_id": t} for (f, si), t in id_map.items()])


def _build_df(valid):
    rows = []
    for fi, spots in enumerate(valid):
        for s in spots:
            rows.append({"frame": fi, "frame_index": s.frame_index, "cx": s.centroid_x_px,
                        "cy": s.centroid_y_px, "spot_id": s.spot_id, "fov_id": s.fov_id})
    return pd.DataFrame(rows)


def _assign(valid, track_df):
    # track_df's "frame" column is POSITIONAL (list index into `valid`, set by
    # _build_df's enumerate()), not the real frame_index. This matters once
    # `valid` has been filtered (e.g. by min_elapsed_min) and no longer starts
    # at position 0 == frame_index 0, or contains leading empty lists - using
    # frame_index as the lookup key here would silently mismatch every spot
    # against the wrong (or no) track. Must use the same positional index
    # _build_df used, not frame_index, for the (frame, spot_index) lookup key.
    flat = []
    imap = {(int(r["frame"]), int(r["spot_index"])): int(r["track_id"]) for _, r in track_df.iterrows()}
    for fi, spots in enumerate(valid):
        for pos, s in enumerate(spots):
            s.track_id = imap.get((fi, pos), -1)
            flat.append(s)
    return flat


def _make_tracks(spots, valid_ids, px, last):
    by = defaultdict(list)
    for s in spots:
        if s.track_id in valid_ids:
            by[s.track_id].append(s)
    return [_summary(tid, sl, px, last) for tid, sl in by.items()]


def _summary(tid, spots, px, last):
    from config import TrackRecord
    spots.sort(key=lambda s: s.frame_index)
    fr = [s.frame_index for s in spots]
    ts = [s.elapsed_min for s in spots]
    xs = [s.centroid_x_px for s in spots]
    ys = [s.centroid_y_px for s in spots]
    ar = [s.area_um2 for s in spots]
    ci = [s.circularity for s in spots]
    fs = set(fr)
    gaps = sum(1 for i in range(fr[0], fr[-1]) if i not in fs)
    disps = [np.sqrt((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2) for i in range(1, len(spots))]
    td = float(sum(disps))
    nd = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
    conf = nd / td if td > 0 else 0.0
    dt = [max(ts[i + 1] - ts[i], 0.01) for i in range(len(disps))]
    vels = [d * px / dt[i] for i, d in enumerate(disps)]
    aslope = _slope(np.array(ts), np.array(ar))
    return TrackRecord(
        fov_id=spots[0].fov_id, track_id=tid,
        first_frame=fr[0], last_frame=fr[-1], first_elapsed_min=ts[0], last_elapsed_min=ts[-1],
        lifespan_frames=len(fr), lifespan_min=round(ts[-1] - ts[0], 2), n_gaps=gaps,
        arrived_in_frame=(fr[0] == 0), survived_to_end=(fr[-1] >= last),
        total_displacement_px=round(td, 2), total_displacement_um=round(td * px, 2),
        net_displacement_px=round(nd, 2), net_displacement_um=round(nd * px, 2),
        confinement_ratio=round(conf, 4),
        mean_velocity_um_min=round(float(np.mean(vels)) if vels else 0.0, 4),
        max_velocity_um_min=round(float(np.max(vels)) if vels else 0.0, 4),
        area_first_um2=round(ar[0], 2), area_last_um2=round(ar[-1], 2),
        area_change_um2=round(ar[-1] - ar[0], 2), area_change_rate_um2_min=round(aslope, 4),
        circularity_first=round(ci[0], 4), circularity_last=round(ci[-1], 4),
        circularity_change=round(ci[-1] - ci[0], 4),
        mean_circularity=round(float(np.mean(ci)), 4), mean_area_um2=round(float(np.mean(ar)), 2),
    )


def _slope(x, y):
    if len(x) < 2:
        return 0.0
    xv = np.var(x)
    return 0.0 if xv < 1e-10 else float(np.cov(x, y)[0, 1] / xv)
