"""Per-muscle glow weights on a welded manifold: no UVs, no projection, no shells.

Seeds come from muscle IDs on the unwelded mesh (nearest-mapped), then a
geodesic (graph-distance) falloff softens the sheet edge on the welded
adjacency. The shader mixes those float attributes into peach / purple / pink
emissive sheets on top of the porcelain clay.
"""
from __future__ import annotations
import heapq
import numpy as np

from .weld import world_verts_basis, nearest_map


GROUPS_DEFAULT = (
    {"name": "lats", "ids": ["lats"], "attr": "glow_lats"},
    {"name": "mid_back", "ids": ["rhomboids", "mid_trap"], "attr": "glow_mid_back"},
    {"name": "rear_delt", "ids": ["rear_delt"], "attr": "glow_rear_delt"},
)


def _adjacency(me):
    nv = len(me.vertices)
    ev = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    adj = [[] for _ in range(nv)]
    for a, b in ev:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def _anatomical_seeds(seed, pts, kind):
    """Keep the muscle-ID core that matches the requested silhouette.

    pts are rest-pose world metres (Z-up). kind is lats / mid_back / rear_delt.
    """
    seed = np.asarray(seed, dtype=bool)
    if int(seed.sum()) < 12:
        return seed
    z = pts[:, 2]
    z0 = float(np.percentile(z[seed], 5))
    z1 = float(np.percentile(z[seed], 95))
    zh = (z - z0) / max(z1 - z0, 1e-6)
    cx = float(np.median(pts[:, 0]))
    ax = np.abs(pts[:, 0] - cx)
    if kind == "lats":
        # V / fan from the armpit down the side to the lower back.
        # Keep lateral verts at every height (the side panel); allow more
        # medial extent only toward the lower back. Do not drop the armpit.
        seed_ax = ax[seed]
        lat_max = float(np.percentile(seed_ax, 92))
        t = np.clip(zh, 0.0, 1.0)
        min_ax = (0.10 + 0.30 * t) * lat_max
        return seed & (ax >= min_ax) & (zh > 0.02) & (zh < 0.98)
    if kind == "mid_back":
        # Diamond between the scapulae — not up onto the upper traps.
        seed_ax = ax[seed]
        half = float(np.percentile(seed_ax, 42))
        t = np.clip(zh, 0.0, 1.0)
        diamond = np.where(t < 0.42, t / 0.42, (1.0 - t) / 0.58)
        return seed & (ax <= half * np.maximum(diamond, 0.15) * 1.05) & (t > 0.14) & (t < 0.76)
    if kind == "rear_delt":
        # Posterior deltoid cap only.
        return seed & (zh > 0.22) & (ax >= float(np.percentile(ax[seed], 18)))
    return seed


def geodesic_falloff(seeds, adj, pts, fade_m=0.022):
    """Multi-source Dijkstra on the mesh graph. 1 at seeds, 0 at fade_m."""
    nv = len(adj)
    dist = np.full(nv, np.inf, np.float64)
    heap = []
    for i in np.nonzero(np.asarray(seeds, dtype=bool))[0]:
        dist[int(i)] = 0.0
        heap.append((0.0, int(i)))
    heapq.heapify(heap)
    fade = float(fade_m)
    lim = fade * 1.05
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u] or d > lim:
            continue
        pu = pts[u]
        for v in adj[u]:
            step = float(np.linalg.norm(pts[v] - pu))
            nd = d + step
            if nd < dist[v] and nd <= lim:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    w = np.zeros(nv, np.float32)
    finite = np.isfinite(dist)
    w[finite] = np.clip(1.0 - dist[finite] / max(fade, 1e-6), 0.0, 1.0).astype(np.float32)
    return w


def _write_float(me, name, vals):
    if name in me.attributes:
        try:
            me.attributes.remove(me.attributes[name])
        except Exception:
            pass
    attr = me.attributes.new(name, "FLOAT", "POINT")
    attr.data.foreach_set("value", np.asarray(vals, np.float32).ravel())


def hues_rgb_from_cfg(cfg):
    import muscle_ids as MIDS
    hues = dict(MIDS.DEFAULT_HUES)
    yh = (cfg.get("grade") or {}).get("muscle_hues") or {}
    if isinstance(yh, dict):
        hues.update(yh)
    out = {}
    mapping = {
        "lats": "lats",
        "mid_back": "rhomboids",
        "rear_delt": "rear_delt",
    }
    for key, src in mapping.items():
        spec = hues.get(src) or hues.get(key) or {}
        if isinstance(spec, dict):
            out[key] = MIDS.hsv_to_lin_rgb(
                float(spec.get("h", 10)), float(spec.get("s", 200)), float(spec.get("v", 244)))
        else:
            out[key] = (0.9, 0.35, 0.08)
    return out


def write_weights(welded, src_body, tid, names, cfg, shorts=None, exclude=None, log=print):
    kit = (cfg.get("kit") or {})
    mus = kit.get("muscles") or {}
    groups = mus.get("groups") or GROUPS_DEFAULT
    fade = float(mus.get("fade_m", 0.022))

    src_w = world_verts_basis(src_body)
    dst_w = world_verts_basis(welded)
    if "src_vi" in welded.data.attributes:
        nearest = np.zeros(len(welded.data.vertices), np.int32)
        welded.data.attributes["src_vi"].data.foreach_get("value", nearest)
    else:
        nearest = nearest_map(src_w, dst_w)

    name_to_id = {n: i for i, n in enumerate(names or [])}
    adj = _adjacency(welded.data)
    info = {"fade_m": fade, "groups": {}}
    combined = np.zeros(len(dst_w), np.float32)

    shorts_w = None
    if shorts is not None:
        shorts_w = np.asarray(shorts, dtype=bool)[nearest]
    excl_w = None
    if exclude is not None:
        excl_w = np.asarray(exclude, dtype=bool)[nearest]

    for g in groups:
        ids = list(g.get("ids") or [g.get("name")])
        seed = np.zeros(len(dst_w), dtype=bool)
        if tid is not None:
            src_tid = np.asarray(tid)
            for nm in ids:
                if nm in name_to_id:
                    seed |= src_tid[nearest] == name_to_id[nm]
        if shorts_w is not None:
            seed &= ~shorts_w
        if excl_w is not None:
            seed &= ~excl_w
        seed = _anatomical_seeds(seed, dst_w, str(g.get("name") or ""))
        w = geodesic_falloff(seed, adj, dst_w, fade_m=fade)
        if shorts_w is not None:
            w[shorts_w] = 0.0
        attr_name = g.get("attr") or f"glow_{g['name']}"
        _write_float(welded.data, attr_name, w)
        combined = np.maximum(combined, w)
        rec = {
            "seeds": int(seed.sum()),
            "n_gt_05": int((w > 0.5).sum()),
            "n_gt_0": int((w > 0.02).sum()),
            "max": float(w.max()) if w.size else 0.0,
            "mean": float(w.mean()) if w.size else 0.0,
        }
        info["groups"][attr_name] = rec
        log(f"glow attr {attr_name} seeds={rec['seeds']} >0.5={rec['n_gt_05']} "
            f">0={rec['n_gt_0']} max={rec['max']:.3f}")

    _write_float(welded.data, "glow_w", combined)
    if shorts_w is not None:
        _write_float(welded.data, "shorts", shorts_w.astype(np.float32))
    info["combined_gt_05"] = int((combined > 0.5).sum())
    info["combined_max"] = float(combined.max()) if combined.size else 0.0
    return info
