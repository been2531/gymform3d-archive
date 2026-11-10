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
    cy = float(np.median(pts[:, 1]))
    # Lats live on the back; the sign of (lat Y - torso Y) is the posterior side.
    back_sign = np.sign(float(np.median(pts[seed, 1]) - cy)) or -1.0
    post = (pts[:, 1] - cy) * back_sign
    if kind == "lats":
        # V / fan on the BACK from the posterior armpit to the lower back.
        # Cap lateral extent so the sheet does not become a rib-side block.
        seed_ax = ax[seed]
        lat_max = float(np.percentile(seed_ax, 92))
        t = np.clip(zh, 0.0, 1.0)
        post_min = float(np.percentile(post[seed], 20))
        post_span = max(float(np.percentile(post[seed], 90) - post_min), 1e-6)
        backness = np.clip((post - post_min) / post_span, 0.0, 1.0)
        # Inner edge stays off the spine and moves out toward the armpit.
        min_ax = (0.06 + 0.20 * t) * lat_max
        # Outer edge is wide on the BACK so mid-lat connects into a V,
        # tight on the side so the sheet is not a rib-wall block.
        max_ax = (0.48 + 0.16 * t + 0.40 * backness) * lat_max
        return (seed & (ax >= min_ax) & (ax <= max_ax) & (post >= post_min)
                & (zh > 0.02) & (zh < 0.96))
    if kind == "mid_back":
        # Diamond between the scapulae — not up onto the upper traps.
        seed_ax = ax[seed]
        half = float(np.percentile(seed_ax, 38))
        t = np.clip(zh, 0.0, 1.0)
        diamond = np.where(t < 0.40, t / 0.40, (1.0 - t) / 0.60)
        return seed & (ax <= half * np.maximum(diamond, 0.14) * 1.00) & (t > 0.16) & (t < 0.74)
    if kind == "rear_delt":
        # Posterior deltoid cap only.
        return seed & (zh > 0.22) & (ax >= float(np.percentile(ax[seed], 22)))
    return seed


def _close_seed(seed, adj, rings=2):
    """Mesh-graph close so seed silhouettes are solid, not ragged islands."""
    s = np.asarray(seed, dtype=bool)
    if int(s.sum()) < 8:
        return s
    for _ in range(int(rings)):
        nxt = s.copy()
        for i in np.flatnonzero(s):
            nxt[adj[i]] = True
        s = nxt
    for _ in range(int(rings)):
        nxt = s.copy()
        for i in np.flatnonzero(s):
            nbrs = adj[i]
            if nbrs and (not all(s[j] for j in nbrs)):
                nxt[i] = False
        s = nxt
    return s


def _dilate_seed(seed, adj, rings=1):
    s = np.asarray(seed, dtype=bool).copy()
    for _ in range(int(rings)):
        nxt = s.copy()
        for i in np.flatnonzero(s):
            nxt[adj[i]] = True
        s = nxt
    return s


def _bridge_components(seed, adj, pts, allowed=None):
    """Reconnect split islands of one muscle along an allowed corridor."""
    s = np.asarray(seed, dtype=bool).copy()
    idx = np.flatnonzero(s)
    if len(idx) < 24:
        return s
    nv = len(s)
    comp = np.full(nv, -1, np.int32)
    cid = 0
    for i in idx:
        if comp[i] >= 0:
            continue
        stack = [int(i)]
        comp[i] = cid
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if s[v] and comp[v] < 0:
                    comp[v] = cid
                    stack.append(int(v))
        cid += 1
    if cid <= 1:
        return s
    sizes = sorted(((int((comp == c).sum()), c) for c in range(cid)), reverse=True)
    if sizes[1][0] < 24:
        return s
    a, b = sizes[0][1], sizes[1][1]
    pa, pb = pts[comp == a], pts[comp == b]
    ia = int(np.flatnonzero(comp == a)[np.argmin(np.linalg.norm(pa - pb.mean(0), axis=1))])
    ib = int(np.flatnonzero(comp == b)[np.argmin(np.linalg.norm(pb - pa.mean(0), axis=1))])
    dist = np.full(nv, np.inf)
    prev = np.full(nv, -1, np.int32)
    dist[ia] = 0.0
    heap = [(0.0, ia)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == ib:
            break
        if d > dist[u]:
            continue
        pu = pts[u]
        for v in adj[u]:
            if allowed is not None and (not allowed[v]):
                continue
            nd = d + float(np.linalg.norm(pts[v] - pu))
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, int(v)))
    if not np.isfinite(dist[ib]):
        return s
    cur = ib
    guard = 0
    while cur != ia and cur >= 0 and guard < nv:
        s[cur] = True
        cur = int(prev[cur])
        guard += 1
    return s


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


def _smooth_weights(w, adj, iters=12, lam=0.50):
    """Laplacian smooth on the welded mesh so each sheet is a clean fill."""
    w = np.asarray(w, np.float32).copy()
    nv = len(w)
    pairs = [(i, int(j)) for i, nbrs in enumerate(adj) for j in nbrs if i < j]
    if not pairs:
        return w
    ev = np.asarray(pairs, np.int64)
    a, b = ev[:, 0], ev[:, 1]
    for _ in range(int(iters)):
        acc = np.zeros(nv, np.float32)
        cnt = np.zeros(nv, np.float32)
        np.add.at(acc, a, w[b])
        np.add.at(acc, b, w[a])
        np.add.at(cnt, a, 1.0)
        np.add.at(cnt, b, 1.0)
        ok = cnt > 0
        avg = w.copy()
        avg[ok] = acc[ok] / cnt[ok]
        w = (1.0 - float(lam)) * w + float(lam) * avg
    return np.clip(w, 0.0, 1.0)


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
        raw = seed.copy()
        kind = str(g.get("name") or "")
        seed = _anatomical_seeds(seed, dst_w, kind)
        if kind == "lats" and int(seed.sum()) > 24:
            # Stay on original lat IDs that face back, so the bridge cannot
            # walk around the rib wall (that was the side block).
            z = dst_w[:, 2]
            z0 = float(np.percentile(z[raw], 5)) if int(raw.sum()) else 0.0
            z1 = float(np.percentile(z[raw], 95)) if int(raw.sum()) else 1.0
            zh = (z - z0) / max(z1 - z0, 1e-6)
            cy = float(np.median(dst_w[:, 1]))
            back_sign = np.sign(float(np.median(dst_w[raw, 1]) - cy)) or -1.0
            post = (dst_w[:, 1] - cy) * back_sign
            post_min = float(np.percentile(post[raw], 20))
            corridor = raw & (post >= post_min) & (zh > 0.02) & (zh < 0.96)
            seed = _bridge_components(seed, adj, dst_w, allowed=corridor)
            seed = _dilate_seed(seed, adj, rings=3)
        seed = _close_seed(seed, adj, rings=3)
        w = geodesic_falloff(seed, adj, dst_w, fade_m=fade)
        w = _smooth_weights(w, adj, iters=12, lam=0.50)
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
