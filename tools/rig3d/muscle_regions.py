"""Shared anatomical muscle-region weights (rig3d).

Regions are painted in a torso-local frame (spine / shoulder line / back) so they
follow a deformed mesh instead of a rest-pose world AABB. Rest-space AABBs lose
mid-trap / rhomboid / rear-delt coverage when the shoulder retracts or the elbow
tucks — those verts bunch toward the spine and newly visible back surface was
never painted.

Channels (match the mask pass): R = lats, G = rhomboids, B = mid-trap | rear delt.
`flex` (0 = start / extended, 1 = peak contraction) grows the painted set along
the back-facing trunk so peak glow is a larger continuous region than start.
"""
from __future__ import annotations
import hashlib, json, os
import numpy as np

EXCLUDE_SUB = (
    "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "Toe", "Head", "neck", "eye",
    "Index", "Thumb", "Middle", "Ring", "Pinky",
)


def adjacency(nv, edges):
    """List of neighbor-index arrays from an (E, 2) edge array."""
    adj = [[] for _ in range(int(nv))]
    for a, b in np.asarray(edges, dtype=np.int64):
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    return [np.asarray(n, dtype=np.int32) for n in adj]


def exclude_mask(dom, group_names):
    """True = this vertex is a limb / head / finger and must not glow."""
    out = np.zeros(len(dom), dtype=bool)
    for i, gi in enumerate(dom):
        nm = (group_names.get(int(gi), "") if gi >= 0 else "").lower()
        if any(s.lower() in nm for s in EXCLUDE_SUB):
            out[i] = True
    return out


def _bone(bones, *names):
    """Case-insensitive lookup; Mixamo assets often use `neck` not `Neck`."""
    cmap = {str(k).lower(): v for k, v in bones.items() if v is not None}
    for n in names:
        if n in bones and bones[n] is not None:
            return np.asarray(bones[n], dtype=np.float64)
        if n.lower() in cmap:
            return np.asarray(cmap[n.lower()], dtype=np.float64)
    return None

def torso_frame(bones):
    """Return hip, spine_hat, side_hat, back_hat, spine_len from posed or rest bone heads."""
    hip = _bone(bones, "Hips", "Spine")
    if hip is None:
        hip = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # Never use Spine02 as the neck: on this Grab3D rig Spine02 is the *lower* spine
    # (14 cm above the hips). Prefer neck/Head so `t` spans the whole back.
    neck = _bone(bones, "Neck", "Head", "head_end")
    if neck is None:
        candidates = [ _bone(bones, n) for n in ("Spine", "Spine01", "Spine02") ]
        candidates = [c for c in candidates if c is not None]
        neck = max(candidates, key=lambda c: float(np.linalg.norm(c - hip))) if candidates else hip + np.array([0.0, 0.0, 0.55])
    spine = neck - hip; slen = float(np.linalg.norm(spine)) or 1e-6; spine = spine / slen
    # Laterality must NOT come from the working Arm: that bone swings through the row and
    # tilts the back axis, dropping mid-trap coverage at the start. Planted hip sockets
    # (UpLeg heads) are stable; Shoulder/Arm are fallbacks (Shoulder span is only ~7 cm here).
    sh_l = _bone(bones, "LeftUpLeg", "LeftShoulder", "LeftArm")
    sh_r = _bone(bones, "RightUpLeg", "RightShoulder", "RightArm")
    if sh_l is None: sh_l = hip + np.array([-0.18, 0, 0.0])
    if sh_r is None: sh_r = hip + np.array([0.18, 0, 0.0])
    side = sh_r - sh_l; side = side - side.dot(spine) * spine
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float64); side = side - side.dot(spine) * spine
    side = side / (np.linalg.norm(side) + 1e-12)
    back = np.cross(side, spine); back = back / (np.linalg.norm(back) + 1e-12)
    return hip, spine, side, back, slen


def _dilate(mask, adj, steps, allowed):
    m = mask.copy()
    for _ in range(int(steps)):
        grow = np.zeros_like(m)
        for i in np.flatnonzero(m):
            nb = adj[int(i)]
            if len(nb):
                grow[nb] = True
        m = m | (grow & allowed)
    return m


def assign(co, nr, bones, adj, shorts, exclude, flex=0.0, peak_dilate=3, base_dilate=4, prior=None):
    """Paint (N, 3) float32 region weights for the given (possibly deformed) vertex positions.

    `shorts` / `exclude` are rest-stable boolean masks (vertex indices do not change).
    `flex` in [0, 1] adds extra mesh-ring dilation at peak contraction.
    """
    co = np.asarray(co, dtype=np.float64); nr = np.asarray(nr, dtype=np.float64)
    nv = len(co)
    hip, spine, side, back, slen = torso_frame(bones)
    # pick the back-facing sign from trunk normals near the sagittal plane
    rel = co - hip
    t = (rel @ spine) / slen
    lat = rel @ side
    bck = rel @ back
    nb = nr @ back
    medial = (np.abs(lat) < 0.10) & (t > 0.25) & (t < 0.90) & ~shorts & ~exclude
    if medial.any() and float(nb[medial].mean()) < 0.0:
        back = -back; side = -side
        lat = rel @ side; bck = rel @ back; nb = nr @ back
    keep = ~shorts & ~exclude
    # Start (flex=0): the hinge tips upper-back normals off the torso "back" axis, so a
    # tight backish test drops mid-trap / rhomboids (v3.1 start holes). Loosen at low
    # flex; at peak (flex=1) keep the original tight window so glow_frac stays capped.
    fx = float(np.clip(flex, 0.0, 1.0))
    nb_lo = -0.10 + 0.12 * fx
    bck_lo = -0.14 + 0.08 * fx
    backish = (nb > nb_lo) & (bck > bck_lo)
    sideish = np.abs(nr @ side) > 0.45

    # mid-trap extends to t=0.94 (upper medial trap near the neck). The previous
    # t<=0.86 left a gap that only peak-flex dilation filled — that was the start-frame hole.
    mid_trap = keep & backish & (t >= 0.48) & (t <= 0.94) & (np.abs(lat) <= 0.16)
    rhomb = keep & backish & (t >= 0.38) & (t < 0.74) & (np.abs(lat) >= 0.008) & (np.abs(lat) <= 0.12) & ~mid_trap
    lat_lo = 0.032 + 0.20 * np.clip(t - 0.08, 0.0, 0.60)
    lat_hi = 0.090 + 0.50 * np.clip(t - 0.08, 0.0, 0.60)
    lats = keep & (t >= 0.08) & (t <= 0.68) & (np.abs(lat) >= lat_lo) & (np.abs(lat) <= lat_hi)
    lats &= ((nb > 0.00) | sideish) & (nb > -0.30) & ~mid_trap & ~rhomb
    rear_delt = keep & backish & (t >= 0.58) & (t <= 0.94) & (np.abs(lat) >= 0.10) & (np.abs(lat) <= 0.22)
    rear_delt &= (bck > -0.02) & ~mid_trap & ~rhomb

    # Rig-skinning prior: verts weighted to spine/shoulder groups are anatomically
    # the back sheet. This does not depend on posed normals, so a hinged start
    # still covers mid-trap / upper trap that the backish test drops.
    if prior is not None:
        pr = np.asarray(prior, dtype=bool)[:nv] & keep
        sheet = pr & (t >= 0.12) & (t <= 0.96) & (np.abs(lat) < 0.28) & (nb > -0.28)
        mid_trap = mid_trap | (sheet & (t >= 0.46) & (np.abs(lat) <= 0.18))
        rhomb = rhomb | (sheet & (t >= 0.36) & (t < 0.74) & (np.abs(lat) >= 0.006) & (np.abs(lat) <= 0.13) & ~mid_trap)
        rear_delt = rear_delt | (sheet & (t >= 0.56) & (np.abs(lat) >= 0.10) & ~mid_trap & ~rhomb)
        lats = lats | (sheet & (t <= 0.70) & (np.abs(lat) >= 0.06) & ~mid_trap & ~rhomb & ~rear_delt)

    allowed = keep & (nb > (nb_lo - 0.06)) & (t >= 0.06) & (t <= 0.92) & (np.abs(lat) < 0.28)
    # fill pinholes and connect lats / rhomb / mid-trap / rear-delt into one sheet
    mid_trap = _dilate(mid_trap, adj, base_dilate, allowed & (t >= 0.46))
    rhomb = _dilate(rhomb, adj, base_dilate, allowed & (t >= 0.34) & (t <= 0.80) & ~mid_trap)
    lats = _dilate(lats, adj, base_dilate, allowed & (t <= 0.72) & ~mid_trap & ~rhomb)
    rear_delt = _dilate(rear_delt, adj, base_dilate, allowed & (t >= 0.52) & ~mid_trap & ~rhomb)

    extra = 2 + int(round(float(peak_dilate) * float(np.clip(flex, 0.0, 1.0))))
    if extra > 0:
        any0 = lats | rhomb | mid_trap | rear_delt
        grown = _dilate(any0, adj, extra, allowed)
        # new verts go to the nearest existing channel (by priority: rhomb, mid_trap, lats, rear_delt)
        new = grown & ~any0
        mid_trap = mid_trap | (new & (t >= 0.56))
        rhomb = rhomb | (new & (t >= 0.40) & (t < 0.74) & ~mid_trap)
        rear_delt = rear_delt | (new & (t >= 0.58) & ~mid_trap & ~rhomb)
        lats = lats | (new & ~mid_trap & ~rhomb & ~rear_delt)

    blue = mid_trap | rear_delt
    # exclusive-ish: keep overlaps (grade uses max) but prefer rhomb over lats on the same vert
    lats = lats & ~rhomb & ~mid_trap
    reg = np.stack([lats, rhomb, blue], 1).astype(np.float32)
    info = {"lats": int(lats.sum()), "rhomboids": int(rhomb.sum()), "mid_trap": int(mid_trap.sum()),
            "rear_delt": int(rear_delt.sum()), "flex": round(float(flex), 3), "extra_dilate": extra}
    return reg, info


def write_mregion(me, reg):
    """Write an (N, 3) region array into the mesh's `mregion` POINT FLOAT_COLOR attribute."""
    nv = len(reg)
    col = np.concatenate([np.asarray(reg, dtype=np.float32), np.ones((nv, 1), np.float32)], 1).ravel()
    va = me.color_attributes.get("mregion")
    if va is None:
        va = me.color_attributes.new("mregion", "FLOAT_COLOR", "POINT")
    va.data.foreach_set("color", col)
    me.update()


def smooth_vertex_groups(body, names, adj, factor=0.5, repeat=6):
    """Laplacian-smooth named vertex groups without bpy.ops (headless-safe)."""
    nv = len(body.data.vertices)
    did = []
    for name in names:
        vg = body.vertex_groups.get(name)
        if vg is None:
            continue
        gi = vg.index
        w = np.zeros(nv, dtype=np.float64)
        for v in body.data.vertices:
            for g in v.groups:
                if g.group == gi:
                    w[v.index] = g.weight
        for _ in range(int(repeat)):
            nw = w.copy()
            for i, nb in enumerate(adj):
                if len(nb):
                    nw[i] = (1.0 - factor) * w[i] + factor * float(w[nb].mean())
            w = nw
        for i, val in enumerate(w):
            if val > 1e-5:
                vg.add([i], float(min(val, 1.0)), "REPLACE")
            elif val <= 1e-5:
                try:
                    vg.remove([i])
                except RuntimeError:
                    pass
        did.append(name)
    return did


def vertex_group_weights(body, names):
    """Read named vertex-group weights as {name: (N,) float64}."""
    nv = len(body.data.vertices)
    want = {}
    for name in names:
        vg = body.vertex_groups.get(name)
        if vg is not None:
            want[vg.index] = name
    out = {n: np.zeros(nv, dtype=np.float64) for n in want.values()}
    for v in body.data.vertices:
        for g in v.groups:
            if g.group in want:
                out[want[g.group]][v.index] = g.weight
    return out


def apply_vertex_group_weights(body, weights):
    """Replace named vertex-group weights from a {name: (N,)} mapping."""
    for name, w in weights.items():
        vg = body.vertex_groups.get(name)
        if vg is None:
            continue
        w = np.asarray(w, dtype=np.float64)
        for i, val in enumerate(w):
            if val > 1e-5:
                vg.add([i], float(min(val, 1.0)), "REPLACE")
            else:
                try:
                    vg.remove([i])
                except RuntimeError:
                    pass


def asset_cache_key(glb_path, params):
    """Stable id: size + head/tail bytes of the GLB plus the smooth params."""
    st = os.stat(glb_path)
    h = hashlib.sha256()
    h.update(str(int(st.st_size)).encode())
    with open(glb_path, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > (2 << 20):
            f.seek(-1 << 20, 2)
            h.update(f.read())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def cached_smooth_vertex_groups(body, names, adj, factor, repeat, cache_dir, glb_path):
    """Laplacian-smooth named groups, cached per asset hash + params (one-time ~21s)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = asset_cache_key(glb_path, {"names": list(names), "factor": float(factor), "repeat": int(repeat), "v": 1})
    path = os.path.join(cache_dir, f"vg_smooth_{key}.npz")
    if os.path.exists(path):
        data = {k: np.asarray(v) for k, v in np.load(path).items()}
        apply_vertex_group_weights(body, data)
        return list(data.keys()), "hit", path
    did = smooth_vertex_groups(body, names, adj, factor=factor, repeat=repeat)
    weights = vertex_group_weights(body, did)
    if weights:
        np.savez_compressed(path, **weights)
    return did, "miss", path
