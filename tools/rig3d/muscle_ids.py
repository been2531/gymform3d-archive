"""Persistent per-muscle IDs for a rigged Grab3D-style body (rig3d v3.3).

The Grab3D GLB is a single mesh / single material. Muscles are separate UV
islands painted on a terracotta atlas (rig_basecolor). We segment those islands
in UV space, classify each island from rest-pose 3D landmarks + dominant skin
bone, and store a uint8 per vertex keyed by asset hash. The IDs ride with the
mesh (vertex attribute) and are rendered as an ID pass — no per-frame positional
heuristics and no dilation.
"""
from __future__ import annotations
import hashlib, json, os, struct
import numpy as np

MUSCLE_NAMES = [
    "none", "lats", "rhomboids", "mid_trap", "upper_trap",
    "rear_delt", "side_delt", "front_delt",
    "biceps", "triceps", "forearm",
    "pecs", "abs", "obliques",
    "glutes", "quads", "hams", "calves", "erectors",
]
NAME_TO_ID = {n: i for i, n in enumerate(MUSCLE_NAMES)}

# Well-separated RGB so nearest-colour decode survives resize.
PALETTE = np.array([
    [0,   0,   0],     # none
    [220, 32,  36],    # lats
    [46,  196, 48],    # rhomboids
    [36,  72,  220],   # mid_trap
    [0,   180, 180],   # upper_trap
    [240, 140, 16],    # rear_delt
    [180, 80,  220],   # side_delt
    [255, 80,  140],   # front_delt
    [80,  160, 255],   # biceps
    [160, 40,  40],    # triceps
    [200, 180, 40],    # forearm
    [255, 120, 90],    # pecs
    [90,  220, 160],   # abs
    [40,  120, 80],    # obliques
    [140, 90,  50],    # glutes
    [60,  80,  160],   # quads
    [120, 50,  140],   # hams
    [50,  50,  90],    # calves
    [20,  140, 90],    # erectors
], dtype=np.uint8)

def asset_key(glb_path):
    st = os.stat(glb_path)
    h = hashlib.sha256()
    h.update(str(int(st.st_size)).encode())
    with open(glb_path, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > (2 << 20):
            f.seek(-1 << 20, 2)
            h.update(f.read())
    h.update(b"muscle_ids_v5")
    return h.hexdigest()[:20]


def _read_glb(glb_path):
    data = open(glb_path, "rb").read()
    chunk_len = struct.unpack_from("<I", data, 12)[0]
    js = json.loads(data[20:20 + chunk_len])
    off = 20 + chunk_len
    blob = data[off + 8:off + 8 + struct.unpack_from("<I", data, off)[0]]
    prim = js["meshes"][0]["primitives"][0]
    accs, bvs = js["accessors"], js["bufferViews"]

    def view(attr, ncomp, dtype, byte_off_fallback):
        a = accs[prim["attributes"][attr]]
        bv = bvs[a["bufferView"]]
        stride = int(bv.get("byteStride") or (np.dtype(dtype).itemsize * ncomp))
        base = int(bv.get("byteOffset", 0)) + int(a.get("byteOffset", byte_off_fallback))
        raw = np.frombuffer(blob[int(bv.get("byteOffset", 0)):int(bv.get("byteOffset", 0)) + int(bv["byteLength"])], dtype=np.uint8)
        n = int(a["count"])
        raw = raw.reshape(n, stride)
        rel = int(a.get("byteOffset", byte_off_fallback))
        sl = raw[:, rel:rel + np.dtype(dtype).itemsize * ncomp]
        return np.ascontiguousarray(sl).view(dtype).reshape(n, ncomp)

    pos = view("POSITION", 3, np.float32, 0)
    nrm = view("NORMAL", 3, np.float32, 12)
    uv = view("TEXCOORD_0", 2, np.float32, 24)
    joints = view("JOINTS_0", 4, np.uint8, 32)
    weights = view("WEIGHTS_0", 4, np.float32, 36)
    ia = accs[prim["indices"]]; ibv = bvs[ia["bufferView"]]
    idx = np.frombuffer(blob[int(ibv.get("byteOffset", 0)):int(ibv.get("byteOffset", 0)) + int(ibv["byteLength"])], dtype=np.uint32).copy()
    jnames = [js["nodes"][j].get("name", str(j)) for j in js["skins"][0]["joints"]]
    # albedo jpeg
    img_info = [im for im in js["images"] if "basecolor" in im.get("name", "").lower()][0]
    bv = bvs[img_info["bufferView"]]
    jpeg = bytes(blob[int(bv.get("byteOffset", 0)):int(bv.get("byteOffset", 0)) + int(bv["byteLength"])])
    return {"pos": pos, "nrm": nrm, "uv": uv, "joints": joints, "weights": weights, "idx": idx,
            "jnames": jnames, "albedo_jpeg": jpeg, "license": (js.get("nodes") or [{}])[-1].get("extras") if js.get("nodes") else None}


def _albedo_fg(jpeg):
    import cv2
    im = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    # Terracotta atlas background (BGR ~ 21,37,134)
    bg = np.array([24.0, 38.0, 135.0], np.float32)
    dist = np.linalg.norm(im.astype(np.float32) - bg, axis=-1)
    fg = (dist > 38).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    nlab, lab = cv2.connectedComponents(fg, connectivity=8)
    return im, fg, lab, nlab


def _classify_vertex(y, x, z, nz, ny, bone):
    """Rest-pose Y-up metres (feet ~0, head ~1.80). +Z is the chest (verified on Grab3D)."""
    ax = abs(float(x))
    z = float(z); nz = float(nz); ny = float(ny); y = float(y)
    back = (z < -0.018) or (nz < -0.16 and z < 0.02)
    front = (z > 0.022) or (nz > 0.16 and z > -0.015)
    if z < -0.03: back, front = True, False
    if z > 0.03: back, front = False, True
    up = ny > 0.40
    b = (bone or "").lower().replace(" ", "")
    if "foot" in b or "toe" in b or "head" in b:
        return "none"
    if "hand" in b and "fore" not in b:
        return "none"
    if "forearm" in b:
        return "forearm"
    if y < 0.52 and ("leg" in b and "up" not in b):
        return "calves"
    if 0.46 <= y <= 1.02 and "upleg" in b:
        if back and y >= 0.86 and ax >= 0.04:
            return "glutes"
        return "hams" if back else "quads"
    # Proximal Arm weights paint the lat insertion / armpit — keep those as lats (teres
    # major rides with the lat insertion; it has no separate ID). Distal/posterior arm = tri/bi.
    if "arm" in b and "fore" not in b and "shoulder" not in b and "upleg" not in b:
        if ax >= 0.20 or y < 1.28:
            return "triceps" if back else "biceps"
        if back:
            return "lats"
    if "shoulder" in b or (1.28 <= y <= 1.58 and ax >= 0.13):
        # side delt is the lateral cap; the old "any back-facing → rear_delt" swallowed it
        # (only 25 verts survived as side_delt).
        if 1.32 <= y <= 1.55 and ax >= 0.155 and abs(nz) <= 0.45:
            return "side_delt"
        if back and ax >= 0.14 and y >= 1.42:
            return "rear_delt"
        if front and ax >= 0.12:
            return "front_delt"
        if ax >= 0.15:
            return "side_delt"
    # Exclusive back-torso partition so target muscles (lats/rhomb/mid-trap/rear-delt) meet.
    if back and 0.80 <= y <= 1.68:
        if 0.80 <= y < 1.00 and ax >= 0.045:
            return "glutes"
        if ax < 0.058:
            if y < 1.34: return "erectors"
            if y < 1.54: return "mid_trap"
            return "upper_trap"
        if ax < 0.128:
            if y < 1.18: return "lats"
            if y < 1.50: return "rhomboids"
            return "upper_trap"
        # lateral back: y < 1.42 includes teres major / lat insertion with the lats
        # (both are primary row targets). Rear delt stays on the shoulder cap (y >= 1.42).
        if y < 1.42: return "lats"
        if y < 1.58: return "rear_delt"
        return "upper_trap"
    if front:
        if 0.92 <= y <= 1.28 and ax < 0.12: return "abs"
        if 0.92 <= y <= 1.34 and ax >= 0.10: return "obliques"
        if 1.18 <= y <= 1.48 and ax <= 0.22: return "pecs"
        if 1.32 <= y <= 1.56 and ax >= 0.12: return "front_delt"
    if (not front) and (not back):
        if 1.00 <= y <= 1.44 and ax >= 0.08: return "lats"
        if 1.30 <= y <= 1.58 and ax >= 0.12: return "side_delt"
        if 0.92 <= y <= 1.32 and ax >= 0.10: return "obliques"
    if "neck" in b:
        return "upper_trap" if (back or up) else "none"
    if "spine" in b or "hip" in b:
        if back and ax < 0.08: return "erectors" if y < 1.36 else "mid_trap"
        if back: return "lats"
        if front and y < 1.28: return "abs"
        if front: return "pecs"
    return "none"


def build_from_glb(glb_path):
    import cv2
    M = _read_glb(glb_path)
    pos, nrm, uv = M["pos"], M["nrm"], M["uv"]
    joints, weights, idx = M["joints"], M["weights"], M["idx"]
    jnames = list(M["jnames"])
    nv = len(pos)
    dom_i = joints[np.arange(nv), np.argmax(weights, axis=1)]
    bones = np.array([jnames[int(i)] if int(i) < len(jnames) else "" for i in dom_i])
    # facing: mean normal of mid-torso
    torso = (pos[:, 1] > 1.05) & (pos[:, 1] < 1.40) & (np.abs(pos[:, 0]) < 0.15)
    face = nrm[torso].mean(0) if torso.any() else np.array([0, 0, 1.0])
    # if the chest points toward -Z, flip so +nz is front
    if face[2] < 0:
        nrm = nrm.copy(); nrm[:, 2] *= -1
    # +Z is front on this asset — do not flip (front+back torso normals cancel).
    raw = np.array([_classify_vertex(float(pos[i, 1]), float(pos[i, 0]), float(pos[i, 2]),
                                    float(nrm[i, 2]), float(nrm[i, 1]), bones[i]) for i in range(nv)])

    faces = idx.reshape(-1, 3)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        A = coo_matrix((np.ones(len(e0), np.uint8), (e0, e1)), shape=(nv, nv))
        _ncc, island = connected_components(A, directed=False)
    except Exception:
        parent = np.arange(nv, dtype=np.int32)
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for a, b, c in faces:
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
            ra, rc = find(a), find(c)
            if ra != rc: parent[rc] = ra
        island = np.array([find(i) for i in range(nv)])
    # majority vote per mesh UV-island (verts are split at UV seams)
    voted = raw.copy()
    for k in np.unique(island):
        sel = np.flatnonzero(island == k)
        if len(sel) < 8:
            continue
        # Large charts wrap multiple muscles (whole arm, torso quarter) — keep per-vert labels.
        # Small charts are single painted islands; majority-vote them, but split by facing
        # so a side-wrap chart cannot paint the chest as a back muscle.
        if len(sel) > 280:
            continue
        zsel = pos[sel, 2]
        for mask in (zsel > 0.02, zsel < -0.02, np.abs(zsel) <= 0.02):
            sub = sel[mask]
            if len(sub) < 8:
                continue
            vals, cnts = np.unique(raw[sub], return_counts=True)
            order = np.argsort(-cnts)
            pick = vals[order[0]]
            for j in order:
                if vals[j] != "none" and cnts[j] >= 0.22 * len(sub):
                    pick = vals[j]; break
            voted[sub] = pick

    # mesh-adjacency majority smooth; "none" votes are weaker so labelled muscle sheets hold
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
                        faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]])
    src, dst = e[:, 0], e[:, 1]
    ids = np.array([NAME_TO_ID.get(n, 0) for n in voted], dtype=np.uint8)
    nlab = len(MUSCLE_NAMES)
    TARGET = {NAME_TO_ID[n] for n in ("lats", "rhomboids", "mid_trap", "rear_delt")}
    for _ in range(5):
        votes = np.zeros((nv, nlab), np.float32)
        np.add.at(votes, (src, ids[dst]), 1)
        votes[:, 0] *= 0.28
        m = votes.sum(1) > 0
        ids = ids.copy()
        ids[m] = votes[m].argmax(1).astype(np.uint8)
    # Fill none holes whose neighbours are target muscles (keeps start-frame sheet continuous)
    votes = np.zeros((nv, nlab), np.float32)
    np.add.at(votes, (src, ids[dst]), 1)
    votes[:, 0] = 0
    tgt_votes = votes[:, list(TARGET)].sum(1)
    hole = (ids == 0) & (tgt_votes >= 2)
    if hole.any():
        ids[hole] = votes[hole].argmax(1).astype(np.uint8)

    counts = {MUSCLE_NAMES[i]: int((ids == i).sum()) for i in range(len(MUSCLE_NAMES))}
    return {
        "vert_id": ids, "names": MUSCLE_NAMES, "palette": PALETTE, "counts": counts,
        "nv": int(nv), "key": asset_key(glb_path), "src": os.path.abspath(glb_path),
    }


def palette_rgb(vert_id):
    vid = np.clip(np.asarray(vert_id, dtype=np.int32), 0, len(PALETTE) - 1)
    return PALETTE[vid]


def decode_id_image(bgr):
    """Decode an ID-pass image.

    Preferred encoding (v3.3+): Raw view-transform emission of `mid` (R = muscle
    index 0-18, G=B=0). Palette RGB is the fallback for older passes; distances
    MUST use int32 — int16 overflows on 255^2 and maps black to the wrong muscle.
    Transparent pixels are always `none`.
    """
    im = np.asarray(bgr)
    alpha = im[..., 3] if im.ndim == 3 and im.shape[-1] == 4 else None
    bgr3 = im[..., :3].astype(np.int32)
    R, G, B = bgr3[..., 2], bgr3[..., 1], bgr3[..., 0]
    opaque = (alpha > 10) if alpha is not None else (bgr3.max(-1) > 0)
    core = opaque & (G < 20) & (B < 20) & (R <= 22)
    index_mode = bool(opaque.any() and (core.sum() > 0.35 * max(int(opaque.sum()), 1)))
    if index_mode:
        ids = np.clip(np.rint(R.astype(np.float32)), 0, len(MUSCLE_NAMES) - 1).astype(np.uint8)
        ids[(G > 20) | (B > 20) | (R > 22)] = 0
    else:
        rgb = bgr3[..., ::-1]
        pal = PALETTE.astype(np.int32)
        d = ((rgb[..., None, :] - pal[None, None, :, :]) ** 2).sum(-1)
        ids = d.argmin(-1).astype(np.uint8)
    if alpha is not None:
        ids = ids.copy(); ids[alpha <= 10] = 0
    return ids


def write_mid_attribute(me, vert_id):
    """Write numeric id (R) + palette colour into mesh colour attributes `mid` and `mid_rgb`."""
    nv = len(vert_id)
    rgb = palette_rgb(vert_id).astype(np.float32) / 255.0
    rgba = np.concatenate([rgb, np.ones((nv, 1), np.float32)], 1).ravel()
    # encode id in R of `mid` as id/255
    mid = np.zeros((nv, 4), np.float32)
    mid[:, 0] = np.asarray(vert_id, np.float32) / 255.0
    mid[:, 3] = 1.0
    for name, col in (("mid", mid.ravel()), ("mid_rgb", rgba)):
        va = me.color_attributes.get(name)
        if va is None:
            va = me.color_attributes.new(name, "FLOAT_COLOR", "POINT")
        va.data.foreach_set("color", col)
    me.update()


def write_mid_target(me, vert_id, names, targets):
    """FLOAT point attribute `mid_tgt` = 1 on grade target muscles (drives method D)."""
    name_i = {n: i for i, n in enumerate(names)}
    tids = [name_i[n] for n in targets if n in name_i]
    fac = np.isin(np.asarray(vert_id), tids).astype(np.float32)
    col = np.zeros((len(fac), 4), np.float32); col[:, 0] = fac; col[:, 3] = 1.0
    va = me.color_attributes.get("mid_tgt")
    if va is None:
        va = me.color_attributes.new("mid_tgt", "FLOAT_COLOR", "POINT")
    va.data.foreach_set("color", col.ravel())
    me.update()
    return int(fac.sum())



def blur_mask(mask, adj, iters=7, core_hold=0.94, dilate=2):
    """Dilate a few rings then adjacency-blur. Interior stays near 1; borders feather."""
    w = np.asarray(mask, np.float32).copy()
    core = (w > 0.5).astype(np.float32)
    nv = int(w.shape[0])
    counts = np.array([len(a) for a in adj], np.int32)
    if nv == 0:
        return w
    parts = [a for a in adj if len(a)]
    idx = np.concatenate(parts) if parts else np.zeros(0, np.int32)
    owners = np.repeat(np.arange(nv, dtype=np.int32), counts)
    for _ in range(int(dilate)):
        mx = w.copy()
        if len(idx):
            np.maximum.at(mx, owners, w[idx])
        w = np.maximum(w, mx)
    # `core` stays the pre-dilate interior so only the original sheet is held at core_hold
    for _ in range(int(iters)):
        sums = np.zeros(nv, np.float32)
        if len(idx):
            np.add.at(sums, owners, w[idx])
        nbr = sums / np.maximum(counts.astype(np.float32), 1.0)
        w = 0.40 * w + 0.60 * nbr
        w = np.maximum(w, core * float(core_hold))
    return np.clip(w, 0.0, 1.0)


def write_glow_w(me, weights):
    """FLOAT_COLOR point attribute `glow_w` (R=G=B = weight) for the in-scene mix."""
    w = np.asarray(weights, np.float32)
    col = np.zeros((len(w), 4), np.float32)
    col[:, 0] = w; col[:, 1] = w; col[:, 2] = w; col[:, 3] = 1.0
    va = me.color_attributes.get("glow_w")
    if va is None:
        va = me.color_attributes.new("glow_w", "FLOAT_COLOR", "POINT")
    va.data.foreach_set("color", col.ravel())
    me.update()
    return float(w.mean()), int((w > 0.5).sum())


# OpenCV HSV (H 0-179). Sampled from demicstory crops 2026-11-10 (output/_refs_color/crop_demic_*.jpg).
# Shared fixed table: an exercise only names target_muscles (and may override a hue).
DEFAULT_HUES = {
    "lats":       {"h": 10,  "s": 200, "v": 242},
    "rhomboids":  {"h": 141, "s": 220, "v": 245},
    "mid_trap":   {"h": 141, "s": 220, "v": 245},
    "upper_trap": {"h": 29,  "s": 140, "v": 248},
    "rear_delt":  {"h": 162, "s": 240, "v": 235},
    "side_delt":  {"h": 10,  "s": 210, "v": 235},
    "front_delt": {"h": 10,  "s": 200, "v": 240},
    "biceps":     {"h": 95,  "s": 180, "v": 230},
    "triceps":    {"h": 22,  "s": 170, "v": 245},
    "forearm":    {"h": 29,  "s": 140, "v": 240},
    "pecs":       {"h": 66,  "s": 170, "v": 225},
    "abs":        {"h": 66,  "s": 160, "v": 220},
    "obliques":   {"h": 66,  "s": 150, "v": 210},
    "glutes":     {"h": 10,  "s": 190, "v": 230},
    "quads":      {"h": 95,  "s": 170, "v": 220},
    "hams":       {"h": 141, "s": 180, "v": 220},
    "calves":     {"h": 162, "s": 200, "v": 220},
    "erectors":   {"h": 29,  "s": 150, "v": 245},
}


def hsv_to_lin_rgb(h, s, v):
    """OpenCV HSV -> linear RGB triple for a Standard view-transform emission/tint."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(float(h) / 180.0, float(s) / 255.0, float(v) / 255.0)

    def _lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return (_lin(r), _lin(g), _lin(b))


def _fill_interior_holes(inside, counts, idx, owners, rounds=4, frac=0.66):
    """Fill small unclassified speckles surrounded by the ID. Does not grow the silhouette."""
    ins = np.asarray(inside, dtype=bool).copy()
    nv = int(ins.shape[0])
    need = np.maximum(np.ceil(counts.astype(np.float32) * float(frac)).astype(np.int32), 1)
    for _ in range(int(rounds)):
        votes = np.zeros(nv, np.int32)
        if len(idx):
            np.add.at(votes, owners, ins[idx].astype(np.int32))
        cand = (~ins) & (counts > 0) & (votes >= need)
        if not cand.any():
            break
        ins[cand] = True
    return ins


def _pack_adj(adj):
    nv = len(adj)
    counts = np.array([len(a) for a in adj], np.int32)
    parts = [a for a in adj if len(a)]
    idx = np.concatenate(parts) if parts else np.zeros(0, np.int32)
    owners = np.repeat(np.arange(nv, dtype=np.int32), counts)
    return counts, idx, owners


def weld_adj(adj, pos, eps=4e-4):
    """Connect UV-split copies of the same surface point so a muscle is one sheet."""
    pos = np.asarray(pos, np.float32)
    nv = len(adj)
    keys = np.round(pos / float(eps)).astype(np.int64)
    # 64-bit spatial hash
    h = keys[:, 0] * 73856093 + keys[:, 1] * 19349663 + keys[:, 2] * 83492791
    order = np.argsort(h)
    hs = h[order]
    extra = [[] for _ in range(nv)]
    i = 0
    while i < nv:
        j = i + 1
        while j < nv and hs[j] == hs[i]:
            j += 1
        if j - i > 1:
            grp = order[i:j]
            for a in grp:
                for b in grp:
                    if a != b:
                        extra[int(a)].append(int(b))
        i = j
    out = []
    for k in range(nv):
        if extra[k]:
            out.append(np.unique(np.concatenate([np.asarray(adj[k], np.int32), np.asarray(extra[k], np.int32)])))
        else:
            out.append(np.asarray(adj[k], np.int32))
    return out


def _dilate_mask(mask, counts, idx, owners, rings):
    w = np.asarray(mask, np.float32)
    for _ in range(int(max(rings, 0))):
        mx = w.copy()
        if len(idx):
            np.maximum.at(mx, owners, w[idx])
        w = np.maximum(w, mx)
    return w > 0.5


def _erode_mask(mask, counts, idx, owners, rings):
    return ~_dilate_mask(~np.asarray(mask, dtype=bool), counts, idx, owners, rings)


def _laplacian(w, counts, idx, owners, iters, retain=None):
    w = np.asarray(w, np.float32).copy()
    nv = int(w.shape[0])
    for _ in range(int(max(iters, 0))):
        sums = np.zeros(nv, np.float32)
        if len(idx):
            np.add.at(sums, owners, w[idx])
        nbr = sums / np.maximum(counts.astype(np.float32), 1.0)
        w = 0.35 * w + 0.65 * nbr
        if retain is not None:
            w[~retain] = 0.0
    return w


def _components(mask, adj):
    inside = np.asarray(mask, dtype=bool)
    nv = int(inside.shape[0])
    labels = np.full(nv, -1, np.int32)
    sizes = []
    cid = 0
    for seed in np.flatnonzero(inside):
        if labels[seed] >= 0:
            continue
        stack = [int(seed)]
        labels[seed] = cid
        n = 0
        while stack:
            i = stack.pop()
            n += 1
            for j in adj[i]:
                j = int(j)
                if inside[j] and labels[j] < 0:
                    labels[j] = cid
                    stack.append(j)
        sizes.append(n)
        cid += 1
    return labels, np.asarray(sizes, np.int32)



def _convex_hull_2d(pts):
    """Monotone-chain convex hull of (N,2) points -> (M,2) CCW ring."""
    pts = np.unique(np.asarray(pts, np.float32), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for raw in pts:
        q = (float(raw[0]), float(raw[1]))
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for raw in pts[::-1]:
        q = (float(raw[0]), float(raw[1]))
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1] + upper[:-1], np.float32)


def _inside_convex(xy, hull, eps=1e-7):
    if hull is None or len(hull) < 3:
        return np.zeros(len(xy), dtype=bool)
    inside = np.ones(len(xy), dtype=bool)
    n = len(hull)
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        cross = (b[0] - a[0]) * (xy[:, 1] - a[1]) - (b[1] - a[1]) * (xy[:, 0] - a[0])
        inside &= cross >= -eps
    return inside


def _hull_fill_xy(mask, pos, allow):
    """Fill the X-Y convex hull of `mask` (smooth rounded silhouette), clipped to allow."""
    idx = np.flatnonzero(mask)
    if len(idx) < 8:
        return np.asarray(mask, dtype=bool)
    pts = np.asarray(pos, np.float32)[idx][:, [0, 1]]
    hull = _convex_hull_2d(pts)
    if len(hull) < 3:
        return np.asarray(mask, dtype=bool)
    cand = np.flatnonzero(allow)
    if cand.size == 0:
        return np.asarray(mask, dtype=bool)
    xy = np.asarray(pos, np.float32)[cand][:, [0, 1]]
    out = np.asarray(mask, dtype=bool).copy()
    out[cand[_inside_convex(xy, hull)]] = True
    return out


def _feather_solid(solid, counts, idx, owners, fade_rings=3, power=1.0, radial=False, labels=None, plateau=0.35):
    """Weight from graph-distance to the region exterior.

    Default (radial=False): interior stays 1; only the outer fade_rings hops drop off.
    radial=True: full distance-to-boundary / max, cosine-smoothed and raised to
    `power`, so each muscle is an airbrushed sheet (hot core, soft transparent edge).
    """
    filled = np.asarray(solid, dtype=bool)
    nv = int(filled.shape[0])
    w = np.zeros(nv, np.float32)
    if not filled.any():
        return w
    d = np.zeros(nv, np.int32)
    reached = ~filled
    for hop in range(1, 80):
        flag = np.zeros(nv, np.int8)
        if len(idx):
            np.maximum.at(flag, owners, reached[idx].astype(np.int8))
        nxt = (flag > 0) & (~reached) & filled
        if not nxt.any():
            break
        d[nxt] = hop
        reached |= nxt
    if radial:
        # Per-component dmax so a thin crescent arm (upper lat) is not crushed
        # by a thick belly's radius. Inner `plateau` of each component stays hot.
        if labels is None:
            groups = [filled]
        else:
            groups = [labels == int(c) for c in np.unique(labels) if int(c) >= 0]
        for m in groups:
            if not np.any(m):
                continue
            dmax = max(float(d[m].max()), 1.0)
            ref = max(dmax * (1.0 - float(plateau)), 1.0)
            t = np.clip(d[m].astype(np.float32) / ref, 0.0, 1.0)
            w[m] = np.sin(0.5 * np.pi * t)
    else:
        fade = max(float(fade_rings), 1e-4)
        raw = d.astype(np.float32) / fade
        w[filled] = np.clip(raw[filled], 0.0, 1.0)
    if float(power) != 1.0:
        pos = w > 0
        w[pos] = np.power(w[pos], float(power))
    return np.clip(w, 0.0, 1.0)


def anatomy_allow(pos, vert_id, names):
    """Rest-pose metres (scaled to the mesh). Clips purple below the scapular spine."""
    pos = np.asarray(pos, np.float32)
    y, x, z = pos[:, 1], pos[:, 0], pos[:, 2]
    ax = np.abs(x)
    scale = float(y.max()) / 1.8 if float(y.max()) > 0.5 else 1.0
    name_i = {n: i for i, n in enumerate(names)}
    tid = np.asarray(vert_id)
    ut = tid == name_i["upper_trap"] if "upper_trap" in name_i else np.zeros(len(tid), dtype=bool)
    rd = tid == name_i["rear_delt"] if "rear_delt" in name_i else np.zeros(len(tid), dtype=bool)
    spine_y = 1.45 * scale
    if rd.any():
        spine_y = float(np.percentile(y[rd], 25))
    if ut.any():
        spine_y = min(spine_y, float(np.percentile(y[ut], 5)) - 0.02 * scale)
    back = z < 0.02 * scale
    masks = {
        "lats": (z < 0.10 * scale) & (y >= 0.97 * scale) & (y <= max(spine_y, 1.43 * scale)) & (ax >= 0.030 * scale) & (ax <= 0.28 * scale),
        "rhomboids": back & (y >= 1.16 * scale) & (y <= spine_y) & (ax <= 0.145 * scale),
        "mid_trap": back & (y >= 1.30 * scale) & (y <= spine_y) & (ax <= 0.090 * scale),
        "rear_delt": (z < 0.040 * scale) & (y >= spine_y - 0.050 * scale) & (y <= spine_y + 0.13 * scale)
                     & (ax >= 0.095 * scale) & (ax <= 0.240 * scale),
    }
    return masks, {"spine_y": round(spine_y, 4), "scale": round(scale, 4)}


def clean_muscle_weight(seed, adj, allow, fade_rings=3, smooth_iters=40, thresh=0.32,
                        close_rings=5, keep=2, min_size=80, power=1.0, radial=False,
                        pos=None, hull_fill=True):
    """Rebuild one (or L/R) smooth solid region from a jagged ID seed, then short-feather.

    Morphological close fills island holes; Laplacian smoothing rounds torn atlas edges;
    threshold + largest components give a single anatomical silhouette per side.
    Weight never leaves `allow` (anatomical bounds + not a neighbour muscle).
    """
    seed = np.asarray(seed, dtype=bool)
    allow = np.asarray(allow, dtype=bool)
    nv = int(seed.shape[0])
    w0 = np.zeros(nv, np.float32)
    seed = seed & allow
    if not seed.any():
        return w0, {"solid": 0, "cc": 0}
    counts, idx, owners = _pack_adj(adj)
    solid = _fill_interior_holes(seed, counts, idx, owners, rounds=5, frac=0.60)
    if int(close_rings) > 0:
        grown = _dilate_mask(solid, counts, idx, owners, close_rings) & allow
        solid = _erode_mask(grown, counts, idx, owners, close_rings) & allow
        solid |= seed
    field = solid.astype(np.float32)
    field = _laplacian(field, counts, idx, owners, int(smooth_iters), retain=allow)
    solid = (field >= float(thresh)) & allow
    labels, sizes = _components(solid, adj)
    kept = 0
    if len(sizes):
        order = np.argsort(-sizes)
        pick = []
        for c in order:
            if sizes[c] < int(min_size):
                continue
            pick.append(int(c))
            if len(pick) >= int(keep):
                break
        keep_m = np.isin(labels, np.array(pick, np.int32)) if pick else np.zeros(nv, dtype=bool)
        kept = len(pick)
    else:
        keep_m = np.zeros(nv, dtype=bool)
    # Convex-hull fill per component (X-Y back plane) -> one smooth wing / diamond / oval
    # instead of jagged atlas islands, still clipped to `allow`.
    if hull_fill and pos is not None and keep_m.any():
        xs = np.abs(np.asarray(pos, np.float32)[keep_m, 0])
        # Medial mass (rhomboid/mid-trap) -> one hull = one diamond across the spine.
        # Lateral muscles (lats, rear delt) stay per-component so L/R do not merge.
        if float(np.percentile(xs, 75)) < 0.16:
            keep_m = _hull_fill_xy(keep_m, pos, allow)
        else:
            labs_h, _ = _components(keep_m, adj)
            filled = np.zeros_like(keep_m)
            for c in np.unique(labs_h):
                if int(c) < 0:
                    continue
                filled |= _hull_fill_xy(labs_h == int(c), pos, allow)
            keep_m = filled
    # one extra ring inside allow so the silhouette reaches the visual belly edge
    keep_m = _dilate_mask(keep_m, counts, idx, owners, 1) & allow
    labels, _sz = _components(keep_m, adj)
    w = _feather_solid(keep_m, counts, idx, owners, fade_rings=fade_rings, power=power,
                       radial=radial, labels=labels)
    if radial:
        w = _laplacian(w, counts, idx, owners, 8, retain=allow)
    return w, {"solid": int(keep_m.sum()), "cc": kept}


def center_weight(mask, adj, erode_rings=0, power=1.0, fade_rings=3, blur_iters=2):
    """Fallback feather: graph-distance from the ID exterior with a short rim fade."""
    inside0 = np.asarray(mask, dtype=bool)
    nv = int(inside0.shape[0])
    w = np.zeros(nv, np.float32)
    if not inside0.any():
        return w
    counts, idx, owners = _pack_adj(adj)
    filled = _fill_interior_holes(inside0, counts, idx, owners)
    if int(erode_rings) > 0:
        filled = _erode_mask(filled, counts, idx, owners, erode_rings)
    w = _feather_solid(filled, counts, idx, owners, fade_rings=fade_rings, power=power)
    for _ in range(int(blur_iters)):
        w = _laplacian(w, counts, idx, owners, 1, retain=filled)
    return np.clip(w, 0.0, 1.0)


def write_glow_attrs(me, weights, rgb):
    """Write `glow_w` and per-vertex linear `glow_rgb` (baked once on the rest mesh)."""
    mean, n = write_glow_w(me, weights)
    col = np.zeros((len(weights), 4), np.float32)
    col[:, :3] = np.asarray(rgb, np.float32)
    col[:, 3] = 1.0
    va = me.color_attributes.get("glow_rgb")
    if va is None:
        va = me.color_attributes.new("glow_rgb", "FLOAT_COLOR", "POINT")
    va.data.foreach_set("color", col.ravel())
    me.update()
    return mean, n


def bake_target_glow(vert_id, names, targets, hues, strengths, adj, exclude=None, shorts=None,
                     erode_rings=0, power=1.0, fade_rings=3, pos=None,
                     smooth_iters=40, solid_thresh=0.28, close_rings=5, radial=False):
    """Once-per-mesh bake: clean solid region per hue-group + that group's linear RGB."""
    nv = len(vert_id)
    name_i = {n: i for i, n in enumerate(names)}
    tid = np.asarray(vert_id)
    gw = np.zeros(nv, np.float32)
    rgb = np.zeros((nv, 3), np.float32)
    exclude = np.zeros(nv, dtype=bool) if exclude is None else np.asarray(exclude, dtype=bool)
    if shorts is None:
        sh = np.zeros(nv, dtype=bool)
    else:
        sh = np.asarray(shorts)
        sh = sh if sh.dtype == bool else (sh > 0.5)
    stats = {}
    hues = hues or {}
    strengths = strengths or {}
    bounds, meta = ({}, {})
    if pos is not None:
        bounds, meta = anatomy_allow(pos, tid, names)
        stats["_clip"] = meta
    forbid_ids = [name_i[n] for n in ("upper_trap", "front_delt", "side_delt", "erectors",
                                      "pecs", "abs", "obliques") if n in name_i]
    # group targets that share a hue so mid-back (rhomboids+mid_trap) becomes one shape
    groups = []
    used = set()
    for name in targets:
        if name in used or name not in name_i:
            continue
        spec = hues.get(name) or DEFAULT_HUES.get(name)
        if not spec:
            continue
        key = (int(spec["h"]), int(spec["s"]), int(spec["v"]))
        members = []
        for n2 in targets:
            if n2 in used or n2 not in name_i:
                continue
            s2 = hues.get(n2) or DEFAULT_HUES.get(n2)
            if s2 and (int(s2["h"]), int(s2["s"]), int(s2["v"])) == key:
                members.append(n2)
                used.add(n2)
        groups.append((members, spec))
    use_adj = weld_adj(adj, pos, eps=4e-4) if pos is not None else adj
    for members, spec in groups:
        seed = np.zeros(nv, dtype=bool)
        allow = np.zeros(nv, dtype=bool)
        for name in members:
            idm = tid == name_i[name]
            clip = bounds[name] if name in bounds else np.ones(nv, dtype=bool)
            seed |= idm & clip
            allow |= clip
        allow &= ~sh & ~exclude
        skip_forbid = set()
        if any(n in ("rhomboids", "mid_trap") for n in members):
            # midline of the diamond sits on erector verts — let the hull fill across the spine
            skip_forbid.add("erectors")
        for fid, fname in zip(forbid_ids, [n for n in ("upper_trap", "front_delt", "side_delt", "erectors",
                                                       "pecs", "abs", "obliques") if n in name_i]):
            if fname in skip_forbid:
                continue
            allow &= tid != fid
        for name in targets:
            if name not in members and name in name_i:
                # Only reserve verts the other group will actually paint. A rhomboid
                # atlas island that sits too far lateral (outside its clip) can be
                # claimed by the lat hull so the wing and diamond do not leave a clay gap.
                other_clip = bounds[name] if name in bounds else np.ones(nv, dtype=bool)
                allow &= ~((tid == name_i[name]) & other_clip)
        seed &= ~sh & ~exclude & allow
        stren = float(np.mean([float(strengths.get(n, 1.0)) for n in members]))
        wt, info = clean_muscle_weight(
            seed, use_adj, allow,
            fade_rings=fade_rings, smooth_iters=smooth_iters,
            thresh=solid_thresh, close_rings=close_rings,
            keep=2, min_size=80, power=power, radial=radial,
            pos=pos, hull_fill=True,
        )
        wt = wt * stren
        col = hsv_to_lin_rgb(spec["h"], spec["s"], spec["v"])
        better = wt > gw
        rgb[better] = col
        gw = np.maximum(gw, wt)
        for name in members:
            stats[name] = int(((tid == name_i[name]) & (wt > 0.5)).sum())
        stats["+".join(members) + "_solid"] = info.get("solid", 0)
        stats["+".join(members) + "_cc"] = info.get("cc", 0)
    gw[sh] = 0.0
    gw[exclude] = 0.0
    return gw, rgb, stats



def cache_path(cache_dir, key):
    return os.path.join(cache_dir, f"muscle_ids_{key}.npz")


def load_or_build(glb_path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    key = asset_key(glb_path)
    path = cache_path(cache_dir, key)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {"vert_id": z["vert_id"], "names": [str(s) for s in z["names"]], "palette": z["palette"],
                "counts": json.loads(str(z["counts_json"])), "nv": int(z["nv"]), "key": key, "src": glb_path,
                "cache": path, "hit": True}
    built = build_from_glb(glb_path)
    np.savez_compressed(path, vert_id=built["vert_id"], names=np.array(built["names"]),
                        palette=built["palette"], counts_json=json.dumps(built["counts"]), nv=built["nv"])
    built["cache"] = path
    built["hit"] = False
    return built


def render_label_preview(pos, vert_id, out_path, size=1400):
    """Orthographic back + front preview of rest-pose IDs (no Blender)."""
    import cv2
    ids = np.asarray(vert_id)
    rgb = palette_rgb(ids)
    y, x, z = pos[:, 1], pos[:, 0], pos[:, 2]
    canv = np.full((size, size * 2 + 80, 3), 18, np.uint8)
    def blit(xs, ys, zs, xoff, nearer_is_greater):
        u = ((xs - xs.min()) / max(float(np.ptp(xs)), 1e-6) * (size - 80) + 40).astype(np.int32)
        v = ((1.0 - (ys - ys.min()) / max(float(np.ptp(ys)), 1e-6)) * (size - 80) + 40).astype(np.int32)
        depth = np.full((size, size), -1e9, np.float32)
        key = zs if nearer_is_greater else -zs
        for i in range(len(xs)):
            uu, vv = int(u[i]), int(v[i])
            if not (1 <= uu < size - 1 and 1 <= vv < size - 1):
                continue
            if key[i] >= depth[vv, uu]:
                depth[vv, uu] = key[i]
                canv[vv-1:vv+2, xoff+uu-1:xoff+uu+2] = rgb[i][::-1]
    blit(x, y, z, 0, nearer_is_greater=False)          # back: smaller z is nearer
    blit(-x, y, z, size + 80, nearer_is_greater=True)  # front: larger z is nearer
    # legend
    for i, name in enumerate(MUSCLE_NAMES):
        yy = 20 + i * 22
        if yy >= size - 10: break
        cv2.rectangle(canv, (size + 8, yy - 12), (size + 70, yy + 4), PALETTE[i][::-1].tolist(), -1)
        cv2.putText(canv, name, (size + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canv, "rest back", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canv, "rest front", (size + 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canv)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--preview", default="")
    a = ap.parse_args()
    cache = a.cache or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output/_rig3d/cache")
    built = load_or_build(a.glb, cache)
    print("cache", "HIT" if built.get("hit") else "MISS", built.get("cache"))
    print("nv", built["nv"], "counts", built["counts"])
    if a.preview:
        M = _read_glb(a.glb)
        render_label_preview(M["pos"], built["vert_id"], a.preview)
        print("preview", a.preview)
