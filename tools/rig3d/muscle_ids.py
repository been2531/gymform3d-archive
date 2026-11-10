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
    h.update(b"muscle_ids_v4")
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
    # Proximal Arm weights paint the lat insertion / armpit — those fall through to torso.
    if "arm" in b and "fore" not in b and "shoulder" not in b and "upleg" not in b:
        if ax >= 0.22 or y < 1.16:
            return "triceps" if back else "biceps"
    if "shoulder" in b or (1.30 <= y <= 1.58 and ax >= 0.14):
        if back and ax >= 0.11: return "rear_delt"
        if front and ax >= 0.11: return "front_delt"
        if ax >= 0.11: return "side_delt"
    # Exclusive back-torso partition so target muscles (lats/rhomb/mid-trap/rear-delt) meet.
    if back and 0.80 <= y <= 1.68:
        if 0.80 <= y < 1.00 and ax >= 0.045:
            return "glutes"
        if ax < 0.058:
            if y < 1.34: return "erectors"
            if y < 1.54: return "mid_trap"
            return "upper_trap"
        if ax < 0.128:
            if y < 1.16: return "lats"
            if y < 1.50: return "rhomboids"
            return "upper_trap"
        # lateral back
        if y < 1.34: return "lats"
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
