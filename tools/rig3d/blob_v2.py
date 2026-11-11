"""Symmetric, muscle-shaped 3D glow helpers (blob v2) plus cheaper alts.

The v1 blob mirrored anchors across a camera-derived 'side' axis, so a rear
3/4 view painted only the near flank. v2 builds an anatomic left-right from
the shoulder-band PCA, mirrors across that midline, and uses tapered
elliptical capsules oriented along each muscle instead of round spots.
"""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Euler, Vector

from eval_muscle_colour import (  # noqa: E402
    GREY,
    _all_world,
    _edges_np,
    _world_co_no,
    estimate_torso_frame,
)

MUSCLE_ORDER = ("lats", "mid_back", "rear_delt", "biceps")

# Warmer and more saturated than v1 (peach / pink / purple / yellow).
COL = {
    "lats": np.array([1.00, 0.34, 0.08], np.float32),
    "mid_back": np.array([1.00, 0.22, 0.52], np.float32),
    "rear_delt": np.array([0.55, 0.16, 1.00], np.float32),
    "biceps": np.array([1.00, 0.72, 0.04], np.float32),
}

EM_STR = 0.55
ROUGH = 0.93
SPEC = 0.06


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / np.maximum(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_range(x, lo, hi, margin):
    """1 inside [lo, hi], fades to 0 over `margin` on each side."""
    return smoothstep(lo - margin, lo, x) * (1.0 - smoothstep(hi, hi + margin, x))


def clear_to_rest_pose(arms):
    for arm in arms:
        if getattr(arm, "animation_data", None):
            arm.animation_data_clear()
    sc = bpy.context.scene
    if getattr(sc, "animation_data", None):
        sc.animation_data_clear()
    for arm in arms:
        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        try:
            bpy.ops.object.mode_set(mode="POSE")
            bpy.ops.pose.select_all(action="SELECT")
            bpy.ops.pose.transforms_clear()
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
    try:
        bpy.context.scene.frame_set(0)
    except Exception:
        pass
    bpy.context.view_layer.update()


def estimate_anatomic_frame(meshes, cam_dir):
    """Left-right from shoulder-band PCA; back oriented toward the camera.

    Mirroring across `side` is then a true bilateral flip, not a camera flip.
    """
    hip, spine, _side0, _back0, slen, info = estimate_torso_frame(meshes, cam_dir)
    chunks = _all_world(meshes)
    all_co = np.concatenate([w for _, w, _ in chunks], 0)
    rel = all_co - hip
    t = (rel @ spine) / max(slen, 1e-8)
    band = (t > 0.48) & (t < 0.92)
    if int(band.sum()) < 40:
        band = (t > 0.30) & (t < 0.95)
    pts = rel[band]
    pts_plane = pts - np.outer(pts @ spine, spine)
    if len(pts_plane) >= 8:
        cov = np.cov(pts_plane.T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        side = evecs[:, order[0]].astype(np.float64)
        info["side_pca_var"] = [float(evals[i]) for i in order]
    else:
        side = np.asarray(_side0, np.float64)
        info["side_pca_var"] = None
    side = side - (side @ spine) * spine
    sn = float(np.linalg.norm(side)) or 1.0
    side = side / sn
    back = np.cross(side, spine)
    back = back / (float(np.linalg.norm(back)) or 1.0)
    cam = np.asarray(cam_dir, dtype=np.float64)
    if float(back @ cam) < 0.0:
        back = -back
        side = -side
    # Re-orthonormalize, right-handed (side, back, spine).
    side = np.cross(spine, back)
    side = side / (float(np.linalg.norm(side)) or 1.0)
    back = np.cross(side, spine)
    back = back / (float(np.linalg.norm(back)) or 1.0)
    torso = (t > 0.22) & (t < 0.88)
    if torso.any():
        lat = (all_co[torso] - hip) @ side
        hip = hip + float(np.median(lat)) * side
    info.update({
        "hip": hip.tolist(),
        "spine": spine.tolist(),
        "side": side.tolist(),
        "back": back.tolist(),
        "frame_mode": "anatomic_pca",
        "back_dot_cam": float(back @ cam),
        "spine_len": float(slen),
    })
    return hip, spine, side, back, slen, info




def surface_extents(meshes, hip, spine, side, back, slen):
    """Metre offsets from the hip-neck line to back / front / side / outer-arm skin."""
    chunks = _all_world(meshes)
    all_co = np.concatenate([w for _, w, _ in chunks], 0)
    rel = all_co - hip
    t = (rel @ spine) / max(slen, 1e-8)
    torso = (t > 0.28) & (t < 0.82)
    if int(torso.sum()) < 80:
        torso = (t > 0.18) & (t < 0.92)
    bck_vals = rel[torso] @ back
    lat_abs = np.abs(rel[torso] @ side)
    backish = bck_vals >= np.percentile(bck_vals, 50)
    side_from_back = float(np.percentile(lat_abs[backish], 86)) if backish.any() else float(np.percentile(lat_abs, 75))
    ext = {
        "back_surf": float(np.percentile(bck_vals, 93)),
        "front_surf": float(np.percentile(bck_vals, 7)),
        "side_surf": side_from_back,
        "arm_lat": float(np.percentile(lat_abs, 94)),
        "back_p50": float(np.median(bck_vals)),
        "n": int(torso.sum()),
    }
    if ext["back_surf"] < 0.04:
        ext["back_surf"] = max(abs(ext["back_surf"]), 0.10)
    if ext["side_surf"] < 0.08:
        ext["side_surf"] = 0.18
    if ext["arm_lat"] < ext["side_surf"]:
        ext["arm_lat"] = ext["side_surf"] * 1.25
    return ext

def pt_fn(hip, spine, side, back, slen, ext):
    ss = float(ext["side_surf"])
    bs = float(ext["back_surf"])
    fs = float(abs(ext["front_surf"])) or bs

    def pt(t, lat_frac, bck_frac):
        bck_m = (bck_frac * bs) if bck_frac >= 0.0 else (bck_frac * fs)
        return hip + (t * slen) * spine + (lat_frac * ss) * side + bck_m * back

    return pt, ss, bs, fs

def mirror_across_midline(p, hip, side):
    p = np.asarray(p, np.float64)
    lat = float((p - hip) @ side)
    return p - (2.0 * lat) * side


def tapered_capsule_weight(world, a, b, r_maj_a, r_maj_b, r_min_a, r_min_b, thick_dir,
                           inner=0.50):
    """Soft elliptical falloff around segment AB, radii lerped from A to B."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ab = b - a
    length = float(np.linalg.norm(ab))
    n = len(world)
    if length < 1e-8:
        return np.zeros(n, np.float32)
    u = ab / length
    w = np.asarray(thick_dir, np.float64)
    w = w - (w @ u) * u
    wn = float(np.linalg.norm(w))
    if wn < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        w = tmp - (tmp @ u) * u
        wn = float(np.linalg.norm(w)) or 1.0
    w = w / wn
    v = np.cross(u, w)
    v = v / (float(np.linalg.norm(v)) or 1.0)

    rel = world - a
    tt = rel @ u
    t_clamped = np.clip(tt, 0.0, length)
    closest = a + t_clamped[:, None] * u
    d = world - closest
    dv = d @ v
    dw = d @ w
    frac = t_clamped / length
    r_maj = r_maj_a + (r_maj_b - r_maj_a) * frac
    r_min = r_min_a + (r_min_b - r_min_a) * frac
    r = np.sqrt((dv / np.maximum(r_maj, 1e-6)) ** 2 + (dw / np.maximum(r_min, 1e-6)) ** 2)
    return (1.0 - smoothstep(inner, 1.0, r)).astype(np.float32)





def muscle_capsules(hip, spine, side, back, slen, ext):
    """Tapered capsules. Placement uses the v1 metre frame (s = slen/0.55)
    that already reached this mannequin's skin; shapes are elongated and
    mirrored across the anatomic midline."""
    s = float(slen / 0.55)

    def pt(t, lat, bck):
        return hip + (t * slen) * spine + (lat * s) * side + (bck * s) * back

    caps = {name: [] for name in MUSCLE_ORDER}

    def add(name, a, b, r_maj_a, r_maj_b, r_min_a, r_min_b, thick, mirror=True):
        rec = {
            "a": np.asarray(a, np.float64),
            "b": np.asarray(b, np.float64),
            "r_maj_a": float(r_maj_a),
            "r_maj_b": float(r_maj_b),
            "r_min_a": float(r_min_a),
            "r_min_b": float(r_min_b),
            "thick": np.asarray(thick, np.float64),
        }
        caps[name].append(rec)
        if mirror:
            caps[name].append({
                "a": mirror_across_midline(a, hip, side),
                "b": mirror_across_midline(b, hip, side),
                "r_maj_a": rec["r_maj_a"],
                "r_maj_b": rec["r_maj_b"],
                "r_min_a": rec["r_min_a"],
                "r_min_b": rec["r_min_b"],
                "thick": rec["thick"],
            })

    # Lat: long fan, armpit -> lower back. Thickness matches v1's ~0.20 m reach.
    add("lats",
        pt(0.64, 0.16, 0.05), pt(0.26, 0.11, 0.08),
        0.11 * s, 0.16 * s, 0.15 * s, 0.16 * s, back, True)
    add("lats",
        pt(0.54, 0.13, 0.07), pt(0.32, 0.09, 0.09),
        0.10 * s, 0.13 * s, 0.14 * s, 0.15 * s, back, True)
    # Mid-back diamond between the scapulae.
    add("mid_back",
        pt(0.54, 0.00, 0.08), pt(0.80, 0.00, 0.075),
        0.12 * s, 0.11 * s, 0.15 * s, 0.14 * s, back, False)
    add("mid_back",
        pt(0.67, -0.07, 0.08), pt(0.67, 0.07, 0.08),
        0.10 * s, 0.10 * s, 0.14 * s, 0.14 * s, back, False)
    # Rear delt cap.
    add("rear_delt",
        pt(0.80, 0.19, 0.055), pt(0.86, 0.23, 0.040),
        0.080 * s, 0.070 * s, 0.090 * s, 0.080 * s, back, True)
    # Biceps along the front of the upper arm.
    add("biceps",
        pt(0.62, 0.24, -0.050), pt(0.46, 0.26, -0.075),
        0.065 * s, 0.060 * s, 0.080 * s, 0.075 * s, -back, True)
    return caps


def muscle_strokes(hip, spine, side, back, slen, ext):
    """Thinner painted-stroke polylines (simulate a one-time vertex paint)."""
    s = float(slen / 0.55)

    def pt(t, lat, bck):
        return hip + (t * slen) * spine + (lat * s) * side + (bck * s) * back

    strokes = {name: [] for name in MUSCLE_ORDER}

    def add_poly(name, points, radius, thick, mirror=True):
        pts = [np.asarray(p, np.float64) for p in points]
        strokes[name].append({"points": pts, "radius": float(radius), "thick": np.asarray(thick, np.float64)})
        if mirror:
            mids = [mirror_across_midline(p, hip, side) for p in pts]
            strokes[name].append({"points": mids, "radius": float(radius), "thick": np.asarray(thick, np.float64)})

    add_poly("lats",
             [pt(0.66, 0.17, 0.05), pt(0.52, 0.15, 0.065),
              pt(0.38, 0.12, 0.08), pt(0.24, 0.10, 0.085)],
             0.11 * s, back, True)
    add_poly("lats",
             [pt(0.58, 0.13, 0.07), pt(0.42, 0.10, 0.085),
              pt(0.28, 0.08, 0.09)],
             0.10 * s, back, True)
    add_poly("mid_back",
             [pt(0.54, 0.00, 0.085), pt(0.67, 0.065, 0.08),
              pt(0.80, 0.00, 0.075), pt(0.67, -0.065, 0.08),
              pt(0.54, 0.00, 0.085)],
             0.095 * s, back, False)
    add_poly("rear_delt",
             [pt(0.79, 0.18, 0.055), pt(0.83, 0.21, 0.045),
              pt(0.86, 0.235, 0.035)],
             0.075 * s, back, True)
    add_poly("biceps",
             [pt(0.64, 0.235, -0.045), pt(0.54, 0.25, -0.062),
              pt(0.46, 0.26, -0.075)],
             0.065 * s, -back, True)
    return strokes

def polyline_weight(world, points, radius, thick_dir, inner=0.50):
    acc = np.zeros(len(world), np.float32)
    for a, b in zip(points[:-1], points[1:]):
        acc = np.maximum(acc, tapered_capsule_weight(
            world, a, b, radius, radius, radius * 0.72, radius * 0.72, thick_dir, inner=inner
        ))
    return acc





def muscle_gates(world, normals, hip, spine, side, back, slen, ext):
    """Soft anatomical windows + hand/forearm/hip suppressors."""
    s = float(slen / 0.55)
    rel = world - hip
    t = (rel @ spine) / max(slen, 1e-8)
    lat = rel @ side
    bck = rel @ back
    abs_lat = np.abs(lat)
    nb = normals @ back if normals is not None and len(normals) == len(world) else np.zeros(len(world))

    hand = soft_range(t, -2.0, 0.34, 0.05) * soft_range(abs_lat, 0.20 * s, 8.0, 0.03 * s)
    hip_keep = smoothstep(0.20, 0.28, t)
    forearm = soft_range(t, -2.0, 0.38, 0.05) * soft_range(abs_lat, 0.18 * s, 8.0, 0.03 * s)

    lats = (soft_range(t, 0.18, 0.74, 0.06)
            * soft_range(abs_lat, 0.055 * s, 0.22 * s, 0.03 * s)
            * smoothstep(-0.04 * s, 0.01 * s, bck)
            * (1.0 - hand) * hip_keep)
    mid = (soft_range(t, 0.46, 0.92, 0.05)
           * (1.0 - smoothstep(0.14 * s, 0.20 * s, abs_lat))
           * smoothstep(-0.04 * s, 0.02 * s, bck)
           * (1.0 - hand))
    delt = (soft_range(t, 0.68, 0.98, 0.05)
            * soft_range(abs_lat, 0.12 * s, 0.30 * s, 0.03 * s)
            * smoothstep(-0.06 * s, 0.02 * s, bck)
            * (1.0 - hand))
    bic = (soft_range(t, 0.40, 0.76, 0.05)
           * soft_range(abs_lat, 0.16 * s, 0.36 * s, 0.03 * s)
           * (1.0 - smoothstep(0.02 * s, 0.07 * s, bck))
           * (1.0 - forearm))
    return {"lats": lats.astype(np.float32), "mid_back": mid.astype(np.float32),
            "rear_delt": delt.astype(np.float32), "biceps": bic.astype(np.float32),
            "t": t, "lat": lat, "bck": bck, "nb": nb}

def weights_from_capsules(world, normals, caps, hip, spine, side, back, slen, ext):
    gates = muscle_gates(world, normals, hip, spine, side, back, slen, ext)
    w = np.zeros((len(world), len(MUSCLE_ORDER)), np.float32)
    for i, name in enumerate(MUSCLE_ORDER):
        acc = np.zeros(len(world), np.float32)
        for c in caps[name]:
            acc = np.maximum(acc, tapered_capsule_weight(
                world, c["a"], c["b"], c["r_maj_a"], c["r_maj_b"],
                c["r_min_a"], c["r_min_b"], c["thick"],
            ))
        w[:, i] = acc * gates[name]
    return w, gates


def weights_from_strokes(world, normals, strokes, hip, spine, side, back, slen, ext):
    gates = muscle_gates(world, normals, hip, spine, side, back, slen, ext)
    w = np.zeros((len(world), len(MUSCLE_ORDER)), np.float32)
    for i, name in enumerate(MUSCLE_ORDER):
        acc = np.zeros(len(world), np.float32)
        for st in strokes[name]:
            acc = np.maximum(acc, polyline_weight(world, st["points"], st["radius"], st["thick"]))
        w[:, i] = acc * gates[name]
    return w, gates

def write_mcol_soft(obj, weights, colours=None):
    """Blend overlapping muscles instead of winner-takes-most hard seams."""
    pal = colours or COL
    n = len(weights)
    acc = np.zeros((n, 3), np.float32)
    for i, name in enumerate(MUSCLE_ORDER):
        acc += weights[:, i:i + 1] * pal[name]
    total = weights.sum(axis=1, keepdims=True)
    col = np.zeros((n, 4), np.float32)
    col[:, 3] = 1.0
    col[:, :3] = acc / np.maximum(total, 1.0)
    me = obj.data
    va = me.color_attributes.get("mcol")
    if va is None:
        va = me.color_attributes.new("mcol", "FLOAT_COLOR", "POINT")
    va.data.foreach_set("color", col.ravel())
    me.update()
    return {name: int((weights[:, i] > 1e-4).sum()) for i, name in enumerate(MUSCLE_ORDER)}


def paint_from_builder(meshes, builder):
    totals = {n: 0 for n in MUSCLE_ORDER}
    per = []
    for o in meshes:
        world, no = _world_co_no(o)
        if not len(world):
            continue
        w, _ = builder(world, no)
        counts = write_mcol_soft(o, w)
        for k, v in counts.items():
            totals[k] += v
        per.append({"name": o.name, **counts, "verts": int(len(world))})
    return totals, per




def sphere_weight(world, center, radius, inner=0.40):
    dist = np.linalg.norm(world - np.asarray(center, np.float64), axis=1)
    return (1.0 - smoothstep(inner * radius, radius, dist)).astype(np.float32)


def muscle_spheres(hip, spine, side, back, slen):
    """v1-style large smooth spheres, mirrored across the anatomic midline."""
    s = float(slen / 0.55)

    def pt(t, lat, bck):
        return hip + (t * slen) * spine + (lat * s) * side + (bck * s) * back

    groups = {
        "lats": {
            "radius": 0.17 * s,
            "points": [pt(0.60, 0.16, 0.05), pt(0.48, 0.15, 0.06),
                       pt(0.36, 0.13, 0.07), pt(0.28, 0.11, 0.08)],
            "mirror": True,
        },
        "mid_back": {
            "radius": 0.16 * s,
            "points": [pt(0.62, 0.00, 0.08), pt(0.74, 0.045, 0.07),
                       pt(0.74, -0.045, 0.07)],
            "mirror": False,
        },
        "rear_delt": {
            "radius": 0.095 * s,
            "points": [pt(0.82, 0.20, 0.055)],
            "mirror": True,
        },
        "biceps": {
            "radius": 0.09 * s,
            "points": [pt(0.54, 0.25, -0.065), pt(0.48, 0.26, -0.075)],
            "mirror": True,
        },
    }
    # Expand mirrored copies.
    for name, g in groups.items():
        if not g["mirror"]:
            continue
        extra = [mirror_across_midline(p, hip, side) for p in g["points"]]
        g["points"] = list(g["points"]) + extra
    return groups


def weights_from_spheres(world, normals, groups, hip, spine, side, back, slen, ext):
    gates = muscle_gates(world, normals, hip, spine, side, back, slen, ext)
    w = np.zeros((len(world), len(MUSCLE_ORDER)), np.float32)
    for i, name in enumerate(MUSCLE_ORDER):
        g = groups[name]
        acc = np.zeros(len(world), np.float32)
        for p in g["points"]:
            acc = np.maximum(acc, sphere_weight(world, p, g["radius"]))
        w[:, i] = acc * gates[name]
    return w, gates


def paint_blob(meshes, hip, spine, side, back, slen):
    ext = surface_extents(meshes, hip, spine, side, back, slen)
    print("EXTENTS", ext, "slen", float(slen), flush=True)
    groups = muscle_spheres(hip, spine, side, back, slen)

    def builder(world, no):
        return weights_from_spheres(world, no, groups, hip, spine, side, back, slen, ext)

    return paint_from_builder(meshes, builder)


def paint_strokes(meshes, hip, spine, side, back, slen):
    ext = surface_extents(meshes, hip, spine, side, back, slen)
    print("EXTENTS", ext, "slen", float(slen), flush=True)
    strokes = muscle_strokes(hip, spine, side, back, slen, ext)

    def builder(world, no):
        return weights_from_strokes(world, no, strokes, hip, spine, side, back, slen, ext)

    return paint_from_builder(meshes, builder)

def _smooth_field(w, edges, steps=3):
    w = w.astype(np.float32, copy=True)
    for _ in range(int(steps)):
        acc = np.zeros_like(w)
        cnt = np.zeros_like(w)
        acc[edges[:, 0]] += w[edges[:, 1]]
        acc[edges[:, 1]] += w[edges[:, 0]]
        cnt[edges[:, 0]] += 1.0
        cnt[edges[:, 1]] += 1.0
        avg = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), w)
        w = 0.45 * w + 0.55 * avg
    return w


def classify_vertex_group(name):
    n = (name or "").lower().replace(" ", "")
    if any(k in n for k in ("finger", "thumb", "toe", "foot", "hand", "wrist",
                            "head", "neck", "eye", "jaw", "tongue")):
        return "ignore"
    if any(k in n for k in ("upleg", "up_leg", "thigh", "calf", "shin", "knee",
                            "butt", "glute", "pelvis", "hip")):
        return "ignore"
    # 'leg' after the above still catches LeftLeg / RightLeg
    if "leg" in n:
        return "ignore"
    if any(k in n for k in ("fore", "lowerarm", "lower_arm")):
        return "ignore"
    if any(k in n for k in ("upperarm", "upper_arm", "uparm", "humerus")):
        return "biceps"
    if "arm" in n:
        return "biceps"
    if any(k in n for k in ("shoulder", "clavicle", "collar", "deltoid", "scapula")):
        return "rear_delt"
    if any(k in n for k in ("spine", "chest", "torso", "rib")):
        return "torso"
    return None


def vg_weight_array(obj, group_indices):
    n = len(obj.data.vertices)
    w = np.zeros(n, np.float32)
    if not group_indices:
        return w
    want = set(group_indices)
    for vi, v in enumerate(obj.data.vertices):
        s = 0.0
        for g in v.groups:
            if g.group in want:
                s += g.weight
        w[vi] = s
    return w


def paint_rig_weights(meshes, hip, spine, side, back, slen):
    """Colour from skin weights; no hand-placed anchors. Requires vertex groups."""
    totals = {n: 0 for n in MUSCLE_ORDER}
    per = []
    used = []
    ext = surface_extents(meshes, hip, spine, side, back, slen)
    print("EXTENTS", ext, "slen", float(slen), flush=True)
    for o in meshes:
        world, no = _world_co_no(o)
        if not len(world):
            continue
        buckets = {"biceps": [], "rear_delt": [], "torso": []}
        for gi, vg in enumerate(o.vertex_groups):
            kind = classify_vertex_group(vg.name)
            if kind in buckets:
                buckets[kind].append(gi)
                used.append({"object": o.name, "group": vg.name, "kind": kind})
        if not any(buckets.values()):
            per.append({"name": o.name, "verts": int(len(world)), "note": "no classified vertex groups"})
            continue
        w_bic = vg_weight_array(o, buckets["biceps"])
        w_delt = vg_weight_array(o, buckets["rear_delt"])
        w_torso = vg_weight_array(o, buckets["torso"])
        gates = muscle_gates(world, no, hip, spine, side, back, slen, ext)
        # Split the torso weights into lat vs mid-back by where they sit.
        w_lats = w_torso * gates["lats"]
        w_mid = w_torso * gates["mid_back"]
        w_delt = w_delt * np.clip(gates["rear_delt"] + 0.25 * smoothstep(-0.05, 0.20, gates["nb"]), 0, 1)
        w_bic = w_bic * np.maximum(gates["biceps"], 0.15 * (1.0 - smoothstep(0.02, 0.10, gates["bck"])))
        # Extra: kill hands / hips leftover from fat weights.
        w_lats *= (1.0 - (gates["t"] < 0.16).astype(np.float32))
        edges = _edges_np(o.data)
        w_stack = np.stack([
            _smooth_field(w_lats, edges, 3),
            _smooth_field(w_mid, edges, 3),
            _smooth_field(w_delt, edges, 3),
            _smooth_field(w_bic, edges, 3),
        ], 1)
        # Re-apply gates lightly after smoothing so blur cannot walk onto a hand.
        w_stack[:, 0] *= np.clip(gates["lats"] + 0.15, 0, 1)
        w_stack[:, 1] *= np.clip(gates["mid_back"] + 0.15, 0, 1)
        w_stack[:, 2] *= np.clip(gates["rear_delt"] + 0.20, 0, 1)
        w_stack[:, 3] *= np.clip(gates["biceps"] + 0.20, 0, 1)
        counts = write_mcol_soft(o, w_stack)
        for k, v in counts.items():
            totals[k] += v
        per.append({"name": o.name, **counts, "verts": int(len(world)),
                    "groups": {k: len(v) for k, v in buckets.items()}})
    return totals, per, used


def punch_material(em_str=None):
    mat = bpy.data.materials.new("MuscleBlobV2")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    grey = nt.nodes.new("ShaderNodeBsdfPrincipled")
    grey.inputs["Base Color"].default_value = (*GREY, 1)
    grey.inputs["Roughness"].default_value = ROUGH
    if "Specular IOR Level" in grey.inputs:
        grey.inputs["Specular IOR Level"].default_value = SPEC
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "mcol"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(attr.outputs["Color"], sep.inputs["Color"])
    mx1 = nt.nodes.new("ShaderNodeMath")
    mx1.operation = "MAXIMUM"
    nt.links.new(sep.outputs["Red"], mx1.inputs[0])
    nt.links.new(sep.outputs["Green"], mx1.inputs[1])
    mx2 = nt.nodes.new("ShaderNodeMath")
    mx2.operation = "MAXIMUM"
    nt.links.new(mx1.outputs[0], mx2.inputs[0])
    nt.links.new(sep.outputs["Blue"], mx2.inputs[1])
    tint = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(attr.outputs["Color"], tint.inputs["Base Color"])
    tint.inputs["Roughness"].default_value = 0.70
    if "Specular IOR Level" in tint.inputs:
        tint.inputs["Specular IOR Level"].default_value = 0.08
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(attr.outputs["Color"], em.inputs["Color"])
    stm = nt.nodes.new("ShaderNodeMath")
    stm.operation = "MULTIPLY"
    nt.links.new(mx2.outputs[0], stm.inputs[0])
    stm.inputs[1].default_value = float(EM_STR if em_str is None else em_str)
    nt.links.new(stm.outputs[0], em.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tint.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mx2.outputs[0], mix.inputs["Fac"])
    nt.links.new(grey.outputs[0], mix.inputs[1])
    nt.links.new(add.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def rim_material():
    """Glowing rim (fresnel) plus a light interior tint, masked by mcol."""
    mat = bpy.data.materials.new("MuscleRimV2")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    grey = nt.nodes.new("ShaderNodeBsdfPrincipled")
    grey.inputs["Base Color"].default_value = (*GREY, 1)
    grey.inputs["Roughness"].default_value = ROUGH
    if "Specular IOR Level" in grey.inputs:
        grey.inputs["Specular IOR Level"].default_value = SPEC
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "mcol"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(attr.outputs["Color"], sep.inputs["Color"])
    mx1 = nt.nodes.new("ShaderNodeMath")
    mx1.operation = "MAXIMUM"
    nt.links.new(sep.outputs["Red"], mx1.inputs[0])
    nt.links.new(sep.outputs["Green"], mx1.inputs[1])
    mx2 = nt.nodes.new("ShaderNodeMath")
    mx2.operation = "MAXIMUM"
    nt.links.new(mx1.outputs[0], mx2.inputs[0])
    nt.links.new(sep.outputs["Blue"], mx2.inputs[1])
    fres = nt.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.85
    # rim_str = fresnel * 2.4 + interior 0.20
    mul_f = nt.nodes.new("ShaderNodeMath")
    mul_f.operation = "MULTIPLY"
    nt.links.new(fres.outputs[0], mul_f.inputs[0])
    mul_f.inputs[1].default_value = 2.4
    add_i = nt.nodes.new("ShaderNodeMath")
    add_i.operation = "ADD"
    nt.links.new(mul_f.outputs[0], add_i.inputs[0])
    add_i.inputs[1].default_value = 0.20
    mul_m = nt.nodes.new("ShaderNodeMath")
    mul_m.operation = "MULTIPLY"
    nt.links.new(add_i.outputs[0], mul_m.inputs[0])
    nt.links.new(mx2.outputs[0], mul_m.inputs[1])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(attr.outputs["Color"], em.inputs["Color"])
    nt.links.new(mul_m.outputs[0], em.inputs["Strength"])
    tint = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(attr.outputs["Color"], tint.inputs["Base Color"])
    tint.inputs["Roughness"].default_value = 0.78
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tint.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mx2.outputs[0], mix.inputs["Fac"])
    nt.links.new(grey.outputs[0], mix.inputs[1])
    nt.links.new(add.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def assign_mat(meshes, mat):
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        try:
            if hasattr(o.data, "attributes") and "custom_normal" in o.data.attributes:
                o.data.attributes.remove(o.data.attributes["custom_normal"])
        except Exception:
            pass
        try:
            o.data.use_auto_smooth = False
        except Exception:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            pass
    for img in list(bpy.data.images):
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass


def dim_white_lights(key=16.0, fill=5.0, rim=7.0):
    for name, energy in (("Key", key), ("Fill", fill), ("Rim", rim)):
        o = bpy.data.objects.get(name)
        if o is not None and o.type == "LIGHT":
            o.data.energy = float(energy)



def add_colored_spots(meshes, hip, spine, side, back, slen):
    """Project colour with spots aimed at each capsule; no mesh edits."""
    ext = surface_extents(meshes, hip, spine, side, back, slen)
    print("EXTENTS", ext, "slen", float(slen), flush=True)
    caps = muscle_capsules(hip, spine, side, back, slen, ext)
    created = []
    idx = 0
    for name in MUSCLE_ORDER:
        color = [float(c) for c in COL[name]]
        for c in caps[name]:
            a, b = c["a"], c["b"]
            r_avg = 0.5 * (c["r_maj_a"] + c["r_maj_b"])
            thick = c["thick"]
            tn = float(np.linalg.norm(thick)) or 1.0
            thick = thick / tn
            axis = b - a
            length = float(np.linalg.norm(axis)) or 0.12
            fracs = (0.32, 0.72) if length > 0.14 else (0.50,)
            for fr in fracs:
                target = a + fr * (b - a)
                loc = target + thick * (0.14 + 0.55 * r_avg)
                dist = float(np.linalg.norm(loc - target)) or 0.14
                spot_size = 2.0 * math.atan((1.15 * r_avg) / dist)
                spot_size = float(max(0.22, min(1.15, spot_size)))
                d = bpy.data.lights.new(f"MusSpot_{name}_{idx}", "SPOT")
                d.color = color
                d.energy = 95.0 if name != "mid_back" else 80.0
                d.spot_size = spot_size
                d.spot_blend = 0.88
                d.shadow_soft_size = 0.07
                o = bpy.data.objects.new(d.name, d)
                bpy.context.scene.collection.objects.link(o)
                o.location = Vector((float(loc[0]), float(loc[1]), float(loc[2])))
                direction = Vector((float(target[0] - loc[0]), float(target[1] - loc[1]),
                                    float(target[2] - loc[2])))
                o.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
                created.append(o.name)
                idx += 1
    return created

def slight_bloom(png, sigma=1.35, amount=0.20):
    img = bpy.data.images.load(png)
    w, h = img.size
    ch = img.channels
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)
    rgb = px[..., :3].copy()
    mx = rgb.max(axis=2)
    chroma = mx - rgb.min(axis=2)
    mask = np.clip((chroma - 0.10) / 0.22, 0.0, 1.0) * (mx > 0.18).astype(np.float32)
    glow = rgb * mask[..., None]
    try:
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            glow[..., c] = gaussian_filter(glow[..., c], sigma=sigma)
    except Exception:
        pass
    out = np.clip(rgb + glow * amount, 0.0, 1.0)
    if ch == 4:
        px[..., :3] = out
    else:
        px = out
    img.pixels.foreach_set(px.astype(np.float32).ravel())
    img.filepath_raw = png
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return png


def try_one_arm_row(arms):
    """Best-effort FK one-arm row on a Mixamo-like rig. Inspect the still."""
    if not arms:
        return {"ok": False, "reason": "no armature"}
    arm = arms[0]
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    lower = {b.name.lower(): b for b in arm.pose.bones}

    def find(*needles, exclude=("fore", "hand", "twist", "end", "toe")):
        for name, b in lower.items():
            if any(x in name for x in exclude):
                continue
            if all(n in name for n in needles):
                return b
        return None

    r_arm = find("right", "arm")
    r_fore = None
    for name, b in lower.items():
        if "right" in name and "fore" in name:
            r_fore = b
            break
    r_sh = find("right", "shoulder")
    hips = find("hips") or find("hip")
    # Prefer the upper-most spine for a slight hinge.
    spine = None
    for key in ("spine_011", "spine01_010", "spine02_09", "spine"):
        for name, b in lower.items():
            if key in name.replace(" ", ""):
                spine = b
                break
        if spine is not None:
            break

    applied = []

    def rot(b, xyz, label):
        if b is None:
            return
        b.rotation_mode = "XYZ"
        b.rotation_euler = Euler(xyz, "XYZ")
        applied.append({"bone": b.name, "label": label, "xyz_deg": [round(math.degrees(v), 1) for v in xyz]})

    rot(hips, (math.radians(30), 0.0, 0.0), "hips_hinge")
    rot(spine, (math.radians(12), 0.0, 0.0), "spine_hinge")
    rot(r_sh, (math.radians(-8), math.radians(-12), math.radians(-18)), "r_shoulder")
    rot(r_arm, (math.radians(-20), math.radians(18), math.radians(75)), "r_upper_arm")
    rot(r_fore, (math.radians(100), 0.0, 0.0), "r_forearm")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    return {
        "ok": bool(applied),
        "applied": applied,
        "found": {
            "r_arm": None if r_arm is None else r_arm.name,
            "r_fore": None if r_fore is None else r_fore.name,
            "r_shoulder": None if r_sh is None else r_sh.name,
            "hips": None if hips is None else hips.name,
            "spine": None if spine is None else spine.name,
        },
    }


# --- hybrid v3: V-fan lats + diamond mid-back (blob) and rig-weight delts/biceps ---

COL_HYBRID = {
    "lats": np.array([1.00, 0.26, 0.03], np.float32),
    "mid_back": np.array([1.00, 0.16, 0.46], np.float32),
    "rear_delt": np.array([0.64, 0.10, 1.00], np.float32),
    "biceps": np.array([1.00, 0.66, 0.00], np.float32),
}
EM_STR_HYBRID = 0.82


def lat_fan_capsules(hip, spine, side, back, slen, ext=None):
    """V-shaped lat fans: wide at the armpit, tapering down the flank.

    Bottom ends well above the waist so left and right never meet as a lumbar belt.
    Centres sit near the back surface so the capsules actually reach the skin.
    """
    s = float(slen / 0.55)
    ss = min(float((ext or {}).get("side_surf", 0.22)), 0.23)
    bs = max(float((ext or {}).get("back_surf", 0.14)), 0.10)

    def pt(t, lat_frac, bck_frac):
        return hip + (t * slen) * spine + (lat_frac * ss) * side + (bck_frac * bs) * back

    segs = [
        # upper fan (armpit / teres -> mid flank), on the back surface
        dict(a=pt(0.80, 0.88, 0.82), b=pt(0.60, 0.82, 0.90),
             r_maj_a=0.11 * s, r_maj_b=0.090 * s,
             r_min_a=0.055 * s, r_min_b=0.048 * s),
        # lower taper (mid flank -> above iliac, still lateral)
        dict(a=pt(0.60, 0.82, 0.90), b=pt(0.46, 0.62, 0.84),
             r_maj_a=0.090 * s, r_maj_b=0.055 * s,
             r_min_a=0.048 * s, r_min_b=0.032 * s),
    ]
    caps = []
    for seg in segs:
        seg = dict(seg)
        seg["thick"] = back
        caps.append(seg)
        mir = {
            "a": mirror_across_midline(seg["a"], hip, side),
            "b": mirror_across_midline(seg["b"], hip, side),
            "r_maj_a": seg["r_maj_a"], "r_maj_b": seg["r_maj_b"],
            "r_min_a": seg["r_min_a"], "r_min_b": seg["r_min_b"],
            "thick": back,
        }
        caps.append(mir)
    return caps


def diamond_midback_weight(world, hip, spine, side, back, slen):
    """Pink diamond between the scapulae — pointed top and bottom, not a lumbar smear."""
    s = float(slen / 0.55)
    rel = world - hip
    t = (rel @ spine) / max(slen, 1e-8)
    abs_lat = np.abs(rel @ side)
    bck = rel @ back
    dt = np.abs(t - 0.74)
    h = 1.0 - smoothstep(0.14, 0.22, dt)
    half_w = (0.030 * s + 0.115 * s * h)
    lat_g = 1.0 - smoothstep(half_w * 0.40, half_w, abs_lat)
    back_g = smoothstep(-0.01 * s, 0.04 * s, bck)
    return (h * lat_g * back_g).astype(np.float32)


def vfan_lat_weight(world, hip, spine, side, back, slen, caps):
    """Soft V-fan from capsules × anatomical gate that keeps a lumbar midline gap."""
    s = float(slen / 0.55)
    rel = world - hip
    t = (rel @ spine) / max(slen, 1e-8)
    abs_lat = np.abs(rel @ side)
    bck = rel @ back
    acc = np.zeros(len(world), np.float32)
    for c in caps:
        acc = np.maximum(acc, tapered_capsule_weight(
            world, c["a"], c["b"], c["r_maj_a"], c["r_maj_b"],
            c["r_min_a"], c["r_min_b"], c["thick"], inner=0.58,
        ))
    # Fade out ~8–10 cm above the waistband (hip origin ≈ pelvis / shorts top).
    # 0.09 m / slen ≈ 0.14–0.17 in t; start fading by t~0.34, full by t~0.42.
    waist_keep = smoothstep(0.32, 0.40, t)
    t_n = np.clip((t - 0.42) / 0.38, 0.0, 1.0)
    min_lat = (0.055 - 0.030 * t_n) * s
    max_lat = (0.160 + 0.105 * t_n) * s
    margin = 0.028 * s
    lat_g = smoothstep(min_lat - margin, min_lat, abs_lat) * (1.0 - smoothstep(max_lat, max_lat + margin, abs_lat))
    back_g = smoothstep(-0.015 * s, 0.025 * s, bck)
    hand = soft_range(t, -2.0, 0.34, 0.05) * soft_range(abs_lat, 0.20 * s, 8.0, 0.03 * s)
    return (acc * waist_keep * lat_g * back_g * (1.0 - hand)).astype(np.float32)


def _vg_side_boost(obj, working_side):
    """1.15 on the working side, 0.92 on the other, from Left*/Right* groups."""
    n = len(obj.data.vertices)
    if not working_side:
        return np.ones(n, np.float32)
    left_i, right_i = [], []
    for gi, vg in enumerate(obj.vertex_groups):
        nm = (vg.name or "").lower()
        if "left" in nm:
            left_i.append(gi)
        elif "right" in nm:
            right_i.append(gi)
    wl = vg_weight_array(obj, left_i)
    wr = vg_weight_array(obj, right_i)
    want_right = str(working_side).lower() in {"right", "r", "+1", "1"}
    primary, other = (wr, wl) if want_right else (wl, wr)
    boost = np.ones(n, np.float32)
    mask_p = primary > other
    mask_o = other > primary
    boost[mask_p] = 1.16
    boost[mask_o] = 0.90
    return boost


def paint_hybrid(meshes, hip, spine, side, back, slen, working_side=None):
    """Lats + mid-back from a V-fan / diamond blob; rear delts + biceps from rig weights."""
    ext = surface_extents(meshes, hip, spine, side, back, slen)
    print("EXTENTS hybrid", ext, "slen", float(slen), flush=True)
    caps = lat_fan_capsules(hip, spine, side, back, slen, ext)
    totals = {n: 0 for n in MUSCLE_ORDER}
    per = []
    used = []
    for o in meshes:
        world, no = _world_co_no(o)
        if not len(world):
            continue
        w_lats = vfan_lat_weight(world, hip, spine, side, back, slen, caps)
        w_mid = diamond_midback_weight(world, hip, spine, side, back, slen)
        buckets = {"biceps": [], "rear_delt": [], "torso": []}
        for gi, vg in enumerate(o.vertex_groups):
            kind = classify_vertex_group(vg.name)
            if kind in buckets:
                buckets[kind].append(gi)
                used.append({"object": o.name, "group": vg.name, "kind": kind})
        w_bic = vg_weight_array(o, buckets["biceps"])
        w_delt = vg_weight_array(o, buckets["rear_delt"])
        gates = muscle_gates(world, no, hip, spine, side, back, slen, ext)
        # Rear-delt cap + a little posterior upper-arm wrap (optional rear arm).
        rear_arm = w_bic * smoothstep(-0.02 * (slen / 0.55), 0.06 * (slen / 0.55), gates["bck"])
        w_delt = w_delt * np.clip(gates["rear_delt"] + 0.30, 0, 1)
        w_delt = np.maximum(w_delt, 0.45 * rear_arm * gates["rear_delt"])
        w_bic = w_bic * np.maximum(gates["biceps"], 0.18 * (1.0 - smoothstep(0.02, 0.10, gates["bck"])))
        edges = _edges_np(o.data)
        w_stack = np.stack([
            w_lats,
            w_mid,
            _smooth_field(w_delt, edges, 2),
            _smooth_field(w_bic, edges, 2),
        ], 1)
        w_stack[:, 2] *= np.clip(gates["rear_delt"] + 0.22, 0, 1)
        w_stack[:, 3] *= np.clip(gates["biceps"] + 0.22, 0, 1)
        boost = _vg_side_boost(o, working_side)
        w_stack *= boost[:, None]
        counts = write_mcol_soft(o, w_stack, colours=COL_HYBRID)
        for k, v in counts.items():
            totals[k] += v
        per.append({"name": o.name, **counts, "verts": int(len(world)),
                    "groups": {k: len(v) for k, v in buckets.items()}})
    return totals, per, used
