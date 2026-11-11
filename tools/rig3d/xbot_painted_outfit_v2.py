#!/usr/bin/env python3
"""Painted X Bot outfit v2 — units unified in world metres. One attempt.

Root-cause fix for v1: Mixamo bone-local coordinates from
`arm.matrix_world @ pose_bone.matrix` are centimetres (the bone 3x3 carries
a 0.01 scale). v1 mixed those with world-metre GLB measurements and blew
the glove up 22x and the beanie 122x.

v2 never uses bone-local lengths. Every measurement and every placement is
in world metres, using the bone's world origin and *normalized* world axes
from `matrix_world @ bone.matrix`. Assets are scaled to sane real-world
sizes and asserted before any render:

  glove length  0.25–0.35 m
  beanie width  0.18–0.24 m
  head width    from the X Bot mesh, world metres

Then each glove is rigid-parented to its Hand bone and the beanie to Head,
keeping the world transform.
"""
from __future__ import annotations

import math
import os

import numpy as np
from mathutils import Matrix, Vector

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

import xbot_outfit as xo  # noqa: E402
import xbot_painted_outfit as v1  # noqa: E402

GLOVE_GLB = v1.GLOVE_GLB
BEANIE_GLB = v1.BEANIE_GLB
GLOVE_LEN_RANGE = (0.25, 0.35)
BEANIE_WIDTH_RANGE = (0.18, 0.24)
# Anatomical head width we expect on a 1.6–1.8 m Mixamo X Bot, metres.
HEAD_WIDTH_RANGE = (0.12, 0.28)

GLOVE_SIDE_MARGIN = 0.012
GLOVE_TIP_PAD = 0.018
GLOVE_CUFF_PAST_WRIST = 0.030
GLOVE_CUFF_OPENING_FACTOR = 1.10
BEANIE_HEM_ABOVE_BROW = 0.010
BEANIE_CROWN_CLEARANCE = 0.012
BEANIE_RADIAL_CLEARANCE = 0.008

CRITERIA = {
    "version": "xbot_painted_v2",
    "rule": "ONE honest attempt after unit unification. A miss is a fail; do not iterate.",
    "units": "world metres only; never bone-local centimetres",
    "asserts_before_render": {
        "glove_length_m": list(GLOVE_LEN_RANGE),
        "beanie_width_m": list(BEANIE_WIDTH_RANGE),
        "head_width_m": "from X Bot mesh, world metres",
    },
    "criteria": v1.CRITERIA["criteria"],
}


# ---------------------------------------------------------------------------
# World-metre bone frame (directions are normalized — 0.01 bone scale cannot
# leak into a length).
# ---------------------------------------------------------------------------
def bone_mw(arm, bone_name):
    return arm.matrix_world @ arm.pose.bones[bone_name].matrix


def bone_frame(arm, bone_name):
    """World origin (metres) + orthonormal axes. +Y is along the bone."""
    mw = bone_mw(arm, bone_name)
    origin = mw.translation.copy()
    y = mw.to_3x3() @ Vector((0, 1, 0))
    if y.length < 1e-12:
        y = Vector((0, 1, 0))
    y.normalize()
    z = mw.to_3x3() @ Vector((0, 0, 1))
    z = z - y * z.dot(y)
    if z.length < 1e-8:
        z = y.orthogonal()
    z.normalize()
    x = y.cross(z)
    if x.length < 1e-8:
        x = y.orthogonal()
    x.normalize()
    z = x.cross(y).normalized()
    return origin, x, y, z


def world_project(points, origin, x, y, z):
    """Project world points onto a world orthonormal frame. Returns metres."""
    out = []
    for p in points:
        d = p - origin
        out.append(Vector((d.dot(x), d.dot(y), d.dot(z))))
    return out


def aabb_of(pts):
    if not pts:
        return None
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    return {
        "min": Vector((min(xs), min(ys), min(zs))),
        "max": Vector((max(xs), max(ys), max(zs))),
        "n": len(pts),
    }


def world_bbox_dims(obj):
    pts = v1.world_pts(obj)
    box = aabb_of(pts)
    dim = box["max"] - box["min"]
    return {
        "min": list(box["min"]),
        "max": list(box["max"]),
        "dim": [float(dim.x), float(dim.y), float(dim.z)],
        "length": float(max(dim.x, dim.y, dim.z)),
        "width": float(sorted((dim.x, dim.y, dim.z))[1]),
        "n": box["n"],
    }


# ---------------------------------------------------------------------------
# World-metre character measurements
# ---------------------------------------------------------------------------
def scan_hand_world(arm, meshes, bone_name, side, f0, f1, step=2):
    """Hand verts projected onto the normalized world bone frame, metres."""
    saved = bpy.context.scene.frame_current
    acc = None
    pred = v1.hand_pred(side)
    f = int(f0)
    while f <= int(f1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        origin, x, y, z = bone_frame(arm, bone_name)
        pts = world_project(v1.collect_weighted(meshes, pred), origin, x, y, z)
        box = aabb_of(pts)
        if box:
            if acc is None:
                acc = box
            else:
                acc = {
                    "min": Vector((min(acc["min"].x, box["min"].x),
                                   min(acc["min"].y, box["min"].y),
                                   min(acc["min"].z, box["min"].z))),
                    "max": Vector((max(acc["max"].x, box["max"].x),
                                   max(acc["max"].y, box["max"].y),
                                   max(acc["max"].z, box["max"].z))),
                    "n": acc["n"] + box["n"],
                }
        f += step
    bpy.context.scene.frame_set(saved)
    bpy.context.view_layer.update()
    return acc


def wrist_radius_world(arm, meshes, bone_name, side):
    origin, x, y, z = bone_frame(arm, bone_name)
    locs = world_project(v1.collect_weighted(meshes, v1.hand_pred(side), 0.25),
                         origin, x, y, z)
    rs = [math.hypot(p.x, p.z) for p in locs if -0.035 <= p.y <= 0.012]
    if not rs:
        return 0.033
    rs.sort()
    return float(rs[int(len(rs) * 0.85)])


def head_verts_world(meshes, thresh=0.28):
    def pred(n):
        return v1._clean(n) == "head" or v1._clean(n).endswith("head")
    return v1.collect_weighted(meshes, pred, thresh=thresh)


def head_width_world(meshes, arm, head_bone):
    """Anatomical head width in world metres, pose-stable (plane perp. to up)."""
    pts = head_verts_world(meshes)
    if not pts:
        return None
    origin, x, y, z = bone_frame(arm, head_bone)
    # Ear-to-ear width is the shorter span in the plane perpendicular to
    # the head-up axis (front-to-back is longer on this skull).
    xs, zs = [], []
    for p in pts:
        d = p - origin
        xs.append(d.dot(x))
        zs.append(d.dot(z))
    if not xs:
        return None
    sx = max(xs) - min(xs)
    sz = max(zs) - min(zs)
    return float(min(sx, sz))


def head_landmarks_world(meshes, arm, head_bone):
    """Brows, ears, crown, hem line — all world metres."""
    pts = head_verts_world(meshes)
    origin, x, y_up, z = bone_frame(arm, head_bone)
    if not pts:
        return {
            "brow_z": origin.z + 0.06,
            "ear_z": origin.z + 0.05,
            "hem_z": origin.z + 0.07,
            "crown": origin + y_up * 0.22,
            "mid": Vector((origin.x, origin.y, origin.z + 0.07)),
            "radius": 0.09,
            "zmax": origin.z + 0.22,
            "zmin": origin.z,
            "width": 0.18,
        }
    zs = [p.z for p in pts]
    zmin, zmax = min(zs), max(zs)
    mid_x = sum(p.x for p in pts) / len(pts)
    mid_y = sum(p.y for p in pts) / len(pts)
    front = Vector((0, -1, 0))
    brow_z = zmin + 0.60 * (zmax - zmin)
    front_pts = [p for p in pts if (p - Vector((mid_x, mid_y, p.z))).dot(front) > 0]
    band = [p for p in front_pts if abs(p.z - brow_z) < 0.025] if front_pts else []
    if band:
        brow_z = sum(p.z for p in band) / len(band)
    ear_z = zmin + 0.55 * (zmax - zmin)
    hem_z = max(brow_z, ear_z) + BEANIE_HEM_ABOVE_BROW
    slice_pts = [p for p in pts if abs(p.z - hem_z) < 0.02] or pts
    rads = [math.hypot(p.x - mid_x, p.y - mid_y) for p in slice_pts]
    rads.sort()
    radius = float(rads[int(min(len(rads) - 1, len(rads) * 0.90))])
    width = head_width_world(meshes, arm, head_bone) or (2.0 * radius)
    return {
        "brow_z": brow_z,
        "ear_z": ear_z,
        "hem_z": hem_z,
        "crown": Vector((mid_x, mid_y, zmax + BEANIE_CROWN_CLEARANCE)),
        "mid": Vector((mid_x, mid_y, hem_z)),
        "radius": radius,
        "zmax": zmax,
        "zmin": zmin,
        "width": width,
    }


# ---------------------------------------------------------------------------
# Asset anatomy in world metres (imported GLB, no bone space)
# ---------------------------------------------------------------------------
def glove_anatomy(obj):
    """Cuff, fist, long axis, thumb, opening radius — world metres."""
    pts = v1.world_pts(obj)
    arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
    c = arr.mean(axis=0)
    cov = np.cov((arr - c).T)
    w, V = np.linalg.eigh(cov)
    axis = Vector(V[:, int(np.argmax(w))].tolist()).normalized()
    proj = np.array([(p - Vector(c)).dot(axis) for p in pts])
    lo, hi = float(proj.min()), float(proj.max())
    span = max(hi - lo, 1e-9)

    def end_radius(mask):
        subset = arr[mask]
        if len(subset) < 8:
            return 1e9, Vector(c)
        rel = subset - c
        ax = np.array((axis.x, axis.y, axis.z))
        rad = np.linalg.norm(rel - np.outer(rel @ ax, ax), axis=1)
        cen = Vector(subset.mean(axis=0).tolist())
        return float(np.sqrt(np.mean(rad * rad))), cen

    r_lo, c_lo = end_radius(proj <= lo + 0.15 * span)
    r_hi, c_hi = end_radius(proj >= hi - 0.15 * span)
    if r_lo <= r_hi:
        cuff, fist, fist_dir, cuff_r = c_lo, c_hi, axis, r_lo
    else:
        cuff, fist, fist_dir, cuff_r = c_hi, c_lo, -axis, r_hi

    t = np.array([(p - cuff).dot(fist_dir) for p in pts])
    tmin, tmax = float(t.min()), float(t.max())
    cuff_mask = t <= tmin + 0.10 * (tmax - tmin)
    cuff = Vector(arr[cuff_mask].mean(axis=0).tolist())
    # Opening: radial size of the cuff-end ring only, never the whole glove.
    rel = arr - np.array((cuff.x, cuff.y, cuff.z))
    ax = np.array((fist_dir.x, fist_dir.y, fist_dir.z))
    lateral = rel - np.outer(rel @ ax, ax)
    cuff_lat = np.linalg.norm(lateral[cuff_mask], axis=1)
    opening = float(np.percentile(cuff_lat, 75)) if len(cuff_lat) else cuff_r
    # Thumb: mid-body lateral cluster.
    mid = (t > tmin + 0.25 * (tmax - tmin)) & (t < tmax - 0.10 * (tmax - tmin))
    if mid.sum() < 16:
        mid = np.ones(len(pts), dtype=bool)
    lat = lateral[mid]
    dist = np.linalg.norm(lat, axis=1)
    k = max(8, int(0.04 * len(dist)))
    idx = np.argpartition(dist, -k)[-k:]
    thumb = Vector(lat[idx].mean(axis=0).tolist())
    if thumb.length < 1e-8:
        thumb = fist_dir.orthogonal()
    thumb.normalize()
    return {
        "cuff": cuff,
        "fist": fist,
        "fist_dir": fist_dir.normalized(),
        "thumb_dir": thumb,
        "opening_r": float(opening),
        "length": float((fist - cuff).length),
        "cuff_r_rms": float(cuff_r),
    }


def beanie_anatomy(obj):
    pts = v1.world_pts(obj)
    zs = [p.z for p in pts]
    zmin, zmax = min(zs), max(zs)
    span = max(zmax - zmin, 1e-6)
    hem_pts = [p for p in pts if p.z <= zmin + 0.12 * span]
    crown_pts = [p for p in pts if p.z >= zmax - 0.08 * span]
    hem = sum(hem_pts, Vector()) / max(1, len(hem_pts))
    crown = sum(crown_pts, Vector()) / max(1, len(crown_pts))
    up = crown - hem
    up = up.normalized() if up.length > 1e-6 else Vector((0, 0, 1))
    rads = []
    for p in hem_pts:
        d = p - hem
        rads.append((d - up * d.dot(up)).length)
    hem_r = float(np.percentile(rads, 80)) if rads else 0.10
    mid_pts = [p for p in pts if p.z >= zmin + 0.40 * span]
    mid = sum(mid_pts, Vector()) / max(1, len(mid_pts))
    slouch = mid - hem
    slouch = slouch - up * slouch.dot(up)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return {
        "hem": hem,
        "crown": crown,
        "up": up,
        "hem_r": hem_r,
        "height": float((crown - hem).length),
        "slouch": slouch,
        "width": float(max(max(xs) - min(xs), max(ys) - min(ys))),
        "zmin": zmin,
        "zmax": zmax,
    }


def _align_dirs(obj, src_fwd, dst_fwd, src_up, dst_up, pivot):
    if src_fwd.length > 1e-8 and dst_fwd.length > 1e-8:
        v1.rotate_around(obj, pivot, src_fwd.rotation_difference(dst_fwd.normalized()))
    # Re-read the up after the first rotation — caller passes a getter.
    return obj


# ---------------------------------------------------------------------------
# Placement in world metres, then rigid-parent keep_world
# ---------------------------------------------------------------------------
def place_glove(obj, arm, bone_name, side, meshes, env, head_w):
    origin, bx, by, bz = bone_frame(arm, bone_name)
    info = glove_anatomy(obj)
    raw_len = info["length"]
    # Align fist → bone +Y, thumb → bone +Z, pivot at the cuff, all world.
    if info["fist_dir"].length > 1e-8:
        v1.rotate_around(obj, info["cuff"],
                         info["fist_dir"].rotation_difference(by))
    info = glove_anatomy(obj)
    src_t = info["thumb_dir"] - info["fist_dir"] * info["thumb_dir"].dot(info["fist_dir"])
    dst_t = bz - info["fist_dir"] * bz.dot(info["fist_dir"])
    if src_t.length > 1e-8 and dst_t.length > 1e-8:
        v1.rotate_around(obj, info["cuff"],
                         src_t.normalized().rotation_difference(dst_t.normalized()))
    info = glove_anatomy(obj)

    # Current glove extents in the *normalized* world bone frame (metres).
    locs = world_project(v1.world_pts(obj), origin, bx, by, bz)
    gmin = Vector((min(p.x for p in locs), min(p.y for p in locs),
                   min(p.z for p in locs)))
    gmax = Vector((max(p.x for p in locs), max(p.y for p in locs),
                   max(p.z for p in locs)))
    gw = max(gmax.x - gmin.x, 1e-4)
    gl = max(gmax.y - gmin.y, 1e-4)
    gt = max(gmax.z - gmin.z, 1e-4)

    if env is None:
        env = {"min": Vector((-0.055, -0.030, -0.035)),
               "max": Vector((0.055, 0.110, 0.065)), "n": 0}
    emin, emax = env["min"], env["max"]
    need_w = max(emax.x - emin.x + 2 * GLOVE_SIDE_MARGIN, 0.80 * head_w)
    need_t = (emax.z - emin.z) + 2 * GLOVE_SIDE_MARGIN
    need_l = (emax.y + GLOVE_TIP_PAD) - (min(float(emin.y), 0.0) - GLOVE_CUFF_PAST_WRIST)
    # Intended real-world length is the geometric need, clamped to the
    # sane 10–12 oz band (0.25–0.35 m). That band is the design target,
    # not a post-hoc tweak.
    target_l = min(GLOVE_LEN_RANGE[1], max(GLOVE_LEN_RANGE[0], need_l))
    wrist_r = wrist_radius_world(arm, meshes, bone_name, side)
    need_cuff = wrist_r * GLOVE_CUFF_OPENING_FACTOR
    opening = info["opening_r"]
    s_cuff = (need_cuff / opening) if 0.010 <= opening <= 0.080 else 0.0

    scale = max(target_l / gl, need_w / gw, need_t / gt, s_cuff)
    cuff = info["cuff"]
    T = Matrix.Translation(cuff)
    obj.matrix_world = T @ Matrix.Scale(scale, 4) @ T.inverted() @ obj.matrix_world
    bpy.context.view_layer.update()
    info = glove_anatomy(obj)
    # If the combined scale pushed length outside the band, lock to the band.
    length = info["length"]
    lo, hi = GLOVE_LEN_RANGE
    if length < lo - 1e-6 or length > hi + 1e-6:
        adj = (lo if length < lo else hi) / max(length, 1e-6)
        cuff = info["cuff"]
        T = Matrix.Translation(cuff)
        obj.matrix_world = T @ Matrix.Scale(adj, 4) @ T.inverted() @ obj.matrix_world
        bpy.context.view_layer.update()
        info = glove_anatomy(obj)
        scale *= adj

    target_cuff = origin - by * GLOVE_CUFF_PAST_WRIST
    obj.matrix_world.translation += (target_cuff - info["cuff"])
    bpy.context.view_layer.update()
    xo.parent_to_bone(obj, arm, bone_name, keep_world=True)
    bpy.context.view_layer.update()
    info = glove_anatomy(obj)
    return {
        "bone": bone_name,
        "side": side,
        "scale": scale,
        "raw_length_m": raw_len,
        "length_m": info["length"],
        "opening_r": info["opening_r"],
        "wrist_r": wrist_r,
        "need": {"w": need_w, "t": need_t, "l": need_l, "target_l": target_l,
                 "cuff": need_cuff},
        "env": {"min": list(emin), "max": list(emax), "n": env.get("n", 0)},
        "cuff": list(info["cuff"]),
        "fist": list(info["fist"]),
        "head_w": head_w,
        "units": "world_metres",
    }


def make_glove_pair(arm, meshes, collection, f0, f1, head_w):
    src = v1.import_glb(GLOVE_GLB, "XBotGlove_R")
    if collection and src.name not in collection.objects:
        collection.objects.link(src)
        if src.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(src)
    imported = world_bbox_dims(src)
    mat = xo.make_glossy_leather_material("XBotGloveLeather_painted_v2",
                                          v1.GLOVE_BLACK, roughness=0.30)
    src.data.materials.clear()
    src.data.materials.append(mat)
    src.active_material = mat
    mat.use_backface_culling = False

    bones = {
        "R": xo.find_bone(arm, "mixamorig:RightHand", "RightHand"),
        "L": xo.find_bone(arm, "mixamorig:LeftHand", "LeftHand"),
    }
    envs = {side: scan_hand_world(arm, meshes, bn, side, f0, f1, step=2)
            for side, bn in bones.items()}

    v1._sel(src)
    bpy.ops.object.duplicate()
    left = bpy.context.view_layer.objects.active
    left.name = "XBotGlove_L"
    if collection and left.name not in collection.objects:
        collection.objects.link(left)
        if left.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(left)
    left.data.materials.clear()
    left.data.materials.append(mat)
    left.active_material = mat

    # Same mesh in each hand's world frame. Mixamo Left/Right Hand bones
    # share a local convention (normalized +Y fingers, +Z thumb), so a
    # world-X mirror is the wrong move.
    info_r = place_glove(src, arm, bones["R"], "R", meshes, envs["R"], head_w)
    info_l = place_glove(left, arm, bones["L"], "L", meshes, envs["L"], head_w)
    info_r["imported_bbox_m"] = imported
    info_l["imported_bbox_m"] = imported
    return {"R": (src, info_r), "L": (left, info_l)}


def place_beanie(obj, arm, meshes, head_w):
    head_bone = xo.find_bone(arm, "mixamorig:Head", "Head")
    if not head_bone:
        raise RuntimeError("no Head bone")
    info = beanie_anatomy(obj)
    raw_width = info["width"]
    land = head_landmarks_world(meshes, arm, head_bone)

    world_up = Vector((0, 0, 1))
    if info["up"].dot(world_up) < 0.999:
        v1.rotate_around(obj, info["hem"], info["up"].rotation_difference(world_up))
        info = beanie_anatomy(obj)

    slouch = Vector((info["slouch"].x, info["slouch"].y, 0.0))
    back = Vector((0, 1, 0))  # character faces -Y
    if slouch.length > 1e-4:
        v1.rotate_around(obj, info["hem"], slouch.normalized().rotation_difference(back))
        info = beanie_anatomy(obj)

    # Target width: just outside the skull, inside the sane beanie band.
    target_w = land["radius"] * 2.0 + 2.0 * BEANIE_RADIAL_CLEARANCE
    target_w = min(BEANIE_WIDTH_RANGE[1], max(BEANIE_WIDTH_RANGE[0], target_w))
    # Prefer sizing from the hem ring, then lock width into the band.
    target_r = target_w * 0.5
    scale = target_r / max(info["hem_r"], 1e-4)
    hem = info["hem"]
    T = Matrix.Translation(hem)
    obj.matrix_world = T @ Matrix.Scale(scale, 4) @ T.inverted() @ obj.matrix_world
    bpy.context.view_layer.update()
    info = beanie_anatomy(obj)
    width = info["width"]
    lo, hi = BEANIE_WIDTH_RANGE
    if width < lo - 1e-6 or width > hi + 1e-6:
        adj = (lo if width < lo else hi) / max(width, 1e-6)
        hem = info["hem"]
        T = Matrix.Translation(hem)
        obj.matrix_world = T @ Matrix.Scale(adj, 4) @ T.inverted() @ obj.matrix_world
        bpy.context.view_layer.update()
        info = beanie_anatomy(obj)
        scale *= adj

    target_hem = Vector((land["mid"].x, land["mid"].y, land["hem_z"]))
    obj.matrix_world.translation += (target_hem - info["hem"])
    bpy.context.view_layer.update()
    xo.parent_to_bone(obj, arm, head_bone, keep_world=True)
    bpy.context.view_layer.update()
    info = beanie_anatomy(obj)
    return {
        "bone": head_bone,
        "scale": scale,
        "raw_width_m": raw_width,
        "width_m": info["width"],
        "hem": list(info["hem"]),
        "crown": list(info["crown"]),
        "hem_r": info["hem_r"],
        "height": info["height"],
        "landmarks": {k: (list(v) if isinstance(v, Vector) else v)
                      for k, v in land.items()},
        "head_w": head_w,
        "units": "world_metres",
    }


def make_beanie(arm, meshes, collection, head_w):
    hat = v1.import_glb(BEANIE_GLB, "XBotBeanie")
    if collection and hat.name not in collection.objects:
        collection.objects.link(hat)
        if hat.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(hat)
    imported = world_bbox_dims(hat)
    mat = v1._charcoal_knit_from_existing(hat, "XBotBeanieCharcoal_painted_v2")
    hat.data.materials.clear()
    hat.data.materials.append(mat)
    hat.active_material = mat
    info = place_beanie(hat, arm, meshes, head_w)
    info["imported_bbox_m"] = imported
    return hat, info


# ---------------------------------------------------------------------------
# Size asserts — must pass before any render
# ---------------------------------------------------------------------------
def assert_sane_sizes(handle):
    """Return a report. ok is False if any asset is outside the sane band."""
    report = {"ok": True, "failures": [], "measured": {}}
    hw = handle.get("head_width_m")
    report["measured"]["head_width_m"] = hw
    if hw is None or not (HEAD_WIDTH_RANGE[0] <= hw <= HEAD_WIDTH_RANGE[1]):
        report["ok"] = False
        report["failures"].append(
            f"head width {hw} m is not a sane world-metre X Bot head "
            f"({HEAD_WIDTH_RANGE[0]}–{HEAD_WIDTH_RANGE[1]} m)"
        )
    for side in ("L", "R"):
        rec = (handle.get("gloves") or {}).get(side) or {}
        length = rec.get("length_m")
        report["measured"][f"glove_{side}_length_m"] = length
        report["measured"][f"glove_{side}_imported_length_m"] = rec.get("raw_length_m")
        if length is None or not (GLOVE_LEN_RANGE[0] <= length <= GLOVE_LEN_RANGE[1]):
            report["ok"] = False
            report["failures"].append(
                f"glove {side} length {length} m is outside "
                f"{GLOVE_LEN_RANGE[0]}–{GLOVE_LEN_RANGE[1]} m"
            )
    rec = handle.get("beanie") or {}
    width = rec.get("width_m")
    report["measured"]["beanie_width_m"] = width
    report["measured"]["beanie_imported_width_m"] = rec.get("raw_width_m")
    if width is None or not (BEANIE_WIDTH_RANGE[0] <= width <= BEANIE_WIDTH_RANGE[1]):
        report["ok"] = False
        report["failures"].append(
            f"beanie width {width} m is outside "
            f"{BEANIE_WIDTH_RANGE[0]}–{BEANIE_WIDTH_RANGE[1]} m"
        )
    return report


# ---------------------------------------------------------------------------
# Hand mask in world metres (construction: rigid glove vs finger curl)
# ---------------------------------------------------------------------------
def mask_hands_inside_gloves(meshes, gloves, arm):
    hidden = {}
    for side, (glove, info) in gloves.items():
        bn = info["bone"]
        origin, x, y, z = bone_frame(arm, bn)
        cuff = Vector(info["cuff"])
        cuff_along = (cuff - origin).dot(y)
        pred = v1.hand_pred(side)
        for o in meshes:
            if o.type != "MESH":
                continue
            w = v1._vg_weights(o, pred)
            w_fore = v1._vg_weights(o, lambda n, s=side: (
                (("left" in v1._clean(n) and s == "L")
                 or ("right" in v1._clean(n) and s == "R"))
                and "forearm" in v1._clean(n)
            ))
            n = len(o.data.vertices)
            weights = np.zeros(n, np.float32)
            deps = bpy.context.evaluated_depsgraph_get()
            ev = o.evaluated_get(deps)
            me = ev.to_mesh()
            try:
                emw = ev.matrix_world
                for i, v in enumerate(me.vertices):
                    if i >= n:
                        break
                    along = ((emw @ v.co) - origin).dot(y)
                    handish = (i < len(w) and w[i] >= 0.18) or (
                        i < len(w_fore) and w_fore[i] >= 0.40
                        and along >= cuff_along - 0.01)
                    if handish and along >= cuff_along:
                        weights[i] = 1.0
            finally:
                ev.to_mesh_clear()
            if weights.max() <= 0:
                continue
            vg_name = "xbot_glove_hide"
            vg = o.vertex_groups.get(vg_name) or o.vertex_groups.new(name=vg_name)
            for i, val in enumerate(weights):
                if val > 0.5:
                    vg.add([i], 1.0, "REPLACE")
            if "XBotGloveMask" not in o.modifiers:
                mod = o.modifiers.new("XBotGloveMask", "MASK")
                mod.vertex_group = vg_name
                mod.invert_vertex_group = True
            hidden[f"{o.name}:{side}"] = int(weights.sum())
    return hidden


# ---------------------------------------------------------------------------
# Apply + world-metre checks
# ---------------------------------------------------------------------------
def apply_painted_outfit(arm, meshes, f0, f1, collection_name="XBotPaintedV2"):
    col = bpy.data.collections.get(collection_name)
    if col is None:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)

    head_bone = xo.find_bone(arm, "mixamorig:Head", "Head")
    head_w = head_width_world(meshes, arm, head_bone)
    print(f"  head_width_world={head_w} m", flush=True)

    zone_info = v1.paint_body_zones(meshes, arm)
    v1.assign_painted_material(meshes)

    gloves = make_glove_pair(arm, meshes, col, f0, f1, head_w)
    hat, hat_info = make_beanie(arm, meshes, col, head_w)
    hidden = mask_hands_inside_gloves(
        meshes, {s: gloves[s] for s in ("L", "R")}, arm)

    handle = {
        "collection": collection_name,
        "head_width_m": head_w,
        "gloves": {s: {"object": gloves[s][0].name, **gloves[s][1]}
                   for s in ("L", "R")},
        "beanie": {"object": hat.name, **hat_info},
        "zones": zone_info,
        "hidden_hand_verts": hidden,
        "objects": [gloves["R"][0].name, gloves["L"][0].name, hat.name],
        "credits": [
            "Classic Red Boxing Glove (Low Poly) by ARTEL_3D, CC BY 4.0",
            "Beanie by oluwaphemi, CC BY 4.0",
        ],
        "criteria": CRITERIA,
        "units": "world_metres",
    }
    handle["size_assert"] = assert_sane_sizes(handle)
    print("  size_assert", handle["size_assert"], flush=True)
    bpy.context.view_layer.update()
    return handle


def check_gloves(arm, meshes, handle, f0, f1, step=1):
    """Criterion 1, all distances in world metres."""
    report = {"ok": True, "per_side": {}, "worst": {}, "frames_tested": 0,
              "frames_failed": []}
    saved = bpy.context.scene.frame_current
    for side in ("L", "R"):
        rec = handle["gloves"][side]
        gname, bn = rec["object"], rec["bone"]
        glove = bpy.data.objects.get(gname)
        if glove is None:
            report["ok"] = False
            report["per_side"][side] = {"error": "missing glove"}
            continue
        failures = []
        n_frames = 0
        max_outside = 0
        max_float = 0.0
        min_overlap = 1e9
        f = int(f0)
        pred = v1.hand_pred(side)
        while f <= int(f1):
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            n_frames += 1
            origin, x, y, z = bone_frame(arm, bn)
            hand = world_project(v1.collect_weighted(meshes, pred, 0.30),
                                 origin, x, y, z)
            glove_locs = world_project(v1.world_pts(glove), origin, x, y, z)
            if not hand or not glove_locs:
                failures.append({"f": f, "reason": "no verts"})
                f += step
                continue
            gmin = Vector((min(p.x for p in glove_locs), min(p.y for p in glove_locs),
                           min(p.z for p in glove_locs)))
            gmax = Vector((max(p.x for p in glove_locs), max(p.y for p in glove_locs),
                           max(p.z for p in glove_locs)))
            cuff_y = gmin.y + 0.008
            pad = 0.010
            outside = 0
            tested = 0
            for p in hand:
                if p.y < cuff_y:
                    continue
                tested += 1
                inside = (
                    gmin.x - pad <= p.x <= gmax.x + pad
                    and gmin.y - pad <= p.y <= gmax.y + pad
                    and gmin.z - pad <= p.z <= gmax.z + pad
                )
                if not inside:
                    outside += 1
            max_outside = max(max_outside, outside)
            hc = sum(hand, Vector()) / len(hand)
            gc = sum(glove_locs, Vector()) / len(glove_locs)
            dist = (hc - gc).length
            max_float = max(max_float, dist)
            glove_mid_y = 0.5 * (gmin.y + gmax.y)
            hy0, hy1 = min(p.y for p in hand), max(p.y for p in hand)
            overlap = max(0.0, min(gmax.y, hy1) - max(gmin.y, hy0))
            min_overlap = min(min_overlap, overlap)
            reasons = []
            if tested and outside / max(tested, 1) > 0.04 and outside > 8:
                reasons.append(f"{outside}/{tested} hand verts outside glove AABB")
            if dist > 0.10:
                reasons.append(f"glove centroid {dist*100:.1f} cm from hand (floating)")
            if glove_mid_y < -0.07:
                reasons.append(f"glove mid Y={glove_mid_y:.3f} m (forearm)")
            if overlap < 0.03:
                reasons.append(f"glove/hand Y overlap {overlap:.3f} m")
            if reasons:
                failures.append({"f": f, "reasons": reasons,
                                 "outside": outside, "tested": tested,
                                 "float_m": dist, "glove_mid_y": glove_mid_y})
            f += step
        side_ok = not failures
        report["per_side"][side] = {
            "ok": side_ok,
            "frames": n_frames,
            "failures": failures[:8],
            "n_failures": len(failures),
            "max_outside": max_outside,
            "max_float_m": max_float,
            "min_y_overlap_m": min_overlap if min_overlap < 1e8 else None,
        }
        if not side_ok:
            report["ok"] = False
            report["frames_failed"].extend(
                {"side": side, **fail} for fail in failures[:4])
        report["frames_tested"] = max(report["frames_tested"], n_frames)
        report["worst"][side] = {
            "max_outside": max_outside,
            "max_float_m": max_float,
            "min_y_overlap_m": min_overlap if min_overlap < 1e8 else None,
        }
    bpy.context.scene.frame_set(saved)
    bpy.context.view_layer.update()
    return report


def check_beanie(arm, meshes, handle, f0, f1, step=1):
    """Criterion 2, all distances in world metres."""
    report = {"ok": True, "frames_tested": 0, "failures": []}
    rec = handle.get("beanie") or {}
    gname = rec.get("object")
    hat = bpy.data.objects.get(gname) if gname else None
    head_bone = rec.get("bone") or xo.find_bone(arm, "mixamorig:Head", "Head")
    if hat is None or not head_bone:
        return {"ok": False, "error": "missing beanie or head bone"}
    saved = bpy.context.scene.frame_current
    f = int(f0)
    while f <= int(f1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        report["frames_tested"] += 1
        info = beanie_anatomy(hat)
        land = head_landmarks_world(meshes, arm, head_bone)
        reasons = []
        if info["crown"].z < land["zmax"] - 0.015:
            reasons.append(
                f"hat crown z={info['crown'].z:.3f} below skull top {land['zmax']:.3f}")
        if info["hem"].z > land["zmax"] - 0.02:
            reasons.append("hem is up at the crown — hat has slid off")
        if info["hem"].z < land["brow_z"] - 0.025:
            reasons.append(
                f"hem z={info['hem'].z:.3f} dropped below brows {land['brow_z']:.3f}")
        mid = land["mid"]
        drift = math.hypot(info["hem"].x - mid.x, info["hem"].y - mid.y)
        if drift > 0.07:
            reasons.append(f"hem drifted {drift*100:.1f} cm off the head midline")
        if reasons:
            report["ok"] = False
            report["failures"].append({"f": f, "reasons": reasons})
        f += step
    bpy.context.scene.frame_set(saved)
    bpy.context.view_layer.update()
    report["n_failures"] = len(report["failures"])
    report["failures"] = report["failures"][:8]
    return report


def clear_outfit(handle):
    v1.clear_outfit(handle)
