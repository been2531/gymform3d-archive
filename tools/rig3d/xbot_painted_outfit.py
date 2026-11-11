#!/usr/bin/env python3
"""Painted X Bot outfit (v1) — one honest attempt, no patching.

Pass criteria (written before any render; a miss is a fail, not a cue to iterate):
  (1) Gloves enclose the fist on every frame of all 4 v4 punches,
      with no floating and no forearm placement.
  (2) The beanie stays on the crown on every frame.
  (3) Paint zones are clean and symmetric, with no tearing
      (by construction: attributes on the body, no garment meshes).
  (4) boxing_qc motion numbers are identical to bare X Bot.

Outfit:
  * Gloves — ARTEL_3D Classic Red Boxing Glove (CC BY 4.0), mirrored pair,
    glossy black, rigid-parented to mixamorig:LeftHand / RightHand.
  * Beanie — oluwaphemi Rib Beanie (CC BY 4.0), charcoal, rigid-parented
    to mixamorig:Head, hem just above the brows / ears (not a helmet).
  * Trunks + tank — no garment meshes. Rest-pose vertex zones on the X Bot
    surface (attribute "ozone") so the paint deforms with the skin.
  * Existing mcol glow composites over the painted base.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

import xbot_outfit as xo  # noqa: E402

ASSETS = os.path.join(ROOT, "output/_assets/outfit_candidates")
GLOVE_GLB = os.path.join(ASSETS, "gloves/artel_classic_red_glove/model.glb")
BEANIE_GLB = os.path.join(ASSETS, "beanie/oluwaphemi_beanie/model.glb")

# Linear colours (AgX-safe). Trunks burgundy is the locked outfit decision.
CLAY = (0.65, 0.656, 0.664)
CHARCOAL = (0.070, 0.068, 0.072)
BURGUNDY = (0.18, 0.025, 0.04)
GLOVE_BLACK = xo.GLOVE_V4_BLACK

# ozone ids written on the body and read by the body material.
ZONE_SKIN = 0.0
ZONE_TANK = 1.0
ZONE_TRUNKS = 2.0
ZONE_BAND = 3.0

# Construction clearances, metres (world, after scale_ground_center).
GLOVE_SIDE_MARGIN = 0.012      # leather thickness past the hand AABB
GLOVE_TIP_PAD = 0.018          # past the fingertips
GLOVE_CUFF_PAST_WRIST = 0.030  # cuff extends this far behind the wrist (Y=0)
GLOVE_CUFF_OPENING_FACTOR = 1.10  # opening vs measured wrist radius
BEANIE_HEM_ABOVE_BROW = 0.010
BEANIE_CROWN_CLEARANCE = 0.012
BEANIE_RADIAL_CLEARANCE = 0.008
WAISTBAND_H = 0.038

CRITERIA = {
    "version": "xbot_painted_v1",
    "rule": "ONE honest attempt. A missed criterion is a fail; do not iterate.",
    "criteria": {
        "1_gloves": (
            "The gloves enclose the fist on every frame of all 4 punches, "
            "with no floating and no forearm placement."
        ),
        "2_beanie": "The beanie stays on the crown on every frame.",
        "3_paint": (
            "The paint zones are clean and symmetric, with no tearing "
            "(by construction: body-surface attributes, no garment meshes)."
        ),
        "4_motion": "The boxing_qc motion numbers are identical to bare X Bot.",
    },
}


def _clean(name: str) -> str:
    return (name.lower().replace("mixamorig:", "").replace("mixamorig_", "")
            .replace(":", "").replace(".", ""))


def _sel(*objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        if o is not None:
            o.select_set(True)
            bpy.context.view_layer.objects.active = o


def _apply_transforms(obj, loc=False, rot=True, scale=True):
    _sel(obj)
    bpy.ops.object.transform_apply(location=loc, rotation=rot, scale=scale)
    bpy.context.view_layer.update()


def _shade_smooth(obj):
    mesh = obj.data
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = math.radians(60)
    for p in mesh.polygons:
        p.use_smooth = True


def _join(meshes, name):
    meshes = [m for m in meshes if m and m.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no meshes to join for {name}")
    if len(meshes) == 1:
        meshes[0].name = name
        return meshes[0]
    _sel(*meshes)
    bpy.ops.object.join()
    out = bpy.context.view_layer.objects.active
    out.name = name
    return out


def import_glb(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    bpy.context.view_layer.update()
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh in {path}")
    obj = _join(meshes, name)
    # Drop leftover empties / armature from the glTF scene.
    for o in list(new):
        if o != obj and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    _apply_transforms(obj, loc=False, rot=True, scale=True)
    _shade_smooth(obj)
    return obj


def world_pts(obj):
    mw = obj.matrix_world
    return [mw @ v.co for v in obj.data.vertices]


def rotate_around(obj, pivot, quat):
    T = Matrix.Translation(pivot)
    R = quat.to_matrix().to_4x4()
    obj.matrix_world = T @ R @ T.inverted() @ obj.matrix_world
    bpy.context.view_layer.update()


def bone_mw(arm, bone_name):
    pb = arm.pose.bones[bone_name]
    return arm.matrix_world @ pb.matrix


def bone_axes(arm, bone_name):
    mw = bone_mw(arm, bone_name)
    origin = mw.translation.copy()
    x = (mw.to_3x3() @ Vector((1, 0, 0))).normalized()
    y = (mw.to_3x3() @ Vector((0, 1, 0))).normalized()
    z = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
    return origin, x, y, z


# ---------------------------------------------------------------------------
# Glove anatomy (object space, after import)
# ---------------------------------------------------------------------------
def glove_anatomy(obj):
    """Cuff end, fist end, long axis, thumb side, cuff opening radius.

    The ARTEL glove stands along +Z at import (cuff near Z=0, fist at Z~16 cm).
    We do not trust open-edge loops: the mesh has thousands of non-manifold
    seams. The cuff is the long-axis end with the smaller radial spread.
    The thumb is the largest lateral lobe off that axis.
    """
    pts = world_pts(obj)
    arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
    c = arr.mean(axis=0)
    cov = np.cov((arr - c).T)
    w, V = np.linalg.eigh(cov)
    axis = Vector(V[:, int(np.argmax(w))].tolist()).normalized()
    proj = np.array([(p - Vector(c)).dot(axis) for p in pts])
    lo, hi = float(proj.min()), float(proj.max())
    # Radial RMS of the extreme 15% at each end.
    def end_radius(mask):
        subset = arr[mask]
        if len(subset) < 8:
            return 1e9, Vector(c)
        cen = subset.mean(axis=0)
        # distance to the axis through centroid
        rel = subset - c
        ax = np.array((axis.x, axis.y, axis.z))
        rad = np.linalg.norm(rel - np.outer(rel @ ax, ax), axis=1)
        return float(np.sqrt(np.mean(rad * rad))), Vector(cen.tolist())

    span = hi - lo
    r_lo, c_lo = end_radius(proj <= lo + 0.15 * span)
    r_hi, c_hi = end_radius(proj >= hi - 0.15 * span)
    if r_lo <= r_hi:
        cuff, fist, fist_dir = c_lo, c_hi, axis if (hi > 0 or True) else axis
        # axis points from centroid toward the high-proj end. Cuff is low-proj,
        # so fist direction is +axis (from cuff toward fist) only if cuff is lo.
        fist_dir = axis
        cuff_r = r_lo
    else:
        cuff, fist, fist_dir = c_hi, c_lo, -axis
        cuff_r = r_hi
    # Recompute cuff as the mean of verts near the cuff end along fist_dir.
    t = np.array([(p - cuff).dot(fist_dir) for p in pts])
    tmin, tmax = float(t.min()), float(t.max())
    cuff_mask = t <= tmin + 0.10 * (tmax - tmin)
    cuff = Vector(arr[cuff_mask].mean(axis=0).tolist())
    # Thumb: largest one-sided lateral cluster.
    rel = arr - np.array((cuff.x, cuff.y, cuff.z))
    ax = np.array((fist_dir.x, fist_dir.y, fist_dir.z))
    lateral = rel - np.outer(rel @ ax, ax)
    # Mid-body slice (avoid cuff and very tip).
    mid = (t > tmin + 0.25 * (tmax - tmin)) & (t < tmax - 0.10 * (tmax - tmin))
    if mid.sum() < 16:
        mid = np.ones(len(pts), dtype=bool)
    lat = lateral[mid]
    # Find the direction of the 4% farthest lateral verts; that is the thumb.
    dist = np.linalg.norm(lat, axis=1)
    k = max(8, int(0.04 * len(dist)))
    idx = np.argpartition(dist, -k)[-k:]
    thumb = Vector(lat[idx].mean(axis=0).tolist())
    if thumb.length < 1e-8:
        thumb = fist_dir.orthogonal()
    thumb.normalize()
    # Opening radius: mean radial size of the cuff-end ring.
    cuff_lat = np.linalg.norm(lateral[cuff_mask], axis=1)
    opening = float(np.percentile(cuff_lat, 75)) if len(cuff_lat) else cuff_r
    length = (fist - cuff).length
    return {
        "cuff": cuff,
        "fist": fist,
        "fist_dir": fist_dir.normalized(),
        "thumb_dir": thumb,
        "opening_r": opening,
        "length": length,
        "cuff_r_rms": cuff_r,
    }


def hand_pred(side):
    def pred(n, s=side):
        c = _clean(n)
        if s == "L" and "left" not in c:
            return False
        if s == "R" and "right" not in c:
            return False
        return any(k in c for k in (
            "hand", "thumb", "index", "middle", "ring", "pinky"))
    return pred


def collect_weighted(meshes, pred, thresh=0.25):
    return list(xo.iter_weighted_world_verts(meshes, pred, thresh=thresh))


def aabb_of(pts):
    if not pts:
        return None
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    return {
        "min": Vector((min(xs), min(ys), min(zs))),
        "max": Vector((max(xs), max(ys), max(zs))),
        "n": len(pts),
    }


def scan_hand_local(arm, meshes, bone_name, side, f0, f1, step=2):
    """Hand verts in bone-local metres across the clip."""
    saved = bpy.context.scene.frame_current
    acc = None
    f = int(f0)
    pred = hand_pred(side)
    while f <= int(f1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        inv = bone_mw(arm, bone_name).inverted()
        pts = [inv @ p for p in collect_weighted(meshes, pred)]
        box = aabb_of(pts)
        if box:
            if acc is None:
                acc = box
            else:
                acc = {
                    "min": Vector((min(acc["min"][i], box["min"][i]) for i in range(3))),
                    "max": Vector((max(acc["max"][i], box["max"][i]) for i in range(3))),
                    "n": acc["n"] + box["n"],
                }
        f += step
    bpy.context.scene.frame_set(saved)
    bpy.context.view_layer.update()
    return acc


def wrist_radius_m(arm, meshes, bone_name, side):
    inv = bone_mw(arm, bone_name).inverted()
    rs = []
    for wp in collect_weighted(meshes, hand_pred(side), thresh=0.25):
        loc = inv @ wp
        if -0.035 <= loc.y <= 0.010:
            rs.append(math.hypot(loc.x, loc.z))
    if not rs:
        return 0.033
    rs.sort()
    return float(rs[int(len(rs) * 0.85)])


def head_width_m(meshes, arm, head_bone):
    pts = list(xo._head_points(meshes, arm, head_bone, thresh=0.30))
    if not pts:
        return 0.176
    return max(p.x for p in pts) - min(p.x for p in pts)


def _align_glove_to_bone(obj, info, origin, y_axis, z_axis):
    """Rotate so fist_dir → bone Y, thumb_dir → bone Z, pivot at cuff."""
    src_f = info["fist_dir"]
    dst_f = y_axis.normalized()
    if src_f.length > 1e-8 and dst_f.length > 1e-8:
        if src_f.dot(dst_f) < -0.999:
            pivot = info["cuff"]
            ortho = src_f.orthogonal().normalized()
            rotate_around(obj, pivot, (src_f.rotation_difference(ortho)
                                       if False else src_f.rotation_difference(dst_f)))
        else:
            rotate_around(obj, info["cuff"], src_f.rotation_difference(dst_f))
    # Refresh directions after the first rotation.
    info2 = glove_anatomy(obj)
    src_t = info2["thumb_dir"]
    # Project both onto the plane perpendicular to the (now aligned) fist dir.
    fwd = info2["fist_dir"]
    src_t = src_t - fwd * src_t.dot(fwd)
    dst_t = z_axis - fwd * z_axis.dot(fwd)
    if src_t.length > 1e-8 and dst_t.length > 1e-8:
        src_t.normalize()
        dst_t.normalize()
        rotate_around(obj, info2["cuff"], src_t.rotation_difference(dst_t))
    return glove_anatomy(obj)


def place_glove(obj, arm, bone_name, side, meshes, env, head_w):
    """Rigid-place one glove in world, then bone-parent with keep_world."""
    origin, bx, by, bz = bone_axes(arm, bone_name)
    info = glove_anatomy(obj)
    info = _align_glove_to_bone(obj, info, origin, by, bz)

    # Bone-local AABB of the aligned, unscaled glove.
    inv = bone_mw(arm, bone_name).inverted()
    locs = [inv @ p for p in world_pts(obj)]
    gmin = Vector((min(p.x for p in locs), min(p.y for p in locs), min(p.z for p in locs)))
    gmax = Vector((max(p.x for p in locs), max(p.y for p in locs), max(p.z for p in locs)))
    gw = max(gmax.x - gmin.x, 1e-4)
    gl = max(gmax.y - gmin.y, 1e-4)
    gt = max(gmax.z - gmin.z, 1e-4)

    if env is None:
        env = {"min": Vector((-0.055, -0.030, -0.035)),
               "max": Vector((0.055, 0.110, 0.065)), "n": 0}
    emin, emax = env["min"], env["max"]
    need_w = max(emax.x - emin.x + 2 * GLOVE_SIDE_MARGIN, 0.80 * head_w)
    need_t = (emax.z - emin.z) + 2 * GLOVE_SIDE_MARGIN
    need_l = (emax.y + GLOVE_TIP_PAD) - (min(emin.y, 0.0) - GLOVE_CUFF_PAST_WRIST)
    wrist_r = wrist_radius_m(arm, meshes, bone_name, side)
    need_cuff = wrist_r * GLOVE_CUFF_OPENING_FACTOR
    opening = max(info["opening_r"], 1e-4)

    scale = max(need_w / gw, need_t / gt, need_l / gl, need_cuff / opening)
    # Uniform scale about the cuff so the opening stays at the cuff end.
    cuff = info["cuff"]
    T = Matrix.Translation(cuff)
    obj.matrix_world = T @ Matrix.Scale(scale, 4) @ T.inverted() @ obj.matrix_world
    bpy.context.view_layer.update()
    info = glove_anatomy(obj)

    # Translate: cuff sits GLOVE_CUFF_PAST_WRIST behind the wrist along -Y.
    target_cuff = origin - by.normalized() * GLOVE_CUFF_PAST_WRIST
    obj.matrix_world.translation += (target_cuff - info["cuff"])
    bpy.context.view_layer.update()
    xo.parent_to_bone(obj, arm, bone_name, keep_world=True)
    bpy.context.view_layer.update()
    info = glove_anatomy(obj)
    return {
        "bone": bone_name,
        "side": side,
        "scale": scale,
        "opening_r": info["opening_r"],
        "wrist_r": wrist_r,
        "need": {"w": need_w, "t": need_t, "l": need_l, "cuff": need_cuff},
        "env": {"min": list(emin), "max": list(emax), "n": env.get("n", 0)},
        "cuff": list(info["cuff"]),
        "fist": list(info["fist"]),
        "length": info["length"],
        "head_w": head_w,
    }


def make_glove_pair(arm, meshes, collection, f0, f1):
    src = import_glb(GLOVE_GLB, "XBotGlove_R")
    if collection:
        if src.name not in collection.objects:
            collection.objects.link(src)
            if src.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(src)
    mat = xo.make_glossy_leather_material("XBotGloveLeather_painted_v1",
                                          GLOVE_BLACK, roughness=0.30)
    src.data.materials.clear()
    src.data.materials.append(mat)
    src.active_material = mat
    mat.use_backface_culling = False

    head_bone = xo.find_bone(arm, "mixamorig:Head", "Head")
    head_w = head_width_m(meshes, arm, head_bone)
    bones = {
        "R": xo.find_bone(arm, "mixamorig:RightHand", "RightHand"),
        "L": xo.find_bone(arm, "mixamorig:LeftHand", "LeftHand"),
    }
    envs = {}
    for side, bn in bones.items():
        envs[side] = scan_hand_local(arm, meshes, bn, side, f0, f1, step=2)

    # Duplicate for the left glove *before* placing, then place each in its
    # own bone space. A world-X mirror is the wrong move: Mixamo Left/Right
    # Hand bones already share a local convention (Y along the fingers, +Z
    # toward the thumb), so the same mesh, aligned in each bone's local
    # frame, is already a matched pair.
    _sel(src)
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

    info_r = place_glove(src, arm, bones["R"], "R", meshes, envs["R"], head_w)
    info_l = place_glove(left, arm, bones["L"], "L", meshes, envs["L"], head_w)
    return {"R": (src, info_r), "L": (left, info_l), "head_w": head_w}


# ---------------------------------------------------------------------------
# Beanie
# ---------------------------------------------------------------------------
def beanie_anatomy(obj):
    """Hem plane, crown, up axis. The imported hat already stands along +Z."""
    pts = world_pts(obj)
    zs = [p.z for p in pts]
    zmin, zmax = min(zs), max(zs)
    span = max(zmax - zmin, 1e-6)
    # Hem = lowest 12% (folded cuff). Crown = highest 8% (pinched bun).
    hem_pts = [p for p in pts if p.z <= zmin + 0.12 * span]
    crown_pts = [p for p in pts if p.z >= zmax - 0.08 * span]
    hem = sum(hem_pts, Vector()) / max(1, len(hem_pts))
    crown = sum(crown_pts, Vector()) / max(1, len(crown_pts))
    up = crown - hem
    if up.length < 1e-6:
        up = Vector((0, 0, 1))
    else:
        up.normalize()
    # Radial size of the hem ring about the up axis.
    rads = []
    for p in hem_pts:
        d = p - hem
        rads.append((d - up * d.dot(up)).length)
    hem_r = float(np.percentile(rads, 80)) if rads else 0.10
    height = (crown - hem).length
    # Slouch: horizontal offset of the mid/upper centroid from the hem centre.
    mid_pts = [p for p in pts if p.z >= zmin + 0.40 * span]
    mid = sum(mid_pts, Vector()) / max(1, len(mid_pts))
    slouch = mid - hem
    slouch = slouch - up * slouch.dot(up)
    return {
        "hem": hem, "crown": crown, "up": up, "hem_r": hem_r,
        "height": height, "slouch": slouch,
        "zmin": zmin, "zmax": zmax,
    }


def _head_landmarks(meshes, arm, head_bone):
    """Brow height, ear height, crown, midline, head radius at brow — current pose."""
    pts = list(xo._head_points(meshes, arm, head_bone, thresh=0.28))
    if not pts:
        origin, _, y, _ = bone_axes(arm, head_bone)
        return {
            "brow_z": origin.z + 0.06, "ear_z": origin.z + 0.05,
            "crown": origin + y * 0.22, "mid": origin,
            "radius": 0.09, "front": Vector((0, -1, 0)),
        }
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    zmin, zmax = min(zs), max(zs)
    mid = Vector(((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5,
                  (zmin + zmax) * 0.5))
    # Character faces -Y after the Mixamo importer.
    front = Vector((0, -1, 0))
    # Brows: front half of the head, about 60% up from chin to crown.
    brow_z = zmin + 0.60 * (zmax - zmin)
    front_pts = [p for p in pts if (p - mid).dot(front) > 0]
    if front_pts:
        # Take verts near that height (2 cm band) and use their mean Z.
        band = [p for p in front_pts if abs(p.z - brow_z) < 0.025]
        if band:
            brow_z = sum(p.z for p in band) / len(band)
    # Ears: sides, similar height. Use a slightly lower bound so the hem
    # clears both brows and ears.
    ear_z = zmin + 0.55 * (zmax - zmin)
    side_pts = [p for p in pts if abs(p.x - mid.x) > 0.35 * (max(xs) - min(xs)) * 0.5]
    if side_pts:
        # ear height ≈ max z of the side cluster's lower-mid band
        ear_z = min(max(p.z for p in side_pts) * 0.0 + ear_z, brow_z)
    hem_z = max(brow_z, ear_z) + BEANIE_HEM_ABOVE_BROW
    crown_pt = Vector((mid.x, mid.y, zmax + BEANIE_CROWN_CLEARANCE))
    # Head radius in the horizontal plane at hem height.
    slice_pts = [p for p in pts if abs(p.z - hem_z) < 0.02]
    if not slice_pts:
        slice_pts = pts
    rads = [math.hypot(p.x - mid.x, p.y - mid.y) for p in slice_pts]
    radius = float(np.percentile(rads, 90)) if rads else 0.09
    return {
        "brow_z": brow_z, "ear_z": ear_z, "hem_z": hem_z,
        "crown": crown_pt, "mid": Vector((mid.x, mid.y, hem_z)),
        "radius": radius, "front": front, "zmax": zmax, "zmin": zmin,
    }


def place_beanie(obj, arm, meshes, collection):
    head_bone = xo.find_bone(arm, "mixamorig:Head", "Head")
    if not head_bone:
        raise RuntimeError("no Head bone")
    info = beanie_anatomy(obj)
    land = _head_landmarks(meshes, arm, head_bone)

    # Up: align hat up to world +Z (already close) so the hem is horizontal.
    world_up = Vector((0, 0, 1))
    if info["up"].dot(world_up) < 0.999:
        rotate_around(obj, info["hem"], info["up"].rotation_difference(world_up))
        info = beanie_anatomy(obj)

    # Yaw: send any slouch toward the back of the head (+Y, since the figure
    # faces -Y). A watch cap slumps behind the crown, not over the face.
    slouch = Vector((info["slouch"].x, info["slouch"].y, 0.0))
    back = Vector((0, 1, 0))
    if slouch.length > 1e-4:
        slouch.normalize()
        # Rotation around +Z from slouch to back.
        ang = math.atan2(slouch.x * back.y - slouch.y * back.x,
                         slouch.dot(back))
        # rotation_difference on horizontal vectors
        rotate_around(obj, info["hem"], slouch.rotation_difference(back))
        info = beanie_anatomy(obj)

    # Uniform scale so the hem ring sits just outside the skull at hem height.
    target_r = land["radius"] + BEANIE_RADIAL_CLEARANCE
    scale = target_r / max(info["hem_r"], 1e-4)
    # Also keep the crown from falling short of the skull top, and from
    # becoming a helmet (height much larger than the upper-head).
    need_h = (land["crown"].z - land["hem_z"])
    if info["height"] > 1e-4:
        scale = max(scale, need_h / info["height"])
    hem = info["hem"]
    T = Matrix.Translation(hem)
    obj.matrix_world = T @ Matrix.Scale(scale, 4) @ T.inverted() @ obj.matrix_world
    bpy.context.view_layer.update()
    info = beanie_anatomy(obj)

    # Sit the hem on the brow/ear line, centred on the head midline.
    target_hem = Vector((land["mid"].x, land["mid"].y, land["hem_z"]))
    obj.matrix_world.translation += (target_hem - info["hem"])
    bpy.context.view_layer.update()
    xo.parent_to_bone(obj, arm, head_bone, keep_world=True)
    bpy.context.view_layer.update()
    info = beanie_anatomy(obj)
    return {
        "bone": head_bone,
        "scale": scale,
        "hem": list(info["hem"]),
        "crown": list(info["crown"]),
        "hem_r": info["hem_r"],
        "height": info["height"],
        "landmarks": {k: (list(v) if isinstance(v, Vector) else v)
                      for k, v in land.items()},
    }


def make_beanie(arm, meshes, collection):
    hat = import_glb(BEANIE_GLB, "XBotBeanie")
    if collection and hat.name not in collection.objects:
        collection.objects.link(hat)
        if hat.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(hat)
    # Recolour charcoal; keep the asset's non-colour maps (rib normal / roughness).
    mat = _charcoal_knit_from_existing(hat, "XBotBeanieCharcoal_painted_v1")
    hat.data.materials.clear()
    hat.data.materials.append(mat)
    hat.active_material = mat
    info = place_beanie(hat, arm, meshes, collection)
    return hat, info


def _charcoal_knit_from_existing(obj, name):
    """Charcoal albedo, reuse any non-colour maps already on the mesh."""
    src = obj.active_material or (obj.data.materials[0] if obj.data.materials else None)
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (280, 0)
    bsdf.inputs["Base Color"].default_value = (*CHARCOAL, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.82
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.18
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    # Steal non-colour image textures from the source material (normal / rough).
    if src and src.use_nodes:
        stolen = []
        for n in src.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None:
                stolen.append(n)
        # Prefer images flagged Non-Color.
        noncolor = [n for n in stolen
                    if n.image.colorspace_settings.name not in {"sRGB", "Filmic sRGB"}]
        use = noncolor or stolen
        texcoord = nt.nodes.new("ShaderNodeTexCoord")
        texcoord.location = (-520, -40)
        y = 80
        for i, src_n in enumerate(use[:2]):
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.location = (-280, y - 220 * i)
            tex.image = src_n.image
            tex.image.colorspace_settings.name = "Non-Color"
            nt.links.new(texcoord.outputs["UV"], tex.inputs["Vector"])
            # First non-colour map → normal, second → roughness if it looks like it.
            if i == 0:
                nrm = nt.nodes.new("ShaderNodeNormalMap")
                nrm.location = (40, -160)
                nrm.inputs["Strength"].default_value = 0.55
                nt.links.new(tex.outputs["Color"], nrm.inputs["Color"])
                if "Normal" in bsdf.inputs:
                    nt.links.new(nrm.outputs["Normal"], bsdf.inputs["Normal"])
            else:
                # Map to roughness around charcoal knit (0.7–0.9).
                ramp = nt.nodes.new("ShaderNodeMapRange")
                ramp.location = (40, -400)
                ramp.inputs["To Min"].default_value = 0.70
                ramp.inputs["To Max"].default_value = 0.90
                nt.links.new(tex.outputs["Color"], ramp.inputs["Value"])
                if "Roughness" in bsdf.inputs:
                    nt.links.new(ramp.outputs["Result"], bsdf.inputs["Roughness"])
    mat.use_backface_culling = False
    return mat


# ---------------------------------------------------------------------------
# Paint zones (rest-pose attributes on the body)
# ---------------------------------------------------------------------------
def _vg_weights(obj, pred):
    n = len(obj.data.vertices)
    w = np.zeros(n, np.float32)
    for vg in obj.vertex_groups:
        if not pred(vg.name):
            continue
        idx = vg.index
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == idx and g.weight > w[v.index]:
                    w[v.index] = g.weight
    return w


def paint_body_zones(meshes, arm):
    """Write ozone / pcol in REST pose. Returns zone counts."""
    saved = arm.data.pose_position
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()

    # Landmarks from the largest body mesh.
    body = max((o for o in meshes if "joint" not in o.name.lower()),
               key=lambda o: len(o.data.vertices), default=None)
    if body is None:
        body = meshes[0]
    deps = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        mw = ev.matrix_world
        pts = [mw @ v.co for v in me.vertices]
    finally:
        ev.to_mesh_clear()

    def pts_w(pred, thresh=0.30):
        w = _vg_weights(body, pred)
        return [p for i, p in enumerate(pts) if i < len(w) and w[i] >= thresh]

    def z_span(plist):
        if not plist:
            return None, None
        zs = [p.z for p in plist]
        return min(zs), max(zs)

    hips_z = z_span(pts_w(lambda n: any(k in _clean(n) for k in ("hips", "pelvis"))))
    upleg_z = z_span(pts_w(lambda n: "upleg" in _clean(n)))
    head_z = z_span(pts_w(lambda n: "head" in _clean(n)))
    # Neck bone (no verts on X Bot); fall back to just below the head.
    neck_bn = xo.find_bone(arm, "mixamorig:Neck", "Neck")
    if neck_bn:
        tank_top = bone_mw(arm, neck_bn).translation.z
    else:
        tank_top = (head_z[0] - 0.01) if head_z[0] is not None else 1.50
    hips_bn = xo.find_bone(arm, "mixamorig:Hips", "Hips")
    hips_head_z = bone_mw(arm, hips_bn).translation.z if hips_bn else 1.03
    # Mid-thigh: midpoint of the upper-leg z-span (hip to knee).
    if upleg_z[0] is not None:
        mid_thigh = 0.50 * (upleg_z[0] + upleg_z[1])
    else:
        mid_thigh = 0.75
    # Waistband sits at the navel / upper iliac — between the hip joint and
    # the lower rib, using the upper end of the Hips group plus a little.
    if hips_z[1] is not None:
        band_top = hips_z[1] + 0.012
    else:
        band_top = hips_head_z + 0.04
    band_bot = band_top - WAISTBAND_H
    hem_z = mid_thigh

    # Midline from the rest-pose spine / hips (not world origin).
    torso_pts = pts_w(lambda n: any(k in _clean(n) for k in ("hips", "spine", "head")))
    if torso_pts:
        mid_x = sum(p.x for p in torso_pts) / len(torso_pts)
    else:
        mid_x = 0.0

    counts = defaultdict(int)
    for o in meshes:
        if o.type != "MESH" or not o.data.vertices:
            continue
        ev = o.evaluated_get(deps)
        eme = ev.to_mesh()
        try:
            omw = ev.matrix_world
            world = np.array([list(omw @ v.co) for v in eme.vertices], np.float32)
        finally:
            ev.to_mesh_clear()
        n = len(o.data.vertices)
        if len(world) != n:
            # Evaluated vert count should match; if not, use bind verts.
            world = np.array([list(o.matrix_world @ v.co) for v in o.data.vertices],
                             np.float32)
            n = len(world)
        z = world[:, 2]
        x = world[:, 0]

        w_torso = _vg_weights(o, lambda n: any(
            k in _clean(n) for k in ("hips", "spine", "chest", "abdomen", "pelvis")))
        w_sh = _vg_weights(o, lambda n: "shoulder" in _clean(n) or "clavicle" in _clean(n)
                           or "collar" in _clean(n))
        w_arm = _vg_weights(o, lambda n: any(
            k in _clean(n) for k in ("arm", "hand", "thumb", "index", "middle",
                                     "ring", "pinky", "fore"))
            and "shoulder" not in _clean(n))
        w_upleg = _vg_weights(o, lambda n: "upleg" in _clean(n) or "thigh" in _clean(n))
        w_leg = _vg_weights(o, lambda n: any(
            k in _clean(n) for k in ("leftleg", "rightleg", "shin", "calf",
                                     "foot", "toe"))
            and "upleg" not in _clean(n) and "thigh" not in _clean(n))
        w_head = _vg_weights(o, lambda n: "head" in _clean(n) or "neck" in _clean(n))

        # Pad arrays if the mesh has no groups (joint shells sometimes don't).
        def _fit(a):
            if len(a) == n:
                return a
            b = np.zeros(n, np.float32)
            b[:min(n, len(a))] = a[:min(n, len(a))]
            return b
        w_torso, w_sh, w_arm = _fit(w_torso), _fit(w_sh), _fit(w_arm)
        w_upleg, w_leg, w_head = _fit(w_upleg), _fit(w_leg), _fit(w_head)

        zone = np.zeros(n, np.float32)

        # Trunks: waistband bottom down to mid-thigh, hips + upper thighs,
        # not lower legs, not arms, not head. Symmetric by construction
        # (thresholds are in Z and in already-symmetric Mixamo weights).
        trunks = (
            (z >= hem_z) & (z < band_bot)
            & (w_leg < 0.35)
            & (w_arm < 0.40)
            & (w_head < 0.25)
            & (w_sh < 0.50)
            & ((w_torso + w_upleg) > 0.15)
        )
        # Waistband stripe (charcoal), same horizontal coverage as the trunks.
        band = (
            (z >= band_bot) & (z <= band_top)
            & (w_leg < 0.35)
            & (w_arm < 0.40)
            & (w_head < 0.25)
            & (w_sh < 0.50)
            & ((w_torso + w_upleg) > 0.15)
        )
        # Tank: torso from the top of the waistband to the neck, sleeveless.
        # Deltoids (shoulder group) and arms stay clay so the glow reads.
        tank = (
            (z > band_top) & (z <= tank_top)
            & (w_torso > 0.20)
            & (w_sh < 0.28)
            & (w_arm < 0.18)
            & (w_head < 0.25)
            & (w_upleg < 0.25)
            & (w_leg < 0.25)
        )

        zone[trunks] = ZONE_TRUNKS
        zone[band] = ZONE_BAND
        zone[tank] = ZONE_TANK

        # One-ring smooth of isolated sprinkles, then restore hard zone
        # identity (no blend between burgundy and charcoal — that would
        # blotch). Only promote a skin vert if the majority of its
        # neighbours agree on one painted zone; never demote a zone vert
        # that has painted neighbours. This cleans raster jaggies without
        # changing the intended regions.
        try:
            import blob_v2
            edges = blob_v2._edges_np(o.data)
            # neighbour majority vote for skin verts only
            from collections import Counter
            # build adjacency
            adj = [[] for _ in range(n)]
            # edges is (E,2) int array
            for a, b in edges:
                if 0 <= a < n and 0 <= b < n:
                    adj[int(a)].append(int(b))
                    adj[int(b)].append(int(a))
            new_zone = zone.copy()
            for i in range(n):
                if zone[i] != ZONE_SKIN:
                    continue
                neigh = [zone[j] for j in adj[i] if zone[j] != ZONE_SKIN]
                if len(neigh) >= 3:
                    # majority of painted neighbours
                    vals, counts_n = np.unique(neigh, return_counts=True)
                    if counts_n.max() >= 3:
                        new_zone[i] = float(vals[counts_n.argmax()])
            zone = new_zone
        except Exception:
            pass

        # Write ozone (float) and pcol (rgb of the zone) as attributes.
        ozone = o.data.attributes.get("ozone")
        if ozone is None or ozone.data_type != "FLOAT" or ozone.domain != "POINT":
            if ozone is not None:
                o.data.attributes.remove(ozone)
            ozone = o.data.attributes.new("ozone", "FLOAT", "POINT")
        ozone.data.foreach_set("value", zone.astype(np.float32).ravel())

        pcol = np.zeros((n, 4), np.float32)
        pcol[:, 3] = 1.0
        pcol[:, :3] = CLAY
        pcol[zone == ZONE_TANK, :3] = CHARCOAL
        pcol[zone == ZONE_BAND, :3] = CHARCOAL
        pcol[zone == ZONE_TRUNKS, :3] = BURGUNDY
        va = o.data.color_attributes.get("pcol")
        if va is None:
            va = o.data.color_attributes.new("pcol", "FLOAT_COLOR", "POINT")
        va.data.foreach_set("color", pcol.ravel())
        o.data.update()

        counts["skin"] += int((zone == ZONE_SKIN).sum())
        counts["tank"] += int((zone == ZONE_TANK).sum())
        counts["trunks"] += int((zone == ZONE_TRUNKS).sum())
        counts["band"] += int((zone == ZONE_BAND).sum())
        counts["meshes"] += 1

    arm.data.pose_position = saved
    bpy.context.view_layer.update()
    counts["landmarks"] = {
        "mid_thigh_z": float(hem_z),
        "band_bot_z": float(band_bot),
        "band_top_z": float(band_top),
        "tank_top_z": float(tank_top),
        "mid_x": float(mid_x),
    }
    return dict(counts)


def painted_body_material(em_str=1.35):
    """Clay / charcoal / burgundy base, with existing mcol glow on top."""
    name = "XBotPaintedBody_v1"
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (920, 0)

    ozone = nt.nodes.new("ShaderNodeAttribute")
    ozone.attribute_name = "ozone"
    ozone.location = (-620, 200)

    # Zone masks.
    def cmp_eq(from_sock, value, loc):
        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.location = (loc[0], loc[1] + 40)
        nt.links.new(from_sock, sub.inputs[0])
        sub.inputs[1].default_value = value
        absn = nt.nodes.new("ShaderNodeMath")
        absn.operation = "ABSOLUTE"
        absn.location = (loc[0] + 160, loc[1] + 40)
        nt.links.new(sub.outputs[0], absn.inputs[0])
        less = nt.nodes.new("ShaderNodeMath")
        less.operation = "LESS_THAN"
        less.location = (loc[0] + 320, loc[1])
        nt.links.new(absn.outputs[0], less.inputs[0])
        less.inputs[1].default_value = 0.45
        return less.outputs[0]

    is_tank = cmp_eq(ozone.outputs["Fac"], ZONE_TANK, (-400, 260))
    is_trunks = cmp_eq(ozone.outputs["Fac"], ZONE_TRUNKS, (-400, 60))
    is_band = cmp_eq(ozone.outputs["Fac"], ZONE_BAND, (-400, -140))

    # painted = tank or band or trunks
    or1 = nt.nodes.new("ShaderNodeMath")
    or1.operation = "MAXIMUM"
    or1.location = (80, 160)
    nt.links.new(is_tank, or1.inputs[0])
    nt.links.new(is_band, or1.inputs[1])
    is_paint = nt.nodes.new("ShaderNodeMath")
    is_paint.operation = "MAXIMUM"
    is_paint.location = (240, 80)
    nt.links.new(or1.outputs[0], is_paint.inputs[0])
    nt.links.new(is_trunks, is_paint.inputs[1])

    # Base colour: clay, then charcoal for tank/band, burgundy for trunks.
    charcoal_col = (*CHARCOAL, 1.0)
    burg_col = (*BURGUNDY, 1.0)
    clay_col = (*CLAY, 1.0)

    mix_ch = nt.nodes.new("ShaderNodeMix")
    mix_ch.data_type = "RGBA"
    mix_ch.location = (80, 400)
    # Fac = tank or band
    or_ch = nt.nodes.new("ShaderNodeMath")
    or_ch.operation = "MAXIMUM"
    or_ch.location = (-40, 400)
    nt.links.new(is_tank, or_ch.inputs[0])
    nt.links.new(is_band, or_ch.inputs[1])
    nt.links.new(or_ch.outputs[0], mix_ch.inputs["Factor"])
    mix_ch.inputs["A"].default_value = clay_col
    mix_ch.inputs["B"].default_value = charcoal_col

    mix_bu = nt.nodes.new("ShaderNodeMix")
    mix_bu.data_type = "RGBA"
    mix_bu.location = (280, 400)
    nt.links.new(is_trunks, mix_bu.inputs["Factor"])
    nt.links.new(mix_ch.outputs["Result"], mix_bu.inputs["A"])
    mix_bu.inputs["B"].default_value = burg_col

    # Roughness: clay 0.86, tank 0.62, band 0.78 (rib), trunks satin 0.22.
    r_clay = nt.nodes.new("ShaderNodeMath")
    r_clay.operation = "MAXIMUM"
    r_clay.location = (80, -280)
    r_clay.inputs[0].default_value = 0.86
    r_clay.inputs[1].default_value = 0.86
    # stepwise mix
    r1 = nt.nodes.new("ShaderNodeMix")
    r1.data_type = "FLOAT"
    r1.location = (80, -80)
    nt.links.new(is_tank, r1.inputs["Factor"])
    r1.inputs["A"].default_value = 0.86
    r1.inputs["B"].default_value = 0.62
    r2 = nt.nodes.new("ShaderNodeMix")
    r2.data_type = "FLOAT"
    r2.location = (260, -80)
    nt.links.new(is_band, r2.inputs["Factor"])
    nt.links.new(r1.outputs["Result"], r2.inputs["A"])
    r2.inputs["B"].default_value = 0.78
    r3 = nt.nodes.new("ShaderNodeMix")
    r3.data_type = "FLOAT"
    r3.location = (440, -80)
    nt.links.new(is_trunks, r3.inputs["Factor"])
    nt.links.new(r2.outputs["Result"], r3.inputs["A"])
    r3.inputs["B"].default_value = 0.22

    base = nt.nodes.new("ShaderNodeBsdfPrincipled")
    base.location = (520, 280)
    nt.links.new(mix_bu.outputs["Result"], base.inputs["Base Color"])
    nt.links.new(r3.outputs["Result"], base.inputs["Roughness"])
    if "Specular IOR Level" in base.inputs:
        base.inputs["Specular IOR Level"].default_value = 0.22

    # Glow (identical to blob_v2.punch_material): mcol drives tint + emission.
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "mcol"
    attr.location = (-400, -420)
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    sep.location = (-200, -420)
    nt.links.new(attr.outputs["Color"], sep.inputs["Color"])
    mx1 = nt.nodes.new("ShaderNodeMath")
    mx1.operation = "MAXIMUM"
    mx1.location = (0, -400)
    nt.links.new(sep.outputs["Red"], mx1.inputs[0])
    nt.links.new(sep.outputs["Green"], mx1.inputs[1])
    mx2 = nt.nodes.new("ShaderNodeMath")
    mx2.operation = "MAXIMUM"
    mx2.location = (160, -400)
    nt.links.new(mx1.outputs[0], mx2.inputs[0])
    nt.links.new(sep.outputs["Blue"], mx2.inputs[1])

    tint = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tint.location = (360, -280)
    nt.links.new(attr.outputs["Color"], tint.inputs["Base Color"])
    tint.inputs["Roughness"].default_value = 0.70
    if "Specular IOR Level" in tint.inputs:
        tint.inputs["Specular IOR Level"].default_value = 0.08
    em = nt.nodes.new("ShaderNodeEmission")
    em.location = (360, -500)
    nt.links.new(attr.outputs["Color"], em.inputs["Color"])
    stm = nt.nodes.new("ShaderNodeMath")
    stm.operation = "MULTIPLY"
    stm.location = (200, -560)
    nt.links.new(mx2.outputs[0], stm.inputs[0])
    stm.inputs[1].default_value = float(em_str)
    nt.links.new(stm.outputs[0], em.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    add.location = (560, -400)
    nt.links.new(tint.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (740, 40)
    nt.links.new(mx2.outputs[0], mix.inputs["Fac"])
    nt.links.new(base.outputs[0], mix.inputs[1])
    nt.links.new(add.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def assign_painted_material(meshes, em_str=1.35):
    mat = painted_body_material(em_str=em_str)
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        o.active_material = mat
        n = len(o.data.vertices)
        if n == 0:
            continue
        va = o.data.color_attributes.get("mcol")
        if va is None:
            va = o.data.color_attributes.new("mcol", "FLOAT_COLOR", "POINT")
            col = np.zeros((n, 4), np.float32)
            col[:, 3] = 1.0
            va.data.foreach_set("color", col.ravel())
        o.data.update()
    return mat


# ---------------------------------------------------------------------------
# Hide the clay fist inside the rigid glove (construction, not a patch:
# a rigid glove cannot follow Mixamo finger curl).
# ---------------------------------------------------------------------------
def mask_hands_inside_gloves(meshes, gloves, arm):
    hidden = {}
    for side, (glove, info) in gloves.items():
        bn = info["bone"]
        inv = bone_mw(arm, bn).inverted()
        cuff = inv @ Vector(info["cuff"])
        cuff_y = cuff.y + 0.004  # a few mm inside the opening
        pred = hand_pred(side)
        for o in meshes:
            if o.type != "MESH":
                continue
            w = _vg_weights(o, pred)
            # also hide a bit of the forearm near the wrist so the cuff covers it
            w_fore = _vg_weights(o, lambda n, s=side: (
                (("left" in _clean(n) and s == "L") or ("right" in _clean(n) and s == "R"))
                and "forearm" in _clean(n)
            ))
            n = len(o.data.vertices)
            weights = np.zeros(n, np.float32)
            mw = o.matrix_world
            # Use bind-ish current evaluated positions.
            deps = bpy.context.evaluated_depsgraph_get()
            ev = o.evaluated_get(deps)
            me = ev.to_mesh()
            try:
                emw = ev.matrix_world
                for i, v in enumerate(me.vertices):
                    if i >= n:
                        break
                    loc = inv @ (emw @ v.co)
                    handish = (i < len(w) and w[i] >= 0.18) or (
                        i < len(w_fore) and w_fore[i] >= 0.40 and loc.y >= cuff_y - 0.01)
                    if handish and loc.y >= cuff_y:
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
# Public apply + criterion checks
# ---------------------------------------------------------------------------
def apply_painted_outfit(arm, meshes, f0, f1, collection_name="XBotPaintedV1"):
    col = bpy.data.collections.get(collection_name)
    if col is None:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)

    zone_info = paint_body_zones(meshes, arm)
    assign_painted_material(meshes)

    gloves = make_glove_pair(arm, meshes, col, f0, f1)
    hat, hat_info = make_beanie(arm, meshes, col)
    hidden = mask_hands_inside_gloves(
        meshes, {s: gloves[s] for s in ("L", "R")}, arm)

    handle = {
        "collection": collection_name,
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
    }
    bpy.context.view_layer.update()
    return handle


def _pts_in_glove_local(glove, arm, bone_name, world_pts):
    inv = bone_mw(arm, bone_name).inverted()
    return [inv @ p for p in world_pts]


def check_gloves(arm, meshes, handle, f0, f1, step=1):
    """Criterion 1. Returns a report; ok is True only if every frame passes."""
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
        pred = hand_pred(side)
        while f <= int(f1):
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            n_frames += 1
            inv = bone_mw(arm, bn).inverted()
            hand = [inv @ p for p in collect_weighted(meshes, pred, thresh=0.30)]
            glove_locs = [inv @ p for p in world_pts(glove)]
            if not hand or not glove_locs:
                failures.append({"f": f, "reason": "no verts"})
                f += step
                continue
            gmin = Vector((min(p.x for p in glove_locs), min(p.y for p in glove_locs),
                           min(p.z for p in glove_locs)))
            gmax = Vector((max(p.x for p in glove_locs), max(p.y for p in glove_locs),
                           max(p.z for p in glove_locs)))
            # Cuff plane: allow verts with Y below the cuff lip to leave
            # through the wrist opening.
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
            # Floating: glove centroid vs hand centroid (bone-local).
            hc = sum(hand, Vector()) / len(hand)
            gc = sum(glove_locs, Vector()) / len(glove_locs)
            dist = (hc - gc).length
            max_float = max(max_float, dist)
            # Forearm placement: most of the glove should sit at Y >= -0.06
            # (6 cm behind the wrist is the end of a long cuff, not mid-arm).
            glove_mid_y = 0.5 * (gmin.y + gmax.y)
            # Overlap of glove Y-range with the hand Y-range.
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


def check_beanie(arm, meshes, handle, f0, f1, step=2):
    """Criterion 2. Crown coverage and hem staying above the brows."""
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
        land = _head_landmarks(meshes, arm, head_bone)
        reasons = []
        # Crown of the hat must be at or above the skull top, and the hat
        # must still cover the crown point (hem below crown, hat over it).
        if info["crown"].z < land["zmax"] - 0.015:
            reasons.append(
                f"hat crown z={info['crown'].z:.3f} below skull top {land['zmax']:.3f}")
        if info["hem"].z > land["zmax"] - 0.02:
            reasons.append("hem is up at the crown — hat has slid off")
        # Hem should stay near / just above the brow line, not drop over the face.
        if info["hem"].z < land["brow_z"] - 0.025:
            reasons.append(
                f"hem z={info['hem'].z:.3f} dropped below brows {land['brow_z']:.3f}")
        # Horizontal: hem centre should stay over the head, not float off.
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
    if not handle:
        return
    for name in handle.get("objects") or []:
        o = bpy.data.objects.get(name)
        if o:
            bpy.data.objects.remove(o, do_unlink=True)
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        if "XBotGloveMask" in o.modifiers:
            o.modifiers.remove(o.modifiers["XBotGloveMask"])
        vg = o.vertex_groups.get("xbot_glove_hide")
        if vg:
            o.vertex_groups.remove(vg)
