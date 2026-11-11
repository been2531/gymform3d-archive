#!/usr/bin/env python3
"""Fit a Rigify human metarig to Tishkok's A-pose, bind shells, retarget ACCAD JabRight.

  nice -n 19 blender -b --python tools/rig3d/eval_tishkok_jab.py -- \
      --out-dir output/_basemodels/eval --name boxer_jab_tishkok
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

import addon_utils
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval_base_model import (  # noqa: E402
    _render,
    _reset,
    _setup_camera,
    _setup_world_lights,
    _world_dims,
    _argv,
)
import blob_v2  # noqa: E402
import eval_boxer_jab as v1  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
TISHKOK = os.path.join(
    ROOT, "output/_basemodels/boxer/tishkok_stylized_boxer/scene.gltf")
ACCAD_BVH = os.path.join(
    ROOT, "output/_basemodels/boxer/accad/Male2_E2_JabRight.bvh")

YAW = -40.0  # 3/4 from punching (character-right, -X) side; faces -Y
ELEV = 10.0
RES = (540, 960)
WELD = 0.0001
EM_STR = 1.30

# ACCAD → Rigify DEF (primary bones, not twist .001). Resolved at runtime.
ACCAD_CANDIDATES = {
    "Hips": ("DEF-hips", "DEF-spine", "hips", "spine"),
    "ToSpine": ("DEF-spine.001", "spine.001"),
    "Spine": ("DEF-spine.002", "spine.002"),
    "Spine1": ("DEF-spine.003", "spine.003"),
    "Neck": ("DEF-neck", "DEF-spine.004", "neck", "spine.004"),
    "Head": ("DEF-head", "DEF-spine.005", "DEF-spine.006", "head", "spine.005", "spine.006"),
    "LeftShoulder": ("DEF-shoulder.L", "shoulder.L"),
    "LeftArm": ("DEF-upper_arm.L", "upper_arm.L"),
    "LeftForeArm": ("DEF-forearm.L", "forearm.L"),
    "LeftHand": ("DEF-hand.L", "hand.L"),
    "RightShoulder": ("DEF-shoulder.R", "shoulder.R"),
    "RightArm": ("DEF-upper_arm.R", "upper_arm.R"),
    "RightForeArm": ("DEF-forearm.R", "forearm.R"),
    "RightHand": ("DEF-hand.R", "hand.R"),
    "LeftUpLeg": ("DEF-thigh.L", "thigh.L"),
    "LeftLeg": ("DEF-shin.L", "shin.L"),
    "LeftFoot": ("DEF-foot.L", "foot.L"),
    "LeftToeBase": ("DEF-toe.L", "toe.L"),
    "RightUpLeg": ("DEF-thigh.R", "thigh.R"),
    "RightLeg": ("DEF-shin.R", "shin.R"),
    "RightFoot": ("DEF-foot.R", "foot.R"),
    "RightToeBase": ("DEF-toe.R", "toe.R"),
}


def hide_obj(o):
    o.hide_render = True
    try:
        o.hide_set(True)
    except Exception:
        pass


def show_obj(o):
    o.hide_render = False
    try:
        o.hide_set(False)
    except Exception:
        pass


def apply_tf(objs, loc=True, rot=True, scale=True):
    if not objs:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=loc, rotation=rot, scale=scale)


def centroid(obj):
    ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
    acc = Vector((0, 0, 0))
    for p in ws:
        acc += p
    return acc / max(len(ws), 1)


def glove_wrist(obj):
    """Opening of the sculpted glove: high-Z, more medial verts."""
    ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not ws:
        return centroid(obj)
    zs = sorted(p.z for p in ws)
    zcut = zs[int(0.82 * (len(zs) - 1))]
    top = [p for p in ws if p.z >= zcut]
    return min(top, key=lambda p: abs(p.x))


def join_group(objs, name):
    if not objs:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    out = bpy.context.view_layer.objects.active
    out.name = name
    return out


def classify_part(p):
    ws = [p.matrix_world @ v.co for v in p.data.vertices]
    xs = [q.x for q in ws]
    zs = [q.z for q in ws]
    cen = Vector((sum(xs) / len(xs), 0.0, sum(zs) / len(zs)))
    dimx = max(xs) - min(xs)
    dimz = max(zs) - min(zs)
    tris = len(p.data.polygons)
    if cen.z < 0.25 and abs(cen.x) > 0.05:
        return "shoe_r" if cen.x > 0 else "shoe_l"
    if 0.68 < cen.z < 1.05 and abs(cen.x) > 0.28:
        return "glove_r" if cen.x > 0 else "glove_l"
    # shorts: mid-height slab, narrower than the full body
    if tris > 2000 and abs(cen.x) < 0.12 and 0.65 < cen.z < 1.05 and dimx < 0.50 and dimz < 0.55:
        return "shorts"
    return "body"


def weld_separate(obj, threshold=WELD):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    dup = bpy.context.view_layer.objects.active
    src_name = obj.name
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    parts = [o for o in bpy.data.objects if o.type == "MESH" and o != obj]
    bpy.data.objects.remove(obj, do_unlink=True)
    return parts


def prepare_tishkok(path):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("Tishkok import produced no meshes")
    apply_tf(meshes)
    raw = meshes[0]
    # if more than one mesh, join first
    if len(meshes) > 1:
        raw = join_group(meshes, "tishkok_raw")
    parts = weld_separate(raw)
    buckets = {k: [] for k in (
        "body", "shorts", "glove_l", "glove_r", "shoe_l", "shoe_r")}
    for p in parts:
        buckets[classify_part(p)].append(p)
    groups = {}
    for key, objs in buckets.items():
        groups[key] = join_group(objs, f"tishkok_{key}") if objs else None
        print(f"SHELL {key}: {len(objs)} parts -> "
              f"{'none' if groups[key] is None else groups[key].name} "
              f"tris={0 if groups[key] is None else len(groups[key].data.polygons)}",
              flush=True)
    visible = [o for o in groups.values() if o is not None]
    apply_tf(visible)
    return groups



def world_dims_verts(objs):
    """Vertex-based world AABB — safe after bone parenting (bound_box can lie)."""
    xs, ys, zs = [], [], []
    for o in objs:
        if o is None:
            continue
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co
            xs.append(p.x); ys.append(p.y); zs.append(p.z)
    if not xs:
        return Vector((0, 0, 0)), Vector((0, 0, 0)), Vector((0, 0, 0))
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    return mn, mx, mx - mn


def shoulder_landmarks(body):
    ws = [body.matrix_world @ v.co for v in body.data.vertices]
    zmin = min(p.z for p in ws)
    zmax = max(p.z for p in ws)
    h = zmax - zmin
    band = [p for p in ws if zmin + 0.76 * h <= p.z <= zmin + 0.88 * h]
    left = [p for p in band if p.x < -0.04]
    right = [p for p in band if p.x > 0.04]
    if not left or not right:
        raise RuntimeError("shoulder band empty")

    def lateral(pts, sign):
        pts = sorted(pts, key=lambda p: sign * p.x)
        pick = pts[-max(1, len(pts) // 8):]
        acc = Vector((0, 0, 0))
        for q in pick:
            acc += q
        return acc / len(pick)

    return lateral(left, -1), lateral(right, 1)


def xf_descendants(bone, M):
    for c in bone.children:
        c.head = M @ c.head
        c.tail = M @ c.tail
        xf_descendants(c, M)


def place_arm(eb, side, shoulder, elbow, wrist, fist):
    shb = eb.get(f"shoulder.{side}")
    if shb is not None:
        shb.tail = shoulder.copy()
    ua = eb[f"upper_arm.{side}"]
    ua.head = shoulder.copy()
    ua.tail = elbow.copy()
    fa = eb[f"forearm.{side}"]
    if not fa.use_connect:
        fa.head = elbow.copy()
    fa.tail = wrist.copy()
    ha = eb[f"hand.{side}"]
    old_h, old_t = ha.head.copy(), ha.tail.copy()
    if not ha.use_connect:
        ha.head = wrist.copy()
    ha.tail = fist.copy()
    new_h, new_t = ha.head.copy(), ha.tail.copy()
    old_y = old_t - old_h
    new_y = new_t - new_h
    if old_y.length > 1e-8 and new_y.length > 1e-8:
        rot = old_y.normalized().rotation_difference(new_y.normalized()).to_matrix().to_4x4()
        M = Matrix.Translation(new_h) @ rot @ Matrix.Translation(-old_h)
        xf_descendants(ha, M)


def punch_metrics_any(arm, punch_hand):
    hand = v1.bone_world(arm, punch_hand)
    pelvis = (v1.bone_world(arm, "pelvis") or v1.bone_world(arm, "DEF-hips")
              or v1.bone_world(arm, "DEF-spine") or v1.bone_world(arm, "spine")
              or v1.bone_world(arm, "torso") or v1.bone_world(arm, "Hips"))
    head = (v1.bone_world(arm, "Head") or v1.bone_world(arm, "DEF-head")
            or v1.bone_world(arm, "DEF-spine.005") or v1.bone_world(arm, "spine.005")
            or v1.bone_world(arm, "spine.006") or v1.bone_world(arm, "DEF-spine.006"))
    if hand is None or pelvis is None:
        return None
    return {
        "hand": list(hand),
        "pelvis": list(pelvis),
        "head": list(head) if head else None,
        "reach": float((hand - pelvis).length),
        "hand_z": float(hand.z),
    }


def enable_rigify():
    try:
        bpy.ops.preferences.addon_enable(module="rigify")
        print("RIGIFY addon_enable ok", flush=True)
    except Exception as e:
        print("RIGIFY addon_enable", type(e).__name__, e, flush=True)
        try:
            addon_utils.enable("rigify", default_set=True)
            print("RIGIFY addon_utils.enable fallback", flush=True)
        except Exception as e2:
            print("RIGIFY enable failed", type(e2).__name__, e2, flush=True)


def scale_arm_bones(meta, factor):
    bpy.ops.object.select_all(action="DESELECT")
    meta.select_set(True)
    bpy.context.view_layer.objects.active = meta
    bpy.ops.object.mode_set(mode="EDIT")
    eb = meta.data.edit_bones

    def move_desc(bone, delta, skip_head=False):
        if not skip_head:
            bone.head += delta
        bone.tail += delta
        for c in bone.children:
            move_desc(c, delta)

    for side in ("L", "R"):
        ua = eb[f"upper_arm.{side}"]
        fa = eb[f"forearm.{side}"]
        ha = eb[f"hand.{side}"]
        origin = ua.head.copy()
        ua.tail = origin + (ua.tail - origin) * factor
        if not fa.use_connect:
            fa.head = ua.tail.copy()
        old_fa_tail = fa.tail.copy()
        fa.tail = fa.head + (fa.tail - fa.head) * factor
        delta = fa.tail - old_fa_tail
        if not ha.use_connect:
            ha.head += delta
        move_desc(ha, delta, skip_head=ha.use_connect)
    bpy.ops.object.mode_set(mode="OBJECT")


def _pb_world(arm, name, attr="head"):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    return arm.matrix_world @ (pb.head if attr == "head" else pb.tail)


def fit_metarig(groups):
    enable_rigify()
    bpy.ops.object.armature_human_metarig_add()
    meta = bpy.context.active_object
    meta.name = "metarig"

    body = groups["body"]
    vis = [o for o in groups.values() if o is not None]
    mn, mx, dim = world_dims_verts(vis)
    bpy.context.view_layer.update()

    head_bone = "spine.006" if "spine.006" in meta.pose.bones else "spine.005"
    head_z = _pb_world(meta, head_bone, "tail").z
    heel = _pb_world(meta, "heel.02.L") or _pb_world(meta, "foot.L")
    meta_h = head_z - heel.z
    factor = float(dim.z) / max(meta_h, 1e-6)
    meta.scale *= factor
    apply_tf([meta], loc=False, rot=False, scale=True)
    bpy.context.view_layer.update()

    heel = _pb_world(meta, "heel.02.L") or _pb_world(meta, "foot.L")
    hip = _pb_world(meta, "spine")
    meta.location.z += (mn.z - heel.z)
    meta.location.x += ((mn.x + mx.x) * 0.5) - hip.x
    meta.location.y += ((mn.y + mx.y) * 0.5) - hip.y
    apply_tf([meta], loc=True, rot=False, scale=False)
    bpy.context.view_layer.update()

    wrist_L = glove_wrist(groups["glove_l"])
    wrist_R = glove_wrist(groups["glove_r"])
    fist_L = centroid(groups["glove_l"])
    fist_R = centroid(groups["glove_r"])
    sh_L, sh_R = shoulder_landmarks(body)
    print("LANDMARK wrist_L", list(wrist_L), "fist_L", list(fist_L), flush=True)
    print("LANDMARK wrist_R", list(wrist_R), "fist_R", list(fist_R), flush=True)
    print("LANDMARK sh_L", list(sh_L), "sh_R", list(sh_R), flush=True)

    def elbow(sh, wrist):
        mid = (sh + wrist) * 0.5
        # character faces -Y; elbow goes back (+Y)
        return mid + Vector((0.0, 0.09, 0.0))

    el_L = elbow(sh_L, wrist_L)
    el_R = elbow(sh_R, wrist_R)
    bpy.ops.object.select_all(action="DESELECT")
    meta.select_set(True)
    bpy.context.view_layer.objects.active = meta
    bpy.ops.object.mode_set(mode="EDIT")
    eb = meta.data.edit_bones
    place_arm(eb, "L", sh_L, el_L, wrist_L, fist_L)
    place_arm(eb, "R", sh_R, el_R, wrist_R, fist_R)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    for side, fist, sh in (("L", fist_L, sh_L), ("R", fist_R, sh_R)):
        ht = _pb_world(meta, f"hand.{side}", "tail")
        hh = _pb_world(meta, f"hand.{side}", "head")
        uh = _pb_world(meta, f"upper_arm.{side}")
        print(f"PLACE {side} hand_head={list(hh)} hand_tail={list(ht)} "
              f"fist={list(fist)} tail_err={(ht - fist).length:.3f} "
              f"sh={list(uh)} sh_err={(uh - sh).length:.3f}", flush=True)
    return meta


def generate_rig(meta):
    bpy.ops.object.select_all(action="DESELECT")
    meta.select_set(True)
    bpy.context.view_layer.objects.active = meta
    before = set(bpy.data.objects)
    try:
        bpy.ops.pose.rigify_generate()
    except Exception as e:
        print("GENERATE_FAIL", type(e).__name__, e, flush=True)
        return meta, False
    new_arms = [o for o in bpy.data.objects
                if o not in before and o.type == "ARMATURE"]
    rig = None
    for o in new_arms:
        if o != meta:
            rig = o
            break
    if rig is None:
        rig = bpy.data.objects.get("rig")
    if rig is None:
        print("GENERATE_NO_RIG — falling back to metarig", flush=True)
        return meta, False
    hide_obj(meta)
    # hide widgets
    for o in bpy.data.objects:
        n = o.name
        if n.startswith("WGT") or n.startswith("WGTS"):
            hide_obj(o)
    print("GENERATED", rig.name, "bones", len(rig.data.bones),
          "DEF", sum(1 for b in rig.data.bones if b.name.startswith("DEF-")),
          flush=True)
    return rig, True


def auto_weight(mesh, arm):
    mw = mesh.matrix_world.copy()
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)
    except TypeError:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    mesh.matrix_world = mw
    bpy.context.view_layer.update()
    print(f"AUTOWEIGHT {mesh.name} groups={len(mesh.vertex_groups)} "
          f"kept_world dimz={world_dims_verts([mesh])[2].z:.3f}", flush=True)


def rigid_parent_bone(mesh, arm, bone_name):
    if arm.pose.bones.get(bone_name) is None:
        raise KeyError(f"missing bone {bone_name}")
    before = centroid(mesh)
    mw = mesh.matrix_world.copy()
    bone = arm.pose.bones[bone_name]
    # Blender bone-parent origin is the tail of the bone.
    tail_space = arm.matrix_world @ bone.matrix @ Matrix.Translation(
        (0.0, bone.length, 0.0))
    mesh.parent = arm
    mesh.parent_type = "BONE"
    mesh.parent_bone = bone_name
    mesh.matrix_parent_inverse = tail_space.inverted() @ mw
    bpy.context.view_layer.update()
    mesh.matrix_world = mw
    bpy.context.view_layer.update()
    after = centroid(mesh)
    err = (after - before).length
    print(f"RIGID {mesh.name} -> {bone_name} jump={err:.4f}m "
          f"before={list(before)} after={list(after)}", flush=True)
    return err



def pick_bone(arm, *names):
    for n in names:
        if arm.pose.bones.get(n):
            return n
    return None


def resolve_mapping(arm):
    mapping = {}
    missing = []
    for src, cands in ACCAD_CANDIDATES.items():
        name = pick_bone(arm, *cands)
        if name:
            mapping[src] = name
        else:
            missing.append(src)
    print("MAP", mapping, flush=True)
    print("MAP_MISSING", missing, flush=True)
    return mapping, missing


def rest_y_world(arm, pb):
    """World-space rest matrix of pb given the parent's *current* pose."""
    if pb.parent:
        parent_pose = arm.matrix_world @ pb.parent.matrix
        parent_rest = arm.matrix_world @ pb.parent.bone.matrix_local
        child_rest = arm.matrix_world @ pb.bone.matrix_local
        # parent_pose and parent_rest live in world already if we multiply
        # bone.matrix_local is armature-space, so:
        parent_pose = arm.matrix_world @ pb.parent.matrix
        rest_arm = pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        rest_world = parent_pose @ rest_arm
    else:
        rest_world = arm.matrix_world @ pb.bone.matrix_local
    return rest_world


def record_rest_ys(arm):
    saved = {pb.name: pb.matrix_basis.copy() for pb in arm.pose.bones}
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    rest = {}
    for pb in arm.pose.bones:
        y = (arm.matrix_world @ pb.matrix).to_3x3() @ Vector((0.0, 1.0, 0.0))
        rest[pb.name] = y.normalized() if y.length > 1e-8 else Vector((0.0, 1.0, 0.0))
    for name, mb in saved.items():
        arm.pose.bones[name].matrix_basis = mb
    bpy.context.view_layer.update()
    return rest


def retarget_rest_swing(src_arm, tgt_arm, mapping, src_rest_ys):
    """Copy each source bone's rest-to-pose swing onto the target rest +Y."""
    for src_name, tgt_name in mapping.items():
        src = src_arm.pose.bones.get(src_name)
        tgt = tgt_arm.pose.bones.get(tgt_name)
        if src is None or tgt is None:
            continue
        src_y = ((src_arm.matrix_world @ src.matrix).to_3x3()
                 @ Vector((0.0, 1.0, 0.0)))
        if src_y.length < 1e-8:
            continue
        src_y.normalize()
        rest_y = src_rest_ys.get(src_name)
        if rest_y is None or rest_y.length < 1e-8:
            desired = src_y
        else:
            desired = rest_y.rotation_difference(src_y) @ (
                rest_y_world(tgt_arm, tgt).to_3x3() @ Vector((0.0, 1.0, 0.0)))
        if desired.length < 1e-8:
            continue
        desired.normalize()
        rest_world = rest_y_world(tgt_arm, tgt)
        local_target = rest_world.to_3x3().inverted() @ desired
        if local_target.length < 1e-8:
            continue
        tgt.rotation_mode = "QUATERNION"
        tgt.rotation_quaternion = Vector((0.0, 1.0, 0.0)).rotation_difference(
            local_target.normalized())


ARM_SRC = {
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
}


def retarget_world_aim_arms(src_arm, tgt_arm, mapping):
    """Absolute +Y aim for arm bones only — leaves hips/spine/legs at rest.

    ACCAD bind OFFSETS are not an anatomical T-pose, so rest-relative swing
    of the spine/hips dumps a standing target on its back.
    """
    for src_name, tgt_name in mapping.items():
        if src_name not in ARM_SRC:
            continue
        src = src_arm.pose.bones.get(src_name)
        tgt = tgt_arm.pose.bones.get(tgt_name)
        if src is None or tgt is None:
            continue
        src_y = ((src_arm.matrix_world @ src.matrix).to_3x3()
                 @ Vector((0.0, 1.0, 0.0)))
        if src_y.length < 1e-8:
            continue
        src_y.normalize()
        rest_world = rest_y_world(tgt_arm, tgt)
        local_target = rest_world.to_3x3().inverted() @ src_y
        if local_target.length < 1e-8:
            continue
        tgt.rotation_mode = "QUATERNION"
        tgt.rotation_quaternion = Vector((0.0, 1.0, 0.0)).rotation_difference(
            local_target.normalized())


def bake_world_aim(src_arm, tgt_arm, mapping, f0, f1):
    arm_map = {s: t for s, t in mapping.items() if s in ARM_SRC}
    print("ARM_MAP", arm_map, flush=True)
    for pb in tgt_arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        retarget_world_aim_arms(src_arm, tgt_arm, mapping)
        bpy.context.view_layer.update()
        for _src, tgt_name in arm_map.items():
            pb = tgt_arm.pose.bones.get(tgt_name)
            if pb is None:
                continue
            pb.keyframe_insert(data_path="rotation_quaternion", frame=f)


def import_accad(path):
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(
        filepath=path,
        global_scale=0.01,
        use_fps_scale=True,
        update_scene_fps=False,
        update_scene_duration=True,
        axis_forward="-Y",
        axis_up="Z",
        rotate_mode="NATIVE",
    )
    new = [o for o in bpy.data.objects if o not in before]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("ACCAD import produced no armature")
    arm = arms[0]
    arm.name = "accad_jab_right"
    hide_obj(arm)
    # stand-up check
    bpy.context.view_layer.update()
    heads = [arm.matrix_world @ b.head for b in arm.pose.bones]
    zs = [p.z for p in heads]
    ys = [p.y for p in heads]
    xs = [p.x for p in heads]
    span = {
        "x": max(xs) - min(xs),
        "y": max(ys) - min(ys),
        "z": max(zs) - min(zs),
    }
    print("ACCAD span", span, "bones", [b.name for b in arm.data.bones], flush=True)
    # If Z is not the long axis, rotate the object
    longest = max(span, key=span.get)
    if longest != "z":
        print("ACCAD not Z-up, rotating", longest, "-> Z", flush=True)
        if longest == "y":
            arm.rotation_euler[0] = math.radians(90)
        elif longest == "x":
            arm.rotation_euler[1] = math.radians(-90)
        apply_tf([arm], loc=False, rot=True, scale=False)
    return arm


def face_accad_minus_y(arm, frame):
    """Rotate ACCAD object so rest-pose chest faces -Y (Tishkok front)."""
    saved = {pb.name: pb.matrix_basis.copy() for pb in arm.pose.bones}
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    left = v1.bone_world(arm, "LeftShoulder") or v1.bone_world(arm, "LeftArm")
    right = v1.bone_world(arm, "RightShoulder") or v1.bone_world(arm, "RightArm")
    hips = v1.bone_world(arm, "Hips")
    head = v1.bone_world(arm, "Head")
    if not (left and right and hips and head):
        for name, mb in saved.items():
            arm.pose.bones[name].matrix_basis = mb
        return {"facing": "skip"}
    right_vec = right - left
    up = head - hips
    fwd = right_vec.cross(up)
    if fwd.length < 1e-6:
        fwd = up.cross(right_vec)
    fwd.z = 0
    if fwd.length < 1e-6:
        for name, mb in saved.items():
            arm.pose.bones[name].matrix_basis = mb
        return {"facing": "degenerate"}
    fwd.normalize()
    want = Vector((0.0, -1.0, 0.0))
    ang = math.atan2(want.x, want.y) - math.atan2(fwd.x, fwd.y)
    arm.rotation_euler[2] += ang
    apply_tf([arm], loc=False, rot=True, scale=False)
    for name, mb in saved.items():
        arm.pose.bones[name].matrix_basis = mb
    bpy.context.view_layer.update()
    print(f"ACCAD yaw {math.degrees(ang):.1f} deg  was fwd={list(fwd)}", flush=True)
    return {"facing_yaw_deg": float(math.degrees(ang)), "fwd_before": list(fwd)}


def paint_jab_glow_strong(meshes, punch_side="right"):
    """Stronger punching-side glow: lateral front delt, triceps, true ext. obliques."""
    dummy_cam = Vector((0.0, -1.0, 0.2))
    hip, spine, side, back, slen, info = blob_v2.estimate_anatomic_frame(
        meshes, dummy_cam)
    totals = {n: 0 for n in v1.JAB_ORDER}
    per = []
    want_right = str(punch_side).lower().startswith("r")
    for o in meshes:
        world, no = blob_v2._world_co_no(o)
        if not len(world):
            continue
        rel = world - hip
        t = (rel @ spine) / max(slen, 1e-8)
        lat = rel @ side
        nb = no @ back
        s = float(slen / 0.55)

        r_idx = v1.vg_named(o, lambda n: n.endswith("_r") or n.endswith(".r")
                            or "right" in n)
        l_idx = v1.vg_named(o, lambda n: n.endswith("_l") or n.endswith(".l")
                            or "left" in n)
        wr = blob_v2.vg_weight_array(o, r_idx)
        wl = blob_v2.vg_weight_array(o, l_idx)
        if wr.sum() > 1 and wl.sum() > 1:
            r_med = float(np.median(lat[wr > 0.25])) if (wr > 0.25).any() else 0.0
            l_med = float(np.median(lat[wl > 0.25])) if (wl > 0.25).any() else 0.0
            right_sign = 1.0 if r_med >= l_med else -1.0
            punch_lat = right_sign if want_right else -right_sign
        else:
            punch_lat = 1.0 if want_right else -1.0

        side_coord = lat * punch_lat

        upper_i = v1.vg_named(o, lambda n: any(k in n for k in (
            "upperarm", "upper_arm", "uparm", "humerus")))
        clav_i = v1.vg_named(o, lambda n: any(k in n for k in (
            "clavicle", "shoulder", "collar")))
        w_upper = blob_v2.vg_weight_array(o, upper_i)
        w_clav = blob_v2.vg_weight_array(o, clav_i)
        if want_right:
            side_w = wr / np.maximum(wr + wl, 1e-6)
        else:
            side_w = wl / np.maximum(wr + wl, 1e-6)
        if wr.sum() + wl.sum() < 1:
            side_w = blob_v2.smoothstep(0.02 * s, 0.08 * s, side_coord)

        def pt(tt, lat_m, bck_m):
            return (hip + (tt * slen) * spine
                    + (lat_m * s * punch_lat) * side
                    + (bck_m * s) * back)

        # front delt: more lateral so it does not spill onto the pec
        delt_caps = [
            (pt(0.80, 0.26, -0.04), pt(0.76, 0.34, -0.06), 0.062 * s, 0.048 * s),
            (pt(0.79, 0.24, -0.01), pt(0.75, 0.32, -0.05), 0.054 * s, 0.042 * s),
        ]
        # triceps: along the punching-side upper-arm bone (not spine-t, which
        # in A-pose puts waist-height capsules on the hanging wrist)
        ua = v1.bone_world(bpy.context.view_layer.objects.get("metarig") or
                           bpy.context.view_layer.objects.get("rig"),
                           "upper_arm.R" if want_right else "upper_arm.L")
        el = v1.bone_world(bpy.context.view_layer.objects.get("metarig") or
                           bpy.context.view_layer.objects.get("rig"),
                           "forearm.R" if want_right else "forearm.L")
        # fallback names
        if ua is None:
            ua = v1.bone_world(bpy.data.objects.get("metarig"), "DEF-upper_arm.R" if want_right else "DEF-upper_arm.L")
        if el is None:
            el = v1.bone_world(bpy.data.objects.get("metarig"), "DEF-forearm.R" if want_right else "DEF-forearm.L")
        if ua is not None and el is not None:
            axis = el - ua
            bk = Vector((float(back[0]), float(back[1]), float(back[2])))
            tri_a = ua + axis * 0.20 + bk * (0.035 * s)
            tri_b = ua + axis * 0.78 + bk * (0.030 * s)
            tri_caps = [
                (tri_a, tri_b, 0.055 * s, 0.046 * s),
                (tri_a + bk * (0.015 * s), tri_b + bk * (0.012 * s), 0.048 * s, 0.040 * s),
            ]
        else:
            tri_caps = [
                (pt(0.80, 0.22, 0.05), pt(0.70, 0.20, 0.04), 0.050 * s, 0.042 * s),
            ]
        # true external obliques: lower waist flank (ribs → iliac), not armpit
        obl_caps = [
            (pt(0.46, 0.14, -0.02), pt(0.36, 0.15, 0.00), 0.058 * s, 0.046 * s),
            (pt(0.44, 0.13, -0.04), pt(0.36, 0.14, -0.02), 0.048 * s, 0.040 * s),
        ]

        def caps_w(caps):
            acc = np.zeros(len(world), np.float32)
            for a, b, ra, rb in caps:
                acc = np.maximum(acc, blob_v2.tapered_capsule_weight(
                    world, a, b, ra, rb, ra * 0.72, rb * 0.72, back))
            return acc

        w_delt = caps_w(delt_caps)
        w_tri = caps_w(tri_caps)
        w_obl = caps_w(obl_caps)

        front = blob_v2.smoothstep(0.02, -0.02, nb)
        rear = blob_v2.smoothstep(-0.02, 0.06, nb)
        if w_upper.sum() > 0 or w_clav.sum() > 0:
            w_delt = np.maximum(
                w_delt, 0.90 * (0.55 * w_clav + 0.50 * w_upper) * front * side_w)
            w_tri = np.maximum(w_tri, 0.95 * w_upper * rear * side_w)

        punch_gate = blob_v2.smoothstep(-0.02 * s, 0.04 * s, side_coord)
        head_kill = 1.0 - blob_v2.smoothstep(0.88, 0.96, t)
        # kill hip/TFL: drop anything below ~0.34 spine-t for obliques
        hip_kill = blob_v2.smoothstep(0.32, 0.38, t)
        pec_kill = 1.0 - (
            blob_v2.smoothstep(0.14 * s, 0.22 * s, -side_coord * 0 + (0.18 * s - np.abs(lat)))
            * blob_v2.smoothstep(0.68, 0.80, t) * front
        )
        # more direct pec gate: suppress medial chest (small |lat|) at pec height
        pec_gate = 1.0 - (
            (1.0 - blob_v2.smoothstep(0.14 * s, 0.20 * s, np.abs(lat)))
            * blob_v2.smoothstep(0.68, 0.82, t)
            * front
        )
        hand_kill = 1.0 - blob_v2.smoothstep(0.34 * s, 0.42 * s, np.abs(lat)) * \
            blob_v2.smoothstep(0.35, 0.55, t)
        w_delt = w_delt * punch_gate * head_kill * pec_gate
        w_tri = w_tri * punch_gate * head_kill * hand_kill
        w_obl = w_obl * punch_gate * head_kill * hip_kill

        edges = blob_v2._edges_np(o.data)
        w_stack = np.stack([
            blob_v2._smooth_field(w_delt, edges, 2),
            blob_v2._smooth_field(w_tri, edges, 2),
            blob_v2._smooth_field(w_obl, edges, 2),
        ], 1)
        counts = v1.write_jab_mcol(o, w_stack)
        for k, v in counts.items():
            totals[k] += v
        per.append({"name": o.name, **counts, "verts": int(len(world))})
    return totals, per, info


def pick_keyframes_reach(arm, punch_hand, f0, f1):
    rows = []
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        m = punch_metrics_any(arm, punch_hand)
        if m:
            rows.append((f, m))
    if not rows:
        mid = int((f0 + f1) // 2)
        return {"guard": int(f0), "mid": mid, "full": int(f1), "series": []}
    full_f, full_m = max(rows, key=lambda t: t[1]["reach"])
    before = [r for r in rows if r[0] <= full_f]
    guard_f, guard_m = min(before, key=lambda t: t[1]["reach"])
    target = guard_m["reach"] + 0.5 * (full_m["reach"] - guard_m["reach"])
    mid_f, mid_m = min(before, key=lambda t: abs(t[1]["reach"] - target))
    return {
        "guard": int(guard_f),
        "mid": int(mid_f),
        "full": int(full_f),
        "guard_reach": float(guard_m["reach"]),
        "mid_reach": float(mid_m["reach"]),
        "full_reach": float(full_m["reach"]),
        "guard_hand_z": float(guard_m["hand_z"]),
        "full_hand_z": float(full_m["hand_z"]),
        "series": [{"f": int(f), "reach": float(m["reach"]),
                    "hand_z": float(m["hand_z"])} for f, m in rows],
    }


def render_view(meshes, out_png, yaw=YAW, elev=ELEV, frame_meshes=None):
    # remove old cameras/lights so each still shares the same recipe
    for o in list(bpy.data.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)
    frame = frame_meshes or meshes
    # temporarily swap bound_box camera for vertex AABB by passing a
    # proxy: _setup_camera uses _world_dims(bound_box). We monkeypatch
    # look-at using vertex dims via a tiny wrapper object list — instead,
    # call _setup_camera on frame meshes after printing debug.
    mn, mx, dim = world_dims_verts(frame)
    print(f"CAM frame dim={list(dim)} mn={list(mn)} mx={list(mx)}", flush=True)
    cam = _setup_camera(frame, yaw, elev)
    # If bound_box lied, rebuild camera from vertex dims
    look = Vector(((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, mn.z + 0.52 * dim.z))
    height = max(dim.z, 0.01)
    v_ang = 2 * math.atan(12.0 / 50.0)
    dist = (height * (1 + 2 * 0.08) / 2) / math.tan(v_ang / 2)
    dist *= 0.92
    yaw_r = math.radians(yaw)
    elev_r = math.radians(elev)
    direction = Vector((
        math.sin(yaw_r) * math.cos(elev_r),
        -math.cos(yaw_r) * math.cos(elev_r),
        math.sin(elev_r),
    ))
    loc = look + direction.normalized() * dist
    cam.location = loc
    cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    print(f"CAM loc={list(loc)} look={list(look)} dist={dist:.3f}", flush=True)
    _setup_world_lights(cam)
    _render(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="boxer_jab_tishkok")
    ap.add_argument("--gltf", default=TISHKOK)
    ap.add_argument("--bvh", default=ACCAD_BVH)
    args = ap.parse_args(_argv())
    os.makedirs(args.out_dir, exist_ok=True)
    facts = {"ok": False, "name": args.name}

    _reset()
    try:
        groups = prepare_tishkok(args.gltf)
        facts["shells"] = {
            k: (None if v is None else {
                "name": v.name,
                "verts": len(v.data.vertices),
                "tris": len(v.data.polygons),
            })
            for k, v in groups.items()
        }
        vis = [o for o in groups.values() if o is not None]
        mn, mx, dim = _world_dims(vis)
        facts["height_m"] = float(dim.z)
        facts["bbox_min"] = list(mn)
        facts["bbox_max"] = list(mx)

        meta = fit_metarig(groups)
        # Generate works after addon_enable, but the generated DEF bones are
        # locked by ORG/MCH constraints, so keyframing them does nothing.
        # The brief asked to fit the human metarig to the A-pose and bind —
        # deform with that fitted metarig and keyframe it directly.
        generated = False
        generate_note = "skipped; deform with fitted metarig"
        try:
            gen, gok = generate_rig(meta)
            generate_note = (
                f"generated={gok} name={gen.name if gen else None}; "
                "not used for deform (DEF bones are constrained)"
            )
            if gok and gen is not None and gen != meta:
                hide_obj(gen)
            # fit_metarig's metarig must stay visible/usable
            show_obj(meta)
            meta.hide_render = True  # armature itself not rendered
        except Exception as e:
            generate_note = f"generate exception {type(e).__name__}: {e}"
        rig = meta
        facts["rig"] = {
            "name": rig.name,
            "generated": generated,
            "generate_note": generate_note,
            "bones": [b.name for b in rig.data.bones if b.use_deform],
            "n_deform": sum(1 for b in rig.data.bones if b.use_deform),
            "n_total": len(rig.data.bones),
        }

        # auto-weight body shells (body + shorts); ears were joined into body
        for key in ("body", "shorts"):
            if groups.get(key) is not None:
                auto_weight(groups[key], rig)

        hand_L = pick_bone(rig, "DEF-hand.L", "hand.L")
        hand_R = pick_bone(rig, "DEF-hand.R", "hand.R")
        foot_L = pick_bone(rig, "DEF-foot.L", "foot.L")
        foot_R = pick_bone(rig, "DEF-foot.R", "foot.R")
        facts["parent_bones"] = {
            "glove_l": hand_L, "glove_r": hand_R,
            "shoe_l": foot_L, "shoe_r": foot_R,
        }
        jumps = {}
        if groups["glove_l"] and hand_L:
            jumps["glove_l"] = rigid_parent_bone(groups["glove_l"], rig, hand_L)
        if groups["glove_r"] and hand_R:
            jumps["glove_r"] = rigid_parent_bone(groups["glove_r"], rig, hand_R)
        if groups["shoe_l"] and foot_L:
            jumps["shoe_l"] = rigid_parent_bone(groups["shoe_l"], rig, foot_L)
        if groups["shoe_r"] and foot_R:
            jumps["shoe_r"] = rigid_parent_bone(groups["shoe_r"], rig, foot_R)
        facts["parent_jumps_m"] = jumps

        # glow at rest, body shells only (not gloves/shoes)
        glow_meshes = [groups[k] for k in ("body", "shorts") if groups.get(k)]
        totals, per, ginfo = paint_jab_glow_strong(glow_meshes, punch_side="right")
        facts["glow"] = {
            "totals": totals, "per": per,
            "em_str": EM_STR,
            "frame": {k: (list(v) if hasattr(v, "__iter__") else v)
                      for k, v in (ginfo or {}).items()
                      if k in ("note",) or True},
        }
        # ginfo may contain ndarrays — sanitize
        safe_info = {}
        for k, v in (ginfo or {}).items():
            try:
                json.dumps(v)
                safe_info[k] = v
            except TypeError:
                try:
                    safe_info[k] = list(v)
                except Exception:
                    safe_info[k] = str(v)
        facts["glow"]["frame"] = safe_info
        mat = blob_v2.punch_material(em_str=EM_STR)
        blob_v2.assign_mat(vis, mat)

        # rest still — bind check
        rest_png = os.path.join(args.out_dir, f"{args.name}_rest.png")
        frame_m = [groups[k] for k in ("body", "shorts") if groups.get(k)]
        for k in ("glove_l", "glove_r", "shoe_l", "shoe_r"):
            if groups.get(k):
                print(f"CENTROID {k}", list(centroid(groups[k])), flush=True)
        render_view(vis, rest_png, frame_meshes=frame_m)
        facts["rest_png"] = rest_png
        print("WROTE", rest_png, flush=True)

        # ACCAD import + retarget
        accad = import_accad(args.bvh)
        action = accad.animation_data.action if accad.animation_data else None
        if action is None:
            raise RuntimeError("ACCAD armature has no action")
        f0, f1 = int(action.frame_range[0]), int(action.frame_range[1])
        facts["accad"] = {
            "frames": [f0, f1],
            "action": action.name,
            "n_bones": len(accad.data.bones),
        }
        face_info = face_accad_minus_y(accad, f0)
        facts["accad"]["facing"] = face_info

        mapping, missing = resolve_mapping(rig)
        facts["mapping"] = mapping
        facts["mapping_missing"] = missing
        bake_world_aim(accad, rig, mapping, f0, f1)
        facts["retarget"] = "arm_only_world_aim"

        punch_hand = mapping.get("RightHand") or hand_R
        keys = pick_keyframes_reach(rig, punch_hand, f0, f1)
        facts["keyframes"] = {k: keys[k] for k in keys if k != "series"}
        facts["reach_series"] = keys.get("series")
        facts["punch_side"] = "right"
        facts["punch_hand"] = punch_hand

        # per-shot metrics vs head (guard height / extension)
        shot_metrics = {}
        for label in ("guard", "mid", "full"):
            f = keys[label]
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            m = punch_metrics_any(rig, punch_hand)
            guard_hand = mapping.get("LeftHand") or hand_L
            g = punch_metrics_any(rig, guard_hand) if guard_hand else None
            head = v1.bone_world(rig, mapping.get("Head") or pick_bone(
                rig, "DEF-head", "DEF-spine.006", "head"))
            shot_metrics[label] = {
                "frame": f,
                "punch": m,
                "guard_hand": g,
                "head_z": float(head.z) if head is not None else None,
                "punch_vs_head": (
                    float(m["hand_z"] - head.z) if m and head is not None else None),
                "guard_vs_head": (
                    float(g["hand_z"] - head.z) if g and head is not None else None),
            }
            png = os.path.join(args.out_dir, f"{args.name}_{label}.png")
            render_view(vis, png, frame_meshes=frame_m)
            shot_metrics[label]["png"] = png
            print("WROTE", png, "reach", m["reach"] if m else None, flush=True)
        facts["shots"] = shot_metrics
        facts["ok"] = True
        facts["gloves_are_fists"] = True
        facts["motion"] = "ACCAD Male2_E2_JabRight.bvh (CC BY 3.0)"
        facts["credit"] = "Tishkok (CC BY 4.0); ACCAD/OSU mocap (CC BY 3.0)"
    except Exception:
        facts["ok"] = False
        facts["error"] = traceback.format_exc()
        print("ERROR", facts["error"], flush=True)

    fact_path = os.path.join(args.out_dir, f"{args.name}_facts.json")
    with open(fact_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("FACTS", fact_path, flush=True)


if __name__ == "__main__":
    main()
