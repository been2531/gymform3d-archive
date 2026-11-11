#!/usr/bin/env python3
"""Inspect two boxer bodies, retarget the same UAL jab, render 3 clay stills each.

  nice -n 19 blender -b --python tools/rig3d/eval_boxer_jab.py -- \
      --out-dir output/_basemodels/eval --name boxer_jab
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval_base_model import (  # noqa: E402
    _apply_scale_to_metres,
    _import,
    _render,
    _reset,
    _setup_camera,
    _setup_world_lights,
    _world_dims,
    _argv,
    measure,
)
import blob_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
TISHKOK = os.path.join(
    ROOT, "output/_basemodels/boxer/tishkok_stylized_boxer/scene.gltf")
QUATERNIUS = os.path.join(
    ROOT,
    "output/_basemodels/boxer/quaternius_ubc/"
    "Universal Base Characters[Standard]/Base Characters/"
    "Godot - UE/Superhero_Male_FullBody.gltf")
UAL_GLB = os.path.join(
    ROOT,
    "output/_basemodels/boxer/ual_standard/"
    "Animation Library[Standard]/Godot/AnimationLibrary_Godot_Standard.glb")
ACCAD_BVH = os.path.join(
    ROOT, "output/_basemodels/boxer/accad/Male2_E2_JabRight.bvh")

JAB_ACTION = "Punch_Jab_Rig"
YAW = 40.0   # 3/4; refined after facing detect
ELEV = 10.0
RES = (540, 960)

# UAL Rigify DEF-*  ->  Quaternius / UE mannequin
UAL_TO_UE = {
    "DEF-hips": "pelvis",
    "DEF-spine.001": "spine_01",
    "DEF-spine.002": "spine_02",
    "DEF-spine.003": "spine_03",
    "DEF-neck": "neck_01",
    "DEF-head": "Head",
    "DEF-shoulder.L": "clavicle_l",
    "DEF-upper_arm.L": "upperarm_l",
    "DEF-forearm.L": "lowerarm_l",
    "DEF-hand.L": "hand_l",
    "DEF-shoulder.R": "clavicle_r",
    "DEF-upper_arm.R": "upperarm_r",
    "DEF-forearm.R": "lowerarm_r",
    "DEF-hand.R": "hand_r",
    "DEF-thigh.L": "thigh_l",
    "DEF-shin.L": "calf_l",
    "DEF-foot.L": "foot_l",
    "DEF-toe.L": "ball_l",
    "DEF-thigh.R": "thigh_r",
    "DEF-shin.R": "calf_r",
    "DEF-foot.R": "foot_r",
    "DEF-toe.R": "ball_r",
}

JAB_ORDER = ("front_delt", "triceps", "obliques")
JAB_COL = {
    "front_delt": np.array([0.55, 0.16, 1.00], np.float32),
    "triceps": np.array([1.00, 0.72, 0.04], np.float32),
    "obliques": np.array([1.00, 0.34, 0.08], np.float32),
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


def apply_object_transforms(objs):
    if not objs:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def bone_world(arm, name):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    return arm.matrix_world @ pb.head


def bone_tail_world(arm, name):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    return arm.matrix_world @ pb.tail


def list_bones(arm):
    return [b.name for b in arm.data.bones]


def hide_named_meshes(meshes, keys):
    kept, hidden = [], []
    keys = [k.lower() for k in keys]
    for o in meshes:
        n = (o.name or "").lower()
        if any(k in n for k in keys):
            hide_obj(o)
            hidden.append(o.name)
        else:
            kept.append(o)
    return kept, hidden


def detect_facing(meshes):
    """+1 if the character's face/chest points +Y, else use bbox (fallback +Y)."""
    mn, mx, dim = _world_dims(meshes)
    hip_z = mn.z + 0.52 * dim.z
    # gather verts in head band
    pts = []
    for o in meshes:
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co
            if p.z > mn.z + 0.82 * dim.z:
                pts.append(p)
    if len(pts) < 20:
        return Vector((0, 1, 0)), "fallback_+Y"
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    # nose / face usually the farthest outlier from the head centroid in XY
    farthest = max(pts, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)
    face = Vector((farthest.x - cx, farthest.y - cy, 0.0))
    if face.length < 1e-6:
        return Vector((0, 1, 0)), "fallback_+Y"
    face.normalize()
    return face, "head_outlier"


def retarget_frame(src_arm, tgt_arm, mapping):
    """Copy rest-relative *rotations* only.

    Copying locations as well stretches the target: UAL Rigify bones and the
    UE mannequin do not share rest-pose lengths or parent-space offsets.
    matrix_basis is the pose-on-top-of-rest transform; both skeletons use
    Blender's +Y-along-bone convention after import, so the quaternion copy
    preserves a T-pose-to-T-pose jab without exploding limb lengths.
    """
    for src_name, tgt_name in mapping.items():
        src = src_arm.pose.bones.get(src_name)
        tgt = tgt_arm.pose.bones.get(tgt_name)
        if src is None or tgt is None:
            continue
        tgt.rotation_mode = "QUATERNION"
        tgt.rotation_quaternion = src.matrix_basis.to_quaternion()


def bake_retarget(src_arm, tgt_arm, mapping, f0, f1):
    for pb in tgt_arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    missing = []
    for s, t in mapping.items():
        if src_arm.pose.bones.get(s) is None:
            missing.append(f"src:{s}")
        if tgt_arm.pose.bones.get(t) is None:
            missing.append(f"tgt:{t}")
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        retarget_frame(src_arm, tgt_arm, mapping)
        bpy.context.view_layer.update()
        for _src, tgt_name in mapping.items():
            pb = tgt_arm.pose.bones.get(tgt_name)
            if pb is None:
                continue
            pb.keyframe_insert(data_path="rotation_quaternion", frame=f)
    return missing


def align_and_transfer(donor_mesh, target_mesh, armature):
    """Scale/place target onto donor, copy weights, bind to armature."""
    notes = {}
    dmn, dmx, ddim = _world_dims([donor_mesh])
    tmn, tmx, tdim = _world_dims([target_mesh])
    src_h = max(float(tdim.z), 1e-6)
    dst_h = max(float(ddim.z), 1e-6)
    factor = dst_h / src_h
    target_mesh.scale *= factor
    apply_object_transforms([target_mesh])
    tmn, tmx, tdim = _world_dims([target_mesh])
    # match feet (z) and mid-hip XY
    d_hip = Vector(((dmn.x + dmx.x) * 0.5, (dmn.y + dmx.y) * 0.5, dmn.z))
    t_hip = Vector(((tmn.x + tmx.x) * 0.5, (tmn.y + tmx.y) * 0.5, tmn.z))
    target_mesh.location += (d_hip - t_hip)
    apply_object_transforms([target_mesh])
    notes["scale"] = float(factor)
    notes["donor_height"] = float(dst_h)
    notes["target_height_before"] = float(src_h)

    for vg in donor_mesh.vertex_groups:
        if target_mesh.vertex_groups.get(vg.name) is None:
            target_mesh.vertex_groups.new(name=vg.name)
    bpy.ops.object.select_all(action="DESELECT")
    target_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_mesh
    mod = target_mesh.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    mod.object = donor_mesh
    mod.use_vert_data = True
    try:
        mod.data_types_verts = {"VGROUP_WEIGHTS"}
    except Exception:
        pass
    try:
        mod.vert_mapping = "POLYINTERP_NEAREST"
    except Exception:
        mod.vert_mapping = "NEAREST_FACE"
    try:
        mod.layers_vgroup_select_src = "ALL"
        mod.layers_vgroup_select_dst = "NAME"
    except Exception:
        pass
    try:
        bpy.ops.object.datalayout_transfer(modifier=mod.name)
    except Exception as e:
        notes["datalayout_error"] = f"{type(e).__name__}: {e}"
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        notes["transfer"] = "data_transfer"
    except Exception as e:
        notes["transfer_apply_error"] = f"{type(e).__name__}: {e}"
        try:
            target_mesh.modifiers.remove(mod)
        except Exception:
            pass
        # fallback: automatic weights
        bpy.ops.object.select_all(action="DESELECT")
        target_mesh.select_set(True)
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        notes["transfer"] = "armature_auto_fallback"
        return notes

    existing = [m for m in target_mesh.modifiers if m.type == "ARMATURE"]
    if existing:
        existing[0].object = armature
    else:
        am = target_mesh.modifiers.new("Armature", "ARMATURE")
        am.object = armature
    target_mesh.parent = armature
    notes["groups"] = len(target_mesh.vertex_groups)
    return notes


def write_jab_mcol(obj, weights):
    n = len(weights)
    acc = np.zeros((n, 3), np.float32)
    for i, name in enumerate(JAB_ORDER):
        acc += weights[:, i:i + 1] * JAB_COL[name]
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
    return {name: int((weights[:, i] > 1e-4).sum()) for i, name in enumerate(JAB_ORDER)}


def vg_named(obj, pred):
    idxs = []
    for gi, vg in enumerate(obj.vertex_groups):
        if pred(vg.name.lower()):
            idxs.append(gi)
    return idxs


def paint_jab_glow(meshes, punch_side="right"):
    """Soft blobs + rig-weight tint on punching-side front delt, triceps, obliques."""
    dummy_cam = Vector((0.0, -1.0, 0.2))
    hip, spine, side, back, slen, info = blob_v2.estimate_anatomic_frame(
        meshes, dummy_cam)
    # Force `back` to be anatomical back: opposite the detected face if we can.
    totals = {n: 0 for n in JAB_ORDER}
    per = []
    want_right = str(punch_side).lower().startswith("r")
    for o in meshes:
        world, no = blob_v2._world_co_no(o)
        if not len(world):
            continue
        rel = world - hip
        t = (rel @ spine) / max(slen, 1e-8)
        lat = rel @ side
        bck = rel @ back
        nb = no @ back
        s = float(slen / 0.55)

        # which sign of `lat` is the punching side, from vertex groups if present
        r_idx = vg_named(o, lambda n: n.endswith("_r") or n.endswith(".r")
                         or "right" in n)
        l_idx = vg_named(o, lambda n: n.endswith("_l") or n.endswith(".l")
                         or "left" in n)
        wr = blob_v2.vg_weight_array(o, r_idx)
        wl = blob_v2.vg_weight_array(o, l_idx)
        punch_lat = 1.0
        if wr.sum() > 1 and wl.sum() > 1:
            r_med = float(np.median(lat[wr > 0.25])) if (wr > 0.25).any() else 0.0
            l_med = float(np.median(lat[wl > 0.25])) if (wl > 0.25).any() else 0.0
            # if right groups sit on +lat, punch_sign = +1 for right
            right_sign = 1.0 if r_med >= l_med else -1.0
            punch_lat = right_sign if want_right else -right_sign
        else:
            punch_lat = 1.0 if want_right else -1.0

        side_coord = lat * punch_lat  # positive on punching side

        # Rig-weight hints
        upper_i = vg_named(o, lambda n: any(k in n for k in (
            "upperarm", "upper_arm", "uparm", "humerus")))
        clav_i = vg_named(o, lambda n: any(k in n for k in (
            "clavicle", "shoulder", "collar")))
        spine_i = vg_named(o, lambda n: any(k in n for k in (
            "spine", "torso", "chest", "pelvis")))
        w_upper = blob_v2.vg_weight_array(o, upper_i)
        w_clav = blob_v2.vg_weight_array(o, clav_i)
        w_spine = blob_v2.vg_weight_array(o, spine_i)
        if want_right:
            side_w = wr / np.maximum(wr + wl, 1e-6)
        else:
            side_w = wl / np.maximum(wr + wl, 1e-6)
        if wr.sum() + wl.sum() < 1:
            side_w = blob_v2.smoothstep(0.02 * s, 0.08 * s, side_coord)

        # Capsules (anatomic), punching side only — no mirror.
        def pt(tt, lat_m, bck_m):
            return hip + (tt * slen) * spine + (lat_m * s * punch_lat) * side + (bck_m * s) * back

        # front delt: shoulder cap, forward of the body
        delt_caps = [
            (pt(0.82, 0.18, -0.06), pt(0.78, 0.26, -0.08), 0.075 * s, 0.055 * s),
            (pt(0.80, 0.16, -0.02), pt(0.76, 0.24, -0.07), 0.065 * s, 0.050 * s),
        ]
        # triceps: posterior upper arm
        tri_caps = [
            (pt(0.78, 0.24, 0.02), pt(0.58, 0.28, 0.01), 0.050 * s, 0.042 * s),
            (pt(0.74, 0.22, 0.04), pt(0.56, 0.26, 0.03), 0.046 * s, 0.038 * s),
        ]
        # obliques: punching-side waist
        obl_caps = [
            (pt(0.48, 0.13, -0.01), pt(0.28, 0.14, 0.01), 0.070 * s, 0.055 * s),
            (pt(0.42, 0.11, -0.04), pt(0.26, 0.13, -0.02), 0.055 * s, 0.045 * s),
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

        # Rig-weight tint (follows bind) blended in.
        front = blob_v2.smoothstep(0.02, -0.02, nb)  # facing away from anatomical back
        rear = blob_v2.smoothstep(-0.02, 0.06, nb)
        if w_upper.sum() > 0 or w_clav.sum() > 0:
            w_delt = np.maximum(w_delt, 0.85 * (0.65 * w_clav + 0.55 * w_upper) * front * side_w)
            w_tri = np.maximum(w_tri, 0.90 * w_upper * rear * side_w)
        if w_spine.sum() > 0:
            waist = blob_v2.soft_range(t, 0.22, 0.50, 0.06)
            w_obl = np.maximum(w_obl, 0.75 * w_spine * waist * blob_v2.smoothstep(0.06 * s, 0.12 * s, side_coord))

        # Keep colour off the opposite flank and off the hands/head.
        punch_gate = blob_v2.smoothstep(-0.02 * s, 0.04 * s, side_coord)
        head_kill = 1.0 - blob_v2.smoothstep(0.88, 0.96, t)
        hand_kill = 1.0 - blob_v2.smoothstep(0.34 * s, 0.42 * s, np.abs(lat)) * blob_v2.smoothstep(0.35, 0.55, t)
        w_delt = w_delt * punch_gate * head_kill
        w_tri = w_tri * punch_gate * head_kill * hand_kill
        w_obl = w_obl * punch_gate * head_kill

        edges = blob_v2._edges_np(o.data)
        w_stack = np.stack([
            blob_v2._smooth_field(w_delt, edges, 2),
            blob_v2._smooth_field(w_tri, edges, 2),
            blob_v2._smooth_field(w_obl, edges, 2),
        ], 1)
        counts = write_jab_mcol(o, w_stack)
        for k, v in counts.items():
            totals[k] += v
        per.append({"name": o.name, **counts, "verts": int(len(world))})
    return totals, per, info


def punch_metrics(arm, punch_hand):
    hand = bone_world(arm, punch_hand)
    pelvis = bone_world(arm, "pelvis") or bone_world(arm, "DEF-hips")
    head = bone_world(arm, "Head") or bone_world(arm, "DEF-head")
    if hand is None or pelvis is None:
        return None
    reach = (hand - pelvis).length
    return {
        "hand": list(hand),
        "pelvis": list(pelvis),
        "head": list(head) if head else None,
        "reach": float(reach),
        "hand_z": float(hand.z),
    }


def pick_punch_hand(arm, f0, f1):
    best = {"L": 0.0, "R": 0.0}
    for side, bname in (("L", "hand_l"), ("R", "hand_r")):
        reaches = []
        for f in range(int(f0), int(f1) + 1):
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            m = punch_metrics(arm, bname)
            if m:
                reaches.append(m["reach"])
        if reaches:
            best[side] = max(reaches) - min(reaches)
    side = "R" if best["R"] >= best["L"] else "L"
    return side, best


def pick_keyframes(arm, punch_hand, f0, f1):
    rows = []
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        m = punch_metrics(arm, punch_hand)
        if m:
            rows.append((f, m["reach"]))
    if not rows:
        mid = int((f0 + f1) // 2)
        return {"guard": int(f0), "mid": mid, "full": int(f1), "series": []}
    # peak = max reach
    full_f, full_r = max(rows, key=lambda t: t[1])
    before = [r for r in rows if r[0] <= full_f]
    guard_f, guard_r = min(before, key=lambda t: t[1])
    target = guard_r + 0.5 * (full_r - guard_r)
    mid_f, _ = min(before, key=lambda t: abs(t[1] - target))
    return {
        "guard": int(guard_f),
        "mid": int(mid_f),
        "full": int(full_f),
        "guard_reach": float(guard_r),
        "mid_reach": float(target),
        "full_reach": float(full_r),
        "series": [{"f": int(f), "reach": float(r)} for f, r in rows],
    }


def assign_glow(meshes):
    mat = blob_v2.punch_material(em_str=0.82)
    blob_v2.assign_mat(meshes, mat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="boxer_jab")
    ap.add_argument("--tishkok", default=TISHKOK)
    ap.add_argument("--quaternius", default=QUATERNIUS)
    ap.add_argument("--ual", default=UAL_GLB)
    ap.add_argument("--yaw", type=float, default=YAW)
    ap.add_argument("--elev", type=float, default=ELEV)
    args = ap.parse_args(_argv())
    os.makedirs(args.out_dir, exist_ok=True)
    facts = {"blender": bpy.app.version_string, "clip": JAB_ACTION,
             "clip_source": "Quaternius UAL Standard (CC0) via OpenGameArt"}

    _reset()

    # --- Quaternius ---
    print("import quaternius", args.quaternius, flush=True)
    q_meshes, q_arms = _import(args.quaternius)
    if q_arms:
        blob_v2.clear_to_rest_pose(q_arms)
    q_scale = _apply_scale_to_metres(q_meshes)
    q_body, q_hidden = hide_named_meshes(
        q_meshes, ("eyebrow", "eye", "hair", "lash", "icosphere", "sphere"))
    print("quaternius meshes", [o.name for o in q_meshes],
          "hidden", q_hidden, "arms", [a.name for a in q_arms], flush=True)
    q_metrics = measure(q_body or q_meshes, q_arms)
    facts["quaternius"] = {
        "scale": q_scale,
        "hidden": q_hidden,
        "bones": list_bones(q_arms[0]) if q_arms else [],
        "bone_count": int(q_metrics.get("armature_bones") or 0),
        "metrics": q_metrics,
        "license": "CC0 1.0 (Quaternius Universal Base Characters Standard)",
    }

    # --- Tishkok ---
    print("import tishkok", args.tishkok, flush=True)
    t_meshes, t_arms = _import(args.tishkok)
    if t_arms:
        blob_v2.clear_to_rest_pose(t_arms)
    t_scale = _apply_scale_to_metres(t_meshes)
    t_body, t_hidden = hide_named_meshes(t_meshes, ("plinth", "stand", "base"))
    print("tishkok meshes", [o.name for o in t_meshes],
          "arms", [a.name for a in t_arms], flush=True)
    t_metrics = measure(t_body or t_meshes, t_arms)
    facts["tishkok"] = {
        "scale": t_scale,
        "hidden": t_hidden,
        "bones": list_bones(t_arms[0]) if t_arms else [],
        "bone_count": int(t_metrics.get("armature_bones") or 0),
        "metrics": t_metrics,
        "license": "CC BY 4.0 (Tishkok / Sketchfab uid 537003fb4788494fbb48c66412eb5dda)",
    }

    if not q_arms:
        raise SystemExit("Quaternius imported with no armature")
    q_arm = q_arms[0]
    q_main = next((o for o in q_body if "superhero" in o.name.lower()
                   or "male" in o.name.lower()), q_body[0])
    t_main = t_body[0]

    # Align Tishkok onto Quaternius and steal its weights (requested auto-rig path).
    print("transfer weights quaternius -> tishkok", flush=True)
    xfer = align_and_transfer(q_main, t_main, q_arm)
    facts["tishkok"]["weight_transfer"] = xfer
    print("transfer", xfer, flush=True)

    # --- UAL clip ---
    print("import ual", args.ual, flush=True)
    before_objs = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    u_meshes, u_arms = _import(args.ual)
    new_actions = [a for a in bpy.data.actions if a not in before_actions]
    print("ual meshes", [o.name for o in u_meshes],
          "arms", [a.name for a in u_arms],
          "new_actions", [a.name for a in new_actions], flush=True)
    for o in u_meshes:
        hide_obj(o)
    if not u_arms:
        raise SystemExit("UAL imported with no armature")
    u_arm = u_arms[0]
    # Scale ONLY the UAL armature/mesh. Do not call _apply_scale_to_metres —
    # that function rotates every scene root and would tumble the bodies.
    umn, umx, udim = _world_dims(u_meshes or [u_arm])
    u_h = float(max(udim.x, udim.y, udim.z, 1e-6))
    u_scale = {"height_before": u_h}
    if u_h > 50:
        factor = 0.001
    elif 8 < u_h < 40:
        factor = 0.01
    else:
        factor = 1.0
    u_roots = [o for o in (u_meshes + u_arms) if o.parent is None] or ([u_arm] + list(u_meshes))
    if factor != 1.0:
        for o in u_roots:
            o.scale = o.scale * factor
        apply_object_transforms(u_roots)
        umn, umx, udim = _world_dims(u_meshes or [u_arm])
        u_h = float(max(udim.x, udim.y, udim.z, 1e-6))
    # If UAL came in Y-up, stand just those roots.
    if udim.y > udim.z * 1.15:
        bpy.ops.object.select_all(action="DESELECT")
        for o in u_roots:
            o.select_set(True)
        bpy.context.view_layer.objects.active = u_roots[0]
        bpy.ops.transform.rotate(value=math.radians(90), orient_axis="X", orient_type="GLOBAL")
        apply_object_transforms(u_roots)
        u_scale["upright"] = "Y->Z"
    u_scale["height_m"] = u_h
    u_scale["factor"] = factor
    facts["ual"] = {
        "armature": u_arm.name,
        "bones": list_bones(u_arm),
        "bone_count": len(u_arm.data.bones),
        "scale": u_scale,
        "actions": [a.name for a in bpy.data.actions if "Punch" in a.name or "punch" in a.name],
    }
    act = bpy.data.actions.get(JAB_ACTION)
    if act is None:
        raise SystemExit(f"missing action {JAB_ACTION}")
    if u_arm.animation_data is None:
        u_arm.animation_data_create()
    u_arm.animation_data.action = act
    f0, f1 = int(round(act.frame_range[0])), int(round(act.frame_range[1]))
    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f1
    print(f"jab frames {f0}-{f1}", flush=True)

    hide_obj(u_arm)

    # Paint in rest pose so colour follows the rig once the jab is applied.
    # UAL jab is a lead (typically left) punch; glow the left side.
    punch_side_guess = "left"
    q_glow = paint_jab_glow([q_main], punch_side_guess)
    t_glow = paint_jab_glow([t_main], punch_side_guess)
    assign_glow([q_main, t_main])
    facts["quaternius"]["glow"] = {"totals": q_glow[0], "per": q_glow[1],
                                   "frame": q_glow[2]}
    facts["tishkok"]["glow"] = {"totals": t_glow[0], "per": t_glow[1],
                                "frame": t_glow[2]}
    print("glow Q", q_glow[0], "T", t_glow[0], flush=True)

    missing = bake_retarget(u_arm, q_arm, UAL_TO_UE, f0, f1)
    facts["retarget_missing"] = missing
    print("retarget missing", missing, flush=True)

    side, travel = pick_punch_hand(q_arm, f0, f1)
    punch_hand = "hand_r" if side == "R" else "hand_l"
    punch_side = "right" if side == "R" else "left"
    facts["punch_side_guess"] = punch_side_guess
    facts["punch_side"] = punch_side
    facts["hand_travel"] = travel
    keys = pick_keyframes(q_arm, punch_hand, f0, f1)
    facts["keyframes"] = {k: keys[k] for k in ("guard", "mid", "full",
                                               "guard_reach", "mid_reach",
                                               "full_reach")}
    print("punch", punch_side, "keys", facts["keyframes"], "travel", travel, flush=True)

    # Facing / camera from Quaternius at guard.
    bpy.context.scene.frame_set(keys["guard"])
    bpy.context.view_layer.update()
    face, face_how = detect_facing([q_main])
    facts["facing"] = {"dir": list(face), "how": face_how}
    # _setup_camera: yaw=0 puts camera at -Y looking toward origin.
    # We want a 3/4 front from the punching side.
    # Character forward ~ `face` in XY.
    fwd = Vector((face.x, face.y, 0.0))
    if fwd.length < 1e-6:
        fwd = Vector((0, 1, 0))
    fwd.normalize()
    # yaw where camera direction matches: dir = (sin yaw, -cos yaw)
    # We want camera in front of the character (along +fwd) and toward punch side.
    right = Vector((fwd.y, -fwd.x, 0.0))  # +90 deg yaw of fwd
    punch_vec = right if punch_side == "right" else -right
    # 3/4 front: 0.75*fwd + 0.55*punch_side
    cam_off = (0.78 * fwd + 0.62 * punch_vec)
    # Convert to the yaw convention of _setup_camera
    # direction = (sin yaw, -cos yaw) should equal cam_off.xy normalized
    d = Vector((cam_off.x, cam_off.y, 0.0)).normalized()
    yaw = math.degrees(math.atan2(d.x, -d.y))
    facts["camera_yaw"] = float(yaw)
    print("facing", facts["facing"], "yaw", yaw, flush=True)

    bodies = [
        ("tishkok", t_main, [q_main]),
        ("quaternius", q_main, [t_main]),
    ]
    shots = [("guard", keys["guard"]), ("mid", keys["mid"]), ("full", keys["full"])]
    rendered = {}
    for body_name, visible, others in bodies:
        show_obj(visible)
        for o in others:
            hide_obj(o)
        # camera from this body's bbox at guard, then reuse
        bpy.context.scene.frame_set(keys["guard"])
        bpy.context.view_layer.update()
        # remove old camera/lights if any
        for ob in list(bpy.data.objects):
            if ob.type in {"CAMERA", "LIGHT"} or ob.name in {"Cam", "Key", "Fill", "Rim"}:
                bpy.data.objects.remove(ob, do_unlink=True)
        cam = _setup_camera([visible], yaw, args.elev, margin=0.12)
        _setup_world_lights(cam)
        for shot, frame in shots:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            out = os.path.join(args.out_dir, f"{args.name}_{body_name}_{shot}.png")
            _render(out, exposure=-0.20)
            rendered.setdefault(body_name, {})[shot] = out
            print("wrote", out, flush=True)

    facts["stills"] = rendered
    facts_path = os.path.join(args.out_dir, f"{args.name}_facts.json")
    with open(facts_path, "w") as f:
        json.dump(facts, f, indent=2, default=str)
    print("facts", facts_path, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
