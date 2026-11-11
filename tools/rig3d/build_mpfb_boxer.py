#!/usr/bin/env python3
"""Build an MPFB male boxer, retarget the v4 jab, fit outfit, render a sheet.

  nice -n 19 blender -b --python tools/rig3d/build_mpfb_boxer.py -- --phase all

Phases: body, retarget, outfit, qc, render, all.
Writes gitignored blends/stills under output/_basemodels/mpfb/ and
output/_basemodels/eval/mpfb_boxer_v1.png (+ jab preview mp4).
Do not commit third-party assets or output/.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import eval_xbot_boxing as xb  # noqa: E402
import boxing_qc as qc  # noqa: E402
from pick_xbot_punches_v3 import analyze_rich, enrich_bones  # noqa: E402

ROOT = Path(xb.ROOT) if hasattr(xb, "ROOT") else Path(__file__).resolve().parents[2]
SRC_DIR = Path(xb.SRC_DIR)
EVAL_DIR = Path(xb.OUT_DIR)
OUT_DIR = ROOT / "output/_basemodels/mpfb"
BLEND = OUT_DIR / "mpfb_boxer_v1.blend"
FACTS = OUT_DIR / "mpfb_boxer_v1_facts.json"
SHEET = EVAL_DIR / "mpfb_boxer_v1.png"
PREVIEW = EVAL_DIR / "mpfb_boxer_v1_jab_preview.mp4"
ASSETS = ROOT / "output/_assets/outfit_candidates"

JAB_FILE = SRC_DIR / "alts/alt04_Jab Cross_BoxingJabCrossMedium.fbx"
JAB = dict(id="alt04_jab_L_f14", side="L", guard=9, full=14, ret=22, ptype="jab")
TARGET_HEIGHT = 1.78

CLAY = (0.65, 0.656, 0.664, 1.0)
CLAY_ROUGH = 0.86
KEY_ENERGY = 78.0
FILL_RATIO = 0.36
RIM_RATIO = 0.55
YAW, ELEV, FRAME_FILL = 35.0, 10.0, 0.72
SAMPLES, EXPOSURE = 96, 0.0
BURGUNDY = (0.18, 0.025, 0.04, 1.0)
CHARCOAL = (0.12, 0.12, 0.125, 1.0)

# v4 X Bot jab reference (from xbot_punch_pick_facts_v4.json)
XBOT_REF = dict(
    fist_max_dev_m=0.0111,
    elbow_above_shoulder_max_m=-0.0601,
    foot_slide_l_m=0.3748,
    foot_slide_r_m=0.3814,
    root_drift_m=0.4444,
    lead_foot_slide_m=0.3748,
)

# Tolerances: motion must not degrade vs X Bot (absolute metres).
TOL = dict(
    fist_max_dev_m=0.020,
    elbow_above_shoulder_max_m=0.020,
    foot_slide_m=0.060,
    root_drift_m=0.080,
)

BOXER_TARGETS = [
    ("torso-vshape-incr", 0.70),
    ("measure-shoulder-dist-incr", 0.35),
    ("measure-waist-circ-decr", 0.45),
    ("torso-muscle-pectoral-incr", 0.30),
    ("torso-muscle-dorsi-incr", 0.40),
    ("stomach-tone-incr", 0.45),
    ("hip-scale-horiz-decr", 0.20),
    ("l-upperarm-shoulder-muscle-incr", 0.40),
    ("r-upperarm-shoulder-muscle-incr", 0.40),
    ("l-upperarm-muscle-incr", 0.28),
    ("r-upperarm-muscle-incr", 0.28),
]


def dyn(pkg, key):
    for amod in sys.modules:
        if amod.endswith(pkg):
            mod = importlib.import_module(amod)
            if hasattr(mod, key):
                return getattr(mod, key)
    raise ValueError(f"no {key} in *{pkg}")


def enable_mpfb():
    import addon_utils
    for name in ("bl_ext.user_default.mpfb", "mpfb"):
        try:
            addon_utils.enable(name, default_set=True, persistent=True)
        except Exception:
            pass
    if "bl_ext.user_default.mpfb" not in bpy.context.preferences.addons and \
       "mpfb" not in bpy.context.preferences.addons:
        raise RuntimeError("MPFB is not installed (expected bl_ext.user_default.mpfb)")


def _sel(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def eval_height(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        zs = [(ev.matrix_world @ v.co).z for v in me.vertices]
        return (max(zs) - min(zs)) if zs else 0.0
    finally:
        ev.to_mesh_clear()


def eval_bounds(objs):
    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in objs:
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        try:
            mw = ev.matrix_world
            pts.extend(mw @ v.co for v in me.vertices)
        finally:
            ev.to_mesh_clear()
    if not pts:
        return Vector(), Vector()
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def mesh_closed_stats(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    ev.to_mesh_clear()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    open_e = sum(1 for e in bm.edges if len(e.link_faces) < 2)
    nonman = sum(1 for e in bm.edges if not e.is_manifold)
    nvert, nface = len(bm.verts), len(bm.faces)
    bm.free()
    return dict(verts=nvert, faces=nface, open_edges=open_e,
                nonmanifold=nonman, closed=(open_e == 0))


def assign_clay(meshes, name="ClayV2"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = CLAY
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = CLAY_ROUGH
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.22
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.22
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        o.active_material = mat
    return mat


def principled(name, color, roughness, metallic=0.0, specular=0.3, coat=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = specular
    if coat and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = 0.12
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def apply_object_mode():
    if bpy.context.view_layer.objects.active and bpy.context.object and \
            bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def apply_shape_keys(obj):
    """Bake current shapekey mix into the mesh and drop the keys."""
    if not obj.data.shape_keys:
        return
    _sel(obj)
    apply_object_mode()
    obj.shape_key_add(name="_mix", from_mix=True)
    while obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1:
        obj.active_shape_key_index = 0
        bpy.ops.object.shape_key_remove(all=False)
    if obj.data.shape_keys:
        obj.active_shape_key_index = 0
        bpy.ops.object.shape_key_remove(all=False)


def feet_on_ground(obj):
    bpy.context.view_layer.update()
    mn, mx = eval_bounds([obj])
    obj.location.z -= mn.z
    bpy.context.view_layer.update()
    _sel(obj)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


# ---------------------------------------------------------------------------
# Phase: body
# ---------------------------------------------------------------------------
def phase_body():
    enable_mpfb()
    HumanService = dyn("mpfb.services.humanservice", "HumanService")
    TargetService = dyn("mpfb.services.targetservice", "TargetService")
    HumanObjectProperties = dyn("mpfb.entities.objectproperties", "HumanObjectProperties")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_mpfb()
    # re-bind after factory reset
    HumanService = dyn("mpfb.services.humanservice", "HumanService")
    TargetService = dyn("mpfb.services.targetservice", "TargetService")
    HumanObjectProperties = dyn("mpfb.entities.objectproperties", "HumanObjectProperties")

    macros = TargetService.get_default_macro_info_dict()
    macros.update(dict(
        gender=1.0, age=0.5, muscle=0.68, weight=0.48,
        proportions=0.58, height=0.54, cupsize=0.0, firmness=0.45,
    ))
    macros["race"] = dict(asian=0.15, caucasian=0.55, african=0.30)
    print("CREATE human macros", macros, flush=True)
    basemesh = HumanService.create_human(
        mask_helpers=True, detailed_helpers=False, extra_vertex_groups=True,
        feet_on_ground=True, scale=0.1, macro_detail_dict=macros)
    basemesh.name = "MPFBBoxer"

    for name, w in BOXER_TARGETS:
        path = TargetService.target_full_path(name)
        if not path:
            print(f"MISSING target {name}", flush=True)
            continue
        TargetService.load_target(basemesh, path, weight=float(w))
        print(f"  target {name} w={w}", flush=True)

    # Calibrate height after targets (they can change stature slightly).
    for i in range(5):
        TargetService.reapply_macro_details(basemesh)
        feet_on_ground(basemesh)
        h = eval_height(basemesh)
        err = TARGET_HEIGHT - h
        print(f"  height iter {i}: {h:.3f}m err={err:+.3f}", flush=True)
        if abs(err) <= 0.008:
            break
        # ~0.151 m per 0.12 macro from earlier calibration
        step = err * (0.12 / 0.151)
        macros["height"] = float(max(0.0, min(1.0, macros["height"] + step)))
        HumanObjectProperties.set_value("height", macros["height"], entity_reference=basemesh)

    TargetService.reapply_macro_details(basemesh)
    feet_on_ground(basemesh)
    height = eval_height(basemesh)
    print(f"FINAL height {height:.3f}m (macro={macros['height']:.3f})", flush=True)

    rig = HumanService.add_builtin_rig(basemesh, "mixamo", import_weights=True)
    rig.name = "MPFBBoxer.rig"
    print(f"RIG {rig.name} bones={len(rig.data.bones)}", flush=True)

    # Bake the boxer shape, but do NOT apply the helper Mask and do NOT
    # merge-by-distance. Applying the mask after the armature (or welding)
    # remaps Mixamo weights and stretches helper-adjacent verts into spikes.
    # The live Mask stays first in the stack; the evaluated surface is already
    # a closed body (verified: 0 open edges).
    apply_shape_keys(basemesh)
    # Ensure Mask evaluates before the armature.
    mods = list(basemesh.modifiers)
    mask = next((m for m in mods if m.type == "MASK"), None)
    if mask is not None:
        idx = basemesh.modifiers.find(mask.name)
        if idx > 0:
            basemesh.modifiers.move(idx, 0)
        print(f"  modifier order: {[m.type for m in basemesh.modifiers]}", flush=True)

    stats = mesh_closed_stats(basemesh)
    print("MESH (evaluated, helpers masked)", stats, flush=True)

    assign_clay([basemesh])
    # Do not apply location on a mesh already parented to the rig.

    # Store pointers
    basemesh["mpfb_role"] = "body"
    rig["mpfb_role"] = "rig"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    facts = {
        "phase": "body",
        "height_m": round(height, 4),
        "height_macro": macros["height"],
        "macros": macros,
        "targets": BOXER_TARGETS,
        "mesh": stats,
        "rig": rig.name,
        "n_bones": len(rig.data.bones),
        "bone_sample": [b.name for b in rig.data.bones][:12],
    }
    FACTS.write_text(json.dumps(facts, indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND))
    print("WROTE", BLEND, flush=True)
    return facts


# ---------------------------------------------------------------------------
# Phase: retarget
# ---------------------------------------------------------------------------
def _find_by_role(role):
    for o in bpy.data.objects:
        if o.get("mpfb_role") == role:
            return o
    return None


def import_xbot_clip(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path))
    bpy.context.view_layer.update()
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError(f"no armature in {path}")
    arm = arms[0]
    arm.name = "XBot.rig"
    for i, m in enumerate(meshes):
        m.name = f"XBot.mesh{i}"
        m["mpfb_role"] = "xbot_mesh"
    arm["mpfb_role"] = "xbot_rig"
    return meshes, arm


def copy_mixamo_action(src_arm, dst_arm, location_scale=1.0):
    """Duplicate the Mixamo action onto the MPFB mixamo rig.

    Mixamo FBX typically has object scale 0.01 (cm) and a 90° X rotation.
    Pose-bone fcurves are in bone-local units of the *source* armature, so every
    location channel must be multiplied by the source object's world scale
    (and an optional stature ratio). Rotations copy as-is: both rigs use the
    Mixamo Y-along-bone convention and nearly identical rest axes.
    """
    ad = src_arm.animation_data
    if not ad or not ad.action:
        raise RuntimeError("source armature has no action")
    src = ad.action
    dst = src.copy()
    dst.name = "MPFB_jab_alt04"
    src_scale = src_arm.matrix_world.to_scale()
    loc_scale = float(location_scale) * float(src_scale.x)
    print(f"  src object scale={tuple(src_scale)} loc_scale={loc_scale:.6f}", flush=True)
    if abs(loc_scale - 1.0) > 1e-6:
        for fc in dst.fcurves:
            data = fc.data_path or ""
            if data.endswith(".location"):
                for kp in fc.keyframe_points:
                    kp.co[1] *= loc_scale
                    if hasattr(kp, "handle_left"):
                        kp.handle_left[1] *= loc_scale
                        kp.handle_right[1] *= loc_scale
    if dst_arm.animation_data is None:
        dst_arm.animation_data_create()
    dst_arm.animation_data.action = dst
    ad = dst_arm.animation_data
    while ad.nla_tracks:
        ad.nla_tracks.remove(ad.nla_tracks[0])
    for pb in dst_arm.pose.bones:
        for c in list(pb.constraints):
            pb.constraints.remove(c)
    return dst



def bake_world_retarget(src_arm, dst_arm, f0, f1):
    """World-space retarget: copy rotations on every shared bone, copy hips location.

    This absorbs the Mixamo 0.01 / 90deg object transform and small rest-axis
    differences (hands, feet) that a raw fcurve copy cannot.
    """
    if dst_arm.animation_data:
        dst_arm.animation_data_clear()
    _sel(dst_arm)
    apply_object_mode()
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    n_rot = n_loc = 0
    for pb in dst_arm.pose.bones:
        if pb.name not in src_arm.pose.bones:
            continue
        # Finger / toe bones have incompatible rest axes; leave them in rest
        # relative to the hand/foot (gloves hide the fingers).
        short = pb.name.split(":")[-1]
        if "Hand" in short or "Toe" in short:
            continue
        cr = pb.constraints.new("COPY_ROTATION")
        cr.name = "RT_rot"
        cr.target = src_arm
        cr.subtarget = pb.name
        cr.target_space = "WORLD"
        cr.owner_space = "WORLD"
        n_rot += 1
        if pb.name == "mixamorig:Hips":
            cl = pb.constraints.new("COPY_LOCATION")
            cl.name = "RT_loc"
            cl.target = src_arm
            cl.subtarget = pb.name
            cl.target_space = "WORLD"
            cl.owner_space = "WORLD"
            n_loc += 1
    print(f"  constraints rot={n_rot} loc={n_loc} bake {f0}-{f1}", flush=True)
    # MPFB Mixamo hand bones are ~47 cm (X Bot is ~8 cm). The oversized
    # tail pulls skinned verts into spikes. Shorten in-place; fingers stay
    # parented at the head.
    bpy.ops.object.mode_set(mode="EDIT")
    for name in ("mixamorig:LeftHand", "mixamorig:RightHand"):
        eb = dst_arm.data.edit_bones.get(name)
        if eb is None:
            continue
        direction = (eb.tail - eb.head)
        if direction.length > 0.10:
            eb.tail = eb.head + direction.normalized() * 0.08
            print(f"  shortened {name} to 0.08m", flush=True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.nla.bake(
        frame_start=int(f0), frame_end=int(f1), step=1,
        only_selected=True, visual_keying=True,
        clear_constraints=True, clear_parents=False,
        use_current_action=False, bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    if not dst_arm.animation_data or not dst_arm.animation_data.action:
        raise RuntimeError("bake produced no action")
    act = dst_arm.animation_data.action
    act.name = "MPFB_jab_alt04"
    # Bake still keys every selected bone. Drop Hand/Toe channels so those
    # bones stay at rest relative to the forearm / foot.
    dropped = 0
    for fc in list(act.fcurves):
        path = fc.data_path or ""
        if "Hand" in path or "Toe" in path:
            act.fcurves.remove(fc)
            dropped += 1
    print(f"  stripped {dropped} Hand/Toe fcurves", flush=True)
    for pb in dst_arm.pose.bones:
        if "Hand" in pb.name or "Toe" in pb.name:
            pb.rotation_euler = (0.0, 0.0, 0.0)
            pb.location = (0.0, 0.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    return act


def hide_role(role, hide=True):
    for o in bpy.data.objects:
        if o.get("mpfb_role") == role:
            o.hide_set(hide)
            o.hide_render = hide
            o.hide_viewport = hide


def phase_retarget():
    enable_mpfb()
    if not BLEND.is_file():
        phase_body()
    else:
        bpy.ops.wm.open_mainfile(filepath=str(BLEND))
        enable_mpfb()

    body = _find_by_role("body")
    rig = _find_by_role("rig")
    if body is None or rig is None:
        raise RuntimeError("body/rig not in blend; run --phase body")

    # Drop a previous X Bot import / stale action so this phase is idempotent.
    for o in list(bpy.data.objects):
        if o.get("mpfb_role") in {"xbot_mesh", "xbot_rig"}:
            bpy.data.objects.remove(o, do_unlink=True)
    if rig.animation_data:
        rig.animation_data_clear()

    xmeshes, xarm = import_xbot_clip(JAB_FILE)
    xbones = xb.bone_map(xarm)
    f0, f1 = xb.frame_range(xarm)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    xscale = xb.scale_ground_center(xarm, xmeshes, xbones)
    xb.face_minus_y(xarm, xbones)
    assign_clay(xmeshes, name="ClayV2")
    print(f"XBOT height={xscale['height_m']:.3f} frames {f0}-{f1} scale={tuple(xarm.scale)}", flush=True)

    mpfb_h = eval_height(body)
    action = bake_world_retarget(xarm, rig, f0, f1)
    print(f"ACTION {action.name}", flush=True)

    # If the bind left the feet off the floor, lift the armature (not the mesh).
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    vis = [body] + [o for o in bpy.data.objects if o.get("mpfb_role") == "body"]
    mn, mx = eval_bounds([body])
    if abs(mn.z) > 0.005:
        print(f"  ground lift {-mn.z:+.4f}m (min_z={mn.z:.4f})", flush=True)
        rig.location.z -= mn.z
        bpy.context.view_layer.update()

    # Ground MPFB after animation binding; leave XY as authored.
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = f0, f1
    sc.render.fps = 30

    facts = json.loads(FACTS.read_text()) if FACTS.is_file() else {}
    facts.update({
        "phase": "retarget",
        "jab_file": str(JAB_FILE),
        "xbot_height_m": xscale["height_m"],
        "mpfb_height_m": round(mpfb_h, 4),
        "location_scale": 1.0,
        "xbot_object_scale": list(xarm.scale),
        "frames": [f0, f1],
        "jab": JAB,
    })
    FACTS.write_text(json.dumps(facts, indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND))
    print("WROTE", BLEND, flush=True)
    return facts


# ---------------------------------------------------------------------------
# Phase: QC
# ---------------------------------------------------------------------------
def qc_armature(arm, label):
    bones = enrich_bones(arm, xb.bone_map(arm))
    f0, f1 = xb.frame_range(arm)
    rows = analyze_rich(arm, bones, f0, f1)
    punch = dict(JAB)
    path = qc.punch_path_metrics(rows, punch, side=JAB["side"])
    # foot slide / root drift over the punch window
    sl = "l" if JAB["side"] == "L" else "r"
    win = [r for r in rows if JAB["guard"] <= r["f"] <= JAB["ret"]]

    def span_xy(key):
        pts = [r[key] for r in win if r.get(key)]
        if not pts:
            return 0.0
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return float(math.hypot(max(xs) - min(xs), max(ys) - min(ys)))

    extra = {
        "foot_slide_l_m": span_xy("foot_l"),
        "foot_slide_r_m": span_xy("foot_r"),
        "root_drift_m": span_xy("hips"),
        "lead_foot_slide_m": span_xy("foot_" + sl),
    }
    path.update(extra)
    print(f"QC {label} path_dev={path['fist_max_dev_m']} "
          f"elb={path['elbow_above_shoulder_max_m']} "
          f"slideL={extra['foot_slide_l_m']:.3f} drift={extra['root_drift_m']:.3f}",
          flush=True)
    return path


def compare_qc(mpfb_qc, xbot_qc=None):
    xbot_qc = xbot_qc or XBOT_REF
    cmp = {}
    pairs = [
        ("fist_max_dev_m", "fist_max_dev_m", TOL["fist_max_dev_m"], False),
        ("elbow_above_shoulder_max_m", "elbow_above_shoulder_max_m",
         TOL["elbow_above_shoulder_max_m"], False),
        ("foot_slide_l_m", "foot_slide_l_m", TOL["foot_slide_m"], False),
        ("root_drift_m", "root_drift_m", TOL["root_drift_m"], False),
    ]
    ok = True
    # "must not degrade": path dev / elbow-above must not get worse by more than tol.
    # For elbow, X Bot is already negative (good). Degradation = becoming larger
    # (more above the shoulder).
    notes = []
    for mk, xk, tol, _ in pairs:
        mv = float(mpfb_qc.get(mk, 0) or 0)
        xv = float(xbot_qc.get(xk, 0) or 0)
        d = mv - xv
        # For signed elbow metric, worse means more positive.
        within = abs(d) <= tol
        worse = False
        if mk in ("fist_max_dev_m",):
            worse = mv > xv + tol
        elif mk == "elbow_above_shoulder_max_m":
            worse = mv > xv + tol
        elif mk in ("foot_slide_l_m", "root_drift_m"):
            worse = mv > xv + tol
        rec = dict(mpfb=round(mv, 4), xbot=round(xv, 4),
                   delta=round(d, 4), tol=tol, within=within, worse=worse)
        cmp[mk] = rec
        if worse:
            ok = False
            notes.append(f"{mk} degraded ({mv:.3f} vs X Bot {xv:.3f}, tol {tol})")
        elif not within:
            notes.append(f"{mk} differs by {d:+.3f} (tol {tol}) but not worse")
    cmp["ok"] = ok
    cmp["notes"] = notes
    return cmp


def phase_qc():
    enable_mpfb()
    if not BLEND.is_file():
        phase_retarget()
    else:
        bpy.ops.wm.open_mainfile(filepath=str(BLEND))
        enable_mpfb()
    rig = _find_by_role("rig")
    xarm = _find_by_role("xbot_rig")
    if rig is None:
        raise RuntimeError("missing MPFB rig")
    mpfb_qc = qc_armature(rig, "mpfb")
    xbot_qc = qc_armature(xarm, "xbot") if xarm else dict(XBOT_REF)
    cmp = compare_qc(mpfb_qc, xbot_qc)
    facts = json.loads(FACTS.read_text()) if FACTS.is_file() else {}
    # traj is large; drop it from the saved facts
    def slim(q):
        q = dict(q)
        q.pop("traj", None)
        return q
    facts["qc"] = {"mpfb": slim(mpfb_qc), "xbot": slim(xbot_qc), "compare": cmp}
    facts["phase"] = "qc"
    FACTS.write_text(json.dumps(facts, indent=2))
    print("COMPARE", json.dumps(cmp, indent=2), flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND))
    return facts


# ---------------------------------------------------------------------------
# Phase: outfit
# ---------------------------------------------------------------------------
def import_glb(path, role):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    # Bake parent (Sketchfab) transforms into the mesh before deleting empties.
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
        _sel(o)
        try:
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        except Exception as e:
            print(f"  apply visual failed for {o.name}: {e}", flush=True)
        o["mpfb_role"] = role
        mn, mx = eval_bounds([o])
        print(f"  imported {o.name} span={tuple(round(x,3) for x in (mx-mn))} loc={tuple(round(c,3) for c in o.location)}", flush=True)
    for o in list(new):
        if o.type != "MESH":
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
    return meshes, [o for o in meshes if o.name in bpy.data.objects]


def _merge_meshes(meshes, name):
    if not meshes:
        return None
    if len(meshes) == 1:
        meshes[0].name = name
        return meshes[0]
    _sel(meshes[0])
    for o in meshes[1:]:
        o.select_set(True)
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    return obj


def _place_and_scale(obj, target_span, axis="max"):
    """Scale and optionally keep the current center. Mesh-data rewrite (no ops)."""
    return normalize_mesh(obj, target_span, place_at=None, axis=axis)


def normalize_mesh(obj, target_span, place_at=None, axis="max"):
    """Bake transforms, scale max (or given) span, optionally move the center."""
    apply_object_mode()
    mw = obj.matrix_world.copy()
    obj.data.transform(mw)
    obj.matrix_world = Matrix.Identity(4)
    obj.data.update()
    bpy.context.view_layer.update()
    pts = [v.co.copy() for v in obj.data.vertices]
    if not pts:
        return 1.0
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    dim = mx - mn
    if axis == "max":
        span = max(dim.x, dim.y, dim.z)
    elif axis == "x":
        span = dim.x
    elif axis == "y":
        span = dim.y
    else:
        span = dim.z
    s = (target_span / span) if span > 1e-10 else 1.0
    center = (mn + mx) * 0.5
    dest = Vector(place_at) if place_at is not None else center
    for v in obj.data.vertices:
        v.co = dest + (v.co - center) * s
    obj.data.update()
    bpy.context.view_layer.update()
    mn2, mx2 = eval_bounds([obj])
    print(f"  normalize {obj.name}: {span:.3f}m -> {target_span:.3f}m "
          f"(s={s:.6f}) z={mn2.z:.3f}:{mx2.z:.3f} at {tuple(round(c,3) for c in dest)}",
          flush=True)
    return s


def _world_set_location(obj, loc):
    obj.location = loc
    bpy.context.view_layer.update()


def estimate_head_width(arm, body):
    """Skull width only, from the evaluated (masked) mesh. Typical male ~0.15-0.16 m."""
    head_mw = bone_matrix(arm, "mixamorig:Head")
    if head_mw is None:
        return 0.16
    hloc = head_mw.translation
    xs = []
    deps = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(deps)
    me = ev.to_mesh()
    mw = ev.matrix_world
    try:
        for v in me.vertices:
            p = mw @ v.co
            if (p.z > hloc.z - 0.04) and (p - hloc).length < 0.12:
                xs.append(p.x)
    finally:
        ev.to_mesh_clear()
    if len(xs) >= 8:
        w = max(xs) - min(xs)
        return float(min(0.18, max(0.14, w)))
    return 0.16


def bone_matrix(arm, name):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    return arm.matrix_world @ pb.matrix


def parent_to_bone(obj, arm, bone_name, keep=True):
    """Bone-parent and keep the current world matrix.

    Blender's BONE parent origin is the bone *tail* (pose matrix × +Y × length).
    """
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        raise RuntimeError(f"missing bone {bone_name}")
    bpy.context.view_layer.update()
    mw = obj.matrix_world.copy()
    apply_object_mode()
    parent_mw = arm.matrix_world @ pb.matrix @ Matrix.Translation((0.0, pb.length, 0.0))
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    if keep:
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)
        obj.matrix_parent_inverse = parent_mw.inverted() @ mw
    bpy.context.view_layer.update()
    print(f"  parented {obj.name} -> {bone_name} world={tuple(round(c,4) for c in obj.matrix_world.translation)}", flush=True)
    mn, mx = eval_bounds([obj])
    print(f"    bounds z={mn.z:.3f}:{mx.z:.3f} span={tuple(round(c,3) for c in (mx-mn))}", flush=True)



def _glove_cuff_and_axis(obj):
    """Use the open boundary (cuff hole) and the farthest vert as the fist axis."""
    mw = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    hole = [e for e in bm.edges if len(e.link_faces) < 2]
    cuff = None
    if hole:
        pts = []
        for e in hole:
            pts.append(mw @ e.verts[0].co)
            pts.append(mw @ e.verts[1].co)
        cuff = sum(pts, Vector()) / len(pts)
    verts = [mw @ v.co for v in obj.data.vertices]
    if cuff is None:
        # fallback: centroid of the lowest 15% along the longest bbox axis
        mn, mx = eval_bounds([obj])
        dim = mx - mn
        axis = "xyz"[max(range(3), key=lambda i: dim[i])]
        lo = mn[axis] + 0.15 * dim[axis]
        sel = [p for p in verts if p[axis] <= lo]
        cuff = sum(sel, Vector()) / max(1, len(sel))
    fist = max(verts, key=lambda p: (p - cuff).length)
    axis = (fist - cuff)
    if axis.length < 1e-6:
        axis = Vector((0, 1, 0))
    else:
        axis.normalize()
    bm.free()
    return cuff, fist, axis


def _orient_object_axis(obj, src_dir, dst_dir):
    src = Vector(src_dir)
    dst = Vector(dst_dir)
    if src.length < 1e-8 or dst.length < 1e-8:
        return
    src.normalize(); dst.normalize()
    if src.dot(dst) > 0.999:
        return
    if src.dot(dst) < -0.999:
        rot = src.rotation_difference(dst)
    else:
        rot = src.rotation_difference(dst)
    mn, mx = eval_bounds([obj])
    center = (mn + mx) * 0.5
    R = rot.to_matrix().to_4x4()
    T = Matrix.Translation(center)
    obj.matrix_world = T @ R @ T.inverted() @ obj.matrix_world
    bpy.context.view_layer.update()


def make_glove_pair(arm, body):
    path = ASSETS / "gloves/artel_classic_red_glove/model.glb"
    meshes, _ = import_glb(path, "glove")
    src = _merge_meshes(meshes, "Glove.R")
    mat = principled("GloveBlackLeather", (0.02, 0.02, 0.022, 1.0),
                     roughness=0.22, metallic=0.18, specular=0.55, coat=0.35)
    src.data.materials.clear()
    src.data.materials.append(mat)

    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    bpy.context.view_layer.update()
    head_w = estimate_head_width(arm, body)
    target = 0.80 * head_w
    _place_and_scale(src, target, "max")
    _sel(src)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print(f"GLOVE head_w={head_w:.3f} target_span={target:.3f}", flush=True)

    def place_on_hand(g, bone_name):
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
        bpy.context.view_layer.update()
        mw = bone_matrix(arm, bone_name)
        if mw is None:
            raise RuntimeError(f"missing {bone_name}")
        # Hand bone Y is along the bone (wrist -> fingers) on both rigs.
        hand_dir = (mw.to_3x3() @ Vector((0, 1, 0))).normalized()
        wrist = mw.translation
        cuff, fist, gaxis = _glove_cuff_and_axis(g)
        _orient_object_axis(g, gaxis, hand_dir)
        cuff, fist, gaxis = _glove_cuff_and_axis(g)
        # Sit the cuff a little past the wrist so the hand is inside the glove.
        g.location += (wrist - 0.015 * hand_dir) - cuff
        bpy.context.view_layer.update()
        parent_to_bone(g, arm, bone_name, keep=True)

    # Duplicate + mirror X for the left glove before parenting.
    _sel(src)
    bpy.ops.object.duplicate()
    left = bpy.context.view_layer.objects.active
    left.name = "Glove.L"
    left["mpfb_role"] = "glove"
    # Apply a -X scale around the glove centroid so the thumb flips sides.
    left.scale.x *= -1
    bpy.context.view_layer.update()
    _sel(left)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    place_on_hand(src, "mixamorig:RightHand")
    place_on_hand(left, "mixamorig:LeftHand")
    return src, left


def make_beanie(arm, body):
    path = ASSETS / "beanie/oluwaphemi_beanie/model.glb"
    meshes, _ = import_glb(path, "beanie")
    hat = _merge_meshes(meshes, "Beanie")
    mat = principled("BeanieCharcoal", CHARCOAL, roughness=0.82, metallic=0.0, specular=0.18)
    hat.data.materials.clear()
    hat.data.materials.append(mat)

    hw = estimate_head_width(arm, body)
    head_mw = bone_matrix(arm, "mixamorig:Head")
    hloc = head_mw.translation if head_mw else Vector((0, 0, 1.6))
    body_mn, body_mx = eval_bounds([body])
    # Place so the hat center sits on the crown, slightly dropped onto the skull.
    place = Vector((hloc.x, hloc.y, min(body_mx.z, hloc.z + 0.12) - 0.02))
    print(f"  beanie head_w={hw:.3f} place={tuple(round(c,3) for c in place)}", flush=True)
    normalize_mesh(hat, hw * 1.22, place_at=place, axis="max")

    # Wrap only the lower cuff so the crown does not collapse into the skull.
    me = hat.data
    vg = hat.vertex_groups.new(name="cuff")
    mn, mx = eval_bounds([hat])
    cut = mn.z + 0.35 * (mx.z - mn.z)
    for i, v in enumerate(me.vertices):
        p = hat.matrix_world @ v.co
        if p.z <= cut:
            vg.add([i], 1.0, "REPLACE")
    sw = hat.modifiers.new("HatWrap", "SHRINKWRAP")
    sw.target = body
    sw.wrap_method = "NEAREST_SURFACEPOINT"
    sw.wrap_mode = "ABOVE_SURFACE"
    sw.offset = 0.006
    sw.vertex_group = "cuff"
    parent_to_bone(hat, arm, "mixamorig:Head", keep=True)
    return hat


def make_trunks(arm, body):
    path = ASSETS / "trunks/sutton_shorts_boxer/model.glb"
    meshes, _ = import_glb(path, "trunks")
    shorts = _merge_meshes(meshes, "Trunks")
    # Drop the BØCK albedo: burgundy satin + charcoal trim by vertex height.
    # After scale we'll assign materials via a simple two-slot approach:
    # use a single burgundy satin (logo gone). A thin charcoal waistband is
    # added with a mix based on a generated vertex color from rest-pose Z.
    mat = bpy.data.materials.new("TrunksBurgundy")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "trim"
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = BURGUNDY          # body (A)  — socket names vary; use Factor from attr
    mix.inputs[7].default_value = (0.07, 0.07, 0.075, 1.0)  # charcoal trim
    # Factor
    try:
        mix.inputs["Factor"].default_value = 0.0
        nt.links.new(attr.outputs["Fac"], mix.inputs["Factor"])
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    except Exception:
        # Blender 4.2 Mix node: Factor is inputs[0], A=inputs[6], B=inputs[7], Result=outputs[2]
        nt.links.new(attr.outputs["Fac"], mix.inputs[0])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.28
    bsdf.inputs["Metallic"].default_value = 0.12
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.55
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.20
        bsdf.inputs["Coat Roughness"].default_value = 0.15
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    shorts.data.materials.clear()
    shorts.data.materials.append(mat)

    # Fit scale: body hip width
    hips = []
    for v in body.data.vertices:
        p = body.matrix_world @ v.co
        if 0.85 <= p.z <= 1.05:
            hips.append(p.x)
    hip_w = (max(hips) - min(hips)) if hips else 0.36
    hips_mw = bone_matrix(arm, "mixamorig:Hips")
    hips_loc = hips_mw.translation if hips_mw else Vector((0, 0, 1.0))
    # Shorts are taller than they are wide: scale by height ~0.42 m, center at pelvis.
    normalize_mesh(shorts, 0.46, place_at=Vector((hips_loc.x, hips_loc.y, hips_loc.z - 0.08)), axis="max")

    # Vertex color "trim" for the top ~8 cm (waistband) and side stripes via X edges.
    me = shorts.data
    if "trim" not in me.color_attributes:
        me.color_attributes.new("trim", "FLOAT_COLOR", "POINT")
    col = me.color_attributes["trim"]
    mn, mx = eval_bounds([shorts])
    waist_cut = mx.z - 0.055
    n = len(me.vertices)
    buf = np.zeros((n, 4), np.float32)
    for i, v in enumerate(me.vertices):
        p = shorts.matrix_world @ v.co
        t = 1.0 if p.z >= waist_cut else 0.0
        buf[i] = (t, t, t, 1.0)
    col.data.foreach_set("color", buf.ravel())

    # Shrinkwrap + surface deform + data transfer weights.
    bind_clothing(shorts, body, arm, offset=0.008, name="Trunks",
                  keep_bones=("Hips", "Spine", "Spine1", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"))
    return shorts


def make_tank(arm, body):
    path = ASSETS / "tank/zyadelbrdan74_sleeveless_shirt/model.glb"
    meshes, _ = import_glb(path, "tank")
    tank = _merge_meshes(meshes, "Tank")
    mat = principled("TankCharcoal", CHARCOAL, roughness=0.68, metallic=0.0, specular=0.2)
    tank.data.materials.clear()
    tank.data.materials.append(mat)

    # Flatten baked pecs slightly before wrap.
    smooth = tank.modifiers.new("FlattenPecs", "SMOOTH")
    smooth.factor = 0.6
    smooth.iterations = 8

    # Scale to chest width
    xs = []
    for v in body.data.vertices:
        p = body.matrix_world @ v.co
        if 1.15 <= p.z <= 1.40:
            xs.append(p.x)
    chest_w = (max(xs) - min(xs)) if xs else 0.38
    spine2 = bone_matrix(arm, "mixamorig:Spine2")
    hips_mw = bone_matrix(arm, "mixamorig:Hips")
    mid = spine2.translation if spine2 else Vector((0, 0, 1.3))
    if hips_mw:
        mid = (mid + hips_mw.translation) * 0.5
        mid.z = (mid.z + (spine2.translation.z if spine2 else mid.z)) * 0.5
    normalize_mesh(tank, 0.52, place_at=mid, axis="max")

    bind_clothing(tank, body, arm, offset=0.008, name="Tank",
                  keep_bones=("Hips", "Spine", "Spine1", "Spine2", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm"))
    return tank


def filter_deform_groups(obj, keep_bones):
    """Keep only Mixamo deform groups whose short name is in keep_bones.

    Short name is the part after the last colon (mixamorig:LeftUpLeg -> LeftUpLeg).
    """
    keep = set(keep_bones)
    removed = []
    for vg in list(obj.vertex_groups):
        short = vg.name.split(":")[-1]
        if short not in keep:
            removed.append(vg.name)
            obj.vertex_groups.remove(vg)
    print(f"  {obj.name}: kept {len(obj.vertex_groups)} vgroups, dropped {len(removed)} "
          f"kept={ [g.name for g in obj.vertex_groups] }", flush=True)


def bind_clothing(cloth, body, arm, offset=0.006, name="Cloth", keep_bones=None):

    """Fit at the bind frame, transfer weights, armature-deform, keep a live shrinkwrap.

    Surface Deform often fails on the MH body (concave ngons). A live shrinkwrap
    after the armature is the reliable no-poke path; corrective smooth is last.
    """
    sc = bpy.context.scene
    sc.frame_set(sc.frame_start)
    bpy.context.view_layer.update()

    # Apply any pre-fit deformers (smooth / shrinkwrap) so the rest mesh is the
    # already-fitted shape. Move each to the top first so we do not bake the armature.
    for mod in list(cloth.modifiers):
        if mod.type in {"SMOOTH", "SHRINKWRAP"}:
            _sel(cloth)
            idx = cloth.modifiers.find(mod.name)
            if idx > 0:
                cloth.modifiers.move(idx, 0)
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as e:
                print(f"  apply {mod.name} skipped: {e}", flush=True)

    # A one-shot shrinkwrap to seat the garment, then apply.
    sw = cloth.modifiers.new(f"{name}WrapBind", "SHRINKWRAP")
    sw.target = body
    sw.wrap_method = "NEAREST_SURFACEPOINT"
    sw.wrap_mode = "ABOVE_SURFACE"
    sw.offset = offset
    _sel(cloth)
    try:
        idx = cloth.modifiers.find(sw.name)
        if idx > 0:
            cloth.modifiers.move(idx, 0)
        bpy.ops.object.modifier_apply(modifier=sw.name)
        print(f"  {name}: bind shrinkwrap applied", flush=True)
    except Exception as e:
        print(f"  {name}: shrinkwrap apply skipped: {e}", flush=True)

    # Weights from the body.
    dt = cloth.modifiers.new(f"{name}Weights", "DATA_TRANSFER")
    dt.object = body
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    _sel(cloth)
    try:
        bpy.ops.object.datalayout_transfer(modifier=dt.name)
        bpy.ops.object.modifier_apply(modifier=dt.name)
        print(f"  {name}: weights transferred, vgroups={len(cloth.vertex_groups)}", flush=True)
    except Exception as e:
        print(f"  {name}: data transfer failed: {e}", flush=True)
    # Drop finger/helper/far-bone groups so the garment cannot spike toward the fists.
    if keep_bones:
        filter_deform_groups(cloth, keep_bones)

    if not any(m.type == "ARMATURE" for m in cloth.modifiers):
        am = cloth.modifiers.new(f"{name}Arm", "ARMATURE")
        am.object = arm

    # Live shrinkwrap after the armature to kill poke-through in every pose.
    live = cloth.modifiers.new(f"{name}WrapLive", "SHRINKWRAP")
    live.target = body
    live.wrap_method = "NEAREST_SURFACEPOINT"
    live.wrap_mode = "ABOVE_SURFACE"
    live.offset = offset

    cs = cloth.modifiers.new(f"{name}Corrective", "CORRECTIVE_SMOOTH")
    cs.factor = 0.4
    cs.iterations = 5
    cs.use_only_smooth = True

    if cloth.parent is not arm:
        mw = cloth.matrix_world.copy()
        cloth.parent = arm
        cloth.parent_type = "OBJECT"
        cloth.matrix_world = mw

    print(f"  {name}: mods={[m.type for m in cloth.modifiers]}", flush=True)


def poke_through_count(cloth, body, frames):
    """Count cloth verts that fall clearly inside the body at given frames."""
    report = {}
    for f in frames:
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        bev = body.evaluated_get(deps)
        cev = cloth.evaluated_get(deps)
        bme = bev.to_mesh()
        cme = cev.to_mesh()
        # Build a BVH from the body.
        try:
            from mathutils.bvhtree import BVHTree
            bvh = BVHTree.FromObject(bev, deps)
        except Exception:
            bev.to_mesh_clear(); cev.to_mesh_clear()
            report[str(f)] = None
            continue
        inside = 0
        total = len(cme.vertices)
        bmw = bev.matrix_world
        cmw = cev.matrix_world
        for v in cme.vertices:
            p = cmw @ v.co
            loc, normal, idx, dist = bvh.find_nearest(p)
            if loc is None:
                continue
            # If the cloth point is more than 3 mm on the inner side of the
            # body surface (against the outward normal), count as poke-through.
            # BVH normal is in world? FromObject returns world coords.
            n = Vector(normal)
            if n.length < 1e-8:
                continue
            # "inside" if (p - loc) dots inward (against outward normal)
            # After shrinkwrap ABOVE_SURFACE, normals should point out.
            if n.dot(p - loc) < -0.003:
                inside += 1
        bev.to_mesh_clear(); cev.to_mesh_clear()
        report[str(f)] = dict(inside=inside, total=total,
                              frac=round(inside / max(1, total), 4))
    return report


def phase_outfit():
    enable_mpfb()
    if not BLEND.is_file():
        phase_retarget()
    else:
        bpy.ops.wm.open_mainfile(filepath=str(BLEND))
        enable_mpfb()
    body = _find_by_role("body")
    rig = _find_by_role("rig")
    if body is None or rig is None:
        raise RuntimeError("run body+retarget first")

    # Idempotent: drop a previous outfit attempt.
    for o in list(bpy.data.objects):
        if o.get("mpfb_role") in {"glove", "beanie", "trunks", "tank"}:
            bpy.data.objects.remove(o, do_unlink=True)

    # Bind clothes in a rest-like frame. Frame 1 of alt04 may already be a
    # boxing guard; that's ok — we wrap to the posed body at bind, then surface
    # deform follows later poses.
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    bpy.context.view_layer.update()

    gloves = make_glove_pair(rig, body)
    hat = make_beanie(rig, body)
    trunks = make_trunks(rig, body)
    tank = make_tank(rig, body)

    frames = [JAB["guard"], JAB["full"], JAB["ret"]]
    poke = {
        "trunks": poke_through_count(trunks, body, frames),
        "tank": poke_through_count(tank, body, frames),
    }
    print("POKE", json.dumps(poke), flush=True)

    facts = json.loads(FACTS.read_text()) if FACTS.is_file() else {}
    facts["outfit"] = {
        "gloves": "ARTEL_3D Classic Red (mirrored, recolored glossy black)",
        "beanie": "oluwaphemi Beanie recolored charcoal",
        "trunks": "Ginny Sutton Shorts Boxer, BØCK albedo dropped, burgundy satin + charcoal waist",
        "tank": "zyadelbrdan74 sleeveless shirt, charcoal, smoothed pecs, shrinkwrap+SDef",
        "poke_through": poke,
    }
    facts["phase"] = "outfit"
    FACTS.write_text(json.dumps(facts, indent=2))
    try:
        workbench_preview()
    except Exception as e:
        print("preview failed:", e, flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND))
    print("WROTE", BLEND, flush=True)
    return facts


def workbench_preview():
    """Fast visual check of outfit at guard / extension / return."""
    body = _find_by_role("body")
    set_column("mpfb_outfit")
    bpy.context.scene.frame_set(JAB["full"])
    bpy.context.view_layer.update()
    mn, mx = eval_bounds([body])
    look = Vector(((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.48))
    setup_v2_camera(look, max(0.5, (mx.z - mn.z) * 1.05))
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "TEXTURE"
    sc.render.resolution_x, sc.render.resolution_y = 360, 640
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    dest_dir = OUT_DIR / "_preview"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, f in (("guard", JAB["guard"]), ("extension", JAB["full"]), ("return", JAB["ret"])):
        sc.frame_set(f)
        bpy.context.view_layer.update()
        dest = dest_dir / f"outfit_{name}.png"
        sc.render.filepath = str(dest)
        bpy.ops.render.render(write_still=True)
        print("  preview", dest, flush=True)


# ---------------------------------------------------------------------------
# Phase: render
# ---------------------------------------------------------------------------
def _area_light(name, loc, size, energy, look_at, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, "AREA")
    d.size = size
    d.energy = energy
    d.spread = 150
    d.color = color
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector(look_at) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def setup_v2_camera(look, height):
    for o in list(bpy.data.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)
    yaw_r, elev_r = math.radians(YAW), math.radians(ELEV)
    direction = Vector((
        math.sin(yaw_r) * math.cos(elev_r),
        -math.cos(yaw_r) * math.cos(elev_r),
        math.sin(elev_r),
    ))
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = 24.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 40
    v_ang = 2 * math.atan((12.0) / 50.0)
    dist = (height * 0.5) / (FRAME_FILL * math.tan(v_ang * 0.5))
    loc = look + direction.normalized() * dist
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    world = bpy.data.worlds.get("Charcoal") or bpy.data.worlds.new("Charcoal")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.018, 0.018, 0.020, 1)
    bg.inputs["Strength"].default_value = 1.0
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    if not any(l.to_node == out for l in nt.links):
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = world

    right = cam.matrix_world.to_quaternion() @ Vector((1, 0, 0))
    view = (Vector(look) - cam.location).normalized()
    up = Vector((0, 0, 1))
    _area_light("Key", cam.location + right * 0.90 + up * 0.55, 2.5, KEY_ENERGY, look)
    _area_light("Fill", cam.location + view * 0.25 - right * 0.12 + up * 0.12,
                3.8, KEY_ENERGY * FILL_RATIO, look)
    _area_light("Rim", Vector(look) - view * 1.85 + up * 1.15 - right * 0.25,
                2.1, KEY_ENERGY * RIM_RATIO, look, color=(0.82, 0.90, 1.0))

    if "StudioFloor" not in bpy.data.objects:
        me = bpy.data.meshes.new("StudioFloor")
        me.from_pydata([(-8, -8, 0), (8, -8, 0), (8, 8, 0), (-8, 8, 0)], [], [(0, 1, 2, 3)])
        floor = bpy.data.objects.new("StudioFloor", me)
        bpy.context.scene.collection.objects.link(floor)
        fmat = bpy.data.materials.new("StudioFloorMat")
        fmat.use_nodes = True
        bsdf = fmat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.026, 0.026, 0.029, 1)
            bsdf.inputs["Roughness"].default_value = 0.95
        floor.data.materials.append(fmat)
        floor.is_shadow_catcher = True
        floor.visible_glossy = False
    return cam


def setup_cycles(res=(540, 960), samples=SAMPLES):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
    except Exception:
        pass
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.use_adaptive_sampling = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.film_transparent = False
    try:
        sc.view_settings.view_transform = "AgX"
    except Exception:
        pass
    sc.view_settings.exposure = EXPOSURE
    sc.render.fps = 30


def set_column(col):
    """col: 'xbot' | 'mpfb_plain' | 'mpfb_outfit'"""
    roles_x = {"xbot_mesh"}
    roles_body = {"body"}
    roles_out = {"glove", "beanie", "trunks", "tank"}
    for o in bpy.data.objects:
        role = o.get("mpfb_role")
        if not role:
            continue
        if col == "xbot":
            hide = role not in roles_x
        elif col == "mpfb_plain":
            hide = role not in roles_body
        else:
            hide = role not in roles_body | roles_out
        # Keep armatures but hidden
        if role in {"rig", "xbot_rig"}:
            hide = True
        o.hide_set(hide)
        o.hide_render = hide
        o.hide_viewport = hide


def render_still(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print("  wrote", path, flush=True)


def phase_render():
    enable_mpfb()
    if not BLEND.is_file():
        raise RuntimeError("missing blend; run earlier phases")
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    enable_mpfb()

    body = _find_by_role("body")
    xmeshes = [o for o in bpy.data.objects if o.get("mpfb_role") == "xbot_mesh"]
    # Shared camera: frame the MPFB body at jab extension, 72% fill.
    bpy.context.scene.frame_set(JAB["full"])
    bpy.context.view_layer.update()
    mn, mx = eval_bounds([body])
    look = Vector(((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.48))
    height = max(0.5, (mx.z - mn.z) * 1.05)
    setup_v2_camera(look, height)
    setup_cycles(res=(540, 960), samples=SAMPLES)

    cell_dir = OUT_DIR / "_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    rows = [("guard", JAB["guard"]), ("extension", JAB["full"]), ("return", JAB["ret"])]
    cols = [("xbot", "X Bot"), ("mpfb_plain", "MPFB plain"), ("mpfb_outfit", "MPFB outfit")]
    cells = {}
    for rname, frame in rows:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for ckey, _ in cols:
            set_column(ckey)
            dest = cell_dir / f"{rname}_{ckey}.png"
            render_still(dest)
            cells[f"{rname}_{ckey}"] = str(dest)

    # Close-ups: glove, beanie, trunks, open deltoid (outfit, extension).
    bpy.context.scene.frame_set(JAB["full"])
    set_column("mpfb_outfit")
    bpy.context.view_layer.update()
    close = {}
    targets = {
        "close_glove": "mixamorig:LeftHand",
        "close_beanie": "mixamorig:Head",
        "close_trunks": "mixamorig:Hips",
        "close_delt": "mixamorig:LeftArm",
    }
    close_fill = {"close_glove": 0.12, "close_beanie": 0.22,
                  "close_trunks": 0.40, "close_delt": 0.28}
    for name, bone in targets.items():
        mw = bone_matrix(_find_by_role("rig"), bone)
        if mw is None:
            continue
        look = mw.translation.copy()
        if name == "close_trunks":
            look.z += 0.05
        if name == "close_delt":
            look += Vector((0.04, 0.0, 0.06))
        setup_v2_camera(look, close_fill[name])
        setup_cycles(res=(540, 540), samples=SAMPLES)
        dest = cell_dir / f"{name}.png"
        render_still(dest)
        close[name] = str(dest)

    # Short jab preview (outfit only), 30 fps, frames guard-2 .. ret+4
    set_column("mpfb_outfit")
    # Reset a 3/4 body camera
    bpy.context.scene.frame_set(JAB["full"])
    bpy.context.view_layer.update()
    mn, mx = eval_bounds([body])
    look = Vector(((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.48))
    setup_v2_camera(look, max(0.5, (mx.z - mn.z) * 1.05))
    setup_cycles(res=(540, 960), samples=48)
    sc = bpy.context.scene
    f0 = max(sc.frame_start, JAB["guard"] - 2)
    f1 = min(sc.frame_end, JAB["ret"] + 4)
    sc.frame_start, sc.frame_end = f0, f1
    seq_dir = OUT_DIR / "_jab_preview"
    seq_dir.mkdir(parents=True, exist_ok=True)
    sc.render.filepath = str(seq_dir / "f")
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True)
    # stitch
    import subprocess
    mp4 = str(PREVIEW)
    cmd = [
        "ffmpeg", "-y", "-framerate", "30",
        "-start_number", str(f0),
        "-i", str(seq_dir / "f%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4,
    ]
    # Blender may write f0009.png etc.
    # Check actual pattern
    frames = sorted(seq_dir.glob("*.png"))
    print("preview frames", len(frames), "first", frames[0] if frames else None, flush=True)
    if frames:
        # detect pattern
        name = frames[0].name
        # try common blender patterns
        pat = str(seq_dir / "f%04d.png")
        if not Path(str(seq_dir / f"f{f0:04d}.png")).exists():
            # use a concat from the actual names via ffmpeg -pattern_type glob
            cmd = [
                "ffmpeg", "-y", "-framerate", "30", "-pattern_type", "glob",
                "-i", str(seq_dir / "*.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4,
            ]
        try:
            subprocess.check_call(cmd)
            print("WROTE", mp4, flush=True)
        except Exception as e:
            print("ffmpeg failed:", e, flush=True)

    compose_sheet(cells, close)
    facts = json.loads(FACTS.read_text()) if FACTS.is_file() else {}
    facts["sheet"] = str(SHEET)
    facts["preview"] = str(PREVIEW)
    facts["phase"] = "render"
    FACTS.write_text(json.dumps(facts, indent=2))
    print("SHEET", SHEET, flush=True)


def compose_sheet(cells, close):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL is not in Blender's Python; compose the sheet with system Python.", flush=True)
        return
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        fontt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        fonts = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = fontb = fontt = fonts = ImageFont.load_default()

    def load(p, size=None):
        if not p or not Path(p).is_file():
            im = Image.new("RGB", size or (540, 960), (30, 30, 32))
            return im
        im = Image.open(p).convert("RGB")
        if size and im.size != size:
            im = im.resize(size, Image.Resampling.LANCZOS)
        return im

    Wcell, Hcell = 540, 960
    pad, title_h, label_h, close_h = 16, 100, 28, 540
    cols = 3
    rows = 3
    W = pad + cols * (Wcell + pad)
    H = title_h + rows * (Hcell + label_h) + pad + close_h + 48
    im = Image.new("RGB", (W, H), (18, 18, 20))
    dr = ImageDraw.Draw(im)
    fg, muted = (236, 236, 238), (160, 162, 168)
    dr.text((pad, 16), "MPFB boxer v1   —   X Bot  |  MPFB plain  |  MPFB outfit",
            fill=fg, font=fontt)
    dr.text((pad, 52),
            "v2 lighting, alt04 left jab (guard / extension / return). "
            "Outfit: black gloves, charcoal beanie + tank, burgundy trunks (logo removed).",
            fill=muted, font=fonts)
    col_titles = ["X Bot (Mixamo)", "MPFB plain", "MPFB outfit"]
    row_names = ["guard", "extension", "return"]
    row_labels = ["GUARD", "JAB EXTENSION", "RETURN"]
    for r, rname in enumerate(row_names):
        for c, ckey in enumerate(("xbot", "mpfb_plain", "mpfb_outfit")):
            x = pad + c * (Wcell + pad)
            y = title_h + r * (Hcell + label_h)
            cell = load(cells.get(f"{rname}_{ckey}"), (Wcell, Hcell))
            im.paste(cell, (x, y))
            cap = f"{row_labels[r]}  ·  {col_titles[c]}"
            dr.text((x, y + Hcell + 4), cap, fill=fg, font=font)
    # close-ups
    cy = title_h + rows * (Hcell + label_h) + 8
    dr.text((pad, cy), "Close-ups (MPFB outfit, jab extension)", fill=fg, font=fontb)
    cy += 28
    cw = (W - pad * 5) // 4
    ch = close_h - 8
    for i, key in enumerate(("close_glove", "close_beanie", "close_trunks", "close_delt")):
        x = pad + i * (cw + pad)
        crop = load(close.get(key), (cw, ch))
        im.paste(crop, (x, cy))
        labels = {"close_glove": "glove", "close_beanie": "beanie",
                  "close_trunks": "trunks", "close_delt": "deltoid / arm"}
        dr.text((x, cy + ch + 2), labels[key], fill=fg, font=font)
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    im.save(SHEET, "PNG")
    print(f"composed {im.size} -> {SHEET}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["body", "retarget", "outfit", "qc", "render", "all",
                             "body_retarget_qc"])
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase in ("body", "all", "body_retarget_qc"):
        phase_body()
    if args.phase in ("retarget", "all", "body_retarget_qc"):
        phase_retarget()
    if args.phase in ("qc", "all", "body_retarget_qc"):
        phase_qc()
    if args.phase in ("outfit", "all"):
        phase_outfit()
    if args.phase in ("render", "all"):
        phase_render()
    print(f"DONE {args.phase} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
