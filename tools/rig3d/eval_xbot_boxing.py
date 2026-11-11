#!/usr/bin/env python3
"""Render Mixamo X Bot boxing clips as clay + bone-weight blob glow stills.

  nice -n 19 blender -b --python tools/rig3d/eval_xbot_boxing.py

Reads (gitignored, Mixamo licence — do not commit or redistribute raw):
  output/_basemodels/boxer/mixamo_xbot/{Lead Jab,Cross Punch,Hook,Uppercut,Fighting Idle}.fbx

Writes (also gitignored):
  output/_basemodels/eval/xbot_*.png
  output/_basemodels/eval/xbot_boxing_facts.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval_base_model import (  # noqa: E402
    _reset,
    _setup_world_lights,
    _world_dims,
)
import blob_v2  # noqa: E402
from eval_boxer_jab import bone_world, bone_tail_world  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
SRC_DIR = os.path.join(ROOT, "output/_basemodels/boxer/mixamo_xbot")
OUT_DIR = os.path.join(ROOT, "output/_basemodels/eval")

YAW = 35.0
ELEV = 10.0
RES = (540, 960)
SAMPLES = 16
EM_STR = 1.50
CANON_HEIGHT = 1.85
CANON_LOOK_Z = 0.96
ROOT_DRIFT_LOCK_M = 0.40

CLIPS = (
    {
        "key": "idle",
        "file": "Fighting Idle.fbx",
        "title": "Fighting Idle",
        "muscles": (),
        "pick": "idle",
    },
    {
        "key": "jab",
        "file": "Lead Jab.fbx",
        "title": "Lead Jab",
        "muscles": ("front_delt", "triceps"),
        "pick": "punch",
    },
    {
        "key": "cross",
        "file": "Cross Punch.fbx",
        "title": "Cross Punch",
        "muscles": ("chest", "front_delt", "obliques", "calf", "hip"),
        "pick": "punch",
    },
    {
        "key": "hook",
        "file": "Hook.fbx",
        "title": "Hook",
        "muscles": ("front_delt", "obliques"),
        "pick": "punch",
    },
    {
        "key": "uppercut",
        "file": "Uppercut.fbx",
        "title": "Uppercut",
        "muscles": ("biceps", "front_delt", "quads", "glutes"),
        "pick": "uppercut",
    },
)

COL = {
    "front_delt": np.array([0.55, 0.16, 1.00], np.float32),
    "triceps": np.array([1.00, 0.72, 0.04], np.float32),
    "biceps": np.array([1.00, 0.88, 0.18], np.float32),
    "chest": np.array([1.00, 0.22, 0.52], np.float32),
    "obliques": np.array([1.00, 0.34, 0.08], np.float32),
    "calf": np.array([0.12, 0.82, 0.88], np.float32),
    "hip": np.array([0.22, 0.95, 0.38], np.float32),
    "quads": np.array([0.48, 1.00, 0.16], np.float32),
    "glutes": np.array([1.00, 0.32, 0.58], np.float32),
}


def find_bone(arm, *cands):
    lower = {b.name.lower(): b.name for b in arm.pose.bones}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in cands:
        cl = c.lower().replace("mixamorig:", "").replace("mixamorig_", "")
        for ln, orig in lower.items():
            tail = ln.replace("mixamorig:", "").replace("mixamorig_", "")
            if tail == cl or tail.endswith(cl) or cl in tail:
                return orig
    return None


def bone_map(arm):
    m = {
        "hips": find_bone(arm, "mixamorig:Hips", "Hips", "hips", "pelvis"),
        "head": find_bone(arm, "mixamorig:Head", "Head", "head"),
        "neck": find_bone(arm, "mixamorig:Neck", "Neck"),
        "spine": find_bone(arm, "mixamorig:Spine", "Spine"),
        "spine1": find_bone(arm, "mixamorig:Spine1", "Spine1"),
        "spine2": find_bone(arm, "mixamorig:Spine2", "Spine2"),
        "hand_l": find_bone(arm, "mixamorig:LeftHand", "LeftHand", "hand_l"),
        "hand_r": find_bone(arm, "mixamorig:RightHand", "RightHand", "hand_r"),
        "arm_l": find_bone(arm, "mixamorig:LeftArm", "LeftArm", "upper_arm.L"),
        "arm_r": find_bone(arm, "mixamorig:RightArm", "RightArm", "upper_arm.R"),
        "fore_l": find_bone(arm, "mixamorig:LeftForeArm", "LeftForeArm"),
        "fore_r": find_bone(arm, "mixamorig:RightForeArm", "RightForeArm"),
        "sh_l": find_bone(arm, "mixamorig:LeftShoulder", "LeftShoulder"),
        "sh_r": find_bone(arm, "mixamorig:RightShoulder", "RightShoulder"),
        "upleg_l": find_bone(arm, "mixamorig:LeftUpLeg", "LeftUpLeg"),
        "upleg_r": find_bone(arm, "mixamorig:RightUpLeg", "RightUpLeg"),
        "leg_l": find_bone(arm, "mixamorig:LeftLeg", "LeftLeg"),
        "leg_r": find_bone(arm, "mixamorig:RightLeg", "RightLeg"),
        "foot_l": find_bone(arm, "mixamorig:LeftFoot", "LeftFoot", "foot_l"),
        "foot_r": find_bone(arm, "mixamorig:RightFoot", "RightFoot", "foot_r"),
    }
    print("BONES", {k: v for k, v in m.items()}, flush=True)
    return m


def frame_range(arm):
    f0, f1 = 1, 1
    ad = arm.animation_data
    if ad and ad.action:
        fr = ad.action.frame_range
        f0, f1 = int(round(fr[0])), int(round(fr[1]))
    sc = bpy.context.scene
    sc.frame_start = f0
    sc.frame_end = f1
    sc.render.fps = 30
    return f0, f1


def world_dims_eval(meshes):
    bpy.context.view_layer.update()
    xs, ys, zs = [], [], []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = o.matrix_world
        for v in me.vertices:
            p = mw @ v.co
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
        ev.to_mesh_clear()
    if not xs:
        return Vector((0, 0, 0)), Vector((0, 0, 0)), Vector((0, 0, 0))
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    return mn, mx, mx - mn


def scale_ground_center(arm, meshes, bones):
    bpy.context.view_layer.update()
    mn, mx, dim = world_dims_eval(meshes)
    height = dim.z if dim.z > 1e-4 else max(dim.x, dim.y, dim.z)
    note = f"height_before={height:.4f}"
    scale = 1.0
    if height > 50:
        scale = 0.01
        note += " treated as centimetres"
    elif height > 8 and height < 40:
        scale = 0.01
        note += " treated as centimetres"
    else:
        note += " treated as metres"
    if scale != 1.0:
        arm.scale *= scale
        bpy.context.view_layer.update()
    mn, mx, dim = world_dims_eval(meshes)
    arm.location.z -= mn.z
    bpy.context.view_layer.update()
    if bones.get("hips"):
        h = bone_world(arm, bones["hips"])
        if h is not None:
            arm.location.x -= h.x
            arm.location.y -= h.y
            bpy.context.view_layer.update()
    mn, mx, dim = world_dims_eval(meshes)
    return {"scale": scale, "note": note, "height_m": float(dim.z),
            "bbox_min": list(mn), "bbox_max": list(mx), "bbox_dim": list(dim)}


def face_minus_y(arm, bones):
    """Mixamo X Bot already faces -Y after Blender's FBX importer (object X=90).

    A posed boxing stance rotates the shoulders, so using (right-left) x up
    as 'forward' yaws the character 120–160 deg into a side or rear view.
    Leave the importer orientation alone.
    """
    print("FACE skip extra yaw (Mixamo FBX already -Y)", flush=True)
    return {"yaw_deg": 0.0, "how": "mixamo_importer_already_-Y"}


def hips_xy(arm, bones):
    h = bone_world(arm, bones["hips"])
    return Vector((h.x, h.y, 0.0)) if h else Vector((0, 0, 0))


def analyze_motion(arm, bones, f0, f1):
    rows = []
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        hl = bone_world(arm, bones["hand_l"])
        hr = bone_world(arm, bones["hand_r"])
        hd = bone_world(arm, bones["head"])
        hp = bone_world(arm, bones["hips"])
        fl = bone_world(arm, bones["foot_l"])
        fr = bone_world(arm, bones["foot_r"])
        rec = {"f": f}
        if hl and hd:
            rec["dist_l"] = float((hl - hd).length)
            rec["hand_l"] = list(hl)
        if hr and hd:
            rec["dist_r"] = float((hr - hd).length)
            rec["hand_r"] = list(hr)
        if hd:
            rec["head"] = list(hd)
            rec["head_z"] = float(hd.z)
        if hp:
            rec["hips"] = list(hp)
        if fl:
            rec["foot_l_z"] = float(fl.z)
        if fr:
            rec["foot_r_z"] = float(fr.z)
        rec["min_foot_z"] = min(
            [z for z in (rec.get("foot_l_z"), rec.get("foot_r_z")) if z is not None],
            default=None)
        rows.append(rec)
    return rows


def pick_punch_hand(rows):
    """The punching hand is the one that travels farthest horizontally
    from the hips (a dropping guard hand moves mainly in Z).
    """
    span = {"L": 0.0, "R": 0.0}
    for side, hkey in (("L", "hand_l"), ("R", "hand_r")):
        vals = []
        for r in rows:
            h, hp = r.get(hkey), r.get("hips")
            if h and hp:
                vals.append(math.hypot(h[0] - hp[0], h[1] - hp[1]))
        if vals:
            span[side] = max(vals) - min(vals)
    side = "R" if span["R"] >= span["L"] else "L"
    return side, span


def pick_keyframes(rows, side):
    key = "dist_l" if side == "L" else "dist_r"
    usable = [(r["f"], r[key], r) for r in rows if key in r]
    if not usable:
        f0, f1 = rows[0]["f"], rows[-1]["f"]
        return {"guard": f0, "mid": (f0 + f1) // 2, "full": f1}
    full_f, full_d, full_r = max(usable, key=lambda t: t[1])
    before = [t for t in usable if t[0] <= full_f]
    guard_f, guard_d, guard_r = min(before, key=lambda t: t[1])
    between = [t for t in usable if guard_f <= t[0] <= full_f]
    if not between:
        between = before or usable
    target = guard_d + 0.5 * (full_d - guard_d)
    mid_f, mid_d, mid_r = min(between, key=lambda t: abs(t[1] - target))
    return {
        "guard": int(guard_f), "mid": int(mid_f), "full": int(full_f),
        "guard_dist": float(guard_d), "mid_dist": float(mid_d),
        "full_dist": float(full_d),
        "guard_hand_z": float((guard_r.get("hand_l") if side == "L"
                               else guard_r.get("hand_r") or [0, 0, 0])[2]),
        "full_hand_z": float((full_r.get("hand_l") if side == "L"
                              else full_r.get("hand_r") or [0, 0, 0])[2]),
        "full_head_z": float(full_r.get("head_z") or 0.0),
    }


def pick_uppercut_keys(rows, side):
    """An uppercut's fist rises toward the chin, so hand-to-head distance
    *shrinks* at the finish. Use punching-hand height instead.
    """
    hkey = "hand_l" if side == "L" else "hand_r"
    dkey = "dist_l" if side == "L" else "dist_r"
    usable = []
    for r in rows:
        h = r.get(hkey)
        if h:
            usable.append((r["f"], float(h[2]), r))
    if not usable:
        f0, f1 = rows[0]["f"], rows[-1]["f"]
        return {"guard": f0, "mid": (f0 + f1) // 2, "full": f1}
    full_f, full_z, full_r = max(usable, key=lambda t: t[1])
    before = [t for t in usable if t[0] <= full_f]
    guard_f, guard_z, guard_r = min(before, key=lambda t: t[1])
    between = [t for t in usable if guard_f <= t[0] <= full_f] or before
    target = guard_z + 0.5 * (full_z - guard_z)
    mid_f, mid_z, mid_r = min(between, key=lambda t: abs(t[1] - target))
    return {
        "guard": int(guard_f), "mid": int(mid_f), "full": int(full_f),
        "guard_dist": float(guard_r.get(dkey) or 0.0),
        "mid_dist": float(mid_r.get(dkey) or 0.0),
        "full_dist": float(full_r.get(dkey) or 0.0),
        "guard_hand_z": float(guard_z),
        "mid_hand_z": float(mid_z),
        "full_hand_z": float(full_z),
        "full_head_z": float(full_r.get("head_z") or 0.0),
        "picker": "hand_z",
    }


def pick_idle_frame(rows):
    # tightest two-hand guard (both hands close to the head)
    best = None
    for r in rows:
        if "dist_l" in r and "dist_r" in r:
            score = r["dist_l"] + r["dist_r"]
            if best is None or score < best[0]:
                best = (score, r)
    if best is None:
        return {"guard": rows[len(rows) // 2]["f"]}
    r = best[1]
    return {
        "guard": int(r["f"]),
        "guard_dist_l": float(r["dist_l"]),
        "guard_dist_r": float(r["dist_r"]),
    }


def lock_root_xy(arm, bones, origin_xy):
    bpy.context.view_layer.update()
    now = hips_xy(arm, bones)
    arm.location.x += origin_xy.x - now.x
    arm.location.y += origin_xy.y - now.y
    bpy.context.view_layer.update()


def vg_mask(obj, pred):
    idxs = [i for i, vg in enumerate(obj.vertex_groups) if pred(vg.name.lower())]
    return blob_v2.vg_weight_array(obj, idxs)


def _is_side(n, side):
    n = n.lower()
    if side == "L":
        return ("left" in n) or n.endswith(".l") or n.endswith("_l")
    return ("right" in n) or n.endswith(".r") or n.endswith("_r")


def _clean(n):
    return (n.lower().replace("mixamorig:", "").replace("mixamorig_", "")
            .replace(":", "").replace(".", ""))


def paint_glow(meshes, arm, bones, punch_side, muscles):
    """Rig-following glow: bone-axis capsules gated by Mixamo vertex weights."""
    dummy_cam = Vector((0.0, -1.0, 0.2))
    hip, spine, side_ax, back, slen, info = blob_v2.estimate_anatomic_frame(
        meshes, dummy_cam)
    want_right = punch_side == "R"
    totals = {n: 0 for n in muscles}
    per = []

    def bhead(key):
        n = bones.get(key)
        return bone_world(arm, n) if n else None

    def btail(key):
        n = bones.get(key)
        return bone_tail_world(arm, n) if n else None

    ua = bhead("arm_r" if want_right else "arm_l")
    el = bhead("fore_r" if want_right else "fore_l") or btail("arm_r" if want_right else "arm_l")
    sh = bhead("sh_r" if want_right else "sh_l")
    upleg = bhead("upleg_r" if want_right else "upleg_l")
    knee = bhead("leg_r" if want_right else "leg_l") or btail("upleg_r" if want_right else "upleg_l")
    ankle = bhead("foot_r" if want_right else "foot_l") or (btail("leg_r" if want_right else "leg_l"))
    hips = bhead("hips")
    # opposite (rear/drive) leg — used for cross calf/hip and uppercut drive
    # For orthodox: punch R (cross) → rear is R; punch L (jab) → lead is L.
    # Cross "rear calf/hip" = punching side. Uppercut quads/glutes: both,
    # stronger on punching side.
    s = float(slen / 0.55) if slen else 1.0
    bk = Vector((float(back[0]), float(back[1]), float(back[2])))

    for o in meshes:
        if "joint" in o.name.lower():
            continue
        world, no = blob_v2._world_co_no(o)
        if not len(world):
            continue
        nb = no @ back
        front = blob_v2.smoothstep(0.05, -0.08, nb)
        rear = blob_v2.smoothstep(-0.05, 0.08, nb)
        nlow = o.name.lower()

        def side_w(side):
            return vg_mask(o, lambda n: _is_side(n, side))

        wr = side_w("R")
        wl = side_w("L")
        if wr.sum() + wl.sum() > 1:
            punch_w = (wr if want_right else wl) / np.maximum(wr + wl, 1e-6)
        else:
            punch_w = np.ones(len(world), np.float32)

        def w_keys(keys, side=None, exclude=()):
            def pred(n):
                c = _clean(n)
                if any(ex in c for ex in exclude):
                    return False
                if side and not _is_side(n, side):
                    return False
                return any(k in c for k in keys)
            return vg_mask(o, pred)

        sd = punch_side
        w_sh = w_keys(("shoulder", "clavicle", "collar"), sd)
        w_ua = w_keys(("upperarm", "upper_arm", "leftarm", "rightarm"), sd,
                      exclude=("fore", "hand", "shoulder"))
        # Mixamo LeftArm cleans to "leftarm" — already matched. ForeArm contains "arm"
        # but also "fore"; excluded.
        w_spine = w_keys(("spine", "spine1", "spine2", "chest"))
        w_hips = w_keys(("hips", "pelvis"))
        w_up = w_keys(("upleg", "thigh", "upperleg"), sd)
        w_shin = w_keys(("leftleg", "rightleg", "shin", "calf"), sd,
                        exclude=("upleg", "upper", "foot", "toe"))

        def cap(a, b, ra, rb):
            if a is None or b is None:
                return np.zeros(len(world), np.float32)
            return blob_v2.tapered_capsule_weight(
                world, a, b, ra, rb, ra * 0.72, rb * 0.72, back)

        out = {}
        if "front_delt" in muscles and ua is not None and sh is not None:
            a = sh + (ua - sh) * 0.15 - bk * (0.02 * s)
            b = ua + ((el - ua) if el is not None else (ua - sh)) * 0.18 - bk * (0.035 * s)
            out["front_delt"] = np.maximum(
                cap(a, b, 0.062 * s, 0.050 * s),
                0.88 * (0.60 * w_sh + 0.55 * w_ua) * front,
            ) * punch_w
        if "triceps" in muscles and ua is not None and el is not None:
            axis = el - ua
            a = ua + axis * 0.18 + bk * (0.032 * s)
            b = ua + axis * 0.82 + bk * (0.028 * s)
            out["triceps"] = np.maximum(
                cap(a, b, 0.050 * s, 0.042 * s),
                0.95 * w_ua * rear,
            ) * punch_w
        if "biceps" in muscles and ua is not None and el is not None:
            axis = el - ua
            a = ua + axis * 0.22 - bk * (0.028 * s)
            b = ua + axis * 0.78 - bk * (0.022 * s)
            out["biceps"] = np.maximum(
                cap(a, b, 0.046 * s, 0.038 * s),
                0.92 * w_ua * front,
            ) * punch_w
        if "chest" in muscles:
            # punching-side pec: medial-to-shoulder, front of upper spine
            pec_a = None
            pec_b = None
            if hips is not None and sh is not None:
                pec_a = hips + (sh - hips) * 0.62 - bk * (0.04 * s)
                pec_b = sh - (sh - hips) * 0.08 - bk * (0.05 * s)
            out["chest"] = np.maximum(
                cap(pec_a, pec_b, 0.070 * s, 0.055 * s) if pec_a is not None
                else np.zeros(len(world), np.float32),
                0.85 * w_spine * front * punch_w,
            ) * (0.55 + 0.45 * punch_w)
        if "obliques" in muscles and hips is not None and sh is not None:
            mid = hips + (sh - hips) * 0.42
            # offset to punching flank
            lat = (bhead("sh_r") - bhead("sh_l")) if (bhead("sh_r") and bhead("sh_l")) else None
            if lat is not None and lat.length > 1e-6:
                lat = lat.normalized() * (0.12 * s * (1.0 if want_right else -1.0))
                a = hips + (sh - hips) * 0.28 + lat - bk * (0.01 * s)
                b = hips + (sh - hips) * 0.52 + lat - bk * (0.015 * s)
            else:
                a = mid
                b = mid + Vector((0, 0, 0.12 * s))
            obl = np.maximum(
                cap(a, b, 0.058 * s, 0.048 * s),
                0.80 * (0.45 * w_spine + 0.25 * w_hips) * punch_w,
            )
            if hips is not None:
                dz = world[:, 2] - float(hips.z)
                waist = blob_v2.smoothstep(-0.02 * s, 0.06 * s, dz) * (
                    1.0 - blob_v2.smoothstep(0.26 * s, 0.36 * s, dz))
                obl = obl * waist
            out["obliques"] = obl
        if "calf" in muscles and knee is not None and ankle is not None:
            a = knee + (ankle - knee) * 0.15 + bk * (0.02 * s)
            b = knee + (ankle - knee) * 0.85 + bk * (0.018 * s)
            out["calf"] = np.maximum(
                cap(a, b, 0.046 * s, 0.038 * s),
                0.95 * w_shin * rear,
            ) * punch_w
        if "hip" in muscles and hips is not None and upleg is not None:
            a = hips + bk * (0.03 * s)
            b = upleg + (knee - upleg) * 0.15 + bk * (0.025 * s) if knee is not None else upleg
            out["hip"] = np.maximum(
                cap(a, b, 0.070 * s, 0.055 * s),
                0.85 * (0.50 * w_hips + 0.40 * w_up) * rear,
            ) * punch_w
        if "quads" in muscles and upleg is not None and knee is not None:
            a = upleg + (knee - upleg) * 0.12 - bk * (0.025 * s)
            b = upleg + (knee - upleg) * 0.88 - bk * (0.018 * s)
            # both thighs, stronger on punch/drive side
            w_up_both = w_keys(("upleg", "thigh", "upperleg"))
            out["quads"] = np.maximum(
                cap(a, b, 0.060 * s, 0.050 * s) * punch_w,
                0.70 * w_up_both * front + 0.35 * w_up * front,
            )
        if "glutes" in muscles and hips is not None and upleg is not None:
            a = hips + bk * (0.04 * s)
            b = upleg + bk * (0.03 * s)
            w_up_both = w_keys(("upleg", "thigh", "upperleg"))
            out["glutes"] = np.maximum(
                cap(a, b, 0.075 * s, 0.058 * s) * punch_w,
                0.75 * (0.55 * w_hips + 0.40 * w_up_both) * rear,
            )

        if not out:
            continue
        edges = blob_v2._edges_np(o.data)
        names = list(out)
        stack = np.stack([blob_v2._smooth_field(out[n], edges, 2) for n in names], 1)
        # write using names/colors
        n = len(world)
        acc = np.zeros((n, 3), np.float32)
        totw = np.zeros((n, 1), np.float32)
        for i, name in enumerate(names):
            w = np.clip(stack[:, i:i + 1], 0.0, 1.0)
            acc += w * COL[name]
            totw += w
            totals[name] = totals.get(name, 0) + int((stack[:, i] > 1e-4).sum())
        col = np.zeros((n, 4), np.float32)
        col[:, 3] = 1.0
        col[:, :3] = acc / np.maximum(totw, 1.0)
        va = o.data.color_attributes.get("mcol")
        if va is None:
            va = o.data.color_attributes.new("mcol", "FLOAT_COLOR", "POINT")
        va.data.foreach_set("color", col.ravel())
        o.data.update()
        per.append({"name": o.name, **{k: int((stack[:, i] > 1e-4).sum())
                                       for i, k in enumerate(names)}})
    return totals, per, info


def clear_glow(meshes):
    for o in meshes:
        n = len(o.data.vertices)
        if n == 0:
            continue
        col = np.zeros((n, 4), np.float32)
        col[:, 3] = 1.0
        va = o.data.color_attributes.get("mcol")
        if va is None:
            va = o.data.color_attributes.new("mcol", "FLOAT_COLOR", "POINT")
        va.data.foreach_set("color", col.ravel())
        o.data.update()


def assign_clay(meshes):
    mat = blob_v2.punch_material(em_str=EM_STR)
    # slightly lighter clay so glow pops
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED" and node.inputs["Base Color"].default_value[0] < 0.6:
            node.inputs["Base Color"].default_value = (0.46, 0.468, 0.478, 1.0)
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        if "joint" in o.name.lower():
            # joints stay clay — they get an empty mcol
            n = len(o.data.vertices)
            va = o.data.color_attributes.get("mcol")
            if va is None:
                va = o.data.color_attributes.new("mcol", "FLOAT_COLOR", "POINT")
            col = np.zeros((n, 4), np.float32)
            col[:, 3] = 1.0
            va.data.foreach_set("color", col.ravel())


def setup_fixed_camera():
    for o in list(bpy.data.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)
    look = Vector((0.0, 0.0, CANON_LOOK_Z))
    height = CANON_HEIGHT
    v_ang = 2 * math.atan(12.0 / 50.0)
    dist = (height * (1 + 2 * 0.08) / 2) / math.tan(v_ang / 2)
    dist *= 0.92
    yaw_r = math.radians(YAW)
    elev_r = math.radians(ELEV)
    direction = Vector((
        math.sin(yaw_r) * math.cos(elev_r),
        -math.cos(yaw_r) * math.cos(elev_r),
        math.sin(elev_r),
    ))
    loc = look + direction.normalized() * dist
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50
    cam_data.clip_start = 0.05
    cam_data.clip_end = 50
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    _setup_world_lights(cam)
    return cam


def render_png(path):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.filepath = path
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = -0.12
    sc.view_settings.gamma = 1.0
    sc.cycles.filter_width = 1.5
    bpy.ops.render.render(write_still=True)
    return path


def import_clip(path):
    _reset()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    bpy.context.view_layer.update()
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError(f"no armature in {path}")
    arm = arms[0]
    return meshes, arm


def process_clip(spec, origin_xy=None):
    path = os.path.join(SRC_DIR, spec["file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    meshes, arm = import_clip(path)
    bones = bone_map(arm)
    f0, f1 = frame_range(arm)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    scale_info = scale_ground_center(arm, meshes, bones)
    face_info = face_minus_y(arm, bones)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    start_xy = hips_xy(arm, bones)

    rows = analyze_motion(arm, bones, f0, f1)
    xs = [r["hips"][0] for r in rows if "hips" in r]
    ys = [r["hips"][1] for r in rows if "hips" in r]
    drift = 0.0
    if xs:
        drift = float(math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    lock = drift > ROOT_DRIFT_LOCK_M
    print(f"{spec['key']} frames {f0}-{f1} drift={drift:.3f}m lock={lock} "
          f"height={scale_info['height_m']:.3f}", flush=True)

    feet_min = [r["min_foot_z"] for r in rows if r.get("min_foot_z") is not None]
    both_high = 0
    for r in rows:
        fl, fr = r.get("foot_l_z"), r.get("foot_r_z")
        if fl is not None and fr is not None and fl > 0.12 and fr > 0.12:
            both_high += 1

    punch_side, span = pick_punch_hand(rows)
    if spec["pick"] == "idle":
        keys = pick_idle_frame(rows)
    elif spec["pick"] == "uppercut":
        keys = pick_uppercut_keys(rows, punch_side)
    else:
        keys = pick_keyframes(rows, punch_side)
    print(f"{spec['key']} punch_side={punch_side} span={span} keys={ {k: keys[k] for k in keys if k in ('guard','mid','full')} }",
          flush=True)

    assign_clay(meshes)
    setup_fixed_camera()

    stills = {}
    glow_counts = {}
    stages = ("guard",) if spec["pick"] == "idle" else ("guard", "mid", "full")
    for stage in stages:
        f = int(keys[stage])
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        if lock:
            lock_root_xy(arm, bones, start_xy)
        if spec["muscles"]:
            totals, per, _info = paint_glow(
                meshes, arm, bones, punch_side, spec["muscles"])
            glow_counts[stage] = totals
        else:
            clear_glow(meshes)
            glow_counts[stage] = {}
        out = os.path.join(OUT_DIR, f"xbot_{spec['key']}_{stage}.png")
        render_png(out)
        stills[stage] = out
        print(f"WROTE {out} ({os.path.getsize(out)} bytes)", flush=True)

    return {
        "key": spec["key"],
        "title": spec["title"],
        "file": spec["file"],
        "frames": [f0, f1],
        "scale": scale_info,
        "face": face_info,
        "root_drift_m": drift,
        "root_locked": lock,
        "min_foot_z": float(min(feet_min)) if feet_min else None,
        "max_min_foot_z": float(max(feet_min)) if feet_min else None,
        "frames_both_feet_ungrounded": both_high,
        "punch_side": punch_side,
        "hand_span": span,
        "keys": keys,
        "stills": stills,
        "glow": glow_counts,
        "muscles": list(spec["muscles"]),
        "n_mesh": len(meshes),
        "n_verts": sum(len(o.data.vertices) for o in meshes),
    }


def try_jab_preview(facts):
    """Low-res EEVEE loop of the jab; skip if it will take more than ~3 min."""
    jab = next((c for c in facts["clips"] if c["key"] == "jab"), None)
    if not jab:
        return {"skipped": True, "reason": "no jab clip facts"}
    t0 = time.time()
    path = os.path.join(SRC_DIR, "Lead Jab.fbx")
    meshes, arm = import_clip(path)
    bones = bone_map(arm)
    f0, f1 = frame_range(arm)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    scale_ground_center(arm, meshes, bones)
    face_minus_y(arm, bones)
    rows = analyze_motion(arm, bones, f0, f1)
    side, _span = pick_punch_hand(rows)
    keys = pick_keyframes(rows, side)
    start_xy = hips_xy(arm, bones)
    xs = [r["hips"][0] for r in rows if "hips" in r]
    ys = [r["hips"][1] for r in rows if "hips" in r]
    drift = float(math.hypot(max(xs) - min(xs), max(ys) - min(ys))) if xs else 0.0
    lock = drift > ROOT_DRIFT_LOCK_M
    assign_clay(meshes)
    setup_fixed_camera()

    n = f1 - f0 + 1
    # target ~2 s at 15 fps = 30 frames. Prefer the full clip if short.
    fps = 15
    if n <= 40:
        frames = list(range(f0, f1 + 1))
        fps = max(12, min(30, int(round(n / 2.0)))) or 15
    else:
        # guard → a bit past full, then hold last
        a, b = keys["guard"], min(f1, keys["full"] + max(2, (keys["full"] - keys["guard"]) // 6))
        raw = list(range(a, b + 1))
        step = max(1, int(math.ceil(len(raw) / 30)))
        frames = raw[::step]

    # Estimate: EEVEE 360x640 ~0.4–1.0s/frame on this box. Bail if > 40 frames
    # would clearly exceed 3 min, or after the first few frames if pace is slow.
    if len(frames) > 50:
        return {"skipped": True, "reason": f"too many frames ({len(frames)})"}

    seq_dir = os.path.join(OUT_DIR, "_xbot_jab_seq")
    os.makedirs(seq_dir, exist_ok=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.render.resolution_x, sc.render.resolution_y = 360, 640
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.exposure = -0.12
    if hasattr(sc, "eevee"):
        try:
            sc.eevee.taa_render_samples = 8
        except Exception:
            pass

    written = []
    for i, f in enumerate(frames):
        if time.time() - t0 > 150 and i > 4:
            return {"skipped": True, "reason": "pacing would exceed 3 min",
                    "elapsed_s": time.time() - t0, "done": i}
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        if lock:
            lock_root_xy(arm, bones, start_xy)
        paint_glow(meshes, arm, bones, side, ("front_delt", "triceps"))
        fp = os.path.join(seq_dir, f"f_{i:04d}.png")
        sc.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        written.append(fp)
    if not written:
        return {"skipped": True, "reason": "no frames"}

    mp4 = os.path.join(OUT_DIR, "xbot_jab_preview.mp4")
    # Loop to ~2s if the clip is short: duplicate the sequence in the concat.
    import subprocess
    pattern = os.path.join(seq_dir, "f_%04d.png")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-i", pattern,
        "-vf", "loop=loop=1:size=" + str(len(written)) + ":start=0",
        "-frames:v", str(max(len(written), int(2 * fps))),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        mp4,
    ]
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        # simpler encode without loop filter
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps), "-i", pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4,
        ]
        try:
            subprocess.check_call(cmd)
        except Exception as e2:
            return {"skipped": True, "reason": f"ffmpeg failed: {e2}",
                    "elapsed_s": time.time() - t0}
    return {
        "skipped": False,
        "path": mp4,
        "bytes": os.path.getsize(mp4) if os.path.isfile(mp4) else 0,
        "n_src": len(written),
        "fps": fps,
        "elapsed_s": time.time() - t0,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    facts = {
        "ok": False,
        "yaw": YAW, "elev": ELEV, "samples": SAMPLES, "res": list(RES),
        "em_str": EM_STR,
        "src_dir": SRC_DIR,
        "clips": [],
    }
    t0 = time.time()
    for spec in CLIPS:
        rec = process_clip(spec)
        facts["clips"].append(rec)
    facts["stills_elapsed_s"] = time.time() - t0
    # video only if stills left budget
    remaining = 180.0 - (time.time() - t0)
    if remaining > 60:
        facts["jab_preview"] = try_jab_preview(facts)
    else:
        facts["jab_preview"] = {
            "skipped": True,
            "reason": f"stills used {facts['stills_elapsed_s']:.0f}s; "
                      f"no 3-min budget left for video",
        }
    facts["elapsed_s"] = time.time() - t0
    facts["ok"] = all(
        os.path.isfile(p)
        for c in facts["clips"]
        for p in c.get("stills", {}).values()
    )
    out_json = os.path.join(OUT_DIR, "xbot_boxing_facts.json")
    with open(out_json, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("FACTS", out_json, "ok", facts["ok"], flush=True)
    if not facts["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
