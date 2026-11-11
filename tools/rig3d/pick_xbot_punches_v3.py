#!/usr/bin/env python3
"""Pick the most textbook Mixamo X Bot jab / cross / lead hook / uppercut.

  nice -n 19 blender -b --python tools/rig3d/pick_xbot_punches_v3.py -- --phase analyze
  nice -n 19 blender -b --python tools/rig3d/pick_xbot_punches_v3.py -- --phase render

Reads gitignored Mixamo FBX (do not commit). Writes gitignored eval stills + JSON.
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

import eval_xbot_boxing as xb  # noqa: E402
from eval_boxer_jab import bone_world  # noqa: E402

SRC = xb.SRC_DIR
OUT = xb.OUT_DIR
THUMB_DIR = os.path.join(OUT, "xbot_picks_thumbs")
FACTS_PATH = os.path.join(OUT, "xbot_punch_pick_facts_v3.json")
MARGIN = 0.05
MUSCLES = {
    "jab": ("front_delt", "triceps"),
    "cross": ("chest", "front_delt", "obliques", "calf", "hip"),
    "hook": ("front_delt", "obliques"),
    "uppercut": ("biceps", "front_delt", "quads", "glutes"),
}

CATALOG = [
    {"id": "orig_jab", "file": "Lead Jab.fbx", "hint": "jab",
     "title": "Lead Jab (current)"},
    {"id": "orig_cross", "file": "Cross Punch.fbx", "hint": "cross",
     "title": "Cross Punch (current)"},
    {"id": "orig_hook", "file": "Hook.fbx", "hint": "hook",
     "title": "Hook (current)"},
    {"id": "orig_upper", "file": "Uppercut.fbx", "hint": "uppercut",
     "title": "Uppercut (current)"},
    {"id": "alt01", "file": "alts/alt01_Lead Jab_MedHeadJab.fbx", "hint": "jab",
     "title": "alt01 MedHeadJab"},
    {"id": "alt02", "file": "alts/alt02_Lead Jab_ShortBodyJab.fbx", "hint": "jab",
     "title": "alt02 ShortBodyJab"},
    {"id": "alt03", "file": "alts/alt03_Lead Jab_ShortHeadJab.fbx", "hint": "jab",
     "title": "alt03 ShortHeadJab"},
    {"id": "alt04", "file": "alts/alt04_Jab Cross_BoxingJabCrossMedium.fbx",
     "hint": "combo", "title": "alt04 JabCross Medium"},
    {"id": "alt05", "file": "alts/alt05_Body Jab Cross_BoxingBodyJabCrossMedium.fbx",
     "hint": "combo", "title": "alt05 BodyJabCross"},
    {"id": "alt06", "file": "alts/alt06_Jab Cross_ShortJabCross.fbx",
     "hint": "combo", "title": "alt06 ShortJabCross"},
    {"id": "alt07", "file": "alts/alt07_Hook_MidHookToHead.fbx", "hint": "hook",
     "title": "alt07 MidHookToHead"},
    {"id": "alt08", "file": "alts/alt08_Hook_LongHookToHead.fbx", "hint": "hook",
     "title": "alt08 LongHookToHead"},
    {"id": "alt09", "file": "alts/alt09_Boxing_LeadHandHook.fbx", "hint": "hook",
     "title": "alt09 LeadHandHook"},
    {"id": "alt10", "file": "alts/alt10_Boxing_BackHandHook.fbx", "hint": "hook",
     "title": "alt10 BackHandHook"},
    {"id": "alt11", "file": "alts/alt11_Boxing_LeadingHandJab.fbx", "hint": "jab",
     "title": "alt11 LeadingHandJab"},
    {"id": "alt12", "file": "alts/alt12_Boxing_JabCrossCombo.fbx", "hint": "combo",
     "title": "alt12 JabCrossCombo"},
    {"id": "alt13", "file": "alts/alt13_Boxing_BackHandUppercut.fbx",
     "hint": "uppercut", "title": "alt13 BackHandUppercut"},
    {"id": "alt14", "file": "alts/alt14_Boxing_LeadHandUppercut.fbx",
     "hint": "uppercut", "title": "alt14 LeadHandUppercut"},
    {"id": "alt15", "file": "alts/alt15_Boxing_BackHandCross.fbx", "hint": "cross",
     "title": "alt15 BackHandCross"},
    {"id": "alt16", "file": "alts/alt16_Punch Combo_FourPunchCombo.fbx",
     "hint": "combo", "title": "alt16 FourPunchCombo"},
    {"id": "alt17", "file": "alts/alt17_Combo Punch_EightPunchCombo.fbx",
     "hint": "combo", "title": "alt17 EightPunchCombo"},
    {"id": "alt18", "file": "alts/alt18_Punching_CrossPunch.fbx", "hint": "cross",
     "title": "alt18 CrossPunch"},
]


def _cli():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    phase = "analyze"
    winners = {}
    i = 0
    while i < len(argv):
        if argv[i] == "--phase" and i + 1 < len(argv):
            phase = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--winners" and i + 1 < len(argv):
            for part in argv[i + 1].split(","):
                if ":" in part:
                    k, v = part.split(":", 1)
                    winners[k.strip()] = v.strip()
            i += 2
            continue
        i += 1
    return phase, winners


def gauss(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lo_better(x, good, bad):
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return 1.0 - (x - good) / (bad - good)


def hi_better(x, good, bad):
    if x >= good:
        return 1.0
    if x <= bad:
        return 0.0
    return (x - bad) / (good - bad)


def angle_at(a, pivot, b):
    u = (a - pivot).normalized()
    v = (b - pivot).normalized()
    return math.degrees(math.acos(max(-1.0, min(1.0, u.dot(v)))))


def bone_axis(arm, name, axis=(0, 1, 0)):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    mw = arm.matrix_world @ pb.matrix
    return (mw.to_3x3() @ Vector(axis)).normalized()


def bone_yaw_deg(arm, name):
    ax = bone_axis(arm, name)
    if ax is None:
        return None
    return math.degrees(math.atan2(ax.y, ax.x))


def enrich_bones(arm, bones):
    bones = dict(bones)
    bones["toe_l"] = xb.find_bone(arm, "mixamorig:LeftToeBase", "LeftToeBase")
    bones["toe_r"] = xb.find_bone(arm, "mixamorig:RightToeBase", "RightToeBase")
    return bones


def analyze_rich(arm, bones, f0, f1):
    rows = []
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        rec = {"f": f}
        for key in ("hand_l", "hand_r", "head", "hips", "foot_l", "foot_r",
                    "arm_l", "arm_r", "fore_l", "fore_r", "sh_l", "sh_r",
                    "upleg_l", "upleg_r", "neck"):
            n = bones.get(key)
            p = bone_world(arm, n) if n else None
            if p is not None:
                rec[key] = [float(p.x), float(p.y), float(p.z)]
        hd, hl, hr, hp = rec.get("head"), rec.get("hand_l"), rec.get("hand_r"), rec.get("hips")
        if hl and hd:
            rec["dist_l"] = float(Vector(hl).length if False else (Vector(hl) - Vector(hd)).length)
        if hr and hd:
            rec["dist_r"] = float((Vector(hr) - Vector(hd)).length)
        if hl and hp:
            rec["reach_l"] = math.hypot(hl[0] - hp[0], hl[1] - hp[1])
        if hr and hp:
            rec["reach_r"] = math.hypot(hr[0] - hp[0], hr[1] - hp[1])
        for side, shk, elk, wrk in (
            ("l", "arm_l", "fore_l", "hand_l"),
            ("r", "arm_r", "fore_r", "hand_r"),
        ):
            sh, el, wr = rec.get(shk), rec.get(elk), rec.get(wrk)
            if sh and el and wr:
                rec[f"elbow_{side}"] = angle_at(Vector(sh), Vector(el), Vector(wr))
                ua = Vector(el) - Vector(sh)
                if ua.length > 1e-6:
                    rec[f"arm_elev_{side}"] = math.degrees(math.asin(max(-1.0, min(1.0, ua.normalized().z))))
        shl, shr = rec.get("sh_l") or rec.get("arm_l"), rec.get("sh_r") or rec.get("arm_r")
        if shl and shr:
            rec["sh_yaw"] = math.degrees(math.atan2(shr[1] - shl[1], shr[0] - shl[0]))
        ul, ur = rec.get("upleg_l"), rec.get("upleg_r")
        if ul and ur:
            rec["hip_yaw"] = math.degrees(math.atan2(ur[1] - ul[1], ur[0] - ul[0]))
        hn = bones.get("head")
        if hn:
            along = bone_axis(arm, hn)
            if along is not None:
                rec["head_up_z"] = float(along.z)
                rec["head_axis"] = [float(along.x), float(along.y), float(along.z)]
                rec["head_tilt_fwd"] = float(along.dot(Vector((0.0, -1.0, 0.0))))
        for side, fk in (("l", "foot_l"), ("r", "foot_r")):
            fn = bones.get(fk)
            if fn:
                rec[f"foot_{side}_yaw"] = bone_yaw_deg(arm, fn)
            if rec.get(fk):
                rec[f"foot_{side}_z"] = rec[fk][2]
        rec["min_foot_z"] = min(
            [z for z in (rec.get("foot_l_z"), rec.get("foot_r_z")) if z is not None],
            default=None)
        rows.append(rec)
    return rows


def detect_lead(rows):
    """Facing -Y: more-forward hand/foot has smaller Y. Sample early frames."""
    sample = rows[: max(3, min(12, len(rows) // 5))] or rows
    ly, ry, lfy, rfy = [], [], [], []
    for r in sample:
        if r.get("hand_l"):
            ly.append(r["hand_l"][1])
        if r.get("hand_r"):
            ry.append(r["hand_r"][1])
        if r.get("foot_l"):
            lfy.append(r["foot_l"][1])
        if r.get("foot_r"):
            rfy.append(r["foot_r"][1])
    votes = 0
    if ly and ry:
        votes += -1 if (sum(ly) / len(ly)) < (sum(ry) / len(ry)) else 1
    if lfy and rfy:
        votes += -1 if (sum(lfy) / len(lfy)) < (sum(rfy) / len(rfy)) else 1
    lead = "L" if votes <= 0 else "R"
    return lead, votes


def _local_peaks(values, min_dist=6, min_rise=0.10, win=14):
    n = len(values)
    peaks = []
    i = 1
    while i < n - 1:
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            left = values[max(0, i - win): i]
            if left:
                rise = values[i] - min(left)
                if rise >= min_rise:
                    if not peaks or i - peaks[-1] >= min_dist:
                        peaks.append(i)
                    elif rise > values[peaks[-1]] - min(values[max(0, peaks[-1] - win): peaks[-1]] or [values[peaks[-1]]]):
                        peaks[-1] = i
                    i += min_dist
                    continue
        i += 1
    return peaks


def _guard_before(values, peak_i, win=22):
    lo = max(0, peak_i - win)
    window = values[lo: peak_i + 1]
    if not window:
        return peak_i, values[peak_i]
    j = int(np.argmin(window))
    return lo + j, window[j]


def _return_after(values, peak_i, win=22):
    hi = min(len(values) - 1, peak_i + win)
    window = values[peak_i: hi + 1]
    if not window:
        return peak_i, values[peak_i]
    j = int(np.argmin(window))
    return peak_i + j, window[j]


def unwrap_delta(a, b):
    d = b - a
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def span_xy(rows, key_xyz, f0, f1):
    pts = []
    for r in rows:
        if f0 <= r["f"] <= f1 and r.get(key_xyz):
            pts.append((r[key_xyz][0], r[key_xyz][1]))
    if not pts:
        return 0.0
    xs, ys = zip(*pts)
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def yaw_span(rows, key, f0, f1):
    ys = [r[key] for r in rows if f0 <= r["f"] <= f1 and r.get(key) is not None]
    if len(ys) < 2:
        return 0.0
    # unwrap relative to first
    acc = [0.0]
    for i in range(1, len(ys)):
        acc.append(acc[-1] + unwrap_delta(ys[i - 1], ys[i]))
    return max(acc) - min(acc)


def measure_punch(rows, side, full_i, ptype, lead):
    hkey = "hand_l" if side == "L" else "hand_r"
    rkey = "reach_l" if side == "L" else "reach_r"
    ekey = "elbow_l" if side == "L" else "elbow_r"
    akey = "arm_elev_l" if side == "L" else "arm_elev_r"
    dkey = "dist_l" if side == "L" else "dist_r"
    opp_d = "dist_r" if side == "L" else "dist_l"
    opp_h = "hand_r" if side == "L" else "hand_l"
    zseries = [((r.get(hkey) or [0, 0, 0])[2]) for r in rows]
    rseries = [r.get(rkey) or 0.0 for r in rows]

    if ptype == "uppercut":
        g_i, _ = _guard_before(zseries, full_i)
        ret_i, _ = _return_after(zseries, full_i)
    else:
        g_i, _ = _guard_before(rseries, full_i)
        ret_i, _ = _return_after(rseries, full_i)
    if g_i >= full_i:
        g_i = max(0, full_i - 4)
    mid_i = (g_i + full_i) // 2
    g, mid, full, ret = rows[g_i], rows[mid_i], rows[full_i], rows[ret_i]
    f0, f1 = g["f"], ret["f"]

    fh = full.get(hkey)
    hd = full.get("head")
    fist_vs_head = (fh[2] - hd[2]) if (fh and hd) else 0.0
    elbow = float(full.get(ekey) or 0.0)
    arm_elev = float(full.get(akey) or 0.0)
    rear_or_opp = float(full.get(opp_d) or 0.0)
    punch_dist = float(full.get(dkey) or 0.0)
    reach_full = float(full.get(rkey) or 0.0)
    reach_g = float(g.get(rkey) or 0.0)
    hz_full = fh[2] if fh else 0.0
    hz_g = (g.get(hkey) or [0, 0, 0])[2]
    rise_z = hz_full - hz_g
    rise_xy = reach_full - reach_g

    hip_rot = yaw_span(rows, "hip_yaw", g["f"], full["f"])
    sh_rot = yaw_span(rows, "sh_yaw", g["f"], full["f"])
    head_tilt = float(full.get("head_tilt_fwd") or 0.0)
    root_drift = span_xy(rows, "hips", f0, f1)
    slide_l = span_xy(rows, "foot_l", f0, f1)
    slide_r = span_xy(rows, "foot_r", f0, f1)
    yaw_l = yaw_span(rows, "foot_l_yaw", g["f"], full["f"])
    yaw_r = yaw_span(rows, "foot_r_yaw", g["f"], full["f"])
    rear_side = "R" if lead == "L" else "L"
    rear_slide = slide_r if rear_side == "R" else slide_l
    rear_yaw = yaw_r if rear_side == "R" else yaw_l
    lead_slide = slide_l if lead == "L" else slide_r
    feet_z = [r.get("min_foot_z") for r in rows if f0 <= r["f"] <= f1 and r.get("min_foot_z") is not None]
    both_high = 0
    for r in rows:
        if not (f0 <= r["f"] <= f1):
            continue
        fl, fr = r.get("foot_l_z"), r.get("foot_r_z")
        if fl is not None and fr is not None and fl > 0.16 and fr > 0.16:
            both_high += 1

    # return-to-guard: both hands come nearer the head after the peak
    ret_punch = float(ret.get(dkey) or 99)
    ret_opp = float(ret.get(opp_d) or 99)
    ret_score = 0.5 * lo_better(ret_punch, 0.32, 0.70) + 0.5 * lo_better(ret_opp, 0.32, 0.65)

    target = "body" if fist_vs_head < -0.28 else "head"
    return {
        "side": side,
        "is_lead_hand": side == lead,
        "ptype": ptype,
        "target": target,
        "guard": int(g["f"]),
        "mid": int(mid["f"]),
        "full": int(full["f"]),
        "ret": int(ret["f"]),
        "elbow_deg": elbow,
        "arm_elev_deg": arm_elev,
        "fist_vs_head_m": fist_vs_head,
        "fist_z_m": hz_full,
        "head_z_m": hd[2] if hd else None,
        "opp_hand_to_head_m": rear_or_opp,
        "punch_hand_to_head_m": punch_dist,
        "reach_full_m": reach_full,
        "rise_xy_m": rise_xy,
        "rise_z_m": rise_z,
        "hip_rot_deg": hip_rot,
        "sh_rot_deg": sh_rot,
        "head_tilt_fwd": head_tilt,
        "root_drift_m": root_drift,
        "foot_slide_l_m": slide_l,
        "foot_slide_r_m": slide_r,
        "rear_foot_slide_m": rear_slide,
        "lead_foot_slide_m": lead_slide,
        "rear_foot_yaw_deg": rear_yaw,
        "min_foot_z": min(feet_z) if feet_z else None,
        "max_min_foot_z": max(feet_z) if feet_z else None,
        "frames_both_feet_high": both_high,
        "return_score": ret_score,
        "return_punch_dist_m": ret_punch,
        "return_opp_dist_m": ret_opp,
    }


def classify(side, lead, elbow, rise_xy, rise_z, fist_vs_head):
    if rise_z >= 0.14 and rise_z >= rise_xy * 0.75 and elbow < 155:
        return "uppercut"
    if 60.0 <= elbow <= 125.0 and fist_vs_head > -0.28:
        return "hook"
    if elbow >= 132.0:
        return "jab" if side == lead else "cross"
    if rise_z >= 0.12 and elbow < 160:
        return "uppercut"
    if elbow < 130:
        return "hook"
    return "jab" if side == lead else "cross"


def extract_punches(rows, lead):
    punches = []
    n = len(rows)
    if n < 4:
        return punches
    used = set()  # (side, full_index) roughly
    for side, rkey, zget in (
        ("L", "reach_l", lambda r: (r.get("hand_l") or [0, 0, 0])[2]),
        ("R", "reach_r", lambda r: (r.get("hand_r") or [0, 0, 0])[2]),
    ):
        rseries = [r.get(rkey) or 0.0 for r in rows]
        zseries = [zget(r) for r in rows]
        ekey = "elbow_l" if side == "L" else "elbow_r"
        cand = []
        for i in _local_peaks(rseries, min_dist=6, min_rise=0.09, win=14):
            cand.append(("reach", i))
        for i in _local_peaks(zseries, min_dist=6, min_rise=0.10, win=14):
            cand.append(("z", i))
        # merge nearby
        cand.sort(key=lambda t: t[1])
        merged = []
        for kind, i in cand:
            if merged and abs(i - merged[-1][1]) <= 4:
                # keep the one with clearer signal later; mark both kinds
                prev = merged[-1]
                merged[-1] = (prev[0] + "+" + kind, prev[1] if rseries[prev[1]] >= rseries[i] else i)
            else:
                merged.append((kind, i))
        for kind, i in merged:
            key = (side, i)
            if key in used:
                continue
            used.add(key)
            elbow = float(rows[i].get(ekey) or 0.0)
            hd = rows[i].get("head")
            hh = rows[i].get("hand_l" if side == "L" else "hand_r")
            fvh = (hh[2] - hd[2]) if (hh and hd) else 0.0
            g_r, _ = _guard_before(rseries, i)
            g_z, _ = _guard_before(zseries, i)
            rise_xy = rseries[i] - rseries[g_r]
            rise_z = zseries[i] - zseries[g_z]
            ptype = classify(side, lead, elbow, rise_xy, rise_z, fvh)
            m = measure_punch(rows, side, i, ptype, lead)
            m["peak_kind"] = kind
            punches.append(m)
    punches.sort(key=lambda p: p["full"])
    return punches


def score_punch(p, want):
    parts = {}
    if want == "jab":
        if not p["is_lead_hand"]:
            return 0.0, {"reject": "not lead hand"}
        parts["elbow"] = 22 * gauss(p["elbow_deg"], 170, 18)
        parts["fist_h"] = 18 * gauss(p["fist_vs_head_m"], -0.06, 0.12)
        parts["opp_guard"] = 14 * lo_better(p["opp_hand_to_head_m"], 0.28, 0.58)
        parts["chin"] = 8 * hi_better(p["head_tilt_fwd"], 0.08, -0.12)
        parts["return"] = 10 * p["return_score"]
        parts["drift"] = 8 * lo_better(p["root_drift_m"], 0.12, 0.45)
        parts["slide"] = 8 * lo_better(max(p["foot_slide_l_m"], p["foot_slide_r_m"]), 0.07, 0.22)
        parts["reach"] = 7 * hi_better(p["rise_xy_m"], 0.28, 0.10)
        parts["hip"] = 5 * gauss(p["hip_rot_deg"], 18, 16)
        if p["target"] == "body":
            parts["body_pen"] = -18
    elif want == "cross":
        if p["is_lead_hand"]:
            return 0.0, {"reject": "lead hand (not rear)"}
        parts["elbow"] = 18 * gauss(p["elbow_deg"], 170, 18)
        parts["fist_h"] = 16 * gauss(p["fist_vs_head_m"], -0.06, 0.12)
        parts["opp_guard"] = 12 * lo_better(p["opp_hand_to_head_m"], 0.30, 0.62)
        parts["hip"] = 14 * hi_better(p["hip_rot_deg"], 28, 8)
        parts["shoulder"] = 8 * hi_better(p["sh_rot_deg"], 30, 8)
        # rear foot should pivot (yaw) more than it slides
        pivot = hi_better(p["rear_foot_yaw_deg"], 25, 5) * lo_better(p["rear_foot_slide_m"], 0.10, 0.28)
        parts["pivot"] = 10 * pivot
        parts["return"] = 8 * p["return_score"]
        parts["drift"] = 6 * lo_better(p["root_drift_m"], 0.18, 0.55)
        parts["chin"] = 8 * hi_better(p["head_tilt_fwd"], 0.08, -0.12)
        if p["target"] == "body":
            parts["body_pen"] = -16
    elif want == "hook":
        if not p["is_lead_hand"]:
            return 0.0, {"reject": "rear-hand hook (want lead hook)"}
        parts["elbow"] = 22 * gauss(p["elbow_deg"], 90, 16)
        parts["horizontal"] = 14 * gauss(p["arm_elev_deg"], 5, 22)
        parts["fist_h"] = 16 * gauss(p["fist_vs_head_m"], -0.05, 0.12)
        parts["opp_guard"] = 12 * lo_better(p["opp_hand_to_head_m"], 0.30, 0.60)
        parts["hip"] = 10 * hi_better(p["hip_rot_deg"], 22, 6)
        parts["return"] = 8 * p["return_score"]
        parts["slide"] = 6 * lo_better(max(p["foot_slide_l_m"], p["foot_slide_r_m"]), 0.08, 0.24)
        parts["chin"] = 8 * hi_better(p["head_tilt_fwd"], 0.08, -0.12)
        parts["drift"] = 4 * lo_better(p["root_drift_m"], 0.15, 0.50)
    elif want == "uppercut":
        parts["finish"] = 24 * gauss(p["fist_vs_head_m"], -0.07, 0.11)
        parts["short"] = 12 * lo_better(max(0.0, -p["rise_z_m"] + p["fist_vs_head_m"] + 0.3), 0.22, 0.55)
        # prefer a compact rise, not a huge looping lift
        parts["rise"] = 8 * gauss(p["rise_z_m"], 0.28, 0.16)
        parts["elbow"] = 12 * gauss(p["elbow_deg"], 125, 28)
        parts["opp_guard"] = 12 * lo_better(p["opp_hand_to_head_m"], 0.32, 0.65)
        parts["return"] = 10 * p["return_score"]
        parts["hip"] = 8 * hi_better(p["hip_rot_deg"], 18, 5)
        parts["slide"] = 6 * lo_better(max(p["foot_slide_l_m"], p["foot_slide_r_m"]), 0.08, 0.24)
        parts["chin"] = 8 * hi_better(p["head_tilt_fwd"], 0.08, -0.12)
        if p["fist_vs_head_m"] > 0.18:
            parts["overshoot"] = -14
    else:
        return 0.0, {"reject": "unknown type"}
    total = float(sum(parts.values()))
    return total, parts


def setup_fitted_camera(meshes, frames, arm, bones, start_xy, lock):
    for o in list(bpy.data.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)
    xs, ys, zs = [], [], []
    for f in frames:
        bpy.context.scene.frame_set(int(f))
        bpy.context.view_layer.update()
        if lock:
            xb.lock_root_xy(arm, bones, start_xy)
        mn, mx, _ = xb.world_dims_eval(meshes)
        xs += [mn.x, mx.x]
        ys += [mn.y, mx.y]
        zs += [mn.z, mx.z]
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    look = (mn + mx) * 0.5
    yaw_r = math.radians(xb.YAW)
    elev_r = math.radians(xb.ELEV)
    direction = Vector((
        math.sin(yaw_r) * math.cos(elev_r),
        -math.cos(yaw_r) * math.cos(elev_r),
        math.sin(elev_r),
    )).normalized()
    view = -direction
    world_up = Vector((0, 0, 1))
    right = view.cross(world_up)
    if right.length < 1e-6:
        right = Vector((1, 0, 0))
    right.normalize()
    cam_up = right.cross(view).normalized()
    corners = [Vector((x, y, z))
               for x in (mn.x, mx.x)
               for y in (mn.y, mx.y)
               for z in (mn.z, mx.z)]
    half_w = max(abs((c - look).dot(right)) for c in corners)
    half_h = max(abs((c - look).dot(cam_up)) for c in corners)
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = 24.0
    cam_data.clip_start = 0.05
    cam_data.clip_end = 50
    sensor_w = 24.0 * (xb.RES[0] / xb.RES[1])
    v_half = math.atan((24.0 * 0.5) / 50.0)
    h_half = math.atan((sensor_w * 0.5) / 50.0)
    fill = 1.0 - 2.0 * MARGIN
    dist = max(half_h / (fill * math.tan(v_half)),
               half_w / (fill * math.tan(h_half)),
               2.8)
    loc = look + direction * dist
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    xb.setup_studio(cam, look)
    print(f"fitted cam look={tuple(round(v,3) for v in look)} "
          f"half_w={half_w:.3f} half_h={half_h:.3f} dist={dist:.3f}", flush=True)
    return cam


def load_clip(spec):
    path = os.path.join(SRC, spec["file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    meshes, arm = xb.import_clip(path)
    bones = enrich_bones(arm, xb.bone_map(arm))
    f0, f1 = xb.frame_range(arm)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    scale = xb.scale_ground_center(arm, meshes, bones)
    xb.face_minus_y(arm, bones)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    start_xy = xb.hips_xy(arm, bones)
    rows = analyze_rich(arm, bones, f0, f1)
    xs = [r["hips"][0] for r in rows if r.get("hips")]
    ys = [r["hips"][1] for r in rows if r.get("hips")]
    drift = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if xs else 0.0
    lock = drift > xb.ROOT_DRIFT_LOCK_M
    lead, votes = detect_lead(rows)
    return {
        "meshes": meshes, "arm": arm, "bones": bones, "rows": rows,
        "f0": f0, "f1": f1, "scale": scale, "start_xy": start_xy,
        "clip_drift": drift, "lock": lock, "lead": lead, "lead_votes": votes,
        "path": path,
    }


def render_eevee(path, res=(270, 480), samples=8):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.filepath = path
    xb.apply_color_mgmt(sc)
    if hasattr(sc, "eevee"):
        try:
            sc.eevee.taa_render_samples = samples
        except Exception:
            pass
    bpy.ops.render.render(write_still=True)
    return path


def phase_analyze():
    os.makedirs(THUMB_DIR, exist_ok=True)
    facts = {"ok": False, "clips": [], "punches": [], "rankings": {}}
    t0 = time.time()
    for spec in CATALOG:
        print(f"ANALYZE {spec['id']} {spec['file']}", flush=True)
        try:
            ctx = load_clip(spec)
        except Exception as e:
            print(f"FAIL {spec['id']}: {e}", flush=True)
            facts["clips"].append({"id": spec["id"], "error": str(e), **spec})
            continue
        punches = extract_punches(ctx["rows"], ctx["lead"])
        clip_rec = {
            **spec,
            "frames": [ctx["f0"], ctx["f1"]],
            "lead": ctx["lead"],
            "lead_votes": ctx["lead_votes"],
            "clip_root_drift_m": ctx["clip_drift"],
            "root_locked": ctx["lock"],
            "height_m": ctx["scale"].get("height_m"),
            "n_punches": len(punches),
        }
        facts["clips"].append(clip_rec)

        xb.assign_clay(ctx["meshes"])
        # one camera fitted to all full frames (or whole clip if none)
        fulls = [p["full"] for p in punches] or [ctx["f0"], ctx["f1"]]
        setup_fitted_camera(ctx["meshes"], fulls, ctx["arm"], ctx["bones"],
                            ctx["start_xy"], ctx["lock"])

        for idx, p in enumerate(punches):
            pid = f"{spec['id']}_{p['ptype']}_{p['side']}_f{p['full']}"
            p["id"] = pid
            p["clip_id"] = spec["id"]
            p["clip_title"] = spec["title"]
            p["file"] = spec["file"]
            p["lead_side"] = ctx["lead"]
            scores = {}
            parts = {}
            for want in ("jab", "cross", "hook", "uppercut"):
                sc, br = score_punch(p, want)
                scores[want] = sc
                parts[want] = {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in br.items()}
            p["scores"] = {k: round(v, 2) for k, v in scores.items()}
            p["score_parts"] = parts
            p["best_for"] = p["ptype"]
            p["score"] = round(scores.get(p["ptype"], 0.0), 2)

            bpy.context.scene.frame_set(p["full"])
            bpy.context.view_layer.update()
            if ctx["lock"]:
                xb.lock_root_xy(ctx["arm"], ctx["bones"], ctx["start_xy"])
            thumb = os.path.join(THUMB_DIR, f"{pid}.png")
            render_eevee(thumb)
            p["thumb"] = thumb
            print(f"  punch {pid} type={p['ptype']} side={p['side']} "
                  f"elbow={p['elbow_deg']:.1f} fist_vs_head={p['fist_vs_head_m']:.3f} "
                  f"opp={p['opp_hand_to_head_m']:.3f} score={p['score']:.1f}",
                  flush=True)
            facts["punches"].append(p)

    rankings = {}
    for want in ("jab", "cross", "hook", "uppercut"):
        ranked = []
        for p in facts["punches"]:
            if p.get("ptype") != want:
                continue
            ranked.append(p)
        ranked.sort(key=lambda x: x["scores"].get(want, 0.0), reverse=True)
        rankings[want] = [
            {
                "id": p["id"], "clip_id": p["clip_id"], "title": p["clip_title"],
                "score": p["scores"].get(want, 0.0),
                "side": p["side"], "is_lead_hand": p["is_lead_hand"],
                "elbow_deg": p["elbow_deg"],
                "fist_vs_head_m": p["fist_vs_head_m"],
                "opp_hand_to_head_m": p["opp_hand_to_head_m"],
                "hip_rot_deg": p["hip_rot_deg"],
                "root_drift_m": p["root_drift_m"],
                "guard": p["guard"], "mid": p["mid"], "full": p["full"],
                "thumb": p.get("thumb"),
                "target": p.get("target"),
            }
            for p in ranked
        ]
        print(f"RANK {want}:", flush=True)
        for i, r in enumerate(rankings[want][:8]):
            print(f"  {i+1:2d}. {r['score']:5.1f}  {r['title']}  "
                  f"{r['id']}  elbow={r['elbow_deg']:.0f} "
                  f"fvh={r['fist_vs_head_m']:+.3f} opp={r['opp_hand_to_head_m']:.2f}",
                  flush=True)
    facts["rankings"] = rankings
    # default winners = top of each ranking (render phase may keep this)
    facts["winners"] = {
        k: (v[0]["id"] if v else None) for k, v in rankings.items()
    }
    facts["elapsed_s"] = time.time() - t0
    facts["ok"] = True
    os.makedirs(OUT, exist_ok=True)
    with open(FACTS_PATH, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("FACTS", FACTS_PATH, "punches", len(facts["punches"]),
          f"in {facts['elapsed_s']:.1f}s", flush=True)
    return facts


def _find_punch(facts, pid):
    for p in facts.get("punches", []):
        if p.get("id") == pid:
            return p
    return None


def phase_render(facts=None):
    if facts is None:
        with open(FACTS_PATH) as fh:
            facts = json.load(fh)
    winners = facts.get("winners") or {}
    t0 = time.time()
    rendered = {}
    for ptype, pid in winners.items():
        p = _find_punch(facts, pid)
        if not p:
            print(f"missing winner {ptype} {pid}", flush=True)
            continue
        spec = next(c for c in CATALOG if c["id"] == p["clip_id"])
        print(f"RENDER winner {ptype} {p['clip_title']} {pid}", flush=True)
        ctx = load_clip(spec)
        xb.assign_clay(ctx["meshes"])
        frames = [p["guard"], p["mid"], p["full"]]
        setup_fitted_camera(ctx["meshes"], frames, ctx["arm"], ctx["bones"],
                            ctx["start_xy"], ctx["lock"])
        stills = {}
        for stage in ("guard", "mid", "full"):
            f = int(p[stage])
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            if ctx["lock"]:
                xb.lock_root_xy(ctx["arm"], ctx["bones"], ctx["start_xy"])
            xb.paint_glow(ctx["meshes"], ctx["arm"], ctx["bones"],
                          p["side"], MUSCLES[ptype])
            out = os.path.join(OUT, f"xbot_v3_{ptype}_{stage}.png")
            xb.render_png(out)
            stills[stage] = out
            print(f"  WROTE {out}", flush=True)
        rendered[ptype] = {"stills": stills, "punch": p}

    # runner-up thumbs already exist from analyze; copy refs into facts
    facts["winner_stills"] = {k: v["stills"] for k, v in rendered.items()}
    facts["render_elapsed_s"] = time.time() - t0

    # preview: four winners in sequence, ~3 s at 15 fps
    preview = render_preview(facts, winners)
    facts["preview"] = preview
    facts["elapsed_s"] = facts.get("elapsed_s", 0) + (time.time() - t0)
    with open(FACTS_PATH, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("RENDER done", preview, flush=True)
    return facts


def render_preview(facts, winners):
    seq_dir = os.path.join(OUT, "_xbot_v3_seq")
    os.makedirs(seq_dir, exist_ok=True)
    order = ("jab", "cross", "hook", "uppercut")
    written = []
    t0 = time.time()
    fps = 15
    # budget ~12 frames per punch → ~3.2 s
    for ptype in order:
        pid = winners.get(ptype)
        p = _find_punch(facts, pid)
        if not p:
            continue
        if time.time() - t0 > 150 and written:
            break
        spec = next(c for c in CATALOG if c["id"] == p["clip_id"])
        ctx = load_clip(spec)
        xb.assign_clay(ctx["meshes"])
        frames_key = list(range(int(p["guard"]), int(p["full"]) + 2))
        if not frames_key:
            continue
        setup_fitted_camera(ctx["meshes"], [p["guard"], p["full"]],
                            ctx["arm"], ctx["bones"], ctx["start_xy"], ctx["lock"])
        step = max(1, int(math.ceil(len(frames_key) / 10)))
        pick = frames_key[::step][:10]
        if pick[-1] != frames_key[-1]:
            pick.append(frames_key[-1])
        for f in pick:
            if time.time() - t0 > 170:
                break
            bpy.context.scene.frame_set(int(f))
            bpy.context.view_layer.update()
            if ctx["lock"]:
                xb.lock_root_xy(ctx["arm"], ctx["bones"], ctx["start_xy"])
            xb.paint_glow(ctx["meshes"], ctx["arm"], ctx["bones"],
                          p["side"], MUSCLES[ptype])
            fp = os.path.join(seq_dir, f"f_{len(written):04d}.png")
            render_eevee(fp, res=(360, 640), samples=8)
            written.append(fp)
        # hold last frame 2 extra beats
        if written:
            hold = written[-1]
            for _ in range(2):
                fp = os.path.join(seq_dir, f"f_{len(written):04d}.png")
                if hold != fp:
                    import shutil
                    shutil.copy2(hold, fp)
                written.append(fp)
    if not written:
        return {"skipped": True, "reason": "no frames"}
    mp4 = os.path.join(OUT, "xbot_boxing_picks_v3_preview.mp4")
    import subprocess
    pattern = os.path.join(seq_dir, "f_%04d.png")
    n = len(written)
    # target 2–4 s
    duration = n / float(fps)
    if duration < 2.0:
        fps_use = max(8, int(round(n / 2.2)))
    elif duration > 4.0:
        fps_use = int(math.ceil(n / 3.8))
    else:
        fps_use = fps
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps_use), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4,
    ]
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        return {"skipped": True, "reason": str(e), "n": n}
    return {
        "skipped": False, "path": mp4, "bytes": os.path.getsize(mp4),
        "n_src": n, "fps": fps_use,
        "duration_s": n / float(fps_use),
        "elapsed_s": time.time() - t0,
    }


def main():
    phase, winners = _cli()
    if phase == "analyze":
        phase_analyze()
    elif phase == "render":
        facts = json.load(open(FACTS_PATH))
        if winners:
            facts["winners"].update(winners)
            with open(FACTS_PATH, "w") as fh:
                json.dump(facts, fh, indent=2, default=str)
        phase_render(facts)
    elif phase == "all":
        facts = phase_analyze()
        if winners:
            facts["winners"].update(winners)
        phase_render(facts)
    else:
        raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    main()
