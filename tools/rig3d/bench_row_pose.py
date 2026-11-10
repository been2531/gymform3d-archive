"""Classic bench-supported one-arm dumbbell row from explicit world targets.

World frame (Blender): +Z up, bench long axis = +Y (head toward +Y),
+X = working (right) side. Targets are written before any IK.
"""
from __future__ import annotations
import math
import numpy as np


def _n(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _ik(sh, wt, la, lf, pole_hint):
    """Two-bone IK; returns (elbow, wrist)."""
    sh = np.asarray(sh, np.float64)
    wt = np.asarray(wt, np.float64)
    pole_hint = np.asarray(pole_hint, np.float64)
    dvec = wt - sh
    d = float(np.linalg.norm(dvec))
    reach = 0.98 * (la + lf)
    if d < 1e-4:
        dvec = np.array([0.0, 0.0, -1e-4])
        d = 1e-4
    if d > reach:
        dvec = dvec * (reach / d)
        wt = sh + dvec
        d = reach
    u = dvec / d
    x = (la * la - lf * lf + d * d) / (2.0 * d)
    h = math.sqrt(max(la * la - x * x, 0.0))
    pole = pole_hint - sh
    pole = pole - np.dot(pole, u) * u
    pn = np.linalg.norm(pole)
    if pn < 1e-6:
        pole = np.array([1.0, 0.0, 0.0])
        pole = pole - np.dot(pole, u) * u
        pn = np.linalg.norm(pole)
    pole = pole / max(pn, 1e-9)
    return sh + u * x + pole * h, wt


def build_landmarks(u, L, side=12):
    """Return (B[n,33,3], spec) for an eased hang(u=0) -> peak(u=1) cycle.

    L keys: thigh, shin, upper_arm, forearm, torso, shoulder_half, hip_half, pad_z
    """
    u = np.asarray(u, np.float64)
    n = int(u.shape[0])
    Lt = float(L["thigh"]); Ls = float(L["shin"])
    La = float(L["upper_arm"]); Lf = float(L["forearm"])
    torso = float(L["torso"])
    half_w = float(L["shoulder_half"])
    hip_w = float(L["hip_half"])
    pad_z = float(L.get("pad_z", 0.43))
    hand_ahead = float(L.get("hand_ahead_m", 0.55))  # fallback only
    spec_pose = L.get("spec_pose") or {}
    align = spec_pose.get("alignment") or {}
    max_pitch = float(align.get("torso_from_horizontal_max_deg", 10.0))

    fwd = np.array([0.0, 1.0, 0.0])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    # Left support chain sits on the pad centerline (x=0). Body midline is
    # therefore +shoulder_half so the right shoulder hangs off the pad.
    x_sup = 0.0
    mid_x = x_sup + half_w
    y_knee = -0.10
    kn_z = pad_z + 0.04
    hand_z = pad_z + 0.02

    # Maximize thigh rise so the support shoulder sits as high as the skeleton
    # allows (straight vertical support arm needs ~La+Lf above the pad). Keep a
    # small +Y offset so the thigh reads as kneeling, not standing on the pad.
    lhip_x = mid_x - hip_w
    rhip_x = mid_x + hip_w
    dx = lhip_x - x_sup
    hip_y = y_knee + 0.08
    dy = hip_y - y_knee
    horiz = math.hypot(dx, dy)
    if horiz >= Lt * 0.98:
        rest = math.sqrt(max((Lt * 0.95) ** 2 - dx * dx, 1e-6))
        hip_y = y_knee + rest
        horiz = math.hypot(dx, hip_y - y_knee)
    rise = math.sqrt(max(Lt * Lt - horiz * horiz, 1e-4))
    hip_z = kn_z + rise
    # Cap hip height so the standing leg can still plant flat on the floor.
    max_hip_for_foot = 0.055 + 0.985 * (Lt + Ls)
    hip_z = min(hip_z, max_hip_for_foot)

    lhip = np.array([lhip_x, hip_y, hip_z])
    rhip = np.array([rhip_x, hip_y, hip_z])
    hip_mid = 0.5 * (lhip + rhip)

    # Keep the back flat. Retarget's spine chain adds a few degrees of pitch,
    # so do not spend the spec's 10° budget here. Leftover support-arm bend is
    # a Grab3D thigh-vs-arm proportion limit, not a placement error.
    sh_z = hip_z
    _ = (max_pitch, hand_z, La, Lf)  # kept in scope for readability of nearby code

    lsh = np.array([x_sup, hip_y + torso, sh_z])
    rsh = np.array([mid_x + half_w, hip_y + torso, sh_z])
    sh_mid = 0.5 * (lsh + rsh)

    lkn = np.array([x_sup, y_knee, kn_z])
    # Shin flat on the pad along -Y (toward the near end).
    lank = np.array([x_sup, y_knee - Ls, pad_z + 0.03])
    lheel = np.array([x_sup, y_knee - Ls + 0.04, pad_z + 0.02])
    ltoe = np.array([x_sup, y_knee - Ls - 0.08, pad_z + 0.02])

    # Support hand flat on the pad, directly under the support shoulder.
    lwr = np.array([x_sup, float(lsh[1]), hand_z])
    lel, lwr = _ik(lsh, lwr, La, Lf, lsh + np.array([0.0, -0.04, -0.15]))

    # Opposite foot planted flat beside the bench, slightly wide, knee soft.
    reach_leg = 0.99 * (Lt + Ls)
    down = hip_z - 0.055
    if down >= reach_leg:
        rank = np.array([rhip_x + 0.18, hip_y - 0.04, hip_z - reach_leg])
    else:
        leftover = math.sqrt(max(reach_leg * reach_leg - down * down, 0.0))
        rank = np.array([rhip_x + min(0.22, 0.55 * leftover + 0.12),
                         hip_y - min(0.10, 0.35 * leftover), 0.055])
    rkn, rank = _ik(rhip, rank, Lt, Ls, rhip + np.array([0.16, 0.02, -0.18]))
    rtoe = np.array([rank[0], rank[1] + 0.16, 0.015])
    rheel = np.array([rank[0], rank[1] - 0.05, 0.015])

    # Working arm: hang straight down; peak = elbow driven back past the torso,
    # dumbbell sitting beside the hip.
    wr_hang = rsh + np.array([0.02, 0.0, -(La + Lf) * 0.98])
    el_hang, wr_hang = _ik(rsh, wr_hang, La, Lf, rsh + np.array([0.04, 0.0, -0.40]))
    wr_peak = rhip + np.array([0.12, 0.02, -0.03])
    el_peak, wr_peak = _ik(rsh, wr_peak, La, Lf, rsh + np.array([0.08, -0.35, 0.16]))
    _ = hand_ahead  # kept so older configs still load

    # Head / face: continue the spine; a small downward look keeps the gaze on the floor.
    spine_dir = _n(sh_mid - hip_mid)
    head = sh_mid + spine_dir * 0.20
    nose = sh_mid + spine_dir * 0.16 + np.array([0.0, 0.0, -0.04])
    lear = head + np.array([-0.07, 0.0, 0.0])
    rear = head + np.array([0.07, 0.0, 0.0])

    B = np.zeros((n, 33, 3), np.float64)
    for i, uu in enumerate(u):
        # static body
        B[i, 23] = lhip; B[i, 24] = rhip
        B[i, 11] = lsh; B[i, 12] = rsh
        B[i, 25] = lkn; B[i, 27] = lank; B[i, 29] = lheel; B[i, 31] = ltoe
        B[i, 26] = rkn; B[i, 28] = rank; B[i, 30] = rheel; B[i, 32] = rtoe
        B[i, 13] = lel; B[i, 15] = lwr; B[i, 19] = lwr + fwd * 0.06
        B[i, 0] = nose
        B[i, 7] = lear; B[i, 8] = rear
        for j in (1, 2, 3, 4, 5, 6, 9, 10):
            B[i, j] = head
        # working arm eases hang -> peak
        B[i, 14] = el_hang + (el_peak - el_hang) * uu
        B[i, 16] = wr_hang + (wr_peak - wr_hang) * uu
        B[i, 20] = B[i, 16] + np.array([0.0, 0.0, -0.05])
        B[i, 18] = B[i, 16]
        B[i, 22] = B[i, 16]
        B[i, 17] = B[i, 15]
        B[i, 21] = B[i, 15]

    # Bench: local +X (length) -> world +Y, origin at pad centre.
    yaw_z = math.pi / 2.0
    spec = {
        "origin_xy": (0.0, 0.0),
        "origin_z": 0.39,
        "yaw_z": yaw_z,
        "support_knee": lkn.tolist(),
        "support_ankle": lank.tolist(),
        "support_hand": lwr.tolist(),
        "support_left": True,
        "fwd": fwd.tolist(),
        "right": right.tolist(),
        "pad_z": pad_z,
        "pad_half_xy": (0.15, 0.60),  # world (|x|, |y|) extents after +Y alignment (30 x 120 cm)
        "hip_z": float(hip_z),
        "shoulder": rsh.tolist(),
        "hip_r": rhip.tolist(),
        "wr_hang": wr_hang.tolist(),
        "wr_peak": wr_peak.tolist(),
        "look": "gunmetal_black",
        "world_targets": True,
    }
    return B, spec


def pad_contains(p, spec, margin=0.02):
    hx, hy = spec["pad_half_xy"]
    ox, oy = spec["origin_xy"]
    return (abs(p[0] - ox) <= hx + margin) and (abs(p[1] - oy) <= hy + margin)


def qc_from_bones(rec_pts, spec, u, cam_ndc_db=None, db_visible_frac=None, db_contrast=None):
    """Geometric QC against explicit targets. rec_pts uses bone-head world coords."""
    pad_z = float(spec["pad_z"])
    lkn = rec_pts["lkn"]; lwr = rec_pts["lwr"]; rank = rec_pts["rank"]
    rsh = rec_pts["rsh"]; rwr = rec_pts["rwr"]; rhip = rec_pts["rhip"]
    hip = rec_pts["hip"]; sh = rec_pts["sh"]; head = rec_pts.get("head", sh)
    rel = rec_pts.get("rel")
    lsh = rec_pts.get("lsh")
    lel = rec_pts.get("lel")
    lank = rec_pts.get("lank")
    rtoe = rec_pts.get("rtoe")
    rkn = rec_pts.get("rkn")
    tv = sh - hip
    torso_from_vert = math.degrees(math.atan2(math.hypot(tv[0], tv[1]), tv[2]))
    torso_from_horiz = abs(90.0 - torso_from_vert)
    spine = _n(tv)
    head_dir = _n(head - sh)
    head_spine = math.degrees(math.acos(float(np.clip(np.dot(spine, head_dir), -1, 1))))

    elbow_deg = None
    if rel is not None:
        ue = rsh - rel; ve = rwr - rel
        nu, nv = float(np.linalg.norm(ue)), float(np.linalg.norm(ve))
        if nu > 1e-6 and nv > 1e-6:
            elbow_deg = math.degrees(math.acos(float(np.clip(np.dot(ue, ve) / (nu * nv), -1, 1))))
    ua = (rel - rsh) if rel is not None else (rwr - rsh)
    over_head = bool(np.dot(ua, spine) > 0.12 and ua[2] > 0.12)
    tn = spine
    wrel = rwr - hip
    tt = float(np.clip(np.dot(wrel, tn), 0.0, float(np.linalg.norm(tv))))
    near = hip + tn * tt
    clear = float(np.linalg.norm(rwr - near))

    hand_on_pad = pad_contains(lwr, spec) and abs(lwr[2] - pad_z) < 0.06
    knee_on_pad = pad_contains(lkn, spec) and abs(lkn[2] - pad_z) < 0.08
    foot_on_floor = rank[2] < 0.16 and rank[0] > spec["origin_xy"][0] + spec["pad_half_xy"][0] - 0.02
    below = (rsh[2] - rwr[2]) > 0.22
    near_hip = float(np.linalg.norm(rwr - rhip)) < 0.22

    support_arm_from_vertical_deg = None
    support_hand_under_shoulder = None
    support_elbow_deg = None
    if lsh is not None:
        arm = np.asarray(lwr, float) - np.asarray(lsh, float)
        an = float(np.linalg.norm(arm))
        if an > 1e-6:
            down = np.array([0.0, 0.0, -1.0])
            support_arm_from_vertical_deg = math.degrees(
                math.acos(float(np.clip(np.dot(arm / an, down), -1, 1))))
        support_hand_under_shoulder = bool(
            math.hypot(float(lwr[0] - lsh[0]), float(lwr[1] - lsh[1])) <= 0.08)
        if lel is not None:
            ue = np.asarray(lsh, float) - np.asarray(lel, float)
            ve = np.asarray(lwr, float) - np.asarray(lel, float)
            nu, nv = float(np.linalg.norm(ue)), float(np.linalg.norm(ve))
            if nu > 1e-6 and nv > 1e-6:
                support_elbow_deg = math.degrees(
                    math.acos(float(np.clip(np.dot(ue, ve) / (nu * nv), -1, 1))))
    shin_on_pad = bool(
        knee_on_pad and lank is not None
        and pad_contains(lank, spec) and abs(lank[2] - pad_z) < 0.08)
    standing_foot_flat = bool(
        foot_on_floor and rank[2] < 0.12
        and (rtoe is None or float(rtoe[2]) < 0.06))
    elbow_past_torso = None
    if rel is not None:
        elbow_past_torso = bool(float(rel[1]) <= float(rsh[1]) + 0.01)
    standing_knee_deg = None
    if rkn is not None:
        ua = np.asarray(rhip, float) - np.asarray(rkn, float)
        va = np.asarray(rank, float) - np.asarray(rkn, float)
        nu, nv = float(np.linalg.norm(ua)), float(np.linalg.norm(va))
        if nu > 1e-6 and nv > 1e-6:
            standing_knee_deg = math.degrees(
                math.acos(float(np.clip(np.dot(ua, va) / (nu * nv), -1, 1))))

    out = {
        "torso_from_horizontal_deg": round(torso_from_horiz, 2),
        "torso_from_vertical_deg": round(torso_from_vert, 2),
        "head_spine_deg": round(head_spine, 2),
        "support_hand_on_pad": bool(hand_on_pad),
        "support_hand_dz": round(float(lwr[2] - pad_z), 4),
        "knee_on_pad": bool(knee_on_pad),
        "knee_dz": round(float(lkn[2] - pad_z), 4),
        "foot_on_floor": bool(foot_on_floor),
        "foot_z": round(float(rank[2]), 4),
        "working_hand_below_shoulder": bool(below),
        "working_hand_drop_m": round(float(rsh[2] - rwr[2]), 4),
        "working_hand_near_hip": bool(near_hip),
        "working_hand_hip_m": round(float(np.linalg.norm(rwr - rhip)), 4),
        "working_elbow_deg": None if elbow_deg is None else round(float(elbow_deg), 1),
        "support_hand_under_shoulder": support_hand_under_shoulder,
        "support_arm_from_vertical_deg": None if support_arm_from_vertical_deg is None
            else round(float(support_arm_from_vertical_deg), 2),
        "support_elbow_deg": None if support_elbow_deg is None else round(float(support_elbow_deg), 1),
        "shin_on_pad": bool(shin_on_pad),
        "standing_foot_flat": bool(standing_foot_flat),
        "elbow_past_torso": elbow_past_torso,
        "standing_knee_deg": None if standing_knee_deg is None else round(float(standing_knee_deg), 1),
        "shoulder_over_head": bool(over_head),
        "arm_torso_clear_m": round(clear, 4),
        "u": round(float(u), 3),
    }
    if cam_ndc_db is not None:
        xs, ys = cam_ndc_db
        in_frame = (0.04 < xs < 0.96) and (0.04 < ys < 0.96)
        out["db_in_frame"] = bool(in_frame)
        out["db_ndc"] = [round(float(xs), 3), round(float(ys), 3)]
    if db_visible_frac is not None:
        out["db_visible_frac"] = round(float(db_visible_frac), 3)
    if db_contrast is not None:
        out["db_contrast"] = round(float(db_contrast), 2)

    gates = {
        "torso_horizontal": bool(torso_from_horiz <= 10.0),
        "head_spine": bool(head_spine <= 22.0),
        "support_hand_on_pad": bool(hand_on_pad),
        "knee_on_pad": bool(knee_on_pad),
        "foot_on_floor": bool(foot_on_floor),
        "elbow_rom": True if elbow_deg is None else bool(40.0 <= elbow_deg <= 170.0),
        "shoulder_rom": bool(not over_head),
        "no_arm_torso_penetration": bool(clear >= 0.12 and rwr[0] >= hip[0] - 0.02),
    }
    if float(u) < 0.15:
        gates["working_hand_below_shoulder"] = bool(below)
    if float(u) > 0.85:
        gates["working_hand_near_hip"] = bool(near_hip)
    if cam_ndc_db is not None:
        gates["db_in_frame"] = bool(out.get("db_in_frame", False))
    if db_visible_frac is not None:
        gates["db_visible"] = bool(db_visible_frac >= 0.35)
    if db_contrast is not None:
        gates["db_contrast"] = bool(db_contrast >= 12.0)
    out["gates"] = {k: bool(v) for k, v in gates.items()}
    out["pass"] = bool(all(out["gates"].values()))
    return out
