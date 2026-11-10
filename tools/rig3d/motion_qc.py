#!/usr/bin/env python3
"""Motion smoothness QC (joint jerk, foot slide, head spin, loop seam, elbow lockout speed).

Limits live in cfg['qc']['motion_smooth']. Missing key = that check is skipped.
Designed so a raw MediaPipe take fails jerk / seam / lockout, and an authored cycle passes.
"""
from __future__ import annotations
import json, math, os
import numpy as np


def _ck(name, value, limit, ok, fix=None):
    return {"name": name, "group": "motion_smooth", "value": value, "limit": limit,
            "pass": bool(ok), "fix": None if ok else fix}


def _heads(P, ix, n):
    return P[:, ix[n], :3, 3]


def metrics(npz, fps):
    D = np.load(npz, allow_pickle=True)
    names = [str(n) for n in D["names"]]
    ix = {n: i for i, n in enumerate(names)}
    P = D["pose"]
    fps = float(fps)
    dt = 1.0 / max(fps, 1e-6)
    out = {"fps": fps, "frames": int(P.shape[0])}

    def jerk(X):
        v = np.gradient(X, dt, axis=0)
        a = np.gradient(v, dt, axis=0)
        j = np.gradient(a, dt, axis=0)
        return np.linalg.norm(j, axis=1)

    jerks = {}
    for n in ("Head", "neck", "LeftHand", "RightHand", "LeftForeArm", "RightForeArm",
              "LeftArm", "RightArm", "LeftFoot", "RightFoot", "Hips"):
        if n not in ix:
            continue
        jm = jerk(_heads(P, ix, n))
        jerks[n] = {"max": float(jm.max()), "p95": float(np.percentile(jm, 95))}
    out["jerk"] = jerks

    slides = {}
    for n in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"):
        if n not in ix:
            continue
        X = _heads(P, ix, n)
        slides[n] = float(np.linalg.norm(X - np.median(X, 0), axis=1).max() * 1000.0)
    out["foot_slide_mm"] = slides

    if "Head" in ix:
        R = P[:, ix["Head"], :3, :3]
        angs = []
        for i in range(1, len(R)):
            d = R[i - 1].T @ R[i]
            tr = float(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0))
            angs.append(math.degrees(math.acos(tr)) * fps)
        out["head_ang_vel_max_deg_s"] = float(max(angs)) if angs else 0.0
    else:
        out["head_ang_vel_max_deg_s"] = None

    seam = {}
    for n in ("Head", "Hips", "LeftFoot", "RightFoot", "LeftHand", "RightHand"):
        if n not in ix:
            continue
        seam[n] = float(np.linalg.norm(P[0, ix[n], :3, 3] - P[-1, ix[n], :3, 3]) * 1000.0)
    out["seam_mm"] = seam
    out["seam_max_mm"] = float(max(seam.values())) if seam else 0.0

    # working elbow: prefer the side with larger ROM
    flex = None
    which = None
    for s in ("Right", "Left"):
        if not all(b in ix for b in (s + "Arm", s + "ForeArm", s + "Hand")):
            continue
        sh, el, wt = _heads(P, ix, s + "Arm"), _heads(P, ix, s + "ForeArm"), _heads(P, ix, s + "Hand")
        interior = np.degrees(np.arccos(np.clip(
            ((sh - el) * (wt - el)).sum(1)
            / (np.linalg.norm(sh - el, axis=1) * np.linalg.norm(wt - el, axis=1) + 1e-12),
            -1, 1)))
        fl = 180.0 - interior
        if flex is None or (fl.max() - fl.min()) > (flex.max() - flex.min()):
            flex, which = fl, s
    if flex is not None:
        spd = np.abs(np.gradient(flex, dt))
        near = flex < 25.0
        out["elbow_flex_deg"] = [float(flex.min()), float(flex.max())]
        out["elbow_side"] = which
        out["elbow_speed_max_deg_s"] = float(spd.max())
        out["elbow_lockout_speed_deg_s"] = float(spd[near].max()) if near.any() else 0.0
        # humerus vs world-down (0 = hanging, 90 = horizontal)
        sh, el = _heads(P, ix, which + "Arm"), _heads(P, ix, which + "ForeArm")
        hum = el - sh
        hum /= np.linalg.norm(hum, axis=1, keepdims=True) + 1e-12
        down = np.array([0.0, 0.0, -1.0])
        out["humerus_from_down_deg"] = [
            float(np.degrees(np.arccos(np.clip((hum * down).sum(1), -1, 1))).min()),
            float(np.degrees(np.arccos(np.clip((hum * down).sum(1), -1, 1))).max()),
        ]
        # forearm vs world-vertical (0 = perfectly vertical)
        wt = _heads(P, ix, which + "Hand")
        fore = wt - el
        fore /= np.linalg.norm(fore, axis=1, keepdims=True) + 1e-12
        out["forearm_from_vertical_max_deg"] = float(np.degrees(np.arccos(np.clip((fore * down).sum(1), -1, 1))).max())
    return out


def checks(rdir, Q, fps):
    bz = os.path.join(rdir, "bones.npz")
    if not os.path.exists(bz):
        return [_ck("motion_bones_present", False, True, False)]
    M = metrics(bz, fps)
    out = []
    r = lambda v, k=2: round(float(v), k)
    if "joint_jerk_max" in Q:
        joints = Q.get("jerk_joints") or ["Head", "RightHand", "RightForeArm", "LeftHand", "LeftForeArm"]
        vals = {n: r(M["jerk"][n]["max"], 1) for n in joints if n in M["jerk"]}
        mx = max(vals.values()) if vals else 0.0
        out.append(_ck("joint_jerk_max", {"max": mx, "per": vals}, f"<= {Q['joint_jerk_max']} m/s^3",
                       mx <= float(Q["joint_jerk_max"]), "authored_motion"))
    if "head_jerk_max" in Q and "Head" in M["jerk"]:
        v = M["jerk"]["Head"]["max"]
        out.append(_ck("head_jerk_max", r(v, 2), f"<= {Q['head_jerk_max']} m/s^3",
                       v <= float(Q["head_jerk_max"]), "authored_motion"))
    if "foot_slide_mm" in Q:
        v = max(M["foot_slide_mm"].values()) if M["foot_slide_mm"] else 0.0
        out.append(_ck("foot_slide_mm", r(v, 2), f"<= {Q['foot_slide_mm']} mm",
                       v <= float(Q["foot_slide_mm"]), "lock_feet"))
    if "head_ang_vel_max_deg_s" in Q and M.get("head_ang_vel_max_deg_s") is not None:
        v = M["head_ang_vel_max_deg_s"]
        out.append(_ck("head_ang_vel_max_deg_s", r(v, 1), f"<= {Q['head_ang_vel_max_deg_s']}",
                       v <= float(Q["head_ang_vel_max_deg_s"]), "authored_motion"))
    if "seam_pos_mm" in Q:
        v = M["seam_max_mm"]
        out.append(_ck("segment_seam_pose_delta_mm", {"max": r(v, 2), "per": {k: r(x, 2) for k, x in M["seam_mm"].items()}},
                       f"<= {Q['seam_pos_mm']} mm (frame 0 vs last)", v <= float(Q["seam_pos_mm"]), "authored_loop"))
    if "elbow_lockout_speed_max_deg_s" in Q and "elbow_lockout_speed_deg_s" in M:
        v = M["elbow_lockout_speed_deg_s"]
        out.append(_ck("elbow_lockout_speed_deg_s", r(v, 1),
                       f"<= {Q['elbow_lockout_speed_max_deg_s']} when flex < 25°",
                       v <= float(Q["elbow_lockout_speed_max_deg_s"]), "soft_lockout"))
    json.dump({k: (M[k] if not isinstance(M[k], dict) else M[k]) for k in M},
              open(os.path.join(rdir, "motion_smooth.json"), "w"), default=float, indent=1)
    return out


if __name__ == "__main__":
    import sys
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 25
    print(json.dumps(metrics(sys.argv[1], fps), indent=2, default=float))
