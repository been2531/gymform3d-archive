#!/usr/bin/env python3
"""Generic rig-level posture QC (pure geometry from the evaluated armature, no AI, no pixels). Shared by every exercise.
Input: render/bones.npz written by retarget_render_v2.py: names (B,), rest (B,4,4) world rest matrices, pose (F,B,4,4)
world pose matrices per frame. Limits come from the exercise config `qc.posture` (missing key = check skipped).

Head / neck (relative to the chest bone the neck hangs from, so a hinged torso is not counted as a bent neck):
  neck_line_dev_deg   angle between the thoracic line (Spine02 -> neck base) and the neck line (neck base -> head top),
                      minus the same angle at rest = head-to-torso line deviation (0 = head continues the spine)
  neck_flex_deg       signed sagittal part of that deviation (+ = chin toward chest, - = head tipped back / gaze up)
  head_roll_deg       ear-to-shoulder tilt of the head relative to the chest (rotation about the chest's forward axis)
  head_yaw_deg        head turned left/right relative to the chest (twist about the neck axis)
  neck_steady         max |neck_flex - median| over the rep
Legs:
  feet planted        toe-base height above the floor (lowest rest toe) and foot pitch vs rest, both feet, every frame
  foot drift          max distance of ankle / toe-base from their per-rep median
  knee angle          hip-knee-ankle angle per leg (range) + steadiness
  thigh split         sagittal angle between the two thighs (a running stride is one thigh forward, one back)
  knees soft+balanced both knees inside a soft range and |median L - median R| small (v5)
Dumbbell (v5, needs the dumbbell matrices + palm axis in the npz):
  handle orientation  handle axis vs the across-palm axis of the working hand and vs perpendicular-to-forearm
  neutral grip        handle (horizontal part) vs the body's horizontal forward axis (row: handle front-to-back)
  support hand        non-working wrist to the nearest thigh bone segment (braced on the thigh, not floating)
usage: python3 tools/rig3d/posture_qc.py RENDER_DIR/bones.npz [exercise.yaml]"""
import json, math, sys
import numpy as np

def _u(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
def _ang(a, b): return np.degrees(np.arccos(np.clip((_u(a) * _u(b)).sum(-1), -1, 1)))

def measure(npz):
    D = np.load(npz, allow_pickle=True); names = [str(n) for n in D["names"]]; P = D["pose"]; R0 = D["rest"]; ix = {n: i for i, n in enumerate(names)}
    head = lambda n: P[:, ix[n], :3, 3]; rhead = lambda n: R0[ix[n], :3, 3]
    rot = lambda n: P[:, ix[n], :3, :3]; rrot = lambda n: R0[ix[n], :3, :3]
    chest = "Spine"; low = "Spine02"; top = "head_end" if "head_end" in ix else "Head"
    out = {"frames": int(P.shape[0])}
    # head/neck: thoracic line vs neck line, expressed in the chest frame (X lateral, Y up the spine, Z = X x Y forward)
    Rc = rot(chest); thor = head("neck") - head(low); nl = head(top) - head("neck")
    thor0 = rhead("neck") - rhead(low); nl0 = rhead(top) - rhead("neck")
    loc = lambda v: np.einsum("fji,fj->fi", Rc, v)                     # world -> chest-local
    t_l, n_l = loc(thor), loc(nl); Rc0 = rrot(chest); t0, n0 = Rc0.T @ thor0, Rc0.T @ nl0
    def sag(v): return np.degrees(np.arctan2(v[..., 2], v[..., 1]))   # angle in the Y-Z (sagittal) plane, + toward forward
    def cor(v): return np.degrees(np.arctan2(v[..., 0], v[..., 1]))   # angle in the Y-X (coronal) plane
    flex = (sag(n_l) - sag(t_l)) - (sag(n0) - sag(t0))
    # forward = chest -Z or +Z? decide from the rig's own 'headfront' bone (face direction at rest) if present
    if "headfront" in ix:
        fz = (Rc0.T @ (rhead("headfront") - rhead("Head")))[2]
        if fz < 0: flex = -flex
    line_dev = np.abs(_ang(thor, nl) - _ang(thor0[None], nl0[None]))
    # head orientation relative to chest, vs rest: Q = (Rc^T Rh)(Rc0^T Rh0)^T in chest-local axes
    Rh, Rh0 = rot("Head"), rrot("Head"); Q = np.einsum("fji,fjk->fik", Rc, Rh) @ (Rc0.T @ Rh0).T
    # swing-twist about the chest Y (spine) axis: yaw = twist, roll = coronal tilt of the swung Y axis
    y = Q[:, :, 1]; roll = np.degrees(np.arcsin(np.clip(y[:, 0], -1, 1)))   # sideways lean of the head axis (bounded +-90)
    x = Q[:, :, 0]; xs = x - y * (x * y).sum(1, keepdims=True)          # lateral axis, made perpendicular to the head axis
    yaw = np.degrees(np.arctan2(-xs[:, 2], xs[:, 0]))
    out.update(neck_flex_deg=flex, neck_line_dev_deg=line_dev, head_roll_deg=roll, head_yaw_deg=yaw)
    # legs
    floor = min(rhead("LeftToeBase")[2], rhead("RightToeBase")[2])
    for s in ("Left", "Right"):
        hp, kn, an, to = head(s + "UpLeg"), head(s + "Leg"), head(s + "Foot"), head(s + "ToeBase")
        knee = _ang(hp - kn, an - kn); fv, fv0 = to - an, rhead(s + "ToeBase") - rhead(s + "Foot")
        pitch = np.degrees(np.arctan2(-fv[:, 2], np.hypot(fv[:, 0], fv[:, 1]))); pitch0 = math.degrees(math.atan2(-fv0[2], math.hypot(fv0[0], fv0[1])))
        drift = max(np.linalg.norm(an - np.median(an, 0), axis=1).max(), np.linalg.norm(to - np.median(to, 0), axis=1).max())
        out[s] = {"knee_deg": knee, "toe_h_m": to[:, 2] - floor, "pitch_dev_deg": pitch - pitch0, "drift_m": float(drift), "thigh": hp - kn}
    # thigh split in the body's sagittal plane: lateral axis from the two hip joints
    lat = _u(head("LeftUpLeg") - head("RightUpLeg"))
    def proj(v): return _u(v - lat * (v * lat).sum(1, keepdims=True))
    out["thigh_split_deg"] = _ang(proj(-out["Left"]["thigh"]), proj(-out["Right"]["thigh"]))
    # support hand (v5): distance from the non-working wrist to the nearest thigh segment (bone line, not skin)
    if "working_hand" in D.files:
        sp = "Right" if str(D["working_hand"]).startswith("Left") else "Left"; w = head(sp + "Hand"); best = None
        for s in ("Left", "Right"):
            a_, b_ = head(s + "UpLeg"), head(s + "Leg"); ab = b_ - a_; t = np.clip(((w - a_) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
            dd = np.linalg.norm(w - (a_ + ab * t[:, None]), axis=1); best = dd if best is None else np.minimum(best, dd)
        out["support_hand_to_thigh_m"] = best
    # dumbbell handle vs the working hand (v5): needs dumbbell (F,4,4) + handle_axis_local + palm_<side> in the npz
    if "dumbbell" in D.files and "palm_left" in D.files:
        hb = str(D["working_hand"]); fb = hb.replace("Hand", "ForeArm"); pl = D["palm_left" if hb.startswith("Left") else "palm_right"]
        hax = _u(np.einsum("fij,j->fi", D["dumbbell"][:, :3, :3], D["handle_axis_local"]))
        Rh = rot(hb); hy = Rh[:, :, 1]; palm = _u(np.einsum("fij,j->fi", Rh, pl)); palm = _u(palm - hy * (palm * hy).sum(1, keepdims=True))
        fore = _u(head(hb) - head(fb))
        out["handle_vs_palm_deg"] = np.minimum(_ang(hax, palm), 180 - _ang(hax, palm))            # handle has no direction
        out["handle_vs_forearm_perp_deg"] = np.abs(90 - _ang(hax, fore))
        tv = np.mean(head("Spine") - head("Hips"), 0); fw = np.array([tv[0], tv[1], 0.0]); fw /= np.linalg.norm(fw)
        hh = hax.copy(); hh[:, 2] = 0; out["handle_vs_body_forward_deg"] = np.minimum(_ang(hh, fw[None]), 180 - _ang(hh, fw[None]))
    return out

def _ck(name, value, limit, ok, fix=None):
    return {"name": name, "group": "posture", "value": value, "limit": limit, "pass": bool(ok), "fix": None if ok else fix}

def checks(npz, Q):
    """Q = cfg['qc']['posture']. Returns (list of check dicts, summary of per-frame series)."""
    M = measure(npz); out = []; r = lambda v, n=1: round(float(v), n)
    absmax = lambda a: float(np.abs(a).max()); wf = lambda a: int(np.abs(a).argmax()) + 1
    if "neck_line_dev_max_deg" in Q:
        v = M["neck_line_dev_deg"]; out.append(_ck("neck_in_line_with_spine_deg", r(v.max()), f"<= {Q['neck_line_dev_max_deg']}", v.max() <= Q["neck_line_dev_max_deg"], "neck_follow_spine")); out[-1]["where"] = {"frame": wf(v)}
    if "neck_flex_range_deg" in Q:
        v = M["neck_flex_deg"]; lo, hi = Q["neck_flex_range_deg"]
        out.append(_ck("neck_flex_deg_min_max", [r(v.min()), r(v.max())], [lo, hi], lo <= v.min() and v.max() <= hi, "neck_follow_spine"))
    if "head_roll_max_deg" in Q:
        v = M["head_roll_deg"]; out.append(_ck("head_roll_deg", r(absmax(v)), f"<= {Q['head_roll_max_deg']}", absmax(v) <= Q["head_roll_max_deg"], "neck_follow_spine")); out[-1]["where"] = {"frame": wf(v)}
    if "head_yaw_max_deg" in Q:
        v = M["head_yaw_deg"]; out.append(_ck("head_yaw_deg", r(absmax(v)), f"<= {Q['head_yaw_max_deg']}", absmax(v) <= Q["head_yaw_max_deg"], "neck_follow_spine"))
    if "neck_steady_max_deg" in Q:
        v = M["neck_flex_deg"]; d = absmax(v - np.median(v)); out.append(_ck("neck_steady_maxdev_deg", r(d, 2), f"<= {Q['neck_steady_max_deg']}", d <= Q["neck_steady_max_deg"], "neck_follow_spine"))
    if "toe_lift_max_m" in Q:
        v = max(absmax(M[s]["toe_h_m"]) for s in ("Left", "Right")); p = max(absmax(M[s]["pitch_dev_deg"]) for s in ("Left", "Right"))
        ok = v <= Q["toe_lift_max_m"] and p <= Q["foot_pitch_dev_max_deg"]
        out.append(_ck("both_feet_planted", {"toe_off_floor_m": r(v, 3), "foot_pitch_dev_deg": r(p)}, f"toe <= {Q['toe_lift_max_m']} m, pitch dev <= {Q['foot_pitch_dev_max_deg']}", ok, "plant_legs"))
    if "foot_drift_max_m" in Q:
        v = max(M[s]["drift_m"] for s in ("Left", "Right")); out.append(_ck("foot_drift_m", r(v, 4), f"<= {Q['foot_drift_max_m']}", v <= Q["foot_drift_max_m"], "plant_legs"))
    if "knee_range_deg" in Q:
        lo, hi = Q["knee_range_deg"]; kv = {s[0]: [r(M[s]["knee_deg"].min()), r(M[s]["knee_deg"].max())] for s in ("Left", "Right")}
        ok = all(lo <= a and b <= hi for a, b in kv.values()); out.append(_ck("knee_angles_deg", kv, [lo, hi], ok, "plant_legs"))
    if "knee_steady_max_deg" in Q:
        d = max(absmax(M[s]["knee_deg"] - np.median(M[s]["knee_deg"])) for s in ("Left", "Right"))
        out.append(_ck("knee_steady_maxdev_deg", r(d, 2), f"<= {Q['knee_steady_max_deg']}", d <= Q["knee_steady_max_deg"], "plant_legs"))
    if "knee_soft_range_deg" in Q:      # v5: soft AND balanced (no locked-straight leg next to a bent one)
        lo, hi = Q["knee_soft_range_deg"]; med = {s[0]: float(np.median(M[s]["knee_deg"])) for s in ("Left", "Right")}
        mn = {s[0]: float(M[s]["knee_deg"].min()) for s in ("Left", "Right")}; mx = {s[0]: float(M[s]["knee_deg"].max()) for s in ("Left", "Right")}
        dlr = abs(med["L"] - med["R"]); ok = all(lo <= mn[k] and mx[k] <= hi for k in mn) and dlr <= Q.get("knee_lr_diff_max_deg", 20)
        out.append(_ck("knees_soft_and_balanced", {"L": [r(mn["L"]), r(mx["L"])], "R": [r(mn["R"]), r(mx["R"])], "LR_diff": r(dlr)},
                       f"both in [{lo}, {hi}], |L-R| <= {Q.get('knee_lr_diff_max_deg', 20)}", ok, "balance_knees"))
    if "handle_palm_max_deg" in Q:
        if "handle_vs_palm_deg" not in M: out.append(_ck("dumbbell_handle_orientation", None, "bones.npz without dumbbell", False))
        else:
            a1, a2 = M["handle_vs_palm_deg"], M["handle_vs_forearm_perp_deg"]; v = np.maximum(a1, a2)
            out.append(_ck("dumbbell_handle_orientation", {"vs_palm_axis": r(a1.max()), "vs_forearm_perp": r(a2.max())}, f"<= {Q['handle_palm_max_deg']} every frame",
                           v.max() <= Q["handle_palm_max_deg"], "grip_palm")); out[-1]["where"] = {"frame": int(v.argmax()) + 1}
    if "handle_body_forward_max_deg" in Q and "handle_vs_body_forward_deg" in M:
        v = M["handle_vs_body_forward_deg"]; out.append(_ck("dumbbell_neutral_grip_deg", r(v.max()), f"<= {Q['handle_body_forward_max_deg']} every frame",
                                                            v.max() <= Q["handle_body_forward_max_deg"], "grip_palm")); out[-1]["where"] = {"frame": int(v.argmax()) + 1}
    if "support_hand_thigh_max_m" in Q and "support_hand_to_thigh_m" in M:
        v = M["support_hand_to_thigh_m"]; out.append(_ck("support_hand_on_thigh_m", r(v.max(), 3), f"<= {Q['support_hand_thigh_max_m']} every frame", v.max() <= Q["support_hand_thigh_max_m"], "support_hand"))
    if "thigh_split_max_deg" in Q:
        v = M["thigh_split_deg"]; out.append(_ck("thigh_split_deg", r(np.median(v)), f"<= {Q['thigh_split_max_deg']} (median)", np.median(v) <= Q["thigh_split_max_deg"], "plant_legs"))
    series = {k: [round(float(x), 2) for x in M[k]] for k in ("neck_flex_deg", "neck_line_dev_deg", "head_roll_deg", "head_yaw_deg", "thigh_split_deg",
              "handle_vs_palm_deg", "handle_vs_forearm_perp_deg", "handle_vs_body_forward_deg") if k in M}
    for s in ("Left", "Right"): series[s + "_knee_deg"] = [round(float(x), 1) for x in M[s]["knee_deg"]]; series[s + "_toe_h_m"] = [round(float(x), 4) for x in M[s]["toe_h_m"]]
    return out, series

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else "exercises/db_row.yaml"))
    cs, _ = checks(sys.argv[1], cfg["qc"]["posture"])
    for c in cs: print(("PASS " if c["pass"] else "FAIL ") + c["name"], json.dumps(c["value"]), "limit", c["limit"])
    print("POSTURE", "PASS" if all(c["pass"] for c in cs) else "FAIL", f"{sum(c['pass'] for c in cs)}/{len(cs)}")
