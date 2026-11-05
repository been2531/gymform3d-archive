#!/usr/bin/env python3
"""Generic proactive anatomy / readability audit (v6) - pure geometry, no AI. Limits in the exercise config `qc.anatomy`
(missing key = check skipped). Inputs from the render dir: bones.npz, hand.npz, checks.json (+ grade/look.json for
pixel checks, done in qc3d).
  joint ranges        signed flexion per hinge joint (elbow, knee: + = flexion as at rest; hyperextension = negative
                      past a tolerance), wrist deviation vs forearm, ankle (shin-foot) change vs rest
  self-intersection   capsules per bone (rest radius from the mesh), non-adjacent pairs (> 2 hops apart in the
                      hierarchy): penetration depth = r1 + r2 - segment distance; allowed contact pairs from config
  mesh spikes         deformed/rest edge length max (spike) and min (collapse), from checks.json
  balance             whole-body centre of mass (Dempster segment fractions + dumbbell) inside the feet polygon
  proportions         left/right segment lengths at rest, and every frame vs rest
  grip                fingers wrap the handle (wrap angle range), fingertip gap, thumb gap + opposition, no hand
                      vertex inside the handle (tools/rig3d/hand_grip.metrics)
  prop readability    dumbbell heads not hidden by the body (visible share) and not stacked on screen (centre distance
                      / head diameter), extremities (head top, fingers, toes, dumbbell heads) inside the frame margin"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_grip

def _u(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
def _ck(name, value, limit, ok, fix, group="anatomy"):
    return {"name": name, "group": group, "value": value, "limit": limit, "pass": bool(ok), "fix": None if ok else fix}
MASS = {"Head": 0.081, "Hips": 0.142, "Spine02": 0.12, "Spine01": 0.12, "Spine": 0.115, "LeftArm": 0.028, "RightArm": 0.028, "LeftForeArm": 0.016,
        "RightForeArm": 0.016, "LeftHand": 0.006, "RightHand": 0.006, "LeftUpLeg": 0.1, "RightUpLeg": 0.1, "LeftLeg": 0.0465, "RightLeg": 0.0465,
        "LeftFoot": 0.0145, "RightFoot": 0.0145}
CHILD = {"Hips": "Spine02", "Spine02": "Spine01", "Spine01": "Spine", "Spine": "neck", "Head": "head_end", "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand",
         "RightArm": "RightForeArm", "RightForeArm": "RightHand", "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "RightUpLeg": "RightLeg", "RightLeg": "RightFoot",
         "LeftFoot": "LeftToeBase", "RightFoot": "RightToeBase"}

def run(rdir, A, cfg):
    out = []; D = np.load(os.path.join(rdir, "bones.npz"), allow_pickle=True); names = [str(n) for n in D["names"]]; ix = {n: i for i, n in enumerate(names)}
    P, R0 = D["pose"], D["rest"]; H = lambda n: P[:, ix[n], :3, 3]; H0 = lambda n: R0[ix[n], :3, 3]; Rot = lambda n: P[:, ix[n], :3, :3]
    r = lambda v, k=1: round(float(v), k)
    # ---- joint ranges -------------------------------------------------------------------------------------------
    if "joint_ranges" in A:
        J = A["joint_ranges"]; bad = {}; vals = {}
        def hinge(a, b, c):
            p, q = H(b) - H(a), H(c) - H(b); p0, q0 = H0(b) - H0(a), H0(c) - H0(b); ax = Rot(a)[:, :, 0]            # parent bone X = hinge axis (the forearm's own X turns with the grip twist)
            ang = np.degrees(np.arctan2((np.cross(p, q) * ax).sum(1), (p * q).sum(1)))
            a0 = np.degrees(np.arctan2(np.cross(p0, q0) @ R0[ix[a], :3, 0], p0 @ q0)); sgn = 1.0 if a0 >= 0 else -1.0
            return sgn * ang                                                     # + = flexion (same sense as the rest bend)
        for s in ("Left", "Right"):
            for jn, (a, b, c) in (("elbow", (s + "Arm", s + "ForeArm", s + "Hand")), ("knee", (s + "UpLeg", s + "Leg", s + "Foot"))):
                fl = hinge(a, b, c); lo, hi = J[jn + "_flex_deg"]; vals[f"{s[0]}_{jn}"] = [r(fl.min()), r(fl.max())]
                if fl.min() < lo or fl.max() > hi: bad[f"{s[0]}_{jn}"] = vals[f"{s[0]}_{jn}"]
            wd = np.degrees(np.arccos(np.clip((Rot(s + "Hand")[:, :, 1] * _u(H(s + "Hand") - H(s + "ForeArm"))).sum(1), -1, 1)))
            vals[f"{s[0]}_wrist_dev"] = r(wd.max())
            if wd.max() > J["wrist_dev_max_deg"]: bad[f"{s[0]}_wrist_dev"] = r(wd.max())
            sv, fv = _u(H(s + "Foot") - H(s + "Leg")), _u(H(s + "ToeBase") - H(s + "Foot")); sv0, fv0 = _u(H0(s + "Foot") - H0(s + "Leg")), _u(H0(s + "ToeBase") - H0(s + "Foot"))
            ad = np.abs(np.degrees(np.arccos(np.clip((sv * fv).sum(1), -1, 1))) - np.degrees(np.arccos(np.clip(sv0 @ fv0, -1, 1))))
            vals[f"{s[0]}_ankle_dev"] = r(ad.max())
            if ad.max() > J["ankle_dev_max_deg"]: bad[f"{s[0]}_ankle_dev"] = r(ad.max())
        out.append(_ck("joint_ranges_no_hyperextension", bad or "ok", J, not bad, "joint_range")); out[-1]["values"] = vals
    # ---- self-intersection (capsules) ---------------------------------------------------------------------------
    if "capsule_penetration_max_m" in A and "capsule_names" in D.files:
        cr = dict(zip([str(n) for n in D["capsule_names"]], D["capsule_r"].astype(float))); par = {}
        segs = [b for b in cr if b in CHILD and CHILD[b] in ix]
        tree = {"Hips": None, "Spine02": "Hips", "Spine01": "Spine02", "Spine": "Spine01", "neck": "Spine", "Head": "neck", "LeftShoulder": "Spine", "RightShoulder": "Spine",
                "LeftArm": "LeftShoulder", "RightArm": "RightShoulder", "LeftForeArm": "LeftArm", "RightForeArm": "RightArm", "LeftHand": "LeftForeArm", "RightHand": "RightForeArm",
                "LeftUpLeg": "Hips", "RightUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "RightLeg": "RightUpLeg", "LeftFoot": "LeftLeg", "RightFoot": "RightLeg"}
        def chain(b):
            c = []
            while b: c.append(b); b = tree.get(b)
            return c
        def hops(a, b):
            ca, cb = chain(a), chain(b); com = next((x for x in ca if x in cb), None)
            return (ca.index(com) + cb.index(com)) if com else 99
        allow = {tuple(sorted(p)) for p in A.get("capsule_allowed_pairs", [])}
        def segdist(p1, q1, p2, q2):             # closest distance between segments, vectorised over frames (sampled)
            ts = np.linspace(0, 1, 9); a_ = p1[:, None] + (q1 - p1)[:, None] * ts[None, :, None]; b_ = p2[:, None] + (q2 - p2)[:, None] * ts[None, :, None]
            return np.linalg.norm(a_[:, :, None] - b_[:, None, :], axis=-1).min(axis=(1, 2))
        worst = {}
        for i, a in enumerate(segs):
            for b in segs[i + 1:]:
                if hops(a, b) <= 2 or tuple(sorted((a, b))) in allow: continue
                d = segdist(H(a), H(CHILD[a]), H(b), H(CHILD[b])); pen = cr[a] + cr[b] - d
                if pen.max() > 0: worst[f"{a}/{b}"] = round(float(pen.max()), 3)
        mx = max(worst.values()) if worst else 0.0; top = dict(sorted(worst.items(), key=lambda kv: -kv[1])[:4])
        out.append(_ck("self_intersection_capsule_m", {"max": mx, "pairs": top} if top else 0.0, f"<= {A['capsule_penetration_max_m']}", mx <= A["capsule_penetration_max_m"], "limb_clearance"))
    # ---- balance ---------------------------------------------------------------------------------------------------
    if "com_margin_min_m" in A:
        tot = np.zeros((P.shape[0], 3)); m = 0.0
        for b, w in MASS.items():
            if b in ix and CHILD.get(b, None) in ix: tot += w * (H(b) + H(CHILD[b])) / 2; m += w
            elif b in ix: tot += w * H(b); m += w
        dbm = float(A.get("dumbbell_mass_frac", 0.15))
        if "dumbbell" in D.files: tot += dbm * D["dumbbell"][:, :3, 3]; m += dbm
        com = tot / m
        def foot_pts(s):
            an, to = H(s + "Foot"), H(s + "ToeBase"); fw = _u(to - an); fw[:, 2] = 0; fw = _u(fw); lt = np.cross(fw, [0, 0, 1.0])
            pts = [an - 0.06 * fw, to + 0.05 * fw]; return [p + sgn * 0.045 * lt for p in pts for sgn in (-1, 1)]
        margins = []
        for f in range(P.shape[0]):
            pts = np.array([p[f, :2] for s in ("Left", "Right") for p in foot_pts(s)])
            hull = _hull(pts); margins.append(_inside_margin(hull, com[f, :2]))
        mg = float(min(margins)); out.append(_ck("center_of_mass_over_feet_m", round(mg, 3), f">= {A['com_margin_min_m']} (inside the feet polygon)", mg >= A["com_margin_min_m"], "com_balance"))
    # ---- proportions -----------------------------------------------------------------------------------------------
    if "proportion_tol" in A:
        bad = {}
        for s in ("Arm", "ForeArm", "UpLeg", "Leg"):
            l0, r0_ = np.linalg.norm(H0("Left" + s) - H0(CHILD["Left" + s])), np.linalg.norm(H0("Right" + s) - H0(CHILD["Right" + s]))
            if abs(l0 / r0_ - 1) > A["proportion_tol"]: bad[s + "_LR"] = round(float(l0 / r0_), 3)
            for side in ("Left", "Right"):
                lf = np.linalg.norm(H(side + s) - H(CHILD[side + s]), axis=1); l0_ = np.linalg.norm(H0(side + s) - H0(CHILD[side + s]))
                if np.abs(lf / l0_ - 1).max() > A["proportion_tol"]: bad[side + s + "_vs_rest"] = round(float(np.abs(lf / l0_ - 1).max()), 3)
        out.append(_ck("proportions_symmetric_and_rigid", bad or "ok", f"L/R and frame/rest within {A['proportion_tol']:.0%}", not bad, None))
    # ---- grip ------------------------------------------------------------------------------------------------------
    hz = os.path.join(rdir, "hand.npz")
    if "grip" in A:
        G = A["grip"]
        if not os.path.exists(hz): out.append(_ck("grip_present", False, True, False, "grip_wrap"))
        else:
            Hn = np.load(hz, allow_pickle=True); fr = list(Hn["frames"]); M = []
            ax_l = np.array(D["handle_axis_local"]) if "handle_axis_local" in D.files else np.array([1.0, 0, 0])
            for i, f in enumerate(fr):
                Md = D["dumbbell"][f - 1]; M.append(hand_grip.metrics(Hn["P"][i].astype(float), Hn["labels"], Hn["s_finger"], Md[:3, 3], Md[:3, :3] @ ax_l, float(Hn["handle_r"])))
            wr = np.array([m_["finger_wrap_deg"] for m_ in M]); gp = np.array([m_["fingertip_gap_mm"] for m_ in M])
            lo, hi = G["finger_wrap_deg"]
            out.append(_ck("grip_finger_wrap_deg", {"min": r(wr.min()), "max": r(wr.max())}, [lo, hi], lo <= wr.min() and wr.max() <= hi, "grip_wrap", "grip"))
            out.append(_ck("grip_fingertip_gap_mm", {"min": r(gp.min()), "max": r(gp.max())}, f"[{G['tip_gap_mm'][0]}, {G['tip_gap_mm'][1]}]", G["tip_gap_mm"][0] <= gp.min() and gp.max() <= G["tip_gap_mm"][1], "grip_wrap", "grip"))
            tg = np.array([m_.get("thumb_gap_mm", 99) for m_ in M]); to_ = np.array([m_.get("thumb_opposition_deg", 0) for m_ in M])
            out.append(_ck("grip_thumb_opposing", {"gap_mm_max": r(tg.max()), "opposition_deg_min": r(to_.min())}, f"gap <= {G['thumb_gap_max_mm']} mm, opposition >= {G['thumb_opposition_min_deg']}",
                           tg.max() <= G["thumb_gap_max_mm"] and tg.min() >= -1.0 and to_.min() >= G["thumb_opposition_min_deg"], "grip_wrap", "grip"))
            iv = max(m_["handle_intersect_verts"] for m_ in M)
            out.append(_ck("grip_no_hand_handle_intersection", iv, 0, iv == 0, "grip_wrap", "grip"))
    # ---- mesh spikes (checks.json) + prop readability + extremities -----------------------------------------------
    C = json.load(open(os.path.join(rdir, "checks.json")))["frames"]; T = [f for f in C if "tear_max_ratio" in f]
    if "edge_ratio_max" in A:
        vk = "tear_max_visible" if all("tear_max_visible" in f for f in T) else "tear_max_ratio"   # edges under the shorts are hidden
        mx = max(f[vk] for f in T); mn = min(f.get("tear_min_ratio", 1.0) for f in T) if any("tear_min_ratio" in f for f in T) else None
        ok = mx <= A["edge_ratio_max"] and (mn is None or mn >= A["edge_ratio_min"])
        out.append(_ck("mesh_no_spikes_edge_ratio", {"max": r(mx, 2), "min": None if mn is None else r(mn, 3), "measured": vk, "worst_groups": max(T, key=lambda f: f[vk]).get("spike_max_by_group")}, f"max <= {A['edge_ratio_max']}, min >= {A['edge_ratio_min']}", ok, "spine_spread"))
    V = [f for f in T if "db_head_visible" in f]
    if "db_head_visible_min" in A:
        if not V: out.append(_ck("prop_dumbbell_heads_readable", None, "needs v6 checks", False, "camera_yaw"))
        else:
            vm = min(min(f["db_head_visible"]) for f in V); sp = min(f["db_head_sep_rel"] for f in V)
            out.append(_ck("prop_dumbbell_heads_readable", {"visible_min": r(vm, 2), "screen_sep_min": r(sp, 2)}, f"visible >= {A['db_head_visible_min']}, separation >= {A['db_head_sep_min']} head diameters",
                           vm >= A["db_head_visible_min"] and sp >= A["db_head_sep_min"], "camera_yaw"))
    if "extremity_margin_px" in A:
        Wd, Hd = cfg["grade"]["out_res"]; E = [f["extrem2d"] for f in C if "extrem2d" in f]
        if not E: out.append(_ck("extremities_in_frame_margin_px", None, "needs v6 checks", False, "camera_margin"))
        else:
            mg = min(min(p[0], Wd - p[0], p[1], Hd - p[1]) for e in E for p in e.values())
            out.append(_ck("extremities_in_frame_margin_px", r(mg, 0), f">= {A['extremity_margin_px']}", mg >= A["extremity_margin_px"], "camera_margin"))
    return out

def _hull(p):
    p = sorted(map(tuple, p)); 
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0: lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0: up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])

def _inside_margin(h, c):
    """signed distance of c to the convex polygon h (CCW): + inside."""
    d = []
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]; e = b - a; nrm = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-12)   # outward for CCW
        d.append(-float((c - a) @ nrm))
    return min(d)
