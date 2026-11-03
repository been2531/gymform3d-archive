#!/usr/bin/env python3
"""Raw-clip drift gate G1-G10 (docs/mission/2D_CONSISTENCY_PLAN.md §3d), low-res, for the Day 7 smoke test.
usage: nice -n 19 python3 tools/source_gate_smoke.py <out.json> name:clip:start:keyframe:key_t [...]
Clips of one reel are given together (G7/G8 between-clip checks)."""
import sys, json, math
import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim
sys.path.insert(0, "/workspace/daily_info_reels/tools"); sys.path.insert(0, "/workspace/daily_info_reels")
import consistency_metrics as cm
import startframe_gates as sg
from gymform import motion_check as mc

W, H = cm.W, cm.H
def rs(im): return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
def gray(im): return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)

def hold(ref, fr):
    """tone-normalised SSIM + edge IoU (both at 360x640)."""
    a, b = gray(ref), gray(fr)
    bn = (b - b.mean()) / max(b.std(), 1e-6) * a.std() + a.mean()
    s = float(ssim(a, bn, data_range=255))
    ea = cv2.dilate(cv2.Canny(a.astype(np.uint8), 20, 60), np.ones((3, 3))) > 0
    eb = cv2.dilate(cv2.Canny(bn.clip(0, 255).astype(np.uint8), 20, 60), np.ones((3, 3))) > 0
    return round(s, 3), round(float((ea & eb).sum() / max((ea | eb).sum(), 1)), 3)

def glow(m): return (m.get("glow_red_frac") or 0) + (m.get("glow_green_frac") or 0)

def read_all(path):
    cap = cv2.VideoCapture(path); frs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frs.append(rs(f))
    return frs, cap.get(cv2.CAP_PROP_FPS)

KP = {"sh": (12, 11), "elb": (14, 13), "wr": (16, 15), "hip": (24, 23), "kn": (26, 25), "ank": (28, 27)}
def kps(fr):
    res = cm.detector().detect(cm.mp.Image(image_format=cm.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    if not res.pose_landmarks: return None
    lm = res.pose_landmarks[0]; P = lambda i: (lm[i].x * W, lm[i].y * H)
    d = {f"{n}_{s}": P(i) for n, (ir, il) in KP.items() for s, i in (("r", ir), ("l", il))}
    d["_ear"] = (P(8), P(7)); return d

def clip_gate(name, clip, start, key, key_t):
    frs, fps = read_all(clip); n = len(frs)
    st = rs(cv2.imread(start)); ky = rs(cv2.imread(key)); ki = min(n - 1, int(round(key_t * fps)))
    idx = sorted(set([int(round(i * (n - 1) / 12)) for i in range(13)] + [ki]))
    per = {i: cm.frame_metrics(frs[i]) for i in idx}
    ms_start, _ = cm.frame_metrics(st); ms_key, _ = cm.frame_metrics(ky)
    union = np.zeros((H, W), bool)
    for i in idx:
        if per[i][1] is not None: union |= per[i][1]
    union = cv2.dilate(union.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    g0 = gray(frs[0]); bgss = []
    for i in idx[1:]:
        _, full = ssim(g0, gray(frs[i]), full=True, data_range=255); bgss.append(float(full[~union].mean()))
    M = [per[i][0] for i in idx]
    r = {"clip": clip, "frames": n, "fps": fps, "dur_s": round(n / fps, 2), "key_frame_idx": ki, "sampled": idx}
    # G1 start hold, G2 end + keyframe hold
    s1 = hold(st, frs[0]); s2 = hold(st, frs[-1]); sk = hold(ky, frs[ki])
    # best keyframe match within +-6 frames (timing tolerance)
    skb = max((hold(ky, frs[j]) for j in range(max(0, ki - 6), min(n, ki + 7))), key=lambda t: t[0] + t[1])
    r["G1"] = {"ssim": s1[0], "edge_iou": s1[1], "pass": s1[0] >= 0.97 and s1[1] >= 0.88}
    r["G2"] = {"end_ssim": s2[0], "end_edge_iou": s2[1], "key_ssim": sk[0], "key_edge_iou": sk[1],
               "key_best_pm6": list(skb),
               "pass": s2[0] >= 0.97 and s2[1] >= 0.88 and skb[0] >= 0.97 and skb[1] >= 0.88}
    gs = [glow(m) for m in M]; g0s = glow(ms_start); gk = glow(ms_key)
    arr = np.array(gs); cv = float(arr.std() / arr.mean() * 100) if arr.mean() > 0 else 0
    if g0s >= 0.02:
        ok3 = all(0.6 * g0s <= g <= 1.6 * g0s for g in gs) and cv <= 30
    else:
        ok3 = (max(gs) - min(gs)) <= 0.015
    r["G3"] = {"start_glow_pct": round(g0s * 100, 2), "keyframe_glow_pct": round(gk * 100, 2),
               "clip_min_pct": round(min(gs) * 100, 2), "clip_max_pct": round(max(gs) * 100, 2),
               "ratio_min": round(min(gs) / max(g0s, 1e-6), 2), "ratio_max": round(max(gs) / max(g0s, 1e-6), 2),
               "cv_pct": round(cv, 1), "series_pct": [round(g * 100, 1) for g in gs], "pass": bool(ok3)}
    off = max(m.get("glow_off_body_frac") or 0 for m in M)
    r["G4"] = {"max_off_body_pct": round(off * 100, 3), "pass": off <= 0.005}
    r["G5"] = {"bg_ssim_min": round(min(bgss), 3), "pass": min(bgss) >= 0.90}
    bl = [float(np.mean(m["bg_rgb"])) for m in M]
    r["G6"] = {"bg_lum_drift": round(max(bl) - min(bl), 1), "bg_rgb_f0": M[0]["bg_rgb"], "bg_rgb_start": ms_start["bg_rgb"], "pass": max(bl) - min(bl) <= 4}
    vr = [m.get("view_ratio") for m in M if m.get("view_ratio")]
    r["view_mean"] = round(float(np.mean(vr)), 3); r["view_start"] = ms_start.get("view_ratio")
    cf = np.array([m.get("cobalt_frac") or 0 for m in M]); crgb = [m["cobalt_rgb"] for m in M if m.get("cobalt_rgb")]
    ccv = float(cf.std() / cf.mean() * 100) if cf.mean() > 0 else 999
    cmed = np.median(crgb, 0) if crgb else None
    cd = float(np.linalg.norm(cmed - np.array(ms_start["cobalt_rgb"]))) if (cmed is not None and ms_start.get("cobalt_rgb")) else None
    r["G8_clip"] = {"cobalt_frac_mean": round(float(cf.mean()), 4), "cobalt_cv_pct": round(ccv, 1),
                    "cobalt_rgb": cmed.round(0).tolist() if cmed is not None else None, "start_cobalt_rgb": ms_start.get("cobalt_rgb"),
                    "cobalt_rgb_dist_vs_start": round(cd, 1) if cd is not None else None}
    # G9 motion: keypoints on 25 frames
    kidx = [int(round(i * (n - 1) / 24)) for i in range(25)]
    K = [k for k in (kps(frs[i]) for i in kidx) if k]
    def elbow(k): return mc._agg(mc.joint_angles(k)["elbow_flexion"], "mean")
    el = [elbow(k) for k in K]; el = [x for x in el if x == x]
    tr = [mc.joint_angles(k)["trunk_from_vertical"] for k in K]; tr = [x for x in tr if x == x]
    def shrug(k):
        (er, el_), sr, sl = k["_ear"], k["sh_r"], k["sh_l"]
        tl = math.dist(((sr[0]+sl[0])/2, (sr[1]+sl[1])/2), ((k["hip_r"][0]+k["hip_l"][0])/2, (k["hip_r"][1]+k["hip_l"][1])/2))
        return ((sr[1] - er[1]) + (sl[1] - el_[1])) / 2 / max(tl, 1)
    sh = [shrug(k) for k in K]
    def flare(k):  # elbow lateral distance from shoulder line, / shoulder width
        sw = max(abs(k["sh_r"][0] - k["sh_l"][0]), 1)
        return (abs(k["elb_r"][0] - k["hip_r"][0]) + abs(k["elb_l"][0] - k["hip_l"][0])) / 2 / sw
    fl = [flare(k) for k in K]
    def wr_y(k): return (k["wr_r"][1] + k["wr_l"][1]) / 2 / H
    wy = [wr_y(k) for k in K]
    r["motion"] = {"kp_frames": len(K), "elbow_flex_min": round(min(el), 1) if el else None, "elbow_flex_max": round(max(el), 1) if el else None,
                   "trunk_from_vertical_range": round(max(tr) - min(tr), 1) if tr else None, "trunk_max": round(max(tr), 1) if tr else None,
                   "ear_to_shoulder_norm_start_vs_min": [round(sh[0], 3), round(min(sh), 3)],
                   "elbow_flare_start_vs_max": [round(fl[0], 3), round(max(fl), 3)],
                   "elbow_series": [round(x) for x in el]}
    std = mc.load_standards()["exercises"]["seated_cable_row"] if "exercises" in mc.load_standards() else None
    r["_K"] = K
    # G10 separation at f0 (startframe_gates on the upscaled frame 0)
    cap = cv2.VideoCapture(clip); ok, f0 = cap.read(); cv2.imwrite("/workspace/daily_info_reels/_tmp_gate_f0.png", f0)
    g = sg.gates("/workspace/daily_info_reels/_tmp_gate_f0.png")
    import os; os.remove("/workspace/daily_info_reels/_tmp_gate_f0.png")
    r["G10"] = {"fig_minus_bg": g.get("fig"), "plate_vs_bg_p05": g.get("plate_vs_bg_p05"), "bench_vs_bg": g.get("bench_vs_bg"),
                "bench_vs_skin": g.get("bench_vs_skin"), "startframe_gates_pass": g.get("pass"),
                "pass": (g.get("fig") or 0) >= 33}
    r["skin_series"] = [m.get("skin_L_p50") for m in M]; r["skin_start"] = ms_start.get("skin_L_p50")
    return r

def main():
    out = sys.argv[1]; res = {}
    for spec in sys.argv[2:]:
        name, clip, start, key, kt = spec.split(":"); res[name] = clip_gate(name, clip, start, key, float(kt))
    names = list(res)
    std = mc.load_standards(); ex = (std.get("exercises") or std)["seated_cable_row"]
    for nm in names:
        K = res[nm].pop("_K")
        if "wrong" in nm:
            items, summ = mc.check_wrong_step(ex, K, view="3/4")
            res[nm]["G9"] = {"kind": "wrong", "fired": summ["value"], "pass": summ["pass"], "severity": summ["severity"],
                             "items": [(i["id"], i.get("value"), i.get("expected"), i.get("pass")) for i in items]}
        else:
            items = mc.check_right_step(ex, K, view="3/4")
            res[nm]["G9"] = {"kind": "right", "items": [(i["id"], None if i.get("value") is None else round(i["value"], 1), i.get("expected"), i.get("pass"), i.get("severity")) for i in items],
                             "pass": all(i.get("pass") is not False for i in items if i.get("severity") == "fail")}
    if len(names) >= 2:
        v = [res[n]["view_mean"] for n in names]; cfs = [res[n]["G8_clip"]["cobalt_frac_mean"] for n in names]
        rel = (max(cfs) - min(cfs)) / np.mean(cfs) * 100 if np.mean(cfs) else 999
        between = {"view_spread": round(max(v) - min(v), 3), "cobalt_frac_rel_spread_pct": round(rel, 1)}
        for n in names:
            c = res[n]["G8_clip"]
            res[n]["G7"] = {"view_spread_between_clips": between["view_spread"], "view_mean": res[n]["view_mean"], "view_start": res[n]["view_start"], "pass": between["view_spread"] <= 0.05}
            res[n]["G8"] = dict(c, between_rel_pct=between["cobalt_frac_rel_spread_pct"],
                                pass_=None)
            res[n]["G8"]["pass"] = bool(c["cobalt_cv_pct"] <= 10 and rel <= 20 and (c["cobalt_rgb_dist_vs_start"] is not None and c["cobalt_rgb_dist_vs_start"] <= 20))
            del res[n]["G8"]["pass_"]
            del res[n]["G8_clip"]
    for n in names:
        res[n]["G1_G8_all_pass"] = all(res[n][f"G{i}"]["pass"] for i in range(1, 9))
    json.dump(res, open(out, "w"), indent=1, default=float)
    for n in names:
        print(n, {k: res[n][k]["pass"] for k in sorted(res[n]) if k.startswith("G") and isinstance(res[n][k], dict)})

if __name__ == "__main__":
    main()
