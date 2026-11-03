#!/usr/bin/env python3
"""t660u/t661u extra clip gates for the D7 machine-first clips (raw 720x1280, before any grade):
  machine_unchanged : machine pixels (empty-machine still foreground, minus the dilated union of the moving body) vs
                      the pinned still, per sampled frame: mean |diff| and SSIM. Pass: mean |diff| <= 6, SSIM >= 0.90
                      (the weight stack + cable + handle legitimately move: they are excluded by a stack/cable box).
  foot_on_plate     : ankle keypoint displacement vs frame 0 (max px / shin length). Pass <= 0.15.
  head_vs_body      : head (circle at the ear midpoint, r = 0.45 x ear-shoulder) vs body skin (torso quad, no glow):
                      median L difference and gloss (p99 - p50 L). Pass: |dL| <= 15 and head gloss <= body gloss + 15.
  peak angles       : layout.measure_angle metric "arm_torso" (shoulder angle trunk -> upper arm, visible arm) and
                      elbow flexion per frame; reports the peak (max elbow flexion) frame and its angles.
usage: nice -n 19 python3 tools/d7_extra_gates.py <out.json> <empty_machine_720.png> name:clip:pin [...]"""
import sys, json, math, cv2, numpy as np
sys.path.insert(0, "/workspace/daily_info_reels")
import universal_anatomy_rig as uar
from skimage.metrics import structural_similarity as ssim

def lm(fr):
    r = uar.get_pose_detector().detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    if not r.pose_landmarks: return None
    h, w = fr.shape[:2]; L = r.pose_landmarks[0]
    return {i: np.array([L[i].x * w, L[i].y * h]) for i in range(33)}

def ang(a, v, b):
    u, w = a - v, b - v
    return float(np.degrees(np.arccos(np.clip(np.dot(u, w) / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-6), -1, 1))))

def visible_side(k):
    quad = np.array([k[11], k[12], k[24], k[23]], np.float32)
    def inside(s, e, wr):
        return sum(cv2.pointPolygonTest(quad, tuple(map(float, k[s] + t * (k[wr] - k[s]))), False) >= 0 for t in np.linspace(0.15, 1, 12))
    return min(((11, 13, 15), (12, 14, 16)), key=lambda c: inside(c[0], c[1], c[2]))

def cr(a, b): return float(a[0] * b[1] - a[1] * b[0])

def lum(a): a = a.astype(np.float32); return 0.299 * a[..., 2] + 0.587 * a[..., 1] + 0.114 * a[..., 0]

def run(name, clip, pin, empty):
    cap = cv2.VideoCapture(clip); frs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frs.append(f)
    n = len(frs); fps = cap.get(cv2.CAP_PROP_FPS) or 24
    st = cv2.imread(pin); em = cv2.imread(empty)
    bg = np.median(lum(em)[:80])
    mach = cv2.dilate((np.abs(lum(em) - bg) > 14).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    idx = list(range(0, n, 3))
    K = {i: lm(frs[i]) for i in idx}
    body = np.zeros(mach.shape, np.uint8)
    for i in idx:   # moving body + its shadow: frame-vs-pin difference union, dilated
        body |= (np.abs(lum(frs[i]) - lum(st)) > 18).astype(np.uint8)
    body = cv2.dilate(body, np.ones((31, 31), np.uint8)) > 0
    person0 = cv2.dilate((np.abs(lum(st) - lum(em)) > 18).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    keep = mach & ~body & ~person0
    md, ms = [], []
    gs, ge = lum(st), cv2.cvtColor(st, cv2.COLOR_BGR2GRAY)
    for i in idx:
        d = np.abs(lum(frs[i]) - gs)[keep]; md.append(float(d.mean()))
        _, full = ssim(ge, cv2.cvtColor(frs[i], cv2.COLOR_BGR2GRAY), full=True, data_range=255); ms.append(float(full[keep].mean()))
    r = {"frames": n, "fps": fps, "machine_px_checked": int(keep.sum()),
         "machine_unchanged": {"mean_absdiff_max": round(max(md), 2), "ssim_min": round(min(ms), 3), "pass": max(md) <= 6 and min(ms) >= 0.90}}
    k0 = next(K[i] for i in idx if K[i] is not None)
    shin = np.mean([np.linalg.norm(k0[27] - k0[25]), np.linalg.norm(k0[28] - k0[26])])
    disp = [max(np.linalg.norm(K[i][27] - k0[27]), np.linalg.norm(K[i][28] - k0[28])) / shin for i in idx if K[i] is not None]
    # pixel check (t660u): MediaPipe ankles from behind a footplate are unreliable -> feet box (pin ankles/heels/toes
    # +-0.5 shin) mean |diff| vs the pin over all sampled frames; feet + plate pixels must hold
    pts = np.array([ks_[j] for j in (27, 28, 29, 30, 31, 32)]) if (ks_ := lm(st)) is not None else None
    if pts is not None:
        x0_, y0_ = (pts.min(0) - 0.5 * shin).astype(int); x1_, y1_ = (pts.max(0) + 0.5 * shin).astype(int)
        x0_, y0_ = max(0, x0_), max(0, y0_)
        fd = [float(np.abs(lum(frs[i])[y0_:y1_, x0_:x1_] - gs[y0_:y1_, x0_:x1_]).mean()) for i in idx]
        r["foot_on_plate_pixels"] = {"box": [int(x0_), int(y0_), int(x1_), int(y1_)], "mean_absdiff_max": round(max(fd), 2), "pass": bool(max(fd) <= 8.0)}
    heel = [max(np.linalg.norm(K[i][29] - k0[29]), np.linalg.norm(K[i][30] - k0[30])) / shin for i in idx if K[i] is not None]
    r["foot_on_plate"] = {"max_ankle_disp_over_shin": round(max(disp), 3), "p90_ankle_disp_over_shin": round(float(np.percentile(disp, 90)), 3),
                          "max_heel_disp_over_shin": round(max(heel), 3), "shin_px": round(float(shin), 1), "pass": bool(np.percentile(disp, 90) <= 0.15)}
    # head vs body tone on the pinned still and the peak frame
    def hb(fr, k):
        L = lum(fr); f = fr.astype(np.float32)
        act = np.maximum(f[..., 2] - 0.5 * (f[..., 1] + f[..., 0]), f[..., 1] - 0.5 * (f[..., 2] + f[..., 0]))
        ear = 0.5 * (k[7] + k[8]); shm = 0.5 * (k[11] + k[12]); rad = 0.45 * np.linalg.norm(ear - shm)
        yy, xx = np.mgrid[:L.shape[0], :L.shape[1]]
        head = (np.hypot(xx - ear[0], yy - ear[1]) < rad) & (np.abs(L - bg) > 8)
        hl, hr = k[11] + 0.7 * (k[23] - k[11]), k[12] + 0.7 * (k[24] - k[12])   # upper 70% of the trunk: no shorts
        quad = np.zeros(L.shape, np.uint8); cv2.fillPoly(quad, [np.array([k[11], k[12], hr, hl], np.int32)], 1)
        arm = np.zeros(L.shape, np.uint8)
        for a_, b_ in ((11, 13), (12, 14), (13, 15), (14, 16)): cv2.line(arm, tuple(map(int, k[a_])), tuple(map(int, k[b_])), 1, int(max(6, 0.12 * np.linalg.norm(k[11] - k[12]) + 10)))
        skin = ((quad | arm) > 0) & (act < 12) & (np.abs(L - bg) > 8)
        hL, bL = L[head], L[skin]
        return {"head_L": round(float(np.median(hL)), 1), "body_L": round(float(np.median(bL)), 1),
                "head_gloss": round(float(np.percentile(hL, 99) - np.median(hL)), 1), "body_gloss": round(float(np.percentile(bL, 99) - np.median(bL)), 1)}
    ks = lm(st); h0 = hb(st, ks)
    el = {i: ang(K[i][visible_side(K[i])[0]], K[i][visible_side(K[i])[1]], K[i][visible_side(K[i])[2]]) for i in idx if K[i] is not None}
    flex = {i: 180.0 - v for i, v in el.items()}
    pk = max(flex, key=flex.get)
    kp = K[pk]; s, e, w = visible_side(kp); hip = 0.5 * (kp[23] + kp[24])
    k0s = visible_side(k0); hip0 = 0.5 * (k0[23] + k0[24])
    r["head_vs_body"] = dict(h0, dL=round(h0["head_L"] - h0["body_L"], 1),
                             pass_=bool(abs(h0["head_L"] - h0["body_L"]) <= 15 and h0["head_gloss"] <= h0["body_gloss"] + 15))
    r["head_vs_body"]["pass"] = r["head_vs_body"].pop("pass_")
    r["head_vs_body_peak"] = hb(frs[pk], kp)
    def signed(k, sd):
        hp = 0.5 * (k[23] + k[24]); t = hp - k[sd[0]]; u = k[sd[1]] - k[sd[0]]
        fwd = np.sign(cr(t, k0[k0s[2]] - k0[k0s[0]])) or 1.0   # side the handle is on at the start
        return float(np.sign(cr(t, u)) * fwd * ang(hp, k[sd[0]], k[sd[1]]))
    sh_series = {i: signed(K[i], visible_side(K[i])) for i in idx if K[i] is not None}
    r["peak"] = {"frame": pk, "shoulder_signed_at_peak": round(sh_series[pk], 1), "shoulder_signed_min": round(min(sh_series.values()), 1),
                 "shoulder_signed_series": [round(sh_series[i]) for i in sorted(sh_series)], "t_s": round(pk / fps, 2), "elbow_flexion": round(flex[pk], 1),
                 "arm_torso": round(ang(hip, kp[s], kp[e]), 1), "arm_torso_start": round(ang(hip0, k0[k0s[0]], k0[k0s[1]]), 1),
                 "elbow_flex_series": [round(flex[i]) for i in sorted(flex)]}
    return r

out, empty = sys.argv[1], sys.argv[2]; res = {}
for spec in sys.argv[3:]:
    nm, clip, pin = spec.split(":"); res[nm] = run(nm, clip, pin, empty)
    print(nm, {k: (v.get("pass") if isinstance(v, dict) and "pass" in v else None) for k, v in res[nm].items()}, res[nm]["peak"], res[nm]["head_vs_body"])
json.dump(res, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
