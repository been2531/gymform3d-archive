#!/usr/bin/env python3
"""2D source-clip consistency metrics (read-only, low-res). No generation, no rendering.

For each raw clip: sample N frames, downscale, and measure
  backdrop RGB (top strip + 4 corners), backdrop corner span,
  person silhouette (MediaPipe segmentation), mannequin tone (torso-polygon median L, glow excluded),
  upper-arm / forearm width normalised by torso length, glow fraction of the body, cobalt (palette A) pixels,
  framing (hip centre, body height fraction), camera-view proxy (shoulder width / torso length),
  SSIM vs frame 0 (whole frame and backdrop-only), and frame-0 SSIM between clips of a day.
Usage: nice -n 19 python3 tools/consistency_metrics.py  -> output/_consistency/metrics.json
"""
import json, os, sys, math
import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = "/workspace/daily_info_reels"
OUT = f"{ROOT}/output/_consistency"
N = 9            # sampled frames per clip
W, H = 360, 640  # analysis size (low-res)
BG_REF = np.array([35, 39, 43], float)  # graded target (RGB)

CLIPS = {
 "D4": [("s1 i2v (base)", "output/day4_rdl_step1_i2v.mp4", 0),
        ("s1 locked", "output/day4_rdl_step1_locked_720p.mp4", 1),
        ("s2 ecorche", "output/day4_rdl_step2_ecorche_720p.mp4", 1),
        ("s3=s1 ecorche", "output/day4_rdl_step1_ecorche_720p.mp4", 1),
        ("s4 locked", "output/day4_rdl_step4_locked_720p.mp4", 1),
        ("s3 ecorche (unused)", "output/day4_rdl_step3_ecorche_720p.mp4", 0),
        ("s4 ecorche (unused)", "output/day4_rdl_step4_ecorche_720p.mp4", 0)],
 "D5": [("s1 bone", "output/assets/day5_lat_step1_bone_720p.mp4", 1),
        ("s2 muscle", "output/assets/day5_lat_step2_muscle_720p.mp4", 1),
        ("s3 bone v3", "output/assets/day5_lat_step3_bone_v3_720p.mp4", 1),
        ("s4 muscle", "output/assets/day5_lat_step4_muscle_720p.mp4", 1),
        ("s3 bone v1", "output/assets/day5_lat_step3_bone_720p.mp4", 0),
        ("s3 bone v2", "output/assets/day5_lat_step3_bone_v2_720p.mp4", 0)],
 "D6": [("v5 wrong", "output/assets/day6_v5_wrong_raw.mp4", 1),
        ("v5 right", "output/assets/day6_v5_right_raw.mp4", 1)],
 "D7": [("v3 wrong", "output/assets/day7_wrong_raw.mp4", 1),
        ("v3 right", "output/assets/day7_right_raw.mp4", 1),
        ("v1 wrong", "output/assets/day7_wrong_raw_v1.mp4", 0),
        ("v1 right", "output/assets/day7_right_raw_v1.mp4", 0),
        ("v2 wrong", "output/assets/day7_wrong_raw_v2.mp4", 0),
        ("v2 right", "output/assets/day7_right_raw_v2.mp4", 0)],
}

_det = None
def detector():
    global _det
    if _det is None:
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=f"{ROOT}/pose_landmarker_heavy.task"),
            running_mode=mp_vision.RunningMode.IMAGE, output_segmentation_masks=True, num_poses=1)
        _det = mp_vision.PoseLandmarker.create_from_options(opts)
    return _det

def read_samples(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    idx = [int(round(i * (n - 1) / (N - 1))) for i in range(N)]
    frs = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok: continue
        frs.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    return frs, idx, n, fps

def width_at(mask, a, b, t=0.5):
    """Silhouette cross-section width perpendicular to segment a->b at fraction t (px)."""
    a = np.array(a, float); b = np.array(b, float); d = b - a; L = np.linalg.norm(d)
    if L < 4: return None
    u = d / L; p = a + t * d; nrm = np.array([-u[1], u[0]])
    w = 0
    for s in (1, -1):
        k = 0
        while k < 120:
            q = p + s * (k + 1) * nrm; x, y = int(round(q[0])), int(round(q[1]))
            if not (0 <= x < W and 0 <= y < H) or mask[y, x] == 0: break
            k += 1
        w += k
    return w + 1

def frame_metrics(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    m = {}
    f = rgb.astype(float); r, g, b = f[..., 0], f[..., 1], f[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # backdrop: top strip rows 3-9 %, plus corner medians
    top = rgb[int(.03 * H):int(.09 * H), int(.1 * W):int(.9 * W)].reshape(-1, 3)
    m["bg_rgb"] = np.median(top, 0).round(1).tolist()
    cs = int(.07 * W)
    corners = [rgb[int(.03*H):int(.03*H)+cs, :cs], rgb[int(.03*H):int(.03*H)+cs, -cs:],
               rgb[int(.6*H):int(.6*H)+cs, :cs], rgb[int(.6*H):int(.6*H)+cs, -cs:]]
    cl = [float(np.median(0.299*c[...,0]+0.587*c[...,1]+0.114*c[...,2])) for c in corners]
    m["bg_corner_span"] = round(max(cl) - min(cl), 1)
    if not res.pose_landmarks:
        m["pose"] = False; return m, None
    m["pose"] = True
    seg = res.segmentation_masks[0].numpy_view()
    seg = (seg[..., 0] if seg.ndim == 3 else seg) > 0.5
    seg_u8 = seg.astype(np.uint8)
    lm = res.pose_landmarks[0]
    P = lambda i: (lm[i].x * W, lm[i].y * H)
    ls, rs, lh, rh = P(11), P(12), P(23), P(24)
    sm = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2); hm = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
    torso = math.dist(sm, hm)
    m["torso_px"] = round(torso, 1)
    m["view_ratio"] = round(math.dist(ls, rs) / max(torso, 1), 3)   # shoulder width / torso length
    area = int(seg.sum())
    m["sil_area_frac"] = round(area / (W * H), 4)
    m["sil_area_norm"] = round(area / max(torso, 1) ** 2, 3)       # body bulk, zoom-independent
    ys, xs = np.nonzero(seg)
    if len(ys):
        m["body_h_frac"] = round((ys.max() - ys.min()) / H, 3)
        m["body_cx"] = round(float(xs.mean()) / W, 3); m["body_cy"] = round(float(ys.mean()) / H, 3)
    m["hip_xy"] = [round(hm[0] / W, 3), round(hm[1] / H, 3)]
    # mannequin tone: torso polygon, glow-free pixels
    poly = np.array([ls, rs, rh, lh], np.int32)
    tm = np.zeros((H, W), np.uint8); cv2.fillPoly(tm, [poly], 1)
    chroma = f.max(-1) - f.min(-1)
    sel = (tm > 0) & seg & (chroma < 18)
    m["skin_L_p50"] = round(float(np.median(lum[sel])), 1) if sel.sum() > 50 else None
    m["skin_L_p90"] = round(float(np.percentile(lum[sel], 90)), 1) if sel.sum() > 50 else None
    # limb widths / torso
    ws = []
    for (a, bb) in ((11, 13), (12, 14)):
        if min(lm[a].visibility, lm[bb].visibility) > 0.5:
            w = width_at(seg_u8, P(a), P(bb)); ws.append(w) if w else None
    m["upperarm_w_norm"] = round(float(np.median(ws)) / torso, 3) if ws and torso > 10 else None
    ws = []
    for (a, bb) in ((13, 15), (14, 16)):
        if min(lm[a].visibility, lm[bb].visibility) > 0.5:
            w = width_at(seg_u8, P(a), P(bb)); ws.append(w) if w else None
    m["forearm_w_norm"] = round(float(np.median(ws)) / torso, 3) if ws and torso > 10 else None
    # glow on body
    red = (r - np.maximum(g, b) > 25) & (r > 70); grn = (g - np.maximum(r, b) > 20) & (g > 60)
    m["glow_red_frac"] = round(float((red & seg).sum()) / max(area, 1), 4)
    m["glow_green_frac"] = round(float((grn & seg).sum()) / max(area, 1), 4)
    m["glow_off_body_frac"] = round(float(((red | grn) & ~seg).sum()) / (W * H), 4)
    cob = (b - np.maximum(r, g) > 20) & (b > 60)
    m["cobalt_frac"] = round(float(cob.sum()) / (W * H), 4)
    m["cobalt_rgb"] = np.median(rgb[cob], 0).round(0).tolist() if cob.sum() > 30 else None
    return m, seg

def agg(vals):
    v = [x for x in vals if x is not None]
    if not v: return None
    a = np.array(v, float)
    return {"mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "cv_pct": round(float(a.std() / abs(a.mean()) * 100), 1) if a.mean() else None}

def clip_metrics(path):
    frs, idx, n, fps = read_samples(f"{ROOT}/{path}")
    per, segs = [], []
    for fr in frs:
        m, s = frame_metrics(fr); per.append(m); segs.append(s)
    g0 = cv2.cvtColor(frs[0], cv2.COLOR_BGR2GRAY)
    union = np.zeros((H, W), bool)
    for s in segs:
        if s is not None: union |= s
    union = cv2.dilate(union.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    ss, ssb = [], []
    for fr in frs[1:]:
        gi = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        v, full = ssim(g0, gi, full=True, data_range=255)
        ss.append(v); ssb.append(float(full[~union].mean()) if (~union).sum() > 1000 else None)
    keys = ["skin_L_p50", "skin_L_p90", "upperarm_w_norm", "forearm_w_norm", "sil_area_norm", "view_ratio",
            "body_h_frac", "body_cx", "body_cy", "glow_red_frac", "glow_green_frac", "glow_off_body_frac",
            "cobalt_frac", "bg_corner_span", "torso_px"]
    out = {"path": path, "frames": n, "fps": round(fps, 2), "dur_s": round(n / fps, 2), "sampled": idx,
           "pose_frames": sum(1 for p in per if p.get("pose")),
           "bg_rgb_mean": np.mean([p["bg_rgb"] for p in per], 0).round(1).tolist(),
           "bg_dist_to_35_39_43": round(float(np.linalg.norm(np.mean([p["bg_rgb"] for p in per], 0) - BG_REF)), 1),
           "bg_rgb_drift": round(float(np.ptp([np.mean(p["bg_rgb"]) for p in per])), 1),
           "ssim_vs_f0": agg(ss), "ssim_bg_vs_f0": agg(ssb), "per_frame": per}
    for k in keys: out[k] = agg([p.get(k) for p in per])
    hips = [p["hip_xy"] for p in per if p.get("hip_xy")]
    if hips:
        h = np.array(hips); out["hip_drift_px720"] = round(float(np.ptp(h[:, 0]) * 720), 1), round(float(np.ptp(h[:, 1]) * 1280), 1)
    cobs = [p["cobalt_rgb"] for p in per if p.get("cobalt_rgb")]
    out["cobalt_rgb"] = np.median(cobs, 0).round(0).tolist() if cobs else None
    return out, frs

def main():
    res = {}
    f0 = {}
    for day, lst in CLIPS.items():
        res[day] = {}
        for name, path, used in lst:
            if not os.path.exists(f"{ROOT}/{path}"): continue
            print(day, name, flush=True)
            m, frs = clip_metrics(path); m["used_in_master"] = bool(used)
            res[day][name] = m
            f0[(day, name)] = cv2.cvtColor(frs[0], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{OUT}/f_{day}_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '-')}.jpg",
                        np.hstack([frs[0], frs[len(frs)//2], frs[-1]]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # between clips (used in master): frame-0 SSIM pairs and spread of clip means
    between = {}
    for day, lst in CLIPS.items():
        used = [n for n, p, u in lst if u and n in res[day]]
        pairs = []
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                pairs.append(round(float(ssim(f0[(day, used[i])], f0[(day, used[j])], data_range=255)), 3))
        def spread(k):
            v = [res[day][n][k]["mean"] for n in used if res[day][n].get(k)]
            return round(max(v) - min(v), 3) if len(v) > 1 else None
        def rel(k):
            v = [res[day][n][k]["mean"] for n in used if res[day][n].get(k)]
            return round((max(v) - min(v)) / np.mean(v) * 100, 1) if len(v) > 1 and np.mean(v) else None
        between[day] = {"clips": used, "f0_ssim_pairs": pairs,
                        "f0_ssim_min": min(pairs) if pairs else None,
                        "skin_L_p50_spread": spread("skin_L_p50"),
                        "upperarm_w_rel_spread_pct": rel("upperarm_w_norm"),
                        "forearm_w_rel_spread_pct": rel("forearm_w_norm"),
                        "sil_area_norm_rel_spread_pct": rel("sil_area_norm"),
                        "view_ratio_spread": spread("view_ratio"),
                        "body_h_frac_spread": spread("body_h_frac"),
                        "body_cx_spread": spread("body_cx"),
                        "bg_dist_spread": round(float(np.ptp([res[day][n]["bg_dist_to_35_39_43"] for n in used])), 1) if len(used) > 1 else None}
    json.dump({"clips": res, "between": between}, open(f"{OUT}/metrics.json", "w"), indent=1)
    print(json.dumps(between, indent=1))

if __name__ == "__main__":
    main()
