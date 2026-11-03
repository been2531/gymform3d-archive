#!/usr/bin/env python3
"""Reusable still pre-check before any i2v (t693u). One JSON config per exercise/camera (configs/precheck/*.json).
Checks, each PASS/FAIL with the number:
  machine   SSIM vs the empty plate per ROI, outside the dilated body mask and the moving handle/cable band
  cable     dark thin line from the pulley point to the hands (fraction of samples darker than both side neighbours)
  arm       START: elbow flexion <= max (arms extended);  PEAK: elbow behind the shoulder by >= frac x torso length
  torso     lean from vertical <= max per still; |start - peak| <= delta (pair mode)
  handle    rectangle-frame detector: bright metal pixels in a box ABOVE the hands (a V handle has none there)
  head      head vs body luminance |dL|
  light     shaded side (p10 of upper-body luminance) minus backdrop median >= min
  glow      green/red dominance pixels on the body
usage: python3 tools/still_precheck.py CONFIG --start S.png [--peak P.png] [--out json]"""
import sys, json, argparse, cv2, numpy as np, mediapipe as mp
from skimage.metrics import structural_similarity as ssim
sys.path.insert(0, "/workspace/daily_info_reels")
from universal_anatomy_rig import get_pose_detector
R = "/workspace/daily_info_reels/"
def pose(img):
    r = get_pose_detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks: return None, None
    h, w = img.shape[:2]; L = r.pose_landmarks[0]
    return np.array([[p.x * w, p.y * h, p.visibility] for p in L]), (r.segmentation_masks[0].numpy_view()[..., 0] > 0.3).astype(np.uint8)
def ang(a, b, c):
    v1, v2 = a - b, c - b; return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))))
def check(cfg, path, kind):
    img = cv2.imread(path); H, W = cfg["size"][1], cfg["size"][0]
    if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    plate = cv2.imread(R + cfg["plate"]); out = {"file": path, "kind": kind}
    kp, seg = pose(img)
    if kp is None: return {**out, "pass": False, "error": "no pose"}
    g, pg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    excl = cv2.dilate(seg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool)
    y0, y1, x0, x1 = cfg["moving_band"]; band = np.zeros_like(excl); band[y0:y1, x0:x1] = True
    _, sm = ssim(pg, g, full=True); res = {}
    for k, (a, b, c, d) in cfg["machine_rois"].items():
        m = ~(excl | band)[a:b, c:d]; res[k] = round(float(sm[a:b, c:d][m].mean()), 3) if m.sum() > 500 else None
    allm = ~(excl | band); res["all_static"] = round(float(sm[allm].mean()), 3)
    vals = [v for v in res.values() if v is not None]
    out["machine"] = {"ssim": res, "pass": min(vals) >= cfg["machine_ssim_min"]}
    side = 11 if kp[11, 2] >= kp[12, 2] else 12
    s, e, wr = kp[side, :2], kp[side + 2, :2], kp[side + 4, :2]
    sh = (kp[11, :2] + kp[12, :2]) / 2; hp = (kp[23, :2] + kp[24, :2]) / 2; tl = float(np.linalg.norm(sh - hp))
    sgn = 1 if cfg.get("tower_side", "left") == "left" else -1   # +x = away from the tower = backward
    lean = float(np.degrees(np.arctan2(sgn * (sh[0] - hp[0]), hp[1] - sh[1])))
    flex = 180 - ang(s, e, wr); past = sgn * float(e[0] - s[0]) / tl
    out["pose"] = {"torso_lean_deg": round(lean, 1), "elbow_flex_deg": round(flex, 1), "elbow_behind_shoulder_frac": round(past, 3), "torso_px": round(tl, 1)}
    out["torso"] = {"pass": abs(lean) <= cfg["torso_lean_max"]}
    if kind == "start": out["arm"] = {"rule": f"elbow flex <= {cfg['start']['elbow_flex_max']}", "pass": flex <= cfg["start"]["elbow_flex_max"]}
    else: out["arm"] = {"rule": f"elbow behind shoulder >= {cfg['peak']['elbow_past_shoulder_min_frac']} torso", "pass": past >= cfg["peak"]["elbow_past_shoulder_min_frac"]}
    # cable: samples from the pulley to the hand midpoint, skip the last 12% (hands/handle) and the first 4% (pulley)
    hand = (kp[15, :2] + kp[16, :2]) / 2 if min(kp[15, 2], kp[16, 2]) > 0.2 else wr
    gb = cv2.GaussianBlur(g, (3, 3), 0).astype(float)
    def cover(p0):
        v = hand - p0; L = np.linalg.norm(v); u = v / (L + 1e-6); nrm = np.array([-u[1], u[0]]); hits = tot = 0
        for t in np.linspace(0.06, 0.88, 60):
            c = p0 + v * t; best = 0
            for off in range(-3, 4):
                q = c + nrm * off; xi, yi = int(q[0]), int(q[1])
                if not (3 <= xi < W - 3 and 3 <= yi < H - 3): continue
                sa, sb = c + nrm * (off - 5), c + nrm * (off + 5)
                va = gb[int(np.clip(sa[1], 0, H - 1)), int(np.clip(sa[0], 0, W - 1))]; vb = gb[int(np.clip(sb[1], 0, H - 1)), int(np.clip(sb[0], 0, W - 1))]
                best = max(best, min(va, vb) - gb[yi, xi])
            tot += 1; hits += best >= 6
        return hits / max(tot, 1), L
    pr = cfg.get("pulley_search_px", 24); cands = []
    for dx in range(-pr, pr + 1, 4):
        for dy in range(-pr, pr + 1, 4):
            p0 = np.array(cfg["pulley_xy"], float) + [dx, dy]; cv_, L = cover(p0); cands.append((cv_, L, p0))
    cov, L, p0b = max(cands, key=lambda t: t[0]); hits = tot = 1
    out["cable"] = {"cover": round(cov, 2), "len_px": round(float(L), 1), "pulley_used": p0b.round(0).tolist(), "pass": cov >= cfg["cable_min_cover"]}
    # handle: bright low-sat metal pixels in a box above the hands (rectangle frame top bar); a V handle sits below/at the grip
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); metal = (hsv[..., 2] > 150) & (hsv[..., 1] < 45) & ~seg.astype(bool)
    hx, hy = int(hand[0]), int(hand[1]); bx = metal[max(0, hy - 75):max(0, hy - 22), max(0, hx - 70):hx + 70]
    out["handle"] = {"rect_top_metal_px": int(bx.sum()), "pass": int(bx.sum()) <= cfg["handle_rect_top_px_max"]}
    # head vs body tone, light, glow
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float) * 100 / 255
    head = np.zeros_like(seg); hc = kp[[0, 7, 8], :2].mean(0); hr = int(0.22 * tl)
    cv2.circle(head, (int(hc[0]), int(hc[1])), hr, 1, -1); head &= seg
    upper = seg.copy(); upper[int(hp[1]) - 10:, :] = 0; upper &= (1 - cv2.dilate(head, np.ones((9, 9), np.uint8)))
    hL, bL = float(np.median(lum[head > 0])), float(np.median(lum[upper > 0]))
    out["head"] = {"head_L": round(hL, 1), "body_L": round(bL, 1), "dL": round(abs(hL - bL), 1), "pass": abs(hL - bL) <= cfg["head_body_dL_max"]}
    bgm = ~cv2.dilate(seg, np.ones((61, 61), np.uint8)).astype(bool); bgm[:, :240] = False; bgm[1000:, :] = False
    bgL = float(np.median(lum[bgm]))
    # shaded SIDE, not crevices: low-pass luminance (sigma 8) inside the eroded upper body, then p10
    er = cv2.erode(upper, np.ones((11, 11), np.uint8)).astype(bool)
    lp = cv2.GaussianBlur(lum * upper, (0, 0), 8) / np.maximum(cv2.GaussianBlur(upper.astype(float), (0, 0), 8), 1e-3)
    p10 = float(np.percentile(lp[er], 10))
    out["light"] = {"body_p10_L": round(p10, 1), "backdrop_L": round(bgL, 1), "diff": round(p10 - bgL, 1), "pass": p10 - bgL >= cfg["shaded_vs_backdrop_min"]}
    b, gr, r = [img[..., i].astype(int) for i in range(3)]
    glow = ((gr - np.maximum(r, b) > 25) | (r - np.maximum(gr, b) > 35)) & seg.astype(bool)
    gf = float(glow.sum() / max(seg.sum(), 1)); out["glow"] = {"frac": round(gf, 4), "pass": gf <= cfg["glow_frac_max"]}
    out["kp"] = {"shoulder": s.round(1).tolist(), "elbow": e.round(1).tolist(), "wrist": wr.round(1).tolist(), "hand_mid": hand.round(1).tolist(), "sh_mid": sh.round(1).tolist(), "hip_mid": hp.round(1).tolist()}
    keys = ["machine", "torso", "arm", "cable", "handle", "head", "light", "glow"]
    out["fails"] = [k for k in keys if not out[k]["pass"]]; out["pass"] = not out["fails"]
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--start"); ap.add_argument("--peak"); ap.add_argument("--out")
    a = ap.parse_args(); cfg = json.load(open(a.config)); res = {}
    if a.start: res["start"] = check(cfg, a.start, "start")
    if a.peak: res["peak"] = check(cfg, a.peak, "peak")
    if a.start and a.peak and "pose" in res["start"] and "pose" in res["peak"]:
        dlt = abs(res["start"]["pose"]["torso_lean_deg"] - res["peak"]["pose"]["torso_lean_deg"])
        res["pair"] = {"torso_delta_deg": round(dlt, 1), "pass": dlt <= cfg["torso_delta_max"]}
    def _py(o):
        if isinstance(o, dict): return {k: _py(v) for k, v in o.items()}
        if isinstance(o, list): return [_py(v) for v in o]
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.generic): return o.item()
        return o
    res = _py(res)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items(): print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("kp",)}))
