#!/usr/bin/env python3
"""t654u: Day 4 vs D7 muscle look, measured at the SAME on-screen muscle scale.
Scale = torso length (shoulder midpoint -> hip midpoint, MediaPipe on the final 1080 frame); the D7 frame is resized
so its torso length equals the Day 4 frame's. Metrics on the step's glow (wrong: r - max(g,b), right: g - max(r,b)):
  core   = dom >= 0.6 x p95(dom)  -> hue (deg), saturation, value of the solid colour
  fibre  = luminance band-pass (sigma 1.2 - sigma 4) inside the core: contrast (std) and ridge density
           (sign changes per 100 px, horizontal + vertical)
  falloff= median distance (px, Day 4 scale) from the core edge to pixels at 10-60% of p95 (core-to-grey ramp)
  cover  = glow px (dom > 40) / torso_len^2
usage: python3 tools/day4_look_metrics.py <name>=<frame.jpg>:<wrong|right> ...  (first = Day 4 reference); <name>=<frame>:<form>:<torso_px> fixes the torso length"""
import sys, json, cv2, numpy as np
sys.path.insert(0, "/workspace/daily_info_reels")
import universal_anatomy_rig as uar


def torso_len(bgr):
    r = uar.get_pose_detector().detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
    lm = r.pose_landmarks[0]; h, w = bgr.shape[:2]
    P = lambda i: np.array([lm[i].x * w, lm[i].y * h])
    return float(np.linalg.norm(0.5 * (P(11) + P(12)) - 0.5 * (P(23) + P(24))))


def metrics(bgr, form, scale):
    f = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC).astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    dom = (r - np.maximum(g, b)) if form == "wrong" else (g - np.maximum(r, b))
    p95 = float(np.percentile(dom[dom > 20], 95)) if (dom > 20).sum() > 500 else 1.0
    core = dom >= 0.6 * p95
    hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[..., 0][core] * 2.0
    hue_c = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(hue)))))) % 360
    L = 0.299 * r + 0.587 * g + 0.114 * b
    bp = cv2.GaussianBlur(L, (0, 0), 1.2) - cv2.GaussianBlur(L, (0, 0), 4.0)
    ce = cv2.erode(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    sx = (np.sign(bp[:, 1:]) != np.sign(bp[:, :-1])) & ce[:, 1:] & ce[:, :-1]
    sy = (np.sign(bp[1:, :]) != np.sign(bp[:-1, :])) & ce[1:, :] & ce[:-1, :]
    dens = 100.0 * (sx.sum() + sy.sum()) / max(2 * ce.sum(), 1)
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    ramp = (dom > 0.1 * p95) & (dom < 0.6 * p95)
    return {"core_hue_deg": round(hue_c, 1), "core_sat": round(float(hsv[..., 1][core].mean()), 1), "core_val": round(float(hsv[..., 2][core].mean()), 1),
            "fibre_contrast": round(float(bp[ce].std()), 2), "fibre_density_per100px": round(float(dens), 2),
            "falloff_px": round(float(np.median(dist[ramp])) if ramp.sum() else 0.0, 2), "core_px": int(core.sum()), "glow_px": int((dom > 40).sum())}


out, ref_t = {}, None
for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1); parts = rest.split(":")
    path, form = parts[0], parts[1]
    im = cv2.imread(path)
    t = float(parts[2]) if len(parts) > 2 else torso_len(im)   # optional fixed torso px (detector miss on a glow frame)
    ref_t = ref_t or t
    m = metrics(im, form, ref_t / t)
    m["torso_px_onscreen"] = round(t, 1); m["scale_to_ref"] = round(ref_t / t, 3)
    m["cover_glow_per_torso2"] = round(m["glow_px"] / (ref_t ** 2), 3)
    out[name] = m
print(json.dumps(out, indent=1))
