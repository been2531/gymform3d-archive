"""LOCKED studio background: minimal charcoal plate, floor contact shadow, per-step size/floor transform, temporal backdrop plate (cached)."""
import math
import cv2
import numpy as np
from gymform.ctx import *  # shared state: W,H,CFG,LAST_MASKS,CUR_PLATE,STEP_SCALE,_XX,_YY,d5,b4,p62,v30
import os, hashlib
from gymform import ctx

_STUDIO_BG = {}


def studio_bg() -> np.ndarray:
    p = CFG["studio_bg"]
    if p not in _STUDIO_BG:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        _STUDIO_BG[p] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return _STUDIO_BG[p]


def contact_shadow(kp: dict) -> np.ndarray:
    ank_r, ank_l = kp["ank_r"], kp["ank_l"]
    ay = max(ank_r[1], ank_l[1]) + 30.0
    cx = 0.5 * (ank_r[0] + ank_l[0])
    span = abs(ank_l[0] - ank_r[0])
    sh = np.zeros((H, W), dtype=np.float32)
    cv2.ellipse(sh, (int(cx), int(ay)), (int(0.5 * span + 110), 28), 0, 0, 360, 0.62, -1, cv2.LINE_AA)
    for a in (ank_r, ank_l):  # tighter, darker core under each foot
        cv2.ellipse(sh, (int(a[0] + 10), int(a[1] + 36)), (72, 15), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    return np.clip(cv2.GaussianBlur(sh, (0, 0), 11.0), 0.0, 1.0)


def transform(bgr_720p, kp_cached, y_sh, step_idx):
    """720p clip -> 1080x1920 canvas: per-step scale (equal skeletal size x target zoom), DX/DY framing.
    Single Lanczos resample straight from 720p (no double resize) + the locked light unsharp."""
    sc = STEP_SCALE.get(step_idx, b4.SCALE)
    x0, y0, nw, nh = ctx.clip_rect(step_idx)
    rgb720 = cv2.cvtColor(bgr_720p, cv2.COLOR_BGR2RGB).astype(np.float32)
    if sc <= 1.0:   # approved Day 1-6 path, unchanged: 1080 upscale + light unsharp, then downscale
        up = cv2.resize(rgb720, (W, H), interpolation=cv2.INTER_LANCZOS4)
        up = up * 1.22 - cv2.GaussianBlur(up, (0, 0), 0.9) * 0.22
        scaled = cv2.resize(up, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    else:           # target zoom: ONE Lanczos resample 720p -> final size (no double resample), unsharp at final scale
        scaled = cv2.resize(rgb720, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        scaled = scaled * 1.22 - cv2.GaussianBlur(scaled, (0, 0), 0.9 * sc) * 0.22
    bg_med = np.median(np.concatenate([rgb720[:18].reshape(-1, 3), rgb720[:, :18].reshape(-1, 3), rgb720[:, -18:].reshape(-1, 3)]), axis=0)
    canvas = np.ones((H, W, 3), np.float32) * bg_med[None, None, :]
    sx0, sy0, dx0, dy0 = max(0, -x0), max(0, -y0), max(0, x0), max(0, y0)
    cw, ch = min(nw - sx0, W - dx0), min(nh - sy0, H - dy0)
    sub = scaled[sy0:sy0 + ch, sx0:sx0 + cw]
    yy_s, xx_s = np.mgrid[:ch, :cw].astype(np.float32)
    f = 36.0
    # feather only edges where the clip ends inside the canvas (a clip edge outside the canvas is not a seam)
    wl = np.clip(xx_s / f, 0, 1) if x0 > 0 else 1.0
    wr = np.clip((cw - 1 - xx_s) / f, 0, 1) if x0 + nw < W else 1.0
    wt = np.clip(yy_s / f, 0, 1) if y0 > 0 else 1.0
    wb = np.clip((ch - 1 - yy_s) / f, 0, 1) if y0 + nh < H else 1.0
    wm = (np.minimum(wl, wr) * np.minimum(wt, wb) * np.ones_like(xx_s))[:, :, None]
    canvas[dy0:dy0 + ch, dx0:dx0 + cw] = canvas[dy0:dy0 + ch, dx0:dx0 + cw] * (1.0 - wm) + sub * wm
    kp_out = {}
    for k, (px, py_c) in kp_cached.items():
        kp_out[k] = (float((px - W * 0.5) * sc + W * 0.5 + int(STEP_DX.get(step_idx, 0))),
                     float((py_c - y_sh - H * 0.5) * sc + H * 0.5 + int(b4.STEP_DY.get(step_idx, 0))))
    return np.clip(canvas, 0, 255), kp_out


b4.transform_frame_and_kp_1080p = transform


def set_step_plate(seq, kps, step_idx, src=None):
    """Per-step temporal median backdrop plate (720p native -> the step's own 1080 transform), cached on disk.
    src = clip that seq comes from when it is not the step's own clip (motion base of an X-Ray edit)."""
    fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(step_idx)
    fn = src or fn
    key = hashlib.md5(repr((fn, i0, i1)).encode()).hexdigest()[:10]
    cp = os.path.join(ctx.CACHE_DIR, f"median_{CFG['day']:02d}_s{step_idx}_{key}.npy")   # exact 720p median (uint8)
    if os.path.exists(cp):
        med = np.load(cp)
    else:
        med = np.median(np.stack(seq).astype(np.uint8), axis=0).astype(np.uint8)
        np.save(cp, med)
    mp = cp[:-4] + "_motion.npy"
    if os.path.exists(mp):
        mot = np.load(mp)
    else:   # fraction of frames where a pixel differs from the median (moving body / equipment / their ghosts)
        mot = np.mean([np.abs(f.astype(np.int16) - med.astype(np.int16)).max(axis=-1) > 10 for f in seq], axis=0).astype(np.float32)
        mot = cv2.dilate(mot, np.ones((5, 5), np.uint8))
        np.save(mp, mot)
    rgb = b4.transform_frame_and_kp_1080p(med, kps[0], y_sh, step_idx)[0]
    mot3 = np.repeat(np.clip(mot * 255.0, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    CUR_PLATE["motion"] = b4.transform_frame_and_kp_1080p(mot3, kps[0], y_sh, step_idx)[0][:, :, 0] / 255.0
    CUR_PLATE["rgb"] = rgb.astype(np.float32)
    CUR_PLATE["step"] = step_idx
