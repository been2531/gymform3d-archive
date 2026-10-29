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


_orig_transform = b4.transform_frame_and_kp_1080p


def transform(bgr_720p, kp_cached, y_sh, step_idx):
    """Per-step scale (equal on-screen mannequin height) + per-step DY (feet on one floor line)."""
    old = b4.SCALE
    b4.SCALE = STEP_SCALE.get(step_idx, old)
    try:
        return _orig_transform(bgr_720p, kp_cached, y_sh, step_idx)
    finally:
        b4.SCALE = old


b4.transform_frame_and_kp_1080p = transform


def set_step_plate(seq, kps, step_idx):
    """Per-step temporal median backdrop plate (720p native -> the step's own 1080 transform), cached on disk."""
    fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(step_idx)
    key = hashlib.md5(repr((fn, i0, i1)).encode()).hexdigest()[:10]
    cp = os.path.join(ctx.CACHE_DIR, f"median_{CFG['day']:02d}_s{step_idx}_{key}.npy")   # exact 720p median (uint8)
    if os.path.exists(cp):
        med = np.load(cp)
    else:
        med = np.median(np.stack(seq).astype(np.uint8), axis=0).astype(np.uint8)
        np.save(cp, med)
    rgb = b4.transform_frame_and_kp_1080p(med, kps[0], y_sh, step_idx)[0]
    CUR_PLATE["rgb"] = rgb.astype(np.float32)
    CUR_PLATE["step"] = step_idx
