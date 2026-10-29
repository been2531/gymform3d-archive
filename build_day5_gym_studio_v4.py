#!/usr/bin/env python3
"""
@gymform.3d 4-step master reel — V4 studio pipeline (Day 5 Dumbbell Lateral Raise = config 5, output *_v4b).

Builds on (does NOT modify) build_day5_lateral_raise_master (d5), build_4step_bone_muscle_master (b4)
and preview_v62_seamless_day2_master (p62). Overrides only:
  1. Clean anatomical foreground mask (row_bg from inner clip strips, keypoint envelope, glow-robust bright key,
     shorts polygon, equipment hull fill from per-day equipment zones) composited onto a studio plate
     (default: output/assets/private_gym_studio_minimal_graded_1080p.jpg) with a soft floor contact shadow.
  2. Crisp activation colour (core_w sigma 1.0, no +58*bone_core white-hot boost), colour only inside w_fg;
     Step 1 = crimson bone X-Ray gated to clavicle/acromion/humeral head, Step 3 green bone X-Ray lifted.
  3. Crisp laser renderer (core 2.8px / halo 6px s3.5 / faint glow s10 a45, small vertex rings).
  4. Title depth-behind + translucent top preview driven by the clean w_fg (mannequin never frosted).
  5. Per-step scale/DY so the mannequin has the same on-screen height and floor line in every step.
Tempo, slices, callouts, TTS and the 4-slot preview are identical to d5.build_all.
Writes a QC sidecar (CFG["qc_dir"]) consumed by qc_gate_gymform.py.

Usage:  python3 build_day5_gym_studio_v4.py            # full render (mp4 + sheet + cover + QC sidecar)
        python3 build_day5_gym_studio_v4.py --peaks    # quick QC frames (start/peak) + sheet, no video
        python3 qc_gate_gymform.py output/day5_dumbbell_lateral_raise_4step_master_v4d.mp4 --qc-dir output/qc/day5_v4b
"""
import os
import time
import sys
import gc
import math
import pickle
import asyncio
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import build_day5_lateral_raise_master as d5  # noqa: E402  (sets b4.STEP_DY for Day 5)
import build_4step_bone_muscle_master as b4  # noqa: E402
import preview_v62_seamless_day2_master as p62  # noqa: E402
import build_v30_pure_graphic_numbers as v30  # noqa: E402

cv2.setNumThreads(2)

OUT_DIR = d5.OUT_DIR
ASSETS_DIR = d5.ASSETS_DIR
TMP_DIR = "/workspace/daily_info_reels/_tmp_day5_v4"
os.makedirs(TMP_DIR, exist_ok=True)
d5.TMP_DIR = TMP_DIR  # TTS scratch files go to the V4 tmp dir

W, H, FPS, DURATION = d5.W, d5.H, d5.FPS, d5.DURATION
# ---------------------------------------------------------------------------------------------
# Per-day config (reuse for future days: add an entry, point step_specs/kp cache at that day's builder).
# equipment zones (all in 1080x1920 frame coords, resolved from keypoints per frame):
#   {"type": "dumbbell", "anchor": "wr_r", "radius": 140}                  circle around a keypoint
#   {"type": "barbell",  "segment": ["wr_r", "wr_l"], "pad": 90, "ext": 0.6}  capsule along two keypoints (extended)
#   {"type": "cable",    "segment": ["wr_r", [540, 200]], "pad": 40}       keypoint to fixed anchor point
#   {"type": "bench",    "roi": [x0, y0, x1, y1]}                          fixed rectangle (machines/benches/racks)
# Inside each zone, equipment pixels (bright OR darker-than-backdrop) are morphologically closed and hole-filled
# into the solid foreground so matte-black / metal objects are never see-through or clipped by the body mask.
# ---------------------------------------------------------------------------------------------
DAY_CONFIGS = {
    5: {
        "exercise_name": "Dumbbell Lateral Raise",
        "studio_bg": os.path.join(ASSETS_DIR, "private_gym_studio_minimal_graded_1080p.jpg"),  # minimal @demicstory-style plate, no props
        # Source clips are shot at different camera distances (Steps 3/4 ~14% larger than 1/2), so equal on-screen
        # mannequin size needs a per-step scale compensation. Step 1 keeps the base 0.90; others are normalised to
        # Step 1's head-to-ankle height, and DY lands every step's feet on the same floor line (max ankle y ~1660).
        "step_scale": {1: 0.90, 2: 0.938, 3: 0.789, 4: 0.797},
        "step_dy": {1: 80, 2: 87, 3: 157, 4: 148},
        "equipment": [
            {"type": "dumbbell", "anchor": "wr_r", "radius": 250},
            {"type": "dumbbell", "anchor": "wr_l", "radius": 250},
        ],
        "bone_steps": (1, 3),
        "xray_style": "day1_v36",
        "callout_ty": {2: 408, 3: 440},  # Step 3 text block fixed above the raised left dumbbell (clear of plates/title)  # reusable X-Ray module (xray_day1_v36) = approved Day 1 v36 native X-Ray look
        "muscle_steps": (2, 4),
        "out_mp4": os.path.join(OUT_DIR, "day5_dumbbell_lateral_raise_4step_master_v4d.mp4"),
        "out_sheet": os.path.join(OUT_DIR, "day5_v4d_4step_sheet.jpg"),
        "out_cover": os.path.join(OUT_DIR, "day5_v4d_cover.jpg"),
        "qc_dir": os.path.join(OUT_DIR, "qc", "day5_v4d"),
    },
}
CFG = DAY_CONFIGS[5]
# run_day.py auto-fix loop: JSON overrides (step_scale / step_dy / equipment radius / callout_ty / outputs)
_ovr_p = os.environ.get("GYMFORM_OVERRIDES")
if _ovr_p and os.path.exists(_ovr_p):
    import json as _json
    _ovr = _json.load(open(_ovr_p))
    for _k, _v in _ovr.items():
        if isinstance(_v, dict) and isinstance(CFG.get(_k), dict):
            CFG[_k].update({int(a) if str(a).isdigit() else a: b for a, b in _v.items()})
        else:
            CFG[_k] = _v
STUDIO_BG_PATH = CFG["studio_bg"]
OUT_MP4, OUT_SHEET, OUT_COVER = CFG["out_mp4"], CFG["out_sheet"], CFG["out_cover"]

# Step 1 vertical placement. Feet already sit at y~1697 with d5's +80, so the peak dumbbells (y~255)
# cannot be pushed below y~430 without the feet leaving the y<1700 safe zone; keep +80 (see report).
b4.STEP_DY = dict(b4.STEP_DY)
# Same SCALE (0.90) for every step so mannequin size is identical; only vertical offsets are tuned so the
# feet land on the same floor line (max ankle y ~1660). Step 1 peak dumbbells overlapping the preview bar are
# handled by the translucent preview (mannequin shown sharp through it), not by shrinking.
b4.STEP_DY.update(CFG["step_dy"])
STEP_SCALE = dict(CFG.get("step_scale", {}))

_orig_transform = b4.transform_frame_and_kp_1080p


def transform_v4(bgr_720p, kp_cached, old_y_sh, step_idx):
    old = b4.SCALE
    b4.SCALE = STEP_SCALE.get(step_idx, old)
    try:
        return _orig_transform(bgr_720p, kp_cached, old_y_sh, step_idx)
    finally:
        b4.SCALE = old


b4.transform_frame_and_kp_1080p = transform_v4

_STUDIO_BG = None


def studio_bg() -> np.ndarray:
    global _STUDIO_BG
    if _STUDIO_BG is None:
        bgr = cv2.imread(STUDIO_BG_PATH, cv2.IMREAD_COLOR)
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        _STUDIO_BG = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return _STUDIO_BG


# ---------------------------------------------------------------------------------------------
# 1. Anatomical envelope from keypoints
# ---------------------------------------------------------------------------------------------
_YY, _XX = np.mgrid[:H, :W].astype(np.float32)


def _seg_dist(p0, p1, x0, y0, x1, y1):
    """Distance field to segment p0-p1 restricted to ROI [x0:x1, y0:y1]."""
    xx = _XX[y0:y1, x0:x1]
    yy = _YY[y0:y1, x0:x1]
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay
    L2 = max(dx * dx + dy * dy, 1e-6)
    t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(xx - (ax + t * dx), yy - (ay + t * dy))


def _ext(p0, p1, t):
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def xray_window(kp: dict, is_wrong: bool) -> np.ndarray:
    """
    Local X-Ray window (soft 0..1) around the target bones only: near-side shoulder girdle (clavicle to the
    midline, scapula/acromion, humeral head) + near upper arm (humerus). Step 1 (wrong posture, both arms
    overhead, rib cage in the native X-Ray) also opens both upper arms and the upper rib cage.
    """
    sh_r, sh_l = kp["sh_r"], kp["sh_l"]
    sw = abs(sh_l[0] - sh_r[0])
    sh_mid = (0.5 * (sh_r[0] + sh_l[0]), 0.5 * (sh_r[1] + sh_l[1]))
    win = np.zeros((H, W), np.float32)
    soft = 20.0
    win = np.maximum(win, np.clip((95.0 - np.hypot(_XX - sh_r[0], _YY - sh_r[1])) / soft, 0.0, 1.0))
    win = np.maximum(win, np.clip((48.0 - _seg_dist(sh_r, (sh_mid[0], sh_mid[1] + 10.0), 0, 0, W, H)) / soft, 0.0, 1.0))
    win = np.maximum(win, np.clip((62.0 - _seg_dist(sh_r, _ext(sh_r, kp["elb_r"], 1.05), 0, 0, W, H)) / soft, 0.0, 1.0))
    if is_wrong:
        win = np.maximum(win, np.clip((95.0 - np.hypot(_XX - sh_l[0], _YY - sh_l[1])) / soft, 0.0, 1.0))
        win = np.maximum(win, np.clip((48.0 - _seg_dist(sh_l, (sh_mid[0], sh_mid[1] + 10.0), 0, 0, W, H)) / soft, 0.0, 1.0))
        win = np.maximum(win, np.clip((62.0 - _seg_dist(sh_l, _ext(sh_l, kp["elb_l"], 1.05), 0, 0, W, H)) / soft, 0.0, 1.0))
        rib = np.hypot((_XX - sh_mid[0]) / max(110.0, 0.42 * sw + 30.0), (_YY - (sh_mid[1] + 150.0)) / 170.0)
        win = np.maximum(win, np.clip((1.0 - rib) / 0.18, 0.0, 1.0))
    return win


def anatomical_envelope(kp: dict) -> np.ndarray:
    sh_r, sh_l = kp["sh_r"], kp["sh_l"]
    hip_r, hip_l = kp["hip_r"], kp["hip_l"]
    sh_m = (0.5 * (sh_r[0] + sh_l[0]), 0.5 * (sh_r[1] + sh_l[1]))
    hip_m = (0.5 * (hip_r[0] + hip_l[0]), 0.5 * (hip_r[1] + hip_l[1]))
    torso_len = math.hypot(sh_m[0] - hip_m[0], sh_m[1] - hip_m[1])
    head = (sh_m[0] + (sh_m[0] - hip_m[0]) * 0.42, sh_m[1] - 0.42 * torso_len)

    segs = [  # (p0, p1, normaliser)
        (head, hip_m, 185.0),
        (sh_r, sh_l, 185.0 * 0.62),
        (hip_r, hip_l, 185.0 * 0.62),
    ]
    for side in ("r", "l"):
        sh, el, wr = kp[f"sh_{side}"], kp[f"elb_{side}"], kp[f"wr_{side}"]
        hip, kn, ank = kp[f"hip_{side}"], kp[f"kn_{side}"], kp[f"ank_{side}"]
        segs.append((sh, el, 92.0))                    # upper arm
        segs.append((el, _ext(el, wr, 1.30), 120.0))   # forearm incl. hand + dumbbell
        segs.append((hip, kn, 108.0))                  # thigh
        segs.append((kn, _ext(kn, ank, 1.12), 95.0))   # shin + foot

    pts = np.array([p for s in segs for p in s[:2]], dtype=np.float32)
    pad = int(1.35 * 185.0) + 8
    x0 = int(max(0, pts[:, 0].min() - pad)); x1 = int(min(W, pts[:, 0].max() + pad))
    y0 = int(max(0, pts[:, 1].min() - pad)); y1 = int(min(H, pts[:, 1].max() + pad))
    dmin = np.full((y1 - y0, x1 - x0), 99.0, dtype=np.float32)
    for p0, p1, r in segs:
        np.minimum(dmin, _seg_dist(p0, p1, x0, y0, x1, y1) / r, out=dmin)
    env = np.zeros((H, W), dtype=np.float32)
    env[y0:y1, x0:x1] = np.clip((1.35 - dmin) / 0.30, 0.0, 1.0)
    return env


# ---------------------------------------------------------------------------------------------
# 2. Shading + clean foreground composite
# ---------------------------------------------------------------------------------------------
def equipment_zones(kp: dict):
    """Yields (x0, y0, x1, y1, zone_mask_uint8) per configured equipment zone."""
    for eq in CFG["equipment"]:
        z = np.zeros((H, W), np.uint8)
        if "anchor" in eq:
            ax, ay = kp[eq["anchor"]]
            cv2.circle(z, (int(ax), int(ay)), int(eq["radius"]), 1, -1)
        elif "segment" in eq:
            pa, pb = [kp[p] if isinstance(p, str) else tuple(p) for p in eq["segment"]]
            e = eq.get("ext", 0.0)
            pa2 = (pa[0] - (pb[0] - pa[0]) * e, pa[1] - (pb[1] - pa[1]) * e)
            pb2 = (pb[0] + (pb[0] - pa[0]) * e, pb[1] + (pb[1] - pa[1]) * e)
            cv2.line(z, (int(pa2[0]), int(pa2[1])), (int(pb2[0]), int(pb2[1])), 1, int(2 * eq["pad"]))
        elif "roi" in eq:
            x0, y0, x1, y1 = eq["roi"]
            z[y0:y1, x0:x1] = 1
        ys, xs = np.where(z > 0)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        yield x0, y0, x1, y1, z[y0:y1, x0:x1]


def equipment_soft_zone(kp: dict) -> np.ndarray:
    zz = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1, z in equipment_zones(kp):
        zz[y0:y1, x0:x1] = np.maximum(zz[y0:y1, x0:x1], z.astype(np.float32))
    # full weight ~25px inside the zone edge, fading to 0 at the edge
    return np.clip(cv2.distanceTransform((zz > 0).astype(np.uint8), cv2.DIST_L2, 3) / 25.0, 0.0, 1.0)


LAST_MASKS = {}  # filled by shade_day5_v4 for QC sidecars
CUR_PLATE = {}   # {"rgb": float32 1080x1920x3 temporal clean plate of the current step}


def set_step_plate(seq, kps, step_idx):
    """Per-step temporal median plate (720p native, then the step's own 1080 transform)."""
    fn, i0, i1, is_w, v_mode, y_sh = d5.STEP_SPECS_DAY5[step_idx]
    med = np.median(np.stack(seq).astype(np.uint8), axis=0).astype(np.uint8)
    CUR_PLATE["rgb"] = b4.transform_frame_and_kp_1080p(med, kps[0], y_sh, step_idx)[0].astype(np.float32)
    CUR_PLATE["step"] = step_idx


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


def shorts_polygon(kp: dict, bright_m: np.ndarray, hip_y: float, kn_y: float) -> np.ndarray:
    """Opaque boxer-shorts region from body extents at the waist (above) and thighs (below) + crotch notch."""
    hx = int(0.5 * (kp["hip_r"][0] + kp["hip_l"][0]))
    xa, xb = max(0, hx - 300), min(W, hx + 300)
    top = int(hip_y - 125.0)
    bot = int(hip_y + 0.55 * (kn_y - hip_y))
    crotch = int(hip_y + 0.36 * (kn_y - hip_y))

    # exclude hanging forearms / hands / dumbbells next to the hips from the body-width measurement
    y_a, y_b = max(0, int(hip_y) - 140), min(H, int(kn_y) + 10)
    arm_x = np.zeros((y_b - y_a, W), bool)
    for side in ("r", "l"):
        el, wr = kp[f"elb_{side}"], kp[f"wr_{side}"]
        d1 = _seg_dist(el, _ext(el, wr, 1.35), 0, y_a, W, y_b)
        d2 = np.hypot(_XX[y_a:y_b] - wr[0], _YY[y_a:y_b] - wr[1])
        arm_x |= (d1 < 55.0) | (d2 < 105.0)
    bright_m = bright_m.copy()
    bright_m[y_a:y_b] &= ~arm_x

    def cols(r):
        return np.where(bright_m[max(0, r - 4) : r + 5, xa:xb].any(axis=0))[0] + xa

    def occ_row(r):
        o = bright_m[max(0, r - 4) : r + 5, :].any(axis=0).astype(np.uint8)
        return cv2.morphologyEx(o[None, :], cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))[0] > 0

    def run_at(o, x):
        """Contiguous occupied run containing (or nearest to, within 40px) column x."""
        x = int(np.clip(x, 0, W - 1))
        if not o[x]:
            near = np.where(o[max(0, x - 40) : x + 41])[0]
            if len(near) == 0:
                return None
            x = max(0, x - 40) + int(near[np.argmin(np.abs(near - 40))])
        l = x
        while l > 0 and o[l - 1]:
            l -= 1
        r = x
        while r < W - 1 and o[r + 1]:
            r += 1
        return l, r

    # waist: the skin run through the body centre (arms hanging beside it are separate runs)
    rt_run = run_at(occ_row(top), hx)
    ob = occ_row(bot)
    rl = run_at(ob, kp["hip_r"][0])
    rr = run_at(ob, kp["hip_l"][0])
    out = np.zeros((H, W), dtype=np.float32)
    if rt_run is None or rl is None or rr is None:
        return out
    lt, rt = rt_run[0] - 4, rt_run[1] + 4
    lb, rb = rl[0] - 4, rr[1] + 4
    pts = [(lt, top), (rt, top), (rb, bot)]
    occ = bright_m[max(0, bot - 4) : bot + 5, :].any(axis=0)
    gl = gr = None
    if not occ[hx]:
        gl, gr = hx, hx
        while gl > lb and not occ[gl]:
            gl -= 1
        while gr < rb and not occ[gr]:
            gr += 1
    if gl is not None and gr - gl > 8:
        pts += [(gr - 2, bot), (hx, crotch), (gl + 2, bot)]
    pts += [(lb, bot)]
    cv2.fillPoly(out, [np.array(pts, dtype=np.int32)], 1.0, cv2.LINE_AA)
    return out


def xray_day1_v36(rgb: np.ndarray, mono: np.ndarray, gate: np.ndarray, shorts_mask: np.ndarray, tint: str = "crimson",
                  glass_gate: np.ndarray | None = None, glass_floor: float = 0.0) -> np.ndarray:
    """
    Reusable X-Ray module = the APPROVED Day 1 v36 'Step 1 native 3D X-Ray' method
    (gymform_engine.prepare_xray_bone_frame + render_xray_skeletal_frame), generalised to any tint/gate:
      1. inside the tinted X-Ray zone the body luminance is replaced by a dark glassy cavity whose only bright
         structure is the native bone detail (xray_cavity_lum = 28 + 135 * bone_sharp^1.15) -> see-through body;
      2. bone pixels (strongly tinted + bright in the native render) are filled with the Day 1 luminous bone ramp
         (tint -> hot core) at alpha bone_mask^0.8 * 0.96, with the Day 1 0.48 X-Ray window darkening around them.
    Pure per-pixel shading of the native 3D frame: no warps, no drawn shapes. Window/bone-mask blur kept tiny.
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    if tint == "crimson":
        t_any = np.clip((r - b - 12.0) / 95.0, 0.0, 1.0)
        detail = 0.38 * r + 0.62 * g
        bone_m = np.clip((r - b - 82.0) / 95.0, 0.0, 1.0) * np.clip((r - 130.0) / 75.0, 0.0, 1.0)
    else:
        # emerald source glow is ~2x weaker than the crimson one (g-r p50~52 vs r-b p50~74): thresholds rescaled
        t_any = np.clip((g - r - 8.0) / 50.0, 0.0, 1.0)
        detail = 0.38 * g + 0.62 * r
        bone_m = np.clip((g - r - 30.0) / 40.0, 0.0, 1.0) * np.clip((g - 120.0) / 70.0, 0.0, 1.0)
    bone_sharp = np.clip((detail + (detail - cv2.GaussianBlur(detail, (0, 0), 1.5)) * 2.1 - 20.0) / 205.0, 0.0, 1.0)
    # soft, structureless glow (e.g. armpit haze) is not bone: weight the colour-based bone mask by native
    # high-frequency structure so only crisp 3D bone edges/fills light up
    hf = np.abs(detail - cv2.GaussianBlur(detail, (0, 0), 2.5))
    structured = np.clip((hf - 1.5) / 5.0, 0.0, 1.0)
    bone_m = bone_m * (0.20 + 0.80 * structured)
    # native bright bone edges inside the tinted zone also count as bone (Day 5 sources tint bones less than Day 1)
    bone_m = np.maximum(bone_m, 0.9 * np.clip(t_any * 1.6, 0.0, 1.0) * np.clip((bone_sharp - 0.50) / 0.22, 0.0, 1.0))
    if glass_gate is not None and glass_floor > 0.0 and tint != "crimson":
        # Step 3 source tints only thin bone outlines: the native 3D bone/skeletal relief inside the glass
        # window (positive high-frequency ridges of the render: clavicle, scapular spine/acromion, humeral head,
        # humerus, rib arcs) is lit as luminous bone. Per-pixel on the native frame, no drawn shapes, no blur.
        lum_n = 0.299 * r + 0.587 * g + 0.114 * b
        ridge = lum_n - cv2.GaussianBlur(lum_n, (0, 0), 3.0)
        ridge_m = np.clip((ridge - 1.5) / 6.0, 0.0, 1.0)
        bone_m = np.maximum(bone_m, ridge_m * glass_gate)
        bone_m = np.maximum(bone_m, np.clip((g - r - 14.0) / 30.0, 0.0, 1.0) * glass_gate)
        bone_sharp = np.maximum(bone_sharp, np.clip(0.72 + ridge / 25.0, 0.0, 1.0) * ridge_m * glass_gate)
        # native emerald-tinted bone (clavicle/humeral head outlines) gets the same luminous core as Step 1
        bone_sharp = np.maximum(bone_sharp, np.clip((g - r - 14.0) / 30.0, 0.0, 1.0) * 0.85 * glass_gate)
    zone2 = t_any * gate
    if glass_gate is not None and glass_floor > 0.0:
        # see-through glass body over the whole shoulder girdle + upper arm (Day 1 look), even where the
        # native emerald tint is faint; bones stay native structure only
        zone2 = np.maximum(zone2, glass_floor * glass_gate)
    zone = (zone2 * (1.0 - shorts_mask))[:, :, None]

    if glass_gate is not None and glass_floor > 0.0:
        # outside the native tint the body detail is plain skin, not bone: the glass there keeps only the native
        # surface structure (high-pass edges of the 3D render), so the body reads dark and see-through
        t_w = np.clip(t_any * 1.6, 0.0, 1.0)
        surf = np.clip((hf - 1.0) / 16.0, 0.0, 1.0) * 0.55
        bone_sharp_g = t_w * bone_sharp + (1.0 - t_w) * surf
    else:
        bone_sharp_g = bone_sharp
    cav = 28.0 + 135.0 * (bone_sharp_g ** 1.15)
    norm = (cav / 255.0) ** 0.92
    glass = np.stack([30.0 + 202.0 * norm, 32.0 + 204.0 * norm, 36.0 + 206.0 * norm], axis=-1)
    out = mono * (1.0 - zone) + glass * zone

    bm = cv2.GaussianBlur(bone_m.astype(np.float32), (0, 0), 1.0) * gate * (1.0 - shorts_mask)
    window = np.clip(cv2.GaussianBlur(bm, (0, 0), 6.0) * 1.1, 0.0, 0.65)
    out = out * (1.0 - window[:, :, None] * 0.48)

    bv = np.clip((bone_sharp - 0.12) / 0.78, 0.0, 1.0)
    hot = 110.0 * (bv ** 1.25) + 120.0 * (bv ** 2.6)
    if tint == "crimson":   # Day 1 ramp: deep crimson -> orange-gold -> white-hot core
        col = np.stack([95.0 + 160.0 * (bv ** 0.55), 14.0 + hot, 10.0 + 45.0 * (bv ** 1.5) + 135.0 * (bv ** 3.0)], axis=-1)
    else:                   # same ramp mirrored to emerald: deep green -> mint -> white core
        col = np.stack([10.0 + 45.0 * (bv ** 1.5) + 135.0 * (bv ** 3.0), 95.0 + 160.0 * (bv ** 0.55), 14.0 + hot], axis=-1)
    a = np.clip((bm ** 0.80) * 0.96, 0.0, 0.96)[:, :, None]
    out = out * (1.0 - a) + np.clip(col, 0, 255) * a
    return out


def shade_day5_v4(rgb_1080: np.ndarray, step_idx: int, is_wrong: bool, kp: dict, pulse_phase: float = 0.0):
    """Returns (composited_rgb_uint8, shorts_mask, w_fg)."""
    r_ch = rgb_1080[:, :, 0].astype(np.float32)
    g_ch = rgb_1080[:, :, 1].astype(np.float32)
    b_ch = rgb_1080[:, :, 2].astype(np.float32)
    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    min_c = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    sat = (max_c - min_c) / (max_c + 1.0)
    red_dom = np.clip(r_ch - 0.50 * (g_ch + b_ch), 0.0, 255.0)
    grn_dom = np.clip(g_ch - 0.50 * (r_ch + b_ch), 0.0, 255.0)
    act_dom = np.maximum(red_dom, grn_dom)

    yy, xx = _YY, _XX
    hip_y = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1])
    kn_y = 0.5 * (kp["kn_r"][1] + kp["kn_l"][1])
    ank_max = max(kp["ank_r"][1], kp["ank_l"][1])

    # --- clean fg mask (no act_dom bloom) ---
    sx0 = (W - int(round(W * STEP_SCALE.get(step_idx, b4.SCALE)))) // 2  # clip left edge (54 at scale 0.90)
    strips = np.concatenate([lum[:, sx0 + 16 : sx0 + 91], lum[:, W - sx0 - 91 : W - sx0 - 16]], axis=1)
    row_bg = cv2.GaussianBlur(np.percentile(strips, 35, axis=1).astype(np.float32)[:, None], (0, 0), 16.0)
    env = anatomical_envelope(kp)
    floor_gate = np.clip(((ank_max + 52.0) - yy) / 24.0, 0.0, 1.0)
    # Coloured glow over the backdrop has high act_dom but a backdrop-level min channel; tinted body pixels keep a
    # high min channel. Taking the max of both keeps the body solid while the glow fringe drops out.
    bright_sig = np.maximum(lum - 0.85 * act_dom, min_c - 6.0)
    bright_diff = np.maximum(0.0, bright_sig - row_bg) * env

    # Platform / floor spill below the ankles: only clearly bright foot pixels count there.
    ank_min = min(kp["ank_r"][1], kp["ank_l"][1])
    below = np.clip((yy - (ank_min - 45.0)) / 12.0, 0.0, 1.0)
    bright_diff = np.maximum(0.0, bright_diff - 18.0 * below)

    near_wrist = equipment_soft_zone(kp)  # equipment zones from CFG (Day 5: 140px discs at wrists, full inside 115px)
    shorts_band = np.clip((yy - (hip_y - 140.0)) / 24.0, 0.0, 1.0) * np.clip(((kn_y - 32.0) - yy) / 34.0, 0.0, 1.0)
    dark_gate = shorts_band * env  # equipment is handled by convex-hull fill below (dark_diff near wrists picked up backdrop shadows)
    dark_diff = np.maximum(0.0, row_bg - lum) * dark_gate

    fg_diff = (bright_diff + dark_diff) * floor_gate
    w_raw = np.clip((fg_diff - 8.5) / 10.5, 0.0, 1.0).astype(np.float32)

    # Matte-black dumbbells and shorts are nearly the same luminance as the source backdrop, so a pure
    # luminance key leaves them see-through. Close them with shape priors (wrist zone hole-fill, shorts polygon).
    solid = np.zeros((H, W), dtype=np.float32)
    shorts_poly = shorts_polygon(kp, (bright_diff * floor_gate) > 14.0, hip_y, kn_y)
    # Equipment footprint from a TEMPORAL clean plate: the source backdrop is static while the equipment moves,
    # so the per-step median plate (CUR_PLATE) is empty backdrop where the dumbbells are. Matte-black plates that
    # match the backdrop in luminance still differ from the plate (noise p99 ~3), giving solid, complete plates.
    # Coloured activation glow is excluded (equipment is neutral), and only configured equipment zones are used.
    foot = np.zeros((H, W), dtype=np.float32)
    plate = CUR_PLATE.get("rgb")
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for x0, y0, x1, y1, zone in equipment_zones(kp):
        if plate is None:
            break
        d = np.abs(rgb_1080[y0:y1, x0:x1].astype(np.float32) - plate[y0:y1, x0:x1]).max(axis=-1)
        # plate must be backdrop there (dark): where the plate holds the body (arm rest positions) the difference
        # is a body ghost, not equipment
        plate_lum = plate[y0:y1, x0:x1].max(axis=-1)
        # the source backdrop also carries a soft, smooth light halo around the moving arms (+10..25 vs plate):
        # equipment = differs from the plate AND (strongly, or with native 3D surface texture)
        lz = lum[y0:y1, x0:x1]
        tex = cv2.GaussianBlur(np.abs(lz - cv2.GaussianBlur(lz, (0, 0), 2.0)), (0, 0), 1.2)
        # (matte outer plates can sit only ~8-10 above the plate in max channel but keep crisp 3D texture)
        # black metal is DARKER than the backdrop plate, while the light halo is brighter: darker-than-plate
        # pixels are equipment even where the plate face is smooth
        pl_l = 0.299 * plate[y0:y1, x0:x1, 0] + 0.587 * plate[y0:y1, x0:x1, 1] + 0.114 * plate[y0:y1, x0:x1, 2]
        # (activation light tints the metal in one channel, so count channels: >=2 of R,G,B darker than plate)
        dec = plate[y0:y1, x0:x1] - rgb_1080[y0:y1, x0:x1].astype(np.float32)
        # the source backdrop is a cool blue-grey (B-R ~ +10) while metal/rubber plates are neutral (B-R ~ +2):
        # the chroma drop vs the plate outlines the full plate discs even where their luminance equals the backdrop
        cz = rgb_1080[y0:y1, x0:x1].astype(np.float32)
        # (min of the B-R and B-G drops: a red glow only lowers B-R, a green glow only lowers B-G; neutral metal
        # lowers both)
        pz = plate[y0:y1, x0:x1]
        drop_br = (pz[:, :, 2] - pz[:, :, 0]) - (cz[:, :, 2] - cz[:, :, 0])
        drop_bg = (pz[:, :, 2] - pz[:, :, 1]) - (cz[:, :, 2] - cz[:, :, 1])
        chroma_drop = cv2.GaussianBlur(np.minimum(drop_br, drop_bg), (0, 0), 1.5)
        darker = ((pl_l - lz) > 7.0) | ((dec > 5.5).sum(axis=-1) >= 2)
        # textured differences are native 3D surface (metal or glow-lit forearm/hand), never the smooth glow
        # halo, so they carry no tint limit; the smooth neutral/chroma paths keep a strict tint limit
        m = ((((d > 34.0) & (tex > 2.0)) | ((d > 11.0) & (tex > 4.0))) & (zone > 0) & (plate_lum < 95.0)) | (
              (((d > 11.0) & (act_dom[y0:y1, x0:x1] < 7.0) & (plate_lum < 52.0))
               | ((chroma_drop > 4.5) & (d > 3.0) & (act_dom[y0:y1, x0:x1] < 12.0) & (plate_lum < 52.0))) & (zone > 0))
        # darker-than-backdrop pixels are never glow (glow only brightens): no tint limit, so glow-lit dark
        # metal and the glow-lit dark underside of the hand/forearm next to the equipment stay solid
        m = (m | ((d > 6.0) & darker & (plate_lum < 52.0) & (zone > 0))).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for ci in range(1, n_c):
            if st_c[ci, cv2.CC_STAT_AREA] >= 250:
                keep[lab_c == ci] = 1
        ff = np.pad(keep, 1).copy()
        cv2.floodFill(ff, None, (0, 0), 2)
        filled = (ff[1:-1, 1:-1] != 2)
        holes = (filled & (keep == 0)).astype(np.uint8)
        n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(holes)
        for hi in range(1, n_h):  # only small enclosed holes (plate centres / handle gaps), never body windows
            if st_h[hi, cv2.CC_STAT_AREA] > 2500:
                filled[lab_h == hi] = False
        # rigid-object completion: the dark metal parts (plates) of one piece of equipment are convex in
        # silhouette, so their joint convex hull restores plate rims that fade into the backdrop. Thin glow arcs
        # are removed by an opening first; the hull is only accepted when it stays compact (not body-sized).
        metal = (filled & (lum[y0:y1, x0:x1] < 100.0)).astype(np.uint8)
        metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        n_m, lab_m, st_m, _ = cv2.connectedComponentsWithStats(metal)
        big = [mi for mi in range(1, n_m) if st_m[mi, cv2.CC_STAT_AREA] >= 1200]
        if big and CFG.get("equipment_hull", False):
            pts = np.concatenate([np.column_stack(np.where(lab_m == mi))[:, ::-1] for mi in big]).astype(np.int32)
            hull = cv2.convexHull(pts)
            hm = np.zeros_like(metal)
            cv2.fillConvexPoly(hm, hull, 1)
            area_big = sum(st_m[mi, cv2.CC_STAT_AREA] for mi in big)
            if hm.sum() <= 2.2 * area_big and hm.sum() < 60000:
                filled = filled | (hm > 0)
        # plate completion: dark metal rings (C-shaped where a matte face meets the backdrop) are closed with a
        # plate-scale kernel and their enclosed faces filled; gaps wider than the kernel (handle span between
        # left/right plate stacks, arm-to-plate) stay open, so no backdrop slab is added
        ring = (filled & (lum[y0:y1, x0:x1] - 0.85 * act_dom[y0:y1, x0:x1] < 100.0)).astype(np.uint8)
        ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        rf = np.pad(ring, 1).copy()
        cv2.floodFill(rf, None, (0, 0), 2)
        ring_h = ((rf[1:-1, 1:-1] != 2) & (ring == 0)).astype(np.uint8)
        n_r, lab_r, st_r, _ = cv2.connectedComponentsWithStats(ring_h)
        ring_f = ring > 0
        for ri in range(1, n_r):
            if st_r[ri, cv2.CC_STAT_AREA] <= 9000:
                ring_f |= lab_r == ri
        # per-plate-stack convex completion: break thin handle links, then each plate stack (disc-like blob)
        # is replaced by its convex hull when it is compact and plate-sized (not a bridge across the hand)
        rs = cv2.morphologyEx(ring_f.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n_s, lab_s, st_s, _ = cv2.connectedComponentsWithStats(rs)
        for si in range(1, n_s):
            a_s = st_s[si, cv2.CC_STAT_AREA]
            if a_s < 600:
                continue
            pts = np.column_stack(np.where(lab_s == si))[:, ::-1].astype(np.int32)
            hull = cv2.convexHull(pts)
            hm = np.zeros_like(rs)
            cv2.fillConvexPoly(hm, hull, 1)
            ha = int(hm.sum())
            if len(pts) >= 5:
                (_, _), (ea, eb), _ = cv2.fitEllipse(pts)
                aspect = max(ea, eb) / max(1.0, min(ea, eb))
            else:
                aspect = 9.0
            if ha <= 22000 and a_s >= 0.33 * ha and aspect <= 3.2:
                ring_f |= hm > 0
        # only keep completion additions that are dark (metal-like), never bright body/glow halo
        add = ring_f & ~filled & ((lum[y0:y1, x0:x1] - 0.85 * act_dom[y0:y1, x0:x1]) < 70.0)
        # completion pixels must also be neutral (metal), not the cool-blue backdrop beside the arm
        # (inside activation glow the chroma test is unreliable: there the convex plate completion is trusted)
        filled = filled | (add & ((chroma_drop > 2.5) | (act_dom[y0:y1, x0:x1] >= 12.0)))
        # difference-matte alpha (root-cause fix for bloated equipment): motion-blur smear and soft halos around a
        # moving object differ only weakly from the backdrop plate. Treating them as solid made a fast-moving
        # dumbbell render as an opaque, oversized blob. Only confident evidence (strong difference / native
        # texture / clearly darker than the backdrop, plus the faces it encloses) is solid; weak-difference
        # pixels keep a proportional alpha, so the composite reproduces the source silhouette and size.
        strong = ((d > 18.0) | ((tex > 4.0) & (d > 8.0)) | ((dec > 10.0).sum(axis=-1) >= 2)) & filled
        strong = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        sf = np.pad(strong, 1).copy()
        cv2.floodFill(sf, None, (0, 0), 2)
        s_h = ((sf[1:-1, 1:-1] != 2) & (strong == 0)).astype(np.uint8)
        n_sh, lab_sh, st_sh, _ = cv2.connectedComponentsWithStats(s_h)
        strong_f = strong > 0
        for hi in range(1, n_sh):
            if st_sh[hi, cv2.CC_STAT_AREA] <= 9000:
                strong_f |= lab_sh == hi
        alpha = np.where(strong_f, 1.0, np.clip((d - 5.0) / 15.0, 0.0, 1.0)).astype(np.float32)
        f = ((filled & (zone > 0)) * alpha).astype(np.float32)
        foot[y0:y1, x0:x1] = np.maximum(foot[y0:y1, x0:x1], f)
    # skin (hands) inside the footprint is body anyway; the equipment-dark shading only applies to metal
    solid = np.maximum(solid, foot)
    eq_solid = solid.copy()

    solid = np.maximum(solid, shorts_poly)
    w_fg = cv2.GaussianBlur(np.maximum(w_raw, solid), (0, 0), float(CFG.get("mask_feather", 0.85)))
    LAST_MASKS["eq_solid"] = eq_solid
    LAST_MASKS["eq_foot"] = foot
    LAST_MASKS["eq_zone"] = near_wrist
    LAST_MASKS["src"] = rgb_1080

    # --- opaque matte-black shorts (lum 10..22) ---
    dark_low_sat = np.clip((68.0 - lum) / 18.0, 0.0, 1.0) * np.clip((0.28 - sat) / 0.14, 0.0, 1.0) * np.clip((18.0 - act_dom) / 12.0, 0.0, 1.0)
    shorts_mask = cv2.GaussianBlur((shorts_band * dark_low_sat * (w_fg > 0.25)).astype(np.float32), (0, 0), 2.0)
    shorts_mask = np.clip(shorts_mask * 1.35, 0.0, 1.0)

    # --- bright silver-grey mannequin curve (p75~165, p90~191, max 226) ---
    skin_m = (w_fg > 0.6) & (shorts_mask < 0.2) & (act_dom < 15.0) & (lum > 42.0)
    p90_raw = float(np.percentile(lum[skin_m], 90)) if skin_m.sum() > 500 else 180.0
    hv_silver = np.clip((lum - 24.0) / max(p90_raw - 24.0, 95.0), 0.0, 1.28)
    sulc = np.clip((cv2.GaussianBlur(lum, (0, 0), 2.8) - lum) / 14.0, 0.0, 1.0)
    hf_ridge = lum - cv2.GaussianBlur(lum, (0, 0), 1.6)
    grey = np.clip(48.0 + 143.0 * (hv_silver ** 0.88) + hf_ridge * 0.70 - sulc * 11.0, 44.0, 226.0)
    # Lock skin tone to the Bright Silver-Grey standard (p75~165, p90~191, max 226) with a max-anchored remap.
    skin_q = (w_fg > 0.9) & (shorts_mask < 0.2) & (act_dom < 12.0) & (lum > 46.0)
    if step_idx in CFG["bone_steps"]:   # colour lock is defined on the body OUTSIDE the X-Ray window
        skin_q &= cv2.dilate((xray_window(kp, is_wrong) > 0.02).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
    if skin_q.sum() > 2000:
        p75_g = float(np.percentile(grey[skin_q], 75))
        k = float(np.clip((226.0 - 165.0) / max(226.0 - p75_g, 1.0), 0.90, 1.30))
        grey = np.clip(226.0 - (226.0 - grey) * k, 44.0, 226.0)
        # second anchor: highlights locked to p90~191 so every step has the same silver (colour lock)
        p90_g = float(np.percentile(grey[skin_q], 90))
        if p90_g > 170.0:
            k90 = float(np.clip((191.0 - 165.0) / (p90_g - 165.0), 0.75, 1.6))
            hi = grey > 165.0
            grey[hi] = np.clip(165.0 + (grey[hi] - 165.0) * k90, 165.0, 226.0)
    shorts_lum = np.clip(lum * 0.30 + 8.0, 10.0, 22.0)
    # Dumbbells keep a matte gunmetal-black look (dark faces stay dark instead of being lifted to grey).
    # (activation light tints the metal: use the tint-discounted luminance so glow-lit plates stay black)
    lum_eq = np.maximum(lum - 0.85 * act_dom, min_c)
    dark_t = np.clip((lum_eq - 38.0) / 40.0, 0.0, 1.0)
    db_lum = np.clip(lum_eq * 0.95 + 9.0, 12.0, 60.0)
    eq_w = cv2.GaussianBlur(eq_solid, (0, 0), 1.5)  # equipment hull only (not skin inside the zone)
    grey = grey * (1.0 - eq_w * (1.0 - dark_t)) + db_lum * eq_w * (1.0 - dark_t)
    matte = grey * (1.0 - shorts_mask) + shorts_lum * shorts_mask
    mono = np.stack([matte - 1.5, matte + 0.5, matte + 3.0], axis=-1)

    # --- step-specific anatomical gate (same as d5) ---
    sh_r, sh_l = kp["sh_r"], kp["sh_l"]
    sh_y = 0.5 * (sh_r[1] + sh_l[1])
    if step_idx in (1, 3):
        d_shr = np.hypot((xx - sh_r[0]) / 1.15, (yy - sh_r[1]) / 0.95)
        d_shl = np.hypot((xx - sh_l[0]) / 1.15, (yy - sh_l[1]) / 0.95)
        cx_clav = 0.5 * (sh_r[0] + sh_l[0])
        rx_clav = max(90.0, 0.55 * abs(sh_l[0] - sh_r[0]) + 42.0)
        d_clav = np.hypot((xx - cx_clav) / rx_clav, (yy - (sh_y + 12.0)) / 72.0)
        joint = np.maximum(np.clip((135.0 - np.minimum(d_shr, d_shl)) / 42.0, 0.0, 1.0), np.clip((1.18 - d_clav) / 0.35, 0.0, 1.0))
        anat = joint * np.clip((yy - (sh_y - 95.0)) / 28.0, 0.0, 1.0) * np.clip(((sh_y + 135.0) - yy) / 36.0, 0.0, 1.0)
    elif step_idx == 2:
        anat = np.clip((yy - (sh_y - 120.0)) / 28.0, 0.0, 1.0) * np.clip(((sh_y + 135.0) - yy) / 40.0, 0.0, 1.0)
    else:
        d_r = np.hypot((xx - (sh_r[0] - 10.0)) / 92.0, (yy - (sh_r[1] + 22.0)) / 108.0)
        d_l = np.hypot((xx - (sh_l[0] + 10.0)) / 92.0, (yy - (sh_l[1] + 22.0)) / 108.0)
        anat = np.clip((1.35 - np.minimum(d_r, d_l)) / 0.40, 0.0, 1.0)
    anat = anat * w_fg  # colour only inside the clean body mask

    w_act = cv2.GaussianBlur((np.clip((act_dom - 10.0) / 32.0, 0.0, 1.0) ** 0.85).astype(np.float32), (0, 0), 1.8) * (1.0 - shorts_mask) * anat
    core_seed = (np.clip((act_dom - 16.0) / 42.0, 0.0, 1.0) ** 1.08) * anat
    pulse = 1.0 + 0.06 * math.sin(2.0 * math.pi * pulse_phase)
    core_w = np.clip(cv2.GaussianBlur(core_seed.astype(np.float32), (0, 0), 1.0) * w_act * 0.90 * pulse, 0.0, 0.90)

    act_ch = r_ch if is_wrong else g_ch  # Step 4 source glow is green: read fibers from G (R left dark holes)
    gb_ch = 0.55 * g_ch + 0.45 * b_ch
    fiber_fine = (act_ch - cv2.GaussianBlur(act_ch, (0, 0), 4.8)) * 2.05 + (gb_ch - cv2.GaussianBlur(gb_ch, (0, 0), 4.2)) * 1.45
    fiber_bundle = (cv2.GaussianBlur(act_ch, (0, 0), 1.3) - cv2.GaussianBlur(act_ch, (0, 0), 13.0)) * 1.05
    sulc_3d = np.clip(-fiber_fine / 13.0, 0.0, 1.0)
    ridge_3d = np.clip(fiber_fine / 15.0, 0.0, 1.0)
    vol_3d = np.clip((0.58 * act_ch + 0.42 * gb_ch) / 170.0, 0.18, 1.20)
    fiber_mod = np.clip(vol_3d + 0.54 * ridge_3d - 0.58 * sulc_3d + 0.004 * fiber_bundle, 0.12, 1.42)

    bone_edge = np.clip(lum - cv2.GaussianBlur(lum, (0, 0), 5.0), -24.0, 38.0)
    bone_lum = np.clip(cv2.GaussianBlur(lum, (0, 0), 0.6) + 0.85 * bone_edge, 0.0, 255.0)
    L_bone = np.clip((bone_lum - 24.0) / 140.0, 0.0, 1.0)
    bone_core = (np.clip((bone_lum - 115.0) / 55.0, 0.0, 1.0) * np.clip((act_dom - 12.0) / 26.0, 0.0, 1.0) * anat) if step_idx in (1, 3) else 0.0

    if is_wrong:
        if step_idx == 1:
            col_r = np.clip(bone_lum * 1.45 + 38.0 * (L_bone ** 0.85), 22.0, 242.0)
            col_g = np.clip(bone_lum * 0.32, 10.0, 155.0)
            col_b = np.clip(bone_lum * 0.34, 12.0, 142.0)
        else:
            col_r = np.clip(36.0 + 174.0 * fiber_mod + 52.0 * (ridge_3d ** 1.25) - 46.0 * sulc_3d, 22.0, 245.0)
            col_g = np.clip(14.0 + 40.0 * fiber_mod + 28.0 * (ridge_3d ** 1.5) - 18.0 * sulc_3d, 10.0, 102.0)
            col_b = np.clip(16.0 + 42.0 * fiber_mod + 26.0 * (ridge_3d ** 1.5) - 18.0 * sulc_3d, 10.0, 102.0)
    else:
        if step_idx == 3:
            col_g = np.clip(bone_lum * 1.45 + 36.0 * (L_bone ** 0.85), 22.0, 242.0)
            col_r = np.clip(bone_lum * 0.32, 10.0, 138.0)
            col_b = np.clip(bone_lum * 0.76, 14.0, 185.0)
        else:
            col_r = np.clip(14.0 + 36.0 * fiber_mod + 30.0 * (ridge_3d ** 1.5) - 14.0 * sulc_3d, 10.0, 94.0)
            col_g = np.clip(34.0 + 180.0 * fiber_mod + 58.0 * (ridge_3d ** 1.25) - 48.0 * sulc_3d, 24.0, 248.0)
            col_b = np.clip(22.0 + 96.0 * fiber_mod + 42.0 * (ridge_3d ** 1.35) - 28.0 * sulc_3d, 16.0, 170.0)

    act_rgb = np.stack([col_r, col_g, col_b], axis=-1)
    if step_idx == 1:
        # Step 1 = WRONG-posture BONE X-Ray: keep only crisp bone structure (clavicle, acromion/scapula, humeral head),
        # suppress the broad red soft-tissue tint.
        bone_lines = np.clip((bone_edge - 4.0) / 10.0, 0.0, 1.0) * np.clip((bone_lum - 140.0) / 40.0, 0.0, 1.0) * np.clip((act_dom - 10.0) / 25.0, 0.0, 1.0) * anat * (1.0 - shorts_mask)
        core_w = np.clip(0.15 * core_w + 0.90 * bone_lines, 0.0, 0.92)
        # tight gate: clavicle line + acromion / humeral-head discs only (no pec / trap soft tissue)
        d_j = np.minimum(np.hypot(xx - sh_r[0], yy - sh_r[1]), np.hypot(xx - sh_l[0], yy - sh_l[1]))
        clav_d = np.hypot((xx - 0.5 * (sh_r[0] + sh_l[0])) / max(60.0, 0.55 * abs(sh_l[0] - sh_r[0])), (yy - (sh_y + 6.0)) / 34.0)
        tight = np.maximum(np.clip((92.0 - d_j) / 24.0, 0.0, 1.0), np.clip((1.05 - clav_d) / 0.25, 0.0, 1.0))
        core_w = core_w * tight
        bone_core = bone_core * tight
    if step_idx == 3:
        # Step 3 source glow is faint: lift the crisp green bone X-Ray so clavicle/scapula/humeral head read clearly.
        bone_w3 = np.clip((act_dom - 6.0) / 18.0, 0.0, 1.0) * anat * (1.0 - shorts_mask)
        core_w = np.maximum(core_w, 0.80 * cv2.GaussianBlur(bone_w3.astype(np.float32), (0, 0), 0.8))
        bone_core = np.clip(1.8 * bone_core, 0.0, 1.0)
    blend_w = np.clip(core_w + 0.45 * bone_core, 0.0, 0.92) * (1.0 - shorts_mask)
    mannequin = mono * (1.0 - blend_w[:, :, None]) + act_rgb * blend_w[:, :, None]
    if step_idx in CFG["bone_steps"] and CFG.get("xray_style") == "day1_v36":
        # Mannequin colour is LOCKED across all steps: only a local see-through X-Ray window around the target
        # bones becomes translucent (Day 1 style); the rest of the body keeps the silver-grey standard.
        win = xray_window(kp, is_wrong) * w_fg
        # equipment is never part of the X-Ray window (no glassy/ghosted dumbbells)
        win = win * (1.0 - cv2.GaussianBlur(cv2.dilate(eq_solid, np.ones((9, 9), np.uint8)), (0, 0), 2.0))
        LAST_MASKS["xwin"] = win
        if is_wrong:
            mannequin_x = xray_day1_v36(rgb_1080, mono, anat * win, shorts_mask, tint="crimson")
        else:
            mannequin_x = xray_day1_v36(rgb_1080, mono, np.maximum(anat, win) * win, shorts_mask, tint="emerald",
                                        glass_gate=win, glass_floor=0.80)
        wv = win[:, :, None]
        mannequin = mono * (1.0 - wv) + mannequin_x * wv
    else:
        LAST_MASKS["xwin"] = np.zeros((H, W), np.float32)

    bg = studio_bg() * (1.0 - 0.80 * contact_shadow(kp)[:, :, None])
    wf = w_fg[:, :, None]
    out = bg * (1.0 - wf) + mannequin * wf
    return np.clip(out, 0, 255).astype(np.uint8), shorts_mask, w_fg


# ---------------------------------------------------------------------------------------------
# 3. Crisp laser renderer (replaces p62.draw_demic_laser_joint_axis for this build only)
# ---------------------------------------------------------------------------------------------
def draw_crisp_laser_joint_axis(canvas, main_chains_1x, aux_ext_chains_1x, vertex_pts_1x, is_wrong):
    all_pts = [pt for ch in (main_chains_1x + aux_ext_chains_1x) for pt in ch] + list(vertex_pts_1x)
    if not all_pts:
        return
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    pad = 44
    rx0 = max(0, int(min(xs)) - pad); ry0 = max(0, int(min(ys)) - pad)
    rx1 = min(W, int(max(xs)) + pad); ry1 = min(H, int(max(ys)) + pad)
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw <= 4 or rh <= 4:
        return
    sh = lambda chains: [[(px - rx0, py - ry0) for (px, py) in ch] for ch in chains]  # noqa: E731
    m_loc, a_loc = sh(main_chains_1x), sh(aux_ext_chains_1x)
    v_loc = [(vx - rx0, vy - ry0) for (vx, vy) in vertex_pts_1x]
    col_halo = (255, 70, 72) if is_wrong else (52, 255, 140)
    col_core = (255, 236, 234) if is_wrong else (226, 255, 238)
    taper = p62._draw_tapered_laser_polyline_ss

    glow = Image.new("RGBA", (rw, rh), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    for ch in m_loc:
        taper(gd, ch, base_w_1x=9.0, col_rgba=(*col_halo, 45), ss=1, fade_start=0.15, fade_end=0.15)
    glow_arr = cv2.GaussianBlur(np.array(glow), (0, 0), 10.0)

    halo = Image.new("RGBA", (rw, rh), (0, 0, 0, 0)); hd = ImageDraw.Draw(halo)
    for ch in m_loc:
        taper(hd, ch, base_w_1x=6.0, col_rgba=(*col_halo, 150), ss=1, fade_start=0.15, fade_end=0.15)
    for ch in a_loc:
        taper(hd, ch, base_w_1x=4.0, col_rgba=(*col_halo, 110), ss=1, fade_start=0.20, fade_end=0.22)
    halo_arr = cv2.GaussianBlur(np.array(halo), (0, 0), 3.5)

    SS = 3
    core = Image.new("RGBA", (rw * SS, rh * SS), (0, 0, 0, 0)); cd = ImageDraw.Draw(core)
    for ch in a_loc:
        taper(cd, ch, base_w_1x=2.2, col_rgba=(*col_halo, 210), ss=SS, fade_start=0.20, fade_end=0.22)
        taper(cd, ch, base_w_1x=1.1, col_rgba=(*col_core, 230), ss=SS, fade_start=0.20, fade_end=0.22)
    for ch in m_loc:
        taper(cd, ch, base_w_1x=2.8, col_rgba=(*col_halo, 255), ss=SS, fade_start=0.15, fade_end=0.15)
        taper(cd, ch, base_w_1x=1.4, col_rgba=(*col_core, 255), ss=SS, fade_start=0.15, fade_end=0.15)
    for vx, vy in v_loc:  # small crisp vertex rings
        r_o = 6.5 * SS
        cd.ellipse([vx * SS - r_o, vy * SS - r_o, vx * SS + r_o, vy * SS + r_o], outline=(*col_halo, 255), width=int(1.8 * SS))
        r_i = 2.4 * SS
        cd.ellipse([vx * SS - r_i, vy * SS - r_i, vx * SS + r_i, vy * SS + r_i], fill=(*col_core, 255))
    core_arr = cv2.resize(np.array(core), (rw, rh), interpolation=cv2.INTER_AREA)

    roi = canvas.crop((rx0, ry0, rx1, ry1))
    for arr in (glow_arr, halo_arr, core_arr):
        roi.alpha_composite(Image.fromarray(arr))
    canvas.paste(roi, (rx0, ry0))


# d5.draw_day5_laser_and_callout looks the laser function up on the p62 module at call time;
# patch it in-process only (files on disk are untouched).
p62.draw_demic_laser_joint_axis = draw_crisp_laser_joint_axis


# ---------------------------------------------------------------------------------------------
# 4. Title depth-behind + top preview that never frosts the mannequin
# ---------------------------------------------------------------------------------------------
def composite_title_and_preview_v4(base_rgb, fg_mask, torso_icons, active_slot, is_wrong, exercise_name):
    # Title: identical to b4 (depth-behind with fg_front = 0.92 * fg)
    out = b4.composite_depth_behind_title_and_translucent_preview(
        base_rgb, fg_mask, torso_icons, active_slot=active_slot, is_wrong=is_wrong, exercise_name=exercise_name
    )
    # Preview ROI: wherever the mannequin/dumbbells touch the pill, show them sharp (pre-preview pixels with title).
    ty0, ty1 = 410, 565
    titled = base_rgb.copy()
    t_rgb, t_a = b4._TITLE_ROI_CACHE[exercise_name]
    eff_a = t_a * (1.0 - np.clip(fg_mask[ty0:ty1, :, None] * 0.92, 0.0, 0.92))
    titled[ty0:ty1] = np.clip(titled[ty0:ty1].astype(np.float32) * (1.0 - eff_a) + t_rgb * eff_a, 0, 255).astype(np.uint8)
    py0, py1 = 230, 425
    touch = np.clip(fg_mask[py0:py1, :, None] * 1.15, 0.0, 0.92)
    eqm = LAST_MASKS.get("eq_front")
    if eqm is not None:  # equipment renders fully in front of the preview pill (no preview pixels on it)
        touch = np.maximum(touch, eqm[py0:py1, :, None])
    out[py0:py1] = np.clip(out[py0:py1].astype(np.float32) * (1.0 - touch) + titled[py0:py1].astype(np.float32) * touch, 0, 255).astype(np.uint8)
    return out


def fg_for_preview(w_fg: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(np.clip((w_fg - 0.05) / 0.6, 0.0, 1.0).astype(np.float32), (0, 0), 1.2)


# ---------------------------------------------------------------------------------------------
# 5. Build (tempo / slices / callouts / TTS / 4-slot preview identical to d5.build_all)
# ---------------------------------------------------------------------------------------------
def _callout_for(step_idx, pose_frac):
    if step_idx == 1:
        return 0, f"{int(round(18 + 100 * pose_frac))}°", "HIGH FLARE", f"{int(round(76 + 12 * pose_frac))}% JOINT PINCH"
    if step_idx == 2:
        return 1, f"{int(round(8 + 20 * pose_frac))}°", "TRAP SHRUG", f"{int(round(68 + 8 * pose_frac))}% TRAP OVERLOAD"
    if step_idx == 3:
        return 2, "0°", "SCAPULA LOCK", f"{int(round(90 + 6 * pose_frac))}% JOINT SAFE"
    return 3, f"{int(round(12 + 18 * pose_frac))}°", "SCAPULAR PLANE", f"{int(round(91 + 7 * pose_frac))}% LATERAL DELT"


def _tempo_wave(u):
    if u < 0.06:
        return 0.0
    if u < 0.36:
        return 0.5 - 0.5 * math.cos(math.pi * ((u - 0.06) / 0.30))
    if u < 0.62:
        return 1.0
    if u < 0.92:
        return 0.5 + 0.5 * math.cos(math.pi * ((u - 0.62) / 0.30))
    return 0.0


def draw_laser_and_callout_v4(base_rgb, shorts_mask_1080, step_idx, kp, is_wrong, deg_str, l1_str, l2_str):
    """d5.draw_day5_laser_and_callout with an optional fixed text block per step (CFG['callout_ty'])."""
    ov = CFG.get("callout_ty", {}).get(step_idx)
    if ov is None:
        return d5.draw_day5_laser_and_callout(base_rgb, shorts_mask_1080, step_idx, kp, is_wrong, deg_str, l1_str, l2_str)
    # laser only (empty callout strings), then our own text block + leader
    canvas = Image.fromarray(base_rgb).convert("RGBA")
    sh_r, sh_l, elb_r, wr_r = kp["sh_r"], kp["sh_l"], kp["elb_r"], kp["wr_r"]
    elb_l, wr_l = kp["elb_l"], kp["wr_l"]
    _, wr_r_ext = p62._extend_segment(elb_r, wr_r, 0.0, 1.12)
    _, wr_l_ext = p62._extend_segment(elb_l, wr_l, 0.0, 1.12)
    sh_r_ext, sh_l_ext = p62._extend_segment(sh_r, sh_l, -0.08, 1.08)
    clav_mid = (0.52 * sh_r[0] + 0.48 * sh_l[0], 0.52 * sh_r[1] + 0.48 * sh_l[1] + 4.0)
    if step_idx == 2:   # identical laser chains / targets to d5.draw_day5_laser_and_callout
        p62.draw_demic_laser_joint_axis(canvas, main_chains_1x=[[sh_r, elb_r, wr_r_ext], [sh_l, elb_l, wr_l_ext]],
                                        aux_ext_chains_1x=[[sh_r_ext, sh_l_ext]], vertex_pts_1x=[sh_r, elb_r, sh_l, elb_l],
                                        is_wrong=is_wrong)
        target_pt = (int(0.65 * sh_r[0] + 0.35 * sh_l[0]), int(sh_r[1] - 22))
    else:
        p62.draw_demic_laser_joint_axis(canvas, main_chains_1x=[[sh_r, elb_r, wr_r_ext]], aux_ext_chains_1x=[[sh_r_ext, clav_mid]],
                                        vertex_pts_1x=[sh_r, elb_r], is_wrong=is_wrong)
        if step_idx in (1, 3):
            target_pt = (int(sh_r[0] + 8), int(sh_r[1] + 6))
        else:
            target_pt = (int(0.76 * sh_r[0] + 0.24 * elb_r[0]), int(0.76 * sh_r[1] + 0.24 * elb_r[1] + 10))
    laser_rgb = np.array(canvas.convert("RGB")).astype(np.float32)
    g = np.clip(shorts_mask_1080 * 0.90, 0.0, 0.90)[:, :, None]
    blended = np.clip(laser_rgb * (1.0 - g) + base_rgb.astype(np.float32) * g, 0, 255).astype(np.uint8)
    base_pil = Image.fromarray(blended).convert("RGBA")
    ovl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ovl)
    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    tx, ty = 54, int(ov)
    elbow_y = ty + 76
    elbow_x = max(254, min(target_pt[0] - 40, 300))
    d5.bcd._draw_precision_leader(rd, target_pt, (elbow_x, elbow_y), (242, elbow_y), col)
    rd.text((tx + 2, ty + 3), deg_str, font=v30.f_pretendard(84, "extrabold"), fill=(0, 0, 0, 210))
    rd.text((tx, ty), deg_str, font=v30.f_pretendard(84, "extrabold"), fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 90), l1_str, font=v30.f_pretendard(21, "extrabold"), fill=(*col, 255))
    rd.text((tx + 2, ty + 117), l2_str, font=v30.f_pretendard(16, "bold"), fill=(215, 222, 232, 235))
    return np.array(Image.alpha_composite(base_pil, ovl).convert("RGB"))


def render_pose(step_idx, seq, kps, p_i, icons, qc=False):
    fn, i0, i1, is_w, v_mode, y_sh = d5.STEP_SPECS_DAY5[step_idx]
    n_f = len(seq)
    pose_frac = p_i / float(max(1, n_f - 1))
    if CUR_PLATE.get("step") != step_idx:
        set_step_plate(seq, kps, step_idx)
    rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(seq[p_i], kps[p_i], y_sh, step_idx)
    base_rgb, sh_mask, w_fg = shade_day5_v4(rgb_1080, step_idx, is_w, kp_1080, pulse_phase=pose_frac * 0.75)
    slot_i, deg_s, l1_s, l2_s = _callout_for(step_idx, pose_frac)
    fr = draw_laser_and_callout_v4(base_rgb, sh_mask, step_idx, kp_1080, is_w, deg_s, l1_s, l2_s)
    LAST_MASKS["pre_ui"] = fr
    LAST_MASKS["eq_front"] = cv2.GaussianBlur(cv2.dilate(LAST_MASKS["eq_foot"], np.ones((5, 5), np.uint8)), (0, 0), 1.0)
    fr = composite_title_and_preview_v4(fr, fg_for_preview(w_fg), icons, slot_i, is_w, CFG["exercise_name"])
    if qc:
        return fr, base_rgb, sh_mask, w_fg, kp_1080
    return fr


def load_inputs():
    raw_seqs, kps_seqs = {}, {}
    with open(d5.CACHE_KP_PKL, "rb") as f:
        kps_seqs = pickle.load(f)
    for s_idx, (fn, i0, i1, is_w, v_mode, y_sh) in d5.STEP_SPECS_DAY5.items():
        seq = d5.load_native_slice_720p(os.path.join(ASSETS_DIR, fn), i0, i1)
        raw_seqs[s_idx] = seq
        if s_idx not in kps_seqs or len(kps_seqs[s_idx]) != len(seq):
            kps_seqs[s_idx] = d5.extract_day5_kps_1080(seq, s_idx, y_sh)
    return raw_seqs, kps_seqs


def build_icons(raw_seqs, kps_seqs):
    icons = []
    for s_idx in [1, 2, 3, 4]:
        fn, i0, i1, is_w, v_mode, y_sh = d5.STEP_SPECS_DAY5[s_idx]
        set_step_plate(raw_seqs[s_idx], kps_seqs[s_idx], s_idx)
        rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(raw_seqs[s_idx][-1], kps_seqs[s_idx][-1], y_sh, s_idx)
        sh_rgb, sh_m, w_fg = shade_day5_v4(rgb_1080, s_idx, is_w, kp_1080, pulse_phase=0.75)
        lum = 0.299 * sh_rgb[:, :, 0] + 0.587 * sh_rgb[:, :, 1] + 0.114 * sh_rgb[:, :, 2]
        f = sh_rgb.astype(np.float32)
        act_d = np.maximum(np.clip(f[:, :, 0] - 0.5 * (f[:, :, 1] + f[:, :, 2]), 0, 255), np.clip(f[:, :, 1] - 0.5 * (f[:, :, 0] + f[:, :, 2]), 0, 255))
        skin = (w_fg > 0.9) & (sh_m < 0.2) & (act_d < 12.0) & (lum > 46.0)
        shorts = (sh_m > 0.6) & (w_fg > 0.9)
        print(f"   Step {s_idx} skin p50={np.percentile(lum[skin], 50):.1f} p75={np.percentile(lum[skin], 75):.1f} "
              f"p90={np.percentile(lum[skin], 90):.1f} max={lum[skin].max():.1f} | shorts p50={np.percentile(lum[shorts], 50):.1f}", flush=True)
        icons.append(b4._make_icon_pair(sh_rgb, kp_1080))
    return icons


def make_sheet(peaks: dict):
    pw, ph, pad, header_h = 520, 924, 22, 96
    sheet = Image.new("RGB", (pad + 4 * (pw + pad), header_h + ph + pad), (14, 16, 20))
    sd = ImageDraw.Draw(sheet)
    sd.text((28, 18), "Day 5 Dumbbell Lateral Raise — V4 Private Gym Studio · Clean Mask (zero bloom) · Crisp Laser",
            font=v30.f_pretendard(26, "extrabold"), fill=(242, 246, 252))
    sd.text((28, 54), "Step 1: 오자세 뼈 X-Ray | Step 2: 오자세 승모 근육결 | Step 3: 정자세 뼈 X-Ray | Step 4: 정자세 측면삼각근 근육결",
            font=v30.f_pretendard(18, "bold"), fill=(185, 225, 255))
    titles = [(1, True, "Step 1: 오자세 뼈 X-Ray (HIGH FLARE)"), (2, True, "Step 2: 오자세 근육결 (TRAP SHRUG)"),
              (3, False, "Step 3: 정자세 뼈 X-Ray (SCAPULA LOCK)"), (4, False, "Step 4: 정자세 근육결 (SCAPULAR PLANE)")]
    for i, (s_i, is_w, lbl) in enumerate(titles):
        x0, y0 = pad + i * (pw + pad), header_h
        sheet.paste(peaks[s_i].resize((pw, ph), Image.Resampling.LANCZOS), (x0, y0))
        col = (245, 65, 65) if is_w else (35, 235, 130)
        sd.rectangle([x0 - 2, y0 - 2, x0 + pw + 1, y0 + ph + 1], outline=col, width=3)
    sheet.save(OUT_SHEET, quality=94)
    return OUT_SHEET


def build_peaks_only():
    raw_seqs, kps_seqs = load_inputs()
    icons = build_icons(raw_seqs, kps_seqs)
    peaks = {}
    for s in [1, 2, 3, 4]:
        n_f = len(raw_seqs[s])
        p_i = int(round(_tempo_wave(50 / 150.0) * (n_f - 1)))
        fr, base, shm, w_fg, kp = render_pose(s, raw_seqs[s], kps_seqs[s], p_i, icons, qc=True)
        Image.fromarray(fr).save(os.path.join(TMP_DIR, f"qc_step{s}_peak.jpg"), quality=93)
        cv2.imwrite(os.path.join(TMP_DIR, f"qc_step{s}_wfg.png"), (w_fg * 255).astype(np.uint8))
        peaks[s] = Image.fromarray(fr)
        # halo check: coloured pixels outside the body mask
        f = base.astype(np.float32)
        act = np.maximum(f[:, :, 0] - 0.5 * (f[:, :, 1] + f[:, :, 2]), f[:, :, 1] - 0.5 * (f[:, :, 0] + f[:, :, 2]))
        bgf = studio_bg()
        bact = np.maximum(bgf[:, :, 0] - 0.5 * (bgf[:, :, 1] + bgf[:, :, 2]), bgf[:, :, 1] - 0.5 * (bgf[:, :, 0] + bgf[:, :, 2]))
        outside = w_fg < 0.02
        print(f"   QC step {s}: p_i={p_i} coloured px outside body (act>{20}) = {int(((act - bact) > 20)[outside].sum())} | "
              f"wrists y={kp['wr_r'][1]:.0f}/{kp['wr_l'][1]:.0f} ankles y={kp['ank_r'][1]:.0f}/{kp['ank_l'][1]:.0f}", flush=True)
    for s in [1, 2, 3, 4]:
        fr = render_pose(s, raw_seqs[s], kps_seqs[s], 0, icons)
        Image.fromarray(fr).save(os.path.join(TMP_DIR, f"qc_step{s}_start.jpg"), quality=93)
    make_sheet(peaks)
    print("QC peaks + sheet written to", TMP_DIR, OUT_SHEET)


# QC sidecar frames per step (f_local -> tag): start/peak feed every check; the extra mid-motion frames feed the
# per-frame equipment checks (d)/(h) so equipment is verified across the whole movement, not just two poses
QC_TAGS = {4: "start", 20: "t20", 35: "t35", 50: "peak", 70: "t70", 100: "t100", 130: "t130"}


def save_qc_sidecar(step_idx, tag, frame_idx, fr, base, shorts_m, w_fg, kp):
    """Per-step QC sidecar consumed by qc_gate_gymform.py (masks + pre-UI base + keypoints)."""
    import json
    d = CFG["qc_dir"]
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(d, f"step{step_idx}_{tag}")
    cv2.imwrite(pre + "_base.png", cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
    cv2.imwrite(pre + "_wfg.png", (np.clip(w_fg, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(pre + "_shorts.png", (np.clip(shorts_m, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(pre + "_eq.png", (np.clip(LAST_MASKS.get("eq_solid", np.zeros_like(w_fg)), 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(pre + "_eqfoot.png", (np.clip(LAST_MASKS.get("eq_foot", np.zeros_like(w_fg)), 0, 1) * 255).astype(np.uint8))
    # independent inputs for the QC's own expected-equipment detector: the raw (transformed) source frame and
    # the step's temporal backdrop plate
    cv2.imwrite(pre + "_src.png", cv2.cvtColor(LAST_MASKS["src"], cv2.COLOR_RGB2BGR))
    plate_p = os.path.join(d, f"step{step_idx}_plate.png")
    if not os.path.exists(plate_p) or CUR_PLATE.get("saved") != (d, step_idx):
        cv2.imwrite(plate_p, cv2.cvtColor(np.clip(CUR_PLATE["rgb"], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        CUR_PLATE["saved"] = (d, step_idx)
    cv2.imwrite(pre + "_preui.png", cv2.cvtColor(LAST_MASKS["pre_ui"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(pre + "_xwin.png", (np.clip(LAST_MASKS.get("xwin", np.zeros_like(w_fg)), 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(pre + "_eqzone.png", (np.clip(LAST_MASKS.get("eq_zone", np.zeros_like(w_fg)), 0, 1) * 255).astype(np.uint8))
    meta_p = os.path.join(d, f"meta_step{step_idx}.json")   # per-step (parallel workers); merged by merge_qc_meta
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.setdefault("frames", {})[f"step{step_idx}_{tag}"] = {"frame_idx": frame_idx, "step": step_idx, "tag": tag, "kp": kp}
    meta.update({
        "exercise_name": CFG["exercise_name"], "studio_bg": CFG["studio_bg"], "equipment": CFG["equipment"],
        "bone_steps": list(CFG["bone_steps"]), "muscle_steps": list(CFG["muscle_steps"]),
        "title_roi": [0, 410, W, 565], "preview_roi": [190, 232, 890, 424], "fps": FPS, "qc_tags": QC_TAGS,
    })
    json.dump(meta, open(meta_p, "w"), indent=1)


N_WORKERS = int(os.environ.get("GYMFORM_WORKERS", "2"))      # modest: Shorts Manager has box priority
X264_PRESET = os.environ.get("GYMFORM_X264_PRESET", "veryfast")
_G = {}


def render_step_to_mp4(step_idx):
    """Render one step (150 frames, unique poses cached) straight into an x264 pipe. Runs in a worker."""
    cv2.setNumThreads(1)
    seq, kps = _G["raw"][step_idx], _G["kps"][step_idx]
    icons = _G["icons"][step_idx]
    n_f = len(seq)
    out = os.path.join(TMP_DIR, f"_step{step_idx}.mp4")
    ff = subprocess.Popen(["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", X264_PRESET,
                           "-crf", "17", "-pix_fmt", "yuv420p", "-threads", "1", out], stdin=subprocess.PIPE)
    cache, peak_p = {}, os.path.join(TMP_DIR, f"_peak{step_idx}.png")
    for f_local in range(150):
        p_i = int(round(_tempo_wave(f_local / 150.0) * (n_f - 1)))
        if f_local in QC_TAGS:
            fr_q, base_q, shm_q, wfg_q, kp_q = render_pose(step_idx, seq, kps, p_i, icons, qc=True)
            save_qc_sidecar(step_idx, QC_TAGS[f_local], (step_idx - 1) * 150 + f_local, fr_q, base_q, shm_q, wfg_q, kp_q)
            cache[p_i] = fr_q
        if p_i not in cache:
            cache[p_i] = render_pose(step_idx, seq, kps, p_i, icons)
        fr = cache[p_i]
        if f_local == 50:
            Image.fromarray(fr).save(peak_p)
        ff.stdin.write(np.ascontiguousarray(fr).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"   Step {step_idx}/4 done ({len(cache)} unique poses)", flush=True)
    return step_idx, peak_p


def merge_qc_meta():
    import json
    d = CFG["qc_dir"]
    meta = {}
    for s in [1, 2, 3, 4]:
        mp_ = os.path.join(d, f"meta_step{s}.json")
        if os.path.exists(mp_):
            m = json.load(open(mp_))
            frames = meta.get("frames", {})
            frames.update(m.get("frames", {}))
            meta.update(m)
            meta["frames"] = frames
            os.remove(mp_)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)


def build_all():
    print("1. Loading Day 5 slices + cached keypoints...", flush=True)
    raw_seqs, kps_seqs = load_inputs()
    print("2. Building 4-slot preview icons (studio composite) + luminance audit...", flush=True)
    icons = build_icons(raw_seqs, kps_seqs)
    icons_for_step = {1: icons, 2: icons, 3: icons, 4: icons}

    print("3. TTS (identical lines / timing to d5)...", flush=True)
    wav_path = os.path.join(TMP_DIR, "day5_v4_audio.wav")
    vo_lines = [
        (0.20, "Dumbbell lateral raise. Avoid flaring straight out and pinching your shoulder joint."),
        (5.20, "Shrugging the weights up overloads your upper traps instead of your side delts."),
        (10.20, "Depress your shoulder blades and lock your shoulder joint in a safe groove."),
        (15.20, "Raise thirty degrees forward in the scapular plane to fire your lateral deltoid fibers."),
    ]
    import threading
    t_tts = threading.Thread(target=lambda: asyncio.run(d5.synthesize_day5_voiceover_wav(wav_path, vo_lines, DURATION)))
    t_tts.start()   # TTS runs while frames render

    print(f"4. Rendering 600 frames ({N_WORKERS} step workers, x264 {X264_PRESET})...", flush=True)
    t_r = time.time()
    steps = [1, 2, 3, 4]
    # step workers = independent subprocesses (no fork-with-threads deadlocks); icons shared via a pickle
    with open(os.path.join(TMP_DIR, "_icons.pkl"), "wb") as f:
        pickle.dump(icons, f)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    pending, running, res = list(steps), [], []
    while pending or running:
        while pending and len(running) < N_WORKERS:
            s_ = pending.pop(0)
            lf = open(os.path.join(TMP_DIR, f"_step{s_}.log"), "w")
            running.append((s_, subprocess.Popen(["nice", "-n", "10", sys.executable, os.path.abspath(__file__), "--render-step", str(s_)],
                                                 stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
        time.sleep(1.0)
        for item in list(running):
            s_, pr, lf = item
            if pr.poll() is not None:
                lf.close()
                running.remove(item)
                if pr.returncode != 0:
                    raise RuntimeError(f"step {s_} worker failed (see {TMP_DIR}/_step{s_}.log)")
                print(f"   Step {s_}/4 done", flush=True)
                res.append((s_, os.path.join(TMP_DIR, f"_peak{s_}.png")))
    peaks = {s: Image.open(pk) for s, pk in sorted(res)}
    peaks[4].save(OUT_COVER, quality=95)
    merge_qc_meta()
    print(f"   frames rendered in {time.time() - t_r:.1f}s", flush=True)
    t_tts.join()

    print("5. Concat + mux AAC (stream copy video)...", flush=True)
    lst = os.path.join(TMP_DIR, "_steps.txt")
    with open(lst, "w") as f:
        for s in steps:
            f.write(f"file '{os.path.join(TMP_DIR, f'_step{s}.mp4')}'\n")
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-y", "-threads", "2", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav_path,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT_MP4],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for s in steps:
        os.remove(os.path.join(TMP_DIR, f"_step{s}.mp4"))
    os.remove(lst)
    make_sheet(peaks)
    print("Saved:", OUT_MP4, OUT_SHEET, OUT_COVER, flush=True)


if __name__ == "__main__":
    if "--render-step" in sys.argv:
        _s = int(sys.argv[sys.argv.index("--render-step") + 1])
        cv2.setNumThreads(1)
        _raw, _kps = load_inputs()
        _ic = pickle.load(open(os.path.join(TMP_DIR, "_icons.pkl"), "rb"))
        _G.update({"raw": _raw, "kps": _kps, "icons": {k: _ic for k in (1, 2, 3, 4)}})
        render_step_to_mp4(_s)
        sys.exit(0)
    if "--peaks" in sys.argv:
        build_peaks_only()
    else:
        build_all()
