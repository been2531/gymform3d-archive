#!/usr/bin/env python3
"""
Day 1 — v24 Posture-Diff & Lower-Left UI/UX Concept Generator
Fixes:
1. 100% removes the lower-left artificial black darkening (`ll_dim` removed) so the cable machine base
   and studio floor lighting look 100% natural.
2. Makes Step 2 (`✕ ELBOW SWING`) vs Step 3 (`✓ PIN ELBOWS`) posture difference unmistakable:
   - True biomechanical shoulder-hinge rotation swings the upper arm & forearm forward by +24° (68px forward)
     with forward torso lean in Step 2 (`✕`), plus an on-body kinematic angle arc (`+24° SWING` vs `0° LOCKED`).
3. Places the `✕ / ✓` badge ONLY at the top-right (next to the 3-torso dock, matching `@demicstory`).
4. Renders 3 distinct, world-class Lower-Left UI/UX Concepts (A, B, C) for user selection:
   - Concept A: "Dual-Bar Versus Gauge" (TRICEPS % vs FRONT DELT % instant visual comparison bar)
   - Concept B: "Radial Arc Ring + Floating Pill" (Frameless 270° circular dial + compact 2-line capsule)
   - Concept C: "3-Head Anatomy Equalizer" (3 side-by-side anatomical head meters + 1-line coaching cue)
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
BASE_DIR = "/workspace/daily_info_reels"
OUT_DIR = os.path.join(BASE_DIR, "output")

CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")

BG_CHARCOAL = (14, 16, 20)
COL_RED = (255, 62, 50)
COL_AMBER = (255, 152, 48)
COL_GREEN = (24, 226, 105)
COL_CYAN = (110, 235, 255)


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    paths = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def place_in_studio_canvas_clean(
    mono_res: np.ndarray,
    extra_maps: dict[str, np.ndarray],
    scale: float = 0.84,
    x_shift: int = 34,
    y_shift: int = 50,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Places the resized frame into the 1080x1920 studio canvas with ZERO artificial black patch
    on the bottom-left (`ll_dim` completely removed!).
    """
    target_h, target_w = mono_res.shape[:2]
    x0 = max(0, min(W - target_w, (W - target_w) // 2 + x_shift))
    y0 = max(0, min(H - target_h, (H - target_h) // 2 + y_shift))
    pad_top = y0
    pad_bot = H - (y0 + target_h)
    pad_left = x0
    pad_right = W - (x0 + target_w)

    full_mono = cv2.copyMakeBorder(
        mono_res.astype(np.uint8),
        pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_REPLICATE,
    )

    small_bg = cv2.resize(full_mono, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small_bg = cv2.GaussianBlur(small_bg, (0, 0), sigmaX=6.0, sigmaY=6.0)
    bg_blur = cv2.resize(small_bg, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    vig = 1.0 - 0.28 * (((yy - H * 0.50) / (H * 0.55)) ** 2 + ((xx - W * 0.54) / (W * 0.60)) ** 2)
    vig = np.clip(vig, 0.56, 1.0)[:, :, None]
    bg_blur = bg_blur * vig

    mask = np.ones((target_h, target_w), dtype=np.float32)
    feather = 85
    for i in range(feather):
        f = 0.5 - 0.5 * math.cos(math.pi * (i / float(feather)))
        mask[i, :] *= f
        mask[target_h - 1 - i, :] *= f
        mask[:, i] *= f
        mask[:, target_w - 1 - i] *= f

    full_mask = np.zeros((H, W), dtype=np.float32)
    full_mask[y0:y0 + target_h, x0:x0 + target_w] = mask
    full_mask3 = full_mask[:, :, None]

    canvas = (bg_blur * (1.0 - full_mask3) + full_mono.astype(np.float32) * full_mask3).astype(np.uint8)

    out_maps = {}
    for k, arr in extra_maps.items():
        full_arr = np.zeros((H, W), dtype=np.float32)
        full_arr[y0:y0 + target_h, x0:x0 + target_w] = arr * mask
        out_maps[k] = full_arr

    return canvas, out_maps


def prepare_muscle_frame_v24(bgr: np.ndarray) -> dict[str, np.ndarray | float]:
    scale = 0.84
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(W * scale), int(H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]
    h_r, w_r = r_ch.shape
    yy_r, xx_r = np.mgrid[:h_r, :w_r].astype(np.float32)

    red_any = np.clip((r_ch - b_ch - 10.0) / 90.0, 0.0, 1.0)
    green_any = np.clip((g_ch - np.maximum(r_ch, b_ch) - 12.0) / 90.0, 0.0, 1.0)

    fiber_lum = 0.32 * r_ch + 0.68 * g_ch
    blur_fine = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=1.3, sigmaY=1.3)
    blur_med = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=4.2, sigmaY=4.2)
    blur_wide = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=11.0, sigmaY=11.0)

    fiber_detail = fiber_lum + (fiber_lum - blur_fine) * 2.50 + (blur_fine - blur_med) * 1.30
    fiber_norm_res = np.clip((fiber_detail - 26.0) / 190.0, 0.0, 1.0)
    natural_sulcus_res = np.clip((blur_wide - blur_fine - 4.0) / 28.0, 0.0, 1.0)

    non_red_lum = 0.25 * g_ch + 0.75 * b_ch
    sculpted_grey_muscle = 42.0 + 136.0 * (fiber_norm_res ** 1.06)

    arm_ellipse = np.exp(-(((xx_r - 658.0) / 76.0) ** 2 + ((yy_r - 612.0) / 155.0) ** 2))
    arm_zone = np.clip((arm_ellipse - 0.12) / 0.55, 0.0, 1.0)

    red_replacement = non_red_lum * (1.0 - arm_zone) + sculpted_grey_muscle * arm_zone
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    neutral_lum = base_lum * (1.0 - red_any) + red_replacement * red_any

    hand_sculpt_lum = 65.0 + 0.65 * r_ch
    neutral_lum = neutral_lum * (1.0 - green_any) + hand_sculpt_lum * green_any
    neutral_lum = np.clip(neutral_lum, 14.0, 240.0)

    rb_strict = np.clip((r_ch - b_ch - 108.0) / 88.0, 0.0, 1.0)
    r_strict = np.clip((r_ch - 152.0) / 62.0, 0.0, 1.0)
    red_core_res = rb_strict * r_strict * arm_zone
    red_core_res = cv2.GaussianBlur(red_core_res, (0, 0), sigmaX=1.2, sigmaY=1.2)

    norm = neutral_lum / 255.0
    mono_r = BG_CHARCOAL[0] + (234 - BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = BG_CHARCOAL[1] + (238 - BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = BG_CHARCOAL[2] + (244 - BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    canvas, maps = place_in_studio_canvas_clean(
        mono_res,
        {
            "fiber_norm": fiber_norm_res,
            "red_core": red_core_res,
            "natural_sulcus": natural_sulcus_res,
        },
        scale=scale,
        x_shift=34,
        y_shift=50,
    )

    rc = maps["red_core"]
    m_ys, m_xs = np.where(rc > 0.22)
    if len(m_ys) > 40:
        y_top = float(np.percentile(m_ys, 4))
        y_bot = float(np.percentile(m_ys, 96))
        x_med = float(np.median(m_xs))
    else:
        y_top, y_bot, x_med = 630.0, 855.0, 730.0

    return {
        "mono": canvas,
        "fiber_norm": maps["fiber_norm"],
        "red_core": rc,
        "natural_sulcus": maps["natural_sulcus"],
        "y_top": y_top,
        "y_bot": y_bot,
        "x_med": x_med,
    }


def apply_dramatic_elbow_swing_posture(
    frame_data: dict[str, np.ndarray | float],
    swing_amount: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """
    Creates an unmistakable biomechanical posture difference for Step 2 (`✕ ELBOW SWING`):
    1. Rotates the entire upper arm, elbow, forearm, and hands forward (`+24°` clockwise around the shoulder joint
       at `px=718, py=605`), shifting the elbow 62px forward (left) and raising the forearm/hands!
    2. Slightly tilts the upper torso/head forward (`-22px` left, `+14px` down) so the whole upper body visibly
       swings forward into the cable compared to Step 3 (`✓ PIN ELBOWS`)!
    """
    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    # Shoulder pivot in 1080x1920 canvas
    px, py = 718.0, 605.0
    dx = xx - px
    dy = yy - py

    # 1. Arm + Forearm + Hands region (`x = 380..830, y = 570..1020`): rotate forward by +23 degrees!
    arm_mask = (
        np.exp(-(((xx - 640.0) / 185.0) ** 2 + ((yy - 780.0) / 210.0) ** 2))
        * np.clip((yy - 565.0) / 55.0, 0.0, 1.0)
        * np.clip((845.0 - xx) / 70.0, 0.0, 1.0)
    )
    theta = math.radians(23.0 * swing_amount) * arm_mask
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Clockwise rotation in screen coords (moves bottom of arm left/forward and slightly up)
    src_x_arm = px + dx * cos_t - dy * sin_t
    src_y_arm = py + dx * sin_t + dy * cos_t

    # 2. Upper torso & head forward lean (`y = 400..850, x = 520..880`)
    torso_mask = np.exp(-(((xx - 690.0) / 170.0) ** 2 + ((yy - 600.0) / 220.0) ** 2))
    lean_dx = 24.0 * swing_amount * torso_mask * np.clip((850.0 - yy) / 300.0, 0.0, 1.0)
    lean_dy = -12.0 * swing_amount * torso_mask * np.clip((850.0 - yy) / 300.0, 0.0, 1.0)

    map_x = (src_x_arm + lean_dx).astype(np.float32)
    map_y = (src_y_arm + lean_dy).astype(np.float32)

    out: dict[str, np.ndarray | float] = {}
    for k, v in frame_data.items():
        if isinstance(v, np.ndarray):
            out[k] = cv2.remap(v, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            out[k] = v

    # Update muscle bounding coordinates after warp
    rc = out["red_core"]
    m_ys, m_xs = np.where(rc > 0.22)
    if len(m_ys) > 40:
        out["y_top"] = float(np.percentile(m_ys, 4))
        out["y_bot"] = float(np.percentile(m_ys, 96))
        out["x_med"] = float(np.median(m_xs))
    return out


def render_targeted_muscle_v24(
    frame_data: dict[str, np.ndarray | float],
    target_mode: str,
    is_wrong: bool,
    alpha_scale: float = 1.0,
) -> tuple[np.ndarray, list[tuple[float, float, str]], tuple[float, float], tuple[float, float]]:
    """
    Renders the targeted muscle head(s) and returns:
    - `rendered_rgb`
    - `targets`: list of `(x, y, label)` for active muscle heads
    - `shoulder_xy`: `(sx, sy)` shoulder pivot
    - `elbow_xy`: `(ex, ey)` current elbow pivot (for kinematic posture comparison overlay)
    """
    working = apply_dramatic_elbow_swing_posture(frame_data, swing_amount=1.0) if target_mode == "anterior_deltoid" else frame_data

    mono = working["mono"].astype(np.float32)
    fiber_norm = working["fiber_norm"]
    red_core = working["red_core"]
    natural_sulcus = working["natural_sulcus"]

    y_top = float(working["y_top"])
    y_bot = float(working["y_bot"])
    x_med = float(working["x_med"])
    span_y = max(130.0, y_bot - y_top)

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    sculpt_gate = np.clip((red_core ** 0.75) * 1.25, 0.0, 1.0)

    dx_arm = xx - (x_med - 6.0)
    y_delt_v = (y_top + 0.42 * span_y) - 0.0050 * (dx_arm ** 2) + 0.14 * dx_arm

    is_deltoid = np.clip((y_delt_v - yy) / 11.0, 0.0, 1.0)
    is_triceps = np.clip((yy - (y_delt_v - 2.0)) / 11.0, 0.0, 1.0)

    x_septum = (x_med + 2.0) - (yy - (y_top + 0.43 * span_y)) * 0.05
    y_med_arch = (y_bot - 0.25 * span_y) + ((xx - x_med) ** 2) * 0.007

    is_med_head = is_triceps * np.clip((yy - y_med_arch) / 9.0, 0.0, 1.0)
    is_lat_head = is_triceps * np.clip((x_septum - xx + 2.0) / 8.0, 0.0, 1.0) * (1.0 - is_med_head * 0.55)
    is_long_head = is_triceps * np.clip((xx - x_septum + 2.0) / 8.0, 0.0, 1.0) * (1.0 - is_med_head * 0.55)

    groove_delt = np.exp(-((yy - y_delt_v) ** 2) / (2.0 * 2.8 ** 2))
    groove_septum = np.exp(-((xx - x_septum) ** 2) / (2.0 * 2.2 ** 2)) * is_triceps
    groove_med = np.exp(-((yy - y_med_arch) ** 2) / (2.0 * 2.2 ** 2)) * is_triceps

    if target_mode == "anterior_deltoid":
        primary_mask = is_deltoid * sculpt_gate
        sulcus_active = np.clip(groove_delt * 0.75 + natural_sulcus * 0.55, 0.0, 1.0)
    elif target_mode == "triceps_lat_long":
        primary_mask = np.maximum(is_lat_head, is_long_head * 0.96) * sculpt_gate * (1.0 - is_med_head * 0.30)
        sulcus_active = np.clip(groove_delt * 0.65 + groove_septum * 0.90 + natural_sulcus * 0.65, 0.0, 1.0)
    else:
        primary_mask = is_triceps * sculpt_gate
        sulcus_active = np.clip(groove_delt * 0.65 + groove_septum * 0.92 + groove_med * 0.85 + natural_sulcus * 0.68, 0.0, 1.0)

    total_mask = primary_mask * (1.0 - 0.80 * sulcus_active)
    total_mask = cv2.GaussianBlur(total_mask, (0, 0), sigmaX=1.4, sigmaY=1.4)
    total_mask = np.clip(total_mask * 1.16, 0.0, 1.0) * alpha_scale

    f_contrast = np.clip((fiber_norm - 0.17) / 0.70, 0.0, 1.0)
    f_val = (f_contrast ** 0.84) * (1.0 - 0.78 * sulcus_active)

    if is_wrong:
        col_r = 74.0 + 181.0 * (f_val ** 0.60)
        col_g = 10.0 + 80.0 * (f_val ** 1.28) + 150.0 * (f_val ** 2.7)
        col_b = 8.0 + 34.0 * (f_val ** 1.40) + 128.0 * (f_val ** 3.0)
    else:
        col_r = 8.0 + 50.0 * (f_val ** 1.32) + 172.0 * (f_val ** 2.8)
        col_g = 64.0 + 191.0 * (f_val ** 0.60)
        col_b = 12.0 + 56.0 * (f_val ** 1.22) + 132.0 * (f_val ** 2.8)

    painted = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)

    out = mono.copy()
    fill_alpha = np.clip(total_mask * 0.95, 0.0, 0.97)[:, :, None]
    out = out * (1.0 - fill_alpha) + painted * fill_alpha

    sulcus_alpha = (sulcus_active * total_mask * 0.72)[:, :, None]
    out = out * (1.0 - sulcus_alpha) + 10.0 * sulcus_alpha

    def head_centroid(h_mask: np.ndarray, default_xy: tuple[float, float]) -> tuple[float, float]:
        ys, xs = np.where((h_mask * sculpt_gate) > 0.32)
        if len(ys) > 20:
            return float(np.median(xs)), float(np.median(ys))
        return default_xy

    if target_mode == "anterior_deltoid":
        cx, cy = head_centroid(is_deltoid, (x_med - 18.0, y_top + 0.22 * span_y))
        targets = [(cx, cy, "FRONT DELT")]
        shoulder_xy = (702.0, 608.0)
        elbow_xy = (646.0, 818.0)
    elif target_mode == "triceps_lat_long":
        lx, ly = head_centroid(is_lat_head, (x_med - 14.0, y_top + 0.58 * span_y))
        gx, gy = head_centroid(is_long_head, (x_med + 16.0, y_top + 0.62 * span_y))
        targets = [(lx, ly, "LATERAL HEAD"), (gx, gy, "LONG HEAD")]
        shoulder_xy = (722.0, 605.0)
        elbow_xy = (722.0, 835.0)
    else:
        lx, ly = head_centroid(is_lat_head, (x_med - 14.0, y_top + 0.56 * span_y))
        gx, gy = head_centroid(is_long_head, (x_med + 16.0, y_top + 0.60 * span_y))
        mx, my = head_centroid(is_med_head, (x_med - 4.0, y_top + 0.82 * span_y))
        targets = [(lx, ly, "LATERAL HEAD"), (gx, gy, "LONG HEAD"), (mx, my, "MEDIAL HEAD")]
        shoulder_xy = (722.0, 605.0)
        elbow_xy = (722.0, 835.0)

    return np.clip(out, 0, 255).astype(np.uint8), targets, shoulder_xy, elbow_xy


def draw_kinematic_posture_indicator(
    canvas: Image.Image,
    shoulder_xy: tuple[float, float],
    elbow_xy: tuple[float, float],
    is_wrong: bool,
):
    """
    Draws a super-clear biomechanical posture alignment indicator on the upper arm/elbow so the viewer
    immediately sees WHY Step 2 (`+24° ELBOW SWING`) is wrong and Step 3 (`0° PINNED ELBOW`) is right!
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    sx, sy = int(round(shoulder_xy[0])), int(round(shoulder_xy[1]))
    ex, ey = int(round(elbow_xy[0])), int(round(elbow_xy[1]))

    if is_wrong:
        # 1. Vertical reference dashed line (where the upper arm SHOULD stay pinned: straight down from shoulder)
        ref_x = sx + 14
        for y_d in range(sy + 15, ey + 25, 16):
            od.line([(ref_x, y_d), (ref_x, min(ey + 25, y_d + 8))], fill=(210, 220, 235, 135), width=2)

        # 2. Actual swung upper-arm vector (`sx, sy` -> `ex, ey`) in glowing red
        od.line([(sx, sy), (ex, ey)], fill=(*COL_RED, 225), width=4)

        # 3. Forward swing arc + arrow from `ref_x` to `ex` at the elbow level (`ey`)
        arc_y = ey - 10
        od.line([(ref_x, arc_y), (ex + 6, arc_y)], fill=(*COL_RED, 245), width=4)
        # Arrowhead pointing left (forward swing)
        od.polygon(
            [(ex, arc_y), (ex + 14, arc_y - 7), (ex + 14, arc_y + 7)],
            fill=(*COL_RED, 255),
        )
        # Elbow joint warning ring
        od.ellipse([ex - 11, ey - 11, ex + 11, ey + 11], fill=(16, 18, 24, 210), outline=(*COL_RED, 255), width=3)
    else:
        # Vertical locked alignment axis + green elbow lock ring (`0° PINNED`)
        for y_d in range(sy + 15, ey + 20, 14):
            od.line([(ex, y_d), (ex, min(ey + 20, y_d + 8))], fill=(*COL_GREEN, 175), width=2)
        od.ellipse([ex - 13, ey - 13, ex + 13, ey + 13], fill=(14, 20, 18, 210), outline=(*COL_GREEN, 255), width=3)
        od.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(255, 255, 255, 255))

    canvas.alpha_composite(ov)


def draw_reticle_and_leader(
    od: ImageDraw.ImageDraw,
    targets: list[tuple[float, float, str]],
    card_anchor_xy: tuple[int, int],
    accent: tuple[int, int, int],
):
    px, py, _ = targets[0]
    px_i, py_i = int(round(px)), int(round(py))

    if len(targets) > 1:
        for k in range(1, len(targets)):
            nx, ny, _ = targets[k]
            nx_i, ny_i = int(round(nx)), int(round(ny))
            od.line([(px_i, py_i), (nx_i, ny_i)], fill=(*accent, 160), width=2)
            od.ellipse([nx_i - 8, ny_i - 8, nx_i + 8, ny_i + 8], fill=(12, 15, 20, 195), outline=(*accent, 240), width=2)
            od.ellipse([nx_i - 3, ny_i - 3, nx_i + 3, ny_i + 3], fill=(255, 255, 255, 255))

    r_out = 17
    od.ellipse([px_i - r_out - 2, py_i - r_out - 2, px_i + r_out + 2, py_i + r_out + 2], outline=(10, 12, 16, 185), width=3)
    od.ellipse([px_i - r_out, py_i - r_out, px_i + r_out, py_i + r_out], outline=(*accent, 250), width=2)
    od.ellipse([px_i - 5, py_i - 5, px_i + 5, py_i + 5], fill=(255, 255, 255, 255), outline=(*accent, 255), width=2)

    ax, ay = card_anchor_xy
    elbow_x = ax
    elbow_y = py_i + 30
    pts = [(px_i - r_out - 3, py_i), (elbow_x + 24, py_i), (elbow_x, elbow_y), (ax, ay)]
    od.line(pts, fill=(10, 12, 16, 185), width=5, joint="curve")
    od.line(pts, fill=(*accent, 225), width=2, joint="curve")
    od.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=(*accent, 255))


def draw_concept_a_dual_versus_bar(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    is_wrong: bool,
):
    """
    Concept A: "Dual-Bar Versus Gauge" (직관적 2단 비교 바 — Whoop / Apple Fitness 스타일)
    - Instantly answers:
      1. WHAT posture: `ELBOW SWING` vs `PIN ELBOWS`
      2. WHERE & HOW MUCH:
         * `TRICEPS` bar (`26%` vs `96%`)
         * `FRONT DELT` bar (`74%` vs `4%`)
      3. SETS × REPS badge (`4 SETS × 12 REPS`)
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    cx, cy = 84, 1175
    cw, ch = 530, 236
    draw_reticle_and_leader(od, targets, (cx + 150, cy), accent)

    # Sleek translucent glass card (leaves background machine contours gently visible through frosted glass)
    od.rounded_rectangle([cx + 4, cy + 6, cx + cw + 4, cy + ch + 6], radius=26, fill=(4, 6, 10, 145))
    od.rounded_rectangle(
        [cx, cy, cx + cw, cy + ch],
        radius=24,
        fill=(16, 20, 28, 212),
        outline=(225, 235, 250, 68),
        width=2,
    )
    od.rounded_rectangle([cx + 14, cy + 20, cx + 20, cy + ch - 20], radius=3, fill=(*accent, 255))

    ix = cx + 36
    # Header: Big bold coaching title + right-aligned posture angle pill
    title = "ELBOW SWING" if is_wrong else "PIN ELBOWS"
    od.text((ix, cy + 20), title, font=load_font(34, bold=True), fill=(255, 255, 255, 255))

    pill_txt = "+24° DRIFT" if is_wrong else "0° LOCKED"
    pill_w = 134
    od.rounded_rectangle(
        [cx + cw - pill_w - 20, cy + 22, cx + cw - 20, cy + 58],
        radius=18,
        fill=(*accent, 42),
        outline=(*accent, 215),
        width=2,
    )
    od.text((cx + cw - pill_w - 6, cy + 30), pill_txt, font=load_font(17, bold=True), fill=(*accent, 255))

    # Row 1: TRICEPS (Target Muscle) Bar
    rows = (
        [
            ("TRICEPS (TARGET)", 26, (130, 142, 160)),
            ("FRONT DELT (STEAL)", 74, COL_RED),
        ]
        if is_wrong
        else [
            ("TRICEPS (3 HEADS)", 96, COL_GREEN),
            ("FRONT DELT (LOAD)", 4, (110, 122, 138)),
        ]
    )

    for idx_r, (r_label, r_pct, r_col) in enumerate(rows):
        ry = cy + 78 + idx_r * 58
        od.text((ix, ry), r_label, font=load_font(18, bold=True), fill=(225, 232, 245, 245))
        pct_s = f"{r_pct}%"
        od.text((cx + cw - 82, ry - 4), pct_s, font=load_font(24, bold=True), fill=(*r_col, 255))

        # Smooth rounded progress track
        bx1, by1 = ix, ry + 28
        bw_total = cw - 60
        bh = 14
        od.rounded_rectangle([bx1, by1, bx1 + bw_total, by1 + bh], radius=7, fill=(32, 38, 48, 210))
        fill_w = max(14, int(round(bw_total * (r_pct / 100.0))))
        od.rounded_rectangle([bx1, by1, bx1 + fill_w, by1 + bh], radius=7, fill=(*r_col, 255))

    # Bottom footer tag
    sub_txt = "DELTS STEAL TENSION FROM TRICEPS" if is_wrong else "4 SETS × 12 REPS · FULL LOCKOUT"
    od.text((ix, cy + ch - 34), sub_txt, font=load_font(17, bold=True), fill=(*accent, 245))

    canvas.alpha_composite(ov)


def draw_concept_b_radial_ring_pill(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    is_wrong: bool,
):
    """
    Concept B: "Radial Arc Ring + Floating Pill" (무베젤 원형 다이얼 + 플로팅 캡슐 — Porsche / Peloton 스타일)
    - Eliminates the heavy box feel by pairing a 4x-supersampled 270° circular gauge on the left
      with a sleek floating capsule on the right.
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    ring_cx, ring_cy = 168, 1285
    ring_r = 74
    draw_reticle_and_leader(od, targets, (ring_cx, ring_cy - ring_r - 6), accent)

    # Floating pill extending to the right of the circular dial
    px1, py1 = 232, 1212
    px2, py2 = 628, 1358
    od.rounded_rectangle([px1 + 4, py1 + 6, px2 + 4, py2 + 6], radius=36, fill=(4, 6, 10, 145))
    od.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=36,
        fill=(16, 20, 28, 215),
        outline=(225, 235, 250, 68),
        width=2,
    )

    # 4x Supersampled 270-degree Circular Dial
    ss = 4
    dsz = (ring_r * 2 + 24) * ss
    dc = dsz // 2
    dr = ring_r * ss
    dial = Image.new("RGBA", (dsz, dsz), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)

    dd.ellipse([dc - dr, dc - dr, dc + dr, dc + dr], fill=(14, 18, 25, 242), outline=(220, 230, 245, 75), width=2 * ss)
    arc_box = [dc - dr + 10 * ss, dc - dr + 10 * ss, dc + dr - 10 * ss, dc + dr - 10 * ss]
    dd.arc(arc_box, start=135, end=405, fill=(36, 42, 54, 230), width=10 * ss)

    pct_main = 74 if is_wrong else 96
    end_deg = 135 + int(round(270 * (pct_main / 100.0)))
    dd.arc(arc_box, start=135, end=end_deg, fill=(*accent, 255), width=10 * ss)

    dial_down = dial.resize((ring_r * 2 + 24, ring_r * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(dial_down, (ring_cx - ring_r - 12, ring_cy - ring_r - 12))

    # Percentage inside dial
    p_str = f"{pct_main}%"
    f_dial = load_font(36, bold=True)
    pb = od.textbbox((0, 0), p_str, font=f_dial)
    od.text((ring_cx - (pb[2] - pb[0]) // 2, ring_cy - 26), p_str, font=f_dial, fill=(255, 255, 255, 255))
    sub_d = "DELT STEAL" if is_wrong else "TRICEPS"
    f_sub = load_font(14, bold=True)
    sb = od.textbbox((0, 0), sub_d, font=f_sub)
    od.text((ring_cx - (sb[2] - sb[0]) // 2, ring_cy + 18), sub_d, font=f_sub, fill=(*accent, 255))

    # Right capsule text (3 ultra-clean lines)
    tx = 262
    kicker = "ANTERIOR DELTOID OVERLOAD" if is_wrong else "LATERAL · LONG · MEDIAL"
    main_cue = "ELBOW SWING" if is_wrong else "PIN ELBOWS"
    sub_cue = "TRICEPS DROPS TO 26%" if is_wrong else "4 SETS × 12 REPS LOCKOUT"

    od.text((tx, py1 + 20), kicker, font=load_font(17, bold=True), fill=(*accent, 255))
    od.text((tx, py1 + 48), main_cue, font=load_font(36, bold=True), fill=(255, 255, 255, 255))
    od.text((tx, py1 + 98), sub_cue, font=load_font(18, bold=True), fill=(210, 220, 235, 240))

    canvas.alpha_composite(ov)


def draw_concept_c_3head_equalizer(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    is_wrong: bool,
):
    """
    Concept C: "3-Head Anatomy Equalizer" (3갈래 부위별 직관 미니 카드 — Muscle & Motion Pro 스타일)
    - Shows the 3 anatomical heads/zones as 3 side-by-side visual battery/equalizer pills so the viewer
      immediately sees which exact head is firing!
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    cx, cy = 84, 1170
    cw, ch = 536, 242
    draw_reticle_and_leader(od, targets, (cx + 150, cy), accent)

    od.rounded_rectangle([cx + 4, cy + 6, cx + cw + 4, cy + ch + 6], radius=24, fill=(4, 6, 10, 145))
    od.rounded_rectangle(
        [cx, cy, cx + cw, cy + ch],
        radius=22,
        fill=(16, 20, 28, 212),
        outline=(225, 235, 250, 68),
        width=2,
    )

    ix = cx + 24
    title = "ELBOW SWING · DELT STEAL" if is_wrong else "PIN ELBOWS · 4 × 12 REPS"
    od.text((ix, cy + 18), title, font=load_font(28, bold=True), fill=(255, 255, 255, 255))

    # 3 Equalizer Pillars
    cols_data = (
        [
            ("FRONT DELT", 74, COL_RED),
            ("LAT HEAD", 16, (120, 130, 145)),
            ("LONG HEAD", 10, (120, 130, 145)),
        ]
        if is_wrong
        else [
            ("LAT HEAD", 98, COL_GREEN),
            ("LONG HEAD", 95, COL_GREEN),
            ("MEDIAL", 92, COL_GREEN),
        ]
    )

    box_w = 152
    box_h = 148
    gap = 16
    by = cy + 68
    for i, (h_name, h_pct, h_col) in enumerate(cols_data):
        bx = ix + i * (box_w + gap)
        od.rounded_rectangle(
            [bx, by, bx + box_w, by + box_h],
            radius=16,
            fill=(24, 30, 40, 215),
            outline=(*h_col, 185 if h_pct > 50 else 65),
            width=2,
        )
        # Bottom-up fill bar inside the mini card
        bar_x1, bar_y2 = bx + 14, by + box_h - 14
        bar_max_h = 54
        bar_y1 = bar_y2 - bar_max_h
        od.rounded_rectangle([bar_x1, bar_y1, bx + box_w - 14, bar_y2], radius=6, fill=(14, 18, 24, 230))
        fill_h = max(8, int(round(bar_max_h * (h_pct / 100.0))))
        od.rounded_rectangle([bar_x1, bar_y2 - fill_h, bx + box_w - 14, bar_y2], radius=6, fill=(*h_col, 245))

        od.text((bx + 14, by + 12), h_name, font=load_font(16, bold=True), fill=(220, 228, 242, 245))
        od.text((bx + 14, by + 36), f"{h_pct}%", font=load_font(30, bold=True), fill=(255, 255, 255, 255) if h_pct > 50 else (160, 170, 185, 220))

    canvas.alpha_composite(ov)


def draw_top_dock_and_right_status(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
):
    """
    Matches `@demicstory`:
    - Top-left: 3-torso thumbnail dock (`x = 84`).
    - Top-right: Luxury circular `✕ / ✓` status badge at `x = 910, y = 317` (strictly in top-right Safe Zone!).
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    tw, th = 144, 114
    gap = 12
    pad = 12
    bar_x = 84
    bar_y = 248
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + 2 * pad

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=22,
        fill=(14, 18, 24, 220),
        outline=(210, 222, 240, 62),
        width=2,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy() if i == active_idx else inact_im.copy()

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle(
                [tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                radius=16,
                outline=(*i_col, 255),
                width=3,
            )

    # Top-Right Status Emblem (aligned cleanly on the right side like `@demicstory`)
    bx = 912
    by = bar_y + bar_h // 2
    br = 46
    ss = 4
    sz = (br * 2 + 24) * ss
    c = sz // 2
    r = br * ss
    em = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    ed = ImageDraw.Draw(em)
    ed.ellipse([c - r, c - r, c + r, c + r], fill=(16, 19, 26, 242), outline=(215, 224, 238, 95), width=2 * ss)
    ri = r - 6 * ss
    ed.ellipse([c - ri, c - ri, c + ri, c + ri], fill=(20, 24, 32, 248), outline=(*accent, 245), width=3 * ss)

    if is_wrong:
        d = int(15 * ss)
        w = int(6.5 * ss)
        ed.line([(c - d, c - d), (c + d, c + d)], fill=(255, 255, 255, 255), width=w)
        ed.line([(c - d, c + d), (c + d, c - d)], fill=(255, 255, 255, 255), width=w)
    else:
        pts = [(c - 18 * ss, c + 1 * ss), (c - 4 * ss, c + 15 * ss), (c + 20 * ss, c - 13 * ss)]
        ed.line(pts, fill=(255, 255, 255, 255), width=int(7 * ss), joint="curve")

    em_down = em.resize((br * 2 + 24, br * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(em_down, (bx - br - 12, by - br - 12))
    canvas.alpha_composite(ov)


def main():
    cap_c = cv2.VideoCapture(CORRECT_I2V)
    cap_c.set(cv2.CAP_PROP_POS_FRAMES, 52)
    _, bgr_c = cap_c.read()
    cap_c.release()

    m_data = prepare_muscle_frame_v24(bgr_c)

    # Render Wrong (Step 2: Elbow Swing +24°) and Correct (Step 4: Pinned Elbows 0° Lockout)
    wrong_rgb, wrong_targets, w_sh, w_el = render_targeted_muscle_v24(m_data, "anterior_deltoid", True, 1.0)
    corr_rgb, corr_targets, c_sh, c_el = render_targeted_muscle_v24(m_data, "triceps_all_three", False, 1.0)

    # Build top dock icons
    size = (144, 114)
    crop_box = (555, 550, 905, 950)
    inact = Image.fromarray(m_data["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 120)))
    im_w = Image.fromarray(wrong_rgb).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im_c = Image.fromarray(corr_rgb).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im_w, inact.copy()), (im_w, inact.copy()), (im_c, inact.copy())]

    concepts = [
        ("A", "CONCEPT A: DUAL-BAR VERSUS GAUGE", draw_concept_a_dual_versus_bar),
        ("B", "CONCEPT B: RADIAL RING + FLOATING PILL", draw_concept_b_radial_ring_pill),
        ("C", "CONCEPT C: 3-HEAD ANATOMY EQUALIZER", draw_concept_c_3head_equalizer),
    ]

    full_panels = []
    for code, label, fn_draw in concepts:
        # Render Wrong (Step 2) and Correct (Step 3/4) for each concept
        im_wrong = Image.fromarray(wrong_rgb).convert("RGBA")
        draw_kinematic_posture_indicator(im_wrong, w_sh, w_el, is_wrong=True)
        draw_top_dock_and_right_status(im_wrong, torso_icons, 1, is_wrong=True)
        fn_draw(im_wrong, wrong_targets, is_wrong=True)

        im_corr = Image.fromarray(corr_rgb).convert("RGBA")
        draw_kinematic_posture_indicator(im_corr, c_sh, c_el, is_wrong=False)
        draw_top_dock_and_right_status(im_corr, torso_icons, 2, is_wrong=False)
        fn_draw(im_corr, corr_targets, is_wrong=False)

        p_w_out = os.path.join(OUT_DIR, f"v24_concept_{code}_wrong.jpg")
        p_c_out = os.path.join(OUT_DIR, f"v24_concept_{code}_correct.jpg")
        im_wrong.convert("RGB").save(p_w_out, quality=95)
        im_corr.convert("RGB").save(p_c_out, quality=95)
        print("Saved:", p_w_out, p_c_out)

        # Side-by-side Wrong vs Correct card for this concept
        pair = Image.new("RGB", (W * 2 + 24, H + 90), (10, 12, 16))
        pd = ImageDraw.Draw(pair)
        pd.text((24, 22), f"{label}  —  LEFT: ✕ ELBOW SWING (+24° DRIFT)   vs   RIGHT: ✓ PINNED ELBOWS (0° LOCK)", font=load_font(32, bold=True), fill=(240, 245, 255))
        pair.paste(im_wrong.convert("RGB"), (0, 82))
        pair.paste(im_corr.convert("RGB"), (W + 24, 82))
        pair_out = os.path.join(OUT_DIR, f"v24_concept_{code}_pair.jpg")
        pair.resize((1200, int((H + 90) * (1200 / (W * 2 + 24)))), Image.Resampling.LANCZOS).save(pair_out, quality=95)
        full_panels.append((code, label, im_wrong.convert("RGB"), im_corr.convert("RGB")))

    # Build a 3-Concept Master Comparison Sheet (Focusing on the lower half & full posture)
    sheet_w = 520 * 3 + 24 * 4
    sheet_h = 924 + 80
    sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 12, 16))
    sd = ImageDraw.Draw(sheet)
    for i, (code, label, _, im_c) in enumerate(full_panels):
        x = 24 + i * (520 + 24)
        im_r = im_c.resize((520, 924), Image.Resampling.LANCZOS)
        sheet.paste(im_r, (x, 64))
        sd.rectangle([x, 64, x + 519, 64 + 923], outline=COL_GREEN, width=3)
        sd.text((x + 8, 18), label, font=load_font(22, bold=True), fill=COL_GREEN)
    sheet_out = os.path.join(OUT_DIR, "v24_3concepts_overview.jpg")
    sheet.save(sheet_out, quality=95)
    print("Saved 3-concept overview:", sheet_out)


if __name__ == "__main__":
    main()
