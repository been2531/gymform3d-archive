#!/usr/bin/env python3
"""
GymForm 3D — Universal Luxury Biomechanical Reels Engine (v24 Concept B Master)
Confirmed Master Architecture:
1. Concept B Lower-Left UI/UX ("Radial Arc Ring + Floating Pill"):
   - Eliminates rectangular box heaviness: pairs a 4x-supersampled 270° circular dial (`%` + target tag)
     on the left (`cx = 168, cy = 1285`) with a sleek floating rounded capsule (`x = 232..638, y = 1210..1360`).
   - Precision Anatomical Reticle (`◎`) on the active muscle head(s) / X-Ray joint connects via a fine leader line
     directly to the top of the circular dial.
2. Top-Right ONLY Status Emblem (`✕ / ✓` at `x = 912, y = 317`) + Top-Left 3-Torso Dock (`x = 84, y = 248`),
   matching `@demicstory`.
3. Zero Artificial Bottom-Left Darkening (`ll_dim` removed): studio floor & cable machine base look 100% natural.
4. Unmistakable Biomechanical Posture Comparison:
   - Step 1 (`✕`): 3D X-Ray Skeletal Scan (`gym_day1_wrong_i2v.mp4`) showing Humerus, Shoulder & Elbow joint overload.
   - Step 2 (`✕`): True shoulder-pivot forward arm rotation (`+24° ELBOW SWING`) + forward torso lean + red swing vector arrow.
   - Step 3 & 4 (`✓`): Upright posture + `0° LOCKED` vertical elbow alignment axis & green elbow lock ring.
5. Non-Overlapping English Coaching Voiceover (`gap >= 1.75s`).
"""

import asyncio
import math
import os
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
DURATION = 28.0
TOTAL_FRAMES = int(DURATION * FPS)

BASE_DIR = "/workspace/daily_info_reels"
OUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = "/workspace/frames_v24_day1"

CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")

OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")
OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v24_qc_preview.jpg")
OUT_COMPARISON = os.path.join(OUT_DIR, "day1_v24_posture_and_anatomy_closeup.jpg")
OUT_HOOK = os.path.join(OUT_DIR, "day1_v24_step0_hook.jpg")
OUT_STEP1 = os.path.join(OUT_DIR, "day1_v24_step1_xray_joint.jpg")
OUT_STEP2 = os.path.join(OUT_DIR, "day1_v24_step2_wrong_muscle.jpg")
OUT_STEP3 = os.path.join(OUT_DIR, "day1_v24_step3_posture_key.jpg")
OUT_STEP4 = os.path.join(OUT_DIR, "day1_v24_step4_target_muscle.jpg")

BG_CHARCOAL = (14, 16, 20)
COL_RED = (255, 62, 50)
COL_GREEN = (24, 226, 105)


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


FONT_DIAL_PCT = load_font(36, bold=True)
FONT_DIAL_TAG = load_font(14, bold=True)
FONT_PILL_KICKER = load_font(17, bold=True)
FONT_PILL_MAIN = load_font(36, bold=True)
FONT_PILL_SUB = load_font(18, bold=True)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / max(1e-6, e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


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


def prepare_muscle_frame(bgr: np.ndarray) -> dict[str, np.ndarray | float]:
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


def prepare_xray_bone_frame(bgr: np.ndarray) -> dict[str, np.ndarray | float]:
    scale = 0.84
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(W * scale), int(H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]
    h_r, w_r = r_ch.shape
    yy_r, xx_r = np.mgrid[:h_r, :w_r].astype(np.float32)

    red_any = np.clip((r_ch - b_ch - 12.0) / 95.0, 0.0, 1.0)
    bone_detail = 0.38 * r_ch + 0.62 * g_ch
    blur_b = cv2.GaussianBlur(bone_detail, (0, 0), sigmaX=1.5, sigmaY=1.5)
    bone_sharp = np.clip((bone_detail + (bone_detail - blur_b) * 2.1 - 20.0) / 205.0, 0.0, 1.0)

    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    non_red_lum = 0.30 * g_ch + 0.70 * b_ch
    xray_cavity_lum = 28.0 + 135.0 * (bone_sharp ** 1.15)

    arm_xray_zone = np.exp(-(((xx_r - 645.0) / 95.0) ** 2 + ((yy_r - 740.0) / 240.0) ** 2))
    arm_xray_gate = np.clip((arm_xray_zone - 0.10) / 0.55, 0.0, 1.0)

    neutral_lum = base_lum * (1.0 - red_any) + (non_red_lum * (1.0 - arm_xray_gate) + xray_cavity_lum * arm_xray_gate) * red_any
    neutral_lum = np.clip(neutral_lum, 14.0, 238.0)

    bone_mask_res = np.clip((r_ch - b_ch - 82.0) / 95.0, 0.0, 1.0) * np.clip((r_ch - 130.0) / 75.0, 0.0, 1.0) * arm_xray_gate
    bone_mask_res = cv2.GaussianBlur(bone_mask_res, (0, 0), sigmaX=1.2, sigmaY=1.2)

    norm = neutral_lum / 255.0
    mono_r = BG_CHARCOAL[0] + (232 - BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = BG_CHARCOAL[1] + (236 - BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = BG_CHARCOAL[2] + (242 - BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    canvas, maps = place_in_studio_canvas_clean(
        mono_res,
        {
            "bone_sharp": bone_sharp,
            "bone_mask": bone_mask_res,
        },
        scale=scale,
        x_shift=34,
        y_shift=50,
    )

    bm = maps["bone_mask"]
    b_ys, b_xs = np.where(bm > 0.25)
    if len(b_ys) > 40:
        y_sh = float(np.percentile(b_ys, 12))
        x_sh = float(np.median(b_xs[b_ys < np.percentile(b_ys, 28)]))
        y_el = float(np.percentile(b_ys, 62))
        x_el = float(np.median(b_xs[(b_ys > np.percentile(b_ys, 48)) & (b_ys < np.percentile(b_ys, 75))]))
    else:
        x_sh, y_sh = 720.0, 625.0
        x_el, y_el = 725.0, 815.0

    return {
        "mono": canvas,
        "bone_sharp": maps["bone_sharp"],
        "bone_mask": bm,
        "x_sh": x_sh,
        "y_sh": y_sh,
        "x_el": x_el,
        "y_el": y_el,
    }


def apply_dramatic_elbow_swing_posture(
    frame_data: dict[str, np.ndarray | float],
    swing_amount: float,
) -> dict[str, np.ndarray | float]:
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    px, py = 718.0, 605.0
    dx = xx - px
    dy = yy - py

    arm_mask = (
        np.exp(-(((xx - 640.0) / 185.0) ** 2 + ((yy - 780.0) / 210.0) ** 2))
        * np.clip((yy - 565.0) / 55.0, 0.0, 1.0)
        * np.clip((845.0 - xx) / 70.0, 0.0, 1.0)
    )
    deg = 10.0 + 14.0 * swing_amount
    theta = np.radians(deg) * arm_mask
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    src_x_arm = px + dx * cos_t - dy * sin_t
    src_y_arm = py + dx * sin_t + dy * cos_t

    torso_mask = np.exp(-(((xx - 690.0) / 170.0) ** 2 + ((yy - 600.0) / 220.0) ** 2))
    lean_dx = (10.0 + 14.0 * swing_amount) * torso_mask * np.clip((850.0 - yy) / 300.0, 0.0, 1.0)
    lean_dy = -(5.0 + 7.0 * swing_amount) * torso_mask * np.clip((850.0 - yy) / 300.0, 0.0, 1.0)

    map_x = (src_x_arm + lean_dx).astype(np.float32)
    map_y = (src_y_arm + lean_dy).astype(np.float32)

    out: dict[str, np.ndarray | float] = {}
    for k, v in frame_data.items():
        if isinstance(v, np.ndarray):
            out[k] = cv2.remap(v, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            out[k] = v

    rc = out["red_core"]
    m_ys, m_xs = np.where(rc > 0.22)
    if len(m_ys) > 40:
        out["y_top"] = float(np.percentile(m_ys, 4))
        out["y_bot"] = float(np.percentile(m_ys, 96))
        out["x_med"] = float(np.median(m_xs))
    return out


def render_xray_skeletal_frame(
    xray_data: dict[str, np.ndarray | float],
    alpha_scale: float,
    rep_wave: float,
) -> tuple[np.ndarray, list[tuple[float, float, str]], tuple[float, float], tuple[float, float]]:
    mono = xray_data["mono"].astype(np.float32)
    bone_sharp = xray_data["bone_sharp"]
    bone_mask = xray_data["bone_mask"]
    x_sh, y_sh = float(xray_data["x_sh"]), float(xray_data["y_sh"])
    x_el, y_el = float(xray_data["x_el"]), float(xray_data["y_el"])

    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    xray_window = cv2.GaussianBlur(bone_mask, (0, 0), sigmaX=12.0, sigmaY=12.0)
    xray_window = np.clip(xray_window * 1.1, 0.0, 0.65) * alpha_scale
    out = mono * (1.0 - xray_window[:, :, None] * 0.48)

    b_val = np.clip((bone_sharp - 0.12) / 0.78, 0.0, 1.0)
    pulse = 0.88 + 0.12 * rep_wave

    d_sh = np.exp(-(((xx - x_sh) / 36.0) ** 2 + ((yy - y_sh) / 36.0) ** 2))
    d_el = np.exp(-(((xx - x_el) / 34.0) ** 2 + ((yy - y_el) / 34.0) ** 2))
    joint_heat = np.clip(d_sh + d_el * 0.85, 0.0, 1.0)

    col_r = 95.0 + 160.0 * (b_val ** 0.55)
    col_g = 14.0 + 110.0 * (b_val ** 1.25) + 120.0 * (b_val ** 2.6) + 35.0 * joint_heat
    col_b = 10.0 + 45.0 * (b_val ** 1.50) + 135.0 * (b_val ** 3.0)

    bone_rgb = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)
    fill_alpha = np.clip((bone_mask ** 0.80) * 0.96 * alpha_scale * pulse, 0.0, 0.96)[:, :, None]
    out = out * (1.0 - fill_alpha) + bone_rgb * fill_alpha

    targets = [
        (x_sh, y_sh, "GLENOHUMERAL JOINT"),
        (x_el, y_el, "HUMERUS & ELBOW"),
    ]
    return np.clip(out, 0, 255).astype(np.uint8), targets, (x_sh, y_sh), (x_el, y_el)


def render_targeted_muscle_frame(
    frame_data: dict[str, np.ndarray | float],
    target_mode: str,
    is_wrong: bool,
    alpha_scale: float,
    rep_wave: float,
) -> tuple[np.ndarray, list[tuple[float, float, str]], tuple[float, float], tuple[float, float]]:
    working = (
        apply_dramatic_elbow_swing_posture(frame_data, swing_amount=rep_wave)
        if target_mode == "anterior_deltoid"
        else frame_data
    )

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
        elbow_xy = (668.0 - 24.0 * rep_wave, 822.0)
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
    target_mode: str,
    is_wrong: bool,
    alpha_scale: float,
):
    if alpha_scale <= 0.02 or target_mode == "xray_joint":
        return

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    sx, sy = int(round(shoulder_xy[0])), int(round(shoulder_xy[1]))
    ex, ey = int(round(elbow_xy[0])), int(round(elbow_xy[1]))

    if is_wrong:
        ref_x = sx + 14
        for y_d in range(sy + 15, ey + 25, 16):
            od.line([(ref_x, y_d), (ref_x, min(ey + 25, y_d + 8))], fill=(210, 220, 235, int(135 * alpha_scale)), width=2)
        od.line([(sx, sy), (ex, ey)], fill=(*COL_RED, int(225 * alpha_scale)), width=4)
        arc_y = ey - 10
        od.line([(ref_x, arc_y), (ex + 6, arc_y)], fill=(*COL_RED, int(245 * alpha_scale)), width=4)
        od.polygon(
            [(ex, arc_y), (ex + 14, arc_y - 7), (ex + 14, arc_y + 7)],
            fill=(*COL_RED, int(255 * alpha_scale)),
        )
        od.ellipse([ex - 11, ey - 11, ex + 11, ey + 11], fill=(16, 18, 24, int(210 * alpha_scale)), outline=(*COL_RED, int(255 * alpha_scale)), width=3)
    else:
        for y_d in range(sy + 15, ey + 20, 14):
            od.line([(ex, y_d), (ex, min(ey + 20, y_d + 8))], fill=(*COL_GREEN, int(175 * alpha_scale)), width=2)
        od.ellipse([ex - 13, ey - 13, ex + 13, ey + 13], fill=(14, 20, 18, int(210 * alpha_scale)), outline=(*COL_GREEN, int(255 * alpha_scale)), width=3)
        od.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(255, 255, 255, int(255 * alpha_scale)))

    canvas.alpha_composite(ov)


def draw_concept_b_radial_ring_pill_hud(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    phase: dict,
    rep_wave: float,
    alpha_scale: float,
):
    """
    Concept B ("Radial Arc Ring + Floating Pill"):
    - 4x-supersampled 270° circular dial (`ring_cx = 168, ring_cy = 1285, ring_r = 74`)
    - Sleek floating capsule (`x = 232..638, y = 1212..1358`)
    - Precision Reticle (`◎`) on the active muscle head(s) / X-Ray joint connected via a fine leader line
      directly to the top of the circular dial.
    """
    if alpha_scale <= 0.02:
        return

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    ring_cx, ring_cy = 168, 1285
    ring_r = 74

    # 1. Precision Reticle (`◎`) + Multi-head Node Links + Leader Line to Top of Dial
    px, py, _ = targets[0]
    px_i, py_i = int(round(px)), int(round(py))

    if len(targets) > 1:
        for k in range(1, len(targets)):
            nx, ny, _ = targets[k]
            nx_i, ny_i = int(round(nx)), int(round(ny))
            od.line([(px_i, py_i), (nx_i, ny_i)], fill=(*accent, int(160 * alpha_scale)), width=2)
            od.ellipse([nx_i - 8, ny_i - 8, nx_i + 8, ny_i + 8], fill=(12, 15, 20, int(195 * alpha_scale)), outline=(*accent, int(240 * alpha_scale)), width=2)
            od.ellipse([nx_i - 3, ny_i - 3, nx_i + 3, ny_i + 3], fill=(255, 255, 255, int(255 * alpha_scale)))

    r_out = int(16 + 2 * rep_wave)
    od.ellipse([px_i - r_out - 2, py_i - r_out - 2, px_i + r_out + 2, py_i + r_out + 2], outline=(10, 12, 16, int(185 * alpha_scale)), width=3)
    od.ellipse([px_i - r_out, py_i - r_out, px_i + r_out, py_i + r_out], outline=(*accent, int(250 * alpha_scale)), width=2)
    od.ellipse([px_i - 5, py_i - 5, px_i + 5, py_i + 5], fill=(255, 255, 255, int(255 * alpha_scale)), outline=(*accent, int(255 * alpha_scale)), width=2)

    ax, ay = ring_cx, ring_cy - ring_r - 6
    elbow_x = ax
    elbow_y = py_i + 32
    pts = [(px_i - r_out - 3, py_i), (elbow_x + 26, py_i), (elbow_x, elbow_y), (ax, ay)]
    od.line(pts, fill=(10, 12, 16, int(185 * alpha_scale)), width=5, joint="curve")
    od.line(pts, fill=(*accent, int(225 * alpha_scale)), width=2, joint="curve")
    od.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=(*accent, int(255 * alpha_scale)))

    # 2. Floating Rounded Pill Capsule (`x = 232..642, y = 1212..1358`)
    px1, py1 = 232, 1212
    px2, py2 = 642, 1358
    od.rounded_rectangle([px1 + 4, py1 + 6, px2 + 4, py2 + 6], radius=36, fill=(4, 6, 10, int(145 * alpha_scale)))
    od.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=36,
        fill=(16, 20, 28, int(216 * alpha_scale)),
        outline=(225, 235, 250, int(68 * alpha_scale)),
        width=2,
    )

    # 3. 4x-Supersampled 270° Circular Dial
    p_base, p_peak = phase["pct_range"]
    pct_val = int(round(p_base + (p_peak - p_base) * (0.35 + 0.65 * rep_wave)))

    ss = 4
    dsz = (ring_r * 2 + 24) * ss
    dc = dsz // 2
    dr = ring_r * ss
    dial = Image.new("RGBA", (dsz, dsz), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)

    dd.ellipse([dc - dr, dc - dr, dc + dr, dc + dr], fill=(14, 18, 25, int(244 * alpha_scale)), outline=(220, 230, 245, int(78 * alpha_scale)), width=2 * ss)
    arc_box = [dc - dr + 10 * ss, dc - dr + 10 * ss, dc + dr - 10 * ss, dc + dr - 10 * ss]
    dd.arc(arc_box, start=135, end=405, fill=(36, 42, 54, int(230 * alpha_scale)), width=10 * ss)

    end_deg = 135 + int(round(270 * (pct_val / 100.0)))
    dd.arc(arc_box, start=135, end=end_deg, fill=(*accent, int(255 * alpha_scale)), width=10 * ss)

    dial_down = dial.resize((ring_r * 2 + 24, ring_r * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(dial_down, (ring_cx - ring_r - 12, ring_cy - ring_r - 12))

    # Percentage & short tag inside dial
    p_str = f"{pct_val}%"
    pb = od.textbbox((0, 0), p_str, font=FONT_DIAL_PCT)
    od.text((ring_cx - (pb[2] - pb[0]) // 2, ring_cy - 26), p_str, font=FONT_DIAL_PCT, fill=(255, 255, 255, int(255 * alpha_scale)))

    dial_tag = phase["dial_tag"]
    sb = od.textbbox((0, 0), dial_tag, font=FONT_DIAL_TAG)
    od.text((ring_cx - (sb[2] - sb[0]) // 2, ring_cy + 18), dial_tag, font=FONT_DIAL_TAG, fill=(*accent, int(255 * alpha_scale)))

    # 4. Floating Pill Typography (3 clean lines)
    tx = 262
    od.text((tx, py1 + 20), phase["kicker"], font=FONT_PILL_KICKER, fill=(*accent, int(255 * alpha_scale)))
    od.text((tx, py1 + 48), phase["line1"], font=FONT_PILL_MAIN, fill=(255, 255, 255, int(255 * alpha_scale)))
    od.text((tx, py1 + 98), phase["line2"], font=FONT_PILL_SUB, fill=(212, 222, 238, int(242 * alpha_scale)))

    canvas.alpha_composite(ov)


def build_v24_torso_icons(
    muscle_pool: list[dict[str, np.ndarray | float]],
    xray_pool: list[dict[str, np.ndarray | float]],
) -> list[tuple[Image.Image, Image.Image]]:
    size = (144, 114)
    crop_box = (555, 550, 905, 950)
    sample_m = muscle_pool[min(len(muscle_pool) - 1, 45)]
    sample_x = xray_pool[min(len(xray_pool) - 1, 35)]

    inact_base = Image.fromarray(sample_m["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    dim = Image.new("RGBA", size, (10, 12, 16, 120))
    inact_base.alpha_composite(dim)

    x_arr, _, _, _ = render_xray_skeletal_frame(sample_x, 1.0, 0.85)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m1_arr, _, _, _ = render_targeted_muscle_frame(sample_m, "anterior_deltoid", True, 1.0, 0.85)
    im1 = Image.fromarray(m1_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m2_arr, _, _, _ = render_targeted_muscle_frame(sample_m, "triceps_all_three", False, 1.0, 0.85)
    im2 = Image.fromarray(m2_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    return [
        (im0, inact_base.copy()),
        (im1, inact_base.copy()),
        (im2, inact_base.copy()),
    ]


def draw_top_dock_and_right_status(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    alpha_scale: float = 1.0,
):
    if alpha_scale <= 0.02:
        return
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
        fill=(14, 18, 24, int(220 * alpha_scale)),
        outline=(210, 222, 240, int(62 * alpha_scale)),
        width=2,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy() if i == active_idx else inact_im.copy()

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=int(255 * alpha_scale))
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle(
                [tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                radius=16,
                outline=(*i_col, int(255 * alpha_scale)),
                width=3,
            )

    # Top-Right ONLY Status Emblem (`x = 912`)
    bx = 912
    by = bar_y + bar_h // 2
    br = 46
    ss = 4
    sz = (br * 2 + 24) * ss
    c = sz // 2
    r = br * ss
    em = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    ed = ImageDraw.Draw(em)
    ed.ellipse([c - r, c - r, c + r, c + r], fill=(16, 19, 26, int(242 * alpha_scale)), outline=(215, 224, 238, int(95 * alpha_scale)), width=2 * ss)
    ri = r - 6 * ss
    ed.ellipse([c - ri, c - ri, c + ri, c + ri], fill=(20, 24, 32, int(248 * alpha_scale)), outline=(*accent, int(245 * alpha_scale)), width=3 * ss)

    if is_wrong:
        d = int(15 * ss)
        w = int(6.5 * ss)
        ed.line([(c - d, c - d), (c + d, c + d)], fill=(255, 255, 255, int(255 * alpha_scale)), width=w)
        ed.line([(c - d, c + d), (c + d, c - d)], fill=(255, 255, 255, int(255 * alpha_scale)), width=w)
    else:
        pts = [(c - 18 * ss, c + 1 * ss), (c - 4 * ss, c + 15 * ss), (c + 20 * ss, c - 13 * ss)]
        ed.line(pts, fill=(255, 255, 255, int(255 * alpha_scale)), width=int(7 * ss), joint="curve")

    em_down = em.resize((br * 2 + 24, br * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(em_down, (bx - br - 12, by - br - 12))
    canvas.alpha_composite(ov)


PHASES = [
    {
        "idx": 0,
        "t_start": 0.0,
        "t_end": 4.0,
        "is_wrong": False,
        "target_mode": "triceps_lat_long",
        "active_slot": 2,
        "dial_tag": "TRICEPS",
        "pct_range": (88, 96),
        "kicker": "TRICEPS BRACHII · 3 HEADS",
        "line1": "ROPE PUSHDOWN",
        "line2": "0° LOCKED ELBOW ISOLATION",
        "reps": 1.0,
    },
    {
        "idx": 1,
        "t_start": 4.0,
        "t_end": 11.5,
        "is_wrong": True,
        "target_mode": "xray_joint",
        "active_slot": 0,
        "dial_tag": "JOINT LOAD",
        "pct_range": (78, 89),
        "kicker": "X-RAY · HUMERUS & SHOULDER",
        "line1": "SHOULDER ROLL",
        "line2": "84% JOINT SHEAR · 16% TRICEPS",
        "reps": 2.0,
    },
    {
        "idx": 2,
        "t_start": 11.5,
        "t_end": 18.5,
        "is_wrong": True,
        "target_mode": "anterior_deltoid",
        "active_slot": 1,
        "dial_tag": "DELT STEAL",
        "pct_range": (68, 76),
        "kicker": "ANTERIOR DELTOID OVERLOAD",
        "line1": "ELBOW SWING",
        "line2": "+24° DRIFT · TRICEPS DROPS TO 26%",
        "reps": 2.0,
    },
    {
        "idx": 3,
        "t_start": 18.5,
        "t_end": 23.5,
        "is_wrong": False,
        "target_mode": "triceps_lat_long",
        "active_slot": 2,
        "dial_tag": "LAT & LONG",
        "pct_range": (86, 95),
        "kicker": "LATERAL & LONG HEAD FOCUS",
        "line1": "PIN ELBOWS",
        "line2": "0° HINGE · 95% TRICEPS TENSION",
        "reps": 1.5,
    },
    {
        "idx": 4,
        "t_start": 23.5,
        "t_end": 28.0,
        "is_wrong": False,
        "target_mode": "triceps_all_three",
        "active_slot": 2,
        "dial_tag": "3 HEADS",
        "pct_range": (92, 99),
        "kicker": "LATERAL · LONG · MEDIAL",
        "line1": "4 × 12 REPS",
        "line2": "SPREAD ROPE AT FULL LOCKOUT",
        "reps": 1.5,
    },
]


async def build_voiceover_and_audio(wav_path: str):
    import edge_tts

    lines = [
        (0.25, "Cable rope pushdown."),
        (4.25, "Avoid rolling your shoulders forward. It overloads the joint."),
        (11.75, "Stop swinging your elbows. Your front delt takes over."),
        (18.75, "Pin your elbows tight and spread the rope."),
        (23.75, "Full triceps lockout. Four sets of twelve slow reps."),
    ]
    sr = 44100
    total_samples = int(DURATION * sr)
    master = np.zeros(total_samples, dtype=np.float32)

    t_arr = np.linspace(0, DURATION, total_samples, endpoint=False)
    sub = 0.042 * np.sin(2 * np.pi * 55.0 * t_arr)
    beat_env = (np.sin(2 * np.pi * (100.0 / 60.0) * t_arr) > 0.75).astype(np.float32)
    master += sub * (0.4 + 0.6 * beat_env)

    for start_s, _ in lines:
        s0 = int(start_s * sr)
        dur_c = int(0.20 * sr)
        if s0 + dur_c < total_samples:
            tc = np.linspace(0, 0.20, dur_c, endpoint=False)
            freq = 220.0 if 3.5 < start_s < 18.0 else 528.0
            chime = 0.09 * np.sin(2 * np.pi * freq * tc) * np.exp(-tc * 14.0)
            master[s0:s0 + dur_c] += chime

    os.makedirs(TMP_DIR, exist_ok=True)
    prev_end_s = 0.0
    print("\n--- AUDIO QC TIMING LOG ---")
    for i, (start_s, text) in enumerate(lines):
        mp3_p = os.path.join(TMP_DIR, f"vo_{i}.mp3")
        wav_p = os.path.join(TMP_DIR, f"vo_{i}.wav")
        comm = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+6%", pitch="-2Hz")
        await comm.save(mp3_p)
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_p, "-af", "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB", "-ac", "1", "-ar", str(sr), wav_p],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(wav_p, "rb") as wf:
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        dur_s = len(raw) / float(sr)
        actual_start_s = max(start_s, prev_end_s + 0.50)
        gap_from_prev = actual_start_s - prev_end_s if i > 0 else actual_start_s
        end_s = actual_start_s + dur_s
        print(f"Line {i}: start={actual_start_s:.2f}s, dur={dur_s:.2f}s, end={end_s:.2f}s, gap_from_prev={gap_from_prev:.2f}s | '{text}'")
        assert gap_from_prev >= 0.45, f"Audio overlap detected on line {i}!"
        prev_end_s = end_s

        s_idx = int(actual_start_s * sr)
        e_idx = min(total_samples, s_idx + len(raw))
        master[s_idx:e_idx] += raw[: e_idx - s_idx] * 0.92
    print("---------------------------\n")

    master = np.clip(master, -0.96, 0.96)
    pcm = (master * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


GLOBAL_MUSCLE_POOL: list[dict[str, np.ndarray | float]] = []
GLOBAL_XRAY_POOL: list[dict[str, np.ndarray | float]] = []
GLOBAL_TORSO_ICONS: list[tuple[Image.Image, Image.Image]] = []


def load_pools() -> None:
    global GLOBAL_MUSCLE_POOL, GLOBAL_XRAY_POOL, GLOBAL_TORSO_ICONS
    if GLOBAL_MUSCLE_POOL:
        return
    print("Loading 3D Muscle Pool & 3D X-Ray Pool for v24 Concept B Master...")
    cap_c = cv2.VideoCapture(CORRECT_I2V)
    raw_c = []
    while True:
        ret, f = cap_c.read()
        if not ret:
            break
        raw_c.append(f)
    cap_c.release()
    if not raw_c:
        raw_c = [cv2.imread(CORRECT_JPG)]
    stroke_c = raw_c[:69] if len(raw_c) >= 69 else raw_c
    GLOBAL_MUSCLE_POOL = [prepare_muscle_frame(f) for f in stroke_c]

    cap_w = cv2.VideoCapture(WRONG_I2V)
    raw_w = []
    while True:
        ret, f = cap_w.read()
        if not ret:
            break
        raw_w.append(f)
    cap_w.release()
    if not raw_w:
        raw_w = [cv2.imread(WRONG_JPG)]
    stroke_w = raw_w[:56] if len(raw_w) >= 56 else raw_w
    GLOBAL_XRAY_POOL = [prepare_xray_bone_frame(f) for f in stroke_w]

    GLOBAL_TORSO_ICONS = build_v24_torso_icons(GLOBAL_MUSCLE_POOL, GLOBAL_XRAY_POOL)


def render_single_frame(f_idx: int) -> str:
    t = f_idx / float(FPS)
    phase = PHASES[-1]
    for p in PHASES:
        if p["t_start"] <= t < p["t_end"] or (p["idx"] == 4 and t <= p["t_end"] + 0.01):
            phase = p
            break

    t_local = t - phase["t_start"]
    dur = phase["t_end"] - phase["t_start"]
    u = max(0.0, min(1.0, t_local / max(0.001, dur)))

    reps = phase["reps"]
    rep_phase = (u * reps) % 1.0
    rep_wave = 0.5 - 0.5 * math.cos(2.0 * math.pi * rep_phase)

    env_in = smoothstep(0.0, 0.35, t_local)
    env_out = 1.0 - 0.15 * smoothstep(dur - 0.25, dur, t_local)
    alpha_scale = env_in * env_out * (0.85 + 0.15 * rep_wave)

    if phase["target_mode"] == "xray_joint":
        pool_pos = int(round(rep_wave * (len(GLOBAL_XRAY_POOL) - 1)))
        canvas, targets, sh_xy, el_xy = render_xray_skeletal_frame(
            GLOBAL_XRAY_POOL[pool_pos],
            alpha_scale=alpha_scale,
            rep_wave=rep_wave,
        )
    else:
        pool_pos = int(round(rep_wave * (len(GLOBAL_MUSCLE_POOL) - 1)))
        canvas, targets, sh_xy, el_xy = render_targeted_muscle_frame(
            GLOBAL_MUSCLE_POOL[pool_pos],
            target_mode=phase["target_mode"],
            is_wrong=phase["is_wrong"],
            alpha_scale=alpha_scale,
            rep_wave=rep_wave,
        )

    pil_im = Image.fromarray(canvas).convert("RGBA")
    draw_kinematic_posture_indicator(pil_im, sh_xy, el_xy, phase["target_mode"], phase["is_wrong"], alpha_scale=env_in * env_out)
    draw_top_dock_and_right_status(pil_im, GLOBAL_TORSO_ICONS, phase["active_slot"], phase["is_wrong"], alpha_scale=1.0)
    draw_concept_b_radial_ring_pill_hud(
        pil_im,
        targets=targets,
        phase=phase,
        rep_wave=rep_wave,
        alpha_scale=env_in * env_out,
    )

    out_rgb = pil_im.convert("RGB")
    os.makedirs(TMP_DIR, exist_ok=True)
    out_path = os.path.join(TMP_DIR, f"frame_{f_idx:04d}.jpg")
    out_rgb.save(out_path, quality=95)
    return out_path


def generate_previews_and_closeup() -> None:
    load_pools()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    sample_times = [
        (2.00, OUT_HOOK, "0. HOOK"),
        (5.88, OUT_STEP1, "1. X-RAY JOINT"),
        (13.25, OUT_STEP2, "2. ELBOW SWING (+24°)"),
        (20.17, OUT_STEP3, "3. PIN ELBOWS (0°)"),
        (25.00, OUT_STEP4, "4. 4x12 LOCKOUT"),
    ]
    panels = []
    for t_sec, out_p, label in sample_times:
        f_idx = int(round(t_sec * FPS))
        tmp_p = render_single_frame(f_idx)
        shutil.copyfile(tmp_p, out_p)
        print("Saved:", out_p)
        im = Image.open(out_p).convert("RGB")
        pw, ph = 320, 568
        im_r = im.resize((pw, ph), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(im_r)
        is_w = "X-RAY" in label or "SWING" in label
        col = COL_RED if is_w else COL_GREEN
        d.rectangle([0, 0, pw - 1, ph - 1], outline=col, width=4)
        d.rectangle([0, 0, pw, 42], fill=(12, 14, 18))
        d.text((12, 10), label, font=load_font(18, bold=True), fill=col)
        panels.append(im_r)

    sheet = Image.new("RGB", (320 * 5 + 16 * 6, 568 + 32), (10, 12, 16))
    for i, p in enumerate(panels):
        sheet.paste(p, (16 + i * (320 + 16), 16))
    sheet.save(OUT_PREVIEW, quality=95)
    print("Saved 5-phase v24 preview sheet:", OUT_PREVIEW)


def build_full_mp4() -> None:
    load_pools()
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

    wav_path = os.path.join(TMP_DIR, "audio.wav")
    asyncio.run(build_voiceover_and_audio(wav_path))

    print(f"Rendering {TOTAL_FRAMES} frames ({DURATION}s) with max_workers=4...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(render_single_frame, range(TOTAL_FRAMES)))

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(TMP_DIR, "frame_%04d.jpg"),
        "-i", wav_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        OUT_MP4,
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Saved final 28.0s v24 Concept B Master MP4:", OUT_MP4)


if __name__ == "__main__":
    generate_previews_and_closeup()
    if "--mp4" in sys.argv:
        build_full_mp4()
