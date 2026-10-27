#!/usr/bin/env python3
"""
GymForm 3D — Universal Luxury Biomechanical Reels Engine (v23 Master)
Designed for:
1. Universal Reuse Across All Workouts (Day 1 ~ Day 7) with Minimum Token Cost:
   - Automatic chroma + multi-scale sulcus (근육 고랑) segmentation separates proximal/compensator heads
     from primary/multi-head target muscles directly from the 3D render's native muscle fibers.
   - Native X-Ray Skeletal Joint integration (`gym_day*_wrong_i2v.mp4`) reveals internal bone & joint overload
     (Humerus, Scapula/Glenohumeral joint, Elbow/Olecranon) during mistake phases.
2. Ultra-Clean Luxury Obsidian Telemetry UI/UX (Apple / Porsche / @demicstory Pro Benchmark):
   - Zero body occlusion: Mannequin sits in the upper-right studio zone (`x = 510..880, y = 420..1140`) with
     ZERO text or boxes over the torso or limbs.
   - Precision Anatomical Reticle (`◎`) locks onto the exact active muscle head(s) / X-Ray joint and connects via
     a fine architectural leader line to the Lower-Left Obsidian Telemetry Console (`x = 88..600, y = 1140..1445`).
   - Real-Time Biomechanical Telemetry: Shows EXACTLY WHERE (`Target Head` vs `Compensator/Joint`) and
     HOW MUCH (`% EMG Activation / Strain` + 24-segment micro-LED tension bar + Target vs. Steal split).
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
TMP_DIR = "/workspace/frames_v23_day1"

CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")

OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")
OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v23_qc_preview.jpg")
OUT_COMPARISON = os.path.join(OUT_DIR, "day1_v23_vs_avatar_closeup.jpg")
OUT_HOOK = os.path.join(OUT_DIR, "day1_v23_step0_hook.jpg")
OUT_STEP1 = os.path.join(OUT_DIR, "day1_v23_step1_xray_joint.jpg")
OUT_STEP2 = os.path.join(OUT_DIR, "day1_v23_step2_wrong_muscle.jpg")
OUT_STEP3 = os.path.join(OUT_DIR, "day1_v23_step3_posture_key.jpg")
OUT_STEP4 = os.path.join(OUT_DIR, "day1_v23_step4_target_muscle.jpg")

BG_CHARCOAL = (14, 16, 20)
COL_RED = (255, 62, 50)
COL_AMBER = (255, 148, 42)
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


FONT_LABEL_XS = load_font(16, bold=True)
FONT_TAG_SM = load_font(19, bold=True)
FONT_BAR_SM = load_font(18, bold=True)
FONT_CUE_MD = load_font(34, bold=True)
FONT_PCT_LG = load_font(48, bold=True)
FONT_SETS_LG = load_font(44, bold=True)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / max(1e-6, e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def place_in_studio_canvas(
    mono_res: np.ndarray,
    extra_maps: dict[str, np.ndarray],
    scale: float = 0.84,
    x_shift: int = 34,
    y_shift: int = 50,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    vig = 1.0 - 0.35 * (((yy - H * 0.50) / (H * 0.52)) ** 2 + ((xx - W * 0.56) / (W * 0.58)) ** 2)
    vig = np.clip(vig, 0.46, 1.0)[:, :, None]
    bg_blur = bg_blur * vig

    mask = np.ones((target_h, target_w), dtype=np.float32)
    feather = 90
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

    # Darken the lower-left studio corner (`x < 560, y > 1080`) into deep luxury obsidian charcoal
    ll_wx = 1.0 - np.clip((xx - 480.0) / 110.0, 0.0, 1.0)
    ll_wy = np.clip((yy - 1060.0) / 110.0, 0.0, 1.0)
    ll_dim = (ll_wx * ll_wy * 0.85)[:, :, None]
    bg_arr = np.array(BG_CHARCOAL, dtype=np.float32)[None, None, :]
    canvas = (canvas.astype(np.float32) * (1.0 - ll_dim) + bg_arr * ll_dim).astype(np.uint8)

    out_maps = {}
    for k, arr in extra_maps.items():
        full_arr = np.zeros((H, W), dtype=np.float32)
        full_arr[y0:y0 + target_h, x0:x0 + target_w] = arr * mask
        out_maps[k] = full_arr

    return canvas, out_maps


def prepare_muscle_frame(bgr: np.ndarray) -> dict[str, np.ndarray | float]:
    """
    Universal 3D Muscle & Sulcus Extractor:
    - Neutralizes ambient red/green reflections into matte studio charcoal.
    - Extracts high-chroma 3D muscle mask (`red_core`) and real intermuscular dark sulci (`sulcus_map`)
      directly from the 3D render's native muscle fiber valleys.
    """
    scale = 0.84
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(W * scale), int(H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]
    h_r, w_r = r_ch.shape
    yy_r, xx_r = np.mgrid[:h_r, :w_r].astype(np.float32)

    red_any = np.clip((r_ch - b_ch - 10.0) / 90.0, 0.0, 1.0)
    green_any = np.clip((g_ch - np.maximum(r_ch, b_ch) - 12.0) / 90.0, 0.0, 1.0)

    # Multi-scale muscle fiber luminance & natural dark sulcus detection
    fiber_lum = 0.32 * r_ch + 0.68 * g_ch
    blur_fine = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=1.3, sigmaY=1.3)
    blur_med = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=4.2, sigmaY=4.2)
    blur_wide = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=11.0, sigmaY=11.0)

    fiber_detail = fiber_lum + (fiber_lum - blur_fine) * 2.50 + (blur_fine - blur_med) * 1.30
    fiber_norm_res = np.clip((fiber_detail - 26.0) / 190.0, 0.0, 1.0)

    # Natural dark anatomical valley (sulcus) detector: where local luminance is darker than the surrounding muscle belly
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

    canvas, maps = place_in_studio_canvas(
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
    """
    Extracts the authentic 3D X-Ray Skeletal Arm & Joint structure from `gym_day1_wrong_i2v.mp4`:
    - Preserves the glowing Humerus, Glenohumeral (Shoulder) Joint, Elbow (Olecranon) Joint, and Forearm bones
      inside a semi-translucent matte charcoal arm!
    """
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

    # Neutral matte skin with subtle translucent X-ray cavity
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    non_red_lum = 0.30 * g_ch + 0.70 * b_ch
    xray_cavity_lum = 28.0 + 135.0 * (bone_sharp ** 1.15)

    arm_xray_zone = np.exp(-(((xx_r - 645.0) / 95.0) ** 2 + ((yy_r - 740.0) / 240.0) ** 2))
    arm_xray_gate = np.clip((arm_xray_zone - 0.10) / 0.55, 0.0, 1.0)

    neutral_lum = base_lum * (1.0 - red_any) + (non_red_lum * (1.0 - arm_xray_gate) + xray_cavity_lum * arm_xray_gate) * red_any
    neutral_lum = np.clip(neutral_lum, 14.0, 238.0)

    # True X-Ray Bone & Joint mask inside the arm (`(r_ch - b_ch > 85) * (r_ch > 135)`)
    bone_mask_res = np.clip((r_ch - b_ch - 82.0) / 95.0, 0.0, 1.0) * np.clip((r_ch - 130.0) / 75.0, 0.0, 1.0) * arm_xray_gate
    bone_mask_res = cv2.GaussianBlur(bone_mask_res, (0, 0), sigmaX=1.2, sigmaY=1.2)

    norm = neutral_lum / 255.0
    mono_r = BG_CHARCOAL[0] + (232 - BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = BG_CHARCOAL[1] + (236 - BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = BG_CHARCOAL[2] + (242 - BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    canvas, maps = place_in_studio_canvas(
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
        # Upper humerus / shoulder joint & elbow joint coordinates
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


def render_xray_skeletal_frame(
    xray_data: dict[str, np.ndarray | float],
    alpha_scale: float,
    rep_wave: float,
) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """
    Renders the 3D X-Ray Skeletal Joint Overload frame:
    - Shows the glowing Humerus shaft, Shoulder Joint capsule, and Elbow Joint (Olecranon) inside the translucent arm.
    - Returns the exact 3D joint target coordinates `[(x_sh, y_sh, "SHOULDER JOINT"), (x_el, y_el, "ELBOW JOINT")]`.
    """
    mono = xray_data["mono"].astype(np.float32)
    bone_sharp = xray_data["bone_sharp"]
    bone_mask = xray_data["bone_mask"]
    x_sh, y_sh = float(xray_data["x_sh"]), float(xray_data["y_sh"])
    x_el, y_el = float(xray_data["x_el"]), float(xray_data["y_el"])

    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    # Darken the arm interior slightly to create a true medical X-ray window effect
    xray_window = cv2.GaussianBlur(bone_mask, (0, 0), sigmaX=12.0, sigmaY=12.0)
    xray_window = np.clip(xray_window * 1.1, 0.0, 0.65) * alpha_scale
    out = mono * (1.0 - xray_window[:, :, None] * 0.48)

    # Glowing bone & joint shader (hot ivory/amber core -> crimson red stress contour)
    b_val = np.clip((bone_sharp - 0.12) / 0.78, 0.0, 1.0)
    pulse = 0.88 + 0.12 * rep_wave

    # Emphasize stress hotspots at the Shoulder Joint (`x_sh, y_sh`) and Elbow Joint (`x_el, y_el`)
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
    return np.clip(out, 0, 255).astype(np.uint8), targets


def render_targeted_muscle_frame(
    frame_data: dict[str, np.ndarray | float],
    target_mode: str,
    is_wrong: bool,
    alpha_scale: float,
    rep_wave: float,
) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """
    Data-Driven Anatomical Head & Sulcus Segmentation:
    - Uses BOTH the natural intermuscular dark sulcus map (`natural_sulcus`) from the 3D render AND
      the anatomical head topology (`Deltoid` vs `Triceps Lateral Head`, `Long Head`, `Medial Head`)
      so every muscle head follows the actual 3D fiber bundles and dark anatomical valleys!
    - Returns `(rendered_rgb, target_nodes)` where `target_nodes` are the exact `(x, y, label)` centroids
      of each active muscle head for the precision reticle UI.
    """
    mono = frame_data["mono"].astype(np.float32)
    fiber_norm = frame_data["fiber_norm"]
    red_core = frame_data["red_core"]
    natural_sulcus = frame_data["natural_sulcus"]

    y_top = float(frame_data["y_top"])
    y_bot = float(frame_data["y_bot"])
    x_med = float(frame_data["x_med"])
    span_y = max(130.0, y_bot - y_top)

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    sculpt_gate = np.clip((red_core ** 0.75) * 1.25, 0.0, 1.0)

    # 1. Anatomical Deltoid-Triceps Sulcus (follows the V-shaped deltoid tuberosity & natural dark groove at ~41% span)
    dx_arm = xx - (x_med - 6.0)
    y_delt_v = (y_top + 0.42 * span_y) - 0.0050 * (dx_arm ** 2) + 0.14 * dx_arm

    is_deltoid = np.clip((y_delt_v - yy) / 11.0, 0.0, 1.0)
    is_triceps = np.clip((yy - (y_delt_v - 2.0)) / 11.0, 0.0, 1.0)

    # 2. Lateral Head (outer horseshoe, left of septum) vs Long Head (posterior inner belly, right of septum)
    # In our 3D render, the natural dark septum runs along `x_septum`
    x_septum = (x_med + 2.0) - (yy - (y_top + 0.43 * span_y)) * 0.05
    y_med_arch = (y_bot - 0.25 * span_y) + ((xx - x_med) ** 2) * 0.007

    is_med_head = is_triceps * np.clip((yy - y_med_arch) / 9.0, 0.0, 1.0)
    is_lat_head = is_triceps * np.clip((x_septum - xx + 2.0) / 8.0, 0.0, 1.0) * (1.0 - is_med_head * 0.55)
    is_long_head = is_triceps * np.clip((xx - x_septum + 2.0) / 8.0, 0.0, 1.0) * (1.0 - is_med_head * 0.55)

    groove_delt = np.exp(-((yy - y_delt_v) ** 2) / (2.0 * 2.8 ** 2))
    groove_septum = np.exp(-((xx - x_septum) ** 2) / (2.0 * 2.2 ** 2)) * is_triceps
    groove_med = np.exp(-((yy - y_med_arch) ** 2) / (2.0 * 2.2 ** 2)) * is_triceps

    # Combine geometric head boundaries with the real 3D dark sulcus valleys (`natural_sulcus`)
    if target_mode == "anterior_deltoid":
        primary_mask = is_deltoid * sculpt_gate
        sulcus_active = np.clip(groove_delt * 0.75 + natural_sulcus * 0.55, 0.0, 1.0)
    elif target_mode == "triceps_lat_long":
        primary_mask = np.maximum(is_lat_head, is_long_head * 0.96) * sculpt_gate * (1.0 - is_med_head * 0.30)
        sulcus_active = np.clip(groove_delt * 0.65 + groove_septum * 0.90 + natural_sulcus * 0.65, 0.0, 1.0)
    else:
        # All 3 Triceps Heads (Lateral + Long + Medial)
        primary_mask = is_triceps * sculpt_gate
        sulcus_active = np.clip(groove_delt * 0.65 + groove_septum * 0.92 + groove_med * 0.85 + natural_sulcus * 0.68, 0.0, 1.0)

    total_mask = primary_mask * (1.0 - 0.80 * sulcus_active)
    total_mask = cv2.GaussianBlur(total_mask, (0, 0), sigmaX=1.4, sigmaY=1.4)
    total_mask = np.clip(total_mask * 1.16, 0.0, 1.0) * alpha_scale

    # Deep 3D muscle fiber ridge shader (`f_val`): carves dark inter-fiber valleys and bright ridge striations
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

    # Compute exact centroid coordinates of each active head for the Precision Reticle (`◎`)
    def head_centroid(h_mask: np.ndarray, default_xy: tuple[float, float]) -> tuple[float, float]:
        ys, xs = np.where((h_mask * sculpt_gate) > 0.32)
        if len(ys) > 20:
            return float(np.median(xs)), float(np.median(ys))
        return default_xy

    if target_mode == "anterior_deltoid":
        cx, cy = head_centroid(is_deltoid, (x_med - 12.0, y_top + 0.22 * span_y))
        targets = [(cx, cy, "ANTERIOR DELTOID")]
    elif target_mode == "triceps_lat_long":
        lx, ly = head_centroid(is_lat_head, (x_med - 14.0, y_top + 0.58 * span_y))
        gx, gy = head_centroid(is_long_head, (x_med + 16.0, y_top + 0.62 * span_y))
        targets = [(lx, ly, "LATERAL HEAD"), (gx, gy, "LONG HEAD")]
    else:
        lx, ly = head_centroid(is_lat_head, (x_med - 14.0, y_top + 0.56 * span_y))
        gx, gy = head_centroid(is_long_head, (x_med + 16.0, y_top + 0.60 * span_y))
        mx, my = head_centroid(is_med_head, (x_med - 4.0, y_top + 0.82 * span_y))
        targets = [(lx, ly, "LATERAL HEAD"), (gx, gy, "LONG HEAD"), (mx, my, "MEDIAL HEAD")]

    return np.clip(out, 0, 255).astype(np.uint8), targets


def draw_luxury_biomechanical_hud(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    phase: dict,
    rep_wave: float,
    alpha_scale: float,
):
    """
    Draws the Luxury Obsidian Telemetry HUD + Precision Anatomical Reticle (`◎`):
    1. On the exact active muscle head(s) / X-Ray joint:
       - Primary surgical dual-ring reticle (`◎`) + secondary head micro-nodes connected by a fine vector link,
         with ZERO text over the mannequin body.
    2. Ultra-thin architectural leader line stepping left from the primary reticle into the open left studio space
       (`x = 240`) and connecting smoothly to the Lower-Left Obsidian Telemetry Console (`x = 84..620, y = 1150..1445`).
    3. Inside the Lower-Left Obsidian Telemetry Console:
       - Shows EXACTLY WHERE is targeted (`Target Head` vs `Secondary / Compensator`)
       - Shows EXACTLY HOW MUCH load/activation (`Dynamic %` + 22-bar precision LED meter + Primary vs Secondary split).
    """
    if alpha_scale <= 0.02:
        return

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    sub_accent = COL_AMBER if is_wrong else COL_CYAN

    # 1. Draw Precision Anatomical Reticles (`◎`) on the target muscle heads / joints
    px, py, _ = targets[0]
    px_i, py_i = int(round(px)), int(round(py))

    # Connect multi-head nodes (e.g. Lateral Head <-> Long Head <-> Medial Head) with a subtle surgical hairline
    if len(targets) > 1:
        for k in range(1, len(targets)):
            nx, ny, _ = targets[k]
            nx_i, ny_i = int(round(nx)), int(round(ny))
            od.line(
                [(px_i, py_i), (nx_i, ny_i)],
                fill=(*accent, int(155 * alpha_scale)),
                width=2,
            )
            # Secondary head micro-reticle
            r_sub = 9
            od.ellipse(
                [nx_i - r_sub, ny_i - r_sub, nx_i + r_sub, ny_i + r_sub],
                fill=(12, 15, 20, int(190 * alpha_scale)),
                outline=(*accent, int(235 * alpha_scale)),
                width=2,
            )
            od.ellipse(
                [nx_i - 3, ny_i - 3, nx_i + 3, ny_i + 3],
                fill=(255, 255, 255, int(250 * alpha_scale)),
            )

    # Primary surgical dual-ring reticle (`◎`)
    r_outer = int(16 + 3 * rep_wave)
    r_inner = 5
    od.ellipse(
        [px_i - r_outer - 3, py_i - r_outer - 3, px_i + r_outer + 3, py_i + r_outer + 3],
        outline=(10, 12, 16, int(180 * alpha_scale)),
        width=3,
    )
    od.ellipse(
        [px_i - r_outer, py_i - r_outer, px_i + r_outer, py_i + r_outer],
        outline=(*accent, int(245 * alpha_scale)),
        width=2,
    )
    # Crosshair ticks on the outer reticle ring
    for dx_t, dy_t in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        od.line(
            [
                (px_i + dx_t * (r_outer - 4), py_i + dy_t * (r_outer - 4)),
                (px_i + dx_t * (r_outer + 5), py_i + dy_t * (r_outer + 5)),
            ],
            fill=(255, 255, 255, int(220 * alpha_scale)),
            width=2,
        )
    od.ellipse(
        [px_i - r_inner, py_i - r_inner, px_i + r_inner, py_i + r_inner],
        fill=(255, 255, 255, int(255 * alpha_scale)),
        outline=(*accent, int(255 * alpha_scale)),
        width=2,
    )

    # 2. Architectural Leader Line from Reticle `(px_i, py_i)` -> Left Studio Space -> Telemetry Console (`y = 1148`)
    card_x, card_y = 84, 1148
    card_w, card_h = 548, 294
    elbow_x = 248
    elbow_y = py_i + 34

    leader_pts = [
        (px_i - r_outer - 4, py_i),
        (elbow_x + 28, py_i),
        (elbow_x, elbow_y),
        (elbow_x, card_y),
    ]
    od.line(leader_pts, fill=(10, 12, 16, int(190 * alpha_scale)), width=5, joint="curve")
    od.line(leader_pts, fill=(*accent, int(215 * alpha_scale)), width=2, joint="curve")
    od.ellipse(
        [elbow_x - 4, card_y - 4, elbow_x + 4, card_y + 4],
        fill=(*accent, int(255 * alpha_scale)),
    )

    # 3. Lower-Left Luxury Obsidian Telemetry Console (`x = 84..632, y = 1148..1442`)
    # Outer shadow + frosted obsidian card
    od.rounded_rectangle(
        [card_x + 4, card_y + 6, card_x + card_w + 4, card_y + card_h + 6],
        radius=22,
        fill=(4, 6, 9, int(165 * alpha_scale)),
    )
    od.rounded_rectangle(
        [card_x, card_y, card_x + card_w, card_y + card_h],
        radius=20,
        fill=(14, 18, 24, int(232 * alpha_scale)),
        outline=(210, 222, 240, int(62 * alpha_scale)),
        width=2,
    )
    # Left vertical titanium/accent spine
    od.rounded_rectangle(
        [card_x + 14, card_y + 18, card_x + 20, card_y + card_h - 18],
        radius=3,
        fill=(*accent, int(255 * alpha_scale)),
    )

    ix = card_x + 36
    iy = card_y + 18

    # Dynamic percentage computation (pulsing realistically with rep contraction `rep_wave`)
    p_base, p_peak = phase["pct_range"]
    pct_val = int(round(p_base + (p_peak - p_base) * (0.35 + 0.65 * rep_wave)))

    # Top Header: Category Badge + Target Muscle/Bone Name + Large % Readout on the right
    od.text(
        (ix, iy),
        phase["hud_kicker"],
        font=FONT_LABEL_XS,
        fill=(168, 180, 198, int(235 * alpha_scale)),
    )
    od.text(
        (ix, iy + 22),
        phase["target_name"],
        font=FONT_TAG_SM,
        fill=(*accent, int(255 * alpha_scale)),
    )

    pct_str = f"{pct_val}%"
    pct_bbox = od.textbbox((0, 0), pct_str, font=FONT_PCT_LG)
    pct_w = pct_bbox[2] - pct_bbox[0]
    od.text(
        (card_x + card_w - pct_w - 24, iy - 2),
        pct_str,
        font=FONT_PCT_LG,
        fill=(255, 255, 255, int(255 * alpha_scale)),
    )

    # 22-Segment Precision Micro-LED EMG / Load Bar
    bar_y = iy + 56
    n_segs = 22
    seg_w = 18
    seg_h = 12
    seg_gap = 5
    active_segs = int(round((pct_val / 100.0) * n_segs))
    for s in range(n_segs):
        sx = ix + s * (seg_w + seg_gap)
        if s < active_segs:
            s_col = accent if s < n_segs - 4 else (255, 255, 255)
            od.rounded_rectangle(
                [sx, bar_y, sx + seg_w, bar_y + seg_h],
                radius=3,
                fill=(*s_col, int(245 * alpha_scale)),
            )
        else:
            od.rounded_rectangle(
                [sx, bar_y, sx + seg_w, bar_y + seg_h],
                radius=3,
                fill=(34, 40, 50, int(190 * alpha_scale)),
            )

    # Secondary Split Telemetry Row (shows exact distribution across heads / compensator)
    split_y = bar_y + 24
    s1_label, s1_pct = phase["split_1"]
    s2_label, s2_pct = phase["split_2"]
    od.text(
        (ix, split_y),
        f"{s1_label}  {s1_pct}%",
        font=FONT_BAR_SM,
        fill=(235, 240, 248, int(240 * alpha_scale)),
    )
    od.text(
        (ix + 255, split_y),
        f"{s2_label}  {s2_pct}%",
        font=FONT_BAR_SM,
        fill=(*sub_accent, int(235 * alpha_scale)),
    )

    # Divider hairline
    div_y = split_y + 32
    od.line(
        [(ix, div_y), (card_x + card_w - 22, div_y)],
        fill=(255, 255, 255, int(38 * alpha_scale)),
        width=1,
    )

    # Bottom Coaching Action / Sets × Reps Block
    cue_y = div_y + 12
    od.text(
        (ix, cue_y),
        phase["line1"],
        font=FONT_SETS_LG if phase["is_sets_reps"] else FONT_CUE_MD,
        fill=(255, 255, 255, int(255 * alpha_scale)),
    )
    od.text(
        (ix, cue_y + (50 if phase["is_sets_reps"] else 44)),
        phase["line2"],
        font=FONT_TAG_SM if phase["is_sets_reps"] else FONT_CUE_MD,
        fill=(*accent, int(255 * alpha_scale)),
    )

    canvas.alpha_composite(ov)


def build_v23_torso_icons(
    muscle_pool: list[dict[str, np.ndarray | float]],
    xray_pool: list[dict[str, np.ndarray | float]],
) -> list[tuple[Image.Image, Image.Image]]:
    size = (148, 118)
    crop_box = (555, 550, 905, 950)
    sample_m = muscle_pool[min(len(muscle_pool) - 1, 45)]
    sample_x = xray_pool[min(len(xray_pool) - 1, 35)]

    inact_base = Image.fromarray(sample_m["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    dim = Image.new("RGBA", size, (10, 12, 16, 120))
    inact_base.alpha_composite(dim)

    # Slot 0: X-Ray Bone Joint Overload
    x_arr, _ = render_xray_skeletal_frame(sample_x, 1.0, 0.85)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    # Slot 1: Anterior Deltoid Compensation
    m1_arr, _ = render_targeted_muscle_frame(sample_m, "anterior_deltoid", True, 1.0, 0.85)
    im1 = Image.fromarray(m1_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    # Slot 2: Triceps 3-Head Isolation
    m2_arr, _ = render_targeted_muscle_frame(sample_m, "triceps_all_three", False, 1.0, 0.85)
    im2 = Image.fromarray(m2_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    return [
        (im0, inact_base.copy()),
        (im1, inact_base.copy()),
        (im2, inact_base.copy()),
    ]


def draw_luxury_obsidian_status_emblem(
    ov: Image.Image,
    bx: int,
    by: int,
    br: int,
    accent: tuple[int, int, int],
    is_wrong: bool,
    alpha_scale: float,
):
    ss = 4
    sz = (br * 2 + 24) * ss
    c = sz // 2
    r = br * ss

    em = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    ed = ImageDraw.Draw(em)

    ed.ellipse(
        [c - r, c - r, c + r, c + r],
        fill=(16, 19, 26, int(242 * alpha_scale)),
        outline=(215, 224, 238, int(95 * alpha_scale)),
        width=2 * ss,
    )
    ri = r - 6 * ss
    ed.ellipse(
        [c - ri, c - ri, c + ri, c + ri],
        fill=(20, 24, 32, int(248 * alpha_scale)),
        outline=(*accent, int(245 * alpha_scale)),
        width=3 * ss,
    )

    if is_wrong:
        d = int(15 * ss)
        w = int(6.5 * ss)
        ed.line([(c - d, c - d), (c + d, c + d)], fill=(*accent, int(110 * alpha_scale)), width=w + 2 * ss)
        ed.line([(c - d, c + d), (c + d, c - d)], fill=(*accent, int(110 * alpha_scale)), width=w + 2 * ss)
        ed.line([(c - d, c - d), (c + d, c + d)], fill=(255, 255, 255, int(255 * alpha_scale)), width=w)
        ed.line([(c - d, c + d), (c + d, c - d)], fill=(255, 255, 255, int(255 * alpha_scale)), width=w)
    else:
        pts = [(c - 18 * ss, c + 1 * ss), (c - 4 * ss, c + 15 * ss), (c + 20 * ss, c - 13 * ss)]
        w = int(7 * ss)
        ed.line(pts, fill=(*accent, int(115 * alpha_scale)), width=w + 2 * ss, joint="curve")
        ed.line(pts, fill=(255, 255, 255, int(255 * alpha_scale)), width=w, joint="curve")

    em_down = em.resize((br * 2 + 24, br * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(em_down, (bx - br - 12, by - br - 12))


def draw_glassmorphic_3torso_dock_v23(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_dock_idx: int,
    is_wrong: bool,
    alpha_scale: float = 1.0,
):
    if alpha_scale <= 0.02:
        return
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    tw, th = 148, 118
    gap = 14
    pad = 12
    bar_x = 84
    bar_y = 246
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + 2 * pad

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=22,
        fill=(14, 18, 24, int(225 * alpha_scale)),
        outline=(210, 222, 240, int(62 * alpha_scale)),
        width=2,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy() if i == active_dock_idx else inact_im.copy()

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=int(255 * alpha_scale))
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_dock_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle(
                [tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                radius=16,
                outline=(*i_col, int(255 * alpha_scale)),
                width=3,
            )

    bx = bar_x + bar_w + 82
    by = bar_y + bar_h // 2
    draw_luxury_obsidian_status_emblem(ov, bx, by, 46, accent, is_wrong, alpha_scale)

    canvas.alpha_composite(ov)


PHASES = [
    {
        "idx": 0,
        "t_start": 0.0,
        "t_end": 4.0,
        "is_wrong": False,
        "target_mode": "triceps_lat_long",
        "active_slot": 2,
        "hud_kicker": "TARGET BIOMECHANICS · EMG ACTIVATION",
        "target_name": "TRICEPS BRACHII (3 HEADS)",
        "pct_range": (88, 96),
        "split_1": ("LATERAL & LONG", 92),
        "split_2": ("FRONT DELT", 8),
        "line1": "ROPE PUSHDOWN",
        "line2": "3-HEAD ISOLATION",
        "is_sets_reps": False,
        "reps": 1.0,
    },
    {
        "idx": 1,
        "t_start": 4.0,
        "t_end": 11.5,
        "is_wrong": True,
        "target_mode": "xray_joint",
        "active_slot": 0,
        "hud_kicker": "X-RAY SKELETAL SCAN · JOINT SHEAR",
        "target_name": "SHOULDER & ELBOW JOINT",
        "pct_range": (78, 89),
        "split_1": ("JOINT IMPINGEMENT", 84),
        "split_2": ("TRICEPS LOAD", 16),
        "line1": "SHOULDER ROLL",
        "line2": "BONE & JOINT OVERLOAD",
        "is_sets_reps": False,
        "reps": 2.0,
    },
    {
        "idx": 2,
        "t_start": 11.5,
        "t_end": 18.5,
        "is_wrong": True,
        "target_mode": "anterior_deltoid",
        "active_slot": 1,
        "hud_kicker": "COMPENSATION ERROR · LOAD LEAK",
        "target_name": "ANTERIOR DELTOID (FRONT)",
        "pct_range": (68, 79),
        "split_1": ("FRONT DELT STEAL", 74),
        "split_2": ("TRICEPS LOAD", 26),
        "line1": "ELBOW SWING",
        "line2": "DELTS STEAL TENSION",
        "is_sets_reps": False,
        "reps": 2.0,
    },
    {
        "idx": 3,
        "t_start": 18.5,
        "t_end": 23.5,
        "is_wrong": False,
        "target_mode": "triceps_lat_long",
        "active_slot": 2,
        "hud_kicker": "PRIMARY ISOLATION · LOCKED HINGE",
        "target_name": "LATERAL & LONG HEAD",
        "pct_range": (86, 95),
        "split_1": ("LATERAL HEAD", 58),
        "split_2": ("LONG HEAD", 37),
        "line1": "PIN ELBOWS TIGHT",
        "line2": "ISOLATE TRICEPS HEADS",
        "is_sets_reps": False,
        "reps": 1.5,
    },
    {
        "idx": 4,
        "t_start": 23.5,
        "t_end": 28.0,
        "is_wrong": False,
        "target_mode": "triceps_all_three",
        "active_slot": 2,
        "hud_kicker": "PEAK LOCKOUT · ALL 3 HEADS ACTIVE",
        "target_name": "LATERAL · LONG · MEDIAL",
        "pct_range": (92, 99),
        "split_1": ("LAT & LONG HEAD", 72),
        "split_2": ("MEDIAL HEAD", 27),
        "line1": "4 SETS × 12 REPS",
        "line2": "SPREAD ROPE AT LOCKOUT",
        "is_sets_reps": True,
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
    print("Loading 3D Muscle Pool (Frames 0..68) & 3D X-Ray Skeletal Pool (Frames 0..55)...")
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

    GLOBAL_TORSO_ICONS = build_v23_torso_icons(GLOBAL_MUSCLE_POOL, GLOBAL_XRAY_POOL)


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
        canvas, targets = render_xray_skeletal_frame(
            GLOBAL_XRAY_POOL[pool_pos],
            alpha_scale=alpha_scale,
            rep_wave=rep_wave,
        )
    else:
        pool_pos = int(round(rep_wave * (len(GLOBAL_MUSCLE_POOL) - 1)))
        canvas, targets = render_targeted_muscle_frame(
            GLOBAL_MUSCLE_POOL[pool_pos],
            target_mode=phase["target_mode"],
            is_wrong=phase["is_wrong"],
            alpha_scale=alpha_scale,
            rep_wave=rep_wave,
        )

    pil_im = Image.fromarray(canvas).convert("RGBA")
    draw_glassmorphic_3torso_dock_v23(pil_im, GLOBAL_TORSO_ICONS, phase["active_slot"], phase["is_wrong"], alpha_scale=1.0)
    draw_luxury_biomechanical_hud(
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
        (13.25, OUT_STEP2, "2. DELT STEAL"),
        (20.17, OUT_STEP3, "3. LAT & LONG HEAD"),
        (25.00, OUT_STEP4, "4. ALL 3 HEADS 4x12"),
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
        is_w = "X-RAY" in label or "DELT" in label
        col = COL_RED if is_w else COL_GREEN
        d.rectangle([0, 0, pw - 1, ph - 1], outline=col, width=4)
        d.rectangle([0, 0, pw, 42], fill=(12, 14, 18))
        d.text((12, 10), label, font=load_font(18, bold=True), fill=col)
        panels.append(im_r)

    sheet = Image.new("RGB", (320 * 5 + 16 * 6, 568 + 32), (10, 12, 16))
    for i, p in enumerate(panels):
        sheet.paste(p, (16 + i * (320 + 16), 16))
    sheet.save(OUT_PREVIEW, quality=95)
    print("Saved 5-phase v23 preview sheet:", OUT_PREVIEW)

    # 1:1 Anatomy & X-Ray Close-up Comparison
    crop_rect = (530, 520, 910, 900)
    s1_im = Image.open(OUT_STEP1).convert("RGB").crop((510, 540, 890, 920)).resize((420, 420), Image.Resampling.LANCZOS)
    s2_im = Image.open(OUT_STEP2).convert("RGB").crop(crop_rect).resize((420, 420), Image.Resampling.LANCZOS)
    s3_im = Image.open(OUT_STEP3).convert("RGB").crop(crop_rect).resize((420, 420), Image.Resampling.LANCZOS)
    s4_im = Image.open(OUT_STEP4).convert("RGB").crop(crop_rect).resize((420, 420), Image.Resampling.LANCZOS)

    comp = Image.new("RGB", (420 * 4 + 20 * 5, 420 + 70), (12, 14, 18))
    cd = ImageDraw.Draw(comp)
    crops = [
        (s1_im, "1. X-RAY BONE & JOINT OVERLOAD", COL_RED),
        (s2_im, "2. COMPENSATION: FRONT DELT (74%)", COL_RED),
        (s3_im, "3. ISOLATION: LAT & LONG HEAD (95%)", COL_GREEN),
        (s4_im, "4. PEAK: ALL 3 TRICEPS HEADS (99%)", COL_GREEN),
    ]
    for i, (cim, title, tcol) in enumerate(crops):
        x = 20 + i * (420 + 20)
        comp.paste(cim, (x, 52))
        cd.rectangle([x, 52, x + 419, 52 + 419], outline=tcol, width=3)
        cd.text((x + 8, 16), title, font=load_font(17, bold=True), fill=tcol)
    comp.save(OUT_COMPARISON, quality=96)
    print("Saved close-up comparison:", OUT_COMPARISON)


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
    print("Saved final 28.0s v23 Master MP4:", OUT_MP4)


if __name__ == "__main__":
    generate_previews_and_closeup()
    if "--mp4" in sys.argv:
        build_full_mp4()
