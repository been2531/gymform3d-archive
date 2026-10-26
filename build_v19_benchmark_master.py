#!/usr/bin/env python3
"""
@gymform.3d — Day 1 Cable Rope Pushdown — v19 Benchmark Master Format (28.0s Full Reel)
Benchmarked directly against De9qPwyGFvK & @demicstory:
1. Full-Bleed 1080x1920 Seamless Studio Extension (Zero cropped box edges!)
2. Anatomical Muscle Contour Neon Shader (Iso-contour & muscle-fiber ridge neon lines that hug the 3D muscle boundaries dynamically on every frame — zero floating PowerPoint lines!)
3. 0.0-3.5s Close-Up 3D Muscle Contour Hook -> 3.5-11.0s Wrong 1 (2.5 slow reps) -> 11.0-18.0s Wrong 2 (2.5 slow reps) -> 18.0-28.0s Correct Technique & Triceps Isolation (3.5 slow reps) = 28.0s full reel!
4. Top Glassmorphic 3-Torso Anatomy Dock (3 anatomical torso miniatures with glowing target muscle zone + active neon border, matching De9qPwyGFvK).
5. Minimal Editorial Typography (No floating boxes) + Signature Glowing "4 × 12" at Bottom-Left.
"""
from __future__ import annotations

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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 30
DURATION = 28.0
TOTAL_FRAMES = int(FPS * DURATION)

BASE_DIR = "/workspace/daily_info_reels"
OUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = "/workspace/frames_v19_day1"

WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")

OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v19_benchmark_preview.jpg")
OUT_HOOK = os.path.join(OUT_DIR, "day1_v19_step0_hook.jpg")
OUT_STEP1 = os.path.join(OUT_DIR, "day1_v19_step1_injury.jpg")
OUT_STEP2 = os.path.join(OUT_DIR, "day1_v19_step2_wrong_muscle.jpg")
OUT_STEP3 = os.path.join(OUT_DIR, "day1_v19_step3_posture_key.jpg")
OUT_STEP4 = os.path.join(OUT_DIR, "day1_v19_step4_target_muscle.jpg")
OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")

BG_CHARCOAL = (18, 20, 24)
COL_RED = (255, 59, 48)       # #FF3B30
COL_GREEN = (24, 226, 105)    # #18E269
COL_WHITE = (255, 255, 255)

PRETENDARD_BLACK = "/home/box/.fonts/Pretendard-Black.otf"
PRETENDARD_BOLD = "/home/box/.fonts/Pretendard-ExtraBold.otf"
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = [
        PRETENDARD_BLACK if bold else PRETENDARD_BOLD,
        "/home/ubuntu/.fonts/Pretendard-Black.otf",
        FALLBACK_BOLD,
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_KICKER = load_font(26, bold=False)
FONT_MAIN = load_font(50, bold=True)
FONT_SUB = load_font(28, bold=False)
FONT_SETS = load_font(86, bold=True)
FONT_SETS_SUB = load_font(26, bold=False)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def cubic_bezier_vec(p0, p1, p2, p3, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    om = 1.0 - u
    x = (om ** 3) * p0[0] + 3 * (om ** 2) * u * p1[0] + 3 * om * (u ** 2) * p2[0] + (u ** 3) * p3[0]
    y = (om ** 3) * p0[1] + 3 * (om ** 2) * u * p1[1] + 3 * om * (u ** 2) * p2[1] + (u ** 3) * p3[1]
    return x, y


def cubic_bezier_tan_scalar(p0, p1, p2, p3, u: float) -> tuple[float, float]:
    om = 1.0 - u
    dx = 3 * (om ** 2) * (p1[0] - p0[0]) + 6 * om * u * (p2[0] - p1[0]) + 3 * (u ** 2) * (p3[0] - p2[0])
    dy = 3 * (om ** 2) * (p1[1] - p0[1]) + 6 * om * u * (p2[1] - p1[1]) + 3 * (u ** 2) * (p3[1] - p2[1])
    n = math.hypot(dx, dy) + 1e-6
    return dx / n, dy / n


def prepare_fullbleed_frame(bgr: np.ndarray, is_wrong: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a 100% seamless Full-Bleed 1080x1920 monochrome 3D frame with ZERO visible box edges,
    plus precomputed muscle mask, hand mask, and anatomical muscle fiber ridge map!
    """
    scale = 0.86 if is_wrong else 0.90
    y_shift = 65 if is_wrong else 55

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(W * scale), int(H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]

    red_diff = np.clip((r_ch - np.maximum(g_ch, b_ch)) / 130.0, 0.0, 1.0)
    green_diff = np.clip((g_ch - np.maximum(r_ch, b_ch)) / 115.0, 0.0, 1.0)

    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    blue_excess = np.clip(b_ch - np.maximum(r_ch, g_ch), 0.0, 255.0)
    sculpt_lum = np.clip(lum - 0.35 * blue_excess + 0.25 * (r_ch * red_diff), 0.0, 255.0)

    norm = sculpt_lum / 255.0
    mono_r = BG_CHARCOAL[0] + (235 - BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = BG_CHARCOAL[1] + (239 - BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = BG_CHARCOAL[2] + (245 - BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    # Compute anatomical fiber ridge & boundary map on sculpt_lum
    gx = cv2.Sobel(sculpt_lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sculpt_lum, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    ridge_res = np.clip(grad_mag / 95.0, 0.0, 1.0)

    # Instead of pasting onto a flat charcoal canvas (which creates a visible box),
    # pad using BORDER_REPLICATE and blur the outer studio environment so it is 100% seamless full-bleed!
    x0 = (W - target_w) // 2
    y0 = max(0, min(H - target_h, (H - target_h) // 2 + y_shift))
    pad_top = y0
    pad_bot = H - (y0 + target_h)
    pad_left = x0
    pad_right = W - (x0 + target_w)

    full_mono = cv2.copyMakeBorder(
        mono_res.astype(np.uint8),
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE,
    )
    # Clean any faint vertical pulley frame text on the far-left column before blurring
    pulley_roi = full_mono[180:920, 40:310]
    pulley_smooth = cv2.GaussianBlur(pulley_roi, (31, 31), sigmaX=12, sigmaY=12)
    # Replace high-frequency bright letters on the dark pulley column
    diff_p = pulley_roi.astype(np.float32) - pulley_smooth.astype(np.float32)
    letter_mask = np.clip((np.max(diff_p, axis=2) - 10.0) / 18.0, 0.0, 1.0)[:, :, None]
    full_mono[180:920, 40:310] = (pulley_roi * (1.0 - letter_mask) + pulley_smooth * letter_mask).astype(np.uint8)

    small_bg = cv2.resize(full_mono, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small_bg = cv2.GaussianBlur(small_bg, (0, 0), sigmaX=6.0, sigmaY=6.0)
    bg_blur = cv2.resize(small_bg, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # Darken very top/bottom subtly like a studio floor/ceiling vignette
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    vig = 1.0 - 0.28 * (((yy - H * 0.52) / (H * 0.55)) ** 2 + ((xx - W * 0.5) / (W * 0.65)) ** 2)
    vig = np.clip(vig, 0.62, 1.0)[:, :, None]
    bg_blur = bg_blur * vig

    mask = np.ones((target_h, target_w), dtype=np.float32)
    feather = 110
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

    red_full = np.zeros((H, W), dtype=np.float32)
    green_full = np.zeros((H, W), dtype=np.float32)
    ridge_full = np.zeros((H, W), dtype=np.float32)

    red_full[y0:y0 + target_h, x0:x0 + target_w] = red_diff * mask
    green_full[y0:y0 + target_h, x0:x0 + target_w] = green_diff * mask
    ridge_full[y0:y0 + target_h, x0:x0 + target_w] = ridge_res * mask

    return canvas, red_full, green_full, ridge_full


def load_clean_video_pool(mp4_path: str, jpg_path: str, is_wrong: bool):
    pool = []
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        while True:
            ret, f = cap.read()
            if not ret:
                break
            pool.append(prepare_fullbleed_frame(f, is_wrong))
        cap.release()
    if not pool and os.path.exists(jpg_path):
        f = cv2.imread(jpg_path)
        if f is not None:
            pool.append(prepare_fullbleed_frame(f, is_wrong))
    return pool


print("Loading 100% full-bleed 3D video pools with anatomical contour maps...")
WRONG_POOL = load_clean_video_pool(WRONG_I2V, WRONG_JPG, True)
CORRECT_POOL = load_clean_video_pool(CORRECT_I2V, CORRECT_JPG, False)


def render_anatomical_contour_3d_frame(
    pool,
    rep_phase: float,
    focus_xy: tuple[int, int],
    focus_sigma: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    t: float,
    use_hand_mask: bool = False,
    closeup_zoom: float = 1.0,
    closeup_center: tuple[int, int] = (680, 820),
) -> np.ndarray:
    """
    Renders the 3D mannequin with:
    1) Selective Deep Muscle Internal Glow (Red for Wrong, Green for Correct)
    2) De9qPwyGFvK-style Volumetric Muscle Contour Neon Lines that wrap directly around
       the active 3D muscle boundary and internal fiber separation ridges!
    """
    n = len(pool)
    idx_f = rep_phase * (n - 1)
    i0 = int(math.floor(idx_f))
    i1 = min(n - 1, i0 + 1)
    frac = idx_f - i0
    if frac < 1e-3 or n == 1:
        mono, r_mask, g_mask, ridge = pool[i0]
        mono = mono.copy()
    else:
        m0, r0, g0, rg0 = pool[i0]
        m1, r1, g1, rg1 = pool[i1]
        mono = cv2.addWeighted(m0, 1.0 - frac, m1, frac, 0)
        r_mask = r0 * (1.0 - frac) + r1 * frac
        g_mask = g0 * (1.0 - frac) + g1 * frac
        ridge = rg0 * (1.0 - frac) + rg1 * frac

    yy, xx = np.ogrid[:H, :W]
    fx, fy = focus_xy
    sx, sy = focus_sigma
    spatial = np.exp(-(((xx - fx) ** 2) / (2.0 * sx * sx) + ((yy - fy) ** 2) / (2.0 * sy * sy))).astype(np.float32)

    # Focused anatomical mask
    muscle_focus = r_mask * (0.22 + 0.78 * spatial)
    if use_hand_mask:
        muscle_focus = np.maximum(muscle_focus * 0.45, g_mask * 0.95 + spatial * 0.55 * r_mask)

    # Smooth muscle field M for extracting De9qPwyGFvK-style neon contour lines!
    M = cv2.GaussianBlur(muscle_focus, (0, 0), sigmaX=5.5, sigmaY=5.5)
    gx = cv2.Sobel(M, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(M, cv2.CV_32F, 0, 1, ksize=3)
    grad_M = np.clip(np.sqrt(gx * gx + gy * gy) * 4.2, 0.0, 1.0)

    # Iso-contour band around the active muscle boundary (wraps the muscle organically!)
    iso_band = np.exp(-(((M - 0.24) / 0.075) ** 2)) * grad_M
    # Inner muscle head separation lines (where ridge is strong inside the active muscle)
    inner_fiber_line = np.clip((M - 0.25) * 1.8, 0.0, 1.0) * (ridge ** 1.6) * spatial

    pulse = 0.85 + 0.15 * math.sin(t * 6.0)
    contour_core = np.clip((iso_band * 1.35 + inner_fiber_line * 1.15) * alpha_scale * pulse, 0.0, 1.0)
    contour_bloom = cv2.GaussianBlur(contour_core, (0, 0), sigmaX=9.0, sigmaY=9.0)
    contour_bloom = np.clip(contour_bloom * 1.8, 0.0, 1.0)

    # Base muscle internal fill
    active_fill = np.clip(muscle_focus * 1.35 * (0.35 + 0.65 * alpha_scale), 0.0, 1.0)

    out = mono.astype(np.float32)
    lum_factor = (out[:, :, 0] / 255.0) * 0.75 + 0.32
    col_arr = np.array(col, dtype=np.float32)

    for c_i in range(3):
        target_c = col_arr[c_i] * lum_factor
        # 1) Muscle body tint
        ch = out[:, :, c_i] * (1.0 - active_fill * 0.82) + target_c * (active_fill * 0.82)
        # 2) Volumetric neon contour bloom around the muscle border
        ch = ch * (1.0 - contour_bloom * 0.65) + col_arr[c_i] * (contour_bloom * 0.85)
        # 3) White-hot neon contour core line tracing the muscle edge (De9qPwyGFvK signature!)
        white_mix = 0.65 * 255.0 + 0.35 * col_arr[c_i]
        ch = ch * (1.0 - contour_core * 0.90) + white_mix * (contour_core * 0.90)
        out[:, :, c_i] = ch

    res = np.clip(out, 0, 255).astype(np.uint8)

    # If closeup_zoom > 1.0 (e.g., for the 0-3.5s Hook), smoothly crop & zoom into the muscle!
    if closeup_zoom > 1.02:
        cx, cy = closeup_center
        crop_w = int(W / closeup_zoom)
        crop_h = int(H / closeup_zoom)
        x0 = max(0, min(W - crop_w, cx - crop_w // 2))
        y0 = max(0, min(H - crop_h, cy - crop_h // 2))
        cropped = res[y0:y0 + crop_h, x0:x0 + crop_w]
        res = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    return res


def render_volumetric_limb_arc(
    canvas_rgb: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    u_end: float = 1.0,
    max_radius: float = 5.0,
):
    """
    Renders a compact, body-hugging 3D biomechanical motion arc right next to the joint/limb
    (like @demicstory's green/red motion sheath) — NOT a long line across the screen!
    """
    if alpha_scale <= 0.02 or u_end <= 0.05:
        return

    u_max = min(1.0, max(0.15, u_end))
    us = np.linspace(0.0, u_max, 180, dtype=np.float32)
    bx, by = cubic_bezier_vec(p0, p1, p2, p3, us)
    rel_u = us / u_max
    radii = max_radius * (0.20 + 0.80 * np.sin(rel_u * math.pi * 0.86) ** 0.75).astype(np.float32)

    hx, hy = float(bx[-1]), float(by[-1])
    tx, ty = cubic_bezier_tan_scalar(p0, p1, p2, p3, float(u_max))
    nx, ny = -ty, tx
    wing_len = 26.0
    wing_flare = 15.0
    w_us = np.linspace(0.0, 1.0, 45, dtype=np.float32)
    for sign in (-1.0, 1.0):
        wx = hx - tx * (wing_len * w_us) + sign * nx * (wing_flare * (w_us ** 1.25))
        wy = hy - ty * (wing_len * w_us) + sign * ny * (wing_flare * (w_us ** 1.25))
        w_rad = max_radius * 0.85 * (1.0 - 0.75 * (w_us ** 1.4))
        bx = np.concatenate([bx, wx])
        by = np.concatenate([by, wy])
        radii = np.concatenate([radii, w_rad])

    pad = int(max_radius * 5.0 + 30)
    x0, x1 = max(0, int(np.min(bx)) - pad), min(W, int(np.max(bx)) + pad)
    y0, y1 = max(0, int(np.min(by)) - pad), min(H, int(np.max(by)) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    min_d2 = np.full(xx.shape, 1e12, dtype=np.float32)
    nearest_r = np.full(xx.shape, max_radius, dtype=np.float32)

    for i in range(0, len(bx), 64):
        cx = bx[i : i + 64, None, None]
        cy = by[i : i + 64, None, None]
        cr = radii[i : i + 64, None, None]
        d2 = (xx[None, :, :] - cx) ** 2 + (yy[None, :, :] - cy) ** 2
        idx = np.argmin(d2, axis=0)
        d2_min = np.take_along_axis(d2, idx[None, :, :], axis=0)[0]
        r_min = np.take_along_axis(cr, idx[None, :, :], axis=0)[0]
        upd = d2_min < min_d2
        min_d2 = np.where(upd, d2_min, min_d2)
        nearest_r = np.where(upd, r_min, nearest_r)

    dist = np.sqrt(min_d2)
    bloom = np.exp(-((dist / (nearest_r * 3.0 + 1e-4)) ** 1.6)) * 0.50 * alpha_scale
    body = np.exp(-((dist / (nearest_r * 1.0 + 1e-4)) ** 2.3)) * 0.92 * alpha_scale
    core = np.exp(-((dist / (nearest_r * 0.40 + 1e-4)) ** 2.5)) * 0.90 * alpha_scale

    roi = canvas_rgb[y0:y1, x0:x1].astype(np.float32)
    col_arr = np.array(col, dtype=np.float32)[None, None, :]
    white_arr = np.array([255.0, 255.0, 255.0], dtype=np.float32)[None, None, :]

    roi = roi * (1.0 - bloom[:, :, None] * 0.6) + col_arr * bloom[:, :, None]
    roi = roi * (1.0 - body[:, :, None]) + col_arr * body[:, :, None]
    roi = roi * (1.0 - core[:, :, None]) + white_arr * core[:, :, None]
    canvas_rgb[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def render_volumetric_target_node(
    canvas_rgb: np.ndarray,
    center: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    t: float,
    ring_radius: float = 26.0,
):
    if alpha_scale <= 0.02:
        return
    cx, cy = center
    pulse = 1.0 + 0.08 * math.sin(t * 5.5)
    r_ring = ring_radius * pulse
    pad = int(r_ring * 2.5 + 24)
    x0, x1 = max(0, int(cx) - pad), min(W, int(cx) + pad)
    y0, y1 = max(0, int(cy) - pad), min(H, int(cy) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    d_ring = np.abs(d - r_ring)

    ring_glow = np.exp(-((d_ring / 8.0) ** 1.6)) * 0.45 * alpha_scale
    ring_body = np.exp(-((d_ring / 2.2) ** 2.2)) * 0.90 * alpha_scale
    ring_core = np.exp(-((d_ring / 1.0) ** 2.4)) * 0.75 * alpha_scale

    pearl_glow = np.exp(-((d / 14.0) ** 1.7)) * 0.60 * alpha_scale
    pearl_body = np.exp(-((d / 6.5) ** 2.4)) * 0.95 * alpha_scale
    pearl_core = np.exp(-((d / 3.2) ** 2.4)) * 0.95 * alpha_scale

    roi = canvas_rgb[y0:y1, x0:x1].astype(np.float32)
    col_arr = np.array(col, dtype=np.float32)[None, None, :]
    white_arr = np.array([255.0, 255.0, 255.0], dtype=np.float32)[None, None, :]

    tot_glow = np.clip(ring_glow + pearl_glow, 0.0, 1.0)[:, :, None]
    tot_body = np.clip(ring_body + pearl_body, 0.0, 1.0)[:, :, None]
    tot_core = np.clip(ring_core + pearl_core, 0.0, 1.0)[:, :, None]

    roi = roi * (1.0 - tot_glow * 0.6) + col_arr * tot_glow
    roi = roi * (1.0 - tot_body) + col_arr * tot_body
    roi = roi * (1.0 - tot_core) + white_arr * tot_core
    canvas_rgb[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def make_torso_anatomy_icon(
    pool,
    focus_xy: tuple[int, int],
    focus_sigma: tuple[float, float],
    col: tuple[int, int, int],
    size: tuple[int, int] = (158, 128),
) -> tuple[Image.Image, Image.Image]:
    """
    Creates the De9qPwyGFvK-style 3D Anatomical Torso Bust Icon ( both active neon-highlighted
    and inactive grayscale versions ) cropped tightly around the upper torso & arm anatomy!
    """
    active_arr = render_anatomical_contour_3d_frame(
        pool,
        rep_phase=0.82,
        focus_xy=focus_xy,
        focus_sigma=focus_sigma,
        col=col,
        alpha_scale=1.0,
        t=1.0,
        use_hand_mask=False,
    )
    # Tight upper-body anatomical bust crop (showing shoulder, scapula, delt, and triceps)
    crop_box = (520, 560, 920, 1020)
    act_im = Image.fromarray(active_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    mono_arr, _, _, _ = pool[int(0.82 * (len(pool) - 1))]
    inact_im = Image.fromarray(mono_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    dim = Image.new("RGBA", size, (10, 12, 16, 115))
    inact_im.alpha_composite(dim)
    return act_im, inact_im


# 3 Torso Anatomy Icons for the Glassmorphic Top Dock (matching De9qPwyGFvK's 3-Torso Bar):
# Icon 0: Shoulder/Rotator Cuff Joint (Mistake 1)
# Icon 1: Anterior Deltoid / Upper Arm Swing (Mistake 2)
# Icon 2: Triceps Lateral & Long Head Isolation (Correct Technique & Target Muscle)
TORSO_ICONS = [
    make_torso_anatomy_icon(WRONG_POOL, (775, 815), (90.0, 90.0), COL_RED),
    make_torso_anatomy_icon(WRONG_POOL, (742, 875), (95.0, 95.0), COL_RED),
    make_torso_anatomy_icon(CORRECT_POOL, (712, 792), (130.0, 145.0), COL_GREEN),
]


def draw_glassmorphic_3torso_dock(canvas: Image.Image, active_dock_idx: int, is_wrong: bool, alpha_scale: float = 1.0):
    """
    Benchmarked directly from De9qPwyGFvK (t_05s, t_15s, t_20s, t_30s):
    A sleek glassmorphic horizontal dock at the top holding 3 anatomical 3D torso busts,
    with the currently targeted muscle zone glowing in neon and framed by a glowing neon border,
    plus our circular ✕ / ✓ badge on the right!
    """
    if alpha_scale <= 0.02:
        return
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    accent = COL_RED if is_wrong else COL_GREEN

    tw, th = 158, 128
    gap = 16
    pad = 14
    bar_x = 92
    bar_y = 246
    bar_w = 3 * tw + 2 * gap + 2 * pad  # 534px
    bar_h = th + 2 * pad                # 156px

    # Frosted glassmorphic dock container
    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=24,
        fill=(24, 28, 36, int(195 * alpha_scale)),
        outline=(255, 255, 255, int(48 * alpha_scale)),
        width=2,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        act_im, inact_im = TORSO_ICONS[i]
        icon_im = act_im.copy() if i == active_dock_idx else inact_im.copy()

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=16, fill=int(255 * alpha_scale))
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_dock_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            # Outer neon glow around the active torso bust (exact De9qPwyGFvK style!)
            glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(glow).rounded_rectangle(
                [tx - 5, ty - 5, tx + tw + 5, ty + th + 5],
                radius=19,
                outline=(*i_col, int(180 * alpha_scale)),
                width=6,
            )
            glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
            canvas.alpha_composite(glow)

            od.rounded_rectangle(
                [tx - 3, ty - 3, tx + tw + 3, ty + th + 3],
                radius=18,
                outline=(*i_col, int(255 * alpha_scale)),
                width=4,
            )

    # Right Circular ✕ / ✓ Status Badge (x = 770, balanced with dock)
    bx = bar_x + bar_w + 84
    by = bar_y + bar_h // 2
    br = 50
    b_glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(b_glow).ellipse([bx - br - 10, by - br - 10, bx + br + 10, by + br + 10], fill=(*accent, int(95 * alpha_scale)))
    b_glow = b_glow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas.alpha_composite(b_glow)

    od.ellipse(
        [bx - br, by - br, bx + br, by + br],
        fill=(*accent, int(245 * alpha_scale)),
        outline=(255, 255, 255, int(230 * alpha_scale)),
        width=3,
    )
    if is_wrong:
        d = 19
        od.line([(bx - d, by - d), (bx + d, by + d)], fill=(255, 255, 255, int(255 * alpha_scale)), width=9)
        od.line([(bx - d, by + d), (bx + d, by - d)], fill=(255, 255, 255, int(255 * alpha_scale)), width=9)
    else:
        pts = [(bx - 22, by + 2), (bx - 4, by + 19), (bx + 24, by - 17)]
        od.line(pts, fill=(255, 255, 255, int(255 * alpha_scale)), width=9, joint="curve")

    canvas.alpha_composite(ov)


def draw_minimal_editorial_copy(
    canvas: Image.Image,
    kicker: str,
    line1: str,
    line2: str,
    sub_line: str,
    col: tuple[int, int, int],
    ly: int,
    alpha_scale: float,
):
    """
    Zero boxes, zero floating arrows across the screen:
    Just a sleek vertical glowing bar + crisp high-contrast editorial typography on the left!
    """
    if alpha_scale <= 0.02:
        return

    spot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(spot).rounded_rectangle([64, ly - 28, 540, ly + 220], radius=45, fill=(10, 12, 16, int(165 * alpha_scale)))
    spot = spot.filter(ImageFilter.GaussianBlur(radius=26))
    canvas.alpha_composite(spot)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bar_x = 92
    od.rounded_rectangle([bar_x, ly + 4, bar_x + 6, ly + 194], radius=3, fill=(*col, int(255 * alpha_scale)))

    lx = 114
    od.text(
        (lx, ly),
        kicker,
        font=FONT_KICKER,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(225 * alpha_scale)),
    )
    od.text(
        (lx, ly + 36),
        line1,
        font=FONT_MAIN,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )
    od.text(
        (lx, ly + 92),
        line2,
        font=FONT_MAIN,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )
    od.text(
        (lx, ly + 158),
        sub_line,
        font=FONT_SUB,
        fill=(225, 232, 240, int(245 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(235 * alpha_scale)),
    )
    canvas.alpha_composite(ov)


def draw_signature_sets_reps(canvas: Image.Image, alpha_scale: float):
    """
    Benchmarked directly from De9qPwyGFvK's glowing bottom-left "3x10" / "3x12":
    Glowing white-and-emerald "4 × 12" with subtle neon bloom at bottom-left Safe Zone!
    """
    if alpha_scale <= 0.02:
        return
    x, y = 94, 1185

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text((x, y), "4×12", font=FONT_SETS, fill=(*COL_GREEN, int(140 * alpha_scale)))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas.alpha_composite(glow)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.text(
        (x, y),
        "4×12",
        font=FONT_SETS,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(235 * alpha_scale)),
    )
    od.text(
        (x + 4, y + 92),
        "SETS × SLOW REPS",
        font=FONT_SETS_SUB,
        fill=(*COL_GREEN, int(255 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(230 * alpha_scale)),
    )
    canvas.alpha_composite(ov)


def get_rep_progress(local_t: float, period: float = 3.0) -> float:
    u = (local_t % period) / period
    if u < 0.45:
        return 0.5 - 0.5 * math.cos(math.pi * (u / 0.45))
    elif u < 0.72:
        return 1.0
    else:
        return 0.5 + 0.5 * math.cos(math.pi * ((u - 0.72) / 0.28))


# 28.0s Full Benchmark Timeline (5 Phases: 0-3.5s Hook + 4 Multi-Rep Biomechanical Phases)
PHASES = [
    {
        # Phase 0 (0.0 - 3.5s): De9qPwyGFvK-style Close-Up Muscle Contour Hook!
        "idx": 0,
        "t_start": 0.0,
        "t_end": 3.5,
        "is_wrong": False,
        "dock_idx": 2,
        "show_dock": False,  # Just like De9qPwyGFvK 0-2.5s: pure close-up muscle contour hook!
        "col": COL_GREEN,
        "anchor_xy": (712, 792),
        "focus_sigma": (155.0, 165.0),
        "use_hand_mask": False,
        "closeup_zoom": 1.48,
        "closeup_center": (675, 815),
        "rep_period": 3.5,
        "label_y": 1150,
        "kicker": "3D BIOMECHANICS GUIDE",
        "line1": "CABLE ROPE",
        "line2": "PUSHDOWN",
        "sub_line": "TRICEPS LATERAL & LONG HEAD",
        "limb_arc": None,
        "show_node": False,
        "show_sets": False,
    },
    {
        # Phase 1 (3.5 - 11.0s, 7.5s = 2.5 Slow Reps): Mistake 1 — Shoulders Roll Forward
        "idx": 1,
        "t_start": 3.5,
        "t_end": 11.0,
        "is_wrong": True,
        "dock_idx": 0,
        "show_dock": True,
        "col": COL_RED,
        "anchor_xy": (775, 815),
        "focus_sigma": (105.0, 105.0),
        "use_hand_mask": False,
        "closeup_zoom": 1.0,
        "closeup_center": (680, 820),
        "rep_period": 3.0,
        "label_y": 545,
        "kicker": "✕ FORM MISTAKE 01",
        "line1": "SHOULDERS",
        "line2": "ROLL FORWARD",
        "sub_line": "→ GRINDS ROTATOR CUFF JOINT",
        # Compact body-hugging motion arc right over the shoulder joint showing forward roll
        "limb_arc": ((835, 745), (785, 710), (725, 730), (695, 790)),
        "show_node": True,
        "show_sets": False,
    },
    {
        # Phase 2 (11.0 - 18.0s, 7.0s = 2.3 Slow Reps): Mistake 2 — Elbows Swing Up & Down
        "idx": 2,
        "t_start": 11.0,
        "t_end": 18.0,
        "is_wrong": True,
        "dock_idx": 1,
        "show_dock": True,
        "col": COL_RED,
        "anchor_xy": (742, 875),
        "focus_sigma": (115.0, 115.0),
        "use_hand_mask": False,
        "closeup_zoom": 1.0,
        "closeup_center": (680, 820),
        "rep_period": 3.0,
        "label_y": 610,
        "kicker": "✕ FORM MISTAKE 02",
        "line1": "ELBOWS SWING",
        "line2": "UP & DOWN",
        "sub_line": "→ FRONT DELT STEALS TENSION",
        # Compact body-hugging arc right along the swinging elbow/forearm path
        "limb_arc": ((665, 965), (620, 900), (620, 825), (665, 765)),
        "show_node": True,
        "show_sets": False,
    },
    {
        # Phase 3 (18.0 - 23.0s, 5.0s = 2 Slow Reps): Correct Technique — Pin Elbows & Spread Rope
        "idx": 3,
        "t_start": 18.0,
        "t_end": 23.0,
        "is_wrong": False,
        "dock_idx": 2,
        "show_dock": True,
        "col": COL_GREEN,
        "anchor_xy": (690, 912),
        "focus_sigma": (125.0, 125.0),
        "use_hand_mask": True,
        "closeup_zoom": 1.0,
        "closeup_center": (680, 820),
        "rep_period": 2.5,
        "label_y": 615,
        "kicker": "✓ CORRECT TECHNIQUE",
        "line1": "PIN ELBOWS &",
        "line2": "SPREAD ROPE",
        "sub_line": "→ LOCKS JOINT · ZERO STRAIN",
        # Compact outward J-curve right at the rope/hands showing the spread at the bottom
        "limb_arc": ((495, 775), (465, 845), (420, 895), (355, 922)),
        "show_node": True,
        "show_sets": False,
    },
    {
        # Phase 4 (23.0 - 28.0s, 5.0s = 2 Slow Reps): Target Muscle Isolation + 4x12 Signature
        "idx": 4,
        "t_start": 23.0,
        "t_end": 28.0,
        "is_wrong": False,
        "dock_idx": 2,
        "show_dock": True,
        "col": COL_GREEN,
        "anchor_xy": (712, 792),
        "focus_sigma": (145.0, 160.0),
        "use_hand_mask": False,
        "closeup_zoom": 1.08,
        "closeup_center": (660, 840),
        "rep_period": 2.5,
        "label_y": 545,
        "kicker": "✓ 100% MUSCLE ISOLATION",
        "line1": "TRICEPS LATERAL",
        "line2": "& LONG HEAD",
        "sub_line": "→ FULL CONTRACTION AT BOTTOM",
        "limb_arc": None,
        "show_node": True,
        "show_sets": True,
    },
]


def render_frame_at_time(t: float) -> Image.Image:
    st = PHASES[-1]
    for p in PHASES:
        if p["t_start"] <= t < p["t_end"]:
            st = p
            break

    local_t = t - st["t_start"]
    seg_dur = st["t_end"] - st["t_start"]
    rep_p = get_rep_progress(local_t, period=st["rep_period"])
    pool = WRONG_POOL if st["is_wrong"] else CORRECT_POOL

    fade_in = smoothstep(0.05, 0.38, local_t)
    fade_out = 1.0 - smoothstep(seg_dur - 0.35, seg_dur - 0.04, local_t)
    alpha_scale = fade_in * fade_out

    # Smooth zoom transition at the end of the 0-3.5s Hook
    z = st["closeup_zoom"]
    if st["idx"] == 0 and local_t > 2.6:
        pullback = smoothstep(2.6, 3.45, local_t)
        z = z * (1.0 - pullback) + 1.0 * pullback

    # 1. Render 100% Full-Bleed 3D frame with De9qPwyGFvK-style Muscle Contour Neon Lines!
    arr = render_anatomical_contour_3d_frame(
        pool,
        rep_phase=rep_p,
        focus_xy=st["anchor_xy"],
        focus_sigma=st["focus_sigma"],
        col=st["col"],
        alpha_scale=alpha_scale,
        t=t,
        use_hand_mask=st["use_hand_mask"],
        closeup_zoom=z,
        closeup_center=st["closeup_center"],
    )

    # 2. Optional compact body-hugging biomechanical motion arc (never crossing the screen!)
    if st["limb_arc"] is not None:
        p0, p1, p2, p3 = st["limb_arc"]
        arc_prog = smoothstep(0.12, 0.70, local_t)
        render_volumetric_limb_arc(arr, p0, p1, p2, p3, st["col"], alpha_scale, u_end=arc_prog, max_radius=5.2)

    # 3. Volumetric Orbital Target Ring (◎) directly on the anatomical joint/muscle
    if st["show_node"]:
        render_volumetric_target_node(arr, st["anchor_xy"], st["col"], alpha_scale, t, ring_radius=24.0)

    base = Image.fromarray(arr).convert("RGBA")

    # 4. Top Glassmorphic 3-Torso Anatomy Dock (Benchmarked from De9qPwyGFvK)
    if st["show_dock"]:
        draw_glassmorphic_3torso_dock(base, st["dock_idx"], st["is_wrong"], alpha_scale=1.0)

    # 5. Minimal Editorial Typography (No floating boxes or cross-screen lines)
    draw_minimal_editorial_copy(
        base,
        st["kicker"],
        st["line1"],
        st["line2"],
        st["sub_line"],
        st["col"],
        st["label_y"],
        alpha_scale,
    )

    # 6. Signature Bottom-Left Glowing "4×12" in Phase 4
    if st["show_sets"]:
        draw_signature_sets_reps(base, alpha_scale)

    return base.convert("RGB")


def generate_previews():
    os.makedirs(OUT_DIR, exist_ok=True)
    sample_times = [1.6, 5.5, 13.0, 20.0, 25.0]
    out_paths = [OUT_HOOK, OUT_STEP1, OUT_STEP2, OUT_STEP3, OUT_STEP4]
    titles = [
        "0. HOOK: 3D CONTOUR (0-3.5s)",
        "1. WRONG: SHOULDER (3.5-11s)",
        "2. WRONG: ELBOW SWING (11-18s)",
        "3. CORRECT: PIN & SPREAD (18-23s)",
        "4. ISOLATION + 4x12 (23-28s)",
    ]

    imgs = []
    for t, p in zip(sample_times, out_paths):
        im = render_frame_at_time(t)
        im.save(p, quality=95)
        imgs.append(im)
        print("Saved:", p)

    pw, ph = 360, 640
    pad = 20
    header_h = 64
    sheet_w = 5 * pw + 6 * pad
    sheet_h = ph + header_h + 2 * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (12, 14, 18))
    sd = ImageDraw.Draw(sheet)
    f_hdr = load_font(20, bold=True)

    for i, (im, title) in enumerate(zip(imgs, titles)):
        x = pad + i * (pw + pad)
        y = header_h + pad
        col = COL_RED if i in (1, 2) else COL_GREEN
        sd.text((x + 6, 20), title, font=f_hdr, fill=col)
        thumb = im.resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        sd.rounded_rectangle([x - 2, y - 2, x + pw + 2, y + ph + 2], radius=14, outline=col, width=3)

    sheet.save(OUT_PREVIEW, quality=95)
    print("Saved 5-phase benchmark preview sheet:", OUT_PREVIEW)


async def build_voiceover_and_audio(wav_path: str):
    import edge_tts

    lines = [
        (0.15, "Cable rope pushdown. Let's fix your form for maximum triceps growth."),
        (3.70, "Mistake one: rolling your shoulders forward grinds the rotator cuff joint."),
        (11.20, "Mistake two: swinging your elbows up and down lets the front delt steal the load."),
        (18.20, "Instead, pin your elbows tight to your ribs and spread the rope at the bottom."),
        (23.20, "This locks in one hundred percent triceps isolation. Perform four sets of twelve slow reps."),
    ]
    sr = 44100
    total_samples = int(DURATION * sr)
    master = np.zeros(total_samples, dtype=np.float32)

    t_arr = np.linspace(0, DURATION, total_samples, endpoint=False)
    sub = 0.045 * np.sin(2 * np.pi * 55.0 * t_arr)
    beat_env = (np.sin(2 * np.pi * (100.0 / 60.0) * t_arr) > 0.75).astype(np.float32)
    master += sub * (0.4 + 0.6 * beat_env)

    for start_s, _ in lines:
        s0 = int(start_s * sr)
        dur_c = int(0.22 * sr)
        if s0 + dur_c < total_samples:
            tc = np.linspace(0, 0.22, dur_c, endpoint=False)
            freq = 220.0 if 3.0 < start_s < 17.0 else 528.0
            chime = 0.10 * np.sin(2 * np.pi * freq * tc) * np.exp(-tc * 14.0)
            master[s0:s0 + dur_c] += chime

    os.makedirs(TMP_DIR, exist_ok=True)
    for i, (start_s, text) in enumerate(lines):
        mp3_p = os.path.join(TMP_DIR, f"vo_{i}.mp3")
        wav_p = os.path.join(TMP_DIR, f"vo_{i}.wav")
        comm = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+4%", pitch="-2Hz")
        await comm.save(mp3_p)
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_p, "-ac", "1", "-ar", str(sr), wav_p],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(wav_p, "rb") as wf:
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        s_idx = int(start_s * sr)
        e_idx = min(total_samples, s_idx + len(raw))
        master[s_idx:e_idx] += raw[: e_idx - s_idx] * 0.92

    master = np.clip(master, -0.96, 0.96)
    pcm = (master * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _render_one_frame(idx: int) -> str:
    t = idx / float(FPS)
    im = render_frame_at_time(t)
    out_p = os.path.join(TMP_DIR, f"frame_{idx:04d}.jpg")
    im.save(out_p, quality=92)
    return out_p


def render_mp4():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

    wav_path = os.path.join(TMP_DIR, "audio.wav")
    asyncio.run(build_voiceover_and_audio(wav_path))

    print(f"Rendering {TOTAL_FRAMES} frames (28.0s) with max_workers=4...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(_render_one_frame, range(TOTAL_FRAMES)))

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        os.path.join(TMP_DIR, "frame_%04d.jpg"),
        "-i",
        wav_path,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        OUT_MP4,
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Saved final 28.0s Benchmark Master MP4:", OUT_MP4)


if __name__ == "__main__":
    generate_previews()
    if "--mp4" in sys.argv:
        render_mp4()
