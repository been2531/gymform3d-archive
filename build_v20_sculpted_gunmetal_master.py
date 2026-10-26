#!/usr/bin/env python3
"""
@gymform.3d — Day 1 Cable Rope Pushdown — v20 Sculpted Gunmetal 3D Wrap Master (28.0s)
Incorporates user feedback (t84s383, t85u, t86u):
1. Crisp Zero-Bleed Muscle Contour (No muddy neon blur/bloom! Razor-sharp muscle fiber striations & 1.5px crisp anatomical contour line).
2. Dark Gunmetal / Matte Graphite (`#22262E`) 3D Body-Wrapping Trajectory Ribbons & Orbital Cuffs:
   - Never uses Red/Green for arrows so it NEVER clashes or blends into the Red/Green muscle!
   - Casts a realistic 3D contact shadow onto the muscle underneath, with a sculpted dark gunmetal body and a crisp 1px titanium-silver bevel rim highlight (`#E2E8F0`).
   - Parameterized in 3D cylindrical space around the shoulder/arm/elbow so the ribbon emerges from behind the limb's 3D contour, wraps over the curved surface with true perspective foreshortening, and finishes in a sleek 3D beveled arrowhead.
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
TMP_DIR = "/workspace/frames_v20_day1"

WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")

OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v20_gunmetal_preview.jpg")
OUT_HOOK = os.path.join(OUT_DIR, "day1_v20_step0_hook.jpg")
OUT_STEP1 = os.path.join(OUT_DIR, "day1_v20_step1_injury.jpg")
OUT_STEP2 = os.path.join(OUT_DIR, "day1_v20_step2_wrong_muscle.jpg")
OUT_STEP3 = os.path.join(OUT_DIR, "day1_v20_step3_posture_key.jpg")
OUT_STEP4 = os.path.join(OUT_DIR, "day1_v20_step4_target_muscle.jpg")
OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")

BG_CHARCOAL = (18, 20, 24)
COL_RED = (255, 59, 48)          # #FF3B30 (Muscle & UI Status only)
COL_GREEN = (24, 226, 105)       # #18E269 (Muscle & UI Status only)
COL_WHITE = (255, 255, 255)

# Dark Gunmetal / Titanium Palette for 3D Wrapping Arrows & Rings (High contrast over Red/Green muscles)
GUNMETAL_DARK = (26, 30, 38)     # #1A1E26 Deep Gunmetal Core
GUNMETAL_MID = (48, 54, 66)      # #303642 Matte Anodized Graphite Body
GUNMETAL_RIM = (218, 226, 238)   # #DAE2EE Crisp Titanium Bevel Edge Highlight
SHADOW_COL = (6, 8, 12)          # Contact occlusion shadow on muscle

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


def prepare_fullbleed_frame(bgr: np.ndarray, is_wrong: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = 0.86 if is_wrong else 0.90
    y_shift = 65 if is_wrong else 55

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(W * scale), int(H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]

    raw_red = np.clip((r_ch - np.maximum(g_ch, b_ch)) / 130.0, 0.0, 1.0)
    raw_green = np.clip((g_ch - np.maximum(r_ch, b_ch)) / 115.0, 0.0, 1.0)

    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    blue_excess = np.clip(b_ch - np.maximum(r_ch, g_ch), 0.0, 255.0)
    # Remove any source red/green glow from background luminance so background stays pure dark charcoal
    body_lum = np.clip(lum - 0.45 * blue_excess - 0.55 * (r_ch - np.maximum(g_ch, b_ch)) * (lum < 75.0), 0.0, 255.0)
    sculpt_lum = np.clip(lum - 0.35 * blue_excess + 0.15 * (r_ch * raw_red), 0.0, 255.0)

    # Strict 3D body silhouette gate: prevents ANY color from spilling into the air around the arm!
    body_gate = np.clip((body_lum - 46.0) / 16.0, 0.0, 1.0)
    # High threshold + morphological erosion to strip away the soft neon halo baked into the source clip
    red_core = np.clip((raw_red - 0.36) / 0.44, 0.0, 1.0) * body_gate
    green_core = np.clip((raw_green - 0.32) / 0.46, 0.0, 1.0) * body_gate
    kernel3 = np.ones((3, 3), np.uint8)
    red_diff = cv2.erode(red_core, kernel3, iterations=1)
    green_diff = cv2.erode(green_core, kernel3, iterations=1)

    norm = sculpt_lum / 255.0
    mono_r = BG_CHARCOAL[0] + (235 - BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = BG_CHARCOAL[1] + (239 - BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = BG_CHARCOAL[2] + (245 - BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    gx = cv2.Sobel(sculpt_lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sculpt_lum, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    ridge_res = np.clip(grad_mag / 90.0, 0.0, 1.0)

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
    pulley_roi = full_mono[180:920, 40:310]
    pulley_smooth = cv2.GaussianBlur(pulley_roi, (31, 31), sigmaX=12, sigmaY=12)
    diff_p = pulley_roi.astype(np.float32) - pulley_smooth.astype(np.float32)
    letter_mask = np.clip((np.max(diff_p, axis=2) - 10.0) / 18.0, 0.0, 1.0)[:, :, None]
    full_mono[180:920, 40:310] = (pulley_roi * (1.0 - letter_mask) + pulley_smooth * letter_mask).astype(np.uint8)

    small_bg = cv2.resize(full_mono, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small_bg = cv2.GaussianBlur(small_bg, (0, 0), sigmaX=6.0, sigmaY=6.0)
    bg_blur = cv2.resize(small_bg, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

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


print("Loading 100% full-bleed 3D video pools with crisp zero-bleed anatomical contours...")
WRONG_POOL = load_clean_video_pool(WRONG_I2V, WRONG_JPG, True)
CORRECT_POOL = load_clean_video_pool(CORRECT_I2V, CORRECT_JPG, False)


def render_crisp_anatomical_3d_frame(
    pool,
    rep_phase: float,
    focus_xy: tuple[int, int],
    focus_sigma: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    t: float,
    use_hand_mask: bool = False,
) -> np.ndarray:
    """
    Renders the 3D mannequin with ZERO muddy neon bloom (t86u: "네온이 너무 번지지않게"):
    - Muscle coloring stays strictly inside the anatomical boundary.
    - Internal muscle fibers/striations remain razor-sharp.
    - Only a crisp 1.5px boundary rim line traces the muscle contour (no wide Gaussian haze!).
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

    # Strictly bounded anatomical muscle mask (zero spill outside the physical muscle belly!)
    raw_focus = r_mask * spatial
    if use_hand_mask:
        raw_focus = np.maximum(raw_focus * 0.55, g_mask * 0.90 + spatial * 0.65 * r_mask)

    muscle_focus = np.clip((raw_focus - 0.16) / 0.58, 0.0, 1.0)

    # Crisp 1px anatomical boundary line with ZERO Gaussian bloom!
    M = cv2.GaussianBlur(muscle_focus, (0, 0), sigmaX=1.2, sigmaY=1.2)
    gx = cv2.Sobel(M, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(M, cv2.CV_32F, 0, 1, ksize=3)
    grad_M = np.clip(np.sqrt(gx * gx + gy * gy) * 3.2, 0.0, 1.0)

    iso_line = np.exp(-(((M - 0.35) / 0.045) ** 2)) * grad_M
    fiber_striation = np.clip((M - 0.30) * 1.5, 0.0, 1.0) * (ridge ** 1.8) * spatial * 0.55
    crisp_contour = np.clip((iso_line + fiber_striation) * alpha_scale, 0.0, 1.0)

    # Satin-matte anatomical muscle tint (controlled intensity so it never looks like glowing neon)
    active_fill = np.clip(muscle_focus * (0.25 + 0.52 * alpha_scale), 0.0, 0.76)

    out = mono.astype(np.float32)
    lum_norm = out[:, :, 0] / 255.0
    sculpt_shade = (lum_norm ** 1.20) * 0.82 + 0.12
    col_arr = np.array(col, dtype=np.float32)

    for c_i in range(3):
        target_c = col_arr[c_i] * sculpt_shade
        ch = out[:, :, c_i] * (1.0 - active_fill) + target_c * active_fill
        rim_col = 0.35 * 255.0 + 0.65 * col_arr[c_i]
        ch = ch * (1.0 - crisp_contour * 0.68) + rim_col * (crisp_contour * 0.68)
        out[:, :, c_i] = ch

    return np.clip(out, 0, 255).astype(np.uint8)


def render_3d_wrapping_gunmetal_ribbon(
    canvas_rgb: np.ndarray,
    center_xy: tuple[float, float],
    axis_angle_deg: float,
    cyl_radius_px: float,
    axial_span_px: float,
    theta_start_deg: float,
    theta_end_deg: float,
    alpha_scale: float,
    progress: float = 1.0,
    ribbon_half_width: float = 8.5,
):
    """
    Renders a TRUE 3D Cylindrical Body-Wrapping Ribbon in Dark Gunmetal Charcoal (`#22262E`)
    with a Titanium-Silver 3D Bevel Edge (`#DAE2EE`) and Contact Occlusion Shadow!
    - Parameterized on a 3D cylinder of radius `cyl_radius_px` tilted at `axis_angle_deg`
      (matching the exact 3D orientation of the shoulder, upper arm, or forearm!).
    - As angle `theta` sweeps around the limb cylinder from `theta_start_deg` (emerging from
      behind the limb edge, `theta = -95 deg`) across the front surface (`theta = 0 deg`) to
      `theta_end_deg`, the ribbon width foreshortens naturally by `cos(theta)` and the 3D surface
      normal shades from deep charcoal at the limb's curved edge to titanium-accented gunmetal at the apex!
    - Terminates in a sculpted 3D Gunmetal Arrowhead with a crisp silver bevel edge.
    """
    if alpha_scale <= 0.02 or progress <= 0.05:
        return

    prog = min(1.0, max(0.18, progress))
    t0 = math.radians(theta_start_deg)
    t1 = math.radians(theta_start_deg + (theta_end_deg - theta_start_deg) * prog)

    n_pts = 220
    us = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)
    thetas = t0 + (t1 - t0) * us

    # Local 3D cylinder coordinates:
    # u_axis: along the limb cylinder axis
    # v_perp: perpendicular to the limb axis (across the cylinder diameter: R * sin(theta))
    # z_depth: out of the screen toward the viewer (R * cos(theta))
    u_local = -0.5 * axial_span_px + axial_span_px * prog * (us ** 1.05)
    # Add a graceful helical S-camber so it flows organically along the muscle belly
    v_local = cyl_radius_px * np.sin(thetas)
    z_depth = np.cos(thetas)  # >0 is front of limb, <=0 is wrapping behind the limb edge!

    # Rotate (u_local, v_local) into screen (x, y) coordinates
    ang = math.radians(axis_angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    cx, cy = center_xy
    bx = cx + u_local * ca - v_local * sa
    by = cy + u_local * sa + v_local * ca

    # Perspective foreshortening of ribbon width as it wraps around the cylinder edge:
    # At the limb edge (z_depth near 0), the ribbon is viewed edge-on so it narrows naturally!
    foreshorten = np.clip(z_depth + 0.25, 0.18, 1.0)
    # Also taper gracefully near the start tail
    tail_taper = np.clip(us / 0.22, 0.25, 1.0)
    radii = (ribbon_half_width * foreshorten * tail_taper).astype(np.float32)

    # 3D Beveled Arrowhead at the end of the wrapping ribbon (u = 1.0)
    hx, hy = float(bx[-1]), float(by[-1])
    dx = float(bx[-1] - bx[-4])
    dy = float(by[-1] - by[-4])
    dn = math.hypot(dx, dy) + 1e-6
    tx, ty = dx / dn, dy / dn
    nx, ny = -ty, tx

    wing_len = 32.0
    wing_flare = 19.0
    w_us = np.linspace(0.0, 1.0, 55, dtype=np.float32)
    for sign in (-1.0, 1.0):
        wx = hx - tx * (wing_len * w_us) + sign * nx * (wing_flare * (w_us ** 1.15))
        wy = hy - ty * (wing_len * w_us) + sign * ny * (wing_flare * (w_us ** 1.15))
        w_rad = ribbon_half_width * 0.82 * (1.0 - 0.72 * (w_us ** 1.3))
        w_z = np.full_like(w_us, float(z_depth[-1]))
        bx = np.concatenate([bx, wx])
        by = np.concatenate([by, wy])
        radii = np.concatenate([radii, w_rad])
        z_depth = np.concatenate([z_depth, w_z])

    pad = int(ribbon_half_width * 4.0 + 36)
    x0, x1 = max(0, int(np.min(bx)) - pad), min(W, int(np.max(bx)) + pad)
    y0, y1 = max(0, int(np.min(by)) - pad), min(H, int(np.max(by)) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

    # 1) Distance field for the ribbon & swept arrowhead
    min_d2 = np.full(xx.shape, 1e12, dtype=np.float32)
    nearest_r = np.full(xx.shape, ribbon_half_width, dtype=np.float32)
    nearest_z = np.full(xx.shape, 1.0, dtype=np.float32)

    # Also compute a shifted distance field for the 3D drop/contact shadow on the muscle (+5px x, +7px y)
    sh_min_d2 = np.full(xx.shape, 1e12, dtype=np.float32)

    for i in range(0, len(bx), 64):
        px = bx[i : i + 64, None, None]
        py = by[i : i + 64, None, None]
        pr = radii[i : i + 64, None, None]
        pz = z_depth[i : i + 64, None, None]

        d2 = (xx[None, :, :] - px) ** 2 + (yy[None, :, :] - py) ** 2
        idx = np.argmin(d2, axis=0)
        d2_min = np.take_along_axis(d2, idx[None, :, :], axis=0)[0]
        r_min = np.take_along_axis(pr, idx[None, :, :], axis=0)[0]
        z_min = np.take_along_axis(pz, idx[None, :, :], axis=0)[0]

        upd = d2_min < min_d2
        min_d2 = np.where(upd, d2_min, min_d2)
        nearest_r = np.where(upd, r_min, nearest_r)
        nearest_z = np.where(upd, z_min, nearest_z)

        sd2 = (xx[None, :, :] - (px + 5.0)) ** 2 + (yy[None, :, :] - (py + 7.0)) ** 2
        sd2_min = np.min(sd2, axis=0)
        sh_min_d2 = np.minimum(sh_min_d2, sd2_min)

    dist = np.sqrt(min_d2)
    sh_dist = np.sqrt(sh_min_d2)

    # Occlude the portion wrapping behind the limb cylinder (z < -0.15 fades smoothly into the back edge)
    vis_mask = np.clip((nearest_z + 0.18) / 0.32, 0.0, 1.0) * alpha_scale

    # 3D Contact Shadow under the ribbon (separates the ribbon from the glowing red/green muscle!)
    shadow_alpha = np.exp(-((sh_dist / (nearest_r * 1.65 + 1e-4)) ** 2.0)) * 0.68 * vis_mask

    # Sculpted Dark Gunmetal Body (crisp anti-aliased edge, zero neon blur!)
    body_alpha = (1.0 / (1.0 + np.exp(np.clip((dist - nearest_r) * 2.2, -40.0, 40.0)))) * 0.96 * vis_mask

    # 3D Cylindrical Shading on the Gunmetal Body:
    # Brighter anodized graphite on the front crest of the cylinder (z near 1), deeper charcoal toward the curved sides,
    # plus cross-sectional tube curvature shading!
    cross_u = np.clip(dist / (nearest_r + 1e-4), 0.0, 1.0)
    cyl_light = np.clip(nearest_z, 0.0, 1.0) * (1.0 - 0.45 * (cross_u ** 2))

    # Crisp 1.3px Titanium-Silver Bevel Rim along the outer edges of the dark gunmetal ribbon!
    rim_band = np.exp(-(((dist - (nearest_r - 1.2)) / 1.1) ** 2)) * 0.92 * vis_mask * np.clip(nearest_z + 0.1, 0.2, 1.0)
    # Subtle center spine sheen
    center_sheen = np.exp(-((dist / (nearest_r * 0.28 + 1e-4)) ** 2)) * 0.28 * vis_mask * np.clip(nearest_z, 0.0, 1.0)

    roi = canvas_rgb[y0:y1, x0:x1].astype(np.float32)

    # Step A: Apply contact shadow onto the underlying muscle
    sh_col = np.array(SHADOW_COL, dtype=np.float32)[None, None, :]
    roi = roi * (1.0 - shadow_alpha[:, :, None]) + sh_col * shadow_alpha[:, :, None]

    # Step B: Composite Dark Gunmetal 3D Ribbon Body
    gm_dark = np.array(GUNMETAL_DARK, dtype=np.float32)[None, None, :]
    gm_mid = np.array(GUNMETAL_MID, dtype=np.float32)[None, None, :]
    gm_rim = np.array(GUNMETAL_RIM, dtype=np.float32)[None, None, :]

    body_col = gm_dark * (1.0 - cyl_light[:, :, None]) + gm_mid * cyl_light[:, :, None]
    body_col = body_col * (1.0 - center_sheen[:, :, None]) + gm_rim * center_sheen[:, :, None]
    roi = roi * (1.0 - body_alpha[:, :, None]) + body_col * body_alpha[:, :, None]

    # Step C: Composite Crisp Titanium-Silver Bevel Rim Highlight (gives unmistakable luxury 3D metal/carbon look!)
    roi = roi * (1.0 - rim_band[:, :, None]) + gm_rim * rim_band[:, :, None]

    canvas_rgb[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def render_3d_gunmetal_orbital_cuff(
    canvas_rgb: np.ndarray,
    center: tuple[float, float],
    tilt_deg: float,
    rx: float,
    ry: float,
    accent_col: tuple[int, int, int],
    alpha_scale: float,
):
    """
    Renders a 3D perspective-tilted Gunmetal Orbital Cuff Ring that wraps around the joint
    (back half occluded behind the joint, front half wrapping over the joint in dark gunmetal
    with a crisp silver bevel edge and a tiny precision center dot).
    """
    if alpha_scale <= 0.02:
        return
    cx, cy = center
    pad = int(max(rx, ry) + 28)
    x0, x1 = max(0, int(cx) - pad), min(W, int(cx) + pad)
    y0, y1 = max(0, int(cy) - pad), min(H, int(cy) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    ang = math.radians(tilt_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca

    # Elliptical distance
    ell_r = np.sqrt((u / rx) ** 2 + (v / ry) ** 2 + 1e-6)
    # Approximate radial pixel distance from the ellipse perimeter
    scale_r = 0.5 * (rx + ry)
    d_ring = np.abs(ell_r - 1.0) * scale_r

    # Front half of the 3D tilted cuff has v > -0.15*ry (back half is attenuated as it goes behind the joint)
    front_vis = np.clip((v / (ry + 1e-4) + 0.35) / 0.45, 0.22, 1.0) * alpha_scale

    ring_w = 4.2
    sh_d = np.abs(np.sqrt(((u - 3.0) / rx) ** 2 + ((v - 5.0) / ry) ** 2 + 1e-6) - 1.0) * scale_r
    sh_alpha = np.exp(-((sh_d / (ring_w * 1.7)) ** 2)) * 0.55 * front_vis

    body_alpha = (1.0 / (1.0 + np.exp(np.clip((d_ring - ring_w) * 2.2, -40.0, 40.0)))) * 0.94 * front_vis
    rim_alpha = np.exp(-(((d_ring - (ring_w - 1.0)) / 0.9) ** 2)) * 0.88 * front_vis

    # Small center precision pin (4.5px gunmetal ring + 2.5px accent core)
    d_center = np.sqrt(dx * dx + dy * dy)
    pin_body = (1.0 / (1.0 + np.exp(np.clip((d_center - 6.5) * 2.2, -40.0, 40.0)))) * 0.95 * alpha_scale
    pin_core = (1.0 / (1.0 + np.exp(np.clip((d_center - 3.2) * 2.2, -40.0, 40.0)))) * 0.95 * alpha_scale

    roi = canvas_rgb[y0:y1, x0:x1].astype(np.float32)
    sh_col = np.array(SHADOW_COL, dtype=np.float32)[None, None, :]
    gm_mid = np.array(GUNMETAL_MID, dtype=np.float32)[None, None, :]
    gm_rim = np.array(GUNMETAL_RIM, dtype=np.float32)[None, None, :]
    acc_arr = np.array(accent_col, dtype=np.float32)[None, None, :]

    roi = roi * (1.0 - sh_alpha[:, :, None]) + sh_col * sh_alpha[:, :, None]
    roi = roi * (1.0 - body_alpha[:, :, None]) + gm_mid * body_alpha[:, :, None]
    roi = roi * (1.0 - rim_alpha[:, :, None]) + gm_rim * rim_alpha[:, :, None]
    roi = roi * (1.0 - pin_body[:, :, None]) + gm_mid * pin_body[:, :, None]
    roi = roi * (1.0 - pin_core[:, :, None]) + acc_arr * pin_core[:, :, None]

    canvas_rgb[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def make_torso_anatomy_icon(
    pool,
    focus_xy: tuple[int, int],
    focus_sigma: tuple[float, float],
    col: tuple[int, int, int],
    size: tuple[int, int] = (158, 128),
) -> tuple[Image.Image, Image.Image]:
    active_arr = render_crisp_anatomical_3d_frame(
        pool,
        rep_phase=0.82,
        focus_xy=focus_xy,
        focus_sigma=focus_sigma,
        col=col,
        alpha_scale=1.0,
        t=1.0,
        use_hand_mask=False,
    )
    crop_box = (520, 560, 920, 1020)
    act_im = Image.fromarray(active_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    mono_arr, _, _, _ = pool[int(0.82 * (len(pool) - 1))]
    inact_im = Image.fromarray(mono_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    dim = Image.new("RGBA", size, (10, 12, 16, 115))
    inact_im.alpha_composite(dim)
    return act_im, inact_im


TORSO_ICONS = [
    make_torso_anatomy_icon(WRONG_POOL, (775, 815), (90.0, 90.0), COL_RED),
    make_torso_anatomy_icon(WRONG_POOL, (742, 875), (95.0, 95.0), COL_RED),
    make_torso_anatomy_icon(CORRECT_POOL, (712, 792), (130.0, 145.0), COL_GREEN),
]


def draw_glassmorphic_3torso_dock(canvas: Image.Image, active_dock_idx: int, is_wrong: bool, alpha_scale: float = 1.0):
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
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + 2 * pad

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=24,
        fill=(22, 26, 34, int(215 * alpha_scale)),
        outline=(255, 255, 255, int(55 * alpha_scale)),
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
            # Crisp border without muddy neon blur
            od.rounded_rectangle(
                [tx - 3, ty - 3, tx + tw + 3, ty + th + 3],
                radius=18,
                outline=(*i_col, int(255 * alpha_scale)),
                width=4,
            )

    bx = bar_x + bar_w + 84
    by = bar_y + bar_h // 2
    br = 50
    od.ellipse(
        [bx - br, by - br, bx + br, by + br],
        fill=(*accent, int(248 * alpha_scale)),
        outline=(255, 255, 255, int(235 * alpha_scale)),
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
    if alpha_scale <= 0.02:
        return

    spot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(spot).rounded_rectangle([64, ly - 28, 540, ly + 220], radius=45, fill=(10, 12, 16, int(165 * alpha_scale)))
    spot = spot.filter(ImageFilter.GaussianBlur(radius=24))
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
        stroke_fill=(0, 0, 0, int(230 * alpha_scale)),
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
    if alpha_scale <= 0.02:
        return
    x, y = 94, 1185
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.text(
        (x, y),
        "4×12",
        font=FONT_SETS,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(240 * alpha_scale)),
    )
    od.text(
        (x + 4, y + 92),
        "SETS × SLOW REPS",
        font=FONT_SETS_SUB,
        fill=(*COL_GREEN, int(255 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(235 * alpha_scale)),
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


# 28.0s Full Benchmark Timeline with 3D Cylindrical Body-Wrapping Gunmetal Ribbons
PHASES = [
    {
        # Phase 0 (0.0 - 3.5s): Close-Up 3D Muscle Contour + Gunmetal Helical Wrap around Triceps!
        "idx": 0,
        "t_start": 0.0,
        "t_end": 3.5,
        "is_wrong": False,
        "dock_idx": 2,
        "show_dock": False,
        "col": COL_GREEN,
        "anchor_xy": (712, 792),
        "focus_sigma": (150.0, 160.0),
        "use_hand_mask": False,
        "closeup_zoom": 1.46,
        "closeup_center": (675, 815),
        "rep_period": 3.5,
        "label_y": 1150,
        "kicker": "3D BIOMECHANICS GUIDE",
        "line1": "CABLE ROPE",
        "line2": "PUSHDOWN",
        "sub_line": "TRICEPS LATERAL & LONG HEAD",
        # 3D Cylindrical Wrap around the upper arm / triceps belly
        "wrap_cfg": {
            "center_xy": (710, 790),
            "axis_angle_deg": 78.0,
            "cyl_radius_px": 78.0,
            "axial_span_px": 95.0,
            "theta_start_deg": -95.0,
            "theta_end_deg": 65.0,
        },
        "cuff_cfg": None,
        "show_sets": False,
    },
    {
        # Phase 1 (3.5 - 11.0s): Mistake 1 — Shoulders Roll Forward
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
        # 3D Cylindrical Wrap over the deltoid/shoulder ball showing forward rolling rotation!
        "wrap_cfg": {
            "center_xy": (772, 800),
            "axis_angle_deg": 32.0,
            "cyl_radius_px": 84.0,
            "axial_span_px": -65.0,
            "theta_start_deg": -92.0,
            "theta_end_deg": 68.0,
        },
        "cuff_cfg": {"center": (775, 815), "tilt_deg": -22.0, "rx": 36.0, "ry": 18.0},
        "show_sets": False,
    },
    {
        # Phase 2 (11.0 - 18.0s): Mistake 2 — Elbows Swing Up & Down
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
        # 3D Cylindrical Wrap around the upper arm & elbow sweeping forward/upward!
        "wrap_cfg": {
            "center_xy": (728, 885),
            "axis_angle_deg": -68.0,
            "cyl_radius_px": 76.0,
            "axial_span_px": 110.0,
            "theta_start_deg": -90.0,
            "theta_end_deg": 62.0,
        },
        "cuff_cfg": {"center": (742, 875), "tilt_deg": 18.0, "rx": 38.0, "ry": 19.0},
        "show_sets": False,
    },
    {
        # Phase 3 (18.0 - 23.0s): Correct Technique — Pin Elbows & Spread Rope
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
        # 3D Cylindrical Wrap directly around the pinned elbow/forearm cylinder flaring outward!
        "wrap_cfg": {
            "center_xy": (668, 885),
            "axis_angle_deg": 122.0,
            "cyl_radius_px": 74.0,
            "axial_span_px": 95.0,
            "theta_start_deg": -92.0,
            "theta_end_deg": 66.0,
        },
        # Locked Gunmetal Orbital Cuff Ring around the pinned elbow joint!
        "cuff_cfg": {"center": (690, 912), "tilt_deg": 15.0, "rx": 42.0, "ry": 20.0},
        "show_sets": False,
    },
    {
        # Phase 4 (23.0 - 28.0s): Target Muscle Isolation + 4x12 Signature
        "idx": 4,
        "t_start": 23.0,
        "t_end": 28.0,
        "is_wrong": False,
        "dock_idx": 2,
        "show_dock": True,
        "col": COL_GREEN,
        "anchor_xy": (712, 792),
        "focus_sigma": (140.0, 155.0),
        "use_hand_mask": False,
        "closeup_zoom": 1.08,
        "closeup_center": (660, 840),
        "rep_period": 2.5,
        "label_y": 545,
        "kicker": "✓ 100% MUSCLE ISOLATION",
        "line1": "TRICEPS LATERAL",
        "line2": "& LONG HEAD",
        "sub_line": "→ FULL CONTRACTION AT BOTTOM",
        # 3D Cylindrical Wrap hugging the sculpted Triceps Lateral & Long Head!
        "wrap_cfg": {
            "center_xy": (712, 795),
            "axis_angle_deg": 76.0,
            "cyl_radius_px": 74.0,
            "axial_span_px": 95.0,
            "theta_start_deg": -92.0,
            "theta_end_deg": 64.0,
        },
        "cuff_cfg": {"center": (712, 792), "tilt_deg": 14.0, "rx": 36.0, "ry": 17.0},
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

    # 1. Render Full-Bleed 3D frame with Crisp Zero-Bleed Muscle Contour
    arr = render_crisp_anatomical_3d_frame(
        pool,
        rep_phase=rep_p,
        focus_xy=st["anchor_xy"],
        focus_sigma=st["focus_sigma"],
        col=st["col"],
        alpha_scale=alpha_scale,
        t=t,
        use_hand_mask=st["use_hand_mask"],
    )

    # 2. Render 3D Gunmetal Orbital Joint Cuff (before zoom crop so it scales naturally with the limb!)
    if st["cuff_cfg"] is not None:
        cc = st["cuff_cfg"]
        render_3d_gunmetal_orbital_cuff(
            arr,
            center=cc["center"],
            tilt_deg=cc["tilt_deg"],
            rx=cc["rx"],
            ry=cc["ry"],
            accent_col=st["col"],
            alpha_scale=alpha_scale,
        )

    # 3. Render 3D Cylindrical Body-Wrapping Dark Gunmetal Ribbon (emerges from behind limb, wraps over muscle!)
    if st["wrap_cfg"] is not None:
        wc = st["wrap_cfg"]
        arc_prog = smoothstep(0.10, 0.68, local_t)
        render_3d_wrapping_gunmetal_ribbon(
            arr,
            center_xy=wc["center_xy"],
            axis_angle_deg=wc["axis_angle_deg"],
            cyl_radius_px=wc["cyl_radius_px"],
            axial_span_px=wc["axial_span_px"],
            theta_start_deg=wc["theta_start_deg"],
            theta_end_deg=wc["theta_end_deg"],
            alpha_scale=alpha_scale,
            progress=arc_prog,
            ribbon_half_width=8.5,
        )

    # 4. Apply smooth camera zoom if closeup_zoom > 1.0 (e.g., 0-3.5s Hook)
    z = st["closeup_zoom"]
    if st["idx"] == 0 and local_t > 2.6:
        pullback = smoothstep(2.6, 3.45, local_t)
        z = z * (1.0 - pullback) + 1.0 * pullback
    if z > 1.02:
        cx, cy = st["closeup_center"]
        crop_w, crop_h = int(W / z), int(H / z)
        x0 = max(0, min(W - crop_w, cx - crop_w // 2))
        y0 = max(0, min(H - crop_h, cy - crop_h // 2))
        arr = cv2.resize(arr[y0:y0 + crop_h, x0:x0 + crop_w], (W, H), interpolation=cv2.INTER_LANCZOS4)

    base = Image.fromarray(arr).convert("RGBA")

    # 5. Top Glassmorphic 3-Torso Anatomy Dock
    if st["show_dock"]:
        draw_glassmorphic_3torso_dock(base, st["dock_idx"], st["is_wrong"], alpha_scale=1.0)

    # 6. Minimal Editorial Typography
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

    # 7. Crisp Signature Bottom-Left "4×12" in Phase 4
    if st["show_sets"]:
        draw_signature_sets_reps(base, alpha_scale)

    return base.convert("RGB")


def generate_previews():
    os.makedirs(OUT_DIR, exist_ok=True)
    sample_times = [1.6, 5.5, 13.0, 20.0, 25.0]
    out_paths = [OUT_HOOK, OUT_STEP1, OUT_STEP2, OUT_STEP3, OUT_STEP4]
    titles = [
        "0. HOOK: 3D WRAP (0-3.5s)",
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
    print("Saved 5-phase v20 gunmetal preview sheet:", OUT_PREVIEW)


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
    print("Saved final 28.0s v20 Gunmetal Master MP4:", OUT_MP4)


if __name__ == "__main__":
    generate_previews()
    if "--mp4" in sys.argv:
        render_mp4()
