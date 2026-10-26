#!/usr/bin/env python3
"""
@gymform.3d — Day 1 Cable Rope Pushdown — v18 Volumetric Shader Master Format
True Sub-Pixel Signed Distance Field (SDF) 3D Volumetric Light Ribbons & Orbital Rings
+ Upgraded 3-Tier Cause-and-Effect Biomechanical Editorial Copy.
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
DURATION = 16.0
TOTAL_FRAMES = int(FPS * DURATION)

BASE_DIR = "/workspace/daily_info_reels"
OUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = "/workspace/frames_v18_day1"

WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")

OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v18_volumetric_preview.jpg")
OUT_STEP1 = os.path.join(OUT_DIR, "day1_v18_step1_injury.jpg")
OUT_STEP2 = os.path.join(OUT_DIR, "day1_v18_step2_wrong_muscle.jpg")
OUT_STEP3 = os.path.join(OUT_DIR, "day1_v18_step3_posture_key.jpg")
OUT_STEP4 = os.path.join(OUT_DIR, "day1_v18_step4_target_muscle.jpg")
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


FONT_KICKER = load_font(28, bold=False)
FONT_MAIN = load_font(52, bold=True)
FONT_IMPACT = load_font(30, bold=True)
FONT_SETS = load_font(78, bold=True)
FONT_SETS_SUB = load_font(28, bold=False)
FONT_STEP_NUM = load_font(20, bold=True)


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


def render_volumetric_bezier_ribbon(
    canvas_rgb: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    u_end: float = 1.0,
    max_radius: float = 6.5,
    with_swept_head: bool = True,
):
    """
    True Sub-Pixel Signed Distance Field (SDF) Volumetric 3D Light Ribbon Shader!
    Evaluates exact Euclidean distance to the smooth cubic Bezier curve + curved swept wings,
    producing zero jagged edges, a white-hot 3D specular tube core, rich colored glass body,
    and soft atmospheric neon bloom.
    """
    if alpha_scale <= 0.02 or u_end <= 0.05:
        return

    u_max = min(1.0, max(0.15, u_end))
    n_samples = 260
    us = np.linspace(0.0, u_max, n_samples, dtype=np.float32)
    bx, by = cubic_bezier_vec(p0, p1, p2, p3, us)

    # Smooth width taper along curve parameter u:
    # starts slim (0.28 * max_radius) at tail, swells to full max_radius at u = 0.78*u_max, tapers slightly at tip
    rel_u = us / u_max
    radii = max_radius * (0.25 + 0.75 * np.sin(rel_u * math.pi * 0.88) ** 0.70).astype(np.float32)

    # If swept arrowhead is enabled, add 2 curved wing arcs meeting at the tip!
    if with_swept_head:
        hx, hy = float(bx[-1]), float(by[-1])
        tx, ty = cubic_bezier_tan_scalar(p0, p1, p2, p3, float(u_max))
        nx, ny = -ty, tx
        wing_len = 34.0
        wing_flare = 20.0
        w_us = np.linspace(0.0, 1.0, 65, dtype=np.float32)
        # Curved swept wing flare: flares outward subtly like an aerodynamic fin
        for sign in (-1.0, 1.0):
            # Wing goes from tip (w_u=0) to back wingtip (w_u=1)
            wx = hx - tx * (wing_len * w_us) + sign * nx * (wing_flare * (w_us ** 1.25))
            wy = hy - ty * (wing_len * w_us) + sign * ny * (wing_flare * (w_us ** 1.25))
            w_rad = max_radius * 0.92 * (1.0 - 0.75 * (w_us ** 1.4))
            bx = np.concatenate([bx, wx])
            by = np.concatenate([by, wy])
            radii = np.concatenate([radii, w_rad])

    pad = int(max_radius * 5.5 + 36)
    x0 = max(0, int(np.min(bx)) - pad)
    x1 = min(W, int(np.max(bx)) + pad)
    y0 = max(0, int(np.min(by)) - pad)
    y1 = min(H, int(np.max(by)) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    # Compute distance field on a 0.5x grid for ultra-fast bloom + 1x grid for razor-sharp core
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    # Chunked nearest-point distance calculation to keep memory tiny
    min_d2 = np.full(xx.shape, 1e12, dtype=np.float32)
    nearest_r = np.full(xx.shape, max_radius, dtype=np.float32)

    chunk = 64
    for i in range(0, len(bx), chunk):
        cx = bx[i : i + chunk, None, None]
        cy = by[i : i + chunk, None, None]
        cr = radii[i : i + chunk, None, None]
        d2 = (xx[None, :, :] - cx) ** 2 + (yy[None, :, :] - cy) ** 2
        idx = np.argmin(d2, axis=0)
        d2_min = np.take_along_axis(d2, idx[None, :, :], axis=0)[0]
        r_min = np.take_along_axis(cr, idx[None, :, :], axis=0)[0]
        upd = d2_min < min_d2
        min_d2 = np.where(upd, d2_min, min_d2)
        nearest_r = np.where(upd, r_min, nearest_r)

    dist = np.sqrt(min_d2)

    # Volumetric 3-layer shader:
    # 1) Outer atmospheric bloom
    bloom = np.exp(-((dist / (nearest_r * 3.2 + 1e-4)) ** 1.65)) * 0.55 * alpha_scale
    # 2) Rich colored 3D glass ribbon body (smooth sub-pixel anti-aliased edge)
    body = np.exp(-((dist / (nearest_r * 1.05 + 1e-4)) ** 2.4)) * 0.95 * alpha_scale
    # 3) White-hot specular core spine
    core = np.exp(-((dist / (nearest_r * 0.42 + 1e-4)) ** 2.6)) * 0.92 * alpha_scale

    roi = canvas_rgb[y0:y1, x0:x1].astype(np.float32)
    col_arr = np.array(col, dtype=np.float32)[None, None, :]
    white_arr = np.array([255.0, 255.0, 255.0], dtype=np.float32)[None, None, :]

    # Screen/additive blend bloom, then alpha-composite body and specular core
    roi = roi * (1.0 - bloom[:, :, None] * 0.65) + col_arr * bloom[:, :, None]
    roi = roi * (1.0 - body[:, :, None]) + col_arr * body[:, :, None]
    roi = roi * (1.0 - core[:, :, None]) + white_arr * core[:, :, None]

    canvas_rgb[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def render_volumetric_target_node(
    canvas_rgb: np.ndarray,
    center: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    t: float,
    ring_radius: float = 28.0,
):
    """
    Renders a luxury 3D Volumetric Orbital Target Ring (◎) + Glowing Core Pearl
    using exact radial distance fields (zero jagged edges).
    """
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

    # Outer ring distance field
    d_ring = np.abs(d - r_ring)
    ring_glow = np.exp(-((d_ring / 8.5) ** 1.6)) * 0.45 * alpha_scale
    ring_body = np.exp(-((d_ring / 2.4) ** 2.2)) * 0.90 * alpha_scale
    ring_core = np.exp(-((d_ring / 1.1) ** 2.4)) * 0.75 * alpha_scale

    # Center pearl node distance field
    pearl_glow = np.exp(-((d / 16.0) ** 1.7)) * 0.65 * alpha_scale
    pearl_body = np.exp(-((d / 7.5) ** 2.4)) * 0.96 * alpha_scale
    pearl_core = np.exp(-((d / 3.8) ** 2.4)) * 0.95 * alpha_scale

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


def prepare_pulled_back_frame(bgr: np.ndarray, is_wrong: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = 0.76 if is_wrong else 0.84
    y_shift = 85 if is_wrong else 70

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(W * scale), int(H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]

    red_diff = np.clip((r_ch - np.maximum(g_ch, b_ch)) / 135.0, 0.0, 1.0)
    green_diff = np.clip((g_ch - np.maximum(r_ch, b_ch)) / 120.0, 0.0, 1.0)

    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    blue_excess = np.clip(b_ch - np.maximum(r_ch, g_ch), 0.0, 255.0)
    sculpt_lum = np.clip(lum - 0.35 * blue_excess + 0.25 * (r_ch * red_diff), 0.0, 255.0)

    norm = sculpt_lum / 255.0
    mono_r = BG_CHARCOAL[0] + (232 - BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = BG_CHARCOAL[1] + (236 - BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = BG_CHARCOAL[2] + (242 - BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    canvas = np.full((H, W, 3), BG_CHARCOAL, dtype=np.uint8)
    red_full = np.zeros((H, W), dtype=np.float32)
    green_full = np.zeros((H, W), dtype=np.float32)

    x0 = (W - target_w) // 2
    y0 = max(0, min(H - target_h, (H - target_h) // 2 + y_shift))

    mask = np.ones((target_h, target_w), dtype=np.float32)
    feather = 64
    for i in range(feather):
        f = (i / feather) ** 1.5
        mask[i, :] *= f
        mask[target_h - 1 - i, :] *= f
        mask[:, i] *= f
        mask[:, target_w - 1 - i] *= f
    mask3 = mask[:, :, None]

    roi = canvas[y0:y0 + target_h, x0:x0 + target_w].astype(np.float32)
    canvas[y0:y0 + target_h, x0:x0 + target_w] = (roi * (1.0 - mask3) + mono_res * mask3).astype(np.uint8)
    red_full[y0:y0 + target_h, x0:x0 + target_w] = red_diff * mask
    green_full[y0:y0 + target_h, x0:x0 + target_w] = green_diff * mask

    return canvas, red_full, green_full


def load_clean_video_pool(mp4_path: str, jpg_path: str, is_wrong: bool) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pool = []
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        while True:
            ret, f = cap.read()
            if not ret:
                break
            pool.append(prepare_pulled_back_frame(f, is_wrong))
        cap.release()
    if not pool and os.path.exists(jpg_path):
        f = cv2.imread(jpg_path)
        if f is not None:
            pool.append(prepare_pulled_back_frame(f, is_wrong))
    return pool


print("Loading and neutralizing full-shot 3D video pools...")
WRONG_POOL = load_clean_video_pool(WRONG_I2V, WRONG_JPG, True)
CORRECT_POOL = load_clean_video_pool(CORRECT_I2V, CORRECT_JPG, False)


def render_focused_3d_array(
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rep_phase: float,
    focus_xy: tuple[int, int],
    focus_sigma: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    use_hand_mask: bool = False,
) -> np.ndarray:
    n = len(pool)
    idx_f = rep_phase * (n - 1)
    i0 = int(math.floor(idx_f))
    i1 = min(n - 1, i0 + 1)
    frac = idx_f - i0
    if frac < 1e-3 or n == 1:
        mono, r_mask, g_mask = pool[i0]
        mono = mono.copy()
    else:
        m0, r0, g0 = pool[i0]
        m1, r1, g1 = pool[i1]
        mono = cv2.addWeighted(m0, 1.0 - frac, m1, frac, 0)
        r_mask = r0 * (1.0 - frac) + r1 * frac
        g_mask = g0 * (1.0 - frac) + g1 * frac

    yy, xx = np.ogrid[:H, :W]
    fx, fy = focus_xy
    sx, sy = focus_sigma
    spatial = np.exp(-(((xx - fx) ** 2) / (2.0 * sx * sx) + ((yy - fy) ** 2) / (2.0 * sy * sy))).astype(np.float32)

    active_3d = r_mask * (0.28 + 0.72 * spatial)
    if use_hand_mask:
        active_3d = np.maximum(active_3d * 0.30, g_mask * 0.95 + spatial * 0.35 * r_mask)

    active_3d = np.clip(active_3d * 1.45 * alpha_scale, 0.0, 1.0)

    out = mono.astype(np.float32)
    lum_factor = (out[:, :, 0] / 255.0) * 0.75 + 0.35
    for c_i in range(3):
        target_c = col[c_i] * lum_factor
        out[:, :, c_i] = out[:, :, c_i] * (1.0 - active_3d) + target_c * active_3d

    return np.clip(out, 0, 255).astype(np.uint8)


def get_rep_progress(local_t: float, period: float = 4.0) -> float:
    u = (local_t % period) / period
    if u < 0.45:
        return 0.5 - 0.5 * math.cos(math.pi * (u / 0.45))
    elif u < 0.72:
        return 1.0
    else:
        return 0.5 + 0.5 * math.cos(math.pi * ((u - 0.72) / 0.28))


# 4 Sequential Steps with Upgraded 3-Tier Cause-and-Effect Explanations & Sweeping Volumetric Curves
STEPS = [
    {
        "step_idx": 0,
        "t_start": 0.0,
        "t_end": 4.0,
        "is_wrong": True,
        "col": COL_RED,
        "anchor_xy": (775, 815),
        "focus_sigma": (110.0, 110.0),
        "use_hand_mask": False,
        "label_y": 540,
        "kicker": "✕ FORM MISTAKE 01",
        "line1": "SHOULDERS",
        "line2": "ROLL FORWARD",
        "impact": "GRINDS ROTATOR CUFF JOINT",
        # High-arch sweeping volumetric Bezier ribbon from typography block curving down into the shoulder joint
        "curve_p0": (495, 625),
        "curve_p1": (635, 555),
        "curve_p2": (740, 640),
        "curve_p3": (772, 795),
        "show_sets": False,
    },
    {
        "step_idx": 1,
        "t_start": 4.0,
        "t_end": 8.0,
        "is_wrong": True,
        "col": COL_RED,
        "anchor_xy": (742, 875),
        "focus_sigma": (115.0, 115.0),
        "use_hand_mask": False,
        "label_y": 620,
        "kicker": "✕ FORM MISTAKE 02",
        "line1": "ELBOWS SWING",
        "line2": "UP & DOWN",
        "impact": "FRONT DELT STEALS THE LOAD",
        # Sweeping volumetric Bezier ribbon arcing from typography into the anterior deltoid / upper arm
        "curve_p0": (505, 705),
        "curve_p1": (625, 645),
        "curve_p2": (705, 725),
        "curve_p3": (738, 855),
        "show_sets": False,
    },
    {
        "step_idx": 2,
        "t_start": 8.0,
        "t_end": 12.0,
        "is_wrong": False,
        "col": COL_GREEN,
        "anchor_xy": (690, 912),
        "focus_sigma": (120.0, 120.0),
        "use_hand_mask": True,
        "label_y": 640,
        "kicker": "✓ CORRECT TECHNIQUE",
        "line1": "PIN ELBOWS &",
        "line2": "SPREAD ROPE",
        "impact": "LOCKS JOINT · ZERO STRAIN",
        # Sweeping volumetric Bezier ribbon curving into the locked elbow joint + secondary rope spread arc
        "curve_p0": (495, 725),
        "curve_p1": (605, 685),
        "curve_p2": (665, 775),
        "curve_p3": (686, 892),
        "extra_motion_curve": ((505, 770), (470, 840), (425, 895), (355, 925)),
        "show_sets": False,
    },
    {
        "step_idx": 3,
        "t_start": 12.0,
        "t_end": 16.0,
        "is_wrong": False,
        "col": COL_GREEN,
        "anchor_xy": (712, 792),
        "focus_sigma": (140.0, 155.0),
        "use_hand_mask": False,
        "label_y": 545,
        "kicker": "✓ 100% MUSCLE ISOLATION",
        "line1": "TRICEPS LATERAL",
        "line2": "& LONG HEAD",
        "impact": "FULL CONTRACTION AT BOTTOM",
        # Sweeping volumetric Bezier ribbon arcing into the sculpted Triceps Lateral & Long head belly
        "curve_p0": (535, 630),
        "curve_p1": (635, 575),
        "curve_p2": (695, 655),
        "curve_p3": (710, 772),
        "show_sets": True,
    },
]


def make_mini_thumb(step_cfg: dict, size: tuple[int, int] = (142, 118)) -> Image.Image:
    pool = WRONG_POOL if step_cfg["is_wrong"] else CORRECT_POOL
    arr = render_focused_3d_array(
        pool,
        rep_phase=0.85,
        focus_xy=step_cfg["anchor_xy"],
        focus_sigma=step_cfg["focus_sigma"],
        col=step_cfg["col"],
        alpha_scale=1.0,
        use_hand_mask=step_cfg["use_hand_mask"],
    )
    render_volumetric_target_node(arr, step_cfg["anchor_xy"], step_cfg["col"], 1.0, 1.0, ring_radius=24.0)
    im = Image.fromarray(arr).convert("RGBA")
    cropped = im.crop((240, 480, 900, 1180)).resize(size, Image.Resampling.LANCZOS)
    return cropped


THUMBS = [make_mini_thumb(s) for s in STEPS]


def draw_top_balanced_preview_bar(canvas: Image.Image, step_idx: int, is_wrong: bool):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    accent = COL_RED if is_wrong else COL_GREEN

    tw, th = 142, 118
    gap = 12
    pad = 12
    bar_x = 86
    bar_y = 244
    bar_w = 4 * tw + 3 * gap + 2 * pad
    bar_h = th + 2 * pad

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=22,
        fill=(22, 25, 32, 220),
        outline=(255, 255, 255, 42),
        width=2,
    )

    for i, thumb in enumerate(THUMBS):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        t_img = thumb.copy()
        if i != step_idx:
            dim = Image.new("RGBA", t_img.size, (10, 12, 16, 125))
            t_img.alpha_composite(dim)
        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=255)
        ov.paste(t_img, (tx, ty), mask)

        t_col = COL_RED if i < 2 else COL_GREEN
        mark = "✕" if i < 2 else "✓"
        od.rounded_rectangle([tx + 6, ty + 6, tx + 48, ty + 30], radius=7, fill=(12, 14, 18, 200))
        od.text((tx + 12, ty + 7), f"{mark}{i+1}", font=FONT_STEP_NUM, fill=(*t_col, 240))

        if i == step_idx:
            od.rounded_rectangle(
                [tx - 3, ty - 3, tx + tw + 3, ty + th + 3],
                radius=16,
                outline=(*t_col, 255),
                width=4,
            )

    bx = bar_x + bar_w + 68
    by = bar_y + bar_h // 2
    br = 48
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([bx - br - 10, by - br - 10, bx + br + 10, by + br + 10], fill=(*accent, 90))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas.alpha_composite(glow)

    od.ellipse([bx - br, by - br, bx + br, by + br], fill=(*accent, 245), outline=(255, 255, 255, 225), width=3)
    if is_wrong:
        d = 18
        od.line([(bx - d, by - d), (bx + d, by + d)], fill=COL_WHITE, width=9)
        od.line([(bx - d, by + d), (bx + d, by - d)], fill=COL_WHITE, width=9)
    else:
        pts = [(bx - 21, by + 2), (bx - 4, by + 18), (bx + 23, by - 16)]
        od.line(pts, fill=COL_WHITE, width=9, joint="curve")

    canvas.alpha_composite(ov)


def draw_editorial_typography_block(canvas: Image.Image, step_cfg: dict, alpha_scale: float):
    """
    Draws the luxury 3-tier Cause -> Main Action/Target -> Anatomical Impact Pill
    with a glowing vertical accent bar on the left.
    """
    if alpha_scale <= 0.02:
        return

    col = step_cfg["col"]
    ly = step_cfg["label_y"]

    spot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spot)
    sd.rounded_rectangle([62, ly - 32, 555, ly + 235], radius=48, fill=(8, 10, 14, int(190 * alpha_scale)))
    spot = spot.filter(ImageFilter.GaussianBlur(radius=28))
    canvas.alpha_composite(spot)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # Vertical glowing accent bar
    bar_x = 86
    od.rounded_rectangle([bar_x, ly + 4, bar_x + 7, ly + 206], radius=3, fill=(*col, int(255 * alpha_scale)))

    lx = 110
    # 1) Kicker (e.g. "✕ FORM MISTAKE 01")
    od.text(
        (lx, ly),
        step_cfg["kicker"],
        font=FONT_KICKER,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(230 * alpha_scale)),
    )
    # 2) Main Line 1 (Pure White)
    od.text(
        (lx, ly + 38),
        step_cfg["line1"],
        font=FONT_MAIN,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )
    # 3) Main Line 2 (Accent Color)
    od.text(
        (lx, ly + 96),
        step_cfg["line2"],
        font=FONT_MAIN,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )
    # 4) Anatomical Impact Sub-Pill (Clear explanation of WHY / RESULT)
    impact_txt = step_cfg["impact"]
    imp_w = int(od.textlength(impact_txt, font=FONT_IMPACT))
    pill_y = ly + 164
    od.rounded_rectangle(
        [lx, pill_y, lx + imp_w + 32, pill_y + 46],
        radius=12,
        fill=(20, 24, 30, int(225 * alpha_scale)),
        outline=(*col, int(185 * alpha_scale)),
        width=2,
    )
    od.text(
        (lx + 16, pill_y + 7),
        impact_txt,
        font=FONT_IMPACT,
        fill=(255, 255, 255, int(250 * alpha_scale)),
    )

    canvas.alpha_composite(ov)


def draw_sets_reps_badge(canvas: Image.Image, alpha_scale: float):
    if alpha_scale <= 0.02:
        return
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    x, y = 92, 1185
    spot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(spot).rounded_rectangle([x - 20, y - 16, x + 310, y + 130], radius=32, fill=(8, 10, 14, int(175 * alpha_scale)))
    spot = spot.filter(ImageFilter.GaussianBlur(radius=20))
    canvas.alpha_composite(spot)

    od.rounded_rectangle([86, y + 8, 93, y + 114], radius=3, fill=(*COL_GREEN, int(255 * alpha_scale)))
    od.text(
        (110, y),
        "4 × 12",
        font=FONT_SETS,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(240 * alpha_scale)),
    )
    od.text(
        (112, y + 86),
        "SETS × SLOW REPS",
        font=FONT_SETS_SUB,
        fill=(*COL_GREEN, int(255 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(230 * alpha_scale)),
    )
    canvas.alpha_composite(ov)


def render_frame_at_time(t: float) -> Image.Image:
    step_idx = min(3, int(t // 4.0))
    st = STEPS[step_idx]
    local_t = t - st["t_start"]

    rep_p = get_rep_progress(local_t, period=4.0)
    pool = WRONG_POOL if st["is_wrong"] else CORRECT_POOL

    fade_in = smoothstep(0.08, 0.42, local_t)
    fade_out = 1.0 - smoothstep(3.65, 3.96, local_t)
    alpha_scale = fade_in * fade_out

    # 1. Render 3D monochrome frame with active muscle/bone highlighted in pure Red or Green
    arr = render_focused_3d_array(
        pool,
        rep_phase=rep_p,
        focus_xy=st["anchor_xy"],
        focus_sigma=st["focus_sigma"],
        col=st["col"],
        alpha_scale=0.35 + 0.65 * alpha_scale,
        use_hand_mask=st["use_hand_mask"],
    )

    # 2. Render True Sub-Pixel SDF Volumetric Bezier Ribbon sweeping from typography into the anatomical node!
    curve_prog = smoothstep(0.10, 0.65, local_t)
    render_volumetric_bezier_ribbon(
        arr,
        st["curve_p0"],
        st["curve_p1"],
        st["curve_p2"],
        st["curve_p3"],
        col=st["col"],
        alpha_scale=alpha_scale,
        u_end=curve_prog,
        max_radius=6.2,
        with_swept_head=True,
    )

    # Optional secondary motion arc on the rope/hands in Step 3
    if "extra_motion_curve" in st:
        ep0, ep1, ep2, ep3 = st["extra_motion_curve"]
        render_volumetric_bezier_ribbon(
            arr,
            ep0,
            ep1,
            ep2,
            ep3,
            col=st["col"],
            alpha_scale=alpha_scale * 0.90,
            u_end=curve_prog,
            max_radius=5.2,
            with_swept_head=True,
        )

    # 3. Render Volumetric Orbital Target Ring (◎) at the anatomical anchor & start pearl at curve_p0
    render_volumetric_target_node(arr, st["anchor_xy"], st["col"], alpha_scale, t, ring_radius=28.0)
    render_volumetric_target_node(arr, st["curve_p0"], st["col"], alpha_scale, t, ring_radius=10.0)

    base = Image.fromarray(arr).convert("RGBA")

    # 4. Top Balanced 4-Step Posture Preview Bar + Circular ✕/✓ Badge
    draw_top_balanced_preview_bar(base, step_idx, st["is_wrong"])

    # 5. Upgraded 3-Tier Editorial Typography Block
    draw_editorial_typography_block(base, st, alpha_scale)

    # 6. In Step 4, clean "4 × 12" Sets × Reps at bottom-left
    if st["show_sets"]:
        draw_sets_reps_badge(base, alpha_scale)

    return base.convert("RGB")


def generate_previews():
    os.makedirs(OUT_DIR, exist_ok=True)
    peak_times = [2.0, 6.0, 10.0, 14.0]
    out_paths = [OUT_STEP1, OUT_STEP2, OUT_STEP3, OUT_STEP4]
    titles = [
        "1. WRONG: SHOULDER ROLL (0-4s)",
        "2. WRONG: ELBOW SWING (4-8s)",
        "3. CORRECT: PIN & SPREAD (8-12s)",
        "4. CORRECT: TRICEPS + 4x12 (12-16s)",
    ]

    imgs = []
    for t, p in zip(peak_times, out_paths):
        im = render_frame_at_time(t)
        im.save(p, quality=95)
        imgs.append(im)
        print("Saved:", p)

    pw, ph = 420, 746
    pad = 24
    header_h = 68
    sheet_w = 4 * pw + 5 * pad
    sheet_h = ph + header_h + 2 * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (12, 14, 18))
    sd = ImageDraw.Draw(sheet)
    f_hdr = load_font(23, bold=True)

    for i, (im, title) in enumerate(zip(imgs, titles)):
        x = pad + i * (pw + pad)
        y = header_h + pad
        col = COL_RED if i < 2 else COL_GREEN
        sd.text((x + 8, 22), title, font=f_hdr, fill=col)
        thumb = im.resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        sd.rounded_rectangle([x - 2, y - 2, x + pw + 2, y + ph + 2], radius=14, outline=col, width=3)

    sheet.save(OUT_PREVIEW, quality=95)
    print("Saved 4-step volumetric preview sheet:", OUT_PREVIEW)


async def build_voiceover_and_audio(wav_path: str):
    import edge_tts

    lines = [
        (0.12, "Rolling your shoulders forward grinds the rotator cuff joint."),
        (4.12, "Swinging your elbows shifts tension into your front delts."),
        (8.12, "Instead, pin your elbows tight and spread the rope at the bottom."),
        (12.12, "This isolates the triceps lateral and long head. Four sets of twelve slow reps."),
    ]
    sr = 44100
    total_samples = int(DURATION * sr)
    master = np.zeros(total_samples, dtype=np.float32)

    t_arr = np.linspace(0, DURATION, total_samples, endpoint=False)
    sub = 0.045 * np.sin(2 * np.pi * 55.0 * t_arr)
    beat_env = (np.sin(2 * np.pi * (100.0 / 60.0) * t_arr) > 0.75).astype(np.float32)
    master += sub * (0.4 + 0.6 * beat_env)

    for step_i, (_, _) in enumerate(lines):
        t0 = step_i * 4.0
        s0 = int(t0 * sr)
        dur_c = int(0.22 * sr)
        if s0 + dur_c < total_samples:
            tc = np.linspace(0, 0.22, dur_c, endpoint=False)
            freq = 220.0 if step_i < 2 else 528.0
            chime = 0.10 * np.sin(2 * np.pi * freq * tc) * np.exp(-tc * 14.0)
            master[s0:s0 + dur_c] += chime

    os.makedirs(TMP_DIR, exist_ok=True)
    for i, (start_s, text) in enumerate(lines):
        mp3_p = os.path.join(TMP_DIR, f"vo_{i}.mp3")
        wav_p = os.path.join(TMP_DIR, f"vo_{i}.wav")
        comm = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+6%", pitch="-2Hz")
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

    print(f"Rendering {TOTAL_FRAMES} frames with max_workers=4...")
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
    print("Saved final MP4:", OUT_MP4)


if __name__ == "__main__":
    generate_previews()
    if "--mp4" in sys.argv:
        render_mp4()
