#!/usr/bin/env python3
"""
@gymform.3d — Day 1 Cable Rope Pushdown — v17 Luxury Bezier Motion Master Format
Upgrades over v16 based on user feedback (t78u–t81u):
1. ZERO clunky 2D polygonal arrows:
   - Replaced with 3x Supersampled (SSAA) fluid cubic Bezier motion ribbons (variable-width tapered glow + white-hot specular core + swept aerodynamic tip)
   - Plus an ultra-sleek, hair-thin silk S-curve leader filament with a concentric anatomical target ring (◎) and vertical glass accent bar.
2. Balanced Full-Width Top 4-Step Posture Preview Bar (x = 86..890, y = 244..386):
   - Harmoniously spans the Safe Zone width with 4 mini posture thumbnails (matching the 4 sequential steps 1:1) + sleek circular ✕/✓ emblem.
3. Improved 4-Step Cause-and-Effect Biomechanical Copy & Voiceover:
   - Step 1 (0–4s): ✕ SHOULDERS ROLL FORWARD -> ROTATOR CUFF STRAIN
   - Step 2 (4–8s): ✕ ELBOWS SWING UP & DOWN -> FRONT DELT STEALS LOAD
   - Step 3 (8–12s): ✓ PIN ELBOWS TO RIBS -> SPREAD ROPE AT BOTTOM
   - Step 4 (12–16s): ✓ 100% TRICEPS ISOLATION -> LATERAL & LONG HEAD (+ 4 × 12 SETS × REPS)
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
TMP_DIR = "/workspace/frames_v17_day1"

WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")

OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v17_luxury_preview.jpg")
OUT_STEP1 = os.path.join(OUT_DIR, "day1_v17_step1_injury.jpg")
OUT_STEP2 = os.path.join(OUT_DIR, "day1_v17_step2_wrong_muscle.jpg")
OUT_STEP3 = os.path.join(OUT_DIR, "day1_v17_step3_posture_key.jpg")
OUT_STEP4 = os.path.join(OUT_DIR, "day1_v17_step4_target_muscle.jpg")
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


FONT_CAUSE = load_font(31, bold=False)
FONT_EFFECT = load_font(50, bold=True)
FONT_SETS = load_font(76, bold=True)
FONT_SETS_SUB = load_font(28, bold=False)
FONT_STEP_NUM = load_font(20, bold=True)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def cubic_bezier(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], u: float) -> tuple[float, float]:
    om = 1.0 - u
    x = (om ** 3) * p0[0] + 3 * (om ** 2) * u * p1[0] + 3 * om * (u ** 2) * p2[0] + (u ** 3) * p3[0]
    y = (om ** 3) * p0[1] + 3 * (om ** 2) * u * p1[1] + 3 * om * (u ** 2) * p2[1] + (u ** 3) * p3[1]
    return x, y


def cubic_bezier_tangent(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], u: float) -> tuple[float, float]:
    om = 1.0 - u
    dx = 3 * (om ** 2) * (p1[0] - p0[0]) + 6 * om * u * (p2[0] - p1[0]) + 3 * (u ** 2) * (p3[0] - p2[0])
    dy = 3 * (om ** 2) * (p1[1] - p0[1]) + 6 * om * u * (p2[1] - p1[1]) + 3 * (u ** 2) * (p3[1] - p2[1])
    norm = math.hypot(dx, dy) + 1e-6
    return dx / norm, dy / norm


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


def render_focused_3d_frame(
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rep_phase: float,
    focus_xy: tuple[int, int],
    focus_sigma: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    use_hand_mask: bool = False,
) -> Image.Image:
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

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def get_rep_progress(local_t: float, period: float = 4.0) -> float:
    u = (local_t % period) / period
    if u < 0.45:
        return 0.5 - 0.5 * math.cos(math.pi * (u / 0.45))
    elif u < 0.72:
        return 1.0
    else:
        return 0.5 + 0.5 * math.cos(math.pi * ((u - 0.72) / 0.28))


STEPS = [
    {
        "step_idx": 0,
        "t_start": 0.0,
        "t_end": 4.0,
        "is_wrong": True,
        "col": COL_RED,
        "mode": "shoulder_roll_arc",
        "anchor_xy": (775, 815),
        "focus_sigma": (110.0, 110.0),
        "use_hand_mask": False,
        "label_y": 695,
        "cause": "✕ SHOULDERS ROLL FORWARD",
        "line1": "ROTATOR CUFF",
        "line2": "IMPINGEMENT",
        # Curved forward-rolling Bezier motion arc around the shoulder joint
        "motion_curve": ((835, 760), (775, 710), (700, 745), (675, 825)),
        "show_sets": False,
    },
    {
        "step_idx": 1,
        "t_start": 4.0,
        "t_end": 8.0,
        "is_wrong": True,
        "col": COL_RED,
        "mode": "elbow_swing_arc",
        "anchor_xy": (742, 875),
        "focus_sigma": (115.0, 115.0),
        "use_hand_mask": False,
        "label_y": 775,
        "cause": "✕ ELBOWS SWING UP & DOWN",
        "line1": "FRONT DELT",
        "line2": "STEALS LOAD",
        # Curved upward/forward swinging Bezier trajectory from elbow toward front delt
        "motion_curve": ((745, 1045), (675, 995), (655, 910), (705, 845)),
        "show_sets": False,
    },
    {
        "step_idx": 2,
        "t_start": 8.0,
        "t_end": 12.0,
        "is_wrong": False,
        "col": COL_GREEN,
        "mode": "rope_spread_arc",
        "anchor_xy": (690, 912),
        "focus_sigma": (120.0, 120.0),
        "use_hand_mask": True,
        "label_y": 805,
        "cause": "✓ PIN ELBOWS TIGHT TO RIBS",
        "line1": "SPREAD ROPE",
        "line2": "AT BOTTOM",
        # Fluid downward & outward J-curve Bezier motion ribbon showing rope flare at bottom
        "motion_curve": ((540, 745), (495, 825), (445, 895), (365, 935)),
        "show_sets": False,
    },
    {
        "step_idx": 3,
        "t_start": 12.0,
        "t_end": 16.0,
        "is_wrong": False,
        "col": COL_GREEN,
        "mode": "triceps_fiber_flow",
        "anchor_xy": (712, 792),
        "focus_sigma": (140.0, 155.0),
        "use_hand_mask": False,
        "label_y": 685,
        "cause": "✓ 100% TRICEPS ISOLATION",
        "line1": "LATERAL &",
        "line2": "LONG HEAD",
        # Sculptural energy contraction curve sweeping up the Triceps Lateral & Long Head belly
        "motion_curve": ((685, 905), (748, 855), (762, 775), (718, 705)),
        "show_sets": True,
    },
]


def draw_ssaa_tapered_bezier_ribbon(
    canvas: Image.Image,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    draw_progress: float = 1.0,
    max_width: float = 11.0,
    with_arrow_tip: bool = True,
):
    """
    Renders a luxury 3D tapered Bezier motion ribbon using high-density polygon strips +
    soft Gaussian bloom + razor-sharp white-hot specular filament core + sculpted swept wing tip.
    """
    if alpha_scale <= 0.02 or draw_progress <= 0.05:
        return

    steps = 90
    u_max = min(1.0, max(0.15, draw_progress))
    us = np.linspace(0.0, u_max, steps)

    left_pts = []
    right_pts = []
    center_pts = []

    for u in us:
        x, y = cubic_bezier(p0, p1, p2, p3, float(u))
        tx, ty = cubic_bezier_tangent(p0, p1, p2, p3, float(u))
        nx, ny = -ty, tx
        # Smoothly taper width from 1.0px at tail (u=0) to max_width near head (u=0.82*u_max)
        rel_u = u / u_max
        w = max_width * (math.sin(rel_u * math.pi * 0.86) ** 0.75)
        left_pts.append((x + nx * w * 0.5, y + ny * w * 0.5))
        right_pts.append((x - nx * w * 0.5, y - ny * w * 0.5))
        center_pts.append((x, y))

    poly = left_pts + right_pts[::-1]

    # 1. Soft Outer Neon Bloom
    bloom = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bloom)
    bd.polygon(poly, fill=(*col, int(185 * alpha_scale)))
    bloom = bloom.filter(ImageFilter.GaussianBlur(radius=14))
    canvas.alpha_composite(bloom)

    # 2. Crisp Sculpted Ribbon Body + Specular Core
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.polygon(poly, fill=(*col, int(245 * alpha_scale)))

    # Specular white-tinted inner spine (from 20% to 92% of the curve)
    core_pts = center_pts[int(steps * 0.18) : int(steps * 0.94)]
    if len(core_pts) >= 2:
        od.line(core_pts, fill=(255, 255, 255, int(220 * alpha_scale)), width=3, joint="curve")

    # 3. Sleek Aerodynamic Swept Chevron Tip (modern motion-graphics style, not a clunky triangle)
    if with_arrow_tip:
        hx, hy = cubic_bezier(p0, p1, p2, p3, u_max)
        tx, ty = cubic_bezier_tangent(p0, p1, p2, p3, u_max)
        nx, ny = -ty, tx

        tip_len = 28.0
        wing_span = 16.0
        notch_back = 18.0

        nose = (hx + tx * 8.0, hy + ty * 8.0)
        left_wing = (hx - tx * tip_len + nx * wing_span, hy - ty * tip_len + ny * wing_span)
        notch = (hx - tx * notch_back, hy - ty * notch_back)
        right_wing = (hx - tx * tip_len - nx * wing_span, hy - ty * tip_len - ny * wing_span)

        od.polygon([nose, left_wing, notch, right_wing], fill=(255, 255, 255, int(250 * alpha_scale)))
        # Inner colored chevron accent
        nose_in = (hx + tx * 5.0, hy + ty * 5.0)
        od.polygon(
            [
                nose_in,
                (hx - tx * (tip_len - 4) + nx * (wing_span - 3), hy - ty * (tip_len - 4) + ny * (wing_span - 3)),
                (hx - tx * (notch_back - 3), hy - ty * (notch_back - 3)),
                (hx - tx * (tip_len - 4) - nx * (wing_span - 3), hy - ty * (tip_len - 4) - ny * (wing_span - 3)),
            ],
            fill=(*col, int(255 * alpha_scale)),
        )

    canvas.alpha_composite(ov)


def draw_luxury_callout_and_silk_curve(
    canvas: Image.Image,
    step_cfg: dict,
    alpha_scale: float,
    local_t: float,
    t: float,
):
    """
    1. Draws the refined 3-line Editorial Typography Block on the left (x = 92) with a glowing vertical bar (┃).
    2. Connects the vertical bar to the anatomical target ring (◎) using a hair-thin, fluid S-curve Bezier filament (NO clunky arrowhead!).
    3. Renders the fluid 3D Bezier Motion Trajectory Ribbon right on the body.
    """
    if alpha_scale <= 0.02:
        return

    col = step_cfg["col"]
    ax, ay = step_cfg["anchor_xy"]
    ly = step_cfg["label_y"]
    cause = step_cfg["cause"]
    line1 = step_cfg["line1"]
    line2 = step_cfg["line2"]

    # 1. Feathered studio vignette behind the typography for 100% contrast without boxes
    spot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spot)
    sd.rounded_rectangle([62, ly - 32, 545, ly + 185], radius=48, fill=(8, 10, 14, int(185 * alpha_scale)))
    spot = spot.filter(ImageFilter.GaussianBlur(radius=28))
    canvas.alpha_composite(spot)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # Sleek vertical glowing accent bar on the left of the typography (x = 86..92)
    bar_x = 86
    bar_y0, bar_y1 = ly + 4, ly + 158
    od.rounded_rectangle([bar_x, bar_y0, bar_x + 6, bar_y1], radius=3, fill=(*col, int(255 * alpha_scale)))

    lx = 108
    # Cause Sub-Headline (e.g. "✕ SHOULDERS ROLL FORWARD")
    od.text(
        (lx, ly),
        cause,
        font=FONT_CAUSE,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(230 * alpha_scale)),
    )
    # Effect Main Line 1 (Pure White)
    od.text(
        (lx, ly + 44),
        line1,
        font=FONT_EFFECT,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )
    # Effect Main Line 2 (Accent Color)
    od.text(
        (lx, ly + 102),
        line2,
        font=FONT_EFFECT,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )

    # 2. Sleek S-Curve Silk Leader Filament from Typography to Anatomical Ring (◎) — NO clunky arrowhead!
    w1 = od.textlength(line1, font=FONT_EFFECT)
    w2 = od.textlength(line2, font=FONT_EFFECT)
    text_right = lx + max(w1, w2)
    start_x = min(525.0, float(text_right + 22))
    start_y = float(ly + 102)

    end_x, end_y = float(ax), float(ay)
    mid_x = (start_x + end_x) * 0.5
    c1 = (mid_x, start_y)
    c2 = (mid_x, end_y)

    # Sample smooth S-curve points
    curve_pts = [cubic_bezier((start_x, start_y), c1, c2, (end_x, end_y), u) for u in np.linspace(0.0, 1.0, 60)]
    # Soft outer glow on the silk filament
    fil_glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(fil_glow).line(curve_pts, fill=(*col, int(140 * alpha_scale)), width=8, joint="curve")
    fil_glow = fil_glow.filter(ImageFilter.GaussianBlur(radius=6))
    canvas.alpha_composite(fil_glow)

    # Crisp 2.5px silk curve + luminous pearl terminal at text & concentric target ring at muscle
    od.line(curve_pts, fill=(255, 255, 255, int(225 * alpha_scale)), width=3, joint="curve")
    od.ellipse([start_x - 5, start_y - 5, start_x + 5, start_y + 5], fill=(*col, int(255 * alpha_scale)))

    # Refined Concentric Anatomical Target Ring (◎) at (ax, ay)
    pulse_r = 22 + int(4 * math.sin(t * 6.0))
    od.ellipse([ax - pulse_r, ay - pulse_r, ax + pulse_r, ay + pulse_r], outline=(*col, int(230 * alpha_scale)), width=3)
    od.ellipse([ax - 10, ay - 10, ax + 10, ay + 10], fill=(*col, int(255 * alpha_scale)), outline=(255, 255, 255, int(240 * alpha_scale)), width=2)
    od.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=(255, 255, 255, int(255 * alpha_scale)))

    canvas.alpha_composite(ov)

    # 3. Fluid 3D Bezier Motion Trajectory Ribbon on the body (animated sweep over each 2s cycle)
    p0, p1, p2, p3 = step_cfg["motion_curve"]
    sweep_prog = smoothstep(0.15, 0.85, (local_t % 2.0) / 1.1)
    draw_ssaa_tapered_bezier_ribbon(
        canvas,
        p0,
        p1,
        p2,
        p3,
        col=col,
        alpha_scale=alpha_scale,
        draw_progress=0.35 + 0.65 * sweep_prog,
        max_width=12.0,
        with_arrow_tip=True,
    )


def make_mini_thumb(step_cfg: dict, size: tuple[int, int] = (142, 118)) -> Image.Image:
    pool = WRONG_POOL if step_cfg["is_wrong"] else CORRECT_POOL
    im = render_focused_3d_frame(
        pool,
        rep_phase=0.85,
        focus_xy=step_cfg["anchor_xy"],
        focus_sigma=step_cfg["focus_sigma"],
        col=step_cfg["col"],
        alpha_scale=1.0,
        use_hand_mask=step_cfg["use_hand_mask"],
    ).convert("RGBA")
    ax, ay = step_cfg["anchor_xy"]
    gd = ImageDraw.Draw(im)
    gd.ellipse([ax - 22, ay - 22, ax + 22, ay + 22], fill=(*step_cfg["col"], 220))
    cropped = im.crop((240, 480, 900, 1180)).resize(size, Image.Resampling.LANCZOS)
    return cropped


THUMBS = [make_mini_thumb(s) for s in STEPS]


def draw_top_balanced_preview_bar(canvas: Image.Image, step_idx: int, is_wrong: bool):
    """
    Balanced Top 4-Step Posture Preview Capsule spanning across the Safe Zone (x = 86..888, y = 244..386):
    - Contains 4 mini posture preview thumbnails (matching the 4 sequential steps 1:1)
    - Plus a sleek circular ✕ / ✓ status emblem on the right end of the bar.
    """
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

    # Smoked glass capsule
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

        # Tiny step badge in top-left of each thumbnail ("✕ 1", "✕ 2", "✓ 3", "✓ 4")
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

    # Right Circular ✕ / ✓ Status Badge at x = 808 (balanced right next to the 4-thumbnail bar, inside Safe Zone <= 885)
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

    od.rounded_rectangle([86, y + 8, 92, y + 112], radius=3, fill=(*COL_GREEN, int(255 * alpha_scale)))
    od.text(
        (108, y),
        "4 × 12",
        font=FONT_SETS,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(240 * alpha_scale)),
    )
    od.text(
        (110, y + 84),
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

    fade_in = smoothstep(0.10, 0.45, local_t)
    fade_out = 1.0 - smoothstep(3.65, 3.96, local_t)
    alpha_scale = fade_in * fade_out

    # 1. Monochrome 3D mannequin with ONLY the active anatomical muscle/bone illuminated in Red or Green
    base = render_focused_3d_frame(
        pool,
        rep_phase=rep_p,
        focus_xy=st["anchor_xy"],
        focus_sigma=st["focus_sigma"],
        col=st["col"],
        alpha_scale=0.35 + 0.65 * alpha_scale,
        use_hand_mask=st["use_hand_mask"],
    ).convert("RGBA")

    # 2. Top Balanced 4-Step Posture Preview Bar + Right Circular ✕/✓ Badge
    draw_top_balanced_preview_bar(base, step_idx, st["is_wrong"])

    # 3. Luxury Editorial Callout + Silk S-Curve Filament + Fluid 3D Bezier Motion Trajectory Ribbon
    draw_luxury_callout_and_silk_curve(base, st, alpha_scale, local_t, t)

    # 4. In Step 4, clean "4 × 12" Sets × Reps at bottom-left
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
    print("Saved 4-step luxury preview sheet:", OUT_PREVIEW)


async def build_voiceover_and_audio(wav_path: str):
    import edge_tts

    lines = [
        (0.12, "Rolling your shoulders forward grinds the joint and strains your rotator cuff."),
        (4.12, "Swinging your elbows shifts tension into your front delts."),
        (8.12, "Instead, pin your elbows to your ribs and spread the rope at the bottom."),
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
