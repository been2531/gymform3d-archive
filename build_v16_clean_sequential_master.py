#!/usr/bin/env python3
"""
@gymform.3d — Day 1 Cable Rope Pushdown — v16 Clean Sequential Master Format
Directly follows @demicstory & De9qPwyGFvK reference structure:
1. Unzoomed Full-Shot 3D Studio Framing (matched full-body scale for Wrong=0.76 and Correct=0.84 inside seamless dark studio #121418).
2. Strict Single Accent Color Palette:
   - Neutral Matte White/Grey 3D Mannequin on Dark Charcoal (#121418)
   - Preserves the real 3D sculpted muscle/bone anatomy from the 3D render while eliminating mixed blue/orange/yellow colors!
   - Wrong Phase (0–8s): ONLY Crimson Red (#FF3B30) + Pure White (#FFFFFF)
   - Correct Phase (8–16s): ONLY Emerald Green (#18E269) + Pure White (#FFFFFF)
3. One Point at a Time (Sequential 4 Steps × 4.0s = 16.0s):
   - Step 1 (0.0–4.0s): ✕ INJURY RISK -> SHOULDER IMPINGEMENT (1 red arrow to shoulder joint)
   - Step 2 (4.0–8.0s): ✕ WRONG TARGET -> FRONT DELTOID (1 red arrow to anterior deltoid)
   - Step 3 (8.0–12.0s): ✓ POSTURE KEY -> LOCK ELBOW AT SIDE (1 green arrow to pinned elbow)
   - Step 4 (12.0–16.0s): ✓ TARGET MUSCLE -> TRICEPS BRACHII + "4 × 12" Sets×Reps (1 green arrow to 3D triceps belly)
4. Top @demicstory-style 3-Thumbnail Posture Preview Bar + Right Circular ✕/✓ Badge inside Instagram Reels Safe Zone.
5. Calm, concise English trainer voiceover synced to the 4 sequential steps.
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
TMP_DIR = "/workspace/frames_v16_day1"

WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")

OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v16_sequential_preview.jpg")
OUT_STEP1 = os.path.join(OUT_DIR, "day1_v16_step1_injury.jpg")
OUT_STEP2 = os.path.join(OUT_DIR, "day1_v16_step2_wrong_muscle.jpg")
OUT_STEP3 = os.path.join(OUT_DIR, "day1_v16_step3_posture_key.jpg")
OUT_STEP4 = os.path.join(OUT_DIR, "day1_v16_step4_target_muscle.jpg")
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


FONT_TAG = load_font(34, bold=False)
FONT_MAIN = load_font(54, bold=True)
FONT_SETS = load_font(78, bold=True)
FONT_SETS_SUB = load_font(30, bold=False)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def prepare_pulled_back_frame(bgr: np.ndarray, is_wrong: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - mono_canvas (H, W, 3): Clean monochrome matte white/grey 3D mannequin on #121418 studio background,
        with matched full-body scale (0.76 for wrong, 0.84 for correct) so neither is overly zoomed in.
      - red_mask (H, W) float32 0..1: Real 3D anatomical muscle/bone highlight mask from the render.
      - green_mask (H, W) float32 0..1: Real 3D hand/grip mask from the render.
    """
    scale = 0.76 if is_wrong else 0.84
    y_shift = 85 if is_wrong else 70

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(W * scale), int(H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]

    # Extract real 3D muscle/bone redness mask before neutralizing colors
    red_diff = np.clip((r_ch - np.maximum(g_ch, b_ch)) / 135.0, 0.0, 1.0)
    green_diff = np.clip((g_ch - np.maximum(r_ch, b_ch)) / 120.0, 0.0, 1.0)

    # Neutralize blue neon outlines and all colors into sculpted matte white/grey luminance
    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    # Suppress blue neon edge glow so it doesn't leave a ring
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
print(f"Loaded pools: wrong={len(WRONG_POOL)} frames, correct={len(CORRECT_POOL)} frames")


def render_focused_3d_frame(
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rep_phase: float,
    focus_xy: tuple[int, int],
    focus_sigma: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
    use_hand_mask: bool = False,
) -> Image.Image:
    """
    Samples the 3D monochrome frame and illuminates the REAL 3D sculpted anatomical muscle/bone
    within a spatial spotlight around `focus_xy` using ONLY the single semantic color `col`.
    """
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

    # Build smooth 2D Gaussian spatial window around the step's active anatomical focus
    yy, xx = np.ogrid[:H, :W]
    fx, fy = focus_xy
    sx, sy = focus_sigma
    spatial = np.exp(-(((xx - fx) ** 2) / (2.0 * sx * sx) + ((yy - fy) ** 2) / (2.0 * sy * sy))).astype(np.float32)

    # Combine the 3D render's actual anatomical mask with the spatial focus window
    active_3d = (r_mask * (0.30 + 0.70 * spatial))
    if use_hand_mask:
        active_3d = np.maximum(active_3d * 0.35, g_mask * 0.95 + spatial * 0.35 * r_mask)

    active_3d = np.clip(active_3d * 1.45 * alpha_scale, 0.0, 1.0)

    # Colorize the 3D muscle fibers / bone structure in pure `col` (Red or Green) while preserving 3D shading!
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


# Exact pixel coordinates verified on the scaled 1080x1920 canvases:
# Wrong canvas (scale=0.76, y_shift=85):
#   - Shoulder joint: (775, 815)
#   - Anterior Deltoid (front shoulder): (742, 875)
# Correct canvas (scale=0.84, y_shift=70):
#   - Pinned Elbow joint: (690, 912)
#   - Triceps Brachii muscle belly: (712, 792)
STEPS = [
    {
        "step_idx": 0,
        "t_start": 0.0,
        "t_end": 4.0,
        "is_wrong": True,
        "col": COL_RED,
        "mode": "joint_injury",
        "anchor_xy": (775, 815),
        "focus_sigma": (115.0, 115.0),
        "use_hand_mask": False,
        "label_y": 725,
        "tag": "✕ JOINT INJURY RISK",
        "line1": "SHOULDER",
        "line2": "IMPINGEMENT",
        "show_sets": False,
    },
    {
        "step_idx": 1,
        "t_start": 4.0,
        "t_end": 8.0,
        "is_wrong": True,
        "col": COL_RED,
        "mode": "wrong_muscle",
        "anchor_xy": (742, 875),
        "focus_sigma": (110.0, 110.0),
        "use_hand_mask": False,
        "label_y": 790,
        "tag": "✕ WRONG TARGET",
        "line1": "FRONT",
        "line2": "DELTOID",
        "show_sets": False,
    },
    {
        "step_idx": 2,
        "t_start": 8.0,
        "t_end": 12.0,
        "is_wrong": False,
        "col": COL_GREEN,
        "mode": "posture_lock",
        "anchor_xy": (690, 912),
        "focus_sigma": (120.0, 120.0),
        "use_hand_mask": True,
        "label_y": 825,
        "tag": "✓ POSTURE KEY",
        "line1": "LOCK ELBOW",
        "line2": "AT SIDE",
        "show_sets": False,
    },
    {
        "step_idx": 3,
        "t_start": 12.0,
        "t_end": 16.0,
        "is_wrong": False,
        "col": COL_GREEN,
        "mode": "target_muscle",
        "anchor_xy": (712, 792),
        "focus_sigma": (140.0, 155.0),
        "use_hand_mask": False,
        "label_y": 705,
        "tag": "✓ TARGET MUSCLE",
        "line1": "TRICEPS",
        "line2": "BRACHII",
        "show_sets": True,
    },
]


def make_mini_thumb(step_cfg: dict, size: tuple[int, int] = (108, 124)) -> Image.Image:
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
    # Add subtle dot at anchor_xy so thumbnail clearly previews the posture & active spot
    ax, ay = step_cfg["anchor_xy"]
    gd = ImageDraw.Draw(im)
    gd.ellipse([ax - 24, ay - 24, ax + 24, ay + 24], fill=(*step_cfg["col"], 220))
    # Crop full upper body + cable rope (x: 240..900, y: 460..1220)
    cropped = im.crop((240, 460, 900, 1220)).resize(size, Image.Resampling.LANCZOS)
    return cropped


THUMB_1 = make_mini_thumb(STEPS[0])
THUMB_2 = make_mini_thumb(STEPS[2])
THUMB_3 = make_mini_thumb(STEPS[3])


def draw_top_demic_bar_and_badge(canvas: Image.Image, step_idx: int, is_wrong: bool):
    """
    Draws the @demicstory / De9qPwyGFvK top UI inside the Instagram Reels Safe Zone (y = 246..394):
    - Left/Center: 3-Thumbnail Posture Preview Capsule (active thumbnail framed in Red or Green)
    - Right: Clean circular ✕ (Red) or ✓ (Green) status badge
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    accent = COL_RED if is_wrong else COL_GREEN
    active_thumb = 0 if step_idx in (0, 1) else (1 if step_idx == 2 else 2)

    tw, th = 108, 124
    gap = 14
    pad = 12
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + 2 * pad
    bar_x = 88
    bar_y = 246

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=22,
        fill=(26, 30, 36, 215),
        outline=(255, 255, 255, 40),
        width=2,
    )

    thumbs = [THUMB_1, THUMB_2, THUMB_3]
    for i, thumb in enumerate(thumbs):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        t_img = thumb.copy()
        if i != active_thumb:
            dim = Image.new("RGBA", t_img.size, (12, 14, 18, 115))
            t_img.alpha_composite(dim)
        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=255)
        ov.paste(t_img, (tx, ty), mask)

        if i == active_thumb:
            t_col = COL_RED if i == 0 else COL_GREEN
            od.rounded_rectangle(
                [tx - 3, ty - 3, tx + tw + 3, ty + th + 3],
                radius=16,
                outline=(*t_col, 255),
                width=5,
            )

    bx, by = 815, bar_y + bar_h // 2
    br = 54
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([bx - br - 10, by - br - 10, bx + br + 10, by + br + 10], fill=(*accent, 90))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas.alpha_composite(glow)

    od.ellipse([bx - br, by - br, bx + br, by + br], fill=(*accent, 245), outline=(255, 255, 255, 220), width=4)
    if is_wrong:
        d = 21
        od.line([(bx - d, by - d), (bx + d, by + d)], fill=COL_WHITE, width=10)
        od.line([(bx - d, by + d), (bx + d, by - d)], fill=COL_WHITE, width=10)
    else:
        pts = [(bx - 24, by + 2), (bx - 5, by + 21), (bx + 26, by - 18)]
        od.line(pts, fill=COL_WHITE, width=10, joint="curve")

    canvas.alpha_composite(ov)


def draw_subtle_joint_ring(
    canvas: Image.Image,
    mode: str,
    center: tuple[int, int],
    col: tuple[int, int, int],
    alpha_scale: float,
    t: float,
):
    """Draws a clean, minimal ring on the joint for injury/posture steps (no clutter)."""
    if alpha_scale <= 0.02:
        return
    cx, cy = center
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    pulse = 1.0 + 0.06 * math.sin(t * 5.5)

    if mode == "joint_injury":
        for k in range(2):
            p = (t * 1.3 + k * 0.5) % 1.0
            r = int(20 + p * 54)
            a = int((1.0 - p) * 235 * alpha_scale)
            od.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*col, a), width=4)
    elif mode == "posture_lock":
        r = int(36 * pulse)
        od.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*col, int(250 * alpha_scale)), width=5)

    canvas.alpha_composite(ov)


def draw_single_clean_callout(
    canvas: Image.Image,
    anchor_xy: tuple[int, int],
    label_y: int,
    tag_text: str,
    main_line1: str,
    main_line2: str,
    col: tuple[int, int, int],
    alpha_scale: float,
):
    """
    Draws ONE clean, high-contrast 2-line anatomical label on the left studio space (x = 88)
    and ONE bold 3D pointer arrow (◀━━━━━━●) from the exact anatomical anchor directly to the label!
    """
    if alpha_scale <= 0.02:
        return

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    spot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spot)
    sd.rounded_rectangle(
        [64, label_y - 30, 520, label_y + 180],
        radius=44,
        fill=(8, 10, 14, int(185 * alpha_scale)),
    )
    spot = spot.filter(ImageFilter.GaussianBlur(radius=24))
    canvas.alpha_composite(spot)

    lx = 88
    od.text(
        (lx, label_y),
        tag_text,
        font=FONT_TAG,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=5,
        stroke_fill=(0, 0, 0, int(235 * alpha_scale)),
    )
    od.text(
        (lx, label_y + 46),
        main_line1,
        font=FONT_MAIN,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )
    od.text(
        (lx, label_y + 108),
        main_line2,
        font=FONT_MAIN,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )

    w1 = od.textlength(main_line1, font=FONT_MAIN)
    w2 = od.textlength(main_line2, font=FONT_MAIN)
    text_right = lx + max(w1, w2)
    arrow_tip_x = int(min(515, text_right + 24))
    arrow_tip_y = label_y + 98

    ax, ay = anchor_xy
    elbow_x = arrow_tip_x + 56
    elbow_y = arrow_tip_y

    pts = [(ax, ay), (elbow_x, elbow_y), (arrow_tip_x + 18, arrow_tip_y)]
    od.line(pts, fill=(0, 0, 0, int(225 * alpha_scale)), width=14, joint="curve")
    od.line(pts, fill=(*col, int(255 * alpha_scale)), width=7, joint="curve")

    # Arrowhead pointing LEFT at the muscle name (◀)
    ah = [
        (arrow_tip_x, arrow_tip_y),
        (arrow_tip_x + 28, arrow_tip_y - 16),
        (arrow_tip_x + 22, arrow_tip_y),
        (arrow_tip_x + 28, arrow_tip_y + 16),
    ]
    od.polygon(ah, fill=(*col, int(255 * alpha_scale)), outline=(0, 0, 0, int(225 * alpha_scale)))

    # Glowing anchor node (●) right on the muscle/joint
    od.ellipse([ax - 13, ay - 13, ax + 13, ay + 13], fill=(*col, int(255 * alpha_scale)), outline=(0, 0, 0, int(225 * alpha_scale)), width=3)
    od.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=(255, 255, 255, int(255 * alpha_scale)))

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

    od.text(
        (x, y),
        "4 × 12",
        font=FONT_SETS,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(240 * alpha_scale)),
    )
    od.text(
        (x + 4, y + 86),
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

    # 1. Render monochrome 3D mannequin with ONLY the active anatomical muscle/bone illuminated in Red or Green
    base = render_focused_3d_frame(
        pool,
        rep_phase=rep_p,
        focus_xy=st["anchor_xy"],
        focus_sigma=st["focus_sigma"],
        col=st["col"],
        alpha_scale=0.35 + 0.65 * alpha_scale,
        use_hand_mask=st["use_hand_mask"],
    ).convert("RGBA")

    # 2. Subtle joint ring for joint steps
    draw_subtle_joint_ring(base, st["mode"], st["anchor_xy"], st["col"], alpha_scale, t)

    # 3. Top @demicstory 3-Thumbnail Posture Preview Bar + Circular ✕/✓ Badge
    draw_top_demic_bar_and_badge(base, step_idx, st["is_wrong"])

    # 4. Single clean arrow + single 2-line muscle/posture callout
    draw_single_clean_callout(
        base,
        anchor_xy=st["anchor_xy"],
        label_y=st["label_y"],
        tag_text=st["tag"],
        main_line1=st["line1"],
        main_line2=st["line2"],
        col=st["col"],
        alpha_scale=alpha_scale,
    )

    # 5. In Step 4, clean "4 × 12" Sets × Reps at bottom-left
    if st["show_sets"]:
        draw_sets_reps_badge(base, alpha_scale)

    return base.convert("RGB")


def generate_previews():
    os.makedirs(OUT_DIR, exist_ok=True)
    peak_times = [2.0, 6.0, 10.0, 14.0]
    out_paths = [OUT_STEP1, OUT_STEP2, OUT_STEP3, OUT_STEP4]
    titles = [
        "1. WRONG: INJURY RISK (0-4s)",
        "2. WRONG: FRONT DELT (4-8s)",
        "3. CORRECT: POSTURE KEY (8-12s)",
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
    f_hdr = load_font(24, bold=True)

    for i, (im, title) in enumerate(zip(imgs, titles)):
        x = pad + i * (pw + pad)
        y = header_h + pad
        col = COL_RED if i < 2 else COL_GREEN
        sd.text((x + 8, 22), title, font=f_hdr, fill=col)
        thumb = im.resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        sd.rounded_rectangle([x - 2, y - 2, x + pw + 2, y + ph + 2], radius=14, outline=col, width=3)

    sheet.save(OUT_PREVIEW, quality=95)
    print("Saved 4-step sequential preview sheet:", OUT_PREVIEW)


async def build_voiceover_and_audio(wav_path: str):
    import edge_tts

    lines = [
        (0.15, "Swinging your elbows overloads the shoulder joint."),
        (4.15, "And shifts the tension into your front deltoid."),
        (8.15, "Instead, lock your elbows tight at your sides."),
        (12.15, "To isolate the triceps brachii. Four sets of twelve reps."),
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
