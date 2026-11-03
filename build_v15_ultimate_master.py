#!/usr/bin/env python3
"""
@gymform.3d — Day 1 Cable Rope Pushdown — v15 Ultimate Master Format
15.0s Reel: Phase 1 (0–7.5s ✕ WRONG) + Phase 2 (7.5–15.0s ✓ CORRECT)
Dual-posture header, left-side 3D pointer callouts, bottom activation bar, English TTS.
"""
from __future__ import annotations

import asyncio
import math
import os
import shutil
import subprocess
import wave
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from build_3scene_ox_preview import (
    W,
    H,
    crop_camera_view,
    draw_bold_3d_arrow,
    draw_lock_on_joint_ring,
    enhance_plate,
    lerp,
    load_i2v_frames,
)

OUT_DIR = "/workspace/output"
ARTIFACTS_DIR = "/opt/cursor/artifacts"
FPS = 30
DURATION = 15.0
TOTAL_FRAMES = int(FPS * DURATION)
TMP_DIR = "/workspace/frames_v15_day1"

SAFE_X0, SAFE_X1 = 82, 890
SAFE_Y0, SAFE_Y1 = 236, 1355

PRETENDARD_BLACK = "/home/ubuntu/.fonts/Pretendard-Black.otf"
PRETENDARD_EXTRABOLD = "/home/ubuntu/.fonts/Pretendard-ExtraBold.otf"
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v15_ultimate_preview.jpg")
OUT_WRONG_PEAK = os.path.join(OUT_DIR, "day1_v15_wrong_peak.jpg")
OUT_CORRECT_PEAK = os.path.join(OUT_DIR, "day1_v15_correct_peak.jpg")
OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")

VO_WRONG = (
    "Stop swinging your elbows on cable pushdowns. "
    "Rolling your shoulders forward shifts sixty-eight percent of the load to your front delts "
    "and risks shoulder impingement."
)
VO_CORRECT = (
    "Instead, pin your elbows tight to your ribs and spread the rope apart at the bottom. "
    "This locks the joint and isolates ninety-eight percent of your triceps lateral and long head. "
    "Perform four sets of twelve slow reps."
)

WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    for p in (path, PRETENDARD_BLACK, PRETENDARD_EXTRABOLD, FALLBACK_BOLD):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TITLE = load_font(PRETENDARD_BLACK, 50)
FONT_RX_PILL = load_font(PRETENDARD_EXTRABOLD, 34)
FONT_CALLOUT_SM = load_font(PRETENDARD_EXTRABOLD, 46)
FONT_CALLOUT_LG = load_font(PRETENDARD_BLACK, 48)
FONT_CALLOUT_ML = load_font(PRETENDARD_BLACK, 52)
FONT_CALLOUT_M = load_font(PRETENDARD_BLACK, 50)
FONT_MINI_LBL = load_font(PRETENDARD_EXTRABOLD, 22)
FONT_BOTTOM_HERO = load_font(PRETENDARD_BLACK, 44)
FONT_BOTTOM_PILL = load_font(PRETENDARD_EXTRABOLD, 28)
FONT_BOTTOM_BAR = load_font(PRETENDARD_EXTRABOLD, 26)
FONT_IG = load_font(PRETENDARD_EXTRABOLD, 24)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    if hasattr(draw, "textlength"):
        return draw.textlength(text, font=font)
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def draw_text_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    shadow: bool = True,
):
    x, y = xy
    if shadow:
        draw.text((x + 2, y + 3), text, font=font, fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=font, fill=fill)


def build_scene_pool(is_wrong: bool) -> list:
    tag = "wrong" if is_wrong else "correct"
    pool: list = []
    for i in (1, 2, 3):
        p = os.path.join(OUT_DIR, f"day1_scene{i}_{tag}.jpg")
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                pool.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4))
    if pool:
        return pool
    mp4 = WRONG_I2V if is_wrong else CORRECT_I2V
    jpg = WRONG_JPG if is_wrong else CORRECT_JPG
    return load_i2v_frames(mp4, jpg)


WRONG_POOL = build_scene_pool(True)
CORRECT_POOL = build_scene_pool(False)


def rep_value(local_t: float, rep_dur: float = 3.75, hold: float = 0.85) -> tuple[float, bool]:
    """Returns (rep 0..1, at_peak_hold)."""
    cycle = local_t % rep_dur
    motion_t = rep_dur - hold
    if cycle < motion_t:
        u = cycle / motion_t
        return 0.5 - 0.5 * math.cos(math.pi * u), False
    return 1.0, True


def sample_motion_frame(pool: list, rep: float, is_wrong: bool) -> np.ndarray:
    n = len(pool)
    if n == 1:
        raw = pool[0]
    else:
        f_idx = rep * (n - 1)
        i0 = int(math.floor(f_idx))
        i1 = min(n - 1, i0 + 1)
        frac = f_idx - i0
        if frac < 1e-3:
            raw = pool[i0].copy()
        else:
            raw = cv2.addWeighted(pool[i0], 1.0 - frac, pool[i1], frac, 0)
    return enhance_plate(raw, is_wrong=is_wrong, boost_level=0.42 + 0.58 * rep)


def load_plate_thumb(path: str, size: tuple[int, int]) -> Image.Image:
    if not os.path.exists(path):
        return Image.new("RGB", size, (20, 24, 32))
    im = Image.open(path).convert("RGB")
    return im.resize(size, Image.Resampling.LANCZOS)


WRONG_THUMB = load_plate_thumb(WRONG_JPG, (130, 130))
CORRECT_THUMB = load_plate_thumb(CORRECT_JPG, (130, 130))


def draw_muscle_contour(
    canvas: Image.Image,
    center: tuple,
    col_rgb: tuple,
    prog: float,
    pulse_t: float,
):
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    cx, cy = int(center[0]), int(center[1])
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    rw, rh = 58, 44
    pulse = 1.0 + 0.06 * math.sin(pulse_t * 6.0)
    rw = int(rw * pulse)
    rh = int(rh * pulse)
    gd.ellipse([cx - rw - 6, cy - rh - 6, cx + rw + 6, cy + rh + 6], fill=(*col_rgb, int(120 * a)))
    od.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], outline=(*col_rgb, int(240 * a)), width=4)
    for i in range(-5, 6):
        ang = math.radians(-35 + i * 14)
        x0 = cx + int(12 * math.cos(ang))
        y0 = cy + int(8 * math.sin(ang))
        x1 = cx + int((rw - 4) * math.cos(ang))
        y1 = cy + int((rh - 4) * math.sin(ang))
        od.line([(x0, y0), (x1, y1)], fill=(*col_rgb, int(200 * a)), width=2)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=12))
    canvas.alpha_composite(glow)
    canvas.alpha_composite(ov)


def draw_shockwave_rings(
    canvas: Image.Image,
    center: tuple,
    col_rgb: tuple,
    t: float,
    prog: float,
):
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    cx, cy = int(center[0]), int(center[1])
    ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for k in range(3):
        phase = (t * 1.8 + k * 0.35) % 1.0
        r = int(18 + phase * 72)
        alpha = int((1.0 - phase) * 220 * a)
        od.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*col_rgb, alpha), width=4)
    canvas.alpha_composite(ov)


def draw_pointer_arrow_to_label(
    canvas: Image.Image,
    muscle_pt: tuple,
    label_right_x: float,
    label_cy: float,
    col_rgb: tuple,
    prog: float,
):
    """Bold ◀━━━━━━● pointer from muscle to left-side label block."""
    if prog <= 0.05:
        return
    p = min(1.0, prog)
    mx, my = muscle_pt
    lx = label_right_x
    ly = label_cy
    mid_x = lerp(mx, lx, 0.55)
    pts = [(mx, my), (mid_x, ly), (lx, ly)]
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    draw_len = max(2, int(len(pts) * p))
    sub = pts[:draw_len]
    if len(sub) >= 2:
        gd.line(sub, fill=(*col_rgb, 180), width=16)
        od.line(sub, fill=(8, 10, 14, 240), width=14)
        od.line(sub, fill=(*col_rgb, 255), width=8)
        od.line(sub, fill=(255, 255, 255, 220), width=2)
    bx, by = int(mx), int(my)
    od.ellipse([bx - 10, by - 10, bx + 10, by + 10], fill=(255, 255, 255, 255), outline=(*col_rgb, 255), width=4)
    # Arrow head at label end (◀)
    if p > 0.85:
        ax = int(lx)
        ay = int(ly)
        od.polygon([(ax, ay), (ax + 22, ay - 12), (ax + 22, ay + 12)], fill=(*col_rgb, 255))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(glow)
    canvas.alpha_composite(ov)


def draw_left_callout(
    canvas: Image.Image,
    y_center: int,
    muscle_pt: tuple,
    line1: str,
    line2: str,
    col_rgb: tuple,
    font1: ImageFont.FreeTypeFont,
    font2: ImageFont.FreeTypeFont,
    prog: float,
    t: float,
):
    ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    a = min(1.0, prog)
    x0 = SAFE_X0 + 4
    w1 = measure_text(od, line1, font1)
    w2 = measure_text(od, line2, font2)
    label_right = x0 + max(w1, w2) + 8
    y1 = y_center - 28
    y2 = y_center + 18
    draw_text_shadow(od, (x0, y1), line1, font1, (*col_rgb, int(255 * a)))
    draw_text_shadow(od, (x0, y2), line2, font2, (245, 248, 255, int(255 * a)))
    canvas.alpha_composite(ov)
    draw_pointer_arrow_to_label(canvas, muscle_pt, label_right, y_center, col_rgb, prog)
    draw_muscle_contour(canvas, muscle_pt, col_rgb, prog, t)


def draw_v15_top_header(canvas: Image.Image, is_correct: bool, t: float):
    """Dual posture preview strip + title + RX pill inside safe zone y=238..395."""
    ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    vign = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    vd = ImageDraw.Draw(vign)
    vd.rounded_rectangle([SAFE_X0, 238, 890, 398], radius=28, fill=(6, 10, 16, 150))
    vign = vign.filter(ImageFilter.GaussianBlur(radius=22))
    canvas.alpha_composite(vign)

    title = "CABLE ROPE PUSHDOWN"
    tw = measure_text(od, title, FONT_TITLE)
    tx = max(SAFE_X0 + 8, min(560 - tw // 2, 520))
    draw_text_shadow(od, (tx, 248), title, FONT_TITLE, (248, 252, 255, 255))

    rx = "4 SETS × 12 REPS · SLOW TEMPO"
    rw = measure_text(od, rx, FONT_RX_PILL) + 36
    rx0 = SAFE_X0 + 8
    ry0 = 312
    od.rounded_rectangle([rx0, ry0, rx0 + rw, ry0 + 44], radius=22, fill=(255, 200, 60, 235), outline=(0, 230, 255, 220), width=2)
    draw_text_shadow(od, (rx0 + 18, ry0 + 4), rx, FONT_RX_PILL, (12, 16, 22, 255), shadow=False)

    # Dual mini previews (590..885, 242..392)
    slots = [
        (598, 248, False, "✕ WRONG"),
        (738, 248, True, "✓ CORRECT"),
    ]
    for sx, sy, is_c, lbl in slots:
        active = (not is_correct and not is_c) or (is_correct and is_c)
        accent = (40, 250, 125) if is_c else (255, 58, 48)
        border = accent if active else (80, 90, 110)
        bw = 138
        od.rounded_rectangle([sx, sy, sx + bw, sy + 144], radius=16, outline=(*border, 255 if active else 140), width=5 if active else 2)
        if active:
            glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            gd.rounded_rectangle([sx - 4, sy - 4, sx + bw + 4, sy + 148], radius=18, fill=(*accent, 90))
            glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
            canvas.alpha_composite(glow)
        thumb = CORRECT_THUMB if is_c else WRONG_THUMB
        canvas.alpha_composite(thumb.convert("RGBA"), dest=(sx + 4, sy + 4))
        badge_col = (255, 58, 48) if not is_c else (40, 250, 125)
        od.rounded_rectangle([sx + 96, sy + 6, sx + bw - 6, sy + 32], radius=10, fill=(*badge_col, 240))
        draw_text_shadow(od, (sx + 102, sy + 6), lbl.split()[0], FONT_MINI_LBL, (255, 255, 255, 255), shadow=False)

    canvas.alpha_composite(ov)


def draw_bottom_coaching(canvas: Image.Image, is_correct: bool, prog: float):
    """Spotlight coaching cue + RX pill + 12-segment activation bar (y=1135..1345)."""
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    cx = (SAFE_X0 + SAFE_X1) // 2

    vign = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    vd = ImageDraw.Draw(vign)
    vd.ellipse([SAFE_X0 - 20, 1120, SAFE_X1 + 20, 1360], fill=(2, 5, 10, int(210 * a)))
    vign = vign.filter(ImageFilter.GaussianBlur(radius=38))
    canvas.alpha_composite(vign)

    ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    emblem = "✓" if is_correct else "✕"
    od.text((SAFE_X0 + 12, 1148), emblem, font=FONT_TITLE, fill=(*accent, int(255 * a)))

    hero = "LOCK ELBOW IN & SPREAD ROPE OUT" if is_correct else "ELBOW SWINGS & SHOULDER ROLLS"
    hw = measure_text(od, hero, FONT_BOTTOM_HERO)
    draw_text_shadow(od, (cx - hw // 2, 1158), hero, FONT_BOTTOM_HERO, (255, 255, 255, int(255 * a)))

    pill = "4 SETS × 12 SLOW REPS · 2s SQUEEZE"
    pw = measure_text(od, pill, FONT_BOTTOM_PILL) + 32
    px0 = cx - pw // 2
    py0 = 1220
    od.rounded_rectangle([px0, py0, px0 + pw, py0 + 38], radius=19, fill=(0, 0, 0, int(180 * a)), outline=(255, 210, 80, int(220 * a)), width=2)
    draw_text_shadow(od, (px0 + 16, py0 + 4), pill, FONT_BOTTOM_PILL, (255, 220, 100, int(255 * a)))

    bar_label = "98% TRICEPS ISOLATION" if is_correct else "24% TRICEPS (LEAK)"
    blw = measure_text(od, bar_label, FONT_BOTTOM_BAR)
    draw_text_shadow(od, (cx - blw // 2, 1272), bar_label, FONT_BOTTOM_BAR, (*accent, int(255 * a)))

    bar_x0, bar_x1 = SAFE_X0 + 40, SAFE_X1 - 40
    bar_y, bar_h = 1310, 22
    n_seg = 12
    gap = 5
    seg_w = (bar_x1 - bar_x0 - gap * (n_seg - 1)) / float(n_seg)
    pct = 98 if is_correct else 24
    active = int(round((pct / 100.0) * n_seg * a))
    for i in range(n_seg):
        sx0 = int(bar_x0 + i * (seg_w + gap))
        sx1 = int(sx0 + seg_w)
        if i < active:
            od.rounded_rectangle([sx0, bar_y, sx1, bar_y + bar_h], radius=5, fill=(*accent, int(250 * a)))
        else:
            od.rounded_rectangle([sx0, bar_y, sx1, bar_y + bar_h], radius=5, fill=(30, 38, 52, int(220 * a)))

    canvas.alpha_composite(ov)


def draw_ig_reels_overlay(canvas: Image.Image):
    """Simulator chrome outside the content safe zone (top bar, side actions, bottom caption)."""
    ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.rectangle([0, 0, W, 180], fill=(0, 0, 0, 110))
    od.text((36, 52), "Reels", font=FONT_IG, fill=(255, 255, 255, 230))
    for i, y in enumerate([420, 540, 660, 780]):
        od.ellipse([962, y, 1012, y + 50], outline=(255, 255, 255, 180), width=2)
    od.rectangle([0, 1460, W, H], fill=(0, 0, 0, 150))
    od.text((36, 1488), "@gymform.3d", font=FONT_IG, fill=(255, 255, 255, 240))
    od.text((36, 1524), "Cable Rope Pushdown · Biomechanics", font=FONT_MINI_LBL, fill=(200, 210, 225, 220))
    canvas.alpha_composite(ov)


def render_frame(t: float) -> np.ndarray:
    is_correct = t >= 7.5
    local_t = t - 7.5 if is_correct else t
    rep, at_peak = rep_value(local_t)
    pool = CORRECT_POOL if is_correct else WRONG_POOL
    raw = sample_motion_frame(pool, rep, is_wrong=not is_correct)

    zoom = lerp(1.18, 1.28, rep) if is_correct else lerp(1.15, 1.24, rep)
    cy = lerp(800, 780, rep)
    framed, tf = crop_camera_view(raw, cx=700, cy=cy, zoom=zoom)
    canvas = Image.fromarray(cv2.cvtColor(framed, cv2.COLOR_BGR2RGB)).convert("RGBA")

    prog = smoothstep(0.0, 0.35, local_t % 3.75)

    if not is_correct:
        sh = tf((640, 492))
        delt = tf((615, 548))
        elbow = tf((680, 745))
        draw_shockwave_rings(canvas, sh, (255, 50, 60), t, prog)
        draw_left_callout(
            canvas,
            470,
            sh,
            "⚠ JOINT INJURY RISK",
            "SHOULDER IMPINGEMENT",
            (255, 55, 60),
            FONT_CALLOUT_LG,
            FONT_CALLOUT_ML,
            prog,
            t,
        )
        draw_left_callout(
            canvas,
            690,
            delt,
            "✕ WRONG MUSCLE · 68% LOAD",
            "ANTERIOR DELTOID",
            (255, 120, 45),
            FONT_CALLOUT_M,
            FONT_CALLOUT_ML,
            prog,
            t,
        )
        draw_shockwave_rings(canvas, elbow, (255, 180, 40), t + 0.2, prog)
        draw_left_callout(
            canvas,
            910,
            elbow,
            "⚠ TRICEPS TENSION LOST (24%)",
            "ELBOW SWING & WRIST STRAIN",
            (255, 190, 50),
            FONT_CALLOUT_SM,
            FONT_CALLOUT_SM,
            prog,
            t,
        )
        draw_lock_on_joint_ring(canvas, sh, is_correct=False, r_outer=48, prog=prog)
    else:
        lat = tf((648, 578))
        long_h = tf((682, 625))
        elbow = tf((654, 718))
        draw_left_callout(
            canvas,
            470,
            lat,
            "✓ PRIMARY TARGET · 98% LOAD",
            "TRICEPS LATERAL HEAD",
            (40, 255, 120),
            FONT_CALLOUT_M,
            FONT_CALLOUT_M,
            prog,
            t,
        )
        draw_left_callout(
            canvas,
            690,
            long_h,
            "✓ FULL STRETCH · 0% STRAIN",
            "TRICEPS LONG HEAD",
            (120, 255, 80),
            FONT_CALLOUT_M,
            FONT_CALLOUT_M,
            prog,
            t,
        )
        draw_lock_on_joint_ring(canvas, elbow, is_correct=True, r_outer=52, prog=prog)
        draw_left_callout(
            canvas,
            910,
            elbow,
            "✓ ANATOMICAL KEY · ELBOW PINNED",
            "SCAPULA & ELBOW LOCKED",
            (40, 250, 125),
            FONT_CALLOUT_SM,
            FONT_CALLOUT_SM,
            prog,
            t,
        )
        if at_peak:
            wh = tf((520, 980))
            draw_bold_3d_arrow(canvas, tf((520, 980)), tf((420, 1060)), tf((320, 1140)), (40, 250, 125), prog=0.9)
            draw_bold_3d_arrow(canvas, wh, tf((620, 1065)), tf((720, 1175)), (40, 250, 125), prog=0.9)

    draw_v15_top_header(canvas, is_correct, t)
    draw_bottom_coaching(canvas, is_correct, smoothstep(0.2, 0.55, local_t))

    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def render_frame_idx(idx: int) -> int:
    t = idx / FPS
    bgr = render_frame(t)
    os.makedirs(TMP_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return idx


async def _tts_file(text: str, path: str, voice: str = "en-US-ChristopherNeural"):
    import edge_tts

    comm = edge_tts.Communicate(text, voice)
    await comm.save(path)


def synthesize_voiceover(wav_out: str):
    import tempfile

    vdir = tempfile.mkdtemp(prefix="v15_tts_")
    wrong_mp3 = os.path.join(vdir, "wrong.mp3")
    correct_mp3 = os.path.join(vdir, "correct.mp3")
    asyncio.run(_tts_file(VO_WRONG, wrong_mp3))
    asyncio.run(_tts_file(VO_CORRECT, correct_mp3))

    mixed = os.path.join(vdir, "mixed.wav")
    # Wrong VO at 0s, correct at 7.5s; bed + SFX from numpy, duck under speech
    sr = 44100
    n = int(sr * DURATION)
    t_arr = np.arange(n, dtype=np.float32) / sr
    bed = np.zeros(n, dtype=np.float32)
    beat = 0.5 + 0.5 * np.sin(2 * math.pi * 1.333 * t_arr)
    bed += 0.07 * np.sin(2 * math.pi * 54 * t_arr) * beat
    bed += 0.035 * np.sin(2 * math.pi * 110 * t_arr) * beat

    for t_cut in (7.5,):
        m = (t_arr >= t_cut - 0.08) & (t_arr < t_cut + 0.25)
        dt = t_arr[m] - (t_cut - 0.08)
        bed[m] += 0.12 * np.sin(math.pi * dt / 0.33) * np.sin(2 * math.pi * 220 * dt)

    for ts in (0.0, 7.5):
        m = (t_arr >= ts) & (t_arr < ts + 0.35)
        dt = t_arr[m] - ts
        bed[m] += 0.18 * np.exp(-dt * 20) * np.sin(2 * math.pi * 880 * dt)

    bed_path = os.path.join(vdir, "bed.wav")
    bed_i16 = np.clip(bed / max(1e-6, np.max(np.abs(bed))) * 0.55, -1, 1)
    with wave.open(bed_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((bed_i16 * 32767).astype(np.int16).tobytes())

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            bed_path,
            "-i",
            wrong_mp3,
            "-i",
            correct_mp3,
            "-filter_complex",
            "[1:a]adelay=0|0,volume=1.05[vo1];"
            "[2:a]adelay=7500|7500,volume=1.05[vo2];"
            "[0:a][vo1][vo2]amix=inputs=3:duration=first:dropout_transition=2[aout]",
            "-map",
            "[aout]",
            "-ar",
            str(sr),
            "-ac",
            "1",
            wav_out,
        ],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(vdir, ignore_errors=True)


def build_preview_sheet(wrong_peak: np.ndarray, correct_peak: np.ndarray):
    def add_ig(bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb).convert("RGBA")
        draw_ig_reels_overlay(canvas)
        return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)

    p3 = add_ig(wrong_peak.copy())
    p4 = add_ig(correct_peak.copy())
    pw, ph = 540, 960
    pad = 16
    sheet_w = pw * 2 + pad * 3
    sheet_h = ph * 2 + pad * 3 + 56
    sheet = Image.new("RGB", (sheet_w, sheet_h), (8, 11, 18))
    sd = ImageDraw.Draw(sheet)
    labels = [
        "Panel 1 · ✕ Wrong Peak",
        "Panel 2 · ✓ Correct Peak",
        "Panel 3 · Wrong + IG Reels UI",
        "Panel 4 · Correct + IG Reels UI",
    ]
    panels = [
        cv2.cvtColor(cv2.resize(wrong_peak, (pw, ph)), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.resize(correct_peak, (pw, ph)), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.resize(p3, (pw, ph)), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.resize(p4, (pw, ph)), cv2.COLOR_BGR2RGB),
    ]
    positions = [(pad, 56), (pad * 2 + pw, 56), (pad, 56 + ph + pad), (pad * 2 + pw, 56 + ph + pad)]
    for (x, y), panel, lbl in zip(positions, panels, labels):
        sheet.paste(Image.fromarray(panel), (x, y))
        sd.text((x + 8, y - 28 if y < 100 else y - 22), lbl, fill=(220, 230, 245))
    sheet.save(OUT_PREVIEW, quality=95)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)

    existing = (
        os.path.isdir(TMP_DIR)
        and os.path.exists(os.path.join(TMP_DIR, f"frame_{TOTAL_FRAMES - 1:04d}.jpg"))
    )
    if not existing:
        print(f"[v15] Rendering {TOTAL_FRAMES} frames @ {FPS}fps...")
        with ProcessPoolExecutor(max_workers=4) as ex:
            list(ex.map(render_frame_idx, range(TOTAL_FRAMES)))
    else:
        print(f"[v15] Reusing {TOTAL_FRAMES} cached frames in {TMP_DIR}")

    wrong_peak_t = 5.625
    correct_peak_t = 7.5 + 5.625
    wrong_peak = render_frame(wrong_peak_t)
    correct_peak = render_frame(correct_peak_t)
    cv2.imwrite(OUT_WRONG_PEAK, wrong_peak, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(OUT_CORRECT_PEAK, correct_peak, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    build_preview_sheet(wrong_peak, correct_peak)

    wav_path = os.path.join(OUT_DIR, "day1_v15_audio.wav")
    print("[v15] Synthesizing voiceover + gym bed...")
    synthesize_voiceover(wav_path)

    print("[v15] Muxing MP4...")
    subprocess.run(
        [
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
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            OUT_MP4,
        ],
        check=True,
        capture_output=True,
    )

    for name in (
        "day1_v15_ultimate_preview.jpg",
        "day1_v15_wrong_peak.jpg",
        "day1_v15_correct_peak.jpg",
        "demic_gym_day1.mp4",
    ):
        src = os.path.join(OUT_DIR, name)
        dst = os.path.join(ARTIFACTS_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    os.remove(wav_path)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("[v15] DONE")
    print(" ", OUT_PREVIEW)
    print(" ", OUT_WRONG_PEAK)
    print(" ", OUT_CORRECT_PEAK)
    print(" ", OUT_MP4)


if __name__ == "__main__":
    main()
