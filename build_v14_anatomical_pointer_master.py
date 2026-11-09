#!/usr/bin/env python3
"""
@gymform.3d — v14 Slow-Rep 3D Anatomical Muscle Contour & Arrow-Pointer Master
Addresses 중현 님's core requirements (t68u, t69u, t70u):
1. Posture-Dependent Target Muscle Shift & 3D Anatomical Visualization:
   - Sculpted 3D muscle belly contours + striated muscle fiber lines drawn directly onto the 3D mannequin:
     * WRONG FORM (✕): Anterior Deltoid (compensating muscle stealing load) glows orange-red with 3D fiber contour,
       while Triceps stays dim/inactive. Shoulder Joint (Acromion/Rotator Cuff) & Wrist Joint pulse with 3D X-Ray
       Shockwave rings showing INJURY RISK.
     * CORRECT FORM (✓): Triceps Lateral Head & Triceps Long Head light up with 3D sculpted muscle belly contours
       and glowing fiber striations (98% Peak Activation), while the Elbow Joint is locked in a green 3D hinge ring.
2. Direct Arrow Pointers from Muscle/Joint to Giant Muscle Name Typography ("근육부위에 대한게 화살표로 나와서 명칭을 가리키도록"):
   - Bold 3D glowing arrows originate at a glowing anchor node (●) inside the 3D muscle belly / joint and point
     directly at the giant, ultra-legible muscle name & injury risk typography (`◀━━━━━━●`).
3. Maximum Visual Legibility for Typography ("타이포가 시각적으로 잘보여야해"):
   - Deep spotlight vignette behind text zones + 60px~72px Pretendard-Black + 7px 3D black outline stroke.
   - 100% inside Instagram Reels Safe Zone (x: 92..888, y: 236..1355).
4. Slow, Deliberate Repetitions ("너무 빠른 전환이 아닌 조금 느린 반복 자세를 통한 정보전달"):
   - 15.0s Reel = 7.5s WRONG FORM (2 slow 3.75s reps with peak-load pause) -> 7.5s CORRECT FORM (2 slow 3.75s reps with peak-contraction pause).
"""
import os
import math
import wave
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import build_3scene_ox_preview as d1
import build_v9_master_preview as v9

W, H = 1080, 1920
FPS = 30
DURATION = 15.0
TOTAL_FRAMES = int(FPS * DURATION)  # 450 frames
OUT_DIR = "/workspace/daily_info_reels/output"
TMP_DIR = "/workspace/daily_info_reels/frames_day1_v14"
OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")
OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v14_anatomical_pointer_preview.jpg")

PRETENDARD_BLACK = "/home/box/.fonts/Pretendard-Black.otf"
PRETENDARD_EXTRABOLD = "/home/box/.fonts/Pretendard-ExtraBold.otf"
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(path: str, size: int):
    for p in [path, PRETENDARD_BLACK, PRETENDARD_EXTRABOLD, FALLBACK_BOLD]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Ultra-High-Visibility Typography Hierarchy (v14)
FONT_TOP_MODE = load_font(PRETENDARD_BLACK, 34)
FONT_TOP_TITLE = load_font(PRETENDARD_BLACK, 46)

FONT_CALLOUT_TAG = load_font(PRETENDARD_BLACK, 34)
FONT_CALLOUT_MAIN = load_font(PRETENDARD_BLACK, 58)

FONT_BOTTOM_SUB = load_font(PRETENDARD_BLACK, 36)
FONT_BOTTOM_HERO = load_font(PRETENDARD_BLACK, 64)
FONT_SHEET_HDR = load_font(PRETENDARD_BLACK, 30)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def slow_deliberate_rep_curve(local_t: float, rep_period: float = 3.75) -> float:
    """
    Produces a slow, deliberate repetition curve (0.0 -> 1.0 -> 0.0) over `rep_period` seconds (2 slow reps in 7.5s),
    with a distinct 0.85s isometric hold at peak contraction/extension so the viewer can clearly study the anatomy!
    """
    phase = (local_t % rep_period) / rep_period  # 0.0 .. 1.0
    if phase < 0.38:
        # Slow controlled eccentric/concentric drive to peak
        return smoothstep(0.0, 0.38, phase)
    elif phase < 0.64:
        # Deliberate isometric hold at the peak so anatomy & injury risk are crystal clear
        return 1.0 - 0.04 * math.sin(math.pi * (phase - 0.38) / 0.26)
    else:
        # Controlled return
        return 1.0 - smoothstep(0.64, 1.0, phase)


def measure_tracked_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, tracking: float = 0.0) -> float:
    if not text:
        return 0.0
    total = 0.0
    for i, ch in enumerate(text):
        adv = draw.textlength(ch, font=font) if hasattr(draw, "textlength") else (draw.textbbox((0, 0), ch, font=font)[2])
        total += adv + (tracking if i < len(text) - 1 else 0.0)
    return total


def draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    pos: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    tracking: float = 0.0,
    stroke_w: int = 7,
):
    x, y = float(pos[0]), float(pos[1])
    for ch in text:
        ix, iy = int(round(x)), int(round(y))
        if stroke_w > 0:
            draw.text((ix + 4, iy + 6), ch, font=font, fill=(0, 0, 0, 245), stroke_width=stroke_w, stroke_fill=(0, 0, 0, 245))
        draw.text(
            (ix, iy),
            ch,
            font=font,
            fill=fill,
            stroke_width=stroke_w,
            stroke_fill=(3, 6, 12, 255) if stroke_w > 0 else None,
        )
        bbox = draw.textbbox((0, 0), ch, font=font)
        adv = draw.textlength(ch, font=font) if hasattr(draw, "textlength") else (bbox[2] - bbox[0])
        x += adv + tracking


def draw_centered_tracked_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    tracking: float = 0.0,
    stroke_w: int = 7,
):
    w = measure_tracked_text(draw, text, font, tracking=tracking)
    start_x = center_x - w / 2.0
    draw_tracked_text(draw, (start_x, y), text, font, fill=fill, tracking=tracking, stroke_w=stroke_w)
    return start_x, start_x + w


def draw_3d_muscle_contour_and_fibers(
    canvas_rgba: Image.Image,
    center_pt: tuple,
    rx: float,
    ry: float,
    angle_deg: float,
    col_rgb: tuple,
    intensity: float = 1.0,
    is_injury_joint: bool = False,
    pulse_phase: float = 0.0,
):
    """
    Renders either:
    - A 3D Sculpted Muscle Belly Contour + internal anatomical muscle fiber striations (`is_injury_joint=False`), OR
    - A 3D Joint Injury X-Ray Shockwave + jagged impingement warning reticle (`is_injury_joint=True`).
    """
    cx, cy = float(center_pt[0]), float(center_pt[1])
    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    ang_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(ang_rad), math.sin(ang_rad)

    def rot_pt(lx: float, ly: float) -> tuple:
        return (
            int(round(cx + lx * cos_a - ly * sin_a)),
            int(round(cy + lx * sin_a + ly * cos_a)),
        )

    if is_injury_joint:
        # 3D Joint Injury X-Ray Shockwave (concentric pulsing danger rings + anatomical crosshairs)
        for ring_i in range(3):
            r_scale = 1.0 + 0.28 * ring_i + 0.14 * math.sin(pulse_phase * 2.0 * math.pi + ring_i)
            r_cur = int(round(rx * r_scale))
            alpha = int(max(45, (235 - ring_i * 65) * intensity))
            bbox = [int(cx - r_cur), int(cy - r_cur), int(cx + r_cur), int(cy + r_cur)]
            gd.ellipse(bbox, outline=(*col_rgb, alpha), width=9 - ring_i * 2)
            od.ellipse(bbox, outline=(*col_rgb, alpha), width=4 if ring_i == 0 else 2)

        # Inner high-voltage impingement core glow
        core_r = int(rx * 0.55)
        gd.ellipse([int(cx - core_r), int(cy - core_r), int(cx + core_r), int(cy + core_r)], fill=(255, 45, 35, int(185 * intensity)))
        od.ellipse([int(cx - 10), int(cy - 10), int(cx + 10), int(cy + 10)], fill=(255, 255, 255, 255), outline=(*col_rgb, 255), width=3)

        # 4 diagonal impingement shock spikes around the joint
        for deg in [45, 135, 225, 315]:
            ar = math.radians(deg)
            p0 = (int(cx + (rx * 0.65) * math.cos(ar)), int(cy + (rx * 0.65) * math.sin(ar)))
            p1 = (int(cx + (rx * 1.25) * math.cos(ar)), int(cy + (rx * 1.25) * math.sin(ar)))
            od.line([p0, p1], fill=(255, 235, 90, int(240 * intensity)), width=4)
    else:
        # 3D Sculpted Muscle Belly Contour (anatomical spindle polygon + glowing muscle fiber striations)
        pts_outer = []
        pts_inner = []
        n_pts = 36
        for i in range(n_pts):
            theta = 2.0 * math.pi * i / n_pts
            # Anatomical muscle spindle shape (tapered at tendons, fuller in muscle belly)
            taper = 1.0 - 0.22 * (math.cos(theta) ** 2)
            lx = rx * math.cos(theta)
            ly = ry * math.sin(theta) * taper
            pts_outer.append(rot_pt(lx, ly))
            pts_inner.append(rot_pt(lx * 0.72, ly * 0.72))

        # Volumetric muscle fill + outer contour rim
        gd.polygon(pts_outer, fill=(*col_rgb, int(95 * intensity)), outline=(*col_rgb, int(245 * intensity)))
        gd.line(pts_outer + [pts_outer[0]], fill=(*col_rgb, int(240 * intensity)), width=12, joint="curve")
        od.line(pts_outer + [pts_outer[0]], fill=(*col_rgb, int(255 * intensity)), width=5, joint="curve")
        od.line(pts_inner + [pts_inner[0]], fill=(255, 255, 255, int(145 * intensity)), width=2, joint="curve")

        # Internal 3D Striated Muscle Fiber Lines (근섬유 결 시각화)
        for f_off in [-0.55, -0.28, 0.0, 0.28, 0.55]:
            fiber_pts = []
            for s in range(15):
                u = -0.82 + 1.64 * (s / 14.0)
                lx = rx * u
                arch = math.sqrt(max(0.0, 1.0 - u * u))
                ly = ry * f_off * arch
                fiber_pts.append(rot_pt(lx, ly))
            od.line(fiber_pts, fill=(255, 240, 195, int(175 * intensity)), width=2, joint="curve")

    glow = glow.filter(ImageFilter.GaussianBlur(radius=11))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)


def draw_muscle_pointer_arrow_to_label(
    canvas_rgba: Image.Image,
    muscle_pt: tuple,
    elbow_pt: tuple,
    label_anchor_pt: tuple,
    tag_text: str,
    line1_text: str,
    line2_text: str,
    col_rgb: tuple,
    tag_col_rgb: tuple = None,
    prog: float = 1.0,
):
    """
    Draws a bold 3D arrow emerging from `muscle_pt` (inside the highlighted 3D muscle or joint)
    and pointing directly at the giant high-contrast typography block on the left (`◀━━━━━━●`)!
    - `muscle_pt`: origin `(mx, my)` on the 3D muscle/joint
    - `elbow_pt`: intermediate bend point `(ex, ey)`
    - `label_anchor_pt`: tip of the arrowhead `(ax, ay)` right beside the giant typography!
    """
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    if tag_col_rgb is None:
        tag_col_rgb = col_rgb

    mx, my = int(muscle_pt[0]), int(muscle_pt[1])
    ex, ey = int(elbow_pt[0]), int(elbow_pt[1])
    ax, ay = int(label_anchor_pt[0]), int(label_anchor_pt[1])

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # 1. Glowing Origin Node inside the 3D Muscle / Joint (`●`)
    gd.ellipse([mx - 16, my - 16, mx + 16, my + 16], fill=(*col_rgb, int(230 * a)))
    od.ellipse([mx - 10, my - 10, mx + 10, my + 10], fill=(255, 255, 255, int(255 * a)), outline=(3, 6, 12, int(255 * a)), width=3)
    od.ellipse([mx - 5, my - 5, mx + 5, my + 5], fill=(*col_rgb, int(255 * a)))

    # 2. Bold 3D Arrow Shaft from Muscle -> Elbow -> Arrow Tip beside Typography
    # Shorten shaft end slightly so the arrowhead tip lands cleanly at (ax, ay)
    dx = ax - ex
    dy = ay - ey
    seg_len = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / seg_len, dy / seg_len
    px, py = -uy, ux

    shaft_end = (int(round(ax - ux * 22)), int(round(ay - uy * 22)))
    pts_shaft = [(mx, my), (ex, ey), shaft_end]

    # Dark 3D outline under shaft for 100% contrast over any background
    od.line(pts_shaft, fill=(3, 6, 12, int(245 * a)), width=14, joint="curve")
    gd.line(pts_shaft, fill=(*col_rgb, int(225 * a)), width=16, joint="curve")
    od.line(pts_shaft, fill=(*col_rgb, int(255 * a)), width=7, joint="curve")
    od.line(pts_shaft, fill=(255, 255, 255, int(235 * a)), width=2, joint="curve")

    # 3. Sculpted 3D Arrowhead (`◀`) pointing directly at the Muscle Name Typography!
    head_len = 30.0
    head_w = 19.0
    tip = (ax, ay)
    base_c = (ax - ux * head_len, ay - uy * head_len)
    left_w = (int(round(base_c[0] + px * head_w)), int(round(base_c[1] + py * head_w)))
    right_w = (int(round(base_c[0] - px * head_w)), int(round(base_c[1] - py * head_w)))
    notch = (int(round(ax - ux * (head_len * 0.72))), int(round(ay - uy * (head_len * 0.72))))
    arrow_poly = [tip, left_w, notch, right_w]

    gd.polygon(arrow_poly, fill=(*col_rgb, int(240 * a)))
    od.line(arrow_poly + [tip], fill=(3, 6, 12, int(250 * a)), width=6, joint="curve")
    od.polygon(arrow_poly, fill=(*col_rgb, int(255 * a)), outline=(255, 255, 255, int(235 * a)))

    # 4. Giant High-Contrast Typography Block on the Left (`x = 96 .. ax - 18`)
    text_left = 96
    top_y = ay - 78

    # Vertical glowing accent bar on the left edge of the typography block (`x = 84`)
    bar_x = 82
    gd.line([(bar_x, top_y + 6), (bar_x, top_y + 158)], fill=(*col_rgb, int(225 * a)), width=12)
    od.line([(bar_x, top_y + 6), (bar_x, top_y + 158)], fill=(*col_rgb, int(255 * a)), width=6)

    # Overline Tag (e.g. "⚠ INJURY RISK" or "✓ TARGET MUSCLE · 98%") in 34px Bold
    draw_tracked_text(
        od,
        (text_left, top_y),
        tag_text,
        FONT_CALLOUT_TAG,
        fill=(*tag_col_rgb, int(255 * a)),
        tracking=1.0,
        stroke_w=6,
    )

    # Main Anatomical Name Line 1 (58px Pretendard-Black Pure White)
    draw_tracked_text(
        od,
        (text_left, top_y + 42),
        line1_text,
        FONT_CALLOUT_MAIN,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.0,
        stroke_w=7,
    )

    # Main Anatomical Name Line 2 (58px Pretendard-Black Vivid Accent)
    if line2_text:
        draw_tracked_text(
            od,
            (text_left, top_y + 104),
            line2_text,
            FONT_CALLOUT_MAIN,
            fill=(*col_rgb, int(255 * a)),
            tracking=1.0,
            stroke_w=7,
        )

    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)


def draw_v14_top_header(
    canvas_rgba: Image.Image,
    exercise_name: str,
    is_correct: bool,
    rep_progress: float,
):
    """
    Top Header inside Instagram Reels Safe Zone (`y = 238..340`, `cx = 495`):
    - Shows slow-rep progress bar + mode indicator ("✕ WRONG FORM · INJURY RISK" vs "✓ CORRECT FORM · TARGET ISOLATION")
    - Giant 46px Exercise Title ("CABLE ROPE PUSHDOWN")
    """
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    cx = 495

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    mode_str = "✓ CORRECT FORM · MUSCLE ISOLATION" if is_correct else "✕ WRONG FORM · INJURY & FORCE LEAK"
    draw_centered_tracked_text(
        od,
        center_x=cx,
        y=236,
        text=mode_str,
        font=FONT_TOP_MODE,
        fill=(*accent, 255),
        tracking=1.2,
        stroke_w=6,
    )

    draw_centered_tracked_text(
        od,
        center_x=cx,
        y=280,
        text=exercise_name,
        font=FONT_TOP_TITLE,
        fill=(255, 255, 255, 255),
        tracking=1.4,
        stroke_w=7,
    )

    # 2-Phase Slow Rep Progress Indicator Bar at y = 342
    bar_w = 260
    gap = 20
    x0_w = cx - bar_w - gap // 2
    x0_c = cx + gap // 2
    by = 342
    bh = 10

    if not is_correct:
        od.rounded_rectangle([x0_w, by, x0_w + bar_w, by + bh], radius=5, fill=(255, 255, 255, 65))
        fill_w = int(bar_w * min(1.0, max(0.08, rep_progress)))
        gd.rounded_rectangle([x0_w - 3, by - 3, x0_w + fill_w + 3, by + bh + 3], radius=6, fill=(*accent, 220))
        od.rounded_rectangle([x0_w, by, x0_w + fill_w, by + bh], radius=5, fill=(*accent, 255))
        od.rounded_rectangle([x0_c, by, x0_c + bar_w, by + bh], radius=5, fill=(255, 255, 255, 55))
    else:
        od.rounded_rectangle([x0_w, by, x0_w + bar_w, by + bh], radius=5, fill=(255, 80, 70, 140))
        od.rounded_rectangle([x0_c, by, x0_c + bar_w, by + bh], radius=5, fill=(255, 255, 255, 65))
        fill_w = int(bar_w * min(1.0, max(0.08, rep_progress)))
        gd.rounded_rectangle([x0_c - 3, by - 3, x0_c + fill_w + 3, by + bh + 3], radius=6, fill=(*accent, 220))
        od.rounded_rectangle([x0_c, by, x0_c + fill_w, by + bh], radius=5, fill=(*accent, 255))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)


def draw_v14_bottom_posture_summary(
    canvas_rgba: Image.Image,
    is_correct: bool,
    sub_text: str,
    hero_line1: str,
    hero_line2: str,
    prog: float = 1.0,
):
    """
    Lower-Center Posture Action Summary (`y = 1135..1345`, 100% inside IG Reels Safe Zone):
    Displays what the posture is doing and how it affects the target muscle.
    """
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    cx = 495

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    draw_centered_tracked_text(
        od,
        center_x=cx,
        y=1142,
        text=sub_text,
        font=FONT_BOTTOM_SUB,
        fill=(255, 215, 75, int(255 * a)) if not is_correct else (*accent, int(255 * a)),
        tracking=1.2,
        stroke_w=6,
    )

    draw_centered_tracked_text(
        gd,
        center_x=cx,
        y=1190,
        text=hero_line1,
        font=FONT_BOTTOM_HERO,
        fill=(*accent, int(130 * a)),
        tracking=1.4,
        stroke_w=0,
    )
    draw_centered_tracked_text(
        od,
        center_x=cx,
        y=1190,
        text=hero_line1,
        font=FONT_BOTTOM_HERO,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.4,
        stroke_w=7,
    )

    draw_centered_tracked_text(
        gd,
        center_x=cx,
        y=1262,
        text=hero_line2,
        font=FONT_BOTTOM_HERO,
        fill=(*accent, int(160 * a)),
        tracking=1.4,
        stroke_w=0,
    )
    draw_centered_tracked_text(
        od,
        center_x=cx,
        y=1262,
        text=hero_line2,
        font=FONT_BOTTOM_HERO,
        fill=(*accent, int(255 * a)),
        tracking=1.4,
        stroke_w=7,
    )

    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)


def render_v14_day1_frame(t: float) -> np.ndarray:
    """
    Renders a single frame at time `t` (0.0 .. 15.0s) for Day 1 (Cable Rope Pushdown):
    - 0.0s .. 7.5s: ✕ WRONG FORM (2 Slow Deliberate Reps, 3.75s each)
      Shows:
        1) 3D Joint Impingement Shockwave on Shoulder Joint (`SHOULDER IMPINGEMENT` — Injury Risk!)
        2) 3D Muscle Contour on Anterior Deltoid (`ANTERIOR DELTOID · 68%` — Wrong Muscle Targeted!)
        3) 3D Wrist Strain highlight (`WRIST EXTENSION STRAIN`)
        4) Bold 3D arrows pointing directly from the Shoulder Joint & Anterior Deltoid to their giant labels!
    - 7.5s .. 15.0s: ✓ CORRECT FORM (2 Slow Deliberate Reps, 3.75s each)
      Shows:
        1) 3D Sculpted Muscle Contour + Striated Fibers on `TRICEPS LATERAL HEAD (98%)`
        2) 3D Sculpted Muscle Contour on `TRICEPS LONG HEAD` + Green Hinge Ring on `ELBOW PINNED`
        3) Bold 3D arrows pointing directly from the Triceps Lateral Head & Long Head/Elbow to their giant labels!
    """
    is_correct = (t >= 7.5)
    local_t = t if not is_correct else (t - 7.5)
    phase_progress = local_t / 7.5
    prog = smoothstep(0.0, 0.45, local_t)
    rep = slow_deliberate_rep_curve(local_t, rep_period=3.75)

    frames_src = d1.CORRECT_FRAMES if is_correct else d1.WRONG_FRAMES
    n = len(frames_src)
    u = min(1.0, max(0.0, rep * 0.72))
    f_idx = u * (n - 1)
    i0 = int(math.floor(f_idx))
    i1 = min(n - 1, i0 + 1)
    frac = f_idx - i0
    base = frames_src[i0].copy() if frac < 1e-3 else cv2.addWeighted(frames_src[i0], 1.0 - frac, frames_src[i1], frac, 0)
    enh = d1.enhance_plate(base, is_wrong=not is_correct, boost_level=0.55 + 0.45 * rep)

    # Camera framing shifted slightly right (`cx=660`) so the mannequin stands on `x=520..880`,
    # leaving a generous dark studio stage on the left (`x=80..500`) for giant muscle typography & arrows!
    z = lerp(1.16, 1.22, smoothstep(0.0, 3.5, local_t))
    framed_bgr, tf = d1.crop_camera_view(enh, cx=660, cy=835, zoom=z)
    canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Deep studio spotlight vignette behind the left typography stage, top header, and bottom cue
    # so every letter has 100% contrast without rectangular boxes!
    vign = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vign)
    vd.ellipse([-60, 380, 560, 1120], fill=(2, 5, 10, int(225 * prog)))
    vd.ellipse([50, 185, 940, 385], fill=(2, 5, 10, 210))
    vd.ellipse([40, 1095, 950, 1385], fill=(2, 5, 10, int(235 * prog)))
    vign = vign.filter(ImageFilter.GaussianBlur(radius=38))
    canvas.alpha_composite(vign)

    if not is_correct:
        # --- ✕ WRONG FORM: Injury Risk (Shoulder Impingement) + Wrong Muscle Targeted (Anterior Deltoid) ---
        shoulder_joint = tf((775, lerp(575, 595, rep)))
        anterior_delt = tf((715, lerp(655, 675, rep)))
        elbow_drift = tf((775, 825))
        wrist_strain = tf((535, 955))

        # 1. Posture fault line along drifting humerus & bent wrist
        d1.draw_single_posture_line(canvas, [shoulder_joint, elbow_drift, wrist_strain], col_rgb=(255, 58, 48), width=11, prog=prog)

        # 2. 3D Joint Injury X-Ray Shockwave on Shoulder Joint + Wrist Strain
        draw_3d_muscle_contour_and_fibers(
            canvas,
            center_pt=shoulder_joint,
            rx=56,
            ry=56,
            angle_deg=0,
            col_rgb=(255, 45, 38),
            intensity=0.75 + 0.25 * rep,
            is_injury_joint=True,
            pulse_phase=local_t * 1.6,
        )
        draw_3d_muscle_contour_and_fibers(
            canvas,
            center_pt=wrist_strain,
            rx=36,
            ry=36,
            angle_deg=0,
            col_rgb=(255, 165, 35),
            intensity=0.65 + 0.35 * rep,
            is_injury_joint=True,
            pulse_phase=local_t * 1.6 + 0.5,
        )

        # 3. 3D Sculpted Muscle Contour on the WRONG Compensating Muscle (Anterior Deltoid)
        draw_3d_muscle_contour_and_fibers(
            canvas,
            center_pt=anterior_delt,
            rx=64,
            ry=42,
            angle_deg=-58,
            col_rgb=(255, 125, 35),
            intensity=0.75 + 0.25 * rep,
            is_injury_joint=False,
        )

        # 4. Motion Fault Arrow showing shoulder rolling forward/down
        d1.draw_bold_3d_arrow(
            canvas,
            p0=tf((835, 485)),
            p1=tf((725, 515)),
            p2=tf((655, 615)),
            col_rgb=(255, 58, 48),
            prog=min(1.0, 0.45 + 0.55 * rep),
        )

        # 5. Direct 3D Arrow Pointer #1: Shoulder Joint -> "⚠ INJURY RISK / SHOULDER IMPINGEMENT"
        draw_muscle_pointer_arrow_to_label(
            canvas,
            muscle_pt=shoulder_joint,
            elbow_pt=(int(shoulder_joint[0] - 95), 535),
            label_anchor_pt=(525, 535),
            tag_text="⚠ JOINT INJURY RISK",
            line1_text="SHOULDER",
            line2_text="IMPINGEMENT",
            col_rgb=(255, 62, 52),
            tag_col_rgb=(255, 215, 65),
            prog=prog,
        )

        # 6. Direct 3D Arrow Pointer #2: Anterior Deltoid Belly -> "✕ WRONG TARGET / ANTERIOR DELTOID"
        delt_pct = int(round(lerp(56, 68, rep)))
        draw_muscle_pointer_arrow_to_label(
            canvas,
            muscle_pt=anterior_delt,
            elbow_pt=(int(anterior_delt[0] - 85), 815),
            label_anchor_pt=(495, 815),
            tag_text=f"✕ WRONG MUSCLE ({delt_pct}%)",
            line1_text="ANTERIOR",
            line2_text="DELTOID",
            col_rgb=(255, 135, 40),
            tag_col_rgb=(255, 110, 95),
            prog=prog,
        )

        # 7. Lower-Center Slow-Rep Biomechanical Summary
        tri_pct = int(round(lerp(34, 24, rep)))
        draw_v14_bottom_posture_summary(
            canvas,
            is_correct=False,
            sub_text=f"✕ TRICEPS TENSION LOST ({tri_pct}%) · WRIST STRAIN",
            hero_line1="ELBOW SWINGS",
            hero_line2="& SHOULDER ROLLS",
            prog=prog,
        )

    else:
        # --- ✓ CORRECT FORM: 3D Triceps Lateral & Long Head Isolation + Joint Locked ---
        triceps_lateral = tf((725, lerp(695, 710, rep)))
        triceps_long = tf((782, lerp(735, 750, rep)))
        elbow_pinned = tf((735, 845))
        wrist_neutral = tf((515, 965))

        # 1. Clean green alignment axis from pinned elbow to neutral wrist
        d1.draw_single_posture_line(canvas, [elbow_pinned, wrist_neutral], col_rgb=(40, 250, 125), width=11, prog=prog)

        # 2. 3D Sculpted Muscle Contours & Striated Fibers on TRICEPS LATERAL HEAD & LONG HEAD
        draw_3d_muscle_contour_and_fibers(
            canvas,
            center_pt=triceps_lateral,
            rx=82,
            ry=44,
            angle_deg=78,
            col_rgb=(40, 250, 125),
            intensity=0.75 + 0.25 * rep,
            is_injury_joint=False,
        )
        draw_3d_muscle_contour_and_fibers(
            canvas,
            center_pt=triceps_long,
            rx=74,
            ry=36,
            angle_deg=82,
            col_rgb=(160, 255, 90),
            intensity=0.70 + 0.30 * rep,
            is_injury_joint=False,
        )

        # 3. Biomechanical Lock-On Pivot Ring on Pinned Elbow Joint
        d1.draw_lock_on_joint_ring(canvas, elbow_pinned, is_correct=True, r_outer=52, prog=prog)

        # 4. Rope Spread 3D Arrow at bottom lockout
        d1.draw_bold_3d_arrow(
            canvas,
            p0=tf((515, 965)),
            p1=tf((425, 1035)),
            p2=tf((335, 1120)),
            col_rgb=(40, 250, 125),
            prog=min(1.0, 0.40 + 0.60 * rep),
        )
        d1.draw_bold_3d_arrow(
            canvas,
            p0=tf((515, 965)),
            p1=tf((605, 1045)),
            p2=tf((695, 1135)),
            col_rgb=(40, 250, 125),
            prog=min(1.0, 0.40 + 0.60 * rep),
        )

        # 5. Direct 3D Arrow Pointer #1: Triceps Lateral Head -> "✓ PRIMARY TARGET (98%) / TRICEPS LATERAL HEAD"
        act_pct = int(round(lerp(86, 98, rep)))
        draw_muscle_pointer_arrow_to_label(
            canvas,
            muscle_pt=triceps_lateral,
            elbow_pt=(int(triceps_lateral[0] - 95), 545),
            label_anchor_pt=(520, 545),
            tag_text=f"✓ PRIMARY TARGET ({act_pct}%)",
            line1_text="TRICEPS",
            line2_text="LATERAL HEAD",
            col_rgb=(40, 250, 125),
            tag_col_rgb=(130, 255, 175),
            prog=prog,
        )

        # 6. Direct 3D Arrow Pointer #2: Triceps Long Head & Pinned Elbow -> "✓ SECONDARY + JOINT LOCK / TRICEPS LONG HEAD"
        draw_muscle_pointer_arrow_to_label(
            canvas,
            muscle_pt=triceps_long,
            elbow_pt=(int(triceps_long[0] - 115), 825),
            label_anchor_pt=(495, 825),
            tag_text="✓ FULL STRETCH · 0% STRAIN",
            line1_text="TRICEPS",
            line2_text="LONG HEAD",
            col_rgb=(145, 255, 95),
            tag_col_rgb=(40, 250, 125),
            prog=prog,
        )

        # 7. Lower-Center Slow-Rep Biomechanical Summary
        draw_v14_bottom_posture_summary(
            canvas,
            is_correct=True,
            sub_text="✓ ELBOW PINNED AT SIDE · SCAPULA LOCKED",
            hero_line1="LOCK ELBOW IN",
            hero_line2="& SPREAD ROPE OUT",
            prog=prog,
        )

    draw_v14_top_header(canvas, "CABLE ROPE PUSHDOWN", is_correct=is_correct, rep_progress=phase_progress)
    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def build_preview_comparison():
    """Generates a 4-panel preview sheet comparing Wrong vs Correct (Clean & with IG Reels UI Safe-Zone Check)."""
    w_frame = render_v14_day1_frame(t=1.85)   # Peak of Slow Rep 1 (Wrong Form)
    c_frame = render_v14_day1_frame(t=9.35)   # Peak of Slow Rep 1 (Correct Form)
    w_ig = v9.overlay_ig_reels_ui_simulator(w_frame)
    c_ig = v9.overlay_ig_reels_ui_simulator(c_frame)

    cv2.imwrite(os.path.join(OUT_DIR, "day1_v14_wrong_peak.jpg"), w_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(os.path.join(OUT_DIR, "day1_v14_correct_peak.jpg"), c_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    scale = 0.55
    pw, ph = int(W * scale), int(H * scale)
    pad = 24
    hdr = 60
    comp = Image.new("RGB", (pad * 5 + pw * 4, pad * 2 + ph + hdr), (7, 9, 14))
    cd = ImageDraw.Draw(comp)
    panels = [
        (w_frame, "1. ✕ WRONG (INJURY & DELT)", (255, 105, 95)),
        (c_frame, "2. ✓ CORRECT (3D TRICEPS)", (85, 255, 150)),
        (w_ig, "3. ✕ WRONG + IG SAFE ZONE", (100, 235, 255)),
        (c_ig, "4. ✓ CORRECT + IG SAFE ZONE", (100, 235, 255)),
    ]
    for idx, (frm, lbl, col_t) in enumerate(panels):
        x0 = pad + idx * (pw + pad)
        draw_tracked_text(cd, (x0 + 8, pad + 8), lbl, FONT_SHEET_HDR, fill=col_t, tracking=0.8, stroke_w=4)
        im = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)).resize((pw, ph), Image.Resampling.LANCZOS)
        comp.paste(im, (x0, pad + hdr))

    comp.save(OUT_PREVIEW, quality=95)
    print("Saved v14 preview:", OUT_PREVIEW)


def _render_worker(idx: int) -> int:
    t = idx / float(FPS)
    bgr = render_v14_day1_frame(t)
    cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return idx


def render_full_mp4():
    os.makedirs(TMP_DIR, exist_ok=True)
    print(f"Rendering {TOTAL_FRAMES} frames (15.0s Slow-Rep v14 Day 1)...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(_render_worker, range(TOTAL_FRAMES)))

    from build_gym_day1 import build_audio_track
    wav_path = os.path.join(OUT_DIR, "day1_v14_audio.wav")
    build_audio_track(wav_path)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", os.path.join(TMP_DIR, "frame_%04d.jpg"),
            "-i", wav_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            OUT_MP4,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(wav_path)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Saved v14 MP4:", OUT_MP4)


if __name__ == "__main__":
    import sys
    build_preview_comparison()
    if "--mp4" in sys.argv:
        render_full_mp4()
