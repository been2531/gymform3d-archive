#!/usr/bin/env python3
"""
@gymform.3d — v11.1 Refined Pure Sculptural Graphic Master Module
("박스/테두리/가로줄 0% · 스마트 비간섭 45° 홀로그램 % 콜아웃 + 에너지 펄스 라이트 스레드 + 시네마 타이포")
"""
import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)

PRETENDARD_BLACK = "/home/box/.fonts/Pretendard-Black.otf"
PRETENDARD_EXTRABOLD = "/home/box/.fonts/Pretendard-ExtraBold.otf"
RAJDHANI_BOLD = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Bold.ttf"
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(path: str, size: int):
    for p in [path, PRETENDARD_BLACK, PRETENDARD_EXTRABOLD, FALLBACK_BOLD]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TOP_TITLE = load_font(PRETENDARD_EXTRABOLD, 28)
FONT_TOP_STEP = load_font(RAJDHANI_BOLD, 22)

FONT_RING_PCT = load_font(RAJDHANI_BOLD, 56)
FONT_RING_SUB = load_font(RAJDHANI_BOLD, 21)

FONT_OVERLINE = load_font(PRETENDARD_EXTRABOLD, 25)
FONT_HERO_CUE = load_font(PRETENDARD_BLACK, 54)
FONT_SHEET_HDR = load_font(PRETENDARD_EXTRABOLD, 26)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


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
    shadow: bool = False,
):
    x, y = float(pos[0]), float(pos[1])
    for ch in text:
        if shadow:
            draw.text((int(round(x)) + 2, int(round(y)) + 3), ch, font=font, fill=(0, 0, 0, 215))
            draw.text((int(round(x)) + 1, int(round(y)) + 1), ch, font=font, fill=(0, 0, 0, 180))
        draw.text((int(round(x)), int(round(y))), ch, font=font, fill=fill)
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
    shadow: bool = True,
):
    w = measure_tracked_text(draw, text, font, tracking=tracking)
    start_x = center_x - w / 2.0
    draw_tracked_text(draw, (start_x, y), text, font, fill=fill, tracking=tracking, shadow=shadow)
    return start_x, start_x + w


def draw_v11_cinema_top(
    canvas_rgba: Image.Image,
    exercise_name: str,
    active_idx: int,
    is_correct: bool,
):
    """
    Pure Floating Cinema Masthead (centered at x = 495, y = 250..316):
    - NO box, NO pill, NO long underline!
    - 3 sleek glowing step bars at y = 254 (with subtle "STEP 01/03" micro tag)
    - Floating wide-tracked exercise title at y = 276
    """
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    cx = 495

    vign = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    vd = ImageDraw.Draw(vign)
    vd.ellipse([130, 210, 860, 355], fill=(3, 6, 11, 145))
    vign = vign.filter(ImageFilter.GaussianBlur(radius=38))
    canvas_rgba.alpha_composite(vign)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bar_w, gap = 58, 14
    total_w = 3 * bar_w + 2 * gap
    start_x = cx - total_w // 2
    bar_y = 254

    for i in range(3):
        bx0 = start_x + i * (bar_w + gap)
        bx1 = bx0 + bar_w
        if i == active_idx:
            gd.rounded_rectangle([bx0 - 3, bar_y - 4, bx1 + 3, bar_y + 8], radius=4, fill=(*accent, 220))
            od.rounded_rectangle([bx0, bar_y - 1, bx1, bar_y + 5], radius=3, fill=(*accent, 255))
            od.line([(bx0 + 4, bar_y + 1), (bx1 - 4, bar_y + 1)], fill=(255, 255, 255, 220), width=2)
        elif i < active_idx:
            od.rounded_rectangle([bx0, bar_y, bx1, bar_y + 4], radius=2, fill=(*accent, 150))
        else:
            od.rounded_rectangle([bx0, bar_y, bx1, bar_y + 4], radius=2, fill=(255, 255, 255, 58))

    draw_centered_tracked_text(
        od,
        center_x=cx,
        y=274,
        text=exercise_name,
        font=FONT_TOP_TITLE,
        fill=(244, 248, 255, 248),
        tracking=3.2,
        shadow=True,
    )

    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)


def draw_v11_sculptural_in_scene_graphic(
    canvas_rgba: Image.Image,
    target_pt: tuple,
    is_correct: bool,
    badge_text: str,
    anatomy_tag: str,
    title_text: str,
    metric_label: str,
    metric_pct: int,
    prog: float = 1.0,
):
    """
    v11.1 Refined Pure Sculptural Graphic & Floating Cinema Typography:
    1. On-Body 270° Radial Arc Gauge around `target_pt` with animated sweep (`prog`).
    2. Smart Non-Overlapping 45° Callout Arm (`o──╱───`) placing the sculptural `%` readout
       in clear studio space so it never collides with the 3D arrow or posture line.
    3. Curved Light-Thread with an animated photon energy pulse gliding from the joint to the title.
    4. Center-Lower Floating Cinema Callout (centered at x = 495, y = 1210..1325, 100% Safe Zone).
    """
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    # Smooth gauge sweep animation during entrance
    sweep = min(1.0, prog * 1.15)
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    cx = 495

    tx, ty = int(target_pt[0]), int(target_pt[1])

    # Smart Clear-Space Placement for the On-Body % Readout:
    # Place it in the upper-left studio space (around x=235..310) when target is on the right/center,
    # and upper-right (around x=740) when target is on the far left, staying well clear of central arrows!
    if tx >= 460:
        badge_side = -1
        rx_center = max(215, min(340, tx - 235))
    else:
        badge_side = 1
        rx_center = min(785, max(640, tx + 235))
    ry_center = max(410, min(1010, ty - 78))

    # Edge-free studio shadow behind floating lower typography & on-body % readout
    vign = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    vd = ImageDraw.Draw(vign)
    vd.ellipse([70, 1145, 920, 1385], fill=(2, 5, 10, int(205 * a)))
    vd.ellipse([rx_center - 110, ry_center - 62, rx_center + 110, ry_center + 58], fill=(2, 5, 10, int(185 * a)))
    vign = vign.filter(ImageFilter.GaussianBlur(radius=36))
    canvas_rgba.alpha_composite(vign)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # --- 1. ON-BODY 3D RADIAL ARC GAUGE AROUND TARGET JOINT ---
    arc_r = 74
    start_deg = 135
    span_deg = 270
    disp_pct = int(round(metric_pct * sweep))
    end_active_deg = start_deg + int(round(span_deg * (disp_pct / 100.0)))

    od.arc(
        [tx - arc_r, ty - arc_r, tx + arc_r, ty + arc_r],
        start=start_deg,
        end=start_deg + span_deg,
        fill=(255, 255, 255, int(55 * a)),
        width=4,
    )
    if end_active_deg > start_deg:
        gd.arc(
            [tx - arc_r, ty - arc_r, tx + arc_r, ty + arc_r],
            start=start_deg,
            end=end_active_deg,
            fill=(*accent, int(230 * a)),
            width=13,
        )
        od.arc(
            [tx - arc_r, ty - arc_r, tx + arc_r, ty + arc_r],
            start=start_deg,
            end=end_active_deg,
            fill=(*accent, int(255 * a)),
            width=6,
        )
        od.arc(
            [tx - arc_r, ty - arc_r, tx + arc_r, ty + arc_r],
            start=start_deg,
            end=end_active_deg,
            fill=(255, 255, 255, int(215 * a)),
            width=2,
        )

    # Outer precision radial degree ticks
    n_ticks = 24
    active_ticks = int(round((disp_pct / 100.0) * n_ticks))
    for i in range(n_ticks):
        ang = math.radians(start_deg + i * (span_deg / float(n_ticks - 1)))
        r0 = arc_r + 9
        r1 = arc_r + (18 if i % 4 == 0 else 13)
        p0 = (tx + int(r0 * math.cos(ang)), ty + int(r0 * math.sin(ang)))
        p1 = (tx + int(r1 * math.cos(ang)), ty + int(r1 * math.sin(ang)))
        if i <= active_ticks:
            od.line([p0, p1], fill=(*accent, int(250 * a)), width=3 if i % 4 == 0 else 2)
        else:
            od.line([p0, p1], fill=(255, 255, 255, int(60 * a)), width=2)

    # Smart 45° Holographic Callout Arm (`o──╱───`) to the Floating % Readout
    pct_str = f"{disp_pct}%"
    short_metric = (
        metric_label.upper()
        .replace("LOAD", "")
        .replace("ISOLATION", "")
        .replace("TENSION", "")
        .replace("ACTIVATION", "")
        .replace("STRESS", "")
        .replace("SHEAR", "")
        .strip()
    )
    if len(short_metric) > 14:
        short_metric = short_metric[:14].strip()

    anchor_ang = math.radians(212 if badge_side < 0 else -32)
    ring_edge_pt = (tx + int((arc_r + 16) * math.cos(anchor_ang)), ty + int((arc_r + 16) * math.sin(anchor_ang)))
    elbow_arm = (
        int(lerp(ring_edge_pt[0], rx_center + (62 if badge_side < 0 else -62), 0.55)),
        ry_center + 16,
    )
    arm_end = (rx_center + (56 if badge_side < 0 else -56), ry_center + 16)

    gd.line([ring_edge_pt, elbow_arm, arm_end], fill=(*accent, int(175 * a)), width=7, joint="curve")
    od.line([ring_edge_pt, elbow_arm, arm_end], fill=(*accent, int(245 * a)), width=3, joint="curve")
    od.ellipse(
        [ring_edge_pt[0] - 5, ring_edge_pt[1] - 5, ring_edge_pt[0] + 5, ring_edge_pt[1] + 5],
        fill=(255, 255, 255, int(255 * a)),
        outline=(*accent, int(255 * a)),
        width=2,
    )
    od.ellipse(
        [arm_end[0] - 4, arm_end[1] - 4, arm_end[0] + 4, arm_end[1] + 4],
        fill=(*accent, int(255 * a)),
    )

    draw_centered_tracked_text(
        gd,
        center_x=rx_center,
        y=ry_center - 32,
        text=pct_str,
        font=FONT_RING_PCT,
        fill=(*accent, int(215 * a)),
        tracking=1.0,
        shadow=False,
    )
    draw_centered_tracked_text(
        od,
        center_x=rx_center,
        y=ry_center - 32,
        text=pct_str,
        font=FONT_RING_PCT,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.0,
        shadow=True,
    )
    draw_centered_tracked_text(
        od,
        center_x=rx_center,
        y=ry_center + 24,
        text=short_metric,
        font=FONT_RING_SUB,
        fill=(*accent, int(250 * a)),
        tracking=1.8,
        shadow=True,
    )

    # --- 2. FADING CURVED LIGHT-THREAD + ANIMATED PHOTON PULSE ---
    thread_start = (tx, min(ty + arc_r + 12, 1165))
    thread_end = (cx, 1192)
    ctrl = (int(lerp(tx, cx, 0.35)), int(lerp(thread_start[1], thread_end[1], 0.65)))
    n_seg = 30
    pulse_pos = (prog * 1.45) % 1.0

    for s in range(n_seg):
        t0 = s / float(n_seg)
        t1 = (s + 1) / float(n_seg)
        x0_b = (1 - t0) ** 2 * thread_start[0] + 2 * (1 - t0) * t0 * ctrl[0] + t0 ** 2 * thread_end[0]
        y0_b = (1 - t0) ** 2 * thread_start[1] + 2 * (1 - t0) * t0 * ctrl[1] + t0 ** 2 * thread_end[1]
        x1_b = (1 - t1) ** 2 * thread_start[0] + 2 * (1 - t1) * t1 * ctrl[0] + t1 ** 2 * thread_end[0]
        y1_b = (1 - t1) ** 2 * thread_start[1] + 2 * (1 - t1) * t1 * ctrl[1] + t1 ** 2 * thread_end[1]

        dist_pulse = abs(t0 - pulse_pos)
        pulse_boost = math.exp(-((dist_pulse / 0.14) ** 2))
        seg_a = int(min(255, (195 * (1.0 - 0.45 * t0) + 60 * pulse_boost)) * a)
        gd.line([(x0_b, y0_b), (x1_b, y1_b)], fill=(*accent, int(seg_a * 0.75)), width=8 if pulse_boost > 0.5 else 6)
        od.line([(x0_b, y0_b), (x1_b, y1_b)], fill=(*accent, seg_a), width=3)
        if pulse_boost > 0.65:
            od.line([(x0_b, y0_b), (x1_b, y1_b)], fill=(255, 255, 255, int(235 * a)), width=2)

    gd.ellipse([cx - 11, 1183, cx + 11, 1205], fill=(*accent, int(230 * a)))
    od.ellipse([cx - 5, 1189, cx + 5, 1199], fill=(255, 255, 255, int(255 * a)), outline=(*accent, int(255 * a)), width=2)

    # --- 3. FLOATING CINEMA CALLOUT (y = 1212..1325, ZERO BOXES, ZERO BARS) ---
    clean_badge = badge_text.replace("✕ ", "").replace("✓ ", "").strip()
    overline_str = f"{clean_badge}  ·  {anatomy_tag}"
    ov_w = measure_tracked_text(od, overline_str, FONT_OVERLINE, tracking=2.2)
    icon_gap = 38
    total_ov_w = ov_w + icon_gap
    ov_start_x = cx - total_ov_w / 2.0
    ov_y = 1212

    ic_cx = int(ov_start_x + 14)
    ic_cy = ov_y + 15
    gd.ellipse([ic_cx - 18, ic_cy - 18, ic_cx + 18, ic_cy + 18], fill=(*accent, int(195 * a)))
    od.ellipse([ic_cx - 14, ic_cy - 14, ic_cx + 14, ic_cy + 14], fill=(*accent, int(255 * a)))
    ic_col = (8, 12, 18, int(255 * a)) if is_correct else (255, 255, 255, int(255 * a))
    if is_correct:
        od.line([(ic_cx - 6, ic_cy + 1), (ic_cx - 1, ic_cy + 6), (ic_cx + 7, ic_cy - 5)], fill=ic_col, width=4, joint="curve")
    else:
        od.line([(ic_cx - 5, ic_cy - 5), (ic_cx + 5, ic_cy + 5)], fill=ic_col, width=4)
        od.line([(ic_cx - 5, ic_cy + 5), (ic_cx + 5, ic_cy - 5)], fill=ic_col, width=4)

    draw_tracked_text(
        od,
        (ov_start_x + icon_gap, ov_y),
        overline_str,
        FONT_OVERLINE,
        fill=(*accent, int(255 * a)),
        tracking=2.2,
        shadow=True,
    )

    # Monumental Floating Hero Cue at y = 1254
    draw_centered_tracked_text(
        gd,
        center_x=cx,
        y=1254,
        text=title_text,
        font=FONT_HERO_CUE,
        fill=(*accent, int(115 * a)),
        tracking=1.8,
        shadow=False,
    )
    draw_centered_tracked_text(
        od,
        center_x=cx,
        y=1254,
        text=title_text,
        font=FONT_HERO_CUE,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.8,
        shadow=True,
    )

    # Subtle centered laser accent flare beneath the Hero Cue
    flare_y = 1326
    for i in range(-18, 19):
        t = abs(i) / 18.0
        fa = int(215 * ((1.0 - t) ** 1.8) * a)
        x_seg = cx + i * 6
        od.line([(x_seg, flare_y), (x_seg + 6, flare_y)], fill=(*accent, fa), width=3 if abs(i) < 6 else 2)

    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)
