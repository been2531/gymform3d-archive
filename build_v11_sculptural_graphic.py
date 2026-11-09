#!/usr/bin/env python3
"""
@gymform.3d — v13 Zero-Micro-Text Giant Sculptural Typography Master Module
("작은 부가 글씨 0개 · 모든 글자 초대형 46px~104px 시네마 디스플레이 타이포 + 박스 0% 인-씬 그래픽")
Why v13 solves "글자들은 잘 보여야해. 너무 작은건 있으나 마나" 100%:
- Deletes ALL small secondary labels ("FORCE LEAK", "PEAK CONTRACTION", long anatomy strings).
- Every single text element on screen is MASSIVE (minimum 44px Pretendard-Black!):
  1) Top Title ("CABLE ROPE PUSHDOWN"): 48px Pretendard-Black + 3 thick 104px glowing step bars (y = 238..325).
  2) On-Body Radial Gauge Readout: 102px Pretendard-Black ("68%" / "98%") + 44px Pretendard-Black ("FRONT DELT" / "TRICEPS").
  3) Center-Lower Hero Cue (y = 1105..1310, well above IG bottom UI):
     Giant 74px Glowing ✕/✓ Disc + 2-Line Stacked 80px Pretendard-Black Hero Cue ("SHOULDER / TAKEOVER", "SPREAD / ROPE OUT")!
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
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(path: str, size: int):
    for p in [path, PRETENDARD_BLACK, PRETENDARD_EXTRABOLD, FALLBACK_BOLD]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Every single font is HUGE (44px to 102px Pretendard-Black) — zero small text!
FONT_TOP_TITLE_48 = load_font(PRETENDARD_BLACK, 48)
FONT_TOP_TITLE_44 = load_font(PRETENDARD_BLACK, 44)

FONT_RING_PCT = load_font(PRETENDARD_BLACK, 98)
FONT_RING_SUB = load_font(PRETENDARD_BLACK, 44)

FONT_HERO_2LINE = load_font(PRETENDARD_BLACK, 78)
FONT_SHEET_HDR = load_font(PRETENDARD_BLACK, 32)


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
    shadow: bool = True,
    stroke_w: int = 6,
):
    x, y = float(pos[0]), float(pos[1])
    for ch in text:
        ix, iy = int(round(x)), int(round(y))
        if shadow:
            draw.text((ix + 4, iy + 6), ch, font=font, fill=(0, 0, 0, 245), stroke_width=stroke_w, stroke_fill=(0, 0, 0, 245))
        draw.text(
            (ix, iy),
            ch,
            font=font,
            fill=fill,
            stroke_width=stroke_w if shadow else 0,
            stroke_fill=(4, 7, 12, 250) if shadow else None,
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
    shadow: bool = True,
    stroke_w: int = 6,
):
    w = measure_tracked_text(draw, text, font, tracking=tracking)
    start_x = center_x - w / 2.0
    draw_tracked_text(draw, (start_x, y), text, font, fill=fill, tracking=tracking, shadow=shadow, stroke_w=stroke_w)
    return start_x, start_x + w


def shorten_muscle_tag(anatomy_tag: str, metric_label: str) -> str:
    src = (anatomy_tag + " " + metric_label).upper()
    if "DELT" in src:
        return "FRONT DELT"
    if "WRIST" in src:
        return "WRIST"
    if "TRICEPS" in src or "ELBOW" in src or "HUMERUS" in src:
        return "TRICEPS"
    if "TRAP" in src and "UPPER" in src:
        return "TRAPS"
    if "LUMBAR" in src or "LOWER BACK" in src:
        return "LOW BACK"
    if "BICEPS" in src:
        return "BICEPS"
    if "LAT" in src or "SCAPULA" in src or "THORACIC" in src:
        return "LATS"
    words = anatomy_tag.upper().split()
    return words[0] if words else "TARGET"


def split_hero_into_two_lines(title_text: str) -> tuple:
    """Splits a 2-4 word hero cue into 2 balanced, punchy lines so it can be rendered at a gigantic 78px font size!"""
    words = title_text.strip().split()
    if len(words) <= 1:
        return (title_text, "")
    if len(words) == 2:
        return (words[0], words[1])
    if len(words) == 3:
        # Balance by character length
        if len(words[0]) >= len(words[1]) + len(words[2]) - 1:
            return (words[0], f"{words[1]} {words[2]}")
        return (f"{words[0]} {words[1]}", words[2])
    mid = len(words) // 2
    return (" ".join(words[:mid]), " ".join(words[mid:]))


def draw_v11_cinema_top(
    canvas_rgba: Image.Image,
    exercise_name: str,
    active_idx: int,
    is_correct: bool,
):
    """
    v13 Giant Floating Cinema Masthead (centered at x = 495, y = 238..330):
    - 3 thick glowing step bars (104px wide, 12px tall each) at y = 240
    - Giant 48px Pretendard-Black exercise title at y = 266
    """
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    cx = 495

    vign = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    vd = ImageDraw.Draw(vign)
    vd.ellipse([65, 190, 925, 380], fill=(2, 5, 10, 205))
    vign = vign.filter(ImageFilter.GaussianBlur(radius=36))
    canvas_rgba.alpha_composite(vign)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bar_w, gap = 104, 20
    total_w = 3 * bar_w + 2 * gap
    start_x = cx - total_w // 2
    bar_y = 240

    for i in range(3):
        bx0 = start_x + i * (bar_w + gap)
        bx1 = bx0 + bar_w
        if i == active_idx:
            gd.rounded_rectangle([bx0 - 5, bar_y - 5, bx1 + 5, bar_y + 16], radius=6, fill=(*accent, 240))
            od.rounded_rectangle([bx0, bar_y, bx1, bar_y + 11], radius=5, fill=(*accent, 255))
            od.line([(bx0 + 6, bar_y + 2), (bx1 - 6, bar_y + 2)], fill=(255, 255, 255, 240), width=2)
        elif i < active_idx:
            od.rounded_rectangle([bx0, bar_y + 1, bx1, bar_y + 10], radius=5, fill=(*accent, 185))
        else:
            od.rounded_rectangle([bx0, bar_y + 1, bx1, bar_y + 10], radius=5, fill=(255, 255, 255, 80))

    t_font = FONT_TOP_TITLE_48
    if measure_tracked_text(od, exercise_name, t_font, tracking=1.2) > 770:
        t_font = FONT_TOP_TITLE_44

    draw_centered_tracked_text(
        od,
        center_x=cx,
        y=266,
        text=exercise_name,
        font=t_font,
        fill=(255, 255, 255, 255),
        tracking=1.2,
        shadow=True,
        stroke_w=5,
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
    v13 Zero-Micro-Text Giant Sculptural Graphic:
    - ZERO small text! Minimum font size on the entire screen is 44px Pretendard-Black.
    - 1) On-Body 270° Radial Arc Gauge + Giant 98px % Readout + 44px Muscle Name ("FRONT DELT" / "TRICEPS").
    - 2) Animated Photon Pulse Light-Thread connecting the joint ring to the lower-center Hero Cue.
    - 3) Center-Lower Hero Callout (y = 1110..1315, well clear of IG UI):
         Giant 74px Glowing ✕/✓ Emblem + 2-Line Stacked 78px Hero Title ("SHOULDER / TAKEOVER", "SPREAD / ROPE OUT")!
    """
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    sweep = min(1.0, prog * 1.15)
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    cx = 495

    tx, ty = int(target_pt[0]), int(target_pt[1])

    # Smart Clear-Space Placement for the Giant 98px % Readout + 44px Muscle Tag
    if tx >= 450:
        badge_side = -1
        rx_center = max(245, min(340, tx - 255))
    else:
        badge_side = 1
        rx_center = min(745, max(640, tx + 255))
    ry_center = max(445, min(950, ty - 70))

    # Deep feathered studio backdrop behind floating giant typography (zero box!)
    vign = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    vd = ImageDraw.Draw(vign)
    vd.ellipse([50, 1080, 940, 1365], fill=(2, 5, 10, int(235 * a)))
    vd.ellipse([rx_center - 165, ry_center - 95, rx_center + 165, ry_center + 95], fill=(2, 5, 10, int(225 * a)))
    vign = vign.filter(ImageFilter.GaussianBlur(radius=36))
    canvas_rgba.alpha_composite(vign)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # --- 1. ON-BODY 3D RADIAL ARC GAUGE AROUND TARGET JOINT ---
    arc_r = 78
    start_deg = 135
    span_deg = 270
    disp_pct = int(round(metric_pct * sweep))
    end_active_deg = start_deg + int(round(span_deg * (disp_pct / 100.0)))

    od.arc(
        [tx - arc_r, ty - arc_r, tx + arc_r, ty + arc_r],
        start=start_deg,
        end=start_deg + span_deg,
        fill=(255, 255, 255, int(70 * a)),
        width=6,
    )
    if end_active_deg > start_deg:
        gd.arc(
            [tx - arc_r, ty - arc_r, tx + arc_r, ty + arc_r],
            start=start_deg,
            end=end_active_deg,
            fill=(*accent, int(240 * a)),
            width=16,
        )
        od.arc(
            [tx - arc_r, ty - arc_r, tx + arc_r, ty + arc_r],
            start=start_deg,
            end=end_active_deg,
            fill=(*accent, int(255 * a)),
            width=8,
        )
        od.arc(
            [tx - arc_r, ty - arc_r, tx + arc_r, ty + arc_r],
            start=start_deg,
            end=end_active_deg,
            fill=(255, 255, 255, int(230 * a)),
            width=3,
        )

    n_ticks = 20
    active_ticks = int(round((disp_pct / 100.0) * n_ticks))
    for i in range(n_ticks):
        ang = math.radians(start_deg + i * (span_deg / float(n_ticks - 1)))
        r0 = arc_r + 11
        r1 = arc_r + (22 if i % 4 == 0 else 15)
        p0 = (tx + int(r0 * math.cos(ang)), ty + int(r0 * math.sin(ang)))
        p1 = (tx + int(r1 * math.cos(ang)), ty + int(r1 * math.sin(ang)))
        if i <= active_ticks:
            od.line([p0, p1], fill=(*accent, int(255 * a)), width=4 if i % 4 == 0 else 3)
        else:
            od.line([p0, p1], fill=(255, 255, 255, int(70 * a)), width=2)

    # Smart 45° Holographic Arm (`o──╱───`) to the Giant 98px % Readout
    pct_str = f"{disp_pct}%"
    short_tag = shorten_muscle_tag(anatomy_tag, metric_label)

    anchor_ang = math.radians(210 if badge_side < 0 else -30)
    ring_edge_pt = (tx + int((arc_r + 18) * math.cos(anchor_ang)), ty + int((arc_r + 18) * math.sin(anchor_ang)))
    arm_end = (rx_center + (95 if badge_side < 0 else -95), ry_center + 8)
    elbow_arm = (int(lerp(ring_edge_pt[0], arm_end[0], 0.5)), ry_center + 8)

    gd.line([ring_edge_pt, elbow_arm, arm_end], fill=(*accent, int(200 * a)), width=9, joint="curve")
    od.line([ring_edge_pt, elbow_arm, arm_end], fill=(*accent, int(255 * a)), width=4, joint="curve")
    od.ellipse(
        [ring_edge_pt[0] - 6, ring_edge_pt[1] - 6, ring_edge_pt[0] + 6, ring_edge_pt[1] + 6],
        fill=(255, 255, 255, int(255 * a)),
        outline=(*accent, int(255 * a)),
        width=3,
    )

    # Giant 98px % Number + Bold 44px Muscle Tag
    draw_centered_tracked_text(
        gd,
        center_x=rx_center,
        y=ry_center - 68,
        text=pct_str,
        font=FONT_RING_PCT,
        fill=(*accent, int(230 * a)),
        tracking=1.0,
        shadow=False,
    )
    draw_centered_tracked_text(
        od,
        center_x=rx_center,
        y=ry_center - 68,
        text=pct_str,
        font=FONT_RING_PCT,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.0,
        shadow=True,
        stroke_w=6,
    )
    draw_centered_tracked_text(
        od,
        center_x=rx_center,
        y=ry_center + 28,
        text=short_tag,
        font=FONT_RING_SUB,
        fill=(*accent, int(255 * a)),
        tracking=1.2,
        shadow=True,
        stroke_w=5,
    )

    # --- 2. FADING CURVED LIGHT-THREAD + ANIMATED PHOTON PULSE ---
    thread_start = (tx, min(ty + arc_r + 14, 1085))
    thread_end = (cx, 1118)
    ctrl = (int(lerp(tx, cx, 0.35)), int(lerp(thread_start[1], thread_end[1], 0.65)))
    n_seg = 28
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
        seg_a = int(min(255, (205 * (1.0 - 0.35 * t0) + 50 * pulse_boost)) * a)
        gd.line([(x0_b, y0_b), (x1_b, y1_b)], fill=(*accent, int(seg_a * 0.8)), width=9 if pulse_boost > 0.5 else 6)
        od.line([(x0_b, y0_b), (x1_b, y1_b)], fill=(*accent, seg_a), width=4)

    # --- 3. GIANT 2-LINE STACKED HERO CALLOUT (y = 1125..1315, ZERO SMALL TEXT) ---
    line1, line2 = split_hero_into_two_lines(title_text)
    w1 = measure_tracked_text(od, line1, FONT_HERO_2LINE, tracking=1.5)
    w2 = measure_tracked_text(od, line2, FONT_HERO_2LINE, tracking=1.5) if line2 else 0.0
    max_text_w = max(w1, w2)

    # Place a Giant 74px Glowing ✕ / ✓ Disc directly to the left of the 2-line stacked Hero Title!
    disc_r = 37
    gap_disc = 28
    block_w = (disc_r * 2) + gap_disc + max_text_w
    block_start_x = max(96.0, cx - block_w / 2.0)

    ic_cx = int(block_start_x + disc_r)
    ic_cy = 1222 if line2 else 1185

    gd.ellipse([ic_cx - disc_r - 8, ic_cy - disc_r - 8, ic_cx + disc_r + 8, ic_cy + disc_r + 8], fill=(*accent, int(225 * a)))
    od.ellipse(
        [ic_cx - disc_r, ic_cy - disc_r, ic_cx + disc_r, ic_cy + disc_r],
        fill=(*accent, int(255 * a)),
        outline=(255, 255, 255, int(245 * a)),
        width=3,
    )
    ic_col = (8, 12, 18, int(255 * a)) if is_correct else (255, 255, 255, int(255 * a))
    if is_correct:
        od.line([(ic_cx - 16, ic_cy + 2), (ic_cx - 4, ic_cy + 15), (ic_cx + 18, ic_cy - 13)], fill=ic_col, width=8, joint="curve")
    else:
        d = 14
        od.line([(ic_cx - d, ic_cy - d), (ic_cx + d, ic_cy + d)], fill=ic_col, width=8)
        od.line([(ic_cx - d, ic_cy + d), (ic_cx + d, ic_cy - d)], fill=ic_col, width=8)

    text_left_x = block_start_x + disc_r * 2 + gap_disc
    y1_line = 1132
    y2_line = 1218

    # Line 1 in Pure White (78px Pretendard-Black)
    draw_tracked_text(
        gd,
        (text_left_x, y1_line),
        line1,
        FONT_HERO_2LINE,
        fill=(*accent, int(130 * a)),
        tracking=1.5,
        shadow=False,
    )
    draw_tracked_text(
        od,
        (text_left_x, y1_line),
        line1,
        FONT_HERO_2LINE,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.5,
        shadow=True,
        stroke_w=7,
    )

    # Line 2 in Vivid Neon Accent (78px Pretendard-Black) so the 2-line stack looks like a Nike/Apple poster!
    if line2:
        draw_tracked_text(
            gd,
            (text_left_x, y2_line),
            line2,
            FONT_HERO_2LINE,
            fill=(*accent, int(165 * a)),
            tracking=1.5,
            shadow=False,
        )
        draw_tracked_text(
            od,
            (text_left_x, y2_line),
            line2,
            FONT_HERO_2LINE,
            fill=(*accent, int(255 * a)),
            tracking=1.5,
            shadow=True,
            stroke_w=7,
        )

    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)
