#!/usr/bin/env python3
"""
@gymform.3d — v9 Master Template:
1. 100% Instagram Reels Safe-Zone Compliant:
   - Horizontal Safe Zone: x in [92 .. 888] (leaves 92px left margin for 19.5:9 phone edge crop, and 192px right margin so Instagram's Like/Comment/Share/Save icon column at x=915..1080 NEVER touches our UI).
   - Vertical Safe Zone: y in [246 .. 1388] (leaves 246px top margin below Instagram's Reels/Camera header, and 532px bottom margin above Instagram's @username, Follow button, caption, and audio marquee; also 100% inside the 4:5 profile grid crop y=240..1680).
2. Unified Single-Focus Visual Flow ("시선이 분리되지 않는 원-포커스 고급 레이아웃"):
   - Top Minimal Floating Pill (y = 248..322, height 74px): Combines the ✕/✓ status icon + Exercise Title + 3-Step progress dots in ONE slim smoked-glass pill right above the figure. No separate top-right badge.
   - Center 3D Anatomy + 1 Posture Line + 1 Bold 3D Arrow (y = 360..1130).
   - Proximity Floating Glass Scanner HUD (y = 1156..1376, width 788px, x = 96..884): Attached directly beneath the active muscle/joint via a short glowing pointer stem so the 3D muscle, arrow, and cue title read in ONE single eye fixation!
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
RAJDHANI_SEMIBOLD = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-SemiBold.ttf"
MONTSERRAT_VAR = "/usr/share/fonts/truetype/sand-box/google/Montserrat/Montserrat-VariableFont_wght.ttf"
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(path: str, size: int):
    for p in [path, PRETENDARD_BLACK, PRETENDARD_EXTRABOLD, FALLBACK_BOLD]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Refined Luxury Typography Hierarchy (v9)
FONT_PILL_TITLE = load_font(PRETENDARD_EXTRABOLD, 27)
FONT_PILL_STEP = load_font(RAJDHANI_BOLD, 25)

FONT_HUD_BADGE = load_font(PRETENDARD_EXTRABOLD, 23)
FONT_HUD_ANATOMY = load_font(RAJDHANI_BOLD, 24)
FONT_HUD_HERO = load_font(PRETENDARD_BLACK, 50)
FONT_HUD_METRIC_LBL = load_font(PRETENDARD_EXTRABOLD, 27)
FONT_HUD_METRIC_PCT = load_font(PRETENDARD_BLACK, 40)

FONT_SHEET_TITLE = load_font(PRETENDARD_BLACK, 30)
FONT_IG_UI = load_font(PRETENDARD_EXTRABOLD, 26)
FONT_IG_SMALL = load_font(PRETENDARD_EXTRABOLD, 22)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


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
            draw.text((int(round(x)) + 2, int(round(y)) + 2), ch, font=font, fill=(0, 0, 0, 180))
        draw.text((int(round(x)), int(round(y))), ch, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), ch, font=font)
        adv = draw.textlength(ch, font=font) if hasattr(draw, "textlength") else (bbox[2] - bbox[0])
        x += adv + tracking


def measure_tracked_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, tracking: float = 0.0) -> float:
    if not text:
        return 0.0
    total = 0.0
    for i, ch in enumerate(text):
        adv = draw.textlength(ch, font=font) if hasattr(draw, "textlength") else (draw.textbbox((0, 0), ch, font=font)[2])
        total += adv + (tracking if i < len(text) - 1 else 0.0)
    return total


def draw_v9_unified_top_pill(
    canvas_rgba: Image.Image,
    exercise_name: str,
    active_idx: int,
    is_correct: bool,
):
    """
    v9 Unified Top Floating Pill:
    - Positioned at x = 96..884 (width 788px, centered at x=490 so it clears the right-hand IG icon column)
    - Vertically at y = 248..322 (height 74px), well below Instagram's top camera/Reels bar (y=0..220).
    - Integrates:
      [ ✕ / ✓ Neon Status Circle ]  +  "CABLE ROPE PUSHDOWN"  +  [ STEP 1/3  ● ○ ○ ]
      into ONE sleek, low-profile VisionOS smoked-glass capsule so the viewer's eye isn't split across top corners.
    """
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    x0, y0 = 96, 246
    x1, y1 = 884, 322
    h = y1 - y0
    cy = (y0 + y1) // 2

    sh = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([x0 + 3, y0 + 8, x1 + 3, y1 + 10], radius=h // 2, fill=(0, 0, 0, 175))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=12))
    canvas_rgba.alpha_composite(sh)

    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # Frosted dark glass capsule with subtle top specular rim
    od.rounded_rectangle([x0, y0, x1, y1], radius=h // 2, fill=(10, 14, 22, 215), outline=(255, 255, 255, 45), width=2)
    od.line([(x0 + 38, y0 + 2), (x1 - 38, y0 + 2)], fill=(255, 255, 255, 105), width=2)

    # 1. Left Integrated ✕ / ✓ Status Disc inside the capsule
    ic_cx = x0 + 38
    ic_r = 25
    od.ellipse([ic_cx - ic_r, cy - ic_r, ic_cx + ic_r, cy + ic_r], fill=(*accent, 255))
    if is_correct:
        pts = [(ic_cx - 11, cy + 1), (ic_cx - 3, cy + 9), (ic_cx + 12, cy - 8)]
        od.line(pts, fill=(8, 12, 18, 255), width=5, joint="curve")
    else:
        d = 9
        od.line([(ic_cx - d, cy - d), (ic_cx + d, cy + d)], fill=(255, 255, 255, 255), width=5)
        od.line([(ic_cx - d, cy + d), (ic_cx + d, cy - d)], fill=(255, 255, 255, 255), width=5)

    # 2. Exercise Title (clean, understated luxury typography)
    draw_tracked_text(
        od,
        (x0 + 80, y0 + 21),
        exercise_name,
        FONT_PILL_TITLE,
        fill=(245, 248, 255, 255),
        tracking=1.4,
        shadow=True,
    )

    # 3. Right Integrated Step Counter + 3 Glowing Progress Pills
    step_lbl = f"0{active_idx + 1}/03"
    lbl_w = measure_tracked_text(od, step_lbl, FONT_PILL_STEP, tracking=1.0)
    dots_w = 3 * 22 + 2 * 7
    right_start = x1 - 26 - dots_w
    draw_tracked_text(
        od,
        (right_start - lbl_w - 14, y0 + 22),
        step_lbl,
        FONT_PILL_STEP,
        fill=(*accent, 255),
        tracking=1.0,
        shadow=True,
    )
    for i in range(3):
        dx0 = right_start + i * (22 + 7)
        dx1 = dx0 + 22
        if i == active_idx:
            od.rounded_rectangle([dx0, cy - 5, dx1, cy + 5], radius=5, fill=(*accent, 255))
        elif i < active_idx:
            od.rounded_rectangle([dx0, cy - 5, dx1, cy + 5], radius=5, fill=(200, 215, 235, 195))
        else:
            od.rounded_rectangle([dx0, cy - 5, dx1, cy + 5], radius=5, fill=(52, 64, 82, 200))

    canvas_rgba.alpha_composite(ov)


def draw_v9_proximity_scanner_hud(
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
    v9 Luxury Proximity Scanner HUD:
    - Positioned at x = 96..884 (width = 788px, center x = 490) and y = 1158..1374 (height = 216px, center y = 1266).
    - Why this position is optimal:
      1) Sits directly beneath the 3D mannequin's active muscle/joint (y ~ 720..1020), so the leader stem is short and direct.
         The viewer sees the 3D muscle, the 3D arrow, and the diagnostic text in a SINGLE central gaze!
      2) Bottom edge is y = 1374 — leaving a massive 546px buffer at the bottom for Instagram's username, Follow button,
         caption, and audio ticker, and 100% inside the 4:5 feed crop (y <= 1635)!
      3) Right edge is x = 884 — leaving a full 196px buffer on the right for Instagram's Like/Comment/Share/Save buttons!
    """
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    accent = (40, 250, 125) if is_correct else (255, 58, 48)

    x0, y0 = 96, 1158
    x1, y1 = 884, 1374
    cw, ch = x1 - x0, y1 - y0

    tx, ty = int(target_pt[0]), int(target_pt[1])
    anchor_y = y0
    anchor_x = max(x0 + 110, min(x1 - 110, tx + 24))
    elbow_x = int(lerp(tx, anchor_x, 0.58))
    elbow_y = int(lerp(ty, anchor_y, 0.45))

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # Soft luxury elevation shadow
    sh = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([x0 + 4, y0 + 12, x1 + 4, y1 + 14], radius=30, fill=(0, 0, 0, int(205 * a)))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=16))
    canvas_rgba.alpha_composite(sh)

    # Short, direct technical pointer stem from anatomy to card top edge
    leader_pts = [(tx, ty), (elbow_x, elbow_y), (anchor_x, anchor_y)]
    od.line(leader_pts, fill=(5, 8, 14, int(240 * a)), width=10, joint="curve")
    gd.line(leader_pts, fill=(*accent, int(205 * a)), width=14, joint="curve")
    od.line(leader_pts, fill=(*accent, int(255 * a)), width=5, joint="curve")
    od.line(leader_pts, fill=(255, 255, 255, int(245 * a)), width=2, joint="curve")

    od.ellipse([anchor_x - 8, anchor_y - 8, anchor_x + 8, anchor_y + 8], fill=(255, 255, 255, int(255 * a)), outline=(*accent, int(255 * a)), width=3)
    gd.ellipse([tx - 18, ty - 18, tx + 18, ty + 18], fill=(*accent, int(215 * a)))
    od.ellipse([tx - 10, ty - 10, tx + 10, ty + 10], fill=(255, 255, 255, int(255 * a)), outline=(*accent, int(255 * a)), width=4)

    # Subtle ambient rim glow around card
    gd.rounded_rectangle([x0 - 5, y0 - 5, x1 + 5, y1 + 5], radius=33, fill=(*accent, int(95 * a)))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=12))
    canvas_rgba.alpha_composite(glow)

    # Smoked dark glass card body + refined 3px border + inner specular rim
    od.rounded_rectangle([x0, y0, x1, y1], radius=28, fill=(8, 12, 20, int(236 * a)), outline=(*accent, int(235 * a)), width=3)
    od.rounded_rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], radius=25, outline=(255, 255, 255, int(28 * a)), width=1)
    od.line([(x0 + 36, y0 + 2), (x1 - 36, y0 + 2)], fill=(255, 255, 255, int(135 * a)), width=2)

    pad_x = 30
    # --- ROW 1: Compact Status Pill + Right Anatomical Target Label ---
    clean_badge = badge_text.replace("✕ ", "").replace("✓ ", "").strip()
    pill_w = int(measure_tracked_text(od, clean_badge, FONT_HUD_BADGE, tracking=1.2)) + 62
    pill_h = 38
    px0, py0 = x0 + pad_x, y0 + 18
    od.rounded_rectangle([px0, py0, px0 + pill_w, py0 + pill_h], radius=11, fill=(*accent, int(250 * a)))
    pill_txt_col = (8, 12, 18, int(255 * a)) if is_correct else (255, 255, 255, int(255 * a))

    ic_cx, ic_cy = px0 + 22, py0 + pill_h // 2
    if is_correct:
        od.line([(ic_cx - 8, ic_cy + 1), (ic_cx - 2, ic_cy + 7), (ic_cx + 9, ic_cy - 6)], fill=pill_txt_col, width=4, joint="curve")
    else:
        od.line([(ic_cx - 6, ic_cy - 6), (ic_cx + 6, ic_cy + 6)], fill=pill_txt_col, width=4)
        od.line([(ic_cx - 6, ic_cy + 6), (ic_cx + 6, ic_cy - 6)], fill=pill_txt_col, width=4)
    draw_tracked_text(od, (px0 + 40, py0 + 6), clean_badge, FONT_HUD_BADGE, fill=pill_txt_col, tracking=1.2, shadow=False)

    anat_w = measure_tracked_text(od, anatomy_tag, FONT_HUD_ANATOMY, tracking=1.5)
    draw_tracked_text(
        od,
        (x1 - pad_x - anat_w, py0 + 7),
        anatomy_tag,
        FONT_HUD_ANATOMY,
        fill=(185, 202, 225, int(240 * a)),
        tracking=1.5,
        shadow=True,
    )

    # --- ROW 2: Crisp 50px Hero Cue Title (Left) + Large 40px Percentage Readout (Right) ---
    draw_tracked_text(
        od,
        (x0 + pad_x, y0 + 68),
        title_text,
        FONT_HUD_HERO,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.1,
        shadow=True,
    )

    # --- ROW 3: Integrated Luxury Telemetry Bar + Metric Label + Percentage ---
    lbl_y = y0 + 138
    draw_tracked_text(
        od,
        (x0 + pad_x, lbl_y),
        metric_label,
        FONT_HUD_METRIC_LBL,
        fill=(210, 224, 242, int(250 * a)),
        tracking=0.6,
        shadow=True,
    )

    pct_str = f"{metric_pct}%"
    pct_w = measure_tracked_text(od, pct_str, FONT_HUD_METRIC_PCT, tracking=0.8)
    draw_tracked_text(
        od,
        (x1 - pad_x - pct_w, y0 + 126),
        pct_str,
        FONT_HUD_METRIC_PCT,
        fill=(*accent, int(255 * a)),
        tracking=0.8,
        shadow=True,
    )

    # Sleek 20-Segment Telemetry Bar at y0 + 176 (height 20px)
    bar_x0 = x0 + pad_x
    bar_x1 = x1 - pad_x
    bar_y0 = y0 + 176
    bar_h = 20
    n_segs = 20
    gap = 6
    total_w = bar_x1 - bar_x0
    seg_w = (total_w - gap * (n_segs - 1)) / float(n_segs)
    active_segs = int(round((metric_pct / 100.0) * n_segs * a))

    for i in range(n_segs):
        sx0 = int(round(bar_x0 + i * (seg_w + gap)))
        sx1 = int(round(sx0 + seg_w))
        if i < active_segs:
            od.rounded_rectangle([sx0, bar_y0, sx1, bar_y0 + bar_h], radius=5, fill=(*accent, int(250 * a)))
            od.line([(sx0 + 2, bar_y0 + 2), (sx1 - 2, bar_y0 + 2)], fill=(255, 255, 255, int(160 * a)), width=2)
        else:
            od.rounded_rectangle([sx0, bar_y0, sx1, bar_y0 + bar_h], radius=5, fill=(28, 36, 50, int(215 * a)), outline=(255, 255, 255, int(24 * a)), width=1)

    canvas_rgba.alpha_composite(ov)


def overlay_ig_reels_ui_simulator(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Draws the actual Instagram Reels mobile UI overlay (top bar, right icon column, bottom username/caption/audio bar,
    and dashed cyan Safe-Zone boundaries) on top of a rendered frame to visually prove 0% clipping and 0% overlap!
    """
    canvas = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # 1. Shaded danger zones (Top 220px, Bottom 480px, Right 165px, Left 60px)
    od.rectangle([0, 0, W, 220], fill=(0, 0, 0, 115))
    od.rectangle([0, 1440, W, H], fill=(0, 0, 0, 135))
    od.rectangle([915, 220, W, 1440], fill=(0, 0, 0, 95))

    # Safe zone border (Cyan rectangle x=90..890, y=240..1390)
    od.rounded_rectangle([88, 238, 892, 1384], radius=24, outline=(0, 230, 255, 220), width=3)
    od.rounded_rectangle([96, 200, 520, 234], radius=8, fill=(0, 190, 220, 230))
    od.text((108, 204), "IG REELS 100% SAFE ZONE", font=FONT_IG_SMALL, fill=(8, 14, 22, 255))

    # 2. Fake Instagram Top Bar (Reels / Camera icon)
    od.text((64, 110), "<  Reels", font=FONT_IG_UI, fill=(255, 255, 255, 210))
    od.rounded_rectangle([960, 105, 1020, 150], radius=10, outline=(255, 255, 255, 200), width=3)

    # 3. Fake Instagram Right-Side Action Column (Heart, Comment, Share, Save, Audio Disc at x=985)
    for idx_btn, (by, lbl) in enumerate([(960, "24.8K"), (1100, "412"), (1240, "8.9K"), (1380, "Save")]):
        od.ellipse([955, by - 32, 1019, by + 32], fill=(255, 255, 255, 45), outline=(255, 255, 255, 195), width=3)
        od.text((955, by + 40), lbl, font=FONT_IG_SMALL, fill=(255, 255, 255, 210))

    # 4. Fake Instagram Bottom Username + Follow + Caption + Audio Bar (y = 1480..1840)
    od.ellipse([64, 1500, 128, 1564], fill=(40, 250, 125, 220), outline=(255, 255, 255, 240), width=3)
    od.text((146, 1516), "gymform.3d", font=FONT_IG_UI, fill=(255, 255, 255, 240))
    od.rounded_rectangle([340, 1510, 470, 1556], radius=10, outline=(255, 255, 255, 220), width=2)
    od.text((366, 1519), "Follow", font=FONT_IG_SMALL, fill=(255, 255, 255, 240))

    od.text((64, 1590), "Stop letting your shoulders steal your triceps gains...", font=FONT_IG_SMALL, fill=(235, 240, 248, 220))
    od.text((64, 1630), "#tricepsworkout #gymform #biomechanics ... more", font=FONT_IG_SMALL, fill=(185, 195, 210, 200))
    od.rounded_rectangle([64, 1690, 520, 1740], radius=24, fill=(255, 255, 255, 38))
    od.text((88, 1702), "♫  gymform.3d · Original Audio", font=FONT_IG_SMALL, fill=(255, 255, 255, 220))

    canvas.alpha_composite(ov)
    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
