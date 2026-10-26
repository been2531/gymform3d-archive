#!/usr/bin/env python3
"""
@gymform.3d — Day 1: Cable Rope Triceps Pushdown (v7 — Luxury Athletic HUD & High-Legibility Typography)
Upgrades:
1. Top Luxury Header Bar:
   - Aligned horizontally with the top-right ✕/✓ circular badge (cy = 122).
   - Displays "STEP 01/03 · TORSO & SCAPULA" + 3 segmented progress bars on the overline,
     and "CABLE ROPE PUSHDOWN" in crisp luxury display typography (Pretendard-Black / BarlowSemiCondensed).
2. On-Body Visuals:
   - Single clean posture line + 1 bold 3D directional arrow + 1 target joint lock-on ring.
3. Bottom Anatomical Target & Force-Leak Scanner Card (@gymform.3d Signature):
   - Wide, balanced dark-glass card (948 x 276) with specular top border, glowing leader line from the exact joint/muscle,
     huge 58px hero cue title ("SHOULDER TAKEOVER", "LOCK ELBOW AT SIDE", "SPREAD ROPE OUT"),
     and a 20-segment glowing EMG / Joint-Load telemetry bar with a large 46px percentage readout.
"""
import os
import math
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)

PRETENDARD_BLACK = "/home/box/.fonts/Pretendard-Black.otf"
PRETENDARD_EXTRABOLD = "/home/box/.fonts/Pretendard-ExtraBold.otf"
BARLOW_BLACK = "/usr/share/fonts/truetype/sand-box/google/Barlow Semi Condensed/BarlowSemiCondensed-Black.ttf"
BARLOW_EXTRABOLD = "/usr/share/fonts/truetype/sand-box/google/Barlow Semi Condensed/BarlowSemiCondensed-ExtraBold.ttf"
BARLOW_BOLD = "/usr/share/fonts/truetype/sand-box/google/Barlow Semi Condensed/BarlowSemiCondensed-Bold.ttf"
RAJDHANI_BOLD = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Bold.ttf"
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(path: str, size: int):
    for p in [path, PRETENDARD_BLACK, PRETENDARD_EXTRABOLD, BARLOW_EXTRABOLD, FALLBACK_BOLD]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Carefully tuned luxury typography hierarchy
FONT_HEADER_STEP = load_font(RAJDHANI_BOLD, 25)
FONT_HEADER_TITLE = load_font(PRETENDARD_BLACK, 38)

FONT_CARD_PILL = load_font(PRETENDARD_EXTRABOLD, 27)
FONT_CARD_ANATOMY = load_font(RAJDHANI_BOLD, 26)
FONT_CARD_HERO = load_font(PRETENDARD_BLACK, 58)
FONT_CARD_METRIC_LBL = load_font(PRETENDARD_EXTRABOLD, 32)
FONT_CARD_METRIC_PCT = load_font(PRETENDARD_BLACK, 46)

FONT_SHEET_HEADER = load_font(PRETENDARD_BLACK, 32)


def draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    pos: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    tracking: float = 0.0,
    shadow: bool = False,
):
    """Draws text with custom letter-spacing (tracking) and optional soft crisp drop-shadow."""
    x, y = float(pos[0]), float(pos[1])
    for ch in text:
        if shadow:
            draw.text((int(round(x)) + 2, int(round(y)) + 3), ch, font=font, fill=(0, 0, 0, 185))
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


WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")


def load_i2v_frames(mp4_path: str, fallback_jpg: str) -> list:
    frames = []
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            up = cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4)
            blur = cv2.GaussianBlur(up, (0, 0), 1.05)
            sharp = cv2.addWeighted(up, 1.22, blur, -0.22, 0)
            frames.append(sharp)
        cap.release()
    if not frames and os.path.exists(fallback_jpg):
        im = cv2.imread(fallback_jpg)
        up = cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4)
        frames = [up] * 60
    return frames


WRONG_FRAMES = load_i2v_frames(WRONG_I2V, WRONG_JPG)
CORRECT_FRAMES = load_i2v_frames(CORRECT_I2V, CORRECT_JPG)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def enhance_plate(frame_bgr: np.ndarray, is_wrong: bool, boost_level: float = 0.85) -> np.ndarray:
    """Clean studio contrast + realistic glowing anatomical highlight on active muscle/joint."""
    out = frame_bgr.astype(np.float32) * 1.04
    h, w, _ = out.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    if is_wrong:
        g1 = np.exp(-((xx - 785.0) ** 2 / (2 * 82.0 ** 2) + (yy - 690.0) ** 2 / (2 * 82.0 ** 2)))
        g2 = np.exp(-((xx - 555.0) ** 2 / (2 * 70.0 ** 2) + (yy - 975.0) ** 2 / (2 * 70.0 ** 2)))
        glow = np.clip(g1 + 0.85 * g2, 0.0, 1.0) * (0.42 * boost_level)
        out[:, :, 2] += glow * 210.0
        out[:, :, 1] += glow * 35.0
        out[:, :, 0] += glow * 20.0
    else:
        g1 = np.exp(-((xx - 745.0) ** 2 / (2 * 68.0 ** 2) + (yy - 735.0) ** 2 / (2 * 98.0 ** 2)))
        g2 = np.exp(-((xx - 525.0) ** 2 / (2 * 64.0 ** 2) + (yy - 965.0) ** 2 / (2 * 64.0 ** 2)))
        glow = np.clip(g1 + 0.65 * g2, 0.0, 1.0) * (0.45 * boost_level)
        out[:, :, 2] += glow * 195.0
        out[:, :, 1] += glow * 95.0
        out[:, :, 0] += glow * 18.0

    # Subtle bottom vignette so the bottom scanner card pops with crystal contrast
    vignette_y = np.clip((yy - h * 0.62) / (h * 0.38), 0.0, 1.0) ** 1.35
    out *= (1.0 - 0.52 * vignette_y[:, :, None])
    # Subtle top vignette for header legibility
    top_vig = np.clip((h * 0.16 - yy) / (h * 0.16), 0.0, 1.0) ** 1.4
    out *= (1.0 - 0.38 * top_vig[:, :, None])

    return np.clip(out, 0, 255).astype(np.uint8)


def crop_camera_view(frame_bgr: np.ndarray, cx: float, cy: float, zoom: float):
    """Crops a (W/zoom, H/zoom) window centered at (cx, cy) and resizes to (1080, 1920)."""
    crop_w = W / zoom
    crop_h = H / zoom
    x0 = max(0.0, min(float(W) - crop_w, cx - crop_w / 2.0))
    y0 = max(0.0, min(float(H) - crop_h, cy - crop_h / 2.0))
    ix0, iy0 = int(round(x0)), int(round(y0))
    ix1, iy1 = min(W, int(round(x0 + crop_w))), min(H, int(round(y0 + crop_h)))
    cropped = frame_bgr[iy0:iy1, ix0:ix1]
    resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    def transform_pt(pt):
        px, py = pt
        return ((px - x0) * zoom, (py - y0) * zoom)

    return resized, transform_pt


def draw_single_posture_line(
    canvas_rgba: Image.Image,
    pts: list,
    col_rgb: tuple,
    width: int = 13,
    prog: float = 1.0,
):
    """Draws 1 clean, high-contrast posture/alignment line directly on the body."""
    if len(pts) < 2 or prog <= 0.05:
        return
    a = min(1.0, prog)
    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line(pts, fill=(0, 0, 0, int(220 * a)), width=width + 14, joint="curve")
    gd.line(pts, fill=(*col_rgb, int(220 * a)), width=width + 10, joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(glow)

    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    cd.line(pts, fill=(8, 12, 18, int(240 * a)), width=width + 6, joint="curve")
    cd.line(pts, fill=(*col_rgb, int(255 * a)), width=width, joint="curve")
    cd.line(pts, fill=(255, 255, 255, int(245 * a)), width=max(3, width // 3), joint="curve")
    for p in pts:
        px, py = int(p[0]), int(p[1])
        r = width // 2 + 5
        cd.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255, int(255 * a)), outline=(*col_rgb, int(255 * a)), width=4)
    canvas_rgba.alpha_composite(core)


def draw_lock_on_joint_ring(
    canvas_rgba: Image.Image,
    center: tuple,
    is_correct: bool,
    r_outer: int = 54,
    prog: float = 1.0,
):
    """Draws 1 clean anatomical target ring on the key joint/muscle."""
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    col = (40, 250, 125) if is_correct else (255, 58, 48)
    cx, cy = int(center[0]), int(center[1])
    ro = int(r_outer * (0.85 + 0.15 * a))

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([cx - ro - 8, cy - ro - 8, cx + ro + 8, cy + ro + 8], outline=(*col, int(215 * a)), width=12)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    canvas_rgba.alpha_composite(glow)

    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    cd.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], outline=(8, 12, 18, int(230 * a)), width=9)
    cd.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], outline=(*col, int(255 * a)), width=5)
    cd.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(255, 255, 255, int(255 * a)), outline=(*col, int(255 * a)), width=4)
    canvas_rgba.alpha_composite(core)


def draw_bold_3d_arrow(
    canvas_rgba: Image.Image,
    p0: tuple,
    p1: tuple,
    p2: tuple,
    col_rgb: tuple,
    prog: float = 1.0,
    tail_w: float = 18.0,
    neck_w: float = 38.0,
    head_w: float = 82.0,
    head_len: float = 64.0,
):
    """High-contrast 3D curved/linear arrow with dark outline casing + neon bloom + white spine."""
    if prog <= 0.05:
        return
    p = min(1.0, prog)

    def bez(u):
        omt = 1.0 - u
        return (
            omt * omt * p0[0] + 2 * omt * u * p1[0] + u * u * p2[0],
            omt * omt * p0[1] + 2 * omt * u * p1[1] + u * u * p2[1],
        )

    def bez_tan(u):
        dx = 2 * (1.0 - u) * (p1[0] - p0[0]) + 2 * u * (p2[0] - p1[0])
        dy = 2 * (1.0 - u) * (p1[1] - p0[1]) + 2 * u * (p2[1] - p1[1])
        L = math.hypot(dx, dy) + 1e-6
        return (dx / L, dy / L)

    steps = 32
    u_max = p * 0.82
    left_pts, right_pts, spine_pts = [], [], []
    for i in range(steps + 1):
        u = u_max * (i / steps)
        bx, by = bez(u)
        tx, ty = bez_tan(u)
        nx, ny = -ty, tx
        frac = i / steps
        half_w = 0.5 * lerp(tail_w, neck_w, frac ** 0.85)
        left_pts.append((bx + nx * half_w, by + ny * half_w))
        right_pts.append((bx - nx * half_w, by - ny * half_w))
        spine_pts.append((bx, by))

    neck_x, neck_y = bez(u_max)
    tx, ty = bez_tan(u_max)
    nx, ny = -ty, tx
    scale_h = min(1.0, p * 1.25)
    hw = 0.5 * head_w * scale_h
    hl = head_len * scale_h
    back_sweep = 12.0 * scale_h

    tip = (neck_x + tx * hl, neck_y + ty * hl)
    wing_L = (neck_x - tx * back_sweep + nx * hw, neck_y - ty * back_sweep + ny * hw)
    wing_R = (neck_x - tx * back_sweep - nx * hw, neck_y - ty * back_sweep - ny * hw)

    poly = left_pts + [wing_L, tip, wing_R] + right_pts[::-1]
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    pad = 54
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(W, int(max(xs)) + pad)
    y1 = min(H, int(max(ys)) + pad)
    pw, ph = x1 - x0, y1 - y0
    if pw <= 4 or ph <= 4:
        return

    SS = 2
    local_poly = [((px - x0) * SS, (py - y0) * SS) for (px, py) in poly]
    sh_poly = [((px - x0 + 6) * SS, (py - y0 + 9) * SS) for (px, py) in poly]

    sh_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(sh_im).polygon(sh_poly, fill=(0, 0, 0, 215))
    sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=7 * SS))

    casing_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(casing_im).polygon(local_poly, fill=(5, 8, 14, 245), outline=(5, 8, 14, 250), width=6 * SS)

    glow_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(glow_im).polygon(local_poly, fill=(*col_rgb, 210))
    glow_im = glow_im.filter(ImageFilter.GaussianBlur(radius=9 * SS))

    body_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body_im)
    bd.polygon(local_poly, fill=(*col_rgb, 255), outline=(255, 255, 255, 255), width=3 * SS)

    local_spine = [((px - x0) * SS, (py - y0) * SS) for (px, py) in spine_pts]
    local_tip = ((tip[0] - x0) * SS, (tip[1] - y0) * SS)
    if len(local_spine) >= 2:
        bd.line(local_spine + [local_tip], fill=(255, 255, 255, 240), width=max(3, int(neck_w * 0.24 * SS)), joint="curve")

    comp = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    comp.alpha_composite(sh_im)
    comp.alpha_composite(glow_im)
    comp.alpha_composite(casing_im)
    comp.alpha_composite(body_im)
    comp_down = comp.resize((pw, ph), resample=Image.Resampling.LANCZOS)
    canvas_rgba.alpha_composite(comp_down, dest=(x0, y0))


def draw_top_header_and_step_bar(canvas_rgba: Image.Image, active_idx: int, is_correct: bool):
    """
    Replaces the tiny ambiguous top silhouette capsule with a luxury Apple Fitness+ / Whoop style
    Workout Header & 3-Step Progress Bar:
      - Overline: "STEP 01 / 03  ·  TORSO & SCAPULA" + 3 glowing step segments
      - Main Title: "CABLE ROPE PUSHDOWN" in crisp Pretendard-Black 38px with letter-spacing
    """
    step_focus_names = [
        "TORSO & SCAPULA",
        "ELBOW HINGE AXIS",
        "WRIST & LOCKOUT",
    ]
    focus_name = step_focus_names[max(0, min(2, active_idx))]
    accent = (40, 250, 125) if is_correct else (255, 58, 48)

    # Positioned cleanly on top-left/center, vertically aligned with the top-right ✕/✓ badge (cy=124)
    x0, y0 = 48, 56
    x1, y1 = 826, 192

    sh = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([x0 + 4, y0 + 8, x1 + 4, y1 + 10], radius=32, fill=(0, 0, 0, 185))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=14))
    canvas_rgba.alpha_composite(sh)

    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # Dark frosted glass pill with subtle outer border + top specular highlight
    od.rounded_rectangle([x0, y0, x1, y1], radius=30, fill=(10, 14, 22, 232), outline=(255, 255, 255, 42), width=2)
    od.line([(x0 + 32, y0 + 2), (x1 - 32, y0 + 2)], fill=(255, 255, 255, 95), width=2)

    # Left vertical glowing status accent bar inside the header
    bar_x = x0 + 24
    od.rounded_rectangle([bar_x, y0 + 24, bar_x + 8, y1 - 24], radius=4, fill=(*accent, 255))

    text_x = x0 + 50
    # Overline: STEP 01 / 03 + glowing vector dot + FOCUS
    step_part1 = f"STEP 0{active_idx + 1} / 03"
    draw_tracked_text(od, (text_x, y0 + 20), step_part1, FONT_HEADER_STEP, fill=(*accent, 255), tracking=1.8, shadow=True)
    w_p1 = measure_tracked_text(od, step_part1, FONT_HEADER_STEP, tracking=1.8)
    dot_cx = int(round(text_x + w_p1 + 18))
    dot_cy = y0 + 34
    od.ellipse([dot_cx - 4, dot_cy - 4, dot_cx + 4, dot_cy + 4], fill=(*accent, 255))
    draw_tracked_text(od, (dot_cx + 18, y0 + 20), focus_name, FONT_HEADER_STEP, fill=(215, 228, 245, 255), tracking=1.8, shadow=True)

    # 3 Segmented Progress Bars on top-right of the header pill
    seg_w, seg_h, seg_gap = 46, 10, 8
    seg_start_x = x1 - 28 - (3 * seg_w + 2 * seg_gap)
    seg_y = y0 + 29
    for i in range(3):
        sx0 = seg_start_x + i * (seg_w + seg_gap)
        sx1 = sx0 + seg_w
        if i < active_idx:
            # Completed step
            od.rounded_rectangle([sx0, seg_y, sx1, seg_y + seg_h], radius=5, fill=(210, 225, 245, 190))
        elif i == active_idx:
            # Active step (glowing accent)
            od.rounded_rectangle([sx0 - 2, seg_y - 2, sx1 + 2, seg_y + seg_h + 2], radius=6, fill=(*accent, 90))
            od.rounded_rectangle([sx0, seg_y, sx1, seg_y + seg_h], radius=5, fill=(*accent, 255))
        else:
            # Upcoming step
            od.rounded_rectangle([sx0, seg_y, sx1, seg_y + seg_h], radius=5, fill=(45, 56, 72, 210))

    # Main Exercise Title: CABLE ROPE PUSHDOWN
    draw_tracked_text(
        od,
        (text_x, y0 + 62),
        "CABLE ROPE PUSHDOWN",
        FONT_HEADER_TITLE,
        fill=(255, 255, 255, 255),
        tracking=1.5,
        shadow=True,
    )

    canvas_rgba.alpha_composite(ov)


def draw_ox_status_badge(canvas_rgba: Image.Image, is_correct: bool):
    """Upper-right circular ✕ / ✓ neon badge, horizontally aligned with the top workout header."""
    cx, cy = 946, 124
    r = 64
    col = (40, 250, 125) if is_correct else (255, 55, 45)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([cx - r - 12, cy - r - 12, cx + r + 12, cy + r + 12], fill=(*col, 145))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=16))
    canvas_rgba.alpha_composite(glow)

    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(10, 14, 22, 238), outline=(*col, 255), width=6)
    od.ellipse([cx - r + 7, cy - r + 7, cx + r - 7, cy + r - 7], outline=(255, 255, 255, 45), width=2)

    if is_correct:
        pts = [(cx - 28, cy + 2), (cx - 7, cy + 24), (cx + 31, cy - 22)]
        od.line(pts, fill=(*col, 255), width=15, joint="curve")
        od.line(pts, fill=(255, 255, 255, 220), width=5, joint="curve")
    else:
        d = 25
        od.line([(cx - d, cy - d), (cx + d, cy + d)], fill=(*col, 255), width=15)
        od.line([(cx - d, cy + d), (cx + d, cy - d)], fill=(*col, 255), width=15)
        od.line([(cx - d, cy - d), (cx + d, cy + d)], fill=(255, 255, 255, 210), width=4)
        od.line([(cx - d, cy + d), (cx + d, cy - d)], fill=(255, 255, 255, 210), width=4)

    canvas_rgba.alpha_composite(ov)


def draw_anatomical_scanner_callout(
    canvas_rgba: Image.Image,
    target_pt: tuple,
    card_center: tuple,
    is_correct: bool,
    badge_text: str,
    anatomy_tag: str,
    title_text: str,
    metric_label: str,
    metric_pct: int,
    prog: float = 1.0,
):
    """
    @gymform.3d Signature — v7 Luxury High-Legibility Anatomical Target Scanner Card:
      - Wide 960 x 284 dark-glass HUD panel at the bottom with specular glass edge & neon accent border.
      - Connected directly to the active muscle/joint via a sleek 2-segment glowing leader line.
      - Huge 58px Hero Cue Title (e.g. "SHOULDER TAKEOVER" / "LOCK ELBOW AT SIDE" / "SPREAD ROPE OUT")
        that is 100% legible even on small mobile screens.
      - 20-segment glowing telemetry bar + bold 46px percentage readout.
    """
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    bg_col = (8, 12, 20, int(242 * a))

    cw, ch = 964, 286
    cx, cy = int(card_center[0]), int(card_center[1])
    x0, y0 = cx - cw // 2, cy - ch // 2
    x1, y1 = x0 + cw, y0 + ch

    # Leader line from `target_pt` on the body to the top edge of the scanner card
    tx, ty = int(target_pt[0]), int(target_pt[1])
    anchor_y = y0
    anchor_x = max(x0 + 140, min(x1 - 140, tx + 40))
    elbow_x = int(lerp(tx, anchor_x, 0.62))
    elbow_y = int(lerp(ty, anchor_y, 0.42))

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # Drop shadow behind card
    sh = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([x0 + 6, y0 + 14, x1 + 6, y1 + 16], radius=36, fill=(0, 0, 0, int(210 * a)))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=18))
    canvas_rgba.alpha_composite(sh)

    # Leader line with dark casing + neon glow + crisp white core
    leader_pts = [(tx, ty), (elbow_x, elbow_y), (anchor_x, anchor_y)]
    od.line(leader_pts, fill=(5, 8, 14, int(245 * a)), width=12, joint="curve")
    gd.line(leader_pts, fill=(*accent, int(215 * a)), width=16, joint="curve")
    od.line(leader_pts, fill=(*accent, int(255 * a)), width=6, joint="curve")
    od.line(leader_pts, fill=(255, 255, 255, int(245 * a)), width=2, joint="curve")

    # Anchor node on card top border + target node on anatomy
    od.ellipse([anchor_x - 9, anchor_y - 9, anchor_x + 9, anchor_y + 9], fill=(255, 255, 255, int(255 * a)), outline=(*accent, int(255 * a)), width=4)
    gd.ellipse([tx - 22, ty - 22, tx + 22, ty + 22], fill=(*accent, int(220 * a)))
    od.ellipse([tx - 12, ty - 12, tx + 12, ty + 12], fill=(255, 255, 255, int(255 * a)), outline=(*accent, int(255 * a)), width=4)

    # Card outer neon ambient glow
    gd.rounded_rectangle([x0 - 8, y0 - 8, x1 + 8, y1 + 8], radius=38, fill=(*accent, int(115 * a)))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas_rgba.alpha_composite(glow)

    # Card dark glass body + dual rim (outer accent border + inner subtle glass highlight)
    od.rounded_rectangle([x0, y0, x1, y1], radius=34, fill=bg_col, outline=(*accent, int(245 * a)), width=4)
    od.rounded_rectangle([x0 + 6, y0 + 6, x1 - 6, y1 - 6], radius=29, outline=(255, 255, 255, int(32 * a)), width=2)
    od.line([(x0 + 42, y0 + 3), (x1 - 42, y0 + 3)], fill=(255, 255, 255, int(130 * a)), width=2)

    pad_x = 38
    # --- ROW 1: Top-left Status Pill (with custom vector ✕/✓ icon) + Top-right Anatomical Structure Tag ---
    clean_badge = badge_text.replace("✕ ", "").replace("✓ ", "").strip()
    pill_w = int(measure_tracked_text(od, clean_badge, FONT_CARD_PILL, tracking=1.4)) + 74
    pill_h = 46
    px0, py0 = x0 + pad_x, y0 + 24
    od.rounded_rectangle([px0, py0, px0 + pill_w, py0 + pill_h], radius=14, fill=(*accent, int(250 * a)))
    pill_txt_col = (8, 12, 18, int(255 * a)) if is_correct else (255, 255, 255, int(255 * a))
    ic_cx, ic_cy = px0 + 26, py0 + pill_h // 2
    if is_correct:
        od.line([(ic_cx - 9, ic_cy + 1), (ic_cx - 2, ic_cy + 8), (ic_cx + 10, ic_cy - 7)], fill=pill_txt_col, width=4, joint="curve")
    else:
        od.line([(ic_cx - 7, ic_cy - 7), (ic_cx + 7, ic_cy + 7)], fill=pill_txt_col, width=4)
        od.line([(ic_cx - 7, ic_cy + 7), (ic_cx + 7, ic_cy - 7)], fill=pill_txt_col, width=4)
    draw_tracked_text(od, (px0 + 48, py0 + 8), clean_badge, FONT_CARD_PILL, fill=pill_txt_col, tracking=1.4, shadow=False)

    # Top-right anatomical target tag (e.g. "TARGET · TRICEPS LATERAL HEAD")
    anat_w = measure_tracked_text(od, anatomy_tag, FONT_CARD_ANATOMY, tracking=1.6)
    draw_tracked_text(
        od,
        (x1 - pad_x - anat_w, py0 + 10),
        anatomy_tag,
        FONT_CARD_ANATOMY,
        fill=(185, 202, 225, int(240 * a)),
        tracking=1.6,
        shadow=True,
    )

    # --- ROW 2: Huge Hero Biomechanical Cue Title (58px Pretendard-Black) ---
    draw_tracked_text(
        od,
        (x0 + pad_x, y0 + 84),
        title_text,
        FONT_CARD_HERO,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.4,
        shadow=True,
    )

    # Subtle horizontal divider line
    div_y = y0 + 164
    od.line([(x0 + pad_x, div_y), (x1 - pad_x, div_y)], fill=(255, 255, 255, int(34 * a)), width=2)

    # --- ROW 3: Metric Label (left) + Big Percentage (right) + 20-Segment Telemetry Bar ---
    lbl_y = y0 + 178
    draw_tracked_text(
        od,
        (x0 + pad_x, lbl_y),
        metric_label,
        FONT_CARD_METRIC_LBL,
        fill=(225, 234, 248, int(255 * a)),
        tracking=0.6,
        shadow=True,
    )

    pct_str = f"{metric_pct}%"
    pct_w = measure_tracked_text(od, pct_str, FONT_CARD_METRIC_PCT, tracking=1.0)
    draw_tracked_text(
        od,
        (x1 - pad_x - pct_w, y0 + 168),
        pct_str,
        FONT_CARD_METRIC_PCT,
        fill=(*accent, int(255 * a)),
        tracking=1.0,
        shadow=True,
    )

    # 20-Segment Telemetry Gauge Bar at the bottom of the card
    bar_x0 = x0 + pad_x
    bar_x1 = x1 - pad_x
    bar_y0 = y0 + 230
    bar_h = 28
    n_segs = 20
    gap = 7
    total_w = bar_x1 - bar_x0
    seg_w = (total_w - gap * (n_segs - 1)) / float(n_segs)
    active_segs = int(round((metric_pct / 100.0) * n_segs * a))

    for i in range(n_segs):
        sx0 = int(round(bar_x0 + i * (seg_w + gap)))
        sx1 = int(round(sx0 + seg_w))
        if i < active_segs:
            od.rounded_rectangle([sx0, bar_y0, sx1, bar_y0 + bar_h], radius=6, fill=(*accent, int(250 * a)))
            # Inner top highlight on lit segment
            od.line([(sx0 + 3, bar_y0 + 3), (sx1 - 3, bar_y0 + 3)], fill=(255, 255, 255, int(165 * a)), width=2)
        else:
            od.rounded_rectangle([sx0, bar_y0, sx1, bar_y0 + bar_h], radius=6, fill=(26, 34, 48, int(220 * a)), outline=(255, 255, 255, int(28 * a)), width=1)

    canvas_rgba.alpha_composite(ov)




# --- v11.1 PURE SCULPTURAL GRAPHIC MASTER OVERRIDE ---
import build_v11_sculptural_graphic as _v11

def draw_top_header_and_step_bar(canvas_rgba, active_idx: int, is_correct: bool = True):
    _v11.draw_v11_cinema_top(canvas_rgba, "CABLE ROPE PUSHDOWN", active_idx, is_correct)

def draw_ox_status_badge(canvas_rgba, is_correct: bool = True, pulse: float = 0.5):
    pass

def draw_anatomical_scanner_callout(
    canvas_rgba,
    target_pt: tuple,
    card_center: tuple,
    is_correct: bool,
    badge_text: str,
    anatomy_tag: str,
    title_text: str,
    metric_label: str,
    metric_pct: int,
    prog: float = 1.0,
):
    _v11.draw_v11_sculptural_in_scene_graphic(
        canvas_rgba=canvas_rgba,
        target_pt=target_pt,
        is_correct=is_correct,
        badge_text=badge_text,
        anatomy_tag=anatomy_tag,
        title_text=title_text,
        metric_label=metric_label,
        metric_pct=metric_pct,
        prog=prog,
    )


def build_scene_frame(scene_idx: int, is_correct: bool) -> np.ndarray:
    """
    Renders a 1080x1920 frame for the given scene (0, 1, 2) and state (Wrong=False, Correct=True)
    with v7 Luxury Header + 1 Line + 1 Arrow + High-Legibility Scanner Card.
    """
    if scene_idx == 0:
        raw = CORRECT_FRAMES[26] if is_correct else WRONG_FRAMES[28]
        enh = enhance_plate(raw, is_wrong=not is_correct, boost_level=0.9)
        framed_bgr, tf = crop_camera_view(enh, cx=730, cy=820, zoom=1.10)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            hip = tf((895, 995))
            hunched_mid = tf((855, 760))
            shoulder_fwd = tf((775, 600))
            target_sh = tf((795, 675))

            draw_single_posture_line(canvas, [hip, hunched_mid, shoulder_fwd], col_rgb=(255, 55, 45), width=13)
            draw_lock_on_joint_ring(canvas, target_sh, is_correct=False, r_outer=54)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((835, 495)),
                p1=tf((725, 535)),
                p2=tf((665, 650)),
                col_rgb=(255, 55, 45),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_sh,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ FORCE LEAK",
                anatomy_tag="ANTERIOR DELTOID",
                title_text="SHOULDER TAKEOVER",
                metric_label="Front Delt Load",
                metric_pct=68,
            )
        else:
            hip = tf((785, 965))
            mid_back = tf((765, 735))
            shoulder_top = tf((745, 520))
            target_sc = tf((765, 635))

            draw_single_posture_line(canvas, [hip, mid_back, shoulder_top], col_rgb=(40, 250, 125), width=13)
            draw_lock_on_joint_ring(canvas, target_sc, is_correct=True, r_outer=54)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((735, 470)),
                p1=tf((845, 555)),
                p2=tf((835, 720)),
                col_rgb=(40, 250, 125),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_sc,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ POSTURE LOCKED",
                anatomy_tag="SCAPULA & TRICEPS",
                title_text="SCAPULA DEPRESSED",
                metric_label="Triceps Isolation",
                metric_pct=92,
            )

    elif scene_idx == 1:
        raw = CORRECT_FRAMES[34] if is_correct else WRONG_FRAMES[36]
        enh = enhance_plate(raw, is_wrong=not is_correct, boost_level=0.95)
        framed_bgr, tf = crop_camera_view(enh, cx=680, cy=790, zoom=1.28)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            shoulder = tf((810, 680))
            drifted_elbow = tf((745, 910))

            draw_single_posture_line(canvas, [shoulder, drifted_elbow], col_rgb=(255, 55, 45), width=13)
            draw_lock_on_joint_ring(canvas, drifted_elbow, is_correct=False, r_outer=56)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((830, 935)),
                p1=tf((710, 925)),
                p2=tf((595, 845)),
                col_rgb=(255, 55, 45),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=drifted_elbow,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ TENSION LOST",
                anatomy_tag="HUMERUS DRIFT",
                title_text="ELBOW SWINGING",
                metric_label="Triceps Tension",
                metric_pct=32,
            )
        else:
            shoulder = tf((775, 600))
            elbow = tf((735, 815))
            wrist = tf((485, 985))

            draw_single_posture_line(canvas, [shoulder, elbow, wrist], col_rgb=(40, 250, 125), width=13)
            draw_lock_on_joint_ring(canvas, elbow, is_correct=True, r_outer=56)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((455, 760)),
                p1=tf((430, 915)),
                p2=tf((525, 1060)),
                col_rgb=(40, 250, 125),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=elbow,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ PURE HINGE AXIS",
                anatomy_tag="ELBOW JOINT PINNED",
                title_text="LOCK ELBOW AT SIDE",
                metric_label="Triceps Tension",
                metric_pct=96,
            )

    else:
        raw = CORRECT_FRAMES[48] if is_correct else WRONG_FRAMES[48]
        enh = enhance_plate(raw, is_wrong=not is_correct, boost_level=1.0)
        framed_bgr, tf = crop_camera_view(enh, cx=600, cy=930, zoom=1.45)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            elbow = tf((825, 885))
            wrist = tf((535, 1005))
            bent_hand = tf((415, 960))

            draw_single_posture_line(canvas, [elbow, wrist, bent_hand], col_rgb=(255, 55, 45), width=13)
            draw_lock_on_joint_ring(canvas, wrist, is_correct=False, r_outer=56)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((495, 1085)),
                p1=tf((415, 1025)),
                p2=tf((395, 905)),
                col_rgb=(255, 55, 45),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=wrist,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ JOINT STRAIN",
                anatomy_tag="WRIST EXTENSION",
                title_text="WRIST BENT BACK",
                metric_label="Wrist Joint Stress",
                metric_pct=85,
            )
        else:
            shoulder_triceps = tf((740, 745))
            elbow = tf((735, 845))
            wrist = tf((515, 965))

            draw_single_posture_line(canvas, [elbow, wrist], col_rgb=(40, 250, 125), width=13)
            draw_lock_on_joint_ring(canvas, shoulder_triceps, is_correct=True, r_outer=58)

            draw_bold_3d_arrow(
                canvas,
                p0=tf((515, 965)),
                p1=tf((410, 1045)),
                p2=tf((315, 1145)),
                col_rgb=(40, 250, 125),
            )
            draw_bold_3d_arrow(
                canvas,
                p0=tf((515, 965)),
                p1=tf((615, 1065)),
                p2=tf((715, 1175)),
                col_rgb=(40, 250, 125),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=shoulder_triceps,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ PEAK CONTRACTION",
                anatomy_tag="TRICEPS LATERAL HEAD",
                title_text="SPREAD ROPE OUT",
                metric_label="Lateral Head EMG",
                metric_pct=98,
            )

    draw_top_header_and_step_bar(canvas, active_idx=scene_idx, is_correct=is_correct)
    draw_ox_status_badge(canvas, is_correct=is_correct)
    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    scenes_wrong = []
    scenes_correct = []
    for s_idx in range(3):
        w_bgr = build_scene_frame(s_idx, is_correct=False)
        c_bgr = build_scene_frame(s_idx, is_correct=True)
        cv2.imwrite(os.path.join(OUT_DIR, f"day1_scene{s_idx+1}_wrong.jpg"), w_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        cv2.imwrite(os.path.join(OUT_DIR, f"day1_scene{s_idx+1}_correct.jpg"), c_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        scenes_wrong.append(w_bgr)
        scenes_correct.append(c_bgr)

    pw, ph = 540, 960
    pad = 24
    header_h = 86
    sheet_w = pw * 3 + pad * 4
    sheet_h = ph * 2 + pad * 3 + header_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 13, 18))
    sd = ImageDraw.Draw(sheet)

    col_titles = [
        "STEP 1/3  |  TORSO & SCAPULA",
        "STEP 2/3  |  ELBOW HINGE AXIS",
        "STEP 3/3  |  WRIST & ROPE SPREAD",
    ]
    for c_idx, title in enumerate(col_titles):
        cx = pad + c_idx * (pw + pad) + pw // 2
        tb = sd.textbbox((0, 0), title, font=FONT_SHEET_HEADER)
        tw = tb[2] - tb[0]
        sd.text((cx - tw // 2, 26), title, font=FONT_SHEET_HEADER, fill=(235, 242, 252))

    for c_idx in range(3):
        x = pad + c_idx * (pw + pad)
        y_top = header_h
        w_rgb = cv2.cvtColor(cv2.resize(scenes_wrong[c_idx], (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(w_rgb), (x, y_top))
        sd.rounded_rectangle([x - 2, y_top - 2, x + pw + 2, y_top + ph + 2], radius=14, outline=(255, 60, 50), width=3)

        y_bot = header_h + ph + pad
        c_rgb = cv2.cvtColor(cv2.resize(scenes_correct[c_idx], (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(c_rgb), (x, y_bot))
        sd.rounded_rectangle([x - 2, y_bot - 2, x + pw + 2, y_bot + ph + 2], radius=14, outline=(40, 250, 125), width=3)

    sheet_path = os.path.join(OUT_DIR, "day1_3scene_ox_preview_sheet.jpg")
    sheet.save(sheet_path, quality=96)
    print("Saved v7 preview sheet:", sheet_path, sheet.size)


if __name__ == "__main__":
    main()
