#!/usr/bin/env python3
"""
GymForm 3D — v28 Luxury Minimal Biomechanical Master (Concept A & Concept B)
Eliminates ALL tacky/arcade graphics ("촌스러운 그래픽은 버리라니까? 그리고 더 개선시켜봐"):
1. Typography Upgrade:
   - Replaces dated Linux `DejaVuSans` with `Pretendard` (Apple SF Pro / Inter geometry) + `Barlow Condensed` (athletic telemetry numerals).
2. Zero Camcorder/Arcade Clutter:
   - Removes fake "CAM 4 • 1.45x" top status bars.
   - Removes filled neon pie-slice protractors, dashed arcade lines, and thick neon borders.
   - Removes sticker labels and crosshairs from Concept B's optical zoom loupe.
3. Refined Luxury Visual Hierarchy (Apple Fitness+ / Whoop / @demicstory aesthetic):
   - Top: Pure `@demicstory` 3-stage dark-glass dock + 4x-supersampled top-right `✕ / ✓` titanium emblem.
   - Center (Multi-Camera Framing + Surgical Micro-Callout):
     * Concept A: Dynamic multi-camera framing (1.0x Wide -> 1.36x Shoulder/Delt -> 1.18x Mid Hinge -> 1.42x Triceps Lockout)
       with a single 4x-supersampled hairline pointer (`1.5px`) and a sleek frosted-obsidian micro-pill (`180° FULL LOCKOUT · 1s SQUEEZE`).
     * Concept B: Full-body posture paired with a frameless, crosshair-free **4x Optical Glass Loupe (`2.2×`)** connected by a fine tangent line.
   - Bottom (Refined Concept B Radial Dial + Frosted Titanium Capsule):
     * 1px specular titanium rim (no neon border), 4x-supersampled rounded-cap radial gauge,
       crystal-clear 3-level typography, and a clean frameless inline muscle-head & set/rep strip.
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import gymform_engine as eng
import build_v25_fast_preview as v25

W, H = eng.W, eng.H
OUT_DIR = eng.OUT_DIR

# Refined luxury sports-science palette (crimson coral & emerald mint, never harsh MS-Paint neon)
COL_RED = (255, 69, 58)       # Apple iOS/watchOS System Red
COL_GREEN = (48, 209, 88)     # Apple Fitness+ / Whoop Emerald Green
COL_AMBER = (255, 159, 10)    # Apple System Orange
COL_SILVER = (216, 224, 236)
COL_MUTED = (142, 154, 174)

PRETENDARD_BLACK = "/home/box/.fonts/Pretendard-Black.otf"
PRETENDARD_EXTRABOLD = "/home/box/.fonts/Pretendard-ExtraBold.otf"
PRETENDARD_BOLD = "/usr/share/fonts/truetype/sand-box/custom/Pretendard/Pretendard-Bold.otf"
PRETENDARD_SEMIBOLD = "/usr/share/fonts/truetype/sand-box/custom/Pretendard/Pretendard-SemiBold.otf"
PRETENDARD_MEDIUM = "/usr/share/fonts/truetype/sand-box/custom/Pretendard/Pretendard-Medium.otf"
BARLOW_BOLD = "/usr/share/fonts/truetype/sand-box/google/Barlow Condensed/BarlowCondensed-Bold.ttf"
BARLOW_EXTRABOLD = "/usr/share/fonts/truetype/sand-box/google/Barlow Condensed/BarlowCondensed-ExtraBold.ttf"


def f_pretendard(size: int, weight: str = "bold") -> ImageFont.FreeTypeFont:
    mapping = {
        "black": PRETENDARD_BLACK,
        "extrabold": PRETENDARD_EXTRABOLD,
        "bold": PRETENDARD_BOLD,
        "semibold": PRETENDARD_SEMIBOLD,
        "medium": PRETENDARD_MEDIUM,
    }
    p = mapping.get(weight, PRETENDARD_BOLD)
    if os.path.exists(p):
        return ImageFont.truetype(p, size)
    return eng.load_font(size, bold=True)


def f_barlow(size: int, extrabold: bool = True) -> ImageFont.FreeTypeFont:
    p = BARLOW_EXTRABOLD if extrabold else BARLOW_BOLD
    if os.path.exists(p):
        return ImageFont.truetype(p, size)
    return f_pretendard(size, "extrabold")


# Typography Scale
F_HERO = f_pretendard(38, "extrabold")
F_KICKER = f_pretendard(15, "bold")
F_TIP = f_pretendard(19, "semibold")
F_METRIC_LBL = f_pretendard(13, "semibold")
F_METRIC_VAL = f_pretendard(16, "extrabold")

F_DIAL_NUM = f_barlow(52, extrabold=True)
F_DIAL_PCT = f_pretendard(22, "bold")
F_DIAL_SUB = f_pretendard(13, "bold")

F_CALLOUT_DEG = f_barlow(34, extrabold=True)
F_CALLOUT_TIT = f_pretendard(18, "bold")
F_CALLOUT_SUB = f_pretendard(14, "medium")


def apply_cinema_camera(
    rgb_arr: np.ndarray,
    zoom: float = 1.0,
    focus_xy: tuple[float, float] = (680.0, 760.0),
    dest_xy: tuple[float, float] = (680.0, 760.0),
) -> tuple[np.ndarray, callable]:
    """
    Smooth cinema camera framing with subtle studio vignette polish.
    """
    fx, fy = focus_xy
    dx, dy = dest_xy
    M = np.float32([
        [zoom, 0.0, dx - fx * zoom],
        [0.0, zoom, dy - fy * zoom],
    ])
    out = cv2.warpAffine(
        rgb_arr,
        M,
        (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(14, 16, 20),
    )

    def map_pt(x: float, y: float) -> tuple[float, float]:
        return (dx + (x - fx) * zoom, dy + (y - fy) * zoom)

    return out, map_pt


def draw_subtle_onbody_vector(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
):
    """
    Draws ONLY ultra-clean, 4x-supersampled minimal biomechanical vectors on the body
    (NO arcade pie-slices, NO dashed lines, NO clutter).
    """
    ss = 4
    ov = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    def mp(x: float, y: float) -> tuple[int, int]:
        mx, my = map_pt(x, y)
        return int(round(mx * ss)), int(round(my * ss))

    if step_idx == 1:
        # Subtle spine curve arc showing hunched upper back
        col = COL_RED
        pts = [mp(722, 575), mp(776, 650), mp(815, 748), mp(828, 845)]
        d.line(pts, fill=(0, 0, 0, 140), width=5 * ss, joint="curve")
        d.line(pts, fill=(*col, 225), width=2 * ss, joint="curve")
        for pt in [pts[0], pts[2]]:
            r = 5 * ss
            d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=(255, 255, 255, 255), outline=(*col, 255), width=2 * ss)

    elif step_idx == 2:
        # Subtle 90° forearm stop line vs downward extension arrow
        col = COL_RED
        sh = mp(726, 634)
        el = mp(722, 836)
        wr = mp(592, 886)
        d.line([sh, el, wr], fill=(0, 0, 0, 150), width=5 * ss, joint="curve")
        d.line([sh, el, wr], fill=(*col, 230), width=2 * ss, joint="curve")
        for pt in [sh, el, wr]:
            r = 5 * ss
            d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=(255, 255, 255, 255), outline=(*col, 255), width=2 * ss)

    elif step_idx == 3:
        # Clean vertical humerus alignment line + locked elbow pivot ring
        col = COL_GREEN
        sh = mp(726, 632)
        el = mp(724, 832)
        wr = mp(616, 938)
        d.line([sh, el, wr], fill=(0, 0, 0, 150), width=5 * ss, joint="curve")
        d.line([sh, el, wr], fill=(*col, 235), width=2 * ss, joint="curve")
        # Precision double ring at pinned elbow hinge
        r_out = 16 * ss
        d.ellipse([el[0] - r_out, el[1] - r_out, el[0] + r_out, el[1] + r_out], outline=(*col, 220), width=2 * ss)
        for pt in [sh, el]:
            r = 5 * ss
            d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=(255, 255, 255, 255), outline=(*col, 255), width=2 * ss)

    elif step_idx == 4:
        # Clean 180° full extension axis + sleek outward rope-spread arrows at wrist
        col = COL_GREEN
        sh = mp(726, 632)
        el = mp(716, 828)
        wr = mp(590, 966)
        d.line([sh, el, wr], fill=(0, 0, 0, 150), width=5 * ss, joint="curve")
        d.line([sh, el, wr], fill=(*col, 235), width=2 * ss, joint="curve")

        # Two crisp outward chevrons at the hands showing rope spread
        spread_l = mp(544, 996)
        spread_r = mp(638, 1004)
        d.line([wr, spread_l], fill=(*col, 245), width=3 * ss)
        d.line([wr, spread_r], fill=(*col, 245), width=3 * ss)
        for pt in [el, wr]:
            r = 5 * ss
            d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=(255, 255, 255, 255), outline=(*col, 255), width=2 * ss)

    canvas.alpha_composite(ov.resize((W, H), Image.Resampling.LANCZOS))


def draw_luxury_top_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
):
    """
    Pure, uncluttered `@demicstory` 3-stage top dock + 4x-supersampled titanium `✕ / ✓` emblem.
    Zero camcorder bars, zero sticker text inside thumbnails.
    """
    ss = 4
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    tw, th = 142, 114
    gap = 12
    pad = 12
    bar_x = 84
    bar_y = 242
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + 2 * pad

    # Soft ambient shadow + frosted obsidian container (`1px` subtle titanium border)
    od.rounded_rectangle([bar_x + 2, bar_y + 6, bar_x + bar_w + 2, bar_y + bar_h + 6], radius=24, fill=(0, 0, 0, 125))
    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=24,
        fill=(16, 19, 25, 215),
        outline=(255, 255, 255, 38),
        width=1,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy().resize((tw, th), Image.Resampling.LANCZOS) if i == active_idx else inact_im.copy().resize((tw, th), Image.Resampling.LANCZOS)

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], radius=15, outline=(*i_col, 240), width=2)
            # Subtle bottom pill indicator dot
            od.rounded_rectangle([tx + tw // 2 - 16, ty + th - 6, tx + tw // 2 + 16, ty + th - 2], radius=2, fill=(*i_col, 255))
        else:
            od.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=14, outline=(255, 255, 255, 22), width=1)

    # Top-Right 4x-Supersampled Titanium Emblem (`x = 912`)
    bx = 912
    by = bar_y + bar_h // 2
    br = 46
    sz = (br * 2 + 24) * ss
    c = sz // 2
    r = br * ss
    em = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    ed = ImageDraw.Draw(em)

    ed.ellipse([c - r, c - r, c + r, c + r], fill=(16, 19, 25, 238), outline=(255, 255, 255, 48), width=int(1.5 * ss))
    ri = r - 6 * ss
    ed.ellipse([c - ri, c - ri, c + ri, c + ri], fill=(20, 24, 32, 248), outline=(*accent, 235), width=int(2.5 * ss))

    if is_wrong:
        d = int(14 * ss)
        w = int(5.5 * ss)
        ed.line([(c - d, c - d), (c + d, c + d)], fill=(255, 255, 255, 255), width=w)
        ed.line([(c - d, c + d), (c + d, c - d)], fill=(255, 255, 255, 255), width=w)
    else:
        pts = [(c - 16 * ss, c + 1 * ss), (c - 3 * ss, c + 14 * ss), (c + 18 * ss, c - 12 * ss)]
        ed.line(pts, fill=(255, 255, 255, 255), width=int(6 * ss), joint="curve")

    ov.alpha_composite(em.resize((br * 2 + 24, br * 2 + 24), Image.Resampling.LANCZOS), (bx - br - 12, by - br - 12))
    canvas.alpha_composite(ov)


def draw_surgical_micro_callout(
    canvas: Image.Image,
    joint_xy: tuple[float, float],
    card_xy: tuple[int, int],
    deg_val: str,
    title_str: str,
    sub_str: str,
    is_wrong: bool,
):
    """
    Concept A's signature mid-screen visual:
    A sleek, frosted-obsidian floating micro-pill in the open left studio space with a 1.5px surgical
    hairline pointer to the exact joint/muscle. Zero pie-slice clutter, zero arcade drop-shadow text.
    """
    ss = 4
    ov_ss = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d_ss = ImageDraw.Draw(ov_ss)
    accent = COL_RED if is_wrong else COL_GREEN

    jx, jy = int(round(joint_xy[0] * ss)), int(round(joint_xy[1] * ss))
    cx, cy = card_xy

    # Measure micro-pill width
    dummy = ImageDraw.Draw(canvas)
    b_deg = dummy.textbbox((0, 0), deg_val, font=F_CALLOUT_DEG)
    b_tit = dummy.textbbox((0, 0), title_str, font=F_CALLOUT_TIT)
    b_sub = dummy.textbbox((0, 0), sub_str, font=F_CALLOUT_SUB)

    deg_w = b_deg[2] - b_deg[0]
    right_w = max(b_tit[2] - b_tit[0], b_sub[2] - b_sub[0])
    pw = deg_w + right_w + 58
    ph = 68

    # Hairline surgical pointer from right edge of micro-pill to joint
    px_end = cx + pw
    py_mid = cy + ph // 2
    elbow_x = px_end + 34
    pts = [
        (px_end * ss, py_mid * ss),
        (elbow_x * ss, py_mid * ss),
        (jx, jy),
    ]
    d_ss.line(pts, fill=(255, 255, 255, 135), width=int(1.5 * ss), joint="curve")
    # Small precision target ring on joint
    d_ss.ellipse([jx - 11 * ss, jy - 11 * ss, jx + 11 * ss, jy + 11 * ss], outline=(*accent, 235), width=2 * ss)
    d_ss.ellipse([jx - 3 * ss, jy - 3 * ss, jx + 3 * ss, jy + 3 * ss], fill=(255, 255, 255, 255))

    ov = ov_ss.resize((W, H), Image.Resampling.LANCZOS)
    od = ImageDraw.Draw(ov)

    # Frosted obsidian micro-pill (`1px` subtle titanium border + left accent bar)
    od.rounded_rectangle([cx + 2, cy + 5, cx + pw + 2, cy + ph + 5], radius=18, fill=(0, 0, 0, 120))
    od.rounded_rectangle([cx, cy, cx + pw, cy + ph], radius=18, fill=(14, 17, 23, 224), outline=(255, 255, 255, 42), width=1)
    od.rounded_rectangle([cx + 8, cy + 14, cx + 12, cy + ph - 14], radius=2, fill=(*accent, 255))

    # Big condensed numeral on left + 2-line crisp label on right
    od.text((cx + 22, cy + 13), deg_val, font=F_CALLOUT_DEG, fill=(*accent, 255))
    tx = cx + 22 + deg_w + 16
    od.line([(tx - 8, cy + 14), (tx - 8, cy + ph - 14)], fill=(255, 255, 255, 28), width=1)
    od.text((tx, cy + 13), title_str, font=F_CALLOUT_TIT, fill=(255, 255, 255, 255))
    od.text((tx, cy + 38), sub_str, font=F_CALLOUT_SUB, fill=COL_SILVER)

    canvas.alpha_composite(ov)


def draw_optical_glass_loupe(
    canvas: Image.Image,
    base_rgb: np.ndarray,
    focus_xy: tuple[float, float],
    loupe_xy: tuple[int, int],
    deg_val: str,
    title_str: str,
    is_wrong: bool,
):
    """
    Concept B's signature mid-screen visual:
    A pure, frameless-feeling **4x-Supersampled Optical Glass Loupe (`2.2×`)** with zero crosshairs
    and zero sticker clutter—just a razor-sharp specular titanium rim and an integrated caption pill below.
    """
    ss = 4
    accent = COL_RED if is_wrong else COL_GREEN
    lcx, lcy = loupe_xy
    lr = 148
    fx, fy = int(round(focus_xy[0])), int(round(focus_xy[1]))

    # Extract 2.25x magnified patch around the anatomical target
    half_crop = int(lr / 2.25)
    x0 = max(0, min(W - 2 * half_crop, fx - half_crop))
    y0 = max(0, min(H - 2 * half_crop, fy - half_crop))
    patch = Image.fromarray(base_rgb[y0 : y0 + 2 * half_crop, x0 : y0 + 2 * half_crop]).resize((lr * 2, lr * 2), Image.Resampling.LANCZOS)

    # Subtle optical inner vignette on patch so the rim looks like curved crystal glass
    yy, xx = np.mgrid[: lr * 2, : lr * 2].astype(np.float32)
    rn = np.sqrt(((xx - lr) / lr) ** 2 + ((yy - lr) / lr) ** 2)
    vig = np.clip(1.0 - 0.35 * np.clip((rn - 0.72) / 0.28, 0.0, 1.0), 0.65, 1.0)[..., None]
    patch_vig = Image.fromarray(np.clip(np.array(patch, dtype=np.float32) * vig, 0, 255).astype(np.uint8)).convert("RGBA")

    mask = Image.new("L", (lr * 2, lr * 2), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, lr * 2 - 1, lr * 2 - 1], fill=255)

    # 4x supersampled tangent line + optical rim
    ov_ss = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d_ss = ImageDraw.Draw(ov_ss)

    # Soft drop shadow
    d_ss.ellipse([(lcx - lr - 6) * ss, (lcy - lr + 4) * ss, (lcx + lr + 6) * ss, (lcy + lr + 16) * ss], fill=(0, 0, 0, 155))

    # Fine surgical connection line from loupe edge to joint ring
    r_joint = 22
    angle = math.atan2(fy - lcy, fx - lcx)
    sx = int(round((lcx + (lr + 2) * math.cos(angle)) * ss))
    sy = int(round((lcy + (lr + 2) * math.sin(angle)) * ss))
    ex = int(round((fx - r_joint * math.cos(angle)) * ss))
    ey = int(round((fy - r_joint * math.sin(angle)) * ss))

    d_ss.line([(sx, sy), (ex, ey)], fill=(255, 255, 255, 145), width=int(1.5 * ss))
    d_ss.ellipse([(fx - r_joint) * ss, (fy - r_joint) * ss, (fx + r_joint) * ss, (fy + r_joint) * ss], outline=(*accent, 235), width=2 * ss)

    ov = ov_ss.resize((W, H), Image.Resampling.LANCZOS)
    ov.paste(patch_vig, (lcx - lr, lcy - lr), mask)

    # High-res specular rim drawn at 4x
    rim_ss = Image.new("RGBA", ((lr * 2 + 16) * ss, (lr * 2 + 16) * ss), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rim_ss)
    rc = (lr + 8) * ss
    rr = lr * ss
    rd.ellipse([rc - rr, rc - rr, rc + rr, rc + rr], outline=(255, 255, 255, 75), width=int(1.5 * ss))
    rd.arc([rc - rr - 2 * ss, rc - rr - 2 * ss, rc + rr + 2 * ss, rc + rr + 2 * ss], start=-55, end=65, fill=(*accent, 240), width=int(2.5 * ss))
    ov.alpha_composite(rim_ss.resize((lr * 2 + 16, lr * 2 + 16), Image.Resampling.LANCZOS), (lcx - lr - 8, lcy - lr - 8))

    # Sleek, minimal floating caption pill tucked cleanly below the loupe
    od = ImageDraw.Draw(ov)
    lbl_str = f"{deg_val}  ·  {title_str}"
    bb = od.textbbox((0, 0), lbl_str, font=F_CALLOUT_SUB)
    pw = (bb[2] - bb[0]) + 32
    ph = 36
    px0, py0 = lcx - pw // 2, lcy + lr + 12
    od.rounded_rectangle([px0, py0, px0 + pw, py0 + ph], radius=18, fill=(14, 17, 23, 232), outline=(255, 255, 255, 42), width=1)
    od.text((px0 + 16, py0 + 9), lbl_str, font=F_CALLOUT_SUB, fill=(255, 255, 255, 255))

    canvas.alpha_composite(ov)


def draw_luxury_concept_b_hud(
    canvas: Image.Image,
    target_xy: tuple[float, float],
    phase: dict,
):
    """
    The definitive, luxury Concept B Lower-Left HUD (`Radial Dial + Frosted Obsidian Capsule`):
    - Zero thick neon borders (`1px` subtle specular titanium rim `rgba(255,255,255,0.16)`).
    - 4x-supersampled 270° radial progress ring with smooth rounded stroke caps and `Barlow Condensed` numerals.
    - Clean 3-tier coaching typography (`Kicker` -> `38px Hero Cue` -> `19px Why/Do Coaching Line`).
    - Frameless inline muscle-head & prescription strip separated by a `1px` hairline divider (zero sub-boxes!).
    """
    ss = 4
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    ring_cx, ring_cy, ring_r = 172, 1312, 86
    px1, py1, px2, py2 = 236, 1208, 976, 1416

    # 1. Hairline leader from active muscle/joint to top of radial dial
    tx, ty = int(round(target_xy[0])), int(round(target_xy[1]))
    ax, ay = ring_cx, ring_cy - ring_r - 4
    pts = [(tx, ty), (ax + 24, ty), (ax, ty + 24), (ax, ay)]
    od.line(pts, fill=(255, 255, 255, 115), width=2, joint="curve")
    od.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=(*accent, 255))

    # 2. Frosted Obsidian Floating Capsule (soft ambient shadow + 1px titanium specular rim)
    od.rounded_rectangle([px1 + 2, py1 + 8, px2 + 2, py2 + 8], radius=42, fill=(0, 0, 0, 145))
    od.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=42,
        fill=(14, 17, 23, 232),
        outline=(255, 255, 255, 42),
        width=1,
    )
    # Subtle top specular highlight along the capsule's upper edge
    od.line([(px1 + 42, py1), (px2 - 42, py1)], fill=(255, 255, 255, 78), width=1)

    # 3. 4x-Supersampled Radial Progress Ring with Rounded Caps
    pct_val = phase["pct"]
    dsz = (ring_r * 2 + 24) * ss
    dc = dsz // 2
    dr = ring_r * ss
    dial = Image.new("RGBA", (dsz, dsz), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)

    dd.ellipse([dc - dr, dc - dr, dc + dr, dc + dr], fill=(12, 15, 20, 250), outline=(255, 255, 255, 48), width=int(1.5 * ss))
    pad_arc = 11 * ss
    arc_box = [dc - dr + pad_arc, dc - dr + pad_arc, dc + dr - pad_arc, dc + dr - pad_arc]
    stroke_w = 9 * ss
    dd.arc(arc_box, start=135, end=405, fill=(34, 40, 52, 240), width=stroke_w)

    end_deg = 135 + int(round(270 * (pct_val / 100.0)))
    dd.arc(arc_box, start=135, end=end_deg, fill=(*accent, 255), width=stroke_w)

    # Rounded end-caps on the arc
    r_mid = dr - pad_arc - stroke_w // 2
    for deg_cap, col_cap in [(135, accent), (end_deg, accent)]:
        rad = math.radians(deg_cap)
        cx_c = dc + r_mid * math.cos(rad)
        cy_c = dc + r_mid * math.sin(rad)
        rc_w = stroke_w // 2
        dd.ellipse([cx_c - rc_w, cy_c - rc_w, cx_c + rc_w, cy_c + rc_w], fill=(*col_cap, 255))

    ov.alpha_composite(dial.resize((ring_r * 2 + 24, ring_r * 2 + 24), Image.Resampling.LANCZOS), (ring_cx - ring_r - 12, ring_cy - ring_r - 12))

    # Crisp condensed numeral inside dial
    n_str = f"{pct_val}"
    nb = od.textbbox((0, 0), n_str, font=F_DIAL_NUM)
    pb = od.textbbox((0, 0), "%", font=F_DIAL_PCT)
    tot_w = (nb[2] - nb[0]) + (pb[2] - pb[0]) + 3
    nx = ring_cx - tot_w // 2
    od.text((nx, ring_cy - 36), n_str, font=F_DIAL_NUM, fill=(255, 255, 255, 255))
    od.text((nx + (nb[2] - nb[0]) + 3, ring_cy - 18), "%", font=F_DIAL_PCT, fill=COL_SILVER)

    sb = od.textbbox((0, 0), phase["dial_tag"], font=F_DIAL_SUB)
    od.text((ring_cx - (sb[2] - sb[0]) // 2, ring_cy + 22), phase["dial_tag"], font=F_DIAL_SUB, fill=(*accent, 255))

    # 4. Clean 3-Tier Typography inside Capsule (`tx = 282`)
    tx = 282
    od.text((tx, py1 + 20), phase["kicker"], font=F_KICKER, fill=(*accent, 255))
    od.text((tx, py1 + 44), phase["line1"], font=F_HERO, fill=(255, 255, 255, 255))
    od.text((tx, py1 + 96), phase["line2"], font=F_TIP, fill=COL_SILVER)

    # 5. Frameless Inline Telemetry Strip below a 1px hairline divider (`y = 1344..1396`)
    div_y = py1 + 134
    od.line([(tx, div_y), (px2 - 36, div_y)], fill=(255, 255, 255, 26), width=1)

    mx = tx
    my = div_y + 14
    for idx_m, (m_lbl, m_val, m_col) in enumerate(phase["splits"]):
        od.text((mx, my + 2), m_lbl, font=F_METRIC_LBL, fill=COL_MUTED)
        lb = od.textbbox((0, 0), m_lbl, font=F_METRIC_LBL)
        vx = mx + (lb[2] - lb[0]) + 8
        v_str = f"{m_val}%"
        od.text((vx, my - 1), v_str, font=F_METRIC_VAL, fill=(255, 255, 255, 255) if m_col == accent else COL_SILVER)
        vb = od.textbbox((0, 0), v_str, font=F_METRIC_VAL)

        # Ultra-sleek 4px micro progress bar underneath
        bar_w = (lb[2] - lb[0]) + (vb[2] - vb[0]) + 8
        by_bar = my + 24
        od.rounded_rectangle([mx, by_bar, mx + bar_w, by_bar + 4], radius=2, fill=(38, 44, 56, 255))
        fw = max(6, int(bar_w * (m_val / 100.0)))
        od.rounded_rectangle([mx, by_bar, mx + fw, by_bar + 4], radius=2, fill=(*m_col, 255))

        mx += bar_w + 28
        if idx_m < len(phase["splits"]) - 1:
            od.line([(mx - 14, my + 2), (mx - 14, my + 26)], fill=(255, 255, 255, 24), width=1)

    # Right-aligned prescription pill (`4 × 12 REPS` or `15° HINGE`)
    rx_str = phase["rx_badge"]
    rxb = od.textbbox((0, 0), rx_str, font=F_METRIC_LBL)
    rx_w = (rxb[2] - rxb[0]) + 24
    od.rounded_rectangle([px2 - rx_w - 28, my - 2, px2 - 28, my + 28], radius=15, fill=(255, 255, 255, 18), outline=(255, 255, 255, 38), width=1)
    od.text((px2 - rx_w - 16, my + 6), rx_str, font=F_METRIC_LBL, fill=(255, 255, 255, 255))

    canvas.alpha_composite(ov)


def main():
    f_w45 = eng.prepare_xray_bone_frame(v25.read_specific_frame(eng.WRONG_I2V, 45))
    f_c14 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 14))
    f_c42 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 42))
    f_c64 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 64))

    size = (142, 114)
    crop_box = (555, 550, 905, 950)
    inact_base = Image.fromarray(f_c42["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 125)))

    x_arr, x_targets, _, _ = eng.render_xray_skeletal_frame(f_w45, 1.0, 0.90)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m1_arr, m1_targets = v25.render_natural_muscle_frame(f_c14, "anterior_deltoid", True, 1.0)
    im1 = Image.fromarray(m1_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m2_arr, m2_targets = v25.render_natural_muscle_frame(f_c42, "triceps_lat_long", False, 1.0)
    m3_arr, m3_targets = v25.render_natural_muscle_frame(f_c64, "triceps_all_three", False, 1.0)
    im2 = Image.fromarray(m3_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    steps = [
        {
            "idx": 1,
            "is_wrong": True,
            "base_rgb": x_arr,
            "raw_targets": x_targets,
            "active_slot": 0,
            "cam_zoom": 1.04,
            "cam_focus": (690.0, 735.0),
            "cam_dest": (690.0, 745.0),
            "pct": 86,
            "dial_tag": "JOINT RISK",
            "kicker": "MISTAKE 01  ·  TORSO & SHOULDER POSTURE",
            "line1": "DON'T HUNCH SHOULDERS",
            "line2": "Pressing with rounded back impinges the shoulder joint",
            "deg_val": "35°",
            "deg_title": "ROUNDED UPPER BACK",
            "deg_sub": "Acromion joint impingement",
            "joint_pt": (728.0, 618.0),
            "callout_xy": (84, 535),
            "loupe_xy": (244, 710),
            "splits": [
                ("SHOULDER LOAD", 86, COL_RED),
                ("TRICEPS WORK", 18, COL_MUTED),
            ],
            "rx_badge": "FIX: CHEST UP",
        },
        {
            "idx": 2,
            "is_wrong": True,
            "base_rgb": m1_arr,
            "raw_targets": m1_targets,
            "active_slot": 1,
            "cam_zoom": 1.34,
            "cam_focus": (675.0, 735.0),
            "cam_dest": (705.0, 755.0),
            "pct": 72,
            "dial_tag": "DELT STEAL",
            "kicker": "MISTAKE 02  ·  INCOMPLETE RANGE OF MOTION",
            "line1": "AVOID 90° HALF REPS",
            "line2": "Stopping at 90° & lifting elbows shifts load to front delts",
            "deg_val": "90°",
            "deg_title": "STOPPED SHORT",
            "deg_sub": "72% stolen by front deltoid",
            "joint_pt": (602.0, 882.0),
            "callout_xy": (84, 775),
            "loupe_xy": (244, 725),
            "splits": [
                ("FRONT DELT", 72, COL_RED),
                ("LATERAL HEAD", 28, COL_MUTED),
            ],
            "rx_badge": "FIX: FULL RANGE",
        },
        {
            "idx": 3,
            "is_wrong": False,
            "base_rgb": m2_arr,
            "raw_targets": m2_targets,
            "active_slot": 2,
            "cam_zoom": 1.16,
            "cam_focus": (685.0, 745.0),
            "cam_dest": (695.0, 755.0),
            "pct": 95,
            "dial_tag": "LAT & LONG",
            "kicker": "PRO TIP 01  ·  SCAPULA & ELBOW HINGE SETUP",
            "line1": "PIN ELBOWS TO RIBS",
            "line2": "Hinge 15° forward, depress shoulders & move forearms only",
            "deg_val": "0°",
            "deg_title": "FIXED ELBOW HINGE",
            "deg_sub": "15° slight torso lean",
            "joint_pt": (724.0, 832.0),
            "callout_xy": (84, 755),
            "loupe_xy": (244, 725),
            "splits": [
                ("LAT HEAD", 95, COL_GREEN),
                ("LONG HEAD", 91, COL_GREEN),
            ],
            "rx_badge": "15° TORSO LEAN",
        },
        {
            "idx": 4,
            "is_wrong": False,
            "base_rgb": m3_arr,
            "raw_targets": m3_targets,
            "active_slot": 2,
            "cam_zoom": 1.40,
            "cam_focus": (665.0, 790.0),
            "cam_dest": (700.0, 755.0),
            "pct": 99,
            "dial_tag": "ALL 3 HEADS",
            "kicker": "PRO TIP 02  ·  FULL EXTENSION & PEAK SQUEEZE",
            "line1": "SPREAD ROPE AT BOTTOM",
            "line2": "Pull handles apart at 180° lockout & hold 1 sec squeeze",
            "deg_val": "180°",
            "deg_title": "FULL ROPE SPREAD",
            "deg_sub": "1.0s isometric peak hold",
            "joint_pt": (588.0, 968.0),
            "callout_xy": (84, 775),
            "loupe_xy": (244, 735),
            "splits": [
                ("LAT 99%", 99, COL_GREEN),
                ("LONG 95%", 95, COL_GREEN),
                ("MED 96%", 96, COL_GREEN),
            ],
            "rx_badge": "4 × 12 REPS",
        },
    ]

    frames_A = []
    frames_B = []

    for sp in steps:
        # --- Render Concept A (Multi-Camera Framing + Surgical Frosted Micro-Callout) ---
        cam_rgb, map_pt = apply_cinema_camera(sp["base_rgb"], zoom=sp["cam_zoom"], focus_xy=sp["cam_focus"], dest_xy=sp["cam_dest"])
        im_a = Image.fromarray(cam_rgb).convert("RGBA")
        draw_subtle_onbody_vector(im_a, sp["idx"], map_pt)
        draw_luxury_top_dock(im_a, torso_icons, sp["active_slot"], sp["is_wrong"])

        mj = map_pt(*sp["joint_pt"])
        mt = map_pt(*sp["raw_targets"][0][:2])
        draw_surgical_micro_callout(im_a, mj, sp["callout_xy"], sp["deg_val"], sp["deg_title"], sp["deg_sub"], sp["is_wrong"])
        draw_luxury_concept_b_hud(im_a, mt, sp)

        out_a = os.path.join(OUT_DIR, f"v28_concept_A_step{sp['idx']}.jpg")
        im_a.convert("RGB").save(out_a, quality=95)
        frames_A.append(im_a.convert("RGB"))

        # --- Render Concept B (Full-Body + 4x Optical Glass 2.2x Loupe) ---
        im_b = Image.fromarray(sp["base_rgb"]).convert("RGBA")
        draw_subtle_onbody_vector(im_b, sp["idx"], lambda x, y: (float(x), float(y)))
        draw_luxury_top_dock(im_b, torso_icons, sp["active_slot"], sp["is_wrong"])

        raw_t = sp["raw_targets"][0][:2]
        draw_optical_glass_loupe(im_b, sp["base_rgb"], raw_t, sp["loupe_xy"], sp["deg_val"], sp["deg_title"], sp["is_wrong"])
        draw_luxury_concept_b_hud(im_b, raw_t, sp)

        out_b = os.path.join(OUT_DIR, f"v28_concept_B_step{sp['idx']}.jpg")
        im_b.convert("RGB").save(out_b, quality=95)
        frames_B.append(im_b.convert("RGB"))

    # Build Side-by-Side 4-Step Storyboard Comparison (`day1_v28_luxury_A_vs_B_board.jpg`)
    pw, ph = 390, 693
    board = Image.new("RGB", (pw * 4 + 24 * 5, ph * 2 + 185), (10, 12, 16))
    bd = ImageDraw.Draw(board)

    f_row_tit = f_pretendard(22, "extrabold")
    f_col_tit = f_pretendard(15, "bold")

    bd.text((28, 22), "CONCEPT A  ·  다이내믹 다각도 카메라 줌 (1.0× ↔ 1.4×) + 무박스 마이크로 각도 필 + 럭셔리 글래스 HUD", font=f_row_tit, fill=COL_GREEN)
    bd.text((28, ph + 112), "CONCEPT B  ·  전신 자세 + 2.2배 옵티컬 크리스탈 돋보기 렌즈(Optical Loupe) + 럭셔리 글래스 HUD", font=f_row_tit, fill=(110, 215, 255))

    step_titles = [
        "STEP 1 · 35° HUNCHED BACK (✕)",
        "STEP 2 · 90° HALF REP STEAL (✕)",
        "STEP 3 · 0° PINNED ELBOWS (✓)",
        "STEP 4 · 180° ROPE SPREAD (✓)",
    ]

    for i in range(4):
        x = 24 + i * (pw + 24)
        col = COL_RED if i < 2 else COL_GREEN

        # Row 1: Concept A
        bd.text((x + 4, 58), step_titles[i], font=f_col_tit, fill=col)
        ra = frames_A[i].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 84))

        # Row 2: Concept B
        rb = frames_B[i].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rb).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(rb, (x, ph + 148))

    board_p = os.path.join(OUT_DIR, "day1_v28_luxury_A_vs_B_board.jpg")
    board.save(board_p, quality=95)
    print("Saved v28 Luxury A vs B Board:", board_p)


if __name__ == "__main__":
    main()
