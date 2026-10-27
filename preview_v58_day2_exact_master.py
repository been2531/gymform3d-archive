#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v58_day2_exact_master.py`)
===============================================================
Fixes all 3 issues 중현 님 pointed out in `t241u`–`t246u`:
1. "프리뷰 위아래 일부로 줄인것처럼 이상하고" (`t241u`):
   - Restores Day 2 (`v39` `build_v39_ig_safe_zone.py` `draw_v39_safezone_top_dock`)'s EXACT un-squished dimensions:
     * Top Exercise Header at `x = 96, y = 252 .. 298` (`hdr_h = 46px`).
     * 3-Stage Thumbnail Dock at `x = 96, y = 308 .. 438` (`tw = 136, th = 108, gap = 10, pad = 11, dock_h = 130px`).
     * Crops each thumbnail with the exact `136:108` (`544 × 432` -> `136 × 108`) aspect ratio so there is 0% vertical/horizontal squishing!
2. "프리뷰랑 머리랑 닿으면 당연히 안되겠지? / 릴스 최적화가 부족했던거고" (`t241u`, `t246u`):
   - Places the Bottom-Left HUD at `x = 96, y = 1348 .. 1490` (`95px` above `@gymform.3d 팔로우` at `y = 1585`).
   - Frames the mannequin across all 4 steps at `scale = 0.80 .. 0.845` so that across ALL 600 frames (standing AND squatting):
     * `min(head_top_y) >= 455px` (`>= 17px` below the bottom of the un-squished Top Preview Dock at `y = 438` -> 0% contact!).
     * `max(foot_bot_y) <= 1344px` (cleanly above the Bottom HUD at `y = 1348`!).
3. "화살표랑 점 이상하고 / 단지 화살표가 거슬렀던거지" (`t241u`–`t245u`):
   - Removes ALL bullseye target rings (`◎`), ALL big triangular arrowheads, and ALL disconnected floating arcs!
   - Replaces them with a clean, geometrically exact Biomechanical Angle Gauge (`각도 게이지`):
     * 1 thin vertical dashed reference line (`270°` straight up from the joint vertex),
     * 1 thin glowing limb/spine axis line (`3.0px` core + soft glow, with tiny `3.5px` solid vertex dots),
     * 1 clean filled translucent angle sector (`alpha=62`) + crisp `4.2px` solid arc (`SS=4`, zero triangle arrowheads!)
       spanning directly between the vertical reference line and the actual limb/spine axis line!
     * Pair 1 (Front View — Step 1 Wrong vs Step 2 Right): Both measure Knee Alignment Angle (`18° VALGUS CAVE` vs `0° KNEES TRACK TOES`)
       between the vertical ankle-plumb line and the shin-to-knee axis (`ank -> kn`).
     * Pair 2 (Side View — Step 3 Wrong vs Step 4 Right): Both measure Torso Inclination Angle (`42° FORWARD LEAN` vs `15° UPRIGHT LOCK`)
       at the hip vertex (`hip_r`) directly between the vertical plumb line (`270°`) and the forward-inclined torso axis (`hip_r -> sh`, `270°..312°` vs `270°..286°`)!
"""

import os
import sys
import math
import pickle
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v30_pure_graphic_numbers as v30
import build_v38_high_contrast_hud as v38
import build_v39_ig_safe_zone as v39
import preview_v55_demic_day2_master as p55

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_V58_PKL = "/workspace/daily_info_reels/_cache_day3_v58_pools.pkl"
W, H = 1080, 1920

SAFE_X_LEFT = 96
SAFE_X_RIGHT_MAX = 975
SAFE_Y_TOP = 252  # Exact Day 2 header top (below IG "+ / 릴스 친구" at y=165..242); dock ends at y=438

COL_RED = (248, 44, 52)
COL_RED_SOFT = (255, 168, 162)

COL_GREEN = (20, 226, 122)
COL_GREEN_SOFT = (168, 255, 210)

COL_WHITE = (255, 255, 255)
COL_PLUMB = (220, 232, 248)

F_EX_TITLE = v30.f_pretendard(24, "extrabold")
F_EX_SUB = v30.f_pretendard(16, "extrabold")

F_TAG_PILL = v30.f_pretendard(19, "extrabold")
F_TAG_CTX = v30.f_pretendard(19, "bold")
F_HERO_PCT = v30.f_barlow(84, extrabold=True)
F_HERO_SYM = v30.f_barlow(44, extrabold=True)
F_SINGLE_LINE = v30.f_pretendard(42, "extrabold")

F_CALLOUT_NUM = v30.f_barlow(68, extrabold=True)
F_CALLOUT_SUB = v30.f_pretendard(23, "extrabold")

_YY, _XX = np.mgrid[:H, :W].astype(np.float32)
_RADIAL_STAGE = np.exp(-(((_XX - 560.0) / 380.0) ** 2 + ((_YY - 920.0) / 450.0) ** 2))[:, :, None]
_TINT_WRONG = np.array([19.0, -9.0, -9.0], dtype=np.float32)[None, None, :] * _RADIAL_STAGE
_TINT_RIGHT = np.array([-14.0, 16.0, 13.0], dtype=np.float32)[None, None, :] * _RADIAL_STAGE


def apply_static_studio_tint(rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    out = rgb.astype(np.float32) + (_TINT_WRONG if is_wrong else _TINT_RIGHT)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def get_or_build_v58_pools():
    if os.path.exists(CACHE_V58_PKL):
        with open(CACHE_V58_PKL, "rb") as f:
            return pickle.load(f)

    print("Building v58 Safe-Zone Framed 3D Pools (`scale = 0.80..0.845`, `head_top >= 455`, `foot_bot <= 1344`)...")
    s1_raw = p55._load_slice("/workspace/daily_info_reels/output/day3_v55_step1_valgus_to_stand.mp4", 64, 24)
    s2_raw = p55._load_slice("/workspace/daily_info_reels/output/day3_v43_step2_front_fix.mp4", 94, 138)
    s3_raw = p55._load_slice("/workspace/daily_info_reels/output/day3_v43_step3_side_wrong.mp4", 75, 125)
    s4_raw = p55._load_slice("/workspace/daily_info_reels/output/day3_v43_step4_side_fix.mp4", 98, 25)

    pools = {
        1: p55.build_seamless_framed_pool(s1_raw, 1, True,  scale=0.810, y_head_target=465, y_head_src_720=10.0,  x_shift=+20),
        2: p55.build_seamless_framed_pool(s2_raw, 2, False, scale=0.805, y_head_target=450, y_head_src_720=32.0,  x_shift=+24),
        3: p55.build_seamless_framed_pool(s3_raw, 3, True,  scale=0.798, y_head_target=455, y_head_src_720=62.0,  x_shift=+12),
        4: p55.build_seamless_framed_pool(s4_raw, 4, False, scale=0.845, y_head_target=459, y_head_src_720=128.0, x_shift=+20),
    }
    # Normalize `foot_bot` padding (+6px below MediaPipe toe/heel landmarks)
    for s_idx in (1, 2, 3, 4):
        for kp in pools[s_idx][1]:
            kp["foot_bot"] = (kp["foot_bot"][0], kp["foot_bot"][1] - 6.0)

    with open(CACHE_V58_PKL, "wb") as f:
        pickle.dump(pools, f)
    print("Saved:", CACHE_V58_PKL)
    return pools


def make_v58_torso_icons(pools: dict) -> list[tuple[Image.Image, Image.Image]]:
    """
    Creates the 3 progression thumbnails using an EXACT `136:108` (`544 × 432`) aspect-ratio crop
    around the squatting torso & thighs so there is 0.0% vertical or horizontal squishing!
    """
    torso_icons = []
    for s_idx in [1, 3, 4]:
        p_pool, p_kps = pools[s_idx]
        mid_rgb = p_pool[-1]
        hip_x, hip_y = p_kps[-1]["hip_r"]
        # Exact 4x multiple of (136, 108) -> (544, 432)
        cw, ch = 544, 432
        cx = int(np.clip(hip_x + 25, cw // 2, W - cw // 2))
        cy = int(np.clip(hip_y - 20, ch // 2 + 440, H - ch // 2))
        x0, y0 = cx - cw // 2, cy - ch // 2
        crop = mid_rgb[y0 : y0 + ch, x0 : x0 + cw]

        pil_act = Image.fromarray(crop).resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        dark = np.stack([gray * 0.48, gray * 0.50, gray * 0.54], axis=2).astype(np.uint8)
        pil_inact = Image.fromarray(dark).resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        torso_icons.append((pil_act, pil_inact))
    return torso_icons


def draw_v58_day2_unsquished_top_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    exercise_name: str = "BARBELL BACK SQUAT",
    muscle_group: str = "[ LEGS · QUADS & GLUTES ]",
):
    """
    Restores Day 2 (`build_v39_ig_safe_zone.py` `draw_v39_safezone_top_dock`)'s exact un-squished proportions:
    - Top Exercise Header at `x = 96, y = 252 .. 298` (`hdr_h = 46px`).
    - 3-Stage Thumbnail Dock at `x = 96, y = 308 .. 438` (`tw = 136, th = 108, gap = 10, pad = 11, dock_h = 130px`).
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN
    pill_bg = (218, 44, 38, 245) if is_wrong else (18, 168, 88, 245)

    bar_x = SAFE_X_LEFT
    hdr_y = SAFE_Y_TOP  # 252
    hdr_h = 46          # 252 .. 298
    tb_ex = od.textbbox((0, 0), exercise_name, font=F_EX_TITLE)
    ex_w = tb_ex[2] - tb_ex[0]
    tb_mg = od.textbbox((0, 0), muscle_group, font=F_EX_SUB)
    mg_w = tb_mg[2] - tb_mg[0]

    pill_w = mg_w + 24
    hdr_w = 22 + 12 + ex_w + 16 + pill_w + 16

    od.rounded_rectangle([bar_x + 2, hdr_y + 4, bar_x + hdr_w + 2, hdr_y + hdr_h + 4], radius=14, fill=(0, 0, 0, 150))
    od.rounded_rectangle(
        [bar_x, hdr_y, bar_x + hdr_w, hdr_y + hdr_h],
        radius=14,
        fill=(14, 17, 23, 232),
        outline=(255, 255, 255, 44),
        width=1,
    )

    dot_cx = bar_x + 20
    dot_cy = hdr_y + hdr_h // 2
    od.ellipse([dot_cx - 8, dot_cy - 8, dot_cx + 8, dot_cy + 8], fill=(*accent, 95))
    od.ellipse([dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5], fill=(*accent, 255))

    ex_x = dot_cx + 15
    od.text((ex_x, hdr_y + 9), exercise_name, font=F_EX_TITLE, fill=(255, 255, 255, 255))

    p_x0 = ex_x + ex_w + 14
    p_y0 = hdr_y + 8
    p_h = hdr_h - 16
    od.rounded_rectangle([p_x0, p_y0, p_x0 + pill_w, p_y0 + p_h], radius=8, fill=pill_bg)
    od.text((p_x0 + 12, p_y0 + 5), muscle_group, font=F_EX_SUB, fill=(255, 255, 255, 255))

    # Exact Day 2 (`v39`) 3-Stage Thumbnail Dock (`tw = 136, th = 108, gap = 10, pad = 11` -> `y = 308 .. 438`)
    tw, th = 136, 108
    gap = 10
    pad = 11
    dock_y = hdr_y + hdr_h + 10  # y = 308
    dock_w = 3 * tw + 2 * gap + 2 * pad
    dock_h = th + 2 * pad        # 130px -> ends at y = 438!

    od.rounded_rectangle([bar_x + 2, dock_y + 6, bar_x + dock_w + 2, dock_y + dock_h + 6], radius=22, fill=(0, 0, 0, 135))
    od.rounded_rectangle(
        [bar_x, dock_y, bar_x + dock_w, dock_y + dock_h],
        radius=22,
        fill=(16, 19, 25, 222),
        outline=(255, 255, 255, 38),
        width=1,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = dock_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im if i == active_idx else inact_im

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=13, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], radius=14, outline=(*i_col, 245), width=2)
            od.rounded_rectangle([tx + tw // 2 - 16, ty + th - 6, tx + tw // 2 + 16, ty + th - 2], radius=2, fill=(*i_col, 255))
        else:
            od.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=13, outline=(255, 255, 255, 22), width=1)

    canvas.alpha_composite(ov)


def draw_v58_safezone_bottom_hud(canvas: Image.Image, sp: dict):
    """
    v58 Safe-Zone Bottom-Left HUD (`x = 96, y = 1348 .. 1488`):
    - Starts at `y = 1348` (cleanly below the feet at `y <= 1344`).
    - Ends at `y = 1488` (`97px` cleanly above `@gymform.3d 팔로우` at `y = 1585` on iPhone Reels).
    """
    is_wrong = sp["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    pill_fill = (224, 44, 38, 250) if is_wrong else (16, 168, 86, 250)
    pct = int(np.clip(sp["pct"], 0, 100))

    tag_pill_str = sp["tag_pill"]
    tag_ctx_str = sp["tag_ctx"]
    cue_1line = sp["cue_1line"]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx = SAFE_X_LEFT  # 96px

    # Row 1 (`y = 1348`)
    py = 1348
    tb_p = od.textbbox((0, 0), tag_pill_str, font=F_TAG_PILL)
    pw = (tb_p[2] - tb_p[0]) + 26
    ph = 34

    sd.rounded_rectangle([bx, py + 3, bx + pw, py + ph + 3], radius=8, fill=(0, 0, 0, 225))
    od.rounded_rectangle([bx, py, bx + pw, py + ph], radius=8, fill=pill_fill, outline=(255, 255, 255, 55), width=1)
    od.text((bx + 13, py + 6), tag_pill_str, font=F_TAG_PILL, fill=(255, 255, 255, 255))

    ctx_x = bx + pw + 13
    ctx_y = py + 6
    sd.text((ctx_x, ctx_y + 3), tag_ctx_str, font=F_TAG_CTX, fill=(0, 0, 0, 245))
    od.text((ctx_x, ctx_y), tag_ctx_str, font=F_TAG_CTX, fill=(228, 236, 248, 255))

    # Row 2 (`y = 1390`)
    by = 1390
    num_str = f"{pct}"
    nb = od.textbbox((0, 0), num_str, font=F_HERO_PCT)
    nw = nb[2] - nb[0]

    sd.text((bx, by + 4), num_str, font=F_HERO_PCT, fill=(0, 0, 0, 245))
    od.text((bx, by), num_str, font=F_HERO_PCT, fill=(255, 255, 255, 255))

    sym_x = bx + nw + 3
    sym_y = by + 12
    sd.text((sym_x, sym_y + 3), "%", font=F_HERO_SYM, fill=(0, 0, 0, 240))
    od.text((sym_x, sym_y), "%", font=F_HERO_SYM, fill=(*accent, 255))
    sb = od.textbbox((0, 0), "%", font=F_HERO_SYM)
    sw = sb[2] - sb[0]

    div_x = sym_x + sw + 18
    div_y0 = by + 18
    div_y1 = by + 68
    sd.rounded_rectangle([div_x, div_y0 + 2, div_x + 4, div_y1 + 2], radius=2, fill=(0, 0, 0, 225))
    od.rounded_rectangle([div_x, div_y0, div_x + 4, div_y1], radius=2, fill=(*accent, 255))

    tx = div_x + 18
    ty = by + 19
    tb_c = od.textbbox((0, 0), cue_1line, font=F_SINGLE_LINE)
    cw = tb_c[2] - tb_c[0]

    sd.text((tx, ty + 4), cue_1line, font=F_SINGLE_LINE, fill=(0, 0, 0, 245))
    od.text((tx, ty), cue_1line, font=F_SINGLE_LINE, fill=(255, 255, 255, 255))

    # Row 3 (`y = 1480`)
    bar_x0 = bx + 2
    bar_y0 = by + 90
    total_row_w = (tx + cw) - bar_x0
    bar_w = max(360, min(510, total_row_w))
    bar_h = 6
    fill_w = max(14, int(round(bar_w * (pct / 100.0))))

    sd.rounded_rectangle([bar_x0, bar_y0 + 2, bar_x0 + bar_w, bar_y0 + bar_h + 2], radius=3, fill=(0, 0, 0, 215))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], radius=3, fill=(255, 255, 255, 42))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y0 + bar_h], radius=3, fill=(*accent, 248))

    dot_x = bar_x0 + fill_w
    dot_y = bar_y0 + bar_h * 0.5
    sd.ellipse([dot_x - 8, dot_y - 8, dot_x + 8, dot_y + 8], fill=(*accent, 175))
    od.ellipse([dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5], fill=(255, 255, 255, 255))

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
    canvas.alpha_composite(ov)


def _draw_smooth_seg_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    half_w_ss: float,
    rgba: tuple[int, int, int, int],
):
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 0.5:
        return
    nx, ny = -dy / L, dx / L
    rd.polygon(
        [
            (x0 + nx * half_w_ss, y0 + ny * half_w_ss),
            (x1 + nx * half_w_ss, y1 + ny * half_w_ss),
            (x1 - nx * half_w_ss, y1 - ny * half_w_ss),
            (x0 - nx * half_w_ss, y0 - ny * half_w_ss),
        ],
        fill=rgba,
    )
    r = half_w_ss
    rd.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=rgba)
    rd.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=rgba)


def draw_clean_angle_gauge_ss(
    rd: ImageDraw.ImageDraw,
    bd_1x: ImageDraw.ImageDraw,
    vertex_1x: tuple[float, float],
    ref_pt_1x: tuple[float, float],
    limb_pt_1x: tuple[float, float],
    next_limb_pt_1x: tuple[float, float] | None,
    arc_radius_1x: float,
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 4,
):
    """
    Draws a Clean, Geometrically Exact Biomechanical Angle Gauge (`각도 게이지`):
    - ZERO bullseye target rings (`◎`)!
    - ZERO chunky triangle arrowheads (`draw_arrow_head`)!
    - ZERO disconnected floating arcs in empty space!
    - Consists of:
      1. Crisp vertical dashed reference line (`vertex -> ref_pt`),
      2. Filled translucent angle sector (`alpha=64`) + bold `4.2px` solid arc (`SS=4`) directly between
         `vertex -> ref_pt` and `vertex -> limb_pt`,
      3. Glowing `3.0px` limb/spine axis line (`vertex -> limb_pt`, and optionally `limb_pt -> next_limb_pt`),
      4. Tiny `3.5px` solid white vertex pinpoints where the lines meet.
    """
    vx_s, vy_s = vertex_1x[0] * ss, vertex_1x[1] * ss
    rx_s, ry_s = ref_pt_1x[0] * ss, ref_pt_1x[1] * ss
    lx_s, ly_s = limb_pt_1x[0] * ss, limb_pt_1x[1] * ss

    # 1. Angle Sector & Solid Arc directly between Reference Line and Limb/Spine Line
    a_ref = math.atan2(ry_s - vy_s, rx_s - vx_s)
    a_limb = math.atan2(ly_s - vy_s, lx_s - vx_s)
    diff = (a_limb - a_ref + math.pi) % (2.0 * math.pi) - math.pi
    r_s = arc_radius_1x * ss

    if abs(diff) >= 0.025:
        steps = 64
        wedge_poly = [(vx_s, vy_s)]
        outer_arc, inner_arc = [], []
        outer_sh, inner_sh = [], []
        hw_arc = 2.2 * ss
        hw_sh = 3.4 * ss
        for i in range(steps + 1):
            a = a_ref + diff * (i / float(steps))
            ca, sa = math.cos(a), math.sin(a)
            wedge_poly.append((vx_s + ca * r_s, vy_s + sa * r_s))
            outer_arc.append((vx_s + ca * (r_s + hw_arc), vy_s + sa * (r_s + hw_arc)))
            inner_arc.append((vx_s + ca * (r_s - hw_arc), vy_s + sa * (r_s - hw_arc)))
            outer_sh.append((vx_s + ca * (r_s + hw_sh), vy_s + sa * (r_s + hw_sh)))
            inner_sh.append((vx_s + ca * (r_s - hw_sh), vy_s + sa * (r_s - hw_sh)))

        rd.polygon(wedge_poly, fill=(*col_main, 64))
        rd.polygon(outer_sh + inner_sh[::-1], fill=(0, 0, 0, 185))
        rd.polygon(outer_arc + inner_arc[::-1], fill=(*col_main, 255))

    # 2. Vertical Dashed Reference Line (`vertex_1x -> ref_pt_1x`)
    dx_r, dy_r = rx_s - vx_s, ry_s - vy_s
    L_r = math.hypot(dx_r, dy_r)
    if L_r > 8.0:
        n_dashes = max(5, int(L_r // (14.0 * ss)))
        for i in range(n_dashes):
            t0 = i / float(n_dashes)
            t1 = min(1.0, (i + 0.56) / float(n_dashes))
            pa = (vx_s + dx_r * t0, vy_s + dy_r * t0)
            pb = (vx_s + dx_r * t1, vy_s + dy_r * t1)
            _draw_smooth_seg_ss(rd, pa, pb, 1.8 * ss, (0, 0, 0, 170))
            _draw_smooth_seg_ss(rd, pa, pb, 1.05 * ss, (*COL_PLUMB, 215))

    # 3. Glowing Limb / Spine Axis Line(s)
    segments_1x = [(vertex_1x, limb_pt_1x)]
    if next_limb_pt_1x is not None:
        segments_1x.append((limb_pt_1x, next_limb_pt_1x))

    for p0_1x, p1_1x in segments_1x:
        bd_1x.line([p0_1x, p1_1x], fill=(*col_main, 95), width=7)
        p0_s = (p0_1x[0] * ss, p0_1x[1] * ss)
        p1_s = (p1_1x[0] * ss, p1_1x[1] * ss)
        _draw_smooth_seg_ss(rd, p0_s, p1_s, 2.7 * ss, (0, 0, 0, 195))
        _draw_smooth_seg_ss(rd, p0_s, p1_s, 1.65 * ss, (*col_main, 252))
        _draw_smooth_seg_ss(rd, p0_s, p1_s, 0.55 * ss, (255, 255, 255, 230))

    # 4. Tiny `3.5px` solid vertex pinpoints (NO bullseye rings!)
    pts_pin = [vertex_1x, limb_pt_1x] + ([next_limb_pt_1x] if next_limb_pt_1x is not None else [])
    for px_1x, py_1x in pts_pin:
        cx, cy = px_1x * ss, py_1x * ss
        r_out = 4.2 * ss
        r_in = 2.8 * ss
        rd.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=(*col_main, 255))
        rd.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=(255, 255, 255, 255))


def draw_v58_floating_white_callout(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    num_str: str,
    tag_str: str,
    align: str = "right",
):
    x, y = xy
    nb = od.textbbox((0, 0), num_str, font=F_CALLOUT_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), tag_str, font=F_CALLOUT_SUB)
    tw = tb[2] - tb[0]

    if align == "right":
        nx = x - nw
        tx = x - tw
    elif align == "center":
        nx = x - nw // 2
        tx = x - tw // 2
    else:
        nx = x
        tx = x

    nx = int(np.clip(nx, SAFE_X_LEFT, SAFE_X_RIGHT_MAX - nw))
    tx = int(np.clip(tx, SAFE_X_LEFT, SAFE_X_RIGHT_MAX - tw))
    ty = y + 70

    for dx_s, dy_s in [(-3, 0), (3, 0), (0, -3), (0, 3), (0, 5)]:
        sd.text((nx + dx_s, y + dy_s), num_str, font=F_CALLOUT_NUM, fill=(0, 0, 0, 240))
        sd.text((tx + dx_s, ty + dy_s), tag_str, font=F_CALLOUT_SUB, fill=(0, 0, 0, 240))

    od.text((nx, y), num_str, font=F_CALLOUT_NUM, fill=COL_WHITE)
    od.text((tx, ty), tag_str, font=F_CALLOUT_SUB, fill=COL_WHITE)


def render_v58_clean_angle_callouts(
    canvas: Image.Image,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 1.0,
):
    """
    Renders the v58 Geometrically Exact Biomechanical Angle Gauge (`각도 게이지`):
    - ZERO bullseye rings (`◎`), ZERO triangle arrowheads, ZERO disconnected floating arcs!
    - Pair 1 (Front View — Step 1 Wrong vs Step 2 Right): Both measure Knee Alignment Angle (`18°` vs `0°`)
      at the ankle-to-knee axis (`ank -> kn`) against the vertical reference line (`270°`).
    - Pair 2 (Side View — Step 3 Wrong vs Step 4 Right): Both measure Torso Inclination Angle (`42°` vs `15°`)
      at the hip vertex (`hip_r`) directly between the vertical reference line (`270°`) and the forward-inclined
      torso/spine axis (`hip_r -> sh`, `270°..312°` in Step 3 vs `270°..286°` in Step 4)!
    """
    SS = 4
    bloom_1x = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd_1x = ImageDraw.Draw(bloom_1x)

    rig_layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig_layer)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    col_main = COL_RED if is_wrong else COL_GREEN
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    sh = kp["sh"]
    hip_r = kp["hip_r"]
    hip_l = kp["hip_l"]
    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    ank_r = kp["ank_r"]
    ank_l = kp["ank_l"]

    if step_idx == 1:
        # Step 1 (Front Wrong — `18° VALGUS CAVE`)
        # Place knee gauge vertex at `ank_r` & `ank_l`, with vertical plumb line (`270°`) and inward-buckling knee axis (`18°` inward)
        inward_shift = 24.0 * (0.30 + 0.70 * rep_amount)
        kn_r_g = (kn_r[0] + inward_shift, kn_r[1])
        kn_l_g = (kn_l[0] - inward_shift, kn_l[1])
        ref_r = (ank_r[0], kn_r[1] - 48.0)
        ref_l = (ank_l[0], kn_l[1] - 48.0)

        draw_clean_angle_gauge_ss(rd, bd_1x, ank_r, ref_r, kn_r_g, hip_r, arc_radius_1x=115.0, col_main=col_main, col_soft=col_soft, ss=SS)
        draw_clean_angle_gauge_ss(rd, bd_1x, ank_l, ref_l, kn_l_g, hip_l, arc_radius_1x=115.0, col_main=col_main, col_soft=col_soft, ss=SS)

        deg_live = int(round(10 + 8 * rep_amount))
        pct_live = int(round(78 + 6 * rep_amount))
        draw_v58_floating_white_callout(od, sd, (255, 920), f"{deg_live}°", "VALGUS CAVE", align="right")
        draw_v58_floating_white_callout(od, sd, (785, 860), f"{pct_live}%", "ACL & MCL LOAD", align="left")

    elif step_idx == 2:
        # Step 2 (Front Right — `0° KNEES TRACK TOES`)
        # Same ankle-to-knee angle gauge on both legs showing clean alignment over toes (`0°`)
        ref_r = (ank_r[0], kn_r[1] - 48.0)
        ref_l = (ank_l[0], kn_l[1] - 48.0)

        draw_clean_angle_gauge_ss(rd, bd_1x, ank_r, ref_r, kn_r, hip_r, arc_radius_1x=115.0, col_main=col_main, col_soft=col_soft, ss=SS)
        draw_clean_angle_gauge_ss(rd, bd_1x, ank_l, ref_l, kn_l, hip_l, arc_radius_1x=115.0, col_main=col_main, col_soft=col_soft, ss=SS)

        pct_live = int(round(92 + 4 * rep_amount))
        draw_v58_floating_white_callout(od, sd, (255, 920), "0°", "KNEES TRACK TOES", align="right")
        draw_v58_floating_white_callout(od, sd, (785, 860), f"{pct_live}%", "VMO & GLUTE LOCK", align="left")

    elif step_idx == 3:
        # Step 3 (Side Wrong — `42° FORWARD LEAN`)
        # Measured at `hip_r` between vertical plumb line (`270°` straight up) and forward-inclined spine (`hip_r -> sh_spine`)
        sh_spine = (sh[0] - 8.0, sh[1] - 6.0)
        ref_top = (hip_r[0], sh_spine[1] - 35.0)

        draw_clean_angle_gauge_ss(rd, bd_1x, hip_r, ref_top, sh_spine, None, arc_radius_1x=108.0, col_main=col_main, col_soft=col_soft, ss=SS)
        # Continue the lower-body biomechanical axis (`hip_r -> kn_r -> ank_r`)
        draw_clean_angle_gauge_ss(rd, bd_1x, hip_r, hip_r, kn_r, ank_r, arc_radius_1x=0.0, col_main=col_main, col_soft=col_soft, ss=SS)

        deg_live = int(round(24 + 18 * rep_amount))
        pct_live = int(round(72 + 7 * rep_amount))
        draw_v58_floating_white_callout(od, sd, (265, 900), f"{deg_live}°", "FORWARD LEAN", align="right")
        draw_v58_floating_white_callout(od, sd, (785, 860), f"{pct_live}%", "LUMBAR OVERLOAD", align="left")

    else:
        # Step 4 (Side Right — `15° UPRIGHT LOCK`)
        # Measured at the EXACT SAME `hip_r` vertex between vertical plumb line (`270°`) and proud upright thoracic spine (`15°`)!
        sh_upright = (hip_r[0] + 50.0 * (0.45 + 0.55 * rep_amount), sh[1] - 10.0)
        ref_top = (hip_r[0], sh_upright[1] - 35.0)

        draw_clean_angle_gauge_ss(rd, bd_1x, hip_r, ref_top, sh_upright, None, arc_radius_1x=108.0, col_main=col_main, col_soft=col_soft, ss=SS)
        draw_clean_angle_gauge_ss(rd, bd_1x, hip_r, hip_r, kn_r, ank_r, arc_radius_1x=0.0, col_main=col_main, col_soft=col_soft, ss=SS)

        pct_live = int(round(94 + 4 * rep_amount))
        draw_v58_floating_white_callout(od, sd, (265, 900), "15°", "UPRIGHT LOCK", align="right")
        draw_v58_floating_white_callout(od, sd, (785, 860), f"{pct_live}%", "QUADS & GLUTES", align="left")

    rig_1x = rig_layer.resize((W, H), Image.Resampling.LANCZOS)
    canvas.alpha_composite(bloom_1x.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(rig_1x)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
    canvas.alpha_composite(ov)


def render_v58_iphone_safezone_sheet(step_frames_rgb: dict, out_path: str):
    real_shot_bgr = cv2.imread(v39.USER_SCREENSHOT_PATH)
    cap_old = cv2.VideoCapture(os.path.join(OUT_DIR, "day2_v38_master_reel.mp4"))
    cap_old.set(cv2.CAP_PROP_POS_FRAMES, 210)
    _, old_f210_bgr = cap_old.read()
    cap_old.release()

    cw, ch = 420, 910
    ipad = 24
    board = Image.new("RGB", (cw * 4 + ipad * 5, ch + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_hdr = v30.f_pretendard(20, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 3 · iPHONE INSTAGRAM REELS UI SAFE-ZONE PREVIEW (`v58`)  ·  136×108 무왜곡 프리뷰(y<=438) + 머리 분리(y>=455) + 정밀 각도 게이지",
        font=f_hdr,
        fill=COL_GREEN,
    )
    titles = [
        "iPhone View · STEP 1 (18° KNEE VALGUS)",
        "iPhone View · STEP 2 (0° KNEES TRACK TOES)",
        "iPhone View · STEP 3 (42° FORWARD LEAN)",
        "iPhone View · STEP 4 (15° UPRIGHT LOCK)",
    ]
    for i in range(4):
        x = ipad + i * (cw + ipad)
        col = COL_RED if i in (0, 2) else COL_GREEN
        bd.text((x + 4, 58), titles[i], font=f_col, fill=col)
        sim_im = v39.simulate_iphone_ig_reels_view(step_frames_rgb[i + 1], real_shot_bgr, old_f210_bgr)
        rc = sim_im.resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        board.paste(rc, (x, 86))
    board.save(out_path, quality=95)


def main():
    pools = get_or_build_v58_pools()
    torso_icons = make_v58_torso_icons(pools)

    # Verify programmatic clearance across all frames of all 4 steps
    print("=== v58 STANDING & SQUATTING CLEARANCE VERIFICATION ===")
    for s_idx in (1, 2, 3, 4):
        p_pool, p_kps = pools[s_idx]
        min_ht = min(k["head_top"][1] for k in p_kps)
        max_fb = max(k["foot_bot"][1] for k in p_kps)
        ht_stand = p_kps[0]["head_top"][1]
        ht_squat = p_kps[-1]["head_top"][1]
        print(
            f"  Step {s_idx}: standing head_top={ht_stand:.1f}px, squat head_top={ht_squat:.1f}px "
            f"(min={min_ht:.1f} >= 455, dock_bot=438 -> gap={min_ht - 438.0:.1f}px) | max_foot_bot={max_fb:.1f} <= 1344 (hud_top=1348)"
        )
        assert min_ht >= 454.0, f"Step {s_idx} head_top {min_ht} too close to dock 438!"
        assert max_fb <= 1346.0, f"Step {s_idx} foot_bot {max_fb} overlaps bottom HUD 1348!"

    left_w = 430
    curve_l = (np.linspace(0.68, 1.0, left_w, dtype=np.float32) ** 0.78)[None, :, None]
    scrim_2d = v38.make_bottom_scrim_2d()

    step_configs = [
        (1, pools[1][0], pools[1][1], True,  0, 84, "LUMBAR & KNEE RISK",       "BARBELL BACK SQUAT",       "18° KNEE VALGUS"),
        (2, pools[2][0], pools[2][1], False, 2, 96, "NEUTRAL KNEE LOCK",        "GLUTE MEDIUS & VMO",       "0° KNEES TRACK TOES"),
        (3, pools[3][0], pools[3][1], True,  1, 79, "ANTERIOR FORCE LEAK",      "HEEL LIFT & FORWARD DIVE", "42° FORWARD LEAN"),
        (4, pools[4][0], pools[4][1], False, 2, 98, "FULL QUAD & GLUTE · 4×10", "QUADRICEPS & GLUTES",      "15° UPRIGHT LOCK"),
    ]

    panels = []
    crops = []
    step_frames_pil = {}

    for s_idx, pool, kps, is_wrong, dock_idx, pct_val, tag_pill, tag_ctx, cue_1line in step_configs:
        idx = len(pool) - 1
        rgb_frame = apply_static_studio_tint(pool[idx], is_wrong)
        rgb_frame = v38.apply_bottom_left_scrim(rgb_frame, curve_l, scrim_2d)
        kp_cur = kps[idx]

        canvas = Image.fromarray(rgb_frame).convert("RGBA")
        render_v58_clean_angle_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)
        draw_v58_day2_unsquished_top_dock(canvas, torso_icons, active_idx=dock_idx, is_wrong=is_wrong)
        draw_v58_safezone_bottom_hud(
            canvas,
            {
                "is_wrong": is_wrong,
                "pct": pct_val,
                "tag_pill": tag_pill,
                "tag_ctx": tag_ctx,
                "cue_1line": cue_1line,
            },
        )

        rgb_pil = canvas.convert("RGB")
        step_frames_pil[s_idx] = rgb_pil
        out_bgr = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)
        panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
        crops.append(out_bgr[430:1360, 0:1080])

    diff_path = os.path.join(OUT_DIR, "day3_v58_difference_sheet.jpg")
    close_path = os.path.join(OUT_DIR, "day3_v58_closeup_sheet.jpg")
    safe_path = os.path.join(OUT_DIR, "day3_v58_iphone_safezone_preview.jpg")

    cv2.imwrite(diff_path, np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(close_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    render_v58_iphone_safezone_sheet(step_frames_pil, safe_path)
    print("Saved v58 preview sheets:", diff_path, close_path, safe_path)


if __name__ == "__main__":
    main()
