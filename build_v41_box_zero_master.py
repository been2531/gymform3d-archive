#!/usr/bin/env python3
"""
v41 Pure Box-Zero Editorial Master (`build_v41_box_zero_master.py`)
===================================================================
Implements all of 중현 님's design & biomechanics instructions:
1. ZERO boxes, ZERO pills, ZERO dark background cards, ZERO sci-fi rings (`AI 티나는 박스나 이런건 버려야해`).
2. ZERO arbitrary degree (`°`) numbers or protractor arcs (`각도같은거 정확하지 않은거면 빼`).
3. Pure Reference-Style Animated Curved Arrow Pointers (`Red Se` style):
   - Clean, bold white floating typography (`QUADRICEPS`, `GLUTEUS MAXIMUS` / `LOWER BACK`, `PATELLAR TENDON`)
     with a single concise color-tinted role line (`PRIMARY TARGET` vs `JOINT OVERLOAD`).
   - Directly beneath the text, a smooth tapered white curved arrow (`╭──→`) dynamically draws on (`0% -> 100%` over `0.45s`)
     and tracks the exact muscle belly or overloaded joint in real time.
4. Smooth Camera Flow & Editorial Cross-Dissolves (`카메라 전환이든 흐름이 깔끔하고 정보성을 줘야해`):
   - Smooth 0.35s cinema cross-dissolve & subtle camera push between the 4 distinct 3D camera angles.
5. Fast localized rendering (completes 600 frames + sheets in ~25 seconds).
"""

import os
import sys
import math
import pickle
import shutil
import asyncio
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import build_v37_minimal_luxury as v37
import build_v38_high_contrast_hud as v38
import build_v39_ig_safe_zone as v39
import build_day3_v39_squat as d3

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_AUDIO = "/workspace/daily_info_reels/_tmp_day3_v41_audio"
CACHE_PKL = "/workspace/daily_info_reels/_cache_day3_v41_pools.pkl"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_AUDIO, exist_ok=True)

W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)

SAFE_X_LEFT = 96
SAFE_Y_TOP = 258

COL_RED = (235, 42, 48)
COL_RED_SOFT = (255, 142, 136)
COL_GREEN = (16, 205, 110)
COL_GREEN_SOFT = (148, 248, 195)
COL_WHITE = (255, 255, 255)

F_EX_TITLE = v30.f_pretendard(26, "extrabold")
F_EX_SUB = v30.f_pretendard(18, "extrabold")

F_CALLOUT_MAIN = v30.f_pretendard(38, "extrabold")
F_CALLOUT_SUB = v30.f_pretendard(17, "extrabold")

F_HUD_TOP = v30.f_pretendard(19, "extrabold")
F_HERO_PCT = v30.f_barlow(84, extrabold=True)
F_HERO_SYM = v30.f_barlow(44, extrabold=True)
F_SINGLE_LINE = v30.f_pretendard(44, "extrabold")


def load_or_build_day3_pools():
    if os.path.exists(CACHE_PKL):
        print("Loading cached Day 3 shaded pools & smoothed keypoints...")
        with open(CACHE_PKL, "rb") as f:
            return pickle.load(f)

    print("Building Day 3 shaded pools & smoothed keypoints (one-time)...")
    raw_s1 = d3._load_clip_frames(d3.DAY3_STEP1_VALGUS_I2V)
    raw_s2 = d3._load_clip_frames(d3.DAY3_STEP2_WRONG_I2V)
    raw_s3 = d3._load_clip_frames(d3.DAY3_STEP3_KNEELOCK_I2V)
    raw_s4 = d3._load_clip_frames(d3.DAY3_STEP4_CORRECT_I2V)

    stroke_s1 = raw_s1[54:119][::-1]
    stroke_s2 = raw_s2[64:121]
    stroke_s3 = raw_s3[56:126][::-1]
    stroke_s4 = raw_s4[90:145]

    data = {}
    for idx, (stroke, is_w) in enumerate([(stroke_s1, True), (stroke_s2, True), (stroke_s3, False), (stroke_s4, False)], start=1):
        pool, kp_raw = [], []
        for fr in stroke:
            rgb, kp = d3.render_day3_i2v_frame_v39(fr, idx, is_w)
            pool.append(rgb)
            kp_raw.append(kp)
        data[idx] = (pool, d3._smooth_kpts_seq(kp_raw))

    with open(CACHE_PKL, "wb") as f:
        pickle.dump(data, f)
    return data


def draw_frameless_top_header_and_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    exercise_name: str = "BARBELL BACK SQUAT",
    muscle_group: str = "QUADS & GLUTES",
):
    """
    100% Box-Zero Top Header & 3-Thumbnail Progression Strip (`x=96, y=258..424`):
    - No dark background rectangle behind the title, no colored pill box!
    - Pure floating typography with crisp drop shadow + 3 clean frameless thumbnails.
    """
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    bx = SAFE_X_LEFT
    hy = SAFE_Y_TOP

    # Glowing status dot + Pure floating Exercise Name + Target Muscle Group
    dot_cx, dot_cy = bx + 8, hy + 15
    sd.ellipse([dot_cx - 8, dot_cy - 8, dot_cx + 8, dot_cy + 8], fill=(0, 0, 0, 220))
    od.ellipse([dot_cx - 8, dot_cy - 8, dot_cx + 8, dot_cy + 8], fill=(*accent, 90))
    od.ellipse([dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5], fill=(*accent, 255))

    tx = dot_cx + 16
    sd.text((tx, hy + 3), exercise_name, font=F_EX_TITLE, fill=(0, 0, 0, 245))
    od.text((tx, hy), exercise_name, font=F_EX_TITLE, fill=(255, 255, 255, 255))
    tb_ex = od.textbbox((0, 0), exercise_name, font=F_EX_TITLE)
    ex_w = tb_ex[2] - tb_ex[0]

    # Subtle vertical hairline divider + colored target group text (NO BOX)
    div_x = tx + ex_w + 14
    sd.line([(div_x, hy + 6), (div_x, hy + 26)], fill=(0, 0, 0, 220), width=2)
    od.line([(div_x, hy + 5), (div_x, hy + 25)], fill=(255, 255, 255, 110), width=2)

    mg_x = div_x + 14
    sd.text((mg_x, hy + 7), muscle_group, font=F_EX_SUB, fill=(0, 0, 0, 245))
    od.text((mg_x, hy + 4), muscle_group, font=F_EX_SUB, fill=(*accent_soft, 255))

    # 3 Frameless Rounded Thumbnails (`y=302..406`) — No outer dark container box!
    tw, th = 128, 100
    gap = 12
    ty = hy + 42
    for i in range(3):
        ux = bx + i * (tw + gap)
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy().resize((tw, th), Image.Resampling.LANCZOS) if i == active_idx else inact_im.copy().resize((tw, th), Image.Resampling.LANCZOS)

        sd.rounded_rectangle([ux + 2, ty + 4, ux + tw + 2, ty + th + 4], radius=14, fill=(0, 0, 0, 180))
        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=255)
        ov.paste(icon_im, (ux, ty), mask)

        if i == active_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle([ux, ty, ux + tw, ty + th], radius=14, outline=(*i_col, 245), width=2)
            # Clean active bar beneath active thumbnail
            od.rounded_rectangle([ux + 28, ty + th + 6, ux + tw - 28, ty + th + 10], radius=2, fill=(*i_col, 255))
        else:
            od.rounded_rectangle([ux, ty, ux + tw, ty + th], radius=14, outline=(255, 255, 255, 35), width=1)

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=5)))
    canvas.alpha_composite(ov)


def build_cached_v41_top_docks(torso_icons) -> dict:
    cache = {}
    for active_slot, is_wrong in [(0, True), (1, True), (2, False)]:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_frameless_top_header_and_dock(layer, torso_icons, active_slot, is_wrong)
        cache[(active_slot, is_wrong)] = layer
    return cache


def draw_frameless_bottom_hud(canvas: Image.Image, sp: dict):
    """
    100% Box-Zero Bottom-Left Telemetry HUD (`x=96, y=1306..1438`):
    - NO solid colored pill box!
    - Row 1 (`y=1308`): Glowing LED dot + pure colored status/target text (`JOINT OVERLOAD  ·  MCL & ACL LIGAMENTS`
      or `PRIMARY TARGET  ·  QUADRICEPS & GLUTES  ·  4×10`).
    - Row 2 (`y=1342`): Huge `%` + 3px vertical bar + 1-line bold hero cue (`DEEP SQUAT DRIVE`).
    - Row 3 (`y=1432`): Single 4px slim progress line.
    """
    is_wrong = sp["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT
    pct = int(np.clip(sp["pct"], 0, 100))

    top_status = sp["top_status"]
    top_target = sp["top_target"]
    cue_1line = sp["cue_1line"]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx = SAFE_X_LEFT  # 96px

    # Row 1: Pure Floating Status & Target Muscle Line (ZERO BOX)
    py = 1308
    dot_cx, dot_cy = bx + 7, py + 11
    sd.ellipse([dot_cx - 6, dot_cy - 6, dot_cx + 6, dot_cy + 6], fill=(0, 0, 0, 220))
    od.ellipse([dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5], fill=(*accent, 255))

    st_x = dot_cx + 13
    sd.text((st_x, py + 3), top_status, font=F_HUD_TOP, fill=(0, 0, 0, 245))
    od.text((st_x, py), top_status, font=F_HUD_TOP, fill=(*accent_soft, 255))
    tb_st = od.textbbox((0, 0), top_status, font=F_HUD_TOP)
    st_w = tb_st[2] - tb_st[0]

    sep_str = "  ·  "
    sep_x = st_x + st_w
    od.text((sep_x, py), sep_str, font=F_HUD_TOP, fill=(255, 255, 255, 130))
    tb_sep = od.textbbox((0, 0), sep_str, font=F_HUD_TOP)
    tg_x = sep_x + (tb_sep[2] - tb_sep[0])

    sd.text((tg_x, py + 3), top_target, font=F_HUD_TOP, fill=(0, 0, 0, 245))
    od.text((tg_x, py), top_target, font=F_HUD_TOP, fill=(238, 243, 250, 255))

    # Row 2: Huge `%` + 3px Vertical Bar + Single-Line Bold Hero Cue (`y=1342`)
    by = 1342
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
    div_y0, div_y1 = by + 18, by + 68
    sd.rounded_rectangle([div_x, div_y0 + 2, div_x + 3, div_y1 + 2], radius=2, fill=(0, 0, 0, 225))
    od.rounded_rectangle([div_x, div_y0, div_x + 3, div_y1], radius=2, fill=(*accent, 255))

    tx = div_x + 18
    ty = by + 18
    tb_c = od.textbbox((0, 0), cue_1line, font=F_SINGLE_LINE)
    cw = tb_c[2] - tb_c[0]

    sd.text((tx, ty + 4), cue_1line, font=F_SINGLE_LINE, fill=(0, 0, 0, 245))
    od.text((tx, ty), cue_1line, font=F_SINGLE_LINE, fill=(255, 255, 255, 255))

    # Row 3: Single 4px Slim Glowing Line (`y=1432`)
    bar_x0 = bx + 2
    bar_y0 = by + 90
    total_row_w = (tx + cw) - bar_x0
    bar_w = max(350, min(510, total_row_w))
    bar_h = 4
    fill_w = max(14, int(round(bar_w * (pct / 100.0))))

    sd.rounded_rectangle([bar_x0, bar_y0 + 2, bar_x0 + bar_w, bar_y0 + bar_h + 2], radius=2, fill=(0, 0, 0, 210))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], radius=2, fill=(255, 255, 255, 38))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y0 + bar_h], radius=2, fill=(*accent, 248))

    dot_x = bar_x0 + fill_w
    dot_y = bar_y0 + bar_h * 0.5
    sd.ellipse([dot_x - 7, dot_y - 7, dot_x + 7, dot_y + 7], fill=(*accent, 165))
    od.ellipse([dot_x - 4, dot_y - 4, dot_x + 4, dot_y + 4], fill=(255, 255, 255, 255))

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(ov)


def _bezier_pt(p0, p1, p2, t: float) -> tuple[float, float]:
    u = 1.0 - t
    return (
        u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
    )


def draw_pure_curved_arrow_callout(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    anchor_xy: tuple[int, int],
    target_xy: tuple[float, float],
    main_title: str,
    sub_label: str,
    is_wrong: bool,
    align: str = "left",
    t_local: float = 2.0,
):
    """
    Pure Reference-Style Animated Curved Arrow (`Red Se` style, 100% Box-Zero):
    - NO background box, NO pill, NO rings!
    - Bold white anatomical name (`QUADRICEPS`, `GLUTEUS MAXIMUS` / `LOWER BACK`, `PATELLAR TENDON`)
      + compact 1-line colored subtitle (`PRIMARY TARGET` / `JOINT OVERLOAD`).
    - Smoothly draws on (`0% -> 100%` over `0.45s` with cubic ease-out) along a graceful quadratic Bezier curve
      ending in a crisp open chevron arrowhead pointing directly at the muscle belly or joint.
    """
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    # Smooth fade/slide-in at start of step (0.05..0.35s) and gentle fade-out at very end (4.65..4.95s)
    fade_in = float(np.clip((t_local - 0.04) / 0.28, 0.0, 1.0))
    fade_out = float(np.clip((4.92 - t_local) / 0.25, 0.0, 1.0))
    alpha_f = fade_in * fade_out
    if alpha_f <= 0.02:
        return

    slide_dy = int(round((1.0 - fade_in) * 12.0))
    arrow_prog = float(np.clip((t_local - 0.12) / 0.46, 0.0, 1.0))
    arrow_prog = 1.0 - (1.0 - arrow_prog) ** 3  # Cubic ease-out draw-on

    ax, ay = anchor_xy[0], anchor_xy[1] + slide_dy
    tb_m = od.textbbox((0, 0), main_title, font=F_CALLOUT_MAIN)
    mw, mh = tb_m[2] - tb_m[0], tb_m[3] - tb_m[1]
    tb_s = od.textbbox((0, 0), sub_label, font=F_CALLOUT_SUB)
    sw, sh = tb_s[2] - tb_s[0], tb_s[3] - tb_s[1]

    a_txt = int(round(255 * alpha_f))
    a_sh = int(round(240 * alpha_f))

    # 1. Bold White Muscle/Joint Name (NO BOX)
    mx = ax if align == "left" else (ax - mw)
    sd.text((mx, ay + 4), main_title, font=F_CALLOUT_MAIN, fill=(0, 0, 0, a_sh))
    od.text((mx, ay), main_title, font=F_CALLOUT_MAIN, fill=(255, 255, 255, a_txt))

    # 2. Clean 1-Line Color Subtitle (`PRIMARY TARGET` / `JOINT OVERLOAD`)
    sy = ay + mh + 7
    sx = ax if align == "left" else (ax - sw)
    sd.text((sx, sy + 3), sub_label, font=F_CALLOUT_SUB, fill=(0, 0, 0, a_sh))
    od.text((sx, sy), sub_label, font=F_CALLOUT_SUB, fill=(*col_soft, a_txt))

    # 3. Smooth Tapered White Curved Arrow (`╭──→`) from beneath the text to `target_xy`
    if arrow_prog <= 0.04:
        return

    tail_y = sy + sh + 12
    if align == "left":
        p0 = (ax + min(mw, 150) * 0.48, tail_y)
        p2 = (target_xy[0] - 10.0, target_xy[1] - 2.0)
        p1 = (p0[0] + (p2[0] - p0[0]) * 0.25, p0[1] + (p2[1] - p0[1]) * 0.72 + 22.0)
    else:
        p0 = (ax - min(mw, 150) * 0.48, tail_y)
        p2 = (target_xy[0] + 10.0, target_xy[1] - 2.0)
        p1 = (p0[0] + (p2[0] - p0[0]) * 0.25, p0[1] + (p2[1] - p0[1]) * 0.72 + 22.0)

    n_seg = 36
    max_i = max(2, int(round(n_seg * arrow_prog)))
    pts = [_bezier_pt(p0, p1, p2, i / float(n_seg)) for i in range(max_i + 1)]

    # Draw tapered curve segments
    for i in range(len(pts) - 1):
        u = i / float(n_seg)
        w_line = int(round(3.5 + 1.8 * math.sin(u * math.pi * 0.85)))
        pa, pb = pts[i], pts[i + 1]
        sd.line([(pa[0], pa[1] + 4), (pb[0], pb[1] + 4)], fill=(0, 0, 0, a_sh), width=w_line + 3)
        od.line([pa, pb], fill=(255, 255, 255, a_txt), width=w_line)

    # Open V-chevron arrowhead at leading tip (`pts[-1]`) exactly like user's `Red Se` reference
    tip = pts[-1]
    prev = pts[max(0, len(pts) - 3)]
    ang = math.atan2(tip[1] - prev[1], tip[0] - prev[0])
    hl = 21.0 * min(1.0, arrow_prog * 2.5)
    spread = math.radians(32.0)

    w1 = (tip[0] - hl * math.cos(ang - spread), tip[1] - hl * math.sin(ang - spread))
    w2 = (tip[0] - hl * math.cos(ang + spread), tip[1] - hl * math.sin(ang + spread))

    sd.line([(w1[0], w1[1] + 4), (tip[0], tip[1] + 4), (w2[0], w2[1] + 4)], fill=(0, 0, 0, a_sh), width=7, joint="curve")
    od.line([w1, tip, w2], fill=(255, 255, 255, a_txt), width=5, joint="curve")


def render_v41_step_callouts(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
    kp: dict,
    is_wrong: bool,
    t_local: float,
):
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    if step_idx == 1:
        # Step 1 (Direct Front View — Knee Valgus Collapse)
        pt_left = map_pt(kp["kn_r"][0] + 6.0, kp["kn_r"][1] - 4.0)
        pt_right = map_pt(kp["kn_l"][0] - 6.0, kp["kn_l"][1] - 4.0)
        draw_pure_curved_arrow_callout(
            od, sd,
            anchor_xy=(102, 855),
            target_xy=pt_left,
            main_title="MCL & MENISCUS",
            sub_label="INWARD KNEE BUCKLE",
            is_wrong=True,
            align="left",
            t_local=t_local,
        )
        draw_pure_curved_arrow_callout(
            od, sd,
            anchor_xy=(962, 775),
            target_xy=pt_right,
            main_title="ACL LIGAMENT",
            sub_label="JOINT OVERLOAD",
            is_wrong=True,
            align="right",
            t_local=t_local,
        )

    elif step_idx == 2:
        # Step 2 (3/4 Side Profile — Heel Lift & Forward Lean)
        pt_left = map_pt(kp["hip"][0] + 15.0, kp["hip"][1] - 78.0)
        pt_right = map_pt(kp["kn_r"][0] + 14.0, kp["kn_r"][1] - 6.0)
        draw_pure_curved_arrow_callout(
            od, sd,
            anchor_xy=(102, 755),
            target_xy=pt_left,
            main_title="LOWER BACK",
            sub_label="L4-L5 SPINAL STRAIN",
            is_wrong=True,
            align="left",
            t_local=t_local,
        )
        draw_pure_curved_arrow_callout(
            od, sd,
            anchor_xy=(962, 785),
            target_xy=pt_right,
            main_title="KNEE TENDON",
            sub_label="ANTERIOR SHEAR LOAD",
            is_wrong=True,
            align="right",
            t_local=t_local,
        )

    elif step_idx == 3:
        # Step 3 (Lower-Body Medium Close-Up — Knees Track Toes)
        pt_left = map_pt(kp["kn_r"][0] - 15.0, kp["kn_r"][1] - 92.0)
        pt_right = map_pt(kp["kn_l"][0] - 35.0, kp["kn_l"][1] - 82.0)
        draw_pure_curved_arrow_callout(
            od, sd,
            anchor_xy=(102, 610),
            target_xy=pt_left,
            main_title="VMO & OUTER QUAD",
            sub_label="STABILIZER LOCK-IN",
            is_wrong=False,
            align="left",
            t_local=t_local,
        )
        draw_pure_curved_arrow_callout(
            od, sd,
            anchor_xy=(962, 610),
            target_xy=pt_right,
            main_title="ADDUCTOR & GLUTE",
            sub_label="EXTERNAL HIP TORQUE",
            is_wrong=False,
            align="right",
            t_local=t_local,
        )

    else:
        # Step 4 (Full-Body Wide 3/4 View — Deep Squat Drive)
        pt_left = map_pt(kp["hip"][0] - 18.0, kp["hip"][1] + 15.0)
        pt_right = map_pt((kp["hip"][0] + kp["kn_r"][0]) * 0.5 + 22.0, (kp["hip"][1] + kp["kn_r"][1]) * 0.5 - 12.0)
        draw_pure_curved_arrow_callout(
            od, sd,
            anchor_xy=(102, 765),
            target_xy=pt_left,
            main_title="GLUTEUS MAXIMUS",
            sub_label="HIP EXTENSION DRIVE",
            is_wrong=False,
            align="left",
            t_local=t_local,
        )
        draw_pure_curved_arrow_callout(
            od, sd,
            anchor_xy=(962, 780),
            target_xy=pt_right,
            main_title="QUADRICEPS",
            sub_label="PRIMARY THIGH SWEEP",
            is_wrong=False,
            align="right",
            t_local=t_local,
        )

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=5)))
    canvas.alpha_composite(ov)


def sample_step_camera_plate(
    step_idx: int,
    t_local: float,
    pools_data: dict,
) -> tuple[np.ndarray, callable, dict, float, int]:
    """
    Samples the 3D mannequin plate & cinema camera transform for a given step (`1..4`) and local time (`0..5.0s`),
    with smooth progressive cinema push-in for natural editorial flow.
    """
    pool, smooth_kp = pools_data[step_idx]
    u = float(np.clip(t_local / 5.0, 0.0, 1.0))
    rep_w = v37.controlled_single_rep_wave(u)
    pos_f = rep_w * (len(pool) - 1)
    base_rgb = v37.sample_pool_interpolated(pool, pos_f)
    kp = v37.sample_kpts_interpolated(smooth_kp, pos_f)

    # Smooth editorial camera drift + contraction zoom
    drift = u * 0.018
    if step_idx == 1:
        cam_zoom = 1.015 + drift + 0.022 * rep_w
        cam_focus, cam_dest = (540.0, 980.0), (550.0, 980.0)
        pct = int(round(78 + 6 * rep_w))
    elif step_idx == 2:
        cam_zoom = 1.020 + drift + 0.022 * rep_w
        cam_focus, cam_dest = (575.0, 970.0), (585.0, 970.0)
        pct = int(round(72 + 7 * rep_w))
    elif step_idx == 3:
        cam_zoom = 1.015 + drift + 0.022 * rep_w
        cam_focus, cam_dest = (560.0, 940.0), (570.0, 940.0)
        pct = int(round(92 + 4 * rep_w))
    else:
        cam_zoom = 1.020 + drift + 0.025 * rep_w
        cam_focus, cam_dest = (580.0, 965.0), (590.0, 965.0)
        pct = int(round(94 + 4 * rep_w))

    cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
    return cam_rgb, map_pt, kp, rep_w, pct


def build_v41_master_reel():
    pools_data = load_or_build_day3_pools()
    pool_s1, smooth_kp_s1 = pools_data[1]
    pool_s2, smooth_kp_s2 = pools_data[2]
    pool_s3, smooth_kp_s3 = pools_data[3]
    pool_s4, smooth_kp_s4 = pools_data[4]

    size = (128, 100)
    im0_act = Image.fromarray(pool_s1[-1]).convert("RGBA").crop((220, 740, 860, 1480)).resize(size, Image.Resampling.LANCZOS)
    im1_act = Image.fromarray(pool_s2[-1]).convert("RGBA").crop((310, 680, 910, 1460)).resize(size, Image.Resampling.LANCZOS)
    im2_act = Image.fromarray(pool_s4[-1]).convert("RGBA").crop((310, 680, 910, 1460)).resize(size, Image.Resampling.LANCZOS)

    def _make_inact(act_im: Image.Image) -> Image.Image:
        out = act_im.copy()
        out.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 142)))
        return out

    torso_icons = [
        (im0_act, _make_inact(im0_act)),
        (im1_act, _make_inact(im1_act)),
        (im2_act, _make_inact(im2_act)),
    ]
    dock_cache = build_cached_v41_top_docks(torso_icons)
    hud_cache = {}

    wav_day3 = os.path.join(TMP_AUDIO, "day3_v41_audio.wav")
    vo_day3 = [
        (0.12, "Stop letting your knees cave inward and overloading your knee ligaments."),
        (5.12, "Lifting your heels pitches you forward and strains your lower back."),
        (10.12, "Root your feet and track knees over toes to fire your quads and glutes."),
        (15.12, "Drop to full depth and drive up for maximum quad and glute growth."),
    ]
    asyncio.run(d3.synthesize_day3_voiceover_wav(wav_day3, vo_day3, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day3_v41_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day3_v41_box_zero_master.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]
    scrim_2d = v38.make_bottom_scrim_2d()

    step_meta = {
        1: (True, 0, "JOINT OVERLOAD", "MCL · MENISCUS · ACL", "KNEE VALGUS CAVE"),
        2: (True, 1, "SPINE & KNEE LOAD", "LOWER BACK & PATELLA", "HEELS LIFTING UP"),
        3: (False, 2, "TARGET MUSCLES", "VMO · OUTER QUAD · GLUTE", "KNEES TRACK TOES"),
        4: (False, 2, "PRIMARY TARGET", "QUADRICEPS & GLUTES · 4×10", "DEEP SQUAT DRIVE"),
    }

    step_static_frames = {}
    anim_timeline_frames = {}

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            step_idx, t_local = 1, t
        elif t < 10.0:
            step_idx, t_local = 2, t - 5.0
        elif t < 15.0:
            step_idx, t_local = 3, t - 10.0
        else:
            step_idx, t_local = 4, t - 15.0

        cam_rgb, map_pt, kp, rep_w, pct = sample_step_camera_plate(step_idx, t_local, pools_data)

        # Smooth 0.35s camera cross-dissolve across step transitions (`t_local < 0.35s` for steps 2, 3, 4)
        if step_idx > 1 and t_local < 0.35:
            alpha_tr = float(np.clip(t_local / 0.35, 0.0, 1.0))
            alpha_tr = alpha_tr * alpha_tr * (3.0 - 2.0 * alpha_tr)  # Smoothstep
            prev_rgb, _, _, _, _ = sample_step_camera_plate(step_idx - 1, 5.0, pools_data)
            cam_rgb = np.clip(prev_rgb.astype(np.float32) * (1.0 - alpha_tr) + cam_rgb.astype(np.float32) * alpha_tr, 0, 255).astype(np.uint8)

        clean_rgb = v38.apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)
        im = Image.fromarray(clean_rgb).convert("RGBA")

        is_wrong, active_slot, top_status, top_target, cue_1line = step_meta[step_idx]
        render_v41_step_callouts(im, step_idx, map_pt, kp, is_wrong, t_local=t_local)
        im.alpha_composite(dock_cache[(active_slot, is_wrong)])

        hud_key = (is_wrong, pct, top_status, top_target, cue_1line)
        if hud_key not in hud_cache:
            h_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_frameless_bottom_hud(
                h_layer,
                {
                    "is_wrong": is_wrong,
                    "pct": pct,
                    "top_status": top_status,
                    "top_target": top_target,
                    "cue_1line": cue_1line,
                },
            )
            hud_cache[hud_key] = h_layer
        im.alpha_composite(hud_cache[hud_key])

        rgb_frame = im.convert("RGB")
        if f_idx in (int(2.2 * FPS), int(7.2 * FPS), int(12.2 * FPS), int(17.2 * FPS)):
            step_static_frames[step_idx] = rgb_frame.copy()
        if step_idx == 4 and f_idx in (int(15.22 * FPS), int(15.42 * FPS), int(17.20 * FPS)):
            anim_timeline_frames[f_idx] = rgb_frame.copy()

        vw.write(cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR))

    vw.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-threads", "4",
            "-i", tmp_mp4, "-i", wav_day3,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_mp4)
    shutil.rmtree(TMP_AUDIO, ignore_errors=True)
    shutil.copyfile(final_mp4, os.path.join(OUT_DIR, "day3_v39_master_reel.mp4"))

    # Save 4-Step Master Sheet (`day3_v41_box_zero_master_sheet.jpg`)
    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 3 · BARBELL BACK SQUAT (v41 PURE BOX-ZERO MASTER)  ·  박스/캡슐/각도(°) 0% · 드로우온 곡선 포인터 & 스무스 카메라 전환",
        font=f_row,
        fill=COL_GREEN,
    )
    col_titles = [
        "STEP 1 (전면 뷰) · MCL & ACL 무릎 내반 과부하",
        "STEP 2 (측면 뷰) · LOWER BACK & KNEE TENDON",
        "STEP 3 (하체 근접) · VMO & ADDUCTOR 타겟 정렬",
        "STEP 4 (전신 풀샷) · GLUTE MAX & QUADRICEPS 타격",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = COL_RED if i < 2 else COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = step_static_frames[i + 1].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=col, width=2)
        board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day3_v41_box_zero_master_sheet.jpg")
    board.save(sheet_path, quality=95)
    print("Saved v41 Box-Zero Master Video:", final_mp4)
    print("Saved v41 Box-Zero Master Sheet:", sheet_path)


if __name__ == "__main__":
    build_v41_master_reel()
