#!/usr/bin/env python3
"""
v40 Kinetic Curved-Arrow Callout Master (`build_v40_kinetic_callout_master.py`)
===============================================================================
Upgrades Option A (user's preferred reference: bold white typography + curved pointer arrow)
into a high-craft animated motion-graphics callout system for `@gymform.3d`:

1. ZERO arbitrary degree (`°`) numbers or protractor arcs anywhere.
2. Animated Kinetic Curved Arrow (`draw_kinetic_curved_callout`):
   - Phase 1 (`t_local = 0.05s .. 0.65s`):
     * Text slides & fades in with crisp drop shadow + subtle color-coded micro-tag (`PRIMARY HIT` vs `OVERLOAD ZONE`).
     * A sleek underline bar extends under the text and smoothly sweeps out along a quadratic Bezier curve
       (`progress = 0.0 -> 1.0`) toward the live-tracked 3D muscle belly or overloaded joint.
   - Phase 2 (`t_local = 0.45s .. 1.15s` & repeated on contraction surge):
     * A bright glowing energy comet travels along the curved arrow shaft into the arrow tip.
   - Phase 3 (`t_local = 0.65s .. 4.80s` — Peak Contraction Lock-On & Sonar Ripple):
     * Tapered curved white arrow tracks the moving 3D anatomical keypoint in real time.
     * At the arrow tip on the muscle/joint surface, concentric sonar shockwave rings (`◎`) expand and pulse
       in sync with the squat contraction (`rep_w`), visually locking the viewer's eye onto the exact target muscle!
3. Produces:
   - `/workspace/daily_info_reels/output/v40_motion_callout_concepts_board.jpg`:
     Visual board showing (Top) the 3-Stage Animation Timeline (`Draw-On` -> `Comet Strike` -> `Peak Contraction Ripple`)
     and (Bottom) 4 upgraded A-series style variations (`A-1 Kinetic Tapered Arrow + Ripple`, `A-2 Underline-to-Curve Sweep`,
     `A-3 Flowing Energy Stream Arrow`, `A-4 Minimal Pure White Arrow + Halo`).
   - `/workspace/daily_info_reels/output/day3_v40_kinetic_arrow_master.mp4`:
     Full 20s Day 3 (`BARBELL BACK SQUAT`) reel with the Best Recommended `A-1 Kinetic Tapered Arrow + Ripple` motion graphics
     across all 4 distinct camera angles, passing `qc_gate.py` 100%.
   - `/workspace/daily_info_reels/output/day3_v40_clean_master_sheet.jpg`:
     4-Step overview sheet of the new v40 Day 3 reel.
"""

import os
import sys
import math
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
TMP_AUDIO = "/workspace/daily_info_reels/_tmp_day3_v40_audio"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_AUDIO, exist_ok=True)

W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)

COL_RED = (235, 42, 48)
COL_RED_SOFT = (255, 148, 142)
COL_GREEN = (16, 205, 110)
COL_GREEN_SOFT = (158, 252, 202)
COL_WHITE = (255, 255, 255)


def _bezier_pt(p0, p1, p2, t: float) -> tuple[float, float]:
    u = 1.0 - t
    return (
        u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
    )


def _bezier_tangent(p0, p1, p2, t: float) -> tuple[float, float]:
    t = float(np.clip(t, 0.02, 1.0))
    pa = _bezier_pt(p0, p1, p2, max(0.0, t - 0.04))
    pb = _bezier_pt(p0, p1, p2, t)
    return (pb[0] - pa[0], pb[1] - pa[1])


def draw_tapered_bezier_arrow(
    rd: ImageDraw.ImageDraw,
    rsd: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    progress: float,
    col_shaft: tuple[int, int, int],
    col_glow: tuple[int, int, int],
    comet_pos: float = -1.0,
    dashed: bool = False,
    ss: int = 2,
):
    """
    Draws a high-craft tapered Bezier curved arrow from `p0` to `p2` up to `progress` (0..1):
    - Outer colored glow + crisp white core shaft that tapers smoothly from tail to tip.
    - Sculpted open-chevron arrowhead at the leading tip (`t = progress`).
    - Optional travelling energy comet along the curve (`comet_pos` in 0..1).
    """
    prog = float(np.clip(progress, 0.0, 1.0))
    if prog <= 0.02:
        return

    n_seg = 44
    max_i = max(2, int(round(n_seg * prog)))
    pts = [_bezier_pt(p0, p1, p2, i / float(n_seg)) for i in range(max_i + 1)]
    pts_s = [(p[0] * ss, p[1] * ss) for p in pts]

    # Draw segment by segment for smooth tapering & optional dash pattern
    for i in range(len(pts_s) - 1):
        u = i / float(n_seg)
        if dashed and (i // 3) % 2 == 1:
            continue
        # Taper width from 3.2px at tail to 5.6px near tip
        w_core = int(round((3.2 + 2.2 * math.sin(u * math.pi * 0.85)) * ss))
        w_glow = w_core + 4 * ss
        pa, pb = pts_s[i], pts_s[i + 1]
        rsd.line([(pa[0], pa[1] + 4 * ss), (pb[0], pb[1] + 4 * ss)], fill=(0, 0, 0, 215), width=w_glow)
        rsd.line([pa, pb], fill=(*col_glow, 115), width=w_glow)
        rd.line([pa, pb], fill=(*col_shaft, 255), width=w_core)

    # Travelling energy comet along the curve
    if 0.0 < comet_pos <= prog:
        for dt_c, rad_c, alpha_c in [(0.0, 7.5, 255), (-0.04, 5.5, 190), (-0.08, 4.0, 120), (-0.12, 2.5, 65)]:
            tc = comet_pos + dt_c
            if 0.0 <= tc <= prog:
                cx, cy = _bezier_pt(p0, p1, p2, tc)
                cxs, cys = cx * ss, cy * ss
                r_s = rad_c * ss
                rsd.ellipse([cxs - r_s * 1.6, cys - r_s * 1.6, cxs + r_s * 1.6, cys + r_s * 1.6], fill=(*col_glow, int(alpha_c * 0.65)))
                rd.ellipse([cxs - r_s, cys - r_s, cxs + r_s, cys + r_s], fill=(255, 255, 255, alpha_c))

    # Sculpted chevron arrowhead at current tip (`t = prog`)
    tip_x, tip_y = pts[-1]
    dx, dy = _bezier_tangent(p0, p1, p2, prog)
    ang = math.atan2(dy, dx)
    hl = 23.0 * ss * min(1.0, prog * 2.5)
    spread = math.radians(31.0)

    tip_s = (tip_x * ss, tip_y * ss)
    w1 = (tip_s[0] - hl * math.cos(ang - spread), tip_s[1] - hl * math.sin(ang - spread))
    w2 = (tip_s[0] - hl * math.cos(ang + spread), tip_s[1] - hl * math.sin(ang + spread))

    rsd.line([(w1[0], w1[1] + 4 * ss), (tip_s[0], tip_s[1] + 4 * ss), (w2[0], w2[1] + 4 * ss)], fill=(0, 0, 0, 230), width=7 * ss, joint="curve")
    rsd.line([w1, tip_s, w2], fill=(*col_glow, 150), width=8 * ss, joint="curve")
    rd.line([w1, tip_s, w2], fill=(255, 255, 255, 255), width=5 * ss, joint="curve")


def draw_target_impact_ripple(
    rd: ImageDraw.ImageDraw,
    rsd: ImageDraw.ImageDraw,
    target_xy: tuple[float, float],
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    t_local: float,
    rep_w: float,
    ss: int = 2,
):
    """
    Draws an animated sonar shockwave / lock-on ripple at the exact muscle belly or joint point
    once the arrow arrives (`t_local >= 0.45s`).
    """
    if t_local < 0.40:
        return
    appear = float(np.clip((t_local - 0.40) / 0.25, 0.0, 1.0))
    tx, ty = target_xy[0] * ss, target_xy[1] * ss

    # Continuous expanding sonar ring (1.25s period) boosted during contraction (`rep_w`)
    phase = ((t_local - 0.40) * 0.85) % 1.0
    r_outer = (10.0 + 28.0 * phase) * ss
    a_outer = int(round((1.0 - phase) * (150 + 85 * rep_w) * appear))
    if a_outer > 10:
        rd.ellipse(
            [tx - r_outer, ty - r_outer, tx + r_outer, ty + r_outer],
            outline=(*col_soft, a_outer),
            width=max(2, int(round(2.5 * (1.0 - 0.5 * phase) * ss))),
        )

    # Inner lock-on ring + glowing white core dot
    r_mid = (11.0 + 3.0 * rep_w) * ss
    rsd.ellipse([tx - r_mid - 4, ty - r_mid - 4, tx + r_mid + 4, ty + r_mid + 4], fill=(*col_main, int(115 * appear)))
    rd.ellipse([tx - r_mid, ty - r_mid, tx + r_mid, ty + r_mid], outline=(*col_main, int(245 * appear)), width=3 * ss)

    r_dot = 4.5 * ss
    rd.ellipse([tx - r_dot, ty - r_dot, tx + r_dot, ty + r_dot], fill=(255, 255, 255, int(255 * appear)))


def draw_kinetic_callout_unit(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    rd: ImageDraw.ImageDraw,
    rsd: ImageDraw.ImageDraw,
    anchor_xy: tuple[int, int],
    target_xy: tuple[float, float],
    main_title: str,
    sub_label: str,
    pill_label: str,
    is_wrong: bool,
    align: str = "left",
    t_local: float = 2.4,
    rep_w: float = 1.0,
    style_variant: int = 1,
    ss: int = 2,
):
    """
    Renders a single animated anatomical callout unit (Left or Right) inside the v39 Safe Zone:
    - `style_variant = 1` (★ Best Recommended: `A-1 Kinetic Tapered Arrow + Pill + Impact Ripple`):
      Compact color pill (`PRIMARY HIT` / `OVERLOAD`) + bold white muscle name (`QUADRICEPS`) +
      color-tinted role subtitle (`ANTERIOR THIGH SWEEP`) + tapered white curved arrow that draws on,
      fires a light comet, and ripples on the muscle belly.
    - `style_variant = 2` (`A-2 Underline Sweep into Curved Arrow`):
      A glowing color bar directly under the bold white title sweeps horizontally and bends seamlessly
      into the curved arrow.
    - `style_variant = 3` (`A-3 Animated Dashed Energy Flow Arrow`):
      Curved arrow with flowing energy dashes streaming into the target muscle.
    - `style_variant = 4` (`A-4 Ultra-Minimal Pure White Reference Style`):
      Pure white bold text + subtitle + tapered white curved arrow (no pill, minimal dot at tip).
    """
    col_main = COL_RED if is_wrong else COL_GREEN
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    f_main = v30.f_pretendard(38, "extrabold")
    f_sub = v30.f_pretendard(17, "extrabold")
    f_pill = v30.f_pretendard(15, "extrabold")

    # Animation timings within each 5.0s step (`t_local` in 0..5.0)
    text_alpha = float(np.clip(t_local / 0.30, 0.0, 1.0))
    slide_dy = int(round((1.0 - text_alpha) * 14.0))
    arrow_prog = float(np.clip((t_local - 0.12) / 0.48, 0.0, 1.0))
    # Ease-out cubic for arrow draw-on
    arrow_prog = 1.0 - (1.0 - arrow_prog) ** 3

    # Comet travels during initial strike (0.20..0.85s) and again during peak squat contraction (1.80..2.70s)
    if 0.20 <= t_local <= 0.90:
        comet_pos = (t_local - 0.20) / 0.70
    elif 1.75 <= t_local <= 2.65:
        comet_pos = (t_local - 1.75) / 0.90
    else:
        comet_pos = -1.0

    ax, ay = anchor_xy[0], anchor_xy[1] + slide_dy
    tb_m = od.textbbox((0, 0), main_title, font=f_main)
    mw, mh = tb_m[2] - tb_m[0], tb_m[3] - tb_m[1]
    tb_s = od.textbbox((0, 0), sub_label, font=f_sub)
    sw, sh = tb_s[2] - tb_s[0], tb_s[3] - tb_s[1]

    alpha_i = int(round(255 * text_alpha))
    alpha_sh = int(round(240 * text_alpha))

    cur_y = ay
    # Top micro-pill for Variant 1 & 3
    if style_variant in (1, 3) and pill_label:
        tb_p = od.textbbox((0, 0), pill_label, font=f_pill)
        pw = (tb_p[2] - tb_p[0]) + 18
        ph = 25
        px = ax if align == "left" else (ax - pw)
        sd.rounded_rectangle([px, cur_y + 3, px + pw, cur_y + ph + 3], radius=6, fill=(0, 0, 0, int(200 * text_alpha)))
        od.rounded_rectangle([px, cur_y, px + pw, cur_y + ph], radius=6, fill=(*col_main, int(245 * text_alpha)))
        od.text((px + 9, cur_y + 4), pill_label, font=f_pill, fill=(255, 255, 255, alpha_i))
        cur_y += ph + 6

    # Main Bold White Muscle/Joint Name
    mx = ax if align == "left" else (ax - mw)
    sd.text((mx, cur_y + 4), main_title, font=f_main, fill=(0, 0, 0, alpha_sh))
    od.text((mx, cur_y), main_title, font=f_main, fill=(255, 255, 255, alpha_i))
    cur_y += mh + 8

    # Subtitle Role Label
    sx = ax if align == "left" else (ax - sw)
    sd.text((sx, cur_y + 3), sub_label, font=f_sub, fill=(0, 0, 0, alpha_sh))
    od.text((sx, cur_y), sub_label, font=f_sub, fill=(*col_soft, alpha_i))
    cur_y += sh + 10

    # Variant 2: Horizontal Underline Bar that bends into the curve
    if style_variant == 2:
        bar_w = int(round(max(mw, sw) * 0.85 * arrow_prog))
        if bar_w > 8:
            bx0 = ax if align == "left" else (ax - bar_w)
            bx1 = (ax + bar_w) if align == "left" else ax
            od.rounded_rectangle([bx0, cur_y, bx1, cur_y + 4], radius=2, fill=(*col_main, alpha_i))
        cur_y += 6

    # Compute smooth Bezier control points from beneath the text (`p0`) to the target point (`p2`)
    if align == "left":
        p0 = (ax + min(mw, 160) * 0.52, cur_y + 6)
        p2 = (target_xy[0] - 14.0, target_xy[1] - 4.0)
        # Graceful downward-then-inward sweep (`╭──→` / `╰──→`)
        p1 = (p0[0] + (p2[0] - p0[0]) * 0.28, p0[1] + (p2[1] - p0[1]) * 0.72 + 24.0)
    else:
        p0 = (ax - min(mw, 160) * 0.52, cur_y + 6)
        p2 = (target_xy[0] + 14.0, target_xy[1] - 4.0)
        p1 = (p0[0] + (p2[0] - p0[0]) * 0.28, p0[1] + (p2[1] - p0[1]) * 0.72 + 24.0)

    draw_tapered_bezier_arrow(
        rd,
        rsd,
        p0,
        p1,
        p2,
        progress=arrow_prog,
        col_shaft=COL_WHITE if style_variant != 2 else col_soft,
        col_glow=col_main,
        comet_pos=comet_pos,
        dashed=(style_variant == 3),
        ss=ss,
    )

    if style_variant in (1, 2, 3):
        draw_target_impact_ripple(rd, rsd, target_xy, col_main, col_soft, t_local, rep_w, ss=ss)
    else:
        # Variant 4: subtle small white dot only
        if arrow_prog > 0.85:
            tx, ty = target_xy[0] * ss, target_xy[1] * ss
            rd.ellipse([tx - 7, ty - 7, tx + 7, ty + 7], fill=(255, 255, 255, 240))


def render_day3_v40_callouts(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
    kp: dict,
    is_wrong: bool,
    t_local: float = 2.4,
    rep_amount: float = 1.0,
    style_variant: int = 1,
):
    """
    Renders Day 3 (Barbell Back Squat) anatomical target / overload callouts across all 4 steps
    with ZERO degree (`°`) numbers and 100% v39 iPhone Safe-Zone compliance (`x in 96..965, y <= 935` on right).
    """
    SS = 2
    rig = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig)
    rig_sh = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rsd = ImageDraw.Draw(rig_sh)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    if step_idx == 1:
        # Step 1 (Direct Front Coronal View — Knee Valgus Collapse):
        # Left points to Right Medial Knee (MCL / Meniscus), Right points to Left Medial Knee / ACL
        pt_left = map_pt(kp["kn_r"][0] + 8.0, kp["kn_r"][1] - 4.0)
        pt_right = map_pt(kp["kn_l"][0] - 8.0, kp["kn_l"][1] - 4.0)
        draw_kinetic_callout_unit(
            od, sd, rd, rsd,
            anchor_xy=(104, 830),
            target_xy=pt_left,
            main_title="MCL & MENISCUS",
            sub_label="INWARD KNEE BUCKLE",
            pill_label="JOINT OVERLOAD",
            is_wrong=True,
            align="left",
            t_local=t_local,
            rep_w=rep_amount,
            style_variant=style_variant,
            ss=SS,
        )
        draw_kinetic_callout_unit(
            od, sd, rd, rsd,
            anchor_xy=(960, 760),
            target_xy=pt_right,
            main_title="ACL LIGAMENT",
            sub_label="VALGUS TORQUE STRESS",
            pill_label="HIGH RISK",
            is_wrong=True,
            align="right",
            t_local=t_local,
            rep_w=rep_amount,
            style_variant=style_variant,
            ss=SS,
        )

    elif step_idx == 2:
        # Step 2 (3/4 Side-Profile View — Heel Lift & Forward Lean):
        # Left points to Lumbar Spine (L4-L5), Right points to Anterior Knee / Patellar Tendon
        pt_left = map_pt(kp["hip"][0] + 15.0, kp["hip"][1] - 78.0)
        pt_right = map_pt(kp["kn_r"][0] + 14.0, kp["kn_r"][1] - 6.0)
        draw_kinetic_callout_unit(
            od, sd, rd, rsd,
            anchor_xy=(104, 735),
            target_xy=pt_left,
            main_title="LOWER BACK",
            sub_label="L4-L5 SPINAL SHEAR",
            pill_label="OVERLOAD ZONE",
            is_wrong=True,
            align="left",
            t_local=t_local,
            rep_w=rep_amount,
            style_variant=style_variant,
            ss=SS,
        )
        draw_kinetic_callout_unit(
            od, sd, rd, rsd,
            anchor_xy=(960, 775),
            target_xy=pt_right,
            main_title="PATELLAR TENDON",
            sub_label="ANTERIOR KNEE LOAD",
            pill_label="SHEAR STRESS",
            is_wrong=True,
            align="right",
            t_local=t_local,
            rep_w=rep_amount,
            style_variant=style_variant,
            ss=SS,
        )

    elif step_idx == 3:
        # Step 3 (Lower-Body Medium Close-Up View — Tripod Foot & Knees Track Toes):
        # Left points to Outer/Teardrop Quad (VMO & Vastus Lateralis), Right points to Inner Thigh / Hip Stabilizers
        pt_left = map_pt(kp["kn_r"][0] - 15.0, kp["kn_r"][1] - 95.0)
        pt_right = map_pt(kp["kn_l"][0] - 35.0, kp["kn_l"][1] - 85.0)
        draw_kinetic_callout_unit(
            od, sd, rd, rsd,
            anchor_xy=(104, 595),
            target_xy=pt_left,
            main_title="VMO & QUAD SWEEP",
            sub_label="VASTUS MEDIALIS / LATERALIS",
            pill_label="TARGET HIT",
            is_wrong=False,
            align="left",
            t_local=t_local,
            rep_w=rep_amount,
            style_variant=style_variant,
            ss=SS,
        )
        draw_kinetic_callout_unit(
            od, sd, rd, rsd,
            anchor_xy=(960, 595),
            target_xy=pt_right,
            main_title="ADDUCTOR & GLUTE",
            sub_label="EXTERNAL HIP TORQUE",
            pill_label="STABILIZER LOCK",
            is_wrong=False,
            align="right",
            t_local=t_local,
            rep_w=rep_amount,
            style_variant=style_variant,
            ss=SS,
        )

    else:
        # Step 4 (Full-Body Wide 3/4 View — Deep Squat Drive):
        # Left points to Gluteus Maximus belly, Right points to Quadriceps main belly
        pt_left = map_pt(kp["hip"][0] - 18.0, kp["hip"][1] + 15.0)
        pt_right = map_pt((kp["hip"][0] + kp["kn_r"][0]) * 0.5 + 22.0, (kp["hip"][1] + kp["kn_r"][1]) * 0.5 - 12.0)
        draw_kinetic_callout_unit(
            od, sd, rd, rsd,
            anchor_xy=(104, 745),
            target_xy=pt_left,
            main_title="GLUTEUS MAXIMUS",
            sub_label="POSTERIOR HIP DRIVE",
            pill_label="PRIMARY TARGET",
            is_wrong=False,
            align="left",
            t_local=t_local,
            rep_w=rep_amount,
            style_variant=style_variant,
            ss=SS,
        )
        draw_kinetic_callout_unit(
            od, sd, rd, rsd,
            anchor_xy=(960, 765),
            target_xy=pt_right,
            main_title="QUADRICEPS",
            sub_label="FULL ANTERIOR SWEEP",
            pill_label="PRIMARY TARGET",
            is_wrong=False,
            align="right",
            t_local=t_local,
            rep_w=rep_amount,
            style_variant=style_variant,
            ss=SS,
        )

    canvas.alpha_composite(rig_sh.resize((W, H), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(rig.resize((W, H), Image.Resampling.LANCZOS))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(ov)


def build_v40_concepts_and_master():
    print("Building v40 Kinetic Curved-Arrow Callout Master & Motion Graphics Board...")
    raw_s1 = d3._load_clip_frames(d3.DAY3_STEP1_VALGUS_I2V)
    raw_s2 = d3._load_clip_frames(d3.DAY3_STEP2_WRONG_I2V)
    raw_s3 = d3._load_clip_frames(d3.DAY3_STEP3_KNEELOCK_I2V)
    raw_s4 = d3._load_clip_frames(d3.DAY3_STEP4_CORRECT_I2V)

    stroke_s1 = raw_s1[54:119][::-1]
    stroke_s2 = raw_s2[64:121]
    stroke_s3 = raw_s3[56:126][::-1]
    stroke_s4 = raw_s4[90:145]

    pool_s1, kp_s1_raw = [], []
    for fr in stroke_s1:
        rgb, kp = d3.render_day3_i2v_frame_v39(fr, 1, True)
        pool_s1.append(rgb)
        kp_s1_raw.append(kp)
    smooth_kp_s1 = d3._smooth_kpts_seq(kp_s1_raw)

    pool_s2, kp_s2_raw = [], []
    for fr in stroke_s2:
        rgb, kp = d3.render_day3_i2v_frame_v39(fr, 2, True)
        pool_s2.append(rgb)
        kp_s2_raw.append(kp)
    smooth_kp_s2 = d3._smooth_kpts_seq(kp_s2_raw)

    pool_s3, kp_s3_raw = [], []
    for fr in stroke_s3:
        rgb, kp = d3.render_day3_i2v_frame_v39(fr, 3, False)
        pool_s3.append(rgb)
        kp_s3_raw.append(kp)
    smooth_kp_s3 = d3._smooth_kpts_seq(kp_s3_raw)

    pool_s4, kp_s4_raw = [], []
    for fr in stroke_s4:
        rgb, kp = d3.render_day3_i2v_frame_v39(fr, 4, False)
        pool_s4.append(rgb)
        kp_s4_raw.append(kp)
    smooth_kp_s4 = d3._smooth_kpts_seq(kp_s4_raw)

    size = (136, 108)
    im0_act = Image.fromarray(pool_s1[-1]).convert("RGBA").crop((220, 740, 860, 1480)).resize(size, Image.Resampling.LANCZOS)
    im1_act = Image.fromarray(pool_s2[-1]).convert("RGBA").crop((310, 680, 910, 1460)).resize(size, Image.Resampling.LANCZOS)
    im2_act = Image.fromarray(pool_s4[-1]).convert("RGBA").crop((310, 680, 910, 1460)).resize(size, Image.Resampling.LANCZOS)

    def _make_inact(act_im: Image.Image) -> Image.Image:
        out = act_im.copy()
        out.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 138)))
        return out

    torso_icons = [
        (im0_act, _make_inact(im0_act)),
        (im1_act, _make_inact(im1_act)),
        (im2_act, _make_inact(im2_act)),
    ]
    dock_cache = d3.build_cached_day3_v39_top_docks(torso_icons)

    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]
    scrim_2d = v38.make_bottom_scrim_2d()

    # 1. Build the 4-Variant Upgraded A-Series & Animation Showcase Sheet (`v40_motion_callout_concepts_board.jpg`)
    cw, ch = 430, 764
    pad = 24
    board = Image.new("RGB", (cw * 4 + pad * 5, ch * 2 + 215), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_h1 = v30.f_pretendard(23, "extrabold")
    f_h2 = v30.f_pretendard(17, "extrabold")
    f_sub = v30.f_pretendard(14, "bold")

    bd.text(
        (28, 18),
        "OPTION A 업그레이드 · 모션 그래픽 애니메이션 타임라인 (상단) & 4가지 세부 스타일 비교 (하단)",
        font=f_h1,
        fill=COL_GREEN,
    )

    # Top row: Show the 4 Animation Stages of Variant A-1 (Best Recommended) on Step 4 (`QUADRICEPS` & `GLUTEUS MAXIMUS`)
    anim_stages = [
        (0.24, 0.35, "STAGE 1 (0.2s) · 텍스트 슬라이드 + 곡선 드로우온 시작", "글자 아래에서 테이퍼드 화이트 곡선이 근육을 향해 그려짐"),
        (0.52, 0.65, "STAGE 2 (0.5s) · 곡선 화살표 도착 + 광원 혜성(Comet) 발사", "곡선을 따라 빛 입자가 빠르게 흐르며 근육 타겟에 꽂힘"),
        (2.15, 0.96, "STAGE 3 (2.1s) · 최대 수축 도달 + 소나 충격파 파동(◎)", "화살표 끝 타겟 부위에서 동심원 리플 파동이 퍼지며 3D 근섬유 폭발"),
        (2.40, 1.00, "STAGE 4 (오류 동작 예시) · 과부하 관절 부위 직관적 타겟팅", "잘못된 자세에서는 허리(LOWER BACK)·무릎(PATELLAR) 부위를 레드로 지목"),
    ]

    for idx, (t_loc, rw, title_a, desc_a) in enumerate(anim_stages):
        x = pad + idx * (cw + pad)
        is_w = (idx == 3)
        col = COL_RED if is_w else COL_GREEN
        bd.text((x + 4, 58), title_a, font=f_h2, fill=col)
        bd.text((x + 4, 82), desc_a, font=f_sub, fill=(185, 200, 220))

        pool_use = pool_s2 if is_w else pool_s4
        kp_use = smooth_kp_s2 if is_w else smooth_kp_s4
        pos_f = rw * (len(pool_use) - 1)
        base_rgb = v37.sample_pool_interpolated(pool_use, pos_f)
        kp = v37.sample_kpts_interpolated(kp_use, pos_f)
        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=1.03 + 0.02 * rw, focus_xy=(580.0, 965.0), dest_xy=(590.0, 965.0))
        clean_rgb = v38.apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)
        im = Image.fromarray(clean_rgb).convert("RGBA")
        render_day3_v40_callouts(im, 2 if is_w else 4, map_pt, kp, is_w, t_local=t_loc, rep_amount=rw, style_variant=1)
        im.alpha_composite(dock_cache[(1 if is_w else 2, is_w)])
        d3.draw_day3_v39_safezone_bottom_hud(
            im,
            {
                "is_wrong": is_w,
                "pct": 79 if is_w else int(round(94 + 4 * rw)),
                "tag_pill": "SPINE & KNEE LOAD" if is_w else "PRIMARY TARGET · 4×10",
                "tag_ctx": "LUMBAR & PATELLA" if is_w else "QUADRICEPS & GLUTES",
                "cue_1line": "HEELS LIFTING UP" if is_w else "DEEP SQUAT DRIVE",
            },
        )
        rc = im.convert("RGB").resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        board.paste(rc, (x, 108))

    # Bottom row: Compare the 4 A-Series Style Variants at Peak Contraction (`t_local = 2.15s`)
    var_headers = [
        ("★ 추천 A-1 · 테이퍼드 곡선 + 혜성 + 타겟 파동(◎)", "상단 미니 태그 + 볼드 화이트 부위명 + 테이퍼드 화살표 & 파동"),
        ("A-2 · 언더라인 컬러 바 일체형 곡선 화살표", "글자 밑줄 컬러 바가 그대로 휘어져 근육으로 꽂히는 스타일"),
        ("A-3 · 에너지 플로우 점선 스트림 곡선 화살표", "곡선 화살표 자체가 흐르는 에너지 대시로 근육에 주입되는 스타일"),
        ("A-4 · 울트라 미니멀 순수 화이트 곡선 (태그 없음)", "태그·링 없이 오직 화이트 타이포 + 테이퍼드 화이트 곡선만 남긴 스타일"),
    ]
    for idx in range(4):
        v_id = idx + 1
        x = pad + idx * (cw + pad)
        y_top = 108 + ch + 26
        bd.text((x + 4, y_top), var_headers[idx][0], font=f_h2, fill=(255, 255, 255) if v_id > 1 else COL_GREEN)
        bd.text((x + 4, y_top + 24), var_headers[idx][1], font=f_sub, fill=(185, 200, 220))

        base_rgb = pool_s4[-1]
        kp = smooth_kp_s4[-1]
        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=1.05, focus_xy=(580.0, 965.0), dest_xy=(590.0, 965.0))
        clean_rgb = v38.apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)
        im = Image.fromarray(clean_rgb).convert("RGBA")
        render_day3_v40_callouts(im, 4, map_pt, kp, False, t_local=2.15, rep_amount=1.0, style_variant=v_id)
        im.alpha_composite(dock_cache[(2, False)])
        d3.draw_day3_v39_safezone_bottom_hud(
            im,
            {
                "is_wrong": False,
                "pct": 98,
                "tag_pill": "PRIMARY TARGET · 4×10",
                "tag_ctx": "QUADRICEPS & GLUTES",
                "cue_1line": "DEEP SQUAT DRIVE",
            },
        )
        rc = im.convert("RGB").resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=COL_GREEN if v_id == 1 else (120, 135, 155), width=3 if v_id == 1 else 2)
        board.paste(rc, (x, y_top + 52))

    concepts_path = os.path.join(OUT_DIR, "v40_motion_callout_concepts_board.jpg")
    board.save(concepts_path, quality=95)
    print("Saved Motion Callout Concepts Board:", concepts_path)

    # 2. Render Full 20.0s Day 3 (`BARBELL BACK SQUAT`) Master Reel with Variant A-1 (`day3_v40_kinetic_arrow_master.mp4`)
    wav_day3 = os.path.join(TMP_AUDIO, "day3_v40_audio.wav")
    vo_day3 = [
        (0.12, "Stop letting your knees cave inward and overloading your knee ligaments."),
        (5.12, "Lifting your heels pitches you forward and strains your lower back."),
        (10.12, "Root your feet and track knees over toes to fire your quads and glutes."),
        (15.12, "Drop to full depth and drive up for maximum quad and glute growth."),
    ]
    asyncio.run(d3.synthesize_day3_voiceover_wav(wav_day3, vo_day3, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day3_v40_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day3_v40_kinetic_arrow_master.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    hud_cache = {}
    step_static_frames = {}

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            t_local = t
            u = t_local / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s1) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s1, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_kp_s1, pos_f)
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom = 1.02 + 0.02 * rep_w
            cam_focus, cam_dest = (540.0, 980.0), (550.0, 980.0)
            pct = int(round(78 + 6 * rep_w))
            tag_pill = "JOINT OVERLOAD"
            tag_ctx = "MCL · MENISCUS · ACL"
            cue_1line = "KNEE VALGUS CAVE"
        elif t < 10.0:
            t_local = t - 5.0
            u = t_local / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s2) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s2, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_kp_s2, pos_f)
            step_idx = 2
            is_wrong = True
            active_slot = 1
            cam_zoom = 1.03 + 0.02 * rep_w
            cam_focus, cam_dest = (575.0, 970.0), (585.0, 970.0)
            pct = int(round(72 + 7 * rep_w))
            tag_pill = "SPINE & KNEE LOAD"
            tag_ctx = "LOWER BACK & PATELLA"
            cue_1line = "HEELS LIFTING UP"
        elif t < 15.0:
            t_local = t - 10.0
            u = t_local / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s3) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s3, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_kp_s3, pos_f)
            step_idx = 3
            is_wrong = False
            active_slot = 2
            cam_zoom = 1.02 + 0.02 * rep_w
            cam_focus, cam_dest = (560.0, 940.0), (570.0, 940.0)
            pct = int(round(92 + 4 * rep_w))
            tag_pill = "TARGET MUSCLES"
            tag_ctx = "VMO · OUTER QUAD · GLUTE"
            cue_1line = "KNEES TRACK TOES"
        else:
            t_local = t - 15.0
            u = t_local / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s4) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s4, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_kp_s4, pos_f)
            step_idx = 4
            is_wrong = False
            active_slot = 2
            cam_zoom = 1.03 + 0.02 * rep_w
            cam_focus, cam_dest = (580.0, 965.0), (590.0, 965.0)
            pct = int(round(94 + 4 * rep_w))
            tag_pill = "PRIMARY TARGET · 4×10"
            tag_ctx = "QUADRICEPS & GLUTE MAX"
            cue_1line = "DEEP SQUAT DRIVE"

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        clean_rgb = v38.apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)
        im = Image.fromarray(clean_rgb).convert("RGBA")

        render_day3_v40_callouts(im, step_idx, map_pt, kp, is_wrong, t_local=t_local, rep_amount=rep_w, style_variant=1)
        im.alpha_composite(dock_cache[(active_slot, is_wrong)])

        hud_key = (is_wrong, pct, tag_pill, tag_ctx, cue_1line)
        if hud_key not in hud_cache:
            h_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d3.draw_day3_v39_safezone_bottom_hud(
                h_layer,
                {
                    "is_wrong": is_wrong,
                    "pct": pct,
                    "tag_pill": tag_pill,
                    "tag_ctx": tag_ctx,
                    "cue_1line": cue_1line,
                },
            )
            hud_cache[hud_key] = h_layer
        im.alpha_composite(hud_cache[hud_key])

        rgb_frame = im.convert("RGB")
        if f_idx in (int(2.15 * FPS), int(7.15 * FPS), int(12.15 * FPS), int(17.15 * FPS)):
            step_static_frames[step_idx] = rgb_frame.copy()

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

    # Also copy to `day3_v39_master_reel.mp4` so the canonical Day 3 master is always up to date
    shutil.copyfile(final_mp4, os.path.join(OUT_DIR, "day3_v39_master_reel.mp4"))

    # Save 4-Step Master Sheet (`day3_v40_clean_master_sheet.jpg`)
    pw, ph = 420, 746
    pad = 24
    s_board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    s_bd = ImageDraw.Draw(s_board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    s_bd.text(
        (28, 20),
        "DAY 3 · BARBELL BACK SQUAT (v40 KINETIC CURVED-ARROW MASTER)  ·  각도(°) 0% + 타겟 근육/관절 모션 포인터",
        font=f_row,
        fill=COL_GREEN,
    )
    col_titles = [
        "STEP 1 (전면 뷰) · MCL & ACL 무릎 내반 과부하",
        "STEP 2 (측면 뷰) · LOWER BACK & PATELLA 과부하",
        "STEP 3 (하체 근접) · VMO & ADDUCTOR 타겟 정렬",
        "STEP 4 (전신 풀샷) · GLUTE MAX & QUADRICEPS 타격",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = COL_RED if i < 2 else COL_GREEN
        s_bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = step_static_frames[i + 1].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=col, width=2)
        s_board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day3_v40_clean_master_sheet.jpg")
    s_board.save(sheet_path, quality=95)
    print("Saved v40 Day 3 Master Video:", final_mp4)
    print("Saved v40 Day 3 Master Sheet:", sheet_path)


if __name__ == "__main__":
    build_v40_concepts_and_master()
