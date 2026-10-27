#!/usr/bin/env python3
"""
GymForm 3D — v30 Pure Graphic & Number-Driven Biomechanical Master
Fixes:
1. "팔 쪽에 있는 철심같은게 거슬리네?"
   -> Completely removes the vertebral cross-tick line (which passed right over the arm in side profile and looked like a surgical iron pin/철심),
      removes all on-arm lines/dots/ticks, and removes long pointer lines poking into the arm.
   -> The 3D muscle striations (and Step 1 X-Ray bone) on the arm are now 100% pure and untouched!
2. "글은 최소화하고 그래픽적으로 풀어가야해. 숫자나"
   -> Places clean, smooth exterior posture & motion arcs in the dark space OUTSIDE the body silhouette
      (behind the back and in front of the forearm/rope), paired directly with bold sculpted numbers (35°, 90°, 0°, 15°, 180°, 25°).
   -> Lower-left open studio space features a frameless Giant % Number + 20-Segment Visual EMG Bar Gauge + Set×Rep Number (4×12).
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import gymform_engine as eng
import build_v25_fast_preview as v25
import build_v28_luxury_ab as v28

W, H = eng.W, eng.H
OUT_DIR = eng.OUT_DIR

COL_RED = (255, 69, 58)
COL_RED_SOFT = (255, 138, 128)
COL_GREEN = (46, 228, 118)
COL_GREEN_SOFT = (158, 248, 195)
COL_SILVER = (224, 232, 242)
COL_MUTED = (142, 154, 174)

f_pretendard = v28.f_pretendard
f_barlow = v28.f_barlow

F_GIANT_NUM = f_barlow(82, extrabold=True)
F_ARC_NUM = f_barlow(52, extrabold=True)
F_ARC_TAG = f_pretendard(14, "bold")
F_CUE_SHORT = f_pretendard(34, "extrabold")
F_TAG_TINY = f_pretendard(14, "bold")
F_BAR_NUM = f_barlow(28, extrabold=True)
F_RX_NUM = f_barlow(34, extrabold=True)


def draw_arrow_head(rd: ImageDraw.ImageDraw, tip: tuple[float, float], angle_deg: float, size: float, fill: tuple):
    ang = math.radians(angle_deg)
    a1 = ang + math.radians(152)
    a2 = ang - math.radians(152)
    p1 = (tip[0] + size * math.cos(a1), tip[1] + size * math.sin(a1))
    p2 = (tip[0] + size * math.cos(a2), tip[1] + size * math.sin(a2))
    rd.polygon([tip, p1, p2], fill=fill)


def draw_floating_number_badge(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    num_str: str,
    tag_str: str,
    accent: tuple,
    accent_soft: tuple,
    align: str = "right",
):
    """Draws a crisp frameless number + 1-word micro-tag right next to an exterior arc."""
    x, y = xy
    nb = od.textbbox((0, 0), num_str, font=F_ARC_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), tag_str, font=F_ARC_TAG)
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

    sd.text((nx, y + 3), num_str, font=F_ARC_NUM, fill=(0, 0, 0, 245))
    od.text((nx, y), num_str, font=F_ARC_NUM, fill=(255, 255, 255, 255))

    sd.text((tx, y + 54), tag_str, font=F_ARC_TAG, fill=(0, 0, 0, 245))
    od.text((tx, y + 52), tag_str, font=F_ARC_TAG, fill=(*accent_soft, 255))


def render_zero_occlusion_exterior_graphics(
    canvas: Image.Image,
    step_idx: int,
    mp: callable,
    is_wrong: bool,
):
    """
    Renders 100% exterior posture & motion graphics in the dark studio space OUTSIDE the mannequin's arm and body:
    - ZERO lines, ticks, or rods crossing the upper arm or forearm!
    - Smooth exterior back-posture curve behind the torso (`x > 855`) + vertical reference line.
    - Smooth exterior forearm/rope sweep arc in front of the arm (`x < 610`).
    - Bold sculpted numbers (`35°`, `90°`, `0°`, `15°`, `180°`, `25°`) placed right next to each exterior arc.
    """
    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    SS = 2
    rig = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    def S(pt):
        return (pt[0] * SS, pt[1] * SS)

    if step_idx == 1:
        # 1. Exterior Dorsal Back Curve (placed 28px OUTSIDE the hunched back silhouette at x=915..955)
        b_bot = S(mp(935, 1045))
        b_top = S(mp(845, 665))
        # Vertical 0-deg reference datum behind the back
        for yd in range(int(b_top[1]), int(b_bot[1]), 14 * SS):
            rd.line([(b_bot[0], yd), (b_bot[0], min(int(b_bot[1]), yd + 6 * SS))], fill=(215, 225, 238, 120), width=2 * SS)
        # Smooth 35-deg hunched posture arc behind the back (zero cross-ticks!)
        arc_r = 260 * SS
        rd.arc([b_bot[0] - arc_r, b_bot[1] - arc_r, b_bot[0] + arc_r, b_bot[1] + arc_r], start=235, end=270, fill=(*COL_RED, 245), width=4 * SS)
        ang_t = math.radians(235)
        tip_b = (b_bot[0] + arc_r * math.cos(ang_t), b_bot[1] + arc_r * math.sin(ang_t))
        draw_arrow_head(rd, tip_b, 235 - 90, 12 * SS, (*COL_RED_SOFT, 255))

        n1_xy = mp(960, 690)
        draw_floating_number_badge(od, sd, (min(965, int(n1_xy[0])), int(n1_xy[1])), "35°", "BACK HUNCH", COL_RED, COL_RED_SOFT, align="left")

        # 2. Exterior Front Cable/Wrist Flare Arc (in open space in front of hands at x=440..540)
        w_c = S(mp(560, 995))
        r_w = 78 * SS
        rd.arc([w_c[0] - r_w, w_c[1] - r_w, w_c[0] + r_w, w_c[1] + r_w], start=145, end=225, fill=(*COL_RED, 245), width=4 * SS)
        ang_w = math.radians(145)
        tip_w = (w_c[0] + r_w * math.cos(ang_w), w_c[1] + r_w * math.sin(ang_w))
        draw_arrow_head(rd, tip_w, 145 - 90, 11 * SS, (*COL_RED_SOFT, 255))

        n2_xy = mp(445, 915)
        draw_floating_number_badge(od, sd, (int(n2_xy[0]), int(n2_xy[1])), "28°", "WRIST BEND", COL_RED, COL_RED_SOFT, align="right")

    elif step_idx == 2:
        # Step 2: 90-deg Half Rep Stop (Arc placed in the open left space IN FRONT of the forearm/rope, x=410..560!)
        # Horizontal 90-deg stop line in front of the hands (never touching the arm!)
        h_left = S(mp(425, 882))
        h_right = S(mp(535, 882))
        rd.line([h_left, h_right], fill=(*COL_RED, 245), width=4 * SS)
        rd.line([(h_left[0], h_left[1] - 12 * SS), (h_left[0], h_left[1] + 12 * SS)], fill=(*COL_RED_SOFT, 255), width=3 * SS)

        # Curved downward sweep arc in front of the rope showing the missing 90->180 bottom half
        arc_c = S(mp(645, 882))
        r_m = 155 * SS
        rd.arc([arc_c[0] - r_m, arc_c[1] - r_m, arc_c[0] + r_m, arc_c[1] + r_m], start=132, end=180, fill=(*COL_RED, 245), width=4 * SS)
        # Ghost green arc below showing where the rep SHOULD go
        for a_start in range(96, 130, 8):
            rd.arc([arc_c[0] - r_m, arc_c[1] - r_m, arc_c[0] + r_m, arc_c[1] + r_m], start=a_start, end=a_start + 4, fill=(*COL_GREEN_SOFT, 190), width=3 * SS)

        n1_xy = mp(445, 815)
        draw_floating_number_badge(od, sd, (int(n1_xy[0]), int(n1_xy[1])), "90°", "STOP SHORT", COL_RED, COL_RED_SOFT, align="right")

        n2_xy = mp(495, 615)
        draw_floating_number_badge(od, sd, (int(n2_xy[0]), int(n2_xy[1])), "72%", "DELT STEAL", COL_RED, COL_RED_SOFT, align="right")

    elif step_idx == 3:
        # Step 3: 15-deg Torso Lean (exterior back arc at x=865..915) + Forearm Sweep Arc in front of rope (`x=450..560`)
        b_bot = S(mp(865, 1045))
        b_top = S(mp(845, 685))
        for yd in range(int(b_top[1]), int(b_bot[1]), 14 * SS):
            rd.line([(b_bot[0], yd), (b_bot[0], min(int(b_bot[1]), yd + 6 * SS))], fill=(215, 225, 238, 115), width=2 * SS)
        arc_r = 250 * SS
        rd.arc([b_bot[0] - arc_r, b_bot[1] - arc_r, b_bot[0] + arc_r, b_bot[1] + arc_r], start=255, end=270, fill=(*COL_GREEN, 245), width=4 * SS)

        n_back = mp(905, 720)
        draw_floating_number_badge(od, sd, (min(955, int(n_back[0])), int(n_back[1])), "15°", "HIP HINGE", COL_GREEN, COL_GREEN_SOFT, align="left")

        # Exterior forearm extension arc in open space in front of the rope (`x=455..565`)
        arc_c = S(mp(655, 885))
        r_f = 155 * SS
        rd.arc([arc_c[0] - r_f, arc_c[1] - r_f, arc_c[0] + r_f, arc_c[1] + r_f], start=128, end=174, fill=(*COL_GREEN, 245), width=4 * SS)
        ang_tip = math.radians(128)
        tip_f = (arc_c[0] + r_f * math.cos(ang_tip), arc_c[1] + r_f * math.sin(ang_tip))
        draw_arrow_head(rd, tip_f, 128 - 90, 12 * SS, (*COL_GREEN_SOFT, 255))

        n_front = mp(465, 865)
        draw_floating_number_badge(od, sd, (int(n_front[0]), int(n_front[1])), "0°", "ELBOW FIXED", COL_GREEN, COL_GREEN_SOFT, align="right")

    else:
        # Step 4: 180-deg Full Lockout Arc + 25-deg Outward Rope Spread Arrows (100% outside the arm!)
        arc_c = S(mp(665, 885))
        r_f = 175 * SS
        rd.arc([arc_c[0] - r_f, arc_c[1] - r_f, arc_c[0] + r_f, arc_c[1] + r_f], start=112, end=168, fill=(*COL_GREEN, 245), width=4 * SS)
        ang_tip = math.radians(112)
        tip_f = (arc_c[0] + r_f * math.cos(ang_tip), arc_c[1] + r_f * math.sin(ang_tip))
        draw_arrow_head(rd, tip_f, 112 - 90, 13 * SS, (*COL_GREEN_SOFT, 255))

        n_lock = mp(455, 845)
        draw_floating_number_badge(od, sd, (int(n_lock[0]), int(n_lock[1])), "180°", "LOCKOUT", COL_GREEN, COL_GREEN_SOFT, align="right")

        # Outward 25-deg rope spread curved arrows around the bottom of the rope (`y=1110..1170`, below the hand)
        w_c = S(mp(640, 1115))
        r_sp = 54 * SS
        rd.arc([w_c[0] - r_sp, w_c[1] - r_sp, w_c[0] + r_sp, w_c[1] + r_sp], start=130, end=215, fill=(*COL_GREEN_SOFT, 250), width=4 * SS)
        ang_l = math.radians(215)
        tip_l = (w_c[0] + r_sp * math.cos(ang_l), w_c[1] + r_sp * math.sin(ang_l))
        draw_arrow_head(rd, tip_l, 215 + 90, 11 * SS, (*COL_GREEN_SOFT, 255))

        rd.arc([w_c[0] - r_sp, w_c[1] - r_sp, w_c[0] + r_sp, w_c[1] + r_sp], start=325, end=40, fill=(*COL_GREEN_SOFT, 250), width=4 * SS)
        ang_r = math.radians(325)
        tip_r = (w_c[0] + r_sp * math.cos(ang_r), w_c[1] + r_sp * math.sin(ang_r))
        draw_arrow_head(rd, tip_r, 325 - 90, 11 * SS, (*COL_GREEN_SOFT, 255))

        n_sp = mp(525, 1085)
        draw_floating_number_badge(od, sd, (int(n_sp[0]), int(n_sp[1])), "25°", "ROPE SPREAD", COL_GREEN, COL_GREEN_SOFT, align="right")

    rig_smooth = rig.resize((W, H), Image.Resampling.LANCZOS)
    shadow_blur = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(rig_smooth)
    canvas.alpha_composite(shadow_blur)
    canvas.alpha_composite(ov)


def draw_v30_graphic_telemetry_bottom_left(canvas: Image.Image, sp: dict, mode: str = "A"):
    """
    Pure Graphic + Number Telemetry Cluster in the lower-left studio space (x=68..475, y=1265..1545):
    - 0% boxes, 0% overlap with legs, 0% long sentences!
    """
    is_wrong = sp["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT
    pct = sp["pct"]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx, by = 68, 1265

    if mode == "A":
        # 1. Giant Sculpted Percentage + 2-Word Cue
        pct_str = f"{pct}%"
        pb = od.textbbox((0, 0), pct_str, font=F_GIANT_NUM)
        pw = pb[2] - pb[0]
        sd.text((bx, by + 3), pct_str, font=F_GIANT_NUM, fill=(0, 0, 0, 245))
        od.text((bx, by), pct_str, font=F_GIANT_NUM, fill=(*accent, 255))

        tx = bx + pw + 18
        sd.text((tx, by + 14), sp["dial_tag"], font=F_TAG_TINY, fill=(0, 0, 0, 240))
        od.text((tx, by + 12), sp["dial_tag"], font=F_TAG_TINY, fill=(*accent_soft, 255))

        sd.text((tx, by + 36), sp["short_cue"], font=F_CUE_SHORT, fill=(0, 0, 0, 245))
        od.text((tx, by + 33), sp["short_cue"], font=F_CUE_SHORT, fill=(255, 255, 255, 255))

        # 2. High-Tech 20-Segment LED Equalizer / EMG Bar Spectrum
        gy = by + 104
        for m_lbl, m_val, m_col in sp["splits"]:
            sd.text((bx, gy + 5), m_lbl, font=F_TAG_TINY, fill=(0, 0, 0, 240))
            od.text((bx, gy + 3), m_lbl, font=F_TAG_TINY, fill=COL_SILVER)

            bar_x0 = bx + 112
            n_segs = 20
            seg_w, seg_gap = 8, 4
            active_segs = int(round((m_val / 100.0) * n_segs))
            for s_i in range(n_segs):
                sx_i = bar_x0 + s_i * (seg_w + seg_gap)
                col_s = (*m_col, 255) if s_i < active_segs else (42, 48, 60, 180)
                od.rounded_rectangle([sx_i, gy + 3, sx_i + seg_w, gy + 19], radius=2, fill=col_s)

            vx = bar_x0 + n_segs * (seg_w + seg_gap) + 8
            v_str = f"{m_val}%"
            sd.text((vx, gy - 2), v_str, font=F_BAR_NUM, fill=(0, 0, 0, 245))
            od.text((vx, gy - 4), v_str, font=F_BAR_NUM, fill=(255, 255, 255, 255) if m_col == accent else COL_MUTED)
            gy += 38

        if sp.get("rx_num"):
            ry = gy + 8
            od.rounded_rectangle([bx, ry, bx + 186, ry + 42], radius=21, fill=(255, 255, 255, 18), outline=(*accent_soft, 160), width=2)
            od.text((bx + 20, ry + 4), sp["rx_num"], font=F_RX_NUM, fill=(255, 255, 255, 255))
            od.text((bx + 108, ry + 13), sp["rx_unit"], font=F_TAG_TINY, fill=(*accent_soft, 255))

    else:
        # Mode B: 4x-Supersampled Radial Ring Gauge + Segmented Bars
        cx, cy = bx + 74, by + 76
        SS = 4
        d_sz = 160
        dial = Image.new("RGBA", (d_sz * SS, d_sz * SS), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dial)
        dc = (d_sz * SS) // 2
        r_arc = 60 * SS
        dd.ellipse([dc - 68 * SS, dc - 68 * SS, dc + 68 * SS, dc + 68 * SS], fill=(10, 12, 16, 155))
        start_a, span_a = 135, 270
        dd.arc([dc - r_arc, dc - r_arc, dc + r_arc, dc + r_arc], start=start_a, end=start_a + span_a, fill=(45, 52, 66, 200), width=9 * SS)
        active_end = start_a + int(span_a * (pct / 100.0))
        dd.arc([dc - r_arc, dc - r_arc, dc + r_arc, dc + r_arc], start=start_a, end=active_end, fill=(*accent, 255), width=10 * SS)
        ov.alpha_composite(dial.resize((d_sz, d_sz), Image.Resampling.LANCZOS), (cx - d_sz // 2, cy - d_sz // 2))

        num_s = f"{pct}%"
        nb = od.textbbox((0, 0), num_s, font=F_ARC_NUM)
        od.text((cx - (nb[2] - nb[0]) // 2, cy - 26), num_s, font=F_ARC_NUM, fill=(255, 255, 255, 255))
        tb = od.textbbox((0, 0), sp["dial_tag"], font=F_TAG_TINY)
        od.text((cx - (tb[2] - tb[0]) // 2, cy + 24), sp["dial_tag"], font=F_TAG_TINY, fill=(*accent_soft, 245))

        rx = bx + 168
        ry = by + 10
        sd.text((rx, ry + 3), sp["short_cue"], font=F_CUE_SHORT, fill=(0, 0, 0, 245))
        od.text((rx, ry), sp["short_cue"], font=F_CUE_SHORT, fill=(255, 255, 255, 255))

        gy = ry + 52
        for m_lbl, m_val, m_col in sp["splits"]:
            od.text((rx, gy + 2), m_lbl, font=F_TAG_TINY, fill=COL_SILVER)
            bar_x0 = rx + 96
            n_segs = 12
            seg_w, seg_gap = 8, 4
            active_segs = int(round((m_val / 100.0) * n_segs))
            for s_i in range(n_segs):
                sx_i = bar_x0 + s_i * (seg_w + seg_gap)
                col_s = (*m_col, 255) if s_i < active_segs else (42, 48, 60, 180)
                od.rounded_rectangle([sx_i, gy + 2, sx_i + seg_w, gy + 16], radius=2, fill=col_s)
            od.text((bar_x0 + n_segs * (seg_w + seg_gap) + 6, gy - 4), f"{m_val}%", font=F_BAR_NUM, fill=(255, 255, 255, 255))
            gy += 32

    shadow_blur = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(shadow_blur)
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

    x_arr, _, _, _ = eng.render_xray_skeletal_frame(f_w45, 1.0, 0.90)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m1_arr, _ = v25.render_natural_muscle_frame(f_c14, "anterior_deltoid", True, 1.0)
    im1 = Image.fromarray(m1_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m2_arr, _ = v25.render_natural_muscle_frame(f_c42, "triceps_lat_long", False, 1.0)
    m3_arr, _ = v25.render_natural_muscle_frame(f_c64, "triceps_all_three", False, 1.0)
    im2 = Image.fromarray(m3_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    steps = [
        {
            "idx": 1,
            "is_wrong": True,
            "base_rgb": x_arr,
            "active_slot": 0,
            "cam_zoom": 1.04,
            "cam_focus": (690.0, 750.0),
            "cam_dest": (700.0, 750.0),
            "pct": 86,
            "dial_tag": "JOINT RISK",
            "short_cue": "✕ HUNCHED BACK",
            "splits": [
                ("JOINT LOAD", 86, COL_RED),
                ("TRICEPS", 18, COL_MUTED),
            ],
            "rx_num": "35°",
            "rx_unit": "KYPHOSIS",
        },
        {
            "idx": 2,
            "is_wrong": True,
            "base_rgb": m1_arr,
            "active_slot": 1,
            "cam_zoom": 1.22,
            "cam_focus": (675.0, 775.0),
            "cam_dest": (715.0, 775.0),
            "pct": 72,
            "dial_tag": "DELT STEAL",
            "short_cue": "✕ 90° HALF REP",
            "splits": [
                ("FRONT DELT", 72, COL_RED),
                ("TRICEPS", 28, COL_MUTED),
            ],
            "rx_num": "90°",
            "rx_unit": "STOP SHORT",
        },
        {
            "idx": 3,
            "is_wrong": False,
            "base_rgb": m2_arr,
            "active_slot": 2,
            "cam_zoom": 1.14,
            "cam_focus": (685.0, 775.0),
            "cam_dest": (710.0, 775.0),
            "pct": 95,
            "dial_tag": "TRICEPS ISO",
            "short_cue": "✓ PIN ELBOWS",
            "splits": [
                ("LAT HEAD", 95, COL_GREEN),
                ("LONG HEAD", 91, COL_GREEN),
            ],
            "rx_num": "0°",
            "rx_unit": "FIXED AXIS",
        },
        {
            "idx": 4,
            "is_wrong": False,
            "base_rgb": m3_arr,
            "active_slot": 2,
            "cam_zoom": 1.26,
            "cam_focus": (675.0, 825.0),
            "cam_dest": (720.0, 790.0),
            "pct": 99,
            "dial_tag": "ALL 3 HEADS",
            "short_cue": "✓ 180° SPREAD",
            "splits": [
                ("LAT HEAD", 99, COL_GREEN),
                ("LONG HEAD", 95, COL_GREEN),
                ("MED HEAD", 96, COL_GREEN),
            ],
            "rx_num": "4×12",
            "rx_unit": "REPS · 1s",
        },
    ]

    frames_v30a, frames_v30b = [], []

    for sp in steps:
        cam_rgb, map_pt = v28.apply_cinema_camera(
            sp["base_rgb"],
            zoom=sp["cam_zoom"],
            focus_xy=sp["cam_focus"],
            dest_xy=sp["cam_dest"],
        )

        out = cam_rgb.astype(np.float32)
        left_w = 460
        curve_l = np.linspace(0.62, 1.0, left_w, dtype=np.float32) ** 0.75
        out[480:1780, :left_w] *= curve_l[None, :, None]
        clean_rgb = np.clip(out, 0, 255).astype(np.uint8)

        # v30-A: 100% Exterior Motion/Posture Arcs (Zero Lines on Arm!) + Giant Numbers + 20-Segment LED Bars
        im_a = Image.fromarray(clean_rgb).convert("RGBA")
        render_zero_occlusion_exterior_graphics(im_a, sp["idx"], map_pt, sp["is_wrong"])
        v28.draw_luxury_top_dock(im_a, torso_icons, sp["active_slot"], sp["is_wrong"])
        draw_v30_graphic_telemetry_bottom_left(im_a, sp, mode="A")
        pa = os.path.join(OUT_DIR, f"v30_A_step{sp['idx']}.jpg")
        im_a.convert("RGB").save(pa, quality=95)
        frames_v30a.append(im_a.convert("RGB"))

        # v30-B: 100% Exterior Motion/Posture Arcs + Giant Numbers + Radial Ring Gauge & Bars
        im_b = Image.fromarray(clean_rgb).convert("RGBA")
        render_zero_occlusion_exterior_graphics(im_b, sp["idx"], map_pt, sp["is_wrong"])
        v28.draw_luxury_top_dock(im_b, torso_icons, sp["active_slot"], sp["is_wrong"])
        draw_v30_graphic_telemetry_bottom_left(im_b, sp, mode="B")
        pb = os.path.join(OUT_DIR, f"v30_B_step{sp['idx']}.jpg")
        im_b.convert("RGB").save(pb, quality=95)
        frames_v30b.append(im_b.convert("RGB"))

    pw, ph = 390, 693
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph * 2 + 185), (10, 12, 16))
    bd = ImageDraw.Draw(board)

    f_row = f_pretendard(22, "extrabold")
    f_col = f_pretendard(15, "bold")

    bd.text((28, 22), "v30-A (추천)  ·  팔 위 철심·지시선 100% 제거(근질 무결점 보존) + 몸 바깥 궤적 아크 & 대형 숫자 + 20단 LED 게이지", font=f_row, fill=COL_GREEN)
    bd.text((28, ph + 112), "v30-B  ·  팔 위 철심·지시선 100% 제거 + 몸 바깥 궤적 아크 & 대형 숫자 + 원형 링 다이얼 게이지", font=f_row, fill=(110, 218, 255))

    col_titles = [
        "STEP 1 · 35° HUNCH / 86% RISK (✕)",
        "STEP 2 · 90° STOP / 72% DELT (✕)",
        "STEP 3 · 15° HINGE / 0° ELBOW (✓)",
        "STEP 4 · 180° LOCKOUT / 25° SPREAD (✓)",
    ]

    for i in range(4):
        x = pad + i * (pw + pad)
        col = COL_RED if i < 2 else COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = frames_v30a[i].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 84))

        rb = frames_v30b[i].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rb).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(rb, (x, ph + 148))

    board_p = os.path.join(OUT_DIR, "day1_v30_graphic_numbers_board.jpg")
    board.save(board_p, quality=95)
    print("Saved v30 Graphic & Numbers Board:", board_p)


if __name__ == "__main__":
    main()
