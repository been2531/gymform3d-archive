#!/usr/bin/env python3
"""
GymForm 3D — v26 Information Architecture & Value Upgrade Preview Builder
Answers: "내용은 하단에 있는걸로 충분한지? 우리가 제공하고자하는게 그걸로 충분한가?"
Why v25 bottom pill alone was NOT enough:
1. Top Area Waste & Ambiguity:
   - In v25, the top-left 3-thumbnail dock had no labels, so viewers couldn't tell what the 3 stages meant,
     NOR did the screen ever state the Exercise Name & Target Muscle at the top (`CABLE ROPE PUSHDOWN · TRICEPS 3 HEADS`).
2. Missing 3-Head Anatomical Breakdown (Where & How Much per Head):
   - Saying "95% TRICEPS" in a single circle doesn't tell a lifter WHICH head is working (`LATERAL HEAD`, `LONG HEAD`, `MEDIAL HEAD`)
     or how much the `FRONT DELT` is stealing (`72% STEAL`).
3. Missing Actionable Biomechanical Prescription (Angles, Tempo, Breathing, Setup):
   - Lifters save a reel when it gives concrete, actionable numbers they can use in the gym right now:
     * Setup angle: `15° TORSO LEAN` / `90°–180° ELBOW ARC`
     * Tempo & Breathing: `1s PUSH (EXHALE) · 1s HOLD · 3s RETURN`
     * Routine Prescription: `4 SETS × 12 REPS · REST 60s`
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

import gymform_engine as eng
import build_v25_fast_preview as v25

W, H = eng.W, eng.H
OUT_DIR = eng.OUT_DIR
COL_RED = (255, 62, 50)
COL_GREEN = (24, 226, 105)
COL_AMBER = (255, 184, 48)
COL_CYAN = (72, 210, 255)

FONT_TOP_EX = eng.load_font(26, bold=True)
FONT_TOP_SUB = eng.load_font(15, bold=True)
FONT_TAB_LBL = eng.load_font(13, bold=True)

FONT_CALLOUT_TAG = eng.load_font(15, bold=True)
FONT_CALLOUT_SUB = eng.load_font(13, bold=True)

FONT_DIAL_PCT = eng.load_font(36, bold=True)
FONT_DIAL_TAG = eng.load_font(13, bold=True)

FONT_PILL_KICKER = eng.load_font(16, bold=True)
FONT_PILL_MAIN = eng.load_font(33, bold=True)
FONT_PILL_TIP = eng.load_font(17, bold=True)

FONT_BAR_LBL = eng.load_font(14, bold=True)
FONT_BAR_PCT = eng.load_font(15, bold=True)

FONT_RX_VAL = eng.load_font(17, bold=True)
FONT_RX_LBL = eng.load_font(12, bold=True)


def draw_v26_top_header_and_labeled_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    step_title: str,
):
    """
    Upgraded Top Architecture:
    1. Top Exercise Title Bar (`y = 148..214`):
       Clearly states `CABLE ROPE PUSHDOWN` + `TRICEPS BRACHII (LATERAL · LONG · MEDIAL)` so anyone scrolling knows the exercise immediately.
    2. Labeled 3-Stage Diagnostic Dock (`y = 228..388`):
       Each of the 3 anatomical thumbnails has a crisp bottom label (`✕ JOINT LOAD`, `✕ DELT STEAL`, `✓ 3-HEAD LOCK`)
       so the viewer understands the 3-step progression at a glance.
    3. Top-Right Status Emblem (`✕ / ✓` at `x = 912, y = 308`).
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    # 1. Exercise Header Pill (`x = 84..780, y = 148..214`)
    hx1, hy1, hx2, hy2 = 84, 148, 780, 214
    od.rounded_rectangle(
        [hx1, hy1, hx2, hy2],
        radius=18,
        fill=(14, 18, 25, 225),
        outline=(215, 225, 242, 65),
        width=2,
    )
    od.rounded_rectangle([hx1 + 14, hy1 + 14, hx1 + 22, hy2 - 14], radius=4, fill=(*accent, 255))
    od.text((hx1 + 36, hy1 + 10), "CABLE ROPE PUSHDOWN", font=FONT_TOP_EX, fill=(255, 255, 255, 255))
    od.text((hx1 + 36, hy1 + 41), step_title, font=FONT_TOP_SUB, fill=(*accent, 255))

    # 2. Labeled 3-Stage Dock (`x = 84, y = 228`)
    tw, th = 144, 108
    lbl_h = 26
    gap = 12
    pad = 12
    bar_x = 84
    bar_y = 228
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + lbl_h + 2 * pad

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=22,
        fill=(14, 18, 24, 224),
        outline=(210, 222, 240, 62),
        width=2,
    )

    tab_labels = ["1. JOINT LOAD", "2. DELT STEAL", "3. 3-HEAD FIX"]
    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy().resize((tw, th), Image.Resampling.LANCZOS) if i == active_idx else inact_im.copy().resize((tw, th), Image.Resampling.LANCZOS)

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=12, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        i_col = COL_RED if i < 2 else COL_GREEN
        if i == active_idx:
            od.rounded_rectangle(
                [tx - 2, ty - 2, tx + tw + 2, ty + th + lbl_h + 2],
                radius=14,
                outline=(*i_col, 255),
                width=3,
            )
            od.rounded_rectangle(
                [tx, ty + th - 4, tx + tw, ty + th + lbl_h],
                radius=10,
                fill=(*i_col, 235),
            )
            tb = od.textbbox((0, 0), tab_labels[i], font=FONT_TAB_LBL)
            txt_col = (10, 14, 18, 255) if i == 2 else (255, 255, 255, 255)
            od.text((tx + (tw - (tb[2] - tb[0])) // 2, ty + th + 4), tab_labels[i], font=FONT_TAB_LBL, fill=txt_col)
        else:
            tb = od.textbbox((0, 0), tab_labels[i], font=FONT_TAB_LBL)
            od.text((tx + (tw - (tb[2] - tb[0])) // 2, ty + th + 4), tab_labels[i], font=FONT_TAB_LBL, fill=(155, 168, 188, 215))

    # 3. Top-Right Status Emblem (`x = 912, y = 308`)
    bx = 912
    by = bar_y + bar_h // 2
    br = 46
    ss = 4
    sz = (br * 2 + 24) * ss
    c = sz // 2
    r = br * ss
    em = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    ed = ImageDraw.Draw(em)
    ed.ellipse([c - r, c - r, c + r, c + r], fill=(16, 19, 26, 242), outline=(215, 224, 238, 95), width=2 * ss)
    ri = r - 6 * ss
    ed.ellipse([c - ri, c - ri, c + ri, c + ri], fill=(20, 24, 32, 248), outline=(*accent, 245), width=3 * ss)

    if is_wrong:
        d = int(15 * ss)
        w = int(6.5 * ss)
        ed.line([(c - d, c - d), (c + d, c + d)], fill=(255, 255, 255, 255), width=w)
        ed.line([(c - d, c + d), (c + d, c - d)], fill=(255, 255, 255, 255), width=w)
    else:
        pts = [(c - 18 * ss, c + 1 * ss), (c - 4 * ss, c + 15 * ss), (c + 20 * ss, c - 13 * ss)]
        ed.line(pts, fill=(255, 255, 255, 255), width=int(7 * ss), joint="curve")

    em_down = em.resize((br * 2 + 24, br * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(em_down, (bx - br - 12, by - br - 12))
    canvas.alpha_composite(ov)


def draw_margin_anatomical_callouts(
    canvas: Image.Image,
    callouts: list[dict],
    is_wrong: bool,
):
    """
    Places sleek, minimal anatomical/angle callout pills in the EMPTY dark studio margins
    (left of the cable or right of the back, NEVER covering the arm/triceps!) connected via
    1px surgical leader lines to the exact joint/muscle head.
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    for c in callouts:
        tx, ty = int(c["target"][0]), int(c["target"][1])
        bx, by = int(c["box_center"][0]), int(c["box_center"][1])
        col = c.get("color", COL_RED if is_wrong else COL_GREEN)
        title = c["title"]
        sub = c["sub"]

        tb1 = od.textbbox((0, 0), title, font=FONT_CALLOUT_TAG)
        tb2 = od.textbbox((0, 0), sub, font=FONT_CALLOUT_SUB)
        pw = max(tb1[2] - tb1[0], tb2[2] - tb2[0]) + 28
        ph = 50
        x0, y0 = bx - pw // 2, by - ph // 2
        x1, y1 = bx + pw // 2, by + ph // 2

        # Anchor point on pill edge closest to target
        ax = x1 if tx > bx else x0
        ay = by
        mx = int(bx + (tx - bx) * 0.55)

        od.line([(tx, ty), (mx, ty), (ax, ay)], fill=(8, 10, 14, 210), width=4, joint="curve")
        od.line([(tx, ty), (mx, ty), (ax, ay)], fill=(*col, 235), width=2, joint="curve")
        od.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(255, 255, 255, 255), outline=(*col, 255), width=2)

        od.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=(14, 18, 26, 232), outline=(*col, 210), width=2)
        od.text((x0 + 14, y0 + 7), title, font=FONT_CALLOUT_TAG, fill=(255, 255, 255, 255))
        od.text((x0 + 14, y0 + 27), sub, font=FONT_CALLOUT_SUB, fill=(*col, 255))

    canvas.alpha_composite(ov)


def draw_v26_complete_concept_b_deck(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    phase: dict,
):
    """
    Upgraded Concept B Lower-Left Deck (`y = 1185..1515`, safely left of IG right buttons and above bottom caption):
    1. Row 1 (`y = 1190..1346`):
       - 270° Circular Ring Dial (`cx = 168, cy = 1268, r = 74`) with `%` + `DIAL TAG`
       - Sleek Floating Coaching Pill (`x = 258..785, y = 1192..1344`) with:
         * Kicker: Mistake / Form Tip Category
         * Main Cue: Bold Actionable Coaching Rule
         * Action Tip: Exact angle / breathing / posture cue (`✓ DO:` or `✕ WHY:`)
    2. Row 2 (`y = 1362..1448`):
       - 3-Part Anatomical EMG Split Bar Pill (`x = 94..785`):
         Shows exact activation split across **Lateral Head**, **Long/Medial Head**, and **Front Delt / Joint Shear**
         so the viewer immediately sees *where* the tension goes and *how much*!
    3. Row 3 (`y = 1462..1518`):
       - Save-Worthy Workout Prescription Strip (`x = 94..785`):
         `SETS & REPS: 4 × 12` · `TEMPO: 1s-1s-3s` · `ELBOW ARC: 90°→180°`
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    ring_cx, ring_cy = 168, 1268
    ring_r = 74

    # 1. Precision Reticle (`◎`) on Anatomy -> Top of Ring Dial
    px, py, _ = targets[0]
    px_i, py_i = int(round(px)), int(round(py))

    if len(targets) > 1:
        for k in range(1, len(targets)):
            nx, ny, _ = targets[k]
            nx_i, ny_i = int(round(nx)), int(round(ny))
            od.line([(px_i, py_i), (nx_i, ny_i)], fill=(*accent, 165), width=2)
            od.ellipse([nx_i - 8, ny_i - 8, nx_i + 8, ny_i + 8], fill=(12, 15, 20, 195), outline=(*accent, 240), width=2)
            od.ellipse([nx_i - 3, ny_i - 3, nx_i + 3, ny_i + 3], fill=(255, 255, 255, 255))

    r_out = 17
    od.ellipse([px_i - r_out - 2, py_i - r_out - 2, px_i + r_out + 2, py_i + r_out + 2], outline=(10, 12, 16, 185), width=3)
    od.ellipse([px_i - r_out, py_i - r_out, px_i + r_out, py_i + r_out], outline=(*accent, 250), width=2)
    od.ellipse([px_i - 5, py_i - 5, px_i + 5, py_i + 5], fill=(255, 255, 255, 255), outline=(*accent, 255), width=2)

    ax, ay = ring_cx, ring_cy - ring_r - 6
    elbow_x = ax
    elbow_y = py_i + 32
    pts = [(px_i - r_out - 3, py_i), (elbow_x + 26, py_i), (elbow_x, elbow_y), (ax, ay)]
    od.line(pts, fill=(10, 12, 16, 185), width=5, joint="curve")
    od.line(pts, fill=(*accent, 225), width=2, joint="curve")
    od.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=(*accent, 255))

    # 2. Main Floating Coaching Capsule (`x = 232..785, y = 1192..1344`)
    px1, py1 = 232, 1192
    px2, py2 = 785, 1344
    od.rounded_rectangle([px1 + 4, py1 + 6, px2 + 4, py2 + 6], radius=36, fill=(4, 6, 10, 150))
    od.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=36,
        fill=(16, 20, 28, 226),
        outline=(225, 235, 250, 72),
        width=2,
    )

    # 3. 4x-Supersampled 270° Circular Ring Dial
    pct_val = phase["pct"]
    ss = 4
    dsz = (ring_r * 2 + 24) * ss
    dc = dsz // 2
    dr = ring_r * ss
    dial = Image.new("RGBA", (dsz, dsz), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)

    dd.ellipse([dc - dr, dc - dr, dc + dr, dc + dr], fill=(14, 18, 25, 246), outline=(220, 230, 245, 82), width=2 * ss)
    arc_box = [dc - dr + 10 * ss, dc - dr + 10 * ss, dc + dr - 10 * ss, dc + dr - 10 * ss]
    dd.arc(arc_box, start=135, end=405, fill=(36, 42, 54, 230), width=10 * ss)

    end_deg = 135 + int(round(270 * (pct_val / 100.0)))
    dd.arc(arc_box, start=135, end=end_deg, fill=(*accent, 255), width=10 * ss)

    dial_down = dial.resize((ring_r * 2 + 24, ring_r * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(dial_down, (ring_cx - ring_r - 12, ring_cy - ring_r - 12))

    p_str = f"{pct_val}%"
    pb = od.textbbox((0, 0), p_str, font=FONT_DIAL_PCT)
    od.text((ring_cx - (pb[2] - pb[0]) // 2, ring_cy - 26), p_str, font=FONT_DIAL_PCT, fill=(255, 255, 255, 255))

    dial_tag = phase["dial_tag"]
    sb = od.textbbox((0, 0), dial_tag, font=FONT_DIAL_TAG)
    od.text((ring_cx - (sb[2] - sb[0]) // 2, ring_cy + 18), dial_tag, font=FONT_DIAL_TAG, fill=(*accent, 255))

    # Main Capsule Typography
    tx = 262
    od.text((tx, py1 + 18), phase["kicker"], font=FONT_PILL_KICKER, fill=(*accent, 255))
    od.text((tx, py1 + 46), phase["line1"], font=FONT_PILL_MAIN, fill=(255, 255, 255, 255))
    od.text((tx, py1 + 96), phase["line2"], font=FONT_PILL_TIP, fill=(220, 230, 245, 248))

    # 4. Row 2: 3-Muscle / Joint Split Telemetry Pill (`x = 94..785, y = 1358..1436`)
    bx1, by1, bx2, by2 = 94, 1358, 785, 1436
    od.rounded_rectangle(
        [bx1, by1, bx2, by2],
        radius=22,
        fill=(14, 18, 26, 224),
        outline=(215, 225, 242, 60),
        width=2,
    )
    splits = phase["splits"]  # list of 3 tuples: (label, pct, color)
    col_w = (bx2 - bx1 - 36) // 3
    for i, (s_lbl, s_pct, s_col) in enumerate(splits):
        sx = bx1 + 18 + i * col_w
        sy = by1 + 14
        od.text((sx, sy), s_lbl, font=FONT_BAR_LBL, fill=(205, 215, 230, 240))
        p_txt = f"{s_pct}%"
        ptb = od.textbbox((0, 0), p_txt, font=FONT_BAR_PCT)
        od.text((sx + col_w - 16 - (ptb[2] - ptb[0]), sy - 1), p_txt, font=FONT_BAR_PCT, fill=(*s_col, 255))
        # Progress bar track
        bar_y = sy + 26
        bar_w = col_w - 16
        od.rounded_rectangle([sx, bar_y, sx + bar_w, bar_y + 12], radius=6, fill=(32, 38, 50, 230))
        fill_w = max(10, int(round(bar_w * (s_pct / 100.0))))
        od.rounded_rectangle([sx, bar_y, sx + fill_w, bar_y + 12], radius=6, fill=(*s_col, 255))

    # 5. Row 3: Actionable Gym Prescription Strip (`x = 94..785, y = 1448..1508`)
    rx1, ry1, rx2, ry2 = 94, 1448, 785, 1508
    od.rounded_rectangle(
        [rx1, ry1, rx2, ry2],
        radius=18,
        fill=(14, 18, 26, 218),
        outline=(*accent, 110),
        width=2,
    )
    rx_items = phase["rx"]  # 3 tuples: (label, value)
    rc_w = (rx2 - rx1 - 24) // 3
    for i, (r_lbl, r_val) in enumerate(rx_items):
        rx_x = rx1 + 18 + i * rc_w
        if i > 0:
            od.line([(rx_x - 10, ry1 + 12), (rx_x - 10, ry2 - 12)], fill=(210, 220, 235, 50), width=1)
        od.text((rx_x, ry1 + 9), r_lbl, font=FONT_RX_LBL, fill=(160, 175, 195, 230))
        od.text((rx_x, ry1 + 26), r_val, font=FONT_RX_VAL, fill=(255, 255, 255, 255))

    canvas.alpha_composite(ov)


def main():
    f_w45 = eng.prepare_xray_bone_frame(v25.read_specific_frame(eng.WRONG_I2V, 45))
    f_c14 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 14))
    f_c42 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 42))
    f_c64 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 64))

    size = (144, 114)
    crop_box = (555, 550, 905, 950)
    inact_base = Image.fromarray(f_c42["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 120)))

    x_arr, _, _, _ = eng.render_xray_skeletal_frame(f_w45, 1.0, 0.90)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m1_arr, _ = v25.render_natural_muscle_frame(f_c14, "anterior_deltoid", True, 1.0)
    im1 = Image.fromarray(m1_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m2_arr, _ = v25.render_natural_muscle_frame(f_c64, "triceps_all_three", False, 1.0)
    im2 = Image.fromarray(m2_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    phases = [
        {
            "idx": 0,
            "is_wrong": False,
            "frame": f_c64,
            "mode": "triceps_all_three",
            "active_slot": 2,
            "step_title": "COMPLETE BIOMECHANICS & 3-HEAD ISOLATION GUIDE",
            "pct": 98,
            "dial_tag": "TRICEPS",
            "kicker": "TRICEPS 3-HEAD ANATOMY GUIDE",
            "line1": "2 MISTAKES & 2 PRO TIPS",
            "line2": "✓ DO: 15° LEAN · PIN ELBOWS · SPREAD ROPE AT BOTTOM",
            "splits": [
                ("LATERAL HEAD", 98, COL_GREEN),
                ("LONG & MEDIAL", 94, COL_GREEN),
                ("JOINT STRESS", 8, COL_CYAN),
            ],
            "rx": [
                ("ROUTINE VOLUME", "4 SETS × 12 REPS"),
                ("REP TEMPO", "1s PUSH · 1s HOLD · 3s UP"),
                ("SETUP ANGLE", "15° TORSO LEAN"),
            ],
            "callouts": [
                {"target": (718, 705), "box_center": (235, 675), "title": "LATERAL & LONG HEAD", "sub": "PRIMARY TARGET (98% EMG)", "color": COL_GREEN},
                {"target": (588, 968), "box_center": (235, 968), "title": "EXTERNAL WRIST FLARE", "sub": "PEAK CONTRACTION CUE", "color": COL_GREEN},
            ],
            "out": os.path.join(OUT_DIR, "day1_v26_step0_hook.jpg"),
        },
        {
            "idx": 1,
            "is_wrong": True,
            "frame": f_w45,
            "mode": "xray_joint",
            "active_slot": 0,
            "step_title": "MISTAKE 1 · ROUNDED BACK & SHOULDER SHRUGGING",
            "pct": 86,
            "dial_tag": "JOINT LOAD",
            "kicker": "MISTAKE 1 · LEANING OVER & SHRUGGING",
            "line1": "DON'T HUNCH SHOULDERS",
            "line2": "✕ WHY: BODYWEIGHT PRESSING IMPINGES SHOULDER JOINT",
            "splits": [
                ("SHOULDER SHEAR", 86, COL_RED),
                ("ELBOW STRAIN", 74, COL_AMBER),
                ("TRICEPS WORK", 18, COL_RED),
            ],
            "rx": [
                ("SPINE ERROR", "35° HUNCHED BACK"),
                ("FORCE LEAK", "82% TENSION LOST"),
                ("FIX CUE", "DEPRESS SCAPULA DOWN"),
            ],
            "callouts": [
                {"target": (728, 618), "box_center": (245, 610), "title": "GLENOHUMERAL JOINT", "sub": "86% IMPINGEMENT RISK", "color": COL_RED},
                {"target": (812, 740), "box_center": (245, 755), "title": "ROUNDED UPPER BACK", "sub": "35° EXCESSIVE LEAN", "color": COL_RED},
            ],
            "out": os.path.join(OUT_DIR, "day1_v26_step1_mistake1.jpg"),
        },
        {
            "idx": 2,
            "is_wrong": True,
            "frame": f_c14,
            "mode": "anterior_deltoid",
            "active_slot": 1,
            "step_title": "MISTAKE 2 · 90° HALF REPS & ELBOW LIFTING",
            "pct": 72,
            "dial_tag": "DELT STEAL",
            "kicker": "MISTAKE 2 · STOPPING SHORT & MOVING ELBOWS",
            "line1": "AVOID 90° HALF REPS",
            "line2": "✕ WHY: LIFTING ELBOWS SHIFTS 72% LOAD TO FRONT DELTS",
            "splits": [
                ("FRONT DELTOID", 72, COL_RED),
                ("LATERAL HEAD", 28, COL_AMBER),
                ("LONG HEAD", 19, COL_RED),
            ],
            "rx": [
                ("ROM ERROR", "STOPPED AT 90° BEND"),
                ("COMPENSATION", "ANTERIOR DELT STEAL"),
                ("FIX CUE", "LOCK ELBOWS AT RIBS"),
            ],
            "callouts": [
                {"target": (712, 665), "box_center": (245, 650), "title": "ANTERIOR DELTOID", "sub": "72% LOAD STEAL", "color": COL_RED},
                {"target": (600, 885), "box_center": (245, 885), "title": "INCOMPLETE LOCKOUT", "sub": "STOPPED AT 90° ANGLE", "color": COL_RED},
            ],
            "out": os.path.join(OUT_DIR, "day1_v26_step2_mistake2.jpg"),
        },
        {
            "idx": 3,
            "is_wrong": False,
            "frame": f_c42,
            "mode": "triceps_lat_long",
            "active_slot": 2,
            "step_title": "PRO TIP 1 · POSTURE SETUP & ELBOW HINGE LOCK",
            "pct": 95,
            "dial_tag": "LAT & LONG",
            "kicker": "FORM TIP 1 · STEP BACK & LOCK UPPER ARMS",
            "line1": "PIN ELBOWS TO RIBS",
            "line2": "✓ DO: 15° HINGE · SHOULDERS DOWN · MOVE ONLY FOREARMS",
            "splits": [
                ("LATERAL HEAD", 95, COL_GREEN),
                ("LONG HEAD", 91, COL_GREEN),
                ("FRONT DELT", 9, COL_CYAN),
            ],
            "rx": [
                ("TORSO SETUP", "15° SLIGHT HIP HINGE"),
                ("ELBOW AXIS", "0° FIXED AT RIBS"),
                ("BREATHING", "EXHALE ON PUSH DOWN"),
            ],
            "callouts": [
                {"target": (776, 595), "box_center": (245, 600), "title": "SCAPULA DEPRESSED", "sub": "CHEST UP · 15° LEAN", "color": COL_GREEN},
                {"target": (724, 832), "box_center": (245, 832), "title": "FIXED ELBOW HINGE", "sub": "0° UPPER ARM DRIFT", "color": COL_GREEN},
            ],
            "out": os.path.join(OUT_DIR, "day1_v26_step3_tip1.jpg"),
        },
        {
            "idx": 4,
            "is_wrong": False,
            "frame": f_c64,
            "mode": "triceps_all_three",
            "active_slot": 2,
            "step_title": "PRO TIP 2 · FULL 180° LOCKOUT & ROPE SPREAD",
            "pct": 99,
            "dial_tag": "3 HEADS",
            "kicker": "FORM TIP 2 · FULL EXTENSION & 1-SEC SQUEEZE",
            "line1": "SPREAD ROPE AT BOTTOM",
            "line2": "✓ DO: PULL HANDLES APART AT 180° & HOLD 1 SECOND",
            "splits": [
                ("LATERAL HEAD", 99, COL_GREEN),
                ("LONG HEAD", 95, COL_GREEN),
                ("MEDIAL HEAD", 96, COL_GREEN),
            ],
            "rx": [
                ("TARGET VOLUME", "4 SETS × 12 REPS"),
                ("PEAK HOLD", "1.0s SQUEEZE AT BOTTOM"),
                ("ECCENTRIC", "3.0s SLOW RETURN TO 90°"),
            ],
            "callouts": [
                {"target": (728, 725), "box_center": (245, 710), "title": "ALL 3 TRICEPS HEADS", "sub": "99% PEAK CONTRACTION", "color": COL_GREEN},
                {"target": (588, 968), "box_center": (245, 968), "title": "180° FULL EXTENSION", "sub": "SPREAD HANDLES OUT", "color": COL_GREEN},
            ],
            "out": os.path.join(OUT_DIR, "day1_v26_step4_tip2.jpg"),
        },
    ]

    panels = []
    for ph in phases:
        if ph["mode"] == "xray_joint":
            canvas, targets, _, _ = eng.render_xray_skeletal_frame(ph["frame"], 1.0, 0.90)
        else:
            canvas, targets = v25.render_natural_muscle_frame(ph["frame"], ph["mode"], ph["is_wrong"], 1.0)

        pil_im = Image.fromarray(canvas).convert("RGBA")
        v25.draw_clean_on_body_coaching_geometry(pil_im, ph["idx"], 1.0)
        draw_v26_top_header_and_labeled_dock(pil_im, torso_icons, ph["active_slot"], ph["is_wrong"], ph["step_title"])
        draw_margin_anatomical_callouts(pil_im, ph["callouts"], ph["is_wrong"])
        draw_v26_complete_concept_b_deck(pil_im, targets, ph)

        rgb_im = pil_im.convert("RGB")
        rgb_im.save(ph["out"], quality=95)
        print("Saved:", ph["out"])

        pw, ph_h = 320, 568
        im_r = rgb_im.resize((pw, ph_h), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(im_r)
        col = COL_RED if ph["is_wrong"] else COL_GREEN
        d.rectangle([0, 0, pw - 1, ph_h - 1], outline=col, width=4)
        d.rectangle([0, 0, pw, 42], fill=(12, 14, 18))
        d.text((12, 10), f"{ph['idx']}. {ph['line1']}", font=eng.load_font(17, bold=True), fill=col)
        panels.append(im_r)

    sheet = Image.new("RGB", (320 * 5 + 16 * 6, 568 + 32), (10, 12, 16))
    for i, p in enumerate(panels):
        sheet.paste(p, (16 + i * (320 + 16), 16))
    sheet_p = os.path.join(OUT_DIR, "day1_v26_value_upgrade_preview.jpg")
    sheet.save(sheet_p, quality=95)
    print("Saved v26 preview sheet:", sheet_p)

    # Side-by-side comparison: v25 (Sparse) vs v26 (Complete Coaching & Anatomy Guide)
    v25_tip1 = Image.open(os.path.join(OUT_DIR, "day1_v25_step3_tip1.jpg")).convert("RGB").resize((520, 924), Image.Resampling.LANCZOS)
    v26_tip1 = Image.open(os.path.join(OUT_DIR, "day1_v26_step3_tip1.jpg")).convert("RGB").resize((520, 924), Image.Resampling.LANCZOS)
    comp = Image.new("RGB", (520 * 2 + 48, 924 + 80), (10, 12, 16))
    cd = ImageDraw.Draw(comp)
    cd.text((24, 22), "BEFORE (v25): Bottom Pill Only (Sparse Info)", font=eng.load_font(24, bold=True), fill=(190, 200, 215))
    cd.text((520 + 36, 22), "AFTER (v26): Complete Anatomy, Angles, 3-Head Split & Tempo", font=eng.load_font(24, bold=True), fill=COL_GREEN)
    comp.paste(v25_tip1, (16, 64))
    comp.paste(v26_tip1, (520 + 32, 64))
    comp_p = os.path.join(OUT_DIR, "day1_v25_vs_v26_comparison.jpg")
    comp.save(comp_p, quality=95)
    print("Saved comparison:", comp_p)


if __name__ == "__main__":
    main()
