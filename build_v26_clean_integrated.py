#!/usr/bin/env python3
"""
GymForm 3D — v26.1 Frameless Integrated Coaching Master
Combines:
1. Labeled Top 3-Thumbnail Dock (`1. JOINT LOAD` / `2. DELT STEAL` / `3. 3-HEAD FIX`) + Top-Right `✕/✓` Emblem
2. Frameless In-Studio Angle & Anatomy Callouts (ZERO boxes on the middle screen — crisp floating text + thin leader line)
3. Unified Concept B Lower-Left HUD (270° Circular Ring Dial + Single Cohesive Floating Capsule containing:
   - Mistake / Pro Tip Headline
   - Actionable `✕ WHY` / `✓ DO` Coaching Cue
   - Inline 3-Head Muscle Split Bars + `4×12 · 1s HOLD · 15° LEAN` Prescription Chips)
"""

import os
from PIL import Image, ImageDraw

import gymform_engine as eng
import build_v25_fast_preview as v25

W, H = eng.W, eng.H
OUT_DIR = eng.OUT_DIR
COL_RED = (255, 62, 50)
COL_GREEN = (24, 226, 105)
COL_AMBER = (255, 184, 48)
COL_CYAN = (72, 210, 255)

FONT_TAB_LBL = eng.load_font(13, bold=True)
FONT_FLOAT_BIG = eng.load_font(22, bold=True)
FONT_FLOAT_SUB = eng.load_font(14, bold=True)

FONT_DIAL_PCT = eng.load_font(38, bold=True)
FONT_DIAL_TAG = eng.load_font(13, bold=True)

FONT_PILL_KICKER = eng.load_font(15, bold=True)
FONT_PILL_MAIN = eng.load_font(34, bold=True)
FONT_PILL_TIP = eng.load_font(16, bold=True)
FONT_CHIP = eng.load_font(14, bold=True)


def draw_labeled_top_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    tw, th = 144, 108
    lbl_h = 26
    gap = 12
    pad = 12
    bar_x = 84
    bar_y = 236
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + lbl_h + 2 * pad

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=24,
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
        tb = od.textbbox((0, 0), tab_labels[i], font=FONT_TAB_LBL)
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
            txt_col = (10, 14, 18, 255) if i == 2 else (255, 255, 255, 255)
            od.text((tx + (tw - (tb[2] - tb[0])) // 2, ty + th + 4), tab_labels[i], font=FONT_TAB_LBL, fill=txt_col)
        else:
            od.text((tx + (tw - (tb[2] - tb[0])) // 2, ty + th + 4), tab_labels[i], font=FONT_TAB_LBL, fill=(155, 168, 188, 215))

    # Top-Right ONLY Status Emblem (`x = 912`)
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


def draw_frameless_biomechanical_cue(
    canvas: Image.Image,
    target_xy: tuple[int, int],
    anchor_xy: tuple[int, int],
    big_text: str,
    sub_text: str,
    is_wrong: bool,
):
    """
    Draws a sleek, FRAMELESS floating biomechanical angle/posture cue in the open dark studio space
    connected by a fine 2px pointer line to the exact joint. ZERO rectangular box!
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    col = COL_RED if is_wrong else COL_GREEN

    tx, ty = target_xy
    ax, ay = anchor_xy

    # Measure text width for underline
    tb1 = od.textbbox((0, 0), big_text, font=FONT_FLOAT_BIG)
    tb2 = od.textbbox((0, 0), sub_text, font=FONT_FLOAT_SUB)
    w_line = max(tb1[2] - tb1[0], tb2[2] - tb2[0])

    line_end_x = ax + w_line + 10
    pts = [(tx, ty), (line_end_x + 28, ty), (line_end_x, ay + 28), (ax, ay + 28)]
    od.line(pts, fill=(8, 10, 14, 210), width=5, joint="curve")
    od.line(pts, fill=(*col, 235), width=2, joint="curve")
    od.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(255, 255, 255, 255), outline=(*col, 255), width=2)

    # Crisp text with subtle dark drop-shadow (zero box!)
    od.text((ax + 2, ay + 2), big_text, font=FONT_FLOAT_BIG, fill=(0, 0, 0, 210))
    od.text((ax, ay), big_text, font=FONT_FLOAT_BIG, fill=(255, 255, 255, 255))

    od.text((ax + 2, ay + 34), sub_text, font=FONT_FLOAT_SUB, fill=(0, 0, 0, 210))
    od.text((ax, ay + 32), sub_text, font=FONT_FLOAT_SUB, fill=(*col, 255))

    canvas.alpha_composite(ov)


def draw_unified_concept_b_hud(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    phase: dict,
):
    """
    Single cohesive Concept B HUD (`Radial Ring + Unified Floating Capsule`):
    - Ring Dial (`cx = 168, cy = 1305, r = 82`)
    - Unified Floating Pill (`x = 242..792, y = 1205..1405`, radius = 44) containing:
      1. Kicker (`MISTAKE 1 · LEANING & SHRUGGING` or `FORM TIP 1 · POSTURE & HINGE`)
      2. Main Action Headline (`DON'T HUNCH OVER` / `PIN ELBOWS TO RIBS`)
      3. Concrete Why/Do Coaching Cue (`✓ DO: 15° HINGE · SHOULDERS DOWN · FOREARMS ONLY`)
      4. Integrated Bottom Pill Chips inside the capsule (`LAT 95% · LONG 91%` | `4×12 REPS` | `1s HOLD · 3s UP`)
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    ring_cx, ring_cy = 168, 1305
    ring_r = 82

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

    # 2. Unified Floating Pill Capsule (`x = 238..796, y = 1205..1405`, radius = 44)
    px1, py1 = 238, 1205
    px2, py2 = 796, 1405
    od.rounded_rectangle([px1 + 4, py1 + 8, px2 + 4, py2 + 8], radius=44, fill=(4, 6, 10, 155))
    od.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=44,
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

    dd.ellipse([dc - dr, dc - dr, dc + dr, dc + dr], fill=(14, 18, 25, 248), outline=(220, 230, 245, 84), width=2 * ss)
    arc_box = [dc - dr + 11 * ss, dc - dr + 11 * ss, dc + dr - 11 * ss, dc + dr - 11 * ss]
    dd.arc(arc_box, start=135, end=405, fill=(36, 42, 54, 235), width=11 * ss)

    end_deg = 135 + int(round(270 * (pct_val / 100.0)))
    dd.arc(arc_box, start=135, end=end_deg, fill=(*accent, 255), width=11 * ss)

    dial_down = dial.resize((ring_r * 2 + 24, ring_r * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(dial_down, (ring_cx - ring_r - 12, ring_cy - ring_r - 12))

    p_str = f"{pct_val}%"
    pb = od.textbbox((0, 0), p_str, font=FONT_DIAL_PCT)
    od.text((ring_cx - (pb[2] - pb[0]) // 2, ring_cy - 28), p_str, font=FONT_DIAL_PCT, fill=(255, 255, 255, 255))

    dial_tag = phase["dial_tag"]
    sb = od.textbbox((0, 0), dial_tag, font=FONT_DIAL_TAG)
    od.text((ring_cx - (sb[2] - sb[0]) // 2, ring_cy + 18), dial_tag, font=FONT_DIAL_TAG, fill=(*accent, 255))

    # 4. Typography + Integrated Pill Chips inside the single capsule
    tx = 274
    od.text((tx, py1 + 18), phase["kicker"], font=FONT_PILL_KICKER, fill=(*accent, 255))
    od.text((tx, py1 + 44), phase["line1"], font=FONT_PILL_MAIN, fill=(255, 255, 255, 255))
    od.text((tx, py1 + 92), phase["line2"], font=FONT_PILL_TIP, fill=(220, 230, 245, 248))

    # Subtle divider + 2 rounded mini-pills inside the bottom of the capsule (`y = 1338..1384`)
    cy1, cy2 = py1 + 132, py1 + 176
    cx_cur = tx
    for idx_c, (chip_txt, is_hi) in enumerate(phase["chips"]):
        cb = od.textbbox((0, 0), chip_txt, font=FONT_CHIP)
        cw = (cb[2] - cb[0]) + 28
        bg_c = (*accent, 42) if is_hi else (28, 34, 46, 210)
        bd_c = (*accent, 185) if is_hi else (200, 212, 230, 55)
        fg_c = (255, 255, 255, 255) if is_hi else (205, 216, 232, 240)
        od.rounded_rectangle([cx_cur, cy1, cx_cur + cw, cy2], radius=22, fill=bg_c, outline=bd_c, width=2)
        od.text((cx_cur + 14, cy1 + 12), chip_txt, font=FONT_CHIP, fill=fg_c)
        cx_cur += cw + 10

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
            "pct": 98,
            "dial_tag": "TRICEPS",
            "kicker": "CABLE ROPE PUSHDOWN · 3-HEAD GUIDE",
            "line1": "2 MISTAKES & 2 PRO TIPS",
            "line2": "✓ DO: 15° LEAN · PIN ELBOWS · SPREAD ROPE",
            "chips": [
                ("LAT 98% · LONG 95% · MED 96%", True),
                ("4 SETS × 12 REPS", False),
            ],
            "float_cue": ((588, 968), (96, 860), "180° FULL LOCKOUT", "SPREAD HANDLES OUT"),
            "out": os.path.join(OUT_DIR, "day1_v26_clean_step0.jpg"),
        },
        {
            "idx": 1,
            "is_wrong": True,
            "frame": f_w45,
            "mode": "xray_joint",
            "active_slot": 0,
            "pct": 86,
            "dial_tag": "JOINT LOAD",
            "kicker": "MISTAKE 1 · LEANING OVER & SHRUGGING",
            "line1": "DON'T HUNCH SHOULDERS",
            "line2": "✕ WHY: BODYWEIGHT PRESSING STRAINS JOINTS",
            "chips": [
                ("SHOULDER SHEAR 86%", True),
                ("TRICEPS DROPS TO 18%", False),
            ],
            "float_cue": ((728, 618), (96, 560), "35° ROUNDED BACK", "ACROMION IMPINGEMENT"),
            "out": os.path.join(OUT_DIR, "day1_v26_clean_step1.jpg"),
        },
        {
            "idx": 2,
            "is_wrong": True,
            "frame": f_c14,
            "mode": "anterior_deltoid",
            "active_slot": 1,
            "pct": 72,
            "dial_tag": "DELT STEAL",
            "kicker": "MISTAKE 2 · STOPPING SHORT AT 90°",
            "line1": "AVOID 90° HALF REPS",
            "line2": "✕ WHY: LIFTING ELBOWS SHIFTS LOAD TO DELTS",
            "chips": [
                ("FRONT DELT STEAL 72%", True),
                ("TRICEPS ONLY 28%", False),
            ],
            "float_cue": ((600, 885), (96, 825), "STOPPED AT 90°", "ZERO LOWER EXTENSION"),
            "out": os.path.join(OUT_DIR, "day1_v26_clean_step2.jpg"),
        },
        {
            "idx": 3,
            "is_wrong": False,
            "frame": f_c42,
            "mode": "triceps_lat_long",
            "active_slot": 2,
            "pct": 95,
            "dial_tag": "LAT & LONG",
            "kicker": "PRO TIP 1 · CHEST UP & SCAPULA DOWN",
            "line1": "PIN ELBOWS TO RIBS",
            "line2": "✓ DO: 15° HINGE · LOCK UPPER ARM · FOREARM ONLY",
            "chips": [
                ("LAT 95% · LONG 91%", True),
                ("DELT STEAL < 9%", False),
            ],
            "float_cue": ((724, 832), (96, 780), "0° ELBOW HINGE", "15° SLIGHT TORSO LEAN"),
            "out": os.path.join(OUT_DIR, "day1_v26_clean_step3.jpg"),
        },
        {
            "idx": 4,
            "is_wrong": False,
            "frame": f_c64,
            "mode": "triceps_all_three",
            "active_slot": 2,
            "pct": 99,
            "dial_tag": "3 HEADS",
            "kicker": "PRO TIP 2 · FULL EXTENSION & SQUEEZE",
            "line1": "SPREAD ROPE AT BOTTOM",
            "line2": "✓ DO: PULL HANDLES APART & HOLD 1 SECOND",
            "chips": [
                ("ALL 3 HEADS 99% EMG", True),
                ("4 × 12 · 1s HOLD · 3s UP", False),
            ],
            "float_cue": ((588, 968), (96, 860), "180° ROPE SPREAD", "1.0s PEAK SQUEEZE"),
            "out": os.path.join(OUT_DIR, "day1_v26_clean_step4.jpg"),
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
        draw_labeled_top_dock(pil_im, torso_icons, ph["active_slot"], ph["is_wrong"])
        fc = ph["float_cue"]
        draw_frameless_biomechanical_cue(pil_im, fc[0], fc[1], fc[2], fc[3], ph["is_wrong"])
        draw_unified_concept_b_hud(pil_im, targets, ph)

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
    sheet_p = os.path.join(OUT_DIR, "day1_v26_clean_integrated_preview.jpg")
    sheet.save(sheet_p, quality=95)
    print("Saved v26.1 clean integrated preview sheet:", sheet_p)


if __name__ == "__main__":
    main()
