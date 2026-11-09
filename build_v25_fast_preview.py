#!/usr/bin/env python3
"""
GymForm 3D — v25 Clean Natural Posture & Practical Coaching Tips (Fast Preview Generator)
Only loads the exact frames needed for the 5 preview panels so it completes in ~4 seconds!
Key Refinements:
1. ZERO artificial 2D arm bending (`apply_dramatic_elbow_swing_posture` removed).
2. Realistic Common Mistakes (`✕`) with distinct natural postures:
   - Step 1 (`✕ DON'T HUNCH OVER`): Natural 3D hunched posture (`WRONG_I2V` f=45) with X-Ray shoulder/elbow joint overload + red rounded-back curve & forward shoulder-roll arrow.
   - Step 2 (`✕ AVOID HALF REPS`): Natural 3D upper-range bent-arm posture (`CORRECT_I2V` f=14, stopped short at 90°) with Anterior Deltoid takeover highlighted in red + red half-rep stop barrier & upward elbow-pull arrow.
3. Actionable Posture Tips (`✓`) with distinct natural postures:
   - Step 3 (`✓ PIN ELBOWS TO RIBS`): Natural 3D mid-extension posture (`CORRECT_I2V` f=42) with green upright spine alignment, scapula-down arrow, and locked elbow hinge ring (`Lateral & Long Head 95%`).
   - Step 4 (`✓ SPREAD ROPE AT BOTTOM`): Natural 3D full bottom lockout (`CORRECT_I2V` f=64) with all 3 Triceps heads (`Lateral · Long · Medial 99%`) + outward rope-spread arrows (`◀ ● ▶`) at the hands (`4 × 12 REPS`).
4. Zero floating text boxes on the mannequin (`y = 400..1180`); all coaching copy lives cleanly inside the Concept B Radial Ring + Floating Pill HUD.
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

import gymform_engine as eng

W, H = eng.W, eng.H
OUT_DIR = eng.OUT_DIR
COL_RED = (255, 62, 50)
COL_GREEN = (24, 226, 105)

FONT_DIAL_PCT = eng.load_font(36, bold=True)
FONT_DIAL_TAG = eng.load_font(13, bold=True)
FONT_PILL_KICKER = eng.load_font(16, bold=True)
FONT_PILL_MAIN = eng.load_font(34, bold=True)
FONT_PILL_SUB = eng.load_font(16, bold=True)


def read_specific_frame(mp4_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, f = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {mp4_path}")
    return f


def render_natural_muscle_frame(
    frame_data: dict,
    target_mode: str,
    is_wrong: bool,
    alpha_scale: float = 1.0,
) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """
    Renders 3D muscle striations on the natural 3D frame WITHOUT any 2D pixel warping.
    """
    mono = frame_data["mono"].astype(np.float32)
    fiber_norm = frame_data["fiber_norm"]
    red_core = frame_data["red_core"]
    natural_sulcus = frame_data["natural_sulcus"]

    y_top = float(frame_data["y_top"])
    y_bot = float(frame_data["y_bot"])
    x_med = float(frame_data["x_med"])
    span_y = max(130.0, y_bot - y_top)

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    sculpt_gate = np.clip((red_core ** 0.75) * 1.25, 0.0, 1.0)

    dx_arm = xx - (x_med - 6.0)
    y_delt_v = (y_top + 0.42 * span_y) - 0.0050 * (dx_arm ** 2) + 0.14 * dx_arm

    is_deltoid = np.clip((y_delt_v - yy) / 11.0, 0.0, 1.0)
    is_triceps = np.clip((yy - (y_delt_v - 2.0)) / 11.0, 0.0, 1.0)

    x_septum = (x_med + 2.0) - (yy - (y_top + 0.43 * span_y)) * 0.05
    y_med_arch = (y_bot - 0.25 * span_y) + ((xx - x_med) ** 2) * 0.007

    is_med_head = is_triceps * np.clip((yy - y_med_arch) / 9.0, 0.0, 1.0)
    is_lat_head = is_triceps * np.clip((x_septum - xx + 2.0) / 8.0, 0.0, 1.0) * (1.0 - is_med_head * 0.55)
    is_long_head = is_triceps * np.clip((xx - x_septum + 2.0) / 8.0, 0.0, 1.0) * (1.0 - is_med_head * 0.55)

    groove_delt = np.exp(-((yy - y_delt_v) ** 2) / (2.0 * 2.8 ** 2))
    groove_septum = np.exp(-((xx - x_septum) ** 2) / (2.0 * 2.2 ** 2)) * is_triceps
    groove_med = np.exp(-((yy - y_med_arch) ** 2) / (2.0 * 2.2 ** 2)) * is_triceps

    if target_mode == "anterior_deltoid":
        primary_mask = is_deltoid * sculpt_gate
        sulcus_active = np.clip(groove_delt * 0.75 + natural_sulcus * 0.55, 0.0, 1.0)
    elif target_mode == "triceps_lat_long":
        primary_mask = np.maximum(is_lat_head, is_long_head * 0.96) * sculpt_gate * (1.0 - is_med_head * 0.30)
        sulcus_active = np.clip(groove_delt * 0.65 + groove_septum * 0.90 + natural_sulcus * 0.65, 0.0, 1.0)
    else:
        primary_mask = is_triceps * sculpt_gate
        sulcus_active = np.clip(groove_delt * 0.65 + groove_septum * 0.92 + groove_med * 0.85 + natural_sulcus * 0.68, 0.0, 1.0)

    total_mask = primary_mask * (1.0 - 0.80 * sulcus_active)
    total_mask = cv2.GaussianBlur(total_mask, (0, 0), sigmaX=1.4, sigmaY=1.4)
    total_mask = np.clip(total_mask * 1.16, 0.0, 1.0) * alpha_scale

    f_contrast = np.clip((fiber_norm - 0.17) / 0.70, 0.0, 1.0)
    f_val = (f_contrast ** 0.84) * (1.0 - 0.78 * sulcus_active)

    if is_wrong:
        col_r = 74.0 + 181.0 * (f_val ** 0.60)
        col_g = 10.0 + 80.0 * (f_val ** 1.28) + 150.0 * (f_val ** 2.7)
        col_b = 8.0 + 34.0 * (f_val ** 1.40) + 128.0 * (f_val ** 3.0)
    else:
        col_r = 8.0 + 50.0 * (f_val ** 1.32) + 172.0 * (f_val ** 2.8)
        col_g = 64.0 + 191.0 * (f_val ** 0.60)
        col_b = 12.0 + 56.0 * (f_val ** 1.22) + 132.0 * (f_val ** 2.8)

    painted = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)

    out = mono.copy()
    fill_alpha = np.clip(total_mask * 0.95, 0.0, 0.97)[:, :, None]
    out = out * (1.0 - fill_alpha) + painted * fill_alpha

    sulcus_alpha = (sulcus_active * total_mask * 0.72)[:, :, None]
    out = out * (1.0 - sulcus_alpha) + 10.0 * sulcus_alpha

    def head_centroid(h_mask: np.ndarray, default_xy: tuple[float, float]) -> tuple[float, float]:
        ys, xs = np.where((h_mask * sculpt_gate) > 0.32)
        if len(ys) > 20:
            return float(np.median(xs)), float(np.median(ys))
        return default_xy

    if target_mode == "anterior_deltoid":
        cx, cy = head_centroid(is_deltoid, (x_med - 18.0, y_top + 0.22 * span_y))
        targets = [(cx, cy, "FRONT DELT")]
    elif target_mode == "triceps_lat_long":
        lx, ly = head_centroid(is_lat_head, (x_med - 14.0, y_top + 0.58 * span_y))
        gx, gy = head_centroid(is_long_head, (x_med + 16.0, y_top + 0.62 * span_y))
        targets = [(lx, ly, "LATERAL HEAD"), (gx, gy, "LONG HEAD")]
    else:
        lx, ly = head_centroid(is_lat_head, (x_med - 14.0, y_top + 0.56 * span_y))
        gx, gy = head_centroid(is_long_head, (x_med + 16.0, y_top + 0.60 * span_y))
        mx, my = head_centroid(is_med_head, (x_med - 4.0, y_top + 0.82 * span_y))
        targets = [(lx, ly, "LATERAL HEAD"), (gx, gy, "LONG HEAD"), (mx, my, "MEDIAL HEAD")]

    return np.clip(out, 0, 255).astype(np.uint8), targets


def draw_curved_arrow(
    od: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float = 1.0,
    width: int = 5,
):
    steps = 24
    pts = []
    for i in range(steps + 1):
        u = i / float(steps)
        omt = 1.0 - u
        x = omt * omt * p0[0] + 2 * omt * u * p1[0] + u * u * p2[0]
        y = omt * omt * p0[1] + 2 * omt * u * p1[1] + u * u * p2[1]
        pts.append((x, y))

    od.line(pts, fill=(10, 12, 18, int(210 * alpha_scale)), width=width + 4, joint="curve")
    od.line(pts, fill=(*col, int(245 * alpha_scale)), width=width, joint="curve")

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    L = math.hypot(dx, dy) + 1e-6
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    tip = p2
    b1 = (p2[0] - ux * 18 + nx * 10, p2[1] - uy * 18 + ny * 10)
    b2 = (p2[0] - ux * 18 - nx * 10, p2[1] - uy * 18 - ny * 10)
    od.polygon([tip, b1, b2], fill=(*col, int(255 * alpha_scale)))


def draw_clean_on_body_coaching_geometry(
    canvas: Image.Image,
    phase_idx: int,
    alpha_scale: float = 1.0,
):
    """
    Pure, intuitive visual geometry on the mannequin (ZERO text boxes on the body!):
    - Phase 1 (Mistake 1 `✕`): Rounded back posture curve + forward shoulder-roll arrow
    - Phase 2 (Mistake 2 `✕`): Upper-arm upward pull arrow + red half-rep stop barrier line at 90° forearm height
    - Phase 3 (Form Tip 1 `✓`): Upright neutral spine line + scapula-down arrow + locked elbow hinge ring at ribs
    - Phase 4 (Form Tip 2 `✓` & Hook): Outward rope-spread arrows (`◀ ● ▶`) at full bottom lockout
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    if phase_idx == 1:
        back_pts = [(832, 910), (812, 740), (776, 610), (728, 565)]
        od.line(back_pts, fill=(10, 12, 18, int(200 * alpha_scale)), width=8, joint="curve")
        od.line(back_pts, fill=(*COL_RED, int(230 * alpha_scale)), width=4, joint="curve")
        for bx, by in back_pts[:3]:
            od.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=(255, 255, 255, int(245 * alpha_scale)), outline=(*COL_RED, int(255 * alpha_scale)), width=2)
        draw_curved_arrow(od, (775, 515), (710, 500), (665, 565), COL_RED, alpha_scale, width=5)

    elif phase_idx == 2:
        # Half-rep / elbow lifting mistake: horizontal red barrier under the high hands + upward elbow drift arrow
        stop_y = 885
        for x_d in range(520, 680, 16):
            od.line([(x_d, stop_y), (min(680, x_d + 9), stop_y)], fill=(*COL_RED, int(220 * alpha_scale)), width=3)
        draw_curved_arrow(od, (724, 835), (680, 790), (635, 735), COL_RED, alpha_scale, width=5)
        od.ellipse([724 - 13, 830 - 13, 724 + 13, 830 + 13], outline=(*COL_RED, int(245 * alpha_scale)), width=3)

    elif phase_idx == 3:
        # Upright spine line + Scapula back & down arrow + Pinned elbow lock ring
        spine_pts = [(806, 910), (792, 745), (776, 595)]
        od.line(spine_pts, fill=(10, 12, 18, int(200 * alpha_scale)), width=8, joint="curve")
        od.line(spine_pts, fill=(*COL_GREEN, int(225 * alpha_scale)), width=4, joint="curve")
        draw_curved_arrow(od, (755, 520), (815, 555), (815, 650), COL_GREEN, alpha_scale, width=5)
        ex, ey = 724, 832
        od.ellipse([ex - 16, ey - 16, ex + 16, ey + 16], fill=(12, 18, 16, int(200 * alpha_scale)), outline=(*COL_GREEN, int(255 * alpha_scale)), width=3)
        od.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(255, 255, 255, int(255 * alpha_scale)))

    elif phase_idx in (0, 4):
        # Full bottom lockout + outward rope spread arrows (`◀ ● ▶`)
        hx, hy = 588, 968
        od.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], fill=(255, 255, 255, int(250 * alpha_scale)), outline=(*COL_GREEN, int(255 * alpha_scale)), width=3)
        draw_curved_arrow(od, (hx - 10, hy - 8), (hx - 48, hy + 10), (hx - 86, hy + 34), COL_GREEN, alpha_scale, width=5)
        draw_curved_arrow(od, (hx + 10, hy + 4), (hx + 52, hy + 22), (hx + 92, hy + 42), COL_GREEN, alpha_scale, width=5)

    canvas.alpha_composite(ov)


def draw_concept_b_hud(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    phase: dict,
    pct_val: int,
):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    ring_cx, ring_cy = 168, 1285
    ring_r = 74

    px, py, _ = targets[0]
    px_i, py_i = int(round(px)), int(round(py))

    if len(targets) > 1:
        for k in range(1, len(targets)):
            nx, ny, _ = targets[k]
            nx_i, ny_i = int(round(nx)), int(round(ny))
            od.line([(px_i, py_i), (nx_i, ny_i)], fill=(*accent, 160), width=2)
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

    px1, py1 = 232, 1210
    px2, py2 = 686, 1360
    od.rounded_rectangle([px1 + 4, py1 + 6, px2 + 4, py2 + 6], radius=36, fill=(4, 6, 10, 145))
    od.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=36,
        fill=(16, 20, 28, 218),
        outline=(225, 235, 250, 68),
        width=2,
    )

    ss = 4
    dsz = (ring_r * 2 + 24) * ss
    dc = dsz // 2
    dr = ring_r * ss
    dial = Image.new("RGBA", (dsz, dsz), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)

    dd.ellipse([dc - dr, dc - dr, dc + dr, dc + dr], fill=(14, 18, 25, 244), outline=(220, 230, 245, 78), width=2 * ss)
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

    tx = 262
    od.text((tx, py1 + 20), phase["kicker"], font=FONT_PILL_KICKER, fill=(*accent, 255))
    od.text((tx, py1 + 48), phase["line1"], font=FONT_PILL_MAIN, fill=(255, 255, 255, 255))
    od.text((tx, py1 + 98), phase["line2"], font=FONT_PILL_SUB, fill=(212, 222, 238, 242))

    canvas.alpha_composite(ov)


def main():
    # Prepare ONLY the 4 needed frames (takes ~3 seconds total!)
    f_w45 = eng.prepare_xray_bone_frame(read_specific_frame(eng.WRONG_I2V, 45))
    f_c14 = eng.prepare_muscle_frame(read_specific_frame(eng.CORRECT_I2V, 14))
    f_c42 = eng.prepare_muscle_frame(read_specific_frame(eng.CORRECT_I2V, 42))
    f_c64 = eng.prepare_muscle_frame(read_specific_frame(eng.CORRECT_I2V, 64))

    size = (144, 114)
    crop_box = (555, 550, 905, 950)
    inact_base = Image.fromarray(f_c42["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 120)))

    x_arr, _, _, _ = eng.render_xray_skeletal_frame(f_w45, 1.0, 0.90)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m1_arr, _ = render_natural_muscle_frame(f_c14, "anterior_deltoid", True, 1.0)
    im1 = Image.fromarray(m1_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m2_arr, _ = render_natural_muscle_frame(f_c64, "triceps_all_three", False, 1.0)
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
            "kicker": "CABLE ROPE PUSHDOWN · FORM GUIDE",
            "line1": "2 MISTAKES & TIPS",
            "line2": "STOP WASTING REPS · ISOLATE 3 HEADS",
            "out": os.path.join(OUT_DIR, "day1_v25_step0_hook.jpg"),
        },
        {
            "idx": 1,
            "is_wrong": True,
            "frame": f_w45,
            "mode": "xray_joint",
            "active_slot": 0,
            "pct": 86,
            "dial_tag": "JOINT LOAD",
            "kicker": "MISTAKE 1 · LEANING & SHRUGGING",
            "line1": "DON'T HUNCH OVER",
            "line2": "ROUNDED SHOULDERS OVERLOAD JOINTS",
            "out": os.path.join(OUT_DIR, "day1_v25_step1_mistake1.jpg"),
        },
        {
            "idx": 2,
            "is_wrong": True,
            "frame": f_c14,
            "mode": "anterior_deltoid",
            "active_slot": 1,
            "pct": 72,
            "dial_tag": "DELT STEAL",
            "kicker": "MISTAKE 2 · HALF REPS & ELBOW LIFT",
            "line1": "AVOID HALF REPS",
            "line2": "STOPPING AT 90° SHIFTS LOAD TO DELTS",
            "out": os.path.join(OUT_DIR, "day1_v25_step2_mistake2.jpg"),
        },
        {
            "idx": 3,
            "is_wrong": False,
            "frame": f_c42,
            "mode": "triceps_lat_long",
            "active_slot": 2,
            "pct": 95,
            "dial_tag": "LAT & LONG",
            "kicker": "FORM TIP 1 · CHEST UP & SCAPULA DOWN",
            "line1": "PIN ELBOWS TO RIBS",
            "line2": "LOCK UPPER ARM · MOVE ONLY FOREARMS",
            "out": os.path.join(OUT_DIR, "day1_v25_step3_tip1.jpg"),
        },
        {
            "idx": 4,
            "is_wrong": False,
            "frame": f_c64,
            "mode": "triceps_all_three",
            "active_slot": 2,
            "pct": 99,
            "dial_tag": "3 HEADS",
            "kicker": "FORM TIP 2 · FULL LOCKOUT · 4 × 12",
            "line1": "SPREAD ROPE APART",
            "line2": "PULL HANDLES OUT & SQUEEZE 1 SEC",
            "out": os.path.join(OUT_DIR, "day1_v25_step4_tip2.jpg"),
        },
    ]

    panels = []
    for ph in phases:
        if ph["mode"] == "xray_joint":
            canvas, targets, _, _ = eng.render_xray_skeletal_frame(ph["frame"], 1.0, 0.90)
        else:
            canvas, targets = render_natural_muscle_frame(ph["frame"], ph["mode"], ph["is_wrong"], 1.0)

        pil_im = Image.fromarray(canvas).convert("RGBA")
        draw_clean_on_body_coaching_geometry(pil_im, ph["idx"], 1.0)
        eng.draw_top_dock_and_right_status(pil_im, torso_icons, ph["active_slot"], ph["is_wrong"], 1.0)
        draw_concept_b_hud(pil_im, targets, ph, ph["pct"])

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
    sheet_p = os.path.join(OUT_DIR, "day1_v25_coaching_preview.jpg")
    sheet.save(sheet_p, quality=95)
    print("Saved v25 preview sheet:", sheet_p)


if __name__ == "__main__":
    main()
