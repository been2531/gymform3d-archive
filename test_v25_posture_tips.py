#!/usr/bin/env python3
"""
GymForm 3D — v25 Natural Posture & Actionable Coaching Tips Preview Builder
- ZERO artificial 2D limb warping (`apply_dramatic_elbow_swing_posture` completely removed).
- Uses natural 3D `WRONG_I2V` (hunched torso, rolled shoulders, forward-drifted elbows) for `✕` Mistake 1 & Mistake 2,
  and natural 3D `CORRECT_I2V` (upright chest, depressed scapula, pinned elbows, bottom rope spread) for `✓` Tip 1 & Tip 2.
- Adds clear, intuitive on-body biomechanical coaching overlays (spine posture line, shoulder-down cue, elbow lock hinge,
  and bottom rope-spread arrows) that do not clutter or cover the triceps striations.
- Concept B Lower-Left HUD (270° Circular Ring Dial + Sleek Floating Capsule) with practical gym coaching tips.
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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
FONT_CUE_TAG = eng.load_font(15, bold=True)


def render_wrong_delt_steal_frame(
    xray_data: dict,
    alpha_scale: float,
    rep_wave: float,
) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """
    Uses the natural 3D `WRONG_I2V` posture (hunched torso, forward elbow drift) with ZERO pixel warping.
    Neutralizes the lower-arm bone glow into sculpted charcoal anatomy and highlights the Anterior Deltoid
    (front shoulder takeover) + upper humerus drift in crimson/orange striations.
    """
    mono = xray_data["mono"].astype(np.float32)
    bone_sharp = xray_data["bone_sharp"]
    bone_mask = xray_data["bone_mask"]
    x_sh, y_sh = float(xray_data["x_sh"]), float(xray_data["y_sh"])
    x_el, y_el = float(xray_data["x_el"]), float(xray_data["y_el"])

    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    # Anterior Deltoid anatomical cap around the front/top of the shoulder (`x_sh - 10, y_sh - 18`)
    delt_cap = np.exp(-(((xx - (x_sh - 8.0)) / 42.0) ** 2 + ((yy - (y_sh - 12.0)) / 54.0) ** 2))
    # Upper-arm drift zone along humerus
    upper_arm = np.exp(-(((xx - (x_sh + 2.0)) / 32.0) ** 2 + ((yy - (y_sh + 65.0)) / 78.0) ** 2))

    # Create directional deltoid striations (-24 deg pennation)
    angle = math.radians(-24.0)
    rot_x = (xx - x_sh) * math.cos(angle) + (yy - y_sh) * math.sin(angle)
    striation_wave = 0.5 + 0.5 * np.sin(rot_x * 0.65)
    f_val = np.clip((bone_sharp * 0.75 + striation_wave * 0.25), 0.0, 1.0)

    active_mask = np.clip(delt_cap * 1.15 + upper_arm * np.clip(bone_mask * 0.85, 0.0, 1.0), 0.0, 1.0)
    active_mask = cv2.GaussianBlur(active_mask, (0, 0), sigmaX=2.0, sigmaY=2.0) * alpha_scale

    col_r = 85.0 + 170.0 * (f_val ** 0.58)
    col_g = 12.0 + 88.0 * (f_val ** 1.25) + 125.0 * (delt_cap ** 1.8)
    col_b = 10.0 + 38.0 * (f_val ** 1.45) + 90.0 * (delt_cap ** 2.4)
    painted = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)

    out = mono.copy()
    fill_a = np.clip(active_mask * 0.92, 0.0, 0.95)[:, :, None]
    out = out * (1.0 - fill_a) + painted * fill_a

    targets = [
        (x_sh - 6.0, y_sh - 10.0, "ANTERIOR DELTOID"),
        (x_el, y_el, "DRIFTED ELBOW"),
    ]
    return np.clip(out, 0, 255).astype(np.uint8), targets


def draw_curved_coaching_arrow(
    od: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    col: tuple[int, int, int],
    alpha_scale: float,
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

    # Arrowhead at p2 along tangent from p1 -> p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    L = math.hypot(dx, dy) + 1e-6
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    tip = p2
    b1 = (p2[0] - ux * 18 + nx * 10, p2[1] - uy * 18 + ny * 10)
    b2 = (p2[0] - ux * 18 - nx * 10, p2[1] - uy * 18 - ny * 10)
    od.polygon([tip, b1, b2], fill=(*col, int(255 * alpha_scale)))


def draw_coaching_tag(
    od: ImageDraw.ImageDraw,
    center_xy: tuple[int, int],
    text: str,
    col: tuple[int, int, int],
    alpha_scale: float,
):
    tb = od.textbbox((0, 0), text, font=FONT_CUE_TAG)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    px, py = center_xy
    x0, y0 = px - tw // 2 - 12, py - th // 2 - 6
    x1, y1 = px + tw // 2 + 12, py + th // 2 + 6
    od.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=12,
        fill=(14, 18, 25, int(225 * alpha_scale)),
        outline=(*col, int(220 * alpha_scale)),
        width=2,
    )
    od.text((px - tw // 2, py - th // 2 - 1), text, font=FONT_CUE_TAG, fill=(255, 255, 255, int(255 * alpha_scale)))


def draw_biomechanical_coaching_overlay(
    canvas: Image.Image,
    phase_idx: int,
    alpha_scale: float,
):
    """
    Draws clear, realistic posture coaching cues on the mannequin WITHOUT covering the triceps striations:
    - Phase 1 (Mistake 1 `✕`): Hunched spine curve along back + forward shoulder roll arrow + `ROUNDED BACK` cue
    - Phase 2 (Mistake 2 `✕`): Forward elbow drift axis + upward/forward swing arrow at elbow + `ELBOW DRIFTS UP` cue
    - Phase 3 (Form Tip 1 `✓`): Upright neutral spine line (`15° HINGE`) + scapula down arrow + pinned elbow lock ring (`0° LOCK`)
    - Phase 4 (Form Tip 2 `✓`): Bottom rope spread outward arrows (`◀ SPREAD OUT ▶`) at the hands + 1s squeeze cue
    """
    if alpha_scale <= 0.02:
        return
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    if phase_idx == 1:
        # Mistake 1: Hunched back curve (behind the arm so it never covers the X-ray arm) + forward shoulder roll arrow
        back_pts = [(832, 910), (812, 740), (776, 610), (728, 565)]
        od.line(back_pts, fill=(10, 12, 18, int(200 * alpha_scale)), width=8, joint="curve")
        od.line(back_pts, fill=(*COL_RED, int(225 * alpha_scale)), width=4, joint="curve")
        for bx, by in [back_pts[0], back_pts[1], back_pts[2]]:
            od.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=(255, 255, 255, int(245 * alpha_scale)), outline=(*COL_RED, int(255 * alpha_scale)), width=2)
        # Curved arrow above shoulder showing forward shoulder roll
        draw_curved_coaching_arrow(od, (775, 515), (710, 500), (665, 565), COL_RED, alpha_scale, width=5)
        draw_coaching_tag(od, (765, 478), "✕ HUNCHED & SHRUGGING", COL_RED, alpha_scale)

    elif phase_idx == 2:
        # Mistake 2: Elbow drifting forward/upward toward cable column + front delt takeover
        sh_x, sh_y = 720, 615
        el_x, el_y = 710, 815
        # Vertical reference line showing where elbow SHOULD stay (pinned back)
        ref_x = 748
        for y_d in range(sh_y + 10, el_y + 20, 16):
            od.line([(ref_x, y_d), (ref_x, min(el_y + 20, y_d + 8))], fill=(210, 220, 235, int(150 * alpha_scale)), width=2)
        # Forward swing arrow from reference axis to drifted elbow and forward toward pulley
        draw_curved_coaching_arrow(od, (752, 835), (685, 835), (625, 780), COL_RED, alpha_scale, width=5)
        od.ellipse([el_x - 13, el_y - 13, el_x + 13, el_y + 13], outline=(*COL_RED, int(250 * alpha_scale)), width=3)
        draw_coaching_tag(od, (615, 742), "✕ ELBOW LIFTS FORWARD", COL_RED, alpha_scale)

    elif phase_idx == 3:
        # Form Tip 1: Upright neutral spine (15° slight hinge) + Scapula pulled back & down + Elbow pinned at side
        spine_pts = [(806, 910), (792, 745), (776, 595)]
        od.line(spine_pts, fill=(10, 12, 18, int(200 * alpha_scale)), width=8, joint="curve")
        od.line(spine_pts, fill=(*COL_GREEN, int(225 * alpha_scale)), width=4, joint="curve")
        # Scapula depression arrow (back & down)
        draw_curved_coaching_arrow(od, (755, 520), (815, 555), (815, 650), COL_GREEN, alpha_scale, width=5)
        draw_coaching_tag(od, (775, 485), "✓ CHEST UP · SHOULDERS DOWN", COL_GREEN, alpha_scale)
        # Elbow lock ring at side
        ex, ey = 724, 832
        od.ellipse([ex - 16, ey - 16, ex + 16, ey + 16], fill=(12, 18, 16, int(200 * alpha_scale)), outline=(*COL_GREEN, int(255 * alpha_scale)), width=3)
        od.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(255, 255, 255, int(255 * alpha_scale)))

    elif phase_idx in (0, 4):
        # Form Tip 2 (and Hook): Full Lockout + Spread Rope Apart at Bottom (`◀ ● ▶`)
        hx, hy = 588, 968
        od.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], fill=(255, 255, 255, int(250 * alpha_scale)), outline=(*COL_GREEN, int(255 * alpha_scale)), width=3)
        # Outward spread arrows at the rope handles
        draw_curved_coaching_arrow(od, (hx - 10, hy - 8), (hx - 48, hy + 10), (hx - 86, hy + 34), COL_GREEN, alpha_scale, width=5)
        draw_curved_coaching_arrow(od, (hx + 10, hy + 4), (hx + 52, hy + 22), (hx + 92, hy + 42), COL_GREEN, alpha_scale, width=5)
        if phase_idx == 4:
            draw_coaching_tag(od, (585, 1042), "✓ SPREAD ROPE & SQUEEZE 1s", COL_GREEN, alpha_scale)

    canvas.alpha_composite(ov)


def draw_concept_b_coaching_hud(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    phase: dict,
    rep_wave: float,
    alpha_scale: float,
):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    ring_cx, ring_cy = 168, 1285
    ring_r = 74

    # 1. Precision Reticle (`◎`) + Multi-head Node Links + Leader Line to Top of Dial
    px, py, _ = targets[0]
    px_i, py_i = int(round(px)), int(round(py))

    if len(targets) > 1:
        for k in range(1, len(targets)):
            nx, ny, _ = targets[k]
            nx_i, ny_i = int(round(nx)), int(round(ny))
            od.line([(px_i, py_i), (nx_i, ny_i)], fill=(*accent, int(160 * alpha_scale)), width=2)
            od.ellipse([nx_i - 8, ny_i - 8, nx_i + 8, ny_i + 8], fill=(12, 15, 20, int(195 * alpha_scale)), outline=(*accent, int(240 * alpha_scale)), width=2)
            od.ellipse([nx_i - 3, ny_i - 3, nx_i + 3, ny_i + 3], fill=(255, 255, 255, int(255 * alpha_scale)))

    r_out = int(16 + 2 * rep_wave)
    od.ellipse([px_i - r_out - 2, py_i - r_out - 2, px_i + r_out + 2, py_i + r_out + 2], outline=(10, 12, 16, int(185 * alpha_scale)), width=3)
    od.ellipse([px_i - r_out, py_i - r_out, px_i + r_out, py_i + r_out], outline=(*accent, int(250 * alpha_scale)), width=2)
    od.ellipse([px_i - 5, py_i - 5, px_i + 5, py_i + 5], fill=(255, 255, 255, int(255 * alpha_scale)), outline=(*accent, int(255 * alpha_scale)), width=2)

    ax, ay = ring_cx, ring_cy - ring_r - 6
    elbow_x = ax
    elbow_y = py_i + 32
    pts = [(px_i - r_out - 3, py_i), (elbow_x + 26, py_i), (elbow_x, elbow_y), (ax, ay)]
    od.line(pts, fill=(10, 12, 16, int(185 * alpha_scale)), width=5, joint="curve")
    od.line(pts, fill=(*accent, int(225 * alpha_scale)), width=2, joint="curve")
    od.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=(*accent, int(255 * alpha_scale)))

    # 2. Floating Rounded Pill Capsule (`x = 232..686, y = 1210..1360`)
    px1, py1 = 232, 1210
    px2, py2 = 686, 1360
    od.rounded_rectangle([px1 + 4, py1 + 6, px2 + 4, py2 + 6], radius=36, fill=(4, 6, 10, int(145 * alpha_scale)))
    od.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=36,
        fill=(16, 20, 28, int(218 * alpha_scale)),
        outline=(225, 235, 250, int(68 * alpha_scale)),
        width=2,
    )

    # 3. 4x-Supersampled 270° Circular Dial
    p_base, p_peak = phase["pct_range"]
    pct_val = int(round(p_base + (p_peak - p_base) * (0.35 + 0.65 * rep_wave)))

    ss = 4
    dsz = (ring_r * 2 + 24) * ss
    dc = dsz // 2
    dr = ring_r * ss
    dial = Image.new("RGBA", (dsz, dsz), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)

    dd.ellipse([dc - dr, dc - dr, dc + dr, dc + dr], fill=(14, 18, 25, int(244 * alpha_scale)), outline=(220, 230, 245, int(78 * alpha_scale)), width=2 * ss)
    arc_box = [dc - dr + 10 * ss, dc - dr + 10 * ss, dc + dr - 10 * ss, dc + dr - 10 * ss]
    dd.arc(arc_box, start=135, end=405, fill=(36, 42, 54, int(230 * alpha_scale)), width=10 * ss)

    end_deg = 135 + int(round(270 * (pct_val / 100.0)))
    dd.arc(arc_box, start=135, end=end_deg, fill=(*accent, int(255 * alpha_scale)), width=10 * ss)

    dial_down = dial.resize((ring_r * 2 + 24, ring_r * 2 + 24), Image.Resampling.LANCZOS)
    ov.alpha_composite(dial_down, (ring_cx - ring_r - 12, ring_cy - ring_r - 12))

    p_str = f"{pct_val}%"
    pb = od.textbbox((0, 0), p_str, font=FONT_DIAL_PCT)
    od.text((ring_cx - (pb[2] - pb[0]) // 2, ring_cy - 26), p_str, font=FONT_DIAL_PCT, fill=(255, 255, 255, int(255 * alpha_scale)))

    dial_tag = phase["dial_tag"]
    sb = od.textbbox((0, 0), dial_tag, font=FONT_DIAL_TAG)
    od.text((ring_cx - (sb[2] - sb[0]) // 2, ring_cy + 18), dial_tag, font=FONT_DIAL_TAG, fill=(*accent, int(255 * alpha_scale)))

    # 4. Floating Pill Typography (Clear Mistake / Actionable Posture Tip)
    tx = 262
    od.text((tx, py1 + 20), phase["kicker"], font=FONT_PILL_KICKER, fill=(*accent, int(255 * alpha_scale)))
    od.text((tx, py1 + 48), phase["line1"], font=FONT_PILL_MAIN, fill=(255, 255, 255, int(255 * alpha_scale)))
    od.text((tx, py1 + 98), phase["line2"], font=FONT_PILL_SUB, fill=(212, 222, 238, int(242 * alpha_scale)))

    canvas.alpha_composite(ov)


V25_PHASES = [
    {
        "idx": 0,
        "is_wrong": False,
        "source": "correct",
        "pool_idx": 58,
        "target_mode": "triceps_all_three",
        "active_slot": 2,
        "dial_tag": "TRICEPS",
        "pct_range": (92, 98),
        "kicker": "CABLE ROPE PUSHDOWN · FORM GUIDE",
        "line1": "2 MISTAKES & TIPS",
        "line2": "HOW TO ISOLATE ALL 3 TRICEPS HEADS",
    },
    {
        "idx": 1,
        "is_wrong": True,
        "source": "wrong_xray",
        "pool_idx": 45,
        "target_mode": "xray_joint",
        "active_slot": 0,
        "dial_tag": "JOINT LOAD",
        "pct_range": (78, 86),
        "kicker": "MISTAKE 1 · LEANING & SHRUGGING",
        "line1": "DON'T HUNCH OVER",
        "line2": "PRESSING WITH SHOULDERS STRAINS JOINTS",
    },
    {
        "idx": 2,
        "is_wrong": True,
        "source": "wrong_delt",
        "pool_idx": 20,
        "target_mode": "anterior_deltoid",
        "active_slot": 1,
        "dial_tag": "DELT STEAL",
        "pct_range": (68, 74),
        "kicker": "MISTAKE 2 · MOVING UPPER ARM",
        "line1": "DON'T LIFT ELBOWS",
        "line2": "FRONT DELT TAKES OVER · TENSION LOST",
    },
    {
        "idx": 3,
        "is_wrong": False,
        "source": "correct",
        "pool_idx": 38,
        "target_mode": "triceps_lat_long",
        "active_slot": 2,
        "dial_tag": "TRICEPS",
        "pct_range": (90, 95),
        "kicker": "FORM TIP 1 · CHEST UP & SHOULDERS DOWN",
        "line1": "PIN ELBOWS TO RIBS",
        "line2": "LOCK UPPER ARM · MOVE ONLY FOREARMS",
    },
    {
        "idx": 4,
        "is_wrong": False,
        "source": "correct",
        "pool_idx": 62,
        "target_mode": "triceps_all_three",
        "active_slot": 2,
        "dial_tag": "3 HEADS",
        "pct_range": (94, 99),
        "kicker": "FORM TIP 2 · FULL LOCKOUT · 4 × 12",
        "line1": "SPREAD ROPE APART",
        "line2": "PULL HANDLES OUT & SQUEEZE 1 SEC",
    },
]


def main():
    eng.load_pools()
    # Update torso icon #1 to use natural wrong_delt_steal instead of warped arm
    sample_m = eng.GLOBAL_MUSCLE_POOL[45]
    sample_x = eng.GLOBAL_XRAY_POOL[35]
    size = (144, 114)
    crop_box = (555, 550, 905, 950)
    inact_base = Image.fromarray(sample_m["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 120)))

    x_arr, _, _, _ = eng.render_xray_skeletal_frame(sample_x, 1.0, 0.85)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    d_arr, _ = render_wrong_delt_steal_frame(eng.GLOBAL_XRAY_POOL[20], 1.0, 0.85)
    im1 = Image.fromarray(d_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m2_arr, _, _, _ = eng.render_targeted_muscle_frame(sample_m, "triceps_all_three", False, 1.0, 0.85)
    im2 = Image.fromarray(m2_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    panels = []
    out_files = [
        os.path.join(OUT_DIR, "day1_v25_step0_hook.jpg"),
        os.path.join(OUT_DIR, "day1_v25_step1_mistake1.jpg"),
        os.path.join(OUT_DIR, "day1_v25_step2_mistake2.jpg"),
        os.path.join(OUT_DIR, "day1_v25_step3_tip1.jpg"),
        os.path.join(OUT_DIR, "day1_v25_step4_tip2.jpg"),
    ]

    for ph, out_p in zip(V25_PHASES, out_files):
        if ph["source"] == "wrong_xray":
            canvas, targets, _, _ = eng.render_xray_skeletal_frame(
                eng.GLOBAL_XRAY_POOL[ph["pool_idx"]], alpha_scale=1.0, rep_wave=0.90
            )
        elif ph["source"] == "wrong_delt":
            canvas, targets = render_wrong_delt_steal_frame(
                eng.GLOBAL_XRAY_POOL[ph["pool_idx"]], alpha_scale=1.0, rep_wave=0.90
            )
        else:
            canvas, targets, _, _ = eng.render_targeted_muscle_frame(
                eng.GLOBAL_MUSCLE_POOL[ph["pool_idx"]],
                target_mode=ph["target_mode"],
                is_wrong=False,
                alpha_scale=1.0,
                rep_wave=0.92,
            )

        pil_im = Image.fromarray(canvas).convert("RGBA")
        draw_biomechanical_coaching_overlay(pil_im, ph["idx"], alpha_scale=1.0)
        eng.draw_top_dock_and_right_status(pil_im, torso_icons, ph["active_slot"], ph["is_wrong"], alpha_scale=1.0)
        draw_concept_b_coaching_hud(pil_im, targets, ph, rep_wave=0.92, alpha_scale=1.0)

        rgb_im = pil_im.convert("RGB")
        rgb_im.save(out_p, quality=95)
        print("Saved:", out_p)

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
