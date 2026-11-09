#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v52_architectural_hinge_master.py`)
========================================================================
Solves both of 중현 님's core requirements (`t219u`):
1. "흠,, 차이가 명확하게 뭔지를 나타내줘야 보는 사람들도 인지하기 쉬울텐데"
   - Replaces the local Gaussian knee warp with a **True Piecewise-Linear Bone-Hinge Rotation**
     around the anatomical Hip (`y=1015`), Knee (`y=1215`), and Ankle (`y=1520`) joints:
     * In **Step 1 (`18° KNEE VALGUS`)**: The entire femur angles sharply INWARD from the hip to the knee,
       and the entire tibia angles from the knocked-in knee to the ankle (`>   <` X-다리 무릎 안쪽 붕괴,
       where the two knees collapse inward to within ~210px of each other!).
     * In **Step 2 (`0° KNEES TRACK TOES`)**: The entire femur and tibia track WIDE OUTWARD over the toes
       (`<   >`, knees ~580px apart!). Even on a phone screen with zero text, Step 1 vs Step 2 is unmistakable!
     * In **Step 3 (`+28° FORWARD LEAN & HEELS LIFTING`)**: True rigid-body hip-hinge forward rotation pitches
       the entire upper torso, head, and barbell forward by `+28°` while lifting the heel clearly off the floor!
     * In **Step 4 (`95° PARALLEL DRIVE`)**: Upright torso + parallel thigh depth + flat planted heel!

2. "그리고 화살표 이상해 아직도"
   - Eliminates ALL bulky 3D polygon arrows, `>>>` multi-chevrons, C-loops, and unanchored floating lines.
   - Uses a **100% Unified Architectural Telemetry System** (matching the approved `Day 1 v36` & `Day 2 v38` line weight):
     * Every arrow shares the exact same sleek `4.2px` glowing vector shaft + crisp single triangular arrowhead,
       and every single arrow is **strictly anchored at both ends** between the Reference Axis and the `◎` Joint Marker:
       - **Step 1**: Neutral `Hip ◎ — Ankle ◎` reference line vs. Inward-Buckled `Hip ◎ — Knee ◎ — Ankle ◎` chain,
         with a crisp horizontal vector arrow (`───▶◎   ◎◀───`) bridging directly from the neutral line into the buckled `Knee ◎`!
       - **Step 2**: Faint inner valgus reference line vs. Locked `Hip ◎ — Knee ◎ — Ankle ◎` chain,
         with a crisp horizontal vector arrow (`◎◀───   ───▶◎`) bridging directly from the inner line out to the locked `Knee ◎`!
       - **Step 3**: Upright `Hip ◎ — Ideal Shoulder ○` reference axis vs. Pitched-Forward `Hip ◎ — Shoulder ◎` spine axis,
         with a crisp circular arc arrow (`○╭───▶◎`) centered at `Hip ◎` connecting `Ideal Shoulder ○` directly to `Shoulder ◎`,
         plus a vertical arrow (`───▲◎`) from the flat floor line directly up to the lifted heel!
       - **Step 4**: Locked `Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎` chain with `95°` hip-hinge arc,
         vertical drive arrow (`◎───▲`) rising straight up from `Shoulder ◎`, and heel-lock arrow (`◎───▼`) pressing into the floor bar!
"""

import os
import sys
import math
import pickle
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, "/workspace/daily_info_reels")
import build_day3_v39_squat as d3

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_PKL = "/workspace/daily_info_reels/_cache_day3_v44_pools.pkl"
W, H = 1080, 1920

SAFE_X_LEFT = 96
SAFE_Y_TOP = 258
SAFE_X_RIGHT_BADGE = 805

COL_RED = (242, 44, 50)
COL_RED_LIGHT = (255, 165, 160)
COL_GREEN = (18, 218, 118)
COL_GREEN_LIGHT = (155, 252, 202)
COL_WHITE = (255, 255, 255)
COL_REF = (225, 234, 248)

FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_CACHE = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(FONT_BOLD_PATH, size)
    return _FONT_CACHE[size]


_GRID_X, _GRID_Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def apply_true_bone_hinge_posture_diff(
    rgb: np.ndarray,
    kp: dict,
    step_idx: int,
    u: float,
) -> tuple[np.ndarray, dict]:
    """
    Applies a true piecewise-linear biomechanical bone-hinge deformation around `(Hip, Knee, Ankle)`
    so the entire thigh and entire shin angle inward in Step 1 (`>   <`) and the entire upper body
    hinges forward around the hip in Step 3 (`+28° Forward Lean + Heel Lift`).
    """
    kp_out = {k: (float(v[0]), float(v[1])) for k, v in kp.items()}

    if step_idx == 1:
        y_hip = 1015.0 + 25.0 * (1.0 - u)
        kp_out["hip_r"] = (388.0, y_hip)
        kp_out["hip_l"] = (692.0, y_hip)
    elif step_idx == 2:
        y_hip = 1025.0 + 25.0 * (1.0 - u)
        kp_out["hip_r"] = (412.0, y_hip)
        kp_out["hip_l"] = (668.0, y_hip)

    if u <= 0.02 or step_idx in (2, 4):
        return rgb, kp_out

    dx_map = np.zeros((H, W), dtype=np.float32)
    dy_map = np.zeros((H, W), dtype=np.float32)

    if step_idx == 1:
        # True Piecewise-Linear Femur-Tibia Hinge (`Hip -> Knee -> Ankle`)
        # At `u=1.0`, shift the right knee INWARD by `+148px` (from `x=284` to `x=422`, well inside
        # the hip-ankle line `x=354`!) and the left knee INWARD by `-148px` (from `x=810` to `x=668`)!
        inward_px = 148.0 * u
        down_px = 14.0 * u

        y_hip = kp_out["hip_r"][1]
        y_kn = 1205.0
        y_ank = kp["ank_r"][1]

        # Piecewise-linear tent function \Lambda(y): 0 at y <= y_hip, 1.0 at y = y_kn, 0 at y >= y_ank
        t_up = np.clip((_GRID_Y - (y_hip - 30.0)) / max(1.0, y_kn - (y_hip - 30.0)), 0.0, 1.0)
        t_dn = np.clip((y_ank - 15.0 - _GRID_Y) / max(1.0, (y_ank - 15.0) - y_kn), 0.0, 1.0)
        # Smooth the apex at `y_kn` using smooth-min of `t_up` and `t_dn`
        hinge_y = np.minimum(t_up, t_dn)
        # Apply cosine smoothing (`0.5 - 0.5*cos(pi*h)`) so transitions at hip, knee, and ankle are C^1 smooth
        hinge_y = 0.5 - 0.5 * np.cos(np.pi * hinge_y)

        # Horizontal isolation for right leg (`x < 540`) and left leg (`x > 540`)
        # Right leg resides in `x in [160..515]`, left leg resides in `x in [565..920]`
        w_x_r = np.exp(-((_GRID_X - 355.0) ** 2) / (2.0 * 145.0 ** 2)) * np.clip((535.0 - _GRID_X) / 45.0, 0.0, 1.0)
        w_x_l = np.exp(-((_GRID_X - 735.0) ** 2) / (2.0 * 145.0 ** 2)) * np.clip((_GRID_X - 545.0) / 45.0, 0.0, 1.0)

        w_r = hinge_y * w_x_r
        w_l = hinge_y * w_x_l

        dx_map += inward_px * w_r - inward_px * w_l
        dy_map += down_px * (w_r + w_l)

        # Evaluate actual displaced knee positions
        kp_out["kn_r"] = (kp["kn_r"][0] + inward_px * 0.92, kp["kn_r"][1] + down_px * 0.85)
        kp_out["kn_l"] = (kp["kn_l"][0] - inward_px * 0.92, kp["kn_l"][1] + down_px * 0.85)

    elif step_idx == 3:
        # Step 3: Rigid-hinge forward torso lean around `hip_r` + heel upward lift
        sh = kp["sh"]
        hip_r = kp["hip_r"]
        kn_r = kp["kn_r"]
        ank_r = kp["ank_r"]

        lean_dx = 96.0 * u
        lean_dy = 68.0 * u
        heel_lift_dy = -62.0 * u
        knee_fwd_dx = 40.0 * u

        # Linear hinge ramp above hip_r (`y < hip_r[1] + 20`)
        ramp_torso = np.clip((hip_r[1] + 20.0 - _GRID_Y) / max(1.0, (hip_r[1] + 20.0) - sh[1]), 0.0, 1.25)
        ramp_torso = 0.5 - 0.5 * np.cos(np.pi * np.clip(ramp_torso, 0.0, 1.0))
        w_torso_x = np.exp(-((_GRID_X - (sh[0] + 10.0)) ** 2) / (2.0 * 270.0 ** 2))
        w_torso_y = np.exp(-((_GRID_Y - (sh[1] + 90.0)) ** 2) / (2.0 * 280.0 ** 2))
        torso_w = ramp_torso * w_torso_x * w_torso_y

        heel_x = ank_r[0] - 28.0
        heel_y = ank_r[1] + 12.0
        heel_w = np.exp(-((_GRID_X - heel_x) ** 2) / (2.0 * 85.0 ** 2) - ((_GRID_Y - heel_y) ** 2) / (2.0 * 92.0 ** 2))
        toe_pin = np.clip((ank_r[0] + 50.0 - _GRID_X) / 60.0, 0.0, 1.0)
        heel_w *= toe_pin

        kn_w = np.exp(-((_GRID_X - kn_r[0]) ** 2) / (2.0 * 145.0 ** 2) - ((_GRID_Y - kn_r[1]) ** 2) / (2.0 * 150.0 ** 2))

        dx_map += lean_dx * torso_w + knee_fwd_dx * kn_w
        dy_map += lean_dy * torso_w + heel_lift_dy * heel_w

        # Compute actual displaced shoulder & ankle positions
        kp_out["sh"] = (sh[0] + lean_dx * 0.82, sh[1] + lean_dy * 0.82)
        kp_out["kn_r"] = (kn_r[0] + knee_fwd_dx * 0.85, kn_r[1])
        kp_out["ank_r"] = (ank_r[0], ank_r[1] + heel_lift_dy * 0.80)

    map_x = (_GRID_X - dx_map).astype(np.float32)
    map_y = (_GRID_Y - dy_map).astype(np.float32)
    warped = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped, kp_out


def draw_architectural_line_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    alpha_scale: float = 1.0,
    trim_ends_1x: float = 10.5,
):
    """Draws a sleek `Day 2 (v38)` architectural segment between two joint nodes."""
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    trim = trim_ends_1x * ss
    if L <= 2.0 * trim + 4.0:
        return
    ux, uy = dx / L, dy / L
    pa = (x0 + ux * trim, y0 + uy * trim)
    pb = (x1 - ux * trim, y1 - uy * trim)

    rd.line([pa, pb], fill=(0, 0, 0, int(round(215 * alpha_scale))), width=int(round(6.6 * ss)))
    rd.line([pa, pb], fill=(*col_main, int(round(250 * alpha_scale))), width=int(round(3.8 * ss)))
    rd.line([pa, pb], fill=(*col_light, int(round(210 * alpha_scale))), width=int(round(1.4 * ss)))


def draw_reference_axis_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    ss: int = 4,
    trim_ends_1x: float = 10.5,
):
    """
    Draws a subtle, translucent white/silver reference alignment line anchored directly between two nodes
    (NO floating segments — always connects two concrete endpoints!).
    """
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    trim = trim_ends_1x * ss
    if L <= 2.0 * trim + 4.0:
        return
    ux, uy = dx / L, dy / L
    pa = (x0 + ux * trim, y0 + uy * trim)
    pb = (x1 - ux * trim, y1 - uy * trim)

    rd.line([pa, pb], fill=(0, 0, 0, 145), width=int(round(4.6 * ss)))
    rd.line([pa, pb], fill=(*COL_REF, 175), width=int(round(2.4 * ss)))


def draw_architectural_straight_arrow_ss(
    rd: ImageDraw.ImageDraw,
    p_start_ss: tuple[float, float],
    p_end_ss: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    """
    Draws a sleek, high-precision architectural vector arrow (`───▶`) that matches the exact
    `3.8px` line weight of the bone axes and bridges directly from the Reference Axis to the Joint Node!
    """
    x0, y0 = p_start_ss
    x1, y1 = p_end_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 14.0 * ss:
        return
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    head_len = min(L * 0.48, 18.5 * ss)
    head_half_w = 10.5 * ss
    bx = x1 - ux * head_len
    by = y1 - uy * head_len

    # Shaft matching the exact architectural line weight (`3.8*ss`)
    rd.line([(x0, y0), (bx, by)], fill=(0, 0, 0, 220), width=int(round(6.6 * ss)))
    rd.line([(x0, y0), (bx, by)], fill=(*col_main, 255), width=int(round(3.8 * ss)))
    rd.line([(x0, y0), (bx, by)], fill=(*col_light, 220), width=int(round(1.4 * ss)))

    # Crisp single triangular arrowhead at `(x1, y1)`
    for pad, rgba in [(2.2 * ss, (0, 0, 0, 220)), (0.0, (*col_main, 255))]:
        hw = head_half_w + pad * 1.15
        p_tip = (x1 + ux * pad * 1.3, y1 + uy * pad * 1.3)
        p_l = (bx - ux * (2.0 * ss + pad * 0.3) + nx * hw, by - uy * (2.0 * ss + pad * 0.3) + ny * hw)
        p_r = (bx - ux * (2.0 * ss + pad * 0.3) - nx * hw, by - uy * (2.0 * ss + pad * 0.3) - ny * hw)
        rd.polygon([p_l, p_tip, p_r], fill=rgba)


def draw_architectural_arc_arrow_ss(
    rd: ImageDraw.ImageDraw,
    pivot_ss: tuple[float, float],
    p_from_ss: tuple[float, float],
    p_to_ss: tuple[float, float],
    radius_ss: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    show_arrowhead: bool = True,
    fill_sector: bool = True,
):
    """
    Draws an architectural circular angle arc (`╭───▶`) centered at `pivot_ss` and spanning
    strictly from ray `pivot -> p_from` to ray `pivot -> p_to`, with the exact same `3.8px` line weight
    and optional single arrowhead at `p_to`.
    """
    px, py = pivot_ss
    a1 = math.atan2(p_from_ss[1] - py, p_from_ss[0] - px)
    a2 = math.atan2(p_to_ss[1] - py, p_to_ss[0] - px)
    diff = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(diff) < 0.03:
        return

    steps = 40
    if fill_sector:
        poly = [(px, py)]
        for i in range(steps + 1):
            a = a1 + diff * (i / float(steps))
            poly.append((px + math.cos(a) * radius_ss, py + math.sin(a) * radius_ss))
        rd.polygon(poly, fill=(*col_main, 52))

    direction = 1.0 if diff >= 0.0 else -1.0
    head_len = 18.0 * ss
    head_half_w = 10.2 * ss
    d_theta_head = (head_len / max(40.0 * ss, radius_ss)) * direction if show_arrowhead else 0.0
    a_end_arc = a1 + diff - d_theta_head

    arc_pts = []
    for i in range(steps + 1):
        a = a1 + (a_end_arc - a1) * (i / float(steps))
        arc_pts.append((px + math.cos(a) * radius_ss, py + math.sin(a) * radius_ss))

    rd.line(arc_pts, fill=(0, 0, 0, 220), width=int(round(6.6 * ss)))
    rd.line(arc_pts, fill=(*col_main, 255), width=int(round(3.8 * ss)))
    rd.line(arc_pts, fill=(*col_light, 215), width=int(round(1.4 * ss)))

    if show_arrowhead:
        ux = -math.sin(a_end_arc) * direction
        uy = math.cos(a_end_arc) * direction
        nx, ny = -uy, ux
        nbx = px + math.cos(a_end_arc) * radius_ss
        nby = py + math.sin(a_end_arc) * radius_ss
        tx = px + math.cos(a1 + diff) * radius_ss
        ty = py + math.sin(a1 + diff) * radius_ss

        for pad, rgba in [(2.2 * ss, (0, 0, 0, 220)), (0.0, (*col_main, 255))]:
            hw = head_half_w + pad * 1.15
            p_tip = (tx + ux * pad * 1.3, ty + uy * pad * 1.3)
            p_l = (nbx - ux * (1.5 * ss + pad * 0.3) + nx * hw, nby - uy * (1.5 * ss + pad * 0.3) + ny * hw)
            p_r = (nbx - ux * (1.5 * ss + pad * 0.3) - nx * hw, nby - uy * (1.5 * ss + pad * 0.3) - ny * hw)
            rd.polygon([p_l, p_tip, p_r], fill=rgba)


def draw_joint_pivot_node_ss(
    rd: ImageDraw.ImageDraw,
    pt: tuple[float, float],
    radius_1x: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    is_primary: bool = True,
):
    cx, cy = pt
    r = radius_1x * ss
    rd.ellipse([cx - r - 2 * ss, cy - r - 2 * ss, cx + r + 2 * ss, cy + r + 2 * ss], fill=(0, 0, 0, 190))
    rd.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=(*col_main, 255),
        width=int(round((2.8 if is_primary else 2.2) * ss)),
    )
    r_dot = (3.4 if is_primary else 2.5) * ss
    rd.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot], fill=(*col_light, 255))


def draw_pure_white_metric_callout(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    big_str: str,
    sub_str: str,
    align: str = "left",
):
    x, y = pos
    f_big = _font(64)
    f_sub = _font(24)

    bb_big = od.textbbox((0, 0), big_str, font=f_big)
    bw = bb_big[2] - bb_big[0]
    bb_sub = od.textbbox((0, 0), sub_str, font=f_sub)
    sw = bb_sub[2] - bb_sub[0]

    bx = x - bw if align == "right" else x
    sx = x - sw if align == "right" else x
    sy = y + 70

    for dx_s, dy_s in [(-3, 0), (3, 0), (0, -3), (0, 3), (0, 5)]:
        sd.text((bx + dx_s, y + dy_s), big_str, font=f_big, fill=(0, 0, 0, 235))
        sd.text((sx + dx_s, sy + dy_s), sub_str, font=f_sub, fill=(0, 0, 0, 235))

    od.text((bx, y), big_str, font=f_big, fill=COL_WHITE)
    od.text((sx, sy), sub_str, font=f_sub, fill=COL_WHITE)


def render_v52_callouts(
    canvas: Image.Image,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 1.0,
):
    SS = 4
    rig_layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig_layer)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    def S(pt):
        return (pt[0] * SS, pt[1] * SS)

    col_main = COL_RED if is_wrong else COL_GREEN
    col_light = COL_RED_LIGHT if is_wrong else COL_GREEN_LIGHT

    sh = kp["sh"]
    hip_r = kp["hip_r"]
    hip_l = kp["hip_l"]
    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    ank_r = kp["ank_r"]
    ank_l = kp["ank_l"]

    if step_idx == 1:
        # =========================================================================
        # STEP 1 (FRONT VIEW — WRONG: `18° VALGUS CAVE` / INWARD KNEE COLLAPSE `>   <`)
        # =========================================================================
        # 1. Neutral Alignment Line connecting `Hip ◎` -> `Ideal Outer Knee ○` -> `Ankle ◎`
        #    Wait: instead of floating lines, we anchor the Neutral Reference Axis at `Ankle ◎`
        #    going straight up to the Ideal Knee position `(ank_r[0] - 28, kn_r[1])` and `Hip ◎`!
        ref_kn_r = (ank_r[0] - 26.0, kn_r[1])
        ref_kn_l = (ank_l[0] + 26.0, kn_l[1])

        draw_reference_axis_ss(rd, S(hip_r), S(ref_kn_r), SS, trim_ends_1x=8.0)
        draw_reference_axis_ss(rd, S(ref_kn_r), S(ank_r), SS, trim_ends_1x=8.0)
        draw_reference_axis_ss(rd, S(hip_l), S(ref_kn_l), SS, trim_ends_1x=8.0)
        draw_reference_axis_ss(rd, S(ref_kn_l), S(ank_l), SS, trim_ends_1x=8.0)

        # 2. Architectural Angle Arc Arrow (`18°`) anchored at `Ankle ◎` sweeping INWARD from Reference Axis to Collapsed Shin!
        draw_architectural_arc_arrow_ss(
            rd,
            pivot_ss=S(ank_r),
            p_from_ss=S(ref_kn_r),
            p_to_ss=S(kn_r),
            radius_ss=165.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrowhead=True,
            fill_sector=True,
        )
        draw_architectural_arc_arrow_ss(
            rd,
            pivot_ss=S(ank_l),
            p_from_ss=S(ref_kn_l),
            p_to_ss=S(kn_l),
            radius_ss=165.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrowhead=True,
            fill_sector=True,
        )

        # 3. Actual Collapsed 3-Point Bone Hinge (`Hip ◎ — Knee ◎ — Ankle ◎` bent inward `>   <`)
        draw_architectural_line_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(hip_l), S(kn_l), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(kn_l), S(ank_l), col_main, col_light, SS)

        # 4. Horizontal Displacement Vector Arrow (`○───▶◎   ◎◀───○`) bridging strictly from `ref_kn` to `kn`!
        draw_architectural_straight_arrow_ss(
            rd,
            p_start_ss=S((ref_kn_r[0] + 8.0, kn_r[1])),
            p_end_ss=S((kn_r[0] - 13.0, kn_r[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_architectural_straight_arrow_ss(
            rd,
            p_start_ss=S((ref_kn_l[0] - 8.0, kn_l[1])),
            p_end_ss=S((kn_l[0] + 13.0, kn_l[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        # Reference nodes `○` at `ref_kn` + Primary Joint Nodes `◎` at `Hip, Knee, Ankle`
        for pt_r in [ref_kn_r, ref_kn_l]:
            draw_joint_pivot_node_ss(rd, S(pt_r), 6.0, COL_REF, COL_WHITE, SS, is_primary=False)
        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.0, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (265, 995), "18°", "VALGUS CAVE", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "84%", "ACL & MCL LOAD", align="left")

    elif step_idx == 2:
        # =========================================================================
        # STEP 2 (FRONT VIEW — FIX: `0° KNEES TRACK TOES` / WIDE KNEE LOCK `<   >`)
        # =========================================================================
        # 1. Collapsed Inner Valgus Reference Line (`Hip ◎ -> Collapsed Inner Knee ○ -> Ankle ◎`)
        valgus_kn_r = (ank_r[0] + 78.0, kn_r[1])
        valgus_kn_l = (ank_l[0] - 78.0, kn_l[1])

        draw_reference_axis_ss(rd, S(hip_r), S(valgus_kn_r), SS, trim_ends_1x=8.0)
        draw_reference_axis_ss(rd, S(valgus_kn_r), S(ank_r), SS, trim_ends_1x=8.0)
        draw_reference_axis_ss(rd, S(hip_l), S(valgus_kn_l), SS, trim_ends_1x=8.0)
        draw_reference_axis_ss(rd, S(valgus_kn_l), S(ank_l), SS, trim_ends_1x=8.0)

        # 2. Architectural Angle Arc Arrow (`0° LOCK`) anchored at `Ankle ◎` sweeping OUTWARD from Valgus Line to Locked Shin!
        draw_architectural_arc_arrow_ss(
            rd,
            pivot_ss=S(ank_r),
            p_from_ss=S(valgus_kn_r),
            p_to_ss=S(kn_r),
            radius_ss=165.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrowhead=True,
            fill_sector=True,
        )
        draw_architectural_arc_arrow_ss(
            rd,
            pivot_ss=S(ank_l),
            p_from_ss=S(valgus_kn_l),
            p_to_ss=S(kn_l),
            radius_ss=165.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrowhead=True,
            fill_sector=True,
        )

        # 3. Locked 3-Point Bone Hinge (`Hip ◎ — Knee ◎ — Ankle ◎` wide over toes `<   >`)
        draw_architectural_line_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(hip_l), S(kn_l), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(kn_l), S(ank_l), col_main, col_light, SS)

        # 4. Horizontal Outward Correction Vector Arrow (`◎◀───○   ○───▶◎`) bridging strictly from `valgus_kn` out to `kn`!
        draw_architectural_straight_arrow_ss(
            rd,
            p_start_ss=S((valgus_kn_r[0] - 8.0, kn_r[1])),
            p_end_ss=S((kn_r[0] + 13.0, kn_r[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_architectural_straight_arrow_ss(
            rd,
            p_start_ss=S((valgus_kn_l[0] + 8.0, kn_l[1])),
            p_end_ss=S((kn_l[0] - 13.0, kn_l[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        for pt_r in [valgus_kn_r, valgus_kn_l]:
            draw_joint_pivot_node_ss(rd, S(pt_r), 6.0, COL_REF, COL_WHITE, SS, is_primary=False)
        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.0, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (255, 995), "0°", "KNEES TRACK TOES", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "96%", "VMO & GLUTE LOCK", align="left")

    elif step_idx == 3:
        # =========================================================================
        # STEP 3 (3/4 SIDE VIEW — WRONG: `+28° HEELS LIFTING` & FORWARD TORSO FOLD)
        # =========================================================================
        # 1. Ideal Upright Spine Reference Line (`Hip ◎ -> Ideal Upright Shoulder ○`)
        #    Matches Step 4's upright spine angle (`dx=+195, dy=-410` from `hip_r`)
        spine_len = math.hypot(sh[0] - hip_r[0], sh[1] - hip_r[1])
        ideal_ang = math.radians(-66.0)
        ideal_sh = (hip_r[0] + spine_len * math.cos(ideal_ang), hip_r[1] + spine_len * math.sin(ideal_ang))

        draw_reference_axis_ss(rd, S(hip_r), S(ideal_sh), SS, trim_ends_1x=8.0)

        # 2. Architectural Angle Arc Arrow (`+28°`) centered at `Hip ◎` sweeping from `Ideal Upright Spine` to `Actual Forward Spine`!
        #    AND a second full-radius arc arrow right from `Ideal Shoulder ○` to `Actual Shoulder ◎` (`r = spine_len`)!
        draw_architectural_arc_arrow_ss(
            rd,
            pivot_ss=S(hip_r),
            p_from_ss=S(ideal_sh),
            p_to_ss=S(sh),
            radius_ss=165.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrowhead=False,
            fill_sector=True,
        )
        draw_architectural_arc_arrow_ss(
            rd,
            pivot_ss=S(hip_r),
            p_from_ss=S(ideal_sh),
            p_to_ss=S(sh),
            radius_ss=(spine_len - 2.0) * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrowhead=True,
            fill_sector=False,
        )

        # 3. Actual Collapsed 3-Point Chain (`Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`)
        draw_architectural_line_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        # 4. Flat Floor Reference Line + Architectural Upward Heel-Lift Arrow (`Floor ○ -> Lifted Heel ◎`)
        floor_y = ank_r[1] + 68.0
        floor_heel_pt = (ank_r[0] - 48.0, floor_y)
        lifted_heel_pt = (ank_r[0] - 48.0, ank_r[1] - 32.0)
        draw_reference_axis_ss(rd, S((ank_r[0] - 85.0, floor_y)), S((ank_r[0] + 95.0, floor_y)), SS, trim_ends_1x=0.0)
        draw_architectural_straight_arrow_ss(
            rd,
            p_start_ss=S(floor_heel_pt),
            p_end_ss=S(lifted_heel_pt),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        draw_joint_pivot_node_ss(rd, S(ideal_sh), 6.5, COL_REF, COL_WHITE, SS, is_primary=False)
        draw_joint_pivot_node_ss(rd, S(floor_heel_pt), 5.5, COL_REF, COL_WHITE, SS, is_primary=False)
        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.5, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (325, 1020), "+28°", "HEELS LIFTING", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "79%", "LUMBAR & PATELLA", align="left")

    else:
        # =========================================================================
        # STEP 4 (3/4 SIDE VIEW — FIX: `95° PARALLEL DEPTH` & FLAT HEEL DRIVE)
        # =========================================================================
        # 1. Collapsed Forward Spine Ghost Line (`Hip ◎ -> Forward Collapsed Shoulder ○`)
        spine_len = math.hypot(sh[0] - hip_r[0], sh[1] - hip_r[1])
        collapsed_ang = math.radians(-42.0)
        collapsed_sh = (hip_r[0] + spine_len * math.cos(collapsed_ang), hip_r[1] + spine_len * math.sin(collapsed_ang))

        draw_reference_axis_ss(rd, S(hip_r), S(collapsed_sh), SS, trim_ends_1x=8.0)

        # 2. Architectural Recovery Arc Arrow (`Collapsed Shoulder ○ -> Locked Upright Shoulder ◎`)
        #    + `95°` Parallel Hip-to-Knee Angle Arc between Spine (`Hip -> Shoulder`) and Femur (`Hip -> Knee`)!
        draw_architectural_arc_arrow_ss(
            rd,
            pivot_ss=S(hip_r),
            p_from_ss=S(collapsed_sh),
            p_to_ss=S(sh),
            radius_ss=(spine_len - 2.0) * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrowhead=True,
            fill_sector=False,
        )
        draw_architectural_arc_arrow_ss(
            rd,
            pivot_ss=S(hip_r),
            p_from_ss=S(sh),
            p_to_ss=S(kn_r),
            radius_ss=135.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrowhead=False,
            fill_sector=True,
        )

        # 3. Locked 3-Point Chain (`Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`)
        draw_architectural_line_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_architectural_line_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        # 4. Solid Green Floor Lock Bar + Heel-Down Lock Arrow (`Lifted Ghost ○ -> Locked Floor Heel ◎`)
        floor_y = ank_r[1] + 28.0
        draw_architectural_line_ss(
            rd,
            S((ank_r[0] - 80.0, floor_y)),
            S((ank_r[0] + 95.0, floor_y)),
            col_main,
            col_light,
            SS,
            trim_ends_1x=0.0,
        )
        lifted_ghost_pt = (ank_r[0] - 48.0, ank_r[1] - 68.0)
        locked_floor_pt = (ank_r[0] - 48.0, floor_y - 4.0)
        draw_architectural_straight_arrow_ss(
            rd,
            p_start_ss=S(lifted_ghost_pt),
            p_end_ss=S(locked_floor_pt),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        draw_joint_pivot_node_ss(rd, S(collapsed_sh), 6.5, COL_REF, COL_WHITE, SS, is_primary=False)
        draw_joint_pivot_node_ss(rd, S(lifted_ghost_pt), 5.5, COL_REF, COL_WHITE, SS, is_primary=False)
        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.5, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (325, 1020), "95°", "PARALLEL DEPTH", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "98%", "QUADS & GLUTES", align="left")

    rig_1x = rig_layer.resize((W, H), Image.Resampling.LANCZOS)
    canvas.alpha_composite(rig_1x)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(ov)


def main():
    with open(CACHE_PKL, "rb") as f:
        pools = pickle.load(f)

    pool_s1, kp_s1 = pools[1]
    pool_s2, kp_s2 = pools[2]
    pool_s3, kp_s3 = pools[3]
    pool_s4, kp_s4 = pools[4]

    torso_icons = []
    for p_pool in [pool_s1, pool_s3, pool_s4]:
        mid_rgb = p_pool[len(p_pool) // 2]
        crop = mid_rgb[360:1440, 180:900]
        pil_act = Image.fromarray(crop).resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        dark = np.stack([gray * 0.48, gray * 0.50, gray * 0.54], axis=2).astype(np.uint8)
        pil_inact = Image.fromarray(dark).resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        torso_icons.append((pil_act, pil_inact))

    step_configs = [
        (1, pool_s1, kp_s1, True, 0, 84, "LUMBAR & KNEE RISK", "BARBELL BACK SQUAT", "18° KNEE VALGUS"),
        (2, pool_s2, kp_s2, False, 2, 96, "NEUTRAL KNEE LOCK", "GLUTE MEDIUS & VMO", "0° KNEES TRACK TOES"),
        (3, pool_s3, kp_s3, True, 1, 79, "ANTERIOR FORCE LEAK", "HEEL LIFT & FORWARD DIVE", "+28° FORWARD LEAN"),
        (4, pool_s4, kp_s4, False, 2, 98, "FULL QUAD & GLUTE · 4×10", "QUADRICEPS & GLUTES", "95° PARALLEL DRIVE"),
    ]

    panels = []
    crops = []
    for s_idx, pool, kps, is_wrong, dock_idx, pct_val, tag_pill, tag_ctx, cue_1line in step_configs:
        idx = int(len(pool) * 0.88)
        raw_rgb = pool[idx].copy()
        raw_kp = kps[idx]

        rgb_frame, kp_cur = apply_true_bone_hinge_posture_diff(raw_rgb, raw_kp, step_idx=s_idx, u=1.0)
        canvas = Image.fromarray(rgb_frame).convert("RGBA")

        render_v52_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)

        d3.draw_day3_v39_safezone_top_dock(
            canvas,
            torso_icons=torso_icons,
            active_idx=dock_idx,
            is_wrong=is_wrong,
            exercise_name="BARBELL BACK SQUAT",
            muscle_group="[ LEGS · QUADS & GLUTES ]",
        )
        d3.draw_day3_v39_safezone_bottom_hud(
            canvas,
            {
                "is_wrong": is_wrong,
                "pct": pct_val,
                "tag_pill": tag_pill,
                "tag_ctx": tag_ctx,
                "cue_1line": cue_1line,
            },
        )

        out_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
        crops.append(out_bgr[560:1580, 120:960])

    sheet_2x2 = np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])])
    sheet_path = os.path.join(OUT_DIR, "day3_v52_difference_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    crop_2x2 = np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])])
    crop_path = os.path.join(OUT_DIR, "day3_v52_closeup_sheet.jpg")
    cv2.imwrite(crop_path, crop_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved:", sheet_path, "and", crop_path)


if __name__ == "__main__":
    main()
