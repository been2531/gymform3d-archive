#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v53_crystal_clear_contrast.py`)
=====================================================================
Addresses every single point from 중현 님 (`t219u`) and the independent visual QC:
1. Unmistakable Posture & Anatomical Highlight Contrast (`차이가 명확하게`):
   - **Step 1 (Front Wrong — `18° KNEE VALGUS`)**: True bone-hinge inward collapse (`>   <`) brings the knees
     sharply inward (`kn_r.x=422, kn_l.x=668`) + vivid crimson 3D surface overload glow across the inner thighs/knees/lumbar.
   - **Step 2 (Front Right — `0° KNEES TRACK TOES`)**: Wide athletic knee tracking over toes (`<   >`, `kn_r.x=266, kn_l.x=847`)
     + vivid emerald 3D muscle activation across the quads (VMO & vastus lateralis) and glutes.
   - **Step 3 (Side Wrong — `+28° FORWARD LEAN & HEELS LIFTING`)**: True hip-hinge forward torso fold (`+28°`) +
     heel lifted clearly off the floor + vivid crimson overload on lumbar spine, anterior knee, and calves.
   - **Step 4 (Side Right — `95° PARALLEL DRIVE`)**: Upright torso + `95°` parallel thigh depth + flat planted heel
     on solid emerald floor bar + vivid emerald quad & glute activation.

2. Zero-Clutter, 100% Intuitive Arrows in Open Space (`화살표가 겹치지 않고 한눈에 직관적`):
   - ZERO duplicate white spiderweb triangles on the legs!
   - Each step has ONLY ONE clean 3-point Joint Alignment Chain (`Hip ◎ — Knee ◎ — Ankle ◎` or `Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`).
   - Arrows sit in **100% unobstructed dark studio space** with zero overlap against bone lines:
     * **Step 1 (Inward Collapse)**: `───▶ ◎     ◎ ◀───` (Sleek crimson inward arrows outside each knee pointing straight INWARD into the collapsing `Knee ◎` + `18°` knee valgus sector).
     * **Step 2 (Outward Lock)**: `◎ ◀───     ───▶ ◎` (Sleek emerald outward arrows in the wide open space between the knees pointing straight OUTWARD into the locked `Knee ◎` + `0°` alignment).
     * **Step 3 (Forward Dive & Heel Lift)**: Sleek `Day 2 (v38)` style exterior circular arc arrow (`╭───▶`) above/ahead of the shoulder showing the `+28°` forward torso pitch + crisp upward arrow (`▲`) under the lifted heel.
     * **Step 4 (Vertical Drive & Flat Heel)**: Sleek vertical upward drive arrow (`▲`) above the shoulder + `95°` parallel hip-hinge sector + downward lock arrow (`▼`) pressing the heel onto the solid green floor bar.
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

COL_RED = (248, 46, 52)
COL_RED_LIGHT = (255, 172, 168)
COL_GREEN = (22, 224, 122)
COL_GREEN_LIGHT = (162, 255, 208)
COL_WHITE = (255, 255, 255)

FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_CACHE = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(FONT_BOLD_PATH, size)
    return _FONT_CACHE[size]


_GRID_X, _GRID_Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def apply_posture_and_crimson_boost(
    rgb: np.ndarray,
    kp: dict,
    step_idx: int,
    u: float,
) -> tuple[np.ndarray, dict]:
    """
    1. Applies true piecewise-linear bone-hinge posture contrast (`Step 1` inward knee valgus `>   <`,
       `Step 3` forward torso fold + heel lift).
    2. Enhances the crimson 3D anatomical highlight on the overloaded thighs/knees/lumbar in Steps 1 & 3
       so the Red (Incorrect) panels match the visual richness of the Green (Correct) panels.
    """
    kp_out = {k: (float(v[0]), float(v[1])) for k, v in kp.items()}

    if step_idx == 1:
        y_hip = 1015.0 + 25.0 * (1.0 - u)
        kp_out["hip_r"] = (392.0, y_hip)
        kp_out["hip_l"] = (688.0, y_hip)
    elif step_idx == 2:
        y_hip = 1025.0 + 25.0 * (1.0 - u)
        kp_out["hip_r"] = (375.0, y_hip)
        kp_out["hip_l"] = (705.0, y_hip)

    if u <= 0.02 or step_idx in (2, 4):
        return rgb, kp_out

    dx_map = np.zeros((H, W), dtype=np.float32)
    dy_map = np.zeros((H, W), dtype=np.float32)

    if step_idx == 1:
        inward_px = 152.0 * u
        down_px = 14.0 * u

        y_hip = kp_out["hip_r"][1]
        y_kn = 1205.0
        y_ank = kp["ank_r"][1]

        t_up = np.clip((_GRID_Y - (y_hip - 30.0)) / max(1.0, y_kn - (y_hip - 30.0)), 0.0, 1.0)
        t_dn = np.clip((y_ank - 15.0 - _GRID_Y) / max(1.0, (y_ank - 15.0) - y_kn), 0.0, 1.0)
        hinge_y = np.minimum(t_up, t_dn)
        hinge_y = 0.5 - 0.5 * np.cos(np.pi * hinge_y)

        w_x_r = np.exp(-((_GRID_X - 355.0) ** 2) / (2.0 * 148.0 ** 2)) * np.clip((535.0 - _GRID_X) / 45.0, 0.0, 1.0)
        w_x_l = np.exp(-((_GRID_X - 735.0) ** 2) / (2.0 * 148.0 ** 2)) * np.clip((_GRID_X - 545.0) / 45.0, 0.0, 1.0)

        w_r = hinge_y * w_x_r
        w_l = hinge_y * w_x_l

        dx_map += inward_px * w_r - inward_px * w_l
        dy_map += down_px * (w_r + w_l)

        kp_out["kn_r"] = (kp["kn_r"][0] + inward_px * 0.92, kp["kn_r"][1] + down_px * 0.85)
        kp_out["kn_l"] = (kp["kn_l"][0] - inward_px * 0.92, kp["kn_l"][1] + down_px * 0.85)

    elif step_idx == 3:
        sh = kp["sh"]
        hip_r = kp["hip_r"]
        kn_r = kp["kn_r"]
        ank_r = kp["ank_r"]

        lean_dx = 98.0 * u
        lean_dy = 70.0 * u
        heel_lift_dy = -64.0 * u
        knee_fwd_dx = 42.0 * u

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

        kp_out["sh"] = (sh[0] + lean_dx * 0.82, sh[1] + lean_dy * 0.82)
        kp_out["kn_r"] = (kn_r[0] + knee_fwd_dx * 0.85, kn_r[1])
        kp_out["ank_r"] = (ank_r[0], ank_r[1] + heel_lift_dy * 0.80)

    map_x = (_GRID_X - dx_map).astype(np.float32)
    map_y = (_GRID_Y - dy_map).astype(np.float32)
    warped = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Boost intrinsic 3D crimson muscle/joint surface highlight on the mannequin skin in Steps 1 & 3
    # so Red panels have the same rich anatomical area highlight as Green panels
    out = warped.astype(np.float32)
    luma = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    body_mask = np.clip((luma - 28.0) / 55.0, 0.0, 1.0)

    if step_idx == 1:
        kn_r = kp_out["kn_r"]
        kn_l = kp_out["kn_l"]
        g_r = np.exp(-((_GRID_X - (kn_r[0] + 10.0)) ** 2) / (2.0 * 68.0 ** 2) - ((_GRID_Y - (kn_r[1] - 35.0)) ** 2) / (2.0 * 145.0 ** 2))
        g_l = np.exp(-((_GRID_X - (kn_l[0] - 10.0)) ** 2) / (2.0 * 68.0 ** 2) - ((_GRID_Y - (kn_l[1] - 35.0)) ** 2) / (2.0 * 145.0 ** 2))
        glow = np.clip((g_r + g_l) * body_mask * u * 0.58, 0.0, 0.65)[:, :, None]
        target_red = np.array([245.0, 38.0, 42.0], dtype=np.float32)[None, None, :]
        out = out * (1.0 - glow) + (target_red * (0.45 + 0.55 * (luma[:, :, None] / 255.0))) * glow
    elif step_idx == 3:
        sh = kp_out["sh"]
        hip_r = kp_out["hip_r"]
        kn_r = kp_out["kn_r"]
        mid_back = ((sh[0] + hip_r[0]) * 0.5 - 25.0, (sh[1] + hip_r[1]) * 0.5)
        g_back = np.exp(-((_GRID_X - mid_back[0]) ** 2) / (2.0 * 85.0 ** 2) - ((_GRID_Y - mid_back[1]) ** 2) / (2.0 * 135.0 ** 2))
        g_knee = np.exp(-((_GRID_X - kn_r[0]) ** 2) / (2.0 * 65.0 ** 2) - ((_GRID_Y - kn_r[1]) ** 2) / (2.0 * 95.0 ** 2))
        glow = np.clip((g_back + g_knee * 0.85) * body_mask * u * 0.52, 0.0, 0.60)[:, :, None]
        target_red = np.array([245.0, 38.0, 42.0], dtype=np.float32)[None, None, :]
        out = out * (1.0 - glow) + (target_red * (0.45 + 0.55 * (luma[:, :, None] / 255.0))) * glow

    return np.clip(out, 0, 255).astype(np.uint8), kp_out


def draw_bone_axis_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    trim_ends_1x: float = 10.5,
):
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

    rd.line([pa, pb], fill=(0, 0, 0, 215), width=int(round(6.8 * ss)))
    rd.line([pa, pb], fill=(*col_main, 252), width=int(round(4.0 * ss)))
    rd.line([pa, pb], fill=(*col_light, 215), width=int(round(1.5 * ss)))


def draw_joint_angle_sector_ss(
    rd: ImageDraw.ImageDraw,
    vertex_ss: tuple[float, float],
    ray1_pt_ss: tuple[float, float],
    ray2_pt_ss: tuple[float, float],
    radius_1x: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    vx, vy = vertex_ss
    r = radius_1x * ss
    a1 = math.atan2(ray1_pt_ss[1] - vy, ray1_pt_ss[0] - vx)
    a2 = math.atan2(ray2_pt_ss[1] - vy, ray2_pt_ss[0] - vx)
    diff = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(diff) < 0.04:
        return

    steps = 32
    poly = [(vx, vy)]
    arc_pts = []
    for i in range(steps + 1):
        a = a1 + diff * (i / float(steps))
        px = vx + math.cos(a) * r
        py = vy + math.sin(a) * r
        poly.append((px, py))
        arc_pts.append((px, py))

    rd.polygon(poly, fill=(*col_main, 62))
    rd.line(arc_pts, fill=(0, 0, 0, 210), width=int(round(5.4 * ss)))
    rd.line(arc_pts, fill=(*col_main, 250), width=int(round(3.2 * ss)))
    rd.line(arc_pts, fill=(*col_light, 210), width=int(round(1.2 * ss)))


def draw_clean_exterior_vector_arrow_ss(
    rd: ImageDraw.ImageDraw,
    p_start_ss: tuple[float, float],
    p_end_ss: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    """
    Draws a sleek, unmistakable directional vector arrow (`───▶`) in clean unobstructed space:
    - Tapered continuous one-piece polygon shaft (`4.5px -> 9.0px`) + crisp single swept arrowhead (`24px` long, `24px` wide)
    - Framed with a dark studio drop-shadow and bright specular center spine so it pops at a glance!
    """
    x0, y0 = p_start_ss
    x1, y1 = p_end_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 20.0 * ss:
        return
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    head_len = 24.0 * ss
    head_half_w = 13.5 * ss
    neck_x = x1 - ux * head_len
    neck_y = y1 - uy * head_len

    w_tail = 2.4 * ss
    w_neck = 4.6 * ss

    for pad, rgba in [(2.6 * ss, (0, 0, 0, 225)), (0.0, (*col_main, 255))]:
        wt = w_tail + pad
        wn = w_neck + pad
        hw = head_half_w + pad * 1.15
        p_tl = (x0 + nx * wt, y0 + ny * wt)
        p_tr = (x0 - nx * wt, y0 - ny * wt)
        p_nl = (neck_x + nx * wn, neck_y + ny * wn)
        p_nr = (neck_x - nx * wn, neck_y - ny * wn)
        p_wl = (neck_x - ux * (3.0 * ss) + nx * hw, neck_y - uy * (3.0 * ss) + ny * hw)
        p_wr = (neck_x - ux * (3.0 * ss) - nx * hw, neck_y - uy * (3.0 * ss) - ny * hw)
        p_tip = (x1 + ux * pad * 1.3, y1 + uy * pad * 1.3)
        rd.polygon([p_tl, p_nl, p_wl, p_tip, p_wr, p_nr, p_tr], fill=rgba)

    # Specular center spine + origin anchor dot so the start of the vector is anchored
    rd.line([(x0 + ux * 4 * ss, y0 + uy * 4 * ss), (neck_x, neck_y)], fill=(*col_light, 230), width=int(round(2.0 * ss)))
    r_o = 4.2 * ss
    rd.ellipse([x0 - r_o - 1.5 * ss, y0 - r_o - 1.5 * ss, x0 + r_o + 1.5 * ss, y0 + r_o + 1.5 * ss], fill=(0, 0, 0, 200))
    rd.ellipse([x0 - r_o, y0 - r_o, x0 + r_o, y0 + r_o], fill=(*col_light, 255))


def draw_clean_exterior_arc_arrow_ss(
    rd: ImageDraw.ImageDraw,
    center_ss: tuple[float, float],
    radius_ss: float,
    deg_start: float,
    deg_end: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    """
    Draws a sleek one-piece curved trajectory arrow (`╭───▶`) in unobstructed exterior space
    (matching the approved `Day 1 v36` and `Day 2 v38` exterior arc style).
    """
    cx, cy = center_ss
    total_deg = deg_end - deg_start
    if abs(total_deg) < 5.0:
        return
    direction = 1.0 if total_deg > 0 else -1.0

    head_len = 24.0 * ss
    head_half_w = 13.5 * ss
    d_deg_head = math.degrees(head_len / max(40.0 * ss, radius_ss)) * direction
    deg_neck = deg_end - d_deg_head

    steps = 48
    for pad, rgba in [(2.6 * ss, (0, 0, 0, 225)), (0.0, (*col_main, 255))]:
        left_pts = []
        right_pts = []
        for i in range(steps + 1):
            u = i / float(steps)
            ang = math.radians(deg_start + (deg_neck - deg_start) * u)
            rx = math.cos(ang)
            ry = math.sin(ang)
            hw = (2.4 + 2.2 * u) * ss + pad
            px = cx + rx * radius_ss
            py = cy + ry * radius_ss
            left_pts.append((px + rx * hw, py + ry * hw))
            right_pts.append((px - rx * hw, py - ry * hw))

        ang_n = math.radians(deg_neck)
        rx_n, ry_n = math.cos(ang_n), math.sin(ang_n)
        tx_n, ty_n = -math.sin(ang_n) * direction, math.cos(ang_n) * direction
        nx_p, ny_p = cx + rx_n * radius_ss, cy + ry_n * radius_ss
        hw_head = head_half_w + pad * 1.15
        p_wl = (nx_p - tx_n * (3.0 * ss) + rx_n * hw_head, ny_p - ty_n * (3.0 * ss) + ry_n * hw_head)
        p_wr = (nx_p - tx_n * (3.0 * ss) - rx_n * hw_head, ny_p - ty_n * (3.0 * ss) - ry_n * hw_head)

        ang_t = math.radians(deg_end)
        tx_t, ty_t = -math.sin(ang_t) * direction, math.cos(ang_t) * direction
        tip_x = cx + math.cos(ang_t) * radius_ss + tx_t * pad * 1.3
        tip_y = cy + math.sin(ang_t) * radius_ss + ty_t * pad * 1.3

        poly = left_pts + [p_wl, (tip_x, tip_y), p_wr] + right_pts[::-1]
        rd.polygon(poly, fill=rgba)

    spine_pts = []
    for i in range(steps + 1):
        u = i / float(steps)
        ang = math.radians(deg_start + (deg_neck - deg_start) * u)
        spine_pts.append((cx + math.cos(ang) * radius_ss, cy + math.sin(ang) * radius_ss))
    rd.line(spine_pts, fill=(*col_light, 230), width=int(round(2.0 * ss)))


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
    rd.ellipse([cx - r - 2 * ss, cy - r - 2 * ss, cx + r + 2 * ss, cy + r + 2 * ss], fill=(0, 0, 0, 195))
    rd.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=(*col_main, 255),
        width=int(round((2.8 if is_primary else 2.2) * ss)),
    )
    r_dot = (3.5 if is_primary else 2.6) * ss
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


def render_v53_callouts(
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
        # 1. Single Clean 3-Point Bone Hinge (`Hip ◎ — Knee ◎ — Ankle ◎` bent inward `>   <`)
        #    + `18°` Knee Valgus Angle Sector inside the bent knee angle
        draw_joint_angle_sector_ss(rd, S(kn_r), S(hip_r), S(ank_r), 54.0, col_main, col_light, SS)
        draw_joint_angle_sector_ss(rd, S(kn_l), S(hip_l), S(ank_l), 54.0, col_main, col_light, SS)

        draw_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(hip_l), S(kn_l), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_l), S(ank_l), col_main, col_light, SS)

        # 2. Sleek Unobstructed Inward Collapse Arrows (`───▶ ◎     ◎ ◀───`) in the open dark studio space
        #    OUTSIDE each knee, pointing straight INWARD into the collapsing `Knee ◎`!
        draw_clean_exterior_vector_arrow_ss(
            rd,
            p_start_ss=S((kn_r[0] - 148.0, kn_r[1])),
            p_end_ss=S((kn_r[0] - 20.0, kn_r[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_clean_exterior_vector_arrow_ss(
            rd,
            p_start_ss=S((kn_l[0] + 148.0, kn_l[1])),
            p_end_ss=S((kn_l[0] + 20.0, kn_l[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.0, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (255, 995), "18°", "VALGUS CAVE", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "84%", "ACL & MCL LOAD", align="left")

    elif step_idx == 2:
        # =========================================================================
        # STEP 2 (FRONT VIEW — FIX: `0° KNEES TRACK TOES` / WIDE KNEE LOCK `<   >`)
        # =========================================================================
        # 1. Single Clean 3-Point Bone Hinge (`Hip ◎ — Knee ◎ — Ankle ◎` wide over toes `<   >`)
        #    + `0°` Neutral Alignment Sector at the knee
        draw_joint_angle_sector_ss(rd, S(kn_r), S(hip_r), S(ank_r), 52.0, col_main, col_light, SS)
        draw_joint_angle_sector_ss(rd, S(kn_l), S(hip_l), S(ank_l), 52.0, col_main, col_light, SS)

        draw_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(hip_l), S(kn_l), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_l), S(ank_l), col_main, col_light, SS)

        # 2. Sleek Unobstructed Outward Drive Arrows (`◎ ◀───     ───▶ ◎`) in the wide open dark space
        #    BETWEEN the knees, pointing straight OUTWARD into the locked `Knee ◎`!
        draw_clean_exterior_vector_arrow_ss(
            rd,
            p_start_ss=S((kn_r[0] + 152.0, kn_r[1])),
            p_end_ss=S((kn_r[0] + 22.0, kn_r[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_clean_exterior_vector_arrow_ss(
            rd,
            p_start_ss=S((kn_l[0] - 152.0, kn_l[1])),
            p_end_ss=S((kn_l[0] - 22.0, kn_l[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.0, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (245, 995), "0°", "KNEES TRACK TOES", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "96%", "VMO & GLUTE LOCK", align="left")

    elif step_idx == 3:
        # =========================================================================
        # STEP 3 (3/4 SIDE VIEW — WRONG: `+28° HEELS LIFTING` & FORWARD TORSO FOLD)
        # =========================================================================
        # 1. Single Clean 3-Point Chain (`Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`) + Hip Fold Sector
        draw_joint_angle_sector_ss(rd, S(hip_r), S(sh), S(kn_r), 64.0, col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        # 2. Upper Exterior Forward-Lean Arc Arrow (`╭───▶`) in the dark studio space behind/above the back
        #    (sweeping from upright `-112°` forward/downward to `-62°`, 100% outside the body!)
        draw_clean_exterior_arc_arrow_ss(
            rd,
            center_ss=S((hip_r[0] + 15.0, hip_r[1] - 60.0)),
            radius_ss=340.0 * SS,
            deg_start=-118.0,
            deg_end=-68.0,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        # 3. Lower Exterior Upward Heel-Lift Arrow (`▲`) in the open space behind the lifted heel
        heel_arrow_x = ank_r[0] - 58.0
        floor_y = ank_r[1] + 74.0
        rd.line(
            [S((ank_r[0] - 95.0, floor_y)), S((ank_r[0] + 95.0, floor_y))],
            fill=(225, 235, 248, 175),
            width=int(round(3.0 * SS)),
        )
        draw_clean_exterior_vector_arrow_ss(
            rd,
            p_start_ss=S((heel_arrow_x, floor_y - 4.0)),
            p_end_ss=S((heel_arrow_x, ank_r[1] - 42.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.5, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (325, 1020), "+28°", "HEELS LIFTING", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "79%", "LUMBAR & PATELLA", align="left")

    else:
        # =========================================================================
        # STEP 4 (3/4 SIDE VIEW — FIX: `95° PARALLEL DEPTH` & FLAT HEEL DRIVE)
        # =========================================================================
        # 1. Single Clean 3-Point Chain (`Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`) + `95°` Parallel Sector
        draw_joint_angle_sector_ss(rd, S(hip_r), S(sh), S(kn_r), 68.0, col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        # 2. Upper Exterior Vertical Chest-Drive Arrow (`▲`) rising straight up above the shoulder
        draw_clean_exterior_vector_arrow_ss(
            rd,
            p_start_ss=S((sh[0], sh[1] - 26.0)),
            p_end_ss=S((sh[0], sh[1] - 158.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        # 3. Lower Exterior Downward Heel-Lock Arrow (`▼`) pressing straight down onto the solid green floor bar
        heel_arrow_x = ank_r[0] - 58.0
        floor_y = ank_r[1] + 28.0
        draw_bone_axis_ss(
            rd,
            S((ank_r[0] - 90.0, floor_y)),
            S((ank_r[0] + 95.0, floor_y)),
            col_main,
            col_light,
            SS,
            trim_ends_1x=0.0,
        )
        draw_clean_exterior_vector_arrow_ss(
            rd,
            p_start_ss=S((heel_arrow_x, floor_y - 122.0)),
            p_end_ss=S((heel_arrow_x, floor_y - 8.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
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

        rgb_frame, kp_cur = apply_posture_and_crimson_boost(raw_rgb, raw_kp, step_idx=s_idx, u=1.0)
        canvas = Image.fromarray(rgb_frame).convert("RGBA")

        render_v53_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)

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
    sheet_path = os.path.join(OUT_DIR, "day3_v53_difference_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    crop_2x2 = np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])])
    crop_path = os.path.join(OUT_DIR, "day3_v53_closeup_sheet.jpg")
    cv2.imwrite(crop_path, crop_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved:", sheet_path, "and", crop_path)


if __name__ == "__main__":
    main()
