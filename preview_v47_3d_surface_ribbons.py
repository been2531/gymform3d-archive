#!/usr/bin/env python3
"""
Day 3 v47 — Continuous 3D Shader Ribbon Arrows & Purposeful 3D Biomechanical Markers (`시안`)
=============================================================================================
Directly resolves 중현 님's feedback (`t215u`):
1. "각도 퍼센티지 밑에 밑줄은 필요없고":
   - 0% underline bars under degree (`°`) or percentage (`%`) numbers.
   - Clean 58pt Extrabold Pure White number + 22pt Extrabold Pure White sub-label directly below.
2. "화살표시나 이런게 입체적으로 따라가야하는데 아직은 어설픈것 같아. 마커점도 의미가 있어야 하는데":
   - Every single marker dot on the body now has a clear, unmistakable 1:1 biomechanical role
     (NO orphan decorative dots, and NO messy stick-lines cluttering the legs):
     (A) **3D Goniometer Hinge Pivot Marker (`각도 회전축 피벗 마커`)**:
         Sits at the exact vertex of the measured joint angle (`18°`, `0°`, `+28°`, `95°`), featuring a
         3D perspective-tilted goniometer ring, a 3D lit metallic-core pivot sphere, two short precision
         axis rays (`Reference Axis` vs `Actual Bone Axis`), and a translucent 3D perspective angle sector,
         linked directly to the degree badge.
     (B) **3D Surface-Normal Load Sensor Marker (`부하·근활성 타겟 센서 마커`)**:
         Sits flush on the 3D surface plane of the target ligament/muscle (`ACL/MCL`, `VMO`, `Lumbar/Patella`,
         `Quads/Glutes`) as a perspective-foreshortened elliptical sensor ring + core node, linked directly
         to the `%` badge.
   - **True 3D Perspective Surface-Wrapping Ribbon Arrows (`3D 입체 곡면 리본 화살표`)**:
     Rendered with a continuous 4x-supersampled 3D ribbon shader:
     - Wraps around the 3D cylindrical contour of the thigh/knee/torso/heel in perspective,
     - Casts a realistic soft 3D elevation shadow onto the anatomical surface underneath so the ribbon
       floats in real 3D space above the limb,
     - Features a dark 3D extruded thickness wall (`3D 두께 옆면`), cylindrical normal-mapped studio shading
       across the ribbon surface, a razor-sharp white specular bevel along the leading edge, and a 3D
       two-facet sculpted delta arrowhead.
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
import build_day3_v39_squat as d3

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_PKL = "/workspace/daily_info_reels/_cache_day3_v44_pools.pkl"
W, H = 1080, 1920

COL_RED = (242, 42, 50)
COL_RED_DARK = (98, 8, 14)
COL_RED_SOFT = (255, 162, 156)

COL_GREEN = (16, 218, 116)
COL_GREEN_DARK = (5, 78, 40)
COL_GREEN_SOFT = (165, 255, 208)

F_V47_NUM = v30.f_barlow(58, extrabold=True)
F_V47_SUB = v30.f_pretendard(22, "extrabold")


def draw_v47_frameless_white_badge(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    num_str: str,
    tag_str: str,
    align: str = "right",
):
    """
    Frameless Pure White Number (`58pt`) + Enlarged Pure White Sub-Label (`22pt`)
    with ZERO underline bar ("각도 퍼센티지 밑에 밑줄은 필요없고").
    """
    x, y = xy
    nb = od.textbbox((0, 0), num_str, font=F_V47_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), tag_str, font=F_V47_SUB)
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

    for dx_s, dy_s in [(0, 3), (0, -2), (3, 0), (-3, 0), (2, 2)]:
        sd.text((nx + dx_s, y + dy_s), num_str, font=F_V47_NUM, fill=(0, 0, 0, 240))
        sd.text((tx + dx_s, y + 55 + dy_s), tag_str, font=F_V47_SUB, fill=(0, 0, 0, 245))

    od.text((nx, y), num_str, font=F_V47_NUM, fill=(255, 255, 255, 255))
    od.text((tx, y + 55), tag_str, font=F_V47_SUB, fill=(255, 255, 255, 255))


def draw_3d_elliptical_surface_ring(
    rd: ImageDraw.ImageDraw,
    center: tuple[float, float],
    rx: float,
    ry: float,
    tilt_deg: float,
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 4,
    fill_alpha: int = 32,
    extrude_px: float = 2.6,
):
    """
    Draws a 3D perspective elliptical ring lying on an anatomical or floor plane (`rx`, `ry`, `tilt_deg`),
    with 3D depth modulation (receding back arc vs thick extruded front rim).
    """
    cx, cy = center[0] * ss, center[1] * ss
    rx_s, ry_s = rx * ss, ry * ss
    tilt = math.radians(tilt_deg)
    ct, st = math.cos(tilt), math.sin(tilt)

    n_pts = 140
    angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=True)
    pts = []
    for a in angles:
        lx = rx_s * math.cos(a)
        ly = ry_s * math.sin(a)
        px = cx + lx * ct - ly * st
        py = cy + lx * st + ly * ct
        # Front-facing factor (`sin(a)` in -1..+1)
        pts.append((px, py, math.sin(a)))

    if fill_alpha > 0:
        rd.polygon([(p[0], p[1]) for p in pts], fill=(*col_main, fill_alpha))

    ex_dy = extrude_px * ss
    for i in range(n_pts - 1):
        x0, y0, d0 = pts[i]
        x1, y1, d1 = pts[i + 1]
        front = 0.5 * (d0 + d1)  # -1 (back) .. +1 (front)
        w_seg = int(round((1.3 + 2.2 * max(0.0, front)) * ss))
        alpha_seg = int(round(105 + 150 * (0.5 + 0.5 * front)))

        if front > -0.1:
            # 3D extruded dark wall under front-facing arc
            rd.line([(x0, y0 + ex_dy), (x1, y1 + ex_dy)], fill=(0, 0, 0, int(alpha_seg * 0.82)), width=w_seg + ss)
            rd.line([(x0, y0 + ex_dy * 0.65), (x1, y1 + ex_dy * 0.65)], fill=(*col_dark, alpha_seg), width=w_seg)
        rd.line([(x0, y0), (x1, y1)], fill=(*col_main, alpha_seg), width=w_seg)
        if front > 0.2:
            rd.line([(x0, y0 - 0.5 * ss), (x1, y1 - 0.5 * ss)], fill=(255, 255, 255, int(215 * front)), width=max(1, int(0.9 * ss)))


def draw_3d_goniometer_joint_hub(
    rd: ImageDraw.ImageDraw,
    pivot: tuple[float, float],
    deg_ref: float,
    deg_actual: float,
    ray_len_ref: float,
    ray_len_actual: float,
    wedge_rx: float,
    wedge_ry: float,
    ring_tilt_deg: float,
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 4,
):
    """
    Renders a **Meaningful 3D Goniometer Joint Hub (`의미 있는 3D 각도기 관절 축 마커`)**:
    1. A 3D perspective angle sector wedge (`wedge_rx x wedge_ry`) between `deg_ref` (ideal reference axis)
       and `deg_actual` (actual limb axis), with a 3D extruded outer rim.
    2. A crisp 3D Reference Axis Ray (`deg_ref`, dashed white/silver architectural ray) and a 3D Actual Limb
       Axis Ray (`deg_actual`, 3D cylindrical lit rod along the bone direction) meeting right at `pivot`.
    3. A 3D Perspective Orbital Ring + 3D Lit Hinge Sphere right at the vertex `pivot`, so the marker dot has
       an unmistakable mechanical purpose: it is the exact rotational hinge of the angle gauge!
    """
    cx, cy = pivot[0] * ss, pivot[1] * ss
    rx, ry = wedge_rx * ss, wedge_ry * ss

    # 1. 3D Perspective Sector Wedge + Extruded Outer Arc
    n_arc = 56
    a_vals = np.linspace(math.radians(deg_ref), math.radians(deg_actual), n_arc)
    outer_pts = [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in a_vals]
    inner_pts = [(cx + rx * 0.20 * math.cos(a), cy + ry * 0.20 * math.sin(a)) for a in reversed(a_vals)]

    rd.polygon(outer_pts + inner_pts, fill=(*col_main, 64))

    ex_dy = 3.0 * ss
    for i in range(len(outer_pts) - 1):
        p0, p1 = outer_pts[i], outer_pts[i + 1]
        rd.line([(p0[0], p0[1] + ex_dy), (p1[0], p1[1] + ex_dy)], fill=(0, 0, 0, 210), width=int(4.8 * ss))
        rd.line([(p0[0], p0[1] + ex_dy * 0.6), (p1[0], p1[1] + ex_dy * 0.6)], fill=(*col_dark, 245), width=int(4.2 * ss))

    for i in range(len(outer_pts) - 1):
        p0, p1 = outer_pts[i], outer_pts[i + 1]
        u = i / float(max(1, len(outer_pts) - 2))
        w_a = int(round((2.6 + 2.0 * u) * ss))
        rd.line([p0, p1], fill=(*col_main, 255), width=w_a)
        rd.line([(p0[0], p0[1] - 0.7 * ss), (p1[0], p1[1] - 0.7 * ss)], fill=(255, 255, 255, int(165 + 85 * u)), width=max(1, int(1.1 * ss)))

    # 2. Reference Axis Ray (`deg_ref`) — sleek architectural dashed reference ray
    rad_ref = math.radians(deg_ref)
    len_r = ray_len_ref * ss
    n_dash = 6
    for i in range(1, n_dash):
        t0 = (i + 0.05) / n_dash
        t1 = (i + 0.65) / n_dash
        p0 = (cx + len_r * t0 * math.cos(rad_ref), cy + len_r * t0 * math.sin(rad_ref))
        p1 = (cx + len_r * t1 * math.cos(rad_ref), cy + len_r * t1 * math.sin(rad_ref))
        rd.line([p0, p1], fill=(0, 0, 0, 195), width=int(3.6 * ss))
        rd.line([p0, p1], fill=(240, 246, 255, 225), width=int(1.8 * ss))

    # 3. Actual Bone Axis Ray (`deg_actual`) — 3D cylindrical lit axis rod
    rad_act = math.radians(deg_actual)
    len_a = ray_len_actual * ss
    pa0 = (cx + 10.0 * ss * math.cos(rad_act), cy + 10.0 * ss * math.sin(rad_act))
    pa1 = (cx + len_a * math.cos(rad_act), cy + len_a * math.sin(rad_act))
    rd.line([(pa0[0], pa0[1] + 2.2 * ss), (pa1[0], pa1[1] + 2.2 * ss)], fill=(0, 0, 0, 215), width=int(5.8 * ss))
    rd.line([pa0, pa1], fill=(*col_main, 250), width=int(3.8 * ss))
    rd.line([pa0, pa1], fill=(255, 255, 255, 230), width=int(1.4 * ss))
    # Small 3D end-cap bead at `pa1`
    r_cap = 3.8 * ss
    rd.ellipse([pa1[0] - r_cap, pa1[1] - r_cap, pa1[0] + r_cap, pa1[1] + r_cap], fill=(255, 255, 255, 255), outline=(*col_main, 255), width=ss)

    # 4. 3D Perspective Orbital Ring + 3D Shaded Hinge Sphere at `pivot`
    draw_3d_elliptical_surface_ring(
        rd, pivot, rx=22.0, ry=11.0, tilt_deg=ring_tilt_deg,
        col_main=col_main, col_dark=col_dark, col_soft=col_soft,
        ss=ss, fill_alpha=24, extrude_px=2.2
    )
    r_sp = 8.8 * ss
    rd.ellipse([cx - r_sp * 1.15, cy - r_sp * 0.9 + 2.8 * ss, cx + r_sp * 1.15, cy + r_sp * 1.25 + 2.8 * ss], fill=(0, 0, 0, 205))
    rd.ellipse([cx - r_sp, cy - r_sp, cx + r_sp, cy + r_sp], fill=(*col_dark, 255), outline=(255, 255, 255, 235), width=int(1.2 * ss))
    r_m = r_sp * 0.72
    rd.ellipse([cx - r_m - 0.12 * r_sp, cy - r_m - 0.12 * r_sp, cx + r_m - 0.12 * r_sp, cy + r_m - 0.12 * r_sp], fill=(*col_main, 255))
    r_sp_hi = r_sp * 0.32
    rd.ellipse([cx - r_sp_hi - 0.25 * r_sp, cy - r_sp_hi - 0.25 * r_sp, cx + r_sp_hi - 0.25 * r_sp, cy + r_sp_hi - 0.25 * r_sp], fill=(255, 255, 255, 255))


def draw_3d_surface_load_sensor(
    rd: ImageDraw.ImageDraw,
    pt: tuple[float, float],
    tilt_deg: float,
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 4,
):
    """
    Renders a **3D Surface-Normal Load/Activation Sensor (`3D 표면 밀착 타겟 센서 마커`)**
    for the `%` callout (`84% ACL & MCL LOAD`, `96% VMO & GLUTE LOCK`, etc.):
    - Foreshortened 3D elliptical ring tilted to lie flush on the muscle/ligament surface,
    - Glowing 3D sensor core at the center from which the `%` leader line originates.
    """
    draw_3d_elliptical_surface_ring(
        rd, pt, rx=18.5, ry=9.5, tilt_deg=tilt_deg,
        col_main=col_main, col_dark=col_dark, col_soft=col_soft,
        ss=ss, fill_alpha=38, extrude_px=2.0
    )
    cx, cy = pt[0] * ss, pt[1] * ss
    r_c = 6.5 * ss
    rd.ellipse([cx - r_c, cy - r_c + 2.0 * ss, cx + r_c, cy + r_c + 2.0 * ss], fill=(0, 0, 0, 195))
    rd.ellipse([cx - r_c, cy - r_c, cx + r_c, cy + r_c], fill=(*col_main, 255), outline=(255, 255, 255, 240), width=int(1.1 * ss))
    r_w = 2.6 * ss
    rd.ellipse([cx - r_w - 0.2 * r_c, cy - r_w - 0.2 * r_c, cx + r_w - 0.2 * r_c, cy + r_w - 0.2 * r_c], fill=(255, 255, 255, 255))


def draw_3d_perspective_wrap_ribbon_arrow(
    rd: ImageDraw.ImageDraw,
    ctrl_pts: list[tuple[float, float]],
    w_tail: float,
    w_head: float,
    extrude_vec: tuple[float, float],
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    head_len: float = 34.0,
    head_wid: float = 30.0,
    ss: int = 4,
):
    """
    Renders a **Cinema-Grade 3D Perspective Surface-Wrapping Ribbon Arrow** at 4x supersampling:
    1. **3D Limb-Wrap Foreshortening**:
       - Starts at `u=0` as a thin dark-shaded edge emerging from around the side of the 3D limb contour,
       - Twists smoothly in 3D space across the front of the joint/muscle so its ribbon face opens up toward
         the camera (`w_tail -> w_head`).
    2. **3D Elevation Contact Shadow**:
       - Casts a soft dark shadow offset onto the limb underneath (`extrude_vec * 1.8`), giving the unmistakable
         optical illusion that the 3D arrow is hovering in 3D space right in front of the knee/thigh/torso!
    3. **3D Extruded Side Thickness Wall + Transverse Cylindrical Shading**:
       - Renders the physical 3D thickness wall (`extrude_vec`) in deep `col_dark`,
       - Renders 5 transverse lighting strips across the width of the ribbon (from shaded bottom bevel ->
         saturated mid-body -> bright `col_soft` upper face -> pure white specular rim along the top bevel edge).
    4. **Sculpted 3D Two-Facet Arrowhead**:
       - Split along the central 3D ridge line into a bright studio-lit upper facet and a rich lower facet,
         with a 3D extruded dark cliff underneath and a razor-sharp white bevel outline.
    """
    pts_np = np.array(ctrl_pts, dtype=np.float32) * ss
    n_samples = 140
    t_vals = np.linspace(0.0, 1.0, n_samples)

    if len(pts_np) == 3:
        p0, p1, p2 = pts_np
        curve = np.array([((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2 for t in t_vals])
    else:
        p0, p1, p2, p3 = pts_np[:4]
        curve = np.array([
            ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
            for t in t_vals
        ])

    diffs = np.diff(curve, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum_len[-1])
    head_len_s = head_len * ss
    shaft_max_len = max(total_len * 0.42, total_len - head_len_s * 0.76)

    shaft_pts = curve[cum_len <= shaft_max_len]
    if len(shaft_pts) < 8:
        shaft_pts = curve[:-16]

    n_s = len(shaft_pts)
    tangents = np.zeros_like(shaft_pts)
    tangents[1:-1] = shaft_pts[2:] - shaft_pts[:-2]
    tangents[0] = shaft_pts[1] - shaft_pts[0]
    tangents[-1] = shaft_pts[-1] - shaft_pts[-2]
    tnorms = np.hypot(tangents[:, 0], tangents[:, 1])[:, None] + 1e-6
    tangents = tangents / tnorms
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Ensure `normals` points toward the studio light (upward / outward from `extrude_vec`)
    ex_x, ex_y = extrude_vec[0] * ss, extrude_vec[1] * ss
    mean_dot = float(np.mean(normals[:, 0] * (-ex_x) + normals[:, 1] * (-ex_y)))
    if mean_dot < 0:
        normals = -normals

    w0_s, w1_s = w_tail * ss, w_head * ss
    half_widths = np.zeros(n_s, dtype=np.float32)
    for i in range(n_s):
        u = i / float(max(1, n_s - 1))
        # S-curve 3D perspective twist: narrow edge-on tail -> wide camera-facing ribbon body
        twist = math.sin(u * math.pi * 0.5) ** 0.80
        half_widths[i] = 0.5 * (w0_s + (w1_s - w0_s) * twist)

    top_edge = [(float(shaft_pts[i, 0] + normals[i, 0] * half_widths[i]), float(shaft_pts[i, 1] + normals[i, 1] * half_widths[i])) for i in range(n_s)]
    bot_edge = [(float(shaft_pts[i, 0] - normals[i, 0] * half_widths[i]), float(shaft_pts[i, 1] - normals[i, 1] * half_widths[i])) for i in range(n_s)]

    # Arrowhead geometry at tip
    tip_pt = curve[-1]
    base_pt = shaft_pts[-1]
    h_vec = tip_pt - base_pt
    h_norm = float(np.hypot(h_vec[0], h_vec[1])) + 1e-6
    ux, uy = float(h_vec[0] / h_norm), float(h_vec[1] / h_norm)
    # Align arrowhead normal with `normals[-1]` so `wing_top` is on the same lit side as `top_edge`
    nx, ny = float(normals[-1, 0]), float(normals[-1, 1])

    hw_s = 0.5 * head_wid * ss
    wing_top = (float(base_pt[0] - ux * (head_len_s * 0.20) + nx * hw_s), float(base_pt[1] - uy * (head_len_s * 0.20) + ny * hw_s))
    wing_bot = (float(base_pt[0] - ux * (head_len_s * 0.20) - nx * hw_s), float(base_pt[1] - uy * (head_len_s * 0.20) - ny * hw_s))
    notch_pt = (float(base_pt[0] + ux * (head_len_s * 0.12)), float(base_pt[1] + uy * (head_len_s * 0.12)))
    tip_xy = (float(tip_pt[0]), float(tip_pt[1]))

    # 1. Multi-layer 3D Elevation Drop Shadow on the limb underneath
    for sh_mult, sh_alpha in [(2.1, 85), (1.45, 145)]:
        sdx, sdy = ex_x * sh_mult, ex_y * sh_mult
        sh_poly = [(p[0] + sdx, p[1] + sdy) for p in top_edge] + [(p[0] + sdx, p[1] + sdy) for p in reversed(bot_edge)]
        rd.polygon(sh_poly, fill=(0, 0, 0, sh_alpha))
        sh_head = [
            (tip_xy[0] + sdx, tip_xy[1] + sdy),
            (wing_top[0] + sdx, wing_top[1] + sdy),
            (notch_pt[0] + sdx, notch_pt[1] + sdy),
            (wing_bot[0] + sdx, wing_bot[1] + sdy),
        ]
        rd.polygon(sh_head, fill=(0, 0, 0, min(210, int(sh_alpha * 1.25))))

    # 2. 3D Extruded Side/Bottom Thickness Wall (`bot_edge` extruded by `extrude_vec`)
    for i in range(n_s - 1):
        u = i / float(max(1, n_s - 1))
        a_wall = int(round(125 + 130 * (u ** 0.7)))
        b0, b1 = bot_edge[i], bot_edge[i + 1]
        wall_q = [b0, b1, (b1[0] + ex_x, b1[1] + ex_y), (b0[0] + ex_x, b0[1] + ex_y)]
        rd.polygon(wall_q, fill=(*col_dark, a_wall))
        rd.line([(b0[0] + ex_x, b0[1] + ex_y), (b1[0] + ex_x, b1[1] + ex_y)], fill=(0, 0, 0, int(a_wall * 0.9)), width=max(2, int(1.1 * ss)))

    # 3. Transverse 3D Cylindrical Shading across Ribbon Face (4 sub-bands from `bot_edge` to `top_edge`)
    bands = [
        (-1.0, -0.35, 0.25, 0.0),   # Bottom bevel band: darker rich shade
        (-0.35, 0.25, 0.85, 0.15),  # Mid-lower band: pure saturated main color
        (0.25, 0.75, 1.0, 0.48),    # Mid-upper band: lit soft highlight color
        (0.75, 1.0, 1.0, 0.78),     # Top bevel band: hot specular sheen
    ]
    for i in range(n_s - 1):
        u = i / float(max(1, n_s - 1))
        alpha_u = int(round(95 + 160 * (u ** 0.65)))
        # Longitudinal limb-wrap lighting (`u=0` emerges from shadow around the side of the limb -> `u=1` full studio light)
        long_light = 0.28 + 0.72 * (u ** 0.65)

        for v0, v1, main_w, soft_w in bands:
            # Interpolate between `shaft_pts` + `normals * half_width * v`
            def _pt_at(idx, v_frac):
                return (
                    float(shaft_pts[idx, 0] + normals[idx, 0] * half_widths[idx] * v_frac),
                    float(shaft_pts[idx, 1] + normals[idx, 1] * half_widths[idx] * v_frac),
                )

            q = [_pt_at(i, v0), _pt_at(i + 1, v0), _pt_at(i + 1, v1), _pt_at(i, v1)]
            # Color blend
            base_r = col_dark[0] * (1.0 - main_w * long_light) + col_main[0] * (main_w * long_light)
            base_g = col_dark[1] * (1.0 - main_w * long_light) + col_main[1] * (main_w * long_light)
            base_b = col_dark[2] * (1.0 - main_w * long_light) + col_main[2] * (main_w * long_light)

            sw = soft_w * (u ** 0.85)
            cr = int(np.clip(base_r * (1.0 - sw) + (col_soft[0] * 0.55 + 255 * 0.45) * sw, 0, 255))
            cg = int(np.clip(base_g * (1.0 - sw) + (col_soft[1] * 0.55 + 255 * 0.45) * sw, 0, 255))
            cb = int(np.clip(base_b * (1.0 - sw) + (col_soft[2] * 0.55 + 255 * 0.45) * sw, 0, 255))
            rd.polygon(q, fill=(cr, cg, cb, alpha_u))

    # Crisp white specular highlight along `top_edge`
    for i in range(int(n_s * 0.12), n_s - 1):
        u = i / float(max(1, n_s - 1))
        rd.line([top_edge[i], top_edge[i + 1]], fill=(255, 255, 255, int(235 * (u ** 0.9))), width=max(2, int(1.2 * ss)))

    # 4. 3D Extruded Cliff under Arrowhead + Two-Facet Sculpted Delta Head
    for w_pt in (wing_bot, wing_top):
        cliff = [tip_xy, w_pt, (w_pt[0] + ex_x, w_pt[1] + ex_y), (tip_xy[0] + ex_x, tip_xy[1] + ex_y)]
        rd.polygon(cliff, fill=(*col_dark, 255))
    cliff_back = [wing_bot, notch_pt, (notch_pt[0] + ex_x, notch_pt[1] + ex_y), (wing_bot[0] + ex_x, wing_bot[1] + ex_y)]
    rd.polygon(cliff_back, fill=(*col_dark, 255))

    # Upper lit facet (`tip_xy - wing_top - notch_pt`)
    lit_col = (
        int(col_soft[0] * 0.58 + 255 * 0.42),
        int(col_soft[1] * 0.58 + 255 * 0.42),
        int(col_soft[2] * 0.58 + 255 * 0.42),
        255,
    )
    # Lower shaded facet (`tip_xy - notch_pt - wing_bot`)
    shade_col = (*col_main, 255)

    rd.polygon([tip_xy, wing_top, notch_pt], fill=lit_col)
    rd.polygon([tip_xy, notch_pt, wing_bot], fill=shade_col)

    # Crisp 3D bevel edges on arrowhead
    rd.line([wing_top, tip_xy], fill=(255, 255, 255, 250), width=max(2, int(1.4 * ss)))
    rd.line([notch_pt, tip_xy], fill=(255, 255, 255, 245), width=max(2, int(1.3 * ss)))
    rd.line([wing_bot, tip_xy], fill=(*col_soft, 230), width=max(1, int(1.0 * ss)))


def render_v47_scene(
    rgb_frame: np.ndarray,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 0.85,
) -> Image.Image:
    """
    Renders the v47 3D Perspective Surface-Wrapping Ribbon Arrows & Meaningful 2-Point Joint/Sensor Markers.
    """
    SS = 4
    rig = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig)

    col_main = COL_RED if is_wrong else COL_GREEN
    col_dark = COL_RED_DARK if is_wrong else COL_GREEN_DARK
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    sh = kp["sh"]
    hip_r = kp["hip_r"]
    hip_l = kp["hip_l"]
    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    ank_r = kp["ank_r"]
    ank_l = kp["ank_l"]

    if step_idx == 1:
        # =========================================================================================
        # STEP 1 (Front View — WRONG: `18° VALGUS CAVE` & `84% ACL & MCL LOAD`)
        # - Marker 1 (Right Knee Goniometer Hub): Shows the vertical reference plumb axis (90° down)
        #   vs the inward-buckling tibia/femur valgus angle (`18° VALGUS CAVE`).
        # - Marker 2 (Left Knee Medial Ligament Sensor): Sits flush on the medial ACL/MCL joint line
        #   where `84% ACL & MCL LOAD` concentrates.
        # - 3D Cylindrical Wrap Ribbon Arrows: Wrap from the outer thigh contour around the front of
        #   both knees inward toward the center (`> <`).
        # =========================================================================================
        draw_3d_goniometer_joint_hub(
            rd, pivot=kn_r,
            deg_ref=90.0, deg_actual=128.0,
            ray_len_ref=92.0, ray_len_actual=92.0,
            wedge_rx=74.0, wedge_ry=62.0,
            ring_tilt_deg=-14.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        draw_3d_surface_load_sensor(
            rd, pt=kn_l, tilt_deg=14.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        shift = 8.0 * rep_amount
        # Right thigh/knee 3D wrap-around inward buckle ribbon arrow
        draw_3d_perspective_wrap_ribbon_arrow(
            rd,
            ctrl_pts=[
                (kn_r[0] - 96.0, kn_r[1] - 42.0),  # emerges from outer contour of right thigh
                (kn_r[0] - 82.0, kn_r[1] + 20.0),  # wraps around the front-outer curve of the knee
                (kn_r[0] - 26.0, kn_r[1] + 32.0),
                (kn_r[0] + 38.0 + shift, kn_r[1] + 16.0),  # drives inward
            ],
            w_tail=3.5, w_head=18.5,
            extrude_vec=(1.5, 4.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_len=35.0, head_wid=31.0, ss=SS
        )
        # Left thigh/knee 3D wrap-around inward buckle ribbon arrow
        draw_3d_perspective_wrap_ribbon_arrow(
            rd,
            ctrl_pts=[
                (kn_l[0] + 96.0, kn_l[1] - 42.0),  # emerges from outer contour of left thigh
                (kn_l[0] + 82.0, kn_l[1] + 20.0),  # wraps around front-outer curve of left knee
                (kn_l[0] + 26.0, kn_l[1] + 32.0),
                (kn_l[0] - 38.0 - shift, kn_l[1] + 16.0),  # drives inward
            ],
            w_tail=3.5, w_head=18.5,
            extrude_vec=(-1.5, 4.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_len=35.0, head_wid=31.0, ss=SS
        )

    elif step_idx == 2:
        # =========================================================================================
        # STEP 2 (Front View — FIX: `0° KNEES TRACK TOES` & `96% VMO & GLUTE LOCK`)
        # - Marker 1 (Right Knee Goniometer Hub): Shows locked outward tracking over the ankle/toes (`0°`).
        # - Marker 2 (Left VMO/Quad Sensor): Sits flush on the left VMO / outer quad surface (`96%`).
        # - 3D Floor-Lock Elliptical Rings under both feet + 3D Cylindrical External-Torque Ribbon
        #   Arrows wrapping outward around both thighs/knees (`<--` and `-->`).
        # =========================================================================================
        draw_3d_elliptical_surface_ring(
            rd, (ank_r[0] - 6.0, ank_r[1] + 18.0), rx=44.0, ry=13.5, tilt_deg=-5.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS, fill_alpha=36, extrude_px=2.4
        )
        draw_3d_elliptical_surface_ring(
            rd, (ank_l[0] + 6.0, ank_l[1] + 18.0), rx=44.0, ry=13.5, tilt_deg=5.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS, fill_alpha=36, extrude_px=2.4
        )

        draw_3d_goniometer_joint_hub(
            rd, pivot=kn_r,
            deg_ref=128.0, deg_actual=92.0,
            ray_len_ref=88.0, ray_len_actual=94.0,
            wedge_rx=74.0, wedge_ry=62.0,
            ring_tilt_deg=-12.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        draw_3d_surface_load_sensor(
            rd, pt=(kn_l[0] + 6.0, kn_l[1] - 28.0), tilt_deg=14.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        shift = 8.0 * rep_amount
        # Right thigh/knee 3D external-rotation ribbon arrow wrapping outward
        draw_3d_perspective_wrap_ribbon_arrow(
            rd,
            ctrl_pts=[
                (kn_r[0] + 44.0, kn_r[1] - 26.0),  # starts from inner contour of right thigh
                (kn_r[0] + 14.0, kn_r[1] + 24.0),  # wraps across the front of the knee
                (kn_r[0] - 42.0, kn_r[1] + 24.0),
                (kn_r[0] - 102.0 - shift, kn_r[1] - 4.0),  # sweeps outward in 3D perspective
            ],
            w_tail=3.5, w_head=18.5,
            extrude_vec=(-1.5, 4.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_len=35.0, head_wid=31.0, ss=SS
        )
        # Left thigh/knee 3D external-rotation ribbon arrow wrapping outward
        draw_3d_perspective_wrap_ribbon_arrow(
            rd,
            ctrl_pts=[
                (kn_l[0] - 44.0, kn_l[1] - 26.0),
                (kn_l[0] - 14.0, kn_l[1] + 24.0),
                (kn_l[0] + 42.0, kn_l[1] + 24.0),
                (kn_l[0] + 102.0 + shift, kn_l[1] - 4.0),
            ],
            w_tail=3.5, w_head=18.5,
            extrude_vec=(1.5, 4.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_len=35.0, head_wid=31.0, ss=SS
        )

    elif step_idx == 3:
        # =========================================================================================
        # STEP 3 (3/4 Side View — WRONG: `+28° HEELS LIFTING` & `79% LUMBAR & PATELLA`)
        # - Marker 1 (Hip Hinge Goniometer Hub): Shows upright torso reference ray vs `+28°` excessive
        #   forward torso collapse ray.
        # - Marker 2 (Anterior Knee/Lumbar Shear Sensor): Sits flush on the overloaded patella (`79%`).
        # - 3D Floor Reference Ring under the lifted heel + 3D Upward Heel-Peel Ribbon Arrow +
        #   3D Forward-Pitching Upper-Back Ribbon Arrow.
        # =========================================================================================
        draw_3d_elliptical_surface_ring(
            rd, (ank_r[0] - 8.0, ank_r[1] + 38.0), rx=44.0, ry=13.5, tilt_deg=-8.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS, fill_alpha=36, extrude_px=2.4
        )

        draw_3d_goniometer_joint_hub(
            rd, pivot=hip_r,
            deg_ref=275.0, deg_actual=322.0,
            ray_len_ref=108.0, ray_len_actual=112.0,
            wedge_rx=88.0, wedge_ry=76.0,
            ring_tilt_deg=-18.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        draw_3d_surface_load_sensor(
            rd, pt=kn_l, tilt_deg=-14.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # 1. 3D Forward-Collapse Ribbon Arrow wrapping over the upper back/shoulder in 3D perspective
        draw_3d_perspective_wrap_ribbon_arrow(
            rd,
            ctrl_pts=[
                (sh[0] - 74.0, sh[1] - 42.0),
                (sh[0] + 10.0, sh[1] - 58.0),
                (sh[0] + 84.0, sh[1] - 10.0),
                (sh[0] + 114.0, sh[1] + 78.0),
            ],
            w_tail=3.8, w_head=18.5,
            extrude_vec=(-3.5, 3.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_len=35.0, head_wid=31.0, ss=SS
        )
        # 2. 3D Heel-Lift Ribbon Arrow peeling upward from the 3D floor ring behind the heel
        draw_3d_perspective_wrap_ribbon_arrow(
            rd,
            ctrl_pts=[
                (ank_r[0] - 16.0, ank_r[1] + 40.0),
                (ank_r[0] - 64.0, ank_r[1] + 14.0),
                (ank_r[0] - 68.0, ank_r[1] - 30.0),
                (ank_r[0] - 42.0, ank_r[1] - 88.0),
            ],
            w_tail=3.5, w_head=16.5,
            extrude_vec=(3.5, 2.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_len=31.0, head_wid=28.0, ss=SS
        )

    else:
        # =========================================================================================
        # STEP 4 (3/4 Side View — FIX: `95° PARALLEL DEPTH` & `98% QUADS & GLUTES`)
        # - Marker 1 (Hip Parallel Hinge Goniometer Hub): Shows horizontal parallel reference ray vs
        #   locked `95°` femur-to-torso hinge at the bottom of the squat.
        # - Marker 2 (Quad & Glute Peak Activation Sensor): Sits flush on the 3D quad belly (`98%`).
        # - 3D Floor-Lock Elliptical Ring under the planted foot + 3D Downward Heel-Anchor Ribbon +
        #   3D Upward Torso/Chest Drive Ribbon sweeping up the thigh/torso contour.
        # =========================================================================================
        draw_3d_elliptical_surface_ring(
            rd, (ank_r[0] + 10.0, ank_r[1] + 18.0), rx=54.0, ry=16.0, tilt_deg=-6.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS, fill_alpha=42, extrude_px=2.6
        )

        draw_3d_goniometer_joint_hub(
            rd, pivot=hip_r,
            deg_ref=0.0, deg_actual=304.0,
            ray_len_ref=108.0, ray_len_actual=112.0,
            wedge_rx=88.0, wedge_ry=76.0,
            ring_tilt_deg=-18.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        draw_3d_surface_load_sensor(
            rd, pt=(kn_l[0] - 18.0, kn_l[1] - 32.0), tilt_deg=-16.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # 1. 3D Upward Chest/Barbell Drive Ribbon Arrow curving upward in 3D space
        draw_3d_perspective_wrap_ribbon_arrow(
            rd,
            ctrl_pts=[
                (sh[0] + 42.0, sh[1] + 120.0),
                (sh[0] + 96.0, sh[1] + 66.0),
                (sh[0] + 100.0, sh[1] + 6.0),
                (sh[0] + 74.0, sh[1] - 66.0),
            ],
            w_tail=3.8, w_head=18.5,
            extrude_vec=(-3.5, 3.0),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_len=35.0, head_wid=31.0, ss=SS
        )
        # 2. 3D Downward Heel-Anchor Ribbon Arrow pressing the heel into the 3D floor ring
        draw_3d_perspective_wrap_ribbon_arrow(
            rd,
            ctrl_pts=[
                (ank_r[0] - 40.0, ank_r[1] - 86.0),
                (ank_r[0] - 68.0, ank_r[1] - 42.0),
                (ank_r[0] - 64.0, ank_r[1] + 2.0),
                (ank_r[0] - 20.0, ank_r[1] + 22.0),
            ],
            w_tail=3.5, w_head=16.5,
            extrude_vec=(3.5, 2.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_len=31.0, head_wid=28.0, ss=SS
        )

    rig_1x = rig.resize((W, H), Image.Resampling.LANCZOS)
    canvas = Image.fromarray(rgb_frame).convert("RGBA")
    canvas.alpha_composite(rig_1x)

    # Leader lines + Frameless Pure White Badges (ZERO underline bars!)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    def _leader(p_joint, p_elbow, p_badge):
        sd.line([p_joint, p_elbow, p_badge], fill=(0, 0, 0, 220), width=4)
        od.line([p_joint, p_elbow, p_badge], fill=(255, 255, 255, 220), width=2)
        bx, by = p_badge
        od.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(255, 255, 255, 255))

    if step_idx == 1:
        _leader((kn_r[0] - 18, kn_r[1] - 8), (310, 1075), (335, 1038))
        _leader((kn_l[0] + 18, kn_l[1] - 8), (715, 955), (738, 898))
        draw_v47_frameless_white_badge(od, sd, (335, 1005), "18°", "VALGUS CAVE", align="right")
        draw_v47_frameless_white_badge(od, sd, (745, 860), "84%", "ACL & MCL LOAD", align="left")
    elif step_idx == 2:
        _leader((kn_r[0] - 18, kn_r[1] - 8), (315, 1060), (340, 1022))
        _leader((kn_l[0] + 22, kn_l[1] - 32), (710, 945), (730, 898))
        draw_v47_frameless_white_badge(od, sd, (345, 990), "0°", "KNEES TRACK TOES", align="right")
        draw_v47_frameless_white_badge(od, sd, (735, 860), "96%", "VMO & GLUTE LOCK", align="left")
    elif step_idx == 3:
        _leader((hip_r[0] - 18, hip_r[1] + 10), (315, 1005), (342, 1048))
        _leader((kn_l[0] + 16, kn_l[1] - 10), (705, 955), (725, 898))
        draw_v47_frameless_white_badge(od, sd, (348, 1015), "+28°", "HEELS LIFTING", align="right")
        draw_v47_frameless_white_badge(od, sd, (730, 860), "79%", "LUMBAR & PATELLA", align="left")
    else:
        _leader((hip_r[0] - 18, hip_r[1] + 10), (310, 995), (338, 1038))
        _leader((kn_l[0] - 2, kn_l[1] - 36), (715, 945), (738, 898))
        draw_v47_frameless_white_badge(od, sd, (345, 1005), "95°", "PARALLEL DEPTH", align="right")
        draw_v47_frameless_white_badge(od, sd, (745, 860), "98%", "QUADS & GLUTES", align="left")

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=5)))
    canvas.alpha_composite(ov)
    return canvas


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
        (1, pool_s1, kp_s1, True,  0, 84, "LUMBAR & KNEE RISK",       "BARBELL BACK SQUAT",       "18° KNEE VALGUS"),
        (2, pool_s2, kp_s2, False, 2, 96, "NEUTRAL KNEE LOCK",        "GLUTE MEDIUS & VMO",       "0° KNEES TRACK TOES"),
        (3, pool_s3, kp_s3, True,  1, 79, "ANTERIOR FORCE LEAK",      "HEEL LIFT & FORWARD DIVE", "+28° FORWARD LEAN"),
        (4, pool_s4, kp_s4, False, 2, 98, "FULL QUAD & GLUTE · 4×10", "QUADRICEPS & GLUTES",      "95° PARALLEL DRIVE"),
    ]

    full_panels = []
    closeup_panels = []
    for s_idx, pool, kps, is_w, dock_idx, pct_val, tag_pill, tag_ctx, cue_1line in step_configs:
        idx = int(round(0.82 * (len(pool) - 1)))
        rgb_frame = pool[idx].copy()
        kp_cur = kps[idx]

        canvas = render_v47_scene(rgb_frame, s_idx, kp_cur, is_w, rep_amount=0.85)
        d3.draw_day3_v39_safezone_top_dock(
            canvas,
            torso_icons=torso_icons,
            active_idx=dock_idx,
            is_wrong=is_w,
            exercise_name="BARBELL BACK SQUAT",
            muscle_group="[ LEGS · QUADS & GLUTES ]",
        )
        d3.draw_day3_v39_safezone_bottom_hud(canvas, {
            "is_wrong": is_w,
            "pct": pct_val,
            "tag_pill": tag_pill,
            "tag_ctx": tag_ctx,
            "cue_1line": cue_1line,
        })

        bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        full_panels.append(cv2.resize(bgr, (540, 960), interpolation=cv2.INTER_LANCZOS4))
        crop_zoom = bgr[640:1420, 120:960]
        closeup_panels.append(cv2.resize(crop_zoom, (540, 500), interpolation=cv2.INTER_LANCZOS4))

    sheet_2x2 = np.vstack([np.hstack(full_panels[:2]), np.hstack(full_panels[2:])])
    sheet_path = os.path.join(OUT_DIR, "day3_v47_3d_design_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    closeup_2x2 = np.vstack([np.hstack(closeup_panels[:2]), np.hstack(closeup_panels[2:])])
    closeup_path = os.path.join(OUT_DIR, "day3_v47_3d_closeup_sheet.jpg")
    cv2.imwrite(closeup_path, closeup_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved v47 design sheets:", sheet_path, "and", closeup_path)


if __name__ == "__main__":
    main()
