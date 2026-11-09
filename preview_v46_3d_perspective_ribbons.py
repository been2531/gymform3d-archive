#!/usr/bin/env python3
"""
Day 3 v46 — True 3D Perspective Ribbon Arrows & Meaningful Kinematic Joint Markers (`시안`)
==========================================================================================
Addresses 중현 님's feedback (`t215u`):
1. "각도 퍼센티지 밑에 밑줄은 필요없고":
   - Completely removes the underline bar between the degree/percentage numbers and the pure-white sub-labels.
2. "화살표시나 이런게 입체적으로 따라가야하는데 아직은 어설픈것 같아. 마커점도 의미가 있어야 하는데":
   - Replaces flat 2D screen-space lines with **True 3D Perspective Extruded Ribbon Arrows** that wrap around
     the 3D cylinder of the thigh, knee, torso, and foot in 3D space (with real 3D wall extrusion thickness,
     foreshortened ribbon width, studio specular rim lighting, and a two-facet 3D sculpted delta arrowhead).
   - Gives every **Joint Marker (`마커점`)** clear **biomechanical meaning**:
     * Step 1 (Front Wrong): **Hip -> Knee -> Ankle** kinematic chain vs **Vertical Plumb Reference Axis**,
       visually proving the `18°` inward knee valgus buckle off the load line + 3D cylindrical inward-collapse
       ribbon arrows wrapping around both thighs/knees.
     * Step 2 (Front Fix): **Hip -> Knee -> Ankle** kinematic chain locked straight (`0°`) along the load axis
       + 3D cylindrical external-torque ribbon arrows wrapping outward around the thighs/knees + 3D floor-plane
       foot lock rings.
     * Step 3 (Side Wrong): **Shoulder -> Hip -> Knee -> Heel** kinematic chain with `+28°` forward torso pitch
       sector at the hip pivot + 3D floor-plane reference ellipse under the heel showing the exact heel-lift gap.
     * Step 4 (Side Fix): **Shoulder -> Hip -> Knee -> Heel** kinematic chain with `95°` parallel hip-to-knee
       depth gauge + 3D floor-locked heel ring + 3D upward drive ribbon arrow.
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

COL_RED = (240, 44, 52)
COL_RED_DARK = (115, 12, 18)
COL_RED_SOFT = (255, 158, 152)

COL_GREEN = (18, 212, 114)
COL_GREEN_DARK = (6, 82, 44)
COL_GREEN_SOFT = (162, 252, 204)

F_V46_NUM = v30.f_barlow(58, extrabold=True)
F_V46_SUB = v30.f_pretendard(22, "extrabold")


def draw_v46_clean_white_badge(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    num_str: str,
    tag_str: str,
    align: str = "right",
):
    """
    Draws the degree/percentage callout badge with:
    - ZERO underline bar ("각도 퍼센티지 밑에 밑줄은 필요없고"),
    - 58pt extrabold Pure White number,
    - 22pt extrabold Pure White sub-label tucked cleanly underneath (`y + 56`).
    """
    x, y = xy
    nb = od.textbbox((0, 0), num_str, font=F_V46_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), tag_str, font=F_V46_SUB)
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
        sd.text((nx + dx_s, y + dy_s), num_str, font=F_V46_NUM, fill=(0, 0, 0, 240))
        sd.text((tx + dx_s, y + 56 + dy_s), tag_str, font=F_V46_SUB, fill=(0, 0, 0, 245))

    od.text((nx, y), num_str, font=F_V46_NUM, fill=(255, 255, 255, 255))
    od.text((tx, y + 56), tag_str, font=F_V46_SUB, fill=(255, 255, 255, 255))


def draw_3d_perspective_floor_or_joint_ring(
    rd: ImageDraw.ImageDraw,
    center: tuple[float, float],
    rx: float,
    ry: float,
    tilt_deg: float,
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 3,
    is_floor_lock: bool = False,
):
    """
    Draws a 3D perspective elliptical ring tilted in 3D space (`rx`, `ry`, `tilt_deg`):
    - Back half of the ellipse is slightly dimmer/thinner (receding in 3D depth),
    - Front half of the ellipse has a 3D extruded dark underside + bright lit bevel + white specular rim.
    """
    cx, cy = center[0] * ss, center[1] * ss
    rx_s, ry_s = rx * ss, ry * ss
    tilt = math.radians(tilt_deg)
    ct, st = math.cos(tilt), math.sin(tilt)

    n_pts = 120
    angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=True)
    pts = []
    for a in angles:
        lx = rx_s * math.cos(a)
        ly = ry_s * math.sin(a)
        px = cx + lx * ct - ly * st
        py = cy + lx * st + ly * ct
        pts.append((px, py, math.sin(a)))  # sin(a) > 0 is the front-facing arc toward viewer

    if is_floor_lock:
        # Subtle translucent 3D floor disc fill
        poly_xy = [(p[0], p[1]) for p in pts]
        rd.polygon(poly_xy, fill=(*col_main, 34))

    # Draw segment by segment with 3D depth modulation (`sin(a)` = -1 back .. +1 front)
    for i in range(n_pts - 1):
        x0, y0, d0 = pts[i]
        x1, y1, d1 = pts[i + 1]
        depth_front = 0.5 * (d0 + d1)  # -1 (back) to +1 (front)
        w_seg = int(round((1.4 + 1.6 * max(0.0, depth_front)) * ss))
        alpha_seg = int(round(95 + 155 * (0.5 + 0.5 * depth_front)))

        if depth_front > 0.0:
            # 3D underside extrusion shadow on front-facing rim
            rd.line([(x0, y0 + 2.2 * ss), (x1, y1 + 2.2 * ss)], fill=(0, 0, 0, int(alpha_seg * 0.75)), width=w_seg + ss)
        rd.line([(x0, y0), (x1, y1)], fill=(*col_main, alpha_seg), width=w_seg)
        if depth_front > 0.25:
            rd.line([(x0, y0 - 0.5 * ss), (x1, y1 - 0.5 * ss)], fill=(255, 255, 255, int(190 * depth_front)), width=max(1, ss // 2))


def draw_3d_kinematic_joint_marker(
    rd: ImageDraw.ImageDraw,
    pt: tuple[float, float],
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    radius: float = 10.5,
    ring_rx: float = 22.0,
    ring_ry: float = 11.5,
    ring_tilt_deg: float = -12.0,
    ss: int = 3,
):
    """
    Renders a 3D biomechanical hinge joint marker (`마커점`):
    1. A 3D perspective orbital ring (`ring_rx x ring_ry`) tilted to the joint's 3D cross-sectional plane,
    2. A 3D shaded spherical ball marker at the joint center (dark 3D drop shadow + deep colored hemisphere +
       bright lit inner hemisphere + top-left hot white specular highlight dot), so it looks like a real 3D
       motion-capture sphere attached to the joint axis.
    """
    draw_3d_perspective_floor_or_joint_ring(
        rd, pt, rx=ring_rx, ry=ring_ry, tilt_deg=ring_tilt_deg,
        col_main=col_main, col_soft=col_soft, ss=ss, is_floor_lock=False
    )

    cx, cy = pt[0] * ss, pt[1] * ss
    r = radius * ss

    # 1. 3D drop shadow under sphere
    rd.ellipse([cx - r * 1.15, cy - r * 0.95 + 3.0 * ss, cx + r * 1.15, cy + r * 1.25 + 3.0 * ss], fill=(0, 0, 0, 185))
    # 2. Outer dark-shaded 3D sphere body
    rd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*col_dark, 250), outline=(*col_soft, 235), width=max(2, int(0.9 * ss)))
    # 3. Lit upper-left hemisphere
    r_mid = r * 0.78
    rd.ellipse([cx - r_mid - 0.12 * r, cy - r_mid - 0.12 * r, cx + r_mid - 0.12 * r, cy + r_mid - 0.12 * r], fill=(*col_main, 255))
    # 4. Bright soft highlight
    r_hi = r * 0.48
    rd.ellipse([cx - r_hi - 0.22 * r, cy - r_hi - 0.22 * r, cx + r_hi - 0.22 * r, cy + r_hi - 0.22 * r], fill=(*col_soft, 255))
    # 5. Crisp white 3D specular glint
    r_spec = r * 0.24
    rd.ellipse([cx - r_spec - 0.28 * r, cy - r_spec - 0.28 * r, cx + r_spec - 0.28 * r, cy + r_spec - 0.28 * r], fill=(255, 255, 255, 255))


def draw_3d_kinematic_bone_link(
    rd: ImageDraw.ImageDraw,
    p_a: tuple[float, float],
    p_b: tuple[float, float],
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    is_reference_plumb: bool = False,
    ss: int = 3,
):
    """
    Draws a meaningful biomechanical axis link between two joint markers:
    - If `is_reference_plumb=True`: draws a sleek architectural reference axis (showing the ideal `0°` line
      from which the joint deviated),
    - If `is_reference_plumb=False`: draws a 3D cylindrical bone-axis rod with dark edge cushioning,
      colored core, and white center specular spine connecting the joint markers.
    """
    x0, y0 = p_a[0] * ss, p_a[1] * ss
    x1, y1 = p_b[0] * ss, p_b[1] * ss
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 10.0:
        return
    ux, uy = dx / dist, dy / dist
    # Trim slightly so it connects cleanly at the joint sphere perimeter
    trim = 10.0 * ss
    sx0, sy0 = x0 + ux * trim, y0 + uy * trim
    sx1, sy1 = x1 - ux * trim, y1 - uy * trim

    if is_reference_plumb:
        # Clean dashed/ticked reference plumb line in 3D space
        n_dashes = int(max(6, dist / (16.0 * ss)))
        for i in range(n_dashes):
            t0 = (i + 0.15) / n_dashes
            t1 = (i + 0.68) / n_dashes
            px0, py0 = sx0 + (sx1 - sx0) * t0, sy0 + (sy1 - sy0) * t0
            px1, py1 = sx0 + (sx1 - sx0) * t1, sy0 + (sy1 - sy0) * t1
            rd.line([(px0, py0), (px1, py1)], fill=(0, 0, 0, 160), width=int(3.2 * ss))
            rd.line([(px0, py0), (px1, py1)], fill=(255, 255, 255, 165), width=int(1.5 * ss))
    else:
        # Solid 3D cylindrical bone-axis rod
        rd.line([(sx0, sy0 + 1.5 * ss), (sx1, sy1 + 1.5 * ss)], fill=(0, 0, 0, 195), width=int(6.2 * ss))
        rd.line([(sx0, sy0), (sx1, sy1)], fill=(*col_main, 235), width=int(4.0 * ss))
        rd.line([(sx0, sy0), (sx1, sy1)], fill=(255, 255, 255, 210), width=int(1.4 * ss))


def draw_3d_perspective_angle_wedge(
    rd: ImageDraw.ImageDraw,
    pivot: tuple[float, float],
    deg_ref: float,
    deg_actual: float,
    radius_x: float,
    radius_y: float,
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 3,
):
    """
    Draws a foreshortened 3D perspective angle sector & extruded rim between `deg_ref` and `deg_actual`
    centered at `pivot`, so the angle measurement sits in the same 3D perspective plane as the joint!
    """
    cx, cy = pivot[0] * ss, pivot[1] * ss
    rx, ry = radius_x * ss, radius_y * ss

    n_steps = 48
    angles = np.linspace(math.radians(deg_ref), math.radians(deg_actual), n_steps)
    outer_pts = [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in angles]
    inner_pts = [(cx + rx * 0.22 * math.cos(a), cy + ry * 0.22 * math.sin(a)) for a in reversed(angles)]

    # 1. Translucent 3D sector wedge fill
    wedge_poly = outer_pts + inner_pts
    rd.polygon(wedge_poly, fill=(*col_main, 58))

    # 2. 3D extruded dark underside rim along the outer arc
    extrude_dy = 3.2 * ss
    for i in range(len(outer_pts) - 1):
        p0 = outer_pts[i]
        p1 = outer_pts[i + 1]
        rd.line([(p0[0], p0[1] + extrude_dy), (p1[0], p1[1] + extrude_dy)], fill=(*col_dark, 235), width=int(4.2 * ss))

    # 3. Main lit 3D arc rim + white specular highlight
    for i in range(len(outer_pts) - 1):
        p0 = outer_pts[i]
        p1 = outer_pts[i + 1]
        u = i / float(max(1, len(outer_pts) - 2))
        w_a = int(round((2.6 + 2.2 * u) * ss))
        rd.line([p0, p1], fill=(*col_main, 250), width=w_a)
        rd.line([(p0[0], p0[1] - 0.8 * ss), (p1[0], p1[1] - 0.8 * ss)], fill=(255, 255, 255, int(150 + 95 * u)), width=max(1, int(1.1 * ss)))


def draw_3d_sculpted_ribbon_arrow(
    rd: ImageDraw.ImageDraw,
    ctrl_pts: list[tuple[float, float]],
    width_profile: tuple[float, float],
    extrude_vec: tuple[float, float],
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    head_length: float = 34.0,
    head_width: float = 30.0,
    wrap_behind_start: bool = True,
    ss: int = 3,
):
    """
    Renders a **True 3D Extruded Perspective Ribbon Arrow** (`입체 3D 곡면 리본 화살표`):
    - Evaluates a smooth cubic/quadratic 3D-projected spline through `ctrl_pts`,
    - Computes the ribbon's top surface polygon (`left_edge`, `right_edge`) with perspective foreshortening
      (narrower where wrapping around the limb edge, wider where facing the camera),
    - Extrudes a **3D physical side/bottom thickness wall** (`extrude_vec`) shaded in deep `col_dark`,
    - Shades the ribbon's top surface with a 3D cylindrical lighting gradient + crisp white specular bevel along
      the leading edge,
    - Renders a **Two-Facet 3D Sculpted Delta Arrowhead** at the tip (split down the central 3D ridge line into
      a bright lit facet and a rich shaded facet, plus a 3D extruded underside cliff).
    """
    pts_np = np.array(ctrl_pts, dtype=np.float32) * ss
    n_samples = 90
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

    # Compute arc-length and stop the ribbon shaft right at the base of the 3D arrowhead
    diffs = np.diff(curve, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum_len[-1])
    head_len_s = head_length * ss
    shaft_max_len = max(total_len * 0.45, total_len - head_len_s * 0.78)

    shaft_mask = cum_len <= shaft_max_len
    shaft_pts = curve[shaft_mask]
    if len(shaft_pts) < 6:
        shaft_pts = curve[:-12]

    n_s = len(shaft_pts)
    tangents = np.zeros_like(shaft_pts)
    tangents[1:-1] = shaft_pts[2:] - shaft_pts[:-2]
    tangents[0] = shaft_pts[1] - shaft_pts[0]
    tangents[-1] = shaft_pts[-1] - shaft_pts[-2]
    tnorms = np.hypot(tangents[:, 0], tangents[:, 1])[:, None] + 1e-6
    tangents = tangents / tnorms
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    w_start, w_end = width_profile[0] * ss, width_profile[1] * ss
    ex_x, ex_y = extrude_vec[0] * ss, extrude_vec[1] * ss

    left_pts = []
    right_pts = []
    for i in range(n_s):
        u = i / float(max(1, n_s - 1))
        # Foreshortened 3D perspective width: if wrapping from behind limb, starts razor-thin and swells into full 3D ribbon
        if wrap_behind_start:
            foreshorten = math.sin(u * math.pi * 0.5) ** 0.85
        else:
            foreshorten = 0.35 + 0.65 * u
        half_w = 0.5 * (w_start + (w_end - w_start) * foreshorten)
        lx = shaft_pts[i, 0] + normals[i, 0] * half_w
        ly = shaft_pts[i, 1] + normals[i, 1] * half_w
        rx = shaft_pts[i, 0] - normals[i, 0] * half_w
        ry = shaft_pts[i, 1] - normals[i, 1] * half_w
        left_pts.append((lx, ly))
        right_pts.append((rx, ry))

    # 1. Soft 3D ambient drop shadow under the entire ribbon & arrowhead
    tip_pt = curve[-1]
    base_pt = shaft_pts[-1]
    head_dir = tip_pt - base_pt
    hd_norm = float(np.hypot(head_dir[0], head_dir[1])) + 1e-6
    ux, uy = float(head_dir[0] / hd_norm), float(head_dir[1] / hd_norm)
    nx, ny = -uy, ux

    hw_s = 0.5 * head_width * ss
    # Swept 3D wing corners and recessed notch
    wing_l = (base_pt[0] - ux * (head_len_s * 0.18) + nx * hw_s, base_pt[1] - uy * (head_len_s * 0.18) + ny * hw_s)
    wing_r = (base_pt[0] - ux * (head_len_s * 0.18) - nx * hw_s, base_pt[1] - uy * (head_len_s * 0.18) - ny * hw_s)
    notch_pt = (base_pt[0] + ux * (head_len_s * 0.10), base_pt[1] + uy * (head_len_s * 0.10))
    tip_xy = (float(tip_pt[0]), float(tip_pt[1]))

    # Draw 3D drop shadow first
    sh_dx, sh_dy = ex_x * 1.45, ex_y * 1.45
    shaft_shadow_poly = [(p[0] + sh_dx, p[1] + sh_dy) for p in left_pts] + [(p[0] + sh_dx, p[1] + sh_dy) for p in reversed(right_pts)]
    rd.polygon(shaft_shadow_poly, fill=(0, 0, 0, 150))
    head_shadow_poly = [
        (tip_xy[0] + sh_dx, tip_xy[1] + sh_dy),
        (wing_l[0] + sh_dx, wing_l[1] + sh_dy),
        (notch_pt[0] + sh_dx, notch_pt[1] + sh_dy),
        (wing_r[0] + sh_dx, wing_r[1] + sh_dy),
    ]
    rd.polygon(head_shadow_poly, fill=(0, 0, 0, 175))

    # 2. 3D Extruded Side/Bottom Wall along the ribbon shaft (gives real 3D slab thickness!)
    for i in range(n_s - 1):
        u = i / float(max(1, n_s - 1))
        alpha_wall = int(round(110 + 140 * u)) if wrap_behind_start else 235
        # Extrude both outer edges so whichever edge faces `extrude_vec` shows a crisp 3D cliff wall
        for edge in (left_pts, right_pts):
            q0 = edge[i]
            q1 = edge[i + 1]
            wall_quad = [q0, q1, (q1[0] + ex_x, q1[1] + ex_y), (q0[0] + ex_x, q0[1] + ex_y)]
            rd.polygon(wall_quad, fill=(*col_dark, alpha_wall))

    # 3. Top Ribbon Surface rendered in progressive 3D lit quads (dark limb-wrap tail -> bright 3D body)
    for i in range(n_s - 1):
        u = i / float(max(1, n_s - 1))
        alpha_top = int(round(85 + 170 * (u ** 0.75))) if wrap_behind_start else int(round(190 + 65 * u))
        # Cylindrical 3D sheen: blend from `col_dark` at tail (wrapping out from behind limb) -> `col_main` -> `col_soft` near head
        if u < 0.35:
            k = u / 0.35
            c_r = int(col_dark[0] * (1 - k) + col_main[0] * k)
            c_g = int(col_dark[1] * (1 - k) + col_main[1] * k)
            c_b = int(col_dark[2] * (1 - k) + col_main[2] * k)
        else:
            k = (u - 0.35) / 0.65
            c_r = int(col_main[0] * (1 - 0.45 * k) + col_soft[0] * (0.45 * k))
            c_g = int(col_main[1] * (1 - 0.45 * k) + col_soft[1] * (0.45 * k))
            c_b = int(col_main[2] * (1 - 0.45 * k) + col_soft[2] * (0.45 * k))

        quad = [left_pts[i], left_pts[i + 1], right_pts[i + 1], right_pts[i]]
        rd.polygon(quad, fill=(c_r, c_g, c_b, alpha_top))

    # 4. Crisp 3D Specular Bevel Highlight along the upper/leading ribbon edge and central spine
    for i in range(int(n_s * 0.15), n_s - 1):
        u = i / float(max(1, n_s - 1))
        spec_a = int(round(215 * (u ** 1.1)))
        rd.line([left_pts[i], left_pts[i + 1]], fill=(255, 255, 255, spec_a), width=max(2, int(1.3 * ss)))
        c0 = (0.5 * (left_pts[i][0] + right_pts[i][0]), 0.5 * (left_pts[i][1] + right_pts[i][1]))
        c1 = (0.5 * (left_pts[i + 1][0] + right_pts[i + 1][0]), 0.5 * (left_pts[i + 1][1] + right_pts[i + 1][1]))
        rd.line([c0, c1], fill=(255, 255, 255, int(spec_a * 0.55)), width=max(1, int(0.9 * ss)))

    # 5. Two-Facet 3D Sculpted Delta Arrowhead at `tip_xy`
    # 5a. 3D Extruded cliff wall under arrowhead wings
    for w_pt in (wing_l, wing_r):
        cliff_quad = [
            tip_xy,
            w_pt,
            (w_pt[0] + ex_x, w_pt[1] + ex_y),
            (tip_xy[0] + ex_x, tip_xy[1] + ex_y),
        ]
        rd.polygon(cliff_quad, fill=(*col_dark, 250))

    # 5b. Left/Upper 3D Facet (lit by studio key light -> bright `col_soft` blended with white)
    lit_facet_col = (
        int(col_soft[0] * 0.65 + 255 * 0.35),
        int(col_soft[1] * 0.65 + 255 * 0.35),
        int(col_soft[2] * 0.65 + 255 * 0.35),
        255,
    )
    # 5c. Right/Lower 3D Facet (rich saturated `col_main` for strong 3D bevel contrast)
    shaded_facet_col = (*col_main, 255)

    rd.polygon([tip_xy, wing_l, notch_pt], fill=lit_facet_col)
    rd.polygon([tip_xy, notch_pt, wing_r], fill=shaded_facet_col)

    # 5d. Central 3D Ridge Line & Leading Wing Bevel Highlights
    rd.line([notch_pt, tip_xy], fill=(255, 255, 255, 245), width=max(2, int(1.4 * ss)))
    rd.line([wing_l, tip_xy], fill=(255, 255, 255, 235), width=max(2, int(1.3 * ss)))
    rd.line([wing_r, tip_xy], fill=(*col_soft, 220), width=max(1, int(1.0 * ss)))


def render_v46_3d_biomechanics_overlay(
    rgb_frame: np.ndarray,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 0.85,
) -> Image.Image:
    """
    Renders the complete v46 3D Perspective Ribbon & Meaningful Kinematic Marker suite onto `rgb_frame`.
    """
    SS = 3
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
        # =================================================================================
        # STEP 1 (Front View — WRONG: `18° VALGUS CAVE` & `84% ACL & MCL LOAD`)
        # Meaningful Markers:
        # - Ideal Hip-to-Ankle Vertical Load Axis (reference plumb line) vs
        #   Actual Hip -> Knee -> Ankle kinematic bone chain buckling INWARD by 18°!
        # - 3D Cylindrical Wrap-Around Ribbon Arrows curving from the outer thigh surface
        #   across the front of both knees inward toward the midline.
        # =================================================================================
        # 1. Ideal Hip -> Ankle straight reference plumb lines on both legs
        draw_3d_kinematic_bone_link(rd, hip_r, ank_r, col_main, col_soft, is_reference_plumb=True, ss=SS)
        draw_3d_kinematic_bone_link(rd, hip_l, ank_l, col_main, col_soft, is_reference_plumb=True, ss=SS)

        # 2. Actual buckled Hip -> Knee -> Ankle bone links (`> <` valgus collapse)
        draw_3d_kinematic_bone_link(rd, hip_r, kn_r, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, kn_r, ank_r, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, hip_l, kn_l, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, kn_l, ank_l, col_main, col_soft, is_reference_plumb=False, ss=SS)

        # 3. 3D Perspective Angle Wedge at right knee showing the 18° deviation from plumb axis
        draw_3d_perspective_angle_wedge(
            rd, pivot=kn_r, deg_ref=92.0, deg_actual=135.0,
            radius_x=78.0, radius_y=64.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # 4. 3D Kinematic Joint Markers on Hip — Knee — Ankle chain
        draw_3d_kinematic_joint_marker(rd, hip_r, col_main, col_dark, col_soft, radius=8.5, ring_rx=18.0, ring_ry=9.5, ring_tilt_deg=-15.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, hip_l, col_main, col_dark, col_soft, radius=8.5, ring_rx=18.0, ring_ry=9.5, ring_tilt_deg=15.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, kn_r, col_main, col_dark, col_soft, radius=11.0, ring_rx=26.0, ring_ry=13.0, ring_tilt_deg=-12.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, kn_l, col_main, col_dark, col_soft, radius=11.0, ring_rx=26.0, ring_ry=13.0, ring_tilt_deg=12.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, ank_r, col_main, col_dark, col_soft, radius=8.0, ring_rx=22.0, ring_ry=8.5, ring_tilt_deg=0.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, ank_l, col_main, col_dark, col_soft, radius=8.0, ring_rx=22.0, ring_ry=8.5, ring_tilt_deg=0.0, ss=SS)

        # 5. True 3D Cylindrical Wrap-Around Ribbon Arrows curving around both knees inward
        shift = 10.0 * rep_amount
        draw_3d_sculpted_ribbon_arrow(
            rd,
            ctrl_pts=[
                (kn_r[0] - 92.0, kn_r[1] - 34.0),  # emerges from outer contour of right thigh
                (kn_r[0] - 78.0, kn_r[1] + 22.0),  # wraps around the front-outer cylinder of the knee
                (kn_r[0] - 28.0, kn_r[1] + 28.0),
                (kn_r[0] + 34.0 + shift, kn_r[1] + 14.0),  # drives inward toward midline
            ],
            width_profile=(4.0, 17.5),
            extrude_vec=(1.5, 4.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_length=34.0, head_width=31.0, wrap_behind_start=True, ss=SS
        )
        draw_3d_sculpted_ribbon_arrow(
            rd,
            ctrl_pts=[
                (kn_l[0] + 92.0, kn_l[1] - 34.0),  # emerges from outer contour of left thigh
                (kn_l[0] + 78.0, kn_l[1] + 22.0),  # wraps around front-outer cylinder of left knee
                (kn_l[0] + 28.0, kn_l[1] + 28.0),
                (kn_l[0] - 34.0 - shift, kn_l[1] + 14.0),  # drives inward toward midline
            ],
            width_profile=(4.0, 17.5),
            extrude_vec=(-1.5, 4.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_length=34.0, head_width=31.0, wrap_behind_start=True, ss=SS
        )

    elif step_idx == 2:
        # =================================================================================
        # STEP 2 (Front View — FIX: `0° KNEES TRACK TOES` & `96% VMO & GLUTE LOCK`)
        # Meaningful Markers:
        # - Locked 0° straight Hip -> Knee -> Ankle load axis directly over the 3D floor-locked feet!
        # - 3D Cylindrical External-Torque Ribbon Arrows wrapping from the inner thigh across
        #   the front of the knee outward (`<--` and `-->`).
        # =================================================================================
        # 1. 3D Floor-Lock Elliptical Rings under both ankles/feet
        draw_3d_perspective_floor_or_joint_ring(
            rd, (ank_r[0] - 6.0, ank_r[1] + 18.0), rx=44.0, ry=14.0, tilt_deg=-6.0,
            col_main=col_main, col_soft=col_soft, ss=SS, is_floor_lock=True
        )
        draw_3d_perspective_floor_or_joint_ring(
            rd, (ank_l[0] + 6.0, ank_l[1] + 18.0), rx=44.0, ry=14.0, tilt_deg=6.0,
            col_main=col_main, col_soft=col_soft, ss=SS, is_floor_lock=True
        )

        # 2. Straight locked Hip -> Knee -> Ankle kinematic bone links
        draw_3d_kinematic_bone_link(rd, hip_r, kn_r, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, kn_r, ank_r, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, hip_l, kn_l, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, kn_l, ank_l, col_main, col_soft, is_reference_plumb=False, ss=SS)

        # 3. 3D Perspective Angle Sector at right knee showing outward tracking lock
        draw_3d_perspective_angle_wedge(
            rd, pivot=kn_r, deg_ref=115.0, deg_actual=168.0,
            radius_x=78.0, radius_y=64.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # 4. 3D Kinematic Joint Markers on Hip — Knee — Ankle chain
        draw_3d_kinematic_joint_marker(rd, hip_r, col_main, col_dark, col_soft, radius=8.5, ring_rx=18.0, ring_ry=9.5, ring_tilt_deg=-15.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, hip_l, col_main, col_dark, col_soft, radius=8.5, ring_rx=18.0, ring_ry=9.5, ring_tilt_deg=15.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, kn_r, col_main, col_dark, col_soft, radius=11.0, ring_rx=26.0, ring_ry=13.0, ring_tilt_deg=-12.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, kn_l, col_main, col_dark, col_soft, radius=11.0, ring_rx=26.0, ring_ry=13.0, ring_tilt_deg=12.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, ank_r, col_main, col_dark, col_soft, radius=8.5, ring_rx=22.0, ring_ry=8.5, ring_tilt_deg=-5.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, ank_l, col_main, col_dark, col_soft, radius=8.5, ring_rx=22.0, ring_ry=8.5, ring_tilt_deg=5.0, ss=SS)

        # 5. True 3D Cylindrical External-Torque Ribbon Arrows wrapping OUTWARD around both knees
        shift = 10.0 * rep_amount
        draw_3d_sculpted_ribbon_arrow(
            rd,
            ctrl_pts=[
                (kn_r[0] + 42.0, kn_r[1] - 24.0),  # starts on inner aspect of right knee/thigh
                (kn_r[0] + 12.0, kn_r[1] + 24.0),  # wraps across the front surface of the patella
                (kn_r[0] - 42.0, kn_r[1] + 22.0),
                (kn_r[0] - 96.0 - shift, kn_r[1] - 4.0),  # sweeps outward in 3D space
            ],
            width_profile=(4.0, 17.5),
            extrude_vec=(-1.5, 4.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_length=34.0, head_width=31.0, wrap_behind_start=True, ss=SS
        )
        draw_3d_sculpted_ribbon_arrow(
            rd,
            ctrl_pts=[
                (kn_l[0] - 42.0, kn_l[1] - 24.0),  # starts on inner aspect of left knee/thigh
                (kn_l[0] - 12.0, kn_l[1] + 24.0),  # wraps across the front surface of the patella
                (kn_l[0] + 42.0, kn_l[1] + 22.0),
                (kn_l[0] + 96.0 + shift, kn_l[1] - 4.0),  # sweeps outward in 3D space
            ],
            width_profile=(4.0, 17.5),
            extrude_vec=(1.5, 4.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_length=34.0, head_width=31.0, wrap_behind_start=True, ss=SS
        )

    elif step_idx == 3:
        # =================================================================================
        # STEP 3 (3/4 Side View — WRONG: `+28° HEELS LIFTING` & `79% LUMBAR & PATELLA`)
        # Meaningful Markers:
        # - Ideal Upright Torso Reference Ray vs Collapsed Torso Axis (`Shoulder -> Hip`) showing `+28°` pitch,
        # - Connected `Shoulder -> Hip -> Knee -> Lifted Heel` kinematic chain,
        # - 3D Floor Reference Ring on the ground under the heel + 3D Upward Heel-Lift Ribbon Arrow,
        # - 3D Forward-Pitching Torso Collapse Ribbon Arrow sweeping over the upper back/barbell.
        # =================================================================================
        # 1. Ideal upright torso reference ray from Hip upward + actual collapsed Shoulder->Hip->Knee->Ankle chain
        ideal_sh = (hip_r[0] + 42.0, hip_r[1] - 245.0)
        draw_3d_kinematic_bone_link(rd, hip_r, ideal_sh, col_main, col_soft, is_reference_plumb=True, ss=SS)
        draw_3d_kinematic_bone_link(rd, sh, hip_r, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, hip_r, kn_r, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, kn_r, ank_r, col_main, col_soft, is_reference_plumb=False, ss=SS)

        # 2. 3D Floor Reference Ring showing where the heel SHOULD be on the floor (`ank_r[1] + 38`)
        floor_heel_pt = (ank_r[0] - 8.0, ank_r[1] + 38.0)
        draw_3d_perspective_floor_or_joint_ring(
            rd, floor_heel_pt, rx=42.0, ry=13.5, tilt_deg=-8.0,
            col_main=col_main, col_soft=col_soft, ss=SS, is_floor_lock=True
        )

        # 3. 3D Perspective Angle Wedge at Hip Pivot showing `+28°` forward torso collapse
        draw_3d_perspective_angle_wedge(
            rd, pivot=hip_r, deg_ref=278.0, deg_actual=328.0,
            radius_x=96.0, radius_y=82.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # 4. 3D Kinematic Joint Markers on Shoulder (Barbell Load), Hip Hinge, Knee, and Lifted Heel
        draw_3d_kinematic_joint_marker(rd, sh, col_main, col_dark, col_soft, radius=9.5, ring_rx=21.0, ring_ry=11.0, ring_tilt_deg=18.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, hip_r, col_main, col_dark, col_soft, radius=11.5, ring_rx=26.0, ring_ry=13.5, ring_tilt_deg=-18.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, kn_r, col_main, col_dark, col_soft, radius=10.0, ring_rx=23.0, ring_ry=12.0, ring_tilt_deg=-14.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, ank_r, col_main, col_dark, col_soft, radius=9.5, ring_rx=22.0, ring_ry=10.0, ring_tilt_deg=-8.0, ss=SS)

        # 5. True 3D Forward-Collapse Ribbon Arrow arcing over the upper back/shoulders in 3D perspective
        draw_3d_sculpted_ribbon_arrow(
            rd,
            ctrl_pts=[
                (sh[0] - 68.0, sh[1] - 48.0),
                (sh[0] + 15.0, sh[1] - 58.0),
                (sh[0] + 82.0, sh[1] - 12.0),
                (sh[0] + 112.0, sh[1] + 74.0),
            ],
            width_profile=(4.5, 18.0),
            extrude_vec=(-3.5, 3.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_length=35.0, head_width=31.0, wrap_behind_start=True, ss=SS
        )

        # 6. True 3D Heel-Lift Ribbon Arrow rising from the 3D floor ring up behind the lifting heel
        draw_3d_sculpted_ribbon_arrow(
            rd,
            ctrl_pts=[
                (ank_r[0] - 42.0, ank_r[1] + 40.0),
                (ank_r[0] - 66.0, ank_r[1] + 12.0),
                (ank_r[0] - 68.0, ank_r[1] - 32.0),
                (ank_r[0] - 48.0, ank_r[1] - 88.0),
            ],
            width_profile=(4.0, 16.0),
            extrude_vec=(3.5, 2.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_length=30.0, head_width=27.0, wrap_behind_start=True, ss=SS
        )

    else:
        # =================================================================================
        # STEP 4 (3/4 Side View — FIX: `95° PARALLEL DEPTH` & `98% QUADS & GLUTES`)
        # Meaningful Markers:
        # - Locked `Shoulder -> Hip -> Knee -> Floor-Planted Heel` kinematic chain,
        # - Horizontal parallel depth reference ray from Hip to Knee (`95° PARALLEL DEPTH`),
        # - 3D Floor-Lock Ring planted firmly around the heel + downward heel-anchor 3D ribbon,
        # - 3D Upward Chest/Barbell Drive Ribbon Arrow rising along the torso.
        # =================================================================================
        # 1. Parallel femur reference line + locked Shoulder -> Hip -> Knee -> Ankle kinematic chain
        parallel_ref = (hip_r[0] + 195.0, hip_r[1])
        draw_3d_kinematic_bone_link(rd, hip_r, parallel_ref, col_main, col_soft, is_reference_plumb=True, ss=SS)
        draw_3d_kinematic_bone_link(rd, sh, hip_r, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, hip_r, kn_r, col_main, col_soft, is_reference_plumb=False, ss=SS)
        draw_3d_kinematic_bone_link(rd, kn_r, ank_r, col_main, col_soft, is_reference_plumb=False, ss=SS)

        # 2. 3D Floor-Lock Elliptical Ring planted flat under the heel/foot
        draw_3d_perspective_floor_or_joint_ring(
            rd, (ank_r[0] + 8.0, ank_r[1] + 18.0), rx=54.0, ry=16.5, tilt_deg=-6.0,
            col_main=col_main, col_soft=col_soft, ss=SS, is_floor_lock=True
        )

        # 3. 3D Perspective Angle Wedge at Hip showing the `95°` parallel hip/femur hinge
        draw_3d_perspective_angle_wedge(
            rd, pivot=hip_r, deg_ref=302.0, deg_actual=360.0,
            radius_x=96.0, radius_y=82.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # 4. 3D Kinematic Joint Markers on Shoulder, Hip, Knee, and Planted Heel
        draw_3d_kinematic_joint_marker(rd, sh, col_main, col_dark, col_soft, radius=9.5, ring_rx=21.0, ring_ry=11.0, ring_tilt_deg=-12.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, hip_r, col_main, col_dark, col_soft, radius=11.5, ring_rx=26.0, ring_ry=13.5, ring_tilt_deg=-18.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, kn_r, col_main, col_dark, col_soft, radius=10.5, ring_rx=24.0, ring_ry=12.5, ring_tilt_deg=-14.0, ss=SS)
        draw_3d_kinematic_joint_marker(rd, ank_r, col_main, col_dark, col_soft, radius=9.5, ring_rx=23.0, ring_ry=10.0, ring_tilt_deg=-6.0, ss=SS)

        # 5. True 3D Upward Chest/Torso Drive Ribbon Arrow sweeping upward in 3D space
        draw_3d_sculpted_ribbon_arrow(
            rd,
            ctrl_pts=[
                (sh[0] + 55.0, sh[1] + 115.0),
                (sh[0] + 95.0, sh[1] + 62.0),
                (sh[0] + 98.0, sh[1] + 4.0),
                (sh[0] + 78.0, sh[1] - 64.0),
            ],
            width_profile=(4.5, 18.0),
            extrude_vec=(-3.5, 3.0),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_length=35.0, head_width=31.0, wrap_behind_start=True, ss=SS
        )

        # 6. True 3D Heel-Anchor Ribbon Arrow locking the heel down into the 3D floor ring
        draw_3d_sculpted_ribbon_arrow(
            rd,
            ctrl_pts=[
                (ank_r[0] - 46.0, ank_r[1] - 88.0),
                (ank_r[0] - 66.0, ank_r[1] - 42.0),
                (ank_r[0] - 64.0, ank_r[1] - 4.0),
                (ank_r[0] - 38.0, ank_r[1] + 22.0),
            ],
            width_profile=(4.0, 15.5),
            extrude_vec=(3.5, 2.5),
            col_main=col_main, col_dark=col_dark, col_soft=col_soft,
            head_length=29.0, head_width=26.0, wrap_behind_start=True, ss=SS
        )

    # Downsample 3x supersampled 3D rig onto canvas
    rig_1x = rig.resize((W, H), Image.Resampling.LANCZOS)
    canvas = Image.fromarray(rgb_frame).convert("RGBA")
    canvas.alpha_composite(rig_1x)

    # Draw clean pure-white callout badges (NO underline bars!) + crisp architectural leader lines
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    def _leader(p_joint, p_elbow, p_badge):
        sd.line([p_joint, p_elbow, p_badge], fill=(0, 0, 0, 210), width=4)
        od.line([p_joint, p_elbow, p_badge], fill=(255, 255, 255, 210), width=2)
        bx, by = p_badge
        od.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(255, 255, 255, 255))

    if step_idx == 1:
        _leader((kn_r[0] - 16, kn_r[1] - 10), (310, 1080), (335, 1040))
        _leader((kn_l[0] + 16, kn_l[1] - 10), (715, 955), (735, 898))
        draw_v46_clean_white_badge(od, sd, (335, 1005), "18°", "VALGUS CAVE", align="right")
        draw_v46_clean_white_badge(od, sd, (745, 860), "84%", "ACL & MCL LOAD", align="left")
    elif step_idx == 2:
        _leader((kn_r[0] - 16, kn_r[1] - 10), (315, 1065), (340, 1025))
        _leader((kn_l[0] + 16, kn_l[1] - 10), (710, 955), (730, 898))
        draw_v46_clean_white_badge(od, sd, (345, 990), "0°", "KNEES TRACK TOES", align="right")
        draw_v46_clean_white_badge(od, sd, (735, 860), "96%", "VMO & GLUTE LOCK", align="left")
    elif step_idx == 3:
        _leader((hip_r[0] - 16, hip_r[1] + 12), (315, 1005), (342, 1050))
        _leader((kn_l[0] + 12, kn_l[1] - 18), (705, 955), (725, 898))
        draw_v46_clean_white_badge(od, sd, (348, 1015), "+28°", "HEELS LIFTING", align="right")
        draw_v46_clean_white_badge(od, sd, (730, 860), "79%", "LUMBAR & PATELLA", align="left")
    else:
        _leader((hip_r[0] - 16, hip_r[1] + 12), (310, 995), (338, 1040))
        _leader((kn_l[0] + 12, kn_l[1] - 18), (715, 955), (735, 898))
        draw_v46_clean_white_badge(od, sd, (345, 1005), "95°", "PARALLEL DEPTH", align="right")
        draw_v46_clean_white_badge(od, sd, (745, 860), "98%", "QUADS & GLUTES", align="left")

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

        canvas = render_v46_3d_biomechanics_overlay(rgb_frame, s_idx, kp_cur, is_w, rep_amount=0.85)
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
    sheet_path = os.path.join(OUT_DIR, "day3_v46_3d_design_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    closeup_2x2 = np.vstack([np.hstack(closeup_panels[:2]), np.hstack(closeup_panels[2:])])
    closeup_path = os.path.join(OUT_DIR, "day3_v46_3d_closeup_sheet.jpg")
    cv2.imwrite(closeup_path, closeup_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved v46 design sheets:", sheet_path, "and", closeup_path)


if __name__ == "__main__":
    main()
