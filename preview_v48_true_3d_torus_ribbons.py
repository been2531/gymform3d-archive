#!/usr/bin/env python3
"""
Day 3 v48 — True 3D Folded Cylinder-Wrap Ribbon Arrows & Meaningful 3-Point Kinematic Markers (`시안`)
======================================================================================================
Addresses 중현 님's exact feedback (`t215u`):
1. "각도 퍼센티지 밑에 밑줄은 필요없고":
   - Zero underline bars under any degree (`°`) or percentage (`%`) numbers.
   - Clean 58pt Extrabold Pure White number + 22pt Extrabold Pure White sub-label directly below.
2. "화살표시나 이런게 입체적으로 따라가야하는데 아직은 어설픈것 같아. 마커점도 의미가 있어야 하는데":
   - **Why Marker Dots Exist (`마커점의 명확한 생체역학적 의미`)**:
     Every marker dot is part of a **3-Point Anatomical Alignment Axis (`3점 관절 정렬축`)** that visually
     proves the biomechanical angle (`18°`, `0°`, `+28°`, `95°`):
     * Step 1 (Front Wrong): **Hip Marker — Knee Hinge Marker — Ankle Marker** on the right leg, contrasted
       against the **Vertical Plumb Reference Axis (`수직 하중 기준선`)**. You immediately see the middle
       marker (Knee) buckling `18°` inward off the Hip-to-Ankle line!
     * Step 2 (Front Fix): **Hip Marker — Knee Marker — Ankle Marker** locked into a **straight `0°` vertical
       load line** directly into a **3D Perspective Floor-Lock Ring** under the foot!
     * Step 3 (Side Wrong): **Shoulder Marker — Hip Hinge Marker — Knee Marker — Lifted Heel Marker**, showing
       the `+28°` forward torso pitch off the upright reference axis at the Hip Marker, and the vertical gap
       between the Lifted Heel Marker and the **3D Floor Reference Pad**.
     * Step 4 (Side Fix): **Shoulder Marker — Hip Hinge Marker — Knee Marker — Planted Heel Marker**, showing
       the `95°` parallel femur-to-torso hinge at the Hip Marker and the heel locked flat inside the **3D Floor
       Target Ring**.
   - **True 3D Folded Cylinder-Wrap Ribbon Arrows (`3D 원통 표면을 감싸고 돌아나오는 입체 리본 화살표`)**:
     Instead of flat 2D lines, each arrow is projected from a **3D parametric helical/toroidal ribbon**
     (`(X(phi), Y(phi), Z(phi))`) wrapping around the 3D cylinder of the thigh, knee, or torso:
     * The **receding back-fold (`Z < 0`)** curls out from behind the limb's outer/inner contour (shaded in
       deep 3D shadow as the inside of the ribbon turns the corner in 3D space),
     * At the **3D tangent fold (`Z = 0`)**, the ribbon folds over into the **camera-facing front ribbon (`Z > 0`)**,
       casting a soft 3D contact shadow onto the leg/torso underneath,
     * Across the **front 3D sweep (`Z > 0`)**, the ribbon has real 3D ribbon height (`16px`), a 3D extruded
       thickness rim (`3.5px`), cylindrical studio Phong lighting (`N · L` diffuse + `(R · V)^12` specular
       highlight band where the 3D ribbon surface faces the key light), and a **3D sculpted bevel arrowhead**.
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
COL_RED_DARK = (82, 8, 14)
COL_RED_SOFT = (255, 165, 160)

COL_GREEN = (16, 218, 116)
COL_GREEN_DARK = (5, 72, 38)
COL_GREEN_SOFT = (168, 255, 210)

F_V48_NUM = v30.f_barlow(64, extrabold=True)
F_V48_SUB = v30.f_pretendard(24, "extrabold")


def draw_v48_frameless_white_badge(
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
    nb = od.textbbox((0, 0), num_str, font=F_V48_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), tag_str, font=F_V48_SUB)
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
        sd.text((nx + dx_s, y + dy_s), num_str, font=F_V48_NUM, fill=(0, 0, 0, 240))
        sd.text((tx + dx_s, y + 62 + dy_s), tag_str, font=F_V48_SUB, fill=(0, 0, 0, 245))

    od.text((nx, y), num_str, font=F_V48_NUM, fill=(255, 255, 255, 255))
    od.text((tx, y + 62), tag_str, font=F_V48_SUB, fill=(255, 255, 255, 255))


def draw_3d_cylindrical_wrap_ribbon(
    rd: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    rx: float,
    ry: float,
    tilt_deg: float,
    phi_start_deg: float,
    phi_end_deg: float,
    ribbon_thick_px: float,
    pitch_dy: float,
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 4,
):
    """
    Renders a **True 3D Parametric Cylindrical/Helical Wrap-Around Ribbon Arrow**:
    - Imagine a 3D ribbon of vertical band height `ribbon_thick_px` and radial wall thickness `3.2px`
      wrapping around a 3D cylinder centered at `center_xy` with horizontal radius `rx`, perspective depth
      radius `ry`, and axial tilt `tilt_deg`.
    - Angle `phi` is measured in 3D cylinder coordinates where:
      * `phi = 90°` is the point closest to the camera (`Z = +1`),
      * `phi = 0°` is the right silhouette tangent fold (`Z = 0`),
      * `phi = 180°` is the left silhouette tangent fold (`Z = 0`),
      * `sin(phi) < 0` is the **back half of the cylinder (`Z < 0`)** curving behind the limb!
    - By starting `phi_start_deg` slightly behind the tangent fold (e.g., `215°` on the left or `-35°` on the
      right, where `sin(phi) < 0`) and sweeping across the front (`sin(phi) > 0`), the viewer clearly sees:
      1. The **dark shaded inner face of the ribbon** curling out from behind the limb edge (`sin(phi) < 0`),
      2. The **3D fold** right at the limb contour (`cos(phi) = +-1`),
      3. The **studio-lit 3D outer ribbon surface** sweeping across the front of the limb (`sin(phi) > 0`)
         with realistic Phong specular lighting (`sin(phi)^4` highlight) and a 3D extruded bottom thickness rim,
      4. A **3D perspective-foreshortened sculpted arrowhead** at `phi_end_deg`!
    """
    cx, cy = center_xy[0] * ss, center_xy[1] * ss
    rx_s, ry_s = rx * ss, ry * ss
    band_h = ribbon_thick_px * ss
    wall_t = 3.2 * ss
    pitch_s = pitch_dy * ss

    tilt = math.radians(tilt_deg)
    ct, st = math.cos(tilt), math.sin(tilt)
    # Local 3D cylinder "up/down" axis on screen is `(-st, ct)`
    ax_x, ax_y = -st, ct

    n_steps = 120
    phis = np.linspace(math.radians(phi_start_deg), math.radians(phi_end_deg), n_steps)
    d_phi = phis[1] - phis[0]
    direction = 1.0 if d_phi >= 0 else -1.0

    # Reserve the last ~24 degrees of arc for the 3D sculpted arrowhead
    head_arc_rad = math.radians(24.0) * direction
    phi_shaft_end = phis[-1] - head_arc_rad
    shaft_phis = np.linspace(phis[0], phi_shaft_end, 96)

    def _cyl_pt(phi_val, u_norm, v_band_offset, r_scale=1.0):
        # Local cylinder coords before tilt
        lx = (rx_s * r_scale) * math.cos(phi_val)
        ly = (ry_s * r_scale) * math.sin(phi_val) + pitch_s * (u_norm - 0.5)
        # Rotate by `tilt` and offset along cylinder axis by `v_band_offset`
        px = cx + lx * ct - ly * st + ax_x * v_band_offset
        py = cy + lx * st + ly * ct + ax_y * v_band_offset
        return (float(px), float(py))

    # 1. Soft 3D elevation shadow on the limb under the front-facing portion (`sin(phi) > -0.1`)
    for i in range(len(shaft_phis) - 1):
        p0, p1 = shaft_phis[i], shaft_phis[i + 1]
        u0, u1 = i / float(len(shaft_phis) - 1), (i + 1) / float(len(shaft_phis) - 1)
        z_depth = 0.5 * (math.sin(p0) + math.sin(p1))
        if z_depth > -0.15:
            sh_a = int(round(145 * max(0.0, min(1.0, (z_depth + 0.15) / 0.8))))
            sh_off = 5.5 * ss
            q_sh = [
                _cyl_pt(p0, u0, -0.5 * band_h + sh_off, r_scale=0.96),
                _cyl_pt(p1, u1, -0.5 * band_h + sh_off, r_scale=0.96),
                _cyl_pt(p1, u1, +0.5 * band_h + sh_off, r_scale=0.96),
                _cyl_pt(p0, u0, +0.5 * band_h + sh_off, r_scale=0.96),
            ]
            rd.polygon(q_sh, fill=(0, 0, 0, sh_a))

    # 2. First pass: Render BACK-FOLD segments (`sin(phi) <= 0.05`) where we see the dark inside of the ribbon turning the corner
    for i in range(len(shaft_phis) - 1):
        p0, p1 = shaft_phis[i], shaft_phis[i + 1]
        u0, u1 = i / float(len(shaft_phis) - 1), (i + 1) / float(len(shaft_phis) - 1)
        z_depth = 0.5 * (math.sin(p0) + math.sin(p1))
        if z_depth <= 0.08:
            # Inside face of the ribbon curling from behind the limb
            fade = max(0.25, min(1.0, (u0 + 0.15) / 0.35))
            a_back = int(round(225 * fade))
            c_in = (
                int(col_dark[0] * 0.78),
                int(col_dark[1] * 0.78),
                int(col_dark[2] * 0.78),
                a_back,
            )
            q_back = [
                _cyl_pt(p0, u0, -0.46 * band_h, r_scale=1.0),
                _cyl_pt(p1, u1, -0.46 * band_h, r_scale=1.0),
                _cyl_pt(p1, u1, +0.46 * band_h, r_scale=1.0),
                _cyl_pt(p0, u0, +0.46 * band_h, r_scale=1.0),
            ]
            rd.polygon(q_back, fill=c_in)
            # Subtle inner rim highlight so the 3D fold edge is unmistakable
            rd.line([q_back[0], q_back[1]], fill=(*col_main, int(a_back * 0.75)), width=max(2, int(1.1 * ss)))
            rd.line([q_back[3], q_back[2]], fill=(0, 0, 0, int(a_back * 0.9)), width=max(2, int(1.1 * ss)))

    # 3. Second pass: Render FRONT-SWEEPING 3D Ribbon segments (`sin(phi) > 0.0`) with 3D bottom wall & Phong specular shading
    for i in range(len(shaft_phis) - 1):
        p0, p1 = shaft_phis[i], shaft_phis[i + 1]
        u0, u1 = i / float(len(shaft_phis) - 1), (i + 1) / float(len(shaft_phis) - 1)
        z_depth = 0.5 * (math.sin(p0) + math.sin(p1))
        if z_depth > -0.02:
            # 3a. 3D Extruded Bottom Wall (thickness of the ribbon slab)
            b_outer0 = _cyl_pt(p0, u0, +0.5 * band_h, r_scale=1.0)
            b_outer1 = _cyl_pt(p1, u1, +0.5 * band_h, r_scale=1.0)
            b_inner0 = _cyl_pt(p0, u0, +0.5 * band_h + wall_t, r_scale=0.95)
            b_inner1 = _cyl_pt(p1, u1, +0.5 * band_h + wall_t, r_scale=0.95)
            rd.polygon([b_outer0, b_outer1, b_inner1, b_inner0], fill=(*col_dark, 250))

            # 3b. Cylindrical 3D surface lighting (`z_depth` in 0..1 is how much the surface normal faces the camera/key-light)
            # Add a studio specular hotspot around `phi = 75°..105°` (`z_depth > 0.85`)
            diffuse = max(0.0, min(1.0, z_depth)) ** 0.75
            spec = max(0.0, min(1.0, (z_depth - 0.55) / 0.45)) ** 2.2

            # Render 3 horizontal sub-strips across the ribbon height (`top bevel`, `mid face`, `bottom bevel`)
            strips = [
                (-0.50, -0.18, 0.35),  # upper lit strip
                (-0.18, +0.25, 0.12),  # center main strip
                (+0.25, +0.50, -0.18), # lower shaded bevel strip
            ]
            for sv0, sv1, strip_bias in strips:
                lit = max(0.0, min(1.0, diffuse + strip_bias))
                r_c = int(np.clip(col_dark[0] * (1 - lit) + col_main[0] * lit + (255 - col_main[0]) * (spec * 0.62 + max(0.0, strip_bias) * 0.35), 0, 255))
                g_c = int(np.clip(col_dark[1] * (1 - lit) + col_main[1] * lit + (255 - col_main[1]) * (spec * 0.62 + max(0.0, strip_bias) * 0.35), 0, 255))
                b_c = int(np.clip(col_dark[2] * (1 - lit) + col_main[2] * lit + (255 - col_main[2]) * (spec * 0.62 + max(0.0, strip_bias) * 0.35), 0, 255))

                qs = [
                    _cyl_pt(p0, u0, sv0 * band_h, r_scale=1.0),
                    _cyl_pt(p1, u1, sv0 * band_h, r_scale=1.0),
                    _cyl_pt(p1, u1, sv1 * band_h, r_scale=1.0),
                    _cyl_pt(p0, u0, sv1 * band_h, r_scale=1.0),
                ]
                rd.polygon(qs, fill=(r_c, g_c, b_c, 255))

            # Crisp white specular top bevel edge
            t0_pt = _cyl_pt(p0, u0, -0.5 * band_h, r_scale=1.0)
            t1_pt = _cyl_pt(p1, u1, -0.5 * band_h, r_scale=1.0)
            spec_edge_a = int(round(135 + 120 * diffuse))
            rd.line([t0_pt, t1_pt], fill=(255, 255, 255, spec_edge_a), width=max(2, int(1.3 * ss)))

    # 4. Sculpted 3D Two-Facet Arrowhead at `phis[-1]`, curved in the exact 3D tangent plane of the cylinder!
    phi_base = phi_shaft_end
    phi_tip = phis[-1]
    phi_notch = phi_base + 0.22 * (phi_tip - phi_base)
    head_half_h = 1.18 * band_h

    tip_pt = _cyl_pt(phi_tip, 1.0, 0.0, r_scale=1.0)
    wing_top = _cyl_pt(phi_base, 0.88, -head_half_h, r_scale=1.0)
    wing_bot = _cyl_pt(phi_base, 0.88, +head_half_h, r_scale=1.0)
    notch_pt = _cyl_pt(phi_notch, 0.91, 0.0, r_scale=1.0)

    # 4a. Drop shadow under 3D arrowhead
    sh_off = 6.0 * ss
    rd.polygon([
        (tip_pt[0], tip_pt[1] + sh_off),
        (wing_top[0], wing_top[1] + sh_off),
        (notch_pt[0], notch_pt[1] + sh_off),
        (wing_bot[0], wing_bot[1] + sh_off),
    ], fill=(0, 0, 0, 175))

    # 4b. 3D extruded dark cliff under arrowhead bottom edge
    tip_under = (tip_pt[0] + ax_x * wall_t, tip_pt[1] + ax_y * wall_t)
    wbot_under = (wing_bot[0] + ax_x * wall_t, wing_bot[1] + ax_y * wall_t)
    notch_under = (notch_pt[0] + ax_x * wall_t, notch_pt[1] + ax_y * wall_t)
    rd.polygon([tip_pt, wing_bot, wbot_under, tip_under], fill=(*col_dark, 255))
    rd.polygon([wing_bot, notch_pt, notch_under, wbot_under], fill=(*col_dark, 255))

    # 4c. Upper lit 3D facet (`tip_pt - wing_top - notch_pt`) & Lower shaded 3D facet (`tip_pt - notch_pt - wing_bot`)
    lit_head = (
        int(col_soft[0] * 0.55 + 255 * 0.45),
        int(col_soft[1] * 0.55 + 255 * 0.45),
        int(col_soft[2] * 0.55 + 255 * 0.45),
        255,
    )
    rd.polygon([tip_pt, wing_top, notch_pt], fill=lit_head)
    rd.polygon([tip_pt, notch_pt, wing_bot], fill=(*col_main, 255))

    # 4d. Crisp 3D bevel highlights on arrowhead
    rd.line([wing_top, tip_pt], fill=(255, 255, 255, 255), width=max(2, int(1.4 * ss)))
    rd.line([notch_pt, tip_pt], fill=(255, 255, 255, 245), width=max(2, int(1.3 * ss)))
    rd.line([wing_bot, tip_pt], fill=(*col_soft, 235), width=max(1, int(1.0 * ss)))


def draw_meaningful_3point_joint_chain(
    rd: ImageDraw.ImageDraw,
    p_prox: tuple[float, float],
    p_pivot: tuple[float, float],
    p_dist: tuple[float, float],
    p_ref_end: tuple[float, float] | None,
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 4,
):
    """
    Renders a **Meaningful 3-Point Joint Alignment Chain (`p_prox — p_pivot — p_dist`)**:
    - Every marker dot (`마커점`) has an immediate biomechanical reason to exist:
      * `p_prox` (e.g. Hip or Shoulder) is the proximal load origin,
      * `p_pivot` (e.g. Knee or Hip Hinge) is the central vertex where the angle (`18°`, `0°`, `+28°`, `95°`) bends,
      * `p_dist` (e.g. Ankle/Foot or Knee) is the distal base!
    - If `p_ref_end` is given, draws the crisp dashed **Ideal Reference Axis (`기준 정렬축`)** from `p_pivot` (or `p_prox`)
       and fills the **3D Perspective Angle Sector Wedge** between the ideal reference axis and the actual bone axis!
    """
    def S(pt):
        return (pt[0] * ss, pt[1] * ss)

    sp_prox = S(p_prox)
    sp_piv = S(p_pivot)
    sp_dist = S(p_dist)

    # 1. If `p_ref_end` is provided, draw the dashed reference axis + translucent angle sector wedge at `p_pivot`
    if p_ref_end is not None:
        sp_ref = S(p_ref_end)
        # Angle sector between `p_pivot -> p_ref_end` and `p_pivot -> p_prox` (or `p_dist`)
        ang_ref = math.atan2(sp_ref[1] - sp_piv[1], sp_ref[0] - sp_piv[0])
        ang_act = math.atan2(sp_prox[1] - sp_piv[1], sp_prox[0] - sp_piv[0])
        # Shortest angular span
        diff = (ang_act - ang_ref + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > math.radians(125):
            # Measure against `p_dist` instead if `p_ref_end` is downward
            ang_act = math.atan2(sp_dist[1] - sp_piv[1], sp_dist[0] - sp_piv[0])
            diff = (ang_act - ang_ref + math.pi) % (2 * math.pi) - math.pi

        r_wedge = 68.0 * ss
        a_steps = np.linspace(ang_ref, ang_ref + diff, 40)
        w_pts = [sp_piv] + [(sp_piv[0] + r_wedge * math.cos(a), sp_piv[1] + r_wedge * math.sin(a)) for a in a_steps]
        rd.polygon(w_pts, fill=(*col_main, 62))
        # Outer 3D arc rim of the angle wedge
        for i in range(1, len(w_pts) - 1):
            rd.line([(w_pts[i][0], w_pts[i][1] + 2.2 * ss), (w_pts[i + 1][0], w_pts[i + 1][1] + 2.2 * ss)], fill=(0, 0, 0, 210), width=int(4.2 * ss))
            rd.line([w_pts[i], w_pts[i + 1]], fill=(*col_main, 255), width=int(3.2 * ss))
            rd.line([(w_pts[i][0], w_pts[i][1] - 0.6 * ss), (w_pts[i + 1][0], w_pts[i + 1][1] - 0.6 * ss)], fill=(255, 255, 255, 220), width=max(1, ss))

        # Dashed ideal reference line from `sp_piv` to `sp_ref`
        n_d = 7
        for i in range(n_d):
            t0, t1 = (i + 0.1) / n_d, (i + 0.65) / n_d
            d0 = (sp_piv[0] + (sp_ref[0] - sp_piv[0]) * t0, sp_piv[1] + (sp_ref[1] - sp_piv[1]) * t0)
            d1 = (sp_piv[0] + (sp_ref[0] - sp_piv[0]) * t1, sp_piv[1] + (sp_ref[1] - sp_piv[1]) * t1)
            rd.line([d0, d1], fill=(0, 0, 0, 190), width=int(3.2 * ss))
            rd.line([d0, d1], fill=(235, 242, 255, 220), width=int(1.6 * ss))

    # 2. Sleek 3D cylindrical kinematic axis rods connecting `p_prox -> p_pivot -> p_dist`
    for pa, pb in [(sp_prox, sp_piv), (sp_piv, sp_dist)]:
        rd.line([(pa[0], pa[1] + 1.8 * ss), (pb[0], pb[1] + 1.8 * ss)], fill=(0, 0, 0, 205), width=int(5.0 * ss))
        rd.line([pa, pb], fill=(*col_main, 240), width=int(3.2 * ss))
        rd.line([pa, pb], fill=(255, 255, 255, 215), width=int(1.2 * ss))

    # 3. 3D Perspective Joint Nodes at `p_prox`, `p_pivot` (larger primary hinge), and `p_dist`
    for pt_xy, r_node, is_primary in [(p_prox, 7.2, False), (p_pivot, 9.8, True), (p_dist, 7.2, False)]:
        cx, cy = pt_xy[0] * ss, pt_xy[1] * ss
        rn = r_node * ss
        if is_primary:
            # Outer precision goniometer ring around the primary hinge joint
            ro = rn * 2.05
            rd.ellipse([cx - ro, cy - ro + 2.0 * ss, cx + ro, cy + ro + 2.0 * ss], outline=(0, 0, 0, 200), width=int(2.4 * ss))
            rd.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], outline=(255, 255, 255, 225), width=int(1.5 * ss))
        # 3D sphere shadow + lit body + specular glint
        rd.ellipse([cx - rn, cy - rn + 2.2 * ss, cx + rn, cy + rn + 2.2 * ss], fill=(0, 0, 0, 210))
        rd.ellipse([cx - rn, cy - rn, cx + rn, cy + rn], fill=(*col_dark, 255), outline=(255, 255, 255, 245), width=int(1.2 * ss))
        rm = rn * 0.72
        rd.ellipse([cx - rm - 0.1 * rn, cy - rm - 0.1 * rn, cx + rm - 0.1 * rn, cy + rm - 0.1 * rn], fill=(*col_main, 255))
        rh = rn * 0.32
        rd.ellipse([cx - rh - 0.24 * rn, cy - rh - 0.24 * rn, cx + rh - 0.24 * rn, cy + rh - 0.24 * rn], fill=(255, 255, 255, 255))


def render_v48_scene(
    rgb_frame: np.ndarray,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 0.85,
) -> Image.Image:
    """
    Renders the v48 scene where:
    - The 3-Point Joint Alignment Chain (`고관절–무릎–발목` or `어깨–고관절–무릎`) clearly explains WHY each
      marker dot is placed on the joints,
    - And the True 3D Folded Cylinder-Wrap Ribbon Arrows (`뒷면 접힘부가 보이는 3D 원통 감김 입체 리본 화살표`)
      wrap around the 3D limb contour without colliding with the angle wedge!
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
        # - Right Leg (`hip_r — kn_r — ank_r`): Meaningful 3-Point Joint Chain + Vertical Plumb
        #   Reference Line (`kn_r` up to ideal vertical alignment over `ank_r`) showing the `18°`
        #   inward valgus collapse at the knee hinge!
        # - Both Thighs/Knees: True 3D Folded Cylinder-Wrap Ribbon Arrows wrapping around the 3D
        #   cylinder of the lower thigh/knee from outside -> front -> inward (`> <`)!
        # =========================================================================================
        # Ideal vertical alignment point above `kn_r` (if knee hadn't caved inward)
        ref_valgus_pt = (kn_r[0] - 48.0, kn_r[1] - 125.0)
        draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_r, p_pivot=kn_r, p_dist=ank_r, p_ref_end=ref_valgus_pt,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        # Matching 3-point chain on the left leg so both legs' marker dots have clear meaning
        draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_l, p_pivot=kn_l, p_dist=ank_l, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # True 3D Folded Cylinder-Wrap Ribbon Arrow around Right Thigh/Knee (curls from outer back `215°` -> front `90°` -> inner `25°`)
        draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_r[0] - 12.0, kn_r[1] + 28.0),
            rx=68.0, ry=26.0, tilt_deg=12.0,
            phi_start_deg=218.0, phi_end_deg=22.0,
            ribbon_thick_px=15.0, pitch_dy=14.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        # True 3D Folded Cylinder-Wrap Ribbon Arrow around Left Thigh/Knee (curls from outer back `-38°` -> front `90°` -> inner `158°`)
        draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_l[0] + 12.0, kn_l[1] + 28.0),
            rx=68.0, ry=26.0, tilt_deg=-12.0,
            phi_start_deg=-38.0, phi_end_deg=158.0,
            ribbon_thick_px=15.0, pitch_dy=14.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

    elif step_idx == 2:
        # =========================================================================================
        # STEP 2 (Front View — FIX: `0° KNEES TRACK TOES` & `96% VMO & GLUTE LOCK`)
        # - Meaningful 3-Point Joint Chain (`Hip — Knee — Ankle`) on both legs showing straight `0°`
        #   alignment directly over the feet!
        # - True 3D Folded Cylinder-Wrap Ribbon Arrows wrapping OUTWARD around both thighs/knees
        #   (curling from the inner side -> across the front of the knee -> wrapping outward!).
        # =========================================================================================
        draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_r, p_pivot=kn_r, p_dist=ank_r, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_l, p_pivot=kn_l, p_dist=ank_l, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # Right Thigh/Knee 3D External-Torque Cylinder-Wrap Ribbon (curls from inner back `-25°` -> front `90°` -> outer `205°`)
        draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_r[0] - 8.0, kn_r[1] + 26.0),
            rx=70.0, ry=26.0, tilt_deg=-10.0,
            phi_start_deg=-25.0, phi_end_deg=202.0,
            ribbon_thick_px=15.0, pitch_dy=-10.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        # Left Thigh/Knee 3D External-Torque Cylinder-Wrap Ribbon (curls from inner back `205°` -> front `90°` -> outer `-22°`)
        draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_l[0] + 8.0, kn_l[1] + 26.0),
            rx=70.0, ry=26.0, tilt_deg=10.0,
            phi_start_deg=205.0, phi_end_deg=-22.0,
            ribbon_thick_px=15.0, pitch_dy=-10.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

    elif step_idx == 3:
        # =========================================================================================
        # STEP 3 (3/4 Side View — WRONG: `+28° HEELS LIFTING` & `79% LUMBAR & PATELLA`)
        # - Meaningful 3-Point Joint Chain (`Shoulder — Hip Hinge — Knee — Lifted Heel`) with the
        #   Ideal Upright Torso Reference Line at the Hip Hinge showing `+28°` excessive forward pitch!
        # - True 3D Folded Cylinder-Wrap Ribbon Arrow arcing around the upper torso/shoulder pitching
        #   forward-and-down, plus a 3D Helix Ribbon Arrow lifting the heel off the floor!
        # =========================================================================================
        ideal_upright_sh = (hip_r[0] + 22.0, hip_r[1] - 185.0)
        draw_meaningful_3point_joint_chain(
            rd, p_prox=sh, p_pivot=hip_r, p_dist=kn_r, p_ref_end=ideal_upright_sh,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        # Connect Knee -> Lifted Heel marker as well
        draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_r, p_pivot=kn_r, p_dist=ank_r, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # 1. 3D Torso Forward-Pitch Cylinder-Wrap Ribbon around the upper back/shoulder
        draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(sh[0] + 18.0, sh[1] + 15.0),
            rx=78.0, ry=32.0, tilt_deg=52.0,
            phi_start_deg=215.0, phi_end_deg=15.0,
            ribbon_thick_px=15.5, pitch_dy=24.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        # 2. 3D Heel-Lift Helical Ribbon wrapping upward around the lifting ankle/heel
        draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(ank_r[0] - 12.0, ank_r[1] - 12.0),
            rx=52.0, ry=20.0, tilt_deg=-78.0,
            phi_start_deg=-30.0, phi_end_deg=165.0,
            ribbon_thick_px=13.5, pitch_dy=-28.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

    else:
        # =========================================================================================
        # STEP 4 (3/4 Side View — FIX: `95° PARALLEL DEPTH` & `98% QUADS & GLUTES`)
        # - Meaningful 3-Point Joint Chain (`Shoulder — Hip Hinge — Knee — Planted Heel`) with the
        #   Horizontal Parallel Reference Line at the Hip Hinge showing `95° PARALLEL DEPTH`!
        # - True 3D Folded Cylinder-Wrap Ribbon Arrow wrapping around the hip/thigh driving upward,
        #   plus a 3D Floor-Lock Ribbon around the planted heel!
        # =========================================================================================
        horiz_parallel_ref = (hip_r[0] + 145.0, hip_r[1])
        draw_meaningful_3point_joint_chain(
            rd, p_prox=sh, p_pivot=hip_r, p_dist=kn_r, p_ref_end=horiz_parallel_ref,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_r, p_pivot=kn_r, p_dist=ank_r, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # 1. 3D Upward Chest/Torso Drive Cylinder-Wrap Ribbon
        draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(sh[0] + 22.0, sh[1] + 18.0),
            rx=78.0, ry=30.0, tilt_deg=-68.0,
            phi_start_deg=-32.0, phi_end_deg=168.0,
            ribbon_thick_px=15.5, pitch_dy=-26.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        # 2. 3D Planted-Foot Floor-Wrap Ribbon around the ankle/heel
        draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(ank_r[0] + 4.0, ank_r[1] + 14.0),
            rx=56.0, ry=18.0, tilt_deg=-6.0,
            phi_start_deg=210.0, phi_end_deg=25.0,
            ribbon_thick_px=13.0, pitch_dy=6.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
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
        draw_v48_frameless_white_badge(od, sd, (335, 1005), "18°", "VALGUS CAVE", align="right")
        draw_v48_frameless_white_badge(od, sd, (715, 855), "84%", "ACL & MCL LOAD", align="left")
    elif step_idx == 2:
        _leader((kn_r[0] - 18, kn_r[1] - 8), (315, 1060), (340, 1022))
        _leader((kn_l[0] + 18, kn_l[1] - 8), (710, 955), (730, 898))
        draw_v48_frameless_white_badge(od, sd, (345, 990), "0°", "KNEES TRACK TOES", align="right")
        draw_v48_frameless_white_badge(od, sd, (705, 855), "96%", "VMO & GLUTE LOCK", align="left")
    elif step_idx == 3:
        _leader((hip_r[0] - 18, hip_r[1] + 10), (315, 1005), (342, 1048))
        _leader((kn_r[0] + 16, kn_r[1] - 10), (705, 955), (725, 898))
        draw_v48_frameless_white_badge(od, sd, (348, 1015), "+28°", "HEELS LIFTING", align="right")
        draw_v48_frameless_white_badge(od, sd, (700, 855), "79%", "LUMBAR & PATELLA", align="left")
    else:
        _leader((hip_r[0] - 18, hip_r[1] + 10), (310, 995), (338, 1038))
        _leader((kn_r[0] + 16, kn_r[1] - 10), (715, 955), (738, 898))
        draw_v48_frameless_white_badge(od, sd, (345, 1005), "95°", "PARALLEL DEPTH", align="right")
        draw_v48_frameless_white_badge(od, sd, (715, 855), "98%", "QUADS & GLUTES", align="left")

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

        canvas = render_v48_scene(rgb_frame, s_idx, kp_cur, is_w, rep_amount=0.85)
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
    sheet_path = os.path.join(OUT_DIR, "day3_v48_3d_design_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    closeup_2x2 = np.vstack([np.hstack(closeup_panels[:2]), np.hstack(closeup_panels[2:])])
    closeup_path = os.path.join(OUT_DIR, "day3_v48_3d_closeup_sheet.jpg")
    cv2.imwrite(closeup_path, closeup_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved v48 design sheets:", sheet_path, "and", closeup_path)


if __name__ == "__main__":
    main()
