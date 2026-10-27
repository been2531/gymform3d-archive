#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v54_joint_anchored_master.py`)
===================================================================
1. Replaces ALL `rd.line(arc_pts, width=...)` polyline calls (which produce jagged un-joined segment notches in PIL)
   with **True Closed-Polygon Annular Arcs (`draw_joint_angle_sector_ss`)** and **Closed-Polygon Segments (`draw_smooth_segment_ss`)**
   so every single angle arc, bone axis, and curved arrow is 100% razor-smooth!
2. Unmistakable Full-Body Biomechanical Posture Contrast (`차이가 명확하게`):
   - Step 1 (`18° KNEE VALGUS`) vs Step 2 (`0° KNEES TRACK TOES`): Full-leg femur-tibia hinge brings the knees
     sharply inward (`>   <`) in Step 1 vs wide outward over the toes (`<   >`) in Step 2.
   - Step 3 (`+28° FORWARD LEAN`) vs Step 4 (`95° PARALLEL DRIVE`): True 2D rigid-hinge rotation (`+19.5°` clockwise
     around `hip_r`) tilts the entire upper body forward + `68px` upward heel lift in Step 3 vs `-7.5°` upright
     chest lock + flat planted heel in Step 4.
3. 100% Joint-Anchored Vectors (`◎───▶`):
   - Every arrow starts directly at the `◎` joint ring or spans strictly from the Plumb Reference Line into `◎`.
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
SAFE_X_RIGHT_BADGE = 785

COL_RED = (250, 46, 52)
COL_RED_LIGHT = (255, 175, 170)
COL_GREEN = (22, 226, 124)
COL_GREEN_LIGHT = (165, 255, 210)
COL_WHITE = (255, 255, 255)
COL_PLUMB = (220, 232, 248)

FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_CACHE = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(FONT_BOLD_PATH, size)
    return _FONT_CACHE[size]


_GRID_X, _GRID_Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def apply_unmistakable_posture_contrast(
    rgb: np.ndarray,
    kp: dict,
    step_idx: int,
    u: float,
) -> tuple[np.ndarray, dict]:
    kp_out = {k: (float(v[0]), float(v[1])) for k, v in kp.items()}

    if step_idx == 1:
        y_hip = 1005.0 + 25.0 * (1.0 - u)
        kp_out["hip_r"] = (386.0, y_hip)
        kp_out["hip_l"] = (694.0, y_hip)
    elif step_idx == 2:
        y_hip = 1015.0 + 25.0 * (1.0 - u)
        kp_out["hip_r"] = (368.0, y_hip)
        kp_out["hip_l"] = (712.0, y_hip)

    if u <= 0.02:
        return rgb, kp_out

    dx_map = np.zeros((H, W), dtype=np.float32)
    dy_map = np.zeros((H, W), dtype=np.float32)

    if step_idx == 1:
        inward_px = 162.0 * u
        down_px = 16.0 * u

        y_hip = kp_out["hip_r"][1]
        y_kn = 1205.0
        y_ank = kp["ank_r"][1]

        t_up = np.clip((_GRID_Y - (y_hip - 35.0)) / max(1.0, y_kn - (y_hip - 35.0)), 0.0, 1.0)
        t_dn = np.clip((y_ank - 10.0 - _GRID_Y) / max(1.0, (y_ank - 10.0) - y_kn), 0.0, 1.0)
        hinge_y = np.minimum(t_up, t_dn)
        hinge_y = 0.5 - 0.5 * np.cos(np.pi * hinge_y)

        w_x_r = np.clip((_GRID_X - 120.0) / 80.0, 0.0, 1.0) * np.clip((532.0 - _GRID_X) / 55.0, 0.0, 1.0)
        w_x_l = np.clip((_GRID_X - 548.0) / 55.0, 0.0, 1.0) * np.clip((960.0 - _GRID_X) / 80.0, 0.0, 1.0)

        w_r = hinge_y * w_x_r
        w_l = hinge_y * w_x_l

        dx_map += inward_px * w_r - inward_px * w_l
        dy_map += down_px * (w_r + w_l)

        kp_out["kn_r"] = (kp["kn_r"][0] + inward_px * 0.92, kp["kn_r"][1] + down_px * 0.85)
        kp_out["kn_l"] = (kp["kn_l"][0] - inward_px * 0.92, kp["kn_l"][1] + down_px * 0.85)

    elif step_idx == 2:
        outward_px = 18.0 * u
        y_hip = kp_out["hip_r"][1]
        y_kn = 1248.0
        y_ank = kp["ank_r"][1]
        t_up = np.clip((_GRID_Y - (y_hip - 25.0)) / max(1.0, y_kn - (y_hip - 25.0)), 0.0, 1.0)
        t_dn = np.clip((y_ank - 10.0 - _GRID_Y) / max(1.0, (y_ank - 10.0) - y_kn), 0.0, 1.0)
        hinge_y = 0.5 - 0.5 * np.cos(np.pi * np.minimum(t_up, t_dn))
        w_x_r = np.clip((_GRID_X - 110.0) / 70.0, 0.0, 1.0) * np.clip((520.0 - _GRID_X) / 60.0, 0.0, 1.0)
        w_x_l = np.clip((_GRID_X - 560.0) / 60.0, 0.0, 1.0) * np.clip((970.0 - _GRID_X) / 70.0, 0.0, 1.0)
        dx_map += -outward_px * hinge_y * w_x_r + outward_px * hinge_y * w_x_l
        kp_out["kn_r"] = (kp["kn_r"][0] - outward_px * 0.9, kp["kn_r"][1])
        kp_out["kn_l"] = (kp["kn_l"][0] + outward_px * 0.9, kp["kn_l"][1])

    elif step_idx == 3:
        sh = kp["sh"]
        hip_r = kp["hip_r"]
        kn_r = kp["kn_r"]
        ank_r = kp["ank_r"]

        theta = math.radians(19.5 * u)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        px, py = hip_r[0], hip_r[1]

        rx = _GRID_X - px
        ry = _GRID_Y - py
        src_x_rot = px + (rx * cos_t + ry * sin_t)
        src_y_rot = py + (-rx * sin_t + ry * cos_t)

        w_upper_y = np.clip((1130.0 - _GRID_Y) / 165.0, 0.0, 1.0)
        w_upper_y = 0.5 - 0.5 * np.cos(np.pi * w_upper_y)
        w_upper_x = np.clip((_GRID_X - 180.0) / 110.0, 0.0, 1.0) * np.clip((960.0 - _GRID_X) / 110.0, 0.0, 1.0)
        w_rot = w_upper_y * w_upper_x

        dx_map += (_GRID_X - src_x_rot) * w_rot
        dy_map += (_GRID_Y - src_y_rot) * w_rot

        heel_lift_dy = -68.0 * u
        knee_fwd_dx = 42.0 * u
        heel_x, heel_y = ank_r[0] - 26.0, ank_r[1] + 10.0
        heel_w = np.exp(-((_GRID_X - heel_x) ** 2) / (2.0 * 88.0 ** 2) - ((_GRID_Y - heel_y) ** 2) / (2.0 * 95.0 ** 2))
        toe_pin = np.clip((ank_r[0] + 52.0 - _GRID_X) / 60.0, 0.0, 1.0)
        heel_w *= toe_pin

        kn_w = np.exp(-((_GRID_X - kn_r[0]) ** 2) / (2.0 * 145.0 ** 2) - ((_GRID_Y - kn_r[1]) ** 2) / (2.0 * 150.0 ** 2))
        dx_map += knee_fwd_dx * kn_w
        dy_map += heel_lift_dy * heel_w

        sh_rx, sh_ry = sh[0] - px, sh[1] - py
        kp_out["sh"] = (
            px + (sh_rx * cos_t - sh_ry * sin_t) * 0.90,
            py + (sh_rx * sin_t + sh_ry * cos_t) * 0.90,
        )
        kp_out["kn_r"] = (kn_r[0] + knee_fwd_dx * 0.85, kn_r[1])
        kp_out["ank_r"] = (ank_r[0], ank_r[1] + heel_lift_dy * 0.82)

    elif step_idx == 4:
        sh = kp["sh"]
        hip_r = kp["hip_r"]
        theta = math.radians(-7.5 * u)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        px, py = hip_r[0], hip_r[1]

        rx = _GRID_X - px
        ry = _GRID_Y - py
        src_x_rot = px + (rx * cos_t + ry * sin_t)
        src_y_rot = py + (-rx * sin_t + ry * cos_t)

        w_upper_y = np.clip((1110.0 - _GRID_Y) / 160.0, 0.0, 1.0)
        w_upper_y = 0.5 - 0.5 * np.cos(np.pi * w_upper_y)
        w_upper_x = np.clip((_GRID_X - 180.0) / 110.0, 0.0, 1.0) * np.clip((950.0 - _GRID_X) / 110.0, 0.0, 1.0)
        w_rot = w_upper_y * w_upper_x

        dx_map += (_GRID_X - src_x_rot) * w_rot
        dy_map += (_GRID_Y - src_y_rot) * w_rot

        sh_rx, sh_ry = sh[0] - px, sh[1] - py
        kp_out["sh"] = (
            px + (sh_rx * cos_t - sh_ry * sin_t) * 0.92,
            py + (sh_rx * sin_t + sh_ry * cos_t) * 0.92,
        )

    map_x = (_GRID_X - dx_map).astype(np.float32)
    map_y = (_GRID_Y - dy_map).astype(np.float32)
    warped = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    out = warped.astype(np.float32)
    luma = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    body_mask = np.clip((luma - 28.0) / 55.0, 0.0, 1.0)

    if step_idx == 1:
        kn_r = kp_out["kn_r"]
        kn_l = kp_out["kn_l"]
        g_r = np.exp(-((_GRID_X - (kn_r[0] + 8.0)) ** 2) / (2.0 * 72.0 ** 2) - ((_GRID_Y - (kn_r[1] - 30.0)) ** 2) / (2.0 * 155.0 ** 2))
        g_l = np.exp(-((_GRID_X - (kn_l[0] - 8.0)) ** 2) / (2.0 * 72.0 ** 2) - ((_GRID_Y - (kn_l[1] - 30.0)) ** 2) / (2.0 * 155.0 ** 2))
        glow = np.clip((g_r + g_l) * body_mask * u * 0.62, 0.0, 0.68)[:, :, None]
        target_red = np.array([248.0, 38.0, 44.0], dtype=np.float32)[None, None, :]
        out = out * (1.0 - glow) + (target_red * (0.45 + 0.55 * (luma[:, :, None] / 255.0))) * glow
    elif step_idx == 3:
        sh = kp_out["sh"]
        hip_r = kp_out["hip_r"]
        kn_r = kp_out["kn_r"]
        mid_back = ((sh[0] + hip_r[0]) * 0.5 - 20.0, (sh[1] + hip_r[1]) * 0.5)
        g_back = np.exp(-((_GRID_X - mid_back[0]) ** 2) / (2.0 * 92.0 ** 2) - ((_GRID_Y - mid_back[1]) ** 2) / (2.0 * 145.0 ** 2))
        g_knee = np.exp(-((_GRID_X - kn_r[0]) ** 2) / (2.0 * 68.0 ** 2) - ((_GRID_Y - kn_r[1]) ** 2) / (2.0 * 100.0 ** 2))
        glow = np.clip((g_back + g_knee * 0.90) * body_mask * u * 0.58, 0.0, 0.65)[:, :, None]
        target_red = np.array([248.0, 38.0, 44.0], dtype=np.float32)[None, None, :]
        out = out * (1.0 - glow) + (target_red * (0.45 + 0.55 * (luma[:, :, None] / 255.0))) * glow

    return np.clip(out, 0, 255).astype(np.uint8), kp_out


def draw_smooth_segment_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    half_w_ss: float,
    rgba: tuple[int, int, int, int],
):
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1.0:
        return
    nx, ny = -dy / L, dx / L
    poly = [
        (x0 + nx * half_w_ss, y0 + ny * half_w_ss),
        (x1 + nx * half_w_ss, y1 + ny * half_w_ss),
        (x1 - nx * half_w_ss, y1 - ny * half_w_ss),
        (x0 - nx * half_w_ss, y0 - ny * half_w_ss),
    ]
    rd.polygon(poly, fill=rgba)


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

    draw_smooth_segment_ss(rd, pa, pb, 3.4 * ss, (0, 0, 0, 220))
    draw_smooth_segment_ss(rd, pa, pb, 2.0 * ss, (*col_main, 252))
    draw_smooth_segment_ss(rd, pa, pb, 0.75 * ss, (*col_light, 220))


def draw_plumb_reference_line_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    ss: int = 4,
):
    draw_smooth_segment_ss(rd, p0_ss, p1_ss, 2.3 * ss, (0, 0, 0, 170))
    draw_smooth_segment_ss(rd, p0_ss, p1_ss, 1.2 * ss, (*COL_PLUMB, 200))


def draw_joint_angle_sector_ss(
    rd: ImageDraw.ImageDraw,
    vertex_ss: tuple[float, float],
    ray1_pt_ss: tuple[float, float],
    ray2_pt_ss: tuple[float, float],
    radius_1x: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    show_arrow_at_ray2: bool = False,
):
    vx, vy = vertex_ss
    r = radius_1x * ss
    a1 = math.atan2(ray1_pt_ss[1] - vy, ray1_pt_ss[0] - vx)
    a2 = math.atan2(ray2_pt_ss[1] - vy, ray2_pt_ss[0] - vx)
    diff = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(diff) < 0.04:
        return

    steps = 64
    poly = [(vx, vy)]
    for i in range(steps + 1):
        a = a1 + diff * (i / float(steps))
        poly.append((vx + math.cos(a) * r, vy + math.sin(a) * r))
    rd.polygon(poly, fill=(*col_main, 64))

    direction = 1.0 if diff >= 0.0 else -1.0
    head_len = 22.0 * ss if show_arrow_at_ray2 else 0.0
    head_half_w = 12.5 * ss
    d_theta = (head_len / max(36.0 * ss, r)) * direction
    a_end = a1 + diff - d_theta

    for pad, rgba in [(1.8 * ss, (0, 0, 0, 225)), (0.0, (*col_main, 255))]:
        hw_arc = 2.2 * ss + pad
        outer_pts = []
        inner_pts = []
        for i in range(steps + 1):
            a = a1 + (a_end - a1) * (i / float(steps))
            ca, sa = math.cos(a), math.sin(a)
            outer_pts.append((vx + ca * (r + hw_arc), vy + sa * (r + hw_arc)))
            inner_pts.append((vx + ca * (r - hw_arc), vy + sa * (r - hw_arc)))

        if show_arrow_at_ray2:
            ux = -math.sin(a_end) * direction
            uy = math.cos(a_end) * direction
            rx_n, ry_n = math.cos(a_end), math.sin(a_end)
            nbx = vx + rx_n * r
            nby = vy + ry_n * r
            hw_h = head_half_w + pad * 1.15
            p_outer_wing = (nbx - ux * (2.5 * ss) + rx_n * hw_h, nby - uy * (2.5 * ss) + ry_n * hw_h)
            p_inner_wing = (nbx - ux * (2.5 * ss) - rx_n * hw_h, nby - uy * (2.5 * ss) - ry_n * hw_h)
            tx = vx + math.cos(a1 + diff) * r + ux * pad * 1.2
            ty = vy + math.sin(a1 + diff) * r + uy * pad * 1.2
            ring_poly = outer_pts + [p_outer_wing, (tx, ty), p_inner_wing] + inner_pts[::-1]
        else:
            ring_poly = outer_pts + inner_pts[::-1]
        rd.polygon(ring_poly, fill=rgba)

    hw_spec = 0.75 * ss
    spec_out, spec_in = [], []
    for i in range(steps + 1):
        a = a1 + (a_end - a1) * (i / float(steps))
        ca, sa = math.cos(a), math.sin(a)
        spec_out.append((vx + ca * (r + hw_spec), vy + sa * (r + hw_spec)))
        spec_in.append((vx + ca * (r - hw_spec), vy + sa * (r - hw_spec)))
    rd.polygon(spec_out + spec_in[::-1], fill=(*col_light, 225))


def draw_joint_anchored_vector_arrow_ss(
    rd: ImageDraw.ImageDraw,
    joint_pt_ss: tuple[float, float],
    tip_pt_ss: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    start_offset_1x: float = 11.5,
):
    jx, jy = joint_pt_ss
    x1, y1 = tip_pt_ss
    dx, dy = x1 - jx, y1 - jy
    L_full = math.hypot(dx, dy)
    off = start_offset_1x * ss
    if L_full <= off + 22.0 * ss:
        return
    ux, uy = dx / L_full, dy / L_full
    nx, ny = -uy, ux

    x0 = jx + ux * off
    y0 = jy + uy * off

    head_len = 24.0 * ss
    head_half_w = 13.5 * ss
    neck_x = x1 - ux * head_len
    neck_y = y1 - uy * head_len

    w_tail = 3.0 * ss
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

    draw_smooth_segment_ss(rd, (x0, y0), (neck_x, neck_y), 0.9 * ss, (*col_light, 230))


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
    rd.ellipse([cx - r - 2 * ss, cy - r - 2 * ss, cx + r + 2 * ss, cy + r + 2 * ss], fill=(0, 0, 0, 200))
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

    bx = max(SAFE_X_LEFT, x - bw) if align == "right" else max(SAFE_X_LEFT, x)
    sx = max(SAFE_X_LEFT, x - sw) if align == "right" else max(SAFE_X_LEFT, x)
    sy = y + 70

    for dx_s, dy_s in [(-3, 0), (3, 0), (0, -3), (0, 3), (0, 5)]:
        sd.text((bx + dx_s, y + dy_s), big_str, font=f_big, fill=(0, 0, 0, 235))
        sd.text((sx + dx_s, sy + dy_s), sub_str, font=f_sub, fill=(0, 0, 0, 235))

    od.text((bx, y), big_str, font=f_big, fill=COL_WHITE)
    od.text((sx, sy), sub_str, font=f_sub, fill=COL_WHITE)


def render_v54_callouts(
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
        plumb_r = (ank_r[0], kn_r[1])
        plumb_l = (ank_l[0], kn_l[1])
        draw_plumb_reference_line_ss(rd, S(ank_r), S(plumb_r), SS)
        draw_plumb_reference_line_ss(rd, S(ank_l), S(plumb_l), SS)

        draw_joint_angle_sector_ss(rd, S(ank_r), S(plumb_r), S(kn_r), 130.0, col_main, col_light, SS, show_arrow_at_ray2=False)
        draw_joint_angle_sector_ss(rd, S(ank_l), S(plumb_l), S(kn_l), 130.0, col_main, col_light, SS, show_arrow_at_ray2=False)

        draw_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(hip_l), S(kn_l), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_l), S(ank_l), col_main, col_light, SS)

        draw_joint_anchored_vector_arrow_ss(
            rd,
            joint_pt_ss=S(kn_r),
            tip_pt_ss=S((kn_r[0] + 92.0, kn_r[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_joint_anchored_vector_arrow_ss(
            rd,
            joint_pt_ss=S(kn_l),
            tip_pt_ss=S((kn_l[0] - 92.0, kn_l[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.0, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (SAFE_X_LEFT, 955), "18°", "VALGUS CAVE", align="left")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "84%", "ACL & MCL LOAD", align="left")

    elif step_idx == 2:
        draw_joint_angle_sector_ss(rd, S(kn_r), S(hip_r), S(ank_r), 54.0, col_main, col_light, SS, show_arrow_at_ray2=False)
        draw_joint_angle_sector_ss(rd, S(kn_l), S(hip_l), S(ank_l), 54.0, col_main, col_light, SS, show_arrow_at_ray2=False)

        draw_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(hip_l), S(kn_l), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_l), S(ank_l), col_main, col_light, SS)

        draw_joint_anchored_vector_arrow_ss(
            rd,
            joint_pt_ss=S(kn_r),
            tip_pt_ss=S((kn_r[0] - 115.0, kn_r[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_joint_anchored_vector_arrow_ss(
            rd,
            joint_pt_ss=S(kn_l),
            tip_pt_ss=S((kn_l[0] + 115.0, kn_l[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.0, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (SAFE_X_LEFT, 955), "0°", "KNEES TRACK TOES", align="left")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "96%", "VMO & GLUTE LOCK", align="left")

    elif step_idx == 3:
        plumb_top = (hip_r[0], sh[1] - 40.0)
        draw_plumb_reference_line_ss(rd, S(hip_r), S(plumb_top), SS)

        spine_len = math.hypot(sh[0] - hip_r[0], sh[1] - hip_r[1])
        draw_joint_angle_sector_ss(
            rd,
            vertex_ss=S(hip_r),
            ray1_pt_ss=S(plumb_top),
            ray2_pt_ss=S(sh),
            radius_1x=spine_len * 0.72,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrow_at_ray2=True,
        )

        draw_bone_axis_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        floor_y = ank_r[1] + 72.0
        draw_plumb_reference_line_ss(rd, S((ank_r[0] - 90.0, floor_y)), S((ank_r[0] + 95.0, floor_y)), SS)
        draw_joint_anchored_vector_arrow_ss(
            rd,
            joint_pt_ss=S(ank_r),
            tip_pt_ss=S((ank_r[0], ank_r[1] - 125.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.5, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (SAFE_X_LEFT, 985), "+28°", "HEELS LIFTING", align="left")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "79%", "LUMBAR & PATELLA", align="left")

    else:
        draw_joint_angle_sector_ss(
            rd,
            vertex_ss=S(hip_r),
            ray1_pt_ss=S(sh),
            ray2_pt_ss=S(kn_r),
            radius_1x=95.0,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrow_at_ray2=False,
        )

        draw_bone_axis_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        draw_joint_anchored_vector_arrow_ss(
            rd,
            joint_pt_ss=S(sh),
            tip_pt_ss=S((sh[0], sh[1] - 135.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        floor_y = ank_r[1] + 58.0
        draw_bone_axis_ss(
            rd,
            S((ank_r[0] - 90.0, floor_y)),
            S((ank_r[0] + 95.0, floor_y)),
            col_main,
            col_light,
            SS,
            trim_ends_1x=0.0,
        )
        draw_joint_anchored_vector_arrow_ss(
            rd,
            joint_pt_ss=S(ank_r),
            tip_pt_ss=S((ank_r[0], floor_y - 4.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            start_offset_1x=8.0,
        )

        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.5, col_main, col_light, SS, is_primary=is_p)

        draw_pure_white_metric_callout(od, sd, (SAFE_X_LEFT, 985), "95°", "PARALLEL DEPTH", align="left")
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

        rgb_frame, kp_cur = apply_unmistakable_posture_contrast(raw_rgb, raw_kp, step_idx=s_idx, u=1.0)
        canvas = Image.fromarray(rgb_frame).convert("RGBA")

        render_v54_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)

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
        crops.append(out_bgr[520:1620, 80:1000])

    sheet_2x2 = np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])])
    sheet_path = os.path.join(OUT_DIR, "day3_v54_difference_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    crop_2x2 = np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])])
    crop_path = os.path.join(OUT_DIR, "day3_v54_closeup_sheet.jpg")
    cv2.imwrite(crop_path, crop_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved:", sheet_path, "and", crop_path)


if __name__ == "__main__":
    main()
