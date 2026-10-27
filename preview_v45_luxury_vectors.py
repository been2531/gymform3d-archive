#!/usr/bin/env python3
"""
Day 3 v45 — Broadcast-Luxury SDF Arrows, Joint Nodes & Angle Gauges Design Mockup (`시안`)
==========================================================================================
Renders ultra-clean, cinema-grade (Apple Fitness+ / @demicstory / Nike Training Club 3D HUD style)
SDF (Signed Distance Field) vector graphics over the 4 full-body 3D Squat frames:
1. Luxury Optical Joint Nodes (`점`):
   - Multi-layer radial bloom + precision 1.5px micro-ring + neon corona + hot specular white core.
2. Luxury Gradient Angle Sector & Arc (`각도 아크`):
   - Subtle translucent angular sector wedge (`14%` opacity) between the reference axis and limb axis,
   - Whisper-thin reference ray + progressively thickening/brightening glowing arc + sleek swept chevron tip.
3. Luxury Body-Fitted Tapered Bezier Arrows (`화살표`):
   - Smoothly curved along the 3D anatomy, tapering from `0%` opacity / `2px` tail to `100%` hot white-cored
     neon beam with aerodynamic dual swept-wing chevrons (`❯❯`).
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

SAFE_X_LEFT = 96
SAFE_Y_TOP = 258
SAFE_X_RIGHT_BADGE = 805

COL_RED = (235, 42, 48)
COL_RED_SOFT = (255, 148, 142)
COL_GREEN = (16, 205, 110)
COL_GREEN_SOFT = (152, 250, 198)


def _blend_rgba_onto_canvas(canvas_rgba: np.ndarray, layer_rgb: np.ndarray, layer_alpha: np.ndarray):
    """In-place straight-alpha composite of a float32 RGB layer (0..255) and float32 alpha (0..1) onto uint8 RGBA canvas."""
    a = np.clip(layer_alpha, 0.0, 1.0)[:, :, None]
    base_rgb = canvas_rgba[:, :, :3].astype(np.float32)
    out_rgb = base_rgb * (1.0 - a) + np.clip(layer_rgb, 0.0, 255.0) * a
    canvas_rgba[:, :, :3] = np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)


def draw_sdf_joint_node(
    rgb_buf: np.ndarray,
    pt: tuple[float, float],
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    r_core: float = 5.2,
    r_ring: float = 15.5,
    r_outer: float = 24.0,
):
    """
    Renders a jewel-like 3D optical motion-capture joint node using analytical per-pixel SDF:
    - Soft dark contrast halo so it reads over bright X-Ray / muscle glow,
    - Ambient colored bloom,
    - Razor-sharp 1.6px precision ring at `r_ring` with 4 subtle cardinal gap ticks,
    - Outer subtle segmented arc at `r_outer`,
    - Hot white specular center (`r_core`) with neon colored corona.
    """
    cx, cy = float(pt[0]), float(pt[1])
    pad = int(r_outer + 28)
    x0, x1 = max(0, int(cx - pad)), min(W, int(cx + pad + 1))
    y0, y1 = max(0, int(cy - pad)), min(H, int(cy + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    dist = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)  # -pi..pi

    roi = rgb_buf[y0:y1, x0:x1].astype(np.float32)
    c_main = np.array(col_main, dtype=np.float32)
    c_soft = np.array(col_soft, dtype=np.float32)
    c_white = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    # 1. Subtle dark contrast cushion behind node (`r < r_ring + 4`)
    dark_a = 0.42 * np.exp(-0.5 * (dist / (r_ring * 0.95)) ** 2)
    roi = roi * (1.0 - dark_a[:, :, None])

    # 2. Colored ambient glow
    bloom_a = 0.48 * np.exp(-0.5 * (dist / (r_ring * 1.15)) ** 2)
    roi = roi * (1.0 - bloom_a[:, :, None]) + c_main * bloom_a[:, :, None]

    # 3. Inner precision ring (`r = r_ring`, thickness `1.7px`) with 4 micro-gaps at 45 deg
    ring_sdf = np.abs(dist - r_ring)
    gap_mask = np.clip((np.abs(np.sin(2.0 * ang)) - 0.16) / 0.12, 0.0, 1.0)
    ring_a = np.clip(1.0 - ring_sdf / 1.65, 0.0, 1.0) * gap_mask * 0.92
    ring_col = c_soft * 0.55 + c_white * 0.45
    roi = roi * (1.0 - ring_a[:, :, None]) + ring_col * ring_a[:, :, None]

    # 4. Outer whisper-thin bracket arcs (`r = r_outer`, thickness `1.2px`) on left/right sides
    outer_sdf = np.abs(dist - r_outer)
    bracket_mask = np.clip((np.abs(np.cos(ang)) - 0.55) / 0.15, 0.0, 1.0)
    outer_a = np.clip(1.0 - outer_sdf / 1.35, 0.0, 1.0) * bracket_mask * 0.52
    roi = roi * (1.0 - outer_a[:, :, None]) + c_soft * outer_a[:, :, None]

    # 5. Neon corona around center dot
    corona_a = 0.85 * np.exp(-0.5 * (dist / (r_core * 1.45)) ** 2)
    roi = roi * (1.0 - corona_a[:, :, None]) + c_main * corona_a[:, :, None]

    # 6. Crisp hot-white specular core dot
    core_a = np.clip(1.0 - (dist - (r_core - 0.8)) / 1.4, 0.0, 1.0)
    roi = roi * (1.0 - core_a[:, :, None]) + c_white * core_a[:, :, None]

    rgb_buf[y0:y1, x0:x1] = np.clip(roi, 0.0, 255.0).astype(np.uint8)


def _eval_bezier(p0, p1, p2, n=160):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    pts = ((1.0 - t) ** 2) * p0 + 2.0 * (1.0 - t) * t * p1 + (t ** 2) * p2
    return pts


def draw_sdf_curved_arrow(
    rgb_buf: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    w_tail: float = 1.6,
    w_head: float = 6.2,
    head_len: float = 24.0,
    head_span: float = 16.5,
    double_chevron: bool = True,
):
    """
    Renders a luxury `@demicstory` / Apple Fitness+ style tapered curved arrow using per-pixel SDF:
    - Smooth quadratic Bezier trajectory `p0 -> p1 -> p2`,
    - Progressive opacity & width taper from `u=0` (whisper-thin fade-in tail) to `u=1` (bright head),
    - Hot white core spine inside a glowing colored beam,
    - Swept-wing aerodynamic chevron head (`❯`) at `p2` + optional ghost chevron (`❯❯`) behind it!
    """
    pts = _eval_bezier(p0, p1, p2, n=140)
    # Trim shaft slightly before tip so the chevron head forms the crisp apex
    shaft_pts = pts[:-10]

    all_x = pts[:, 0]
    all_y = pts[:, 1]
    pad = int(max(head_len, head_span) + 26)
    x0, x1 = max(0, int(all_x.min() - pad)), min(W, int(all_x.max() + pad + 1))
    y0, y1 = max(0, int(all_y.min() - pad)), min(H, int(all_y.max() + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    px = xx[:, :, None]
    py = yy[:, :, None]

    # Distance to shaft polyline
    sx = shaft_pts[:, 0][None, None, :]
    sy = shaft_pts[:, 1][None, None, :]
    d2 = (px - sx) ** 2 + (py - sy) ** 2
    idx_min = np.argmin(d2, axis=2)
    d_shaft = np.sqrt(np.min(d2, axis=2))
    u = idx_min.astype(np.float32) / float(len(shaft_pts) - 1)  # 0..1 along shaft

    # Width & alpha taper along shaft (`u=0 -> 1`)
    w_u = w_tail + (w_head - w_tail) * (u ** 1.15)
    alpha_taper = np.clip((u ** 0.75) * 1.15, 0.08, 1.0)

    # Tangent at tip `p2`
    tx = float(pts[-1, 0] - pts[-8, 0])
    ty = float(pts[-1, 1] - pts[-8, 1])
    tnorm = math.hypot(tx, ty) + 1e-6
    ux, uy = tx / tnorm, ty / tnorm
    nx, ny = -uy, ux

    def _chevron_sdf(tip_xy: tuple[float, float], scale: float = 1.0):
        # Two swept wings from `tip_xy` back to `tip_xy - u*L +- n*S`
        L = head_len * scale
        S = head_span * scale
        cx_t, cy_t = tip_xy
        w1 = (cx_t - ux * L + nx * S, cy_t - uy * L + ny * S)
        w2 = (cx_t - ux * L - nx * S, cy_t - uy * L - ny * S)

        def _seg_dist(ax, ay, bx, by):
            vx, vy = bx - ax, by - ay
            l2 = vx * vx + vy * vy + 1e-6
            t_proj = np.clip(((xx - ax) * vx + (yy - ay) * vy) / l2, 0.0, 1.0)
            proj_x = ax + t_proj * vx
            proj_y = ay + t_proj * vy
            return np.hypot(xx - proj_x, yy - proj_y), t_proj

        d_w1, t_w1 = _seg_dist(cx_t, cy_t, w1[0], w1[1])
        d_w2, t_w2 = _seg_dist(cx_t, cy_t, w2[0], w2[1])
        d_c = np.minimum(d_w1, d_w2)
        t_c = np.where(d_w1 < d_w2, t_w1, t_w2)
        # Wing thickness tapers from apex (t=0, thick) to wingtip (t=1, sharp)
        w_wing = (w_head * 0.92 * scale) * (1.0 - 0.55 * t_c)
        return d_c, w_wing

    d_chev1, w_chev1 = _chevron_sdf((float(pts[-1, 0]), float(pts[-1, 1])), scale=1.0)

    roi = rgb_buf[y0:y1, x0:x1].astype(np.float32)
    c_main = np.array(col_main, dtype=np.float32)
    c_soft = np.array(col_soft, dtype=np.float32)
    c_white = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    # Combine SDFs for dark shadow cushion & outer glow
    d_comb = np.minimum(d_shaft, d_chev1)
    if double_chevron:
        ghost_tip = (float(pts[-1, 0]) - ux * (head_len * 0.72), float(pts[-1, 1]) - uy * (head_len * 0.72))
        d_chev2, w_chev2 = _chevron_sdf(ghost_tip, scale=0.82)
        d_comb = np.minimum(d_comb, d_chev2)

    # 1. Soft dark contrast shadow (`sigma = 5.5px`) so arrow pops cleanly off any background
    shadow_a = 0.48 * np.exp(-0.5 * (d_comb / 6.0) ** 2) * (0.35 + 0.65 * alpha_taper)
    roi = roi * (1.0 - shadow_a[:, :, None])

    # 2. Colored neon outer bloom (`sigma = 8.5px`)
    bloom_a = 0.46 * np.exp(-0.5 * (d_comb / 8.5) ** 2) * alpha_taper
    roi = roi * (1.0 - bloom_a[:, :, None]) + c_main * bloom_a[:, :, None]

    # 3. Shaft body (`d_shaft <= w_u`) + hot white inner core (`d_shaft <= w_u * 0.42`)
    shaft_body_a = np.clip(1.0 - (d_shaft - (w_u - 0.7)) / 1.4, 0.0, 1.0) * alpha_taper
    shaft_core_a = np.clip(1.0 - (d_shaft - (w_u * 0.38)) / 1.2, 0.0, 1.0) * (u ** 1.1) * 0.88
    shaft_col = c_main * (1.0 - 0.55 * (u[:, :, None] ** 1.2)) + c_soft * (0.55 * (u[:, :, None] ** 1.2))
    roi = roi * (1.0 - shaft_body_a[:, :, None]) + shaft_col * shaft_body_a[:, :, None]
    roi = roi * (1.0 - shaft_core_a[:, :, None]) + c_white * shaft_core_a[:, :, None]

    # 4. Secondary trailing ghost chevron (`❯`, 60% opacity)
    if double_chevron:
        chev2_body_a = np.clip(1.0 - (d_chev2 - (w_chev2 - 0.6)) / 1.35, 0.0, 1.0) * 0.62
        chev2_core_a = np.clip(1.0 - (d_chev2 - (w_chev2 * 0.36)) / 1.15, 0.0, 1.0) * 0.48
        roi = roi * (1.0 - chev2_body_a[:, :, None]) + c_soft * chev2_body_a[:, :, None]
        roi = roi * (1.0 - chev2_core_a[:, :, None]) + c_white * chev2_core_a[:, :, None]

    # 5. Primary swept-wing chevron head (`❯`, 100% crisp with hot white core)
    chev1_body_a = np.clip(1.0 - (d_chev1 - (w_chev1 - 0.6)) / 1.35, 0.0, 1.0) * 0.98
    chev1_core_a = np.clip(1.0 - (d_chev1 - (w_chev1 * 0.42)) / 1.15, 0.0, 1.0) * 0.92
    roi = roi * (1.0 - chev1_body_a[:, :, None]) + c_soft * chev1_body_a[:, :, None]
    roi = roi * (1.0 - chev1_core_a[:, :, None]) + c_white * chev1_core_a[:, :, None]

    rgb_buf[y0:y1, x0:x1] = np.clip(roi, 0.0, 255.0).astype(np.uint8)


def draw_sdf_angle_gauge(
    rgb_buf: np.ndarray,
    center_pt: tuple[float, float],
    radius: float,
    deg_start: float,
    deg_end: float,
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ccw: bool = False,
):
    """
    Renders a broadcast-luxury biomechanical angle gauge around `center_pt`:
    - Translucent radial-angular gradient sector wedge (`0 -> 20%` opacity) inside the angle span,
    - Crisp 1.4px reference ray at `deg_start` and active limb ray at `deg_end`,
    - Tapered glowing arc (`2.0px -> 5.2px` with hot white core) from `deg_start` to `deg_end`,
    - Sleek tangential chevron arrowhead at `deg_end`!
    """
    cx, cy = float(center_pt[0]), float(center_pt[1])
    pad = int(radius + 36)
    x0, x1 = max(0, int(cx - pad)), min(W, int(cx + pad + 1))
    y0, y1 = max(0, int(cy - pad)), min(H, int(cy + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    dist = np.hypot(dx, dy)
    ang_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

    s_deg = deg_start % 360.0
    e_deg = deg_end % 360.0
    if not ccw:
        span = (e_deg - s_deg) % 360.0
        rel = (ang_deg - s_deg) % 360.0
    else:
        span = (s_deg - e_deg) % 360.0
        rel = (s_deg - ang_deg) % 360.0

    in_span = (rel <= span).astype(np.float32)
    u_ang = np.clip(rel / max(1.0, span), 0.0, 1.0) * in_span

    roi = rgb_buf[y0:y1, x0:x1].astype(np.float32)
    c_main = np.array(col_main, dtype=np.float32)
    c_soft = np.array(col_soft, dtype=np.float32)
    c_white = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    # 1. Subtle translucent gradient sector fill (`r in 18..radius`)
    r_norm = np.clip((dist - 16.0) / max(1.0, radius - 16.0), 0.0, 1.0)
    in_rad = ((dist >= 16.0) & (dist <= radius)).astype(np.float32)
    wedge_a = in_span * in_rad * (0.04 + 0.18 * (r_norm ** 1.3) * (0.35 + 0.65 * u_ang))
    roi = roi * (1.0 - wedge_a[:, :, None]) + c_main * wedge_a[:, :, None]

    # 2. Two crisp boundary rays (`deg_start` and `deg_end`) from `r=18` to `r=radius+8`
    for r_deg, ray_op in [(s_deg, 0.52), (e_deg, 0.82)]:
        rad = math.radians(r_deg)
        rx0, ry0 = cx + 18.0 * math.cos(rad), cy + 18.0 * math.sin(rad)
        rx1, ry1 = cx + (radius + 8.0) * math.cos(rad), cy + (radius + 8.0) * math.sin(rad)
        vx, vy = rx1 - rx0, ry1 - ry0
        l2 = vx * vx + vy * vy + 1e-6
        tp = np.clip(((xx - rx0) * vx + (yy - ry0) * vy) / l2, 0.0, 1.0)
        d_ray = np.hypot(xx - (rx0 + tp * vx), yy - (ry0 + tp * vy))
        ray_a = np.clip(1.0 - d_ray / 1.45, 0.0, 1.0) * ray_op
        roi = roi * (1.0 - ray_a[:, :, None]) + c_soft * ray_a[:, :, None]

    # 3. Tapered glowing arc at `dist == radius` inside `in_span`
    d_arc = np.abs(dist - radius)
    # Soft angular feather at ends
    ang_dist_px = np.minimum(rel, np.abs(span - rel)) * (math.pi / 180.0) * radius
    arc_gate = np.where(in_span > 0.5, 1.0, np.clip(1.0 - (np.minimum(np.abs(ang_deg - s_deg), np.abs(ang_deg - e_deg)) * (math.pi / 180.0) * radius) / 3.0, 0.0, 1.0))

    w_arc = 1.8 + 2.8 * (u_ang ** 1.1)
    arc_shadow = 0.45 * np.exp(-0.5 * (d_arc / 5.0) ** 2) * in_span
    roi = roi * (1.0 - arc_shadow[:, :, None])

    arc_bloom = 0.48 * np.exp(-0.5 * (d_arc / 7.5) ** 2) * in_span * (0.35 + 0.65 * u_ang)
    roi = roi * (1.0 - arc_bloom[:, :, None]) + c_main * arc_bloom[:, :, None]

    arc_body = np.clip(1.0 - (d_arc - (w_arc - 0.6)) / 1.35, 0.0, 1.0) * in_span * (0.45 + 0.55 * u_ang)
    arc_core = np.clip(1.0 - (d_arc - (w_arc * 0.38)) / 1.15, 0.0, 1.0) * in_span * (u_ang ** 1.2) * 0.88
    roi = roi * (1.0 - arc_body[:, :, None]) + c_soft * arc_body[:, :, None]
    roi = roi * (1.0 - arc_core[:, :, None]) + c_white * arc_core[:, :, None]

    rgb_buf[y0:y1, x0:x1] = np.clip(roi, 0.0, 255.0).astype(np.uint8)

    # 4. Tangential micro-chevron at arc tip (`deg_end`)
    rad_e = math.radians(e_deg)
    tip_x = cx + radius * math.cos(rad_e)
    tip_y = cy + radius * math.sin(rad_e)
    tan_deg = (e_deg - 90.0) if ccw else (e_deg + 90.0)
    tan_rad = math.radians(tan_deg)
    p0_t = (tip_x - 18.0 * math.cos(tan_rad), tip_y - 18.0 * math.sin(tan_rad))
    p1_t = (tip_x - 9.0 * math.cos(tan_rad), tip_y - 9.0 * math.sin(tan_rad))
    draw_sdf_curved_arrow(
        rgb_buf, p0_t, p1_t, (tip_x, tip_y),
        col_main, col_soft,
        w_tail=2.0, w_head=4.2, head_len=16.0, head_span=11.5, double_chevron=False
    )


def draw_sdf_leader_pin(
    rgb_buf: np.ndarray,
    p_joint: tuple[float, float],
    p_elbow: tuple[float, float],
    p_badge: tuple[float, float],
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
):
    """
    Draws a razor-sharp 1.3px architectural callout leader line (`p_joint -> p_elbow -> p_badge`)
    with a glowing 3.2px anchor dot at `p_badge`.
    """
    pts = np.vstack([
        np.linspace(p_joint, p_elbow, 45, dtype=np.float32),
        np.linspace(p_elbow, p_badge, 45, dtype=np.float32),
    ])
    pad = 16
    x0, x1 = max(0, int(pts[:, 0].min() - pad)), min(W, int(pts[:, 0].max() + pad + 1))
    y0, y1 = max(0, int(pts[:, 1].min() - pad)), min(H, int(pts[:, 1].max() + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = (xx[:, :, None] - pts[:, 0][None, None, :]) ** 2 + (yy[:, :, None] - pts[:, 1][None, None, :]) ** 2
    d_line = np.sqrt(np.min(d2, axis=2))
    u = np.argmin(d2, axis=2).astype(np.float32) / float(len(pts) - 1)

    roi = rgb_buf[y0:y1, x0:x1].astype(np.float32)
    c_soft = np.array(col_soft, dtype=np.float32)
    c_white = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    line_a = np.clip(1.0 - d_line / 1.45, 0.0, 1.0) * (0.25 + 0.55 * u)
    roi = roi * (1.0 - line_a[:, :, None]) + c_soft * line_a[:, :, None]

    # Small glowing terminal dot at `p_badge`
    d_dot = np.hypot(xx - p_badge[0], yy - p_badge[1])
    dot_glow = 0.55 * np.exp(-0.5 * (d_dot / 5.5) ** 2)
    dot_core = np.clip(1.0 - (d_dot - 2.2) / 1.2, 0.0, 1.0)
    roi = roi * (1.0 - dot_glow[:, :, None]) + np.array(col_main, dtype=np.float32) * dot_glow[:, :, None]
    roi = roi * (1.0 - dot_core[:, :, None]) + c_white * dot_core[:, :, None]

    rgb_buf[y0:y1, x0:x1] = np.clip(roi, 0.0, 255.0).astype(np.uint8)



F_V45_NUM = v30.f_barlow(56, extrabold=True)
F_V45_SUB = v30.f_pretendard(21, "extrabold")

def draw_v45_white_callout_badge(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    num_str: str,
    tag_str: str,
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    align: str = "right",
):
    """
    Renders a high-contrast broadcast callout badge:
    - 56pt extrabold Pure White number (`18°`, `0°`, `+28°`, `95°`, `84%`, etc.)
    - Sleek 3px colored accent underline bar
    - Larger 21pt extrabold PURE WHITE (`#FFFFFF`) sub-label (`VALGUS CAVE`, `KNEES TRACK TOES`, etc.)
      with deep dark shadow cushion for instant readability on mobile Reels.
    """
    x, y = xy
    nb = od.textbbox((0, 0), num_str, font=F_V45_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), tag_str, font=F_V45_SUB)
    tw = tb[2] - tb[0]

    if align == "right":
        nx = x - nw
        tx = x - tw
        bar_x0, bar_x1 = x - min(max(nw, tw), 110), x
    elif align == "center":
        nx = x - nw // 2
        tx = x - tw // 2
        bar_x0, bar_x1 = x - 45, x + 45
    else:
        nx = x
        tx = x
        bar_x0, bar_x1 = x, x + min(max(nw, tw), 110)

    # Deep multi-pass shadow behind number and pure-white sub-label
    for dx_s, dy_s in [(0, 3), (0, -2), (3, 0), (-3, 0), (2, 2)]:
        sd.text((nx + dx_s, y + dy_s), num_str, font=F_V45_NUM, fill=(0, 0, 0, 235))
        sd.text((tx + dx_s, y + 62 + dy_s), tag_str, font=F_V45_SUB, fill=(0, 0, 0, 245))

    # Crisp Pure White Number
    od.text((nx, y), num_str, font=F_V45_NUM, fill=(255, 255, 255, 255))

    # Sleek 3px colored accent bar between number and sub-label
    sd.rounded_rectangle([bar_x0 - 1, y + 55, bar_x1 + 1, y + 59], radius=2, fill=(0, 0, 0, 220))
    od.rounded_rectangle([bar_x0, y + 55, bar_x1, y + 58], radius=2, fill=(*col_main, 255))

    # Crisp Larger (21pt) PURE WHITE Sub-Label
    od.text((tx, y + 62), tag_str, font=F_V45_SUB, fill=(255, 255, 255, 255))


def render_v45_luxury_callouts_and_vectors(
    rgb_frame: np.ndarray,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 0.85,
) -> Image.Image:
    """
    Renders the complete v45 Broadcast-Luxury SDF graphic suite on `rgb_frame`:
    - SDF Optical Joint Nodes (`점`),
    - SDF Gradient Angle Sector & Arc (`각도 아크`),
    - SDF Body-Fitted Curved Tapered Arrows (`화살표`),
    - Architectural Leader Pins + Exterior Safe-Zone `°` & `%` Badges.
    """
    buf = rgb_frame.copy()
    col_main = COL_RED if is_wrong else COL_GREEN
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    sh = kp["sh"]
    hip_r = kp["hip_r"]
    hip_l = kp["hip_l"]
    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    ank_r = kp["ank_r"]
    ank_l = kp["ank_l"]

    if step_idx == 1:
        # STEP 1 (Front View — WRONG: `18° VALGUS CAVE` & `84% ACL & MCL LOAD`)
        # 1. Angle gauge on right knee showing inward valgus tilt
        draw_sdf_angle_gauge(
            buf, center_pt=kn_r, radius=92.0,
            deg_start=275.0, deg_end=338.0 + 14.0 * rep_amount,
            col_main=col_main, col_soft=col_soft, ccw=False
        )

        # 2. Body-fitted curved inward collapse arrows (`╭──❯❯` and `❮❮──╮`) hugging both knees
        shift = 10.0 * rep_amount
        draw_sdf_curved_arrow(
            buf,
            p0=(kn_r[0] - 115.0, kn_r[1] - 28.0),
            p1=(kn_r[0] - 62.0,  kn_r[1] + 8.0),
            p2=(kn_r[0] - 18.0 + shift, kn_r[1] + 6.0),
            col_main=col_main, col_soft=col_soft,
            w_tail=1.5, w_head=6.0, head_len=24.0, head_span=16.0, double_chevron=True
        )
        draw_sdf_curved_arrow(
            buf,
            p0=(kn_l[0] + 115.0, kn_l[1] - 28.0),
            p1=(kn_l[0] + 62.0,  kn_l[1] + 8.0),
            p2=(kn_l[0] + 18.0 - shift, kn_l[1] + 6.0),
            col_main=col_main, col_soft=col_soft,
            w_tail=1.5, w_head=6.0, head_len=24.0, head_span=16.0, double_chevron=True
        )

        # 3. Optical joint nodes on both knees & hips
        draw_sdf_joint_node(buf, kn_r, col_main, col_soft, r_core=5.2, r_ring=16.0, r_outer=25.0)
        draw_sdf_joint_node(buf, kn_l, col_main, col_soft, r_core=5.2, r_ring=16.0, r_outer=25.0)
        draw_sdf_joint_node(buf, hip_r, col_main, col_soft, r_core=4.0, r_ring=12.0, r_outer=19.0)
        draw_sdf_joint_node(buf, hip_l, col_main, col_soft, r_core=4.0, r_ring=12.0, r_outer=19.0)

        # 4. Architectural leader pins to exterior badges
        draw_sdf_leader_pin(buf, (kn_r[0] - 12.0, kn_r[1] - 14.0), (295.0, 1085.0), (320.0, 1045.0), col_main, col_soft)
        draw_sdf_leader_pin(buf, (kn_l[0] + 12.0, kn_l[1] - 14.0), (715.0, 955.0), (735.0, 898.0), col_main, col_soft)

    elif step_idx == 2:
        # STEP 2 (Front View — FIX: `0° KNEES TRACK TOES` & `96% VMO & GLUTE LOCK`)
        # 1. Angle gauge on right knee showing locked neutral tracking over toes
        draw_sdf_angle_gauge(
            buf, center_pt=kn_r, radius=92.0,
            deg_start=205.0, deg_end=145.0,
            col_main=col_main, col_soft=col_soft, ccw=True
        )

        # 2. Body-fitted curved outward torque arrows (`❮❮──╯` and `╰──❯❯`) driving knees wide over toes
        shift = 10.0 * rep_amount
        draw_sdf_curved_arrow(
            buf,
            p0=(kn_r[0] + 68.0, kn_r[1] + 14.0),
            p1=(kn_r[0] - 12.0, kn_r[1] - 12.0),
            p2=(kn_r[0] - 82.0 - shift, kn_r[1] - 4.0),
            col_main=col_main, col_soft=col_soft,
            w_tail=1.5, w_head=6.0, head_len=24.0, head_span=16.0, double_chevron=True
        )
        draw_sdf_curved_arrow(
            buf,
            p0=(kn_l[0] - 68.0, kn_l[1] + 14.0),
            p1=(kn_l[0] + 12.0, kn_l[1] - 12.0),
            p2=(kn_l[0] + 82.0 + shift, kn_l[1] - 4.0),
            col_main=col_main, col_soft=col_soft,
            w_tail=1.5, w_head=6.0, head_len=24.0, head_span=16.0, double_chevron=True
        )

        # 3. Optical joint nodes on knees & ankles
        draw_sdf_joint_node(buf, kn_r, col_main, col_soft, r_core=5.2, r_ring=16.0, r_outer=25.0)
        draw_sdf_joint_node(buf, kn_l, col_main, col_soft, r_core=5.2, r_ring=16.0, r_outer=25.0)
        draw_sdf_joint_node(buf, ank_r, col_main, col_soft, r_core=4.2, r_ring=13.0, r_outer=20.0)
        draw_sdf_joint_node(buf, ank_l, col_main, col_soft, r_core=4.2, r_ring=13.0, r_outer=20.0)

        draw_sdf_leader_pin(buf, (kn_r[0] - 12.0, kn_r[1] - 14.0), (295.0, 1070.0), (320.0, 1030.0), col_main, col_soft)
        draw_sdf_leader_pin(buf, (kn_l[0] + 12.0, kn_l[1] - 14.0), (715.0, 955.0), (735.0, 898.0), col_main, col_soft)

    elif step_idx == 3:
        # STEP 3 (3/4 Side View — WRONG: `+28° HEELS LIFTING` & `79% LUMBAR & PATELLA`)
        # 1. Angle gauge at hip showing excessive forward torso collapse
        draw_sdf_angle_gauge(
            buf, center_pt=hip_r, radius=112.0,
            deg_start=270.0, deg_end=325.0 + 12.0 * rep_amount,
            col_main=col_main, col_soft=col_soft, ccw=False
        )

        # 2. Curved forward torso collapse arrow (`⤵`) + upward heel lift arrow (`↑`)
        draw_sdf_curved_arrow(
            buf,
            p0=(sh[0] - 25.0, sh[1] - 55.0),
            p1=(sh[0] + 75.0, sh[1] - 15.0),
            p2=(sh[0] + 108.0, sh[1] + 78.0),
            col_main=col_main, col_soft=col_soft,
            w_tail=1.5, w_head=6.0, head_len=24.0, head_span=16.0, double_chevron=True
        )
        draw_sdf_curved_arrow(
            buf,
            p0=(ank_r[0] - 46.0, ank_r[1] + 24.0),
            p1=(ank_r[0] - 52.0, ank_r[1] - 26.0),
            p2=(ank_r[0] - 46.0, ank_r[1] - 82.0),
            col_main=col_main, col_soft=col_soft,
            w_tail=1.5, w_head=5.6, head_len=22.0, head_span=14.5, double_chevron=True
        )

        # 3. Optical joint nodes on hip, knee, and ankle
        draw_sdf_joint_node(buf, hip_r, col_main, col_soft, r_core=5.2, r_ring=16.0, r_outer=25.0)
        draw_sdf_joint_node(buf, kn_r, col_main, col_soft, r_core=4.8, r_ring=14.5, r_outer=22.0)
        draw_sdf_joint_node(buf, ank_r, col_main, col_soft, r_core=4.8, r_ring=14.5, r_outer=22.0)

        draw_sdf_leader_pin(buf, (hip_r[0] - 14.0, hip_r[1] + 10.0), (315.0, 1005.0), (340.0, 1055.0), col_main, col_soft)
        draw_sdf_leader_pin(buf, (kn_l[0] + 10.0, kn_l[1] - 20.0), (715.0, 955.0), (735.0, 898.0), col_main, col_soft)

    else:
        # STEP 4 (3/4 Side View — FIX: `95° PARALLEL DEPTH` & `98% QUADS & GLUTES`)
        # 1. Angle gauge at hip/femur showing locked parallel squat depth
        draw_sdf_angle_gauge(
            buf, center_pt=hip_r, radius=112.0,
            deg_start=192.0, deg_end=128.0,
            col_main=col_main, col_soft=col_soft, ccw=True
        )

        # 2. Upright chest drive arrow (`↑`) + flat heel floor lock arrow (`↓`)
        draw_sdf_curved_arrow(
            buf,
            p0=(sh[0] + 82.0, sh[1] + 95.0),
            p1=(sh[0] + 92.0, sh[1] + 25.0),
            p2=(sh[0] + 82.0, sh[1] - 48.0),
            col_main=col_main, col_soft=col_soft,
            w_tail=1.5, w_head=6.0, head_len=24.0, head_span=16.0, double_chevron=True
        )
        draw_sdf_curved_arrow(
            buf,
            p0=(ank_r[0] - 46.0, ank_r[1] - 82.0),
            p1=(ank_r[0] - 52.0, ank_r[1] - 30.0),
            p2=(ank_r[0] - 46.0, ank_r[1] + 18.0),
            col_main=col_main, col_soft=col_soft,
            w_tail=1.5, w_head=5.6, head_len=22.0, head_span=14.5, double_chevron=True
        )

        # 3. Optical joint nodes on hip, knee, and ankle
        draw_sdf_joint_node(buf, hip_r, col_main, col_soft, r_core=5.2, r_ring=16.0, r_outer=25.0)
        draw_sdf_joint_node(buf, kn_r, col_main, col_soft, r_core=4.8, r_ring=14.5, r_outer=22.0)
        draw_sdf_joint_node(buf, ank_r, col_main, col_soft, r_core=4.8, r_ring=14.5, r_outer=22.0)

        draw_sdf_leader_pin(buf, (hip_r[0] - 14.0, hip_r[1] + 10.0), (310.0, 995.0), (332.0, 1045.0), col_main, col_soft)
        draw_sdf_leader_pin(buf, (kn_l[0] + 10.0, kn_l[1] - 20.0), (715.0, 955.0), (735.0, 898.0), col_main, col_soft)

    canvas = Image.fromarray(buf).convert("RGBA")
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    if step_idx == 1:
        draw_v45_white_callout_badge(od, sd, (335, 1010), "18°", "VALGUS CAVE", col_main, col_soft, align="right")
        draw_v45_white_callout_badge(od, sd, (745, 865), "84%", "ACL & MCL LOAD", col_main, col_soft, align="left")
    elif step_idx == 2:
        draw_v45_white_callout_badge(od, sd, (345, 995), "0°", "KNEES TRACK TOES", col_main, col_soft, align="right")
        draw_v45_white_callout_badge(od, sd, (735, 865), "96%", "VMO & GLUTE LOCK", col_main, col_soft, align="left")
    elif step_idx == 3:
        draw_v45_white_callout_badge(od, sd, (348, 1020), "+28°", "HEELS LIFTING", col_main, col_soft, align="right")
        draw_v45_white_callout_badge(od, sd, (730, 865), "79%", "LUMBAR & PATELLA", col_main, col_soft, align="left")
    else:
        draw_v45_white_callout_badge(od, sd, (345, 1010), "95°", "PARALLEL DEPTH", col_main, col_soft, align="right")
        draw_v45_white_callout_badge(od, sd, (745, 865), "98%", "QUADS & GLUTES", col_main, col_soft, align="left")

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=6)))
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
        # Pick peak squat frame (u ~ 0.85)
        idx = int(round(0.82 * (len(pool) - 1)))
        rgb_frame = pool[idx].copy()
        kp_cur = kps[idx]

        canvas = render_v45_luxury_callouts_and_vectors(rgb_frame, s_idx, kp_cur, is_w, rep_amount=0.85)
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
        # Also save a 1:1 close-up crop of the torso+knees+gauges region (y=680..1360, x=140..940) to inspect arrow/dot quality!
        crop_zoom = bgr[640:1420, 120:960]
        closeup_panels.append(cv2.resize(crop_zoom, (540, 500), interpolation=cv2.INTER_LANCZOS4))

    sheet_2x2 = np.vstack([np.hstack(full_panels[:2]), np.hstack(full_panels[2:])])
    sheet_path = os.path.join(OUT_DIR, "day3_v45_luxury_design_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    closeup_2x2 = np.vstack([np.hstack(closeup_panels[:2]), np.hstack(closeup_panels[2:])])
    closeup_path = os.path.join(OUT_DIR, "day3_v45_luxury_closeup_sheet.jpg")
    cv2.imwrite(closeup_path, closeup_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved design sheets:", sheet_path, "and", closeup_path)


if __name__ == "__main__":
    main()
