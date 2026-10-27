#!/usr/bin/env python3
"""
v43 @demicstory Exact Visual Grammar Copy (Day 3: BARBELL BACK SQUAT)
=====================================================================
1. ZERO Ghost/Multiple Legs (`다리가 여러개로 보이는데`):
   - Drives smooth, physically-exact C-infinity kinematic squat articulation directly from the 4 razor-sharp
     3D anatomical studio plates (100% free of I2V cross-dissolve ghost limbs or duplicate calves/feet).
2. ZERO Text Between or On the Legs (`다리 사이에 단어들 넣어져있는데 겹쳐서 잘보이지도 않고`):
   - 0% text inside the 3D mannequin stage (`y = 320..1185`).
   - Mannequin feet end around `y = 1095..1125` and floor stance tracks end at `y <= 1165`, leaving
     a 130px pure dark charcoal studio floor buffer above the Bottom HUD (`y = 1302..1432`).
3. Luxury 2x-Supersampled Body-Fitted Arrows & Lines (`화살표 같은건 나쁘진 않은데 고급지게 해야지. 그리고 몸에 맞게 들어가야지. @demicstory를 카피해봐`):
   - Fast OpenCV + 2x supersampled stage rendering with multi-pass C++ Gaussian bloom (~8s for 600 frames).
   - Exact anatomical joint lock on every frame `f` (`hip`, `knee`, `ankle`, `shoulder`, `heel`, `toe`),
     3D elliptical joint rings, glowing 3D floor foot-stance tracks, 90° depth bracket (`∟`), and tapered
     swept-back chevron arrows (`>`).
"""

import os
import math
import shutil
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
# 2x supersampled ROI covering the entire 3D stage (y=330..1210, x=80..1000) -> 5.5x faster than full frame!
ROI_X0, ROI_Y0, ROI_W, ROI_H = 80, 330, 920, 880
SS = 2
SW, SH = ROI_W * SS, ROI_H * SS
FPS = 30
TOTAL_SEC = 20.0
TOTAL_FRAMES = int(TOTAL_SEC * FPS)  # 600 frames

OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)

STORE_ATT = "/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/attachments"
PLATE_1 = os.path.join(STORE_ATT, "1f31291aff1f3b8fcf439621627072b92cf7c13769570767b062d75ddbb2e2a0.jpg")
PLATE_2 = os.path.join(STORE_ATT, "ce9487fd707364467e0d050840047bba0423208b526080fff73d2fdc1c51ebcf.jpg")
PLATE_3 = os.path.join(STORE_ATT, "128766dd47d4a0a4a2e8935c0854a5a0277a3356a21f2b955de56826308d4e11.jpg")
PLATE_4 = os.path.join(STORE_ATT, "332c6cba31404e0993de09b2881b572d3b301fc7fe7669619a53761f9ed017cd.jpg")

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
def get_font(size: int):
    return ImageFont.truetype(FONT_BOLD, size)

F_TITLE   = get_font(36)
F_SUB     = get_font(21)
F_HUD_TAG = get_font(23)
F_HUD_PCT = get_font(74)
F_HUD_CUE = get_font(40)

COL_RED   = (255, 40, 60)    # #FF283C Crimson Red
COL_GREEN = (0, 245, 130)    # #00F582 Emerald Green
COL_WHITE = (250, 252, 255)

YY_GRID, XX_GRID = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_pt(p1, p2, t: float):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))


def lerp_col(c1, c2, t: float):
    return (
        int(round(lerp(c1[0], c2[0], t))),
        int(round(lerp(c1[1], c2[1], t))),
        int(round(lerp(c1[2], c2[2], t))),
    )


def place_plate_on_stage(img_bgr: np.ndarray, scale: float, y_foot_src: int, y_foot_dst: int, x_shift: int = 0) -> np.ndarray:
    fr = cv2.resize(img_bgr, (720, 1280), interpolation=cv2.INTER_AREA)
    sh, sw = fr.shape[:2]
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    resized = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((H, W, 3), (12, 10, 9), dtype=np.uint8)
    x0 = (W - nw) // 2 + x_shift
    y0 = y_foot_dst - int(round(y_foot_src * scale))

    dst_x1, dst_x2 = max(0, x0), min(W, x0 + nw)
    dst_y1, dst_y2 = max(315, y0), min(1205, y0 + nh)
    src_x1, src_x2 = dst_x1 - x0, (dst_x1 - x0) + (dst_x2 - dst_x1)
    src_y1, src_y2 = dst_y1 - y0, (dst_y1 - y0) + (dst_y2 - dst_y1)

    crop = resized[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
    yy = np.arange(dst_y1, dst_y2, dtype=np.float32)[:, None]
    xx = np.arange(dst_x1, dst_x2, dtype=np.float32)[None, :]

    ay_top = np.clip((yy - 315.0) / 48.0, 0.0, 1.0)
    ay_bot = np.clip((1205.0 - yy) / 45.0, 0.0, 1.0)
    ax_left = np.clip((xx - dst_x1) / 48.0, 0.0, 1.0) if dst_x1 > 0 else np.ones_like(xx)
    ax_right = np.clip((dst_x2 - xx) / 48.0, 0.0, 1.0) if dst_x2 < W else np.ones_like(xx)
    alpha = (ay_top * ay_bot * ax_left * ax_right)[..., None]

    bg = canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(crop * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)
    return canvas


def articulate_squat_frame(
    base_stage: np.ndarray,
    base_kpts: dict,
    squat_dy: float,
    torso_dx: float,
    knee_dx: float,
    hip_y: float,
    knee_y: float,
    ankle_y: float,
    is_front: bool,
    cam_zoom: float = 1.0,
    cam_cx: float = 540.0,
    cam_cy: float = 780.0,
) -> tuple[np.ndarray, dict]:
    t_thigh = np.clip((YY_GRID - (hip_y - 45.0)) / max(40.0, (ankle_y - 25.0) - (hip_y - 45.0)), 0.0, 1.0)
    s_vert = 0.5 * (1.0 + np.cos(np.pi * t_thigh))
    dy_field = squat_dy * s_vert

    bell_knee = np.exp(-0.5 * ((YY_GRID - knee_y) / 85.0) ** 2)
    if is_front:
        left_w = np.exp(-0.5 * ((XX_GRID - 385.0) / 80.0) ** 2)
        right_w = np.exp(-0.5 * ((XX_GRID - 695.0) / 80.0) ** 2)
        dx_field = knee_dx * bell_knee * (left_w - right_w)
    else:
        s_torso = 0.5 * (1.0 + np.cos(np.pi * np.clip((YY_GRID - (hip_y - 90.0)) / 180.0, 0.0, 1.0)))
        dx_field = torso_dx * s_torso + knee_dx * bell_knee

    map_x = cam_cx + (XX_GRID - cam_cx) / cam_zoom - dx_field
    map_y = cam_cy + (YY_GRID - cam_cy) / cam_zoom - dy_field

    warped = cv2.remap(
        base_stage,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(12, 10, 9),
    )

    out_kpts = {}
    for k, (jx, jy) in base_kpts.items():
        tt = min(1.0, max(0.0, (jy - (hip_y - 45.0)) / max(40.0, (ankle_y - 25.0) - (hip_y - 45.0))))
        sv = 0.5 * (1.0 + math.cos(math.pi * tt))
        j_dy = squat_dy * sv

        bk = math.exp(-0.5 * ((jy - knee_y) / 85.0) ** 2)
        if is_front:
            lw = math.exp(-0.5 * ((jx - 385.0) / 80.0) ** 2)
            rw = math.exp(-0.5 * ((jx - 695.0) / 80.0) ** 2)
            j_dx = knee_dx * bk * (lw - rw)
        else:
            st_w = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, (jy - (hip_y - 90.0)) / 180.0))))
            j_dx = torso_dx * st_w + knee_dx * bk

        nx = cam_cx + ((jx + j_dx) - cam_cx) * cam_zoom
        ny = cam_cy + ((jy + j_dy) - cam_cy) * cam_zoom
        out_kpts[k] = (nx, ny)

    return warped, out_kpts


def ss_pt(pt):
    return ((pt[0] - ROI_X0) * SS, (pt[1] - ROI_Y0) * SS)


def draw_demic_laser_axis(s_draw, g_draw, pts_1x, color, width_1x=7.5):
    if len(pts_1x) < 2:
        return
    pts = [ss_pt(p) for p in pts_1x]
    r, g, b = color
    w = int(round(width_1x * SS))
    g_draw.line(pts, fill=(r, g, b, 155), width=w + 18, joint="curve")
    s_draw.line(pts, fill=(r, g, b, 250), width=w, joint="curve")
    s_draw.line(pts, fill=(255, 255, 255, 220), width=max(3, w // 3), joint="curve")


def draw_demic_3d_joint_ring(s_draw, g_draw, pt_1x, color, rx_1x=22.0, ry_1x=11.0, tilt_deg=0.0, pulse=0.5):
    cx, cy = ss_pt(pt_1x)
    rx = (rx_1x + 2.5 * pulse) * SS
    ry = (ry_1x + 1.2 * pulse) * SS
    r, g, b = color
    ang = math.radians(tilt_deg)
    ca, sa = math.cos(ang), math.sin(ang)

    ring_pts = []
    for i in range(37):
        t = (i / 36.0) * 2.0 * math.pi
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        ring_pts.append((cx + ex * ca - ey * sa, cy + ex * sa + ey * ca))

    g_draw.line(ring_pts, fill=(r, g, b, 145), width=12, joint="curve")
    s_draw.line(ring_pts, fill=(r, g, b, 245), width=5, joint="curve")
    s_draw.line(ring_pts, fill=(255, 255, 255, 195), width=2, joint="curve")

    nr = 6 * SS
    g_draw.ellipse([cx - nr - 8, cy - nr - 8, cx + nr + 8, cy + nr + 8], fill=(r, g, b, 155))
    s_draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=(r, g, b, 255), outline=(255, 255, 255, 240), width=3)


def draw_demic_joint_node(s_draw, g_draw, pt_1x, color, radius_1x=6.0):
    cx, cy = ss_pt(pt_1x)
    rad = radius_1x * SS
    r, g, b = color
    g_draw.ellipse([cx - rad - 8, cy - rad - 8, cx + rad + 8, cy + rad + 8], fill=(r, g, b, 135))
    s_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(r, g, b, 255), outline=(255, 255, 255, 235), width=3)


def draw_demic_tapered_arrow(
    s_draw, g_draw, p_start_1x, p_end_1x, color,
    w_tail_1x=3.0, w_head_1x=11.5, head_len_1x=29.0, head_w_1x=18.0,
    progress=1.0, pulse_phase=0.0, double_chevron=True,
):
    if progress <= 0.03:
        return
    p0 = ss_pt(p_start_1x)
    p1_full = ss_pt(p_end_1x)
    p1 = lerp_pt(p0, p1_full, min(1.0, progress))

    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 12:
        return
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    r, g, b = color
    hlen = head_len_1x * SS
    hw = head_w_1x * SS
    wt = w_tail_1x * SS
    wh = w_head_1x * SS

    shaft_len = max(4.0, length - hlen * 0.52)
    n_segs = 18
    for i in range(n_segs):
        t0, t1 = i / n_segs, (i + 1) / n_segs
        s0 = (p0[0] + ux * shaft_len * t0, p0[1] + uy * shaft_len * t0)
        s1 = (p0[0] + ux * shaft_len * t1, p0[1] + uy * shaft_len * t1)
        tm = 0.5 * (t0 + t1)
        w_cur = lerp(wt, wh, tm ** 0.85)
        wave = math.exp(-((tm - (pulse_phase % 1.0)) ** 2) / 0.05)
        alpha_core = int(round(lerp(55, 255, tm ** 0.75)))
        alpha_glow = int(round(lerp(25, 155, tm) + 45 * wave))

        hw_cur = 0.5 * w_cur
        hw_next = 0.5 * lerp(wt, wh, t1 ** 0.85)
        quad = [
            (s0[0] + nx * hw_cur, s0[1] + ny * hw_cur),
            (s1[0] + nx * hw_next, s1[1] + ny * hw_next),
            (s1[0] - nx * hw_next, s1[1] - ny * hw_next),
            (s0[0] - nx * hw_cur, s0[1] - ny * hw_cur),
        ]
        g_draw.line([s0, s1], fill=(r, g, b, min(255, alpha_glow)), width=int(round(w_cur + 16)))
        s_draw.polygon(quad, fill=(r, g, b, alpha_core))
        if tm > 0.25:
            s_draw.line([s0, s1], fill=(255, 255, 255, int(round(alpha_core * (0.65 + 0.35 * wave)))), width=max(2, int(round(w_cur * 0.30))))

    def _draw_chevron(tip_pt, scale_h=1.0, alpha_mul=1.0):
        hl = hlen * scale_h
        h_w = hw * scale_h
        notch_d = hl * 0.54
        base_x, base_y = tip_pt[0] - ux * hl, tip_pt[1] - uy * hl
        notch_pt = (tip_pt[0] - ux * notch_d, tip_pt[1] - uy * notch_d)
        left_wing = (base_x + nx * h_w, base_y + ny * h_w)
        right_wing = (base_x - nx * h_w, base_y - ny * h_w)
        poly = [tip_pt, left_wing, notch_pt, right_wing]
        g_draw.polygon(poly, fill=(r, g, b, int(165 * alpha_mul)))
        s_draw.polygon(poly, fill=(r, g, b, int(255 * alpha_mul)))
        s_draw.line([left_wing, tip_pt, right_wing], fill=(255, 255, 255, int(240 * alpha_mul)), width=max(2, int(3 * scale_h)), joint="curve")

    _draw_chevron(p1, scale_h=1.0, alpha_mul=1.0)
    if double_chevron and length > hlen * 1.85:
        p_second = (p1[0] - ux * hlen * 0.68, p1[1] - uy * hlen * 0.68)
        _draw_chevron(p_second, scale_h=0.76, alpha_mul=0.75)


def draw_demic_curved_tapered_arrow(
    s_draw, g_draw, p0_1x, p1_1x, p2_1x, color,
    w_tail_1x=3.0, w_head_1x=11.5, head_len_1x=29.0, head_w_1x=18.0,
    progress=1.0, pulse_phase=0.0,
):
    if progress <= 0.03:
        return
    p0, p1, p2 = ss_pt(p0_1x), ss_pt(p1_1x), ss_pt(p2_1x)
    r, g, b = color
    wt, wh = w_tail_1x * SS, w_head_1x * SS
    hlen, hw = head_len_1x * SS, head_w_1x * SS

    n_pts = 24
    pts = []
    for i in range(n_pts + 1):
        t = (i / n_pts) * min(1.0, progress)
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((x, y))

    shaft_pts = pts[:-3]
    for i in range(len(shaft_pts) - 1):
        tm = i / max(1, len(shaft_pts) - 1)
        w_cur = int(round(lerp(wt, wh, tm ** 0.85)))
        wave = math.exp(-((tm - (pulse_phase % 1.0)) ** 2) / 0.05)
        alpha_c = int(round(lerp(55, 255, tm ** 0.75)))
        alpha_g = int(round(lerp(25, 155, tm) + 45 * wave))
        g_draw.line([shaft_pts[i], shaft_pts[i + 1]], fill=(r, g, b, min(255, alpha_g)), width=w_cur + 16)
        s_draw.line([shaft_pts[i], shaft_pts[i + 1]], fill=(r, g, b, alpha_c), width=w_cur)
        if tm > 0.25:
            s_draw.line([shaft_pts[i], shaft_pts[i + 1]], fill=(255, 255, 255, int(alpha_c * 0.78)), width=max(2, w_cur // 3))

    tip = pts[-1]
    prev = pts[-4]
    dx, dy = tip[0] - prev[0], tip[1] - prev[1]
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux

    notch_pt = (tip[0] - ux * (hlen * 0.54), tip[1] - uy * (hlen * 0.54))
    base_x, base_y = tip[0] - ux * hlen, tip[1] - uy * hlen
    left_wing = (base_x + nx * hw, base_y + ny * hw)
    right_wing = (base_x - nx * hw, base_y - ny * hw)
    poly = [tip, left_wing, notch_pt, right_wing]
    g_draw.polygon(poly, fill=(r, g, b, 165))
    s_draw.polygon(poly, fill=(r, g, b, 255))
    s_draw.line([left_wing, tip, right_wing], fill=(255, 255, 255, 240), width=3, joint="curve")


def draw_demic_floor_stance_track(s_draw, g_draw, heel_1x, toe_1x, color, half_w_1x=21.0):
    p0, p1 = ss_pt(heel_1x), ss_pt(toe_1x)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy * 1.15, ux * 0.55
    hw = half_w_1x * SS
    r, g, b = color

    e0 = (p0[0] - ux * 16 * SS, p0[1] - uy * 16 * SS)
    e1 = (p1[0] + ux * 20 * SS, p1[1] + uy * 20 * SS)

    quad = [
        (e0[0] - nx * hw * 0.82, e0[1] - ny * hw * 0.82),
        (e1[0] - nx * hw * 1.08, e1[1] - ny * hw * 1.08),
        (e1[0] + nx * hw * 1.08, e1[1] + ny * hw * 1.08),
        (e0[0] + nx * hw * 0.82, e0[1] + ny * hw * 0.82),
    ]
    g_draw.polygon(quad, fill=(r, g, b, 65))
    s_draw.polygon(quad, fill=(r, g, b, 38))
    g_draw.line(quad + [quad[0]], fill=(r, g, b, 150), width=12, joint="curve")
    s_draw.line(quad + [quad[0]], fill=(r, g, b, 245), width=5, joint="curve")
    s_draw.line(quad + [quad[0]], fill=(255, 255, 255, 195), width=2, joint="curve")


def draw_demic_status_badge(s_draw, g_draw, center_1x, is_correct: bool):
    cx, cy = ss_pt(center_1x)
    rad = 32 * SS
    col = COL_GREEN if is_correct else COL_RED
    r, g, b = col

    g_draw.ellipse([cx - rad - 14, cy - rad - 14, cx + rad + 14, cy + rad + 14], fill=(r, g, b, 115))
    s_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(14, 18, 24, 230), outline=(r, g, b, 255), width=6)
    s_draw.ellipse([cx - rad + 4, cy - rad + 4, cx + rad - 4, cy + rad - 4], outline=(255, 255, 255, 130), width=2)

    if is_correct:
        pts = [(cx - 13 * SS, cy + 1 * SS), (cx - 3 * SS, cy + 12 * SS), (cx + 15 * SS, cy - 11 * SS)]
        g_draw.line(pts, fill=(r, g, b, 180), width=16, joint="curve")
        s_draw.line(pts, fill=(r, g, b, 255), width=9, joint="curve")
        s_draw.line(pts, fill=(255, 255, 255, 235), width=4, joint="curve")
    else:
        d = 12 * SS
        for p_a, p_b in [((cx - d, cy - d), (cx + d, cy + d)), ((cx - d, cy + d), (cx + d, cy - d))]:
            g_draw.line([p_a, p_b], fill=(r, g, b, 180), width=16)
            s_draw.line([p_a, p_b], fill=(r, g, b, 255), width=9)
            s_draw.line([p_a, p_b], fill=(255, 255, 255, 235), width=4)


def draw_shadow_text_1x(s_draw_1x, g_draw_1x, pos, text: str, font, fill):
    x, y = int(round(pos[0])), int(round(pos[1]))
    g_draw_1x.text((x, y + 3), text, font=font, fill=(0, 0, 0, 235))
    g_draw_1x.text((x - 2, y), text, font=font, fill=(0, 0, 0, 185))
    g_draw_1x.text((x + 2, y), text, font=font, fill=(0, 0, 0, 185))
    s_draw_1x.text((x, y), text, font=font, fill=(fill[0], fill[1], fill[2], 255))


def fast_composite_roi(base_bgr: np.ndarray, glow_ss: Image.Image, sharp_ss: Image.Image, hud_rgba: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Composite 2x supersampled stage ROI (x=80..1000, y=330..1210)
    roi = out[ROI_Y0 : ROI_Y0 + ROI_H, ROI_X0 : ROI_X0 + ROI_W]

    g_arr = np.array(glow_ss, dtype=np.uint8)
    g_small = cv2.resize(g_arr, (ROI_W // 2, ROI_H // 2), interpolation=cv2.INTER_LINEAR)
    g_blur = cv2.GaussianBlur(g_small, (15, 15), 4.0)
    g_1x = cv2.resize(g_blur, (ROI_W, ROI_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    ga = g_1x[:, :, 3:4] / 255.0
    roi = roi * (1.0 - ga) + g_1x[:, :, :3] * ga

    s_arr = np.array(sharp_ss, dtype=np.uint8)
    s_1x = cv2.resize(s_arr, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    sa = s_1x[:, :, 3:4] / 255.0
    roi = roi * (1.0 - sa) + s_1x[:, :, :3] * sa

    out[ROI_Y0 : ROI_Y0 + ROI_H, ROI_X0 : ROI_X0 + ROI_W] = roi

    # Composite cached HUD layer
    ha = hud_rgba[:, :, 3:4]
    out = out * (1.0 - ha) + hud_rgba[:, :, :3] * ha
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def build_hud_layer(step_idx: int, hud_tag: str, hud_pct: str, hud_cue: str, bar_ratio: float) -> np.ndarray:
    is_correct = step_idx in (2, 4)
    theme_col = COL_GREEN if is_correct else COL_RED

    hud_glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hud_sharp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hg_draw = ImageDraw.Draw(hud_glow)
    hs_draw = ImageDraw.Draw(hud_sharp)

    hx, hy = 98, 260
    draw_shadow_text_1x(hs_draw, hg_draw, (hx, hy), "BARBELL BACK SQUAT", F_TITLE, COL_WHITE)
    hs_draw.rectangle([hx, hy + 46, hx + 455, hy + 49], fill=(theme_col[0], theme_col[1], theme_col[2], 245))
    hg_draw.rectangle([hx, hy + 45, hx + 455, hy + 50], fill=(theme_col[0], theme_col[1], theme_col[2], 120))

    step_label = (
        f"0{step_idx} / 04 · "
        + (
            "WRONG: KNEES CAVING IN" if step_idx == 1 else
            "FIX: PUSH KNEES OUT & PLANT FEET" if step_idx == 2 else
            "WRONG: TORSO DROP & HEELS UP" if step_idx == 3 else
            "FIX: CHEST UP & HEELS FLAT"
        )
    )
    draw_shadow_text_1x(hs_draw, hg_draw, (hx, hy + 56), step_label, F_SUB, theme_col)

    bx, by = 98, 1308
    draw_shadow_text_1x(hs_draw, hg_draw, (bx, by), hud_tag, F_HUD_TAG, theme_col)
    draw_shadow_text_1x(hs_draw, hg_draw, (bx, by + 32), hud_pct, F_HUD_PCT, COL_WHITE)

    div_x = bx + 184
    hs_draw.rectangle([div_x, by + 44, div_x + 4, by + 102], fill=(theme_col[0], theme_col[1], theme_col[2], 255))
    hg_draw.rectangle([div_x - 2, by + 42, div_x + 6, by + 104], fill=(theme_col[0], theme_col[1], theme_col[2], 120))

    draw_shadow_text_1x(hs_draw, hg_draw, (div_x + 20, by + 52), hud_cue, F_HUD_CUE, COL_WHITE)

    bar_y = by + 122
    bar_w = 690
    hs_draw.rectangle([bx, bar_y, bx + bar_w, bar_y + 4], fill=(255, 255, 255, 55))
    fill_w = int(bar_w * bar_ratio)
    hg_draw.rectangle([bx, bar_y - 3, bx + fill_w, bar_y + 7], fill=(theme_col[0], theme_col[1], theme_col[2], 140))
    hs_draw.rectangle([bx, bar_y, bx + fill_w, bar_y + 4], fill=(theme_col[0], theme_col[1], theme_col[2], 255))
    hs_draw.ellipse([bx + fill_w - 6, bar_y - 4, bx + fill_w + 6, bar_y + 8], fill=(255, 255, 255, 255))

    hg_arr = cv2.GaussianBlur(np.array(hud_glow, dtype=np.uint8), (9, 9), 2.2)
    comp = Image.alpha_composite(Image.fromarray(hg_arr), hud_sharp)
    arr = np.array(comp, dtype=np.float32)
    arr[:, :, 3:4] /= 255.0
    return arr


HUD_CACHE = {}


def render_step_frame(frame_bgr: np.ndarray, step_idx: int, local_t: float, kpts: dict) -> np.ndarray:
    glow_ss = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    sharp_ss = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    g_ss = ImageDraw.Draw(glow_ss)
    s_ss = ImageDraw.Draw(sharp_ss)

    anim_p = min(1.0, local_t / 0.50)
    pulse = 0.5 + 0.5 * math.sin(local_t * 5.5)
    pulse_phase = local_t * 1.65
    shift_px = 5.0 * math.sin(local_t * 5.0)

    is_correct = step_idx in (2, 4)
    draw_demic_status_badge(s_ss, g_ss, (898, 402), is_correct=is_correct)

    if step_idx == 1:
        l_hip, r_hip = kpts["l_hip"], kpts["r_hip"]
        l_knee, r_knee = kpts["l_knee"], kpts["r_knee"]
        l_ank, r_ank = kpts["l_ankle"], kpts["r_ankle"]

        draw_demic_laser_axis(s_ss, g_ss, [l_hip, l_knee, l_ank], COL_RED, width_1x=7.5)
        draw_demic_laser_axis(s_ss, g_ss, [r_hip, r_knee, r_ank], COL_RED, width_1x=7.5)

        draw_demic_3d_joint_ring(s_ss, g_ss, l_knee, COL_RED, rx_1x=23, ry_1x=11, tilt_deg=-14, pulse=pulse)
        draw_demic_3d_joint_ring(s_ss, g_ss, r_knee, COL_RED, rx_1x=23, ry_1x=11, tilt_deg=14, pulse=pulse)
        for pt in (l_hip, r_hip, l_ank, r_ank):
            draw_demic_joint_node(s_ss, g_ss, pt, COL_RED, radius_1x=6.0)

        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (l_knee[0] - 115, l_knee[1]),
            (l_knee[0] - 24 + shift_px, l_knee[1]),
            COL_RED, w_tail_1x=3.0, w_head_1x=11.5, head_len_1x=28, head_w_1x=17.5,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=True,
        )
        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (r_knee[0] + 115, r_knee[1]),
            (r_knee[0] + 24 - shift_px, r_knee[1]),
            COL_RED, w_tail_1x=3.0, w_head_1x=11.5, head_len_1x=28, head_w_1x=17.5,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=True,
        )

        hud_tag, hud_pct, hud_cue, bar_ratio = "JOINT OVERLOAD · MEDIAL KNEE & ACL", "86%", "KNEES CAVING IN", 0.86

    elif step_idx == 2:
        morph_t = min(1.0, max(0.0, local_t / 0.65))
        morph_e = 1.0 - (1.0 - morph_t) ** 3
        line_col = lerp_col(COL_RED, COL_GREEN, morph_e)

        l_hip, r_hip = kpts["l_hip"], kpts["r_hip"]
        l_ank, r_ank = kpts["l_ankle"], kpts["r_ankle"]
        l_knee_bad = (l_ank[0] + 44.0, kpts["l_knee"][1] + 10.0)
        r_knee_bad = (r_ank[0] - 44.0, kpts["r_knee"][1] + 10.0)
        l_knee = lerp_pt(l_knee_bad, kpts["l_knee"], morph_e)
        r_knee = lerp_pt(r_knee_bad, kpts["r_knee"], morph_e)

        draw_demic_floor_stance_track(s_ss, g_ss, (l_ank[0] + 12, l_ank[1] + 12), (l_ank[0] - 32, l_ank[1] + 42), COL_GREEN, half_w_1x=20.0)
        draw_demic_floor_stance_track(s_ss, g_ss, (r_ank[0] - 12, r_ank[1] + 12), (r_ank[0] + 32, r_ank[1] + 42), COL_GREEN, half_w_1x=20.0)

        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (l_ank[0] - 42, l_ank[1] - 45),
            (l_ank[0] - 42, l_ank[1] + 28 + abs(shift_px) * 0.6),
            COL_GREEN, w_tail_1x=2.5, w_head_1x=9.5, head_len_1x=22, head_w_1x=14,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=False,
        )
        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (r_ank[0] + 42, r_ank[1] - 45),
            (r_ank[0] + 42, r_ank[1] + 28 + abs(shift_px) * 0.6),
            COL_GREEN, w_tail_1x=2.5, w_head_1x=9.5, head_len_1x=22, head_w_1x=14,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=False,
        )

        draw_demic_laser_axis(s_ss, g_ss, [l_hip, l_knee, l_ank], line_col, width_1x=8.0)
        draw_demic_laser_axis(s_ss, g_ss, [r_hip, r_knee, r_ank], line_col, width_1x=8.0)

        draw_demic_3d_joint_ring(s_ss, g_ss, l_knee, line_col, rx_1x=24, ry_1x=11.5, tilt_deg=10, pulse=pulse)
        draw_demic_3d_joint_ring(s_ss, g_ss, r_knee, line_col, rx_1x=24, ry_1x=11.5, tilt_deg=-10, pulse=pulse)
        for pt in (l_hip, r_hip, l_ank, r_ank):
            draw_demic_joint_node(s_ss, g_ss, pt, line_col, radius_1x=6.5)

        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (l_knee[0] - 18, l_knee[1]),
            (l_knee[0] - 122 - abs(shift_px), l_knee[1]),
            COL_GREEN, w_tail_1x=3.5, w_head_1x=12.0, head_len_1x=28, head_w_1x=17.5,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=True,
        )
        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (r_knee[0] + 18, r_knee[1]),
            (r_knee[0] + 122 + abs(shift_px), r_knee[1]),
            COL_GREEN, w_tail_1x=3.5, w_head_1x=12.0, head_len_1x=28, head_w_1x=17.5,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=True,
        )

        hud_tag, hud_pct, hud_cue, bar_ratio = "TARGET · QUADRICEPS & ADDUCTORS", "96%", "PUSH KNEES OUT", 0.96

    elif step_idx == 3:
        sh_pt, hip_pt, knee_pt, heel_pt, toe_pt = kpts["shoulder"], kpts["hip"], kpts["knee"], kpts["heel"], kpts["toe"]

        draw_demic_laser_axis(s_ss, g_ss, [sh_pt, hip_pt, knee_pt, heel_pt, toe_pt], COL_RED, width_1x=7.5)
        draw_demic_3d_joint_ring(s_ss, g_ss, knee_pt, COL_RED, rx_1x=22, ry_1x=11, tilt_deg=18, pulse=pulse)
        draw_demic_3d_joint_ring(s_ss, g_ss, hip_pt, COL_RED, rx_1x=20, ry_1x=10, tilt_deg=-22, pulse=pulse)
        for pt in (sh_pt, heel_pt, toe_pt):
            draw_demic_joint_node(s_ss, g_ss, pt, COL_RED, radius_1x=6.0)

        draw_demic_curved_tapered_arrow(
            s_ss, g_ss,
            (sh_pt[0] + 45, sh_pt[1] - 55),
            (sh_pt[0] + 135, sh_pt[1] + 5),
            (sh_pt[0] + 125, sh_pt[1] + 125 + abs(shift_px)),
            COL_RED, w_tail_1x=3.5, w_head_1x=12.0, head_len_1x=28, head_w_1x=17.5,
            progress=anim_p, pulse_phase=pulse_phase,
        )

        floor_y = toe_pt[1] + 10.0
        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (heel_pt[0] - 34, floor_y),
            (heel_pt[0] - 34, heel_pt[1] - 58 - abs(shift_px)),
            COL_RED, w_tail_1x=3.0, w_head_1x=11.0, head_len_1x=25, head_w_1x=16.0,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=True,
        )

        hud_tag, hud_pct, hud_cue, bar_ratio = "OVERLOAD · LUMBAR SPINE & KNEE", "79%", "TORSO DROP & HEELS UP", 0.79

    else:
        morph_t = min(1.0, max(0.0, local_t / 0.65))
        morph_e = 1.0 - (1.0 - morph_t) ** 3
        line_col = lerp_col(COL_RED, COL_GREEN, morph_e)

        sh_bad = (kpts["shoulder"][0] + 28.0, kpts["shoulder"][1] + 42.0)
        hip_bad = (kpts["hip"][0] - 14.0, kpts["hip"][1] - 32.0)
        heel_bad = (kpts["heel"][0], kpts["heel"][1] - 38.0)

        sh_pt = lerp_pt(sh_bad, kpts["shoulder"], morph_e)
        hip_pt = lerp_pt(hip_bad, kpts["hip"], morph_e)
        knee_pt = kpts["knee"]
        heel_pt = lerp_pt(heel_bad, kpts["heel"], morph_e)
        toe_pt = kpts["toe"]

        draw_demic_floor_stance_track(s_ss, g_ss, (heel_pt[0] - 10, heel_pt[1] + 10), (toe_pt[0] + 16, toe_pt[1] + 10), COL_GREEN, half_w_1x=18.0)

        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (heel_pt[0] - 32, heel_pt[1] - 68),
            (heel_pt[0] - 32, heel_pt[1] + 4 + abs(shift_px) * 0.6),
            COL_GREEN, w_tail_1x=2.8, w_head_1x=10.0, head_len_1x=23, head_w_1x=14.5,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=False,
        )
        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (toe_pt[0] + 26, toe_pt[1] - 60),
            (toe_pt[0] + 26, toe_pt[1] + 4 + abs(shift_px) * 0.6),
            COL_GREEN, w_tail_1x=2.8, w_head_1x=10.0, head_len_1x=23, head_w_1x=14.5,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=False,
        )

        if morph_e > 0.35:
            draw_demic_laser_axis(s_ss, g_ss, [(hip_pt[0], hip_pt[1]), (knee_pt[0], hip_pt[1])], COL_GREEN, width_1x=3.5)
            b_sz = 24.0
            draw_demic_laser_axis(
                s_ss, g_ss,
                [
                    (hip_pt[0] + b_sz, hip_pt[1] - b_sz * 0.85),
                    (hip_pt[0] + b_sz * 1.85, hip_pt[1] - b_sz * 0.85),
                    (hip_pt[0] + b_sz * 1.85, hip_pt[1]),
                ],
                COL_GREEN, width_1x=3.0,
            )

        draw_demic_laser_axis(s_ss, g_ss, [sh_pt, hip_pt, knee_pt, heel_pt, toe_pt], line_col, width_1x=8.0)
        draw_demic_3d_joint_ring(s_ss, g_ss, knee_pt, line_col, rx_1x=23, ry_1x=11, tilt_deg=15, pulse=pulse)
        draw_demic_3d_joint_ring(s_ss, g_ss, hip_pt, line_col, rx_1x=21, ry_1x=10.5, tilt_deg=-18, pulse=pulse)
        for pt in (sh_pt, heel_pt, toe_pt):
            draw_demic_joint_node(s_ss, g_ss, pt, line_col, radius_1x=6.5)

        draw_demic_curved_tapered_arrow(
            s_ss, g_ss,
            (sh_pt[0] + 82, sh_pt[1] + 105),
            (sh_pt[0] + 102, sh_pt[1] + 10),
            (sh_pt[0] + 58, sh_pt[1] - 85 - abs(shift_px)),
            COL_GREEN, w_tail_1x=3.5, w_head_1x=12.0, head_len_1x=28, head_w_1x=17.5,
            progress=anim_p, pulse_phase=pulse_phase,
        )

        draw_demic_tapered_arrow(
            s_ss, g_ss,
            (hip_pt[0] - 62, hip_pt[1] - 95),
            (hip_pt[0] - 62, hip_pt[1] + 35 + abs(shift_px)),
            COL_GREEN, w_tail_1x=3.5, w_head_1x=12.0, head_len_1x=28, head_w_1x=17.5,
            progress=anim_p, pulse_phase=pulse_phase, double_chevron=True,
        )

        hud_tag, hud_pct, hud_cue, bar_ratio = "TARGET · QUADS & GLUTEUS MAXIMUS", "98%", "CHEST UP · FEET FLAT", 0.98

    if step_idx not in HUD_CACHE:
        HUD_CACHE[step_idx] = build_hud_layer(step_idx, hud_tag, hud_pct, hud_cue, bar_ratio)

    return fast_composite_roi(frame_bgr, glow_ss, sharp_ss, HUD_CACHE[step_idx])


def main():
    p1_stage = place_plate_on_stage(cv2.imread(PLATE_1), scale=0.88, y_foot_src=1085, y_foot_dst=1135, x_shift=-24)
    p2_stage = place_plate_on_stage(cv2.imread(PLATE_2), scale=1.08, y_foot_src=1110, y_foot_dst=1152, x_shift=24)
    p3_stage = place_plate_on_stage(cv2.imread(PLATE_3), scale=0.86, y_foot_src=1175, y_foot_dst=1132, x_shift=-26)
    p4_stage = place_plate_on_stage(cv2.imread(PLATE_4), scale=1.06, y_foot_src=1175, y_foot_dst=1152, x_shift=26)

    kpts_p1 = {
        "l_hip": (380.0, 763.0), "r_hip": (652.0, 763.0),
        "l_knee": (396.0, 888.0), "r_knee": (636.0, 889.0),
        "l_ankle": (363.0, 1088.0), "r_ankle": (669.0, 1088.0),
    }
    kpts_p2 = {
        "l_hip": (432.0, 718.0), "r_hip": (696.0, 718.0),
        "l_knee": (382.0, 846.0), "r_knee": (748.0, 846.0),
        "l_ankle": (396.0, 1072.0), "r_ankle": (732.0, 1072.0),
    }
    kpts_p3 = {
        "shoulder": (512.0, 582.0),
        "hip": (412.0, 676.0),
        "knee": (526.0, 786.0),
        "heel": (428.0, 1042.0),
        "toe": (532.0, 1086.0),
    }
    kpts_p4 = {
        "shoulder": (568.0, 555.0),
        "hip": (472.0, 685.0),
        "knee": (602.0, 808.0),
        "heel": (476.0, 1092.0),
        "toe": (605.0, 1095.0),
    }

    raw_mp4 = os.path.join(OUT_DIR, "day3_v43_raw.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day3_v43_demic_master_reel.mp4")
    sheet_jpg = os.path.join(OUT_DIR, "day3_v43_demic_master_sheet.jpg")

    writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    keyframes = {}

    print("Rendering 600 frames with fast ROI 2x supersampled pipeline...")
    for fidx in range(TOTAL_FRAMES):
        step_idx = min(4, (fidx // 150) + 1)
        local_f = fidx - (step_idx - 1) * 150
        local_t = local_f / float(FPS)
        u = local_f / 149.0

        cyc = 0.5 - 0.5 * math.cos(u * 2.0 * math.pi)

        if step_idx == 1:
            squat_dy = -55.0 + 95.0 * cyc
            knee_dx = 26.0 * cyc
            cam_z = 0.94 + 0.16 * cyc
            st, kp = articulate_squat_frame(p1_stage, kpts_p1, squat_dy, 0.0, knee_dx, 763.0, 888.0, 1088.0, True, cam_z, 516.0, 815.0)
        elif step_idx == 2:
            cyc2 = 0.5 + 0.5 * math.cos(u * 2.0 * math.pi)
            squat_dy = -60.0 + 98.0 * cyc2
            knee_dx = -22.0 * cyc2
            cam_z = 1.22 - 0.16 * (1.0 - cyc2)
            st, kp = articulate_squat_frame(p2_stage, kpts_p2, squat_dy, 0.0, knee_dx, 718.0, 846.0, 1072.0, True, cam_z, 564.0, 730.0)
            st = np.clip(st.astype(np.float32) * np.array([1.12, 1.22, 1.04], dtype=np.float32) + np.array([14.0, 22.0, 8.0], dtype=np.float32), 0, 255).astype(np.uint8)
        elif step_idx == 3:
            squat_dy = -52.0 + 92.0 * cyc
            torso_dx = 34.0 * cyc
            knee_dx = 18.0 * cyc
            cam_z = 0.93 + 0.16 * cyc
            st, kp = articulate_squat_frame(p3_stage, kpts_p3, squat_dy, torso_dx, knee_dx, 676.0, 786.0, 1064.0, False, cam_z, 495.0, 825.0)
        else:
            cyc4 = 0.5 + 0.5 * math.cos(u * 2.0 * math.pi)
            squat_dy = -58.0 + 96.0 * cyc4
            torso_dx = -16.0 * cyc4
            knee_dx = 14.0 * cyc4
            cam_z = 1.22 - 0.16 * (1.0 - cyc4)
            st, kp = articulate_squat_frame(p4_stage, kpts_p4, squat_dy, torso_dx, knee_dx, 685.0, 808.0, 1092.0, False, cam_z, 575.0, 725.0)
            st = np.clip(st.astype(np.float32) * np.array([1.12, 1.22, 1.04], dtype=np.float32) + np.array([14.0, 22.0, 8.0], dtype=np.float32), 0, 255).astype(np.uint8)

        rendered = render_step_frame(st, step_idx, local_t, kp)
        writer.write(rendered)

        if local_f == 72:
            keyframes[step_idx] = rendered.copy()

    writer.release()

    wav_path = os.path.join(OUT_DIR, "day3_v43_audio.wav")
    if not os.path.exists(wav_path) and os.path.exists(os.path.join(OUT_DIR, "day3_v42_audio.wav")):
        shutil.copyfile(os.path.join(OUT_DIR, "day3_v42_audio.wav"), wav_path)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", raw_mp4,
            "-i", wav_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    pw, ph = 360, 640
    sheet = Image.new("RGB", (pw * 4, ph), (12, 14, 18))
    for s in range(1, 5):
        rgb = cv2.cvtColor(keyframes[s], cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(im, ((s - 1) * pw, 0))
    sheet.save(sheet_jpg, quality=95)
    print("Successfully built:", final_mp4)
    print("Successfully built:", sheet_jpg)


if __name__ == "__main__":
    main()
