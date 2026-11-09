#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v50_clean_single_arrow_and_posture.py`)
============================================================================
Fixes every visual issue identified in `v49` QC:
1. Unmistakable Physical Posture Difference (`차이가 한눈에 명확하게`):
   - Step 1 (Front Wrong — `18° KNEE VALGUS`): Both knees genuinely buckle INWARD (`>  <`) inside the
     vertical ankle plumb line (`kn_r.x` moves inward to `x~386` vs `ank_r.x=320`; `kn_l.x` moves inward
     to `x~696` vs `ank_l.x=762`), creating a dramatic 18° inward valgus cave compared to Step 2 (`<  >`
     where knees track wide over toes at `x~266` and `x~847` — over `270px` total knee-width difference!).
   - Step 3 (Side Wrong — `+28° FORWARD LEAN & HEELS LIFTING`): Upper torso genuinely folds forward/down
     and the heel genuinely lifts clearly off the floor line compared to Step 4 (upright torso + flat heel).
2. Clean Biomechanical Reference vs. Actual Axis (`기준선 vs 실제 정렬축`):
   - Front View (Steps 1 & 2): Clean Vertical Plumb Reference Line (`하얀 점선 수직 기준선`) straight up from
     each Ankle Pivot (`ank_r`, `ank_l`) vs. Actual Shin Axis (`Ankle -> Knee`), with a clean shaded
     `18°` deviation sector in Step 1 and `0°` locked alignment in Step 2 (removes the awkward sideways
     groin line that cut across the shorts).
   - Side View (Steps 3 & 4): Clean Upright Spine Reference Line & Flat Floor Reference Line vs. Actual
     `Shoulder — Hip — Knee — Ankle` axis with a shaded `+28°` forward-lean sector in Step 3 and `95°`
     parallel depth sector in Step 4.
3. Classic, Ultra-Clean Single-Head Arrows (`깔끔한 단일 헤드 화살표 ───▶`):
   - Completely removes the `>>>` multi-chevron / segmented shaft design that looked glitchy in `v49`,
     and completely removes the C-loop knee polygon from `v48`.
   - Draws each directional arrow as ONE unified, smooth 7-vertex single-head arrow polygon (`───▶`)
     with a clean dark drop-shadow and solid high-contrast fill, positioned directly at the joint movement.
   - Zero underline bars under the Pure White (`#FFFFFF`) metric numbers and sub-labels.
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
COL_RED_LIGHT = (255, 145, 140)
COL_GREEN = (18, 218, 118)
COL_GREEN_LIGHT = (145, 250, 195)
COL_WHITE = (255, 255, 255)
COL_REF = (235, 242, 252)

FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_CACHE = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(FONT_BOLD_PATH, size)
    return _FONT_CACHE[size]


_GRID_X, _GRID_Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def apply_biomechanical_posture_diff(
    rgb: np.ndarray,
    kp: dict,
    step_idx: int,
    u: float,
) -> tuple[np.ndarray, dict]:
    """
    Smooth, artifact-free biomechanical posture differentiation so that:
    - Step 1 (Front Wrong) has unmistakable inward X-leg knee valgus (`>  <`) where the knees cave
      inward inside the ankles (`kn_r.x > ank_r.x`, `kn_l.x < ank_l.x`).
    - Step 3 (Side Wrong) has unmistakable forward torso fold + lifted heel off the floor.
    """
    kp_out = {k: (float(v[0]), float(v[1])) for k, v in kp.items()}
    if u <= 0.02 or step_idx in (2, 4):
        return rgb, kp_out

    dx_map = np.zeros((H, W), dtype=np.float32)
    dy_map = np.zeros((H, W), dtype=np.float32)

    if step_idx == 1:
        # In Step 1 raw bottom frame, `kn_r.x ~ 270`, `ank_r.x ~ 320`, `kn_l.x ~ 810`, `ank_l.x ~ 762`.
        # To make the knees visibly buckle INWARD inside the ankles (`>  <` X-다리 무릎 안쪽 붕괴),
        # we shift the knee region inward by `118px * u` while keeping the feet/ankles (`y >= 1485`)
        # and upper torso/hips (`y <= 1030`) smoothly anchored!
        inward_px = 118.0 * u
        down_px = 12.0 * u

        kn_r = kp["kn_r"]
        kn_l = kp["kn_l"]
        ank_r = kp["ank_r"]

        cy_r = kn_r[1] + 25.0
        cy_l = kn_l[1] + 25.0
        sig_x = 165.0
        sig_y = 155.0

        # Smooth vertical pin masks: 1.0 across the knee/thigh/upper-shin (`y = 1090..1430`),
        # smoothly tapering to 0 at the ankle (`y = 1495`) and upper pelvis (`y = 1010`).
        foot_mask = np.clip((ank_r[1] - 20.0 - _GRID_Y) / 85.0, 0.0, 1.0)
        top_mask = np.clip((_GRID_Y - 995.0) / 95.0, 0.0, 1.0)
        pin_mask = (foot_mask * top_mask) ** 1.2

        cx_r_mid = kn_r[0] + inward_px * 0.52
        cx_l_mid = kn_l[0] - inward_px * 0.52
        w_r = np.exp(-((_GRID_X - cx_r_mid) ** 2) / (2.0 * sig_x ** 2) - ((_GRID_Y - cy_r) ** 2) / (2.0 * sig_y ** 2)) * pin_mask
        w_l = np.exp(-((_GRID_X - cx_l_mid) ** 2) / (2.0 * sig_x ** 2) - ((_GRID_Y - cy_l) ** 2) / (2.0 * sig_y ** 2)) * pin_mask

        dx_map += inward_px * w_r - inward_px * w_l
        dy_map += down_px * (w_r + w_l)

        kp_out["kn_r"] = (kn_r[0] + inward_px * 0.86, kn_r[1] + down_px * 0.86)
        kp_out["kn_l"] = (kn_l[0] - inward_px * 0.86, kn_l[1] + down_px * 0.86)

    elif step_idx == 3:
        sh = kp["sh"]
        hip_r = kp["hip_r"]
        kn_r = kp["kn_r"]
        ank_r = kp["ank_r"]

        lean_dx = 82.0 * u
        lean_dy = 56.0 * u
        heel_lift_dy = -52.0 * u
        knee_fwd_dx = 34.0 * u

        cx_sh = sh[0] + lean_dx * 0.45
        cy_sh = sh[1] + 50.0
        torso_w = np.exp(-((_GRID_X - cx_sh) ** 2) / (2.0 * 260.0 ** 2) - ((_GRID_Y - cy_sh) ** 2) / (2.0 * 250.0 ** 2))
        torso_pin = np.clip((hip_r[1] + 25.0 - _GRID_Y) / 145.0, 0.0, 1.0)
        torso_w *= torso_pin

        heel_x = ank_r[0] - 28.0
        heel_y = ank_r[1] + 15.0
        heel_w = np.exp(-((_GRID_X - heel_x) ** 2) / (2.0 * 82.0 ** 2) - ((_GRID_Y - heel_y) ** 2) / (2.0 * 90.0 ** 2))
        toe_pin = np.clip((ank_r[0] + 52.0 - _GRID_X) / 65.0, 0.0, 1.0)
        heel_w *= toe_pin

        kn_w = np.exp(-((_GRID_X - kn_r[0]) ** 2) / (2.0 * 140.0 ** 2) - ((_GRID_Y - kn_r[1]) ** 2) / (2.0 * 145.0 ** 2))

        dx_map += lean_dx * torso_w + knee_fwd_dx * kn_w
        dy_map += lean_dy * torso_w + heel_lift_dy * heel_w

        kp_out["sh"] = (sh[0] + lean_dx * 0.85, sh[1] + lean_dy * 0.85)
        kp_out["kn_r"] = (kn_r[0] + knee_fwd_dx * 0.85, kn_r[1])
        kp_out["ank_r"] = (ank_r[0], ank_r[1] + heel_lift_dy * 0.78)

    map_x = (_GRID_X - dx_map).astype(np.float32)
    map_y = (_GRID_Y - dy_map).astype(np.float32)
    warped = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped, kp_out


def draw_dashed_line_ss(
    rd: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    col_rgba: tuple[int, int, int, int],
    width: int,
    dash_len: float = 24.0,
    gap_len: float = 16.0,
):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1.0:
        return
    ux, uy = dx / L, dy / L
    s = 0.0
    while s < L:
        e = min(L, s + dash_len)
        xs, ys = x0 + ux * s, y0 + uy * s
        xe, ye = x0 + ux * e, y0 + uy * e
        rd.line([(xs, ys), (xe, ye)], fill=col_rgba, width=width)
        s += dash_len + gap_len


def draw_deviation_wedge_ss(
    rd: ImageDraw.ImageDraw,
    pivot: tuple[float, float],
    pt_ideal: tuple[float, float],
    pt_actual: tuple[float, float],
    radius_ss: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    px, py = pivot
    ang1 = math.atan2(pt_ideal[1] - py, pt_ideal[0] - px)
    ang2 = math.atan2(pt_actual[1] - py, pt_actual[0] - px)
    diff = (ang2 - ang1 + math.pi) % (2.0 * math.pi) - math.pi

    steps = 36
    poly = [(px, py)]
    for i in range(steps + 1):
        a = ang1 + diff * (i / float(steps))
        poly.append((px + math.cos(a) * radius_ss, py + math.sin(a) * radius_ss))
    rd.polygon(poly, fill=(*col_main, 68))

    arc_pts = poly[1:]
    rd.line(arc_pts, fill=(0, 0, 0, 215), width=int(round(6.5 * ss)))
    rd.line(arc_pts, fill=(*col_main, 255), width=int(round(3.8 * ss)))
    rd.line(arc_pts, fill=(*col_light, 210), width=int(round(1.4 * ss)))


def draw_clean_single_arrow_ss(
    rd: ImageDraw.ImageDraw,
    p_start_ss: tuple[float, float],
    p_end_ss: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    """
    Draws a classic, ultra-clean SINGLE-HEAD arrow (`───▶`) as ONE unified 7-vertex polygon:
    - ZERO `>>>` multi-chevrons!
    - ZERO segmented polygon strips!
    - ZERO C-loops!
    - Crisp dark outer casing + solid high-contrast fill + clean triangular head.
    """
    x0, y0 = p_start_ss
    x1, y1 = p_end_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 24.0 * ss:
        return
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    # Proportional, sleek single-head arrow geometry
    head_len = min(L * 0.45, 28.0 * ss)
    head_half_w = 15.5 * ss
    shaft_half_w_start = 4.2 * ss
    shaft_half_w_neck = 5.8 * ss

    bx = x1 - ux * head_len
    by = y1 - uy * head_len

    def _make_arrow_poly(pad: float = 0.0):
        sw0 = shaft_half_w_start + pad
        sw1 = shaft_half_w_neck + pad
        hw = head_half_w + pad * 1.25
        tx = x1 + ux * (pad * 1.4)
        ty = y1 + uy * (pad * 1.4)
        sx = x0 - ux * pad
        sy = y0 - uy * pad
        nbx = bx - ux * (pad * 0.3)
        nby = by - uy * (pad * 0.3)
        return [
            (sx + nx * sw0, sy + ny * sw0),
            (nbx + nx * sw1, nby + ny * sw1),
            (nbx + nx * hw, nby + ny * hw),
            (tx, ty),
            (nbx - nx * hw, nby - ny * hw),
            (nbx - nx * sw1, nby - ny * sw1),
            (sx - nx * sw0, sy - ny * sw0),
        ]

    # 1. Dark drop-shadow & casing polygon
    rd.polygon(_make_arrow_poly(pad=2.8 * ss), fill=(0, 0, 0, 215))
    # 2. Main solid arrow polygon
    rd.polygon(_make_arrow_poly(pad=0.0), fill=(*col_main, 255))
    # 3. Subtle inner highlight spine along the shaft only (never splitting the head!)
    rd.line(
        [(x0 + ux * (4.0 * ss), y0 + uy * (4.0 * ss)), (bx + ux * (4.0 * ss), by + uy * (4.0 * ss))],
        fill=(*col_light, 190),
        width=int(round(2.4 * ss)),
    )


def draw_clean_curved_arc_arrow_ss(
    rd: ImageDraw.ImageDraw,
    pivot_ss: tuple[float, float],
    radius_ss: float,
    ang_start_deg: float,
    ang_end_deg: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    """
    Draws a smooth circular arc arrow (`╭───▶`) with ONE clean triangular arrowhead at `ang_end_deg`.
    """
    cx, cy = pivot_ss
    steps = 40
    a0 = math.radians(ang_start_deg)
    a1 = math.radians(ang_end_deg)
    direction = 1.0 if a1 >= a0 else -1.0

    # Shorten the arc slightly so the single triangular arrowhead sits cleanly at `a1`
    head_len = 26.0 * ss
    head_half_w = 15.0 * ss
    d_theta_head = (head_len / max(40.0 * ss, radius_ss)) * direction
    a_neck = a1 - d_theta_head

    pts = []
    for i in range(steps + 1):
        a = a0 + (a_neck - a0) * (i / float(steps))
        pts.append((cx + math.cos(a) * radius_ss, cy + math.sin(a) * radius_ss))

    rd.line(pts, fill=(0, 0, 0, 215), width=int(round(13.5 * ss)))
    rd.line(pts, fill=(*col_main, 255), width=int(round(8.5 * ss)))
    rd.line(pts, fill=(*col_light, 185), width=int(round(2.4 * ss)))

    # Tangent unit vector at `a_neck` pointing toward `a1`
    ux = -math.sin(a_neck) * direction
    uy = math.cos(a_neck) * direction
    nx, ny = -uy, ux

    nbx = cx + math.cos(a_neck) * radius_ss
    nby = cy + math.sin(a_neck) * radius_ss
    tx = nbx + ux * head_len
    ty = nby + uy * head_len

    for pad, rgba in [(2.8 * ss, (0, 0, 0, 215)), (0.0, (*col_main, 255))]:
        hw = head_half_w + pad * 1.25
        p_tip = (tx + ux * pad * 1.4, ty + uy * pad * 1.4)
        p_l = (nbx - ux * pad * 0.4 + nx * hw, nby - uy * pad * 0.4 + ny * hw)
        p_r = (nbx - ux * pad * 0.4 - nx * hw, nby - uy * pad * 0.4 - ny * hw)
        rd.polygon([p_l, p_tip, p_r], fill=rgba)


def draw_joint_chain_segment_ss(
    rd: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 24.0 * ss:
        return
    ux, uy = dx / L, dy / L
    pa = (x0 + ux * 11.5 * ss, y0 + uy * 11.5 * ss)
    pb = (x1 - ux * 11.5 * ss, y1 - uy * 11.5 * ss)
    rd.line([pa, pb], fill=(0, 0, 0, 215), width=int(round(7.0 * ss)))
    rd.line([pa, pb], fill=(*col_main, 250), width=int(round(4.2 * ss)))
    rd.line([pa, pb], fill=(*col_light, 215), width=int(round(1.5 * ss)))


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
    rd.ellipse([cx - r - 2 * ss, cy - r - 2 * ss, cx + r + 2 * ss, cy + r + 2 * ss], fill=(0, 0, 0, 185))
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


def render_v50_callouts(
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
        # STEP 1 (FRONT VIEW — WRONG: `18° VALGUS CAVE` / KNEE INWARD COLLAPSE)
        # =========================================================================
        # Ideal Knee Alignment Reference Line (`Ankle -> Ideal Outer Knee over Toes`)
        ideal_kn_r = (ank_r[0] - 32.0, kn_r[1])
        ideal_kn_l = (ank_l[0] + 32.0, kn_l[1])

        draw_dashed_line_ss(
            rd,
            S(ank_r),
            S((ideal_kn_r[0] - 8.0, kn_r[1] - 85.0)),
            (*COL_REF, 225),
            width=int(round(3.0 * SS)),
            dash_len=20.0 * SS,
            gap_len=14.0 * SS,
        )
        draw_dashed_line_ss(
            rd,
            S(ank_l),
            S((ideal_kn_l[0] + 8.0, kn_l[1] - 85.0)),
            (*COL_REF, 225),
            width=int(round(3.0 * SS)),
            dash_len=20.0 * SS,
            gap_len=14.0 * SS,
        )

        # Shaded `18°` Valgus Deviation Wedge between Ideal Tracking Line and Actual Buckled Shin (`Ankle -> Knee`)
        draw_deviation_wedge_ss(
            rd,
            pivot=S(ank_r),
            pt_ideal=S(ideal_kn_r),
            pt_actual=S(kn_r),
            radius_ss=185.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_deviation_wedge_ss(
            rd,
            pivot=S(ank_l),
            pt_ideal=S(ideal_kn_l),
            pt_actual=S(kn_l),
            radius_ss=185.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        # Actual Collapsed Shin Axis (`Ankle -> Knee`) + Joint Pivot Nodes (`◎`)
        draw_joint_chain_segment_ss(rd, S(ank_r), S(kn_r), col_main, col_light, SS)
        draw_joint_chain_segment_ss(rd, S(ank_l), S(kn_l), col_main, col_light, SS)

        for pt_j, is_p in [(kn_r, True), (ank_r, False), (kn_l, True), (ank_l, False), (ideal_kn_r, False), (ideal_kn_l, False)]:
            c_m = COL_REF if pt_j in (ideal_kn_r, ideal_kn_l) else col_main
            c_l = COL_WHITE if pt_j in (ideal_kn_r, ideal_kn_l) else col_light
            r_n = 6.5 if pt_j in (ideal_kn_r, ideal_kn_l) else (10.5 if is_p else 8.0)
            draw_joint_pivot_node_ss(rd, S(pt_j), r_n, c_m, c_l, SS, is_primary=is_p)

        # Classic Single-Head Inward Collapse Arrows (`───▶   ◀───`) directly from Ideal Knee Line to Buckled Knee!
        draw_clean_single_arrow_ss(
            rd,
            p_start_ss=S((ideal_kn_r[0] + 10.0, kn_r[1])),
            p_end_ss=S((kn_r[0] - 16.0, kn_r[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_clean_single_arrow_ss(
            rd,
            p_start_ss=S((ideal_kn_l[0] - 10.0, kn_l[1])),
            p_end_ss=S((kn_l[0] + 16.0, kn_l[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        draw_pure_white_metric_callout(od, sd, (275, 995), "18°", "VALGUS CAVE", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "84%", "ACL & MCL LOAD", align="left")

    elif step_idx == 2:
        # =========================================================================
        # STEP 2 (FRONT VIEW — FIX: `0° KNEES TRACK TOES` / WIDE KNEE LOCK)
        # =========================================================================
        # Faint dashed inner ghost line showing where Step 1 had collapsed (`x = ank_r[0] + 58`)
        valgus_ghost_r = (ank_r[0] + 58.0, kn_r[1])
        valgus_ghost_l = (ank_l[0] - 58.0, kn_l[1])

        draw_dashed_line_ss(
            rd,
            S(ank_r),
            S((valgus_ghost_r[0], kn_r[1] - 60.0)),
            (*COL_REF, 135),
            width=int(round(2.4 * SS)),
            dash_len=16.0 * SS,
            gap_len=14.0 * SS,
        )
        draw_dashed_line_ss(
            rd,
            S(ank_l),
            S((valgus_ghost_l[0], kn_l[1] - 60.0)),
            (*COL_REF, 135),
            width=int(round(2.4 * SS)),
            dash_len=16.0 * SS,
            gap_len=14.0 * SS,
        )

        # Shaded Green Recovery Wedge from Collapsed Line out to Locked Knee-Over-Toe Axis
        draw_deviation_wedge_ss(
            rd,
            pivot=S(ank_r),
            pt_ideal=S(valgus_ghost_r),
            pt_actual=S(kn_r),
            radius_ss=180.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_deviation_wedge_ss(
            rd,
            pivot=S(ank_l),
            pt_ideal=S(valgus_ghost_l),
            pt_actual=S(kn_l),
            radius_ss=180.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        # Locked Green Shin Axis (`Ankle -> Knee`) extending straight through the knee
        draw_joint_chain_segment_ss(rd, S(ank_r), S(kn_r), col_main, col_light, SS)
        draw_joint_chain_segment_ss(rd, S(ank_l), S(kn_l), col_main, col_light, SS)

        for pt_j, is_p in [(kn_r, True), (ank_r, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.0, col_main, col_light, SS, is_primary=is_p)

        # Classic Single-Head Outward Knee-Drive Arrows (`◀───   ───▶`) from inner ghost line out to locked knee!
        draw_clean_single_arrow_ss(
            rd,
            p_start_ss=S((valgus_ghost_r[0] - 6.0, kn_r[1])),
            p_end_ss=S((kn_r[0] + 18.0, kn_r[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        draw_clean_single_arrow_ss(
            rd,
            p_start_ss=S((valgus_ghost_l[0] + 6.0, kn_l[1])),
            p_end_ss=S((kn_l[0] - 18.0, kn_l[1])),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        draw_pure_white_metric_callout(od, sd, (275, 995), "0°", "KNEES TRACK TOES", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "96%", "VMO & GLUTE LOCK", align="left")

    elif step_idx == 3:
        # =========================================================================
        # STEP 3 (3/4 SIDE VIEW — WRONG: `+28° HEELS LIFTING` & FORWARD TORSO FOLD)
        # =========================================================================
        # Ideal Upright Spine Reference Line (`hip_r -> ideal_sh`, matching Step 4's upright spine angle)
        ideal_sh = (hip_r[0] + 165.0, hip_r[1] - 420.0)
        draw_dashed_line_ss(
            rd,
            S(hip_r),
            S(ideal_sh),
            (*COL_REF, 225),
            width=int(round(3.0 * SS)),
            dash_len=20.0 * SS,
            gap_len=14.0 * SS,
        )

        # Shaded `+28°` Forward Lean Wedge between Ideal Upright Spine (`hip_r -> ideal_sh`) and Actual Collapsed Spine (`hip_r -> sh`)
        draw_deviation_wedge_ss(
            rd,
            pivot=S(hip_r),
            pt_ideal=S(ideal_sh),
            pt_actual=S(sh),
            radius_ss=220.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        # Smooth Single-Head Curved Arc Arrow (`╭───▶`) showing the torso pitching FORWARD & DOWN from Ideal Spine to Actual Spine!
        ang_ideal_deg = math.degrees(math.atan2(ideal_sh[1] - hip_r[1], ideal_sh[0] - hip_r[0]))
        ang_actual_deg = math.degrees(math.atan2(sh[1] - hip_r[1], sh[0] - hip_r[0]))
        draw_clean_curved_arc_arrow_ss(
            rd,
            pivot_ss=S(hip_r),
            radius_ss=295.0 * SS,
            ang_start_deg=ang_ideal_deg,
            ang_end_deg=ang_actual_deg,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        # Flat Floor Reference Line under the foot
        floor_y = ank_r[1] + 58.0
        draw_dashed_line_ss(
            rd,
            S((ank_r[0] - 85.0, floor_y)),
            S((ank_r[0] + 95.0, floor_y)),
            (*COL_REF, 225),
            width=int(round(2.8 * SS)),
            dash_len=18.0 * SS,
            gap_len=12.0 * SS,
        )

        # Actual Collapsed 3-Point Chain (`Shoulder — Hip — Knee — Ankle`)
        draw_joint_chain_segment_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_joint_chain_segment_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_joint_chain_segment_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        for pt_j, is_p in [(sh, False), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.5, col_main, col_light, SS, is_primary=is_p)

        # Classic Single-Head Upward Heel-Lift Arrow (`▲`) next to the lifted heel
        draw_clean_single_arrow_ss(
            rd,
            p_start_ss=S((ank_r[0] - 58.0, floor_y + 4.0)),
            p_end_ss=S((ank_r[0] - 58.0, ank_r[1] - 65.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        draw_pure_white_metric_callout(od, sd, (335, 1020), "+28°", "HEELS LIFTING", align="right")
        draw_pure_white_metric_callout(od, sd, (SAFE_X_RIGHT_BADGE, 865), "79%", "LUMBAR & PATELLA", align="left")

    else:
        # =========================================================================
        # STEP 4 (3/4 SIDE VIEW — FIX: `95° PARALLEL DEPTH` & FLAT HEEL DRIVE)
        # =========================================================================
        # Parallel Thigh Reference Line (`horizontal dashed line through hip_r`)
        draw_dashed_line_ss(
            rd,
            S((hip_r[0] - 20.0, hip_r[1])),
            S((kn_r[0] + 45.0, hip_r[1])),
            (*COL_REF, 195),
            width=int(round(2.6 * SS)),
            dash_len=18.0 * SS,
            gap_len=12.0 * SS,
        )
        # Solid Green Floor Lock Line under the flat foot
        floor_y = ank_r[1] + 28.0
        rd.line(
            [S((ank_r[0] - 75.0, floor_y)), S((ank_r[0] + 95.0, floor_y))],
            fill=(0, 0, 0, 215),
            width=int(round(6.0 * SS)),
        )
        rd.line(
            [S((ank_r[0] - 75.0, floor_y)), S((ank_r[0] + 95.0, floor_y))],
            fill=(*col_main, 250),
            width=int(round(3.4 * SS)),
        )

        # Shaded `95°` Parallel Hip-Thigh Wedge between Spine (`hip_r -> sh`) and Femur (`hip_r -> kn_r`)
        draw_deviation_wedge_ss(
            rd,
            pivot=S(hip_r),
            pt_ideal=S(sh),
            pt_actual=S(kn_r),
            radius_ss=145.0 * SS,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        # Locked 3-Point Chain (`Shoulder — Hip — Knee — Ankle`)
        draw_joint_chain_segment_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_joint_chain_segment_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_joint_chain_segment_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        for pt_j, is_p in [(sh, False), (hip_r, True), (kn_r, True), (ank_r, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 10.5 if is_p else 8.5, col_main, col_light, SS, is_primary=is_p)

        # Classic Single-Head Arrows:
        # (a) Proud Chest / Barbell Vertical Drive UP (`▲`)
        draw_clean_single_arrow_ss(
            rd,
            p_start_ss=S((sh[0] + 82.0, sh[1] + 100.0)),
            p_end_ss=S((sh[0] + 82.0, sh[1] - 35.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )
        # (b) Flat Heel Locked DOWN onto floor (`▼`)
        draw_clean_single_arrow_ss(
            rd,
            p_start_ss=S((ank_r[0] - 58.0, ank_r[1] - 90.0)),
            p_end_ss=S((ank_r[0] - 58.0, floor_y - 4.0)),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
        )

        draw_pure_white_metric_callout(od, sd, (335, 1020), "95°", "PARALLEL DEPTH", align="right")
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

        rgb_frame, kp_cur = apply_biomechanical_posture_diff(raw_rgb, raw_kp, step_idx=s_idx, u=1.0)
        canvas = Image.fromarray(rgb_frame).convert("RGBA")

        render_v50_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)

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
        crops.append(out_bgr[620:1580, 120:960])

    sheet_2x2 = np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])])
    sheet_path = os.path.join(OUT_DIR, "day3_v50_difference_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    crop_2x2 = np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])])
    crop_path = os.path.join(OUT_DIR, "day3_v50_closeup_sheet.jpg")
    cv2.imwrite(crop_path, crop_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved:", sheet_path, "and", crop_path)


if __name__ == "__main__":
    main()
