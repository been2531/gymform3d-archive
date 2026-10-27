#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v56_minimal_laser_master.py`)
==================================================================
Addresses 중현 님's exact feedback (`t232u`: "화살표를 입체적으로 했다고 했는데 촌스러운 느낌이야"):
1. DELETES all thick 2.5D/3D clip-art ribbon arrows and chunky filled triangle arrowheads completely!
2. Implements Pure `@demicstory` + Day 2 (`v39`) Minimalist Architectural Laser Telemetry (`SS=4`):
   - Slender `2.8px` glowing anatomical bone-axis laser lines (`Hip ◎ — Knee ◎ — Ankle ◎` & `Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`)
     with a `1.0px` pure white center filament and soft outer neon bloom (ZERO heavy black cartoon outlines!).
   - Clean, slender `3.0px` Day 2 exterior solid protractor arcs (`14°`, `0°`, `+28°`, `95°`) with subtle translucent
     wedge fill (`alpha=28`), capped with a delicate, razor-sharp open stroke micro-chevron (`>`).
   - Minimalist `@demicstory` architectural double micro-chevron ticks (`»` / `«` / `︿` made of `2.5px` glowing open strokes,
     zero thick shaft/ribbon body!) integrated directly onto the joint alignment gauge.
   - Glowing emerald flat-foot floor laser bar flush under the planted soles (`demic_style_ref.jpg` signature).
3. Preserves the realistic, unwarped 3D squat clips (`0%` `cv2.remap`, `0%` background wobble), Day 2 muscle fiber &
   3D bone X-Ray glow, and strict iPhone Reels Safe-Zone framing (`head_top >= 430`, `foot_bot <= 1285`).
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
import build_v39_ig_safe_zone as v39
import preview_v55_demic_day2_master as p55

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_V55_PKL = "/workspace/daily_info_reels/_cache_day3_v55_pools.pkl"
W, H = 1080, 1920

SAFE_X_LEFT = 96
SAFE_X_RIGHT_MAX = 978
SAFE_Y_TOP = 258

COL_RED = (250, 44, 54)
COL_RED_LIGHT = (255, 195, 192)

COL_GREEN = (18, 232, 126)
COL_GREEN_LIGHT = (190, 255, 222)

COL_WHITE = (255, 255, 255)
COL_PLUMB = (210, 226, 245)

F_CALLOUT_NUM = v30.f_barlow(64, extrabold=True)
F_CALLOUT_SUB = v30.f_pretendard(22, "extrabold")

# Precompute a 100% static radial studio key-light contrast field (constant across all frames within a step
# so intra-step background diff == 0.0, while giving rich Crimson vs Emerald studio atmosphere across steps!)
_YY, _XX = np.mgrid[:H, :W].astype(np.float32)
_RADIAL_STAGE = np.exp(-(((_XX - 540.0) / 360.0) ** 2 + ((_YY - 860.0) / 420.0) ** 2))[:, :, None]
_TINT_WRONG = np.array([15.0, -7.0, -7.0], dtype=np.float32)[None, None, :] * _RADIAL_STAGE
_TINT_RIGHT = np.array([-11.0, 13.0, 11.0], dtype=np.float32)[None, None, :] * _RADIAL_STAGE


def apply_static_studio_tint(rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    """Applies a 100% static studio key-light tint (0.0 frame-to-frame change within a step)."""
    out = rgb.astype(np.float32) + (_TINT_WRONG if is_wrong else _TINT_RIGHT)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def draw_smooth_segment_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    half_w_ss: float,
    rgba: tuple[int, int, int, int],
    round_caps: bool = True,
):
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 0.5:
        return
    nx, ny = -dy / L, dx / L
    poly = [
        (x0 + nx * half_w_ss, y0 + ny * half_w_ss),
        (x1 + nx * half_w_ss, y1 + ny * half_w_ss),
        (x1 - nx * half_w_ss, y1 - ny * half_w_ss),
        (x0 - nx * half_w_ss, y0 - ny * half_w_ss),
    ]
    rd.polygon(poly, fill=rgba)
    if round_caps and half_w_ss >= 1.5:
        r = half_w_ss
        rd.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=rgba)
        rd.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=rgba)


def draw_minimal_bone_laser_ss(
    rd: ImageDraw.ImageDraw,
    bd_1x: ImageDraw.ImageDraw,
    p0_1x: tuple[float, float],
    p1_1x: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    trim_ends_1x: float = 10.0,
):
    """
    `@demicstory` Minimalist Glowing Bone-Axis Laser Line:
    - Trimmed cleanly outside the `◎` joint pivot rings (`trim_ends_1x = 10.0px`).
    - Soft outer bloom on `bd_1x` + crisp `2.8px` core + `1.0px` white-hot center laser filament on `rd`.
    - ZERO heavy black cartoon border!
    """
    x0, y0 = p0_1x
    x1, y1 = p1_1x
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L <= 2.0 * trim_ends_1x + 2.0:
        return
    ux, uy = dx / L, dy / L
    pa_1x = (x0 + ux * trim_ends_1x, y0 + uy * trim_ends_1x)
    pb_1x = (x1 - ux * trim_ends_1x, y1 - uy * trim_ends_1x)

    # Soft bloom halo in 1x (will be Gaussian blurred)
    bd_1x.line([pa_1x, pb_1x], fill=(*col_main, 95), width=8)

    pa_ss = (pa_1x[0] * ss, pa_1x[1] * ss)
    pb_ss = (pb_1x[0] * ss, pb_1x[1] * ss)
    draw_smooth_segment_ss(rd, pa_ss, pb_ss, 2.8 * ss, (*col_main, 65))
    draw_smooth_segment_ss(rd, pa_ss, pb_ss, 1.45 * ss, (*col_main, 248))
    draw_smooth_segment_ss(rd, pa_ss, pb_ss, 0.52 * ss, (255, 255, 255, 225))


def draw_minimal_reference_hairline_ss(
    rd: ImageDraw.ImageDraw,
    p0_1x: tuple[float, float],
    p1_1x: tuple[float, float],
    ss: int = 4,
    dashed: bool = True,
):
    """
    Clinical `1.4px` silver-white reference plumb / parallel hairline (`#D2E2F5`).
    """
    p0_ss = (p0_1x[0] * ss, p0_1x[1] * ss)
    p1_ss = (p1_1x[0] * ss, p1_1x[1] * ss)
    if not dashed:
        draw_smooth_segment_ss(rd, p0_ss, p1_ss, 0.75 * ss, (*COL_PLUMB, 185))
        return
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 8.0:
        return
    n_dashes = max(5, int(L // (13.0 * ss)))
    for i in range(n_dashes):
        t0 = i / float(n_dashes)
        t1 = min(1.0, (i + 0.55) / float(n_dashes))
        pa = (x0 + dx * t0, y0 + dy * t0)
        pb = (x0 + dx * t1, y0 + dy * t1)
        draw_smooth_segment_ss(rd, pa, pb, 0.72 * ss, (*COL_PLUMB, 195))


def draw_open_micro_chevron_ss(
    rd: ImageDraw.ImageDraw,
    bd_1x: ImageDraw.ImageDraw | None,
    tip_1x: tuple[float, float],
    dir_ux: float,
    dir_uy: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    wing_len_1x: float = 9.5,
    half_span_1x: float = 6.2,
    stroke_hw_1x: float = 1.25,
    alpha_scale: float = 1.0,
    ss: int = 4,
):
    """
    Draws a delicate, razor-sharp OPEN stroke micro-chevron (`>`) pointing along `(dir_ux, dir_uy)`:
    - ZERO filled triangle body!
    - ZERO black cartoon border!
    - Just two slender intersecting laser strokes (`2.5px` width) with a subtle white center filament.
    """
    norm = math.hypot(dir_ux, dir_uy)
    if norm < 1e-5:
        return
    ux, uy = dir_ux / norm, dir_uy / norm
    nx, ny = -uy, ux

    tx, ty = tip_1x
    bx, by = tx - ux * wing_len_1x, ty - uy * wing_len_1x
    w1_1x = (bx + nx * half_span_1x, by + ny * half_span_1x)
    w2_1x = (bx - nx * half_span_1x, by - ny * half_span_1x)

    if bd_1x is not None:
        bd_1x.line([w1_1x, (tx, ty), w2_1x], fill=(*col_main, int(85 * alpha_scale)), width=6)

    tip_ss = (tx * ss, ty * ss)
    w1_ss = (w1_1x[0] * ss, w1_1x[1] * ss)
    w2_ss = (w2_1x[0] * ss, w2_1x[1] * ss)

    hw_glow = (stroke_hw_1x + 0.9) * ss
    hw_core = stroke_hw_1x * ss
    hw_spec = 0.45 * ss

    a_glow = int(round(65 * alpha_scale))
    a_core = int(round(250 * alpha_scale))
    a_spec = int(round(220 * alpha_scale))

    for wing_ss in (w1_ss, w2_ss):
        draw_smooth_segment_ss(rd, wing_ss, tip_ss, hw_glow, (*col_main, a_glow))
    for wing_ss in (w1_ss, w2_ss):
        draw_smooth_segment_ss(rd, wing_ss, tip_ss, hw_core, (*col_main, a_core))
    for wing_ss in (w1_ss, w2_ss):
        draw_smooth_segment_ss(rd, wing_ss, tip_ss, hw_spec, (255, 255, 255, a_spec))


def draw_minimal_double_chevron_cue_ss(
    rd: ImageDraw.ImageDraw,
    bd_1x: ImageDraw.ImageDraw,
    anchor_1x: tuple[float, float],
    dir_ux: float,
    dir_uy: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    spacing_1x: float = 7.5,
    wing_len_1x: float = 8.0,
    half_span_1x: float = 5.2,
    ss: int = 4,
):
    """
    `@demicstory` Pure Shaftless Double Micro-Chevron Tick (`»` / `«` / `︿`):
    - ONLY two delicate open stroke chevrons (`2.1px` laser strokes), ZERO shaft line, ZERO ribbon body!
    """
    norm = math.hypot(dir_ux, dir_uy)
    if norm < 1e-5:
        return
    ux, uy = dir_ux / norm, dir_uy / norm

    trail_tip_1x = (anchor_1x[0] - ux * spacing_1x, anchor_1x[1] - uy * spacing_1x)
    draw_open_micro_chevron_ss(
        rd, bd_1x, trail_tip_1x, ux, uy, col_main, col_light,
        wing_len_1x=wing_len_1x * 0.88, half_span_1x=half_span_1x * 0.88,
        stroke_hw_1x=0.95, alpha_scale=0.60, ss=ss,
    )
    draw_open_micro_chevron_ss(
        rd, bd_1x, anchor_1x, ux, uy, col_main, col_light,
        wing_len_1x=wing_len_1x, half_span_1x=half_span_1x,
        stroke_hw_1x=1.08, alpha_scale=0.96, ss=ss,
    )


def draw_minimal_protractor_arc_ss(
    rd: ImageDraw.ImageDraw,
    bd_1x: ImageDraw.ImageDraw,
    vertex_1x: tuple[float, float],
    ray1_pt_1x: tuple[float, float],
    ray2_pt_1x: tuple[float, float],
    radius_1x: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    show_micro_chevron: bool = True,
):
    """
    Day 2 (`v39`) Slender Exterior Solid Protractor Arc (`3.0px` stroke + `alpha=28` glass sector fill),
    capped with a delicate open micro-chevron (`>`) tangent to the arc at `ray2`.
    """
    vx, vy = vertex_1x[0] * ss, vertex_1x[1] * ss
    r = radius_1x * ss
    a1 = math.atan2(ray1_pt_1x[1] * ss - vy, ray1_pt_1x[0] * ss - vx)
    a2 = math.atan2(ray2_pt_1x[1] * ss - vy, ray2_pt_1x[0] * ss - vx)
    diff = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(diff) < 0.03:
        return

    steps = 64
    # 1. Subtle clinical glass sector wedge (`alpha=28`)
    poly = [(vx, vy)]
    for i in range(steps + 1):
        a = a1 + diff * (i / float(steps))
        poly.append((vx + math.cos(a) * r, vy + math.sin(a) * r))
    rd.polygon(poly, fill=(*col_main, 28))

    # 2. Slender 3.0px smooth annular arc (`hw = 1.5px`) + 0.5px white center filament, ZERO black border!
    outer_glow, inner_glow = [], []
    outer_core, inner_core = [], []
    outer_spec, inner_spec = [], []
    pts_1x = []
    for i in range(steps + 1):
        a = a1 + diff * (i / float(steps))
        ca, sa = math.cos(a), math.sin(a)
        pts_1x.append((vertex_1x[0] + ca * radius_1x, vertex_1x[1] + sa * radius_1x))
        outer_glow.append((vx + ca * (r + 2.6 * ss), vy + sa * (r + 2.6 * ss)))
        inner_glow.append((vx + ca * (r - 2.6 * ss), vy + sa * (r - 2.6 * ss)))
        outer_core.append((vx + ca * (r + 1.45 * ss), vy + sa * (r + 1.45 * ss)))
        inner_core.append((vx + ca * (r - 1.45 * ss), vy + sa * (r - 1.45 * ss)))
        outer_spec.append((vx + ca * (r + 0.48 * ss), vy + sa * (r + 0.48 * ss)))
        inner_spec.append((vx + ca * (r - 0.48 * ss), vy + sa * (r - 0.48 * ss)))

    bd_1x.line(pts_1x, fill=(*col_main, 80), width=6)
    rd.polygon(outer_glow + inner_glow[::-1], fill=(*col_main, 60))
    rd.polygon(outer_core + inner_core[::-1], fill=(*col_main, 248))
    rd.polygon(outer_spec + inner_spec[::-1], fill=(255, 255, 255, 215))

    # 3. Delicate open stroke micro-chevron (`>`) tangent to the arc at `a2`
    if show_micro_chevron and abs(diff) * radius_1x >= 14.0:
        direction = 1.0 if diff >= 0.0 else -1.0
        a_end = a1 + diff
        tip_1x = (vertex_1x[0] + math.cos(a_end) * radius_1x, vertex_1x[1] + math.sin(a_end) * radius_1x)
        # Tangent vector along arc direction at `a_end`
        t_ux = -math.sin(a_end) * direction
        t_uy = math.cos(a_end) * direction
        draw_open_micro_chevron_ss(
            rd, bd_1x, tip_1x, t_ux, t_uy, col_main, col_light,
            wing_len_1x=8.5, half_span_1x=5.4, stroke_hw_1x=1.18, alpha_scale=1.0, ss=ss,
        )


def draw_minimal_joint_reticle_ss(
    rd: ImageDraw.ImageDraw,
    bd_1x: ImageDraw.ImageDraw,
    pt_1x: tuple[float, float],
    radius_1x: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    is_primary: bool = True,
):
    """
    Concentric Precision Joint Reticle (`◎`):
    - Subtle dark inner lens (`alpha=135`) + `2.0px` glowing ring + `2.5px` white center pinpoint dot.
    """
    x, y = pt_1x
    r_1x = radius_1x
    bd_1x.ellipse([x - r_1x - 2, y - r_1x - 2, x + r_1x + 2, y + r_1x + 2], fill=(*col_main, 70))

    cx, cy = x * ss, y * ss
    r = r_1x * ss
    rd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(8, 10, 14, 145))
    rd.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=(*col_main, 250),
        width=int(round((2.1 if is_primary else 1.7) * ss)),
    )
    r_dot = (2.7 if is_primary else 2.1) * ss
    rd.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot], fill=(255, 255, 255, 255))


def draw_minimal_floor_laser_ss(
    rd: ImageDraw.ImageDraw,
    bd_1x: ImageDraw.ImageDraw,
    ank_r_1x: tuple[float, float],
    ank_l_1x: tuple[float, float] | None,
    y_floor_1x: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    is_lifting: bool = False,
    ss: int = 4,
):
    """
    `@demicstory` Minimalist Floor-Contact Laser Bar (`demic_style_ref.jpg` bottom-right squat reference):
    - Flush horizontal glowing laser bar under the sole(s) (`2.5px` core + soft neon bloom).
    """
    anchors = [ank_r_1x] if ank_l_1x is None else [ank_r_1x, ank_l_1x]
    for ax_1x, _ in anchors:
        p0_1x = (ax_1x - 46.0, y_floor_1x)
        p1_1x = (ax_1x + 46.0, y_floor_1x)
        if is_lifting:
            draw_minimal_reference_hairline_ss(rd, p0_1x, p1_1x, ss=ss, dashed=True)
        else:
            bd_1x.line([p0_1x, p1_1x], fill=(*col_main, 95), width=7)
            p0_ss = (p0_1x[0] * ss, p0_1x[1] * ss)
            p1_ss = (p1_1x[0] * ss, p1_1x[1] * ss)
            draw_smooth_segment_ss(rd, p0_ss, p1_ss, 2.5 * ss, (*col_main, 65))
            draw_smooth_segment_ss(rd, p0_ss, p1_ss, 1.35 * ss, (*col_main, 250))
            draw_smooth_segment_ss(rd, p0_ss, p1_ss, 0.50 * ss, (255, 255, 255, 230))


def draw_minimal_pointer_and_callout(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    joint_pt: tuple[float, float],
    elbow_pt: tuple[float, float],
    anchor_pt: tuple[int, int],
    num_str: str,
    sub_str: str,
    col_main: tuple[int, int, int],
    align: str = "left",
):
    """
    Day 2 (`v39`) `1.5px` slender architectural leader line + Pure White (`#FFFFFF`) `64pt`/`22pt` callout.
    """
    p_j = (int(round(joint_pt[0])), int(round(joint_pt[1])))
    p_e = (int(round(elbow_pt[0])), int(round(elbow_pt[1])))
    sd.line([p_j, p_e], fill=(0, 0, 0, 185), width=3)
    od.line([p_j, p_e], fill=(*col_main, 210), width=2)
    od.ellipse([p_e[0] - 3, p_e[1] - 3, p_e[0] + 3, p_e[1] + 3], fill=(255, 255, 255, 240))

    ax, ay = anchor_pt
    nb = od.textbbox((0, 0), num_str, font=F_CALLOUT_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), sub_str, font=F_CALLOUT_SUB)
    tw = tb[2] - tb[0]

    if align == "right":
        nx = min(SAFE_X_RIGHT_MAX, ax) - nw
        tx = min(SAFE_X_RIGHT_MAX, ax) - tw
    else:
        nx = max(SAFE_X_LEFT, ax)
        tx = max(SAFE_X_LEFT, ax)

    nx = int(np.clip(nx, SAFE_X_LEFT, SAFE_X_RIGHT_MAX - nw))
    tx = int(np.clip(tx, SAFE_X_LEFT, SAFE_X_RIGHT_MAX - tw))
    ty = ay + 66

    for dx_s, dy_s in [(-3, 0), (3, 0), (0, -3), (0, 3), (0, 5)]:
        sd.text((nx + dx_s, ay + dy_s), num_str, font=F_CALLOUT_NUM, fill=(0, 0, 0, 235))
        sd.text((tx + dx_s, ty + dy_s), sub_str, font=F_CALLOUT_SUB, fill=(0, 0, 0, 235))

    od.text((nx, ay), num_str, font=F_CALLOUT_NUM, fill=COL_WHITE)
    od.text((tx, ty), sub_str, font=F_CALLOUT_SUB, fill=COL_WHITE)


def render_v56_minimal_laser_callouts(
    canvas: Image.Image,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 1.0,
):
    """
    Renders Pure `@demicstory` + Day 2 (`v39`) Minimalist Architectural Laser Telemetry (`SS=4`):
    - ZERO floating arrows or thick 2D/3D ribbons!
    - ONLY crisp glowing bone-axis laser lines (`Hip ◎ — Knee ◎ — Ankle ◎` & `Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`),
      clinical reference plumb/parallel hairlines, `@demicstory` floor-contact laser bars, and Day 2's slender
      exterior solid protractor arcs capped with a delicate open micro-chevron (`>`).
    """
    SS = 4
    bloom_1x = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd_1x = ImageDraw.Draw(bloom_1x)

    rig_layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig_layer)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

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
        # Step 1 (Front Wrong — Realistic Knee Valgus `14° VALGUS CAVE`)
        # 1. Vertical Ideal Plumb Reference Hairlines rising from ankles (`ank_r`, `ank_l`)
        plumb_r = (ank_r[0], hip_r[1] + 25.0)
        plumb_l = (ank_l[0], hip_l[1] + 25.0)
        draw_minimal_reference_hairline_ss(rd, ank_r, plumb_r, SS, dashed=True)
        draw_minimal_reference_hairline_ss(rd, ank_l, plumb_l, SS, dashed=True)

        # 2. Slender Day 2 Protractor Arcs (`14°` valgus drift from vertical plumb line, with delicate open micro-chevron)
        draw_minimal_protractor_arc_ss(rd, bd_1x, ank_r, plumb_r, kn_r, 86.0, col_main, col_light, SS, show_micro_chevron=True)
        draw_minimal_protractor_arc_ss(rd, bd_1x, ank_l, plumb_l, kn_l, 86.0, col_main, col_light, SS, show_micro_chevron=True)

        # 3. `@demicstory` Glowing Crimson Bone-Axis Laser Lines (`Hip ◎ — Knee ◎ — Ankle ◎`)
        draw_minimal_bone_laser_ss(rd, bd_1x, hip_r, kn_r, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, kn_r, ank_r, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, hip_l, kn_l, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, kn_l, ank_l, col_main, col_light, SS)

        # 4. Concentric Precision Joint Reticles (`◎`)
        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_minimal_joint_reticle_ss(rd, bd_1x, pt_j, 8.0 if is_p else 6.0, col_main, col_light, SS, is_primary=is_p)

        deg_live = int(round(8 + 6 * rep_amount))
        pct_live = int(round(79 + 5 * rep_amount))
        draw_minimal_pointer_and_callout(od, sd, (kn_r[0] - 10, kn_r[1] - 6), (265, 965), (SAFE_X_LEFT, 910), f"{deg_live}°", "VALGUS CAVE", col_main, align="left")
        draw_minimal_pointer_and_callout(od, sd, (kn_l[0] + 10, kn_l[1] - 6), (765, 915), (SAFE_X_RIGHT_MAX, 855), f"{pct_live}%", "ACL & MCL LOAD", col_main, align="right")

    elif step_idx == 2:
        # Step 2 (Front Right — Knees Track Wide Over Toes `0° KNEES TRACK TOES`)
        # 1. `@demicstory` Glowing Emerald Floor Laser Bars flush under both feet
        draw_minimal_floor_laser_ss(rd, bd_1x, ank_r, ank_l, ank_r[1] + 20.0, col_main, col_light, is_lifting=False, ss=SS)

        # 2. Slender Day 2 Protractor Arcs at both knees (`Hip ◎ — Knee ◎ — Ankle ◎`)
        draw_minimal_protractor_arc_ss(rd, bd_1x, kn_r, hip_r, ank_r, 52.0, col_main, col_light, SS, show_micro_chevron=True)
        draw_minimal_protractor_arc_ss(rd, bd_1x, kn_l, hip_l, ank_l, 52.0, col_main, col_light, SS, show_micro_chevron=True)

        # 3. `@demicstory` Glowing Emerald Bone-Axis Laser Lines (`Hip ◎ — Knee ◎ — Ankle ◎`)
        draw_minimal_bone_laser_ss(rd, bd_1x, hip_r, kn_r, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, kn_r, ank_r, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, hip_l, kn_l, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, kn_l, ank_l, col_main, col_light, SS)

        # 4. Concentric Precision Joint Reticles (`◎`)
        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_minimal_joint_reticle_ss(rd, bd_1x, pt_j, 8.0 if is_p else 6.0, col_main, col_light, SS, is_primary=is_p)

        pct_live = int(round(92 + 4 * rep_amount))
        draw_minimal_pointer_and_callout(od, sd, (kn_r[0] - 10, kn_r[1] - 6), (265, 965), (SAFE_X_LEFT, 910), "0°", "KNEES TRACK TOES", col_main, align="left")
        draw_minimal_pointer_and_callout(od, sd, (kn_l[0] + 10, kn_l[1] - 6), (765, 915), (SAFE_X_RIGHT_MAX, 855), f"{pct_live}%", "VMO & GLUTE LOCK", col_main, align="right")

    elif step_idx == 3:
        # Step 3 (Side Wrong — Realistic Good-Morning Forward Lean `+28°` & Heel Lift)
        floor_y = 1190.0
        draw_minimal_floor_laser_ss(rd, bd_1x, ank_r, None, floor_y, col_main, col_light, is_lifting=True, ss=SS)
        # Vertical gap hairline between lifted ankle and floor reference bar
        draw_minimal_reference_hairline_ss(rd, (ank_r[0], floor_y), ank_r, SS, dashed=True)

        # 1. Vertical Upright Plumb Reference Hairline rising from `hip_r`
        plumb_top = (hip_r[0], sh[1] - 30.0)
        draw_minimal_reference_hairline_ss(rd, hip_r, plumb_top, SS, dashed=True)

        # 2. Slender Day 2 Exterior Protractor Arc (`+28°` forward torso pitch from upright plumb to spine `hip_r -> sh`)
        spine_len = math.hypot(sh[0] - hip_r[0], sh[1] - hip_r[1])
        draw_minimal_protractor_arc_ss(
            rd, bd_1x,
            vertex_1x=hip_r,
            ray1_pt_1x=plumb_top,
            ray2_pt_1x=sh,
            radius_1x=min(96.0, spine_len * 0.68),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_micro_chevron=True,
        )

        # 3. `@demicstory` Glowing Crimson Spine & Leg Laser Chain (`sh — hip_r — kn_r — ank_r`)
        draw_minimal_bone_laser_ss(rd, bd_1x, sh, hip_r, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, hip_r, kn_r, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, kn_r, ank_r, col_main, col_light, SS)

        # 4. Concentric Precision Joint Reticles (`◎`)
        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_minimal_joint_reticle_ss(rd, bd_1x, pt_j, 8.0 if is_p else 6.2, col_main, col_light, SS, is_primary=is_p)

        deg_live = int(round(14 + 14 * rep_amount))
        pct_live = int(round(74 + 5 * rep_amount))
        draw_minimal_pointer_and_callout(od, sd, (hip_r[0] - 10, hip_r[1] + 6), (265, 955), (SAFE_X_LEFT, 900), f"+{deg_live}°", "HEELS LIFTING", col_main, align="left")
        draw_minimal_pointer_and_callout(od, sd, (kn_r[0] + 10, kn_r[1] - 6), (765, 915), (SAFE_X_RIGHT_MAX, 855), f"{pct_live}%", "LUMBAR & PATELLA", col_main, align="right")

    else:
        # Step 4 (Side Right — Proud Upright Chest, Planted Heels & `95°` Parallel Drive)
        floor_y = ank_r[1] + 22.0
        draw_minimal_floor_laser_ss(rd, bd_1x, ank_r, None, floor_y, col_main, col_light, is_lifting=False, ss=SS)

        # 1. Horizontal Parallel Reference Hairline from `hip_r`
        horiz_ref = (kn_r[0] + 24.0, hip_r[1])
        draw_minimal_reference_hairline_ss(rd, hip_r, horiz_ref, SS, dashed=True)

        # 2. Slender Day 2 Exterior Protractor Arc (`95°` Hip-to-Femur Parallel Depth) at `hip_r`
        draw_minimal_protractor_arc_ss(
            rd, bd_1x,
            vertex_1x=hip_r,
            ray1_pt_1x=kn_r,
            ray2_pt_1x=sh,
            radius_1x=72.0,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_micro_chevron=True,
        )

        # 3. `@demicstory` Glowing Emerald Biomechanical Laser Chain (`sh — hip_r — kn_r — ank_r`)
        draw_minimal_bone_laser_ss(rd, bd_1x, sh, hip_r, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, hip_r, kn_r, col_main, col_light, SS)
        draw_minimal_bone_laser_ss(rd, bd_1x, kn_r, ank_r, col_main, col_light, SS)

        # 4. Concentric Precision Joint Reticles (`◎`)
        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_minimal_joint_reticle_ss(rd, bd_1x, pt_j, 8.0 if is_p else 6.2, col_main, col_light, SS, is_primary=is_p)

        deg_live = int(round(125 - 30 * rep_amount))
        pct_live = int(round(94 + 4 * rep_amount))
        draw_minimal_pointer_and_callout(od, sd, (hip_r[0] - 10, hip_r[1] + 6), (265, 955), (SAFE_X_LEFT, 900), f"{deg_live}°", "PARALLEL DEPTH", col_main, align="left")
        draw_minimal_pointer_and_callout(od, sd, (kn_r[0] + 10, kn_r[1] - 6), (765, 915), (SAFE_X_RIGHT_MAX, 855), f"{pct_live}%", "QUADS & GLUTES", col_main, align="right")

    rig_1x = rig_layer.resize((W, H), Image.Resampling.LANCZOS)
    canvas.alpha_composite(bloom_1x.filter(ImageFilter.GaussianBlur(radius=7)))
    canvas.alpha_composite(rig_1x)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(ov)


def main():
    pools = p55.get_or_build_v55_pools()
    torso_icons = p55.make_torso_icons(pools)

    step_configs = [
        (1, pools[1][0], pools[1][1], True,  0, 84, "LUMBAR & KNEE RISK",       "BARBELL BACK SQUAT",       "14° KNEE VALGUS"),
        (2, pools[2][0], pools[2][1], False, 2, 96, "NEUTRAL KNEE LOCK",        "GLUTE MEDIUS & VMO",       "0° KNEES TRACK TOES"),
        (3, pools[3][0], pools[3][1], True,  1, 79, "ANTERIOR FORCE LEAK",      "HEEL LIFT & FORWARD DIVE", "+28° FORWARD LEAN"),
        (4, pools[4][0], pools[4][1], False, 2, 98, "FULL QUAD & GLUTE · 4×10", "QUADRICEPS & GLUTES",      "95° PARALLEL DRIVE"),
    ]

    panels = []
    crops = []
    step_frames_pil = {}

    for s_idx, pool, kps, is_wrong, dock_idx, pct_val, tag_pill, tag_ctx, cue_1line in step_configs:
        idx = len(pool) - 1
        rgb_frame = apply_static_studio_tint(pool[idx], is_wrong)
        kp_cur = kps[idx]

        canvas = Image.fromarray(rgb_frame).convert("RGBA")
        render_v56_minimal_laser_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)
        p55.draw_v55_safezone_top_dock(canvas, torso_icons, active_idx=dock_idx, is_wrong=is_wrong)
        p55.draw_v55_safezone_bottom_hud(
            canvas,
            {
                "is_wrong": is_wrong,
                "pct": pct_val,
                "tag_pill": tag_pill,
                "tag_ctx": tag_ctx,
                "cue_1line": cue_1line,
            },
        )

        rgb_pil = canvas.convert("RGB")
        step_frames_pil[s_idx] = rgb_pil
        out_bgr = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)
        panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
        crops.append(out_bgr[415:1310, 0:1080])

    for ver in ("v55", "v56"):
        diff_path = os.path.join(OUT_DIR, f"day3_{ver}_difference_sheet.jpg")
        close_path = os.path.join(OUT_DIR, f"day3_{ver}_closeup_sheet.jpg")
        safe_path = os.path.join(OUT_DIR, f"day3_{ver}_iphone_safezone_preview.jpg")
        cv2.imwrite(diff_path, np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        cv2.imwrite(close_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        p55.render_iphone_safezone_sheet(step_frames_pil, safe_path)
        print(f"Saved {ver} sheets:", diff_path, close_path, safe_path)


if __name__ == "__main__":
    main()
