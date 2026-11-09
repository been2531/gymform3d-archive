#!/usr/bin/env python3
"""
Renders a 12-second (360 frames @ 30fps, 1080x1920) live motion test video
(`arrow_4concepts_live_test.mp4`) comparing the 4 Auto-Keypoint Visual Cue Concepts
(A: @demicstory Neon Contour, B: Tapered Light Comet >>, C: Zero-Arrow Target Ring + Leader Pin, D: 3D Orbit Torus Ring)
dynamically tracking MediaPipe joint landmarks on moving Front & Side Squat clips.
- 0.0s - 4.0s: 2x2 Quad Live Comparison (Front Squat 2s + Side Squat 2s)
- 4.0s - 6.0s: Full-Screen Concept A (@demicstory Neon Contour)
- 6.0s - 8.0s: Full-Screen Concept B (Tapered Light Comet >>)
- 8.0s - 10.0s: Full-Screen Concept C (Zero-Arrow Target Ring + Leader Pin)
- 10.0s - 12.0s: Full-Screen Concept D (3D Orbit Torus Ring)
"""

import os
import sys
import math
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v30_pure_graphic_numbers as v30
import build_v36_clean_master as v36
import build_v38_high_contrast_hud as v38
import preview_v58_day2_exact_master as p58
import preview_v62_seamless_day2_master as p62

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920
FPS = 30

COL_GREEN = (34, 245, 138)
COL_GREEN_SOFT = (165, 255, 208)
COL_RED = (255, 68, 56)
COL_RED_SOFT = (255, 178, 168)
COL_WHITE = (242, 247, 255)

_LEFT_W = 430
_CURVE_L = (np.linspace(0.68, 1.0, _LEFT_W, dtype=np.float32) ** 0.78)[None, :, None]
_SCRIM_2D = v38.make_bottom_scrim_2d()


def _find_outer_silhouette_contour(raw_rgb: np.ndarray, y_min: int, y_max: int, side: str, offset_px: int = 22) -> list[tuple[float, float]]:
    """Automatically detects the outer edge of the mannequin silhouette between y_min and y_max and offsets into open studio margin."""
    lum = (
        0.299 * raw_rgb[:, :, 0].astype(np.float32)
        + 0.587 * raw_rgb[:, :, 1].astype(np.float32)
        + 0.114 * raw_rgb[:, :, 2].astype(np.float32)
    )
    fg = lum > 25.0
    pts = []
    y0 = max(180, min(H - 180, int(y_min)))
    y1 = max(y0 + 20, min(H - 180, int(y_max)))
    ys = np.linspace(y0, y1, 28)
    raw_xs = []
    valid_ys = []
    for y in ys:
        yi = int(round(y))
        xs = np.where(fg[yi, 140:940])[0]
        if len(xs) > 8:
            if side == "left":
                x_edge = float(xs[0] + 140 - offset_px)
            else:
                x_edge = float(xs[-1] + 140 + offset_px)
            raw_xs.append(x_edge)
            valid_ys.append(float(yi))
    if len(raw_xs) < 5:
        return []
    # Smooth x coordinates with 1D Gaussian kernel
    k = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    xs_arr = np.pad(np.array(raw_xs, dtype=np.float32), (2, 2), mode="edge")
    xs_smooth = np.convolve(xs_arr, k, mode="valid")
    return [(float(x), float(y)) for x, y in zip(xs_smooth, valid_ys)]


def _draw_glowing_polyline(
    glow_draw: ImageDraw.ImageDraw,
    core_draw: ImageDraw.ImageDraw,
    pts: list[tuple[float, float]],
    col_main: tuple[int, int, int],
    width_core: int = 5,
    width_glow: int = 16,
    alpha_scale: float = 1.0,
):
    if len(pts) < 2:
        return
    a_glow = int(max(0, min(255, round(135 * alpha_scale))))
    a_main = int(max(0, min(255, round(245 * alpha_scale))))
    a_white = int(max(0, min(255, round(235 * alpha_scale))))
    glow_draw.line(pts, fill=(*col_main, a_glow), width=width_glow, joint="curve")
    core_draw.line(pts, fill=(0, 0, 0, int(210 * alpha_scale)), width=width_core + 4, joint="curve")
    core_draw.line(pts, fill=(*col_main, a_main), width=width_core, joint="curve")
    core_draw.line(pts, fill=(255, 255, 255, a_white), width=max(2, width_core // 2), joint="curve")


def _draw_tapered_light_comet(
    glow_draw: ImageDraw.ImageDraw,
    core_draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    col_main: tuple[int, int, int],
    phase: float,
    cw: bool = True,
):
    """Concept B: Tapered glowing light stream + animated micro-chevrons (>>), zero bulky triangle head."""
    steps = 44
    cx, cy = center
    for i in range(steps):
        t0 = i / float(steps)
        t1 = (i + 1) / float(steps)
        a0 = math.radians(start_deg + (end_deg - start_deg) * t0)
        a1 = math.radians(start_deg + (end_deg - start_deg) * t1)
        p0 = (cx + radius * math.cos(a0), cy + radius * math.sin(a0))
        p1 = (cx + radius * math.cos(a1), cy + radius * math.sin(a1))
        # Taper width and opacity from tail (t=0) to head (t=1)
        w_core = max(1, int(round(1.5 + 6.5 * (t1 ** 1.15))))
        w_glow = max(3, int(round(4.0 + 16.0 * (t1 ** 1.15))))
        alpha_g = int(round(140 * (t1 ** 1.25)))
        alpha_c = int(round(250 * (t1 ** 1.1)))
        glow_draw.line([p0, p1], fill=(*col_main, alpha_g), width=w_glow)
        core_draw.line([p0, p1], fill=(*col_main, alpha_c), width=w_core)
        if t1 > 0.35:
            core_draw.line([p0, p1], fill=(255, 255, 255, int(round(230 * ((t1 - 0.35) / 0.65)))), width=max(1, w_core // 2))

    # Draw 2 animated micro-chevrons (››) traveling along the stream
    for ch_idx in range(2):
        u_ch = (0.58 + ch_idx * 0.22 + 0.16 * math.sin(phase * 2.0 * math.pi)) % 1.0
        u_ch = 0.48 + 0.46 * u_ch
        ang = math.radians(start_deg + (end_deg - start_deg) * u_ch)
        tang = ang + (math.pi / 2.0 if (end_deg >= start_deg) else -math.pi / 2.0)
        tip = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))
        wing_len = 11.0
        wing_spread = 0.58
        w1 = (
            tip[0] - wing_len * math.cos(tang - wing_spread),
            tip[1] - wing_len * math.sin(tang - wing_spread),
        )
        w2 = (
            tip[0] - wing_len * math.cos(tang + wing_spread),
            tip[1] - wing_len * math.sin(tang + wing_spread),
        )
        glow_draw.line([w1, tip, w2], fill=(*col_main, 180), width=10, joint="curve")
        core_draw.line([w1, tip, w2], fill=(0, 0, 0, 220), width=6, joint="curve")
        core_draw.line([w1, tip, w2], fill=(255, 255, 255, 250), width=3, joint="curve")


def _draw_concentric_target_ring_and_pin(
    glow_draw: ImageDraw.ImageDraw,
    core_draw: ImageDraw.ImageDraw,
    joint_pt: tuple[float, float],
    pin_end_x: float,
    col_main: tuple[int, int, int],
    phase: float,
):
    """Concept C: Concentric radar target ring (◎) + horizontal leader pin (──•) to callout."""
    jx, jy = joint_pt
    pulse_r = 22.0 + 5.0 * math.sin(phase * 2.0 * math.pi)
    r_inner = 7.0
    r_mid = 16.0

    # Pulsing outer radar ring
    glow_draw.ellipse([jx - pulse_r, jy - pulse_r, jx + pulse_r, jy + pulse_r], outline=(*col_main, 125), width=6)
    core_draw.ellipse([jx - pulse_r, jy - pulse_r, jx + pulse_r, jy + pulse_r], outline=(*col_main, 195), width=2)

    # Crisp mid lock ring + glowing center dot
    core_draw.ellipse([jx - r_mid, jy - r_mid, jx + r_mid, jy + r_mid], outline=(0, 0, 0, 230), width=6)
    core_draw.ellipse([jx - r_mid, jy - r_mid, jx + r_mid, jy + r_mid], outline=(*col_main, 255), width=3)
    core_draw.ellipse([jx - r_inner, jy - r_inner, jx + r_inner, jy + r_inner], fill=(255, 255, 255, 255), outline=(*col_main, 255), width=2)

    # Horizontal leader line from outer edge of ring to callout margin
    dir_sign = -1.0 if pin_end_x < jx else 1.0
    start_x = jx + dir_sign * (r_mid + 4.0)
    glow_draw.line([(start_x, jy), (pin_end_x, jy)], fill=(*col_main, 120), width=8)
    core_draw.line([(start_x, jy), (pin_end_x, jy)], fill=(0, 0, 0, 220), width=5)
    core_draw.line([(start_x, jy), (pin_end_x, jy)], fill=(*col_main, 245), width=3)
    core_draw.ellipse([pin_end_x - 5, jy - 5, pin_end_x + 5, jy + 5], fill=(255, 255, 255, 255), outline=(*col_main, 255), width=2)


def _draw_3d_orbit_torus_ring(
    glow_draw: ImageDraw.ImageDraw,
    core_draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    rx: float,
    ry: float,
    tilt_deg: float,
    col_main: tuple[int, int, int],
    phase: float,
    cw_pulse: bool = True,
):
    """Concept D: Foreshortened 3D perspective torus ring wrapping around the joint in 3D space."""
    cx, cy = center
    tilt = math.radians(tilt_deg)
    ct, st = math.cos(tilt), math.sin(tilt)

    def pt_at(theta_deg: float) -> tuple[float, float]:
        th = math.radians(theta_deg)
        x0 = rx * math.cos(th)
        y0 = ry * math.sin(th)
        return (cx + x0 * ct - y0 * st, cy + x0 * st + y0 * ct)

    # Back half of 3D ring (180..360 deg): subtle translucent behind the limb
    back_pts = [pt_at(d) for d in np.linspace(185, 355, 36)]
    core_draw.line(back_pts, fill=(*col_main, 70), width=3, joint="curve")

    # Front half of 3D ring (0..180 deg): thick glowing foreground arc wrapping in front of the joint
    front_pts = [pt_at(d) for d in np.linspace(-8, 188, 48)]
    glow_draw.line(front_pts, fill=(*col_main, 155), width=18, joint="curve")
    core_draw.line(front_pts, fill=(0, 0, 0, 225), width=9, joint="curve")
    core_draw.line(front_pts, fill=(*col_main, 250), width=6, joint="curve")
    core_draw.line(front_pts, fill=(255, 255, 255, 235), width=2, joint="curve")

    # Animated energy bead orbiting along the front arc
    u_orb = (phase if cw_pulse else (1.0 - phase)) % 1.0
    ang_orb = 18.0 + 144.0 * u_orb
    p_orb = pt_at(ang_orb)
    glow_draw.ellipse([p_orb[0] - 13, p_orb[1] - 13, p_orb[0] + 13, p_orb[1] + 13], fill=(*col_main, 190))
    core_draw.ellipse([p_orb[0] - 6, p_orb[1] - 6, p_orb[0] + 6, p_orb[1] + 6], fill=(255, 255, 255, 255))


def render_concept_overlay(
    canvas: Image.Image,
    raw_rgb: np.ndarray,
    concept_id: str,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    deg_val: int,
    pct_val: int,
    phase: float,
    show_concept_banner: bool = True,
):
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    core_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    gd = ImageDraw.Draw(glow_layer)
    cd = ImageDraw.Draw(core_layer)
    sd = ImageDraw.Draw(shadow)
    od = ImageDraw.Draw(ov)

    col_main = COL_RED if is_wrong else COL_GREEN
    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    hip_r = kp["hip_r"]
    sh = kp["sh"]
    ank_r = kp["ank_r"]
    ank_l = kp["ank_l"]

    if concept_id == "A":
        # Concept A: @demicstory Neon Contour + Pure Alignment Axis (Zero Arrowhead)
        if step_idx in (1, 2):
            y_top = min(hip_r[1], kn_r[1]) - 35
            y_bot = max(ank_r[1], kn_r[1]) - 15
            pts_l = _find_outer_silhouette_contour(raw_rgb, y_top, y_bot, side="left", offset_px=20)
            pts_r = _find_outer_silhouette_contour(raw_rgb, y_top, y_bot, side="right", offset_px=20)
            _draw_glowing_polyline(gd, cd, pts_l, col_main, width_core=5, width_glow=16)
            _draw_glowing_polyline(gd, cd, pts_r, col_main, width_core=5, width_glow=16)
            # Subtle glowing knee joint rings
            v36.draw_single_elbow_pivot_ring(cd, kn_r, 8.5, col_main, 1)
            v36.draw_single_elbow_pivot_ring(cd, kn_l, 8.5, col_main, 1)
        else:
            # Side Squat: Pure glowing dorsal contour + clean white/green spinal alignment axis in outer margin
            y_top = sh[1] - 10
            y_bot = hip_r[1] + 25
            pts_back = _find_outer_silhouette_contour(raw_rgb, y_top, y_bot, side="left", offset_px=22)
            _draw_glowing_polyline(gd, cd, pts_back, col_main, width_core=5, width_glow=16)
            # Crisp plumb reference axis in outer left margin
            axis_p0 = (hip_r[0] - 58.0, hip_r[1])
            axis_p1 = (hip_r[0] - 58.0, sh[1] - 25.0)
            _draw_glowing_polyline(gd, cd, [axis_p0, axis_p1], COL_WHITE, width_core=3, width_glow=9, alpha_scale=0.82)
            v36.draw_single_elbow_pivot_ring(cd, hip_r, 8.5, col_main, 1)

    elif concept_id == "B":
        # Concept B: Tapered Light Comet + Micro-Chevrons (>>)
        if step_idx in (1, 2):
            v36.draw_single_elbow_pivot_ring(cd, kn_r, 8.5, col_main, 1)
            v36.draw_single_elbow_pivot_ring(cd, kn_l, 8.5, col_main, 1)
            _draw_tapered_light_comet(gd, cd, (kn_r[0] - 42.0, kn_r[1] + 4.0), 62.0, 225.0, 142.0, col_main, phase)
            _draw_tapered_light_comet(gd, cd, (kn_l[0] + 42.0, kn_l[1] + 4.0), 62.0, 315.0, 398.0, col_main, phase)
        else:
            v36.draw_single_elbow_pivot_ring(cd, hip_r, 8.5, col_main, 1)
            _draw_tapered_light_comet(gd, cd, (hip_r[0] - 52.0, hip_r[1] - 185.0), 118.0, 176.0, 248.0, col_main, phase)

    elif concept_id == "C":
        # Concept C: Zero-Arrow Concentric Target Ring (◎) + Horizontal Leader Pin (──•)
        if step_idx in (1, 2):
            _draw_concentric_target_ring_and_pin(gd, cd, kn_r, 275.0, col_main, phase)
            _draw_concentric_target_ring_and_pin(gd, cd, kn_l, 765.0, col_main, phase)
        else:
            spine_mid = ((sh[0] + hip_r[0]) * 0.5, (sh[1] + hip_r[1]) * 0.5)
            _draw_concentric_target_ring_and_pin(gd, cd, hip_r, 285.0, col_main, phase)
            _draw_concentric_target_ring_and_pin(gd, cd, spine_mid, 765.0, col_main, phase)

    elif concept_id == "D":
        # Concept D: 3D Perspective Orbit Torus Ring wrapping the joint
        if step_idx in (1, 2):
            _draw_3d_orbit_torus_ring(gd, cd, (kn_r[0], kn_r[1] - 18.0), rx=58.0, ry=20.0, tilt_deg=-8.0, col_main=col_main, phase=phase)
            _draw_3d_orbit_torus_ring(gd, cd, (kn_l[0], kn_l[1] - 18.0), rx=58.0, ry=20.0, tilt_deg=8.0, col_main=col_main, phase=phase, cw_pulse=False)
        else:
            torso_mid = ((sh[0] + hip_r[0]) * 0.5, (sh[1] + hip_r[1]) * 0.5 + 15.0)
            _draw_3d_orbit_torus_ring(gd, cd, torso_mid, rx=86.0, ry=28.0, tilt_deg=-24.0, col_main=col_main, phase=phase)

    # Callout typography synchronized with joint Y
    if step_idx == 2:
        p58.draw_v58_floating_white_callout(od, sd, (255, 895), f"{deg_val}°", "KNEE TRACKING", align="right")
        p58.draw_v58_floating_white_callout(od, sd, (785, 855), f"{pct_val}%", "QUAD & VMO LOCK", align="left")
    elif step_idx == 3:
        p58.draw_v58_floating_white_callout(od, sd, (265, 895), f"{deg_val}°", "FORWARD LEAN", align="right")
        p58.draw_v58_floating_white_callout(od, sd, (785, 855), f"{pct_val}%", "LUMBAR OVERLOAD", align="left")
    elif step_idx == 4:
        p58.draw_v58_floating_white_callout(od, sd, (265, 895), f"{deg_val}°", "NEUTRAL SPINE", align="right")
        p58.draw_v58_floating_white_callout(od, sd, (785, 855), f"{pct_val}%", "QUADS & GLUTES", align="left")

    if show_concept_banner:
        titles = {
            "A": ("CONCEPT A · @demicstory NEON CONTOUR", "PURE GLOWING ALIGNMENT AXIS · ZERO ARROWHEAD"),
            "B": ("CONCEPT B · TAPERED LIGHT COMET (››)", "TAPERED LIGHT STREAM + ANIMATED MICRO-CHEVRONS"),
            "C": ("CONCEPT C · ZERO-ARROW TARGET RING", "CONCENTRIC RADAR LOCK RING (◎) + LEADER PIN"),
            "D": ("CONCEPT D · 3D ORBIT TORUS RING", "FORESHORTENED 3D SPATIAL RING WRAPPING JOINT"),
        }
        t_main, t_sub = titles[concept_id]
        f_title = v30.f_pretendard(28, "extrabold")
        f_sub = v30.f_pretendard(19, "bold")
        # Clean pill banner at top safe zone (y=190..286)
        od.rounded_rectangle([64, 186, W - 64, 286], radius=22, fill=(10, 14, 20, 232), outline=(*col_main, 220), width=3)
        od.text((W // 2, 220), t_main, font=f_title, fill=(255, 255, 255, 255), anchor="mm")
        od.text((W // 2, 258), t_sub, font=f_sub, fill=(*col_main, 245), anchor="mm")

    canvas.alpha_composite(glow_layer.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(core_layer)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
    canvas.alpha_composite(ov)


def build_single_concept_frame(
    raw_rgb: np.ndarray,
    kp: dict,
    step_idx: int,
    concept_id: str,
    u_rep: float,
    phase: float,
    include_full_hud: bool = True,
) -> np.ndarray:
    is_wrong = step_idx in (1, 3)
    if step_idx == 2:
        deg_val = 0
        pct_val = int(round(92 + 4 * u_rep))
        tag_pill = "NEUTRAL KNEE LOCK"
        tag_ctx = "QUADRICEPS & VMO"
        cue_1line = f"{deg_val}° KNEE TRACKING"
    elif step_idx == 3:
        deg_val = int(round(25 + 16 * u_rep))
        pct_val = int(round(72 + 7 * u_rep))
        tag_pill = "LUMBAR SPINE RISK"
        tag_ctx = "GOOD-MORNING SQUAT"
        cue_1line = f"{deg_val}° FORWARD LEAN"
    else:
        deg_val = int(round(15 + 7 * u_rep))
        pct_val = int(round(94 + 4 * u_rep))
        tag_pill = "FULL QUAD & GLUTE · 4×10"
        tag_ctx = "QUADRICEPS & GLUTES"
        cue_1line = f"{deg_val}° NEUTRAL SPINE"

    rgb_tinted = p58.apply_static_studio_tint(raw_rgb, is_wrong)
    rgb_scrim = v38.apply_bottom_left_scrim(rgb_tinted, _CURVE_L, _SCRIM_2D)
    canvas = Image.fromarray(rgb_scrim).convert("RGBA")

    render_concept_overlay(
        canvas,
        raw_rgb=raw_rgb,
        concept_id=concept_id,
        step_idx=step_idx,
        kp=kp,
        is_wrong=is_wrong,
        deg_val=deg_val,
        pct_val=pct_val,
        phase=phase,
        show_concept_banner=include_full_hud,
    )

    if include_full_hud:
        p58.draw_v58_safezone_bottom_hud(
            canvas,
            {
                "is_wrong": is_wrong,
                "pct": pct_val,
                "tag_pill": tag_pill,
                "tag_ctx": tag_ctx,
                "cue_1line": cue_1line,
            },
        )
    return np.array(canvas.convert("RGB"))


def _render_video_frame_worker(args):
    g_idx, step_idx, mode, raw_rgb, kp, u_rep, phase = args
    if mode == "QUAD":
        # Render 2x2 grid comparing A, B, C, D simultaneously in motion
        board = Image.new("RGB", (W, H), (8, 10, 14))
        bd = ImageDraw.Draw(board)
        f_hdr = v30.f_pretendard(28, "extrabold")
        f_sub = v30.f_pretendard(19, "bold")
        f_lbl = v30.f_pretendard(22, "extrabold")
        bd.text((W // 2, 56), "4 AUTO-KEYPOINT VISUAL CUES (LIVE MOTION TEST)", font=f_hdr, fill=(255, 255, 255), anchor="mm")
        pose_name = "FRONT SQUAT (0° KNEE TRACKING)" if step_idx == 2 else "SIDE SQUAT (41° LUMBAR OVERLOAD)"
        col_pose = COL_GREEN if step_idx == 2 else COL_RED
        bd.text((W // 2, 96), f"POSE: {pose_name} · ZERO AI TOKENS", font=f_sub, fill=col_pose, anchor="mm")

        concepts = [
            ("A", "A · @demicstory NEON CONTOUR", 20, 134),
            ("B", "B · TAPERED LIGHT COMET (››)", 546, 134),
            ("C", "C · ZERO-ARROW TARGET RING", 20, 1018),
            ("D", "D · 3D ORBIT TORUS RING", 546, 1018),
        ]
        tw, th = 514, 860
        for cid, clabel, x0, y0 in concepts:
            fr_full = build_single_concept_frame(
                raw_rgb, kp, step_idx, cid, u_rep, phase, include_full_hud=False
            )
            # Crop central action region (y=240..1680, x=60..1020 -> 960x1440) and resize to 514x860
            crop = fr_full[250:1690, 60:1020]
            tile = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
            board.paste(Image.fromarray(tile), (x0, y0))
            bd.rounded_rectangle([x0, y0, x0 + tw, y0 + th], radius=16, outline=col_pose, width=2)
            bd.rounded_rectangle([x0 + 10, y0 + 10, x0 + tw - 10, y0 + 54], radius=10, fill=(10, 14, 20))
            bd.text((x0 + tw // 2, y0 + 32), clabel, font=f_lbl, fill=(255, 255, 255), anchor="mm")

        return g_idx, np.array(board)
    else:
        fr = build_single_concept_frame(
            raw_rgb, kp, step_idx, mode, u_rep, phase, include_full_hud=True
        )
        return g_idx, fr


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    import pickle
    cache_p = "/workspace/daily_info_reels/_cache_fast_arrow_test_pools.pkl"
    if os.path.exists(cache_p):
        with open(cache_p, "rb") as fp:
            pools = pickle.load(fp)
    else:
        pools = {}
        for s_idx in (2, 3, 4):
            fn, i0, i1, is_w, sc, y_st, y_sq, ys_st, ys_sq, x_sh = p62.STEP_POOL_SPECS[s_idx]
            raw = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=24)
            pools[s_idx] = p62.build_v62_fullbleed_pool(raw, s_idx, is_w, scale=sc, y_head_target_stand=y_st, y_head_target_squat=y_sq, y_head_src_stand_720=ys_st, y_head_src_squat_720=ys_sq, x_shift=x_sh)
        with open(cache_p, "wb") as fp:
            pickle.dump(pools, fp)
    print(f"[test_video] Loaded cached v62 pools in {time.time() - t0:.2f}s")

    # Timeline (360 frames = 12.0s @ 30fps):
    # 0..59 (2.0s): QUAD Front Squat (Step 2)
    # 60..119 (2.0s): QUAD Side Squat (Step 3)
    # 120..149 (1.0s): Full-Screen Concept A Front (Step 2)
    # 150..179 (1.0s): Full-Screen Concept A Side (Step 3)
    # 180..209 (1.0s): Full-Screen Concept B Front (Step 2)
    # 210..239 (1.0s): Full-Screen Concept B Side (Step 4)
    # 240..269 (1.0s): Full-Screen Concept C Front (Step 2)
    # 270..299 (1.0s): Full-Screen Concept C Side (Step 3)
    # 300..329 (1.0s): Full-Screen Concept D Front (Step 2)
    # 330..359 (1.0s): Full-Screen Concept D Side (Step 4)
    segments = [
        (0, 60, 2, "QUAD"),
        (60, 120, 3, "QUAD"),
        (120, 150, 2, "A"),
        (150, 180, 3, "A"),
        (180, 210, 2, "B"),
        (210, 240, 4, "B"),
        (240, 270, 2, "C"),
        (270, 300, 3, "C"),
        (300, 330, 2, "D"),
        (330, 360, 4, "D"),
    ]

    out_mp4 = os.path.join(OUT_DIR, "arrow_4concepts_live_test.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-crf", "18",
        "-movflags", "+faststart",
        out_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    total_written = 0
    for f_start, f_end, s_idx, mode in segments:
        frames_s, kps_s = pools[s_idx]
        n_p = len(frames_s)
        seg_len = f_end - f_start
        for i in range(seg_len):
            g_idx = f_start + i
            u_seg = i / float(max(1, seg_len - 1))
            u_rep = 0.5 * (1.0 - math.cos(2.0 * math.pi * u_seg))
            p_idx = int(round(u_rep * (n_p - 1)))
            p_idx = max(0, min(n_p - 1, p_idx))
            phase = (g_idx % 30) / 30.0
            _, rgb = _render_video_frame_worker((g_idx, s_idx, mode, frames_s[p_idx], kps_s[p_idx], u_rep, phase))
            proc.stdin.write(rgb.tobytes())
            total_written += 1

    proc.stdin.close()
    proc.wait()
    print(f"[test_video] Rendered {out_mp4} ({total_written} frames) in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
