#!/usr/bin/env python3
"""
GymForm 3D — v29 Luxury 3D Spine/Limb/Wrist Kinematic Rig & Zero-Box Fusion Concepts
Addresses user feedback (t132u ~ t137u):
1. "허리 위치나 팔목 각도 팔 위치 이런거 선 그래픽? 이런걸로 잡아주고 하면 좋자나."
2. "하단에 박스 몸을 가리니까 별로야."
3. "어떻게 구상할지도 시안부터 그려바"
4. "뼈나 근질 같은거 잘표현한 것처럼 고급스럽게 가야지"
5. "Brand CardNews도 지금 자세교정으로 해서 점점 좋아지고 있어. 우리거랑 잘 합쳐봐"

Key Breakthroughs in v29:
- ZERO Bottom Rectangular Boxes: Mannequin's torso, hips, glutes, thighs, and legs are 100% uncovered.
- 3D Subsurface Vertebral Column (허리 요추 L1-L5 & 흉추 T1-T12 마디 투시) + 3D Bone-Laser Kinematic Axis (어깨-팔꿈치-손목-그립):
  Rendered with subsurface glow masked inside the body silhouette, 3D sculpted vertebral segments,
  titanium plumb reference lines, double-ticked precision angle arcs, and concentric optical joint rings.
- 3 Distinct Fusion Concepts (A-1, A-2, A-3) combining @gymform.3d Concept A with Brand CardNews (posture-ep01).
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import gymform_engine as eng
import build_v25_fast_preview as v25
import build_v28_luxury_ab as v28

W, H = eng.W, eng.H
OUT_DIR = eng.OUT_DIR

COL_RED = (255, 69, 58)
COL_RED_SOFT = (255, 138, 128)
COL_GREEN = (46, 228, 118)
COL_GREEN_SOFT = (158, 248, 195)
COL_CYAN = (110, 218, 255)
COL_SILVER = (224, 232, 242)
COL_MUTED = (150, 162, 182)

f_pretendard = v28.f_pretendard
f_barlow = v28.f_barlow

# Typography
F_SERIES = f_pretendard(18, "bold")
F_HERO_TITLE = f_pretendard(42, "extrabold")
F_HERO_SUB = f_pretendard(22, "semibold")
F_LEFT_TITLE = f_pretendard(36, "extrabold")
F_LEFT_SUB = f_pretendard(18, "semibold")

F_JOINT_DEG = f_barlow(34, extrabold=True)
F_JOINT_PART = f_pretendard(13, "bold")
F_JOINT_DESC = f_pretendard(16, "bold")

F_DIAL_NUM = f_barlow(54, extrabold=True)
F_DIAL_PCT = f_pretendard(22, "bold")
F_DIAL_TAG = f_pretendard(13, "bold")

F_METRIC_LBL = f_pretendard(15, "bold")
F_METRIC_VAL = f_barlow(22, extrabold=True)


def cubic_bezier(p0, p1, p2, p3, n=40):
    pts = []
    for i in range(n + 1):
        t = i / float(n)
        u = 1.0 - t
        x = (u**3)*p0[0] + 3*(u**2)*t*p1[0] + 3*u*(t**2)*p2[0] + (t**3)*p3[0]
        y = (u**3)*p0[1] + 3*(u**2)*t*p1[1] + 3*u*(t**2)*p2[1] + (t**3)*p3[1]
        pts.append((x, y))
    return pts


def apply_atmospheric_vignette(rgb_arr: np.ndarray, left_fade: bool = True, bottom_fade: bool = False) -> np.ndarray:
    """
    Pure studio lighting preservation: ZERO bottom black fade over the mannequin's legs!
    Only a subtle, imperceptible left-edge studio falloff (x=0..480) where there is no body.
    """
    out = rgb_arr.astype(np.float32)
    if left_fade:
        left_w = 480
        curve_l = np.linspace(0.62, 1.0, left_w, dtype=np.float32) ** 0.75
        out[480:1820, :left_w] *= curve_l[None, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_3d_subsurface_spine_and_limb_rig(
    canvas: Image.Image,
    cam_rgb: np.ndarray,
    step_idx: int,
    mp: callable,
    is_wrong: bool,
) -> dict:
    """
    Renders a medical-grade 3D anatomical spine/waist, upper-arm/elbow, and wrist alignment rig
    that matches the quality of our 3D X-Ray bone & muscle striations ("뼈나 근질 같은거 잘표현한 것처럼 고급스럽게"):
      1. Subsurface Vertebral Column (체내 3D 척추·허리 마디 디스크 & 발광 코어) along C7 -> Thoracic -> Lumbar (허리) -> Pelvis.
      2. 3D Internal Humerus -> Ulna/Radius -> Wrist/Carpal Bone-Laser Axis with subsurface glow masked to the body.
      3. Titanium Gravity/Plumb Reference Datum Lines + Precision Double-Ticked Angle Arcs + 3D Optical Joint Rings.
    Returns mapped screen coordinates of the 3 key anatomical checkpoints:
      {"waist": (x,y), "elbow": (x,y), "wrist": (x,y), "shoulder": (x,y)}
    """
    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT
    core_col = (255, 238, 235) if is_wrong else (232, 255, 242)

    # Body silhouette mask so subsurface bone/spine glow stays strictly inside the 3D body volume
    luma = cam_rgb.mean(axis=2)
    body_mask_np = np.clip((luma - 32.0) / 38.0, 0.0, 1.0)
    body_mask_np = cv2.GaussianBlur(body_mask_np.astype(np.float32), (15, 15), 4.0)
    body_mask_pil = Image.fromarray((body_mask_np * 255).astype(np.uint8), mode="L")

    # Raw anatomical coordinates before camera transform
    if step_idx == 1:
        # Hunched upper back (35 deg thoracic kyphosis), rounded shoulder, flared elbow, bent wrist
        p_pelvis = mp(868, 1135)
        p_lumbar = mp(892, 1000)
        p_thoracic = mp(876, 845)
        p_c7 = mp(782, 718)
        p_shoulder = mp(796, 818)
        p_elbow = mp(826, 946)
        p_wrist = mp(638, 995)
        p_hand = mp(578, 1025)
    elif step_idx == 2:
        # Upright spine, but elbow lifted & stopped short at 90 deg half rep
        p_pelvis = mp(748, 1120)
        p_lumbar = mp(768, 985)
        p_thoracic = mp(786, 835)
        p_c7 = mp(738, 695)
        p_shoulder = mp(748, 742)
        p_elbow = mp(742, 896)
        p_wrist = mp(586, 880)
        p_hand = mp(532, 886)
    elif step_idx == 3:
        # 15 deg neutral spine hip-hinge, 0 deg pinned vertical humerus, neutral wrist
        p_pelvis = mp(750, 1125)
        p_lumbar = mp(774, 985)
        p_thoracic = mp(798, 835)
        p_c7 = mp(742, 695)
        p_shoulder = mp(752, 745)
        p_elbow = mp(752, 902)
        p_wrist = mp(602, 978)
        p_hand = mp(546, 1012)
    else:
        # Step 4: Locked 15 deg core/spine, 180 deg full arm lockout, 25 deg outward wrist pronation/spread
        p_pelvis = mp(754, 1125)
        p_lumbar = mp(784, 985)
        p_thoracic = mp(812, 835)
        p_c7 = mp(746, 695)
        p_shoulder = mp(762, 748)
        p_elbow = mp(778, 904)
        p_wrist = mp(685, 1068)
        p_hand = mp(638, 1136)

    spine_pts = cubic_bezier(p_pelvis, p_lumbar, p_thoracic, p_c7, n=48)

    # --- LAYER 1: Subsurface Anatomical Bone & Vertebral Glow (Strictly inside body silhouette) ---
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)

    # Spine subsurface aura
    gd.line(spine_pts, fill=(*accent, 155), width=22, joint="curve")
    # Arm bone subsurface aura (Shoulder -> Elbow -> Wrist -> Hand)
    gd.line([p_shoulder, p_elbow, p_wrist, p_hand], fill=(*accent, 145), width=18, joint="curve")

    glow_blurred = glow_layer.filter(ImageFilter.GaussianBlur(radius=16))
    # Mask subsurface glow by body silhouette so it looks like internal X-ray illumination
    r, g, b, a = glow_blurred.split()
    a_masked = Image.composite(a, Image.new("L", (W, H), 0), body_mask_pil)
    glow_blurred.putalpha(a_masked)
    canvas.alpha_composite(glow_blurred)

    # --- LAYER 2: 4x-Supersampled 3D Vertebral Segments & Sculpted Bone-Laser Axes ---
    SS = 2
    sw, sh = W * SS, H * SS
    rig = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig)

    def S(pt):
        return (pt[0] * SS, pt[1] * SS)

    # 2A. Titanium Reference Plumb / Datum Lines (기준 수직·수평선)
    # Spine vertical plumb line from pelvis upward
    px0, py0 = S(p_pelvis)
    py_top = S(p_c7)[1] - 30 * SS
    for y_d in range(int(py_top), int(py0), 14 * SS):
        rd.line([(px0, y_d), (px0, min(int(py0), y_d + 7 * SS))], fill=(215, 225, 238, 105), width=2 * SS)
    # Small plumb tick at pelvis
    rd.line([(px0 - 10 * SS, py0), (px0 + 10 * SS, py0)], fill=(215, 225, 238, 150), width=2 * SS)

    # Upper-arm vertical plumb line from shoulder downward (in Steps 2, 3, 4) or 90-deg horizontal line in Step 2
    sx0, sy0 = S(p_shoulder)
    ex0, ey0 = S(p_elbow)
    wx0, wy0 = S(p_wrist)
    hx0, hy0 = S(p_hand)

    if step_idx == 2:
        # 90-deg horizontal stop datum line from elbow forward + full range target ray downward
        for x_d in range(int(wx0 - 35 * SS), int(ex0), 12 * SS):
            rd.line([(x_d, ey0), (min(int(ex0), x_d + 6 * SS), ey0)], fill=(*COL_RED, 145), width=2 * SS)
        # Ideal full extension ghost vector downward from elbow
        ideal_w = S(mp(675, 1055))
        for t_i in range(0, 10, 2):
            t0, t1 = t_i / 10.0, (t_i + 1) / 10.0
            p_a = (ex0 + (ideal_w[0] - ex0) * t0, ey0 + (ideal_w[1] - ey0) * t0)
            p_b = (ex0 + (ideal_w[0] - ex0) * t1, ey0 + (ideal_w[1] - ey0) * t1)
            rd.line([p_a, p_b], fill=(*COL_GREEN, 135), width=2 * SS)
    elif step_idx in (3, 4):
        # Vertical fixed elbow hinge datum line through shoulder & elbow
        for y_d in range(int(sy0 - 24 * SS), int(ey0 + 120 * SS), 12 * SS):
            rd.line([(sx0, y_d), (sx0, min(int(ey0 + 120 * SS), y_d + 6 * SS))], fill=(215, 225, 238, 115), width=2 * SS)

    # 2B. 3D Sculpted Vertebral Column Segments (허리 요추 L5..L1 ~ 흉추 T12..T1 입체 척추 마디)
    s_pts_ss = [S(p) for p in spine_pts]
    # Outer dark anatomical casing for contrast
    rd.line(s_pts_ss, fill=(10, 14, 18, 190), width=10 * SS, joint="curve")
    # Colored 3D bone shaft
    rd.line(s_pts_ss, fill=(*accent, 215), width=6 * SS, joint="curve")
    # Specular 3D inner ridge
    rd.line(s_pts_ss, fill=(*core_col, 240), width=2 * SS, joint="curve")

    # Draw 13 3D vertebral body discs perpendicular to the spine curve
    for i in range(3, len(s_pts_ss) - 3, 3):
        cx_v, cy_v = s_pts_ss[i]
        dx = s_pts_ss[i + 1][0] - s_pts_ss[i - 1][0]
        dy = s_pts_ss[i + 1][1] - s_pts_ss[i - 1][1]
        length = max(1e-4, math.hypot(dx, dy))
        nx, ny = -dy / length, dx / length  # normal vector perpendicular to spine
        # Lumbar vertebrae (lower) are wider than thoracic vertebrae (upper)
        half_w = (9.5 - 3.5 * (i / len(s_pts_ss))) * SS
        v1 = (cx_v - nx * half_w, cy_v - ny * half_w)
        v2 = (cx_v + nx * half_w, cy_v + ny * half_w)
        # Shadow base of vertebra
        rd.line([(v1[0], v1[1] + 2 * SS), (v2[0], v2[1] + 2 * SS)], fill=(12, 15, 20, 200), width=5 * SS)
        # Sculpted vertebral disc body
        rd.line([v1, v2], fill=(*accent_soft, 235), width=4 * SS)
        # Bright specular center of vertebra
        v1_in = (cx_v - nx * (half_w * 0.45), cy_v - ny * (half_w * 0.45))
        v2_in = (cx_v + nx * (half_w * 0.45), cy_v + ny * (half_w * 0.45))
        rd.line([v1_in, v2_in], fill=(255, 255, 255, 245), width=2 * SS)

    # 2C. 3D Sculpted Upper Arm (Humerus) -> Forearm (Radius/Ulna) -> Wrist/Grip Axis
    arm_chain = [S(p_shoulder), S(p_elbow), S(p_wrist), S(p_hand)]
    # Dark casing shadow
    rd.line(arm_chain, fill=(10, 14, 18, 210), width=9 * SS, joint="curve")
    # Glowing anodized bone axis
    rd.line(arm_chain, fill=(*accent, 235), width=5 * SS, joint="curve")
    # Specular 1.5px core
    rd.line(arm_chain, fill=(*core_col, 250), width=2 * SS, joint="curve")

    # 2D. Precision Angle Arcs (No filled pie slices! Clean technical double-ring arcs)
    # 1) Waist/Spine Angle Arc at Pelvis/Lumbar
    arc_r_spine = 56 * SS
    rd.arc([px0 - arc_r_spine, py0 - arc_r_spine, px0 + arc_r_spine, py0 + arc_r_spine], start=242 if is_wrong else 254, end=270, fill=(*accent_soft, 240), width=3 * SS)

    # 2) Elbow Joint Angle Arc
    arc_r_elbow = 44 * SS
    if step_idx == 1:
        rd.arc([ex0 - arc_r_elbow, ey0 - arc_r_elbow, ex0 + arc_r_elbow, ey0 + arc_r_elbow], start=158, end=256, fill=(*COL_RED_SOFT, 240), width=3 * SS)
    elif step_idx == 2:
        # 90 deg right-angle bracket + arc at elbow
        sq = 22 * SS
        rd.line([(ex0 - sq, ey0), (ex0 - sq, ey0 - sq), (ex0, ey0 - sq)], fill=(*COL_RED_SOFT, 245), width=3 * SS)
    elif step_idx == 3:
        rd.arc([ex0 - arc_r_elbow, ey0 - arc_r_elbow, ex0 + arc_r_elbow, ey0 + arc_r_elbow], start=142, end=270, fill=(*COL_GREEN_SOFT, 240), width=3 * SS)
    else:
        # Step 4: 180 deg lockout arc at elbow + outward wrist spread vectors at wrist
        rd.arc([ex0 - arc_r_elbow, ey0 - arc_r_elbow, ex0 + arc_r_elbow, ey0 + arc_r_elbow], start=118, end=262, fill=(*COL_GREEN_SOFT, 245), width=3 * SS)
        # Dual outward wrist torque arrows at wrist/hand
        sp_l = S(mp(618, 1118))
        sp_r = S(mp(695, 1142))
        rd.line([(wx0, wy0), sp_l], fill=(*COL_GREEN_SOFT, 245), width=3 * SS)
        rd.line([(wx0, wy0), sp_r], fill=(*COL_GREEN_SOFT, 245), width=3 * SS)

    # 3) Wrist Alignment Brackets (손목 각도 중립/외회전 브래킷)
    wr_r = 22 * SS
    rd.arc([wx0 - wr_r, wy0 - wr_r, wx0 + wr_r, wy0 + wr_r], start=0, end=360, fill=(255, 255, 255, 90), width=1 * SS)

    # 2E. 3D Optical Joint Rings at Key Landmarks (Pelvis/Waist, Lumbar, Shoulder, Elbow, Wrist)
    joint_nodes = [
        (p_lumbar, 11, True),     # Waist / Lumbar Spine (허리)
        (p_shoulder, 11, True),   # Shoulder (어깨)
        (p_elbow, 12, True),      # Elbow (팔꿈치)
        (p_wrist, 10, True),      # Wrist (손목)
        (p_pelvis, 7, False),     # Pelvis base
        (p_hand, 6, False),       # Hand / Rope grip
    ]
    for pt, rad, is_major in joint_nodes:
        jx, jy = S(pt)
        r_out = rad * SS
        # Dark optical backing ring
        rd.ellipse([jx - r_out - 2 * SS, jy - r_out - 2 * SS, jx + r_out + 2 * SS, jy + r_out + 2 * SS], fill=(10, 12, 16, 175))
        # Colored precision ring
        rd.ellipse([jx - r_out, jy - r_out, jx + r_out, jy + r_out], outline=(*accent_soft, 245), width=2 * SS if is_major else 1 * SS)
        if is_major:
            r_mid = int(r_out * 0.58)
            rd.ellipse([jx - r_mid, jy - r_mid, jx + r_mid, jy + r_mid], outline=(255, 255, 255, 180), width=1 * SS)
        # Bright center pin
        r_dot = (3 if is_major else 2) * SS
        rd.ellipse([jx - r_dot, jy - r_dot, jx + r_dot, jy + r_dot], fill=(255, 255, 255, 255))

    rig_smooth = rig.resize((W, H), Image.Resampling.LANCZOS)
    canvas.alpha_composite(rig_smooth)

    return {
        "waist": p_lumbar,
        "shoulder": p_shoulder,
        "elbow": p_elbow,
        "wrist": p_wrist,
        "pelvis": p_pelvis,
    }


def draw_frameless_3point_posture_callouts(
    canvas: Image.Image,
    joints: dict,
    posture_specs: list[tuple[str, str, str, str]],
    is_wrong: bool,
    y_positions: tuple[int, int, int] = (555, 795, 1025),
):
    """
    Draws 3 frameless, box-free posture callouts on the open left dark studio margin (x=68..455)
    connected by 1.5px specular hairlines to the 3D Spine/Waist (허리·척추), Elbow/Arm (팔 위치), and Wrist (팔목 각도).
    Zero opaque boxes — uses Brand CardNews style soft optical shadow + razor-sharp typography + 2px vertical accent bar.
    """
    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    x_left = 68
    for idx, (joint_key, deg_str, part_lbl, desc_str) in enumerate(posture_specs):
        ty = y_positions[idx]
        jx, jy = joints[joint_key]

        # 1. Measure text block width
        db = od.textbbox((0, 0), deg_str, font=F_JOINT_DEG)
        deg_w = db[2] - db[0]
        pb = od.textbbox((0, 0), part_lbl, font=F_JOINT_PART)
        dsb = od.textbbox((0, 0), desc_str, font=F_JOINT_DESC)
        right_w = max(pb[2] - pb[0], dsb[2] - dsb[0])
        block_end_x = x_left + 12 + deg_w + 14 + right_w

        # 2. Pure character-only optical shadow (ZERO rectangular shadow box!)
        od.rounded_rectangle([x_left, ty + 2, x_left + 4, ty + 42], radius=2, fill=(*accent, 255))

        sd.text((x_left + 14, ty + 1), deg_str, font=F_JOINT_DEG, fill=(0, 0, 0, 240))
        od.text((x_left + 14, ty - 2), deg_str, font=F_JOINT_DEG, fill=(*accent_soft, 255))

        tx = x_left + 14 + deg_w + 12
        sd.text((tx, ty + 3), part_lbl, font=F_JOINT_PART, fill=(0, 0, 0, 240))
        od.text((tx, ty + 1), part_lbl, font=F_JOINT_PART, fill=(*COL_SILVER, 225))
        sd.text((tx, ty + 23), desc_str, font=F_JOINT_DESC, fill=(0, 0, 0, 240))
        od.text((tx, ty + 20), desc_str, font=F_JOINT_DESC, fill=(255, 255, 255, 255))

        # 6. Precision 1.5px elbow hairline from the text baseline/right out to the 3D joint ring on the body
        line_y = ty + 24
        start_x = min(465, block_end_x + 16)
        knee_x = min(jx - 32, max(start_x + 24, (start_x + jx) * 0.58))
        pts = [(start_x, line_y), (knee_x, line_y), (jx, jy)]
        sd.line(pts, fill=(0, 0, 0, 200), width=5, joint="curve")
        od.line(pts, fill=(*accent_soft, 210), width=2, joint="curve")
        # Small anchor dot at text end
        od.ellipse([start_x - 3, line_y - 3, start_x + 3, line_y + 3], fill=(*accent_soft, 255))

    shadow_blur = shadow.filter(ImageFilter.GaussianBlur(radius=12))
    canvas.alpha_composite(shadow_blur)
    canvas.alpha_composite(ov)


def draw_concept_a1_keynote_bottom(canvas: Image.Image, sp: dict):
    """
    Concept A-1 (Brand CardNews x @gymform.3d Fusion):
    100% Zero-Box Apple-Keynote Typography placed strictly in the open lower-left studio space (x=68..515, y=1285..1495)
    with character-only optical shadow (ZERO dark rectangle, ZERO overlap with the mannequin's legs on x=550..940!).
    """
    is_wrong = sp["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx, by = 68, 1285
    # Thin 2px accent bar above title
    od.rounded_rectangle([bx, by, bx + 48, by + 4], radius=2, fill=(*accent, 255))

    kicker = f"{sp['kicker']}  ·  {sp['pct']}% {sp['dial_tag']}"
    sd.text((bx, by + 18), kicker, font=F_JOINT_PART, fill=(0, 0, 0, 240))
    od.text((bx, by + 16), kicker, font=F_JOINT_PART, fill=(*accent_soft, 255))

    title = sp["line1"]
    sd.text((bx, by + 44), title, font=F_LEFT_TITLE, fill=(0, 0, 0, 245))
    od.text((bx, by + 41), title, font=F_LEFT_TITLE, fill=(255, 255, 255, 255))

    sub = sp["line2"]
    sd.text((bx, by + 91), sub, font=F_LEFT_SUB, fill=(0, 0, 0, 240))
    od.text((bx, by + 88), sub, font=F_LEFT_SUB, fill=(*COL_SILVER, 245))

    parts = [f"{m_lbl} {m_val}%" for (m_lbl, m_val, _) in sp["splits"]]
    parts.append(sp["rx_badge"])
    strip_str = "   |   ".join(parts)
    sd.text((bx, by + 128), strip_str, font=F_JOINT_PART, fill=(0, 0, 0, 240))
    od.text((bx, by + 125), strip_str, font=F_JOINT_PART, fill=(*accent_soft, 245))

    shadow_blur = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(shadow_blur)
    canvas.alpha_composite(ov)


def draw_concept_a2_left_floating_hud(canvas: Image.Image, sp: dict):
    """
    Concept A-2 (Left-Margin Frameless Radial Dial + Coaching Stack · 0% Body/Leg Occlusion):
    Places the 270-deg radial gauge and coaching typography strictly in the open left studio space (x=68..485, y=1220..1620)
    with ZERO rectangular box, leaving the mannequin's hips, glutes, thighs, and legs on the right (x=520..950) 100% visible!
    """
    is_wrong = sp["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT
    pct = sp["pct"]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # 1. 4x-Supersampled Frameless Radial Gauge on the Left Margin (cx=142, cy=1315, r=66)
    cx, cy = 142, 1315
    SS = 4
    d_sz = 180
    dial = Image.new("RGBA", (d_sz * SS, d_sz * SS), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)
    dc = (d_sz * SS) // 2
    r_arc = 64 * SS

    # Subtle dark backing inside ring only
    dd.ellipse([dc - 72 * SS, dc - 72 * SS, dc + 72 * SS, dc + 72 * SS], fill=(10, 12, 16, 170))
    start_a, span_a = 135, 270
    dd.arc([dc - r_arc, dc - r_arc, dc + r_arc, dc + r_arc], start=start_a, end=start_a + span_a, fill=(45, 52, 66, 200), width=9 * SS)
    active_end = start_a + int(span_a * (pct / 100.0))
    dd.arc([dc - r_arc, dc - r_arc, dc + r_arc, dc + r_arc], start=start_a, end=active_end, fill=(*accent, 255), width=10 * SS)

    dial_sm = dial.resize((d_sz, d_sz), Image.Resampling.LANCZOS)
    ov.alpha_composite(dial_sm, (cx - d_sz // 2, cy - d_sz // 2))

    # Numerals inside dial
    num_s = str(pct)
    nb = od.textbbox((0, 0), num_s, font=F_DIAL_NUM)
    pb = od.textbbox((0, 0), "%", font=F_DIAL_PCT)
    tw_d = (nb[2] - nb[0]) + (pb[2] - pb[0]) + 2
    nx = cx - tw_d // 2
    ny = cy - 30
    od.text((nx, ny), num_s, font=F_DIAL_NUM, fill=(255, 255, 255, 255))
    od.text((nx + (nb[2] - nb[0]) + 2, ny + 18), "%", font=F_DIAL_PCT, fill=(*accent_soft, 255))

    tb = od.textbbox((0, 0), sp["dial_tag"], font=F_DIAL_TAG)
    od.text((cx - (tb[2] - tb[0]) // 2, cy + 32), sp["dial_tag"], font=F_DIAL_TAG, fill=(*accent_soft, 240))

    # 2. Frameless Left-Aligned Coaching Stack below/beside the dial (x=68..490, y=1415..1635)
    tx = 68
    ty = 1418
    # zero rectangle shadow

    od.text((tx, ty), sp["kicker"], font=F_JOINT_PART, fill=(*accent_soft, 255))
    od.text((tx, ty + 24), sp["line1"], font=F_LEFT_TITLE, fill=(255, 255, 255, 255))
    od.text((tx, ty + 72), sp["line2"], font=F_LEFT_SUB, fill=(*COL_SILVER, 240))

    # Frameless EMG bars in left column (x=68..470)
    my = ty + 114
    for m_lbl, m_val, m_col in sp["splits"]:
        od.text((tx, my), m_lbl, font=F_JOINT_PART, fill=COL_MUTED)
        od.text((tx + 135, my - 4), f"{m_val}%", font=F_METRIC_VAL, fill=(255, 255, 255, 255) if m_col == accent else COL_SILVER)
        bar_x = tx + 195
        bar_w = 190
        od.rounded_rectangle([bar_x, my + 5, bar_x + bar_w, my + 11], radius=3, fill=(38, 44, 56, 220))
        fw = max(8, int(bar_w * (m_val / 100.0)))
        od.rounded_rectangle([bar_x, my + 5, bar_x + fw, my + 11], radius=3, fill=(*m_col, 255))
        my += 28

    shadow_blur = shadow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas.alpha_composite(shadow_blur)
    canvas.alpha_composite(ov)


def draw_concept_a3_onbody_blueprint(
    canvas: Image.Image,
    joints: dict,
    posture_specs: list[tuple[str, str, str, str]],
    sp: dict,
):
    """
    Concept A-3 (Ultra-Minimal On-Body Anatomical Blueprint + Compact Bottom-Left Caption):
    Places compact degree badges directly beside the 3D Waist/Spine, Elbow, and Wrist joints,
    and keeps only a minimal 2-line frameless title + EMG strip at the bottom-left (x=68..510, y=1560..1760).
    """
    is_wrong = sp["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    accent_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # Compact floating tags right next to each of the 3 joints (shifted left of the body silhouette)
    offsets = [
        (-310, -28),  # Waist / Spine
        (-320, -18),  # Elbow
        (-285, +12),  # Wrist
    ]
    for idx, (joint_key, deg_str, part_lbl, desc_str) in enumerate(posture_specs):
        jx, jy = joints[joint_key]
        ox, oy = offsets[idx]
        tx = max(68, int(jx + ox))
        ty = int(jy + oy)

        label_full = f"{deg_str}  {desc_str}"
        lb = od.textbbox((0, 0), label_full, font=F_JOINT_DESC)
        lw = lb[2] - lb[0]

        # zero rectangle shadow
        od.text((tx, ty - 4), part_lbl, font=F_JOINT_PART, fill=(*accent_soft, 240))
        od.text((tx, ty + 14), label_full, font=F_JOINT_DESC, fill=(255, 255, 255, 255))

        # Short crisp hairline to joint
        end_x = tx + lw + 12
        if end_x < jx - 14:
            od.line([(end_x, ty + 22), (jx - 12, jy)], fill=(*accent_soft, 220), width=2)

    # Bottom-left frameless summary (x=68..510, y=1565..1755) — never touches the legs on the right!
    bx, by = 68, 1575
    # zero rectangle shadow
    od.text((bx, by), f"{sp['kicker']}  ·  {sp['pct']}% {sp['dial_tag']}", font=F_JOINT_PART, fill=(*accent_soft, 255))
    od.text((bx, by + 24), sp["line1"], font=F_LEFT_TITLE, fill=(255, 255, 255, 255))
    od.text((bx, by + 72), sp["line2"], font=F_LEFT_SUB, fill=(*COL_SILVER, 240))

    parts = [f"{m_lbl} {m_val}%" for (m_lbl, m_val, _) in sp["splits"]]
    parts.append(sp["rx_badge"])
    od.text((bx, by + 114), "   ·   ".join(parts), font=F_JOINT_PART, fill=(*accent_soft, 240))

    shadow_blur = shadow.filter(ImageFilter.GaussianBlur(radius=12))
    canvas.alpha_composite(shadow_blur)
    canvas.alpha_composite(ov)


def main():
    f_w45 = eng.prepare_xray_bone_frame(v25.read_specific_frame(eng.WRONG_I2V, 45))
    f_c14 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 14))
    f_c42 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 42))
    f_c64 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 64))

    size = (142, 114)
    crop_box = (555, 550, 905, 950)
    inact_base = Image.fromarray(f_c42["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 125)))

    x_arr, x_targets, _, _ = eng.render_xray_skeletal_frame(f_w45, 1.0, 0.90)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m1_arr, m1_targets = v25.render_natural_muscle_frame(f_c14, "anterior_deltoid", True, 1.0)
    im1 = Image.fromarray(m1_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m2_arr, m2_targets = v25.render_natural_muscle_frame(f_c42, "triceps_lat_long", False, 1.0)
    m3_arr, m3_targets = v25.render_natural_muscle_frame(f_c64, "triceps_all_three", False, 1.0)
    im2 = Image.fromarray(m3_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    # Note how each step specifies the 3 key posture line checkpoints:
    #   1) SPINE & WAIST (허리·척추 위치)
    #   2) ARM & ELBOW (팔·팔꿈치 위치)
    #   3) WRIST & GRIP (팔목 각도·그립)
    steps = [
        {
            "idx": 1,
            "is_wrong": True,
            "base_rgb": x_arr,
            "active_slot": 0,
            "cam_zoom": 1.04,
            "cam_focus": (690.0, 750.0),
            "cam_dest": (715.0, 750.0),
            "pct": 86,
            "dial_tag": "JOINT RISK",
            "kicker": "MISTAKE 01 · TORSO & SHOULDER POSTURE",
            "line1": "DON'T HUNCH SHOULDERS",
            "line2": "Rounding upper back & leaning over cable impinges the shoulder",
            "posture_specs": [
                ("waist", "35°", "SPINE & WAIST AXIS", "HUNCHED THORACIC BACK"),
                ("shoulder", "86%", "SHOULDER JOINT", "ACROMION IMPINGEMENT"),
                ("wrist", "28°", "ELBOW & WRIST ANGLE", "FLARED ELBOW & BENT WRIST"),
            ],
            "splits": [
                ("SHOULDER LOAD", 86, COL_RED),
                ("TRICEPS WORK", 18, COL_MUTED),
            ],
            "rx_badge": "FIX: CHEST UP",
        },
        {
            "idx": 2,
            "is_wrong": True,
            "base_rgb": m1_arr,
            "active_slot": 1,
            "cam_zoom": 1.24,
            "cam_focus": (675.0, 775.0),
            "cam_dest": (725.0, 775.0),
            "pct": 72,
            "dial_tag": "DELT STEAL",
            "kicker": "MISTAKE 02 · INCOMPLETE RANGE OF MOTION",
            "line1": "AVOID 90° HALF REPS",
            "line2": "Stopping at 90° & lifting elbows shifts 72% load to front delts",
            "posture_specs": [
                ("shoulder", "72%", "ANTERIOR DELTOID", "ANTERIOR DELT OVERLOAD"),
                ("elbow", "90°", "ELBOW RANGE OF MOTION", "STOPPED SHORT AT 90°"),
                ("wrist", "HIGH", "WRIST STOP POSITION", "NO BOTTOM EXTENSION"),
            ],
            "splits": [
                ("FRONT DELT", 72, COL_RED),
                ("LATERAL HEAD", 28, COL_MUTED),
            ],
            "rx_badge": "FIX: FULL RANGE",
        },
        {
            "idx": 3,
            "is_wrong": False,
            "base_rgb": m2_arr,
            "active_slot": 2,
            "cam_zoom": 1.14,
            "cam_focus": (685.0, 775.0),
            "cam_dest": (725.0, 775.0),
            "pct": 95,
            "dial_tag": "LAT & LONG",
            "kicker": "PRO TIP 01 · SPINE, ELBOW & WRIST SETUP",
            "line1": "PIN ELBOWS TO RIBS",
            "line2": "Hinge 15° at hips, lock elbows at 0° & keep wrists neutral",
            "posture_specs": [
                ("waist", "15°", "LUMBAR & WAIST HINGE", "NEUTRAL SPINE HIP HINGE"),
                ("elbow", "0°", "UPPER ARM & ELBOW AXIS", "ELBOWS PINNED TO RIBS"),
                ("wrist", "0°", "WRIST ALIGNMENT", "LOCKED NEUTRAL WRIST"),
            ],
            "splits": [
                ("LAT HEAD", 95, COL_GREEN),
                ("LONG HEAD", 91, COL_GREEN),
            ],
            "rx_badge": "15° HIP HINGE",
        },
        {
            "idx": 4,
            "is_wrong": False,
            "base_rgb": m3_arr,
            "active_slot": 2,
            "cam_zoom": 1.30,
            "cam_focus": (675.0, 825.0),
            "cam_dest": (730.0, 790.0),
            "pct": 99,
            "dial_tag": "ALL 3 HEADS",
            "kicker": "PRO TIP 02 · FULL LOCKOUT & WRIST PRONATION",
            "line1": "SPREAD ROPE AT BOTTOM",
            "line2": "Extend to 180° lockout & pronate wrists 25° outward for 1s",
            "posture_specs": [
                ("waist", "15°", "CORE & WAIST LOCK", "ZERO TORSO MOMENTUM"),
                ("elbow", "180°", "ELBOW LOCKOUT ANGLE", "FULL TRICEPS LOCKOUT"),
                ("wrist", "25°", "WRIST PRONATION ANGLE", "OUTWARD ROPE SPREAD"),
            ],
            "splits": [
                ("LAT HEAD", 99, COL_GREEN),
                ("LONG HEAD", 95, COL_GREEN),
                ("MED HEAD", 96, COL_GREEN),
            ],
            "rx_badge": "4 × 12 REPS",
        },
    ]

    frames_a1, frames_a2, frames_a3 = [], [], []

    for sp in steps:
        cam_rgb, map_pt = v28.apply_cinema_camera(
            sp["base_rgb"],
            zoom=sp["cam_zoom"],
            focus_xy=sp["cam_focus"],
            dest_xy=sp["cam_dest"],
        )

        # --- Concept A-1: Brand CardNews x @gymform.3d Fusion (3D Spine/Limb/Wrist Rig + Left Posture Lines + Bottom Keynote Frameless Typography) ---
        rgb_a1 = apply_atmospheric_vignette(cam_rgb, left_fade=True, bottom_fade=True)
        im_a1 = Image.fromarray(rgb_a1).convert("RGBA")
        joints = render_3d_subsurface_spine_and_limb_rig(im_a1, rgb_a1, sp["idx"], map_pt, sp["is_wrong"])
        v28.draw_luxury_top_dock(im_a1, torso_icons, sp["active_slot"], sp["is_wrong"])
        draw_frameless_3point_posture_callouts(im_a1, joints, sp["posture_specs"], sp["is_wrong"], y_positions=(565, 805, 1045))
        draw_concept_a1_keynote_bottom(im_a1, sp)
        p_a1 = os.path.join(OUT_DIR, f"v29_concept_A1_step{sp['idx']}.jpg")
        im_a1.convert("RGB").save(p_a1, quality=95)
        frames_a1.append(im_a1.convert("RGB"))

        # --- Concept A-2: Left-Margin Frameless Radial HUD + 3D Spine/Limb/Wrist Rig (0% Body/Leg Occlusion) ---
        rgb_a2 = apply_atmospheric_vignette(cam_rgb, left_fade=True, bottom_fade=False)
        im_a2 = Image.fromarray(rgb_a2).convert("RGBA")
        joints2 = render_3d_subsurface_spine_and_limb_rig(im_a2, rgb_a2, sp["idx"], map_pt, sp["is_wrong"])
        v28.draw_luxury_top_dock(im_a2, torso_icons, sp["active_slot"], sp["is_wrong"])
        draw_frameless_3point_posture_callouts(im_a2, joints2, sp["posture_specs"], sp["is_wrong"], y_positions=(520, 735, 950))
        draw_concept_a2_left_floating_hud(im_a2, sp)
        p_a2 = os.path.join(OUT_DIR, f"v29_concept_A2_step{sp['idx']}.jpg")
        im_a2.convert("RGB").save(p_a2, quality=95)
        frames_a2.append(im_a2.convert("RGB"))

        # --- Concept A-3: Ultra-Minimal On-Body Anatomical Blueprint + Compact Bottom-Left Summary ---
        rgb_a3 = apply_atmospheric_vignette(cam_rgb, left_fade=True, bottom_fade=False)
        im_a3 = Image.fromarray(rgb_a3).convert("RGBA")
        joints3 = render_3d_subsurface_spine_and_limb_rig(im_a3, rgb_a3, sp["idx"], map_pt, sp["is_wrong"])
        v28.draw_luxury_top_dock(im_a3, torso_icons, sp["active_slot"], sp["is_wrong"])
        draw_concept_a3_onbody_blueprint(im_a3, joints3, sp["posture_specs"], sp)
        p_a3 = os.path.join(OUT_DIR, f"v29_concept_A3_step{sp['idx']}.jpg")
        im_a3.convert("RGB").save(p_a3, quality=95)
        frames_a3.append(im_a3.convert("RGB"))

    # Build 3-Row x 4-Column Master Comparison Board (`day1_v29_3concepts_board.jpg`)
    pw, ph = 370, 658
    pad = 22
    row_h = ph + 78
    board = Image.new("RGB", (pw * 4 + pad * 5, row_h * 3 + 40), (10, 12, 16))
    bd = ImageDraw.Draw(board)

    f_row = f_pretendard(21, "extrabold")
    f_col = f_pretendard(14, "bold")

    row_configs = [
        ("시안 A-1 (Brand CardNews × @gymform.3d 직결합)  ·  3D 척추·팔·손목 입체 투시 정렬선 + 좌측 여백 3점 각도 + 하단 무박스 Keynote 타이포 (몸 가림 0%)", frames_a1, COL_GREEN),
        ("시안 A-2 (좌측 여백 올인원 플로팅 HUD)  ·  우측 마네킹 전신·다리 100% 노출 + 좌측 여백에 3점 자세 라인 & 무박스 원형 다이얼 배치", frames_a2, COL_CYAN),
        ("시안 A-3 (미니멀 온바디 3D 해부학 블루프린트)  ·  관절 바로 옆 밀착 각도 스펙 + 좌하단 미니멀 2줄 요약 (여백 극대화)", frames_a3, (255, 195, 90)),
    ]

    col_titles = [
        "STEP 1 · 허리 35° 굽음 & 어깨 충돌 (✕)",
        "STEP 2 · 팔꿈치 90° 하프렙 & 전면삼각근 개입 (✕)",
        "STEP 3 · 허리 15° 힌지 · 팔꿈치 0° · 손목 중립 (✓)",
        "STEP 4 · 팔 180° 신전 & 손목 25° 외회전 스퀴즈 (✓)",
    ]

    for r_idx, (r_title, r_frames, r_col) in enumerate(row_configs):
        y0 = 20 + r_idx * row_h
        bd.text((pad, y0), r_title, font=f_row, fill=r_col)
        for c_idx in range(4):
            x0 = pad + c_idx * (pw + pad)
            c_col = COL_RED if c_idx < 2 else COL_GREEN
            bd.text((x0 + 4, y0 + 34), col_titles[c_idx], font=f_col, fill=c_col)
            thumb = r_frames[c_idx].resize((pw, ph), Image.Resampling.LANCZOS)
            ImageDraw.Draw(thumb).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
            board.paste(thumb, (x0, y0 + 58))

    board_path = os.path.join(OUT_DIR, "day1_v29_3concepts_board.jpg")
    board.save(board_path, quality=95)
    print("Saved v29 3-concepts board:", board_path)


if __name__ == "__main__":
    main()
