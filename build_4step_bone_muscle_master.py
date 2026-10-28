#!/usr/bin/env python3
"""
Day 3 (`Barbell Back Squat`) — 4-Step Master with:
1. Authentic Day 1 X-Ray Bone + Vascular Energy Flow (`build_clean_d4_variants.py` / `gymform_engine.py` exact formula) for Steps 1 & 3,
   and Authentic Day 1/Day 2 3D Intrinsic Fiber-Strand Muscle Activation (`build_v36_publish_reels.py` exact formula) for Steps 2 & 4.
2. Skin-tight, 100% Silhouette-Fitted 3D Matte Charcoal Short Trunks / Drawers (`드로즈`):
   - Outer edges follow `human_mask` 100% (never slices vertically across the hips during a squat),
   - Snug low-rise elastic waistband + upper-thigh hem that hugs the groin & upper thigh crease in both standing and deep squat,
   - Shaded directly from the mannequin's 3D surface lighting (`sculpt_curve`) with zero blurry waistband text.
3. Large, Screen-Filling Mannequin Scale (`scale = 1.25 ~ 1.30`, feet anchored at `y = 1615`), taking full advantage of the compact top preview (`y = 148..296`).
4. Confirmed Bottom Single-Line Exercise Title:
   - `"Barbell Back Squat"` in Font #1 (`Kaushan Script`, `KaushanScript-Regular.ttf`),
   - NO flanking horizontal lines (`-`) on either side.
"""
import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/workspace/daily_info_reels")
import universal_anatomy_rig as uar
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62
import build_clean_d4_variants as bcd

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920

CURSIVE_FONT_PATH = "/workspace/daily_info_reels/fonts/KaushanScript-Regular.ttf"


def apply_fitted_3d_drawers_720p(
    out_rgb: np.ndarray,
    human_mask: np.ndarray,
    lm,
    sculpt_curve: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    True 3D Femur-Coordinate (`t_fem` + `s_glute`) Glute-Wrapped & Quad-Exposed Athletic Drawers (`드로즈`):
    - Projects every pixel onto the viewer-left (`lm[24] -> lm[26]`) and viewer-right (`lm[23] -> lm[25]`) femur axes:
      * Along the anterior/outer quadriceps belly (`s_glute <= 0`), the leg opening sits high (`t_hem ~ 0.20..0.25`)
        so 75%+ of the quadriceps muscle bellies glow cleanly in both standing and deep squat.
      * Along the inner/posterior glute & groin underside (`s_glute > 0`), the fabric wraps down (`+0.36 * squat_depth`)
        all the way to the bottom of the glute fold (`crotch_y_split + 26px`) so 100% of the buttocks are enclosed!
    """
    sh_h, sw_w = out_rgb.shape[:2]
    yy, xx = np.mgrid[:sh_h, :sw_w].astype(np.float32)

    sh_l = (lm[11].x * sw_w, lm[11].y * sh_h)
    sh_r = (lm[12].x * sw_w, lm[12].y * sh_h)
    hip_vL = (lm[24].x * sw_w, lm[24].y * sh_h)
    hip_vR = (lm[23].x * sw_w, lm[23].y * sh_h)
    kn_vL = (lm[26].x * sw_w, lm[26].y * sh_h)
    kn_vR = (lm[25].x * sw_w, lm[25].y * sh_h)

    sh_cy = 0.5 * (sh_l[1] + sh_r[1])
    hip_cx = 0.5 * (hip_vL[0] + hip_vR[0])
    hip_cy = 0.5 * (hip_vL[1] + hip_vR[1])
    kn_cy = 0.5 * (kn_vL[1] + kn_vR[1])

    torso_h = max(145.0, hip_cy - sh_cy)
    hip_span = max(78.0, abs(hip_vR[0] - hip_vL[0]))
    squat_depth = float(np.clip((240.0 - (kn_cy - hip_cy)) / 165.0, 0.0, 1.0))

    crotch_y_split = hip_cy + 45.0
    x_mid = int(round(hip_cx))
    y_s0 = int(max(0, min(sh_h - 1, hip_cy - 5)))
    y_s1 = int(max(y_s0 + 1, min(sh_h, max(kn_cy + 55, hip_cy + 115))))
    for y_scan in range(y_s0, y_s1):
        strip = human_mask[y_scan, max(0, x_mid - 22):min(sw_w, x_mid + 23)]
        if strip.size > 0 and np.min(strip) < 0.25:
            crotch_y_split = float(y_scan)
            break

    waist_y_center = hip_cy - (0.14 + 0.08 * squat_depth) * torso_h
    dx_w = (xx - hip_cx) / (hip_span * 0.65)
    waist_dip = (5.0 + 7.0 * squat_depth) * (1.0 - np.clip(dx_w ** 2, 0.0, 1.4))
    waist_top_y = waist_y_center + waist_dip
    band_h = 11.5
    waist_bot_y = waist_top_y + band_h
    top_gate = np.clip((yy - waist_top_y + 1.0) / 1.8, 0.0, 1.0)

    def _femur_coords(h_pt, k_pt, is_viewer_left: bool):
        vx, vy = k_pt[0] - h_pt[0], k_pt[1] - h_pt[1]
        l2 = max(100.0, vx * vx + vy * vy)
        t = ((xx - h_pt[0]) * vx + (yy - h_pt[1]) * vy) / l2
        length = math.sqrt(l2)
        if is_viewer_left:
            nx, ny = vy / length, -vx / length
        else:
            nx, ny = -vy / length, vx / length
        s_glute = (xx - h_pt[0]) * nx + (yy - h_pt[1]) * ny
        return t, s_glute

    t_vL, s_glute_vL = _femur_coords(hip_vL, kn_vL, is_viewer_left=True)
    t_vR, s_glute_vR = _femur_coords(hip_vR, kn_vR, is_viewer_left=False)

    def _leg_hem_gate(t, s_glute):
        quad_arch = -0.05 * np.clip((-s_glute) / 42.0, 0.0, 1.2)
        glute_wrap = (0.04 + 0.36 * squat_depth) * np.clip(s_glute / 36.0, 0.0, 1.35)
        t_hem = (0.25 - 0.05 * squat_depth) + quad_arch + glute_wrap
        return np.clip((t_hem - t) / 0.038, 0.0, 1.0)

    gate_vL = _leg_hem_gate(t_vL, s_glute_vL)
    gate_vR = _leg_hem_gate(t_vR, s_glute_vR)

    side_blend = np.clip((xx - hip_cx) / 14.0 + 0.5, 0.0, 1.0)
    thigh_hem_gate = gate_vL * (1.0 - side_blend) + gate_vR * side_blend

    groin_glute_core = (
        np.exp(-(((xx - hip_cx) / (hip_span * 0.50)) ** 4))
        * np.clip(((crotch_y_split + 26.0) - yy) / 6.0, 0.0, 1.0)
    )

    y_glute_max = (hip_cy + 0.48 * (kn_cy - hip_cy)) * (1.0 - squat_depth) + (crotch_y_split + 26.0) * squat_depth
    glute_bot_gate = np.clip((y_glute_max - yy + 1.0) / 2.5, 0.0, 1.0)

    fg_gate = np.clip((human_mask - 0.18) / 0.45, 0.0, 1.0)
    shorts_mask = np.clip(top_gate * np.maximum(thigh_hem_gate, groin_glute_core) * glute_bot_gate * fg_gate, 0.0, 1.0)

    band_mask = (
        np.clip((yy - waist_top_y + 1.0) / 1.6, 0.0, 1.0)
        * np.clip((waist_bot_y - yy + 1.0) / 1.6, 0.0, 1.0)
        * shorts_mask
    )

    base_3d = np.clip(sculpt_curve, 0.16, 0.92)
    dx_h = (xx - hip_cx) / max(55.0, hip_span * 0.72)
    thigh_cyl_L = np.exp(-(((dx_h + 0.48) / 0.38) ** 2))
    thigh_cyl_R = np.exp(-(((dx_h - 0.48) / 0.38) ** 2))
    pouch_3d = np.exp(-(((dx_h) / 0.20) ** 2 + ((yy - (waist_bot_y + 0.42 * (crotch_y_split - waist_bot_y))) / 22.0) ** 2))
    center_seam = np.exp(-((dx_h / 0.018) ** 2)) * np.clip((yy - waist_bot_y) / 6.0, 0.0, 1.0)
    hem_edge = np.exp(-(((thigh_hem_gate - 0.52) / 0.16) ** 2)) * shorts_mask * (1.0 - band_mask)

    fabric_lit = np.clip(
        0.58 * base_3d
        + 0.26 * np.maximum(thigh_cyl_L, thigh_cyl_R)
        + 0.16 * pouch_3d
        - 0.10 * center_seam,
        0.0,
        1.0,
    )

    sh_r = 16.0 + 34.0 * (fabric_lit ** 0.95) + 10.0 * hem_edge
    sh_g = 18.0 + 36.0 * (fabric_lit ** 0.95) + 11.0 * hem_edge
    sh_b = 21.0 + 40.0 * (fabric_lit ** 0.95) + 13.0 * hem_edge

    band_top_hi = np.exp(-(((yy - (waist_top_y + 1.2)) / 1.0) ** 2))
    band_bot_seam = np.exp(-(((yy - (waist_bot_y - 1.0)) / 1.0) ** 2))
    band_lit = 0.62 * base_3d + 0.38 * np.exp(-((dx_w / 0.75) ** 2))

    bd_r = 12.0 + 22.0 * band_lit + 24.0 * band_top_hi + 12.0 * band_bot_seam
    bd_g = 14.0 + 24.0 * band_lit + 26.0 * band_top_hi + 13.0 * band_bot_seam
    bd_b = 17.0 + 27.0 * band_lit + 30.0 * band_top_hi + 16.0 * band_bot_seam

    shorts_rgb = np.stack([sh_r, sh_g, sh_b], axis=-1)
    band_rgb = np.stack([bd_r, bd_g, bd_b], axis=-1)
    bw = np.clip(band_mask, 0.0, 1.0)[:, :, None]
    shorts_rgb = shorts_rgb * (1.0 - bw) + band_rgb * bw

    res = out_rgb * (1.0 - shorts_mask[:, :, None]) + shorts_rgb * shorts_mask[:, :, None]
    return res, shorts_mask


def render_4step_frame_1080p(
    raw_bgr_720p: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    visual_mode: str,  # "bone_blood" (Steps 1 & 3) or "muscle_blood" (Steps 2 & 4)
    pulse_phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Renders the 1080x1920 frame with:
    1. Large, screen-filling mannequin scale (`scale = 1.25 ~ 1.30`, feet at `y = 1615`),
    2. Fitted 3D matte charcoal short drawers (`apply_fitted_3d_drawers_720p`),
    3. Authentic Day 1 X-Ray Bone + Vascular Flow (`build_clean_d4_variants.py` / `gymform_engine.py`) for `bone_blood`,
       and Authentic Day 1/Day 2 3D Intrinsic Fiber-Strand Muscle Activation (`build_v36_publish_reels.py`) for `muscle_blood`.
    """
    if step_idx in (1, 2) and is_wrong:
        raw_bgr_720p = p62._apply_step1_knee_valgus_720p(raw_bgr_720p, 1.0)

    sh_h, sw_w = raw_bgr_720p.shape[:2]
    detector = uar.get_pose_detector()
    rgb_720 = cv2.cvtColor(raw_bgr_720p, cv2.COLOR_BGR2RGB)
    mp_res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb_720))
    if mp_res.pose_landmarks and mp_res.segmentation_masks:
        lm = mp_res.pose_landmarks[0]
        mp_seg = mp_res.segmentation_masks[0].numpy_view()[:, :, 0]
        globals()["_LAST_MP_LM_SEG"] = (lm, mp_seg)
    else:
        lm, mp_seg = globals()["_LAST_MP_LM_SEG"]

    full_fg_mask, human_mask, equip_mask = bcd.make_clean_barbell_and_human_mask(rgb_720, lm, mp_seg)

    rgb = rgb_720.astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    non_red_lum = 0.30 * g_ch + 0.70 * b_ch
    rb_diff = np.maximum(0.0, r_ch - b_ch)
    gb_diff = np.maximum(0.0, g_ch - b_ch)

    sh_l = (lm[11].x * sw_w, lm[11].y * sh_h)
    sh_r = (lm[12].x * sw_w, lm[12].y * sh_h)
    sh_pt = (0.5 * (sh_l[0] + sh_r[0]), 0.5 * (sh_l[1] + sh_r[1]))
    hip_r = (lm[24].x * sw_w, lm[24].y * sh_h)
    hip_l = (lm[23].x * sw_w, lm[23].y * sh_h)
    kn_r = (lm[26].x * sw_w, lm[26].y * sh_h)
    kn_l = (lm[25].x * sw_w, lm[25].y * sh_h)
    ank_r = (lm[28].x * sw_w, lm[28].y * sh_h)
    ank_l = (lm[27].x * sw_w, lm[27].y * sh_h)

    kp_720 = {
        "sh": sh_pt,
        "hip_r": hip_r,
        "hip_l": hip_l,
        "kn_r": kn_r,
        "kn_l": kn_l,
        "ank_r": ank_r,
        "ank_l": ank_l,
    }

    yy, xx = np.mgrid[:sh_h, :sw_w].astype(np.float32)
    hip_mid_y = 0.5 * (hip_r[1] + hip_l[1])

    # --- 1. PEARL-SILVER MATTE MANNEQUIN SKIN (Day 1 / Day 2 Master Tone) ---
    red_any = np.clip((rb_diff - 10.0) / 75.0, 0.0, 1.0)
    neutral_raw = base_lum * (1.0 - red_any) + (non_red_lum * 1.18 + 14.0) * red_any
    lum_smooth = cv2.GaussianBlur(neutral_raw, (0, 0), 1.5)
    lum_clean = lum_smooth + (neutral_raw - lum_smooth) * 0.42

    raw_lo, raw_hi = (21.0, 86.0) if is_wrong else (19.0, 82.0)
    norm_body = np.clip((lum_clean - raw_lo) / (raw_hi - raw_lo), 0.0, 1.15)
    sculpt_curve = np.clip(norm_body, 0.0, 1.0) ** 0.54 + 0.16 * np.clip(norm_body - 0.70, 0.0, 0.45)

    body_r = 82.0 + 106.0 * (sculpt_curve ** 0.82)
    body_g = 85.0 + 108.0 * (sculpt_curve ** 0.82)
    body_b = 89.0 + 111.0 * (sculpt_curve ** 0.82)

    plate_zone = np.clip(equip_mask * (1.0 - human_mask * 0.90), 0.0, 1.0)
    plate_norm = np.clip((lum_clean - 24.0) / 135.0, 0.0, 1.0) ** 0.78
    plate_r = 52.0 + 118.0 * plate_norm
    plate_g = 55.0 + 120.0 * plate_norm
    plate_b = 60.0 + 124.0 * plate_norm

    mono_r = body_r * (1.0 - plate_zone) + plate_r * plate_zone
    mono_g = body_g * (1.0 - plate_zone) + plate_g * plate_zone
    mono_b = body_b * (1.0 - plate_zone) + plate_b * plate_zone
    mono = np.stack([np.clip(mono_r, 0, 238), np.clip(mono_g, 0, 242), np.clip(mono_b, 0, 246)], axis=-1)

    # Pre-compute Day 1 / Day 2 3D anatomical ridge features (`build_v36_publish_reels.py` & `gymform_engine.py`)
    fiber_signal = 0.42 * r_ch + 0.58 * g_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 0.9)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.4)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 6.5)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 14.0)

    hf_ridge = (sigma_micro - sigma_ridge) * 3.2 + (sigma_ridge - sigma_belly) * 1.80
    fs = np.clip(0.48 + hf_ridge / 38.0, 0.0, 1.0) ** 1.32
    hv = np.clip((sigma_belly - 24.0) / 145.0, 0.0, 1.0)
    sulc = np.clip((sigma_broad - sigma_ridge - 2.8) / 19.0, 0.0, 1.0)

    bone_detail = 0.38 * r_ch + 0.62 * g_ch
    blur_b = cv2.GaussianBlur(bone_detail, (0, 0), 1.4)
    bone_sharp = np.clip((bone_detail + (bone_detail - blur_b) * 2.2 - 18.0) / 185.0, 0.0, 1.0)

    d_fem_r, t_fem_r = p62._segment_distance_field(xx, yy, hip_r, kn_r, 0.16, 0.98)
    d_fem_l, t_fem_l = p62._segment_distance_field(xx, yy, hip_l, kn_l, 0.16, 0.98)
    d_tib_r, t_tib_r = p62._segment_distance_field(xx, yy, kn_r, ank_r, 0.02, 0.88)
    d_tib_l, t_tib_l = p62._segment_distance_field(xx, yy, kn_l, ank_l, 0.02, 0.88)

    y_chest = sh_pt[1] + 0.38 * (hip_mid_y - sh_pt[1])
    y_ank_cut = 0.5 * (ank_r[1] + ank_l[1]) + 10.0
    leg_gate = np.clip((yy - y_chest) / 28.0, 0.0, 1.0) * np.clip((y_ank_cut - yy) / 22.0, 0.0, 1.0) * human_mask

    if visual_mode == "bone_blood":
        # ==============================================================================
        # STEPS 1 & 3: AUTHENTIC DAY 1 3D X-RAY BONE + VASCULAR ENERGY FLOW (`build_clean_d4_variants.py`)
        # ==============================================================================
        condyle_r = np.exp(-(((xx - kn_r[0]) / 26.0) ** 2 + ((yy - kn_r[1]) / 24.0) ** 2))
        condyle_l = np.exp(-(((xx - kn_l[0]) / 26.0) ** 2 + ((yy - kn_l[1]) / 24.0) ** 2))

        w_fem_r = 12.5 + 11.5 * np.exp(-((t_fem_r - 0.94) / 0.14) ** 2) + 7.0 * np.exp(-((t_fem_r - 0.12) / 0.12) ** 2)
        w_fem_l = 12.5 + 11.5 * np.exp(-((t_fem_l - 0.94) / 0.14) ** 2) + 7.0 * np.exp(-((t_fem_l - 0.12) / 0.12) ** 2)
        w_tib_r = 10.5 + 10.5 * np.exp(-((t_tib_r - 0.06) / 0.14) ** 2)
        w_tib_l = 10.5 + 10.5 * np.exp(-((t_tib_l - 0.06) / 0.14) ** 2)

        u_fem_r = np.clip(d_fem_r / np.maximum(1.0, w_fem_r), 0.0, 1.5)
        u_fem_l = np.clip(d_fem_l / np.maximum(1.0, w_fem_l), 0.0, 1.5)
        u_tib_r = np.clip(d_tib_r / np.maximum(1.0, w_tib_r), 0.0, 1.5)
        u_tib_l = np.clip(d_tib_l / np.maximum(1.0, w_tib_l), 0.0, 1.5)
        u_bone = np.minimum(np.minimum(u_fem_r, u_fem_l), np.minimum(u_tib_r, u_tib_l))

        bone_cyl = np.clip(1.0 - (u_bone ** 1.8), 0.0, 1.0)
        cortical_edge = np.exp(-((u_bone - 0.74) / 0.16) ** 2) * np.clip(1.15 - u_bone, 0.0, 1.0)

        flow_wave_r = 0.68 + 0.32 * np.cos(2.0 * math.pi * (t_fem_r * 1.8 - pulse_phase))
        flow_wave_l = 0.68 + 0.32 * np.cos(2.0 * math.pi * (t_fem_l * 1.8 - pulse_phase))
        flow_wave = np.maximum(flow_wave_r * (u_fem_r < 1.25), flow_wave_l * (u_fem_l < 1.25))
        joint_heat = np.clip(condyle_r + condyle_l, 0.0, 1.2)

        bone_presence = np.clip((bone_cyl * 0.85 + cortical_edge * 0.55 + joint_heat * 0.78) * leg_gate, 0.0, 1.0)
        # Soft Day 1 translucent X-Ray cavity (`0.46` depth, never a muddy blotch!)
        xray_window = cv2.GaussianBlur(bone_presence, (0, 0), 13.0)
        xray_window = np.clip(xray_window * 1.22, 0.0, 0.66) * leg_gate

        out = mono * (1.0 - xray_window[:, :, None] * 0.46)
        trabecular = np.clip((bone_sharp - 0.24) / 0.64, 0.0, 1.0) * cortical_edge * leg_gate
        b_val = np.clip(
            bone_cyl * (0.45 + 0.55 * bone_sharp) * (0.78 + 0.28 * flow_wave)
            + joint_heat * 0.50
            + cortical_edge * 0.40,
            0.0,
            1.0,
        )

        if is_wrong:
            # Exact Day 1 Crimson-Orange-Gold Thermal X-Ray Bone + Vascular Stress Ridges
            col_r = 105.0 + 150.0 * (b_val ** 0.50)
            col_g = 18.0 + 125.0 * (b_val ** 1.18) + 115.0 * (b_val ** 2.4) + 38.0 * joint_heat
            col_b = 10.0 + 42.0 * (b_val ** 1.45) + 135.0 * (b_val ** 2.8)
            bone_rgb = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)

            musc_stress = np.clip((rb_diff - 14.0) / 42.0, 0.0, 1.0) * leg_gate * (1.0 - bone_presence * 0.65)
            stress_rgb = np.stack([
                65.0 + 180.0 * (fs ** 0.85),
                18.0 + 48.0 * (fs ** 1.15),
                18.0 + 35.0 * (fs ** 1.30),
            ], axis=-1)
            out = out * (1.0 - (musc_stress * 0.48)[:, :, None]) + stress_rgb * (musc_stress * 0.48)[:, :, None]
        else:
            # Exact Day 1 Emerald-Cyan Bio-X-Ray Bone + Vascular Energy Ridges
            col_r = 12.0 + 65.0 * (b_val ** 1.35) + 155.0 * (b_val ** 2.7)
            col_g = 95.0 + 160.0 * (b_val ** 0.52)
            col_b = 48.0 + 115.0 * (b_val ** 1.10) + 92.0 * (b_val ** 2.3) + 35.0 * joint_heat
            bone_rgb = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)

            musc_act = np.clip((rb_diff - 12.0) / 38.0, 0.0, 1.0) * leg_gate * (1.0 - bone_presence * 0.60)
            act_rgb = np.stack([
                16.0 + 55.0 * (fs ** 1.10),
                75.0 + 175.0 * (fs ** 0.85),
                42.0 + 115.0 * (fs ** 1.00),
            ], axis=-1)
            out = out * (1.0 - (musc_act * 0.52)[:, :, None]) + act_rgb * (musc_act * 0.52)[:, :, None]

        fill_alpha = np.clip((bone_presence ** 0.78) * 0.95, 0.0, 0.96)[:, :, None]
        out = out * (1.0 - fill_alpha) + bone_rgb * fill_alpha
        out = np.clip(out + trabecular[:, :, None] * np.array([190.0, 200.0, 215.0], dtype=np.float32) * 0.58, 0.0, 255.0)

    else:
        # ==============================================================================
        # STEPS 2 & 4: AUTHENTIC DAY 1 / DAY 2 3D INTRINSIC FIBER-STRAND MUSCLE SHADER (`build_v36_publish_reels.py`)
        # ==============================================================================
        # Combine the raw 3D anatomical red muscle channel (`rb_diff`) with the quadriceps geometric envelope
        # so the 3D muscle bellies (Vastus Medialis, Rectus Femoris, Vastus Lateralis) and their individual striated fibers
        # light up with deep anatomical volume, dark sulci separation (`sulc`), and glowing blood-flow crests (`crest_col`)!
        w_quad_r = 25.0 + 14.0 * np.sin(np.clip(t_fem_r, 0.0, 1.0) * math.pi)
        w_quad_l = 25.0 + 14.0 * np.sin(np.clip(t_fem_l, 0.0, 1.0) * math.pi)
        q_env_r = np.clip(1.18 - d_fem_r / np.maximum(1.0, w_quad_r), 0.0, 1.0) * (t_fem_r > 0.14) * (t_fem_r < 0.93)
        q_env_l = np.clip(1.18 - d_fem_l / np.maximum(1.0, w_quad_l), 0.0, 1.0) * (t_fem_l > 0.14) * (t_fem_l < 0.93)
        quad_env = cv2.GaussianBlur(np.maximum(q_env_r, q_env_l), (0, 0), 4.0) * leg_gate

        fe_raw = np.clip((rb_diff - 12.0) / 42.0, 0.0, 1.0) * leg_gate
        fe = np.clip(np.maximum(fe_raw, quad_env * 0.88), 0.0, 1.0)
        hc = np.clip((gb_diff - 8.0) / 70.0, 0.0, 1.0) * leg_gate

        # Traveling blood-flow wave along the muscle belly from hip to knee
        blood_wave_r = 0.78 + 0.22 * np.cos(2.0 * math.pi * (t_fem_r * 1.6 - pulse_phase))
        blood_wave_l = 0.78 + 0.22 * np.cos(2.0 * math.pi * (t_fem_l * 1.6 - pulse_phase))
        blood_wave = np.maximum(blood_wave_r * (q_env_r > 0.05), blood_wave_l * (q_env_l > 0.05))

        strand_mod = (0.20 + 0.80 * fs) * (1.0 - 0.68 * sulc)
        fiber_alpha = np.clip(fe * (0.62 + 0.68 * strand_mod), 0.0, 0.95)[:, :, None]
        ridge_boost = np.clip((fs * 0.72 + hc * 0.45) * blood_wave, 0.0, 1.0)[:, :, None]
        vol_shade = np.clip(0.26 + 0.74 * hv * (0.42 + 0.58 * fs), 0.0, 1.0)[:, :, None]

        if is_wrong:
            # Step 2 (`Wrong Muscle`): Lateral quad / tensor fasciae latae shear tension in crimson-orange
            deep_col = np.array([145.0, 16.0, 15.0], dtype=np.float32)
            mid_col = np.array([245.0, 50.0, 32.0], dtype=np.float32)
            crest_col = np.array([255.0, 195.0, 110.0], dtype=np.float32)
        else:
            # Step 4 (`Correct Muscle`): Full Quadriceps (VMO, Rectus Femoris, Vastus Lateralis) activation in rich emerald-mint
            deep_col = np.array([8.0, 108.0, 46.0], dtype=np.float32)
            mid_col = np.array([22.0, 224.0, 96.0], dtype=np.float32)
            crest_col = np.array([185.0, 255.0, 208.0], dtype=np.float32)

        fiber_rgb = (
            deep_col * (1.0 - vol_shade)
            + mid_col * vol_shade * (1.0 - ridge_boost * 0.62)
            + crest_col * (ridge_boost ** 1.35) * 0.88
        )
        out = mono * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha

    # --- 3. DRESS MANNEQUIN IN FITTED 3D SHORT DRAWERS (`드로즈`) ---
    out, shorts_mask_720 = apply_fitted_3d_drawers_720p(out, human_mask, lm, sculpt_curve)

    # --- 4. PLACE INTO 1080x1920 STUDIO CANVAS WITH LARGER MANNEQUIN SCALE (`1.26x ~ 1.30x`) ---
    # Since the top preview capsule is now compact (`y = 148..296`), the mannequin fills the screen
    # from `y ~ 345` (when standing) down to `y_feet = 1615`!
    step_fixed_cam = {
        1: (1.375, 360.0, 1037.0),
        2: (1.375, 360.0, 1037.0),
        3: (1.345, 360.0, 1062.0),
        4: (1.345, 360.0, 1062.0),
    }
    scale, cx_fg, y_bot_fg = step_fixed_cam.get(step_idx, (1.360, 360.0, 1050.0))
    y_anchor = 1658.0

    nw, nh = int(round(sw_w * scale)), int(round(sh_h * scale))
    x0 = int(round(540.0 - cx_fg * scale))
    y0 = int(round(y_anchor - y_bot_fg * scale))

    res_man = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    res_mask = cv2.resize(full_fg_mask, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    res_shorts = cv2.resize(shorts_mask_720, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    canvas = np.full((H, W, 3), [28.0, 31.0, 36.0], dtype=np.float32)
    mask_1080 = np.zeros((H, W), dtype=np.float32)
    shorts_1080 = np.zeros((H, W), dtype=np.float32)

    sy0, sy1 = max(0, -y0), min(nh, H - y0)
    sx0, sx1 = max(0, -x0), min(nw, W - x0)
    dy0, dy1 = max(0, y0), min(H, y0 + nh)
    dx0, dx1 = max(0, x0), min(W, x0 + nw)

    if sy1 > sy0 and sx1 > sx0:
        m_crop = np.clip(res_mask[sy0:sy1, sx0:sx1], 0.0, 1.0)[:, :, None]
        canvas[dy0:dy1, dx0:dx1] = canvas[dy0:dy1, dx0:dx1] * (1.0 - m_crop) + res_man[sy0:sy1, sx0:sx1] * m_crop
        mask_1080[dy0:dy1, dx0:dx1] = res_mask[sy0:sy1, sx0:sx1]
        shorts_1080[dy0:dy1, dx0:dx1] = res_shorts[sy0:sy1, sx0:sx1]

    # Floor shadow under feet
    fl_y, fl_x = np.mgrid[1615:1705, 0:W].astype(np.float32)
    fl_sh = np.exp(-(((fl_x - 540.0) / 305.0) ** 2 + ((fl_y - 1662.0) / 24.0) ** 2)) * 0.42
    canvas[1615:1705, :] *= 1.0 - fl_sh[:, :, None] * (1.0 - mask_1080[1615:1705, :, None])

    def map_pt(pt):
        return (float(pt[0] * scale + x0), float(pt[1] * scale + y0))

    kp_1080 = {k: map_pt(v) for k, v in kp_720.items()}

    # --- 5. @demicstory LASER JOINT-ALIGNMENT LINES (CLIPPED BELOW DRAWERS SO THEY NEVER CROSS THE BLACK TRUNKS) ---
    base_rgb = np.clip(canvas, 0, 255).astype(np.uint8)
    laser_rgb = bcd.bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        base_rgb, mask_1080, kp_1080, is_wrong, with_exterior_callouts=False, with_laser_axis=True
    )
    sh_gate = np.clip(shorts_1080 * 1.4, 0.0, 1.0)[:, :, None]
    out_1080 = laser_rgb.astype(np.float32) * (1.0 - sh_gate) + base_rgb.astype(np.float32) * sh_gate
    return np.clip(out_1080, 0, 255).astype(np.uint8), mask_1080, kp_1080


def draw_compact_demic_top_preview(frame_rgb: np.ndarray, torso_icons: list, active_slot: int, is_wrong: bool) -> np.ndarray:
    """
    Compact Centered @demicstory Top Preview Capsule (`pill_w = 576, pill_h = 148, x = 252..828, y = 148..296`):
    - Translucent frosted dark glass capsule with 3 upright anatomical figure slots (`tw = 142, th = 168`),
    - Active center slot (`x = 540`) highlighted with glowing Red/Orange (Wrong) or Emerald Green (Correct) border,
    - Compact height (`bottom y = 296`) leaves plenty of headroom so the larger `1.285x` mannequin & barbell are never covered!
    """
    from PIL import ImageFilter

    bar_x0, bar_y0, bar_x1, bar_y1 = 258, 132, 822, 274
    tw, th = 136, 160
    slot_centers_x = [354, 540, 726]

    raw_pil = Image.fromarray(frame_rgb).convert("RGBA")
    blur_pil = Image.fromarray(frame_rgb).filter(ImageFilter.GaussianBlur(radius=9)).convert("RGBA")
    frost_mask = Image.new("L", (W, H), 0)
    fmd = ImageDraw.Draw(frost_mask)
    fmd.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=24, fill=190)
    cx_act = slot_centers_x[active_slot]
    fmd.rounded_rectangle([cx_act - tw // 2, bar_y0 - 10, cx_act + tw // 2, bar_y0 - 10 + th], radius=20, fill=210)
    base_pil = Image.composite(blur_pil, raw_pil, frost_mask)

    glass_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glass_ov)
    gd.rounded_rectangle([bar_x0 + 2, bar_y0 + 6, bar_x1 + 2, bar_y1 + 8], radius=26, fill=(0, 0, 0, 75))
    gd.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=24, fill=(22, 25, 30, 138), outline=(140, 150, 168, 125), width=2)
    base_pil = Image.alpha_composite(base_pil, glass_ov)

    yy_t, xx_t = np.mgrid[:th, :tw].astype(np.float32)
    nx = (xx_t - tw * 0.5) / (tw * 0.48)
    ny = (yy_t - th * 0.52) / (th * 0.48)
    vign = np.clip(1.0 - 0.55 * ((nx * nx + ny * ny) ** 1.35), 0.15, 1.0)

    for i in range(3):
        if i == active_slot:
            continue
        cx = slot_centers_x[i]
        tx0, ty0 = cx - tw // 2, bar_y0 - 10
        _, inact_224 = torso_icons[i]
        inact_arr = cv2.resize(inact_224, (tw, th), interpolation=cv2.INTER_AREA)
        inact_cropped = np.clip(inact_arr * vign[:, :, None], 0, 255).astype(np.uint8)
        im_slot = Image.fromarray(inact_cropped).convert("RGBA")
        mask_slot = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask_slot).rounded_rectangle([3, 12, tw - 4, th - 13], radius=16, fill=182)
        mask_slot = mask_slot.filter(ImageFilter.GaussianBlur(radius=3))
        base_pil.paste(im_slot, (tx0, ty0), mask_slot)

    ax0, ay0 = cx_act - tw // 2, bar_y0 - 10
    ax1, ay1 = ax0 + tw, ay0 + th
    act_col = (255, 88, 48) if is_wrong else (42, 245, 135)

    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gld = ImageDraw.Draw(glow_layer)
    gld.rounded_rectangle([ax0 - 6, ay0 - 6, ax1 + 6, ay1 + 6], radius=24, outline=(*act_col, 155), width=7)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=10))
    base_pil = Image.alpha_composite(base_pil, glow_layer)

    act_224, _ = torso_icons[active_slot]
    act_arr = cv2.resize(act_224, (tw, th), interpolation=cv2.INTER_AREA)
    rad_glow = np.exp(-(((xx_t - tw * 0.5) / 50.0) ** 2 + ((yy_t - th * 0.46) / 60.0) ** 2))[:, :, None]
    card_bg = np.array([26.0, 29.0, 35.0], dtype=np.float32) + rad_glow * np.array(act_col, dtype=np.float32) * 0.34
    fg_a = np.clip((act_arr.mean(axis=2, keepdims=True) - 44.0) / 14.0, 0.0, 1.0)
    comp_act = np.clip(card_bg * (1.0 - fg_a) + act_arr * fg_a, 0, 255).astype(np.uint8)

    im_act = Image.fromarray(comp_act).convert("RGBA")
    mask_act = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(mask_act).rounded_rectangle([0, 0, tw - 1, th - 1], radius=20, fill=215)
    base_pil.paste(im_act, (ax0, ay0), mask_act)

    bdr_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd_d = ImageDraw.Draw(bdr_ov)
    bd_d.rounded_rectangle([ax0 - 2, ay0 - 2, ax1 + 2, ay1 + 2], radius=22, outline=(*act_col, 95), width=4)
    bd_d.rounded_rectangle([ax0, ay0, ax1, ay1], radius=20, outline=(*act_col, 255), width=3)
    base_pil = Image.alpha_composite(base_pil, bdr_ov)

    return np.array(base_pil.convert("RGB"))


def draw_step_simple2_callout(
    frame_rgb: np.ndarray,
    kp_1080: dict,
    is_wrong: bool,
    deg_str: str,
    l1_str: str,
    l2_str: str,
) -> np.ndarray:
    base = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    kn_r = (int(kp_1080["kn_r"][0]), int(kp_1080["kn_r"][1]))
    bcd._draw_precision_leader(rd, kn_r, (kn_r[0] - 48, kn_r[1] - 32), (242, kn_r[1] - 32), col)

    tx, ty = 36, kn_r[1] - 108
    f_big = v30.f_pretendard(86, "extrabold")
    f_l1 = v30.f_pretendard(21, "extrabold")
    f_l2 = v30.f_pretendard(16, "bold")

    rd.text((tx + 2, ty + 3), deg_str, font=f_big, fill=(0, 0, 0, 210))
    rd.text((tx, ty), deg_str, font=f_big, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 92), l1_str, font=f_l1, fill=(col[0], col[1], col[2], 255))
    rd.text((tx + 2, ty + 119), l2_str, font=f_l2, fill=(215, 222, 232, 235))

    return np.array(Image.alpha_composite(base, ov).convert("RGB"))


def draw_single_line_kaushan_title(canvas_pil: Image.Image, exercise_name: str = "Barbell Back Squat"):
    """
    Renders the confirmed single-line full exercise name (`Barbell Back Squat`)
    in Font #1 (`Kaushan Script`) at the bottom (`y_center = 1742`),
    with ZERO flanking horizontal lines (`-`) on either side!
    """
    SS = 2
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    try:
        f_scr = ImageFont.truetype(CURSIVE_FONT_PATH, 62 * SS)
    except Exception:
        f_scr = v30.load_font(56 * SS, bold=True)

    bb = od.textbbox((0, 0), exercise_name, font=f_scr)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = (W * SS - tw) // 2 - bb[0]
    ty = 1756 * SS - th // 2 - bb[1]

    for dx_s, dy_s, a_s in [(0, 5 * SS, 185), (-2 * SS, 3 * SS, 110), (2 * SS, 3 * SS, 110)]:
        od.text((tx + dx_s, ty + dy_s), exercise_name, font=f_scr, fill=(8, 10, 14, a_s))
    od.text((tx, ty), exercise_name, font=f_scr, fill=(246, 250, 255, 252))

    ov_1080 = ov.resize((W, H), Image.Resampling.LANCZOS)
    canvas_pil.alpha_composite(ov_1080)


import build_simple_4concepts as s4

def main():
    print("Loading raw sequences...")
    fn1, i0_1, i1_1, *_ = p62.STEP_POOL_SPECS[1]
    fn2, i0_2, i1_2, *_ = p62.STEP_POOL_SPECS[2]
    seq_wrong = p62._load_slice(os.path.join(OUT_DIR, fn1), i0_1, i1_1, target_len=52)
    seq_right = p62._load_slice(os.path.join(OUT_DIR, fn2), i0_2, i1_2, target_len=52)

    # Ensure top preview icons also use our fitted 3D drawers
    bcd.apply_3d_athletic_briefs_720p = apply_fitted_3d_drawers_720p
    s4._UPRIGHT_ANATOMY_CACHE.clear()
    icons_wrong = s4.make_unsquished_day1_top_icons({}, current_step=1)
    icons_right = s4.make_unsquished_day1_top_icons({}, current_step=2)

    steps_cfg = [
        (1, seq_wrong[-1], True,  "bone_blood",   "16°", "KNEE CAVE",   "86% JOINT RISK",  "1. 오자세 뼈·혈류 (X-Ray Bone)"),
        (2, seq_wrong[-1], True,  "muscle_blood", "16°", "VALGUS LOAD", "78% QUAD LOSS",   "2. 오자세 근육·혈류 (Muscle Strain)"),
        (3, seq_right[-1], False, "bone_blood",   "0°",  "KNEE LOCK",   "96% JOINT SAFE",  "3. 정자세 뼈·혈류 (Aligned Bone)"),
        (4, seq_right[-1], False, "muscle_blood", "0°",  "FULL TRACK",  "98% QUAD DRIVE",  "4. 정자세 근육·혈류 (Full Quad Drive)"),
    ]

    panels = []
    for step_idx, raw_fr, is_w, v_mode, deg_s, l1_s, l2_s, title_kr in steps_cfg:
        print(f"Rendering Step {step_idx} ({v_mode})...")
        fr_1080, mask_1080, kp_1080 = render_4step_frame_1080p(raw_fr, step_idx, is_w, v_mode, pulse_phase=0.15)
        fr_1080 = draw_step_simple2_callout(fr_1080, kp_1080, is_w, deg_s, l1_s, l2_s)
        icons = icons_wrong if is_w else icons_right
        fr_1080 = draw_compact_demic_top_preview(fr_1080, icons, active_slot=1, is_wrong=is_w)
        pil = Image.fromarray(fr_1080).convert("RGBA")
        draw_single_line_kaushan_title(pil, "Barbell Back Squat")
        out_rgb = pil.convert("RGB")
        out_rgb.save(os.path.join(OUT_DIR, f"day3_step{step_idx}_{v_mode}.jpg"), quality=95)
        panels.append((title_kr, is_w, out_rgb))

    pw, ph = 520, 924
    pad = 22
    header_h = 96
    sheet_w = pad + 4 * (pw + pad)
    sheet_h = header_h + ph + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 16, 20))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(26, "extrabold")
    f_s = v30.f_pretendard(18, "bold")

    sd.text(
        (28, 18),
        "Day 3 4구간 마스터 — 밀착 3D 드로즈 + 마네킹 확대(1.285x) + Day 1 X-Ray 뼈·근섬유 + 1번 필기체(양옆 - 제거)",
        font=f_t,
        fill=(242, 246, 252),
    )
    sd.text(
        (28, 54),
        "Step 1: 오자세 뼈·혈류(Day 1 X-Ray) | Step 2: 오자세 근육·혈류 | Step 3: 정자세 뼈·혈류(Day 1 X-Ray) | Step 4: 정자세 근육·혈류 + Barbell Back Squat (Kaushan Script)",
        font=f_s,
        fill=(185, 225, 255),
    )

    for idx, (label, is_w, p_im) in enumerate(panels):
        x0 = pad + idx * (pw + pad)
        y0 = header_h
        col = (255, 95, 78) if is_w else (65, 245, 135)
        rs = p_im.resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(rs, (x0, y0))
        sd.rounded_rectangle([x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 2], radius=12, outline=col, width=3)
        sd.rounded_rectangle([x0 + 10, y0 + 10, x0 + pw - 10, y0 + 44], radius=7, fill=(12, 14, 18))
        sd.text((x0 + 18, y0 + 15), label, font=f_s, fill=col)

    out_path = os.path.join(OUT_DIR, "day3_4step_bone_muscle_cursive_sheet.jpg")
    sheet.save(out_path, quality=95)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
