#!/usr/bin/env python3
"""
Day 1 Viral Style (`5,931 views`, `01M60R7X6QEKSCKB67N9E3ERNW.png`) Applied to Day 3 Barbell Back Squat:
1. Exact Day 1 Mannequin Pearl-Silver Matte Skin Tone (`median RGB = [138, 141, 144]`, `p90 RGB = [162, 165, 169]`)
   over exact Day 1 Studio Charcoal Background (`RGB = [42, 45, 49]`), with 100% intact Barbell & Weight Plates (`0% clipping`).
2. Exact Day 1 3D Translucent Neon X-Ray Bone + Flowing Energy Effect (`t325u`: "뼈에 이렇게 네온주고 움직임에 따라 흐르던데 이런거 좋더라"):
   - Extracts the 3D femur, knee joint (patella/condyles), tibia/fibula, and cortical bone edges inside the translucent X-Ray cavity (`xray_window`),
   - Renders the glowing **Amber-Orange-Gold Neon Bone (`#FF5510 -> #FFA828 -> #FFF0A8`)** with a directional energy flow wave pulsing into the overloaded joint on Wrong (`✕`),
   - And renders **Neon Emerald-Cyan Bone + Quad Fiber Flow (`#14DC64 -> #68FFAC -> #E8FFF2`)** flowing upward through the femur & quadriceps on Right (`✓`).
3. Intuitive Meaning-First Graphic Telemetry (`t327u` & `t328u`):
   - Shows WHY the `%` matters in 0.1 seconds by comparing **Joint Stress vs Target Muscle Activation (`KNEE · JOINT 84%` vs `QUADS · VMO 16%`)**
     using Day 1's frameless 20-segment LED equalizer bars, giant sculpted `%`, 5-bar vertical icon (`≣`), and `[ 16°  KNEE VALGUS ]` pill across 4 layout variations (`시안 A, B, C, D`).
"""
import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import universal_anatomy_rig as uar
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62
import build_4variants_comparison_mockups as b4

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920
BG_DAY1 = np.array([42.0, 45.0, 49.0], dtype=np.float32)


def make_day1_studio_bg() -> np.ndarray:
    """Matches Day 1 (`01M60R7X6QEKSCKB67N9E3ERNW.png`) dark charcoal studio background (`median RGB 44,47,51`)."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    rx = (xx - W * 0.50) / (W * 0.62)
    ry = (yy - H * 0.48) / (H * 0.56)
    r2 = np.clip(rx * rx + ry * ry, 0.0, 1.45)
    vig = 1.06 - 0.24 * (r2 ** 0.88)
    bg = np.empty((H, W, 3), dtype=np.float32)
    bg[:, :, 0] = np.clip(43.0 * vig, 26.0, 52.0)
    bg[:, :, 1] = np.clip(46.0 * vig, 28.0, 55.0)
    bg[:, :, 2] = np.clip(50.0 * vig, 31.0, 60.0)
    return bg


STUDIO_BG_DAY1 = make_day1_studio_bg()


def render_day1_pearl_mannequin_and_neon_xray_1080p(
    raw_bgr_720p: np.ndarray,
    step_idx: int,
    is_wrong: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    1. Matches Day 1's bright, sculptural pearl-silver matte mannequin skin (`median RGB = 138, 141, 144`, `p90 = 165`).
    2. Preserves 100% of the barbell and weight plates (`scale <= 1.38`, zero horizontal clipping, zero plate mask holes).
    3. Synthesizes Day 1's signature **Translucent 3D X-Ray Bone + Flowing Neon Energy** inside the femur, knee joint, and tibia!
    """
    if step_idx == 1:
        raw_bgr_720p = p62._apply_step1_knee_valgus_720p(raw_bgr_720p, 1.0)

    sh_h, sw_w = raw_bgr_720p.shape[:2]
    detector = uar.get_pose_detector()
    rgb_720 = cv2.cvtColor(raw_bgr_720p, cv2.COLOR_BGR2RGB)
    mp_res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb_720))
    lm = mp_res.pose_landmarks[0]
    mp_seg = mp_res.segmentation_masks[0].numpy_view()[:, :, 0]

    rgb = rgb_720.astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    non_red_lum = 0.30 * g_ch + 0.70 * b_ch
    rb_diff = np.maximum(0.0, r_ch - b_ch)

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
    bg_ref = float(np.percentile(base_lum[60:260, 10:50], 25))
    fg_contrast = np.clip((base_lum - (bg_ref + 6.5)) / 6.0, 0.0, 1.0)

    # Full Equipment + Human Foreground Mask (100% barbell + weight plates preserved!)
    hip_mid_y = 0.5 * (hip_r[1] + hip_l[1])
    upper_equip_zone = np.clip((hip_mid_y - 10.0 - yy) / 45.0, 0.0, 1.0)
    equip_mask = fg_contrast * upper_equip_zone
    eq_u8 = (equip_mask > 0.25).astype(np.uint8) * 255
    k_eq = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    eq_closed = cv2.morphologyEx(eq_u8, cv2.MORPH_CLOSE, k_eq).astype(np.float32) / 255.0
    equip_mask = np.maximum(equip_mask, eq_closed * np.clip((base_lum - (bg_ref + 3.0)) / 5.0, 0.0, 1.0))

    human_mask = np.clip((mp_seg - 0.38) / 0.28, 0.0, 1.0)
    full_fg_mask = np.clip(np.maximum(human_mask, equip_mask), 0.0, 1.0)
    full_fg_mask = cv2.GaussianBlur(full_fg_mask, (0, 0), 1.0)

    # --- 1. EXACT DAY 1 PEARL-SILVER MATTE MANNEQUIN SKIN (`median RGB = 138, 141, 144`, `p90 = 165`) ---
    red_any = np.clip((rb_diff - 10.0) / 75.0, 0.0, 1.0)
    neutral_raw = base_lum * (1.0 - red_any) + (non_red_lum * 1.18 + 14.0) * red_any
    lum_smooth = cv2.GaussianBlur(neutral_raw, (0, 0), 1.5)
    lum_clean = lum_smooth + (neutral_raw - lum_smooth) * 0.42

    # Calibrate each step's fixed raw torso range (`p10..p92`) so ALL 4 steps have 100% IDENTICAL Day 1 Pearl-Silver skin:
    # Day 1 (`01M60R7X6QEKSCKB67N9E3ERNW.png`) measured:
    # - Mannequin Torso/Head median RGB = `[136, 139, 143]`, p90 RGB = `[162, 165, 169]`
    # - Thigh mid RGB = `[112, 115, 119]`
    # Fixed per-step raw torso calibration constants (no per-frame flicker, 100% locked across the clip):
    step_raw_cal = {
        1: (18.0, 88.0),
        2: (15.0, 76.0),
        3: (16.0, 82.0),
        4: (16.0, 82.0),
    }
    raw_lo, raw_hi = step_raw_cal.get(step_idx, (16.0, 82.0))
    norm_body = np.clip((lum_clean - raw_lo) / (raw_hi - raw_lo), 0.0, 1.15)
    # Smooth sculptural Pearl-Silver curve matching Day 1 (`01M60R7X6QEKSCKB67N9E3ERNW.png`):
    # Shadow floor starts at `RGB [88, 91, 95]`, midtone (`p50`) = `RGB [136, 139, 143]`, highlight (`p90`) = `RGB [166, 169, 174]`!
    sculpt_curve = np.clip(norm_body, 0.0, 1.0) ** 0.52 + 0.16 * np.clip(norm_body - 0.70, 0.0, 0.45)

    body_r = 88.0 + 84.0 * sculpt_curve
    body_g = 91.0 + 85.0 * sculpt_curve
    body_b = 95.0 + 86.0 * sculpt_curve

    # Outer weight plates only (`|x - sh_pt[0]| > 145` in 720p) get subtle metallic contrast so plates look like machined steel
    plate_zone = np.clip((np.abs(xx - sh_pt[0]) - 142.0) / 28.0, 0.0, 1.0) * equip_mask
    plate_norm = np.clip((lum_clean - 16.0) / 135.0, 0.0, 1.0) ** 0.75
    plate_r = 58.0 + 110.0 * plate_norm
    plate_g = 61.0 + 112.0 * plate_norm
    plate_b = 66.0 + 116.0 * plate_norm

    mono_r = body_r * (1.0 - plate_zone) + plate_r * plate_zone
    mono_g = body_g * (1.0 - plate_zone) + plate_g * plate_zone
    mono_b = body_b * (1.0 - plate_zone) + plate_b * plate_zone
    mono = np.stack([np.clip(mono_r, 0, 238), np.clip(mono_g, 0, 242), np.clip(mono_b, 0, 246)], axis=-1)

    # --- 2. DAY 1 TRANSLUCENT 3D NEON X-RAY BONE & ENERGY FLOW (`뼈에 네온 주고 움직임에 따라 흐르는 연출`) ---
    # Build anatomical 3D bone distance fields for Femur (`hip -> knee`) and Tibia (`knee -> ankle`) + Knee Joint Condyles!
    d_fem_r, t_fem_r = p62._segment_distance_field(xx, yy, hip_r, kn_r, 0.06, 0.98)
    d_fem_l, t_fem_l = p62._segment_distance_field(xx, yy, hip_l, kn_l, 0.06, 0.98)
    d_tib_r, t_tib_r = p62._segment_distance_field(xx, yy, kn_r, ank_r, 0.02, 0.88)
    d_tib_l, t_tib_l = p62._segment_distance_field(xx, yy, kn_l, ank_l, 0.02, 0.88)

    # Anatomical bone width profile: slender diaphysis (shaft `r ~ 13px` in 720p) flaring into wide femoral/tibial condyles (`r ~ 24px`) at the knee!
    condyle_r = np.exp(-(((xx - kn_r[0]) / 26.0) ** 2 + ((yy - kn_r[1]) / 24.0) ** 2))
    condyle_l = np.exp(-(((xx - kn_l[0]) / 26.0) ** 2 + ((yy - kn_l[1]) / 24.0) ** 2))
    hip_joint_r = np.exp(-(((xx - hip_r[0]) / 24.0) ** 2 + ((yy - hip_r[1]) / 24.0) ** 2))
    hip_joint_l = np.exp(-(((xx - hip_l[0]) / 24.0) ** 2 + ((yy - hip_l[1]) / 24.0) ** 2))

    w_fem_r = 12.5 + 11.5 * np.exp(-((t_fem_r - 0.94) / 0.14) ** 2) + 7.0 * np.exp(-((t_fem_r - 0.10) / 0.12) ** 2)
    w_fem_l = 12.5 + 11.5 * np.exp(-((t_fem_l - 0.94) / 0.14) ** 2) + 7.0 * np.exp(-((t_fem_l - 0.10) / 0.12) ** 2)
    w_tib_r = 10.5 + 10.5 * np.exp(-((t_tib_r - 0.06) / 0.14) ** 2)
    w_tib_l = 10.5 + 10.5 * np.exp(-((t_tib_l - 0.06) / 0.14) ** 2)

    # 3D Cylindrical Bone Shading + Cortical Rim (like Day 1's translucent X-Ray bone walls!)
    u_fem_r = np.clip(d_fem_r / np.maximum(1.0, w_fem_r), 0.0, 1.5)
    u_fem_l = np.clip(d_fem_l / np.maximum(1.0, w_fem_l), 0.0, 1.5)
    u_tib_r = np.clip(d_tib_r / np.maximum(1.0, w_tib_r), 0.0, 1.5)
    u_tib_l = np.clip(d_tib_l / np.maximum(1.0, w_tib_l), 0.0, 1.5)
    u_bone = np.minimum(np.minimum(u_fem_r, u_fem_l), np.minimum(u_tib_r, u_tib_l))

    bone_cyl = np.clip(1.0 - (u_bone ** 1.8), 0.0, 1.0)
    # Bright cortical X-Ray edge ring at `u_bone ~ 0.72` (exact Day 1 translucent X-Ray bone contour!)
    cortical_edge = np.exp(-((u_bone - 0.74) / 0.16) ** 2) * np.clip(1.15 - u_bone, 0.0, 1.0)

    # Combine with native 3D red/bone detail inside the leg
    bone_detail = 0.38 * r_ch + 0.62 * g_ch
    blur_b = cv2.GaussianBlur(bone_detail, (0, 0), 1.4)
    bone_sharp = np.clip((bone_detail + (bone_detail - blur_b) * 2.2 - 18.0) / 185.0, 0.0, 1.0)

    # Directional Flowing Energy Wave along the bone (`움직임에 따라 뼈를 타고 흐르는 네온 파동`)
    # Flows along `t_fem` (from hip `0.0` -> knee `1.0`) and `t_tib` (from ankle `1.0` -> knee `0.0`)
    flow_wave_r = 0.68 + 0.32 * np.cos(2.0 * math.pi * (t_fem_r * 1.6 - 0.82))
    flow_wave_l = 0.68 + 0.32 * np.cos(2.0 * math.pi * (t_fem_l * 1.6 - 0.82))
    flow_wave = np.maximum(flow_wave_r * (u_fem_r < 1.2), flow_wave_l * (u_fem_l < 1.2))
    joint_heat = np.clip(condyle_r + condyle_l + 0.45 * (hip_joint_r + hip_joint_l), 0.0, 1.2)

    # X-Ray Cavity Window (`xray_window`: darkens the surrounding skin slightly so the internal glowing bone pops in 3D!)
    y_chest = sh_pt[1] + 0.38 * (hip_mid_y - sh_pt[1])
    y_ank_cut = 0.5 * (ank_r[1] + ank_l[1]) + 10.0
    leg_gate = np.clip((yy - y_chest) / 30.0, 0.0, 1.0) * np.clip((y_ank_cut - yy) / 25.0, 0.0, 1.0) * human_mask

    bone_presence = np.clip((bone_cyl * 0.85 + cortical_edge * 0.55 + joint_heat * 0.75) * leg_gate, 0.0, 1.0)
    xray_window = cv2.GaussianBlur(bone_presence, (0, 0), 14.0)
    xray_window = np.clip(xray_window * 1.25, 0.0, 0.68) * leg_gate

    # Darken cavity slightly like Day 1 (`mono * (1 - xray_window * 0.46)`)
    out = mono * (1.0 - xray_window[:, :, None] * 0.46)

    # Also include outer translucent silver X-Ray trabecular/cortical lines around the joint (like Day 1's wrist/shoulder white X-Ray lines!)
    trabecular = np.clip((bone_sharp - 0.25) / 0.65, 0.0, 1.0) * cortical_edge * leg_gate

    b_val = np.clip(bone_cyl * (0.45 + 0.55 * bone_sharp) * (0.78 + 0.28 * flow_wave) + joint_heat * 0.48 + cortical_edge * 0.38, 0.0, 1.0)

    if is_wrong:
        # Day 1's Exact Fiery Neon Amber-Orange-Gold Bone Glow (`gymform_engine.py` lines 363-366)!
        col_r = 105.0 + 150.0 * (b_val ** 0.50)
        col_g = 18.0 + 125.0 * (b_val ** 1.18) + 115.0 * (b_val ** 2.4) + 38.0 * joint_heat
        col_b = 10.0 + 42.0 * (b_val ** 1.45) + 135.0 * (b_val ** 2.8)
        bone_rgb = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)

        # Subtle crimson muscle stress aura around the bone inside the thigh
        musc_stress = np.clip((rb_diff - 14.0) / 45.0, 0.0, 1.0) * leg_gate * (1.0 - bone_presence * 0.7)
        stress_rgb = np.stack([
            55.0 + 175.0 * (bone_sharp ** 0.9),
            20.0 + 42.0 * (bone_sharp ** 1.2),
            22.0 + 35.0 * (bone_sharp ** 1.3),
        ], axis=-1)
        out = out * (1.0 - (musc_stress * 0.42)[:, :, None]) + stress_rgb * (musc_stress * 0.42)[:, :, None]
    else:
        # Right Form (`✓`): Glowing Neon Emerald-Cyan Bone Axis + 3D Quadriceps Fiber Flow!
        col_r = 12.0 + 65.0 * (b_val ** 1.35) + 155.0 * (b_val ** 2.7)
        col_g = 95.0 + 160.0 * (b_val ** 0.52)
        col_b = 48.0 + 115.0 * (b_val ** 1.10) + 92.0 * (b_val ** 2.3) + 35.0 * joint_heat
        bone_rgb = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)

        # Rich 3D Emerald Quadriceps muscle activation around the aligned femur bone
        musc_act = np.clip((rb_diff - 12.0) / 38.0, 0.0, 1.0) * leg_gate * (1.0 - bone_presence * 0.55)
        act_rgb = np.stack([
            16.0 + 52.0 * (bone_sharp ** 1.15),
            65.0 + 185.0 * (bone_sharp ** 0.85),
            40.0 + 115.0 * (bone_sharp ** 1.00),
        ], axis=-1)
        out = out * (1.0 - (musc_act * 0.78)[:, :, None]) + act_rgb * (musc_act * 0.78)[:, :, None]

    # Blend the glowing 3D Neon X-Ray Bone + white-silver cortical X-Ray outlines
    fill_alpha = np.clip((bone_presence ** 0.78) * 0.94, 0.0, 0.95)[:, :, None]
    out = out * (1.0 - fill_alpha) + bone_rgb * fill_alpha
    # Add crisp white-silver X-Ray cortical rim lines (`trabecular`) like Day 1's forearm/shoulder X-Ray contour
    out = np.clip(out + trabecular[:, :, None] * np.array([185.0, 195.0, 210.0], dtype=np.float32) * 0.55, 0.0, 255.0)

    # --- 3. PLACE INTO 1080x1920 STUDIO CANVAS WITH ZERO EQUIPMENT CLIPPING ---
    ys_fg, xs_fg = np.where(full_fg_mask > 0.35)
    x_min_fg, x_max_fg = float(np.percentile(xs_fg, 0.2)), float(np.percentile(xs_fg, 99.8))
    y_top_fg, y_bot_fg = float(np.percentile(ys_fg, 0.5)), float(np.percentile(ys_fg, 99.5))

    scale_h = 735.0 / max(120.0, (y_bot_fg - y_top_fg))
    scale_w = 1004.0 / max(120.0, (x_max_fg - x_min_fg))
    scale = min(scale_h, scale_w)

    nw, nh = int(round(sw_w * scale)), int(round(sh_h * scale))
    cx_fg = 0.5 * (x_min_fg + x_max_fg)
    x0 = int(round(540.0 - cx_fg * scale))
    y0 = int(round(1310.0 - y_bot_fg * scale))

    res_man = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    res_mask = cv2.resize(full_fg_mask, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    canvas = STUDIO_BG_DAY1.copy()

    # Grounding floor contact shadow at `y = 1310px`
    ank_r_1080 = (x0 + kp_720["ank_r"][0] * scale, y0 + kp_720["ank_r"][1] * scale)
    ank_l_1080 = (x0 + kp_720["ank_l"][0] * scale, y0 + kp_720["ank_l"][1] * scale)
    yy_f, xx_f = np.mgrid[:H, :W].astype(np.float32)
    sh_f = np.zeros((H, W), dtype=np.float32)
    for ax_pt in (ank_r_1080, ank_l_1080):
        sx = (xx_f - ax_pt[0]) / 78.0
        sy = (yy_f - 1314.0) / 17.0
        sh_f = np.maximum(sh_f, np.exp(-0.5 * (sx * sx + sy * sy)))
    canvas *= (1.0 - np.clip(sh_f * 0.46, 0.0, 0.52)[:, :, None])

    dst_x1, dst_x2 = max(0, x0), min(W, x0 + nw)
    src_x1 = dst_x1 - x0
    src_x2 = src_x1 + (dst_x2 - dst_x1)

    dst_y1, dst_y2 = max(0, y0), min(H, y0 + nh)
    src_y1 = dst_y1 - y0
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    crop_man = res_man[src_y1:src_y2, src_x1:src_x2, :]
    crop_m = np.clip(res_mask[src_y1:src_y2, src_x1:src_x2], 0.0, 1.0)[:, :, None]
    canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = (
        crop_man * crop_m + canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] * (1.0 - crop_m)
    )

    full_mask_1080 = np.zeros((H, W), dtype=np.float32)
    full_mask_1080[dst_y1:dst_y2, dst_x1:dst_x2] = crop_m[:, :, 0]
    kp_1080 = {k: (x0 + kx * scale, y0 + ky * scale) for k, (kx, ky) in kp_720.items()}

    return np.clip(canvas, 0, 255).astype(np.uint8), full_mask_1080, kp_1080


def draw_day1_bone_energy_flow_and_exterior_arcs(
    frame_rgb: np.ndarray,
    body_mask: np.ndarray,
    kp: dict,
    is_wrong: bool,
    with_exterior_callouts: bool = True,
    with_laser_axis: bool = True,
) -> np.ndarray:
    """
    Combines:
    1. Flowing Neon Bone Energy Stream along the 3D Femur -> Knee -> Tibia axis (`뼈에 네온 주고 움직임에 따라 흐르는 효과`),
    2. Optional exterior curved arrows + sculpted numbers outside the body (`Day 1 screenshot style: 16° KNEE CAVE`),
    3. Optional `@demicstory` razor-thin neon laser joint-axis line.
    """
    canvas = Image.fromarray(frame_rgb).convert("RGBA")
    hip_r, hip_l = kp["hip_r"], kp["hip_l"]
    kn_r, kn_l = kp["kn_r"], kp["kn_l"]
    ank_r, ank_l = kp["ank_r"], kp["ank_l"]

    if with_laser_axis:
        if is_wrong:
            kn_rv, kn_lv = (kn_r[0] + 16.0, kn_r[1]), (kn_l[0] - 16.0, kn_l[1])
            top_r, _ = p62._extend_segment(hip_r, kn_rv, 0.25, 1.0)
            _, bot_r = p62._extend_segment(kn_rv, ank_r, 0.0, 1.12)
            top_l, _ = p62._extend_segment(hip_l, kn_lv, 0.25, 1.0)
            _, bot_l = p62._extend_segment(kn_lv, ank_l, 0.0, 1.12)
            ext_r1, ext_r2 = p62._extend_segment(hip_r, kn_rv, 0.60, 1.44)
            ext_l1, ext_l2 = p62._extend_segment(hip_l, kn_lv, 0.60, 1.44)
            m_ch, a_ch, v_p = [[top_r, kn_rv, bot_r], [top_l, kn_lv, bot_l]], [[ext_r1, ext_r2], [ext_l1, ext_l2]], [kn_rv, kn_lv]
        else:
            top_r, _ = p62._extend_segment(hip_r, kn_r, 0.25, 1.0)
            _, bot_r = p62._extend_segment(kn_r, ank_r, 0.0, 1.12)
            top_l, _ = p62._extend_segment(hip_l, kn_l, 0.25, 1.0)
            _, bot_l = p62._extend_segment(kn_l, ank_l, 0.0, 1.12)
            ext_r1, ext_r2 = p62._extend_segment(hip_r, kn_r, 0.64, 1.36)
            ext_l1, ext_l2 = p62._extend_segment(hip_l, kn_l, 0.64, 1.36)
            m_ch, a_ch, v_p = [[top_r, kn_r, bot_r], [top_l, kn_l, bot_l]], [[ext_r1, ext_r2], [ext_l1, ext_l2]], [kn_r, kn_l]

        b4.draw_subsurface_additive_laser_axis(canvas, body_mask, m_ch, a_ch, v_p, is_wrong, laser_style="hybrid_contrast")

    if with_exterior_callouts:
        SS = 2
        rig_ss = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        rd = ImageDraw.Draw(rig_ss)
        shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)

        col_main = (255, 92, 36) if is_wrong else (32, 245, 122)
        col_soft = (255, 165, 95) if is_wrong else (130, 255, 185)

        # Exterior curved arrow in the dark studio space outside the right knee (`kn_l` on viewer's right side `x ~ 660..760`)
        rx_k, ry_k = int(kn_l[0]), int(kn_l[1])
        arc_cx, arc_cy = (rx_k + 42) * SS, ry_k * SS
        r_arc = 76 * SS
        if is_wrong:
            # Curved inward-collapse arrow outside right knee (`start=-45, end=45`)
            rd.arc([arc_cx - r_arc, arc_cy - r_arc, arc_cx + r_arc, arc_cy + r_arc], start=-42, end=42, fill=(*col_main, 248), width=4 * SS)
            ang_t = math.radians(42)
            tip = (arc_cx + r_arc * math.cos(ang_t), arc_cy + r_arc * math.sin(ang_t))
            v30.draw_arrow_head(rd, tip, 42 + 90, 12 * SS, (*col_soft, 255))
            v30.draw_floating_number_badge(
                od, sd, (rx_k + 108, ry_k - 36), "16°", "KNEE CAVE", col_main, col_soft, align="left"
            )
        else:
            # Clean exterior outward/stabilizing arc outside right knee (`0° KNEE LOCK`)
            rd.arc([arc_cx - r_arc, arc_cy - r_arc, arc_cx + r_arc, arc_cy + r_arc], start=-38, end=38, fill=(*col_main, 248), width=4 * SS)
            ang_t = math.radians(-38)
            tip = (arc_cx + r_arc * math.cos(ang_t), arc_cy + r_arc * math.sin(ang_t))
            v30.draw_arrow_head(rd, tip, -38 - 90, 12 * SS, (*col_soft, 255))
            v30.draw_floating_number_badge(
                od, sd, (rx_k + 108, ry_k - 36), "0°", "KNEE LOCK", col_main, col_soft, align="left"
            )

        canvas.alpha_composite(rig_ss.resize((W, H), Image.Resampling.LANCZOS))
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
        canvas.alpha_composite(ov)

    return np.array(canvas.convert("RGB"))


def draw_day1_5bar_level_icon(od: ImageDraw.ImageDraw, x: int, y: int, col_rgb: tuple, active_bars: int = 5):
    """Draws Day 1's signature 5-horizontal-bar vertical level icon (`≣`) next to the cue title (`01M60R7X6QEKSCKB67N9E3ERNW.png`)."""
    bar_w, bar_h, gap = 24, 4, 3
    for i in range(5):
        # i=0 is top bar, i=4 is bottom bar
        by = y + (4 - i) * (bar_h + gap)
        c = (*col_rgb, 255) if i < active_bars else (90, 98, 112, 165)
        od.rounded_rectangle([x, by, x + bar_w, by + bar_h], radius=1, fill=c)


def draw_hud_variant_A_day1_exact_left_frameless(frame_rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    """
    시안 A: `Day 1 오리지널 좌하단 노박스(Frameless) 20단 듀얼 비교 게이지` (`01M60R7X6QEKSCKB67N9E3ERNW.png` 100% 동일 포맷!)
    - 박스 없이 배경 위에 바로 뜨는 대형 `84%` / `96%` + `≣` 5단 레벨 아이콘 + `INWARD KNEE` / `NEUTRAL LOCK`
    - 2줄 20단 LED 비교 바:
      * 오자세 (`✕`): `KNEE · JOINT` 84% (오렌지-레드) vs `QUADS · VMO` 16% (회색) -> 왜 나쁜지 0.1초 만에 직관적 이해!
      * 정자세 (`✓`): `QUADS · VMO` 96% (네온 그린) vs `KNEE · JOINT` 4% (회색) -> 왜 좋은지 0.1초 만에 직관적 이해!
    - 하단 캡슐: `[ 16°   KNEE VALGUS ]` / `[ 0°   ALIGNED TRACK ]`
    """
    canvas = Image.fromarray(frame_rgb).convert("RGBA")
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    accent = (255, 94, 36) if is_wrong else (32, 245, 122)
    accent_soft = (255, 155, 95) if is_wrong else (115, 255, 175)
    pct = 84 if is_wrong else 96
    kicker = "JOINT RISK" if is_wrong else "TARGET LOCK"
    cue = "INWARD KNEE" if is_wrong else "QUAD & VMO LOCK"
    splits = (
        [("KNEE · JOINT", 84, accent), ("QUADS · VMO", 16, v30.COL_MUTED)]
        if is_wrong
        else [("QUADS · VMO", 96, accent), ("KNEE · JOINT", 4, v30.COL_MUTED)]
    )
    rx_num = "16°" if is_wrong else "0°"
    rx_unit = "KNEE VALGUS" if is_wrong else "ALIGNED TRACK"

    bx, by = 52, 1210

    # 1. Giant Sculpted Percentage (`84%` / `96%`) + `≣` 5-Bar Icon + Bold Cue
    pct_str = f"{pct}%"
    pb = od.textbbox((0, 0), pct_str, font=v30.F_GIANT_NUM)
    pw = pb[2] - pb[0]
    sd.text((bx, by + 3), pct_str, font=v30.F_GIANT_NUM, fill=(0, 0, 0, 245))
    od.text((bx, by), pct_str, font=v30.F_GIANT_NUM, fill=(*accent, 255))

    tx = bx + pw + 18
    sd.text((tx, by + 12), kicker, font=v30.F_TAG_TINY, fill=(0, 0, 0, 240))
    od.text((tx, by + 10), kicker, font=v30.F_TAG_TINY, fill=(*accent_soft, 255))

    draw_day1_5bar_level_icon(od, tx, by + 35, (245, 248, 255), active_bars=4 if is_wrong else 5)
    sd.text((tx + 34, by + 36), cue, font=v30.F_CUE_SHORT, fill=(0, 0, 0, 245))
    od.text((tx + 34, by + 33), cue, font=v30.F_CUE_SHORT, fill=(255, 255, 255, 255))

    # 2. High-Tech 20-Segment LED Equalizer Split Comparison Bars
    gy = by + 102
    for m_lbl, m_val, m_col in splits:
        sd.text((bx, gy + 5), m_lbl, font=v30.F_TAG_TINY, fill=(0, 0, 0, 240))
        od.text((bx, gy + 3), m_lbl, font=v30.F_TAG_TINY, fill=v30.COL_SILVER)

        bar_x0 = bx + 116
        n_segs = 20
        seg_w, seg_gap = 8, 4
        active_segs = int(round((m_val / 100.0) * n_segs))
        for s_i in range(n_segs):
            sx_i = bar_x0 + s_i * (seg_w + seg_gap)
            col_s = (*m_col, 255) if s_i < active_segs else (52, 58, 70, 175)
            od.rounded_rectangle([sx_i, gy + 3, sx_i + seg_w, gy + 19], radius=2, fill=col_s)

        vx = bar_x0 + n_segs * (seg_w + seg_gap) + 10
        v_str = f"{m_val}%"
        sd.text((vx, gy - 2), v_str, font=v30.F_BAR_NUM, fill=(0, 0, 0, 245))
        od.text((vx, gy - 4), v_str, font=v30.F_BAR_NUM, fill=(255, 255, 255, 255) if m_col == accent else v30.COL_MUTED)
        gy += 38

    # 3. Bottom Outlined Angle Pill (`[ 16°   KNEE VALGUS ]`)
    ry = gy + 8
    od.rounded_rectangle([bx, ry, bx + 216, ry + 44], radius=22, fill=(14, 16, 22, 185), outline=(*accent_soft, 185), width=2)
    od.text((bx + 20, ry + 5), rx_num, font=v30.F_RX_NUM, fill=(255, 255, 255, 255))
    od.text((bx + 98, ry + 14), rx_unit, font=v30.F_TAG_TINY, fill=(*accent_soft, 255))

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)
    return np.array(canvas.convert("RGB"))


def draw_hud_variant_B_bottom_fixed_day1_frameless(frame_rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    """
    시안 B: `하단 고정 영역(y=1335..1470) Day 1 노박스 와이드 그래픽 비교 게이지` (`상단·하단 고정 + 중앙 운동동작 완전 분리형`)
    - 다리 주변 여백을 100% 비워 운동 동작이 가운데(`y=430..1310`)에 완벽히 Fit 되게 하고,
    - 하단 고정 영역(`y=1335..1470`)에 무거운 박스 없이 Day 1의 직관적인 `84% + 20단 듀얼 비교 바(KNEE JOINT 84% vs QUADS 16%) + [16° VALGUS]`를 수평으로 깔끔하게 배치!
    """
    canvas = Image.fromarray(frame_rgb).convert("RGBA")
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    accent = (255, 94, 36) if is_wrong else (32, 245, 122)
    accent_soft = (255, 155, 95) if is_wrong else (115, 255, 175)
    pct = 84 if is_wrong else 96
    kicker = "JOINT OVERLOAD" if is_wrong else "MUSCLE DRIVE"
    cue = "INWARD KNEE" if is_wrong else "QUAD & VMO LOCK"
    splits = (
        [("KNEE · JOINT", 84, accent), ("QUADS · VMO", 16, v30.COL_MUTED)]
        if is_wrong
        else [("QUADS · VMO", 96, accent), ("KNEE · JOINT", 4, v30.COL_MUTED)]
    )
    rx_num = "16°" if is_wrong else "0°"
    rx_unit = "KNEE VALGUS" if is_wrong else "NEUTRAL LOCK"

    # Left Block (`x = 76, y = 1340`): Giant `84%` + `≣` + `INWARD KNEE` + `[ 16° KNEE VALGUS ]`
    bx, by = 76, 1338
    pct_str = f"{pct}%"
    pb = od.textbbox((0, 0), pct_str, font=v30.F_GIANT_NUM)
    pw = pb[2] - pb[0]
    sd.text((bx, by + 3), pct_str, font=v30.F_GIANT_NUM, fill=(0, 0, 0, 245))
    od.text((bx, by), pct_str, font=v30.F_GIANT_NUM, fill=(*accent, 255))

    # Pill right below Giant `%`
    ry = by + 92
    od.rounded_rectangle([bx, ry, bx + 216, ry + 42], radius=21, fill=(14, 16, 22, 195), outline=(*accent_soft, 190), width=2)
    od.text((bx + 18, ry + 4), rx_num, font=v30.F_RX_NUM, fill=(255, 255, 255, 255))
    od.text((bx + 94, ry + 13), rx_unit, font=v30.F_TAG_TINY, fill=(*accent_soft, 255))

    # Right Block (`x = 340, y = 1344`): `≣` Cue Header + 2-Row 22-Segment LED Split Bars (`84% vs 16%`)
    rx0 = 336
    sd.text((rx0, by + 6), kicker, font=v30.F_TAG_TINY, fill=(0, 0, 0, 240))
    od.text((rx0, by + 4), kicker, font=v30.F_TAG_TINY, fill=(*accent_soft, 255))

    draw_day1_5bar_level_icon(od, rx0 + 135, by + 3, (245, 248, 255), active_bars=4 if is_wrong else 5)
    sd.text((rx0 + 168, by + 2), cue, font=v30.F_CUE_SHORT, fill=(0, 0, 0, 245))
    od.text((rx0 + 168, by), cue, font=v30.F_CUE_SHORT, fill=(255, 255, 255, 255))

    gy = by + 52
    for m_lbl, m_val, m_col in splits:
        sd.text((rx0, gy + 5), m_lbl, font=v30.F_TAG_TINY, fill=(0, 0, 0, 240))
        od.text((rx0, gy + 3), m_lbl, font=v30.F_TAG_TINY, fill=v30.COL_SILVER)

        bar_x0 = rx0 + 120
        n_segs = 24
        seg_w, seg_gap = 10, 4
        active_segs = int(round((m_val / 100.0) * n_segs))
        for s_i in range(n_segs):
            sx_i = bar_x0 + s_i * (seg_w + seg_gap)
            col_s = (*m_col, 255) if s_i < active_segs else (52, 58, 70, 175)
            od.rounded_rectangle([sx_i, gy + 2, sx_i + seg_w, gy + 20], radius=3, fill=col_s)

        vx = bar_x0 + n_segs * (seg_w + seg_gap) + 12
        v_str = f"{m_val}%"
        sd.text((vx, gy - 2), v_str, font=v30.F_BAR_NUM, fill=(0, 0, 0, 245))
        od.text((vx, gy - 4), v_str, font=v30.F_BAR_NUM, fill=(255, 255, 255, 255) if m_col == accent else v30.COL_MUTED)
        gy += 42

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)
    return np.array(canvas.convert("RGB"))


def draw_hud_variant_C_radial_dial_plus_day1_bars(frame_rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    """
    시안 C: `270° 원형 아크 다이얼 + Day 1 듀얼 비교 바 (84% vs 16%)` (`build_v30_pure_graphic_numbers.py` Mode B 업그레이드)
    - 좌측에 입체적인 270° 원형 아크 링 다이얼(`84% JOINT RISK` / `96% QUAD LOCK`),
    - 우측에 `KNEE · JOINT 84%` vs `QUADS · VMO 16%` 듀얼 LED 바와 `[ 16° KNEE VALGUS ]` 캡슐을 결합!
    """
    canvas = Image.fromarray(frame_rgb).convert("RGBA")
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    accent = (255, 94, 36) if is_wrong else (32, 245, 122)
    accent_soft = (255, 155, 95) if is_wrong else (115, 255, 175)
    pct = 84 if is_wrong else 96
    dial_tag = "JOINT RISK" if is_wrong else "QUAD LOCK"
    short_cue = "16° KNEE VALGUS" if is_wrong else "0° ALIGNED TRACK"
    splits = (
        [("KNEE · JOINT", 84, accent), ("QUADS · VMO", 16, v30.COL_MUTED)]
        if is_wrong
        else [("QUADS · VMO", 96, accent), ("KNEE · JOINT", 4, v30.COL_MUTED)]
    )

    bx, by = 84, 1326
    cx, cy = bx + 74, by + 76
    SS = 4
    d_sz = 160
    dial = Image.new("RGBA", (d_sz * SS, d_sz * SS), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)
    dc = (d_sz * SS) // 2
    r_arc = 60 * SS
    dd.ellipse([dc - 68 * SS, dc - 68 * SS, dc + 68 * SS, dc + 68 * SS], fill=(12, 14, 19, 185))
    start_a, span_a = 135, 270
    dd.arc([dc - r_arc, dc - r_arc, dc + r_arc, dc + r_arc], start=start_a, end=start_a + span_a, fill=(48, 55, 68, 210), width=9 * SS)
    active_end = start_a + int(span_a * (pct / 100.0))
    dd.arc([dc - r_arc, dc - r_arc, dc + r_arc, dc + r_arc], start=start_a, end=active_end, fill=(*accent, 255), width=10 * SS)
    ov.alpha_composite(dial.resize((d_sz, d_sz), Image.Resampling.LANCZOS), (cx - d_sz // 2, cy - d_sz // 2))

    num_s = f"{pct}%"
    nb = od.textbbox((0, 0), num_s, font=v30.F_ARC_NUM)
    od.text((cx - (nb[2] - nb[0]) // 2, cy - 26), num_s, font=v30.F_ARC_NUM, fill=(255, 255, 255, 255))
    tb = od.textbbox((0, 0), dial_tag, font=v30.F_TAG_TINY)
    od.text((cx - (tb[2] - tb[0]) // 2, cy + 24), dial_tag, font=v30.F_TAG_TINY, fill=(*accent_soft, 245))

    rx = bx + 178
    ry = by + 12
    draw_day1_5bar_level_icon(od, rx, ry + 3, accent, active_bars=4 if is_wrong else 5)
    sd.text((rx + 34, ry + 3), short_cue, font=v30.F_CUE_SHORT, fill=(0, 0, 0, 245))
    od.text((rx + 34, ry), short_cue, font=v30.F_CUE_SHORT, fill=(255, 255, 255, 255))

    gy = ry + 54
    for m_lbl, m_val, m_col in splits:
        od.text((rx, gy + 3), m_lbl, font=v30.F_TAG_TINY, fill=v30.COL_SILVER)
        bar_x0 = rx + 118
        n_segs = 20
        seg_w, seg_gap = 9, 4
        active_segs = int(round((m_val / 100.0) * n_segs))
        for s_i in range(n_segs):
            sx_i = bar_x0 + s_i * (seg_w + seg_gap)
            col_s = (*m_col, 255) if s_i < active_segs else (48, 54, 66, 180)
            od.rounded_rectangle([sx_i, gy + 2, sx_i + seg_w, gy + 18], radius=2, fill=col_s)
        od.text((bar_x0 + n_segs * (seg_w + seg_gap) + 10, gy - 4), f"{m_val}%", font=v30.F_BAR_NUM, fill=(255, 255, 255, 255) if m_col == accent else v30.COL_MUTED)
        gy += 38

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)
    return np.array(canvas.convert("RGB"))


def draw_hud_variant_D_side_vertical_columns_and_leader(frame_rgb: np.ndarray, kp: dict, is_wrong: bool) -> np.ndarray:
    """
    시안 D: `좌·우 여백 그래픽 타워 분할형` (`t328u`: "옆에 그래픽적으로 풀어가는것도 여러개 시안")
    - 마네킹 옆의 빈 스튜디오 여백을 활용해:
      * 좌측 여백(`x = 54 .. 285, y = 860 .. 1250`): 관절(`◎`)에서 뻗어나온 얇은 리더선 + 수직 20단 듀얼 비교 타워(`JOINT 84%` vs `QUADS 16%`)
      * 우측 여백(`x = 760 .. 1010`): 곡선 각도 아크 + `16° KNEE CAVE`
    """
    canvas = Image.fromarray(frame_rgb).convert("RGBA")
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    accent = (255, 94, 36) if is_wrong else (32, 245, 122)
    accent_soft = (255, 155, 95) if is_wrong else (115, 255, 175)
    pct = 84 if is_wrong else 96

    # Connect left knee (`kn_r` at `x ~ 370, y ~ 985`) to the Left Margin Graphic HUD (`x = 52 .. 310`) via a fine leader line + reticle (`◎`)
    kx, ky = int(kp["kn_r"][0]), int(kp["kn_r"][1])
    od.ellipse([kx - 10, ky - 10, kx + 10, ky + 10], outline=(*accent, 235), width=2)
    od.ellipse([kx - 3, ky - 3, kx + 3, ky + 3], fill=(255, 255, 255, 255))
    od.line([kx - 12, ky, 260, ky, 235, ky + 35], fill=(*accent_soft, 185), width=2)

    bx, by = 48, ky + 42
    pct_str = f"{pct}%"
    sd.text((bx, by + 3), pct_str, font=v30.F_GIANT_NUM, fill=(0, 0, 0, 245))
    od.text((bx, by), pct_str, font=v30.F_GIANT_NUM, fill=(*accent, 255))

    kicker = "JOINT STRESS" if is_wrong else "QUAD DRIVE"
    od.text((bx + 4, by + 86), kicker, font=v30.F_TAG_TINY, fill=(*accent_soft, 255))

    splits = (
        [("JOINT", 84, accent), ("QUADS", 16, v30.COL_MUTED)]
        if is_wrong
        else [("QUADS", 96, accent), ("JOINT", 4, v30.COL_MUTED)]
    )
    gy = by + 114
    for m_lbl, m_val, m_col in splits:
        od.text((bx + 2, gy + 2), m_lbl, font=v30.F_TAG_TINY, fill=v30.COL_SILVER)
        bar_x0 = bx + 68
        n_segs = 14
        seg_w, seg_gap = 8, 3
        active_segs = int(round((m_val / 100.0) * n_segs))
        for s_i in range(n_segs):
            sx_i = bar_x0 + s_i * (seg_w + seg_gap)
            col_s = (*m_col, 255) if s_i < active_segs else (48, 54, 66, 180)
            od.rounded_rectangle([sx_i, gy + 2, sx_i + seg_w, gy + 16], radius=2, fill=col_s)
        od.text((bar_x0 + n_segs * (seg_w + seg_gap) + 6, gy - 4), f"{m_val}%", font=v30.F_BAR_NUM, fill=(255, 255, 255, 255) if m_col == accent else v30.COL_MUTED)
        gy += 34

    ry = gy + 8
    rx_num = "16°" if is_wrong else "0°"
    rx_unit = "VALGUS" if is_wrong else "LOCKED"
    od.rounded_rectangle([bx, ry, bx + 185, ry + 40], radius=20, fill=(14, 16, 22, 195), outline=(*accent_soft, 185), width=2)
    od.text((bx + 18, ry + 3), rx_num, font=v30.F_RX_NUM, fill=(255, 255, 255, 255))
    od.text((bx + 92, ry + 12), rx_unit, font=v30.F_TAG_TINY, fill=(*accent_soft, 255))

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)
    return np.array(canvas.convert("RGB"))
import build_fast_hud_concepts as bf


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_peaks = {}
    for s_i in (1, 2, 3, 4):
        fn, i0, i1, is_w, *_ = p62.STEP_POOL_SPECS[s_i]
        seq = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=52)
        raw_peaks[s_i] = seq[-1]

    base_frames = {}
    masks = {}
    kps = {}
    thumbs = {}
    for s_i in (1, 2, 3, 4):
        is_w = (s_i in (1, 3))
        b_rgb, b_mask, b_kp = render_day1_pearl_mannequin_and_neon_xray_1080p(raw_peaks[s_i], s_i, is_w)
        base_frames[s_i] = b_rgb
        masks[s_i] = b_mask
        kps[s_i] = b_kp
        thumbs[s_i] = cv2.resize(b_rgb[520:1325, 40:1040], (360, 260), interpolation=cv2.INTER_AREA)

    # Check exact mannequin skin tone on chest/shoulder of Step 1 & Step 2 compared to Day 1 (`[138, 141, 144]`)
    for s_i in (1, 2):
        roi_rgb = base_frames[s_i][700:880, 440:640]
        roi_m = masks[s_i][700:880, 440:640] > 0.7
        vals = roi_rgb[roi_m]
        med = np.median(vals, axis=0)
        p90 = np.percentile(vals, 90, axis=0)
        print(f"Step {s_i} Mannequin Torso Skin: median RGB={np.round(med, 1)}, p90 RGB={np.round(p90, 1)}")

    variants = {"A": {}, "B": {}, "C": {}, "D": {}}
    for s_i in (1, 2):
        is_w = (s_i == 1)
        # Variant A: Exact Day 1 style (3D Neon X-Ray Bone Flow + exterior curved angle callout + Left-Lower Frameless 20-seg dual bar)
        fr_a = draw_day1_bone_energy_flow_and_exterior_arcs(
            base_frames[s_i], masks[s_i], kps[s_i], is_w, with_exterior_callouts=True, with_laser_axis=False
        )
        fr_a = bf.draw_clean_top_carousel(fr_a, s_i, is_w, thumbs, ring_pct=None)
        variants["A"][s_i] = draw_hud_variant_A_day1_exact_left_frameless(fr_a, is_w)

        # Variant B: Bottom-Fixed Day 1 Frameless Wide Dual Bars + @demicstory Laser Axis + Neon X-Ray Bone Flow
        fr_b = draw_day1_bone_energy_flow_and_exterior_arcs(
            base_frames[s_i], masks[s_i], kps[s_i], is_w, with_exterior_callouts=False, with_laser_axis=True
        )
        fr_b = bf.draw_clean_top_carousel(fr_b, s_i, is_w, thumbs, ring_pct=None)
        variants["B"][s_i] = draw_hud_variant_B_bottom_fixed_day1_frameless(fr_b, is_w)

        # Variant C: 270° Circular Arc Ring Dial + Day 1 20-Segment Dual Split Bars (Bottom-Fixed Frameless) + Exterior Callout
        fr_c = draw_day1_bone_energy_flow_and_exterior_arcs(
            base_frames[s_i], masks[s_i], kps[s_i], is_w, with_exterior_callouts=True, with_laser_axis=True
        )
        fr_c = bf.draw_clean_top_carousel(fr_c, s_i, is_w, thumbs, ring_pct=None)
        variants["C"][s_i] = draw_hud_variant_C_radial_dial_plus_day1_bars(fr_c, is_w)

        # Variant D: Side Margin Graphic Split (`옆에 그래픽적으로 풀어가는 시안`: Left Leader + Dual Bars | Right Curved Arc + Angle)
        fr_d = draw_day1_bone_energy_flow_and_exterior_arcs(
            base_frames[s_i], masks[s_i], kps[s_i], is_w, with_exterior_callouts=True, with_laser_axis=True
        )
        fr_d = bf.draw_clean_top_carousel(fr_d, s_i, is_w, thumbs, ring_pct=None)
        variants["D"][s_i] = draw_hud_variant_D_side_vertical_columns_and_leader(fr_d, kps[s_i], is_w)

    # Save Master Comparison Sheet (4 Columns: A, B, C, D x 2 Rows: Row 1 Wrong X-Ray Orange-Gold Bone, Row 2 Right Emerald Bone+Muscle)
    cw, ch = 520, 924
    pad = 22
    hdr_h = 116
    sheet = Image.new("RGB", (cw * 4 + pad * 5, ch * 2 + hdr_h + pad * 3), (10, 12, 16))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(27, "extrabold")
    f_c = v30.f_pretendard(19, "extrabold")
    f_s = v30.f_pretendard(14, "bold")

    sd.text(
        (pad, 18),
        "DAY 1 (조회수 5,931회) 완벽 이식 — 3D 네온 X-Ray 뼈 에너지 흐름 + 밝은 실버 마네킹 + 직관적 듀얼 비교 바(84% vs 16%) 4종",
        font=f_t,
        fill=(255, 255, 255),
    )
    meta_cols = [
        ("시안 A · Day 1 오리지널 100% 동일 레이아웃", "좌하단 노박스 84% + [KNEE·JOINT 84% vs QUADS 16%] 20단 바 + 우측 곡선 각도"),
        ("시안 B · 하단 고정 와이드 노박스 듀얼 비교 바", "중앙 운동영역 100% 분리 + 하단에 [84% | KNEE·JOINT 84% vs QUADS 16%] 수평 배치"),
        ("시안 C · 270° 원형 링 다이얼 + 듀얼 비교 바", "좌측 270° 원형 링 게이지(84%/96%) + 우측 20단 듀얼 비교 바 & 각도 결합"),
        ("시안 D · 좌·우 여백 그래픽 타워 연결형", "무릎 관절(◎) 리더선으로 좌측 여백에 84% 비교 바, 우측 여백에 16° 아크 배치"),
    ]

    for idx, k in enumerate(("A", "B", "C", "D")):
        x0 = pad + idx * (cw + pad)
        sd.text((x0 + 4, 60), meta_cols[idx][0], font=f_c, fill=(255, 165, 70) if idx == 0 else (45, 250, 140))
        sd.text((x0 + 4, 86), meta_cols[idx][1], font=f_s, fill=(195, 205, 220))

        # Row 1: Step 1 Wrong (Orange-Gold Neon X-Ray Bone Flow, like Day 1 screenshot!)
        im_w = Image.fromarray(variants[k][1]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im_w, (x0, hdr_h))
        sd.rounded_rectangle([x0 - 2, hdr_h - 2, x0 + cw + 2, hdr_h + ch + 2], radius=14, outline=(255, 94, 36), width=3)

        # Row 2: Step 2 Right (Emerald-Lime Neon Bone & Quad Activation Flow)
        y2 = hdr_h + ch + pad
        im_r = Image.fromarray(variants[k][2]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im_r, (x0, y2))
        sd.rounded_rectangle([x0 - 2, y2 - 2, x0 + cw + 2, y2 + ch + 2], radius=14, outline=(35, 245, 125), width=3)

    out_master = os.path.join(OUT_DIR, "day3_day1_xray_and_intuitive_hud_4variants.jpg")
    sheet.save(out_master, quality=95)

    # Save 1:1 Close-Up of the Glowing Neon X-Ray Bone + Intuitive Dual Comparison Bars (`84% vs 16%`)
    closeup = Image.new("RGB", (980 * 2 + 60, 740 * 2 + 120), (10, 12, 16))
    cd = ImageDraw.Draw(closeup)
    cd.text(
        (20, 18),
        "1:1 원본 픽셀 확대 — 3D 네온 X-Ray 뼈 에너지 흐름 + 직관적 부하 비교 바 (`KNEE·JOINT 84% vs QUADS 16%`) 4종",
        font=f_t,
        fill=(255, 255, 255),
    )
    pos = [(20, 72), (1020, 72), (20, 842), (1020, 842)]
    for idx, k in enumerate(("A", "B", "C", "D")):
        px, py = pos[idx]
        # Crop `y = 780 .. 1500, x = 30 .. 1010` showing both the glowing X-Ray knee bones and the graphic HUD!
        crop_im = Image.fromarray(variants[k][1][780:1500, 30:1010])
        closeup.paste(crop_im, (px, py + 32))
        cd.text((px + 6, py), meta_cols[idx][0], font=f_c, fill=(255, 165, 70) if idx == 0 else (45, 250, 140))
        cd.rounded_rectangle([px - 2, py + 30, px + 982, py + 752], radius=12, outline=(255, 255, 255, 75), width=2)

    out_closeup = os.path.join(OUT_DIR, "day3_day1_xray_and_hud_1to1_closeup.jpg")
    closeup.save(out_closeup, quality=95)
    print("Saved:", out_master, "and", out_closeup)


if __name__ == "__main__":
    main()
