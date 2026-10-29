#!/usr/bin/env python3
"""
Day 5 (`Dumbbell Lateral Raise`) — 100% Native 3D Multi-Angle 4-Step Master Reel Builder
Applies confirmed V3 + Option 3 + Bright Silver-Grey Studio Mannequin Standard:
1. Bright Silver-Grey Studio Mannequin (`p75 ~ 165, p90 ~ 191, max 226`) + Fixed Charcoal Studio Background (`RGB 38,40,44`).
2. 100% Opaque Solid Matte Jet-Black Drawers (`lum 10..22`).
3. Top 4-slot preview bar in 3:4 profile grid safe zone (y = 248..406) with Translucent Glass See-Through.
4. 'Dumbbell Lateral Raise' Kaushan Script title on the studio back wall behind the 3D mannequin (y = 452).
5. 4-Step Bone X-Ray vs 3D Muscle Fiber Striations:
   - Step 1 (3/4 Front-Side Wrong Bone X-Ray): 90° ARM FLARE / 84% JOINT PINCH (Clavicle & Shoulder Joint Bone X-Ray)
   - Step 2 (3/4 Rear-Side Wrong Muscle Overload): 28° TRAP SHRUG / 76% TRAP OVERLOAD (Upper Trapezius 3D Muscle Striations)
   - Step 3 (3/4 Front-Side Correct Bone X-Ray): 0° SCAPULA LOCK / 96% JOINT SAFE (Depressed Clavicle & Scapula Bone X-Ray)
   - Step 4 (3/4 Front-Side Correct Target Muscle): 30° SCAPULAR PLANE / 98% LATERAL DELT (Lateral Deltoid 3D Muscle Striations)
"""
import os
import sys
import gc
import math
import wave
import pickle
import asyncio
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import universal_anatomy_rig as uar
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62
import build_clean_d4_variants as bcd
import build_4step_bone_muscle_master as b4

OUT_DIR = "/workspace/daily_info_reels/output"
ASSETS_DIR = "/workspace/daily_info_reels/output/assets"
TMP_DIR = "/workspace/daily_info_reels/_tmp_day5_master"
CACHE_KP_PKL = "/workspace/daily_info_reels/_cache_day5_lat_720p_kps.pkl"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)  # 600 frames

STEP_SPECS_DAY5 = {
    1: ("day5_lat_step1_bone_720p.mp4",   2, 71, True,  "bone_blood",   80),
    2: ("day5_lat_step2_muscle_720p.mp4", 2,  55, True,  "muscle_blood", 80),
    3: ("day5_lat_step3_bone_720p.mp4",   2, 106, False, "bone_blood",   80),
    4: ("day5_lat_step4_muscle_720p.mp4", 2,  64, False, "muscle_blood", 80),
}

b4.STEP_DY = {1: 80, 2: 85, 3: 80, 4: 80}


def load_native_slice_720p(path: str, i0: int, i1: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frs = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if fr.shape[1] != 720 or fr.shape[0] != 1280:
            fr = cv2.resize(fr, (720, 1280), interpolation=cv2.INTER_LANCZOS4)
        frs.append(fr)
    cap.release()
    i1 = min(i1, len(frs) - 1)
    sl = frs[i0 : i1 + 1] if i0 <= i1 else frs[i1 : i0 + 1][::-1]
    if len(sl) > 36:
        idxs = [int(round(x)) for x in np.linspace(0, len(sl) - 1, 36)]
        sl = [sl[i] for i in idxs]
    return sl


def extract_day5_kps_1080(seq_bgr_720: list[np.ndarray], step_idx: int, y_sh: int) -> list[dict]:
    detector = uar.get_pose_detector()
    raw_kps = []
    for bgr in seq_bgr_720:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb))
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            sh1 = (lm[11].x * 720.0, lm[11].y * 1280.0)
            sh2 = (lm[12].x * 720.0, lm[12].y * 1280.0)
            el1 = (lm[13].x * 720.0, lm[13].y * 1280.0)
            el2 = (lm[14].x * 720.0, lm[14].y * 1280.0)
            wr1 = (lm[15].x * 720.0, lm[15].y * 1280.0)
            wr2 = (lm[16].x * 720.0, lm[16].y * 1280.0)
            hip1 = (lm[23].x * 720.0, lm[23].y * 1280.0)
            hip2 = (lm[24].x * 720.0, lm[24].y * 1280.0)
            kn1 = (lm[25].x * 720.0, lm[25].y * 1280.0)
            kn2 = (lm[26].x * 720.0, lm[26].y * 1280.0)
            ank1 = (lm[27].x * 720.0, lm[27].y * 1280.0)
            ank2 = (lm[28].x * 720.0, lm[28].y * 1280.0)

            # Screen-left is _r, screen-right is _l for consistent camera-space tracking
            sh_r, sh_l = (sh1, sh2) if sh1[0] < sh2[0] else (sh2, sh1)
            elb_r, elb_l = (el1, el2) if el1[0] < el2[0] else (el2, el1)
            wr_r, wr_l = (wr1, wr2) if wr1[0] < wr2[0] else (wr2, wr1)
            hip_r, hip_l = (hip1, hip2) if hip1[0] < hip2[0] else (hip2, hip1)
            kn_r, kn_l = (kn1, kn2) if kn1[0] < kn2[0] else (kn2, kn1)
            ank_r, ank_l = (ank1, ank2) if ank1[0] < ank2[0] else (ank2, ank1)
            sh = (0.5 * (sh_r[0] + sh_l[0]), 0.5 * (sh_r[1] + sh_l[1]))
        else:
            prev = raw_kps[-1]
            sh, sh_r, sh_l = prev["sh"], prev["sh_r"], prev["sh_l"]
            elb_r, elb_l, wr_r, wr_l = prev["elb_r"], prev["elb_l"], prev["wr_r"], prev["wr_l"]
            hip_r, hip_l, kn_r, kn_l, ank_r, ank_l = prev["hip_r"], prev["hip_l"], prev["kn_r"], prev["kn_l"], prev["ank_r"], prev["ank_l"]

        raw_kps.append({
            "sh": sh, "sh_r": sh_r, "sh_l": sh_l,
            "elb_r": elb_r, "elb_l": elb_l,
            "wr_r": wr_r, "wr_l": wr_l,
            "hip_r": hip_r, "hip_l": hip_l,
            "kn_r": kn_r, "kn_l": kn_l,
            "ank_r": ank_r, "ank_l": ank_l,
        })

    keys = list(raw_kps[0].keys())
    n = len(raw_kps)
    out_1080 = [dict() for _ in range(n)]
    for k in keys:
        xs = gaussian_filter1d([p[k][0] * 1.5 for p in raw_kps], sigma=2.2, mode="nearest")
        ys = gaussian_filter1d([p[k][1] * 1.5 + y_sh for p in raw_kps], sigma=2.2, mode="nearest")
        for i in range(n):
            out_1080[i][k] = (float(xs[i]), float(ys[i]))
    return out_1080


def shade_day5_3d_frame_1080p(
    rgb_1080: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    kp: dict,
    pulse_phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r_ch = rgb_1080[:, :, 0].astype(np.float32)
    g_ch = rgb_1080[:, :, 1].astype(np.float32)
    b_ch = rgb_1080[:, :, 2].astype(np.float32)
    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    min_c = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    sat = (max_c - min_c) / (max_c + 1.0)

    red_dom = np.clip(r_ch - 0.50 * (g_ch + b_ch), 0.0, 255.0)
    grn_dom = np.clip(g_ch - 0.50 * (r_ch + b_ch), 0.0, 255.0)
    act_dom = np.maximum(red_dom, grn_dom)

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    hip_y = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1])
    kn_y = 0.5 * (kp["kn_r"][1] + kp["kn_l"][1])
    ank_y = 0.5 * (kp["ank_r"][1] + kp["ank_l"][1])

    # Row-adaptive background luminance so any subtle studio floor platform is cleanly merged into fixed RGB(38,40,44)
    edge_strips = np.concatenate([lum[:, :64], lum[:, -64:]], axis=1)
    row_bg = cv2.GaussianBlur(np.median(edge_strips, axis=1).astype(np.float32)[:, None], (0, 0), 18.0)
    floor_y_gate = np.clip(((ank_y + 52.0) - yy) / 24.0, 0.0, 1.0)
    fg_diff = np.abs(lum - row_bg) + 0.95 * act_dom
    w_fg = cv2.GaussianBlur((np.clip((fg_diff - 8.0) / 14.0, 0.0, 1.0) * floor_y_gate).astype(np.float32), (0, 0), 2.0)

    # 1. Opaque Matte-Black Drawers Mask (lum 10..22)
    shorts_band = np.clip((yy - (hip_y - 140.0)) / 24.0, 0.0, 1.0) * np.clip(((kn_y - 32.0) - yy) / 34.0, 0.0, 1.0)
    dark_low_sat = np.clip((68.0 - lum) / 18.0, 0.0, 1.0) * np.clip((0.28 - sat) / 0.14, 0.0, 1.0) * np.clip((18.0 - act_dom) / 12.0, 0.0, 1.0)
    shorts_mask_1080 = cv2.GaussianBlur((shorts_band * dark_low_sat * (w_fg > 0.25)).astype(np.float32), (0, 0), 2.0)
    shorts_mask_1080 = np.clip(shorts_mask_1080 * 1.35, 0.0, 1.0)

    # 2. Bright Silver-Grey Studio Mannequin Curve (calibrated so p75 ~ 165, p90 ~ 191, max 226)
    skin_m = (w_fg > 0.6) & (shorts_mask_1080 < 0.2) & (act_dom < 15.0) & (lum > 42.0)
    p90_raw = float(np.percentile(lum[skin_m], 90)) if skin_m.sum() > 500 else 180.0
    hv_silver = np.clip((lum - 24.0) / max(p90_raw - 24.0, 95.0), 0.0, 1.28)
    blur_med = cv2.GaussianBlur(lum, (0, 0), 2.8)
    sulc = np.clip((blur_med - lum) / 14.0, 0.0, 1.0)
    hf_ridge = lum - cv2.GaussianBlur(lum, (0, 0), 1.6)
    grey_back_exact = np.clip(48.0 + 143.0 * (hv_silver ** 0.88) + hf_ridge * 0.70 - sulc * 11.0, 44.0, 226.0)

    shorts_jet_lum = np.clip(lum * 0.30 + 8.0, 10.0, 22.0)
    matte_lum = grey_back_exact * (1.0 - shorts_mask_1080) + shorts_jet_lum * shorts_mask_1080

    mono_r = 38.0 * (1.0 - w_fg) + (matte_lum - 1.5) * w_fg
    mono_g = 40.0 * (1.0 - w_fg) + (matte_lum + 0.5) * w_fg
    mono_b = 44.0 * (1.0 - w_fg) + (matte_lum + 3.0) * w_fg
    mono = np.stack([mono_r, mono_g, mono_b], axis=-1)

    # 3. Step-Specific Anatomical Gate (Bone X-Ray in Steps 1 & 3 vs 3D Muscle Fiber Striations in Steps 2 & 4)
    sh_r, sh_l = kp["sh_r"], kp["sh_l"]
    elb_r, elb_l = kp["elb_r"], kp["elb_l"]
    sh_y = 0.5 * (sh_r[1] + sh_l[1])

    if step_idx in (1, 3):
        # Gate strictly to Clavicle + Scapula + Glenohumeral Shoulder Joint (stop forearm/wrist bleed)
        d_shr = np.hypot((xx - sh_r[0]) / 1.15, (yy - sh_r[1]) / 0.95)
        d_shl = np.hypot((xx - sh_l[0]) / 1.15, (yy - sh_l[1]) / 0.95)
        cx_clav = 0.5 * (sh_r[0] + sh_l[0])
        cy_clav = sh_y + 12.0
        rx_clav = max(90.0, 0.55 * abs(sh_l[0] - sh_r[0]) + 42.0)
        d_clav = np.hypot((xx - cx_clav) / rx_clav, (yy - cy_clav) / 72.0)
        joint_rad_gate = np.maximum(
            np.clip((135.0 - np.minimum(d_shr, d_shl)) / 42.0, 0.0, 1.0),
            np.clip((1.18 - d_clav) / 0.35, 0.0, 1.0),
        )
        y_band_gate = np.clip((yy - (sh_y - 95.0)) / 28.0, 0.0, 1.0) * np.clip(((sh_y + 135.0) - yy) / 36.0, 0.0, 1.0)
        anat_gate = joint_rad_gate * y_band_gate
    elif step_idx == 2:
        # Upper Trapezius gate (neck base & top of shoulders above mid-back)
        anat_gate = np.clip((yy - (sh_y - 120.0)) / 28.0, 0.0, 1.0) * np.clip(((sh_y + 135.0) - yy) / 40.0, 0.0, 1.0)
    else:
        # Step 4: Lateral Deltoid cap gate (shoulder deltoid caps on both sides)
        d_deltr = np.hypot((xx - (sh_r[0] - 10.0)) / 92.0, (yy - (sh_r[1] + 22.0)) / 108.0)
        d_deltl = np.hypot((xx - (sh_l[0] + 10.0)) / 92.0, (yy - (sh_l[1] + 22.0)) / 108.0)
        anat_gate = np.clip((1.35 - np.minimum(d_deltr, d_deltl)) / 0.40, 0.0, 1.0)

    w_act = cv2.GaussianBlur((np.clip((act_dom - 10.0) / 32.0, 0.0, 1.0) ** 0.85).astype(np.float32), (0, 0), 1.8) * (1.0 - shorts_mask_1080) * anat_gate
    core_seed = (np.clip((act_dom - 16.0) / 42.0, 0.0, 1.0) ** 1.08) * anat_gate
    pulse = 1.0 + 0.06 * math.sin(2.0 * math.pi * pulse_phase)
    core_w = np.clip(cv2.GaussianBlur(core_seed.astype(np.float32), (0, 0), 6.0) * w_act * 0.90 * pulse, 0.0, 0.90)

    # Extract 3D muscle fiber ridges & inter-fiber sulci
    act_ch = r_ch if (is_wrong or step_idx == 4) else g_ch
    gb_ch = 0.55 * g_ch + 0.45 * b_ch
    a_blur_fine = cv2.GaussianBlur(act_ch, (0, 0), 1.3)
    a_blur_mid  = cv2.GaussianBlur(act_ch, (0, 0), 4.8)
    a_blur_wide = cv2.GaussianBlur(act_ch, (0, 0), 13.0)
    gb_blur_mid = cv2.GaussianBlur(gb_ch, (0, 0), 4.2)

    fiber_fine   = (act_ch - a_blur_mid) * 2.05 + (gb_ch - gb_blur_mid) * 1.45
    fiber_bundle = (a_blur_fine - a_blur_wide) * 1.05
    sulc_3d  = np.clip(-fiber_fine / 13.0, 0.0, 1.0)
    ridge_3d = np.clip(fiber_fine / 15.0, 0.0, 1.0)
    vol_3d   = np.clip((0.58 * act_ch + 0.42 * gb_ch) / 170.0, 0.18, 1.20)
    fiber_mod = np.clip(vol_3d + 0.54 * ridge_3d - 0.58 * sulc_3d + 0.004 * fiber_bundle, 0.12, 1.42)

    # Crisp 3D skeletal bone structure for Steps 1 & 3 (preserves clavicle, scapula & humeral head bone edges)
    bone_edge = np.clip(lum - cv2.GaussianBlur(lum, (0, 0), 5.0), -24.0, 38.0)
    bone_lum = np.clip(cv2.GaussianBlur(lum, (0, 0), 1.1) + 0.85 * bone_edge, 0.0, 255.0)
    L_bone = np.clip((bone_lum - 24.0) / 140.0, 0.0, 1.0)
    bone_core = (np.clip((bone_lum - 115.0) / 55.0, 0.0, 1.0) * np.clip((act_dom - 12.0) / 26.0, 0.0, 1.0) * anat_gate) if step_idx in (1, 3) else 0.0

    if is_wrong:
        if step_idx == 1:
            # Step 1: Pure Crimson-Red Clavicle & Shoulder Joint Bone X-Ray
            col_r = np.clip(bone_lum * 1.45 + 38.0 * (L_bone ** 0.85) + 58.0 * bone_core, 22.0, 242.0)
            col_g = np.clip(bone_lum * 0.32 + 64.0 * bone_core, 10.0, 155.0)
            col_b = np.clip(bone_lum * 0.34 + 52.0 * bone_core, 12.0, 142.0)
        else:
            # Step 2: Vivid Crimson-Red Upper Trapezius 3D Muscle Fiber Striations
            col_r = np.clip(36.0 + 174.0 * fiber_mod + 52.0 * (ridge_3d ** 1.25) - 46.0 * sulc_3d, 22.0, 245.0)
            col_g = np.clip(14.0 + 40.0 * fiber_mod + 28.0 * (ridge_3d ** 1.5) - 18.0 * sulc_3d, 10.0, 102.0)
            col_b = np.clip(16.0 + 42.0 * fiber_mod + 26.0 * (ridge_3d ** 1.5) - 18.0 * sulc_3d, 10.0, 102.0)
    else:
        if step_idx == 3:
            # Step 3: Pure Emerald-Green Clavicle & Scapula Joint Lock Bone X-Ray
            col_g = np.clip(bone_lum * 1.45 + 36.0 * (L_bone ** 0.85) + 58.0 * bone_core, 22.0, 242.0)
            col_r = np.clip(bone_lum * 0.32 + 48.0 * bone_core, 10.0, 138.0)
            col_b = np.clip(bone_lum * 0.76 + 56.0 * bone_core, 14.0, 185.0)
        else:
            # Step 4: Vivid Emerald-Green Lateral Deltoid 3D Muscle Fiber Striations (근육결)
            col_r = np.clip(14.0 + 36.0 * fiber_mod + 30.0 * (ridge_3d ** 1.5) - 14.0 * sulc_3d, 10.0, 94.0)
            col_g = np.clip(34.0 + 180.0 * fiber_mod + 58.0 * (ridge_3d ** 1.25) - 48.0 * sulc_3d, 24.0, 248.0)
            col_b = np.clip(22.0 + 96.0 * fiber_mod + 42.0 * (ridge_3d ** 1.35) - 28.0 * sulc_3d, 16.0, 170.0)

    act_rgb = np.stack([col_r, col_g, col_b], axis=-1)
    blend_w = np.clip(core_w + 0.45 * bone_core, 0.0, 0.92) * (1.0 - shorts_mask_1080)
    out = mono * (1.0 - blend_w[:, :, None]) + act_rgb * blend_w[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8), shorts_mask_1080


def draw_day5_laser_and_callout(
    base_rgb: np.ndarray,
    shorts_mask_1080: np.ndarray,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    deg_str: str,
    l1_str: str,
    l2_str: str,
) -> np.ndarray:
    canvas = Image.fromarray(base_rgb).convert("RGBA")
    sh_r, sh_l = kp["sh_r"], kp["sh_l"]
    elb_r, elb_l = kp["elb_r"], kp["elb_l"]
    wr_r, wr_l = kp["wr_r"], kp["wr_l"]

    # Extend laser ends slightly along the clavicle and arm vectors (@demicstory style)
    _, wr_r_ext = p62._extend_segment(elb_r, wr_r, 0.0, 1.12)
    _, wr_l_ext = p62._extend_segment(elb_l, wr_l, 0.0, 1.12)
    sh_r_ext, sh_l_ext = p62._extend_segment(sh_r, sh_l, -0.08, 1.08)

    clav_mid = (0.52 * sh_r[0] + 0.48 * sh_l[0], 0.52 * sh_r[1] + 0.48 * sh_l[1] + 4.0)
    if step_idx == 2:
        p62.draw_demic_laser_joint_axis(
            canvas,
            main_chains_1x=[[sh_r, elb_r, wr_r_ext], [sh_l, elb_l, wr_l_ext]],
            aux_ext_chains_1x=[[sh_r_ext, sh_l_ext]],
            vertex_pts_1x=[sh_r, elb_r, sh_l, elb_l],
            is_wrong=is_wrong,
        )
    else:
        p62.draw_demic_laser_joint_axis(
            canvas,
            main_chains_1x=[[sh_r, elb_r, wr_r_ext]],
            aux_ext_chains_1x=[[sh_r_ext, clav_mid]],
            vertex_pts_1x=[sh_r, elb_r],
            is_wrong=is_wrong,
        )

    if step_idx == 2:
        # Point directly to the overloaded upper trapezius ridge on screen-left shoulder
        target_pt = (int(0.65 * sh_r[0] + 0.35 * sh_l[0]), int(sh_r[1] - 22))
    elif step_idx in (1, 3):
        # Point directly to the screen-left acromion / glenohumeral shoulder joint
        target_pt = (int(sh_r[0] + 8), int(sh_r[1] + 6))
    else:
        # Point directly to the screen-left lateral deltoid muscle belly
        target_pt = (int(0.76 * sh_r[0] + 0.24 * elb_r[0]), int(0.76 * sh_r[1] + 0.24 * elb_r[1] + 10))

    laser_rgb = np.array(canvas.convert("RGB")).astype(np.float32)
    sh_gate = np.clip(shorts_mask_1080 * 0.90, 0.0, 0.90)[:, :, None]
    blended = np.clip(laser_rgb * (1.0 - sh_gate) + base_rgb.astype(np.float32) * sh_gate, 0, 255).astype(np.uint8)

    base_pil = Image.fromarray(blended).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    # Place callout cleanly in the upper-left studio space above the raised arm path (y ~ 525..640)
    elbow_x = max(254, min(target_pt[0] - 28, 340))
    elbow_y = int(np.clip(target_pt[1] - 92, 548, 665))
    bcd._draw_precision_leader(rd, target_pt, (elbow_x, elbow_y), (242, elbow_y), col)

    tx = 54
    ty = int(np.clip(elbow_y - 76, 485, 620))
    f_big = v30.f_pretendard(84, "extrabold")
    f_l1 = v30.f_pretendard(21, "extrabold")
    f_l2 = v30.f_pretendard(16, "bold")

    rd.text((tx + 2, ty + 3), deg_str, font=f_big, fill=(0, 0, 0, 210))
    rd.text((tx, ty), deg_str, font=f_big, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 90), l1_str, font=f_l1, fill=(col[0], col[1], col[2], 255))
    rd.text((tx + 2, ty + 117), l2_str, font=f_l2, fill=(215, 222, 232, 235))

    return np.array(Image.alpha_composite(base_pil, ov).convert("RGB"))


async def synthesize_day5_voiceover_wav(
    wav_path: str,
    vo_lines: list[tuple[float, str]],
    duration_s: float = 20.0,
):
    import edge_tts

    sr = 44100
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples, dtype=np.float32) / sr

    beat_env = 0.45 + 0.55 * (np.sin(2.0 * math.pi * (100.0 / 60.0) * t) > 0.72).astype(np.float32)
    sub = 0.040 * np.sin(2.0 * math.pi * 55.0 * t) * beat_env
    master = sub.copy()

    for idx_s, t_step in enumerate([0.05, 5.0, 10.0, 15.0]):
        s0 = int(t_step * sr)
        dur_c = int(0.24 * sr)
        if s0 + dur_c < n_samples:
            tc = np.linspace(0.0, 0.24, dur_c, endpoint=False)
            freq = 220.0 if idx_s < 2 else 528.0
            chime = 0.085 * np.sin(2.0 * math.pi * freq * tc) * np.exp(-tc * 13.0)
            master[s0 : s0 + dur_c] += chime

    prev_end_s = 0.0
    print(f"\n=== DAY 5 VOICEOVER TIMING QC ({os.path.basename(wav_path)}) ===")
    for i, (target_start, text) in enumerate(vo_lines):
        mp3_p = os.path.join(TMP_DIR, f"vo_d5_{i}.mp3")
        wav_p = os.path.join(TMP_DIR, f"vo_d5_{i}.wav")
        comm = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+24%", pitch="-2Hz")
        await comm.save(mp3_p)
        subprocess.run(
            [
                "ffmpeg", "-y", "-threads", "2", "-i", mp3_p,
                "-af", "silenceremove=start_periods=1:start_silence=0.03:start_threshold=-42dB:stop_periods=-1:stop_duration=0.12:stop_threshold=-42dB,areverse,silenceremove=start_periods=1:start_silence=0.04:start_threshold=-42dB,areverse,atempo=1.08",
                "-ac", "1", "-ar", str(sr), wav_p,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(wav_p, "rb") as wf:
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        dur_s = len(raw) / float(sr)
        actual_start = target_start
        gap_s = (actual_start - prev_end_s) if i > 0 else actual_start
        end_s = actual_start + dur_s
        print(f"  Sentence {i+1}: start={actual_start:.2f}s | dur={dur_s:.2f}s | end={end_s:.2f}s | gap_from_prev={gap_s:.2f}s | '{text}'")
        assert end_s <= (i + 1) * 5.0 - 0.85, f"Sentence {i+1} end {end_s:.2f}s too close to step boundary!"
        prev_end_s = end_s

        s_idx = int(actual_start * sr)
        e_idx = min(n_samples, s_idx + len(raw))
        master[s_idx:e_idx] += raw[: e_idx - s_idx] * 0.92
    print("========================================================\n")

    master = np.clip(master, -0.96, 0.96)
    pcm = (master * 32767.0).astype(np.int16)
    interleaved = np.empty(n_samples * 2, dtype=np.int16)
    interleaved[0::2] = pcm
    interleaved[1::2] = pcm
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())


def build_all():
    print("1. Loading native 3D sequences & extracting smoothed MediaPipe keypoints for Day 5 (Dumbbell Lateral Raise)...", flush=True)
    raw_seqs = {}
    kps_seqs = {}
    if os.path.exists(CACHE_KP_PKL):
        with open(CACHE_KP_PKL, "rb") as f:
            kps_seqs = pickle.load(f)

    for s_idx, (fn, i0, i1, is_w, v_mode, y_sh) in STEP_SPECS_DAY5.items():
        seq = load_native_slice_720p(os.path.join(ASSETS_DIR, fn), i0, i1)
        raw_seqs[s_idx] = seq
        if s_idx not in kps_seqs or len(kps_seqs[s_idx]) != len(seq) or "elb_r" not in kps_seqs[s_idx][0]:
            kps_seqs[s_idx] = extract_day5_kps_1080(seq, s_idx, y_sh)

    with open(CACHE_KP_PKL, "wb") as f:
        pickle.dump(kps_seqs, f)

    print("2. Building 4-slot top preview icons from each step's peak frame & auditing skin luminance...", flush=True)
    icons_all_4 = []
    for s_idx in [1, 2, 3, 4]:
        fn, i0, i1, is_w, v_mode, y_sh = STEP_SPECS_DAY5[s_idx]
        seq = raw_seqs[s_idx]
        kps = kps_seqs[s_idx]
        rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(seq[-1], kps[-1], y_sh, s_idx)
        sh_rgb, sh_m = shade_day5_3d_frame_1080p(rgb_1080, s_idx, is_w, kp_1080, pulse_phase=0.75)
        lum_out = 0.299 * sh_rgb[:, :, 0] + 0.587 * sh_rgb[:, :, 1] + 0.114 * sh_rgb[:, :, 2]
        act_d = np.maximum(
            np.clip(sh_rgb[:, :, 0].astype(float) - 0.5 * (sh_rgb[:, :, 1] + sh_rgb[:, :, 2]), 0, 255),
            np.clip(sh_rgb[:, :, 1].astype(float) - 0.5 * (sh_rgb[:, :, 0] + sh_rgb[:, :, 2]), 0, 255),
        )
        skin_chk = (lum_out > 46.0) & (sh_m < 0.2) & (act_d < 12.0)
        print(
            f"   Step {s_idx} Peak Skin Luminance: p50={np.percentile(lum_out[skin_chk], 50):.1f}, "
            f"p75={np.percentile(lum_out[skin_chk], 75):.1f}, p90={np.percentile(lum_out[skin_chk], 90):.1f}, "
            f"max={lum_out[skin_chk].max():.1f} | Shorts p50={np.percentile(lum_out[sh_m > 0.6], 50):.1f}",
            flush=True,
        )
        icons_all_4.append(b4._make_icon_pair(sh_rgb, kp_1080))
    icons_for_step = {1: icons_all_4, 2: icons_all_4, 3: icons_all_4, 4: icons_all_4}

    print("3. Synthesizing 20.0s English Coaching TTS (`en-US-ChristopherNeural`) + Sub-Bass Beat + Step Chimes...", flush=True)
    wav_path = os.path.join(TMP_DIR, "day5_4step_master_audio.wav")
    vo_lines = [
        (0.20, "Dumbbell lateral raise. Avoid flaring straight out and pinching your shoulder joint."),
        (5.20, "Shrugging the weights up overloads your upper traps instead of your side delts."),
        (10.20, "Depress your shoulder blades and lock your shoulder joint in a safe groove."),
        (15.20, "Raise thirty degrees forward in the scapular plane to fire your lateral deltoid fibers."),
    ]
    asyncio.run(synthesize_day5_voiceover_wav(wav_path, vo_lines, DURATION))

    print("4. Streaming 600 frames (20.0s @ 30fps) step-by-step with low memory footprint...", flush=True)
    tmp_mp4 = os.path.join(TMP_DIR, "_day5_video_only.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day5_dumbbell_lateral_raise_4step_master.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    step_peak_rendered = {}
    step_standing_rendered = {}

    for step_idx in [1, 2, 3, 4]:
        fn, i0, i1, is_w, v_mode, y_sh = STEP_SPECS_DAY5[step_idx]
        seq = raw_seqs[step_idx]
        kps = kps_seqs[step_idx]
        n_f = len(seq)
        step_cache = {}

        for f_local in range(150):
            u = f_local / 150.0
            # Controlled 1-Rep Coaching Tempo: 0.3s start -> 1.5s raise -> 1.3s isometric peak hold -> 1.5s lower -> 0.4s bottom
            if u < 0.06:
                wave_p = 0.0
            elif u < 0.36:
                wave_p = 0.5 - 0.5 * math.cos(math.pi * ((u - 0.06) / 0.30))
            elif u < 0.62:
                wave_p = 1.0
            elif u < 0.92:
                wave_p = 0.5 + 0.5 * math.cos(math.pi * ((u - 0.62) / 0.30))
            else:
                wave_p = 0.0
            p_i = int(round(wave_p * (n_f - 1)))

            pose_frac = p_i / float(max(1, n_f - 1))
            if p_i not in step_cache:
                rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(seq[p_i], kps[p_i], y_sh, step_idx)
                pulse_ph = pose_frac * 0.75
                base_rgb, sh_mask = shade_day5_3d_frame_1080p(rgb_1080, step_idx, is_w, kp_1080, pulse_phase=pulse_ph)
                if step_idx == 1:
                    slot_i = 0
                    deg_v = int(round(18 + 100 * pose_frac))
                    pct_v = int(round(76 + 12 * pose_frac))
                    deg_s, l1_s, l2_s = f"{deg_v}°", "HIGH FLARE", f"{pct_v}% JOINT PINCH"
                elif step_idx == 2:
                    slot_i = 1
                    deg_v = int(round(8 + 20 * pose_frac))
                    pct_v = int(round(68 + 8 * pose_frac))
                    deg_s, l1_s, l2_s = f"{deg_v}°", "TRAP SHRUG", f"{pct_v}% TRAP OVERLOAD"
                elif step_idx == 3:
                    slot_i = 2
                    pct_v = int(round(90 + 6 * pose_frac))
                    deg_s, l1_s, l2_s = "0°", "SCAPULA LOCK", f"{pct_v}% JOINT SAFE"
                else:
                    slot_i = 3
                    deg_v = int(round(12 + 18 * pose_frac))
                    pct_v = int(round(91 + 7 * pose_frac))
                    deg_s, l1_s, l2_s = f"{deg_v}°", "SCAPULAR PLANE", f"{pct_v}% LATERAL DELT"

                fr_comp = draw_day5_laser_and_callout(base_rgb, sh_mask, step_idx, kp_1080, is_w, deg_s, l1_s, l2_s)
                fg_m = b4.get_mannequin_fg_mask_1080p(base_rgb)
                fr_comp = b4.composite_depth_behind_title_and_translucent_preview(
                    fr_comp, fg_m, icons_for_step[step_idx], active_slot=slot_i, is_wrong=is_w, exercise_name="Dumbbell Lateral Raise"
                )
                step_cache[p_i] = fr_comp

            fr = step_cache[p_i]
            rgb_out = Image.fromarray(fr)

            if f_local == 4:
                step_standing_rendered[step_idx] = rgb_out.copy()
            if f_local == 50:
                step_peak_rendered[step_idx] = rgb_out.copy()
                rgb_out.save(os.path.join(OUT_DIR, f"day5_step{step_idx}_peak.jpg"), quality=95)
                if step_idx == 4:
                    rgb_out.save(os.path.join(OUT_DIR, "day5_dumbbell_lateral_raise_4step_master_cover.jpg"), quality=95)

            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

        step_cache.clear()
        gc.collect()
        print(f"   Finished Step {step_idx}/4 (150 frames written).", flush=True)

    vw.release()

    print("5. Muxing H.264 video + AAC 192k audio into final MP4...", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-threads", "4",
            "-i", tmp_mp4, "-i", wav_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_mp4)
    print("Saved Day 5 Master MP4:", final_mp4, flush=True)

    pw, ph = 520, 924
    pad = 22
    header_h = 96
    sheet = Image.new("RGB", (pad + 4 * (pw + pad), header_h + ph + pad), (14, 16, 20))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(26, "extrabold")
    f_s = v30.f_pretendard(18, "bold")
    sd.text(
        (28, 18),
        "Day 5 Dumbbell Lateral Raise — Bright Silver-Grey Mannequin (p75~165, p90~191) + 4-Step Bone/Muscle Master",
        font=f_t,
        fill=(242, 246, 252),
    )
    sd.text(
        (28, 54),
        "Step 1: 90° 팔 벌어짐(어깨관절 충돌 X-Ray) | Step 2: 28° 승모근 개입(상부승모 근육결 과부하) | Step 3: 0° 견갑 하강 고정(관절 X-Ray) | Step 4: 30° 견갑면(측면삼각근 근육결)",
        font=f_s,
        fill=(185, 225, 255),
    )
    step_titles = [
        (1, True,  "Step 1: 오자세 뼈 X-Ray (118° HIGH FLARE · 과거상 관절 충돌)"),
        (2, True,  "Step 2: 오자세 근육결 (28° TRAP SHRUG · 승모 과부하)"),
        (3, False, "Step 3: 정자세 뼈 X-Ray (0° SCAPULA LOCK · 견갑 고정)"),
        (4, False, "Step 4: 정자세 근육결 (30° SCAPULAR PLANE · 측면삼각근)"),
    ]
    for idx_p, (s_i, is_w, lbl) in enumerate(step_titles):
        x0 = pad + idx_p * (pw + pad)
        y0 = header_h
        im_p = step_peak_rendered[s_i].resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(im_p, (x0, y0))
        b_col = (245, 65, 65) if is_w else (35, 235, 130)
        sd.rectangle([x0 - 2, y0 - 2, x0 + pw + 1, y0 + ph + 1], outline=b_col, width=3)
        sd.rectangle([x0 + 10, y0 + 10, x0 + pw - 10, y0 + 48], fill=(12, 14, 18))
        sd.text((x0 + 20, y0 + 18), lbl, font=v30.f_pretendard(16, "bold"), fill=b_col)

    sheet_path = os.path.join(OUT_DIR, "day5_4step_multiangle_master_sheet.jpg")
    sheet.save(sheet_path, quality=94)

    caption_text = (
        "Be honest — where do you feel Dumbbell Lateral Raises more when the set gets tough: "
        "1 (Upper Traps & Neck) or 2 (Side Delts)? Drop 1 or 2 in the comments 👇\n\n"
        "Dumbbell Lateral Raise — 4-Step 3D Biomechanics Breakdown:\n\n"
        "• Step 1 (90° Arm Flare — Joint Pinch): Flaring your arms straight out at 90° with internal rotation jams the greater tubercle of the humerus against the acromion, creating up to 84% shoulder joint impingement risk.\n"
        "• Step 2 (28° Trap Shrug — Trap Overload): Elevating your shoulder girdle to heave the dumbbells shifts up to 76% of the load onto your upper trapezius fibers instead of your deltoids.\n"
        "• Step 3 (0° Scapula Lock — Joint Safe): Depressing your shoulder blades and locking the humeral head centered in the glenoid socket keeps the subacromial space wide open (96% joint safety).\n"
        "• Step 4 (30° Scapular Plane — 98% Lateral Delt): Raising the dumbbells 30° forward in the scapular plane while leading with your elbows aligns the line of pull directly with your multipennate lateral deltoid fibers.\n\n"
        "Save this breakdown for your next shoulder day and share it with a training partner."
    )
    with open(os.path.join(OUT_DIR, "day5_caption_en.txt"), "w", encoding="utf-8") as f:
        f.write(caption_text + "\n")
    print("Saved Day 5 4-Step Sheet & Caption:", sheet_path, flush=True)


if __name__ == "__main__":
    build_all()
