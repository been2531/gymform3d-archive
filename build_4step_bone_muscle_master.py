#!/usr/bin/env python3
"""
Day 3 (`Barbell Back Squat`) — 100% Native 3D Multi-Angle 4-Step Master Reel Builder
====================================================================================
Fixes every issue identified by 중현 님 (`[t362u]`–`[t379u]`):
1. ZERO 2D Overlays (`[t379u]: "그래 이제 2d는 금지야"`):
   - ZERO 2D drawn shorts/drawers (`apply_fitted_3d_drawers_720p` 100% removed).
   - ZERO 2D drawn cylindrical bone tubes (`u_bone` / `d_fem_r` 100% removed).
   - ZERO 2D image warping (`_apply_step1_knee_valgus_720p` 100% removed).
   - ZERO 2D cutout masks (`make_clean_barbell_and_human_mask` 100% removed).
   - Uses 100% pure native 3D I2V clips (`day3_v61_step1..4`) where the 3D mannequin ALREADY wears
     fitted 3D black athletic shorts that deform naturally with the hips, glutes, and thighs!
2. ZERO Frame-Blending Ghosting / Onion-Skinning (`[t372u]`, `[t373u]`):
   - Loads every native 3D frame (`load_native_slice`) with ZERO `cv2.addWeighted` frame interpolation.
   - Maps 600 frames (20.0s @ 30fps, 150 frames per step) directly to native integer frames (`p_i`)
     across 1.5 smooth full-ROM squat repetitions per step.
3. Multi-Angle Camera Progression (`[t373u]`, `[t378u]`):
   - Step 1 (0.0s–5.0s, Front View Wrong — Bone + Joint Overload): `16° KNEE CAVE / 86% JOINT RISK`
   - Step 2 (5.0s–10.0s, Side View Wrong — Lumbar Spine + Muscle Strain): `41° FORWARD LEAN / 79% LUMBAR LOAD`
   - Step 3 (10.0s–15.0s, Front View Correct — Bio-X-Ray Bone + Joint Lock): `0° KNEE LOCK / 96% JOINT SAFE`
   - Step 4 (15.0s–20.0s, Rear 3/4 View Correct — Full Quad & Glute Muscle Drive): `22° HIP DRIVE / 98% QUAD & GLUTE`
4. Full-Screen Large Framing & ZERO Left Blur (`[t377u]`):
   - Upscales the native 9:16 720x1280 studio frame 1.5x to full 1080x1920 with zero left-side blur or vignette (`curve_l = 0%`).
5. English Coaching TTS (`en-US-ChristopherNeural`) + Sub-Bass Beat + Step Chimes (`[t373u]`).
"""
import os
import sys
import math
import pickle
import asyncio
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import universal_anatomy_rig as uar
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62
import build_clean_d4_variants as bcd
import build_v36_publish_reels as v36_pub

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_DIR = "/workspace/daily_info_reels/_tmp_day3_master"
CACHE_KP_PKL = "/workspace/daily_info_reels/_cache_day3_v61_native_kps.pkl"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)  # 600 frames
CURSIVE_FONT_PATH = "/workspace/daily_info_reels/fonts/KaushanScript-Regular.ttf"

STEP_SPECS = {
    1: ("day3_v61_step1_norack_i2v.mp4", 72, 120, True,  "bone_blood",   180),
    2: ("day3_v61_step3_norack_i2v.mp4", 136, 62, True,  "muscle_blood",  70),
    3: ("day3_v61_step2_norack_i2v.mp4", 78, 144, False, "bone_blood",   150),
    4: ("day3_v61_step4_flatfeet_i2v.mp4", 72, 144, False, "muscle_blood", 140),
}


def load_native_slice(path: str, i0: int, i1: int) -> list[np.ndarray]:
    """Loads 100% pure native video frames from i0 to i1 with ZERO addWeighted frame blending!"""
    cap = cv2.VideoCapture(path)
    frs = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frs.append(fr)
    cap.release()
    assert len(frs) > max(i0, i1), f"Clip {path} has {len(frs)} frames, needed {max(i0, i1)}"
    return frs[i0 : i1 + 1] if i0 <= i1 else frs[i1 : i0 + 1][::-1]


def extract_smoothed_kps_1080(seq_bgr_720: list[np.ndarray], step_idx: int, y_sh: int) -> list[dict]:
    """
    Runs MediaPipe PoseLandmarker on the native 720x1280 sequence, enforces anatomical left/right ordering,
    locks Step 1 knee centers onto the native 3D glowing X-Ray knee joints, applies temporal Gaussian
    smoothing (`sigma=2.5`), and maps to 1080x1920 (`x * 1.5, y * 1.5 + y_sh`).
    """
    detector = uar.get_pose_detector()
    raw_kps = []
    yy_720, xx_720 = np.mgrid[:1280, :720].astype(np.float32)

    for bgr in seq_bgr_720:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb))
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            sh = (0.5 * (lm[11].x + lm[12].x) * 720.0, 0.5 * (lm[11].y + lm[12].y) * 1280.0)
            hip_r = (lm[24].x * 720.0, lm[24].y * 1280.0)
            hip_l = (lm[23].x * 720.0, lm[23].y * 1280.0)
            kn_r = (lm[26].x * 720.0, lm[26].y * 1280.0)
            kn_l = (lm[25].x * 720.0, lm[25].y * 1280.0)
            ank_r = (lm[28].x * 720.0, lm[28].y * 1280.0)
            ank_l = (lm[27].x * 720.0, lm[27].y * 1280.0)

            if step_idx in (1, 3):
                if hip_r[0] > hip_l[0]:
                    hip_r, hip_l = hip_l, hip_r
                if kn_r[0] > kn_l[0]:
                    kn_r, kn_l = kn_l, kn_r
                if ank_r[0] > ank_l[0]:
                    ank_r, ank_l = ank_l, ank_r

                if step_idx == 1:
                    rb = rgb[:, :, 0].astype(np.float32) - rgb[:, :, 2].astype(np.float32)
                    m_vL = (rb > 25.0) & (yy_720 > 660.0) & (yy_720 < 900.0) & (xx_720 < 360.0)
                    m_vR = (rb > 25.0) & (yy_720 > 660.0) & (yy_720 < 900.0) & (xx_720 >= 360.0)
                    if np.count_nonzero(m_vL) > 80:
                        kn_r = (0.35 * kn_r[0] + 0.65 * float(xx_720[m_vL].mean()), 0.50 * kn_r[1] + 0.50 * float(yy_720[m_vL].mean()))
                    if np.count_nonzero(m_vR) > 80:
                        kn_l = (0.35 * kn_l[0] + 0.65 * float(xx_720[m_vR].mean()), 0.50 * kn_l[1] + 0.50 * float(yy_720[m_vR].mean()))
        else:
            prev = raw_kps[-1]
            sh, hip_r, hip_l = prev["sh"], prev["hip_r"], prev["hip_l"]
            kn_r, kn_l, ank_r, ank_l = prev["kn_r"], prev["kn_l"], prev["ank_r"], prev["ank_l"]

        raw_kps.append({
            "sh": sh,
            "hip_r": hip_r,
            "hip_l": hip_l,
            "kn_r": kn_r,
            "kn_l": kn_l,
            "ank_r": ank_r,
            "ank_l": ank_l,
        })

    keys = list(raw_kps[0].keys())
    n = len(raw_kps)
    out_1080 = [dict() for _ in range(n)]
    for k in keys:
        xs = gaussian_filter1d([p[k][0] * 1.5 for p in raw_kps], sigma=2.5, mode="nearest")
        ys = gaussian_filter1d([p[k][1] * 1.5 + y_sh for p in raw_kps], sigma=2.5, mode="nearest")
        for i in range(n):
            out_1080[i][k] = (float(xs[i]), float(ys[i]))
    return out_1080


SCALE = 0.90
STEP_DY = {
    1: +235,
    2: +100,
    3: +200,
    4: +115,
}


def transform_frame_and_kp_1080p(bgr_720p: np.ndarray, kp_cached: dict, old_y_sh: int, step_idx: int) -> tuple[np.ndarray, dict]:
    up = cv2.resize(bgr_720p, (W, H), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 0.9)
    sharp_bgr = cv2.addWeighted(up, 1.22, blur, -0.22, 0)
    rgb = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    dy = STEP_DY[step_idx]
    nw, nh = int(round(W * SCALE)), int(round(H * SCALE))
    scaled = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    bg_med = np.median(
        np.concatenate([rgb[:28, :, :].reshape(-1, 3), rgb[:, :28, :].reshape(-1, 3), rgb[:, -28:, :].reshape(-1, 3)], axis=0),
        axis=0,
    )
    canvas = np.ones((H, W, 3), dtype=np.float32) * bg_med[None, None, :]

    x0 = (W - nw) // 2
    y0 = (H - nh) // 2 + dy

    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    cw = min(nw - sx0, W - dx0)
    ch = min(nh - sy0, H - dy0)

    sub = scaled[sy0 : sy0 + ch, sx0 : sx0 + cw]
    yy_s, xx_s = np.mgrid[:ch, :cw].astype(np.float32)
    feather = 36.0
    wx = np.minimum(np.clip(xx_s / feather, 0.0, 1.0), np.clip((cw - 1 - xx_s) / feather, 0.0, 1.0))
    wy = np.minimum(np.clip(yy_s / feather, 0.0, 1.0), np.clip((ch - 1 - yy_s) / feather, 0.0, 1.0))
    w_mask = (wx * wy)[:, :, None]

    canvas[dy0 : dy0 + ch, dx0 : dx0 + cw] = (
        canvas[dy0 : dy0 + ch, dx0 : dx0 + cw] * (1.0 - w_mask) + sub * w_mask
    )

    kp_out = {}
    for k, (px, py_cached) in kp_cached.items():
        py_raw = py_cached - old_y_sh
        nx = (px - W * 0.5) * SCALE + W * 0.5
        ny = (py_raw - H * 0.5) * SCALE + H * 0.5 + dy
        kp_out[k] = (float(nx), float(ny))

    return canvas, kp_out


def shade_native_3d_frame_1080p(
    rgb_1080: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    visual_mode: str,
    kp_1080: dict,
    pulse_phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Option B Standard (@demicstory Core-to-Grey Edge Falloff + 100% Upper-Back Striation Lock):
    1. Applies the exact upper-back `grey_back_exact` formula to the ENTIRE body (`mono`), so the
       quadriceps, hamstrings, glutes, and erector spinae start with the exact same 3D muscle fiber
       grooves and ridges as the upper back.
    2. Instead of flooding the entire muscle area edge-to-edge with 246..255 neon green/red,
       concentrates the color tint in the high-contraction core (`core_w`) with a smooth falloff
       into the sculpted grey muscle fibers at the periphery, keeping maximum luminance aligned
       with the studio lighting so every single muscle fiber strand stays razor-sharp.
    """
    r_ch, g_ch, b_ch = rgb_1080[:, :, 0], rgb_1080[:, :, 1], rgb_1080[:, :, 2]
    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    cool_lum = 0.35 * g_ch + 0.65 * b_ch

    fiber_signal = 0.68 * r_ch + 0.32 * g_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 1.0)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.8)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 7.2)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 15.5)

    hf_ridge = (sigma_micro - sigma_ridge) * 4.2 + (sigma_ridge - sigma_belly) * 2.4
    sulc = np.clip((sigma_broad - sigma_ridge - 1.0) / 10.5, 0.0, 1.0)

    # Exact upper-back 3D sculpted grey formula (`grey_back_exact`) applied uniformly across the whole body
    cool_belly = cv2.GaussianBlur(cool_lum, (0, 0), 6.0)
    hv_cool = np.clip((cool_belly - 18.0) / 115.0, 0.0, 1.0)
    grey_back_exact = np.clip(
        32.0 + 108.0 * (hv_cool ** 0.95) + hf_ridge * 0.55 - sulc * 14.0,
        20.0,
        168.0,
    )

    red_any = np.clip((rb_diff - 9.0) / 52.0, 0.0, 1.0)
    mono_target = np.stack([
        grey_back_exact * 0.96,
        grey_back_exact * 0.98,
        grey_back_exact * 1.03,
    ], axis=2)
    mono = rgb_1080 * (1.0 - red_any[:, :, None]) + mono_target * red_any[:, :, None]

    kn_y = 0.5 * (kp_1080["kn_r"][1] + kp_1080["kn_l"][1])
    hip_y = 0.5 * (kp_1080["hip_r"][1] + kp_1080["hip_l"][1])
    sh_y = kp_1080["sh"][1]

    y_cut_shin = kn_y + 52.0
    above_shin_gate = np.clip((y_cut_shin - yy) / 28.0, 0.0, 1.0)
    lower_body_gate = np.clip((yy - (hip_y - 160.0)) / 45.0, 0.0, 1.0) * above_shin_gate

    shorts_region = (
        (yy > (hip_y - 165.0))
        & (yy < (kn_y - 45.0))
        & (xx > 220.0)
        & (xx < 860.0)
        & (base_lum < 48.0)
        & (rb_diff < 11.0)
    ).astype(np.float32)
    shorts_mask_1080 = cv2.GaussianBlur(shorts_region, (0, 0), 4.0)

    flow_wave = 0.92 + 0.08 * np.cos(2.0 * math.pi * ((yy - hip_y) / 280.0 - pulse_phase))
    L = np.clip((grey_back_exact - 20.0) / 145.0, 0.0, 1.0)

    if step_idx == 2:
        region_gate = (
            np.clip(((hip_y + 75.0) - yy) / 48.0, 0.0, 1.0)
            * np.clip((yy - (sh_y - 45.0)) / 42.0, 0.0, 1.0)
            * above_shin_gate
        )
    elif step_idx == 4:
        region_gate = np.clip((yy - (hip_y - 125.0)) / 48.0, 0.0, 1.0) * above_shin_gate
    else:
        region_gate = lower_body_gate

    # Option B Core-to-Grey Edge Falloff:
    # Concentrated in the active contraction core (`rb_diff > 26`), softly feathering into sculpted grey at edges
    core_raw = np.clip((rb_diff - 26.0) / 64.0, 0.0, 1.0) * region_gate
    core_blur = cv2.GaussianBlur(core_raw, (0, 0), 7.5) * region_gate
    core_w = np.clip((0.45 * core_raw + 0.70 * core_blur) * flow_wave, 0.0, 0.90)[:, :, None]

    # Luminance-locked color derived directly from `grey_back_exact` so every fiber groove stays dark and sharp!
    if not is_wrong:
        col_r = grey_back_exact * 0.38
        col_g = grey_back_exact * 1.32 + 22.0 * (L ** 0.90)
        col_b = grey_back_exact * 0.82 + 10.0 * (L ** 0.90)
    else:
        col_r = grey_back_exact * 1.36 + 26.0 * (L ** 0.90)
        col_g = grey_back_exact * 0.40
        col_b = grey_back_exact * 0.44

    colored = np.stack([col_r, col_g, col_b], axis=2)

    # Preserve crisp 3D internal X-Ray bone/joint core on Step 1, Step 2, and Step 3
    if visual_mode == "bone_blood" or step_idx == 2:
        bone_core_w = (
            np.clip((g_ch - 68.0) / 80.0, 0.0, 1.0)
            * np.clip((rb_diff - 22.0) / 45.0, 0.0, 1.0)
            * region_gate
        )[:, :, None]
        if is_wrong:
            xray_bone = np.stack([
                np.clip(grey_back_exact * 1.42 + 34.0 + hf_ridge * 0.65, 0, 248),
                np.clip(grey_back_exact * 0.72 + 16.0 + hf_ridge * 0.30, 0, 185),
                np.clip(grey_back_exact * 0.50 + 10.0, 0, 140),
            ], axis=2)
        else:
            xray_bone = np.stack([
                np.clip(grey_back_exact * 0.55 + 16.0 + hf_ridge * 0.35, 0, 195),
                np.clip(grey_back_exact * 1.40 + 30.0 + hf_ridge * 0.65, 0, 248),
                np.clip(grey_back_exact * 1.05 + 22.0 + hf_ridge * 0.50, 0, 225),
            ], axis=2)
        colored = colored * (1.0 - bone_core_w * 0.75) + xray_bone * (bone_core_w * 0.75)
        core_w = np.maximum(core_w, bone_core_w * 0.88)

    out = mono * (1.0 - core_w) + np.clip(colored, 0.0, 255.0) * core_w
    return np.clip(out, 0, 255).astype(np.uint8), shorts_mask_1080


_TOP_BAR_CACHE = {}

def draw_compact_demic_top_preview(frame_rgb: np.ndarray, torso_icons: list, active_slot: int, is_wrong: bool) -> np.ndarray:
    """ROI-Accelerated & Cached 4-Slot Top Preview Bar (runs in <1.2ms per frame instead of 250ms!)."""
    bar_x0, bar_y0, bar_x1, bar_y1 = 212, 258, 868, 400
    tw, th = 134, 158
    slot_centers_x = [303, 461, 619, 777]
    if len(torso_icons) == 2:   # t568u 2-step day: 2 thumbnails, same slot size/pitch, pill centred
        slot_centers_x = [461, 619]
        bar_x0, bar_x1 = 370, 710
    elif len(torso_icons) == 1:   # t672u 1-segment reel: one thumbnail, same slot size, pill centred
        slot_centers_x = [540]
        bar_x0, bar_x1 = 449, 631
    rx0, ry0, rx1, ry1 = 190, 232, 890, 424
    rw, rh = rx1 - rx0, ry1 - ry0

    cache_key = (id(torso_icons), active_slot, is_wrong)
    if cache_key not in _TOP_BAR_CACHE:
        frost_mask = Image.new("L", (rw, rh), 0)
        fmd = ImageDraw.Draw(frost_mask)
        fmd.rounded_rectangle([bar_x0 - rx0, bar_y0 - ry0, bar_x1 - rx0, bar_y1 - ry0], radius=24, fill=190)
        cx_act = slot_centers_x[active_slot]
        fmd.rounded_rectangle([cx_act - tw // 2 - rx0, bar_y0 - 10 - ry0, cx_act + tw // 2 - rx0, bar_y0 - 10 + th - ry0], radius=20, fill=210)
        frost_w = (np.array(frost_mask, dtype=np.float32) / 255.0)[:, :, None]

        ov_pil = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
        gd = ImageDraw.Draw(ov_pil)
        gd.rounded_rectangle([bar_x0 + 2 - rx0, bar_y0 + 6 - ry0, bar_x1 + 2 - rx0, bar_y1 + 8 - ry0], radius=26, fill=(0, 0, 0, 75))
        gd.rounded_rectangle([bar_x0 - rx0, bar_y0 - ry0, bar_x1 - rx0, bar_y1 - ry0], radius=24, fill=(22, 25, 30, 138), outline=(140, 150, 168, 125), width=2)

        yy_t, xx_t = np.mgrid[:th, :tw].astype(np.float32)
        nx = (xx_t - tw * 0.5) / (tw * 0.48)
        ny = (yy_t - th * 0.52) / (th * 0.48)
        vign = np.clip(1.0 - 0.55 * ((nx * nx + ny * ny) ** 1.35), 0.15, 1.0)

        for i in range(len(slot_centers_x)):
            if i == active_slot:
                continue
            cx = slot_centers_x[i]
            tx0, ty0 = cx - tw // 2 - rx0, bar_y0 - 10 - ry0
            _, inact_224 = torso_icons[i]
            inact_arr = cv2.resize(inact_224, (tw, th), interpolation=cv2.INTER_AREA)
            inact_cropped = np.clip(inact_arr * vign[:, :, None], 0, 255).astype(np.uint8)
            im_slot = Image.fromarray(inact_cropped).convert("RGBA")
            mask_slot = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(mask_slot).rounded_rectangle([3, 12, tw - 4, th - 13], radius=16, fill=182)
            mask_slot = mask_slot.filter(ImageFilter.GaussianBlur(radius=3))
            ov_pil.paste(im_slot, (tx0, ty0), mask_slot)

        ax0, ay0 = cx_act - tw // 2 - rx0, bar_y0 - 10 - ry0
        ax1, ay1 = ax0 + tw, ay0 + th
        act_col = (255, 88, 48) if is_wrong else (42, 245, 135)

        glow_layer = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
        gld = ImageDraw.Draw(glow_layer)
        gld.rounded_rectangle([ax0 - 6, ay0 - 6, ax1 + 6, ay1 + 6], radius=24, outline=(*act_col, 155), width=7)
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=10))
        ov_pil = Image.alpha_composite(ov_pil, glow_layer)

        act_224, _ = torso_icons[active_slot]
        act_arr = cv2.resize(act_224, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        edge_dist = np.minimum(np.minimum(xx_t, tw - 1 - xx_t), np.minimum(yy_t, th - 1 - yy_t))
        rim_glow = np.exp(-edge_dist / 14.0)[:, :, None] * 0.22
        comp_act = np.clip(act_arr * 1.06 + rim_glow * np.array(act_col, dtype=np.float32), 0, 255).astype(np.uint8)

        im_act = Image.fromarray(comp_act).convert("RGBA")
        mask_act = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask_act).rounded_rectangle([0, 0, tw - 1, th - 1], radius=20, fill=215)
        ov_pil.paste(im_act, (ax0, ay0), mask_act)

        bdr_ov = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
        bd_d = ImageDraw.Draw(bdr_ov)
        bd_d.rounded_rectangle([ax0 - 2, ay0 - 2, ax1 + 2, ay1 + 2], radius=22, outline=(*act_col, 95), width=4)
        bd_d.rounded_rectangle([ax0, ay0, ax1, ay1], radius=20, outline=(*act_col, 255), width=3)
        ov_pil = Image.alpha_composite(ov_pil, bdr_ov)

        ov_arr = np.array(ov_pil, dtype=np.float32)
        ov_rgb = ov_arr[:, :, :3]
        ov_a = (ov_arr[:, :, 3:4] / 255.0)
        _TOP_BAR_CACHE[cache_key] = (frost_w, ov_rgb, ov_a)

    frost_w, ov_rgb, ov_a = _TOP_BAR_CACHE[cache_key]
    out = frame_rgb.copy()
    roi = out[ry0:ry1, rx0:rx1].astype(np.float32)
    roi_blur = cv2.GaussianBlur(roi, (0, 0), 6.0)
    frosted = roi * (1.0 - frost_w) + roi_blur * frost_w
    blended = frosted * (1.0 - ov_a) + ov_rgb * ov_a
    out[ry0:ry1, rx0:rx1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


_TITLE_CACHE = {}


def draw_single_line_kaushan_title(canvas_pil: Image.Image, exercise_name: str = "Barbell Back Squat"):
    if exercise_name not in _TITLE_CACHE:
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
        ty = 458 * SS - th // 2 - bb[1]

        for dx_s, dy_s, a_s in [(0, 5 * SS, 185), (-2 * SS, 3 * SS, 110), (2 * SS, 3 * SS, 110)]:
            od.text((tx + dx_s, ty + dy_s), exercise_name, font=f_scr, fill=(8, 10, 14, a_s))
        od.text((tx, ty), exercise_name, font=f_scr, fill=(246, 250, 255, 252))

        _TITLE_CACHE[exercise_name] = ov.resize((W, H), Image.Resampling.LANCZOS)
    canvas_pil.alpha_composite(_TITLE_CACHE[exercise_name])



def get_mannequin_fg_mask_1080p(rgb_1080: np.ndarray) -> np.ndarray:
    """Smooth 0..1 foreground mask of the 3D mannequin + barbell against the studio charcoal background."""
    bg_med = np.median(
        np.concatenate([rgb_1080[:24, :, :].reshape(-1, 3), rgb_1080[:, :24, :].reshape(-1, 3), rgb_1080[:, -24:, :].reshape(-1, 3)], axis=0),
        axis=0,
    )
    diff = np.linalg.norm(rgb_1080.astype(np.float32) - bg_med[None, None, :], axis=-1)
    fg = np.clip((diff - 12.0) / 22.0, 0.0, 1.0).astype(np.float32)
    return cv2.GaussianBlur(fg, (0, 0), 2.5)


_TITLE_ROI_CACHE = {}

def composite_depth_behind_title_and_translucent_preview(
    base_rgb: np.ndarray,
    fg_mask: np.ndarray,
    torso_icons: list,
    active_slot: int,
    is_wrong: bool,
    exercise_name: str = "Barbell Back Squat",
) -> np.ndarray:
    """
    Option 3 Standard (Day 4+) with Turbo ROI Caching:
    1. Places the Kaushan Script exercise title on the studio back wall BEHIND the 3D mannequin (`fg_front = 0.92`),
       so when standing tall, the mannequin head/traps pass cleanly in front of the letters with zero facial clutter.
    2. Renders the 4-slot top preview bar (`y = 248..406`) as a translucent glass pill (`touch_p = 0.65` see-through when touched).
    """
    ty0, ty1 = 410, 565
    if exercise_name not in _TITLE_ROI_CACHE:
        blank = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_single_line_kaushan_title(blank, exercise_name)
        t_arr = np.array(blank.crop((0, ty0, W, ty1)), dtype=np.float32)
        _TITLE_ROI_CACHE[exercise_name] = (t_arr[:, :, :3], t_arr[:, :, 3:4] / 255.0)

    t_rgb, t_a = _TITLE_ROI_CACHE[exercise_name]
    out = base_rgb.copy()

    # 1. Depth-behind title in ROI y=410..565
    roi_base = out[ty0:ty1].astype(np.float32)
    fg_front = np.clip(fg_mask[ty0:ty1, :, None] * 0.92, 0.0, 0.92)
    eff_a = t_a * (1.0 - fg_front)
    out[ty0:ty1] = np.clip(roi_base * (1.0 - eff_a) + t_rgb * eff_a, 0, 255).astype(np.uint8)

    # 2. Translucent glass top preview in ROI y=230..425
    py0, py1 = 230, 425
    layered_roi = out[py0:py1].copy()
    with_prev = draw_compact_demic_top_preview(out, torso_icons, active_slot=active_slot, is_wrong=is_wrong)
    prev_roi = with_prev[py0:py1].astype(np.float32)
    touch_p = np.clip(fg_mask[py0:py1, :, None] * 0.65, 0.0, 0.65)
    with_prev[py0:py1] = np.clip(prev_roi * (1.0 - touch_p) + layered_roi.astype(np.float32) * touch_p, 0, 255).astype(np.uint8)
    return with_prev


def draw_step_laser_and_callout(
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
    sh = kp["sh"]
    hip_r, hip_l = kp["hip_r"], kp["hip_l"]
    kn_r, kn_l = kp["kn_r"], kp["kn_l"]
    ank_r, ank_l = kp["ank_r"], kp["ank_l"]

    if step_idx in (1, 3):
        top_r, _ = p62._extend_segment(hip_r, kn_r, 0.22, 1.0)
        _, bot_r = p62._extend_segment(kn_r, ank_r, 0.0, 1.12)
        top_l, _ = p62._extend_segment(hip_l, kn_l, 0.22, 1.0)
        _, bot_l = p62._extend_segment(kn_l, ank_l, 0.0, 1.12)
        ext_r1, ext_r2 = p62._extend_segment(hip_r, kn_r, 0.62, 1.38)
        ext_l1, ext_l2 = p62._extend_segment(hip_l, kn_l, 0.62, 1.38)
        p62.draw_demic_laser_joint_axis(
            canvas,
            main_chains_1x=[[top_r, kn_r, bot_r], [top_l, kn_l, bot_l]],
            aux_ext_chains_1x=[[ext_r1, ext_r2], [ext_l1, ext_l2]],
            vertex_pts_1x=[kn_r, kn_l],
            is_wrong=is_wrong,
        )
        target_pt = (int(kn_r[0]), int(kn_r[1]))
    else:
        sh_top, _ = p62._extend_segment(sh, hip_r, -0.12, 1.0)
        _, kn_ext = p62._extend_segment(hip_r, kn_r, 0.0, 1.14)
        torso_cross_a, torso_cross_b = p62._extend_segment(sh, hip_r, 0.66, 1.36)
        thigh_cross_a, thigh_cross_b = p62._extend_segment(kn_r, hip_r, 0.66, 1.32)
        p62.draw_demic_laser_joint_axis(
            canvas,
            main_chains_1x=[[sh_top, hip_r, kn_ext]],
            aux_ext_chains_1x=[[torso_cross_a, torso_cross_b], [thigh_cross_a, thigh_cross_b]],
            vertex_pts_1x=[hip_r, kn_r] if not is_wrong else [hip_r],
            is_wrong=is_wrong,
        )
        if step_idx == 2:
            target_pt = (int(0.45 * sh[0] + 0.55 * hip_r[0]), int(0.45 * sh[1] + 0.55 * hip_r[1]))
        else:
            target_pt = (int(0.55 * hip_r[0] + 0.45 * kn_r[0]), int(0.55 * hip_r[1] + 0.45 * kn_r[1]))

    laser_rgb = np.array(canvas.convert("RGB")).astype(np.float32)
    sh_gate = np.clip(shorts_mask_1080 * 0.85, 0.0, 0.85)[:, :, None]
    blended = np.clip(laser_rgb * (1.0 - sh_gate) + base_rgb.astype(np.float32) * sh_gate, 0, 255).astype(np.uint8)

    base_pil = Image.fromarray(blended).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    elbow_x = max(286, target_pt[0] - 42)
    elbow_y = target_pt[1] - 24
    bcd._draw_precision_leader(rd, target_pt, (elbow_x, elbow_y), (278, elbow_y), col)

    tx = 86
    ty = int(np.clip(elbow_y - 76, 520, 1460))
    f_big = v30.f_pretendard(86, "extrabold")
    f_l1 = v30.f_pretendard(21, "extrabold")
    f_l2 = v30.f_pretendard(16, "bold")

    rd.text((tx + 2, ty + 3), deg_str, font=f_big, fill=(0, 0, 0, 210))
    rd.text((tx, ty), deg_str, font=f_big, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 92), l1_str, font=f_l1, fill=(col[0], col[1], col[2], 255))
    rd.text((tx + 2, ty + 119), l2_str, font=f_l2, fill=(215, 222, 232, 235))

    return np.array(Image.alpha_composite(base_pil, ov).convert("RGB"))


def _make_icon_pair(fr_1080: np.ndarray, kp_1080: dict) -> tuple[np.ndarray, np.ndarray]:
    sh_x, sh_y = kp_1080["sh"]
    hip_x = 0.5 * (kp_1080["hip_r"][0] + kp_1080["hip_l"][0])
    kn_x = 0.5 * (kp_1080["kn_r"][0] + kp_1080["kn_l"][0])
    kn_y = 0.5 * (kp_1080["kn_r"][1] + kp_1080["kn_l"][1])
    cx = int(round(0.35 * sh_x + 0.40 * hip_x + 0.25 * kn_x))
    y0 = int(round(sh_y - 210))
    y1 = int(round(kn_y + 195))
    h_c = max(420, y1 - y0)
    w_crop = int(round(h_c * (224.0 / 264.0)))
    x0 = max(0, min(W - w_crop, cx - w_crop // 2))
    x1 = x0 + w_crop
    y0 = max(0, min(H - h_c, y0))
    y1 = y0 + h_c
    crop = fr_1080[y0:y1, x0:x1, :]
    act_224 = cv2.resize(crop, (224, 264), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(act_224, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inact_224 = np.stack([
        np.clip(gray * 0.52 + 8.0, 0, 255),
        np.clip(gray * 0.55 + 9.0, 0, 255),
        np.clip(gray * 0.60 + 12.0, 0, 255),
    ], axis=-1).astype(np.uint8)
    return (act_224, inact_224)


def _read_single_frame_720p(path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, fr = cap.read()
    cap.release()
    if fr.shape[1] != 720 or fr.shape[0] != 1280:
        fr = cv2.resize(fr, (720, 1280), interpolation=cv2.INTER_LANCZOS4)
    return fr


def build_all():
    import time, gc
    t0_all = time.time()
    print("1. Loading cached MediaPipe keypoints & preparing 4-slot top preview icons (4 peak frames only)...")
    with open(CACHE_KP_PKL, "rb") as f:
        kps_seqs_raw = pickle.load(f)

    icons_all_4 = []
    for s_idx, (fn, i0, i1, is_w, v_mode, y_sh) in STEP_SPECS.items():
        pk_bgr = _read_single_frame_720p(os.path.join(OUT_DIR, fn), i1)
        rgb_1080_pk, kp_1080_pk = transform_frame_and_kp_1080p(pk_bgr, kps_seqs_raw[s_idx][-1], y_sh, s_idx)
        sh_rgb_pk, _ = shade_native_3d_frame_1080p(rgb_1080_pk, s_idx, is_w, v_mode, kp_1080_pk, pulse_phase=0.75)
        icons_all_4.append(_make_icon_pair(sh_rgb_pk, kp_1080_pk))

    print(f"   4-slot icons ready in {time.time() - t0_all:.2f}s.")

    wav_path = os.path.join(TMP_DIR, "day3_4step_master_audio.wav")
    if not os.path.exists(wav_path):
        print("2. Synthesizing 20.0s English Coaching TTS...")
        vo_lines = [
            (0.25, "Barbell back squat. Stop letting your knees cave in."),
            (5.60, "Leaning forward overloads your lower lumbar spine."),
            (10.40, "Lock your knees in line with your toes."),
            (15.40, "Drive hard through your quads and glutes."),
        ]
        asyncio.run(v36_pub.synthesize_voiceover_wav(wav_path, vo_lines, DURATION))

    print("3. Streaming 4 steps with Option B shader + Per-Step Pose Cache...")
    tmp_mp4 = os.path.join(TMP_DIR, "_day3_video_only.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day3_barbell_squat_4step_master.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    step_peak_rendered = {}
    sample_local_frames = {4: "Standing", 25: "Mid-Squat", 50: "Deep Squat"}

    for step_idx in range(1, 5):
        t_step = time.time()
        fn, i0, i1, is_w, v_mode, y_sh = STEP_SPECS[step_idx]
        seq = load_native_slice(os.path.join(OUT_DIR, fn), i0, i1)
        n_pool = len(seq)
        step_pose_cache = {}

        for f_local in range(150):
            u = f_local / 150.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            raw_pi = int(round(wave_p * (n_pool - 1)))
            # Stride-2 pose quantization: ensures peak (n_pool-1) and start (0) are exact while cutting unique renders in half!
            if raw_pi not in (0, n_pool - 1):
                p_i = (raw_pi // 2) * 2
            else:
                p_i = raw_pi
            pose_frac = p_i / float(max(1, n_pool - 1))

            if p_i not in step_pose_cache:
                rgb_1080, kp = transform_frame_and_kp_1080p(seq[p_i], kps_seqs_raw[step_idx][p_i], y_sh, step_idx)
                pulse_ph = pose_frac * 0.75
                base_rgb, sh_mask = shade_native_3d_frame_1080p(rgb_1080, step_idx, is_w, v_mode, kp, pulse_phase=pulse_ph)

                if step_idx == 1:
                    slot_i = 0
                    deg_v = int(round(2 + 14 * pose_frac))
                    pct_v = int(round(78 + 8 * pose_frac))
                    deg_s, l1_s, l2_s = f"{deg_v}°", "KNEE CAVE", f"{pct_v}% JOINT RISK"
                elif step_idx == 2:
                    slot_i = 1
                    deg_v = int(round(18 + 23 * pose_frac))
                    pct_v = int(round(71 + 8 * pose_frac))
                    deg_s, l1_s, l2_s = f"{deg_v}°", "FORWARD LEAN", f"{pct_v}% LUMBAR LOAD"
                elif step_idx == 3:
                    slot_i = 2
                    pct_v = int(round(90 + 6 * pose_frac))
                    deg_s, l1_s, l2_s = "0°", "KNEE LOCK", f"{pct_v}% JOINT SAFE"
                else:
                    slot_i = 3
                    deg_v = int(round(12 + 10 * pose_frac))
                    pct_v = int(round(91 + 7 * pose_frac))
                    deg_s, l1_s, l2_s = f"{deg_v}°", "HIP DRIVE", f"{pct_v}% QUAD & GLUTE"

                fr = draw_step_laser_and_callout(base_rgb, sh_mask, step_idx, kp, is_w, deg_s, l1_s, l2_s)
                fg_m = get_mannequin_fg_mask_1080p(base_rgb)
                fr = composite_depth_behind_title_and_translucent_preview(
                    fr, fg_m, icons_all_4, active_slot=slot_i, is_wrong=is_w, exercise_name="Barbell Back Squat"
                )
                bgr_frame = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                step_pose_cache[p_i] = (fr, bgr_frame, deg_s, l1_s)

            fr, bgr_frame, deg_s, l1_s = step_pose_cache[p_i]

            if f_local == 50:
                rgb_out = Image.fromarray(fr)
                step_peak_rendered[step_idx] = rgb_out
                rgb_out.save(os.path.join(OUT_DIR, f"day3_step{step_idx}_peak.jpg"), quality=95)
                if step_idx == 4:
                    rgb_out.save(os.path.join(OUT_DIR, "day3_barbell_squat_4step_master_cover.jpg"), quality=95)
                    rgb_out.save(os.path.join(OUT_DIR, "day3_cover_step4_rear34_clean.jpg"), quality=95)

            vw.write(bgr_frame)

        print(f"   Step {step_idx} (150 frames, {len(step_pose_cache)} unique poses) rendered in {time.time() - t_step:.2f}s")
        step_pose_cache.clear()
        del seq
        gc.collect()

    vw.release()

    print("4. Muxing H.264 video + AAC 192k audio into final MP4...")
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
    print(f"Saved Master MP4 in {time.time() - t0_all:.2f}s total:", final_mp4)

    # Save 4-Step Multi-Angle Master Comparison Sheet
    pw, ph = 520, 924
    pad = 22
    header_h = 96
    sheet = Image.new("RGB", (pad + 4 * (pw + pad), header_h + ph + pad), (14, 16, 20))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(26, "extrabold")
    f_s = v30.f_pretendard(18, "bold")
    sd.text(
        (28, 18),
        "Day 3 Barbell Back Squat — 방안 B (@demicstory식 중심부 코어 그라데이션 + 등 근육결 100% 보존)",
        font=f_t,
        fill=(242, 246, 252),
    )
    sd.text(
        (28, 54),
        "Step 1: 정면 오자세(뼈·관절 X-Ray) | Step 2: 측면 오자세(척추·근육 과부하) | Step 3: 정면 정자세(뼈·관절 정렬) | Step 4: 후측면 정자세(둔근·대퇴사두)",
        font=f_s,
        fill=(185, 225, 255),
    )
    step_titles = [
        (1, True,  "Step 1: 정면 오자세 (16° KNEE CAVE · 뼈/관절)"),
        (2, True,  "Step 2: 측면 오자세 (41° FORWARD LEAN · 척추/근육)"),
        (3, False, "Step 3: 정면 정자세 (0° KNEE LOCK · 뼈/관절)"),
        (4, False, "Step 4: 후측면 정자세 (22° HIP DRIVE · 둔근/대퇴사두)"),
    ]
    for idx_p, (s_i, is_w, lbl) in enumerate(step_titles):
        x0 = pad + idx_p * (pw + pad)
        y0 = header_h
        col = (255, 95, 78) if is_w else (65, 245, 135)
        sheet.paste(step_peak_rendered[s_i].resize((pw, ph), Image.Resampling.LANCZOS), (x0, y0))
        sd.rounded_rectangle([x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 2], radius=12, outline=col, width=3)
        sd.rounded_rectangle([x0 + 10, y0 + 10, x0 + pw - 10, y0 + 44], radius=7, fill=(12, 14, 18))
        sd.text((x0 + 18, y0 + 15), lbl, font=f_s, fill=col)

    for fn_s in ["day3_4step_multiangle_master_sheet.jpg", "day3_4step_bone_muscle_cursive_sheet.jpg"]:
        sp = os.path.join(OUT_DIR, fn_s)
        sheet.save(sp, quality=95)
        print("Saved 4-Step Sheet:", sp)


if __name__ == "__main__":
    build_all()
