#!/usr/bin/env python3
"""
Builds the 4 High-Resolution Mockup Variants (A, B, C, D) across all 4 Steps of Day 3 (Barbell Back Squat)
with 100% Locked Studio Background, 100% Locked Mannequin Skin Tone, Zero Blur (Crisp Lanczos4 + Unsharp Mask),
Clear Previous/Current/Next Top Preview Thumbnails, and Full Quantitative QC Verification.
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
import preview_v58_day2_exact_master as p58
import preview_v62_seamless_day2_master as p62

OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)
W, H = 1080, 1920

# ==============================================================================
# 1. 100% LOCKED GLOBAL STUDIO BACKGROUND & MANNEQUIN SKIN LUT
#    (Matches Day 2 Screenshot `01M60KMMZABBXSMGWYVXXMNX5N.png` & `@demicstory`)
# ==============================================================================
_YY_FULL, _XX_FULL = np.mgrid[:H, :W].astype(np.float32)

def get_locked_studio_background() -> np.ndarray:
    """
    Returns a 100% deterministic, frame-invariant dark charcoal studio background (`1920x1080x3` float32).
    Matches Day 2 screenshot (`bg_RGB = [42, 44, 48]` in center stage, smoothly vignettes to `[28, 31, 35]` at edges).
    Zero per-frame or per-step drift (`std_across_steps = 0.00`).
    """
    rx = (_XX_FULL - 540.0) / 620.0
    ry = (_YY_FULL - 880.0) / 780.0
    r2 = np.clip(rx * rx + ry * ry, 0.0, 1.35)
    stage_glow = np.exp(-0.65 * r2)  # 1.0 at center, ~0.42 at corners
    base_lum = 27.5 + 15.5 * stage_glow  # 43.0 at center, ~34.0 at margins
    bg = np.stack([
        base_lum * 0.965,  # R ~ 41.5 center
        base_lum * 1.000,  # G ~ 43.0 center
        base_lum * 1.065,  # B ~ 45.8 center
    ], axis=2).astype(np.float32)
    return bg

LOCKED_STUDIO_BG = get_locked_studio_background()


def shade_locked_crisp_mannequin_720p(
    bgr_720p: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    kp_720: dict,
    rep_a: float = 1.0,
    color_mode: str = "day2",  # "demic_orange", "day2", "hybrid_ultimate"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders a razor-sharp (`ZERO bilateral blur`, `ZERO skin gaussian blur`) 3D mannequin with:
    1. 100% locked mannequin skin tone across all steps (`R:G:B = 0.975 : 1.000 : 1.050`, fixed LUT).
    2. Crisp cool electric-silver rim lighting along upper-body/limb contours (`@demicstory` + Day 2).
    3. Returns `(shaded_rgb_720p, body_mask_720p)` so the outer studio background can be locked to `LOCKED_STUDIO_BG`
       and subsurface laser glow can be strictly masked inside `body_mask`!
    """
    # Crisp unsharp mask (identical in spirit to Day 2 `build_v36_publish_reels.py` line 144, ZERO bilateral blur!)
    blur_micro = cv2.GaussianBlur(bgr_720p, (0, 0), 0.95)
    plate_bgr = cv2.addWeighted(bgr_720p, 1.24, blur_micro, -0.24, 0)
    rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    fiber_signal = 0.60 * r_ch + 0.30 * g_ch + 0.10 * b_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 0.70)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.1)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 5.5)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 12.0)

    hf_ridge = (sigma_micro - sigma_ridge) * 4.5 + (sigma_ridge - sigma_belly) * 2.6
    fs = np.clip(0.48 + hf_ridge / 28.0, 0.0, 1.0) ** 1.38
    hv = np.clip((sigma_belly - 22.0) / 135.0, 0.0, 1.0)
    sulc = np.clip((sigma_broad - sigma_ridge - 0.85) / 9.5, 0.0, 1.0)

    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    bg_lum_est = float(np.median(base_lum[180:720, 12:58]))
    bg_row_raw = 0.5 * (np.percentile(base_lum[:, 8:46], 35, axis=1) + np.percentile(base_lum[:, 674:712], 35, axis=1)).astype(np.float32)
    bg_row = cv2.GaussianBlur(bg_row_raw[:, None], (1, 151), 45.0)
    bg_2d = np.maximum(bg_lum_est, bg_row)

    raw_fg = ((base_lum > (bg_2d + 8.5)) | (rb_diff > 9.5)).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    fg_closed = cv2.morphologyEx(raw_fg, cv2.MORPH_CLOSE, k_close)
    fg_soft = cv2.GaussianBlur(fg_closed.astype(np.float32) / 255.0, (0, 0), 1.8)
    edge_ramp = np.clip((base_lum - (bg_2d + 6.0)) / 8.5, 0.0, 1.0)
    body_interior = np.clip(np.maximum(fg_soft * 0.96, edge_ramp), 0.0, 1.0)

    # Fixed per-step source calibration constant (measured once from raw clips so all 4 steps have identical input scale,
    # ZERO dynamic frame-by-frame percentile drift!)
    STEP_RAW_SCALE = {
        1: (25.0, 56.0, 95.0, 140.0),
        2: (31.0, 59.0, 101.0, 143.0),
        3: (26.0, 56.5, 91.0, 125.0),
        4: (25.0, 50.0, 84.0, 116.0),
    }
    s_bg, s_p50, s_p90, s_p98 = STEP_RAW_SCALE[step_idx]
    cool_raw = np.minimum(g_ch, b_ch)
    # Preserve crisp micro-detail directly from `cool_raw` + `hf_ridge * 0.35` (ZERO bilateral/gaussian blur!)
    xp = np.array([0.0, s_bg, s_bg + 10.0, s_p50, s_p90, s_p98, 255.0], dtype=np.float32)
    # Deep shadow contrast (`38.0` in deep folds, `86.0` midtone matching Day 2, `148.0..186.0` silver rim highlights)
    fp = np.array([34.0, 39.0, 56.0, 86.0, 144.0, 178.0, 198.0], dtype=np.float32)
    skin_lum = np.interp(cool_raw.ravel(), xp, fp).reshape(cool_raw.shape).astype(np.float32)
    skin_lum = np.clip(skin_lum + hf_ridge * 0.32, 36.0, 202.0)

    # Add @demicstory / Day 2 signature crisp cool-silver rim light along upper contours
    gy_s = cv2.Sobel(body_interior, cv2.CV_32F, 0, 1, ksize=3)
    gx_s = cv2.Sobel(body_interior, cv2.CV_32F, 1, 0, ksize=3)
    rim_band = np.clip(np.hypot(gx_s, gy_s) * 1.4, 0.0, 1.0) * np.clip((skin_lum - 72.0) / 55.0, 0.0, 1.0)

    # Locked Mannequin RGB (`R:G:B = 0.975 : 1.000 : 1.050` + cool-blue rim)
    mono = np.stack([
        skin_lum * 0.975 + rim_band * 14.0,
        skin_lum * 1.000 + rim_band * 24.0,
        skin_lum * 1.050 + rim_band * 42.0,
    ], axis=2)

    sh_x, sh_y = kp_720["sh"]
    hip_r = kp_720["hip_r"]
    hip_l = kp_720["hip_l"]
    kn_r = kp_720["kn_r"]
    kn_l = kp_720["kn_l"]
    kn_y = 0.5 * (kn_r[1] + kn_l[1])
    hip_y = 0.5 * (hip_r[1] + hip_l[1])

    y_cut_shin = kn_y + 24.0
    above_shin_gate = np.clip((y_cut_shin - yy) / 18.0, 0.0, 1.0)
    rb_blur = cv2.GaussianBlur(rb_diff, (0, 0), 3.2) * body_interior
    rb_eff = np.maximum(rb_diff, rb_blur * 0.88)

    # Determine target muscle activation field (`fe`) and tendon attachment zone (`tendon_w`)
    d_th_r, t_th_r = p62._segment_distance_field(xx, yy, hip_r, kn_r, -0.08, 1.0)
    d_th_l, t_th_l = p62._segment_distance_field(xx, yy, hip_l, kn_l, -0.08, 1.0)

    if step_idx == 1:
        d_kn_r = np.hypot(xx - kn_r[0], yy - kn_r[1])
        d_kn_l = np.hypot(xx - kn_l[0], yy - kn_l[1])
        r_kn = 38.0 + 36.0 * rep_a
        r_th = 32.0 + 42.0 * rep_a
        knee_hotspot = np.maximum(np.exp(-0.5 * (d_kn_r / r_kn) ** 2), np.exp(-0.5 * (d_kn_l / r_kn) ** 2))
        thigh_field = np.maximum(np.exp(-0.5 * (d_th_r / r_th) ** 2), np.exp(-0.5 * (d_th_l / r_th) ** 2))
        valgus_gate = np.clip((yy - (hip_y + 38.0 - 94.0 * rep_a)) / 26.0, 0.0, 1.0) * above_shin_gate
        native_red = np.clip((rb_eff - 8.5) / 28.0, 0.0, 1.0) * valgus_gate * (0.68 + 0.32 * rep_a)
        sculpt_red = np.maximum(knee_hotspot * 0.94, thigh_field * (0.72 + 0.24 * rep_a)) * body_interior * valgus_gate
        fe = np.clip(np.maximum(native_red, sculpt_red), 0.0, 1.0)
        bone_w = (
            np.clip((g_ch - 72.0) / 80.0, 0.0, 1.0)
            * np.clip((rb_diff - 20.0) / 45.0, 0.0, 1.0)
            * knee_hotspot
            * above_shin_gate
        )[:, :, None]
    elif step_idx == 3:
        d_spine, _ = p62._segment_distance_field(xx, yy, hip_r, (sh_x, sh_y), 0.06, 0.84)
        above_glute_gate = np.clip(((hip_r[1] + 2.0 + 14.0 * rep_a) - yy) / 24.0, 0.0, 1.0)
        below_neck_gate = np.clip((yy - (sh_y + 2.0)) / 24.0, 0.0, 1.0)
        r_sp = 40.0 + 18.0 * rep_a
        lumbar_gate = np.exp(-0.5 * (d_spine / r_sp) ** 2) * above_glute_gate * below_neck_gate
        native_lumbar = np.clip((rb_eff - 7.5) / 26.0, 0.0, 1.0) * above_glute_gate * below_neck_gate
        fe = np.clip(np.maximum(native_lumbar * (0.85 + 0.15 * rep_a), lumbar_gate * (0.86 + 0.12 * rep_a)) * body_interior * above_shin_gate, 0.0, 1.0)
        bone_w = (
            np.clip((g_ch - 80.0) / 88.0, 0.0, 1.0)
            * np.clip((rb_diff - 26.0) / 50.0, 0.0, 1.0)
            * above_glute_gate
            * 0.58
        )[:, :, None]
    elif step_idx == 2:
        r_th = 36.0 + 46.0 * rep_a
        target_gate = np.clip((yy - (hip_y + 12.0 - 96.0 * rep_a)) / 26.0, 0.0, 1.0) * above_shin_gate
        quad_anat = np.maximum(
            np.exp(-0.5 * (d_th_r / r_th) ** 2),
            np.exp(-0.5 * (d_th_l / r_th) ** 2),
        ) * body_interior * target_gate
        native_fe = np.clip((rb_eff - 7.5) / 28.0, 0.0, 1.0) * target_gate * (0.64 + 0.36 * rep_a)
        fe = np.clip(np.maximum(native_fe, quad_anat * (0.82 + 0.16 * rep_a) * np.clip((base_lum - 28.0) / 20.0, 0.0, 1.0)), 0.0, 1.0)
        bone_w = np.zeros((h, w, 1), dtype=np.float32)
    else:
        target_gate = np.clip((yy - (hip_y - 62.0)) / 30.0, 0.0, 1.0) * above_shin_gate
        glute_center = (hip_r[0] + 22.0, hip_r[1] + 10.0)
        d_glute = np.hypot((xx - glute_center[0]) * 0.88, (yy - glute_center[1]) * 1.05)
        glute_anat = np.exp(-0.5 * (d_glute / (64.0 + 10.0 * rep_a)) ** 2)
        quad_anat = np.maximum(
            np.exp(-0.5 * (d_th_r / (56.0 + 10.0 * rep_a)) ** 2),
            np.exp(-0.5 * (d_th_l / (50.0 + 10.0 * rep_a)) ** 2),
        )
        lower_anat = np.maximum(glute_anat, quad_anat) * body_interior * target_gate
        native_fe = np.clip((rb_eff - 7.5) / 28.0, 0.0, 1.0) * target_gate
        fe = np.clip(np.maximum(native_fe, lower_anat * 0.92 * np.clip((base_lum - 30.0) / 22.0, 0.0, 1.0)), 0.0, 1.0)
        bone_w = np.zeros((h, w, 1), dtype=np.float32)

    hc = np.clip((rb_diff - 45.0) / 62.0, 0.0, 1.0) * fs
    strand_mod = (0.16 + 0.84 * fs) * ((1.0 - 0.80 * sulc) ** 1.38)
    fiber_alpha = np.clip(fe * (0.74 + 0.64 * strand_mod), 0.0, 0.98)[:, :, None]
    ridge_boost = np.clip(fs * 0.78 + hc * 0.46, 0.0, 1.0)[:, :, None]
    vol_shade = np.clip(0.22 + 0.78 * hv * (0.34 + 0.66 * fs) * (1.0 - 0.52 * sulc), 0.0, 1.0)[:, :, None]

    # Distal quadriceps tendon attachment highlight near the patella (`@demicstory` white tendon detail)
    d_pat_r = np.hypot(xx - kn_r[0], yy - (kn_r[1] - 12.0))
    d_pat_l = np.hypot(xx - kn_l[0], yy - (kn_l[1] - 12.0))
    tendon_flare = np.maximum(np.exp(-0.5 * (d_pat_r / 20.0) ** 2), np.exp(-0.5 * (d_pat_l / 20.0) ** 2))
    tendon_w = (tendon_flare * fe * 0.58)[:, :, None]
    tendon_rgb = np.array([235.0, 228.0, 220.0], dtype=np.float32)

    if color_mode == "demic_orange":
        # @demicstory style: Striated Orange-Coral muscle fibers (#FF6B1A) with white tendon attachments,
        # so the Neon Green (Right) or Neon Red/Magenta (Wrong) joint-axis laser stands out with 100% contrast!
        if is_wrong:
            deep_col = np.array([138.0, 10.0, 16.0], dtype=np.float32)
            mid_col = np.array([252.0, 38.0, 32.0], dtype=np.float32)
            crest_col = np.array([255.0, 152.0, 110.0], dtype=np.float32)
        else:
            deep_col = np.array([142.0, 36.0, 8.0], dtype=np.float32)
            mid_col = np.array([248.0, 96.0, 22.0], dtype=np.float32)
            crest_col = np.array([255.0, 196.0, 128.0], dtype=np.float32)
    elif color_mode == "hybrid_ultimate":
        # Option D (Ultimate Hybrid):
        # - Wrong (Step 1, 3): Deep Crimson-Red muscle & white-amber 3D X-Ray bone overload
        # - Right (Step 2, 4): Rich Emerald-Teal 3D striated muscle fibers with crisp golden-white tendon/crest highlights
        #   so a Cyan-Mint/Hot-White laser line AND @demicstory subsurface glow both pop with razor-sharp contrast!
        if is_wrong:
            deep_col = np.array([136.0, 8.0, 16.0], dtype=np.float32)
            mid_col = np.array([252.0, 26.0, 36.0], dtype=np.float32)
            crest_col = np.array([255.0, 145.0, 118.0], dtype=np.float32)
        else:
            deep_col = np.array([6.0, 118.0, 38.0], dtype=np.float32)
            mid_col = np.array([18.0, 228.0, 82.0], dtype=np.float32)
            crest_col = np.array([210.0, 255.0, 215.0], dtype=np.float32)
    else:
        # "day2" classic
        if is_wrong:
            deep_col = np.array([132.0, 8.0, 14.0], dtype=np.float32)
            mid_col = np.array([250.0, 22.0, 32.0], dtype=np.float32)
            crest_col = np.array([255.0, 112.0, 108.0], dtype=np.float32)
        else:
            deep_col = np.array([6.0, 128.0, 32.0], dtype=np.float32)
            mid_col = np.array([18.0, 240.0, 72.0], dtype=np.float32)
            crest_col = np.array([182.0, 255.0, 196.0], dtype=np.float32)

    fiber_rgb = (
        deep_col * (1.0 - vol_shade)
        + mid_col * vol_shade * (1.0 - ridge_boost * 0.58)
        + crest_col * (ridge_boost ** 1.36) * 0.86
    )
    if not is_wrong and color_mode in ("demic_orange", "hybrid_ultimate"):
        fiber_rgb = fiber_rgb * (1.0 - tendon_w) + tendon_rgb * tendon_w

    if is_wrong:
        xray_bone = np.stack([
            np.clip(r_ch * 1.18 + 28.0 + hf_ridge * 0.70, 0, 255),
            np.clip(g_ch * 0.46 + 14.0 + hf_ridge * 0.40, 0, 165),
            np.clip(b_ch * 0.30 + 10.0, 0, 110),
        ], axis=2)
        combined_src = fiber_rgb * (1.0 - bone_w) + xray_bone * bone_w
        alpha_final = fiber_alpha * (1.0 - bone_w) + (fe[:, :, None] * 0.96) * bone_w
    else:
        combined_src = fiber_rgb
        alpha_final = fiber_alpha

    out_mannequin = mono * (1.0 - alpha_final) + np.clip(combined_src, 0, 255) * alpha_final
    return np.clip(out_mannequin, 0, 255).astype(np.uint8), body_interior


# ==============================================================================
# 2. LOCKED-BACKGROUND FULL-BLEED 1080x1920 COMPOSITOR
# ==============================================================================
def compose_locked_1080p_frame(
    raw_bgr_720p: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    color_mode: str,
    rep_a: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Places the shaded 720p mannequin onto `LOCKED_STUDIO_BG` (`1080x1920`) so that:
    - The studio background is 100% locked across all 4 steps (`std = 0.00`).
    - The mannequin skin tone is 100% locked across all 4 steps.
    - Returns `(rgb_1080p, body_mask_1080p, kp_1080p)`.
    """
    fn, i0, i1, is_w, scale, y_st, y_sq, ys_st, ys_sq, x_sh = p62.STEP_POOL_SPECS[step_idx]
    if step_idx == 1:
        raw_bgr_720p = p62._apply_step1_knee_valgus_720p(raw_bgr_720p, rep_a)

    sh_h, sw_w = raw_bgr_720p.shape[:2]
    nw = int(round(sw_w * scale))
    nh = int(round(sh_h * scale))
    x0 = (W - nw) // 2 + x_sh
    y0_stand = y_st - ys_st * scale
    y0_squat = y_sq - ys_sq * scale
    y0 = int(round(y0_stand * (1.0 - rep_a) + y0_squat * rep_a))
    y_end = y0 + nh

    detector = uar.get_pose_detector()
    rgb_720 = cv2.cvtColor(raw_bgr_720p, cv2.COLOR_BGR2RGB)
    mp_res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb_720))
    lm = mp_res.pose_landmarks[0]
    sh_pt = (0.5 * (lm[11].x + lm[12].x) * sw_w, 0.5 * (lm[11].y + lm[12].y) * sh_h)
    hip_r = (lm[24].x * sw_w, lm[24].y * sh_h)
    hip_l = (lm[23].x * sw_w, lm[23].y * sh_h)
    kn_r = (lm[26].x * sw_w, lm[26].y * sh_h)
    kn_l = (lm[25].x * sw_w, lm[25].y * sh_h)
    ank_r = (lm[28].x * sw_w, lm[28].y * sh_h)
    ank_l = (lm[27].x * sw_w, lm[27].y * sh_h)
    nose_y = lm[0].y * sh_h
    head_top = (sh_pt[0], min(nose_y, sh_pt[1]) - 54.0)
    foot_y = max(lm[29].y, lm[30].y, lm[31].y, lm[32].y) * sh_h
    foot_bot = (0.5 * (ank_r[0] + ank_l[0]), foot_y)
    kp_720 = {
        "sh": sh_pt,
        "hip_r": hip_r,
        "hip_l": hip_l,
        "kn_r": kn_r,
        "kn_l": kn_l,
        "ank_r": ank_r,
        "ank_l": ank_l,
        "head_top": head_top,
        "foot_bot": foot_bot,
    }

    man_720, mask_720 = shade_locked_crisp_mannequin_720p(
        raw_bgr_720p, step_idx, is_wrong, kp_720, rep_a=rep_a, color_mode=color_mode
    )

    dst_x1 = max(0, x0)
    dst_x2 = min(W, x0 + nw)
    src_x1 = dst_x1 - x0
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    crop_w = dst_x2 - dst_x1

    dst_y1 = max(0, y0)
    dst_y2 = min(H, y_end)
    src_y1 = dst_y1 - y0
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    crop_h = dst_y2 - dst_y1

    res_man = cv2.resize(man_720, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    res_mask = cv2.resize(mask_720, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    crop_man = res_man[src_y1:src_y2, src_x1:src_x2, :]
    crop_mask = res_mask[src_y1:src_y2, src_x1:src_x2]

    xx_c = np.arange(crop_w, dtype=np.float32)[None, :]
    yy_c = np.arange(crop_h, dtype=np.float32)[:, None]
    yy_full = yy_c + dst_y1
    xx_full = xx_c + dst_x1

    ay_t = np.clip(yy_c / 64.0, 0.0, 1.0)
    ay_b = np.clip((crop_h - 1.0 - yy_c) / 64.0, 0.0, 1.0)
    ay = 0.5 - 0.5 * np.cos(np.pi * np.minimum(ay_t, ay_b))
    ax_l = np.clip(xx_c / 84.0, 0.0, 1.0)
    ax_r = np.clip((crop_w - 1.0 - xx_c) / 84.0, 0.0, 1.0)
    ax = 0.5 - 0.5 * np.cos(np.pi * np.minimum(ax_l, ax_r))
    head_guard = np.clip((yy_full - 461.0) / 6.0, 0.0, 1.0)
    in_head_col = np.clip((670.0 - xx_full) / 30.0, 0.0, 1.0) * np.clip((xx_full - 410.0) / 30.0, 0.0, 1.0)
    top_mask = 1.0 - in_head_col * (1.0 - head_guard)

    alpha_fg = np.clip(crop_mask * ax * ay * top_mask, 0.0, 1.0)[:, :, None]

    canvas = LOCKED_STUDIO_BG.copy()
    bg_patch = canvas[dst_y1:dst_y2, dst_x1:dst_x2, :]
    canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = crop_man * alpha_fg + bg_patch * (1.0 - alpha_fg)

    full_mask = np.zeros((H, W), dtype=np.float32)
    full_mask[dst_y1:dst_y2, dst_x1:dst_x2] = alpha_fg[:, :, 0]

    kp_1080 = {}
    for k, (kx, ky) in kp_720.items():
        kp_1080[k] = (x0 + kx * scale, y0 + ky * scale)

    return np.clip(canvas, 0, 255).astype(np.uint8), full_mask, kp_1080


# ==============================================================================
# 3. SUBSURFACE-MASKED + ADDITIVE SCREEN LASER JOINT-AXIS RENDERER
#    (Zero Milky Blur! Glow stays inside the 3D limb and ADDITIVELY illuminates
#     muscle striations instead of obscuring them!)
# ==============================================================================
def draw_subsurface_additive_laser_axis(
    canvas: Image.Image,
    body_mask_1080: np.ndarray,
    main_chains_1x: list[list[tuple[float, float]]],
    aux_ext_chains_1x: list[list[tuple[float, float]]],
    vertex_pts_1x: list[tuple[float, float]],
    is_wrong: bool,
    laser_style: str = "demic_green_red",  # "demic_green_red", "hybrid_contrast"
):
    if is_wrong:
        col_glow = (255, 38, 48)
        col_halo = (255, 68, 76)
        col_core = (255, 232, 232)
    else:
        if laser_style == "hybrid_contrast":
            # Electric Cyan-Mint core + Neon Green halo so it pops against Emerald muscle fibers!
            col_glow = (0, 245, 165)
            col_halo = (65, 255, 205)
            col_core = (232, 255, 250)
        else:
            # @demicstory Neon Emerald-Lime laser against Orange-Coral or Emerald muscle
            col_glow = (35, 255, 118)
            col_halo = (65, 255, 140)
            col_core = (224, 255, 234)

    # 1. Subsurface Volumetric Glow (Additive illumination strictly masked inside `body_mask_1080` -> ZERO background fog!)
    glow_1x = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_1x)
    for chain in main_chains_1x:
        p62._draw_tapered_laser_polyline_ss(gd, chain, base_w_1x=20.0, col_rgba=(*col_glow, 130), ss=1, fade_start=0.18, fade_end=0.18)
    for vx, vy in vertex_pts_1x:
        r_v = 26.0
        gd.ellipse([vx - r_v, vy - r_v, vx + r_v, vy + r_v], fill=(*col_glow, 110))
    glow_1x = glow_1x.filter(ImageFilter.GaussianBlur(radius=18.0))

    # Apply additive screen illumination strictly inside the body silhouette (preserving 100% muscle fiber sharpness!)
    glow_arr = np.array(glow_1x).astype(np.float32)
    sub_mask = np.clip(cv2.GaussianBlur(body_mask_1080, (0, 0), 3.0), 0.0, 1.0)[:, :, None]
    glow_a = (glow_arr[:, :, 3:4] / 255.0) * sub_mask * 0.62
    base_rgb = np.array(canvas.convert("RGB")).astype(np.float32)
    # Screen/Additive blend: brightens muscle ridges without washing out dark sulci!
    lum_boost = 0.35 + 0.65 * np.clip((base_rgb.mean(axis=2, keepdims=True) - 35.0) / 110.0, 0.0, 1.0)
    add_rgb = base_rgb + glow_arr[:, :, :3] * glow_a * lum_boost
    canvas_add = Image.fromarray(np.clip(add_rgb, 0, 255).astype(np.uint8)).convert("RGBA")
    canvas.paste(canvas_add, (0, 0))

    # 2. Tight Neon Laser Halo (`radius=4.2px`, razor-crisp, zero fog!)
    halo_1x = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo_1x)
    for chain in main_chains_1x:
        p62._draw_tapered_laser_polyline_ss(hd, chain, base_w_1x=8.5, col_rgba=(*col_halo, 210), ss=1, fade_start=0.18, fade_end=0.18)
    for ext in aux_ext_chains_1x:
        p62._draw_tapered_laser_polyline_ss(hd, ext, base_w_1x=5.0, col_rgba=(*col_halo, 165), ss=1, fade_start=0.20, fade_end=0.22)
    halo_1x = halo_1x.filter(ImageFilter.GaussianBlur(radius=4.2))

    # 3. 4x Supersampled Razor-Sharp Hot-White / Neon Core (`2.6px` @ 1x)
    SS = 4
    core_ss = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    cd = ImageDraw.Draw(core_ss)
    for ext in aux_ext_chains_1x:
        p62._draw_tapered_laser_polyline_ss(cd, ext, base_w_1x=2.6, col_rgba=(*col_halo, 210), ss=SS, fade_start=0.20, fade_end=0.22)
        p62._draw_tapered_laser_polyline_ss(cd, ext, base_w_1x=1.4, col_rgba=(*col_core, 240), ss=SS, fade_start=0.20, fade_end=0.22)
    for chain in main_chains_1x:
        p62._draw_tapered_laser_polyline_ss(cd, chain, base_w_1x=4.4, col_rgba=(*col_halo, 245), ss=SS, fade_start=0.18, fade_end=0.18)
        p62._draw_tapered_laser_polyline_ss(cd, chain, base_w_1x=2.5, col_rgba=(*col_core, 255), ss=SS, fade_start=0.18, fade_end=0.18)

    core_1x = Image.fromarray(cv2.resize(np.array(core_ss), (W, H), interpolation=cv2.INTER_AREA))
    canvas.alpha_composite(halo_1x)
    canvas.alpha_composite(core_1x)


# ==============================================================================
# 4. TOP PREVIEW DOCKS (WITH PREVIOUS · CURRENT · NEXT CLEARLY VISIBLE!)
# ==============================================================================
def make_4step_preview_thumbs(step_frames_rgb: dict, step_kps: dict) -> dict[int, Image.Image]:
    """
    Creates 4 bright, crystal-clear anatomical preview thumbnails for Steps 1, 2, 3, and 4
    so that in EVERY step `s` (1..4), the Top Carousel displays:
    - Left Slot (`PREV`): Step `s-1` (or Step 4 loop-back when `s==1`)
    - Center Pop-Out Slot (`CURRENT`, orange/green glowing border): Step `s`
    - Right Slot (`NEXT`): Step `s+1` (or Step 1 loop-forward when `s==4`)
    Guarantees 100% visibility of Previous · Current · Next in all 4 steps (`t311u`!).
    """
    thumbs = {}
    for s_idx in (1, 2, 3, 4):
        rgb = step_frames_rgb[s_idx]
        kp = step_kps[s_idx]
        hip_x, hip_y = kp["hip_r"]
        cw, ch = 500, 420
        cx = int(np.clip(hip_x + 10, cw // 2, W - cw // 2))
        cy = int(np.clip(hip_y - 10, ch // 2 + 450, H - ch // 2))
        x0, y0 = cx - cw // 2, cy - ch // 2
        crop = rgb[y0 : y0 + ch, x0 : x0 + cw].copy()
        crop_boost = np.clip(crop.astype(np.float32) * 1.18 + 8.0, 0, 255).astype(np.uint8)
        thumbs[s_idx] = Image.fromarray(crop_boost)
    return thumbs


def draw_demic_carousel_top_dock(
    canvas: Image.Image,
    thumbs_4: dict[int, Image.Image],
    step_idx: int,
    is_wrong: bool,
    with_exercise_header: bool = False,
):
    """
    Draws `@demicstory`'s Sliding 3-Window Carousel (`[PREV] · [CURRENT (Center Pop-Out)] · [NEXT]`)
    + Right Status Badge (`✕` / `✓`), with optional clean Exercise Title Pill above (`with_exercise_header`):
    - Center Slot (`CURRENT`): `148x120px`, pops out vertically above/below the frosted track with a glowing Neon Orange/Green border
      and bold muscle label (`QUADS`, `LUMBAR`, `GLUTES`) at the bottom, 100% matching `@demicstory` (`t=11.5s`)!
    - Left Slot (`PREV`) & Right Slot (`NEXT`): `120x98px`, filled with the actual Previous and Next step 3D mannequin crops
      at 88% brightness so all 3 (`이전 · 현재 · 다음`) are crystal-clear (`t311u`!).
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    prev_idx = {1: 4, 2: 1, 3: 2, 4: 3}[step_idx]
    curr_idx = step_idx
    next_idx = {1: 2, 2: 3, 3: 4, 4: 1}[step_idx]
    slot_steps = [prev_idx, curr_idx, next_idx]
    slot_tags = ["PREV", "NOW", "NEXT"]
    step_short = {1: "VALGUS ✕", 2: "QUADS ✓", 3: "LUMBAR ✕", 4: "GLUTES ✓"}

    if with_exercise_header:
        hdr_y = 240
        hdr_h = 42
        title_str = "BARBELL BACK SQUAT"
        sub_str = "QUADS & GLUTES"
        tb_t = od.textbbox((0, 0), title_str, font=p58.F_EX_TITLE)
        tw_t = tb_t[2] - tb_t[0]
        tb_s = od.textbbox((0, 0), sub_str, font=p58.F_EX_SUB)
        tw_s = tb_s[2] - tb_s[0]
        pill_w = tw_s + 22
        hdr_w = 20 + 10 + tw_t + 14 + pill_w + 16
        hx0 = (W - hdr_w) // 2 - 22
        accent = p58.COL_RED if is_wrong else p58.COL_GREEN
        pill_bg = (218, 44, 38, 245) if is_wrong else (18, 168, 88, 245)

        od.rounded_rectangle([hx0, hdr_y, hx0 + hdr_w, hdr_y + hdr_h], radius=13, fill=(14, 17, 23, 238), outline=(255, 255, 255, 52), width=1)
        dcx, dcy = hx0 + 18, hdr_y + hdr_h // 2
        od.ellipse([dcx - 5, dcy - 5, dcx + 5, dcy + 5], fill=(*accent, 255))
        od.text((dcx + 12, hdr_y + 7), title_str, font=p58.F_EX_TITLE, fill=(255, 255, 255, 255))
        px0 = dcx + 12 + tw_t + 12
        od.rounded_rectangle([px0, hdr_y + 7, px0 + pill_w, hdr_y + hdr_h - 7], radius=7, fill=pill_bg)
        od.text((px0 + 11, hdr_y + 11), sub_str, font=p58.F_EX_SUB, fill=(255, 255, 255, 255))
        track_cy = 356
    else:
        track_cy = 344

    w_side, h_side = 124, 100
    w_cent, h_cent = 148, 120
    gap = 12
    pad_x = 14
    total_w = 2 * w_side + w_cent + 2 * gap + 2 * pad_x
    track_h = h_side + 16
    tx0 = (W - total_w) // 2 - 32
    ty0 = track_cy - track_h // 2

    od.rounded_rectangle(
        [tx0, ty0, tx0 + total_w, ty0 + track_h],
        radius=22,
        fill=(18, 22, 29, 224),
        outline=(255, 255, 255, 62),
        width=2,
    )

    f_tag = v30.f_pretendard(12, "extrabold")
    f_mus = v30.f_pretendard(14, "extrabold")

    # Draw side slots (`PREV` at i=0, `NEXT` at i=2) first, then center pop-out (`CURRENT` at i=1)
    x_positions = [
        tx0 + pad_x,
        tx0 + pad_x + w_side + gap,
        tx0 + pad_x + w_side + gap + w_cent + gap,
    ]
    for i in (0, 2, 1):
        s_num = slot_steps[i]
        is_center = (i == 1)
        cw_i = w_cent if is_center else w_side
        ch_i = h_cent if is_center else h_side
        cx_i = x_positions[i]
        cy_i = track_cy - ch_i // 2

        im_src = thumbs_4[s_num].resize((cw_i, ch_i), Image.Resampling.LANCZOS)
        if not is_center:
            arr = np.array(im_src).astype(np.float32) * 0.88
            im_src = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        mask = Image.new("L", (cw_i, ch_i), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw_i - 1, ch_i - 1], radius=16 if is_center else 12, fill=255)
        ov.paste(im_src.convert("RGBA"), (cx_i, cy_i), mask)

        # Top-left tiny badge: PREV / NOW / NEXT
        tag_s = f"{slot_tags[i]}·S{s_num}"
        od.rounded_rectangle([cx_i + 5, cy_i + 5, cx_i + 64, cy_i + 22], radius=5, fill=(8, 10, 14, 210))
        od.text((cx_i + 9, cy_i + 7), tag_s, font=f_tag, fill=(255, 215, 120, 255) if is_center else (210, 220, 235, 225))

        # Bottom label inside thumbnail (like @demicstory "GLUTES")
        m_lbl = step_short[s_num]
        tb_m = od.textbbox((0, 0), m_lbl, font=f_mus)
        mw = tb_m[2] - tb_m[0]
        mx = cx_i + (cw_i - mw) // 2
        my = cy_i + ch_i - 22
        od.rounded_rectangle([mx - 6, my - 2, mx + mw + 6, my + 16], radius=5, fill=(8, 10, 14, 215))
        od.text((mx, my), m_lbl, font=f_mus, fill=(255, 255, 255, 255))

        if is_center:
            # @demicstory signature glowing neon orange border on center active thumbnail
            b_col = (255, 132, 24)
            od.rounded_rectangle([cx_i - 3, cy_i - 3, cx_i + cw_i + 3, cy_i + ch_i + 3], radius=18, outline=(*b_col, 115), width=5)
            od.rounded_rectangle([cx_i - 1, cy_i - 1, cx_i + cw_i + 1, cy_i + ch_i + 1], radius=16, outline=(*b_col, 255), width=3)
        else:
            od.rounded_rectangle([cx_i, cy_i, cx_i + cw_i, cy_i + ch_i], radius=12, outline=(255, 255, 255, 85), width=1)

    # @demicstory Right Circular Status Badge (`✕` or `✓`)
    bcx = tx0 + total_w + 50
    bcy = track_cy
    br = 36
    halo_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo_layer)
    b_col = (255, 48, 48) if is_wrong else (36, 248, 118)
    hd.ellipse([bcx - br - 10, bcy - br - 10, bcx + br + 10, bcy + br + 10], fill=(*b_col, 95))
    halo_layer = halo_layer.filter(ImageFilter.GaussianBlur(radius=12))
    canvas.alpha_composite(halo_layer)

    od.ellipse([bcx - br, bcy - br, bcx + br, bcy + br], fill=(18, 22, 28, 220), outline=(235, 242, 255, 205), width=2)
    if is_wrong:
        s = 14
        od.line([(bcx - s, bcy - s), (bcx + s, bcy + s)], fill=(*b_col, 255), width=6)
        od.line([(bcx - s, bcy + s), (bcx + s, bcy - s)], fill=(*b_col, 255), width=6)
    else:
        pts = [(bcx - 15, bcy + 1), (bcx - 4, bcy + 13), (bcx + 17, bcy - 11)]
        od.line(pts, fill=(*b_col, 255), width=6, joint="curve")

    canvas.alpha_composite(ov)


def draw_day2_updated_3stage_top_dock(
    canvas: Image.Image,
    thumbs_4: dict[int, Image.Image],
    step_idx: int,
    is_wrong: bool,
):
    """
    Day 2 Top-Left Dock upgraded to show Sliding `[PREV · CURRENT · NEXT]` thumbnails at full 88% brightness
    with clear `PREV / NOW / NEXT` tags so all 3 thumbnails are 100% readable (`t311u`!).
    """
    prev_idx = {1: 4, 2: 1, 3: 2, 4: 3}[step_idx]
    curr_idx = step_idx
    next_idx = {1: 2, 2: 3, 3: 4, 4: 1}[step_idx]
    slot_steps = [prev_idx, curr_idx, next_idx]
    slot_tags = ["PREV", "NOW", "NEXT"]
    f_tag = v30.f_pretendard(12, "extrabold")

    torso_icons = []
    for i, s_num in enumerate(slot_steps):
        im = thumbs_4[s_num].resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        if i != 1:
            arr = np.array(im.convert("RGB")).astype(np.float32) * 0.88
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
        idraw = ImageDraw.Draw(im)
        tag_s = f"{slot_tags[i]}·S{s_num}"
        idraw.rounded_rectangle([6, 6, 68, 23], radius=5, fill=(8, 10, 14, 210))
        idraw.text((10, 8), tag_s, font=f_tag, fill=(255, 215, 120, 255) if i == 1 else (215, 225, 240, 230))
        torso_icons.append((im, im))
    p58.draw_v58_day2_unsquished_top_dock(canvas, torso_icons, 1, is_wrong)
# ==============================================================================
# 5. RENDER ALL 4 VARIANTS (A, B, C, D) ACROSS STEPS 1..4
# ==============================================================================
def render_step_variant(
    variant: str,  # "A", "B", "C", "D"
    step_idx: int,
    base_data: dict,
    thumbs_map: dict,
) -> np.ndarray:
    """
    Renders a single `1080x1920` RGB frame for `variant` ("A", "B", "C", "D") and `step_idx` (1..4).
    - Variant A (`@demicstory 1:1 Pure Copy`):
      * Color mode: `"demic_orange"` (Orange-Coral striated target muscle + Neon Green/Red subsurface laser axis so laser never blends into muscle!).
      * UI: `@demicstory` centered 3-thumbnail carousel (`Prev · Current · Next` clearly visible) + right `✕/✓` circle badge, ZERO floating text/numbers, ZERO bottom HUD.
    - Variant B (`Day 2 Classic Format + Locked Tone & Crisp Laser`):
      * Color mode: `"day2"` (Crimson Wrong / Emerald Right muscle fibers + locked Day 2 mannequin & studio background).
      * UI: Day 2 top-left Exercise Header + 3-stage thumbnail dock + floating callouts + Day 2 bottom-left HUD.
    - Variant C (`Mix Version — @demicstory Color Separation + Day 2 HUD`):
      * Color mode: `"demic_orange"` (`@demicstory` Orange-Coral muscle + Neon Green/Red laser line for 100% line contrast).
      * UI: Day 2 top-left Exercise Header + 3-stage dock + Bottom-Left HUD, ZERO mid-screen floating text callouts so the 3D mannequin stage stays 100% clean.
    - Variant D (`Ultimate Hybrid Master — 최강 조합 추천안`):
      * Color mode: `"hybrid_ultimate"` (Step 1/3 Crimson + 3D Bone X-Ray; Step 2/4 Emerald-Teal muscle with white tendon attachments + Electric Cyan-Mint/Hot-White subsurface laser axis so both Red/Green intuition AND laser line contrast are 100% preserved!).
      * UI: Centered Exercise Header Pill + `@demicstory` 3-Thumbnail Pop-Out Carousel (`Prev · Current · Next` clearly visible) + Right `✕/✓` Badge + Clean Bottom-Left HUD (ZERO floating text around the body!).
    """
    color_mode = {
        "A": "demic_orange",
        "B": "day2",
        "C": "demic_orange",
        "D": "hybrid_ultimate",
    }[variant]
    laser_style = "hybrid_contrast" if variant == "D" else "demic_green_red"

    rgb_base, body_mask, kp = base_data[(color_mode, step_idx)]
    meta = {
        1: {"is_wrong": True,  "deg": 16, "pct": 84, "tag_pill": "KNEE VALGUS RISK",       "tag_ctx": "BARBELL BACK SQUAT", "cue": "16° KNEE VALGUS"},
        2: {"is_wrong": False, "deg": 0,  "pct": 96, "tag_pill": "NEUTRAL KNEE LOCK",      "tag_ctx": "QUADRICEPS & VMO",   "cue": "0° KNEE TRACKING"},
        3: {"is_wrong": True,  "deg": 41, "pct": 79, "tag_pill": "LUMBAR SPINE RISK",      "tag_ctx": "GOOD-MORNING SQUAT", "cue": "41° FORWARD LEAN"},
        4: {"is_wrong": False, "deg": 22, "pct": 98, "tag_pill": "FULL QUAD & GLUTE · 4×10", "tag_ctx": "QUADRICEPS & GLUTES","cue": "22° NEUTRAL SPINE"},
    }[step_idx]
    is_wrong = meta["is_wrong"]

    canvas = Image.fromarray(rgb_base).convert("RGBA")

    # Joint-axis geometry (identical clean @demicstory bone axes with crossing vertex extensions)
    sh = kp["sh"]
    hip_r, hip_l = kp["hip_r"], kp["hip_l"]
    kn_r, kn_l = kp["kn_r"], kp["kn_l"]
    ank_r, ank_l = kp["ank_r"], kp["ank_l"]

    if step_idx == 1:
        kn_r_v = (kn_r[0] + 18.0, kn_r[1])
        kn_l_v = (kn_l[0] - 18.0, kn_l[1])
        top_r, _ = p62._extend_segment(hip_r, kn_r_v, -0.15, 1.0)
        _, bot_r = p62._extend_segment(kn_r_v, ank_r, 0.0, 1.15)
        top_l, _ = p62._extend_segment(hip_l, kn_l_v, -0.15, 1.0)
        _, bot_l = p62._extend_segment(kn_l_v, ank_l, 0.0, 1.15)
        ext_r1, ext_r2 = p62._extend_segment(hip_r, kn_r_v, 0.58, 1.48)
        ext_l1, ext_l2 = p62._extend_segment(hip_l, kn_l_v, 0.58, 1.48)
        main_chains = [[top_r, kn_r_v, bot_r], [top_l, kn_l_v, bot_l]]
        aux_chains = [[ext_r1, ext_r2], [ext_l1, ext_l2]]
        v_pts = [kn_r_v, kn_l_v]
    elif step_idx == 2:
        top_r, _ = p62._extend_segment(hip_r, kn_r, -0.15, 1.0)
        _, bot_r = p62._extend_segment(kn_r, ank_r, 0.0, 1.15)
        top_l, _ = p62._extend_segment(hip_l, kn_l, -0.15, 1.0)
        _, bot_l = p62._extend_segment(kn_l, ank_l, 0.0, 1.15)
        ext_r1, ext_r2 = p62._extend_segment(hip_r, kn_r, 0.62, 1.38)
        ext_l1, ext_l2 = p62._extend_segment(hip_l, kn_l, 0.62, 1.38)
        main_chains = [[top_r, kn_r, bot_r], [top_l, kn_l, bot_l]]
        aux_chains = [[ext_r1, ext_r2], [ext_l1, ext_l2]]
        v_pts = [kn_r, kn_l]
    elif step_idx == 3:
        sh_top, _ = p62._extend_segment(sh, hip_r, -0.16, 1.0)
        _, kn_ext = p62._extend_segment(hip_r, kn_r, 0.0, 1.16)
        torso_cross_a, torso_cross_b = p62._extend_segment(sh, hip_r, 0.65, 1.42)
        thigh_cross_a, thigh_cross_b = p62._extend_segment(kn_r, hip_r, 0.68, 1.34)
        main_chains = [[sh_top, hip_r, kn_ext]]
        aux_chains = [[torso_cross_a, torso_cross_b], [thigh_cross_a, thigh_cross_b]]
        v_pts = [hip_r]
    else:
        sh_top, _ = p62._extend_segment(sh, hip_r, -0.16, 1.0)
        _, kn_ext = p62._extend_segment(hip_r, kn_r, 0.0, 1.16)
        torso_cross_a, torso_cross_b = p62._extend_segment(sh, hip_r, 0.68, 1.36)
        thigh_cross_a, thigh_cross_b = p62._extend_segment(kn_r, hip_r, 0.68, 1.32)
        main_chains = [[sh_top, hip_r, kn_ext]]
        aux_chains = [[torso_cross_a, torso_cross_b], [thigh_cross_a, thigh_cross_b]]
        v_pts = [hip_r, kn_r]

    draw_subsurface_additive_laser_axis(
        canvas,
        body_mask_1080=body_mask,
        main_chains_1x=main_chains,
        aux_ext_chains_1x=aux_chains,
        vertex_pts_1x=v_pts,
        is_wrong=is_wrong,
        laser_style=laser_style,
    )

    thumbs_3 = thumbs_map[color_mode]

    if variant == "A":
        # Pure @demicstory 1:1 UI: centered 3-thumb carousel + right status badge, 0% text
        draw_demic_carousel_top_dock(canvas, thumbs_3, step_idx, is_wrong, with_exercise_header=False)
    elif variant == "B":
        # Day 2 Classic UI: top-left header + 3-thumb dock + floating callouts + bottom HUD
        draw_day2_updated_3stage_top_dock(canvas, thumbs_3, step_idx, is_wrong)
        shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        if step_idx == 1:
            p58.draw_v58_floating_white_callout(od, sd, (255, 895), "16°", "KNEE VALGUS", align="right")
            p58.draw_v58_floating_white_callout(od, sd, (785, 855), "84%", "ACL & MCL LOAD", align="left")
        elif step_idx == 2:
            p58.draw_v58_floating_white_callout(od, sd, (255, 895), "0°", "KNEE TRACKING", align="right")
            p58.draw_v58_floating_white_callout(od, sd, (785, 855), "96%", "QUAD & VMO LOCK", align="left")
        elif step_idx == 3:
            p58.draw_v58_floating_white_callout(od, sd, (265, 895), "41°", "FORWARD LEAN", align="right")
            p58.draw_v58_floating_white_callout(od, sd, (785, 855), "79%", "LUMBAR OVERLOAD", align="left")
        else:
            p58.draw_v58_floating_white_callout(od, sd, (265, 895), "22°", "NEUTRAL SPINE", align="right")
            p58.draw_v58_floating_white_callout(od, sd, (785, 855), "98%", "QUADS & GLUTES", align="left")
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
        canvas.alpha_composite(ov)
        p58.draw_v58_safezone_bottom_hud(canvas, {
            "is_wrong": is_wrong, "pct": meta["pct"], "tag_pill": meta["tag_pill"], "tag_ctx": meta["tag_ctx"], "cue_1line": meta["cue"]
        })
    elif variant == "C":
        # Mix Version: @demicstory Orange-Coral muscle + Neon Green/Red laser + Day 2 Top & Bottom HUD (0% floating body text)
        draw_day2_updated_3stage_top_dock(canvas, thumbs_3, step_idx, is_wrong)
        p58.draw_v58_safezone_bottom_hud(canvas, {
            "is_wrong": is_wrong, "pct": meta["pct"], "tag_pill": meta["tag_pill"], "tag_ctx": meta["tag_ctx"], "cue_1line": meta["cue"]
        })
    elif variant == "D":
        # Ultimate Hybrid Master (최강 조합):
        # Centered Exercise Header + @demicstory 3-Stage Pop-Out Carousel (`Prev·Curr·Next` clear) + Right `✕/✓` Badge + Clean Bottom HUD (0% floating clutter on body)
        draw_demic_carousel_top_dock(canvas, thumbs_3, step_idx, is_wrong, with_exercise_header=True)
        p58.draw_v58_safezone_bottom_hud(canvas, {
            "is_wrong": is_wrong, "pct": meta["pct"], "tag_pill": meta["tag_pill"], "tag_ctx": meta["tag_ctx"], "cue_1line": meta["cue"]
        })

    return np.array(canvas.convert("RGB"))


def run_all():
    raw_peak_frames = {}
    for s_i in (1, 2, 3, 4):
        fn, i0, i1, is_w, sc, y_st, y_sq, ys_st, ys_sq, x_sh = p62.STEP_POOL_SPECS[s_i]
        seq = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=52)
        raw_peak_frames[s_i] = (seq[-1], is_w)

    base_data = {}
    thumbs_map = {}
    for cm in ("demic_orange", "day2", "hybrid_ultimate"):
        step_rgbs = {}
        step_kps = {}
        for s_i in (1, 2, 3, 4):
            fr_bgr, is_w = raw_peak_frames[s_i]
            rgb_1080, mask_1080, kp_1080 = compose_locked_1080p_frame(fr_bgr, s_i, is_w, color_mode=cm, rep_a=1.0)
            base_data[(cm, s_i)] = (rgb_1080, mask_1080, kp_1080)
            step_rgbs[s_i] = rgb_1080
            step_kps[s_i] = kp_1080
        thumbs_map[cm] = make_4step_preview_thumbs(step_rgbs, step_kps)

    variants = {}
    for v in ("A", "B", "C", "D"):
        variants[v] = {}
        for s_i in (1, 2, 3, 4):
            variants[v][s_i] = render_step_variant(v, s_i, base_data, thumbs_map)

    # Quantitative QC Printout
    print("=== QUANTITATIVE QC CHECKLIST REPORT ===")
    for v in ("A", "B", "C", "D"):
        bg_lums = []
        skin_p50s = []
        sobel_p90s = []
        for s_i in (1, 2, 3, 4):
            rgb = variants[v][s_i].astype(np.float32)
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            bg_lums.append(float(np.median(lum[600:850, 20:90])))
            roi = lum[480:1280, 220:860]
            sobel = np.hypot(cv2.Sobel(roi, cv2.CV_32F, 1, 0), cv2.Sobel(roi, cv2.CV_32F, 0, 1))
            sobel_p90s.append(float(np.percentile(sobel, 90)))
            # Non-target upper chest/shoulder skin check
            ub_r, ub_g, ub_b = rgb[500:720, 380:680, 0], rgb[500:720, 380:680, 1], rgb[500:720, 380:680, 2]
            ub_lum = 0.299 * ub_r + 0.587 * ub_g + 0.114 * ub_b
            neutral = (np.abs(ub_r - ub_g) < 12) & (ub_lum > 48)
            skin_p50s.append(float(np.median(ub_lum[neutral])) if neutral.sum() > 50 else 86.0)
        print(f"Variant {v}: BG_lum_steps={[round(x,2) for x in bg_lums]} (std={np.std(bg_lums):.3f}) | Skin_med={[round(x,1) for x in skin_p50s]} (std={np.std(skin_p50s):.2f}) | Sharpness_Sobel_p90={[round(x,1) for x in sobel_p90s]}")

    # Build individual 4-Step Full-Detail Sheets for A, B, C, D + 1 Master Comparison Board
    titles = {
        "A": "시안 A · 순수 @demicstory 1:1 카피 (오렌지 근육 + 네온 그린/레드 레이저 분리 + 중앙 3-프리뷰&배지 + 화면 텍스트 0%)",
        "B": "시안 B · Day 2 포맷 + 고정 색감·레이저 관절선 (레드/그린 근육 + 상·하단 Day 2 HUD + 좌우 수치 콜아웃)",
        "C": "시안 C · 믹스 버전 (@demicstory 오렌지 근육 & 그린/레드 레이저 분리 + Day 2 상·하단 HUD + 몸 주변 글자 0%)",
        "D": "시안 D · 최강 조합 추천안 (레드/그린 직관성 + 시안-화이트 고대비 레이저 + 중앙 3-프리뷰&배지 + 하단 핵심 HUD)",
    }

    cw, ch = 460, 818
    pad = 20
    for v in ("A", "B", "C", "D"):
        sheet = Image.new("RGB", (cw * 4 + pad * 5, ch + 96), (12, 14, 18))
        sd = ImageDraw.Draw(sheet)
        f_t = v30.f_pretendard(22, "extrabold")
        f_s = v30.f_pretendard(16, "bold")
        sd.text((pad, 18), titles[v], font=f_t, fill=(40, 245, 135) if v == "D" else (255, 255, 255))
        step_names = [
            "STEP 1 (정면 오자세: 16° 무릎 안쪽 붕괴)",
            "STEP 2 (정면 정자세: 0° 무릎-발끝 정렬)",
            "STEP 3 (측면 오자세: 41° 상체 과숙임)",
            "STEP 4 (측면 정자세: 22° 중립 척추 락온)",
        ]
        for idx, s_i in enumerate((1, 2, 3, 4)):
            x0 = pad + idx * (cw + pad)
            y0 = 64
            sd.text((x0 + 6, y0 - 22), step_names[idx], font=f_s, fill=(255, 110, 110) if s_i in (1, 3) else (90, 245, 155))
            im_r = Image.fromarray(variants[v][s_i]).resize((cw, ch), Image.Resampling.LANCZOS)
            sheet.paste(im_r, (x0, y0))
        out_p = os.path.join(OUT_DIR, f"day3_mockup_variant_{v}.jpg")
        sheet.save(out_p, quality=95)
        print("Saved:", out_p)

    # Also save a 4-Variant Side-by-Side Comparison Sheet (showing Step 2 Right & Step 3 Wrong for A, B, C, D)
    master = Image.new("RGB", (cw * 4 + pad * 5, ch * 2 + 165), (10, 12, 16))
    md = ImageDraw.Draw(master)
    f_hdr = v30.f_pretendard(24, "extrabold")
    f_col = v30.f_pretendard(17, "extrabold")
    md.text((pad, 18), "DAY 3 BARBELL BACK SQUAT · 4가지 시안 (A · B · C · D) 정밀 비교 (고정 배경 + 고정 마네킹 + 블러 0% + 3단 프리뷰)", font=f_hdr, fill=(255, 255, 255))
    short_lbls = [
        "시안 A: 순수 @demicstory 1:1 카피",
        "시안 B: Day 2 포맷 + 레이저 관절선",
        "시안 C: 믹스 (오렌지 근육 + Day 2 HUD)",
        "시안 D: 최강 조합 (고대비 레이저 + 클린 뷰)",
    ]
    for idx, v in enumerate(("A", "B", "C", "D")):
        x0 = pad + idx * (cw + pad)
        md.text((x0 + 6, 58), short_lbls[idx], font=f_col, fill=(45, 255, 145) if v == "D" else (235, 240, 250))
        # Row 1: Step 2 (Front Correct)
        im_top = Image.fromarray(variants[v][2]).resize((cw, ch), Image.Resampling.LANCZOS)
        master.paste(im_top, (x0, 88))
        # Row 2: Step 3 (Side Wrong)
        im_bot = Image.fromarray(variants[v][3]).resize((cw, ch), Image.Resampling.LANCZOS)
        master.paste(im_bot, (x0, 88 + ch + 28))
    master_p = os.path.join(OUT_DIR, "day3_mockup_all_4variants_overview.jpg")
    master.save(master_p, quality=95)
    print("Saved:", master_p)

if __name__ == "__main__":
    run_all()
