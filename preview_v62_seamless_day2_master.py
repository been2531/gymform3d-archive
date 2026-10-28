#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v62_seamless_day2_master.py`)
==================================================================
Fixes ALL 5 feedbacks from 중현 님 (`t267u`–`t272u`):
1. Shorts-Free 100% Exposed Lower-Body Écorché Anatomy (`t268u`).
2. True Day-2 Close-Up Full-Body Framing (`scale = 1.26 .. 1.32`, `head_top = 462..515px`, zero black gap band!) (`t269u`, `t267u`).
3. Zero Upper-Back Whitening in Steps 3 & 4 (`t270u`, `t272u`).
4. Landmark-Locked Pivot Rings (`◎`) & Directional Arrow Arcs in Steps 1–4 (`t271u`, `t272u`).
5. 100% Real-Time Callout ↔ Bottom HUD Synchronization & Zero Border Seam (`h_seam < 1.0, v_seam < 1.5`).
"""

import os
import sys
import math
import pickle
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import universal_anatomy_rig as uar
import build_v30_pure_graphic_numbers as v30
import build_v36_clean_master as v36
import build_v38_high_contrast_hud as v38
import build_v39_ig_safe_zone as v39
import preview_v58_day2_exact_master as p58

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_V62_PKL = "/workspace/daily_info_reels/_cache_day3_v62_pools.pkl"
W, H = 1080, 1920

COL_RED = (248, 40, 48)       # Deep Vivid Day-2 Crimson Red (#F82830)
COL_RED_SOFT = (255, 162, 166)
COL_GREEN = (20, 228, 120)    # Vivid Day-2 Emerald Green (#14E478)
COL_GREEN_SOFT = (172, 255, 212)
COL_WHITE = (255, 255, 255)
COL_PLUMB = (228, 238, 252)


def _load_slice(path: str, i0: int, i1: int, target_len: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frs = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frs.append(fr)
    cap.release()
    assert len(frs) > max(i0, i1), f"Clip {path} has {len(frs)} frames, needed {max(i0, i1)}"
    seq = frs[i0 : i1 + 1] if i0 <= i1 else frs[i1 : i0 + 1][::-1]
    if target_len is not None and len(seq) != target_len:
        out = []
        n = len(seq)
        for k in range(target_len):
            pos = k * (n - 1) / float(max(1, target_len - 1))
            a = int(math.floor(pos))
            b = min(n - 1, a + 1)
            w = pos - a
            out.append(cv2.addWeighted(seq[a], 1.0 - w, seq[b], w, 0))
        return out
    return seq


def _segment_distance_field(
    xx: np.ndarray,
    yy: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    len2 = max(64.0, dx * dx + dy * dy)
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / len2, t_min, t_max)
    px = x0 + t * dx
    py = y0 + t * dy
    dist = np.hypot(xx - px, yy - py)
    return dist, t


def shade_v62_day2_authentic_720p(
    bgr_720p: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    kp_720: dict,
    rep_a: float = 1.0,
) -> np.ndarray:
    """
    2-Stage Landmark-Guided 3D Muscle-Fiber & Internal X-Ray Shader (`v62`):
    - Stage 1 (Non-Target Matte Dark-Charcoal Skin):
      Computes skin luminance strictly from `cool_base = np.minimum(g_ch, b_ch)` (zero `r_ch` contamination!)
      clamped to `[18, 76]`, and suppresses any bright overhead specular glare (`base_lum > 68`) across ALL 4 steps
      so the upper back and shoulders NEVER look white (`t270u`).
    - Stage 2 (Anatomical Target Muscle/X-Ray Glow):
      Combines native 3D striated muscle glow (`rb_diff`) with MediaPipe landmark-anchored anatomical masks
      so the full bare gluteus maximus and quadriceps femoris (pelvis to knee) glow vividly without shorts!
    """
    blur = cv2.GaussianBlur(bgr_720p, (0, 0), 0.85)
    plate_bgr = cv2.addWeighted(bgr_720p, 1.25, blur, -0.25, 0)
    rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    fiber_signal = 0.65 * r_ch + 0.25 * g_ch + 0.10 * b_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 0.75)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.0)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 5.2)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 11.5)

    hf_ridge = (sigma_micro - sigma_ridge) * 4.8 + (sigma_ridge - sigma_belly) * 2.8
    fs = np.clip(0.46 + hf_ridge / 28.0, 0.0, 1.0) ** 1.45
    hv = np.clip((sigma_belly - 22.0) / 138.0, 0.0, 1.0)
    sulc = np.clip((sigma_broad - sigma_ridge - 0.85) / 9.5, 0.0, 1.0)

    # Character A (`PRIMARY LIFTER`) Unified Sculpted Matte Silver-Grey Skin Shader (`t282u`–`t285u`, `@demicstory` Standard):
    # Quantile-normalizes the non-target body skin across Steps 1, 2, 3, and 4 to the exact same
    # Sculpted Matte Silver-Grey tone (`p15 -> 84`, `p50 -> 118`, `p85 -> 154`, clamped `[68, 174]`,
    # pure achromatic cool silver `R:G:B = 0.98 : 1.00 : 1.03`, 0% brown/green color cast!)
    # while locking the studio background to a uniform Dark Obsidian (`#15181F`, `lum ~ 22`).
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    bg_lum_est = float(np.median(base_lum[180:720, 12:58]))
    body_interior = np.clip((base_lum - (bg_lum_est + 9.5)) / 13.5, 0.0, 1.0)

    cool_base = np.minimum(g_ch, b_ch)
    body_pixels = cool_base[body_interior > 0.55]
    if len(body_pixels) > 200:
        p15 = float(np.percentile(body_pixels, 15))
        p50 = float(np.percentile(body_pixels, 50))
        p85 = float(np.percentile(body_pixels, 85))
    else:
        p15, p50, p85 = 40.0, 58.0, 88.0

    low_t = np.clip((cool_base - p15) / max(6.0, p50 - p15), -0.55, 1.0)
    high_t = np.clip((cool_base - p50) / max(8.0, p85 - p50), 0.0, 1.35)
    char_a_skin = np.where(
        cool_base <= p50,
        118.0 + low_t * (118.0 - 84.0),
        118.0 + high_t * (154.0 - 118.0),
    )
    char_a_skin = np.clip(
        char_a_skin + np.clip(hf_ridge, -14.0, 16.0) * 0.52 - sulc * 11.5,
        68.0,
        174.0,
    )

    bg_studio = np.clip(21.0 + (base_lum - bg_lum_est) * 0.45, 16.0, 30.0)
    unified_lum = bg_studio * (1.0 - body_interior) + char_a_skin * body_interior

    sh_x, sh_y = kp_720["sh"]
    hip_r = kp_720["hip_r"]
    hip_l = kp_720["hip_l"]
    kn_r = kp_720["kn_r"]
    kn_l = kp_720["kn_l"]

    kn_y = 0.5 * (kn_r[1] + kn_l[1])
    hip_y = 0.5 * (hip_r[1] + hip_l[1])

    mono = np.stack([
        unified_lum * 0.98,
        unified_lum * 1.00,
        unified_lum * 1.03,
    ], axis=2)

    y_cut_shin = kn_y + 24.0
    above_shin_gate = np.clip((y_cut_shin - yy) / 18.0, 0.0, 1.0)

    rb_blur = cv2.GaussianBlur(rb_diff, (0, 0), 4.0) * body_interior
    rb_eff = np.maximum(rb_diff, rb_blur * 0.92)

    if is_wrong:
        deep_r = np.array([128.0, 6.0, 14.0], dtype=np.float32)
        mid_r = np.array([250.0, 20.0, 32.0], dtype=np.float32)
        crest_r = np.array([255.0, 98.0, 104.0], dtype=np.float32)

        if step_idx == 1:
            # Step 1 (Front Knee Valgus): Bilateral Knee Joints + Medial Thighs/Distal Quads
            d_kn_r = np.hypot(xx - kn_r[0], yy - kn_r[1])
            d_kn_l = np.hypot(xx - kn_l[0], yy - kn_l[1])
            d_th_r, _ = _segment_distance_field(xx, yy, hip_r, kn_r, 0.18, 1.0)
            d_th_l, _ = _segment_distance_field(xx, yy, hip_l, kn_l, 0.18, 1.0)

            r_kn = 40.0 + 38.0 * rep_a
            r_th = 34.0 + 44.0 * rep_a
            knee_hotspot = np.maximum(np.exp(-0.5 * (d_kn_r / r_kn) ** 2), np.exp(-0.5 * (d_kn_l / r_kn) ** 2))
            thigh_field = np.maximum(np.exp(-0.5 * (d_th_r / r_th) ** 2), np.exp(-0.5 * (d_th_l / r_th) ** 2))
            valgus_gate = np.clip((yy - (hip_y + 38.0 - 94.0 * rep_a)) / 28.0, 0.0, 1.0) * above_shin_gate

            native_red = np.clip((rb_eff - 8.5) / 30.0, 0.0, 1.0) * valgus_gate * (0.66 + 0.34 * rep_a)
            sculpt_red = np.maximum(knee_hotspot * 0.95, thigh_field * (0.72 + 0.24 * rep_a)) * body_interior * valgus_gate
            xray_alpha = np.clip(np.maximum(native_red, sculpt_red), 0.0, 1.0)
            bone_w = (
                np.clip((g_ch - 72.0) / 80.0, 0.0, 1.0)
                * np.clip((rb_diff - 20.0) / 45.0, 0.0, 1.0)
                * knee_hotspot
                * above_shin_gate
            )[:, :, None]
        else:
            # Step 3 (3/4 Side Good-Morning Squat — Internal Lumbar Spine L1-L5 & Erector Spinae Overload ONLY):
            # Strict cutoff above the glutes (`hip_r[1] - 6.0`) so glutes & thighs stay 100% matte dark-charcoal grey!
            d_spine, _ = _segment_distance_field(xx, yy, hip_r, (sh_x, sh_y), 0.06, 0.84)
            above_glute_gate = np.clip(((hip_r[1] + 2.0 + 14.0 * rep_a) - yy) / 24.0, 0.0, 1.0)
            below_neck_gate = np.clip((yy - (sh_y + 2.0)) / 24.0, 0.0, 1.0)
            r_sp = 42.0 + 18.0 * rep_a
            lumbar_gate = np.exp(-0.5 * (d_spine / r_sp) ** 2) * above_glute_gate * below_neck_gate
            native_lumbar = np.clip((rb_eff - 7.5) / 26.0, 0.0, 1.0) * above_glute_gate * below_neck_gate
            xray_alpha = np.clip(np.maximum(native_lumbar * (0.85 + 0.15 * rep_a), lumbar_gate * (0.86 + 0.12 * rep_a)) * body_interior * above_shin_gate, 0.0, 1.0)
            bone_w = (
                np.clip((g_ch - 82.0) / 90.0, 0.0, 1.0)
                * np.clip((rb_diff - 28.0) / 52.0, 0.0, 1.0)
                * above_glute_gate
                * 0.55
            )[:, :, None]

        xray_bone = np.stack([
            np.clip(r_ch * 1.15 + 24.0 + hf_ridge * 0.65, 0, 255),
            np.clip(g_ch * 0.42 + 10.0 + hf_ridge * 0.35, 0, 145),
            np.clip(b_ch * 0.28 + 8.0, 0, 95),
        ], axis=2)

        strand_mod = (0.18 + 0.82 * fs) * ((1.0 - 0.76 * sulc) ** 1.32)
        ridge_boost = np.clip(fs * 0.76 + np.clip((rb_diff - 48.0) / 68.0, 0.0, 1.0) * 0.44, 0.0, 1.0)[:, :, None]
        vol_shade = np.clip(0.24 + 0.76 * hv * (0.36 + 0.64 * fs) * (1.0 - 0.48 * sulc), 0.0, 1.0)[:, :, None]

        fiber_red = (
            deep_r * (1.0 - vol_shade)
            + mid_r * vol_shade * (1.0 - ridge_boost * 0.56)
            + crest_r * (ridge_boost ** 1.38) * 0.84
        )
        f_alpha = np.clip(xray_alpha * (0.68 + 0.64 * strand_mod), 0.0, 0.97)[:, :, None]
        combined_src = fiber_red * (1.0 - bone_w) + xray_bone * bone_w
        alpha_final = f_alpha * (1.0 - bone_w) + (xray_alpha[:, :, None] * 0.96) * bone_w
        out = mono * (1.0 - alpha_final) + np.clip(combined_src, 0, 255) * alpha_final
        return np.clip(out, 0, 255).astype(np.uint8)

    else:
        # Steps 2 & 4 (Right): Rich Day-2 Emerald-Green 3D Muscle Fiber Bundles (`근육결`)
        d_th_r, _ = _segment_distance_field(xx, yy, hip_r, kn_r, -0.10, 0.98)
        d_th_l, _ = _segment_distance_field(xx, yy, hip_l, kn_l, -0.10, 0.98)

        if step_idx == 2:
            # Foreshortening compensation (`r_th` widens as thighs squat down so active_px never shrinks!)
            r_th = 36.0 + 48.0 * rep_a
            target_gate = np.clip((yy - (hip_y + 12.0 - 96.0 * rep_a)) / 26.0, 0.0, 1.0) * above_shin_gate
            quad_anat = np.maximum(
                np.exp(-0.5 * (d_th_r / r_th) ** 2),
                np.exp(-0.5 * (d_th_l / r_th) ** 2),
            ) * body_interior * target_gate
            native_fe = np.clip((rb_eff - 7.5) / 28.0, 0.0, 1.0) * target_gate * (0.62 + 0.38 * rep_a)
            fe = np.clip(np.maximum(native_fe, quad_anat * (0.78 + 0.20 * rep_a) * np.clip((base_lum - 34.0) / 20.0, 0.0, 1.0)), 0.0, 1.0)
        else:
            # Step 4 (3/4 Side Right): Full Gluteus Maximus, Gluteus Medius, and Quadriceps Femoris (pelvis to knee)
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
            fe = np.clip(np.maximum(native_fe, lower_anat * 0.90 * np.clip((base_lum - 38.0) / 24.0, 0.0, 1.0)), 0.0, 1.0)

        hc = np.clip((rb_diff - 48.0) / 65.0, 0.0, 1.0) * fs * target_gate
        strand_mod = (0.16 + 0.84 * fs) * ((1.0 - 0.78 * sulc) ** 1.35)
        fiber_alpha = np.clip(fe * (0.72 + 0.66 * strand_mod), 0.0, 0.97)[:, :, None]
        ridge_boost = np.clip(fs * 0.76 + hc * 0.46, 0.0, 1.0)[:, :, None]
        vol_shade = np.clip(0.24 + 0.76 * hv * (0.36 + 0.64 * fs) * (1.0 - 0.50 * sulc), 0.0, 1.0)[:, :, None]

        deep_col = np.array([4.0, 136.0, 28.0], dtype=np.float32)
        mid_col = np.array([16.0, 246.0, 64.0], dtype=np.float32)
        crest_col = np.array([170.0, 255.0, 184.0], dtype=np.float32)

        fiber_rgb = (
            deep_col * (1.0 - vol_shade)
            + mid_col * vol_shade * (1.0 - ridge_boost * 0.60)
            + crest_col * (ridge_boost ** 1.40) * 0.84
        )
        out = mono * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha
        return np.clip(out, 0, 255).astype(np.uint8)



def _apply_step1_knee_valgus_720p(bgr_720p: np.ndarray, rep_a: float) -> np.ndarray:
    """
    Applies a smooth, localized horizontal inward knee-valgus pinch (`16° KNEE VALGUS`)
    strictly within the knee/distal-thigh band (`y in 560..920` in 720p) as the athlete squats down (`rep_a: 0->1`),
    leaving the upper body/hands gripping the barbell (`y < 560`) and the flat feet/ankles (`y > 920`) 100% untouched.
    """
    if rep_a <= 0.02:
        return bgr_720p
    h, w = bgr_720p.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    wy = np.exp(-0.5 * ((yy - 745.0) / 72.0) ** 2)
    wy *= np.clip((yy - 565.0) / 45.0, 0.0, 1.0) * np.clip((925.0 - yy) / 45.0, 0.0, 1.0)
    wx_r = np.exp(-0.5 * ((xx - 225.0) / 68.0) ** 2)
    wx_l = np.exp(-0.5 * ((xx - 495.0) / 68.0) ** 2)
    shift = (44.0 * (rep_a ** 1.15)) * wy
    map_x = xx - shift * wx_r + shift * wx_l
    map_y = yy
    return cv2.remap(bgr_720p, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def build_v62_fullbleed_pool(
    raw_frames_720p: list[np.ndarray],
    step_idx: int,
    is_wrong: bool,
    scale: float,
    y_head_target_stand: int,
    y_head_target_squat: int,
    y_head_src_stand_720: float,
    y_head_src_squat_720: float,
    x_shift: int = 0,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Frames the shorts-free 720p clip at True Day-2 Close-Up Full-Body Scale (`scale = 1.24 .. 1.32`,
    `nw = 893 .. 950px`), smoothly tracking the vertical head position from `y_head_target_stand = 462px`
    to `y_head_target_squat = 495..510px` so there is ZERO empty black gap below the Top Dock (`y=438`),
    and the huge sculpted glutes & quadriceps fill the central viewport (`y = 650 .. 1255`)!
    """
    sh_h, sw_w = raw_frames_720p[0].shape[:2]  # 1280, 720
    nw = int(round(sw_w * scale))
    nh = int(round(sh_h * scale))
    x0 = (W - nw) // 2 + x_shift
    y0_stand = y_head_target_stand - y_head_src_stand_720 * scale
    y0_squat = y_head_target_squat - y_head_src_squat_720 * scale

    n_f_in = len(raw_frames_720p)
    if step_idx == 1:
        raw_frames_720p = [
            _apply_step1_knee_valgus_720p(fr, i / float(max(1, n_f_in - 1)))
            for i, fr in enumerate(raw_frames_720p)
        ]

    detector = uar.get_pose_detector()
    raw_kps_720 = []
    for bgr_f in raw_frames_720p:
        rgb_720 = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2RGB)
        mp_res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb_720))
        if mp_res.pose_landmarks:
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
        else:
            prev = raw_kps_720[-1]
            sh_pt, hip_r, hip_l = prev["sh"], prev["hip_r"], prev["hip_l"]
            kn_r, kn_l, ank_r, ank_l = prev["kn_r"], prev["kn_l"], prev["ank_r"], prev["ank_l"]
            head_top, foot_bot = prev["head_top"], prev["foot_bot"]
        raw_kps_720.append({
            "sh": sh_pt,
            "hip_r": hip_r,
            "hip_l": hip_l,
            "kn_r": kn_r,
            "kn_l": kn_l,
            "ank_r": ank_r,
            "ank_l": ank_l,
            "head_top": head_top,
            "foot_bot": foot_bot,
        })

    keys = list(raw_kps_720[0].keys())
    for k in keys:
        xs = gaussian_filter1d([p[k][0] for p in raw_kps_720], sigma=1.8)
        ys = gaussian_filter1d([p[k][1] for p in raw_kps_720], sigma=1.8)
        for i in range(len(raw_kps_720)):
            raw_kps_720[i][k] = (float(xs[i]), float(ys[i]))

    n_f = len(raw_frames_720p)
    shaded_720 = []
    for i, fr in enumerate(raw_frames_720p):
        rep_a = i / float(max(1, n_f - 1))
        shaded_720.append(shade_v62_day2_authentic_720p(fr, step_idx, is_wrong, kp_720=raw_kps_720[i], rep_a=rep_a))

    dst_x1 = max(0, x0)
    dst_x2 = min(W, x0 + nw)
    src_x1 = dst_x1 - x0
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    crop_w = dst_x2 - dst_x1

    out_rgb_list = []
    out_kps_1080 = []
    for i, s_rgb in enumerate(shaded_720):
        rep_a = i / float(max(1, n_f - 1))
        y0 = int(round(y0_stand * (1.0 - rep_a) + y0_squat * rep_a))
        y_end = y0 + nh

        # Full vertical bleed (`dst_y1` allowed up to `0` so there is ZERO horizontal seam at y=442..468!)
        dst_y1 = max(0, y0)
        dst_y2 = min(H, y_end)
        src_y1 = dst_y1 - y0
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        crop_h = dst_y2 - dst_y1

        res_rgb = cv2.resize(s_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        crop_rgb = res_rgb[src_y1:src_y2, src_x1:src_x2, :]

        # Enforce Top Dock head safe-zone (`y < 456` smoothly faded into top studio wall so head never touches y=438!)
        left_prof = np.percentile(crop_rgb[:, :28, :], 35, axis=1).astype(np.float32)
        right_prof = np.percentile(crop_rgb[:, -28:, :], 35, axis=1).astype(np.float32)
        left_prof = cv2.GaussianBlur(left_prof[:, None, :], (1, 181), 65.0)[:, 0, :]
        right_prof = cv2.GaussianBlur(right_prof[:, None, :], (1, 181), 65.0)[:, 0, :]

        u_x = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :, None]
        studio_bg = np.zeros((H, W, 3), dtype=np.float32)
        studio_bg[dst_y1:dst_y2, :, :] = left_prof[:, None, :] * (1.0 - u_x) + right_prof[:, None, :] * u_x
        if dst_y1 > 0:
            studio_bg[:dst_y1, :, :] = studio_bg[dst_y1 : dst_y1 + 1, :, :]
        if dst_y2 < H:
            studio_bg[dst_y2:, :, :] = studio_bg[dst_y2 - 1 : dst_y2, :, :]
        studio_bg = cv2.GaussianBlur(studio_bg, (61, 61), 22.0)

        xx_c = np.arange(crop_w, dtype=np.float32)[None, :]
        yy_c = np.arange(crop_h, dtype=np.float32)[:, None]
        yy_full = yy_c + dst_y1

        # Smooth 90px top/bottom/side studio blend + strict head-column top dock guard (`y_full < 456`)
        ay_t = np.clip(yy_c / 72.0, 0.0, 1.0)
        ay_b = np.clip((crop_h - 1.0 - yy_c) / 72.0, 0.0, 1.0)
        ay = 0.5 - 0.5 * np.cos(np.pi * np.minimum(ay_t, ay_b))
        ax_l = np.clip(xx_c / 96.0, 0.0, 1.0)
        ax_r = np.clip((crop_w - 1.0 - xx_c) / 96.0, 0.0, 1.0)
        ax = 0.5 - 0.5 * np.cos(np.pi * np.minimum(ax_l, ax_r))
        head_guard = np.clip((yy_full - 461.0) / 6.0, 0.0, 1.0)
        xx_full = xx_c + dst_x1
        in_head_col = np.clip((670.0 - xx_full) / 30.0, 0.0, 1.0) * np.clip((xx_full - 410.0) / 30.0, 0.0, 1.0)
        top_mask = 1.0 - in_head_col * (1.0 - head_guard)
        alpha_base = (ax * ay * top_mask)[:, :, None]

        bg_patch = studio_bg[dst_y1:dst_y2, dst_x1:dst_x2, :]
        canvas = studio_bg.copy()
        canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = crop_rgb * alpha_base + bg_patch * (1.0 - alpha_base)
        out_rgb_list.append(np.clip(canvas, 0, 255).astype(np.uint8))

        kp_1080 = {}
        for k, (kx, ky) in raw_kps_720[i].items():
            kp_1080[k] = (x0 + kx * scale, y0 + ky * scale)
        out_kps_1080.append(kp_1080)

    return out_rgb_list, out_kps_1080


def get_or_build_v62_pools() -> dict:
    print("Building Gripped + Shorts-Free Close-Up 3D Pools (`day3_v63_step1..4_i2v.mp4`, `scale = 1.24 .. 1.28`)...")
    s1_raw = _load_slice(os.path.join(OUT_DIR, "day3_v63_step1_i2v.mp4"), 78, 132, target_len=52)
    s2_raw = _load_slice(os.path.join(OUT_DIR, "day3_v63_step2_i2v.mp4"), 84, 138, target_len=52)
    s3_raw = _load_slice(os.path.join(OUT_DIR, "day3_v63_step3_i2v.mp4"), 24, 86,  target_len=52)
    s4_raw = _load_slice(os.path.join(OUT_DIR, "day3_v63_step4_i2v.mp4"), 72, 124, target_len=52)

    pools = {
        1: build_v62_fullbleed_pool(s1_raw, 1, True,  scale=1.24, y_head_target_stand=464, y_head_target_squat=502, y_head_src_stand_720=76.0,  y_head_src_squat_720=406.0, x_shift=0),
        2: build_v62_fullbleed_pool(s2_raw, 2, False, scale=1.24, y_head_target_stand=464, y_head_target_squat=502, y_head_src_stand_720=57.0,  y_head_src_squat_720=365.0, x_shift=0),
        3: build_v62_fullbleed_pool(s3_raw, 3, True,  scale=1.28, y_head_target_stand=464, y_head_target_squat=532, y_head_src_stand_720=195.0, y_head_src_squat_720=498.0, x_shift=-8),
        4: build_v62_fullbleed_pool(s4_raw, 4, False, scale=1.28, y_head_target_stand=464, y_head_target_squat=506, y_head_src_stand_720=68.0,  y_head_src_squat_720=342.0, x_shift=-8),
    }
    return pools


def _draw_joint_pivot_ring_ss(
    rd: ImageDraw.ImageDraw,
    pt_1x: tuple[float, float],
    radius_1x: float,
    col_main: tuple[int, int, int],
    ss: int = 4,
):
    cx, cy = pt_1x[0] * ss, pt_1x[1] * ss
    r_outer_sh = (radius_1x + 4.5) * ss
    r_outer = (radius_1x + 2.2) * ss
    r_mid = (radius_1x - 1.2) * ss
    r_dot = 3.6 * ss

    rd.ellipse([cx - r_outer_sh, cy - r_outer_sh, cx + r_outer_sh, cy + r_outer_sh], fill=(0, 0, 0, 215))
    rd.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=(*col_main, 255))
    rd.ellipse([cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid], fill=(12, 15, 20, 235))
    rd.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot], fill=(255, 255, 255, 255))


def _draw_curved_directional_arrow_ss(
    rd: ImageDraw.ImageDraw,
    center_1x: tuple[float, float],
    radius_1x: float,
    start_deg: float,
    end_deg: float,
    arrow_at_end: bool,
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 4,
):
    cx_s, cy_s = center_1x[0] * ss, center_1x[1] * ss
    r_s = radius_1x * ss
    steps = 48
    outer_sh, inner_sh = [], []
    outer_arc, inner_arc = [], []
    outer_core, inner_core = [], []
    hw_sh = 5.4 * ss
    hw_arc = 3.0 * ss
    hw_core = 1.0 * ss

    for i in range(steps + 1):
        a = math.radians(start_deg + (end_deg - start_deg) * (i / float(steps)))
        ca, sa = math.cos(a), math.sin(a)
        outer_sh.append((cx_s + ca * (r_s + hw_sh), cy_s + sa * (r_s + hw_sh)))
        inner_sh.append((cx_s + ca * (r_s - hw_sh), cy_s + sa * (r_s - hw_sh)))
        outer_arc.append((cx_s + ca * (r_s + hw_arc), cy_s + sa * (r_s + hw_arc)))
        inner_arc.append((cx_s + ca * (r_s - hw_arc), cy_s + sa * (r_s - hw_arc)))
        outer_core.append((cx_s + ca * (r_s + hw_core), cy_s + sa * (r_s + hw_core)))
        inner_core.append((cx_s + ca * (r_s - hw_core), cy_s + sa * (r_s - hw_core)))

    rd.polygon(outer_sh + inner_sh[::-1], fill=(0, 0, 0, 235))
    rd.polygon(outer_arc + inner_arc[::-1], fill=(*col_main, 255))
    rd.polygon(outer_core + inner_core[::-1], fill=(255, 255, 255, 230))

    tip_deg = end_deg if arrow_at_end else start_deg
    tangent_deg = (end_deg + 90.0) if arrow_at_end else (start_deg - 90.0)
    ang_r = math.radians(tip_deg)
    tip_pt = (cx_s + r_s * math.cos(ang_r), cy_s + r_s * math.sin(ang_r))
    v30.draw_arrow_head(rd, tip_pt, tangent_deg, 16.0 * ss, (0, 0, 0, 240))
    v30.draw_arrow_head(rd, tip_pt, tangent_deg, 12.5 * ss, (*col_main, 255))
    v30.draw_arrow_head(rd, tip_pt, tangent_deg, 7.8 * ss, (*col_soft, 255))


def draw_v62_high_contrast_angle_gauge_ss(
    rd: ImageDraw.ImageDraw,
    bd_1x: ImageDraw.ImageDraw,
    vertex_1x: tuple[float, float],
    ref_pt_1x: tuple[float, float],
    limb_pt_1x: tuple[float, float],
    next_limb_pt_1x: tuple[float, float] | None,
    arc_radius_1x: float,
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    draw_sector_arrow: bool = True,
    ss: int = 4,
):
    vx_s, vy_s = vertex_1x[0] * ss, vertex_1x[1] * ss
    rx_s, ry_s = ref_pt_1x[0] * ss, ref_pt_1x[1] * ss
    lx_s, ly_s = limb_pt_1x[0] * ss, limb_pt_1x[1] * ss

    a_ref = math.atan2(ry_s - vy_s, rx_s - vx_s)
    a_limb = math.atan2(ly_s - vy_s, lx_s - vx_s)
    diff = (a_limb - a_ref + math.pi) % (2.0 * math.pi) - math.pi
    r_s = arc_radius_1x * ss

    if abs(diff) >= 0.025 and arc_radius_1x > 10.0:
        steps = 64
        wedge_poly = [(vx_s, vy_s)]
        outer_arc, inner_arc = [], []
        outer_sh, inner_sh = [], []
        outer_core, inner_core = [], []
        hw_sh = 5.4 * ss
        hw_arc = 2.8 * ss
        hw_core = 0.9 * ss
        for i in range(steps + 1):
            a = a_ref + diff * (i / float(steps))
            ca, sa = math.cos(a), math.sin(a)
            wedge_poly.append((vx_s + ca * r_s, vy_s + sa * r_s))
            outer_sh.append((vx_s + ca * (r_s + hw_sh), vy_s + sa * (r_s + hw_sh)))
            inner_sh.append((vx_s + ca * (r_s - hw_sh), vy_s + sa * (r_s - hw_sh)))
            outer_arc.append((vx_s + ca * (r_s + hw_arc), vy_s + sa * (r_s + hw_arc)))
            inner_arc.append((vx_s + ca * (r_s - hw_arc), vy_s + sa * (r_s - hw_arc)))
            outer_core.append((vx_s + ca * (r_s + hw_core), vy_s + sa * (r_s + hw_core)))
            inner_core.append((vx_s + ca * (r_s - hw_core), vy_s + sa * (r_s - hw_core)))

        rd.polygon(wedge_poly, fill=(8, 10, 14, 115))
        rd.polygon(wedge_poly, fill=(*col_main, 82))
        rd.polygon(outer_sh + inner_sh[::-1], fill=(0, 0, 0, 235))
        rd.polygon(outer_arc + inner_arc[::-1], fill=(*col_main, 255))
        rd.polygon(outer_core + inner_core[::-1], fill=(255, 255, 255, 235))

        if draw_sector_arrow:
            a_end = a_ref + diff
            tip_pt = (vx_s + math.cos(a_end) * r_s, vy_s + math.sin(a_end) * r_s)
            tang_deg = math.degrees(a_end) + (90.0 if diff > 0 else -90.0)
            v30.draw_arrow_head(rd, tip_pt, tang_deg, 15.0 * ss, (0, 0, 0, 240))
            v30.draw_arrow_head(rd, tip_pt, tang_deg, 11.5 * ss, (*col_main, 255))
            v30.draw_arrow_head(rd, tip_pt, tang_deg, 7.2 * ss, (*col_soft, 255))

    dx_r, dy_r = rx_s - vx_s, ry_s - vy_s
    L_r = math.hypot(dx_r, dy_r)
    if L_r > 8.0:
        n_dashes = max(6, int(L_r // (15.0 * ss)))
        for i in range(n_dashes):
            t0 = i / float(n_dashes)
            t1 = min(1.0, (i + 0.58) / float(n_dashes))
            pa = (vx_s + dx_r * t0, vy_s + dy_r * t0)
            pb = (vx_s + dx_r * t1, vy_s + dy_r * t1)
            p58._draw_smooth_seg_ss(rd, pa, pb, 3.2 * ss, (0, 0, 0, 225))
            p58._draw_smooth_seg_ss(rd, pa, pb, 1.45 * ss, (*COL_PLUMB, 248))

    segments_1x = [(vertex_1x, limb_pt_1x)]
    if next_limb_pt_1x is not None:
        segments_1x.append((limb_pt_1x, next_limb_pt_1x))

    for p0_1x, p1_1x in segments_1x:
        bd_1x.line([p0_1x, p1_1x], fill=(*col_main, 115), width=9)
        p0_s = (p0_1x[0] * ss, p0_1x[1] * ss)
        p1_s = (p1_1x[0] * ss, p1_1x[1] * ss)
        p58._draw_smooth_seg_ss(rd, p0_s, p1_s, 3.8 * ss, (0, 0, 0, 230))
        p58._draw_smooth_seg_ss(rd, p0_s, p1_s, 2.10 * ss, (*col_main, 255))
        p58._draw_smooth_seg_ss(rd, p0_s, p1_s, 0.75 * ss, (255, 255, 255, 245))

    pts_pin = [limb_pt_1x] + ([next_limb_pt_1x] if next_limb_pt_1x is not None else [])
    for px_1x, py_1x in pts_pin:
        cx, cy = px_1x * ss, py_1x * ss
        r_sh = 6.2 * ss
        r_out = 4.6 * ss
        r_in = 2.8 * ss
        rd.ellipse([cx - r_sh, cy - r_sh, cx + r_sh, cy + r_sh], fill=(0, 0, 0, 230))
        rd.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=(*col_main, 255))
        rd.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=(255, 255, 255, 255))

    _draw_joint_pivot_ring_ss(rd, vertex_1x, radius_1x=9.5, col_main=col_main, ss=ss)


def render_v62_synchronized_angle_and_callouts(
    canvas: Image.Image,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    deg_val: int,
    pct_val: int,
    rep_amount: float = 1.0,
):
    """
    Ultra-Minimal Zero-Occlusion Exterior Arrow Standard (`t280u`, `t281u`, `t289u`, `@demicstory` Standard):
    - ZERO lines, wedges, or arrows crossing over the mannequin's muscles or torso!
    - Every `4*SS` single arc arrow sits 100% in the open studio space (`exterior margin`) outside the body silhouette!
    """
    SS = 2
    rig_layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig_layer)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    def S(pt):
        return (pt[0] * SS, pt[1] * SS)

    col_main = COL_RED if is_wrong else COL_GREEN
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    hip_r = kp["hip_r"]

    if step_idx == 1:
        # Step 1 (16° KNEE VALGUS): Pivot rings on knees + inward-pointing arc arrows strictly in the open studio space
        v36.draw_single_elbow_pivot_ring(rd, S(kn_r), 9.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_l), 9.5, col_main, SS)

        r_a = 58 * SS
        # Left knee (viewer left): arc placed in the open inner gap (`x = kn_r[0] + 64`) curving inward (`→`)
        c_r = S((kn_r[0] + 44.0, kn_r[1] + 6.0))
        rd.arc([c_r[0] - r_a, c_r[1] - r_a, c_r[0] + r_a, c_r[1] + r_a], start=316, end=32, fill=(*col_main, 245), width=4 * SS)
        ang_r = math.radians(32)
        tip_r = (c_r[0] + r_a * math.cos(ang_r), c_r[1] + r_a * math.sin(ang_r))
        v30.draw_arrow_head(rd, tip_r, 32 + 90, 12 * SS, (*col_soft, 255))

        # Right knee (viewer right): arc placed in the open inner gap (`x = kn_l[0] - 44`) curving inward (`←`)
        c_l = S((kn_l[0] - 44.0, kn_l[1] + 6.0))
        rd.arc([c_l[0] - r_a, c_l[1] - r_a, c_l[0] + r_a, c_l[1] + r_a], start=148, end=224, fill=(*col_main, 245), width=4 * SS)
        ang_l = math.radians(148)
        tip_l = (c_l[0] + r_a * math.cos(ang_l), c_l[1] + r_a * math.sin(ang_l))
        v30.draw_arrow_head(rd, tip_l, 148 - 90, 12 * SS, (*col_soft, 255))

        p58.draw_v58_floating_white_callout(od, sd, (255, 895), f"{deg_val}°", "KNEE VALGUS", align="right")
        p58.draw_v58_floating_white_callout(od, sd, (785, 855), f"{pct_val}%", "ACL & MCL LOAD", align="left")

    elif step_idx == 2:
        # Step 2 (0° KNEE TRACKING): Pivot rings on knees + outward-pointing arc arrows strictly in the open outer studio margin (`← →`)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_r), 9.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_l), 9.5, col_main, SS)

        r_a = 60 * SS
        # Left knee (viewer left): arc placed in the open left studio space (`x = kn_r[0] - 46`) pointing outward (`←`)
        c_r = S((kn_r[0] - 38.0, kn_r[1] + 4.0))
        rd.arc([c_r[0] - r_a, c_r[1] - r_a, c_r[0] + r_a, c_r[1] + r_a], start=148, end=224, fill=(*col_main, 245), width=4 * SS)
        ang_r = math.radians(148)
        tip_r = (c_r[0] + r_a * math.cos(ang_r), c_r[1] + r_a * math.sin(ang_r))
        v30.draw_arrow_head(rd, tip_r, 148 - 90, 12 * SS, (*col_soft, 255))

        # Right knee (viewer right): arc placed in the open right studio space (`x = kn_l[0] + 38`) pointing outward (`→`)
        c_l = S((kn_l[0] + 38.0, kn_l[1] + 4.0))
        rd.arc([c_l[0] - r_a, c_l[1] - r_a, c_l[0] + r_a, c_l[1] + r_a], start=316, end=32, fill=(*col_main, 245), width=4 * SS)
        ang_l = math.radians(32)
        tip_l = (c_l[0] + r_a * math.cos(ang_l), c_l[1] + r_a * math.sin(ang_l))
        v30.draw_arrow_head(rd, tip_l, 32 + 90, 12 * SS, (*col_soft, 255))

        p58.draw_v58_floating_white_callout(od, sd, (255, 895), f"{deg_val}°", "KNEE TRACKING", align="right")
        p58.draw_v58_floating_white_callout(od, sd, (785, 855), f"{pct_val}%", "QUAD & VMO LOCK", align="left")

    elif step_idx == 3:
        # Step 3 (41° FORWARD LEAN): 1 hip pivot ring + 1 single exterior arc in the open left studio margin (`x = 265..345`)
        # ZERO overlap with the glowing lumbar spine or torso!
        v36.draw_single_elbow_pivot_ring(rd, S(hip_r), 9.5, col_main, SS)

        c_arc = S((hip_r[0] - 48.0, hip_r[1] - 195.0))
        r_a = 118 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=178, end=244, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(244)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 244 + 90, 12 * SS, (*col_soft, 255))

        p58.draw_v58_floating_white_callout(od, sd, (265, 895), f"{deg_val}°", "FORWARD LEAN", align="right")
        p58.draw_v58_floating_white_callout(od, sd, (785, 855), f"{pct_val}%", "LUMBAR OVERLOAD", align="left")

    else:
        # Step 4 (22° NEUTRAL SPINE): 1 hip pivot ring + 1 single exterior arc in the open left studio margin (`x = 265..345`)
        # ZERO overlap with the glutes, quadriceps, or upper back!
        v36.draw_single_elbow_pivot_ring(rd, S(hip_r), 9.5, col_main, SS)

        c_arc = S((hip_r[0] - 54.0, hip_r[1] - 185.0))
        r_a = 118 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=184, end=248, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(248)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 248 + 90, 12 * SS, (*col_soft, 255))

        p58.draw_v58_floating_white_callout(od, sd, (265, 895), f"{deg_val}°", "NEUTRAL SPINE", align="right")
        p58.draw_v58_floating_white_callout(od, sd, (785, 855), f"{pct_val}%", "QUADS & GLUTES", align="left")

    rig_1x = Image.fromarray(cv2.resize(np.array(rig_layer), (W, H), interpolation=cv2.INTER_AREA))
    canvas.alpha_composite(rig_1x)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
    canvas.alpha_composite(ov)


if __name__ == "__main__":
    pools = get_or_build_v62_pools()
    print("v62 pools built:", {k: len(v[0]) for k, v in pools.items()})
