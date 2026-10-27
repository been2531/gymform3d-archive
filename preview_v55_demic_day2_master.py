#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v55_demic_day2_master.py`)
===============================================================
Synthesizes:
1. 100% Real, Unwarped 3D Squat Postures (`0%` `cv2.remap`, `0%` background warping, `diff == 0.0` on background):
   - Step 1 (Front View — WRONG): Realistic gym knee valgus (`14° VALGUS CAVE`) where knees drift inside the toe line.
   - Step 2 (Front View — RIGHT): Wide external rotation (`0° KNEES TRACK TOES`) where knees track cleanly over toes.
   - Step 3 (Side View — WRONG): Realistic "Good-Morning Squat" forward torso pitch (`+28° FORWARD LEAN`) & heel lift.
   - Step 4 (Side View — RIGHT): Proud upright chest, planted heels & parallel depth (`95° PARALLEL DRIVE`).
2. Strict iPhone Instagram Reels Safe-Zone Framing (`head_top >= 435`, `foot_bot <= 1275`):
   - Top Exercise Header (`y = 258..302`) & Compact 3-Thumbnail Progression Dock (`y = 310..406`) stay 100% above the
     standing mannequin's head (`y >= 435`) on every single frame (`0px` overlap!).
   - Mannequin feet (`y <= 1275`) stay 100% above the Bottom-Left HUD (`y = 1315..1444`).
3. Day 2 (`v39`) Muscle Fiber Striation & 3D Bone X-Ray Shader:
   - Steps 1 & 3: High-contrast matte charcoal-silver 3D body with glowing Crimson-Red (`#FA2E34`) 3D skeletal X-Ray
     (knees, lumbar spine, ankles) & overloaded muscle fiber striations.
   - Steps 2 & 4: High-contrast matte charcoal-silver 3D body with unified, luminous Day 2 Emerald-Green (`#16E27C`)
     3D muscle fiber striations across the Quadriceps (VMO, Rectus Femoris, Vastus Lateralis), Glutes, and neutral Spine.
4. `@demicstory` + Day 2 (`v39`) + `v48` Intuitive 3D Biomechanical Overlays:
   - Glowing anatomical bone-axis alignment chains (`Hip ◎ — Knee ◎ — Ankle ◎` & `Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`),
   - Day 2 (`v39`) exterior solid angle sectors/arcs (`14°`, `0°`, `+28°`, `95°`),
   - Compact, body-fitted `v48` 3D volumetric curved ribbon arrows (`SS=4`) showing exact movement/correction direction,
   - Slender exterior pointer lines to Pure White (`#FFFFFF`) `64pt` numbers + `22pt` sub-labels (`x in 96..978`, zero underlines).
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
import gymform_engine as eng
import universal_anatomy_rig as uar
import build_v30_pure_graphic_numbers as v30
import build_v38_high_contrast_hud as v38
import build_v39_ig_safe_zone as v39
import preview_v48_true_3d_torus_ribbons as v48

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_V55_PKL = "/workspace/daily_info_reels/_cache_day3_v55_pools.pkl"
W, H = 1080, 1920

SAFE_X_LEFT = 96
SAFE_X_RIGHT_MAX = 978
SAFE_Y_TOP = 258

COL_RED = (246, 42, 50)
COL_RED_DARK = (82, 8, 14)
COL_RED_LIGHT = (255, 172, 168)

COL_GREEN = (18, 222, 120)
COL_GREEN_DARK = (5, 74, 38)
COL_GREEN_LIGHT = (168, 255, 212)

COL_WHITE = (255, 255, 255)
COL_PLUMB = (215, 228, 245)

F_EX_TITLE = v30.f_pretendard(23, "extrabold")
F_EX_SUB = v30.f_pretendard(15, "extrabold")

F_TAG_PILL = v30.f_pretendard(19, "extrabold")
F_TAG_CTX = v30.f_pretendard(19, "bold")
F_HERO_PCT = v30.f_barlow(82, extrabold=True)
F_HERO_SYM = v30.f_barlow(42, extrabold=True)
F_SINGLE_LINE = v30.f_pretendard(42, "extrabold")

F_CALLOUT_NUM = v30.f_barlow(64, extrabold=True)
F_CALLOUT_SUB = v30.f_pretendard(22, "extrabold")


def _load_slice(path: str, i0: int, i1: int) -> list[np.ndarray]:
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


def apply_day2_v39_anatomy_shader_720p(bgr_720p: np.ndarray, step_idx: int, is_wrong: bool) -> np.ndarray:
    """
    Applies the Day 2 (`v39`) high-contrast matte charcoal-silver 3D body sculpt +
    multi-scale muscle fiber striation & 3D skeletal X-ray shader directly on the 720x1280 frame:
    - Steps 1 & 3 (`is_wrong=True`): Warm crimson studio rim/key lighting + vivid Crimson-Red 3D skeletal X-Ray
      (knees, lumbar spine, ankles) & crimson overloaded muscle fiber striations.
    - Steps 2 & 4 (`is_wrong=False`): Cool emerald-steel studio rim/key lighting + unified, luminous Day 2
      Emerald-Green (`#16DE78`) 3D muscle fiber striations with bright `#B9FFD4` fiber crests!
    """
    blur = cv2.GaussianBlur(bgr_720p, (0, 0), 0.9)
    sharp_bgr = cv2.addWeighted(bgr_720p, 1.18, blur, -0.18, 0)
    rgb = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    gb_diff = np.maximum(0.0, g_ch - b_ch)
    gr_diff = np.maximum(0.0, g_ch - r_ch * 0.80)

    fiber_signal = 0.45 * r_ch + 0.55 * g_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 0.8)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.2)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 6.0)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 13.0)

    hf_ridge = (sigma_micro - sigma_ridge) * 3.4 + (sigma_ridge - sigma_belly) * 1.9
    fs = np.clip(0.48 + hf_ridge / 38.0, 0.0, 1.0) ** 1.30
    hv = np.clip((sigma_belly - 24.0) / 155.0, 0.0, 1.0)
    sulc = np.clip((sigma_broad - sigma_ridge - 3.0) / 19.0, 0.0, 1.0)

    chroma_any = np.clip((np.maximum(rb_diff, gb_diff) - 10.0) / 72.0, 0.0, 1.0)
    grey_sculpt = np.clip(36.0 + 158.0 * (hv ** 1.02) + hf_ridge * 1.05, 14.0, 242.0)
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    neutral_lum = np.clip(base_lum * (1.0 - chroma_any) + grey_sculpt * chroma_any, 12.0, 244.0)

    bg_weight = np.clip((base_lum - 11.0) / 15.0, 0.0, 1.0)[:, :, None]
    norm = neutral_lum / 255.0
    if is_wrong:
        # Subtle warm crimson-obsidian studio key lighting on sculpted body & rack
        mono_sculpt = np.stack([
            18.0 + (242.0 - 18.0) * (norm ** 0.88),
            12.0 + (216.0 - 12.0) * (norm ** 0.94),
            14.0 + (220.0 - 14.0) * (norm ** 0.94),
        ], axis=2)
    else:
        # Subtle cool emerald-steel studio key lighting on sculpted body & rack
        mono_sculpt = np.stack([
            11.0 + (212.0 - 11.0) * (norm ** 0.94),
            17.0 + (240.0 - 17.0) * (norm ** 0.88),
            19.0 + (245.0 - 19.0) * (norm ** 0.88),
        ], axis=2)
    mono = rgb * (1.0 - bg_weight) + mono_sculpt * bg_weight

    strand_mod = (0.26 + 0.74 * fs) * (1.0 - 0.58 * sulc)
    vol_shade = np.clip(0.30 + 0.70 * hv * (0.45 + 0.55 * fs), 0.0, 1.0)[:, :, None]

    if is_wrong:
        fe = np.clip((rb_diff - 12.0) / 68.0, 0.0, 1.0)
        hc = np.clip((gb_diff - 10.0) / 78.0, 0.0, 1.0)
        fiber_alpha = np.clip(fe * (0.74 + 0.62 * strand_mod), 0.0, 0.97)[:, :, None]
        ridge_boost = np.clip(fs * 0.68 + hc * 0.58, 0.0, 1.0)[:, :, None]

        deep_col = np.array([158.0, 14.0, 18.0], dtype=np.float32)
        mid_col = np.array([248.0, 38.0, 46.0], dtype=np.float32)
        crest_col = np.array([255.0, 200.0, 140.0], dtype=np.float32)
    else:
        fe = np.clip((np.maximum(gb_diff, rb_diff * 0.95) - 11.0) / 66.0, 0.0, 1.0)
        hc = np.clip((np.maximum(gr_diff, gb_diff * 0.88) - 8.0) / 66.0, 0.0, 1.0)
        fiber_alpha = np.clip(fe * (0.76 + 0.62 * strand_mod), 0.0, 0.97)[:, :, None]
        ridge_boost = np.clip(fs * 0.70 + hc * 0.58, 0.0, 1.0)[:, :, None]

        deep_col = np.array([6.0, 114.0, 52.0], dtype=np.float32)
        mid_col = np.array([18.0, 224.0, 120.0], dtype=np.float32)
        crest_col = np.array([196.0, 255.0, 222.0], dtype=np.float32)

    fiber_rgb = (
        deep_col * (1.0 - vol_shade)
        + mid_col * vol_shade * (1.0 - ridge_boost * 0.62)
        + crest_col * (ridge_boost ** 1.30) * 0.90
    )
    out_rgb = mono * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha
    return np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)


def build_seamless_framed_pool(
    raw_frames_720p: list[np.ndarray],
    step_idx: int,
    is_wrong: bool,
    scale: float,
    y_head_target: int,
    y_head_src_720: float,
    x_shift: int = 0,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    1. Applies Day 2 (`v39`) Muscle Fiber & Bone X-Ray shader to each 720x1280 frame.
    2. Places the frame onto the 1080x1920 studio stage with a 100% STATIC, UNIFORM `scale` (`scale_x == scale_y`)
       so `head_top >= 435` (below the Top Preview Dock `y <= 406`) and `foot_bot <= 1275` (above Bottom HUD `y >= 1315`),
       with `0.00` background pixel diff across the entire step!
    3. Extracts and Gaussian-smooths anatomical keypoints (`sh`, `hip_r`, `hip_l`, `kn_r`, `kn_l`, `ank_r`, `ank_l`, `head_top`, `foot_bot`).
    """
    sh, sw = raw_frames_720p[0].shape[:2]  # 1280, 720
    nw = int(round(sw * scale))
    nh = int(round(sh * scale))
    x0 = (W - nw) // 2 + x_shift
    y0 = int(round(y_head_target - y_head_src_720 * scale))
    y_end = y0 + nh

    shaded_720 = [apply_day2_v39_anatomy_shader_720p(fr, step_idx, is_wrong) for fr in raw_frames_720p]
    fr0_res = cv2.resize(shaded_720[0], (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    bg_color = np.array([10.0, 11.0, 14.0], dtype=np.float32)
    bg_canvas = np.broadcast_to(bg_color, (H, W, 3)).copy()

    xx_sub = np.arange(nw, dtype=np.float32)[None, :]
    ax_l = np.clip(xx_sub / 46.0, 0.0, 1.0)
    ax_r = np.clip((nw - 1.0 - xx_sub) / 46.0, 0.0, 1.0)
    ax_1d = (0.5 - 0.5 * np.cos(np.pi * (ax_l * ax_r)))[:, :, None]

    top_strip = np.mean(fr0_res[4:18, :, :].astype(np.float32), axis=0, keepdims=True)
    cx_norm = (np.arange(nw, dtype=np.float32) / float(nw))[None, :, None]
    rack_only_mask = np.clip(np.abs(cx_norm - 0.50) - 0.18, 0.0, 0.12) / 0.12
    top_strip_clean = top_strip * rack_only_mask + bg_color[None, None, :] * (1.0 - rack_only_mask)

    bot_strip = np.mean(fr0_res[nh - 18 : nh - 4, :, :].astype(np.float32), axis=0, keepdims=True)

    if y0 > 0:
        yy_top = np.arange(y0, dtype=np.float32)[:, None, None]
        fade_top = np.clip((yy_top - 110.0) / max(1.0, float(y0 - 110.0)), 0.0, 1.0) ** 1.4
        ext_top = top_strip_clean * fade_top + bg_color[None, None, :] * (1.0 - fade_top)
        bg_canvas[0:y0, x0 : x0 + nw] = ext_top * ax_1d + bg_canvas[0:y0, x0 : x0 + nw] * (1.0 - ax_1d)

    if y_end < H:
        yy_bot = np.arange(H - y_end, dtype=np.float32)[:, None, None]
        fade_bot = np.clip(1.0 - yy_bot / 210.0, 0.0, 1.0) ** 1.3
        ext_bot = bot_strip * fade_bot + bg_color[None, None, :] * (1.0 - fade_bot)
        bg_canvas[y_end:H, x0 : x0 + nw] = ext_bot * ax_1d + bg_canvas[y_end:H, x0 : x0 + nw] * (1.0 - ax_1d)

    yy_in = np.arange(nh, dtype=np.float32)[:, None]
    ay_top = np.clip(yy_in / 26.0, 0.0, 1.0)
    ay_bot = np.clip((nh - 1.0 - yy_in) / 34.0, 0.0, 1.0)
    ay_1d = (0.5 - 0.5 * np.cos(np.pi * (ay_top * ay_bot)))[:, :, None]
    alpha_full = ax_1d * ay_1d

    for r_i in range(nh):
        t_r = r_i / float(max(1, nh - 1))
        top_row = top_strip_clean[0] * ax_1d[0] + bg_canvas[y0 + r_i, x0 : x0 + nw] * (1.0 - ax_1d[0])
        bot_row = bot_strip[0] * ax_1d[0] + bg_canvas[y0 + r_i, x0 : x0 + nw] * (1.0 - ax_1d[0])
        bg_canvas[y0 + r_i, x0 : x0 + nw] = (1.0 - t_r) * top_row + t_r * bot_row

    detector = uar.get_pose_detector()
    out_rgb_list = []
    raw_kps = []

    for idx_f, s_rgb in enumerate(shaded_720):
        res_rgb = cv2.resize(s_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        canvas = bg_canvas.copy()
        canvas[y0:y_end, x0 : x0 + nw] = res_rgb * alpha_full + canvas[y0:y_end, x0 : x0 + nw] * (1.0 - alpha_full)
        stage_rgb = np.clip(canvas, 0, 255).astype(np.uint8)
        out_rgb_list.append(stage_rgb)

        # Detect pose on raw_frames_720p[idx_f] for maximum MediaPipe accuracy, then map via (x0 + x*scale, y0 + y*scale)
        raw_rgb_720 = cv2.cvtColor(raw_frames_720p[idx_f], cv2.COLOR_BGR2RGB)
        mp_res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=raw_rgb_720))

        def M(nx_norm: float, ny_norm: float) -> tuple[float, float]:
            return (float(x0 + (nx_norm * sw) * scale), float(y0 + (ny_norm * sh) * scale))

        if mp_res.pose_landmarks:
            lms = mp_res.pose_landmarks[0]
            nose_p = M(lms[0].x, lms[0].y)
            if step_idx in (1, 2):
                sh_p = M(0.5 * (lms[11].x + lms[12].x), 0.5 * (lms[11].y + lms[12].y))
            else:
                # In 3/4 side view, use mid-thoracic/shoulder pivot along the spine
                sh_p = M(0.65 * lms[12].x + 0.35 * lms[11].x - 0.045, 0.5 * (lms[12].y + lms[11].y) + 0.01)
            hip_r = M(lms[24].x, lms[24].y)
            hip_l = M(lms[23].x, lms[23].y)
            kn_r = M(lms[26].x, lms[26].y)
            kn_l = M(lms[25].x, lms[25].y)
            ank_r = M(lms[28].x, lms[28].y)
            ank_l = M(lms[27].x, lms[27].y)
            head_top_y = nose_p[1] - 0.82 * (sh_p[1] - nose_p[1])
            foot_bot_y = max(
                M(lms[29].x, lms[29].y)[1],
                M(lms[30].x, lms[30].y)[1],
                M(lms[31].x, lms[31].y)[1],
                M(lms[32].x, lms[32].y)[1],
            ) + 12.0
        else:
            assert len(raw_kps) > 0, f"MediaPipe failed on frame 0 of step {step_idx}"
            raw_kps.append(dict(raw_kps[-1]))
            continue

        raw_kps.append({
            "sh": sh_p,
            "hip": hip_r,
            "hip_r": hip_r,
            "hip_l": hip_l,
            "kn_r": kn_r,
            "kn_l": kn_l,
            "ank_r": ank_r,
            "ank_l": ank_l,
            "head_top": (nose_p[0], head_top_y),
            "foot_bot": (ank_r[0], foot_bot_y),
        })

    # Smooth keypoints temporally across the pool
    n = len(raw_kps)
    smooth_kps = [dict() for _ in range(n)]
    for k in raw_kps[0].keys():
        arr = np.array([rk[k] for rk in raw_kps], dtype=np.float32)
        ax = gaussian_filter1d(arr[:, 0], sigma=2.2, mode="nearest")
        ay = gaussian_filter1d(arr[:, 1], sigma=2.2, mode="nearest")
        for i in range(n):
            smooth_kps[i][k] = (float(ax[i]), float(ay[i]))

    return out_rgb_list, smooth_kps


def get_or_build_v55_pools():
    if os.path.exists(CACHE_V55_PKL):
        with open(CACHE_V55_PKL, "rb") as f:
            return pickle.load(f)

    print("Building v55 unwarped safe-zone 3D pools...")
    s1_raw = _load_slice("/workspace/daily_info_reels/output/day3_v55_step1_valgus_to_stand.mp4", 64, 24)
    s2_raw = _load_slice("/workspace/daily_info_reels/output/day3_v43_step2_front_fix.mp4", 94, 138)
    s3_raw = _load_slice("/workspace/daily_info_reels/output/day3_v43_step3_side_wrong.mp4", 75, 125)
    s4_raw = _load_slice("/workspace/daily_info_reels/output/day3_v43_step4_side_fix.mp4", 98, 25)

    pools = {
        1: build_seamless_framed_pool(s1_raw, 1, True,  scale=0.77, y_head_target=436, y_head_src_720=10.0,  x_shift=-16),
        2: build_seamless_framed_pool(s2_raw, 2, False, scale=0.75, y_head_target=436, y_head_src_720=32.0,  x_shift=+16),
        3: build_seamless_framed_pool(s3_raw, 3, True,  scale=0.76, y_head_target=436, y_head_src_720=62.0,  x_shift=-22),
        4: build_seamless_framed_pool(s4_raw, 4, False, scale=0.80, y_head_target=436, y_head_src_720=128.0, x_shift=+14),
    }
    with open(CACHE_V55_PKL, "wb") as f:
        pickle.dump(pools, f)
    print("Saved:", CACHE_V55_PKL)
    return pools


def draw_smooth_segment_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    half_w_ss: float,
    rgba: tuple[int, int, int, int],
):
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1.0:
        return
    nx, ny = -dy / L, dx / L
    poly = [
        (x0 + nx * half_w_ss, y0 + ny * half_w_ss),
        (x1 + nx * half_w_ss, y1 + ny * half_w_ss),
        (x1 - nx * half_w_ss, y1 - ny * half_w_ss),
        (x0 - nx * half_w_ss, y0 - ny * half_w_ss),
    ]
    rd.polygon(poly, fill=rgba)


def draw_demic_glowing_bone_axis_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    trim_ends_1x: float = 9.5,
):
    """
    Draws `@demicstory`'s signature glowing anatomical bone-axis alignment line between two `◎` joint pivots:
    - Outer neon bloom halo + dark contrast edge + saturated crimson/emerald core + thin white specular center hairline.
    """
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    trim = trim_ends_1x * ss
    if L <= 2.0 * trim + 4.0:
        return
    ux, uy = dx / L, dy / L
    pa = (x0 + ux * trim, y0 + uy * trim)
    pb = (x1 - ux * trim, y1 - uy * trim)

    draw_smooth_segment_ss(rd, pa, pb, 5.2 * ss, (*col_main, 68))
    draw_smooth_segment_ss(rd, pa, pb, 3.3 * ss, (0, 0, 0, 215))
    draw_smooth_segment_ss(rd, pa, pb, 2.2 * ss, (*col_main, 252))
    draw_smooth_segment_ss(rd, pa, pb, 0.8 * ss, (*col_light, 235))


def draw_plumb_reference_line_ss(
    rd: ImageDraw.ImageDraw,
    p0_ss: tuple[float, float],
    p1_ss: tuple[float, float],
    ss: int = 4,
    dashed: bool = True,
):
    if not dashed:
        draw_smooth_segment_ss(rd, p0_ss, p1_ss, 2.2 * ss, (0, 0, 0, 175))
        draw_smooth_segment_ss(rd, p0_ss, p1_ss, 1.2 * ss, (*COL_PLUMB, 210))
        return
    x0, y0 = p0_ss
    x1, y1 = p1_ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 8.0:
        return
    n_dashes = max(4, int(L // (16.0 * ss)))
    for i in range(n_dashes):
        t0 = i / float(n_dashes)
        t1 = min(1.0, (i + 0.58) / float(n_dashes))
        pa = (x0 + dx * t0, y0 + dy * t0)
        pb = (x0 + dx * t1, y0 + dy * t1)
        draw_smooth_segment_ss(rd, pa, pb, 2.1 * ss, (0, 0, 0, 175))
        draw_smooth_segment_ss(rd, pa, pb, 1.15 * ss, (*COL_PLUMB, 215))


def draw_day2_angle_sector_ss(
    rd: ImageDraw.ImageDraw,
    vertex_ss: tuple[float, float],
    ray1_pt_ss: tuple[float, float],
    ray2_pt_ss: tuple[float, float],
    radius_1x: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    show_arrow_at_ray2: bool = False,
):
    """
    Day 2 (`v39`) Exterior Solid Angle Sector & Smooth Closed-Polygon Annular Arc (`SS=4`).
    """
    vx, vy = vertex_ss
    r = radius_1x * ss
    a1 = math.atan2(ray1_pt_ss[1] - vy, ray1_pt_ss[0] - vx)
    a2 = math.atan2(ray2_pt_ss[1] - vy, ray2_pt_ss[0] - vx)
    diff = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(diff) < 0.03:
        return

    steps = 64
    poly = [(vx, vy)]
    for i in range(steps + 1):
        a = a1 + diff * (i / float(steps))
        poly.append((vx + math.cos(a) * r, vy + math.sin(a) * r))
    rd.polygon(poly, fill=(*col_main, 58))

    direction = 1.0 if diff >= 0.0 else -1.0
    head_len = 20.0 * ss if show_arrow_at_ray2 else 0.0
    head_half_w = 11.5 * ss
    d_theta = (head_len / max(36.0 * ss, r)) * direction
    a_end = a1 + diff - d_theta

    for pad, rgba in [(1.7 * ss, (0, 0, 0, 225)), (0.0, (*col_main, 255))]:
        hw_arc = 2.1 * ss + pad
        outer_pts, inner_pts = [], []
        for i in range(steps + 1):
            a = a1 + (a_end - a1) * (i / float(steps))
            ca, sa = math.cos(a), math.sin(a)
            outer_pts.append((vx + ca * (r + hw_arc), vy + sa * (r + hw_arc)))
            inner_pts.append((vx + ca * (r - hw_arc), vy + sa * (r - hw_arc)))

        if show_arrow_at_ray2:
            ux = -math.sin(a_end) * direction
            uy = math.cos(a_end) * direction
            rx_n, ry_n = math.cos(a_end), math.sin(a_end)
            nbx, nby = vx + rx_n * r, vy + ry_n * r
            hw_h = head_half_w + pad * 1.15
            p_outer_wing = (nbx - ux * (2.5 * ss) + rx_n * hw_h, nby - uy * (2.5 * ss) + ry_n * hw_h)
            p_inner_wing = (nbx - ux * (2.5 * ss) - rx_n * hw_h, nby - uy * (2.5 * ss) - ry_n * hw_h)
            tx = vx + math.cos(a1 + diff) * r + ux * pad * 1.2
            ty = vy + math.sin(a1 + diff) * r + uy * pad * 1.2
            ring_poly = outer_pts + [p_outer_wing, (tx, ty), p_inner_wing] + inner_pts[::-1]
        else:
            ring_poly = outer_pts + inner_pts[::-1]
        rd.polygon(ring_poly, fill=rgba)

    hw_spec = 0.7 * ss
    spec_out, spec_in = [], []
    for i in range(steps + 1):
        a = a1 + (a_end - a1) * (i / float(steps))
        ca, sa = math.cos(a), math.sin(a)
        spec_out.append((vx + ca * (r + hw_spec), vy + sa * (r + hw_spec)))
        spec_in.append((vx + ca * (r - hw_spec), vy + sa * (r - hw_spec)))
    rd.polygon(spec_out + spec_in[::-1], fill=(*col_light, 225))


def draw_joint_pivot_node_ss(
    rd: ImageDraw.ImageDraw,
    pt_ss: tuple[float, float],
    radius_1x: float,
    col_main: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
    is_primary: bool = True,
):
    cx, cy = pt_ss
    r = radius_1x * ss
    rd.ellipse([cx - r - 3 * ss, cy - r - 3 * ss, cx + r + 3 * ss, cy + r + 3 * ss], fill=(*col_main, 65))
    rd.ellipse([cx - r - 1.5 * ss, cy - r - 1.5 * ss, cx + r + 1.5 * ss, cy + r + 1.5 * ss], fill=(0, 0, 0, 210))
    rd.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=(*col_main, 255),
        width=int(round((2.6 if is_primary else 2.0) * ss)),
    )
    r_dot = (3.4 if is_primary else 2.5) * ss
    rd.ellipse([cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot], fill=(*col_light, 255))


def draw_3d_floor_lock_bar_ss(
    rd: ImageDraw.ImageDraw,
    ank_r_1x: tuple[float, float],
    ank_l_1x: tuple[float, float] | None,
    y_floor_1x: float,
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_light: tuple[int, int, int],
    is_lifting: bool = False,
    ss: int = 4,
):
    """
    Draws `@demicstory`'s signature glowing 3D floor-plane anchor pad & foot-stance laser bar.
    """
    anchors = [ank_r_1x] if ank_l_1x is None else [ank_r_1x, ank_l_1x]
    for ax_1x, _ in anchors:
        cx, cy = ax_1x * ss, y_floor_1x * ss
        rx_s, ry_s = 46.0 * ss, 12.0 * ss
        rd.ellipse([cx - rx_s * 1.05, cy - ry_s * 1.05 + 2 * ss, cx + rx_s * 1.05, cy + ry_s * 1.05 + 2 * ss], fill=(0, 0, 0, 145))
        rd.ellipse(
            [cx - rx_s, cy - ry_s, cx + rx_s, cy + ry_s],
            fill=(*col_main, 42),
            outline=(*col_main, 235 if not is_lifting else 180),
            width=int(2.2 * ss),
        )
        # Glowing horizontal floor bar flush under the sole (`@demicstory` signature)
        p0 = ((ax_1x - 52.0) * ss, y_floor_1x * ss)
        p1 = ((ax_1x + 52.0) * ss, y_floor_1x * ss)
        if is_lifting:
            draw_plumb_reference_line_ss(rd, p0, p1, ss=ss, dashed=True)
        else:
            draw_smooth_segment_ss(rd, p0, p1, 4.5 * ss, (*col_main, 70))
            draw_smooth_segment_ss(rd, p0, p1, 2.4 * ss, (*col_main, 250))
            draw_smooth_segment_ss(rd, p0, p1, 0.9 * ss, (*col_light, 235))


def draw_3d_sculpted_vertical_ribbon_arrow_ss(
    rd: ImageDraw.ImageDraw,
    p_start_1x: tuple[float, float],
    p_end_1x: tuple[float, float],
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    col_light: tuple[int, int, int],
    ss: int = 4,
):
    """
    Draws a sculpted 3D bevel ribbon arrow for vertical drive / heel lift cues (`SS=4`),
    with 3D drop shadow, dark bevel side wall, Phong gradient core, specular edge highlight,
    and two-facet 3D arrowhead.
    """
    x0, y0 = p_start_1x[0] * ss, p_start_1x[1] * ss
    x1, y1 = p_end_1x[0] * ss, p_end_1x[1] * ss
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 16.0 * ss:
        return
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    head_len = 22.0 * ss
    head_hw = 13.0 * ss
    w0 = 3.2 * ss
    w1 = 5.6 * ss
    bx, by = x1 - ux * head_len, y1 - uy * head_len
    notch_x, notch_y = x1 - ux * (head_len * 0.76), y1 - uy * (head_len * 0.76)

    # 1. Soft 3D drop shadow
    sh_dx, sh_dy = 3.0 * ss, 4.0 * ss
    rd.polygon([
        (x0 + nx * w0 + sh_dx, y0 + ny * w0 + sh_dy),
        (bx + nx * w1 + sh_dx, by + ny * w1 + sh_dy),
        (bx + nx * head_hw + sh_dx, by + ny * head_hw + sh_dy),
        (x1 + sh_dx, y1 + sh_dy),
        (bx - nx * head_hw + sh_dx, by - ny * head_hw + sh_dy),
        (bx - nx * w1 + sh_dx, by - ny * w1 + sh_dy),
        (x0 - nx * w0 + sh_dx, y0 - ny * w0 + sh_dy),
    ], fill=(0, 0, 0, 175))

    # 2. Dark 3D extruded bevel rim
    pad = 1.8 * ss
    rd.polygon([
        (x0 + nx * (w0 + pad), y0 + ny * (w0 + pad)),
        (bx + nx * (w1 + pad), by + ny * (w1 + pad)),
        (bx + nx * (head_hw + pad), by + ny * (head_hw + pad)),
        (x1 + ux * pad, y1 + uy * pad),
        (bx - nx * (head_hw + pad), by - ny * (head_hw + pad)),
        (bx - nx * (w1 + pad), by - ny * (w1 + pad)),
        (x0 - nx * (w0 + pad), y0 - ny * (w0 + pad)),
    ], fill=(*col_dark, 250))

    # 3. Left lit facet & Right main facet on shaft + head
    lit_col = (
        int(col_light[0] * 0.55 + 255 * 0.45),
        int(col_light[1] * 0.55 + 255 * 0.45),
        int(col_light[2] * 0.55 + 255 * 0.45),
        255,
    )
    rd.polygon([(x0, y0), (bx, by), (bx + nx * w1, by + ny * w1), (x0 + nx * w0, y0 + ny * w0)], fill=(*col_light, 245))
    rd.polygon([(x0, y0), (bx, by), (bx - nx * w1, by - ny * w1), (x0 - nx * w0, y0 - ny * w0)], fill=(*col_main, 255))

    rd.polygon([(x1, y1), (bx + nx * head_hw, by + ny * head_hw), (notch_x, notch_y)], fill=lit_col)
    rd.polygon([(x1, y1), (notch_x, notch_y), (bx - nx * head_hw, by - ny * head_hw)], fill=(*col_main, 255))
    rd.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, 245), width=max(2, int(1.1 * ss)))


def draw_day2_pointer_and_callout(
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
    Day 2 (`v39`) slender architectural pointer line (`joint ◎ ── elbow ── dot`) +
    Pure White (`#FFFFFF`) `64pt` number & `22pt` sub-label clamped strictly inside `[96 .. 978]`,
    with ZERO underline bars.
    """
    # 1. Slender architectural pointer line
    p_j = (int(round(joint_pt[0])), int(round(joint_pt[1])))
    p_e = (int(round(elbow_pt[0])), int(round(elbow_pt[1])))
    sd.line([p_j, p_e], fill=(0, 0, 0, 215), width=4)
    od.line([p_j, p_e], fill=(*col_main, 230), width=2)
    od.ellipse([p_e[0] - 3, p_e[1] - 3, p_e[0] + 3, p_e[1] + 3], fill=(255, 255, 255, 245))

    # 2. Pure White Callout Typography
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
        sd.text((nx + dx_s, ay + dy_s), num_str, font=F_CALLOUT_NUM, fill=(0, 0, 0, 240))
        sd.text((tx + dx_s, ty + dy_s), sub_str, font=F_CALLOUT_SUB, fill=(0, 0, 0, 240))

    od.text((nx, ay), num_str, font=F_CALLOUT_NUM, fill=COL_WHITE)
    od.text((tx, ty), sub_str, font=F_CALLOUT_SUB, fill=COL_WHITE)


def render_v55_demic_day2_callouts(
    canvas: Image.Image,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 1.0,
):
    """
    Renders the complete `@demicstory` + Day 2 (`v39`) + `v48` 3D biomechanical overlay system in `SS=4`:
    - Glowing anatomical bone-axis alignment chains (`Hip ◎ — Knee ◎ — Ankle ◎` & `Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`),
    - Day 2 (`v39`) exterior solid angle sectors (`14°`, `0°`, `+28°`, `95°`),
    - Compact, body-hugging `v48` 3D volumetric curved ribbon arrows (`SS=4`),
    - Slender exterior pointer lines to Pure White `64pt`/`22pt` callouts.
    """
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
    col_dark = COL_RED_DARK if is_wrong else COL_GREEN_DARK
    col_light = COL_RED_LIGHT if is_wrong else COL_GREEN_LIGHT

    sh = kp["sh"]
    hip_r = kp["hip_r"]
    hip_l = kp["hip_l"]
    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    ank_r = kp["ank_r"]
    ank_l = kp["ank_l"]

    sweep_k = 0.65 + 0.35 * min(1.0, max(0.0, rep_amount * 1.15))

    if step_idx == 1:
        # Step 1 (Front Wrong — Realistic Knee Valgus `14° VALGUS CAVE`)
        # 1. Floor reference pads under both ankles
        draw_3d_floor_lock_bar_ss(rd, ank_r, ank_l, ank_r[1] + 22.0, col_main, col_dark, col_light, is_lifting=False, ss=SS)

        # 2. Vertical Ideal Plumb Alignment Lines rising from ankles (`ank_r`, `ank_l`)
        plumb_r = (ank_r[0], hip_r[1] + 15.0)
        plumb_l = (ank_l[0], hip_l[1] + 15.0)
        draw_plumb_reference_line_ss(rd, S(ank_r), S(plumb_r), SS, dashed=True)
        draw_plumb_reference_line_ss(rd, S(ank_l), S(plumb_l), SS, dashed=True)

        # 3. Day 2 Angle Sector showing inward valgus drift from vertical plumb line
        draw_day2_angle_sector_ss(rd, S(ank_r), S(plumb_r), S(kn_r), 108.0, col_main, col_light, SS, show_arrow_at_ray2=False)
        draw_day2_angle_sector_ss(rd, S(ank_l), S(plumb_l), S(kn_l), 108.0, col_main, col_light, SS, show_arrow_at_ray2=False)

        # 4. `@demicstory` Glowing Crimson-Red Bone-Axis Alignment Lines (`Hip ◎ — Knee ◎ — Ankle ◎`)
        draw_demic_glowing_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(hip_l), S(kn_l), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(kn_l), S(ank_l), col_main, col_light, SS)

        # 5. Compact 3D Volumetric Curved Ribbon Arrows (`╭──▶   ◀──╮`) wrapping around outer knees pointing INWARD
        phi_e_r = 212.0 + (26.0 - 212.0) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_r[0] - 8.0, kn_r[1] + 14.0),
            rx=48.0, ry=18.0, tilt_deg=10.0,
            phi_start_deg=212.0, phi_end_deg=phi_e_r,
            ribbon_thick_px=12.5, pitch_dy=10.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_light, ss=SS,
        )
        phi_e_l = -32.0 + (154.0 - (-32.0)) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_l[0] + 8.0, kn_l[1] + 14.0),
            rx=48.0, ry=18.0, tilt_deg=-10.0,
            phi_start_deg=-32.0, phi_end_deg=phi_e_l,
            ribbon_thick_px=12.5, pitch_dy=10.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_light, ss=SS,
        )

        # 6. Joint Pivot Rings (`◎`)
        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 9.5 if is_p else 7.2, col_main, col_light, SS, is_primary=is_p)

        deg_live = int(round(8 + 6 * rep_amount))
        pct_live = int(round(79 + 5 * rep_amount))
        draw_day2_pointer_and_callout(od, sd, (kn_r[0] - 12, kn_r[1] - 6), (265, 965), (SAFE_X_LEFT, 910), f"{deg_live}°", "VALGUS CAVE", col_main, align="left")
        draw_day2_pointer_and_callout(od, sd, (kn_l[0] + 12, kn_l[1] - 6), (765, 915), (SAFE_X_RIGHT_MAX, 855), f"{pct_live}%", "ACL & MCL LOAD", col_main, align="right")

    elif step_idx == 2:
        # Step 2 (Front Right — Knees Track Wide Over Toes `0° KNEES TRACK TOES`)
        # 1. `@demicstory` Glowing Emerald Floor Tripod Lock Bars under both feet
        draw_3d_floor_lock_bar_ss(rd, ank_r, ank_l, ank_r[1] + 22.0, col_main, col_dark, col_light, is_lifting=False, ss=SS)

        # 2. Day 2 Exterior Solid Angle Sectors at both knees
        draw_day2_angle_sector_ss(rd, S(kn_r), S(hip_r), S(ank_r), 48.0, col_main, col_light, SS, show_arrow_at_ray2=False)
        draw_day2_angle_sector_ss(rd, S(kn_l), S(hip_l), S(ank_l), 48.0, col_main, col_light, SS, show_arrow_at_ray2=False)

        # 3. `@demicstory` Glowing Emerald Bone-Axis Alignment Lines (`Hip ◎ — Knee ◎ — Ankle ◎`)
        draw_demic_glowing_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(hip_l), S(kn_l), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(kn_l), S(ank_l), col_main, col_light, SS)

        # 4. Compact 3D Volumetric Curved Ribbon Arrows (`◀──╭   ╮──▶`) wrapping OUTWARD over the toes
        phi_e_r = -22.0 + (198.0 - (-22.0)) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_r[0] - 6.0, kn_r[1] + 14.0),
            rx=50.0, ry=18.0, tilt_deg=-8.0,
            phi_start_deg=-22.0, phi_end_deg=phi_e_r,
            ribbon_thick_px=12.5, pitch_dy=-8.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_light, ss=SS,
        )
        phi_e_l = 202.0 + (-18.0 - 202.0) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_l[0] + 6.0, kn_l[1] + 14.0),
            rx=50.0, ry=18.0, tilt_deg=8.0,
            phi_start_deg=202.0, phi_end_deg=phi_e_l,
            ribbon_thick_px=12.5, pitch_dy=-8.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_light, ss=SS,
        )

        # 5. Joint Pivot Rings (`◎`)
        for pt_j, is_p in [(hip_r, False), (kn_r, True), (ank_r, False), (hip_l, False), (kn_l, True), (ank_l, False)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 9.5 if is_p else 7.2, col_main, col_light, SS, is_primary=is_p)

        pct_live = int(round(92 + 4 * rep_amount))
        draw_day2_pointer_and_callout(od, sd, (kn_r[0] - 14, kn_r[1] - 6), (265, 965), (SAFE_X_LEFT, 910), "0°", "KNEES TRACK TOES", col_main, align="left")
        draw_day2_pointer_and_callout(od, sd, (kn_l[0] + 14, kn_l[1] - 6), (765, 915), (SAFE_X_RIGHT_MAX, 855), f"{pct_live}%", "VMO & GLUTE LOCK", col_main, align="right")

    elif step_idx == 3:
        # Step 3 (Side Wrong — Realistic Good-Morning Forward Lean `+28°` & Heel Lift)
        floor_y = 1192.0
        draw_3d_floor_lock_bar_ss(rd, ank_r, None, floor_y, col_main, col_dark, col_light, is_lifting=True, ss=SS)

        # 1. Vertical Upright Plumb Reference Line rising from `hip_r`
        plumb_top = (hip_r[0], sh[1] - 35.0)
        draw_plumb_reference_line_ss(rd, S(hip_r), S(plumb_top), SS, dashed=True)

        # 2. Day 2 Exterior Solid Angle Sector (`+28°`) from upright plumb to forward-pitched spine (`hip_r -> sh`)
        spine_len = math.hypot(sh[0] - hip_r[0], sh[1] - hip_r[1])
        draw_day2_angle_sector_ss(
            rd,
            vertex_ss=S(hip_r),
            ray1_pt_ss=S(plumb_top),
            ray2_pt_ss=S(sh),
            radius_1x=min(115.0, spine_len * 0.72),
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrow_at_ray2=True,
        )

        # 3. `@demicstory` Glowing Crimson-Red Spine & Leg Chain (`sh — hip_r — kn_r — ank_r`)
        draw_demic_glowing_bone_axis_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        # 4. Compact 3D Volumetric Curved Ribbon Arrow arching forward over upper back (`╭───▶`)
        phi_e_sh = 212.0 + (22.0 - 212.0) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(sh[0] + 12.0, sh[1] + 8.0),
            rx=56.0, ry=22.0, tilt_deg=48.0,
            phi_start_deg=212.0, phi_end_deg=phi_e_sh,
            ribbon_thick_px=12.5, pitch_dy=16.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_light, ss=SS,
        )

        # 5. Sculpted 3D upward heel-lift arrow showing heel rising off floor (`floor_y -> ank_r`)
        draw_3d_sculpted_vertical_ribbon_arrow_ss(
            rd,
            p_start_1x=(ank_r[0] - 34.0, floor_y - 4.0),
            p_end_1x=(ank_r[0] - 34.0, ank_r[1] - 18.0),
            col_main=col_main, col_dark=col_dark, col_light=col_light, ss=SS,
        )

        # 6. Joint Pivot Rings (`◎`)
        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 9.5 if is_p else 7.5, col_main, col_light, SS, is_primary=is_p)

        deg_live = int(round(14 + 14 * rep_amount))
        pct_live = int(round(74 + 5 * rep_amount))
        draw_day2_pointer_and_callout(od, sd, (hip_r[0] - 12, hip_r[1] + 6), (265, 955), (SAFE_X_LEFT, 900), f"+{deg_live}°", "HEELS LIFTING", col_main, align="left")
        draw_day2_pointer_and_callout(od, sd, (kn_r[0] + 12, kn_r[1] - 6), (765, 915), (SAFE_X_RIGHT_MAX, 855), f"{pct_live}%", "LUMBAR & PATELLA", col_main, align="right")

    else:
        # Step 4 (Side Right — Proud Upright Chest, Planted Heels & `95°` Parallel Drive)
        floor_y = ank_r[1] + 24.0
        draw_3d_floor_lock_bar_ss(rd, ank_r, None, floor_y, col_main, col_dark, col_light, is_lifting=False, ss=SS)

        # 1. Day 2 Exterior Solid Angle Sector (`95°` Hip-to-Thigh Parallel Depth) at `hip_r`
        draw_day2_angle_sector_ss(
            rd,
            vertex_ss=S(hip_r),
            ray1_pt_ss=S(sh),
            ray2_pt_ss=S(kn_r),
            radius_1x=78.0,
            col_main=col_main,
            col_light=col_light,
            ss=SS,
            show_arrow_at_ray2=False,
        )

        # 2. `@demicstory` Glowing Emerald Biomechanical Alignment Chain (`sh — hip_r — kn_r — ank_r`)
        draw_demic_glowing_bone_axis_ss(rd, S(sh), S(hip_r), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(hip_r), S(kn_r), col_main, col_light, SS)
        draw_demic_glowing_bone_axis_ss(rd, S(kn_r), S(ank_r), col_main, col_light, SS)

        # 3. Sculpted 3D Upward Drive Ribbon Arrow positioned cleanly BESIDE the barbell/shoulder (`x = sh[0] - 88`, 0% overlap with face/head!)
        drive_x = hip_r[0] - 58.0
        draw_3d_sculpted_vertical_ribbon_arrow_ss(
            rd,
            p_start_1x=(drive_x, sh[1] + 45.0),
            p_end_1x=(drive_x, sh[1] - 65.0),
            col_main=col_main, col_dark=col_dark, col_light=col_light, ss=SS,
        )

        # 4. Sculpted 3D Downward Heel-Lock Arrow onto the emerald floor bar
        draw_3d_sculpted_vertical_ribbon_arrow_ss(
            rd,
            p_start_1x=(ank_r[0] - 36.0, ank_r[1] - 42.0),
            p_end_1x=(ank_r[0] - 36.0, floor_y - 4.0),
            col_main=col_main, col_dark=col_dark, col_light=col_light, ss=SS,
        )

        # 5. Joint Pivot Rings (`◎`)
        for pt_j, is_p in [(sh, True), (hip_r, True), (kn_r, True), (ank_r, True)]:
            draw_joint_pivot_node_ss(rd, S(pt_j), 9.5 if is_p else 7.5, col_main, col_light, SS, is_primary=is_p)

        deg_live = int(round(125 - 30 * rep_amount))
        pct_live = int(round(94 + 4 * rep_amount))
        draw_day2_pointer_and_callout(od, sd, (hip_r[0] - 12, hip_r[1] + 6), (265, 955), (SAFE_X_LEFT, 900), f"{deg_live}°", "PARALLEL DEPTH", col_main, align="left")
        draw_day2_pointer_and_callout(od, sd, (kn_r[0] + 12, kn_r[1] - 6), (765, 915), (SAFE_X_RIGHT_MAX, 855), f"{pct_live}%", "QUADS & GLUTES", col_main, align="right")

    rig_1x = rig_layer.resize((W, H), Image.Resampling.LANCZOS)
    canvas.alpha_composite(rig_1x)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(ov)


def draw_v55_safezone_top_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    exercise_name: str = "BARBELL BACK SQUAT",
    muscle_group: str = "[ LEGS · QUADS & GLUTES ]",
):
    """
    v55 Compact Safe-Zone Top HUD (`x = 96, y = 258..406`):
    - Ends strictly at `y = 406` (`<= 410`), leaving `27px+` of dark studio margin above the
      standing mannequin's head (`head_top >= 433`) so the head is NEVER covered by the preview dock!
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN
    pill_bg = (*COL_RED, 245) if is_wrong else (*COL_GREEN, 245)

    bar_x = SAFE_X_LEFT
    hdr_y = SAFE_Y_TOP  # 258
    hdr_h = 42          # 258..300
    tb_ex = od.textbbox((0, 0), exercise_name, font=F_EX_TITLE)
    ex_w = tb_ex[2] - tb_ex[0]
    tb_mg = od.textbbox((0, 0), muscle_group, font=F_EX_SUB)
    mg_w = tb_mg[2] - tb_mg[0]

    pill_w = mg_w + 20
    hdr_w = 20 + 12 + ex_w + 13 + pill_w + 15

    od.rounded_rectangle([bar_x + 2, hdr_y + 4, bar_x + hdr_w + 2, hdr_y + hdr_h + 4], radius=13, fill=(0, 0, 0, 150))
    od.rounded_rectangle(
        [bar_x, hdr_y, bar_x + hdr_w, hdr_y + hdr_h],
        radius=13,
        fill=(14, 17, 23, 232),
        outline=(255, 255, 255, 44),
        width=1,
    )

    dot_cx = bar_x + 19
    dot_cy = hdr_y + hdr_h // 2
    od.ellipse([dot_cx - 7, dot_cy - 7, dot_cx + 7, dot_cy + 7], fill=(*accent, 95))
    od.ellipse([dot_cx - 4, dot_cy - 4, dot_cx + 4, dot_cy + 4], fill=(*accent, 255))

    ex_x = dot_cx + 14
    od.text((ex_x, hdr_y + 8), exercise_name, font=F_EX_TITLE, fill=(255, 255, 255, 255))

    p_x0 = ex_x + ex_w + 12
    p_y0 = hdr_y + 7
    p_h = hdr_h - 14
    od.rounded_rectangle([p_x0, p_y0, p_x0 + pill_w, p_y0 + p_h], radius=7, fill=pill_bg)
    od.text((p_x0 + 10, p_y0 + 5), muscle_group, font=F_EX_SUB, fill=(255, 255, 255, 255))

    # Compact 3-Stage Thumbnail Dock (`y = 308..406`, height = 98px)
    tw, th = 118, 82
    gap = 9
    pad = 8
    dock_y = hdr_y + hdr_h + 8  # y = 308
    dock_w = 3 * tw + 2 * gap + 2 * pad
    dock_h = th + 2 * pad       # 98px -> ends at y = 406!

    od.rounded_rectangle([bar_x + 2, dock_y + 5, bar_x + dock_w + 2, dock_y + dock_h + 5], radius=18, fill=(0, 0, 0, 135))
    od.rounded_rectangle(
        [bar_x, dock_y, bar_x + dock_w, dock_y + dock_h],
        radius=18,
        fill=(16, 19, 25, 224),
        outline=(255, 255, 255, 38),
        width=1,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = dock_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy().resize((tw, th), Image.Resampling.LANCZOS) if i == active_idx else inact_im.copy().resize((tw, th), Image.Resampling.LANCZOS)

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=11, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], radius=12, outline=(*i_col, 245), width=2)
            od.rounded_rectangle([tx + tw // 2 - 14, ty + th - 5, tx + tw // 2 + 14, ty + th - 2], radius=2, fill=(*i_col, 255))
        else:
            od.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=11, outline=(255, 255, 255, 22), width=1)

    canvas.alpha_composite(ov)


def draw_v55_safezone_bottom_hud(canvas: Image.Image, sp: dict):
    """
    v55 Safe-Zone Bottom-Left HUD (`x = 96, y = 1315..1448`):
    - Starts at `y = 1315` (`>= 1315`, leaving `55px+` above the HUD for the feet at `y <= 1260`).
    - Ends at `y = 1448` (`135px+` above Instagram's `@gymform.3d 팔로우` bar at `y = 1585`).
    """
    is_wrong = sp["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    pill_fill = (*COL_RED, 250) if is_wrong else (*COL_GREEN, 250)
    pct = int(np.clip(sp["pct"], 0, 100))

    tag_pill_str = sp["tag_pill"]
    tag_ctx_str = sp["tag_ctx"]
    cue_1line = sp["cue_1line"]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx = SAFE_X_LEFT  # 96px

    # Row 1 (`y = 1315`)
    py = 1315
    tb_p = od.textbbox((0, 0), tag_pill_str, font=F_TAG_PILL)
    pw = (tb_p[2] - tb_p[0]) + 26
    ph = 32

    sd.rounded_rectangle([bx, py + 3, bx + pw, py + ph + 3], radius=8, fill=(0, 0, 0, 225))
    od.rounded_rectangle([bx, py, bx + pw, py + ph], radius=8, fill=pill_fill, outline=(255, 255, 255, 55), width=1)
    od.text((bx + 13, py + 5), tag_pill_str, font=F_TAG_PILL, fill=(255, 255, 255, 255))

    ctx_x = bx + pw + 13
    ctx_y = py + 5
    sd.text((ctx_x, ctx_y + 3), tag_ctx_str, font=F_TAG_CTX, fill=(0, 0, 0, 245))
    od.text((ctx_x, ctx_y), tag_ctx_str, font=F_TAG_CTX, fill=(228, 236, 248, 255))

    # Row 2 (`y = 1355`)
    by = 1355
    num_str = f"{pct}"
    nb = od.textbbox((0, 0), num_str, font=F_HERO_PCT)
    nw = nb[2] - nb[0]

    sd.text((bx, by + 4), num_str, font=F_HERO_PCT, fill=(0, 0, 0, 245))
    od.text((bx, by), num_str, font=F_HERO_PCT, fill=(255, 255, 255, 255))

    sym_x = bx + nw + 3
    sym_y = by + 11
    sd.text((sym_x, sym_y + 3), "%", font=F_HERO_SYM, fill=(0, 0, 0, 240))
    od.text((sym_x, sym_y), "%", font=F_HERO_SYM, fill=(*accent, 255))
    sb = od.textbbox((0, 0), "%", font=F_HERO_SYM)
    sw = sb[2] - sb[0]

    div_x = sym_x + sw + 18
    div_y0 = by + 16
    div_y1 = by + 64
    sd.rounded_rectangle([div_x, div_y0 + 2, div_x + 3, div_y1 + 2], radius=2, fill=(0, 0, 0, 225))
    od.rounded_rectangle([div_x, div_y0, div_x + 3, div_y1], radius=2, fill=(*accent, 255))

    tx = div_x + 18
    ty = by + 17
    tb_c = od.textbbox((0, 0), cue_1line, font=F_SINGLE_LINE)
    cw = tb_c[2] - tb_c[0]

    sd.text((tx, ty + 4), cue_1line, font=F_SINGLE_LINE, fill=(0, 0, 0, 245))
    od.text((tx, ty), cue_1line, font=F_SINGLE_LINE, fill=(255, 255, 255, 255))

    # Row 3 (`y = 1442`)
    bar_x0 = bx + 2
    bar_y0 = by + 87
    total_row_w = (tx + cw) - bar_x0
    bar_w = max(360, min(520, total_row_w))
    bar_h = 5
    fill_w = max(14, int(round(bar_w * (pct / 100.0))))

    sd.rounded_rectangle([bar_x0, bar_y0 + 2, bar_x0 + bar_w, bar_y0 + bar_h + 2], radius=3, fill=(0, 0, 0, 215))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], radius=3, fill=(255, 255, 255, 42))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y0 + bar_h], radius=3, fill=(*accent, 248))

    dot_x = bar_x0 + fill_w
    dot_y = bar_y0 + bar_h * 0.5
    sd.ellipse([dot_x - 8, dot_y - 8, dot_x + 8, dot_y + 8], fill=(*accent, 175))
    od.ellipse([dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5], fill=(255, 255, 255, 255))

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
    canvas.alpha_composite(ov)


def make_torso_icons(pools: dict) -> list[tuple[Image.Image, Image.Image]]:
    torso_icons = []
    for s_idx in [1, 3, 4]:
        p_pool = pools[s_idx][0]
        mid_rgb = p_pool[-1]
        crop = mid_rgb[520:1240, 260:820]
        pil_act = Image.fromarray(crop).resize((118, 82), Image.Resampling.LANCZOS).convert("RGBA")
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        dark = np.stack([gray * 0.48, gray * 0.50, gray * 0.54], axis=2).astype(np.uint8)
        pil_inact = Image.fromarray(dark).resize((118, 82), Image.Resampling.LANCZOS).convert("RGBA")
        torso_icons.append((pil_act, pil_inact))
    return torso_icons


def render_iphone_safezone_sheet(step_frames_rgb: dict, out_path: str):
    """
    Overlays the actual iPhone Instagram Reels 19.5:9 UI mask (`v39.simulate_iphone_ig_reels_view`)
    onto all 4 steps and saves `day3_v55_iphone_safezone_preview.jpg`.
    """
    real_shot_bgr = cv2.imread(v39.USER_SCREENSHOT_PATH)
    cap_old = cv2.VideoCapture(os.path.join(OUT_DIR, "day2_v38_master_reel.mp4"))
    cap_old.set(cv2.CAP_PROP_POS_FRAMES, 210)
    _, old_f210_bgr = cap_old.read()
    cap_old.release()

    cw, ch = 420, 910
    ipad = 24
    board = Image.new("RGB", (cw * 4 + ipad * 5, ch + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_hdr = v30.f_pretendard(20, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 3 · iPHONE INSTAGRAM REELS UI SAFE-ZONE PREVIEW (`v55`)  ·  상단 프리뷰(y<=406) / 머리(y>=430) / 발끝(y<=1280) / 하단 HUD(y>=1315) 간섭 0%",
        font=f_hdr,
        fill=COL_GREEN,
    )
    titles = [
        "iPhone View · STEP 1 (14° KNEE VALGUS)",
        "iPhone View · STEP 2 (0° KNEES TRACK TOES)",
        "iPhone View · STEP 3 (+28° FORWARD LEAN)",
        "iPhone View · STEP 4 (95° PARALLEL DRIVE)",
    ]
    for i in range(4):
        x = ipad + i * (cw + ipad)
        col = COL_RED if i in (0, 2) else COL_GREEN
        bd.text((x + 4, 58), titles[i], font=f_col, fill=col)
        sim_im = v39.simulate_iphone_ig_reels_view(step_frames_rgb[i + 1], real_shot_bgr, old_f210_bgr)
        rc = sim_im.resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        board.paste(rc, (x, 86))
    board.save(out_path, quality=95)


def main():
    pools = get_or_build_v55_pools()
    torso_icons = make_torso_icons(pools)

    step_configs = [
        (1, pools[1][0], pools[1][1], True,  0, 84, "LUMBAR & KNEE RISK",       "BARBELL BACK SQUAT",      "14° KNEE VALGUS"),
        (2, pools[2][0], pools[2][1], False, 2, 96, "NEUTRAL KNEE LOCK",        "GLUTE MEDIUS & VMO",      "0° KNEES TRACK TOES"),
        (3, pools[3][0], pools[3][1], True,  1, 79, "ANTERIOR FORCE LEAK",      "HEEL LIFT & FORWARD DIVE","+28° FORWARD LEAN"),
        (4, pools[4][0], pools[4][1], False, 2, 98, "FULL QUAD & GLUTE · 4×10", "QUADRICEPS & GLUTES",     "95° PARALLEL DRIVE"),
    ]

    panels = []
    crops = []
    step_frames_pil = {}

    for s_idx, pool, kps, is_wrong, dock_idx, pct_val, tag_pill, tag_ctx, cue_1line in step_configs:
        idx = len(pool) - 1  # Peak squat depth frame (`u = 1.0`)
        rgb_frame = pool[idx].copy()
        kp_cur = kps[idx]

        canvas = Image.fromarray(rgb_frame).convert("RGBA")
        render_v55_demic_day2_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)
        draw_v55_safezone_top_dock(canvas, torso_icons, active_idx=dock_idx, is_wrong=is_wrong)
        draw_v55_safezone_bottom_hud(
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
        # Full-width x=0..1080 crop of the stage (`y=415..1310`) so ZERO callout text is ever clipped!
        crops.append(out_bgr[415:1310, 0:1080])

    diff_path = os.path.join(OUT_DIR, "day3_v55_difference_sheet.jpg")
    close_path = os.path.join(OUT_DIR, "day3_v55_closeup_sheet.jpg")
    safe_path = os.path.join(OUT_DIR, "day3_v55_iphone_safezone_preview.jpg")

    cv2.imwrite(diff_path, np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(close_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    render_iphone_safezone_sheet(step_frames_pil, safe_path)
    print("Saved v55 preview sheets:", diff_path, close_path, safe_path)


if __name__ == "__main__":
    main()
