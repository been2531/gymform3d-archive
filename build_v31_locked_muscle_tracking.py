#!/usr/bin/env python3
"""
v31 Motion-Locked 3D Striated Muscle Mask & Pure Graphic Master
Fixes:
1. Eliminates dark inter-muscular wire cracks ('철심') and synthetic groove lines while preserving 100% sharp 3D muscle fiber striations (근질).
2. Locks the Deltoid-to-Triceps anatomical V-insertion to the shoulder acromion (y_top + 104px) so Deltoid never stretches into Triceps during elbow extension.
3. Strictly blocks torso/lat bounce-light bleed and detached forearm patches across all motion frames (f=0..120).
4. Differentiates Step 3 (Lateral + Long Head recruitment during mid-extension) vs Step 4 (Full 3-Head Lateral + Long + Medial recruitment down to the elbow tendon at 180° lockout).
"""

import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v25_fast_preview as v25
import build_v30_pure_graphic_numbers as v30

OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)


def prepare_clean_muscle_frame_v31(bgr: np.ndarray) -> dict:
    scale = 0.84
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(eng.W * scale), int(eng.H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]
    h_r, w_r = r_ch.shape
    yy_r, xx_r = np.mgrid[:h_r, :w_r].astype(np.float32)

    red_any = np.clip((r_ch - b_ch - 10.0) / 90.0, 0.0, 1.0)
    green_any = np.clip((g_ch - np.maximum(r_ch, b_ch) - 12.0) / 90.0, 0.0, 1.0)

    # 1. Anatomical Upper-Arm Corridor:
    #    Strictly blocks torso/serratus/lat bounce light (x_r < 573 below armpit y_r > 562)
    #    while giving full room to posterior Long Head (up to x_r=755) and lower Medial Head (down to y_r=845).
    x_left_bound = np.where(yy_r < 562.0, 560.0, 573.0 + (yy_r - 562.0) * 0.055)
    x_right_bound = 755.0
    arm_corridor = (
        np.clip((xx_r - x_left_bound) / 9.0, 0.0, 1.0)
        * np.clip((x_right_bound - xx_r) / 12.0, 0.0, 1.0)
        * np.clip((yy_r - 455.0) / 10.0, 0.0, 1.0)
        * np.clip((845.0 - yy_r) / 12.0, 0.0, 1.0)
    )

    # 2. Inclusive muscle chrominance inside the arm corridor (captures shaded borders without harsh clipping)
    rb_muscle = np.clip((r_ch - b_ch - 65.0) / 70.0, 0.0, 1.0)
    rg_muscle = np.clip((r_ch - g_ch - 25.0) / 46.0, 0.0, 1.0)
    r_muscle = np.clip((r_ch - 104.0) / 60.0, 0.0, 1.0)
    raw_muscle = rb_muscle * rg_muscle * r_muscle * arm_corridor

    # 3. Morphological closing + connected-component filtering:
    #    - Bridges the thin dark inter-head crack ('철심') between Lateral & Long heads
    #    - Removes any detached small forearm island (such as at f=28 near the elbow)
    bw = (raw_muscle > 0.16).astype(np.uint8) * 255
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((bw_closed > 127).astype(np.uint8))
    clean_bw = np.zeros_like(bw_closed, dtype=np.float32)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        y0_c = stats[lbl, cv2.CC_STAT_TOP]
        if area >= 1800 and y0_c < 705:
            clean_bw[labels == lbl] = 1.0

    interior_bridge = cv2.erode(clean_bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    red_core_res = np.maximum(raw_muscle * clean_bw, interior_bridge * 0.88)
    red_core_res = (
        cv2.GaussianBlur(red_core_res, (0, 0), sigmaX=2.0, sigmaY=2.0)
        * cv2.GaussianBlur(clean_bw, (0, 0), sigmaX=2.2, sigmaY=2.2)
    )

    # 4. Preserve 100% sharp 3D muscle fiber striations (근질) while selectively lifting ONLY deep dark cracks ('철심')!
    fiber_lum_raw = 0.34 * r_ch + 0.66 * g_ch
    blur_med_raw = cv2.GaussianBlur(fiber_lum_raw, (0, 0), sigmaX=4.8, sigmaY=4.8)
    # A deep dark crack or cyan separator line has fiber_lum_raw significantly below blur_med_raw (or high blue channel)
    deep_crack = np.clip((blur_med_raw - fiber_lum_raw - 12.0) / 18.0, 0.0, 1.0) * clean_bw
    cyan_line = np.clip((b_ch - r_ch + 35.0) / 45.0, 0.0, 1.0) * clean_bw
    crack_fix_weight = np.clip(np.maximum(deep_crack, cyan_line * 0.85), 0.0, 1.0)
    fiber_lum = fiber_lum_raw * (1.0 - crack_fix_weight * 0.85) + (blur_med_raw * 0.96) * (crack_fix_weight * 0.85)

    blur_fine = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=1.25, sigmaY=1.25)
    blur_med = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=4.0, sigmaY=4.0)
    # Strong multi-scale unsharp mask so 3D muscle striations (근질) pop clearly through the color!
    fiber_detail = fiber_lum + (fiber_lum - blur_fine) * 2.45 + (blur_fine - blur_med) * 1.25
    fiber_norm_res = np.clip((fiber_detail - 24.0) / 185.0, 0.0, 1.0)

    non_red_lum = 0.25 * g_ch + 0.75 * b_ch
    sculpted_grey_muscle = 44.0 + 136.0 * (fiber_norm_res ** 1.04)

    muscle_zone_soft = cv2.GaussianBlur(clean_bw, (0, 0), sigmaX=3.0, sigmaY=3.0)
    red_replacement = non_red_lum * (1.0 - muscle_zone_soft) + sculpted_grey_muscle * muscle_zone_soft
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    neutral_lum = base_lum * (1.0 - red_any) + red_replacement * red_any

    hand_sculpt_lum = 65.0 + 0.65 * r_ch
    neutral_lum = neutral_lum * (1.0 - green_any) + hand_sculpt_lum * green_any
    neutral_lum = np.clip(neutral_lum, 14.0, 240.0)

    norm = neutral_lum / 255.0
    mono_r = eng.BG_CHARCOAL[0] + (234 - eng.BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = eng.BG_CHARCOAL[1] + (238 - eng.BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = eng.BG_CHARCOAL[2] + (244 - eng.BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    canvas, maps = eng.place_in_studio_canvas_clean(
        mono_res,
        {
            "fiber_norm": fiber_norm_res,
            "red_core": red_core_res,
        },
        scale=scale,
        x_shift=34,
        y_shift=50,
    )

    rc = maps["red_core"]
    m_ys, m_xs = np.where(rc > 0.20)
    if len(m_ys) > 40:
        y_top = float(np.percentile(m_ys, 2))
        y_bot = float(np.percentile(m_ys, 98))
        x_med = float(np.median(m_xs))
    else:
        y_top, y_bot, x_med = 677.0, 950.0, 745.0

    return {
        "mono": canvas,
        "fiber_norm": maps["fiber_norm"],
        "red_core": rc,
        "y_top": y_top,
        "y_bot": y_bot,
        "x_med": x_med,
    }


def render_clean_muscle_frame_v31(fd: dict, mode: str, is_wrong: bool, pulse: float = 1.0):
    mono = fd["mono"].astype(np.float32)
    rc = fd["red_core"]
    fn = fd["fiber_norm"]
    y_top, y_bot, x_med = fd["y_top"], fd["y_bot"], fd["x_med"]

    yy, xx = np.mgrid[:eng.H, :eng.W].astype(np.float32)

    # Fixed-distance anatomical Deltoid V-insertion (104px below shoulder acromion y_top)
    # Never stretches downward when y_bot extends!
    y_delt_v = (y_top + 104.0) - ((xx - x_med) ** 2) * 0.0018
    delt_weight = np.clip((y_delt_v - yy + 6.0) / 12.0, 0.0, 1.0)
    triceps_weight = 1.0 - delt_weight

    if mode == "anterior_deltoid":
        # Full Deltoid cap highlighted; Triceps completely neutral grey
        active_mask = rc * delt_weight
    elif mode == "triceps_lat_long":
        # Upper/Mid Triceps (Lateral & Long Heads) primary activation during mid-extension,
        # with smooth taper into the distal medial head near the elbow
        y_distal = (y_top + 104.0) + 0.78 * max(80.0, (y_bot - (y_top + 104.0)))
        distal_taper = np.clip((y_distal - yy + 22.0) / 38.0, 0.35, 1.0)
        active_mask = rc * triceps_weight * distal_taper
    else:  # triceps_all_three
        # Full 100% 3-Head Triceps (Lateral + Long + Medial Head) all the way to elbow tendon
        active_mask = np.clip(rc * triceps_weight * 1.06, 0.0, 1.0)

    active_mask = cv2.GaussianBlur(active_mask, (0, 0), sigmaX=1.8, sigmaY=1.8)
    alpha = np.clip(active_mask * (0.85 + 0.08 * pulse), 0.0, 0.94)[:, :, None]

    # High-contrast 3D striated muscle shading (preserves sculptural volume + fine muscle fibers)
    shade = (0.30 + 0.78 * (fn ** 1.08))[:, :, None]
    spec = (np.clip((fn - 0.58) / 0.36, 0.0, 1.0) ** 1.75)[:, :, None]

    if is_wrong:
        base_col = np.array([246.0, 44.0, 36.0], dtype=np.float32)
        hi_col = np.array([255.0, 175.0, 150.0], dtype=np.float32)
    else:
        base_col = np.array([16.0, 212.0, 92.0], dtype=np.float32)
        hi_col = np.array([165.0, 255.0, 195.0], dtype=np.float32)

    muscle_rgb = np.clip(base_col * shade + hi_col * spec * 0.44, 0.0, 255.0)
    out = mono * (1.0 - alpha) + muscle_rgb * alpha

    # Subtle 1px inner anatomical rim glow so muscle boundary sits cleanly inside the arm contour
    edge = cv2.GaussianBlur(active_mask, (0, 0), 1.2) - cv2.GaussianBlur(active_mask, (0, 0), 3.6)
    edge = np.clip(edge * 2.2, 0.0, 1.0)[:, :, None]
    rim_col = np.array([255.0, 130.0, 115.0]) if is_wrong else np.array([130.0, 255.0, 175.0])
    out = np.clip(out + edge * rim_col * 0.22, 0.0, 255.0)

    return out.astype(np.uint8), (int(x_med), int((y_top + y_bot) / 2))


def main():
    font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    test_frames = [8, 16, 28, 42, 54, 64, 78, 96]
    cw, ch = 380, 520
    c_box = (530, 545, 910, 1065)

    grid = Image.new("RGB", (cw * len(test_frames), ch * 4 + 90), (10, 12, 16))
    gd = ImageDraw.Draw(grid)
    gd.text(
        (28, 26),
        "DAY 1 MUSCLE TARGETING & MOTION TRACKING QC Across Full Range of Motion (f=8 Flexion -> f=96 Full Lockout)",
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28),
        fill=(245, 247, 252),
    )

    for i, fi in enumerate(test_frames):
        bgr = v25.read_specific_frame(eng.CORRECT_I2V, fi)
        fd_old = eng.prepare_muscle_frame(bgr)
        fd_new = prepare_clean_muscle_frame_v31(bgr)

        old_delt, _ = v25.render_natural_muscle_frame(fd_old, "anterior_deltoid", True, 1.0)
        new_delt, _ = render_clean_muscle_frame_v31(fd_new, "anterior_deltoid", True, 1.0)
        old_tri, _ = v25.render_natural_muscle_frame(fd_old, "triceps_all_three", False, 1.0)
        new_tri, _ = render_clean_muscle_frame_v31(fd_new, "triceps_all_three", False, 1.0)

        rows = [
            (old_delt, f"f={fi} [OLD v30 DELT] stretches down", (255, 115, 115)),
            (new_delt, f"f={fi} [NEW v31 DELT] locked to shoulder", (255, 215, 110)),
            (old_tri,  f"f={fi} [OLD v30 TRICEPS] dark crack + gap", (255, 115, 115)),
            (new_tri,  f"f={fi} [NEW v31 TRICEPS] seamless + striated", (90, 255, 150)),
        ]
        for r_idx, (arr, lbl, col) in enumerate(rows):
            p = Image.fromarray(arr).crop(c_box)
            pd = ImageDraw.Draw(p)
            pd.rectangle([0, 0, cw - 1, ch - 1], outline=(38, 44, 54), width=1)
            pd.rectangle([8, 8, cw - 8, 36], fill=(12, 14, 18))
            pd.text((14, 13), lbl, font=font_b, fill=col)
            grid.paste(p, (i * cw, 90 + r_idx * ch))

    qc_path = os.path.join(OUT_DIR, "day1_v31_muscle_tracking_8frame_qc.jpg")
    grid.save(qc_path, quality=93)
    print("Saved:", qc_path)

    # Now build the 4 full 9:16 frames of v31 (using v30-A's approved Pure Graphic & Number layout + v31 muscle engine)
    f_step1 = v25.read_specific_frame(eng.WRONG_I2V, 55)
    f_step2 = v25.read_specific_frame(eng.CORRECT_I2V, 24)
    f_step3 = v25.read_specific_frame(eng.CORRECT_I2V, 36)
    f_step4 = v25.read_specific_frame(eng.CORRECT_I2V, 64)

    thumbs = v30.make_top_thumbnails(f_step1, f_step2, f_step4)

    # Step 1: X-Ray Joint Overload
    fd1 = eng.prepare_xray_bone_frame(f_step1)
    m1, _ = eng.render_xray_skeletal_frame(fd1, pulse=1.0)
    # Step 2: Mistake 2 (Deltoid Takeover + +24° Forward Elbow Swing)
    fd2 = prepare_clean_muscle_frame_v31(f_step2)
    m2, _ = render_clean_muscle_frame_v31(fd2, "anterior_deltoid", is_wrong=True, pulse=1.0)
    m2 = v25.apply_elbow_swing_warp(m2, swing_deg=24.0, torso_lean_px=18.0)
    # Step 3: Tip 1 (Pinned Elbow + Upper/Mid Lateral & Long Head Triceps Ignition)
    fd3 = prepare_clean_muscle_frame_v31(f_step3)
    m3, _ = render_clean_muscle_frame_v31(fd3, "triceps_lat_long", is_wrong=False, pulse=1.0)
    # Step 4: Tip 2 (Full 180° Extension + All 3 Triceps Heads Peak Lockout)
    fd4 = prepare_clean_muscle_frame_v31(f_step4)
    m4, _ = render_clean_muscle_frame_v31(fd4, "triceps_all_three", is_wrong=False, pulse=1.0)

    raw_frames = [m1, m2, m3, m4]
    final_imgs = []
    for s_idx in range(4):
        im = v30.render_step_v30(raw_frames[s_idx], s_idx + 1, "A", thumbs)
        out_p = os.path.join(OUT_DIR, f"v31_step{s_idx+1}.jpg")
        im.save(out_p, quality=95)
        final_imgs.append(im)

    # Build 4-step sheet
    sw, sh = 540, 960
    sheet = Image.new("RGB", (sw * 4 + 50, sh + 90), (10, 12, 16))
    sd = ImageDraw.Draw(sheet)
    sd.text(
        (24, 24),
        "v31 MASTER — Motion-Locked 3D Muscle Mask (Zero Iron-Pin Crack, Zero Bleed, Striated 3D Relief) + Pure Graphic & Numbers",
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24),
        fill=(245, 247, 252),
    )
    for idx, im in enumerate(final_imgs):
        r_im = im.resize((sw, sh), Image.Resampling.LANCZOS)
        sheet.paste(r_im, (10 + idx * (sw + 10), 80))

    sheet_path = os.path.join(OUT_DIR, "day1_v31_4steps_sheet.jpg")
    sheet.save(sheet_path, quality=94)
    print("Saved:", sheet_path)


if __name__ == "__main__":
    main()
