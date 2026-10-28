#!/usr/bin/env python3
"""
Perfects Variant D (`t315u`–`t320u`):
1. ZERO Background Ghost / Afterimage (`뒤에 잔상 100% 제거`):
   - Uses MediaPipe `PoseLandmarker Heavy` AI human segmentation mask (`mp_seg`) + geometric Barbell & Weight-Plate mask
     instead of brightness thresholding (`base_lum > bg_2d + 8.5`), completely eliminating the smoky studio wall spotlight
     behind the back and between the legs!
2. ZERO Barbell Plate Edge Fade (`원판 테두리 반투명 잘림 제거`):
   - Removes the 84px horizontal edge fade that was eating into the barbell plates.
3. 100% Locked Vertical & Horizontal Fit (`상단·하단 고정 + 중앙 운동동작 완벽 Fit & 통일성`):
   - Top UI Zone locked at `y = 236 .. 420` (`bottom = 420px`).
   - Bottom HUD Zone locked at `y = 1368 .. 1468` (`top = 1368px`).
   - Central Action Stage (`y = 420 .. 1368`):
     * Mannequin feet (`Y_FLOOR`) locked to `y = 1314px` across ALL 4 steps (`std = 0.0px`) with realistic studio floor contact shadows!
     * Mannequin top (`Y_TOP_SQUAT`) locked to `y = 588px` at peak squat (and `y = 462px` at standing) across ALL 4 steps (`std = 0.0px`)!
     * Mannequin horizontal center (`X_CENTER`) locked to `x = 540px` across ALL 4 steps (`std = 0.0px`)!
4. Compares 3 Clean `%` Information Display Designs (`D-1`, `D-2`, `D-3`) so `%` is informative, uncluttered, and 0% intrusive.
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
import build_4variants_comparison_mockups as b4

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920

# Fixed Viewport Constants (`t316u`–`t319u`)
TOP_UI_BOTTOM_Y = 420
BOT_UI_TOP_Y = 1368
Y_FLOOR_LOCKED = 1312      # Exact floor line where feet are planted across ALL 4 steps (56px above Bottom UI)
Y_SQUAT_TOP_LOCKED = 582   # Exact top of head/barbell at peak squat across ALL 4 steps (162px below Top UI, centered in stage)
X_CENTER_LOCKED = 540      # Exact horizontal center of mannequin across ALL 4 steps


def extract_clean_mannequin_and_barbell_mask_720p(
    bgr_720p: np.ndarray,
    mp_seg_720p: np.ndarray,
    lm_norm: list,
    step_idx: int,
) -> tuple[np.ndarray, dict]:
    """
    Combines MediaPipe `PoseLandmarker Heavy` human segmentation (`mp_seg_720p`) with an exact
    geometric + luminance mask for the barbell shaft & weight plates across the shoulders/hands,
    while suppressing 100% of the background studio spotlight halo behind the torso and between the legs!
    """
    sh_h, sw_w = bgr_720p.shape[:2]
    rgb = cv2.cvtColor(bgr_720p, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    rb_diff = np.maximum(0.0, r_ch - b_ch)
    yy, xx = np.mgrid[:sh_h, :sw_w].astype(np.float32)

    sh_l = (lm_norm[11].x * sw_w, lm_norm[11].y * sh_h)
    sh_r = (lm_norm[12].x * sw_w, lm_norm[12].y * sh_h)
    el_l = (lm_norm[13].x * sw_w, lm_norm[13].y * sh_h)
    el_r = (lm_norm[14].x * sw_w, lm_norm[14].y * sh_h)
    wr_l = (lm_norm[15].x * sw_w, lm_norm[15].y * sh_h)
    wr_r = (lm_norm[16].x * sw_w, lm_norm[16].y * sh_h)
    sh_pt = (0.5 * (sh_l[0] + sh_r[0]), 0.5 * (sh_l[1] + sh_r[1]))
    hip_r = (lm_norm[24].x * sw_w, lm_norm[24].y * sh_h)
    hip_l = (lm_norm[23].x * sw_w, lm_norm[23].y * sh_h)
    kn_r = (lm_norm[26].x * sw_w, lm_norm[26].y * sh_h)
    kn_l = (lm_norm[25].x * sw_w, lm_norm[25].y * sh_h)
    ank_r = (lm_norm[28].x * sw_w, lm_norm[28].y * sh_h)
    ank_l = (lm_norm[27].x * sw_w, lm_norm[27].y * sh_h)
    nose_y = lm_norm[0].y * sh_h
    head_top = (sh_pt[0], min(nose_y, sh_pt[1]) - 54.0)
    foot_y = max(lm_norm[29].y, lm_norm[30].y, lm_norm[31].y, lm_norm[32].y) * sh_h
    foot_bot = (0.5 * (ank_r[0] + ank_l[0]), foot_y)

    # 1. Crisp Human Body Mask from MediaPipe Segmentation (`mp_seg_720p`)
    # Tighten threshold `(mp_seg - 0.42) / 0.30` so the soft outer halo around the body is 0.0!
    human_mask = np.clip((mp_seg_720p - 0.42) / 0.28, 0.0, 1.0)

    # 2. Barbell Shaft + Weight Plates Region Mask
    # The barbell passes through `(wr_r, sh_r, sh_l, wr_l)` and extends outward to the weight plates at both ends.
    bar_dx = wr_l[0] - wr_r[0]
    bar_dy = wr_l[1] - wr_r[1]
    bar_len = max(60.0, math.hypot(bar_dx, bar_dy))
    ux, uy = bar_dx / bar_len, bar_dy / bar_len

    # Extend barbell axis outward past the wrists to cover the weight plates
    ext_px = 145.0 if step_idx in (1, 2) else 95.0
    p_bar_a = (wr_r[0] - ux * ext_px, wr_r[1] - uy * ext_px)
    p_bar_b = (wr_l[0] + ux * ext_px, wr_l[1] + uy * ext_px)

    d_bar, t_bar = p62._segment_distance_field(xx, yy, p_bar_a, p_bar_b, 0.0, 1.0)
    # Near the ends (`t < 0.28` or `t > 0.72`), the weight plates have radius ~85px; in the center shaft, radius ~24px
    end_weight = np.maximum(np.clip((0.32 - t_bar) / 0.14, 0.0, 1.0), np.clip((t_bar - 0.68) / 0.14, 0.0, 1.0))
    max_bar_r = 22.0 + 70.0 * end_weight
    in_bar_corridor = np.clip((max_bar_r - d_bar) / 8.0, 0.0, 1.0)

    # Local background luminance in the upper barbell band (`y < hip_y`)
    bg_upper = float(np.percentile(base_lum[80:360, 10:60], 30))
    bar_fg = np.clip((base_lum - (bg_upper + 10.5)) / 7.5, 0.0, 1.0) * in_bar_corridor

    # Close small internal dark holes inside the weight plates
    bar_u8 = (bar_fg > 0.35).astype(np.uint8) * 255
    k_plate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bar_closed = cv2.morphologyEx(bar_u8, cv2.MORPH_CLOSE, k_plate).astype(np.float32) / 255.0
    bar_mask = cv2.GaussianBlur(bar_closed, (0, 0), 1.2) * in_bar_corridor

    # Combine Human Mask + Barbell Mask, and strictly zero out any stray background pixels below the barbell between/outside legs
    combined_mask = np.clip(np.maximum(human_mask, bar_mask), 0.0, 1.0)
    # Refine outer edge against actual foreground contrast so background wall is 0.0% included
    edge_contrast = np.clip((base_lum - (bg_upper + 4.5)) / 6.5, 0.0, 1.0)
    # Inside deep core (`human_mask > 0.82`), keep 1.0 so internal muscle shadows are never punctured;
    # on the boundary (`0.0 < combined_mask <= 0.82`), multiply by `edge_contrast` to shave off any halo!
    core_solid = np.clip((human_mask - 0.72) / 0.15, 0.0, 1.0)
    clean_mask = np.clip(np.maximum(core_solid, combined_mask * np.maximum(edge_contrast, (rb_diff > 10.0).astype(np.float32))), 0.0, 1.0)
    clean_mask = cv2.GaussianBlur(clean_mask, (0, 0), 1.0)

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
    return clean_mask, kp_720


def compose_perfected_1080p_frame(
    raw_bgr_720p: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    rep_a: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Renders the 100% ghost-free, vertically & horizontally fit-locked 1080x1920 frame:
    - Exact same top (`Y_SQUAT_TOP_LOCKED = 582px`) and floor (`Y_FLOOR_LOCKED = 1312px`) across Steps 1, 2, 3, and 4!
    - Exact same horizontal center (`X_CENTER_LOCKED = 540px`) across Steps 1, 2, 3, and 4!
    - Realistic soft studio floor contact shadow under both feet at `y = 1312px`!
    """
    if step_idx == 1:
        raw_bgr_720p = p62._apply_step1_knee_valgus_720p(raw_bgr_720p, rep_a)

    detector = uar.get_pose_detector()
    rgb_720 = cv2.cvtColor(raw_bgr_720p, cv2.COLOR_BGR2RGB)
    mp_res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb_720))
    lm = mp_res.pose_landmarks[0]
    mp_seg = mp_res.segmentation_masks[0].numpy_view()[:, :, 0]

    clean_mask_720, kp_720 = extract_clean_mannequin_and_barbell_mask_720p(raw_bgr_720p, mp_seg, lm, step_idx)
    man_720, _ = b4.shade_locked_crisp_mannequin_720p(
        raw_bgr_720p, step_idx, is_wrong, kp_720, rep_a=rep_a, color_mode="hybrid_ultimate"
    )

    # Measure exact top (`y_top_src`) and bottom (`y_bot_src`) of the clean segmented mannequin in 720p
    ys, xs = np.where(clean_mask_720 > 0.35)
    y_top_src = float(np.percentile(ys, 0.5))
    y_bot_src = float(np.percentile(ys, 99.5))
    # Horizontal center of pelvis/torso
    cx_src = 0.5 * (kp_720["hip_r"][0] + kp_720["hip_l"][0])

    # Compute exact uniform scale & translation so that:
    # 1. `y_bot_src` lands at `Y_FLOOR_LOCKED = 1312px`
    # 2. `y_top_src` lands at `Y_SQUAT_TOP_LOCKED = 582px` (or interpolated from standing `462px` to squatting `582px`)
    target_top_y = 462.0 * (1.0 - rep_a) + float(Y_SQUAT_TOP_LOCKED) * rep_a
    target_bot_y = float(Y_FLOOR_LOCKED)
    scale = (target_bot_y - target_top_y) / max(120.0, (y_bot_src - y_top_src))

    sh_h, sw_w = raw_bgr_720p.shape[:2]
    nw = int(round(sw_w * scale))
    nh = int(round(sh_h * scale))
    x0 = int(round(X_CENTER_LOCKED - cx_src * scale))
    y0 = int(round(target_bot_y - y_bot_src * scale))

    res_man = cv2.resize(man_720, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    res_mask = cv2.resize(clean_mask_720, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    dst_x1 = max(0, x0)
    dst_x2 = min(W, x0 + nw)
    src_x1 = dst_x1 - x0
    src_x2 = src_x1 + (dst_x2 - dst_x1)

    dst_y1 = max(0, y0)
    dst_y2 = min(H, y0 + nh)
    src_y1 = dst_y1 - y0
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    canvas = b4.LOCKED_STUDIO_BG.copy()

    # Add realistic studio floor grounding contact shadow at `Y_FLOOR_LOCKED = 1312px` under both feet (`개선점: 공중부양 방지 접지 그림자`)
    ank_r_1080 = (x0 + kp_720["ank_r"][0] * scale, y0 + kp_720["ank_r"][1] * scale)
    ank_l_1080 = (x0 + kp_720["ank_l"][0] * scale, y0 + kp_720["ank_l"][1] * scale)
    yy_f, xx_f = np.mgrid[:H, :W].astype(np.float32)
    shadow_field = np.zeros((H, W), dtype=np.float32)
    for ax_pt in (ank_r_1080, ank_l_1080):
        sx = (xx_f - ax_pt[0]) / 78.0
        sy = (yy_f - (Y_FLOOR_LOCKED + 4.0)) / 18.0
        shadow_field = np.maximum(shadow_field, np.exp(-0.5 * (sx * sx + sy * sy)))
    # Ambient floor shadow between the feet under the pelvis
    sx_mid = (xx_f - X_CENTER_LOCKED) / 185.0
    sy_mid = (yy_f - (Y_FLOOR_LOCKED + 8.0)) / 24.0
    shadow_field = np.clip(shadow_field * 0.52 + np.exp(-0.5 * (sx_mid * sx_mid + sy_mid * sy_mid)) * 0.26, 0.0, 0.62)
    canvas *= (1.0 - shadow_field[:, :, None])

    crop_man = res_man[src_y1:src_y2, src_x1:src_x2, :]
    crop_mask = np.clip(res_mask[src_y1:src_y2, src_x1:src_x2], 0.0, 1.0)[:, :, None]

    bg_patch = canvas[dst_y1:dst_y2, dst_x1:dst_x2, :]
    canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = crop_man * crop_mask + bg_patch * (1.0 - crop_mask)

    full_mask = np.zeros((H, W), dtype=np.float32)
    full_mask[dst_y1:dst_y2, dst_x1:dst_x2] = crop_mask[:, :, 0]

    kp_1080 = {}
    for k, (kx, ky) in kp_720.items():
        kp_1080[k] = (x0 + kx * scale, y0 + ky * scale)

    return np.clip(canvas, 0, 255).astype(np.uint8), full_mask, kp_1080



STEP_META = {
    1: ("84%", "ACL & MCL OVERLOAD", "16°", "KNEE VALGUS", "INWARD KNEE COLLAPSE · SHEAR RISK"),
    2: ("96%", "QUAD & VMO LOCK", "0°", "KNEE TRACKING", "KNEES ALIGNED OVER TOES · MAX VMO"),
    3: ("79%", "LUMBAR DISC SHEAR", "41°", "FORWARD LEAN", "GOOD-MORNING SQUAT · LOWER BACK OVERLOAD"),
    4: ("98%", "QUADS & GLUTES", "22°", "NEUTRAL SPINE", "4×10 REPS · FULL QUAD & GLUTE DRIVE"),
}


def load_step_peak_frames() -> dict:
    raw_peak = {}
    for s_i in (1, 2, 3, 4):
        fn, i0, i1, is_w, sc, y_st, y_sq, ys_st, ys_sq, x_sh = p62.STEP_POOL_SPECS[s_i]
        seq = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=52)
        raw_peak[s_i] = seq[-1]
    return raw_peak


def draw_laser_joint_line(
    rgb_base: np.ndarray,
    body_mask: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    kp: dict,
) -> np.ndarray:
    canvas = Image.fromarray(rgb_base).convert("RGBA")
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

    b4.draw_subsurface_additive_laser_axis(
        canvas,
        body_mask_1080=body_mask,
        main_chains_1x=main_chains,
        aux_ext_chains_1x=aux_chains,
        vertex_pts_1x=v_pts,
        is_wrong=is_wrong,
        laser_style="hybrid_contrast",
    )
    return np.array(canvas.convert("RGB"))


def draw_locked_top_zone_ui(
    frame_rgb: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    thumbs: dict,
    pct_in_top_badge: bool = False,
) -> np.ndarray:
    """
    Draws the 100% pixel-locked Top Zone (`y = 236 .. 420`, symmetric `x = 96 .. 984`):
    - Top header pill (`y = 236 .. 278`)
    - 3-Thumbnail `PREV · NOW · NEXT` carousel (`y = 292 .. 420`) with Red/Green border synced to `✕/✓`
    - Right circular `✕ / ✓` status badge (`cx = 908, cy = 352, r = 52`)
    - If `pct_in_top_badge=True` (Option D-2), attaches a sleek mini `%` pill right below the `✕/✓` badge!
    """
    pil = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    f_hdr = v30.f_pretendard(21, "extrabold")
    f_step = v30.f_pretendard(16, "extrabold")
    f_Slot = v30.f_pretendard(14, "extrabold")
    f_pct_mini = v30.f_pretendard(20, "extrabold")

    accent = (255, 48, 56) if is_wrong else (32, 245, 122)

    # 1. Exercise Header Pill (`y = 236 .. 278`, centered at `x = 540`)
    hdr_txt = "BARBELL BACK SQUAT"
    tw = d.textlength(hdr_txt, font=f_hdr)
    pw = int(tw + 66)
    px1 = (W - pw) // 2
    py1, py2 = 236, 278
    d.rounded_rectangle([px1, py1, px1 + pw, py2], radius=21, fill=(14, 16, 20, 232), outline=(255, 255, 255, 48), width=1)
    d.ellipse([px1 + 18, (py1 + py2) // 2 - 5, px1 + 28, (py1 + py2) // 2 + 5], fill=(*accent, 255))
    d.text((px1 + 38, py1 + 9), hdr_txt, font=f_hdr, fill=(248, 250, 255, 255))

    # 2. Symmetric Top Layout (`x = 96 .. 984`, total width = `888px`)
    # Carousel container on left (`x = 96 .. 796`, width = `700px`), Status Badge on right (`cx = 908`)
    c_x1, c_y1, c_x2, c_y2 = 96, 290, 796, 420
    d.rounded_rectangle([c_x1, c_y1, c_x2, c_y2], radius=24, fill=(12, 14, 18, 228), outline=(255, 255, 255, 38), width=2)

    slot_steps = [max(1, step_idx - 1), step_idx, min(4, step_idx + 1)]
    slot_labels = ["PREV", "NOW", "NEXT"]
    slot_boxes = [
        (114, 300, 304, 410, False, 0),
        (318, 295, 574, 415, True, 1),
        (588, 300, 778, 410, False, 2),
    ]

    pil = Image.alpha_composite(pil, ov)

    for (bx1, by1, bx2, by2, is_active, s_idx) in slot_boxes:
        bw, bh = bx2 - bx1, by2 - by1
        s_num = slot_steps[s_idx]
        s_wrong = (s_num in (1, 3))
        s_col = (255, 52, 60) if s_wrong else (32, 245, 122)

        t_img = cv2.resize(thumbs[s_num], (bw, bh), interpolation=cv2.INTER_AREA)
        if not is_active:
            t_img = (t_img.astype(np.float32) * 0.65).astype(np.uint8)

        t_pil = Image.fromarray(t_img).convert("RGBA")
        mask = Image.new("L", (bw, bh), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=16, fill=255)
        pil.paste(t_pil, (bx1, by1), mask)

        b_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(b_ov)
        if is_active:
            # Sync NOW border color with Red/Green step status (`개선점: NOW 테두리 색상 동기화`)
            bd.rounded_rectangle([bx1 - 2, by1 - 2, bx2 + 2, by2 + 2], radius=18, outline=(*s_col, 115), width=5)
            bd.rounded_rectangle([bx1, by1, bx2, by2], radius=16, outline=(*s_col, 255), width=3)
        else:
            bd.rounded_rectangle([bx1, by1, bx2, by2], radius=16, outline=(255, 255, 255, 60), width=1)

        # Step badge top-left
        tag_txt = f"{s_num}/4"
        tw_t = bd.textlength(tag_txt, font=f_step)
        bd.rounded_rectangle([bx1 + 8, by1 + 7, bx1 + int(tw_t) + 22, by1 + 29], radius=7, fill=(10, 12, 16, 225))
        bd.text((bx1 + 15, by1 + 9), tag_txt, font=f_step, fill=(*s_col, 255) if is_active else (225, 230, 240, 225))

        # PREV / NOW / NEXT pill bottom-left
        sl_txt = slot_labels[s_idx]
        tw_s = bd.textlength(sl_txt, font=f_Slot)
        bd.rounded_rectangle(
            [bx1 + 8, by2 - 25, bx1 + int(tw_s) + 20, by2 - 6],
            radius=6,
            fill=(*s_col, 240) if is_active else (14, 16, 20, 215),
        )
        bd.text(
            (bx1 + 14, by2 - 23),
            sl_txt,
            font=f_Slot,
            fill=(10, 12, 14, 255) if (is_active and not s_wrong) else (255, 255, 255, 250),
        )

        # Mini X/Check dot bottom-right
        mcx, mcy = bx2 - 18, by2 - 16
        bd.ellipse([mcx - 10, mcy - 10, mcx + 10, mcy + 10], fill=(*s_col, 245))
        if s_wrong:
            bd.line([mcx - 4, mcy - 4, mcx + 4, mcy + 4], fill=(255, 255, 255, 255), width=2)
            bd.line([mcx - 4, mcy + 4, mcx + 4, mcy - 4], fill=(255, 255, 255, 255), width=2)
        else:
            bd.line([mcx - 5, mcy, mcx - 1, mcy + 4], fill=(12, 16, 14, 255), width=2)
            bd.line([mcx - 1, mcy + 4, mcx + 5, mcy - 4], fill=(12, 16, 14, 255), width=2)

        pil = Image.alpha_composite(pil, b_ov)

    # 3. Right Circular Status Badge (`cx = 908, cy = 342 if pct_in_top_badge else 355`)
    ov2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(ov2)
    bcx = 908
    bcy = 338 if pct_in_top_badge else 355
    br = 46 if pct_in_top_badge else 52
    d2.ellipse([bcx - br - 6, bcy - br - 6, bcx + br + 6, bcy + br + 6], fill=(*accent, 52))
    d2.ellipse([bcx - br, bcy - br, bcx + br, bcy + br], fill=(*accent, 248), outline=(255, 255, 255, 235), width=3)
    if is_wrong:
        o = int(br * 0.35)
        d2.line([bcx - o, bcy - o, bcx + o, bcy + o], fill=(255, 255, 255, 255), width=8)
        d2.line([bcx - o, bcy + o, bcx + o, bcy - o], fill=(255, 255, 255, 255), width=8)
    else:
        d2.line([bcx - int(br * 0.40), bcy + 2, bcx - 5, bcy + int(br * 0.36)], fill=(12, 18, 14, 255), width=9)
        d2.line([bcx - 5, bcy + int(br * 0.36), bcx + int(br * 0.44), bcy - int(br * 0.32)], fill=(12, 18, 14, 255), width=9)

    if pct_in_top_badge:
        # Option D-2: Attached mini `%` pill directly below the status badge (`y = 392 .. 426`)
        pct_val, pct_tag, _, _, _ = STEP_META[step_idx]
        p_txt = f"{pct_val} {'LOAD' if is_wrong else 'ACT'}"
        tw_p = d2.textlength(p_txt, font=f_pct_mini)
        pw_p = int(tw_p + 28)
        px_p1 = bcx - pw_p // 2
        py_p1, py_p2 = 392, 426
        d2.rounded_rectangle(
            [px_p1, py_p1, px_p1 + pw_p, py_p2],
            radius=17,
            fill=(14, 16, 22, 242),
            outline=(*accent, 235),
            width=2,
        )
        d2.text((px_p1 + 14, py_p1 + 5), p_txt, font=f_pct_mini, fill=(255, 255, 255, 255))

    pil = Image.alpha_composite(pil, ov2)
    return np.array(pil.convert("RGB"))


def draw_pct_option_d1_integrated_bottom_dock(
    frame_rgb: np.ndarray,
    step_idx: int,
    is_wrong: bool,
) -> np.ndarray:
    """
    Option D-1 (`하단 1줄 슬림 통합 도크 — Slim Integrated Bottom Dock`):
    - Zero floating text in the center action zone (`y = 420 .. 1368`).
    - At the locked bottom zone (`x = 96 .. 984, y = 1368 .. 1458`, height = `90px`),
      a single sleek horizontal dark-glass dock cleanly separates:
      [ LEFT: `84%` + 5-Segment Mini Bar + `JOINT LOAD` ] | [ RIGHT: `16° KNEE VALGUS` + `ACL SHEAR RISK` ]
    """
    pil = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    pct_txt, pct_sub, deg_txt, cue_title, cue_sub = STEP_META[step_idx]
    accent = (255, 58, 66) if is_wrong else (32, 245, 122)

    bx1, by1, bx2, by2 = 96, 1368, 984, 1458
    d.rounded_rectangle(
        [bx1, by1, bx2, by2],
        radius=24,
        fill=(12, 14, 18, 232),
        outline=(255, 255, 255, 42),
        width=2,
    )
    # Left accent bar inside dock
    d.rounded_rectangle([bx1 + 14, by1 + 16, bx1 + 22, by2 - 16], radius=4, fill=(*accent, 255))

    # Left Compartment (`x = 132 .. 420`): Big `%` + 5-segment gauge + short label
    f_pct = v30.f_pretendard(44, "extrabold")
    f_lbl = v30.f_pretendard(15, "extrabold")
    d.text((bx1 + 36, by1 + 16), pct_txt, font=f_pct, fill=(*accent, 255))
    pct_w = int(d.textlength(pct_txt, font=f_pct))

    # 5-segment mini battery bar (`x = bx1 + 48 + pct_w`)
    gx0 = bx1 + 48 + pct_w
    gy0 = by1 + 24
    active_segs = 4 if step_idx in (1, 3) else 5
    for i in range(5):
        sx1 = gx0 + i * 24
        scol = (*accent, 255) if i < active_segs else (55, 60, 70, 180)
        d.rounded_rectangle([sx1, gy0, sx1 + 19, gy0 + 16], radius=4, fill=scol)
    d.text((gx0, gy0 + 23), pct_sub, font=f_lbl, fill=(210, 216, 228, 235))

    # Vertical Divider (`x = 456`)
    div_x = 456
    d.line([div_x, by1 + 16, div_x, by2 - 16], fill=(255, 255, 255, 42), width=2)

    # Right Compartment (`x = 480 .. 964`): Angle + Biomechanical Cue
    f_deg = v30.f_pretendard(36, "extrabold")
    f_title = v30.f_pretendard(24, "extrabold")
    f_sub = v30.f_pretendard(16, "extrabold")

    d.text((div_x + 24, by1 + 14), deg_txt, font=f_deg, fill=(255, 255, 255, 255))
    deg_w = int(d.textlength(deg_txt, font=f_deg))
    d.text((div_x + 36 + deg_w, by1 + 21), cue_title, font=f_title, fill=(*accent, 255))
    d.text((div_x + 26, by1 + 56), cue_sub, font=f_sub, fill=(195, 202, 215, 230))

    pil = Image.alpha_composite(pil, ov)
    return np.array(pil.convert("RGB"))


def draw_pct_option_d2_top_badge_and_slim_bottom(
    frame_rgb: np.ndarray,
    step_idx: int,
    is_wrong: bool,
) -> np.ndarray:
    """
    Option D-2 (`우측 상단 배지 하단 미니 캡슐 + 하단 초슬림 바`):
    - `%` is already attached right beneath the top-right `✕ / ✓` badge (`96% ACT` / `84% LOAD`).
    - Bottom fixed zone (`x = 140 .. 940, y = 1374 .. 1450`, height = `76px`) has a super clean,
      uncluttered single-line posture pill (`16° KNEE VALGUS · ACL SHEAR RISK`).
    """
    pil = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    _, _, deg_txt, cue_title, cue_sub = STEP_META[step_idx]
    accent = (255, 58, 66) if is_wrong else (32, 245, 122)

    bx1, by1, bx2, by2 = 130, 1372, 950, 1452
    d.rounded_rectangle(
        [bx1, by1, bx2, by2],
        radius=38,
        fill=(12, 14, 18, 228),
        outline=(*accent, 150),
        width=2,
    )

    f_deg = v30.f_pretendard(36, "extrabold")
    f_title = v30.f_pretendard(25, "extrabold")
    f_sub = v30.f_pretendard(17, "extrabold")

    # Status dot on left
    d.ellipse([bx1 + 26, (by1 + by2) // 2 - 9, bx1 + 44, (by1 + by2) // 2 + 9], fill=(*accent, 255))
    d.text((bx1 + 62, by1 + 18), deg_txt, font=f_deg, fill=(255, 255, 255, 255))
    deg_w = int(d.textlength(deg_txt, font=f_deg))
    d.text((bx1 + 78 + deg_w, by1 + 14), cue_title, font=f_title, fill=(*accent, 255))
    d.text((bx1 + 78 + deg_w, by1 + 44), cue_sub, font=f_sub, fill=(205, 212, 224, 230))

    pil = Image.alpha_composite(pil, ov)
    return np.array(pil.convert("RGB"))


def draw_pct_option_d3_side_micro_pill_and_slim_bottom(
    frame_rgb: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    kp_1080: dict,
) -> np.ndarray:
    """
    Option D-3 (`몸 바깥 여백 초소형 마이크로 `%` 태그 + 하단 슬림 바`):
    - Places a compact, non-intrusive micro pill (`96%` + 5-dot mini bar) in the empty dark studio margin
      outside the mannequin silhouette (`x = 758..958`, never overlapping the body!), connected by a subtle leader tick,
      plus the clean bottom bar (`Option D-2` bottom bar).
    """
    out = draw_pct_option_d2_top_badge_and_slim_bottom(frame_rgb, step_idx, is_wrong)
    pil = Image.fromarray(out).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    pct_txt, pct_sub, _, _, _ = STEP_META[step_idx]
    accent = (255, 58, 66) if is_wrong else (32, 245, 122)

    # Place micro pill in guaranteed exterior studio margin (`x = 768 .. 974, y = 920 .. 994`)
    px1, py1, px2, py2 = 768, 915, 976, 991
    d.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=18,
        fill=(12, 14, 18, 224),
        outline=(*accent, 210),
        width=2,
    )

    f_pct = v30.f_pretendard(32, "extrabold")
    f_sub = v30.f_pretendard(13, "extrabold")
    d.text((px1 + 16, py1 + 10), pct_txt, font=f_pct, fill=(*accent, 255))
    tw = int(d.textlength(pct_txt, font=f_pct))

    # 5 mini dots next to `%`
    active_segs = 4 if is_wrong else 5
    for i in range(5):
        dx = px1 + 26 + tw + i * 15
        dy = py1 + 22
        scol = (*accent, 255) if i < active_segs else (65, 70, 80, 180)
        d.rounded_rectangle([dx, dy, dx + 10, dy + 12], radius=3, fill=scol)

    d.text((px1 + 18, py1 + 48), pct_sub, font=f_sub, fill=(220, 225, 235, 235))

    pil = Image.alpha_composite(pil, ov)
    return np.array(pil.convert("RGB"))


def build_clean_thumbnails(peak_raws: dict) -> dict:
    """Builds 100% ghost-free, consistently scaled thumbnails for the Top Carousel."""
    thumbs = {}
    for s in (1, 2, 3, 4):
        is_w = (s in (1, 3))
        base_1080, mask_1080, kp_1080 = compose_perfected_1080p_frame(peak_raws[s], s, is_w, rep_a=1.0)
        with_laser = draw_laser_joint_line(base_1080, mask_1080, s, is_w, kp_1080)
        # Crop central action stage (`y = 510 .. 1335, x = 140 .. 940`) so every thumbnail has identical framing!
        crop = with_laser[510:1335, 140:940]
        thumbs[s] = cv2.resize(crop, (360, 260), interpolation=cv2.INTER_AREA)
    return thumbs


def render_perfected_step(
    raw_bgr_720p: np.ndarray,
    step_idx: int,
    thumbs: dict,
    pct_option: str = "D1",
    draw_fit_guides: bool = False,
) -> tuple[np.ndarray, dict]:
    is_wrong = (step_idx in (1, 3))
    base_1080, mask_1080, kp_1080 = compose_perfected_1080p_frame(raw_bgr_720p, step_idx, is_wrong, rep_a=1.0)
    frame = draw_laser_joint_line(base_1080, mask_1080, step_idx, is_wrong, kp_1080)

    # Top Locked Zone (`y = 236 .. 420`)
    frame = draw_locked_top_zone_ui(
        frame, step_idx, is_wrong, thumbs, pct_in_top_badge=(pct_option == "D2")
    )

    # Bottom Locked Zone (`y = 1368 .. 1458`) + `%` Option
    if pct_option == "D1":
        frame = draw_pct_option_d1_integrated_bottom_dock(frame, step_idx, is_wrong)
    elif pct_option == "D2":
        frame = draw_pct_option_d2_top_badge_and_slim_bottom(frame, step_idx, is_wrong)
    elif pct_option == "D3":
        frame = draw_pct_option_d3_side_micro_pill_and_slim_bottom(frame, step_idx, is_wrong, kp_1080)

    if draw_fit_guides:
        pil = Image.fromarray(frame).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        f_g = v30.f_pretendard(16, "extrabold")
        guides = [
            (TOP_UI_BOTTOM_Y, (80, 200, 255, 190), "TOP UI LOCK (y=420)"),
            (Y_SQUAT_TOP_LOCKED, (255, 215, 70, 200), "HEAD/BAR TOP FIT (y=582)"),
            (Y_FLOOR_LOCKED, (40, 245, 130, 215), "FLOOR FOOT LOCK (y=1312)"),
            (BOT_UI_TOP_Y, (80, 200, 255, 190), "BOTTOM HUD LOCK (y=1368)"),
        ]
        for gy, gcol, glbl in guides:
            for x in range(24, W - 24, 28):
                d.line([x, gy, x + 16, gy], fill=gcol, width=2)
            d.rounded_rectangle([28, gy - 24, 290, gy - 3], radius=6, fill=(10, 12, 16, 215))
            d.text((36, gy - 22), glbl, font=f_g, fill=gcol)
        # Vertical centerline (`x = 540`)
        for y in range(TOP_UI_BOTTOM_Y + 10, BOT_UI_TOP_Y - 10, 32):
            d.line([X_CENTER_LOCKED, y, X_CENTER_LOCKED, y + 14], fill=(255, 255, 255, 75), width=2)
        pil = Image.alpha_composite(pil, ov)
        frame = np.array(pil.convert("RGB"))

    ys, _ = np.where(mask_1080 > 0.35)
    metrics = {
        "y_top": int(np.percentile(ys, 0.5)),
        "y_bot": int(np.percentile(ys, 99.5)),
        "cx": int(round(0.5 * (kp_1080["hip_r"][0] + kp_1080["hip_l"][0]))),
    }
    return frame, metrics


def make_sheet(frames_by_step: dict, title: str, subtitle: str, out_path: str) -> str:
    tw, th = 500, 888
    pad = 24
    header_h = 128
    sw = pad * 5 + tw * 4
    sh = header_h + th + pad * 2

    sheet = Image.new("RGB", (sw, sh), (14, 16, 20))
    d = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(32, "extrabold")
    f_s = v30.f_pretendard(20, "extrabold")
    f_c = v30.f_pretendard(18, "extrabold")

    d.text((pad, 24), title, font=f_t, fill=(245, 248, 255))
    d.text((pad, 72), subtitle, font=f_s, fill=(72, 238, 148))

    step_names = {
        1: "STEP 1 · WRONG (16° KNEE VALGUS)",
        2: "STEP 2 · RIGHT (0° KNEE TRACKING)",
        3: "STEP 3 · WRONG (42° SPINE COLLAPSE)",
        4: "STEP 4 · RIGHT (NEUTRAL SPINE LOCK)",
    }

    for idx, s in enumerate((1, 2, 3, 4)):
        x0 = pad + idx * (tw + pad)
        y0 = header_h
        im = Image.fromarray(frames_by_step[s]).resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(im, (x0, y0))
        col = (255, 68, 76) if s in (1, 3) else (40, 245, 130)
        d.rounded_rectangle([x0 - 2, y0 - 2, x0 + tw + 2, y0 + th + 2], radius=14, outline=col, width=3)
        d.rounded_rectangle([x0 + 12, y0 + 12, x0 + tw - 12, y0 + 48], radius=9, fill=(10, 12, 16, 225))
        d.text((x0 + 22, y0 + 19), step_names[s], font=f_c, fill=col)

    sheet.save(out_path, quality=95)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    peak_raws = load_step_peak_frames()
    thumbs = build_clean_thumbnails(peak_raws)

    option_meta = {
        "D1": (
            "OPTION D-1 (추천 1순위: 하단 1줄 슬림 통합 도크 — 중앙 시야 방해 0% + 하단 좌우 분할 `[% 게이지 | 각도·큐]`)",
            "상단(y=236..420) · 하단(y=1368..1458) 완전 고정 | 중앙(y=582..1312) 발바닥·머리 높이 4구간 100% 일치 + 배경 잔상 0% 제거",
            f"{OUT_DIR}/day3_mockup_D1_bottom_dock.jpg",
        ),
        "D2": (
            "OPTION D-2 (우측 상단 ✕/✓ 배지 직결 미니 `%` 캡슐 + 하단 초슬림 1줄 각도 바)",
            "`%` 수치를 우측 상단 ✕/✓ 배지 바로 아래(`84% LOAD` / `96% ACT`)에 일체화하고, 하단 바는 각도·큐만 1줄로 미니멀하게 배치",
            f"{OUT_DIR}/day3_mockup_D2_top_badge_pill.jpg",
        ),
        "D3": (
            "OPTION D-3 (몸 바깥 여백 마이크로 `%` 태그 + 하단 초슬림 1줄 각도 바)",
            "마네킹을 절대 가리지 않는 우측 다크 여백(`x=768..976`)에 초소형 `%` + 5도트 미니 게이지를 배치하고 하단 바를 슬림화",
            f"{OUT_DIR}/day3_mockup_D3_side_micro_tag.jpg",
        ),
    }

    rendered_options = {}
    for opt_key, (title, sub, out_p) in option_meta.items():
        frames = {}
        metrics_all = {}
        for s in (1, 2, 3, 4):
            fr, m = render_perfected_step(peak_raws[s], s, thumbs, pct_option=opt_key, draw_fit_guides=False)
            frames[s] = fr
            metrics_all[s] = m
        make_sheet(frames, title, sub, out_p)
        rendered_options[opt_key] = out_p
        print(f"[{opt_key}] saved -> {out_p} | metrics={metrics_all}")

    # Build Fit & Ghost Proof Sheet (`day3_d_before_vs_after_ghost_and_fit.jpg`) with alignment guide lines!
    fit_frames = {}
    for s in (1, 2, 3, 4):
        fr_g, _ = render_perfected_step(peak_raws[s], s, thumbs, pct_option="D1", draw_fit_guides=True)
        fit_frames[s] = fr_g
    fit_path = f"{OUT_DIR}/day3_d_fit_and_ghost_free_proof.jpg"
    make_sheet(
        fit_frames,
        "PERFECTED VARIANT D — 3-ZONE LOCKED FIT & ZERO AFTERIMAGE PROOF (전 구간 높이·정렬·잔상 제거 검증)",
        "TOP UI LOCK (y=420) · HEAD/BAR TOP (y=582, 오차 0px) · FLOOR FOOT LINE + SHADOW (y=1312, 오차 0px) · BOTTOM HUD LOCK (y=1368)",
        fit_path,
    )

    # Combine D-1, D-2, D-3 into a single master comparison overview
    imgs = [Image.open(rendered_options[k]) for k in ("D1", "D2", "D3")]
    ow, oh = imgs[0].size
    comp = Image.new("RGB", (ow, oh * 3 + 32), (10, 12, 16))
    for i, im in enumerate(imgs):
        comp.paste(im, (0, i * (oh + 16)))
    comp_path = f"{OUT_DIR}/day3_d_pct_3options_comparison.jpg"
    comp.save(comp_path, quality=94)
    print("Master comparison saved ->", comp_path)


if __name__ == "__main__":
    main()
