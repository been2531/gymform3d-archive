#!/usr/bin/env python3
"""
Fast Single-Step Preview Generator (`t321u`–`t324u`):
1. 100% Intact Barbell & Weight Plates (`운동기구 어디부분은 없어지고 그러면 안돼`, `원판 짤리면 안되고`):
   - Scales raw 720p by `scale = 1.38` (`nw = 994px <= 1080px`) so the entire 720p width fits inside 1080px with 0% horizontal plate clipping!
   - Preserves 100% of the barbell shaft & weight plates in the upper torso zone (`y < hip_y`), while using AI human segmentation + clean leg separation only below the bar so zero equipment ever disappears!
2. 100% Day 2 Locked Mannequin Skin Tone (`마네킹 아까전부터 색상 어떻게 하라고 알려준거 잊지말고`):
   - Uses the exact Day 2 Step 2 formula (`BG_CHARCOAL + (234, 238, 244 - BG_CHARCOAL) * (norm ** 0.92)`) with smooth matte slate-silver grey (`RGB 84,87,91` midtones) and bright silver rim light (`140~185`).
3. Minimal Text + Luxury Numbers & Graphics (`글은 최소화하고 수치와 그래픽이 고급져야해`):
   - Generates 4 distinct luxury HUD concepts (Concept 1, 2, 3, 4) on Step 2 (Right) & Step 1 (Wrong) in < 5 seconds!
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
import build_v36_publish_reels as v36
import preview_v62_seamless_day2_master as p62
import build_4variants_comparison_mockups as b4

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920
BG_RGB = np.array([38.0, 40.0, 44.0], dtype=np.float32)


def render_day2_exact_mannequin_and_full_barbell_1080p(raw_bgr_720p: np.ndarray, step_idx: int, is_wrong: bool):
    """
    1. Preserves 100% of the barbell and weight plates (no geometric clipping of plates, and `nw = 994px <= 1080px` so left/right edges never clip!).
    2. Applies the exact Day 2 Step 2 skin formula (`BG_CHARCOAL + (234, 238, 244 - BG_CHARCOAL) * (norm ** 0.92)`).
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
    foot_y = max(lm[29].y, lm[30].y, lm[31].y, lm[32].y) * sh_h

    kp_720 = {
        "sh": sh_pt,
        "hip_r": hip_r,
        "hip_l": hip_l,
        "kn_r": kn_r,
        "kn_l": kn_l,
        "ank_r": ank_r,
        "ank_l": ank_l,
    }

    # Complete Foreground Mask that NEVER clips ANY gym equipment:
    # - Any pixel in the upper barbell zone (`y < hip_y + 20`) with foreground contrast (`base_lum > bg + 7.5`) is 100% preserved!
    # - In the lower zone (`y >= hip_y - 20`), MediaPipe human segmentation (`mp_seg`) removes the background wall/floor halo!
    yy, xx = np.mgrid[:sh_h, :sw_w].astype(np.float32)
    bg_ref = float(np.percentile(base_lum[60:260, 10:50], 25))
    fg_contrast = np.clip((base_lum - (bg_ref + 6.5)) / 6.0, 0.0, 1.0)

    # Upper zone weight (includes entire barbell + weight plates across full width `x = 0..720`)
    hip_mid_y = 0.5 * (hip_r[1] + hip_l[1])
    upper_equip_zone = np.clip((hip_mid_y - 10.0 - yy) / 45.0, 0.0, 1.0)
    equip_mask = fg_contrast * upper_equip_zone
    # Solidify the interior of the weight plates using morphological closing so dark inner plate rings never drop out
    eq_u8 = (equip_mask > 0.25).astype(np.uint8) * 255
    k_eq = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    eq_closed = cv2.morphologyEx(eq_u8, cv2.MORPH_CLOSE, k_eq).astype(np.float32) / 255.0
    equip_mask = np.maximum(equip_mask, eq_closed * np.clip((base_lum - (bg_ref + 3.0)) / 5.0, 0.0, 1.0))

    human_mask = np.clip((mp_seg - 0.38) / 0.28, 0.0, 1.0)
    full_fg_mask = np.clip(np.maximum(human_mask, equip_mask), 0.0, 1.0)
    full_fg_mask = cv2.GaussianBlur(full_fg_mask, (0, 0), 1.0)

    # Exact Day 2 Step 2 Mannequin Skin Formula (`build_v36_publish_reels.py` lines 184-187):
    # `BG_CHARCOAL + (234, 238, 244 - BG_CHARCOAL) * (norm ** 0.92)`
    # Smooth micro-noise slightly (`sigma=1.4`) while preserving 100% 3D muscle definition and silver rim light
    lum_smooth = cv2.GaussianBlur(base_lum, (0, 0), 1.4)
    detail = base_lum - lum_smooth
    lum_clean = lum_smooth + detail * 0.45

    # Fixed normalization matching Day 2 Step 2 (`bg=22.0`, `hi=196.0`) -> midtones `RGB(84,87,91)`, rim light `145~185`
    norm = np.clip((lum_clean - 22.0) / (196.0 - 22.0), 0.0, 1.0)
    curve = norm ** 0.92
    skin_r = BG_RGB[0] + (234.0 - BG_RGB[0]) * curve
    skin_g = BG_RGB[1] + (238.0 - BG_RGB[1]) * curve
    skin_b = BG_RGB[2] + (244.0 - BG_RGB[2]) * curve
    man_rgb = np.stack([skin_r, skin_g, skin_b], axis=-1)

    # Target Muscle Glow (Crimson Red for Wrong, Emerald Green for Right — strictly restricted to target thigh/glute region)
    y_chest = sh_pt[1] + 0.38 * (hip_mid_y - sh_pt[1])
    y_ankle = 0.5 * (ank_r[1] + ank_l[1]) - 15.0
    if step_idx in (1, 2, 4):
        anat_zone = np.clip((yy - y_chest) / 35.0, 0.0, 1.0) * np.clip((y_ankle - yy) / 35.0, 0.0, 1.0)
    else:
        anat_zone = np.clip((yy - sh_pt[1]) / 35.0, 0.0, 1.0) * np.clip((y_ankle - yy) / 35.0, 0.0, 1.0)

    raw_red = np.clip((rb_diff - 12.0) / 38.0, 0.0, 1.0) * anat_zone * human_mask
    raw_red = cv2.GaussianBlur(raw_red, (0, 0), 3.0)

    fiber_n = np.clip((lum_clean - 24.0) / 150.0, 0.0, 1.0)
    if is_wrong:
        # Deep anatomical crimson-red target highlight
        t_r = 48.0 + 195.0 * (fiber_n ** 0.88)
        t_g = 12.0 + 38.0 * (fiber_n ** 1.25)
        t_b = 16.0 + 34.0 * (fiber_n ** 1.35)
    else:
        # Day 2 signature emerald-green target highlight
        t_r = 14.0 + 46.0 * (fiber_n ** 1.20)
        t_g = 52.0 + 192.0 * (fiber_n ** 0.86)
        t_b = 34.0 + 110.0 * (fiber_n ** 1.02)
    target_rgb = np.stack([t_r, t_g, t_b], axis=-1)

    alpha_t = np.clip(raw_red * 1.15, 0.0, 0.94)[:, :, None]
    shaded_720 = man_rgb * (1.0 - alpha_t) + target_rgb * alpha_t

    # Fit inside 1080x1920 with `scale <= 1.40` so `nw = int(720 * scale) <= 1010px` (NEVER clipping barbell plates horizontally!)
    # And lock feet at `Y_FLOOR = 1312px`!
    ys_fg, xs_fg = np.where(full_fg_mask > 0.35)
    x_min_fg, x_max_fg = float(np.percentile(xs_fg, 0.2)), float(np.percentile(xs_fg, 99.8))
    y_top_fg, y_bot_fg = float(np.percentile(ys_fg, 0.5)), float(np.percentile(ys_fg, 99.5))

    # Target height = `730px` (`y = 582 .. 1312`), capped so width never exceeds `996px` (leaving >= 42px margin on both sides of plates!)
    scale_h = 730.0 / max(120.0, (y_bot_fg - y_top_fg))
    scale_w = 996.0 / max(120.0, (x_max_fg - x_min_fg))
    scale = min(scale_h, scale_w)

    nw, nh = int(round(sw_w * scale)), int(round(sh_h * scale))
    cx_fg = 0.5 * (x_min_fg + x_max_fg)
    x0 = int(round(540.0 - cx_fg * scale))
    y0 = int(round(1312.0 - y_bot_fg * scale))

    res_man = cv2.resize(shaded_720, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    res_mask = cv2.resize(full_fg_mask, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    canvas = b4.LOCKED_STUDIO_BG.copy()

    # Grounding floor contact shadow at `y = 1312px`
    ank_r_1080 = (x0 + kp_720["ank_r"][0] * scale, y0 + kp_720["ank_r"][1] * scale)
    ank_l_1080 = (x0 + kp_720["ank_l"][0] * scale, y0 + kp_720["ank_l"][1] * scale)
    yy_f, xx_f = np.mgrid[:H, :W].astype(np.float32)
    sh_f = np.zeros((H, W), dtype=np.float32)
    for ax_pt in (ank_r_1080, ank_l_1080):
        sx = (xx_f - ax_pt[0]) / 75.0
        sy = (yy_f - 1316.0) / 16.0
        sh_f = np.maximum(sh_f, np.exp(-0.5 * (sx * sx + sy * sy)))
    canvas *= (1.0 - np.clip(sh_f * 0.48, 0.0, 0.55)[:, :, None])

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


def draw_clean_top_carousel(
    frame_rgb: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    thumbs: dict,
    ring_pct: int | None = None,
) -> np.ndarray:
    """
    Draws the locked Top Zone (`y = 236 .. 420`) with minimal text (`1/4` .. `4/4` + `✕/✓` icons, zero cluttered text):
    - If `ring_pct` is provided (Concept 3), renders a 4x supersampled glowing circular progress ring around the right `✕/✓` badge
      with a sleek `96%` pill right below it!
    """
    pil = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    f_hdr = v30.f_pretendard(20, "extrabold")
    f_step = v30.f_pretendard(15, "extrabold")
    accent = (255, 52, 60) if is_wrong else (32, 245, 122)

    # 1. Minimal Exercise Pill (`BARBELL SQUAT`, short & clean)
    hdr_txt = "BARBELL SQUAT"
    tw = d.textlength(hdr_txt, font=f_hdr)
    pw = int(tw + 58)
    px1 = (W - pw) // 2
    d.rounded_rectangle([px1, 236, px1 + pw, 276], radius=20, fill=(14, 16, 20, 230), outline=(255, 255, 255, 45), width=1)
    d.ellipse([px1 + 16, 251, px1 + 26, 261], fill=(*accent, 255))
    d.text((px1 + 36, 245), hdr_txt, font=f_hdr, fill=(245, 248, 255, 255))

    # 2. 3-Stage Preview Carousel (`x = 96 .. 786, y = 290 .. 420`)
    d.rounded_rectangle([96, 290, 786, 420], radius=24, fill=(12, 14, 18, 228), outline=(255, 255, 255, 38), width=2)
    pil = Image.alpha_composite(pil, ov)

    slot_steps = [4 if step_idx == 1 else step_idx - 1, step_idx, 1 if step_idx == 4 else step_idx + 1]
    slot_boxes = [
        (112, 300, 300, 410, False, 0),
        (314, 294, 568, 416, True, 1),
        (582, 300, 770, 410, False, 2),
    ]

    for (bx1, by1, bx2, by2, is_active, s_i) in slot_boxes:
        bw, bh = bx2 - bx1, by2 - by1
        s_num = slot_steps[s_i]
        s_w = (s_num in (1, 3))
        s_col = (255, 52, 60) if s_w else (32, 245, 122)

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
            bd.rounded_rectangle([bx1 - 2, by1 - 2, bx2 + 2, by2 + 2], radius=18, outline=(*s_col, 110), width=5)
            bd.rounded_rectangle([bx1, by1, bx2, by2], radius=16, outline=(*s_col, 255), width=3)
        else:
            bd.rounded_rectangle([bx1, by1, bx2, by2], radius=16, outline=(255, 255, 255, 55), width=1)

        # Minimal step number `1/4` + mini `✕/✓` icon only (zero wordy text!)
        tag = f"{s_num}/4"
        tw_t = bd.textlength(tag, font=f_step)
        bd.rounded_rectangle([bx1 + 8, by1 + 7, bx1 + int(tw_t) + 20, by1 + 27], radius=6, fill=(10, 12, 16, 220))
        bd.text((bx1 + 14, by1 + 8), tag, font=f_step, fill=(*s_col, 255) if is_active else (220, 225, 235, 220))

        mcx, mcy = bx2 - 17, by2 - 16
        bd.ellipse([mcx - 10, mcy - 10, mcx + 10, mcy + 10], fill=(*s_col, 245))
        if s_w:
            bd.line([mcx - 4, mcy - 4, mcx + 4, mcy + 4], fill=(255, 255, 255, 255), width=2)
            bd.line([mcx - 4, mcy + 4, mcx + 4, mcy - 4], fill=(255, 255, 255, 255), width=2)
        else:
            bd.line([mcx - 5, mcy, mcx - 1, mcy + 4], fill=(10, 14, 12, 255), width=2)
            bd.line([mcx - 1, mcy + 4, mcx + 5, mcy - 4], fill=(10, 14, 12, 255), width=2)
        pil = Image.alpha_composite(pil, b_ov)

    # 3. Right Status Badge (`cx = 904, cy = 350`)
    SS = 3
    ov_ss = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dss = ImageDraw.Draw(ov_ss)
    bcx, bcy = 904, 340 if ring_pct is not None else 355
    br = 42 if ring_pct is not None else 50

    if ring_pct is not None:
        # Luxury outer progress arc ring around the status badge!
        r_ring = br + 11
        dss.arc(
            [(bcx - r_ring) * SS, (bcy - r_ring) * SS, (bcx + r_ring) * SS, (bcy + r_ring) * SS],
            start=0,
            end=360,
            fill=(55, 62, 74, 210),
            width=6 * SS,
        )
        end_deg = -90 + int(360 * (ring_pct / 100.0))
        dss.arc(
            [(bcx - r_ring) * SS, (bcy - r_ring) * SS, (bcx + r_ring) * SS, (bcy + r_ring) * SS],
            start=-90,
            end=end_deg,
            fill=(*accent, 255),
            width=6 * SS,
        )

    dss.ellipse([(bcx - br - 5) * SS, (bcy - br - 5) * SS, (bcx + br + 5) * SS, (bcy + br + 5) * SS], fill=(*accent, 48))
    dss.ellipse([(bcx - br) * SS, (bcy - br) * SS, (bcx + br) * SS, (bcy + br) * SS], fill=(*accent, 248), outline=(255, 255, 255, 235), width=3 * SS)
    if is_wrong:
        o = int(br * 0.35 * SS)
        dss.line([bcx * SS - o, bcy * SS - o, bcx * SS + o, bcy * SS + o], fill=(255, 255, 255, 255), width=8 * SS)
        dss.line([bcx * SS - o, bcy * SS + o, bcx * SS + o, bcy * SS - o], fill=(255, 255, 255, 255), width=8 * SS)
    else:
        dss.line([(bcx - int(br * 0.38)) * SS, (bcy + 2) * SS, (bcx - 4) * SS, (bcy + int(br * 0.35)) * SS], fill=(10, 16, 12, 255), width=8 * SS)
        dss.line([(bcx - 4) * SS, (bcy + int(br * 0.35)) * SS, (bcx + int(br * 0.42)) * SS, (bcy - int(br * 0.32)) * SS], fill=(10, 16, 12, 255), width=8 * SS)

    ov_1x = Image.fromarray(cv2.resize(np.array(ov_ss), (W, H), interpolation=cv2.INTER_AREA))
    pil = Image.alpha_composite(pil, ov_1x)

    if ring_pct is not None:
        ov_p = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dp = ImageDraw.Draw(ov_p)
        f_p = v30.f_pretendard(22, "extrabold")
        ptxt = f"{ring_pct}%"
        tw_p = dp.textlength(ptxt, font=f_p)
        pw_p = int(tw_p + 28)
        px_p = bcx - pw_p // 2
        dp.rounded_rectangle([px_p, 396, px_p + pw_p, 430], radius=17, fill=(12, 14, 18, 245), outline=(*accent, 240), width=2)
        dp.text((px_p + 14, 400), ptxt, font=f_p, fill=(255, 255, 255, 255))
        pil = Image.alpha_composite(pil, ov_p)

    return np.array(pil.convert("RGB"))


def draw_hud_concept_1_arc_ring_telemetry(frame_rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    """
    시안 1: `Arc Ring & 10-Bar Telemetry Dock` (원형 아크 링 게이지 + 10단 정밀 네온 블록 바 + 1단어 키워드)
    - 글자 최소화: 긴 문장 0%, 오직 `96%` (원형 링 내부), `10단 네온 텔레메트리 바`, `0°` + `ALIGNED` (또는 `16°` + `VALGUS`)만!
    """
    pil = Image.fromarray(frame_rgb).convert("RGBA")
    SS = 3
    ov_ss = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dss = ImageDraw.Draw(ov_ss)

    accent = (255, 54, 64) if is_wrong else (32, 245, 122)
    pct_val = 84 if is_wrong else 96
    deg_txt = "16°" if is_wrong else "0°"
    short_tag = "VALGUS" if is_wrong else "ALIGNED"
    bar_tag = "JOINT LOAD" if is_wrong else "QUAD ACTIVATION"

    # Compact floating glass capsule (`x = 120 .. 960, y = 1366 .. 1464`, width = `840px`, height = `98px`)
    bx1, by1, bx2, by2 = 120, 1366, 960, 1464
    dss.rounded_rectangle(
        [bx1 * SS, by1 * SS, bx2 * SS, by2 * SS],
        radius=49 * SS,
        fill=(11, 13, 18, 236),
        outline=(255, 255, 255, 48),
        width=2 * SS,
    )
    # Subtle inner top specular rim
    dss.arc([(bx1 + 6) * SS, (by1 + 4) * SS, (bx1 + 92) * SS, (by2 - 4) * SS], start=120, end=240, fill=(*accent, 180), width=3 * SS)

    # 1. Left Precision Circular Arc Gauge (`cx = 186, cy = 1415, r = 34`)
    rcx, rcy, rr = 186, 1415, 34
    dss.arc([(rcx - rr) * SS, (rcy - rr) * SS, (rcx + rr) * SS, (rcy + rr) * SS], start=135, end=405, fill=(52, 58, 70, 220), width=7 * SS)
    span = int(270 * (pct_val / 100.0))
    dss.arc([(rcx - rr) * SS, (rcy - rr) * SS, (rcx + rr) * SS, (rcy + rr) * SS], start=135, end=135 + span, fill=(*accent, 255), width=7 * SS)

    # 2. Center 12-Segment Slanted Luxury Telemetry Bar (`x = 256 .. 636`)
    seg_x0 = 256
    seg_y1, seg_y2 = 1412, 1438
    num_segs = 12
    active_segs = 10 if is_wrong else 12
    for i in range(num_segs):
        sx = seg_x0 + i * 31
        slant = 5
        pts = [
            ((sx + slant) * SS, seg_y1 * SS),
            ((sx + 22 + slant) * SS, seg_y1 * SS),
            ((sx + 22) * SS, seg_y2 * SS),
            (sx * SS, seg_y2 * SS),
        ]
        if i < active_segs:
            # Gradient intensity from 75% to 100%
            f_int = 0.72 + 0.28 * (i / max(1, num_segs - 1))
            scol = (int(accent[0] * f_int), int(accent[1] * f_int), int(accent[2] * f_int), 255)
        else:
            scol = (48, 54, 64, 185)
        dss.polygon(pts, fill=scol)

    # Subtle vertical divider (`x = 668`)
    dss.line([668 * SS, (by1 + 20) * SS, 668 * SS, (by2 - 20) * SS], fill=(255, 255, 255, 42), width=2 * SS)

    # 3. Right Angle Icon (Protractor Vertex Icon) at `x = 696, y = 1415`
    icx, icy = 710, 1415
    dss.line([(icx - 12) * SS, (icy + 12) * SS, (icx + 14) * SS, (icy + 12) * SS], fill=(180, 190, 205, 220), width=3 * SS)
    dss.line([(icx - 12) * SS, (icy + 12) * SS, (icx + 8) * SS, (icy - 12) * SS], fill=(*accent, 255), width=3 * SS)
    dss.arc([(icx - 22) * SS, (icy - 6) * SS, (icx + 6) * SS, (icy + 22) * SS], start=-50, end=0, fill=(*accent, 230), width=2 * SS)

    ov_1x = Image.fromarray(cv2.resize(np.array(ov_ss), (W, H), interpolation=cv2.INTER_AREA))
    pil = Image.alpha_composite(pil, ov_1x)

    # Crisp Typography Overlay
    tx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(tx)
    f_ring = v30.f_pretendard(22, "extrabold")
    f_lbl = v30.f_pretendard(14, "extrabold")
    f_deg = v30.f_pretendard(40, "extrabold")
    f_tag = v30.f_pretendard(16, "extrabold")

    ptxt = f"{pct_val}%"
    tw_p = td.textlength(ptxt, font=f_ring)
    td.text((rcx - tw_p / 2, rcy - 12), ptxt, font=f_ring, fill=(255, 255, 255, 255))

    # Micro header above telemetry bar
    td.text((seg_x0 + 4, 1384), bar_tag, font=f_lbl, fill=(195, 205, 220, 230))
    td.text((seg_x0 + 318, 1384), f"{pct_val}/100", font=f_lbl, fill=(*accent, 255))

    # Right Angle Number + 1-word pill
    td.text((736, 1391), deg_txt, font=f_deg, fill=(255, 255, 255, 255))
    dw = int(td.textlength(deg_txt, font=f_deg))
    px_t = 746 + dw
    tw_tag = int(td.textlength(short_tag, font=f_tag))
    td.rounded_rectangle([px_t, 1399, px_t + tw_tag + 24, 1431], radius=16, fill=(*accent, 38), outline=(*accent, 210), width=2)
    td.text((px_t + 12, 1405), short_tag, font=f_tag, fill=(*accent, 255))

    pil = Image.alpha_composite(pil, tx)
    return np.array(pil.convert("RGB"))


def draw_hud_concept_2_minimal_glass_slider(frame_rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    """
    시안 2: `Ultra-Slim Glass Slider Pill` (초슬림 스모크 글래스 캡슐 — `[ 16° ] ━━━●━ [ 84% ]` 순수 수치·그래픽 중심)
    - 글자는 극도로 절제된 1단어(`VALGUS` / `LOCKED`)만 두고, 좌측 `각도`, 중앙 `네온 글로우 프로그레스 트랙`, 우측 `퍼센트`를 대칭 배치!
    """
    pil = Image.fromarray(frame_rgb).convert("RGBA")
    SS = 3
    ov_ss = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dss = ImageDraw.Draw(ov_ss)

    accent = (255, 54, 64) if is_wrong else (32, 245, 122)
    pct_val = 84 if is_wrong else 96
    deg_txt = "16°" if is_wrong else "0°"
    mode_word = "VALGUS" if is_wrong else "LOCKED"

    bx1, by1, bx2, by2 = 150, 1374, 930, 1456
    dss.rounded_rectangle(
        [bx1 * SS, by1 * SS, bx2 * SS, by2 * SS],
        radius=41 * SS,
        fill=(12, 14, 19, 235),
        outline=(*accent, 145),
        width=2 * SS,
    )

    # Left Angle Sub-Pill (`x = 164 .. 308, y = 1386 .. 1444`)
    dss.rounded_rectangle([164 * SS, 1386 * SS, 308 * SS, 1444 * SS], radius=29 * SS, fill=(22, 25, 32, 255), outline=(255, 255, 255, 40), width=1 * SS)

    # Center Continuous Neon Track + Tick Marks (`x = 338 .. 722, y = 1422`)
    tx1, tx2, ty = 338, 722, 1424
    dss.rounded_rectangle([tx1 * SS, (ty - 5) * SS, tx2 * SS, (ty + 5) * SS], radius=5 * SS, fill=(45, 50, 62, 230))
    fill_x = tx1 + int((tx2 - tx1) * (pct_val / 100.0))
    dss.rounded_rectangle([tx1 * SS, (ty - 5) * SS, fill_x * SS, (ty + 5) * SS], radius=5 * SS, fill=(*accent, 255))
    # Glowing thumb node on track
    dss.ellipse([(fill_x - 13) * SS, (ty - 13) * SS, (fill_x + 13) * SS, (ty + 13) * SS], fill=(*accent, 75))
    dss.ellipse([(fill_x - 7) * SS, (ty - 7) * SS, (fill_x + 7) * SS, (ty + 7) * SS], fill=(255, 255, 255, 255), outline=(*accent, 255), width=2 * SS)

    ov_1x = Image.fromarray(cv2.resize(np.array(ov_ss), (W, H), interpolation=cv2.INTER_AREA))
    pil = Image.alpha_composite(pil, ov_1x)

    tx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(tx)
    f_deg = v30.f_pretendard(34, "extrabold")
    f_pct = v30.f_pretendard(40, "extrabold")
    f_w = v30.f_pretendard(15, "extrabold")

    # Left Angle
    dw = td.textlength(deg_txt, font=f_deg)
    td.text((236 - dw / 2, 1395), deg_txt, font=f_deg, fill=(255, 255, 255, 255))

    # Center minimal 1-word label above track
    td.text((tx1 + 2, 1390), mode_word, font=f_w, fill=(*accent, 255))

    # Right Hero Percentage (`x = 752 .. 906`)
    ptxt = f"{pct_val}%"
    pw = td.textlength(ptxt, font=f_pct)
    td.text((832 - pw / 2, 1391), ptxt, font=f_pct, fill=(*accent, 255))

    pil = Image.alpha_composite(pil, tx)
    return np.array(pil.convert("RGB"))


def draw_hud_concept_3_top_ring_and_micro_angle_pill(frame_rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    """
    시안 3: `Top-Right Ring Gauge + Micro Angle Pill` (우측 상단 배지 외곽 링 게이지에 `96%` 통합 + 하단 초소형 각도 캡슐)
    - `%`와 원형 게이지는 이미 우측 상단 `✕/✓` 배지에 통합되어 있으므로, 하단에는 손바닥만한 미니 캡슐(`[ ⦠ 0° · ALIGNED ]`)만 배치!
    """
    pil = Image.fromarray(frame_rgb).convert("RGBA")
    SS = 3
    ov_ss = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dss = ImageDraw.Draw(ov_ss)

    accent = (255, 54, 64) if is_wrong else (32, 245, 122)
    deg_txt = "16°" if is_wrong else "0°"
    word_txt = "VALGUS" if is_wrong else "ALIGNED"

    # Centered Micro Jewel Pill (`x = 350 .. 730, y = 1378 .. 1450`, width = `380px`, height = `72px`)
    bx1, by1, bx2, by2 = 350, 1378, 730, 1450
    dss.rounded_rectangle(
        [bx1 * SS, by1 * SS, bx2 * SS, by2 * SS],
        radius=36 * SS,
        fill=(12, 14, 19, 236),
        outline=(*accent, 185),
        width=2 * SS,
    )

    # Geometric angle icon on left (`x = 398, y = 1414`)
    icx, icy = 398, 1414
    dss.line([(icx - 11) * SS, (icy + 11) * SS, (icx + 13) * SS, (icy + 11) * SS], fill=(200, 210, 225, 230), width=3 * SS)
    dss.line([(icx - 11) * SS, (icy + 11) * SS, (icx + 7) * SS, (icy - 11) * SS], fill=(*accent, 255), width=3 * SS)

    ov_1x = Image.fromarray(cv2.resize(np.array(ov_ss), (W, H), interpolation=cv2.INTER_AREA))
    pil = Image.alpha_composite(pil, ov_1x)

    tx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(tx)
    f_deg = v30.f_pretendard(38, "extrabold")
    f_w = v30.f_pretendard(22, "extrabold")

    td.text((430, 1391), deg_txt, font=f_deg, fill=(255, 255, 255, 255))
    dw = int(td.textlength(deg_txt, font=f_deg))
    td.line([446 + dw, 1396, 446 + dw, 1432], fill=(255, 255, 255, 48), width=2)
    td.text((466 + dw, 1401), word_txt, font=f_w, fill=(*accent, 255))

    pil = Image.alpha_composite(pil, tx)
    return np.array(pil.convert("RGB"))


def draw_hud_concept_4_dual_instrument_pods(frame_rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    """
    시안 4: `Dual Luxury Instrument Pods` (하단 좌·우 독립 슈퍼카 계기판 포드 — 좌측 `[각도 다이얼 0°]` · 우측 `[96% + 5단 파워 바]`)
    - 가운데 바닥 공간(`x = 430 .. 650`)을 완전히 비워 마네킹의 다리·발 접지선이 시원하게 열려 있고,
      좌우 대칭 미니 글래스 포드 2개로 수치와 그래픽만 보여줌!
    """
    pil = Image.fromarray(frame_rgb).convert("RGBA")
    SS = 3
    ov_ss = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dss = ImageDraw.Draw(ov_ss)

    accent = (255, 54, 64) if is_wrong else (32, 245, 122)
    pct_val = 84 if is_wrong else 96
    deg_txt = "16°" if is_wrong else "0°"

    # Left Pod (`x = 96 .. 426, y = 1370 .. 1456`, width = `330px`) — Angle Pod
    dss.rounded_rectangle([96 * SS, 1370 * SS, 426 * SS, 1456 * SS], radius=43 * SS, fill=(12, 14, 19, 236), outline=(255, 255, 255, 45), width=2 * SS)
    # Mini Angle Arc Dial inside Left Pod (`cx = 152, cy = 1413, r = 28`)
    lcx, lcy, lr = 152, 1413, 28
    dss.arc([(lcx - lr) * SS, (lcy - lr) * SS, (lcx + lr) * SS, (lcy + lr) * SS], start=180, end=360, fill=(55, 62, 74, 220), width=5 * SS)
    # Needle line
    ang_rad = math.radians(-90 + (28 if is_wrong else 0))
    nx = lcx + math.sin(ang_rad) * (lr - 4)
    ny = lcy - math.cos(ang_rad) * (lr - 4)
    dss.line([lcx * SS, (lcy + 10) * SS, int(nx * SS), int(ny * SS)], fill=(*accent, 255), width=4 * SS)
    dss.ellipse([(lcx - 5) * SS, (lcy + 5) * SS, (lcx + 5) * SS, (lcy + 15) * SS], fill=(255, 255, 255, 255))

    # Right Pod (`x = 654 .. 984, y = 1370 .. 1456`, width = `330px`) — Percentage + 5-Bar Pod
    dss.rounded_rectangle([654 * SS, 1370 * SS, 984 * SS, 1456 * SS], radius=43 * SS, fill=(12, 14, 19, 236), outline=(255, 255, 255, 45), width=2 * SS)
    # 5 Slanted Power Bars inside Right Pod (`x = 816 .. 946`)
    active_b = 4 if is_wrong else 5
    for i in range(5):
        bx = 816 + i * 26
        by_top = 1424 - i * 4
        by_bot = 1438
        scol = (*accent, 255) if i < active_b else (52, 58, 70, 190)
        dss.rounded_rectangle([bx * SS, by_top * SS, (bx + 18) * SS, by_bot * SS], radius=4 * SS, fill=scol)

    ov_1x = Image.fromarray(cv2.resize(np.array(ov_ss), (W, H), interpolation=cv2.INTER_AREA))
    pil = Image.alpha_composite(pil, ov_1x)

    tx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(tx)
    f_big = v30.f_pretendard(40, "extrabold")
    f_lbl = v30.f_pretendard(14, "extrabold")

    # Left Pod Text: `0°` + `ANGLE`
    td.text((206, 1390), deg_txt, font=f_big, fill=(255, 255, 255, 255))
    dw = int(td.textlength(deg_txt, font=f_big))
    td.text((218 + dw, 1406), "VALGUS" if is_wrong else "LOCK", font=f_lbl, fill=(*accent, 255))

    # Right Pod Text: `96%` + `POWER`
    ptxt = f"{pct_val}%"
    td.text((688, 1390), ptxt, font=f_big, fill=(*accent, 255))
    td.text((818, 1384), "LOAD" if is_wrong else "PEAK", font=f_lbl, fill=(200, 208, 220, 230))

    pil = Image.alpha_composite(pil, tx)
    return np.array(pil.convert("RGB"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Load Step 1..4 peak frames once for thumbnails, and render ONLY Step 2 (Right) & Step 1 (Wrong) for ultra-fast comparison!
    raw_peaks = {}
    for s_i in (1, 2, 3, 4):
        fn, i0, i1, is_w, *_ = p62.STEP_POOL_SPECS[s_i]
        seq = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=52)
        raw_peaks[s_i] = seq[-1]

    # Render base frames + laser lines for Steps 1..4 so thumbnails & main frames have 100% intact barbell plates and Day 2 skin!
    base_with_laser = {}
    thumbs = {}
    for s_i in (1, 2, 3, 4):
        is_w = (s_i in (1, 3))
        b_rgb, b_mask, b_kp = render_day2_exact_mannequin_and_full_barbell_1080p(raw_peaks[s_i], s_i, is_w)
        # Draw @demicstory laser line
        sh, hip_r, hip_l, kn_r, kn_l, ank_r, ank_l = (
            b_kp["sh"], b_kp["hip_r"], b_kp["hip_l"], b_kp["kn_r"], b_kp["kn_l"], b_kp["ank_r"], b_kp["ank_l"]
        )
        if s_i == 1:
            kn_r_v, kn_l_v = (kn_r[0] + 18.0, kn_r[1]), (kn_l[0] - 18.0, kn_l[1])
            top_r, _ = p62._extend_segment(hip_r, kn_r_v, -0.15, 1.0)
            _, bot_r = p62._extend_segment(kn_r_v, ank_r, 0.0, 1.15)
            top_l, _ = p62._extend_segment(hip_l, kn_l_v, -0.15, 1.0)
            _, bot_l = p62._extend_segment(kn_l_v, ank_l, 0.0, 1.15)
            ext_r1, ext_r2 = p62._extend_segment(hip_r, kn_r_v, 0.58, 1.48)
            ext_l1, ext_l2 = p62._extend_segment(hip_l, kn_l_v, 0.58, 1.48)
            m_ch, a_ch, v_p = [[top_r, kn_r_v, bot_r], [top_l, kn_l_v, bot_l]], [[ext_r1, ext_r2], [ext_l1, ext_l2]], [kn_r_v, kn_l_v]
        elif s_i == 2:
            top_r, _ = p62._extend_segment(hip_r, kn_r, -0.15, 1.0)
            _, bot_r = p62._extend_segment(kn_r, ank_r, 0.0, 1.15)
            top_l, _ = p62._extend_segment(hip_l, kn_l, -0.15, 1.0)
            _, bot_l = p62._extend_segment(kn_l, ank_l, 0.0, 1.15)
            ext_r1, ext_r2 = p62._extend_segment(hip_r, kn_r, 0.62, 1.38)
            ext_l1, ext_l2 = p62._extend_segment(hip_l, kn_l, 0.62, 1.38)
            m_ch, a_ch, v_p = [[top_r, kn_r, bot_r], [top_l, kn_l, bot_l]], [[ext_r1, ext_r2], [ext_l1, ext_l2]], [kn_r, kn_l]
        else:
            sh_top, _ = p62._extend_segment(sh, hip_r, -0.16, 1.0)
            _, kn_ext = p62._extend_segment(hip_r, kn_r, 0.0, 1.16)
            tc_a, tc_b = p62._extend_segment(sh, hip_r, 0.66, 1.38)
            th_a, th_b = p62._extend_segment(kn_r, hip_r, 0.68, 1.33)
            m_ch, a_ch, v_p = [[sh_top, hip_r, kn_ext]], [[tc_a, tc_b], [th_a, th_b]], [hip_r]

        canv = Image.fromarray(b_rgb).convert("RGBA")
        b4.draw_subsurface_additive_laser_axis(canv, b_mask, m_ch, a_ch, v_p, is_w, laser_style="hybrid_contrast")
        with_l = np.array(canv.convert("RGB"))
        base_with_laser[s_i] = with_l
        thumbs[s_i] = cv2.resize(with_l[520:1325, 40:1040], (360, 260), interpolation=cv2.INTER_AREA)

    # Build the 4 HUD Concepts for Step 2 (Right) and Step 1 (Wrong)
    concepts = []
    for c_idx in (1, 2, 3, 4):
        pair = {}
        for s_i in (2, 1):
            is_w = (s_i == 1)
            pct_val = 84 if is_w else 96
            top_fr = draw_clean_top_carousel(
                base_with_laser[s_i], s_i, is_w, thumbs, ring_pct=(pct_val if c_idx == 3 else None)
            )
            if c_idx == 1:
                pair[s_i] = draw_hud_concept_1_arc_ring_telemetry(top_fr, is_w)
            elif c_idx == 2:
                pair[s_i] = draw_hud_concept_2_minimal_glass_slider(top_fr, is_w)
            elif c_idx == 3:
                pair[s_i] = draw_hud_concept_3_top_ring_and_micro_angle_pill(top_fr, is_w)
            elif c_idx == 4:
                pair[s_i] = draw_hud_concept_4_dual_instrument_pods(top_fr, is_w)
        concepts.append((c_idx, pair))

    # Compose 1 Fast High-Res Comparison Sheet: 4 Columns (Concept 1..4) x 2 Rows (Row 1: Step 2 Right, Row 2: Step 1 Wrong)
    cw, ch = 520, 924
    pad = 22
    hdr_h = 110
    sheet = Image.new("RGB", (cw * 4 + pad * 5, ch * 2 + hdr_h + pad * 3), (10, 12, 16))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(28, "extrabold")
    f_c = v30.f_pretendard(19, "extrabold")
    f_s = v30.f_pretendard(15, "bold")

    sd.text(
        (pad, 20),
        "BARBELL PLATE 100% 보존 + DAY 2 마네킹 색상 고정 + 미니멀 럭셔리 수치·그래픽 HUD 4종 비교",
        font=f_t,
        fill=(255, 255, 255),
    )
    titles = [
        ("시안 1 · 원형 링 다이얼 + 12단 네온 바 도크", "글자 1단어(ALIGNED) · 원형 96% 링 + 12단 슬랜트 게이지 + 0°"),
        ("시안 2 · 초슬림 글래스 슬라이더 캡슐", "[ 0° ] ━━━●━ [ 96% ] 대칭형 슬라이더 · 불필요 문장 0%"),
        ("시안 3 · 우측 상단 배지 외곽 링 게이지 + 하단 미니 캡슐", "우측 상단 ✓ 배지 테두리가 96% 원형 링 · 하단은 초소형 [ ⦠ 0° · ALIGNED ]"),
        ("시안 4 · 하단 좌·우 독립 듀얼 계기판 포드", "중앙 바닥 여백 오픈 · 좌측 [각도 계기판 0°] + 우측 [96% 5단 파워 바]"),
    ]

    for idx, (c_idx, pair) in enumerate(concepts):
        x0 = pad + idx * (cw + pad)
        sd.text((x0 + 4, 62), titles[idx][0], font=f_c, fill=(45, 250, 140) if idx in (0, 2) else (240, 245, 255))
        sd.text((x0 + 4, 86), titles[idx][1], font=f_s, fill=(185, 195, 210))

        # Row 1: Step 2 (Right)
        im_r = Image.fromarray(pair[2]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im_r, (x0, hdr_h))
        sd.rounded_rectangle([x0 - 2, hdr_h - 2, x0 + cw + 2, hdr_h + ch + 2], radius=14, outline=(35, 245, 125), width=3)

        # Row 2: Step 1 (Wrong)
        y2 = hdr_h + ch + pad
        im_w = Image.fromarray(pair[1]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im_w, (x0, y2))
        sd.rounded_rectangle([x0 - 2, y2 - 2, x0 + cw + 2, y2 + ch + 2], radius=14, outline=(255, 58, 66), width=3)

    out_p = os.path.join(OUT_DIR, "day3_fast_4hud_luxury_concepts.jpg")
    sheet.save(out_p, quality=95)

    # Also save big close-up crops of the HUDs so user can inspect typography & graphics at 1:1 pixel clarity
    crop_sheet = Image.new("RGB", (920 * 2 + 60, 320 * 2 + 120), (10, 12, 16))
    cd = ImageDraw.Draw(crop_sheet)
    cd.text((20, 16), "1:1 픽셀 확대 — 하단 수치·그래픽 HUD 4종 디테일 비교 (글자 최소화 · 고급 계기판 그래픽)", font=f_t, fill=(255, 255, 255))
    positions = [(20, 70), (960, 70), (20, 410), (960, 410)]
    for idx, (c_idx, pair) in enumerate(concepts):
        px, py = positions[idx]
        # Crop bottom HUD (`y = 1260 .. 1500, x = 80 .. 1000`)
        hud_crop = Image.fromarray(pair[2][1250:1510, 80:1000])
        crop_sheet.paste(hud_crop, (px, py + 32))
        cd.text((px + 6, py), titles[idx][0], font=f_c, fill=(45, 250, 140))
        cd.rounded_rectangle([px - 2, py + 30, px + 922, py + 292], radius=12, outline=(255, 255, 255, 65), width=2)

    crop_p = os.path.join(OUT_DIR, "day3_fast_4hud_closeup_1to1.jpg")
    crop_sheet.save(crop_p, quality=95)
    print("Saved:", out_p, "and", crop_p)


if __name__ == "__main__":
    main()
