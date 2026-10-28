#!/usr/bin/env python3
import os
import sys
import cv2
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import build_4step_bone_muscle_master as b4
import preview_v62_seamless_day2_master as p62
import build_clean_d4_variants as bcd
import build_v30_pure_graphic_numbers as v30

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920
CURSIVE_FONT_PATH = "/workspace/daily_info_reels/fonts/KaushanScript-Regular.ttf"
IPHONE_CAP_PATH = "/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/attachments/01M61ABFAJ4VZFGFZNZKS350GA.png"

SCALE = 0.90
# Raw 1080p (dy=0) deep squat ankle y & standing shoulder y:
# Step 1: stand sh=387, deep ank=1451 -> with SCALE=0.90, dy=+235: deep ank = (1451-960)*0.9 + 960 + 235 = 1637 (foot bottom ~ 1680)
# Step 2: stand sh=513, deep ank=1604 -> with SCALE=0.90, dy=+100: deep ank = (1604-960)*0.9 + 960 + 100 = 1640 (foot bottom ~ 1680)
# Step 3: stand sh=389, deep ank=1492 -> with SCALE=0.90, dy=+200: deep ank = (1492-960)*0.9 + 960 + 200 = 1639 (foot bottom ~ 1680)
# Step 4: stand sh=398, deep ank=1588 -> with SCALE=0.90, dy=+115: deep ank = (1588-960)*0.9 + 960 + 115 = 1640 (foot bottom ~ 1680)
STEP_DY = {
    1: +235,
    2: +100,
    3: +200,
    4: +115,
}


def transform_frame_and_kp_fast(bgr_720p: np.ndarray, kp_cached: dict, old_y_sh: int, step_idx: int) -> tuple[np.ndarray, dict]:
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


def draw_grid_safe_top_preview(frame_rgb: np.ndarray, torso_icons: list, active_slot: int, is_wrong: bool, bar_y0: int = 258) -> np.ndarray:
    """
    Places the 4-slot top preview bar at `bar_y0 = 258` (`y = 248..406`),
    which is 100% INSIDE the Instagram 3:4 profile grid crop (`y = 240..1680`)!
    Zero clipping in the profile grid AND sits cleanly below the Reels player top icons!
    """
    bar_x0, bar_x1 = 212, 868
    bar_y1 = bar_y0 + 142
    tw, th = 134, 158
    slot_centers_x = [303, 461, 619, 777]

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

    for i in range(4):
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
    act_arr = cv2.resize(act_224, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    edge_dist = np.minimum(np.minimum(xx_t, tw - 1 - xx_t), np.minimum(yy_t, th - 1 - yy_t))
    rim_glow = np.exp(-edge_dist / 14.0)[:, :, None] * 0.22
    comp_act = np.clip(act_arr * 1.06 + rim_glow * np.array(act_col, dtype=np.float32), 0, 255).astype(np.uint8)

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


def draw_top_cursive_title(canvas_pil: Image.Image, exercise_name: str = "Barbell Back Squat", center_y: int = 456):
    SS = 2
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    try:
        f_scr = ImageFont.truetype(CURSIVE_FONT_PATH, 58 * SS)
    except Exception:
        f_scr = v30.load_font(54 * SS, bold=True)

    bb = od.textbbox((0, 0), exercise_name, font=f_scr)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = (W * SS - tw) // 2 - bb[0]
    ty = center_y * SS - th // 2 - bb[1]

    for dx_s, dy_s, a_s in [(0, 5 * SS, 190), (-2 * SS, 3 * SS, 115), (2 * SS, 3 * SS, 115)]:
        od.text((tx + dx_s, ty + dy_s), exercise_name, font=f_scr, fill=(8, 10, 14, a_s))
    od.text((tx, ty), exercise_name, font=f_scr, fill=(246, 250, 255, 252))

    ov_1080 = ov.resize((W, H), Image.Resampling.LANCZOS)
    canvas_pil.alpha_composite(ov_1080)


def draw_step_laser_and_safe_callout(
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
    tx = 86
    elbow_x = max(286, target_pt[0] - 42)
    elbow_y = target_pt[1] - 24
    bcd._draw_precision_leader(rd, target_pt, (elbow_x, elbow_y), (278, elbow_y), col)

    ty = int(np.clip(elbow_y - 76, 520, 1460))
    f_big = v30.f_pretendard(84, "extrabold")
    f_l1 = v30.f_pretendard(21, "extrabold")
    f_l2 = v30.f_pretendard(16, "bold")

    rd.text((tx + 2, ty + 3), deg_str, font=f_big, fill=(0, 0, 0, 215))
    rd.text((tx, ty), deg_str, font=f_big, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 90), l1_str, font=f_l1, fill=(col[0], col[1], col[2], 255))
    rd.text((tx + 2, ty + 117), l2_str, font=f_l2, fill=(215, 222, 232, 240))

    return np.array(Image.alpha_composite(base_pil, ov).convert("RGB"))


def run_fast():
    with open(b4.CACHE_KP_PKL, "rb") as f:
        kps_cached = pickle.load(f)

    peak_kp = {}
    peak_shaded = {}
    icons_all_4 = []

    for s_idx, (fn, i0, i1, is_w, v_mode, old_y_sh) in b4.STEP_SPECS.items():
        seq = b4.load_native_slice(os.path.join(OUT_DIR, fn), i0, i1)
        bgr_peak = seq[-1]
        kp_c = kps_cached[s_idx][-1]
        rgb_1080, kp_1080 = transform_frame_and_kp_fast(bgr_peak, kp_c, old_y_sh, s_idx)
        sh_rgb, sh_mask = b4.shade_native_3d_frame_1080p(rgb_1080, s_idx, is_w, v_mode, kp_1080, pulse_phase=0.75)
        peak_kp[s_idx] = kp_1080
        peak_shaded[s_idx] = (sh_rgb, sh_mask)
        icons_all_4.append(b4._make_icon_pair(sh_rgb, kp_1080))

    step_meta = {
        1: (True,  0, "16°", "KNEE CAVE",    "86% JOINT RISK"),
        2: (True,  1, "41°", "FORWARD LEAN", "79% LUMBAR LOAD"),
        3: (False, 2, "0°",  "KNEE LOCK",    "96% JOINT SAFE"),
        4: (False, 3, "22°", "HIP DRIVE",    "98% QUAD & GLUTE"),
    }

    rendered = {}
    for s_idx in [1, 2, 3, 4]:
        is_w, slot_i, deg_s, l1_s, l2_s = step_meta[s_idx]
        sh_rgb, sh_mask = peak_shaded[s_idx]
        kp = peak_kp[s_idx]
        fr = draw_step_laser_and_safe_callout(sh_rgb, sh_mask, s_idx, kp, is_w, deg_s, l1_s, l2_s)
        # Place 4-slot preview at bar_y0 = 258 (y = 248..406 -> 100% inside 3:4 profile grid crop y=240..1680!)
        fr = draw_grid_safe_top_preview(fr, icons_all_4, active_slot=slot_i, is_wrong=is_w, bar_y0=258)
        pil = Image.fromarray(fr).convert("RGBA")
        # Place single-line cursive title right below the 4-slot preview at center_y = 458
        draw_top_cursive_title(pil, "Barbell Back Squat", center_y=458)
        rendered[s_idx] = pil.convert("RGB")

    # Save the clean Cover 1 (Step 4 Rear 3/4 Green)
    cov_path = os.path.join(OUT_DIR, "day3_cover_step4_rear34_clean.jpg")
    rendered[4].save(cov_path, quality=95)

    # Simulate iPhone Reels screen + 3:4 Profile Grid next to Day 2 & Day 1
    iphone_cap = Image.open(IPHONE_CAP_PATH).convert("RGB")
    iw, ih = iphone_cap.size  # 1179 x 2556

    def simulate_iphone_screen(reel_1080x1920: Image.Image) -> Image.Image:
        target_h = 2306
        target_w = int(round(1080 * (target_h / 1920.0)))
        scaled = reel_1080x1920.resize((target_w, target_h), Image.Resampling.LANCZOS)
        crop_x0 = (target_w - iw) // 2
        cropped = scaled.crop((crop_x0, 0, crop_x0 + iw, target_h))
        screen = iphone_cap.copy()
        screen.paste(cropped, (0, 0))
        screen.paste(iphone_cap.crop((0, 0, iw, 145)), (0, 0))
        screen.paste(iphone_cap.crop((0, 2035, iw, ih)), (0, 2035))
        return screen

    sim_s3 = simulate_iphone_screen(rendered[3])
    sim_s4 = simulate_iphone_screen(rendered[4])

    day1_cov = Image.open(os.path.join(OUT_DIR, "day1_v36_cover.jpg")).convert("RGB")
    day2_cov = Image.open(os.path.join(OUT_DIR, "day2_v38_cover.jpg")).convert("RGB")

    def crop_grid_3x4(img_1080x1920: Image.Image, tw: int = 380, th: int = 507) -> Image.Image:
        c = img_1080x1920.crop((0, 240, 1080, 1680))
        return c.resize((tw, th), Image.Resampling.LANCZOS)

    # Build Combined Final Verification Sheet:
    # Left: Actual iPhone Screen (Step 3 Front & Step 4 Rear 3/4, Zero Bottom HUD, Feet filling down to @gymform.3d)
    # Right: @gymform.3d 3-Column Profile Grid (Cover 1: Step 4 Rear 3/4 with 100% Unclipped Top Preview + Title, next to Day 2 & Day 1)
    board = Image.new("RGB", (2160, 1140), (12, 14, 18))
    bd = ImageDraw.Draw(board)
    f_h1 = v30.f_pretendard(28, "extrabold")
    f_h2 = v30.f_pretendard(18, "bold")
    f_lbl = v30.f_pretendard(18, "extrabold")

    bd.text(
        (32, 18),
        "Day 3 최종 교정 시안 — ① 하단 HUD 0%(전신 하단 밀착) · ② 커버 1안(Step 4 후측면 3/4) 상단 프리뷰 잘림 0% 해결",
        font=f_h1,
        fill=(245, 248, 255),
    )
    bd.text(
        (32, 58),
        "✓ 4칸 프리뷰를 그리드 세이프존(y=248..406)으로 내려 Day 2 프리뷰(y=228..366)와 동일한 높이로 정렬 (3:4 그리드 상단 잘림 0%)",
        font=f_h2,
        fill=(85, 245, 155),
    )

    # Left two iPhone screens (440 x 954 each)
    pw, ph = 440, 954
    for i, (im_s, lbl_s, col_s) in enumerate([
        (sim_s3, "아이폰 실제 화면 · Step 3 (하단 텍스트 0% · 발끝 밀착)", (65, 245, 135)),
        (sim_s4, "아이폰 실제 화면 · Step 4 (후측면 3/4 · 커버 1안 장면)", (85, 215, 255)),
    ]):
        x0 = 32 + i * (pw + 24)
        y0 = 115
        board.paste(im_s.resize((pw, ph), Image.Resampling.LANCZOS), (x0, y0))
        bd.rounded_rectangle([x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 2], radius=14, outline=col_s, width=3)
        bd.rounded_rectangle([x0 + 8, y0 + 8, x0 + pw - 8, y0 + 44], radius=8, fill=(10, 12, 16))
        bd.text((x0 + 16, y0 + 14), lbl_s, font=f_lbl, fill=col_s)

    # Right: @gymform.3d 3-Column Profile Grid (360 x 480 per tile)
    gx0 = 980
    gy0 = 150
    bd.text((gx0, 112), "@gymform.3d 3열 프로필 그리드 실제 크롭 화면 (3:4 비율 y=240..1680 · 상단 프리뷰 잘림 0%)", font=f_lbl, fill=(255, 215, 90))

    tw, th = 365, 487
    g_pad = 16
    cols = [
        (rendered[4], "NEW · Day 3 (커버 1안 · 프리뷰 100% 노출)", (65, 245, 135)),
        (day2_cov,    "Day 2 (Lat Pulldown · 후측면 3/4)", (150, 160, 175)),
        (day1_cov,    "Day 1 (Rope Pushdown · 측면 3/4)", (150, 160, 175)),
    ]
    for c_i, (c_im, c_lbl, c_col) in enumerate(cols):
        x_img = gx0 + c_i * (tw + g_pad)
        y_img = gy0
        tile = crop_grid_3x4(c_im, tw, th)
        board.paste(tile, (x_img, y_img))
        bd.rounded_rectangle([x_img - 2, y_img - 2, x_img + tw + 2, y_img + th + 2], radius=8, outline=c_col, width=3 if c_i == 0 else 1)
        bd.rounded_rectangle([x_img + 8, y_img + th - 36, x_img + tw - 8, y_img + th - 8], radius=6, fill=(10, 12, 16))
        bd.text((x_img + 14, y_img + th - 32), c_lbl, font=v30.f_pretendard(14, "bold"), fill=c_col)

    # Below the 3-column grid on the right: show all 4 steps peak frames side-by-side so user sees the whole 4-step flow
    bd.text((gx0, 675), "Day 3 전 구간(Step 1~4) 상단 프리뷰(y=248) + 상단 타이틀(y=458) + 하단 무텍스트 적용 미리보기", font=f_lbl, fill=(185, 225, 255))
    sw, sh_p = 212, 377
    for s_i in [1, 2, 3, 4]:
        x_s = gx0 + (s_i - 1) * (sw + 16)
        y_s = 710
        col_s = (255, 95, 78) if s_i in (1, 2) else (65, 245, 135)
        board.paste(rendered[s_i].resize((sw, sh_p), Image.Resampling.LANCZOS), (x_s, y_s))
        bd.rounded_rectangle([x_s - 2, y_s - 2, x_s + sw + 2, y_s + sh_p + 2], radius=8, outline=col_s, width=2)

    out_p = os.path.join(OUT_DIR, "_mockup_v3_final_grid_unclipped.jpg")
    board.save(out_p, quality=95)
    print("Saved V3 Final Mockup:", out_p)


if __name__ == "__main__":
    run_fast()
