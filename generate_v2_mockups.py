#!/usr/bin/env python3
"""
Generates V2 Mockup Boards addressing 중현 님's feedback ([t392u]–[t395u]):
1. Title at Top (`y = 352`, right below the 4-slot preview bar `y = 132..274`)
2. Bottom Area Filled (`[t393u]: "하단에 있는걸 상단으로 올렸으니까 하단영역 비어보이지 않게"`):
   - Shifts the 3D squatting mannequin down into the lower-center sweet spot so the head sits cleanly below the top cursive title (`y ~ 550`) and the feet extend right down to `y ~ 1650` (just above `@gymform.3d` at `y = 1694`), leaving ZERO empty gap at the bottom!
   - Also compares:
     * 하단 시안 A (마네킹 하향 풀사이즈 밀착 + 좌측 하단 각도 콜아웃 재배치): 마네킹 전신이 상단 타이틀(`y=352`) 아래부터 `@gymform.3d`(`y=1694`) 바로 위(`y=1660`)까지 꽉 차고, 좌측 각도/위험도 콜아웃(`0° KNEE LOCK` / `22° HIP DRIVE`)도 하단 밸런스 영역에 배치
     * 하단 시안 B (마네킹 하향 밀착 + 하단 세이프존 네온 근활성 게이지 바 `y=1585..1648`): Day 1·Day 2처럼 하단 세이프존(`@gymform.3d` 바로 위)에 타겟 근육 활성도 게이지 바(`QUADS 98% · GLUTES 95%`)를 함께 배치해 하단 밀도를 꽉 채움
3. Dynamic Non-Front Grid Covers (`[t394u]: "매번 정면으로 그리드 커버가 들어가면 심심할듯"`):
   - Compares 3 dynamic 3D perspective covers right next to Day 2 (Rear 3/4 Lats) and Day 1 (Side 3/4 Triceps) on the `@gymform.3d` 3-column profile grid:
     * 그리드 커버 1안 (Step 4 · 후측면 3/4 대각선 앵글 · 네온 그린 `22° HIP DRIVE / 98% QUAD & GLUTE`)
     * 그리드 커버 2안 (Step 2 · 측면 3/4 대각선 앵글 · 네온 레드 X-Ray `41° FORWARD LEAN / 79% LUMBAR LOAD`)
     * 그리드 커버 3안 (Step 3 · 정면 앵글 + 하단 게이지 바 · 네온 그린 `0° KNEE LOCK` 비교용)
"""
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

# Look at why horizontal plates clipped in original: only Step 1 & Step 3 (front view wide barbell) had barbell ends near x=20..1060.
# If we use horiz_scale = 0.88 and vert_scale = 0.96 (or uniform 0.90 scale anchored near the bottom y=1660!),
# wait: let's keep 100% pure uniform 3D aspect ratio (zero distortion!) with scale = 0.90,
# and set the vertical shift `STEP_DY_BOTTOM_FULL` so that in every step:
# - The head/barbell top is around y = 550..590 (right below the top cursive title "Barbell Back Squat" at y = 352..415)
# - The feet/ankles reach right down to y = 1620..1655 (right above @gymform.3d at y = 1694)!
# Let's check exact coordinates with uniform scale = 0.90 and custom center y!

SCALE = 0.90
# With uniform scale = 0.90 around (W/2, H/2=960) plus dy:
# y_new = (y_1080_dy0 - 960) * 0.90 + 960 + dy
# Wait: for Step 1 (dy0: sh_y=869, ank_y=1451, head_top~710, foot_bot~1520):
#   If dy = +165: head_top = (710-960)*0.9 + 960 + 165 = 900, ank_y = (1451-960)*0.9 + 960 + 165 = 1567 (foot_bot ~ 1630)!
#   And in standing frame (f_local=4, sh_y=510): head_top = (360-960)*0.9 + 960 + 165 = 585 (cleanly below y=352 title!).
# Let's check each step's standing sh_y and deep-squat ank_y so both standing and deep-squat fit y=440..1665!

STEP_DY_FULL = {
    1: +155,  # Standing head ~ 560, deep squat foot bottom ~ 1635
    2: +35,   # Step 2 native camera is already lower (ank_y=1604 -> foot bottom ~ 1645)
    3: +125,  # Standing head ~ 555, deep squat foot bottom ~ 1640
    4: +50,   # Step 4 native ank_y=1588 -> foot bottom ~ 1645
}


def transform_frame_and_kp_bottom_full(bgr_720p: np.ndarray, kp_dy0: dict, step_idx: int) -> tuple[np.ndarray, dict]:
    """
    Uniform 0.90x scale (zero aspect distortion, 100% pure 3D geometry) so barbell plates stay inside
    iPhone 19.5:9 horizontal safe zone (vx = 49..1031), while positioning the mannequin lower (STEP_DY_FULL)
    so the top title at y=352 has breathing room and the mannequin's feet fill the bottom down to y=1645!
    """
    up = cv2.resize(bgr_720p, (W, H), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 0.9)
    sharp_bgr = cv2.addWeighted(up, 1.22, blur, -0.22, 0)
    rgb = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    dy = STEP_DY_FULL[step_idx]
    nw, nh = int(round(W * SCALE)), int(round(H * SCALE))
    scaled = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # Fill studio background seamlessly using perimeter charcoal tone
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

    # Soft feather on outer edges so background transition is 100% invisible
    yy_s, xx_s = np.mgrid[:ch, :cw].astype(np.float32)
    feather = 36.0
    wx = np.minimum(np.clip(xx_s / feather, 0.0, 1.0), np.clip((cw - 1 - xx_s) / feather, 0.0, 1.0))
    wy = np.minimum(np.clip(yy_s / feather, 0.0, 1.0), np.clip((ch - 1 - yy_s) / feather, 0.0, 1.0))
    w_mask = (wx * wy)[:, :, None]

    canvas[dy0 : dy0 + ch, dx0 : dx0 + cw] = (
        canvas[dy0 : dy0 + ch, dx0 : dx0 + cw] * (1.0 - w_mask) + sub * w_mask
    )

    kp_out = {}
    for k, (px, py) in kp_dy0.items():
        nx = (px - W * 0.5) * SCALE + W * 0.5
        ny = (py - H * 0.5) * SCALE + H * 0.5 + dy
        kp_out[k] = (float(nx), float(ny))

    return canvas, kp_out


def draw_top_cursive_title(canvas_pil: Image.Image, exercise_name: str = "Barbell Back Squat", center_y: int = 348):
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


def draw_bottom_activation_bar(canvas_pil: Image.Image, step_idx: int, is_wrong: bool):
    """
    Sleek bottom safe-zone biomechanical gauge bar (`y = 1568..1648`, right above `@gymform.3d` at `y = 1694`)
    inspired by Day 1 / Day 2 so the bottom area looks authoritative, dense, and zero-empty!
    Stays strictly inside horizontal safe zone (`x = 86 .. 994`).
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    bx0, by0, bx1, by1 = 86, 1566, 994, 1650
    col = (255, 92, 42) if is_wrong else (38, 245, 132)

    # Frosted dark glass pill
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=20, fill=(16, 19, 24, 215), outline=(*col, 150), width=2)

    if step_idx == 1:
        tag_l = "VALGUS TORQUE · ACL STRAIN"
        tag_r = "KNEE STABILITY  24%"
        bar_ratio = 0.24
    elif step_idx == 2:
        tag_l = "LUMBAR SHEAR · SPINE OVERLOAD"
        tag_r = "SPINE SAFETY  29%"
        bar_ratio = 0.29
    elif step_idx == 3:
        tag_l = "NEUTRAL KNEE TRACKING · HIP TORQUE"
        tag_r = "JOINT ALIGNMENT  96%"
        bar_ratio = 0.96
    else:
        tag_l = "FULL DEPTH · QUAD & GLUTE DRIVE"
        tag_r = "MUSCLE ACTIVATION  98%"
        bar_ratio = 0.98

    f_l = v30.f_pretendard(18, "extrabold")
    f_r = v30.f_pretendard(18, "extrabold")

    # Status dot + left label
    d.ellipse([bx0 + 20, by0 + 19, bx0 + 32, by0 + 31], fill=(*col, 255))
    d.text((bx0 + 42, by0 + 14), tag_l, font=f_l, fill=(238, 243, 250, 255))

    # Right metric
    bb_r = d.textbbox((0, 0), tag_r, font=f_r)
    rw = bb_r[2] - bb_r[0]
    d.text((bx1 - 22 - rw, by0 + 14), tag_r, font=f_r, fill=(*col, 255))

    # Progress bar track
    px0, py0, px1, py1 = bx0 + 22, by0 + 50, bx1 - 22, by0 + 64
    d.rounded_rectangle([px0, py0, px1, py1], radius=7, fill=(42, 47, 56, 230))
    fill_w = int(round((px1 - px0) * bar_ratio))
    d.rounded_rectangle([px0, py0, px0 + max(18, fill_w), py1], radius=7, fill=(*col, 255))

    canvas_pil.alpha_composite(ov)


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
    # Safe-zone horizontal placement: tx = 86 (well inside vx = 49.2 left crop!)
    tx = 86
    elbow_x = max(286, target_pt[0] - 42)
    elbow_y = target_pt[1] - 24
    bcd._draw_precision_leader(rd, target_pt, (elbow_x, elbow_y), (278, elbow_y), col)

    ty = int(np.clip(elbow_y - 76, 480, 1400))
    f_big = v30.f_pretendard(84, "extrabold")
    f_l1 = v30.f_pretendard(21, "extrabold")
    f_l2 = v30.f_pretendard(16, "bold")

    rd.text((tx + 2, ty + 3), deg_str, font=f_big, fill=(0, 0, 0, 215))
    rd.text((tx, ty), deg_str, font=f_big, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 90), l1_str, font=f_l1, fill=(col[0], col[1], col[2], 255))
    rd.text((tx + 2, ty + 117), l2_str, font=f_l2, fill=(215, 222, 232, 240))

    return np.array(Image.alpha_composite(base_pil, ov).convert("RGB"))


def build_v2_boards():
    peak_raw = {}
    peak_kp = {}
    peak_shaded = {}
    icons_all_4 = []

    for s_idx, (fn, i0, i1, is_w, v_mode, _) in b4.STEP_SPECS.items():
        seq = b4.load_native_slice(os.path.join(OUT_DIR, fn), i0, i1)
        kps_0 = b4.extract_smoothed_kps_1080(seq, s_idx, 0)
        bgr_peak = seq[-1]
        kp_peak_0 = kps_0[-1]
        rgb_1080, kp_1080 = transform_frame_and_kp_bottom_full(bgr_peak, kp_peak_0, s_idx)
        sh_rgb, sh_mask = b4.shade_native_3d_frame_1080p(rgb_1080, s_idx, is_w, v_mode, kp_1080, pulse_phase=0.75)
        peak_raw[s_idx] = rgb_1080
        peak_kp[s_idx] = kp_1080
        peak_shaded[s_idx] = (sh_rgb, sh_mask)
        icons_all_4.append(b4._make_icon_pair(sh_rgb, kp_1080))

    step_meta = {
        1: (True,  0, "16°", "KNEE CAVE",    "86% JOINT RISK"),
        2: (True,  1, "41°", "FORWARD LEAN", "79% LUMBAR LOAD"),
        3: (False, 2, "0°",  "KNEE LOCK",    "96% JOINT SAFE"),
        4: (False, 3, "22°", "HIP DRIVE",    "98% QUAD & GLUTE"),
    }

    rendered_clean_bottom = {}
    rendered_bar_bottom = {}

    for s_idx in [1, 2, 3, 4]:
        is_w, slot_i, deg_s, l1_s, l2_s = step_meta[s_idx]
        sh_rgb, sh_mask = peak_shaded[s_idx]
        kp = peak_kp[s_idx]
        fr = draw_step_laser_and_safe_callout(sh_rgb, sh_mask, s_idx, kp, is_w, deg_s, l1_s, l2_s)
        fr = b4.draw_compact_demic_top_preview(fr, icons_all_4, active_slot=slot_i, is_wrong=is_w)

        # Version A: Mannequin shifted down to fill bottom (`y=1645` feet right above `@gymform.3d`)
        pil_a = Image.fromarray(fr).convert("RGBA")
        draw_top_cursive_title(pil_a, "Barbell Back Squat", center_y=348)
        rendered_clean_bottom[s_idx] = pil_a.convert("RGB")

        # Version B: Mannequin + Sleek Bottom Safe-Zone Muscle Activation Bar (`y=1566..1650`)
        pil_b = Image.fromarray(fr).convert("RGBA")
        draw_top_cursive_title(pil_b, "Barbell Back Squat", center_y=348)
        draw_bottom_activation_bar(pil_b, s_idx, is_w)
        rendered_bar_bottom[s_idx] = pil_b.convert("RGB")

    # Simulate exact iPhone 1179x2556 Reels viewport with actual Instagram bottom/top/right UI overlays
    iphone_cap = Image.open(IPHONE_CAP_PATH).convert("RGB")
    iw, ih = iphone_cap.size  # 1179 x 2556

    def simulate_iphone_screen(reel_1080x1920: Image.Image) -> Image.Image:
        # iPhone scales 1080x1920 to height 2306 (width 1297) and crops center 1179px horizontally
        target_h = 2306
        target_w = int(round(1080 * (target_h / 1920.0)))  # 1297
        scaled = reel_1080x1920.resize((target_w, target_h), Image.Resampling.LANCZOS)
        crop_x0 = (target_w - iw) // 2  # 59
        cropped = scaled.crop((crop_x0, 0, crop_x0 + iw, target_h))

        screen = iphone_cap.copy()
        screen.paste(cropped, (0, 0))

        # Overlay actual Instagram Reels UI elements from user's screenshot (status bar, back/add icons, right icons, bottom @gymform.3d + Edits bar)
        # 1. Top status bar (y = 0..145)
        screen.paste(iphone_cap.crop((0, 0, iw, 145)), (0, 0))
        # 2. Bottom @gymform.3d profile & Edits insights bar (y = 2030..2556)
        # Blend y=2010..2306 smoothly where the @gymform.3d row sits
        ui_bot = iphone_cap.crop((0, 2035, iw, ih))
        screen.paste(ui_bot, (0, 2035))
        # Also draw a subtle indicator box around @gymform.3d row so user can see zero overlap
        return screen

    sim_prev_empty = Image.open(os.path.join(OUT_DIR, "_mockup_iphone_safezone_comparison.jpg"))
    sim_a_step3 = simulate_iphone_screen(rendered_clean_bottom[3])
    sim_b_step3 = simulate_iphone_screen(rendered_bar_bottom[3])
    sim_b_step4 = simulate_iphone_screen(rendered_bar_bottom[4])

    # Build Board 1: Bottom Area Fill Comparison on Actual iPhone Screen (3 panels side-by-side)
    pw, ph = 460, 997
    pad = 26
    head_h = 118
    board1 = Image.new("RGB", (pad + 3 * (pw + pad), head_h + ph + pad), (14, 16, 20))
    b1d = ImageDraw.Draw(board1)
    f_h1 = v30.f_pretendard(28, "extrabold")
    f_h2 = v30.f_pretendard(18, "bold")
    f_lbl = v30.f_pretendard(19, "extrabold")

    b1d.text(
        (28, 18),
        "상단 타이틀(y=348) 고정 + 하단 영역 빈틈 해결 시안 비교 (아이폰 실제 화면 기준)",
        font=f_h1,
        fill=(245, 248, 255),
    )
    b1d.text(
        (28, 60),
        "✓ 마네킹을 하향 밸런스 배치해 상단 타이틀과 겹치지 않으면서 발끝이 @gymform.3d 바로 위(y=1650)까지 꽉 차게 설계",
        font=f_h2,
        fill=(110, 245, 165),
    )

    panels_b1 = [
        (sim_a_step3, "하단 A안 · 마네킹 전신 하향 밀착 (발끝 y=1650)", (65, 245, 135)),
        (sim_b_step3, "하단 B안 (추천) · 전신 밀착 + 하단 근활성 게이지 바", (55, 215, 255)),
        (sim_b_step4, "하단 B안 (Step 4 후측면 3/4 앵글 적용 예시)", (255, 195, 70)),
    ]
    for i, (im_s, title_s, col_s) in enumerate(panels_b1):
        x0 = pad + i * (pw + pad)
        y0 = head_h
        board1.paste(im_s.resize((pw, ph), Image.Resampling.LANCZOS), (x0, y0))
        b1d.rounded_rectangle([x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 2], radius=14, outline=col_s, width=3)
        b1d.rounded_rectangle([x0 + 10, y0 + 10, x0 + pw - 10, y0 + 48], radius=8, fill=(10, 12, 16))
        b1d.text((x0 + 18, y0 + 17), title_s, font=f_lbl, fill=col_s)

    p_b1 = os.path.join(OUT_DIR, "_mockup_v2_bottom_filled_safezone.jpg")
    board1.save(p_b1, quality=95)
    print("Saved Board 1:", p_b1)

    # Build Board 2: Dynamic Non-Front Grid Covers next to Day 2 (Rear 3/4) & Day 1 (Side 3/4)
    day1_cov = Image.open(os.path.join(OUT_DIR, "day1_v36_cover.jpg")).convert("RGB")
    day2_cov = Image.open(os.path.join(OUT_DIR, "day2_v38_cover.jpg")).convert("RGB")

    # Option 1: Step 4 Rear 3/4 Dynamic Angle (Green Quad & Glute Drive + Bottom Gauge Bar)
    cov_opt1 = rendered_bar_bottom[4].copy()
    # Option 2: Step 2 Side 3/4 Dynamic Angle (Red Lumbar Spine X-Ray Overload + Bottom Gauge Bar)
    cov_opt2 = rendered_bar_bottom[2].copy()
    # Option 3: Dynamic Diagonal Split (Left: Step 2 Side Red 41° Forward Lean | Right: Step 4 Rear 3/4 Green 22° Hip Drive)
    cov_opt3 = Image.new("RGB", (W, H), (22, 24, 28))
    arr_s2 = np.array(rendered_bar_bottom[2])
    arr_s4 = np.array(rendered_bar_bottom[4])
    # Center-crop each mannequin into its half so both full dynamic 3/4 bodies are clearly visible in the 3:4 grid!
    left_half = arr_s2[:, 210:750, :]
    right_half = arr_s4[:, 270:810, :]
    comp_split = np.concatenate([left_half, right_half], axis=1)
    cov_opt3 = Image.fromarray(comp_split)
    d_sp = ImageDraw.Draw(cov_opt3)
    d_sp.line([(W // 2, 280), (W // 2, 1650)], fill=(255, 255, 255), width=4)
    # Re-draw top preview & cursive title cleanly across the split cover
    fr_sp = b4.draw_compact_demic_top_preview(np.array(cov_opt3), icons_all_4, active_slot=3, is_wrong=False)
    pil_sp = Image.fromarray(fr_sp).convert("RGBA")
    draw_top_cursive_title(pil_sp, "Barbell Back Squat", center_y=348)
    draw_bottom_activation_bar(pil_sp, 4, False)
    cov_opt3 = pil_sp.convert("RGB")

    def crop_grid_3x4(img_1080x1920: Image.Image, tw: int = 360, th: int = 480) -> Image.Image:
        # Instagram profile Reels grid displays a centered 3:4 vertical crop (1080 x 1440, y = 240..1680)
        c = img_1080x1920.crop((0, 240, 1080, 1680))
        return c.resize((tw, th), Image.Resampling.LANCZOS)

    tw, th = 360, 480
    g_pad = 16
    row_h = th + 76
    board2_w = 40 + 3 * tw + 2 * g_pad + 40
    board2_h = 115 + 3 * row_h + 25
    board2 = Image.new("RGB", (board2_w, board2_h), (12, 14, 18))
    b2d = ImageDraw.Draw(board2)

    b2d.text(
        (32, 18),
        "@gymform.3d 프로필 그리드 썸네일 비교 — 정면 탈피 · 입체 대각선(3/4) 앵글 시안",
        font=f_h1,
        fill=(245, 248, 255),
    )
    b2d.text(
        (32, 58),
        "Day 1(측면 3/4 삼두) · Day 2(후측면 3/4 광배근) 옆에 놓였을 때 입체감과 시선 집중도가 가장 높은 3/4 앵글 비교",
        font=f_h2,
        fill=(175, 225, 255),
    )

    rows = [
        (
            cov_opt1,
            "커버 1안 (추천 · Step 4 후측면 3/4 그린 · 22° HIP DRIVE + 대퇴사두·둔근 발광)",
            (65, 245, 135),
        ),
        (
            cov_opt2,
            "커버 2안 (Step 2 측면 3/4 레드 X-Ray · 41° FORWARD LEAN + 요추 과부하 경고)",
            (255, 95, 78),
        ),
        (
            cov_opt3,
            "커버 3안 (입체 앵글 좌·우 반반 비교 · 측면 41° RED 오자세 vs 후측면 22° GREEN 정자세)",
            (85, 210, 255),
        ),
    ]

    for r_i, (cov_im, r_title, r_col) in enumerate(rows):
        y_top = 115 + r_i * row_h
        b2d.text((36, y_top), r_title, font=f_lbl, fill=r_col)
        y_img = y_top + 34

        cols = [
            (cov_im, "NEW · Day 3 (Squat)", r_col),
            (day2_cov, "Day 2 (Lat Pulldown · 후측면 3/4)", (140, 150, 165)),
            (day1_cov, "Day 1 (Rope Pushdown · 측면 3/4)", (140, 150, 165)),
        ]
        for c_i, (c_im, c_lbl, c_col) in enumerate(cols):
            x_img = 40 + c_i * (tw + g_pad)
            tile = crop_grid_3x4(c_im, tw, th)
            board2.paste(tile, (x_img, y_img))
            b2d.rounded_rectangle(
                [x_img - 2, y_img - 2, x_img + tw + 2, y_img + th + 2],
                radius=8,
                outline=c_col,
                width=3 if c_i == 0 else 1,
            )
            b2d.rounded_rectangle([x_img + 8, y_img + th - 34, x_img + 260, y_img + th - 8], radius=6, fill=(10, 12, 16))
            b2d.text((x_img + 14, y_img + th - 31), c_lbl, font=v30.f_pretendard(14, "bold"), fill=c_col)

    p_b2 = os.path.join(OUT_DIR, "_mockup_v2_dynamic_angle_grid_covers.jpg")
    board2.save(p_b2, quality=95)
    print("Saved Board 2:", p_b2)


if __name__ == "__main__":
    build_v2_boards()
