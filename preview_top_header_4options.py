#!/usr/bin/env python3
import os
import sys
import pickle
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import build_4step_bone_muscle_master as b4
import build_v30_pure_graphic_numbers as v30

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920
CURSIVE_FONT_PATH = "/workspace/daily_info_reels/fonts/KaushanScript-Regular.ttf"


def get_mannequin_fg_mask(rgb_1080: np.ndarray) -> np.ndarray:
    """Extracts smooth 0..1 foreground mask of the 3D mannequin + barbell against the studio charcoal background."""
    bg_med = np.median(
        np.concatenate([rgb_1080[:24, :, :].reshape(-1, 3), rgb_1080[:, :24, :].reshape(-1, 3), rgb_1080[:, -24:, :].reshape(-1, 3)], axis=0),
        axis=0,
    )
    diff = np.linalg.norm(rgb_1080.astype(np.float32) - bg_med[None, None, :], axis=-1)
    fg = np.clip((diff - 12.0) / 22.0, 0.0, 1.0).astype(np.float32)
    fg = cv2.GaussianBlur(fg, (0, 0), 2.5)
    return fg


def draw_translucent_preview_bar(
    frame_rgb: np.ndarray,
    torso_icons: list,
    active_slot: int,
    is_wrong: bool,
    bar_y0: int = 252,
    tw: int = 122,
    th: int = 138,
    bar_alpha: int = 68,
    thumb_alpha: int = 135,
    active_alpha: int = 175,
) -> np.ndarray:
    """
    Ultra-clean translucent glass 4-slot preview bar:
    - Mannequin head/barbell remains clearly visible through the glass and thumbnails!
    """
    bar_h = th - 14
    bar_y1 = bar_y0 + bar_h
    gap = 18
    total_w = 4 * tw + 3 * gap + 36
    bar_x0 = (W - total_w) // 2
    bar_x1 = bar_x0 + total_w
    slot_centers_x = [bar_x0 + 18 + tw // 2 + i * (tw + gap) for i in range(4)]

    base_pil = Image.fromarray(frame_rgb).convert("RGBA")

    # Subtle translucent glass pill (NO heavy opaque background blur so mannequin stays sharp underneath!)
    glass_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glass_ov)
    gd.rounded_rectangle(
        [bar_x0, bar_y0, bar_x1, bar_y1],
        radius=22,
        fill=(18, 20, 26, bar_alpha),
        outline=(180, 195, 215, 85),
        width=2,
    )
    base_pil = Image.alpha_composite(base_pil, glass_ov)

    yy_t, xx_t = np.mgrid[:th, :tw].astype(np.float32)
    nx = (xx_t - tw * 0.5) / (tw * 0.48)
    ny = (yy_t - th * 0.52) / (th * 0.48)
    vign = np.clip(1.0 - 0.55 * ((nx * nx + ny * ny) ** 1.35), 0.15, 1.0)

    for i in range(4):
        cx = slot_centers_x[i]
        tx0, ty0 = cx - tw // 2, bar_y0 - 7
        if i != active_slot:
            _, inact_224 = torso_icons[i]
            inact_arr = cv2.resize(inact_224, (tw, th), interpolation=cv2.INTER_AREA)
            inact_cropped = np.clip(inact_arr * vign[:, :, None], 0, 255).astype(np.uint8)
            im_slot = Image.fromarray(inact_cropped).convert("RGBA")
            mask_slot = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(mask_slot).rounded_rectangle([3, 10, tw - 4, th - 11], radius=14, fill=thumb_alpha)
            mask_slot = mask_slot.filter(ImageFilter.GaussianBlur(radius=2))
            base_pil.paste(im_slot, (tx0, ty0), mask_slot)

    cx_act = slot_centers_x[active_slot]
    ax0, ay0 = cx_act - tw // 2, bar_y0 - 7
    ax1, ay1 = ax0 + tw, ay0 + th
    act_col = (255, 88, 48) if is_wrong else (42, 245, 135)

    act_224, _ = torso_icons[active_slot]
    act_arr = cv2.resize(act_224, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    edge_dist = np.minimum(np.minimum(xx_t, tw - 1 - xx_t), np.minimum(yy_t, th - 1 - yy_t))
    rim_glow = np.exp(-edge_dist / 12.0)[:, :, None] * 0.22
    comp_act = np.clip(act_arr * 1.06 + rim_glow * np.array(act_col, dtype=np.float32), 0, 255).astype(np.uint8)

    im_act = Image.fromarray(comp_act).convert("RGBA")
    mask_act = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(mask_act).rounded_rectangle([0, 0, tw - 1, th - 1], radius=18, fill=active_alpha)
    base_pil.paste(im_act, (ax0, ay0), mask_act)

    bdr_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd_d = ImageDraw.Draw(bdr_ov)
    bd_d.rounded_rectangle([ax0 - 2, ay0 - 2, ax1 + 2, ay1 + 2], radius=20, outline=(*act_col, 90), width=4)
    bd_d.rounded_rectangle([ax0, ay0, ax1, ay1], radius=18, outline=(*act_col, 240), width=3)
    base_pil = Image.alpha_composite(base_pil, bdr_ov)

    return np.array(base_pil.convert("RGB"))


def draw_cursive_title_custom(
    canvas_pil: Image.Image,
    exercise_name: str,
    center_y: int,
    font_size: int = 58,
    text_alpha: int = 245,
    hollow_glass: bool = False,
):
    SS = 2
    ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    f_scr = ImageFont.truetype(CURSIVE_FONT_PATH, font_size * SS)
    bb = od.textbbox((0, 0), exercise_name, font=f_scr)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = (W * SS - tw) // 2 - bb[0]
    ty = center_y * SS - th // 2 - bb[1]

    if hollow_glass:
        # Translucent glass fill + crisp neon/white rim so mannequin head shines right through the letters!
        for dx_s, dy_s in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)]:
            od.text((tx + dx_s, ty + dy_s), exercise_name, font=f_scr, fill=(255, 255, 255, 210))
        od.text((tx, ty), exercise_name, font=f_scr, fill=(185, 220, 255, 95))
    else:
        for dx_s, dy_s, a_s in [(0, 4 * SS, int(text_alpha * 0.65)), (-2 * SS, 2 * SS, int(text_alpha * 0.4))]:
            od.text((tx + dx_s, ty + dy_s), exercise_name, font=f_scr, fill=(8, 10, 14, a_s))
        od.text((tx, ty), exercise_name, font=f_scr, fill=(246, 250, 255, text_alpha))

    ov_1080 = ov.resize((W, H), Image.Resampling.LANCZOS)
    canvas_pil.alpha_composite(ov_1080)


def main():
    with open(b4.CACHE_KP_PKL, "rb") as f:
        kps_cached = pickle.load(f)

    # Load Step 3 (Front Correct) at STANDING (idx 0, where head/barbell is highest!) and DEEP SQUAT (idx -1)
    # Also load Step 1..4 peaks for the 4-slot icons
    icons_all_4 = []
    for s_idx, (fn, i0, i1, is_w, v_mode, y_sh) in b4.STEP_SPECS.items():
        seq = b4.load_native_slice(os.path.join(OUT_DIR, fn), i0, i1)
        rgb_p, kp_p = b4.transform_frame_and_kp_1080p(seq[-1], kps_cached[s_idx][-1], y_sh, s_idx)
        sh_p, _ = b4.shade_native_3d_frame_1080p(rgb_p, s_idx, is_w, v_mode, kp_p, pulse_phase=0.75)
        icons_all_4.append(b4._make_icon_pair(sh_p, kp_p))

    # Step 3 Standing frame (idx 0) — the exact moment the mannequin stands tall and touches the upper zone!
    fn3, i0_3, i1_3, is_w3, v_mode3, y_sh3 = b4.STEP_SPECS[3]
    seq3 = b4.load_native_slice(os.path.join(OUT_DIR, fn3), i0_3, i1_3)
    rgb_stand, kp_stand = b4.transform_frame_and_kp_1080p(seq3[0], kps_cached[3][0], y_sh3, 3)
    sh_stand, mask_stand = b4.shade_native_3d_frame_1080p(rgb_stand, 3, False, v_mode3, kp_stand, pulse_phase=0.1)
    base_stand = b4.draw_step_laser_and_callout(sh_stand, mask_stand, 3, kp_stand, False, "0°", "KNEE LOCK", "90% JOINT SAFE")
    fg_stand = get_mannequin_fg_mask(sh_stand)

    # Step 4 Deep Squat frame (idx -1) — for checking how the 3:4 Grid Cover looks!
    fn4, i0_4, i1_4, is_w4, v_mode4, y_sh4 = b4.STEP_SPECS[4]
    seq4 = b4.load_native_slice(os.path.join(OUT_DIR, fn4), i0_4, i1_4)
    rgb_deep, kp_deep = b4.transform_frame_and_kp_1080p(seq4[-1], kps_cached[4][-1], y_sh4, 4)
    sh_deep, mask_deep = b4.shade_native_3d_frame_1080p(rgb_deep, 4, False, v_mode4, kp_deep, pulse_phase=0.75)
    base_deep = b4.draw_step_laser_and_callout(sh_deep, mask_deep, 4, kp_deep, False, "22°", "HIP DRIVE", "98% QUAD & GLUTE")
    fg_deep = get_mannequin_fg_mask(sh_deep)

    # -------------------------------------------------------------------------
    # Build 4 Options on BOTH Standing (where overlap occurs) and Deep Squat
    # -------------------------------------------------------------------------
    # Option 1: 시스루 글래스 프리뷰 + 시스루 운동명 (마네킹이 닿으면 마네킹이 65% 투과되어 비침)
    def render_opt1(base_fr, fg_m, slot_i):
        ui_fr = draw_translucent_preview_bar(base_fr, icons_all_4, slot_i, False, bar_y0=252, tw=126, th=144, bar_alpha=65, thumb_alpha=130, active_alpha=175)
        pil = Image.fromarray(ui_fr).convert("RGBA")
        draw_cursive_title_custom(pil, "Barbell Back Squat", center_y=438, font_size=56, text_alpha=215, hollow_glass=False)
        ui_rgb = np.array(pil.convert("RGB")).astype(np.float32)
        # Wherever the mannequin foreground (fg_m) touches the top UI, let 55% of the mannequin shine right through!
        touch_w = (fg_m[:, :, None] * 0.55)
        out = np.clip(ui_rgb * (1.0 - touch_w) + base_fr.astype(np.float32) * touch_w, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    # Option 2: 운동명을 프리뷰 '위쪽'(y=218)으로 올리고 + 슬림 반투명 프리뷰(y=256..378) -> 마네킹 얼굴/가슴 침범 0%!
    def render_opt2(base_fr, fg_m, slot_i):
        ui_fr = draw_translucent_preview_bar(base_fr, icons_all_4, slot_i, False, bar_y0=258, tw=120, th=134, bar_alpha=70, thumb_alpha=140, active_alpha=185)
        pil = Image.fromarray(ui_fr).convert("RGBA")
        # Place title ABOVE the preview bar at y=214 (or top of grid at y=220) so y=390..1680 is 100% open for the mannequin!
        draw_cursive_title_custom(pil, "Barbell Back Squat", center_y=214, font_size=56, text_alpha=245, hollow_glass=False)
        ui_rgb = np.array(pil.convert("RGB")).astype(np.float32)
        touch_w = (fg_m[:, :, None] * 0.50)
        out = np.clip(ui_rgb * (1.0 - touch_w) + base_fr.astype(np.float32) * touch_w, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    # Option 3: 3D 백월(Depth-Behind) 레이어링 — 운동명이 마네킹 '뒤쪽 스튜디오 벽'에 배치되어 일어설 때 머리가 글자 앞으로 지나감!
    def render_opt3(base_fr, fg_m, slot_i):
        # First draw the title on the background BEFORE the mannequin foreground!
        bg_with_title = Image.fromarray(base_fr).convert("RGBA")
        draw_cursive_title_custom(bg_with_title, "Barbell Back Squat", center_y=438, font_size=60, text_alpha=235, hollow_glass=False)
        title_rgb = np.array(bg_with_title.convert("RGB")).astype(np.float32)
        # Composite mannequin IN FRONT OF the title (with 25% subtle glass bleed so letters are still readable)
        fg_front = np.clip(fg_m[:, :, None] * 0.78, 0.0, 0.78)
        layered = np.clip(title_rgb * (1.0 - fg_front) + base_fr.astype(np.float32) * fg_front, 0, 255).astype(np.uint8)
        # Then draw translucent preview bar (also letting mannequin shine through 50%)
        ui_fr = draw_translucent_preview_bar(layered, icons_all_4, slot_i, False, bar_y0=252, tw=124, th=140, bar_alpha=60, thumb_alpha=125, active_alpha=170)
        touch_p = (fg_m[:, :, None] * 0.48)
        out = np.clip(ui_fr.astype(np.float32) * (1.0 - touch_p) + layered.astype(np.float32) * touch_p, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    # Option 4: 초슬림 통합 캡슐 헤더 (프리뷰 상단 림 안에 운동명을 작고 세련되게 일체화, y=248..392에서 헤더 완전 종료!)
    def render_opt4(base_fr, fg_m, slot_i):
        ui_fr = draw_translucent_preview_bar(base_fr, icons_all_4, slot_i, False, bar_y0=282, tw=116, th=128, bar_alpha=65, thumb_alpha=135, active_alpha=180)
        pil = Image.fromarray(ui_fr).convert("RGBA")
        # Title at y=248 right inside top of 3:4 grid (y=240..1680), compact preview ends at y=403 -> zero overlap below y=403!
        draw_cursive_title_custom(pil, "Barbell Back Squat", center_y=246, font_size=48, text_alpha=245, hollow_glass=False)
        ui_rgb = np.array(pil.convert("RGB")).astype(np.float32)
        touch_w = (fg_m[:, :, None] * 0.50)
        out = np.clip(ui_rgb * (1.0 - touch_w) + base_fr.astype(np.float32) * touch_w, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    opts = [
        ("1안: 시스루 글래스 (닿으면 55% 투명 투과)", "현재 위치 유지 + 마네킹이 닿을 때 프리뷰·글자가 반투명하게 비침", render_opt1),
        ("2안: 상단 타이틀(y=214) + 반투명 프리뷰", "운동명을 프리뷰 위로 올려 서 있을 때도 얼굴·바벨 가림 0%", render_opt2),
        ("3안: 3D 인물 뒤쪽 배치 (Depth-Behind)", "운동명을 마네킹 뒤 스튜디오 벽면에 두어 머리가 글자 앞으로 통과", render_opt3),
        ("4안: 그리드 세이프 일체형 슬림 헤더", "운동명(y=246)+슬림 프리뷰(y=282)로 압축해 3:4 그리드·전신 모두 확보", render_opt4),
    ]

    pw, ph = 460, 818
    pad = 24
    board = Image.new("RGB", (pad + 4 * (pw + pad), 115 + ph + 55 + 420 + pad), (12, 14, 18))
    bd = ImageDraw.Draw(board)
    f_h1 = v30.f_pretendard(28, "extrabold")
    f_h2 = v30.f_pretendard(17, "bold")
    f_t = v30.f_pretendard(18, "extrabold")
    f_s = v30.f_pretendard(14, "bold")

    bd.text((28, 18), "상단 프리뷰 & 운동명 가림 해결 4가지 비교 시안 (서 있을 때 마네킹 머리·바벨 겹침 비교)", font=f_h1, fill=(245, 248, 255))
    bd.text((28, 56), "위쪽: 마네킹이 완전히 일어섰을 때(가장 높이 올라왔을 때) | 아래쪽: 상단 헤더 영역 확대 크롭 (y=160..540)", font=f_h2, fill=(85, 245, 155))

    for idx, (title_s, sub_s, fn_render) in enumerate(opts):
        im_stand = fn_render(base_stand, fg_stand, 2)
        x0 = pad + idx * (pw + pad)
        y0 = 105
        col = (65, 245, 135) if idx in (0, 2, 3) else (85, 215, 255)
        board.paste(im_stand.resize((pw, ph), Image.Resampling.LANCZOS), (x0, y0))
        bd.rounded_rectangle([x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 2], radius=12, outline=col, width=3)
        bd.rounded_rectangle([x0 + 8, y0 + 8, x0 + pw - 8, y0 + 64], radius=8, fill=(10, 12, 16))
        bd.text((x0 + 16, y0 + 14), title_s, font=f_t, fill=col)
        bd.text((x0 + 16, y0 + 40), sub_s, font=f_s, fill=(210, 220, 232))

        # Bottom row: 1:1 zoomed crop of the upper zone (y=160..540, x=140..940 -> 800x380 -> resized to 460x218)
        # AND 3:4 grid crop preview next to it
        crop_top = im_stand.crop((140, 160, 940, 540)).resize((pw, 218), Image.Resampling.LANCZOS)
        yc = y0 + ph + 42
        bd.text((x0 + 4, yc - 28), f"▲ 상단 겹침 구간 확대 (일어선 자세)", font=f_s, fill=(255, 215, 90))
        board.paste(crop_top, (x0, yc))
        bd.rounded_rectangle([x0 - 2, yc - 2, x0 + pw + 2, yc + 218 + 2], radius=10, outline=col, width=2)

        # Also show how the Deep Squat (Cover frame) looks in the 3:4 Grid crop (y=240..620)
        im_deep = fn_render(base_deep, fg_deep, 3)
        crop_cov = im_deep.crop((140, 220, 940, 560)).resize((pw, 195), Image.Resampling.LANCZOS)
        yc2 = yc + 218 + 34
        bd.text((x0 + 4, yc2 - 24), f"▲ 3:4 프로필 그리드 커버 상단 (앉은 자세)", font=f_s, fill=(180, 225, 255))
        board.paste(crop_cov, (x0, yc2))
        bd.rounded_rectangle([x0 - 2, yc2 - 2, x0 + pw + 2, yc2 + 195 + 2], radius=10, outline=(140, 160, 185), width=2)

    out_path = os.path.join(OUT_DIR, "_mockup_top_header_4options.jpg")
    board.save(out_path, quality=95)
    print("Saved 4-option header comparison:", out_path)


if __name__ == "__main__":
    main()
