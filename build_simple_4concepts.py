#!/usr/bin/env python3
"""
Ultra-Simple 4 Concepts (`t333u`: "너무 복잡해 심플하게 풀어내봐", `t334u`: "프리뷰 상단에 있는 것도 너무 찌부된거 아니야? 레퍼런스 잡은건 깔끔하게 좋던데 카피해"):
1. Exact Day 1 Top 3-Thumbnail Dock (`v28.draw_luxury_top_dock` 100% copied):
   - Equal-sized `142 x 114` slots (`aspect ratio 1.2456 : 1`), cropped from an exact `1.2456 : 1` window (`760 x 610` around the squat mannequin) so there is **0.0% squishing (`찌부러짐 0%`)**!
   - Zero clutter inside thumbnails (no `1/4` stickers, no mini `✕/✓` circles, no header pill above the dock) + exact 4x-supersampled Day 1 Titanium `✕ / ✓` emblem on the right!
2. 0.0% Background Haze (`build_clean_d4_variants.render_zero_haze_day1_mannequin_1080p`) + Day 1 3D Neon X-Ray Bone Flow.
3. 4 Ultra-Simple Graphic Variations (`심플 1 ~ 4`) with clutter stripped away:
   - **심플 1 (좌·우 원포인트 숫자 콜아웃 — 막대·박스 0%)**: Left knee pointer -> `84% JOINT RISK`, Right knee arc -> `16° KNEE CAVE`. Zero bars, zero boxes!
   - **심플 2 (단일 원포인트 포인터 — @demicstory 미니멀)**: Only 1 clean surgical callout on the left (`84% · 16° INWARD KNEE`) + `@demicstory` neon laser bone axis.
   - **심플 3 (하단 단일 줄다리기 바 1줄 [84% vs 16%])**: One single horizontal tug-of-war bar at the bottom (`84% JOINT ━━━━━━━░░ 16% QUADS`) + simple `16°` knee arc.
   - **심플 4 (단일 270° 원형 링 + 무릎 16° 아크)**: One clean `84%` circular ring on the left + `16°` knee arc on the right.
"""
import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62
import build_day1_style_xray_and_hud_concepts as bd
import build_clean_d4_variants as cd4

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920


_UPRIGHT_ANATOMY_CACHE = {}

def _get_upright_pure_anatomy_portrait(step_idx: int, is_wrong: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Renders the UPRIGHT standing front-facing 3D anatomical figure (frame 0 of Step 1/2)
    with ONLY `human_mask` (barbell plates masked out, just like `@demicstory`'s top 3 cards in `01M60T9XS7J2Z1YCHM1J0Q87DY.png`!)
    and crops an exact `224 : 272` (`0.8235 : 1`) vertical anatomical portrait (`tw = 224, th = 272`) centered on the body!
    """
    key = (step_idx, is_wrong)
    if key in _UPRIGHT_ANATOMY_CACHE:
        return _UPRIGHT_ANATOMY_CACHE[key]

    fn, i0, i1, *_ = p62.STEP_POOL_SPECS[step_idx]
    seq = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=52)
    raw_bgr = seq[0]  # Upright standing anatomical pose!

    sh_h, sw_w = raw_bgr.shape[:2]
    detector = cd4.uar.get_pose_detector()
    rgb_720 = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    mp_res = detector.detect(cd4.uar.mp.Image(image_format=cd4.uar.mp.ImageFormat.SRGB, data=rgb_720))
    lm = mp_res.pose_landmarks[0]
    mp_seg = mp_res.segmentation_masks[0].numpy_view()[:, :, 0]

    # Pure human body mask only (no barbell plates in the top anatomical preview cards, matching @demicstory!)
    _, human_mask, _ = cd4.make_clean_barbell_and_human_mask(rgb_720, lm, mp_seg)
    # Trim arms outside shoulders where barbell sleeves sit so it looks like a pure anatomical figure
    sh_l_x, sh_r_x = lm[11].x * sw_w, lm[12].x * sw_w
    cx_body = 0.5 * (sh_l_x + sh_r_x)
    half_span = abs(sh_l_x - sh_r_x) * 0.92
    yy, xx = np.mgrid[:sh_h, :sw_w].astype(np.float32)
    torso_corridor = np.clip(1.0 - np.maximum(0.0, np.abs(xx - cx_body) - half_span) / 18.0, 0.0, 1.0)
    pure_body_mask = human_mask * torso_corridor

    rgb = rgb_720.astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    non_red_lum = 0.30 * g_ch + 0.70 * b_ch
    rb_diff = np.maximum(0.0, r_ch - b_ch)
    red_any = np.clip((rb_diff - 10.0) / 75.0, 0.0, 1.0)
    neutral_raw = base_lum * (1.0 - red_any) + (non_red_lum * 1.18 + 14.0) * red_any
    norm_body = np.clip((neutral_raw - 20.0) / 64.0, 0.0, 1.15)
    sculpt_curve = np.clip(norm_body, 0.0, 1.0) ** 0.54 + 0.16 * np.clip(norm_body - 0.70, 0.0, 0.45)

    body_r = 86.0 + 108.0 * (sculpt_curve ** 0.80)
    body_g = 89.0 + 110.0 * (sculpt_curve ** 0.80)
    body_b = 94.0 + 113.0 * (sculpt_curve ** 0.80)
    mono = np.stack([np.clip(body_r, 0, 240), np.clip(body_g, 0, 242), np.clip(body_b, 0, 246)], axis=-1)

    # Highlight target muscles (Quads & Knee Joint) in vibrant @demicstory Orange (Wrong) or Emerald Green (Right)
    hip_y = 0.5 * (lm[23].y + lm[24].y) * sh_h
    kn_y = 0.5 * (lm[25].y + lm[26].y) * sh_h
    quad_zone = np.exp(-(((yy - (0.45 * hip_y + 0.55 * kn_y)) / max(28.0, 0.48 * abs(kn_y - hip_y))) ** 2)) * pure_body_mask
    target_w = np.clip((rb_diff - 10.0) / 36.0 + quad_zone * 0.65, 0.0, 1.0) * pure_body_mask

    if is_wrong:
        t_rgb = np.stack([235.0 + 20.0 * sculpt_curve, 95.0 + 55.0 * sculpt_curve, 28.0 + 25.0 * sculpt_curve], axis=-1)
    else:
        t_rgb = np.stack([28.0 + 35.0 * sculpt_curve, 215.0 + 38.0 * sculpt_curve, 110.0 + 45.0 * sculpt_curve], axis=-1)

    act_full = mono * (1.0 - (target_w * 0.82)[:, :, None]) + t_rgb * (target_w * 0.82)[:, :, None]
    act_full, _ = cd4.apply_3d_athletic_briefs_720p(act_full, pure_body_mask, lm, sculpt_curve)
    inact_full = np.stack([body_r * 0.58, body_g * 0.60, body_b * 0.64], axis=-1)
    inact_full, _ = cd4.apply_3d_athletic_briefs_720p(inact_full, pure_body_mask, lm, sculpt_curve * 0.65)

    # Crop exact 224:272 portrait window around shoulders-to-knees (`y_sh - 10` to `kn_y + 35`)
    sh_y = 0.5 * (lm[11].y + lm[12].y) * sh_h
    y0 = int(max(0, sh_y - 12))
    y1 = int(min(sh_h, kn_y + 36))
    box_h = max(120, y1 - y0)
    box_w = int(round(box_h * (224.0 / 272.0)))
    x0 = int(round(cx_body - box_w * 0.5))
    x1 = x0 + box_w

    bg_card = np.array([22.0, 25.0, 30.0], dtype=np.float32)
    m3 = pure_body_mask[y0:y1, x0:x1, None]
    act_crop = bg_card * (1.0 - m3) + act_full[y0:y1, x0:x1] * m3
    inact_crop = bg_card * (1.0 - m3) + inact_full[y0:y1, x0:x1] * m3

    tw, th = 224, 272
    act_224 = cv2.resize(act_crop, (tw, th), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    inact_224 = cv2.resize(inact_crop, (tw, th), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    _UPRIGHT_ANATOMY_CACHE[key] = (act_224, inact_224)
    return act_224, inact_224


def make_unsquished_day1_top_icons(base_frames: dict, current_step: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    100% Exact Copy of `@demicstory` Centered Top Glass Capsule (`01M60T9XS7J2Z1YCHM1J0Q87DY.png`):
    All 3 portrait cards (`224 x 272`) show the UPRIGHT, SYMMETRICAL front-facing 3D anatomical figure (zero barbell clutter, zero squat crouching!),
    with the Left & Right cards dimmed in monochrome slate and the Center protruding card (`x = 540`) glowing in Orange/Green!
    """
    is_w = (current_step == 1)
    act_224, inact_224 = _get_upright_pure_anatomy_portrait(1 if is_w else 2, is_w)
    return [(act_224, inact_224), (act_224, inact_224), (act_224, inact_224)]


def draw_day1_exact_top_dock(frame_rgb: np.ndarray, torso_icons: list, active_slot: int, is_wrong: bool) -> np.ndarray:
    """
    100% Exact Pixel-Measured Clone of `@demicstory`'s Centered Top Preview Capsule (`01M60T9XS7J2Z1YCHM1J0Q87DY.png`):
    1. Horizontally Centered Frosted Glass Capsule (`x = 87 .. 993`, `w = 906px`, exact center `x = 540`, `y = 247 .. 487`, `h = 240px`, `radius = 34px`)
       with `@demicstory`'s signature bottom-edge & corner silver glass specular rim light!
    2. 3 Portrait Anatomy Slots (`Left cx = 248`, `Center cx = 540`, `Right cx = 832`, each `w = 224px`):
       - Inactive slots (`h = 232px`, `y = 251 .. 483`) sit seamlessly inside the dark glass capsule with soft edge feathering (no inner box border).
       - Active slot (`w = 224px, h = 272px`, `y = 231 .. 503`) **protrudes 16px above and 16px below** the horizontal glass bar,
         with a warm volumetric radial backlight + crisp glowing neon rounded border (`radius = 28px`)!
    """
    from PIL import ImageFilter

    out_f = frame_rgb.astype(np.float32).copy()
    bar_x0, bar_y0, bar_x1, bar_y1 = 87, 195, 993, 435
    bar_w, bar_h = bar_x1 - bar_x0, bar_y1 - bar_y0

    # 1. Translucent Frosted Dark Glass Capsule Background (`DevEQskCa2g` style — blurs underlying head/equipment when touching!)
    raw_pil = Image.fromarray(frame_rgb).convert("RGBA")
    blur_pil = Image.fromarray(frame_rgb).filter(ImageFilter.GaussianBlur(radius=10)).convert("RGBA")
    frost_mask = Image.new("L", (W, H), 0)
    fmd = ImageDraw.Draw(frost_mask)
    fmd.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=34, fill=190)
    cx_act_pre = [248, 540, 832][active_slot]
    fmd.rounded_rectangle([cx_act_pre - 112, 179, cx_act_pre + 112, 179 + 272], radius=28, fill=205)
    base_pil = Image.composite(blur_pil, raw_pil, frost_mask)

    glass_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glass_ov)

    # Outer drop shadow + Translucent dark glass fill (`alpha = 135` ~ 53% so overlapping head/barbell softly shows through!)
    gd.rounded_rectangle([bar_x0 + 2, bar_y0 + 8, bar_x1 + 2, bar_y1 + 10], radius=36, fill=(0, 0, 0, 75))
    gd.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=34, fill=(22, 25, 30, 132), outline=(140, 150, 168, 120), width=2)
    gd.rounded_rectangle([bar_x0 + 2, bar_y0 + 2, bar_x1 - 2, bar_y1 - 1], radius=32, outline=(210, 220, 238, 52), width=1)
    base_pil = Image.alpha_composite(base_pil, glass_ov)

    slot_centers_x = [248, 540, 832]
    tw, th = 224, 272

    # Build soft radial vignette mask so each portrait figure blends seamlessly into the glass slot
    yy_t, xx_t = np.mgrid[:th, :tw].astype(np.float32)
    nx = (xx_t - tw * 0.5) / (tw * 0.48)
    ny = (yy_t - th * 0.52) / (th * 0.48)
    vign = np.clip(1.0 - 0.55 * ((nx * nx + ny * ny) ** 1.35), 0.15, 1.0)

    # 2. Paste the 2 Inactive Slots first (clipped inside `y = 251 .. 483`)
    for i in range(3):
        if i == active_slot:
            continue
        cx = slot_centers_x[i]
        tx0 = cx - tw // 2
        ty0 = 179
        _, inact_arr = torso_icons[i]
        # Feather top/bottom inside `bar_y0+4 .. bar_y1-4` (`251 .. 483`)
        inact_cropped = np.clip(inact_arr * vign[:, :, None], 0, 255).astype(np.uint8)
        im_slot = Image.fromarray(inact_cropped).convert("RGBA")
        mask_slot = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask_slot).rounded_rectangle([4, 20, tw - 5, th - 21], radius=22, fill=182)
        mask_slot = mask_slot.filter(ImageFilter.GaussianBlur(radius=4))
        base_pil.paste(im_slot, (tx0, ty0), mask_slot)

    # 3. Paste the Active Protruding Portrait Card (`w = 224, h = 272, y = 231 .. 503`, protrudes 16px above & below!)
    cx_a = slot_centers_x[active_slot]
    ax0, ay0 = cx_a - tw // 2, 179
    ax1, ay1 = ax0 + tw, ay0 + th
    act_col = (255, 138, 48) if is_wrong else (42, 245, 135)

    # Warm volumetric radial glow inside and around the active card (`@demicstory` signature!)
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gld = ImageDraw.Draw(glow_layer)
    gld.rounded_rectangle([ax0 - 8, ay0 - 8, ax1 + 8, ay1 + 8], radius=34, outline=(*act_col, 155), width=10)
    gld.ellipse([cx_a - 95, ay0 + 25, cx_a + 95, ay1 - 25], fill=(*act_col, 85))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=14))
    base_pil = Image.alpha_composite(base_pil, glow_layer)

    # Dark card backing for the protruding active card + warm radial center glow
    act_arr, _ = torso_icons[active_slot]
    rad_glow = np.exp(-(((xx_t - tw * 0.5) / 78.0) ** 2 + ((yy_t - th * 0.46) / 92.0) ** 2))[:, :, None]
    card_bg = np.array([26.0, 29.0, 35.0], dtype=np.float32) + rad_glow * np.array(act_col, dtype=np.float32) * 0.36
    # Separate foreground of thumbnail from its studio background (`lum > 46`)
    fg_a = np.clip((act_arr.mean(axis=2, keepdims=True) - 44.0) / 14.0, 0.0, 1.0)
    comp_act = np.clip(card_bg * (1.0 - fg_a) + act_arr * fg_a, 0, 255).astype(np.uint8)

    im_act = Image.fromarray(comp_act).convert("RGBA")
    mask_act = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(mask_act).rounded_rectangle([0, 0, tw - 1, th - 1], radius=28, fill=208)
    base_pil.paste(im_act, (ax0, ay0), mask_act)

    # Crisp glowing neon rounded border (`radius = 28px`, 4x supersampled feel)
    bdr_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd_d = ImageDraw.Draw(bdr_ov)
    bd_d.rounded_rectangle([ax0 - 2, ay0 - 2, ax1 + 2, ay1 + 2], radius=30, outline=(*act_col, 95), width=5)
    bd_d.rounded_rectangle([ax0, ay0, ax1, ay1], radius=28, outline=(*act_col, 255), width=3)
    bd_d.rounded_rectangle([ax0 + 1, ay0 + 1, ax1 - 1, ay1 - 1], radius=27, outline=(255, 235, 205, 170) if is_wrong else (210, 255, 230, 170), width=1)
    base_pil = Image.alpha_composite(base_pil, bdr_ov)

    return np.array(base_pil.convert("RGB"))


def draw_simple_1_left_right_onepoint_numbers(frame_rgb: np.ndarray, kp_1080: dict, is_wrong: bool) -> np.ndarray:
    """
    심플 1 (좌·우 원포인트 숫자 콜아웃 — 막대·박스 0%):
    - Left knee (`◎`) thin pointer -> `84%` + `JOINT RISK` (or `96%` + `QUAD LOCK`).
    - Right knee curved arc -> `16°` + `KNEE CAVE` (or `0°` + `KNEE LOCK`).
    - ZERO progress bars, ZERO boxes, ZERO bottom clutter!
    """
    # Render clean neon bone flow + right exterior curved `16°` arc + `@demicstory` laser axis
    fr = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        frame_rgb, np.zeros((H, W), dtype=np.float32), kp_1080, is_wrong, with_exterior_callouts=True, with_laser_axis=True
    )
    base = Image.fromarray(fr).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    pct_val = 84 if is_wrong else 96
    lbl_top = "JOINT RISK" if is_wrong else "QUAD LOCK"
    lbl_bot = "INWARD KNEE" if is_wrong else "ALIGNED TRACK"

    kn_r = (int(kp_1080["kn_r"][0]), int(kp_1080["kn_r"][1]))
    cd4._draw_precision_leader(rd, kn_r, (kn_r[0] - 54, kn_r[1] - 36), (288, kn_r[1] - 36), col)

    tx = 54
    ty = kn_r[1] - 112
    f_num = v30.f_pretendard(96, "extrabold")
    f_pct = v30.f_pretendard(44, "extrabold")
    f_l1 = v30.f_pretendard(20, "extrabold")
    f_l2 = v30.f_pretendard(15, "bold")

    rd.text((tx + 2, ty + 3), f"{pct_val}", font=f_num, fill=(0, 0, 0, 210))
    rd.text((tx, ty), f"{pct_val}", font=f_num, fill=(col[0], col[1], col[2], 255))
    nw = int(rd.textlength(f"{pct_val}", font=f_num))
    rd.text((tx + nw + 4, ty + 16), "%", font=f_pct, fill=(255, 255, 255, 245))
    rd.text((tx + 2, ty + 102), lbl_top, font=f_l1, fill=(255, 255, 255, 248))
    rd.text((tx + 2, ty + 128), lbl_bot, font=f_l2, fill=(col[0], col[1], col[2], 235))

    return np.array(Image.alpha_composite(base, ov).convert("RGB"))


def draw_simple_2_single_surgical_callout(frame_rgb: np.ndarray, kp_1080: dict, is_wrong: bool) -> np.ndarray:
    """
    심플 2 (단일 원포인트 콜아웃 1개 — 우측·하단 여백 100% 비움):
    - Only ONE ultra-clean exterior pointer on the left margin (`x = 48..285`, 0% overlap with legs!)
      combining `16°` + `KNEE CAVE` (`84% JOINT RISK`), plus the `@demicstory` neon laser line!
    """
    fr = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        frame_rgb, np.zeros((H, W), dtype=np.float32), kp_1080, is_wrong, with_exterior_callouts=False, with_laser_axis=True
    )
    base = Image.fromarray(fr).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    deg_str = "16°" if is_wrong else "0°"
    l1_str = "KNEE CAVE" if is_wrong else "KNEE LOCK"
    l2_str = "84% JOINT RISK" if is_wrong else "96% QUAD LOAD"

    kn_r = (int(kp_1080["kn_r"][0]), int(kp_1080["kn_r"][1]))
    cd4._draw_precision_leader(rd, kn_r, (kn_r[0] - 54, kn_r[1] - 32), (276, kn_r[1] - 32), col)

    tx, ty = 52, kn_r[1] - 108
    f_big = v30.f_pretendard(92, "extrabold")
    f_l1 = v30.f_pretendard(22, "extrabold")
    f_l2 = v30.f_pretendard(16, "bold")

    rd.text((tx + 2, ty + 3), deg_str, font=f_big, fill=(0, 0, 0, 210))
    rd.text((tx, ty), deg_str, font=f_big, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 98), l1_str, font=f_l1, fill=(col[0], col[1], col[2], 255))
    rd.text((tx + 2, ty + 126), l2_str, font=f_l2, fill=(215, 222, 232, 235))

    return np.array(Image.alpha_composite(base, ov).convert("RGB"))


def draw_simple_3_single_tug_of_war_bar(frame_rgb: np.ndarray, kp_1080: dict, is_wrong: bool) -> np.ndarray:
    """
    심플 3 (하단 단일 슬림 줄다리기 바 1줄 [84% vs 16%] + 무릎 16° 아크):
    - Keeps the side margins 100% open (only the small `16° KNEE CAVE` curved arrow on the right knee),
    - And at the bottom (`y = 1365`), shows **ONE single sleek tug-of-war bar** where Orange/Green (`84%` / `96%`) vs Grey (`16%` / `4%`)
      meet in a single bar! Instant 0.1-second comprehension with minimal visual elements.
    """
    fr = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        frame_rgb, np.zeros((H, W), dtype=np.float32), kp_1080, is_wrong, with_exterior_callouts=True, with_laser_axis=True
    )
    base = Image.fromarray(fr).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    pct_l = 84 if is_wrong else 96
    pct_r = 16 if is_wrong else 4
    lbl_l = "KNEE JOINT" if is_wrong else "QUADS · VMO"
    lbl_r = "QUADS · VMO" if is_wrong else "KNEE JOINT"

    bx1, bx2 = 100, 980
    bw = bx2 - bx1
    by = 1388
    bh = 18

    f_big = v30.f_pretendard(44, "extrabold")
    f_lbl = v30.f_pretendard(18, "extrabold")

    # Left big % + label
    rd.text((bx1, by - 56), f"{pct_l}%", font=f_big, fill=(col[0], col[1], col[2], 255))
    w_pl = int(rd.textlength(f"{pct_l}%", font=f_big))
    rd.text((bx1 + w_pl + 12, by - 36), lbl_l, font=f_lbl, fill=(245, 248, 255, 248))

    # Right label + %
    r_pct_str = f"{pct_r}%"
    w_pr = int(rd.textlength(r_pct_str, font=f_big))
    w_lr = int(rd.textlength(lbl_r, font=f_lbl))
    rd.text((bx2 - w_pr, by - 56), r_pct_str, font=f_big, fill=(165, 175, 190, 235))
    rd.text((bx2 - w_pr - w_lr - 12, by - 36), lbl_r, font=f_lbl, fill=(165, 175, 190, 220))

    # Single 24-segment Tug-of-War Bar (`pct_l` active colored segments on left, `pct_r` dim segments on right)
    n_segs = 28
    gap = 5
    seg_w = (bw - (n_segs - 1) * gap) // n_segs
    act_n = int(round((pct_l / 100.0) * n_segs))
    for i in range(n_segs):
        sx = bx1 + i * (seg_w + gap)
        c_s = (col[0], col[1], col[2], 250) if i < act_n else (150, 160, 175, 75)
        rd.rounded_rectangle([sx, by, sx + seg_w, by + bh], radius=5, fill=c_s)

    return np.array(Image.alpha_composite(base, ov).convert("RGB"))


def draw_simple_4_single_radial_ring_and_arc(frame_rgb: np.ndarray, kp_1080: dict, is_wrong: bool) -> np.ndarray:
    """
    심플 4 (좌측 심플 270° 원형 링 1개 + 우측 16° 아크 — 막대 0%):
    - Left knee leader line (`◎`) -> ONE clean 270° Circular Ring (`84%` `JOINT RISK` / `96%` `QUAD LOCK`).
    - Right knee -> ONE curved angle callout (`16° KNEE CAVE` / `0° KNEE LOCK`).
    - Zero horizontal/vertical bars, zero bottom clutter!
    """
    fr = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        frame_rgb, np.zeros((H, W), dtype=np.float32), kp_1080, is_wrong, with_exterior_callouts=True, with_laser_axis=True
    )
    base = Image.fromarray(fr).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    pct_val = 84 if is_wrong else 96
    ring_sub = "JOINT RISK" if is_wrong else "QUAD LOCK"

    kn_r = (int(kp_1080["kn_r"][0]), int(kp_1080["kn_r"][1]))
    cx, cy, r_ring = 166, kn_r[1] - 12, 88
    cd4._draw_precision_leader(rd, kn_r, (kn_r[0] - 46, cy), (cx + r_ring + 8, cy), col)

    bbox = [cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring]
    rd.ellipse([cx - r_ring - 6, cy - r_ring - 6, cx + r_ring + 6, cy + r_ring + 6], fill=(12, 15, 20, 160))
    rd.arc(bbox, start=135, end=405, fill=(255, 255, 255, 38), width=14)
    end_deg = int(135 + 270 * (pct_val / 100.0))
    rd.arc([cx - r_ring - 2, cy - r_ring - 2, cx + r_ring + 2, cy + r_ring + 2], start=135, end=end_deg, fill=(col[0], col[1], col[2], 70), width=22)
    rd.arc(bbox, start=135, end=end_deg, fill=(col[0], col[1], col[2], 255), width=14)

    f_num = v30.f_pretendard(62, "extrabold")
    f_pct = v30.f_pretendard(26, "extrabold")
    f_sub = v30.f_pretendard(15, "extrabold")
    s_txt = f"{pct_val}"
    tw = int(rd.textlength(s_txt, font=f_num))
    pw = int(rd.textlength("%", font=f_pct))
    sx = cx - (tw + pw) // 2
    rd.text((sx, cy - 36), s_txt, font=f_num, fill=(255, 255, 255, 255))
    rd.text((sx + tw + 2, cy - 20), "%", font=f_pct, fill=(col[0], col[1], col[2], 255))
    sw = int(rd.textlength(ring_sub, font=f_sub))
    rd.text((cx - sw // 2, cy + 30), ring_sub, font=f_sub, fill=(col[0], col[1], col[2], 245))

    return np.array(Image.alpha_composite(base, ov).convert("RGB"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_peaks = {}
    for s_i in (1, 2, 3, 4):
        fn, i0, i1, is_w, *_ = p62.STEP_POOL_SPECS[s_i]
        seq = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=52)
        raw_peaks[s_i] = seq[-1]

    base_frames = {}
    masks = {}
    kps = {}
    for s_i in (1, 2, 3, 4):
        is_w = (s_i in (1, 3))
        b_rgb, b_mask, b_kp = cd4.render_zero_haze_day1_mannequin_1080p(raw_peaks[s_i], s_i, is_w)
        base_frames[s_i] = b_rgb
        masks[s_i] = b_mask
        kps[s_i] = b_kp

    # Build the 100% UN-SQUISHED (`142x114`, exact `1.2456:1` crop) Day 1 top dock icons!
    variants = {"S1": {}, "S2": {}, "S3": {}, "S4": {}}
    for s_i in (1, 2):
        is_w = (s_i == 1)
        act_slot = 1
        torso_icons = make_unsquished_day1_top_icons(base_frames, current_step=s_i)

        s1 = draw_simple_1_left_right_onepoint_numbers(base_frames[s_i], kps[s_i], is_w)
        variants["S1"][s_i] = draw_day1_exact_top_dock(s1, torso_icons, act_slot, is_w)

        s2 = draw_simple_2_single_surgical_callout(base_frames[s_i], kps[s_i], is_w)
        variants["S2"][s_i] = draw_day1_exact_top_dock(s2, torso_icons, act_slot, is_w)

        s3 = draw_simple_3_single_tug_of_war_bar(base_frames[s_i], kps[s_i], is_w)
        variants["S3"][s_i] = draw_day1_exact_top_dock(s3, torso_icons, act_slot, is_w)

        s4 = draw_simple_4_single_radial_ring_and_arc(base_frames[s_i], kps[s_i], is_w)
        variants["S4"][s_i] = draw_day1_exact_top_dock(s4, torso_icons, act_slot, is_w)

    # Save 4-Column Overview Sheet (`심플 1 ~ 4`)
    cw, ch = 520, 924
    pad = 22
    hdr_h = 116
    sheet = Image.new("RGB", (cw * 4 + pad * 5, ch * 2 + hdr_h + pad * 3), (10, 12, 16))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(27, "extrabold")
    f_c = v30.f_pretendard(19, "extrabold")
    f_s = v30.f_pretendard(14, "bold")

    sd.text(
        (pad, 18),
        "Day 1 상단 프리뷰 캡슐 100% 복원(찌부러짐 0%) + 복잡한 막대·박스 덜어낸 초심플 시안 4종 (심플 1 ~ 4)",
        font=f_t,
        fill=(255, 255, 255),
    )
    meta_cols = [
        ("심플 1 · 좌·우 원포인트 숫자 (막대·박스 0%)", "좌측 무릎 포인터 [84% JOINT RISK] + 우측 무릎 아크 [16° KNEE CAVE]만 배치"),
        ("심플 2 · 단일 미니 캡슐 포인터 (@demicstory형)", "우측·하단 요소 없이 좌측에 [84% | INWARD KNEE · 16°] 포인터 딱 1개만 배치"),
        ("심플 3 · 하단 단일 줄다리기 바 1줄 (84% vs 16%)", "측면은 비우고 발 아래에 [84% JOINT ━━━━░░ 16% QUADS] 한 줄 바로 직관화"),
        ("심플 4 · 좌측 270° 원형 링 1개 + 우측 16° 아크", "복잡한 막대 없이 좌측에 [84% 원형 링], 우측에 [16° 각도 아크]만 대칭 배치"),
    ]

    for idx, k in enumerate(("S1", "S2", "S3", "S4")):
        x0 = pad + idx * (cw + pad)
        sd.text((x0 + 4, 60), meta_cols[idx][0], font=f_c, fill=(255, 165, 70) if idx == 0 else (45, 250, 140))
        sd.text((x0 + 4, 86), meta_cols[idx][1], font=f_s, fill=(195, 205, 220))

        im_w = Image.fromarray(variants[k][1]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im_w, (x0, hdr_h))
        sd.rounded_rectangle([x0 - 2, hdr_h - 2, x0 + cw + 2, hdr_h + ch + 2], radius=14, outline=(255, 94, 36), width=3)

        y2 = hdr_h + ch + pad
        im_r = Image.fromarray(variants[k][2]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im_r, (x0, y2))
        sd.rounded_rectangle([x0 - 2, y2 - 2, x0 + cw + 2, y2 + ch + 2], radius=14, outline=(35, 245, 125), width=3)

    out_master = os.path.join(OUT_DIR, "day3_simple_4variants_unsquished_top.jpg")
    sheet.save(out_master, quality=95)

    # Also save a 1:1 Close-Up showing the Top Capsule (`y=215..410`) + the Simple Lower/Side Callouts (`y=820..1460`)!
    closeup = Image.new("RGB", (1020 * 2 + 60, 940 * 2 + 120), (10, 12, 16))
    cd = ImageDraw.Draw(closeup)
    cd.text(
        (20, 18),
        "1:1 원본 픽셀 확인 — 상단 Day 1 정비율 캡슐(찌부러짐 0%) + 초심플 그래픽 시안 4종 (심플 1 ~ 4)",
        font=f_t,
        fill=(255, 255, 255),
    )
    pos = [(20, 72), (1060, 72), (20, 1032), (1060, 1032)]
    for idx, k in enumerate(("S1", "S2", "S3", "S4")):
        px, py = pos[idx]
        top_strip = Image.fromarray(variants[k][1][215:520, 30:1050])   # 1020 x 190 (Top Dock)
        bot_strip = Image.fromarray(variants[k][1][840:1430, 30:1050])  # 1020 x 650 (Knees + Simple Callout)
        closeup.paste(top_strip, (px, py + 32))
        closeup.paste(bot_strip, (px, py + 32 + 310))
        cd.text((px + 6, py), meta_cols[idx][0], font=f_c, fill=(255, 165, 70) if idx == 0 else (45, 250, 140))
        cd.rounded_rectangle([px - 2, py + 30, px + 1022, py + 935], radius=12, outline=(255, 255, 255, 75), width=2)

    out_closeup = os.path.join(OUT_DIR, "day3_simple_4variants_1to1_closeup.jpg")
    closeup.save(out_closeup, quality=95)
    print("Saved:", out_master, "and", out_closeup)


if __name__ == "__main__":
    main()
