#!/usr/bin/env python3
"""
v38 Broadcast Luxury Master — Zero Amateur Icons + Clear Exercise Header + High-Contrast 1-Line Hero HUD
========================================================================================================
Implements all of the user's feedback (`t170u`, `t171u`, `t172u`, `t173u`):
1. "체크 표시같은 아마추어같은 아이콘 같은건 사용하지마" (`t173u`):
   - ZERO checkmark (`✓`) or cross (`✕`) symbols anywhere in the video frames, top dock, pills, or sheet headers!
   - Removes the top-right `✕ / ✓` circle badge completely.
   - Uses pure broadcast-grade Crimson (`#FF453A`) vs Emerald (`#2EE476`) color coding and clean typography.
2. "또한 지금이 무슨 운동동작인지를 알려줘야하고" (`t172u`):
   - Adds a crystal-clear, architectural **Top Exercise Title Header** (`WIDE-GRIP LAT PULLDOWN` / `CABLE ROPE PUSHDOWN`
     with target muscle badge `BACK · LATS` / `TRICEPS`) directly above the 3-thumbnail dock (`y=168..216`),
     so viewers immediately know what exercise movement is being performed from frame 0 to frame 600.
3. "2단 타이포 글자가 잘 안들어와" + "글자가 있어야할건 있어야지" (`t170u`, `t171u`):
   - Eliminates the cramped, hard-to-read 2-tier stacked text to the right of `84%`!
   - Next to `84%` and the vertical divider, there is ONLY **ONE single giant `46px` ExtraBold white line**
     (`84%  |  42° BACK SWING`, `98%  |  ELBOWS TO HIPS`) vertically centered on the exact same baseline!
   - All essential anatomical target, injury risk, and set/rep information (`LUMBAR OVERLOAD`, `BICEPS & TRAP STEAL`,
     `UPPER LAT & TERES`, `FULL LAT ISOLATION · 4 × 12 REPS`) is preserved in a **full-width High-Contrast Solid Accent Capsule**
     (`20px` ExtraBold white text on a solid Crimson/Emerald pill) placed cleanly ABOVE the `84%` hero row!
"""

import os
import sys
import math
import asyncio
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import build_v35_repeatable_master as v35
import build_v36_clean_master as v36
import build_v36_publish_reels as v36p
import build_v37_minimal_luxury as v37

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_AUDIO = "/workspace/daily_info_reels/_tmp_v38_audio"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_AUDIO, exist_ok=True)

W, H = eng.W, eng.H

# Speed up 2x supersampled RGBA box-downsample (2160x3840 -> 1080x1920)
_orig_resize = Image.Image.resize
def _fast_2x_resize(self, size, resample=None, box=None, reducing_gap=None):
    if self.size == (2160, 3840) and size == (1080, 1920):
        return _orig_resize(self, size, resample=Image.Resampling.BILINEAR)
    return _orig_resize(self, size, resample=resample, box=box, reducing_gap=reducing_gap)
Image.Image.resize = _fast_2x_resize

FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)  # 600 frames

F_EX_TITLE = v30.f_pretendard(26, "extrabold")
F_EX_SUB = v30.f_pretendard(17, "extrabold")

F_TAG_PILL = v30.f_pretendard(20, "extrabold")
F_TAG_CTX = v30.f_pretendard(20, "bold")
F_HERO_PCT = v30.f_barlow(90, extrabold=True)
F_HERO_SYM = v30.f_barlow(48, extrabold=True)
F_SINGLE_LINE = v30.f_pretendard(46, "extrabold")


def draw_v38_icon_free_top_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    exercise_name: str,
    muscle_group: str,
):
    """
    v38 Broadcast-Grade Top HUD (`y=164..376`):
    1. Top Exercise Header Bar (`x=84, y=164`):
       - Frosted obsidian capsule containing a glowing 10px status LED dot (`Crimson` or `Emerald`),
         bold `26px` Pure White Exercise Name (`WIDE-GRIP LAT PULLDOWN` / `CABLE ROPE PUSHDOWN`),
         and a crisp accent pill (`BACK · LATS` / `TRICEPS`).
    2. 3-Thumbnail Progression Dock (`x=84, y=232`):
       - Clean 3-stage visual progression dock with ZERO amateur checkmark (`✓`) or cross (`✕`) circle badges!
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = v30.COL_RED if is_wrong else v30.COL_GREEN
    pill_bg = (218, 44, 38, 245) if is_wrong else (18, 168, 88, 245)

    bar_x = 84
    # --- 1. Top Exercise Title Header (`y=164..216`) ---
    hdr_y = 164
    hdr_h = 50
    tb_ex = od.textbbox((0, 0), exercise_name, font=F_EX_TITLE)
    ex_w = tb_ex[2] - tb_ex[0]
    tb_mg = od.textbbox((0, 0), muscle_group, font=F_EX_SUB)
    mg_w = tb_mg[2] - tb_mg[0]

    pill_w = mg_w + 26
    hdr_w = 24 + 12 + ex_w + 18 + pill_w + 18

    od.rounded_rectangle([bar_x + 2, hdr_y + 4, bar_x + hdr_w + 2, hdr_y + hdr_h + 4], radius=16, fill=(0, 0, 0, 145))
    od.rounded_rectangle(
        [bar_x, hdr_y, bar_x + hdr_w, hdr_y + hdr_h],
        radius=16,
        fill=(14, 17, 23, 228),
        outline=(255, 255, 255, 42),
        width=1,
    )

    # Glowing status LED dot (no amateur checkmark/cross icon!)
    dot_cx = bar_x + 22
    dot_cy = hdr_y + hdr_h // 2
    od.ellipse([dot_cx - 8, dot_cy - 8, dot_cx + 8, dot_cy + 8], fill=(*accent, 95))
    od.ellipse([dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5], fill=(*accent, 255))

    ex_x = dot_cx + 16
    od.text((ex_x, hdr_y + 10), exercise_name, font=F_EX_TITLE, fill=(255, 255, 255, 255))

    p_x0 = ex_x + ex_w + 16
    p_y0 = hdr_y + 9
    p_h = hdr_h - 18
    od.rounded_rectangle([p_x0, p_y0, p_x0 + pill_w, p_y0 + p_h], radius=9, fill=pill_bg)
    od.text((p_x0 + 13, p_y0 + 5), muscle_group, font=F_EX_SUB, fill=(255, 255, 255, 255))

    # --- 2. 3-Stage Thumbnail Dock (`y=228..366`) ---
    tw, th = 142, 114
    gap = 12
    pad = 12
    dock_y = 228
    dock_w = 3 * tw + 2 * gap + 2 * pad
    dock_h = th + 2 * pad

    od.rounded_rectangle([bar_x + 2, dock_y + 6, bar_x + dock_w + 2, dock_y + dock_h + 6], radius=24, fill=(0, 0, 0, 130))
    od.rounded_rectangle(
        [bar_x, dock_y, bar_x + dock_w, dock_y + dock_h],
        radius=24,
        fill=(16, 19, 25, 218),
        outline=(255, 255, 255, 38),
        width=1,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = dock_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy().resize((tw, th), Image.Resampling.LANCZOS) if i == active_idx else inact_im.copy().resize((tw, th), Image.Resampling.LANCZOS)

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_idx:
            i_col = v30.COL_RED if i < 2 else v30.COL_GREEN
            od.rounded_rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], radius=15, outline=(*i_col, 245), width=2)
            od.rounded_rectangle([tx + tw // 2 - 18, ty + th - 6, tx + tw // 2 + 18, ty + th - 2], radius=2, fill=(*i_col, 255))
        else:
            od.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=14, outline=(255, 255, 255, 22), width=1)

    canvas.alpha_composite(ov)


def draw_v38_complete_bottom_hud(canvas: Image.Image, sp: dict):
    """
    v38 Complete & Crystal-Clear Luxury Bottom-Left HUD:
    - ZERO amateur checkmark (`✓`) or cross (`✕`) icons!
    - ZERO cramped 2-tier stacked text to the right of `84%`!
    - Structure:
      1. Top Header Strip (`y=1322`): Solid High-Contrast Accent Capsule (`[ LUMBAR OVERLOAD ]` / `[ FULL LAT ISOLATION ]`
         in `20px` ExtraBold Pure White on Crimson/Emerald fill) + `20px` Bold Silver-White Context Label (`LAT PULLDOWN` / `4 × 12 REPS · 1s HOLD`).
      2. Main Hero 1-Line Row (`y=1368`): Sculpted `90px` Hero Number + `48px` `%` + `3.5px` Vertical Accent Bar +
         SINGLE GIANT `46px` ExtraBold White Cue (`42° BACK SWING` / `ELBOWS TO HIPS`) on the exact same horizontal line!
      3. Bottom Sleek `6px` Continuous Progress Bar (`y=1470`) with glowing leading-edge jewel dot.
    """
    is_wrong = sp["is_wrong"]
    accent = v30.COL_RED if is_wrong else v30.COL_GREEN
    pill_fill = (224, 44, 38, 250) if is_wrong else (16, 168, 86, 250)
    pct = int(np.clip(sp["pct"], 0, 100))

    tag_pill_str = sp["tag_pill"]
    tag_ctx_str = sp["tag_ctx"]
    cue_1line = sp["cue_1line"]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx = 68

    # --- 1. Top Full-Width High-Contrast Solid Accent Capsule + Context Label (`y=1322`) ---
    py = 1322
    tb_p = od.textbbox((0, 0), tag_pill_str, font=F_TAG_PILL)
    pw = (tb_p[2] - tb_p[0]) + 28
    ph = 36

    sd.rounded_rectangle([bx, py + 3, bx + pw, py + ph + 3], radius=9, fill=(0, 0, 0, 225))
    od.rounded_rectangle([bx, py, bx + pw, py + ph], radius=9, fill=pill_fill, outline=(255, 255, 255, 55), width=1)
    od.text((bx + 14, py + 6), tag_pill_str, font=F_TAG_PILL, fill=(255, 255, 255, 255))

    ctx_x = bx + pw + 14
    ctx_y = py + 6
    sd.text((ctx_x, ctx_y + 3), tag_ctx_str, font=F_TAG_CTX, fill=(0, 0, 0, 245))
    od.text((ctx_x, ctx_y), tag_ctx_str, font=F_TAG_CTX, fill=(228, 236, 248, 255))

    # --- 2. Main Hero 1-Line Row (`y=1368`) ---
    by = 1368
    num_str = f"{pct}"
    nb = od.textbbox((0, 0), num_str, font=F_HERO_PCT)
    nw = nb[2] - nb[0]

    sd.text((bx, by + 4), num_str, font=F_HERO_PCT, fill=(0, 0, 0, 245))
    od.text((bx, by), num_str, font=F_HERO_PCT, fill=(255, 255, 255, 255))

    sym_x = bx + nw + 3
    sym_y = by + 13
    sd.text((sym_x, sym_y + 3), "%", font=F_HERO_SYM, fill=(0, 0, 0, 240))
    od.text((sym_x, sym_y), "%", font=F_HERO_SYM, fill=(*accent, 255))
    sb = od.textbbox((0, 0), "%", font=F_HERO_SYM)
    sw = sb[2] - sb[0]

    div_x = sym_x + sw + 20
    div_y0 = by + 19
    div_y1 = by + 73
    sd.rounded_rectangle([div_x, div_y0 + 2, div_x + 4, div_y1 + 2], radius=2, fill=(0, 0, 0, 225))
    od.rounded_rectangle([div_x, div_y0, div_x + 4, div_y1], radius=2, fill=(*accent, 255))

    tx = div_x + 20
    ty = by + 20
    tb_c = od.textbbox((0, 0), cue_1line, font=F_SINGLE_LINE)
    cw = tb_c[2] - tb_c[0]

    sd.text((tx, ty + 4), cue_1line, font=F_SINGLE_LINE, fill=(0, 0, 0, 245))
    od.text((tx, ty), cue_1line, font=F_SINGLE_LINE, fill=(255, 255, 255, 255))

    # --- 3. Bottom Sleek Continuous Progress Hairline (`y=1466`) ---
    bar_x0 = bx + 2
    bar_y0 = by + 96
    total_row_w = (tx + cw) - bar_x0
    bar_w = max(380, min(540, total_row_w))
    bar_h = 6
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


def build_cached_v38_top_docks(torso_icons, exercise_name: str, muscle_group: str) -> dict:
    cache = {}
    for active_slot, is_wrong in [(0, True), (1, True), (2, False)]:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_v38_icon_free_top_dock(layer, torso_icons, active_slot, is_wrong, exercise_name, muscle_group)
        cache[(active_slot, is_wrong)] = layer
    return cache


def apply_bottom_left_scrim(rgb_arr: np.ndarray, curve_l: np.ndarray, scrim_2d: np.ndarray) -> np.ndarray:
    out = rgb_arr.astype(np.float32)
    left_w = curve_l.shape[1]
    out[480:1780, :left_w] *= curve_l
    out[1240:1560, :680] *= scrim_2d[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_bottom_scrim_2d() -> np.ndarray:
    """Smooth dark vignette scrim under the bottom-left HUD (`y=1240..1560, x=0..680`) for 10:1 text contrast."""
    yy = np.linspace(0.0, 1.0, 320, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, 680, dtype=np.float32)[None, :]
    vy = np.clip(np.sin(yy * math.pi), 0.0, 1.0) ** 0.85
    vx = np.clip((1.0 - xx) * 1.35, 0.0, 1.0) ** 0.9
    atten = 1.0 - 0.42 * (vy * vx)
    return atten


def build_day2_v38_master_reel():
    print("Building Day 2 (Wide-Grip Lat Pulldown) v38 Broadcast Luxury Master Reel...")
    cap_w = cv2.VideoCapture(v36p.DAY2_WRONG_I2V)
    raw_w = []
    while True:
        ret, fr = cap_w.read()
        if not ret:
            break
        raw_w.append(fr)
    cap_w.release()

    cap_c = cv2.VideoCapture(v36p.DAY2_CORRECT_I2V)
    raw_c = []
    while True:
        ret, fr = cap_c.read()
        if not ret:
            break
        raw_c.append(fr)
    cap_c.release()

    raw_w_stroke = raw_w[22:56]
    pool_s1, pool_s2, raw_el_w = [], [], []
    for fr in raw_w_stroke:
        rgb1, el1 = v36p.render_day2_i2v_frame_v36(fr, 1, True)
        rgb2, _   = v36p.render_day2_i2v_frame_v36(fr, 2, True)
        pool_s1.append(rgb1)
        pool_s2.append(rgb2)
        raw_el_w.append(el1)

    el_w_arr = np.array(raw_el_w, dtype=np.float32)
    el_w_x = gaussian_filter1d(el_w_arr[:, 0], sigma=2.0, mode="nearest")
    el_w_y = gaussian_filter1d(el_w_arr[:, 1], sigma=2.0, mode="nearest")
    smooth_el_w = [(float(el_w_x[i]), float(el_w_y[i])) for i in range(len(raw_w_stroke))]

    raw_c_stroke = raw_c[40:121][::-1]
    pool_s3, pool_s4, raw_el_c = [], [], []
    for fr in raw_c_stroke:
        rgb3, el3 = v36p.render_day2_i2v_frame_v36(fr, 3, False)
        rgb4, _   = v36p.render_day2_i2v_frame_v36(fr, 4, False)
        pool_s3.append(rgb3)
        pool_s4.append(rgb4)
        raw_el_c.append(el3)

    el_c_arr = np.array(raw_el_c, dtype=np.float32)
    el_c_x = gaussian_filter1d(el_c_arr[:, 0], sigma=2.2, mode="nearest")
    el_c_y = gaussian_filter1d(el_c_arr[:, 1], sigma=2.2, mode="nearest")
    smooth_el_c = [(float(el_c_x[i]), float(el_c_y[i])) for i in range(len(raw_c_stroke))]

    size = (142, 114)
    crop_box = (460, 620, 880, 1080)
    inact_base = Image.fromarray(pool_s3[55]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 135)))

    im0 = Image.fromarray(pool_s1[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im1 = Image.fromarray(pool_s2[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im2 = Image.fromarray(pool_s4[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]
    dock_cache = build_cached_v38_top_docks(torso_icons, "WIDE-GRIP LAT PULLDOWN", "BACK · LATS")
    hud_cache = {}

    wav_day2 = os.path.join(TMP_AUDIO, "day2_v38_audio.wav")
    vo_day2 = [
        (0.25, "Wide grip lat pulldown. Stop swinging your lower back."),
        (5.93, "Shrugging your shoulders lets your biceps steal the load."),
        (10.40, "Set a fifteen degree lean and depress your scapula first."),
        (15.40, "Drive your elbows to your hips. Four sets of twelve."),
    ]
    asyncio.run(v36p.synthesize_voiceover_wav(wav_day2, vo_day2, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day2_v38_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day2_v38_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]
    scrim_2d = make_bottom_scrim_2d()

    step_static_frames = {}
    cover_pil = None

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            u = t / 5.0
            swing_w = v37.back_swing_momentum_wave(u)
            pos_f = swing_w * (len(pool_s1) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s1, pos_f)
            base_rgb = v37.amplify_back_swing_posture(base_rgb, swing_w)
            el_pt = v37.sample_pt_interpolated(smooth_el_w, pos_f)
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom = 1.02 + 0.025 * swing_w
            cam_focus, cam_dest = (650.0, 940.0), (665.0, 940.0)
            pct = int(round(74 + 10 * swing_w))
            tag_pill = "LUMBAR OVERLOAD"
            tag_ctx = "MOMENTUM SWING"
            cue_1line = "42° BACK SWING"
        elif t < 10.0:
            u = (t - 5.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s2) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s2, pos_f)
            el_pt = v37.sample_pt_interpolated(smooth_el_w, pos_f)
            step_idx = 2
            is_wrong = True
            active_slot = 1
            swing_w = rep_w
            cam_zoom = 1.04
            cam_focus, cam_dest = (635.0, 900.0), (655.0, 900.0)
            pct = int(round(66 + 8 * rep_w))
            tag_pill = "BICEPS & TRAP STEAL"
            tag_ctx = "ARM PULL ERROR"
            cue_1line = "+22° TRAP SHRUG"
        elif t < 15.0:
            u = (t - 10.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s3) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s3, pos_f)
            el_pt = v37.sample_pt_interpolated(smooth_el_c, pos_f)
            step_idx = 3
            is_wrong = False
            active_slot = 2
            swing_w = rep_w
            cam_zoom = 1.03
            cam_focus, cam_dest = (645.0, 910.0), (660.0, 910.0)
            pct = int(round(88 + 8 * rep_w))
            tag_pill = "UPPER LAT & TERES"
            tag_ctx = "SCAPULA DEPRESSION"
            cue_1line = "15° SCAPULA LOCK"
        else:
            u = (t - 15.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s4) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s4, pos_f)
            el_pt = v37.sample_pt_interpolated(smooth_el_c, pos_f)
            step_idx = 4
            is_wrong = False
            active_slot = 2
            swing_w = rep_w
            cam_zoom = 1.04
            cam_focus, cam_dest = (650.0, 915.0), (665.0, 915.0)
            pct = int(round(91 + 7 * rep_w))
            tag_pill = "FULL LAT ISOLATION"
            tag_ctx = "4 × 12 REPS  ·  1s HOLD"
            cue_1line = "ELBOWS TO HIPS"

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        clean_rgb = apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)
        im = Image.fromarray(clean_rgb).convert("RGBA")

        v37.render_day2_v37_exterior_callouts(im, step_idx, map_pt, el_pt, is_wrong, swing_amount=swing_w)
        im.alpha_composite(dock_cache[(active_slot, is_wrong)])

        hud_key = (is_wrong, pct, tag_pill, tag_ctx, cue_1line)
        if hud_key not in hud_cache:
            h_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_v38_complete_bottom_hud(
                h_layer,
                {
                    "is_wrong": is_wrong,
                    "pct": pct,
                    "tag_pill": tag_pill,
                    "tag_ctx": tag_ctx,
                    "cue_1line": cue_1line,
                },
            )
            hud_cache[hud_key] = h_layer
        im.alpha_composite(hud_cache[hud_key])

        rgb_frame = im.convert("RGB")
        if f_idx in (int(1.35 * FPS), int(7.4 * FPS), int(12.4 * FPS), int(17.4 * FPS)):
            step_static_frames[step_idx] = rgb_frame.copy()
        if f_idx == int(17.4 * FPS):
            cover_pil = rgb_frame.copy()

        vw.write(cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR))

    vw.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-threads", "4",
            "-i", tmp_mp4, "-i", wav_day2,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_mp4)

    cover_path = os.path.join(OUT_DIR, "day2_v38_cover.jpg")
    cover_pil.save(cover_path, quality=95)

    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 2 · WIDE-GRIP LAT PULLDOWN (v38 BROADCAST LUXURY MASTER)  ·  체크/X 아이콘 0% · 상단 운동명 헤더 + 고대비 태그 캡슐 + 1단 초대형 타이포",
        font=f_row,
        fill=v30.COL_GREEN,
    )
    col_titles = [
        "STEP 1 · 84% | 42° BACK SWING",
        "STEP 2 · 74% | +22° TRAP SHRUG",
        "STEP 3 · 96% | 15° SCAPULA LOCK",
        "STEP 4 · 98% | ELBOWS TO HIPS",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if i < 2 else v30.COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = step_static_frames[i + 1].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day2_v38_clean_master_sheet.jpg")
    board.save(sheet_path, quality=95)
    print("Saved Day 2 v38 Master Reel & Sheet:", final_mp4, sheet_path)
    return step_static_frames


def build_day1_v38_master_reel():
    print("Building Day 1 (Cable Rope Pushdown) v38 Broadcast Luxury Master Reel...")
    cap_w = cv2.VideoCapture(eng.WRONG_I2V)
    raw_w = []
    while True:
        ret, fr = cap_w.read()
        if not ret:
            break
        raw_w.append(fr)
    cap_w.release()
    raw_w_stroke = raw_w[:56]

    cap_c = cv2.VideoCapture(eng.CORRECT_I2V)
    raw_c = []
    while True:
        ret, fr = cap_c.read()
        if not ret:
            break
        raw_c.append(fr)
    cap_c.release()
    raw_c_stroke = raw_c[:65]

    pool_s1_rgb = []
    for fr in raw_w_stroke:
        rgb_out, _ = v35.render_v35_clean_3d_frame(fr, 1, "xray", True)
        pool_s1_rgb.append(rgb_out)

    keys_order = ["ear", "sh", "sh_far", "el", "wr", "hd", "hip", "hip_far", "knee", "ank"]
    pool_s2_rgb, pool_s3_rgb, pool_s4_rgb, raw_c_kpts = [], [], [], []
    for fr in raw_c_stroke:
        rgb2, kp = v35.render_v35_clean_3d_frame(fr, 2, "anterior_deltoid", True)
        rgb3, _  = v35.render_v35_clean_3d_frame(fr, 3, "triceps_lat_long", False)
        rgb4, _  = v35.render_v35_clean_3d_frame(fr, 4, "triceps_all_three", False)
        pool_s2_rgb.append(rgb2)
        pool_s3_rgb.append(rgb3)
        pool_s4_rgb.append(rgb4)
        raw_c_kpts.append(kp)

    n_c = len(raw_c_stroke)
    smooth_c_kpts = [dict() for _ in range(n_c)]
    for k in keys_order:
        arr = np.array([rk[k] for rk in raw_c_kpts], dtype=np.float32)
        ax = gaussian_filter1d(arr[:, 0], sigma=1.8, mode="nearest")
        ay = gaussian_filter1d(arr[:, 1], sigma=1.8, mode="nearest")
        for i in range(n_c):
            smooth_c_kpts[i][k] = (float(ax[i]), float(ay[i]))

    size = (142, 114)
    crop_box = (555, 550, 905, 950)
    f_c42_mono = eng.prepare_muscle_frame(raw_c_stroke[42])["mono"]
    inact_base = Image.fromarray(f_c42_mono).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 125)))

    im0 = Image.fromarray(pool_s1_rgb[45]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im1 = Image.fromarray(pool_s2_rgb[14]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im2 = Image.fromarray(pool_s4_rgb[60]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]
    dock_cache = build_cached_v38_top_docks(torso_icons, "CABLE ROPE PUSHDOWN", "TRICEPS")
    hud_cache = {}

    wav_day1 = os.path.join(TMP_AUDIO, "day1_v38_audio.wav")
    vo_day1 = [
        (0.25, "Cable rope pushdown. Never hunch your spine."),
        (5.25, "Stopping at ninety degrees lets your front delt take over."),
        (10.35, "Hinge fifteen degrees and pin your elbows tight."),
        (15.35, "Spread the rope at full lockout. Four sets of twelve."),
    ]
    asyncio.run(v36p.synthesize_voiceover_wav(wav_day1, vo_day1, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day1_v38_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day1_v38_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    kp_s1_fixed = smooth_c_kpts[14]
    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]
    scrim_2d = make_bottom_scrim_2d()

    step_static_frames = {}
    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            u = t / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s1_rgb) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s1_rgb, pos_f)
            kp = kp_s1_fixed
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom, cam_focus, cam_dest = 1.02, (690.0, 920.0), (690.0, 920.0)
            pct = int(round(80 + 6 * rep_w))
            tag_pill = "SHOULDER & WRIST SHEAR"
            tag_ctx = "SPINE KYPHOSIS"
            cue_1line = "35° BACK HUNCH"
        elif t < 10.0:
            u = (t - 5.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = 4.0 + rep_w * 22.0
            base_rgb = v37.sample_pool_interpolated(pool_s2_rgb, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_c_kpts, pos_f)
            step_idx = 2
            is_wrong = True
            active_slot = 1
            cam_zoom, cam_focus, cam_dest = 1.05, (685.0, 910.0), (695.0, 910.0)
            pct = int(round(66 + 6 * rep_w))
            tag_pill = "FRONT DELT STEAL"
            tag_ctx = "INCOMPLETE ROM"
            cue_1line = "90° HALF REP"
        elif t < 15.0:
            u = (t - 10.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s3_rgb) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s3_rgb, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_c_kpts, pos_f)
            step_idx = 3
            is_wrong = False
            active_slot = 2
            cam_zoom, cam_focus, cam_dest = 1.04, (685.0, 920.0), (695.0, 920.0)
            pct = int(round(89 + 6 * rep_w))
            tag_pill = "LATERAL & LONG HEAD"
            tag_ctx = "ELBOW HINGE LOCK"
            cue_1line = "15° HINGE · PIN"
        else:
            u = (t - 15.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s4_rgb) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s4_rgb, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_c_kpts, pos_f)
            step_idx = 4
            is_wrong = False
            active_slot = 2
            cam_zoom, cam_focus, cam_dest = 1.06, (690.0, 925.0), (700.0, 920.0)
            pct = int(round(93 + 6 * rep_w))
            tag_pill = "ALL 3 TRICEPS HEADS"
            tag_ctx = "4 × 12 REPS  ·  1s HOLD"
            cue_1line = "180° LOCKOUT"

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        clean_rgb = apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)
        im = Image.fromarray(clean_rgb).convert("RGBA")

        v36.render_v36_clean_exterior_overlay(im, step_idx, map_pt, kp, is_wrong, dynamic_elbow_deg=None)
        im.alpha_composite(dock_cache[(active_slot, is_wrong)])

        hud_key = (is_wrong, pct, tag_pill, tag_ctx, cue_1line)
        if hud_key not in hud_cache:
            h_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_v38_complete_bottom_hud(
                h_layer,
                {
                    "is_wrong": is_wrong,
                    "pct": pct,
                    "tag_pill": tag_pill,
                    "tag_ctx": tag_ctx,
                    "cue_1line": cue_1line,
                },
            )
            hud_cache[hud_key] = h_layer
        im.alpha_composite(hud_cache[hud_key])

        rgb_frame = im.convert("RGB")
        if f_idx in (int(2.4 * FPS), int(7.4 * FPS), int(12.4 * FPS), int(17.4 * FPS)):
            step_static_frames[step_idx] = rgb_frame.copy()
        vw.write(cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR))

    vw.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-threads", "4",
            "-i", tmp_mp4, "-i", wav_day1,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_mp4)

    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 1 · CABLE ROPE PUSHDOWN (v38 BROADCAST LUXURY MASTER)  ·  체크/X 아이콘 0% · 상단 운동명 헤더 + 고대비 태그 캡슐 + 1단 초대형 타이포",
        font=f_row,
        fill=v30.COL_GREEN,
    )
    col_titles = [
        "STEP 1 · 86% | 35° BACK HUNCH",
        "STEP 2 · 72% | 90° HALF REP",
        "STEP 3 · 95% | 15° HINGE · PIN",
        "STEP 4 · 99% | 180° LOCKOUT",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if i < 2 else v30.COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = step_static_frames[i + 1].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day1_v38_clean_master_sheet.jpg")
    board.save(sheet_path, quality=95)
    print("Saved Day 1 v38 Master Reel & Sheet:", final_mp4, sheet_path)
    return step_static_frames


def build_v38_complete_hud_comparison(d2_v38_frames: dict):
    """
    Builds `/workspace/daily_info_reels/output/v38_complete_hud_comparison.jpg` comparing:
    - Top Row (`v37`): Amateur `✕/✓` circle badge + No exercise name + Cramped 2-tier text right of `%` with faint 15px top line.
    - Bottom Row (`v38`): Zero `✕/✓` icons + Clear Top Exercise Title Header (`WIDE-GRIP LAT PULLDOWN · BACK · LATS`) +
      Full-Width Solid Accent Tag Capsule (`LUMBAR OVERLOAD · MOMENTUM SWING`) + Single-Line `84% | 42° BACK SWING` Hero Row!
    """
    v37_comp = Image.open(os.path.join(OUT_DIR, "v37_bottom_hud_comparison.jpg"))
    v37_s1_bot = v37_comp.crop((24, 512, 24 + 560, 512 + 390))
    v37_s4_bot = v37_comp.crop((24 + 560 + 24, 512, 24 + 560 + 24 + 560, 512 + 390))

    v38_s1_top = d2_v38_frames[1].crop((40, 140, 820, 410))
    v38_s4_top = d2_v38_frames[4].crop((40, 140, 820, 410))
    v38_s1_bot = d2_v38_frames[1].crop((30, 1275, 670, 1545))
    v38_s4_bot = d2_v38_frames[4].crop((30, 1275, 670, 1545))

    cw = 580
    pad = 24
    comp = Image.new("RGB", (cw * 2 + pad * 3, 1120), (10, 12, 16))
    cd = ImageDraw.Draw(comp)
    f_title = v30.f_pretendard(21, "extrabold")
    f_sub = v30.f_pretendard(15, "bold")
    cd.text(
        (24, 18),
        "v37 vs v38 BROADCAST LUXURY HUD  ·  체크/X 아이콘 제거 + 상단 운동명 헤더 + 고대비 솔리드 캡슐 & 1단 초대형 타이포",
        font=f_title,
        fill=v30.COL_GREEN,
    )

    # Row 1: v38 New Top Exercise Title Header + Icon-Free 3-Thumbnail Dock
    r1_y = 80
    r1_h = 220
    for c_i, (im_t, lbl, col) in enumerate([
        (v38_s1_top, "1A. v38 상단 헤더 (운동명 WIDE-GRIP LAT PULLDOWN 명시 + 우측 X 아이콘 제거)", v30.COL_RED),
        (v38_s4_top, "1B. v38 상단 헤더 (운동명 + BACK · LATS 타겟 캡슐 + 우측 체크 아이콘 제거)", v30.COL_GREEN),
    ]):
        x = pad + c_i * (cw + pad)
        cd.text((x + 4, r1_y - 24), lbl, font=f_sub, fill=col)
        rc = im_t.resize((cw, r1_h), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, r1_h - 1], outline=col, width=2)
        comp.paste(rc, (x, r1_y))

    # Row 2: v37 Old Cramped 2-Tier Bottom HUD
    r2_y = 350
    r2_h = 330
    for c_i, (im_b, lbl) in enumerate([
        (v37_s1_bot, "2A. v37 기존 하단 (% 우측 2단 타이포: 상단 15px 회색 글씨가 작고 흐려 안 읽힘)"),
        (v37_s4_bot, "2B. v37 기존 하단 (% 우측 2단 타이포: 상·하단 시선 분산 및 운동명 부재)"),
    ]):
        x = pad + c_i * (cw + pad)
        cd.text((x + 4, r2_y - 24), lbl, font=f_sub, fill=(175, 185, 200))
        rc = im_b.resize((cw, r2_h), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, r2_h - 1], outline=(120, 130, 145), width=2)
        comp.paste(rc, (x, r2_y))

    # Row 3: v38 New Complete & Crystal-Clear Bottom HUD
    r3_y = 730
    r3_h = 350
    for c_i, (im_b, lbl, col) in enumerate([
        (v38_s1_bot, "3A. v38 NEW 하단 (상단 고대비 솔리드 캡슐 + % 우측 1단 초대형 46px 타이포!)", v30.COL_RED),
        (v38_s4_bot, "3B. v38 NEW 하단 (FULL LAT ISOLATION · 4×12 REPS + 98% | ELBOWS TO HIPS)", v30.COL_GREEN),
    ]):
        x = pad + c_i * (cw + pad)
        cd.text((x + 4, r3_y - 24), lbl, font=f_sub, fill=col)
        rc = im_b.resize((cw, r3_h), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, r3_h - 1], outline=col, width=2)
        comp.paste(rc, (x, r3_y))

    for fn in ["v38_complete_hud_comparison.jpg", "v38_1line_hud_comparison.jpg"]:
        p = os.path.join(OUT_DIR, fn)
        comp.save(p, quality=95)
        print("Saved:", p)


if __name__ == "__main__":
    d2_v38 = build_day2_v38_master_reel()
    build_day1_v38_master_reel()
    build_v38_complete_hud_comparison(d2_v38)
