#!/usr/bin/env python3
"""
v37 Minimal Luxury Master — Ultra-Clean Bottom HUD + Vivid Step 1 Back-Swing Momentum + Controlled 1x Full-ROM Reps
====================================================================================================================
Implements the user's feedback (`t166u`, `t167u`, `t168u`):
1. "그리고 하단에 있는 내용들 조금 더 간단하게 그리고 멋스럽게 표현해줄래?" (`v37 Minimal Luxury Telemetry`):
   - Removes the cluttered 2-to-3 rows of 20-segment equalizer bricks (`LOW BACK`, `LATS`, `TERES MAJ`, `LOW TRAP`)
     and removes the extra bottom pill.
   - Replaces them with a breathtaking, architectural 3-element Minimal Luxury HUD:
     (a) Sculpted Hero Percentage (`96px` Barlow Condensed ExtraBold) on the left with deep studio shadow.
     (b) Vertical `2.5px` luminous accent hairline divider (`Crimson` for `✕`, `Emerald` for `✓`) +
         tracked-out uppercase micro-kicker (`15px`, e.g. `LUMBAR OVERLOAD · ✕ MISTAKE` / `FULL LAT ISOLATION · 4 × 12`) +
         bold sculptural cue title (`36px`, e.g. `42° BACK SWING` / `ELBOWS TO HIPS`).
     (c) Single ultra-sleek continuous `4.5px` hairline progress track (`width=336px`) with a glowing leading-edge jewel dot.
2. "처음꺼 일부로 빠르게 동작하게 한거야?" + "배치기 반동 표현 잘 보여주고":
   - Fixes the unnatural 3x ping-pong (`u * 1.5`) across a double-rep clip!
   - In Day 2 Step 1 (`0.0s–5.0s`, `✕ 42° BACK SWING`), isolates the single clean upright-to-back-swing stroke (`f=22 -> f=55`)
     and applies a realistic **Whip-and-Hold Back-Swing Momentum Curve (`배치기 반동`)**:
     * Starts upright with arms extended overhead (`f=22`, `18°` setup)
     * Whips the entire torso backward into the `42° BACK SWING` (`f=22 -> f=55`), with a subtle rotational torso-whip emphasis
       and dynamic exterior swing angle counter (`18° -> 42° BACK SWING`) + pulsing crimson lumbar X-Ray flare!
     * Holds the `42°` leaned-back lumbar overload briefly so the viewer clearly sees the back-swing cheat, then recovers smoothly
       and performs a second clear back-swing rep!
   - In Day 2 Steps 3 & 4 (`✓ Correct`), maps a **single, heavy, controlled 1.0x repetition per 5.0s step** (`f=120` overhead stretch
     -> `f=38` chest pull lockout -> `f=120` controlled eccentric return) with sub-frame blending (`cv2.addWeighted`) so the movement
     is 100% smooth and controlled!
   - Updates Day 1 (`day1_v37_master_reel.mp4` & `day1_v37_clean_master_sheet.jpg`) to the exact same `v37 Minimal Luxury Bottom HUD`
     and smooth 1.0x controlled repetition tempo.
"""

import os
import sys
import math
import wave
import asyncio
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v25_fast_preview as v25
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import universal_anatomy_rig as uar
import build_v35_repeatable_master as v35
import build_v36_clean_master as v36
import build_v36_publish_reels as v36p

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_AUDIO = "/workspace/daily_info_reels/_tmp_v37_audio"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_AUDIO, exist_ok=True)

W, H = eng.W, eng.H
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)  # 600 frames

F_HERO_PCT = v30.f_barlow(92, extrabold=True)
F_HERO_SYM = v30.f_barlow(48, extrabold=True)
F_KICKER = v30.f_pretendard(15, "bold")
F_MAIN_TITLE = v30.f_pretendard(36, "extrabold")


def draw_tracked_kicker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill: tuple,
    tracking: int = 3,
):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        bb = draw.textbbox((0, 0), ch, font=font)
        x += (bb[2] - bb[0]) + tracking


def draw_v37_minimal_luxury_bottom_hud(canvas: Image.Image, sp: dict):
    """
    v37 Ultra-Minimal, High-Fashion Luxury Bottom-Left HUD:
    - ZERO multi-row 20-segment equalizer bricks!
    - ZERO extra bottom pill!
    - Only 3 crisp, breathable architectural elements in the open bottom-left studio space (`x=68..430, y=1385..1510`):
      1. Sculpted Hero Metric (`92px` number + `48px` `%` superscript)
      2. Vertical `3px` luminous accent bar + tracked uppercase micro-kicker (`15px`) + bold cue title (`36px`)
      3. Single sleek `5px` continuous progress hairline (`width=346px`) with a glowing leading-edge jewel dot.
    """
    is_wrong = sp["is_wrong"]
    accent = v30.COL_RED if is_wrong else v30.COL_GREEN
    accent_soft = v30.COL_RED_SOFT if is_wrong else v30.COL_GREEN_SOFT
    pct = int(np.clip(sp["pct"], 0, 100))

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx, by = 68, 1380

    # 1. Sculpted Hero Number (`92px`) + Superscript `%` (`48px`)
    num_str = f"{pct}"
    nb = od.textbbox((0, 0), num_str, font=F_HERO_PCT)
    nw = nb[2] - nb[0]

    sd.text((bx, by + 4), num_str, font=F_HERO_PCT, fill=(0, 0, 0, 245))
    od.text((bx, by), num_str, font=F_HERO_PCT, fill=(255, 255, 255, 255))

    sym_x = bx + nw + 3
    sym_y = by + 12
    sd.text((sym_x, sym_y + 3), "%", font=F_HERO_SYM, fill=(0, 0, 0, 240))
    od.text((sym_x, sym_y), "%", font=F_HERO_SYM, fill=(*accent, 255))
    sb = od.textbbox((0, 0), "%", font=F_HERO_SYM)
    sw = sb[2] - sb[0]

    # 2. Luminous Vertical Accent Hairline (`3px` wide)
    div_x = sym_x + sw + 18
    div_y0 = by + 16
    div_y1 = by + 84
    sd.rounded_rectangle([div_x, div_y0 + 2, div_x + 3, div_y1 + 2], radius=1, fill=(0, 0, 0, 220))
    od.rounded_rectangle([div_x, div_y0, div_x + 3, div_y1], radius=1, fill=(*accent, 255))

    # 3. Tracked Micro-Kicker (`15px`) + Bold Cue Title (`36px`)
    tx = div_x + 16
    kicker_str = sp.get("kicker_v37", sp.get("dial_tag", "BIOMECHANICS"))
    cue_str = sp["short_cue"].replace("✕ ", "").replace("✓ ", "")

    draw_tracked_kicker(sd, (tx, by + 18), kicker_str, F_KICKER, (0, 0, 0, 240), tracking=2)
    draw_tracked_kicker(od, (tx, by + 16), kicker_str, F_KICKER, (*accent_soft, 245), tracking=2)

    sd.text((tx, by + 44), cue_str, font=F_MAIN_TITLE, fill=(0, 0, 0, 245))
    od.text((tx, by + 41), cue_str, font=F_MAIN_TITLE, fill=(255, 255, 255, 255))

    # 4. Single Ultra-Sleek Continuous Progress Hairline (`width=348px, height=5px`)
    bar_x0 = bx + 2
    bar_y0 = by + 102
    bar_w = 348
    bar_h = 5
    fill_w = max(12, int(round(bar_w * (pct / 100.0))))

    sd.rounded_rectangle([bar_x0, bar_y0 + 2, bar_x0 + bar_w, bar_y0 + bar_h + 2], radius=3, fill=(0, 0, 0, 200))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], radius=3, fill=(255, 255, 255, 38))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y0 + bar_h], radius=3, fill=(*accent, 245))

    # Leading-edge luminous jewel dot on the progress track
    dot_x = bar_x0 + fill_w
    dot_y = bar_y0 + bar_h * 0.5
    sd.ellipse([dot_x - 7, dot_y - 7, dot_x + 7, dot_y + 7], fill=(*accent, 160))
    od.ellipse([dot_x - 4.5, dot_y - 4.5, dot_x + 4.5, dot_y + 4.5], fill=(255, 255, 255, 255))

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)


def sample_pool_interpolated(pool_rgbs: list[np.ndarray], pos_float: float) -> np.ndarray:
    """Sub-frame linear interpolation between adjacent frames for 100% buttery-smooth slow/controlled motion."""
    n = len(pool_rgbs)
    pos = float(np.clip(pos_float, 0.0, n - 1.0001))
    i0 = int(math.floor(pos))
    i1 = min(n - 1, i0 + 1)
    frac = pos - i0
    if frac < 1e-3 or i0 == i1:
        return pool_rgbs[i0]
    return cv2.addWeighted(pool_rgbs[i0], 1.0 - frac, pool_rgbs[i1], frac, 0)


def sample_pt_interpolated(pts: list[tuple[float, float]], pos_float: float) -> tuple[float, float]:
    n = len(pts)
    pos = float(np.clip(pos_float, 0.0, n - 1.0001))
    i0 = int(math.floor(pos))
    i1 = min(n - 1, i0 + 1)
    frac = pos - i0
    return (
        pts[i0][0] * (1.0 - frac) + pts[i1][0] * frac,
        pts[i0][1] * (1.0 - frac) + pts[i1][1] * frac,
    )


def sample_kpts_interpolated(kpts_list: list[dict], pos_float: float) -> dict:
    n = len(kpts_list)
    pos = float(np.clip(pos_float, 0.0, n - 1.0001))
    i0 = int(math.floor(pos))
    i1 = min(n - 1, i0 + 1)
    frac = pos - i0
    out = {}
    for k in kpts_list[0].keys():
        p0 = kpts_list[i0][k]
        p1 = kpts_list[i1][k]
        out[k] = (p0[0] * (1.0 - frac) + p1[0] * frac, p0[1] * (1.0 - frac) + p1[1] * frac)
    return out


def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def controlled_single_rep_wave(u: float) -> float:
    """
    Maps `u in [0, 1]` (a 5.0-second step) to a single, heavy, textbook gym repetition:
    - `u = 0.00 .. 0.42` (2.1s): Smooth concentric pull / extension (`0.0 -> 1.0`)
    - `u = 0.42 .. 0.56` (0.7s): Peak isometric squeeze / lockout hold (`0.96 -> 1.0 -> 0.96`)
    - `u = 0.56 .. 1.00` (2.2s): Controlled eccentric return (`1.0 -> 0.0`)
    """
    if u < 0.42:
        return smoothstep01(u / 0.42)
    elif u < 0.56:
        mid = (u - 0.42) / 0.14
        return 0.96 + 0.04 * math.sin(mid * math.pi)
    else:
        return 1.0 - smoothstep01((u - 0.56) / 0.44)


def back_swing_momentum_wave(u: float) -> float:
    """
    Maps `u in [0, 1]` (Day 2 Step 1 `0.0s..5.0s`) to a vivid, realistic **Back-Swing Momentum Cheat (`배치기 반동`)**:
    Shows 2 distinct momentum back-swing reps where the lifter starts upright (`0.0`),
    whips the upper body backward fast into the `42° BACK SWING` (`1.0`), holds/strains at the bottom,
    and rebounds up!
    """
    cycle = (u * 1.65) % 1.0
    if cycle < 0.10:
        # Brief upright setup at top stretch
        return 0.05 * (cycle / 0.10)
    elif cycle < 0.36:
        # Fast explosive torso-whip back swing (`배치기 반동` whip!)
        t_whip = (cycle - 0.10) / 0.26
        return 0.05 + 0.95 * (t_whip ** 0.72)
    elif cycle < 0.58:
        # Heavy strained hold at 42° backward lean so viewer clearly sees the back-swing posture
        t_hold = (cycle - 0.36) / 0.22
        return 1.0 - 0.04 * math.sin(t_hold * math.pi)
    else:
        # Springy eccentric return upward
        t_ret = (cycle - 0.58) / 0.42
        return 0.96 * (1.0 - smoothstep01(t_ret))


def amplify_back_swing_posture(rgb_frame: np.ndarray, swing_amount: float) -> np.ndarray:
    """
    Subtly amplifies the upper-torso backward lean (`배치기 반동`) around the hip/pelvis pivot `(615, 1340)`
    as `swing_amount` goes from `0.0` (upright setup) to `1.0` (`42° BACK SWING` peak),
    making the backward torso throw unmistakably visible!
    """
    if swing_amount <= 0.02:
        return rgb_frame
    # Upper body rotates clockwise (to the right in rear-3/4 view) and shifts right/down as torso leans back
    ang_deg = -7.5 * swing_amount
    M = cv2.getRotationMatrix2D((615.0, 1280.0), ang_deg, 1.0)
    M[0, 2] += 18.0 * swing_amount
    M[1, 2] += 10.0 * swing_amount
    warped = cv2.warpAffine(rgb_frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # Blend smoothly across the waist (`y=700..1320`) so the lower body/seat stays anchored
    yy = np.linspace(0, H - 1, H, dtype=np.float32)[:, None, None]
    upper_mask = np.clip((1320.0 - yy) / 260.0, 0.0, 1.0)
    out = rgb_frame.astype(np.float32) * (1.0 - upper_mask) + warped.astype(np.float32) * upper_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def render_day2_v37_exterior_callouts(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
    el_pt: tuple[float, float],
    is_wrong: bool,
    swing_amount: float = 1.0,
):
    """
    Renders Day 2 exterior solid arcs & numbers in v37:
    - In Step 1 (`✕ 42° BACK SWING`), dynamically sweeps the exterior back-swing arc and angle counter
      from `18°` -> `42° BACK SWING` in sync with the mannequin's backward torso whip (`배치기 반동`)!
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

    col_main = v30.COL_RED if is_wrong else v30.COL_GREEN
    col_soft = v30.COL_RED_SOFT if is_wrong else v30.COL_GREEN_SOFT
    el_s = S(map_pt(el_pt[0], el_pt[1]))

    if step_idx == 1:
        # Dynamic Back-Swing Arc behind/beside the torso showing the momentum whip (`18° -> 42° BACK SWING`)
        cur_deg = int(round(18 + 24 * swing_amount))
        cur_lumbar = int(round(58 + 26 * swing_amount))

        c_arc = S(map_pt(470, 940))
        r_a = 172 * SS
        # Dynamic arc sweep that grows wider as the torso swings backward!
        end_ang = 246
        start_ang = int(round(230 - 38 * swing_amount))
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=5 * SS)
        ang_t = math.radians(start_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, start_ang - 90, 13 * SS, (*col_soft, 255))

        # Second dorsal momentum whip arc on the right side of the upper back to make the backward throw unmistakable!
        c_back = S(map_pt(745, 920))
        r_b = 165 * SS
        b_start = 285
        b_end = int(round(302 + 34 * swing_amount))
        rd.arc([c_back[0] - r_b, c_back[1] - r_b, c_back[0] + r_b, c_back[1] + r_b], start=b_start, end=b_end, fill=(*col_main, 240), width=4 * SS)
        ang_b = math.radians(b_end)
        tip_b = (c_back[0] + r_b * math.cos(ang_b), c_back[1] + r_b * math.sin(ang_b))
        v30.draw_arrow_head(rd, tip_b, b_end + 90, 12 * SS, (*col_soft, 255))

        p_b = map_pt(405, 755)
        v30.draw_floating_number_badge(od, sd, (int(p_b[0]), int(p_b[1])), f"{cur_deg}°", "BACK SWING", col_main, col_soft, align="right")

        p_l = map_pt(855, 1085)
        v30.draw_floating_number_badge(od, sd, (min(955, int(p_l[0])), int(p_l[1])), f"{cur_lumbar}%", "LUMBAR LOAD", col_main, col_soft, align="left")

    elif step_idx == 2:
        v36.draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

        c_arc = S(map_pt(315, 795))
        r_a = 128 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=205, end=268, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(268)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 268 + 90, 12 * SS, (*col_soft, 255))

        p_shrug = map_pt(425, 595)
        v30.draw_floating_number_badge(od, sd, (int(p_shrug[0]), int(p_shrug[1])), "+22°", "TRAP SHRUG", col_main, col_soft, align="right")

        p_bi = map_pt(385, 935)
        v30.draw_floating_number_badge(od, sd, (int(p_bi[0]), int(p_bi[1])), "68%", "BICEPS STEAL", col_main, col_soft, align="right")

    elif step_idx == 3:
        v36.draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

        c_arc = S(map_pt(435, 835))
        r_a = 148 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=118, end=170, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(118)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 118 - 90, 12 * SS, (*col_soft, 255))

        p_sc = map_pt(395, 745)
        v30.draw_floating_number_badge(od, sd, (int(p_sc[0]), int(p_sc[1])), "15°", "SCAPULA LOCK", col_main, col_soft, align="right")

        p_pl = map_pt(885, 760)
        v30.draw_floating_number_badge(od, sd, (min(955, int(p_pl[0])), int(p_pl[1])), "30°", "ELBOW PLANE", col_main, col_soft, align="left")

    else:
        v36.draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

        c_arc = S(map_pt(435, 865))
        r_a = 158 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=116, end=172, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(116)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 116 - 90, 13 * SS, (*col_soft, 255))

        p_dr = map_pt(405, 825)
        v30.draw_floating_number_badge(od, sd, (int(p_dr[0]), int(p_dr[1])), "100%", "ELBOW TO HIP", col_main, col_soft, align="right")

        p_arch = map_pt(885, 790)
        v30.draw_floating_number_badge(od, sd, (min(955, int(p_arch[0])), int(p_arch[1])), "15°", "THORACIC LOCK", col_main, col_soft, align="left")

    canvas.alpha_composite(rig_layer.resize((W, H), Image.Resampling.LANCZOS))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)


def build_day2_v37_master_reel():
    print("Building Day 2 (Wide-Grip Lat Pulldown) v37 Minimal Luxury Master Reel...")
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

    # Exact monotonic single-stroke slice from `gym_day2_wrong_i2v.mp4`:
    # `f=22` is upright with arms extended high overhead (`el_y=635, wr_y=450`);
    # `f=55` is the peak backward-swing pull (`el_y=1148, wr_y=913`).
    # Using ONLY `raw_w[22:56]` (34 frames) eliminates the double-rep fast oscillation completely!
    raw_w_stroke = raw_w[22:56]
    pool_s1 = []
    pool_s2 = []
    raw_el_w = []
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

    # Exact monotonic single-stroke slice from `gym_day2_correct_i2v.mp4`:
    # `f=40` is full chest lockout (`el_y=1065, wr_y=942`), `f=120` is full overhead lat stretch (`el_y=453, wr_y=213`).
    # Reversing `raw_c[40:121][::-1]` gives a pure 81-frame monotonic stroke from Overhead Stretch (idx 0) -> Chest Lockout (idx 80)!
    raw_c_stroke = raw_c[40:121][::-1]
    pool_s3 = []
    pool_s4 = []
    raw_el_c = []
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

    # Top 3-thumbnail dock icons
    size = (142, 114)
    crop_box = (460, 620, 880, 1080)
    inact_base = Image.fromarray(pool_s3[55]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 135)))

    im0 = Image.fromarray(pool_s1[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im1 = Image.fromarray(pool_s2[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im2 = Image.fromarray(pool_s4[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    wav_day2 = os.path.join(TMP_AUDIO, "day2_v37_audio.wav")
    vo_day2 = [
        (0.25, "Wide grip lat pulldown. Stop swinging your lower back."),
        (5.93, "Shrugging your shoulders lets your biceps steal the load."),
        (10.40, "Set a fifteen degree lean and depress your scapula first."),
        (15.40, "Drive your elbows to your hips. Four sets of twelve."),
    ]
    asyncio.run(v36p.synthesize_voiceover_wav(wav_day2, vo_day2, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day2_v37_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day2_v37_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]

    step_static_frames = {}
    cover_pil = None

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            # Step 1 (0.0s - 5.0s): Vivid Back-Swing Momentum (`배치기 반동`)
            u = t / 5.0
            swing_w = back_swing_momentum_wave(u)
            pos_f = swing_w * (len(pool_s1) - 1)
            base_rgb = sample_pool_interpolated(pool_s1, pos_f)
            base_rgb = amplify_back_swing_posture(base_rgb, swing_w)
            el_pt = sample_pt_interpolated(smooth_el_w, pos_f)
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom = 1.02 + 0.025 * swing_w
            cam_focus, cam_dest = (650.0, 940.0), (665.0, 940.0)
            pct = int(round(74 + 10 * swing_w))
            sp_cfg = {
                "is_wrong": True,
                "pct": pct,
                "kicker_v37": "LUMBAR OVERLOAD  ·  MOMENTUM CHEAT",
                "short_cue": "42° BACK SWING",
            }
        elif t < 10.0:
            # Step 2 (5.0s - 10.0s): 1 Smooth Controlled Wrong Rep (Trap Shrug & Biceps Steal)
            u = (t - 5.0) / 5.0
            rep_w = controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s2) - 1)
            base_rgb = sample_pool_interpolated(pool_s2, pos_f)
            el_pt = sample_pt_interpolated(smooth_el_w, pos_f)
            step_idx = 2
            is_wrong = True
            active_slot = 1
            swing_w = rep_w
            cam_zoom = 1.04
            cam_focus, cam_dest = (635.0, 900.0), (655.0, 900.0)
            pct = int(round(66 + 8 * rep_w))
            sp_cfg = {
                "is_wrong": True,
                "pct": pct,
                "kicker_v37": "BICEPS & UPPER TRAP STEAL",
                "short_cue": "+22° TRAP SHRUG",
            }
        elif t < 15.0:
            # Step 3 (10.0s - 15.0s): 1 Smooth, Heavy, Controlled Textbook Rep (Overhead Stretch -> Scapula Lock -> Chest Pull)
            u = (t - 10.0) / 5.0
            rep_w = controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s3) - 1)
            base_rgb = sample_pool_interpolated(pool_s3, pos_f)
            el_pt = sample_pt_interpolated(smooth_el_c, pos_f)
            step_idx = 3
            is_wrong = False
            active_slot = 2
            swing_w = rep_w
            cam_zoom = 1.03
            cam_focus, cam_dest = (645.0, 910.0), (660.0, 910.0)
            pct = int(round(88 + 8 * rep_w))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "kicker_v37": "UPPER LAT & TERES MAJOR",
                "short_cue": "15° LEAN · DEPRESS",
            }
        else:
            # Step 4 (15.0s - 20.0s): 1 Smooth, Heavy, Controlled Peak-Contraction Rep (Elbows to Hips · 4×12)
            u = (t - 15.0) / 5.0
            rep_w = controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s4) - 1)
            base_rgb = sample_pool_interpolated(pool_s4, pos_f)
            el_pt = sample_pt_interpolated(smooth_el_c, pos_f)
            step_idx = 4
            is_wrong = False
            active_slot = 2
            swing_w = rep_w
            cam_zoom = 1.04
            cam_focus, cam_dest = (650.0, 915.0), (665.0, 915.0)
            pct = int(round(91 + 7 * rep_w))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "kicker_v37": "FULL LAT ISOLATION  ·  4 × 12 REPS",
                "short_cue": "ELBOWS TO HIPS",
            }

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        out = cam_rgb.astype(np.float32)
        out[480:1780, :left_w] *= curve_l
        im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")

        render_day2_v37_exterior_callouts(im, step_idx, map_pt, el_pt, is_wrong, swing_amount=swing_w)
        v28.draw_luxury_top_dock(im, torso_icons, active_slot, is_wrong)
        draw_v37_minimal_luxury_bottom_hud(im, sp_cfg)

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
    # Also update day2_v36_master_reel.mp4 so both paths point to the latest master
    subprocess.run(["cp", final_mp4, os.path.join(OUT_DIR, "day2_v36_master_reel.mp4")], check=True)

    cover_path = os.path.join(OUT_DIR, "day2_v37_cover.jpg")
    cover_pil.save(cover_path, quality=95)

    # 1. Build 4-Step Full-Body 9:16 Master Sheet for Day 2 (`day2_v37_clean_master_sheet.jpg`)
    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 2 · WIDE-GRIP LAT PULLDOWN (v37 MINIMAL LUXURY MASTER)  ·  배치기 반동(Step 1) 대비 + 미니멀 럭셔리 하단 HUD + 1x 정속 풀가동범위",
        font=f_row,
        fill=v30.COL_GREEN,
    )
    col_titles = [
        "STEP 1 · 42° BACK SWING / 배치기 반동 (✕)",
        "STEP 2 · +22° TRAP SHRUG / BICEPS (✕)",
        "STEP 3 · 15° LEAN / SCAPULA LOCK (✓)",
        "STEP 4 · ELBOWS TO HIPS / 4×12 (✓)",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if i < 2 else v30.COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = step_static_frames[i + 1].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day2_v37_clean_master_sheet.jpg")
    board.save(sheet_path, quality=95)
    print("Saved Day 2 v37 Master Reel & Sheet:", final_mp4, sheet_path)
    return step_static_frames


def build_day1_v37_master_reel():
    print("Building Day 1 (Cable Rope Pushdown) v37 Minimal Luxury Master Reel...")
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

    wav_day1 = os.path.join(TMP_AUDIO, "day1_v37_audio.wav")
    vo_day1 = [
        (0.25, "Cable rope pushdown. Never hunch your spine."),
        (5.25, "Stopping at ninety degrees lets your front delt take over."),
        (10.35, "Hinge fifteen degrees and pin your elbows tight."),
        (15.35, "Spread the rope at full lockout. Four sets of twelve."),
    ]
    asyncio.run(v36p.synthesize_voiceover_wav(wav_day1, vo_day1, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day1_v37_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day1_v37_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    kp_s1_fixed = smooth_c_kpts[14]
    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]

    step_static_frames = {}
    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            u = t / 5.0
            rep_w = controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s1_rgb) - 1)
            base_rgb = sample_pool_interpolated(pool_s1_rgb, pos_f)
            kp = kp_s1_fixed
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom, cam_focus, cam_dest = 1.02, (690.0, 920.0), (690.0, 920.0)
            pct = int(round(80 + 6 * rep_w))
            sp_cfg = {
                "is_wrong": True,
                "pct": pct,
                "kicker_v37": "SHOULDER & WRIST SHEAR",
                "short_cue": "35° HUNCHED SPINE",
            }
        elif t < 10.0:
            u = (t - 5.0) / 5.0
            rep_w = controlled_single_rep_wave(u)
            pos_f = 4.0 + rep_w * 22.0
            base_rgb = sample_pool_interpolated(pool_s2_rgb, pos_f)
            kp = sample_kpts_interpolated(smooth_c_kpts, pos_f)
            step_idx = 2
            is_wrong = True
            active_slot = 1
            cam_zoom, cam_focus, cam_dest = 1.05, (685.0, 910.0), (695.0, 910.0)
            pct = int(round(66 + 6 * rep_w))
            sp_cfg = {
                "is_wrong": True,
                "pct": pct,
                "kicker_v37": "ANTERIOR DELTOID STEAL",
                "short_cue": "90° HALF REP",
            }
        elif t < 15.0:
            u = (t - 10.0) / 5.0
            rep_w = controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s3_rgb) - 1)
            base_rgb = sample_pool_interpolated(pool_s3_rgb, pos_f)
            kp = sample_kpts_interpolated(smooth_c_kpts, pos_f)
            step_idx = 3
            is_wrong = False
            active_slot = 2
            cam_zoom, cam_focus, cam_dest = 1.04, (685.0, 920.0), (695.0, 920.0)
            pct = int(round(89 + 6 * rep_w))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "kicker_v37": "LATERAL & LONG HEAD ISO",
                "short_cue": "15° HINGE · PIN",
            }
        else:
            u = (t - 15.0) / 5.0
            rep_w = controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s4_rgb) - 1)
            base_rgb = sample_pool_interpolated(pool_s4_rgb, pos_f)
            kp = sample_kpts_interpolated(smooth_c_kpts, pos_f)
            step_idx = 4
            is_wrong = False
            active_slot = 2
            cam_zoom, cam_focus, cam_dest = 1.06, (690.0, 925.0), (700.0, 920.0)
            pct = int(round(93 + 6 * rep_w))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "kicker_v37": "ALL 3 HEADS  ·  4 × 12 REPS",
                "short_cue": "180° LOCKOUT",
            }

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        out = cam_rgb.astype(np.float32)
        out[480:1780, :left_w] *= curve_l
        im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")

        v36.render_v36_clean_exterior_overlay(im, step_idx, map_pt, kp, is_wrong, dynamic_elbow_deg=None)
        v28.draw_luxury_top_dock(im, torso_icons, active_slot, is_wrong)
        draw_v37_minimal_luxury_bottom_hud(im, sp_cfg)

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
        "DAY 1 · CABLE ROPE PUSHDOWN (v37 MINIMAL LUXURY MASTER)  ·  미니멀 럭셔리 하단 HUD + 1x 정속 풀가동범위 + 순수 3D 근섬유 결",
        font=f_row,
        fill=v30.COL_GREEN,
    )
    col_titles = [
        "STEP 1 · 35° HUNCHED SPINE (✕)",
        "STEP 2 · 90° HALF REP / DELT (✕)",
        "STEP 3 · 15° HINGE · PIN (✓)",
        "STEP 4 · 180° LOCKOUT · 4×12 (✓)",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if i < 2 else v30.COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = step_static_frames[i + 1].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day1_v37_clean_master_sheet.jpg")
    board.save(sheet_path, quality=95)
    print("Saved Day 1 v37 Master Reel & Sheet:", final_mp4, sheet_path)
    return step_static_frames


def build_bottom_hud_comparison_sheet(d2_v37_frames: dict):
    """Builds a close-up comparison of v36 Multi-Bar Bottom HUD vs v37 Minimal Luxury Bottom HUD."""
    old_sheet = Image.open(os.path.join(OUT_DIR, "day2_v36_clean_master_sheet.jpg"))
    # Also render direct crops from v36_step1/step4 vs v37 Step 1/Step 4
    v36_s1 = Image.open(os.path.join(OUT_DIR, "v36_step1.jpg")).crop((30, 1210, 620, 1620))
    v36_s4 = Image.open(os.path.join(OUT_DIR, "v36_step4.jpg")).crop((30, 1210, 620, 1620))
    v37_d2_s1 = d2_v37_frames[1].crop((30, 1210, 620, 1620))
    v37_d2_s4 = d2_v37_frames[4].crop((30, 1210, 620, 1620))

    cw, ch = 560, 390
    pad = 24
    comp = Image.new("RGB", (cw * 2 + pad * 3, ch * 2 + 150), (10, 12, 16))
    cd = ImageDraw.Draw(comp)
    f_title = v30.f_pretendard(22, "extrabold")
    f_sub = v30.f_pretendard(15, "bold")
    cd.text(
        (24, 18),
        "v36 vs v37 BOTTOM HUD COMPARISON  ·  상단(v36): 복잡한 다중 막대그래프+하단 박스 vs 하단(v37): 미니멀 럭셔리 원라인 게이지",
        font=f_title,
        fill=v30.COL_GREEN,
    )

    panels = [
        (v36_s1, 0, 0, "v36 기존 하단 HUD (✕ 다중 막대그래프 + 하단 캡슐로 복잡함)", (180, 190, 205)),
        (v36_s4, 0, 1, "v36 기존 하단 HUD (✓ 3줄 막대그래프 + 4×12 박스로 시선 분산)", (180, 190, 205)),
        (v37_d2_s1, 1, 0, "v37 NEW 미니멀 럭셔리 HUD (✕ 숫자 + 2단 타이포 + 단일 슬림 게이지)", v30.COL_RED),
        (v37_d2_s4, 1, 1, "v37 NEW 미니멀 럭셔리 HUD (✓ 여백이 살아나는 하이엔드 1-Line 구조)", v30.COL_GREEN),
    ]
    for im_c, r_i, c_i, lbl, col in panels:
        x = pad + c_i * (cw + pad)
        y = 80 + r_i * (ch + 42)
        cd.text((x + 4, y - 24), lbl, font=f_sub, fill=col)
        rc = im_c.resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        comp.paste(rc, (x, y))

    p_comp = os.path.join(OUT_DIR, "v37_bottom_hud_comparison.jpg")
    comp.save(p_comp, quality=95)
    print("Saved Bottom HUD Comparison:", p_comp)


if __name__ == "__main__":
    d2_frames = build_day2_v37_master_reel()
    build_day1_v37_master_reel()
    build_bottom_hud_comparison_sheet(d2_frames)
