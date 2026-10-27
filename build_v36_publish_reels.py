#!/usr/bin/env python3
"""
v36 Pure Studio Master — Full-ROM Real 3D I2V Reels Builder for Day 1 & Day 2 (@gymform.3d)
===========================================================================================
Upgrades:
1. Real 3D I2V Full-ROM Exercise Motion (`121 frames` per I2V clip):
   - Day 1 (`Cable Rope Pushdown`): Uses `gym_day1_wrong_i2v.mp4` and `gym_day1_correct_i2v.mp4`
     with full-range-of-motion (Full-ROM) continuous reps across all 4 steps!
   - Day 2 (`Wide-Grip Lat Pulldown`): Uses the newly generated real 3D I2V motion clips
     `/workspace/daily_info_reels/output/gym_day2_wrong_i2v.mp4` (`121 frames`) and
     `/workspace/daily_info_reels/output/gym_day2_correct_i2v.mp4` (`121 frames`), where the mannequin
     actually performs the full overhead-to-chest lat pulldown repetition!
2. Strict v36 Pure Studio Master Standard:
   - ZERO dashed alignment lines or scattered body dots (`점선 정렬축 100% 제거`).
   - Intrinsic 3D Fiber-Strand Coloring locked dynamically to the moving 3D muscle ridges + Step 1 Native 3D X-Ray.
   - Clean exterior solid angle arcs & sculpted numbers outside the body + single clean `◎` elbow pivot ring on Steps 2-4
     tracking the moving elbow via temporally smoothed (`sigma=2.2`) MediaPipe keypoints.
   - English voiceover (`en-US-ChristopherNeural`) with verified `>= 1.78s` pauses between sentences.
   - DOES NOT upload automatically (`업로드전에 나한테 검토받아`).
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

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_AUDIO = "/workspace/daily_info_reels/_tmp_v36_audio"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_AUDIO, exist_ok=True)

W, H = eng.W, eng.H
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)  # 600 frames

DAY2_WRONG_I2V = os.path.join(OUT_DIR, "gym_day2_wrong_i2v.mp4")
DAY2_CORRECT_I2V = os.path.join(OUT_DIR, "gym_day2_correct_i2v.mp4")


async def synthesize_voiceover_wav(
    wav_path: str,
    vo_lines: list[tuple[float, str]],
    duration_s: float = 20.0,
):
    import edge_tts

    sr = 44100
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples, dtype=np.float32) / sr

    beat_env = 0.45 + 0.55 * (np.sin(2.0 * math.pi * (100.0 / 60.0) * t) > 0.72).astype(np.float32)
    sub = 0.040 * np.sin(2.0 * math.pi * 55.0 * t) * beat_env
    master = sub.copy()

    for idx_s, t_step in enumerate([0.05, 5.0, 10.0, 15.0]):
        s0 = int(t_step * sr)
        dur_c = int(0.24 * sr)
        if s0 + dur_c < n_samples:
            tc = np.linspace(0.0, 0.24, dur_c, endpoint=False)
            freq = 220.0 if idx_s < 2 else 528.0
            chime = 0.085 * np.sin(2.0 * math.pi * freq * tc) * np.exp(-tc * 13.0)
            master[s0:s0 + dur_c] += chime

    prev_end_s = 0.0
    print(f"\n=== VOICEOVER TIMING QC ({os.path.basename(wav_path)}) ===")
    for i, (target_start, text) in enumerate(vo_lines):
        mp3_p = os.path.join(TMP_AUDIO, f"vo_{os.path.basename(wav_path)}_{i}.mp3")
        wav_p = os.path.join(TMP_AUDIO, f"vo_{os.path.basename(wav_path)}_{i}.wav")
        comm = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+8%", pitch="-2Hz")
        await comm.save(mp3_p)
        subprocess.run(
            [
                "ffmpeg", "-y", "-threads", "2", "-i", mp3_p,
                "-af", "silenceremove=start_periods=1:start_silence=0.04:start_threshold=-42dB,areverse,silenceremove=start_periods=1:start_silence=0.04:start_threshold=-42dB,areverse",
                "-ac", "1", "-ar", str(sr), wav_p,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(wav_p, "rb") as wf:
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        dur_s = len(raw) / float(sr)
        actual_start = max(target_start, prev_end_s + 1.78 if i > 0 else target_start)
        gap_s = (actual_start - prev_end_s) if i > 0 else actual_start
        end_s = actual_start + dur_s
        print(f"  Sentence {i+1}: start={actual_start:.2f}s | dur={dur_s:.2f}s | end={end_s:.2f}s | pause_from_prev={gap_s:.2f}s | '{text}'")
        if i > 0:
            assert gap_s >= 1.70, f"Pause between sentences too short: {gap_s:.2f}s < 1.70s"
        prev_end_s = end_s

        s_idx = int(actual_start * sr)
        e_idx = min(n_samples, s_idx + len(raw))
        master[s_idx:e_idx] += raw[: e_idx - s_idx] * 0.92
    print("========================================================\n")

    master = np.clip(master, -0.96, 0.96)
    pcm = (master * 32767.0).astype(np.int16)
    interleaved = np.empty(n_samples * 2, dtype=np.int16)
    interleaved[0::2] = pcm
    interleaved[1::2] = pcm
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())


# ==============================================================================
# DAY 2 REAL 3D I2V FRAME SHADER (`gym_day2_wrong_i2v.mp4` & `gym_day2_correct_i2v.mp4`)
# ==============================================================================
def render_day2_i2v_frame_v36(
    bgr_720p: np.ndarray,
    step_idx: int,
    is_wrong: bool,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Takes a raw frame from `gym_day2_wrong_i2v.mp4` or `gym_day2_correct_i2v.mp4`,
    upscales to 1080x1920 with crisp unsharp mask, extracts the moving left elbow keypoint,
    and applies the v36 Pure Studio Master shader:
    - Step 1 (`✕ 42° BACK SWING`): Native 3D X-Ray lumbar spine & shoulder overload built into `gym_day2_wrong_i2v.mp4`!
    - Step 2 (`✕ TRAP SHRUG · BICEPS STEAL`): Intrinsic 3D Fiber-Strand crimson coloring on Upper Traps & Posterior Delts/Arms during the wrong pull!
    - Step 3 (`✓ 15° LEAN · SCAPULA DEPRESS`): Intrinsic 3D Fiber-Strand emerald coloring on Upper Latissimus Dorsi, Teres Major & Scapular Retractors!
    - Step 4 (`✓ ELBOWS TO HIPS · FULL LAT CONTRACT`): Intrinsic 3D Fiber-Strand emerald coloring across the entire V-taper Latissimus Dorsi, Teres Major & Lower Traps!
    """
    up = cv2.resize(bgr_720p, (W, H), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    plate_bgr = cv2.addWeighted(up, 1.16, blur, -0.16, 0)

    rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    # Detect moving left elbow keypoint via MediaPipe
    detector = uar.get_pose_detector()
    mp_im = uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb.astype(np.uint8))
    res = detector.detect(mp_im)
    if res.pose_landmarks:
        lms = res.pose_landmarks[0]
        el_x = float(np.clip(lms[13].x * W, 210.0, 480.0))
        el_y = float(np.clip(lms[13].y * H, 450.0, 1120.0))
        sh_x = float(np.clip(lms[11].x * W, 420.0, 680.0))
        sh_y = float(np.clip(lms[11].y * H, 580.0, 900.0))
    else:
        el_x, el_y = 320.0, 880.0
        sh_x, sh_y = 540.0, 740.0

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    gb_diff = np.maximum(0.0, g_ch - b_ch)

    fiber_signal = 0.45 * r_ch + 0.55 * g_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 1.0)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.6)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 7.0)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 15.0)

    hf_ridge = (sigma_micro - sigma_ridge) * 3.1 + (sigma_ridge - sigma_belly) * 1.75
    fs = np.clip(0.48 + hf_ridge / 42.0, 0.0, 1.0) ** 1.35
    hv = np.clip((sigma_belly - 28.0) / 165.0, 0.0, 1.0)
    sulc = np.clip((sigma_broad - sigma_ridge - 3.5) / 22.0, 0.0, 1.0)

    red_any = np.clip((rb_diff - 12.0) / 85.0, 0.0, 1.0)
    grey_sculpt = np.clip(34.0 + 148.0 * (hv ** 1.05) + hf_ridge * 0.95, 16.0, 235.0)
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    neutral_lum = np.clip(base_lum * (1.0 - red_any) + grey_sculpt * red_any, 14.0, 240.0)

    norm = neutral_lum / 255.0
    mono = np.stack([
        eng.BG_CHARCOAL[0] + (234 - eng.BG_CHARCOAL[0]) * (norm ** 0.92),
        eng.BG_CHARCOAL[1] + (238 - eng.BG_CHARCOAL[1]) * (norm ** 0.92),
        eng.BG_CHARCOAL[2] + (244 - eng.BG_CHARCOAL[2]) * (norm ** 0.92),
    ], axis=2)

    if step_idx == 1:
        # Step 1 (`gym_day2_wrong_i2v.mp4`): Preserve the native 3D X-Ray lumbar spine & shoulder overload
        # as the mannequin swings its back dynamically!
        xray_alpha = np.clip((rb_diff - 20.0) / 90.0, 0.0, 1.0)
        xray_boost = rgb.copy()
        xray_boost[:, :, 0] = np.clip(r_ch * 1.15 + 26.0, 0, 255)
        xray_boost[:, :, 1] = np.clip(g_ch * 0.94, 0, 255)
        out = mono * (1.0 - xray_alpha[:, :, None]) + xray_boost * xray_alpha[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8), (el_x, el_y)

    if step_idx == 2:
        # Step 2 (`gym_day2_wrong_i2v.mp4`): Focus crimson 3D fiber-strands on the Upper Traps, Shoulder & Arms
        # while suppressing the lower lumbar glow so Step 2 distinctly highlights Upper Trap/Arm compensation!
        upper_gate = np.clip((980.0 - yy) / 85.0, 0.0, 1.0)
        arm_flare = np.exp(-(((xx - el_x) / 145.0) ** 2 + ((yy - el_y) / 145.0) ** 2)) * np.clip((base_lum - 38.0) / 90.0, 0.0, 1.0)
        fe = np.clip(np.clip((rb_diff - 22.0) / 105.0, 0.0, 1.0) * upper_gate + arm_flare * 0.72, 0.0, 1.0)
        hc = np.clip((gb_diff - 10.0) / 90.0, 0.0, 1.0) * upper_gate
    elif step_idx == 3:
        # Step 3 (`gym_day2_correct_i2v.mp4`): Upper Lat, Teres Major & Scapula Depression
        lat_gate = np.clip((yy - (sh_y - 35.0)) / 55.0, 0.0, 1.0) * np.clip((1045.0 - yy) / 75.0, 0.0, 1.0)
        fe = np.clip((rb_diff - 26.0) / 115.0, 0.0, 1.0) * lat_gate
        hc = np.clip((gb_diff - 10.0) / 90.0, 0.0, 1.0) * lat_gate
    else:
        # Step 4 (`gym_day2_correct_i2v.mp4`): Full V-Taper Latissimus Dorsi, Teres Major & Lower Traps
        fe = np.clip((rb_diff - 22.0) / 110.0, 0.0, 1.0)
        hc = np.clip((gb_diff - 10.0) / 88.0, 0.0, 1.0)

    strand_mod = (0.22 + 0.78 * fs) * (1.0 - 0.65 * sulc)
    fiber_alpha = np.clip(fe * (0.64 + 0.66 * strand_mod), 0.0, 0.96)[:, :, None]
    ridge_boost = np.clip(fs * 0.65 + hc * 0.55, 0.0, 1.0)[:, :, None]
    vol_shade = np.clip(0.28 + 0.72 * hv * (0.45 + 0.55 * fs), 0.0, 1.0)[:, :, None]

    if is_wrong:
        deep_col = np.array([155.0, 18.0, 16.0], dtype=np.float32)
        mid_col = np.array([248.0, 52.0, 34.0], dtype=np.float32)
        crest_col = np.array([255.0, 195.0, 110.0], dtype=np.float32)
    else:
        deep_col = np.array([8.0, 110.0, 48.0], dtype=np.float32)
        mid_col = np.array([22.0, 224.0, 98.0], dtype=np.float32)
        crest_col = np.array([185.0, 255.0, 205.0], dtype=np.float32)

    fiber_rgb = (
        deep_col * (1.0 - vol_shade)
        + mid_col * vol_shade * (1.0 - ridge_boost * 0.65)
        + crest_col * (ridge_boost ** 1.35) * 0.85
    )
    out = mono * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha
    return np.clip(out, 0, 255).astype(np.uint8), (el_x, el_y)


def render_day2_v36_exterior_callouts(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
    el_pt: tuple[float, float],
    is_wrong: bool,
):
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
        c_arc = S(map_pt(470, 940))
        r_a = 165 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=192, end=246, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(192)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 192 - 90, 12 * SS, (*col_soft, 255))

        p_b = map_pt(405, 760)
        v30.draw_floating_number_badge(od, sd, (int(p_b[0]), int(p_b[1])), "42°", "BACK SWING", col_main, col_soft, align="right")

        p_l = map_pt(855, 1085)
        v30.draw_floating_number_badge(od, sd, (min(955, int(p_l[0])), int(p_l[1])), "82%", "LUMBAR LOAD", col_main, col_soft, align="left")

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


def build_day2_v36_master_reel():
    print("Building Day 2 (Wide-Grip Lat Pulldown) v36 Master Reel from REAL 3D I2V Videos...")
    cap_w = cv2.VideoCapture(DAY2_WRONG_I2V)
    raw_w = []
    while True:
        ret, fr = cap_w.read()
        if not ret:
            break
        raw_w.append(fr)
    cap_w.release()

    cap_c = cv2.VideoCapture(DAY2_CORRECT_I2V)
    raw_c = []
    while True:
        ret, fr = cap_c.read()
        if not ret:
            break
        raw_c.append(fr)
    cap_c.release()

    print(f"Loaded Day 2 Real 3D I2V clips: wrong={len(raw_w)} frames, correct={len(raw_c)} frames")

    # Pre-render the 4 step pools across the real 3D I2V frames:
    # - Wrong I2V frames 0..72 cover the cleanest full-ROM backward-swing & shrug repetition
    # - Correct I2V frames 0..120 cover the full overhead stretch (f=120) -> chest pull (f=40) -> return repetition
    raw_w_sub = raw_w[:72]
    pool_s1 = []
    pool_s2 = []
    raw_el_w = []
    for fr in raw_w_sub:
        rgb1, el1 = render_day2_i2v_frame_v36(fr, 1, True)
        rgb2, _   = render_day2_i2v_frame_v36(fr, 2, True)
        pool_s1.append(rgb1)
        pool_s2.append(rgb2)
        raw_el_w.append(el1)

    el_w_arr = np.array(raw_el_w, dtype=np.float32)
    el_w_x = gaussian_filter1d(el_w_arr[:, 0], sigma=2.2, mode="nearest")
    el_w_y = gaussian_filter1d(el_w_arr[:, 1], sigma=2.2, mode="nearest")
    smooth_el_w = [(float(el_w_x[i]), float(el_w_y[i])) for i in range(len(raw_w_sub))]

    pool_s3 = []
    pool_s4 = []
    raw_el_c = []
    for fr in raw_c:
        rgb3, el3 = render_day2_i2v_frame_v36(fr, 3, False)
        rgb4, _   = render_day2_i2v_frame_v36(fr, 4, False)
        pool_s3.append(rgb3)
        pool_s4.append(rgb4)
        raw_el_c.append(el3)

    el_c_arr = np.array(raw_el_c, dtype=np.float32)
    el_c_x = gaussian_filter1d(el_c_arr[:, 0], sigma=2.2, mode="nearest")
    el_c_y = gaussian_filter1d(el_c_arr[:, 1], sigma=2.2, mode="nearest")
    smooth_el_c = [(float(el_c_x[i]), float(el_c_y[i])) for i in range(len(raw_c))]

    # Top 3-thumbnail dock icons
    size = (142, 114)
    crop_box = (460, 620, 880, 1080)
    inact_base = Image.fromarray(pool_s3[35]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 135)))

    im0 = Image.fromarray(pool_s1[45]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im1 = Image.fromarray(pool_s2[50]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im2 = Image.fromarray(pool_s4[40]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    # Synthesize 20.0s Day 2 English voiceover with >=1.78s pauses between sentences
    wav_day2 = os.path.join(TMP_AUDIO, "day2_v36_audio.wav")
    vo_day2 = [
        (0.25, "Wide grip lat pulldown. Stop swinging your lower back."),
        (5.93, "Shrugging your shoulders lets your biceps steal the load."),
        (10.40, "Set a fifteen degree lean and depress your scapula first."),
        (15.40, "Drive your elbows to your hips. Four sets of twelve."),
    ]
    asyncio.run(synthesize_voiceover_wav(wav_day2, vo_day2, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day2_v36_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day2_v36_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]

    step_static_frames = {}
    motion_sheet_samples = []
    cover_pil = None

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            # Step 1 (0.0s - 5.0s): Full-ROM Wrong Back Swing & Lumbar Overload (2 full dynamic rep cycles across raw_w_sub)
            u = t / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            p_i = int(round(wave_p * (len(pool_s1) - 1)))
            base_rgb = pool_s1[p_i]
            el_pt = smooth_el_w[p_i]
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom = 1.03
            cam_focus, cam_dest = (650.0, 940.0), (665.0, 940.0)
            pct = int(round(76 + 8 * wave_p))
            sp_cfg = {
                "is_wrong": True,
                "pct": pct,
                "dial_tag": "LUMBAR RISK",
                "short_cue": "✕ 42° BACK SWING",
                "splits": [("LOW BACK", pct, v30.COL_RED), ("LATS", 100 - pct, v30.COL_MUTED)],
                "rx_num": "42°",
                "rx_unit": "SWING",
            }
        elif t < 10.0:
            # Step 2 (5.0s - 10.0s): Full-ROM Shrugged Shoulders & Biceps/Arm Pull
            u = (t - 5.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            p_i = int(round(wave_p * (len(pool_s2) - 1)))
            base_rgb = pool_s2[p_i]
            el_pt = smooth_el_w[p_i]
            step_idx = 2
            is_wrong = True
            active_slot = 1
            cam_zoom = 1.04
            cam_focus, cam_dest = (635.0, 900.0), (655.0, 900.0)
            pct = int(round(66 + 8 * wave_p))
            sp_cfg = {
                "is_wrong": True,
                "pct": pct,
                "dial_tag": "BICEPS STEAL",
                "short_cue": "✕ TRAP SHRUG · CURL",
                "splits": [("BICEPS · TRAP", pct, v30.COL_RED), ("LATS", 100 - pct, v30.COL_MUTED)],
                "rx_num": "+22°",
                "rx_unit": "SHRUG",
            }
        elif t < 15.0:
            # Step 3 (10.0s - 15.0s): Full-ROM Scapula Depression & Lat Pull (sweeping from overhead f=118 down to chest pull f=36 and back!)
            u = (t - 10.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            # At wave_p=0 -> f=118 (full overhead stretch); at wave_p=1 -> f=36 (full chest pull contraction)
            p_i = int(round(118 - wave_p * 82))
            base_rgb = pool_s3[p_i]
            el_pt = smooth_el_c[p_i]
            step_idx = 3
            is_wrong = False
            active_slot = 2
            cam_zoom = 1.03
            cam_focus, cam_dest = (645.0, 910.0), (660.0, 910.0)
            pct = int(round(88 + 8 * wave_p))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "dial_tag": "SCAPULA LOCK",
                "short_cue": "✓ 15° LEAN · DEPRESS",
                "splits": [("UPPER LAT", pct, v30.COL_GREEN), ("TERES MAJ", pct - 3, v30.COL_GREEN)],
                "rx_num": "15°",
                "rx_unit": "LEAN LOCK",
            }
        else:
            # Step 4 (15.0s - 20.0s): Full-ROM Elbows-to-Hips Peak Lat Contraction (sweeping f=118 -> f=38 -> f=118)
            u = (t - 15.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            p_i = int(round(118 - wave_p * 80))
            base_rgb = pool_s4[p_i]
            el_pt = smooth_el_c[p_i]
            step_idx = 4
            is_wrong = False
            active_slot = 2
            cam_zoom = 1.04
            cam_focus, cam_dest = (650.0, 915.0), (665.0, 915.0)
            pct = int(round(91 + 7 * wave_p))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "dial_tag": "FULL LAT ISO",
                "short_cue": "✓ ELBOWS TO HIPS",
                "splits": [
                    ("LAT DORSI", pct, v30.COL_GREEN),
                    ("TERES MAJ", max(88, pct - 3), v30.COL_GREEN),
                    ("LOW TRAP", max(86, pct - 5), v30.COL_GREEN),
                ],
                "rx_num": "4×12",
                "rx_unit": "REPS · 1s",
            }

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        out = cam_rgb.astype(np.float32)
        out[480:1780, :left_w] *= curve_l
        im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")

        render_day2_v36_exterior_callouts(im, step_idx, map_pt, el_pt, is_wrong)
        v28.draw_luxury_top_dock(im, torso_icons, active_slot, is_wrong)
        v30.draw_v30_graphic_telemetry_bottom_left(im, sp_cfg, mode="A")

        rgb_frame = im.convert("RGB")
        if f_idx in (int(1.6 * FPS), int(6.6 * FPS), int(11.6 * FPS), int(16.6 * FPS)):
            step_static_frames[step_idx] = rgb_frame.copy()
        if f_idx == int(16.6 * FPS):
            cover_pil = rgb_frame.copy()

        # Collect 8 dynamic frames showing the full ROM across Wrong & Correct reps
        sample_targets = [
            (int(0.2 * FPS), "1. STEP 1 (f=06) · TOP START (✕)", True),
            (int(1.6 * FPS), "2. STEP 1 (f=48) · 42° BACK SWING PEAK (✕)", True),
            (int(6.6 * FPS), "3. STEP 2 (f=198) · +22° SHRUG & ARM PULL (✕)", True),
            (int(10.1 * FPS), "4. STEP 3 (f=303) · OVERHEAD LAT STRETCH (✓)", False),
            (int(10.9 * FPS), "5. STEP 3 (f=327) · SCAPULA DEPRESS INIT (✓)", False),
            (int(11.6 * FPS), "6. STEP 3 (f=348) · MID ELBOW DRIVE (✓)", False),
            (int(16.6 * FPS), "7. STEP 4 (f=498) · FULL CHEST LOCKOUT 98% (✓)", False),
            (int(18.2 * FPS), "8. STEP 4 (f=546) · CONTROLLED ECCENTRIC UP (✓)", False),
        ]
        for st_idx, st_lbl, st_w in sample_targets:
            if f_idx == st_idx:
                motion_sheet_samples.append((rgb_frame.copy(), st_lbl, st_w))

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

    cover_path = os.path.join(OUT_DIR, "day2_v36_cover.jpg")
    cover_pil.save(cover_path, quality=95)

    # 1. Build 4-Step Full-Body 9:16 Master Sheet for Day 2 (`day2_v36_clean_master_sheet.jpg`)
    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 2 · WIDE-GRIP LAT PULLDOWN (v36 FULL-ROM I2V MASTER)  ·  실제 풀가동범위(Full-ROM) 3D 동작 + 순수 근섬유 결 + 외곽 솔리드 아크",
        font=f_row,
        fill=v30.COL_GREEN,
    )
    col_titles = [
        "STEP 1 · 42° BACK SWING / LUMBAR (✕)",
        "STEP 2 · +22° TRAP SHRUG / BICEPS (✕)",
        "STEP 3 · 15° LEAN / SCAPULA LOCK (✓)",
        "STEP 4 · ELBOWS TO HIPS / FULL LAT (✓)",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if i < 2 else v30.COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = step_static_frames[i + 1].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day2_v36_clean_master_sheet.jpg")
    board.save(sheet_path, quality=95)

    # 2. Build 8-Frame Full-ROM Motion Proof Sheet (`day2_v36_dynamic_8frame_motion_sheet.jpg`)
    mw, mh = 360, 640
    mpad = 20
    msheet = Image.new("RGB", (mw * 4 + mpad * 5, mh * 2 + 165), (10, 12, 16))
    msd = ImageDraw.Draw(msheet)
    msd.text(
        (24, 18),
        "DAY 2 v36 FULL-ROM 3D MOTION PROOF  ·  상완 최대 신전(Overhead Stretch) → 견갑 하강 → 쇄골 하단 수축(Chest Lockout) → 이완 복귀",
        font=f_row,
        fill=v30.COL_GREEN,
    )
    for idx_s, (im_p, lbl, is_w) in enumerate(motion_sheet_samples):
        r_i = idx_s // 4
        c_i = idx_s % 4
        x = mpad + c_i * (mw + mpad)
        y = 75 + r_i * (mh + 42)
        col = v30.COL_RED if is_w else v30.COL_GREEN
        msd.text((x + 4, y - 24), lbl, font=v30.f_pretendard(14, "bold"), fill=col)
        thumb = im_p.resize((mw, mh), Image.Resampling.LANCZOS)
        ImageDraw.Draw(thumb).rectangle([0, 0, mw - 1, mh - 1], outline=(255, 255, 255, 48), width=2)
        msheet.paste(thumb, (x, y))

    p_msheet = os.path.join(OUT_DIR, "day2_v36_dynamic_8frame_motion_sheet.jpg")
    msheet.save(p_msheet, quality=95)
    print("Saved Day 2 Full-ROM v36 Master Reel, Cover, 4-Step Sheet & 8-Frame Motion Sheet:", final_mp4, p_msheet)


if __name__ == "__main__":
    build_day2_v36_master_reel()
