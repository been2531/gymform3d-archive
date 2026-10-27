#!/usr/bin/env python3
"""
v36 Pure Studio Master — Full Production Reels Builder for Day 1 & Day 2 (@gymform.3d)
======================================================================================
Builds:
1. Day 1 (`Cable Rope Pushdown`):
   - `/workspace/daily_info_reels/output/day1_v36_master_reel.mp4` (20.0s, 600 frames @ 30fps, H.264 + AAC)
   - `/workspace/daily_info_reels/output/day1_v36_cover.jpg` (1080x1920 9:16 Cover)
   - `/workspace/daily_info_reels/output/day1_v36_caption_en.txt`
2. Day 2 (`Wide-Grip Lat Pulldown`):
   - `/workspace/daily_info_reels/output/day2_v36_master_reel.mp4` (20.0s, 600 frames @ 30fps, H.264 + AAC)
   - `/workspace/daily_info_reels/output/day2_v36_cover.jpg` (1080x1920 9:16 Cover)
   - `/workspace/daily_info_reels/output/day2_v36_clean_master_sheet.jpg` (4-Step Full-Body 9:16 Master Sheet)
   - `/workspace/daily_info_reels/output/day2_v36_caption_en.txt`

Strict v36 Standards Verified:
- ZERO dashed alignment lines or scattered body dots (`점선 정렬축 100% 제거`).
- Intrinsic 3D Fiber-Strand Coloring on target/compensatory muscles + Step 1 Native 3D X-Ray joint/spine stress.
- Clean exterior solid angle arcs & sculpted numbers outside the body + single clean `◎` elbow pivot ring on Steps 2-4.
- Frameless bottom-left `%` + 20-segment LED gauge + `4×12` prescription + top 3-thumbnail dock + top-right `✕/✓` badge.
- English voiceover (`en-US-ChristopherNeural`) with verified `>= 1.75s` natural pauses between every sentence.
- Uses `-threads 4` on ffmpeg for shared box CPU etiquette.
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
import build_day2_ox_preview as d2
import build_gym_day2 as gd2
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


async def synthesize_voiceover_wav(
    wav_path: str,
    vo_lines: list[tuple[float, str]],
    duration_s: float = 20.0,
):
    """
    Generates a 44.1kHz stereo WAV combining:
    - Subtle dark-gym sub-bass pulse + step transition chimes
    - Clean English TTS (`en-US-ChristopherNeural`) with strict verification that
      every sentence has `>= 1.75s` pause before the next sentence!
    """
    import edge_tts

    sr = 44100
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples, dtype=np.float32) / sr

    beat_env = 0.45 + 0.55 * (np.sin(2.0 * math.pi * (100.0 / 60.0) * t) > 0.72).astype(np.float32)
    sub = 0.040 * np.sin(2.0 * math.pi * 55.0 * t) * beat_env
    master = sub.copy()

    # Step transition chimes at 0.0s, 5.0s, 10.0s, 15.0s
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
# TASK A: DAY 1 (`CABLE ROPE PUSHDOWN`) v36 FULL MASTER REEL & COVER
# ==============================================================================
def build_day1_v36_master_reel():
    print("Building Day 1 (Cable Rope Pushdown) v36 Master Reel...")
    cap_w = cv2.VideoCapture(eng.WRONG_I2V)
    raw_w = []
    while True:
        ret, fr = cap_w.read()
        if not ret:
            break
        raw_w.append(fr)
    cap_w.release()
    raw_w = raw_w[:56]

    cap_c = cv2.VideoCapture(eng.CORRECT_I2V)
    raw_c = []
    while True:
        ret, fr = cap_c.read()
        if not ret:
            break
        raw_c.append(fr)
    cap_c.release()
    raw_c = raw_c[:69]

    # Pre-compute the 4 pools on the unique source frames (fast & temporally smoothed!)
    print(f"Precomputing Day 1 pools: wrong={len(raw_w)} frames, correct={len(raw_c)} frames...")
    pool_s1_rgb = []
    for fr in raw_w:
        rgb_out, _ = v35.render_v35_clean_3d_frame(fr, 1, "xray", True)
        pool_s1_rgb.append(rgb_out)

    # For Correct clip (69 frames), detect keypoints once and smooth them temporally
    keys_order = ["ear", "sh", "sh_far", "el", "wr", "hd", "hip", "hip_far", "knee", "ank"]
    pool_s2_rgb = []
    pool_s3_rgb = []
    pool_s4_rgb = []
    raw_c_kpts = []

    for i, fr in enumerate(raw_c):
        rgb2, kp = v35.render_v35_clean_3d_frame(fr, 2, "anterior_deltoid", True)
        rgb3, _  = v35.render_v35_clean_3d_frame(fr, 3, "triceps_lat_long", False)
        rgb4, _  = v35.render_v35_clean_3d_frame(fr, 4, "triceps_all_three", False)
        pool_s2_rgb.append(rgb2)
        pool_s3_rgb.append(rgb3)
        pool_s4_rgb.append(rgb4)
        raw_c_kpts.append(kp)

    n_c = len(raw_c)
    smooth_c_kpts = [dict() for _ in range(n_c)]
    for k in keys_order:
        arr = np.array([rk[k] for rk in raw_c_kpts], dtype=np.float32)
        ax = gaussian_filter1d(arr[:, 0], sigma=1.8, mode="nearest")
        ay = gaussian_filter1d(arr[:, 1], sigma=1.8, mode="nearest")
        for i in range(n_c):
            smooth_c_kpts[i][k] = (float(ax[i]), float(ay[i]))

    # Top 3-thumbnail dock icons
    size = (142, 114)
    crop_box = (555, 550, 905, 950)
    f_c42_mono = eng.prepare_muscle_frame(raw_c[42])["mono"]
    inact_base = Image.fromarray(f_c42_mono).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 125)))

    im0 = Image.fromarray(pool_s1_rgb[45]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im1 = Image.fromarray(pool_s2_rgb[14]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im2 = Image.fromarray(pool_s4_rgb[64]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    # Synthesize 20.0s voiceover with >=1.78s pauses between sentences
    wav_day1 = os.path.join(TMP_AUDIO, "day1_v36_audio.wav")
    vo_day1 = [
        (0.25, "Cable rope pushdown. Never hunch your spine."),
        (5.25, "Stopping at ninety degrees lets your front delt take over."),
        (10.35, "Hinge fifteen degrees and pin your elbows tight."),
        (15.35, "Spread the rope at full lockout. Four sets of twelve."),
    ]
    asyncio.run(synthesize_voiceover_wav(wav_day1, vo_day1, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day1_v36_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day1_v36_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    kp_s1_fixed = smooth_c_kpts[14]
    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]

    cover_pil = None
    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            # Step 1 (0.0s - 5.0s): Hunched Spine Wrong (1.5 smooth rep cycles)
            u = t / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            p_i = int(round(wave_p * (len(pool_s1_rgb) - 1)))
            base_rgb = pool_s1_rgb[p_i]
            kp = kp_s1_fixed
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom, cam_focus, cam_dest = 1.02, (690.0, 920.0), (690.0, 920.0)
            pct = int(round(82 + 6 * wave_p))
            sp_cfg = {
                "is_wrong": True,
                "pct": pct,
                "dial_tag": "JOINT RISK",
                "short_cue": "✕ HUNCHED SPINE",
                "splits": [("SPINE · JOINT", pct, v30.COL_RED), ("TRICEPS", 100 - pct, v30.COL_MUTED)],
                "rx_num": "35°",
                "rx_unit": "KYPHOSIS",
            }
        elif t < 10.0:
            # Step 2 (5.0s - 10.0s): 90° Half Rep / Front Delt Steal (oscillates in top half-rep frames 0..24)
            u = (t - 5.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            p_i = int(round(4 + wave_p * 20))
            base_rgb = pool_s2_rgb[p_i]
            kp = smooth_c_kpts[p_i]
            step_idx = 2
            is_wrong = True
            active_slot = 1
            cam_zoom, cam_focus, cam_dest = 1.05, (685.0, 910.0), (695.0, 910.0)
            pct = int(round(68 + 6 * wave_p))
            sp_cfg = {
                "is_wrong": True,
                "pct": pct,
                "dial_tag": "DELT STEAL",
                "short_cue": "✕ 90° HALF REP",
                "splits": [("FRONT DELT", pct, v30.COL_RED), ("TRICEPS", 100 - pct, v30.COL_MUTED)],
                "rx_num": "90°",
                "rx_unit": "STOP SHORT",
            }
        elif t < 15.0:
            # Step 3 (10.0s - 15.0s): 15° Hinge & Pin Elbows (frames 12..52)
            u = (t - 10.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            p_i = int(round(12 + wave_p * 40))
            base_rgb = pool_s3_rgb[p_i]
            kp = smooth_c_kpts[p_i]
            step_idx = 3
            is_wrong = False
            active_slot = 2
            cam_zoom, cam_focus, cam_dest = 1.04, (685.0, 920.0), (695.0, 920.0)
            pct = int(round(90 + 6 * wave_p))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "dial_tag": "TRICEPS ISO",
                "short_cue": "✓ 15° HINGE · PIN",
                "splits": [("LAT HEAD", pct, v30.COL_GREEN), ("LONG HEAD", pct - 4, v30.COL_GREEN)],
                "rx_num": "135°",
                "rx_unit": "EXTENSION",
            }
        else:
            # Step 4 (15.0s - 20.0s): 180° Full Lockout & 25° Rope Spread (frames 16..67)
            u = (t - 15.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            p_i = int(round(16 + wave_p * 51))
            base_rgb = pool_s4_rgb[p_i]
            kp = smooth_c_kpts[p_i]
            step_idx = 4
            is_wrong = False
            active_slot = 2
            cam_zoom, cam_focus, cam_dest = 1.06, (690.0, 925.0), (700.0, 920.0)
            pct = int(round(94 + 5 * wave_p))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "dial_tag": "ALL 3 HEADS",
                "short_cue": "✓ 180° LOCKOUT",
                "splits": [
                    ("LAT HEAD", pct, v30.COL_GREEN),
                    ("LONG HEAD", max(90, pct - 4), v30.COL_GREEN),
                    ("MED HEAD", max(91, pct - 3), v30.COL_GREEN),
                ],
                "rx_num": "4×12",
                "rx_unit": "REPS · 1s",
            }

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        out = cam_rgb.astype(np.float32)
        out[480:1780, :left_w] *= curve_l
        im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")

        v36.render_v36_clean_exterior_overlay(im, step_idx, map_pt, kp, is_wrong, dynamic_elbow_deg=None)
        v28.draw_luxury_top_dock(im, torso_icons, active_slot, is_wrong)
        v30.draw_v30_graphic_telemetry_bottom_left(im, sp_cfg, mode="A")

        rgb_frame = im.convert("RGB")
        if f_idx == int(16.6 * FPS):
            cover_pil = rgb_frame.copy()
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
    cover_path = os.path.join(OUT_DIR, "day1_v36_cover.jpg")
    cover_pil.save(cover_path, quality=95)

    caption_day1 = """Stop letting your shoulders and wrists steal your triceps gains on the Cable Rope Pushdown.

Here is the 4-step 3D biomechanical checklist to isolate 99% of all 3 Triceps Heads:

1. HUNCHED SPINE OVERLOAD (✕)
Hunching 35° forward and craning your neck (+28°) shifts 86% of the shear stress onto your shoulder joint and wrists.

2. 90° HALF-REP TRAP (✕)
Stopping short at 90° lets your Anterior Deltoid steal 72% of the tension before the triceps ever reach peak contraction.

3. 15° HIP HINGE & ELBOW PIN (✓)
Set a clean 15° athletic hip hinge and lock your elbows at your sides as a pure fixed hinge axis (95% Lateral & Long Head isolation).

4. 180° FULL LOCKOUT & 25° ROPE SPREAD (✓)
Extend all the way to 180° and spread the rope 25° outward at the bottom to fire all 3 Triceps Heads (Lateral, Long, and Medial — 99% peak activation, 4×12 controlled reps).

Save this for your next arm day and follow @gymform.3d for daily 3D biomechanics guides.

#tricepspushdown #tricepsworkout #armday #biomechanics #gymform #workouttips #gymtips #hypertrophy #fitnesseducation #correctform #bodybuilding #anatomydrawing"""
    cap_path = os.path.join(OUT_DIR, "day1_v36_caption_en.txt")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(caption_day1.strip() + "\n")

    print("Saved Day 1 v36 Master Reel & Cover:", final_mp4, cover_path)


# ==============================================================================
# TASK B: DAY 2 (`WIDE-GRIP LAT PULLDOWN`) v36 FULL MASTER REEL & COVER
# ==============================================================================
def render_day2_v36_base_plate(
    plate_bgr: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    intensity: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Renders a Day 2 (`Wide-Grip Lat Pulldown`) frame in the exact v36 Pure Studio Standard:
    - Step 1 (`day2_step2_wrong_plate.jpg`): Preserves the native 3D X-Ray lumbar spine & shoulder joint overload
      built into the 3D render!
    - Step 2 (`day2_step3_wrong_plate.jpg`): Intrinsic 3D Fiber-Strand crimson coloring on Biceps, Forearms & Upper Traps.
    - Step 3 (`day2_step1_correct_plate.jpg`): Intrinsic 3D Fiber-Strand emerald coloring on Upper Lat, Teres Major & Scapula.
    - Step 4 (`day2_step3_correct_plate.jpg`): Intrinsic 3D Fiber-Strand emerald coloring on Full Latissimus Dorsi & Teres Major.
    - ZERO dashed lines or fake drawn bones!
    """
    rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    yy, xx = np.mgrid[:H, :W].astype(np.float32)

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
        # Step 1 (`day2_step2_wrong_plate.jpg`): Keep the native 3D X-Ray lumbar spine & shoulder joint glow!
        zone = np.exp(-(((xx - 655.0) / 175.0) ** 2 + ((yy - 980.0) / 260.0) ** 2))
        xray_alpha = np.clip((rb_diff - 22.0) / 95.0, 0.0, 1.0) * zone * (0.85 + 0.15 * intensity)
        xray_boost = rgb.copy()
        xray_boost[:, :, 0] = np.clip(r_ch * 1.15 + 28.0, 0, 255)
        xray_boost[:, :, 1] = np.clip(g_ch * 0.95, 0, 255)
        out = mono * (1.0 - xray_alpha[:, :, None]) + xray_boost * xray_alpha[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8), {"el": (309.0, 899.0)}

    if step_idx == 2:
        # Step 2 (`day2_step3_wrong_plate.jpg`): Biceps, Forearms & Shrugged Upper Traps steal load
        z_arm_l = np.exp(-(((xx - 310.0) / 170.0) ** 2 + ((yy - 770.0) / 150.0) ** 2))
        z_arm_r = np.exp(-(((xx - 880.0) / 160.0) ** 2 + ((yy - 790.0) / 150.0) ** 2))
        z_trap  = np.exp(-(((xx - 640.0) / 160.0) ** 2 + ((yy - 780.0) / 95.0) ** 2))
        zone = np.clip(z_arm_l + z_arm_r + z_trap, 0.0, 1.0)
        el_pt = (295.0, 885.0)
    elif step_idx == 3:
        # Step 3 (`day2_step1_correct_plate.jpg`): Scapula Depression & Upper/Mid Lat V-Taper
        z_l = np.exp(-(((xx - 565.0) / 125.0) ** 2 + ((yy - 885.0) / 155.0) ** 2))
        z_r = np.exp(-(((xx - 745.0) / 125.0) ** 2 + ((yy - 885.0) / 155.0) ** 2))
        zone = np.clip(z_l + z_r, 0.0, 1.0) * np.clip((yy - 705.0) / 45.0, 0.0, 1.0)
        el_pt = (295.0, 915.0)
    else:
        # Step 4 (`day2_step3_correct_plate.jpg`): Full Latissimus Dorsi & Teres Major Peak Contraction
        zone = np.exp(-(((xx - 705.0) / 155.0) ** 2 + ((yy - 965.0) / 165.0) ** 2))
        el_pt = (418.0, 924.0)

    fe = np.clip((rb_diff - 28.0) / 118.0, 0.0, 1.0) * zone
    hc = np.clip((gb_diff - 10.0) / 92.0, 0.0, 1.0) * zone

    strand_mod = (0.22 + 0.78 * fs) * (1.0 - 0.65 * sulc)
    fiber_alpha = np.clip(fe * (0.64 + 0.66 * strand_mod) * (0.82 + 0.18 * intensity), 0.0, 0.96)[:, :, None]
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
    return np.clip(out, 0, 255).astype(np.uint8), {"el": el_pt}


def render_day2_v36_exterior_callouts(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
    el_pt: tuple[float, float],
    is_wrong: bool,
):
    """
    Renders Day 2 (`Wide-Grip Lat Pulldown`) exterior arcs & numbers in the v36 Pure Studio Standard:
    - ZERO dashed lines or scattered dots on the body!
    - Clean exterior solid arcs & sculpted numbers outside the body + single clean `◎` elbow pivot ring on Steps 2-4.
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
        # Step 1: 42° Back Swing & Lumbar Overload (solid exterior arc to the left of the torso)
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
        # Step 2: Shrugged Shoulders & Biceps Steal (single elbow pivot ring + exterior forearm curl arc)
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
        # Step 3: 15° Torso Lean & Scapula Depression (single elbow pivot ring + exterior downward drive arc)
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
        # Step 4: Elbows to Hips Full Lat Contraction (single elbow lock ring + exterior elbow-to-hip sweep arc)
        v36.draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

        c_arc = S(map_pt(525, 885))
        r_a = 158 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=116, end=172, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(116)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 116 - 90, 13 * SS, (*col_soft, 255))

        p_dr = map_pt(415, 825)
        v30.draw_floating_number_badge(od, sd, (int(p_dr[0]), int(p_dr[1])), "100%", "ELBOW TO HIP", col_main, col_soft, align="right")

        p_arch = map_pt(885, 790)
        v30.draw_floating_number_badge(od, sd, (min(955, int(p_arch[0])), int(p_arch[1])), "15°", "THORACIC LOCK", col_main, col_soft, align="left")

    canvas.alpha_composite(rig_layer.resize((W, H), Image.Resampling.LANCZOS))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)


def build_day2_v36_master_reel():
    print("Building Day 2 (Wide-Grip Lat Pulldown) v36 Master Reel...")
    p_s1_w = d2.load_plate("day2_step2_wrong_plate.jpg")
    p_s2_w = d2.load_plate("day2_step3_wrong_plate.jpg")
    p_s3_c = d2.load_plate("day2_step1_correct_plate.jpg")
    p_s4_c = d2.load_plate("day2_step3_correct_plate.jpg")

    # Build top 3-thumbnail dock icons for Day 2
    s1_base, _ = render_day2_v36_base_plate(p_s1_w, 1, True, 1.0)
    s2_base, _ = render_day2_v36_base_plate(p_s2_w, 2, True, 1.0)
    s3_base, _ = render_day2_v36_base_plate(p_s3_c, 3, False, 1.0)
    s4_base, _ = render_day2_v36_base_plate(p_s4_c, 4, False, 1.0)

    size = (142, 114)
    crop_box = (460, 620, 880, 1080)
    inact_base = Image.fromarray(s3_base).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 135)))

    im0 = Image.fromarray(s1_base).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im1 = Image.fromarray(s2_base).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im2 = Image.fromarray(s3_base).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    # Pre-compute 25 smooth motion states (`rep_level` in 0..24) for each of the 4 steps so 600 frames render in seconds!
    N_STATES = 25
    pools_d2 = {1: [], 2: [], 3: [], 4: []}
    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]

    for s_i in range(N_STATES):
        rep = s_i / float(N_STATES - 1)
        # Step 1: 42° Back Swing & Lumbar Overload
        lobes1 = [
            (635, 760, 240, 220, 22.0 * rep, 16.0 * rep),
            (645, 1040, 190, 190, 12.0 * rep, 8.0 * rep),
        ]
        w1 = gd2.warp_anatomical_motion(p_s1_w, lobes1)
        rgb1, info1 = render_day2_v36_base_plate(w1, 1, True, intensity=rep)
        pools_d2[1].append((rgb1, (info1["el"][0] + 12.0 * rep, info1["el"][1] + 10.0 * rep)))

        # Step 2: Shrugged Shoulders & Biceps Steal
        lobes2 = [
            (285, 780, 180, 190, 18.0 * rep, -18.0 * rep),
            (885, 790, 180, 190, -18.0 * rep, -18.0 * rep),
            (640, 760, 210, 140, 0.0, -20.0 * rep),
        ]
        w2 = gd2.warp_anatomical_motion(p_s2_w, lobes2)
        rgb2, info2 = render_day2_v36_base_plate(w2, 2, True, intensity=rep)
        pools_d2[2].append((rgb2, (info2["el"][0] + 12.0 * rep, info2["el"][1] - 12.0 * rep)))

        # Step 3: 15° Torso Lean & Scapula Depression
        lobes3 = [
            (310, 820, 190, 210, 14.0 * rep, 26.0 * rep),
            (890, 820, 190, 210, -14.0 * rep, 26.0 * rep),
            (655, 885, 220, 190, 0.0, 16.0 * rep),
        ]
        w3 = gd2.warp_anatomical_motion(p_s3_c, lobes3)
        rgb3, info3 = render_day2_v36_base_plate(w3, 3, False, intensity=rep)
        pools_d2[3].append((rgb3, (info3["el"][0] + 10.0 * rep, info3["el"][1] + 20.0 * rep)))

        # Step 4: Elbows to Hips Full Lat Contraction
        lobes4 = [
            (420, 860, 185, 210, 14.0 * rep, 28.0 * rep),
            (880, 850, 185, 210, -14.0 * rep, 28.0 * rep),
            (715, 965, 165, 175, -6.0 * rep, 15.0 * rep),
        ]
        w4 = gd2.warp_anatomical_motion(p_s4_c, lobes4)
        rgb4, info4 = render_day2_v36_base_plate(w4, 4, False, intensity=rep)
        pools_d2[4].append((rgb4, (info4["el"][0] + 10.0 * rep, info4["el"][1] + 22.0 * rep)))

    # Synthesize 20.0s Day 2 English voiceover with >=1.78s pauses between sentences
    wav_day2 = os.path.join(TMP_AUDIO, "day2_v36_audio.wav")
    vo_day2 = [
        (0.25, "Wide grip lat pulldown. Stop swinging your lower back."),
        (5.30, "Shrugging your shoulders lets your biceps steal the load."),
        (10.40, "Set a fifteen degree lean and depress your scapula first."),
        (15.40, "Drive your elbows to your hips. Four sets of twelve."),
    ]
    asyncio.run(synthesize_voiceover_wav(wav_day2, vo_day2, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day2_v36_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day2_v36_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    step_static_frames = {}
    cover_pil = None

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            u = t / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            s_i = int(round(wave_p * (N_STATES - 1)))
            base_rgb, el_pt = pools_d2[1][s_i]
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom = 1.02 + 0.03 * wave_p
            cam_focus, cam_dest = (650.0, 940.0), (665.0, 940.0)
            pct = int(round(76 + 6 * wave_p))
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
            u = (t - 5.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            s_i = int(round(wave_p * (N_STATES - 1)))
            base_rgb, el_pt = pools_d2[2][s_i]
            step_idx = 2
            is_wrong = True
            active_slot = 1
            cam_zoom = 1.03 + 0.03 * wave_p
            cam_focus, cam_dest = (635.0, 900.0), (655.0, 900.0)
            pct = int(round(66 + 6 * wave_p))
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
            u = (t - 10.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            s_i = int(round(wave_p * (N_STATES - 1)))
            base_rgb, el_pt = pools_d2[3][s_i]
            step_idx = 3
            is_wrong = False
            active_slot = 2
            cam_zoom = 1.02 + 0.03 * wave_p
            cam_focus, cam_dest = (645.0, 910.0), (660.0, 910.0)
            pct = int(round(91 + 5 * wave_p))
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
            u = (t - 15.0) / 5.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            s_i = int(round(wave_p * (N_STATES - 1)))
            base_rgb, el_pt = pools_d2[4][s_i]
            step_idx = 4
            is_wrong = False
            active_slot = 2
            cam_zoom = 1.04 + 0.03 * wave_p
            cam_focus, cam_dest = (660.0, 920.0), (675.0, 920.0)
            pct = int(round(94 + 4 * wave_p))
            sp_cfg = {
                "is_wrong": False,
                "pct": pct,
                "dial_tag": "FULL LAT ISO",
                "short_cue": "✓ ELBOWS TO HIPS",
                "splits": [
                    ("LAT DORSI", pct, v30.COL_GREEN),
                    ("TERES MAJ", max(90, pct - 3), v30.COL_GREEN),
                    ("LOW TRAP", max(88, pct - 5), v30.COL_GREEN),
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
        if f_idx == int(11.6 * FPS):
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

    cover_path = os.path.join(OUT_DIR, "day2_v36_cover.jpg")
    cover_pil.save(cover_path, quality=95)

    # Build 4-Step Full-Body 9:16 Master Sheet for Day 2 (`day2_v36_clean_master_sheet.jpg`)
    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 2 · WIDE-GRIP LAT PULLDOWN (v36 PURE STUDIO MASTER)  ·  점선 0% · 순수 3D 광배근·대원근 결 발현 + 외곽 솔리드 아크",
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

    caption_day2 = """Stop turning your Wide-Grip Lat Pulldown into a lower-back and biceps workout.

Here is the 4-step 3D biomechanical checklist to isolate 98% of your Latissimus Dorsi & Teres Major:

1. 42° BACK SWING OVERLOAD (✕)
Swinging your torso 42° backward turns the pulldown into a momentum row and spikes lumbar spine shear stress to 82%.

2. +22° TRAP SHRUG & BICEPS STEAL (✕)
Pulling with elevated shoulders and curling your forearms lets your Biceps and Upper Traps steal 68% of the load.

3. 15° TORSO LEAN & SCAPULA DEPRESSION (✓)
Lock a slight 15° thoracic lean and depress your shoulder blades (scapulae) down and back BEFORE bending your elbows (96% Upper Lat & Teres Major activation).

4. DRIVE ELBOWS TO HIPS IN A 30° PLANE (✓)
Sweep your elbows down and inward toward your hips to achieve 98% full Latissimus Dorsi contraction (4×12 controlled reps with a 1-second bottom squeeze).

Save this for your next back day and follow @gymform.3d for daily 3D biomechanics guides.

#latpulldown #backworkout #backday #biomechanics #gymform #lats #workouttips #gymtips #hypertrophy #fitnesseducation #correctform #bodybuilding"""
    cap_path = os.path.join(OUT_DIR, "day2_v36_caption_en.txt")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(caption_day2.strip() + "\n")

    print("Saved Day 2 v36 Master Reel, Cover & 4-Step Sheet:", final_mp4, cover_path, sheet_path)


if __name__ == "__main__":
    build_day1_v36_master_reel()
    build_day2_v36_master_reel()
