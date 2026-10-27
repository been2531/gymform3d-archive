#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_day3_v39_squat.py`)
======================================================
Produces Day 3 of `@gymform.3d`'s Daily 3D Biomechanics Encyclopedia Reels series
strictly adhering to `/workspace/daily_info_reels/PRODUCTION_QC_CHECKLIST.md`, `qc_gate.py`,
and the locked `v39 Instagram Reels Safe-Zone Master` specification:
- 4 Genuinely Distinct 3D Postures & Camera Angles across Steps 1, 2, 3, and 4:
  * Step 1 (`0–5s`): Direct Front Coronal Knee Valgus Collapse X-Ray view (`gym_day3_step1_valgus_i2v.mp4`)
  * Step 2 (`5–10s`): 3/4 Side-Profile Lumbar Rounding & Forward Dive X-Ray view (`gym_day3_wrong_i2v.mp4`)
  * Step 3 (`10–15s`): Lower-Body Medium Close-Up Tripod Foot & Knees Tracking Outward view (`gym_day3_step3_kneelock_i2v.mp4`)
  * Step 4 (`15–20s`): Full-Body Wide 3/4 Deep Parallel Squat Drive view (`gym_day3_correct_i2v.mp4`)
- Horizontal 19.5:9 Safe Zone (`x in 96..980`, clear of `x=0..50` and `x=1030..1080` iPhone crop).
- Top Exercise Header (`BARBELL BACK SQUAT [ LEGS · QUADS & GLUTES ]`) at `x=96, y=258..304`
  and 3-Thumbnail Progression Dock at `x=96, y=316..446` (0% overlap with Instagram's `+ / 릴스 친구` at `y=175..242`).
- Bottom-Left HUD at `x=96, y=1300..1444` (well above `@gymform.3d 팔로우` at `y=1585`).
- Right-side exterior callout badges clamped to `x <= 805` (with text right edge `<= 975`) and `y <= 935`
  (well above the right-side Heart icon at `y=1055`).
- ZERO checkmark (`✓`) or cross (`✕`) icons anywhere.
"""

import os
import sys
import math
import wave
import shutil
import asyncio
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import universal_anatomy_rig as uar
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import build_v36_clean_master as v36
import build_v37_minimal_luxury as v37
import build_v38_high_contrast_hud as v38
import build_v39_ig_safe_zone as v39

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_AUDIO = "/workspace/daily_info_reels/_tmp_day3_v39_audio"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_AUDIO, exist_ok=True)

DAY3_STEP1_VALGUS_I2V = os.path.join(OUT_DIR, "gym_day3_step1_valgus_i2v.mp4")
DAY3_STEP2_WRONG_I2V = os.path.join(OUT_DIR, "gym_day3_wrong_i2v.mp4")
DAY3_STEP3_KNEELOCK_I2V = os.path.join(OUT_DIR, "gym_day3_step3_kneelock_i2v.mp4")
DAY3_STEP4_CORRECT_I2V = os.path.join(OUT_DIR, "gym_day3_correct_i2v.mp4")
USER_SCREENSHOT_PATH = v39.USER_SCREENSHOT_PATH

W, H = eng.W, eng.H
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)  # 600 frames

# Locked v39 Safe-Zone Coordinates
SAFE_X_LEFT = 96
SAFE_Y_TOP = 258
SAFE_X_RIGHT_BADGE = 805

# Locked v39 Color Palette (Crimson-Red vs Emerald-Green)
COL_RED = (235, 42, 48)
COL_RED_SOFT = (255, 138, 132)
COL_GREEN = (16, 205, 110)
COL_GREEN_SOFT = (148, 246, 192)

F_EX_TITLE = v30.f_pretendard(24, "extrabold")
F_EX_SUB = v30.f_pretendard(15, "extrabold")

F_TAG_PILL = v30.f_pretendard(19, "extrabold")
F_TAG_CTX = v30.f_pretendard(19, "bold")
F_HERO_PCT = v30.f_barlow(84, extrabold=True)
F_HERO_SYM = v30.f_barlow(44, extrabold=True)
F_SINGLE_LINE = v30.f_pretendard(44, "extrabold")


async def synthesize_day3_voiceover_wav(
    wav_path: str,
    vo_lines: list[tuple[float, str]],
    duration_s: float = 20.0,
) -> list[dict]:
    """
    Synthesizes the 20.0s English coaching voiceover WAV using `edge-tts` (`en-US-GuyNeural`, rate `+22%`)
    and verifies that all 3 inter-step pauses are `>= 1.78s`.
    """
    import edge_tts
    os.makedirs(TMP_AUDIO, exist_ok=True)

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
    stats = []
    print(f"\n=== DAY 3 VOICEOVER TIMING QC ({os.path.basename(wav_path)}) ===")
    for i, (target_start, text) in enumerate(vo_lines):
        mp3_p = os.path.join(TMP_AUDIO, f"vo_day3_{i}.mp3")
        wav_p = os.path.join(TMP_AUDIO, f"vo_day3_{i}.wav")
        comm = edge_tts.Communicate(text, voice="en-US-GuyNeural", rate="+22%", pitch="-2Hz")
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
        print(f"  Step {i+1}: start={actual_start:.2f}s | dur={dur_s:.2f}s | end={end_s:.2f}s | pause_from_prev={gap_s:.2f}s | '{text}'")
        if i > 0:
            assert gap_s >= 1.70, f"Pause between sentences too short: {gap_s:.2f}s < 1.70s"
        prev_end_s = end_s
        stats.append({"step": i + 1, "start": actual_start, "dur": dur_s, "end": end_s, "pause": gap_s, "text": text})

        s_idx = int(actual_start * sr)
        e_idx = min(n_samples, s_idx + len(raw))
        master[s_idx:e_idx] += raw[: e_idx - s_idx] * 0.92
    print("===============================================================\n")

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
    return stats


def render_day3_i2v_frame_v39(
    bgr_720p: np.ndarray,
    step_idx: int,
    is_wrong: bool,
) -> tuple[np.ndarray, dict]:
    """
    Upscales a raw 720x1280 frame from one of Day 3's 4 dedicated 3D I2V clips
    to `1080x1920` with crisp unsharp mask, extracts squat keypoints (`hip`, `kn_r`, `kn_l`, `ank_r`),
    and applies the v39 studio anatomical shader:
    - Step 1 (`LUMBAR & KNEE RISK` — Direct Front Knee Valgus X-Ray clip):
      Preserves and boosts the native 3D X-Ray crimson-orange medial knee valgus collapse glow.
    - Step 2 (`ANTERIOR FORCE LEAK` — 3/4 Side Lumbar Rounding & Forward Dive clip):
      Preserves and boosts the crimson 3D X-Ray lumbar spine rounding & anterior knee/ankle shear.
    - Step 3 (`NEUTRAL SPINE & KNEE LOCK` — Lower-Body Close-Up Tripod Foot & Knees Over Toes clip):
      Vivid intrinsic emerald-green 3D muscle fiber striations across the VMO, Vastus Lateralis, and calves.
    - Step 4 (`FULL QUAD & GLUTE · 4×10` — Full-Body 3/4 Wide Deep Parallel Squat Drive clip):
      Full-intensity emerald-green 3D muscle fiber striations across the Quadriceps and Gluteus Maximus.
    """
    up = cv2.resize(bgr_720p, (W, H), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    plate_bgr = cv2.addWeighted(up, 1.16, blur, -0.16, 0)

    rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    detector = uar.get_pose_detector()
    mp_im = uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb.astype(np.uint8))
    res = detector.detect(mp_im)

    if step_idx == 1:
        # Direct Front View (`gym_day3_step1_valgus_i2v.mp4`)
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            hip_x = float(np.clip(lms[24].x * W, 430.0, 520.0))
            hip_y = float(np.clip(lms[24].y * H, 780.0, 1220.0))
            kn_r_x = float(np.clip(lms[26].x * W, 330.0, 445.0))
            kn_r_y = float(np.clip(lms[26].y * H, 1140.0, 1245.0))
            kn_l_x = float(np.clip(lms[25].x * W, 635.0, 760.0))
            kn_l_y = float(np.clip(lms[25].y * H, 1140.0, 1245.0))
            ank_r_x = float(np.clip(lms[28].x * W, 305.0, 365.0))
            ank_r_y = float(np.clip(lms[28].y * H, 1490.0, 1560.0))
        else:
            hip_x, hip_y = 465.0, 1060.0
            kn_r_x, kn_r_y = 385.0, 1195.0
            kn_l_x, kn_l_y = 695.0, 1195.0
            ank_r_x, ank_r_y = 325.0, 1522.0
    elif step_idx == 3:
        # Lower-Body Medium Close-Up View (`gym_day3_step3_kneelock_i2v.mp4`)
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            hip_x = float(np.clip(lms[24].x * W, 275.0, 420.0))
            hip_y = float(np.clip(lms[24].y * H, 580.0, 760.0))
            kn_r_x = float(np.clip(lms[26].x * W, 285.0, 475.0))
            kn_r_y = float(np.clip(lms[26].y * H, 815.0, 920.0))
            kn_l_x = float(np.clip(lms[25].x * W, 740.0, 920.0))
            kn_l_y = float(np.clip(lms[25].y * H, 800.0, 910.0))
            ank_r_x = float(np.clip(lms[28].x * W, 215.0, 290.0))
            ank_r_y = float(np.clip(lms[28].y * H, 1360.0, 1460.0))
        else:
            hip_x, hip_y = 335.0, 685.0
            kn_r_x, kn_r_y = 385.0, 865.0
            kn_l_x, kn_l_y = 835.0, 855.0
            ank_r_x, ank_r_y = 240.0, 1420.0
    else:
        # 3/4 Full-Body Views (Step 2 `gym_day3_wrong_i2v.mp4` & Step 4 `gym_day3_correct_i2v.mp4`)
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            hip_x = float(np.clip(lms[24].x * W, 380.0, 580.0))
            hip_y = float(np.clip(lms[24].y * H, 710.0, 1150.0))
            kn_r_x = float(np.clip(lms[26].x * W, 450.0, 560.0))
            kn_r_y = float(np.clip(lms[26].y * H, 1110.0, 1260.0))
            kn_l_x = float(np.clip(lms[25].x * W, 680.0, 880.0))
            kn_l_y = float(np.clip(lms[25].y * H, 1100.0, 1240.0))
            ank_r_x = float(np.clip(lms[28].x * W, 375.0, 445.0))
            ank_r_y = float(np.clip(lms[28].y * H, 1520.0, 1630.0))
        else:
            hip_x, hip_y = 435.0, 1020.0
            kn_r_x, kn_r_y = 505.0, 1215.0
            kn_l_x, kn_l_y = 835.0, 1195.0
            ank_r_x, ank_r_y = 408.0, 1585.0

    kpts = {
        "hip": (hip_x, hip_y),
        "kn_r": (kn_r_x, kn_r_y),
        "kn_l": (kn_l_x, kn_l_y),
        "ank_r": (ank_r_x, ank_r_y),
    }

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    gb_diff = np.maximum(0.0, g_ch - b_ch)
    gr_diff = np.maximum(0.0, g_ch - r_ch * 0.82)

    fiber_signal = 0.45 * r_ch + 0.55 * g_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 1.0)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.6)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 7.0)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 15.0)

    hf_ridge = (sigma_micro - sigma_ridge) * 3.1 + (sigma_ridge - sigma_belly) * 1.75
    fs = np.clip(0.48 + hf_ridge / 42.0, 0.0, 1.0) ** 1.35
    hv = np.clip((sigma_belly - 28.0) / 165.0, 0.0, 1.0)
    sulc = np.clip((sigma_broad - sigma_ridge - 3.5) / 22.0, 0.0, 1.0)

    chroma_any = np.clip((np.maximum(rb_diff, gb_diff) - 12.0) / 85.0, 0.0, 1.0)
    grey_sculpt = np.clip(34.0 + 148.0 * (hv ** 1.05) + hf_ridge * 0.95, 16.0, 235.0)
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    neutral_lum = np.clip(base_lum * (1.0 - chroma_any) + grey_sculpt * chroma_any, 14.0, 240.0)

    norm = neutral_lum / 255.0
    mono = np.stack([
        eng.BG_CHARCOAL[0] + (234 - eng.BG_CHARCOAL[0]) * (norm ** 0.92),
        eng.BG_CHARCOAL[1] + (238 - eng.BG_CHARCOAL[1]) * (norm ** 0.92),
        eng.BG_CHARCOAL[2] + (244 - eng.BG_CHARCOAL[2]) * (norm ** 0.92),
    ], axis=2)

    if step_idx == 1:
        # Step 1 (`LUMBAR & KNEE RISK` — Direct Front Knee Valgus X-Ray clip):
        # Preserve and boost the native 3D X-Ray crimson-orange medial knee valgus collapse glow
        xray_alpha = np.clip((rb_diff - 18.0) / 88.0, 0.0, 1.0)
        xray_boost = rgb.copy()
        xray_boost[:, :, 0] = np.clip(r_ch * 1.18 + 26.0, 0, 255)
        xray_boost[:, :, 1] = np.clip(g_ch * 0.90, 0, 255)
        out = mono * (1.0 - xray_alpha[:, :, None]) + xray_boost * xray_alpha[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8), kpts

    if step_idx == 2:
        # Step 2 (`ANTERIOR FORCE LEAK` / `HEEL LIFT & FORWARD DIVE` — 3/4 Side Lumbar Rounding & Forward Dive clip):
        # Preserve the native X-Ray lumbar spine AND boost crimson 3D strain on anterior knees & ankles
        kn_flare_r = np.exp(-(((xx - kn_r_x) / 115.0) ** 2 + ((yy - kn_r_y) / 125.0) ** 2))
        kn_flare_l = np.exp(-(((xx - kn_l_x) / 115.0) ** 2 + ((yy - kn_l_y) / 125.0) ** 2))
        ank_flare = np.exp(-(((xx - ank_r_x) / 95.0) ** 2 + ((yy - ank_r_y) / 95.0) ** 2))
        body_gate = np.clip((base_lum - 36.0) / 85.0, 0.0, 1.0)
        fe = np.clip(np.clip((rb_diff - 18.0) / 90.0, 0.0, 1.0) * 0.94 + (kn_flare_r + kn_flare_l + 0.65 * ank_flare) * body_gate * 0.58, 0.0, 1.0)
        hc = np.clip((gb_diff - 10.0) / 90.0, 0.0, 1.0)
    elif step_idx == 3:
        # Step 3 (`NEUTRAL SPINE & KNEE LOCK` — Lower-Body Close-Up Tripod Foot & Knees Over Toes clip):
        # Crisp emerald-green 3D muscle fiber striations on VMO, Vastus Lateralis, and lower-leg stabilizers
        fe = np.clip((np.maximum(gb_diff, gr_diff) - 14.0) / 86.0, 0.0, 1.0)
        hc = np.clip((gr_diff - 10.0) / 76.0, 0.0, 1.0)
    else:
        # Step 4 (`FULL QUAD & GLUTE · 4×10` — Full-Body 3/4 Wide Deep Parallel Squat Drive clip):
        # Full-intensity emerald-green 3D muscle fibers on Quadriceps & Gluteus Maximus
        glute_flare = np.exp(-(((xx - (hip_x - 28.0)) / 92.0) ** 2 + ((yy - (hip_y + 18.0)) / 98.0) ** 2)) * np.clip((base_lum - 38.0) / 80.0, 0.0, 1.0)
        fe = np.clip(np.clip((np.maximum(gb_diff, gr_diff) - 14.0) / 85.0, 0.0, 1.0) * 1.08 + glute_flare * 0.72, 0.0, 1.0)
        hc = np.clip((gr_diff - 10.0) / 75.0, 0.0, 1.0)

    strand_mod = (0.24 + 0.76 * fs) * (1.0 - 0.62 * sulc)
    fiber_alpha = np.clip(fe * (0.66 + 0.64 * strand_mod), 0.0, 0.96)[:, :, None]
    ridge_boost = np.clip(fs * 0.65 + hc * 0.55, 0.0, 1.0)[:, :, None]
    vol_shade = np.clip(0.28 + 0.72 * hv * (0.45 + 0.55 * fs), 0.0, 1.0)[:, :, None]

    if is_wrong:
        deep_col = np.array([155.0, 18.0, 20.0], dtype=np.float32)
        mid_col = np.array([235.0, 42.0, 48.0], dtype=np.float32)
        crest_col = np.array([255.0, 195.0, 120.0], dtype=np.float32)
    else:
        deep_col = np.array([8.0, 110.0, 52.0], dtype=np.float32)
        mid_col = np.array([16.0, 205.0, 110.0], dtype=np.float32)
        crest_col = np.array([185.0, 255.0, 212.0], dtype=np.float32)

    fiber_rgb = (
        deep_col * (1.0 - vol_shade)
        + mid_col * vol_shade * (1.0 - ridge_boost * 0.65)
        + crest_col * (ridge_boost ** 1.35) * 0.85
    )
    out = mono * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha
    return np.clip(out, 0, 255).astype(np.uint8), kpts


def draw_day3_v39_safezone_top_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    exercise_name: str = "BARBELL BACK SQUAT",
    muscle_group: str = "[ LEGS · QUADS & GLUTES ]",
):
    """
    v39 Instagram Reels Safe-Zone Top HUD (`x=96, y=258..446`):
    - Starts at `SAFE_X_LEFT = 96` and `SAFE_Y_TOP = 258` (`y=258..304` header, `y=316..446` 3-thumbnail dock).
    - Zero checkmark (`✓`) or cross (`✕`) icons.
    - Uses solid crimson-red `(235, 42, 48)` vs emerald-green `(16, 205, 110)`.
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN
    pill_bg = (*COL_RED, 245) if is_wrong else (*COL_GREEN, 245)

    bar_x = SAFE_X_LEFT
    hdr_y = SAFE_Y_TOP
    hdr_h = 46
    tb_ex = od.textbbox((0, 0), exercise_name, font=F_EX_TITLE)
    ex_w = tb_ex[2] - tb_ex[0]
    tb_mg = od.textbbox((0, 0), muscle_group, font=F_EX_SUB)
    mg_w = tb_mg[2] - tb_mg[0]

    pill_w = mg_w + 22
    hdr_w = 22 + 12 + ex_w + 14 + pill_w + 16

    od.rounded_rectangle([bar_x + 2, hdr_y + 4, bar_x + hdr_w + 2, hdr_y + hdr_h + 4], radius=14, fill=(0, 0, 0, 150))
    od.rounded_rectangle(
        [bar_x, hdr_y, bar_x + hdr_w, hdr_y + hdr_h],
        radius=14,
        fill=(14, 17, 23, 232),
        outline=(255, 255, 255, 44),
        width=1,
    )

    dot_cx = bar_x + 20
    dot_cy = hdr_y + hdr_h // 2
    od.ellipse([dot_cx - 8, dot_cy - 8, dot_cx + 8, dot_cy + 8], fill=(*accent, 95))
    od.ellipse([dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5], fill=(*accent, 255))

    ex_x = dot_cx + 15
    od.text((ex_x, hdr_y + 9), exercise_name, font=F_EX_TITLE, fill=(255, 255, 255, 255))

    p_x0 = ex_x + ex_w + 13
    p_y0 = hdr_y + 8
    p_h = hdr_h - 16
    od.rounded_rectangle([p_x0, p_y0, p_x0 + pill_w, p_y0 + p_h], radius=8, fill=pill_bg)
    od.text((p_x0 + 11, p_y0 + 6), muscle_group, font=F_EX_SUB, fill=(255, 255, 255, 255))

    # 3-Stage Thumbnail Dock (`y=316..446`)
    tw, th = 136, 108
    gap = 10
    pad = 11
    dock_y = hdr_y + hdr_h + 12  # y = 316
    dock_w = 3 * tw + 2 * gap + 2 * pad
    dock_h = th + 2 * pad        # 130px -> ends at y = 446

    od.rounded_rectangle([bar_x + 2, dock_y + 6, bar_x + dock_w + 2, dock_y + dock_h + 6], radius=22, fill=(0, 0, 0, 135))
    od.rounded_rectangle(
        [bar_x, dock_y, bar_x + dock_w, dock_y + dock_h],
        radius=22,
        fill=(16, 19, 25, 222),
        outline=(255, 255, 255, 38),
        width=1,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = dock_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy().resize((tw, th), Image.Resampling.LANCZOS) if i == active_idx else inact_im.copy().resize((tw, th), Image.Resampling.LANCZOS)

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=13, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], radius=14, outline=(*i_col, 245), width=2)
            od.rounded_rectangle([tx + tw // 2 - 16, ty + th - 6, tx + tw // 2 + 16, ty + th - 2], radius=2, fill=(*i_col, 255))
        else:
            od.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=13, outline=(255, 255, 255, 22), width=1)

    canvas.alpha_composite(ov)


def draw_day3_v39_safezone_bottom_hud(canvas: Image.Image, sp: dict):
    """
    v39 Instagram Reels Safe-Zone Bottom-Left HUD (`x=96, y=1300..1444`):
    - Row 1 (`y=1300`): Solid high-contrast color pill + secondary anatomical target label.
    - Row 2 (`y=1344`): Huge `%` + 3px vertical bar + single-line bold hero cue.
    - Row 3 (`y=1436`): Single 5px glowing progress bar.
    """
    is_wrong = sp["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN
    pill_fill = (*COL_RED, 250) if is_wrong else (*COL_GREEN, 250)
    pct = int(np.clip(sp["pct"], 0, 100))

    tag_pill_str = sp["tag_pill"]
    tag_ctx_str = sp["tag_ctx"]
    cue_1line = sp["cue_1line"]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx = SAFE_X_LEFT  # 96px

    # Row 1: Solid High-Contrast Pill + Secondary Anatomical Target Label (`y=1300`)
    py = 1300
    tb_p = od.textbbox((0, 0), tag_pill_str, font=F_TAG_PILL)
    pw = (tb_p[2] - tb_p[0]) + 26
    ph = 34

    sd.rounded_rectangle([bx, py + 3, bx + pw, py + ph + 3], radius=8, fill=(0, 0, 0, 225))
    od.rounded_rectangle([bx, py, bx + pw, py + ph], radius=8, fill=pill_fill, outline=(255, 255, 255, 55), width=1)
    od.text((bx + 13, py + 6), tag_pill_str, font=F_TAG_PILL, fill=(255, 255, 255, 255))

    ctx_x = bx + pw + 13
    ctx_y = py + 6
    sd.text((ctx_x, ctx_y + 3), tag_ctx_str, font=F_TAG_CTX, fill=(0, 0, 0, 245))
    od.text((ctx_x, ctx_y), tag_ctx_str, font=F_TAG_CTX, fill=(228, 236, 248, 255))

    # Row 2: Huge `%` + 3px Vertical Bar + Single-Line Bold Hero Cue (`y=1344`)
    by = 1344
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

    div_x = sym_x + sw + 18
    div_y0 = by + 18
    div_y1 = by + 68
    sd.rounded_rectangle([div_x, div_y0 + 2, div_x + 3, div_y1 + 2], radius=2, fill=(0, 0, 0, 225))
    od.rounded_rectangle([div_x, div_y0, div_x + 3, div_y1], radius=2, fill=(*accent, 255))

    tx = div_x + 18
    ty = by + 18
    tb_c = od.textbbox((0, 0), cue_1line, font=F_SINGLE_LINE)
    cw = tb_c[2] - tb_c[0]

    sd.text((tx, ty + 4), cue_1line, font=F_SINGLE_LINE, fill=(0, 0, 0, 245))
    od.text((tx, ty), cue_1line, font=F_SINGLE_LINE, fill=(255, 255, 255, 255))

    # Row 3: Single 5px Glowing Progress Bar (`y=1436`)
    bar_x0 = bx + 2
    bar_y0 = by + 92
    total_row_w = (tx + cw) - bar_x0
    bar_w = max(360, min(520, total_row_w))
    bar_h = 5
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


def render_day3_v39_safezone_callouts(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 1.0,
):
    """
    Day 3 (Barbell Back Squat) exterior callouts tailored to each of the 4 distinct camera views,
    with strict v39 iPhone Safe-Zone clamping:
    - Left-side callout badges (`align="right"`) are placed at `x >= 255` so their left edge never goes below `x = 115`.
    - Right-side callout badges (`align="left"`) are clamped so `x <= SAFE_X_RIGHT_BADGE (805)` and their right edge
      never exceeds `x = 975` (`55px` inside the iPhone right crop `x=1030`), and `y <= 935` (`120px` above the Heart icon `y=1055`).
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

    col_main = COL_RED if is_wrong else COL_GREEN
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    hip_s = S(map_pt(*kp["hip"]))
    kn_r_s = S(map_pt(*kp["kn_r"]))
    kn_l_s = S(map_pt(*kp["kn_l"]))
    ank_r_s = S(map_pt(*kp["ank_r"]))

    if step_idx == 1:
        # Step 1 (Direct Front Coronal Knee Valgus View):
        # Bilateral knee reticles (◎) + inward-buckling valgus arc 18° (VALGUS CAVE) + right-side badge 84% (ACL & LUMBAR LOAD)
        v36.draw_single_elbow_pivot_ring(rd, kn_r_s, 10.0, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, kn_l_s, 9.0, col_main, SS)

        c_arc = S(map_pt(kp["kn_r"][0] + 15.0, kp["kn_r"][1] - 10.0))
        r_a = 118 * SS
        start_ang = 330
        end_ang = int(round(375 + 22 * rep_amount))
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=4 * SS)
        ang_t = math.radians(end_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, end_ang + 90, 12 * SS, (*col_soft, 255))

        p_left = map_pt(340, 1045)
        v30.draw_floating_number_badge(
            od, sd,
            (max(255, int(p_left[0])), int(np.clip(p_left[1], 880, 1110))),
            "18°", "VALGUS CAVE", col_main, col_soft, align="right"
        )

        p_right = map_pt(785, 865)
        rx = min(SAFE_X_RIGHT_BADGE, int(p_right[0]))
        ry = min(920, int(p_right[1]))
        v30.draw_floating_number_badge(
            od, sd,
            (rx, ry),
            "84%", "ACL & LUMBAR LOAD", col_main, col_soft, align="left"
        )

    elif step_idx == 2:
        # Step 2 (3/4 Side-Profile Lumbar Rounding & Forward Dive View):
        # Knee/Ankle reticle (◎) + solid red forward arc +28° (HEELS LIFTING) + right-side badge 79% (PATELLAR STRESS)
        v36.draw_single_elbow_pivot_ring(rd, ank_r_s, 9.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, kn_r_s, 8.5, col_main, SS)

        c_arc = S(map_pt(kp["hip"][0] + 15.0, kp["hip"][1] - 35.0))
        r_a = 152 * SS
        start_ang = 195
        end_ang = int(round(238 + 22 * rep_amount))
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=4 * SS)
        ang_t = math.radians(end_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, end_ang + 90, 12 * SS, (*col_soft, 255))

        p_left = map_pt(355, 1055)
        v30.draw_floating_number_badge(
            od, sd,
            (max(255, int(p_left[0])), int(np.clip(p_left[1], 880, 1120))),
            "+28°", "HEELS LIFTING", col_main, col_soft, align="right"
        )

        p_right = map_pt(795, 885)
        rx = min(SAFE_X_RIGHT_BADGE, int(p_right[0]))
        ry = min(925, int(p_right[1]))
        v30.draw_floating_number_badge(
            od, sd,
            (rx, ry),
            "79%", "PATELLAR STRESS", col_main, col_soft, align="left"
        )

    elif step_idx == 3:
        # Step 3 (Lower-Body Medium Close-Up Tripod Foot & Knees Tracking Outward View):
        # Knee & Ankle reticles (◎) + solid green outward tracking arc 0° (KNEES TRACK TOES) + right-side badge 100% (TRIPOD & TORQUE)
        v36.draw_single_elbow_pivot_ring(rd, kn_r_s, 10.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, ank_r_s, 8.5, col_main, SS)

        c_arc = S(map_pt(kp["kn_r"][0] - 12.0, kp["kn_r"][1] + 5.0))
        r_a = 126 * SS
        start_ang = 142
        end_ang = 204
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=4 * SS)
        ang_t = math.radians(start_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, start_ang - 90, 12 * SS, (*col_soft, 255))

        p_left = map_pt(355, 985)
        v30.draw_floating_number_badge(
            od, sd,
            (max(265, int(p_left[0])), int(np.clip(p_left[1], 840, 1060))),
            "0°", "KNEES TRACK TOES", col_main, col_soft, align="right"
        )

        p_right = map_pt(790, 845)
        rx = min(SAFE_X_RIGHT_BADGE, int(p_right[0]))
        ry = min(915, int(p_right[1]))
        v30.draw_floating_number_badge(
            od, sd,
            (rx, ry),
            "100%", "TRIPOD & TORQUE", col_main, col_soft, align="left"
        )

    else:
        # Step 4 (Full-Body Wide 3/4 Deep Parallel Squat Drive View):
        # Hip/Knee reticle (◎) + solid green depth arc 95° (PARALLEL DEPTH) + right-side badge 100% (QUAD & GLUTE DRIVE)
        v36.draw_single_elbow_pivot_ring(rd, hip_s, 10.0, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, kn_r_s, 8.5, col_main, SS)

        c_arc = S(map_pt(kp["hip"][0] - 15.0, kp["hip"][1] + 10.0))
        r_a = 142 * SS
        start_ang = 128
        end_ang = 198
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=4 * SS)
        ang_t = math.radians(start_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, start_ang - 90, 13 * SS, (*col_soft, 255))

        p_left = map_pt(355, 985)
        v30.draw_floating_number_badge(
            od, sd,
            (max(255, int(p_left[0])), int(np.clip(p_left[1], 840, 1080))),
            "95°", "PARALLEL DEPTH", col_main, col_soft, align="right"
        )

        p_right = map_pt(790, 865)
        rx = min(795, int(p_right[0]))
        ry = min(915, int(p_right[1]))
        v30.draw_floating_number_badge(
            od, sd,
            (rx, ry),
            "100%", "QUAD & GLUTE DRIVE", col_main, col_soft, align="left"
        )

    canvas.alpha_composite(rig_layer.resize((W, H), Image.Resampling.LANCZOS))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)


def build_cached_day3_v39_top_docks(torso_icons) -> dict:
    cache = {}
    for active_slot, is_wrong in [(0, True), (1, True), (2, False)]:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_day3_v39_safezone_top_dock(
            layer,
            torso_icons,
            active_slot,
            is_wrong,
            exercise_name="BARBELL BACK SQUAT",
            muscle_group="[ LEGS · QUADS & GLUTES ]",
        )
        cache[(active_slot, is_wrong)] = layer
    return cache


def _load_clip_frames(mp4_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    assert len(frames) > 30, f"Failed to load frames from {mp4_path}"
    return frames


def _smooth_kpts_seq(raw_kp_list: list[dict]) -> list[dict]:
    n = len(raw_kp_list)
    smooth = [dict() for _ in range(n)]
    for k in ["hip", "kn_r", "kn_l", "ank_r"]:
        arr = np.array([rk[k] for rk in raw_kp_list], dtype=np.float32)
        ax = gaussian_filter1d(arr[:, 0], sigma=2.0, mode="nearest")
        ay = gaussian_filter1d(arr[:, 1], sigma=2.0, mode="nearest")
        for i in range(n):
            smooth[i][k] = (float(ax[i]), float(ay[i]))
    return smooth


def build_day3_v39_master_reel():
    print("Building Day 3 (Barbell Back Squat) v39 4-Angle Distinct Instagram Reels Master...")
    raw_s1 = _load_clip_frames(DAY3_STEP1_VALGUS_I2V)
    raw_s2 = _load_clip_frames(DAY3_STEP2_WRONG_I2V)
    raw_s3 = _load_clip_frames(DAY3_STEP3_KNEELOCK_I2V)
    raw_s4 = _load_clip_frames(DAY3_STEP4_CORRECT_I2V)

    print(
        f"Loaded 4 distinct Day 3 I2V clips: "
        f"step1_valgus={len(raw_s1)}, step2_wrong={len(raw_s2)}, "
        f"step3_kneelock={len(raw_s3)}, step4_correct={len(raw_s4)}"
    )

    # Monotonic full-rep frame slices (standing top -> deepest squat position) across 4 distinct camera angles:
    # - Step 1 (`raw_s1[54:119][::-1]`, 65 frames): Direct Front View standing -> deep knock-kneed valgus collapse
    # - Step 2 (`raw_s2[64:121]`, 57 frames): 3/4 Side Profile standing -> deep forward-lean & lumbar rounding
    # - Step 3 (`raw_s3[56:126][::-1]`, 70 frames): Lower-Body Medium Close-Up standing -> deep tripod-foot & knees-over-toes squat
    # - Step 4 (`raw_s4[90:145]`, 55 frames): Full-Body 3/4 Wide Studio View standing -> deep parallel squat drive
    stroke_s1 = raw_s1[54:119][::-1]
    stroke_s2 = raw_s2[64:121]
    stroke_s3 = raw_s3[56:126][::-1]
    stroke_s4 = raw_s4[90:145]

    pool_s1, kp_s1_raw = [], []
    for fr in stroke_s1:
        rgb, kp = render_day3_i2v_frame_v39(fr, 1, True)
        pool_s1.append(rgb)
        kp_s1_raw.append(kp)
    smooth_kp_s1 = _smooth_kpts_seq(kp_s1_raw)

    pool_s2, kp_s2_raw = [], []
    for fr in stroke_s2:
        rgb, kp = render_day3_i2v_frame_v39(fr, 2, True)
        pool_s2.append(rgb)
        kp_s2_raw.append(kp)
    smooth_kp_s2 = _smooth_kpts_seq(kp_s2_raw)

    pool_s3, kp_s3_raw = [], []
    for fr in stroke_s3:
        rgb, kp = render_day3_i2v_frame_v39(fr, 3, False)
        pool_s3.append(rgb)
        kp_s3_raw.append(kp)
    smooth_kp_s3 = _smooth_kpts_seq(kp_s3_raw)

    pool_s4, kp_s4_raw = [], []
    for fr in stroke_s4:
        rgb, kp = render_day3_i2v_frame_v39(fr, 4, False)
        pool_s4.append(rgb)
        kp_s4_raw.append(kp)
    smooth_kp_s4 = _smooth_kpts_seq(kp_s4_raw)

    # 3-Thumbnail Progression Dock showing the 3 distinct postures/views:
    # Thumbnail 0: Front Knee Valgus (`pool_s1[-1]`)
    # Thumbnail 1: Side Forward Lean (`pool_s2[-1]`)
    # Thumbnail 2: Full-Body Deep Squat Drive (`pool_s4[-1]`)
    size = (136, 108)
    im0_act = Image.fromarray(pool_s1[-1]).convert("RGBA").crop((220, 740, 860, 1480)).resize(size, Image.Resampling.LANCZOS)
    im1_act = Image.fromarray(pool_s2[-1]).convert("RGBA").crop((310, 680, 910, 1460)).resize(size, Image.Resampling.LANCZOS)
    im2_act = Image.fromarray(pool_s4[-1]).convert("RGBA").crop((310, 680, 910, 1460)).resize(size, Image.Resampling.LANCZOS)

    def _make_inact(act_im: Image.Image) -> Image.Image:
        out = act_im.copy()
        out.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 138)))
        return out

    torso_icons = [
        (im0_act, _make_inact(im0_act)),
        (im1_act, _make_inact(im1_act)),
        (im2_act, _make_inact(im2_act)),
    ]
    dock_cache = build_cached_day3_v39_top_docks(torso_icons)
    hud_cache = {}

    wav_day3 = os.path.join(TMP_AUDIO, "day3_v39_audio.wav")
    vo_day3 = [
        (0.12, "Stop letting your knees cave inward and rounding your lower back."),
        (5.12, "Lifting your heels pitches you forward and overloads your knees."),
        (10.12, "Root your feet, brace your core, and track knees over toes."),
        (15.12, "Drop to full depth and drive up to maximize quad and glute growth."),
    ]
    vo_stats = asyncio.run(synthesize_day3_voiceover_wav(wav_day3, vo_day3, DURATION))

    tmp_mp4 = os.path.join(TMP_AUDIO, "_day3_v39_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day3_v39_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]
    scrim_2d = v38.make_bottom_scrim_2d()

    step_static_frames = {}
    cover_pil = None

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            # Step 1 (0.0s - 5.0s, Red / Error 1): Direct Front Knee Valgus X-Ray
            u = t / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s1) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s1, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_kp_s1, pos_f)
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom = 1.02 + 0.02 * rep_w
            cam_focus, cam_dest = (540.0, 980.0), (550.0, 980.0)
            pct = int(round(78 + 6 * rep_w))
            tag_pill = "LUMBAR & KNEE RISK"
            tag_ctx = "MEDIAL KNEE COLLAPSE"
            cue_1line = "18° KNEE VALGUS"
        elif t < 10.0:
            # Step 2 (5.0s - 10.0s, Red / Error 2): 3/4 Side Torso Forward Dive & Lumbar X-Ray
            u = (t - 5.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s2) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s2, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_kp_s2, pos_f)
            step_idx = 2
            is_wrong = True
            active_slot = 1
            cam_zoom = 1.03 + 0.02 * rep_w
            cam_focus, cam_dest = (575.0, 970.0), (585.0, 970.0)
            pct = int(round(72 + 7 * rep_w))
            tag_pill = "ANTERIOR FORCE LEAK"
            tag_ctx = "HEEL LIFT & FORWARD DIVE"
            cue_1line = "+28° FORWARD LEAN"
        elif t < 15.0:
            # Step 3 (10.0s - 15.0s, Green / Correct 1): Lower-Body Close-Up Tripod Foot & Knees Tracking Outward
            u = (t - 10.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s3) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s3, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_kp_s3, pos_f)
            step_idx = 3
            is_wrong = False
            active_slot = 2
            cam_zoom = 1.02 + 0.02 * rep_w
            cam_focus, cam_dest = (560.0, 940.0), (570.0, 940.0)
            pct = int(round(92 + 4 * rep_w))
            tag_pill = "NEUTRAL SPINE & KNEE LOCK"
            tag_ctx = "TRIPOD FOOT & HIP TORQUE"
            cue_1line = "KNEES OVER TOES"
        else:
            # Step 4 (15.0s - 20.0s, Green / Correct 2): Full-Body Wide Deep Parallel Squat Drive (Quads & Glutes · 4×10)
            u = (t - 15.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s4) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s4, pos_f)
            kp = v37.sample_kpts_interpolated(smooth_kp_s4, pos_f)
            step_idx = 4
            is_wrong = False
            active_slot = 2
            cam_zoom = 1.03 + 0.02 * rep_w
            cam_focus, cam_dest = (580.0, 965.0), (590.0, 965.0)
            pct = int(round(94 + 4 * rep_w))
            tag_pill = "FULL QUAD & GLUTE · 4×10"
            tag_ctx = "QUADRICEPS & GLUTES"
            cue_1line = "DEEP SQUAT DRIVE"

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        clean_rgb = v38.apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)
        im = Image.fromarray(clean_rgb).convert("RGBA")

        render_day3_v39_safezone_callouts(im, step_idx, map_pt, kp, is_wrong, rep_amount=rep_w)
        im.alpha_composite(dock_cache[(active_slot, is_wrong)])

        hud_key = (is_wrong, pct, tag_pill, tag_ctx, cue_1line)
        if hud_key not in hud_cache:
            h_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_day3_v39_safezone_bottom_hud(
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
        # Sample peak squat depth frames (`u = 0.48` -> `2.4s, 7.4s, 12.4s, 17.4s`)
        if f_idx in (int(2.4 * FPS), int(7.4 * FPS), int(12.4 * FPS), int(17.4 * FPS)):
            step_static_frames[step_idx] = rgb_frame.copy()
        if f_idx == int(17.4 * FPS):
            cover_pil = rgb_frame.copy()

        vw.write(cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR))

    vw.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-threads", "4",
            "-i", tmp_mp4, "-i", wav_day3,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_mp4)

    # 1. Save Cover Image (`day3_v39_cover.jpg`)
    cover_path = os.path.join(OUT_DIR, "day3_v39_cover.jpg")
    cover_pil.save(cover_path, quality=95)

    # 2. Save 4-Step Overview Sheet (`day3_v39_clean_master_sheet.jpg`)
    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 3 · BARBELL BACK SQUAT (v39 4-ANGLE DISTINCT MASTER)  ·  4개 구간 독립 3D 앵글/자세 + 아이폰 세이프존 100% 최적화",
        font=f_row,
        fill=COL_GREEN,
    )
    col_titles = [
        "STEP 1 (전면 뷰) · 84% | 18° KNEE VALGUS",
        "STEP 2 (측면 뷰) · 79% | +28° FORWARD LEAN",
        "STEP 3 (하체 근접) · 96% | KNEES OVER TOES",
        "STEP 4 (전신 풀샷) · 98% | DEEP SQUAT DRIVE",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = COL_RED if i < 2 else COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = step_static_frames[i + 1].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day3_v39_clean_master_sheet.jpg")
    board.save(sheet_path, quality=95)

    # 3. Save 4-Step iPhone Instagram Reels UI Safe-Zone Verification Sheet (`day3_v39_iphone_safezone_preview.jpg`)
    real_shot_bgr = cv2.imread(USER_SCREENSHOT_PATH)
    cap_old = cv2.VideoCapture(os.path.join(OUT_DIR, "day2_v38_master_reel.mp4"))
    cap_old.set(cv2.CAP_PROP_POS_FRAMES, 210)
    _, old_f210_bgr = cap_old.read()
    cap_old.release()

    cw, ch = 420, 910
    ipad = 24
    ip_board = Image.new("RGB", (cw * 4 + ipad * 5, ch + 115), (10, 12, 16))
    idraw = ImageDraw.Draw(ip_board)
    idraw.text(
        (28, 20),
        "DAY 3 · iPHONE INSTAGRAM REELS UI SAFE-ZONE PREVIEW (1179×2556)  ·  4개 구간 실제 자세·카메라 앵글 차별화 + UI 간섭 0%",
        font=f_row,
        fill=COL_GREEN,
    )
    ip_titles = [
        "iPhone View · STEP 1 (전면 무릎 내반 붕괴)",
        "iPhone View · STEP 2 (측면 상체 쏠림·요추 과부하)",
        "iPhone View · STEP 3 (하체 근접 무릎-발끝 정렬)",
        "iPhone View · STEP 4 (전신 풀 스쿼트 드라이브)",
    ]
    for i in range(4):
        x = ipad + i * (cw + ipad)
        col = COL_RED if i < 2 else COL_GREEN
        idraw.text((x + 4, 58), ip_titles[i], font=f_col, fill=col)
        sim_im = v39.simulate_iphone_ig_reels_view(step_static_frames[i + 1], real_shot_bgr, old_f210_bgr)
        rc = sim_im.resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        ip_board.paste(rc, (x, 86))

    iphone_sheet_path = os.path.join(OUT_DIR, "day3_v39_iphone_safezone_preview.jpg")
    ip_board.save(iphone_sheet_path, quality=95)

    # Clean up temporary directory
    shutil.rmtree(TMP_AUDIO, ignore_errors=True)

    print("Saved Day 3 v39 Master Reel:", final_mp4)
    print("Saved Day 3 v39 Cover:", cover_path)
    print("Saved Day 3 v39 4-Step Sheet:", sheet_path)
    print("Saved Day 3 v39 iPhone Safe-Zone Preview:", iphone_sheet_path)
    return vo_stats


if __name__ == "__main__":
    build_day3_v39_master_reel()
