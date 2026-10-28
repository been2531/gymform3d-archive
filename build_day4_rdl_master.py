#!/usr/bin/env python3
"""
Day 4 (`Romanian Deadlift`) — 100% Native 3D Multi-Angle 4-Step Master Reel Builder (Low-RAM Streaming)
Applies confirmed V3 + Option 3 standard:
1. Top 4-slot preview bar in 3:4 profile grid safe zone (y = 248..406) with Translucent Glass See-Through.
2. 'Romanian Deadlift' Kaushan Script title on the studio back wall behind the 3D mannequin (y = 458).
3. Zero bottom text / zero 2D overlays / zero frame-blending ghosting / full-body 0.90x scale.
4. Step 4 Rear 3/4 View (97% Hamstring & Glute) as the dynamic 3:4 grid cover.
"""
import os
import sys
import gc
import math
import pickle
import asyncio
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62
import build_clean_d4_variants as bcd
import build_v36_publish_reels as v36_pub
import build_4step_bone_muscle_master as b4

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_DIR = "/workspace/daily_info_reels/_tmp_day4_master"
CACHE_KP_PKL = "/workspace/daily_info_reels/_cache_day4_rdl_ecorche_720p_kps.pkl"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)  # 600 frames

STEP_SPECS_DAY4 = {
    1: ("day4_rdl_step1_ecorche_720p.mp4", 57, 4, True,  "bone_blood",   95),
    2: ("day4_rdl_step2_ecorche_720p.mp4", 39, 0, True,  "muscle_blood", 95),
    3: ("day4_rdl_step3_ecorche_720p.mp4", 46, 6, False, "bone_blood",   95),
    4: ("day4_rdl_step4_ecorche_720p.mp4", 59, 5, False, "muscle_blood", 95),
}

b4.STEP_DY = {1: 95, 2: 95, 3: 95, 4: 95}


def load_native_slice_720p(path: str, i0: int, i1: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frs = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if fr.shape[1] != 720 or fr.shape[0] != 1280:
            fr = cv2.resize(fr, (720, 1280), interpolation=cv2.INTER_LANCZOS4)
        frs.append(fr)
    cap.release()
    return frs[i0 : i1 + 1] if i0 <= i1 else frs[i1 : i0 + 1][::-1]


def shade_day4_3d_frame_1080p(
    rgb_1080: np.ndarray,
    step_idx: int,
    is_wrong: bool,
    kp: dict,
    pulse_phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r_ch = rgb_1080[:, :, 0].astype(np.float32)
    g_ch = rgb_1080[:, :, 1].astype(np.float32)
    b_ch = rgb_1080[:, :, 2].astype(np.float32)
    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    min_c = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    sat = (max_c - min_c) / (max_c + 1.0)

    red_dom = np.clip(r_ch - 0.50 * (g_ch + b_ch), 0.0, 255.0)
    grn_dom = np.clip(g_ch - 0.50 * (r_ch + b_ch), 0.0, 255.0)
    act_dom = np.maximum(red_dom, grn_dom)

    bg_med = np.median(
        np.concatenate([rgb_1080[:24, :, :].reshape(-1, 3), rgb_1080[:, :24, :].reshape(-1, 3), rgb_1080[:, -24:, :].reshape(-1, 3)], axis=0),
        axis=0,
    )
    bg_lum = float(0.299 * bg_med[0] + 0.587 * bg_med[1] + 0.114 * bg_med[2])

    fg_diff = np.abs(lum - bg_lum) + 0.95 * act_dom
    w_fg = cv2.GaussianBlur(np.clip((fg_diff - 6.0) / 14.0, 0.0, 1.0).astype(np.float32), (0, 0), 2.0)

    # 1. Unified Upper-Back Sculpted Grey Base (grey_back_exact) across entire body
    hv_cool = np.clip((lum - 18.0) / 172.0, 0.0, 1.35)
    blur_med = cv2.GaussianBlur(lum, (0, 0), 2.8)
    sulc = np.clip((blur_med - lum) / 14.0, 0.0, 1.0)
    hf_ridge = lum - cv2.GaussianBlur(lum, (0, 0), 1.6)
    grey_back_exact = np.clip(32.0 + 108.0 * (hv_cool ** 0.95) + hf_ridge * 0.55 - sulc * 14.0, 20.0, 168.0)

    yy = np.arange(H, dtype=np.float32)[:, None]
    hip_y = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1])
    kn_y = 0.5 * (kp["kn_r"][1] + kp["kn_l"][1])

    shorts_band = np.clip((yy - (hip_y - 145.0)) / 28.0, 0.0, 1.0) * np.clip(((kn_y - 30.0) - yy) / 38.0, 0.0, 1.0)
    dark_low_sat = np.clip((64.0 - lum) / 20.0, 0.0, 1.0) * np.clip((0.26 - sat) / 0.14, 0.0, 1.0) * np.clip((18.0 - act_dom) / 12.0, 0.0, 1.0)
    shorts_mask_1080 = cv2.GaussianBlur((shorts_band * dark_low_sat * w_fg).astype(np.float32), (0, 0), 2.0)

    # Crisp opaque matte jet-black drawers (@demicstory standard: lum ~ 14..22)
    shorts_jet_lum = np.clip(lum * 0.46 + 3.0, 10.0, 24.0)
    matte_lum = grey_back_exact * (1.0 - 0.88 * shorts_mask_1080) + shorts_jet_lum * (0.88 * shorts_mask_1080)

    mono_r = 38.0 * (1.0 - w_fg) + (matte_lum - 1.5) * w_fg
    mono_g = 40.0 * (1.0 - w_fg) + (matte_lum + 0.5) * w_fg
    mono_b = 44.0 * (1.0 - w_fg) + (matte_lum + 3.0) * w_fg
    mono = np.stack([mono_r, mono_g, mono_b], axis=-1)

    # 2. Option B (@demicstory Core-to-Grey Edge Falloff + 100% Striation Preservation)
    w_act = cv2.GaussianBlur((np.clip((act_dom - 10.0) / 36.0, 0.0, 1.0) ** 0.85).astype(np.float32), (0, 0), 1.8) * (1.0 - 0.90 * shorts_mask_1080)
    core_seed = np.clip((act_dom - 22.0) / 50.0, 0.0, 1.0) ** 1.12
    pulse = 1.0 + 0.06 * math.sin(2.0 * math.pi * pulse_phase)
    core_w = np.clip(cv2.GaussianBlur(core_seed.astype(np.float32), (0, 0), 7.5) * w_act * 0.80 * pulse, 0.0, 0.84)

    L = np.clip((grey_back_exact - 20.0) / 148.0, 0.0, 1.0)
    bone_core = np.clip((lum - 155.0) / 55.0, 0.0, 1.0) * np.clip((act_dom - 14.0) / 28.0, 0.0, 1.0) if step_idx in (1, 3) else 0.0

    if is_wrong:
        col_r = np.clip(grey_back_exact * 1.36 + 26.0 * (L ** 0.90) + 38.0 * bone_core, 16.0, 224.0)
        col_g = np.clip(grey_back_exact * 0.42 * ((1.0 - 0.45 * sulc)) + 45.0 * bone_core, 10.0, 130.0)
        col_b = np.clip(grey_back_exact * 0.44 * ((1.0 - 0.45 * sulc)) + 35.0 * bone_core, 12.0, 120.0)
    else:
        col_g = np.clip(grey_back_exact * 1.32 + 22.0 * (L ** 0.90) + 38.0 * bone_core, 16.0, 220.0)
        col_r = np.clip(grey_back_exact * 0.42 * ((1.0 - 0.45 * sulc)) + 32.0 * bone_core, 10.0, 120.0)
        col_b = np.clip(grey_back_exact * 0.74 * ((1.0 - 0.30 * sulc)) + 42.0 * bone_core, 14.0, 165.0)

    act_rgb = np.stack([col_r, col_g, col_b], axis=-1)
    blend_w = np.clip(core_w + 0.35 * bone_core, 0.0, 0.90)
    out = mono * (1.0 - blend_w[:, :, None]) + act_rgb * blend_w[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8), shorts_mask_1080


def draw_day4_laser_and_callout(
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
    hip_r = kp["hip_r"]
    kn_r = kp["kn_r"]

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

    if step_idx in (1, 2, 3):
        target_pt = (int(0.48 * sh[0] + 0.52 * hip_r[0]), int(0.48 * sh[1] + 0.52 * hip_r[1]))
    else:
        target_pt = (int(0.48 * hip_r[0] + 0.52 * kn_r[0]), int(0.48 * hip_r[1] + 0.52 * kn_r[1]))

    laser_rgb = np.array(canvas.convert("RGB")).astype(np.float32)
    sh_gate = np.clip(shorts_mask_1080 * 0.85, 0.0, 0.85)[:, :, None]
    blended = np.clip(laser_rgb * (1.0 - sh_gate) + base_rgb.astype(np.float32) * sh_gate, 0, 255).astype(np.uint8)

    base_pil = Image.fromarray(blended).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    elbow_x = max(286, target_pt[0] - 42)
    elbow_y = target_pt[1] - 24
    bcd._draw_precision_leader(rd, target_pt, (elbow_x, elbow_y), (278, elbow_y), col)

    tx = 86
    ty = int(np.clip(elbow_y - 76, 520, 1460))
    f_big = v30.f_pretendard(86, "extrabold")
    f_l1 = v30.f_pretendard(21, "extrabold")
    f_l2 = v30.f_pretendard(16, "bold")

    rd.text((tx + 2, ty + 3), deg_str, font=f_big, fill=(0, 0, 0, 210))
    rd.text((tx, ty), deg_str, font=f_big, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 92), l1_str, font=f_l1, fill=(col[0], col[1], col[2], 255))
    rd.text((tx + 2, ty + 119), l2_str, font=f_l2, fill=(215, 222, 232, 235))

    return np.array(Image.alpha_composite(base_pil, ov).convert("RGB"))


def build_all():
    print("1. Loading native 3D sequences & extracting smoothed MediaPipe keypoints for Day 4 (Romanian Deadlift)...", flush=True)
    raw_seqs = {}
    kps_seqs = {}
    if os.path.exists(CACHE_KP_PKL):
        with open(CACHE_KP_PKL, "rb") as f:
            kps_seqs = pickle.load(f)

    for s_idx, (fn, i0, i1, is_w, v_mode, y_sh) in STEP_SPECS_DAY4.items():
        seq = load_native_slice_720p(os.path.join(OUT_DIR, fn), i0, i1)
        raw_seqs[s_idx] = seq
        if s_idx not in kps_seqs or len(kps_seqs[s_idx]) != len(seq):
            kps_seqs[s_idx] = b4.extract_smoothed_kps_1080(seq, s_idx, y_sh)

    with open(CACHE_KP_PKL, "wb") as f:
        pickle.dump(kps_seqs, f)

    print("2. Building 4-slot top preview icons from each step's peak frame...", flush=True)
    icons_all_4 = []
    for s_idx in [1, 2, 3, 4]:
        fn, i0, i1, is_w, v_mode, y_sh = STEP_SPECS_DAY4[s_idx]
        seq = raw_seqs[s_idx]
        kps = kps_seqs[s_idx]
        rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(seq[-1], kps[-1], y_sh, s_idx)
        sh_rgb, _ = shade_day4_3d_frame_1080p(rgb_1080, s_idx, is_w, kp_1080, pulse_phase=0.75)
        icons_all_4.append(b4._make_icon_pair(sh_rgb, kp_1080))
    icons_for_step = {1: icons_all_4, 2: icons_all_4, 3: icons_all_4, 4: icons_all_4}

    print("3. Synthesizing 20.0s English Coaching TTS (`en-US-ChristopherNeural`) + Sub-Bass Beat + Step Chimes...", flush=True)
    wav_path = os.path.join(TMP_DIR, "day4_4step_master_audio.wav")
    vo_lines = [
        (0.25, "Romanian deadlift. Stop rounding your lower back."),
        (5.60, "Letting the barbell drift forward strains your lumbar spine."),
        (10.40, "Lock your spine neutral and push your hips straight back."),
        (15.40, "Keep the bar close to drive maximum tension into your hamstrings and glutes."),
    ]
    asyncio.run(v36_pub.synthesize_voiceover_wav(wav_path, vo_lines, DURATION))

    print("4. Streaming 600 frames (20.0s @ 30fps) step-by-step with low memory footprint...", flush=True)
    tmp_mp4 = os.path.join(TMP_DIR, "_day4_video_only.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day4_romanian_deadlift_4step_master.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    step_peak_rendered = {}
    step_standing_rendered = {}

    for step_idx in [1, 2, 3, 4]:
        fn, i0, i1, is_w, v_mode, y_sh = STEP_SPECS_DAY4[step_idx]
        seq = raw_seqs[step_idx]
        kps = kps_seqs[step_idx]
        n_f = len(seq)
        step_cache = {}

        for f_local in range(150):
            u = f_local / 150.0
            wave_p = 0.5 - 0.5 * math.cos(2.0 * math.pi * (u * 1.5))
            p_i = int(round(wave_p * (n_f - 1)))

            if p_i not in step_cache:
                rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(seq[p_i], kps[p_i], y_sh, step_idx)
                pulse_ph = (p_i / float(max(1, n_f - 1))) * 0.75
                sh_rgb, sh_mask = shade_day4_3d_frame_1080p(rgb_1080, step_idx, is_w, kp_1080, pulse_phase=pulse_ph)
                step_cache[p_i] = (sh_rgb, sh_mask, kp_1080)

            base_rgb, sh_mask, kp = step_cache[p_i]

            if step_idx == 1:
                slot_i = 0
                deg_v = int(round(4 + 20 * wave_p))
                pct_v = int(round(80 + 8 * wave_p))
                deg_s, l1_s, l2_s = f"{deg_v}°", "SPINE ROUND", f"{pct_v}% DISC RISK"
            elif step_idx == 2:
                slot_i = 1
                deg_v = int(round(8 + 23 * wave_p))
                pct_v = int(round(74 + 8 * wave_p))
                deg_s, l1_s, l2_s = f"{deg_v}°", "BAR DRIFT", f"{pct_v}% BACK STRAIN"
            elif step_idx == 3:
                slot_i = 2
                pct_v = int(round(90 + 6 * wave_p))
                deg_s, l1_s, l2_s = "0°", "SPINE LOCK", f"{pct_v}% JOINT SAFE"
            else:
                slot_i = 3
                deg_v = int(round(6 + 12 * wave_p))
                pct_v = int(round(90 + 7 * wave_p))
                deg_s, l1_s, l2_s = f"{deg_v}°", "HIP HINGE", f"{pct_v}% HAMSTRING & GLUTE"

            fr = draw_day4_laser_and_callout(base_rgb, sh_mask, step_idx, kp, is_w, deg_s, l1_s, l2_s)
            fg_m = b4.get_mannequin_fg_mask_1080p(base_rgb)
            fr = b4.composite_depth_behind_title_and_translucent_preview(
                fr, fg_m, icons_for_step[step_idx], active_slot=slot_i, is_wrong=is_w, exercise_name="Romanian Deadlift"
            )
            rgb_out = Image.fromarray(fr)

            if f_local == 4:
                step_standing_rendered[step_idx] = rgb_out.copy()
            if f_local == 50:
                step_peak_rendered[step_idx] = rgb_out.copy()
                rgb_out.save(os.path.join(OUT_DIR, f"day4_step{step_idx}_peak.jpg"), quality=95)
                if step_idx == 4:
                    rgb_out.save(os.path.join(OUT_DIR, "day4_romanian_deadlift_4step_master_cover.jpg"), quality=95)

            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

        step_cache.clear()
        gc.collect()
        print(f"   Finished Step {step_idx}/4 (150 frames written).", flush=True)

    vw.release()

    print("5. Muxing H.264 video + AAC 192k audio into final MP4...", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-threads", "4",
            "-i", tmp_mp4, "-i", wav_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_mp4)
    print("Saved Day 4 Master MP4:", final_mp4, flush=True)

    pw, ph = 520, 924
    pad = 22
    header_h = 96
    sheet = Image.new("RGB", (pad + 4 * (pw + pad), header_h + ph + pad), (14, 16, 20))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(26, "extrabold")
    f_s = v30.f_pretendard(18, "bold")
    sd.text(
        (28, 18),
        "Day 4 Romanian Deadlift — 100% Native 3D 4-Step Multi-Angle Master (3안 벽면 타이틀 + 반투명 글래스 프리뷰)",
        font=f_t,
        fill=(242, 246, 252),
    )
    sd.text(
        (28, 54),
        "Step 1: 측면 오자세(요추 디스크 X-Ray) | Step 2: 전측면 오자세(바벨 이탈·승모/요추 과부하) | Step 3: 측면 정자세(척추 중립 X-Ray) | Step 4: 후측면 정자세(햄스트링·둔근)",
        font=f_s,
        fill=(185, 225, 255),
    )
    step_titles = [
        (1, True,  "Step 1: 측면 오자세 (24° SPINE ROUND · 디스크 위험)"),
        (2, True,  "Step 2: 전측면 오자세 (31° BAR DRIFT · 허리 과부하)"),
        (3, False, "Step 3: 측면 정자세 (0° SPINE LOCK · 척추 중립)"),
        (4, False, "Step 4: 후측면 정자세 (18° HIP HINGE · 햄스트링/둔근)"),
    ]
    for idx_p, (s_i, is_w, lbl) in enumerate(step_titles):
        x0 = pad + idx_p * (pw + pad)
        y0 = header_h
        im_p = step_peak_rendered[s_i].resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(im_p, (x0, y0))
        b_col = (245, 65, 65) if is_w else (35, 235, 130)
        sd.rectangle([x0 - 2, y0 - 2, x0 + pw + 1, y0 + ph + 1], outline=b_col, width=3)
        sd.rectangle([x0 + 10, y0 + 10, x0 + pw - 10, y0 + 48], fill=(12, 14, 18))
        sd.text((x0 + 20, y0 + 18), lbl, font=v30.f_pretendard(16, "bold"), fill=b_col)

    sheet_path = os.path.join(OUT_DIR, "day4_4step_multiangle_master_sheet.jpg")
    sheet.save(sheet_path, quality=94)

    comp = Image.new("RGB", (540 * 2, 960), (14, 16, 20))
    comp.paste(step_peak_rendered[4].resize((540, 960), Image.Resampling.LANCZOS), (0, 0))
    comp.paste(step_standing_rendered[4].resize((540, 960), Image.Resampling.LANCZOS), (540, 0))
    comp.save(os.path.join(OUT_DIR, "day4_opt3_hinged_vs_standing_check.jpg"), quality=94)

    caption_text = (
        "Be honest — where do you feel Romanian Deadlifts more when the weight gets heavy: "
        "1 (Lower Back) or 2 (Hamstrings & Glutes)? Drop 1 or 2 in the comments 👇\n\n"
        "Romanian Deadlift — 4-Step 3D Biomechanics Breakdown:\n\n"
        "• Step 1 (Side-Oblique View — Spine Round): Letting your lumbar spine flex under load (24° spine round) shifts up to 88% of the shear stress directly onto your lower back discs.\n"
        "• Step 2 (Front-Oblique View — Bar Drift): Allowing the barbell to drift forward away from your shins increases the moment arm on your torso, overloading the erector spinae and upper traps (82% back strain).\n"
        "• Step 3 (Side View — Spine Lock): Bracing your core with a 0° neutral spine and pushing your hips straight back turns the movement into a pure hip hinge (96% joint safety).\n"
        "• Step 4 (Rear 3/4 View — Hip Hinge): Shaving the barbell down your thighs and shins locks tension directly into the biceps femoris, semitendinosus, and gluteus maximus (97% hamstring & glute activation).\n\n"
        "Save this breakdown for your next posterior-chain session and share it with a lifting partner."
    )
    with open(os.path.join(OUT_DIR, "day4_caption_en.txt"), "w", encoding="utf-8") as f:
        f.write(caption_text + "\n")
    print("Saved 4-Step Sheet & Caption:", sheet_path, flush=True)


if __name__ == "__main__":
    build_all()
