#!/usr/bin/env python3
"""
@gymform.3d — Day 1: Cable Rope Triceps Pushdown (v7 — Luxury Header + 1 Line + 1 Arrow + High-Legibility Scanner Card)
Renders the full 15.0s (450 frames @ 30fps) 9:16 vertical Reel:
- Scene 1 (0.0 – 4.8s): STEP 01/03 · TORSO & SCAPULA (0.0–2.4s ✕ SHOULDER TAKEOVER -> 2.4–4.8s ✓ SCAPULA DEPRESSED)
- Scene 2 (4.8 – 9.6s): STEP 02/03 · ELBOW HINGE AXIS (4.8–7.2s ✕ ELBOW SWINGING -> 7.2–9.6s ✓ LOCK ELBOW AT SIDE)
- Scene 3 (9.6 – 15.0s): STEP 03/03 · WRIST & LOCKOUT (9.6–12.0s ✕ WRIST BENT BACK -> 12.0–15.0s ✓ SPREAD ROPE OUT · 98%)
"""
import os
import math
import wave
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image

from build_3scene_ox_preview import (
    W, H, OUT_DIR, WRONG_FRAMES, CORRECT_FRAMES,
    lerp, enhance_plate, crop_camera_view,
    draw_single_posture_line, draw_lock_on_joint_ring,
    draw_bold_3d_arrow, draw_anatomical_scanner_callout,
    draw_top_header_and_step_bar, draw_ox_status_badge,
)

FPS = 30
DURATION = 15.0
TOTAL_FRAMES = int(FPS * DURATION)  # 450 frames
TMP_DIR = "/workspace/daily_info_reels/frames_gym_day1_v7"
OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")
OUT_KEYFRAMES = os.path.join(OUT_DIR, "demic_gym_day1_keyframes.jpg")


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def sample_clip(frames: list, rep: float, is_wrong: bool) -> np.ndarray:
    n = len(frames)
    u = min(1.0, max(0.0, rep * 0.62))
    f_idx = u * (n - 1)
    i0 = int(math.floor(f_idx))
    i1 = min(n - 1, i0 + 1)
    frac = f_idx - i0
    if frac < 1e-3:
        base = frames[i0].copy()
    else:
        base = cv2.addWeighted(frames[i0], 1.0 - frac, frames[i1], frac, 0)
    return enhance_plate(base, is_wrong=is_wrong, boost_level=0.45 + 0.55 * rep)


def render_single_frame(idx: int) -> int:
    t = idx / FPS

    if t < 4.8:
        scene_idx = 0
        is_correct = (t >= 2.4)
        local_t = t if not is_correct else (t - 2.4)
        prog = smoothstep(0.0, 0.42, local_t)
        rep = 0.25 + 0.45 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))

        raw = sample_clip(CORRECT_FRAMES if is_correct else WRONG_FRAMES, rep, not is_correct)
        z = lerp(1.15, 1.08, smoothstep(0.0, 0.7, local_t))
        framed_bgr, tf = crop_camera_view(raw, cx=730, cy=820, zoom=z)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            hip = tf((895, 995))
            hunched_mid = tf((855, 760))
            shoulder_fwd = tf((lerp(785, 768, rep), lerp(585, 608, rep)))
            target_sh = tf((795, 675))

            draw_single_posture_line(canvas, [hip, hunched_mid, shoulder_fwd], col_rgb=(255, 55, 45), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, target_sh, is_correct=False, r_outer=54, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((835, 495)),
                p1=tf((725, 535)),
                p2=tf((665, 650)),
                col_rgb=(255, 55, 45),
                prog=prog,
            )
            pct = int(round(lerp(54, 68, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_sh,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ FORCE LEAK",
                anatomy_tag="ANTERIOR DELTOID",
                title_text="SHOULDER TAKEOVER",
                metric_label="Front Delt Load",
                metric_pct=pct,
                prog=prog,
            )
        else:
            hip = tf((785, 965))
            mid_back = tf((765, 735))
            shoulder_top = tf((745, 520))
            target_sc = tf((765, 635))

            draw_single_posture_line(canvas, [hip, mid_back, shoulder_top], col_rgb=(40, 250, 125), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, target_sc, is_correct=True, r_outer=54, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((735, 470)),
                p1=tf((845, 555)),
                p2=tf((835, 720)),
                col_rgb=(40, 250, 125),
                prog=prog,
            )
            pct = int(round(lerp(82, 92, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_sc,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ POSTURE LOCKED",
                anatomy_tag="SCAPULA & TRICEPS",
                title_text="SCAPULA DEPRESSED",
                metric_label="Triceps Isolation",
                metric_pct=pct,
                prog=prog,
            )

    elif t < 9.6:
        scene_idx = 1
        is_correct = (t >= 7.2)
        local_t = (t - 4.8) if not is_correct else (t - 7.2)
        prog = smoothstep(0.0, 0.42, local_t)
        rep = 0.20 + 0.70 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))

        raw = sample_clip(CORRECT_FRAMES if is_correct else WRONG_FRAMES, rep, not is_correct)
        z = lerp(1.24, 1.32, smoothstep(0.0, 0.6, local_t))
        framed_bgr, tf = crop_camera_view(raw, cx=680, cy=790, zoom=z)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            shoulder = tf((810, 680))
            drifted_elbow = tf((lerp(765, 730, rep), lerp(900, 915, rep)))

            draw_single_posture_line(canvas, [shoulder, drifted_elbow], col_rgb=(255, 55, 45), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, drifted_elbow, is_correct=False, r_outer=56, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((830, 935)),
                p1=tf((710, 925)),
                p2=tf((595, 845)),
                col_rgb=(255, 55, 45),
                prog=prog,
            )
            pct = int(round(lerp(42, 32, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=drifted_elbow,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ TENSION LOST",
                anatomy_tag="HUMERUS DRIFT",
                title_text="ELBOW SWINGING",
                metric_label="Triceps Tension",
                metric_pct=pct,
                prog=prog,
            )
        else:
            shoulder = tf((775, 600))
            elbow = tf((735, 815))
            wrist = tf((lerp(455, 515, rep), lerp(885, 1045, rep)))

            draw_single_posture_line(canvas, [shoulder, elbow, wrist], col_rgb=(40, 250, 125), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, elbow, is_correct=True, r_outer=56, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((455, 760)),
                p1=tf((430, 915)),
                p2=tf((525, 1060)),
                col_rgb=(40, 250, 125),
                prog=min(1.0, 0.35 + 0.75 * rep),
            )
            pct = int(round(lerp(78, 96, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=elbow,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ PURE HINGE AXIS",
                anatomy_tag="ELBOW JOINT PINNED",
                title_text="LOCK ELBOW AT SIDE",
                metric_label="Triceps Tension",
                metric_pct=pct,
                prog=prog,
            )

    else:
        scene_idx = 2
        is_correct = (t >= 12.0)
        local_t = (t - 9.6) if not is_correct else (t - 12.0)
        prog = smoothstep(0.0, 0.42, local_t)

        if not is_correct:
            rep = 0.40 + 0.55 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))
            raw = sample_clip(WRONG_FRAMES, rep, True)
            z = lerp(1.40, 1.48, smoothstep(0.0, 0.6, local_t))
            framed_bgr, tf = crop_camera_view(raw, cx=600, cy=930, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            elbow = tf((825, 885))
            wrist = tf((535, 1005))
            bent_hand = tf((415, lerp(975, 950, rep)))

            draw_single_posture_line(canvas, [elbow, wrist, bent_hand], col_rgb=(255, 55, 45), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, wrist, is_correct=False, r_outer=56, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((495, 1085)),
                p1=tf((415, 1025)),
                p2=tf((395, 905)),
                col_rgb=(255, 55, 45),
                prog=prog,
            )
            pct = int(round(lerp(72, 85, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=wrist,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ JOINT STRAIN",
                anatomy_tag="WRIST EXTENSION",
                title_text="WRIST BENT BACK",
                metric_label="Wrist Joint Stress",
                metric_pct=pct,
                prog=prog,
            )
        else:
            rep = 0.35 + 0.65 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 1.5)))
            raw = sample_clip(CORRECT_FRAMES, rep, False)
            z = lerp(1.42, 1.48, smoothstep(0.0, 0.5, local_t))
            framed_bgr, tf = crop_camera_view(raw, cx=600, cy=930, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            shoulder_triceps = tf((740, 745))
            elbow = tf((735, 845))
            wrist = tf((515, 965))

            draw_single_posture_line(canvas, [elbow, wrist], col_rgb=(40, 250, 125), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, shoulder_triceps, is_correct=True, r_outer=58, prog=prog)

            spread_p = min(1.0, max(0.30, rep))
            draw_bold_3d_arrow(
                canvas,
                p0=tf((515, 965)),
                p1=tf((410, 1045)),
                p2=tf((315, 1145)),
                col_rgb=(40, 250, 125),
                prog=spread_p,
            )
            draw_bold_3d_arrow(
                canvas,
                p0=tf((515, 965)),
                p1=tf((615, 1065)),
                p2=tf((715, 1175)),
                col_rgb=(40, 250, 125),
                prog=spread_p,
            )
            pct = int(round(lerp(86, 98, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=shoulder_triceps,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ PEAK CONTRACTION",
                anatomy_tag="TRICEPS LATERAL HEAD",
                title_text="SPREAD ROPE OUT",
                metric_label="Lateral Head EMG",
                metric_pct=pct,
                prog=prog,
            )

    draw_top_header_and_step_bar(canvas, active_idx=scene_idx, is_correct=is_correct)
    draw_ox_status_badge(canvas, is_correct=is_correct)

    out_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}.jpg"), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return idx


def build_audio_track(wav_path: str):
    sr = 44100
    n_samples = int(sr * DURATION)
    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32) / sr

    beat_env = 0.5 + 0.5 * np.sin(2.0 * math.pi * 1.333 * t)
    sub = 0.08 * np.sin(2.0 * math.pi * 54.0 * t) * beat_env
    left += sub
    right += sub

    for (s0, s1) in [(0.1, 2.3), (4.9, 7.1), (9.7, 11.9)]:
        mask = (t >= s0) & (t <= s1)
        buzz = 0.055 * np.sin(2.0 * math.pi * 92.0 * t[mask]) + 0.025 * np.sin(2.0 * math.pi * 184.0 * t[mask])
        left[mask] += buzz
        right[mask] += buzz * 0.9

    for t_cut in [4.8, 9.6]:
        mask = (t >= t_cut - 0.12) & (t < t_cut + 0.22)
        dt = t[mask] - (t_cut - 0.12)
        whoosh = 0.14 * np.sin(math.pi * dt / 0.34) * np.sin(2.0 * math.pi * (180.0 + 320.0 * dt) * dt)
        left[mask] += whoosh
        right[mask] += whoosh

    for t_snap in [2.4, 7.2, 12.0]:
        mask = (t >= t_snap) & (t < t_snap + 0.45)
        dt = t[mask] - t_snap
        snap = 0.25 * np.exp(-dt * 28.0) * np.sin(2.0 * math.pi * (880.0 + 440.0 * np.exp(-dt * 40.0)) * dt)
        chime = 0.16 * np.exp(-dt * 6.5) * np.sin(2.0 * math.pi * 659.25 * dt)
        left[mask] += snap + chime
        right[mask] += snap + chime

    notes = [523.25, 659.25, 783.99, 1046.50]
    for i, freq in enumerate(notes):
        t0 = 12.15 + i * 0.18
        mask = (t >= t0) & (t < t0 + 0.70)
        dt = t[mask] - t0
        c_wave = 0.13 * np.exp(-dt * 5.5) * (np.sin(2.0 * math.pi * freq * dt) + 0.35 * np.sin(2.0 * math.pi * freq * 2.0 * dt))
        left[mask] += c_wave * (0.85 + 0.04 * i)
        right[mask] += c_wave * (1.0 - 0.03 * i)

    peak = max(1e-4, float(max(np.max(np.abs(left)), np.max(np.abs(right)))))
    left = np.clip(left / peak * 0.88, -1.0, 1.0)
    right = np.clip(right / peak * 0.88, -1.0, 1.0)

    interleaved = np.empty(n_samples * 2, dtype=np.int16)
    interleaved[0::2] = (left * 32767).astype(np.int16)
    interleaved[1::2] = (right * 32767).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    print(f"Rendering {TOTAL_FRAMES} frames (15.0s @ 30fps) for v7...")
    with ProcessPoolExecutor(max_workers=8) as ex:
        list(ex.map(render_single_frame, range(TOTAL_FRAMES)))

    wav_path = os.path.join(OUT_DIR, "gym_day1_audio.wav")
    build_audio_track(wav_path)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", os.path.join(TMP_DIR, "frame_%04d.jpg"),
            "-i", wav_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            OUT_MP4,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(wav_path)

    pw, ph = 360, 640
    top_row = [
        cv2.resize(cv2.imread(os.path.join(TMP_DIR, f"frame_{int(1.2*FPS):04d}.jpg")), (pw, ph)),
        cv2.resize(cv2.imread(os.path.join(TMP_DIR, f"frame_{int(6.0*FPS):04d}.jpg")), (pw, ph)),
        cv2.resize(cv2.imread(os.path.join(TMP_DIR, f"frame_{int(10.8*FPS):04d}.jpg")), (pw, ph)),
    ]
    bot_row = [
        cv2.resize(cv2.imread(os.path.join(TMP_DIR, f"frame_{int(3.6*FPS):04d}.jpg")), (pw, ph)),
        cv2.resize(cv2.imread(os.path.join(TMP_DIR, f"frame_{int(8.4*FPS):04d}.jpg")), (pw, ph)),
        cv2.resize(cv2.imread(os.path.join(TMP_DIR, f"frame_{int(13.5*FPS):04d}.jpg")), (pw, ph)),
    ]
    grid = np.vstack([np.hstack(top_row), np.hstack(bot_row)])
    cv2.imwrite(OUT_KEYFRAMES, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("DONE v7 MP4! Saved:", OUT_MP4, OUT_KEYFRAMES)


if __name__ == "__main__":
    main()
