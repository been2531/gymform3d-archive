#!/usr/bin/env python3
"""
@gymform.3d — Day 2: Wide-Grip Lat Pulldown (v8 — Multi-Angle 3D Biomechanical Motion + Selective Muscle Shader + Luxury HUD)
Renders the full 15.0s (450 frames @ 30fps) 9:16 vertical Reel (`output/demic_gym_day2.mp4`) and keyframes grid (`output/demic_gym_day2_keyframes.jpg`).
Uses smooth anatomical displacement fields (`cv2.remap`) synchronized with joint-tracked posture lines, 3D arrows, and 20-segment EMG telemetry.
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
    W, H, OUT_DIR,
    lerp, crop_camera_view,
    draw_single_posture_line, draw_lock_on_joint_ring,
    draw_bold_3d_arrow, draw_ox_status_badge,
    draw_anatomical_scanner_callout,
)
from build_day2_ox_preview import (
    load_plate, selective_muscle_grade, draw_day2_header,
)

FPS = 30
DURATION = 15.0
TOTAL_FRAMES = int(FPS * DURATION)  # 450 frames
TMP_DIR = "/workspace/daily_info_reels/frames_gym_day2_v8"
OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day2.mp4")
OUT_KEYFRAMES = os.path.join(OUT_DIR, "demic_gym_day2_keyframes.jpg")

PLATES = {
    (0, False): load_plate("day2_step1_wrong_plate.jpg"),
    (0, True):  load_plate("day2_step1_correct_plate.jpg"),
    (1, False): load_plate("day2_step2_wrong_plate.jpg"),
    (1, True):  load_plate("day2_step2_correct_plate.jpg"),
    (2, False): load_plate("day2_step3_wrong_plate.jpg"),
    (2, True):  load_plate("day2_step3_correct_plate.jpg"),
}

BASE_Y, BASE_X = np.mgrid[0:H, 0:W].astype(np.float32)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def warp_anatomical_motion(plate_bgr: np.ndarray, lobes: list) -> np.ndarray:
    """
    Applies smooth localized Gaussian displacement lobes `(cx, cy, rx, ry, dx, dy)`
    to animate realistic anatomical joint/muscle motion while keeping 1080p studio sharpness.
    """
    map_x = BASE_X.copy()
    map_y = BASE_Y.copy()
    for (cx, cy, rx, ry, dx, dy) in lobes:
        w = np.exp(-((BASE_X - cx) ** 2 / (2.0 * rx ** 2) + (BASE_Y - cy) ** 2 / (2.0 * ry ** 2)))
        map_x -= dx * w
        map_y -= dy * w
    return cv2.remap(plate_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def render_single_frame(idx: int) -> int:
    t = idx / FPS

    if t < 4.8:
        scene_idx = 0
        is_correct = (t >= 2.4)
        local_t = t if not is_correct else (t - 2.4)
        prog = smoothstep(0.0, 0.40, local_t)
        rep = 0.20 + 0.80 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))

        raw = PLATES[(0, is_correct)]
        if not is_correct:
            # Upward shoulder/trap shrug motion
            shrug_dy = -26.0 * rep
            warped = warp_anatomical_motion(
                raw,
                [
                    (665, 760, 220, 160, 0.0, shrug_dy),
                    (520, 680, 160, 180, -6.0 * rep, shrug_dy * 0.85),
                    (820, 680, 160, 180, 6.0 * rep, shrug_dy * 0.85),
                ],
            )
            graded = selective_muscle_grade(
                warped,
                is_correct=False,
                hot_zones=[
                    (660, 775 + shrug_dy, 118, 68, 0.42 + 0.32 * rep),
                    (585, 780 + shrug_dy, 78, 58, 0.35 + 0.25 * rep),
                    (755, 785 + shrug_dy, 78, 58, 0.35 + 0.25 * rep),
                ],
                suppress_zones=[(675, 960, 220, 130)],
            )
            z = lerp(1.04, 1.10, smoothstep(0.0, 2.2, local_t))
            framed_bgr, tf = crop_camera_view(graded, cx=640, cy=860, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            left_sh = tf((525, 800 + shrug_dy * 0.8))
            neck_shrug = tf((665, 745 + shrug_dy))
            right_sh = tf((825, 805 + shrug_dy * 0.8))
            target_trap = tf((665, 765 + shrug_dy))

            draw_single_posture_line(canvas, [left_sh, neck_shrug, right_sh], col_rgb=(255, 55, 45), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, target_trap, is_correct=False, r_outer=56, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((665, 910)),
                p1=tf((665, 790)),
                p2=tf((665, 645 + shrug_dy * 0.5)),
                col_rgb=(255, 55, 45),
                prog=min(1.0, 0.35 + 0.75 * rep),
            )
            pct = int(round(lerp(58, 74, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_trap,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ FORCE LEAK",
                anatomy_tag="UPPER TRAPEZIUS",
                title_text="SHRUGGING SHOULDERS",
                metric_label="Upper Trap Takeover",
                metric_pct=pct,
                prog=prog,
            )
        else:
            # Downward & inward scapular depression motion
            dep_dy = 24.0 * rep
            warped = warp_anatomical_motion(
                raw,
                [
                    (655, 835, 230, 175, 0.0, dep_dy),
                    (520, 740, 160, 170, 10.0 * rep, dep_dy * 0.9),
                    (790, 740, 160, 170, -10.0 * rep, dep_dy * 0.9),
                ],
            )
            graded = selective_muscle_grade(
                warped,
                is_correct=True,
                hot_zones=[
                    (575, 910 + dep_dy * 0.5, 98, 128, 0.30 + 0.28 * rep),
                    (735, 910 + dep_dy * 0.5, 98, 128, 0.30 + 0.28 * rep),
                    (655, 865 + dep_dy * 0.6, 88, 98, 0.25 + 0.25 * rep),
                ],
                suppress_zones=[(655, 690, 130, 55)],
            )
            z = lerp(1.04, 1.10, smoothstep(0.0, 2.2, local_t))
            framed_bgr, tf = crop_camera_view(graded, cx=640, cy=860, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            left_sc = tf((545, 785 + dep_dy * 0.7))
            mid_sc = tf((655, 875 + dep_dy))
            right_sc = tf((775, 785 + dep_dy * 0.7))
            target_sc = tf((655, 865 + dep_dy * 0.8))

            draw_single_posture_line(canvas, [left_sc, mid_sc, right_sc], col_rgb=(40, 250, 125), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, target_sc, is_correct=True, r_outer=56, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((655, 705)),
                p1=tf((655, 835)),
                p2=tf((655, 985 + dep_dy * 0.4)),
                col_rgb=(40, 250, 125),
                prog=min(1.0, 0.35 + 0.75 * rep),
            )
            pct = int(round(lerp(78, 91, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_sc,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ SCAPULA LOCKED",
                anatomy_tag="LOWER TRAPS & LATS",
                title_text="DEPRESS SCAPULA FIRST",
                metric_label="Latissimus Activation",
                metric_pct=pct,
                prog=prog,
            )

    elif t < 9.6:
        scene_idx = 1
        is_correct = (t >= 7.2)
        local_t = (t - 4.8) if not is_correct else (t - 7.2)
        prog = smoothstep(0.0, 0.40, local_t)
        rep = 0.20 + 0.80 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))

        raw = PLATES[(1, is_correct)]
        if not is_correct:
            # Excessive backward momentum swing
            swing_dx = 26.0 * rep
            swing_dy = 14.0 * rep
            warped = warp_anatomical_motion(
                raw,
                [
                    (675, 860, 240, 220, swing_dx, swing_dy),
                    (645, 1085, 160, 150, swing_dx * 0.45, swing_dy * 0.4),
                ],
            )
            graded = selective_muscle_grade(
                warped,
                is_correct=False,
                hot_zones=[
                    (650 + swing_dx * 0.4, 1080 + swing_dy * 0.4, 88, 148, 0.35 + 0.30 * rep),
                    (755 + swing_dx * 0.7, 800 + swing_dy * 0.7, 78, 78, 0.32 + 0.28 * rep),
                ],
                suppress_zones=[],
            )
            z = lerp(1.10, 1.18, smoothstep(0.0, 2.2, local_t))
            framed_bgr, tf = crop_camera_view(graded, cx=650, cy=930, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            lumbar = tf((635, 1185))
            mid_spine = tf((655 + swing_dx * 0.5, 985 + swing_dy * 0.5))
            upper_spine = tf((695 + swing_dx * 0.9, 780 + swing_dy * 0.9))
            target_lumbar = tf((645 + swing_dx * 0.3, 1095 + swing_dy * 0.3))

            draw_single_posture_line(canvas, [lumbar, mid_spine, upper_spine], col_rgb=(255, 55, 45), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, target_lumbar, is_correct=False, r_outer=56, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((560, 760)),
                p1=tf((695, 785)),
                p2=tf((835 + swing_dx * 0.5, 885 + swing_dy * 0.5)),
                col_rgb=(255, 55, 45),
                prog=min(1.0, 0.35 + 0.75 * rep),
            )
            pct = int(round(lerp(66, 82, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_lumbar,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ MOMENTUM SWING",
                anatomy_tag="LUMBAR & ROTATOR CUFF",
                title_text="EXCESSIVE BACK SWING",
                metric_label="Lower Back Shear",
                metric_pct=pct,
                prog=prog,
            )
        else:
            # Locked 15° thoracic arch while bar & arms pull cleanly toward upper chest
            pull_dy = 26.0 * rep
            warped = warp_anatomical_motion(
                raw,
                [
                    (575, 760, 180, 190, 8.0 * rep, pull_dy),
                    (715, 955, 140, 150, -6.0 * rep, 12.0 * rep),
                ],
            )
            graded = selective_muscle_grade(
                warped,
                is_correct=True,
                hot_zones=[(715, 955, 115, 135, 0.35 + 0.32 * rep)],
                suppress_zones=[],
            )
            z = lerp(1.10, 1.18, smoothstep(0.0, 2.2, local_t))
            framed_bgr, tf = crop_camera_view(graded, cx=650, cy=910, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            hip = tf((655, 1180))
            thoracic = tf((675, 940))
            shoulder = tf((695, 745))
            target_lat = tf((715, 955))

            draw_single_posture_line(canvas, [hip, thoracic, shoulder], col_rgb=(40, 250, 125), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, target_lat, is_correct=True, r_outer=56, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((535, 620)),
                p1=tf((555, 765)),
                p2=tf((605, 905 + pull_dy * 0.5)),
                col_rgb=(40, 250, 125),
                prog=min(1.0, 0.35 + 0.75 * rep),
            )
            pct = int(round(lerp(82, 95, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_lat,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ 15° THORACIC ARCH",
                anatomy_tag="UPPER CHEST BAR PATH",
                title_text="PULL TO UPPER CHEST",
                metric_label="Latissimus Tension",
                metric_pct=pct,
                prog=prog,
            )

    else:
        scene_idx = 2
        is_correct = (t >= 12.0)
        local_t = (t - 9.6) if not is_correct else (t - 12.0)
        prog = smoothstep(0.0, 0.40, local_t)

        raw = PLATES[(2, is_correct)]
        if not is_correct:
            rep = 0.20 + 0.80 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))
            curl_dx = 22.0 * rep
            curl_dy = -16.0 * rep
            warped = warp_anatomical_motion(
                raw,
                [
                    (255, 745, 165, 155, curl_dx, curl_dy),
                    (855, 765, 165, 155, -curl_dx, curl_dy),
                ],
            )
            graded = selective_muscle_grade(
                warped,
                is_correct=False,
                hot_zones=[
                    (265 + curl_dx * 0.5, 735 + curl_dy * 0.5, 115, 98, 0.36 + 0.30 * rep),
                    (855 - curl_dx * 0.5, 765 + curl_dy * 0.5, 115, 98, 0.36 + 0.30 * rep),
                ],
                suppress_zones=[(540, 920, 195, 160)],
            )
            z = lerp(1.14, 1.22, smoothstep(0.0, 2.2, local_t))
            framed_bgr, tf = crop_camera_view(graded, cx=560, cy=860, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            wrist_L = tf((175 + curl_dx * 0.7, 645 + curl_dy * 0.5))
            elbow_L = tf((245 + curl_dx * 0.4, 815 + curl_dy * 0.4))
            shoulder_L = tf((395, 785))
            target_bicep = tf((295 + curl_dx * 0.4, 775 + curl_dy * 0.4))

            draw_single_posture_line(canvas, [wrist_L, elbow_L, shoulder_L], col_rgb=(255, 55, 45), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, target_bicep, is_correct=False, r_outer=56, prog=prog)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((180, 740)),
                p1=tf((255, 665)),
                p2=tf((375, 710)),
                col_rgb=(255, 55, 45),
                prog=min(1.0, 0.35 + 0.75 * rep),
            )
            pct = int(round(lerp(46, 34, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_bicep,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ TENSION LOST",
                anatomy_tag="BICEPS & FOREARMS",
                title_text="CURLING WITH BICEPS",
                metric_label="Latissimus Isolation",
                metric_pct=pct,
                prog=prog,
            )
        else:
            # 2 peak-contraction reps in 3.0s
            rep = 0.25 + 0.75 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 1.5)))
            drive_dy = 28.0 * rep
            drive_dx = -16.0 * rep
            warped = warp_anatomical_motion(
                raw,
                [
                    (885, 835, 175, 195, drive_dx, drive_dy),
                    (465, 815, 175, 195, -drive_dx, drive_dy),
                    (725, 975, 155, 165, -8.0 * rep, 14.0 * rep),
                ],
            )
            graded = selective_muscle_grade(
                warped,
                is_correct=True,
                hot_zones=[
                    (725, 975, 128, 148, 0.38 + 0.35 * rep),
                    (615, 965, 98, 132, 0.28 + 0.26 * rep),
                ],
                suppress_zones=[(445, 745, 90, 90)],
            )
            z = lerp(1.14, 1.22, smoothstep(0.0, 2.5, local_t))
            framed_bgr, tf = crop_camera_view(graded, cx=650, cy=900, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            hand_R = tf((915 + drive_dx * 0.5, 635 + drive_dy * 0.7))
            elbow_R = tf((885 + drive_dx * 0.8, 855 + drive_dy * 0.8))
            target_lat = tf((735, 965))

            draw_single_posture_line(canvas, [hand_R, elbow_R], col_rgb=(40, 250, 125), width=13, prog=prog)
            draw_lock_on_joint_ring(canvas, target_lat, is_correct=True, r_outer=58, prog=prog)

            arc_p = min(1.0, max(0.32, rep))
            draw_bold_3d_arrow(
                canvas,
                p0=tf((895, 825)),
                p1=tf((845, 965)),
                p2=tf((755, 1095)),
                col_rgb=(40, 250, 125),
                prog=arc_p,
            )
            draw_bold_3d_arrow(
                canvas,
                p0=tf((455, 805)),
                p1=tf((505, 945)),
                p2=tf((595, 1075)),
                col_rgb=(40, 250, 125),
                prog=arc_p,
            )
            pct = int(round(lerp(85, 97, rep)))
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_lat,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ PEAK CONTRACTION",
                anatomy_tag="LATISSIMUS DORSI",
                title_text="DRIVE ELBOWS TO HIPS",
                metric_label="Latissimus Dorsi EMG",
                metric_pct=pct,
                prog=prog,
            )

    draw_day2_header(canvas, active_idx=scene_idx, is_correct=is_correct)
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
    print(f"Rendering {TOTAL_FRAMES} frames (15.0s @ 30fps) for Day 2 v8...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(render_single_frame, range(TOTAL_FRAMES)))

    wav_path = os.path.join(OUT_DIR, "gym_day2_audio.wav")
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
    print("DONE Day 2 v8 MP4! Saved:", OUT_MP4, OUT_KEYFRAMES)


if __name__ == "__main__":
    main()
