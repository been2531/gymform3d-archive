#!/usr/bin/env python3
import os
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image

import build_demic_plus1 as bp1
from build_demic_1to1 import (
    W, H, FPS, DURATION, TOTAL_FRAMES,
    S1_MASTER, S2_NECK, OUT_DIR,
    extract_fullscreen_9x16_plates,
    build_minimap_thumbnails,
    animate_plate,
    draw_demic_status_badge,
    draw_demic_minimap_pill,
    draw_demic_3d_arrow,
    synthesize_demic_audio,
)

WRONG_PLATE, CORRECT_PLATE = extract_fullscreen_9x16_plates(S2_NECK)
THUMBS = build_minimap_thumbnails(S1_MASTER)
TMP_DIR = "/workspace/daily_info_reels/frames_plus1"
os.makedirs(TMP_DIR, exist_ok=True)


def render_one_frame(f: int):
    t = f / FPS
    pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.0 * t)

    if t < 4.6:
        plate = animate_plate(WRONG_PLATE, t, is_wrong=True)
        is_corr = False
    elif t < 5.0:
        alpha = (t - 4.6) / 0.4
        p_w = animate_plate(WRONG_PLATE, t, is_wrong=True).astype(np.float32)
        p_c = animate_plate(CORRECT_PLATE, t, is_wrong=False).astype(np.float32)
        plate = (p_w * (1.0 - alpha) + p_c * alpha).astype(np.uint8)
        is_corr = (alpha >= 0.5)
    else:
        plate = animate_plate(CORRECT_PLATE, t, is_wrong=False)
        is_corr = True

    if t < 1.8:
        u = t / 1.8
        ease = 1.0 - (1.0 - u) ** 3
        zoom = 1.72 - 0.72 * ease
        focus_x = 540.0
        focus_y = 820.0 * (1.0 - ease) + 960.0 * ease
    else:
        zoom = 1.0 + 0.04 * math.sin(math.pi * (t - 1.8) / (DURATION - 1.8))
        focus_x, focus_y = 540.0, 960.0

    if zoom > 1.001:
        zw = int(W / zoom)
        zh = int(H / zoom)
        cx0 = int(np.clip(focus_x - zw / 2, 0, W - zw))
        cy0 = int(np.clip(focus_y - zh / 2, 0, H - zh))
        cropped = plate[cy0:cy0 + zh, cx0:cx0 + zw]
        plate = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    frame_rgba = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)).convert("RGBA")

    if 2.0 <= t < 4.6:
        prog = min(1.0, (t - 2.0) / 0.6)
        draw_demic_3d_arrow(frame_rgba, (350, 690), (265, 760), (255, 65, 45), prog)
    elif t >= 5.1:
        prog = min(1.0, (t - 5.1) / 0.6)
        draw_demic_3d_arrow(frame_rgba, (280, 695), (395, 695), (40, 255, 115), prog)
        draw_demic_3d_arrow(frame_rgba, (575, 1020), (575, 865), (40, 255, 115), prog)

    if 1.8 <= t < 4.6:
        diff_prog = min(1.0, (t - 1.8) / 0.45)
        bp1.draw_signature_differentiator(frame_rgba, plate, is_correct=False, t=t, pulse=pulse, appear_prog=diff_prog)
    elif t >= 5.0:
        diff_prog = min(1.0, (t - 5.0) / 0.45)
        bp1.draw_signature_differentiator(frame_rgba, plate, is_correct=True, t=t, pulse=pulse, appear_prog=diff_prog)

    if t >= 1.0:
        draw_demic_minimap_pill(frame_rgba, THUMBS, is_correct=is_corr, pulse=pulse)
        draw_demic_status_badge(frame_rgba, is_correct=is_corr, pulse=pulse)

    rgb_out = frame_rgba.convert("RGB")
    rgb_out.save(f"{TMP_DIR}/f_{f:04d}.jpg", quality=94)


def main():
    with ProcessPoolExecutor() as ex:
        list(ex.map(render_one_frame, range(TOTAL_FRAMES)))

    wav_path = f"{TMP_DIR}/audio.wav"
    synthesize_demic_audio(wav_path)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", f"{TMP_DIR}/f_%04d.jpg",
        "-i", wav_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        bp1.OUT_MP4,
    ]
    subprocess.run(cmd, check=True)

    kf_indices = [int(0.4 * FPS), int(2.8 * FPS), int(6.2 * FPS)]
    kfs = [Image.open(f"{TMP_DIR}/f_{k:04d}.jpg") for k in kf_indices]

    strip = Image.new("RGB", (W * 3, H))
    for i, im in enumerate(kfs):
        strip.paste(im, (i * W, 0))
    strip.resize((1620, 960), Image.Resampling.LANCZOS).save(bp1.OUT_KEYFRAME, quality=92)

    basic_kf = Image.open(os.path.join(OUT_DIR, "demic_1to1_day1_keyframes.jpg"))
    basic_wrong = basic_kf.crop((540, 0, 1080, 960))
    basic_corr = basic_kf.crop((1080, 0, 1620, 960))
    plus1_wrong = kfs[1].resize((540, 960), Image.Resampling.LANCZOS)
    plus1_corr = kfs[2].resize((540, 960), Image.Resampling.LANCZOS)

    comp = Image.new("RGB", (540 * 4, 960))
    comp.paste(basic_wrong, (0, 0))
    comp.paste(plus1_wrong, (540, 0))
    comp.paste(basic_corr, (1080, 0))
    comp.paste(plus1_corr, (1620, 0))
    comp.save(bp1.OUT_DIFF_COMP, quality=92)
    print("FAST RENDER DONE:", bp1.OUT_MP4)


if __name__ == "__main__":
    main()
