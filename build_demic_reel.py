import os
import math
import time
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Import our exact post-processing UI functions so there is ZERO text and 100% @demicstory style
import sys
sys.path.insert(0, "/workspace/daily_info_reels")
from generate_demic_3d import (
    apply_neon_bloom,
    draw_status_badge,
    draw_joint_chain_o_o_o,
    draw_neon_arrow,
    draw_muscle_map_inset,
)

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_DIR = "/workspace/daily_info_reels/frames_demic"

def make_full_frame_from_raw(raw_img_path):
    """Expands a 1080x960 Blender EEVEE studio render into a full 1080x1920 vertical studio frame with seamless ceiling & floor reflection."""
    raw = Image.open(raw_img_path).convert("RGB")
    # Crop central 540x960 (9:16 aspect ratio) around the mannequin so it fills the full 1080x1920 vertical frame!
    # The mannequin is centered around x=490 in the 1080x960 render.
    x0 = 220
    crop = raw.crop((x0, 0, x0 + 540, 960))
    full = crop.resize((1080, 1920), Image.Resampling.LANCZOS)
    return full

def map_pt(pt_1080x960, x0=220):
    return ((pt_1080x960[0] - x0) * 2.0, pt_1080x960[1] * 2.0)

def main():
    t0 = time.time()
    base_wrong = make_full_frame_from_raw(f"{TMP_DIR}/raw_a_wrong.png")
    base_right = make_full_frame_from_raw(f"{TMP_DIR}/raw_a_right.png")

    arr_wrong = np.array(base_wrong, dtype=np.float32)
    arr_right = np.array(base_right, dtype=np.float32)

    # Exact projected 2D coordinates on the 1080x960 render for Sample A:
    # Wrong posture joint chain (ear -> C7 -> shoulder -> hip)
    pts_w_raw = [(402, 238), (496, 336), (484, 388), (528, 692)]
    pts_r_raw = [(508, 178), (528, 304), (534, 362), (542, 692)]

    pts_w = [map_pt(p) for p in pts_w_raw]
    pts_r = [map_pt(p) for p in pts_r_raw]

    head_f_w = map_pt((358, 250))
    head_f_r = map_pt((456, 192))
    mid_b_r = map_pt((595, 430))

    total_frames = 240  # 8.0 seconds @ 30fps
    for f in range(total_frames):
        sec = f / 30.0
        # 0.0 - 3.5s: Wrong form (Red X) with animated pulsing neon red cervical spine glow & moving shear arrow
        # 3.5 - 4.3s: Smooth biomechanical transition from Wrong (Red X) to Right (Green ✓)
        # 4.3 - 8.0s: Correct form (Green ✓) with animated pulsing neon green deep-neck/mid-back glow & chin-tuck arrows
        if sec < 3.4:
            u = 0.0
        elif sec < 4.4:
            t = (sec - 3.4) / 1.0
            u = t * t * (3.0 - 2.0 * t)
        else:
            u = 1.0

        # Subtle camera zoom / breathing micro-motion so every single frame is dynamically alive
        zoom = 1.0 + 0.025 * math.sin(sec * math.pi * 0.5)
        frame_arr = (1.0 - u) * arr_wrong + u * arr_right
        frame_img = Image.fromarray(np.clip(frame_arr, 0, 255).astype(np.uint8))

        # Pulse neon bloom intensity smoothly at 2Hz
        pulse = 0.85 + 0.35 * (0.5 + 0.5 * math.sin(sec * math.pi * 4.0))
        frame_img = apply_neon_bloom(frame_img, radius=22, intensity=pulse)

        # Interpolate `o—o—o` joint chain positions smoothly
        cur_pts = []
        for pw, pr in zip(pts_w, pts_r):
            cx = int((1.0 - u) * pw[0] + u * pr[0])
            cy = int((1.0 - u) * pw[1] + u * pr[1])
            cur_pts.append((cx, cy))

        is_corr = (u >= 0.5)
        dot_c = (0, 230, 95) if is_corr else (255, 50, 25)
        frame_img = draw_joint_chain_o_o_o(frame_img, cur_pts, line_color=(255, 255, 255), dot_fill=dot_c)

        # Animated directional arrow (grows and pulses along force direction)
        arrow_phase = 0.65 + 0.35 * math.sin(sec * math.pi * 3.0)
        if not is_corr:
            p0 = (int(head_f_w[0] + 20), int(head_f_w[1] - 70))
            p1 = (int(p0[0] - 135 * arrow_phase), int(p0[1] + 95 * arrow_phase))
            frame_img = draw_neon_arrow(frame_img, p0, p1, color=(255, 45, 20), width=14, head_len=38)
        else:
            # Green chin-tuck horizontal retraction arrow
            p0 = (int(head_f_r[0] - 150), int(head_f_r[1] + 10))
            p1 = (int(p0[0] + 135 * arrow_phase), int(p0[1] - 12 * arrow_phase))
            frame_img = draw_neon_arrow(frame_img, p0, p1, color=(0, 255, 102), width=14, head_len=38)
            # Green thoracic extension upward arrow
            m0 = (int(mid_b_r[0] + 40), int(mid_b_r[1] + 110))
            m1 = (int(m0[0] - 25 * arrow_phase), int(m0[1] - 170 * arrow_phase))
            frame_img = draw_neon_arrow(frame_img, m0, m1, color=(0, 255, 102), width=14, head_len=38)

        # Top-Right Status Badge (Red X -> Green Checkmark)
        frame_img = draw_status_badge(frame_img, 1080 - 120, 130, radius=64, is_correct=is_corr)
        # Top-Left Muscle Map Inset
        frame_img = draw_muscle_map_inset(frame_img, 48, 64, size=168, highlight_zone="neck",
                                          highlight_color=(0, 255, 102) if is_corr else (255, 45, 20))

        # Apply subtle smooth studio camera push-in
        if zoom > 1.001:
            zw = int(1080 / zoom)
            zh = int(1920 / zoom)
            zx0 = (1080 - zw) // 2
            zy0 = (1920 - zh) // 2
            frame_img = frame_img.crop((zx0, zy0, zx0 + zw, zy0 + zh)).resize((1080, 1920), Image.Resampling.BILINEAR)

        frame_img.save(f"{TMP_DIR}/reel_frame_{f:04d}.jpg", quality=94)

    mp4_path = f"{OUT_DIR}/demic_clone_day1.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", f"{TMP_DIR}/reel_frame_%04d.jpg",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        mp4_path
    ]
    subprocess.run(cmd, check=True)
    print(f"MP4 Reel generated in {time.time()-t0:.2f}s -> {mp4_path}")

if __name__ == "__main__":
    main()
