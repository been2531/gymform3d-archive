#!/usr/bin/env python3
"""
@demicstory 1:1 Basic Format + 1 Signature Visual Differentiator (특이점):
- Basic Format (kept 100% intact):
  * Full-bleed 9:16 dark charcoal studio (#14161B), sequential scenes (Macro Hook -> Wrong -> Correct)
  * Zero text, zero rectangular boxes
  * Top-center 3-thumbnail translucent pill
  * Top-right circular translucent ring with glowing red '✕' / green '✓'
  * 3D glowing directional arrow
- +1 Signature Differentiator (특이점):
  * "2.2x Live X-Ray Disc Magnifier Loupe + Plumb-Line Laser Axis" (관절 2.2배 확대 X-Ray 돋보기 링 & 중력 정렬 레이저 축):
    1) A crisp vertical gravity plumb-line passing up from the shoulder joint, visually exposing the forward head drift (red offset arc & shear zone) in Wrong mode, and snapping into a locked neon-green axial laser line in Correct mode.
    2) A floating circular 2.2x X-Ray Magnifier Loupe connected to C5-C6 that magnifies the actual cervical vertebrae & nerve root, showing red disc compression pulses vs green disc decompression.
"""
import os
import math
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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

OUT_MP4 = os.path.join(OUT_DIR, "demic_plus1_day1.mp4")
OUT_KEYFRAME = os.path.join(OUT_DIR, "demic_plus1_day1_keyframes.jpg")
OUT_DIFF_COMP = os.path.join(OUT_DIR, "demic_basic_vs_plus1.jpg")


def draw_signature_differentiator(
    canvas_rgba: Image.Image,
    plate_bgr: np.ndarray,
    is_correct: bool,
    t: float,
    pulse: float,
    appear_prog: float,
):
    """
    Draws the ONE Signature Visual Differentiator (특이점) on top of the @demicstory Basic layout:
    A zero-text 'Plumb-Line Laser Axis + 2.2x Live X-Ray Joint Magnifier Loupe'.
    """
    if appear_prog <= 0.02:
        return
    alpha_scale = min(1.0, appear_prog)

    # Key anatomical coordinates on the 1080x1920 full-screen plate
    # Shoulder acromion / C7 base anchor:
    base_x, base_y = 548, 1085
    if not is_correct:
        # Forward head posture: ear/upper cervical is drifted forward (leftward in profile)
        head_x, head_y = 435, 650
        c5_x, c5_y = 488, 865
        col = (255, 58, 42)
        bright_col = (255, 195, 180)
    else:
        # Neutral aligned posture: ear/upper cervical aligns vertically over C7/shoulder axis
        head_x, head_y = 536, 645
        c5_x, c5_y = 542, 855
        col = (35, 255, 115)
        bright_col = (210, 255, 225)

    # -------------------------------------------------------------------------
    # PART A: Gravity Plumb-Line Laser Axis & Deviation Wedge (Zero Text)
    # -------------------------------------------------------------------------
    axis_glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    agd = ImageDraw.Draw(axis_glow)
    axis_core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    acd = ImageDraw.Draw(axis_core)

    # Vertical reference gravity line (dashed white/cyan reference line straight up from base_x, base_y)
    ref_top_y = 560
    dash_len, gap_len = 16, 12
    y_cur = ref_top_y
    while y_cur < base_y:
        y_next = min(base_y, y_cur + dash_len)
        acd.line(
            [(base_x, y_cur), (base_x, y_next)],
            fill=(210, 230, 255, int(135 * alpha_scale)),
            width=3,
        )
        y_cur += dash_len + gap_len

    # Filled translucent deviation wedge between vertical reference line and actual neck axis
    wedge_pts = [(base_x, base_y), (base_x, head_y), (head_x, head_y)]
    wedge_alpha = int((65 + 35 * pulse) * alpha_scale) if not is_correct else int(40 * alpha_scale)
    acd.polygon(wedge_pts, fill=(*col, wedge_alpha))

    # Actual cervical posture laser line from base to head
    agd.line([(base_x, base_y), (head_x, head_y)], fill=(*col, int(190 * alpha_scale)), width=14)
    acd.line([(base_x, base_y), (head_x, head_y)], fill=(*col, int(255 * alpha_scale)), width=5)
    acd.line([(base_x, base_y), (head_x, head_y)], fill=(*bright_col, int(230 * alpha_scale)), width=2)

    # Glowing geometric joint nodes at (base_x, base_y), (c5_x, c5_y), (head_x, head_y)
    for jx, jy, jr in [(base_x, base_y, 10), (c5_x, c5_y, 13), (head_x, head_y, 10)]:
        agd.ellipse([jx - jr - 6, jy - jr - 6, jx + jr + 6, jy + jr + 6], fill=(*col, int(180 * alpha_scale)))
        acd.ellipse([jx - jr, jy - jr, jx + jr, jy + jr], fill=(*col, int(255 * alpha_scale)))
        acd.ellipse([jx - 4, jy - 4, jx + 4, jy + 4], fill=(255, 255, 255, int(255 * alpha_scale)))

    # Expanding shockwave rings on C5-C6 stress point in Wrong mode (or calm alignment ring in Correct mode)
    wave_phase = (t * 1.6) % 1.0
    wr = int(16 + 44 * wave_phase)
    wa = int((1.0 - wave_phase) * 195 * alpha_scale)
    acd.ellipse([c5_x - wr, c5_y - wr, c5_x + wr, c5_y + wr], outline=(*col, wa), width=3)

    axis_glow = axis_glow.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(axis_glow)
    canvas_rgba.alpha_composite(axis_core)

    # -------------------------------------------------------------------------
    # PART B: Floating 2.2x Live X-Ray Magnifier Loupe (Right Side, Zero Text)
    # -------------------------------------------------------------------------
    loupe_cx, loupe_cy = 855, 1280
    loupe_r = 152

    # 1. Extract 2.2x magnified crop around (c5_x, c5_y) from plate_bgr
    src_r = int(loupe_r / 2.15)  # ~70px radius -> 2.15x magnification
    x0 = max(0, c5_x - src_r)
    y0 = max(0, c5_y - src_r)
    x1 = min(W, c5_x + src_r)
    y1 = min(H, c5_y + src_r)
    crop_bgr = plate_bgr[y0:y1, x0:x1].copy()

    # Enhance internal X-ray vertebral contrast inside the loupe
    crop_zoom = cv2.resize(crop_bgr, (loupe_r * 2, loupe_r * 2), interpolation=cv2.INTER_CUBIC)
    # Apply subtle dark vignette near lens edge + boost bone/glow contrast
    crop_f = crop_zoom.astype(np.float32) * 1.18
    yy, xx = np.mgrid[0 : loupe_r * 2, 0 : loupe_r * 2].astype(np.float32)
    dist = np.sqrt((xx - loupe_r) ** 2 + (yy - loupe_r) ** 2) / loupe_r
    vig = np.clip(1.08 - 0.45 * (dist ** 2), 0.55, 1.05)[:, :, None]
    crop_f = np.clip(crop_f * vig, 0, 255).astype(np.uint8)
    loupe_img = Image.fromarray(cv2.cvtColor(crop_f, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Draw subtle medical crosshair & vertebral compression/decompression chevrons inside the loupe (purely graphical, zero text)
    ld = ImageDraw.Draw(loupe_img)
    c = loupe_r
    # Crosshair ticks around the perimeter of the lens
    for deg in range(0, 360, 30):
        rad = math.radians(deg)
        r_in = loupe_r - (16 if deg % 90 == 0 else 9)
        r_out = loupe_r - 3
        ld.line(
            [
                (c + r_in * math.cos(rad), c + r_in * math.sin(rad)),
                (c + r_out * math.cos(rad), c + r_out * math.sin(rad)),
            ],
            fill=(255, 255, 255, 130 if deg % 90 == 0 else 65),
            width=2 if deg % 90 == 0 else 1,
        )

    # Inside the loupe: animated graphical disc-pinch arrows (Wrong) vs disc-decompression arrows (Correct)
    p_shift = int(6 * pulse)
    if not is_correct:
        # Two red inward-pinching chevrons squeezing the cervical disc
        # Top chevron pointing DOWN into the disc
        y_top_ch = c - 36 + p_shift
        ld.line([(c - 18, y_top_ch - 14), (c, y_top_ch), (c + 18, y_top_ch - 14)], fill=(255, 80, 65, 240), width=5, joint="curve")
        # Bottom chevron pointing UP into the disc
        y_bot_ch = c + 36 - p_shift
        ld.line([(c - 18, y_bot_ch + 14), (c, y_bot_ch), (c + 18, y_bot_ch + 14)], fill=(255, 80, 65, 240), width=5, joint="curve")
    else:
        # Two neon-green outward-opening chevrons showing disc space relief/decompression
        y_top_ch = c - 30 - p_shift
        ld.line([(c - 18, y_top_ch + 14), (c, y_top_ch), (c + 18, y_top_ch + 14)], fill=(60, 255, 135, 240), width=5, joint="curve")
        y_bot_ch = c + 30 + p_shift
        ld.line([(c - 18, y_bot_ch - 14), (c, y_bot_ch), (c + 18, y_bot_ch - 14)], fill=(60, 255, 135, 240), width=5, joint="curve")

    # Circular mask for the loupe
    mask = Image.new("L", (loupe_r * 2, loupe_r * 2), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([0, 0, loupe_r * 2 - 1, loupe_r * 2 - 1], fill=int(255 * alpha_scale))

    # Outer glow and callout connector line from C5-C6 to the loupe
    loupe_glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    lgd = ImageDraw.Draw(loupe_glow)
    loupe_ring = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    lrd = ImageDraw.Draw(loupe_ring)

    # Elbow connector line from (c5_x, c5_y) -> (655, 1110) -> edge of loupe
    ang_to_c5 = math.atan2(c5_y - loupe_cy, c5_x - loupe_cx)
    edge_x = int(loupe_cx + (loupe_r + 2) * math.cos(ang_to_c5))
    edge_y = int(loupe_cy + (loupe_r + 2) * math.sin(ang_to_c5))
    mid_x, mid_y = 655, 1025

    lgd.line([(c5_x, c5_y), (mid_x, mid_y), (edge_x, edge_y)], fill=(*col, int(150 * alpha_scale)), width=10, joint="curve")
    lrd.line([(c5_x, c5_y), (mid_x, mid_y), (edge_x, edge_y)], fill=(*col, int(235 * alpha_scale)), width=3, joint="curve")

    # Small target ring around C5-C6 origin
    lrd.ellipse([c5_x - 24, c5_y - 24, c5_x + 24, c5_y + 24], outline=(255, 255, 255, int(190 * alpha_scale)), width=2)

    # Loupe outer neon halo & crisp double rim
    lgd.ellipse(
        [loupe_cx - loupe_r - 12, loupe_cy - loupe_r - 12, loupe_cx + loupe_r + 12, loupe_cy + loupe_r + 12],
        outline=(*col, int((160 + 60 * pulse) * alpha_scale)),
        width=14,
    )
    loupe_glow = loupe_glow.filter(ImageFilter.GaussianBlur(radius=12))
    canvas_rgba.alpha_composite(loupe_glow)

    # Paste magnified X-ray lens
    canvas_rgba.paste(loupe_img, (loupe_cx - loupe_r, loupe_cy - loupe_r), mask)

    # Crisp glass rim + active colored ring
    lrd.ellipse(
        [loupe_cx - loupe_r, loupe_cy - loupe_r, loupe_cx + loupe_r, loupe_cy + loupe_r],
        outline=(*col, int(245 * alpha_scale)),
        width=5,
    )
    lrd.ellipse(
        [loupe_cx - loupe_r - 5, loupe_cy - loupe_r - 5, loupe_cx + loupe_r + 5, loupe_cy + loupe_r + 5],
        outline=(235, 245, 255, int(150 * alpha_scale)),
        width=2,
    )
    canvas_rgba.alpha_composite(loupe_ring)


def main():
    wrong_plate, correct_plate = extract_fullscreen_9x16_plates(S2_NECK)
    thumbs = build_minimap_thumbnails(S1_MASTER)

    tmp_dir = "/workspace/daily_info_reels/frames_plus1"
    os.makedirs(tmp_dir, exist_ok=True)

    saved_keyframes = {}

    for f in range(TOTAL_FRAMES):
        t = f / FPS
        pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.0 * t)

        if t < 4.6:
            plate = animate_plate(wrong_plate, t, is_wrong=True)
            is_corr = False
        elif t < 5.0:
            alpha = (t - 4.6) / 0.4
            p_w = animate_plate(wrong_plate, t, is_wrong=True).astype(np.float32)
            p_c = animate_plate(correct_plate, t, is_wrong=False).astype(np.float32)
            plate = (p_w * (1.0 - alpha) + p_c * alpha).astype(np.uint8)
            is_corr = (alpha >= 0.5)
        else:
            plate = animate_plate(correct_plate, t, is_wrong=False)
            is_corr = True

        # Camera movement (0.0s - 1.8s: macro X-ray zoom hook -> full 9:16 frame)
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

        # 1. @demicstory Basic 3D directional arrows
        if 2.0 <= t < 4.6:
            prog = min(1.0, (t - 2.0) / 0.6)
            draw_demic_3d_arrow(frame_rgba, (350, 690), (265, 760), (255, 65, 45), prog)
        elif t >= 5.1:
            prog = min(1.0, (t - 5.1) / 0.6)
            draw_demic_3d_arrow(frame_rgba, (280, 695), (395, 695), (40, 255, 115), prog)
            draw_demic_3d_arrow(frame_rgba, (575, 1020), (575, 865), (40, 255, 115), prog)

        # 2. +1 Signature Differentiator (특이점): Live 2.2x X-Ray Magnifier Loupe + Plumb-Line Laser Axis
        if 1.8 <= t < 4.6:
            diff_prog = min(1.0, (t - 1.8) / 0.45)
            draw_signature_differentiator(frame_rgba, plate, is_correct=False, t=t, pulse=pulse, appear_prog=diff_prog)
        elif t >= 5.0:
            diff_prog = min(1.0, (t - 5.0) / 0.45)
            draw_signature_differentiator(frame_rgba, plate, is_correct=True, t=t, pulse=pulse, appear_prog=diff_prog)

        # 3. @demicstory Basic Top-Center 3-Thumbnail Pill & Top-Right Status Badge
        if t >= 1.0:
            draw_demic_minimap_pill(frame_rgba, thumbs, is_correct=is_corr, pulse=pulse)
            draw_demic_status_badge(frame_rgba, is_correct=is_corr, pulse=pulse)

        rgb_out = frame_rgba.convert("RGB")
        rgb_out.save(f"{tmp_dir}/f_{f:04d}.jpg", quality=94)

        if f in [int(0.4 * FPS), int(2.8 * FPS), int(6.2 * FPS)]:
            saved_keyframes[f] = rgb_out.copy()

    wav_path = f"{tmp_dir}/audio.wav"
    synthesize_demic_audio(wav_path)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", f"{tmp_dir}/f_%04d.jpg",
        "-i", wav_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        OUT_MP4,
    ]
    subprocess.run(cmd, check=True)

    # Save 3-keyframe strip
    k_keys = sorted(saved_keyframes.keys())
    strip = Image.new("RGB", (W * 3, H))
    for i, k in enumerate(k_keys):
        strip.paste(saved_keyframes[k], (i * W, 0))
    strip.resize((1620, 960), Image.Resampling.LANCZOS).save(OUT_KEYFRAME, quality=92)

    # Save Basic vs +1 Differentiator comparison (Wrong & Correct side by side)
    basic_kf = Image.open(os.path.join(OUT_DIR, "demic_1to1_day1_keyframes.jpg"))
    # basic_kf is 1620x960 (3 panels of 540x960)
    basic_wrong = basic_kf.crop((540, 0, 1080, 960))
    basic_corr = basic_kf.crop((1080, 0, 1620, 960))
    plus1_wrong = saved_keyframes[k_keys[1]].resize((540, 960), Image.Resampling.LANCZOS)
    plus1_corr = saved_keyframes[k_keys[2]].resize((540, 960), Image.Resampling.LANCZOS)

    comp = Image.new("RGB", (540 * 4, 960))
    comp.paste(basic_wrong, (0, 0))
    comp.paste(plus1_wrong, (540, 0))
    comp.paste(basic_corr, (1080, 0))
    comp.paste(plus1_corr, (1620, 0))
    comp.save(OUT_DIFF_COMP, quality=92)
    print("DONE PLUS1:", OUT_MP4, OUT_KEYFRAME, OUT_DIFF_COMP)


if __name__ == "__main__":
    main()
