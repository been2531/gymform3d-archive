#!/usr/bin/env python3
"""
1:1 Exact Replica of @demicstory Visual Format (Zero Text, Full-Screen 9:16 Sequential Scenes).
Matches De77y3KCRvi.mp4 & De1ZjBfDzns.mp4 frame-for-frame in layout, lighting, and minimal UI:
1. Full-bleed 9:16 dark charcoal studio (#14161B), NEVER a top/bottom split screen.
2. Shot 1 (0.0s - 1.8s): Macro X-ray zoom on glowing orange-red cervical vertebrae & trapezius fibers, pulling back.
3. Shot 2 (1.8s - 4.8s): Full-screen 9:16 Wrong Posture — pulsing orange-red muscle/spine glow, cool electric-blue rim light, top-center 3-thumbnail translucent pill (orange square border on active thumb, zero text), top-right small translucent white ring with glowing red '✕'.
4. Shot 3 (4.8s - 8.5s): Full-screen 9:16 Correct Posture — vibrant neon-green spine/shoulder glow, single clean glowing green directional arrow for corrective path, top-center 3-thumbnail pill, top-right small translucent white ring with glowing green '✓'.
5. ZERO text, ZERO numbers, ZERO rectangular callout boxes.
"""
import os
import math
import wave
import struct
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1080, 1920
FPS = 30
DURATION = 8.5
TOTAL_FRAMES = int(FPS * DURATION)

S1_MASTER = "/workspace/daily_info_reels/sheet_r1/s1_master_sheet.png"
S2_NECK = "/workspace/daily_info_reels/sheet_r1/s2_rehab_neck.png"
OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MP4 = os.path.join(OUT_DIR, "demic_1to1_day1.mp4")
OUT_COMP = os.path.join(OUT_DIR, "demic_1to1_vs_original.jpg")
OUT_KEYFRAME = os.path.join(OUT_DIR, "demic_1to1_day1_keyframes.jpg")


def extract_fullscreen_9x16_plates(s2_path: str):
    """
    Extracts the Top panel (Wrong form) and Bottom panel (Correct form) from s2_rehab_neck.png (5120x2880)
    and converts each into a seamless FULL-SCREEN 9:16 (1080x1920) dark charcoal studio plate (#14161B).
    """
    raw = cv2.imread(s2_path)
    if raw is None:
        raise FileNotFoundError(s2_path)
    rh, rw = raw.shape[:2]  # 5120, 2880

    # Top panel is roughly y=40..2510, Bottom panel is y=2590..5080
    # Let's crop away any outer border and center the figure in a full 9:16 vertical studio frame
    top_panel = raw[60:2500, 120:2760]   # 2440 x 2640
    bot_panel = raw[2620:5060, 120:2760] # 2440 x 2640

    def make_full_9x16(panel_bgr, is_wrong=True):
        ph, pw = panel_bgr.shape[:2]
        # Resize panel so width is 1080 (or slightly wider 1160 cropped to 1080)
        scale = 1140.0 / pw
        nh = int(ph * scale)
        nw = 1140
        resized = cv2.resize(panel_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = (nw - W) // 2
        body = resized[:, x0:x0 + W]  # nh x 1080 (nh is ~1053)

        # Clean stray green from wrong panel or stray red from correct panel
        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        if is_wrong:
            mask_stray = cv2.inRange(hsv, (35, 60, 50), (95, 255, 255))
        else:
            mask_stray = cv2.inRange(hsv, (0, 90, 70), (22, 255, 255)) | cv2.inRange(hsv, (165, 90, 70), (180, 255, 255))
        if np.count_nonzero(mask_stray) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask_stray = cv2.dilate(mask_stray, k, iterations=1)
            body = cv2.inpaint(body, mask_stray, 7, cv2.INPAINT_TELEA)

        # Now place `body` (height ~1053) seamlessly inside a full 1920px tall #14161B studio canvas
        # Position body vertically around y_top = 440 so the head/neck/shoulders fill the center of the 9:16 screen
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        y_top = 430
        y_bot = y_top + nh

        # Top studio background extension: sample top 40 rows of body and gradient-fade upward to #14161B
        top_sample = cv2.GaussianBlur(body[10:50, :], (51, 51), 0)
        top_ext = cv2.resize(top_sample, (W, y_top), interpolation=cv2.INTER_LINEAR)
        studio_dark = np.array([27, 22, 20], dtype=np.float32)  # BGR for #14161B
        t_alpha = np.linspace(0.0, 1.0, y_top)[:, None, None] ** 1.4
        top_blend = (studio_dark * (1.0 - t_alpha) + top_ext.astype(np.float32) * t_alpha).astype(np.uint8)

        # Bottom studio extension: sample bottom 40 rows of body and gradient-fade downward so torso continues naturally
        rem_bot = H - y_bot
        bot_sample = cv2.GaussianBlur(body[-55:-15, :], (31, 31), 0)
        bot_ext = cv2.resize(bot_sample, (W, rem_bot), interpolation=cv2.INTER_LINEAR)
        b_alpha = np.linspace(1.0, 0.08, rem_bot)[:, None, None] ** 1.3
        bot_blend = (studio_dark * (1.0 - b_alpha) + bot_ext.astype(np.float32) * b_alpha).astype(np.uint8)

        canvas[:y_top] = top_blend
        canvas[y_top:y_bot] = body
        canvas[y_bot:] = bot_blend

        # Feather the seam regions at y_top and y_bot so there is zero visible horizontal line
        for seam_y in [y_top, y_bot]:
            band = 48
            s0, s1 = max(0, seam_y - band), min(H, seam_y + band)
            blurred = cv2.GaussianBlur(canvas[s0:s1], (1, 45), 0)
            w_curve = (1.0 - np.abs(np.linspace(-1.0, 1.0, s1 - s0)))[:, None, None]
            canvas[s0:s1] = (canvas[s0:s1].astype(np.float32) * (1.0 - w_curve) + blurred.astype(np.float32) * w_curve).astype(np.uint8)

        return canvas

    wrong_plate = make_full_9x16(top_panel, is_wrong=True)
    correct_plate = make_full_9x16(bot_panel, is_wrong=False)
    return wrong_plate, correct_plate


def build_minimap_thumbnails(s1_path: str):
    """
    Extracts 3 square miniature thumbnails from s1_master_sheet.png (profile, front, back)
    for @demicstory's signature top-center translucent pill (ZERO text).
    """
    raw = cv2.imread(s1_path)
    rh, rw = raw.shape[:2]  # 5120 x 2880
    # Three figures sit at x: [0..960], [960..1920], [1920..2880], upper body y: [500..2300]
    thumbs = []
    x_ranges = [(80, 960), (1000, 1880), (1920, 2800)]
    for x0, x1 in x_ranges:
        crop = raw[520:1760, x0:x1]  # upper body + neck/head square-ish crop
        ch, cw = crop.shape[:2]
        side = min(ch, cw)
        cy0 = (ch - side) // 2
        cx0 = (cw - side) // 2
        sq = crop[cy0:cy0 + side, cx0:cx0 + side]
        sq_rgb = cv2.cvtColor(cv2.resize(sq, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        thumbs.append(Image.fromarray(sq_rgb))
    return thumbs


def animate_plate(base_bgr: np.ndarray, t: float, is_wrong: bool) -> np.ndarray:
    """
    Applies subtle 2.5D biomechanical posture breathing/movement + @demicstory's signature
    pulsing anatomical glow & electric-blue rim highlight.
    """
    out = base_bgr.astype(np.float32)

    # Subtle 2.5D head/neck displacement field around the head/neck zone (y=460..1150, x=200..900)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = 560.0, 780.0
    dist2 = ((xx - cx) / 280.0) ** 2 + ((yy - cy) / 310.0) ** 2
    influence = np.exp(-dist2 * 1.4).astype(np.float32)

    if is_wrong:
        # Forward head droop / strain oscillation
        phase = 0.5 + 0.5 * math.sin(2.0 * math.pi * 0.85 * t)
        dx = -12.0 * phase * influence
        dy = 10.0 * phase * influence
    else:
        # Smooth chin-tuck retraction & axial elongation settling into place
        local_t = max(0.0, t - 4.8)
        settle = 1.0 - math.exp(-local_t * 2.2)
        breath = 0.15 * math.sin(2.0 * math.pi * 0.6 * local_t)
        dx = 14.0 * (1.0 - settle + breath) * influence
        dy = -8.0 * settle * influence

    map_x = (xx - dx).astype(np.float32)
    map_y = (yy - dy).astype(np.float32)
    warped = cv2.remap(base_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = warped.astype(np.float32)

    # Dynamic anatomical glow pulse (orange-red for wrong, neon-green for correct)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    if is_wrong:
        glow_mask = cv2.inRange(hsv, (0, 80, 80), (26, 255, 255)) | cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
        pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.1 * t)
        boost_color = np.array([15.0, 85.0, 255.0], dtype=np.float32)  # BGR bright orange-red
        strength = 0.25 + 0.42 * pulse
    else:
        glow_mask = cv2.inRange(hsv, (35, 65, 65), (98, 255, 255))
        pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 1.4 * t)
        boost_color = np.array([110.0, 255.0, 35.0], dtype=np.float32)  # BGR vibrant neon green
        strength = 0.22 + 0.35 * pulse

    glow_float = cv2.GaussianBlur(glow_mask.astype(np.float32) / 255.0, (45, 45), 0)[:, :, None]
    out = np.clip(out + glow_float * boost_color * strength, 0, 255)

    return out.astype(np.uint8)


def draw_demic_status_badge(canvas_rgba: Image.Image, is_correct: bool, pulse: float):
    """
    Draws @demicstory's exact top-right status icon (seen in De77y3KCRvi_t4s.png & De1ZjBfDzns_t4s.png):
    - A small, clean translucent white ring (center=(955, 195), radius=46px)
    - Inside: a glowing red '✕' (wrong) or glowing neon-green '✓' (correct) with a soft outer halo.
    - ZERO text.
    """
    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    cx, cy, r = 955, 195, 46

    # Soft outer halo
    halo_layer = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo_layer)
    if not is_correct:
        halo_col = (255, 45, 35, int(95 + 45 * pulse))
        icon_col = (255, 55, 45, 255)
    else:
        halo_col = (30, 255, 110, int(95 + 45 * pulse))
        icon_col = (40, 255, 115, 255)

    hd.ellipse([cx - r - 14, cy - r - 14, cx + r + 14, cy + r + 14], fill=halo_col)
    halo_layer = halo_layer.filter(ImageFilter.GaussianBlur(radius=16))
    canvas_rgba.alpha_composite(halo_layer)

    # Dark translucent inner circle + thin translucent white ring (exact @demicstory look)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 24, 30, 165), outline=(235, 242, 255, 190), width=3)

    # Inner glowing symbol ('✕' or '✓')
    sym_glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    sgd = ImageDraw.Draw(sym_glow)
    if not is_correct:
        s = 19
        sgd.line([(cx - s, cy - s), (cx + s, cy + s)], fill=icon_col, width=8)
        sgd.line([(cx - s, cy + s), (cx + s, cy - s)], fill=icon_col, width=8)
        d.line([(cx - s, cy - s), (cx + s, cy + s)], fill=(255, 190, 180, 255), width=4)
        d.line([(cx - s, cy + s), (cx + s, cy - s)], fill=(255, 190, 180, 255), width=4)
    else:
        pts = [(cx - 21, cy + 2), (cx - 5, cy + 18), (cx + 23, cy - 15)]
        sgd.line(pts, fill=icon_col, width=9, joint="curve")
        d.line(pts, fill=(200, 255, 215, 255), width=4, joint="curve")

    sym_glow = sym_glow.filter(ImageFilter.GaussianBlur(radius=6))
    canvas_rgba.alpha_composite(sym_glow)
    canvas_rgba.alpha_composite(overlay)


def draw_demic_minimap_pill(canvas_rgba: Image.Image, thumbs: list, is_correct: bool, pulse: float):
    """
    Draws @demicstory's top-center floating translucent pill containing 3 square miniature
    3D mannequin thumbnails, with the active thumbnail highlighted by a glowing orange/green square border.
    ZERO text inside or outside the pill.
    """
    pill_w, pill_h = 348, 116
    px0 = (W - pill_w) // 2
    py0 = 96

    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Frosted dark translucent rounded pill container
    d.rounded_rectangle(
        [px0, py0, px0 + pill_w, py0 + pill_h],
        radius=24,
        fill=(22, 26, 34, 185),
        outline=(255, 255, 255, 55),
        width=2,
    )
    canvas_rgba.alpha_composite(overlay)

    # Paste the 3 square thumbnails (96x96 each, spaced by 14px)
    pad_x = 16
    gap = 14
    ty = py0 + (pill_h - 96) // 2
    for i, th in enumerate(thumbs):
        tx = px0 + pad_x + i * (96 + gap)
        # Round corners of thumbnail
        mask = Image.new("L", (96, 96), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle([0, 0, 95, 95], radius=14, fill=255)
        canvas_rgba.paste(th, (tx, ty), mask)

        # Border around thumbnail: index 0 (side profile) is the active view!
        b_overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(b_overlay)
        if i == 0:
            active_col = (40, 255, 115, 245) if is_correct else (255, 115, 30, 245)
            bd.rounded_rectangle([tx - 2, ty - 2, tx + 98, ty + 98], radius=15, outline=active_col, width=4)
        else:
            bd.rounded_rectangle([tx, ty, tx + 96, ty + 96], radius=14, outline=(255, 255, 255, 45), width=1)
        canvas_rgba.alpha_composite(b_overlay)


def draw_demic_3d_arrow(canvas_rgba: Image.Image, p0: tuple, p1: tuple, color_rgb: tuple, progress: float):
    """
    Draws a single clean glowing 3D-style directional arrow (zero text) showing the biomechanics vector.
    """
    if progress <= 0.05:
        return
    prog = min(1.0, progress)
    x0, y0 = p0
    x1 = x0 + (p1[0] - x0) * prog
    y1 = y0 + (p1[1] - y0) * prog

    angle = math.atan2(y1 - y0, x1 - x0)
    head_len = 34
    head_ang = 0.52
    hx1 = x1 - head_len * math.cos(angle - head_ang)
    hy1 = y1 - head_len * math.sin(angle - head_ang)
    hx2 = x1 - head_len * math.cos(angle + head_ang)
    hy2 = y1 - head_len * math.sin(angle + head_ang)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line([(x0, y0), (x1, y1)], fill=(*color_rgb, 180), width=18, joint="curve")
    gd.polygon([(x1, y1), (hx1, hy1), (hx2, hy2)], fill=(*color_rgb, 200))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    canvas_rgba.alpha_composite(glow)

    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    cd.line([(x0, y0), (x1, y1)], fill=(*color_rgb, 255), width=9, joint="curve")
    cd.line([(x0, y0), (x1, y1)], fill=(235, 255, 240, 220), width=3, joint="curve")
    cd.polygon([(x1, y1), (hx1, hy1), (hx2, hy2)], fill=(235, 255, 240, 255))
    canvas_rgba.alpha_composite(core)


def synthesize_demic_audio(wav_path: str):
    """
    Synthesizes subtle, tactile medical 3D sound effects matching @demicstory:
    - 0.0s: deep X-ray sub-bass zoom whoosh
    - 1.8s: low warning pulse tone on wrong posture reveal
    - 4.8s: crisp mechanical transition whoosh + harmonic emerald chime on correct alignment
    """
    sr = 44100
    n_samples = int(sr * DURATION)
    audio = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32) / sr

    # Ambient dark studio drone
    drone = 0.025 * np.sin(2 * np.pi * 55.0 * t) + 0.015 * np.sin(2 * np.pi * 110.0 * t)
    audio += drone

    # 0.0s - 1.5s: X-ray zoom pull-back whoosh
    m_whoosh = (t >= 0.0) & (t < 1.5)
    tw = t[m_whoosh]
    env_w = np.sin(np.pi * tw / 1.5) ** 1.5
    freq_w = 180.0 - 90.0 * (tw / 1.5)
    audio[m_whoosh] += 0.14 * env_w * np.sin(2 * np.pi * freq_w * tw)

    # 1.8s, 2.8s, 3.8s: subtle red tension heartbeat pulses
    for pt in [1.8, 2.8, 3.8]:
        mp = (t >= pt) & (t < pt + 0.35)
        tp = t[mp] - pt
        env_p = np.exp(-tp * 10.0)
        audio[mp] += 0.16 * env_p * np.sin(2 * np.pi * 92.0 * tp)

    # 4.8s: Transition whoosh + clean green alignment chime (528Hz + 792Hz)
    mc = (t >= 4.75) & (t < 6.0)
    tc = t[mc] - 4.75
    env_c = np.exp(-tc * 3.2) * (1.0 - np.exp(-tc * 40.0))
    audio[mc] += 0.18 * env_c * (np.sin(2 * np.pi * 528.0 * tc) + 0.6 * np.sin(2 * np.pi * 792.0 * tc))

    audio = np.clip(audio, -0.95, 0.95)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main():
    wrong_plate, correct_plate = extract_fullscreen_9x16_plates(S2_NECK)
    thumbs = build_minimap_thumbnails(S1_MASTER)

    tmp_dir = "/workspace/daily_info_reels/frames_1to1"
    os.makedirs(tmp_dir, exist_ok=True)

    saved_keyframes = {}

    for f in range(TOTAL_FRAMES):
        t = f / FPS
        pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.0 * t)

        if t < 4.6:
            # Phase 1 & 2: Wrong Form (Full-Screen 9:16)
            plate = animate_plate(wrong_plate, t, is_wrong=True)
            is_corr = False
        elif t < 5.0:
            # Crisp 0.4s cross-dissolve from Wrong full-screen to Correct full-screen
            alpha = (t - 4.6) / 0.4
            p_w = animate_plate(wrong_plate, t, is_wrong=True).astype(np.float32)
            p_c = animate_plate(correct_plate, t, is_wrong=False).astype(np.float32)
            plate = (p_w * (1.0 - alpha) + p_c * alpha).astype(np.uint8)
            is_corr = (alpha >= 0.5)
        else:
            # Phase 3: Correct Form (Full-Screen 9:16)
            plate = animate_plate(correct_plate, t, is_wrong=False)
            is_corr = True

        # Camera movement:
        # 0.0s - 1.8s: @demicstory's signature tight macro X-ray close-up on the glowing cervical spine,
        # smoothly pulling back to the full 9:16 studio framing!
        if t < 1.8:
            u = t / 1.8
            # Smooth ease-out cubic
            ease = 1.0 - (1.0 - u) ** 3
            zoom = 1.72 - 0.72 * ease
            # Focus center on cervical spine (x=540, y=820) pulling back to (540, 960)
            focus_x = 540.0
            focus_y = 820.0 * (1.0 - ease) + 960.0 * ease
        else:
            # Subtle slow studio breathing push-in (1.00 -> 1.05)
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

        # Draw @demicstory's minimal 3D directional arrow (zero text)
        if 2.0 <= t < 4.6:
            # Red downward/forward strain vector on forward head posture
            prog = min(1.0, (t - 2.0) / 0.6)
            draw_demic_3d_arrow(frame_rgba, (350, 690), (265, 760), (255, 65, 45), prog)
        elif t >= 5.1:
            # Green horizontal chin-tuck & upward axial alignment vector
            prog = min(1.0, (t - 5.1) / 0.6)
            draw_demic_3d_arrow(frame_rgba, (280, 695), (395, 695), (40, 255, 115), prog)
            draw_demic_3d_arrow(frame_rgba, (575, 1020), (575, 865), (40, 255, 115), prog)

        # Draw @demicstory's top-center 3-thumbnail translucent pill & top-right circular status badge
        # (Fade in smoothly after the initial 1.2s macro X-ray hook, just like De77y3KCRvi.mp4!)
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

    # Build 3-keyframe progression strip (Hook -> Wrong -> Correct)
    k_keys = sorted(saved_keyframes.keys())
    strip = Image.new("RGB", (W * 3, H))
    for i, k in enumerate(k_keys):
        strip.paste(saved_keyframes[k], (i * W, 0))
    strip_small = strip.resize((1620, 960), Image.Resampling.LANCZOS)
    strip_small.save(OUT_KEYFRAME, quality=92)

    # Build Side-by-Side Comparison with @demicstory original frames (De77y3KCRvi & De1ZjBfDzns)
    orig_wrong = Image.open("/workspace/daily_info_reels/demic_bench/frames/De77y3KCRvi_t4s.png").convert("RGB").resize((540, 960), Image.Resampling.LANCZOS)
    orig_corr = Image.open("/workspace/daily_info_reels/demic_bench/frames/De1ZjBfDzns_t4s.png").convert("RGB").resize((540, 960), Image.Resampling.LANCZOS)
    our_wrong = saved_keyframes[k_keys[1]].resize((540, 960), Image.Resampling.LANCZOS)
    our_corr = saved_keyframes[k_keys[2]].resize((540, 960), Image.Resampling.LANCZOS)

    comp = Image.new("RGB", (540 * 4, 960))
    comp.paste(orig_wrong, (0, 0))
    comp.paste(our_wrong, (540, 0))
    comp.paste(orig_corr, (1080, 0))
    comp.paste(our_corr, (1620, 0))
    comp.save(OUT_COMP, quality=92)
    print("DONE:", OUT_MP4, OUT_KEYFRAME, OUT_COMP)


if __name__ == "__main__":
    main()
