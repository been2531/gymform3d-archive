#!/usr/bin/env python3
"""
@gymform.3d — Day 1: Cable Rope Triceps Pushdown (v4)
Incorporating 중현 님's feedback from https://www.instagram.com/p/De77WaXAiXd/:
1. Longer, unhurried 16.0s duration (480 frames @ 30fps) with multi-checkpoint step-by-step coaching.
2. Clear O / X (✓ / ✕) posture & joint-angle information delivery:
   - Checkpoint 1 (0.0–4.2s): Torso & Spine Angle Alignment Line (✕ Hunched/Leaning vs ✓ Locked Upright Angle)
   - Checkpoint 2 (4.2–8.4s): Elbow Hinge & Wrist Axis Alignment Line (✕ Flared Elbow & Bent Wrist vs ✓ Pinned Elbow & Neutral Wrist)
   - Checkpoint 3 (8.4–16.0s): Full-ROM Execution Comparison (✕ Force Leak Rep vs ✓ 2 Full Peak-Contraction Reps with Rope Spread)
3. Both on-body Alignment Line Graphics (선 그래픽) like De77WaXAiXd AND a synchronized Bottom Real-Time Biomechanical Line Graph (선 그래프) with live tracking playhead comparing ✕ vs ✓ tension curves + 5-bar gauge.
4. Sculpted 4x-supersampled 3D volumetric tapered arrows.
"""
import os
import math
import wave
import struct
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1080, 1920
FPS = 30
DURATION = 16.0
TOTAL_FRAMES = int(FPS * DURATION)  # 480 frames

OUT_DIR = "/workspace/daily_info_reels/output"
WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")
TMP_DIR = "/workspace/daily_info_reels/frames_gym_day1"

OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")
OUT_KEYFRAMES = os.path.join(OUT_DIR, "demic_gym_day1_keyframes.jpg")
OUT_COMP = os.path.join(OUT_DIR, "demic_gym_vs_original.jpg")


def load_i2v_frames(mp4_path: str, fallback_jpg: str) -> list:
    frames = []
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            up = cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4)
            blur = cv2.GaussianBlur(up, (0, 0), 1.1)
            sharp = cv2.addWeighted(up, 1.22, blur, -0.22, 0)
            frames.append(sharp)
        cap.release()
    if not frames:
        im = cv2.imread(fallback_jpg)
        up = cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4)
        frames = [up] * 60
    return frames


WRONG_FRAMES = load_i2v_frames(WRONG_I2V, WRONG_JPG)
CORRECT_FRAMES = load_i2v_frames(CORRECT_I2V, CORRECT_JPG)


def make_gym_thumbs():
    c_rgb = cv2.cvtColor(CORRECT_FRAMES[36], cv2.COLOR_BGR2RGB)
    w_rgb = cv2.cvtColor(WRONG_FRAMES[36], cv2.COLOR_BGR2RGB)
    c_top = cv2.cvtColor(CORRECT_FRAMES[6], cv2.COLOR_BGR2RGB)
    crops = [
        c_top[380:920, 420:960],   # Checkpoint 1: Upper torso / spine posture
        w_rgb[520:1040, 380:900],  # Checkpoint 2: Elbow & wrist hinge axis
        c_rgb[420:1060, 320:960],  # Checkpoint 3: Full triceps extension & rope spread
    ]
    thumbs = []
    for cr in crops:
        th = cv2.resize(cr, (96, 96), interpolation=cv2.INTER_AREA)
        thumbs.append(Image.fromarray(th))
    return thumbs


THUMBS = make_gym_thumbs()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_pt(p0: tuple, p1: tuple, t: float) -> tuple:
    return (lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t))


def get_wrong_landmarks(rep: float) -> dict:
    return {
        "head_top": (lerp(740, 710, rep), lerp(430, 455, rep)),
        "shoulder": (lerp(780, 760, rep), lerp(685, 705, rep)),
        "back_mid": (lerp(955, 935, rep), lerp(740, 760, rep)),
        "hip": (lerp(965, 955, rep), lerp(985, 995, rep)),
        "elbow": (lerp(845, 815, rep), lerp(890, 910, rep)),
        "wrist": (lerp(565, 535, rep), lerp(995, 1065, rep)),
        "hand": (lerp(495, 465, rep), lerp(975, 1045, rep)),
    }


def get_correct_landmarks(rep: float) -> dict:
    return {
        "head_top": (lerp(690, 695, rep), lerp(395, 392, rep)),
        "shoulder": (lerp(735, 742, rep), lerp(615, 612, rep)),
        "back_mid": (lerp(845, 855, rep), lerp(735, 735, rep)),
        "hip": (lerp(835, 845, rep), lerp(975, 975, rep)),
        "triceps": (lerp(752, 765, rep), lerp(685, 675, rep)),
        "elbow": (lerp(735, 740, rep), lerp(815, 812, rep)),
        "wrist": (lerp(430, 555, rep), lerp(690, 955, rep)),
        "rope_L": (lerp(395, 475, rep), lerp(705, 980, rep)),
        "rope_R": (lerp(455, 635, rep), lerp(710, 970, rep)),
    }


def sample_clip_frame(frames: list, rep: float, is_wrong: bool, t: float) -> np.ndarray:
    """Samples the I2V clip smoothly at normalized repetition extension `rep` in [0, 1]."""
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

    out = base.astype(np.float32)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.0 * t)

    red_mask = cv2.inRange(hsv, (0, 85, 85), (25, 255, 255)) | cv2.inRange(hsv, (165, 85, 85), (180, 255, 255))
    red_float = cv2.GaussianBlur(red_mask.astype(np.float32) / 255.0, (45, 45), 0)[:, :, None]

    if is_wrong:
        out = np.clip(out + red_float * np.array([15.0, 55.0, 255.0], dtype=np.float32) * (0.20 + 0.42 * (0.4 + 0.6 * rep)), 0, 255)
    else:
        boost = 0.22 + 0.55 * rep + 0.15 * pulse
        out = np.clip(out + red_float * np.array([20.0, 110.0, 255.0], dtype=np.float32) * boost, 0, 255)
        grn_mask = cv2.inRange(hsv, (35, 70, 70), (95, 255, 255))
        grn_float = cv2.GaussianBlur(grn_mask.astype(np.float32) / 255.0, (39, 39), 0)[:, :, None]
        out = np.clip(out + grn_float * np.array([90.0, 255.0, 35.0], dtype=np.float32) * (0.22 + 0.40 * rep), 0, 255)

    return out.astype(np.uint8)


def draw_3d_curved_arrow(
    canvas_rgba: Image.Image,
    p0: tuple,
    p1: tuple,
    p2: tuple,
    col_rgb: tuple,
    prog: float,
    tail_w: float = 10.0,
    neck_w: float = 26.0,
    head_w: float = 58.0,
    head_len: float = 50.0,
):
    """4x-supersampled sculpted 3D volumetric tapered ribbon arrow with drop shadow, neon bloom, and bevel spine."""
    if prog <= 0.04:
        return
    p = min(1.0, max(0.05, prog))

    def bez(u):
        omt = 1.0 - u
        return (
            omt * omt * p0[0] + 2 * omt * u * p1[0] + u * u * p2[0],
            omt * omt * p0[1] + 2 * omt * u * p1[1] + u * u * p2[1],
        )

    def bez_tan(u):
        dx = 2 * (1.0 - u) * (p1[0] - p0[0]) + 2 * u * (p2[0] - p1[0])
        dy = 2 * (1.0 - u) * (p1[1] - p0[1]) + 2 * u * (p2[1] - p1[1])
        L = math.hypot(dx, dy) + 1e-6
        return (dx / L, dy / L)

    steps = 36
    u_max = p * 0.84
    left_pts, right_pts, spine_pts = [], [], []
    for i in range(steps + 1):
        u = u_max * (i / steps)
        bx, by = bez(u)
        tx, ty = bez_tan(u)
        nx, ny = -ty, tx
        frac = i / steps
        half_w = 0.5 * lerp(tail_w, neck_w, frac ** 0.85)
        left_pts.append((bx + nx * half_w, by + ny * half_w))
        right_pts.append((bx - nx * half_w, by - ny * half_w))
        spine_pts.append((bx - nx * (half_w * 0.18), by - ny * (half_w * 0.18)))

    neck_x, neck_y = bez(u_max)
    tx, ty = bez_tan(u_max)
    nx, ny = -ty, tx
    scale_h = min(1.0, p * 1.25)
    hw = 0.5 * head_w * scale_h
    hl = head_len * scale_h
    back_sweep = 10.0 * scale_h

    tip = (neck_x + tx * hl, neck_y + ty * hl)
    wing_L = (neck_x - tx * back_sweep + nx * hw, neck_y - ty * back_sweep + ny * hw)
    wing_R = (neck_x - tx * back_sweep - nx * hw, neck_y - ty * back_sweep - ny * hw)

    poly = left_pts + [wing_L, tip, wing_R] + right_pts[::-1]

    # Bounding box for fast 2x supersampled local patch
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    pad = 48
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(W, int(max(xs)) + pad)
    y1 = min(H, int(max(ys)) + pad)
    pw, ph = x1 - x0, y1 - y0
    if pw <= 4 or ph <= 4:
        return

    SS = 2
    local_poly = [((px - x0) * SS, (py - y0) * SS) for (px, py) in poly]
    sh_poly = [((px - x0 + 6) * SS, (py - y0 + 9) * SS) for (px, py) in poly]

    # 1. Drop shadow
    sh_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(sh_im).polygon(sh_poly, fill=(0, 0, 0, 155))
    sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=8 * SS))

    # 2. Outer neon bloom
    glow_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(glow_im).polygon(local_poly, fill=(*col_rgb, 185))
    glow_im = glow_im.filter(ImageFilter.GaussianBlur(radius=9 * SS))

    # 3. Main 3D body + bright bevel + glossy specular spine
    body_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body_im)
    bd.polygon(local_poly, fill=(*col_rgb, 248), outline=(245, 255, 250, 235), width=2 * SS)

    local_spine = [((px - x0) * SS, (py - y0) * SS) for (px, py) in spine_pts]
    local_tip = ((tip[0] - x0) * SS, (tip[1] - y0) * SS)
    if len(local_spine) >= 2:
        bd.line(local_spine + [local_tip], fill=(245, 255, 250, 215), width=max(2, int(neck_w * 0.22 * SS)), joint="curve")

    comp = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    comp.alpha_composite(sh_im)
    comp.alpha_composite(glow_im)
    comp.alpha_composite(body_im)
    comp_down = comp.resize((pw, ph), resample=Image.Resampling.LANCZOS)
    canvas_rgba.alpha_composite(comp_down, dest=(x0, y0))


def draw_alignment_guide_line(
    canvas_rgba: Image.Image,
    pts: list,
    col_rgb: tuple,
    prog: float,
    width: int = 8,
    dashed: bool = False,
):
    """Draws the glowing biomechanical posture/joint alignment line like De77WaXAiXd."""
    if prog <= 0.03 or len(pts) < 2:
        return
    a = min(1.0, prog)
    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)

    if not dashed:
        gd.line(pts, fill=(*col_rgb, int(205 * a)), width=width * 3, joint="curve")
        cd.line(pts, fill=(*col_rgb, int(255 * a)), width=width, joint="curve")
        cd.line(pts, fill=(245, 255, 250, int(235 * a)), width=max(2, width // 2), joint="curve")
    else:
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            n_seg = max(4, int(dist // 28))
            for k in range(0, n_seg, 2):
                s0 = lerp_pt(p0, p1, k / n_seg)
                s1 = lerp_pt(p0, p1, min(1.0, (k + 1.15) / n_seg))
                gd.line([s0, s1], fill=(*col_rgb, int(190 * a)), width=width * 2)
                cd.line([s0, s1], fill=(235, 255, 245, int(235 * a)), width=max(3, width - 2))

    # Draw glowing joint anchor nodes at endpoints and vertices
    for (px, py) in pts:
        r = width + 4
        gd.ellipse([px - r - 6, py - r - 6, px + r + 6, py + r + 6], fill=(*col_rgb, int(185 * a)))
        cd.ellipse([px - r, py - r, px + r, py + r], fill=(245, 255, 250, int(250 * a)), outline=(*col_rgb, int(255 * a)), width=3)

    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(core)


def draw_ox_status_badge(canvas_rgba: Image.Image, is_correct: bool, pulse: float, cx: int = 895, cy: int = 285, r: int = 54):
    """Prominent glowing O/X (✓ / ✕) badge positioned just like De77WaXAiXd (upper-right above the figure)."""
    halo = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    col = (40, 255, 115) if is_correct else (255, 48, 38)

    hd.ellipse([cx - r - 20, cy - r - 20, cx + r + 20, cy + r + 20], fill=(*col, int(125 + 55 * pulse)))
    halo = halo.filter(ImageFilter.GaussianBlur(radius=18))
    canvas_rgba.alpha_composite(halo)

    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*col, 235), outline=(250, 255, 252, 245), width=4)

    if not is_correct:
        s = int(r * 0.42)
        d.line([(cx - s, cy - s), (cx + s, cy + s)], fill=(255, 255, 255, 255), width=10)
        d.line([(cx - s, cy + s), (cx + s, cy - s)], fill=(255, 255, 255, 255), width=10)
    else:
        pts = [(cx - int(r * 0.44), cy + 3), (cx - int(r * 0.10), cy + int(r * 0.36)), (cx + int(r * 0.46), cy - int(r * 0.32))]
        d.line(pts, fill=(255, 255, 255, 255), width=11, joint="curve")

    canvas_rgba.alpha_composite(overlay)


def draw_demic_minimap_pill(canvas_rgba: Image.Image, step_idx: int, is_correct: bool):
    """Top-center 3-checkpoint thumbnail capsule highlighting which checkpoint is active (0, 1, or 2)."""
    pill_w, pill_h = 356, 118
    px0 = (W - pill_w) // 2
    py0 = 88

    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle(
        [px0, py0, px0 + pill_w, py0 + pill_h],
        radius=26,
        fill=(16, 20, 28, 205),
        outline=(255, 255, 255, 75),
        width=2,
    )
    canvas_rgba.alpha_composite(overlay)

    pad_x, gap = 18, 16
    ty = py0 + (pill_h - 96) // 2
    for i, th in enumerate(THUMBS):
        tx = px0 + pad_x + i * (96 + gap)
        mask = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, 95, 95], radius=15, fill=255)
        canvas_rgba.paste(th, (tx, ty), mask)

        b_ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(b_ov)
        if i == step_idx:
            col = (45, 255, 120, 255) if is_correct else (255, 65, 45, 255)
            bd.rounded_rectangle([tx - 3, ty - 3, tx + 99, ty + 99], radius=16, outline=col, width=4)
        else:
            bd.rounded_rectangle([tx, ty, tx + 96, ty + 96], radius=15, outline=(255, 255, 255, 50), width=1)
        canvas_rgba.alpha_composite(b_ov)


def draw_biomechanical_line_graph_hud(
    canvas_rgba: Image.Image,
    is_correct: bool,
    rep_phase: float,
    charge_level: float,
    pulse: float,
):
    """
    Bottom HUD Real-Time Biomechanical Line Graph (선 그래프) + 5-Bar Muscle Gauge (Zero Text):
    - Left/Center area: Translucent dark glass Line Graph showing two curves over the repetition cycle:
      * Red ✕ Curve: erratic joint-stress spike followed by collapsing muscle tension (~20%).
      * Green ✓ Curve: smooth biomechanical S-curve climbing to a sustained 100% peak-contraction plateau.
      * Live glowing playhead dot + vertical scanline moving in 1:1 sync with the mannequin's repetition!
    - Right area: 5-Bar Muscle Activation Battery Gauge synchronized with the live curve height.
    """
    panel_x0, panel_y0 = 64, 1490
    panel_x1, panel_y1 = 1016, 1790

    base = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(base)
    bd.rounded_rectangle(
        [panel_x0, panel_y0, panel_x1, panel_y1],
        radius=30,
        fill=(14, 18, 25, 215),
        outline=(255, 255, 255, 70),
        width=2,
    )

    # Graph plot area inside panel
    gx0, gy0 = panel_x0 + 36, panel_y0 + 36
    gx1, gy1 = panel_x1 - 150, panel_y1 - 36
    gw, gh = gx1 - gx0, gy1 - gy0

    # Subtle gridlines (4 horizontal, 5 vertical)
    for r_i in range(5):
        yy = int(gy0 + gh * (r_i / 4))
        bd.line([(gx0, yy), (gx1, yy)], fill=(255, 255, 255, 26 if r_i in (0, 4) else 18), width=1)
    for c_i in range(6):
        xx = int(gx0 + gw * (c_i / 5))
        bd.line([(xx, gy0), (xx, gy1)], fill=(255, 255, 255, 16), width=1)

    canvas_rgba.alpha_composite(base)

    # Define the two biomechanical curves for u in [0, 1]
    def wrong_curve_val(u: float) -> float:
        # Spikes early from elbow/wrist jerk, then collapses to 0.18
        return 0.16 + 0.34 * math.exp(-((u - 0.24) / 0.11) ** 2) + 0.06 * math.sin(10.0 * u)

    def correct_curve_val(u: float) -> float:
        # Smooth S-curve rise to sustained peak contraction plateau (0.92), then controlled return
        if u <= 0.65:
            s = u / 0.65
            return 0.15 + 0.78 * (s * s * (3.0 - 2.0 * s))
        else:
            s = (u - 0.65) / 0.35
            return 0.93 - 0.45 * (s * s)

    N = 64
    wrong_pts, correct_pts = [], []
    for i in range(N + 1):
        u = i / N
        px = gx0 + u * gw
        wy = gy1 - wrong_curve_val(u) * gh
        cy = gy1 - correct_curve_val(u) * gh
        wrong_pts.append((px, wy))
        correct_pts.append((px, cy))

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)

    # Area fill under active curve
    u_play = min(1.0, max(0.04, rep_phase))
    k_play = max(1, int(u_play * N))

    if not is_correct:
        # Highlight Red ✕ curve, show Green ✓ curve as subtle reference
        cd.line(correct_pts, fill=(45, 255, 120, 65), width=3, joint="curve")
        poly_fill = wrong_pts[: k_play + 1] + [(wrong_pts[k_play][0], gy1), (wrong_pts[0][0], gy1)]
        gd.polygon(poly_fill, fill=(255, 50, 40, 55))
        gd.line(wrong_pts, fill=(255, 50, 40, 175), width=10, joint="curve")
        cd.line(wrong_pts, fill=(255, 75, 65, 240), width=5, joint="curve")
        cd.line(wrong_pts[: k_play + 1], fill=(255, 220, 215, 255), width=6, joint="curve")
        play_pt = wrong_pts[k_play]
        act_col = (255, 55, 45)
    else:
        # Show Red ✕ curve as dim baseline, blaze the Green ✓ curve!
        cd.line(wrong_pts, fill=(255, 65, 55, 80), width=3, joint="curve")
        poly_fill = correct_pts[: k_play + 1] + [(correct_pts[k_play][0], gy1), (correct_pts[0][0], gy1)]
        gd.polygon(poly_fill, fill=(40, 255, 120, 65))
        gd.line(correct_pts, fill=(40, 255, 120, 195), width=11, joint="curve")
        cd.line(correct_pts, fill=(55, 255, 130, 245), width=5, joint="curve")
        cd.line(correct_pts[: k_play + 1], fill=(230, 255, 240, 255), width=6, joint="curve")
        play_pt = correct_pts[k_play]
        act_col = (45, 255, 120)

    # Small ✕ and ✓ badge markers anchored right on the two curves inside the graph
    w_tag = wrong_pts[int(0.24 * N)]
    c_tag = correct_pts[int(0.65 * N)]
    # Mini red X circle on red peak
    cd.ellipse([w_tag[0] - 13, w_tag[1] - 32, w_tag[0] + 13, w_tag[1] - 6], fill=(255, 50, 40, 230), outline=(255, 255, 255, 220), width=2)
    cd.line([(w_tag[0] - 5, w_tag[1] - 24), (w_tag[0] + 5, w_tag[1] - 14)], fill=(255, 255, 255, 255), width=3)
    cd.line([(w_tag[0] - 5, w_tag[1] - 14), (w_tag[0] + 5, w_tag[1] - 24)], fill=(255, 255, 255, 255), width=3)
    # Mini green check circle on green peak
    cd.ellipse([c_tag[0] - 13, c_tag[1] - 32, c_tag[0] + 13, c_tag[1] - 6], fill=(40, 255, 115, 235), outline=(255, 255, 255, 230), width=2)
    cd.line([(c_tag[0] - 6, c_tag[1] - 19), (c_tag[0] - 1, c_tag[1] - 14), (c_tag[0] + 6, c_tag[1] - 24)], fill=(255, 255, 255, 255), width=3)

    # Live playhead vertical scanline & glowing tracking dot
    px, py = play_pt
    cd.line([(px, gy0), (px, gy1)], fill=(*act_col, 130), width=2)
    gd.ellipse([px - 18, py - 18, px + 18, py + 18], fill=(*act_col, 220))
    cd.ellipse([px - 9, py - 9, px + 9, py + 9], fill=(255, 255, 255, 255), outline=(*act_col, 255), width=3)

    # Right-side 5-Bar Muscle Activation Gauge inside the HUD panel
    mx0 = gx1 + 28
    mw = 84
    bar_h = 30
    gap = 11
    for b in range(5):
        by1 = gy1 - b * (bar_h + gap)
        by0 = by1 - bar_h
        lit = b < int(round(charge_level))
        if lit:
            b_col = (255, 50, 40) if not is_correct else (
                (40, 255, 120) if b < 3 else ((160, 255, 60) if b == 3 else (255, 145, 30))
            )
            gd.rounded_rectangle([mx0 - 3, by0 - 3, mx0 + mw + 3, by1 + 3], radius=9, fill=(*b_col, 150))
            cd.rounded_rectangle([mx0, by0, mx0 + mw, by1], radius=8, fill=(*b_col, 245), outline=(245, 255, 250, 220), width=2)
        else:
            cd.rounded_rectangle([mx0, by0, mx0 + mw, by1], radius=8, fill=(32, 38, 48, 170), outline=(255, 255, 255, 40), width=1)

    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(core)


def draw_pinned_elbow_lock_ring(canvas_rgba: Image.Image, pt: tuple, prog: float, pulse: float):
    """Clean 3D neon-green lock-on ring at the elbow pivot."""
    if prog <= 0.03:
        return
    a = min(1.0, prog)
    cx, cy = int(pt[0]), int(pt[1])
    r = 44
    grn = (40, 255, 120)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)

    gd.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*grn, int(210 * a)), width=11)
    cd.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(235, 255, 242, int(250 * a)), width=4)
    for deg in [0, 90, 180, 270]:
        rad = math.radians(deg)
        x0, y0 = cx + (r - 10) * math.cos(rad), cy + (r - 10) * math.sin(rad)
        x1, y1 = cx + (r + 10) * math.cos(rad), cy + (r + 10) * math.sin(rad)
        cd.line([(x0, y0), (x1, y1)], fill=(240, 255, 245, int(245 * a)), width=4)
    cd.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(240, 255, 245, int(255 * a)))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(core)


def render_single_frame(idx: int) -> int:
    t = idx / FPS
    pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.2 * t)

    # 16.0-second 3-Checkpoint Coaching Timeline (inspired by De77WaXAiXd):
    # - Checkpoint 1 (0.0 – 4.2s): Torso & Spine Posture Alignment
    #     * 0.0 – 2.2s (✕ Wrong): Excessive forward lean, red spine alignment line + green corrective arrow
    #     * 2.2 – 4.2s (✓ Correct): Locked upright spine alignment line + green ✓ confirmation
    # - Checkpoint 2 (4.2 – 8.6s): Elbow Hinge & Wrist Axis Alignment
    #     * 4.2 – 6.4s (✕ Wrong): Elbow flaring forward/out + wrist bending back, red broken axis line + 3D red arrows
    #     * 6.4 – 8.6s (✓ Correct): Pinned Elbow Lock-On Ring + straight green upper-arm & forearm alignment lines
    # - Checkpoint 3 (8.6 – 16.0s): Full-ROM Execution & Live Muscle Tension Curve Comparison
    #     * 8.6 – 11.4s (✕ Full Rep): Full wrong rep showing force leak & red ✕ curve on Line Graph
    #     * 11.4 – 16.0s (✓ Full Reps): 2 full peak-contraction reps with 3D curved green arrow + outward rope-spread arrows + 5/5 green curve!

    if t < 4.2:
        step_idx = 0
        is_correct = (t >= 2.2)
        local_t = t if not is_correct else (t - 2.2)
        rep = 0.15 + 0.20 * (0.5 - 0.5 * math.cos(2.0 * math.pi * 0.5 * local_t))
        frame_bgr = sample_clip_frame(CORRECT_FRAMES if is_correct else WRONG_FRAMES, rep, not is_correct, t)

        # Gentle macro zoom on upper body for first 1.2s
        if t < 1.2:
            s = t / 1.2
            ease = s * s * (3.0 - 2.0 * s)
            z = lerp(1.22, 1.0, ease)
            cw, ch = int(W / z), int(H / z)
            cx, cy = 720, 720
            x0 = max(0, min(W - cw, cx - cw // 2))
            y0 = max(0, min(H - ch, cy - ch // 2))
            frame_bgr = cv2.resize(frame_bgr[y0 : y0 + ch, x0 : x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)

        canvas = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        prog = min(1.0, local_t / 0.45)

        if not is_correct:
            lm = get_wrong_landmarks(rep)
            # Red curved/hunched spine line + target dashed green line
            draw_alignment_guide_line(canvas, [(845, 510), (860, 740), (855, 980)], (45, 255, 120), prog * 0.85, width=6, dashed=True)
            draw_alignment_guide_line(canvas, [lm["head_top"], lm["back_mid"], lm["hip"]], (255, 50, 40), prog, width=9)
            # Bold 3D Green corrective arrow pushing hunched upper back toward upright target line
            draw_3d_curved_arrow(
                canvas,
                p0=(965, 650),
                p1=(910, 610),
                p2=(845, 605),
                col_rgb=(45, 255, 120),
                prog=prog,
                tail_w=11,
                neck_w=26,
                head_w=56,
                head_len=48,
            )
            charge = 1.0
        else:
            lm = get_correct_landmarks(rep)
            # Crisp Neon-Green Spine Alignment Line confirming locked posture!
            draw_alignment_guide_line(canvas, [lm["head_top"], lm["back_mid"], lm["hip"]], (45, 255, 120), prog, width=10)
            charge = 3.0

        draw_demic_minimap_pill(canvas, step_idx=0, is_correct=is_correct)
        draw_ox_status_badge(canvas, is_correct=is_correct, pulse=pulse)
        draw_biomechanical_line_graph_hud(canvas, is_correct=is_correct, rep_phase=min(1.0, local_t / 2.0), charge_level=charge, pulse=pulse)

    elif t < 8.6:
        step_idx = 1
        is_correct = (t >= 6.4)
        local_t = (t - 4.2) if not is_correct else (t - 6.4)
        rep = 0.5 - 0.5 * math.cos(2.0 * math.pi * 0.48 * local_t)
        frame_bgr = sample_clip_frame(CORRECT_FRAMES if is_correct else WRONG_FRAMES, rep, not is_correct, t)
        canvas = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        prog = min(1.0, local_t / 0.45)

        if not is_correct:
            lm = get_wrong_landmarks(rep)
            # Red misaligned Shoulder -> Flared Elbow -> Bent Wrist line
            draw_alignment_guide_line(canvas, [lm["shoulder"], lm["elbow"], lm["wrist"]], (255, 50, 40), prog, width=9)
            # 3D Red arrow showing elbow flaring out & wrist snapping up
            ex, ey = lm["elbow"]
            wx, wy = lm["wrist"]
            draw_3d_curved_arrow(
                canvas,
                p0=(ex - 10, ey + 15),
                p1=(ex + 65, ey - 10),
                p2=(ex + 135, ey - 55),
                col_rgb=(255, 50, 40),
                prog=prog,
                tail_w=10,
                neck_w=24,
                head_w=54,
                head_len=46,
            )
            draw_3d_curved_arrow(
                canvas,
                p0=(wx + 15, wy + 5),
                p1=(wx - 45, wy + 45),
                p2=(wx - 105, wy + 100),
                col_rgb=(255, 50, 40),
                prog=prog,
                tail_w=9,
                neck_w=21,
                head_w=46,
                head_len=40,
            )
            charge = 1.0
        else:
            lm = get_correct_landmarks(rep)
            # Green aligned Shoulder -> Pinned Elbow -> Neutral Wrist axis + Lock-On Ring!
            draw_alignment_guide_line(canvas, [lm["shoulder"], lm["elbow"], lm["wrist"]], (45, 255, 120), prog, width=9)
            draw_pinned_elbow_lock_ring(canvas, lm["elbow"], prog, pulse)
            charge = 4.0

        draw_demic_minimap_pill(canvas, step_idx=1, is_correct=is_correct)
        draw_ox_status_badge(canvas, is_correct=is_correct, pulse=pulse)
        draw_biomechanical_line_graph_hud(canvas, is_correct=is_correct, rep_phase=min(1.0, local_t / 2.2), charge_level=charge, pulse=pulse)

    else:
        step_idx = 2
        is_correct = (t >= 11.4)
        local_t = (t - 8.6) if not is_correct else (t - 11.4)
        if not is_correct:
            # 1 full wrong rep (2.8s)
            rep_cycle = min(1.0, local_t / 2.6)
            rep = 0.5 - 0.5 * math.cos(2.0 * math.pi * rep_cycle)
            frame_bgr = sample_clip_frame(WRONG_FRAMES, rep, True, t)
            canvas = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
            prog = min(1.0, local_t / 0.4)
            lm = get_wrong_landmarks(rep)
            draw_alignment_guide_line(canvas, [lm["shoulder"], lm["elbow"], lm["wrist"]], (255, 50, 40), prog, width=8)
            ex, ey = lm["elbow"]
            draw_3d_curved_arrow(
                canvas,
                p0=(ex - 10, ey + 15),
                p1=(ex + 65, ey - 10),
                p2=(ex + 135, ey - 55),
                col_rgb=(255, 50, 40),
                prog=prog,
            )
            charge = 1.0
            graph_phase = rep_cycle
        else:
            # 2 full smooth Correct reps over 4.6s (2.3s per rep)
            rep_cycle = (local_t / 2.3) % 1.0
            rep = 0.5 - 0.5 * math.cos(2.0 * math.pi * rep_cycle)
            frame_bgr = sample_clip_frame(CORRECT_FRAMES, rep, False, t)
            canvas = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
            prog = min(1.0, local_t / 0.45)
            lm = get_correct_landmarks(rep)

            # 1) Upright posture & upper-arm alignment line + Pinned Elbow Lock Ring
            draw_alignment_guide_line(canvas, [lm["shoulder"], lm["elbow"]], (45, 255, 120), prog, width=9)
            draw_pinned_elbow_lock_ring(canvas, lm["elbow"], prog, pulse)

            # 2) Bold 3D Curved Tapered Green Arrow sweeping along forearm extension path
            draw_3d_curved_arrow(
                canvas,
                p0=(375, 650),
                p1=(365, 835),
                p2=(465, 995),
                col_rgb=(45, 255, 120),
                prog=min(1.0, 0.35 + 0.85 * rep),
                tail_w=12,
                neck_w=28,
                head_w=62,
                head_len=54,
            )

            # 3) Dual symmetrical 3D Rope-Spread Arrows at bottom extension
            spread_prog = max(0.0, min(1.0, (rep - 0.30) / 0.55))
            if spread_prog > 0.05:
                lx, ly = lm["rope_L"]
                rx, ry = lm["rope_R"]
                draw_3d_curved_arrow(
                    canvas,
                    p0=(lx + 15, ly - 15),
                    p1=(lx - 35, ly + 25),
                    p2=(lx - 95, ly + 75),
                    col_rgb=(45, 255, 120),
                    prog=spread_prog,
                    tail_w=9,
                    neck_w=22,
                    head_w=48,
                    head_len=42,
                )
                draw_3d_curved_arrow(
                    canvas,
                    p0=(rx - 15, ry - 15),
                    p1=(rx + 35, ry + 25),
                    p2=(rx + 95, ry + 75),
                    col_rgb=(45, 255, 120),
                    prog=spread_prog,
                    tail_w=9,
                    neck_w=22,
                    head_w=48,
                    head_len=42,
                )

            charge = min(5.0, 1.5 + 4.2 * rep) if local_t < 1.8 else 5.0
            graph_phase = min(1.0, 0.15 + 0.65 * rep)

        draw_demic_minimap_pill(canvas, step_idx=2, is_correct=is_correct)
        draw_ox_status_badge(canvas, is_correct=is_correct, pulse=pulse)
        draw_biomechanical_line_graph_hud(canvas, is_correct=is_correct, rep_phase=graph_phase, charge_level=charge, pulse=pulse)

    out_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}.jpg"), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return idx


def build_audio_track(wav_path: str):
    sr = 44100
    n_samples = int(sr * DURATION)
    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32) / sr

    # Deep gym studio sub-bass pulse
    beat_env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 1.25 * t)
    sub = 0.08 * np.sin(2.0 * np.pi * 52.0 * t) * beat_env
    left += sub
    right += sub

    # Warning buzz on ✕ segments (0.0-2.2s, 4.2-6.4s, 8.6-11.4s)
    for (s0, s1) in [(0.1, 2.1), (4.2, 6.3), (8.6, 11.3)]:
        mask = (t >= s0) & (t <= s1)
        buzz = 0.06 * np.sin(2.0 * np.pi * 88.0 * t[mask]) + 0.03 * np.sin(2.0 * np.pi * 176.0 * t[mask])
        left[mask] += buzz
        right[mask] += buzz * 0.9

    # Crisp Lock-On Snap + Positive Chime at ✓ transitions (2.2s, 6.4s, 11.4s)
    for t_snap in [2.2, 6.4, 11.4]:
        mask = (t >= t_snap) & (t < t_snap + 0.45)
        dt = t[mask] - t_snap
        snap = 0.24 * np.exp(-dt * 28.0) * np.sin(2.0 * np.pi * (880.0 + 440.0 * np.exp(-dt * 40.0)) * dt)
        drop = 0.18 * np.exp(-dt * 7.5) * np.sin(2.0 * np.pi * (95.0 - 40.0 * dt) * dt)
        left[mask] += snap + drop
        right[mask] += snap + drop

    # Rising 5-bar harmonic chime during final full-ROM peak contraction (11.5s - 13.0s)
    notes = [523.25, 587.33, 659.25, 783.99, 1046.50]
    for i, freq in enumerate(notes):
        t0 = 11.55 + i * 0.22
        mask = (t >= t0) & (t < t0 + 0.65)
        dt = t[mask] - t0
        chime = 0.14 * np.exp(-dt * 6.0) * (np.sin(2.0 * np.pi * freq * dt) + 0.4 * np.sin(2.0 * np.pi * freq * 2.0 * dt))
        left[mask] += chime * (0.8 + 0.05 * i)
        right[mask] += chime * (1.0 - 0.04 * i)

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
    print(f"Rendering {TOTAL_FRAMES} frames (16.0s @ 30fps)...")
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

    # Build 4-panel Keyframe Sheet showing Checkpoint 1 (✕ -> ✓), Checkpoint 2 (✕), and Checkpoint 3 (✓ Full ROM + Line Graph)
    sample_indices = [int(1.2 * FPS), int(3.1 * FPS), int(5.2 * FPS), int(12.6 * FPS)]
    panels = []
    for fi in sample_indices:
        im = cv2.imread(os.path.join(TMP_DIR, f"frame_{fi:04d}.jpg"))
        panels.append(cv2.resize(im, (450, 800), interpolation=cv2.INTER_AREA))
    sheet = np.hstack(panels)
    cv2.imwrite(OUT_KEYFRAMES, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Build Side-by-Side Comparison against De77WaXAiXd reference frame
    ref_path = "/workspace/daily_info_reels/demic_bench/frames_De77WaXAiXd/t_24s.jpg"
    if os.path.exists(ref_path):
        ref_im = cv2.resize(cv2.imread(ref_path), (540, 960), interpolation=cv2.INTER_AREA)
        our_w = cv2.resize(cv2.imread(os.path.join(TMP_DIR, f"frame_{int(5.2*FPS):04d}.jpg")), (540, 960), interpolation=cv2.INTER_AREA)
        our_c = cv2.resize(cv2.imread(os.path.join(TMP_DIR, f"frame_{int(12.6*FPS):04d}.jpg")), (540, 960), interpolation=cv2.INTER_AREA)
        comp = np.hstack([ref_im, our_w, our_c])
        cv2.imwrite(OUT_COMP, comp, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("DONE! Saved:", OUT_MP4, OUT_KEYFRAMES)


if __name__ == "__main__":
    main()
