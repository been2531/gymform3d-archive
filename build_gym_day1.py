#!/usr/bin/env python3
"""
@gymform.3d — Day 1: Cable Rope Triceps Pushdown (v5 — Approved v2 3-Scene Multi-Angle O/X Format)
Features:
1. Dynamic 3-Scene Multi-Angle O/X Sequencing (15.0s @ 30fps = 450 frames):
   - Scene 1 (0.0 – 4.8s): Upper Body / Torso Posture Angle (Wide-Medium Side View, Zoom 1.08x)
       * 0.0 – 2.4s (✕): Hunched torso lean, white/cyan dashed target ray + red wrong ray + 26.4° pill badge + arc-fitted 3D arrow
       * 2.4 – 4.8s (✓): Upright locked spine alignment ray (no degree numbers!) + Lock-On Target Ring + coaxial 3D green arrow
   - Scene 2 (4.8 – 9.6s): Elbow Pivot & Humerus Axis (Medium-Tight Upper-Arm Zoom, 1.32x)
       * 4.8 – 7.2s (✕): Flared/drifting elbow axis, dashed vertical target ray + red drifted ray + 19.8° pill badge + arc-fitted 3D arrow
       * 7.2 – 9.6s (✓): Pinned vertical humerus + Lock-On Ring at elbow + arc-fitted 3D green arrow sweeping along exact forearm extension arc
   - Scene 3 (9.6 – 15.0s): Wrist Bend vs Full Lockout & Rope Spread (Close-Up Zoom, 1.50x)
       * 9.6 – 12.0s (✕): Bent/hyperextended wrist at bottom, dashed neutral forearm ray + red bent wrist ray + 22.5° pill badge + arc-fitted 3D arrow
       * 12.0 – 15.0s (✓): Full lockout neutral wrist + V-spread green rays + dual coaxial 3D green Rope-Spread arrows + 100% triceps glow
2. Zero bottom chart clutter: sleek left-edge vertical gauge bar + top 4-icon checkpoint capsule + upper-right circular ✕/✓ badge.
3. High-contrast dark casing + neon bloom on all lines, arcs, arrows, and degree badges.
"""
import os
import math
import wave
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 30
DURATION = 15.0
TOTAL_FRAMES = int(FPS * DURATION)  # 450 frames

OUT_DIR = "/workspace/daily_info_reels/output"
WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")
TMP_DIR = "/workspace/daily_info_reels/frames_gym_day1_v5"

OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")
OUT_KEYFRAMES = os.path.join(OUT_DIR, "demic_gym_day1_keyframes.jpg")


def load_i2v_frames(mp4_path: str, fallback_jpg: str) -> list:
    frames = []
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            up = cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4)
            blur = cv2.GaussianBlur(up, (0, 0), 1.05)
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


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_56 = get_font(56)


def sample_and_enhance(frames: list, rep: float, is_wrong: bool, pulse: float) -> np.ndarray:
    """Smoothly interpolates the 3D workout clip at normalized extension `rep` in [0, 1]."""
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
    red_mask = cv2.inRange(hsv, (0, 80, 80), (25, 255, 255)) | cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
    red_float = cv2.GaussianBlur(red_mask.astype(np.float32) / 255.0, (43, 43), 0)[:, :, None]

    if is_wrong:
        out = np.clip(out + red_float * np.array([15.0, 45.0, 255.0], dtype=np.float32) * (0.24 + 0.28 * rep), 0, 255)
    else:
        boost = 0.26 + 0.44 * rep + 0.12 * pulse
        out = np.clip(out + red_float * np.array([20.0, 115.0, 255.0], dtype=np.float32) * boost, 0, 255)
        grn_mask = cv2.inRange(hsv, (35, 65, 65), (95, 255, 255))
        grn_float = cv2.GaussianBlur(grn_mask.astype(np.float32) / 255.0, (37, 37), 0)[:, :, None]
        out = np.clip(out + grn_float * np.array([85.0, 255.0, 40.0], dtype=np.float32) * (0.28 + 0.36 * rep), 0, 255)

    return out.astype(np.uint8)


def crop_camera_view(frame_bgr: np.ndarray, cx: float, cy: float, zoom: float) -> tuple:
    if zoom <= 1.001:
        return frame_bgr.copy(), (lambda pt: (float(pt[0]), float(pt[1])))
    cw = int(round(W / zoom))
    ch = int(round(H / zoom))
    x0 = max(0, min(W - cw, int(round(cx - cw / 2))))
    y0 = max(0, min(H - ch, int(round(cy - ch / 2))))
    cropped = frame_bgr[y0 : y0 + ch, x0 : x0 + cw]
    resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    def tf(pt: tuple) -> tuple:
        return ((pt[0] - x0) * (W / cw), (pt[1] - y0) * (H / ch))

    return resized, tf


def ray_pt(pivot: tuple, deg: float, r: float) -> tuple:
    rad = math.radians(deg)
    return (pivot[0] + r * math.cos(rad), pivot[1] + r * math.sin(rad))


def draw_high_contrast_dashed_ray(
    canvas_rgba: Image.Image,
    p0: tuple,
    p1: tuple,
    col_rgb: tuple = (180, 255, 220),
    width: int = 8,
    dash_len: int = 24,
    gap_len: int = 14,
    prog: float = 1.0,
):
    if prog <= 0.03:
        return
    p_end = (lerp(p0[0], p1[0], prog), lerp(p0[1], p1[1], prog))
    dx, dy = p_end[0] - p0[0], p_end[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return
    ux, uy = dx / L, dy / L

    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    cur = 0.0
    while cur < L:
        end = min(L, cur + dash_len)
        s0 = (p0[0] + ux * cur, p0[1] + uy * cur)
        s1 = (p0[0] + ux * end, p0[1] + uy * end)
        d.line([s0, s1], fill=(5, 8, 14, 235), width=width + 6)
        d.line([s0, s1], fill=(*col_rgb, 250), width=width)
        cur += dash_len + gap_len

    canvas_rgba.alpha_composite(ov)


def draw_high_contrast_solid_ray(
    canvas_rgba: Image.Image,
    pts: list,
    col_rgb: tuple,
    width: int = 12,
    prog: float = 1.0,
):
    if len(pts) < 2 or prog <= 0.03:
        return
    a = min(1.0, prog)
    casing = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(casing)
    cd.line(pts, fill=(5, 8, 14, int(235 * a)), width=width + 10, joint="curve")

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line(pts, fill=(*col_rgb, int(225 * a)), width=width * 3, joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))

    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    crd = ImageDraw.Draw(core)
    crd.line(pts, fill=(*col_rgb, int(255 * a)), width=width, joint="curve")
    crd.line(pts, fill=(255, 255, 255, int(250 * a)), width=max(3, width // 3), joint="curve")

    for (px, py) in pts:
        r = width + 4
        cd.ellipse([px - r - 4, py - r - 4, px + r + 4, py + r + 4], fill=(5, 8, 14, int(240 * a)))
        crd.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255, int(255 * a)), outline=(*col_rgb, int(255 * a)), width=4)

    canvas_rgba.alpha_composite(casing)
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(core)


def draw_arc_fitted_3d_arrow(
    canvas_rgba: Image.Image,
    pivot: tuple,
    deg_start: float,
    deg_end: float,
    radius: float,
    col_rgb: tuple,
    prog: float = 1.0,
    tail_w: float = 18.0,
    neck_w: float = 36.0,
    head_w: float = 76.0,
    head_len_deg: float = 10.0,
):
    if prog <= 0.05:
        return
    p = min(1.0, prog)
    cur_end = lerp(deg_start, deg_end, p)
    total_span = cur_end - deg_start
    if abs(total_span) < 2.0:
        return
    sign = 1.0 if total_span > 0 else -1.0
    actual_head_deg = min(abs(total_span) * 0.42, head_len_deg) * sign
    neck_deg = cur_end - actual_head_deg

    steps = 32
    left_pts, right_pts, spine_pts = [], [], []
    for i in range(steps + 1):
        u = i / steps
        ang = lerp(deg_start, neck_deg, u)
        rad = math.radians(ang)
        cx = pivot[0] + radius * math.cos(rad)
        cy = pivot[1] + radius * math.sin(rad)
        nx, ny = math.cos(rad), math.sin(rad)
        half_w = 0.5 * lerp(tail_w, neck_w, u ** 0.85)
        left_pts.append((cx + nx * half_w, cy + ny * half_w))
        right_pts.append((cx - nx * half_w, cy - ny * half_w))
        spine_pts.append((cx, cy))

    rad_neck = math.radians(neck_deg)
    rad_tip = math.radians(cur_end)
    nx, ny = math.cos(rad_neck), math.sin(rad_neck)
    tx, ty = -math.sin(rad_neck) * sign, math.cos(rad_neck) * sign
    neck_cx = pivot[0] + radius * math.cos(rad_neck)
    neck_cy = pivot[1] + radius * math.sin(rad_neck)

    hw = 0.5 * head_w * min(1.0, p * 1.25)
    back_sweep = 10.0
    wing_L = (neck_cx - tx * back_sweep + nx * hw, neck_cy - ty * back_sweep + ny * hw)
    wing_R = (neck_cx - tx * back_sweep - nx * hw, neck_cy - ty * back_sweep - ny * hw)
    tip = (pivot[0] + radius * math.cos(rad_tip), pivot[1] + radius * math.sin(rad_tip))

    poly = left_pts + [wing_L, tip, wing_R] + right_pts[::-1]
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    pad = 50
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(W, int(max(xs)) + pad)
    y1 = min(H, int(max(ys)) + pad)
    pw, ph = x1 - x0, y1 - y0
    if pw <= 4 or ph <= 4:
        return

    SS = 2
    local_poly = [((px - x0) * SS, (py - y0) * SS) for (px, py) in poly]
    sh_poly = [((px - x0 + 6) * SS, (py - y0 + 8) * SS) for (px, py) in poly]

    sh_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(sh_im).polygon(sh_poly, fill=(0, 0, 0, 210))
    sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=6 * SS))

    casing_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(casing_im).polygon(local_poly, fill=(5, 8, 14, 245), outline=(5, 8, 14, 250), width=5 * SS)

    glow_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(glow_im).polygon(local_poly, fill=(*col_rgb, 210))
    glow_im = glow_im.filter(ImageFilter.GaussianBlur(radius=8 * SS))

    body_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body_im)
    bd.polygon(local_poly, fill=(*col_rgb, 255), outline=(255, 255, 255, 255), width=3 * SS)

    local_spine = [((px - x0) * SS, (py - y0) * SS) for (px, py) in spine_pts]
    local_tip = ((tip[0] - x0) * SS, (tip[1] - y0) * SS)
    if len(local_spine) >= 2:
        bd.line(local_spine + [local_tip], fill=(255, 255, 255, 235), width=max(3, int(neck_w * 0.24 * SS)), joint="curve")

    comp = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    comp.alpha_composite(sh_im)
    comp.alpha_composite(glow_im)
    comp.alpha_composite(casing_im)
    comp.alpha_composite(body_im)
    comp_down = comp.resize((pw, ph), resample=Image.Resampling.LANCZOS)
    canvas_rgba.alpha_composite(comp_down, dest=(x0, y0))


def draw_linear_fitted_3d_arrow(
    canvas_rgba: Image.Image,
    p0: tuple,
    p2: tuple,
    col_rgb: tuple,
    prog: float = 1.0,
    tail_w: float = 15.0,
    neck_w: float = 32.0,
    head_w: float = 68.0,
    head_len: float = 54.0,
):
    if prog <= 0.05:
        return
    p = min(1.0, prog)
    cur_p2 = (lerp(p0[0], p2[0], p), lerp(p0[1], p2[1], p))
    dx, dy = cur_p2[0] - p0[0], cur_p2[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 20.0:
        return
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx
    hl = min(head_len, L * 0.42)
    neck_L = L - hl

    steps = 20
    left_pts, right_pts, spine_pts = [], [], []
    for i in range(steps + 1):
        u = i / steps
        d_cur = u * neck_L
        bx, by = p0[0] + tx * d_cur, p0[1] + ty * d_cur
        half_w = 0.5 * lerp(tail_w, neck_w, u ** 0.85)
        left_pts.append((bx + nx * half_w, by + ny * half_w))
        right_pts.append((bx - nx * half_w, by - ny * half_w))
        spine_pts.append((bx, by))

    neck_x, neck_y = p0[0] + tx * neck_L, p0[1] + ty * neck_L
    hw = 0.5 * head_w * min(1.0, p * 1.25)
    back_sweep = 10.0
    wing_L = (neck_x - tx * back_sweep + nx * hw, neck_y - ty * back_sweep + ny * hw)
    wing_R = (neck_x - tx * back_sweep - nx * hw, neck_y - ty * back_sweep - ny * hw)
    tip = cur_p2

    poly = left_pts + [wing_L, tip, wing_R] + right_pts[::-1]
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
    sh_poly = [((px - x0 + 6) * SS, (py - y0 + 8) * SS) for (px, py) in poly]

    sh_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(sh_im).polygon(sh_poly, fill=(0, 0, 0, 210))
    sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=6 * SS))

    casing_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(casing_im).polygon(local_poly, fill=(5, 8, 14, 245), outline=(5, 8, 14, 250), width=5 * SS)

    glow_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(glow_im).polygon(local_poly, fill=(*col_rgb, 205))
    glow_im = glow_im.filter(ImageFilter.GaussianBlur(radius=7 * SS))

    body_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body_im)
    bd.polygon(local_poly, fill=(*col_rgb, 255), outline=(255, 255, 255, 255), width=3 * SS)
    local_spine = [((px - x0) * SS, (py - y0) * SS) for (px, py) in spine_pts]
    local_tip = ((tip[0] - x0) * SS, (tip[1] - y0) * SS)
    bd.line(local_spine + [local_tip], fill=(255, 255, 255, 235), width=max(3, int(neck_w * 0.24 * SS)), joint="curve")

    comp = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    comp.alpha_composite(sh_im)
    comp.alpha_composite(glow_im)
    comp.alpha_composite(casing_im)
    comp.alpha_composite(body_im)
    comp_down = comp.resize((pw, ph), resample=Image.Resampling.LANCZOS)
    canvas_rgba.alpha_composite(comp_down, dest=(x0, y0))


def draw_wrong_angle_protract_system(
    canvas_rgba: Image.Image,
    pivot: tuple,
    deg_target: float,
    deg_wrong: float,
    ray_len: float,
    arc_radius: float,
    arrow_radius: float,
    deg_label: str,
    label_radius: float,
    prog: float = 1.0,
    label_deg_offset: float = 0.0,
):
    if prog <= 0.03:
        return
    cur_wrong = lerp(deg_target, deg_wrong, min(1.0, prog * 1.15))
    a0 = min(deg_target, cur_wrong)
    a1 = max(deg_target, cur_wrong)

    # 1. Angle sector + protractor arc
    if abs(a1 - a0) > 1.0:
        sec = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sec)
        bbox_arc = [pivot[0] - arc_radius, pivot[1] - arc_radius, pivot[0] + arc_radius, pivot[1] + arc_radius]
        sd.pieslice(bbox_arc, start=a0, end=a1, fill=(255, 42, 32, int(95 * prog)))
        sd.arc(bbox_arc, start=a0, end=a1, fill=(5, 8, 14, int(240 * prog)), width=12)
        sd.arc(bbox_arc, start=a0, end=a1, fill=(255, 190, 180, int(255 * prog)), width=6)
        canvas_rgba.alpha_composite(sec)

    # 2. Target dashed ray
    p_target = ray_pt(pivot, deg_target, ray_len)
    p_target_back = ray_pt(pivot, deg_target + 180.0, 65.0)
    draw_high_contrast_dashed_ray(canvas_rgba, p_target_back, p_target, col_rgb=(180, 255, 220), width=8, prog=prog)

    # 3. Wrong solid red ray
    p_wrong = ray_pt(pivot, cur_wrong, ray_len)
    draw_high_contrast_solid_ray(canvas_rgba, [pivot, p_wrong], col_rgb=(255, 48, 38), width=12, prog=prog)

    # 4. Arc-fitted 3D arrow sweeping from Wrong Ray -> Target Ray!
    draw_arc_fitted_3d_arrow(
        canvas_rgba,
        pivot=pivot,
        deg_start=deg_wrong,
        deg_end=deg_target,
        radius=arrow_radius,
        col_rgb=(255, 58, 45),
        prog=prog,
        tail_w=18.0,
        neck_w=36.0,
        head_w=76.0,
        head_len_deg=max(10.0, abs(deg_target - deg_wrong) * 0.38),
    )

    # 5. High-contrast Degree Pill Badge (ONLY on ✕!)
    mid_deg = 0.5 * (deg_target + deg_wrong) + label_deg_offset
    lx, ly = ray_pt(pivot, mid_deg, label_radius)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    tb = od.textbbox((0, 0), deg_label, font=FONT_56)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad_x, pad_y = 24, 14
    bx0 = int(lx - tw / 2 - pad_x)
    by0 = int(ly - th / 2 - pad_y)
    bx1 = int(lx + tw / 2 + pad_x)
    by1 = int(ly + th / 2 + pad_y)

    glow_pill = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow_pill).rounded_rectangle([bx0 - 8, by0 - 8, bx1 + 8, by1 + 8], radius=24, fill=(255, 45, 35, int(180 * prog)))
    glow_pill = glow_pill.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(glow_pill)

    od.rounded_rectangle([bx0, by0, bx1, by1], radius=20, fill=(10, 14, 20, int(242 * prog)), outline=(255, 75, 65, int(255 * prog)), width=4)
    od.text((int(lx - tw / 2 - tb[0]), int(ly - th / 2 - tb[1])), deg_label, font=FONT_56, fill=(255, 255, 255, int(255 * prog)))
    canvas_rgba.alpha_composite(ov)


def draw_lock_on_target_ring(canvas_rgba: Image.Image, pt: tuple, r_outer: int = 54, prog: float = 1.0):
    if prog <= 0.03:
        return
    a = min(1.0, prog)
    cx, cy = int(pt[0]), int(pt[1])
    grn = (40, 255, 120)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)

    cd.ellipse([cx - r_outer - 4, cy - r_outer - 4, cx + r_outer + 4, cy + r_outer + 4], outline=(5, 8, 14, int(230 * a)), width=12)
    gd.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(*grn, int(220 * a)), width=14)
    cd.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(230, 255, 240, int(255 * a)), width=5)

    r_in = int(r_outer * 0.62)
    cd.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], outline=(*grn, int(245 * a)), width=4)

    for deg in [0, 90, 180, 270]:
        rad = math.radians(deg)
        x0, y0 = cx + (r_outer - 12) * math.cos(rad), cy + (r_outer - 12) * math.sin(rad)
        x1, y1 = cx + (r_outer + 12) * math.cos(rad), cy + (r_outer + 12) * math.sin(rad)
        cd.line([(x0, y0), (x1, y1)], fill=(5, 8, 14, int(235 * a)), width=8)
        cd.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, int(255 * a)), width=4)

    cd.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(255, 255, 255, int(255 * a)), outline=(*grn, int(255 * a)), width=4)

    glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(core)


def draw_top_silhouette_capsule(canvas_rgba: Image.Image, active_idx: int, is_correct: bool):
    pill_w, pill_h = 370, 86
    px0 = (W - pill_w) // 2
    py0 = 66

    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rounded_rectangle(
        [px0, py0, px0 + pill_w, py0 + pill_h],
        radius=43,
        fill=(14, 18, 24, 230),
        outline=(255, 255, 255, 130),
        width=3,
    )

    centers = [px0 + 55 + i * 86 for i in range(4)]
    cy = py0 + pill_h // 2
    act_col = (40, 245, 115) if is_correct else (255, 52, 42)

    for i, cx in enumerate(centers):
        if i == active_idx:
            d.ellipse([cx - 31, cy - 31, cx + 31, cy + 31], fill=(*act_col, 250), outline=(255, 255, 255, 235), width=2)
            ic_col = (255, 255, 255, 255)
        else:
            ic_col = (180, 190, 205, 155)

        if i == 0:
            d.line([(cx - 6, cy - 16), (cx - 6, cy + 12), (cx + 10, cy + 12)], fill=ic_col, width=4, joint="curve")
            d.ellipse([cx - 10, cy - 22, cx - 2, cy - 14], fill=ic_col)
        elif i == 1:
            d.ellipse([cx - 5, cy - 19, cx + 5, cy - 9], fill=ic_col)
            d.line([(cx, cy - 8), (cx, cy + 8)], fill=ic_col, width=4)
            d.line([(cx, cy - 4), (cx - 12, cy + 4), (cx - 6, cy + 16)], fill=ic_col, width=3)
            d.line([(cx, cy - 4), (cx + 11, cy + 8)], fill=ic_col, width=3)
        elif i == 2:
            d.line([(cx - 14, cy + 6), (cx + 2, cy - 4), (cx + 14, cy - 4)], fill=ic_col, width=4, joint="curve")
            d.ellipse([cx - 2, cy - 8, cx + 6, cy], fill=ic_col)
        else:
            d.ellipse([cx - 5, cy - 18, cx + 5, cy - 8], fill=ic_col)
            d.line([(cx, cy - 7), (cx, cy + 18)], fill=ic_col, width=4)

    canvas_rgba.alpha_composite(ov)


def draw_ox_status_badge(canvas_rgba: Image.Image, is_correct: bool, pulse: float, cx: int = 925, cy: int = 235, r: int = 58):
    halo = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    col = (40, 245, 115) if is_correct else (255, 48, 38)
    hd.ellipse([cx - r - 18, cy - r - 18, cx + r + 18, cy + r + 18], fill=(*col, int(135 + 45 * pulse)))
    halo = halo.filter(ImageFilter.GaussianBlur(radius=16))
    canvas_rgba.alpha_composite(halo)

    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*col, 250), outline=(255, 255, 255, 250), width=5)

    if not is_correct:
        s = int(r * 0.40)
        d.line([(cx - s, cy - s), (cx + s, cy + s)], fill=(255, 255, 255, 255), width=12)
        d.line([(cx - s, cy + s), (cx + s, cy - s)], fill=(255, 255, 255, 255), width=12)
    else:
        pts = [(cx - int(r * 0.42), cy + 3), (cx - int(r * 0.08), cy + int(r * 0.36)), (cx + int(r * 0.45), cy - int(r * 0.32))]
        d.line(pts, fill=(255, 255, 255, 255), width=13, joint="curve")

    canvas_rgba.alpha_composite(ov)


def draw_left_slim_vertical_bar(canvas_rgba: Image.Image, is_correct: bool, fill_ratio: float):
    bx0, by0 = 58, 720
    bw, bh = 24, 460
    bx1, by1 = bx0 + bw, by0 + bh

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)

    cd.rounded_rectangle([bx0, by0, bx1, by1], radius=12, fill=(14, 18, 24, 230), outline=(255, 255, 255, 90), width=2)

    col = (40, 245, 115) if is_correct else (255, 52, 42)
    fill_h = max(24, int(bh * min(0.95, max(0.12, fill_ratio))))
    fy0 = by1 - fill_h
    gd.rounded_rectangle([bx0 - 6, fy0 - 6, bx1 + 6, by1 + 6], radius=14, fill=(*col, 180))
    cd.rounded_rectangle([bx0 + 3, fy0, bx1 - 3, by1 - 3], radius=9, fill=(*col, 255))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(core)


def render_single_frame(idx: int) -> int:
    t = idx / FPS
    pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.2 * t)

    # 3-Scene Multi-Angle O/X Timeline (15.0s total):
    # - Scene 1 (0.0 – 4.8s): Upper Body / Torso Posture Angle (0.0–2.4s ✕ -> 2.4–4.8s ✓)
    # - Scene 2 (4.8 – 9.6s): Elbow Pivot & Humerus Axis (4.8–7.2s ✕ -> 7.2–9.6s ✓)
    # - Scene 3 (9.6 – 15.0s): Wrist Bend vs Full Lockout & Rope Spread (9.6–12.0s ✕ -> 12.0–15.0s ✓)

    if t < 4.8:
        scene_idx = 0
        is_correct = (t >= 2.4)
        local_t = t if not is_correct else (t - 2.4)
        prog = smoothstep(0.0, 0.45, local_t)
        rep = 0.25 + 0.35 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))

        raw = sample_and_enhance(CORRECT_FRAMES if is_correct else WRONG_FRAMES, rep, not is_correct, pulse)
        z = lerp(1.16, 1.08, smoothstep(0.0, 0.8, local_t))
        framed_bgr, tf = crop_camera_view(raw, cx=730, cy=820, zoom=z)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            hip_pivot = tf((lerp(905, 885, rep), lerp(990, 1000, rep)))
            deg_w = lerp(-105.0, -109.4, 0.5 + 0.5 * math.sin(2.0 * math.pi * 0.5 * local_t))
            draw_wrong_angle_protract_system(
                canvas,
                pivot=hip_pivot,
                deg_target=-83.0,
                deg_wrong=deg_w,
                ray_len=480.0,
                arc_radius=240.0,
                arrow_radius=370.0,
                deg_label="26.4°",
                label_radius=465.0,
                prog=prog,
            )
            bar_fill = 0.62 + 0.08 * math.sin(2.0 * math.pi * 1.2 * local_t)
        else:
            hip = tf((785, 965))
            mid_back = tf((765, 735))
            shoulder_top = tf((745, 515))
            draw_high_contrast_solid_ray(canvas, [hip, mid_back, shoulder_top], col_rgb=(40, 250, 120), width=12, prog=prog)
            draw_lock_on_target_ring(canvas, mid_back, r_outer=54, prog=prog)
            p_arr0 = (hip[0] + 78, hip[1] - 40)
            p_arr1 = (shoulder_top[0] + 78, shoulder_top[1] + 40)
            draw_linear_fitted_3d_arrow(canvas, p_arr0, p_arr1, col_rgb=(40, 250, 120), prog=prog, tail_w=16, neck_w=32, head_w=68, head_len=56)
            bar_fill = 0.28 + 0.05 * rep

    elif t < 9.6:
        scene_idx = 1
        is_correct = (t >= 7.2)
        local_t = (t - 4.8) if not is_correct else (t - 7.2)
        prog = smoothstep(0.0, 0.45, local_t)
        rep = 0.25 + 0.65 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))

        raw = sample_and_enhance(CORRECT_FRAMES if is_correct else WRONG_FRAMES, rep, not is_correct, pulse)
        z = lerp(1.24, 1.32, smoothstep(0.0, 0.6, local_t))
        framed_bgr, tf = crop_camera_view(raw, cx=680, cy=790, zoom=z)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            shoulder_pivot = tf((810, 680))
            deg_w = lerp(108.5, 111.8, rep)
            draw_wrong_angle_protract_system(
                canvas,
                pivot=shoulder_pivot,
                deg_target=92.0,
                deg_wrong=deg_w,
                ray_len=410.0,
                arc_radius=210.0,
                arrow_radius=325.0,
                deg_label="19.8°",
                label_radius=415.0,
                prog=prog,
                label_deg_offset=4.0,
            )
            bar_fill = 0.66 + 0.06 * rep
        else:
            shoulder = tf((775, 600))
            elbow = tf((735, 815))
            wrist_end = tf((lerp(465, 515, rep), lerp(890, 1045, rep)))

            deg_start_forearm = 172.0
            deg_end_forearm = math.degrees(math.atan2(wrist_end[1] - elbow[1], wrist_end[0] - elbow[0]))
            p_start_forearm = ray_pt(elbow, deg_start_forearm, 310.0)
            draw_high_contrast_dashed_ray(canvas, elbow, p_start_forearm, col_rgb=(180, 255, 220), width=7, prog=prog)

            draw_high_contrast_solid_ray(canvas, [shoulder, elbow, wrist_end], col_rgb=(40, 250, 120), width=12, prog=prog)
            draw_lock_on_target_ring(canvas, elbow, r_outer=54, prog=prog)

            draw_arc_fitted_3d_arrow(
                canvas,
                pivot=elbow,
                deg_start=deg_start_forearm,
                deg_end=deg_end_forearm,
                radius=245.0,
                col_rgb=(40, 250, 120),
                prog=min(1.0, 0.35 + 0.75 * rep),
                tail_w=16.0,
                neck_w=34.0,
                head_w=72.0,
                head_len_deg=12.0,
            )
            bar_fill = 0.30 + 0.06 * rep

    else:
        scene_idx = 2
        is_correct = (t >= 12.0)
        local_t = (t - 9.6) if not is_correct else (t - 12.0)
        prog = smoothstep(0.0, 0.45, local_t)

        if not is_correct:
            rep = 0.45 + 0.50 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 2.4)))
            raw = sample_and_enhance(WRONG_FRAMES, rep, True, pulse)
            z = lerp(1.42, 1.50, smoothstep(0.0, 0.6, local_t))
            framed_bgr, tf = crop_camera_view(raw, cx=600, cy=930, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            elbow_in = tf((825, 885))
            wrist_pivot = tf((535, 1005))
            draw_high_contrast_solid_ray(canvas, [elbow_in, wrist_pivot], col_rgb=(255, 140, 60), width=10, prog=prog)
            draw_wrong_angle_protract_system(
                canvas,
                pivot=wrist_pivot,
                deg_target=152.0,
                deg_wrong=174.5,
                ray_len=360.0,
                arc_radius=185.0,
                arrow_radius=275.0,
                deg_label="22.5°",
                label_radius=355.0,
                prog=prog,
                label_deg_offset=-2.0,
            )
            bar_fill = 0.70 + 0.06 * rep
        else:
            # 2 full peak-contraction reps in 3.0s
            rep = 0.40 + 0.60 * (0.5 - 0.5 * math.cos(2.0 * math.pi * (local_t / 1.5)))
            raw = sample_and_enhance(CORRECT_FRAMES, rep, False, pulse)
            z = lerp(1.44, 1.50, smoothstep(0.0, 0.5, local_t))
            framed_bgr, tf = crop_camera_view(raw, cx=600, cy=930, zoom=z)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            elbow = tf((735, 845))
            rope_apex = tf((515, 965))
            hand_L = tf((lerp(395, 355, rep), lerp(1015, 1065, rep)))
            hand_R = tf((lerp(610, 660, rep), lerp(1110, 1165, rep)))

            draw_high_contrast_solid_ray(canvas, [elbow, rope_apex], col_rgb=(40, 250, 120), width=12, prog=prog)
            draw_high_contrast_solid_ray(canvas, [rope_apex, hand_L], col_rgb=(40, 250, 120), width=11, prog=prog)
            draw_high_contrast_solid_ray(canvas, [rope_apex, hand_R], col_rgb=(40, 250, 120), width=11, prog=prog)
            draw_lock_on_target_ring(canvas, rope_apex, r_outer=50, prog=prog)

            spread_p = min(1.0, max(0.25, rep))
            draw_linear_fitted_3d_arrow(
                canvas,
                p0=(hand_L[0] + 65, hand_L[1] + 55),
                p2=(hand_L[0] - 95, hand_L[1] + 150),
                col_rgb=(40, 250, 120),
                prog=spread_p,
                tail_w=15,
                neck_w=32,
                head_w=68,
                head_len=54,
            )
            draw_linear_fitted_3d_arrow(
                canvas,
                p0=(hand_R[0] - 45, hand_R[1] + 35),
                p2=(hand_R[0] + 115, hand_R[1] + 135),
                col_rgb=(40, 250, 120),
                prog=spread_p,
                tail_w=15,
                neck_w=32,
                head_w=68,
                head_len=54,
            )
            bar_fill = 0.28 + 0.06 * rep

    draw_top_silhouette_capsule(canvas, active_idx=scene_idx, is_correct=is_correct)
    draw_ox_status_badge(canvas, is_correct=is_correct, pulse=pulse)
    draw_left_slim_vertical_bar(canvas, is_correct=is_correct, fill_ratio=bar_fill)

    out_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}.jpg"), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return idx


def build_audio_track(wav_path: str):
    sr = 44100
    n_samples = int(sr * DURATION)
    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32) / sr

    # Clean driving studio sub-bass pulse
    beat_env = 0.5 + 0.5 * np.sin(2.0 * math.pi * 1.333 * t)
    sub = 0.08 * np.sin(2.0 * np.pi * 54.0 * t) * beat_env
    left += sub
    right += sub

    # Subtle warning tension pulse during ✕ segments (0.0-2.4s, 4.8-7.2s, 9.6-12.0s)
    for (s0, s1) in [(0.1, 2.3), (4.9, 7.1), (9.7, 11.9)]:
        mask = (t >= s0) & (t <= s1)
        buzz = 0.055 * np.sin(2.0 * math.pi * 92.0 * t[mask]) + 0.025 * np.sin(2.0 * math.pi * 184.0 * t[mask])
        left[mask] += buzz
        right[mask] += buzz * 0.9

    # Camera scene transition whoosh at 4.8s and 9.6s
    for t_cut in [4.8, 9.6]:
        mask = (t >= t_cut - 0.12) & (t < t_cut + 0.22)
        dt = t[mask] - (t_cut - 0.12)
        whoosh = 0.14 * np.sin(math.pi * dt / 0.34) * np.sin(2.0 * math.pi * (180.0 + 320.0 * dt) * dt)
        left[mask] += whoosh
        right[mask] += whoosh

    # Crisp Lock-On Snap + Positive Chime at ✓ transitions (2.4s, 7.2s, 12.0s)
    for t_snap in [2.4, 7.2, 12.0]:
        mask = (t >= t_snap) & (t < t_snap + 0.45)
        dt = t[mask] - t_snap
        snap = 0.25 * np.exp(-dt * 28.0) * np.sin(2.0 * math.pi * (880.0 + 440.0 * np.exp(-dt * 40.0)) * dt)
        chime = 0.16 * np.exp(-dt * 6.5) * np.sin(2.0 * math.pi * 659.25 * dt)
        left[mask] += snap + chime
        right[mask] += snap + chime

    # Final lockout harmonic chord at 12.2s
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
    print(f"Rendering {TOTAL_FRAMES} frames (15.0s @ 30fps)...")
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

    # Save 6-panel keyframe verification grid from the rendered video frames
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
    print("DONE! Saved:", OUT_MP4, OUT_KEYFRAMES)


if __name__ == "__main__":
    main()
