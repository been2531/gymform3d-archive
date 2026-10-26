#!/usr/bin/env python3
"""
Builds the 3-Scene O/X (✕ vs ✓) Comparison Preview Sheet (v2 — High-Visibility & 100% Angle-Fitted Arrows):
Addresses 중현 님's feedback:
1. "각도나 화살표나 잘보여야지":
   - Every alignment line, dashed reference ray, angle sector, and 3D arrow has a dark high-contrast outline/shadow backing + bright specular core so it pops with 100% clarity even over glowing red/orange anatomy.
   - Degree numbers (ONLY on ✕ panels: 26.4°, 19.8°, 22.5°) are rendered in crisp high-contrast dark-glass pills with glowing neon borders and bold white/red-accent typography.
2. "그리고 맞는 각도에 화살표가 fit하게 그려져야 하지 않을까?":
   - Every 3D arrow is mathematically parameterized on the EXACT circular arc (or coaxial ray) centered at the anatomical joint pivot (Hip pivot, Shoulder/Elbow pivot, Wrist/Rope pivot), starting flush on one angle ray and terminating flush on the target angle ray!
"""
import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
OUT_DIR = "/workspace/daily_info_reels/output"
WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")

OUT_SHEET = os.path.join(OUT_DIR, "day1_3scene_ox_preview_sheet.jpg")


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


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def enhance_plate(base_bgr: np.ndarray, is_wrong: bool, boost_level: float = 0.65) -> np.ndarray:
    out = base_bgr.astype(np.float32)
    hsv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 80, 80), (25, 255, 255)) | cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
    red_float = cv2.GaussianBlur(red_mask.astype(np.float32) / 255.0, (43, 43), 0)[:, :, None]

    if is_wrong:
        out = np.clip(out + red_float * np.array([15.0, 45.0, 255.0], dtype=np.float32) * (0.28 + 0.25 * boost_level), 0, 255)
    else:
        out = np.clip(out + red_float * np.array([20.0, 115.0, 255.0], dtype=np.float32) * (0.35 + 0.38 * boost_level), 0, 255)
        grn_mask = cv2.inRange(hsv, (35, 65, 65), (95, 255, 255))
        grn_float = cv2.GaussianBlur(grn_mask.astype(np.float32) / 255.0, (37, 37), 0)[:, :, None]
        out = np.clip(out + grn_float * np.array([85.0, 255.0, 40.0], dtype=np.float32) * (0.35 + 0.35 * boost_level), 0, 255)

    return out.astype(np.uint8)


def crop_camera_view(frame_bgr: np.ndarray, cx: int, cy: int, zoom: float) -> tuple:
    if zoom <= 1.001:
        return frame_bgr.copy(), (lambda pt: (float(pt[0]), float(pt[1])))
    cw = int(round(W / zoom))
    ch = int(round(H / zoom))
    x0 = max(0, min(W - cw, int(round(cx - cw / 2))))
    y0 = max(0, min(H - ch, int(round(cy - ch / 2))))
    cropped = frame_bgr[y0 : y0 + ch, x0 : x0 + cw]
    resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

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
    col_rgb: tuple = (235, 250, 255),
    width: int = 7,
    dash_len: int = 24,
    gap_len: int = 14,
):
    """High-visibility dashed reference ray with dark outline casing so it never disappears over bright areas."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
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
        # Dark casing first
        d.line([s0, s1], fill=(5, 8, 14, 235), width=width + 6)
        # Bright core
        d.line([s0, s1], fill=(*col_rgb, 250), width=width)
        cur += dash_len + gap_len

    canvas_rgba.alpha_composite(ov)


def draw_high_contrast_solid_ray(
    canvas_rgba: Image.Image,
    pts: list,
    col_rgb: tuple,
    width: int = 11,
):
    """High-visibility solid anatomical alignment ray with dark contrast casing, neon bloom, and white core."""
    if len(pts) < 2:
        return
    casing = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(casing)
    cd.line(pts, fill=(5, 8, 14, 230), width=width + 10, joint="curve")

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line(pts, fill=(*col_rgb, 225), width=width * 3, joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))

    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    crd = ImageDraw.Draw(core)
    crd.line(pts, fill=(*col_rgb, 255), width=width, joint="curve")
    crd.line(pts, fill=(255, 255, 255, 250), width=max(3, width // 3), joint="curve")

    for (px, py) in pts:
        r = width + 4
        cd.ellipse([px - r - 4, py - r - 4, px + r + 4, py + r + 4], fill=(5, 8, 14, 240))
        crd.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255, 255), outline=(*col_rgb, 255), width=4)

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
    tail_w: float = 16.0,
    neck_w: float = 34.0,
    head_w: float = 74.0,
    head_len_deg: float = 9.5,
):
    """
    Draws a sculpted 3D volumetric tapered ribbon arrow that FITS 100% snugly along the circular arc
    centered at `pivot` with radius `radius`, starting flush at `deg_start` ray and ending with its tip
    touching the `deg_end` ray!
    Includes a thick dark contrast outline + drop shadow + neon bloom + bright white spine so it is unmistakable.
    """
    total_span = deg_end - deg_start
    if abs(total_span) < 2.0:
        return
    sign = 1.0 if total_span > 0 else -1.0
    # Reserve angular room for the arrowhead so the tip lands EXACTLY on `deg_end`
    actual_head_deg = min(abs(total_span) * 0.42, head_len_deg) * sign
    neck_deg = deg_end - actual_head_deg

    steps = 36
    left_pts, right_pts, spine_pts = [], [], []
    for i in range(steps + 1):
        u = i / steps
        ang = lerp(deg_start, neck_deg, u)
        rad = math.radians(ang)
        cx = pivot[0] + radius * math.cos(rad)
        cy = pivot[1] + radius * math.sin(rad)
        # Radial normal (pointing outward from pivot)
        nx, ny = math.cos(rad), math.sin(rad)
        half_w = 0.5 * lerp(tail_w, neck_w, u ** 0.85)
        left_pts.append((cx + nx * half_w, cy + ny * half_w))
        right_pts.append((cx - nx * half_w, cy - ny * half_w))
        spine_pts.append((cx, cy))

    # Arrowhead at neck_deg -> tip on deg_end
    rad_neck = math.radians(neck_deg)
    rad_tip = math.radians(deg_end)
    nx, ny = math.cos(rad_neck), math.sin(rad_neck)
    # Tangent direction along sweep
    tx, ty = -math.sin(rad_neck) * sign, math.cos(rad_neck) * sign
    neck_cx = pivot[0] + radius * math.cos(rad_neck)
    neck_cy = pivot[1] + radius * math.sin(rad_neck)

    hw = 0.5 * head_w
    back_sweep = 10.0
    wing_L = (neck_cx - tx * back_sweep + nx * hw, neck_cy - ty * back_sweep + ny * hw)
    wing_R = (neck_cx - tx * back_sweep - nx * hw, neck_cy - ty * back_sweep - ny * hw)
    tip = (pivot[0] + radius * math.cos(rad_tip), pivot[1] + radius * math.sin(rad_tip))

    poly = left_pts + [wing_L, tip, wing_R] + right_pts[::-1]
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    pad = 54
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

    # 1. Dark drop shadow & dark contrast casing
    sh_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh_im)
    sd.polygon(sh_poly, fill=(0, 0, 0, 210))
    sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=7 * SS))

    casing_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(casing_im).polygon(local_poly, fill=(5, 8, 14, 245), outline=(5, 8, 14, 250), width=6 * SS)

    # 2. Neon bloom
    glow_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(glow_im).polygon(local_poly, fill=(*col_rgb, 210))
    glow_im = glow_im.filter(ImageFilter.GaussianBlur(radius=9 * SS))

    # 3. Crisp 3D arrow body + white bevel + bright spine
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
    tail_w: float = 14.0,
    neck_w: float = 30.0,
    head_w: float = 66.0,
    head_len: float = 54.0,
):
    """Draws a straight/coaxial 3D arrow aligned directly with a vector p0 -> p2, with high-contrast dark casing."""
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 20.0:
        return
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx
    hl = min(head_len, L * 0.42)
    neck_L = L - hl

    steps = 24
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
    hw = 0.5 * head_w
    back_sweep = 10.0
    wing_L = (neck_x - tx * back_sweep + nx * hw, neck_y - ty * back_sweep + ny * hw)
    wing_R = (neck_x - tx * back_sweep - nx * hw, neck_y - ty * back_sweep - ny * hw)
    tip = p2

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
    sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=7 * SS))

    casing_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(casing_im).polygon(local_poly, fill=(5, 8, 14, 245), outline=(5, 8, 14, 250), width=5 * SS)

    glow_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
    ImageDraw.Draw(glow_im).polygon(local_poly, fill=(*col_rgb, 205))
    glow_im = glow_im.filter(ImageFilter.GaussianBlur(radius=8 * SS))

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
    label_deg_offset: float = 0.0,
):
    """
    Unified, 100% Angle-Fitted Protractor + Arrow System for ✕ (Wrong Posture) scenes:
    1. Angle sector fill + glowing inner arc at `arc_radius` between `deg_target` and `deg_wrong`
    2. High-contrast dashed white/green Target Ray at `deg_target` (the "맞는 각도")
    3. High-contrast solid Red Wrong Ray at `deg_wrong` (the current misaligned posture)
    4. Sculpted 3D Arrow fitted directly along the circular arc at `arrow_radius` from `deg_wrong` -> `deg_target` (or showing the angular error)
    5. Ultra-legible Degree Pill Badge (`26.4°`, `19.8°`, `22.5°`) anchored right at the angle!
    """
    a0 = min(deg_target, deg_wrong)
    a1 = max(deg_target, deg_wrong)

    # 1. Translucent angle sector & inner protractor arc
    sec = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sec)
    bbox_arc = [pivot[0] - arc_radius, pivot[1] - arc_radius, pivot[0] + arc_radius, pivot[1] + arc_radius]
    sd.pieslice(bbox_arc, start=a0, end=a1, fill=(255, 42, 32, 95))
    sd.arc(bbox_arc, start=a0, end=a1, fill=(5, 8, 14, 240), width=12)
    sd.arc(bbox_arc, start=a0, end=a1, fill=(255, 190, 180, 255), width=6)
    canvas_rgba.alpha_composite(sec)

    # 2. Target Ray ("맞는 각도" — bright dashed cyan-white/green line)
    p_target = ray_pt(pivot, deg_target, ray_len)
    p_target_back = ray_pt(pivot, deg_target + 180.0, 65.0)
    draw_high_contrast_dashed_ray(canvas_rgba, p_target_back, p_target, col_rgb=(180, 255, 220), width=8, dash_len=24, gap_len=14)

    # 3. Wrong Ray ("틀어진 각도" — glowing Red solid line with dark casing)
    p_wrong = ray_pt(pivot, deg_wrong, ray_len)
    draw_high_contrast_solid_ray(canvas_rgba, [pivot, p_wrong], col_rgb=(255, 48, 38), width=12)

    # 4. Angle-Fitted 3D Arrow sweeping along the exact circular arc between the two rays!
    draw_arc_fitted_3d_arrow(
        canvas_rgba,
        pivot=pivot,
        deg_start=deg_wrong,
        deg_end=deg_target,
        radius=arrow_radius,
        col_rgb=(255, 58, 45),
        tail_w=18.0,
        neck_w=36.0,
        head_w=76.0,
        head_len_deg=max(10.0, abs(deg_target - deg_wrong) * 0.38),
    )

    # 5. High-Contrast Degree Readout Badge (ONLY on ✕!)
    mid_deg = 0.5 * (deg_target + deg_wrong) + label_deg_offset
    lx, ly = ray_pt(pivot, mid_deg, label_radius)
    font = get_font(56)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    tb = od.textbbox((0, 0), deg_label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad_x, pad_y = 24, 14
    bx0 = int(lx - tw / 2 - pad_x)
    by0 = int(ly - th / 2 - pad_y)
    bx1 = int(lx + tw / 2 + pad_x)
    by1 = int(ly + th / 2 + pad_y)

    # Outer red glow + dark high-contrast pill + crisp white-red border + bold white text
    glow_pill = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow_pill).rounded_rectangle([bx0 - 8, by0 - 8, bx1 + 8, by1 + 8], radius=24, fill=(255, 45, 35, 180))
    glow_pill = glow_pill.filter(ImageFilter.GaussianBlur(radius=10))
    canvas_rgba.alpha_composite(glow_pill)

    od.rounded_rectangle([bx0, by0, bx1, by1], radius=20, fill=(10, 14, 20, 240), outline=(255, 75, 65, 255), width=4)
    od.text(( int(lx - tw / 2 - tb[0]), int(ly - th / 2 - tb[1]) ), deg_label, font=font, fill=(255, 255, 255, 255))
    canvas_rgba.alpha_composite(ov)


def draw_lock_on_target_ring(canvas_rgba: Image.Image, pt: tuple, r_outer: int = 54):
    """High-contrast Neon-Green concentric Lock-On Target Ring at a stabilized joint pivot."""
    cx, cy = int(pt[0]), int(pt[1])
    grn = (40, 255, 120)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)

    # Dark casing behind rings
    cd.ellipse([cx - r_outer - 4, cy - r_outer - 4, cx + r_outer + 4, cy + r_outer + 4], outline=(5, 8, 14, 230), width=12)
    gd.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(*grn, 220), width=14)
    cd.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(230, 255, 240, 255), width=5)

    r_in = int(r_outer * 0.62)
    cd.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], outline=(*grn, 245), width=4)

    # Crosshair ticks
    for deg in [0, 90, 180, 270]:
        rad = math.radians(deg)
        x0, y0 = cx + (r_outer - 12) * math.cos(rad), cy + (r_outer - 12) * math.sin(rad)
        x1, y1 = cx + (r_outer + 12) * math.cos(rad), cy + (r_outer + 12) * math.sin(rad)
        cd.line([(x0, y0), (x1, y1)], fill=(5, 8, 14, 235), width=8)
        cd.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, 255), width=4)

    cd.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(255, 255, 255, 255), outline=(*grn, 255), width=4)

    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))
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


def draw_ox_status_badge(canvas_rgba: Image.Image, is_correct: bool, cx: int = 925, cy: int = 235, r: int = 58):
    halo = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    col = (40, 245, 115) if is_correct else (255, 48, 38)
    hd.ellipse([cx - r - 18, cy - r - 18, cx + r + 18, cy + r + 18], fill=(*col, 165))
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


def draw_left_slim_vertical_bar(canvas_rgba: Image.Image, is_correct: bool):
    bx0, by0 = 58, 720
    bw, bh = 24, 460
    bx1, by1 = bx0 + bw, by0 + bh

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    core = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)

    cd.rounded_rectangle([bx0, by0, bx1, by1], radius=12, fill=(14, 18, 24, 230), outline=(255, 255, 255, 90), width=2)

    if not is_correct:
        col = (255, 52, 42)
        fill_h = int(bh * 0.66)
        fy0 = by1 - fill_h
        gd.rounded_rectangle([bx0 - 6, fy0 - 6, bx1 + 6, by1 + 6], radius=14, fill=(*col, 180))
        cd.rounded_rectangle([bx0 + 3, fy0, bx1 - 3, by1 - 3], radius=9, fill=(*col, 255))
    else:
        col = (40, 245, 115)
        fill_h = int(bh * 0.30)
        fy0 = by1 - fill_h
        gd.rounded_rectangle([bx0 - 6, fy0 - 6, bx1 + 6, by1 + 6], radius=14, fill=(*col, 180))
        cd.rounded_rectangle([bx0 + 3, fy0, bx1 - 3, by1 - 3], radius=9, fill=(*col, 255))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=9))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(core)


def build_scene_panel(scene_idx: int, is_correct: bool) -> np.ndarray:
    """
    Renders one 1080x1920 panel for Scene `scene_idx` in {0, 1, 2} and `is_correct` in {False, True}.
    Every angle and arrow is 100% fitted to the actual anatomical joint pivot and limb rays!
    """
    if scene_idx == 0:
        # =========================================================================
        # SCENE 1: TORSO / BACK LEAN ANGLE (Hip Pivot) — Wide Side View (Zoom 1.08x)
        # =========================================================================
        raw = CORRECT_FRAMES[18] if is_correct else WRONG_FRAMES[22]
        enh = enhance_plate(raw, is_wrong=not is_correct, boost_level=0.65)
        framed_bgr, tf = crop_camera_view(enh, cx=730, cy=820, zoom=1.08)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            # Pivot at Hip/Pelvis (895, 995):
            # - Target Upright Spine Angle = -83.0° (nearly vertical slight athletic tilt)
            # - Wrong Hunched Torso Angle = -109.4° (leaning forward by 26.4°!)
            # - The 3D curved arrow sweeps directly along the 26.4° circular arc between the two rays!
            hip_pivot = tf((895, 995))
            draw_wrong_angle_protract_system(
                canvas,
                pivot=hip_pivot,
                deg_target=-83.0,
                deg_wrong=-109.4,
                ray_len=480.0,
                arc_radius=240.0,
                arrow_radius=370.0,
                deg_label="26.4°",
                label_radius=465.0,
            )
        else:
            # Correct: Torso locked upright right along the -98° thoracic-to-hip alignment line!
            # NO degree numbers! Crisp Neon-Green alignment ray + Lock-On Target Ring + coaxial posture arrow!
            hip = tf((785, 965))
            mid_back = tf((765, 735))
            shoulder_top = tf((745, 515))
            draw_high_contrast_solid_ray(canvas, [hip, mid_back, shoulder_top], col_rgb=(40, 250, 120), width=12)
            draw_lock_on_target_ring(canvas, mid_back, r_outer=54)
            # Coaxial 3D green arrow fitted parallel right beside the locked spine axis
            p_arr0 = (hip[0] + 78, hip[1] - 40)
            p_arr1 = (shoulder_top[0] + 78, shoulder_top[1] + 40)
            draw_linear_fitted_3d_arrow(canvas, p_arr0, p_arr1, col_rgb=(40, 250, 120), tail_w=16, neck_w=32, head_w=68, head_len=56)

    elif scene_idx == 1:
        # =========================================================================
        # SCENE 2: ELBOW PIVOT & HUMERUS AXIS (Medium Upper-Arm Zoom, 1.32x)
        # =========================================================================
        raw = CORRECT_FRAMES[32] if is_correct else WRONG_FRAMES[34]
        enh = enhance_plate(raw, is_wrong=not is_correct, boost_level=0.75)
        framed_bgr, tf = crop_camera_view(enh, cx=680, cy=790, zoom=1.32)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            # Pivot at Shoulder Joint (810, 680):
            # - Target Pinned Upper-Arm Ray = +92.0° (straight vertical down the ribcage)
            # - Wrong Flared/Forward Upper-Arm Ray = +111.8° (drifted forward by 19.8°!)
            # - 3D Arrow sweeps directly along the circular arc between the Wrong Ray and Target Ray!
            shoulder_pivot = tf((810, 680))
            draw_wrong_angle_protract_system(
                canvas,
                pivot=shoulder_pivot,
                deg_target=92.0,
                deg_wrong=111.8,
                ray_len=410.0,
                arc_radius=210.0,
                arrow_radius=325.0,
                deg_label="19.8°",
                label_radius=415.0,
                label_deg_offset=4.0,
            )
        else:
            # Correct: Shoulder -> Elbow is locked vertical, and Forearm rotates cleanly around the Pinned Elbow Pivot!
            shoulder = tf((775, 600))
            elbow = tf((735, 815))
            wrist_end = tf((515, 1045))

            # Dashed starting forearm ray (at 90° flexion) to show the exact joint hinge angle
            deg_start_forearm = 172.0
            deg_end_forearm = math.degrees(math.atan2(wrist_end[1] - elbow[1], wrist_end[0] - elbow[0]))
            p_start_forearm = ray_pt(elbow, deg_start_forearm, 310.0)
            draw_high_contrast_dashed_ray(canvas, elbow, p_start_forearm, col_rgb=(180, 255, 220), width=7)

            # Solid Neon-Green locked Humerus + extended Forearm rays
            draw_high_contrast_solid_ray(canvas, [shoulder, elbow, wrist_end], col_rgb=(40, 250, 120), width=12)
            draw_lock_on_target_ring(canvas, elbow, r_outer=54)

            # 3D Neon-Green Arrow fitted 100% on the circular arc centered at the Pinned Elbow Pivot,
            # sweeping from the dashed start forearm ray right onto the locked-out green forearm ray!
            draw_arc_fitted_3d_arrow(
                canvas,
                pivot=elbow,
                deg_start=deg_start_forearm,
                deg_end=deg_end_forearm,
                radius=245.0,
                col_rgb=(40, 250, 120),
                tail_w=16.0,
                neck_w=34.0,
                head_w=72.0,
                head_len_deg=12.0,
            )

    else:
        # =========================================================================
        # SCENE 3: WRIST BEND vs FULL LOCKOUT & ROPE SPREAD (Close-Up Zoom, 1.50x)
        # =========================================================================
        raw = CORRECT_FRAMES[44] if is_correct else WRONG_FRAMES[44]
        enh = enhance_plate(raw, is_wrong=not is_correct, boost_level=0.85)
        framed_bgr, tf = crop_camera_view(enh, cx=600, cy=930, zoom=1.50)
        canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

        if not is_correct:
            # Pivot at Wrist Joint (535, 1005):
            # - Forearm comes in from Elbow (825, 885) -> Wrist (535, 1005)
            # - Target Neutral Wrist Ray continues straight along forearm angle (+156.0°)
            # - Wrong Bent-Back Wrist Ray bends upward to +178.5° (22.5° hyperextension!)
            elbow_in = tf((825, 885))
            wrist_pivot = tf((535, 1005))
            draw_high_contrast_solid_ray(canvas, [elbow_in, wrist_pivot], col_rgb=(255, 140, 60), width=10)
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
                label_deg_offset=-2.0,
            )
        else:
            # Correct: Full lockout straight line + V-angle Rope Spread fitted arrows!
            elbow = tf((735, 845))
            rope_apex = tf((515, 965))
            hand_L = tf((355, 1065))
            hand_R = tf((660, 1165))

            # Glowing green locked extension & V-spread rays from rope apex to both hands
            draw_high_contrast_solid_ray(canvas, [elbow, rope_apex], col_rgb=(40, 250, 120), width=12)
            draw_high_contrast_solid_ray(canvas, [rope_apex, hand_L], col_rgb=(40, 250, 120), width=11)
            draw_high_contrast_solid_ray(canvas, [rope_apex, hand_R], col_rgb=(40, 250, 120), width=11)
            draw_lock_on_target_ring(canvas, rope_apex, r_outer=50)

            # Dual 3D Neon-Green Arrows fitted directly along the left and right rope-spread rays!
            draw_linear_fitted_3d_arrow(
                canvas,
                p0=( hand_L[0] + 65, hand_L[1] + 55 ),
                p2=( hand_L[0] - 95, hand_L[1] + 150 ),
                col_rgb=(40, 250, 120),
                tail_w=15,
                neck_w=32,
                head_w=68,
                head_len=54,
            )
            draw_linear_fitted_3d_arrow(
                canvas,
                p0=( hand_R[0] - 45, hand_R[1] + 35 ),
                p2=( hand_R[0] + 115, hand_R[1] + 135 ),
                col_rgb=(40, 250, 120),
                tail_w=15,
                neck_w=32,
                head_w=68,
                head_len=54,
            )

    # Common clean UI chrome
    draw_top_silhouette_capsule(canvas, active_idx=scene_idx, is_correct=is_correct)
    draw_ox_status_badge(canvas, is_correct=is_correct)
    draw_left_slim_vertical_bar(canvas, is_correct=is_correct)

    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pw, ph = 540, 960
    top_row = []
    bot_row = []

    for s_idx in range(3):
        w_bgr = build_scene_panel(s_idx, is_correct=False)
        c_bgr = build_scene_panel(s_idx, is_correct=True)
        cv2.imwrite(os.path.join(OUT_DIR, f"day1_scene{s_idx+1}_wrong.jpg"), w_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(os.path.join(OUT_DIR, f"day1_scene{s_idx+1}_correct.jpg"), c_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        top_row.append(cv2.resize(w_bgr, (pw, ph), interpolation=cv2.INTER_AREA))
        bot_row.append(cv2.resize(c_bgr, (pw, ph), interpolation=cv2.INTER_AREA))

    v_sep = np.full((ph, 8, 3), 12, dtype=np.uint8)
    r_top = np.hstack([top_row[0], v_sep, top_row[1], v_sep, top_row[2]])
    r_bot = np.hstack([bot_row[0], v_sep, bot_row[1], v_sep, bot_row[2]])
    h_sep = np.full((8, r_top.shape[1], 3), 12, dtype=np.uint8)
    grid = np.vstack([r_top, h_sep, r_bot])

    cv2.imwrite(OUT_SHEET, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("Saved 3x2 sheet v2:", OUT_SHEET, grid.shape)


if __name__ == "__main__":
    main()
