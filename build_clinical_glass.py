#!/usr/bin/env python3
"""
Reusable clinical biomechanical overlay & video reel builder for translucent frosted-glass anatomy.
Style: @sciepro / @voka.io / @demicstory clinical upgrade.
"""
import os
import math
import wave
import struct
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1920
FPS = 30
DURATION = 8.0
TOTAL_FRAMES = int(FPS * DURATION)

RAW_PATH = "/workspace/info_concepts_r1/c1_glass_sleep.png"
OUT_DIR = "/workspace/daily_info_reels/output"
OUT_PNG = os.path.join(OUT_DIR, "clinical_glass_day1_sleep.png")
OUT_MP4 = os.path.join(OUT_DIR, "clinical_glass_day1_sleep.mp4")
C1_FINAL_JPG = "/workspace/info_concepts_r1/c1_final.jpg"


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/home/box/.fonts/NotoSansKR[wght].ttf",
        "/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-ExtraBold.ttf" if bold else "/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=size)
                if "NotoSansKR[wght]" in path:
                    try:
                        font.set_variation_by_name("Bold" if bold else "Medium")
                    except Exception:
                        pass
                return font
            except Exception:
                continue
    return ImageFont.load_default()


def prepare_clean_50_50_base(raw_path: str) -> np.ndarray:
    """
    Loads the 4K raw image, removes outer white borders and re-frames into a symmetric
    50/50 split (Top half y=0..960, Bottom half y=960..1920) with zero stray green checkmarks
    in the top half. Also inpaints any stray green pixels in the top half just in case.
    Returns BGR uint8 image (1920, 1080, 3).
    """
    raw = cv2.imread(raw_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read {raw_path}")
    rh, rw = raw.shape[:2]

    # Scale coordinates from 1080x1920 reference space to 4K raw space
    sx = rw / 1080.0
    sy = rh / 1920.0

    # Top half: crop y=236..1196 (960px tall in 1080p space), x=14..1066 (removes outer white frame)
    x0, x1 = int(14 * sx), int(1066 * sx)
    ty0, ty1 = int(236 * sy), int(1196 * sy)
    top_crop = cv2.resize(raw[ty0:ty1, x0:x1], (W, H // 2), interpolation=cv2.INTER_AREA)

    # Clean/inpaint any stray green pixels in the top half so only red/orange + neutral glass exist
    hsv_top = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
    stray_green = cv2.inRange(hsv_top, (35, 70, 60), (95, 255, 255))
    if np.count_nonzero(stray_green) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        stray_green = cv2.dilate(stray_green, kernel, iterations=1)
        top_crop = cv2.inpaint(top_crop, stray_green, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Bottom half: raw bottom panel is y=1216..1904 (688px tall in 1080p space)
    by0, by1 = int(1216 * sy), int(1904 * sy)
    bot_H_body = 696
    bot_pad_top = (H // 2) - bot_H_body  # 264px of clean dark studio headroom
    bot_body = cv2.resize(raw[by0:by1, x0:x1], (W, bot_H_body), interpolation=cv2.INTER_AREA)

    # Extend clean dark studio background seamlessly above bottom body
    bg_strip = cv2.resize(bot_body[16:72, :], (W, bot_pad_top), interpolation=cv2.INTER_LINEAR)
    # Add subtle vertical vignette to top of bg_strip so it looks like a studio backdrop
    grad = np.linspace(0.82, 1.0, bot_pad_top)[:, None, None]
    bg_strip = np.clip(bg_strip.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    bot_crop = np.vstack([bg_strip, bot_body])

    # Smooth seam at y = bot_pad_top
    for dy in range(-24, 24):
        y = bot_pad_top + dy
        alpha = (dy + 24) / 48.0
        bot_crop[y] = (
            bg_strip[-1].astype(np.float32) * (1.0 - alpha)
            + bot_body[max(0, 16 + dy)].astype(np.float32) * alpha
        ).astype(np.uint8)

    base = np.vstack([top_crop, bot_crop])
    return base


def build_glow_layers(base_bgr: np.ndarray):
    """
    Precompute extra emissive glow layers for Top (red-orange C4-C7 compression)
    and Bottom (emerald C1-C7 neutral alignment) to pulse smoothly in the video.
    """
    top_roi = base_bgr[: H // 2].astype(np.float32)
    # Isolate red-orange glow in top half
    r_excess = np.clip(top_roi[:, :, 2] - top_roi[:, :, 0] * 0.75, 0, 255)
    top_glow = np.zeros_like(top_roi)
    top_glow[:, :, 2] = r_excess * 1.15  # Red
    top_glow[:, :, 1] = r_excess * 0.38  # Orange tint
    top_glow[:, :, 0] = r_excess * 0.08
    top_glow = cv2.GaussianBlur(top_glow, (0, 0), sigmaX=18, sigmaY=18)

    bot_roi = base_bgr[H // 2 :].astype(np.float32)
    # Isolate emerald-teal glow in bottom half
    g_excess = np.clip(bot_roi[:, :, 1] - bot_roi[:, :, 2] * 0.65, 0, 255)
    bot_glow = np.zeros_like(bot_roi)
    bot_glow[:, :, 1] = g_excess * 1.10  # Emerald green
    bot_glow[:, :, 0] = g_excess * 0.65  # Teal tint
    bot_glow[:, :, 2] = g_excess * 0.12
    bot_glow = cv2.GaussianBlur(bot_glow, (0, 0), sigmaX=18, sigmaY=18)

    return top_glow, bot_glow


def draw_badge(img_pil: Image.Image, cx: int, cy: int, r: int, kind: str = "x", scale: float = 1.0):
    if scale <= 0.01:
        return img_pil
    r_eff = max(4, int(r * scale))
    glow = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    color = (255, 46, 36) if kind == "x" else (0, 228, 128)
    gd.ellipse(
        [cx - r_eff - 14, cy - r_eff - 14, cx + r_eff + 14, cy + r_eff + 14],
        fill=(*color, int(95 * min(1.0, scale))),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(12))
    img_pil = Image.alpha_composite(img_pil, glow)

    d = ImageDraw.Draw(img_pil)
    d.ellipse(
        [cx - r_eff, cy - r_eff, cx + r_eff, cy + r_eff],
        fill=(*color, 240),
        outline=(255, 255, 255, 235),
        width=max(2, int(4 * scale)),
    )
    s = int(r_eff * 0.42)
    w_line = max(2, int(10 * scale))
    if kind == "x":
        d.line([cx - s, cy - s, cx + s, cy + s], fill=(255, 255, 255, 255), width=w_line)
        d.line([cx - s, cy + s, cx + s, cy - s], fill=(255, 255, 255, 255), width=w_line)
    else:
        pts = [
            (cx - int(r_eff * 0.44), cy + int(r_eff * 0.04)),
            (cx - int(r_eff * 0.10), cy + int(r_eff * 0.38)),
            (cx + int(r_eff * 0.48), cy - int(r_eff * 0.34)),
        ]
        d.line([pts[0], pts[1]], fill=(255, 255, 255, 255), width=w_line)
        d.line([pts[1], pts[2]], fill=(255, 255, 255, 255), width=w_line)
    return img_pil


def draw_dashed_line(draw: ImageDraw.ImageDraw, p1, p2, color, width=3, dash_len=12, gap_len=8):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        end_pos = min(dist, pos + dash_len)
        x_s, y_s = p1[0] + ux * pos, p1[1] + uy * pos
        x_e, y_e = p1[0] + ux * end_pos, p1[1] + uy * end_pos
        draw.line([(x_s, y_s), (x_e, y_e)], fill=color, width=width)
        pos += dash_len + gap_len


def draw_small_arrow(draw: ImageDraw.ImageDraw, p1, p2, color, width=3, head_size=9):
    draw.line([p1, p2], fill=color, width=width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    a1 = ang + math.pi * 0.82
    a2 = ang - math.pi * 0.82
    h1 = (p2[0] + math.cos(a1) * head_size, p2[1] + math.sin(a1) * head_size)
    h2 = (p2[0] + math.cos(a2) * head_size, p2[1] + math.sin(a2) * head_size)
    draw.polygon([p2, h1, h2], fill=color)


def render_clinical_frame(
    base_bgr: np.ndarray,
    top_glow: np.ndarray,
    bot_glow: np.ndarray,
    fonts: dict,
    top_prog: float = 1.0,
    bot_prog: float = 1.0,
    top_pulse: float = 0.35,
    bot_pulse: float = 0.35,
) -> Image.Image:
    """
    Renders a single 1080x1920 clinical overlay frame with animation progress parameters.
    For the static high-res comparison PNG, top_prog=1.0 and bot_prog=1.0.
    """
    frame_f = base_bgr.astype(np.float32).copy()
    if top_pulse > 0:
        frame_f[: H // 2] = np.clip(frame_f[: H // 2] + top_glow * top_pulse, 0, 255)
    if bot_pulse > 0:
        frame_f[H // 2 :] = np.clip(frame_f[H // 2 :] + bot_glow * bot_pulse, 0, 255)

    img_pil = Image.fromarray(cv2.cvtColor(frame_f.astype(np.uint8), cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    gd = ImageDraw.Draw(glow_layer)

    # Center horizontal surgical divider line
    d.line([(0, H // 2), (W, H // 2)], fill=(220, 235, 255, 135), width=3)

    # =========================================================================
    # 1. TOP HALF (y = 0..960): ❌ Wrong Pillow Height (+18.4° Cervical Kink)
    # =========================================================================
    # Anatomical vertebral landmarks in top half (after 236px upward shift):
    # T4 -> T1 (Horizontal thoracic neutral axis at y=556) -> C7 -> C5 -> C2 -> Skull
    top_nodes = [
        (195, 556, "T4"),
        (305, 556, "T1"),
        (395, 556, "C7"),
        (485, 526, "C5"),
        (575, 496, "C2"),
        (685, 459, "Skull"),
    ]
    # Note: vector from C7 (395, 556) to Skull (685, 459) has dx=290, dy=-97 -> atan2(97, 290) = 18.49° (~18.4°)

    if top_prog > 0.01:
        tp_line = min(1.0, top_prog / 0.55)
        tp_arc = max(0.0, min(1.0, (top_prog - 0.25) / 0.45))
        tp_label = max(0.0, min(1.0, (top_prog - 0.40) / 0.45))

        # Horizontal neutral reference dashed line extending right from C7 (395, 556) to x=745
        ref_alpha = int(195 * tp_line)
        draw_dashed_line(
            d,
            (195, 556),
            (745, 556),
            color=(210, 225, 245, ref_alpha),
            width=3,
            dash_len=10,
            gap_len=8,
        )

        # Spine axis o-o-o segments (progressive draw)
        n_segs = len(top_nodes) - 1
        seg_float = tp_line * n_segs
        for i in range(n_segs):
            seg_p = max(0.0, min(1.0, seg_float - i))
            if seg_p <= 0:
                continue
            x0, y0, _ = top_nodes[i]
            x1, y1, _ = top_nodes[i + 1]
            xe = x0 + (x1 - x0) * seg_p
            ye = y0 + (y1 - y0) * seg_p
            is_cervical = i >= 2
            line_col = (255, 72, 42, 245) if is_cervical else (235, 242, 255, 215)
            glow_col = (255, 50, 20, 150) if is_cervical else (180, 210, 255, 80)
            gd.line([(x0, y0), (xe, ye)], fill=glow_col, width=14)
            d.line([(x0, y0), (xe, ye)], fill=line_col, width=5)

        # Draw joint nodes (o-o-o) and subtle segmental micro-tags
        for i, (nx, ny, v_tag) in enumerate(top_nodes):
            if tp_line * n_segs < i - 0.1:
                continue
            is_stress = i in (2, 3, 4)  # C7, C5, C2
            r_n = 9 if is_stress else 7
            fill_c = (255, 65, 35, 255) if is_stress else (240, 248, 255, 255)
            gd.ellipse([nx - r_n - 6, ny - r_n - 6, nx + r_n + 6, ny + r_n + 6], fill=(255, 60, 30, 130))
            d.ellipse(
                [nx - r_n, ny - r_n, nx + r_n, ny + r_n],
                fill=fill_c,
                outline=(255, 255, 255, 255),
                width=3,
            )
            if v_tag in ("T4", "C7", "C5", "C2"):
                d.text(
                    (nx - 14, ny + 14),
                    v_tag,
                    font=fonts["micro"],
                    fill=(255, 215, 205, int(225 * tp_line)),
                )

        # Angle Arc at C7 pivot (395, 556) showing +18.4° upward lateral flexion
        if tp_arc > 0.01:
            piv_x, piv_y = 395, 556
            arc_r = 215
            end_deg = 18.4 * tp_arc
            # In PIL, angles are clockwise from 3 o'clock; upward is -18.4° (341.6° to 360°)
            bbox = [piv_x - arc_r, piv_y - arc_r, piv_x + arc_r, piv_y + arc_r]
            # Shaded wedge fill between -end_deg and 0°
            gd.pieslice(bbox, start=360.0 - end_deg, end=360.0, fill=(255, 55, 30, int(65 * tp_arc)))
            d.pieslice(bbox, start=360.0 - end_deg, end=360.0, fill=(255, 60, 35, int(52 * tp_arc)))
            d.arc(bbox, start=360.0 - end_deg, end=360.0, fill=(255, 95, 60, int(255 * tp_arc)), width=5)

            # Compression arrows pinching toward C4-C7 intervertebral foramen (around 485, 526)
            c_alpha = int(245 * tp_arc)
            draw_small_arrow(d, (465, 472), (480, 508), (255, 85, 55, c_alpha), width=4, head_size=10)
            draw_small_arrow(d, (505, 580), (490, 544), (255, 85, 55, c_alpha), width=4, head_size=10)

            # Angle pill readout right beside the arc
            ang_val = 18.4 * tp_arc
            pill_x, pill_y = 626, 492
            pill_w, pill_h = 268, 52
            d.rounded_rectangle(
                [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                radius=10,
                fill=(24, 12, 14, int(225 * tp_arc)),
                outline=(255, 78, 52, int(235 * tp_arc)),
                width=2,
            )
            d.text(
                (pill_x + 14, pill_y + 8),
                f"+{ang_val:04.1f}° 경추 과굴곡·측굴",
                font=fonts["pill"],
                fill=(255, 120, 95, int(255 * tp_arc)),
            )

        # Surgical Leader Callout pointing to glowing red C4-C7 zone
        if tp_label > 0.01:
            la = int(255 * tp_label)
            # Target point on C4-C6 nerve root compression zone: (485, 526)
            # Elbow at (415, 235), horizontal shelf to (445, 235)
            p_target = (485, 526)
            p_elbow = (415, 222)
            p_shelf = (225, 222)
            gd.line([p_target, p_elbow, p_shelf], fill=(255, 65, 35, int(110 * tp_label)), width=6)
            d.line([p_target, p_elbow, p_shelf], fill=(255, 110, 85, la), width=3)
            # Target ring on C4-C7 foramen
            d.ellipse(
                [p_target[0] - 18, p_target[1] - 18, p_target[0] + 18, p_target[1] + 18],
                outline=(255, 120, 90, la),
                width=2,
            )

            # Glass surgical HUD callout card at top-right of badge (x=205..1030, y=52..208)
            bx0, by0, bx1, by1 = 200, 48, 1032, 204
            d.rounded_rectangle(
                [bx0, by0, bx1, by1],
                radius=14,
                fill=(14, 16, 22, int(218 * tp_label)),
                outline=(255, 72, 48, int(210 * tp_label)),
                width=2,
            )
            # Left accent bar
            d.rounded_rectangle(
                [bx0, by0, bx0 + 8, by1],
                radius=4,
                fill=(255, 58, 38, la),
            )
            # Top clinical tag
            d.text(
                (bx0 + 24, by0 + 14),
                "PATHOLOGY · CERVICAL HYPERFLEXION (+18.4°)",
                font=fonts["tag"],
                fill=(255, 130, 105, la),
            )
            # Main Korean anatomical diagnosis
            d.text(
                (bx0 + 24, by0 + 44),
                "C4–C7 추간공 및 신경근 압박",
                font=fonts["title"],
                fill=(255, 255, 255, la),
            )
            # Secondary biomechanical metric
            d.text(
                (bx0 + 24, by0 + 102),
                "상부승모근·사각근 과긴장  |  추간판 하중 +240%",
                font=fonts["sub"],
                fill=(255, 195, 180, la),
            )

    # =========================================================================
    # 2. BOTTOM HALF (y = 960..1920): ✓ Cervical Contour Alignment (0.0° Axis)
    # =========================================================================
    # Anatomical vertebral landmarks in bottom half (spine is horizontal along y=1616):
    bot_nodes = [
        (195, 1616, "T4"),
        (305, 1616, "T1"),
        (395, 1616, "C7"),
        (485, 1616, "C5"),
        (575, 1616, "C2"),
        (685, 1616, "Skull"),
    ]

    if bot_prog > 0.01:
        bp_line = min(1.0, bot_prog / 0.55)
        bp_arc = max(0.0, min(1.0, (bot_prog - 0.25) / 0.45))
        bp_label = max(0.0, min(1.0, (bot_prog - 0.40) / 0.45))

        # Extended 0.0° neutral reference dashed line across entire spine
        draw_dashed_line(
            d,
            (150, 1616),
            (795, 1616),
            color=(160, 255, 210, int(190 * bp_line)),
            width=3,
            dash_len=12,
            gap_len=8,
        )

        # Spine axis o-o-o segments in glowing emerald green
        n_segs = len(bot_nodes) - 1
        seg_float = bp_line * n_segs
        for i in range(n_segs):
            seg_p = max(0.0, min(1.0, seg_float - i))
            if seg_p <= 0:
                continue
            x0, y0, _ = bot_nodes[i]
            x1, y1, _ = bot_nodes[i + 1]
            xe = x0 + (x1 - x0) * seg_p
            ye = y0 + (y1 - y0) * seg_p
            gd.line([(x0, y0), (xe, ye)], fill=(0, 235, 135, 155), width=14)
            d.line([(x0, y0), (xe, ye)], fill=(65, 255, 175, 245), width=5)

        for i, (nx, ny, v_tag) in enumerate(bot_nodes):
            if bp_line * n_segs < i - 0.1:
                continue
            r_n = 8
            gd.ellipse([nx - r_n - 6, ny - r_n - 6, nx + r_n + 6, ny + r_n + 6], fill=(0, 230, 130, 130))
            d.ellipse(
                [nx - r_n, ny - r_n, nx + r_n, ny + r_n],
                fill=(0, 235, 138, 255),
                outline=(255, 255, 255, 255),
                width=3,
            )
            if v_tag in ("T4", "C7", "C5", "C2"):
                d.text(
                    (nx - 14, ny + 14),
                    v_tag,
                    font=fonts["micro"],
                    fill=(195, 255, 225, int(225 * bp_line)),
                )

        # Decompression arrows & 0.0° Neutral Axis pill
        if bp_arc > 0.01:
            ba = int(245 * bp_arc)
            # Decompression vectors (axial distraction / ergonomic neck support arrows)
            draw_small_arrow(d, (455, 1580), (420, 1580), (65, 255, 175, ba), width=4, head_size=9)
            draw_small_arrow(d, (515, 1580), (550, 1580), (65, 255, 175, ba), width=4, head_size=9)
            # Ergonomic pillow contour support upward arrow under C4-C6
            draw_small_arrow(d, (505, 1682), (505, 1642), (65, 255, 175, ba), width=4, head_size=9)

            pill_x, pill_y = 610, 1532
            pill_w, pill_h = 338, 52
            d.rounded_rectangle(
                [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                radius=10,
                fill=(10, 24, 18, int(225 * bp_arc)),
                outline=(0, 230, 135, int(235 * bp_arc)),
                width=2,
            )
            d.text(
                (pill_x + 14, pill_y + 8),
                "0.0° 해부학적 중립 정렬 (0° Axis)",
                font=fonts["pill"],
                fill=(110, 255, 190, int(255 * bp_arc)),
            )

        # Surgical Leader Callout pointing to glowing emerald C1-C7 curve
        if bp_label > 0.01:
            la = int(255 * bp_label)
            p_target = (485, 1616)
            p_elbow = (415, 1184)
            p_shelf = (225, 1184)
            gd.line([p_target, p_elbow, p_shelf], fill=(0, 230, 135, int(110 * bp_label)), width=6)
            d.line([p_target, p_elbow, p_shelf], fill=(80, 255, 180, la), width=3)
            d.ellipse(
                [p_target[0] - 18, p_target[1] - 18, p_target[0] + 18, p_target[1] + 18],
                outline=(95, 255, 185, la),
                width=2,
            )

            bx0, by0, bx1, by1 = 200, 1008, 1032, 1164
            d.rounded_rectangle(
                [bx0, by0, bx1, by1],
                radius=14,
                fill=(12, 20, 18, int(218 * bp_label)),
                outline=(0, 228, 132, int(210 * bp_label)),
                width=2,
            )
            d.rounded_rectangle(
                [bx0, by0, bx0 + 8, by1],
                radius=4,
                fill=(0, 230, 132, la),
            )
            d.text(
                (bx0 + 24, by0 + 14),
                "BIOMECHANICS · 0.0° NEUTRAL CERVICAL AXIS",
                font=fonts["tag"],
                fill=(105, 255, 185, la),
            )
            d.text(
                (bx0 + 24, by0 + 44),
                "C1–C7 경추 전만(C-Curve) 유지 & 추간공 감압",
                font=fonts["title"],
                fill=(255, 255, 255, la),
            )
            d.text(
                (bx0 + 24, by0 + 102),
                "심부경추굴곡근 이완  ·  미주신경 안정 (하중 0% 기준축)",
                font=fonts["sub"],
                fill=(195, 248, 225, la),
            )

    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(8))
    img_pil = Image.alpha_composite(img_pil, glow_layer)
    img_pil = Image.alpha_composite(img_pil, overlay)

    # Draw clean badges at top-left of each half
    img_pil = draw_badge(img_pil, 110, 126, 54, "x", scale=min(1.0, top_prog * 2.2))
    img_pil = draw_badge(img_pil, 110, H // 2 + 126, 54, "check", scale=min(1.0, max(0.25, bot_prog * 2.0)))

    return img_pil


def generate_medical_audio(wav_path: str):
    """
    Generates an 8.0-second subtle medical HUD pulse + chime stereo/mono WAV track.
    - 0.0s - 3.5s: Low clinical tension pulse (72 BPM heartbeat/scanner hum + soft warning ping at 0.3s)
    - 3.5s: Crisp surgical lock-in chime (emerald transition)
    - 3.5s - 8.0s: Calm resonant harmonic pad + soft telemetry ticks
    """
    sr = 44100
    n_samples = int(sr * DURATION)
    t = np.linspace(0, DURATION, n_samples, endpoint=False)
    audio = np.zeros_like(t)

    # Subtle ambient sub-bass drone
    audio += 0.04 * np.sin(2 * np.pi * 65.0 * t)

    # 0.0s - 3.5s: Top half red compression pulse (at 0.25s, 1.15s, 2.05s, 2.95s)
    for pt in [0.25, 1.15, 2.05, 2.95]:
        mask = (t >= pt) & (t < pt + 0.45)
        dt = t[mask] - pt
        env = np.exp(-dt * 9.5) * np.sin(np.pi * dt / 0.45)
        # Dual-tone medical monitor pulse
        audio[mask] += 0.14 * env * (np.sin(2 * np.pi * 220.0 * dt) + 0.5 * np.sin(2 * np.pi * 440.0 * dt))

    # 3.5s: Emerald alignment lock-in chime (C5 -> G5 -> B5 warm crystalline chord)
    chime_t = 3.52
    mask_c = (t >= chime_t) & (t < DURATION)
    dtc = t[mask_c] - chime_t
    env_c = np.exp(-dtc * 2.2) * (1.0 - np.exp(-dtc * 45.0))
    audio[mask_c] += 0.16 * env_c * (
        0.50 * np.sin(2 * np.pi * 523.25 * dtc)
        + 0.35 * np.sin(2 * np.pi * 783.99 * dtc)
        + 0.25 * np.sin(2 * np.pi * 987.77 * dtc)
    )

    # Gentle telemetry ticks when nodes lock in
    for tick_t in [0.4, 0.55, 0.7, 0.85, 3.7, 3.85, 4.0, 4.15]:
        m_tk = (t >= tick_t) & (t < tick_t + 0.04)
        dtk = t[m_tk] - tick_t
        audio[m_tk] += 0.06 * np.exp(-dtk * 90.0) * np.sin(2 * np.pi * 1600.0 * dtk)

    # Smooth fade in/out
    fade_len = int(0.15 * sr)
    audio[:fade_len] *= np.linspace(0, 1, fade_len)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    audio_int16 = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fonts = {
        "tag": load_font(20, bold=True),
        "title": load_font(36, bold=True),
        "sub": load_font(25, bold=False),
        "pill": load_font(24, bold=True),
        "micro": load_font(19, bold=True),
    }

    print("[1/4] Preparing clean 50/50 glass anatomy base (inpainting stray green check)...")
    base_bgr = prepare_clean_50_50_base(RAW_PATH)
    top_glow, bot_glow = build_glow_layers(base_bgr)

    print("[2/4] Rendering static high-res 1080x1920 clinical comparison PNG...")
    static_pil = render_clinical_frame(
        base_bgr, top_glow, bot_glow, fonts, top_prog=1.0, bot_prog=1.0, top_pulse=0.35, bot_pulse=0.35
    )
    static_rgb = static_pil.convert("RGB")
    static_rgb.save(OUT_PNG, quality=96)
    static_rgb.save(C1_FINAL_JPG, quality=94)
    print(f"      Saved: {OUT_PNG}")

    print("[3/4] Synthesizing medical HUD audio track...")
    wav_path = "/tmp/clinical_glass_audio.wav"
    generate_medical_audio(wav_path)

    print(f"[4/4] Rendering {DURATION}s {FPS}fps MP4 reel ({TOTAL_FRAMES} frames)...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-i", wav_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        OUT_MP4,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Pre-render key static states or stream smoothly
    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / FPS
        if t < 3.5:
            # 0.0s - 3.5s: Focus/pulse on Top Half
            top_prog = min(1.0, t / 1.25)
            bot_prog = 0.15  # Subtle preview of bottom nodes, full lock-in after 3.5s
            top_pulse = 0.35 + 0.35 * math.sin(2 * math.pi * 1.35 * t) ** 2
            bot_pulse = 0.05
        else:
            # 3.5s - 8.0s: Focus/pulse on Bottom Half
            top_prog = 1.0
            bot_prog = min(1.0, (t - 3.5) / 1.25)
            top_pulse = 0.15
            bot_pulse = 0.35 + 0.35 * math.sin(2 * math.pi * 1.15 * (t - 3.5)) ** 2

        frame_pil = render_clinical_frame(
            base_bgr,
            top_glow,
            bot_glow,
            fonts,
            top_prog=top_prog,
            bot_prog=bot_prog,
            top_pulse=top_pulse,
            bot_pulse=bot_pulse,
        ).convert("RGB")
        proc.stdin.write(frame_pil.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"      Saved: {OUT_MP4}")


if __name__ == "__main__":
    main()
