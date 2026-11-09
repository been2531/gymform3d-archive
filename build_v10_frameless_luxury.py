#!/usr/bin/env python3
"""
@gymform.3d — v10 Frameless Luxury Graphic Template ("박스 0% · 순수 시네마틱 그래픽 HUD")
- Completely eliminates rectangular boxes and pill borders from both the top and bottom.
- Top (y = 248..330, x = 110..870):
  Frameless Editorial Masthead with a razor-thin laser timeline (STEP 01 ──●── 02 ──○── 03)
  and floating architectural typography ("CABLE ROPE PUSHDOWN") + frameless neon ✕/✓ emblem.
- Center-Lower (y = 1140..1355, x = 110..870):
  Frameless Holographic Projection Callout anchored directly to the active muscle/joint via a
  chamfered laser pointer line, featuring:
  1) Radial Biometric Arc ticks around the target joint + precision laser bracket
  2) Floating neon status overline ("✕ FORCE LEAK  //  ANTERIOR DELTOID") + sculpted 54px "68%" readout
  3) Glowing chamfered laser horizon line (no box!)
  4) Floating 56px Hero Cue ("SHOULDER TAKEOVER" / "DRIVE ELBOWS TO HIPS") in pure white over a feathered atmospheric shadow
  5) Precision millimeter tick spectrum meter underneath
- 100% inside the Instagram Reels Safe Zone (x: 96..884, y: 246..1374).
"""
import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import build_3scene_ox_preview as d1
import build_day2_ox_preview as d2
import build_v9_master_preview as v9

W, H = 1080, 1920
OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)

PRETENDARD_BLACK = "/home/box/.fonts/Pretendard-Black.otf"
PRETENDARD_EXTRABOLD = "/home/box/.fonts/Pretendard-ExtraBold.otf"
RAJDHANI_BOLD = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-Bold.ttf"
RAJDHANI_SEMIBOLD = "/usr/share/fonts/truetype/sand-box/google/Rajdhani/Rajdhani-SemiBold.ttf"
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(path: str, size: int):
    for p in [path, PRETENDARD_BLACK, PRETENDARD_EXTRABOLD, FALLBACK_BOLD]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TOP_STEP = load_font(RAJDHANI_BOLD, 24)
FONT_TOP_TITLE = load_font(PRETENDARD_BLACK, 32)

FONT_STATUS = load_font(PRETENDARD_EXTRABOLD, 25)
FONT_ANAT = load_font(RAJDHANI_BOLD, 25)
FONT_HERO = load_font(PRETENDARD_BLACK, 54)
FONT_PCT_BIG = load_font(RAJDHANI_BOLD, 64)
FONT_METRIC_SUB = load_font(PRETENDARD_EXTRABOLD, 25)
FONT_JOINT_TAG = load_font(RAJDHANI_BOLD, 22)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def draw_feathered_backdrop(canvas_rgba: Image.Image, box: tuple, alpha: int = 165, blur: int = 38):
    """
    Creates an invisible, edge-free atmospheric studio vignette behind floating typography
    so text has 100% contrast without ANY visible box or border.
    """
    vign = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    vd = ImageDraw.Draw(vign)
    x0, y0, x1, y1 = box
    vd.ellipse([x0, y0, x1, y1], fill=(4, 7, 12, alpha))
    vign = vign.filter(ImageFilter.GaussianBlur(radius=blur))
    canvas_rgba.alpha_composite(vign)


def draw_v10_frameless_top_masthead(
    canvas_rgba: Image.Image,
    exercise_name: str,
    active_idx: int,
    is_correct: bool,
):
    """
    Frameless Architectural Masthead (y = 248..336, x = 110..870):
    - ZERO boxes or borders!
    - Top row (y = 252): Precision laser step track:  01 ●─────── 02 ○─────── 03 ○
    - Main row (y = 286): Frameless glowing ✕ / ✓ neon emblem + floating "CABLE ROPE PUSHDOWN"
      with a razor-thin gradient underline that fades into space.
    """
    accent = (40, 250, 125) if is_correct else (255, 58, 48)
    x_left = 110
    x_right = 860

    # Soft invisible studio vignette behind masthead (no visible edge)
    draw_feathered_backdrop(canvas_rgba, (60, 220, 920, 365), alpha=150, blur=36)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # 1. Top Laser Step Track (y = 258)
    track_y = 258
    step_xs = [x_left + 72, x_left + 215, x_left + 358]
    # Connecting hairline between step nodes
    od.line([(step_xs[0], track_y), (step_xs[2], track_y)], fill=(255, 255, 255, 45), width=2)
    if active_idx > 0:
        od.line([(step_xs[0], track_y), (step_xs[active_idx], track_y)], fill=(*accent, 220), width=3)
        gd.line([(step_xs[0], track_y), (step_xs[active_idx], track_y)], fill=(*accent, 140), width=8)

    for i, sx in enumerate(step_xs):
        lbl = f"0{i+1}"
        if i == active_idx:
            # Glowing diamond/circle node
            gd.ellipse([sx - 12, track_y - 12, sx + 12, track_y + 12], fill=(*accent, 210))
            od.ellipse([sx - 7, track_y - 7, sx + 7, track_y + 7], fill=(255, 255, 255, 255), outline=(*accent, 255), width=3)
            v9.draw_tracked_text(od, (sx + 13, track_y - 13), lbl, FONT_TOP_STEP, fill=(*accent, 255), tracking=1.5, shadow=True)
        elif i < active_idx:
            od.ellipse([sx - 5, track_y - 5, sx + 5, track_y + 5], fill=(*accent, 220))
            v9.draw_tracked_text(od, (sx + 11, track_y - 13), lbl, FONT_TOP_STEP, fill=(200, 215, 232, 210), tracking=1.5, shadow=True)
        else:
            od.ellipse([sx - 5, track_y - 5, sx + 5, track_y + 5], fill=(80, 95, 115, 200))
            v9.draw_tracked_text(od, (sx + 11, track_y - 13), lbl, FONT_TOP_STEP, fill=(120, 135, 155, 185), tracking=1.5, shadow=True)

    # 2. Frameless 3D Neon ✕ / ✓ Emblem on the left (cx = x_left + 22, cy = 292)
    ic_cx, ic_cy = x_left + 22, 294
    # Outer precision reticle arc around the ✕/✓ emblem (no solid box!)
    gd.ellipse([ic_cx - 28, ic_cy - 28, ic_cx + 28, ic_cy + 28], fill=(*accent, 155))
    od.arc([ic_cx - 23, ic_cy - 23, ic_cx + 23, ic_cy + 23], start=35, end=325, fill=(*accent, 255), width=3)
    if is_correct:
        pts = [(ic_cx - 11, ic_cy + 1), (ic_cx - 2, ic_cy + 10), (ic_cx + 13, ic_cy - 8)]
        gd.line(pts, fill=(*accent, 255), width=10, joint="curve")
        od.line(pts, fill=(255, 255, 255, 255), width=5, joint="curve")
    else:
        d = 10
        gd.line([(ic_cx - d, ic_cy - d), (ic_cx + d, ic_cy + d)], fill=(*accent, 255), width=10)
        gd.line([(ic_cx - d, ic_cy + d), (ic_cx + d, ic_cy - d)], fill=(*accent, 255), width=10)
        od.line([(ic_cx - d, ic_cy - d), (ic_cx + d, ic_cy + d)], fill=(255, 255, 255, 255), width=5)
        od.line([(ic_cx - d, ic_cy + d), (ic_cx + d, ic_cy - d)], fill=(255, 255, 255, 255), width=5)

    # 3. Floating Architectural Exercise Title (y = 282)
    v9.draw_tracked_text(
        od,
        (x_left + 68, 280),
        exercise_name,
        FONT_TOP_TITLE,
        fill=(255, 255, 255, 255),
        tracking=2.4,
        shadow=True,
    )

    # 4. Fading Laser Horizon Hairline under the title (y = 326)
    line_y = 328
    for seg in range(40):
        t0 = seg / 40.0
        t1 = (seg + 1) / 40.0
        sx0 = int(lerp(x_left + 68, x_right - 60, t0))
        sx1 = int(lerp(x_left + 68, x_right - 60, t1))
        alpha = int(220 * (1.0 - t0 ** 1.3))
        col = accent if seg < 10 else (200, 220, 245)
        od.line([(sx0, line_y), (sx1, line_y)], fill=(*col, alpha), width=2)

    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)


def draw_v10_frameless_holographic_hud(
    canvas_rgba: Image.Image,
    target_pt: tuple,
    is_correct: bool,
    badge_text: str,
    anatomy_tag: str,
    title_text: str,
    metric_label: str,
    metric_pct: int,
    prog: float = 1.0,
):
    """
    v10 Frameless Holographic Projection HUD (y = 1145..1360, x = 104..876):
    - ZERO rectangular boxes!
    - Enhances the target joint with an outer Biometric Radial Telemetry Arc.
    - Connects the joint to a floating Chamfered Laser Horizon Bracket at y = 1218.
    - Above the laser horizon:
        Left: Neon status label ("✕ FORCE LEAK  /  ANTERIOR DELTOID")
        Right: Sculptural 64px Percentage Readout ("68%" or "98%") with glowing neon aura
    - Below the laser horizon:
        Floating 54px Hero Typography ("SHOULDER TAKEOVER" / "DRIVE ELBOWS TO HIPS")
    - Underneath (y = 1308):
        Frameless Precision Spectrum Ruler (fine vertical millimeter ticks + glowing laser beam)
    """
    if prog <= 0.05:
        return
    a = min(1.0, prog)
    accent = (40, 250, 125) if is_correct else (255, 58, 48)

    # Soft feathered atmospheric backdrop behind the floating graphic typography (ZERO box edges!)
    draw_feathered_backdrop(canvas_rgba, (45, 1105, 935, 1415), alpha=int(185 * a), blur=42)

    glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    tx, ty = int(target_pt[0]), int(target_pt[1])

    # 1. Outer Radial Biometric Telemetry Arc around the Target Joint
    arc_r = 76
    total_ticks = 28
    active_ticks = int(round((metric_pct / 100.0) * total_ticks * a))
    for i in range(total_ticks):
        ang_deg = -210 + i * (240.0 / (total_ticks - 1))
        ang_rad = math.radians(ang_deg)
        r_in = arc_r - 6 if i % 4 == 0 else arc_r - 3
        r_out = arc_r + 6 if i < active_ticks else arc_r + 2
        x_in = tx + int(r_in * math.cos(ang_rad))
        y_in = ty + int(r_in * math.sin(ang_rad))
        x_out = tx + int(r_out * math.cos(ang_rad))
        y_out = ty + int(r_out * math.sin(ang_rad))
        if i < active_ticks:
            gd.line([(x_in, y_in), (x_out, y_out)], fill=(*accent, int(190 * a)), width=6)
            od.line([(x_in, y_in), (x_out, y_out)], fill=(*accent, int(255 * a)), width=3)
        else:
            od.line([(x_in, y_in), (x_out, y_out)], fill=(255, 255, 255, int(65 * a)), width=2)

    # 2. Chamfered Laser Pointer from Joint Ring down to the Floating Laser Horizon (y = 1216)
    x_left = 108
    x_right = 872
    horizon_y = 1216

    anchor_x = max(x_left + 90, min(x_right - 160, tx + 20))
    elbow_x = int(lerp(tx, anchor_x, 0.55))
    elbow_y = int(lerp(ty + 52, horizon_y - 44, 0.55))
    start_y = min(ty + 54, horizon_y - 50)

    pointer_pts = [(tx, start_y), (elbow_x, elbow_y), (anchor_x, horizon_y - 44), (anchor_x, horizon_y)]
    od.line(pointer_pts, fill=(4, 8, 14, int(220 * a)), width=8, joint="curve")
    gd.line(pointer_pts, fill=(*accent, int(210 * a)), width=12, joint="curve")
    od.line(pointer_pts, fill=(*accent, int(255 * a)), width=4, joint="curve")
    od.line(pointer_pts, fill=(255, 255, 255, int(240 * a)), width=2, joint="curve")

    # Glowing diamond node at horizon intersection
    dm = 8
    diamond = [(anchor_x, horizon_y - dm), (anchor_x + dm, horizon_y), (anchor_x, horizon_y + dm), (anchor_x - dm, horizon_y)]
    gd.polygon(diamond, fill=(*accent, int(240 * a)))
    od.polygon(diamond, fill=(255, 255, 255, int(255 * a)), outline=(*accent, int(255 * a)))

    # 3. Left Vertical Architectural Accent Bar + Chamfered Laser Horizon Line (NO BOX!)
    # Sleek glowing vertical accent bar on the left (x = x_left, y = 1166..1342)
    bar_top, bar_bot = 1166, 1342
    gd.line([(x_left, bar_top), (x_left, bar_bot)], fill=(*accent, int(220 * a)), width=14)
    od.line([(x_left, bar_top), (x_left, bar_bot)], fill=(*accent, int(255 * a)), width=6)
    od.line([(x_left, bar_top), (x_left, bar_bot)], fill=(255, 255, 255, int(220 * a)), width=2)
    # Top and bottom 45-degree technical chamfer ticks on the left bar
    od.line([(x_left, bar_top), (x_left + 18, bar_top - 10)], fill=(*accent, int(255 * a)), width=4)
    od.line([(x_left, bar_bot), (x_left + 18, bar_bot + 10)], fill=(*accent, int(255 * a)), width=4)

    # Horizontal Laser Horizon Line separating Status Overline from Hero Cue
    gd.line([(x_left + 22, horizon_y), (x_right, horizon_y)], fill=(*accent, int(145 * a)), width=8)
    for seg in range(48):
        t0 = seg / 48.0
        t1 = (seg + 1) / 48.0
        sx0 = int(lerp(x_left + 22, x_right, t0))
        sx1 = int(lerp(x_left + 22, x_right, t1))
        fade = int(255 * (1.0 - 0.65 * (t0 ** 1.4)) * a)
        col = accent if seg < 28 else (210, 225, 245)
        od.line([(sx0, horizon_y), (sx1, horizon_y)], fill=(*col, fade), width=3 if seg < 20 else 2)

    text_x = x_left + 32

    # 4. ABOVE HORIZON (y = 1168): Frameless Status Tag + Anatomy Target + Sculpted Big % Readout
    clean_badge = badge_text.replace("✕ ", "").replace("✓ ", "").strip()
    prefix = "✕ " if not is_correct else "✓ "
    status_str = f"{prefix}{clean_badge}"
    v9.draw_tracked_text(
        od,
        (text_x, 1170),
        status_str,
        FONT_STATUS,
        fill=(*accent, int(255 * a)),
        tracking=1.6,
        shadow=True,
    )
    st_w = v9.measure_tracked_text(od, status_str, FONT_STATUS, tracking=1.6)

    sep_str = "  //  "
    v9.draw_tracked_text(
        od,
        (text_x + st_w, 1170),
        sep_str,
        FONT_ANAT,
        fill=(130, 150, 175, int(220 * a)),
        tracking=1.0,
        shadow=True,
    )
    sep_w = v9.measure_tracked_text(od, sep_str, FONT_ANAT, tracking=1.0)

    v9.draw_tracked_text(
        od,
        (text_x + st_w + sep_w, 1170),
        anatomy_tag,
        FONT_ANAT,
        fill=(215, 230, 248, int(245 * a)),
        tracking=1.8,
        shadow=True,
    )

    # Sculptural Big Percentage on the right (x aligned to x_right, spanning across the right end)
    pct_str = f"{metric_pct}%"
    pct_w = v9.measure_tracked_text(od, pct_str, FONT_PCT_BIG, tracking=1.0)
    pct_x = x_right - pct_w
    pct_y = 1142
    v9.draw_tracked_text(
        gd,
        (pct_x, pct_y),
        pct_str,
        FONT_PCT_BIG,
        fill=(*accent, int(200 * a)),
        tracking=1.0,
        shadow=False,
    )
    v9.draw_tracked_text(
        od,
        (pct_x, pct_y),
        pct_str,
        FONT_PCT_BIG,
        fill=(*accent, int(255 * a)),
        tracking=1.0,
        shadow=True,
    )

    # 5. BELOW HORIZON (y = 1230): Floating 54px Hero Typography (Pure Graphic, Zero Box!)
    v9.draw_tracked_text(
        gd,
        (text_x, 1230),
        title_text,
        FONT_HERO,
        fill=(*accent, int(95 * a)),
        tracking=1.4,
        shadow=False,
    )
    v9.draw_tracked_text(
        od,
        (text_x, 1230),
        title_text,
        FONT_HERO,
        fill=(255, 255, 255, int(255 * a)),
        tracking=1.4,
        shadow=True,
    )

    # 6. BOTTOM ROW (y = 1306): Frameless Precision Spectrum Tick Ruler + Metric Label
    v9.draw_tracked_text(
        od,
        (text_x, 1304),
        metric_label.upper(),
        FONT_METRIC_SUB,
        fill=(195, 212, 232, int(245 * a)),
        tracking=1.4,
        shadow=True,
    )
    lbl_w = v9.measure_tracked_text(od, metric_label.upper(), FONT_METRIC_SUB, tracking=1.4)

    # Precision vertical tick ruler from (text_x + lbl_w + 26) to x_right
    ruler_x0 = int(text_x + lbl_w + 28)
    ruler_x1 = x_right
    ruler_y = 1318
    n_ticks = 36
    active_ruler = int(round((metric_pct / 100.0) * n_ticks * a))
    for i in range(n_ticks):
        t = i / float(n_ticks - 1)
        rx = int(lerp(ruler_x0, ruler_x1, t))
        th = 11 if i % 5 == 0 else 6
        if i < active_ruler:
            gd.line([(rx, ruler_y - th), (rx, ruler_y + th)], fill=(*accent, int(195 * a)), width=5)
            od.line([(rx, ruler_y - th), (rx, ruler_y + th)], fill=(*accent, int(255 * a)), width=3)
        else:
            od.line([(rx, ruler_y - th // 2), (rx, ruler_y + th // 2)], fill=(255, 255, 255, int(55 * a)), width=2)

    # Glowing laser baseline through the active portion of the ruler
    act_x1 = int(lerp(ruler_x0, ruler_x1, (metric_pct / 100.0) * a))
    gd.line([(ruler_x0, ruler_y), (act_x1, ruler_y)], fill=(*accent, int(210 * a)), width=8)
    od.line([(ruler_x0, ruler_y), (act_x1, ruler_y)], fill=(255, 255, 255, int(245 * a)), width=2)

    glow = glow.filter(ImageFilter.GaussianBlur(radius=11))
    canvas_rgba.alpha_composite(glow)
    canvas_rgba.alpha_composite(ov)


def patch_modules_to_v10():
    def d1_v10_header(canvas_rgba, active_idx, is_correct=True):
        draw_v10_frameless_top_masthead(canvas_rgba, "CABLE ROPE PUSHDOWN", active_idx, is_correct)

    def d2_v10_header(canvas_rgba, active_idx, is_correct=True):
        draw_v10_frameless_top_masthead(canvas_rgba, "WIDE-GRIP LAT PULLDOWN", active_idx, is_correct)

    def v10_noop_badge(canvas_rgba, is_correct=True, pulse=0.5):
        pass

    def v10_scanner_callout(
        canvas_rgba,
        target_pt,
        card_center,
        is_correct,
        badge_text,
        anatomy_tag,
        title_text,
        metric_label,
        metric_pct,
        prog=1.0,
    ):
        draw_v10_frameless_holographic_hud(
            canvas_rgba=canvas_rgba,
            target_pt=target_pt,
            is_correct=is_correct,
            badge_text=badge_text,
            anatomy_tag=anatomy_tag,
            title_text=title_text,
            metric_label=metric_label,
            metric_pct=metric_pct,
            prog=prog,
        )

    d1.draw_top_header_and_step_bar = d1_v10_header
    d1.draw_ox_status_badge = v10_noop_badge
    d1.draw_anatomical_scanner_callout = v10_scanner_callout

    d2.draw_day2_header = d2_v10_header
    d2.draw_ox_status_badge = v10_noop_badge
    d2.draw_anatomical_scanner_callout = v10_scanner_callout


def main():
    patch_modules_to_v10()

    d1_wrong = [d1.build_scene_frame(i, is_correct=False) for i in range(3)]
    d1_correct = [d1.build_scene_frame(i, is_correct=True) for i in range(3)]

    from generate_v9_previews import build_sheet
    build_sheet(
        d1_wrong,
        d1_correct,
        ["STEP 1 · TORSO & SCAPULA", "STEP 2 · ELBOW HINGE", "STEP 3 · ROPE SPREAD"],
        os.path.join(OUT_DIR, "v10_day1_preview_sheet.jpg"),
    )

    d2_wrong = [d2.build_day2_scene_frame(i, is_correct=False) for i in range(3)]
    d2_correct = [d2.build_day2_scene_frame(i, is_correct=True) for i in range(3)]
    build_sheet(
        d2_wrong,
        d2_correct,
        ["STEP 1 · SCAPULA SET", "STEP 2 · TORSO & PATH", "STEP 3 · ELBOW DRIVE"],
        os.path.join(OUT_DIR, "v10_day2_preview_sheet.jpg"),
    )

    # Build 4-Panel Comparison: v10 Frameless Clean vs IG Reels UI Safe-Zone Check
    p1 = d1_wrong[0]
    p2 = v9.overlay_ig_reels_ui_simulator(d1_wrong[0])
    p3 = d2_correct[2]
    p4 = v9.overlay_ig_reels_ui_simulator(d2_correct[2])

    scale = 0.55
    pw, ph = int(W * scale), int(H * scale)
    pad = 24
    hdr = 68
    comp = Image.new("RGB", (pad * 5 + pw * 4, pad * 2 + ph + hdr), (8, 11, 16))
    cd = ImageDraw.Draw(comp)
    panels = [
        (p1, "1. DAY 1 (v10 FRAMELESS)", (255, 115, 105), (48, 16, 18), (255, 58, 48)),
        (p2, "2. DAY 1 + IG REELS UI CHECK", (100, 235, 255), (12, 36, 48), (0, 220, 255)),
        (p3, "3. DAY 2 (v10 FRAMELESS)", (95, 255, 160), (14, 44, 28), (40, 250, 125)),
        (p4, "4. DAY 2 + IG REELS UI CHECK", (100, 235, 255), (12, 36, 48), (0, 220, 255)),
    ]
    for idx, (frm, lbl, tcol, bgcol, bcol) in enumerate(panels):
        x0 = pad + idx * (pw + pad)
        cd.rounded_rectangle([x0, pad, x0 + pw, pad + hdr - 10], radius=12, fill=bgcol, outline=bcol, width=2)
        v9.draw_tracked_text(cd, (x0 + 16, pad + 13), lbl, v9.FONT_SHEET_TITLE, fill=tcol, tracking=0.8)
        im = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)).resize((pw, ph), Image.Resampling.LANCZOS)
        comp.paste(im, (x0, pad + hdr))

    comp_path = os.path.join(OUT_DIR, "v10_ig_safe_zone_comparison.jpg")
    comp.save(comp_path, quality=94)
    print(f"Saved: {comp_path}")


if __name__ == "__main__":
    main()
