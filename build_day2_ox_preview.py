#!/usr/bin/env python3
"""
@gymform.3d — Day 2: Wide-Grip Lat Pulldown (v8 Upgraded 3x2 O/X Preview Sheet)
Key Upgrades over Day 1 (v7):
1. 6 Dedicated Multi-Angle 3D Anatomical Plates (Step 1 Rear-3/4 Scapula, Step 2 Torso/Spine Angle, Step 3 Close-Up Elbow-to-Hip Drive).
2. Selective Compensatory vs Target Muscle Color Separation:
   - In ✕ Wrong panels, lower-back/lat glow is suppressed to matte grey while the compensatory muscle/joint
     (Step 1: Upper Traps, Step 2: Lumbar Spine & Shoulder Capsule, Step 3: Biceps & Forearms) glows warning crimson-red.
   - In ✓ Correct panels, upper-trap/bicep glow is suppressed while the fan-shaped Latissimus Dorsi & Teres Major blaze with orange-gold + neon-green rim light.
3. Luxury Top Header ("STEP 01/03 · SCAPULA INITIATION" + "WIDE-GRIP LAT PULLDOWN") + 1 Body Line + 1 Fitted 3D Arrow + High-Legibility 58px Scanner Card.
"""
import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from build_3scene_ox_preview import (
    W, H, OUT_DIR,
    FONT_HEADER_STEP, FONT_HEADER_TITLE, FONT_SHEET_HEADER,
    lerp, crop_camera_view,
    draw_tracked_text, measure_tracked_text,
    draw_single_posture_line, draw_lock_on_joint_ring,
    draw_bold_3d_arrow, draw_ox_status_badge,
    draw_anatomical_scanner_callout,
)


def load_plate(filename: str) -> np.ndarray:
    p = os.path.join(OUT_DIR, filename)
    im = cv2.imread(p)
    if im is None:
        raise FileNotFoundError(p)
    up = cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    return cv2.addWeighted(up, 1.18, blur, -0.18, 0)


def selective_muscle_grade(
    frame_bgr: np.ndarray,
    is_correct: bool,
    hot_zones: list,
    suppress_zones: list,
) -> np.ndarray:
    """
    Upgraded v8 Selective Muscle Shader:
    - Suppresses unwanted orange/red saturation in `suppress_zones` (turning inactive muscles back to sculpted matte grey)
    - Amplifies either Warning Crimson-Red (✕) or Target Fiery Orange-Gold (✓) inside `hot_zones`.
    """
    out = frame_bgr.astype(np.float32) * 1.04
    h, w, _ = out.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 1. Desaturate / cool down suppressed zones back to sculpted matte grey
    if suppress_zones:
        sup_mask = np.zeros((h, w), dtype=np.float32)
        for (cx, cy, rx, ry) in suppress_zones:
            g = np.exp(-((xx - cx) ** 2 / (2.0 * rx ** 2) + (yy - cy) ** 2 / (2.0 * ry ** 2)))
            sup_mask = np.maximum(sup_mask, g)
        b, gr, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        warm_excess = np.clip((r - b) / 140.0, 0.0, 1.0) * sup_mask
        lum = 0.22 * r + 0.58 * gr + 0.20 * b
        out[:, :, 2] = r * (1.0 - 0.85 * warm_excess) + (lum * 0.95) * (0.85 * warm_excess)
        out[:, :, 1] = gr * (1.0 - 0.70 * warm_excess) + (lum * 0.96) * (0.70 * warm_excess)
        out[:, :, 0] = b * (1.0 - 0.50 * warm_excess) + (lum * 1.02) * (0.50 * warm_excess)

    # 2. Amplify active target or compensatory overload zones
    if hot_zones:
        hot_mask = np.zeros((h, w), dtype=np.float32)
        for (cx, cy, rx, ry, strength) in hot_zones:
            g = np.exp(-((xx - cx) ** 2 / (2.0 * rx ** 2) + (yy - cy) ** 2 / (2.0 * ry ** 2))) * strength
            hot_mask = np.clip(hot_mask + g, 0.0, 1.15)
        if not is_correct:
            # Crimson-Red X-Ray Joint/Compensatory Overload
            out[:, :, 2] += hot_mask * 195.0
            out[:, :, 1] += hot_mask * 22.0
            out[:, :, 0] += hot_mask * 18.0
        else:
            # Target Latissimus Dorsi Fiery Orange-Gold Activation
            out[:, :, 2] += hot_mask * 185.0
            out[:, :, 1] += hot_mask * 92.0
            out[:, :, 0] += hot_mask * 14.0

    # 3. Top and Bottom Vignette for crystal HUD legibility
    vignette_y = np.clip((yy - h * 0.62) / (h * 0.38), 0.0, 1.0) ** 1.35
    out *= (1.0 - 0.54 * vignette_y[:, :, None])
    top_vig = np.clip((h * 0.16 - yy) / (h * 0.16), 0.0, 1.0) ** 1.4
    out *= (1.0 - 0.40 * top_vig[:, :, None])

    return np.clip(out, 0, 255).astype(np.uint8)


def draw_day2_header(canvas_rgba: Image.Image, active_idx: int, is_correct: bool):
    """Luxury Top Header customized for Day 2: WIDE-GRIP LAT PULLDOWN."""
    step_focus_names = [
        "SCAPULA INITIATION",
        "TORSO & BAR PATH",
        "ELBOW DRIVE ARC",
    ]
    focus_name = step_focus_names[max(0, min(2, active_idx))]
    accent = (40, 250, 125) if is_correct else (255, 58, 48)

    x0, y0 = 48, 56
    x1, y1 = 826, 192

    sh = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([x0 + 4, y0 + 8, x1 + 4, y1 + 10], radius=32, fill=(0, 0, 0, 185))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=14))
    canvas_rgba.alpha_composite(sh)

    ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    od.rounded_rectangle([x0, y0, x1, y1], radius=30, fill=(10, 14, 22, 234), outline=(255, 255, 255, 44), width=2)
    od.line([(x0 + 32, y0 + 2), (x1 - 32, y0 + 2)], fill=(255, 255, 255, 95), width=2)

    bar_x = x0 + 24
    od.rounded_rectangle([bar_x, y0 + 24, bar_x + 8, y1 - 24], radius=4, fill=(*accent, 255))

    text_x = x0 + 50
    step_part1 = f"STEP 0{active_idx + 1} / 03"
    draw_tracked_text(od, (text_x, y0 + 20), step_part1, FONT_HEADER_STEP, fill=(*accent, 255), tracking=1.8, shadow=True)
    w_p1 = measure_tracked_text(od, step_part1, FONT_HEADER_STEP, tracking=1.8)
    dot_cx = int(round(text_x + w_p1 + 18))
    dot_cy = y0 + 34
    od.ellipse([dot_cx - 4, dot_cy - 4, dot_cx + 4, dot_cy + 4], fill=(*accent, 255))
    draw_tracked_text(od, (dot_cx + 18, y0 + 20), focus_name, FONT_HEADER_STEP, fill=(215, 228, 245, 255), tracking=1.8, shadow=True)

    seg_w, seg_h, seg_gap = 46, 10, 8
    seg_start_x = x1 - 28 - (3 * seg_w + 2 * seg_gap)
    seg_y = y0 + 29
    for i in range(3):
        sx0 = seg_start_x + i * (seg_w + seg_gap)
        sx1 = sx0 + seg_w
        if i < active_idx:
            od.rounded_rectangle([sx0, seg_y, sx1, seg_y + seg_h], radius=5, fill=(210, 225, 245, 190))
        elif i == active_idx:
            od.rounded_rectangle([sx0 - 2, seg_y - 2, sx1 + 2, seg_y + seg_h + 2], radius=6, fill=(*accent, 90))
            od.rounded_rectangle([sx0, seg_y, sx1, seg_y + seg_h], radius=5, fill=(*accent, 255))
        else:
            od.rounded_rectangle([sx0, seg_y, sx1, seg_y + seg_h], radius=5, fill=(45, 56, 72, 210))

    draw_tracked_text(
        od,
        (text_x, y0 + 62),
        "WIDE-GRIP LAT PULLDOWN",
        FONT_HEADER_TITLE,
        fill=(255, 255, 255, 255),
        tracking=1.4,
        shadow=True,
    )

    canvas_rgba.alpha_composite(ov)




# --- v11.1 PURE SCULPTURAL GRAPHIC MASTER OVERRIDE ---
import build_v11_sculptural_graphic as _v11

def draw_day2_header(canvas_rgba, active_idx: int, is_correct: bool = True):
    _v11.draw_v11_cinema_top(canvas_rgba, "WIDE-GRIP LAT PULLDOWN", active_idx, is_correct)


def build_day2_scene_frame(scene_idx: int, is_correct: bool) -> np.ndarray:
    """
    Renders each of the 6 panels for Day 2: Wide-Grip Lat Pulldown (v8):
    - Scene 0: STEP 1/3 · SCAPULA INITIATION (Rear 3/4 Back View)
        ✕: SHRUGGING SHOULDERS (Upper Trap Load: 74%)
        ✓: DEPRESS SCAPULA FIRST (Lat Activation: 91%)
    - Scene 1: STEP 2/3 · TORSO & BAR PATH (Side-Rear 3/4 Spine & Shoulder View)
        ✕: EXCESSIVE BACK SWING (Lower Back Stress: 82%)
        ✓: 15° CHEST-UP LEAN (Upper Chest Path / Lat Tension: 95%)
    - Scene 2: STEP 3/3 · ELBOW DRIVE ARC (Close-Up Rear-Side 3/4 Upper-Body View)
        ✕: CURLING WITH BICEPS (Lat Isolation: 34%)
        ✓: DRIVE ELBOWS TO HIPS (Latissimus Dorsi EMG: 97%)
    """
    if scene_idx == 0:
        if not is_correct:
            raw = load_plate("day2_step1_wrong_plate.jpg")
            # Suppress lower lats, ignite Upper Trapezius & Neck base in Crimson Red
            graded = selective_muscle_grade(
                raw,
                is_correct=False,
                hot_zones=[(660, 775, 115, 65, 0.58), (585, 780, 75, 55, 0.45), (755, 785, 75, 55, 0.45)],
                suppress_zones=[(675, 960, 220, 130)],
            )
            framed_bgr, tf = crop_camera_view(graded, cx=640, cy=860, zoom=1.06)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            left_sh = tf((525, 800))
            neck_shrug = tf((665, 745))
            right_sh = tf((825, 805))
            target_trap = tf((665, 765))

            draw_single_posture_line(canvas, [left_sh, neck_shrug, right_sh], col_rgb=(255, 55, 45), width=13)
            draw_lock_on_joint_ring(canvas, target_trap, is_correct=False, r_outer=56)
            # Upward shrug arrow at upper traps
            draw_bold_3d_arrow(
                canvas,
                p0=tf((665, 900)),
                p1=tf((665, 790)),
                p2=tf((665, 650)),
                col_rgb=(255, 55, 45),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_trap,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ FORCE LEAK",
                anatomy_tag="UPPER TRAPEZIUS",
                title_text="SHRUGGING SHOULDERS",
                metric_label="Upper Trap Takeover",
                metric_pct=74,
            )
        else:
            raw = load_plate("day2_step1_correct_plate.jpg")
            # Suppress upper traps near neck, amplify V-taper Latissimus Dorsi & Lower Traps
            graded = selective_muscle_grade(
                raw,
                is_correct=True,
                hot_zones=[(575, 910, 95, 125, 0.42), (735, 910, 95, 125, 0.42), (655, 865, 85, 95, 0.35)],
                suppress_zones=[(655, 690, 130, 55)],
            )
            framed_bgr, tf = crop_camera_view(graded, cx=640, cy=860, zoom=1.06)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            left_sc = tf((545, 785))
            mid_sc = tf((655, 875))
            right_sc = tf((775, 785))
            target_sc = tf((655, 865))

            # Clean V-depression scapular lock line
            draw_single_posture_line(canvas, [left_sc, mid_sc, right_sc], col_rgb=(40, 250, 125), width=13)
            draw_lock_on_joint_ring(canvas, target_sc, is_correct=True, r_outer=56)
            # Downward scapular depression arrow
            draw_bold_3d_arrow(
                canvas,
                p0=tf((655, 705)),
                p1=tf((655, 835)),
                p2=tf((655, 985)),
                col_rgb=(40, 250, 125),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_sc,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ SCAPULA LOCKED",
                anatomy_tag="LOWER TRAPS & LATS",
                title_text="DEPRESS SCAPULA FIRST",
                metric_label="Latissimus Activation",
                metric_pct=91,
            )

    elif scene_idx == 1:
        if not is_correct:
            raw = load_plate("day2_step2_wrong_plate.jpg")
            graded = selective_muscle_grade(
                raw,
                is_correct=False,
                hot_zones=[(650, 1080, 85, 145, 0.45), (755, 800, 75, 75, 0.42)],
                suppress_zones=[],
            )
            framed_bgr, tf = crop_camera_view(graded, cx=650, cy=930, zoom=1.14)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            lumbar = tf((635, 1185))
            mid_spine = tf((655, 985))
            upper_spine = tf((695, 780))
            target_lumbar = tf((645, 1095))

            draw_single_posture_line(canvas, [lumbar, mid_spine, upper_spine], col_rgb=(255, 55, 45), width=13)
            draw_lock_on_joint_ring(canvas, target_lumbar, is_correct=False, r_outer=56)
            # Backward momentum swing arrow
            draw_bold_3d_arrow(
                canvas,
                p0=tf((560, 760)),
                p1=tf((695, 785)),
                p2=tf((835, 885)),
                col_rgb=(255, 55, 45),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_lumbar,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ MOMENTUM SWING",
                anatomy_tag="LUMBAR & ROTATOR CUFF",
                title_text="EXCESSIVE BACK SWING",
                metric_label="Lower Back Shear",
                metric_pct=82,
            )
        else:
            raw = load_plate("day2_step2_correct_plate.jpg")
            graded = selective_muscle_grade(
                raw,
                is_correct=True,
                hot_zones=[(715, 955, 110, 130, 0.48)],
                suppress_zones=[],
            )
            framed_bgr, tf = crop_camera_view(graded, cx=650, cy=910, zoom=1.14)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            hip = tf((655, 1180))
            thoracic = tf((675, 940))
            shoulder = tf((695, 745))
            target_lat = tf((715, 955))

            draw_single_posture_line(canvas, [hip, thoracic, shoulder], col_rgb=(40, 250, 125), width=13)
            draw_lock_on_joint_ring(canvas, target_lat, is_correct=True, r_outer=56)
            # Clean bar/elbow path arrow down toward upper chest
            draw_bold_3d_arrow(
                canvas,
                p0=tf((535, 620)),
                p1=tf((555, 765)),
                p2=tf((605, 905)),
                col_rgb=(40, 250, 125),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_lat,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ 15° THORACIC ARCH",
                anatomy_tag="UPPER CHEST BAR PATH",
                title_text="PULL TO UPPER CHEST",
                metric_label="Latissimus Tension",
                metric_pct=95,
            )

    else:
        if not is_correct:
            raw = load_plate("day2_step3_wrong_plate.jpg")
            # Suppress back glow, keep Biceps & Forearms blazing crimson red
            graded = selective_muscle_grade(
                raw,
                is_correct=False,
                hot_zones=[(265, 735, 110, 95, 0.48), (855, 765, 110, 95, 0.48)],
                suppress_zones=[(540, 920, 195, 160)],
            )
            framed_bgr, tf = crop_camera_view(graded, cx=560, cy=860, zoom=1.18)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            wrist_L = tf((175, 645))
            elbow_L = tf((245, 815))
            shoulder_L = tf((395, 785))
            target_bicep = tf((295, 775))

            draw_single_posture_line(canvas, [wrist_L, elbow_L, shoulder_L], col_rgb=(255, 55, 45), width=13)
            draw_lock_on_joint_ring(canvas, target_bicep, is_correct=False, r_outer=56)
            # Inward arm-curl arrow showing biceps takeover
            draw_bold_3d_arrow(
                canvas,
                p0=tf((180, 740)),
                p1=tf((255, 665)),
                p2=tf((375, 710)),
                col_rgb=(255, 55, 45),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_bicep,
                card_center=(540, 1590),
                is_correct=False,
                badge_text="✕ TENSION LOST",
                anatomy_tag="BICEPS & FOREARMS",
                title_text="CURLING WITH BICEPS",
                metric_label="Latissimus Isolation",
                metric_pct=34,
            )
        else:
            raw = load_plate("day2_step3_correct_plate.jpg")
            # Amplify full fan-shaped Latissimus Dorsi & Teres Major
            graded = selective_muscle_grade(
                raw,
                is_correct=True,
                hot_zones=[(725, 975, 125, 145, 0.55), (615, 965, 95, 130, 0.38)],
                suppress_zones=[(445, 745, 90, 90)],
            )
            framed_bgr, tf = crop_camera_view(graded, cx=650, cy=900, zoom=1.18)
            canvas = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

            hand_R = tf((915, 635))
            elbow_R = tf((885, 855))
            target_lat = tf((735, 965))

            # Vertical forearm stack + elbow-to-hip sweep arrow
            draw_single_posture_line(canvas, [hand_R, elbow_R], col_rgb=(40, 250, 125), width=13)
            draw_lock_on_joint_ring(canvas, target_lat, is_correct=True, r_outer=58)
            draw_bold_3d_arrow(
                canvas,
                p0=tf((895, 825)),
                p1=tf((845, 965)),
                p2=tf((755, 1095)),
                col_rgb=(40, 250, 125),
            )
            draw_bold_3d_arrow(
                canvas,
                p0=tf((455, 805)),
                p1=tf((505, 945)),
                p2=tf((595, 1075)),
                col_rgb=(40, 250, 125),
            )
            draw_anatomical_scanner_callout(
                canvas,
                target_pt=target_lat,
                card_center=(540, 1590),
                is_correct=True,
                badge_text="✓ PEAK CONTRACTION",
                anatomy_tag="LATISSIMUS DORSI",
                title_text="DRIVE ELBOWS TO HIPS",
                metric_label="Latissimus Dorsi EMG",
                metric_pct=97,
            )

    draw_day2_header(canvas, active_idx=scene_idx, is_correct=is_correct)
    draw_ox_status_badge(canvas, is_correct=is_correct)
    return cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    scenes_wrong = []
    scenes_correct = []
    for s_idx in range(3):
        w_bgr = build_day2_scene_frame(s_idx, is_correct=False)
        c_bgr = build_day2_scene_frame(s_idx, is_correct=True)
        cv2.imwrite(os.path.join(OUT_DIR, f"day2_scene{s_idx+1}_wrong.jpg"), w_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        cv2.imwrite(os.path.join(OUT_DIR, f"day2_scene{s_idx+1}_correct.jpg"), c_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        scenes_wrong.append(w_bgr)
        scenes_correct.append(c_bgr)

    pw, ph = 540, 960
    pad = 24
    header_h = 86
    sheet_w = pw * 3 + pad * 4
    sheet_h = ph * 2 + pad * 3 + header_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 13, 18))
    sd = ImageDraw.Draw(sheet)

    col_titles = [
        "STEP 1/3  |  SCAPULA INITIATION",
        "STEP 2/3  |  TORSO & BAR PATH",
        "STEP 3/3  |  ELBOW DRIVE ARC",
    ]
    for c_idx, title in enumerate(col_titles):
        cx = pad + c_idx * (pw + pad) + pw // 2
        tb = sd.textbbox((0, 0), title, font=FONT_SHEET_HEADER)
        tw = tb[2] - tb[0]
        sd.text((cx - tw // 2, 26), title, font=FONT_SHEET_HEADER, fill=(235, 242, 252))

    for c_idx in range(3):
        x = pad + c_idx * (pw + pad)
        y_top = header_h
        w_rgb = cv2.cvtColor(cv2.resize(scenes_wrong[c_idx], (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(w_rgb), (x, y_top))
        sd.rounded_rectangle([x - 2, y_top - 2, x + pw + 2, y_top + ph + 2], radius=14, outline=(255, 60, 50), width=3)

        y_bot = header_h + ph + pad
        c_rgb = cv2.cvtColor(cv2.resize(scenes_correct[c_idx], (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(c_rgb), (x, y_bot))
        sd.rounded_rectangle([x - 2, y_bot - 2, x + pw + 2, y_bot + ph + 2], radius=14, outline=(40, 250, 125), width=3)

    sheet_path = os.path.join(OUT_DIR, "day2_3scene_ox_preview_sheet.jpg")
    sheet.save(sheet_path, quality=96)
    print("Saved Day 2 v8 preview sheet:", sheet_path, sheet.size)


if __name__ == "__main__":
    main()
