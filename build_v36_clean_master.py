#!/usr/bin/env python3
"""
v36 Pure Studio Master — Zero Dashed Lines · Pure 3D Fiber-Strand + Exterior Arcs & Numbers
===========================================================================================
Refinement from user (`t157u`):
- "점선 정렬축 깔끔하지 못한거같아. 제외할까? ㅋㅋ"
  (The dashed alignment axes don't look clean. Shall we remove them?)

Key Upgrades in v36:
1. 100% Removal of All Dashed Posture/Alignment Axis Lines & Scattered Keypoint Dots:
   - Zero dashed lines along the neck, spine/waist, upper arm, forearm, or thigh/shin.
   - Zero scattered keypoint circles along the neck, spine, hip, knee, wrist, or ankle.
   - The 3D mannequin's body, neck, waist, and legs remain 100% unobstructed and ultra-clean!
2. Pure 3D Muscle Fiber-Strand Coloring (`Intrinsic 3D Fiber-Strand`) + Step 1 Native 3D X-Ray:
   - Step 1 (`✕`): Native 3D X-Ray shoulder/humerus/forearm glow inside the arm + exterior dorsal `35° BACK HUNCH` arc + exterior `28° NECK CRANE` & `28° WRIST BEND` micro-callouts outside the body.
   - Step 2 (`✕`): Pure 3D fiber-strand crimson coloring on the Anterior Deltoid (`DELT STEAL`) + single clean `◎` elbow pivot ring + exterior `90° STOP SHORT` forearm arc outside the arm.
   - Step 3 (`✓`): Pure 3D fiber-strand emerald coloring on the Lateral & Long Head Triceps + single clean `◎` elbow lock ring + exterior `135°` extension arc + exterior `15° HIP HINGE` micro-badge outside the back contour.
   - Step 4 (`✓`): Pure 3D fiber-strand emerald coloring on All 3 Triceps Heads + single clean `◎` elbow lock ring + exterior `180° LOCKOUT` arc & `25° ROPE SPREAD` pronation arrows at the rope ends.
3. Frameless Bottom-Left Number & 20-Segment LED Gauge + Top 3-Thumbnail Capsule + Top-Right `✕/✓` Badge (`v30-A` HUD).
"""

import os
import sys
import math
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v25_fast_preview as v25
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import build_day2_ox_preview as d2
import universal_anatomy_rig as uar
import build_v35_repeatable_master as v35

OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)
H, W = eng.H, eng.W


def draw_single_elbow_pivot_ring(
    rd: ImageDraw.ImageDraw,
    center_ss: tuple[float, float],
    r_outer_px: float,
    col_rgb: tuple[int, int, int],
    SS: int = 2,
):
    """Draws a single subtle, razor-sharp elbow pivot reticle (`◎`) without any dashed lines."""
    cx, cy = center_ss
    ro = r_outer_px * SS
    ri = max(2.2 * SS, ro * 0.34)
    rd.ellipse([cx - ro - 2, cy - ro - 2, cx + ro + 2, cy + ro + 2], outline=(0, 0, 0, 175), width=2 * SS)
    rd.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], outline=(*col_rgb, 220), width=2 * SS)
    rd.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=(255, 255, 255, 245))


def render_v36_clean_exterior_overlay(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
    kpts: dict,
    is_wrong: bool,
    dynamic_elbow_deg: int = None,
):
    """
    Renders the v36 Ultra-Clean Studio Overlay:
    - ZERO dashed lines on the body!
    - ZERO scattered keypoint dots on the neck, spine, waist, hip, knee, or ankle!
    - Uses ONLY clean exterior arcs & sculpted numbers outside the silhouette + a single clean `◎` elbow pivot ring on Steps 2-4.
    """
    SS = 2
    rig_layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig_layer)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    def S(pt):
        return (pt[0] * SS, pt[1] * SS)

    def M_SS(pt):
        mp = map_pt(pt[0], pt[1])
        return (mp[0] * SS, mp[1] * SS)

    col_main = v30.COL_RED if is_wrong else v30.COL_GREEN
    col_soft = v30.COL_RED_SOFT if is_wrong else v30.COL_GREEN_SOFT
    el_s = M_SS(kpts["el"])
    wr_s = M_SS(kpts["wr"])

    if dynamic_elbow_deg is None:
        # Static 4-Step Master: use v30's clean exterior arcs outside the body silhouette (with solid exterior reference only)
        if step_idx == 1:
            # 1. Exterior Dorsal Back Curve outside hunched back (`x=915..955`)
            b_bot = S(map_pt(935, 1045))
            arc_r = 260 * SS
            rd.arc([b_bot[0] - arc_r, b_bot[1] - arc_r, b_bot[0] + arc_r, b_bot[1] + arc_r], start=235, end=270, fill=(*v30.COL_RED, 245), width=4 * SS)
            ang_t = math.radians(235)
            tip_b = (b_bot[0] + arc_r * math.cos(ang_t), b_bot[1] + arc_r * math.sin(ang_t))
            v30.draw_arrow_head(rd, tip_b, 235 - 90, 12 * SS, (*v30.COL_RED_SOFT, 255))

            n1_xy = map_pt(960, 690)
            v30.draw_floating_number_badge(od, sd, (min(965, int(n1_xy[0])), int(n1_xy[1])), "35°", "BACK HUNCH", v30.COL_RED, v30.COL_RED_SOFT, align="left")

            # Exterior Neck Crane micro-callout outside the head/neck
            p_nk = map_pt(715, 485)
            v30.draw_floating_number_badge(od, sd, (int(p_nk[0]), int(p_nk[1])), "28°", "NECK CRANE", v30.COL_RED, v30.COL_RED_SOFT, align="left")

            # 2. Exterior Front Wrist Flare Arc outside the hands (`x=440..540`)
            w_c = S(map_pt(560, 995))
            r_w = 78 * SS
            rd.arc([w_c[0] - r_w, w_c[1] - r_w, w_c[0] + r_w, w_c[1] + r_w], start=145, end=225, fill=(*v30.COL_RED, 245), width=4 * SS)
            ang_w = math.radians(145)
            tip_w = (w_c[0] + r_w * math.cos(ang_w), w_c[1] + r_w * math.sin(ang_w))
            v30.draw_arrow_head(rd, tip_w, 145 - 90, 11 * SS, (*v30.COL_RED_SOFT, 255))

            n2_xy = map_pt(445, 915)
            v30.draw_floating_number_badge(od, sd, (int(n2_xy[0]), int(n2_xy[1])), "28°", "WRIST BEND", v30.COL_RED, v30.COL_RED_SOFT, align="right")

        elif step_idx == 2:
            # Single clean `◎` elbow pivot ring + exterior 90° stop bar & arc in open left space (`x=410..560`)
            draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

            h_left = S(map_pt(425, 882))
            h_right = S(map_pt(535, 882))
            rd.line([h_left, h_right], fill=(*v30.COL_RED, 245), width=4 * SS)
            rd.line([(h_left[0], h_left[1] - 12 * SS), (h_left[0], h_left[1] + 12 * SS)], fill=(*v30.COL_RED_SOFT, 255), width=3 * SS)

            arc_c = S(map_pt(645, 882))
            r_m = 155 * SS
            rd.arc([arc_c[0] - r_m, arc_c[1] - r_m, arc_c[0] + r_m, arc_c[1] + r_m], start=132, end=180, fill=(*v30.COL_RED, 245), width=4 * SS)

            n1_xy = map_pt(445, 815)
            v30.draw_floating_number_badge(od, sd, (int(n1_xy[0]), int(n1_xy[1])), "90°", "STOP SHORT", v30.COL_RED, v30.COL_RED_SOFT, align="right")

            n2_xy = map_pt(495, 615)
            v30.draw_floating_number_badge(od, sd, (int(n2_xy[0]), int(n2_xy[1])), "72%", "DELT STEAL", v30.COL_RED, v30.COL_RED_SOFT, align="right")

        elif step_idx == 3:
            # Single clean `◎` elbow lock ring + exterior 15° Hip Hinge arc behind back + exterior 135° extension arc in front
            draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

            b_bot = S(map_pt(865, 1045))
            arc_r = 250 * SS
            rd.arc([b_bot[0] - arc_r, b_bot[1] - arc_r, b_bot[0] + arc_r, b_bot[1] + arc_r], start=255, end=270, fill=(*v30.COL_GREEN, 245), width=4 * SS)

            n_back = map_pt(905, 720)
            v30.draw_floating_number_badge(od, sd, (min(955, int(n_back[0])), int(n_back[1])), "15°", "HIP HINGE", v30.COL_GREEN, v30.COL_GREEN_SOFT, align="left")

            arc_c = S(map_pt(655, 885))
            r_f = 155 * SS
            rd.arc([arc_c[0] - r_f, arc_c[1] - r_f, arc_c[0] + r_f, arc_c[1] + r_f], start=128, end=174, fill=(*v30.COL_GREEN, 245), width=4 * SS)
            ang_tip = math.radians(128)
            tip_f = (arc_c[0] + r_f * math.cos(ang_tip), arc_c[1] + r_f * math.sin(ang_tip))
            v30.draw_arrow_head(rd, tip_f, 128 - 90, 12 * SS, (*v30.COL_GREEN_SOFT, 255))

            n_front = map_pt(465, 865)
            v30.draw_floating_number_badge(od, sd, (int(n_front[0]), int(n_front[1])), "135°", "ELBOW FIXED", v30.COL_GREEN, v30.COL_GREEN_SOFT, align="right")

        else:
            # Step 4: Single clean `◎` elbow lock ring + exterior 180° Full Lockout Arc + 25° Outward Rope Spread Arrows
            draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

            arc_c = S(map_pt(665, 885))
            r_f = 175 * SS
            rd.arc([arc_c[0] - r_f, arc_c[1] - r_f, arc_c[0] + r_f, arc_c[1] + r_f], start=112, end=168, fill=(*v30.COL_GREEN, 245), width=4 * SS)
            ang_tip = math.radians(112)
            tip_f = (arc_c[0] + r_f * math.cos(ang_tip), arc_c[1] + r_f * math.sin(ang_tip))
            v30.draw_arrow_head(rd, tip_f, 112 - 90, 13 * SS, (*v30.COL_GREEN_SOFT, 255))

            n_lock = map_pt(455, 845)
            v30.draw_floating_number_badge(od, sd, (int(n_lock[0]), int(n_lock[1])), "180°", "LOCKOUT", v30.COL_GREEN, v30.COL_GREEN_SOFT, align="right")

            w_c = S(map_pt(640, 1115))
            r_sp = 54 * SS
            rd.arc([w_c[0] - r_sp, w_c[1] - r_sp, w_c[0] + r_sp, w_c[1] + r_sp], start=130, end=215, fill=(*v30.COL_GREEN_SOFT, 250), width=4 * SS)
            ang_l = math.radians(215)
            tip_l = (w_c[0] + r_sp * math.cos(ang_l), w_c[1] + r_sp * math.sin(ang_l))
            v30.draw_arrow_head(rd, tip_l, 215 + 90, 11 * SS, (*v30.COL_GREEN_SOFT, 255))

            rd.arc([w_c[0] - r_sp, w_c[1] - r_sp, w_c[0] + r_sp, w_c[1] + r_sp], start=325, end=40, fill=(*v30.COL_GREEN_SOFT, 250), width=4 * SS)
            ang_r = math.radians(325)
            tip_r = (w_c[0] + r_sp * math.cos(ang_r), w_c[1] + r_sp * math.sin(ang_r))
            v30.draw_arrow_head(rd, tip_r, 325 - 90, 11 * SS, (*v30.COL_GREEN_SOFT, 255))

            n_sp = map_pt(525, 1085)
            v30.draw_floating_number_badge(od, sd, (int(n_sp[0]), int(n_sp[1])), "25°", "ROPE SPREAD", v30.COL_GREEN, v30.COL_GREEN_SOFT, align="right")

    else:
        # Dynamic 30fps video mode (ZERO dashed lines — only exterior arc & single elbow pivot ring)
        if is_wrong:
            b_bot = M_SS((935, 1045))
            arc_r = 250 * SS
            rd.arc([b_bot[0] - arc_r, b_bot[1] - arc_r, b_bot[0] + arc_r, b_bot[1] + arc_r], start=235, end=270, fill=(*col_main, 240), width=4 * SS)
            p_b = map_pt(950, 700)
            v30.draw_floating_number_badge(od, sd, (min(960, int(p_b[0])), int(p_b[1])), "35°", "BACK HUNCH", col_main, col_soft, align="left")
            p_nk = map_pt(715, 485)
            v30.draw_floating_number_badge(od, sd, (int(p_nk[0]), int(p_nk[1])), "28°", "NECK CRANE", col_main, col_soft, align="left")
        else:
            draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)
            r_f = 152 * SS
            ang_wr = math.degrees(math.atan2(wr_s[1] - el_s[1], wr_s[0] - el_s[0]))
            start_a = min(ang_wr, 172.0)
            end_a = max(ang_wr, 172.0)
            rd.arc([el_s[0] - r_f, el_s[1] - r_f, el_s[0] + r_f, el_s[1] + r_f], start=start_a, end=end_a, fill=(*col_main, 240), width=4 * SS)
            p_el = map_pt(465, 855)
            tag_el = "LOCKOUT" if dynamic_elbow_deg >= 170 else ("EXTENSION" if dynamic_elbow_deg >= 120 else "90° START")
            v30.draw_floating_number_badge(od, sd, (int(p_el[0]), int(p_el[1])), f"{dynamic_elbow_deg}°", tag_el, col_main, col_soft, align="right")
            p_back = map_pt(895, 720)
            v30.draw_floating_number_badge(od, sd, (min(955, int(p_back[0])), int(p_back[1])), "15°", "HIP HINGE", col_main, col_soft, align="left")

    canvas.alpha_composite(rig_layer.resize((W, H), Image.Resampling.LANCZOS))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)


def render_day2_v36_clean_panel(plate_fn: str, scene_type: str, is_wrong: bool) -> Image.Image:
    """
    Day 2 Wide-Grip Lat Pulldown rendered in the v36 Pure Studio Standard:
    - Intrinsic 3D Fiber-Strand Coloring on the back musculature.
    - ZERO dashed lines or keypoint dots crossing the back or arms!
    - Clean exterior posture/elbow sweep arc & sculpted number badge + frameless bottom-left telemetry.
    """
    bgr = d2.load_plate(plate_fn)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    gb_diff = np.maximum(0.0, g_ch - b_ch)

    fiber_signal = 0.45 * r_ch + 0.55 * g_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 1.0)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.8)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 7.5)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 16.0)

    hf_ridge = (sigma_micro - sigma_ridge) * 3.0 + (sigma_ridge - sigma_belly) * 1.7
    fs = np.clip(0.48 + hf_ridge / 42.0, 0.0, 1.0) ** 1.35
    hv = np.clip((sigma_belly - 28.0) / 165.0, 0.0, 1.0)
    sulc = np.clip((sigma_broad - sigma_ridge - 3.5) / 22.0, 0.0, 1.0)

    red_any = np.clip((rb_diff - 12.0) / 85.0, 0.0, 1.0)
    grey_sculpt = np.clip(34.0 + 148.0 * (hv ** 1.05) + hf_ridge * 0.95, 16.0, 235.0)
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    neutral_lum = np.clip(base_lum * (1.0 - red_any) + grey_sculpt * red_any, 14.0, 240.0)

    norm = neutral_lum / 255.0
    mono = np.stack([
        eng.BG_CHARCOAL[0] + (234 - eng.BG_CHARCOAL[0]) * (norm ** 0.92),
        eng.BG_CHARCOAL[1] + (238 - eng.BG_CHARCOAL[1]) * (norm ** 0.92),
        eng.BG_CHARCOAL[2] + (244 - eng.BG_CHARCOAL[2]) * (norm ** 0.92),
    ], axis=2)

    rig = uar.solve_universal_anatomical_rig(rgb.astype(np.uint8), "rear_back")
    body_mask = rig["body_mask"]

    if scene_type == "step1_correct":
        z_l = np.exp(-(((xx - 565.0) / 115.0) ** 2 + ((yy - 885.0) / 145.0) ** 2))
        z_r = np.exp(-(((xx - 745.0) / 115.0) ** 2 + ((yy - 885.0) / 145.0) ** 2))
        zone = np.clip(z_l + z_r, 0.0, 1.0) * np.clip((yy - 715.0) / 45.0, 0.0, 1.0)
    else:
        zone = np.exp(-(((xx - 635.0) / 135.0) ** 2 + ((yy - 835.0) / 110.0) ** 2))

    fe = np.clip((rb_diff - 32.0) / 125.0, 0.0, 1.0) * zone * body_mask
    hc = np.clip((gb_diff - 12.0) / 95.0, 0.0, 1.0) * zone * body_mask

    strand_mod = (0.22 + 0.78 * fs) * (1.0 - 0.65 * sulc)
    fiber_alpha = np.clip(fe * (0.62 + 0.66 * strand_mod), 0.0, 0.95)[:, :, None]
    ridge_boost = np.clip(fs * 0.65 + hc * 0.55, 0.0, 1.0)[:, :, None]
    vol_shade = np.clip(0.28 + 0.72 * hv * (0.45 + 0.55 * fs), 0.0, 1.0)[:, :, None]

    if is_wrong:
        deep_col = np.array([155.0, 18.0, 16.0], dtype=np.float32)
        mid_col = np.array([248.0, 52.0, 34.0], dtype=np.float32)
        crest_col = np.array([255.0, 195.0, 110.0], dtype=np.float32)
    else:
        deep_col = np.array([8.0, 110.0, 48.0], dtype=np.float32)
        mid_col = np.array([22.0, 224.0, 98.0], dtype=np.float32)
        crest_col = np.array([185.0, 255.0, 205.0], dtype=np.float32)

    fiber_rgb = (
        deep_col * (1.0 - vol_shade)
        + mid_col * vol_shade * (1.0 - ridge_boost * 0.65)
        + crest_col * (ridge_boost ** 1.35) * 0.85
    )
    out = np.clip(mono * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha, 0, 255).astype(np.uint8)
    im = Image.fromarray(out).convert("RGBA")

    # Clean exterior arc & single elbow pivot ring (ZERO dashed lines!)
    SS = 2
    rig_layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig_layer)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    col_main = v30.COL_RED if is_wrong else v30.COL_GREEN
    col_soft = v30.COL_RED_SOFT if is_wrong else v30.COL_GREEN_SOFT

    p_el_L = (rig["el_L"][0] * SS, rig["el_L"][1] * SS)
    draw_single_elbow_pivot_ring(rd, p_el_L, 9.5, col_main, SS)

    if is_wrong:
        c_arc = (450 * SS, 920 * SS)
        r_a = 155 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=195, end=248, fill=(*col_main, 240), width=4 * SS)
        v30.draw_floating_number_badge(od, sd, (395, 760), "42°", "BACK SWING", col_main, col_soft, align="right")
        sp_cfg = {
            "is_wrong": True,
            "pct": 78,
            "dial_tag": "LUMBAR RISK",
            "short_cue": "✕ 42° BACK SWING",
            "splits": [("LOW BACK", 78, v30.COL_RED), ("LATS", 24, v30.COL_MUTED)],
            "rx_num": "42°",
            "rx_unit": "SWING",
        }
    else:
        c_arc = (430 * SS, 825 * SS)
        r_a = 140 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=118, end=168, fill=(*col_main, 240), width=4 * SS)
        ang_tip = math.radians(118)
        tip_p = (c_arc[0] + r_a * math.cos(ang_tip), c_arc[1] + r_a * math.sin(ang_tip))
        v30.draw_arrow_head(rd, tip_p, 118 - 90, 12 * SS, (*col_soft, 255))
        v30.draw_floating_number_badge(od, sd, (395, 760), "15°", "SCAPULA LOCK", col_main, col_soft, align="right")
        sp_cfg = {
            "is_wrong": False,
            "pct": 96,
            "dial_tag": "LAT ISOLATION",
            "short_cue": "✓ SCAPULA DEPRESS",
            "splits": [("LAT DORSI", 96, v30.COL_GREEN), ("TERES MAJ", 92, v30.COL_GREEN)],
            "rx_num": "4×10",
            "rx_unit": "REPS · 1s",
        }

    im.alpha_composite(rig_layer.resize((W, H), Image.Resampling.LANCZOS))
    im.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    im.alpha_composite(ov)
    v30.draw_v30_graphic_telemetry_bottom_left(im, sp_cfg, mode="A")
    return im.convert("RGB")


def build_v36_dynamic_preview(torso_icons):
    """Generates `day1_v36_dynamic_preview.mp4` (5.0s, 150 frames @ 30fps) with ZERO dashed lines."""
    cap_w = cv2.VideoCapture(eng.WRONG_I2V)
    raw_w_frames = []
    idx = 0
    while True:
        ret, fr = cap_w.read()
        if not ret:
            break
        if 15 <= idx < 75:
            raw_w_frames.append(fr)
        idx += 1
    cap_w.release()

    cap_c = cv2.VideoCapture(eng.CORRECT_I2V)
    raw_c_frames = []
    idx = 0
    while True:
        ret, fr = cap_c.read()
        if not ret:
            break
        if 5 <= idx < 95:
            raw_c_frames.append(fr)
        idx += 1
    cap_c.release()

    keys_order = ["ear", "sh", "sh_far", "el", "wr", "hd", "hip", "hip_far", "knee", "ank"]

    def process_seq(bgr_list, is_wrong_seq):
        base_rgbs = []
        raw_kpts = []
        n = len(bgr_list)
        for i, bgr in enumerate(bgr_list):
            if is_wrong_seq:
                rgb_out, kp = v35.render_v35_clean_3d_frame(bgr, 1, "xray", True)
            else:
                prog = float(np.clip(1.0 - abs(i - 58.0) / 58.0, 0.15, 1.0))
                rgb_out, kp = v35.render_v35_clean_3d_frame(bgr, 4, "dynamic_blend", False, fiber_intensity=prog)
            base_rgbs.append(rgb_out)
            raw_kpts.append(kp)

        smoothed_kpts = [dict() for _ in range(n)]
        for k in keys_order:
            arr = np.array([rk[k] for rk in raw_kpts], dtype=np.float32)
            arr_x = gaussian_filter1d(arr[:, 0], sigma=1.8, mode="nearest")
            arr_y = gaussian_filter1d(arr[:, 1], sigma=1.8, mode="nearest")
            for i in range(n):
                smoothed_kpts[i][k] = (float(arr_x[i]), float(arr_y[i]))

        rendered_pil = []
        for i in range(n):
            kp = smoothed_kpts[i]
            if is_wrong_seq:
                cam_rgb, map_pt = v28.apply_cinema_camera(base_rgbs[i], zoom=1.02, focus_xy=(690.0, 920.0), dest_xy=(690.0, 920.0))
                step_i = 1
                el_deg = 105
                sp_cfg = {
                    "is_wrong": True,
                    "pct": 86,
                    "dial_tag": "JOINT RISK",
                    "short_cue": "✕ HUNCHED SPINE",
                    "splits": [("SPINE · JOINT", 86, v30.COL_RED), ("TRICEPS", 18, v30.COL_MUTED)],
                    "rx_num": "35°",
                    "rx_unit": "KYPHOSIS",
                }
                active_slot = 0
            else:
                cam_rgb, map_pt = v28.apply_cinema_camera(base_rgbs[i], zoom=1.05, focus_xy=(688.0, 920.0), dest_xy=(696.0, 920.0))
                ext_norm = float(np.clip((kp["wr"][1] - 880.0) / (1135.0 - 880.0), 0.0, 1.0))
                el_deg = int(round(92 + 88 * ext_norm))
                pct_now = int(round(78 + 21 * ext_norm))
                step_i = 4 if el_deg >= 165 else 3
                sp_cfg = {
                    "is_wrong": False,
                    "pct": pct_now,
                    "dial_tag": "ALL 3 HEADS" if el_deg >= 165 else "TRICEPS ISO",
                    "short_cue": "✓ 180° LOCKOUT" if el_deg >= 165 else "✓ 15° HINGE · PIN",
                    "splits": [
                        ("LAT HEAD", pct_now, v30.COL_GREEN),
                        ("LONG HEAD", max(72, pct_now - 4), v30.COL_GREEN),
                        ("MED HEAD", max(70, pct_now - 3), v30.COL_GREEN),
                    ],
                    "rx_num": f"{el_deg}°",
                    "rx_unit": "LOCKOUT" if el_deg >= 165 else "EXTENSION",
                }
                active_slot = 2

            out = cam_rgb.astype(np.float32)
            left_w = 430
            curve_l = np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75
            out[480:1780, :left_w] *= curve_l[None, :, None]
            im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")

            render_v36_clean_exterior_overlay(im, step_i, map_pt, kp, is_wrong_seq, dynamic_elbow_deg=el_deg)
            v28.draw_luxury_top_dock(im, torso_icons, active_slot, is_wrong_seq)
            v30.draw_v30_graphic_telemetry_bottom_left(im, sp_cfg, mode="A")
            rendered_pil.append(im.convert("RGB"))

        return rendered_pil

    wrong_rendered = process_seq(raw_w_frames, True)
    correct_rendered = process_seq(raw_c_frames, False)

    tmp_mp4 = os.path.join(OUT_DIR, "_tmp_v36_dyn.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day1_v36_dynamic_preview.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for im_pil in (wrong_rendered + correct_rendered):
        vw.write(cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR))
    vw.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_mp4, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast", final_mp4],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_mp4)
    print("Saved v36 Dynamic MP4:", final_mp4)


def main():
    raw_w45 = v25.read_specific_frame(eng.WRONG_I2V, 45)
    raw_c14 = v25.read_specific_frame(eng.CORRECT_I2V, 14)
    raw_c42 = v25.read_specific_frame(eng.CORRECT_I2V, 42)
    raw_c64 = v25.read_specific_frame(eng.CORRECT_I2V, 64)

    s1_rgb, kp1 = v35.render_v35_clean_3d_frame(raw_w45, 1, "xray", True)
    s2_rgb, kp2 = v35.render_v35_clean_3d_frame(raw_c14, 2, "anterior_deltoid", True)
    s3_rgb, kp3 = v35.render_v35_clean_3d_frame(raw_c42, 3, "triceps_lat_long", False)
    s4_rgb, kp4 = v35.render_v35_clean_3d_frame(raw_c64, 4, "triceps_all_three", False)

    size = (142, 114)
    crop_box = (555, 550, 905, 950)
    f_c42_mono = eng.prepare_muscle_frame(raw_c42)["mono"]
    inact_base = Image.fromarray(f_c42_mono).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 125)))

    im0 = Image.fromarray(s1_rgb).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im1 = Image.fromarray(s2_rgb).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im2 = Image.fromarray(s4_rgb).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    steps = [
        {
            "idx": 1,
            "is_wrong": True,
            "base_rgb": s1_rgb,
            "kpts": kp1,
            "active_slot": 0,
            "cam_zoom": 1.02,
            "cam_focus": (690.0, 920.0),
            "cam_dest": (690.0, 920.0),
            "pct": 86,
            "dial_tag": "JOINT RISK",
            "short_cue": "✕ HUNCHED SPINE",
            "splits": [
                ("SPINE · JOINT", 86, v30.COL_RED),
                ("TRICEPS", 18, v30.COL_MUTED),
            ],
            "rx_num": "35°",
            "rx_unit": "KYPHOSIS",
        },
        {
            "idx": 2,
            "is_wrong": True,
            "base_rgb": s2_rgb,
            "kpts": kp2,
            "active_slot": 1,
            "cam_zoom": 1.05,
            "cam_focus": (685.0, 910.0),
            "cam_dest": (695.0, 910.0),
            "pct": 72,
            "dial_tag": "DELT STEAL",
            "short_cue": "✕ 90° HALF REP",
            "splits": [
                ("FRONT DELT", 72, v30.COL_RED),
                ("TRICEPS", 28, v30.COL_MUTED),
            ],
            "rx_num": "90°",
            "rx_unit": "STOP SHORT",
        },
        {
            "idx": 3,
            "is_wrong": False,
            "base_rgb": s3_rgb,
            "kpts": kp3,
            "active_slot": 2,
            "cam_zoom": 1.04,
            "cam_focus": (685.0, 920.0),
            "cam_dest": (695.0, 920.0),
            "pct": 95,
            "dial_tag": "TRICEPS ISO",
            "short_cue": "✓ 15° HINGE · PIN",
            "splits": [
                ("LAT HEAD", 95, v30.COL_GREEN),
                ("LONG HEAD", 91, v30.COL_GREEN),
            ],
            "rx_num": "135°",
            "rx_unit": "EXTENSION",
        },
        {
            "idx": 4,
            "is_wrong": False,
            "base_rgb": s4_rgb,
            "kpts": kp4,
            "active_slot": 2,
            "cam_zoom": 1.06,
            "cam_focus": (690.0, 925.0),
            "cam_dest": (700.0, 920.0),
            "pct": 99,
            "dial_tag": "ALL 3 HEADS",
            "short_cue": "✓ 180° LOCKOUT",
            "splits": [
                ("LAT HEAD", 99, v30.COL_GREEN),
                ("LONG HEAD", 95, v30.COL_GREEN),
                ("MED HEAD", 96, v30.COL_GREEN),
            ],
            "rx_num": "4×12",
            "rx_unit": "REPS · 1s",
        },
    ]

    frames_v36 = []
    for sp in steps:
        cam_rgb, map_pt = v28.apply_cinema_camera(
            sp["base_rgb"],
            zoom=sp["cam_zoom"],
            focus_xy=sp["cam_focus"],
            dest_xy=sp["cam_dest"],
        )
        out = cam_rgb.astype(np.float32)
        left_w = 430
        curve_l = np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75
        out[480:1780, :left_w] *= curve_l[None, :, None]
        clean_rgb = np.clip(out, 0, 255).astype(np.uint8)

        im = Image.fromarray(clean_rgb).convert("RGBA")
        render_v36_clean_exterior_overlay(im, sp["idx"], map_pt, sp["kpts"], sp["is_wrong"])
        v28.draw_luxury_top_dock(im, torso_icons, sp["active_slot"], sp["is_wrong"])
        v30.draw_v30_graphic_telemetry_bottom_left(im, sp, mode="A")

        p_out = os.path.join(OUT_DIR, f"v36_step{sp['idx']}.jpg")
        im.convert("RGB").save(p_out, quality=95)
        frames_v36.append(im.convert("RGB"))

    # 1. Build 4-Step Full-Body 9:16 Master Sheet (`day1_v36_clean_master_sheet.jpg`)
    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "v36 PURE STUDIO MASTER  ·  점선 정렬축 100% 제거 · 순수 3D 근섬유 결 발현 + 외곽 솔리드 아크 & 숫자 (@demicstory 클린 스탠다드)",
        font=f_row,
        fill=v30.COL_GREEN,
    )

    col_titles = [
        "STEP 1 · +28° NECK / 35° HUNCH (✕)",
        "STEP 2 · 90° STOP SHORT / DELT (✕)",
        "STEP 3 · 15° HINGE / 135° EXT (✓)",
        "STEP 4 · 180° LOCKOUT / 25° SPREAD (✓)",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if i < 2 else v30.COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = frames_v36[i].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    p_master = os.path.join(OUT_DIR, "day1_v36_clean_master_sheet.jpg")
    board.save(p_master, quality=95)
    print("Saved:", p_master)

    # 2. Build Top-vs-Bottom Comparison Sheet (`v36_vs_v35_clean_comparison_sheet.jpg`)
    cw, ch = 460, 560
    comp = Image.new("RGB", (cw * 4 + 50, ch * 2 + 135), (10, 12, 16))
    cd = ImageDraw.Draw(comp)
    cd.text(
        (24, 18),
        "v35 vs v36 CLEAN COMPARISON  ·  상단(v35): 점선 정렬축 포함 vs 하단(v36): 점선 100% 제거 · 순수 3D 마네킹 + 외곽 솔리드 아크",
        font=f_row,
        fill=(245, 247, 252),
    )
    crop_rect = (380, 440, 980, 1520)
    for i in range(4):
        v35_im = Image.open(os.path.join(OUT_DIR, f"v35_step{i+1}.jpg")).crop(crop_rect).resize((cw, ch), Image.Resampling.LANCZOS)
        d35 = ImageDraw.Draw(v35_im)
        d35.rectangle([0, 0, cw - 1, ch - 1], outline=(142, 154, 174), width=2)
        d35.rectangle([10, 10, cw - 10, 44], fill=(12, 14, 18))
        d35.text((16, 16), f"v35 Step {i+1} (점선 정렬축 및 다중 포인트 포함)", font=v30.f_pretendard(14, "bold"), fill=(200, 210, 225))
        comp.paste(v35_im, (10 + i * (cw + 10), 68))

        v36_im = frames_v36[i].crop(crop_rect).resize((cw, ch), Image.Resampling.LANCZOS)
        d36 = ImageDraw.Draw(v36_im)
        d36.rectangle([0, 0, cw - 1, ch - 1], outline=(46, 228, 118), width=2)
        d36.rectangle([10, 10, cw - 10, 44], fill=(12, 14, 18))
        d36.text((16, 16), f"v36 Step {i+1} (✓ 점선 100% 제거 · 울트라 클린 3D)", font=v30.f_pretendard(14, "bold"), fill=v30.COL_GREEN)
        comp.paste(v36_im, (10 + i * (cw + 10), 68 + ch + 18))

    p_comp = os.path.join(OUT_DIR, "v36_vs_v35_clean_comparison_sheet.jpg")
    comp.save(p_comp, quality=95)
    print("Saved:", p_comp)

    # 3. Build Cross-Exercise Sheet (`v36_cross_exercise_repeatability_sheet.jpg`)
    d2_w = render_day2_v36_clean_panel("day2_step2_wrong_plate.jpg", "step2_wrong", True)
    d2_c = render_day2_v36_clean_panel("day2_step1_correct_plate.jpg", "step1_correct", False)

    cross_board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    cbd = ImageDraw.Draw(cross_board)
    cbd.text(
        (28, 20),
        "v36 CROSS-EXERCISE PURE STUDIO STANDARD  ·  점선 0% · 순수 3D 근섬유 결 + 외곽 솔리드 아크 (Day 1 푸시다운 & Day 2 랫풀다운)",
        font=f_row,
        fill=v30.COL_GREEN,
    )
    cross_panels = [
        (frames_v36[0], "DAY 1 · 35° HUNCH PUSHDOWN (✕)", True),
        (frames_v36[3], "DAY 1 · 180° LOCKOUT PUSHDOWN (✓)", False),
        (d2_w, "DAY 2 · LAT PULLDOWN 42° SWING (✕)", True),
        (d2_c, "DAY 2 · LAT PULLDOWN SCAPULA LOCK (✓)", False),
    ]
    for i, (im_p, title_c, is_w) in enumerate(cross_panels):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if is_w else v30.COL_GREEN
        cbd.text((x + 4, 58), title_c, font=f_col, fill=col)
        rc_im = im_p.resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc_im).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        cross_board.paste(rc_im, (x, 86))

    p_cross = os.path.join(OUT_DIR, "v36_cross_exercise_repeatability_sheet.jpg")
    cross_board.save(p_cross, quality=95)
    print("Saved:", p_cross)

    # 4. Build Dynamic 30fps MP4 Preview (`day1_v36_dynamic_preview.mp4`)
    build_v36_dynamic_preview(torso_icons)


if __name__ == "__main__":
    main()
