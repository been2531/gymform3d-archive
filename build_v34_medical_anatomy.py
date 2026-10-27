#!/usr/bin/env python3
"""
v34 Master — Universal Pose & Silhouette-Fitted 3D Medical Osteology + Intrinsic 3D Fiber-Strand Engine
=======================================================================================================
1. Powered by MediaPipe PoseLandmarker Heavy (33 keypoints + pixel segmentation mask) fused with
   Arm-Excluded Dorsal Contour Tracking (`universal_anatomy_rig.py`).
2. Continuous Unified 3D Osteology SDF Meshes:
   - Unified Vertebral Column (`Occiput + C1..L5 + Sacrum`) with continuous longitudinal ligament core,
     intervertebral disc waisting, and posterior spinous crest undulations (zero square blocks, zero fake bar ribs!).
   - Unified Pelvis & Single Foreground Lower-Limb Complex (`Ilium/Acetabulum + Unified Femur +
     Integrated Anterior Patella + Unified Tibia & Fibula`) with continuous knee articulation (zero floating dots!).
   - Unified Upper-Limb Complex (`Scapular Acromion/Glenoid + Humerus + Ulnar Olecranon Hook/Trochlear Notch +
     Radial Head + Dual Tapering Radius & Ulna with dynamic pronation crossover + Carpal/Metacarpal Hand`).
   - In Step 1 (`raw_w45` Wrong), preserves the native 3D X-Ray shoulder/humerus/forearm inside the mannequin's
     arms (`render_xray_skeletal_frame`) while adding the silhouette-locked 3D kyphotic spine, pelvis, and lower limb!
"""

import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v25_fast_preview as v25
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import build_day2_ox_preview as d2
import universal_anatomy_rig as uar

OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)
H, W = eng.H, eng.W


def render_v34_hybrid_biomechanics_frame(
    bgr: np.ndarray,
    step_idx: int,
    muscle_mode: str,
    is_wrong: bool,
) -> tuple[np.ndarray, dict]:
    """
    Renders a single Day 1 frame combining:
    1. Intrinsic 3D Fiber-Strand Muscle Coloring (or native 3D X-Ray arm in Step 1).
    2. Universal Pose & Silhouette-Locked 3D Medical Osteology (`universal_anatomy_rig.py`).
    """
    scale = 0.84
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    tw, th = int(W * scale), int(H * scale)
    rgb_r = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_r[:, :, 0], rgb_r[:, :, 1], rgb_r[:, :, 2]
    h_r, w_r = r_ch.shape
    yy_r, xx_r = np.mgrid[:h_r, :w_r].astype(np.float32)

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    gb_diff = np.maximum(0.0, g_ch - b_ch)

    fiber_signal = 0.45 * r_ch + 0.55 * g_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 0.9)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.4)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 6.5)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 14.0)

    hf_ridge = (sigma_micro - sigma_ridge) * 3.2 + (sigma_ridge - sigma_belly) * 1.8
    fiber_strand = np.clip(0.48 + hf_ridge / 42.0, 0.0, 1.0) ** 1.35
    head_volume = np.clip((sigma_belly - 28.0) / 165.0, 0.0, 1.0)
    inter_head_sulcus = np.clip((sigma_broad - sigma_ridge - 3.5) / 22.0, 0.0, 1.0)

    red_any = np.clip((rb_diff - 10.0) / 85.0, 0.0, 1.0)
    green_any = np.clip((g_ch - np.maximum(r_ch, b_ch) - 12.0) / 85.0, 0.0, 1.0)
    non_red_lum = 0.28 * g_ch + 0.72 * b_ch
    grey_sculpt = np.clip(34.0 + 148.0 * (head_volume ** 1.05) + hf_ridge * 0.95, 16.0, 235.0)

    x_left_bound = np.where(yy_r < 562.0, 560.0, 573.0 + (yy_r - 562.0) * 0.055)
    arm_corridor = (
        np.clip((xx_r - x_left_bound) / 8.0, 0.0, 1.0)
        * np.clip((770.0 - xx_r) / 10.0, 0.0, 1.0)
        * np.clip((yy_r - 455.0) / 10.0, 0.0, 1.0)
        * np.clip((735.0 - yy_r) / 14.0, 0.0, 1.0)
    )

    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    neutral_lum = base_lum * (1.0 - red_any) + (non_red_lum * (1.0 - arm_corridor) + grey_sculpt * arm_corridor) * red_any
    hand_lum = 62.0 + 0.62 * r_ch
    neutral_lum = np.clip(neutral_lum * (1.0 - green_any) + hand_lum * green_any, 14.0, 240.0)

    norm = neutral_lum / 255.0
    mono_r = eng.BG_CHARCOAL[0] + (234 - eng.BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = eng.BG_CHARCOAL[1] + (238 - eng.BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = eng.BG_CHARCOAL[2] + (244 - eng.BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    raw_chroma_gate = np.clip((rb_diff - 62.0) / 68.0, 0.0, 1.0) * np.clip((r_ch - 102.0) / 58.0, 0.0, 1.0) * arm_corridor
    bw_gate = (cv2.GaussianBlur(raw_chroma_gate, (0, 0), 2.0) > 0.14).astype(np.uint8)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(bw_gate)
    cc_gate = np.zeros_like(raw_chroma_gate)
    for idx_l in range(1, n_lbl):
        if stats[idx_l, cv2.CC_STAT_AREA] >= 900 and stats[idx_l, cv2.CC_STAT_TOP] < 680:
            cc_gate[lbls == idx_l] = 1.0
    cc_gate = cv2.GaussianBlur(cc_gate, (0, 0), 2.0)

    native_fiber_energy = np.clip((rb_diff - 45.0) / 145.0, 0.0, 1.0) * cc_gate
    native_hot_crest = np.clip((gb_diff - 18.0) / 105.0, 0.0, 1.0) * cc_gate

    canvas, maps = eng.place_in_studio_canvas_clean(
        mono_res,
        {
            "fiber_energy": native_fiber_energy,
            "hot_crest": native_hot_crest,
            "fiber_strand": fiber_strand,
            "head_volume": head_volume,
            "sulcus": inter_head_sulcus,
        },
        scale=scale,
        x_shift=34,
        y_shift=50,
    )
    raw_canvas_rgb, _ = eng.place_in_studio_canvas_clean(rgb_r, {}, scale=scale, x_shift=34, y_shift=50)

    # Solve universal anatomical rig via MediaPipe PoseLandmarker Heavy + Dorsal Contour Tracking
    rig = uar.solve_universal_anatomical_rig(raw_canvas_rgb, "sagittal_right", step_idx)
    body_mask = rig["body_mask"]

    fe = maps["fiber_energy"]
    hc = maps["hot_crest"]
    fs = maps["fiber_strand"]
    hv = maps["head_volume"]
    sulc = maps["sulcus"]

    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    y_elbow_cut = rig["el"][1] - 10.0
    above_elbow_gate = np.clip((y_elbow_cut - yy) / 16.0, 0.0, 1.0)
    fe = fe * above_elbow_gate

    ys, xs = np.where(fe > 0.18)
    y_top = float(np.percentile(ys, 2)) if len(ys) > 40 else 677.0
    y_bot = float(np.percentile(ys, 98)) if len(ys) > 40 else 915.0
    x_med = float(np.median(xs)) if len(xs) > 40 else 755.0

    if step_idx == 1:
        # Step 1: Preserve the native 3D X-Ray shoulder, humerus, elbow & white radius/ulna forearm bones
        # already rendered inside the mannequin's arms (`render_xray_skeletal_frame`), so there is ZERO duplicate bone!
        fd_x = eng.prepare_xray_bone_frame(bgr)
        base_out, _, _, _ = eng.render_xray_skeletal_frame(fd_x, 1.0, 0.92)
        out = base_out.astype(np.float32)
        # Protect the native X-ray arm zone (`fd_x["bone_mask"]`) so our spine/pelvis overlay doesn't clash with the humerus head!
        muscle_protect = np.clip(fd_x["bone_mask"] * 1.35, 0.0, 0.85)
    else:
        y_delt_v = (y_top + 104.0) - ((xx - x_med) ** 2) * 0.0018
        delt_w = np.clip((y_delt_v - yy + 5.0) / 10.0, 0.0, 1.0)
        tri_w = 1.0 - delt_w

        if muscle_mode == "anterior_deltoid":
            region_w = delt_w
        elif muscle_mode == "triceps_lat_long":
            y_distal = (y_top + 104.0) + 0.78 * max(80.0, (y_bot - (y_top + 104.0)))
            distal_taper = np.clip((y_distal - yy + 22.0) / 38.0, 0.35, 1.0)
            region_w = tri_w * distal_taper
        else:
            region_w = tri_w

        strand_modulation = (0.22 + 0.78 * fs) * (1.0 - 0.65 * sulc)
        fiber_alpha = np.clip(fe * region_w * (0.58 + 0.66 * strand_modulation), 0.0, 0.96)[:, :, None]
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
        out = canvas.astype(np.float32) * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha
        muscle_protect = np.clip(fe * region_w * 1.45, 0.0, 1.0) * np.clip((y_elbow_cut - 14.0 - yy) / 14.0, 0.0, 1.0)

    # =========================================================================
    # 2. UNIVERSAL SILHOUETTE-LOCKED 3D MEDICAL OSTEOLOGY SDF RENDERING
    # =========================================================================
    bone_layer = np.zeros((H, W, 3), dtype=np.float32)
    xray_win = np.zeros((H, W), dtype=np.float32)

    if is_wrong:
        col_cort = (255.0, 202.0, 172.0)
        col_trab = (242.0, 68.0, 50.0)
        col_sp_cort = (255.0, 188.0, 160.0)
        col_sp_trab = (228.0, 58.0, 44.0)
    else:
        col_cort = (212.0, 252.0, 234.0)
        col_trab = (30.0, 198.0, 128.0)
        col_sp_cort = (198.0, 244.0, 226.0)
        col_sp_trab = (26.0, 178.0, 114.0)

    if step_idx == 2:
        col_sp_cort = (210.0, 234.0, 248.0)
        col_sp_trab = (95.0, 142.0, 178.0)

    # (A) Continuous Unified 3D Spinal Column (`Occiput + C1..L5 + Sacrum`)
    uar.draw_unified_3d_spinal_column(
        bone_layer, xray_win,
        spine_ctrl=rig["spine"],
        dorsal_side_sign=rig["dorsal_sign"],
        col_cortical=col_sp_cort,
        col_trabecular=col_sp_trab,
        opacity=0.68 if step_idx == 1 else 0.58,
        is_rear_view=False,
    )

    # (B) Continuous Unified Pelvis + Single Foreground Lower-Limb Complex (`Ilium/Acetabulum + Femur + Patella + Tibia/Fibula`)
    uar.draw_unified_3d_pelvis_and_leg(
        bone_layer, xray_win,
        sacrum_pt=rig["sacrum"],
        hip_pt=rig["hip"],
        knee_pt=rig["knee"],
        ankle_pt=rig["ank"],
        col_cortical=col_sp_cort,
        col_trabecular=col_sp_trab,
        opacity=0.68 if step_idx == 1 else 0.62,
    )

    # (C) Unified Upper-Limb Complex (`Scapular Acromion/Glenoid + Humerus + Olecranon Hook + Radius & Ulna + Hand`)
    # Note: In Step 1, `preserve_native_arm_xray=True`, so we keep `raw_w45`'s native 3D X-ray arm untouched!
    if not rig["preserve_native_arm_xray"]:
        uar.draw_unified_3d_shoulder_and_arm(
            bone_layer, xray_win,
            sh_pt=rig["sh"],
            el_pt=rig["el"],
            wr_pt=rig["wr"],
            hd_pt=rig["hd"],
            pronation=rig["pronation"],
            col_cortical=col_cort,
            col_trabecular=col_trab,
            draw_full_humerus=rig["draw_full_humerus"],
            opacity=0.84,
        )

    containment = body_mask * (1.0 - muscle_protect * 0.88)
    xray_win = xray_win * containment
    bone_layer = bone_layer * containment[:, :, None]

    out = out * (1.0 - xray_win[:, :, None] * 0.44) + bone_layer
    return np.clip(out, 0.0, 255.0).astype(np.uint8), rig


def render_v34_fullbody_exterior_callouts(canvas: Image.Image, step_idx: int, map_pt, is_wrong: bool):
    """Combines v30's exterior posture & forearm arcs with full-body Neck & Knee/Stance alignment callouts."""
    v30.render_zero_occlusion_exterior_graphics(canvas, step_idx, map_pt, is_wrong)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    col_main = v30.COL_RED if is_wrong else v30.COL_GREEN
    col_soft = v30.COL_RED_SOFT if is_wrong else v30.COL_GREEN_SOFT

    if step_idx == 1:
        p_nk = map_pt(715, 485)
        v30.draw_floating_number_badge(od, sd, (int(p_nk[0]), int(p_nk[1])), "28°", "NECK CRANE", col_main, col_soft, align="left")
    elif step_idx == 2:
        p_nk = map_pt(785, 485)
        v30.draw_floating_number_badge(od, sd, (int(p_nk[0]), int(p_nk[1])), "0°", "NECK ALIGN", (180, 230, 255), (200, 240, 255), align="left")
    elif step_idx in (3, 4):
        p_kn = map_pt(805, 1300)
        v30.draw_floating_number_badge(od, sd, (int(p_kn[0]), int(p_kn[1])), "165°", "KNEE BASE", col_main, col_soft, align="left")

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=9)))
    canvas.alpha_composite(ov)


def render_day2_universal_frame(plate_fn: str, scene_type: str, is_wrong: bool) -> np.ndarray:
    """
    Applies the EXACT SAME Universal MediaPipe Pose + Silhouette-Fitted 3D Medical Osteology Engine
    to Day 2: Wide-Grip Lat Pulldown (Seated Rear & 3/4 Back Views)!
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

    if scene_type == "step1_wrong":
        zone = np.exp(-(((xx - 665.0) / 165.0) ** 2 + ((yy - 815.0) / 85.0) ** 2))
    elif scene_type == "step1_correct":
        z_l = np.exp(-(((xx - 565.0) / 115.0) ** 2 + ((yy - 885.0) / 145.0) ** 2))
        z_r = np.exp(-(((xx - 745.0) / 115.0) ** 2 + ((yy - 885.0) / 145.0) ** 2))
        zone = np.clip(z_l + z_r, 0.0, 1.0) * np.clip((yy - 715.0) / 45.0, 0.0, 1.0)
    elif scene_type == "step2_wrong":
        zone = np.exp(-(((xx - 635.0) / 135.0) ** 2 + ((yy - 835.0) / 110.0) ** 2))
    else:
        zone = np.exp(-(((xx - 695.0) / 135.0) ** 2 + ((yy - 915.0) / 145.0) ** 2))

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
        col_cort, col_trab = (255.0, 202.0, 172.0), (242.0, 68.0, 50.0)
    else:
        deep_col = np.array([8.0, 110.0, 48.0], dtype=np.float32)
        mid_col = np.array([22.0, 224.0, 98.0], dtype=np.float32)
        crest_col = np.array([185.0, 255.0, 205.0], dtype=np.float32)
        col_cort, col_trab = (212.0, 252.0, 234.0), (30.0, 198.0, 128.0)

    fiber_rgb = (
        deep_col * (1.0 - vol_shade)
        + mid_col * vol_shade * (1.0 - ridge_boost * 0.65)
        + crest_col * (ridge_boost ** 1.35) * 0.85
    )
    out = mono * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha

    bone_layer = np.zeros((H, W, 3), dtype=np.float32)
    xray_win = np.zeros((H, W), dtype=np.float32)

    # Render Unified 3D Spinal Column along the MediaPipe-solved back midline
    uar.draw_unified_3d_spinal_column(
        bone_layer, xray_win,
        spine_ctrl=rig["spine"],
        dorsal_side_sign=1.0,
        col_cortical=col_cort,
        col_trabecular=col_trab,
        opacity=0.64,
        is_rear_view=True,
    )
    # Render Left & Right Upper Limbs fitted to MediaPipe shoulders, elbows, and wrists
    uar.draw_unified_3d_shoulder_and_arm(
        bone_layer, xray_win,
        sh_pt=rig["sh_L"], el_pt=rig["el_L"], wr_pt=rig["wr_L"], hd_pt=rig["hd_L"],
        pronation=0.35, col_cortical=col_cort, col_trabecular=col_trab,
        draw_full_humerus=True, opacity=0.76,
    )
    uar.draw_unified_3d_shoulder_and_arm(
        bone_layer, xray_win,
        sh_pt=rig["sh_R"], el_pt=rig["el_R"], wr_pt=rig["wr_R"], hd_pt=rig["hd_R"],
        pronation=0.35, col_cortical=col_cort, col_trabecular=col_trab,
        draw_full_humerus=True, opacity=0.72,
    )

    protect = np.clip(fe * 1.15, 0.0, 0.75)
    containment = body_mask * (1.0 - protect)
    xray_win = xray_win * containment
    bone_layer = bone_layer * containment[:, :, None]

    out = out * (1.0 - xray_win[:, :, None] * 0.44) + bone_layer
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def main():
    raw_w45 = v25.read_specific_frame(eng.WRONG_I2V, 45)
    raw_c14 = v25.read_specific_frame(eng.CORRECT_I2V, 14)
    raw_c42 = v25.read_specific_frame(eng.CORRECT_I2V, 42)
    raw_c64 = v25.read_specific_frame(eng.CORRECT_I2V, 64)

    s1_rgb, rig1 = render_v34_hybrid_biomechanics_frame(raw_w45, 1, "xray", True)
    s2_rgb, rig2 = render_v34_hybrid_biomechanics_frame(raw_c14, 2, "anterior_deltoid", True)
    s3_rgb, rig3 = render_v34_hybrid_biomechanics_frame(raw_c42, 3, "triceps_lat_long", False)
    s4_rgb, rig4 = render_v34_hybrid_biomechanics_frame(raw_c64, 4, "triceps_all_three", False)

    # Top 3-thumbnail dock icons
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

    frames_v34 = []
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
        render_v34_fullbody_exterior_callouts(im, sp["idx"], map_pt, sp["is_wrong"])
        v28.draw_luxury_top_dock(im, torso_icons, sp["active_slot"], sp["is_wrong"])
        v30.draw_v30_graphic_telemetry_bottom_left(im, sp, mode="A")

        p_out = os.path.join(OUT_DIR, f"v34_step{sp['idx']}.jpg")
        im.convert("RGB").save(p_out, quality=95)
        frames_v34.append(im.convert("RGB"))

    # 1. Build 4-Step Full-Body 9:16 Master Sheet (v34 + v33 compatible path)
    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "v34 MEDICAL ANATOMY MASTER  ·  MediaPipe 33-Keypoint + 실루엣 자동 추적 일체형 3D 척추·견갑·주두(Olecranon)·요골/척골·골반·대퇴·슬개·경비골",
        font=f_row,
        fill=v30.COL_GREEN,
    )

    col_titles = [
        "STEP 1 · +28° NECK / 35° HUNCH (✕)",
        "STEP 2 · 90° FOREARM STOP / DELT (✕)",
        "STEP 3 · 15° SPINE / 135° RADIUS·ULNA (✓)",
        "STEP 4 · 180° LOCKOUT / 25° PRONATION (✓)",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if i < 2 else v30.COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = frames_v34[i].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    for fn in ["day1_v34_medical_anatomy_sheet.jpg", "day1_v33_medical_anatomy_sheet.jpg"]:
        p = os.path.join(OUT_DIR, fn)
        board.save(p, quality=95)
        print("Saved:", p)

    # 2. Build High-Res Close-Up Osteology QC Sheet
    cw, ch = 460, 560
    comp = Image.new("RGB", (cw * 4 + 50, ch * 2 + 135), (10, 12, 16))
    cd = ImageDraw.Draw(comp)
    cd.text(
        (24, 18),
        "v34 HIGH-RES OSTEOLOGY QC  ·  상단: 일체형 3D 경추·흉요추·주두(Olecranon)·요골/척골  |  하단: 골반(비구)·대퇴골(골두·경부·대전자)·일체형 무릎(슬개골·경비골)",
        font=f_row,
        fill=(245, 247, 252),
    )
    raw_steps = [s1_rgb, s2_rgb, s3_rgb, s4_rgb]
    upper_boxes = [
        (460, 480, 950, 1260),
        (460, 480, 880, 1140),
        (480, 480, 880, 1200),
        (480, 480, 900, 1220),
    ]
    lower_boxes = [
        (560, 1160, 960, 1740),
        (520, 980, 840, 1680),
        (520, 980, 840, 1680),
        (520, 980, 840, 1680),
    ]
    u_labels = [
        "1A. 35° 등 라인 밀착 3D 척추 + 네이티브 X-Ray 팔 일체화",
        "2A. 후방 경추·흉추 + 삼각근 결 + 90° 주두·요골/척골",
        "3A. 15° 척추 정렬 + 외측·장두 결 + 135° 요골·척골",
        "4A. 3두 전체 근섬유 결 + 180° 요골·척골 회내 교차",
    ]
    l_labels = [
        "1B. 골반 비구 + 일체형 대퇴골(대전자·골수강)·무릎",
        "2B. 골반 장골·비구 + 대퇴골두·경부 + 일체형 슬개·경비골",
        "3B. 165° 일체형 무릎 관절(대퇴과·슬개골·경골고평부)",
        "4B. 해부학 단일 전면 다리 골격(대퇴골·슬개골·경골·비골)",
    ]
    for i in range(4):
        pu = Image.fromarray(raw_steps[i]).crop(upper_boxes[i]).resize((cw, ch), Image.Resampling.LANCZOS)
        pdu = ImageDraw.Draw(pu)
        pdu.rectangle([0, 0, cw - 1, ch - 1], outline=(50, 58, 72), width=2)
        pdu.rectangle([10, 10, cw - 10, 44], fill=(12, 14, 18))
        pdu.text((16, 16), u_labels[i], font=v30.f_pretendard(14, "bold"), fill=v30.COL_RED if i < 2 else v30.COL_GREEN)
        comp.paste(pu, (10 + i * (cw + 10), 68))

        pl = Image.fromarray(raw_steps[i]).crop(lower_boxes[i]).resize((cw, ch), Image.Resampling.LANCZOS)
        pdl = ImageDraw.Draw(pl)
        pdl.rectangle([0, 0, cw - 1, ch - 1], outline=(50, 58, 72), width=2)
        pdl.rectangle([10, 10, cw - 10, 44], fill=(12, 14, 18))
        pdl.text((16, 16), l_labels[i], font=v30.f_pretendard(14, "bold"), fill=v30.COL_RED if i < 2 else v30.COL_GREEN)
        comp.paste(pl, (10 + i * (cw + 10), 68 + ch + 18))

    for fn in ["day1_v34_closeup_osteology_qc.jpg", "day1_v33_closeup_osteology_qc.jpg"]:
        p = os.path.join(OUT_DIR, fn)
        comp.save(p, quality=95)
        print("Saved:", p)

    # 3. Build Universal Cross-Exercise Fit Verification Sheet (Day 1 Pushdown + Day 2 Wide-Grip Lat Pulldown)
    d2_s1_c = render_day2_universal_frame("day2_step1_correct_plate.jpg", "step1_correct", False)
    d2_s2_w = render_day2_universal_frame("day2_step2_wrong_plate.jpg", "step2_wrong", True)

    cross_board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    cbd = ImageDraw.Draw(cross_board)
    cbd.text(
        (28, 20),
        "v34 UNIVERSAL CROSS-EXERCISE ENGINE  ·  MediaPipe 포즈+실루엣 엔진 Day 1 (스탠딩 푸시다운) & Day 2 (시티드 랫풀다운) 자동 피팅",
        font=f_row,
        fill=v30.COL_GREEN,
    )
    cross_panels = [
        (s1_rgb, "DAY 1 · 35° HUNCH PUSHDOWN (✕)", True),
        (s4_rgb, "DAY 1 · 180° LOCKOUT PUSHDOWN (✓)", False),
        (d2_s2_w, "DAY 2 · LAT PULLDOWN BACK SWING (✕)", True),
        (d2_s1_c, "DAY 2 · LAT PULLDOWN SCAPULA LOCK (✓)", False),
    ]
    for i, (arr_rgb, title_c, is_w) in enumerate(cross_panels):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if is_w else v30.COL_GREEN
        cbd.text((x + 4, 58), title_c, font=f_col, fill=col)
        rc_im = Image.fromarray(arr_rgb).resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc_im).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        cross_board.paste(rc_im, (x, 86))

    for fn in ["v34_universal_cross_exercise_fit_qc.jpg", "v33_universal_cross_exercise_fit_qc.jpg"]:
        p = os.path.join(OUT_DIR, fn)
        cross_board.save(p, quality=95)
        print("Saved:", p)


if __name__ == "__main__":
    main()
