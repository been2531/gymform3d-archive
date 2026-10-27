#!/usr/bin/env python3
"""
v32 Master:
1. True 3D Fiber-Intrinsic Muscle Coloring ('타겟팅되는 근육결 자체에 색상이 들어감'):
   - Zero morphological closing, zero flat paint overlay.
   - Injects color directly into each individual 3D muscle fiber ridge (native R-B/G-B chroma x high-frequency fiber ridges),
     keeping inter-fiber grooves (sulci) and inter-head anatomical clefts dark sculpted charcoal.
2. Synchronized 3D X-Ray Bone Articulation ('움직이는 동작에 맞춰서 뼈도 같이 어떻게 움직여지는지'):
   - Renders volumetric 3D X-Ray bones (cortical bone rim + trabecular epiphysis + dark medullary marrow cavity)
     for the Shoulder Socket, Elbow Hinge Condyles, and Moving Forearm Dual Bones (Radius & Ulna + Carpal Wrist)
     rotating from 90° flexion (f=14) -> 135° extension (f=42) -> 180° full lockout (f=64),
     WITHOUT drawing any line/rod over the upper-arm Triceps/Deltoid muscle fibers!
3. Full-Body Posture Skeleton ('다리나 허리나 목이나 이런데도 자세가 중요하겠지'):
   - Renders 3D X-Ray Cervical Spine (Neck), Thoracic-Lumbar Vertebrae (Back/Waist), Pelvis/Hip Socket,
     and Femur-Knee-Tibia Lower-Body Stance Chain across all 4 steps in a full-body 3/4 framing.
"""

import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v25_fast_preview as v25
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30

OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)
H, W = eng.H, eng.W


def _trabecular_3d_texture(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Generates fine multi-scale 3D spongy trabecular bone micro-architecture."""
    t1 = np.sin(xx * 0.37 + yy * 0.23) * np.cos(xx * 0.19 - yy * 0.41)
    t2 = np.sin(xx * 0.83 - yy * 0.67) * np.cos(xx * 0.53 + yy * 0.79)
    t3 = np.sin(xx * 1.45 + yy * 1.12)
    return np.clip(0.55 + 0.25 * t1 + 0.14 * t2 + 0.06 * t3, 0.15, 1.0)


def draw_volumetric_3d_bone_segment(
    bone_layer: np.ndarray,
    xray_win: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    r_prox: float,
    r_shaft: float,
    r_dist: float,
    col_cortical: tuple[float, float, float],
    col_trabecular: tuple[float, float, float],
    opacity: float = 0.72,
):
    """
    Renders a solid, 3D-lit anatomical bone (NOT a hollow white outline!):
    - True 3D cylindrical volume shading (sqrt(1 - u^2)) + directional key light
    - Flared anatomical condyle/epiphysis heads at t=0 and t=1 with spongy trabecular texture
    - Smoothly embedded inside the translucent X-Ray cavity window.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    pad = int(max(r_prox, r_dist) + 26)
    xmin = max(0, int(min(x0, x1)) - pad)
    xmax = min(W, int(max(x0, x1)) + pad)
    ymin = max(0, int(min(y0, y1)) - pad)
    ymax = min(H, int(max(y0, y1)) + pad)
    if xmin >= xmax or ymin >= ymax:
        return

    yy, xx = np.mgrid[ymin:ymax, xmin:xmax].astype(np.float32)
    rx, ry = xx - x0, yy - y0

    proj = rx * ux + ry * uy
    t = np.clip(proj / length, 0.0, 1.0)
    perp = rx * nx + ry * ny
    dist_axis = np.hypot(rx - t * dx, ry - t * dy)

    # Smooth hourglass bone profile: flared epiphyses at ends, slender diaphysis shaft in center
    end_flare = np.clip(np.sin(t * math.pi), 0.0, 1.0) ** 1.6  # 1 at center, 0 at ends
    r_ends = r_prox * (1.0 - t) + r_dist * t
    radius = r_shaft * end_flare + r_ends * (1.0 - end_flare)

    u = dist_axis / np.maximum(1.0, radius)
    inside = np.clip((1.0 - u) / 0.18, 0.0, 1.0)

    # Solid 3D cylindrical shading (highest along the 3D bone ridge, rolling off smoothly to edges)
    cyl_z = np.sqrt(np.clip(1.0 - (u * 0.92) ** 2, 0.0, 1.0))
    # 3D directional light across cylinder normal
    u_signed = np.clip(perp / np.maximum(1.0, radius), -1.0, 1.0)
    light_3d = np.clip(0.68 * cyl_z + 0.32 * np.clip(0.35 - u_signed, 0.0, 1.0), 0.0, 1.0)

    trab = _trabecular_3d_texture(xx, yy)
    bone_val = np.clip(light_3d * (0.76 + 0.24 * trab), 0.0, 1.0) * inside

    win_soft = np.exp(-((dist_axis / (radius * 1.7)) ** 2)) * 0.42 * opacity
    xray_win[ymin:ymax, xmin:xmax] = np.maximum(xray_win[ymin:ymax, xmin:xmax], win_soft)

    c_cort = np.array(col_cortical, dtype=np.float32)
    c_trab = np.array(col_trabecular, dtype=np.float32)
    # Rich 3D bone gradient: deep trabecular tone in shadow -> warm/cool cortical ivory highlight on 3D ridge crest
    rgb_seg = (
        c_trab * ((bone_val ** 0.85)[:, :, None] * 0.65)
        + c_cort * ((bone_val ** 1.85)[:, :, None] * 0.75)
    )
    bone_layer[ymin:ymax, xmin:xmax] = np.maximum(
        bone_layer[ymin:ymax, xmin:xmax],
        rgb_seg * (inside[:, :, None] * opacity),
    )


def draw_3d_vertebrae_chain(
    bone_layer: np.ndarray,
    xray_win: np.ndarray,
    pts: list[tuple[float, float]],
    n_vertebrae: int,
    r_width: float,
    r_height: float,
    col_cortical: tuple[float, float, float],
    col_trabecular: tuple[float, float, float],
    opacity: float = 0.68,
):
    """
    Renders realistic sagittal 3D vertebral bodies (brick/wedge-shaped vertebral drums + posterior spinous processes
    + continuous spinal canal cord, NEVER round circles!).
    """
    pts_arr = np.array(pts, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, len(pts))
    # First draw a subtle continuous spinal column core so vertebrae are connected anatomically
    t_fine = np.linspace(0.0, 1.0, 40)
    xs_f = np.interp(t_fine, t_in, pts_arr[:, 0])
    ys_f = np.interp(t_fine, t_in, pts_arr[:, 1])
    for k in range(len(t_fine) - 1):
        draw_volumetric_3d_bone_segment(
            bone_layer, xray_win,
            (float(xs_f[k]), float(ys_f[k])),
            (float(xs_f[k + 1]), float(ys_f[k + 1])),
            r_prox=6.5, r_shaft=6.0, r_dist=6.5,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=opacity * 0.65,
        )

    t_out = np.linspace(0.0, 1.0, n_vertebrae)
    xs = np.interp(t_out, t_in, pts_arr[:, 0])
    ys = np.interp(t_out, t_in, pts_arr[:, 1])

    c_cort = np.array(col_cortical, dtype=np.float32)
    c_trab = np.array(col_trabecular, dtype=np.float32)

    for i in range(n_vertebrae):
        cx, cy = float(xs[i]), float(ys[i])
        if i < n_vertebrae - 1:
            tx, ty = float(xs[i + 1] - xs[i]), float(ys[i + 1] - ys[i])
        else:
            tx, ty = float(xs[i] - xs[i - 1]), float(ys[i] - ys[i - 1])
        tlen = max(1.0, math.hypot(tx, ty))
        tx, ty = tx / tlen, ty / tlen
        nx, ny = -ty, tx

        scale_v = 0.70 + 0.40 * (i / max(1, n_vertebrae - 1))
        vw = r_width * scale_v
        vh = r_height * (0.78 + 0.22 * (i / max(1, n_vertebrae - 1)))

        pad = int(max(vw, vh) * 2.4 + 12)
        xmin, xmax = max(0, int(cx) - pad), min(W, int(cx) + pad)
        ymin, ymax = max(0, int(cy) - pad), min(H, int(cy) + pad)
        if xmin >= xmax or ymin >= ymax:
            continue

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        u_along = (dx * tx + dy * ty) / vh
        u_perp = (dx * nx + dy * ny) / vw

        # Boxy sagittal vertebral body (high-order squircle |u_along|^4 + |u_perp|^2.5) + angled posterior spinous fin
        box_dist = ((np.abs(u_along) ** 3.6) + (np.abs(u_perp) ** 2.4)) ** (1.0 / 3.0)
        inside_body = np.clip((1.0 - box_dist) / 0.20, 0.0, 1.0)

        # Posterior spinous process fin pointing backward-downward (u_perp > 0.6)
        fin_u = (u_perp - 1.15) / 0.55
        fin_v = (u_along - 0.25 * (u_perp - 0.6)) / 0.38
        inside_fin = np.clip(1.0 - (fin_u ** 2 + fin_v ** 2), 0.0, 1.0) * (u_perp > 0.4).astype(np.float32)

        inside = np.clip(inside_body + inside_fin * 0.78, 0.0, 1.0)
        cyl_3d = np.sqrt(np.clip(1.0 - (u_perp * 0.75) ** 2, 0.0, 1.0))
        endplate_hi = np.exp(-((np.abs(u_along) - 0.75) / 0.22) ** 2) * inside_body * 0.35
        trab = _trabecular_3d_texture(xx, yy)

        b_val = np.clip((cyl_3d * 0.78 + endplate_hi) * (0.78 + 0.22 * trab), 0.0, 1.0) * inside

        win_local = np.exp(-((box_dist / 1.55) ** 2)) * 0.34 * opacity
        xray_win[ymin:ymax, xmin:xmax] = np.maximum(xray_win[ymin:ymax, xmin:xmax], win_local)

        rgb_v = (
            c_trab * ((b_val ** 0.85)[:, :, None] * 0.62)
            + c_cort * ((b_val ** 1.75)[:, :, None] * 0.72)
        )
        bone_layer[ymin:ymax, xmin:xmax] = np.maximum(
            bone_layer[ymin:ymax, xmin:xmax],
            rgb_v * (inside[:, :, None] * opacity),
        )


def draw_3d_joint_epiphysis(
    bone_layer: np.ndarray,
    xray_win: np.ndarray,
    center: tuple[float, float],
    radius: float,
    col_cortical: tuple[float, float, float],
    col_trabecular: tuple[float, float, float],
    opacity: float = 0.74,
):
    """Renders a solid 3D hemispherical articular condyle / joint head with trabecular internal texture."""
    cx, cy = center
    pad = int(radius * 2.0 + 12)
    xmin, xmax = max(0, int(cx) - pad), min(W, int(cx) + pad)
    ymin, ymax = max(0, int(cy) - pad), min(H, int(cy) + pad)
    if xmin >= xmax or ymin >= ymax:
        return

    yy, xx = np.mgrid[ymin:ymax, xmin:xmax].astype(np.float32)
    dist = np.hypot(xx - cx, yy - cy)
    u = dist / max(1.0, radius)
    inside = np.clip((1.0 - u) / 0.20, 0.0, 1.0)

    sphere_z = np.sqrt(np.clip(1.0 - (u * 0.92) ** 2, 0.0, 1.0))
    trab = _trabecular_3d_texture(xx, yy)
    b_val = np.clip(sphere_z * (0.76 + 0.24 * trab), 0.0, 1.0) * inside

    win_local = np.exp(-((u / 1.55) ** 2)) * 0.38 * opacity
    xray_win[ymin:ymax, xmin:xmax] = np.maximum(xray_win[ymin:ymax, xmin:xmax], win_local)

    c_cort = np.array(col_cortical, dtype=np.float32)
    c_trab = np.array(col_trabecular, dtype=np.float32)
    rgb_j = (
        c_trab * ((b_val ** 0.85)[:, :, None] * 0.65)
        + c_cort * ((b_val ** 1.80)[:, :, None] * 0.72)
    )
    bone_layer[ymin:ymax, xmin:xmax] = np.maximum(
        bone_layer[ymin:ymax, xmin:xmax],
        rgb_j * (inside[:, :, None] * opacity),
    )


def render_v32_hybrid_biomechanics_frame(
    bgr: np.ndarray,
    step_idx: int,
    muscle_mode: str,
    is_wrong: bool,
):
    """
    Combines:
    1. True 3D Fiber-Intrinsic Muscle Coloring on the active muscle strands (Deltoid / Triceps).
    2. Synchronized 3D X-Ray Bone Articulation on the moving Forearm (Radius & Ulna), Elbow Hinge, Shoulder Socket,
       AND Full-Body Posture Skeleton (Cervical Neck, Thoracic-Lumbar Spine/Waist, Pelvis/Hip, Femur & Knee).
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

    # Extract 3D muscle fiber ridges (positive on individual fiber strands, zero in grooves)
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
        * np.clip((756.0 - xx_r) / 10.0, 0.0, 1.0)
        * np.clip((yy_r - 455.0) / 10.0, 0.0, 1.0)
        * np.clip((845.0 - yy_r) / 12.0, 0.0, 1.0)
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
        if stats[idx_l, cv2.CC_STAT_AREA] >= 1200 and stats[idx_l, cv2.CC_STAT_TOP] < 705:
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

    fe = maps["fiber_energy"]
    hc = maps["hot_crest"]
    fs = maps["fiber_strand"]
    hv = maps["head_volume"]
    sulc = maps["sulcus"]

    ys, xs = np.where(fe > 0.18)
    y_top = float(np.percentile(ys, 2)) if len(ys) > 40 else 677.0
    y_bot = float(np.percentile(ys, 98)) if len(ys) > 40 else 960.0
    x_med = float(np.median(xs)) if len(xs) > 40 else 745.0

    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    # Step 1 uses native 3D X-Ray bone from WRONG_I2V + full-body compromised skeleton!
    if step_idx == 1:
        fd_x = eng.prepare_xray_bone_frame(bgr)
        base_out, _, _, _ = eng.render_xray_skeletal_frame(fd_x, 1.0, 0.92)
        out = base_out.astype(np.float32)
    else:
        # Anatomically locked Deltoid V-insertion (104px below acromion y_top)
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
        fiber_alpha = np.clip(fe * region_w * (0.56 + 0.66 * strand_modulation), 0.0, 0.96)[:, :, None]
        ridge_boost = np.clip(fs * 0.65 + hc * 0.55, 0.0, 1.0)[:, :, None]
        vol_shade = np.clip(0.28 + 0.72 * hv * (0.45 + 0.55 * fs), 0.0, 1.0)[:, :, None]

        if is_wrong:
            deep_col = np.array([155.0, 18.0, 16.0], dtype=np.float32)
            mid_col  = np.array([248.0, 52.0, 34.0], dtype=np.float32)
            crest_col = np.array([255.0, 195.0, 110.0], dtype=np.float32)
        else:
            deep_col = np.array([8.0, 110.0, 48.0], dtype=np.float32)
            mid_col  = np.array([22.0, 224.0, 98.0], dtype=np.float32)
            crest_col = np.array([185.0, 255.0, 205.0], dtype=np.float32)

        fiber_rgb = (
            deep_col * (1.0 - vol_shade)
            + mid_col * vol_shade * (1.0 - ridge_boost * 0.65)
            + crest_col * (ridge_boost ** 1.35) * 0.85
        )
        out = canvas.astype(np.float32) * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha

    # =========================================================================
    # 2. SYNCHRONIZED 3D X-RAY SKELETAL ARTICULATION (Arm Forearm/Joints + Full-Body Spine/Neck/Waist/Pelvis/Knee)
    # =========================================================================
    bone_layer = np.zeros((H, W, 3), dtype=np.float32)
    xray_win = np.zeros((H, W), dtype=np.float32)

    if is_wrong:
        # Warm anatomical X-Ray bone (ivory-coral cortical rim + crimson-amber trabecular core)
        col_cort = (255.0, 195.0, 165.0)
        col_trab = (240.0, 68.0, 52.0)
        col_spine_cort = (255.0, 175.0, 150.0)
        col_spine_trab = (220.0, 58.0, 46.0)
    else:
        # Cool clinical X-Ray bone (ice-titanium cortical rim + emerald-cyan trabecular core)
        col_cort = (210.0, 250.0, 232.0)
        col_trab = (32.0, 195.0, 128.0)
        col_spine_cort = (195.0, 242.0, 225.0)
        col_spine_trab = (28.0, 175.0, 115.0)

    if step_idx == 1:
        # Step 1 (Hunched Back 35°): Show compromised Cervical Neck + Kyphotic Thoracic/Lumbar Spine + Pelvis + Knee
        spine_pts = [
            (630.0, 520.0),  # C2 Upper Cervical Neck (craned forward)
            (685.0, 585.0),  # C7 Base of Neck
            (748.0, 690.0),  # T5 Mid-Thoracic (kyphotic hump!)
            (756.0, 825.0),  # T12 Thoracolumbar
            (736.0, 965.0),  # L5 Lower Lumbar Waist
        ]
        draw_3d_vertebrae_chain(bone_layer, xray_win, spine_pts, n_vertebrae=14, r_width=16.0, r_height=9.5,
                                col_cortical=col_spine_cort, col_trabecular=col_spine_trab, opacity=0.82)
        # Pelvis / Hip & Knee
        hip_pt = (722.0, 1025.0)
        knee_pt = (688.0, 1345.0)
        ankle_pt = (706.0, 1635.0)
        draw_3d_joint_epiphysis(bone_layer, xray_win, hip_pt, 22.0, col_spine_cort, col_spine_trab, opacity=0.75)
        draw_volumetric_3d_bone_segment(bone_layer, xray_win, hip_pt, knee_pt, 18.0, 10.5, 20.0, col_spine_cort, col_spine_trab, opacity=0.68)
        draw_3d_joint_epiphysis(bone_layer, xray_win, knee_pt, 20.0, col_spine_cort, col_spine_trab, opacity=0.78)
        draw_volumetric_3d_bone_segment(bone_layer, xray_win, knee_pt, ankle_pt, 17.0, 9.5, 14.0, col_spine_cort, col_spine_trab, opacity=0.62)

        # Forearm Radius & Ulna at f=45 (wrong hunched pushdown)
        el_pt = (725.0, 885.0)
        wr_pt = (565.0, 1065.0)
        draw_volumetric_3d_bone_segment(bone_layer, xray_win, (el_pt[0]-5, el_pt[1]-4), (wr_pt[0]-4, wr_pt[1]-5),
                                        14.0, 7.5, 12.0, col_cort, col_trab, opacity=0.82)
        draw_volumetric_3d_bone_segment(bone_layer, xray_win, (el_pt[0]+6, el_pt[1]+5), (wr_pt[0]+5, wr_pt[1]+5),
                                        13.0, 7.0, 11.0, col_cort, col_trab, opacity=0.80)
        draw_3d_joint_epiphysis(bone_layer, xray_win, wr_pt, 14.0, col_cort, col_trab, opacity=0.82)

    else:
        # Steps 2, 3, 4: Upright Neutral Cervical Neck + 15° Hinged Thoracic-Lumbar Spine + Pelvis + Athletic Knee Stance
        spine_pts = [
            (646.0, 492.0),  # C2 Neutral Cervical Neck
            (674.0, 578.0),  # C7 Base of Neck
            (712.0, 695.0),  # T6 Thoracic Spine (neutral)
            (710.0, 835.0),  # L1 Upper Waist
            (698.0, 965.0),  # L5 Lower Lumbar Waist
        ]
        # In Step 2, spine is neutral/slightly loose; in Steps 3 & 4, emerald locked core!
        sp_cort = (210.0, 230.0, 245.0) if step_idx == 2 else col_spine_cort
        sp_trab = (110.0, 145.0, 175.0) if step_idx == 2 else col_spine_trab
        draw_3d_vertebrae_chain(bone_layer, xray_win, spine_pts, n_vertebrae=14, r_width=15.0, r_height=9.0,
                                col_cortical=sp_cort, col_trabecular=sp_trab, opacity=0.72)

        hip_pt = (692.0, 1025.0)
        knee_pt = (668.0, 1345.0)
        ankle_pt = (688.0, 1635.0)
        draw_3d_joint_epiphysis(bone_layer, xray_win, hip_pt, 21.0, sp_cort, sp_trab, opacity=0.68)
        draw_volumetric_3d_bone_segment(bone_layer, xray_win, hip_pt, knee_pt, 17.0, 10.0, 19.0, sp_cort, sp_trab, opacity=0.62)
        draw_3d_joint_epiphysis(bone_layer, xray_win, knee_pt, 19.0, sp_cort, sp_trab, opacity=0.74)
        draw_volumetric_3d_bone_segment(bone_layer, xray_win, knee_pt, ankle_pt, 16.0, 9.0, 13.0, sp_cort, sp_trab, opacity=0.58)

        # Synchronized Moving Arm Joints & Forearm Dual Bones (Radius + Ulna)
        # Note: We DO NOT draw over the upper-arm muscle belly (y=720..920) so the 3D muscle fibers stay 100% pure!
        sh_pt = (757.0, 712.0)
        if step_idx == 2:
            # f=14: 90° Flexion (Stop Short) -> Forearm horizontal
            el_pt = (728.0, 912.0)
            wr_pt = (472.0, 835.0)
        elif step_idx == 3:
            # f=42: 135° Mid-Extension with Pinned Elbow -> Forearm rotating downward
            el_pt = (722.0, 948.0)
            wr_pt = (646.0, 1168.0)
        else:
            # f=64: 180° Full Lockout + 25° Rope Spread -> Forearm locked vertical & pronated outward
            el_pt = (746.0, 956.0)
            wr_pt = (812.0, 1192.0)

        # 3D X-Ray Shoulder Glenohumeral Socket (at top of acromion) & Elbow Hinge Condyles (at bottom of triceps tendon)
        draw_3d_joint_epiphysis(bone_layer, xray_win, sh_pt, 18.0, col_cort, col_trab, opacity=0.76)
        draw_3d_joint_epiphysis(bone_layer, xray_win, el_pt, 19.0, col_cort, col_trab, opacity=0.88)

        # Dual 3D Forearm Bones: Radius (upper/outer) & Ulna (lower/inner) rotating with the forearm!
        dx_f, dy_f = wr_pt[0] - el_pt[0], wr_pt[1] - el_pt[1]
        flen = max(1.0, math.hypot(dx_f, dy_f))
        nx_f, ny_f = -dy_f / flen, dx_f / flen

        rad_p0 = (el_pt[0] + nx_f * 5.5, el_pt[1] + ny_f * 5.5)
        rad_p1 = (wr_pt[0] + nx_f * 6.0, wr_pt[1] + ny_f * 6.0)
        uln_p0 = (el_pt[0] - nx_f * 5.5, el_pt[1] - ny_f * 5.5)
        uln_p1 = (wr_pt[0] - nx_f * 5.0, wr_pt[1] - ny_f * 5.0)

        draw_volumetric_3d_bone_segment(bone_layer, xray_win, rad_p0, rad_p1, 13.0, 6.8, 12.5, col_cort, col_trab, opacity=0.86)
        draw_volumetric_3d_bone_segment(bone_layer, xray_win, uln_p0, uln_p1, 14.0, 6.5, 10.5, col_cort, col_trab, opacity=0.84)
        draw_3d_joint_epiphysis(bone_layer, xray_win, wr_pt, 14.0, col_cort, col_trab, opacity=0.86)

    # Mask out the active upper-arm muscle fibers from xray_win so the 3D muscle fibers stay 100% vibrant!
    if step_idx != 1:
        muscle_protect = np.clip(fe * 1.4, 0.0, 1.0)
        xray_win = xray_win * (1.0 - muscle_protect * 0.88)
        bone_layer = bone_layer * (1.0 - (muscle_protect * 0.85)[:, :, None])

    # Composite X-Ray cavity darkening + additive/screen 3D bone layer
    out = out * (1.0 - xray_win[:, :, None] * 0.45) + bone_layer
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def render_v32_fullbody_exterior_callouts(canvas: Image.Image, step_idx: int, map_pt, is_wrong: bool):
    """
    Combines v30's exterior posture & forearm arcs with full-body Neck & Knee/Stance alignment callouts!
    """
    v30.render_zero_occlusion_exterior_graphics(canvas, step_idx, map_pt, is_wrong)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    col_main = v30.COL_RED if is_wrong else v30.COL_GREEN
    col_soft = v30.COL_RED_SOFT if is_wrong else v30.COL_GREEN_SOFT

    if step_idx == 1:
        # Neck forward crane badge at upper right
        p_nk = map_pt(805, 495)
        v30.draw_floating_number_badge(od, sd, (int(p_nk[0]), int(p_nk[1])), "28°", "NECK CRANE", col_main, col_soft, align="left")
    elif step_idx == 2:
        p_nk = map_pt(805, 485)
        v30.draw_floating_number_badge(od, sd, (int(p_nk[0]), int(p_nk[1])), "0°", "NECK ALIGN", (180, 230, 255), (200, 240, 255), align="left")
    elif step_idx in (3, 4):
        # Lower-body Knee Stance Stability badge on lower right (outside leg silhouette)
        p_kn = map_pt(815, 1315)
        v30.draw_floating_number_badge(od, sd, (int(p_kn[0]), int(p_kn[1])), "165°", "KNEE BASE", col_main, col_soft, align="left")

    from PIL import ImageFilter
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=9)))
    canvas.alpha_composite(ov)


def main():
    raw_w45 = v25.read_specific_frame(eng.WRONG_I2V, 45)
    raw_c14 = v25.read_specific_frame(eng.CORRECT_I2V, 14)
    raw_c42 = v25.read_specific_frame(eng.CORRECT_I2V, 42)
    raw_c64 = v25.read_specific_frame(eng.CORRECT_I2V, 64)

    s1_rgb = render_v32_hybrid_biomechanics_frame(raw_w45, 1, "xray", True)
    s2_rgb = render_v32_hybrid_biomechanics_frame(raw_c14, 2, "anterior_deltoid", True)
    s3_rgb = render_v32_hybrid_biomechanics_frame(raw_c42, 3, "triceps_lat_long", False)
    s4_rgb = render_v32_hybrid_biomechanics_frame(raw_c64, 4, "triceps_all_three", False)

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

    # Notice: Full-body 3/4 camera framing (cam_zoom = 1.02 ~ 1.06) so Neck, Spine/Waist, Pelvis, Arm Bones, AND Legs/Knees are all visible!
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

    frames_v32 = []
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
        render_v32_fullbody_exterior_callouts(im, sp["idx"], map_pt, sp["is_wrong"])
        v28.draw_luxury_top_dock(im, torso_icons, sp["active_slot"], sp["is_wrong"])
        v30.draw_v30_graphic_telemetry_bottom_left(im, sp, mode="A")

        p_out = os.path.join(OUT_DIR, f"v32_step{sp['idx']}.jpg")
        im.convert("RGB").save(p_out, quality=95)
        frames_v32.append(im.convert("RGB"))

    # Build 4-step master board
    pw, ph = 420, 746
    pad = 24
    board = Image.new("RGB", (pw * 4 + pad * 5, ph + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_row = v30.f_pretendard(21, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "v32 MASTER  ·  (1) 3D 근육결 가닥 자체 발광 + (2) 동작 연동 팔 뼈(요골·척골·관절) 회전 + (3) 목·허리·골반·무릎 전신 3D 척추·골격 정렬",
        font=f_row,
        fill=v30.COL_GREEN,
    )

    col_titles = [
        "STEP 1 · +28° NECK / 35° HUNCH (✕)",
        "STEP 2 · 90° FOREARM STOP / DELT (✕)",
        "STEP 3 · 15° SPINE / 135° RADIUS·ULNA (✓)",
        "STEP 4 · 180° LOCKOUT / 165° KNEE BASE (✓)",
    ]
    for i in range(4):
        x = pad + i * (pw + pad)
        col = v30.COL_RED if i < 2 else v30.COL_GREEN
        bd.text((x + 4, 58), col_titles[i], font=f_col, fill=col)
        ra = frames_v32[i].resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(ra).rectangle([0, 0, pw - 1, ph - 1], outline=(255, 255, 255, 45), width=2)
        board.paste(ra, (x, 86))

    sheet_path = os.path.join(OUT_DIR, "day1_v32_fullbody_fiber_bone_sheet.jpg")
    board.save(sheet_path, quality=94)
    print("Saved:", sheet_path)

    # Also build a high-res zoomed detail comparison sheet (OLD v31 Flat Paint vs NEW v32 Fiber-Strand + Moving Bones + Spine)
    cw, ch = 460, 620
    c_box = (480, 480, 940, 1100)
    comp = Image.new("RGB", (cw * 4 + 50, ch + 95), (10, 12, 16))
    cd = ImageDraw.Draw(comp)
    cd.text(
        (24, 22),
        "CLOSE-UP ANATOMICAL QC  ·  3D 근섬유 결 가닥 내재 컬러링 + 경추·흉요추 3D 척추 마디 + 전완골(요골·척골) 관절 회전",
        font=f_row,
        fill=(245, 247, 252),
    )
    raw_steps = [s1_rgb, s2_rgb, s3_rgb, s4_rgb]
    sub_labels = [
        "1. 35° 굽은 척추 + 어깨·상완골 X-Ray 과부하",
        "2. 삼각근 근섬유 결 발현 + 90° 전완골 멈춤",
        "3. 15° 척추 정렬 + 외측·장두 근섬유 결 + 135° 전완골",
        "4. 3두 전체 근섬유 결 수축 + 180° 요골·척골 완전 신전",
    ]
    for i in range(4):
        p = Image.fromarray(raw_steps[i]).crop(c_box)
        pd = ImageDraw.Draw(p)
        pd.rectangle([0, 0, cw - 1, ch - 1], outline=(50, 58, 72), width=2)
        pd.rectangle([10, 10, cw - 10, 44], fill=(12, 14, 18))
        pd.text((18, 16), sub_labels[i], font=v30.f_pretendard(15, "bold"), fill=v30.COL_RED if i < 2 else v30.COL_GREEN)
        comp.paste(p, (10 + i * (cw + 10), 80))

    comp_path = os.path.join(OUT_DIR, "day1_v32_closeup_fiber_and_bone_qc.jpg")
    comp.save(comp_path, quality=94)
    print("Saved:", comp_path)


if __name__ == "__main__":
    main()
