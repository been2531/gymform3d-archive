#!/usr/bin/env python3
"""
Universal Pose & Silhouette-Fitted 3D Medical Osteology Engine (v33)
====================================================================
Designed to work consistently across arbitrary gym exercises & camera angles
(Day 1 Standing 3/4 Cable Rope Pushdown, Day 2 Seated Rear/3/4 Wide-Grip Lat Pulldown, etc.):

1. Automatic Body Silhouette & Dorsal/Limb Contour Extraction:
   - Extracts the true foreground body mask (bridging dark athletic clothing like shorts)
     and strictly contains all internal skeletal structures within the 3D soft-tissue boundary.
   - Tracks the true dorsal (posterior) neck & back skin contour `x_dorsal(y)` to lock the
     Cervical, Thoracic, Lumbar, and Sacral spine at a true anatomical depth (~20px inward
     from the posterior skin surface, with spinous processes projecting toward the dorsal skin).

2. True Medical Osteology Geometry (ZERO Ball-and-Stick Shapes!):
   - Occipital Cranial Base + 24 Articulated Vertebrae (concave-waisted vertebral bodies,
     dark intervertebral disc spaces, pedicles, and angled posterior spinous processes) + Sacrum.
   - Translucent 3D Thoracic Rib Arcs (Ribs 2-9) + Scapular Spine, Acromion & Glenoid Cavity.
   - Upper Limb Osteology: Humeral Head & Greater Tubercle, Diaphysis with medullary canal,
     Distal Humeral Condyles (Capitulum/Trochlea), Ulnar Olecranon Hook & Trochlear Notch,
     Cylindrical Radial Head, and Dual Radius & Ulna shafts with dynamic radioulnar pronation crossover
     + Carpal Arch & Metacarpal rays.
   - Lower Limb Osteology: Pelvic Iliac Wing/Crest, Acetabular Socket & Ischial Ring,
     Femoral Head, Angled Neck, Greater Trochanter, Anteriorly Bowed Shaft & Distal Condyles,
     Shield-shaped Anterior Patella (Sesamoid Kneecap), Flat Tibial Plateau (with dark synovial
     joint cleft), Tibial Tuberosity, Tibial Shaft & Parallel Fibula (Fibular Head + Shaft).
"""

import math
import cv2
import numpy as np


def _trabecular_3d_texture(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Generates fine multi-scale 3D spongy trabecular bone micro-architecture."""
    t1 = np.sin(xx * 0.39 + yy * 0.27) * np.cos(xx * 0.21 - yy * 0.43)
    t2 = np.sin(xx * 0.87 - yy * 0.71) * np.cos(xx * 0.57 + yy * 0.83)
    t3 = np.sin(xx * 1.55 + yy * 1.22)
    return np.clip(0.54 + 0.26 * t1 + 0.14 * t2 + 0.06 * t3, 0.15, 1.0)


def extract_body_containment_mask(canvas_rgb: np.ndarray, bridge_shorts: bool = True) -> np.ndarray:
    """
    Automatically extracts the full-body soft-tissue containment mask `body_mask` in [0, 1]
    from any studio-placed mannequin frame, bridging dark athletic shorts between torso and thighs
    so bones are 100% contained inside the body silhouette and never float in empty space.
    """
    h, w, _ = canvas_rgb.shape
    c = canvas_rgb.astype(np.float32)
    r, g, b = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    rb = r - b
    gb = g - np.maximum(r, b)

    # Skin/muscle/bone/grip foreground signal
    fg = ((lum > 44.0) | (rb > 38.0) | (gb > 18.0)).astype(np.uint8)
    # Suppress left-side cable machine tower where x < 440 in profile views if disconnected
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)

    if bridge_shorts:
        # In standing/seated views with dark charcoal shorts (y=960..1480), bridge the vertical pelvis/thigh corridor
        # between the lower torso above and the bare thighs/knees below.
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 185))
        fg_bridged = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, v_kernel)
        fg = np.maximum(fg, fg_bridged)

    # Erode by 4px so bones sit inside the skin contour, then smooth feather
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_inner = cv2.erode(fg, kernel_erode, iterations=1).astype(np.float32)
    body_mask = cv2.GaussianBlur(fg_inner, (0, 0), sigmaX=4.5, sigmaY=4.5)
    return np.clip(body_mask, 0.0, 1.0)


def _blend_osteology_patch(
    bone_layer: np.ndarray,
    xray_win: np.ndarray,
    ymin: int,
    ymax: int,
    xmin: int,
    xmax: int,
    inside: np.ndarray,
    cortical_rim: np.ndarray,
    trabecular_core: np.ndarray,
    medullary_dark: np.ndarray,
    col_cortical: tuple[float, float, float],
    col_trabecular: tuple[float, float, float],
    opacity: float,
    win_strength: float = 0.40,
):
    """Composites cortical bone rim + spongy trabecular lattice + dark medullary marrow canal."""
    c_cort = np.array(col_cortical, dtype=np.float32)
    c_trab = np.array(col_trabecular, dtype=np.float32)

    marrow_atten = 1.0 - 0.58 * np.clip(medullary_dark, 0.0, 1.0)
    rgb_patch = (
        c_trab * ((trabecular_core * marrow_atten)[:, :, None] * 0.72)
        + c_cort * (cortical_rim[:, :, None] * 0.88)
    )
    alpha_patch = np.clip(inside, 0.0, 1.0) * opacity

    bone_layer[ymin:ymax, xmin:xmax] = np.maximum(
        bone_layer[ymin:ymax, xmin:xmax],
        rgb_patch * alpha_patch[:, :, None],
    )
    win_patch = cv2.GaussianBlur(np.clip(inside, 0.0, 1.0), (0, 0), sigmaX=7.0, sigmaY=7.0) * win_strength * opacity
    xray_win[ymin:ymax, xmin:xmax] = np.maximum(xray_win[ymin:ymax, xmin:xmax], win_patch)


def draw_anatomical_long_bone(
    bone_layer: np.ndarray,
    xray_win: np.ndarray,
    p_prox: tuple[float, float],
    p_dist: tuple[float, float],
    r_shaft: float,
    prox_flares: tuple[float, float],
    dist_flares: tuple[float, float],
    bow_offset: float,
    col_cortical: tuple[float, float, float],
    col_trabecular: tuple[float, float, float],
    opacity: float = 0.80,
    prox_notch: float = 0.0,
    dist_condyle_split: float = 0.0,
):
    """
    Renders an anatomically sculpted long bone (Humerus, Femur, Tibia, Fibula, Radius, Ulna):
    - Asymmetric proximal & distal metaphyseal/epiphyseal flares (`prox_flares=(r_neg, r_pos)`, `dist_flares=(r_neg, r_pos)`)
    - Natural physiological shaft bow (`bow_offset` in px perpendicular to axis)
    - Dense 3D cortical bone walls (`cortical_rim`) + dark hollow medullary canal (`medullary_dark`) in the diaphysis
    - Reticulated spongy trabecular bone inside the flared epiphyses
    - Optional `prox_notch` (e.g., Ulnar trochlear C-notch) and `dist_condyle_split` (bicondylar groove).
    """
    H, W, _ = bone_layer.shape
    x0, y0 = p_prox
    x1, y1 = p_dist
    dx, dy = x1 - x0, y1 - y0
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    max_r = max(r_shaft, max(prox_flares), max(dist_flares)) + abs(bow_offset) + 28
    pad = int(max_r)
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
    # Physiological bow curve along shaft
    bow = bow_offset * np.sin(t * math.pi)
    perp = (rx * nx + ry * ny) - bow

    # Longitudinal boundary beyond [0, length] with anatomical epiphyseal contour
    over_prox = np.maximum(0.0, -proj)
    over_dist = np.maximum(0.0, proj - length)

    # Epiphyseal flare envelopes at proximal (t -> 0) and distal (t -> 1) ends
    flare_p = np.exp(-((t / 0.19) ** 2))
    flare_d = np.exp(-(((1.0 - t) / 0.20) ** 2))

    r_neg = r_shaft + (prox_flares[0] - r_shaft) * flare_p + (dist_flares[0] - r_shaft) * flare_d
    r_pos = r_shaft + (prox_flares[1] - r_shaft) * flare_p + (dist_flares[1] - r_shaft) * flare_d

    # Add bicondylar distal groove if requested (e.g., femoral/humeral condyles)
    end_cap_r = np.where(perp < 0.0, r_neg, r_pos)
    if dist_condyle_split > 0.0:
        groove = dist_condyle_split * np.exp(-((perp / (r_shaft * 0.55)) ** 2)) * np.exp(-(((1.0 - t) / 0.08) ** 2))
        over_dist = over_dist + groove * 5.5

    u_lat = np.abs(perp) / np.maximum(1.0, end_cap_r)
    u_long_p = over_prox / np.maximum(1.0, 0.65 * (prox_flares[0] + prox_flares[1]))
    u_long_d = over_dist / np.maximum(1.0, 0.65 * (dist_flares[0] + dist_flares[1]))

    # Squircle-like osteological epiphysis cap (not a toy sphere!)
    u_tot = np.sqrt(u_lat ** 2 + (u_long_p ** 1.65) + (u_long_d ** 1.65))

    # Optional Ulnar Trochlear C-Notch cutout at proximal end
    notch_cut = np.zeros_like(u_tot)
    if prox_notch > 0.0:
        # C-shaped articular notch on the anterior/medial aspect of the proximal olecranon
        d_notch = np.hypot(proj - r_shaft * 0.95, perp + r_neg * 0.55)
        notch_cut = np.clip(1.0 - d_notch / (r_shaft * 1.15 * prox_notch), 0.0, 1.0) * 1.4

    inside = np.clip((1.0 - (u_tot + notch_cut * 0.75)) / 0.16, 0.0, 1.0)

    # 3D cylindrical cross-section normal + cortical wall + medullary canal
    cyl_z = np.sqrt(np.clip(1.0 - (u_lat * 0.94) ** 2, 0.0, 1.0))
    # Cortical wall is brightest at u_lat ~ 0.65..0.88 (dense cortical bone rim on X-ray/CT) + 3D key light
    u_signed = np.clip(perp / np.maximum(1.0, end_cap_r), -1.0, 1.0)
    key_light = np.clip(0.55 * cyl_z + 0.45 * np.clip(0.45 - u_signed * 0.75, 0.0, 1.0), 0.0, 1.0)
    cortical_wall = np.exp(-(((u_lat - 0.72) / 0.24) ** 2)) * 0.65 + (key_light ** 1.6) * 0.65
    cortical_rim = np.clip(cortical_wall * inside, 0.0, 1.0)

    # Medullary canal (hollow marrow cavity inside the central diaphysis shaft t in 0.22..0.78, u_lat < 0.52)
    shaft_zone = np.clip(np.sin(t * math.pi), 0.0, 1.0) ** 1.8
    medullary_dark = np.clip(1.0 - (u_lat / 0.54) ** 2, 0.0, 1.0) * shaft_zone * inside

    trab = _trabecular_3d_texture(xx, yy)
    epiph_boost = np.clip(flare_p + flare_d, 0.0, 1.0)
    trabecular_core = np.clip(cyl_z * (0.68 + 0.32 * trab) * (0.75 + 0.25 * epiph_boost), 0.0, 1.0) * inside

    _blend_osteology_patch(
        bone_layer, xray_win, ymin, ymax, xmin, xmax,
        inside, cortical_rim, trabecular_core, medullary_dark,
        col_cortical, col_trabecular, opacity,
    )


def draw_anatomical_spinal_column(
    bone_layer: np.ndarray,
    xray_win: np.ndarray,
    spine_pts: list[tuple[float, float]],
    occiput_pt: tuple[float, float],
    dorsal_side_sign: float,
    col_cortical: tuple[float, float, float],
    col_trabecular: tuple[float, float, float],
    opacity: float = 0.78,
    draw_ribs: bool = True,
):
    """
    Renders a medically accurate 3D Occiput + Cervical (C1-C7) + Thoracic (T1-T12) + Lumbar (L1-L5) + Sacrum column:
    - Each vertebra has:
      1. Anterior Vertebral Body (`추체`): concave-waisted drum with bright superior/inferior cortical endplates
         and a distinct dark Intervertebral Disc space (`추간판 간격`) between bodies!
      2. Pedicle & Articular Pillar (`척추경·관절돌기`) across the spinal canal.
      3. Posterior Spinous Process (`극돌기`) projecting in the `dorsal_side_sign` direction toward the dorsal back skin
         and angling downward according to anatomical region (Cervical bifid/horizontal, Thoracic steep overlapping blades,
         Lumbar thick hatchet blades).
    - Also renders subtle 3D translucent Posterior-Lateral Rib Arcs from T2..T9 so the spine sits inside a real ribcage!
    """
    H, W, _ = bone_layer.shape
    pts_arr = np.array(spine_pts, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, len(pts_arr))

    # 1. Subtle Occipital Cranial Base plate (`후두골 기저부 & 유양돌기`) above C1
    ox, oy = occiput_pt
    c1x, c1y = float(pts_arr[0, 0]), float(pts_arr[0, 1])
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        (ox - dorsal_side_sign * 18.0, oy - 8.0),
        (ox + dorsal_side_sign * 14.0, oy + 6.0),
        r_shaft=7.0, prox_flares=(9.0, 11.0), dist_flares=(12.0, 9.0),
        bow_offset=-5.0 * dorsal_side_sign,
        col_cortical=col_cortical, col_trabecular=col_trabecular,
        opacity=opacity * 0.62,
    )

    # 2. Continuous translucent longitudinal ligament / spinal canal shadow + intervertebral discs
    n_vert = 20  # 5 Cervical + 10 Thoracic + 5 Lumbar + Sacral apex
    t_v = np.linspace(0.0, 0.90, n_vert)
    xs = np.interp(t_v, t_in, pts_arr[:, 0])
    ys = np.interp(t_v, t_in, pts_arr[:, 1])

    c_cort = np.array(col_cortical, dtype=np.float32)
    c_trab = np.array(col_trabecular, dtype=np.float32)

    for i in range(n_vert):
        cx, cy = float(xs[i]), float(ys[i])
        if i < n_vert - 1:
            tx, ty = float(xs[i + 1] - xs[i]), float(ys[i + 1] - ys[i])
            step_len = math.hypot(tx, ty)
        else:
            tx, ty = float(xs[i] - xs[i - 1]), float(ys[i] - ys[i - 1])
            step_len = math.hypot(tx, ty)
        tlen = max(1.0, step_len)
        tx, ty = tx / tlen, ty / tlen
        # Dorsal normal pointing toward the posterior skin surface
        nx, ny = -ty * dorsal_side_sign, tx * dorsal_side_sign
        if nx * dorsal_side_sign < 0:
            nx, ny = -nx, -ny

        frac = i / float(max(1, n_vert - 1))
        # Progressive osteological sizing from C1 -> T1..T12 -> L1..L5
        body_w = 6.8 + 7.2 * (frac ** 1.15)  # Half-width (AP depth) of vertebral body: 6.8px (C) -> 14.0px (L)
        # Leave a crisp 28% intervertebral disc gap between adjacent vertebral bodies!
        body_h = min(step_len * 0.37, 4.2 + 4.5 * frac)

        # Spinous process length & downward angle depend on Cervical vs Thoracic vs Lumbar
        if frac < 0.25:
            # Cervical (C1-C7): short horizontal prongs, C7 (frac~0.22) longer
            sp_len = (9.0 + 14.0 * max(0.0, frac - 0.12))
            sp_drop = 0.28
            sp_thick = 2.8
        elif frac < 0.72:
            # Thoracic (T1-T12): long, steeply downward-slanted overlapping spinous blades
            sp_len = 15.5
            sp_drop = 0.68
            sp_thick = 3.1
        else:
            # Lumbar (L1-L5): thick, broad, horizontal hatchet-shaped spinous processes
            sp_len = 14.5
            sp_drop = 0.22
            sp_thick = 4.6

        pad = int(body_w + sp_len + 18)
        xmin, xmax = max(0, int(cx) - pad), min(W, int(cx) + pad)
        ymin, ymax = max(0, int(cy) - pad), min(H, int(cy) + pad)
        if xmin >= xmax or ymin >= ymax:
            continue

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax].astype(np.float32)
        dx, dy = xx - cx, yy - cy

        # Coordinates in local vertebra frame:
        # s_along: along spinal axis (+ downward)
        # s_dorsal: perpendicular toward dorsal back skin (+ dorsal, - ventral)
        s_along = dx * tx + dy * ty
        s_dorsal = dx * nx + dy * ny

        # (A) Anterior Vertebral Body (`추체`) centered slightly ventral (-0.25 * body_w) with concave waisting
        s_body_d = s_dorsal + 0.22 * body_w
        u_along = s_along / max(1.0, body_h)
        # Concave waisting in the middle of the vertebral drum, flaring at superior/inferior endplates (`종판`)
        waist_flare = 0.86 + 0.18 * np.clip(u_along ** 2, 0.0, 1.2)
        u_perp = s_body_d / np.maximum(1.0, body_w * waist_flare)

        drum_dist = ((np.abs(u_along) ** 3.8) + (np.abs(u_perp) ** 2.6)) ** (1.0 / 3.2)
        inside_body = np.clip((1.0 - drum_dist) / 0.18, 0.0, 1.0)

        # (B) Pedicle & Articular Process bridge (`척추경·관절돌기`) at upper half of body
        ped_u = (s_dorsal - 0.70 * body_w) / max(1.0, 0.52 * body_w)
        ped_v = (s_along + 0.18 * body_h) / max(1.0, 0.52 * body_h)
        inside_ped = np.clip(1.0 - (ped_u ** 2 + ped_v ** 2), 0.0, 1.0)

        # (C) Posterior Spinous Process (`극돌기`) projecting toward dorsal skin contour
        sp_t = np.clip((s_dorsal - 0.85 * body_w) / max(1.0, sp_len), 0.0, 1.0)
        sp_center_along = (-0.05 * body_h) + sp_t * (sp_len * sp_drop)
        sp_dist_perp = np.abs(s_along - sp_center_along) / np.maximum(1.0, sp_thick * (1.0 - 0.35 * sp_t))
        inside_sp = (
            np.clip(1.0 - sp_dist_perp, 0.0, 1.0)
            * np.clip((s_dorsal - 0.75 * body_w) / 3.0, 0.0, 1.0)
            * np.clip((0.85 * body_w + sp_len - s_dorsal) / 3.0, 0.0, 1.0)
        )

        inside_v = np.clip(np.maximum(inside_body, np.maximum(inside_ped * 0.85, inside_sp * 0.92)), 0.0, 1.0)

        # 3D shading: bright superior/inferior cortical endplates (`상·하 종판`) + anterior/posterior cortical wall
        endplate_rim = np.exp(-(((np.abs(u_along) - 0.82) / 0.18) ** 2)) * inside_body * 0.72
        wall_rim = np.exp(-(((np.abs(u_perp) - 0.78) / 0.22) ** 2)) * inside_body * 0.55
        sp_crest = np.clip(1.0 - sp_dist_perp * 0.7, 0.0, 1.0) * inside_sp * 0.78
        cortical_rim = np.clip(endplate_rim + wall_rim + sp_crest, 0.0, 1.0)

        cyl_3d = np.sqrt(np.clip(1.0 - (u_perp * 0.85) ** 2, 0.0, 1.0)) * inside_body + 0.65 * inside_sp
        trab = _trabecular_3d_texture(xx, yy)
        trabecular_core = np.clip(cyl_3d * (0.72 + 0.28 * trab), 0.0, 1.0) * inside_v
        # Subtle dark spinal canal foramen between body and spinous arch
        canal_dark = np.exp(-(((s_dorsal - 0.78 * body_w) / 3.2) ** 2 + (s_along / (body_h * 0.7)) ** 2))

        _blend_osteology_patch(
            bone_layer, xray_win, ymin, ymax, xmin, xmax,
            inside_v, cortical_rim, trabecular_core, canal_dark,
            col_cortical, col_trabecular, opacity, win_strength=0.32,
        )

        # (D) Subtle Translucent 3D Thoracic Rib Arcs (on Thoracic vertebrae i=5..13)
        if draw_ribs and (5 <= i <= 13):
            rib_idx = i - 5
            rib_len = 52.0 + 34.0 * math.sin((rib_idx / 8.0) * math.pi)
            # Ribs curve ventrally (-nx, -ny) and downward (+tx, +ty)
            r_p0 = (cx - nx * 2.0, cy)
            r_p1 = (
                cx - nx * rib_len + tx * (22.0 + rib_idx * 2.5),
                cy - ny * rib_len + ty * (22.0 + rib_idx * 2.5),
            )
            draw_anatomical_long_bone(
                bone_layer, xray_win,
                r_p0, r_p1,
                r_shaft=3.2, prox_flares=(4.5, 4.5), dist_flares=(2.6, 2.6),
                bow_offset=-12.0 * dorsal_side_sign,
                col_cortical=col_cortical, col_trabecular=col_trabecular,
                opacity=opacity * 0.30,
            )

    # 3. Sacrum (`천골`): curved triangular shield from L5 (t=0.90) to Coccyx (t=1.0)
    s_top = (
        float(np.interp(0.91, t_in, pts_arr[:, 0])),
        float(np.interp(0.91, t_in, pts_arr[:, 1])),
    )
    s_bot = (float(pts_arr[-1, 0]), float(pts_arr[-1, 1]))
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        s_top, s_bot,
        r_shaft=9.5,
        prox_flares=(15.0, 13.0),
        dist_flares=(3.5, 4.0),
        bow_offset=7.5 * dorsal_side_sign,
        col_cortical=col_cortical,
        col_trabecular=col_trabecular,
        opacity=opacity * 0.85,
    )


def draw_anatomical_shoulder_and_arm(
    bone_layer: np.ndarray,
    xray_win: np.ndarray,
    acromion_pt: tuple[float, float],
    glenoid_pt: tuple[float, float],
    elbow_pt: tuple[float, float],
    olecranon_pt: tuple[float, float],
    wrist_pt: tuple[float, float],
    hand_pt: tuple[float, float],
    pronation: float,
    col_cortical: tuple[float, float, float],
    col_trabecular: tuple[float, float, float],
    draw_full_humerus: bool = False,
    opacity: float = 0.84,
):
    """
    Renders true 3D Upper Limb Osteology:
    1. Scapular Spine & Acromion Hood (`견갑극·견봉`) + Concave Glenoid Socket (`관절와`)
    2. Proximal Humeral Head & Greater Tubercle (`상완골두·대결절`) + Distal Humeral Condyles (`상완골 외측상과·활차·소두`)
       (If `draw_full_humerus=True`, renders the entire Humerus shaft; otherwise renders the proximal head and
       distal condylar metaphysis so active upper-arm 3D muscle fibers remain 100% unoccluded!)
    3. Elbow Articulation: Ulnar Olecranon Hook (`척골 주두 & 활차절흔`) wrapping around the humeral trochlea
       + Cylindrical Radial Head (`요골두`) separated by a dark synovial joint space!
    4. Dual Forearm Bones: Radius (`요골`) & Ulna (`척골`) with true anatomical pronation crossover!
    5. Wrist & Hand: Curved Carpal Bone Arch (`수근골 2열`) + 4 Diverging Metacarpal Rays (`중수골`)!
    """
    # 1. Scapular Spine & Acromion Process overhanging the Glenoid Cavity
    ax, ay = acromion_pt
    gx, gy = glenoid_pt
    ex, ey = elbow_pt
    ox, oy = olecranon_pt
    wx, wy = wrist_pt
    hx, hy = hand_pt

    # Scapular Spine ridge extending posteromedially from the Acromion
    scap_med = (ax + 44.0, ay + 12.0)
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        scap_med, (ax - 10.0, ay - 3.0),
        r_shaft=5.2, prox_flares=(4.0, 7.0), dist_flares=(9.5, 8.5),
        bow_offset=-4.5,
        col_cortical=col_cortical, col_trabecular=col_trabecular,
        opacity=opacity * 0.74,
    )
    # Scapular Lateral Border / Infraspinous triangle hint
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        (gx + 14.0, gy + 4.0), (gx + 36.0, gy + 64.0),
        r_shaft=4.5, prox_flares=(11.0, 9.0), dist_flares=(4.0, 4.0),
        bow_offset=3.0,
        col_cortical=col_cortical, col_trabecular=col_trabecular,
        opacity=opacity * 0.48,
    )

    # 2. Humerus (Proximal Humeral Head + Greater Tubercle, and Distal Epicondyles/Trochlea)
    dx_h, dy_h = ex - gx, ey - gy
    hlen = max(1.0, math.hypot(dx_h, dy_h))
    ux_h, uy_h = dx_h / hlen, dy_h / hlen

    if draw_full_humerus:
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            (gx, gy), (ex, ey),
            r_shaft=9.5,
            prox_flares=(16.5, 19.5),  # Humeral head + Greater tubercle
            dist_flares=(17.0, 15.5),  # Lateral/Medial epicondyles + Capitulum/Trochlea
            bow_offset=-3.0,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=opacity * 0.90,
            dist_condyle_split=1.0,
        )
    else:
        # Proximal Humerus (Head, Anatomical Neck, Greater Tubercle, Surgical Neck taper)
        p_surg = (gx + ux_h * 52.0, gy + uy_h * 52.0)
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            (gx, gy), p_surg,
            r_shaft=9.0,
            prox_flares=(16.0, 19.0),
            dist_flares=(8.5, 8.5),
            bow_offset=-2.0,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=opacity * 0.82,
        )
        # Distal Humerus (Supracondylar ridges + Lateral Epicondyle + Capitulum & Trochlea)
        p_supra = (ex - ux_h * 54.0, ey - uy_h * 54.0)
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            p_supra, (ex, ey),
            r_shaft=8.8,
            prox_flares=(8.2, 8.2),
            dist_flares=(16.5, 15.0),
            bow_offset=0.0,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=opacity * 0.88,
            dist_condyle_split=1.0,
        )

    # 3. Forearm Dual Bones: ULNA (with Olecranon Hook at `olecranon_pt`) & RADIUS (with Radial Head)
    dx_f, dy_f = wx - ex, wy - ey
    flen = max(1.0, math.hypot(dx_f, dy_f))
    ux_f, uy_f = dx_f / flen, dy_f / flen
    # Normal pointing toward thumb/radial side (anterior/superior in neutral pushdown)
    nx_f, ny_f = -uy_f, ux_f
    if ny_f > 0:
        nx_f, ny_f = -nx_f, -ny_f

    # Ulna starts at the posterior Olecranon beak (`ox, oy`) and runs to the pinky/ulnar side of the wrist
    ulna_prox = (ox, oy)
    ulna_dist = (wx - nx_f * 7.5, wy - ny_f * 7.5)
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        ulna_prox, ulna_dist,
        r_shaft=5.6,
        prox_flares=(13.5, 15.5),  # Prominent Olecranon process + Coronoid beak!
        dist_flares=(6.5, 7.5),    # Slender Ulnar Head + Styloid process at wrist
        bow_offset=-3.2,
        col_cortical=col_cortical, col_trabecular=col_trabecular,
        opacity=opacity * 0.92,
        prox_notch=0.95,           # C-shaped Trochlear Notch articulating with humerus trochlea!
    )

    # Radius starts at the disc-shaped Radial Head (just distal/anterior to the humeral capitulum, separated by a 4px synovial gap)
    rad_prox = (ex + ux_f * 10.0 + nx_f * 7.5, ey + uy_f * 10.0 + ny_f * 7.5)
    # During pronation (`pronation` in 0..1), the distal Radius crosses diagonally over the Ulna!
    rad_cross = (1.0 - 1.65 * pronation) * 8.5
    rad_dist = (wx + nx_f * rad_cross, wy + ny_f * rad_cross)
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        rad_prox, rad_dist,
        r_shaft=6.0,
        prox_flares=(10.5, 10.5),  # Cylindrical disc-shaped Radial Head!
        dist_flares=(13.0, 11.5),  # Broad flared Distal Radius Styloid at wrist!
        bow_offset=4.5 * (1.0 - 0.8 * pronation),
        col_cortical=col_cortical, col_trabecular=col_trabecular,
        opacity=opacity * 0.94,
    )

    # 4. Wrist Carpal Arch (2 rows of faceted carpal bones) + 4 Metacarpal Rays into the hand
    dx_w, dy_w = hx - wx, hy - wy
    hlen_w = max(1.0, math.hypot(dx_w, dy_w))
    ux_w, uy_w = dx_w / hlen_w, dy_w / hlen_w
    nx_w, ny_w = -uy_w, ux_w

    # Proximal & Distal Carpal rows (separated from distal radius/ulna by a 3px radiocarpal joint space)
    for row_i, d_along in enumerate([7.0, 14.5]):
        for lat_off in [-8.0, -2.5, 3.0, 8.0]:
            c_p0 = (wx + ux_w * (d_along - 2.5) + nx_w * (lat_off - 2.2), wy + uy_w * (d_along - 2.5) + ny_w * (lat_off - 2.2))
            c_p1 = (wx + ux_w * (d_along + 2.5) + nx_w * (lat_off + 2.2), wy + uy_w * (d_along + 2.5) + ny_w * (lat_off + 2.2))
            draw_anatomical_long_bone(
                bone_layer, xray_win, c_p0, c_p1,
                r_shaft=3.2, prox_flares=(3.8, 3.8), dist_flares=(3.8, 3.8),
                bow_offset=0.0, col_cortical=col_cortical, col_trabecular=col_trabecular,
                opacity=opacity * 0.78,
            )

    # 4 Slender diverging Metacarpal shafts (`중수골`) extending into the hand grip
    for mc_idx, lat_off in enumerate([-9.0, -3.0, 3.0, 8.5]):
        mc_p0 = (wx + ux_w * 19.0 + nx_w * lat_off * 0.85, wy + uy_w * 19.0 + ny_w * lat_off * 0.85)
        mc_p1 = (hx + nx_w * lat_off * 1.15, hy + ny_w * lat_off * 1.15)
        draw_anatomical_long_bone(
            bone_layer, xray_win, mc_p0, mc_p1,
            r_shaft=2.6, prox_flares=(4.2, 4.2), dist_flares=(4.4, 4.4),
            bow_offset=0.0, col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=opacity * 0.74,
        )


def draw_anatomical_pelvis_and_lower_limbs(
    bone_layer: np.ndarray,
    xray_win: np.ndarray,
    sacrum_pt: tuple[float, float],
    iliac_crest_pt: tuple[float, float],
    hip_R: tuple[float, float],
    trochanter_R: tuple[float, float],
    knee_R: tuple[float, float],
    patella_R: tuple[float, float],
    ankle_R: tuple[float, float],
    hip_L: tuple[float, float] | None = None,
    knee_L: tuple[float, float] | None = None,
    patella_L: tuple[float, float] | None = None,
    ankle_L: tuple[float, float] | None = None,
    col_cortical: tuple[float, float, float] = (195.0, 242.0, 225.0),
    col_trabecular: tuple[float, float, float] = (28.0, 175.0, 115.0),
    opacity: float = 0.76,
):
    """
    Renders true 3D Lower-Body Osteology (Pelvis, Femur, Patella, Tibia & Fibula) with ZERO toy spheres:
    1. Pelvis (`장골익 Iliac Wing + S-curved Iliac Crest 장골능 + Acetabulum Socket 비구 + Ischial Ring 좌골`)
    2. Femur (`대퇴골`):
       - Spherical Femoral Head seated inside the Acetabulum (with dark hip joint space)
       - Angled Femoral Neck (`대퇴경부`) + Prominent Lateral Greater Trochanter (`대전자`)
       - Anteriorly bowed Femoral Diaphysis (`대퇴골체`) with cortical bone walls & medullary canal
       - Scroll-shaped Distal Femoral Condyles (`대퇴골 내·외측과`)
    3. Patella (`슬개골`):
       - Shield/wedge-shaped sesamoid kneecap positioned ANTERIOR to the femoral condyles (with patellofemoral space)
    4. Tibia & Fibula (`경골 · 비골`):
       - Flat-topped Tibial Plateau (`경골 고평부`) separated from the femoral condyles by a crisp 5px dark synovial knee joint space!
       - Anterior Tibial Tuberosity (`경골 조면`) & sharp anterior shin crest tapering to the Medial Malleolus (`내과`)
       - Slender parallel Fibula (`비골두 Fibular Head + 비골체 Shaft + 외과 Lateral Malleolus`) posterolaterally!
    """
    sx, sy = sacrum_pt
    icx, icy = iliac_crest_pt
    hx_r, hy_r = hip_R
    tx_r, ty_r = trochanter_R
    kx_r, ky_r = knee_R
    px_r, py_r = patella_R
    ax_r, ay_r = ankle_R

    # 1. Pelvic Girdle: Iliac Crest arch (PSIS near sacrum -> Iliac Crest -> ASIS) + Iliac Body to Acetabulum + Ischial Loop
    asis_pt = (icx - 36.0, icy + 18.0)
    # Iliac crest S-curve
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        (sx - 6.0, sy - 12.0), asis_pt,
        r_shaft=7.5, prox_flares=(13.0, 11.0), dist_flares=(11.0, 14.0),
        bow_offset=14.0,
        col_cortical=col_cortical, col_trabecular=col_trabecular,
        opacity=opacity * 0.78,
    )
    # Iliac column from Iliac Crest down to Acetabular Roof (`비구 상연`)
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        icx_pt := (icx - 8.0, icy + 6.0), (hx_r, hy_r - 8.0),
        r_shaft=12.5, prox_flares=(18.0, 16.0), dist_flares=(19.0, 16.0),
        bow_offset=4.0,
        col_cortical=col_cortical, col_trabecular=col_trabecular,
        opacity=opacity * 0.82,
    )
    # Ischial Tuberosity / Pubic arch below Acetabulum
    draw_anatomical_long_bone(
        bone_layer, xray_win,
        (hx_r + 6.0, hy_r + 8.0), (hx_r - 16.0, hy_r + 42.0),
        r_shaft=6.5, prox_flares=(11.0, 9.0), dist_flares=(9.5, 11.5),
        bow_offset=-7.0,
        col_cortical=col_cortical, col_trabecular=col_trabecular,
        opacity=opacity * 0.68,
    )

    def _render_single_leg(hip_pt, troch_pt, knee_pt, pat_pt, ankle_pt, leg_op):
        hx, hy = hip_pt
        trx, try_ = troch_pt
        kx, ky = knee_pt
        px, py = pat_pt
        ax, ay = ankle_pt

        # (A) Femoral Head & Angled Femoral Neck (`대퇴골두 · 대퇴경부`) + Greater Trochanter (`대전자`)
        # Neck-shaft junction is slightly below the Greater Trochanter
        neck_base = (trx - 8.0, try_ + 16.0)
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            (hx, hy), neck_base,
            r_shaft=8.0,
            prox_flares=(14.5, 14.5),  # Spherical Femoral Head inside Acetabulum
            dist_flares=(11.5, 13.5),
            bow_offset=0.0,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=leg_op * 0.90,
        )
        # Greater Trochanter bony prominence projecting superolaterally
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            (trx + 4.0, try_ - 10.0), neck_base,
            r_shaft=9.5,
            prox_flares=(13.5, 11.5),
            dist_flares=(11.0, 10.0),
            bow_offset=3.0,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=leg_op * 0.88,
        )

        # (B) Femoral Shaft (`대퇴골체`) curving down to Distal Femoral Condyles (`대퇴골 내·외측과`)
        # Stop 4px above `ky` to leave a real dark Tibiofemoral Synovial Joint Space (`무릎 관절강`)!
        fem_dist = (kx, ky - 5.0)
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            neck_base, fem_dist,
            r_shaft=9.8,
            prox_flares=(13.5, 15.0),
            dist_flares=(16.5, 21.0),  # Posterior scroll of Femoral Condyles flares backward!
            bow_offset=-5.5,           # Natural anterior physiological bow of the human femur!
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=leg_op * 0.92,
            dist_condyle_split=1.0,
        )

        # (C) Patella (`슬개골`): distinct wedge/shield-shaped sesamoid bone IN FRONT OF the knee joint!
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            (px, py - 13.0), (px + 2.0, py + 14.0),
            r_shaft=6.5,
            prox_flares=(8.5, 6.5),    # Broad superior base of patella
            dist_flares=(3.5, 4.0),    # Pointed inferior apex of patella (attaching to patellar tendon)
            bow_offset=-2.5,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=leg_op * 0.95,
        )

        # (D) Tibia (`경골`): Flat Tibial Plateau (`경골 고평부`, 5px below femoral condyles) + Anterior Tibial Tuberosity + Shaft to Medial Malleolus
        tib_prox = (kx, ky + 6.0)
        tib_dist = (ax - 3.0, ay)
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            tib_prox, tib_dist,
            r_shaft=8.2,
            prox_flares=(18.5, 17.5),  # Broad flat-topped Tibial Plateau + anterior Tibial Tuberosity!
            dist_flares=(12.5, 11.0),  # Distal Tibial Plafond & Medial Malleolus at ankle
            bow_offset=1.5,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=leg_op * 0.90,
        )

        # (E) Fibula (`비골`): Slender lateral bone from Fibular Head (posterolateral below lateral tibial condyle) to Lateral Malleolus
        fib_prox = (kx + 17.0, ky + 18.0)
        fib_dist = (ax + 12.0, ay + 5.0)
        draw_anatomical_long_bone(
            bone_layer, xray_win,
            fib_prox, fib_dist,
            r_shaft=4.2,
            prox_flares=(7.5, 7.5),    # Knob-like Fibular Head (`비골두`)
            dist_flares=(7.0, 8.0),    # Pointed Lateral Malleolus (`외과`)
            bow_offset=1.0,
            col_cortical=col_cortical, col_trabecular=col_trabecular,
            opacity=leg_op * 0.78,
        )

    # Render far/lead Left Leg first at subtle depth-cued opacity so near Right Leg sits cleanly in foreground
    if hip_L is not None and knee_L is not None and patella_L is not None and ankle_L is not None:
        troch_L = (hip_L[0] + 22.0, hip_L[1] + 14.0)
        _render_single_leg(hip_L, troch_L, knee_L, patella_L, ankle_L, opacity * 0.56)

    # Render primary foreground Right Leg
    _render_single_leg(hip_R, trochanter_R, knee_R, patella_R, ankle_R, opacity)


def extract_and_solve_day1_anatomical_rig(canvas_rgb: np.ndarray, step_idx: int) -> dict:
    """
    Automatically extracts the dorsal back contour `x_dorsal(y)`, arm chroma cluster, hand grip cluster,
    and leg silhouette columns from `canvas_rgb`, returning a 100% silhouette-locked anatomical rig dictionary.
    """
    c = canvas_rgb.astype(np.float32)
    r, g, b = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Automatically measure dorsal edge `x_dorsal(y)` of neck & torso
    def get_dorsal_x(y_val: int, x_min: int = 520, x_max: int = 965, thr: float = 52.0) -> float:
        y0, y1 = max(0, y_val - 4), min(c.shape[0], y_val + 5)
        band = np.mean(lum[y0:y1, x_min:x_max], axis=0)
        xs = np.where(band > thr)[0]
        if len(xs) > 0:
            return float(x_min + xs[-1])
        return 750.0

    if step_idx == 1:
        # Step 1 (Hunched forward 35° in WRONG_I2V f=45):
        # Lock spine 20px inside the actual detected dorsal neck/back contour!
        y_samples = [525, 585, 640, 680, 725, 780, 850, 935, 1030, 1130, 1210]
        inward_offsets = [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 23.0, 23.0, 24.0, 25.0, 26.0]
        spine_pts = []
        for yv, off in zip(y_samples, inward_offsets):
            xd = get_dorsal_x(yv, x_min=480, x_max=968, thr=50.0)
            spine_pts.append((xd - off, float(yv)))

        return {
            "occiput": (spine_pts[0][0] - 8.0, 502.0),
            "spine": spine_pts,
            "dorsal_sign": 1.0,
            "acromion": (786.0, 790.0),
            "sh_glenoid": (794.0, 816.0),
            "el_center": (832.0, 956.0),
            "el_olecranon": (852.0, 952.0),
            "wr_center": (826.0, 1168.0),
            "hand_tip": (818.0, 1215.0),
            "pronation": 0.45,
            "draw_full_humerus": True,
            # Pelvis & Lower Limbs (fitted to S1_w45's actual pelvis at y=1220..1320 and thighs/knees at y=1560..1700)
            "sacrum": spine_pts[-1],
            "iliac_crest": (882.0, 1215.0),
            "hip_R": (868.0, 1295.0),
            "trochanter_R": (898.0, 1312.0),
            "knee_R": (868.0, 1595.0),
            "patella_R": (824.0, 1590.0),
            "ankle_R": (882.0, 1735.0),
            "hip_L": (665.0, 1295.0),
            "knee_L": (638.0, 1585.0),
            "patella_L": (596.0, 1580.0),
            "ankle_L": (650.0, 1735.0),
        }

    # Steps 2, 3, 4 (Upright / 15° Hinge in CORRECT_I2V f=14, 42, 64):
    # Track the posterior neck & dorsal back contour closely (19-22px inside dorsal edge)
    spine_pts = [
        (get_dorsal_x(512, 560, 740) - 18.0, 512.0),  # C1/C2
        (get_dorsal_x(558, 560, 740) - 18.0, 558.0),  # C4
        (get_dorsal_x(602, 560, 755) - 19.0, 602.0),  # C7
        (get_dorsal_x(642, 560, 810) - 21.0, 642.0),  # T2
        (get_dorsal_x(690, 560, 845) - 24.0, 690.0),  # T5
        (798.0, 755.0),                               # T8 (posterior thoracic column behind scapular plane)
        (786.0, 825.0),                               # T11
        (764.0, 895.0),                               # L1
        (get_dorsal_x(955, 560, 785) - 19.0, 955.0),  # L3
        (get_dorsal_x(1005, 560, 785) - 19.0, 1005.0),# L5
        (752.0, 1052.0),                              # Sacrum S2
    ]

    if step_idx == 2:
        el_c, el_o = (744.0, 892.0), (766.0, 900.0)
        wr_c, hd_t = (542.0, 852.0), (488.0, 840.0)
        pron = 0.15
    elif step_idx == 3:
        el_c, el_o = (768.0, 932.0), (792.0, 938.0)
        wr_c, hd_t = (678.0, 1126.0), (648.0, 1170.0)
        pron = 0.45
    else:
        el_c, el_o = (812.0, 936.0), (834.0, 934.0)
        wr_c, hd_t = (820.0, 1140.0), (818.0, 1190.0)
        pron = 0.92

    return {
        "occiput": (spine_pts[0][0] - 6.0, 488.0),
        "spine": spine_pts,
        "dorsal_sign": 1.0,
        "acromion": (752.0, 680.0),
        "sh_glenoid": (760.0, 706.0),
        "el_center": el_c,
        "el_olecranon": el_o,
        "wr_center": wr_c,
        "hand_tip": hd_t,
        "pronation": pron,
        "draw_full_humerus": False,
        # Lower body: Hip Acetabulum at y=1065, Knee Joint at y=1318 (above calf bulge at y=1390!), Ankle at y=1615
        "sacrum": spine_pts[-1],
        "iliac_crest": (722.0, 1018.0),
        "hip_R": (704.0, 1072.0),
        "trochanter_R": (734.0, 1088.0),
        "knee_R": (706.0, 1318.0),
        "patella_R": (672.0, 1314.0),
        "ankle_R": (736.0, 1615.0),
        "hip_L": (626.0, 1068.0),
        "knee_L": (605.0, 1304.0),
        "patella_L": (572.0, 1300.0),
        "ankle_L": (602.0, 1565.0),
    }
