#!/usr/bin/env python3
"""
Universal Landmark-Anchored 3D Posture & Anatomy Engine (`universal_posture_anatomy_engine.py`)
===============================================================================================
Implements 중현 님's core architectural mandate (`t201u`, `t203u`, `t204u`, `t206u`, `t207u`, `t208u`)
unified with `@Gym Manager`'s `v62` 2-Stage Landmark-Guided Shader:
1. ZERO Mannequin Ghosting / Afterimage (`마네킹 잔상 0%`):
   - Every frame is shaded 100% independently (`f0_lum` frame-subtraction and `cv2.remap` warping
     are 100% eliminated!).
2. `@Gym Manager` `v62` 2-Stage Landmark-Guided Muscle & Bone Targeting (`근육 타겟팅`):
   - Stage 1 (`Cool-Channel Non-Target Charcoal Reset`):
     Computes non-target matte dark-charcoal skin strictly from `cool_base = np.minimum(g_ch, b_ch)`
     (`grey_charcoal` in `[18, 86]`), completely erasing any off-target AI orange/red glow (such as
     lower-back bleed in Step 1, mid-back bleed in Step 2, lower-back bleed in Step 3, or outer
     latissimus dorsi bleed in Step 4) into seamless `#2A2E35` matte dark-charcoal skin with 0% pale
     grey smudges!
   - Stage 2 (`Hybrid Native + Landmark-Anchored Anatomical Muscle-Belly Field + 3D Fiber Striations`):
     Combines `native_fe * m_gate` with a tight MediaPipe landmark-anchored Gaussian muscle-belly field
     (`anat_belly * body_interior`) modulated by 4-Band Multi-Scale DoG + directional 3D anatomical
     muscle fiber striations (`fs_eff`, `sulc_eff`), so the exact target muscle belly always lights up
     with razor-sharp Day-2 3D muscle fiber bundles even when the raw AI plate had uneven glow!
   - Fixes the side-view `arm_mask` overlap bug: `arm_mask` only masks lateral arm segments outside
     the central neck/spine corridor, NEVER punching holes through the neck or spine in profile views.
"""

import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

import universal_anatomy_rig as uar


# ==============================================================================
# 1. DECLARATIVE ANATOMICAL TARGET SPECIFICATIONS (EP01 .. EP04 & GYM REELS)
# ==============================================================================
# Segment tuple format:
#   (pt_a_key, pt_b_key, t_start, t_end, radius_scale_of_torso_len, lateral_offset_scale, fiber_angle_deg_offset)
ANATOMY_REGION_SPECS = {
    "cervical_flexion_overload": {
        "title": "60° TEXT NECK (C1-C7 Cervical Spine & Posterior Neck / Upper Trap Overload)",
        "is_wrong": True,
        "prompt_positive": (
            "localized translucent ivory 3D cervical spine vertebrae X-ray (C1-C7) inside the neck, and deeply sculpted "
            "glowing orange-red 3D muscle fiber bundles strictly on the posterior neck extensors (splenius capitis, "
            "levator scapulae) and upper trapezius neck ridge"
        ),
        "prompt_negative": (
            "NO skinless ecorche on face, skull, chest, abdomen, arms, or mid/lower back; NO glowing muscles below T2"
        ),
        # Strictly Posterior Cervical Neck (`ear -> sh`) + C7-T2 Upper Trap Neck Collar (NO mid/lower back!)
        "muscle_segments": [
            ("ear", "sh", -0.04, 1.12, 0.23, +0.05),      # Posterior cervical extensors (splenius / semispinalis / levator)
            ("sh", "hip", -0.10, 0.24, 0.24, +0.04),      # C7-T2 upper trapezius neck-shoulder junction only
        ],
        "bone_segments": [
            ("ear", "sh", 0.08, 1.02, 0.12, +0.03),       # C1-C7 Cervical spine vertebrae X-Ray
            ("sh", "hip", -0.02, 0.12, 0.11, +0.02),      # T1-T2 Thoracic junction X-Ray
        ],
    },
    "deep_cervical_chin_tuck": {
        "title": "0° CHIN TUCK (Deep Cervical Flexors, SCM & Cervical Stabilizer Column)",
        "is_wrong": False,
        "prompt_positive": (
            "ZERO bones and ZERO spine X-ray visible; deeply sculpted glowing orange-red 3D muscle fiber bundles strictly "
            "on the cervical neck stabilizers (deep cervical flexors, sternocleidomastoid, posterior cervical column C1-C7 "
            "and upper C7-T2 collar)"
        ),
        "prompt_negative": (
            "NO bones, NO skeleton X-ray anywhere; NO glowing muscles on mid-back, lower back, lats, arms, or head/face"
        ),
        # Strictly the Cervical Neck Column (`ear -> sh` and `nose/jaw -> sh` SCM/deep flexors + C7-T2 collar)!
        # ZERO mid-back (`sh -> hip` stops at 0.14)!
        "muscle_segments": [
            ("ear", "sh", 0.06, 1.04, 0.19, +0.02),       # Posterior & lateral cervical stabilizer column (scalenes, splenius)
            ("nose", "sh", 0.42, 0.98, 0.15, +0.04),      # Anterior deep cervical flexors & sternocleidomastoid (SCM)
            ("sh", "hip", -0.04, 0.14, 0.18, +0.02),      # C7-T2 lower cervical / upper-trap anchor collar
        ],
        "bone_segments": [],  # 0% bone exposure on Right cuts!
    },
    "upper_trap_protraction_overload": {
        "title": "38° ROUNDED SHOULDERS (C7-T5 Spine X-Ray & Bilateral Upper Trapezius Yoke Strain)",
        "is_wrong": True,
        "prompt_positive": (
            "localized translucent ivory 3D cervical-thoracic spine vertebrae X-ray (C7-T5) along the upper midline, and "
            "deeply sculpted glowing orange-red 3D muscle fiber bundles strictly on the shrugged bilateral upper trapezius "
            "collar and levator scapulae"
        ),
        "prompt_negative": (
            "NO skinless ecorche on head, forearms, lower back, lats, or glutes"
        ),
        # Strictly the Upper Trapezius Yoke (`ear -> sh_l`, `ear -> sh_r`, `sh_l -> sh_r`) + C7-T4 upper thoracic apex!
        "muscle_segments": [
            ("ear", "sh", 0.22, 1.02, 0.17, 0.00),        # Bilateral levator scapulae / nuchal upper trap
            ("ear", "sh_l", 0.35, 1.00, 0.16, 0.00),      # Left upper trapezius descending ridge
            ("ear", "sh_r", 0.35, 1.00, 0.16, 0.00),      # Right upper trapezius descending ridge
            ("sh_l", "sh_r", 0.02, 0.98, 0.18, 0.00),     # Bi-acromial upper trapezius yoke across top of shoulders
            ("sh", "hip", -0.05, 0.12, 0.18, 0.00),       # C7-T3 upper thoracic trapezius apex only (NO lower/mid back!)
        ],
        "bone_segments": [
            ("ear", "sh", 0.32, 1.00, 0.10, 0.00),        # C5-C7 lower cervical vertebrae X-Ray
            ("sh", "hip", 0.00, 0.28, 0.10, 0.00),        # T1-T5 upper thoracic kyphosis spine X-Ray
        ],
    },
    "lower_mid_trap_rhomboid_retraction": {
        "title": "0° SCAPULA W-Y LOCK (Central Inter-Scapular Diamond: Rhomboids & Mid-Lower Trapezius)",
        "is_wrong": False,
        "prompt_positive": (
            "ZERO bones and ZERO spine X-ray visible; deeply sculpted glowing orange-red 3D muscle fiber bundles strictly "
            "on the central diamond/V-shaped rhomboids and middle-to-lower trapezius strictly between the shoulder blades (T1-T10)"
        ),
        "prompt_negative": (
            "NO bones, NO spine X-ray anywhere; NO glowing muscles on lateral latissimus dorsi, arms, deltoids, neck, or waist"
        ),
        # Strictly the Central Inter-Scapular Diamond (`Rhomboids + Middle/Lower Trapezius` between `sh_l` and `sh_r` down to `T10`)!
        # ZERO lateral latissimus dorsi wings!
        "muscle_segments": [
            ("sh", "hip", 0.02, 0.42, 0.21, 0.00),        # Upper-mid inter-scapular Rhomboid Major/Minor & Mid-Trapezius (T1-T6)
            ("sh", "hip", 0.32, 0.68, 0.14, 0.00),        # Tapering V-point of Lower Trapezius (T6-T10) along central spine
            ("sh_l", "sh_r", 0.24, 0.76, 0.17, 0.00),     # Strictly medial inter-scapular bridge (between left & right scapulae)
        ],
        "bone_segments": [],  # 0% bone exposure on Right cuts!
    },
    "anterior_pelvic_tilt_lumbar": {
        "title": "24° ANTERIOR PELVIC TILT (L1-L5 Lumbar Hyperlordosis & Erector Spinae Overload)",
        "is_wrong": True,
        "prompt_positive": (
            "localized translucent ivory 3D lumbar spine vertebrae X-ray (L1-L5) and sacrum, with glowing orange-red 3D "
            "muscle fiber bundles strictly on the tight lumbar erector spinae and hip flexors (iliopsoas)"
        ),
        "prompt_negative": "NO skinless ecorche on head, chest, or arms",
        "muscle_segments": [
            ("sh", "hip", 0.45, 1.02, 0.22, 0.00),
        ],
        "bone_segments": [
            ("sh", "hip", 0.50, 0.98, 0.12, 0.00),
        ],
    },
    "glute_core_posterior_tilt": {
        "title": "0° NEUTRAL PELVIS LOCK (Gluteus Maximus & Deep Transverse Abdominis)",
        "is_wrong": False,
        "prompt_positive": (
            "ZERO bones visible; deeply sculpted glowing orange-red 3D muscle fiber bundles strictly on the gluteus maximus "
            "and lower abdominal core stabilizers"
        ),
        "prompt_negative": "NO bones visible anywhere; NO glow on neck or arms",
        "muscle_segments": [
            ("sh", "hip", 0.62, 1.15, 0.24, 0.00),
        ],
        "bone_segments": [],
    },
}


# ==============================================================================
# 2. MEDIAPIPE POSE LANDMARK EXTRACTOR & TEMPORAL SMOOTHER
# ==============================================================================
def extract_smoothed_pose_sequence(raw_frames_bgr: list[np.ndarray]) -> list[dict]:
    """
    Detects 3D pose landmarks across all frames in `raw_frames_bgr` using MediaPipe Pose,
    rejects sudden tracking glitches, and applies temporal Gaussian smoothing (`sigma=2.0`).
    """
    h, w = raw_frames_bgr[0].shape[:2]
    detector = uar.get_pose_detector()
    raw_kps = []

    for idx_f, bgr_f in enumerate(raw_frames_bgr):
        rgb_f = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2RGB)
        mp_res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb_f))
        valid = False
        if mp_res.pose_landmarks:
            lm = mp_res.pose_landmarks[0]
            ear_pt = (0.5 * (lm[7].x + lm[8].x) * w, 0.5 * (lm[7].y + lm[8].y) * h)
            nose_pt = (lm[0].x * w, lm[0].y * h)
            sh_l = (lm[11].x * w, lm[11].y * h)
            sh_r = (lm[12].x * w, lm[12].y * h)
            sh_pt = (0.5 * (sh_l[0] + sh_r[0]), 0.5 * (sh_l[1] + sh_r[1]))
            el_l = (lm[13].x * w, lm[13].y * h)
            el_r = (lm[14].x * w, lm[14].y * h)
            hip_l = (lm[23].x * w, lm[23].y * h)
            hip_r = (lm[24].x * w, lm[24].y * h)
            hip_pt = (0.5 * (hip_l[0] + hip_r[0]), 0.5 * (hip_l[1] + hip_r[1]))
            kn_l = (lm[25].x * w, lm[25].y * h)
            kn_r = (lm[26].x * w, lm[26].y * h)
            ank_l = (lm[27].x * w, lm[27].y * h)
            ank_r = (lm[28].x * w, lm[28].y * h)

            neck_len = math.hypot(ear_pt[0] - sh_pt[0], ear_pt[1] - sh_pt[1])
            torso_len = math.hypot(hip_pt[0] - sh_pt[0], hip_pt[1] - sh_pt[1])
            if (
                sh_pt[1] < hip_pt[1] - 90.0
                and ear_pt[1] < sh_pt[1] - 12.0
                and ear_pt[1] < h * 0.48
                and 55.0 <= neck_len <= 240.0
                and torso_len >= 130.0
            ):
                valid = True

        if not valid:
            raw_kps.append(None)
            continue

        lum = 0.299 * bgr_f[:, :, 2] + 0.587 * bgr_f[:, :, 1] + 0.114 * bgr_f[:, :, 0]
        cx_h = int(np.clip(0.5 * (ear_pt[0] + sh_pt[0]), 120, w - 120))
        rows_h = np.where(np.percentile(lum[: int(sh_pt[1]), max(0, cx_h - 110) : min(w, cx_h + 110)], 97, axis=1) > 26.0)[0]
        head_top_y = float(rows_h[0]) if len(rows_h) > 0 else (min(ear_pt[1], nose_pt[1]) - 65.0)
        head_top = (ear_pt[0], head_top_y)

        raw_kps.append({
            "head_top": head_top,
            "ear": ear_pt,
            "nose": nose_pt,
            "sh_l": sh_l,
            "sh_r": sh_r,
            "sh": sh_pt,
            "el_l": el_l,
            "el_r": el_r,
            "hip_l": hip_l,
            "hip_r": hip_r,
            "hip": hip_pt,
            "kn_l": kn_l,
            "kn_r": kn_r,
            "ank_l": ank_l,
            "ank_r": ank_r,
        })

    valid_indices = [i for i, k in enumerate(raw_kps) if k is not None]
    if len(valid_indices) == 0:
        fallback = {
            "head_top": (w * 0.50, h * 0.18),
            "ear": (w * 0.50, h * 0.26),
            "nose": (w * 0.48, h * 0.27),
            "sh_l": (w * 0.44, h * 0.35),
            "sh_r": (w * 0.60, h * 0.35),
            "sh": (w * 0.52, h * 0.35),
            "el_l": (w * 0.38, h * 0.48),
            "el_r": (w * 0.66, h * 0.48),
            "hip_l": (w * 0.46, h * 0.60),
            "hip_r": (w * 0.58, h * 0.60),
            "hip": (w * 0.52, h * 0.60),
            "kn_l": (w * 0.44, h * 0.78),
            "kn_r": (w * 0.56, h * 0.78),
            "ank_l": (w * 0.44, h * 0.92),
            "ank_r": (w * 0.56, h * 0.92),
        }
        return [dict(fallback) for _ in range(len(raw_frames_bgr))]

    keys = list(raw_kps[valid_indices[0]].keys())
    out_kps = [{} for _ in range(len(raw_frames_bgr))]
    idx_arr = np.arange(len(raw_frames_bgr), dtype=np.float32)
    for k in keys:
        vx = np.array([raw_kps[i][k][0] for i in valid_indices], dtype=np.float32)
        vy = np.array([raw_kps[i][k][1] for i in valid_indices], dtype=np.float32)
        ix = np.interp(idx_arr, valid_indices, vx)
        iy = np.interp(idx_arr, valid_indices, vy)
        if len(raw_frames_bgr) > 1:
            ix = gaussian_filter1d(ix, sigma=2.2)
            iy = gaussian_filter1d(iy, sigma=2.2)
        for i in range(len(raw_frames_bgr)):
            out_kps[i][k] = (float(ix[i]), float(iy[i]))

    return out_kps


# ==============================================================================
# 3. LANDMARK-ANCHORED DISTANCE FIELD & ANATOMY MASK BUILDER (`v62` STANDARD)
# ==============================================================================
def _segment_distance_field(
    xx: np.ndarray,
    yy: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    radius: float,
    feather: float,
) -> np.ndarray:
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    seg_len_sq = vx * vx + vy * vy + 1e-6
    t = np.clip(((xx - p0[0]) * vx + (yy - p0[1]) * vy) / seg_len_sq, 0.0, 1.0)
    proj_x = p0[0] + t * vx
    proj_y = p0[1] + t * vy
    dist = np.hypot(xx - proj_x, yy - proj_y)
    return np.clip((radius + feather - dist) / max(1.0, feather), 0.0, 1.0)


def _segment_gaussian_field(
    xx: np.ndarray,
    yy: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    sigma: float,
) -> np.ndarray:
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    seg_len_sq = vx * vx + vy * vy + 1e-6
    t = np.clip(((xx - p0[0]) * vx + (yy - p0[1]) * vy) / seg_len_sq, 0.0, 1.0)
    proj_x = p0[0] + t * vx
    proj_y = p0[1] + t * vy
    dist = np.hypot(xx - proj_x, yy - proj_y)
    # Taper slightly at segment endpoints (t near 0 or 1) for organic elliptical muscle bellies
    end_taper = np.sin(np.clip(t, 0.05, 0.95) * math.pi) ** 0.35
    return np.exp(-0.5 * ((dist / max(1.0, sigma)) ** 2.0)).astype(np.float32) * end_taper


def compute_landmark_anchored_anatomy_masks(
    h: int,
    w: int,
    kp: dict,
    target_anatomy_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes `(muscle_gate, bone_gate, forbidden_zone)` dynamically from MediaPipe Pose Landmarks `kp`!
    - Zero hardcoded pixel coordinates!
    - Scales automatically with torso length `torso_len = hypot(hip - sh)`.
    """
    spec = ANATOMY_REGION_SPECS[target_anatomy_id]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    sh = kp["sh"]
    hip = kp["hip"]
    ear = kp["ear"]
    nose = kp["nose"]
    sh_l = kp["sh_l"]
    sh_r = kp["sh_r"]
    head_top = kp["head_top"]

    torso_len = max(140.0, math.hypot(hip[0] - sh[0], hip[1] - sh[1]))
    shoulder_span = abs(sh_r[0] - sh_l[0])
    is_side_view = shoulder_span < (0.42 * torso_len)
    feather = 0.055 * torso_len

    face_dir_x = 1.0 if nose[0] < sh[0] else -1.0
    m_gate = np.zeros((h, w), dtype=np.float32)
    for k_a, k_b, t0, t1, r_scale, lat_scale in spec["muscle_segments"]:
        pa, pb = kp[k_a], kp[k_b]
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        lat_x = face_dir_x * lat_scale * torso_len
        p_start = (pa[0] + dx * t0 + lat_x, pa[1] + dy * t0)
        p_end = (pa[0] + dx * t1 + lat_x, pa[1] + dy * t1)
        rad = r_scale * torso_len
        seg_f = _segment_distance_field(xx, yy, p_start, p_end, radius=rad, feather=feather)
        m_gate = np.maximum(m_gate, seg_f)

    b_gate = np.zeros((h, w), dtype=np.float32)
    for k_a, k_b, t0, t1, r_scale, lat_scale in spec["bone_segments"]:
        pa, pb = kp[k_a], kp[k_b]
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        lat_x = face_dir_x * lat_scale * torso_len
        p_start = (pa[0] + dx * t0 + lat_x, pa[1] + dy * t0)
        p_end = (pa[0] + dx * t1 + lat_x, pa[1] + dy * t1)
        rad = r_scale * torso_len
        seg_f = _segment_distance_field(xx, yy, p_start, p_end, radius=rad, feather=feather * 0.65)
        b_gate = np.maximum(b_gate, seg_f)

    # 1. Cranium & Face exclusion (`head_top` to `ear`/`nose`) — never skinless/glowing!
    skull_center = (0.5 * (head_top[0] + nose[0]), 0.55 * head_top[1] + 0.45 * ear[1])
    skull_rad = 0.21 * torso_len
    skull_mask = _segment_distance_field(xx, yy, head_top, skull_center, radius=skull_rad, feather=feather * 0.45)
    # Also exclude anterior face/jaw in front of `ear -> nose`
    face_front = _segment_distance_field(xx, yy, ear, nose, radius=0.14 * torso_len, feather=feather * 0.40)
    skull_mask = np.maximum(skull_mask, face_front)

    # 2. Lateral Arms exclusion — ONLY active in front/rear views (`not is_side_view`) AND strictly outside
    # the scapular/spinal corridor so it NEVER punches holes through the neck or torso in side views!
    arm_mask = np.zeros((h, w), dtype=np.float32)
    if not is_side_view:
        for sk, ek in [("sh_l", "el_l"), ("sh_r", "el_r")]:
            ps, pe = kp[sk], kp[ek]
            dx, dy = pe[0] - ps[0], pe[1] - ps[1]
            p_a = (ps[0] + 0.35 * dx, ps[1] + 0.35 * dy)
            p_b = (ps[0] + 1.25 * dx, ps[1] + 1.25 * dy)
            arm_mask = np.maximum(
                arm_mask,
                _segment_distance_field(xx, yy, p_a, p_b, radius=0.14 * torso_len, feather=feather * 0.45),
            )

    # 3. Anatomical region-specific structural boundaries (e.g., V-taper inter-scapular diamond for Step 4,
    # monitor exclusion for side views, and vertical cutoffs for cervical/upper-trap steps)
    extra_excl = np.zeros((h, w), dtype=np.float32)
    if target_anatomy_id == "cervical_flexion_overload":
        # Strictly Cervical Neck + C7-T2 Upper Trap Base (`y <= sh_y + 0.20 * torso_len`)
        y_cut_line = sh[1] + 0.24 * (hip[1] - sh[1])
        below_cut = np.clip((yy - y_cut_line) / (0.055 * torso_len), 0.0, 1.0)
        # Exclude anterior chest/throat in front of the neck line
        neck_front_x = np.minimum(ear[0], sh[0]) - 0.06 * torso_len
        front_cut = np.clip((neck_front_x - xx) / (0.05 * torso_len), 0.0, 1.0)
        extra_excl = np.maximum(below_cut, front_cut)

    elif target_anatomy_id == "deep_cervical_chin_tuck":
        # Strictly Cervical Neck Column (`C1-C7` + `C7-T2` collar, `y <= sh_y + 0.16 * torso_len`)!
        # Completely excludes the mid-back and the computer monitor on the left!
        y_cut_line = sh[1] + 0.16 * (hip[1] - sh[1])
        below_cut = np.clip((yy - y_cut_line) / (0.05 * torso_len), 0.0, 1.0)
        monitor_cut = np.clip(((nose[0] - 0.04 * torso_len) - xx) / (0.04 * torso_len), 0.0, 1.0)
        extra_excl = np.maximum(below_cut, monitor_cut)

    elif target_anatomy_id == "upper_trap_protraction_overload":
        # Strictly Upper Trapezius Yoke & C7-T5 Thoracic Apex (`y <= sh_y + 0.26 * torso_len`)!
        # Excludes middle/lower back so Step 3 (`UPPER TRAP STRAIN`) never blurs into Step 4 (`LOWER TRAP & RHOMBOIDS`)!
        y_cut_line = sh[1] + 0.19 * (hip[1] - sh[1])
        below_cut = np.clip((yy - y_cut_line) / (0.06 * torso_len), 0.0, 1.0)
        extra_excl = below_cut

    elif target_anatomy_id == "lower_mid_trap_rhomboid_retraction":
        # Strictly Central Inter-Scapular Diamond (Rhomboids & Middle/Lower Trapezius `T1-T10`)!
        # Excludes the lateral Latissimus Dorsi wings (`abs(x - mid_x) > diamond_half_width(y)`)
        # and excludes the upper neck above `C7` (`y < sh_y - 0.06 * torso_len`)!
        mid_x = 0.5 * (sh[0] + hip[0])
        rel_y = (yy - sh[1]) / max(1.0, hip[1] - sh[1])
        # Diamond width peaks at `rel_y = 0.14` (between scapulae = `0.34 * shoulder_span`) and tapers to `0.10 * shoulder_span` at `rel_y = 0.68`
        half_w_top = 0.34 * shoulder_span
        taper_factor = np.clip(1.0 - np.maximum(0.0, rel_y - 0.16) / 0.56, 0.26, 1.0)
        max_dx = half_w_top * taper_factor
        lat_wing_cut = np.clip((np.abs(xx - mid_x) - max_dx) / (0.055 * torso_len), 0.0, 1.0)
        above_c7_cut = np.clip(((sh[1] - 0.05 * torso_len) - yy) / (0.045 * torso_len), 0.0, 1.0)
        below_t10_cut = np.clip((rel_y - 0.68) / 0.06, 0.0, 1.0)
        extra_excl = np.maximum(np.maximum(lat_wing_cut, above_c7_cut), below_t10_cut)

    else:
        y_cut_line = hip[1] + 0.25 * torso_len
        extra_excl = np.clip((yy - y_cut_line) / (0.07 * torso_len), 0.0, 1.0)

    excl = np.clip(np.maximum(np.maximum(skull_mask, arm_mask), extra_excl), 0.0, 1.0)
    m_gate = np.clip(m_gate * ((1.0 - excl) ** 2.2), 0.0, 1.0)
    b_gate = np.clip(b_gate * ((1.0 - excl) ** 2.2), 0.0, 1.0)

    forbidden_zone = (m_gate < 0.02).astype(np.float32)
    return m_gate, b_gate, forbidden_zone


def _build_anatomical_belly_and_striations(
    h: int,
    w: int,
    kp: dict,
    target_anatomy_id: str,
    m_gate: np.ndarray,
    b_gate: np.ndarray,
    body_interior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements `@Gym Manager` `v62` Stage-2 Anatomical Muscle-Belly Field + 3D Fiber Striations:
    - Generates a smooth Gaussian anatomical muscle-belly intensity field (`anat_belly`) and
      directional 3D muscle fiber ridges (`synth_fs`) + inter-bundle sulci (`synth_sulc`) oriented
      along the exact anatomical muscle fiber direction between the MediaPipe landmarks!
    - Ensures that even if the raw AI plate had weak/missing orange glow inside the true anatomical
      target muscle (such as the anterior/lateral neck in Step 2 `CHIN TUCK` or the central inter-scapular
      Rhomboid/Mid-Trap gap in Step 4 `SCAPULA LOCK`), the target muscle still exhibits crisp, sculpted
      Day-2 3D muscle fiber bundles!
    """
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    sh = kp["sh"]
    hip = kp["hip"]
    ear = kp["ear"]
    nose = kp["nose"]
    sh_l = kp["sh_l"]
    sh_r = kp["sh_r"]
    torso_len = max(140.0, math.hypot(hip[0] - sh[0], hip[1] - sh[1]))

    anat_belly = np.zeros((h, w), dtype=np.float32)
    fiber_wave = np.zeros((h, w), dtype=np.float32)
    sulcus_wave = np.zeros((h, w), dtype=np.float32)

    def _add_Pennate_Bundle(p0, p1, sigma_scale, intensity=1.0, freq_px=8.5):
        nonlocal anat_belly, fiber_wave, sulcus_wave
        sig = sigma_scale * torso_len
        g_field = _segment_gaussian_field(xx, yy, p0, p1, sigma=sig) * intensity
        anat_belly = np.maximum(anat_belly, g_field)
        # Directional coordinates perpendicular (`u_perp`) and parallel (`u_para`) to muscle fiber axis `p0 -> p1`
        vx, vy = p1[0] - p0[0], p1[1] - p0[1]
        L = max(1.0, math.hypot(vx, vy))
        ux, uy = vx / L, vy / L
        nx, ny = -uy, ux
        u_perp = (xx - p0[0]) * nx + (yy - p0[1]) * ny
        u_para = (xx - p0[0]) * ux + (yy - p0[1]) * uy
        # Multi-scale 3D muscle bundle ridges (`freq_px` primary fascicles + `0.48 * freq_px` micro-striations)
        w1 = np.cos(2.0 * math.pi * u_perp / freq_px + 0.15 * np.sin(2.0 * math.pi * u_para / (freq_px * 6.5)))
        w2 = np.cos(2.0 * math.pi * u_perp / (freq_px * 0.52) - 0.25)
        w_bundle = np.cos(2.0 * math.pi * u_perp / (freq_px * 2.4))
        ridge = np.clip(0.52 + 0.34 * w1 + 0.18 * w2, 0.0, 1.0) ** 1.35
        sulc = np.clip(-w_bundle, 0.0, 1.0) ** 1.6
        w_gate = np.clip(g_field * 1.25, 0.0, 1.0)
        fiber_wave = np.maximum(fiber_wave, ridge * w_gate)
        sulcus_wave = np.maximum(sulcus_wave, sulc * w_gate)

    if target_anatomy_id == "cervical_flexion_overload":
        # Posterior cervical extensors (`ear -> sh`) + upper trapezius neck ridge (`C7 -> T2`)
        c7 = (sh[0] + 0.09 * torso_len, sh[1] - 0.01 * torso_len)
        t2 = (sh[0] + 0.11 * torso_len, sh[1] + 0.14 * torso_len)
        occ = (ear[0] + 0.06 * torso_len, ear[1] - 0.05 * torso_len)
        _add_Pennate_Bundle(occ, c7, sigma_scale=0.155, intensity=1.00, freq_px=7.5)
        _add_Pennate_Bundle(ear, sh, sigma_scale=0.145, intensity=0.98, freq_px=7.5)
        _add_Pennate_Bundle(c7, t2, sigma_scale=0.145, intensity=0.96, freq_px=7.8)

    elif target_anatomy_id == "deep_cervical_chin_tuck":
        # Full Cervical Neck Stabilizer Column:
        # 1) Sternocleidomastoid (`ear -> sternal/clavicular sh`)
        # 2) Deep cervical flexors (`jaw/nose_base -> anterior neck base`)
        # 3) Posterior/lateral cervical scalenes & splenius (`ear -> C7`)
        c7 = (sh[0] + 0.05 * torso_len, sh[1] + 0.02 * torso_len)
        st_sh = (sh[0] - 0.04 * torso_len, sh[1] + 0.04 * torso_len)
        jaw = (0.45 * nose[0] + 0.55 * ear[0], 0.55 * nose[1] + 0.45 * ear[1] + 0.05 * torso_len)
        _add_Pennate_Bundle(ear, st_sh, sigma_scale=0.10, intensity=0.98, freq_px=7.0)
        _add_Pennate_Bundle(jaw, st_sh, sigma_scale=0.09, intensity=0.92, freq_px=6.8)
        _add_Pennate_Bundle((ear[0] + 0.04 * torso_len, ear[1]), c7, sigma_scale=0.10, intensity=0.94, freq_px=7.2)

    elif target_anatomy_id == "upper_trap_protraction_overload":
        # Bilateral Upper Trapezius Yoke (`nape -> sh_l`, `nape -> sh_r`, `sh_l -> c7 -> sh_r`)
        nape = (0.45 * ear[0] + 0.55 * sh[0], sh[1] - 0.24 * torso_len)
        t3 = (sh[0], sh[1] + 0.14 * torso_len)
        _add_Pennate_Bundle(nape, sh_l, sigma_scale=0.11, intensity=0.96, freq_px=8.0)
        _add_Pennate_Bundle(nape, sh_r, sigma_scale=0.11, intensity=0.96, freq_px=8.0)
        _add_Pennate_Bundle(sh_l, t3, sigma_scale=0.12, intensity=0.92, freq_px=8.2)
        _add_Pennate_Bundle(sh_r, t3, sigma_scale=0.12, intensity=0.92, freq_px=8.2)

    elif target_anatomy_id == "lower_mid_trap_rhomboid_retraction":
        # Central Inter-Scapular Diamond:
        # 1) Oblique Rhomboid Major/Minor bundles running downward-outward from `T1-T5` midline to medial scapulae
        # 2) Horizontal Middle Trapezius bundles across `T1-T4` between medial scapulae
        # 3) Ascending Lower Trapezius V-taper bundles from `T9` upward-outward to scapular spine
        mid_x = 0.5 * (sh[0] + hip[0])
        t1 = (mid_x, sh[1] - 0.01 * torso_len)
        t4 = (mid_x, sh[1] + 0.20 * torso_len)
        t7 = (mid_x, sh[1] + 0.40 * torso_len)
        t10 = (mid_x, sh[1] + 0.60 * torso_len)
        scap_l_med = (0.42 * sh_l[0] + 0.58 * mid_x, sh[1] + 0.18 * torso_len)
        scap_r_med = (0.42 * sh_r[0] + 0.58 * mid_x, sh[1] + 0.18 * torso_len)
        scap_l_inf = (0.32 * sh_l[0] + 0.68 * mid_x, sh[1] + 0.36 * torso_len)
        scap_r_inf = (0.32 * sh_r[0] + 0.68 * mid_x, sh[1] + 0.36 * torso_len)

        # Rhomboid chevron fibers (`T1 -> scap_med`, `T4 -> scap_inf`)
        _add_Pennate_Bundle(t1, scap_l_med, sigma_scale=0.10, intensity=0.98, freq_px=7.5)
        _add_Pennate_Bundle(t1, scap_r_med, sigma_scale=0.10, intensity=0.98, freq_px=7.5)
        _add_Pennate_Bundle(t4, scap_l_inf, sigma_scale=0.10, intensity=0.96, freq_px=7.5)
        _add_Pennate_Bundle(t4, scap_r_inf, sigma_scale=0.10, intensity=0.96, freq_px=7.5)
        # Lower trapezius V-taper (`T10 -> scap_l_med`, `T10 -> scap_r_med`)
        _add_Pennate_Bundle(t10, scap_l_med, sigma_scale=0.09, intensity=0.92, freq_px=7.2)
        _add_Pennate_Bundle(t10, scap_r_med, sigma_scale=0.09, intensity=0.92, freq_px=7.2)
        # Central spinal groove (`T1 -> T10` median longitudinal sulcus)
        spine_groove = np.exp(-0.5 * (((xx - mid_x) / (0.018 * torso_len)) ** 2.0)).astype(np.float32)
        sulcus_wave = np.maximum(sulcus_wave, spine_groove * 0.85)

    anat_belly = np.clip(anat_belly * m_gate * body_interior, 0.0, 1.0)
    return anat_belly, fiber_wave, sulcus_wave


# ==============================================================================
# 4. `@GYM MANAGER` `v62` 2-STAGE DAY-2-AUTHENTIC SHADER (ZERO GHOSTING!)
# ==============================================================================
def shade_universal_day2_authentic(
    bgr_frame: np.ndarray,
    f0_bgr_frame: np.ndarray | None,
    kp_px: dict,
    target_anatomy_id: str,
    is_wrong: bool,
    rep_a: float = 1.0,
) -> np.ndarray:
    """
    Universal Day 2 (`De_mCcFAVTO` / `v62`) 2-Stage 3D Muscle-Fiber & Localized Bone X-Ray Shader:
    - Stage 1 (`Cool-Channel Matte Dark-Charcoal Reset`):
      Outside `m_gate`, any orange/red AI glow (`red_any`) is replaced strictly using
      `cool_base = np.minimum(g_ch, b_ch)` (`grey_charcoal` in `[18, 86]`), turning off-target glow
      into seamless `#2A2E35` matte dark-charcoal skin with 0.0% bright/pale grey ghost patches!
    - Stage 2 (`Hybrid Native + Landmark-Anchored Anatomical Muscle-Belly Field + 3D Fiber Striations`):
      Inside `m_gate`, combines native orange-red muscle intensity (`native_fe * m_gate`) with the
      landmark-anchored anatomical muscle-belly field (`anat_belly`) and 4-Band Multi-Scale DoG +
      pennate 3D muscle fiber striations (`fs_eff`, `sulc_eff`).
    """
    blur = cv2.GaussianBlur(bgr_frame, (0, 0), 0.85)
    plate_bgr = cv2.addWeighted(bgr_frame, 1.24, blur, -0.24, 0)
    rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, w = rgb.shape[:2]

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    fiber_signal = 0.70 * r_ch + 0.22 * g_ch + 0.08 * b_ch
    sigma_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 0.75)
    sigma_ridge = cv2.GaussianBlur(fiber_signal, (0, 0), 2.0)
    sigma_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 5.2)
    sigma_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 11.5)

    hf_ridge = (sigma_micro - sigma_ridge) * 4.6 + (sigma_ridge - sigma_belly) * 2.6
    fs = np.clip(0.46 + hf_ridge / 29.0, 0.0, 1.0) ** 1.48
    hv = np.clip((sigma_belly - 22.0) / 142.0, 0.0, 1.0)
    sulc = np.clip((sigma_broad - sigma_ridge - 0.9) / 9.8, 0.0, 1.0)

    m_gate, b_gate, _ = compute_landmark_anchored_anatomy_masks(h, w, kp_px, target_anatomy_id)

    # Stage 1: `@Gym Manager` `v62` Cool-Channel Non-Target Charcoal Skin (`cool_base = np.minimum(g_ch, b_ch)`)
    # Because `cool_base` has ZERO red/orange glow contamination, any erased off-target orange glow
    # becomes smooth matte dark-charcoal skin (`20..84`) matching `#2A2E35` with 0% pale grey blotches!
    cool_base = np.minimum(g_ch, b_ch)
    cool_belly = cv2.GaussianBlur(cool_base, (0, 0), 4.2)
    hv_cool = np.clip((cool_belly - 16.0) / 115.0, 0.0, 1.0)
    grey_charcoal = np.clip(
        22.0 + 54.0 * (hv_cool ** 0.95) + np.clip(hf_ridge, -12.0, 14.0) * 0.38 - sulc * 10.0,
        16.0,
        84.0,
    )
    grey_sculpt = np.clip(30.0 + 142.0 * (hv ** 1.05) + hf_ridge * 1.15 - sulc * 26.0, 18.0, 226.0)
    neutral_lum = grey_charcoal * (1.0 - m_gate) + grey_sculpt * m_gate

    mono_target = np.stack([
        neutral_lum * 0.95,
        neutral_lum * 0.98,
        neutral_lum * 1.04,
    ], axis=2)

    red_any = np.clip((rb_diff - 7.5) / 38.0, 0.0, 1.0)
    # In Step 2, also suppress any red code glow on the computer monitor (`x < nose_x - 15`)
    mono = rgb * (1.0 - red_any[:, :, None]) + mono_target * red_any[:, :, None]

    # Compute mannequin `body_interior` mask so synthetic anatomical muscle bellies NEVER bleed onto the dark studio background!
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    body_raw = ((base_lum > 25.0) | (cool_belly > 21.0)).astype(np.uint8)
    body_eroded = cv2.erode(body_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    body_interior = cv2.GaussianBlur(body_eroded.astype(np.float32), (0, 0), 2.2) * np.clip((base_lum - 24.0) / 22.0, 0.0, 1.0)

    # Stage 2: Hybrid Native + Landmark-Anchored Anatomical Muscle-Belly Field + 3D Fiber Striations
    anat_belly, synth_fs, synth_sulc = _build_anatomical_belly_and_striations(
        h, w, kp_px, target_anatomy_id, m_gate, b_gate, body_interior
    )

    rb_blur = cv2.GaussianBlur(rb_diff, (0, 0), 2.6)
    rb_eff = np.maximum(rb_diff, rb_blur * 0.85)
    native_fe = np.clip((rb_eff - 10.0) / 36.0, 0.0, 1.0) * m_gate

    # Blend native DoG fiber ridges (`fs`) with anatomical pennate fiber ridges (`synth_fs`) where native glow was weak
    native_weight = np.clip(native_fe * 1.35, 0.0, 1.0)
    fs_eff = fs * native_weight + np.maximum(fs * 0.65, synth_fs) * (1.0 - native_weight)
    sulc_eff = np.maximum(sulc, synth_sulc * 0.72)
    hv_eff = np.maximum(hv, (0.62 + 0.38 * hv_cool) * (anat_belly ** 0.52))

    if is_wrong:
        # Deep, vivid Day-2 Crimson Red (`#F82830`) + Localized 3D Vertebrae Bone X-Ray
        deep_r = np.array([124.0, 6.0, 14.0], dtype=np.float32)
        mid_r = np.array([248.0, 20.0, 32.0], dtype=np.float32)
        crest_r = np.array([255.0, 118.0, 126.0], dtype=np.float32)

        xray_alpha = np.maximum(native_fe, anat_belly * 0.88) * m_gate

        bone_w = (
            np.clip((g_ch - 76.0) / 82.0, 0.0, 1.0)
            * np.clip((rb_diff - 22.0) / 46.0, 0.0, 1.0)
            * b_gate
        )[:, :, None]
        xray_bone = np.stack([
            np.clip(r_ch * 1.18 + 28.0 + hf_ridge * 0.70, 0, 255),
            np.clip(g_ch * 0.62 + 10.0 + hf_ridge * 0.26, 0, 255),
            np.clip(b_ch * 0.70 + 14.0, 0, 255),
        ], axis=2)

        strand_mod = (0.18 + 0.82 * fs_eff) * ((1.0 - 0.76 * sulc_eff) ** 1.32)
        ridge_boost = np.clip(fs_eff * 0.76 + np.clip((rb_diff - 55.0) / 75.0, 0.0, 1.0) * 0.44, 0.0, 1.0)[:, :, None]
        vol_shade = np.clip(0.24 + 0.76 * hv_eff * (0.36 + 0.64 * fs_eff) * (1.0 - 0.48 * sulc_eff), 0.0, 1.0)[:, :, None]

        fiber_red = (
            deep_r * (1.0 - vol_shade)
            + mid_r * vol_shade * (1.0 - ridge_boost * 0.56)
            + crest_r * (ridge_boost ** 1.38) * 0.84
        )
        f_alpha = np.clip(xray_alpha * (0.64 + 0.68 * strand_mod), 0.0, 0.97)[:, :, None]
        combined_src = fiber_red * (1.0 - bone_w) + xray_bone * bone_w
        alpha_final = f_alpha * (1.0 - bone_w) + (xray_alpha[:, :, None] * 0.96) * bone_w
        out = mono * (1.0 - alpha_final) + np.clip(combined_src, 0, 255) * alpha_final
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        # Right Cuts (Steps 2 & 4): ZERO bone X-Ray (`0%` bone!) + Rich Day-2 Emerald-Green 3D Muscle Fibers
        fe = np.maximum(native_fe, (anat_belly ** 0.68) * 0.96) * m_gate
        hc = np.maximum(np.clip((rb_diff - 58.0) / 72.0, 0.0, 1.0), anat_belly * 0.65) * fs_eff * m_gate
        strand_mod = (0.16 + 0.84 * fs_eff) * ((1.0 - 0.78 * sulc_eff) ** 1.35)
        fiber_alpha = np.clip(fe * (0.68 + 0.68 * strand_mod), 0.0, 0.97)[:, :, None]
        ridge_boost = np.clip(fs_eff * 0.74 + hc * 0.46, 0.0, 1.0)[:, :, None]
        vol_shade = np.clip(0.24 + 0.76 * hv_eff * (0.36 + 0.64 * fs_eff) * (1.0 - 0.50 * sulc_eff), 0.0, 1.0)[:, :, None]

        deep_col = np.array([4.0, 136.0, 28.0], dtype=np.float32)
        mid_col = np.array([16.0, 246.0, 64.0], dtype=np.float32)
        crest_col = np.array([170.0, 255.0, 184.0], dtype=np.float32)

        fiber_rgb = (
            deep_col * (1.0 - vol_shade)
            + mid_col * vol_shade * (1.0 - ridge_boost * 0.60)
            + crest_col * (ridge_boost ** 1.40) * 0.84
        )
        out = mono * (1.0 - fiber_alpha) + np.clip(fiber_rgb, 0.0, 255.0) * fiber_alpha
        return np.clip(out, 0, 255).astype(np.uint8)


# ==============================================================================
# 5. AUTOMATED ANATOMICAL TARGET COMPLIANCE QC CHECK
# ==============================================================================
def verify_anatomical_target_compliance(
    bgr_frame_1080: np.ndarray,
    kp_1080: dict,
    target_anatomy_id: str,
    is_wrong: bool,
) -> dict:
    h, w = bgr_frame_1080.shape[:2]
    m_gate, _, forbidden_zone = compute_landmark_anchored_anatomy_masks(h, w, kp_1080, target_anatomy_id)

    stage_mask = np.zeros((h, w), dtype=bool)
    stage_mask[455:1330, 140:940] = True

    b = bgr_frame_1080[:, :, 0].astype(np.float32)
    g = bgr_frame_1080[:, :, 1].astype(np.float32)
    r = bgr_frame_1080[:, :, 2].astype(np.float32)
    ch = (r - b) if is_wrong else (g - b)
    non_white = np.minimum(np.minimum(r, g), b) < 190.0

    lum_1080 = 0.299 * r + 0.587 * g + 0.114 * b
    raw_act = ((ch > 52.0) & (lum_1080 > 46.0) & non_white & stage_mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_act = cv2.erode(raw_act, kernel, iterations=1) > 0

    total_act = int(eroded_act.sum())
    in_target = int((eroded_act & (m_gate > 0.015)).sum())
    in_forbidden = int((eroded_act & (m_gate <= 0.015)).sum())

    in_target_ratio = in_target / float(max(1, total_act))
    forbidden_leak_ratio = in_forbidden / float(max(1, total_act))
    passed = (total_act >= 9500) and (in_target_ratio >= 0.94) and (forbidden_leak_ratio <= 0.02)

    return {
        "target_anatomy_id": target_anatomy_id,
        "total_muscle_px": total_act,
        "in_target_px": in_target,
        "in_target_ratio": in_target_ratio,
        "forbidden_leak_px": in_forbidden,
        "forbidden_leak_ratio": forbidden_leak_ratio,
        "passed": passed,
    }
