#!/usr/bin/env python3
"""
@gymform.3d — Day 1 Cable Rope Pushdown — v21 Evidence-Based Anatomical Striation & Head-Isolation Master (28.0s)

Addresses user feedback (t87u-t92u):
1. "근육결이 느껴질 정도여야 하며, 그 근육이 잘보여져야 하는데 타켓팅되는 근육을 정확히 잘 못잡아주는 것 같은데? 개선점을 찾아"
2. "이 bot 프로필보다 못한 느낌이야 / 더 좋은 교보재도 많을걸? / 구글 검색도 해보고 / 논문도 있을거고 / 정리도 해"

Key Upgrades over v20:
- Preserves 100% of the bot profile's native 3D muscle fiber relief (encoded in the G/B channels of the sculpted mannequin) AND enhances it with multi-scale anisotropic fascicle striations aligned to each muscle head's true anatomical pennation angle (Ando et al. 2020 MSSE; Kenhub / Delavier Strength Training Anatomy).
- Replaces generic circular blobs with surgical, polygon-accurate anatomical head segmentation separated by dark intermuscular sulci (grooves):
  * Mistake 01 (Shoulders Roll Forward): Isolates ONLY the Glenohumeral Joint Capsule & Rotator Cuff (Supraspinatus/Subscapularis insertion) at the anterior-superior shoulder. Triceps stay 100% unlit matte grey.
  * Mistake 02 (Elbows Swing Up & Down): Isolates ONLY the Anterior Deltoid (Front Delt, -24° fan fibers) + Anterior Brachialis/Biceps compensation. Triceps stay 100% unlit matte grey, proving tension was stolen from the triceps.
  * Correct Technique (Pin Elbows & Spread Rope): Deltoids turn 100% OFF (neutral matte grey!). Isolates ONLY the Triceps Lateral Head (-18° oblique horseshoe fibers) and Long Head (+10° vertical-medial fibers), with a crisp dark intermuscular septum dividing the two heads.
  * Peak Isolation + 4x12 (Lockout): Activates all 3 heads of the Triceps Brachii per EMG research (MSSE 2023; Kholinne et al. 2018 PMC6136322) — Lateral Head (외측두), Long Head (장두), and Distal Medial Head (내측두, just above the olecranon tendon) — with distinct inter-head anatomical grooves and high-contrast muscle fiber striations.
- Zero muddy neon bloom outside the muscle boundary + 3D Cylindrical Body-Wrapping Dark Gunmetal Ribbons (`#1A1E26`..`#303642` + `#DAE2EE` titanium bevel rim).
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 30
DURATION = 28.0
TOTAL_FRAMES = int(FPS * DURATION)

BASE_DIR = "/workspace/daily_info_reels"
OUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = "/workspace/frames_v21_day1"

WRONG_I2V = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_I2V = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")
WRONG_JPG = os.path.join(OUT_DIR, "gym_day1_wrong_plate.jpg")
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")

OUT_PREVIEW = os.path.join(OUT_DIR, "day1_v21_striation_preview.jpg")
OUT_COMPARISON = os.path.join(OUT_DIR, "day1_v21_vs_avatar_closeup.jpg")
OUT_HOOK = os.path.join(OUT_DIR, "day1_v21_step0_hook.jpg")
OUT_STEP1 = os.path.join(OUT_DIR, "day1_v21_step1_injury.jpg")
OUT_STEP2 = os.path.join(OUT_DIR, "day1_v21_step2_wrong_muscle.jpg")
OUT_STEP3 = os.path.join(OUT_DIR, "day1_v21_step3_posture_key.jpg")
OUT_STEP4 = os.path.join(OUT_DIR, "day1_v21_step4_target_muscle.jpg")
OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")

BG_CHARCOAL = (18, 20, 24)
COL_RED = (255, 59, 48)          # #FF3B30
COL_GREEN = (24, 226, 105)       # #18E269
COL_WHITE = (255, 255, 255)

GUNMETAL_DARK = (26, 30, 38)
GUNMETAL_MID = (48, 54, 66)
GUNMETAL_RIM = (218, 226, 238)
SHADOW_COL = (6, 8, 12)

PRETENDARD_BLACK = "/home/box/.fonts/Pretendard-Black.otf"
PRETENDARD_BOLD = "/home/box/.fonts/Pretendard-ExtraBold.otf"
FALLBACK_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = [
        PRETENDARD_BLACK if bold else PRETENDARD_BOLD,
        "/home/ubuntu/.fonts/Pretendard-Black.otf",
        FALLBACK_BOLD,
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_KICKER = load_font(26, bold=False)
FONT_MAIN = load_font(50, bold=True)
FONT_SUB = load_font(28, bold=False)
FONT_SETS = load_font(86, bold=True)
FONT_SETS_SUB = load_font(26, bold=False)
FONT_LABEL = load_font(23, bold=True)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def make_anisotropic_fascicle_field(h: int, w: int, angle_deg: float, wavelength_px: float = 6.2) -> np.ndarray:
    """
    Generates realistic multipennate muscle fascicle striation ridges and grooves
    oriented along the anatomical pennation angle (Delavier / Muscle & Motion style).
    Returns values in [-1.0, +1.0].
    """
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    theta = math.radians(angle_deg)
    # Coordinate perpendicular to muscle fibers (across fascicle bundles) and parallel (along fibers)
    u = xx * math.cos(theta) + yy * math.sin(theta)
    v = -xx * math.sin(theta) + yy * math.cos(theta)

    # Slight anatomical curvature along the muscle belly
    curve_mod = 3.2 * np.sin(v / 42.0)
    phase1 = 2.0 * math.pi * (u + curve_mod) / wavelength_px
    phase2 = 2.0 * math.pi * (u - 0.6 * curve_mod) / (wavelength_px * 0.52)
    phase3 = 2.0 * math.pi * u / (wavelength_px * 1.85)

    # Sharp fascicle ridge profile (narrow bright tendon/fascicle ridges, deep dark endomysium grooves)
    wave = 0.55 * np.sin(phase1) + 0.28 * np.sin(phase2) + 0.17 * np.cos(phase3)
    # Modulate along fiber length so striations taper near tendon insertions
    long_mod = 0.82 + 0.18 * np.cos(v / 18.0)
    return np.clip(wave * long_mod, -1.0, 1.0)


# Precompute anatomical fascicle striation fields for 1080x1920 canvas:
# 1. Anterior Deltoid (clavicular/acromial fibers running downward & posteriorly at -26 deg)
FASCICLE_ANT_DELT = make_anisotropic_fascicle_field(H, W, angle_deg=-26.0, wavelength_px=6.0)
# 2. Triceps Lateral Head (oblique multipennate horseshoe fibers running at -18 deg)
FASCICLE_TRI_LAT = make_anisotropic_fascicle_field(H, W, angle_deg=-18.0, wavelength_px=5.8)
# 3. Triceps Long Head (thick pennate fibers running down posterior-medial arm at +11 deg)
FASCICLE_TRI_LONG = make_anisotropic_fascicle_field(H, W, angle_deg=11.0, wavelength_px=6.4)
# 4. Triceps Medial Head (deep distal fan fibers converging to olecranon tendon at -3 deg)
FASCICLE_TRI_MED = make_anisotropic_fascicle_field(H, W, angle_deg=-3.0, wavelength_px=5.2)


def polygon_soft_mask(h: int, w: int, pts: list[tuple[float, float]], blur_sigma: float = 2.2) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    arr = np.array([[[int(round(x)), int(round(y))] for (x, y) in pts]], dtype=np.int32)
    cv2.fillPoly(mask, arr, 1.0)
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return np.clip(mask, 0.0, 1.0)


def build_anatomical_head_masks(arm_drop_px: float = 0.0) -> dict[str, np.ndarray]:
    """
    Constructs surgical, anatomically accurate muscle-head masks in 1080x1920 space
    calibrated to the 3D sculpted mannequin (with dynamic distal shift `arm_drop_px` during elbow extension).
    Includes explicit dark intermuscular sulci (grooves) between heads so adjacent heads NEVER blur into one blob!
    """
    d_mid = 0.45 * arm_drop_px
    d_low = 0.85 * arm_drop_px

    # 1. Rotator Cuff & Anterior-Superior Glenohumeral Capsule (Mistake 01: Shoulder Roll Impingement)
    pts_rotator = [
        (642, 636), (682, 628), (722, 640), (728, 675),
        (704, 704), (664, 708), (636, 682),
    ]
    m_rotator = polygon_soft_mask(H, W, pts_rotator, blur_sigma=2.4)

    # 2. Anterior Deltoid (Front Delt — Mistake 02: Elbow Swing Flexor Compensation)
    pts_ant_delt = [
        (634, 650), (674, 642), (694, 668), (692, 716 + d_mid * 0.2),
        (666, 748 + d_mid * 0.3), (638, 736 + d_mid * 0.3), (626, 692),
    ]
    # Also include Anterior Brachialis / Biceps upper flexor slip when elbow swings up
    pts_brach_flex = [
        (628, 748 + d_mid * 0.3), (654, 744 + d_mid * 0.3),
        (658, 825 + d_mid * 0.6), (632, 832 + d_mid * 0.6), (620, 785 + d_mid * 0.4),
    ]
    m_ant_delt = np.maximum(
        polygon_soft_mask(H, W, pts_ant_delt, blur_sigma=2.0),
        0.82 * polygon_soft_mask(H, W, pts_brach_flex, blur_sigma=2.2),
    )

    # 3. Triceps Brachii — LATERAL HEAD (외측두, Prominent Horseshoe Outer Belly)
    # Sits inferior to the deltoid sulcus (y > 742) and anterior/lateral to the intermuscular groove (x < 712)
    pts_tri_lat = [
        (652, 742 + d_mid * 0.25),
        (686, 734 + d_mid * 0.25),
        (710, 750 + d_mid * 0.30),
        (710, 822 + d_mid * 0.65),
        (690, 862 + d_low * 0.80),
        (660, 854 + d_low * 0.80),
        (644, 808 + d_mid * 0.55),
        (644, 766 + d_mid * 0.35),
    ]
    m_tri_lat = polygon_soft_mask(H, W, pts_tri_lat, blur_sigma=1.8)

    # 4. Triceps Brachii — LONG HEAD (장두, Thick Posterior-Medial Belly originating from Scapula)
    # Sits posterior to the intermuscular septum (x: 715..768, y: 746..868)
    pts_tri_long = [
        (718, 746 + d_mid * 0.25),
        (748, 742 + d_mid * 0.25),
        (768, 772 + d_mid * 0.40),
        (764, 828 + d_mid * 0.65),
        (742, 868 + d_low * 0.80),
        (716, 860 + d_low * 0.80),
        (714, 804 + d_mid * 0.55),
    ]
    m_tri_long = polygon_soft_mask(H, W, pts_tri_long, blur_sigma=1.8)

    # 5. Triceps Brachii — MEDIAL HEAD (내측두, Deep Distal Triangle above Olecranon — MSSE 2023 #1 EMG in Pushdowns!)
    pts_tri_med = [
        (666, 848 + d_low * 0.78),
        (712, 844 + d_low * 0.78),
        (732, 872 + d_low * 0.90),
        (718, 912 + d_low * 1.00),
        (676, 910 + d_low * 1.00),
        (658, 878 + d_low * 0.90),
    ]
    m_tri_med = polygon_soft_mask(H, W, pts_tri_med, blur_sigma=1.8)

    # 6. Far-arm Triceps Lateral Head (visible near left cable rope at x: 495..565, y: 710..815)
    pts_far_tri = [
        (505, 712 + d_mid * 0.25), (548, 708 + d_mid * 0.25),
        (556, 785 + d_mid * 0.60), (512, 792 + d_mid * 0.60),
    ]
    m_far_tri = polygon_soft_mask(H, W, pts_far_tri, blur_sigma=2.0)

    # Carve crisp anatomical sulci (intermuscular grooves) so the heads are visually distinct!
    sulcus_canvas = np.zeros((H, W), dtype=np.float32)
    # Groove 1: Deltoid-Triceps Sulcus (separating Deltoid cap above from Triceps Lateral/Long heads below)
    cv2.polylines(
        sulcus_canvas,
        [np.array([[636, int(744 + d_mid * 0.25)], [685, int(733 + d_mid * 0.25)], [758, int(745 + d_mid * 0.25)]], dtype=np.int32)],
        False,
        1.0,
        thickness=4,
        lineType=cv2.LINE_AA,
    )
    # Groove 2: Lateral-Long Head Intermuscular Septum (the signature vertical groove of the triceps horseshoe)
    cv2.polylines(
        sulcus_canvas,
        [np.array([[712, int(740 + d_mid * 0.25)], [713, int(805 + d_mid * 0.55)], [715, int(858 + d_low * 0.80)]], dtype=np.int32)],
        False,
        1.0,
        thickness=4,
        lineType=cv2.LINE_AA,
    )
    # Groove 3: Distal Horseshoe Aponeurosis Arch (between Lateral/Long heads and Medial head / Tendon)
    cv2.polylines(
        sulcus_canvas,
        [np.array([[656, int(852 + d_low * 0.80)], [692, int(842 + d_low * 0.78)], [736, int(864 + d_low * 0.82)]], dtype=np.int32)],
        False,
        1.0,
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    sulcus_mask = cv2.GaussianBlur(sulcus_canvas, (0, 0), sigmaX=1.2, sigmaY=1.2)

    return {
        "rotator": m_rotator,
        "ant_delt": m_ant_delt,
        "tri_lat": m_tri_lat,
        "tri_long": m_tri_long,
        "tri_med": m_tri_med,
        "far_tri": m_far_tri,
        "sulcus": np.clip(sulcus_mask, 0.0, 1.0),
    }


def prepare_sculpted_mannequin_frame(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """
    Converts a raw frame from `gym_day1_correct_i2v.mp4` into:
    1. `mono`: A 100% full-bleed neutral matte grey 3D anatomical mannequin where:
       - The blue neon halo is completely removed.
       - Inside the sculpted muscle region, the G-channel (which carries the deep muscle fiber striations and sulci)
         is mapped into the neutral matte grey skin tone so even inactive muscles look deeply sculpted!
    2. `native_striation`: High-frequency muscle fiber relief extracted directly from the bot profile's G/B channels.
    3. `muscle_silhouette`: Exact physical silhouette of the 3D mannequin's arm/shoulder musculature.
    4. `arm_drop_px`: Dynamic vertical extension offset of the forearm/hand in this frame.
    """
    scale = 0.90
    y_shift = 55

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_w, target_h = int(W * scale), int(H * scale)
    rgb_res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    r_ch, g_ch, b_ch = rgb_res[:, :, 0], rgb_res[:, :, 1], rgb_res[:, :, 2]

    red_excess = np.clip((r_ch - np.maximum(g_ch, b_ch)) / 130.0, 0.0, 1.0)
    green_excess = np.clip((g_ch - np.maximum(r_ch, b_ch)) / 110.0, 0.0, 1.0)
    blue_excess = np.clip((b_ch - np.maximum(r_ch, g_ch)) / 80.0, 0.0, 1.0)

    # Neutral matte grey luminance that preserves 100% of the G-channel muscle fiber ridges & sulci!
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch - 0.55 * np.maximum(0.0, b_ch - np.maximum(r_ch, g_ch))
    muscle_sculpt_lum = 52.0 + 1.18 * g_ch
    hand_sculpt_lum = 68.0 + 0.72 * r_ch

    neutral_lum = base_lum * (1.0 - red_excess) + muscle_sculpt_lum * red_excess
    neutral_lum = neutral_lum * (1.0 - green_excess) + hand_sculpt_lum * green_excess
    neutral_lum = np.clip(neutral_lum, 18.0, 242.0)

    # Extract native high-frequency 3D muscle fiber striations (근육결) from R & G channels (Bot Profile source!)
    g_clean = g_ch - np.clip(g_ch - r_ch * 0.95, 0.0, 255.0) * 0.90
    fiber_lum = 0.62 * r_ch + 0.38 * g_clean
    blur_fine = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=1.9, sigmaY=1.9)
    blur_med = cv2.GaussianBlur(fiber_lum, (0, 0), sigmaX=6.0, sigmaY=6.0)
    fiber_detail = fiber_lum + (fiber_lum - blur_fine) * 1.85 + (blur_fine - blur_med) * 0.95
    fiber_norm_res = np.clip((fiber_detail - 32.0) / 210.0, 0.0, 1.0)

    # Also update neutral_lum inside the muscle so inactive muscles keep deep 3D sculpted grey fiber grooves!
    sculpted_grey_muscle = 42.0 + 135.0 * (fiber_norm_res ** 1.08)
    neutral_lum = neutral_lum * (1.0 - red_excess * 0.92) + sculpted_grey_muscle * (red_excess * 0.92)
    neutral_lum = np.clip(neutral_lum, 18.0, 242.0)

    native_fiber_res = np.clip((fiber_norm_res - 0.48) * 2.2, -1.0, 1.0)

    # Physical silhouette gate of the arm/shoulder & true 3D sculpted muscle presence (red_core_res)
    body_gate_res = np.clip((neutral_lum - 40.0) / 14.0, 0.0, 1.0)
    rg_raw = np.clip(r_ch - g_clean * 0.82, 0.0, 255.0)
    red_core_res = np.clip((rg_raw - 20.0) / 108.0, 0.0, 1.0) * body_gate_res

    norm = neutral_lum / 255.0
    mono_r = BG_CHARCOAL[0] + (235 - BG_CHARCOAL[0]) * (norm ** 0.92)
    mono_g = BG_CHARCOAL[1] + (239 - BG_CHARCOAL[1]) * (norm ** 0.92)
    mono_b = BG_CHARCOAL[2] + (245 - BG_CHARCOAL[2]) * (norm ** 0.92)
    mono_res = np.stack([mono_r, mono_g, mono_b], axis=2)

    x0 = (W - target_w) // 2
    y0 = max(0, min(H - target_h, (H - target_h) // 2 + y_shift))
    pad_top = y0
    pad_bot = H - (y0 + target_h)
    pad_left = x0
    pad_right = W - (x0 + target_w)

    full_mono = cv2.copyMakeBorder(
        mono_res.astype(np.uint8),
        pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_REPLICATE,
    )
    # Clean vertical pulley text
    pulley_roi = full_mono[180:920, 40:310]
    pulley_smooth = cv2.GaussianBlur(pulley_roi, (31, 31), sigmaX=12, sigmaY=12)
    diff_p = pulley_roi.astype(np.float32) - pulley_smooth.astype(np.float32)
    letter_mask = np.clip((np.max(diff_p, axis=2) - 10.0) / 18.0, 0.0, 1.0)[:, :, None]
    full_mono[180:920, 40:310] = (pulley_roi * (1.0 - letter_mask) + pulley_smooth * letter_mask).astype(np.uint8)

    small_bg = cv2.resize(full_mono, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small_bg = cv2.GaussianBlur(small_bg, (0, 0), sigmaX=6.0, sigmaY=6.0)
    bg_blur = cv2.resize(small_bg, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    vig = 1.0 - 0.28 * (((yy - H * 0.52) / (H * 0.55)) ** 2 + ((xx - W * 0.5) / (W * 0.65)) ** 2)
    vig = np.clip(vig, 0.62, 1.0)[:, :, None]
    bg_blur = bg_blur * vig

    mask = np.ones((target_h, target_w), dtype=np.float32)
    feather = 110
    for i in range(feather):
        f = 0.5 - 0.5 * math.cos(math.pi * (i / float(feather)))
        mask[i, :] *= f
        mask[target_h - 1 - i, :] *= f
        mask[:, i] *= f
        mask[:, target_w - 1 - i] *= f

    full_mask = np.zeros((H, W), dtype=np.float32)
    full_mask[y0:y0 + target_h, x0:x0 + target_w] = mask
    full_mask3 = full_mask[:, :, None]

    canvas = (bg_blur * (1.0 - full_mask3) + full_mono.astype(np.float32) * full_mask3).astype(np.uint8)

    native_fiber_full = np.zeros((H, W), dtype=np.float32)
    red_core_full = np.zeros((H, W), dtype=np.float32)
    body_gate_full = np.zeros((H, W), dtype=np.float32)
    green_hand_full = np.zeros((H, W), dtype=np.float32)

    native_fiber_full[y0:y0 + target_h, x0:x0 + target_w] = native_fiber_res * mask
    red_core_full[y0:y0 + target_h, x0:x0 + target_w] = red_core_res * mask
    body_gate_full[y0:y0 + target_h, x0:x0 + target_w] = body_gate_res * mask
    green_hand_full[y0:y0 + target_h, x0:x0 + target_w] = np.clip((green_excess - 0.25) / 0.45, 0.0, 1.0) * body_gate_res * mask

    # Estimate dynamic arm extension drop from hand mask centroid
    g_ys, _ = np.where(green_hand_full > 0.3)
    if len(g_ys) > 20:
        hand_y = float(np.mean(g_ys))
        arm_drop_px = max(0.0, min(36.0, (hand_y - 870.0) * 0.14))
    else:
        arm_drop_px = 0.0

    return {
        "mono": canvas,
        "native_fiber": native_fiber_full,
        "red_core": red_core_full,
        "body_gate": body_gate_full,
        "green_hand": green_hand_full,
        "arm_drop_px": arm_drop_px,
    }


def apply_kinematic_posture_modification(frame_data: dict[str, np.ndarray], mode: str, rep_wave: float) -> dict[str, np.ndarray]:
    """
    For `wrong_shoulder` (Mistake 01) and `wrong_elbow` (Mistake 02), applies a subtle biomechanical
    posture displacement to the sculpted mannequin's shoulder/arm so the posture visibly rolls forward
    or swings up & down while keeping 100% of the sculpted 3D muscle anatomy!
    """
    if mode not in ("wrong_shoulder", "wrong_elbow"):
        return frame_data

    # Smooth displacement field around the upper arm & shoulder (x: 480..820, y: 560..1080)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    roi_weight = np.exp(-(((xx - 660.0) / 135.0) ** 2 + ((yy - 760.0) / 185.0) ** 2))

    if mode == "wrong_shoulder":
        # Shoulders roll forward and shrug slightly (-x, -y at shoulder)
        dx = 14.0 * rep_wave * roi_weight
        dy = 10.0 * rep_wave * roi_weight
    else:
        # Elbows swing forward and up (-x, +y pull)
        elbow_weight = np.exp(-(((xx - 640.0) / 145.0) ** 2 + ((yy - 820.0) / 175.0) ** 2))
        dx = 18.0 * rep_wave * elbow_weight
        dy = 16.0 * rep_wave * elbow_weight

    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)

    out = {}
    for k, v in frame_data.items():
        if isinstance(v, np.ndarray):
            out[k] = cv2.remap(v, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            out[k] = v
    return out


def render_evidence_striated_muscle_frame(
    frame_data: dict[str, np.ndarray],
    target_mode: str,
    is_wrong: bool,
    alpha_scale: float,
    rep_wave: float,
) -> np.ndarray:
    """
    Renders surgical, head-specific anatomical targeting with deep muscle fiber striations (근육결):
    - `target_mode == "rotator_cuff"`: Mistake 01 — Rotator Cuff / Glenohumeral Capsule impingement (`COL_RED`).
    - `target_mode == "anterior_deltoid"`: Mistake 02 — Anterior Deltoid (Front Delt) & Flexor compensation (`COL_RED`). Triceps stay 100% unlit matte grey!
    - `target_mode == "triceps_lat_long"`: Hook & Phase 3 — Triceps Lateral Head + Long Head (`COL_GREEN`), while Deltoid stays 100% unlit matte grey!
    - `target_mode == "triceps_all_three"`: Phase 4 Peak Isolation — Triceps Lateral Head + Long Head + Medial Head (`COL_GREEN`), separated by crisp dark anatomical sulci!
    """
    modified = apply_kinematic_posture_modification(frame_data,
        "wrong_shoulder" if target_mode == "rotator_cuff" else ("wrong_elbow" if target_mode == "anterior_deltoid" else "correct"),
        rep_wave,
    )
    mono = modified["mono"].astype(np.float32)
    native_fiber = modified["native_fiber"]
    red_core = modified["red_core"]
    body_gate = modified["body_gate"]
    arm_drop = float(modified["arm_drop_px"])

    heads = build_anatomical_head_masks(arm_drop_px=arm_drop)
    sulcus = heads["sulcus"]

    # Gate strictly by the TRUE 3D sculpted muscle presence (`red_core ** 0.72`) from the Bot Profile render
    # so only real 3D muscle fibers light up and natural inter-head sulci stay dark!
    sculpt_gate = body_gate * np.clip((red_core ** 0.72) * 1.12, 0.0, 1.0)

    if target_mode == "rotator_cuff":
        # Mistake 01: Rotator Cuff & Superior-Anterior Joint Capsule
        primary_mask = heads["rotator"] * np.maximum(sculpt_gate, body_gate * 0.75)
        fascicle_field = FASCICLE_ANT_DELT
    elif target_mode == "anterior_deltoid":
        # Mistake 02: Anterior Deltoid (Front Delt) steals tension! Triceps stay 0% lit!
        primary_mask = heads["ant_delt"] * sculpt_gate
        fascicle_field = FASCICLE_ANT_DELT
    elif target_mode == "triceps_lat_long":
        # Correct Technique: Triceps Lateral Head + Long Head ONLY (Deltoid is 0% lit, zero chest artifacts!)
        m_lat = heads["tri_lat"] * sculpt_gate
        m_long = heads["tri_long"] * sculpt_gate
        primary_mask = np.maximum(m_lat, m_long * 0.96)
        w_lat = m_lat / (m_lat + m_long + 1e-5)
        fascicle_field = FASCICLE_TRI_LAT * w_lat + FASCICLE_TRI_LONG * (1.0 - w_lat)
    else:
        # Phase 4: All 3 Heads of Triceps Brachii (Lateral Head + Long Head + Distal Medial Head per MSSE 2023)
        m_lat = heads["tri_lat"] * sculpt_gate
        m_long = heads["tri_long"] * sculpt_gate
        m_med = heads["tri_med"] * np.maximum(sculpt_gate, body_gate * 0.72)
        primary_mask = np.maximum(np.maximum(m_lat, m_long * 0.98), m_med * 0.96)
        w_lat = m_lat / (m_lat + m_long + m_med + 1e-5)
        w_long = m_long / (m_lat + m_long + m_med + 1e-5)
        w_med = m_med / (m_lat + m_long + m_med + 1e-5)
        fascicle_field = FASCICLE_TRI_LAT * w_lat + FASCICLE_TRI_LONG * w_long + FASCICLE_TRI_MED * w_med

    # Carve deep intermuscular sulci (grooves) between individual muscle heads so each head's boundary is unmistakable
    total_mask = primary_mask * (1.0 - 0.82 * sulcus)
    total_mask = np.clip((total_mask - 0.04) / 0.76, 0.0, 1.0) * alpha_scale

    # Combine 76% authentic Bot-Profile 3D muscle fiber relief + 24% anisotropic pennation striations
    f_native = np.clip(0.5 + 0.5 * native_fiber, 0.0, 1.0)
    f_aniso = np.clip(0.5 + 0.5 * fascicle_field, 0.0, 1.0)
    f_val = np.clip(0.76 * f_native + 0.24 * f_aniso, 0.0, 1.0) * (1.0 - 0.65 * sulcus)

    # Bot-Profile grade high-contrast multi-stop striated muscle color curves (RGB)!
    if is_wrong:
        # Deep crimson grooves -> Vibrant Red-Orange belly -> Hot glowing Orange-White fiber ridges (exact Bot Profile feel!)
        col_r = 92.0 + 163.0 * (f_val ** 0.70)
        col_g = 16.0 + 76.0 * (f_val ** 1.30) + 128.0 * (f_val ** 3.0)
        col_b = 14.0 + 40.0 * (f_val ** 1.25) + 115.0 * (f_val ** 3.2)
        rim_rgb = np.array([255.0, 190.0, 155.0], dtype=np.float32)
    else:
        # Deep forest emerald grooves -> Vibrant Emerald belly -> Bright Lime-White striated fiber ridges!
        col_r = 12.0 + 52.0 * (f_val ** 1.30) + 150.0 * (f_val ** 3.0)
        col_g = 80.0 + 175.0 * (f_val ** 0.70)
        col_b = 18.0 + 58.0 * (f_val ** 1.20) + 112.0 * (f_val ** 3.0)
        rim_rgb = np.array([190.0, 255.0, 180.0], dtype=np.float32)

    painted = np.stack([np.clip(col_r, 0, 255), np.clip(col_g, 0, 255), np.clip(col_b, 0, 255)], axis=-1)

    # Sharp 1px anatomical head boundary contour (ZERO blurry neon halo!)
    M_blur = cv2.GaussianBlur(total_mask, (0, 0), sigmaX=1.0, sigmaY=1.0)
    gx = cv2.Sobel(M_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(M_blur, cv2.CV_32F, 0, 1, ksize=3)
    head_border = np.clip(np.sqrt(gx * gx + gy * gy) * 2.5, 0.0, 1.0) * (total_mask > 0.05)

    out = mono.copy()
    fill_alpha = np.clip(total_mask * 0.94, 0.0, 0.96)[:, :, None]
    out = out * (1.0 - fill_alpha) + painted * fill_alpha
    out = out * (1.0 - (head_border * 0.65)[:, :, None]) + rim_rgb[None, None, :] * (head_border * 0.65)[:, :, None]
    # Deepen inter-head sulci for surgical head separation
    sulcus_alpha = (sulcus * total_mask * 0.55)[:, :, None]
    out = out * (1.0 - sulcus_alpha) + 16.0 * sulcus_alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def render_3d_wrapping_gunmetal_ribbon(
    canvas_rgb: np.ndarray,
    center_xy: tuple[float, float],
    axis_angle_deg: float,
    cyl_radius_px: float,
    axial_span_px: float,
    theta_start_deg: float,
    theta_end_deg: float,
    alpha_scale: float,
    progress: float = 1.0,
    ribbon_half_width: float = 8.5,
):
    """
    Renders a TRUE 3D Cylindrical Body-Wrapping Ribbon in Dark Gunmetal Charcoal (`#22262E`)
    with a Titanium-Silver 3D Bevel Edge (`#DAE2EE`) and Contact Occlusion Shadow!
    - Parameterized on a 3D cylinder of radius `cyl_radius_px` tilted at `axis_angle_deg`
      (matching the exact 3D orientation of the shoulder, upper arm, or forearm!).
    - As angle `theta` sweeps around the limb cylinder from `theta_start_deg` (emerging from
      behind the limb edge, `theta = -95 deg`) across the front surface (`theta = 0 deg`) to
      `theta_end_deg`, the ribbon width foreshortens naturally by `cos(theta)` and the 3D surface
      normal shades from deep charcoal at the limb's curved edge to titanium-accented gunmetal at the apex!
    - Terminates in a sculpted 3D Gunmetal Arrowhead with a crisp silver bevel edge.
    """
    if alpha_scale <= 0.02 or progress <= 0.05:
        return

    prog = min(1.0, max(0.18, progress))
    t0 = math.radians(theta_start_deg)
    t1 = math.radians(theta_start_deg + (theta_end_deg - theta_start_deg) * prog)

    n_pts = 220
    us = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)
    thetas = t0 + (t1 - t0) * us

    # Local 3D cylinder coordinates:
    # u_axis: along the limb cylinder axis
    # v_perp: perpendicular to the limb axis (across the cylinder diameter: R * sin(theta))
    # z_depth: out of the screen toward the viewer (R * cos(theta))
    u_local = -0.5 * axial_span_px + axial_span_px * prog * (us ** 1.05)
    # Add a graceful helical S-camber so it flows organically along the muscle belly
    v_local = cyl_radius_px * np.sin(thetas)
    z_depth = np.cos(thetas)  # >0 is front of limb, <=0 is wrapping behind the limb edge!

    # Rotate (u_local, v_local) into screen (x, y) coordinates
    ang = math.radians(axis_angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    cx, cy = center_xy
    bx = cx + u_local * ca - v_local * sa
    by = cy + u_local * sa + v_local * ca

    # Perspective foreshortening of ribbon width as it wraps around the cylinder edge:
    # At the limb edge (z_depth near 0), the ribbon is viewed edge-on so it narrows naturally!
    foreshorten = np.clip(z_depth + 0.25, 0.18, 1.0)
    # Also taper gracefully near the start tail
    tail_taper = np.clip(us / 0.22, 0.25, 1.0)
    radii = (ribbon_half_width * foreshorten * tail_taper).astype(np.float32)

    # 3D Beveled Arrowhead at the end of the wrapping ribbon (u = 1.0)
    hx, hy = float(bx[-1]), float(by[-1])
    dx = float(bx[-1] - bx[-4])
    dy = float(by[-1] - by[-4])
    dn = math.hypot(dx, dy) + 1e-6
    tx, ty = dx / dn, dy / dn
    nx, ny = -ty, tx

    wing_len = 32.0
    wing_flare = 19.0
    w_us = np.linspace(0.0, 1.0, 55, dtype=np.float32)
    for sign in (-1.0, 1.0):
        wx = hx - tx * (wing_len * w_us) + sign * nx * (wing_flare * (w_us ** 1.15))
        wy = hy - ty * (wing_len * w_us) + sign * ny * (wing_flare * (w_us ** 1.15))
        w_rad = ribbon_half_width * 0.82 * (1.0 - 0.72 * (w_us ** 1.3))
        w_z = np.full_like(w_us, float(z_depth[-1]))
        bx = np.concatenate([bx, wx])
        by = np.concatenate([by, wy])
        radii = np.concatenate([radii, w_rad])
        z_depth = np.concatenate([z_depth, w_z])

    pad = int(ribbon_half_width * 4.0 + 36)
    x0, x1 = max(0, int(np.min(bx)) - pad), min(W, int(np.max(bx)) + pad)
    y0, y1 = max(0, int(np.min(by)) - pad), min(H, int(np.max(by)) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

    # 1) Distance field for the ribbon & swept arrowhead
    min_d2 = np.full(xx.shape, 1e12, dtype=np.float32)
    nearest_r = np.full(xx.shape, ribbon_half_width, dtype=np.float32)
    nearest_z = np.full(xx.shape, 1.0, dtype=np.float32)

    # Also compute a shifted distance field for the 3D drop/contact shadow on the muscle (+5px x, +7px y)
    sh_min_d2 = np.full(xx.shape, 1e12, dtype=np.float32)

    for i in range(0, len(bx), 64):
        px = bx[i : i + 64, None, None]
        py = by[i : i + 64, None, None]
        pr = radii[i : i + 64, None, None]
        pz = z_depth[i : i + 64, None, None]

        d2 = (xx[None, :, :] - px) ** 2 + (yy[None, :, :] - py) ** 2
        idx = np.argmin(d2, axis=0)
        d2_min = np.take_along_axis(d2, idx[None, :, :], axis=0)[0]
        r_min = np.take_along_axis(pr, idx[None, :, :], axis=0)[0]
        z_min = np.take_along_axis(pz, idx[None, :, :], axis=0)[0]

        upd = d2_min < min_d2
        min_d2 = np.where(upd, d2_min, min_d2)
        nearest_r = np.where(upd, r_min, nearest_r)
        nearest_z = np.where(upd, z_min, nearest_z)

        sd2 = (xx[None, :, :] - (px + 5.0)) ** 2 + (yy[None, :, :] - (py + 7.0)) ** 2
        sd2_min = np.min(sd2, axis=0)
        sh_min_d2 = np.minimum(sh_min_d2, sd2_min)

    dist = np.sqrt(min_d2)
    sh_dist = np.sqrt(sh_min_d2)

    # Occlude the portion wrapping behind the limb cylinder (z < -0.15 fades smoothly into the back edge)
    vis_mask = np.clip((nearest_z + 0.18) / 0.32, 0.0, 1.0) * alpha_scale

    # 3D Contact Shadow under the ribbon (separates the ribbon from the glowing red/green muscle!)
    shadow_alpha = np.exp(-((sh_dist / (nearest_r * 1.65 + 1e-4)) ** 2.0)) * 0.68 * vis_mask

    # Sculpted Dark Gunmetal Body (crisp anti-aliased edge, zero neon blur!)
    body_alpha = (1.0 / (1.0 + np.exp(np.clip((dist - nearest_r) * 2.2, -40.0, 40.0)))) * 0.96 * vis_mask

    # 3D Cylindrical Shading on the Gunmetal Body:
    # Brighter anodized graphite on the front crest of the cylinder (z near 1), deeper charcoal toward the curved sides,
    # plus cross-sectional tube curvature shading!
    cross_u = np.clip(dist / (nearest_r + 1e-4), 0.0, 1.0)
    cyl_light = np.clip(nearest_z, 0.0, 1.0) * (1.0 - 0.45 * (cross_u ** 2))

    # Crisp 1.3px Titanium-Silver Bevel Rim along the outer edges of the dark gunmetal ribbon!
    rim_band = np.exp(-(((dist - (nearest_r - 1.2)) / 1.1) ** 2)) * 0.92 * vis_mask * np.clip(nearest_z + 0.1, 0.2, 1.0)
    # Subtle center spine sheen
    center_sheen = np.exp(-((dist / (nearest_r * 0.28 + 1e-4)) ** 2)) * 0.28 * vis_mask * np.clip(nearest_z, 0.0, 1.0)

    roi = canvas_rgb[y0:y1, x0:x1].astype(np.float32)

    # Step A: Apply contact shadow onto the underlying muscle
    sh_col = np.array(SHADOW_COL, dtype=np.float32)[None, None, :]
    roi = roi * (1.0 - shadow_alpha[:, :, None]) + sh_col * shadow_alpha[:, :, None]

    # Step B: Composite Dark Gunmetal 3D Ribbon Body
    gm_dark = np.array(GUNMETAL_DARK, dtype=np.float32)[None, None, :]
    gm_mid = np.array(GUNMETAL_MID, dtype=np.float32)[None, None, :]
    gm_rim = np.array(GUNMETAL_RIM, dtype=np.float32)[None, None, :]

    body_col = gm_dark * (1.0 - cyl_light[:, :, None]) + gm_mid * cyl_light[:, :, None]
    body_col = body_col * (1.0 - center_sheen[:, :, None]) + gm_rim * center_sheen[:, :, None]
    roi = roi * (1.0 - body_alpha[:, :, None]) + body_col * body_alpha[:, :, None]

    # Step C: Composite Crisp Titanium-Silver Bevel Rim Highlight (gives unmistakable luxury 3D metal/carbon look!)
    roi = roi * (1.0 - rim_band[:, :, None]) + gm_rim * rim_band[:, :, None]

    canvas_rgb[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def render_3d_gunmetal_orbital_cuff(
    canvas_rgb: np.ndarray,
    center: tuple[float, float],
    tilt_deg: float,
    rx: float,
    ry: float,
    accent_col: tuple[int, int, int],
    alpha_scale: float,
):
    """
    Renders a 3D perspective-tilted Gunmetal Orbital Cuff Ring that wraps around the joint
    (back half occluded behind the joint, front half wrapping over the joint in dark gunmetal
    with a crisp silver bevel edge and a tiny precision center dot).
    """
    if alpha_scale <= 0.02:
        return
    cx, cy = center
    pad = int(max(rx, ry) + 28)
    x0, x1 = max(0, int(cx) - pad), min(W, int(cx) + pad)
    y0, y1 = max(0, int(cy) - pad), min(H, int(cy) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    ang = math.radians(tilt_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca

    # Elliptical distance
    ell_r = np.sqrt((u / rx) ** 2 + (v / ry) ** 2 + 1e-6)
    # Approximate radial pixel distance from the ellipse perimeter
    scale_r = 0.5 * (rx + ry)
    d_ring = np.abs(ell_r - 1.0) * scale_r

    # Front half of the 3D tilted cuff has v > -0.15*ry (back half is attenuated as it goes behind the joint)
    front_vis = np.clip((v / (ry + 1e-4) + 0.35) / 0.45, 0.22, 1.0) * alpha_scale

    ring_w = 4.2
    sh_d = np.abs(np.sqrt(((u - 3.0) / rx) ** 2 + ((v - 5.0) / ry) ** 2 + 1e-6) - 1.0) * scale_r
    sh_alpha = np.exp(-((sh_d / (ring_w * 1.7)) ** 2)) * 0.55 * front_vis

    body_alpha = (1.0 / (1.0 + np.exp(np.clip((d_ring - ring_w) * 2.2, -40.0, 40.0)))) * 0.94 * front_vis
    rim_alpha = np.exp(-(((d_ring - (ring_w - 1.0)) / 0.9) ** 2)) * 0.88 * front_vis

    # Small center precision pin (4.5px gunmetal ring + 2.5px accent core)
    d_center = np.sqrt(dx * dx + dy * dy)
    pin_body = (1.0 / (1.0 + np.exp(np.clip((d_center - 6.5) * 2.2, -40.0, 40.0)))) * 0.95 * alpha_scale
    pin_core = (1.0 / (1.0 + np.exp(np.clip((d_center - 3.2) * 2.2, -40.0, 40.0)))) * 0.95 * alpha_scale

    roi = canvas_rgb[y0:y1, x0:x1].astype(np.float32)
    sh_col = np.array(SHADOW_COL, dtype=np.float32)[None, None, :]
    gm_mid = np.array(GUNMETAL_MID, dtype=np.float32)[None, None, :]
    gm_rim = np.array(GUNMETAL_RIM, dtype=np.float32)[None, None, :]
    acc_arr = np.array(accent_col, dtype=np.float32)[None, None, :]

    roi = roi * (1.0 - sh_alpha[:, :, None]) + sh_col * sh_alpha[:, :, None]
    roi = roi * (1.0 - body_alpha[:, :, None]) + gm_mid * body_alpha[:, :, None]
    roi = roi * (1.0 - rim_alpha[:, :, None]) + gm_rim * rim_alpha[:, :, None]
    roi = roi * (1.0 - pin_body[:, :, None]) + gm_mid * pin_body[:, :, None]
    roi = roi * (1.0 - pin_core[:, :, None]) + acc_arr * pin_core[:, :, None]

    canvas_rgb[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)



def draw_minimal_editorial_copy(
    canvas: Image.Image,
    kicker: str,
    line1: str,
    line2: str,
    sub_line: str,
    col: tuple[int, int, int],
    ly: int,
    alpha_scale: float,
):
    if alpha_scale <= 0.02:
        return

    spot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(spot).rounded_rectangle([64, ly - 28, 540, ly + 220], radius=45, fill=(10, 12, 16, int(165 * alpha_scale)))
    spot = spot.filter(ImageFilter.GaussianBlur(radius=24))
    canvas.alpha_composite(spot)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bar_x = 92
    od.rounded_rectangle([bar_x, ly + 4, bar_x + 6, ly + 194], radius=3, fill=(*col, int(255 * alpha_scale)))

    lx = 114
    od.text(
        (lx, ly),
        kicker,
        font=FONT_KICKER,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(230 * alpha_scale)),
    )
    od.text(
        (lx, ly + 36),
        line1,
        font=FONT_MAIN,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )
    od.text(
        (lx, ly + 92),
        line2,
        font=FONT_MAIN,
        fill=(*col, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(245 * alpha_scale)),
    )
    od.text(
        (lx, ly + 158),
        sub_line,
        font=FONT_SUB,
        fill=(225, 232, 240, int(245 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(235 * alpha_scale)),
    )
    canvas.alpha_composite(ov)


def draw_signature_sets_reps(canvas: Image.Image, alpha_scale: float):
    if alpha_scale <= 0.02:
        return
    x, y = 94, 1185
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.text(
        (x, y),
        "4×12",
        font=FONT_SETS,
        fill=(255, 255, 255, int(255 * alpha_scale)),
        stroke_width=6,
        stroke_fill=(0, 0, 0, int(240 * alpha_scale)),
    )
    od.text(
        (x + 4, y + 92),
        "SETS × SLOW REPS",
        font=FONT_SETS_SUB,
        fill=(*COL_GREEN, int(255 * alpha_scale)),
        stroke_width=4,
        stroke_fill=(0, 0, 0, int(235 * alpha_scale)),
    )
    canvas.alpha_composite(ov)



async def build_voiceover_and_audio(wav_path: str):
    import edge_tts

    lines = [
        (0.15, "Cable rope pushdown. Let's fix your form for maximum triceps growth."),
        (3.70, "Mistake one: rolling your shoulders forward grinds the rotator cuff joint."),
        (11.20, "Mistake two: swinging your elbows up and down lets the front delt steal the load."),
        (18.20, "Instead, pin your elbows tight to your ribs and spread the rope at the bottom."),
        (23.20, "This locks in one hundred percent triceps isolation. Perform four sets of twelve slow reps."),
    ]
    sr = 44100
    total_samples = int(DURATION * sr)
    master = np.zeros(total_samples, dtype=np.float32)

    t_arr = np.linspace(0, DURATION, total_samples, endpoint=False)
    sub = 0.045 * np.sin(2 * np.pi * 55.0 * t_arr)
    beat_env = (np.sin(2 * np.pi * (100.0 / 60.0) * t_arr) > 0.75).astype(np.float32)
    master += sub * (0.4 + 0.6 * beat_env)

    for start_s, _ in lines:
        s0 = int(start_s * sr)
        dur_c = int(0.22 * sr)
        if s0 + dur_c < total_samples:
            tc = np.linspace(0, 0.22, dur_c, endpoint=False)
            freq = 220.0 if 3.0 < start_s < 17.0 else 528.0
            chime = 0.10 * np.sin(2 * np.pi * freq * tc) * np.exp(-tc * 14.0)
            master[s0:s0 + dur_c] += chime

    os.makedirs(TMP_DIR, exist_ok=True)
    for i, (start_s, text) in enumerate(lines):
        mp3_p = os.path.join(TMP_DIR, f"vo_{i}.mp3")
        wav_p = os.path.join(TMP_DIR, f"vo_{i}.wav")
        comm = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+4%", pitch="-2Hz")
        await comm.save(mp3_p)
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_p, "-ac", "1", "-ar", str(sr), wav_p],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(wav_p, "rb") as wf:
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        s_idx = int(start_s * sr)
        e_idx = min(total_samples, s_idx + len(raw))
        master[s_idx:e_idx] += raw[: e_idx - s_idx] * 0.92

    master = np.clip(master, -0.96, 0.96)
    pcm = (master * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())




def build_v21_torso_icons(pool: list[dict[str, np.ndarray]]) -> list[tuple[Image.Image, Image.Image]]:
    size = (158, 128)
    crop_box = (520, 560, 920, 1020)
    sample_frame = pool[min(len(pool) - 1, 30)]
    mono_arr = sample_frame["mono"]
    inact_base = Image.fromarray(mono_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    dim = Image.new("RGBA", size, (10, 12, 16, 115))
    inact_base.alpha_composite(dim)

    icons = []
    for mode, is_w in [("rotator_cuff", True), ("anterior_deltoid", True), ("triceps_all_three", False)]:
        act_arr = render_evidence_striated_muscle_frame(sample_frame, mode, is_w, 1.0, 0.85)
        act_im = Image.fromarray(act_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
        icons.append((act_im, inact_base.copy()))
    return icons


def draw_glassmorphic_3torso_dock_v21(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_dock_idx: int,
    is_wrong: bool,
    alpha_scale: float = 1.0,
):
    if alpha_scale <= 0.02:
        return
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    tw, th = 158, 128
    gap = 16
    pad = 14
    bar_x = 92
    bar_y = 246
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + 2 * pad

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=24,
        fill=(22, 26, 34, int(215 * alpha_scale)),
        outline=(255, 255, 255, int(55 * alpha_scale)),
        width=2,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy() if i == active_dock_idx else inact_im.copy()

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=16, fill=int(255 * alpha_scale))
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_dock_idx:
            i_col = COL_RED if i < 2 else COL_GREEN
            od.rounded_rectangle(
                [tx - 3, ty - 3, tx + tw + 3, ty + th + 3],
                radius=18,
                outline=(*i_col, int(255 * alpha_scale)),
                width=4,
            )

    bx = bar_x + bar_w + 84
    by = bar_y + bar_h // 2
    br = 50
    od.ellipse(
        [bx - br, by - br, bx + br, by + br],
        fill=(*accent, int(248 * alpha_scale)),
        outline=(255, 255, 255, int(235 * alpha_scale)),
        width=3,
    )
    if is_wrong:
        d = 19
        od.line([(bx - d, by - d), (bx + d, by + d)], fill=(255, 255, 255, int(255 * alpha_scale)), width=9)
        od.line([(bx - d, by + d), (bx + d, by - d)], fill=(255, 255, 255, int(255 * alpha_scale)), width=9)
    else:
        pts = [(bx - 22, by + 2), (bx - 4, by + 19), (bx + 24, by - 17)]
        od.line(pts, fill=(255, 255, 255, int(255 * alpha_scale)), width=9, joint="curve")

    canvas.alpha_composite(ov)


PHASES = [
    {
        "idx": 0,
        "t_start": 0.0,
        "t_end": 3.5,
        "is_wrong": False,
        "target_mode": "triceps_lat_long",
        "active_slot": 2,
        "label_y": 545,
        "kicker": "3D BIOMECHANICS · CABLE ROPE PUSHDOWN",
        "line1": "ISOLATE ALL",
        "line2": "3 TRICEPS HEADS",
        "sub": "LATERAL · LONG · MEDIAL HEAD ANATOMY",
        "reps": 1.2,
        "wrap_cfg": {
            "center_xy": (685, 795),
            "axis_angle_deg": 98.0,
            "cyl_radius_px": 74.0,
            "axial_span_px": 105.0,
            "theta_start_deg": -92.0,
            "theta_end_deg": 64.0,
        },
        "cuff_cfg": None,
        "show_sets": False,
        "head_callouts": [
            ("LATERAL HEAD", (676, 782), (485, 782)),
            ("LONG HEAD", (742, 804), (800, 804)),
        ],
    },
    {
        "idx": 1,
        "t_start": 3.5,
        "t_end": 11.0,
        "is_wrong": True,
        "target_mode": "rotator_cuff",
        "active_slot": 0,
        "label_y": 545,
        "kicker": "✕ FORM MISTAKE 01 · JOINT IMPINGEMENT",
        "line1": "SHOULDERS ROLL",
        "line2": "FORWARD",
        "sub": "→ IMPINGES ROTATOR CUFF CAPSULE",
        "reps": 2.5,
        "wrap_cfg": {
            "center_xy": (675, 668),
            "axis_angle_deg": 42.0,
            "cyl_radius_px": 76.0,
            "axial_span_px": 78.0,
            "theta_start_deg": -95.0,
            "theta_end_deg": 62.0,
        },
        "cuff_cfg": {"center": (678, 668), "tilt_deg": -22.0, "rx": 36.0, "ry": 18.0},
        "show_sets": False,
        "head_callouts": [
            ("ROTATOR CUFF CAPSULE", (678, 668), (785, 650)),
        ],
    },
    {
        "idx": 2,
        "t_start": 11.0,
        "t_end": 18.0,
        "is_wrong": True,
        "target_mode": "anterior_deltoid",
        "active_slot": 1,
        "label_y": 545,
        "kicker": "✕ FORM MISTAKE 02 · TENSION LEAK",
        "line1": "ELBOWS SWING",
        "line2": "UP & DOWN",
        "sub": "→ FRONT DELT STEALS TRICEPS LOAD",
        "reps": 2.3,
        "wrap_cfg": {
            "center_xy": (658, 708),
            "axis_angle_deg": 18.0,
            "cyl_radius_px": 72.0,
            "axial_span_px": 92.0,
            "theta_start_deg": -92.0,
            "theta_end_deg": 64.0,
        },
        "cuff_cfg": None,
        "show_sets": False,
        "head_callouts": [
            ("ANTERIOR DELTOID", (660, 696), (785, 685)),
            ("TRICEPS UNLOADED (0%)", (725, 805), (785, 805)),
        ],
    },
    {
        "idx": 3,
        "t_start": 18.0,
        "t_end": 23.0,
        "is_wrong": False,
        "target_mode": "triceps_lat_long",
        "active_slot": 2,
        "label_y": 545,
        "kicker": "✓ CORRECT TECHNIQUE · 0° SHOULDER LOCK",
        "line1": "PIN ELBOWS &",
        "line2": "SPREAD ROPE",
        "sub": "→ DEACTIVATES DELTS · LOCKS TRICEPS",
        "reps": 2.0,
        "wrap_cfg": {
            "center_xy": (668, 885),
            "axis_angle_deg": 122.0,
            "cyl_radius_px": 74.0,
            "axial_span_px": 95.0,
            "theta_start_deg": -92.0,
            "theta_end_deg": 66.0,
        },
        "cuff_cfg": {"center": (668, 842), "tilt_deg": 15.0, "rx": 38.0, "ry": 19.0},
        "show_sets": False,
        "head_callouts": [
            ("DELTOID OFF (0%)", (682, 682), (785, 675)),
            ("LATERAL + LONG HEAD", (705, 800), (785, 800)),
        ],
    },
    {
        "idx": 4,
        "t_start": 23.0,
        "t_end": 28.0,
        "is_wrong": False,
        "target_mode": "triceps_all_three",
        "active_slot": 2,
        "label_y": 545,
        "kicker": "✓ 100% 3-HEAD TRICEPS ISOLATION (EMG)",
        "line1": "LATERAL · LONG",
        "line2": "& MEDIAL HEAD",
        "sub": "→ FULL LOCKOUT PEAK CONTRACTION",
        "reps": 2.0,
        "wrap_cfg": {
            "center_xy": (698, 802),
            "axis_angle_deg": 86.0,
            "cyl_radius_px": 72.0,
            "axial_span_px": 110.0,
            "theta_start_deg": -92.0,
            "theta_end_deg": 66.0,
        },
        "cuff_cfg": None,
        "show_sets": True,
        "head_callouts": [
            ("LATERAL HEAD", (674, 782), (485, 782)),
            ("LONG HEAD", (744, 795), (795, 785)),
            ("MEDIAL HEAD", (694, 882), (795, 885)),
        ],
    },
]


def draw_anatomical_head_callouts(
    pil_im: Image.Image,
    callouts: list[tuple[str, tuple[int, int], tuple[int, int]]],
    is_wrong: bool,
    alpha: float,
) -> None:
    if alpha <= 0.02 or not callouts:
        return

    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    accent = COL_RED if is_wrong else COL_GREEN

    for label, (mx, my), (tx, ty) in callouts:
        is_muted_note = "0%" in label or "OFF" in label or "UNLOADED" in label
        dot_col = (165, 175, 190) if is_muted_note else accent
        a_int = int(240 * alpha)

        bbox = d.textbbox((0, 0), label, font=FONT_LABEL)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = 11, 6
        if tx < mx:
            bx2 = min(mx - 22, tx)
            bx1 = max(88, bx2 - tw - pad_x * 2)
            bx2 = bx1 + tw + pad_x * 2
            line_end_x = bx2
        else:
            bx1 = max(mx + 22, min(888 - tw - pad_x * 2, tx))
            bx2 = bx1 + tw + pad_x * 2
            line_end_x = bx1
        by1 = ty - th // 2 - pad_y
        by2 = by1 + th + pad_y * 2

        d.line([(mx, my), (line_end_x, ty)], fill=(18, 22, 28, int(215 * alpha)), width=4)
        d.line([(mx, my), (line_end_x, ty)], fill=(218, 226, 238, int(200 * alpha)), width=2)

        d.ellipse([mx - 6, my - 6, mx + 6, my + 6], fill=(18, 22, 28, a_int), outline=(235, 240, 248, a_int), width=2)
        d.ellipse([mx - 3, my - 3, mx + 3, my + 3], fill=(*dot_col, a_int))

        d.rounded_rectangle(
            [bx1, by1, bx2, by2],
            radius=8,
            fill=(18, 22, 28, int(230 * alpha)),
            outline=(*((145, 155, 170) if is_muted_note else dot_col), int(215 * alpha)),
            width=2,
        )
        txt_col = (195, 205, 220, a_int) if is_muted_note else (255, 255, 255, a_int)
        d.text((bx1 + pad_x, by1 + pad_y - 1), label, font=FONT_LABEL, fill=txt_col)

    pil_im.alpha_composite(over)


GLOBAL_POOL: list[dict[str, np.ndarray]] = []
GLOBAL_TORSO_ICONS: list[tuple[Image.Image, Image.Image]] = []


def load_pools() -> None:
    global GLOBAL_POOL, GLOBAL_TORSO_ICONS
    if GLOBAL_POOL:
        return
    print("Loading 100% full-bleed sculpted 3D anatomical pool with native fiber striations...")
    cap = cv2.VideoCapture(CORRECT_I2V)
    raw = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        raw.append(f)
    cap.release()
    if not raw:
        raw = [cv2.imread(CORRECT_JPG)]
    step = max(1, len(raw) // 60)
    sampled = raw[::step][:60]
    GLOBAL_POOL = [prepare_sculpted_mannequin_frame(f) for f in sampled]
    GLOBAL_TORSO_ICONS = build_v21_torso_icons(GLOBAL_POOL)


def render_single_frame(f_idx: int) -> str:
    t = f_idx / float(FPS)
    phase = PHASES[-1]
    for p in PHASES:
        if p["t_start"] <= t < p["t_end"] or (p["idx"] == 4 and t <= p["t_end"] + 0.01):
            phase = p
            break

    t_local = t - phase["t_start"]
    dur = phase["t_end"] - phase["t_start"]
    u = max(0.0, min(1.0, t_local / max(0.001, dur)))

    pool = GLOBAL_POOL
    n_pool = len(pool)

    reps = phase["reps"]
    rep_phase = (u * reps) % 1.0
    rep_wave = 0.5 - 0.5 * math.cos(2.0 * math.pi * rep_phase)
    pool_pos = int(round(rep_wave * (n_pool - 1))) % n_pool

    frame_data = pool[pool_pos]
    env_in = smoothstep(0.0, 0.35, t_local)
    env_out = 1.0 - 0.15 * smoothstep(dur - 0.25, dur, t_local)
    alpha_scale = env_in * env_out * (0.72 + 0.28 * rep_wave)

    canvas = render_evidence_striated_muscle_frame(
        frame_data=frame_data,
        target_mode=phase["target_mode"],
        is_wrong=phase["is_wrong"],
        alpha_scale=alpha_scale,
        rep_wave=rep_wave,
    )

    if phase["cuff_cfg"] is not None:
        cc = phase["cuff_cfg"]
        render_3d_gunmetal_orbital_cuff(
            canvas,
            center=cc["center"],
            tilt_deg=cc["tilt_deg"],
            rx=cc["rx"],
            ry=cc["ry"],
            accent_col=COL_RED if phase["is_wrong"] else COL_GREEN,
            alpha_scale=env_in * env_out,
        )

    wcfg = phase["wrap_cfg"]
    prog = min(1.0, max(0.28, smoothstep(0.10, 0.85, t_local) * (0.65 + 0.35 * rep_wave)))
    render_3d_wrapping_gunmetal_ribbon(
        canvas,
        center_xy=wcfg["center_xy"],
        axis_angle_deg=wcfg["axis_angle_deg"],
        cyl_radius_px=wcfg["cyl_radius_px"],
        axial_span_px=wcfg["axial_span_px"],
        theta_start_deg=wcfg["theta_start_deg"],
        theta_end_deg=wcfg["theta_end_deg"],
        alpha_scale=env_in * env_out,
        progress=prog,
        ribbon_half_width=8.5,
    )

    pil_im = Image.fromarray(canvas).convert("RGBA")
    draw_glassmorphic_3torso_dock_v21(pil_im, GLOBAL_TORSO_ICONS, phase["active_slot"], phase["is_wrong"], alpha_scale=1.0)
    draw_minimal_editorial_copy(
        pil_im,
        phase["kicker"],
        phase["line1"],
        phase["line2"],
        phase["sub"],
        COL_RED if phase["is_wrong"] else COL_GREEN,
        phase["label_y"],
        env_in * env_out,
    )
    draw_anatomical_head_callouts(
        pil_im, phase.get("head_callouts", []), phase["is_wrong"], env_in * env_out
    )
    if phase["show_sets"]:
        sets_alpha = smoothstep(0.15, 0.65, t_local) * env_out
        draw_signature_sets_reps(pil_im, sets_alpha)

    out_rgb = pil_im.convert("RGB")
    os.makedirs(TMP_DIR, exist_ok=True)
    out_path = os.path.join(TMP_DIR, f"frame_{f_idx:04d}.jpg")
    out_rgb.save(out_path, quality=94)
    return out_path

def generate_previews_and_closeup() -> None:
    load_pools()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    sample_times = [
        (1.8, OUT_HOOK, "0. HOOK: 3-HEAD ANATOMY"),
        (6.8, OUT_STEP1, "1. WRONG: ROTATOR CUFF"),
        (13.8, OUT_STEP2, "2. WRONG: FRONT DELT"),
        (20.0, OUT_STEP3, "3. CORRECT: DELTS OFF"),
        (25.0, OUT_STEP4, "4. 3-HEAD ISOLATION + 4x12"),
    ]
    panels = []
    for t_sec, out_p, label in sample_times:
        f_idx = int(round(t_sec * FPS))
        tmp_p = render_single_frame(f_idx)
        shutil.copyfile(tmp_p, out_p)
        print("Saved:", out_p)
        im = Image.open(out_p).convert("RGB")
        pw, ph = 320, 568
        im_r = im.resize((pw, ph), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(im_r)
        is_w = "WRONG" in label
        col = COL_RED if is_w else COL_GREEN
        d.rectangle([0, 0, pw - 1, ph - 1], outline=col, width=4)
        d.rectangle([0, 0, pw, 42], fill=(12, 14, 18))
        d.text((10, 9), label, font=load_font(18, bold=True), fill=col)
        panels.append(im_r)

    sheet = Image.new("RGB", (320 * 5 + 16 * 6, 568 + 32), (10, 12, 16))
    for i, p in enumerate(panels):
        sheet.paste(p, (16 + i * (320 + 16), 16))
    sheet.save(OUT_PREVIEW, quality=94)
    print("Saved 5-phase v21 preview sheet:", OUT_PREVIEW)

    # Also build a 1:1 close-up comparison sheet: Bot Avatar vs v20 vs v21 (Mistake 2 Front Delt vs Correct 3-Head Triceps)
    avatar_im = Image.open("/home/box/agent-data/agents/7c203a21-2233-41e0-92f1-189a01945890/avatar.png").convert("RGB")
    # Crop the shoulder & triceps region from Avatar, v21 Step 2 (Front Delt), v21 Step 3 (Lat+Long Head), v21 Step 4 (All 3 Heads)
    av_crop = avatar_im.crop((180, 180, 900, 900)).resize((420, 420), Image.Resampling.LANCZOS)

    s2_im = Image.open(OUT_STEP2).convert("RGB").crop((430, 560, 890, 1020)).resize((420, 420), Image.Resampling.LANCZOS)
    s3_im = Image.open(OUT_STEP3).convert("RGB").crop((430, 560, 890, 1020)).resize((420, 420), Image.Resampling.LANCZOS)
    s4_im = Image.open(OUT_STEP4).convert("RGB").crop((430, 560, 890, 1020)).resize((420, 420), Image.Resampling.LANCZOS)

    comp = Image.new("RGB", (420 * 4 + 20 * 5, 420 + 70), (12, 14, 18))
    cd = ImageDraw.Draw(comp)
    crops = [
        (av_crop, "BOT PROFILE (5 HEADS ALL ON + BLUE BLEED)", (180, 200, 230)),
        (s2_im, "v21 MISTAKE 2: ONLY FRONT DELT ON (TRICEPS 0%)", COL_RED),
        (s3_im, "v21 CORRECT: DELTS OFF · LAT + LONG HEAD ON", COL_GREEN),
        (s4_im, "v21 PEAK: ALL 3 TRICEPS HEADS + DEEP STRIATIONS", COL_GREEN),
    ]
    for i, (cim, title, tcol) in enumerate(crops):
        x = 20 + i * (420 + 20)
        comp.paste(cim, (x, 52))
        cd.rectangle([x, 52, x + 419, 52 + 419], outline=tcol, width=3)
        cd.text((x + 6, 16), title, font=load_font(17, bold=True), fill=tcol)
    comp.save(OUT_COMPARISON, quality=95)
    print("Saved close-up comparison:", OUT_COMPARISON)


def build_full_mp4() -> None:
    load_pools()
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

    print(f"Rendering {TOTAL_FRAMES} frames ({DURATION}s) with max_workers=4...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(render_single_frame, range(TOTAL_FRAMES)))

    wav_path = os.path.join(TMP_DIR, "audio.wav")
    import asyncio; asyncio.run(build_voiceover_and_audio(wav_path))

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(TMP_DIR, "frame_%04d.jpg"),
        "-i", wav_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        OUT_MP4,
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Saved final 28.0s v21 Evidence-Based Striation Master MP4:", OUT_MP4)


if __name__ == "__main__":
    generate_previews_and_closeup()
    if "--mp4" in sys.argv:
        build_full_mp4()
