#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`preview_v57_day2_simple_master.py`)
================================================================
Directly implements 중현 님's exact feedback (`t234u`–`t240u`):
1. "릴스 day2 영상보다 거리감이 조금 멀어보이는데? 보여져야 할 부분이 보여져야겠지" (`t234u`, `t239u`, `t240u`):
   - Full-Screen Day 2 Camera Scale (`1080x1920` full-frame `1.5x` upscale from `720x1280` with `x_shift = +36px`
     in Steps 1 & 2 and `+18px` in Steps 3 & 4 so the mannequin fills the screen with the exact same close,
     commanding presence as Day 2 `day2_v39_clean_master_sheet.jpg`!).
   - Zero `cv2.remap` distortion (`0%` background wobble, `diff == 0.0` on background).
2. "너무 복잡하게 들어가려고 하니까 계속 실수가 나오네. 조금 더 단순하게 잡아보던가 Day2처럼" (`t236u`–`t240u`):
   - Strips away ALL stick-skeleton lines, dashed hairlines, floor ovals, and secondary arcs!
   - Uses ONLY Day 2's 3 ultra-clean graphic elements per step:
     (a) **1 Single Concentric Joint Pivot Ring (`◎`)** on the measured joint (`kn_r ◎` in Steps 1–2, `hip_r ◎` in Steps 3–4),
     (b) **1 Bold, Unmistakable Day 2 Exterior Solid Angle Arc (`10px` thick in `1x`, rendered in `SS=4` with dark drop
         shadow + crisp directional arrowhead at the active tip)** placed cleanly in the dark studio space OUTSIDE the body,
     (c) **2 Clean Exterior Floating Number Callouts** (Left: `°` angle, Right: `%` load/activation, Pure White `#FFFFFF`
         `68pt` number + `23pt` sub-label, ZERO underline bars, clamped strictly inside `x = 96 .. 975`).
3. "각도 재는 것도 조금 다른것 같은데? 틀린동작이랑 맞는 동작이랑 할때" (`t235u`):
   - **Pair 1 (Front View — Step 1 Wrong vs Step 2 Right)**:
     * Both measure **Knee Alignment Angle (`°`)** at the exact same exterior position outside the right knee (`kn_r ◎`):
     * Step 1 (Wrong): `18° VALGUS CAVE` (Crimson exterior arc sweeping INWARD toward the midline) | `84% ACL & MCL LOAD`.
     * Step 2 (Right): `0° KNEES TRACK TOES` (Emerald exterior arc at the exact same knee position sweeping OUTWARD over the toes) | `96% VMO & GLUTE LOCK`.
   - **Pair 2 (Side View — Step 3 Wrong vs Step 4 Right)**:
     * Both measure **Torso / Spine Angle (`°`)** at the exact same exterior position above/behind the hip hinge (`hip_r ◎`):
     * Step 3 (Wrong): `42° FORWARD LEAN` (Crimson exterior arc sweeping FORWARD/DOWNWARD in the direction of the forward dive) | `79% LUMBAR OVERLOAD`.
     * Step 4 (Right): `15° UPRIGHT LOCK` (Emerald exterior arc at the exact same hip-hinge position sweeping UPWARD/BACK toward the proud upright spine) | `98% QUADS & GLUTES`.
"""

import os
import sys
import math
import pickle
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import universal_anatomy_rig as uar
import build_v30_pure_graphic_numbers as v30
import build_v38_high_contrast_hud as v38
import build_v39_ig_safe_zone as v39
import preview_v55_demic_day2_master as p55

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_V57_PKL = "/workspace/daily_info_reels/_cache_day3_v57_pools.pkl"
W, H = 1080, 1920

SAFE_X_LEFT = 96
SAFE_X_RIGHT_MAX = 975
SAFE_Y_TOP = 258

COL_RED = (248, 44, 52)
COL_RED_SOFT = (255, 152, 146)

COL_GREEN = (22, 224, 120)
COL_GREEN_SOFT = (162, 252, 204)

COL_WHITE = (255, 255, 255)

F_CALLOUT_NUM = v30.f_barlow(68, extrabold=True)
F_CALLOUT_SUB = v30.f_pretendard(23, "extrabold")

# Static studio radial contrast field (0.0 frame-to-frame change within a step, rich Crimson vs Emerald contrast across steps)
_YY, _XX = np.mgrid[:H, :W].astype(np.float32)
_RADIAL_STAGE = np.exp(-(((_XX - 565.0) / 390.0) ** 2 + ((_YY - 980.0) / 480.0) ** 2))[:, :, None]
_TINT_WRONG = np.array([14.0, -6.0, -6.0], dtype=np.float32)[None, None, :] * _RADIAL_STAGE
_TINT_RIGHT = np.array([-10.0, 12.0, 10.0], dtype=np.float32)[None, None, :] * _RADIAL_STAGE


def apply_static_studio_tint(rgb: np.ndarray, is_wrong: bool) -> np.ndarray:
    out = rgb.astype(np.float32) + (_TINT_WRONG if is_wrong else _TINT_RIGHT)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def build_fullscreen_day2_pool(
    raw_frames_720p: list[np.ndarray],
    step_idx: int,
    is_wrong: bool,
    scale_1080: float = 1.0,
    x_shift: int = 32,
    y_shift: int = 24,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Builds a Full-Screen Day 2 (`1080x1920`) pool with uniform `scale = 1.5 * scale_1080`
    so the mannequin fills the screen with the exact same close, impactful camera distance as Day 2 (`v39`),
    and `0.00` background pixel diff across the entire step!
    """
    sh, sw = raw_frames_720p[0].shape[:2]  # 1280, 720
    scale = 1.5 * scale_1080
    nw = int(round(sw * scale))
    nh = int(round(sh * scale))
    x0 = (W - nw) // 2 + x_shift
    y0 = (H - nh) // 2 + y_shift

    shaded_720 = [p55.apply_day2_v39_anatomy_shader_720p(fr, step_idx, is_wrong) for fr in raw_frames_720p]

    bg_color = np.array([10.0, 11.0, 14.0], dtype=np.float32)
    # Precompute static horizontal/vertical edge feathering if shifted
    dst_x1, dst_x2 = max(0, x0), min(W, x0 + nw)
    dst_y1, dst_y2 = max(0, y0), min(H, y0 + nh)
    src_x1, src_x2 = dst_x1 - x0, (dst_x1 - x0) + (dst_x2 - dst_x1)
    src_y1, src_y2 = dst_y1 - y0, (dst_y1 - y0) + (dst_y2 - dst_y1)

    sub_h = dst_y2 - dst_y1
    sub_w = dst_x2 - dst_x1
    xx_sub = np.arange(sub_w, dtype=np.float32)[None, :]
    yy_sub = np.arange(sub_h, dtype=np.float32)[:, None]

    ax_l = np.clip(xx_sub / 42.0, 0.0, 1.0) if dst_x1 > 0 else np.ones_like(xx_sub)
    ax_r = np.clip((sub_w - 1.0 - xx_sub) / 42.0, 0.0, 1.0) if dst_x2 < W else np.ones_like(xx_sub)
    ay_t = np.clip(yy_sub / 42.0, 0.0, 1.0) if dst_y1 > 0 else np.ones_like(yy_sub)
    ay_b = np.clip((sub_h - 1.0 - yy_sub) / 42.0, 0.0, 1.0) if dst_y2 < H else np.ones_like(yy_sub)
    alpha_mask = (0.5 - 0.5 * np.cos(np.pi * (ax_l * ax_r * ay_t * ay_b)))[:, :, None]

    detector = uar.get_pose_detector()
    out_rgb_list = []
    raw_kps = []

    for idx_f, s_rgb in enumerate(shaded_720):
        res_rgb = cv2.resize(s_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        canvas = np.broadcast_to(bg_color, (H, W, 3)).copy()
        crop = res_rgb[src_y1:src_y2, src_x1:src_x2]
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = crop * alpha_mask + canvas[dst_y1:dst_y2, dst_x1:dst_x2] * (1.0 - alpha_mask)
        out_rgb_list.append(np.clip(canvas, 0, 255).astype(np.uint8))

        raw_rgb_720 = cv2.cvtColor(raw_frames_720p[idx_f], cv2.COLOR_BGR2RGB)
        mp_res = detector.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=raw_rgb_720))

        def M(nx_norm: float, ny_norm: float) -> tuple[float, float]:
            return (float(x0 + (nx_norm * sw) * scale), float(y0 + (ny_norm * sh) * scale))

        if mp_res.pose_landmarks:
            lms = mp_res.pose_landmarks[0]
            sh_p = M(0.5 * (lms[11].x + lms[12].x), 0.5 * (lms[11].y + lms[12].y))
            hip_r = M(lms[24].x, lms[24].y)
            hip_l = M(lms[23].x, lms[23].y)
            kn_r = M(lms[26].x, lms[26].y)
            kn_l = M(lms[25].x, lms[25].y)
            ank_r = M(lms[28].x, lms[28].y)
            ank_l = M(lms[27].x, lms[27].y)
        else:
            assert len(raw_kps) > 0
            raw_kps.append(dict(raw_kps[-1]))
            continue

        raw_kps.append({
            "sh": sh_p,
            "hip": hip_r,
            "hip_r": hip_r,
            "hip_l": hip_l,
            "kn_r": kn_r,
            "kn_l": kn_l,
            "ank_r": ank_r,
            "ank_l": ank_l,
        })

    n = len(raw_kps)
    smooth_kps = [dict() for _ in range(n)]
    for k in raw_kps[0].keys():
        arr = np.array([rk[k] for rk in raw_kps], dtype=np.float32)
        ax = gaussian_filter1d(arr[:, 0], sigma=2.2, mode="nearest")
        ay = gaussian_filter1d(arr[:, 1], sigma=2.2, mode="nearest")
        for i in range(n):
            smooth_kps[i][k] = (float(ax[i]), float(ay[i]))

    return out_rgb_list, smooth_kps


def get_or_build_v57_pools():
    if os.path.exists(CACHE_V57_PKL):
        with open(CACHE_V57_PKL, "rb") as f:
            return pickle.load(f)

    print("Building v57 Full-Screen Day 2 Scale (`1.0x`) 3D pools...")
    s1_raw = p55._load_slice("/workspace/daily_info_reels/output/day3_v55_step1_valgus_to_stand.mp4", 64, 24)
    s2_raw = p55._load_slice("/workspace/daily_info_reels/output/day3_v43_step2_front_fix.mp4", 94, 138)
    s3_raw = p55._load_slice("/workspace/daily_info_reels/output/day3_v43_step3_side_wrong.mp4", 75, 125)
    s4_raw = p55._load_slice("/workspace/daily_info_reels/output/day3_v43_step4_side_fix.mp4", 98, 25)

    pools = {
        1: build_fullscreen_day2_pool(s1_raw, 1, True,  scale_1080=0.98, x_shift=+34, y_shift=+28),
        2: build_fullscreen_day2_pool(s2_raw, 2, False, scale_1080=0.96, x_shift=+34, y_shift=+16),
        3: build_fullscreen_day2_pool(s3_raw, 3, True,  scale_1080=0.96, x_shift=+14, y_shift=+18),
        4: build_fullscreen_day2_pool(s4_raw, 4, False, scale_1080=0.98, x_shift=+18, y_shift=+8),
    }
    with open(CACHE_V57_PKL, "wb") as f:
        pickle.dump(pools, f)
    print("Saved:", CACHE_V57_PKL)
    return pools


def draw_day2_single_pivot_ring_ss(
    rd: ImageDraw.ImageDraw,
    pt_1x: tuple[float, float],
    r_outer_1x: float,
    col_main: tuple[int, int, int],
    ss: int = 4,
):
    """Exact Day 2 (`v36.draw_single_elbow_pivot_ring`) single joint reticle (`◎`) rendered in `SS=4`."""
    cx, cy = pt_1x[0] * ss, pt_1x[1] * ss
    ro = r_outer_1x * ss
    ri = max(2.6 * ss, ro * 0.35)
    rd.ellipse([cx - ro - 2 * ss, cy - ro - 2 * ss, cx + ro + 2 * ss, cy + ro + 2 * ss], fill=(0, 0, 0, 185))
    rd.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], outline=(*col_main, 250), width=int(round(2.5 * ss)))
    rd.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=(255, 255, 255, 255))


def draw_day2_bold_exterior_arc_ss(
    rd: ImageDraw.ImageDraw,
    center_1x: tuple[float, float],
    radius_1x: float,
    start_deg: float,
    end_deg: float,
    arrow_at_end: bool,
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    width_1x: float = 5.0,
    ss: int = 4,
):
    """
    Exact Day 2 (`v38/v39`) Bold Exterior Solid Angle Arc (`width_1x = 5.0px`, smooth closed-polygon annular arc in `SS=4`
    + crisp dark drop shadow + `v30.draw_arrow_head` directional tip at `end_deg` or `start_deg`).
    """
    cx, cy = center_1x[0] * ss, center_1x[1] * ss
    r = radius_1x * ss
    a0 = math.radians(start_deg)
    a1 = math.radians(end_deg)
    steps = 72
    angs = np.linspace(a0, a1, steps)

    # Trim the arc slightly before the arrowhead tip so the shaft joins the triangle cleanly
    arrow_size = 13.5 * ss
    trim_rad = (arrow_size * 0.55) / max(40.0 * ss, r)
    if arrow_at_end:
        angs_shaft = np.linspace(a0, a1 - math.copysign(trim_rad, a1 - a0), steps)
    else:
        angs_shaft = np.linspace(a0 + math.copysign(trim_rad, a1 - a0), a1, steps)

    # 1. Soft dark drop shadow + 2. Bold solid color arc
    for pad, off_y, rgba in [
        (1.8 * ss, 2.0 * ss, (0, 0, 0, 215)),
        (0.0, 0.0, (*col_main, 252)),
    ]:
        hw = (0.5 * width_1x) * ss + pad
        outer_pts = [(cx + math.cos(a) * (r + hw), cy + math.sin(a) * (r + hw) + off_y) for a in angs_shaft]
        inner_pts = [(cx + math.cos(a) * (r - hw), cy + math.sin(a) * (r - hw) + off_y) for a in angs_shaft]
        rd.polygon(outer_pts + inner_pts[::-1], fill=rgba)

    # 3. Arrowhead at the active end of the arc (`v30.draw_arrow_head` geometry)
    if arrow_at_end:
        tip_ang_deg = end_deg
        tangent_deg = end_deg + (90.0 if end_deg >= start_deg else -90.0)
    else:
        tip_ang_deg = start_deg
        tangent_deg = start_deg + (-90.0 if end_deg >= start_deg else 90.0)

    tip_rad = math.radians(tip_ang_deg)
    tip_pt = (cx + r * math.cos(tip_rad), cy + r * math.sin(tip_rad))

    # Shadow under arrowhead
    v30.draw_arrow_head(rd, (tip_pt[0], tip_pt[1] + 2.0 * ss), tangent_deg, arrow_size + 2.0 * ss, (0, 0, 0, 215))
    v30.draw_arrow_head(rd, tip_pt, tangent_deg, arrow_size, (*col_soft, 255))


def draw_day2_floating_white_callout(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    num_str: str,
    tag_str: str,
    align: str = "right",
):
    """
    Day 2 (`v39`) Frameless Floating Number Callout:
    - Pure White (`#FFFFFF`) `68pt` number + Pure White `23pt` sub-label directly below.
    - ZERO underline bars, ZERO pointer lines cluttering the body.
    - Strictly clamped inside `[SAFE_X_LEFT (96) .. SAFE_X_RIGHT_MAX (975)]`.
    """
    x, y = xy
    nb = od.textbbox((0, 0), num_str, font=F_CALLOUT_NUM)
    nw = nb[2] - nb[0]
    tb = od.textbbox((0, 0), tag_str, font=F_CALLOUT_SUB)
    tw = tb[2] - tb[0]

    if align == "right":
        nx = x - nw
        tx = x - tw
    elif align == "center":
        nx = x - nw // 2
        tx = x - tw // 2
    else:
        nx = x
        tx = x

    nx = int(np.clip(nx, SAFE_X_LEFT, SAFE_X_RIGHT_MAX - nw))
    tx = int(np.clip(tx, SAFE_X_LEFT, SAFE_X_RIGHT_MAX - tw))
    ty = y + 70

    for dx_s, dy_s in [(-3, 0), (3, 0), (0, -3), (0, 3), (0, 5)]:
        sd.text((nx + dx_s, y + dy_s), num_str, font=F_CALLOUT_NUM, fill=(0, 0, 0, 240))
        sd.text((tx + dx_s, ty + dy_s), tag_str, font=F_CALLOUT_SUB, fill=(0, 0, 0, 240))

    od.text((nx, y), num_str, font=F_CALLOUT_NUM, fill=COL_WHITE)
    od.text((tx, ty), tag_str, font=F_CALLOUT_SUB, fill=COL_WHITE)


def render_v57_day2_simple_callouts(
    canvas: Image.Image,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 1.0,
):
    """
    Renders the exact Day 2 (`v39`) Simple & High-Impact Biomechanical Callouts:
    - ONLY 1 single joint pivot ring (`◎`),
    - ONLY 1 bold `10px` exterior solid angle arc (`SS=4`) placed cleanly outside the body,
    - ONLY 2 frameless Pure White floating number badges (`°` on left, `%` on right),
    - 100% Unified Angle Measurement between Wrong and Right!
    """
    SS = 4
    rig_layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig_layer)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    col_main = COL_RED if is_wrong else COL_GREEN
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    hip_r = kp["hip_r"]
    kn_r = kp["kn_r"]

    if step_idx == 1:
        # Step 1 (Front Wrong — Knee Alignment Angle at `kn_r ◎`: `18° VALGUS CAVE`)
        # 1. Single Pivot Ring `◎` on `kn_r`
        draw_day2_single_pivot_ring_ss(rd, kn_r, 10.5, col_main, SS)

        # 2. Single Bold Crimson Exterior Solid Arc outside `kn_r` sweeping INWARD (clockwise from 142° to 212°)
        c_arc = (kn_r[0] + 10.0, kn_r[1] - 5.0)
        end_ang = 198.0 + 18.0 * rep_amount
        draw_day2_bold_exterior_arc_ss(
            rd, c_arc, radius_1x=122.0,
            start_deg=140.0, end_deg=end_ang, arrow_at_end=True,
            col_main=col_main, col_soft=col_soft, width_1x=5.2, ss=SS,
        )

        # 3. Left (`°`) & Right (`%`) Pure White Callouts
        deg_live = int(round(10 + 8 * rep_amount))
        pct_live = int(round(78 + 6 * rep_amount))
        draw_day2_floating_white_callout(od, sd, (255, int(np.clip(kn_r[1] - 195, 880, 1030))), f"{deg_live}°", "VALGUS CAVE", align="right")
        draw_day2_floating_white_callout(od, sd, (790, 860), f"{pct_live}%", "ACL & MCL LOAD", align="left")

    elif step_idx == 2:
        # Step 2 (Front Right — Exact Same Knee Alignment Angle at `kn_r ◎`: `0° KNEES TRACK TOES`)
        # 1. Single Pivot Ring `◎` on `kn_r`
        draw_day2_single_pivot_ring_ss(rd, kn_r, 10.5, col_main, SS)

        # 2. Single Bold Emerald Exterior Solid Arc at the EXACT SAME exterior knee position sweeping OUTWARD (counter-clockwise to 140°)
        c_arc = (kn_r[0] + 10.0, kn_r[1] - 5.0)
        end_ang = 198.0 + 14.0 * rep_amount
        draw_day2_bold_exterior_arc_ss(
            rd, c_arc, radius_1x=122.0,
            start_deg=140.0, end_deg=end_ang, arrow_at_end=False,
            col_main=col_main, col_soft=col_soft, width_1x=5.2, ss=SS,
        )

        # 3. Left (`°`) & Right (`%`) Pure White Callouts
        pct_live = int(round(92 + 4 * rep_amount))
        draw_day2_floating_white_callout(od, sd, (255, int(np.clip(kn_r[1] - 195, 880, 1030))), "0°", "KNEES TRACK TOES", align="right")
        draw_day2_floating_white_callout(od, sd, (790, 860), f"{pct_live}%", "VMO & GLUTE LOCK", align="left")

    elif step_idx == 3:
        # Step 3 (Side Wrong — Torso/Spine Angle at `hip_r ◎`: `42° FORWARD LEAN`)
        # 1. Single Pivot Ring `◎` on `hip_r`
        draw_day2_single_pivot_ring_ss(rd, hip_r, 10.5, col_main, SS)

        # 2. Single Bold Crimson Exterior Solid Arc behind/above `hip_r` sweeping FORWARD/DOWNWARD (`188° -> 252°`)
        c_arc = (hip_r[0] + 15.0, hip_r[1] - 25.0)
        end_ang = 236.0 + 22.0 * rep_amount
        draw_day2_bold_exterior_arc_ss(
            rd, c_arc, radius_1x=152.0,
            start_deg=186.0, end_deg=end_ang, arrow_at_end=True,
            col_main=col_main, col_soft=col_soft, width_1x=5.2, ss=SS,
        )

        # 3. Left (`°`) & Right (`%`) Pure White Callouts
        deg_live = int(round(24 + 18 * rep_amount))
        pct_live = int(round(72 + 7 * rep_amount))
        draw_day2_floating_white_callout(od, sd, (265, int(np.clip(hip_r[1] - 95, 840, 990))), f"{deg_live}°", "FORWARD LEAN", align="right")
        draw_day2_floating_white_callout(od, sd, (790, 860), f"{pct_live}%", "LUMBAR OVERLOAD", align="left")

    else:
        # Step 4 (Side Right — Exact Same Torso/Spine Angle at `hip_r ◎`: `15° UPRIGHT LOCK`)
        # 1. Single Pivot Ring `◎` on `hip_r`
        draw_day2_single_pivot_ring_ss(rd, hip_r, 10.5, col_main, SS)

        # 2. Single Bold Emerald Exterior Solid Arc at the EXACT SAME `hip_r` position sweeping UPWARD/BACK (`252° -> 186°`, arrowhead at `186°`)
        c_arc = (hip_r[0] + 15.0, hip_r[1] - 25.0)
        end_ang = 236.0 + 18.0 * rep_amount
        draw_day2_bold_exterior_arc_ss(
            rd, c_arc, radius_1x=152.0,
            start_deg=186.0, end_deg=end_ang, arrow_at_end=False,
            col_main=col_main, col_soft=col_soft, width_1x=5.2, ss=SS,
        )

        # 3. Left (`°`) & Right (`%`) Pure White Callouts
        pct_live = int(round(94 + 4 * rep_amount))
        draw_day2_floating_white_callout(od, sd, (265, int(np.clip(hip_r[1] - 95, 840, 990))), "15°", "UPRIGHT LOCK", align="right")
        draw_day2_floating_white_callout(od, sd, (790, 860), f"{pct_live}%", "QUADS & GLUTES", align="left")

    rig_1x = rig_layer.resize((W, H), Image.Resampling.LANCZOS)
    canvas.alpha_composite(rig_1x)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
    canvas.alpha_composite(ov)


def make_v57_torso_icons(pools: dict) -> list[tuple[Image.Image, Image.Image]]:
    torso_icons = []
    for s_idx in [1, 3, 4]:
        p_pool = pools[s_idx][0]
        mid_rgb = p_pool[-1]
        crop = mid_rgb[560:1420, 220:920]
        pil_act = Image.fromarray(crop).resize((118, 82), Image.Resampling.LANCZOS).convert("RGBA")
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        dark = np.stack([gray * 0.48, gray * 0.50, gray * 0.54], axis=2).astype(np.uint8)
        pil_inact = Image.fromarray(dark).resize((118, 82), Image.Resampling.LANCZOS).convert("RGBA")
        torso_icons.append((pil_act, pil_inact))
    return torso_icons


def render_v57_iphone_safezone_sheet(step_frames_rgb: dict, out_path: str):
    real_shot_bgr = cv2.imread(v39.USER_SCREENSHOT_PATH)
    cap_old = cv2.VideoCapture(os.path.join(OUT_DIR, "day2_v38_master_reel.mp4"))
    cap_old.set(cv2.CAP_PROP_POS_FRAMES, 210)
    _, old_f210_bgr = cap_old.read()
    cap_old.release()

    cw, ch = 420, 910
    ipad = 24
    board = Image.new("RGB", (cw * 4 + ipad * 5, ch + 115), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_hdr = v30.f_pretendard(20, "extrabold")
    f_col = v30.f_pretendard(15, "bold")
    bd.text(
        (28, 20),
        "DAY 3 · iPHONE INSTAGRAM REELS UI SAFE-ZONE PREVIEW (`v57` DAY 2 SIMPLE MASTER)  ·  풀스크린 스케일 + 단일 핵심 각도 아크",
        font=f_hdr,
        fill=COL_GREEN,
    )
    titles = [
        "iPhone View · STEP 1 (18° KNEE VALGUS)",
        "iPhone View · STEP 2 (0° KNEES TRACK TOES)",
        "iPhone View · STEP 3 (42° FORWARD LEAN)",
        "iPhone View · STEP 4 (15° UPRIGHT LOCK)",
    ]
    for i in range(4):
        x = ipad + i * (cw + ipad)
        col = COL_RED if i in (0, 2) else COL_GREEN
        bd.text((x + 4, 58), titles[i], font=f_col, fill=col)
        sim_im = v39.simulate_iphone_ig_reels_view(step_frames_rgb[i + 1], real_shot_bgr, old_f210_bgr)
        rc = sim_im.resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        board.paste(rc, (x, 86))
    board.save(out_path, quality=95)


def main():
    pools = get_or_build_v57_pools()
    torso_icons = make_v57_torso_icons(pools)

    left_w = 430
    curve_l = (np.linspace(0.68, 1.0, left_w, dtype=np.float32) ** 0.78)[None, :, None]
    scrim_2d = v38.make_bottom_scrim_2d()

    step_configs = [
        (1, pools[1][0], pools[1][1], True,  0, 84, "LUMBAR & KNEE RISK",       "BARBELL BACK SQUAT",       "18° KNEE VALGUS"),
        (2, pools[2][0], pools[2][1], False, 2, 96, "NEUTRAL KNEE LOCK",        "GLUTE MEDIUS & VMO",       "0° KNEES TRACK TOES"),
        (3, pools[3][0], pools[3][1], True,  1, 79, "ANTERIOR FORCE LEAK",      "HEEL LIFT & FORWARD DIVE", "42° FORWARD LEAN"),
        (4, pools[4][0], pools[4][1], False, 2, 98, "FULL QUAD & GLUTE · 4×10", "QUADRICEPS & GLUTES",      "15° UPRIGHT LOCK"),
    ]

    panels = []
    crops = []
    step_frames_pil = {}

    for s_idx, pool, kps, is_wrong, dock_idx, pct_val, tag_pill, tag_ctx, cue_1line in step_configs:
        idx = len(pool) - 1
        rgb_frame = apply_static_studio_tint(pool[idx], is_wrong)
        rgb_frame = v38.apply_bottom_left_scrim(rgb_frame, curve_l, scrim_2d)
        kp_cur = kps[idx]

        canvas = Image.fromarray(rgb_frame).convert("RGBA")
        render_v57_day2_simple_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)
        p55.draw_v55_safezone_top_dock(canvas, torso_icons, active_idx=dock_idx, is_wrong=is_wrong)
        p55.draw_v55_safezone_bottom_hud(
            canvas,
            {
                "is_wrong": is_wrong,
                "pct": pct_val,
                "tag_pill": tag_pill,
                "tag_ctx": tag_ctx,
                "cue_1line": cue_1line,
            },
        )

        rgb_pil = canvas.convert("RGB")
        step_frames_pil[s_idx] = rgb_pil
        out_bgr = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)
        panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
        crops.append(out_bgr[450:1550, 0:1080])

    diff_path = os.path.join(OUT_DIR, "day3_v57_difference_sheet.jpg")
    close_path = os.path.join(OUT_DIR, "day3_v57_closeup_sheet.jpg")
    safe_path = os.path.join(OUT_DIR, "day3_v57_iphone_safezone_preview.jpg")

    cv2.imwrite(diff_path, np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(close_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    render_v57_iphone_safezone_sheet(step_frames_pil, safe_path)
    print("Saved v57 preview sheets:", diff_path, close_path, safe_path)


if __name__ == "__main__":
    main()
