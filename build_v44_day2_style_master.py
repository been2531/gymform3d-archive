#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_v44_day2_style_master.py`)
=============================================================
Matches the winning `Day 1 / Day 2 (v38/v39 Broadcast Luxury Master)` format 1:1:
1. Full-Screen `1080x1920` Real 3D I2V Full-Body Motion across all 4 steps (`모습은 전체가 나와야지`,
   ZERO stage shrinking, ZERO `cv2.remap` fake stretching, ZERO ghost/duplicate legs):
   - Step 1 (`0–5s`, Front View — WRONG): Full-body front squat with red knee valgus collapse (`18° VALGUS CAVE`)
   - Step 2 (`5–10s`, Front View — FIX): Full-body front squat with green outward knee tracking (`0° KNEES TRACK TOES`)
   - Step 3 (`10–15s`, 3/4 Side View — WRONG): Full-body 3/4 side squat with red lumbar & heel lift strain (`+28° HEELS LIFTING`)
   - Step 4 (`15–20s`, 3/4 Side View — FIX): Full-body 3/4 side squat with green parallel depth & quad/glute drive (`95° PARALLEL DEPTH`)
2. Exact `Day 2 (v38/v39)` Top Exercise Header (`BARBELL BACK SQUAT [ LEGS · QUADS & GLUTES ]`)
   + Top 3-Thumbnail Progression Preview Dock (`x=96, y=258..446`).
3. Exact `Day 1 / Day 2 (v38/v39)` Exterior Solid Angle Arcs + Joint Pivot Rings (`◎`) + Exterior `°` & `%` Badges
   + Body-Fitted Directional Arrows on the outside of the joints (ZERO text between the legs!).
4. Exact `Day 2 (v38/v39)` Bottom-Left High-Contrast Luxury HUD (`x=96, y=1300..1444`).
"""

import os
import sys
import math
import pickle
import asyncio
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import universal_anatomy_rig as uar
import build_v30_pure_graphic_numbers as v30
import build_v36_clean_master as v36
import build_day3_v39_squat as d3

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_AUDIO = "/workspace/daily_info_reels/_tmp_day3_v44_audio"
CACHE_PKL = "/workspace/daily_info_reels/_cache_day3_v44_pools.pkl"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_AUDIO, exist_ok=True)

W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)

SAFE_X_LEFT = 96
SAFE_Y_TOP = 258
SAFE_X_RIGHT_BADGE = 805

COL_RED = (235, 42, 48)
COL_RED_SOFT = (255, 138, 132)
COL_GREEN = (16, 205, 110)
COL_GREEN_SOFT = (148, 246, 192)


def draw_body_fitted_directional_arrow(
    rd: ImageDraw.ImageDraw,
    p_start_ss: tuple[float, float],
    p_end_ss: tuple[float, float],
    col_main: tuple[int, int, int],
    col_soft: tuple[int, int, int],
    ss: int = 2,
):
    """
    Draws a sleek, tapered body-fitted directional arrow in 2x supersampled space (`SS=2`)
    with a crisp dark outline, solid color shaft, and sharp chevron head.
    """
    x0, y0 = p_start_ss
    x1, y1 = p_end_ss
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 8.0:
        return
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux

    head_len = 18.0 * ss
    w0 = 2.2 * ss
    w1 = 5.2 * ss

    bx = x1 - ux * head_len
    by = y1 - uy * head_len

    shaft_poly = [
        (x0 + nx * w0, y0 + ny * w0),
        (bx + nx * w1, by + ny * w1),
        (bx - nx * w1, by - ny * w1),
        (x0 - nx * w0, y0 - ny * w0),
    ]
    rd.line([(x0, y0), (bx, by)], fill=(0, 0, 0, 190), width=int(round((w1 * 2) + 4 * ss)))
    rd.polygon(shaft_poly, fill=(*col_main, 248))

    ang_deg = math.degrees(math.atan2(dy, dx))
    v30.draw_arrow_head(rd, (x1, y1), ang_deg, int(round(13 * ss)), (*col_soft, 255))


def render_v44_day2_style_callouts(
    canvas: Image.Image,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float = 1.0,
):
    """
    Combines the exact `Day 1 / Day 2 (v38/v39)` exterior solid angle arcs + degree (`°`) & `%` badges
    + joint pivot reticles (`◎`) with sleek body-fitted directional arrows on the 4 full-body 3D postures.
    ZERO text between the legs!
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

    col_main = COL_RED if is_wrong else COL_GREEN
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    sh = kp["sh"]
    hip_r = kp["hip_r"]
    hip_l = kp["hip_l"]
    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    ank_r = kp["ank_r"]
    ank_l = kp["ank_l"]

    if step_idx == 1:
        # Step 1 (Full-Body Front View — WRONG: `18° VALGUS CAVE` & `84% ACL & MCL LOAD`)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_r), 10.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_l), 10.5, col_main, SS)

        c_arc = S((kn_r[0] + 15.0, kn_r[1] - 10.0))
        r_a = 112 * SS
        start_ang = 330
        end_ang = int(round(375 + 22 * rep_amount))
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=4 * SS)
        ang_t = math.radians(end_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, end_ang + 90, 12 * SS, (*col_soft, 255))

        # Sleek inward buckle directional arrows fitted to the outside of both knees (ZERO text between legs)
        shift = 12.0 * rep_amount
        draw_body_fitted_directional_arrow(
            rd,
            S((kn_r[0] - 85.0, kn_r[1] + 5.0)),
            S((kn_r[0] - 14.0 + shift, kn_r[1] + 5.0)),
            col_main, col_soft, SS
        )
        draw_body_fitted_directional_arrow(
            rd,
            S((kn_l[0] + 85.0, kn_l[1] + 5.0)),
            S((kn_l[0] + 14.0 - shift, kn_l[1] + 5.0)),
            col_main, col_soft, SS
        )

        v30.draw_floating_number_badge(
            od, sd,
            (325, 1020),
            "18°", "VALGUS CAVE", col_main, col_soft, align="right"
        )
        v30.draw_floating_number_badge(
            od, sd,
            (SAFE_X_RIGHT_BADGE, 875),
            "84%", "ACL & MCL LOAD", col_main, col_soft, align="left"
        )

    elif step_idx == 2:
        # Step 2 (Full-Body Front View — FIX: `0° KNEES TRACK TOES` & `96% VMO & GLUTE LOCK`)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_r), 10.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_l), 10.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(ank_r), 8.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(ank_l), 8.5, col_main, SS)

        c_arc = S((kn_r[0] - 10.0, kn_r[1] + 5.0))
        r_a = 114 * SS
        start_ang = 142
        end_ang = 204
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=4 * SS)
        ang_t = math.radians(start_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, start_ang - 90, 12 * SS, (*col_soft, 255))

        # Sleek outward knee drive arrows fitted to both knees
        shift = 12.0 * rep_amount
        draw_body_fitted_directional_arrow(
            rd,
            S((kn_r[0] + 45.0, kn_r[1] + 5.0)),
            S((kn_r[0] - 42.0 - shift, kn_r[1] + 5.0)),
            col_main, col_soft, SS
        )
        draw_body_fitted_directional_arrow(
            rd,
            S((kn_l[0] - 45.0, kn_l[1] + 5.0)),
            S((kn_l[0] + 42.0 + shift, kn_l[1] + 5.0)),
            col_main, col_soft, SS
        )

        v30.draw_floating_number_badge(
            od, sd,
            (325, 1005),
            "0°", "KNEES TRACK TOES", col_main, col_soft, align="right"
        )
        v30.draw_floating_number_badge(
            od, sd,
            (SAFE_X_RIGHT_BADGE, 875),
            "96%", "VMO & GLUTE LOCK", col_main, col_soft, align="left"
        )

    elif step_idx == 3:
        # Step 3 (Full-Body 3/4 Side View — WRONG: `+28° HEELS LIFTING` & `79% LUMBAR & PATELLA`)
        v36.draw_single_elbow_pivot_ring(rd, S(hip_r), 10.0, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_r), 9.0, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(ank_r), 9.0, col_main, SS)

        c_arc = S((hip_r[0] + 15.0, hip_r[1] - 35.0))
        r_a = 148 * SS
        start_ang = 195
        end_ang = int(round(238 + 22 * rep_amount))
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=4 * SS)
        ang_t = math.radians(end_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, end_ang + 90, 12 * SS, (*col_soft, 255))

        # Heel lifting upward error arrow
        draw_body_fitted_directional_arrow(
            rd,
            S((ank_r[0] - 48.0, ank_r[1] + 15.0)),
            S((ank_r[0] - 48.0, ank_r[1] - 68.0)),
            col_main, col_soft, SS
        )

        v30.draw_floating_number_badge(
            od, sd,
            (345, 1040),
            "+28°", "HEELS LIFTING", col_main, col_soft, align="right"
        )
        v30.draw_floating_number_badge(
            od, sd,
            (SAFE_X_RIGHT_BADGE, 880),
            "79%", "LUMBAR & PATELLA", col_main, col_soft, align="left"
        )

    else:
        # Step 4 (Full-Body 3/4 Side View — FIX: `95° PARALLEL DEPTH` & `98% QUADS & GLUTES`)
        v36.draw_single_elbow_pivot_ring(rd, S(hip_r), 10.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(kn_r), 9.5, col_main, SS)
        v36.draw_single_elbow_pivot_ring(rd, S(ank_r), 8.5, col_main, SS)

        c_arc = S((hip_r[0] - 10.0, hip_r[1] + 5.0))
        r_a = 148 * SS
        start_ang = 128
        end_ang = int(round(192 + 12 * rep_amount))
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=4 * SS)
        ang_t = math.radians(start_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, start_ang - 90, 12 * SS, (*col_soft, 255))

        # Upright chest drive arrow (`↑`) + flat heel floor lock arrow (`↓`)
        draw_body_fitted_directional_arrow(
            rd,
            S((sh[0] + 85.0, sh[1] + 75.0)),
            S((sh[0] + 85.0, sh[1] - 35.0)),
            col_main, col_soft, SS
        )
        draw_body_fitted_directional_arrow(
            rd,
            S((ank_r[0] - 48.0, ank_r[1] - 65.0)),
            S((ank_r[0] - 48.0, ank_r[1] + 12.0)),
            col_main, col_soft, SS
        )

        v30.draw_floating_number_badge(
            od, sd,
            (335, 1030),
            "95°", "PARALLEL DEPTH", col_main, col_soft, align="right"
        )
        v30.draw_floating_number_badge(
            od, sd,
            (SAFE_X_RIGHT_BADGE, 875),
            "98%", "QUADS & GLUTES", col_main, col_soft, align="left"
        )

    rig_1x = rig_layer.resize((W, H), Image.Resampling.LANCZOS)
    canvas.alpha_composite(rig_1x)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(ov)


def build_v44():
    assert os.path.exists(CACHE_PKL), "Missing cached pools!"
    with open(CACHE_PKL, "rb") as f:
        pools = pickle.load(f)

    pool_s1, kp_s1 = pools[1]
    pool_s2, kp_s2 = pools[2]
    pool_s3, kp_s3 = pools[3]
    pool_s4, kp_s4 = pools[4]

    torso_icons = []
    for p_pool in [pool_s1, pool_s3, pool_s4]:
        mid_rgb = p_pool[len(p_pool) // 2]
        crop = mid_rgb[360:1440, 180:900]
        pil_act = Image.fromarray(crop).resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        dark = np.stack([gray * 0.48, gray * 0.50, gray * 0.54], axis=2).astype(np.uint8)
        pil_inact = Image.fromarray(dark).resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        torso_icons.append((pil_act, pil_inact))

    wav_path = os.path.join(OUT_DIR, "day3_v44_voiceover.wav")
    if not os.path.exists(wav_path):
        vo_lines = [
            (0.18, "Stop letting your knees cave inward at the bottom of the squat."),
            (5.12, "Drive both knees wide over your toes and lock your feet flat."),
            (10.12, "Never let your heels lift or fold your lower back forward."),
            (15.12, "Keep your chest proud and drive straight up through your quads and glutes."),
        ]
        asyncio.run(d3.synthesize_day3_voiceover_wav(wav_path, vo_lines, duration_s=DURATION))

    tmp_silent = os.path.join(OUT_DIR, "_tmp_day3_v44_silent.mp4")
    writer = cv2.VideoWriter(tmp_silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    step_configs = [
        {
            "step_idx": 1,
            "pool": pool_s1,
            "kps": kp_s1,
            "is_wrong": True,
            "dock_idx": 0,
            "pct_base": 82,
            "pct_peak": 84,
            "tag_pill": "LUMBAR & KNEE RISK",
            "tag_ctx": "BARBELL BACK SQUAT",
            "cue_1line": "18° KNEE VALGUS",
        },
        {
            "step_idx": 2,
            "pool": pool_s2,
            "kps": kp_s2,
            "is_wrong": False,
            "dock_idx": 2,
            "pct_base": 92,
            "pct_peak": 96,
            "tag_pill": "NEUTRAL KNEE LOCK",
            "tag_ctx": "GLUTE MEDIUS & VMO",
            "cue_1line": "0° KNEES TRACK TOES",
        },
        {
            "step_idx": 3,
            "pool": pool_s3,
            "kps": kp_s3,
            "is_wrong": True,
            "dock_idx": 1,
            "pct_base": 76,
            "pct_peak": 79,
            "tag_pill": "ANTERIOR FORCE LEAK",
            "tag_ctx": "HEEL LIFT & FORWARD DIVE",
            "cue_1line": "+28° FORWARD LEAN",
        },
        {
            "step_idx": 4,
            "pool": pool_s4,
            "kps": kp_s4,
            "is_wrong": False,
            "dock_idx": 2,
            "pct_base": 94,
            "pct_peak": 98,
            "tag_pill": "FULL QUAD & GLUTE · 4×10",
            "tag_ctx": "QUADRICEPS & GLUTES",
            "cue_1line": "95° PARALLEL DRIVE",
        },
    ]

    rep_peaks = {}
    print("Rendering 600 frames of Day 3 v44 (Day 1 & Day 2 Broadcast Luxury Master format)...")
    for fidx in range(TOTAL_FRAMES):
        t_sec = fidx / float(FPS)
        s_i = min(3, int(t_sec // 5.0))
        t_local = t_sec - s_i * 5.0
        cfg = step_configs[s_i]

        pool = cfg["pool"]
        kps = cfg["kps"]
        n_p = len(pool)

        phase = (t_local / 5.0) * 1.25
        u = 0.5 - 0.5 * math.cos(phase * 2.0 * math.pi)
        pos = u * (n_p - 1)
        i0 = int(math.floor(pos))
        i1 = min(n_p - 1, i0 + 1)
        alpha = pos - i0

        if alpha < 1e-4:
            rgb_frame = pool[i0].copy()
        else:
            rgb_frame = cv2.addWeighted(pool[i0], 1.0 - alpha, pool[i1], alpha, 0.0)

        kp_cur = {}
        for k in kps[0].keys():
            kp_cur[k] = (
                kps[i0][k][0] * (1.0 - alpha) + kps[i1][k][0] * alpha,
                kps[i0][k][1] * (1.0 - alpha) + kps[i1][k][1] * alpha,
            )

        canvas = Image.fromarray(rgb_frame).convert("RGBA")

        render_v44_day2_style_callouts(
            canvas,
            step_idx=cfg["step_idx"],
            kp=kp_cur,
            is_wrong=cfg["is_wrong"],
            rep_amount=u,
        )

        d3.draw_day3_v39_safezone_top_dock(
            canvas,
            torso_icons=torso_icons,
            active_idx=cfg["dock_idx"],
            is_wrong=cfg["is_wrong"],
            exercise_name="BARBELL BACK SQUAT",
            muscle_group="[ LEGS · QUADS & GLUTES ]",
        )

        pct_val = int(round(cfg["pct_base"] + (cfg["pct_peak"] - cfg["pct_base"]) * u))
        sp_hud = {
            "is_wrong": cfg["is_wrong"],
            "pct": pct_val,
            "tag_pill": cfg["tag_pill"],
            "tag_ctx": cfg["tag_ctx"],
            "cue_1line": cfg["cue_1line"],
        }
        d3.draw_day3_v39_safezone_bottom_hud(canvas, sp_hud)

        out_rgb = np.array(canvas.convert("RGB"))
        writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

        if fidx in (60, 210, 360, 510):
            rep_peaks[s_i + 1] = out_rgb.copy()

    writer.release()

    final_mp4 = os.path.join(OUT_DIR, "day3_v44_master_reel.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_silent,
            "-i", wav_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(tmp_silent):
        os.remove(tmp_silent)

    p1 = cv2.resize(cv2.cvtColor(rep_peaks[1], cv2.COLOR_RGB2BGR), (540, 960))
    p2 = cv2.resize(cv2.cvtColor(rep_peaks[2], cv2.COLOR_RGB2BGR), (540, 960))
    p3 = cv2.resize(cv2.cvtColor(rep_peaks[3], cv2.COLOR_RGB2BGR), (540, 960))
    p4 = cv2.resize(cv2.cvtColor(rep_peaks[4], cv2.COLOR_RGB2BGR), (540, 960))
    sheet_2x2 = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
    sheet_path = os.path.join(OUT_DIR, "day3_v44_master_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("Saved:", final_mp4, "and", sheet_path)


if __name__ == "__main__":
    build_v44()
