#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_v45_luxury_master.py`)
=========================================================
Upgrades `v44` with the `v45 Broadcast-Luxury SDF Vector & Pure-White Callout` suite:
1. Analytical SDF Optical Joint Nodes (`점`):
   - Specular white core + neon corona + 4-quadrant precision micro-ring + outer bracket arcs.
2. Analytical SDF Tapered Bezier Arrows (`화살표`):
   - Smooth anatomical curvature, width & opacity taper from tail to head, hot white inner core,
     and aerodynamic dual swept-wing chevrons (`❯❯`).
3. Analytical SDF Gradient Angle Sector & Arc (`각도 아크`) + Architectural Callout Leader Pins.
4. Upgraded Exterior Callout Badges (`18° VALGUS CAVE`, `0° KNEES TRACK TOES`, `+28° HEELS LIFTING`, `95° PARALLEL DEPTH`):
   - 56pt extrabold Pure White degree/percentage numbers,
   - Sleek 3px colored accent bar,
   - Larger 21pt extrabold PURE WHITE (`#FFFFFF`) sub-labels for crystal-clear mobile readability.
"""

import os
import sys
import math
import time
import pickle
import asyncio
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/daily_info_reels")
import build_day3_v39_squat as d3
import preview_v45_luxury_vectors as v45

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_PKL = "/workspace/daily_info_reels/_cache_day3_v44_pools.pkl"
W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)


def _render_pool_frame_worker(args):
    s_idx, f_i, n_p, rgb_frame, kp_cur, is_wrong = args
    rep_amount = f_i / float(max(1, n_p - 1))
    canvas = v45.render_v45_luxury_callouts_and_vectors(
        rgb_frame,
        step_idx=s_idx,
        kp=kp_cur,
        is_wrong=is_wrong,
        rep_amount=rep_amount,
    )
    return s_idx, f_i, np.array(canvas.convert("RGB"))


def build_v45():
    t0 = time.time()
    assert os.path.exists(CACHE_PKL), "Missing cached pools!"
    with open(CACHE_PKL, "rb") as f:
        pools = pickle.load(f)

    pool_s1, kp_s1 = pools[1]
    pool_s2, kp_s2 = pools[2]
    pool_s3, kp_s3 = pools[3]
    pool_s4, kp_s4 = pools[4]

    # Build top-dock 3-thumbnail icons from clean base pools before vector overlay
    torso_icons = []
    for p_pool in [pool_s1, pool_s3, pool_s4]:
        mid_rgb = p_pool[len(p_pool) // 2]
        crop = mid_rgb[360:1440, 180:900]
        pil_act = Image.fromarray(crop).resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        dark = np.stack([gray * 0.48, gray * 0.50, gray * 0.54], axis=2).astype(np.uint8)
        pil_inact = Image.fromarray(dark).resize((136, 108), Image.Resampling.LANCZOS).convert("RGBA")
        torso_icons.append((pil_act, pil_inact))

    # Pre-render SDF luxury vectors onto all 4x49 pool frames in parallel across CPU cores
    tasks = []
    for s_idx, (p_pool, p_kps, is_w) in [
        (1, (pool_s1, kp_s1, True)),
        (2, (pool_s2, kp_s2, False)),
        (3, (pool_s3, kp_s3, True)),
        (4, (pool_s4, kp_s4, False)),
    ]:
        n_p = len(p_pool)
        for f_i in range(n_p):
            tasks.append((s_idx, f_i, n_p, p_pool[f_i], p_kps[f_i], is_w))

    print(f"Pre-rendering {len(tasks)} pool frames with SDF luxury vectors across CPU cores...")
    v45_pools = {1: [None] * len(pool_s1), 2: [None] * len(pool_s2), 3: [None] * len(pool_s3), 4: [None] * len(pool_s4)}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for s_idx, f_i, out_rgb in ex.map(_render_pool_frame_worker, tasks, chunksize=4):
            v45_pools[s_idx][f_i] = out_rgb
    print(f"SDF pool pre-render finished in {time.time() - t0:.1f}s")

    wav_path = os.path.join(OUT_DIR, "day3_v44_voiceover.wav")
    if not os.path.exists(wav_path):
        vo_lines = [
            (0.18, "Stop letting your knees cave inward at the bottom of the squat."),
            (5.12, "Drive both knees wide over your toes and lock your feet flat."),
            (10.12, "Never let your heels lift or fold your lower back forward."),
            (15.12, "Keep your chest proud and drive straight up through your quads and glutes."),
        ]
        asyncio.run(d3.synthesize_day3_voiceover_wav(wav_path, vo_lines, duration_s=DURATION))

    tmp_silent = os.path.join(OUT_DIR, "_tmp_day3_v45_silent.mp4")
    writer = cv2.VideoWriter(tmp_silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    step_configs = [
        {
            "step_idx": 1,
            "pool": v45_pools[1],
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
            "pool": v45_pools[2],
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
            "pool": v45_pools[3],
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
            "pool": v45_pools[4],
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
    print("Assembling 600 frames of Day 3 v45 Master Reel...")
    for fidx in range(TOTAL_FRAMES):
        t_sec = fidx / float(FPS)
        s_i = min(3, int(t_sec // 5.0))
        t_local = t_sec - s_i * 5.0
        cfg = step_configs[s_i]

        pool = cfg["pool"]
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

        canvas = Image.fromarray(rgb_frame).convert("RGBA")

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

    final_mp4 = os.path.join(OUT_DIR, "day3_v45_master_reel.mp4")
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
    sheet_path = os.path.join(OUT_DIR, "day3_v45_master_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print(f"Done in {time.time() - t0:.1f}s -> {final_mp4} & {sheet_path}")


if __name__ == "__main__":
    build_v45()
