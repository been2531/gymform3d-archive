#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_v56_master_reel.py`)
=======================================================
Renders the final 600-frame `day3_v56_master_reel.mp4` (and `day3_v55_master_reel.mp4`),
plus `day3_v56_difference_sheet.jpg`, `day3_v56_closeup_sheet.jpg`, and `day3_v56_iphone_safezone_preview.jpg`:
1. 100% Real, Unwarped 3D Squat Postures (`0%` `cv2.remap`, `0%` background wobble, `diff == 0.0` on background):
   - Step 1 (Front View — WRONG): Realistic gym knee valgus (`14° KNEE VALGUS`) where knees drift inside the toe line.
   - Step 2 (Front View — RIGHT): Wide external rotation (`0° KNEES TRACK TOES`) where knees track cleanly over toes.
   - Step 3 (Side View — WRONG): Realistic "Good-Morning Squat" forward torso pitch (`+28° FORWARD LEAN`) & heel lift.
   - Step 4 (Side View — RIGHT): Proud upright chest, planted heels & parallel depth (`95° PARALLEL DRIVE`).
2. Strict iPhone Instagram Reels Safe-Zone Framing (`head_top >= 430`, `foot_bot <= 1285`):
   - Top Exercise Header (`y = 258..300`) & Compact 3-Thumbnail Progression Dock (`y = 308..406`) stay 100% above the
     standing mannequin's head (`y >= 432`) on every single frame (`0px` overlap!).
   - Mannequin feet (`y <= 1260`) stay 100% above the Bottom-Left HUD (`y = 1315..1448`).
3. Day 2 (`v39`) Muscle Fiber Striation & 3D Bone X-Ray Shader + Pure `@demicstory` / Day 2 Minimalist Laser Telemetry:
   - ZERO thick 2D/3D clip-art ribbon arrows or chunky filled triangle arrowheads!
   - Slender `2.8px` glowing anatomical bone-axis laser lines (`Hip ◎ — Knee ◎ — Ankle ◎` & `Shoulder ◎ — Hip ◎ — Knee ◎ — Ankle ◎`),
   - Clinical reference plumb/parallel hairlines & `@demicstory` floor-contact laser bars,
   - Day 2 (`v39`) `3.0px` exterior solid protractor arcs (`14°`, `0°`, `+28°`, `95°`) capped with delicate open micro-chevrons (`>`),
   - Slender `1.5px` exterior pointer lines to Pure White (`#FFFFFF`) `64pt` numbers + `22pt` sub-labels (`x in 96..978`, zero underlines).
"""

import os
import sys
import math
import time
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/daily_info_reels")
import preview_v55_demic_day2_master as p55
import preview_v56_minimal_laser_master as p56

OUT_DIR = "/workspace/daily_info_reels/output"
AUDIO_WAV = "/workspace/daily_info_reels/output/day3_v44_voiceover.wav"
W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)


def _render_pool_frame_worker(args):
    s_idx, f_i, n_p, raw_rgb, kp_cur, is_wrong = args
    rep_amount = f_i / float(max(1, n_p - 1))
    rgb_tinted = p56.apply_static_studio_tint(raw_rgb, is_wrong)
    canvas = Image.fromarray(rgb_tinted).convert("RGBA")
    p56.render_v56_minimal_laser_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=rep_amount)
    return s_idx, f_i, np.array(canvas.convert("RGB"))


def main():
    t0 = time.time()
    pools = p55.get_or_build_v55_pools()
    torso_icons = p55.make_torso_icons(pools)

    # Pre-render all unique frames across the 4 pools in parallel
    tasks = []
    for s_idx, is_w in [(1, True), (2, False), (3, True), (4, False)]:
        p_pool, p_kps = pools[s_idx]
        n_p = len(p_pool)
        for f_i in range(n_p):
            tasks.append((s_idx, f_i, n_p, p_pool[f_i], p_kps[f_i], is_w))

    print(f"Pre-rendering {len(tasks)} unique pool frames with v56 Minimalist Laser Telemetry...")
    rendered_pools = {s_idx: [None] * len(pools[s_idx][0]) for s_idx in (1, 2, 3, 4)}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for s_idx, f_i, out_rgb in ex.map(_render_pool_frame_worker, tasks, chunksize=4):
            rendered_pools[s_idx][f_i] = out_rgb
    print(f"Pool pre-render finished in {time.time() - t0:.1f}s")

    step_configs = [
        {
            "step_idx": 1,
            "pool": rendered_pools[1],
            "is_wrong": True,
            "dock_idx": 0,
            "pct_base": 79,
            "pct_peak": 84,
            "tag_pill": "LUMBAR & KNEE RISK",
            "tag_ctx": "BARBELL BACK SQUAT",
            "cue_1line": "14° KNEE VALGUS",
        },
        {
            "step_idx": 2,
            "pool": rendered_pools[2],
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
            "pool": rendered_pools[3],
            "is_wrong": True,
            "dock_idx": 1,
            "pct_base": 74,
            "pct_peak": 79,
            "tag_pill": "ANTERIOR FORCE LEAK",
            "tag_ctx": "HEEL LIFT & FORWARD DIVE",
            "cue_1line": "+28° FORWARD LEAN",
        },
        {
            "step_idx": 4,
            "pool": rendered_pools[4],
            "is_wrong": False,
            "dock_idx": 2,
            "pct_base": 94,
            "pct_peak": 98,
            "tag_pill": "FULL QUAD & GLUTE · 4×10",
            "tag_ctx": "QUADRICEPS & GLUTES",
            "cue_1line": "95° PARALLEL DRIVE",
        },
    ]

    # Cache top docks and bottom HUDs
    top_dock_cache = {}
    for cfg in step_configs:
        s_idx = cfg["step_idx"]
        td_im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        p55.draw_v55_safezone_top_dock(td_im, torso_icons, active_idx=cfg["dock_idx"], is_wrong=cfg["is_wrong"])
        top_dock_cache[s_idx] = td_im

    hud_cache = {}
    tmp_silent = os.path.join(OUT_DIR, "_tmp_day3_v56_silent.mp4")
    vw = cv2.VideoWriter(tmp_silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    peak_frames_pil = {}
    panels, crops = [], []

    for fidx in range(TOTAL_FRAMES):
        t_sec = fidx / float(FPS)
        s_i = min(3, int(t_sec // 5.0))
        t_local = t_sec - s_i * 5.0
        cfg = step_configs[s_i]
        s_idx = cfg["step_idx"]
        pool = cfg["pool"]
        n_p = len(pool)

        # Smooth cosine rep wave peaking at t_local = 2.0s..2.5s (fidx = 60..72, 210..222, 360..372, 510..522)
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
        canvas.alpha_composite(top_dock_cache[s_idx])

        pct_val = int(round(cfg["pct_base"] + (cfg["pct_peak"] - cfg["pct_base"]) * u))
        hud_key = (s_idx, pct_val)
        if hud_key not in hud_cache:
            bh_im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            p55.draw_v55_safezone_bottom_hud(
                bh_im,
                {
                    "is_wrong": cfg["is_wrong"],
                    "pct": pct_val,
                    "tag_pill": cfg["tag_pill"],
                    "tag_ctx": cfg["tag_ctx"],
                    "cue_1line": cfg["cue_1line"],
                },
            )
            hud_cache[hud_key] = bh_im
        canvas.alpha_composite(hud_cache[hud_key])

        rgb_pil = canvas.convert("RGB")
        out_bgr = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)
        vw.write(out_bgr)

        if fidx in (60, 210, 360, 510):
            peak_frames_pil[s_idx] = rgb_pil
            panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
            crops.append(out_bgr[415:1310, 0:1080])

    vw.release()

    final_v56_mp4 = os.path.join(OUT_DIR, "day3_v56_master_reel.mp4")
    final_v55_mp4 = os.path.join(OUT_DIR, "day3_v55_master_reel.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_silent,
            "-i", AUDIO_WAV,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart",
            final_v56_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(tmp_silent):
        os.remove(tmp_silent)
    shutil.copyfile(final_v56_mp4, final_v55_mp4)

    for ver in ("v55", "v56"):
        diff_path = os.path.join(OUT_DIR, f"day3_{ver}_difference_sheet.jpg")
        close_path = os.path.join(OUT_DIR, f"day3_{ver}_closeup_sheet.jpg")
        safe_path = os.path.join(OUT_DIR, f"day3_{ver}_iphone_safezone_preview.jpg")
        cv2.imwrite(diff_path, np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        cv2.imwrite(close_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        p55.render_iphone_safezone_sheet(peak_frames_pil, safe_path)

    print(f"Done in {time.time() - t0:.1f}s -> {final_v56_mp4} & {final_v55_mp4}")


if __name__ == "__main__":
    main()
