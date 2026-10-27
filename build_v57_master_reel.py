#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_v57_master_reel.py`)
=======================================================
Renders the final 600-frame `day3_v57_master_reel.mp4` (`1080x1920`, 30fps, 20.0s),
plus `day3_v57_difference_sheet.jpg`, `day3_v57_closeup_sheet.jpg`, and `day3_v57_iphone_safezone_preview.jpg`:
1. Full-Screen Day 2 Camera Scale (`1.0x`, `720x1280` -> `1080x1920`) so the barbell, torso, hips, thighs, knees,
   and ankles fill the screen with the exact same close, impactful presence as Day 2 (`day2_v39_clean_master_sheet.jpg`).
2. Ultra-Simple Day 2 (`v39`) Visual Formula (`"조금 더 단순하게 잡아보던가 Day2처럼"`):
   - ONLY 1 Concentric Joint Pivot Ring (`◎`) per step (`kn_r ◎` in Steps 1–2, `hip_r ◎` in Steps 3–4),
   - ONLY 1 Bold `10px` Exterior Solid Angle Arc (`SS=4` closed-polygon arc + directional arrowhead) placed cleanly outside the body,
   - ONLY 2 Frameless Pure White (`#FFFFFF`) Floating Callout Badges (`°` on left, `%` on right, zero underlines).
3. 100% Unified Angle Measurement Between Wrong and Right (`"각도 재는 것도 조금 다른것 같은데? 틀린동작이랑 맞는 동작이랑 할때"`):
   - Pair 1 (Front View — Step 1 Wrong vs Step 2 Right): Both measure Knee Alignment Angle (`°`) outside `kn_r ◎`
     (`18° VALGUS CAVE` vs `0° KNEES TRACK TOES`).
   - Pair 2 (Side View — Step 3 Wrong vs Step 4 Right): Both measure Torso/Spine Angle (`°`) above/behind `hip_r ◎`
     (`42° FORWARD LEAN` vs `15° UPRIGHT LOCK`).
"""

import os
import sys
import math
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v38_high_contrast_hud as v38
import preview_v55_demic_day2_master as p55
import preview_v57_day2_simple_master as p57

OUT_DIR = "/workspace/daily_info_reels/output"
AUDIO_WAV = "/workspace/daily_info_reels/output/day3_v44_voiceover.wav"
W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)

_LEFT_W = 430
_CURVE_L = (np.linspace(0.68, 1.0, _LEFT_W, dtype=np.float32) ** 0.78)[None, :, None]
_SCRIM_2D = v38.make_bottom_scrim_2d()


def _render_pool_frame_worker(args):
    s_idx, f_i, n_p, raw_rgb, kp_cur, is_wrong = args
    rep_amount = f_i / float(max(1, n_p - 1))
    rgb_tinted = p57.apply_static_studio_tint(raw_rgb, is_wrong)
    rgb_scrim = v38.apply_bottom_left_scrim(rgb_tinted, _CURVE_L, _SCRIM_2D)
    canvas = Image.fromarray(rgb_scrim).convert("RGBA")
    p57.render_v57_day2_simple_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=rep_amount)
    return s_idx, f_i, np.array(canvas.convert("RGB"))


def main():
    t0 = time.time()
    pools = p57.get_or_build_v57_pools()
    torso_icons = p57.make_v57_torso_icons(pools)

    tasks = []
    for s_idx, is_w in [(1, True), (2, False), (3, True), (4, False)]:
        p_pool, p_kps = pools[s_idx]
        n_p = len(p_pool)
        for f_i in range(n_p):
            tasks.append((s_idx, f_i, n_p, p_pool[f_i], p_kps[f_i], is_w))

    print(f"Pre-rendering {len(tasks)} unique pool frames for v57 Day 2 Simple Master...")
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
            "pct_base": 78,
            "pct_peak": 84,
            "tag_pill": "LUMBAR & KNEE RISK",
            "tag_ctx": "BARBELL BACK SQUAT",
            "cue_1line": "18° KNEE VALGUS",
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
            "pct_base": 72,
            "pct_peak": 79,
            "tag_pill": "ANTERIOR FORCE LEAK",
            "tag_ctx": "HEEL LIFT & FORWARD DIVE",
            "cue_1line": "42° FORWARD LEAN",
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
            "cue_1line": "15° UPRIGHT LOCK",
        },
    ]

    top_dock_cache = {}
    for cfg in step_configs:
        s_idx = cfg["step_idx"]
        td_im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        p55.draw_v55_safezone_top_dock(td_im, torso_icons, active_idx=cfg["dock_idx"], is_wrong=cfg["is_wrong"])
        top_dock_cache[s_idx] = td_im

    hud_cache = {}
    tmp_silent = os.path.join(OUT_DIR, "_tmp_day3_v57_silent.mp4")
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

        # Complete 1-rep cycle (standing -> bottom of squat at t_local=2.33s -> standing lockout at t_local=4.66s..5.0s)
        phase = min(1.0, t_local / 4.66)
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

        # Sample peak squat depth frames (`t = 2.4s, 7.4s, 12.4s, 17.4s` -> `fidx = 72, 222, 372, 522`)
        if fidx in (72, 222, 372, 522):
            peak_frames_pil[s_idx] = rgb_pil
            panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
            crops.append(out_bgr[450:1550, 0:1080])

    vw.release()

    final_v57_mp4 = os.path.join(OUT_DIR, "day3_v57_master_reel.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_silent,
            "-i", AUDIO_WAV,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart",
            final_v57_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(tmp_silent):
        os.remove(tmp_silent)

    diff_path = os.path.join(OUT_DIR, "day3_v57_difference_sheet.jpg")
    close_path = os.path.join(OUT_DIR, "day3_v57_closeup_sheet.jpg")
    safe_path = os.path.join(OUT_DIR, "day3_v57_iphone_safezone_preview.jpg")
    cv2.imwrite(diff_path, np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(close_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    p57.render_v57_iphone_safezone_sheet(peak_frames_pil, safe_path)

    print(f"Done in {time.time() - t0:.1f}s -> {final_v57_mp4}")


if __name__ == "__main__":
    main()
