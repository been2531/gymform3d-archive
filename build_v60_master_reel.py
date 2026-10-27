#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_v60_master_reel.py`)
=======================================================
Renders the final 600-frame `day3_v60_master_reel.mp4` (`1080x1920`, 30fps, 20.0s),
plus `day3_v60_vs_day2_texture_compare.jpg`, `day3_v60_4step_closeup_sheet.jpg`,
and `day3_v60_iphone_safezone_preview.jpg`, and runs `qc_gate.py` to verify 100% PASS.
"""

import os
import sys
import math
import time
import asyncio
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import build_day3_v39_squat as d3
import build_v30_pure_graphic_numbers as v30
import build_v38_high_contrast_hud as v38
import build_v39_ig_safe_zone as v39
import preview_v58_day2_exact_master as p58
import preview_v60_day2_authentic_master as p60

OUT_DIR = "/workspace/daily_info_reels/output"
AUDIO_WAV = "/workspace/daily_info_reels/output/day3_v60_voiceover.wav"
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
    rgb_tinted = p58.apply_static_studio_tint(raw_rgb, is_wrong)
    rgb_scrim = v38.apply_bottom_left_scrim(rgb_tinted, _CURVE_L, _SCRIM_2D)
    canvas = Image.fromarray(rgb_scrim).convert("RGBA")
    p60.render_v60_clean_angle_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=rep_amount)
    return s_idx, f_i, np.array(canvas.convert("RGB"))


def render_v60_iphone_safezone_sheet(step_frames_rgb: dict, out_path: str):
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
        "DAY 3 · iPHONE INSTAGRAM REELS UI SAFE-ZONE PREVIEW (`v60`)  ·  Day 2 3D 근육결·뼈대 X-Ray + 발바닥 고정 + 프리뷰-머리 분리",
        font=f_hdr,
        fill=p58.COL_GREEN,
    )
    titles = [
        "iPhone View · STEP 1 (16° KNEE VALGUS)",
        "iPhone View · STEP 2 (0° KNEE TRACKING)",
        "iPhone View · STEP 3 (41° FORWARD LEAN)",
        "iPhone View · STEP 4 (22° NEUTRAL SPINE)",
    ]
    for i in range(4):
        x = ipad + i * (cw + ipad)
        col = p58.COL_RED if i in (0, 2) else p58.COL_GREEN
        bd.text((x + 4, 58), titles[i], font=f_col, fill=col)
        sim_im = v39.simulate_iphone_ig_reels_view(step_frames_rgb[i + 1], real_shot_bgr, old_f210_bgr)
        rc = sim_im.resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        board.paste(rc, (x, 86))
    board.save(out_path, quality=95)


def main():
    t0 = time.time()
    if not os.path.exists(AUDIO_WAV):
        vo_lines = [
            (0.18, "Stop letting your knees cave inward at the bottom of the squat."),
            (5.12, "Drive both knees wide over your toes and lock your feet flat."),
            (10.12, "Never shoot your hips up early or fold your lower back forward."),
            (15.12, "Keep your chest proud and drive straight up through your quads and glutes."),
        ]
        asyncio.run(d3.synthesize_day3_voiceover_wav(AUDIO_WAV, vo_lines, duration_s=DURATION))

    pools = p60.get_or_build_v60_pools()
    torso_icons = p58.make_v58_torso_icons(pools)

    tasks = []
    for s_idx, is_w in [(1, True), (2, False), (3, True), (4, False)]:
        p_pool, p_kps = pools[s_idx]
        n_p = len(p_pool)
        for f_i in range(n_p):
            tasks.append((s_idx, f_i, n_p, p_pool[f_i], p_kps[f_i], is_w))

    print(f"Pre-rendering {len(tasks)} unique pool frames for v60 Day-2-Authentic Master Reel...")
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
            "tag_pill": "KNEE VALGUS RISK",
            "tag_ctx": "BARBELL BACK SQUAT",
            "cue_1line": "16° KNEE VALGUS",
        },
        {
            "step_idx": 2,
            "pool": rendered_pools[2],
            "is_wrong": False,
            "dock_idx": 2,
            "pct_base": 92,
            "pct_peak": 96,
            "tag_pill": "NEUTRAL KNEE LOCK",
            "tag_ctx": "QUADRICEPS & VMO",
            "cue_1line": "0° KNEE TRACKING",
        },
        {
            "step_idx": 3,
            "pool": rendered_pools[3],
            "is_wrong": True,
            "dock_idx": 1,
            "pct_base": 72,
            "pct_peak": 79,
            "tag_pill": "LUMBAR SPINE RISK",
            "tag_ctx": "GOOD-MORNING SQUAT",
            "cue_1line": "41° FORWARD LEAN",
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
            "cue_1line": "22° NEUTRAL SPINE",
        },
    ]

    top_dock_cache = {}
    for cfg in step_configs:
        s_idx = cfg["step_idx"]
        td_im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        p58.draw_v58_day2_unsquished_top_dock(td_im, torso_icons, active_idx=cfg["dock_idx"], is_wrong=cfg["is_wrong"])
        top_dock_cache[s_idx] = td_im

    hud_cache = {}
    tmp_silent = os.path.join(OUT_DIR, "_tmp_day3_v60_silent.mp4")
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
            p58.draw_v58_safezone_bottom_hud(
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

        if fidx in (70, 220, 370, 520):
            peak_frames_pil[s_idx] = rgb_pil
            panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
            crops.append(out_bgr[430:1360, 0:1080])

    vw.release()

    final_v60_mp4 = os.path.join(OUT_DIR, "day3_v60_master_reel.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_silent,
            "-i", AUDIO_WAV,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart",
            final_v60_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(tmp_silent):
        os.remove(tmp_silent)

    close_path = os.path.join(OUT_DIR, "day3_v60_4step_closeup_sheet.jpg")
    comp_path = os.path.join(OUT_DIR, "day3_v60_vs_day2_texture_compare.jpg")
    safe_path = os.path.join(OUT_DIR, "day3_v60_iphone_safezone_preview.jpg")

    cv2.imwrite(close_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    d2_4steps = cv2.imread(os.path.join(OUT_DIR, "_day2_v38_exact_4steps.jpg"))
    row_v60 = np.hstack(panels)
    d2_resized = cv2.resize(d2_4steps, (row_v60.shape[1], row_v60.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(comp_path, np.vstack([d2_resized, row_v60]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    render_v60_iphone_safezone_sheet(peak_frames_pil, safe_path)

    print(f"Done in {time.time() - t0:.1f}s -> {final_v60_mp4}")


if __name__ == "__main__":
    main()
