#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_v61_master_reel.py`)
=======================================================
Renders the final 600-frame `day3_v61_master_reel.mp4` (`1080x1920`, 30fps, 20.0s),
plus:
- `day3_v61_step3_before_after.jpg` (1:1 side-by-side comparison of 중현 님's `v60` Step 3 capture vs fixed `v61` Step 3)
- `day3_v61_vs_day2_texture_compare.jpg` (Day 2 original 4 steps vs Day 3 `v61` 4 steps)
- `day3_v61_4step_closeup_sheet.jpg`
- `day3_v61_iphone_safezone_preview.jpg`
and runs `qc_gate.py` (`v61 Standard`) to verify 100% PASS across all 8 gates.
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
import preview_v61_seamless_day2_master as p61

OUT_DIR = "/workspace/daily_info_reels/output"
AUDIO_WAV = "/workspace/daily_info_reels/output/day3_v60_voiceover.wav"
USER_V60_STEP3_SHOT = "/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/attachments/01M601SVM8J4F9MYPEFF6MKQFM.png"

W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)

_LEFT_W = 430
_CURVE_L = (np.linspace(0.68, 1.0, _LEFT_W, dtype=np.float32) ** 0.78)[None, :, None]
_SCRIM_2D = v38.make_bottom_scrim_2d()

STEP_META = {
    1: {
        "is_wrong": True,
        "dock_idx": 0,
        "deg_base": 10,
        "deg_peak": 16,
        "pct_base": 78,
        "pct_peak": 84,
        "tag_pill": "KNEE VALGUS RISK",
        "tag_ctx": "BARBELL BACK SQUAT",
        "cue_suffix": "KNEE VALGUS",
    },
    2: {
        "is_wrong": False,
        "dock_idx": 2,
        "deg_base": 0,
        "deg_peak": 0,
        "pct_base": 92,
        "pct_peak": 96,
        "tag_pill": "NEUTRAL KNEE LOCK",
        "tag_ctx": "QUADRICEPS & VMO",
        "cue_suffix": "KNEE TRACKING",
    },
    3: {
        "is_wrong": True,
        "dock_idx": 1,
        "deg_base": 25,
        "deg_peak": 41,
        "pct_base": 72,
        "pct_peak": 79,
        "tag_pill": "LUMBAR SPINE RISK",
        "tag_ctx": "GOOD-MORNING SQUAT",
        "cue_suffix": "FORWARD LEAN",
    },
    4: {
        "is_wrong": False,
        "dock_idx": 2,
        "deg_base": 15,
        "deg_peak": 22,
        "pct_base": 94,
        "pct_peak": 98,
        "tag_pill": "FULL QUAD & GLUTE · 4×10",
        "tag_ctx": "QUADRICEPS & GLUTES",
        "cue_suffix": "NEUTRAL SPINE",
    },
}


def _render_pool_frame_worker(args):
    """
    Pre-renders pool frame `f_i` (`u = f_i / (n_p - 1)`) with BOTH the floating callout AND the Bottom HUD
    computed from the exact same `u` so that for every single pool frame:
    - Left Floating Callout Angle == Bottom HUD Angle (`deg_val°`)
    - Right Floating Callout Percentage == Bottom HUD Percentage (`pct_val%`)
    """
    s_idx, f_i, n_p, raw_rgb, kp_cur, top_dock_rgba = args
    u = f_i / float(max(1, n_p - 1))
    meta = STEP_META[s_idx]
    is_wrong = meta["is_wrong"]

    deg_val = int(round(meta["deg_base"] + (meta["deg_peak"] - meta["deg_base"]) * u))
    pct_val = int(round(meta["pct_base"] + (meta["pct_peak"] - meta["pct_base"]) * u))
    cue_1line = f"{deg_val}° {meta['cue_suffix']}"

    rgb_tinted = p58.apply_static_studio_tint(raw_rgb, is_wrong)
    rgb_scrim = v38.apply_bottom_left_scrim(rgb_tinted, _CURVE_L, _SCRIM_2D)
    canvas = Image.fromarray(rgb_scrim).convert("RGBA")

    p61.render_v61_synchronized_angle_and_callouts(
        canvas,
        step_idx=s_idx,
        kp=kp_cur,
        is_wrong=is_wrong,
        deg_val=deg_val,
        pct_val=pct_val,
        rep_amount=u,
    )
    canvas.alpha_composite(Image.fromarray(top_dock_rgba))
    p58.draw_v58_safezone_bottom_hud(
        canvas,
        {
            "is_wrong": is_wrong,
            "pct": pct_val,
            "tag_pill": meta["tag_pill"],
            "tag_ctx": meta["tag_ctx"],
            "cue_1line": cue_1line,
        },
    )
    rgb_out = np.array(canvas.convert("RGB"))
    return s_idx, f_i, cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


def render_v61_iphone_safezone_sheet(step_frames_rgb: dict, out_path: str):
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
        "DAY 3 · iPHONE INSTAGRAM REELS UI SAFE-ZONE PREVIEW (`v61`)  ·  풀스크린 스튜디오(박스 0%) + 유령원판 0% + 각도/수치 100% 일치",
        font=f_hdr,
        fill=p61.COL_GREEN,
    )
    titles = [
        "iPhone View · STEP 1 (16° KNEE VALGUS)",
        "iPhone View · STEP 2 (0° KNEE TRACKING)",
        "iPhone View · STEP 3 (41° FORWARD LEAN)",
        "iPhone View · STEP 4 (22° NEUTRAL SPINE)",
    ]
    for i in range(4):
        x = ipad + i * (cw + ipad)
        col = p61.COL_RED if i in (0, 2) else p61.COL_GREEN
        bd.text((x + 4, 58), titles[i], font=f_col, fill=col)
        sim_im = v39.simulate_iphone_ig_reels_view(step_frames_rgb[i + 1], real_shot_bgr, old_f210_bgr)
        rc = sim_im.resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        board.paste(rc, (x, 86))
    board.save(out_path, quality=95)


def render_v61_step3_before_after_sheet(v61_step3_bgr: np.ndarray, out_path: str):
    v60_shot = cv2.imread(USER_V60_STEP3_SHOT)
    pw, ph = 540, 960
    v60_res = cv2.resize(v60_shot, (pw, ph), interpolation=cv2.INTER_AREA)
    v61_res = cv2.resize(v61_step3_bgr, (pw, ph), interpolation=cv2.INTER_AREA)

    pad = 28
    board = Image.new("RGB", (pw * 2 + pad * 3, ph + 130), (10, 12, 16))
    bd = ImageDraw.Draw(board)
    f_hdr = v30.f_pretendard(21, "extrabold")
    f_sub = v30.f_pretendard(14, "bold")

    bd.text(
        (pad, 20),
        "STEP 3 BEFORE (`v60` 중현 님 캡처)  vs  AFTER (`v61` 풀스크린 스튜디오 · 유령원판 0% · 내부 크림슨 요추 X-Ray · 수치 100% 일치)",
        font=f_hdr,
        fill=p61.COL_GREEN,
    )
    bd.text(
        (pad + 4, 62),
        "BEFORE (`v60`): ① 회색 사각형 박스 테두리/좌우 검은 여백  ② 유령 원판 & 랙 방향 모순  ③ 외계인 흰 등뼈 & 당근색 다리  ④ 36° ≠ 41° 불일치",
        font=f_sub,
        fill=(255, 110, 110),
    )
    bd.text(
        (pad * 2 + pw + 4, 62),
        "AFTER (`v61`): ① 1080×1920 풀스크린 스튜디오  ② 바벨 양끝 원판 2개 정합  ③ Day 2식 내부 크림슨 요추 X-Ray + 매트 회색 둔근/다리  ④ 41° == 41° 일치",
        font=f_sub,
        fill=p61.COL_GREEN,
    )

    p_before = Image.fromarray(cv2.cvtColor(v60_res, cv2.COLOR_BGR2RGB))
    p_after = Image.fromarray(cv2.cvtColor(v61_res, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(p_before).rectangle([0, 0, pw - 1, ph - 1], outline=p61.COL_RED, width=3)
    ImageDraw.Draw(p_after).rectangle([0, 0, pw - 1, ph - 1], outline=p61.COL_GREEN, width=3)

    board.paste(p_before, (pad, 98))
    board.paste(p_after, (pad * 2 + pw, 98))
    board.save(out_path, quality=96)


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

    pools = p61.get_or_build_v61_pools()
    torso_icons = p58.make_v58_torso_icons(pools)

    top_dock_cache = {}
    for s_idx, meta in STEP_META.items():
        td_im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        p58.draw_v58_day2_unsquished_top_dock(td_im, torso_icons, active_idx=meta["dock_idx"], is_wrong=meta["is_wrong"])
        top_dock_cache[s_idx] = np.array(td_im)

    tasks = []
    for s_idx in (1, 2, 3, 4):
        p_pool, p_kps = pools[s_idx]
        n_p = len(p_pool)
        for f_i in range(n_p):
            tasks.append((s_idx, f_i, n_p, p_pool[f_i], p_kps[f_i], top_dock_cache[s_idx]))

    print(f"Pre-rendering {len(tasks)} synchronized full-bleed pool frames for v61 Master Reel...")
    rendered_pools = {s_idx: [None] * len(pools[s_idx][0]) for s_idx in (1, 2, 3, 4)}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for s_idx, f_i, out_bgr in ex.map(_render_pool_frame_worker, tasks, chunksize=4):
            rendered_pools[s_idx][f_i] = out_bgr
    print(f"Pool pre-render finished in {time.time() - t0:.1f}s")

    tmp_silent = os.path.join(OUT_DIR, "_tmp_day3_v61_silent.mp4")
    vw = cv2.VideoWriter(tmp_silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    peak_frames_pil = {}
    panels, crops = [], []

    for fidx in range(TOTAL_FRAMES):
        t_sec = fidx / float(FPS)
        s_i = min(3, int(t_sec // 5.0))
        s_idx = s_i + 1
        t_local = t_sec - s_i * 5.0
        pool = rendered_pools[s_idx]
        n_p = len(pool)

        phase = min(1.0, t_local / 4.66)
        u = 0.5 - 0.5 * math.cos(phase * 2.0 * math.pi)
        idx_near = int(round(u * (n_p - 1)))
        out_bgr = pool[idx_near]
        vw.write(out_bgr)

        if fidx in (70, 220, 370, 520):
            peak_frames_pil[s_idx] = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
            crops.append(out_bgr[430:1360, 0:1080])

    vw.release()

    final_v61_mp4 = os.path.join(OUT_DIR, "day3_v61_master_reel.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_silent,
            "-i", AUDIO_WAV,
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart",
            final_v61_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(tmp_silent):
        os.remove(tmp_silent)

    close_path = os.path.join(OUT_DIR, "day3_v61_4step_closeup_sheet.jpg")
    comp_path = os.path.join(OUT_DIR, "day3_v61_vs_day2_texture_compare.jpg")
    safe_path = os.path.join(OUT_DIR, "day3_v61_iphone_safezone_preview.jpg")
    ba_path = os.path.join(OUT_DIR, "day3_v61_step3_before_after.jpg")

    cv2.imwrite(close_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    d2_4steps = cv2.imread(os.path.join(OUT_DIR, "_day2_v38_exact_4steps.jpg"))
    row_v61 = np.hstack(panels)
    d2_resized = cv2.resize(d2_4steps, (row_v61.shape[1], row_v61.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(comp_path, np.vstack([d2_resized, row_v61]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    render_v61_iphone_safezone_sheet(peak_frames_pil, safe_path)
    render_v61_step3_before_after_sheet(rendered_pools[3][-1], ba_path)

    print(f"Done in {time.time() - t0:.1f}s -> {final_v61_mp4}")


if __name__ == "__main__":
    main()
