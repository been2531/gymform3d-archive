#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_v54_master_reel.py`)
=======================================================
Renders the final 600-frame `day3_v54_master_reel.mp4` (`1080x1920`, 30fps, 20.0s)
and the 4-panel comparison sheet `day3_v54_difference_sheet.jpg` + `day3_v54_closeup_sheet.jpg`:
1. Unmistakable Full-Body Biomechanical Posture Contrast (`차이가 명확하게`):
   - Step 1 (`18° KNEE VALGUS`) vs Step 2 (`0° KNEES TRACK TOES`):
     * Step 1 bends the entire femur-tibia hinge sharply INWARD (`>   <`, knees caving together) with vivid crimson
       inner-thigh/knee overload.
     * Step 2 tracks the knees WIDE OUTWARD over the toes (`<   >`) with vivid emerald VMO & glute activation.
   - Step 3 (`+28° FORWARD LEAN & HEELS LIFTING`) vs Step 4 (`95° PARALLEL DRIVE`):
     * Step 3 rotates the entire upper body (spine, chest, head, barbell) forward by `+19.5°` around the hip hinge
       and lifts the heel `68px` off the floor.
     * Step 4 locks the torso upright (`-7.5°` counter-rotation), hits `95°` parallel thigh depth, and plants the heel
       flat on the solid emerald floor bar.
2. 100% Smooth Closed-Polygon Arcs & Joint-Welded Vectors (`◎───▶`):
   - Every arrow is welded directly to the `◎` joint ring (`start_offset_1x = 9.5px`, zero gap!) or spans strictly
     from the Plumb Reference Line right into the spine `Shoulder ◎` via a closed-polygon smooth annular arc.
   - Pure White (`#FFFFFF`) `64pt` numbers + `24pt` sub-labels inside `SAFE_X_LEFT = 96` with zero underline bars.
"""

import os
import sys
import math
import pickle
import subprocess
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/daily_info_reels")
import build_day3_v39_squat as d3
import preview_v54_joint_anchored_master as p54

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_PKL = "/workspace/daily_info_reels/_cache_day3_v44_pools.pkl"
AUDIO_WAV = "/workspace/daily_info_reels/output/day3_v44_voiceover.wav"
W, H = 1080, 1920
FPS = 30
TOTAL_FRAMES = 600


def refine_step_keypoints(kp_cur: dict, step_idx: int, u: float) -> dict:
    """Locks `sh` in Steps 3 & 4 directly onto the glowing thoracic spine axis."""
    kp_fixed = dict(kp_cur)
    if step_idx == 3:
        sh_x, sh_y = kp_cur["sh"]
        kp_fixed["sh"] = (sh_x - 58.0 * u - 18.0 * (1.0 - u), sh_y + 12.0 * u)
    elif step_idx == 4:
        sh_x, sh_y = kp_cur["sh"]
        kp_fixed["sh"] = (sh_x - 24.0, sh_y + 8.0)
    return kp_fixed


def main():
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

    # Weld arrows directly into the `◎` ring (`start_offset_1x = 9.2`)
    orig_arrow_fn = p54.draw_joint_anchored_vector_arrow_ss
    def welded_arrow_fn(rd, joint_pt_ss, tip_pt_ss, col_main, col_light, ss=4, start_offset_1x=9.2):
        return orig_arrow_fn(rd, joint_pt_ss, tip_pt_ss, col_main, col_light, ss=ss, start_offset_1x=start_offset_1x)
    p54.draw_joint_anchored_vector_arrow_ss = welded_arrow_fn

    step_configs = [
        (1, pool_s1, kp_s1, True, 0, 84, "LUMBAR & KNEE RISK", "BARBELL BACK SQUAT", "18° KNEE VALGUS"),
        (2, pool_s2, kp_s2, False, 2, 96, "NEUTRAL KNEE LOCK", "GLUTE MEDIUS & VMO", "0° KNEES TRACK TOES"),
        (3, pool_s3, kp_s3, True, 1, 79, "ANTERIOR FORCE LEAK", "HEEL LIFT & FORWARD DIVE", "+28° FORWARD LEAN"),
        (4, pool_s4, kp_s4, False, 2, 98, "FULL QUAD & GLUTE · 4×10", "QUADRICEPS & GLUTES", "95° PARALLEL DRIVE"),
    ]

    # Pre-cache static top docks and bottom HUDs per step for fast 600-frame rendering
    top_dock_cache = {}
    bot_hud_cache = {}
    for s_idx, _, _, is_wrong, dock_idx, pct_val, tag_pill, tag_ctx, cue_1line in step_configs:
        td_im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d3.draw_day3_v39_safezone_top_dock(
            td_im,
            torso_icons=torso_icons,
            active_idx=dock_idx,
            is_wrong=is_wrong,
            exercise_name="BARBELL BACK SQUAT",
            muscle_group="[ LEGS · QUADS & GLUTES ]",
        )
        top_dock_cache[s_idx] = td_im

        bh_im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d3.draw_day3_v39_safezone_bottom_hud(
            bh_im,
            {
                "is_wrong": is_wrong,
                "pct": pct_val,
                "tag_pill": tag_pill,
                "tag_ctx": tag_ctx,
                "cue_1line": cue_1line,
            },
        )
        bot_hud_cache[s_idx] = bh_im

    # 1. Render 4-panel peak comparison sheet & close-up sheet
    panels, crops = [], []
    for s_idx, pool, kps, is_wrong, _, _, _, _, _ in step_configs:
        idx = int(len(pool) * 0.88)
        rgb_frame, kp_cur = p54.apply_unmistakable_posture_contrast(pool[idx].copy(), kps[idx], step_idx=s_idx, u=1.0)
        kp_cur = refine_step_keypoints(kp_cur, s_idx, 1.0)
        canvas = Image.fromarray(rgb_frame).convert("RGBA")
        p54.render_v54_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=1.0)
        canvas.alpha_composite(top_dock_cache[s_idx])
        canvas.alpha_composite(bot_hud_cache[s_idx])
        out_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        panels.append(cv2.resize(out_bgr, (540, 960), interpolation=cv2.INTER_AREA))
        crops.append(out_bgr[520:1620, 80:1000])

    sheet_path = os.path.join(OUT_DIR, "day3_v54_difference_sheet.jpg")
    crop_path = os.path.join(OUT_DIR, "day3_v54_closeup_sheet.jpg")
    cv2.imwrite(sheet_path, np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], panels[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(crop_path, np.vstack([np.hstack([crops[0], crops[1]]), np.hstack([crops[2], crops[3]])]), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print("Saved sheets:", sheet_path, crop_path)

    # 2. Pre-render the unique ping-pong frames for each step (only ~55-72 unique frames per step!)
    #    Wait: since each 150-frame step ping-pongs through `pool` (length `L`), `idx` and `u` depend ONLY on
    #    the pool index `idx`! So pre-rendering the `L` unique frames per step reduces 600 frame renders to ~249 unique frames!
    unique_rendered_steps = {}
    for s_idx, pool, kps, is_wrong, _, _, _, _, _ in step_configs:
        L = len(pool)
        rendered_pool = []
        for idx in range(L):
            u = float(np.clip((idx - L * 0.15) / max(1.0, L * 0.72), 0.0, 1.0))
            u_smooth = 0.5 - 0.5 * math.cos(math.pi * u)
            rgb_frame, kp_cur = p54.apply_unmistakable_posture_contrast(pool[idx], kps[idx], step_idx=s_idx, u=u_smooth)
            kp_cur = refine_step_keypoints(kp_cur, s_idx, u_smooth)
            canvas = Image.fromarray(rgb_frame).convert("RGBA")
            p54.render_v54_callouts(canvas, step_idx=s_idx, kp=kp_cur, is_wrong=is_wrong, rep_amount=u_smooth)
            canvas.alpha_composite(top_dock_cache[s_idx])
            canvas.alpha_composite(bot_hud_cache[s_idx])
            rendered_pool.append(cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR))
        unique_rendered_steps[s_idx] = rendered_pool
        print(f"Pre-rendered Step {s_idx}: {L} unique frames")

    # 3. Assemble the 600-frame 20.0s video with smooth ping-pong motion
    tmp_mp4 = os.path.join(OUT_DIR, "_tmp_day3_v54_silent.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day3_v54_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    for f_idx in range(TOTAL_FRAMES):
        s_idx = min(4, (f_idx // 150) + 1)
        local_f = f_idx % 150
        r_pool = unique_rendered_steps[s_idx]
        L = len(r_pool)
        phase = (local_f / 149.0) * 2.0
        ping_pong = phase if phase <= 1.0 else (2.0 - phase)
        ping_pong_smooth = 0.5 - 0.5 * math.cos(math.pi * ping_pong)
        p_idx = int(round(ping_pong_smooth * (L - 1)))
        p_idx = max(0, min(L - 1, p_idx))
        vw.write(r_pool[p_idx])
    vw.release()

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_mp4,
        "-i", AUDIO_WAV,
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        final_mp4,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(tmp_mp4):
        os.remove(tmp_mp4)
    print("Saved final video:", final_mp4)


if __name__ == "__main__":
    main()
