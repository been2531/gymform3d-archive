#!/usr/bin/env python3
"""
Day 3 — BARBELL BACK SQUAT (`build_v48_3d_ribbon_master.py`)
============================================================
Renders the full 20.0s (600 frames @ 30fps) Day 3 Master Reel with `v48` True 3D
Cylinder-Wrap Ribbon Arrows & Meaningful 3-Point Kinematic Joint Markers, plus
3 Dynamic 3D Motion Upgrades:
1. Frameless Pure White Degree/Percentage Callouts (ZERO underline bars):
   - 64pt Extrabold Pure White number + 24pt Extrabold Pure White sub-label directly below.
2. Live Real-Time 3-Point Goniometer Joint Chain (`실시간 3점 관절 각도계 연동`):
   - Every marker dot (`고관절 — 무릎 피벗 — 발목` / `어깨 — 고관절 피벗 — 무릎`) is connected by a
     3D kinematic axis rod and contrasted against the Ideal Reference Axis (`기준 정렬축`),
     with the angle sector wedge and degree readout dynamically tracking the squat descent!
3. Animated 3D Cylinder-Wrap Ribbon Arrows with Traveling Specular Pulse (`3D 원통 곡면 드로우온 & 하이라이트 플로우`):
   - The 3D ribbon curls out from behind the limb contour (`Z < 0` dark inner fold), wraps across
     the front of the 3D thigh/knee/torso (`Z > 0`), and sweeps dynamically with the squat rep
     while a studio specular light wave travels along the 3D ribbon toward the sculpted arrowhead.
4. 3D Perspective Floor-Anchor Ring (`3D 지면 반력 앵커 링`):
   - Anchors the distal ankle/foot marker to the 3D studio floor plane so the viewer immediately
     perceives the 3D ground plane and vertical force transfer.
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
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v30_pure_graphic_numbers as v30
import build_day3_v39_squat as d3
import preview_v48_true_3d_torus_ribbons as v48

OUT_DIR = "/workspace/daily_info_reels/output"
CACHE_PKL = "/workspace/daily_info_reels/_cache_day3_v44_pools.pkl"
W, H = 1080, 1920
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)


def draw_3d_floor_force_ring(
    rd: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    rx: float,
    ry: float,
    col_main: tuple[int, int, int],
    col_dark: tuple[int, int, int],
    is_lifting: bool = False,
    rep_amount: float = 0.85,
    ss: int = 4,
):
    """
    Renders a 3D perspective-foreshortened floor-plane ellipse under the foot/ankle
    so the 3-point joint chain (`Hip -> Knee -> Ankle`) visibly terminates on the 3D floor plane.
    """
    cx, cy = center_xy[0] * ss, center_xy[1] * ss
    rx_s = rx * (0.92 + 0.10 * rep_amount) * ss
    ry_s = ry * (0.92 + 0.10 * rep_amount) * ss

    # Soft floor contact shadow
    rd.ellipse(
        [cx - rx_s * 1.06, cy - ry_s * 1.06 + 3 * ss, cx + rx_s * 1.06, cy + ry_s * 1.06 + 3 * ss],
        fill=(0, 0, 0, 135),
    )
    # Translucent 3D ground pad
    rd.ellipse(
        [cx - rx_s, cy - ry_s, cx + rx_s, cy + ry_s],
        fill=(*col_main, 38),
        outline=(*col_dark, 210),
        width=int(2.6 * ss),
    )
    # Front-half 3D lit rim (`0° .. 180°` is closer to camera)
    phis = np.linspace(-math.radians(15), math.radians(195), 48)
    pts_front = [(cx + rx_s * math.cos(p), cy + ry_s * math.sin(p)) for p in phis]
    for i in range(len(pts_front) - 1):
        z_front = math.sin(0.5 * (phis[i] + phis[i + 1]))
        w_line = int(round((2.2 + 1.6 * max(0.0, z_front)) * ss))
        rd.line([pts_front[i], pts_front[i + 1]], fill=(*col_main, 245), width=w_line)
        if z_front > 0.55:
            rd.line([pts_front[i], pts_front[i + 1]], fill=(255, 255, 255, 210), width=max(1, ss))

    if is_lifting:
        # Inner dashed warning ring on the floor showing where the heel lifted away from
        rx_in, ry_in = rx_s * 0.58, ry_s * 0.58
        rd.ellipse(
            [cx - rx_in, cy - ry_in, cx + rx_in, cy + ry_in],
            outline=(255, 255, 255, 195),
            width=int(1.4 * ss),
        )


def render_v48_dynamic_frame(
    rgb_frame: np.ndarray,
    step_idx: int,
    kp: dict,
    is_wrong: bool,
    rep_amount: float,
    pulse_phase: float,
) -> Image.Image:
    """
    Renders one frame of the v48 scene with dynamic 3D ribbon sweep (`sweep_frac`),
    3D floor-plane anchor rings, and live-tracked goniometer degree readouts.
    """
    SS = 4
    rig = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig)

    col_main = v48.COL_RED if is_wrong else v48.COL_GREEN
    col_dark = v48.COL_RED_DARK if is_wrong else v48.COL_GREEN_DARK
    col_soft = v48.COL_RED_SOFT if is_wrong else v48.COL_GREEN_SOFT

    sh = kp["sh"]
    hip_r = kp["hip_r"]
    hip_l = kp["hip_l"]
    kn_r = kp["kn_r"]
    kn_l = kp["kn_l"]
    ank_r = kp["ank_r"]
    ank_l = kp["ank_l"]

    # Dynamic wrap sweep factor (ribbon actively wraps around the limb as the squat deepens)
    sweep_k = 0.62 + 0.38 * min(1.0, max(0.0, rep_amount * 1.15))

    if step_idx == 1:
        # 3D Floor force rings under both ankles
        draw_3d_floor_force_ring(rd, (ank_r[0], ank_r[1] + 22), 48, 14, col_main, col_dark, False, rep_amount, SS)
        draw_3d_floor_force_ring(rd, (ank_l[0], ank_l[1] + 22), 48, 14, col_main, col_dark, False, rep_amount, SS)

        # Ideal vertical alignment point above `kn_r`
        valgus_dx = 16.0 + 34.0 * rep_amount
        ref_valgus_pt = (kn_r[0] - valgus_dx, kn_r[1] - 125.0)
        v48.draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_r, p_pivot=kn_r, p_dist=ank_r, p_ref_end=ref_valgus_pt,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        v48.draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_l, p_pivot=kn_l, p_dist=ank_l, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # Dynamic 3D Cylinder-Wrap Ribbons curling INWARD around both knees
        phi_e_r = 218.0 + (22.0 - 218.0) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_r[0] - 12.0, kn_r[1] + 28.0),
            rx=68.0, ry=26.0, tilt_deg=12.0,
            phi_start_deg=218.0, phi_end_deg=phi_e_r,
            ribbon_thick_px=15.0, pitch_dy=14.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        phi_e_l = -38.0 + (158.0 - (-38.0)) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_l[0] + 12.0, kn_l[1] + 28.0),
            rx=68.0, ry=26.0, tilt_deg=-12.0,
            phi_start_deg=-38.0, phi_end_deg=phi_e_l,
            ribbon_thick_px=15.0, pitch_dy=14.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

    elif step_idx == 2:
        draw_3d_floor_force_ring(rd, (ank_r[0], ank_r[1] + 22), 52, 15, col_main, col_dark, False, rep_amount, SS)
        draw_3d_floor_force_ring(rd, (ank_l[0], ank_l[1] + 22), 52, 15, col_main, col_dark, False, rep_amount, SS)

        v48.draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_r, p_pivot=kn_r, p_dist=ank_r, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        v48.draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_l, p_pivot=kn_l, p_dist=ank_l, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        # Dynamic 3D Cylinder-Wrap Ribbons curling OUTWARD around both knees
        phi_e_r = -25.0 + (202.0 - (-25.0)) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_r[0] - 8.0, kn_r[1] + 26.0),
            rx=70.0, ry=26.0, tilt_deg=-10.0,
            phi_start_deg=-25.0, phi_end_deg=phi_e_r,
            ribbon_thick_px=15.0, pitch_dy=-10.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        phi_e_l = 205.0 + (-22.0 - 205.0) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(kn_l[0] + 8.0, kn_l[1] + 26.0),
            rx=70.0, ry=26.0, tilt_deg=10.0,
            phi_start_deg=205.0, phi_end_deg=phi_e_l,
            ribbon_thick_px=15.0, pitch_dy=-10.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

    elif step_idx == 3:
        # Floor warning ring where the heel lifts away from the floor
        draw_3d_floor_force_ring(rd, (ank_r[0] - 6, ank_r[1] + 28), 54, 16, col_main, col_dark, True, rep_amount, SS)

        ideal_upright_sh = (hip_r[0] + 22.0, hip_r[1] - 185.0)
        v48.draw_meaningful_3point_joint_chain(
            rd, p_prox=sh, p_pivot=hip_r, p_dist=kn_r, p_ref_end=ideal_upright_sh,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        v48.draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_r, p_pivot=kn_r, p_dist=ank_r, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        phi_e_sh = 215.0 + (15.0 - 215.0) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(sh[0] + 18.0, sh[1] + 15.0),
            rx=78.0, ry=32.0, tilt_deg=52.0,
            phi_start_deg=215.0, phi_end_deg=phi_e_sh,
            ribbon_thick_px=15.5, pitch_dy=24.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        phi_e_ank = -30.0 + (165.0 - (-30.0)) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(ank_r[0] - 12.0, ank_r[1] - 12.0),
            rx=52.0, ry=20.0, tilt_deg=-78.0,
            phi_start_deg=-30.0, phi_end_deg=phi_e_ank,
            ribbon_thick_px=13.5, pitch_dy=-28.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

    else:
        draw_3d_floor_force_ring(rd, (ank_r[0] + 4, ank_r[1] + 24), 58, 17, col_main, col_dark, False, rep_amount, SS)

        horiz_parallel_ref = (hip_r[0] + 145.0, hip_r[1])
        v48.draw_meaningful_3point_joint_chain(
            rd, p_prox=sh, p_pivot=hip_r, p_dist=kn_r, p_ref_end=horiz_parallel_ref,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        v48.draw_meaningful_3point_joint_chain(
            rd, p_prox=hip_r, p_pivot=kn_r, p_dist=ank_r, p_ref_end=None,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

        phi_e_sh = -32.0 + (168.0 - (-32.0)) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(sh[0] + 22.0, sh[1] + 18.0),
            rx=78.0, ry=30.0, tilt_deg=-68.0,
            phi_start_deg=-32.0, phi_end_deg=phi_e_sh,
            ribbon_thick_px=15.5, pitch_dy=-26.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )
        phi_e_ank = 210.0 + (25.0 - 210.0) * sweep_k
        v48.draw_3d_cylindrical_wrap_ribbon(
            rd,
            center_xy=(ank_r[0] + 4.0, ank_r[1] + 14.0),
            rx=56.0, ry=18.0, tilt_deg=-6.0,
            phi_start_deg=210.0, phi_end_deg=phi_e_ank,
            ribbon_thick_px=13.0, pitch_dy=6.0,
            col_main=col_main, col_dark=col_dark, col_soft=col_soft, ss=SS
        )

    rig_1x = rig.resize((W, H), Image.Resampling.LANCZOS)
    canvas = Image.fromarray(rgb_frame).convert("RGBA")
    canvas.alpha_composite(rig_1x)

    # Leader lines + Frameless Pure White Badges (ZERO underline bars, live goniometer numbers!)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    def _leader(p_joint, p_elbow, p_badge):
        sd.line([p_joint, p_elbow, p_badge], fill=(0, 0, 0, 220), width=4)
        od.line([p_joint, p_elbow, p_badge], fill=(255, 255, 255, 220), width=2)
        bx, by = p_badge
        od.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(255, 255, 255, 255))

    if step_idx == 1:
        deg_live = int(round(6 + 12 * rep_amount))
        pct_live = int(round(78 + 6 * rep_amount))
        _leader((kn_r[0] - 18, kn_r[1] - 8), (310, 1075), (335, 1038))
        _leader((kn_l[0] + 18, kn_l[1] - 8), (715, 955), (738, 898))
        v48.draw_v48_frameless_white_badge(od, sd, (335, 1005), f"{deg_live}°", "VALGUS CAVE", align="right")
        v48.draw_v48_frameless_white_badge(od, sd, (715, 855), f"{pct_live}%", "ACL & MCL LOAD", align="left")
    elif step_idx == 2:
        pct_live = int(round(91 + 5 * rep_amount))
        _leader((kn_r[0] - 18, kn_r[1] - 8), (315, 1060), (340, 1022))
        _leader((kn_l[0] + 18, kn_l[1] - 8), (710, 955), (730, 898))
        v48.draw_v48_frameless_white_badge(od, sd, (345, 990), "0°", "KNEES TRACK TOES", align="right")
        v48.draw_v48_frameless_white_badge(od, sd, (705, 855), f"{pct_live}%", "VMO & GLUTE LOCK", align="left")
    elif step_idx == 3:
        deg_live = int(round(12 + 16 * rep_amount))
        pct_live = int(round(73 + 6 * rep_amount))
        _leader((hip_r[0] - 18, hip_r[1] + 10), (315, 1005), (342, 1048))
        _leader((kn_r[0] + 16, kn_r[1] - 10), (705, 955), (725, 898))
        v48.draw_v48_frameless_white_badge(od, sd, (348, 1015), f"+{deg_live}°", "HEELS LIFTING", align="right")
        v48.draw_v48_frameless_white_badge(od, sd, (700, 855), f"{pct_live}%", "LUMBAR & PATELLA", align="left")
    else:
        deg_live = int(round(125 - 30 * rep_amount))
        pct_live = int(round(93 + 5 * rep_amount))
        _leader((hip_r[0] - 18, hip_r[1] + 10), (310, 995), (338, 1038))
        _leader((kn_r[0] + 16, kn_r[1] - 10), (715, 955), (738, 898))
        v48.draw_v48_frameless_white_badge(od, sd, (345, 1005), f"{deg_live}°", "PARALLEL DEPTH", align="right")
        v48.draw_v48_frameless_white_badge(od, sd, (715, 855), f"{pct_live}%", "QUADS & GLUTES", align="left")

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=5)))
    canvas.alpha_composite(ov)
    return canvas


def _render_pool_frame_worker(args):
    s_idx, f_i, n_p, rgb_frame, kp_cur, is_wrong = args
    rep_amount = f_i / float(max(1, n_p - 1))
    canvas = render_v48_dynamic_frame(
        rgb_frame,
        step_idx=s_idx,
        kp=kp_cur,
        is_wrong=is_wrong,
        rep_amount=rep_amount,
        pulse_phase=rep_amount,
    )
    return s_idx, f_i, np.array(canvas.convert("RGB"))


def build_v48_master():
    t0 = time.time()
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

    print(f"Pre-rendering {len(tasks)} pool frames with v48 3D ribbon vectors across CPU cores...")
    v48_pools = {1: [None] * len(pool_s1), 2: [None] * len(pool_s2), 3: [None] * len(pool_s3), 4: [None] * len(pool_s4)}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for s_idx, f_i, out_rgb in ex.map(_render_pool_frame_worker, tasks, chunksize=4):
            v48_pools[s_idx][f_i] = out_rgb
    print(f"v48 pool pre-render finished in {time.time() - t0:.1f}s")

    wav_path = os.path.join(OUT_DIR, "day3_v44_voiceover.wav")
    if not os.path.exists(wav_path):
        vo_lines = [
            (0.18, "Stop letting your knees cave inward at the bottom of the squat."),
            (5.12, "Drive both knees wide over your toes and lock your feet flat."),
            (10.12, "Never let your heels lift or fold your lower back forward."),
            (15.12, "Keep your chest proud and drive straight up through your quads and glutes."),
        ]
        asyncio.run(d3.synthesize_day3_voiceover_wav(wav_path, vo_lines, duration_s=DURATION))

    tmp_silent = os.path.join(OUT_DIR, "_tmp_day3_v48_silent.mp4")
    writer = cv2.VideoWriter(tmp_silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    step_configs = [
        {
            "step_idx": 1,
            "pool": v48_pools[1],
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
            "pool": v48_pools[2],
            "is_wrong": False,
            "dock_idx": 2,
            "pct_base": 91,
            "pct_peak": 96,
            "tag_pill": "NEUTRAL KNEE LOCK",
            "tag_ctx": "GLUTE MEDIUS & VMO",
            "cue_1line": "0° KNEES TRACK TOES",
        },
        {
            "step_idx": 3,
            "pool": v48_pools[3],
            "is_wrong": True,
            "dock_idx": 1,
            "pct_base": 73,
            "pct_peak": 79,
            "tag_pill": "ANTERIOR FORCE LEAK",
            "tag_ctx": "HEEL LIFT & FORWARD DIVE",
            "cue_1line": "+28° FORWARD LEAN",
        },
        {
            "step_idx": 4,
            "pool": v48_pools[4],
            "is_wrong": False,
            "dock_idx": 2,
            "pct_base": 93,
            "pct_peak": 98,
            "tag_pill": "FULL QUAD & GLUTE · 4×10",
            "tag_ctx": "QUADRICEPS & GLUTES",
            "cue_1line": "95° PARALLEL DRIVE",
        },
    ]

    rep_peaks = {}
    print("Assembling 600 frames of Day 3 v48 Master Reel...")
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

    final_mp4 = os.path.join(OUT_DIR, "day3_v48_master_reel.mp4")
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
    sheet_path = os.path.join(OUT_DIR, "day3_v48_master_sheet.jpg")
    cv2.imwrite(sheet_path, sheet_2x2, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print(f"Done in {time.time() - t0:.1f}s -> {final_mp4} & {sheet_path}")


if __name__ == "__main__":
    build_v48_master()
