#!/usr/bin/env python3
"""
@demicstory Gym Workout Concept — Day 1 (Upgraded v3): Cable Rope Triceps Pushdown
Improvements based on user feedback ("화살표 그래픽이 아쉬운데? 더 개선할 부분도 고치고, 직관적이면서 단순화해야해"):
  1. Studio-Grade 4x Supersampled 3D Volumetric Tapered Arrows (draw_3d_curved_arrow / draw_3d_volumetric_arrow):
     - Smoothly widening tapered ribbon shaft (tail 9px -> neck 26px) + sculpted swept-back 3D arrowhead (width 58px, len 52px)
     - 3D elevation drop-shadow (+5, +7px), vibrant outer neon bloom, deep bevel edge, luminous core, and glossy 3D center spine highlight.
  2. Radical Visual Simplification ("One Glance, One Point"):
     - Removed all cluttered connecting polylines, dotted cyan tracks, energy beads, and shoulder rings.
     - Wrong (✕, 1.7-4.65s): Clean red overload pulse + single crisp ring on Elbow & Wrist ONLY + 1 bold 3D curved red elbow-flare arrow + 1 compact 3D red wrist-bend arrow + sleek 1/5 Red Battery Meter.
     - Correct (✓, 4.95-8.5s): Single ultra-clean 3D Neon-Green Elbow Lock-On Ring (leaving the glowing 3D triceps unobstructed) + 1 bold 3D curved tapered green forearm-extension arrow + 2 symmetrical 3D tapered outward rope-spread arrows at peak contraction + sleek 5/5 Green Battery Meter.
  3. Smooth 0.0-1.5s macro hook & crisp 4.65-4.95s cross-dissolve with green lock-on flash + automatic temp frame cleanup.
"""
import os
import math
import wave
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1080, 1920
FPS = 30
DURATION = 8.5
TOTAL_FRAMES = int(FPS * DURATION)

OUT_DIR = "/workspace/daily_info_reels/output"
WRONG_VID = os.path.join(OUT_DIR, "gym_day1_wrong_i2v.mp4")
CORRECT_VID = os.path.join(OUT_DIR, "gym_day1_correct_i2v.mp4")

TMP_DIR = "/workspace/daily_info_reels/frames_gym_day1"
os.makedirs(TMP_DIR, exist_ok=True)

OUT_MP4 = os.path.join(OUT_DIR, "demic_gym_day1.mp4")
OUT_KEYFRAMES = os.path.join(OUT_DIR, "demic_gym_day1_keyframes.jpg")
OUT_COMP = os.path.join(OUT_DIR, "demic_gym_vs_original.jpg")


def upscale_1080p(frame_bgr: np.ndarray) -> np.ndarray:
    up = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.15)
    sharp = cv2.addWeighted(up, 1.22, blur, -0.22, 0)
    return sharp


def load_video_frames(vid_path: str) -> list:
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(upscale_1080p(frame))
    cap.release()
    return frames


WRONG_FRAMES = load_video_frames(WRONG_VID)
CORRECT_FRAMES = load_video_frames(CORRECT_VID)


def sample_clip_frame(frames: list, norm_u: float) -> np.ndarray:
    u = float(np.clip(norm_u, 0.0, 1.0))
    pos = u * (len(frames) - 1)
    i0 = int(math.floor(pos))
    i1 = min(len(frames) - 1, i0 + 1)
    frac = pos - i0
    if frac < 1e-4 or i0 == i1:
        return frames[i0].copy()
    return cv2.addWeighted(frames[i0], 1.0 - frac, frames[i1], frac, 0)


def make_gym_thumbs():
    c_rgb = cv2.cvtColor(CORRECT_FRAMES[55], cv2.COLOR_BGR2RGB)
    w_rgb = cv2.cvtColor(WRONG_FRAMES[25], cv2.COLOR_BGR2RGB)
    crops = [
        c_rgb[460:940, 420:900],   # Thumb 0: glowing triceps lateral/long head
        w_rgb[560:1060, 400:900],  # Thumb 1: X-ray elbow/forearm joint overload view
        c_rgb[320:1180, 160:960],  # Thumb 2: wider upper body & cable pulley view
    ]
    thumbs = []
    for cr in crops:
        th = cv2.resize(cr, (96, 96), interpolation=cv2.INTER_AREA)
        thumbs.append(Image.fromarray(th))
    return thumbs


THUMBS = make_gym_thumbs()


def get_wrong_landmarks(norm_u: float):
    """Tracked (elbow_pt, wrist_pt) in 1080x1920 coordinates across Wrong Form I2V clip."""
    us = [0.0, 0.25, 0.50, 0.75, 1.0]
    el_x = np.interp(norm_u, us, [827, 842, 878, 828, 815])
    el_y = np.interp(norm_u, us, [895, 1015, 1045, 965, 890])
    wr_x = np.interp(norm_u, us, [560, 650, 865, 575, 565])
    wr_y = np.interp(norm_u, us, [1025, 1285, 1310, 1175, 1030])
    return (float(el_x), float(el_y)), (float(wr_x), float(wr_y))


def get_correct_landmarks(norm_u: float):
    """Tracked (triceps_peak, elbow_pivot, hands_grip) in 1080x1920 coordinates across Correct Form I2V clip."""
    us = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
    el_x = np.interp(norm_u, us, [735, 744, 755, 765, 776, 779, 774, 754, 732])
    el_y = np.interp(norm_u, us, [815, 823, 832, 835, 834, 834, 835, 833, 824])
    tr_x = np.interp(norm_u, us, [742, 751, 762, 772, 785, 788, 782, 762, 739])
    tr_y = np.interp(norm_u, us, [673, 678, 688, 695, 698, 699, 700, 702, 695])
    gr_x = np.interp(norm_u, us, [415, 402, 465, 653, 790, 822, 737, 486, 412])
    gr_y = np.interp(norm_u, us, [679, 760, 1004, 1152, 1163, 1148, 1154, 1000, 723])
    return (float(tr_x), float(tr_y)), (float(el_x), float(el_y)), (float(gr_x), float(gr_y))


def enhance_i2v_plate(frame_bgr: np.ndarray, t: float, is_wrong: bool) -> np.ndarray:
    """Subtle anatomical luminescence enhancement on the 3D I2V plate."""
    out = frame_bgr.astype(np.float32)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.2 * t)

    red_mask = cv2.inRange(hsv, (0, 85, 85), (25, 255, 255)) | cv2.inRange(hsv, (165, 85, 85), (180, 255, 255))
    red_float = cv2.GaussianBlur(red_mask.astype(np.float32) / 255.0, (43, 43), 0)[:, :, None]

    if is_wrong:
        out = np.clip(out + red_float * np.array([15.0, 55.0, 235.0], dtype=np.float32) * (0.16 + 0.34 * pulse), 0, 255)
    else:
        out = np.clip(out + red_float * np.array([20.0, 95.0, 240.0], dtype=np.float32) * (0.18 + 0.35 * pulse), 0, 255)
        grn_mask = cv2.inRange(hsv, (35, 75, 75), (95, 255, 255))
        grn_float = cv2.GaussianBlur(grn_mask.astype(np.float32) / 255.0, (37, 37), 0)[:, :, None]
        out = np.clip(out + grn_float * np.array([85.0, 255.0, 35.0], dtype=np.float32) * (0.18 + 0.32 * pulse), 0, 255)

    return out.astype(np.uint8)


# =============================================================================
# STUDIO-GRADE 4x SUPERSAMPLED 3D VOLUMETRIC TAPERED ARROWS
# =============================================================================
def _build_tapered_arrow_polygons(
    pts: np.ndarray,
    tail_w: float,
    neck_w: float,
    head_w: float,
    head_len: float,
    inset_ratio: float = 1.0,
    shift_normal: float = 0.0,
):
    """
    Given a dense 2D centerline `pts` (N x 2), constructs a sculpted 3D tapered ribbon shaft
    and swept-back chevron arrowhead polygon.
    `inset_ratio` (< 1.0) creates inner 3D bevel/core layers; `shift_normal` shifts toward the upper rim for 3D specular highlight.
    """
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    s_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(s_cum[-1])
    if total_len < 8.0:
        return None

    actual_head_len = min(head_len, total_len * 0.48)
    shaft_len = max(4.0, total_len - actual_head_len)

    # Compute smooth unit tangents & normals along the curve
    tangents = np.zeros_like(pts)
    tangents[:-1] = diffs
    tangents[-1] = diffs[-1]
    tangents[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    t_norm = np.hypot(tangents[:, 0], tangents[:, 1]) + 1e-6
    tx = tangents[:, 0] / t_norm
    ty = tangents[:, 1] / t_norm
    nx = -ty
    ny = tx

    # Sample shaft up to shaft_len
    shaft_mask = s_cum <= shaft_len
    if np.sum(shaft_mask) < 2:
        shaft_mask[:2] = True
    s_shaft = s_cum[shaft_mask]
    u_shaft = np.clip(s_shaft / max(1e-4, shaft_len), 0.0, 1.0)

    # Smooth non-linear taper (sleek tail -> muscular neck)
    half_w = 0.5 * (tail_w + (neck_w - tail_w) * (u_shaft ** 0.85)) * inset_ratio

    sx = pts[shaft_mask, 0] + nx[shaft_mask] * shift_normal
    sy = pts[shaft_mask, 1] + ny[shaft_mask] * shift_normal
    snx = nx[shaft_mask]
    sny = ny[shaft_mask]

    left_pts = np.column_stack([sx + snx * half_w, sy + sny * half_w])
    right_pts = np.column_stack([sx - snx * half_w, sy - sny * half_w])

    # Neck and tip geometry from the final segment direction
    neck_idx = int(np.where(shaft_mask)[0][-1])
    neck_pt = pts[neck_idx] + np.array([nx[neck_idx], ny[neck_idx]]) * shift_normal
    tip_t = np.array([tx[-1], ty[-1]])
    tip_n = np.array([nx[-1], ny[-1]])

    # Ensure tip_t points from neck_pt to pts[-1]
    vec_to_tip = pts[-1] - pts[neck_idx]
    v_len = float(np.hypot(vec_to_tip[0], vec_to_tip[1]))
    if v_len > 1e-3:
        tip_t = vec_to_tip / v_len
        tip_n = np.array([-tip_t[1], tip_t[0]])

    tip_pt = pts[-1] + tip_n * shift_normal - tip_t * ((1.0 - inset_ratio) * actual_head_len * 0.32)
    hw = 0.5 * head_w * inset_ratio
    barb_back = actual_head_len * 0.22 * inset_ratio

    barb_left = neck_pt + tip_n * hw - tip_t * barb_back
    barb_right = neck_pt - tip_n * hw - tip_t * barb_back

    poly = np.vstack([
        left_pts,
        [barb_left],
        [tip_pt],
        [barb_right],
        right_pts[::-1],
    ])
    return [tuple(map(float, p)) for p in poly]


def draw_3d_curved_arrow(
    canvas_rgba: Image.Image,
    p0: tuple,
    p1: tuple,
    p2: tuple,
    is_correct: bool,
    prog: float = 1.0,
    tail_w: float = 10.0,
    neck_w: float = 26.0,
    head_w: float = 58.0,
    head_len: float = 52.0,
):
    """
    Renders a studio-grade 4x-supersampled 3D volumetric curved (or straight) tapered arrow
    along the quadratic Bezier curve p0 -> p1 -> p2:
      1. Deep 3D drop shadow (+5, +7px offset, soft blur) for elevation above the mannequin
      2. Vibrant outer neon bloom (#28FF78 green or #FF3B2F red)
      3. Rich 3D bevel edge polygon
      4. Luminous inner body core polygon
      5. Glossy 3D specular highlight ribbon shifted slightly toward the upper lighting rim
    """
    if prog <= 0.05:
        return
    p = float(np.clip(prog, 0.0, 1.0))

    # Sample quadratic Bezier curve up to progress p
    n_samples = 64
    t_vals = np.linspace(0.0, p, n_samples)
    p0_arr, p1_arr, p2_arr = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64), np.array(p2, dtype=np.float64)
    pts = (
        ((1.0 - t_vals)[:, None] ** 2) * p0_arr
        + (2.0 * (1.0 - t_vals) * t_vals)[:, None] * p1_arr
        + (t_vals[:, None] ** 2) * p2_arr
    )

    # Compute local bounding box with generous padding for shadow + glow
    pad = int(max(head_w, head_len) + 42)
    xmin = max(0, int(np.floor(pts[:, 0].min())) - pad)
    ymin = max(0, int(np.floor(pts[:, 1].min())) - pad)
    xmax = min(W, int(np.ceil(pts[:, 0].max())) + pad)
    ymax = min(H, int(np.ceil(pts[:, 1].max())) + pad)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 8 or bh <= 8:
        return

    # Local coordinates at 4x supersampling for razor-sharp anti-aliased 3D edges
    SS = 4
    local_pts = (pts - np.array([xmin, ymin])) * SS

    poly_outer = _build_tapered_arrow_polygons(
        local_pts, tail_w * SS, neck_w * SS, head_w * SS, head_len * SS, inset_ratio=1.0, shift_normal=0.0
    )
    poly_mid = _build_tapered_arrow_polygons(
        local_pts, tail_w * SS, neck_w * SS, head_w * SS, head_len * SS, inset_ratio=0.72, shift_normal=0.0
    )
    poly_spec = _build_tapered_arrow_polygons(
        local_pts, tail_w * SS, neck_w * SS, head_w * SS, head_len * SS, inset_ratio=0.34, shift_normal=1.8 * SS
    )
    if not poly_outer:
        return

    if is_correct:
        glow_col = (30, 255, 115)
        edge_col = (16, 185, 78)
        mid_col = (52, 255, 132)
        spec_col = (230, 255, 240)
    else:
        glow_col = (255, 52, 38)
        edge_col = (195, 28, 22)
        mid_col = (255, 78, 52)
        spec_col = (255, 232, 225)

    # 1) 3D Drop Shadow (rendered at 1x patch resolution for fast Gaussian blur)
    pts_1x = pts - np.array([xmin, ymin])
    poly_shadow_1x = _build_tapered_arrow_polygons(
        pts_1x + np.array([5.0, 8.0]), tail_w, neck_w, head_w, head_len, inset_ratio=1.04
    )
    patch = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    if poly_shadow_1x:
        sh_im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        ImageDraw.Draw(sh_im).polygon(poly_shadow_1x, fill=(4, 6, 12, 185))
        sh_im = sh_im.filter(ImageFilter.GaussianBlur(radius=9))
        patch.alpha_composite(sh_im)

    # 2) Outer Neon Bloom (1x patch resolution)
    poly_glow_1x = _build_tapered_arrow_polygons(pts_1x, tail_w, neck_w, head_w, head_len, inset_ratio=1.10)
    if poly_glow_1x:
        gl_im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        ImageDraw.Draw(gl_im).polygon(poly_glow_1x, fill=(*glow_col, 150))
        gl_im = gl_im.filter(ImageFilter.GaussianBlur(radius=13))
        patch.alpha_composite(gl_im)

    # 3) 4x Supersampled 3D Sculpted Arrow Body (Outer Bevel -> Luminous Core -> 3D Specular Spine)
    hi_im = Image.new("RGBA", (bw * SS, bh * SS), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hi_im)
    hd.polygon(poly_outer, fill=(*edge_col, 252))
    if poly_mid:
        hd.polygon(poly_mid, fill=(*mid_col, 255))
    if poly_spec:
        hd.polygon(poly_spec, fill=(*spec_col, 245))

    # Crisp 3D rim outline on the outer bevel
    hd.line(poly_outer + [poly_outer[0]], fill=(*spec_col, 155), width=max(2, SS // 2))

    body_1x = hi_im.resize((bw, bh), resample=Image.Resampling.LANCZOS)
    patch.alpha_composite(body_1x)

    canvas_rgba.alpha_composite(patch, dest=(xmin, ymin))


def draw_3d_volumetric_arrow(
    canvas_rgba: Image.Image,
    p0: tuple,
    p2: tuple,
    is_correct: bool,
    prog: float = 1.0,
    tail_w: float = 10.0,
    neck_w: float = 25.0,
    head_w: float = 56.0,
    head_len: float = 50.0,
):
    """Straight 3D volumetric tapered arrow (special case of quadratic Bezier with midpoint control)."""
    p1 = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]))
    draw_3d_curved_arrow(
        canvas_rgba, p0, p1, p2, is_correct=is_correct, prog=prog,
        tail_w=tail_w, neck_w=neck_w, head_w=head_w, head_len=head_len,
    )


# =============================================================================
# @DEMICSTORY BASIC UI + RADICALLY SIMPLIFIED BIOMECHANICAL OVERLAYS
# =============================================================================
def draw_demic_status_badge(canvas_rgba: Image.Image, is_correct: bool, pulse: float):
    """@demicstory top-right circular translucent ring with glowing red ✕ or green ✓."""
    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    cx, cy, r = 955, 195, 48

    halo = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    if not is_correct:
        h_col = (255, 45, 35, int(105 + 50 * pulse))
        i_col = (255, 55, 45, 255)
    else:
        h_col = (35, 255, 115, int(105 + 50 * pulse))
        i_col = (45, 255, 120, 255)

    hd.ellipse([cx - r - 16, cy - r - 16, cx + r + 16, cy + r + 16], fill=h_col)
    halo = halo.filter(ImageFilter.GaussianBlur(radius=16))
    canvas_rgba.alpha_composite(halo)

    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 22, 28, 180), outline=(235, 242, 255, 195), width=3)

    sym_glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    sgd = ImageDraw.Draw(sym_glow)
    if not is_correct:
        s = 20
        sgd.line([(cx - s, cy - s), (cx + s, cy + s)], fill=i_col, width=9)
        sgd.line([(cx - s, cy + s), (cx + s, cy - s)], fill=i_col, width=9)
        d.line([(cx - s, cy - s), (cx + s, cy + s)], fill=(255, 210, 200, 255), width=4)
        d.line([(cx - s, cy + s), (cx + s, cy - s)], fill=(255, 210, 200, 255), width=4)
    else:
        pts = [(cx - 22, cy + 2), (cx - 5, cy + 19), (cx + 24, cy - 16)]
        sgd.line(pts, fill=i_col, width=10, joint="curve")
        d.line(pts, fill=(220, 255, 232, 255), width=4, joint="curve")

    sym_glow = sym_glow.filter(ImageFilter.GaussianBlur(radius=6))
    canvas_rgba.alpha_composite(sym_glow)
    canvas_rgba.alpha_composite(overlay)


def draw_demic_minimap_pill(canvas_rgba: Image.Image, is_correct: bool):
    """@demicstory top-center 3-thumbnail translucent pill (zero text)."""
    pill_w, pill_h = 348, 116
    px0 = (W - pill_w) // 2
    py0 = 96

    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle(
        [px0, py0, px0 + pill_w, py0 + pill_h],
        radius=24,
        fill=(18, 22, 30, 190),
        outline=(255, 255, 255, 60),
        width=2,
    )
    canvas_rgba.alpha_composite(overlay)

    pad_x, gap = 16, 14
    ty = py0 + (pill_h - 96) // 2
    active_idx = 0 if is_correct else 1
    for i, th in enumerate(THUMBS):
        tx = px0 + pad_x + i * (96 + gap)
        mask = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, 95, 95], radius=14, fill=255)
        canvas_rgba.paste(th, (tx, ty), mask)

        b_ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(b_ov)
        if i == active_idx:
            col = (45, 255, 120, 250) if is_correct else (255, 95, 30, 250)
            bd.rounded_rectangle([tx - 2, ty - 2, tx + 98, ty + 98], radius=15, outline=col, width=4)
        else:
            bd.rounded_rectangle([tx, ty, tx + 96, ty + 96], radius=14, outline=(255, 255, 255, 45), width=1)
        canvas_rgba.alpha_composite(b_ov)


def draw_simplified_gym_differentiator(
    canvas_rgba: Image.Image,
    is_correct: bool,
    t: float,
    norm_u: float,
    pulse: float,
    prog: float,
):
    """
    Radically Simplified, Studio-Grade Biomechanical Visuals ("One Glance, One Point", Zero Text):
      - WRONG (✕, 1.7-4.65s):
          1. Clean crimson-orange overload aura + a single crisp red warning ring ONLY on the flared elbow & strained wrist
             (NO cluttered polylines or shoulder rings).
          2. 1 Bold 3D Curved Tapered Red Arrow at the elbow showing the elbow swinging/flaring out,
             plus 1 compact 3D Curved Red Arrow at the wrist showing the wrist bending backward under load.
          3. Sleek, compact 5-Bar Muscle Activation Meter at bottom-right stalled at 1/5 Red.
      - CORRECT (✓, 4.95-8.5s):
          1. Pinned Elbow Lock-On Ring: 1 ultra-clean glowing neon-green 3D ring with 4 minimal precision ticks
             pinned at the elbow hinge (leaving the glowing 3D triceps muscle completely unobstructed!).
          2. 1 Bold 3D Curved Tapered Green Arrow sweeping cleanly along the forearm's downward extension arc,
             plus at bottom extension (norm_u in 0.28..0.82) 2 symmetrical 3D Tapered Outward Rope-Spread Arrows
             at the bottom of the rope showing the hands pulling the rope apart!
          3. Sleek, compact 5-Bar Muscle Activation Meter at bottom-right charging smoothly to 5/5 Neon Green.
    """
    if prog <= 0.02:
        return
    a = float(min(1.0, prog))

    if not is_correct:
        elbow_pt, wrist_pt = get_wrong_landmarks(norm_u)
        ex, ey = elbow_pt
        wx, wy = wrist_pt

        # 1) Clean Red Overload Aura + Single Crisp Ring on Elbow & Wrist ONLY
        aura = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
        ad = ImageDraw.Draw(aura)
        ring_layer = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring_layer)

        for (jx, jy, r_base) in [(ex, ey, 40), (wx, wy, 32)]:
            r_pulse = int(r_base + 8 * pulse)
            # Soft glowing crimson-orange overload halo
            ad.ellipse(
                [jx - r_pulse - 18, jy - r_pulse - 18, jx + r_pulse + 18, jy + r_pulse + 18],
                fill=(255, 42, 28, int(95 * a)),
            )
            ad.ellipse(
                [jx - r_pulse, jy - r_pulse, jx + r_pulse, jy + r_pulse],
                outline=(255, 55, 38, int(220 * a)),
                width=8,
            )
            # Single crisp inner precision ring
            rd.ellipse(
                [jx - r_pulse, jy - r_pulse, jx + r_pulse, jy + r_pulse],
                outline=(255, 215, 205, int(235 * a)),
                width=3,
            )

        aura = aura.filter(ImageFilter.GaussianBlur(radius=12))
        canvas_rgba.alpha_composite(aura)
        canvas_rgba.alpha_composite(ring_layer)

        # 2) Bold 3D Curved Tapered Red Arrow at Elbow (showing elbow flaring/swinging out & upward)
        # Starts near the elbow joint and curves outward/upward to the right where the elbow flares
        e_p0 = (ex + 22.0, ey + 18.0)
        e_p1 = (ex + 105.0, ey + 4.0)
        e_p2 = (ex + 168.0, ey - 78.0)
        draw_3d_curved_arrow(
            canvas_rgba, e_p0, e_p1, e_p2,
            is_correct=False, prog=a,
            tail_w=11.0, neck_w=27.0, head_w=60.0, head_len=52.0,
        )

        # Compact 3D Curved Red Arrow at Wrist (showing wrist snapping/bending backward under load)
        w_p0 = (wx - 18.0, wy + 12.0)
        w_p1 = (wx - 88.0, wy + 25.0)
        w_p2 = (wx - 135.0, wy - 48.0)
        draw_3d_curved_arrow(
            canvas_rgba, w_p0, w_p1, w_p2,
            is_correct=False, prog=a,
            tail_w=9.0, neck_w=22.0, head_w=48.0, head_len=44.0,
        )

    else:
        _, elbow_pivot, hands_grip = get_correct_landmarks(norm_u)
        ex, ey = elbow_pivot
        gx, gy = hands_grip

        # 1) Single Ultra-Clean Glowing Neon-Green 3D Lock-On Ring at the Pinned Elbow Hinge
        glow_ring = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
        grd = ImageDraw.Draw(glow_ring)
        core_ring = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
        crd = ImageDraw.Draw(core_ring)

        lr = 44
        # Subtle lock-on flash right after transition (4.95 - 5.35s)
        flash_boost = max(0.0, 1.0 - max(0.0, t - 4.95) / 0.40)
        halo_alpha = int(min(255, (110 + 90 * flash_boost) * a))
        grd.ellipse(
            [ex - lr - 14, ey - lr - 14, ex + lr + 14, ey + lr + 14],
            fill=(30, 255, 115, int(55 * a + 65 * flash_boost)),
        )
        grd.ellipse(
            [ex - lr, ey - lr, ex + lr, ey + lr],
            outline=(35, 255, 120, halo_alpha),
            width=11,
        )
        crd.ellipse(
            [ex - lr, ey - lr, ex + lr, ey + lr],
            outline=(230, 255, 240, int(245 * a)),
            width=4,
        )
        # 4 minimal precision lock ticks on the ring
        for deg in [0, 90, 180, 270]:
            rad = math.radians(deg)
            x_in, y_in = ex + (lr - 10) * math.cos(rad), ey + (lr - 10) * math.sin(rad)
            x_out, y_out = ex + (lr + 10) * math.cos(rad), ey + (lr + 10) * math.sin(rad)
            crd.line([(x_in, y_in), (x_out, y_out)], fill=(235, 255, 245, int(240 * a)), width=4)
        # Center pivot dot
        crd.ellipse([ex - 7, ey - 7, ex + 7, ey + 7], fill=(45, 255, 125, int(255 * a)))
        crd.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(245, 255, 250, int(255 * a)))

        glow_ring = glow_ring.filter(ImageFilter.GaussianBlur(radius=10))
        canvas_rgba.alpha_composite(glow_ring)
        canvas_rgba.alpha_composite(core_ring)

        # 2) 1 Bold 3D Curved Tapered Green Arrow sweeping cleanly along the forearm's downward extension arc
        # Positioned cleanly just outside the forearm sweep so it never covers the triceps or hands!
        arc_p0 = (395.0, 745.0)
        arc_p1 = (355.0, 1025.0)
        arc_p2 = (575.0, 1195.0)
        arc_prog = min(1.0, max(0.25, (t - 4.95) / 0.65)) * a
        draw_3d_curved_arrow(
            canvas_rgba, arc_p0, arc_p1, arc_p2,
            is_correct=True, prog=arc_prog,
            tail_w=11.0, neck_w=28.0, head_w=62.0, head_len=54.0,
        )

        # 3) At bottom extension (when hands push the rope down), 2 symmetrical 3D Tapered Outward Rope-Spread Arrows
        # showing the hands pulling the black rope apart left & right at the bottom!
        if 0.26 <= norm_u <= 0.84:
            spread_prog = float(np.clip(min((norm_u - 0.26) / 0.12, (0.84 - norm_u) / 0.12), 0.0, 1.0)) * a
            # Left outward rope-spread 3D arrow
            draw_3d_curved_arrow(
                canvas_rgba,
                (gx - 28.0, gy + 18.0),
                (gx - 92.0, gy + 55.0),
                (gx - 158.0, gy + 108.0),
                is_correct=True,
                prog=spread_prog,
                tail_w=9.0, neck_w=22.0, head_w=48.0, head_len=44.0,
            )
            # Right outward rope-spread 3D arrow
            draw_3d_curved_arrow(
                canvas_rgba,
                (gx + 28.0, gy + 18.0),
                (gx + 92.0, gy + 55.0),
                (gx + 158.0, gy + 108.0),
                is_correct=True,
                prog=spread_prog,
                tail_w=9.0, neck_w=22.0, head_w=48.0, head_len=44.0,
            )

    # -------------------------------------------------------------------------
    # PART 2: Sleek, Compact 5-Bar Muscle Activation Meter (Bottom-Right, Zero Text)
    # -------------------------------------------------------------------------
    meter_ov = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    md = ImageDraw.Draw(meter_ov)
    meter_glow = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    mgd = ImageDraw.Draw(meter_glow)

    mw, mh = 58, 236
    mx0 = 964 - mw // 2
    my0 = 1365

    md.rounded_rectangle(
        [mx0, my0, mx0 + mw, my0 + mh],
        radius=20,
        fill=(16, 20, 28, int(190 * a)),
        outline=(255, 255, 255, int(65 * a)),
        width=2,
    )

    n_bars = 5
    bar_w = mw - 20
    bar_h = 22
    gap_y = 10
    start_y = my0 + mh - 18 - bar_h

    if not is_correct:
        active_bars = 1
        b_col = (255, 52, 38)
    else:
        charge_t = max(0.0, t - 5.0)
        active_bars = min(5, int(1 + charge_t * 4.2))
        b_col = (42, 255, 120)

    for b in range(n_bars):
        bx = mx0 + 10
        by = start_y - b * (bar_h + gap_y)
        if b < active_bars:
            mgd.rounded_rectangle([bx - 3, by - 3, bx + bar_w + 3, by + bar_h + 3], radius=6, fill=(*b_col, int(165 * a)))
            md.rounded_rectangle([bx, by, bx + bar_w, by + bar_h], radius=6, fill=(*b_col, int(250 * a)))
            md.rounded_rectangle([bx + 3, by + 3, bx + bar_w - 3, by + 8], radius=3, fill=(235, 255, 242, int(190 * a)))
        else:
            md.rounded_rectangle([bx, by, bx + bar_w, by + bar_h], radius=6, fill=(42, 48, 60, int(135 * a)))

    lx, ly = mx0 + mw // 2, my0 + 24
    bolt_pts = [(lx + 3, ly - 12), (lx - 8, ly + 1), (lx + 1, ly + 1), (lx - 3, ly + 13), (lx + 8, ly - 1), (lx - 1, ly - 1)]
    bolt_col = (42, 255, 120, int(255 * a)) if is_correct else (255, 72, 52, int(230 * a))
    mgd.polygon(bolt_pts, fill=bolt_col)
    md.polygon(bolt_pts, fill=(255, 255, 255, int(250 * a)))

    meter_glow = meter_glow.filter(ImageFilter.GaussianBlur(radius=7))
    canvas_rgba.alpha_composite(meter_glow)
    canvas_rgba.alpha_composite(meter_ov)


def render_one_frame(f: int):
    t = f / FPS
    pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * 2.2 * t)

    # Timeline mapping:
    # 0.00 - 1.50s: Smooth macro hook pulling back from the glowing joint/muscle to full 9:16 view
    # 1.50 - 4.65s: Wrong Form (✕) full 3D I2V repetition + simplified 3D red volumetric arrows
    # 4.65 - 4.95s: Crisp 0.30s cross-dissolve transition (✕ -> ✓)
    # 4.95 - 8.50s: Correct Form (✓) full 3D I2V repetition + Pinned Elbow Ring + 3D Curved Tapered Green Arrow + Rope Spread Arrows
    if t < 1.1:
        u_wrong = 0.03 * (t / 1.1)
    else:
        u_wrong = min(1.0, 0.03 + 0.97 * ((t - 1.1) / 3.65))

    u_corr = float(np.clip((t - 4.75) / 3.75, 0.0, 1.0))

    if t < 4.65:
        raw_w = sample_clip_frame(WRONG_FRAMES, u_wrong)
        plate = enhance_i2v_plate(raw_w, t, is_wrong=True)
        is_corr = False
    elif t < 4.95:
        alpha = (t - 4.65) / 0.30
        raw_w = sample_clip_frame(WRONG_FRAMES, u_wrong)
        raw_c = sample_clip_frame(CORRECT_FRAMES, u_corr)
        pw = enhance_i2v_plate(raw_w, t, is_wrong=True).astype(np.float32)
        pc = enhance_i2v_plate(raw_c, t, is_wrong=False).astype(np.float32)
        plate = (pw * (1.0 - alpha) + pc * alpha).astype(np.uint8)
        is_corr = (alpha >= 0.5)
    else:
        raw_c = sample_clip_frame(CORRECT_FRAMES, u_corr)
        plate = enhance_i2v_plate(raw_c, t, is_wrong=False)
        is_corr = True

    # 0.0s - 1.5s: Smooth @demicstory macro focus zoom-out hook
    if t < 1.5:
        u = t / 1.5
        ease = 1.0 - (1.0 - u) ** 3
        zoom = 1.72 - 0.72 * ease
        fx = 680.0 * (1.0 - ease) + 540.0 * ease
        fy = 890.0 * (1.0 - ease) + 960.0 * ease
        zw, zh = int(W / zoom), int(H / zoom)
        cx0 = int(np.clip(fx - zw / 2, 0, W - zw))
        cy0 = int(np.clip(fy - zh / 2, 0, H - zh))
        plate = cv2.resize(plate[cy0:cy0 + zh, cx0:cx0 + zw], (W, H), interpolation=cv2.INTER_LINEAR)

    frame_rgba = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)).convert("RGBA")

    if 1.65 <= t < 4.65:
        prog = min(1.0, (t - 1.65) / 0.38, (4.65 - t) / 0.22)
        draw_simplified_gym_differentiator(frame_rgba, is_correct=False, t=t, norm_u=u_wrong, pulse=pulse, prog=prog)
    elif t >= 4.95:
        prog = min(1.0, (t - 4.95) / 0.38)
        draw_simplified_gym_differentiator(frame_rgba, is_correct=True, t=t, norm_u=u_corr, pulse=pulse, prog=prog)

    if t >= 1.0:
        draw_demic_minimap_pill(frame_rgba, is_correct=is_corr)
        draw_demic_status_badge(frame_rgba, is_correct=is_corr, pulse=pulse)

    rgb_out = frame_rgba.convert("RGB")
    rgb_out.save(f"{TMP_DIR}/f_{f:04d}.jpg", quality=95)


def synthesize_gym_audio(wav_path: str):
    """
    Synthesizes a stereo 44.1kHz tactile gym biomechanics soundtrack:
      1. Deep studio sub-bass pulse (48-58Hz rhythmic heartbeat)
      2. Realistic metallic gym cable pulley glide & steel weight-stack clinks during repetitions
      3. Low warning tension buzz + joint-strain crackle during Wrong Form (1.7 - 4.6s)
      4. Crisp mechanical Lock-On Servo Snap + Heavy Sub-Bass Impact Drop at 4.80s
      5. Rising 5-step harmonic battery charge chime (5.0 - 6.2s) + peak contraction shimmer
    """
    sr = 44100
    n = int(sr * DURATION)
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(42)

    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    sub_env = 0.65 + 0.35 * np.sin(2 * np.pi * 1.25 * t)
    sub = sub_env * (0.045 * np.sin(2 * np.pi * 52.0 * t) + 0.022 * np.sin(2 * np.pi * 104.0 * t))
    left += sub
    right += sub

    mw = (t >= 0.0) & (t < 1.5)
    tw = t[mw]
    w_env = np.sin(np.pi * tw / 1.5) ** 1.35
    w_sig = 0.16 * w_env * np.sin(2 * np.pi * (210.0 - 105.0 * tw / 1.5) * tw)
    left[mw] += w_sig * 1.08
    right[mw] += w_sig * 0.92

    white = rng.standard_normal(n).astype(np.float32)
    kernel_sz = 18
    pulley_tex = np.convolve(white, np.ones(kernel_sz, dtype=np.float32) / kernel_sz, mode="same")

    for (g_start, g_end, amp) in [(1.55, 4.35, 0.032), (5.0, 8.1, 0.038)]:
        mg = (t >= g_start) & (t < g_end)
        tg = t[mg] - g_start
        dur = g_end - g_start
        glide_env = np.sin(np.pi * tg / dur) ** 1.2 * (0.65 + 0.35 * np.sin(2 * np.pi * 2.0 * tg / dur) ** 2)
        roller_mod = 0.7 + 0.3 * np.sin(2 * np.pi * 18.0 * tg)
        left[mg] += amp * glide_env * roller_mod * pulley_tex[mg]
        right[mg] += amp * glide_env * roller_mod * np.roll(pulley_tex[mg], 12)

    clink_times = [
        (1.60, 0.11),
        (2.95, 0.14),
        (4.25, 0.10),
        (5.05, 0.12),
        (6.65, 0.09),
        (8.05, 0.11),
    ]
    for c_t, c_amp in clink_times:
        mc = (t >= c_t) & (t < c_t + 0.22)
        tc = t[mc] - c_t
        c_env = np.exp(-tc * 34.0) * (1.0 - np.exp(-tc * 400.0))
        clink = c_amp * c_env * (
            0.50 * np.sin(2 * np.pi * 1180.0 * tc)
            + 0.35 * np.sin(2 * np.pi * 2460.0 * tc)
            + 0.25 * np.sin(2 * np.pi * 3890.0 * tc)
            + 0.18 * white[mc] * np.exp(-tc * 75.0)
        )
        left[mc] += clink * 0.95
        right[mc] += clink * 1.05

    for pt in [1.80, 2.75, 3.70]:
        mp = (t >= pt) & (t < pt + 0.42)
        tp = t[mp] - pt
        p_env = np.exp(-tp * 8.5) * (1.0 - np.exp(-tp * 90.0))
        buzz = 0.19 * p_env * (
            np.sin(2 * np.pi * 86.0 * tp)
            + 0.55 * np.sin(2 * np.pi * 172.0 * tp)
            + 0.25 * np.sin(2 * np.pi * 258.0 * tp)
        )
        crackle = 0.045 * np.exp(-tp * 18.0) * pulley_tex[mp]
        left[mp] += buzz + crackle
        right[mp] += buzz * 0.92 - crackle

    ml = (t >= 4.80) & (t < 5.40)
    tl = t[ml] - 4.80
    snap1 = np.exp(-tl * 65.0) * (np.sin(2 * np.pi * 1650.0 * tl) + 0.7 * np.sin(2 * np.pi * 3100.0 * tl))
    tl2 = np.maximum(0.0, tl - 0.045)
    snap2 = (tl >= 0.045) * np.exp(-tl2 * 55.0) * (np.sin(2 * np.pi * 2250.0 * tl2) + 0.8 * np.sin(2 * np.pi * 4100.0 * tl2))
    drop_env = np.exp(-tl * 5.2) * (1.0 - np.exp(-tl * 120.0))
    bass_drop = 0.28 * drop_env * np.sin(2 * np.pi * (46.0 + 64.0 * np.exp(-tl * 12.0)) * tl)
    lock_sig = 0.18 * (snap1 + snap2) + bass_drop
    left[ml] += lock_sig
    right[ml] += lock_sig

    charge_notes = [
        (5.00, 523.25, 0.12),
        (5.24, 587.33, 0.13),
        (5.48, 659.25, 0.14),
        (5.72, 783.99, 0.15),
        (5.96, 1046.50, 0.20),
    ]
    for n_start, freq, n_amp in charge_notes:
        dur = 0.85 if freq > 900 else 0.45
        mn = (t >= n_start) & (t < min(DURATION, n_start + dur))
        tn = t[mn] - n_start
        n_env = np.exp(-tn * (3.2 if freq > 900 else 6.5)) * (1.0 - np.exp(-tn * 80.0))
        tone = n_amp * n_env * (
            np.sin(2 * np.pi * freq * tn)
            + 0.45 * np.sin(2 * np.pi * (freq * 1.5) * tn)
            + 0.25 * np.sin(2 * np.pi * (freq * 2.0) * tn)
        )
        pan = 0.35 + 0.30 * ((freq - 523.0) / 523.0)
        left[mn] += tone * (1.0 - 0.3 * pan)
        right[mn] += tone * (0.7 + 0.3 * pan)

    ms = (t >= 6.25) & (t < 7.85)
    ts = t[ms] - 6.25
    s_env = np.sin(np.pi * ts / 1.6) ** 1.3 * 0.085
    chord = s_env * (
        np.sin(2 * np.pi * 523.25 * ts)
        + 0.8 * np.sin(2 * np.pi * 659.25 * ts)
        + 0.7 * np.sin(2 * np.pi * 783.99 * ts)
    )
    left[ms] += chord * 1.05
    right[ms] += chord * 0.95

    fade = np.ones(n, dtype=np.float32)
    f_in = int(0.08 * sr)
    f_out = int(0.18 * sr)
    fade[:f_in] = np.linspace(0.0, 1.0, f_in)
    fade[-f_out:] = np.linspace(1.0, 0.0, f_out)

    left = np.tanh(left * fade * 1.35) * 0.92
    right = np.tanh(right * fade * 1.35) * 0.92

    stereo_pcm = np.empty((n, 2), dtype=np.int16)
    stereo_pcm[:, 0] = (left * 32767.0).astype(np.int16)
    stereo_pcm[:, 1] = (right * 32767.0).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo_pcm.tobytes())


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    with ProcessPoolExecutor() as ex:
        list(ex.map(render_one_frame, range(TOTAL_FRAMES)))

    wav_path = f"{TMP_DIR}/audio.wav"
    synthesize_gym_audio(wav_path)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", f"{TMP_DIR}/f_%04d.jpg",
        "-i", wav_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        OUT_MP4,
    ]
    subprocess.run(cmd, check=True)

    # Extract 3-step keyframe sheet:
    # 1) Hook macro X-ray zoom (0.35s)
    # 2) Wrong form mid-pushdown with clean red Elbow & Wrist rings + 3D Curved Tapered Red Arrows (2.8s)
    # 3) Correct form full triceps extension with Elbow Lock Ring + 3D Curved Tapered Green Arrow + 3D Rope-Spread Arrows + 5/5 Battery (6.65s)
    kf_indices = [int(0.35 * FPS), int(2.8 * FPS), int(6.65 * FPS)]
    kfs = [Image.open(f"{TMP_DIR}/f_{k:04d}.jpg").copy() for k in kf_indices]

    strip = Image.new("RGB", (W * 3, H))
    for i, im in enumerate(kfs):
        strip.paste(im, (i * W, 0))
    strip.resize((1620, 960), Image.Resampling.LANCZOS).save(OUT_KEYFRAMES, quality=93)

    if os.path.exists("/workspace/daily_info_reels/demic_bench/frames/De1ZjBfDzns_t1s.png"):
        orig_w = Image.open("/workspace/daily_info_reels/demic_bench/frames/De1ZjBfDzns_t1s.png").convert("RGB").resize((540, 960), Image.Resampling.LANCZOS)
        orig_c = Image.open("/workspace/daily_info_reels/demic_bench/frames/De1ZjBfDzns_t4s.png").convert("RGB").resize((540, 960), Image.Resampling.LANCZOS)
        our_w = kfs[1].resize((540, 960), Image.Resampling.LANCZOS)
        our_c = kfs[2].resize((540, 960), Image.Resampling.LANCZOS)

        comp = Image.new("RGB", (540 * 4, 960))
        comp.paste(orig_w, (0, 0))
        comp.paste(our_w, (540, 0))
        comp.paste(orig_c, (1080, 0))
        comp.paste(our_c, (1620, 0))
        comp.save(OUT_COMP, quality=92)

    # Clean up temporary frame directory so disk never fills up
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("GYM DAY1 V3 DONE (TMP CLEANED):", OUT_MP4, OUT_KEYFRAMES, OUT_COMP)


if __name__ == "__main__":
    main()
