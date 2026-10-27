#!/usr/bin/env python3
"""
v42 @demicstory Authentic Posture-Correction & Biomechanical Vector Master (Day 3: Barbell Back Squat)
- Directly implements the visual grammar extracted from all 8 @demicstory reels (De4oJaICVDI, De77WaXAiXd, De1ZjBfDzns, DePzKDbFB0y, DetDsIDj2BJ):
  1. Same-Angle Red (Wrong Posture) -> Green (Correct Posture) physical posture & color transitions:
     - Pair 1 (Front View, Steps 1 -> 2): Severe Knee Valgus (>< Red) physically pushes outward into Proper Knee-Over-Toe Tracking (| | Green & Orange Quads).
     - Pair 2 (Side Profile View, Steps 3 -> 4): Collapsed Forward Torso & Lifted Heels (Red) physically lifts Chest Upright (↑), drops Hips Deep (↓), and plants Heels/Front Feet Flat (↓) in Green & Orange Quads/Glutes.
  2. Explicit Graphical Expression of Movement Direction ("위로 올렸는지, 아래로 내렸는지, 앞발은 어떻게 하는지"):
     - Skeletal Axis Lines (Red misaligned line morphing into Green aligned line).
     - Directional Action Arrows directly on the body (──▶ ◀── KNEES IN -> ◀══ ══▶ KNEES OUT, ⤵ TORSO DROP -> ↑ CHEST UP, ↓ HIPS DOWN, ↑ HEELS UP -> ↓ HEELS FLAT).
     - Floor Stance & Foot Guides directly under the front feet (Floor foot-angle tracks & flat-foot laser line).
  3. 100% Box-Zero (no semi-transparent boxes, no pill capsules) & Zero arbitrary degree numbers.
  4. Passes all 5 QC gates in qc_gate.py (including adjacent step pixel diff >= 35.0 and v39 Safe Zone).
"""
import os
import math
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1920
FPS = 30
TOTAL_SEC = 20.0
TOTAL_FRAMES = int(TOTAL_SEC * FPS)  # 600 frames

OUT_DIR = "/workspace/daily_info_reels/output"
os.makedirs(OUT_DIR, exist_ok=True)

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
def get_font(size: int):
    return ImageFont.truetype(FONT_BOLD, size)

F_TITLE     = get_font(40)
F_STEP      = get_font(23)
F_VEC_MAIN  = get_font(30)
F_VEC_SUB   = get_font(22)
F_HUD_TAG   = get_font(25)
F_HUD_PCT   = get_font(76)
F_HUD_CUE   = get_font(42)

COL_RED     = (255, 48, 64)
COL_GREEN   = (0, 245, 138)
COL_ORANGE  = (255, 165, 30)
COL_WHITE   = (250, 252, 255)
COL_SILVER  = (200, 210, 222)

# Precompute biomechanical knee-valgus displacement maps for Front View (1080x1920)
YY, XX = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
VY_KNEE = np.exp(-0.5 * ((YY - 1195.0) / 135.0) ** 2)
VY_KNEE = np.clip((VY_KNEE - 0.05) / 0.95, 0.0, 1.0)
VX_LEFT = np.exp(-0.5 * ((XX - 385.0) / 75.0) ** 2) * (XX > 295).astype(np.float32) * (XX < 535).astype(np.float32)
VX_RIGHT = np.exp(-0.5 * ((XX - 695.0) / 75.0) ** 2) * (XX < 785).astype(np.float32) * (XX > 545).astype(np.float32)
VALGUS_FIELD = (-VY_KNEE * VX_LEFT + VY_KNEE * VX_RIGHT)

def warp_knee_valgus(frame_bgr: np.ndarray, shift_px: float) -> np.ndarray:
    if abs(shift_px) < 0.5:
        return frame_bgr
    map_x = XX + shift_px * VALGUS_FIELD
    return cv2.remap(frame_bgr, map_x, YY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def load_clip_frames(path: str) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(fr, (W, H), interpolation=cv2.INTER_CUBIC))
    cap.release()
    return frames

def sample_clip(frames: list, norm_t: float) -> np.ndarray:
    idx = min(len(frames) - 1, max(0, int(round(norm_t * (len(frames) - 1)))))
    return frames[idx].copy()

def apply_camera_crop(frame_bgr: np.ndarray, zoom: float, cx_norm: float = 0.5, cy_norm: float = 0.55) -> np.ndarray:
    if abs(zoom - 1.0) < 1e-3:
        return frame_bgr
    cw = int(round(W / zoom))
    ch = int(round(H / zoom))
    cx = int(round(W * cx_norm))
    cy = int(round(H * cy_norm))
    x0 = max(0, min(W - cw, cx - cw // 2))
    y0 = max(0, min(H - ch, cy - ch // 2))
    crop = frame_bgr[y0:y0+ch, x0:x0+cw]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def lerp_pt(p1, p2, t: float):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))

def lerp_col(c1, c2, t: float):
    return (
        int(round(lerp(c1[0], c2[0], t))),
        int(round(lerp(c1[1], c2[1], t))),
        int(round(lerp(c1[2], c2[2], t))),
    )

def draw_shadow_text(draw: ImageDraw.ImageDraw, glow_draw: ImageDraw.ImageDraw, pos, text: str, font, fill, shadow_alpha=220):
    x, y = int(round(pos[0])), int(round(pos[1]))
    glow_draw.text((x, y + 3), text, font=font, fill=(0, 0, 0, shadow_alpha))
    glow_draw.text((x - 2, y), text, font=font, fill=(0, 0, 0, int(shadow_alpha * 0.75)))
    glow_draw.text((x + 2, y), text, font=font, fill=(0, 0, 0, int(shadow_alpha * 0.75)))
    draw.text((x, y), text, font=font, fill=(fill[0], fill[1], fill[2], 255))

def draw_neon_line(draw: ImageDraw.ImageDraw, glow_draw: ImageDraw.ImageDraw, pts, color, width=7, dashed=False):
    if len(pts) < 2:
        return
    r, g, b = color
    if not dashed:
        glow_draw.line(pts, fill=(r, g, b, 110), width=width + 12, joint="curve")
        draw.line(pts, fill=(r, g, b, 245), width=width, joint="curve")
        draw.line(pts, fill=(255, 255, 255, 190), width=max(2, width // 3), joint="curve")
    else:
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i+1]
            dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            steps = max(2, int(dist // 22))
            for s in range(0, steps, 2):
                t0 = s / steps
                t1 = min(1.0, (s + 1.1) / steps)
                seg = [lerp_pt(p0, p1, t0), lerp_pt(p0, p1, t1)]
                glow_draw.line(seg, fill=(r, g, b, 95), width=width + 8)
                draw.line(seg, fill=(r, g, b, 230), width=width)

def draw_neon_joint_node(draw: ImageDraw.ImageDraw, glow_draw: ImageDraw.ImageDraw, pt, color, radius=9):
    x, y = pt
    r, g, b = color
    glow_draw.ellipse([x - radius - 7, y - radius - 7, x + radius + 7, y + radius + 7], fill=(r, g, b, 100))
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(r, g, b, 245), outline=(255, 255, 255, 230), width=2)

def draw_bold_vector_arrow(draw: ImageDraw.ImageDraw, glow_draw: ImageDraw.ImageDraw, p_start, p_end, color, width=9, head_len=24, head_w=16, progress=1.0):
    """Draws a bold @demicstory-style biomechanical action vector arrow (straight) with neon glow."""
    if progress <= 0.02:
        return
    p_cur = lerp_pt(p_start, p_end, min(1.0, progress))
    r, g, b = color
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    length = math.hypot(dx, dy)
    if length < 4:
        return
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    # Shorten shaft slightly so it meets arrowhead cleanly
    shaft_end = (p_cur[0] - ux * head_len * 0.65, p_cur[1] - uy * head_len * 0.65)
    glow_draw.line([p_start, shaft_end], fill=(r, g, b, 125), width=width + 12)
    draw.line([p_start, shaft_end], fill=(r, g, b, 255), width=width)
    draw.line([p_start, shaft_end], fill=(255, 255, 255, 190), width=max(2, width // 3))

    # Arrowhead triangle at p_cur
    base_cx = p_cur[0] - ux * head_len
    base_cy = p_cur[1] - uy * head_len
    left_pt = (base_cx + nx * head_w, base_cy + ny * head_w)
    right_pt = (base_cx - nx * head_w, base_cy - ny * head_w)
    poly = [p_cur, left_pt, right_pt]
    glow_draw.polygon(poly, fill=(r, g, b, 140))
    draw.polygon(poly, fill=(r, g, b, 255), outline=(255, 255, 255, 220))

def draw_curved_vector_arrow(draw: ImageDraw.ImageDraw, glow_draw: ImageDraw.ImageDraw, p0, p1, p2, color, width=9, head_len=24, head_w=16, progress=1.0):
    """Draws a quadratic Bezier curved biomechanical action arrow (e.g. Chest Up / Torso Drop)."""
    if progress <= 0.02:
        return
    n_pts = max(6, int(32 * min(1.0, progress)))
    pts = []
    for i in range(n_pts + 1):
        t = (i / 32.0)
        if t > progress:
            t = progress
        x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        pts.append((x, y))
    if len(pts) < 2:
        return
    r, g, b = color
    glow_draw.line(pts[:-1], fill=(r, g, b, 125), width=width + 12, joint="curve")
    draw.line(pts[:-1], fill=(r, g, b, 255), width=width, joint="curve")
    draw.line(pts[:-1], fill=(255, 255, 255, 190), width=max(2, width // 3), joint="curve")

    tip = pts[-1]
    prev = pts[max(0, len(pts) - 4)]
    dx, dy = tip[0] - prev[0], tip[1] - prev[1]
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux
    base_cx = tip[0] - ux * head_len
    base_cy = tip[1] - uy * head_len
    poly = [tip, (base_cx + nx * head_w, base_cy + ny * head_w), (base_cx - nx * head_w, base_cy - ny * head_w)]
    glow_draw.polygon(poly, fill=(r, g, b, 140))
    draw.polygon(poly, fill=(r, g, b, 255), outline=(255, 255, 255, 220))

def render_step_graphics(frame_bgr: np.ndarray, step_idx: int, local_t: float) -> np.ndarray:
    """
    Renders @demicstory-style biomechanical posture-correction graphics & Box-Zero typography.
    All text & HUD elements strictly obey the v39 iPhone Instagram Reels Safe Zone:
      X in [96..965], Y in [258..1440].
    """
    base_rgba = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sharp_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow_layer)
    s_draw = ImageDraw.Draw(sharp_layer)

    # Smooth draw-on progress in first 0.55s of each 5.0s step, plus gentle pulse
    anim_p = min(1.0, local_t / 0.55)
    pulse = 0.5 + 0.5 * math.sin(local_t * 5.0)
    shift_px = int(round(6.0 * math.sin(local_t * 4.5)))

    is_correct = step_idx in (2, 4)
    theme_col = COL_GREEN if is_correct else COL_RED

    # 1. Top Frameless Header (Safe Zone y >= 258, x >= 96)
    hx, hy = 98, 262
    draw_shadow_text(s_draw, g_draw, (hx, hy), "BARBELL BACK SQUAT", F_TITLE, COL_WHITE)
    s_draw.rectangle([hx, hy + 50, hx + 485, hy + 54], fill=(theme_col[0], theme_col[1], theme_col[2], 245))
    g_draw.rectangle([hx, hy + 49, hx + 485, hy + 55], fill=(theme_col[0], theme_col[1], theme_col[2], 120))

    step_label = f"0{step_idx} / 04 · " + (
        "WRONG: KNEES CAVING IN" if step_idx == 1 else
        "FIX: PUSH KNEES OUT & PLANT FEET" if step_idx == 2 else
        "WRONG: TORSO DROP & HEELS UP" if step_idx == 3 else
        "FIX: CHEST UP & HEELS FLAT"
    )
    draw_shadow_text(s_draw, g_draw, (hx, hy + 62), step_label, F_STEP, theme_col)

    # 2. Step-Specific @demicstory Biomechanical Graphics on Body & Floor
    if step_idx == 1:
        # STEP 1 (Front View — WRONG: Knees Buckling Inward >< in RED)
        # Hips: (360, 1065) & (720, 1065) | Inward-buckled Knees: (438, 1195) & (642, 1195) | Ankles/Feet: (335, 1535) & (748, 1535)
        l_hip, l_knee, l_foot = (362, 1065), (438, 1195), (338, 1535)
        r_hip, r_knee, r_foot = (718, 1065), (642, 1195), (745, 1535)

        # Dashed white ideal vertical reference lines (showing where knees SHOULD be)
        draw_neon_line(s_draw, g_draw, [(350, 1065), (342, 1535)], (180, 195, 210), width=4, dashed=True)
        draw_neon_line(s_draw, g_draw, [(730, 1065), (740, 1535)], (180, 195, 210), width=4, dashed=True)

        # Glowing RED inward-buckled femur-knee-tibia lines (>< shape)
        draw_neon_line(s_draw, g_draw, [l_hip, l_knee, l_foot], COL_RED, width=8)
        draw_neon_line(s_draw, g_draw, [r_hip, r_knee, r_foot], COL_RED, width=8)
        for pt in [l_hip, l_knee, l_foot, r_hip, r_knee, r_foot]:
            draw_neon_joint_node(s_draw, g_draw, pt, COL_RED, radius=9)

        # Bold RED Inward Collapse Arrows at both knees (──▶  ◀──)
        draw_bold_vector_arrow(s_draw, g_draw, (330, 1195), (422 + shift_px, 1195), COL_RED, width=10, progress=anim_p)
        draw_bold_vector_arrow(s_draw, g_draw, (750, 1195), (658 - shift_px, 1195), COL_RED, width=10, progress=anim_p)

        # Red floor instability line between collapsing inner arches
        draw_neon_line(s_draw, g_draw, [(338, 1555), (745, 1555)], COL_RED, width=6, dashed=True)

        # Frameless Action Callouts directly near the vectors
        draw_shadow_text(s_draw, g_draw, (375, 1105), "KNEES CAVING IN", F_VEC_MAIN, COL_RED)
        draw_shadow_text(s_draw, g_draw, (365, 1235), "INNER ARCH COLLAPSING", F_VEC_SUB, COL_SILVER)

        hud_tag = "JOINT OVERLOAD · MEDIAL KNEE & ACL"
        hud_pct = "86%"
        hud_cue = "KNEES CAVING IN"
        bar_ratio = 0.86

    elif step_idx == 2:
        # STEP 2 (Same Front View Zoomed 1.16x — LIVE RED -> GREEN FIX: Push Knees Out & Plant Front Feet)
        # During local_t 0.0 -> 1.1s, the posture & skeletal lines morph live from RED (buckled in) to GREEN (pushed out over toes)!
        morph_t = min(1.0, max(0.0, local_t / 1.10))
        # Ease-out cubic
        morph_e = 1.0 - (1.0 - morph_t) ** 3
        line_col = lerp_col(COL_RED, COL_GREEN, morph_e)

        # In 1.16x zoom (cx=0.5, cy=0.58), coordinates map cleanly:
        l_hip, l_foot = (350, 1025), (322, 1565)
        r_hip, r_foot = (730, 1025), (758, 1565)
        l_knee_bad, l_knee_good = (432, 1185), (336, 1185)
        r_knee_bad, r_knee_good = (648, 1185), (744, 1185)
        l_knee = lerp_pt(l_knee_bad, l_knee_good, morph_e)
        r_knee = lerp_pt(r_knee_bad, r_knee_good, morph_e)

        # Ghost red buckled line fading out so viewer clearly sees "틀린 자세(빨간색)에서 어떻게 변했는지(초록색)"
        if morph_t < 0.95:
            draw_neon_line(s_draw, g_draw, [l_hip, l_knee_bad, l_foot], COL_RED, width=5, dashed=True)
            draw_neon_line(s_draw, g_draw, [r_hip, r_knee_bad, r_foot], COL_RED, width=5, dashed=True)

        # Live morphing Red -> Green leg alignment lines
        draw_neon_line(s_draw, g_draw, [l_hip, l_knee, l_foot], line_col, width=9)
        draw_neon_line(s_draw, g_draw, [r_hip, r_knee, r_foot], line_col, width=9)
        for pt in [l_hip, l_knee, l_foot, r_hip, r_knee, r_foot]:
            draw_neon_joint_node(s_draw, g_draw, pt, line_col, radius=10)

        # Bold GREEN Outward Push Arrows at both knees (◀══  ══▶ PUSH KNEES OUT)
        draw_bold_vector_arrow(s_draw, g_draw, (445, 1185), (235 - shift_px, 1185), COL_GREEN, width=11, head_len=28, head_w=18, progress=anim_p)
        draw_bold_vector_arrow(s_draw, g_draw, (635, 1185), (845 + shift_px, 1185), COL_GREEN, width=11, head_len=28, head_w=18, progress=anim_p)
        draw_shadow_text(s_draw, g_draw, (382, 1105), "PUSH KNEES OUT", F_VEC_MAIN, COL_GREEN)

        # @demicstory Floor Foot Stance Tracks & Downward Root Vectors ("앞발은 어떻게 하는건지 그래픽 표현")
        # Angled green foot-track laser lines under left and right feet showing 15-30 deg toe-out stance + downward floor press
        draw_neon_line(s_draw, g_draw, [(355, 1545), (275, 1615)], COL_GREEN, width=9)
        draw_neon_line(s_draw, g_draw, [(725, 1545), (805, 1615)], COL_GREEN, width=9)
        draw_bold_vector_arrow(s_draw, g_draw, (315, 1455), (315, 1555 + abs(shift_px)), COL_GREEN, width=9, progress=anim_p)
        draw_bold_vector_arrow(s_draw, g_draw, (765, 1455), (765, 1555 + abs(shift_px)), COL_GREEN, width=9, progress=anim_p)
        draw_shadow_text(s_draw, g_draw, (315, 1235), "PLANT FEET · TOES SLIGHTLY OUT", F_VEC_SUB, COL_GREEN)

        hud_tag = "TARGET · QUADRICEPS & ADDUCTORS"
        hud_pct = f"{int(round(lerp(86, 96, morph_e)))}%"
        hud_cue = "PUSH KNEES OUT"
        bar_ratio = lerp(0.86, 0.96, morph_e)

    elif step_idx == 3:
        # STEP 3 (Side Profile View — WRONG: Torso Dropping Forward & Heels Lifting Off Floor in RED)
        # Shoulder/Upper Back: (595, 855) | Rounded Lumbar: (465, 985) | Hip: (415, 1135) | Forward Knee: (665, 1245) | Lifted Heel: (445, 1585) | Toes: (635, 1645)
        sh_pt, lum_pt, hip_pt, knee_pt, heel_pt, toe_pt = (595, 855), (465, 985), (415, 1135), (665, 1245), (445, 1585), (635, 1645)

        # Glowing RED collapsed Spine-Hip-Knee-Heel skeletal line
        draw_neon_line(s_draw, g_draw, [sh_pt, lum_pt, hip_pt, knee_pt, heel_pt, toe_pt], COL_RED, width=8)
        for pt in [sh_pt, hip_pt, knee_pt, heel_pt]:
            draw_neon_joint_node(s_draw, g_draw, pt, COL_RED, radius=9)

        # 1) RED Curved Downward Arrow at Chest/Torso ("상체가 아래로 무너짐 ⤵")
        draw_curved_vector_arrow(s_draw, g_draw, (575, 760), (705, 830), (690, 995 + shift_px), COL_RED, width=10, progress=anim_p)
        draw_shadow_text(s_draw, g_draw, (575, 715), "CHEST DROPPING", F_VEC_MAIN, COL_RED)
        draw_shadow_text(s_draw, g_draw, (575, 755), "LOWER BACK ROUNDING", F_VEC_SUB, COL_SILVER)

        # 2) RED Upward Heel-Lift Arrow + Floor Gap ("뒤꿈치가 바닥에서 들림 ↑")
        # Floor horizontal reference line
        draw_neon_line(s_draw, g_draw, [(360, 1650), (720, 1650)], (180, 195, 210), width=4, dashed=True)
        draw_bold_vector_arrow(s_draw, g_draw, (415, 1648), (415, 1525 - abs(shift_px)), COL_RED, width=10, progress=anim_p)
        draw_shadow_text(s_draw, g_draw, (115, 1205), "HEELS LIFTING UP", F_VEC_MAIN, COL_RED)
        draw_shadow_text(s_draw, g_draw, (115, 1245), "WEIGHT ON TIPTOES", F_VEC_SUB, COL_SILVER)

        hud_tag = "OVERLOAD · LUMBAR SPINE & KNEE"
        hud_pct = "79%"
        hud_cue = "TORSO DROP & HEELS UP"
        bar_ratio = 0.79

    else:
        # STEP 4 (Same Side Profile View Zoomed 1.15x — LIVE RED -> GREEN FIX: Chest Up ↑, Hips Down ↓, Heels Flat ↓)
        morph_t = min(1.0, max(0.0, local_t / 1.15))
        morph_e = 1.0 - (1.0 - morph_t) ** 3
        line_col = lerp_col(COL_RED, COL_GREEN, morph_e)

        # Bad vs Good landmarks in Side Profile
        sh_bad, hip_bad, knee_bad, heel_bad, toe_bad = (605, 845), (410, 1125), (675, 1235), (445, 1565), (645, 1635)
        sh_good, hip_good, knee_good, heel_good, toe_good = (545, 775), (435, 1185), (655, 1185), (465, 1635), (665, 1635)

        sh_pt   = lerp_pt(sh_bad, sh_good, morph_e)
        hip_pt  = lerp_pt(hip_bad, hip_good, morph_e)
        knee_pt = lerp_pt(knee_bad, knee_good, morph_e)
        heel_pt = lerp_pt(heel_bad, heel_good, morph_e)
        toe_pt  = lerp_pt(toe_bad, toe_good, morph_e)

        # Ghost Red collapsed line fading out during first second so viewer sees exact Red -> Green posture change
        if morph_t < 0.95:
            draw_neon_line(s_draw, g_draw, [sh_bad, hip_bad, knee_bad, heel_bad], COL_RED, width=5, dashed=True)

        # Live morphing Red -> Green Spine-Hip-Knee-Ankle alignment line (with @demicstory 90-deg parallel thigh line)
        draw_neon_line(s_draw, g_draw, [sh_pt, hip_pt, knee_pt, heel_pt, toe_pt], line_col, width=9)
        for pt in [sh_pt, hip_pt, knee_pt, heel_pt, toe_pt]:
            draw_neon_joint_node(s_draw, g_draw, pt, line_col, radius=10)

        # @demicstory right-angle square bracket at hip-knee depth once Green locks in
        if morph_e > 0.6:
            draw_neon_line(s_draw, g_draw, [(hip_pt[0], hip_pt[1] - 32), (hip_pt[0] + 32, hip_pt[1] - 32), (hip_pt[0] + 32, hip_pt[1])], COL_GREEN, width=4)

        # 1) "위로 올렸는지": Bold GREEN Upward Curved Vector in front of Chest/Sternum (↑ CHEST UP)
        draw_curved_vector_arrow(s_draw, g_draw, (655, 955), (675, 815), (605, 695 - abs(shift_px)), COL_GREEN, width=11, head_len=28, head_w=18, progress=anim_p)
        draw_shadow_text(s_draw, g_draw, (635, 685), "CHEST UP", F_VEC_MAIN, COL_GREEN)
        draw_shadow_text(s_draw, g_draw, (635, 725), "NEUTRAL SPINE", F_VEC_SUB, COL_WHITE)

        # 2) "아래로 내렸는지": Bold GREEN Downward Vertical Vector at Hips/Glutes (↓ HIPS DOWN)
        draw_bold_vector_arrow(s_draw, g_draw, (345, 995), (345, 1195 + abs(shift_px)), COL_GREEN, width=11, head_len=28, head_w=18, progress=anim_p)
        draw_shadow_text(s_draw, g_draw, (115, 965), "HIPS DOWN", F_VEC_MAIN, COL_GREEN)
        draw_shadow_text(s_draw, g_draw, (115, 1005), "DEEP PARALLEL", F_VEC_SUB, COL_WHITE)

        # 3) "앞발·뒤꿈치는 어떻게 하는건지": Bold GREEN Floor Track Line + Downward Press Arrows on Heel & Forefoot (↓ FEET & HEELS FLAT)
        draw_neon_line(s_draw, g_draw, [(415, 1650), (715, 1650)], COL_GREEN, width=10)
        draw_bold_vector_arrow(s_draw, g_draw, (455, 1505), (455, 1630 + abs(shift_px)), COL_GREEN, width=10, head_len=24, head_w=16, progress=anim_p)
        draw_bold_vector_arrow(s_draw, g_draw, (645, 1525), (645, 1630 + abs(shift_px)), COL_GREEN, width=10, head_len=24, head_w=16, progress=anim_p)
        draw_shadow_text(s_draw, g_draw, (115, 1205), "FEET & HEELS FLAT", F_VEC_MAIN, COL_GREEN)
        draw_shadow_text(s_draw, g_draw, (115, 1245), "PRESS FULL FOOT DOWN", F_VEC_SUB, COL_ORANGE)

        hud_tag = "TARGET · QUADS & GLUTEUS MAXIMUS"
        hud_pct = f"{int(round(lerp(79, 98, morph_e)))}%"
        hud_cue = "CHEST UP · FEET FLAT"
        bar_ratio = lerp(0.79, 0.98, morph_e)

    # 3. Bottom Frameless Telemetry HUD (Safe Zone y <= 1440, x >= 96, Box-Zero)
    bx, by = 98, 1298
    draw_shadow_text(s_draw, g_draw, (bx, by), hud_tag, F_HUD_TAG, theme_col)
    draw_shadow_text(s_draw, g_draw, (bx, by + 34), hud_pct, F_HUD_PCT, COL_WHITE)

    div_x = bx + 188
    s_draw.rectangle([div_x, by + 46, div_x + 4, by + 106], fill=(theme_col[0], theme_col[1], theme_col[2], 255))
    g_draw.rectangle([div_x - 2, by + 44, div_x + 6, by + 108], fill=(theme_col[0], theme_col[1], theme_col[2], 120))

    draw_shadow_text(s_draw, g_draw, (div_x + 20, by + 54), hud_cue, F_HUD_CUE, COL_WHITE)

    bar_y = by + 124
    bar_w = 700
    s_draw.rectangle([bx, bar_y, bx + bar_w, bar_y + 4], fill=(255, 255, 255, 55))
    fill_w = int(bar_w * bar_ratio)
    g_draw.rectangle([bx, bar_y - 3, bx + fill_w, bar_y + 7], fill=(theme_col[0], theme_col[1], theme_col[2], 140))
    s_draw.rectangle([bx, bar_y, bx + fill_w, bar_y + 4], fill=(theme_col[0], theme_col[1], theme_col[2], 255))
    s_draw.ellipse([bx + fill_w - 6, bar_y - 4, bx + fill_w + 6, bar_y + 8], fill=(255, 255, 255, 255))

    # Composite glow + sharp layers onto base frame
    glow_blurred = glow_layer.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=2.0)).resize((W, H), Image.BILINEAR)
    out_rgba = Image.alpha_composite(base_rgba, glow_blurred)
    out_rgba = Image.alpha_composite(out_rgba, sharp_layer)
    return cv2.cvtColor(np.array(out_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)

def build_audio_track(out_wav: str):
    """Builds clear coaching voiceover + mechanical lock/whoosh SFX matching the 4 steps."""
    import edge_tts
    import asyncio

    lines = [
        (0.25, "Stop letting your knees cave inward on squats."),
        (5.15, "Push both knees wide out and plant your feet flat."),
        (10.25, "Never drop your chest forward or lift your heels up."),
        (15.15, "Lift your chest up, drop your hips down, and drive through flat feet."),
    ]
    tmp_dir = "/workspace/daily_info_reels/output/tmp_v42_tts"
    os.makedirs(tmp_dir, exist_ok=True)

    async def gen():
        for i, (_, txt) in enumerate(lines):
            p = os.path.join(tmp_dir, f"line_{i}.mp3")
            comm = edge_tts.Communicate(txt, voice="en-US-GuyNeural", rate="+6%", pitch="-6Hz")
            await comm.save(p)
    asyncio.run(gen())

    # Mix voiceover + subtle dark gym sub-bass pulse + transition SFX using ffmpeg
    inputs = []
    filter_parts = []
    # Base ambient + step transition chimes
    filter_parts.append(f"aevalsrc='0.03*sin(2*PI*55*t)*exp(-mod(t,2.5)*1.8)+0.06*sin(2*PI*420*t)*exp(-mod(t,5.0)*14)':s=44100:d={TOTAL_SEC}[bg]")
    for i, (start_s, _) in enumerate(lines):
        p = os.path.join(tmp_dir, f"line_{i}.mp3")
        inputs.extend(["-i", p])
        delay_ms = int(round(start_s * 1000))
        filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms},volume=1.55[v{i}]")
    mix_inputs = "[bg]" + "".join(f"[v{i}]" for i in range(len(lines)))
    filter_parts.append(f"{mix_inputs}amix=inputs={len(lines)+1}:normalize=0[outa]")
    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", ";".join(filter_parts), "-map", "[outa]", "-t", str(TOTAL_SEC), out_wav]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    front_frames = load_clip_frames(os.path.join(OUT_DIR, "day3_v42_cue1_front_red_to_green.mp4"))
    side_frames  = load_clip_frames(os.path.join(OUT_DIR, "day3_v42_cue2_side_red_to_green.mp4"))

    raw_mp4 = os.path.join(OUT_DIR, "day3_v42_raw.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day3_v42_demic_master_reel.mp4")
    sheet_jpg = os.path.join(OUT_DIR, "day3_v42_demic_master_sheet.jpg")

    writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    keyframes = {}

    for fidx in range(TOTAL_FRAMES):
        t = fidx / FPS
        step_idx = min(4, int(t // 5.0) + 1)
        local_t = t - (step_idx - 1) * 5.0
        u = local_t / 5.0  # 0.0 -> 1.0 within step

        if step_idx == 1:
            # Step 1 (0-5s): Front View Wrong (Knees Caving In >< in RED)
            # Sample first 36% of front clip (where glow is red) in a smooth down-up-down squat cycle
            clip_u = 0.34 * (0.5 - 0.5 * math.cos(u * 2.0 * math.pi))
            base = sample_clip(front_frames, clip_u)
            # Progressive inward knee-valgus buckle as mannequin squats down (up to 86px inward shift!)
            valgus_shift = 42.0 + 44.0 * math.sin(u * math.pi)
            base = warp_knee_valgus(base, valgus_shift)
            # Wide full-rack framing (zoom=1.0)
        elif step_idx == 2:
            # Step 2 (5-10s): Same Front View -> Live Red-to-Green Posture Correction (Knees Push Out + Full Rep in Green)
            # First 1.1s transitions from Red (clip_u=0.25 -> 0.55) while knee valgus unwinds from 82px -> 0px!
            # Remaining 3.9s performs full deep squat reps in Green/Orange (clip_u=0.55 -> 1.0 -> 0.62 -> 1.0)
            if local_t < 1.1:
                m = local_t / 1.1
                clip_u = lerp(0.25, 0.58, m)
                valgus_shift = lerp(82.0, 0.0, 1.0 - (1.0 - m) ** 3)
            else:
                r_u = (local_t - 1.1) / 3.9
                clip_u = 0.58 + 0.42 * (0.5 - 0.5 * math.cos(r_u * 2.0 * math.pi))
                valgus_shift = 0.0
            base = sample_clip(front_frames, clip_u)
            base = warp_knee_valgus(base, valgus_shift)
            # 1.24x medium-close focus framing on torso, knees, and front feet so posture change & foot tracks are crystal clear
            base = apply_camera_crop(base, zoom=1.24, cx_norm=0.50, cy_norm=0.61)
        elif step_idx == 3:
            # Step 3 (10-15s): Strict Side Profile Wrong (Torso Dropping Forward & Heels Lifting Off Floor in RED)
            clip_u = 0.34 * (0.5 - 0.5 * math.cos(u * 2.0 * math.pi))
            base = sample_clip(side_frames, clip_u)
        else:
            # Step 4 (15-20s): Same Side Profile View -> Live Red-to-Green Posture Correction (Chest Up ↑, Hips Down ↓, Heels Flat ↓)
            if local_t < 1.15:
                m = local_t / 1.15
                clip_u = lerp(0.24, 0.58, m)
            else:
                r_u = (local_t - 1.15) / 3.85
                clip_u = 0.58 + 0.42 * (0.5 - 0.5 * math.cos(r_u * 2.0 * math.pi))
            base = sample_clip(side_frames, clip_u)
            # 1.22x lower-side focus framing so Chest Up, Hips Down, and Flat Heels fill the frame clearly
            base = apply_camera_crop(base, zoom=1.22, cx_norm=0.54, cy_norm=0.59)

        rendered = render_step_graphics(base, step_idx, local_t)
        writer.write(rendered)

        if fidx in (int(2.2 * FPS), int(7.5 * FPS), int(12.2 * FPS), int(17.5 * FPS)):
            keyframes[step_idx] = rendered.copy()

    writer.release()

    # Build audio and mux H.264 MP4
    wav_path = os.path.join(OUT_DIR, "day3_v42_audio.wav")
    build_audio_track(wav_path)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", raw_mp4,
        "-i", wav_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        final_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Save 4-panel comparison sheet (1080x1920 -> 4 columns x 1 row)
    pw, ph = 360, 640
    sheet = Image.new("RGB", (pw * 4, ph), (12, 14, 18))
    for s in range(1, 5):
        rgb = cv2.cvtColor(keyframes[s], cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(im, ((s - 1) * pw, 0))
    sheet.save(sheet_jpg, quality=95)
    print("Successfully built:", final_mp4)
    print("Successfully built:", sheet_jpg)

if __name__ == "__main__":
    main()
