#!/usr/bin/env python3
"""
GymForm 3D — v27 Multi-Camera, Speed-Ramping & High-Completion 4-Concept Suite
Addresses user feedback (t119u–t129u):
1. "시각적으로 잘보이게끔 해" — Ultra-bold typography, high-contrast 4x supersampled vector graphics, zero tiny text.
2. "카메라 전환도 있으면 좋고" & "전환 같은게 멋져야해. 속도감 조절이라든지" —
   - Multi-camera framing across steps (CAM 1 Wide 1.0x -> CAM 2 Joint/Delt Zoom 1.42x -> CAM 3 Mid Hinge 1.16x -> CAM 4 Triceps Lockout Macro 1.46x)
   - Explicit Speed-Ramping curve (`1.8x WHIP-ZOOM IN` -> `0.3x SLOW-MO PEAK FREEZE` -> `1.0x REP TEMPO: 1s PUSH · 1s SQUEEZE · 3s ECCENTRIC`).
3. "그래픽적으로 완성도가 높아야해" & "배치나 구조도 잘잡고" & "카피할 인기많은 헬스 릴스도 찾아보고 최종적으로 시안 여러개 줘바" —
   Generates 4 distinct, benchmark-inspired final concepts (Concept A, Concept B, Concept C, Concept D) +
   a dedicated Multi-Camera & Speed-Ramp Architecture Board.
"""

import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import gymform_engine as eng
import build_v25_fast_preview as v25

W, H = eng.W, eng.H
OUT_DIR = eng.OUT_DIR

COL_RED = (255, 58, 48)
COL_AMBER = (255, 184, 42)
COL_GREEN = (24, 232, 108)
COL_CYAN = (68, 218, 255)
COL_WHITE = (255, 255, 255)

FONT_H1 = eng.load_font(40, bold=True)
FONT_H2 = eng.load_font(28, bold=True)
FONT_H3 = eng.load_font(22, bold=True)
FONT_BODY = eng.load_font(18, bold=True)
FONT_SUB = eng.load_font(15, bold=True)
FONT_SM = eng.load_font(13, bold=True)
FONT_DIAL_BIG = eng.load_font(44, bold=True)


def apply_cinema_camera(
    rgb_arr: np.ndarray,
    zoom: float = 1.0,
    focus_xy: tuple[float, float] = (680.0, 760.0),
    dest_xy: tuple[float, float] = (675.0, 770.0),
    whip_blur_px: int = 0,
) -> tuple[np.ndarray, callable]:
    """
    Applies a virtual cinema camera crop/zoom & focal re-centering with optional directional whip-pan/zoom blur,
    and returns `(transformed_rgb, map_pt_fn)` so any anatomical coordinate `(x, y)` on the base plate
    maps accurately to the new camera frame!
    """
    if abs(zoom - 1.0) < 1e-3 and whip_blur_px == 0:
        return rgb_arr.copy(), (lambda x, y: (float(x), float(y)))

    fx, fy = focus_xy
    dx, dy = dest_xy

    # Affine matrix M mapping destination (x_d, y_d) -> source (x_s, y_s):
    # x_s = fx + (x_d - dx) / zoom
    # Forward map (source -> dest):
    # x_d = dx + (x_s - fx) * zoom
    M = np.float32([
        [zoom, 0.0, dx - fx * zoom],
        [0.0, zoom, dy - fy * zoom],
    ])
    out = cv2.warpAffine(
        rgb_arr,
        M,
        (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(14, 16, 20),
    )

    if whip_blur_px > 0:
        # Subtle radial/directional motion blur on outer periphery to convey fast camera push-in
        k = max(3, int(whip_blur_px) | 1)
        blurred = cv2.GaussianBlur(out, (k, k), sigmaX=whip_blur_px * 0.45)
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r_norm = np.sqrt(((xx - dx) / (W * 0.42)) ** 2 + ((yy - dy) / (H * 0.38)) ** 2)
        edge_w = np.clip((r_norm - 0.55) / 0.65, 0.0, 0.65)[..., None]
        out = np.clip(out * (1.0 - edge_w) + blurred * edge_w, 0, 255).astype(np.uint8)

    def map_pt(x: float, y: float) -> tuple[float, float]:
        return (dx + (x - fx) * zoom, dy + (y - fy) * zoom)

    return out, map_pt


def draw_4x_badge(canvas: Image.Image, bx: int, by: int, br: int, is_wrong: bool):
    accent = COL_RED if is_wrong else COL_GREEN
    ss = 4
    sz = (br * 2 + 28) * ss
    c = sz // 2
    r = br * ss
    em = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    ed = ImageDraw.Draw(em)

    # Outer ambient glow ring
    ed.ellipse([c - r - 6 * ss, c - r - 6 * ss, c + r + 6 * ss, c + r + 6 * ss], fill=(*accent, 34))
    ed.ellipse([c - r, c - r, c + r, c + r], fill=(14, 18, 25, 246), outline=(230, 238, 250, 110), width=2 * ss)
    ri = r - 6 * ss
    ed.ellipse([c - ri, c - ri, c + ri, c + ri], fill=(18, 22, 30, 252), outline=(*accent, 255), width=4 * ss)

    if is_wrong:
        d = int(15 * ss)
        w = int(7 * ss)
        ed.line([(c - d, c - d), (c + d, c + d)], fill=(255, 255, 255, 255), width=w)
        ed.line([(c - d, c + d), (c + d, c - d)], fill=(255, 255, 255, 255), width=w)
    else:
        pts = [(c - 18 * ss, c + 1 * ss), (c - 4 * ss, c + 15 * ss), (c + 20 * ss, c - 13 * ss)]
        ed.line(pts, fill=(255, 255, 255, 255), width=int(7.5 * ss), joint="curve")

    em_down = em.resize((br * 2 + 28, br * 2 + 28), Image.Resampling.LANCZOS)
    canvas.alpha_composite(em_down, (bx - br - 14, by - br - 14))


def draw_top_navigation_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    cam_label: str,
    speed_label: str,
):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = COL_RED if is_wrong else COL_GREEN

    # 1. Camera & Speed-Ramp Top Status Pill (`y = 176..220`)
    cp_x1, cp_y1, cp_x2, cp_y2 = 84, 174, 768, 220
    od.rounded_rectangle([cp_x1, cp_y1, cp_x2, cp_y2], radius=23, fill=(12, 16, 22, 228), outline=(215, 228, 245, 68), width=2)
    # Live recording dot
    od.ellipse([cp_x1 + 16, cp_y1 + 14, cp_x1 + 32, cp_y1 + 30], fill=(*accent, 255))
    od.text((cp_x1 + 42, cp_y1 + 12), cam_label, font=FONT_SUB, fill=(255, 255, 255, 255))

    sp_b = od.textbbox((0, 0), speed_label, font=FONT_SM)
    sp_w = (sp_b[2] - sp_b[0]) + 24
    od.rounded_rectangle([cp_x2 - sp_w - 8, cp_y1 + 7, cp_x2 - 8, cp_y2 - 7], radius=16, fill=(*accent, 48), outline=(*accent, 210), width=2)
    od.text((cp_x2 - sp_w + 4, cp_y1 + 13), speed_label, font=FONT_SM, fill=(*accent, 255))

    # 2. Labeled 3-Stage Diagnostic Dock (`y = 232..392`)
    tw, th = 148, 108
    lbl_h = 28
    gap = 12
    pad = 12
    bar_x = 84
    bar_y = 232
    bar_w = 3 * tw + 2 * gap + 2 * pad
    bar_h = th + lbl_h + 2 * pad

    od.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=24,
        fill=(14, 18, 25, 232),
        outline=(220, 232, 248, 72),
        width=2,
    )

    tab_labels = ["1. JOINT LOAD", "2. DELT STEAL", "3. 3-HEAD FIX"]
    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = bar_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy().resize((tw, th), Image.Resampling.LANCZOS) if i == active_idx else inact_im.copy().resize((tw, th), Image.Resampling.LANCZOS)

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=12, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        i_col = COL_RED if i < 2 else COL_GREEN
        tb = od.textbbox((0, 0), tab_labels[i], font=FONT_SM)
        if i == active_idx:
            od.rounded_rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + lbl_h + 2], radius=14, outline=(*i_col, 255), width=3)
            od.rounded_rectangle([tx, ty + th - 2, tx + tw, ty + th + lbl_h], radius=10, fill=(*i_col, 245))
            txt_col = (10, 14, 18, 255) if i == 2 else (255, 255, 255, 255)
            od.text((tx + (tw - (tb[2] - tb[0])) // 2, ty + th + 5), tab_labels[i], font=FONT_SM, fill=txt_col)
        else:
            od.text((tx + (tw - (tb[2] - tb[0])) // 2, ty + th + 5), tab_labels[i], font=FONT_SM, fill=(160, 174, 194, 220))

    canvas.alpha_composite(ov)
    draw_4x_badge(canvas, 912, bar_y + bar_h // 2, 48, is_wrong)


def draw_4x_protractor_and_callout(
    canvas: Image.Image,
    joint_xy: tuple[float, float],
    anchor_xy: tuple[int, int],
    deg_text: str,
    title_text: str,
    sub_text: str,
    is_wrong: bool,
    arc_angles: tuple[int, int] = (-90, -55),
):
    """
    Draws a 4x-supersampled glowing biomechanical protractor arc around the joint +
    large, ultra-legible frameless typography in the open left studio space.
    """
    ss = 4
    ov_ss = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d_ss = ImageDraw.Draw(ov_ss)
    col = COL_RED if is_wrong else COL_GREEN

    jx, jy = int(round(joint_xy[0] * ss)), int(round(joint_xy[1] * ss))
    r_arc = 68 * ss

    # Concentric precision rings + degree sector
    d_ss.pieslice(
        [jx - r_arc, jy - r_arc, jx + r_arc, jy + r_arc],
        start=arc_angles[0],
        end=arc_angles[1],
        fill=(*col, 58),
        outline=(*col, 245),
        width=3 * ss,
    )
    d_ss.ellipse([jx - 18 * ss, jy - 18 * ss, jx + 18 * ss, jy + 18 * ss], outline=(255, 255, 255, 235), width=2 * ss)
    d_ss.ellipse([jx - 6 * ss, jy - 6 * ss, jx + 6 * ss, jy + 6 * ss], fill=(*col, 255))

    # Reference axis dashed/solid lines
    for a_deg in arc_angles:
        rad = math.radians(a_deg)
        ex = int(jx + (r_arc + 22 * ss) * math.cos(rad))
        ey = int(jy + (r_arc + 22 * ss) * math.sin(rad))
        d_ss.line([(jx, jy), (ex, ey)], fill=(255, 255, 255, 215), width=2 * ss)

    # Leader line from joint to left studio callout
    ax, ay = anchor_xy
    tb_deg = ImageDraw.Draw(canvas).textbbox((0, 0), deg_text, font=FONT_H1)
    tb_tit = ImageDraw.Draw(canvas).textbbox((0, 0), title_text, font=FONT_H3)
    line_end_x = ax + max(tb_deg[2] - tb_deg[0], tb_tit[2] - tb_tit[0]) + 18

    pts = [
        (jx, jy),
        ((line_end_x + 36) * ss, jy),
        (line_end_x * ss, (ay + 42) * ss),
        (ax * ss, (ay + 42) * ss),
    ]
    d_ss.line(pts, fill=(6, 8, 12, 220), width=6 * ss, joint="curve")
    d_ss.line(pts, fill=(*col, 250), width=3 * ss, joint="curve")

    ov = ov_ss.resize((W, H), Image.Resampling.LANCZOS)
    od = ImageDraw.Draw(ov)

    # Giant degree + bold title (Frameless with crisp drop shadow for 100% visibility)
    for dx_s, dy_s in [(-2, 2), (2, 2), (0, 3)]:
        od.text((ax + dx_s, ay - 6 + dy_s), deg_text, font=FONT_H1, fill=(0, 0, 0, 235))
        od.text((ax + dx_s, ay + 48 + dy_s), title_text, font=FONT_H3, fill=(0, 0, 0, 235))
        od.text((ax + dx_s, ay + 78 + dy_s), sub_text, font=FONT_SUB, fill=(0, 0, 0, 235))

    od.text((ax, ay - 6), deg_text, font=FONT_H1, fill=(*col, 255))
    od.text((ax, ay + 48), title_text, font=FONT_H3, fill=(255, 255, 255, 255))
    od.text((ax, ay + 78), sub_text, font=FONT_SUB, fill=(215, 228, 245, 255))

    canvas.alpha_composite(ov)


def draw_concept_A_hud(canvas: Image.Image, targets: list[tuple[float, float, str]], phase: dict):
    """
    Concept A: @demicstory Multi-Cam + High-Contrast Radial Concept B Master
    Big, ultra-legible 270° circular dial (`cx=176, cy=1310, r=90`) +
    Unified Floating Capsule (`x=252..988, y=1196..1424`) with bold `WHY/DO` + visual 3-head mini gauge pills.
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    ring_cx, ring_cy, ring_r = 176, 1310, 90

    # Reticle link from active muscle/joint to top of dial
    px, py, _ = targets[0]
    px_i, py_i = int(round(px)), int(round(py))
    od.ellipse([px_i - 18, py_i - 18, px_i + 18, py_i + 18], outline=(*accent, 255), width=3)
    od.ellipse([px_i - 6, py_i - 6, px_i + 6, py_i + 6], fill=(255, 255, 255, 255))

    ax, ay = ring_cx, ring_cy - ring_r - 6
    pts = [(px_i - 18, py_i), (ax + 28, py_i), (ax, py_i + 32), (ax, ay)]
    od.line(pts, fill=(8, 10, 14, 210), width=6, joint="curve")
    od.line(pts, fill=(*accent, 240), width=3, joint="curve")

    # Unified Floating Capsule (`x = 248..988, y = 1196..1424`)
    px1, py1, px2, py2 = 248, 1196, 988, 1424
    od.rounded_rectangle([px1 + 4, py1 + 10, px2 + 4, py2 + 10], radius=48, fill=(4, 6, 10, 175))
    od.rounded_rectangle([px1, py1, px2, py2], radius=48, fill=(14, 18, 26, 236), outline=(*accent, 150), width=3)

    # 4x Supersampled 270° Circular Dial
    pct_val = phase["pct"]
    ss = 4
    dsz = (ring_r * 2 + 28) * ss
    dc = dsz // 2
    dr = ring_r * ss
    dial = Image.new("RGBA", (dsz, dsz), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dial)
    dd.ellipse([dc - dr, dc - dr, dc + dr, dc + dr], fill=(12, 16, 22, 252), outline=(225, 235, 250, 105), width=3 * ss)
    arc_box = [dc - dr + 12 * ss, dc - dr + 12 * ss, dc + dr - 12 * ss, dc + dr - 12 * ss]
    dd.arc(arc_box, start=135, end=405, fill=(38, 44, 58, 240), width=13 * ss)
    end_deg = 135 + int(round(270 * (pct_val / 100.0)))
    dd.arc(arc_box, start=135, end=end_deg, fill=(*accent, 255), width=13 * ss)
    dial_down = dial.resize((ring_r * 2 + 28, ring_r * 2 + 28), Image.Resampling.LANCZOS)
    ov.alpha_composite(dial_down, (ring_cx - ring_r - 14, ring_cy - ring_r - 14))

    p_str = f"{pct_val}%"
    pb = od.textbbox((0, 0), p_str, font=FONT_DIAL_BIG)
    od.text((ring_cx - (pb[2] - pb[0]) // 2, ring_cy - 34), p_str, font=FONT_DIAL_BIG, fill=(255, 255, 255, 255))
    sb = od.textbbox((0, 0), phase["dial_tag"], font=FONT_SUB)
    od.text((ring_cx - (sb[2] - sb[0]) // 2, ring_cy + 20), phase["dial_tag"], font=FONT_SUB, fill=(*accent, 255))

    # Main Coaching Copy inside Capsule
    tx = 286
    od.text((tx, py1 + 18), phase["kicker"], font=FONT_SUB, fill=(*accent, 255))
    od.text((tx, py1 + 44), phase["line1"], font=FONT_H1, fill=(255, 255, 255, 255))
    od.text((tx, py1 + 98), phase["line2"], font=FONT_BODY, fill=(230, 238, 250, 255))

    # 3 Visual Mini-Gauge Pills at Bottom of Capsule (`y = 1338..1402`)
    gx = tx
    gy1, gy2 = py1 + 142, py1 + 204
    gw = 216
    for lbl, val, c_bar in phase["splits"]:
        od.rounded_rectangle([gx, gy1, gx + gw, gy2], radius=18, fill=(22, 28, 38, 235), outline=(210, 222, 240, 55), width=2)
        od.text((gx + 14, gy1 + 8), lbl, font=FONT_SM, fill=(205, 218, 235, 245))
        v_str = f"{val}%"
        vb = od.textbbox((0, 0), v_str, font=FONT_SUB)
        od.text((gx + gw - (vb[2] - vb[0]) - 14, gy1 + 6), v_str, font=FONT_SUB, fill=(*c_bar, 255))
        # Bar track
        bx1, by1, bx2, by2 = gx + 14, gy1 + 38, gx + gw - 14, gy1 + 50
        od.rounded_rectangle([bx1, by1, bx2, by2], radius=6, fill=(38, 44, 56, 255))
        fill_w = max(12, int((bx2 - bx1) * (val / 100.0)))
        od.rounded_rectangle([bx1, by1, bx1 + fill_w, by2], radius=6, fill=(*c_bar, 255))
        gx += gw + 12

    canvas.alpha_composite(ov)


def draw_concept_B_pip_zoom_lens(
    canvas: Image.Image,
    base_rgb: np.ndarray,
    focus_xy: tuple[float, float],
    phase: dict,
):
    """
    Concept B: Muscle & Motion Style Picture-in-Picture 2.2x Macro Zoom Lens + Full Coaching Deck
    Places a circular 2.2x Anatomical Magnifier Window (`cx=255, cy=735, r=168`) on the left studio margin,
    linked directly to the active joint/muscle head on the full-body mannequin!
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    lcx, lcy, lr = 250, 740, 162
    fx, fy = int(focus_xy[0]), int(focus_xy[1])

    # Crop 2.2x zoom around focus_xy from base_rgb
    half_crop = int(lr / 2.15)
    x0 = max(0, min(W - 2 * half_crop, fx - half_crop))
    y0 = max(0, min(H - 2 * half_crop, fy - half_crop))
    patch = Image.fromarray(base_rgb[y0 : y0 + 2 * half_crop, x0 : x0 + 2 * half_crop]).resize((lr * 2, lr * 2), Image.Resampling.LANCZOS)

    # Circular mask
    mask = Image.new("L", (lr * 2, lr * 2), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, lr * 2 - 1, lr * 2 - 1], fill=255)

    # Drop shadow + paste magnified lens
    od.ellipse([lcx - lr - 10, lcy - lr - 6, lcx + lr + 10, lcy + lr + 14], fill=(0, 0, 0, 185))
    ov.paste(patch, (lcx - lr, lcy - lr), mask)

    # Crosshair grid inside lens + 4x supersampled metallic rim
    od.line([(lcx - lr + 20, lcy), (lcx - 28, lcy)], fill=(*accent, 140), width=2)
    od.line([(lcx + 28, lcy), (lcx + lr - 20, lcy)], fill=(*accent, 140), width=2)
    od.line([(lcx, lcy - lr + 20), (lcx, lcy - 28)], fill=(*accent, 140), width=2)
    od.line([(lcx, lcy + 28), (lcx, lcy + lr - 20)], fill=(*accent, 140), width=2)
    od.ellipse([lcx - lr, lcy - lr, lcx + lr, lcy + lr], outline=(235, 242, 252, 220), width=3)
    od.ellipse([lcx - lr - 4, lcy - lr - 4, lcx + lr + 4, lcy + lr + 4], outline=(*accent, 255), width=4)

    # Connect lens to the mannequin joint with a surgical callout line
    od.ellipse([fx - 22, fy - 22, fx + 22, fy + 22], outline=(*accent, 255), width=3)
    od.line([(lcx + lr + 4, lcy), (fx - 22, fy)], fill=(*accent, 240), width=3)

    # Lens top badge (`2.2× MACRO ZOOM`) & bottom degree tag
    tag_txt = "2.2× ANATOMY ZOOM"
    tb = od.textbbox((0, 0), tag_txt, font=FONT_SM)
    tw = (tb[2] - tb[0]) + 28
    od.rounded_rectangle([lcx - tw // 2, lcy - lr - 20, lcx + tw // 2, lcy - lr + 14], radius=17, fill=(14, 18, 25, 248), outline=(*accent, 245), width=2)
    od.text((lcx - (tb[2] - tb[0]) // 2, lcy - lr - 11), tag_txt, font=FONT_SM, fill=(*accent, 255))

    deg_lbl = f"{phase['deg_badge']} · {phase['deg_title']}"
    db = od.textbbox((0, 0), deg_lbl, font=FONT_SUB)
    dw = (db[2] - db[0]) + 32
    od.rounded_rectangle([lcx - dw // 2, lcy + lr - 18, lcx + dw // 2, lcy + lr + 22], radius=20, fill=(14, 18, 25, 252), outline=(230, 238, 250, 120), width=2)
    od.text((lcx - (db[2] - db[0]) // 2, lcy + lr - 7), deg_lbl, font=FONT_SUB, fill=(255, 255, 255, 255))

    canvas.alpha_composite(ov)
    draw_concept_A_hud(canvas, [(fx, fy, "primary")], phase)


def draw_concept_C_ar_speedramp_hud(
    canvas: Image.Image,
    targets: list[tuple[float, float, str]],
    phase: dict,
):
    """
    Concept C: Squat University / Sports Science 3D AR Biomechanical Grid + Live Speed-Ramp & Tempo Wave HUD
    Features on-body 3D angle protractor + a sleek Speed-Ramp & Rep Tempo Waveform Strip (`1.8× WHIP ZOOM -> 0.3× PEAK FREEZE -> 3.0s ECCENTRIC`)
    integrated directly above the Concept B coaching capsule.
    """
    draw_concept_A_hud(canvas, targets, phase)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    is_wrong = phase["is_wrong"]
    accent = COL_RED if is_wrong else COL_GREEN

    # Speed-Ramp & Biomechanical Tempo Waveform Bar (`x = 84..988, y = 1112..1178`)
    sx1, sy1, sx2, sy2 = 84, 1112, 988, 1178
    od.rounded_rectangle([sx1, sy1, sx2, sy2], radius=24, fill=(12, 16, 22, 232), outline=(215, 228, 245, 72), width=2)
    od.text((sx1 + 22, sy1 + 12), "SPEED RAMP & TEMPO", font=FONT_SM, fill=(*accent, 255))
    od.text((sx1 + 22, sy1 + 34), phase["tempo_str"], font=FONT_SUB, fill=(255, 255, 255, 255))

    # Draw dynamic speed-ramp curve on the right half (`x = 560..964`)
    wx1, wx2 = 565, 962
    wy_mid = sy1 + 34
    pts_wave = []
    for i in range(60):
        t = i / 59.0
        wx = int(wx1 + t * (wx2 - wx1))
        # Speed curve: high on transition (left), dips low for 0.3x slow-mo freeze (center), rises for eccentric (right)
        env = math.sin(t * math.pi * 2.5) * math.exp(-((t - 0.48) ** 2) / 0.08)
        wy = int(wy_mid - env * 18)
        pts_wave.append((wx, wy))
    od.line(pts_wave, fill=(*accent, 245), width=3, joint="curve")
    # Highlight the 0.3x Slow-Mo Freeze Node
    fx_node = int(wx1 + 0.48 * (wx2 - wx1))
    od.ellipse([fx_node - 7, wy_mid - 25, fx_node + 7, wy_mid - 11], fill=(255, 255, 255, 255), outline=(*accent, 255), width=3)
    od.text((fx_node - 54, sy1 + 42), "0.3× SLOW-MO HOLD", font=FONT_SM, fill=(*accent, 255))

    canvas.alpha_composite(ov)


def build_concept_D_split_versus(
    wrong_frame_rgb: np.ndarray,
    correct_frame_rgb: np.ndarray,
    phase_wrong: dict,
    phase_correct: dict,
) -> Image.Image:
    """
    Concept D: Viral Side-by-Side `✕ MISTAKE` vs `✓ PRO FIX` Direct Comparison Format
    Splits the 1080×1920 vertical canvas into Left (`✕ MISTAKE · 86% JOINT LOAD / 72% DELT STEAL`)
    and Right (`✓ PRO FIX · 99% ALL 3 TRICEPS HEADS`), zoomed in with cinema framing so both figures are huge and unmistakable!
    """
    canvas = Image.new("RGBA", (W, H), (12, 14, 18, 255))
    od = ImageDraw.Draw(canvas)

    half_w = W // 2 - 4
    crop_h = 1180
    top_y = 220

    # Zoom and center both figures into their vertical split panes
    w_zoom, _ = apply_cinema_camera(wrong_frame_rgb, zoom=1.22, focus_xy=(690, 770), dest_xy=(540, 760))
    c_zoom, _ = apply_cinema_camera(correct_frame_rgb, zoom=1.22, focus_xy=(670, 770), dest_xy=(540, 760))

    left_im = Image.fromarray(w_zoom[top_y : top_y + crop_h, 270 : 270 + half_w]).convert("RGBA")
    right_im = Image.fromarray(c_zoom[top_y : top_y + crop_h, 270 : 270 + half_w]).convert("RGBA")

    canvas.paste(left_im, (0, top_y))
    canvas.paste(right_im, (W // 2 + 4, top_y))

    # Glowing divider line
    od.rectangle([W // 2 - 3, top_y, W // 2 + 3, top_y + crop_h], fill=(32, 38, 48, 255))

    # Top Title Banner (`y = 140..212`)
    od.rounded_rectangle([64, 136, W - 64, 208], radius=24, fill=(16, 20, 28, 242), outline=(220, 232, 248, 80), width=2)
    tit = "CABLE ROPE PUSHDOWN · FORM COMPARISON"
    tb = od.textbbox((0, 0), tit, font=FONT_H3)
    od.text(((W - (tb[2] - tb[0])) // 2, 158), tit, font=FONT_H3, fill=(255, 255, 255, 255))

    # Top-Left `✕` and Top-Right `✓` Badges + Pane Headers
    draw_4x_badge(canvas, 96, top_y + 68, 42, True)
    draw_4x_badge(canvas, W - 96, top_y + 68, 42, False)

    od.rounded_rectangle([156, top_y + 38, half_w - 20, top_y + 98], radius=18, fill=(20, 14, 18, 228), outline=(*COL_RED, 220), width=2)
    od.text((174, top_y + 46), "MISTAKE · 90° STOP", font=FONT_SUB, fill=(*COL_RED, 255))
    od.text((174, top_y + 69), "ELBOWS LIFTED UP", font=FONT_BODY, fill=(255, 255, 255, 255))

    rx1 = W // 2 + 24
    od.rounded_rectangle([rx1, top_y + 38, W - 156, top_y + 98], radius=18, fill=(14, 22, 18, 228), outline=(*COL_GREEN, 220), width=2)
    od.text((rx1 + 18, top_y + 46), "PRO FIX · 180° LOCK", font=FONT_SUB, fill=(*COL_GREEN, 255))
    od.text((rx1 + 18, top_y + 69), "ELBOWS PINNED IN", font=FONT_BODY, fill=(255, 255, 255, 255))

    # Bottom Side-by-Side Comparison Deck (`y = 1190..1435`)
    # Left Card (Mistake)
    od.rounded_rectangle([36, 1188, half_w - 10, 1432], radius=32, fill=(18, 16, 22, 242), outline=(*COL_RED, 195), width=3)
    od.text((64, 1210), "72% DELT STEAL", font=FONT_H2, fill=(*COL_RED, 255))
    od.text((64, 1252), "AVOID HALF REPS", font=FONT_H3, fill=(255, 255, 255, 255))
    od.text((64, 1286), "✕ FRONT DELT TAKES OVER", font=FONT_SUB, fill=(225, 200, 200, 245))
    # Mini bars
    for idx_b, (b_lbl, b_val, b_col) in enumerate([("FRONT DELT LOAD", 72, COL_RED), ("TRICEPS WORK", 28, COL_AMBER)]):
        by = 1326 + idx_b * 46
        od.text((64, by), f"{b_lbl}: {b_val}%", font=FONT_SM, fill=(220, 228, 240, 245))
        od.rounded_rectangle([64, by + 20, half_w - 38, by + 32], radius=6, fill=(38, 44, 56, 255))
        fw = int((half_w - 102) * (b_val / 100.0))
        od.rounded_rectangle([64, by + 20, 64 + fw, by + 32], radius=6, fill=(*b_col, 255))

    # Right Card (Pro Fix)
    od.rounded_rectangle([W // 2 + 14, 1188, W - 36, 1432], radius=32, fill=(14, 22, 20, 242), outline=(*COL_GREEN, 195), width=3)
    rx = W // 2 + 42
    od.text((rx, 1210), "99% 3-HEAD EMG", font=FONT_H2, fill=(*COL_GREEN, 255))
    od.text((rx, 1252), "PIN & SPREAD 180°", font=FONT_H3, fill=(255, 255, 255, 255))
    od.text((rx, 1286), "✓ 15° LEAN · 4×12 · 1s HOLD", font=FONT_SUB, fill=(205, 245, 220, 245))
    for idx_b, (b_lbl, b_val, b_col) in enumerate([("TRICEPS 3 HEADS", 99, COL_GREEN), ("FRONT DELT STEAL", 6, COL_CYAN)]):
        by = 1326 + idx_b * 46
        od.text((rx, by), f"{b_lbl}: {b_val}%", font=FONT_SM, fill=(220, 228, 240, 245))
        od.rounded_rectangle([rx, by + 20, W - 64, by + 32], radius=6, fill=(38, 44, 56, 255))
        fw = max(12, int((W - 64 - rx) * (b_val / 100.0)))
        od.rounded_rectangle([rx, by + 20, rx + fw, by + 32], radius=6, fill=(*b_col, 255))

    return canvas.convert("RGB")


def main():
    f_w45 = eng.prepare_xray_bone_frame(v25.read_specific_frame(eng.WRONG_I2V, 45))
    f_c14 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 14))
    f_c42 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 42))
    f_c64 = eng.prepare_muscle_frame(v25.read_specific_frame(eng.CORRECT_I2V, 64))

    # Build top 3-thumbnail icons
    size = (148, 108)
    crop_box = (555, 550, 905, 950)
    inact_base = Image.fromarray(f_c42["mono"]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 120)))

    x_arr, x_targets, _, _ = eng.render_xray_skeletal_frame(f_w45, 1.0, 0.90)
    im0 = Image.fromarray(x_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m1_arr, m1_targets = v25.render_natural_muscle_frame(f_c14, "anterior_deltoid", True, 1.0)
    im1 = Image.fromarray(m1_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    m2_arr, m2_targets = v25.render_natural_muscle_frame(f_c42, "triceps_lat_long", False, 1.0)
    m3_arr, m3_targets = v25.render_natural_muscle_frame(f_c64, "triceps_all_three", False, 1.0)
    im2 = Image.fromarray(m3_arr).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)

    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]

    # Add clean on-body geometry to base renders before cinema camera zoom
    def with_onbody_geom(arr: np.ndarray, step_idx: int) -> np.ndarray:
        p = Image.fromarray(arr).convert("RGBA")
        v25.draw_clean_on_body_coaching_geometry(p, step_idx, 1.0)
        return np.array(p.convert("RGB"))

    base_step1 = with_onbody_geom(x_arr, 1)
    base_step2 = with_onbody_geom(m1_arr, 2)
    base_step3 = with_onbody_geom(m2_arr, 3)
    base_step4 = with_onbody_geom(m3_arr, 4)

    steps_spec = [
        {
            "idx": 1,
            "is_wrong": True,
            "base_rgb": base_step1,
            "raw_targets": x_targets,
            "active_slot": 0,
            "cam_zoom": 1.04,
            "cam_focus": (690.0, 735.0),
            "cam_dest": (690.0, 745.0),
            "cam_label": "CAM 1 · WIDE SIDE POSTURE",
            "speed_label": "1.0× -> 0.3× FREEZE",
            "tempo_str": "1.8× WHIP-IN  ▸  0.3× JOINT FREEZE (1.2s)",
            "pct": 86,
            "dial_tag": "JOINT LOAD",
            "kicker": "MISTAKE 1 · LEANING OVER & SHRUGGING",
            "line1": "DON'T HUNCH SHOULDERS",
            "line2": "✕ WHY: BODYWEIGHT PRESSING IMPINGES JOINT",
            "deg_badge": "35° HUNCH",
            "deg_title": "EXCESSIVE SPINE LEAN",
            "deg_sub": "86% ACROMION IMPINGEMENT",
            "joint_pt": (728.0, 618.0),
            "anchor_xy": (88, 520),
            "arc_angles": (-100, -52),
            "splits": [
                ("SHOULDER LOAD", 86, COL_RED),
                ("ELBOW SHEAR", 74, COL_AMBER),
                ("TRICEPS WORK", 18, COL_RED),
            ],
        },
        {
            "idx": 2,
            "is_wrong": True,
            "base_rgb": base_step2,
            "raw_targets": m1_targets,
            "active_slot": 1,
            "cam_zoom": 1.38,
            "cam_focus": (675.0, 725.0),
            "cam_dest": (705.0, 755.0),
            "cam_label": "CAM 2 · 1.4× SHOULDER & DELT ZOOM",
            "speed_label": "1.8× WHIP -> 0.3× HOLD",
            "tempo_str": "1.8× ZOOM-IN  ▸  0.3× DELT STEAL HOLD",
            "pct": 72,
            "dial_tag": "DELT STEAL",
            "kicker": "MISTAKE 2 · STOPPING SHORT AT 90°",
            "line1": "AVOID 90° HALF REPS",
            "line2": "✕ WHY: LIFTING ELBOWS SHIFTS LOAD TO DELTS",
            "deg_badge": "90° STOP",
            "deg_title": "INCOMPLETE LOCKOUT",
            "deg_sub": "72% STOLEN BY FRONT DELT",
            "joint_pt": (602.0, 882.0),
            "anchor_xy": (88, 760),
            "arc_angles": (150, 240),
            "splits": [
                ("FRONT DELTOID", 72, COL_RED),
                ("LATERAL HEAD", 28, COL_AMBER),
                ("LONG HEAD", 19, COL_RED),
            ],
        },
        {
            "idx": 3,
            "is_wrong": False,
            "base_rgb": base_step3,
            "raw_targets": m2_targets,
            "active_slot": 2,
            "cam_zoom": 1.16,
            "cam_focus": (685.0, 745.0),
            "cam_dest": (695.0, 755.0),
            "cam_label": "CAM 3 · 1.2× TORSO & ELBOW HINGE",
            "speed_label": "1.0s PUSH · 3.0s UP",
            "tempo_str": "1.0s EXPLOSIVE PUSH  ▸  3.0s SLOW RETURN",
            "pct": 95,
            "dial_tag": "LAT & LONG",
            "kicker": "PRO TIP 1 · CHEST UP & SCAPULA DOWN",
            "line1": "PIN ELBOWS TO RIBS",
            "line2": "✓ DO: 15° HINGE · LOCK UPPER ARM · FOREARMS ONLY",
            "deg_badge": "0° LOCKED",
            "deg_title": "FIXED ELBOW HINGE",
            "deg_sub": "15° SLIGHT TORSO LEAN",
            "joint_pt": (724.0, 832.0),
            "anchor_xy": (88, 730),
            "arc_angles": (72, 108),
            "splits": [
                ("LATERAL HEAD", 95, COL_GREEN),
                ("LONG HEAD", 91, COL_GREEN),
                ("DELT STEAL", 8, COL_CYAN),
            ],
        },
        {
            "idx": 4,
            "is_wrong": False,
            "base_rgb": base_step4,
            "raw_targets": m3_targets,
            "active_slot": 2,
            "cam_zoom": 1.44,
            "cam_focus": (665.0, 795.0),
            "cam_dest": (700.0, 750.0),
            "cam_label": "CAM 4 · 1.45× LOCKOUT & 3-HEAD MACRO",
            "speed_label": "0.3× PEAK SQUEEZE (1s)",
            "tempo_str": "180° EXTENSION  ▸  0.3× PEAK SQUEEZE (1.0s)  ▸  4×12",
            "pct": 99,
            "dial_tag": "ALL 3 HEADS",
            "kicker": "PRO TIP 2 · FULL EXTENSION & SQUEEZE",
            "line1": "SPREAD ROPE AT BOTTOM",
            "line2": "✓ DO: PULL HANDLES OUT AT 180° · 4 SETS × 12 REPS",
            "deg_badge": "180° FULL",
            "deg_title": "OUTWARD ROPE SPREAD",
            "deg_sub": "99% ALL 3 TRICEPS HEADS",
            "joint_pt": (588.0, 968.0),
            "anchor_xy": (88, 760),
            "arc_angles": (60, 125),
            "splits": [
                ("LATERAL HEAD", 99, COL_GREEN),
                ("LONG HEAD", 95, COL_GREEN),
                ("MEDIAL HEAD", 96, COL_GREEN),
            ],
        },
    ]

    # Render Concept A (4 Multi-Camera Steps)
    concept_A_frames = []
    for sp in steps_spec:
        cam_rgb, map_pt = apply_cinema_camera(sp["base_rgb"], zoom=sp["cam_zoom"], focus_xy=sp["cam_focus"], dest_xy=sp["cam_dest"])
        pil_im = Image.fromarray(cam_rgb).convert("RGBA")
        mapped_targets = [(*map_pt(tx, ty), lbl) for (tx, ty, lbl) in sp["raw_targets"]]
        mapped_joint = map_pt(*sp["joint_pt"])

        draw_top_navigation_dock(pil_im, torso_icons, sp["active_slot"], sp["is_wrong"], sp["cam_label"], sp["speed_label"])
        draw_4x_protractor_and_callout(
            pil_im,
            mapped_joint,
            sp["anchor_xy"],
            sp["deg_badge"],
            sp["deg_title"],
            sp["deg_sub"],
            sp["is_wrong"],
            sp["arc_angles"],
        )
        draw_concept_A_hud(pil_im, mapped_targets, sp)
        out_p = os.path.join(OUT_DIR, f"v27_concept_A_step{sp['idx']}.jpg")
        pil_im.convert("RGB").save(out_p, quality=95)
        concept_A_frames.append(pil_im.convert("RGB"))

    # Render Concept B (Muscle & Motion 2.2x PiP Zoom Lens — Mistake & Fix Hero Frames)
    concept_B_frames = []
    for sp in [steps_spec[1], steps_spec[3]]:
        pil_im = Image.fromarray(sp["base_rgb"]).convert("RGBA")
        draw_top_navigation_dock(pil_im, torso_icons, sp["active_slot"], sp["is_wrong"], "PIP 2.2× MACRO LENS + FULL SHOT", sp["speed_label"])
        draw_concept_B_pip_zoom_lens(pil_im, sp["base_rgb"], sp["raw_targets"][0][:2], sp)
        out_p = os.path.join(OUT_DIR, f"v27_concept_B_step{sp['idx']}.jpg")
        pil_im.convert("RGB").save(out_p, quality=95)
        concept_B_frames.append(pil_im.convert("RGB"))

    # Render Concept C (3D AR Protractor + Live Speed-Ramp & Tempo Waveform HUD)
    concept_C_frames = []
    for sp in [steps_spec[1], steps_spec[3]]:
        cam_rgb, map_pt = apply_cinema_camera(sp["base_rgb"], zoom=sp["cam_zoom"], focus_xy=sp["cam_focus"], dest_xy=sp["cam_dest"])
        pil_im = Image.fromarray(cam_rgb).convert("RGBA")
        mapped_targets = [(*map_pt(tx, ty), lbl) for (tx, ty, lbl) in sp["raw_targets"]]
        mapped_joint = map_pt(*sp["joint_pt"])
        draw_top_navigation_dock(pil_im, torso_icons, sp["active_slot"], sp["is_wrong"], sp["cam_label"], sp["speed_label"])
        draw_4x_protractor_and_callout(
            pil_im,
            mapped_joint,
            (88, 690),
            sp["deg_badge"],
            sp["deg_title"],
            sp["deg_sub"],
            sp["is_wrong"],
            sp["arc_angles"],
        )
        draw_concept_C_ar_speedramp_hud(pil_im, mapped_targets, sp)
        out_p = os.path.join(OUT_DIR, f"v27_concept_C_step{sp['idx']}.jpg")
        pil_im.convert("RGB").save(out_p, quality=95)
        concept_C_frames.append(pil_im.convert("RGB"))

    # Render Concept D (Side-by-Side Split Versus Comparison)
    concept_D_im = build_concept_D_split_versus(base_step2, base_step4, steps_spec[1], steps_spec[3])
    concept_D_p = os.path.join(OUT_DIR, "v27_concept_D_versus.jpg")
    concept_D_im.save(concept_D_p, quality=95)

    # Build 1. Master 4-Concept Comparison Board (`day1_v27_4concepts_master_board.jpg`)
    pw, ph = 420, 746
    board = Image.new("RGB", (pw * 4 + 24 * 5, ph + 118), (10, 12, 16))
    bd = ImageDraw.Draw(board)

    cards = [
        ("CONCEPT A · 다각도 카메라 + 일체형 다이얼 HUD", "4단계 카메라 줌 전환 + 무박스 대형 각도 + 3헤드 미니바", concept_A_frames[3], COL_GREEN),
        ("CONCEPT B · 2.2배 돋보기 렌즈 (Muscle&Motion형)", "전신 자세 + 2.2배 관절/근육 확대 렌즈 동시 비교", concept_B_frames[1], COL_CYAN),
        ("CONCEPT C · 3D AR 각도기 + 스피드램프 파형 HUD", "관절 AR 각도기 + 실시간 속도감(0.3x 슬로우) 파형 바", concept_C_frames[1], COL_AMBER),
        ("CONCEPT D · ✕ vs ✓ 좌우 분할 직관 비교형", "틀린 자세(72% 삼각근) vs 정자세(99% 삼두) 1:1 동시 대조", concept_D_im, COL_RED),
    ]

    for i, (t_main, t_sub, im, col) in enumerate(cards):
        x = 24 + i * (pw + 24)
        bd.text((x + 4, 18), t_main, font=eng.load_font(18, bold=True), fill=col)
        bd.text((x + 4, 46), t_sub, font=eng.load_font(13, bold=True), fill=(200, 212, 230))
        r = im.resize((pw, ph), Image.Resampling.LANCZOS)
        ImageDraw.Draw(r).rectangle([0, 0, pw - 1, ph - 1], outline=col, width=4)
        board.paste(r, (x, 82))

    board_p = os.path.join(OUT_DIR, "day1_v27_4concepts_master_board.jpg")
    board.save(board_p, quality=95)
    print("Saved 4-Concept Master Board:", board_p)

    # Build 2. Multi-Camera Transition & Speed-Ramping Storyboard (`day1_v27_camera_speedramp_board.jpg`)
    sw, sh = 380, 675
    cam_board = Image.new("RGB", (sw * 4 + 24 * 5, sh + 260), (10, 12, 16))
    cbd = ImageDraw.Draw(cam_board)

    cbd.text((28, 18), "GYMFORM 3D · MULTI-CAMERA FRAMING & SPEED-RAMPING ARCHITECTURE (다각도 카메라 전환 & 속도감 설계)", font=eng.load_font(22, bold=True), fill=(255, 255, 255))
    cbd.text((28, 50), "인기 릴스(@demicstory · Muscle & Motion) 기법: 컷 사이에는 1.8× 빠른 스무스 줌/윕팬 전환으로 몰입감을 주고, 핵심 관절·수축 피크에서는 0.3× 슬로우 모션 홀드로 시선을 고정합니다.", font=eng.load_font(15, bold=True), fill=(185, 200, 222))

    cam_headers = [
        ("STEP 1 · CAM 1 (1.0× WIDE)", "전신 측면 · 35° 굽은 등 & 어깨 충돌", COL_RED),
        ("STEP 2 · CAM 2 (1.4× ZOOM-IN)", "상체 클로즈업 · 90° 하프렙 & 전면삼각근 개입", COL_RED),
        ("STEP 3 · CAM 3 (1.2× MID SHOT)", "정자세 셋업 · 15° 힌지 & 0° 팔꿈치 고정축", COL_GREEN),
        ("STEP 4 · CAM 4 (1.45× MACRO)", "삼두근 3갈래 & 180° 로프 스프레드 클로즈업", COL_GREEN),
    ]

    for i in range(4):
        x = 24 + i * (sw + 24)
        h1, h2, col = cam_headers[i]
        cbd.text((x + 4, 88), h1, font=eng.load_font(17, bold=True), fill=col)
        cbd.text((x + 4, 112), h2, font=eng.load_font(13, bold=True), fill=(210, 222, 240))
        r = concept_A_frames[i].resize((sw, sh), Image.Resampling.LANCZOS)
        ImageDraw.Draw(r).rectangle([0, 0, sw - 1, sh - 1], outline=col, width=4)
        cam_board.paste(r, (x, 140))

    # Draw continuous Speed-Ramping Timeline Graph at bottom (`y = 832..985`)
    gx1, gy1, gx2, gy2 = 24, 832, sw * 4 + 24 * 4, 988
    cbd.rounded_rectangle([gx1, gy1, gx2, gy2], radius=20, fill=(16, 20, 28), outline=(70, 82, 102), width=2)
    cbd.text((gx1 + 20, gy1 + 12), "DYNAMIC SPEED-RAMP TIMELINE (속도감 조절 곡선 · 총 15.0초)", font=eng.load_font(16, bold=True), fill=COL_CYAN)

    # Plot velocity curve across the 4 steps
    curve_pts = []
    w_graph = gx2 - gx1 - 60
    for k in range(200):
        t = k / 199.0
        px = int(gx1 + 30 + t * w_graph)
        # 4 transitions with 1.8x spikes at t=0.02, 0.25, 0.50, 0.75 and 0.3x slow-mo valleys at t=0.14, 0.38, 0.63, 0.88
        spike = (
            math.exp(-((t - 0.03) ** 2) / 0.0008)
            + math.exp(-((t - 0.26) ** 2) / 0.0008)
            + math.exp(-((t - 0.51) ** 2) / 0.0008)
            + math.exp(-((t - 0.76) ** 2) / 0.0008)
        )
        valley = (
            math.exp(-((t - 0.14) ** 2) / 0.0025)
            + math.exp(-((t - 0.38) ** 2) / 0.0025)
            + math.exp(-((t - 0.63) ** 2) / 0.0025)
            + math.exp(-((t - 0.88) ** 2) / 0.0025)
        )
        py = int((gy1 + 92) - spike * 42 + valley * 36)
        curve_pts.append((px, py))

    cbd.line([(gx1 + 30, gy1 + 92), (gx2 - 30, gy1 + 92)], fill=(55, 65, 82), width=2)
    cbd.line(curve_pts, fill=COL_CYAN, width=4, joint="curve")

    # Annotate spikes & slow-mo holds on timeline
    for idx_s, t_spike in enumerate([0.26, 0.51, 0.76]):
        sx = int(gx1 + 30 + t_spike * w_graph)
        cbd.ellipse([sx - 6, gy1 + 44, sx + 6, gy1 + 56], fill=(255, 255, 255), outline=COL_CYAN, width=3)
        cbd.text((sx - 68, gy1 + 24), "⚡ 1.8× WHIP-ZOOM 전환", font=eng.load_font(12, bold=True), fill=(255, 255, 255))

    for idx_v, (t_val, lbl_v, col_v) in enumerate([
        (0.14, "0.3× 슬로우 홀드 (어깨 충돌 강조)", COL_RED),
        (0.38, "0.3× 슬로우 홀드 (삼각근 개입 강조)", COL_RED),
        (0.63, "0.3× 슬로우 홀드 (팔꿈치 0° 고정)", COL_GREEN),
        (0.88, "0.3× 피크 수축 홀드 (삼두 3헤드 99%)", COL_GREEN),
    ]):
        vx = int(gx1 + 30 + t_val * w_graph)
        cbd.ellipse([vx - 6, gy1 + 122, vx + 6, gy1 + 134], fill=col_v, outline=(255, 255, 255), width=2)
        cbd.text((vx - 110, gy1 + 136), lbl_v, font=eng.load_font(12, bold=True), fill=col_v)

    cam_board_p = os.path.join(OUT_DIR, "day1_v27_camera_speedramp_board.jpg")
    cam_board.save(cam_board_p, quality=95)
    print("Saved Camera & Speed-Ramp Architecture Board:", cam_board_p)


if __name__ == "__main__":
    main()
