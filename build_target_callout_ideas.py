#!/usr/bin/env python3
"""
Generates a 4-Option Visual Brainstorming Sheet (`target_callout_ideas_board.jpg`)
comparing 4 distinct ways to point out the exact anatomical target / overloaded body part
on @gymform.3d's 3D mannequin (with ZERO arbitrary degree `°` numbers).
Each option shows both:
- Top half / Left: Step 2 (Red — Wrong Posture: Lumbar Spine & Patellar Tendon Overload)
- Bottom half / Right: Step 4 (Green — Correct Posture: Quadriceps & Gluteus Maximus Target Hit)
"""
import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import build_v38_high_contrast_hud as v38
import build_day3_v39_squat as d3

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920

COL_RED = (235, 42, 48)
COL_RED_SOFT = (255, 145, 140)
COL_GREEN = (16, 205, 110)
COL_GREEN_SOFT = (155, 250, 198)
COL_WHITE = (255, 255, 255)


def draw_bezier_curve_arrow(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int = 8,
    head_len: float = 28.0,
    ss: int = 2,
):
    """
    Draws a smooth quadratic Bezier curve from `p0` (under the label) through control point `p1`
    to `p2` (the arrow tip pointing at the muscle/joint), plus a clean open-V or solid arrowhead at `p2`
    matching the reference image style.
    """
    pts = []
    steps = 48
    for i in range(steps + 1):
        t = i / float(steps)
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((x * ss, y * ss))

    draw.line(pts, fill=color, width=width * ss, joint="curve")

    # Tangent direction at t=1 (from p1 to p2)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = math.atan2(dy, dx)
    hl = head_len * ss
    spread = math.radians(32.0)

    tip = (p2[0] * ss, p2[1] * ss)
    w1 = (tip[0] - hl * math.cos(ang - spread), tip[1] - hl * math.sin(ang - spread))
    w2 = (tip[0] - hl * math.cos(ang + spread), tip[1] - hl * math.sin(ang + spread))

    # Crisp V-shaped arrowhead like the user's reference image
    draw.line([w1, tip, w2], fill=color, width=width * ss, joint="curve")


def draw_text_with_shadow(
    od: ImageDraw.ImageDraw,
    sd: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill=(255, 255, 255, 255),
    align: str = "left",
):
    tb = od.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    x, y = xy
    if align == "right":
        x = x - tw
    elif align == "center":
        x = x - tw // 2
    sd.text((x, y + 4), text, font=font, fill=(0, 0, 0, 245))
    od.text((x, y), text, font=font, fill=fill)
    return (x, y, x + tw, y + th)


def render_option_frame(base_rgb: np.ndarray, kp: dict, is_wrong: bool, option_id: int) -> Image.Image:
    """
    Renders one 1080x1920 frame demonstrating Option 1, 2, 3, or 4 on the 3D Squat mannequin.
    """
    cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=1.04, focus_xy=(575.0, 965.0), dest_xy=(585.0, 965.0))
    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]
    scrim_2d = v38.make_bottom_scrim_2d()
    clean_rgb = v38.apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)

    canvas = Image.fromarray(clean_rgb).convert("RGBA")
    SS = 2
    rig = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig)
    rig_sh = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rsd = ImageDraw.Draw(rig_sh)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    col_main = COL_RED if is_wrong else COL_GREEN
    col_soft = COL_RED_SOFT if is_wrong else COL_GREEN_SOFT

    f_big = v30.f_pretendard(40, "extrabold")
    f_med = v30.f_pretendard(32, "extrabold")
    f_sub = v30.f_pretendard(18, "extrabold")
    f_pill = v30.f_pretendard(16, "extrabold")

    # Anatomical target points in mapped screen coords
    if is_wrong:
        # Step 2 Wrong (Lumbar spine + Anterior knee/patella)
        pt_left_target = map_pt(kp["hip"][0] + 18.0, kp["hip"][1] - 75.0)    # Lower back / Lumbar L4-L5
        pt_right_target = map_pt(kp["kn_r"][0] + 12.0, kp["kn_r"][1] - 5.0)   # Anterior Knee / Patellar tendon
        lbl_left_main = "LOWER BACK"
        lbl_left_sub = "L4-L5 DISC STRAIN"
        lbl_left_pill = "OVERLOAD"
        lbl_right_main = "KNEE JOINT"
        lbl_right_sub = "PATELLAR SHEAR"
        lbl_right_pill = "SHEAR LOAD"
    else:
        # Step 4 Correct (Gluteus Maximus + Quadriceps belly)
        pt_left_target = map_pt(kp["hip"][0] - 18.0, kp["hip"][1] + 15.0)     # Gluteus Maximus belly
        pt_right_target = map_pt((kp["hip"][0] + kp["kn_r"][0]) * 0.5 + 22.0, (kp["hip"][1] + kp["kn_r"][1]) * 0.5 - 15.0)  # Quadriceps belly
        lbl_left_main = "GLUTE MAX"
        lbl_left_sub = "HIP EXTENSION DRIVE"
        lbl_left_pill = "PRIMARY HIT"
        lbl_right_main = "QUADRICEPS"
        lbl_right_sub = "ANTERIOR THIGH SWEEP"
        lbl_right_pill = "PRIMARY HIT"

    # OPTION 1: Pure Reference Style — Bold White Text + Smooth Curved Arrow (`Red Se` style)
    if option_id == 1:
        # Left Target (Glute Max or Lower Back)
        lx, ly = 115, 760
        bbox_l = draw_text_with_shadow(od, sd, (lx, ly), lbl_left_main, f_big, fill=(255, 255, 255, 255), align="left")
        draw_text_with_shadow(od, sd, (lx + 2, bbox_l[3] + 6), lbl_left_sub, f_sub, fill=(*col_soft, 255), align="left")
        # Underline bar under text + curved white arrow to anatomical target
        p0_l = (lx + 85, bbox_l[3] + 38)
        p2_l = (pt_left_target[0] - 12, pt_left_target[1] - 6)
        p1_l = ((p0_l[0] + p2_l[0]) * 0.45 - 20, (p0_l[1] + p2_l[1]) * 0.55 + 35)
        draw_bezier_curve_arrow(rsd, (p0_l[0], p0_l[1] + 4), (p1_l[0], p1_l[1] + 4), (p2_l[0], p2_l[1] + 4), (0, 0, 0, 220), width=6, head_len=24, ss=SS)
        draw_bezier_curve_arrow(rd, p0_l, p1_l, p2_l, (255, 255, 255, 250), width=5, head_len=24, ss=SS)

        # Right Target (Quadriceps or Knee Joint)
        rx, ry = 955, 790
        bbox_r = draw_text_with_shadow(od, sd, (rx, ry), lbl_right_main, f_big, fill=(255, 255, 255, 255), align="right")
        draw_text_with_shadow(od, sd, (rx, bbox_r[3] + 6), lbl_right_sub, f_sub, fill=(*col_soft, 255), align="right")
        p0_r = (rx - 95, bbox_r[3] + 38)
        p2_r = (pt_right_target[0] + 14, pt_right_target[1] - 8)
        p1_r = ((p0_r[0] + p2_r[0]) * 0.55 + 25, (p0_r[1] + p2_r[1]) * 0.48 - 25)
        draw_bezier_curve_arrow(rsd, (p0_r[0], p0_r[1] + 4), (p1_r[0], p1_r[1] + 4), (p2_r[0], p2_r[1] + 4), (0, 0, 0, 220), width=6, head_len=24, ss=SS)
        draw_bezier_curve_arrow(rd, p0_r, p1_r, p2_r, (255, 255, 255, 250), width=5, head_len=24, ss=SS)

    # OPTION 2: Hybrid Color-Coded Curved Arrow + Pill Tag + Target Pulse Ring (`╭──→ ◎`)
    elif option_id == 2:
        # Left Target
        lx, ly = 115, 745
        tb_p = od.textbbox((0, 0), lbl_left_pill, font=f_pill)
        pw_l = (tb_p[2] - tb_p[0]) + 20
        od.rounded_rectangle([lx, ly, lx + pw_l, ly + 28], radius=7, fill=(*col_main, 248))
        od.text((lx + 10, ly + 5), lbl_left_pill, font=f_pill, fill=(255, 255, 255, 255))
        bbox_l = draw_text_with_shadow(od, sd, (lx, ly + 34), lbl_left_main, f_big, fill=(255, 255, 255, 255), align="left")
        draw_text_with_shadow(od, sd, (lx + 2, bbox_l[3] + 5), lbl_left_sub, f_sub, fill=(*col_soft, 255), align="left")

        p0_l = (lx + 90, bbox_l[3] + 36)
        p2_l = (pt_left_target[0] - 16, pt_left_target[1] - 4)
        p1_l = ((p0_l[0] + p2_l[0]) * 0.45 - 15, (p0_l[1] + p2_l[1]) * 0.58 + 30)
        draw_bezier_curve_arrow(rsd, (p0_l[0], p0_l[1] + 4), (p1_l[0], p1_l[1] + 4), (p2_l[0], p2_l[1] + 4), (0, 0, 0, 225), width=6, head_len=24, ss=SS)
        draw_bezier_curve_arrow(rd, p0_l, p1_l, p2_l, (*col_main, 255), width=5, head_len=24, ss=SS)
        # Glowing pulse ring at target
        tx_s, ty_s = pt_left_target[0] * SS, pt_left_target[1] * SS
        rd.ellipse([tx_s - 22, ty_s - 22, tx_s + 22, ty_s + 22], outline=(*col_soft, 235), width=3 * SS)
        rd.ellipse([tx_s - 8, ty_s - 8, tx_s + 8, ty_s + 8], fill=(255, 255, 255, 255))

        # Right Target
        rx, ry = 955, 765
        tb_pr = od.textbbox((0, 0), lbl_right_pill, font=f_pill)
        pw_r = (tb_pr[2] - tb_pr[0]) + 20
        od.rounded_rectangle([rx - pw_r, ry, rx, ry + 28], radius=7, fill=(*col_main, 248))
        od.text((rx - pw_r + 10, ry + 5), lbl_right_pill, font=f_pill, fill=(255, 255, 255, 255))
        bbox_r = draw_text_with_shadow(od, sd, (rx, ry + 34), lbl_right_main, f_big, fill=(255, 255, 255, 255), align="right")
        draw_text_with_shadow(od, sd, (rx, bbox_r[3] + 5), lbl_right_sub, f_sub, fill=(*col_soft, 255), align="right")

        p0_r = (rx - 95, bbox_r[3] + 36)
        p2_r = (pt_right_target[0] + 18, pt_right_target[1] - 6)
        p1_r = ((p0_r[0] + p2_r[0]) * 0.55 + 20, (p0_r[1] + p2_r[1]) * 0.48 - 25)
        draw_bezier_curve_arrow(rsd, (p0_r[0], p0_r[1] + 4), (p1_r[0], p1_r[1] + 4), (p2_r[0], p2_r[1] + 4), (0, 0, 0, 225), width=6, head_len=24, ss=SS)
        draw_bezier_curve_arrow(rd, p0_r, p1_r, p2_r, (*col_main, 255), width=5, head_len=24, ss=SS)
        rx_s, ry_s = pt_right_target[0] * SS, pt_right_target[1] * SS
        rd.ellipse([rx_s - 22, ry_s - 22, rx_s + 22, ry_s + 22], outline=(*col_soft, 235), width=3 * SS)
        rd.ellipse([rx_s - 8, ry_s - 8, rx_s + 8, ry_s + 8], fill=(255, 255, 255, 255))

    # OPTION 3: Anatomical Encyclopedia Pin & Elbow Leader Line (`◎──╱─── QUADRICEPS`)
    elif option_id == 3:
        # Left Target Pin
        tx_l, ty_l = pt_left_target
        elbow_l = (tx_l - 85, ty_l - 65)
        end_l = (115, ty_l - 65)
        pts_l = [(tx_l * SS, ty_l * SS), (elbow_l[0] * SS, elbow_l[1] * SS), (end_l[0] * SS, end_l[1] * SS)]
        rsd.line([(p[0], p[1] + 4) for p in pts_l], fill=(0, 0, 0, 220), width=4 * SS)
        rd.line(pts_l, fill=(*col_main, 255), width=3 * SS)
        rd.ellipse([tx_l * SS - 20, ty_l * SS - 20, tx_l * SS + 20, ty_l * SS + 20], fill=(*col_main, 90), outline=(*col_soft, 255), width=3 * SS)
        rd.ellipse([tx_l * SS - 7, ty_l * SS - 7, tx_l * SS + 7, ty_l * SS + 7], fill=(255, 255, 255, 255))
        draw_text_with_shadow(od, sd, (115, int(elbow_l[1]) - 46), lbl_left_main, f_med, fill=(255, 255, 255, 255), align="left")
        draw_text_with_shadow(od, sd, (115, int(elbow_l[1]) + 8), lbl_left_sub, f_sub, fill=(*col_soft, 255), align="left")

        # Right Target Pin
        tx_r, ty_r = pt_right_target
        elbow_r = (tx_r + 80, min(890, ty_r - 75))
        end_r = (955, elbow_r[1])
        pts_r = [(tx_r * SS, ty_r * SS), (elbow_r[0] * SS, elbow_r[1] * SS), (end_r[0] * SS, end_r[1] * SS)]
        rsd.line([(p[0], p[1] + 4) for p in pts_r], fill=(0, 0, 0, 220), width=4 * SS)
        rd.line(pts_r, fill=(*col_main, 255), width=3 * SS)
        rd.ellipse([tx_r * SS - 20, ty_r * SS - 20, tx_r * SS + 20, ty_r * SS + 20], fill=(*col_main, 90), outline=(*col_soft, 255), width=3 * SS)
        rd.ellipse([tx_r * SS - 7, ty_r * SS - 7, tx_r * SS + 7, ty_r * SS + 7], fill=(255, 255, 255, 255))
        draw_text_with_shadow(od, sd, (955, int(elbow_r[1]) - 46), lbl_right_main, f_med, fill=(255, 255, 255, 255), align="right")
        draw_text_with_shadow(od, sd, (955, int(elbow_r[1]) + 8), lbl_right_sub, f_sub, fill=(*col_soft, 255), align="right")

    # OPTION 4: Glass Capsule Card + Curved Arrow Pointer
    else:
        for side, (tx, ty), pill_txt, main_txt, sub_txt in [
            ("left", pt_left_target, lbl_left_pill, lbl_left_main, lbl_left_sub),
            ("right", pt_right_target, lbl_right_pill, lbl_right_main, lbl_right_sub),
        ]:
            card_w, card_h = 270, 92
            if side == "left":
                cx0, cy0 = 105, 760
                p0 = (cx0 + card_w * 0.55, cy0 + card_h + 4)
                p2 = (tx - 14, ty - 4)
                p1 = ((p0[0] + p2[0]) * 0.45, (p0[1] + p2[1]) * 0.58 + 25)
            else:
                cx0, cy0 = 965 - card_w, 770
                p0 = (cx0 + card_w * 0.45, cy0 + card_h + 4)
                p2 = (tx + 14, ty - 4)
                p1 = ((p0[0] + p2[0]) * 0.55, (p0[1] + p2[1]) * 0.48 - 20)

            sd.rounded_rectangle([cx0 + 2, cy0 + 5, cx0 + card_w + 2, cy0 + card_h + 5], radius=16, fill=(0, 0, 0, 185))
            od.rounded_rectangle([cx0, cy0, cx0 + card_w, cy0 + card_h], radius=16, fill=(14, 18, 24, 228), outline=(*col_main, 220), width=2)
            od.rounded_rectangle([cx0 + 12, cy0 + 14, cx0 + 18, cy0 + card_h - 14], radius=3, fill=(*col_main, 255))
            od.text((cx0 + 28, cy0 + 14), main_txt, font=f_med, fill=(255, 255, 255, 255))
            od.text((cx0 + 28, cy0 + 54), sub_txt, font=f_sub, fill=(*col_soft, 255))

            draw_bezier_curve_arrow(rsd, (p0[0], p0[1] + 4), (p1[0], p1[1] + 4), (p2[0], p2[1] + 4), (0, 0, 0, 220), width=5, head_len=22, ss=SS)
            draw_bezier_curve_arrow(rd, p0, p1, p2, (*col_soft, 255), width=4, head_len=22, ss=SS)

    canvas.alpha_composite(rig_sh.resize((W, H), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=5)))
    canvas.alpha_composite(rig.resize((W, H), Image.Resampling.LANCZOS))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=6)))
    canvas.alpha_composite(ov)

    # Also draw clean bottom HUD without angle degrees
    d3.draw_day3_v39_safezone_bottom_hud(
        canvas,
        {
            "is_wrong": is_wrong,
            "pct": 79 if is_wrong else 98,
            "tag_pill": "SPINE & KNEE LOAD" if is_wrong else "PRIMARY TARGET · 4×10",
            "tag_ctx": "LUMBAR & PATELLA" if is_wrong else "QUADRICEPS & GLUTES",
            "cue_1line": "HEELS LIFTING UP" if is_wrong else "DEEP SQUAT DRIVE",
        },
    )
    return canvas.convert("RGB")


def main():
    raw_s2 = d3._load_clip_frames(d3.DAY3_STEP2_WRONG_I2V)
    raw_s4 = d3._load_clip_frames(d3.DAY3_STEP4_CORRECT_I2V)

    rgb_w, kp_w = d3.render_day3_i2v_frame_v39(raw_s2[115], 2, True)
    rgb_c, kp_c = d3.render_day3_i2v_frame_v39(raw_s4[140], 4, False)

    cw, ch = 440, 782
    pad = 24
    board = Image.new("RGB", (cw * 4 + pad * 5, ch * 2 + 190), (10, 12, 16))
    bd = ImageDraw.Draw(board)

    f_title = v30.f_pretendard(24, "extrabold")
    f_col = v30.f_pretendard(17, "extrabold")
    f_desc = v30.f_pretendard(14, "bold")

    bd.text(
        (28, 20),
        "@gymform.3d 타겟 부위(근육/관절) 지시 디자인 4대 아이디어 비교 (각도 ° 표기 0% · 아이폰 세이프존 준수)",
        font=f_title,
        fill=COL_GREEN,
    )

    headers = [
        ("OPTION A · 레퍼런스 순수 곡선 화살표", "볼드 화이트 타이포 + 부드러운 곡선 화살표 (╭─→)"),
        ("OPTION B · 컬러 곡선 화살표 + 타겟 링", "상단 상태 캡슐 + 컬러 곡선 화살표 + 타겟 포인트(◎)"),
        ("OPTION C · 해부학 도감 핀 & 꺾임 리더선", "Muscle & Motion 스타일 타겟 핀(◎) + 직선 지시선"),
        ("OPTION D · 다크 글래스 캡슐 + 곡선 포인터", "반투명 다크 카드 배지 + 곡선 화살표 결합형"),
    ]

    for i in range(4):
        opt_id = i + 1
        x = pad + i * (cw + pad)
        bd.text((x + 4, 62), headers[i][0], font=f_col, fill=(255, 255, 255))
        bd.text((x + 4, 86), headers[i][1], font=f_desc, fill=(180, 195, 215))

        im_c = render_option_frame(rgb_c, kp_c, False, opt_id).resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(im_c).rectangle([0, 0, cw - 1, ch - 1], outline=COL_GREEN, width=2)
        board.paste(im_c, (x, 114))

        im_w = render_option_frame(rgb_w, kp_w, True, opt_id).resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(im_w).rectangle([0, 0, cw - 1, ch - 1], outline=COL_RED, width=2)
        board.paste(im_w, (x, 114 + ch + 28))

    out_path = os.path.join(OUT_DIR, "target_callout_ideas_board.jpg")
    board.save(out_path, quality=95)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
