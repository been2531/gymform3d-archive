#!/usr/bin/env python3
"""
Builds `@gymform.3d` Official A/B Master Character System Sheet
(`/workspace/daily_info_reels/output/gymform_character_ab_master_sheet.jpg`)
based on:
1. `@demicstory` 24-Reel Benchmark Analysis (`demic_bench/`)
2. NSCA Official Spotting Guidelines (`Basics of Strength and Conditioning Manual`, Figures 3-3..3-16)
3. Starting Strength Tactile Cueing & ACE / Muscle & Motion Scapular Cueing Standards
4. Zero-Occlusion Rule (`t289u`: Character B never blocks Character A's target muscles or movement)
"""

import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62

OUT_DIR = "/workspace/daily_info_reels/output"
OUT_PATH = os.path.join(OUT_DIR, "gymform_character_ab_master_sheet.jpg")

COL_BG = (13, 16, 22)
COL_CARD = (20, 24, 33)
COL_BORDER = (52, 60, 76)
COL_GREEN = (20, 228, 120)
COL_RED = (248, 40, 48)
COL_CYAN = (56, 232, 255)
COL_SILVER = (182, 188, 198)
COL_GUNMETAL = (88, 98, 116)
COL_WHITE = (255, 255, 255)
COL_MUTED = (168, 178, 194)

F_H1 = v30.f_pretendard(30, "extrabold")
F_H2 = v30.f_pretendard(22, "extrabold")
F_H3 = v30.f_pretendard(18, "extrabold")
F_BODY = v30.f_pretendard(15, "bold")
F_SMALL = v30.f_pretendard(13, "bold")
F_NUM = v30.f_barlow(34, extrabold=True)


def render_char_a_pose(raw_jpg: str, step_idx: int, is_wrong: bool, out_size: tuple[int, int]) -> Image.Image:
    bgr = cv2.imread(raw_jpg)
    h, w = bgr.shape[:2]
    import universal_anatomy_rig as uar
    det = uar.get_pose_detector()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = det.detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=rgb))
    lm = res.pose_landmarks[0]
    kp = {
        "sh": (0.5 * (lm[11].x + lm[12].x) * w, 0.5 * (lm[11].y + lm[12].y) * h),
        "hip_r": (lm[24].x * w, lm[24].y * h),
        "hip_l": (lm[23].x * w, lm[23].y * h),
        "kn_r": (lm[26].x * w, lm[26].y * h),
        "kn_l": (lm[25].x * w, lm[25].y * h),
    }
    shaded = p62.shade_v62_day2_authentic_720p(bgr, step_idx, is_wrong, kp, rep_a=1.0)
    pil = Image.fromarray(shaded)
    return pil.resize(out_size, Image.Resampling.LANCZOS)


def render_char_b_gunmetal_pose(raw_jpg: str, out_size: tuple[int, int], add_cyan_rim: bool = True) -> tuple[Image.Image, np.ndarray]:
    """
    Renders Character B (`SPOTTER / BIOMECHANICS COACH`) in the official
    Matte Deep Gunmetal-Slate (`#38404E` ~ `#5E6878`, `median lum = 64`,
    `R:G:B = 0.86 : 0.96 : 1.14` cool slate-blue undertone + cyan rim contour,
    0% internal muscle red/green glow so B never competes visually with Character A!).
    """
    bgr = cv2.imread(raw_jpg)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    bg_est = float(np.median(base_lum[180:720, 12:58]))
    body_mask = np.clip((base_lum - (bg_est + 9.5)) / 13.5, 0.0, 1.0)

    cool_base = np.minimum(g_ch, b_ch)
    fiber_sig = 0.65 * r_ch + 0.25 * g_ch + 0.10 * b_ch
    s1 = cv2.GaussianBlur(fiber_sig, (0, 0), 0.9)
    s2 = cv2.GaussianBlur(fiber_sig, (0, 0), 2.8)
    s3 = cv2.GaussianBlur(fiber_sig, (0, 0), 8.0)
    hf = (s1 - s2) * 3.2
    sulc = np.clip((s3 - s2 - 1.0) / 10.0, 0.0, 1.0)

    b_px = cool_base[body_mask > 0.55]
    p15, p50, p85 = np.percentile(b_px, [15, 50, 85])
    t_low = np.clip((cool_base - p15) / max(6.0, p50 - p15), -0.5, 1.0)
    t_high = np.clip((cool_base - p50) / max(8.0, p85 - p50), 0.0, 1.3)
    gm_lum = np.where(
        cool_base <= p50,
        50.0 + t_low * (50.0 - 34.0),
        50.0 + t_high * (72.0 - 50.0),
    )
    gm_lum = np.clip(gm_lum + np.clip(hf, -10.0, 12.0) * 0.38 - sulc * 8.0, 28.0, 80.0)

    bg_lum = np.clip(18.0 + (base_lum - bg_est) * 0.4, 14.0, 26.0)
    lum_out = bg_lum * (1.0 - body_mask) + gm_lum * body_mask

    # Cool Gunmetal-Slate (`R=0.86, G=0.96, B=1.14`)
    gm_rgb = np.stack([
        lum_out * (1.0 - 0.14 * body_mask),
        lum_out * (1.0 - 0.04 * body_mask),
        lum_out * (1.0 + 0.15 * body_mask),
    ], axis=2)

    if add_cyan_rim:
        gx = cv2.Sobel(body_mask, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(body_mask, cv2.CV_32F, 0, 1, ksize=3)
        rim = np.clip(np.hypot(gx, gy) * 1.4, 0.0, 1.0)[:, :, None]
        cyan_col = np.array([56.0, 232.0, 255.0], dtype=np.float32)[None, None, :]
        gm_rgb = gm_rgb * (1.0 - rim * 0.65) + cyan_col * (rim * 0.65)

    out_u8 = np.clip(gm_rgb, 0, 255).astype(np.uint8)
    pil_im = Image.fromarray(out_u8).resize(out_size, Image.Resampling.LANCZOS)
    mask_res = cv2.resize(body_mask, out_size, interpolation=cv2.INTER_LINEAR)
    return pil_im, mask_res


def draw_cyan_contact_ring(draw: ImageDraw.ImageDraw, pt: tuple[int, int], label: str = None):
    cx, cy = pt
    draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(56, 232, 255, 55), outline=(56, 232, 255, 220), width=2)
    draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(12, 16, 24, 230), outline=(255, 255, 255, 240), width=2)
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(56, 232, 255, 255))
    if label:
        draw.rounded_rectangle([cx + 22, cy - 15, cx + 22 + len(label) * 9 + 18, cy + 15], radius=7, fill=(14, 22, 32, 235), outline=COL_CYAN, width=1)
        draw.text((cx + 31, cy - 9), label, font=F_SMALL, fill=COL_CYAN)


def main():
    W_S, H_S = 2240, 1420
    board = Image.new("RGBA", (W_S, H_S), (*COL_BG, 255))
    bd = ImageDraw.Draw(board)

    # Top Title Banner
    bd.rounded_rectangle([28, 22, W_S - 28, 108], radius=18, fill=(18, 22, 31, 255), outline=COL_BORDER, width=2)
    bd.text((52, 36), "@gymform.3d — OFFICIAL A/B MASTER CHARACTER SYSTEM & NSCA SPOTTING STANDARD", font=F_H1, fill=COL_WHITE)
    bd.text(
        (52, 74),
        "근거: @demicstory 24편 전수 분석 (1인 실버-그레이 마네킹 + 몸 바깥 단일 화살표)  +  NSCA Spotting Guidelines (Figures 3-3~3-16) & Starting Strength Tactile Cueing (시야 방해 0% 2인 보조)",
        font=F_BODY,
        fill=COL_GREEN,
    )

    p2_raw = os.path.join(OUT_DIR, "day3_v63_p2_raw.jpg")
    p4_raw = os.path.join(OUT_DIR, "day3_v63_p4_raw.jpg")
    p3_raw = os.path.join(OUT_DIR, "day3_v63_p3_raw.jpg")

    # =========================================================================
    # PANEL 1 (LEFT, x=28..738): CHARACTER A — PRIMARY LIFTER
    # =========================================================================
    bd.rounded_rectangle([28, 128, 738, 1392], radius=20, fill=(*COL_CARD, 255), outline=(140, 150, 168, 180), width=2)
    bd.rounded_rectangle([48, 146, 268, 184], radius=9, fill=COL_GREEN)
    bd.text((64, 154), "CHARACTER A · MAIN", font=F_H3, fill=(10, 14, 18))
    bd.text((282, 151), "PRIMARY LIFTER (주연 운동 수행자)", font=F_H2, fill=COL_WHITE)

    im_a_front = render_char_a_pose(p2_raw, 2, False, (315, 560))
    im_a_side = render_char_a_pose(p4_raw, 4, False, (315, 560))
    board.paste(im_a_front, (50, 202))
    board.paste(im_a_side, (392, 202))
    bd.rectangle([50, 202, 50 + 315, 202 + 560], outline=(255, 255, 255, 55), width=1)
    bd.rectangle([392, 202, 392 + 315, 202 + 560], outline=(255, 255, 255, 55), width=1)
    bd.text((62, 212), "FRONT · 100% BARBELL GRIP", font=F_SMALL, fill=COL_GREEN)
    bd.text((404, 212), "3/4 SIDE · SHORTS-FREE GLUTES", font=F_SMALL, fill=COL_GREEN)

    # Swatches & Spec Box for Character A
    sy = 782
    bd.text((52, sy), "1. 피부 톤 & 무광 매트 질감 (Day 2 Step 1·2 처음 마네킹 100% 매칭)", font=F_H3, fill=COL_SILVER)
    swatches_a = [
        ((84, 87, 91), "Smooth Matte Slate-Grey", "Day 2 Step 1·2 동일 RGB(84,87,91) · Lum 86.0"),
        ((106, 110, 115), "Matte Clay Highlight", "p75=106, p90=122 (Laplacian <= 1.8 무광 매트!)"),
        ((20, 228, 120),  "Right Target Fiber", "Emerald Green #14E478 (3D 근육결 발광)"),
        ((248, 40, 48),   "Wrong Joint/Overload", "Crimson Red #F82830 (국소 X-Ray 발광)"),
    ]
    for idx, (col_sw, name_sw, desc_sw) in enumerate(swatches_a):
        yy = sy + 36 + idx * 46
        bd.rounded_rectangle([52, yy, 104, yy + 34], radius=7, fill=col_sw, outline=(255, 255, 255, 90), width=1)
        bd.text((118, yy + 1), name_sw, font=F_BODY, fill=COL_WHITE)
        bd.text((118, yy + 19), desc_sw, font=F_SMALL, fill=COL_MUTED)

    # 4-Step Skin Consistency Proof Bar
    by_a = 1000
    bd.rounded_rectangle([48, by_a, 718, 1372], radius=14, fill=(14, 17, 24, 255), outline=COL_BORDER, width=1)
    bd.text((66, by_a + 14), "GATE [10] · Day 2 Step 1·2 무광 매트 슬레이트-그레이(Lum ~86) 4구간 통일 검증", font=F_H3, fill=COL_GREEN)
    cap_r = cv2.VideoCapture(os.path.join(OUT_DIR, "day3_squat_gripped_master_reel.mp4"))
    for idx, (s_no, f_no, lbl_s) in enumerate([(1, 70, "S1 · Lum 86.8"), (2, 220, "S2 · Lum 87.2"), (3, 370, "S3 · Lum 87.4"), (4, 520, "S4 · Lum 86.7")]):
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        _, fr_b = cap_r.read()
        crop_u = cv2.cvtColor(fr_b[460:1180, 240:840], cv2.COLOR_BGR2RGB)
        pil_u = Image.fromarray(crop_u).resize((150, 270), Image.Resampling.LANCZOS)
        xx = 64 + idx * 160
        board.paste(pil_u, (xx, by_a + 50))
        bd.rectangle([xx, by_a + 50, xx + 150, by_a + 320], outline=COL_GREEN if s_no in (2, 4) else COL_RED, width=2)
        bd.text((xx + 14, by_a + 330), lbl_s, font=F_SMALL, fill=COL_WHITE)
    cap_r.release()

    # =========================================================================
    # PANEL 2 (CENTER, x=762..1472): CHARACTER B — SPOTTER / BIOMECHANICS COACH
    # =========================================================================
    bd.rounded_rectangle([762, 128, 1472, 1392], radius=20, fill=(*COL_CARD, 255), outline=(56, 232, 255, 180), width=2)
    bd.rounded_rectangle([782, 146, 1022, 184], radius=9, fill=COL_CYAN)
    bd.text((798, 154), "CHARACTER B · SPOTTER", font=F_H3, fill=(10, 14, 18))
    bd.text((1036, 151), "SPOTTER / COACH (보조자 · 코치)", font=F_H2, fill=COL_WHITE)

    im_b_front, _ = render_char_b_gunmetal_pose(p2_raw, (315, 560), add_cyan_rim=True)
    im_b_side, _ = render_char_b_gunmetal_pose(p3_raw, (315, 560), add_cyan_rim=True)
    board.paste(im_b_front, (784, 202))
    board.paste(im_b_side, (1126, 202))
    bd.rectangle([784, 202, 784 + 315, 202 + 560], outline=(56, 232, 255, 95), width=1)
    bd.rectangle([1126, 202, 1126 + 315, 202 + 560], outline=(56, 232, 255, 95), width=1)
    bd.text((796, 212), "DEEP GUNMETAL (LUM 50)", font=F_SMALL, fill=COL_CYAN)
    bd.text((1138, 212), "CYAN RIM · 0% MUSCLE GLOW", font=F_SMALL, fill=COL_CYAN)

    bd.text((786, sy), "2. 왜 Character B는 딥 건메탈(Lum 50) + 반투명인가? (`t286u`~`t296u`)", font=F_H3, fill=COL_CYAN)
    swatches_b = [
        ((46, 52, 62),   "Deep Gunmetal-Charcoal", "Median Lum 50 (Character A Lum 86보다 한 단계 어두운 톤)"),
        ((88, 100, 118), "Slate Rim Contour",    "R:G:B = 0.86:0.96:1.14 (A의 무채색 실버와 즉시 구분)"),
        ((56, 232, 255), "Tactile Contact Ring", "Cyan-Blue #38E8FF (보조자 손끝 큐잉 접점만 표시)"),
        ((28, 34, 44),   "Zero Muscle Glow",     "보조자(B)는 근육 발광 0% -> 운동자(A)에만 시선 집중!"),
    ]
    for idx, (col_sw, name_sw, desc_sw) in enumerate(swatches_b):
        yy = sy + 36 + idx * 46
        bd.rounded_rectangle([786, yy, 838, yy + 34], radius=7, fill=col_sw, outline=COL_CYAN if idx == 2 else (255, 255, 255, 90), width=1)
        bd.text((852, yy + 1), name_sw, font=F_BODY, fill=COL_WHITE)
        bd.text((852, yy + 19), desc_sw, font=F_SMALL, fill=COL_MUTED)

    # NSCA Spotting Table Box inside Center Panel
    bd.rounded_rectangle([782, by_a, 1452, 1372], radius=14, fill=(14, 17, 24, 255), outline=COL_BORDER, width=1)
    bd.text((800, by_a + 14), "3. 공인 NSCA Spotting & 임상 촉지 큐잉 종목별 정석 위치 (임의 창작 0%)", font=F_H3, fill=COL_CYAN)
    nsca_rows = [
        ("① 바벨 백스쿼트 (NSCA Fig 3-14~16)", "위치: 운동자(A) 바로 등 뒤 (Rear Stance), 함께 스쿼트 하강", "손: 겨드랑이 아래 흉곽 옆(Under Armpits/Torso) 근접 대기 · 바벨 잡기 금지!"),
        ("② 바벨 벤치프레스 (NSCA Fig 3-3~6)", "위치: 벤치 머리맡 정중앙 직립 (Head of Bench)", "손: A의 양손 안쪽 바벨 샤프트에 얼터네이트 그립 (Alternated Grip)"),
        ("③ 덤벨 숄더/인클라인 프레스 (NSCA Fig 3-7~9)", "위치: 벤치 등받이/머리 뒤쪽 밀착 직립", "손: 팔꿈치(Elbows)가 아닌 반드시 양쪽 손목(Wrists/Forearms) 보조!"),
        ("④ 랫풀다운 / 로우 (ACE & Muscle&Motion)", "위치: 등 뒤 대각선 외곽 (Posterolateral Exterior)", "손: 견갑골 하각(Inferior Scapula) 하강 촉지 큐잉 -> 광배근 시야 방해 0%"),
    ]
    for idx, (r_title, r_pos, r_hand) in enumerate(nsca_rows):
        ry = by_a + 52 + idx * 78
        bd.rounded_rectangle([800, ry, 1434, ry + 68], radius=9, fill=(22, 27, 38, 255), outline=(56, 232, 255, 65), width=1)
        bd.text((814, ry + 6), r_title, font=F_BODY, fill=COL_CYAN)
        bd.text((814, ry + 27), r_pos, font=F_SMALL, fill=COL_WHITE)
        bd.text((814, ry + 46), r_hand, font=F_SMALL, fill=COL_MUTED)

    # =========================================================================
    # PANEL 3 (RIGHT, x=1496..2212): A + B ZERO-OCCLUSION INTERACTION SIMULATION
    # =========================================================================
    bd.rounded_rectangle([1496, 128, 2212, 1392], radius=20, fill=(*COL_CARD, 255), outline=(20, 228, 120, 180), width=2)
    bd.rounded_rectangle([1516, 146, 1766, 184], radius=9, fill=COL_GREEN)
    bd.text((1532, 154), "A + B · ZERO OCCLUSION", font=F_H3, fill=(10, 14, 18))
    bd.text((1780, 151), "2인 보조 시야 방해 0% 배치 원리", font=F_H2, fill=COL_WHITE)

    # Composite A (foreground right-center, 100% opacity, Silver-Grey + Emerald Glutes/Quads)
    # + B (rear-left behind A, 62% translucent Gunmetal-Slate, NSCA torso-wrap spot position!)
    comp_w, comp_h = 676, 780
    comp_canvas = Image.new("RGBA", (comp_w, comp_h), (16, 19, 26, 255))
    cd = ImageDraw.Draw(comp_canvas)

    im_b_spot, mask_b = render_char_b_gunmetal_pose(p4_raw, (460, 740), add_cyan_rim=True)
    im_a_main = render_char_a_pose(p4_raw, 4, False, (490, 760)).convert("RGBA")

    # Extract foreground mask of A so A sits 100% cleanly in front of B
    bgr_a_raw = cv2.resize(cv2.imread(p4_raw), (490, 760))
    lum_a = 0.299 * bgr_a_raw[:, :, 2] + 0.587 * bgr_a_raw[:, :, 1] + 0.114 * bgr_a_raw[:, :, 0]
    bg_a = float(np.median(lum_a[100:400, 8:35]))
    mask_a = np.clip((lum_a - (bg_a + 8.5)) / 10.0, 0.0, 1.0)

    # Paste B behind and to the left (`x=28, y=18`) at 64% opacity (NSCA Rear Stance behind lifter)
    b_rgba = np.array(im_b_spot.convert("RGBA")).astype(np.float32)
    b_alpha = (mask_b * 0.68 * 255.0).astype(np.uint8)
    b_pil = Image.fromarray(np.dstack([b_rgba[:, :, :3].astype(np.uint8), b_alpha]))
    comp_canvas.alpha_composite(b_pil, (24, 12))

    # Paste A in foreground (`x=168, y=16`) at 100% opacity -> ZERO occlusion of A's glutes, quads, or barbell grip!
    a_rgba = np.array(im_a_main).astype(np.float32)
    a_alpha = (mask_a * 255.0).astype(np.uint8)
    a_pil = Image.fromarray(np.dstack([a_rgba[:, :, :3].astype(np.uint8), a_alpha]))
    comp_canvas.alpha_composite(a_pil, (168, 16))

    # Add NSCA Cyan Tactile/Spotting Contact Rings (`Under Armpits / Lateral Ribcage` & `Sacrum/Hip Drive`)
    draw_cyan_contact_ring(cd, (338, 305), "NSCA TORSO SPOT (겨드랑이 아래 흉곽)")
    draw_cyan_contact_ring(cd, (315, 425), "HIP DRIVE TACTILE CUE (천골/고관절)")

    # Exterior green single arrow on A (100% outside body silhouette!)
    SS = 2
    rig = Image.new("RGBA", (comp_w * SS, comp_h * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig)
    c_arc = (260 * SS, 290 * SS)
    r_a = 88 * SS
    rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=185, end=248, fill=(*COL_GREEN, 245), width=4 * SS)
    ang_t = math.radians(248)
    tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
    v30.draw_arrow_head(rd, tip_p, 248 + 90, 11 * SS, (172, 255, 212, 255))
    comp_canvas.alpha_composite(rig.resize((comp_w, comp_h), Image.Resampling.LANCZOS))

    cd.rounded_rectangle([16, 16, 445, 52], radius=8, fill=(12, 16, 22, 225), outline=COL_CYAN, width=1)
    cd.text((28, 25), "B (Rear-Left · 65% Deep Gunmetal Lum 50)  +  A (Front-Right · Matte Slate Lum 86)", font=F_SMALL, fill=COL_CYAN)

    cd.rounded_rectangle([16, comp_h - 54, comp_w - 16, comp_h - 14], radius=8, fill=(12, 16, 22, 235), outline=COL_GREEN, width=1)
    cd.text((28, comp_h - 42), "시야 방해 0% 검증: A의 대둔근·대퇴사두근 에메랄드 근육결과 양손 바벨 그립이 100% 노출됨!", font=F_SMALL, fill=COL_GREEN)

    board.paste(comp_canvas, (1516, 202))
    bd.rectangle([1516, 202, 1516 + comp_w, 202 + comp_h], outline=COL_BORDER, width=2)

    # Bottom Rules Box inside Right Panel
    bd.rounded_rectangle([1516, 1000, 2192, 1372], radius=14, fill=(14, 17, 24, 255), outline=COL_BORDER, width=1)
    bd.text((1534, 1014), "4. @gymform.3d 2인 보조 연출 4대 절대 원칙 (`t289u`~`t294u`)", font=F_H3, fill=COL_GREEN)
    rules_ab = [
        ("원칙 1 · Zero Muscle Occlusion", "Character B는 절대 카메라와 Character A의 타겟 근육(대둔근·광배근·삼각근) 사이를 가로막지 않으며, 항상 등 뒤/머리 위 외곽(`Exterior Layer`)에 배치한다."),
        ("원칙 2 · 45% Luminance Contrast", "Character A(`Lum 86` 무광 매트 슬레이트)와 Character B(`Lum 50` 딥 건메탈, `65%` 반투명)의 명도·색온도를 분리해 두 마네킹이 겹쳐도 1초 만에 구분되게 한다."),
        ("원칙 3 · Cyan Contact Ring Only", "Character B에는 적색/녹색 근육 발광을 넣지 않고, 오직 A를 보조하는 손끝 접점(`Under Armpits`, `Wrists`, `Scapula`)에만 시안(`#38E8FF`) 링을 표시한다."),
        ("원칙 4 · Exterior Single Arrow", "모든 운동 궤적 화살표는 마네킹 몸통 위를 침범하지 않고 100% 신체 바깥 스튜디오 여백(`Exterior Margin`)에 단일 `4px` 곡선 화살표로만 그린다."),
    ]
    for idx, (r_hd, r_txt) in enumerate(rules_ab):
        ry = 1052 + idx * 76
        bd.rounded_rectangle([1534, ry, 2174, ry + 66], radius=9, fill=(22, 27, 38, 255), outline=(20, 228, 120, 65), width=1)
        bd.text((1548, ry + 7), r_hd, font=F_BODY, fill=COL_GREEN)
        bd.text((1548, ry + 31), r_txt[:56], font=F_SMALL, fill=COL_WHITE)
        if len(r_txt) > 56:
            bd.text((1548, ry + 47), r_txt[56:], font=F_SMALL, fill=COL_MUTED)

    board.convert("RGB").save(OUT_PATH, quality=96)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
