#!/usr/bin/env python3
"""
Ultra-Simple 4 Concepts (`t333u`: "너무 복잡해 심플하게 풀어내봐", `t334u`: "프리뷰 상단에 있는 것도 너무 찌부된거 아니야? 레퍼런스 잡은건 깔끔하게 좋던데 카피해"):
1. Exact Day 1 Top 3-Thumbnail Dock (`v28.draw_luxury_top_dock` 100% copied):
   - Equal-sized `142 x 114` slots (`aspect ratio 1.2456 : 1`), cropped from an exact `1.2456 : 1` window (`760 x 610` around the squat mannequin) so there is **0.0% squishing (`찌부러짐 0%`)**!
   - Zero clutter inside thumbnails (no `1/4` stickers, no mini `✕/✓` circles, no header pill above the dock) + exact 4x-supersampled Day 1 Titanium `✕ / ✓` emblem on the right!
2. 0.0% Background Haze (`build_clean_d4_variants.render_zero_haze_day1_mannequin_1080p`) + Day 1 3D Neon X-Ray Bone Flow.
3. 4 Ultra-Simple Graphic Variations (`심플 1 ~ 4`) with clutter stripped away:
   - **심플 1 (좌·우 원포인트 숫자 콜아웃 — 막대·박스 0%)**: Left knee pointer -> `84% JOINT RISK`, Right knee arc -> `16° KNEE CAVE`. Zero bars, zero boxes!
   - **심플 2 (단일 원포인트 포인터 — @demicstory 미니멀)**: Only 1 clean surgical callout on the left (`84% · 16° INWARD KNEE`) + `@demicstory` neon laser bone axis.
   - **심플 3 (하단 단일 줄다리기 바 1줄 [84% vs 16%])**: One single horizontal tug-of-war bar at the bottom (`84% JOINT ━━━━━━━░░ 16% QUADS`) + simple `16°` knee arc.
   - **심플 4 (단일 270° 원형 링 + 무릎 16° 아크)**: One clean `84%` circular ring on the left + `16°` knee arc on the right.
"""
import os
import sys
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62
import build_day1_style_xray_and_hud_concepts as bd
import build_clean_d4_variants as cd4

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920


def make_unsquished_day1_top_icons(base_frames: dict) -> list[tuple[Image.Image, Image.Image]]:
    """
    Creates the exact `(142, 114)` thumbnail icons for `v28.draw_luxury_top_dock` with 100% preserved aspect ratio (`142 / 114 = 1.2456`):
    - Crops `x = 160 .. 920` (`w = 760px`) and `y = 700 .. 1310` (`h = 610px`, `760 / 610 = 1.2459 == 142 / 114`!)
      so the mannequin & barbell are NEVER squished horizontally or vertically!
    - Inactive thumbnails are clean desaturated monochrome dimmed just like Day 1 (`build_v36_clean_master.py` lines 478-487).
    """
    size = (142, 114)
    crop_box = (160, 700, 920, 1310)  # 760 x 610 -> exact 142:114 aspect ratio!

    icons = []
    for s_i in (1, 2, 4):
        rgb = base_frames[s_i]
        act_im = Image.fromarray(rgb).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
        # Create clean monochrome dimmed version for inactive state (exact Day 1 style)
        lum = np.mean(rgb.astype(np.float32), axis=2, keepdims=True)
        mono = np.clip(lum * np.array([0.96, 0.99, 1.04]), 0, 255).astype(np.uint8)
        inact_im = Image.fromarray(mono).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
        inact_im.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 118)))
        icons.append((act_im, inact_im))
    return icons


def draw_day1_exact_top_dock(frame_rgb: np.ndarray, torso_icons: list, active_slot: int, is_wrong: bool) -> np.ndarray:
    """Applies the exact Day 1 `v28.draw_luxury_top_dock` (zero squishing, zero sticker clutter)."""
    canvas = Image.fromarray(frame_rgb).convert("RGBA")
    v28.draw_luxury_top_dock(canvas, torso_icons, active_idx=active_slot, is_wrong=is_wrong)
    return np.array(canvas.convert("RGB"))


def draw_simple_1_left_right_onepoint_numbers(frame_rgb: np.ndarray, kp_1080: dict, is_wrong: bool) -> np.ndarray:
    """
    심플 1 (좌·우 원포인트 숫자 콜아웃 — 막대·박스 0%):
    - Left knee (`◎`) thin pointer -> `84%` + `JOINT RISK` (or `96%` + `QUAD LOCK`).
    - Right knee curved arc -> `16°` + `KNEE CAVE` (or `0°` + `KNEE LOCK`).
    - ZERO progress bars, ZERO boxes, ZERO bottom clutter!
    """
    # Render clean neon bone flow + right exterior curved `16°` arc + `@demicstory` laser axis
    fr = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        frame_rgb, np.zeros((H, W), dtype=np.float32), kp_1080, is_wrong, with_exterior_callouts=True, with_laser_axis=True
    )
    base = Image.fromarray(fr).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    pct_val = 84 if is_wrong else 96
    lbl_top = "JOINT RISK" if is_wrong else "QUAD LOCK"
    lbl_bot = "INWARD KNEE" if is_wrong else "ALIGNED TRACK"

    kn_r = (int(kp_1080["kn_r"][0]), int(kp_1080["kn_r"][1]))
    cd4._draw_precision_leader(rd, kn_r, (kn_r[0] - 54, kn_r[1] - 36), (288, kn_r[1] - 36), col)

    tx = 54
    ty = kn_r[1] - 112
    f_num = v30.f_pretendard(96, "extrabold")
    f_pct = v30.f_pretendard(44, "extrabold")
    f_l1 = v30.f_pretendard(20, "extrabold")
    f_l2 = v30.f_pretendard(15, "bold")

    rd.text((tx + 2, ty + 3), f"{pct_val}", font=f_num, fill=(0, 0, 0, 210))
    rd.text((tx, ty), f"{pct_val}", font=f_num, fill=(col[0], col[1], col[2], 255))
    nw = int(rd.textlength(f"{pct_val}", font=f_num))
    rd.text((tx + nw + 4, ty + 16), "%", font=f_pct, fill=(255, 255, 255, 245))
    rd.text((tx + 2, ty + 102), lbl_top, font=f_l1, fill=(255, 255, 255, 248))
    rd.text((tx + 2, ty + 128), lbl_bot, font=f_l2, fill=(col[0], col[1], col[2], 235))

    return np.array(Image.alpha_composite(base, ov).convert("RGB"))


def draw_simple_2_single_surgical_callout(frame_rgb: np.ndarray, kp_1080: dict, is_wrong: bool) -> np.ndarray:
    """
    심플 2 (단일 미니 캡슐 포인터 1개 — @demicstory 순수 미니멀):
    - Removes the right-side callout completely so only ONE sleek surgical callout (`v28.draw_surgical_micro_callout` style)
      sits in the left margin pointing to the knee (`84%` | `INWARD KNEE` / `16° VALGUS RISK`), plus the `@demicstory` neon laser line!
    """
    fr = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        frame_rgb, np.zeros((H, W), dtype=np.float32), kp_1080, is_wrong, with_exterior_callouts=False, with_laser_axis=True
    )
    canvas = Image.fromarray(fr).convert("RGBA")
    kn_r = kp_1080["kn_r"]

    if is_wrong:
        v28.draw_surgical_micro_callout(
            canvas,
            joint_xy=kn_r,
            card_xy=(44, int(kn_r[1] - 54)),
            deg_val="84%",
            title_str="INWARD KNEE",
            sub_str="16° JOINT OVERLOAD",
            is_wrong=True,
        )
    else:
        v28.draw_surgical_micro_callout(
            canvas,
            joint_xy=kn_r,
            card_xy=(44, int(kn_r[1] - 54)),
            deg_val="96%",
            title_str="QUAD LOCK",
            sub_str="0° ALIGNED TRACK",
            is_wrong=False,
        )
    return np.array(canvas.convert("RGB"))


def draw_simple_3_single_tug_of_war_bar(frame_rgb: np.ndarray, kp_1080: dict, is_wrong: bool) -> np.ndarray:
    """
    심플 3 (하단 단일 슬림 줄다리기 바 1줄 [84% vs 16%] + 무릎 16° 아크):
    - Keeps the side margins 100% open (only the small `16° KNEE CAVE` curved arrow on the right knee),
    - And at the bottom (`y = 1365`), shows **ONE single sleek tug-of-war bar** where Orange/Green (`84%` / `96%`) vs Grey (`16%` / `4%`)
      meet in a single bar! Instant 0.1-second comprehension with minimal visual elements.
    """
    fr = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        frame_rgb, np.zeros((H, W), dtype=np.float32), kp_1080, is_wrong, with_exterior_callouts=True, with_laser_axis=True
    )
    base = Image.fromarray(fr).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    pct_l = 84 if is_wrong else 96
    pct_r = 16 if is_wrong else 4
    lbl_l = "KNEE JOINT" if is_wrong else "QUADS · VMO"
    lbl_r = "QUADS · VMO" if is_wrong else "KNEE JOINT"

    bx1, bx2 = 100, 980
    bw = bx2 - bx1
    by = 1388
    bh = 18

    f_big = v30.f_pretendard(44, "extrabold")
    f_lbl = v30.f_pretendard(18, "extrabold")

    # Left big % + label
    rd.text((bx1, by - 56), f"{pct_l}%", font=f_big, fill=(col[0], col[1], col[2], 255))
    w_pl = int(rd.textlength(f"{pct_l}%", font=f_big))
    rd.text((bx1 + w_pl + 12, by - 36), lbl_l, font=f_lbl, fill=(245, 248, 255, 248))

    # Right label + %
    r_pct_str = f"{pct_r}%"
    w_pr = int(rd.textlength(r_pct_str, font=f_big))
    w_lr = int(rd.textlength(lbl_r, font=f_lbl))
    rd.text((bx2 - w_pr, by - 56), r_pct_str, font=f_big, fill=(165, 175, 190, 235))
    rd.text((bx2 - w_pr - w_lr - 12, by - 36), lbl_r, font=f_lbl, fill=(165, 175, 190, 220))

    # Single 24-segment Tug-of-War Bar (`pct_l` active colored segments on left, `pct_r` dim segments on right)
    n_segs = 28
    gap = 5
    seg_w = (bw - (n_segs - 1) * gap) // n_segs
    act_n = int(round((pct_l / 100.0) * n_segs))
    for i in range(n_segs):
        sx = bx1 + i * (seg_w + gap)
        c_s = (col[0], col[1], col[2], 250) if i < act_n else (150, 160, 175, 75)
        rd.rounded_rectangle([sx, by, sx + seg_w, by + bh], radius=5, fill=c_s)

    return np.array(Image.alpha_composite(base, ov).convert("RGB"))


def draw_simple_4_single_radial_ring_and_arc(frame_rgb: np.ndarray, kp_1080: dict, is_wrong: bool) -> np.ndarray:
    """
    심플 4 (좌측 심플 270° 원형 링 1개 + 우측 16° 아크 — 막대 0%):
    - Left knee leader line (`◎`) -> ONE clean 270° Circular Ring (`84%` `JOINT RISK` / `96%` `QUAD LOCK`).
    - Right knee -> ONE curved angle callout (`16° KNEE CAVE` / `0° KNEE LOCK`).
    - Zero horizontal/vertical bars, zero bottom clutter!
    """
    fr = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
        frame_rgb, np.zeros((H, W), dtype=np.float32), kp_1080, is_wrong, with_exterior_callouts=True, with_laser_axis=True
    )
    base = Image.fromarray(fr).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ov)

    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    pct_val = 84 if is_wrong else 96
    ring_sub = "JOINT RISK" if is_wrong else "QUAD LOCK"

    kn_r = (int(kp_1080["kn_r"][0]), int(kp_1080["kn_r"][1]))
    cx, cy, r_ring = 166, kn_r[1] - 12, 88
    cd4._draw_precision_leader(rd, kn_r, (kn_r[0] - 46, cy), (cx + r_ring + 8, cy), col)

    bbox = [cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring]
    rd.ellipse([cx - r_ring - 6, cy - r_ring - 6, cx + r_ring + 6, cy + r_ring + 6], fill=(12, 15, 20, 160))
    rd.arc(bbox, start=135, end=405, fill=(255, 255, 255, 38), width=14)
    end_deg = int(135 + 270 * (pct_val / 100.0))
    rd.arc([cx - r_ring - 2, cy - r_ring - 2, cx + r_ring + 2, cy + r_ring + 2], start=135, end=end_deg, fill=(col[0], col[1], col[2], 70), width=22)
    rd.arc(bbox, start=135, end=end_deg, fill=(col[0], col[1], col[2], 255), width=14)

    f_num = v30.f_pretendard(62, "extrabold")
    f_pct = v30.f_pretendard(26, "extrabold")
    f_sub = v30.f_pretendard(15, "extrabold")
    s_txt = f"{pct_val}"
    tw = int(rd.textlength(s_txt, font=f_num))
    pw = int(rd.textlength("%", font=f_pct))
    sx = cx - (tw + pw) // 2
    rd.text((sx, cy - 36), s_txt, font=f_num, fill=(255, 255, 255, 255))
    rd.text((sx + tw + 2, cy - 20), "%", font=f_pct, fill=(col[0], col[1], col[2], 255))
    sw = int(rd.textlength(ring_sub, font=f_sub))
    rd.text((cx - sw // 2, cy + 30), ring_sub, font=f_sub, fill=(col[0], col[1], col[2], 245))

    return np.array(Image.alpha_composite(base, ov).convert("RGB"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_peaks = {}
    for s_i in (1, 2, 3, 4):
        fn, i0, i1, is_w, *_ = p62.STEP_POOL_SPECS[s_i]
        seq = p62._load_slice(os.path.join(OUT_DIR, fn), i0, i1, target_len=52)
        raw_peaks[s_i] = seq[-1]

    base_frames = {}
    masks = {}
    kps = {}
    for s_i in (1, 2, 3, 4):
        is_w = (s_i in (1, 3))
        b_rgb, b_mask, b_kp = cd4.render_zero_haze_day1_mannequin_1080p(raw_peaks[s_i], s_i, is_w)
        base_frames[s_i] = b_rgb
        masks[s_i] = b_mask
        kps[s_i] = b_kp

    # Build the 100% UN-SQUISHED (`142x114`, exact `1.2456:1` crop) Day 1 top dock icons!
    torso_icons = make_unsquished_day1_top_icons(base_frames)

    variants = {"S1": {}, "S2": {}, "S3": {}, "S4": {}}
    for s_i in (1, 2):
        is_w = (s_i == 1)
        act_slot = 0 if s_i == 1 else 2

        s1 = draw_simple_1_left_right_onepoint_numbers(base_frames[s_i], kps[s_i], is_w)
        variants["S1"][s_i] = draw_day1_exact_top_dock(s1, torso_icons, act_slot, is_w)

        s2 = draw_simple_2_single_surgical_callout(base_frames[s_i], kps[s_i], is_w)
        variants["S2"][s_i] = draw_day1_exact_top_dock(s2, torso_icons, act_slot, is_w)

        s3 = draw_simple_3_single_tug_of_war_bar(base_frames[s_i], kps[s_i], is_w)
        variants["S3"][s_i] = draw_day1_exact_top_dock(s3, torso_icons, act_slot, is_w)

        s4 = draw_simple_4_single_radial_ring_and_arc(base_frames[s_i], kps[s_i], is_w)
        variants["S4"][s_i] = draw_day1_exact_top_dock(s4, torso_icons, act_slot, is_w)

    # Save 4-Column Overview Sheet (`심플 1 ~ 4`)
    cw, ch = 520, 924
    pad = 22
    hdr_h = 116
    sheet = Image.new("RGB", (cw * 4 + pad * 5, ch * 2 + hdr_h + pad * 3), (10, 12, 16))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(27, "extrabold")
    f_c = v30.f_pretendard(19, "extrabold")
    f_s = v30.f_pretendard(14, "bold")

    sd.text(
        (pad, 18),
        "Day 1 상단 프리뷰 캡슐 100% 복원(찌부러짐 0%) + 복잡한 막대·박스 덜어낸 초심플 시안 4종 (심플 1 ~ 4)",
        font=f_t,
        fill=(255, 255, 255),
    )
    meta_cols = [
        ("심플 1 · 좌·우 원포인트 숫자 (막대·박스 0%)", "좌측 무릎 포인터 [84% JOINT RISK] + 우측 무릎 아크 [16° KNEE CAVE]만 배치"),
        ("심플 2 · 단일 미니 캡슐 포인터 (@demicstory형)", "우측·하단 요소 없이 좌측에 [84% | INWARD KNEE · 16°] 포인터 딱 1개만 배치"),
        ("심플 3 · 하단 단일 줄다리기 바 1줄 (84% vs 16%)", "측면은 비우고 발 아래에 [84% JOINT ━━━━░░ 16% QUADS] 한 줄 바로 직관화"),
        ("심플 4 · 좌측 270° 원형 링 1개 + 우측 16° 아크", "복잡한 막대 없이 좌측에 [84% 원형 링], 우측에 [16° 각도 아크]만 대칭 배치"),
    ]

    for idx, k in enumerate(("S1", "S2", "S3", "S4")):
        x0 = pad + idx * (cw + pad)
        sd.text((x0 + 4, 60), meta_cols[idx][0], font=f_c, fill=(255, 165, 70) if idx == 0 else (45, 250, 140))
        sd.text((x0 + 4, 86), meta_cols[idx][1], font=f_s, fill=(195, 205, 220))

        im_w = Image.fromarray(variants[k][1]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im_w, (x0, hdr_h))
        sd.rounded_rectangle([x0 - 2, hdr_h - 2, x0 + cw + 2, hdr_h + ch + 2], radius=14, outline=(255, 94, 36), width=3)

        y2 = hdr_h + ch + pad
        im_r = Image.fromarray(variants[k][2]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im_r, (x0, y2))
        sd.rounded_rectangle([x0 - 2, y2 - 2, x0 + cw + 2, y2 + ch + 2], radius=14, outline=(35, 245, 125), width=3)

    out_master = os.path.join(OUT_DIR, "day3_simple_4variants_unsquished_top.jpg")
    sheet.save(out_master, quality=95)

    # Also save a 1:1 Close-Up showing the Top Capsule (`y=215..410`) + the Simple Lower/Side Callouts (`y=820..1460`)!
    closeup = Image.new("RGB", (1020 * 2 + 60, 880 * 2 + 120), (10, 12, 16))
    cd = ImageDraw.Draw(closeup)
    cd.text(
        (20, 18),
        "1:1 원본 픽셀 확인 — 상단 Day 1 정비율 캡슐(찌부러짐 0%) + 초심플 그래픽 시안 4종 (심플 1 ~ 4)",
        font=f_t,
        fill=(255, 255, 255),
    )
    pos = [(20, 72), (1060, 72), (20, 982), (1060, 982)]
    for idx, k in enumerate(("S1", "S2", "S3", "S4")):
        px, py = pos[idx]
        top_strip = Image.fromarray(variants[k][1][215:405, 30:1050])   # 1020 x 190 (Top Dock)
        bot_strip = Image.fromarray(variants[k][1][800:1450, 30:1050])  # 1020 x 650 (Knees + Simple Callout)
        closeup.paste(top_strip, (px, py + 32))
        closeup.paste(bot_strip, (px, py + 32 + 196))
        cd.text((px + 6, py), meta_cols[idx][0], font=f_c, fill=(255, 165, 70) if idx == 0 else (45, 250, 140))
        cd.rounded_rectangle([px - 2, py + 30, px + 1022, py + 878], radius=12, outline=(255, 255, 255, 75), width=2)

    out_closeup = os.path.join(OUT_DIR, "day3_simple_4variants_1to1_closeup.jpg")
    closeup.save(out_closeup, quality=95)
    print("Saved:", out_master, "and", out_closeup)


if __name__ == "__main__":
    main()
