#!/usr/bin/env python3
"""
Builds the Day 3 Simple 2 Master Verification Sheet (`day3_simple2_standing_vs_squat_briefs_check.jpg`)
showing CONTINUOUS vertical frames (zero cut between top preview and mannequin!) for:
1. Step 1 (Wrong) Standing (`frame 0` — highest point of motion, verifying 0% contact with top preview!)
2. Step 1 (Wrong) Squatting (`frame 51` — Simple 2 callout + 3D Athletic Briefs + Day 1 X-Ray Bone Flow)
3. Step 2 (Right) Standing (`frame 0` — highest point of motion, verifying 0% contact with top preview!)
4. Step 2 (Right) Squatting (`frame 51` — Simple 2 callout + 3D Athletic Briefs + Day 1 X-Ray Bone Flow)
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import build_v30_pure_graphic_numbers as v30
import preview_v62_seamless_day2_master as p62
import build_clean_d4_variants as cd4
import build_simple_4concepts as bs4
import build_day1_style_xray_and_hud_concepts as bd

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920


def render_simple2_frame(raw_bgr_720p: np.ndarray, step_idx: int, is_wrong: bool, show_callout: bool = True) -> tuple[np.ndarray, int, int]:
    b_rgb, b_mask, b_kp = cd4.render_zero_haze_day1_mannequin_1080p(raw_bgr_720p, step_idx, is_wrong)
    ys_fg = np.where(b_mask > 0.25)[0]
    y_top_man = int(ys_fg.min()) if len(ys_fg) else 560
    y_bot_man = int(ys_fg.max()) if len(ys_fg) else 1475

    icons = bs4.make_unsquished_day1_top_icons({}, current_step=1 if is_wrong else 2)
    dock_rgb = bs4.draw_day1_exact_top_dock(b_rgb, icons, active_slot=1, is_wrong=is_wrong)

    if show_callout:
        out_rgb = bs4.draw_simple_2_single_surgical_callout(dock_rgb, b_kp, is_wrong)
    else:
        out_rgb = bd.draw_day1_bone_energy_flow_and_exterior_arcs(
            dock_rgb, np.zeros((H, W), dtype=np.float32), b_kp, is_wrong, with_exterior_callouts=False, with_laser_axis=True
        )
    return out_rgb, y_top_man, y_bot_man


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seq1 = p62._load_slice(os.path.join(OUT_DIR, p62.STEP_POOL_SPECS[1][0]), p62.STEP_POOL_SPECS[1][1], p62.STEP_POOL_SPECS[1][2], target_len=52)
    seq2 = p62._load_slice(os.path.join(OUT_DIR, p62.STEP_POOL_SPECS[2][0]), p62.STEP_POOL_SPECS[2][1], p62.STEP_POOL_SPECS[2][2], target_len=52)

    s1_stand, y1_s_top, y1_s_bot = render_simple2_frame(seq1[0], 1, True, show_callout=False)
    s1_squat, y1_q_top, y1_q_bot = render_simple2_frame(seq1[-1], 1, True, show_callout=True)
    s2_stand, y2_s_top, y2_s_bot = render_simple2_frame(seq2[0], 2, False, show_callout=False)
    s2_squat, y2_q_top, y2_q_bot = render_simple2_frame(seq2[-1], 2, False, show_callout=True)

    print(f"Step 1 Standing: top_capsule_bot=451, mannequin_top={y1_s_top} (gap={y1_s_top - 451}px), bot={y1_s_bot}")
    print(f"Step 1 Squatting: top_capsule_bot=451, mannequin_top={y1_q_top} (gap={y1_q_top - 451}px), bot={y1_q_bot}")
    print(f"Step 2 Standing: top_capsule_bot=451, mannequin_top={y2_s_top} (gap={y2_s_top - 451}px), bot={y2_s_bot}")
    print(f"Step 2 Squatting: top_capsule_bot=451, mannequin_top={y2_q_top} (gap={y2_q_top - 451}px), bot={y2_q_bot}")

    # Build a 4-column continuous full-frame comparison sheet (each panel cropped continuously from y=140..1540 -> 1080x1400, scaled to 600x778)
    pw, ph = 600, 778
    pad = 24
    head_h = 86
    sheet_w = pad + 4 * (pw + pad)
    sheet_h = head_h + ph + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 16, 20))
    sd = ImageDraw.Draw(sheet)
    f_t = v30.f_pretendard(26, "extrabold")
    f_c = v30.f_pretendard(19, "bold")

    sd.text(
        (pad, 22),
        "심플 2번 확정 — 상단 프리뷰 미접촉(서 있을 때도 약 105px 여유) + 3D 스포츠 브리프 착용 확인",
        font=f_t,
        fill=(245, 248, 255),
    )

    panels = [
        (f"1. 오자세 — 서 있을 때 (프리뷰와 {y1_s_top - 451}px 여유)", s1_stand, (255, 115, 45)),
        ("2. 오자세 — 앉았을 때 (심플 2번 + 3D 브리프)", s1_squat, (255, 115, 45)),
        (f"3. 정자세 — 서 있을 때 (프리뷰와 {y2_s_top - 451}px 여유)", s2_stand, (45, 245, 135)),
        ("4. 정자세 — 앉았을 때 (심플 2번 + 3D 브리프)", s2_squat, (45, 245, 135)),
    ]

    for idx, (lbl, img_1080, col) in enumerate(panels):
        x0 = pad + idx * (pw + pad)
        y0 = head_h
        sd.text((x0 + 6, y0 - 30), lbl, font=f_c, fill=col)
        # Continuous vertical slice y=140..1540 (zero horizontal cut!)
        cont_slice = img_1080[140:1540, :]
        rs = cv2.resize(cont_slice, (pw, ph), interpolation=cv2.INTER_AREA)
        sheet.paste(Image.fromarray(rs), (x0, y0))
        sd.rounded_rectangle([x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 2], radius=14, outline=col, width=2)

    out_path = os.path.join(OUT_DIR, "day3_simple2_standing_vs_squat_briefs_check.jpg")
    sheet.save(out_path, quality=96)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
