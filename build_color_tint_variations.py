import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/workspace/daily_info_reels/output"
KO_EB = "/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-ExtraBold.ttf"
KO_B  = "/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Bold.ttf"

def shade_pose2_with_tint_option(
    rgb_1080: np.ndarray,
    option_id: int,
    target_y_range: tuple[float, float] = (780.0, 1360.0),
) -> np.ndarray:
    f = rgb_1080.astype(np.float32)
    r_ch, g_ch, b_ch = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    H, W = r_ch.shape
    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    gb_diff = np.maximum(0.0, g_ch - 0.5 * (r_ch + b_ch))
    color_dom = np.maximum(rb_diff, gb_diff)

    # Multi-scale 3D anatomical muscle strand relief (exact formula from upper back!)
    fiber_signal = 0.55 * r_ch + 0.35 * g_ch + 0.10 * b_ch
    s_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 0.85)
    s_strand = cv2.GaussianBlur(fiber_signal, (0, 0), 2.2)
    s_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 6.5)
    s_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 15.0)

    hf_strand = (s_micro - s_strand) * 4.8 + (s_strand - s_belly) * 2.8
    sulcus = np.clip((s_broad - s_strand - 0.5) / 9.5, 0.0, 1.0)

    cool_lum = 0.35 * g_ch + 0.65 * b_ch
    cool_belly = cv2.GaussianBlur(cool_lum, (0, 0), 5.5)
    hv_cool = np.clip((cool_belly - 16.0) / 112.0, 0.0, 1.0)

    # Pure 3D grey sculpted muscle (exact upper-back texture everywhere)
    grey_sculpt = np.clip(
        28.0 + 118.0 * (hv_cool ** 0.92) + hf_strand * 0.95 - sulcus * 22.0,
        18.0,
        210.0,
    )

    any_col = np.clip((color_dom - 8.0) / 42.0, 0.0, 1.0)
    mono_target = np.stack([grey_sculpt * 0.96, grey_sculpt * 0.98, grey_sculpt * 1.03], axis=-1)
    mono = f * (1.0 - any_col[:, :, None]) + mono_target * any_col[:, :, None]

    # Target anatomical region gate (glutes, quads/hamstrings, lower erectors)
    y_min, y_max = target_y_range
    y_gate = np.clip((yy - y_min) / 42.0, 0.0, 1.0) * np.clip((y_max - yy) / 42.0, 0.0, 1.0)

    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    col_blur = cv2.GaussianBlur(color_dom, (0, 0), 4.0) * np.clip((base_lum - 32.0) / 18.0, 0.0, 1.0)
    col_eff = np.maximum(color_dom, col_blur * 0.85)
    target_mask = np.clip((col_eff - 9.0) / 30.0, 0.0, 1.0) * y_gate
    target_mask = cv2.GaussianBlur(target_mask.astype(np.float32), (0, 0), 1.5)[:, :, None]

    hv_muscle = np.clip((s_belly - 20.0) / 145.0, 0.0, 1.0)
    strand_relief = np.clip(
        0.18 + 0.56 * (hv_muscle ** 0.95) + (hf_strand / 42.0) - 0.38 * sulcus,
        0.02,
        1.0,
    )[:, :, None]
    ridge_spec = np.clip((hf_strand - 2.0) / 24.0, 0.0, 1.0)[:, :, None]

    c_groove = np.array([6.0, 26.0, 14.0], dtype=np.float32)
    c_mid    = np.array([14.0, 168.0, 72.0], dtype=np.float32)
    c_ridge  = np.array([155.0, 252.0, 188.0], dtype=np.float32)

    low_t = np.clip(strand_relief / 0.52, 0.0, 1.0)
    high_t = np.clip((strand_relief - 0.52) / 0.48, 0.0, 1.0)
    colored_full = (
        c_groove * (1.0 - low_t)
        + c_mid * low_t * (1.0 - high_t)
        + c_ridge * high_t
        + (c_ridge - c_mid) * (ridge_spec ** 1.25) * 0.55
    )

    # 6 distinct tint levels blending `mono` (exact upper-back grey muscle grain) with `colored_full`
    if option_id == 1:
        # 1번: 28% 아주 은은한 반투명 틴트 (회색 근육결 72% 그대로 투과)
        tinted = mono * 0.72 + colored_full * 0.28
    elif option_id == 2:
        # 2번: 42% 세이지 반투명 틴트 (회색 근육결 58% + 그린 42%)
        tinted = mono * 0.58 + colored_full * 0.42
    elif option_id == 3:
        # 3번: 58% 스튜디오 매트 그린 (회색 근육결 42% + 그린 58% — 가장 안정적인 밸런스)
        tinted = mono * 0.42 + colored_full * 0.58
    elif option_id == 4:
        # 4번: 실버 능선 하이라이트 + 55% 그린 (밝은 근육결 능선은 등처럼 실버로 빛나고 안쪽만 그린 착색)
        silver_keep = np.clip((strand_relief - 0.50) / 0.38, 0.0, 1.0) * 0.70
        w_col = 0.62 * (1.0 - silver_keep)
        tinted = mono * (1.0 - w_col) + colored_full * w_col
    elif option_id == 5:
        # 5번: 75% 선명 에메랄드 (직전 100%에서 눈부신 채도만 25% 낮춘 톤)
        tinted = mono * 0.25 + colored_full * 0.75
    else:
        # 6번: 100% 직전 시안 (비교 기준 — 방금 '색상이 너무 진하다'고 하신 원본)
        tinted = colored_full

    out = mono * (1.0 - target_mask) + np.clip(tinted, 0.0, 255.0) * target_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    raw_bgr = cv2.imread(f"{OUT_DIR}/_ref_v61_s4.jpg")
    raw_1080 = cv2.cvtColor(cv2.resize(raw_bgr, (1080, 1920), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)

    options = [
        (1, "1번 (농도 28%): 은은한 반투명 틴트", "회색 근육결 72% 투과 · 가장 절제된 컬러감"),
        (2, "2번 (농도 42%): 세이지 반투명 그린", "회색 근육결 58% + 그린 42% 차분한 스튜디오 톤"),
        (3, "3번 (농도 58%): 매트 에메랄드 밸런스", "형광기 없이 타겟 부위와 근육결이 동시에 선명"),
        (4, "4번 (실버 능선): 등 근육 실버 결 + 그린", "근육 능선은 등처럼 은색으로 남기고 결 사이만 그린"),
        (5, "5번 (농도 75%): 선명 에메랄드", "직전 100%에서 눈부신 형광기만 25% 감쇠"),
        (6, "6번 (농도 100%): 직전 시안 (비교 기준)", "방금 '색상이 너무 진하다'고 하신 기준점"),
    ]

    cell_w, cell_h = 560, 820
    pad = 20
    header_h = 105
    grid_w = pad * 4 + cell_w * 3
    grid_h = header_h + pad * 3 + cell_h * 2

    canvas = Image.new("RGB", (grid_w, grid_h), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    f_title = ImageFont.truetype(KO_EB, 32)
    f_lbl = ImageFont.truetype(KO_EB, 24)
    f_sub = ImageFont.truetype(KO_B, 18)

    draw.text(
        (pad, 30),
        "2번째 자세(스쿼트 후측면) 색상 농도·채도 6종 비교 시안 (1~5번 저채도/반투명 vs 6번 직전 100%)",
        font=f_title,
        fill=(242, 245, 250),
    )

    for idx, (opt_id, title_txt, sub_txt) in enumerate(options):
        col = idx % 3
        row = idx // 3
        x0 = pad + col * (cell_w + pad)
        y0 = header_h + row * (cell_h + pad)

        rendered = shade_pose2_with_tint_option(raw_1080, opt_id)
        crop = rendered[480:1580, 140:920]
        pil_crop = Image.fromarray(crop).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        canvas.paste(pil_crop, (x0, y0))

        bar = Image.new("RGBA", (cell_w, 82), (14, 16, 20, 225))
        canvas.paste(bar, (x0, y0), bar)
        b_col = (245, 95, 80) if opt_id == 6 else (80, 235, 155)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=b_col, width=3)
        draw.text((x0 + 16, y0 + 12), title_txt, font=f_lbl, fill=b_col)
        draw.text((x0 + 16, y0 + 48), sub_txt, font=f_sub, fill=(218, 224, 232))

    out_path = f"{OUT_DIR}/draft_pose2_6_color_tints.jpg"
    canvas.save(out_path, quality=95)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
