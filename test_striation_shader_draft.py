import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import build_v30_pure_graphic_numbers as v30

OUT_DIR = "/workspace/daily_info_reels/output"


def shade_striated_muscle_and_bone_1080p(
    rgb_1080: np.ndarray,
    is_wrong: bool = False,
    is_bone_xray: bool = False,
    target_y_range: tuple[float, float] = (750.0, 1400.0),
) -> np.ndarray:
    """
    Preserves 100% of the hyper-detailed 3D muscle fiber striations (like the grey back in Day 3 Step 4)
    even inside the colored (Green / Red) target muscles and X-Ray bones!
    Key principle:
    - Never clamp the green/red channel to 136..255 across the whole muscle belly.
    - Keep inter-fiber grooves (sulci) deep dark emerald/crimson (lum ~ 20..42),
      mid-strands rich saturated emerald/crimson (lum ~ 95..165),
      and only the crisp fiber ridges (hf_ridge) bright luminous emerald/crimson (lum ~ 210..248).
    """
    f = rgb_1080.astype(np.float32)
    r_ch, g_ch, b_ch = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    H, W = r_ch.shape
    yy, xx = np.mgrid[:H, :W].astype(np.float32)

    rb_diff = np.maximum(0.0, r_ch - b_ch)
    gb_diff = np.maximum(0.0, g_ch - 0.5 * (r_ch + b_ch))
    color_dom = np.maximum(rb_diff, gb_diff)

    # Extract multi-scale anatomical muscle strand relief (exact structure that makes the grey back look 3D!)
    fiber_signal = 0.55 * r_ch + 0.35 * g_ch + 0.10 * b_ch
    s_micro = cv2.GaussianBlur(fiber_signal, (0, 0), 0.85)
    s_strand = cv2.GaussianBlur(fiber_signal, (0, 0), 2.2)
    s_belly = cv2.GaussianBlur(fiber_signal, (0, 0), 6.5)
    s_broad = cv2.GaussianBlur(fiber_signal, (0, 0), 15.0)

    # High-frequency individual strand ridges (+) and inter-fiber grooves/sulci (-)
    hf_strand = (s_micro - s_strand) * 4.8 + (s_strand - s_belly) * 2.8
    sulcus = np.clip((s_broad - s_strand - 0.5) / 9.5, 0.0, 1.0)

    # Cool non-red base for neutralizing non-target colored areas into sleek sculpted charcoal-silver
    cool_lum = 0.35 * g_ch + 0.65 * b_ch
    cool_belly = cv2.GaussianBlur(cool_lum, (0, 0), 5.5)
    hv_cool = np.clip((cool_belly - 16.0) / 112.0, 0.0, 1.0)

    # Notice how grey_sculpt preserves every single strand ridge and dark groove:
    grey_sculpt = np.clip(
        28.0 + 118.0 * (hv_cool ** 0.92) + hf_strand * 0.95 - sulcus * 22.0,
        18.0,
        210.0,
    )

    any_col = np.clip((color_dom - 8.0) / 42.0, 0.0, 1.0)
    mono_target = np.stack([grey_sculpt * 0.96, grey_sculpt * 0.98, grey_sculpt * 1.03], axis=-1)
    mono = f * (1.0 - any_col[:, :, None]) + mono_target * any_col[:, :, None]

    # Target anatomical region gate
    y_min, y_max = target_y_range
    y_gate = np.clip((yy - y_min) / 42.0, 0.0, 1.0) * np.clip((y_max - yy) / 42.0, 0.0, 1.0)

    # Protect black athletic shorts (low lum, low color_dom)
    base_lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    col_blur = cv2.GaussianBlur(color_dom, (0, 0), 4.0) * np.clip((base_lum - 32.0) / 18.0, 0.0, 1.0)
    col_eff = np.maximum(color_dom, col_blur * 0.85)
    target_mask = np.clip((col_eff - 9.0) / 30.0, 0.0, 1.0) * y_gate
    target_mask = cv2.GaussianBlur(target_mask.astype(np.float32), (0, 0), 1.5)[:, :, None]

    # Normalized 0..1 3D strand relief inside the colored muscle:
    # Dark grooves stay near 0.05..0.20, mid-fibers at 0.35..0.68, sharp fiber ridges at 0.75..1.0!
    hv_muscle = np.clip((s_belly - 20.0) / 145.0, 0.0, 1.0)
    strand_relief = np.clip(
        0.18 + 0.56 * (hv_muscle ** 0.95) + (hf_strand / 42.0) - 0.38 * sulcus,
        0.02,
        1.0,
    )[:, :, None]
    ridge_spec = np.clip((hf_strand - 2.0) / 24.0, 0.0, 1.0)[:, :, None]

    if not is_wrong:
        # Sculpted Emerald-Green with DEEP dark inter-fiber grooves and razor-sharp strand highlights
        c_groove = np.array([6.0, 26.0, 14.0], dtype=np.float32)     # Deep dark emerald groove
        c_mid    = np.array([14.0, 168.0, 72.0], dtype=np.float32)   # Rich 3D emerald fiber body
        c_ridge  = np.array([155.0, 252.0, 188.0], dtype=np.float32) # Crisp individual fiber highlight
    else:
        # Sculpted Crimson-Red with DEEP dark inter-fiber grooves and razor-sharp strand highlights
        c_groove = np.array([32.0, 6.0, 10.0], dtype=np.float32)     # Deep dark crimson groove
        c_mid    = np.array([195.0, 22.0, 28.0], dtype=np.float32)   # Rich 3D crimson fiber body
        c_ridge  = np.array([255.0, 160.0, 130.0], dtype=np.float32) # Crisp individual fiber highlight

    # Piecewise 3D shading so grooves stay dark and ridges pop:
    low_t = np.clip(strand_relief / 0.52, 0.0, 1.0)
    high_t = np.clip((strand_relief - 0.52) / 0.48, 0.0, 1.0)
    colored_3d = (
        c_groove * (1.0 - low_t)
        + c_mid * low_t * (1.0 - high_t)
        + c_ridge * high_t
        + (c_ridge - c_mid) * (ridge_spec ** 1.25) * 0.55
    )

    if is_bone_xray:
        # Enhance internal X-Ray vertebrae / joint core with crisp trabecular ridges
        bone_core = (
            np.clip((g_ch - 55.0) / 85.0, 0.0, 1.0)
            * np.clip((color_dom - 22.0) / 45.0, 0.0, 1.0)
            * y_gate
        )[:, :, None]
        if is_wrong:
            xray_col = np.stack([
                np.clip(215.0 + hf_strand * 1.4, 140.0, 255.0),
                np.clip(75.0 + hf_strand * 0.9, 25.0, 185.0),
                np.clip(45.0 + hf_strand * 0.6, 15.0, 135.0),
            ], axis=-1)
        else:
            xray_col = np.stack([
                np.clip(65.0 + hf_strand * 0.9, 25.0, 185.0),
                np.clip(225.0 + hf_strand * 1.3, 150.0, 255.0),
                np.clip(175.0 + hf_strand * 1.1, 100.0, 245.0),
            ], axis=-1)
        colored_3d = colored_3d * (1.0 - bone_core * 0.80) + xray_col * (bone_core * 0.80)

    out = mono * (1.0 - target_mask) + np.clip(colored_3d, 0.0, 255.0) * target_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    # 1. Test on Day 3 Step 4 raw (_ref_v61_s4.jpg) AND Day 4 Step 4 raw (9c011845...jpg)
    d3_s4_bgr = cv2.imread(f"{OUT_DIR}/_ref_v61_s4.jpg")
    d3_s4_1080 = cv2.cvtColor(cv2.resize(d3_s4_bgr, (1080, 1920), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)

    # Compare Old Day 3 Step 4 vs New Striation-Preserving Shader on Day 3 Step 4
    old_d3_s4 = cv2.cvtColor(cv2.imread(f"{OUT_DIR}/day3_step4_peak.jpg"), cv2.COLOR_BGR2RGB)
    new_d3_s4 = shade_striated_muscle_and_bone_1080p(d3_s4_1080, is_wrong=False, is_bone_xray=False, target_y_range=(780.0, 1360.0))

    # Also test on Day 4 raw keyframes (1280x720 native stills before 480p video blur!)
    d4_s1_raw = cv2.cvtColor(cv2.resize(cv2.imread("/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/assets/ed02268f9928a932f6c31d779fe6e7551bea7516cb1c1d2487d4d4125c4e6459.jpg"), (1080, 1920), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
    d4_s4_raw = cv2.cvtColor(cv2.resize(cv2.imread("/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/assets/9c011845da6043b9865ea529d6432167a4e878089ea9cbfd207369523b1c84b1.jpg"), (1080, 1920), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)

    new_d4_s1 = shade_striated_muscle_and_bone_1080p(d4_s1_raw, is_wrong=True, is_bone_xray=True, target_y_range=(550.0, 1050.0))
    new_d4_s4 = shade_striated_muscle_and_bone_1080p(d4_s4_raw, is_wrong=False, is_bone_xray=False, target_y_range=(820.0, 1380.0))

    # Build a crisp zoom comparison sheet so we can inspect the muscle fiber strands directly!
    # Panel 1: Old Day 3 Step 4 (thigh zoom) | Panel 2: New Striation Shader on Day 3 Step 4 (thigh & back zoom)
    # Panel 3: New Striation Shader on Day 4 Step 1 (Lumbar Spine X-Ray) | Panel 4: New Striation Shader on Day 4 Step 4 (Hamstrings)
    pw, ph = 520, 924
    pad = 20
    header_h = 90
    sheet = Image.new("RGB", (pad + 4 * (pw + pad), header_h + ph + pad), (14, 16, 20))
    sd = ImageDraw.Draw(sheet)
    sd.text((24, 18), "Striation-Preserving 3D Muscle & Bone Shader Test (색상 칠해진 부위 안의 근육결·고랑·뼈 디테일 보존 비교)", font=v30.f_pretendard(24, "extrabold"), fill=(245, 248, 255))
    sd.text((24, 52), "핵심 개선: 그린/레드 채널이 255로 포화되어 근육결이 뭉개지던 현상 제거 -> 등 근육처럼 어두운 고랑(Sulcus)과 밝은 근섬유 결(Ridge)의 3D 대비 100% 보존", font=v30.f_pretendard(16, "bold"), fill=(180, 225, 255))

    panels = [
        (old_d3_s4, "기존 Day 3 Step 4 (그린 포화로 허벅지 결 뭉개짐)", (245, 90, 75)),
        (new_d3_s4, "개선 셰이더 Day 3 Step 4 (그린 안쪽 근육결·고랑 선명 보존)", (40, 240, 135)),
        (new_d4_s1, "Day 4 Step 1 (요추 L1~L5 뼈 X-Ray 마디 + 주변 근육결)", (245, 90, 75)),
        (new_d4_s4, "Day 4 Step 4 (햄스트링·둔근 그린 내부 근육결 보존)", (40, 240, 135)),
    ]
    for idx, (arr, title, col) in enumerate(panels):
        x0 = pad + idx * (pw + pad)
        y0 = header_h
        im_p = Image.fromarray(arr).resize((pw, ph), Image.Resampling.LANCZOS)
        sheet.paste(im_p, (x0, y0))
        sd.rectangle([x0 - 2, y0 - 2, x0 + pw + 1, y0 + ph + 1], outline=col, width=3)
        sd.rectangle([x0 + 8, y0 + 8, x0 + pw - 8, y0 + 44], fill=(12, 14, 18))
        sd.text((x0 + 16, y0 + 15), title, font=v30.f_pretendard(15, "bold"), fill=col)

    out_path = f"{OUT_DIR}/draft_striation_shader_comparison.jpg"
    sheet.save(out_path, quality=95)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
