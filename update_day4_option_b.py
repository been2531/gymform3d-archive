with open("/workspace/daily_info_reels/build_day4_rdl_master.py", "r", encoding="utf-8") as f:
    code = f.read()

old_specs = '''STEP_SPECS_DAY4 = {
    1: ("day4_rdl_step1_i2v.mp4", 48, 122, True,  "bone_blood",   95),
    2: ("day4_rdl_step2_i2v.mp4", 48, 122, True,  "muscle_blood", 95),
    3: ("day4_rdl_step3_i2v.mp4", 50, 0,   False, "bone_blood",   95),
    4: ("day4_rdl_step4_i2v.mp4", 52, 0,   False, "muscle_blood", 95),
}'''

new_specs = '''STEP_SPECS_DAY4 = {
    1: ("day4_rdl_step1_ecorche_720p.mp4", 57, 4, True,  "bone_blood",   95),
    2: ("day4_rdl_step2_ecorche_720p.mp4", 39, 0, True,  "muscle_blood", 95),
    3: ("day4_rdl_step3_ecorche_720p.mp4", 55, 5, False, "bone_blood",   95),
    4: ("day4_rdl_step4_ecorche_720p.mp4", 59, 5, False, "muscle_blood", 95),
}'''

assert old_specs in code, "old_specs not found"
code = code.replace(old_specs, new_specs, 1)

old_cache = 'CACHE_KP_PKL = "/workspace/daily_info_reels/_cache_day4_rdl_native_kps.pkl"'
new_cache = 'CACHE_KP_PKL = "/workspace/daily_info_reels/_cache_day4_rdl_ecorche_720p_kps.pkl"'
assert old_cache in code, "old_cache not found"
code = code.replace(old_cache, new_cache, 1)

old_shader = '''    body_norm = np.clip((lum - bg_lum) / max(220.0 - bg_lum, 40.0), 0.0, 1.30)
    matte_lum = np.clip(38.0 + (body_norm ** 0.88) * 145.0, 34.0, 215.0)

    yy = np.arange(H, dtype=np.float32)[:, None]
    hip_y = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1])
    kn_y = 0.5 * (kp["kn_r"][1] + kp["kn_l"][1])

    shorts_band = np.clip((yy - (hip_y - 120.0)) / 35.0, 0.0, 1.0) * np.clip(((kn_y - 35.0) - yy) / 45.0, 0.0, 1.0)
    dark_low_sat = np.clip((58.0 - lum) / 22.0, 0.0, 1.0) * np.clip((0.22 - sat) / 0.12, 0.0, 1.0) * np.clip((14.0 - act_dom) / 10.0, 0.0, 1.0)
    shorts_mask_1080 = cv2.GaussianBlur((shorts_band * dark_low_sat * w_fg).astype(np.float32), (0, 0), 2.5)

    matte_lum = matte_lum * (1.0 - 0.48 * shorts_mask_1080) + (lum * 0.78 + 4.0) * (0.48 * shorts_mask_1080)

    mono_r = 38.0 * (1.0 - w_fg) + (matte_lum - 1.5) * w_fg
    mono_g = 40.0 * (1.0 - w_fg) + (matte_lum + 0.5) * w_fg
    mono_b = 44.0 * (1.0 - w_fg) + (matte_lum + 3.0) * w_fg
    mono = np.stack([mono_r, mono_g, mono_b], axis=-1)

    w_act = cv2.GaussianBlur((np.clip((act_dom - 11.0) / 40.0, 0.0, 1.0) ** 0.85).astype(np.float32), (0, 0), 1.8)
    pulse = 1.0 + 0.08 * math.sin(2.0 * math.pi * pulse_phase)
    fiber_relief = np.clip(((lum - 18.0) / 172.0) * pulse, 0.12, 1.38)
    act_boost = np.clip((act_dom / 82.0) * pulse, 0.0, 1.38)

    if is_wrong:
        act_r = np.clip(68.0 + 184.0 * fiber_relief + 44.0 * act_boost, 0.0, 255.0)
        act_g = np.clip(10.0 + 38.0 * (fiber_relief ** 1.45), 0.0, 95.0)
        act_b = np.clip(14.0 + 34.0 * (fiber_relief ** 1.45), 0.0, 90.0)
    else:
        act_g = np.clip(70.0 + 182.0 * fiber_relief + 44.0 * act_boost, 0.0, 255.0)
        act_r = np.clip(8.0 + 42.0 * (fiber_relief ** 1.40), 0.0, 100.0)
        act_b = np.clip(28.0 + 95.0 * (fiber_relief ** 1.15) + 20.0 * act_boost, 0.0, 175.0)

    act_rgb = np.stack([act_r, act_g, act_b], axis=-1)
    out = mono * (1.0 - w_act[:, :, None]) + act_rgb * w_act[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8), shorts_mask_1080'''

new_shader = '''    # 1. Unified Upper-Back Sculpted Grey Base (grey_back_exact) across entire body
    hv_cool = np.clip((lum - 18.0) / 172.0, 0.0, 1.35)
    blur_med = cv2.GaussianBlur(lum, (0, 0), 2.8)
    sulc = np.clip((blur_med - lum) / 14.0, 0.0, 1.0)
    hf_ridge = lum - cv2.GaussianBlur(lum, (0, 0), 1.6)
    grey_back_exact = np.clip(32.0 + 108.0 * (hv_cool ** 0.95) + hf_ridge * 0.55 - sulc * 14.0, 20.0, 168.0)

    yy = np.arange(H, dtype=np.float32)[:, None]
    hip_y = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1])
    kn_y = 0.5 * (kp["kn_r"][1] + kp["kn_l"][1])

    shorts_band = np.clip((yy - (hip_y - 120.0)) / 35.0, 0.0, 1.0) * np.clip(((kn_y - 35.0) - yy) / 45.0, 0.0, 1.0)
    dark_low_sat = np.clip((58.0 - lum) / 22.0, 0.0, 1.0) * np.clip((0.22 - sat) / 0.12, 0.0, 1.0) * np.clip((14.0 - act_dom) / 10.0, 0.0, 1.0)
    shorts_mask_1080 = cv2.GaussianBlur((shorts_band * dark_low_sat * w_fg).astype(np.float32), (0, 0), 2.5)

    matte_lum = grey_back_exact * (1.0 - 0.52 * shorts_mask_1080) + (lum * 0.72 + 4.0) * (0.52 * shorts_mask_1080)

    mono_r = 38.0 * (1.0 - w_fg) + (matte_lum - 1.5) * w_fg
    mono_g = 40.0 * (1.0 - w_fg) + (matte_lum + 0.5) * w_fg
    mono_b = 44.0 * (1.0 - w_fg) + (matte_lum + 3.0) * w_fg
    mono = np.stack([mono_r, mono_g, mono_b], axis=-1)

    # 2. Option B (@demicstory Core-to-Grey Edge Falloff + 100% Striation Preservation)
    w_act = cv2.GaussianBlur((np.clip((act_dom - 10.0) / 36.0, 0.0, 1.0) ** 0.85).astype(np.float32), (0, 0), 1.8) * (1.0 - 0.90 * shorts_mask_1080)
    core_seed = np.clip((act_dom - 22.0) / 50.0, 0.0, 1.0) ** 1.12
    pulse = 1.0 + 0.06 * math.sin(2.0 * math.pi * pulse_phase)
    core_w = np.clip(cv2.GaussianBlur(core_seed.astype(np.float32), (0, 0), 7.5) * w_act * 0.80 * pulse, 0.0, 0.84)

    L = np.clip((grey_back_exact - 20.0) / 148.0, 0.0, 1.0)
    bone_core = np.clip((lum - 155.0) / 55.0, 0.0, 1.0) * np.clip((act_dom - 14.0) / 28.0, 0.0, 1.0) if step_idx in (1, 3) else 0.0

    if is_wrong:
        col_r = np.clip(grey_back_exact * 1.36 + 26.0 * (L ** 0.90) + 38.0 * bone_core, 16.0, 224.0)
        col_g = np.clip(grey_back_exact * 0.42 * ((1.0 - 0.45 * sulc)) + 45.0 * bone_core, 10.0, 130.0)
        col_b = np.clip(grey_back_exact * 0.44 * ((1.0 - 0.45 * sulc)) + 35.0 * bone_core, 12.0, 120.0)
    else:
        col_g = np.clip(grey_back_exact * 1.32 + 22.0 * (L ** 0.90) + 38.0 * bone_core, 16.0, 220.0)
        col_r = np.clip(grey_back_exact * 0.42 * ((1.0 - 0.45 * sulc)) + 32.0 * bone_core, 10.0, 120.0)
        col_b = np.clip(grey_back_exact * 0.74 * ((1.0 - 0.30 * sulc)) + 42.0 * bone_core, 14.0, 165.0)

    act_rgb = np.stack([col_r, col_g, col_b], axis=-1)
    blend_w = np.clip(core_w + 0.35 * bone_core, 0.0, 0.90)
    out = mono * (1.0 - blend_w[:, :, None]) + act_rgb * blend_w[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8), shorts_mask_1080'''

assert old_shader in code, "old_shader not found"
code = code.replace(old_shader, new_shader, 1)

with open("/workspace/daily_info_reels/build_day4_rdl_master.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Successfully updated build_day4_rdl_master.py with Option B shader + 720p ecorche clips!")
