with open("/workspace/daily_info_reels/build_day4_rdl_master.py", "r") as f:
    code = f.read()

old_specs = """STEP_SPECS_DAY4 = {
    1: ("day4_rdl_step1_ecorche_720p.mp4", 57, 4, True,  "bone_blood",   95),
    2: ("day4_rdl_step2_ecorche_720p.mp4", 39, 0, True,  "muscle_blood", 95),
    3: ("day4_rdl_step3_ecorche_720p.mp4", 46, 6, False, "bone_blood",   95),
    4: ("day4_rdl_step4_ecorche_720p.mp4", 59, 5, False, "muscle_blood", 95),
}"""

new_specs = """STEP_SPECS_DAY4 = {
    1: ("day4_rdl_step1_locked_720p.mp4",  5, 110, True,  "bone_blood",   95),
    2: ("day4_rdl_step2_ecorche_720p.mp4", 39,  0, True,  "muscle_blood", 95),
    3: ("day4_rdl_step1_ecorche_720p.mp4", 80,  8, False, "bone_blood",   95),
    4: ("day4_rdl_step4_locked_720p.mp4",  5, 140, False, "muscle_blood", 95),
}"""
assert old_specs in code, "old_specs not found"
code = code.replace(old_specs, new_specs, 1)

old_shader = """    # 2. Option B (@demicstory Core-to-Grey Edge Falloff + 100% Striation Preservation)
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
    out = mono * (1.0 - blend_w[:, :, None]) + act_rgb * blend_w[:, :, None]"""

new_shader = """    # 2. Option B (@demicstory Core-to-Grey Edge Falloff + Step-Specific Bone vs Muscle Fiber Striation Shader)
    if step_idx in (1, 3):
        # Pure Spine Bone X-Ray Focus (Steps 1 & 3): Gate strictly to the spinal column above hip_y
        spine_gate = np.clip(((hip_y + 18.0) - yy) / 42.0, 0.0, 1.0)
    else:
        spine_gate = 1.0

    w_act = cv2.GaussianBlur((np.clip((act_dom - 10.0) / 34.0, 0.0, 1.0) ** 0.85).astype(np.float32), (0, 0), 1.8) * (1.0 - 0.95 * shorts_mask_1080) * spine_gate
    core_seed = (np.clip((act_dom - 18.0) / 46.0, 0.0, 1.0) ** 1.10) * spine_gate
    pulse = 1.0 + 0.06 * math.sin(2.0 * math.pi * pulse_phase)
    core_w = np.clip(cv2.GaussianBlur(core_seed.astype(np.float32), (0, 0), 7.0) * w_act * 0.86 * pulse, 0.0, 0.88)

    # Extract 3D muscle fiber ridges & inter-fiber sulci from both R (active channel) and GB
    gb_ch = 0.55 * g_ch + 0.45 * b_ch
    r_blur_fine = cv2.GaussianBlur(r_ch, (0, 0), 1.4)
    r_blur_mid  = cv2.GaussianBlur(r_ch, (0, 0), 5.0)
    r_blur_wide = cv2.GaussianBlur(r_ch, (0, 0), 14.0)
    gb_blur_mid = cv2.GaussianBlur(gb_ch, (0, 0), 4.5)

    fiber_fine   = (r_ch - r_blur_mid) * 1.85 + (gb_ch - gb_blur_mid) * 1.35
    fiber_bundle = (r_blur_fine - r_blur_wide) * 0.95
    sulc_3d  = np.clip(-fiber_fine / 14.0, 0.0, 1.0)
    ridge_3d = np.clip(fiber_fine / 16.0, 0.0, 1.0)
    vol_3d   = np.clip((0.56 * r_ch + 0.44 * gb_ch) / 172.0, 0.18, 1.18)
    fiber_mod = np.clip(vol_3d + 0.50 * ridge_3d - 0.54 * sulc_3d + 0.004 * fiber_bundle, 0.12, 1.38)

    L = np.clip((grey_back_exact - 20.0) / 148.0, 0.0, 1.0)
    bone_core = (np.clip((lum - 140.0) / 55.0, 0.0, 1.0) * np.clip((act_dom - 12.0) / 26.0, 0.0, 1.0) * spine_gate) if step_idx in (1, 3) else 0.0

    if is_wrong:
        if step_idx == 1:
            col_r = np.clip(grey_back_exact * 1.42 + 34.0 * (L ** 0.88) + 52.0 * bone_core, 18.0, 238.0)
            col_g = np.clip(grey_back_exact * 0.38 * (1.0 - 0.45 * sulc) + 58.0 * bone_core, 10.0, 145.0)
            col_b = np.clip(grey_back_exact * 0.40 * (1.0 - 0.45 * sulc) + 46.0 * bone_core, 12.0, 135.0)
        else:
            col_r = np.clip(34.0 + 172.0 * fiber_mod + 48.0 * (ridge_3d ** 1.3) - 42.0 * sulc_3d, 22.0, 242.0)
            col_g = np.clip(14.0 + 42.0 * fiber_mod + 28.0 * (ridge_3d ** 1.5) - 18.0 * sulc_3d, 10.0, 105.0)
            col_b = np.clip(16.0 + 44.0 * fiber_mod + 26.0 * (ridge_3d ** 1.5) - 18.0 * sulc_3d, 10.0, 105.0)
    else:
        if step_idx == 3:
            col_g = np.clip(grey_back_exact * 1.42 + 32.0 * (L ** 0.88) + 54.0 * bone_core, 18.0, 236.0)
            col_r = np.clip(grey_back_exact * 0.38 * (1.0 - 0.45 * sulc) + 42.0 * bone_core, 10.0, 130.0)
            col_b = np.clip(grey_back_exact * 0.78 * (1.0 - 0.30 * sulc) + 52.0 * bone_core, 14.0, 178.0)
        else:
            # Step 4: Vivid Emerald-Green 3D Hamstring & Erector Muscle Fiber Striations (근육결)
            col_r = np.clip(14.0 + 38.0 * fiber_mod + 32.0 * (ridge_3d ** 1.5) - 14.0 * sulc_3d, 10.0, 96.0)
            col_g = np.clip(32.0 + 178.0 * fiber_mod + 56.0 * (ridge_3d ** 1.3) - 46.0 * sulc_3d, 24.0, 248.0)
            col_b = np.clip(22.0 + 96.0 * fiber_mod + 42.0 * (ridge_3d ** 1.4) - 26.0 * sulc_3d, 16.0, 168.0)

    act_rgb = np.stack([col_r, col_g, col_b], axis=-1)
    blend_w = np.clip(core_w + 0.42 * bone_core, 0.0, 0.92) * (1.0 - 0.95 * shorts_mask_1080)
    out = mono * (1.0 - blend_w[:, :, None]) + act_rgb * blend_w[:, :, None]"""
assert old_shader in code, "old_shader not found"
code = code.replace(old_shader, new_shader, 1)

old_vo = """    vo_lines = [
        (0.25, "Romanian deadlift. Stop rounding your lower back."),
        (5.60, "Letting the barbell drift forward strains your lumbar spine."),
        (10.40, "Lock your spine neutral and push your hips straight back."),
        (15.40, "Keep the bar close to drive maximum tension into your hamstrings and glutes."),
    ]"""

new_vo = """    vo_lines = [
        (0.20, "Romanian deadlift. Avoid rounding your lumbar spine and compressing your discs."),
        (5.45, "Letting the barbell drift forward overloads your traps and lower back."),
        (10.35, "Lock your vertebrae in a neutral spine and hinge straight back."),
        (15.30, "Shave the bar down your thighs to stretch and fire your hamstring fibers."),
    ]"""
assert old_vo in code, "old_vo not found"
code = code.replace(old_vo, new_vo, 1)

with open("/workspace/daily_info_reels/build_day4_rdl_master.py", "w") as f:
    f.write(code)
print("Successfully patched build_day4_rdl_master.py!")
