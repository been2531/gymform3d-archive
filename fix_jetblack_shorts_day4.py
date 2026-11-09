with open("/workspace/daily_info_reels/build_day4_rdl_master.py", "r", encoding="utf-8") as f:
    code = f.read()

old_block = '''    shorts_band = np.clip((yy - (hip_y - 120.0)) / 35.0, 0.0, 1.0) * np.clip(((kn_y - 35.0) - yy) / 45.0, 0.0, 1.0)
    dark_low_sat = np.clip((58.0 - lum) / 22.0, 0.0, 1.0) * np.clip((0.22 - sat) / 0.12, 0.0, 1.0) * np.clip((14.0 - act_dom) / 10.0, 0.0, 1.0)
    shorts_mask_1080 = cv2.GaussianBlur((shorts_band * dark_low_sat * w_fg).astype(np.float32), (0, 0), 2.5)

    matte_lum = grey_back_exact * (1.0 - 0.52 * shorts_mask_1080) + (lum * 0.72 + 4.0) * (0.52 * shorts_mask_1080)'''

new_block = '''    shorts_band = np.clip((yy - (hip_y - 145.0)) / 28.0, 0.0, 1.0) * np.clip(((kn_y - 30.0) - yy) / 38.0, 0.0, 1.0)
    dark_low_sat = np.clip((64.0 - lum) / 20.0, 0.0, 1.0) * np.clip((0.26 - sat) / 0.14, 0.0, 1.0) * np.clip((18.0 - act_dom) / 12.0, 0.0, 1.0)
    shorts_mask_1080 = cv2.GaussianBlur((shorts_band * dark_low_sat * w_fg).astype(np.float32), (0, 0), 2.0)

    # Crisp opaque matte jet-black drawers (@demicstory standard: lum ~ 14..22)
    shorts_jet_lum = np.clip(lum * 0.46 + 3.0, 10.0, 24.0)
    matte_lum = grey_back_exact * (1.0 - 0.88 * shorts_mask_1080) + shorts_jet_lum * (0.88 * shorts_mask_1080)'''

assert old_block in code, "old_block not found"
code = code.replace(old_block, new_block, 1)

with open("/workspace/daily_info_reels/build_day4_rdl_master.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Updated build_day4_rdl_master.py with crisp opaque matte jet-black drawers!")
