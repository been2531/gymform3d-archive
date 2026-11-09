with open("/workspace/daily_info_reels/build_4step_bone_muscle_master.py", "r", encoding="utf-8") as f:
    code = f.read()

old_fn = '''def composite_depth_behind_title_and_translucent_preview(
    base_rgb: np.ndarray,
    fg_mask: np.ndarray,
    torso_icons: list,
    active_slot: int,
    is_wrong: bool,
    exercise_name: str = "Barbell Back Squat",
) -> np.ndarray:
    """
    Option 3 Standard (Day 4+):
    1. Places the Kaushan Script exercise title on the studio back wall BEHIND the 3D mannequin (`fg_front = 0.92`),
       so when standing tall, the mannequin head/traps pass cleanly in front of the letters with zero facial clutter.
    2. Renders the 4-slot top preview bar (`y = 248..406`) as a translucent glass pill (`touch_p = 0.65` see-through when touched).
    """
    bg_title = Image.fromarray(base_rgb).convert("RGBA")
    draw_single_line_kaushan_title(bg_title, exercise_name)
    title_rgb = np.array(bg_title.convert("RGB")).astype(np.float32)

    fg_front = np.clip(fg_mask[:, :, None] * 0.92, 0.0, 0.92)
    layered = np.clip(title_rgb * (1.0 - fg_front) + base_rgb.astype(np.float32) * fg_front, 0, 255).astype(np.uint8)

    with_prev = draw_compact_demic_top_preview(layered, torso_icons, active_slot=active_slot, is_wrong=is_wrong)
    touch_p = np.clip(fg_mask[:, :, None] * 0.65, 0.0, 0.65)
    out = np.clip(with_prev.astype(np.float32) * (1.0 - touch_p) + layered.astype(np.float32) * touch_p, 0, 255).astype(np.uint8)
    return out'''

new_fn = '''_TITLE_ROI_CACHE = {}

def composite_depth_behind_title_and_translucent_preview(
    base_rgb: np.ndarray,
    fg_mask: np.ndarray,
    torso_icons: list,
    active_slot: int,
    is_wrong: bool,
    exercise_name: str = "Barbell Back Squat",
) -> np.ndarray:
    """
    Option 3 Standard (Day 4+) with Turbo ROI Caching:
    1. Places the Kaushan Script exercise title on the studio back wall BEHIND the 3D mannequin (`fg_front = 0.92`),
       so when standing tall, the mannequin head/traps pass cleanly in front of the letters with zero facial clutter.
    2. Renders the 4-slot top preview bar (`y = 248..406`) as a translucent glass pill (`touch_p = 0.65` see-through when touched).
    """
    ty0, ty1 = 410, 565
    if exercise_name not in _TITLE_ROI_CACHE:
        blank = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_single_line_kaushan_title(blank, exercise_name)
        t_arr = np.array(blank.crop((0, ty0, W, ty1)), dtype=np.float32)
        _TITLE_ROI_CACHE[exercise_name] = (t_arr[:, :, :3], t_arr[:, :, 3:4] / 255.0)

    t_rgb, t_a = _TITLE_ROI_CACHE[exercise_name]
    out = base_rgb.copy()

    # 1. Depth-behind title in ROI y=410..565
    roi_base = out[ty0:ty1].astype(np.float32)
    fg_front = np.clip(fg_mask[ty0:ty1, :, None] * 0.92, 0.0, 0.92)
    eff_a = t_a * (1.0 - fg_front)
    out[ty0:ty1] = np.clip(roi_base * (1.0 - eff_a) + t_rgb * eff_a, 0, 255).astype(np.uint8)

    # 2. Translucent glass top preview in ROI y=230..425
    py0, py1 = 230, 425
    layered_roi = out[py0:py1].copy()
    with_prev = draw_compact_demic_top_preview(out, torso_icons, active_slot=active_slot, is_wrong=is_wrong)
    prev_roi = with_prev[py0:py1].astype(np.float32)
    touch_p = np.clip(fg_mask[py0:py1, :, None] * 0.65, 0.0, 0.65)
    with_prev[py0:py1] = np.clip(prev_roi * (1.0 - touch_p) + layered_roi.astype(np.float32) * touch_p, 0, 255).astype(np.uint8)
    return with_prev'''

assert old_fn in code, "old_fn not found"
code = code.replace(old_fn, new_fn, 1)
with open("/workspace/daily_info_reels/build_4step_bone_muscle_master.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Patched composite_depth_behind_title_and_translucent_preview with ROI caching!")
