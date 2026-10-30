# Day 6 v5: Incline Dumbbell Press, palette A at the source (t609u)

**Why:**
- Day 6 v4/v5 couldn't separate the plates, bench and backdrop in post: the raw plates had the backdrop's brightness (playbook D17/D18).
- Fix at the source: the clips are generated with palette A built in (`docs/plate_palette_A_reference.jpg`).

**Start frames** (1080×1920, 9:16, front camera, same incline bench and framing as the v3 frames):
- `day6_v5_wrong_start.png`: bottom of the press, elbows flared ~84–90°, red glow on the front deltoids.
- `day6_v5_right_start.png`: bottom of the press, elbows tucked ~45–60°, emerald glow on the upper chest.

**How they were made:**
- **Generation:** GenerateImage (3 tries each), using the v2/v3 Day 6 start frames, the palette A mockup and the Day 4 body as references. The raw outputs are kept as `*_gen_raw.jpg`.
- **Global grade** (`tools/startframe_grade.py`, no masks):
  - a smooth 60 px flat-field on the backdrop (removes the vignette and the soft floor shadow);
  - one luminance tone curve (backdrop → 12, skin p50 → Day 4 level);
  - a cool tint on the darks that depends on luminance only.
- **Gates:** `tools/startframe_gates.py` → `start_frame_gates.jsonl`. Both pass:
  - figure − backdrop 64.5 / 64.3 (≥ 33);
  - backdrop flat (far span 1.4 / 1.0, ≤ 4), RGB (10,12,17);
  - skin p50 66.9 / 65.1 (Day 4 63.9);
  - plate faces vs backdrop p05 42.9 / 43.9, with 0.67% / 0.50% of face area under 40 RGB (≤ 2%);
  - bench (102,94,93) vs backdrop 145 / 136 (≥ 15), vs skin 50 / 44 (≥ 10).

One grok.com image-to-video take per clip (done by the separate browser step).

## WRONG (image-to-video, start frame `day6_v5_wrong_start.png`)
Smooth incline dumbbell press with the elbows flared wide. The anatomical mannequin presses both cobalt-blue hex dumbbells from the deep bottom beside the shoulders, upper arms flared out to the sides at about 85-90 degrees from the torso, straight up to full lockout over the upper chest, then lowers them slowly back to the same deep bottom with the elbows still flared. Full range of motion every rep. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm: ease out of the bottom, smooth through the middle, ease into the top, no pause and no jerk. The same two dumbbells stay in the hands the whole time and never change shape, size or colour: matte cobalt royal blue plates on every face with a thin brushed-gold rim ring, brushed steel handles. The red glow stays only on the front deltoids. The camera is static and locked, front view, no zoom, no pan. Lighting is flat, even and shadowless: the body keeps exactly the same matte charcoal-grey skin tone, no shaded side, no specular change. Keep every colour exactly as in the start frame: flat dark cool-charcoal backdrop with no vignette and no floor shadow, warm light-grey bench, matte-black shorts. No text, no logo.

## RIGHT (image-to-video, start frame `day6_v5_right_start.png`)
Smooth, controlled incline dumbbell press with the elbows tucked. The anatomical mannequin presses both cobalt-blue hex dumbbells from the deep bottom just outside the upper chest, elbows tucked at about 45-60 degrees from the torso with the forearms vertical, straight up to full lockout over the upper chest, then lowers them slowly back to the same deep bottom with the elbows still tucked. Full range of motion every rep. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm: ease out of the bottom, smooth through the middle, ease into the top, no pause and no jerk. The same two dumbbells stay in the hands the whole time and never change shape, size or colour: matte cobalt royal blue plates on every face with a thin brushed-gold rim ring, brushed steel handles. The emerald-green glow stays only on the upper chest. The camera is static and locked, front view, no zoom, no pan. Lighting is flat, even and shadowless: the body keeps exactly the same matte charcoal-grey skin tone, no shaded side, no specular change. Keep every colour exactly as in the start frame: flat dark cool-charcoal backdrop with no vignette and no floor shadow, warm light-grey bench, matte-black shorts. No text, no logo.

## After download
Save the clips as `output/assets/day6_v5_wrong_720p.mp4` and `output/assets/day6_v5_right_720p.mp4`. Then check each clip:
- elbow angle held;
- full depth and full lockout;
- the same plates throughout;
- glow on the right muscle;
- no camera drift.

Then run `tools/startframe_gates.py` on sampled frames: plate faces ≥ 40 RGB from the backdrop throughout the motion.
