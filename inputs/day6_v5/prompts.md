# Day 6 v5: Incline Dumbbell Press, palette A at the source (t609u)

**Why:**
- Day 6 v4/v5 couldn't separate the plates, bench and backdrop in post: the raw plates had the backdrop's brightness (playbook D17/D18).
- Fix at the source: the clips are generated with palette A built in (`docs/plate_palette_A_reference.jpg`).

**Start frames** (1080×1920, 9:16, front camera, same incline bench and framing as the v3 frames):
- `day6_v5_wrong_start.png`: bottom of the press, elbows flared ~84–90°, red glow on the front deltoids.
- `day6_v5_right_start.png`: bottom of the press, elbows tucked ~45–60°, emerald glow on the upper chest.

**How they were made:**
- **Generation:** GenerateImage (3 tries each), using the v2/v3 Day 6 start frames, the palette A mockup and the Day 4 body as references. The raw outputs are kept as `*_gen_raw.jpg`.
- **Global grade v2** (`tools/startframe_grade.py <raw> <out> 38.3 <skin p50> 35,39,43 <skin p75>`), with no spatial masks:
  - a smooth 60 px flat-field on the backdrop;
  - one luminance curve: backdrop to the Day 4 studio level 38.3, skin p50 to the Day 4 level, and the p75 knot lifts only the skin;
  - low-saturation pixels near the backdrop level take the studio chroma, so the backdrop is exactly RGB 35,39,43;
  - a hue-qualified colour function for saturated blue only (cobalt plates): chroma ×1.3–2.5 and a lum floor of 44, so the shaded faces stay cobalt.
- The earlier near-black-backdrop grade (10,12,17) is kept as `*_dark_v1.png`. It broke the Day 4 studio rule.

**Gates** (`tools/startframe_gates.py` → `start_frame_gates.jsonl`), wrong / right:

| Gate | Wrong | Right | Target |
|---|---|---|---|
| Backdrop RGB | 35,39,43 | 35,39,43 | Day 4 studio |
| Figure − backdrop | 41.0 | 38.7 | 33–41, PASS |
| Skin p50 | 63.2 | 63.2 | Day 4 63.9 |
| Bench vs backdrop | 104 | 95 | ≥ 15, PASS |
| Bench vs skin | 60 | 51 | ≥ 10, PASS |
| Backdrop span, ≥ 60 px from objects | 1.0 | 2.0 | PASS |
| Backdrop span, all blocks | 4.0 | 5.4 | ≤ 4; right FAILS (halo / shadow next to the bench) |
| Plate faces vs backdrop, p05 | 35.2 | 41.6 | ≥ 40; wrong FAILS |
| Plate face area < 40 RGB | 3.3% | 2.4% | ≤ 2%; both FAIL |

The plate failures are the dark seam inside the gold ring and the hand-shadowed inner faces next to the grip.

One grok.com image-to-video take per clip (done by the separate browser step).

## WRONG (image-to-video, start frame `day6_v5_wrong_start.png`)
Smooth incline dumbbell press with the elbows flared wide. The anatomical mannequin presses both cobalt-blue hex dumbbells from the deep bottom beside the shoulders, upper arms flared out to the sides at about 85-90 degrees from the torso, straight up to full lockout over the upper chest, then lowers them slowly back to the same deep bottom with the elbows still flared. Full range of motion every rep. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm: ease out of the bottom, smooth through the middle, ease into the top, no pause and no jerk. The same two dumbbells stay in the hands the whole time and never change shape, size or colour: matte cobalt royal blue plates on every face with a thin brushed-gold rim ring, brushed steel handles. The red glow stays only on the front deltoids. The camera is static and locked, front view, no zoom, no pan. Lighting is flat, even and shadowless: the body keeps exactly the same matte charcoal-grey skin tone, no shaded side, no specular change. Keep every colour exactly as in the start frame: flat dark charcoal grey backdrop with no vignette and no floor shadow, warm light-grey bench, matte-black shorts. No text, no logo.

## RIGHT (image-to-video, start frame `day6_v5_right_start.png`)
Smooth, controlled incline dumbbell press with the elbows tucked. The anatomical mannequin presses both cobalt-blue hex dumbbells from the deep bottom just outside the upper chest, elbows tucked at about 45-60 degrees from the torso with the forearms vertical, straight up to full lockout over the upper chest, then lowers them slowly back to the same deep bottom with the elbows still tucked. Full range of motion every rep. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm: ease out of the bottom, smooth through the middle, ease into the top, no pause and no jerk. The same two dumbbells stay in the hands the whole time and never change shape, size or colour: matte cobalt royal blue plates on every face with a thin brushed-gold rim ring, brushed steel handles. The emerald-green glow stays only on the upper chest. The camera is static and locked, front view, no zoom, no pan. Lighting is flat, even and shadowless: the body keeps exactly the same matte charcoal-grey skin tone, no shaded side, no specular change. Keep every colour exactly as in the start frame: flat dark charcoal grey backdrop with no vignette and no floor shadow, warm light-grey bench, matte-black shorts. No text, no logo.

## After download
Save the clips as `output/assets/day6_v5_wrong_720p.mp4` and `output/assets/day6_v5_right_720p.mp4`. Then check each clip:
- elbow angle held;
- full depth and full lockout;
- the same plates throughout;
- glow on the right muscle;
- no camera drift.

Then run `tools/startframe_gates.py` on sampled frames: plate faces ≥ 40 RGB from the backdrop throughout the motion.
