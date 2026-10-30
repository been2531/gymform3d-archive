# Day 7: Seated Cable Row, muscle-only 2-step, palette A at the source (t612u)

**Mistake (muscle type, like Day 6):** pulling with the arms and shrugging. The biceps and upper traps take the load (red).
**Right:** chest up, shoulders down and back, shoulder blades squeezed, elbows driven back past the torso. The lats take it (green).
The joint-type mistake (lumbar rounding / lean-back rocking, `docs/MOTION_STANDARDS.md`) is NOT used: that would need the 4-step bone format. The earlier lean-back plan (commit 936fec0) is superseded.

**Start frames** (1080×1920 graded, plus `*_720.png` 720×1280; 9:16; elevated front-3/4 camera, cable column on the left; end of the pull = the most telling point, D14):
- `day7_wrong_start.png`: handle at the belly, shoulders shrugged to the ears, chest caved, slight lean back. Red glow on the biceps and upper traps.
- `day7_right_start.png`: torso upright, shoulders lowered, handle at the lower ribs, elbows back. Green glow on the lats (side of the ribcage under the armpits).
- Same machine in both: the right frame was generated from the wrong frame as its reference.

**How they were made:**
- **Generation:** Grok Bot image generation. Wrong took 2 tries (try 1 had a black seat). Right took 4 tries (tries 1–3 had a different machine, or glow on the pecs or obliques). The references were the Day 6 v5 start frame (body) and `docs/plate_palette_A_reference.jpg`. Raw outputs: `*_gen_raw.jpg`.
- **Global grade v2:** `tools/startframe_grade.py <raw> <out> 38.3 67 35,39,43 80`, with no masks.

**Gates** (`tools/startframe_gates.py` → `start_frame_gates.jsonl`), wrong / right:

| Gate | Wrong | Right | Target |
|---|---|---|---|
| Backdrop RGB | 35,39,43 | 35,39,43 | Day 4 studio, PASS |
| Figure − backdrop | 38.3 | 37.0 | 33–41, PASS |
| Skin p50 | 63.2 | 62.7 | Day 4 63.9, PASS |
| Bench/frame vs backdrop | 74.7 | 67.5 | ≥ 15, PASS |
| Bench/frame vs skin | 31.2 | 25.2 | ≥ 10, PASS |
| Backdrop span, ≥ 60 px from objects | 1.0 | 2.0 | PASS |
| Backdrop span, all blocks | 4.0 | 5.7 | ≤ 4; right FAILS (block next to the machine, same as Day 6 v5 right 5.4) |
| Plate faces vs backdrop, median | 67.0 | 70.7 | ≥ 40, PASS |
| Plate faces vs backdrop, p05 | 22.2 | 22.4 | ≥ 40, FAIL |
| Plate face area < 40 RGB | 11.8% | 12.3% | ≤ 2%, FAIL |

- **Plate failures:** the dark seams between the stacked plates and the pin-hole column. They are structural, not a cobalt face that melts into the backdrop (crop check).
- **Known source limits:**
  - no visible gold rim on the stack;
  - the footplate, top pulley housing and pulley are dark, not warm grey;
  - the right-frame lat glow is smaller than Day 4 size (the prompt asks for a large glow).

One grok.com image-to-video take per clip, only when the owner approves (shared quota).

## WRONG (image-to-video, start frame `day7_wrong_start.png`)
Smooth seated cable row done with the arms and a shrug. The anatomical mannequin starts at the end of the pull with the steel V-handle at the belly and both shoulders shrugged up to the ears, then lets the handle travel forward until both arms are fully straight and reaching at full stretch, then yanks the handle back to the belly mostly by bending the arms, shrugging the shoulders up high toward the ears, chest caved and upper back rounded, elbows stopping beside the torso. Full range of motion every rep, from full arm extension to the handle touching the belly. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm: ease out of the stretch, smooth through the middle, ease into the contraction, no pause and no jerk. The torso stays seated on the same seat and the feet stay on the footplate. The same machine, cable and V-handle stay identical the whole time and never change shape, size or colour: warm light-grey frame and seat, weight stack of matte cobalt royal blue plates with a thin brushed-gold rim, brushed steel V-handle and guide rods. The weight stack rises and lowers with each pull. The large bright red glow stays only on the biceps and the upper trapezius. The camera is static and locked, elevated front three-quarter view, no zoom, no pan. Lighting is flat, even and shadowless: the body keeps exactly the same matte charcoal-grey skin tone, no shaded side, no specular change. Keep every colour exactly as in the start frame: flat dark charcoal grey backdrop with no vignette and no floor shadow, warm light-grey machine, matte-black shorts. No text, no logo.

## RIGHT (image-to-video, start frame `day7_right_start.png`)
Smooth, controlled seated cable row with the back doing the work. The anatomical mannequin starts at the end of the pull with the steel V-handle at the lower ribs, chest up, shoulders down and the elbows behind the torso, then lets the handle travel forward until both arms are fully straight and the shoulder blades spread at full stretch, torso staying upright, then rows the handle back to the lower ribs by driving both elbows straight back past the torso, shoulders pressed down away from the ears, shoulder blades squeezed together, chest staying up. The torso stays upright and still, no leaning back and no shrug. Full range of motion every rep, from full arm extension to the handle touching the lower ribs. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm: ease out of the stretch, smooth through the middle, ease into the contraction, no pause and no jerk. The feet stay on the footplate. The same machine, cable and V-handle stay identical the whole time and never change shape, size or colour: warm light-grey frame and seat, weight stack of matte cobalt royal blue plates with a thin brushed-gold rim, brushed steel V-handle and guide rods. The weight stack rises and lowers with each pull. A large, bright emerald-green glow covers the whole latissimus dorsi on both sides of the ribcage and the rear deltoids, and stays only there. The camera is static and locked, elevated front three-quarter view, no zoom, no pan. Lighting is flat, even and shadowless: the body keeps exactly the same matte charcoal-grey skin tone, no shaded side, no specular change. Keep every colour exactly as in the start frame: flat dark charcoal grey backdrop with no vignette and no floor shadow, warm light-grey machine, matte-black shorts. No text, no logo.

## After download
Save the clips as `output/assets/day7_wrong_raw.mp4` and `output/assets/day7_right_raw.mp4`. Then check each clip:
- full arm extension at the stretch and full contraction in both clips;
- wrong: shoulders shrug up, and the elbows stop at the torso;
- right: shoulders stay down, the elbows go past the torso, and the trunk range is ≤ 10°;
- glow on the named muscle, ≥ 20k px at the peak;
- the same machine and stack throughout, and no camera drift.

Then set the `i0`/`i1` windows in `configs/day07_muscle2.json`, and run `tools/startframe_gates.py` on sampled frames (plates ≥ 40 RGB from the backdrop throughout the motion).
