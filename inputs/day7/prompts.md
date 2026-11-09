# Day 7: Seated Cable Row, muscle-only 2-step, palette A at the source

## v3 start frames (t612/t613: fix the source-side preflight fails; clips by Grok Bot GenerateVideo, not grok.com)

- **Frames:** `day7_v3_{wrong,right}_start.png` (1080) and `*_720.png`. Each is a GenerateImage edit of the v2 frame, so the pose, camera and machine are the same as v2. The Day 4 step-2 peak frame was used as the glow/striation reference. Wrong took 3 tries; try 1 is kept, because try 2 removed the cobalt plate and try 3 removed the glow. Right took 3 tries; try 3 is kept.
- **Grade:** `tools/startframe_grade.py <raw> <out> 38.3 67 35,39,43 80`. Gates → `start_frame_gates_v3.jsonl`.
- **Known limits:**
  - Wrong: the red also tints the pecs and front delts faintly, and the traps read weaker than the biceps.
  - Right: the green also covers the side delts, and there is a soft green glow halo on the upper-arm edge.
  - The machine did not become visibly lighter in the edits; the footplate stays dark.

### WRONG v3 (image-to-video, start frame `day7_v3_wrong_start_720.png`)
Seated low cable row, done wrong with the arms and a shrug. Start from the stretch exactly as in the frame: arms straight forward, both hands together on the one single V-handle at knee height. Pull the handle straight back along the cable line to the belly, mostly by bending the elbows, with both shoulders shrugging up toward the ears; the elbows stop beside the torso. Then let it travel straight forward again to full arm extension. 1.5 reps in about 5 seconds, smooth and even. The handle moves only forward and back along the straight cable line at belly height or lower; the hands stay together on the single handle; the arms never rise above the shoulders and never spread sideways; no overhead movement. Feet stay on the footplate, seat and machine stay identical. The large bright red glow stays exactly where and as large as in the start frame, covering the whole biceps and the whole upper traps, with the muscle fibres visible. Locked static camera, no zoom, no pan. Flat, even, shadowless lighting; same matte charcoal-grey skin, flat dark charcoal backdrop, light warm-grey machine, cobalt gold-rim plate, matte-black shorts. No text, no logo.

### RIGHT v3 (image-to-video, start frame `day7_v3_right_start_720.png`)
Seated low cable row, done right with the back. Start from the stretch exactly as in the frame: arms straight forward, both hands together on the one single V-handle at knee height, chest up, shoulders down. Row the handle straight back along the cable line to the lower ribs by driving both elbows back close to the sides and past the torso, squeezing the shoulder blades together; the torso stays upright and still, shoulders stay down. Then let it travel straight forward again to full arm extension. 1.5 reps in about 5 seconds, smooth and even. The handle moves only forward and back along the straight cable line at lower-rib height or lower; the hands stay together on the single handle; the arms never rise above the shoulders and never spread sideways; no overhead movement. Feet stay on the footplate, seat and machine stay identical. The large bright emerald-green glow stays exactly where and as large as in the start frame, covering the whole lats on both sides and the rear delts, with the muscle fibres visible, never on the chest, abs or arms. Locked static camera, no zoom, no pan. Flat, even, shadowless lighting; same matte charcoal-grey skin, flat dark charcoal backdrop, light warm-grey machine, cobalt gold-rim plate, matte-black shorts. No text, no logo.

## v2 start frames

**Why v1 failed** (clips kept as `output/assets/day7_{wrong,right}_raw_v1.mp4`):
- Wrong turned into an overhead pulldown and an arm spread. Right turned into a chest-fly spread with green over the whole torso. Both split the V-handle in two.
- **Cause:** the tall stack tower beside the figure read as a lat pulldown, the cable path was unclear, and the frames started at the contraction, so the model couldn't tell the pull direction.

**What changed in v2:**
- A classic low-row station: rail, seat, footplate, then a LOW compact stack box right in front of the feet with one low pulley. One straight cable runs from the pulley to the hands.
- No tower, no high pulley, no overhead bar.
- One single-piece V-handle with the hands together.
- Both frames start at the STRETCH: arms straight forward along the cable, hands at knee height.

**Start frames** (1080×1920 graded, plus `*_720.png` at 720×1280; elevated front-3/4 camera, stack at the lower right):
- `day7_v2_wrong_start.png`: red on the biceps.
- `day7_v2_right_start.png`: chest up, shoulders down, emerald on the lats and rear delts only. It was made from the wrong frame as its reference, so the machine is identical.
- Raw outputs: `*_gen_raw.jpg`.
- Grade: `tools/startframe_grade.py <raw> <out> 38.3 67 35,39,43 80`.
- Wrong took 4 tries: tries 1–3 kept a contraction pose from the reference image. Try 4 used only the Day 6 body reference. Right took 2 tries: try 1 also had green on the biceps.

**Known limits:**
- The wrong frame has no visible upper-trap glow and only a mild shrug. The prompt asks for both.
- The wrong plate carries a small hex emblem, and the right plate has none.
- The footplate is dark grey.
- The stack is one large cobalt gold-rim plate in an open box rather than a stack of plates.

**Gates** (`tools/startframe_gates.py` → `start_frame_gates_v2.jsonl`), wrong / right:

| Gate | Wrong | Right | Target |
|---|---|---|---|
| Overall | PASS | PASS | |
| Backdrop RGB | 35,39,43 | 35,39,43 | PASS |
| Figure − backdrop | 39.3 | 38.6 | 33–41, PASS |
| Skin p50 | 64.1 | 63.2 | Day 4 63.9, PASS |
| Machine vs backdrop | 68.1 | 61.0 | ≥ 15, PASS |
| Machine vs skin | 22.7 | 16.9 | ≥ 10, PASS |
| Backdrop span, ≥ 60 px from objects | 1.0 | 3.1 | PASS |
| Backdrop span, all blocks | 5.7 | 5.6 | ≤ 4, over (blocks next to the machine, as in v1/Day 6) |
| Plates vs backdrop, p05 / median | 41.3 / 79.7 | 46.0 / 82.5 | ≥ 40, PASS |
| Plate faces < 40 RGB | 0.0% | 0.0% | ≤ 2%, PASS |

One grok.com take per clip.

### WRONG v2 (image-to-video, start frame `day7_v2_wrong_start.png`)
Seated low cable row, done wrong with the arms and a shrug. Start from the stretch exactly as in the frame: arms straight forward, both hands together on the one single V-handle at knee height. Pull the handle straight back along the cable line to the belly, mostly by bending the elbows, with both shoulders shrugging up toward the ears; the elbows stop beside the torso. Then let it travel straight forward again to full arm extension. 1.5 reps in about 5 seconds, smooth and even. The handle moves only forward and back along the straight cable line at belly height or lower; the hands stay together on the single handle; the arms never rise above the shoulders and never spread sideways; no overhead movement. Feet stay on the footplate, seat and machine stay identical. The red glow stays exactly where and as large as in the start frame, on the biceps and upper traps only. Locked static camera, no zoom, no pan. Flat, even, shadowless lighting; same matte charcoal-grey skin, flat dark charcoal backdrop, warm-grey machine, cobalt gold-rim plate, matte-black shorts. No text, no logo.

### RIGHT v2 (image-to-video, start frame `day7_v2_right_start.png`)
Seated low cable row, done right with the back. Start from the stretch exactly as in the frame: arms straight forward, both hands together on the one single V-handle at knee height, chest up, shoulders down. Row the handle straight back along the cable line to the lower ribs by driving both elbows back close to the sides and past the torso, squeezing the shoulder blades together; the torso stays upright and still, shoulders stay down. Then let it travel straight forward again to full arm extension. 1.5 reps in about 5 seconds, smooth and even. The handle moves only forward and back along the straight cable line at lower-rib height or lower; the hands stay together on the single handle; the arms never rise above the shoulders and never spread sideways; no overhead movement. Feet stay on the footplate, seat and machine stay identical. The emerald-green glow stays exactly where and as large as in the start frame, on the lats and rear delts only, never on the chest, abs or arms. Locked static camera, no zoom, no pan. Flat, even, shadowless lighting; same matte charcoal-grey skin, flat dark charcoal backdrop, warm-grey machine, cobalt gold-rim plate, matte-black shorts. No text, no logo.

### After download
Save the clips as `output/assets/day7_wrong_raw.mp4` and `output/assets/day7_right_raw.mp4`, overwriting the v1 working copies (the v1 clips are kept as `*_raw_v1.mp4`).

Then check each clip:
- one handle, with the hands together;
- the handle path stays horizontal;
- full extension and full contraction;
- glow on the named muscle;
- same machine and no camera drift.

## v1 (t612u): FAILED on grok.com, kept for reference

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

### WRONG (image-to-video, start frame `day7_wrong_start.png`)
Smooth seated cable row done with the arms and a shrug. The anatomical mannequin starts at the end of the pull with the steel V-handle at the belly and both shoulders shrugged up to the ears, then lets the handle travel forward until both arms are fully straight and reaching at full stretch, then yanks the handle back to the belly mostly by bending the arms, shrugging the shoulders up high toward the ears, chest caved and upper back rounded, elbows stopping beside the torso. Full range of motion every rep, from full arm extension to the handle touching the belly. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm: ease out of the stretch, smooth through the middle, ease into the contraction, no pause and no jerk. The torso stays seated on the same seat and the feet stay on the footplate. The same machine, cable and V-handle stay identical the whole time and never change shape, size or colour: warm light-grey frame and seat, weight stack of matte cobalt royal blue plates with a thin brushed-gold rim, brushed steel V-handle and guide rods. The weight stack rises and lowers with each pull. The large bright red glow stays only on the biceps and the upper trapezius. The camera is static and locked, elevated front three-quarter view, no zoom, no pan. Lighting is flat, even and shadowless: the body keeps exactly the same matte charcoal-grey skin tone, no shaded side, no specular change. Keep every colour exactly as in the start frame: flat dark charcoal grey backdrop with no vignette and no floor shadow, warm light-grey machine, matte-black shorts. No text, no logo.

### RIGHT (image-to-video, start frame `day7_right_start.png`)
Smooth, controlled seated cable row with the back doing the work. The anatomical mannequin starts at the end of the pull with the steel V-handle at the lower ribs, chest up, shoulders down and the elbows behind the torso, then lets the handle travel forward until both arms are fully straight and the shoulder blades spread at full stretch, torso staying upright, then rows the handle back to the lower ribs by driving both elbows straight back past the torso, shoulders pressed down away from the ears, shoulder blades squeezed together, chest staying up. The torso stays upright and still, no leaning back and no shrug. Full range of motion every rep, from full arm extension to the handle touching the lower ribs. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm: ease out of the stretch, smooth through the middle, ease into the contraction, no pause and no jerk. The feet stay on the footplate. The same machine, cable and V-handle stay identical the whole time and never change shape, size or colour: warm light-grey frame and seat, weight stack of matte cobalt royal blue plates with a thin brushed-gold rim, brushed steel V-handle and guide rods. The weight stack rises and lowers with each pull. A large, bright emerald-green glow covers the whole latissimus dorsi on both sides of the ribcage and the rear deltoids, and stays only there. The camera is static and locked, elevated front three-quarter view, no zoom, no pan. Lighting is flat, even and shadowless: the body keeps exactly the same matte charcoal-grey skin tone, no shaded side, no specular change. Keep every colour exactly as in the start frame: flat dark charcoal grey backdrop with no vignette and no floor shadow, warm light-grey machine, matte-black shorts. No text, no logo.

### After download
Save the clips as `output/assets/day7_wrong_raw.mp4` and `output/assets/day7_right_raw.mp4`. Then check each clip:
- full arm extension at the stretch and full contraction in both clips;
- wrong: shoulders shrug up, and the elbows stop at the torso;
- right: shoulders stay down, the elbows go past the torso, and the trunk range is ≤ 10°;
- glow on the named muscle, ≥ 20k px at the peak;
- the same machine and stack throughout, and no camera drift.

Then set the `i0`/`i1` windows in `configs/day07_muscle2.json`, and run `tools/startframe_gates.py` on sampled frames (plates ≥ 40 RGB from the backdrop throughout the motion).
