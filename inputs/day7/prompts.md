# Day 7 — Seated Cable Row, muscle-only 2-step (wrong = rocking lean-back, right = upright torso)

Motion standard: `docs/MOTION_STANDARDS.md` → "Seated Cable Row (Day 7)". ACE and ACSM H&FJ (free) say the torso stays vertical with no leaning back. The mistake is the torso rocking more than 15° per rep.
Start frames, 1080×1920, at the most telling point (end of the pull):
- `day7_wrong_start.png`: torso leaned back about 35° (2D, measured), red glow on the lower-back erectors.
- `day7_right_start.png`: torso upright (about 14° in 2D from this rear-3/4 camera; the 0–15° band), green glow on the lats and mid-back.
The machine, cable, V-handle, seat, foot plates, body and backdrop are identical in both. Re-made flat for t592u (no shading): backdrop block span is 3 / 1 lum (the old frames were 28 / 21, now kept as `*_shaded_v1.png`). Note: the flat re-make turned the wrong frame into a near-side view, while the right frame stays rear-3/4. Run one grok.com take per clip.

Colour separation (t600u–t603u, playbook D17). Required for the start frames and both prompts:
- the mannequin keeps the Day 4 silver skin tone (not brighter);
- the backdrop is flat and seamless, clearly darker than the mannequin, a very slight cool blue-grey, with no vignette and no floor shadow;
- metal (handle, cable, weight stack) is bright brushed steel;
- the seat, frame and foot plates are warm mid-grey, not black;
- red/emerald glow is only on the named muscle, never on the backdrop or machine.

**Start frames:** `day7_*_start.png` still need a re-make with this palette. They were NOT regenerated (no grok.com usage allowed, t600). Before generation, check them with QC (sep): skin p75 − backdrop ≥ 33, equipment ≥ 15 RGB from both the backdrop and the skin.

## WRONG (image-to-video, start frame `day7_wrong_start.png`)
Smooth seated cable row with a big rocking cheat. From the leaned-back end position, the mannequin lets the V-handle travel forward, swinging the whole torso far forward over the thighs with a rounded back. Then he yanks the handle back to the belly by throwing the torso far back to about 45 degrees behind vertical, chin up. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm. The large torso swing is the main motion. The red glow stays only on the lower back. The backdrop is flat, darker than the silver body, faint cool blue-grey; the handle and cable are bright brushed steel; the seat and frame are warm mid-grey, never black. The camera is static and locked. Lighting is flat, even and shadowless: the body is one even matte grey with no shaded side, and the background is one uniform flat charcoal with no vignette, spotlight, gradient or floor shadow. The machine, cable, handle, body and background stay identical. Nothing else enters the frame.

## RIGHT (image-to-video, start frame `day7_right_start.png`)
Smooth, strict seated cable row. The torso stays perfectly upright and still, shoulders stacked over the hips, the whole time. From the end position, the arms extend forward slowly until the elbows are straight and the shoulder blades spread. Then he rows the V-handle back to the lower ribs, driving the elbows past the torso and squeezing the shoulder blades. Only the arms and shoulder blades move. Do 1.5 reps in about 5 seconds with an even, wave-like rhythm. The green glow stays on the lats and mid-back. The backdrop is flat, darker than the silver body, faint cool blue-grey; the handle and cable are bright brushed steel; the seat and frame are warm mid-grey, never black. The camera is static and locked. Lighting is flat, even and shadowless: the body is one even matte grey with no shaded side, and the background is one uniform flat charcoal with no vignette, spotlight, gradient or floor shadow. The machine, cable, handle, body and background stay identical. Nothing else enters the frame.

## After download
Save as `output/assets/day7_wrong_raw.mp4` and `output/assets/day7_right_raw.mp4`. Then check:
- trunk angle range: wrong > 15°, right ≤ 10°;
- full arm extension at the stretch in both clips;
- glow on the named muscle;
- static camera;
- the same machine and handle in both clips.
