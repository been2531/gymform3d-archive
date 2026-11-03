# Day 4 template spec + feet fix (t637u, t638u) — 2026-11-03 KST

Owner: pt637u "이상한데? 위치가 발 밟는 곳과 / 그리고 day4 틀 사용하기로 했을텐데?", t638u (overlays see-through over body/equipment).
Sources inspected: `output/day4_romanian_deadlift_4step_master.mp4` (1080×1920, 30 fps, 20.0 s, AAC), `build_day4_rdl_master.py`,
`output/day4_4step_multiangle_master_sheet.jpg`, 1 fps contact sheet `output/_consistency/day4_spec/day4_master_1fps.jpg`,
`gymform_spec.json` (gymform-v5, locked), `gymform/{layout,render,native,ctx}.py`, `configs/day07_muscle2.json`, memory.

## 1. What the Day 4 template is (measured)
| Element | Day 4 (published 2026-10-29) | Source |
|---|---|---|
| Canvas | 1080×1920, 30 fps, 4 steps × 5.0 s = 20.0 s, hard cuts between steps (no blending/ghosting) | build_day4 `DURATION`, spec.format |
| Step order (Day 4) | 1 wrong bone X-Ray · 2 wrong muscle · 3 right bone X-Ray · 4 right muscle (cover = step 4 peak) | STEP_SPECS_DAY4 + sheet |
| Step order (locked from Day 6) | 1 wrong FRONT · 2 right FRONT · 3 wrong SIDE muscle · 4 right SIDE muscle; pairs 1/3, 2/4 same motion | spec.step_order (t515u/t521u/t556u) |
| 2-step day (t568u) | n_steps 2 = wrong, right; 10.0 s; **one best angle for both steps**; 2 thumbnails; no front/side pair | ctx.duration, tests/test_two_step.py, DAY7_IMPROVEMENTS #5, configs/day07_muscle2.json |
| Motion inside a step | cosine wave 0.5−0.5cos(2π·1.5u) over the clip slice (stretch→contraction→stretch→contraction), reversals at 1.67/3.33 s | spec.tempo_s |
| Top preview "capsule" | frosted-glass rounded pill (r 24) in the 3:4 grid safe zone, ROI x190–890 y232–424, bar y258–400; 4 slots at x 303/461/619/777 (2-step: 2 slots at 461/619, pill x370–710); each slot = that step's peak thumbnail; ACTIVE slot = coloured border (red wrong / green right) | b4.draw_compact_demic_top_preview |
| Exercise name | single-line Kaushan Script on the back wall, title ROI y410–565 (centre ≈ y458), white with soft shadow | b4.draw_single_line_kaushan_title, layout.TITLE_ROI |
| See-through rule (t638u, already locked) | where the mannequin OR equipment overlaps the title/preview they stay visible at **alpha 0.40** (Day 4 measured title-over-body median 0.30, preview ≥ 0.35); under the pill the 3D layer is NOT frosted and keeps 60% of its own pixels; equipment never cut/hidden | layout.composite_title_and_preview_v4, spec.ui.overlap_alpha |
| Wrong/right signal | **no ✕/✓ badges, no bottom text** ("zero 2D overlays"). Wrong = red/crimson glow + red active slot + orange-red callout; right = emerald glow + green slot + green callout | build_day4 header, sheet |
| Callout | one straight laser joint axis + 2 cross ticks + vertex ring(s), horizontal precision leader to a left text block: number 84–86 px at x=86, label 21 px (colour), sub-line 16 px; y ≥ 600 (below the title) | layout.draw_callout_day4 |
| Audio | edge-tts en-US-ChristopherNeural, one coaching line per step at 0.2/5.2/10.2/15.2 s + sub-bass beat + step chimes; from Day 6 the last line = spoken caption question | build_day4 vo_lines, gymform/tts.py, caption_rules |
| Backdrop / mannequin | flat charcoal 35/39/43, no props; mannequin skin lock p50 64 / p75 75 / p90 88; opaque matte-black shorts | spec.background/mannequin |
| From the GRADE (post) | backdrop flatten, tone curve + skin lock, glow recolour (Day 4 fibre shader / native_grade), laser + callout, title, preview pill, see-through compositing, TTS, step cuts, cover | native.py, muscle.day4_fibre, layout.py |
| From the CLIP | mannequin, shorts, equipment, pose/motion, the glow REGION (clip colour dominance is what gets recoloured) | spec.source_rule (native only, 0% 2D overpaint) |

## 2. Why the D7v2 rough cut was not Day 4 (root cause)
`roughcut/d7v2_roughcut_540p.mp4` was a hand-made ffmpeg cut: invented ✕/✓ badges + text pills, hook/slow-mo/punch-in, 11.8 s,
no title, no preview pill, no callout, no TTS, no native grade. The production 2-step Day 4 pipeline (`run_day.py` +
`configs/day07_muscle2.json`, n_steps 2) already exists and was not used. **Fix: every reel is built by the gymform
pipeline; REEL_FORMAT_V2's badges/hook/tail are withdrawn** (superseded by this file).

## 3. Conflicts with "2 clips, front only" and the proposal
- The locked 4-step plan needs a FRONT and a SIDE camera of the same motion (t556u). Two front-only clips cannot make steps 3/4 honestly; cropping or zooming the front clip is not a side view. → **The Day 4 template strictly needs side views for a 4-step day.**
- The template already has a 2-step mode (t568u) that keeps every Day 4 element (title, pill with 2 thumbnails, callouts, laser, see-through, TTS, grade) with exactly 2 generated clips and one angle. **Proposal: Day 7 = 2-step Day 4 template (10 s), 2 clips (wrong, right), elevated front-3/4.** Optional 20 s variant without new generations: replay each clip twice (steps 1/3 = wrong, 2/4 = right) with a target-zoom punch-in on steps 3/4 — but it must NOT be labelled a side view and the step_order lint (front/side) would have to be relaxed by the owner. Recommendation: 2-step.

## 4. Feet root cause (t637u)
Evidence: `output/_consistency/day4_mock/feet_fix_sheet.jpg`.
- Real Matrix Aura low row: the footplate sits directly in front of the pulley housing, leans back toward the tower and its face points along the cable axis at the seat; the seat is a small pad on a post at the far end of the rail, one leg length away; soles press on the plate face.
- Old base (`base_stretch_gen_raw.jpg`, 720×1280): the plate is rotated ~90° (its face points at the camera, long edge running from the pulley (≈x250,y760) down to (≈x380,y960)); the feet are at x≈315–395, y≈870–930, i.e. **on the floor/rail at the plate's lower-right edge, soles turned sideways, not on the face**. The seat is a separate flat bench, not on the rail.
- Prompt: the base prompt carried a single clause "feet flat on the footplate" at the end, with no geometry (plate facing the seat, heel on the lower lip, knee angle, seat-to-plate distance). The machine reference was the rejected REAR-view try, so the plate geometry was re-imagined for the new camera.
- QC: my D7v2 visual check noted "near foot at the footplate edge" and accepted it; none of G1–G10 measures foot contact.
- **Cause = under-specified prompt + wrong reference for the plate + no foot-contact check.** Fix rule: describe the plate orientation and sole contact explicitly, use the REAL photo as the machine reference, and do not use a frame with bad geometry as a reference (it anchors the fault: v3 try 1 re-used the old base and repeated it).

## 5. Fix (≤ 2 base tries) + verification
- Try 1 (`frames/base_v3_try1_rejected_feet_gen_raw.jpg`, ref = Matrix photo + old base): **rejected**, the same plate/feet geometry and a light border.
- Try 2 (`frames/base_v3_feet_gen_raw.jpg`, ref = Matrix photo + a mannequin-only crop): **soles pressed on the footplate face, heels on the lower lip, knees slightly bent, cable horizontal from the low pulley, stack with selector pin, warm-grey frame, cobalt plates.** Defects: a duplicate Matrix post seat on the rail in front of the bench and "MATRIX" lettering on the upright.
- Wrong start (`frames/v3_wrong_start_gen_raw.jpg` → graded once `v3_wrong_start.png` / `_720.png`): edit of try 2 that removes the extra seat and the lettering and adds red on the biceps, front delts and upper traps. Glow 4.8% (smaller than D7v2's 20%).
- Right start (`frames/v3_right_start_gen_raw.jpg` → `v3_right_start.png` / `_720.png`): edit of the wrong start (red → emerald on the lats band + delts), so the machine is identical. First attempt from the base (`v3_right_start_a_rejected_green_on_arms…`) put green on the arms and the stack pegs differed → rejected. Glow 10.6%. Caveat: green covers the whole deltoid cap (front and side delts too), not only the rear delts.
- Start gates: backdrop 34/38/42 far span 1.0 ✓, fig−bg 56.6 / 48.8 ✓, plates vs bg p05 38–39 ✓, bench ✓; **plate_faces_below40 2.86% / 2.66% > 2.0 ✗ (near miss)**. View ratio 0.621 / 0.624 (Δ 0.003). Machine-region SSIM wrong vs right 0.926.
- Not regenerated further (2-try cap); no videos generated.

## 6. Smoke mock (static, real pipeline)
`tools/day4_template_mock.py` turns each 720 start frame into a 12-frame static clip and runs the UNCHANGED pipeline
(native_grade, layout, `configs/day07_muscle2.json`, n_steps 2), frame 0 of each step. Mock framing: target_zoom removed
(floor-line framing, scale 1.0, dy −70) because the Day 7 target_zoom (body_bbox_x) cut the tower off.
- `output/_consistency/day4_mock/mock_day4_template_540.jpg` — wrong | right in the full Day 4 layout (whole machine in frame).
- `mock_day4_template_overlap_540.jpg` + `mock_overlap_zoom.jpg` — overlap stress case (shift x −170, y +60): the head runs under "Row"; the title stays see-through at 0.40 and the head remains visible. The tower is cut in this stress framing only.
- Open issues seen in the mock: (a) the Day 4 callout block (x=86) sits over the tower in this framing → breaks "callouts never overlap equipment"; needs `callout_ty`/x override or mirror (owner call); (b) the grade lifts the warm-grey frame to near-white; (c) the preview pill does not overlap anything in the normal framing (overlap only appears in the stress case).

## 7. Plan for clips (not started)
1. Owner OK on the v3 starts (feet, machine) + the 2-step Day 4 format.
2. Contraction keyframes as raw edits of the v3 starts (no regrade), 1 per form; GenerateVideo 720p 5 s, first = last = start, keyframes at 2.2/2.8 s; ≤ 1 retry each; G1–G10 + a new manual foot-contact check (soles on plate in f0/key/end).
3. Day 7 config: clips → the new mp4s, window = one full rep, framing = whole machine (no tower cut), callout moved off the tower.
4. `run_day.py` 2-step build (10 s, TTS, title, pill, see-through) → QC → sheet → owner review. Nothing published.
