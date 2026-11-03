# D7 seated cable row — machine match analysis + concept (t639u, t640u–t641u) — 2026-11-03 KST

Owner: pt639u "운동기구 사진과 일치해? 구상부터 해 분석부터"; t640u–t641u: tower, footplate, feet, cable, seat and floor rail must lie on
ONE straight line (the machine's long axis) exactly like the real photo; athlete faces the tower head-on; no invented changes.
Analysis only — **no images or videos were generated in this step.**
Evidence: `output/_consistency/machine_ref/machine_match_sheet.jpg`, references + sources in `output/_consistency/machine_ref/SOURCES.md`.
Our frame measured = `smoke_d7v2/frames/v3_wrong_start_720.png` (720×1280; right start is an edit of it, same machine).
Real frame measured = `ref_matrix_aura_low_row_g3ms53_silver.jpg` (1200×1200).


## 0. Hard rules (t643u, t644u) — override everything below
- **Machine geometry is FIXED** = the real Matrix G3-MS53 photo. Never bend, rotate or re-arrange any part (footplate, rail,
  seat, tower, pulley) to show an angle. Only the CAMERA moves. Every frame gets the check "machine geometry fixed".
- **Cameras are defined relative to the fixed machine** (long axis = tower → footplate → rail → seat post):
  - SIDE view: camera beside the rail, perpendicular to the long axis, at seat-to-shoulder height (the real photo's view).
  - FRONT view: camera at or behind the tower end, looking down the long axis at the athlete's front; machine unchanged
    (the tower is then in the foreground/edge of frame, the athlete faces the camera).
  - REAR view: camera behind the seat end on the axis, looking toward the tower over the athlete's back.
  - Any 3/4 = a camera position between these, still around the unchanged machine.
  Prompts state the camera this way ("camera beside the rail, perpendicular to the long axis"), never "turn the machine".

## 1. Component comparison
| # | Component | Real (Matrix Aura Low Row G3-MS53) | Ours (v3) | Verdict |
|---|---|---|---|---|
| **1** | **Long axis** | tower base (330,940) → footplate (560,980) → rail → seat post (835,1030): ONE straight line, ~10° in frame; athlete sits on it facing the tower | rail axis (200,880)→(575,990) ≈ 16° down-right; an invented 4-leg bench at ≈ 19° UP-right (≈ 35° off the rail) and NOT on the rail; legs hip (560,700) → feet (290,830); cable (240,680)→(335,560). 4 different axes | **MISMATCH (critical)** — machine deformed |
| 2 | Tower | 2 round uprights, curved top cap, ~240 cm, full height | 2 square-ish uprights, flat cap, ≈ 700 px tall | shape ≈ match; height vs human wrong (see 12) |
| 3 | Top pulley | 2 pulleys under the cap | 2 pulleys under the cap | match |
| 4 | Weight stack | front of the tower, low (lower third), ≈ 20 plates (134 kg / 295 lb), black | same place, ≈ 20 cobalt plates (palette A, intended recolour) | match (colour = account rule) |
| 5 | Guide rods | 2 chrome rods above the stack | 2 rods | match |
| 6 | Selector pin / horns | orange pin on the top plate; 2 black weight horns left of the stack | orange pin ✓; horns **deleted** by our seat-removal edit | partial mismatch |
| 7 | Low pulley | housing on the tower front, just above the top edge of the footplate (~35–40 cm high), on the axis | housing at the plate top ✓, but the athlete is off-axis | location match, alignment broken |
| 8 | Cable path | stack → top pulleys → down inside the tower → low pulley → straight out along the axis, near-horizontal to the handle at the belly | rises ≈ 50° in frame to a chest-high handle, diagonal to the body | **MISMATCH** |
| 9 | Footplate | ONE large flat ridged plate, mounted on the rail directly in front of the tower, leaning back toward the tower ~60–65°, face toward the seat, centred on the axis | split into 2 shoe-shaped pads, feet sunk into them | **MISMATCH** |
| 10 | Seat | small square pad (~40 × 30 cm) on ONE post at the far end of the rail, ~45 cm high | long flat bench (~2× longer) on 4 legs, beside the rail; the real rail-end seat was REMOVED by our edit (I misjudged it as a duplicate) | **MISMATCH (invented part)** |
| 11 | Floor rail | one rectangular tube from the tower base to the seat post (machine length 230 cm) | runs past the plate and ends in empty floor (no seat on it) | **MISMATCH** |
| 12 | Proportions | seated adult head ≈ 55% of tower height (≈ 135 / 240 cm) | head at ≈ 80% of tower height → mannequin ≈ 1.45× too big for the machine | **MISMATCH** |
| 13 | Handle | double-D / V (triangle) row handle on a snap hook | closed rectangular frame (2 grips + 2 side bars) | **MISMATCH** |
| 14 | Athlete posture | on the seat, on the axis, facing the tower; both soles flat on the plate, knees soft (~15–25°); torso upright; neutral grip on the V handle; handle travels along the axis from arms-length to the belly (navel/lower ribs), elbows close to the sides and past the torso | sits on the invented bench, body yawed toward the camera, feet in pads, handle at chest height | **MISMATCH** |

Root cause of the geometry drift: our base was re-imagined from a text prompt + a previous (wrong) generation; each edit
"fixed" one thing and invented another (bench, pads, handle, horns). The real photo was never the edit source.

## 2. Target machine spec (copy, no invention)
Matrix Aura Low Row G3-MS53 exactly as in `ref_matrix_aura_low_row_g3ms53_silver.jpg`: 2-upright round-tube tower with curved
top cap and 2 top pulleys, stack on 2 chrome guide rods with orange selector pin and 2 weight horns, low-pulley housing on the
tower front just above the footplate, ONE ridged footplate on the rail leaning toward the tower, one rectangular floor rail,
small square seat pad on one post at the rail end, V/triangle handle on the cable. Only allowed changes = account colours:
frame warm grey, stack palette A cobalt (+ thin gold edge), pads black; logo text removed. Nothing added, removed, moved or bent.

## 3. Camera
- **True side view perpendicular to the long axis, camera at about seat-to-shoulder height (slight elevation 5–10°), rotated
  ≤ 15° toward the athlete's BACK.** The axis then reads as one near-horizontal line (tower → plate → feet → rail → seat), the
  cable is a clearly visible straight line from the low pulley to the handle, the elbow travel past the torso is readable, and
  the slight back rotation shows the lats/rear delts and part of the mid back. Pure front/3-4-front (our old view) hides the
  axis and the cable — rejected.
- Generate from the reference's own side (tower left, seat right, as the Matrix photo); in the Day 4 pipeline set the clip
  `mirror` flag (whole-frame flip, Day 4 precedent t543u) so the **tower ends on the RIGHT**: that keeps the Day 4 callout
  block (x = 86) over empty backdrop behind the athlete's back instead of over the tower. A whole-frame flip is a real
  viewpoint (the machine is symmetric about its axis), not a deformation.
- Framing: whole machine in frame (230 cm long, 240 cm tall); 9:16 → machine ≈ 900 px wide, tower ≈ 950 px tall, bottom
  of rail ≈ y 1500; skin/size gates unchanged.

## 4. Pose (both forms on the same machine, same camera)
| | Start (stretch) | Contraction |
|---|---|---|
| WRONG | soles on the plate, knees soft, arms straight, torso slightly forward | handle pulled to the CHEST with the arms: elbows flare out and stay in front of / level with the torso, shoulders shrug up to the ears, head forward; torso upright (no lean-back so the only fault shown is arm pull + shrug) |
| RIGHT | same as wrong start (identical frame except glow) | chest up, shoulders down and back, scapulae retracted, elbows driven straight back close to the sides and clearly PAST the back line, V handle at the navel/lower ribs, torso upright and still, knees soft, soles on the plate |
Handle travels on the cable line (axis) only; feet never leave the plate; seat never moves.

## 5. Glow regions
- WRONG (red): upper trapezius (neck to acromion), biceps, front (anterior) deltoid only.
- RIGHT (emerald): latissimus dorsi (wing from armpit to the waist along the ribcage side), rhomboids / middle trapezius
  (between the shoulder blades — partly visible with the ≤ 15° back rotation), posterior deltoid only.
  **No whole-shoulder cap** (v3 right had green on the full deltoid incl. front/side heads — reject), nothing on arms, abs, pecs, neck.

## 6. Title, preview and callout placement (Day 4 template, 2-step mode)
- After the mirror flip: tower x ≈ 830–1040, athlete centre x ≈ 560–720, empty backdrop x ≈ 60–480 behind the athlete.
- Title (Kaushan, y 410–565, centred) and preview pill (x 370–710, y 232–424): the tower top (≈ y 500) may reach the title's
  right end → see-through alpha 0.40 rule applies (t638u); keep the tower top below y 430 to stay clear of the pill.
- Callout text block (x = 86, y ≥ 600) sits over the empty backdrop behind the athlete; leader to the shoulder joint (callout
  axis torso → shoulder → elbow). It never crosses the tower or the cable.

## 7. Prompt / edit strategy
1. **Edit the REAL photo, don't describe the machine.** IMAGE 1 = `ref_matrix_aura_low_row_g3ms53_silver.jpg` (or the side
   view) as the edit source: "keep every machine part, its position, angle and proportion exactly; recolour only (frame warm
   grey, plates cobalt, pads black), remove the logo text, replace the white studio with flat charcoal 35/39/43."
   This step outputs an EMPTY machine plate (no person) — check it against the photo component by component (table §1)
   before any person is added.
2. Then add the mannequin as a second edit of that plate: IMAGE 1 = empty machine plate, IMAGE 2 = mannequin crop for body/tone,
   IMAGE 3 = REP posture photo; explicit: seated on the rail-end pad, on the axis, facing the tower, soles flat on the plate,
   knees ~20°, V handle at arm's length, cable straight from the low pulley; mannequin height so the seated head ≈ 55% of the
   tower height.
3. Glow as a third, glow-only edit (wrong and right each from the same no-glow athlete frame); contraction keyframes as
   pose edits of those, never using a previous faulty generation as a reference.
4. Verification per frame (before any video): axis check (tower base, plate centre, foot, seat post within ±15 px of one
   line), cable straight from the pulley, plate count/horns/pin present, seat = single post, handle = V, head/tower ratio
   0.50–0.60, glow regions per §5. Owner sees the empty plate + start frames before videos.

## 8. Risks
- The generator re-imagines geometry during recolour or when the person is added (bench re-invented, plate split) → empty-plate
  step + axis check; reject on any invented part, max 2 tries per step, then report instead of iterating blindly.
- Side view hides the far arm and most of the mid back → rhomboid glow small; accept or use the ≤ 15° back rotation.
- Video motion bends the rail/cable or slides the seat; feet lift off the plate → first = last = start pins + new axis/foot gate.
- Mannequin scale vs machine drifts between wrong and right → both from one athlete frame; G7/G10 + head/tower ratio.
- Mirror flip: any text becomes mirrored (none allowed anyway); the pipeline flips keypoints with the frame.
- Tower top under the title band → see-through rule; preview pill must stay clear.
- Reference images are third-party (product photos) — internal reference for edits only; the published reel shows our recoloured render.
- Box load: generation is remote; local checks are light (nice -n 19).

## 9. Image run 1 (t642u, 2026-11-03 ~16:55 KST) — result
- Step 1 empty machine (edit of the real photo, try 1): PASS — same parts/positions (bbox 279–962 × 113–1046 vs real 280–934 × 115–1053),
  one axis, single footplate, rail-end seat, horns + pin + rods kept, recoloured, logo removed; gold ring faint.
- Step 2 mannequin: FAIL after 2 tries (stopped per rule). Seat/facing/feet/axis PASS; the handle stays a rectangular frame
  (not V), the cable starts at the stack instead of the low pulley, head/tower 0.70 / 0.69 (target 0.50–0.60).
  Machine kept by eye; SSIM vs empty: tower 0.83 / 0.70, rail+seat 0.66 / 0.58.
- Step 3 glow edits and the Day 4 mock: not run (blocked on step 2).
- Sheet: `output/_consistency/d7m/d7m_machine_first_sheet.jpg`.
- Next idea (needs owner OK, not executed): split step 2 — (a) put the V handle and the straight low-pulley cable into the EMPTY
  machine first (machine-only edit, easier to verify), (b) then add a smaller mannequin with an explicit size anchor
  ("his seated head is level with the top of the weight stack's guide-rod housing / halfway up the tower").

## One-pass plan (t648u–t651u)

**Stills, locked (no more generation):**
- Machine: `output/_consistency/d7m/empty_machine_t1_gen_raw.jpg` (Matrix G3-MS53 recolour, side camera). Every
  frame below is composited back on it, so the machine pixels are identical in all of them.
- Mannequin: `mannequin_2b_t2_composite.jpg` (glow-free, seated, soles on the footplate).
- Wrong glow source: `wrong_start_t1_composite.jpg`. Right glow source: `right_start_t2_composite.jpg`. Right t1
  fails because it has no glow between the shoulder blades, and the clamp can't add any.
- Pipeline inputs (720x1280): `wrong_start_t1_720x1280.png`, `right_start_t2_720x1280.png`.

**Look = Day 4 by code (configs/day07_muscle2.json):**
- `native_muscle_shader: "day4"`: native grade with its linear glow recolour switched off. Muscle colour comes from
  the UNCHANGED `muscle.day4_fibre` on the source's own glow (act_dom). The existing `_glow_gain` calibration
  (render.load_inputs, Day 4 p50 ref) still applies.
- `muscle_glow_clamp: true` + `target_regions`: `gymform/region_glow.py` builds allowed-region polygons from the
  same frame's keypoints (torso frame hips→shoulders and half the shoulder width; arm capsules on the visible arm
  only). They multiply into act_dom, so w_act and core_w are zero outside: right = lats, rhomboids/mid-trap, rear
  delt, biceps ×0.35; wrong = erector columns, biceps, forearm, upper trap. Both flags are opt-in and default off,
  so Day 4/5/6 render the same as before.
- No bone steps on D7 (`bone_steps: []`), so the Day 4 bone treatment doesn't run here.

**Video: exactly 2 generations.** wrong = i2v from `wrong_start_t1_720x1280.png`, right = i2v from
`right_start_t2_720x1280.png`. In each, first = last = the pinned still and the contraction is mid-clip. The pinned
frames make the clip loop and keep the machine fixed. The clip's own glow moves with the muscles. The clamp is
rebuilt per frame from that frame's keypoints, which costs 45 ms. A static clamp from the pinned frame would cut
glow off the arms during the pull, so don't use one.

**Render: one `run_day.py --day 7` pass** (native_grade, no per-frame cut-out, `nice -n 19`, 2 workers = 1 per step).

**Measured on the stills** (box, nice 19, 1080x1920, `tools/time_d7_grade.py`):
- render_pose per frame: 1.02–1.40 s with day4_fibre + clamp, 0.62–1.01 s with the old linear recolour.
  Inside that, day4_fibre takes 0.33 s, features 0.08 s and the clamp 0.045 s.
- `native.prepare`: ≈0.5 s per source frame. load + icons: 17 s.

**Full 10 s reel estimate:** render_step already caches one render per unique pose (`cache[p_i]`), so the cost
scales with source frames (~50–70 per step), not the 150 output frames.
- Per step: 60 × (0.5 + 1.1) s ≈ 1.6 min. The 2 steps render in parallel.
- Plus load/icons/preflight ≈ 1 min and TTS/encode ≈ 0.5 min.
- Total ≈ 3–4 min, of which day4_fibre adds ≈ 0.5 min.

**Speedups that don't change the look:**
- Already in place: per-pose cache, per-clip kp cache, cached TTS, `--workers 2`.
- Pulse phase is fixed (0.75, like the Day 4 static) so the per-pose cache stays valid.
- Not worth more: the clamp is 4% of the frame cost, and the Day 4 shader blurs are the look.

## Simplified layout + Day 4 look calibration (t652u–t654u)

**Layout** (config flags, all additive and default off elsewhere):
- `hide_lines`: no laser/alignment line, no leader, no dot.
- `callout_single_label`: the big number plus ONE label (`ARMS + SHRUG` / `ELBOWS BACK`).
- `callout_text_y: 560`: text block at a fixed spot on the left, clear of the body and the tower.
- `callout_measure {metric: arm_torso, frame: last}`: the number is measured from keypoints, never typed in. It is
  the shoulder angle between trunk and upper arm on the visible arm, at the slice's last (= peak) frame.
  - Pinned stills (same start pose): wrong 37°, right 36°.
  - Real wrong/right values appear once the i2v slices end at the contraction. Earlier v2 clips measured 48° vs 68°.
- `frame_override {scale 1.53, dx 62, dy 247}`: one whole-frame crop/scale for every step, used for stills and
  clips alike, no cut-out. Mannequin and machine fill the 9:16 frame and the tower stays whole.

**Day 4 look, measured at equal torso scale** (`tools/day4_look_metrics.py`, sheet `d7m_day4_look_sheet.jpg`):
- Knobs only: `glow_gain_override {1: 12, 2: 30}` and `day4_input_scale {1: 0.45, 2: 0.25}`. day4_fibre unchanged.
- Native fix: with the linear recolour off, `native_keep_chroma` had been passing the source glow colour into the
  grey base (too bright, soft edge). It is now excluded on the body's glow regions only, so the machine's orange
  pin is the same in both steps.
- Mock fix: static clips are written lossless (FFV1). mp4v blocking had turned into a fibre-sharpened checkerboard.
- Matches Day 4: hue, glow V median (wrong 153 vs 157, right 190 vs 190), fibre contrast, wrong fibre density.
- Close: saturation (+6–15%).
- Gaps:
  - Right fibre density is about 5% of Day 4.
  - The core edge is 4–5× wider.
  - Both come from the source: the right t2 glow is smooth green, and the figure is small at 720p (2.3× upscale).
  - The fix is in the i2v source: prompt fibre striations visible through the glow and a larger figure (camera
    closer, tower top may crop). Re-measure with the same tool on the clips.

## t660u/t661u run: pins, 2 clips, gates, preflight (2026-11-03 KST)
- Pins (cropped 720x1280, matte-charcoal head folded into the one edit per still): `output/_consistency/d7m/pin_wrong_head_t1_720x1280.png`, `pin_right_fibre_head_t2_720x1280.png` (right t1 kept for comparison).
- Clips (GenerateVideo, 720p 5 s, first=last pinned, 0 retries): `output/assets/day7_wrong_raw_v3.mp4`, `day7_right_raw_v3.mp4`.
- Gates: `gates_v3.json` (G1-G6, G8-G10 pass; G7 view spread 0.051 vs 0.05 marginal, inherited from the pins; G9 wrong lean_back_rock 23 deg trunk range) and `gates_v3_extra.json` (machine unchanged mean|diff| <= 1.67, SSIM 0.99; foot-on-plate pixel gate pass; head vs body tone pass).
- Fix: `layout.measure_angle` read `gymform.render.KPS`, which is empty in the step workers (render runs as `__main__`), so preflight step 2 crashed; it now falls back to `__main__.KPS`.
- run_day with 2 workers was OOM-killed on the shared box (preflight step 2, no traceback); use `--workers 1`.
- Callout deg now = the measured values (wrong 78, right 75 elbow flexion), so (p) passes. frame_override dx 15 (scale < 1 breaks the backdrop/matte: out-of-source band).
- Preflight QC: FAIL 25/33. Remaining fails are Day 4-6 silver-mannequin / dumbbell definitions meeting the charcoal mannequin + cable machine: (f) skin tone (p75=p90=max=141: the measured "skin" is the light-grey machine frame), (sh), (j) "silver skin" arms, (sep)/(pl) dumbbell plates, (d) equipment footprint detector, (oc) bbox includes the machine (figure+machine span ~1000 px, cannot centre at scale 1), (mf) wrong-peak glow sits at x<300 where the QC UI exclusion (`ui[:, :300]`) removes it. No threshold changed; full render not run (render.py refuses without a PASS stamp; an owner override needs an explicit owner message).

## t662s/t663u: cable-machine QC profile + wrong-step red (2026-11-03 KST)
- `qc_profile: "cable_machine"` (DEFECT_PLAYBOOK D20, DAY4_TEMPLATE_SPEC section 8). Day 6 v5 preflight re-checked with the new qc.py: PASS 33/33 (unchanged).
- Wrong red (D21): `muscle_glow_clamp_chroma: "body"` removed the fibre-less source-red smear; `muscle_glow_wfg_from_glow` filled the colour gaps where the charcoal body sits at the backdrop level; knobs step 1 gain 14, input 0.45 (gain 14 -> 30 changed coverage < 1%: the limit is the clamp x source glow, not the gain). `native_deglow_base` tried and left OFF ((a) halo 525 px, (sh) worse).
- Preflight now FAIL 25/30 (was 25/33 without the profile): (oc) PASS (swept subject 54-1051, centre 552), (p) PASS, (sep)/(pl) N/A. Still failing, strict thresholds, on the REAL mannequin: (sh) dark<46 15-28 % p5 19-22 (Day 6: 0.6-2 %, p5 58), (f) p75 73-119 / p90 97-140 (band 69-81 / 81-95), (j) far-arm shadow p50 21 dark fill 436 px, (mf) 21-30 colour regions (<= 16), (d) step2_t35 handle footprint 1444 px (< 1500).
- Root cause of (sh)/(f)/(j): the generated D7 clips' charcoal mannequin is darker than the backdrop on its shaded side (source lum p5-p25 8-19 vs backdrop 45) with strong directional shading; one global tone curve cannot lift it to the Day 4 tone without lifting the swept-area backdrop (scale 0.9 test showed exactly that grey band). Day 4/6 sources were flat-lit and brighter than the backdrop. Fix needs a source with Day 4 flat light (new stills + clips) or owner override; no full render was made.
- Metrics (equal torso scale, peak frames vs Day 4): see `output/_consistency/d7m/d7_t663_look_metrics.json`, sheet `d7_t663_preflight_vs_day4_sheet.jpg`.
