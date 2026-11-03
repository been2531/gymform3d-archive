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
