# Option-B round 2: new stills + automatic pre-check (t693u). STOPPED before the video

**Result: FAIL.** The RIGHT start still failed the pre-check on all 3 attempts (1 + 2 targeted retries). Per the rule:
- the peak still was not generated;
- the video was not generated;
- no loop or grade was made.

Generations used: **3 GenerateImage, 0 GenerateVideo**. No grok.com, no paid API.

## Pipeline (reusable)
- **Pre-check:** `tools/still_precheck.py CONFIG --start S --peak P`.
  - Config: `configs/precheck/day07_seated_cable_row.json`.
  - Criteria: machine SSIM vs the empty plate per region (outside the body and the handle band), torso lean and the start-vs-peak delta, arm (start: elbow flexion ≤ 20°; peak: elbow behind the shoulder ≥ 0.10 × torso), cable (thin dark line from the pulley to the hands, best start point within ±24 px), handle (rectangle top-bar detector), head vs body tone, shaded side vs backdrop, glow.
  - Light uses low-pass p10 so muscle crevices don't count as shade. Calibrated on Day 4: −0.8, so the threshold is −2.
  - The detector was calibrated on the old pins: start pin cable 0.18 = FAIL, clip frame f22 0.75 = PASS, matching the by-eye findings.
- **Plate composite:** `tools/plate_composite_still.py CONFIG GEN OUT`. It is applied to a still once, not per frame.
  - Similarity-aligns the generated still to the verified empty plate from machine features. Gen1 was 3.3% zoomed; the alignment is a camera-only transform.
  - Then keeps plate pixels everywhere except the person and the moving handle/cable, so the machine SSIM is 0.994–1.0.
- **Sheet:** `tools/still_precheck_sheet.py output/_consistency/optB2/sheet_spec.json` → `OPTB2_STILLS_PRECHECK_sheet.jpg`.

## Attempts (RIGHT start)
| | Machine | Torso | Arm extended | Cable | V handle | Head = body | Shaded side vs backdrop | Glow |
|---|---|---|---|---|---|---|---|---|
| gen1 (edit of the old pin + a desaturated Day 4 body reference) | PASS 0.994 | FAIL 10.0° | FAIL 35.7° flexion | FAIL 53% | FAIL rectangular frame | PASS ΔL 2.4 | FAIL −7.8 | PASS 0% |
| gen2 (retry: + V-bar product photo) | PASS 1.0 | PASS 8.7° | FAIL 34.8° | PASS 70% | FAIL rectangular frame | PASS | FAIL −9.0 | PASS |
| gen3 (retry: no-top-bar wording + cropped V-bar photo) | PASS 1.0 | PASS 9.3° | PASS by eye (the detector read 165°, a keypoint error) | FAIL 35% (cable broken near the handle) | FAIL by eye: the plate floats above the grips and the grip is turned sideways (the detector passed it with 0 px, a false pass) | PASS ΔL 1.6 | FAIL −10.9 | PASS |

**Persistent blockers:**
1. **Handle:** the generator keeps drawing a rectangular frame or a broken double-D, even with a product photo as reference.
2. **Light:** the charcoal body's shaded side is 8–11 L below the backdrop on all 3 attempts; Day 4 is about −1.
3. **Arm extension:** about 35° of flexion unless pushed hard. When pushed (gen3), the handle and cable broke.

## Recommended next step (needs approval)
1. **Handle first, on the EMPTY machine** (1 GenerateImage).
   - Edit the verified plate so the V handle hangs on the cable at the low pulley; check it by eye and with the handle detector.
   - Then use that plate as the edit base for the person, so the handle shape is already in the scene.
2. **Light:** ask for a medium-grey clay material (not charcoal), or judge light after the Day 4 skin lock. The `flat_look` skin lock lifts p5 to 52, which is how Day 4 itself passes.
3. Then start + peak (2 GenerateImage), pre-check, and 1 GenerateVideo as planned.

The V-bar reference photo (fitness-hero.com.au product image) is third-party and is not committed; it is kept in the ignored `_scratch/ref/`.
