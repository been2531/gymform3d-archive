# DEFECT PLAYBOOK: problem → defined fix (decided up front, t598u/t599u)

**Read before every preflight.** `run_day.py` prints this file's entries at preflight. From Day 7 it refuses to run
unless the day config has `"defect_playbook_sha": "<first 8 hex of sha1 of this file>"`; the refusal message prints
the exact value.

Rules:
- A defect seen on screen is fixed by its row below. No new symptom patch.
- A defect type not in this table is first added here (symptom, cause, fix, check) and then fixed.
- Evidence for the Day 6 rows: [`ROOTCAUSE_DAY6.md`](ROOTCAUSE_DAY6.md). Measurement tool: `tools/rootcause_metrics.py`
  (raw vs output, flow-compensated). "ratio" means output ÷ aligned raw clip.

Check status:
- **EXISTS**: a gate in `gymform/qc.py` today.
- **MEASURED**: computed by `tools/rootcause_metrics.py`, not yet a gate.
- **TODO**: to add with the fix.

| # | Defect (days seen) | Symptom | Root cause | Defined fix (module / setting / source decision) | Automatic check + threshold | Status |
|---|---|---|---|---|---|---|
| D1 | Shading / vignette (D6 v1–v3) | Backdrop spotlight or vignette; dark shaded side of the body | Vignetted backdrop plate + contact shadow (`bg.studio_bg`, `render.shade`); skin lock ignored lum < 46 | `flat_look: true` + `studio_bg: output/assets/studio_flat_day4_1080p.png` (no floor shadow) + `_flat_skin_lock`. Under fix plan 1: a smooth backdrop gain field instead of a replaced backdrop. Source: prompts ask for flat, shadowless light | (bg) backdrop 120 px block span ≤ 4 lum; (sh) skin < 46 ≤ 6% and p5 ≥ 44 | EXISTS (day ≥ 6) |
| D2 | Dumbbell / bench morph (D5, D6) | Plates change shape; dark faces go black; holes; bench seat melts | Not generation (raw is rigid). Per-frame `equipment.footprint` (plate diff), `metal_mask` + `metal_shade`, static layer; the shorts mask covers the bench | Keep native equipment pixels: no metal re-shade and no separate equipment layer (fix plan 1, `look: native_grade`). Until then: no new thresholds | (d) footprint solid, no holes [EXISTS]; dumbbell-box residual ratio ≤ 1.10 and bench-box ratio ≤ 1.15 [MEASURED: v4 0.90–1.05 / 1.5–1.6] | partly |
| D3 | Hand–dumbbell merge (D6) | Dumbbell melts into the hand; fingers cut | `finalize` near-wrist plate zeroing + footprint vs body-key fight at the grip | Grip zone = native pixels, with no mask decision inside the wrist disc (r = 0.6 × hand length) | Wrist-disc alpha (all frames) min ≥ 0.95, and XOR vs the raw silhouette in the disc ≤ 2% | TODO |
| D4 | Cut-out halo / edge flicker (D4, D5, D6) | Shimmering outline and halo around the arms and torso | Hard per-frame masks with no temporal coherence; replaced backdrop | No cut-out (fix plan 1). If a matte is needed: flow-warped alpha with EMA 0.6, soft ramps only | Silhouette flicker ratio ≤ 1.15 [MEASURED: v4 1.57–1.66, D4 1.56–1.74]; edge residual ratio ≤ 1.10 [MEASURED: 1.17–1.28]; (a) no colour halo [EXISTS] | partly |
| D5 | Fibre boil (D4, D6) | Muscle texture crawls frame to frame | `silver_skin` high-pass emphasis (`hf_ridge*0.70`, `sulc*11`) amplifies generation micro-noise. Not `flat_look` (v3 = v4) | Remove the hf/sulc emphasis; linear tone map only | Interior boil ratio ≤ 1.15 [MEASURED: v4 1.42–1.59, D4 1.20–1.58]; fibre HF std ratio ≤ 1.10 [MEASURED: +20–31%] | MEASURED |
| D6 | Glow or callout sliding (D4–D6) | Red/green glow and anchor dots swim off the muscle | Glow: threshold ramp stretches native glow flicker. Anchor: kp track not surface-locked | Linear glow recolour + flow EMA on the glow weight; LK-flow-tracked anchor seeded from kp, re-seeded when > 12 px apart | Glow residual ratio ≤ 1.30 [MEASURED: v4 1.92–2.40, D4 1.59–3.55]; anchor vs flow track ≤ 8 px [MEASURED: 30.9 / 9.0]; (ra) anchor on the upper arm [EXISTS] | partly |
| D7 | Arm thickness drift / elbow warp (D5, D6) | Forearm width pulses, elbow bends wrongly | `finalize` zeroes arm pixels equal to the temporal median plate (the arm's own resting position); kp skin segments | Drop plate zeroing on body pixels; no kp geometry in the look path | Forearm width (raw silhouette cross-section at 50% elbow→wrist) output/raw within ±4% every frame | TODO |
| D8 | Step-cut jump (D4, D6) | Pose, bench position and brightness jump at the step change | Step N ends at pose n−1 and step N+1 starts at pose 0; separate generations; shoulder-only alignment | Continue the wave phase across the cut; align on hips/ankles/bench; gain-match the body mean. Source: both clips from one start-frame set and camera | Wrist jump ≤ 40 px, body-mean Δ ≤ 3 lum, cut frame diff ≤ 2× typical p90 [MEASURED: 271 px, 5.7 lum, 4.3×] | MEASURED |
| D9 | Floating feet / bench (D6) | Feet and bench base look cut out and hover | The floor is removed while native dark bench/floor-shadow pixels are pasted opaque; low source separation | Keep the floor contact (native pixels + backdrop gain field). Source: a backdrop clearly brighter than the bench | Feet contrast output/raw ≥ 0.85 and raw feet contrast ≥ 30 [MEASURED: 0.74–0.93; raw 22–29]; (m) bench opaque [EXISTS] | partly |
| D10 | Skin tone drift (D1–D6) | Body brightness differs per step or frame | Per-frame percentile lock on a different skin population | `skin_tone_lock` / `_flat_skin_lock` with the SPEC `skin_lock_all_steps` targets | (f) p75 ≈ 75 / p90 ≈ 88 within the spec ranges; (i) p75/p90 across steps within 5% | EXISTS |
| D11 | Bone colour drift (D1, D2, D5) | X-Ray bones change colour or brightness vs Day 1 | Native bone glow re-tinted per clip | `xray.apply` locked to the Day 1 reference; `target_bones` de-glow in `silver_skin` | (g) X-Ray vs the Day 1 reference | EXISTS |
| D12 | (q) specks (D5, D6) | Stray dots, lines or blobs not anchored to anything | Leftover mask islands and UI primitives | Connected-component cleanup (< 800 px islands removed in `shade`) | (q) no stray/unanchored overlay lines or blobs | EXISTS |
| D13 | Ghosts (D6 v2) | Faint duplicate dumbbells or body parts | `clean_plate` reference darker than the lit centre; the kp shin capsule (`static_body_shin_r`) paints body where there is backdrop | `clean_plate: false`; **remove `static_body_shin_r`** (v3/v4 config) | Raw-backdrop pixels rendered opaque ≤ 2% of the object area [MEASURED: v4 11.9%]; shorts ∩ raw backdrop ≤ 2% of the shorts mask [MEASURED: 38.6%] | MEASURED |
| D14 | Weak wrong/right difference (D6 v1) | Wrong and right movements look almost the same | The source clips don't show the mistake strongly enough | Source: start frames at the most telling point; the mistake exceeds the standard by a wide margin (`docs/MOTION_STANDARDS.md`) | `gymform.motion_check`: wrong step fires ≥ 1 mistake signature, right step passes all items; (s)/(z) pair checks [EXISTS]. motion_check is not yet wired into preflight | partly |
| D15 | Shorts edge flicker / shorts slab (D5, D6) | Hem flickers; a dark slab appears beside the thigh or over the bench | `native_shorts` zone has no foreground gate, then is forced opaque in `finalize` | Gate by foreground evidence (Day 4 `× w_fg`) or keep the native shorts pixels (fix plan 1) | Shorts ∩ raw backdrop ≤ 2% [MEASURED: 38.6%]; (n) 0% 2D overpaint [EXISTS] | MEASURED |
| D16 | Judder from the slice cap (D5, D6) | Uneven motion, skipped poses | `d5.load_native_slice_720p` caps slices at 36 frames | Remove the cap (Day 4 loader had none) | Unique poses ≥ 0.9 × slice length [MEASURED: wrong 36/45] | MEASURED |

## Why Day 4 worked (so the defaults stay this way)

Source: `build_day4_rdl_master.py`, commits f63a353 and aeeb806.
- A free-standing, bright silver ecorche with backdrop all around and the barbell outboard. Raw feet contrast is 33–41.
- ONE soft colour-difference matte and a global tone remap. No keypoint geometry, plate, equipment layer or static layer.
- The shorts mask is gated by the matte.
- The glow is native in the clip and only recoloured.
- No frame blending and no slice cap.

Day 4 still amplifies boil, flicker and glow (see D4–D6), which is why its numbers are not the target. The raw clip is.
