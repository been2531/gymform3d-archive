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
| D17 | Figure / equipment / backdrop melt together (D6 v4–v5, t600u–t603u) | Dumbbell plates, bench and shorts read as the same grey as the backdrop; the far plate disappears | Low-separation source: raw plates (median lum 14) and backdrop (15–29, violet vignette) have equal luminance, and they differ only in chroma; a global luminance map can't pull them apart | Mannequin stays at the Day 4 skin tone (never brightened). Backdrop flat, darker than the figure, slight cool tint (35,39,43); static backdrop weight per step; `native_sep_dx/dy` dark knot (B0−8 → BG−10, slope ≤1.25 so no noise boost); `native_dark_warm_rgb` warm-grey darks (bench, frame); glow from clip chroma only (none on backdrop). Source (Day 7+): prompt bright brushed-steel dumbbells, warm-grey bench/machine, a darker cool flat backdrop. A global chroma mix (`native_chroma_k`) was tried and REJECTED: it broke the vignette flat-field and blotched the backdrop | (sep): skin p75 − backdrop ≥ 33 (Day 4 level); dumbbells and bench ≥ 15 RGB from the backdrop and ≥ 10 from the skin, ≤ 25% of their pixels within 6 lum of the backdrop; glow on the raw backdrop ≤ 0.5% | EXISTS (v5 preflight: fig 37–40, dumbbell 28–31, bench 29–31) |
| D18 | Plates melt into the backdrop (D6 v5, t605u) | The far hex plate is invisible; the dumbbell reads as a grey smudge | Raw plate faces have the backdrop's luminance and nearly its chroma (plate c≈2 vs backdrop 3–4); no global map separates them | Custom plates, palette A (owner pick, `docs/plate_palette_A_reference.jpg`): matte cobalt (45,75,170), thin brushed-gold rim (190,160,95), steel handles, no logo. `gymform/plates.py` (flag `plate_recolour`): soft matte in discs on the handle axis; per-blob convex hull; flow-warped EMA forward + backward; colour remap that keeps graded shading; zero weight at skin brightness | (pl): plate vs backdrop ≥ 40 RGB; recolour weight on raw skin or raw backdrop ≤ 0.5%; (tq) edge flicker bound on the render; (d) bloat | EXISTS (gate). The fix FAILS on v5: plate vs bg 100–118 PASS, skin bleed 0 px, but backdrop bleed 0.6–11.7%, and (d) bloat 4.1–5.1% on step 1. Superseded t609u: palette A is now in the SOURCE clips (no recolour); without a plate matte, (pl) measures the source plates' median vs backdrop |
| D19 | Machine highlights take the glow colour (D7 v1 preflight, t612u) | Lit warm-grey top of the stack box turns faintly green on the right step | `native_glow_hue_strict` uses max(red, green dominance) for every step; warm grey has r − max(g,b) ≈ 15 > the 8 glow floor, so it got a small glow weight and the step colour (green) | `native_glow_form_hue`: the glow weight uses the step's OWN hue only (wrong = red dominance, right = green dominance); global colour function, no mask | (pf) native: bench/static pixels = global tone map (\|base − ref\| > 6 on ≤ 300 px) [EXISTS; D7 step 2: 460–560 px before the fix] | EXISTS |
| D20 | Cable machine read as skin / figure (D7 machine preflight, t662s) | (f) "skin" p75 = p90 = max = 141 (the light-grey frame), (oc) bbox includes the machine, (sh)/(j) count seat/pads/handle as dark skin, (sep)/(pl)/(d) look for dumbbells | Day 4–6 QC skin/figure = matte minus palette-A source colours; a cable machine (light-grey frame, black pads, handle, cable) is neither excluded nor a dumbbell | Opt-in `qc_profile: "cable_machine"` (gymform/qc.py `PROFILE`, `machine_colour`, `body_capsule`, `nonbody`): machine = own colour (lum > 120 sat < 30, cobalt, orange pin) ∪ fg outside the keypoint body capsule. (f)(sh)(j) skin and (mf) body = capsule minus machine; (oc) = swept subject (mannequin + machine, union over the step) centre 540±40 / margins ≥ 24, plus the mannequin's own 24 px margin per frame; (d) machine colour inside the zone is expected equipment; (sep)/(pl) N/A without a dumbbell; (mf) UI = title rows + callout block (no x < 300 strip). Thresholds unchanged | QC prints `QC PROFILE: cable_machine` [D7: (oc) PASS 54–1051 centre 552; (f)(sh)(j) now measure the mannequin and FAIL on its real tone — see below] | EXISTS (D7) |
| D21 | Wrong-step red smear without fibres + gaps in the colour (D7, t663u) | Fibre-less source-red smear on the lower back / flank; Day 4 colour only on ridges, gaps where the charcoal body sits at the backdrop level | (1) `native_keep_chroma` exclusion was limited to the clamp regions, so the source glow chroma passed on the rest of the body; (2) day4_fibre multiplies by the lum matte w_fg, which collapses on charcoal skin at backdrop level | `muscle_glow_clamp_chroma: "body"` (source glow chroma never passes on the body capsule; machine colours untouched) + `muscle_glow_wfg_from_glow` (inside the clamp, a pixel the source glows on has body weight 1). day4_fibre unchanged. `native_deglow_base` (glow channel removed from the grey-base tone) exists but is OFF: it raised (a) halo 525 px and (sh) | day4_look_metrics on the peak frames (see D7_MACHINE_CONCEPT) | EXISTS (D7) |

## Why Day 4 worked (so the defaults stay this way)

Source: `build_day4_rdl_master.py`, commits f63a353 and aeeb806.
- A free-standing, bright silver ecorche with backdrop all around and the barbell outboard. Raw feet contrast is 33–41.
- ONE soft colour-difference matte and a global tone remap. No keypoint geometry, plate, equipment layer or static layer.
- The shorts mask is gated by the matte.
- The glow is native in the clip and only recoloured.
- No frame blending and no slice cap.

Day 4 still amplifies boil, flicker and glow (see D4–D6), which is why its numbers are not the target. The raw clip is.

## t600 implementation status (Day 6 v5, `look: native_grade`, `gymform/native.py`)

The look is behind config flags; older days are unchanged.
- **D4/D9/D13/D15:** no per-frame cut-out. Clip pixels are kept, and only backdrop never touched in a step becomes flat (one static weight per step).
  - Shorts need foreground evidence.
  - `static_body_shin_r` and metal re-shading are gone (D2/D13).
- **D5:** `no_sharpen`, plus a flow-warped temporal denoise (0.25 per neighbour, occlusion guard).
- **D6:** glow is linear from clip chroma. Anchors are LK-tracked and clamped to 12 px.
- **D8:** `phase_continuous_cut` plus `target_zoom.align: hips_ankles`.
- **D16:** `no_slice_cap`.
- **QC in native mode:** (d) coverage/holes, (m) opacity and (q) use the `_alpha.png` sidecar (all ones: nothing is cut). (d) bloat measures what is VISIBLE over the raw backdrop. (m) compares against the global tone map (`_ref.png`).
- **New gates:** (sb), (gh), (sep) per frame; (tq) temporal (boil ≤1.20, edge ≤1.56, glow ≤1.59, cut) on the final render.
- **(sb)/(gh) raw backdrop** is now connected to the frame edge. With that definition, v4 measures 0–1.7% shorts-on-backdrop. The old 38.6% was largely a measurement artefact: flat fabric folds at the backdrop luminance were counted.
- **Open:** (d) step-1 right dumbbell bloat 4.3% (t70/t130), and the far hex plate stays low-contrast. Per the owner, it is not waived.
- **t605 plate recolour (D18) did not pass:**
  - The dark plate faces recolour well.
  - The lit faces have the same brightness as the backdrop and the bench backrest, so the matte either misses them (half-painted hex) or spills onto the raw backdrop (≤11.7%).
  - v5 is NOT rendered.
  - Real fix: a source with separable plates, i.e. Day 7+ prompts with palette A in the generation itself.

## t609 palette A source: QC definition + native fixes (Day 6 v5, flags in `configs/day06_muscle2_v5.json`)

Rule: a definition is changed only so it measures the real defect. No threshold was raised.
- **(a) halo:** in native mode, ring pixels that are equipment by their own SOURCE colour are excluded (`qc._equip_colour`):
  - warm-grey bench = r−b > 4, lum > 50 and saturation < 30, so the red glow is never equipment;
  - cobalt = b − max(r,g) > 15;
  - gold rim = min(r,g) − b > 20;
  - dilated 3 px.

  The 12-lum lift limit still applies to every remaining ring pixel.
- **(pf):** native mode pastes no bench layer. The check is now "bench pixels equal the global tone map": static>0.5 pixels with |base − `_ref`| > 6 must be ≤ 300 px.
- **(sep):** "lost" now means within 15 RGB of the backdrop (colour distance, not brightness), limit still 25%. The bench is measured on the bench-colour mask `_benchc.png`; the static layer also holds the bench shadow. Skin excludes source-colour equipment and the Day 4 shorts mask.
- **(mf):** in native mode the ≥4k px area is judged on the step's PEAK (bottom) frame, the Day 4 measurement frame. At lockout the source glow is genuinely small. Regions and strength are still judged on both frames, and no glow is painted.
- **(f):**
  - `native_highlight_rolloff` adds a tanh knee 10 below the lock max, so skin max stays ≤ 145 while p75/p90 are unchanged.
  - The shorts median is measured on the Day 4 foreground-evidence mask (`_shortsev.png`). With `shorts_evidence_strict`, that evidence must be DARKER than the backdrop, so bright thigh skin inside the fabric zone is never "shorts".
- **(sb):**
  - `shorts_evidence_strict`: the render/UI shorts mask also needs unblurred darker-than-backdrop evidence (open 3, erode 5, dilate 3).
  - No shorts treatment applies to pixels without foreground evidence, i.e. backdrop-level and texture-free pixels. The limit stays 2%.
- **(bg)/(sep)/(sh):** `native_bench_map` replaces the legacy static layer with a per-step STATIC bench map:
  - bench body = bench-coloured pixels (warm, > B0+15, sat < 30) that are stable in ≥ 85% of the sampled frames;
  - plus its neutral static shadows (floor/contact shadows outside the skeleton capsule, touching the bench or below the hips, with their connected penumbra);
  - plus static near-black occlusion gaps (source lum < B0−20).

  The same map drives the flat-field weight: bench and shadow never take the flat backdrop colour. The step1_t20 "41" block was a cobalt plate the matte missed, not a non-flat backdrop.
- **w_fg (matte):** `native_wfg_colour`. Cobalt has a NEGATIVE red/green dominance, so the lum+act matte read the plates as backdrop. Where act < 0, the chroma distance to the backdrop tint replaces it; pixels with act ≥ 0 are identical to the legacy matte. (oc) centres the mannequin without the source-colour equipment.
