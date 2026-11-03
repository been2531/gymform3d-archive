# Reel format v2 + real-machine smoke (D7v2, seated cable row) — 2026-11-03 KST

Owner asks: pt633u (2D consistency), t634u–t636u ("실제 운동기구와 동일해야하고", popularity with minimal generation).
Smoke only — no full render, nothing published. Generators: Grok Bot GenerateImage / GenerateVideo only (no grok.com, no xAI API).

## 1. Format (10–12 s, ONE camera, TWO generated clips)

| t | Source | Post |
|---|---|---|
| 0–1.0 s HOOK | wrong clip 2.2–2.8 s (pinned contraction hold) at 0.6× | punch-in 1.15×, red ✕ badge |
| 1–6 s | wrong clip 0–5 s, 1 rep | ✕ badge "SHRUG + ARM PULL" |
| 6–11 s | right clip 0–5 s, 1 rep, same angle | ✓ badge "ELBOWS BACK = LATS" |
| 11–11.8 s TAIL | wrong clip 0–2.2 s at 2.75× | punch-in 1.15×, ✕ badge; ends on the wrong contraction = hook frame 0 → seamless loop |

- Rough cut (540×960, 11.83 s, 284 f): loop SSIM last↔first frame **0.977**. Alt. loop: hard cut at 11 s back to hook (simpler, less seamless).
- The "4-step" feel = re-use of the 2 clips (punch-in, slow-mo hold, tail). Badges/text/slow-mo/punch-in are post only. Glow is generated in the clip from the pinned frames and must be re-checked by G3/G4; the Day 4 native grade is applied unchanged at render.
- Generation cost per reel: 1 base + 1 wrong start + 1 right start + 2 contraction keyframes (images), 2× GenerateVideo 720p 5 s. Retries ≤ 1 per clip.

## 2. Pinning recipe (what made G1–G8 pass)
- Both starts are edits of ONE base frame (`base_stretch_gen_raw`, no glow); right start is a **glow-only** edit (no pose edit — a "chest up" edit rotated the torso, view 0.30 vs 0.69 → G7 fail in t1).
- `startframe_grade.py <raw> <out> 38.3 67 35,39,43 80` applied **once** to the starts only. Contraction keyframes = raw edits of the graded 720 start, just resized to 720×1280, **never graded again** (t1 double-graded keyframes → cobalt B≈210 → G8 ΔRGB 60 fail).
- GenerateVideo 720p, 5 s, image = last_frame = start_720, keyframes = same contraction image at 2.2 s and 2.8 s (forces exactly 1 rep with a short peak hold). 480p cannot pass G1/G2 (edge-IoU ceiling ≈0.89).

## 3. Real equipment + camera
Reference: Matrix Aura Low Row (also Life Fitness MJRW): front tower with 2 uprights + top pulleys, stack on guide rods with selector pin, low pulley housing just above an angled footplate, floor rail, long low padded seat, horizontal cable to a close-grip handle. Generated base matches this layout (stack cobalt palette A, warm-grey frame, charcoal mannequin). Gold ring on plates is barely visible — known gap.
Camera: elevated front-3/4 from the athlete's front side (~55° off the cable axis). Pure front would put the tower between camera and athlete and hide the cable; this view shows tower, pulley, cable line, footplate and the athlete's face side.

## 4. Gate results (raw clips, 360×640 sampling, `tools/source_gate_smoke.py`)
| Gate | t1 wrong | t1 right | **t2 wrong** | **t2 right** |
|---|---|---|---|---|
| G1 start SSIM/edge | .993/.917 ✓ | .993/.921 ✓ | .992/.913 ✓ | .993/.915 ✓ |
| G2 end / key(±6) | ✓ | ✓ | .991/.903, .99/.91 ✓ | .991/.902, .989/.898 ✓ |
| G3 v2 ratio vs pins | 0.99–1.06 ✓ | 0.65–1.36 ✓ | 0.95–1.03 cv 2.4 ✓ | 0.93–1.31 cv 10.7 ✓ (v1 ✗) |
| G4 off-body glow | .02% ✓ | .02% ✓ | .02% ✓ | .03% ✓ |
| G5 bg SSIM | .988 ✓ | .974 ✓ | .988 ✓ | .938 ✓ |
| G6 bg drift | 2.0 ✓ | 0.0 ✓ | 1.0 ✓ | 1.0 ✓ |
| G7 view spread | 0.292 ✗ | 0.292 ✗ | **0.037 ✓** | **0.037 ✓** |
| G8 cobalt ΔRGB | 60.1 ✗ | 27.1 ✗ | **4.6 ✓** | **5.8 ✓** |
| G9 motion_check | not fired | pass | not fired (only lean_back_rock measurable) | pass (elbows_past_sides 50.7° 2D = warn) |
| G10 fig−bg | 42.8 ✓ | 41.9 ✓ | 41.8 ✓ | 41.3 ✓ |
| reps | 1 | 1 | 1 | 1 |

## 5. Honest caveats
- Right glow drops to ~5.5% at the contraction (pins say so: arms cover the lats from this angle). Passes G3 v2, fails v1 — viewers see the green dim mid-rep. Post option: hold the ✓ glow via the Day 4 native glow layer, not regeneration.
- 2D elbow max 50.7° (right) vs 43.5° (wrong): the right rep is deeper but the elbows-past-torso depth is under-read in 2D at 55° yaw; visually the handle reaches the belly.
- Right green covers lats + rear/side delts; a thin edge touches the front shoulder. Wrong red = biceps, front delts, upper traps.
- Wrong start `startframe_gates_pass` false inside G10 recheck on the clip frame (plate p05 31.2 < 33 after video codec), fig−bg still 41.8.
- Rejected generations kept in `frames/` with `_rejected_` names (nothing deleted).

## 6. Files
`output/_consistency/smoke_d7v2/`: `smoke_d7v2_sheet.jpg` (real ref vs base/starts vs t2 clip frames), `gate_t1.json`, `gate_t2.json`, `roughcut/strip.jpg`, badges. Box-only: mp4s, `frames/`, `refs/` (third-party product photos).
