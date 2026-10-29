# @gymform.3d — LOCKED RULES (enforced in code)

`run_day.py` prints a pointer to this file on every run. Rules marked **[guard]** stop the run before rendering;
rules marked **[QC]** fail the QC gate (`gymform/qc.py`). A day that fails is never published.

## 1. Source: native 3D only, 0% 2D overpaint
- The mannequin, its **real 3D matte-black shorts**, all equipment (dumbbells, bench, cable, machine) and the
  anatomy (X-Ray bones / muscle fibres) must be rendered **inside the 3D source** from the first frame.
  The pipeline only regrades native pixels (silver lock, X-Ray ramp, fibre tint). Nothing is drawn afterwards.
- Every `configs/dayNN.json` needs a `source` block:
  `{"type": "native_3d_clip" | "native_3d_blender", "native": ["mannequin","shorts","equipment",...], "overpaint_2d": {}}`.
  Missing / other type / any enabled 2D stage (`shorts_polygon`, `shorts_fill`, `drawn_bones`, `drawn_shorts`,
  `drawn_equipment`, `warp`, `rembg_cutout`) → **[guard]** exit 3, and **[QC] (o) native_3d_source**.
- Shorts mask = the source's own fabric pixels only (`gymform/mask.native_shorts`, pose-aware hem along each
  thigh axis, standing or seated). The 2D shorts polygon was deleted. **[QC] (n) 2d_overpaint_ratio == 0**.
- If a clip lacks native shorts/equipment/anatomy, regenerate that clip (minimal generation, check quota first);
  never paint it in.

## 2. Process
- **Fix the root cause in the shared modules first, then render.** No blind re-render loops; autofix may only
  change a render parameter, everything else is `needs-human` with the exact cause.
- One-pass modular pipeline (`gymform/*`, `run_day.py` stage cache). Fix shared modules, never revert.
- **Measure production time every run** (`output/qc/production_times.jsonl`, `t.total` in the status line) and
  make each reel faster than the last.
- Copy what already works in approved reels (Day 1 v36 X-Ray, Day 2/Day 4 look, @demicstory reference), then
  improve on it.
- Modest CPU (Shorts Manager has box priority): `nice`, ≤2 step workers, never overlapping runs.
- Never commit secrets; nothing important in /tmp; clean large temp files.

## 3. Look (locked)
- Minimal dark charcoal studio, no props. **[QC] (a)** zero bloom / glow halo; zero blur.
- Steps 1 & 3: local Day-1-style X-Ray (Step 1 crimson, Step 3 emerald). Steps 2 & 4: muscle fibres.
  **[QC] (e)(g)**
- Silver skin locked in all 4 steps from the first render (p75 165 / p90 191 / max 226). **[QC] (f)(i)(j)(k)**
- Same mannequin size in all 4 steps: auto-normalised from keypoint bone lengths (`ctx.normalize_scale`), feet
  on one floor line. **[QC] (c)** skeletal size spread ≤ 4%.
- Equipment never cut off, distorted or faded. Moving equipment **[QC] (d)(h)**; static equipment (bench)
  kept as native pixels, opaque **[QC] (m)**.
- Tempo 0.3 / 1.5 / 1.3 / 1.5 / 0.4 s, one rep per step.
- On-screen angle numbers = the coached joint angle the label names; label, config and TTS must agree (`layout.callout_lint`: [guard] + [QC] (p)). deg is "N°" or {"from","to"}, never a bare list.
- Title + preview stay visible at **0.40 translucency** wherever the mannequin/equipment overlaps them
  (Day 4 rule). **[QC] (b1)(b2)(h)**

## 4. Copy & voice
- Caption: exactly 3 lines, English — title / one personal-experience question / save line. 0 hashtags,
  no DM-reward CTA, no vote hooks, no coaching body. (`gymform/caption.lint`)
- TTS: English; the last line is a short spoken experience question (≤ 6 words, ends by 19.85 s); Step 4 line
  ≤ 11 words. **[QC] (l)**

## 5. Publishing
- Never publish without the owner's explicit approval (`approvals/dayNN.json` status `pending` until approved).
  Instagram is read-only otherwise. Posts go up at 10:00 KST.
