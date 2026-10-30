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

## 2b. Preflight gate (LOCKED 2026-10-30): get it right before the full render
- Flow: `run_day.py` = config lint/guards → **preflight** → full render → final QC. Default path, no bypass flag.
- Preflight renders only the QC sample frames (start/t20/t35/peak/t70/t100/t130 × 4 steps, zoom included) through
  the real composite path (`render.render_pose`), then runs **all** QC checks on them (`qc.py --preflight`;
  only the container `format` check waits for the mp4). Time budget is about 60–90 s at nice 10 with 2 workers.
- A PASS writes a stamp `_cache_gymform/preflight/dayNN_<hash>.json`. The hash covers the day config, overrides,
  `gymform_spec.json` and every `gymform/*.py`. `gymform.render --render-step` **refuses** to render without a
  passing stamp for the exact hash (`gymform/preflight.require`). Any config/code change invalidates it.
  Tests: `python3 -m unittest tests.test_preflight`.
- A failing preflight means fixing the root cause and preflighting again. Never "render and see".
- Preflight and render times are logged in `output/qc/production_times.jsonl`.

## 3. Look (locked)
- Minimal dark charcoal studio, no props. **[QC] (a)** zero bloom / glow halo; zero blur.
- **Step order (LOCKED from Day 6):** 1 = WRONG bone X-Ray (crimson), 2 = RIGHT bone X-Ray (emerald),
  3 = WRONG muscle, 4 = RIGHT muscle (3/4 cover). bone_steps [1,2], muscle_steps [3,4], preview slot = step−1.
  Any other order is rejected before render (`ctx.order_lint` [guard] in run_day.py) and in **[QC] (r)**.
  Day 1–5 (published) used the legacy 1/3 bone, 2/4 muscle order. Look code maps every step through `ctx.legacy()`.
  **[QC] (e)(g)**
- **Target zoom:** the per-day `target_zoom` {zoom, focus_x, focus_y, equipment_margin} pulls the camera toward the
  target region (shoulder midpoint at the focus). The same scale applies to every step **[QC] (c)**. It backs off
  automatically before the dumbbells would leave the frame; the bench may be partial. Single Lanczos resample from
  the 720p source, no blur. Keep the zoom moderate (`_normalized.upscale_vs_720p` is logged).
- **No stray lines:** the static-equipment layer ignores moving pixels (per-step motion map) and temporal-median
  ghosts (solid-core reconstruction). Every foreground component must attach to the mannequin or equipment **[QC] (q)**.
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
