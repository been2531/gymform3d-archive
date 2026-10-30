# @gymform.3d pipeline (v5, modular)

> Locked rules (enforced by run_day.py guard + QC): see [RULES.md](RULES.md).

Daily command: `python3 run_day.py --day N`  (prints one JSON line; full log in output/logs/)

| file | role |
|---|---|
| `gymform_spec.json` | locked rules (format, silver skin lock, shorts, studio, X-Ray style, QC thresholds) |
| `configs/dayNN.json` | the ONLY per-day input: exercise, clips, equipment zones, X-Ray bones, target muscles, callouts, TTS lines, caption ({title, question, save} → built + linted by `gymform/caption.py`: EXACTLY 3 lines separated by blank lines — title / ONE experience question ending "?" / save line; no body text, English only, 0 hashtags, no DM or vote CTA), per-step scale/dy |
| `gymform/ctx.py` | shared state, spec defaults + day config -> CFG |
| `gymform/bg.py` | studio plate, contact shadow, per-step size/floor transform, cached temporal backdrop median |
| `gymform/mask.py` | clean fg key, anatomical envelope, NATIVE shorts mask (source fabric pixels, pose-aware hem; no 2D polygon) |
| `gymform/equipment.py` | generic equipment zones + plate-difference footprint (solid metal, soft edge only near the object), matte metal shading |
| `gymform/muscle.py` | locked silver skin (dual anchor p75/p90) + target-muscle fibres |
| `gymform/xray.py` | Day 1 v36 local X-Ray window (crimson / emerald), zero blur |
| `gymform/layout.py` | laser, callouts, see-through title + 4-slot preview (Day 4 rule: visible at `ui.overlap_alpha` 0.40 over mannequin AND equipment, 3D layer sharp under it), sheet |
| `gymform/tempo.py` | 150-frame step tempo, QC frame tags |
| `gymform/tts.py` | locked edge_tts voice; Day 6+ appends the spoken end question after the Step 4 line, writes `<wav>.timing.json` |
| `gymform/qc.py` | QC gate (a)-(i); `qc_gate_gymform.py` is a shim |
| `gymform/render.py` | per-pose orchestrator + stage workers |
| `tools/gymform_proofs.py` | Day1-vs-DayN X-Ray board + equipment strip from the final mp4 |

Stages and caches (`_cache_gymform/`, hashed on config subset + spec + module sources):
kp (kp pkl + 720p medians) -> icons -> composite+overlays per step (x264 pipe, veryfast, 1 thread/worker, 2 workers, nice 10) -> tts (hash of lines) -> encode (concat stream copy + AAC) -> qc.
Re-run one step: `python3 run_day.py --day 5 --only step2 --from composite`. QC only: `--from qc`. Auto-fix maps a failing
item to its module and steps, changes a render parameter (never a threshold) and re-renders only those steps.

New day: drop 4 clips into output/assets, copy configs/day05.json -> dayNN.json and edit it, run the daily command.
Commit rules: scripts/config/spec/plates only; never mp4/wav/tmp/caches/logs/secrets.

## Spoken end question (Day 6+)
The last narration line asks the caption's experience question out loud (e.g. "Where do you feel it most?").
- Config: optional `question_spoken` in `configs/dayNN.json` (<= 6 words, ends with "?"). If missing, `caption.spoken_question()`
  takes the shortest "?" clause of `caption.question` (them -> it), else the spec default.
- Timing (`gymform_spec.json` caption_rules.spoken_question): Steps 1-3 lines/gaps untouched; question starts 0.25 s after the
  Step 4 line ends and must end <= 19.85 s (reel stays 20.0 s). Keep the Step 4 line short (~<= 11 words).
- Checks: `run_day.py --check` lints it (estimated timing, ~0.16 s/syllable); `tts.py` asserts with real durations;
  QC (l) reads `qc_dir/tts_timing.json`. Day <= 5 is unchanged (legacy TTS cache key).

## UI overlap rule (Day 4 see-through, locked 2026-10-29)
- Where the mannequin or any equipment overlaps the exercise title or the top preview strip, the title/preview stay
  VISIBLE see-through at `gymform_spec.json` → `ui.overlap_alpha` (title 0.40, preview 0.40). Never fully hidden.
- Day 4 (RDL, published) measured: title over body median alpha 0.30 (p25 0.16 / p75 0.47, soft colour mask),
  preview >= 0.35. Locked to a constant 0.40 so it is identical over body and equipment.
- Under the overlap the preview frost is dropped (3D layer stays sharp) and the equipment keeps (1-alpha) of its own
  pixels: never cut, hidden or distorted. Off the occluders title/preview are drawn exactly as before.
- QC: render writes `_uia.png` / `_uirgb.png` sidecars; (b1) fits the effective UI alpha over the solid mannequin,
  (h) over equipment — both must be within ±0.12 of the spec alpha with a small reconstruction residual (= visible,
  not opaque, 3D layer intact).

## Step order + target zoom (Day 6+, locked 2026-10-30)
- 1 wrong bone X-Ray, 2 right bone X-Ray, 3 wrong muscle, 4 right muscle (cover). `ctx.order_lint` guard + QC (r).
  Step-specific look code (silver/muscle/laser/callout) keys on `ctx.legacy(step)` = the role, not the position.
- `target_zoom` in the day config → `ctx.normalize_scale` (focus = median shoulder midpoint, equal scale per step,
  dumbbell margin back-off); `bg.transform` uses a single Lanczos 720p→canvas resample with DX/DY offsets.
- QC (q): stray / unanchored foreground components. Static bench = median plate minus motion map (no ghost edges).

## Preflight gate (default, locked 2026-10-30)
```
python3 run_day.py --day N --check       # lint only
python3 run_day.py --day N --preflight   # QC sample frames (7/step) via the real composite path + ALL QC (~60-90 s)
python3 run_day.py --day N               # lint -> preflight (stamp reused if the hash is unchanged) -> render -> QC
```
- The preflight writes the sidecars and `<tag>_final.png` UI frames to `<qc_dir>_preflight/`. QC runs with `--preflight`.
- Stamp: `_cache_gymform/preflight/dayNN_<sha1(config, overrides, spec, gymform/*.py)>.json` = {ok, checks, fails, t}.
  `gymform.render --render-step` exits non-zero ("render_guard") without a passing stamp for the same hash.
- An autofix that changes the overrides changes the hash, so the next attempt preflights again.
- Exit codes: 3 = guard (order/callout/native source), 4 = preflight FAIL, 1 = final QC FAIL.
- Tests: `tests/test_preflight.py` (missing or failing stamp blocks the render; a config or override change
  invalidates the stamp).

## Pair motion (Day 6+, locked t521u)
```json
"motions": {
  "wrong": {"base": "<wrong clip>", "i0": 2, "i1": 62, "y_shift": 80,
            "renders": {"muscle": {"file": "<wrong clip>"}, "bone": {"file": "<edit>", "edit_of": "<wrong clip>"}}},
  "right": {... same for the right clip (Step 4 = cover, 3/4 view)}
}
```
- Derived steps: 1 = wrong/bone, 2 = right/bone, 3 = wrong/muscle, 4 = right/muscle.
- The keypoint track is extracted once per motion (from the base) and shared, which also gives identical framing.
- The second native rendering of a motion comes from a Grok Imagine video edit of the base clip (`grok-imagine-video`
  edit mode keeps duration, framing and motion; source ≤ 8.7 s). Prompt: change only the anatomy overlay, and keep the
  camera, mannequin, shorts, dumbbells, bench, studio, lighting, motion and timing exactly.
- Guard `pair_lint` (exit 3); missing twin files block `--check`, preflight and render (exit 2). QC (s) checks pair
  keypoints and silhouette.
- Tests: `tests/test_pairs.py`.
