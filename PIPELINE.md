# @gymform.3d pipeline (v5, modular)

Daily command: `python3 run_day.py --day N`  (prints one JSON line; full log in output/logs/)

| file | role |
|---|---|
| `gymform_spec.json` | locked rules (format, silver skin lock, shorts, studio, X-Ray style, QC thresholds) |
| `configs/dayNN.json` | the ONLY per-day input: exercise, clips, equipment zones, X-Ray bones, target muscles, callouts, TTS lines, caption ({title, question, save} → built + linted by `gymform/caption.py`: EXACTLY 3 lines separated by blank lines — title / ONE experience question ending "?" / save line; no body text, English only, 0 hashtags, no DM or vote CTA), per-step scale/dy |
| `gymform/ctx.py` | shared state, spec defaults + day config -> CFG |
| `gymform/bg.py` | studio plate, contact shadow, per-step size/floor transform, cached temporal backdrop median |
| `gymform/mask.py` | clean fg key, anatomical envelope, shorts polygon/mask |
| `gymform/equipment.py` | generic equipment zones + plate-difference footprint (solid metal, soft edge only near the object), matte metal shading |
| `gymform/muscle.py` | locked silver skin (dual anchor p75/p90) + target-muscle fibres |
| `gymform/xray.py` | Day 1 v36 local X-Ray window (crimson / emerald), zero blur |
| `gymform/layout.py` | laser, callouts, depth-behind title, translucent 4-slot preview, sheet |
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
