# @gymform.3d pipeline (v5, modular)

Daily command: `python3 run_day.py --day N`  (prints one JSON line; full log in output/logs/)

| file | role |
|---|---|
| `gymform_spec.json` | locked rules (format, silver skin lock, shorts, studio, X-Ray style, QC thresholds) |
| `configs/dayNN.json` | the ONLY per-day input: exercise, clips, equipment zones, X-Ray bones, target muscles, callouts, TTS lines, caption ({title, body[], question, save} → built + linted by `gymform/caption.py`: 0 hashtags, no DM CTA, no vote hook, ONE experience question), per-step scale/dy |
| `gymform/ctx.py` | shared state, spec defaults + day config -> CFG |
| `gymform/bg.py` | studio plate, contact shadow, per-step size/floor transform, cached temporal backdrop median |
| `gymform/mask.py` | clean fg key, anatomical envelope, shorts polygon/mask |
| `gymform/equipment.py` | generic equipment zones + plate-difference footprint (solid metal, soft edge only near the object), matte metal shading |
| `gymform/muscle.py` | locked silver skin (dual anchor p75/p90) + target-muscle fibres |
| `gymform/xray.py` | Day 1 v36 local X-Ray window (crimson / emerald), zero blur |
| `gymform/layout.py` | laser, callouts, depth-behind title, translucent 4-slot preview, sheet |
| `gymform/tempo.py` | 150-frame step tempo, QC frame tags |
| `gymform/tts.py` | edge_tts voice |
| `gymform/qc.py` | QC gate (a)-(i); `qc_gate_gymform.py` is a shim |
| `gymform/render.py` | per-pose orchestrator + stage workers |
| `tools/gymform_proofs.py` | Day1-vs-DayN X-Ray board + equipment strip from the final mp4 |

Stages and caches (`_cache_gymform/`, hashed on config subset + spec + module sources):
kp (kp pkl + 720p medians) -> icons -> composite+overlays per step (x264 pipe, veryfast, 1 thread/worker, 2 workers, nice 10) -> tts (hash of lines) -> encode (concat stream copy + AAC) -> qc.
Re-run one step: `python3 run_day.py --day 5 --only step2 --from composite`. QC only: `--from qc`. Auto-fix maps a failing
item to its module and steps, changes a render parameter (never a threshold) and re-renders only those steps.

New day: drop 4 clips into output/assets, copy configs/day05.json -> dayNN.json and edit it, run the daily command.
Commit rules: scripts/config/spec/plates only; never mp4/wav/tmp/caches/logs/secrets.
