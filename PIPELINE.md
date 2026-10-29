# @gymform.3d one-pass pipeline

```
python3 run_day.py --day N          # build -> QC gate -> auto-fix/retry (max 3) -> READY / NOT READY
python3 run_day.py --day N --skip-build   # QC the existing master only
```

## Files
| file | role |
|---|---|
| `gymform_spec.json` | **Locked rules** (defaults for every day): minimal no-prop studio bg, silver skin lock (p75~165 / p90~191 / max 226) in all steps, local Day-1 X-Ray window for Steps 1 (crimson) / 3 (emerald) with bones as hero, muscle fibers Steps 2/4, zero bloom/blur, depth-behind title, translucent preview with equipment in front, equal height ±4%, intact equipment (dumbbell/barbell/cable/machine/bench/rack), tempo 0.3/1.5/1.3/1.5/0.4 (monotonic), shorts lum 10..22, 3/4 angles, Step 4 cover. |
| `exercise_curriculum.json` | day list (title, slug, metrics). |
| `build_day5_gym_studio_v4.py` | reusable 4-step renderer. Per-day `DAY_CONFIGS` holds only assets, equipment zones, step_scale/step_dy alignment and callout overrides. Reads `GYMFORM_OVERRIDES` (auto-fix). |
| `qc_gate_gymform.py` | QC gate, PASS/FAIL per item (a halo, b1 title, b2 preview, c height, d equipment vs per-frame expected footprint, e bone/muscle presence, f skin/shorts, g X-Ray vs Day 1 reference, h equipment in front of UI, i colour lock across steps). |
| `run_day.py` | one-command entry point; writes `output/qc/dayN_status.json` (`ready: true` only if every check passes). |

## Adding a day
1. Generate the 4 step clips (`output/assets/dayN_*_step{1..4}_*_720p.mp4`, 3/4 angle, same framing).
2. Add a `DAY_CONFIGS[N]` entry (assets, equipment type + zone, first-guess step_scale / step_dy) and register the builder in `run_day.BUILDERS`.
3. `python3 run_day.py --day N`. Auto-fix only changes render parameters (scale, zones, feather) — never QC thresholds. X-Ray look / skin lock failures stop as needs-human.

## How the hard parts work
- **Clean mask**: glow-robust brightness key vs row background + keypoint anatomical envelope + shorts polygon; feather 0.85 px.
- **Equipment**: per-step temporal median backdrop plate; equipment = pixels differing from the plate that are darker (>=2 channels), neutral vs the cool backdrop (min of B-R/B-G drop), or textured; plate faces completed by convex per-stack hulls (neutral/glow-lit pixels only). QC re-detects the expected footprint independently from the raw frame and requires >=98.5% solid coverage, no hole >60 px, inside frame and zone.
- **X-Ray**: Day 1 v36 per-pixel method inside a local window (shoulder girdle + upper arm; Step 1 adds both arms + upper rib cage). Outside the window the body keeps the locked silver skin.
- **UI**: title blended behind the mannequin (fg_front 0.92); preview translucent over the mannequin, equipment pixels fully in front.

## Rules for commits
Scripts, config, spec, plates only. Never mp4/wav/tmp/caches/secrets (`.gitignore`). Threads: nice 10, OpenCV 2 threads, ffmpeg `-threads 2`.
