# One-command Day pipeline (`tools/run_day_pipeline.py`)

Routinizer, 2026-11-03 KST. One pipeline config runs a whole 2-step Day, so the bot only steps in for
judgement (a failure) and for the two video generations, which stay OUTSIDE the script (Grok Bot GenerateVideo).
Nothing publishes. No xAI API (`XAI_API_KEY` is stripped from every child process env and never read), no grok.com,
no `/workspace/bgm`. Read `docs/DEFECT_PLAYBOOK.md` first: the script checks the day config's `defect_playbook_sha`
and refuses on a mismatch, but never writes the sha for you.

## Commands
```bash
cd /workspace/daily_info_reels
P=configs/pipeline/day07_d7v2_t2.json
python3 tools/run_day_pipeline.py --config $P --plan        # validate + list steps + run_day.py --check; renders nothing
python3 tools/run_day_pipeline.py --config $P --manifest    # (re)write the GenerateVideo handoff manifest only
python3 tools/run_day_pipeline.py --config $P               # full run (exit 10 + manifest if a clip is missing)
nohup python3 tools/run_day_pipeline.py --config $P --watch >/dev/null 2>&1 &   # wait for both clips, then run
python3 tools/run_day_pipeline.py --config $P --gates-only  # G1-G10 only, no render
```
Options: `--out-root` (default `output/runs`), `--max-wait-min 120` (idle-box wait before a stage),
`--max-pause-min 180` (total yield to Shorts per stage), `--keep-work`, `--interval 60` / `--watch-timeout-h 24`.

## Stages
| # | Stage | What runs (existing code reused) |
|---|---|---|
| 1 | validate | pipeline + day config, `DEFECT_PLAYBOOK.md` sha, look = `native_grade` (no per-frame cut-out), Day 4 fibre flags reported, stills + keyframes exist, clips readable |
| 2 | handoff | a clip is missing: write `output/runs/dayNN_clips_manifest.json`, append a `needs_clips` alert, exit 10. `--watch` polls until both clips exist and their size is stable, then continues |
| 3 | gates | G1-G10 raw-clip drift gate (`docs/mission/2D_CONSISTENCY_PLAN.md` §3d) via `tools/source_gate_smoke.clip_gate` + `consistency_metrics` + `startframe_gates` + `gymform.motion_check` (exercise = day config `motion_standard`). Hard: G1-G8, G10. Warn-only: G9 (not yet wired, D14) |
| 4 | run_day | `run_day.py --day N --config <run copy>`: the day config with both clips + windows swapped in and every output path pointed into the run dir. It does the Day 4 grade (`native_grade`, `native_muscle_shader: day4`, `muscle_glow_clamp` + `target_regions` via `gymform/region_glow.py`), the Day 4 template (title, preview pill, callout, see-through), TTS + beat/chimes (`gymform/tts.py`), the master mp4, the preflight gate, `gymform.qc` and the step sheet |
| 5 | verdict | `dayNN_result_sheet.jpg` (1 fps contact sheet of the master), `result.md`, `verdict.json`; on failure a short summary + an alert |

Clip window default: `i0 = 0` (stretch, the pinned start) to `i1 = first keyframe time × fps` (pinned contraction);
run_day's cosine wave loops it. Override per form with `"window": {"i0": .., "i1": ..}`.

## Box priority
Everything runs at `nice -n 19`. Before each heavy stage the script waits while a foreign `flipcut.py`,
`geo_template`, `blender` or `ffmpeg` runs, or while the load is above the core count. During a stage it SIGSTOPs its
own process group while such a process runs and resumes it after (our own ffmpeg children are excluded).

## Exit codes (wake the bot only on non-zero, except 10 which is the generation handoff)
| code | meaning | bot action |
|---|---|---|
| 0 | PASS or PASS_WITH_WARNINGS (master built, QC pass) | review `result.md` / sheet (warnings = judgement items) |
| 2 | config invalid (incl. playbook sha) | fix the config |
| 3 | input missing (stills / keyframes / unreadable clip) | make or fix the input |
| 4 | raw-clip gate fail | read `gates.json`; name the failed metric before any retake (no blind re-roll) |
| 5 | preflight fail | fix by DEFECT_PLAYBOOK row (render param, never a threshold) |
| 6 | render / QC fail | same |
| 7 | run_day guard / lint refusal | fix the day config (callouts, step order, camera plan, caption) |
| 8 | box busy too long | re-run later |
| 9 | internal error | read `pipeline.log` |
| 10 | awaiting clips | generate per the manifest |
| 11 | watch timeout | clips never arrived |

Alerts: one JSON line per event in `/home/box/routinizer/alerts.jsonl` with `"source": "gymform_day"`, `kind`
(`needs_clips`, `gate_fail`, `preflight_fail`, `render_fail`, `run_day_guard`, `box_busy`, `config_invalid`,
`input_missing`, `pipeline_error`, `watch_timeout`), `severity`, `message`, `run_dir`, `verdict`, `exit_code`.

## GenerateVideo handoff manifest (`gymform.clips_manifest/v1`)
```json
{"schema": "gymform.clips_manifest/v1", "day": 7, "status": "awaiting_clips",
 "clips": [{"form": "wrong", "target_path": "/abs/.../day07_wrong.mp4", "exists": false,
   "generate_video": {"prompt": "...", "image_path": "/abs/start.png", "last_frame_path": "/abs/start.png",
     "keyframes": [{"image_path": "/abs/contraction.png", "timestamp_s": 2.2}, {"image_path": "/abs/contraction.png", "timestamp_s": 2.8}],
     "duration": 5, "resolution": "720p", "aspect_ratio": "9:16", "filename": "day07_wrong.mp4"},
   "after_generation": "save/copy the returned mp4 to target_path; do not re-grade or re-encode it"}, {"form": "right", "...": "..."}],
 "resume": {"watch": "...--watch", "once": "..."}}
```
`generate_video` maps 1:1 onto GenerateVideo's arguments. One take per clip.

## Pipeline config (`configs/pipeline/dayNN.json`, or a `"pipeline"` block inside the day config)
See `configs/pipeline/day07_d7v2_t2.json`. Keys: `day`, `day_config`, `forms.{wrong,right}.{start, keyframes[{path,t}],
clip, prompt | prompt_vars, window}`, optional `last` (last-frame pin, default = start), `prompt_template`, `generate`,
`gates.{hard,warn}`, `gate_view`, `workers`, `out_root`, `day_config_overrides`.

## Outputs (`output/runs/dayNN_<YYYYmmdd_HHMMSS>/`, a new dir per run, never overwritten)
`dayNN_master.mp4`, `dayNN_step_sheet.jpg`, `dayNN_result_sheet.jpg`, `dayNN_caption.txt`, `gates.json`,
`run_day.json`, `verdict.json`, `result.md`, logs, `qc/`, `qc_preflight/`, and the run copy of the day config.
`_work/` (copied clips, kp cache, frames) is removed at the end unless `--keep-work`. Note: run_day.py itself still
updates `output/qc/dayN_status.json`, `output/qc/production_times.jsonl` and `_cache_gymform/` like any run_day run.
