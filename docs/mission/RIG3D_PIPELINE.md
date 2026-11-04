# rig3d v2 pipeline (Routinizer, 2026-11-04 KST)
One command, no bot tokens, 0 generations, no xAI, never posts (ig_publish.py is not called):
`nice -n 19 python3 tools/rig3d/pipeline.py --exercise db_row`

| # | Stage | Code | Cache key |
|---|---|---|---|
| 1 | config | `exercises/<slug>.yaml` | file text + auto-fix overrides |
| 2 | fetch | YMove clip + Poly Haven CC0 HDRI → `_scratch/` (gitignored) | URLs |
| 3 | pose + source check | `clip_pose.py` (MediaPipe). Hard gates: full body, ≥2 reps. Hinge/jitter are corrected downstream | clip sha |
| 4/5 | retarget + render | `retarget_render_v2.py` (bpy): floor levelled, person scaled to the rig, torso clamped to 45°, feet locked, spine bend spread, dumbbell baked to the hand-bone grip, tight side camera, HDRI, beauty + muscle-mask pass | motion/render cfg + code |
| 6 | Day 4 grade | `grade3d.py`: skin lock p50/75/90 = 64/75/88, `gymform.muscle.day4_fibre` with activation = rendered mask (lats, rhomboids, mid trap → green), flat 35/39/43 backdrop, Kaushan title see-through 0.40 | grade cfg + code |
| 7 | reel | edge-tts (locked voice), caption build + lint, rep looped, 1080×1920 30 fps | tts/caption cfg |
| 8 | QC | `qc3d.py` → `qc.json`, 22 checks (rig, look, reel). A failure with a rule gets 1 retry. Anything else gets an AI-queue item | qc cfg |
| 9 | sheet | `sheet.py` → PNG (frames, Day 4 ref, mask overlay, curves, every check) | — |
| 10 | deliver | `deliver.py` (see below) | — |

Every stage is resumable: stamps are kept in `_cache_rig3d/<slug>/stamps`, and outputs go to `output/_rig3d/<slug>/` (both gitignored). Use `--force <stage>` to re-run a stage and everything after it. `--short N` is a cost probe of N frames. Heavy stages wait in `/home/box/routinizer/rz_loadgate.py` first.

## Delivery (no bot turn)
- **Found on the box:** no SMTP, Gmail app password, Telegram or ntfy setup, in the env or in `/home/box/{routinizer,.seori,.kairo,.soom}`. The only thing that exists is the GitHub Pages repo `been2531/been2531.github.io`.
- **Default: `repo`.** Commits `results/rig3d/<slug>/<run_id>/` (sheet.png, qc.json, caption, run.json, a 540p preview of our own render) to this public repo, and appends one line to `/home/box/routinizer/state/alerts_for_routinizer.jsonl` with the GitHub link. That link opens on a phone. It is a pull, not a push.
- **`ntfy`:** true phone push. Enabled when `GYMFORM_NTFY_TOPIC` is set, after he installs ntfy and subscribes to that private topic.
- **`smtp`:** email with the PNG attached. Enabled when `GYMFORM_SMTP_USER`, `GYMFORM_SMTP_APP_PASSWORD` (a Gmail app password) and `GYMFORM_NOTIFY_TO` are set.
- **`pages`:** dry-run only. Publishing to it would make the sheet public on github.io.

## AI wake
A QC failure with no rule writes one item to `/home/box/routinizer/ai_queue/fitness_studio.jsonl`, in the rz_ai_queue item shape and deduped by id, plus a warn alert.
