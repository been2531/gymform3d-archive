# Automation plan for the dumbbell/bodyweight method (t697u) — 2026-11-04 KST
0 generations, 0 renders. Based on run_day.py, daily_pipeline.py, tools/ig_publish.py, tools/free_weight_precheck.py,
tools/still_precheck.py, tools/plate_composite_still.py, gymform/* and the db1 pilot (output/_consistency/db1).

## Recommendation
- One file-based queue (`queue/`) and one driver script, `python3 tools/reelq.py tick`.
  - The driver runs every pure-Python step.
  - When it needs a picture or a clip, it writes a batch of generation requests and stops.
- One bot wake per batch fulfils all requests of all queued reels, writes the files back, does a short eye check, and goes back to sleep.
- The next tick resumes automatically.
- Upload stays behind 중현 님's approval, which is a standing account rule. Auto-publish is his decision to make.
- Build split:
  - Routinizer builds the scheduler/wake plumbing.
  - This bot builds and owns `reelq.py`, because it is tied to the exercise configs, QC and the Day 4 pipeline.

## Constraints honoured
- No paid xAI API.
  - `gymform/xai_edit.py` reads `XAI_API_KEY` and must stay unused.
  - reelq refuses to import it, and a lint fails if it is referenced.
- Generation happens only inside a bot wake (Grok Bot GenerateImage/GenerateVideo). grok.com and Higgsfield are not used.
- Every script runs under `nice -n 19`, one reel at a time, and is skipped when the box load is high (Shorts/active-chat bots first).

## (1) Steps done by hand today that a script can do
- **Prompts:** built from the exercise config (`exercises/<slug>.yaml`).
  - Inputs: form names, torso/elbow targets, glow colour + muscle list, camera, the master reference path.
  - Template = the db1 prompts. It also emits the i2v prompts (Day 6 prompts.md pattern).
- **Precheck config:** generated from the same YAML (`extends free_weight_base.json`).
- **Still pre-check:** `free_weight_precheck.py`, already reusable. It runs on every returned still.
- **Retry decision:** a rule table.
  - A failing check maps to a prompt patch (torso FAIL → "hinge to 45°, nearly parallel"; glow missing → name the muscle; dumbbell FAIL → "both dumbbells fully visible").
  - At most 1 retry per still and 1 per clip, then a status of `needs_owner`.
- **Video QC:** the existing raw-clip gates G1–G10 plus `motion_check.py`/`temporal_qc.py`. Checks: one rep, no extra reps, equipment same shape, camera locked.
- **Loop/reverse:** `optb_motion_test.py` code (forward → hold → reverse), selected by the config flag `loop: reverse|native`.
- **Day 4 grading, layout, callouts, TTS, encode, final QC:** `run_day.py`, already one command with caches + preflight + targeted auto-fix. The day config is generated from the YAML.
- **Caption:** `gymform/caption.py` rules: 3 English lines, 0 hashtags, last line = spoken question.
- **Review sheet:** stills and video contact sheets (`free_weight_sheet.py`, the optb sheet code).
- **Git:** commit/push with an allow-list of the reel's own paths and a secret-pattern scan. Never `git add -A`.
- **Upload:** `tools/ig_publish.py`, executed only when `queue/<reel>/APPROVED` exists. Writes the permalink + media id and the never-repost list.
- **Alerts:** only for a status of `needs_owner` (QC fail after retry), `ready_for_review` (sheet + mp4 path), or published.

## (2) Steps that must stay bot-woken
- GenerateImage and GenerateVideo: there is no API path without the paid key.
- The eye check of stills and clips. The detectors already misread twice (D7 gen3 arm, handle).
  - The bot looks at the auto-made sheet only and writes `eye.json` (PASS/WARN/FAIL + a 1-line note).
  - Scripts treat an eye FAIL like an auto FAIL.
- Owner messages: topic picks, review, approval.

## (3) Queue design
- `topics.yaml` is the only file 중현 님 edits (or tells the bot). It holds a slug, the exercise and the wrong fault in one line each.
  - The script fills the rest from `exercises/<slug>.yaml`; a missing YAML → request `exercise_yaml` in the next wake.
- `queue/<reel>/state.json` stages:
  1. `planned`
  2. `stills_requested`
  3. `stills_ready`
  4. `stills_ok`
  5. `video_requested`
  6. `video_ready`
  7. `video_ok`
  8. `rendered`
  9. `ready_for_review`
  10. `approved`
  11. `published`
  - Every stage is idempotent and hashed (the run_day cache pattern).
- `queue/requests.jsonl` holds the open generation requests across ALL reels: id, kind image|video, prompt, refs, aspect, first/last frame, out path, attempt.
- **One bot wake** (Routinizer schedule, e.g. 2×/week, or a manual "배치 돌려"):
  1. Read the open requests.
  2. Generate all of them.
  3. Save the files to the out paths and write `queue/responses.jsonl`.
  4. Run `reelq.py tick` once, which does the pre-check + sheets.
  5. Eye-check the sheets → `eye.json`.
  6. Run the tick again. This may queue the videos; if the budget remains in the same wake, do them now.
  7. End.
- Cron/systemd-style `reelq.py tick` (nice 19) runs every 30 min for the pure-Python stages: render, QC, sheets, commit, notify file.
- Budget guard: weekly caps in `queue/budget.json` (e.g. 3 reels ≈ 6 images + 6 videos + ≤3 retries). Requests above the cap wait.

## (4) Expected savings (estimates, with basis)
- **Basis:** this conversation's D7/db1 tasks.
  - Each hand-run step took one parent turn + an executor run of roughly 10–30 tool calls (scripting, checks, sheet, commit).
  - db1 alone (3 stills, checks, sheet, commit) was about 15 calls.
  - A full reel by hand ≈ 5–8 owner/bot round trips (stills → approve → video → check → render → caption → review → publish).
- **With the queue:** about 1 bot wake per batch of 3 reels plus 1 review message per reel. Estimate: bot turns/tokens per reel drop to about 1/4–1/6.
- **Generations** stay the same: they are the floor, 2 images + 2 videos per reel.
- **Box time** is unchanged per reel (a 1–2 min render + QC, existing pipeline), but it runs unattended.
- **Owner time:** pick topics once per week + 1 review per reel.
- These are estimates, not measurements. Measure on the first 3 reels: log the wake count, tool calls and wall time per reel into `queue/metrics.jsonl`.

## (5) Who builds what
- **Routinizer bot:**
  - the schedule that wakes this bot for a batch;
  - the 30-min `tick` timer (nice 19, load guard);
  - the notification path (alert only on `needs_owner` / `ready_for_review`);
  - the budget counter view.
- **This bot (Fitness Studio):**
  - `tools/reelq.py` (state machine, requests/responses);
  - `exercises/*.yaml` + the generators for prompts, precheck and day configs;
  - the retry rule table, the loop module hook, the caption and the sheets;
  - the commit allow-list;
  - the publish gate (APPROVED file + `ig_publish.py`).
- Routinizer must not hold any generation or publish logic.

## (6) Biggest risks
1. **Video model behaviour on dumbbell motion** is untested in this method (db1 stopped at stills). The first batch must be a pilot of 1 reel.
2. **Detector false PASS/FAIL**, which is why the eye check stays. A wrong auto-retry wastes budget, hence the 1-retry cap.
3. **Weak right/wrong contrast** (db1: 22° vs 33°). Make a minimum contrast a config-level check (e.g. torso delta ≥ 20°).
4. **A wake fails or the quota runs out mid-batch.** Requests are idempotent and resume next wake; never regenerate a fulfilled id.
5. **Box contention:** scripts must back off under load. Shorts and active chats come first.
6. **Accidental publish or secret leak:** the APPROVED file gate, the never-repost list (DfGabTBjwt8), the allow-list commits and the secret scan.
7. **Repo drift:** other executors edit `gymform/*` concurrently. Pin the module hashes per reel (the run_day cache already does).

## Decision for 중현 님
- Keep "review before upload" (default), or allow auto-publish when auto QC + eye check both pass.
- The schedule for batch wakes (suggested: Mon + Thu).
