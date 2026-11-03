# 04 — Proposal (awaiting owner approval; nothing implemented)

## Verdict: generated clips vs a Blender-rigged pipeline
Evidence:
- 4 of the top 5 benchmarks (Demic, Muscle & Motion, SynerMuscle, Anatomy Lab) use one consistent rigged 3D character. What they do that we can't do with clips: identical character every reel, A/B synced reps, orbit + layer peel (skin → muscle → bone), seamless loops, labels locked to muscles.
- Our generated route: Day 6 preflight passed 11/82 (13%), full renders 8/14; Day 7 v1 clips changed the exercise (row → pulldown/fly); D8 step-cut jump is "by construction" (separate generations).
- studio3d was rejected for its **look** (smooth MakeHuman body, no muscle form, bone-only X-ray; `output/day6_3d_proof_vs_day4_t562.jpg`), not for motion accuracy. Its motion side (true 3D angles, L1–L4 set lock) worked.
- Render cost: Cycles 3.5 min/frame measured on 1 CPU (box now has 8 CPUs, no GPU). A full 20 s × 30 fps reel in Cycles is hours; key poses are minutes.
Conclusion: do NOT replace generated clips now. Stage it:
1. Stage A (Day 7+, no new tools): generation spec changes — same start frame as first AND last frame (seamless loop, kills the step-cut), pinned bottom/top keyframes for full ROM, one start-frame set per exercise.
2. Stage B (pilot, 1 exercise): Blender = motion truth. Rig the CC0 mannequin with MPFB2/Rigify, IK hands/feet to equipment, wrong/right Actions from `motion_standards.json`, render only key poses → stylise to the Day 4 look → feed as GenerateVideo keyframes. Gate: owner A/B vs Day 4, joint angles within ±10° of the standard.
3. Stage C (only if B wins): Blender look development (muscle-defined mesh or normal map, Day 4 fibre shader as a material), orbit cameras, layer transitions, loops rendered natively.

## Target architecture
Shared infra (all series): `exercise library` (schema below) → `spec` (script, shots, callouts) → `source` (generation or Blender) → `look` (native tone map) → `overlay/caption/TTS` → `QC` → `approval` → `publish` → `insights`.
Independent features: generation adapters (grok.com, Grok Bot video, Blender), look presets (Day 4), overlay templates, per-exercise configs.
Reuse: run_day.py stages + hash cache, gymform/native.py, qc.py/temporal_qc.py, DEFECT_PLAYBOOK, motion_standards.json, tts.py, caption.py, ig_publish.py, ig_analysis/fetch.py, assets3d, studio3d set-lock QC.

## Exercise schema v2 (extends configs/motion_standards.json)
`id, name, primary_muscles[], secondary_muscles[], equipment, start_position, phases[{name, joint_targets{joint: deg}, duration_s}], rom{joint:[min,max]}, right_form[], mistakes[{id, signature, mistake_type}], tempo, reps_loop, cautions[], camera{views[], closeups[]}, assets{start_frames, clips, blender_action}, sources[]`
Coverage of the 10 requested lifts: in motion_standards today → squat, bench press, shoulder press (barbell OHP), biceps curl, lat pulldown, RDL (conventional deadlift missing), Bulgarian split squat (forward lunge missing). New, with sources found: push-up (ACE /41/push-up), pull-up (ACE /191/pull-ups), forward lunge (ACE /8/forward-lunge), deadlift (ACE /6/deadlift + ProSource 2015). Leg extension: source still to find.

## Direction upgrades (ranked by evidence)
1. Frame-0 hook: target muscle already lit + ✗/✓ label (all top 5).
2. Seamless loop (SynerMuscle, Demic) via first = last frame.
3. A/B or split screen wrong vs right with synced tempo (M&M, 3D muscle anatomy).
4. One close-up on the target muscle per reel (M&M, Demic).
5. Layer transition skin → muscle → bone (SynerMuscle, Anatomy Lab) — needs Blender (Stage C).
6. Only measured numbers on screen (angles); remove unsourced % claims.
7. Collision / joint-break QC: Blender stage gets mesh–equipment intersection + joint-limit checks; clip stage keeps (d)/(pl)/D3 grip checks.

## Auto-QC redesign (goal: first-render pass rate)
- Shift left: a RAW-CLIP acceptance gate runs before any grading (exercise identity via motion_check signatures, ROM, handle integrity, camera lock, glow area vs Day 4, backdrop separation). Reject the clip, not the render.
- Three tiers: Source gate → Look gate (deterministic global map, should almost never fail) → Temporal gate (output/raw ratios).
- Wire `gymform.motion_check` into preflight (D14 says it is not wired). Install pytest (free) and run `tests/` in CI-like pre-commit.
- KPI logged per run in `production_times.jsonl`: first-preflight pass, first-render pass, wall time. Target: ≥70% first-preflight after Stage A.
- Caption/callout lint: block numbers that are not measured or sourced.

## Weekly 3-reel operating plan
- Sun: Grok research (benchmark refresh, last week's insights) → pick 3 exercises from the library → specs.
- Mon: start frames + clips for all 3 (batch, 1 take each) → raw gate.
- Tue: render + QC → owner review of finished mp4s (human).
- Post Mon/Wed/Fri at the time from `online_followers` (endpoint already called in `ig_analysis/acct.py`); owner uploads or approves `ig_publish.py --publish --approved`.
- Insights D+1 and D+7 per reel (views, reach, avg watch ÷ length, saves ÷ reach, shares ÷ reach). Baseline: lat pulldown 15,103 views / 104 saves / 10.8 s avg watch; lateral raise 1,822 / 4 / 6.0 s.

## Roles and cost
- Grok Bot: research, specs, generation calls, evaluation, approval requests, insights analysis.
- Cursor: code in this repo (schema, gates, Blender pilot, insights loop), tests.
- Cost: free/included only. Paid flags: none required. Possible later: Remotion Company licence only if the business passes 3 employees.

## Top 5 Cursor work specs (awaiting approval)
1. Loop + ROM generation spec (Stage A): run_day supports a step whose clip loops (first = last), tempo continues across the cut; prompts/keyframes per exercise. Accept: (tq) cut ≤ 25 px hips/ankles, Δbrightness ≤ 3, D8 passes on Day 7.
2. Raw-clip acceptance gate: new `gymform/source_gate.py` + motion_check in preflight; writes `raw_gate.json`; logs first-pass KPI. Accept: Day 7 v1 clips (row → pulldown) are rejected, v3 accepted.
3. Exercise library v2 for the 10 lifts + callout/caption lint (no unsourced %). Accept: schema test passes, all 10 have ≥1 source URL, Day 6/7 callouts pass lint.
4. Blender motion-truth pilot (lat pulldown): MPFB2 CC0 rig + Rigify, IK to bar/seat, wrong/right Actions, 3 key poses × front/side stills + angle JSON + intersection check. Accept: angles within ±10°, 0 mesh–equipment intersections, owner A/B review. No publishing.
5. Insights loop: scheduled read-only fetch → `analytics/reels.jsonl` + weekly scorecard; no token in output. Accept: 5 existing reels backfilled.

## Human-in-the-loop
Publish approval; look changes (character, palette); grok.com quota spend; Blender pilot go/no-go; music choice in the IG app; any Adobe ID / account sign-in; any licence with share-alike (Z-Anatomy).

## Long-term moat
One owned, consistent character and locked set; every number measured or sourced; a sourced exercise library; a defect playbook + gates that raise first-pass rate; an insights loop tying hook/length/exercise to saves.
