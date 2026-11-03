# Fundamental production plan for @gymform.3d (t694u/t695u) — 2026-11-04 KST
0 generations, 0 renders. The deliverable is this note only; the owner asked for no sheet image (t695u).

## Problem (owner, t694u)
Every new machine restarts the plate → still → pre-check → video loop and burns the weekly budget.

## Evidence from the repo
- Days 3–6 (squat, RDL, lateral raise, incline DB press) shipped. The equipment was a free weight moving rigidly with the hands, with no static structure that hands must stay connected to (ROOTCAUSE_D7_MOTION §2). D1/D2 machines were cropped or used as a backdrop.
- Day 4's look came from a generated sculpted-muscle body plus the native Day 4 grade (ROOTCAUSE_MUSCLE_LOOK, recommend A).
- Day 6 needed v1–v5 to pass QC. The failures were look/QC issues (dumbbell bloat, backdrop, separation), not motion.
- Day 7 (cable low row), from the commit log:
  - grok.com v1/v2: wrong exercise, invented machine;
  - 2D smoke 480p+720p: G2/G3 fail;
  - v3 pins: preflight 25/33;
  - machine-first: the mannequin failed 2/2;
  - glass stills: 3 images;
  - option B: 1 video FAIL 2/5;
  - optB2: 3 images FAIL;
  - V handle: 2 images (1 usable with warnings).
  - That is ≥10 images and ≥4 videos with no publishable reel.
  - Causes: the video model redraws the machine every frame; unpinned spans collapse; a cable must join static and moving parts.
- Blender pilots R1–R6: press-like path, floating or crossed arms, smooth body. The look can't reach Day 4 natively; CPU-only Cycles takes about 3.5 min/frame.

## Options
Success rates below are qualitative estimates from the evidence above, not measurements.

### A. Free-weight / bodyweight / barbell exercise catalogue only (drop cable machines)
- **How:** Day 3–6 method. One start frame per form, then i2v, then the unchanged Day 4 pipeline (run_day, native grade, QC).
- **Generations/reel:** 2 images + 2 videos, plus ≤1–2 retries.
- **Success estimate:** high for motion and equipment, because 4 shipped days share this equipment class. Medium for the look on the first try, because Day 6 needed several QC rounds.
- **Box time:** about 1–2 min render plus QC.
- **Reuses:** everything (configs, QC, DEFECT_PLAYBOOK, template).
- **Risks:** topic variety. Machines can still appear if the generator draws them reliably as a static backdrop (bench, rack), with no moving connection.

### B. One master character + per-exercise single first frame with code checks (adds to A)
- **How:** generate the sculpted Day 4 mannequin once as a reference sheet. Each exercise reuses it, so a reel needs only 1 first frame per form. The still pre-check runs only the body checks (head = body, light, glow, pose); no plate or machine checks.
- **Generations/reel:** 2 images + 2 videos (the master is a one-time cost of about 2 images).
- **Success estimate:** higher look consistency than A alone, because the identity is fixed. Not yet tested.
- **Box time:** same as A.
- **Reuses:** tools/still_precheck.py body checks.
- **Risks:** the reference may not hold on the video model; that needs one smoke test.

### C. Keep machines with a real-plate pin per machine (current D7 path)
- **Generations/reel:** at least 1 plate + 2–4 stills + 2 videos, with retries. D7 evidence: ≥14 generations and no pass.
- **Success estimate:** low for cable machines, per the D7 history.
- **Risks:** repeats the budget burn for every new machine.

### D. Blender deterministic machine + pose, Day 4 grade on top
- **Generations/reel:** 0.
- **Success estimate:** low. Pilots R1–R6 failed on anatomy and IK; the owner rejected the smooth body.
- **Box time:** hours/reel on CPU, plus days of setup.
- **Risks:** amateur look.

### E. Reuse the shipped Day 1–6 clips (filler, 0 generations)
- **How:** re-edit passed graded clips into new angles ("3 mistakes", slow-mo breakdown, compare two days).
- **Generations/reel:** 0.
- **Success estimate:** high technically.
- **Risks:** repeated footage. Never repost DfGabTBjwt8 as is; a new edit needs owner approval.

## Recommendation: A + B (E as a budget buffer)
- **Programme:** only free-weight, bodyweight or barbell exercises (dumbbell rows, curls, presses, lunges, hip thrusts, push-ups, ...). Cable and complex machines are dropped.
- **Master character:** made once; then 2 first frames + 2 videos per reel, run through the existing Day 4 pipeline.
- **Weekly cost:** 3 reels ≈ 6 images + 6 videos, plus retries. If the budget is short, make 2 new reels plus 1 E re-edit.
- **D7 low row:** parked, not published.
- **First step (after approval):** a master-character smoke on one dumbbell exercise: about 2 images + 2 videos, then the Day 4 render and owner review.
