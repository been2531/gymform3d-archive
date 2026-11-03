# 2D consistency plan: Day 4 quality every time (t633u, 2026-11-03 KST)

Scope: the existing 2D pipeline (generated clip → Day 4 grade, `shade_day4_3d_frame_1080p` / `gymform/native.py` +
`gymform/muscle.py day4_fibre`). The 3D Blender path is parked. **Evidence only in this pass:** no clip generated, no
render, no grok.com / xAI API, nothing deleted. Measurements are low-res (360×640, 9 frames per clip, `nice -n 19`,
≈25 s CPU in total).

Tools (new, read-only):
- `tools/consistency_metrics.py` → `output/_consistency/metrics.json` (per-clip + between-clip metrics, frame strips)
- `tools/consistency_sheet.py` → `output/_consistency/consistency_sheet_d4_vs_d5_d6_d7.jpg`
- `output/_consistency/startframe_vs_clip.json` (start frame vs clip frame 0 / last frame) and
  `masters_graded.json` (graded masters, `tools/consistency_masters.py`)

Metric definitions:
- **skin** = MediaPipe torso polygon, glow-free pixels (chroma < 18), luminance p50.
- **view** = 2D shoulder width ÷ torso length. This is a camera-yaw proxy: about 0.5 means rear or front 3/4, about 0.8 means an elevated front 3/4.
- **glow** = red/green-dominant pixels as a share of the person silhouette. **cv** = std/mean over the clip.
- **bgΔ** = RGB distance of the top backdrop strip to 35/39/43.
- **SSIM** is computed on grey frames at 360×640. **Backdrop SSIM** uses only pixels outside the dilated union of all silhouettes.
- **Edge IoU** compares Canny edges of the start frame and clip frame 0 after gain normalisation.

## 1. Evidence

### 1a. What Day 4 actually was (provenance: `build_day4_rdl_master.py`, `docs/ROOTCAUSE_DAY6.md`, git aeeb806)
- **One** base i2v generation (`day4_rdl_step1_i2v.mp4`, 480×848). The 720p "ecorche" and "locked" clips are
  **motion-preserving video edits** of the base renderings, not new generations. Step 3 literally re-uses the step 1
  ecorche clip (`STEP_SPECS_DAY4[3] = day4_rdl_step1_ecorche_720p.mp4`).
- Result: the same body, barbell and rear-3/4 camera in all 4 steps. **view 0.52–0.56 (spread 0.037) in the raw clips and 0.556–0.581
  (spread 0.025) in the graded master.** Base vs edit frame 0: view 0.479 vs 0.488.
- Standing, free silhouette with backdrop all around and nothing between the legs (feet contrast 33–41, ROOTCAUSE_DAY6).
- The video edit was the xAI `/v1/videos/edits` endpoint (`gymform/xai_edit.py`, about $0.40 per clip), which is **now off-limits**
  (no paid xAI). This is why "do what Day 4 did" can't be copied literally; the plan below replaces the edit with a
  shared start-frame set plus first/last-frame pins, which are free in Grok Bot GenerateVideo.
- The raw Day 4 tone was **not** stable: skin p50 32–56 across steps (spread 24 L) and 39–65 inside step 4 (lighting
  changes as the torso hinges). The grade absorbed it: the master backdrop is exactly 35/39/43 in every step.
  ⇒ **Global tone drift is the grade's job and is already solved. Geometry, identity and glow drift are not.**

### 1b. Between clips of one reel (clips used in the master)
| Day | Generations | Frame-0 SSIM between clips (min) | view spread | skin spread (raw) | Notes |
|---|---|---|---|---|---|
| D4 RDL | 1 base + edits | 0.74 (pose differs at f0) | **0.037** | 24 L (graded away) | same body/barbell/camera |
| D5 Lat raise | 4 independent | 0.667 | **0.092** | **57 L** (82–138) | two different mannequins (bright silver vs grey), front vs back view, floor mat only in s2. Graded master: skin 64–150, view 0.52–0.72 |
| D6 IDP v5 | 2 from one start-frame set | 0.874 | 0.021 | 5 L | consistent |
| D7 Row v3 | 2 from one start-frame set | 0.856 | 0.037 | 16 L | wrong clip darkened by the generator (see 1d) |

Across days, the mannequin was re-designed every day with no canonical reference. Raw skin: D4 dark ecorche 32–56,
D5 silver 82–138, D6 charcoal 52–59, D7 charcoal 29–45. The equipment changed too: chrome barbell, black hex
dumbbells, then palette A cobalt in D6–D7 (cobalt RGB raw ≈ 20–35 / 29–53 / 123–141).

### 1c. Within a clip (9 samples)
- **Camera is NOT the problem.** Backdrop SSIM vs f0 is ≥ 0.97 for every D5–D7 clip and 0.92–0.93 for D4 (upscaled 480p
  noise). Backdrop gain drift is ≤ 5 lum everywhere.
- **Glow area is the main within-clip drift:**
  - D4 used clips: cv 10–25%.
  - D5 s4: **cv 78%** (green spreads from the delts to the whole arm). D5 s1: 36%.
  - D7 v3 right: **cv 32%**, falling from 8.9% to 3.7% of the body (0.42× the start frame).
  - D7 v1 right: 2.4% at the start frame, then **18.9–22.5%** (8×, green over the whole torso).
  - D6 v5 right: cv 3% (good).
- Silhouette area and limb width vary with pose (sil_area_norm cv 4–17%). At this resolution the limb-width metric is
  pose-confounded (arm foreshortening, bar or handle occlusion), so it is **not** used as a gate. Limb thickness is
  protected by construction (start-frame hold) and by D7's output/raw forearm check in the grade.

### 1d. Does the generator hold the start frame? (`startframe_vs_clip.json`, `tools/consistency_startframe.py`)
| Clip (generator) | SSIM start→f0 | edge IoU | start→last | skin start→f0→last | backdrop f0 | glow start → clip |
|---|---|---|---|---|---|---|
| D6 v5 right (grok.com) | 0.992 | 0.91 | 0.919 | 55.6→52.9→51.9 | 33/37/41 | 6.5% → 6.0–6.8% |
| D7 v2 right (grok.com) | 0.993 | 0.90 | 0.907 | 54.5→52.5→51.4 | 33/37/41 | 3.6% → **1.1**–3.5% |
| D7 v3 wrong (Grok Bot GenerateVideo, first frame only) | **0.938** | **0.83** | 0.846 | 36.3→32.3→**25.5** | **27/31/39** | 22% → 15.6–21.2% |
| D7 v3 right (GenerateVideo, first frame only) | **0.943** | **0.80** | 0.854 | 56.0→50.4→**44.3** | **30/32/38** | 8.9% → **3.7**–8.6% |

- GenerateVideo with only `image_path` re-renders frame 0 instead of copying it: tone-normalised SSIM 0.955–0.959 and
  edge IoU 0.80–0.83, against 0.993 and 0.90–0.91 for grok.com.
- The tone then darkens about 20–30% over 6 s. Sharpness is fine (Laplacian variance 163–290 against 52 for grok.com v2), so this is not
  low resolution.
- Neither generator holds the glow region without a pin.
- No clip so far used `last_frame_path` or `keyframes`, so nothing pulled the appearance back.

## 2. Ranked root causes
1. **Each step is an independent generation with no shared visual anchor** (D5). This changes the mannequin
   (skin spread 57 L, two body types), the view (spread 0.092 against 0.037) and the props (floor mat). Day 4 avoided it
   by construction: one base generation, edits of it, and re-use of step 1 for step 3. *Fix: one locked start-frame set
   per exercise; both clips start from it; re-use clip windows across steps.*
2. **Only the first frame is pinned, so the appearance decays through the clip.** Glow area swings up to cv 78% and 8×;
   GenerateVideo re-renders frame 0 (edge IoU 0.80) and darkens tone 20–30%. *Fix: first = last frame (loop) plus one
   contraction keyframe; 5 s clips; a gate on glow and start/end hold.*
3. **No canonical character or equipment reference across days.** The mannequin and props are redesigned per day
   (skin 29–138 raw across D4–D7). *Fix: one canonical charcoal mannequin + palette A reference sheet, used as the
   reference for every start-frame edit, with start-frame gates (`tools/startframe_gates.py`).*
4. **Motion identity is unguarded at the source.** D7 v1 turned a row into a pulldown/fly (prompts.md), from an
   ambiguous start frame and a free motion path. *Fix: start at the stretch, pin the contraction keyframe, and wire
   `gymform.motion_check` into a raw-clip gate.*
5. **Low-separation scenes** (seated, bench between the legs, dark machine). These are D6 root causes (ROOTCAUSE_DAY6 §3.2)
   and still apply. *Fix: prefer free-standing silhouettes and a light warm-grey machine; keep (sep) D17 at the start frame.*
6. **Not a cause:** the camera inside a clip (backdrop SSIM ≥ 0.92), the clip length (D4 6.04 s and D5 5.04 s both
   drifted or held depending on the anchor), and the Day 4 grade (deterministic; it absorbed D4's 24 L raw spread).

## 3. Fix spec (for Cursor after approval; grade unchanged)

### 3a. Locked reference start frames, one set per exercise
- `inputs/canon/mannequin_ref.png`: one canonical charcoal mannequin (Day 4 skin target p50 64 / p75 75 / p90 88 after
  the grade; never brightened) with matte-black shorts. `inputs/canon/palette_A.png` is
  `docs/plate_palette_A_reference.jpg`: matte cobalt (45,75,170), brushed-gold rim, steel handle, warm-grey frame.
- Per exercise: `inputs/dayNN/start_stretch.png`, made once with GenerateImage using both canon refs as references.
  Then `start_contraction.png` (same image edited: pose only), and the wrong/right variants are edits of the SAME
  frame, so the camera, machine and body are pixel-shared. D7 v2 already did this (frame-0 SSIM between clips 0.856).
- Graded with `tools/startframe_grade.py <raw> <out> 38.3 67 35,39,43 80`, gated with `tools/startframe_gates.py`
  (existing D17/D18 thresholds), and frozen: sha1 recorded in the day config `source.start_frames`.

### 3b. Generation settings (Grok Bot GenerateVideo)
- `image_path` = start_stretch and `last_frame_path` = start_stretch, so the clip is a seamless loop. This kills the step-cut jump (D8) and
  pulls the appearance back at the end.
- `keyframes` = [{start_contraction, t = duration/2}]: pins full ROM and the motion direction.
- `duration` 5, `resolution` "720p" (pins rule out 1080p; 480p for smoke tests), `aspect_ratio` 9:16.
- One take per clip. A rejected take is NOT re-rolled blindly: the gate report names the failed metric first.
- Re-use clip windows across steps (Day 4 pattern): a 2-step muscle reel needs 2 clips (wrong/right); a 4-step reel
  re-uses the same 2 clips for bone and muscle steps via the grade, not 4 generations.

### 3c. Prompt template (short; the frames carry the look)
```
<Exercise>, done <wrong|right>: <one sentence of the joint action>. Start and end exactly as in the first frame;
reach the pinned contraction pose at the middle. One smooth rep, even tempo. Locked static camera, no zoom, no pan.
Same matte charcoal-grey mannequin, same matte-black shorts, same <equipment, palette A> in every frame; nothing
appears or disappears. The <red|emerald> glow stays exactly where and as large as in the first frame, on the
<target muscle> only. Flat, even, shadowless light; flat dark charcoal backdrop. No text, no logo.
```

### 3d. Raw-clip drift gate (`gymform/source_gate.py`, runs before any grading; reject the clip, not the render)
Measured at 360×640 on 9 frames + the pinned frames; code = `tools/consistency_metrics.py` functions.

| # | Check | Threshold | Calibration (measured) |
|---|---|---|---|
| G1 | Start hold: tone-normalised SSIM(start, f0) and edge IoU | ≥ 0.97 and ≥ 0.88 | grok.com 0.993 / 0.90–0.91 pass; GenerateVideo first-frame-only 0.955 / 0.80–0.83 fail |
| G2 | End / keyframe hold (when pinned): same metrics vs the pinned image | ≥ 0.97 and ≥ 0.88 | unpinned ends were 0.85–0.95 |
| G3 | Glow area vs start frame | every sample 0.6–1.6× start and cv ≤ 30% (if start glow < 2%: absolute range ≤ 1.5 pt) | D4 cv 10–25 pass; D5 s4 78, D7 v3 right 0.42×, D7 v1 right 8× fail |
| G4 | Glow off the body | ≤ 0.5% of the frame | all clips ≤ 0.3% |
| G5 | Camera lock: backdrop-only SSIM vs f0 | ≥ 0.90 | 20 of 21 clips 0.90–0.99; D7 v1 right min 0.87 (the clip that also changed the exercise) |
| G6 | Backdrop gain drift in the clip | ≤ 4 lum | 0–5 measured (D5 s1 5.0 fails) |
| G7 | Identity between the reel's clips: view ratio spread | ≤ 0.05 | D4 0.037, D6 0.021, D7 0.037 pass; D5 0.092 fail |
| G8 | Equipment palette: cobalt pixel share cv in clip; share between clips; median cobalt RGB vs start | ≤ 10%; ≤ 20% rel; ≤ 20 RGB | D6/D7 cv 0.2–5%, between 5–9% |
| G9 | Motion identity + ROM: `gymform.motion_check` signatures | wrong fires ≥ 1 signature, right passes all | existing module, not yet wired (D14) |
| G10 | Separation at f0: (sep) D17 | skin p75 − backdrop ≥ 33, etc. | existing gate |

Tone and skin are **reported, not gated**: the Day 4 grade (`skin_tone_lock`, 35/39/43 backdrop) normalises them.
The output gates in `gymform/qc.py` / `temporal_qc.py` stay as they are.

### 3e. Grading
Unchanged from Day 4 (`native_grade` look, `day4_fibre`, skin lock p75 75 / p90 88, backdrop 35/39/43, no cut-outs).
DEFECT_PLAYBOOK rows still apply. Add one row: "D20 source drift (glow decay, start not held, identity change) →
fix: pins + start-frame set; check: G1–G8".

## 4. Generator options (free only)
| Option | Status | Use |
|---|---|---|
| **Grok Bot GenerateVideo** (included) | first + last frame, up to 4 keyframes, ≤ 15 s, 720p with pins | **Preferred.** Every lever in §3b is available here. It needs the pins (see 1d) |
| grok.com Imagine (included, quota shared with Shorts) | holds frame 0 best (0.993) | **Protect the quota.** Not used by default; only with an explicit owner OK as a fallback |
| xAI API incl. video edit (what made Day 4's edits) | paid | **Forbidden** (owner rule) |
| Local open-weights i2v (Wan 2.x FLF2V, LTX-Video, CogVideoX) | free weights, licences to verify per model | Not practical on the box: no GPU (8 CPU / 15 GB) means hours per clip, and it competes with Shorts Studio. Revisit only with a free GPU |
| Non-generative re-use | free | Re-use clip windows, mirror, ping-pong across steps (Day 4 pattern); costs nothing |

## 5. Smoke test (needs the owner's OK; nothing run yet)
- **Exercise:** Seated Cable Row (Day 7, on hold). Re-use the existing gated v3 start frames
  (`inputs/day7/day7_v3_{wrong,right}_start_720.png`, one shared set), so no new start-frame design is needed.
- **Inputs to make:** 2 contraction keyframes, one GenerateImage edit per start frame, pose only. Gate them with
  `startframe_gates.py`.
- **Clips:** 2 GenerateVideo clips (wrong and right), 480p, 5 s. `image_path` = `last_frame_path` = start frame and
  keyframe = contraction at 2.5 s. One take each.
- **Cost:** 2 images + 2 short videos from Grok Bot's own tools. 0 grok.com, 0 xAI API.
- **Check (low-res, light):**
  1. `tools/consistency_metrics.py` + G1–G8 on both clips.
  2. Grade 3 frames per clip with the Day 4 grade at 540×960.
  3. Contact sheet against the v3 clips.
- **Pass:** G1–G8 all pass on both clips, and the glow stays within 0.6–1.6× through the rep. Then show the sheet to the owner.
- **Before any full render:** 720p takes, a full render and the owner's look. **If it fails:** report which metric failed
  and why before any retake.
