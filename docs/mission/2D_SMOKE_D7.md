# 2D consistency smoke test: Day 7 Seated Cable Row (t633s2233, 2026-11-03 KST)

Plan: `2D_CONSISTENCY_PLAN.md` §5. No grok.com, no xAI API, no full render. Everything ran at low priority (`nice -n 19`) at 360×640.
Sheet: `output/_consistency/smoke_d7/smoke_d7_sheet.jpg`. Gate JSON files: `gate_t1.json`, `gate_t2.json`, `gate_v3_baseline.json`.

## What was generated (Grok Bot tools only)
**Start frames (reused):** `inputs/day7/day7_v3_{wrong,right}_start_720.png`.

**Contraction keyframes** (GenerateImage edit of each start frame, saved in `smoke_d7/keyframes/`):
- **Wrong:** 1 try. The handle is pulled to the belly. The flare and shrug are mild.
- **Right:** try 1 barely changed the pose and was rejected. Try 2 (`right_contraction_720.png`) has the handle at the belly with the elbows bent.
- **Right glow fix (try 3):** an extra edit that asked for the start frame's glow size gave 5.1% glow, against 5.2% in try 2, so it was not used.
- Both keyframes used pass `tools/startframe_gates.py`. Backdrop 35/36/41, figure − backdrop 34.6 and 37.6.

**Clips** (GenerateVideo, prompt = plan template; settings below):

| Take | Settings | Files |
|---|---|---|
| try 1 | image_path = last_frame_path = start frame, keyframes [{contraction, 2.5 s}], duration 5, resolution 480p, 9:16 | `d7_smoke_{wrong,right}_480p_t1.mp4` (480×848, 121 f) |
| retry (1 per clip) | same, resolution **720p** | `d7_smoke_{wrong,right}_720p_t2.mp4` (720×1280, 121 f) |

Why the retry was at 720p: at 480p, even a perfect copy of the start frame only reaches edge IoU 0.885–0.894 (start → 480p → back), which is at the G1/G2 bar of 0.88. So 480p can't be judged by G1/G2. Try 1 G1: 0.965/0.81 and 0.969/0.80.

## Gate results (720p retry; v3 for comparison)
| Check | Wrong (new) | Right (new) | v3 wrong | v3 right |
|---|---|---|---|---|
| G1 start hold, SSIM / edge (≥0.97 / ≥0.88) | **PASS** 0.993 / 0.897 | **PASS** 0.992 / 0.894 | fail 0.955 / 0.826 | fail 0.959 / 0.797 |
| G2 end hold, then keyframe ±6 f | **fail** 0.978 / **0.879**; 0.986 / **0.876** | **fail** 0.987 / **0.875**; 0.989 / 0.885 | fail 0.872 / 0.601 | fail 0.881 / 0.643 |
| G3 glow vs start (0.6–1.6×, cv ≤ 30%) | PASS 0.72–1.10×, cv 14.6% | **fail** 0.50–1.06×, cv 27.2% (0.84× of the keyframe) | pass 0.70–0.97× | fail 0.43–0.94× |
| G4 glow off the body (≤ 0.5%) | PASS 0.01% | PASS 0.05% | pass | pass |
| G5 backdrop SSIM (≥ 0.90) | PASS 0.982 | PASS 0.958 | pass | pass |
| G6 backdrop drift (≤ 4) | PASS 1.0 (f0 36/37/44) | PASS 1.0 (f0 35/36/43) | pass 1.0 (f0 27/31/39) | pass 2.3 (f0 30/32/38) |
| G7 view spread (≤ 0.05) | PASS 0.015 | PASS 0.015 | 0.028 | 0.028 |
| G8 cobalt cv / ΔRGB / between | PASS 0.3% / 3.3 / 8.7% | PASS 0.7% / 3.7 / 8.7% | pass (ΔRGB 12.2) | pass (ΔRGB 10.3) |
| G9 motion_check | n/a: the only measurable wrong signature is lean-back, not this mistake | pass (fail items); elbows_past_sides warn, elbow max 40° | n/a | pass, warn 61° |
| G10 separation, figure − backdrop (≥ 33) | PASS 37.8 | PASS 38.9 | 47.1 | 43.9 |
| Skin L p50 over the clip (start) | 35–41 (36.3) | 51–57 (56.0) | 24–33 (36.3) | 43–50 (56.0) |

**Result:** G1–G8 are not all passing. Wrong fails G2 by 0.001–0.004 on edge IoU. Right fails G2 by 0.005 and fails G3.

## Findings
- The pins work for appearance:
  - The start/end hold improved from 0.87/0.60 to 0.98/0.88.
  - The tone stopped darkening: wrong skin range is 6 L, against v3's 9 L and a 30% darkening.
  - The backdrop stays at about 35/37/43 instead of 27/31/39.
  - Equipment ΔRGB fell from 10–12 to 3–4.
- **The right-clip glow fails because of the keyframe, not the generator.** At the contraction the arms hide part of the lats, so the keyframe has 5.2% glow against the start's 8.9%, and the clip follows the pin. G3 as written (vs the start frame only) can't pass for a pose that occludes the target muscle. A spec change for owner/agent review would be to compare against the pinned frame interpolated over time. That rule is not applied here.
- **Motion is the weak point.** The wrong clip does about 2 reps (elbow peaks near 0.8 s and 4 s) despite "one rep". The right rep is shallow: 2D elbow max 40°, against 61° for v3.
- The Day 4 grade (`gymform/native.py`, `configs/day07_muscle2.json` unchanged, only the files and windows swapped) gives the same backdrop and tone on all frames of the new clips.
