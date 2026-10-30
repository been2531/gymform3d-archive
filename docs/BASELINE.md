# @gymform.3d BASELINE TEMPLATE (t530u–t547u)

Owner (t530u–t532u): Day 6 fell apart because too much changed at once. **Build from what already worked in the
posted Day 1–5 reels, then change ONE element per day** (each change = its own preflight + owner look).
Contact sheet of the chosen frames: `output/baseline_best_frames.jpg` (rows: Day 5 v5, Day 4, + Day 1 X-Ray and the Day 5 native bone clip).

Posted Day 1–5 (exercise_curriculum.json): D1 `output/day1_v36_master_reel.mp4` (De_Yl9rAYC4) · D2
`output/day2_v38_master_reel.mp4` (De_mCcFAVTO) · D3 `output/day3_barbell_squat_simple2_master.mp4` (DfCpKW4IIV8) ·
D4 `output/day4_romanian_deadlift_4step_master.mp4` (DfC8EifoxFa) · D5
`output/day5_dumbbell_lateral_raise_4step_master_v5.mp4` (DfFPS3YFLF3). Day 6 (DfGabTBjwt8) deleted, t529u.

**Primary base = Day 4 for EVERYTHING visual (t541u–t547u: "Day 4 is the best so far")**. Day 4 = gold #1.
From Day 5 only: the one-pass modular pipeline (+ hashed caches), the auto QC gates, the native-fabric shorts
masking and the spoken experience-question caption. From Day 1 only: the bright X-Ray bones. Nothing else from Day 1–3/5.
Measured from `output/day4_romanian_deadlift_4step_master.mp4` (script logic in `gymform/tempo.py`, values in `gymform_spec.json`).

| Element | Base (source day) | Measured / source | Shared module now |
|---|---|---|---|
| **Skin tone** | **Day 4** cool charcoal-silver: body L p50 64 / p75 75 / p90 88 / p99.9 ~143, tint RGB −3/+1/+3 (t542u, 6 frames, body-only ROI, bar/plates excluded) | Day 4 master | `spec.mannequin.skin_lock_all_steps`; `render.skin_tone_lock`, `muscle.silver_skin`; QC (f)(i)(j)(k) scale with it |
| **Backdrop / colour grade** | **Day 4** backdrop median RGB 35/39/43 (plate graded per channel on load) | Day 4 master | `bg.studio_bg`; QC (a) uses the same grade |
| **Shorts** | **Day 4** tone L 24–40 (just under the backdrop) on **Day 5** native-fabric masking | Day 4 master; mask 3b584ad | `muscle.silver_skin`, `mask.native_shorts`; QC (f)(n)(k) |
| **Camera angle / framing** | **Day 4**: ONE camera side for all 4 steps, full body, no zoom, same mannequin size | aeeb806 | config `motions.*.mirror` (whole-frame flip of native clips), `ctx.normalize_scale`; zoom 1.0 |
| **Muscle fibre** | **Day 4** RDL fibre grain/colour as reference (current module = Day 5 port of it; next single change if it differs on the sheet) | D4 f63a353 | `gymform/muscle.py` |
| **Title / preview** | **Day 4** depth-behind Kaushan title + 4-slot translucent preview, see-through 0.40 | 70e292c, 328c7b4 | `gymform/layout.py`; QC (b1)(b2)(h) |
| **Angle callout** | **Day 4** (t547u): one straight joint axis (bent only at the joint vertex) + 2 cross ticks, vertex ring(s), horizontal precision leader to a left text block level with the target; number 86 px Pretendard ExtraBold at x 86, label 21 px in the role colour, sub-line 16 px | `build_day4_rdl_master.draw_day4_laser_and_callout` | `layout.draw_callout_day4` (`spec.ui.callout_style` = `day4` or `day4_refined` = Day 5 crisp 2.8 px laser + 84 px number) |
| **Tempo** | **Day 4** (t544u): per 5 s step `0.5−0.5·cos(2π·1.5·u)` = rest → top 1.67 s → rest 3.33 s → top 5.0 s, smooth turnarounds, no flat holds (measured: turnarounds at frames 50/100 of all 4 steps) | `build_day4_rdl_master.py` L284; master | `gymform/tempo.py`; QC **(v)** |
| **X-Ray bones** (Day 1) | **Day 1 v36** bright near-white luminous bones, only a soft tint; a native bone clip of the other colour is colour-graded (`native_tint`), never repainted | Day 1 v36 @1.5 s | `gymform/xray.py`; QC (e)(g)(u) |
| **Pipeline / QC** (Day 5) | one-pass modular `run_day.py`, hashed per-step caches, preflight gate, auto QC | ca6e9f1, 66590f9, 3b584ad | `run_day.py`, `gymform/*` |
| **Caption** (Day 5/6) | 3 English lines, 0 hashtags + spoken experience question at the end | 1eba222, t515u | `caption.py`, `tts.py`; QC (l) |
| **Step order** | 1 wrong bone, 2 right bone, 3 wrong muscle, 4 right muscle | t515u | `ctx.order_lint` |

Superseded by Day 4 (kept in spec as `*_legacy_day5`): Day 5 bright silver lock 165/191/226, shorts 10–22, tempo 0.3/1.5/1.3/1.5/0.4, single arm-laser callout.

## Pair matching (t521u relaxed by t541u: same motion + angle within joint tolerance)
Zero-cost route (t539u, no xAI, no generation): pair each native muscle clip with the native bone clip whose
motion matches best. DTW-aligned keypoints (`_tmp_day6/pair/dtw.py`), mean joint distance in % of skeletal size;
tolerance `spec.step_order.pair_tolerance` = joint ≤ 15 %, elbow/shoulder angle ≤ 15°, view ≤ 15 %
(same-clip noise floor 1.5–2.8 %). Lint (`ctx.pair_lint`) checks the stored `pair_match`; QC (s) gates keypoints at
the same tolerance and reports silhouette IoU (two different native clips never share pixels).
Day 6 result: wrong = step3 bone f23–59 ↔ step2 muscle: 5.6 % / 11.6° / 8.8° / view 6.8 % (good);
right = step1 bone f20–47 ↔ step4 muscle: 14.6 % / 14.5° / 12.9° / view 7.1 % (at the limit, eyeball).
The old 4-clip version had the two bone clips on swapped steps (each matched the other form's muscle clip).
The paid keyframe-edit route (≈ $1.8/attempt) stays a fallback only with the owner's spend approval.

## Bone brightness gates (t530u)
- Render QC **(u)**: bone pixels in the X-Ray window L50 ≥ 140, L90 ≥ 190, contrast vs surround ≥ 25,
  dark-saturated share ≤ 0.20. Day 1 v36 scores L50 150–153, L90 200–206, contrast 27–43, dark 0.08–0.14.
- Source acceptance (`xai_edit.ACCEPT`): bone V50 ≥ 190, S50 ≤ 0.50, white core ≥ 0.10, dark fill ≤ 0.12,
  tint hue red ≥168 or ≤12 / green 40–80.

## One change at a time
Day N = baseline + at most ONE new element, listed in the day config `change` field, with a before/after preflight
sheet for the owner. Order of pending improvements: (1) pair matching via keyframe clips, (2) target zoom.
