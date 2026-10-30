# GOLD registry (t541u–t547u)

Approved / known-good results. **Machine-readable list: `gold.json`** (ranked). `run_day.py` loads it via
`gymform/gold.py` and prints + logs it at the start of every run (missing local files are flagged), so every new
day starts from these references. Media files are local (not in git); configs/docs/code are.

| # | Item | Path | Why it is gold |
|---|---|---|---|
| **1** | **Day 4 RDL master** (posted DfC8EifoxFa) | `output/day4_romanian_deadlift_4step_master.mp4`, cover `…_cover.jpg`, builder `build_day4_rdl_master.py` (aeeb806) | Owner t541u–t547u: best so far. **Primary base for everything visual**: skin tone (L p50 64 / p75 75 / p90 88 / p99.9 143, tint −3/+1/+3), backdrop grade RGB 35/39/43, shorts L 24–40, one camera side for all steps, full-body framing, muscle-fibre look, Kaushan title + translucent preview, angle callout (one joint axis + cross ticks + horizontal leader + 86 px number), tempo 1.5 cosine reps / 5 s step (turnarounds 1.67 / 3.33 s). All measured values live in `gymform_spec.json`. |
| 2 | Day 1 v36 X-Ray bones (posted De_Yl9rAYC4) | `output/day1_v36_master_reel.mp4` @ 1.5 s; QC ref `output/qc/ref/day1_v36_xray_t1.5.jpg` | Bone BRIGHTNESS only: near-white luminous, soft tint (t530u), QC (g)/(u). Scope is now t552u: only the target bones, no glass window, skin untouched. |
| 3 | Day 5 v5 master (posted DfFPS3YFLF3) | `output/day5_dumbbell_lateral_raise_4step_master_v5.mp4`, `configs/day05.json` (3b584ad) | **Mechanics only**: one-pass modular pipeline, auto QC gates, native-fabric shorts masking, spoken experience-question caption. Its silver skin, tempo and arm-laser callout are superseded by Day 4. |
| 4–5 | Day 5 native clips | `output/assets/day5_lat_step1_bone_720p.mp4`, `…step3_bone_v3…`, `…step2_muscle…`, `…step4_muscle…` | Clip-generation template: native i2v, anatomy baked in, bones over OPAQUE skin. |
| 6 | Day 6 wrong pair (candidate) | `day6_idp_step3_bone_720p.mp4` f23–59 ↔ `day6_idp_step2_muscle_720p.mp4` f2–62 | Joint 5.6 % / elbow 11.6° / shoulder 8.8° / view 6.8 % (noise 2.8 %). Promote after the owner approves the sample sheet. |
| 7 | Day 6 right pair (candidate, weaker) | `day6_idp_step1_bone_720p.mp4` f20–47 ↔ `day6_idp_step4_muscle_720p.mp4` f2–58 (both mirrored to the Day 4 camera side) | Joint 14.6 % / 14.5° / 12.9° / view 7.1 % — inside the 15 tolerance but at the limit. |

**Never use**: `output/assets/day6_idp_{wrong,right}_bone_720p.mp4` (xAI edits, rejected t529u).

Adding an entry: only after the owner approves it (posted or explicitly OK'd). Give the path, the source commit,
and why. Remove or demote entries the owner rejects.
