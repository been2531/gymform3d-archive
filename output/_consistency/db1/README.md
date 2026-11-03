# db1: dumbbell pilot (t696u): Dumbbell Bent-Over Row start stills
Plan A+B (docs/mission/FUNDAMENTAL_PLAN.md). 3 Grok Bot GenerateImage, 0 video.
- `master_mannequin_gen1.jpg` is the fixed master (Day 4 écorché body + palette A cobalt hex dumbbells). Reuse it as `reference_image_paths[0]` for every future dumbbell/bodyweight still.
- `db_row_wrong_start_gen1.jpg`: torso too upright + shrug, red upper traps.
- `db_row_right_start_gen1.jpg`: hip hinge, green lats/rhomboids/mid trap.
- Checks: `python3 tools/free_weight_precheck.py configs/precheck/day08_db_bent_over_row.json --still IMG --form wrong|right --out J`.
  - Shared thresholds: `configs/precheck/free_weight_base.json`.
  - A new exercise only needs a small JSON with `extends`, `forms` and `dumbbells_expected` (0 for bodyweight).
  - Sharpness ref = `ref_day4_step4_f1s.png` (Laplacian var 704.7).
- Results:
  - wrong: all 8 PASS (by eye: biceps not red, WARN).
  - right: torso FAIL (33.2° vs ≥35°). By eye the hinge is shallow and the contrast with the wrong still is only 11°. Everything else passes.
- Negative fixtures (in _scratch, not committed): far dumbbell erased → dumbbell FAIL; Gaussian σ2.5 → sharp FAIL.
- Sheet: `DB1_ROW_STILLS_sheet.jpg` (spec `sheet_spec.json`, script `tools/free_weight_sheet.py`).
