# Motion + tempo standards (@gymform.3d)

Why: motion and speed must match what real lifters do (t549u). Every number below links to a free source. Machine-readable copy: `configs/motion_standards.json`. Checker: `gymform/motion_check.py` (see PIPELINE.md).

How to read
- **[S]** = number from the cited source. **[H]** = the source gives the cue but no number, so the value is a pipeline tolerance (safe to tune).
- Angles are 2D image angles from the cached keypoints. The reels use a 3/4 view, so checks carry a ±tolerance (default 10°). Items marked *top/front view* can't be judged in 3/4 view and report n/a.
- Where sources disagree, the range is given and the disagreement is noted.

## Tempo for every exercise (hypertrophy working set)
- Moderate speed: **1–2 s up, 1–2 s down** (ACSM 2009). [ACSM](https://pubmed.ncbi.nlm.nih.gov/19204579/)
- Any rep from **0.5 to 8 s** grows muscle about equally; **>10 s per rep** looks worse. [Schoenfeld 2015](https://pubmed.ncbi.nlm.nih.gov/25601394/) · [ACSM H&FJ 2023](https://journals.lww.com/acsm-healthfitness/fulltext/2023/05000/the_ultimate_guide_for_selecting_repetition_tempos.9.aspx)
- 2025 meta-analysis (14 studies): eccentric 0.25–4.5 s, concentric 0.5–4.5 s; faster (~1 s) vs slower (~3.5 s) reps were about equal. [Enes 2025](https://doi.org/10.1519/jsc.0000000000005302)
- Trained lifters left to choose go a bit faster than a strict 2 s / 2 s. [Nóbrega 2018](https://pubmed.ncbi.nlm.nih.gov/29481458/)
- Research protocols commonly pace **2 s down / 2 s up** (bench, lateral raise). [Rodríguez-Ridao 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/) · [Coratella 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7503819/)
- **Our reels:** Day 4 cosine tempo = 1.67 s up + 1.67 s down, 3.33 s per rep, no hold → inside every range above. Day 5 holds (1.5 / 1.3 hold / 1.5) also fit.

## Cable Rope Pushdown (Day 1)
`cable_rope_pushdown`

- **Targets:** triceps brachii (lateral + medial heads emphasized, long head)
- **Right form**
  - upper arm stays by the side (elbow aligned with trunk midline) through the rep: `arm_torso` max **0–20** [H] — ACE gives no degree value; 20 deg is a pipeline tolerance for 'elbows by sides'. [ace_pushdown](https://www.acefitness.org/resources/everyone/exercise-library/185/triceps-pushdowns/)
  - elbows reach (near) full extension at the bottom without hard lockout: `elbow_flexion` min **0–15** [H] — ACE: 'until your elbows become fully extended (do not lock out)'; 0-15 deg is a pipeline tolerance. [ace_pushdown](https://www.acefitness.org/resources/everyone/exercise-library/185/triceps-pushdowns/)
  - torso aligned vertically, held still: `trunk_from_vertical` range **0–10** [H]. [ace_pushdown](https://www.acefitness.org/resources/everyone/exercise-library/185/triceps-pushdowns/)
- **Common mistakes (what Step 1/3 should show)**
  - elbows drift forward / flare (shoulder flexes, lats + momentum take over): `arm_torso` max > 25 [H]. [ace_pushdown](https://www.acefitness.org/resources/everyone/exercise-library/185/triceps-pushdowns/)
  - leaning the torso over the handle to push with body weight: `trunk_from_vertical` range > 15 [H]. [ace_pushdown](https://www.acefitness.org/resources/everyone/exercise-library/185/triceps-pushdowns/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Wide-Grip Lat Pulldown (Day 2)
`wide_grip_lat_pulldown`

- **Targets:** latissimus dorsi, teres major, lower/middle trapezius, rhomboids, biceps brachii (assist)
- **Right form**
  - slight backward lean, no more than 30 deg: `trunk_from_vertical` max **0–30** [S] — ACE: 'no more than a 30 degree angle'. 2025 EMG study: 30 deg lean did not change lat EMG but raised posterior deltoid. [ace_pulldown](https://www.acefitness.org/resources/everyone/exercise-library/158/seated-lat-pulldown/) · [pulldown2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12452428/)
  - torso angle does not change during the pull (no swing): `trunk_from_vertical` range **0–10** [H] — ACE: 'avoid any additional backwards lean during the pull'; 10 deg is a pipeline tolerance. [ace_pulldown](https://www.acefitness.org/resources/everyone/exercise-library/158/seated-lat-pulldown/)
  - elbows drive down toward the sides; bar to top/mid chest: `arm_torso` min **0–40** [H] — ACE: stop when the elbows stop moving down and start moving back. [ace_pulldown](https://www.acefitness.org/resources/everyone/exercise-library/158/seated-lat-pulldown/)
- **Common mistakes (what Step 1/3 should show)**
  - shrugging (scapulae elevate instead of depress) -> upper traps take over: *not measurable yet* (needs ear/neck keypoints; the cached 13-point skeleton has none. TODO hook: shoulder-to-ear distance). [ace_pulldown](https://www.acefitness.org/resources/everyone/exercise-library/158/seated-lat-pulldown/)
  - leaning back past ~30 deg / swinging the torso to move the weight: `trunk_from_vertical` max > 30 [S]. [ace_pulldown](https://www.acefitness.org/resources/everyone/exercise-library/158/seated-lat-pulldown/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Barbell Back Squat (Day 3)
`barbell_back_squat`

- **Targets:** quadriceps, gluteus maximus, adductor magnus, spinal erectors (isometric)
- **Right form**
  - thighs at least parallel to the floor at the bottom (tops of thighs just below parallel): `thigh_below_horizontal` max **0–20** [S] — CSCCA: femur parallel; SCJ 2020: just below parallel, deeper can cause forward lean / pelvic tilt. Upper bound 20 deg is a pipeline tolerance. [cscca_squat](https://cscca.org/content.aspx?club_id=408309&module_id=731826&page_id=22) · [scj2020_lowbar](https://journals.lww.com/nsca-scj/fulltext/2020/06000/technical_aspects_and_applications_of_the_low_bar.13.aspx)
  - full-depth knee flexion: `knee_flexion` max **115–125** [S] (warning only) — SCJ 2018 recommends 115-125 deg knee flexion; CSCCA only requires parallel, so this is a warning, not a fail. [scj2018_squat](https://journals.lww.com/nsca-scj/fulltext/2018/12000/optimizing_squat_technique_revisited.10.aspx)
  - knees track over the toes, no valgus (frontal view): `knee_ankle_width_ratio` min **0.9–2.0** [H] *(front view only)* — ratio of knee width to ankle width; <0.9 = knees caving inside the feet. Pipeline heuristic; front view only (feet and knees overlap in 3/4 view). [cscca_squat](https://cscca.org/content.aspx?club_id=408309&module_id=731826&page_id=22) · [scj2018_squat](https://journals.lww.com/nsca-scj/fulltext/2018/12000/optimizing_squat_technique_revisited.10.aspx)
- **Common mistakes (what Step 1/3 should show)**
  - knees cave in (valgus): `knee_ankle_width_ratio` min < 0.85 [H] *(front view only)*. [cscca_squat](https://cscca.org/content.aspx?club_id=408309&module_id=731826&page_id=22) · [myer2014_squat](https://pmc.ncbi.nlm.nih.gov/articles/PMC4725067/) · [ffpa_pfp](https://pmc.ncbi.nlm.nih.gov/articles/PMC10324282/)
    - Published 2D FPPA cutoff 8.2 deg is for single-leg squats in women with PFP only, so a ratio heuristic is used here
  - stopping well above parallel: `thigh_below_horizontal` max < -15 [H]. [cscca_squat](https://cscca.org/content.aspx?club_id=408309&module_id=731826&page_id=22)
  - chest drops / excessive forward lean: `trunk_from_vertical` max > 60 [H]. [cscca_squat](https://cscca.org/content.aspx?club_id=408309&module_id=731826&page_id=22) · [scj2020_lowbar](https://journals.lww.com/nsca-scj/fulltext/2020/06000/technical_aspects_and_applications_of_the_low_bar.13.aspx)
    - no source gives a degree cutoff; 60 deg is a pipeline heuristic
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Romanian Deadlift (Day 4)
`romanian_deadlift`

- **Targets:** hamstrings, gluteus maximus, spinal erectors (isometric)
- **Right form**
  - knees stay slightly bent; the movement is a hip hinge: `knee_flexion` max **15–34** [S] — literature: ~15 deg (Piper & Waller via Lee 2018 / NSCA) up to 33.9+/-12.6 deg measured to the floor (Lee 2018). [lee2018_rdl](https://pmc.ncbi.nlm.nih.gov/articles/PMC6323186/) · [nsca_rdl](https://www.nsca.com/education/articles/kinetic-select/romanian-deadlift-rdl/)
  - hip flexion at the bottom: `hip_flexion` max **64–96** [S] (warning only) — Lee 2018 mean 80.0 deg, SD 15.9 (range = mean+/-1 SD); stop where the back would round. [lee2018_rdl](https://pmc.ncbi.nlm.nih.gov/articles/PMC6323186/)
- **Common mistakes (what Step 1/3 should show)**
  - lower back rounds (spine flexes instead of hips): *not measurable yet* (needs a mid-spine keypoint; TODO hook. Proxy: knee bend and hip angle both low.). [nsca_rdl](https://www.nsca.com/education/articles/kinetic-select/romanian-deadlift-rdl/) · [ace_rdl](https://www.acefitness.org/continuing-education/prosource/january-2016/5767/ace-technique-series-romanian-deadlift/)
  - bending the knees a lot (turns into a conventional deadlift / squat): `knee_flexion` max > 50 [H]. [lee2018_rdl](https://pmc.ncbi.nlm.nih.gov/articles/PMC6323186/)
    - conventional deadlift knee flexion 85.2 deg vs RDL 33.9 deg in Lee 2018; 50 deg is a pipeline midpoint heuristic
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Dumbbell Lateral Raise (Day 5)
`dumbbell_lateral_raise`

- **Targets:** lateral (medial) deltoid, anterior deltoid / supraspinatus (assist), upper trapezius (unwanted if shrugging)
- **Right form**
  - arms rise to shoulder level (~90 deg abduction), not above: `arm_torso` max **80–95** [S] — ACE: 'level with your shoulders and approximately parallel with the floor'; Coratella 2020 protocol: abduct up to 90 deg. 80-95 band is a pipeline tolerance around 90. [ace_lateral](https://www.acefitness.org/resources/everyone/exercise-library/26/lateral-raise/) · [coratella2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7503819/)
  - slight, fixed elbow bend: `elbow_flexion` max **0–30** [H]. [ace_lateral](https://www.acefitness.org/resources/everyone/exercise-library/26/lateral-raise/)
  - torso erect, no swing / back arch: `trunk_from_vertical` range **0–10** [H]. [ace_lateral](https://www.acefitness.org/resources/everyone/exercise-library/26/lateral-raise/)
- **Common mistakes (what Step 1/3 should show)**
  - shrugging / hiking the shoulders (upper trap takes over): *not measurable yet* (needs ear keypoints; TODO hook). [ace_lateral](https://www.acefitness.org/resources/everyone/exercise-library/26/lateral-raise/)
  - swinging the torso / using momentum: `trunk_from_vertical` range > 15 [H]. [ace_lateral](https://www.acefitness.org/resources/everyone/exercise-library/26/lateral-raise/)
  - raising well above shoulder height: `arm_torso` max > 110 [H]. [ace_lateral](https://www.acefitness.org/resources/everyone/exercise-library/26/lateral-raise/)
- **Tempo:** down 1–2 s · pause 0–0.5 s · up 1–2 s · rep 2–5 s

## Incline Dumbbell Press (Day 6)
`incline_dumbbell_press`

- **Targets:** pectoralis major (clavicular/upper head), anterior deltoid, triceps brachii
- **Setup bench_incline_deg:** 15–30 [S]. Rodriguez-Ridao 2020: upper pec peaks at 30 deg, >45 deg shifts to anterior deltoid. SBS: low incline 15-30 deg; most gym benches sit at 45 deg (too steep). Sources agree on <=30 deg; the common '30-45 deg' advice is NOT supported for pec emphasis. [rodriguez_ridao2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/) · [sbs_bench](https://www.strongerbyscience.com/how-to-bench/)
- **Right form**
  - upper arm ~45-60 deg from the torso at the bottom: `arm_torso` value **45–60** [S] *(top view only)* — SBS gives 45-60 deg for the flat bench; applied to incline by analogy. Noteboom 2024 found abduction (45/70/90) mattered less than grip width/scapula pose and did not rank a safest angle. [sbs_bench](https://www.strongerbyscience.com/how-to-bench/) · [noteboom2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/)
  - dumbbells lowered to chest level (deep elbow bend): `elbow_flexion` max **80–110** [H] (warning only). [rodriguez_ridao2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/)
- **Common mistakes (what Step 1/3 should show)**
  - elbows flared out to ~90 deg (T-shape): `arm_torso` value > 75 [H] *(top view only)*. [noteboom2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/) · [sbs_bench](https://www.strongerbyscience.com/how-to-bench/)
    - 90 deg was the widest condition tested (Noteboom 2024); 75 deg cutoff is a pipeline heuristic between SBS 60 and 90
  - bench set at 45 deg or more (becomes a shoulder press): `input:bench_incline_deg`  >= 45 [S]. [rodriguez_ridao2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/) · [sbs_bench](https://www.strongerbyscience.com/how-to-bench/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Barbell Bench Press
`barbell_bench_press`

- **Targets:** pectoralis major, anterior deltoid, triceps brachii
- **Setup grip_width_x_shoulder:** 1.5–2.0 [S]. SBS 1.5-2x shoulder width; Noteboom 2024: wider grips (2 BAW) raised glenohumeral and AC joint loads, narrower lowered AC compression. [sbs_bench](https://www.strongerbyscience.com/how-to-bench/) · [noteboom2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/)
- **Right form**
  - upper arm 45-60 deg from torso with the bar on the lower chest: `arm_torso` value **45–60** [S] *(top view only)* — sources disagree on a single safest angle (Noteboom 2024: 45 deg had more superior shear, larger angles more compression). [sbs_bench](https://www.strongerbyscience.com/how-to-bench/) · [noteboom2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/)
- **Common mistakes (what Step 1/3 should show)**
  - elbows flared to ~90 deg: `arm_torso` value > 75 [H] *(top view only)*. [noteboom2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/) · [sbs_bench](https://www.strongerbyscience.com/how-to-bench/)
  - bar touches high on the chest / neck with elbows flared: *not measurable yet* (needs bar position; TODO hook (bar y vs sternum)). [sbs_bench](https://www.strongerbyscience.com/how-to-bench/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Barbell Bent-Over Row
`barbell_bent_over_row`

- **Targets:** latissimus dorsi, middle/lower trapezius, rhomboids, posterior deltoid, biceps (assist), spinal erectors (isometric)
- **Right form**
  - torso hinged 45-90 deg from vertical (almost parallel): `trunk_from_vertical` min **45–90** [S] — ACE: 45-90 deg from vertical; ACSM H&FJ 2017 (abstract only, full text paywalled): slightly above horizontal; J-STAGE study: 65 deg forward flexion gave more lat/lower trap EMG than 40 deg. [ace_barbell_row](https://www.acefitness.org/resources/pros/expert-articles/3802/ace-total-body-barbell-workout/) · [acsm_hfj_row2017](https://journals.lww.com/acsm-healthfitness/fulltext/2017/03000/the_barbell_row_exercise.9.aspx) · [jstage_row](https://www.jstage.jst.go.jp/article/jscicoachingtraining/8/1/8_23/_article/-char/en)
  - torso angle held during the pull (no body english): `trunk_from_vertical` range **0–10** [H]. [ace_barbell_row](https://www.acefitness.org/resources/pros/expert-articles/3802/ace-total-body-barbell-workout/)
- **Common mistakes (what Step 1/3 should show)**
  - torso rises / jerks up to move the bar: `trunk_from_vertical` range > 15 [H]. [ace_barbell_row](https://www.acefitness.org/resources/pros/expert-articles/3802/ace-total-body-barbell-workout/)
  - lower back rounds: *not measurable yet* (needs spine keypoints; TODO hook). [acsm_hfj_row2017](https://journals.lww.com/acsm-healthfitness/fulltext/2017/03000/the_barbell_row_exercise.9.aspx)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Seated Cable Row (Day 7)
`seated_cable_row`

- **Targets:** latissimus dorsi, middle trapezius, rhomboids, posterior deltoid, biceps (assist)
- **Right form**
  - torso vertical and motionless: `trunk_from_vertical` max **0–15** [H] — ACE: 'torso vertical', 'avoid leaning back'; no source gives a degree tolerance. [ace_seated_row](https://www.acefitness.org/resources/everyone/exercise-library/168/seated-row/) · [acsm_hfj_seated_row](https://journals.lww.com/acsm-healthfitness/fulltext/2019/07000/do_it_right__the_seated_cable_row_exercise.9.aspx)
  - no rocking: `trunk_from_vertical` range **0–10** [H]. [ace_seated_row](https://www.acefitness.org/resources/everyone/exercise-library/168/seated-row/)
  - elbows pull back past the sides of the body: `elbow_flexion` max **80–140** [H] (warning only). [ace_seated_row](https://www.acefitness.org/resources/everyone/exercise-library/168/seated-row/)
- **Common mistakes (what Step 1/3 should show)**
  - leaning back / rocking the torso to pull: `trunk_from_vertical` range > 15 [H]. [ace_seated_row](https://www.acefitness.org/resources/everyone/exercise-library/168/seated-row/) · [acsm_hfj_seated_row](https://journals.lww.com/acsm-healthfitness/fulltext/2019/07000/do_it_right__the_seated_cable_row_exercise.9.aspx)
  - rounding the shoulders/back forward at the stretch: *not measurable yet* (needs spine keypoints; TODO hook). [ace_seated_row](https://www.acefitness.org/resources/everyone/exercise-library/168/seated-row/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Bulgarian Split Squat (Day 8)
`bulgarian_split_squat`

- **Targets:** quadriceps (front leg), gluteus maximus, adductors
- **Right form**
  - front knee flexion at the bottom (back knee near the floor): `knee_flexion` max **100–110** [S] — bss2025 upright variant: back knee to floor = ~100-110 deg front knee flexion. Per-leg metric: the more-bent leg is used (agg=max_side). [bss2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12382192/)
- **Common mistakes (what Step 1/3 should show)**
  - front knee caves inward: *not measurable yet* (per-leg frontal plane check TODO). [scj2017_rfess](https://journals.lww.com/nsca-scj/fulltext/2017/12000/technique,_variation,_and_progression_of_the.10.aspx)
  - barely bending (front knee well under 90 deg): `knee_flexion` max < 80 [H]. [bss2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12382192/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s
- **Note:** Trunk angle is a choice, not an error: upright = quad emphasis; ~40 deg forward lean raised glute max EMG (bss2025).

## Standing Barbell Overhead Press
`barbell_overhead_press`

- **Targets:** anterior + lateral deltoid, triceps brachii, upper trapezius / serratus anterior
- **Right form**
  - full elbow extension, arms in line with the ears at the top: `arm_torso` max **160–180** [H] — ISSA: bar, biceps and triceps align with the ears at full extension; degree band is a pipeline heuristic. [issa_ohp](https://www.issaonline.com/blogs/safety-and-injuries/overhead-press-proper-form-variations-and-common-mistakes) · [scj2017_ohp](https://journals.lww.com/nsca-scj/fulltext/2017/12000/exploring_the_standing_barbell_overhead_press.7.aspx)
  - no lower-back arch / leaning back: `trunk_from_vertical` max **0–10** [H]. [issa_ohp](https://www.issaonline.com/blogs/safety-and-injuries/overhead-press-proper-form-variations-and-common-mistakes)
- **Common mistakes (what Step 1/3 should show)**
  - arching the lower back / leaning back: `trunk_from_vertical` max > 15 [H]. [issa_ohp](https://www.issaonline.com/blogs/safety-and-injuries/overhead-press-proper-form-variations-and-common-mistakes) · [scj2017_ohp](https://journals.lww.com/nsca-scj/fulltext/2017/12000/exploring_the_standing_barbell_overhead_press.7.aspx)
  - elbows flare out instead of pointing forward under the bar: *not measurable yet* (needs a front view; TODO hook). [issa_ohp](https://www.issaonline.com/blogs/safety-and-injuries/overhead-press-proper-form-variations-and-common-mistakes)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Barbell Hip Thrust
`barbell_hip_thrust`

- **Targets:** gluteus maximus, hamstrings (assist)
- **Right form**
  - shins vertical at lockout: `shin_from_vertical` value **0–10** [H] — Contreras: shins vertical at lockout, slight angle OK; 10 deg is a pipeline tolerance. [contreras_ht](https://bretcontreras.com/how-to-hip-thrust/) · [contreras_scoop](https://bretcontreras.com/hip-thrust-hinge-vs-scoop-method/)
  - hips reach full extension (neutral to ~10-15 deg past): `hip_flexion` min **-15–5** [S] — Contreras 2022: most people extend ~10-15 deg past neutral with bent legs; end range = peak glute EMG (Contreras 2015). [contreras_scoop](https://bretcontreras.com/hip-thrust-hinge-vs-scoop-method/)
- **Common mistakes (what Step 1/3 should show)**
  - finishing with the lower back arched instead of the hips: *not measurable yet* (needs spine/pelvis keypoints; TODO hook). [contreras_scoop](https://bretcontreras.com/hip-thrust-hinge-vs-scoop-method/)
  - hips stop well short of full extension: `hip_flexion` min > 20 [H]. [contreras_scoop](https://bretcontreras.com/hip-thrust-hinge-vs-scoop-method/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Leg Press
`leg_press`

- **Targets:** quadriceps, gluteus maximus, adductors
- **Right form**
  - knees bend to at least ~90 deg at the bottom: `knee_flexion` max **90–154** [S] (warning only) — ACE sets up at ~90 deg; 2025 study: ~100 deg vs ~154 deg ROM gave similar quad growth, so any depth with a flat sacrum is fine. [ace_legpress](https://www.acefitness.org/resources/everyone/exercise-library/154/seated-leg-press/) · [legpress_rom2025](https://www.tandfonline.com/doi/full/10.1080/02640414.2025.2481534)
  - stop short of locking the knees: `knee_flexion` min **3–20** [H] (warning only). [ace_legpress](https://www.acefitness.org/resources/everyone/exercise-library/154/seated-leg-press/)
- **Common mistakes (what Step 1/3 should show)**
  - hips/sacrum peel off the pad at the bottom (lumbar rounding): *not measurable yet* (needs pelvis vs pad; TODO hook). [ace_legpress](https://www.acefitness.org/resources/everyone/exercise-library/154/seated-leg-press/)
  - snapping the knees into lockout: `knee_flexion` min < 2 [H]. [ace_legpress](https://www.acefitness.org/resources/everyone/exercise-library/154/seated-leg-press/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Dumbbell Biceps Curl
`dumbbell_biceps_curl`

- **Targets:** biceps brachii, brachialis, brachioradialis
- **Right form**
  - elbows stay close to the sides (upper arm still): `arm_torso` max **0–20** [H]. [ace_curl](https://www.acefitness.org/resources/everyone/blog/5241/4-upper-body-movements-for-beginners/)
  - no swinging / leaning: `trunk_from_vertical` range **0–10** [H]. [ace_curl](https://www.acefitness.org/resources/everyone/blog/5241/4-upper-body-movements-for-beginners/)
  - curl through a full elbow range: `elbow_flexion` range **100–150** [H] (warning only). [ace_curl](https://www.acefitness.org/resources/everyone/blog/5241/4-upper-body-movements-for-beginners/)
- **Common mistakes (what Step 1/3 should show)**
  - swinging the torso / hips for momentum: `trunk_from_vertical` range > 15 [H]. [ace_curl](https://www.acefitness.org/resources/everyone/blog/5241/4-upper-body-movements-for-beginners/)
  - elbows drift forward (front delt helps): `arm_torso` max > 30 [H]. [ace_curl](https://www.acefitness.org/resources/everyone/blog/5241/4-upper-body-movements-for-beginners/)
- **Tempo:** down 1–2 s · pause 0–1 s · up 1–2 s · rep 2–5 s

## Could not source (left as TODO / heuristic)
- No free source gives a degree number for: pushdown elbow drift, curl elbow drift, torso sway limits, OHP lockout band, leg-press lockout, row/pulldown 'no swing' range. These are [H].
- Knee valgus: the only published 2D cutoff (8.2° FPPA) is for single-leg squats in women with knee pain, so the squat uses a knee/ankle width heuristic, front view only.
- Shrugging, lumbar rounding, hip-thrust overarch, pelvis lift on leg press, bar height on bench: need ear/spine/pelvis/bar keypoints the 13-point skeleton doesn't have (TODO hooks in motion_check).
- Bench/incline elbow angle (45–60°) comes from a coaching source (Stronger by Science) for the flat bench; the 2024 shoulder-load study found no single safest abduction angle.
- Incline angle: the common '30–45°' advice is NOT backed for chest emphasis; EMG + coaching sources both point to ≤30° (45° shifts work to the front delts).
- The ACSM H&F Journal lat-pulldown / seated-row / barbell-row articles were paywalled or timed out; ACE pages were used for their numbers.
- RDL: a 2025 landmine RDL study abstract gave no numbers, so RDL hip angle uses only Lee 2018 (80 ± 16°).

## Sources
- `schoenfeld2015`: [Schoenfeld, Ogborn, Krieger 2015, Sports Med: repetition duration and hypertrophy meta-analysis (0.5-8 s similar; >10 s likely inferior)](https://pubmed.ncbi.nlm.nih.gov/25601394/)
- `enes2025`: [Enes ... Schoenfeld 2025, JSCR 39(12): How Slow Should You Go? tempo meta-analysis (ECC 0.25-4.5 s, CON 0.5-4.5 s; faster vs slower similar)](https://doi.org/10.1519/jsc.0000000000005302)
- `acsm2009`: [ACSM Position Stand 2009: Progression models in resistance training (moderate velocity 1-2 s CON, 1-2 s ECC; hypertrophy 6-12 RM at moderate velocity)](https://pubmed.ncbi.nlm.nih.gov/19204579/)
- `nobrega2018`: [Nobrega et al. 2018, JSCR: self-selected vs fixed 2 s/2 s repetition duration in trained men (fixed 2:2 had longer TUT per rep than self-selected)](https://pubmed.ncbi.nlm.nih.gov/29481458/)
- `acsm_hfj_tempo2023`: [ACSM's Health & Fitness Journal 2023: The Ultimate Guide for Selecting Repetition Tempos (hypertrophy 0.5-8 s per rep)](https://journals.lww.com/acsm-healthfitness/fulltext/2023/05000/the_ultimate_guide_for_selecting_repetition_tempos.9.aspx)
- `rodriguez_ridao2020`: [Rodriguez-Ridao et al. 2020, IJERPH: five bench inclinations EMG (upper pec max at 30 deg; >45 deg shifts to anterior deltoid; protocol 2 s ECC / 2 s CON)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/)
- `sbs_bench`: [Stronger by Science, How to Bench Press (upper arm-torso angle ~45-60 deg; low incline 15-30 deg; grip 1.5-2x shoulder width)](https://www.strongerbyscience.com/how-to-bench/)
- `noteboom2024`: [Noteboom et al. 2024, Front Physiol: bench press technique variations and shoulder loads (abduction 45/70/90 deg, grip 1-2 BAW)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/)
- `cscca_squat`: [CSCCA Back Squat technique (femur parallel depth; common errors knee valgus, heels rising, chest dropping)](https://cscca.org/content.aspx?club_id=408309&module_id=731826&page_id=22)
- `scj2018_squat`: [Strength & Conditioning Journal 2018: Optimizing Squat Technique Revisited (full depth 115-125 deg knee flexion, knees over toes, no valgus)](https://journals.lww.com/nsca-scj/fulltext/2018/12000/optimizing_squat_technique_revisited.10.aspx)
- `scj2020_lowbar`: [Strength & Conditioning Journal 2020: Technical Aspects of the Low-Bar Back Squat (tops of thighs just below parallel; excess depth -> forward lean / posterior pelvic tilt)](https://journals.lww.com/nsca-scj/fulltext/2020/06000/technical_aspects_and_applications_of_the_low_bar.13.aspx)
- `myer2014_squat`: [Myer et al. 2014, The Back Squat Part 2 (correcting knee valgus etc.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4725067/)
- `lee2018_rdl`: [Lee et al. 2018, J Exerc Sci Fit: EMG and kinetic comparison of conventional and Romanian deadlifts (RDL peak hip flexion 80.0+/-15.9 deg, knee 33.9+/-12.6 deg; RDL knee 15-33 deg)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6323186/)
- `nsca_rdl`: [NSCA Kinetic Select: Romanian Deadlift (RDL)](https://www.nsca.com/education/articles/kinetic-select/romanian-deadlift-rdl/)
- `ace_rdl`: [ACE ProSource: Romanian Deadlift technique guide](https://www.acefitness.org/continuing-education/prosource/january-2016/5767/ace-technique-series-romanian-deadlift/)
- `ace_pushdown`: [ACE Exercise Library: Triceps Pushdowns (elbows by sides, torso vertical, extend fully without locking, pause momentarily)](https://www.acefitness.org/resources/everyone/exercise-library/185/triceps-pushdowns/)
- `ace_pulldown`: [ACE Exercise Library: Seated Lat Pulldown (lean back no more than 30 deg, no additional lean during pull, bar to top/mid chest, scapulae depressed)](https://www.acefitness.org/resources/everyone/exercise-library/158/seated-lat-pulldown/)
- `signorile2002`: [Signorile et al. 2002, JSCR: EMG of hand positions in the lat pull-down (wide grip anterior highest lat activity)](https://doi.org/10.1519/00124278-200211000-00008)
- `pulldown2025`: [2025 J Funct Morphol Kinesiol: lat pulldown grip variations incl. 30 deg trunk inclination (no lat EMG difference; more posterior deltoid)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12452428/)
- `ace_lateral`: [ACE Exercise Library: Lateral Raise (slight elbow bend, raise until arms level with shoulders ~parallel to floor, no low-back arch / shrug)](https://www.acefitness.org/resources/everyone/exercise-library/26/lateral-raise/)
- `coratella2020`: [Coratella et al. 2020, IJERPH: EMG of lateral raise variations in bodybuilders (abduct to 90 deg; protocol 2 s CON / 2 s ECC / 0.5 s isometric)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7503819/)
- `ace_barbell_row`: [ACE Total-body Barbell Workout: bent-over row (torso about 45-90 deg from vertical / almost parallel; hold contraction for a second)](https://www.acefitness.org/resources/pros/expert-articles/3802/ace-total-body-barbell-workout/)
- `acsm_hfj_row2017`: [ACSM's Health & Fitness Journal 2017: The Barbell Row Exercise](https://journals.lww.com/acsm-healthfitness/fulltext/2017/03000/the_barbell_row_exercise.9.aspx)
- `jstage_row`: [J-STAGE (Japan J Coaching/Training): trunk flexion angle 40 vs 65 deg in bent-over row EMG](https://www.jstage.jst.go.jp/article/jscicoachingtraining/8/1/8_23/_article/-char/en)
- `issa_ohp`: [ISSA: Overhead Press proper form and common mistakes (bar aligned with ears at lockout, no lower-back arch, straight bar path)](https://www.issaonline.com/blogs/safety-and-injuries/overhead-press-proper-form-variations-and-common-mistakes)
- `scj2017_ohp`: [Strength & Conditioning Journal 2017: Exploring the Standing Barbell Overhead Press](https://journals.lww.com/nsca-scj/fulltext/2017/12000/exploring_the_standing_barbell_overhead_press.7.aspx)
- `contreras_ht`: [Bret Contreras: How to Hip Thrust (shins vertical at lockout)](https://bretcontreras.com/how-to-hip-thrust/)
- `contreras_scoop`: [Bret Contreras 2022: Hip Thrust Hinge vs Scoop (hip hyperextension ~10-15 deg past neutral with bent legs; avoid spinal overarch)](https://bretcontreras.com/hip-thrust-hinge-vs-scoop-method/)
- `contreras2015_emg`: [Contreras et al. 2015, J Appl Biomech: glute max EMG back squat vs barbell hip thrust](https://pubmed.ncbi.nlm.nih.gov/26214739/)
- `ace_legpress`: [ACE Exercise Library: Seated Leg Press (knees ~90 deg at setup, sacrum flat on backrest, no low-back movement, don't lock knees)](https://www.acefitness.org/resources/everyone/exercise-library/154/seated-leg-press/)
- `legpress_rom2025`: [2025 J Sports Sci: knee flexion ROM (~5-100 vs ~5-154 deg) does not change quad hypertrophy in leg press](https://www.tandfonline.com/doi/full/10.1080/02640414.2025.2481534)
- `ace_curl`: [ACE: 4 Upper-body Movements for Beginners - Biceps Curl (elbows close to sides, straight body line)](https://www.acefitness.org/resources/everyone/blog/5241/4-upper-body-movements-for-beginners/)
- `ace_seated_row`: [ACE Exercise Library: Seated Row (torso vertical, avoid leaning back / arching, elbows past sides, pause momentarily)](https://www.acefitness.org/resources/everyone/exercise-library/168/seated-row/)
- `acsm_hfj_seated_row`: [ACSM's Health & Fitness Journal 2019: Do It Right: The Seated Cable Row Exercise](https://journals.lww.com/acsm-healthfitness/fulltext/2019/07000/do_it_right__the_seated_cable_row_exercise.9.aspx)
- `bss2025`: [2025 EMG study: Bulgarian split squat trunk position (upright: back knee to floor ~100-110 deg front knee flexion; forward trunk lean raises glute max)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12382192/)
- `scj2017_rfess`: [Strength & Conditioning Journal 2017: Technique, Variation, and Progression of the Rear-Foot-Elevated Split Squat](https://journals.lww.com/nsca-scj/fulltext/2017/12000/technique,_variation,_and_progression_of_the.10.aspx)
- `ffpa_pfp`: [2023 IJSPT: frontal plane projection angle cutoff 8.2 deg (single-leg squat, women with patellofemoral pain; NOT a universal threshold)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10324282/)
