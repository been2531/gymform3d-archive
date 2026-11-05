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

## v3 additions (2026-11-04): Day 4 layout + black boxer briefs
- **Briefs (드로즈):** a separate garment mesh built from the body's own pelvis/upper-thigh surface (bisected at
  z 0.80 / 1.03 m for a straight waistband and leg hems), welded, offset 12 mm outward, with an inward lip at
  every open edge so no skin shows under the band. It keeps the body's vertex groups + armature modifier, so it
  is skinned exactly like the body (no cloth sim). Matte black Principled (base 0.012, roughness 0.95); white
  emission in the mask pass, so the grade treats it as the Day 4 shorts and never puts green on it.
- **Briefs QC:** `briefs_body_touch_frac` (BVH clearance garment→deformed body < 0.5 mm, every 3rd frame),
  `briefs_no_tearing_frac2x`, `briefs_no_visible_poke_through` (body pixels inside the closed briefs silhouette,
  every frame), `briefs_coverage_every_frame` (pixels ≥ 0.6× median, 0 green on briefs).
- **Layout:** Day 4 callout (laser joint axis: vertical ray → hip → torso, precision leader, Pretendard 84/21/16
  at x=86, y ≥ 600). The number is the rig-measured median torso-from-vertical, not the 2D projection. Day 4
  shows one callout per step, so there is no separate elbow callout. 1-slot preview pill
  (`layout.composite_title_and_preview_v4`) with the peak-frame torso thumbnail.
- **Overlay QC:** `overlay_in_safe_zone` (title/pill/callout text inside qc.safe_zone), `overlay_text_no_overlap`,
  `callout_text_clear_of_figure`, `callout_angle_matches_measured`.

## v4 (2026-11-04): posture QC (neck / head / legs) + retarget root-cause fixes
Owner feedback on 20261104_2240: the neck looked bent, the legs looked like a running stride, the figure was small.
- **Root cause, neck:** `neck` and `Head` were both damped-tracked to the ear midpoint (landmarks 7/8, labelled
  "nose"). The Head bone starts at the top of the neck, almost level with the ears, so its aim was ill-conditioned
  (head folded toward the chest, flipping on some frames). In addition every spine bone tracked the *shoulder
  midpoint*; the top bone (`Spine`) starts at shoulder height, so the chest itself swung 7-65° off the thoracic line.
- **Fix (config `motion`):** `spine_aim: far` (spine bones aim 3 torso lengths out along hip→shoulder),
  `neck_mode: spine_follow` (no landmark aim: the neck inherits the chest + a fixed `gaze_ext_deg` extension, the Head
  keeps its rest relation → 0 roll / 0 yaw by construction).
- **Root cause, legs:** feet were locked to frame 1 of YMove's long split stance (rear heel up) and the knees came
  from MediaPipe (91-134°, thighs split ~69°) and the rig could not reach its ankle targets (7-8 cm foot drift).
- **Fix:** `legs: planted` — both feet flat at the rig's rest foot pitch, fore-aft stagger × `stance_scale`, hips +
  upper body raised rigidly (refined on the evaluated rig) until the knees reach ~`knee_target_deg`, knees solved
  per frame (2-bone, rig bone lengths, bending toward the toes). Legacy behaviour = `neck_mode: ears_track`,
  `legs: source`, `spine_aim: shmid` (defaults, so old configs reproduce).
- **Framing:** `render.camera.top_y` (was a fixed 0.24) 0.22, `feet_y` 0.90, `fill_w` 0.90.
- **Posture QC (`tools/rig3d/posture_qc.py`, generic, limits in `qc.posture` per exercise):** pure geometry from the
  evaluated armature dumped per frame to `render/bones.npz` (world matrices of all bones + rest):
  `neck_in_line_with_spine_deg`, `neck_flex_deg_min_max`, `head_roll_deg`, `head_yaw_deg`, `neck_steady_maxdev_deg`,
  `both_feet_planted`, `foot_drift_m`, `knee_angles_deg`, `knee_steady_maxdev_deg`, `thigh_split_deg`. Per-frame
  series go to `posture_series.json`. Auto-fix rules: `neck_follow_spine`, `plant_legs`.

## v5 (2026-11-05): title clear of figure, soft balanced knees, neutral-grip dumbbell, braced support hand
Each issue got a generic QC check first (limits in the exercise config), confirmed FAIL on 20261104_2343
(`results/rig3d/db_row/20261104_2343/v5_recheck.json`), then a root-cause fix:
- **`overlay_clear_of_figure_px`** (grade3d stores figure-mask px inside the title / pill boxes per frame; limit
  `qc.overlay_figure_px_max: 0`). 2343: 5822 px of the figure under the title. Fix: `render.camera.top_y` 0.27
  (title box ends at y 491 px = 0.256), `feet_y` 0.92.
- **`knees_soft_and_balanced`** (`qc.posture.knee_soft_range_deg`, `knee_lr_diff_max_deg`). Root causes of the
  175° rear leg: the pelvis was tipped diagonally toward the lean (minimal-rotation track, no yaw) → one hip joint
  ~9 cm higher; YMove's "legs apart" stance is ~0.5 m wide; the pelvis followed the rowing torso every frame.
  Fixes (`motion`): `pelvis_yaw`, `stance_width_m: 0.30`, `lock_hips` (pelvis aims at the rep-median torso
  direction), `balance_knees` (each ankle slides fore-aft until its own knee hits `knee_target_deg: 160`);
  `torso_target_deg` 49 for a slightly deeper hinge.
- **`dumbbell_handle_orientation`** (handle vs the across-palm axis = middle PCA axis of the rest hand mesh, and vs
  perpendicular-to-forearm) and **`dumbbell_neutral_grip_deg`** (handle vs the body's horizontal forward axis).
  Root causes: the handle followed the shoulder line, the bone X/Z axes sit ~45° off the palm, and the hand tracked
  the MediaPipe index point (wrist curl up to 77°). Fixes: `grip_mode: palm_neutral` (handle across the palm;
  forearm 60% + hand 40% twisted per frame so the palm faces the torso), `wrist_neutral`.
- **`support_hand_on_thigh_m`** (non-working wrist to nearest thigh bone line). Fix: `support_hand: front_thigh`
  (lowest reachable point on the front thigh, elbow from a 2-bone solve, pointing out to the side).
- `render/bones.npz` now also stores the dumbbell matrices, handle axis, palm axes and working hand.

## v6 (2026-11-05): power grip, self-healing loop, anatomy audit

Owner: "the script should handle it; the hand has to wrap around the dumbbell."

**Power grip (`tools/rig3d/hand_grip.py`).** The rig has no finger bones, so the fist is a geometric shape key
("fist", working hand only). The hand frame comes from the mesh: the web line from the thumb-cluster gap; the
across-palm axis from the four finger centroids; the palm side = the side the relaxed thumb sits on. The handle axis
runs across the palm at the knuckle line, at the closest distance where every non-thumb vertex clears r + 2 mm.
Each finger hug-bends with arc length kept. The thumb flexes about its base, ramped in from the thenar seam, and is
searched for a 2-6 mm gap with the best opposition. Displacement is Gaussian-blended across finger/palm seams, so the
web doesn't tear. Config key: `motion.grip_wrap`.

**Dumbbell axis bug (v1-v5).** The joined dumbbell kept its (0, 90 deg, 0) rotation, so the handle's long axis was
object Z, not X. The v5 handle-orientation check measured X, so it passed while the handle was really 90 deg off
the palm. That is the "stacked heads" look. `render.dumbbell_axis_fix: true` fixes it, and the bones.npz
`handle_axis_local` field records the true axis. Because the handle has no direction, the neutral-grip twist is
now solved with period pi.

**Anatomy audit (`tools/rig3d/anatomy_qc.py`, `qc.anatomy`).** Checks:
- signed elbow/knee flexion (hyperextension), wrist and ankle deviation;
- bone-capsule self-intersection;
- whole-body + dumbbell centre of mass inside the feet polygon;
- L/R bone-length symmetry;
- grip: wrap per finger, fingertip gap, thumb gap + opposition, hand/handle intersection;
- visible mesh spikes (stretch relative to the median edge, edges under the shorts excluded);
- dumbbell heads visible and separated on screen;
- extremities inside the frame;
- figure height not below target, green-area cap.

**Self-healing loop (`pipeline.py --max-iter 3`).** Every QC check maps to a fix rule or to "human" in
`qc3d.RULE_MAP`; unmapped checks are listed in qc.json. The loop runs:
1. A rig pre-check (seconds, checks-only) until the rig and anatomy checks pass or the rules are exhausted.
2. render, grade, QC.
3. Pixel fix rules (camera / glow), then a re-render, up to N times.

`run.json` records `auto_iterations` and each applied override. The owner is told only what still fails.

**v6 support-arm fixes.** In v5 the support elbow pointed laterally and was hyperextended by 10 deg, and the wrist
was bent 58 deg. Now `support_elbow_mode: back_out` and `support_wrist_neutral: true`. COM fix rule
`stance_mode: feet_forward` puts both ankles in front of the hip joints, so the hips sit back over the feet, and the
support hand braces on the same-side thigh.
