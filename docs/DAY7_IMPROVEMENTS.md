# Day 7 build: improvement points learned from Day 6 (t558u–t573u)

> **Read first:** [`DEFECT_PLAYBOOK.md`](DEFECT_PLAYBOOK.md) (one row per defect: symptom → root cause → defined fix → QC check) and [`ROOTCAUSE_DAY6.md`](ROOTCAUSE_DAY6.md). The Day 6 v4 look path (cut-out over a replaced backdrop) is the root cause of the morph, ghost and flicker defects; Day 7 must not render on it unchanged.

Read this before generating clips or touching configs for Day 7. Day 4 (RDL) stays the gold standard.
Day 6 shipped as `configs/day06_muscle2.json`: muscle-only, 2 steps, 10 s, QC 28/28.

## Clips (fix at the source, before rendering)
1. **One dumbbell type in every step.** Day 6 used hex dumbbells in Step 1 and round-plate dumbbells in Step 2, and that can't be fixed in post. Use the same dumbbell model, size, and colour in every step's prompt and reference frame, and reject a clip that switches.
2. **Full range of motion in both clips.** Day 6's right-form clip never reached a deep bottom, so its turnaround looked shallow. Wrong and right clips both need full ROM, with the same depth and rep length. Check the keypoint min/max of the working joint before accepting a clip.
3. **Day 4-sized glow on the target muscle.** Day 6 lit only about 10–17k px (Step 1 15–17k, Step 2 10–12k). Day 4 lit 21–35k px on the right form and 50–62k px on the wrong form. The Day 6 right-form pec glow was also faint (median about 40 vs about 100 on Day 4), which needed a 2.5× glow gain. In the prompt, ask for a **large, bright, saturated glow covering the whole target muscle** (whole upper pec / whole front delt), not a thin patch. Reject clips under about 20k glow px at the peak.
4. **Glow on the muscle the callout names.** In the Day 6 wrong-form clip, much of the red sits on the triceps/forearm instead of just the front delt. The prompt should name the exact muscle, and the clip should be checked against the callout text.
5. **True front and side cameras.** Day 6 had only one usable angle (elevated front-3/4), so it couldn't follow the fixed camera plan (steps 1·2 front, 3·4 side). For a 4-step day, generate or choose real front *and* real side clips of the **same motion** before you build. For a 2-step day, use one best angle for both steps, the same for wrong and right.

## Rendering (shared modules, already in code)
6. **Always use the Day 4 fibre shader** (`gymform/muscle.day4_fibre`, `muscle_shader: "day4"`, the default from Day 6 on). **Never use the flat fill** (`muscle_fill: native_full`, the t567 change). It made flat, over-saturated patches, red speckles on the neck/jaw, and hard blotches instead of fibres. If a clip's glow is faint, the glow-gain calibration (`day4_native_glow_p50`) scales it before Day 4's cut-offs. It doesn't invent a new look.
7. **Don't apply the skin lock to faint muscle-edge colour.** It made the edges muddy (t572 root cause).
8. **Keep the t567 fixes on:** `static_body` (no pale feet or dark head rim from motionless body parts), `clean_plate` (no dumbbell ghosts in the background), `callout_axis.arm_axis` (laser on shoulder→elbow, not shoulder→abdomen), `target_zoom.focus_mode: body_bbox_x` (centre the whole figure), and small-component filtering in the static layer (no stray blobs).
9. **Day 4 skin tone** (p75 ≈ 75, p90 ≈ 88–90) and the Day 4 cosine tempo (turnarounds at frames 50/100 per step) stay locked.

## Format
10. **Muscle-only, 2 steps, is fine when bones aren't needed** (a muscle mistake, not a joint one): `n_steps: 2`, `bone_steps: []`, `muscle_steps: [1, 2]`, 10 s, and a top preview with 2 thumbnails. Use bones (4 steps) only for joint-type mistakes, and then show only the main bones involved without changing the mannequin colour.
11. **The Blender studio3d mannequin is rejected (t569u–t570u). Don't use it.** Use the generated-clip anatomical mannequin (Day 4 style). (`studio3d/` and its BodyParts3D assets are unused, so no BodyParts3D credit is needed.)
12. A variant day gets its own config (`configs/dayNN_<variant>.json`) with its own `kp_cache`, `out_mp4`, `out_cover`, `out_sheet`, and `out_caption`, so the base day's files are never overwritten.

## Process
13. Lint → preflight sheet on sample frames → show 중현 님 → exactly one render → full QC. Fix root causes in shared modules and pin older days (`configs/day05.json` is pinned to `gymform_v5`) so Day 4/5 output stays bit-identical.
14. Before the preflight, compare the muscle colouring against Day 4 with a side-by-side crop (`tools/preflight_sheet.py --compare`). If it doesn't look like Day 4, stop and fix it before rendering.
15. Caption: 3 lines (title / experience question / save line), English only, 0 hashtags, no DM-reward CTA. The TTS ends with the spoken experience question.
16. No paid xAI API and no `XAI_API_KEY`. No grok.com generation (shared quota). Run under `nice` with ≤2 workers. No publishing without 중현 님's approval.

## 17. Flat Day 4 look is mandatory (t591u/t592u, Day 6 v4)
- The backdrop is a flat 35/39/43 (`flat_look: true` + `studio_bg: output/assets/studio_flat_day4_1080p.png`). No vignetted plate and no floor contact shadow. Day 4 did this with a constant (38,40,44).
- The skin lock under `flat_look` remaps ALL skin (no lum>46 gate), p5 -> 52, so the clip's dark shaded side is lifted to one even matte tone.
- QC gates (bg) (backdrop block span <= 4 lum) and (sh) (skin < 46 <= 6%, p5 >= 44) run for day >= 6. v3 measured span 44.5, dark 13-16%, p5 43 (FAIL). v4 measures span 0.0, dark 0.4-0.9%, p5 50 (PASS).
