# Reusable asset plan: acquire once, reuse forever (t700u) — 2026-11-04 KST
0 generations. Probe = 2 tiny CPU renders (360×640, 16 spp, 1.0 s each, nice 19).

## Recommendation
- **Body:** Grab3D "Human Muscles" (CC0, rigged). It is downloaded and verified on the box.
- **Equipment:** a procedural hex dumbbell built in bpy (palette A cobalt + gold rim, our own code).
- **Motion:** MediaPipe Pose (Apache-2.0) world landmarks from a 10 s phone clip that 중현 님 (or anyone he authorises) films, in side view.
  - Retarget by bone direction, not hand-keyed IK. This targets the R1–R6 failure.
  - The dumbbell is parented to the hand bone; feet are locked.
- **Render:** Cycles CPU on the box at nice 19, or on a Cursor/Cloud agent for long renders.
  - Per-muscle colour masks come from vertex groups painted once, rendered as a pass.
- **Look:** the existing Day 4 grade (native / day4_fibre) on top.
- **Cost:** per new exercise, 0 generations; a phone clip + one script run.
- **Status:** not proven yet. The first proof is a bent-over row (below).

## Candidates (verified sources)
- **Grab3D Human Muscles, rigged:** https://grab3d.com/models/anatomy/human-muscles/
  - License: `LICENSE.txt` = CC0 1.0, commercial OK, no attribution.
  - Disclosure: AI-assisted (Meshy image-to-3D, cleaned in Blender), so IG's AI label may apply.
  - File check on the box (`_scratch/assets_probe/human-muscles-rigged.glb`, 13 MB):
    - mesh `char1` 168,649 verts;
    - armature with 24 bones (Hips, 3 spine, neck/head, shoulders, arms, forearms, hands, legs, feet, toes); no finger bones;
    - actions walk/run.
  - The probe render shows the sculpted superficial muscles (traps, lats, glutes, calves) in charcoal matte: `REUSABLE_ASSET_probe_grab3d_rest.png`.
  - Quality vs Day 4: the forms are close; fibre striation is softer at 360p. Needs a 720p check, and the Day 4 grade adds the fibres.
  - Per-exercise cost: 0. Setup: about 1 day (fist shape key, shorts mesh, mask groups).
  - Risks:
    - no fingers, so the grip needs a one-time fist shape key;
    - auto-rig skinning at the shoulders and hips under a deep hinge.
- **Z-Anatomy:** https://github.com/Z-Anatomy/Models-of-human-anatomy
  - License: CC BY-SA 4.0, from BodyParts3D CC-BY-SA 2.1 JP. Commercial OK with attribution; ShareAlike applies to derivative models, not to our videos.
  - Gives individual anatomical muscles (exact per-muscle masks).
  - Not rigged: it is an atlas. Binding it to a rig is days of work. The authors say the models "need quite a bit of specialized work" ([issue](https://github.com/LluisV/Z-Anatomy/issues/5)).
  - Use: optional later, as a mask/reference source projected onto the Grab3D body.
- **Free écorché scans:**
  - The CC0 Jagiellonian écorché by Marcin Solarz: https://mars0ne.artstation.com/projects/KOgBBr
  - The Sketchfab écorchés: [1](https://sketchfab.com/3d-models/anatomical-figure-ecorche-0b4bf087f6064180895917e4a93f527b), [2](https://sketchfab.com/3d-models/male-full-body-ecorche-ab11ebff89224f03bd75efede1164cf6)
  - Great sculpt, but static, not rigged, and some carry a historic pose. Rigging = days.
- **SMPL / SMPL-X, AMASS, HumanML3D-trained text-to-motion (MDM, MoMask, MotionGPT), WHAM / 4DHumans / GVHMR, Fit3D:**
  - All are non-commercial or SMPL-bound.
  - Sources: [SMPL license](https://smpl.is.tue.mpg.de/modellicense.html), [AMASS](https://amass.is.tue.mpg.de/license.html), [MoMask](https://github.com/ericguo5513/momask-codes), [GVHMR](https://github.com/zju3dv/GVHMR/blob/main/LICENSE), [WHAM issue](https://github.com/yohanshin/WHAM/issues/91), [Fit3D terms](https://fit3d.imar.ro/legal).
  - Fit3D has 47 gym exercises, including dumbbells, but forbids commercial use. **Rejected for a monetised IG account.**
- **CMU mocap:** https://mocap.cs.cmu.edu/
  - License: free incl. commercial products; cannot resell the raw data. BVH via cgspeed.
  - Few gym exercises, so it is a fallback for bodyweight/general moves only.
- **Mixamo:** free, commercial use OK, no redistribution of raw files ([Adobe](https://community.adobe.com/questions-696/characters-and-animation-for-commercial-use-589723)).
  - Has a few exercise clips (bicep curl, air squat, push-up).
  - Needs an Adobe login in the browser (owner sign-in), is not scriptable, and its form quality is not coach-grade.
  - Optional.
- **OpenGym3D pack:** https://sylvassiamah.gumroad.com/l/opengym3d-exercise-pack
  - MIT, 24 looping exercises incl. Bent-Over Row, Bicep Curl, Lateral Raise, RDL, Lunge, Push-Up, with a muscle manifest.
  - It is **$19** (a purchase needs owner approval). The figure is MPFB2 (smooth); the animations were authored from pose definitions, so quality is unknown.
  - Its motions could be retargeted to the Grab3D body. A cheap shortcut if he approves.
- **FreeMoCap:** https://github.com/freemocap/freemocap
  - AGPL-3.0, Blender export. Single-camera output is "generally not usable"; it wants 2+ calibrated cameras.
  - Our own MediaPipe pose code (already on the box, `universal_anatomy_rig.get_pose_detector`) is enough for planar side-view moves.
- **Reuse our generated master mannequin as texture:**
  - Possible as a 2D look reference for the grade only.
  - Projecting it onto 3D is unreliable from one view. Not recommended.

## Why this can work where R1–R6 failed
- R1–R3: smooth MakeHuman body + box machine. Fix: a sculpted CC0 muscle body + a simple rigid dumbbell, with no machine.
- R5/R6: hand-written IK keyframes gave floating/crossed arms. Fix: the motion comes from a real human clip.
  - Each bone is aimed along the measured limb vector; the dumbbell is a child of the hand; feet are pinned.
  - Code gates on every frame: no arm crossing (wrist x-order vs shoulder x-order), the hand–dumbbell distance = 0, the elbow angle curve matches the clip ±10°.
- R-series QC was metric-only. Fix: an eye check vs a Day 4 frame on a sheet before any full render (standing rule).

## Pipeline (step order)
1. **One-time:**
   - import the Grab3D rigged GLB;
   - fist shape key + black shorts (a simple cloth-free mesh shell);
   - paint muscle vertex groups once: lats, rhomboids/mid trap, upper trap, rear delt, biceps, glutes, hamstrings, quads, pecs, front delt, triceps, abs, calves;
   - studio camera rig + flat lights matched to the Day 4 backdrop (35,39,43);
   - the procedural dumbbell.
   - Save as `assets3d/master_scene.blend`, with sources + licenses in `assets3d/LICENSES.md`.
2. **Per exercise:**
   - Inputs: a 10 s side-view phone clip (3 reps, correct form) and, if needed, a second clip with the wrong form.
   - `tools/clip2rig.py`: MediaPipe world landmarks → smoothed bone directions → baked action (one rep, loop). It writes the elbow, torso and knee angle curves to QC.
3. **Render:** `tools/render_rep.py` renders beauty + muscle-mask passes, 720×1280, about 60 frames.
   - The probe scales to about 4–8 s/frame on CPU (estimate from 1.0 s at 360p/16 spp), so about 5–10 min per clip on the box at nice 19.
   - Longer or 1080p renders go to a cloud agent.
4. **Grade:** `run_day.py` with the Day 4 grade. The glow regions = the rendered mask pass (exact, no per-frame cutout of a generated body). Then TTS, layout, caption and QC as today.
5. **Review:** sheet → owner approval → publish (unchanged rules).

## Minimal first proof: Dumbbell Bent-Over Row (no generation)
- **Input:** a 10 s side-view phone clip of 3 correct rows with light dumbbells. That needs 중현 님 or someone he picks to film it.
  - Fallback with no filming: OpenGym3D "Bent-Over Row" ($19, approval).
- **Steps:**
  1. one-time scene setup (step 1, limited to the back + arm mask groups);
  2. `clip2rig` on the clip;
  3. render 1 rep at 360p for the motion check, then 3 stills at 720p (start, mid, peak);
  4. grade one still with Day 4.
- **Pass criteria:**
  - by eye, the body reads as Day 4 sculpted muscle next to a Day 4 frame;
  - no arm crossing on any frame (code + eye);
  - the dumbbell stays in the hand on every frame (distance 0, code);
  - elbows travel past the torso at the peak; torso angle 35–60° and held within ±5° during the rep;
  - no visible mesh tearing at the shoulders or hips;
  - green = lats/rhomboids/mid trap only.
- **Cost:** 0 generations. Box time ≈ 1 day of setup work + about 10 min of render.

## Risks
- Meshy-based mesh: skinning artefacts in a deep hinge, no fingers, AI-content disclosure.
- MediaPipe depth noise for the far arm. Side view and planar exercises keep this small; the code gates catch crossing.
- The look gap to Day 4 may need the grade tuned once. The eye check decides.
- The owner must supply clips: 1 short phone clip per exercise.
