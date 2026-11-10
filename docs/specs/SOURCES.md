# Free exercise reference sources

Catalogue of **zero-cost** textbook stills, motion/pose data, and pose-extraction libraries for per-exercise spec sheets (`docs/specs/`, `exercises/*.spec.yaml`).

Rules:

- Free licences only. No paid packs (OpenGym3D and similar), no Adobe/Mixamo logins, no paid APIs or credit-based generators.
- Do **not** commit third-party images or motion files. Link them, with licence and credit. Local copies live in `_scratch/refs/<exercise>/` (gitignored).
- Demicstory / Muscle & Motion footage is a **quality bar**, not a source. It is not free to copy or to retarget.
- Motion on this project is authored in Blender from keyposes. Mocap is only a study or overlay aid, and only when the licence allows Instagram use.

Verified on 10 Nov 2026 (Asia/Seoul). 404s were dropped or replaced with a working mirror.

---

## Licence cheat-sheet (Instagram and commercial)

Our Instagram posts are original 3D renders. That is usually fine even when the pose was guided by a CC BY-SA still, because the still itself is not being posted. Still, keep the rules below.

| Licence | Use as an internal pose reference | Post the source image on Instagram | Drive production motion for a commercial/social reel |
|---|---|---|---|
| Public domain / CC0 / U.S. government work | Yes | Yes. Credit is good practice, not required. | Yes, if the file is motion. |
| CC BY (2.0 / 2.5 / 3.0 / 4.0) | Yes | Yes, if the caption or description gives credit. | Yes, with credit. |
| CC BY-SA (3.0 / 4.0) | Yes | Avoid posting the source image. Instagram terms make share-alike hard to honour. | Do not adapt and post the image itself. An independently authored 3D render is OK; credit the reference in the spec. |
| CC BY-NC, CC BY-NC-SA, “non-commercial research / education / artistic” | Study only, if at all | No | **No.** Do not retarget these into Instagram production. |
| Unsplash / Pexels | Yes (none needed below) | Yes, under their licences | n/a |

**Safe to post (the still itself):** public domain, CC0, and CC BY with credit.

**Internal only (stills):** Everkinetic diagrams and most Commons photos here are CC BY-SA. Use them to pose the mannequin; do not upload those files to Instagram.

**Do not use for production motion:** Fit3D / AIFit, AMASS and anything derived from it (including HumanML3D motion), KIT (treat as research), Motion-X (NC-SA plus mixed subsets), Bandai Namco Motiondataset (CC BY-NC), FLAG3D, and the SMPL / SMPL-X body models.

---

## 1. Textbook form images

Primary library: [Everkinetic drawings](https://github.com/everkinetic/data) (upstream [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) and the same drawings on [Wikimedia Commons: Weight training diagrams](https://commons.wikimedia.org/wiki/Category:Weight_training_diagrams) (usually marked **CC BY-SA 3.0**). Each exercise has a start and a peak (or bottom) view, almost always from the working side or a slight 3/4.

Attribution, Commons uploads:

> Everkinetic, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), via Wikimedia Commons

Attribution, files taken from the GitHub repo:

> Drawings by Everkinetic, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), https://github.com/everkinetic/data

U.S. Army and Navy photographs are public-domain U.S. government works. CDC illustrations are also public domain.

Back views are scarce. Only the back squat has a clearly licensed back-view photo. One-arm row, RDL, and most upper-body lifts are side or 3/4 only.

### Recommended pick per exercise

#### One-arm dumbbell bench row (`db_row`)

- **Pick:** [File:DumbbellBentOverRow.JPG](https://commons.wikimedia.org/wiki/File:DumbbellBentOverRow.JPG) — side-view diptych (hang and peak), bench-supported one-arm row. **CC BY-SA 3.0**. Author: GeorgeStepanek (model: Emily).
- **Credit:** GeorgeStepanek / Wikimedia Commons / CC BY-SA 3.0 (model: Emily).
- **Also:** already wired in `exercises/db_row.spec.yaml`. Local copy: `_scratch/refs/db_row/source.jpg` (do not commit).
- **Gap:** no free back-view still found. Everkinetic has no one-arm dumbbell row drawing (their “rows” are upright, rear-delt, T-bar, or barbell variants).
- **Motion pick:** none that is both commercial-OK and textbook. Author the motion from this still. Do not use Fit3D even though it covers gym rows.

#### Bent-over barbell row

- **Pick (motion / form):** [File:Bent-over row - exercise demonstration video.webm](https://commons.wikimedia.org/wiki/File:Bent-over_row_-_exercise_demonstration_video.webm) — **CC BY 3.0**, FitnessScape. Overhand (pronated) textbook row. Extract stills locally if needed; do not commit the video.
- **Credit:** FitnessScape, CC BY 3.0, via Wikimedia Commons.
- **Also (diagrams):** [Reverse grips bent over barbell rows 1.svg](https://commons.wikimedia.org/wiki/File:Reverse_grips_bent_over_barbell_rows_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Reverse_grips_bent_over_barbell_rows_2.svg) — Everkinetic, **CC BY-SA 3.0**. Useful for torso angle and bar path, but the grip is supinated. Everkinetic lists a pronated “Bent Over Row with Barbell” (id 0298) but ships no drawing for it.
- **Motion pick:** the FitnessScape clip above (CC BY 3.0, Instagram-OK with credit). No CMU clip is labelled as a barbell row.

#### Dumbbell lateral raise

- **Pick (diagrams):** [Lateral dumbbell raises 1.svg](https://commons.wikimedia.org/wiki/File:Lateral_dumbbell_raises_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Lateral_dumbbell_raises_2.svg) — Everkinetic, **CC BY-SA 3.0**. Front/3/4, arms at the sides then at shoulder height.
- **Also (photo diptych):** [File:DumbbellLateralRaise.JPG](https://commons.wikimedia.org/wiki/File:DumbbellLateralRaise.JPG) — GeorgeStepanek, **CC BY-SA 3.0**.
- **Motion pick:** none commercial-OK and textbook. Author from the stills. CMU subject 86 has general arm raises in mixed fitness takes, not a clean lateral-raise set.

#### Incline dumbbell press

- **Pick:** [Incline dumbbell press 1.svg](https://commons.wikimedia.org/wiki/File:Incline_dumbbell_press_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Incline_dumbbell_press_2.svg) — Everkinetic, **CC BY-SA 3.0**. Start (dumbbells at the shoulders) and lockout.
- **Also:** [Dumbbell incline bench press 1.svg](https://commons.wikimedia.org/wiki/File:Dumbbell_incline_bench_press_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Dumbbell_incline_bench_press_2.svg) (same family); [Incline press - exercise demonstration video.webm](https://commons.wikimedia.org/wiki/File:Incline_press_-_exercise_demonstration_video.webm) — FitnessScape, **CC BY 3.0** (confirm it is dumbbell, not barbell, before using a still).
- **Motion pick:** FitnessScape clip if the implement matches; otherwise author from the SVGs.

#### Barbell bench press

- **Pick (diagrams):** [Bench press 1.svg](https://commons.wikimedia.org/wiki/File:Bench_press_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Bench_press_2.svg) — Everkinetic, **CC BY-SA 3.0**. Clear side view, bar over mid-chest, feet on the floor.
- **Also (postable still):** [File:Bench press.webp](https://commons.wikimedia.org/wiki/File:Bench_press.webp) — Wikideas1, **CC0**.
- **Also (public domain photo):** [FOB Lightning bench press…](https://commons.wikimedia.org/wiki/File:FOB_Lightning_bench_press_compition_121123-A-NS855-002.jpg) — Spc. Alex Amen, U.S. Army, public domain. Competition photo, not a studio textbook angle.
- **Motion pick:** no free commercial-OK textbook mocap. Author from the SVGs.

#### Back squat

- **Pick (diagrams):** Everkinetic barbell squat start/bottom, **CC BY-SA 4.0**, from the official repo: [`0122-relaxation.svg`](https://raw.githubusercontent.com/everkinetic/data/main/dist/svg/0122-relaxation.svg) and [`0122-tension.svg`](https://raw.githubusercontent.com/everkinetic/data/main/dist/svg/0122-tension.svg). Commons also has [Squats-1.png](https://commons.wikimedia.org/wiki/File:Squats-1.png) / [Squats-2.png](https://commons.wikimedia.org/wiki/File:Squats-2.png) (**CC BY-SA 3.0**). Do not use “Narrow stance squat with barbell” as the standard back-squat spec (Commons “Barbell squat 1.svg” redirects there).
- **Pick (back view):** [Woman doing squat workout in gym with barbell, back view.jpg](https://commons.wikimedia.org/wiki/File:Woman_doing_squat_workout_in_gym_with_barbell,_back_view.jpg) — Nenad Stojkovic, **CC BY 2.0**. Instagram-postable with credit.
- **Credit (back view):** Nenad Stojkovic, CC BY 2.0, via Wikimedia Commons.
- **Also:** [Squat - exercise demonstration video.webm](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) and [Barbell back squat exercise demonstration.webm](https://commons.wikimedia.org/wiki/File:Barbell_back_squat_exercise_demonstration.webm) (check each file’s licence before use; the second is CC BY-SA 4.0, Your Move).
- **Motion pick:** CMU subjects 13, 14 and 86 include squat-like motions (often unweighted). Commercial-OK with requested credit, but not a loaded high-bar back squat. Prefer authored keyposes from the diagrams plus the back-view photo.

#### Romanian deadlift

- **Pick:** [Romanian dead lift 1.svg](https://commons.wikimedia.org/wiki/File:Romanian_dead_lift_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Romanian_dead_lift_2.svg) — Everkinetic, **CC BY-SA 3.0**. Side view, soft knees, hips back, bar close to the legs.
- **Also:** [Romanian-deadlift-1.png](https://commons.wikimedia.org/wiki/File:Romanian-deadlift-1.png) / [2.png](https://commons.wikimedia.org/wiki/File:Romanian-deadlift-2.png), same source.
- **Gap:** no free back-view still found.
- **Motion pick:** none commercial-OK and textbook. Author from the SVGs. Do not confuse with a conventional deadlift clip.

#### Deadlift

- **Pick (diagrams):** [Barbell dead lifts 1.svg](https://commons.wikimedia.org/wiki/File:Barbell_dead_lifts_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Barbell_dead_lifts_2.svg) — Everkinetic, **CC BY-SA 3.0**.
- **Pick (public domain photo):** [U.S. Army recruiter demonstrates correct way to properly lift deadlift…](https://commons.wikimedia.org/wiki/File:U.S._Army_recruiter_demonstrates_correct_way_to_properly_lift_deadlift,_Gaithersburg,_Maryland_(2020),_1.jpg) — U.S. Army, public domain. Instagram-postable.
- **Also (side-photo sequence):** [Deadlift-phase 1.JPG](https://commons.wikimedia.org/wiki/File:Deadlift-phase_1.JPG) / [2](https://commons.wikimedia.org/wiki/File:Deadlift-phase_2.JPG) / [3](https://commons.wikimedia.org/wiki/File:Deadlift-phase_3.JPG) — stu_spivack, **CC BY-SA 2.0**. And [Deadlift illustration.jpg](https://commons.wikimedia.org/wiki/File:Deadlift_illustration.jpg) — Luis Javier Rodriguez / Yupi666, **CC BY 2.5**.
- **Motion pick:** CMU trial 79_93 is labelled “weight lifting” (commercial-OK). Treat as a study clip, not a conventional-deadlift oracle. Author the production take from the SVGs and the Army / phase photos.

#### Dumbbell curl

- **Pick (diagrams):** [Alternating bicep curl with dumbbell 1.svg](https://commons.wikimedia.org/wiki/File:Alternating_bicep_curl_with_dumbbell_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Alternating_bicep_curl_with_dumbbell_2.svg) — Everkinetic, **CC BY-SA 3.0**. Front view, start and peak.
- **Also (photo, postable with credit):** [Dumbbell bicep curls.jpg](https://commons.wikimedia.org/wiki/File:Dumbbell_bicep_curls.jpg) — PTPioneer, **CC BY 2.0**.
- **Motion pick:** none commercial-OK and isolated. Author from the SVGs.

#### Overhead press

- **Pick (standing photos):** [How to do an Overhead Press.jpg](https://commons.wikimedia.org/wiki/File:How_to_do_an_Overhead_Press.jpg) — RangerJim, **CC BY-SA 4.0** (internal still). And [US Navy … military press.jpg](https://commons.wikimedia.org/wiki/File:US_Navy_030926-N-4936C-041_Sgt._Hernan_O._Gamboa_does_a_military_press.jpg) — U.S. Navy photo by Journalist David P. Coleman, public domain (Instagram-postable).
- **Also (diagrams):** [Barbell shoulder press 1.svg](https://commons.wikimedia.org/wiki/File:Barbell_shoulder_press_1.svg) / [2.svg](https://commons.wikimedia.org/wiki/File:Barbell_shoulder_press_2.svg) — Everkinetic, **CC BY-SA 3.0**. Confirm seated vs standing before using them as the spec oracle (the related Everkinetic text for id 0011 is a seated press). [Shoulder press - exercise demonstration video.webm](https://commons.wikimedia.org/wiki/File:Shoulder_press_-_exercise_demonstration_video.webm) — FitnessScape, **CC BY 3.0**.
- **Motion pick:** FitnessScape clip (CC BY 3.0, with credit) or authored keyposes from the standing photos.

---

## 2. Free motion and pose data

Do **not** download these datasets onto the box. They are large. Record the licence and coverage here, then fetch a single clip later only if the licence is Instagram-safe.

| Source | Licence / posting | What it actually covers for us | Format | Verdict |
|---|---|---|---|---|
| [CMU Graphics Lab MoCap](http://mocap.cs.cmu.edu/) (site was returning errors on 10 Nov 2026; use mirrors) | Free for research **and commercial** use worldwide. Do not resell the data, even converted. Requested credit below. | Fitness is thin. Trial **79_93** “weight lifting”; squat-like takes on subjects **13, 14, 86** (86_02, 86_08, 13_29, 13_30, 14_06). No labelled bench press, barbell row, RDL, curl, lateral raise, or OHP. | Original ASF/AMC; BVH via mirrors | **Only clearly commercial-OK mocap.** Use for weight and easing study, not as a textbook oracle. Official site was down at verification; mirrors loaded. |
| [cgspeed CMU BVH conversion](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture) and [motion list](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database/bvh-conversion-release-motions-list) | Same rights as CMU (Bruce Hahne added no extra restrictions). | Same clips, MotionBuilder-friendly BVH with a T-pose on frame one. | BVH | Preferred download route if we ever take a CMU clip. |
| [una-dinosauria/cmu-mocap](https://github.com/una-dinosauria/cmu-mocap) | Same as CMU. | Same library, easy HTTP access to BVH. | BVH | Working mirror (verified). |
| [AMASS](https://amass.is.tue.mpg.de/license.html) | MPI **non-commercial** scientific / education / artistic licence. Commercial use needs a separate deal (`ps-license@tue.mpg.de`). | Unified SMPL/SMPL-X of many mocap sets, including CMU. | SMPL / SMPL-H | **Do not use** for Instagram production. If we want CMU, take the original/cgspeed files, not the AMASS retarget. |
| [HumanML3D](https://github.com/EricGuo5513/HumanML3D) | Code is MIT. Motion is rebuilt from AMASS + HumanAct12, so the **motion inherits AMASS terms**. | Text-labelled clips (daily, sport, dance). Not a gym-form library. | Derived SMPL-style | **Do not use** the motion for production. The text labels are not a licence upgrade. |
| [KIT Whole-Body Human Motion Database](https://motion-database.humanoids.kit.edu/faq/) / KIT-ML | Registration; intended for scientific research. Cite the KIT papers. Do not treat as commercial. | Whole-body mocap + some object interaction. Not a textbook lifting set. | C3D / KIT format; KIT-ML is the language pairing | **Research only** unless KIT confirms otherwise in writing. |
| [Fit3D / AIFit](https://fit3d.imar.ro/) — [terms](https://fit3d.imar.ro/legal) | Free for **non-commercial** scientific research, education, or artistic projects. Commercial use, commercial training, and redistribution are prohibited. Ask `licenses@imar.ro` for a commercial licence. | Best coverage of our list: 47 gym exercises, instructor vs trainee, Vicon + RGB, GHUM / SMPL-X. Includes squat, dumbbell biceps, compound lifts, and similar. | MoCap skeletons, SMPL-X, multi-view video | The dataset we would *want*, and **must not** use for Instagram production. Do not download it. |
| [Motion-X](https://motion-x-dataset.github.io/static/license/Motion-X%20License.pdf) | Annotations **CC BY-NC-SA 4.0**. Subsets keep their original licences (AMASS/GRAB = MPI NC; AIST++ / EgoBody / BAUM = CC BY 4.0; HuMMan = S-Lab; IDEA400 = CC BY-SA 4.0). | Huge mixed motion corpus. Not gym-form specific. | SMPL-X / text | **No** for production (NC-SA umbrella). Do not take AMASS subsets out of it either. |
| [Bandai Namco Research Motiondataset](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset) | **CC BY-NC 4.0** | Everyday locomotion and acted motions, not textbook barbell work. | BVH / FBX style releases | **No** for Instagram production. |
| [MM-Fit](https://zenodo.org/records/7607736) | **CC BY 4.0** (commercial OK with attribution). | Workout sessions as video, depth, and wearable IMU. Not a skeletal mocap / BVH library. | Video + IMU | Useful as a **licensed video** form reference if we later extract poses. Do not download the full set (large). Credit the paper/dataset. |
| [FLAG3D](https://andytang15.github.io/FLAG3D/) | Research only; signed academic licence. Commercial use and redistribution prohibited. | Fitness activities with language instructions, 3D. | Dataset (academic access) | **No.** Do not request or download. |

CMU requested credit, when a clip is actually used:

> The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

**Practical motion plan:** author keyposes from the stills in section 1. Optionally study a CMU squat or “weight lifting” take for easing. Optionally extract 2D joints from a CC BY FitnessScape or MM-Fit video for overlay QC. Leave Fit3D, AMASS, and NC datasets alone.

---

## 3. Open-source libraries (pose, retarget, IK)

| Tool | Licence | Role for us | Notes |
|---|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) (BlazePose) | Apache 2.0 | 2D (and light 3D) joints from a **licensed** still or video, for overlay QC | Code is commercially usable. This project already rejected MediaPipe as the *motion authoring* method. Keep it for overlay only. |
| [MMPose](https://github.com/open-mmlab/mmpose) / RTMPose | Apache 2.0 | Stronger 2D pose, if MediaPipe is not enough | Commercially usable. |
| [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | Apache 2.0 | Alternative 2D pose | Commercially usable. |
| [4D-Humans / HMR2.0](https://github.com/shubham-goel/4D-Humans) | Code MIT | Mesh recovery from video | The **SMPL / SMPL-X weights** are MPI non-commercial. Do not bake those meshes into Instagram assets. |
| [WHAM](https://github.com/yohanshin/WHAM) | Code MIT | World-space human motion from video | Same SMPL-weight caveat as above. |
| [SMPL-X tooling](https://github.com/vchoutas/smplx) / [project](https://smpl-x.is.tue.mpg.de/) | Code often + MPI model licence (non-commercial research; registration) | Body model used by many academic pipelines | **Do not** use the SMPL-X model in production renders. Grab3D stays the mannequin. |
| [Rokoko Studio Live for Blender](https://github.com/Rokoko/rokoko-studio-live-blender) | LGPL-3.0 | Free retargeting addon (no paid Rokoko account required for retarget) | Official support page blocked automated checks (HTTP 403); the GitHub repo loaded. Auto-Rig Pro is paid and is excluded. |
| [ReNim](https://blenderartists.org/t/renim-node-based-retarget-animation/1261958) | Free Blender addon (see its repo licence before shipping) | Node-based retarget onto Rigify or a custom rig | Community option if Rokoko mapping fights the Grab3D rig. |
| [Rigify](https://docs.blender.org/manual/en/4.2/addons/rigging/rigify/index.html) | GPL, bundled with Blender 4.2 | IK/FK, foot roll, hand IK | Good contact-locking toolset. We are on Blender 4.2.3; this 4.2 manual URL was verified. The `/latest/addons/rigging/rigify/index.html` path 404s. |
| Blender built-in BVH import + bone constraints | GPL (Blender) | Import a CMU BVH, then visual-key and clean contacts | Prefer this over a paid retargeter. |

---

## 4. Community notes

Searched X, Reddit (r/blender, r/3Dmodeling, r/animation, r/blenderhelp, r/MachineLearning), and BlenderArtists. Links below loaded on 10 Nov 2026 unless noted. X had almost no production write-ups of Demicstory or Muscle & Motion; most hits were retweets of Demicstory’s own form clips (copyrighted, quality bar only).

### Fitness mocap and how people use it

- [patch, 16 Oct 2026](https://x.com/patch_gg/status/2110896376746070448) (also [this restatement](https://x.com/SupersocksIntel/status/2110984661472247825)) — Pull a skeleton from video with MediaPipe, then retarget onto a Blender rig. **Verdict:** actionable as a *fallback for licensed video*, not as our motion authoring path (MediaPipe motion was already rejected).
- [CuteGuy on ExMR, 5 Nov 2026](https://x.com/cutetoxicguy/status/2118410521519296736) — Open-source video-to-3D animation that includes fingers and feet, no suit. **Verdict:** worth a later look for overlay; licence of the code/models must be checked before any production use. Not a gym-form library.
- [r/MachineLearning: MediaPipe push-up tracker](https://www.reddit.com/r/MachineLearning/comments/122pi1m/realtime_push_up_tracking_using_python_and/) — Joint angles from a live camera compared to a reference. **Verdict:** same idea as our overlay QC. Actionable for checks, not for authoring.
- [r/3Dmodeling: mocap retargeting rant](https://www.reddit.com/r/3Dmodeling/comments/vtrf88/rant_motion_capture_is_a_nightmare/) — Bone names, rest poses, and hierarchies are not standard; retargeting is most of the work. **Verdict:** supports keeping motion authored on one Grab3D rig instead of mixing mocap packs.

### How anatomy / muscle-highlight fitness reels are made

- [r/blender: “Identify the tool used for gym muscle animation”](https://www.reddit.com/r/blender/comments/1jlddar/identify_the_tool_used_for_gym_muscle_animation/) — A Muscle & Motion–style clip is described as ordinary armature animation plus shapekeys for contraction, not a special app. **Verdict:** actionable. We already highlight with projection-painted glow on a smooth clay mannequin; no need to buy a medical-animation suite.
- [r/blender: easiest way to animate workout videos](https://www.reddit.com/r/blender/comments/1de48wz/easiest_way_to_make_3d_animation_for_body_workout/) — Mixamo is suggested (login, excluded here). The better comment: learn to pose from reference; a squat can be authored by hand. **Verdict:** actionable, and it matches our keypose method.
- [r/animation: light up muscles on an anatomical doll](https://www.reddit.com/r/animation/comments/180jcqm/rigger_generic_humanoid_animator_help_plz/) — People point to medical-animation specialists, or to a generic rigged body whose materials light up. **Verdict:** useful confirmation, not a pipeline.
- [r/blender: design question, mocap + muscle transparency](https://www.reddit.com/r/blender/comments/1qbv09l/question_for_design/) — Same product idea: a body where muscles can be hidden or highlighted on top of captured motion. **Verdict:** conceptual only.
- [r/blender: muscle sim test](https://www.reddit.com/r/blender/comments/lj2ulz/muscle_sim_test/) — Soft-body muscles look like jello. Real muscle tenses, shortens, and bulges; origins and insertions matter; tendons do not contract. **Verdict:** actionable. Do not simulate jiggly muscle meshes. Keep clay + fibre/groove shading + a glow on the working muscle.
- [Demicstory, 3 Nov 2026: Single Arm Dumbbell Row mistakes](https://x.com/demicstory/status/2117691418462273956) — Their own form reel (clay mannequin, per-muscle colour, short mistake list). **Verdict:** style and cue reference only. Do not download, retarget, or post their frames.

### Believable lifting animation in Blender

- [BlenderArtists: Sit to Stand Exercise](https://blenderartists.org/t/sit-to-stand-exercise/632215) — Block start, mid, and end; let the hips lead; offset joint timing by a frame or two; ease into stops; avoid perfect left/right symmetry; shoulders and head counterbalance the hips. **Verdict:** highly actionable for every lift we author.
- [BlenderArtists: Lock hands to an object](https://blenderartists.org/t/armature-animation-lock-hands-to-object/556674) — Parent or constrain hand IK targets to an empty on the bar; key constraint influence on and off with the grip. **Verdict:** actionable. Use this so the dumbbell or barbell does not float or slide in the palm.
- [r/blenderhelp: toe locked, heel free](https://www.reddit.com/r/blenderhelp/comments/1uuju97/how_do_i_keep_the_toe_position_locked_so_the_heel/) and [IK foot locking](https://www.reddit.com/r/blenderhelp/comments/1c3bju9/ik_foot_locking/) — Plant the foot IK (or Rigify foot roll) on the floor and animate the pelvis down for a squat or hinge. **Verdict:** actionable for squat, deadlift, and RDL contact.
- [BlenderArtists: fix sliding feet](https://blenderartists.org/t/fix-sliding-feet-in-blender/1620939) and [Mocap Bone Lock](https://blenderartists.org/t/addon-mocap-bone-lock-auto-foot-sliding-fix-for-mocap-and-general/1630357) — Detect contact frames and lock the bone; the linked addon is paid, but the method (constrain, then bake visual keys) is not. **Verdict:** use the method, not the paid addon.
- [BlenderArtists: ReNim retarget](https://blenderartists.org/t/renim-node-based-retarget-animation/1261958) — Free node-based retarget that does not assume identical rest poses. **Verdict:** useful if we ever bring in a CMU BVH.
- [r/blender: discovering Rokoko retargeting](https://www.reddit.com/r/blender/comments/1ns4bir/discovering_rokoko_retargeting_and_motion_capture/) — The free Rokoko addon can retarget onto Rigify, but DEF vs control bones still need cleanup. **Verdict:** actionable, with the usual “clean the contacts by hand” caveat.
- [BlenderArtists: Amy the Weight Lifter](https://blenderartists.org/t/amy-the-weight-lifter/372688) — Early community weightlifting animation thread. **Verdict:** historical only; the contact advice in the newer threads is clearer.

### Top five takeaways

1. Highlight contracted muscle (emission, opacity, or a shapekey bulge). Do not soft-body-simulate muscle; it reads as fat or jello.
2. Author lifts from textbook stills and blocked keyposes. Cheap mocap and login packs (Mixamo) are not the quality path, and most “fitness mocap” is not commercially licensed.
3. Weight reads from hip-led blocking, offset joint timing, and eases into the stop — not from every bone arriving on the same frame.
4. Lock contacts on purpose: feet planted with IK, hands constrained to the bar or dumbbell for the grip, then visual-key and release.
5. Demic-like reels are a smooth mannequin, a fixed camera, and per-muscle colour — ordinary Blender work. There is no public free “Muscle & Motion in a box,” and Demicstory footage itself is not a source.

---

## Local copies

Small, licensed stills only (about 2.9 MB plus the existing `db_row` photo). Gitignored, never committed.

```
_scratch/refs/db_row/                 # already present; DumbbellBentOverRow.JPG
_scratch/refs/bb_bent_over_row/       # Everkinetic reverse-grip SVGs
_scratch/refs/db_lateral_raise/       # Everkinetic SVGs + GeorgeStepanek photo
_scratch/refs/db_incline_press/       # Everkinetic SVGs
_scratch/refs/bb_bench_press/         # Everkinetic SVGs + CC0 webp
_scratch/refs/bb_back_squat/          # Everkinetic PNG/SVG + CC BY back-view thumb
_scratch/refs/bb_rdl/                 # Everkinetic SVGs
_scratch/refs/bb_deadlift/            # Everkinetic SVGs, illustration, PD Army photo, phase thumbs
_scratch/refs/db_curl/                # Everkinetic SVGs + CC BY photo thumb
_scratch/refs/bb_overhead_press/      # Everkinetic SVGs, RangerJim photo, PD Navy photo
```

Each folder has `SOURCES.txt` and `SOURCES.json` with page URL, licence, and credit. Large originals were stored as 1280-px thumbs where the file was bigger than about 1 MB. FitnessScape videos and all mocap datasets were **not** downloaded.
