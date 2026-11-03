# 01 — Existing system diagnosis (evidence)

Commands run on the box 2026-11-03 (KST): `which blender` (none), `/workspace/.bpyenv/bin/python -c "import bpy"` → **5.2.3 LTS**,
addons_core lists `rigify`, `io_anim_bvh`, `io_scene_fbx`; `ffmpeg 7.1.5`; `cv2 5.0.0`; `mediapipe 1.1.1`; node v24.22.1;
`nproc`=8, RAM 15 GB, no GPU (`nvidia-smi` missing); `python3 studio3d/fetch_assets.py` → all 10 locked assets `OK`;
`pytest` not installed (tests/ exists, 8 files, not run).

## Capability status
| Capability | Status | Evidence |
|---|---|---|
| Blender env | IMPLEMENTED_AND_VERIFIED (headless bpy only) | bpy 5.2.3 in `/workspace/.bpyenv`; no `blender` binary/GUI; CPU-only Cycles (STUDIO3D.md: ≈3.5 min/frame at 1080×1920, 24 spp, measured on 1 CPU) |
| .blend projects | NOT_IMPLEMENTED | Only Blender's bundled startup/brush .blend files; scene is built in code each run (`studio3d/build_scene.py`) |
| Character mesh | IMPLEMENTED_BUT_UNVERIFIED (rejected look) | MakeHuman CC0 `assets3d/base.obj` + macro morphs; owner rejected vs Day 4 (proof `output/day6_3d_proof_vs_day4_t562.jpg`: smooth plastic body, no muscle form) |
| Armature / rigging | PARTIALLY_IMPLEMENTED | MakeHuman `default.mhskel` + `default_weights.mhw`; posing via `aim()` bone aiming in `build_scene.py` |
| Rigify | NOT_IMPLEMENTED | Addon present in bpy, unused |
| IK/FK | NOT_IMPLEMENTED | FK aim only; no IK constraints, no hand/foot contact solver |
| Exercise actions | PLACEHOLDER | `studio3d/exercises/incline_dumbbell_press.json` = key poses per phase, no Action/NLA library |
| BVH/FBX | NOT_IMPLEMENTED | 0 .bvh/.fbx files on box; importers available |
| Retargeting | NOT_IMPLEMENTED | — |
| Gym env / background | IMPLEMENTED_AND_VERIFIED (2D) | `gymform/bg.py` flat Day 4 backdrop 35/39/43; QC (bg),(sep) |
| Equipment | PARTIALLY_IMPLEMENTED | 3D: procedural bench + hex dumbbells (studio3d). Production: equipment comes from generated clips (palette A cobalt plates in prompt) |
| Materials / lighting | PARTIALLY_IMPLEMENTED | Production look = 2D global tone map `gymform/native.py`; Blender lights fixed in `set_v1.json` (unused in production) |
| Camera / animation | PARTIALLY_IMPLEMENTED | 2 fixed cameras front/side; no moves, no close-ups, no layer transitions |
| Blender Python | IMPLEMENTED_BUT_UNVERIFIED | 552-line `build_scene.py`, L1–L4 set-lock QC (`qc_set.py`); last commit 620817f (t562u); not used since |
| Blender ↔ Remotion | NOT_IMPLEMENTED | gymform uses ffmpeg/x264 piping; Remotion 4.0.528 is declared in `shorts-labs/packages/kinetic` (node_modules not installed); `shorts-labs/packages/hero3d` renders Blender stills for other series |
| TTS / subtitles | IMPLEMENTED_AND_VERIFIED | `gymform/tts.py` edge-tts en-US-ChristopherNeural (locked); `gymform/caption.py` 3-line lint; on-screen text via `layout.py` |
| BGM / SFX | NOT_IMPLEMENTED | No music/SFX mix in `run_day.py`/`render.py`. `/workspace/bgm/` holds commercial tracks (e.g. drake.mp3) — do NOT use (licence) |
| QC / render automation | IMPLEMENTED_AND_VERIFIED (low first-pass) | `run_day.py` staged + hashed cache, preflight, 35 QC items (`gymform/qc.py`), temporal gate (`temporal_qc.py`), `DEFECT_PLAYBOOK.md` D1–D19. `output/qc/production_times.jsonl`: Day 6 preflight 11/82 passed (13%), full renders 8/14; Day 7 preflight 0/3 |
| IG publishing | IMPLEMENTED_AND_VERIFIED (guarded) | `tools/ig_publish.py` (graph.instagram.com v22.0, publishes only with `--publish --approved`) |
| IG insights | IMPLEMENTED_BUT_UNVERIFIED as a loop | `/workspace/ig_analysis/fetch.py` reads views, reach, saved, shares, avg watch time per reel; `follows`/`profile_visits` return API errors; not scheduled |

## Production reality
- Day 4–7 picture = generated i2v clips (grok.com, later Grok Bot video tool) graded by `gymform/native.py` (global tone map, no cut-out, after ROOTCAUSE_DAY6).
- Recurring defects (DEFECT_PLAYBOOK): shape warping / hand–dumbbell merge (D2/D3), glow + callout drift (D6), fibre boil (D5), step-cut jump (D8), figure/backdrop melt (D17/D18). Status today: EXISTS gates D1, D10–D12, D17–D19; partly D2, D4, D6, D9, D14; MEASURED only D5, D8, D13, D15, D16; TODO D3, D7.
- Root cause doc: most temporal defects were amplified by our own masks (ratios 1.4–2.4×), but the **step-cut jump and wrong-motion risk come from separate generations** (D8, Day 7 v1 clips turned rows into pulldowns/flys).
- Accuracy risk: callouts and captions print precise-looking percentages with no source (`"74% FRONT DELT OVERLOAD"` in Day 6 v5 meta; Day 4 caption "97% hamstring and glute recruitment"). Angles are measured, percentages are not.
- Cadence: 141 commits since 10-26; 5 reels published 10-27 → 10-29, none since.

## Teammate assets (read-only)
- `/workspace/shorts-labs` (Shorts Labs desk): `packages/render` (spec → 1080×1920 mp4, ffmpeg+libass), `packages/kinetic` (Remotion), `packages/hero3d` (Blender hero stills), dashboard. Reusable: captions/kinetic type, render spec runner.
- `/workspace/legacy/shorts-studio` (Remotion card-news), `/workspace/legacy/shorts-factory`, `shorts-video`, `/workspace/xshorts`.
- Posture 3D (@posture.3d): no separate repo found on the box (`find` for *posture* dirs → none; only `build_v29_posture_fusion_concepts.py` and `universal_posture_anatomy_engine.py` inside this repo).
- Free assets already on box: `/workspace/assets3d` (MakeHuman CC0 mesh/rig/weights/targets, BodyParts3D CC BY 4.0), `pose_landmarker_heavy.task`, `/workspace/models` (RealESRGAN, LaMa, RVM, SAM2).
