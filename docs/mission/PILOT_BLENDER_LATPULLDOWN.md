# Pilot: Blender Day-4 set — Wide-Grip Lat Pulldown

## Summary

This pilot adds `studio3d/pilot/`, a **deterministic** headless-Blender pipeline for one exercise (wide-grip lat pulldown). It does not replace `studio3d/set_v1.json` or the generated-clip production path.

Deliverables live under `output/pilot/lat/` and `/opt/cursor/artifacts/screenshots/ab_keypose_vs_day4.jpg`.

## Reused from the repo

| Piece | Role |
|---|---|
| `studio3d/build_scene.py` | MakeHuman load, faceless head, shorts shell, bone `aim()` posing, camera helper |
| `studio3d/set_v1.json` | CC0 rig/morph asset URLs and lock hashes (via `fetch_assets.py` → `assets3d/`) |
| `configs/motion_standards.json` | `wide_grip_lat_pulldown` trunk/arm signatures for QC |
| `gymform/qc.py` thresholds | Backdrop flatness (≤4 lum span), separation spirit (skin vs BG vs machine) |
| `output/day2_scene1_wrong.jpg` | Day-4-**look** reference frame for A/B (same exercise family as Day 2) |

## Open-source tools (commercial use verified)

| Tool | Licence | Use in pilot |
|---|---|---|
| **Blender 4.0.2** (system package) | GPL-3.0+; rendered output is yours | Headless Cycles CPU, scene build |
| **MakeHuman** base mesh + default rig + macro targets | CC0 1.0 | Mannequin mesh + deformation |
| **OpenCV**, **NumPy**, **Pillow** | Apache/BSD | QC, A/B sheet, reel concat |
| **FFmpeg** | LGPL/GPL (dynamic link via CLI) | Loop + 4-step test reel |
| **MediaPipe** | Apache 2.0 | Available for future 2D angle cross-check (not required for 3D meta QC) |

**Not used:** SMPL-X, GVHMR, Bandai Namco mocap, Z-Anatomy (share-alike), `/workspace/bgm` music, paid APIs.

**BodyParts3D** (CC BY 4.0) is used in legacy `studio3d` but **not** in this pilot glow path (vertex-group glow shells instead).

## Look vs Day 4 (honest gaps)

| Target | Pilot approach | Gap vs Day 4 master sheet |
|---|---|---|
| Flat RGB 35/39/43 backdrop | World background, no floor shadow catcher | Match is good; slight GPU colour noise at 16 spp |
| Muscle-fibre striations | Procedural Noise + Wave → Bump on Principled BSDF | **Weaker** than photographic / generative fibre in Day 4; reads as subtle bump, not fine écorché striations |
| Faceless écorché silhouette | Same MakeHuman + egg head as `set_v1` | Torso still **smooth** compared with Day 4 reference; owner rejection reason unchanged at mesh level |
| Red / emerald muscle glow | Thin emission shells on rig-weight regions (lats vs biceps/delt) | Area and softness approximate; no flow-tracked native glow |
| Palette A bar | Procedural cobalt plates + gold rim + steel bar | Stable and separable; lacks brushed metal detail |
| Warm-grey machine | Procedural lat stack + seat | Stable; simpler than real gym geometry |

## Motion

- **Wrong:** trunk lean up to ~42° at contraction, elbows forward (high `arm_torso`), red glow on biceps + front delt regions.
- **Right:** trunk ~18–25°, elbows down (`arm_torso` ≤ ~40°), emerald glow on lat-weighted back shells.
- Tempo for loops: cosine 0→1→0, 1.67 s per half-cycle (Day 4 cosine spec), 24 fps.

3D angles and trunk lean are written to `*_meta.json` per frame.

## Render performance (this VM: 8 CPU, no GPU)

| Setting | Wall time (measured) |
|---|---|
| 1080×1920, 16 spp, one beauty frame | **~15 s** (~0.015 s/px-spp effective) |
| 24-frame loop (12 spp) | ~6–8 min |
| 8 key-pose stills | ~2–3 min |

Video rate: **~15 s per frame** → **~0.6 s of 24 fps video per minute** of render wall time at 16 spp (pilot loops use 12 spp).

## Reuse for another exercise

1. Add `studio3d/pilot/exercises/<slug>.json` with `forms.{wrong,right}.stretch|contract` joint targets (same schema as `wide_grip_lat_pulldown.json`).
2. Implement or reuse a `pose_<slug>()` in `build_scene.py` (or split `pose_lat.py` pattern).
3. Add equipment builder if not lat machine (keep `set_pilot.json` equipment block or extend).
4. Set `glow.regions_wrong` / `regions_right` to rig-weight region keys.
5. Register cameras in `set_pilot.json` (`front`, `side` — same four-step layout).
6. Run `render_keyposes.py` → `make_ab_sheet.py` → `qc_pilot.py`.

Optional: wire `meta.json` into `gymform/motion_check.py` for 2D projection checks on composited output.

## Commands

```bash
python3 studio3d/fetch_assets.py
python3 studio3d/pilot/render_keyposes.py
python3 studio3d/pilot/make_ab_sheet.py
python3 studio3d/pilot/render_loop.py right front 24
python3 studio3d/pilot/render_loop.py wrong front 24
python3 studio3d/pilot/assemble_reel.py
python3 studio3d/pilot/qc_pilot.py
```

## QC checklist (automated in `qc_pilot.py`)

- Deterministic re-render (max pixel diff ≤ 1)
- Hand–bar clearance (`grip_distance_m` vs bar half-width)
- Trunk / arm angles vs motion standard
- Backdrop flatness, skin/machine/backdrop separation
- Glow mask area (ID pass)
- Temporal mean frame delta on loop PNG sequence
