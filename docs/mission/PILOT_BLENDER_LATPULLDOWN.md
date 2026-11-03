# Pilot: Blender Day-4 set — Wide-Grip Lat Pulldown

## Summary (v2.1)

`studio3d/pilot/` is a **deterministic** headless-Blender pipeline for **wide-grip lat pulldown**.  
**v1** failed on look (smooth MH body, opaque glow blobs, wrong A/B reference, floating bar plates).  
**v2/v2.1** rebuild the figure from **[Charleeh/écorché](https://github.com/Charleeh/ecorche)** segmented muscle meshes (CC BY-SA 4.0 — **owner approval pending**): shoulder-span scale fit after pose, quadriceps/thigh muscles restored (v2.1 fixed an over-broad `femoris` exclude), dual-band UV striations + albedo fibres, emission modulated so fibres stay visible through glow, lat stack behind the athlete, and A/B against **Day 4 locked RDL** reference frames (look-only; exercise differs).

Deliverables: `output/pilot/lat/` and `/opt/cursor/artifacts/screenshots/ab_keypose_vs_day4_v2.jpg`.

---

## Source shortlist (Step A research, ranked)

Licences verified from primary pages/repos (Nov 2026). **NC = excluded.**

| Rank | Source | What it gives | Licence | Commercial | Fit to Day 4 look | Pilot use |
|---|---|---|---|---|---|---|
| 1 | **[Charleeh/écorché](https://github.com/Charleeh/ecorche)** `ecorche.glb` | ~575 named muscle/bone meshes; lats filled vs raw BP3D; cleaned layering | **CC BY-SA 4.0** | Yes, with attribution + **SA on adapted 3D** | **Best** open segmented écorché we found | **Adopted v2.1** (~90+ upper-body muscles incl. quads) |
| 2 | **BodyParts3D** 4.0 OBJ (ISA/partof) | Individual FMA muscle OBJs | **CC BY-SA 2.1 Japan** | Yes, with SA | Good geometry, weak alone (no lats in ISA); striations need shaders | Already in repo; upstream of écorché |
| 3 | **[Z-Anatomy](https://github.com/Z-Anatomy/Models-of-human-anatomy)** | Blender template, BP3D-derived + extras | **CC BY-SA 4.0** (+ **NC** kidney etc.) | Yes for clean subset; NC parts must be removed | High detail; heavy template; SA | **Not imported** — flagged for owner; écorché is lighter |
| 4 | **MyoGeneratorRemix** ([GPL-3.0](https://github.com/MiguelDLM/myoGeneratorRemix)) | Procedural muscle loft + Voronoi fibre material | GPL addon | Yes for renders; addon is GPL | Fibres are procedural/Voronoi, not Day-4 photographic | Reference only for fibre node ideas |
| 5 | **MakeHuman + MPFB2** CC0 base | Rigged body, no écorché surface | **CC0** | Yes | Poor fibre read (owner rejection) | Rig + shorts + head only (body hidden v2) |
| 6 | **CMU mocap BVH** + Blender import | Motion reference | **Free research use** (CMU terms) | Generally yes for derived animation | Motion only | Not used yet (FK posing from motion standards) |
| 7 | **Mixamo** | Rigged characters + mocap | **[Adobe terms](https://www.adobe.com/legal/terms.html)** — **no redistribution** of assets; game/video use allowed with limits | Restrictive for **locked OSS set** in git | N/A | **Not used** |
| 8 | **Sketchfab CC0/CC-BY écorché** | Various artist models | Per-model | Many OK | Uneven quality; licence hunt cost | Not selected (écorché ranked higher) |
| 9 | **NIH 3D Print Exchange** | Anatomical meshes | Mixed (many NC) | Case-by-case | Not tuned for real-time | Not used |
| 10 | **Poly Haven / ambientCG** | PBR materials, not anatomy | CC0 | Yes | No muscle topology | Not used |
| 11 | **Blender Studio** open assets | General production assets | CC0 / project-specific | Yes | No écorché hero | Not used |
| 12 | **Open3DModel / random mirrors** | Inconsistent provenance | Often **unclear/NC** | Risk | — | **Avoided** |

**Gym equipment:** no satisfactory **CC0 wide-grip lat pulldown** mesh found in time budget → **procedural** warm-grey stack + palette-A pins (`studio3d/pilot/equipment.py`).

**Shaders:** no drop-in Day-4 fibre node group found → custom **`shaders.py`**: UV principal-axis bands + emission modulated by striation ( fibres stay visible through glow).

---

## v2 architecture

| Piece | Module |
|---|---|
| Rig + head + shorts | MakeHuman CC0 via `studio3d/build_scene.py` |
| Visible anatomy | `anatomy_ecorche.py` → glTF import, upper-body filter, bone parenting |
| Fibre + glow | `shaders.py` |
| Lat machine + steel bar | `equipment.py` (stack **behind** +Y; **no** floating end plates) |
| Motion | `exercises/wide_grip_lat_pulldown.json` |
| QC | `qc_pilot.py` |

---

## Honest gap vs Day 4 (after v2.1)

| Day 4 reference | v2.1 pilot | Still blocking “match or beat” |
|---|---|---|
| Continuous écorché with fine fibre **throughout** the silhouette | Segmented BP3D-derived meshes; **dual-band UV** striations + albedo | **Seams** between muscle parts; fibre frequency still **coarser** than generative Day 4 |
| Glow = muscle lights up, fibres visible | Per-muscle emission × striation on **lat / bi/tri/delt/trap only** | Side-view lats read well; front contract glow **small**; not as soft/photographic as Day 4 red on RDL |
| Single cohesive skin tone | Charcoal ~48/50/54 sRGB + albedo swing | Post **skin lock** (Day 4 p90=88) not applied in Blender yet |
| Photographic depth in striations | Bump + dual wave | No artist-authored **normal map** per muscle |
| Lat pulldown scene | Procedural stack + steel bar | Exercise mismatch in A/B (RDL ref = **look** only) |

**Verdict:** v2.1 **approaches** Day 4 on side-view fibre read and lat-local emerald glow (see `ab_keypose_vs_day4_v2.jpg`). It does **not** fully **match or beat** Day 4 polish (continuous surface, glow softness, tone lock). Next levers: Z-Anatomy subset with SA sign-off, baked tangent normals, optional post grade matching `gymform_spec.json` skin lock.

---

## Performance

| Step | Wall time (8 CPU, 1080×1920) |
|---|---|
| glTF import (74 muscles) | ~2 s (per frame today) |
| Cycles 16 spp beauty | ~15–25 s |
| 8 key poses | ~3–4 min total |
| 24-frame loop | ~8–10 min |

---

## Commands

```bash
python3 studio3d/fetch_assets.py
curl -sL -o assets3d/ecorche.glb https://github.com/Charleeh/ecorche/raw/main/model/ecorche.glb
python3 studio3d/pilot/render_keyposes.py
python3 studio3d/pilot/make_ab_sheet.py
python3 studio3d/pilot/assemble_reel.py
python3 studio3d/pilot/qc_pilot.py
```

See `studio3d/pilot/ATTRIBUTION.md` for licence obligations.

---

## Reuse for another exercise

Same as v1: add `exercises/<slug>.json`, extend `glow_targets()` / pose function, keep `set_pilot.json` cameras and QC gates.
