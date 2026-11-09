# Quality upgrade plan — toward @demicstory, with free reusable tools

Date: 2026-11-10 (Asia/Seoul). Request: t727u. 0 generations. No paid assets, no xAI, no grok.com.
Compared: delivered `output/_rig3d/db_row/reel/db_row_rig3d_v2.mp4` (run 20261105_1017) vs local Demic frames under `demic_bench/` (third-party, analysed only, never committed) and Day 4 `output/day4_step2_peak.jpg`.
Probe stills (Grab3D rest pose, 540×960, 20 spp, Cycles CPU, nice 19): [`quality_probe_light_pair.png`](quality_probe_light_pair.png), [`quality_probe_before_current.png`](quality_probe_before_current.png), [`quality_probe_after_light_sss.png`](quality_probe_after_light_sss.png), [`quality_probe_after_light_sss_glow.png`](quality_probe_after_light_sss_glow.png), [`quality_probe_grade_bloom_pair.png`](quality_probe_grade_bloom_pair.png). Same files also live under `output/_rig3d/quality_probe/` (gitignored work dir). Replay: `nice -n 19 /workspace/.bpyenv/bin/python tools/rig3d/quality_probe.py`.

Brand note: gymform’s locked look is charcoal + neon green/red (Day 4). Demic’s look is light-grey clay + saturated orange volumetric glow. This plan steals Demic’s *lighting, material, and glow behaviour* and keeps gymform colour identity unless the owner later asks for a light-clay A/B.

## Ranked visual gaps

Judged by eye next to Demic row/delt/push-up frames (`demic_bench/DetDsIDj2BJ.mp4` t8, `DUlzpgekQRy.mp4` t2 and t10, `DePzKDbFB0y.mp4` t3) and by simple luma/detail stats on 2026-11-10. Metrics are guards, not proof (same limit as [`ROOTCAUSE_MUSCLE_LOOK.md`](ROOTCAUSE_MUSCLE_LOOK.md)).

| Rank | Gap | What we see | Evidence | Impact if fixed |
|---|---|---|---|---|
| 1 | Lighting / form modelling | Demic: soft key from above-front + a cool rim that draws a halo on delts/traps + a studio floor with contact shadow. Ours: three generic white area lights, no dedicated rim, figure floats on a flat `#23272B` field. Non-glow muscles read as flat plastic. | FACT: delivered body luma p50 ≈ 70 (`ours_t2.5.png`); Demic delt close-up p50 ≈ 150. FACT: we have no floor mesh (`film_transparent = True` in `retarget_render_v2.py`, backdrop painted in `grade3d.py`). FACT: probe after (key + cool rim + floor) makes belly/groove shading obvious in [`quality_probe_after_light_sss.png`](quality_probe_after_light_sss.png). | Highest. One shared light rig upgrades every exercise. |
| 2 | Shading (SSS, unused maps) | Grab3D already ships packed 2K `rig_basecolor`, `rig_normal`, and `rig_metallic-rig_roughness`. Production disconnects albedo unless `albedo_detail > 0` (yaml default **0**), forces roughness 0.62, and sets Subsurface Weight = 0. Demic’s clay volume comes from SSS-like scatter and fibre normals that catch the key. | FACT: inspected GLB 2026-11-10 with bpy 5.2.3: `char1` 168,649 verts; three 2048² packed images. FACT: `exercises/db_row.yaml` sets `normal_strength: 1.5` and does not set `albedo_detail`. FACT: probe after (albedo_detail 0.50, normal 2.0, cool SSS 0.16) shows organised fascicles on lats/glutes/hamstrings that the before still washes out. | Highest. No new download. |
| 3 | Glow behaviour | Demic: the target muscle *is* a shader (saturated emission + fibre relief + bloom halo); ridges stay readable. Ours: beauty pass has no emission; `grade3d.py` paints `day4_fibre` in 2D on the mask. Green reads as a noisy sticker, not a lit surface. Delivered glow_frac ≈ 0.24 vs Demic row ≈ 0.09 (ours paints a larger, flatter patch). | FACT: `grade3d.py` lines 58–60 call `gm.day4_fibre` on the mask. FACT: 2D bloom on the delivered frame ([`quality_probe_grade_bloom_pair.png`](quality_probe_grade_bloom_pair.png)) immediately gives a Demic-like halo. INFERENCE: wiring the existing `mregion` mask into a beauty-pass emission + Blender Glare `Bloom` will inherit real form (the conclusion of ROOTCAUSE_MUSCLE_LOOK). The rest-pose glow probe is only directional — its mask is a z-band, not anatomy. | Highest for “this looks expensive”. |
| 4 | Grade skin-lock flattens lighting gains | Even a well-lit beauty pass is remapped to Day 4 charcoal (p50 64 / p75 75 / p90 88). That is why the delivered body stays dark after a relatively bright Cycles render. | FACT: `gymform/native.py` comment and `grade3d.py` `skin_lock_lut`. FACT: raw probe before luma p50 ≈ 182; delivered graded frame p50 ≈ 70. ASSUMPTION (testable): a gentler lock (or lock that preserves local contrast) is required, or P1–P2 gains will be partly undone. | High — blocking for P1. |
| 5 | Prop quality | Ours: candy-cobalt hex dumbbell, no bevel, no knurl. Demic: machined round plates, star collar, brushed steel. | FACT: procedural hex in `retarget_render_v2.py` (`primitive_cylinder` 6-gon heads, base colour `(0.02, 0.05, 0.45)`, roughness 0.6). | Medium. Shared prop code, every dumbbell exercise. |
| 6 | Camera / crop language | Demic hooks with overflow close-ups; full-body shots sit on a platform and fill the frame. Ours is a safe mid-shot (bbox height ≈ 0.80) with empty charcoal above the title. | FACT: `render.camera.fill_w: 0.90`, `top_y: 0.27`. INFERENCE: a shared “hook close-up” camera preset (first ~0.5 s) would move perception more than another 4K texture. | Medium (format, not shader). |
| 7 | Garment | Demic: ribbed branded athletic shorts. Ours: matte black procedural briefs, roughness 0.95, no weave. | FACT: `garment` block in `db_row.yaml`; material in `retarget_render_v2.py`. | Low–medium. |
| 8 | Head / proportions | Demic: featureless mannequin head, heroic bodybuilder volumes. Grab3D: more anatomical head (ears, brow) and slightly softer “Meshy-cleaned” volumes. | FACT: Grab3D LICENSE.txt discloses AI-assisted Meshy origin, CC0. High effort to remesh; out of scope for a shared shader pass. | Lower. Do not rebuild the body. |
| 9 | Motion polish | Not visible in stills. Demic uses slow orbits and clean authored reps. We already smooth MediaPipe (`smooth_sigma_f: 2.5`). Cycles motion blur and a hook push-in are free, later. | — | Later. |

Day 4 vs Demic is a *style fork*, not just a quality gap. Day 4 (`day4_step2_peak.jpg`) is a dark écorché illustration with very high-frequency painted fibres (laplacian var ≈ 25,800). Demic is smooth clay (laplacian var ≈ 450–700) with glow only on the target. Our delivered reel sits in between (laplacian var ≈ 10,200) — Day 4 grade noise on a flatter 3D body — so it inherits neither Day 4’s engraved illustration nor Demic’s volumetric clay.

## Free reusable fixes

All licenses checked 2026-11-10:

- Grab3D Human Muscles: CC0 1.0 (`_scratch/assets_probe/LICENSE.txt`). Already on disk.
- Poly Haven assets: CC0, commercial OK, attribution not required ([polyhaven.com/license](https://polyhaven.com/license)). `studio_small_09` is already in the pipeline (1K HDR); the same asset at 2K/4K is `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/studio_small_09_2k.hdr` (CC0).
- ambientCG materials: CC0 1.0 ([docs.ambientcg.com/license](https://docs.ambientcg.com/license/)). Candidate maps, if we want fabric/metal later: Fabric023 / Fabric004 (briefs weave), Metal032 / Metal038 (handle knurl). Not required for P0.
- Blender 5.2.3 Principled SSS, Area lights, and Compositor Glare `Type = "Bloom"` are built in (verified in bpy 2026-11-10).
- Z-Anatomy remains CC BY-SA 4.0 (share-alike on derived meshes) — do not pull it in for a look upgrade.

Do **not** buy OpenGym3D ($19) or any paid character for this. Do **not** generate per exercise.

### P0 — shared code, no new files (do first)

1. **Light rig** in `retarget_render_v2.py` (one function, every exercise): warm key slightly above the camera, cooler fill, dedicated rim opposite the camera (colour ≈ `(0.65, 0.82, 1.0)`). Hide the HDRI from camera rays (`world.cycles_visibility.camera = False`) so the photographed studio never shows.
2. **Studio floor**: a dark grey plane at foot Z, or a shadow catcher. Keep the grade’s contact shadow as a fallback.
3. **Use the maps we already have**: default `albedo_detail: 0.45–0.55`, `normal_strength: 2.0`, keep the packed roughness instead of flattening to 0.62. Cool (not flesh) SSS: Subsurface Weight ≈ 0.15–0.20, radius ≈ `(0.55, 0.58, 0.62)`.
4. **Glow**: drive a Principled+Emission *Add Shader* from the existing `mregion` mask (not a new geometric band). Composite Glare Bloom (threshold ~0.7, strength ~0.5). Then *dial down* `day4_fibre` so it adds fascicle direction instead of being the whole glow.
5. **Loosen `skin_lock`**: preserve local contrast (high-pass from the beauty pass) while still landing in a charcoal band, *or* offer a “clay” lock near luma p50 140–160. Without this, P0 lighting will be crushed back to p50 ≈ 64.
6. **2D bloom fallback** (already prototyped): if we want a gain on the current db_row reel without waiting for a full re-render, add the bloom from `quality_probe.py`’s grade pair into `grade3d.py`. Mask it to the muscle region so callout lines do not bloom.

### P1 — still free, still shared

- Procedural dumbbell: bevel the hex heads, darker gunmetal plates, wave-texture knurl on the handle, less saturated cobalt (keep a thin brand rim if we want). Optional ambientCG CC0 metal maps.
- Upgrade the HDRI to `studio_small_09` 2K (same CC0 asset).
- Briefs: tile an ambientCG CC0 fabric normal at low strength.
- Shared camera preset: optional first-half-second hook close-up on the glowing muscle.

### P2 — only if P0/P1 still feel short

- spp 16 → 32 on delivery frames (time cost, not a look invention).
- Subtle Cycles motion blur.
- Owner A/B: charcoal (brand) vs lifted clay (closer to Demic). Same rig, two grade luts.
- Do **not** resculpt Grab3D or swap in a new character unless the owner asks. Head smoothness and heroic proportions are the expensive leftover.

## Probe results

Rest-pose, same camera, same body:

- Before (production shader + lights): washed clay, weak fibre read, no floor. 6.2 s.
- After lighting/SSS/albedo/rim/floor: readable fascicles, cool rim on the left arm and hip, grounded on a studio plane, luma p50 ≈ 151 (same ballpark as the Demic delt close-up). 9.2 s. This is the main proof.
- After + rest-pose emission: mint glow with a soft halo, but the mask is a crude upper-back band — **production must use `mregion`**. 10.9 s.
- Grade bloom on the real db_row frame: the current green lats pick up an inner glow and a halo. Cheap, and it works on already-rendered frames. Callout lines also bloomed in the prototype; clamp to the muscle mask.

## How close can free materials get?

Honest estimate, after P0+P1, stills: **about two-thirds to three-quarters of Demic’s “premium product shot” quality** — lighting language, clay volume, floor, bloom, and a less toy-like dumbbell. That is an INFERENCE from the probe and the unused 2K maps, not a guarantee on a full exercise reel.

What we will not match with free, reusable, zero-generation tools:

- Their custom character (featureless head, branded tailored shorts, more heroic volumes).
- Per-muscle authored displacement that is tighter than Grab3D’s Meshy-cleaned sculpt.
- Hand-keyed cinematic camera moves and studio-authored motion.

What we *can* match, because it is shared code rather than per-exercise art: the key/rim/floor, SSS clay, emission+bloom that follows real form, a better dumbbell, and a hook close-up. One upgrade hits every playlist item.

## Suggested next step

Implement P0 in `retarget_render_v2.py` + `grade3d.py` (no new assets), then a single db_row frame — not a full reel — for the owner to judge charcoal-vs-clay and bloom strength before any playlist render.
