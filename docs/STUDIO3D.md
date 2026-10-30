# Locked 3D set (t561u–t562u) — "structure it so it never drifts"

Generated clips drift every time (mannequin shape, dumbbell type, bench angle, range of motion). The fix is
architectural: **one fixed 3D set, rendered natively**, and each day supplies only an exercise config.

| Layer | Locked where | What |
|---|---|---|
| Mannequin | `studio3d/set_v1.json` → `mannequin` | MakeHuman base mesh + default rig + weights (CC0), macro morph male/young/max-muscle/lean (CC0 targets), faceless egg head, Day 4 skin (post lock to spec p50/p75/p90), 4 mm matte black shorts shell from the body's own faces |
| Equipment | `equipment` | procedural incline bench + hex dumbbells (our geometry, fixed sizes/materials) |
| Anatomy | `anatomy_parts` | BodyParts3D bones/muscles (CC BY 4.0, attribution below), fitted once to the rig and parented to rig bones — true 3D, target parts only |
| Look | `lights`, `world_rgb_linear`, `render`, `xray` | fixed lights, Cycles CPU, Day 4 backdrop (`bg.studio_bg`) in post, Day 1-style X-Ray rim shader |
| Cameras | `cameras.front/side` | fixed presets; front and side render the **same posed scene** → views match by construction |
| Contact | `mannequin.contact` | body z from seat contact, bench slid fore-aft to touch the back (no float, no clip) |

Per day: `studio3d/exercises/<exercise>.json` = pose keyframes from `configs/motion_standards.json`
(bench incline, arm–torso angle, elbow flexion at bottom/top), mistake_type, target bones and muscles.

## Flow
1. `python3 studio3d/fetch_assets.py` (once; verifies sha256 vs `set_lock.json`)
2. `/workspace/.bpyenv/bin/python studio3d/build_scene.py --exercise incline_dumbbell_press --form wrong --phase 0 --camera front --out _tmp_3d/x`
   → beauty / flat-ID / X-Ray bone passes + meta (projected joints, **true 3D angles**, set hash)
3. `python3 studio3d/composite.py _tmp_3d/x --step 1 --out <jpg>` → Day 4 skin lock, shorts tone, bones on skin
   pixels only, Day 4 backdrop, Day 4 callout (number = measured 3D angle), title/preview (shared `gymform`)
4. `python3 studio3d/qc_set.py <meta.json ...>`

## QC (locked-set gate)
- (L1) every asset file sha256 == `set_lock.json`
- (L2) set config + builder/compositor hash == lock (materials, lights, cameras, equipment cannot change silently)
- (L3) every rendered frame's meta `set_hash` == lock (all steps / both cameras from the identical set)
- (L4) posed 3D angles meet the motion standard for the form (wrong: arm–torso > 75°; right: 45–60° ±10; elbow flexion 80–110° at the bottom)
- `--write-lock` only when the owner approves a new set version.

## Cost / time (box: 1 CPU, no GPU)
- 1080×1920, 24 samples: beauty ≈ 3.5 min, ID ≈ 15 s, bones ≈ 20 s per frame. A 20 s reel is not rendered per frame
  at this setting: plan = render key poses (tempo phases) at full quality + interpolate, or lower samples + denoise.
- Zero spend: Blender (bpy 5.2, pip), all assets free for commercial use.

## Attribution (required)
BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International.
MakeHuman assets: CC0 1.0.
