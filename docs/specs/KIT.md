# Base asset kit

Authored in `tools/rig3d/kit/`, configured by `modules/kit.yaml`. Exercises opt
in with `kit.enabled: true`. All meshes are generated from code and are CC0.
No third-party mesh binaries are committed.

| Piece | How it is made | Licence |
|---|---|---|
| Clay body | Existing Grab3D Human Muscles (rigged) | CC0 1.0 (unchanged) |
| Muscle glow | Weld the Grab3D body (0.1 mm merge), write per-muscle geodesic weights as float attributes (`glow_lats`, `glow_mid_back`, `glow_rear_delt`), and mix peach-orange / purple / pink emissive sheets in the shader. No UVs, no projection, no copied shells. | CC0 (ours) |
| Boxer shorts | Closed tube sized from the body's own waist, hip and mid-thigh cross-sections (about 10/15/14 mm ease), rest-pose cloth (5 mm collision, self-collision), stubs welded before solidify, then a 10 mm seat-only displace after armature so the posed hip cannot poke through. Separate thicker elastic waistband, mid-thigh hem, matte black, no wordmark. | CC0 (ours) |
| Bench | Same procedural flat bench, with a rounder pad bevel and a top piping when `kit.bench.padded`. Pad top stays at 0.43 m so the posed contacts do not move. | CC0 (ours) |
| Dumbbell | Same procedural plates + handle, with a crossed-sine knurl displaced on the handle verts. | CC0 (ours) |
| Lights | Poly Haven `studio_small_09` HDRI (CC0, fetched to `_scratch/hdri/`, not committed) at low strength, plus an authored three-point area rig and a shadow-receiving charcoal floor. | HDRI: CC0 (Poly Haven). Rig: CC0 (ours) |
| Porcelain clay | Cooler, higher-value base colour, weaker AO/cavity and normal strength. | n/a (shader) |

Projection paint (`tools/rig3d/project_paint.py`) and the body-face offset
briefs stay in the repo for reference. They are not on the hot path when
`render.glow_shape: attr` and `garment.method: cloth`.

The pose (authored pull_row, bench contacts, camera) is unchanged.

Shapekey transfer onto the welded copy reads Key blocks directly (Blender 5
`Object.to_mesh()` returned the basis when the armature was muted, which zeroed
the working-hand fist). The compositor adds Fog Glow on the emission pass only
and composites the spill outside the glow mask as a halo.
