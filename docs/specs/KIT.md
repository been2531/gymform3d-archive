# Base asset kit

Authored in `tools/rig3d/kit/`, configured by `modules/kit.yaml`. Exercises opt
in with `kit.enabled: true`. All meshes are generated from code and are CC0.
No third-party mesh binaries are committed.

| Piece | How it is made | Licence |
|---|---|---|
| Clay body | Existing Grab3D Human Muscles (rigged) | CC0 1.0 (unchanged) |
| Muscle glow | Overlay shells: lats, mid-trap/rhomboid, rear delt. Copied from Grab3D faces, voxel-remeshed, inflated, bound to the same armature. Opaque emission, not projection paint. | CC0 (ours) |
| Boxer shorts | Convex-hull loft from the hip and knee bones: wide waistband, seat that bridges the cleft, mid-thigh hem, matte black, no wordmark. | CC0 (ours) |
| Bench | Same procedural flat bench, with a rounder pad bevel and a top piping when `kit.bench.padded`. Pad top stays at 0.43 m so the posed contacts do not move. | CC0 (ours) |
| Dumbbell | Same procedural plates + handle, with a crossed-sine knurl displaced on the handle verts. | CC0 (ours) |
| Lights | Poly Haven `studio_small_09` HDRI (CC0, fetched to `_scratch/hdri/`, not committed) at low strength, plus an authored three-point area rig and a shadow-receiving charcoal floor. | HDRI: CC0 (Poly Haven). Rig: CC0 (ours) |
| Porcelain clay | Cooler, higher-value base colour, weaker AO/cavity and normal strength. | n/a (shader) |

Projection paint (`tools/rig3d/project_paint.py`) and the body-face offset
briefs stay in the repo for reference. They are not on the hot path when
`render.glow_shape: shells` and `garment.method: loft`.

The pose (authored pull_row, bench contacts, camera) is unchanged.
