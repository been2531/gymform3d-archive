# Écorché anatomy model — license and fetch

We glow **separated muscle meshes** (not painted regions on the Grab3D
one-piece skin). The free model we use:

| Field | Value |
| --- | --- |
| Project | [Charleeh/ecorche](https://github.com/Charleeh/ecorche) |
| File | `model/ecorche.glb` (not `ecorche.meshopt.glb`) |
| Size | 14,314,536 bytes |
| Contents | 634 named meshes: 203 superficial muscles, 159 deep, 272 skeleton |
| Units | metres, Y-up, +Z facing; soles at y=0; root scale `0.0001` (apply after import) |
| License of this derivative | **CC BY-SA 4.0** |
| Upstream | BodyParts3D © The Database Center for Life Science (**CC BY 4.0**); Z-Anatomy by Gauthier Kervyn (**CC BY-SA 4.0**). Latissimus dorsi (missing from raw BodyParts3D) comes from Z-Anatomy. |

## Attribution (caption or bio)

> Écorché (github.com/Charleeh/ecorche), CC BY-SA 4.0. Based on BodyParts3D,
> © DBCLS, CC BY 4.0, and Z-Anatomy by Gauthier Kervyn, CC BY-SA 4.0.

Share-alike applies to redistributed mesh derivatives of this model. Do not
commit the GLB; keep it in gitignored `_scratch/ecorche/`. Rendered frames
should still carry the credit above.

## How to fetch (do not vendor the binary)

```
python3 tools/rig3d/fetch_ecorche.py
```

Writes `_scratch/ecorche/ecorche.glb` and copies this note beside it.
Re-run is a no-op when the file is already the expected size.

## Why not the other free candidates

- **Grab3D Human Muscles** (already in use, CC0): one mesh + UV atlas. No
  separate-muscle variant. That is why earlier looks read as stickers.
- **Z-Anatomy** The-blend / `Z-Anatomy.zip` (~87 MB, CC BY-SA 4.0): excellent
  but much larger; ecorche already incorporates its latissimus fill-in.
- **BodyParts3D / Anatomography** official 4.0 OBJ zip (~136 MB, CC BY-SA 2.1
  JP): raw dump may lack a named latissimus; too heavy for this disk.

Ecorche is the lean high-quality route: English anatomical names, trapezius
split into descending / transverse / ascending, deltoid split into clavicular /
acromial / spinal, plus named rhomboids, teres, infraspinatus, lats.
