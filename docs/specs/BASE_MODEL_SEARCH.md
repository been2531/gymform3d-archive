# Base-model search — replacement for Grab3D char1

Access date: 2026-11-11 (Asia/Seoul). Grab3D Human Muscles (`char1`) is a
declared failure (834 UV islands, baked ankle retinaculum, baked
`rig_normals`). This search evaluates legally reusable replacements
*before* adopting one. No Grab3D patching and no `db_row` render was done.

**Verdict: none of the three downloadable candidates qualifies as the new
production clay body.** Do not adopt any of them. The closest *look* is a
historic statue we cannot pose. The closest *muscle atlas* is not a
watertight skin. The most promising athletic écorché (Diego Luján, CC BY
4.0) could not be downloaded this round (Sketchfab 401).

Comparison sheet (gitignored stills, viewed at full size before this
write-up):
`output/_basemodels/eval/basemodels_vs_demic.png`

Demic reference: left panel of
`output/_rig3d/v4_authored/peach34_air/ours_vs_demic_peak.png`
(box `(58, 84, 482, 804)`, scaled to 540×960). Candidate stills are
porcelain clay (albedo `[0.42, 0.428, 0.442]`, roughness 0.93), charcoal
backdrop, rear 3/4, 540×960, Cycles 24 spp, Blender 4.2.3, nice 19.

Replay (downloads stay under gitignored `output/_basemodels/`):

```
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  --python tools/rig3d/eval_base_model.py -- \
  --input <glb-or-obj-dir> --out-dir output/_basemodels/eval --name <id>
python3 tools/rig3d/build_base_model_sheet.py \
  --out output/_basemodels/eval/basemodels_vs_demic.png ...
```

Weld test: merge-by-distance at 0.1 mm after applying scale to metres.
Boundary edges are edges with ≠2 faces. Numbers below are measured, not
taken from listings.

## Hard requirements (all must pass)

- Free licence that allows commercial Instagram use (CC0 or CC BY). CC
  BY-SA is internal reference only.
- One continuous watertight skin after a 0.1 mm weld (few boundary edges,
  real orifices only).
- Muscle definition in the geometry, *or* separate per-muscle meshes
  usable for glow — but the clay body itself must still be continuous.
- No baked straps, clothing, or garment seams; clean normals.
- Riggable to our armature, or honestly retargetable from a bind pose.

## The three measured candidates

### 1. Innerscene — Full-body muscular system, superficial layer

- Source: <https://www.innerscene.com/tools/library/3d-parts/full-body-muscular-system-superficial-layer-8e9b43f6>
- Licence: **CC0 1.0** (library page, access date 2026-11-11). Commercial
  use allowed, no attribution required.
- File: `full-body-muscular-system-superficial-layer.glb`, 15,895,376 bytes.
- Height 1.75 m, standing, unrigged (0 bones).

Measured (all 157 meshes, including the fascia):

| | |
|---|---|
| Mesh objects | 157 (156 named muscles + one fascia) |
| Verts / tris | 143,809 / 239,224 |
| Boundary edges | 56,438 |
| After 0.1 mm weld | 493 boundary, **455 non-manifold**, 139 UV islands |
| UV islands before weld | 1,535 |
| Armature | none |

The largest object is named `MakeHuman CC0 | continuous fascial
reference`: 70,985 verts, 43,614 boundary edges, **1,163 UV islands**.
That is the same failure class as Grab3D char1. The muscles-only still
(fascia hidden) shows large gaps at the face, abdomen and groin — an
educational atlas, not a clay mannequin. Rendering fascia and muscles
together overlap and blow out. No bind pose, no rig.

**Fails:** watertight skin, clean continuous clay, riggable.
**Keeps as:** optional CC0 muscle-name reference, not a base body.

### 2. Małopolska / Francavilla — Anatomical figure, écorché

- Source: Virtual Museums of Małopolska scan of Pietro Francavilla
  (inventory 3916; 19/III), Jagiellonian University Museum. Sketchfab
  `0b4bf087f6064180895917e4a93f527b`, downloaded from Zenodo
  <https://doi.org/10.5281/zenodo.21528504> (record 21528504).
- Licence: **CC0 1.0** (Sketchfab + Zenodo `cc-zero`, access date
  2026-11-11).
- File: `0b4bf087f6064180895917e4a93f527b.glb`, 15,274,548 bytes.
- Height **0.547 m** (a statuette, not a 1.8 m figure). Unrigged.

Measured:

| | |
|---|---|
| Mesh objects | 3 (`Object_2/3/4`, including the wooden plinth) |
| Verts / tris | 175,096 / 282,480 |
| Boundary edges | 64,962 |
| After 0.1 mm weld | **650** boundary, 0 non-manifold, 912 UV islands |
| UV islands before weld | 1,804 |
| Armature | none |
| Source textures | 9 packed images |

This is the only candidate whose *sculpted muscle grooves* are in the
same league as demic (lats, glutes, hamstrings, calves read as anatomy,
not as a UV sticker). That is not enough. It is a mid-stride theatrical
pose on a baked plinth, in three scan chunks, 650 leftover boundary
edges after the weld (not orifices), and 1,804 UV islands. There is no
bind pose. Putting this into a dumbbell row is a retopology and a rig,
not a download. Two sibling CC0 statues from the same museum (Zenodo
21291060 and 21376387) are the same class of historic pose.

**Fails:** watertight-enough skin, no baked stand, riggable / bind pose,
life-size continuity.
**Keeps as:** CC0 sculpt reference only.

### 3. BodyParts3D — official IS-A skeletal muscle set

- Source: Database Center for Life Science, `isa_BP3D_4.0_obj_99.zip`
  and `isa_parts_list_e.txt` from
  <https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html>
- Licence: the official archive page
  (<https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html>, last
  updated 2025-02-27, re-fetched 2026-11-11) specifies **CC BY 4.0**.
  Required credit: “BodyParts3D, © The Database Center for Life Science
  licensed under CC Attribution 4.0 International.” Also cite Mitsuhashi
  N. et al., *Nucleic Acids Res.* 2008, PMID 18835852.
- Caveat (FACT, not resolved): every 2013 OBJ header still names CC
  BY-SA 2.1 Japan and points at the same `lic.html`, which now says CC
  BY 4.0. Get written confirmation from DBCLS before a commercial
  launch; do not strip the headers until then.
- Selection: 572 muscle concepts → **426 unique OBJ files**.
  **Latissimus dorsi is absent** from the official list (confirmed: no
  `latissimus` string in `isa_parts_list_e.txt`).

Measured (426 objs, scaled mm → m, rotated Y-up to stand on Z):

| | |
|---|---|
| Mesh objects | 426 named muscles |
| Verts / tris | 1,764,961 / 2,448,586 |
| Boundary edges | 968,534 |
| After 0.1 mm weld | **7,165** boundary, **3,261 non-manifold**, 660 loose |
| UV islands | 0 (no UVs) |
| Armature | none |
| Height | 1.61 m |

The rear still shows the missing latissimus as a hole through the mid
back. Head and feet are incomplete. The meshes are the known holey
atlas, not a clay skin. They satisfy “separate per-muscle meshes” only
as a source that would still need days of cleaning, a latissimus fill,
and a separately authored continuous body to shrink-wrap or glow onto.

**Fails:** continuous watertight skin, complete superficial set (no
lats), riggable, header-licence certainty.
**Keeps as:** internal FMA-named muscle reference after the licence
caveat is respected. Do not take the cleaned Z-Anatomy / Charleeh
retopo if we want to stay clear of ShareAlike.

## Searched and not adopted (not in the sheet)

| Asset | Licence | Why it is not a candidate this round |
|---|---|---|
| Grab3D Human Muscles (current) | CC0 | Declared failure. Not re-tested. |
| Diego Luján, *Male Full Body Ecorche*, Sketchfab `ab11ebff…` | CC BY 4.0, downloadable, 640,470 tris, “standard athletic man” | **Most promising unmeasured athletic écorché.** Sketchfab `/download` returned **401** without credentials. Not fetched. |
| CharacterZone, *Male base muscular anatomy*, `0954aa04…` | CC BY 4.0, 37,648 tris, claims “ready for rig” | Same 401. Lower poly; topology unverified. |
| Beatriz Gomez, *Ecorche - Anatomy study*, `e402d3d5…` | CC BY 4.0 | Same 401. |
| Z-Anatomy Blender atlas | CC BY-SA 4.0 | ShareAlike — internal reference only. Not downloaded. |
| Charleeh/ecorche glTF (634 named meshes) | CC BY-SA 4.0 (BodyParts3D + Z-Anatomy) | ShareAlike. Not used as a production candidate. |
| MakeHuman / MPFB2 core body | CC0, rigged | Already rejected: smooth, no muscle grooves. The Innerscene fascia is this body. |
| MB-Lab | AGPL on base assets | Not commercially usable as a closed body. |
| SMPL / SMPL-X / AMASS / Fit3D | non-commercial | Forbidden for a monetised Instagram account. |
| Innerscene male base mesh, rigged A-pose | CC0 | Smooth clay, no muscle grooves. Skipped. |
| Solarz Jagiellonian rebuilds (ArtStation, CC0) | CC0 | Same Francavilla statues; ArtStation JSON was Cloudflare 403. The museum/Zenodo scans already cover this look. |
| Poly Haven, Blend Swap “Large Muscled Man”, OpenSim | mixed | No modern, complete, commercial-use athletic écorché turned up that we could legally fetch and verify this round. |

## Recommendation

**Adopt none of these three.** A muscle atlas and a Renaissance statuette
are not a demic-style gym mannequin, and adopting either would be
another round of patching.

What would actually have a chance, in order:

1. **Download and measure Diego Luján’s athletic écorché (CC BY 4.0)
   and, if useful, CharacterZone’s muscular base** — once a Sketchfab
   login exists. These are the only remaining free, commercial-use,
   standing athletic bodies that looked right on paper and were not
   measured. Do not start production on them until the same weld / UV /
   strap / bind-pose checks in this note pass.
2. If those also fail the watertight-skin test, **there is no free
   drop-in.** The method that works by construction is to author our
   own continuous, welded, unstrapped clay écorché (CC0), using the
   Małopolska sculpt and/or Diego as *reference only*, bind it once,
   and keep per-muscle glow on attributes or on our own overlay
   shells. That is modeling work, not a download, and it should be
   scoped honestly before anyone starts it.
3. A paid athletic écorché would also need 중현 님’s explicit approval
   (standing rule: no paid assets).

Do not go back to Grab3D char1. Do not ship Z-Anatomy or Charleeh
meshes. Do not treat BodyParts3D’s missing latissimus as a small fix.
