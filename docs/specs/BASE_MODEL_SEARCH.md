# Base-model search — replacement for Grab3D char1

Access date: 2026-11-11 (Asia/Seoul). Grab3D Human Muscles (`char1`) is a
declared failure (834 UV islands, baked ankle retinaculum, baked
`rig_normals`). This search evaluates legally reusable replacements
*before* adopting one. No Grab3D patching and no `db_row` render was done.

**Verdict: none of the measured candidates qualifies as the new
production clay body.** Do not adopt any of them. The first-round
closest *look* is a historic statue we cannot pose. The closest *muscle
atlas* is not a watertight skin. The Sketchfab athletic écorché that
looked most promising on paper (Diego Luján, CC BY 4.0) is now
measured: the clay still has real muscle form, but it is 16 ZBrush
subtools / 284 separate shells after a 1.75 m × 0.1 mm weld, not one
skin. A fourth Sketchfab download, *Gladiator*, was excluded before
evaluation (For Honor-inspired design and a NoAI tag on the page;
copyright risk). Its zip and unpacked folder were deleted and it is
not in either comparison sheet.

Comparison sheets (gitignored stills, viewed at full size before this
write-up):

- First round: `output/_basemodels/eval/basemodels_vs_demic.png`
- Sketchfab round: `output/_basemodels/eval/sketchfab_vs_demic.png`

Demic reference: left panel of
`output/_rig3d/v4_authored/peach34_air/ours_vs_demic_peak.png`
(box `(58, 84, 482, 804)`, scaled to 540×960). Candidate stills are
porcelain clay (albedo `[0.42, 0.428, 0.442]`, roughness 0.93), charcoal
backdrop, rear 3/4, 540×960, Cycles 24 spp, Blender 4.2.3, nice 19.

Replay (downloads stay under gitignored `output/_basemodels/`):

```
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  --python tools/rig3d/eval_base_model.py -- \
  --input <glb-or-gltf> --out-dir output/_basemodels/eval --name <id> \
  --exposure -0.20
# optional: scale height to 1.75 m before the 0.1 mm weld, no still
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  --python tools/rig3d/eval_base_model.py -- \
  --input <glb-or-gltf> --out-dir output/_basemodels/eval --name <id>_h175 \
  --target-height 1.75 --no-render --exposure -0.20
python3 tools/rig3d/build_base_model_sheet.py \
  --out output/_basemodels/eval/sketchfab_vs_demic.png ...
```

Weld test: merge-by-distance at 0.1 mm after applying scale to metres.
Boundary edges are edges with ≠2 faces. Loose parts are face-connected
islands. When the importer’s mm/cm heuristic leaves a figure far from
life size (Diego 6.8 cm, Beatriz 2.7 cm), a second weld is run after
normalizing height to 1.75 m so the 0.1 mm threshold is meaningful;
those normalized numbers are the load-bearing weld figures. Numbers
below are measured, not taken from listings.

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
| Diego Luján, *Male Full Body Ecorche*, Sketchfab `ab11ebff…` | CC BY 4.0, 640,470 tris | **Measured this round.** Best look; fails as a skin (284 parts at 1.75 m). See Sketchfab round below. |
| CharacterZone, *Male base muscular anatomy*, `0954aa04…` | CC BY 4.0, 37,648 tris, claims “ready for rig” | Same 401. Lower poly; topology unverified. |
| Beatriz Gomez, *Ecorche - Anatomy study*, `e402d3d5…` | CC BY 4.0, 418,978 tris | **Measured this round.** Tendon/bone study; 8,890 parts at 1.75 m. See below. |
| Z-Anatomy Blender atlas | CC BY-SA 4.0 | ShareAlike — internal reference only. Not downloaded. |
| Charleeh/ecorche glTF (634 named meshes) | CC BY-SA 4.0 (BodyParts3D + Z-Anatomy) | ShareAlike. Not used as a production candidate. |
| MakeHuman / MPFB2 core body | CC0, rigged | Already rejected: smooth, no muscle grooves. The Innerscene fascia is this body. |
| MB-Lab | AGPL on base assets | Not commercially usable as a closed body. |
| SMPL / SMPL-X / AMASS / Fit3D | non-commercial | Forbidden for a monetised Instagram account. |
| Innerscene male base mesh, rigged A-pose | CC0 | Smooth clay, no muscle grooves. Skipped. |
| Solarz Jagiellonian rebuilds (ArtStation, CC0) | CC0 | Same Francavilla statues; ArtStation JSON was Cloudflare 403. The museum/Zenodo scans already cover this look. |
| Poly Haven, Blend Swap “Large Muscled Man”, OpenSim | mixed | No modern, complete, commercial-use athletic écorché turned up that we could legally fetch and verify this round. |

## Sketchfab round (measured 2026-11-11, Asia/Seoul, after login)

Porcelain clay stills: albedo `[0.42, 0.428, 0.442]`, roughness 0.93,
exposure −0.20, charcoal studio, rear 3/4, 540×960, Cycles 24 spp,
Blender 4.2.3, nice 19. Sheet (viewed at full size before this
write-up): `output/_basemodels/eval/sketchfab_vs_demic.png`.

Four glTF zips were unpacked under gitignored
`output/_basemodels/sketchfab/<slug>/` and the Downloads zips were
removed after a successful unzip.

**Excluded, not scored:** *Gladiator*
(<https://sketchfab.com/3d-models/gladiator-0d09ee1dde8d419b83f8014292badc9f>)
by Alexander Blom (https://sketchfab.com/Exiistin), licence file CC-BY-4.0.
Dropped on instruction before evaluation because the Sketchfab page
carries a NoAI tag and the design is For Honor-inspired (game-IP /
copyright risk). A clay still had already been rendered when that
instruction arrived; the still, its metrics, and the unpacked folder
were deleted. It is not in the sheet and its numbers are not used
below. CC BY credit would not have cured the IP issue.

### 1. Diego Luján García — Male Full Body Ecorche

- Source: <https://sketchfab.com/3d-models/male-full-body-ecorche-ab11ebff89224f03bd75efede1164cf6>
- Licence: **CC-BY-4.0** (`license.txt` in the glTF zip). Commercial use
  allowed; author must be credited. Required credit line:

  > This work is based on "Male Full Body Ecorche"
  > (https://sketchfab.com/3d-models/male-full-body-ecorche-ab11ebff89224f03bd75efede1164cf6)
  > by Diego Luján García (https://sketchfab.com/diegoluga) licensed
  > under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

- File: `output/_basemodels/sketchfab/male_full_body_ecorche/scene.gltf`
  (16 meshes from ZBrush `SubTool-0` … `SubTool-15`; 28 packed
  images: 14 painted red-fiber baseColors + 14 normal maps).
- Import height: 68.1605 units. The mm/cm heuristic treated that as
  millimetres → **0.068 m**. 68.16 inches would be 1.73 m, which
  matches a “standard athletic man”; the 1.75 m normalized weld below
  is the one that counts.
- Pose (from the clay still): standing, arms hanging at the sides,
  not a T-pose or A-pose. No plinth. No clothes or baked straps in
  the geometry. Muscle form is in the sculpture — lats, glutes,
  erectors, calves and forearm extensors read in porcelain clay
  without the painted maps.
- Armature: none (0 skins, 0 bones, 0 vertex groups, 0 shape keys).
  Custom split normals on all 16 meshes (baked shading). Rigify /
  auto-weights are not feasible on 284 stacked muscle solids; that
  would need a retopologized single skin first, then a bind from this
  arms-down rest pose (worse than T/A for armpits). That is modeling
  work, not a download.

Measured (all 16 meshes):

| | |
|---|---|
| Mesh objects | 16 |
| Verts / tris | 348,656 / 640,470 |
| Boundary edges | 51,896 |
| Loose parts (sum) | 3,928 (largest piece, Object_29, already 2,211 islands and 0 UVs) |
| UV islands | **1,692** (14 of 16 meshes have UVs; islands follow painted muscle bellies — same failure class as Grab3D’s 834) |
| Armature | none |
| After 0.1 mm weld at heuristic 6.8 cm | 1 part, 33,876 boundary, 29,654 non-manifold (over-weld artifact; **do not use**) |
| After 0.1 mm weld at **1.75 m** | **284 loose parts**, 58 boundary, 21 non-manifold |

The 1.75 m weld is the honest test: the body stays 284 separate
shells, most of them closed volumes. It *looks* like one écorché
because the muscle solids nest visually. It is not a continuous
watertight clay skin.

**Fails:** watertight skin, clean UVs, no baked shading (custom
normals + normal maps), riggable / bind pose.
**Keeps as:** the best CC BY sculpt *reference* of either round.
Do not weld-patch the 16 subtools.

**Recommendation: reject as a production base.** Not “adopt with a
small fix.”

### 2. Beatriz Gomez Santamaria — Ecorche - Anatomy study

- Source: <https://sketchfab.com/3d-models/ecorche-anatomy-study-e402d3d541eb4b199c57d5410f5d3c57>
- Licence: **CC-BY-4.0**. Required credit line:

  > This work is based on "Ecorche - Anatomy study"
  > (https://sketchfab.com/3d-models/ecorche-anatomy-study-e402d3d541eb4b199c57d5410f5d3c57)
  > by Beatriz Gomez Santamaria (https://sketchfab.com/BeatrizGomez)
  > licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

- File: `output/_basemodels/sketchfab/ecorche_anatomy_study/scene.gltf`
  (6 meshes from 3 ZBrush subtools; no texture images; vertex colour
  `Color` on every mesh). Three meshes sit on the 65,532-vertex glTF
  / unsigned-16 index split.
- Import height: **0.0269 m** (2.7 cm). Normalized ×65.03 to 1.75 m
  for the weld test.
- Pose: standing study, arms down, no plinth, no clothes. The clay
  still reads as a thinner tendon-and-bone écorché — more gaps at
  hip, shoulder and knee than Diego or demic. Muscle form exists in
  the geometry but it is a medical study, not a gym mannequin.
- Armature: none. Custom normals and vertex colours on all 6 (painted
  shading). No UVs.

Measured (all 6 meshes):

| | |
|---|---|
| Mesh objects | 6 |
| Verts / tris | 282,894 / 418,978 |
| Boundary edges | 133,280 |
| Loose parts (sum) | 13,786 |
| UV islands | 0 (no UVs) |
| Armature | none |
| After 0.1 mm weld at imported 2.7 cm | 402 parts, 2,650 boundary, 16 non-manifold (too aggressive on a miniature) |
| After 0.1 mm weld at **1.75 m** | **8,890 loose parts**, 95,489 boundary, 11 non-manifold |

**Fails:** watertight skin, complete clay silhouette, UVs, riggable.
**Keeps as:** optional CC BY anatomy-study reference, weaker than Diego.
**Recommendation: reject.**

### 3. blakeallen25 — Gridiron Jumpman (American football player)

- Source: <https://sketchfab.com/3d-models/gridiron-jumpman-an-american-football-player-ebf3acb308a8497482642e5aa08f0312>
- Licence: **CC-BY-4.0**. The zip’s `license.txt` names the author as
  **blakeallen25** (https://sketchfab.com/blakeallen25), not the
  “Stuffed Dog Digital” label from the download note. Required credit
  line:

  > This work is based on "Gridiron Jumpman - An American Football Player"
  > (https://sketchfab.com/3d-models/gridiron-jumpman-an-american-football-player-ebf3acb308a8497482642e5aa08f0312)
  > by blakeallen25 (https://sketchfab.com/blakeallen25) licensed
  > under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

- File: `output/_basemodels/sketchfab/gridiron_jumpman/scene.gltf`
  (32 meshes: helmet, shoulder-pad `covers`, shirt, undershirt,
  pants, shoes, socks, arms, hands, legs, head, eyes, eyelashes,
  towel, tape, football/`Sphere`, loft glass). Several pieces sit on
  the 65,532-vertex split.
- Import height: **1.075 m** (already metres; jump-pose bounding box,
  not standing height). A 1.75 m normalize leaves the weld almost
  unchanged.
- Pose: mid-air jump, ball in one hand, other arm out. Not a bind
  pose. No plinth. Helmet, pads, jersey, pants and cleats hide almost
  all muscle; the jersey reads as a smooth stylized shell, not
  sculpted fibers. Visible arms are stylized, not an écorché.
- Armature: none (Sketchfab export has 0 skins). Custom normals on
  all 32. Shirt/sock/towel also carry normal maps. Rigify on a
  jumping, multi-layer clothed mesh is not a realistic path; stripping
  the kit to chase a hidden body would be another round of patching.

Measured (all 32 meshes):

| | |
|---|---|
| Mesh objects | 32 |
| Verts / tris | 840,514 / 1,237,224 |
| Boundary edges | 344,902 |
| Loose parts (sum) | 61,587 |
| UV islands | **61,586** |
| Armature | none |
| After 0.1 mm weld at 1.075 m | 454 parts, 32,562 boundary, 14,737 non-manifold |
| After 0.1 mm weld at ~1.75 m | 455 parts, 32,604 boundary, 14,737 non-manifold |

**Fails:** exposed muscle form, watertight skin, bind pose, clean UVs,
riggable.
**Keeps as:** nothing for a clay base. The football-player *concept*
can still be authored later as an original character; this asset is
not that body.
**Recommendation: reject.**

## Recommendation

**Adopt none of the measured bodies** — not the first-round three, and
not Diego, Beatriz, or Gridiron Jumpman. A muscle atlas, a
Renaissance statuette, a 16-subtool écorché kit, a fragmented anatomy
study, and a clothed jumping football player are not a demic-style gym
mannequin. Adopting any of them would be another round of patching.

Diego is the closest *look* we have now (real geometric muscle, no
clothes, standing, no plinth) and still fails the skin test that
killed Grab3D. Do not weld the 16 subtools together and call it a
base.

What would actually have a chance, in order:

1. **Vet further original concept-character downloads** the same way
   (weld / UV / strap / clothes / bind-pose / licence, porcelain still,
   full-size sheet vs demic). CharacterZone’s muscular base
   (`0954aa04…`, CC BY 4.0, claims “ready for rig”) is still
   unmeasured if it can be fetched legally. Do not start production
   on any new file until those checks pass.
2. If those also fail the watertight-skin test, **there is no free
   drop-in.** The method that works by construction is to author our
   own continuous, welded, unstrapped clay body (CC0), using Diego
   and/or the Małopolska sculpt as *reference only* (credit Diego
   under CC BY if we derive from that mesh; looking is not deriving),
   bind it once, and keep per-muscle glow on attributes or on our own
   overlay shells. That is modeling work, not a download, and it
   should be scoped honestly before anyone starts it.
3. A paid athletic écorché would also need 중현 님’s explicit approval
   (standing rule: no paid assets).

Do not go back to Grab3D char1. Do not ship Z-Anatomy or Charleeh
meshes. Do not treat BodyParts3D’s missing latissimus as a small fix.
Do not use the excluded Gladiator (NoAI + For Honor-inspired).
