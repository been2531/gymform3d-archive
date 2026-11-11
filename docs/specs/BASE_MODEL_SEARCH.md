# Base-model search — replacement for Grab3D char1

Access date: 2026-11-11 (Asia/Seoul). Grab3D Human Muscles (`char1`) is a
declared failure (834 UV islands, baked ankle retinaculum, baked
`rig_normals`). This search evaluates legally reusable replacements
*before* adopting one. No Grab3D patching.

**Verdict: do not adopt a production clay body yet.** First-round
bodies, Diego, Beatriz, and Gridiron Jumpman stay rejected. The
deepkexz mannequin remains the only mesh that survives the weld test
at **1 loose part, 0 boundary, 0 non-manifold**. minglanders1’s
*Male Mesh - Bodybuilder* is the first **rigged** candidate (25 bones,
walk animation cleared to rest): weld is **1 part, 33 boundary, 20
non-manifold**, no textures, faceless, real muscle form, sculpted
shorts. That is progress, not a green light. *Gladiator* stays
excluded (For Honor-inspired + NoAI).

The v1 3D blob was lopsided because it mirrored anchors across a
camera-derived axis, so a rear 3/4 view painted only the near flank.
v2 mirrors across an anatomic midline (shoulder-band PCA) and blends
overlapping colours instead of winner-takes-most. Both lats and a
centred pink mid-back now read on the mannequin and the bodybuilder.
It is still softer and less saturated than demic; lower lats meet as
a lumbar belt; rear delts and biceps stay weak on the blob. Of the
simpler alts tested on the bodybuilder, rig-weight tint is the only
one that competes (better delts and a visible biceps, weaker lat
fan). Coloured lights blow out; painted strokes are sparse; fresnel
rim is not distinct from the blob.

A later hybrid pass (V-fan blob lats + diamond mid-back + rig-weight
rear delts/biceps) removed the orange lumbar belt. The lat colour now
reads as a V from the armpit down the flank, with a gap above the
shorts; the pink mid-back is still blockier than a true diamond and
the V is thinner than demic. The same bodybuilder was hand-keyed into
a textbook barbell bent-over row (hip hinge ~40° from vertical, knees
22° soft, peak elbows 97°, feet planted). That still is the closest
3D match yet to the demic elevated-back panel. A first IK attempt at
the bench-supported one-arm DB row put the torso near horizontal but
missed pad contacts (hand/knee ~10 cm above the pad) and did not pull
the working elbow to the hip at the peak — do not treat those stills
as pose proof.

Comparison sheets (gitignored stills, viewed at full size before this
write-up):

- First round: `output/_basemodels/eval/basemodels_vs_demic.png`
- Sketchfab clay round: `output/_basemodels/eval/sketchfab_vs_demic.png`
- Precise-mask colour test (superseded): `output/_basemodels/eval/muscle_colour_test.png`
- Blob glow round: `output/_basemodels/eval/blob_glow_test.png`
- Glow methods on the mannequin: `output/_basemodels/eval/blob_glow_methods.png`
- Symmetric blob v2: `output/_basemodels/eval/blob_glow_v2.png`
- Simpler alts on the bodybuilder: `output/_basemodels/eval/blob_glow_v2_alts.png`
- Hybrid colour + barbell row: `output/_basemodels/eval/hybrid_barbell_row_v3.png`

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

# smooth 3D blobs (and --method idpass|shells|xray on one body)
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  --python tools/rig3d/eval_blob_glow.py -- \
  --input <gltf> --out-dir output/_basemodels/eval --name <id> \
  --method blob
python3 tools/rig3d/build_base_model_sheet.py \
  --out output/_basemodels/eval/blob_glow_test.png \
  --title "Blob glow test  |  matte grey + smooth 3D muscle blobs, elevated rear 3/4" ...
python3 tools/rig3d/build_base_model_sheet.py \
  --out output/_basemodels/eval/blob_glow_methods.png --no-demic ...

# symmetric blob v2 + simpler alts (blob|painted|rim|lights|rigweight)
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  --python tools/rig3d/eval_blob_glow.py -- \
  --input <gltf> --out-dir output/_basemodels/eval --name <id>_v2_rear34 \
  --method blob --yaw 135 --elev 14
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  --python tools/rig3d/eval_blob_glow.py -- \
  --input <gltf> --out-dir output/_basemodels/eval --name <id>_v2_back \
  --method blob --yaw 180 --elev 8

# hybrid colour + barbell bent-over row (and optional db_row attempt)
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  --python tools/rig3d/eval_hybrid_row.py -- \
  --input output/_basemodels/sketchfab/male_mesh_bodybuilder/scene.gltf \
  --out-dir output/_basemodels/eval --name bodybuilder_v3 \
  --shots standing,bb_top,bb_bottom
```

Sketchfab downloads for the blob round used
`GET https://api.sketchfab.com/v3/models/<uid>/download` with a token
kept in `~/.sketchfab/env` (mode 600; never committed or printed) and
the temporary `gltf.url` from that response. API zips were extracted
under `output/_basemodels/sketchfab/<slug>/` and then deleted.

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

## Sketchfab blob-glow round (measured 2026-11-11, Asia/Seoul)

The bar from this round is colour readability, not a demic look-alike.
Sports concept characters are in scope. Target colours (emission over
matte grey, body V ≈ 80–110, charcoal backdrop, elevated rear ¾,
540×960, Cycles 24 spp): peach-orange lats, pink rhomboid/mid-traps,
purple rear delt, yellow biceps. Blobs are large smooth radial
falloffs from a few anatomical anchors (`1 − smoothstep(0.4R, R,
dist)`, winner-takes-most into vertex colour `mcol`), not
boundary-precise masks. Grey albedo ≈ `[0.22, 0.224, 0.228]`,
exposure −0.15 with a grey-V auto-tune (except X-ray). Camera yaw 135,
elev 14. Script: `tools/rig3d/eval_blob_glow.py`.

Three further CC BY models were pulled via the Sketchfab download API
(UIDs `cd7974ef71b542cc99f93bacea1ecc6c`,
`163b786c3c65481a822c66eb732b4e4e`,
`e6281bc9c81c4ce59fc53da74b8bcb8f`) and Diego was re-rendered with
the same blob method. Sheet (viewed at full size):
`output/_basemodels/eval/blob_glow_test.png`.

### 4. deepkexz — Muscular human mannequin figure

- Source: <https://sketchfab.com/3d-models/muscular-human-mannequin-figure-cd7974ef71b542cc99f93bacea1ecc6c>
- Licence: **CC-BY-4.0**. The listing at download time described it as
  AI-made (Rodin). Required credit line:

  > This work is based on "Muscular human mannequin figure"
  > (https://sketchfab.com/3d-models/muscular-human-mannequin-figure-cd7974ef71b542cc99f93bacea1ecc6c)
  > by deepkexz (https://sketchfab.com/deepkexz) licensed under
  > CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

- File: `output/_basemodels/sketchfab/muscular_human_mannequin/scene.gltf`
  (10 meshes on the 65,532-vertex glTF split — chunks of one body, not
  separate muscles; no texture images; no UVs).
- Import height: **1.8985 m**. Normalized ×0.922 to 1.75 m for the
  weld and the still.
- Pose (from the blob still): standing, arms hanging slightly off the
  sides, not a T-pose. Sculpted boxer-briefs are part of the mesh
  (there is no texture to hide). Muscle form is a smooth stylized
  clay athlete — lats, delts and glutes read, without écorché fiber
  detail.
- Armature: none. Custom normals on all 10. No vertex groups, no
  shape keys.

Measured (all 10 meshes):

| | |
|---|---|
| Mesh objects | 10 |
| Verts / tris | 609,466 / 1,000,000 |
| Boundary edges (unwelded sum) | 212,966 (split seams) |
| Loose parts (unwelded sum) | 15,229 |
| UV islands | 0 (no UVs) |
| Armature | none |
| After 0.1 mm weld at **1.75 m** | **1 loose part, 0 boundary, 0 non-manifold**, 499,996 verts / 999,988 tris |

This is the first measured candidate that passes the watertight-skin
test. The 10 objects are exporter chunks; after a join they are one
closed surface.

Blob (viewed at full size): peach lat wraps the visible flank, pink
sits on the mid-upper back, purple on the near rear delt, yellow on
the near upper arm. All four colours read. Paint counts: lats 108,625
/ mid-back 28,865 / rear delt 6,153 / biceps 8,629. Grey V p50 = 82
(in range).

**Fails:** bind-ready armature, no clothing (sculpted briefs), clean
recomputable normals, tractable triangle count, non-AI provenance,
proven row pose.
**Keeps as:** the only weld-pass skin so far, and the body used for
the A/B/C method test.
**Recommendation: do not adopt yet.** The honest next experiment is
join + weld + decimate (to roughly 80–150 k tris) + clear custom
normals + Rigify auto-weights from this standing pose, then a
one-arm DB-row still. That is about half a day to a day, not a
download-and-ship. Parenting without weights will not pose it.

### 5. Tulio Portela — Manequin Dummy v0 Basketball Free

- Source: <https://sketchfab.com/3d-models/manequin-dummy-v0-basketball-free-163b786c3c65481a822c66eb732b4e4e>
- Licence: **CC-BY-4.0**. Required credit line:

  > This work is based on "Manequin Dummy v0 Basketball Free"
  > (https://sketchfab.com/3d-models/manequin-dummy-v0-basketball-free-163b786c3c65481a822c66eb732b4e4e)
  > by Tulio Portela (https://sketchfab.com/TulioPortela) licensed
  > under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

- File: `output/_basemodels/sketchfab/manequin_dummy_basketball/scene.gltf`
  (3 meshes; Object_4 / Object_5 are 65,532-vertex body splits;
  Object_6 is 4,646 tris — the basketball. 9 packed images:
  baseColor / metallicRoughness / normal).
- Import height: **2.00 m** (jump-pose bounding box). Normalized
  ×0.875 to 1.75 m. After normalize, bbox width is 1.33 m (arms and
  ball out).
- Pose: mid-air jump shot, ball overhead, other arm out. Not a bind
  pose. Jersey, shorts and shoes. The body is a smooth dummy, not a
  muscular sculpt.
- Armature: none. Custom normals on all 3.

Measured (all 3 meshes):

| | |
|---|---|
| Mesh objects | 3 |
| Verts / tris | 135,666 / 235,198 |
| UV islands | 1,061 before weld / **295 after** |
| Armature | none |
| After 0.1 mm weld at **1.75 m** | **1 loose part, 6 boundary, 2 non-manifold** |

Blob (viewed at full size): peach lat and pink upper back read on the
jersey. Rear delt and biceps do not read — the shooting arm is up and
only 242 verts painted as biceps. Paint: lats 15,357 / mid-back 6,407
/ rear delt 1,412 / biceps 242. Grey V p50 = 86.

**Fails:** bind pose, exposed muscle form, clean UVs, riggable without
a large retarget from a jump.
**Keeps as:** proof that colour-on-clothes can mark a lat and an
upper back on a sports dummy.
**Recommendation: reject as a base.** Same pose class as Gridiron.

### 6. restore50 — Shirtless Athlete in Black Shorts

- Source: <https://sketchfab.com/3d-models/shirtless-athlete-in-black-shorts-e6281bc9c81c4ce59fc53da74b8bcb8f>
- Licence: **CC-BY-4.0**. The listing at download time described it as
  AI-made (Meshy). Required credit line:

  > This work is based on "Shirtless Athlete in Black Shorts"
  > (https://sketchfab.com/3d-models/shirtless-athlete-in-black-shorts-e6281bc9c81c4ce59fc53da74b8bcb8f)
  > by restore50 (https://sketchfab.com/restore50) licensed under
  > CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

- File: `output/_basemodels/sketchfab/shirtless_athlete_black_shorts/scene.gltf`
  (2 meshes on the 65k split; 8 packed 2048² images: emissive,
  baseColor, metallicRoughness, normal on each).
- Import height: **1.899 m**. Normalized ×0.922 to 1.75 m. Bbox width
  0.63 m (arms nearer the sides than the mannequin).
- Pose: standing, hands near the hips. Shirtless with sculpted black
  shorts — the closest *concept* to demic of this round. Muscle form
  is present but smoother than Diego (Meshy clay, not fiber).
- Armature: none. Custom normals on both meshes.

Measured (all 2 meshes):

| | |
|---|---|
| Mesh objects | 2 |
| Verts / tris | 104,134 / 171,578 |
| UV islands | 1,244 before weld / **463 after** |
| Armature | none |
| After 0.1 mm weld at **1.75 m** | **1 loose part, 6 boundary, 3 non-manifold** |

Blob (viewed at full size): peach lats and pink mid-back read; purple
rear delt and yellow biceps are small. The lower lat blob spilled
onto the near hip and hand (hands sit at the waist, inside the large
lat radius). Paint: lats 23,066 / mid-back 4,724 / rear delt 341 /
biceps 595. Grey V p50 = 83 after one exposure retune.

**Fails:** leftover orifice-scale boundary / 3 non-manifold edges,
baked shading (custom normals + normal/emissive maps), 463 UV
islands, no armature, AI provenance, slightly dirty blob spill.
**Keeps as:** lighter standing runner-up if the mannequin’s triangle
count or briefs become a problem.
**Recommendation: do not adopt yet.** Same class of next experiment
as the mannequin (join + weld + Rigify), with more leftover topology
and baked maps to clear.

Diego’s blob still (re-rendered this round, grey V p50 = 81): all
four colours read on the écorché — peach lat, pink mid-back, purple
rear delt, yellow upper arm — with soft edges. That does not change
the 284-shell weld failure. He remains the best sculpt *reference*,
not a skin.

### Glow methods on the mannequin

One still each, same camera, viewed at full size on
`output/_basemodels/eval/blob_glow_methods.png`.

| Method | What was tested | What it reads as |
|---|---|---|
| 3D blob (baseline) | Vertex-colour radial falloff, winner-takes-most, emission over grey | Four distinct colours; lat wraps the flank |
| (A) 2D compositing | Grey beauty + projected soft elliptical masks clipped to the silhouette | Softest Day-4 grading bloom; peach spills onto the briefs; ellipses do not wrap a flank |
| (B) Overlay shells | Keep weight > 0.32, displace 2.5 mm along normals, only shells emit (26 shells) | Ragged hard cutouts and a hole in the lower lat; not shippable |
| (C) Glass X-ray | Principled glass skin (transmission 0.88, alpha 0.22) + 11 inner emission ellipsoids | Glass mannequin with balloon organs; over-bright primitives blow toward yellow |

**Clearest for muscle identity: the 3D blob.** The four targets stay
separable and follow the surface.
**Softest still / closest to old Day-4 grading: (A) 2D compositing.**
Useful as an optional grade on top of a grey beauty pass, not as the
base method — it needs per-frame projection of the same 3D anchors,
and screen-space ellipses foreshorten incorrectly.
**Most reusable: the 3D blob.** Vertex attributes travel with the
mesh through posing and any camera. (A) is camera-locked. (B) would
be reusable only with hand-sculpted or smoothed shells, not this
extract. (C)’s cheap inner spheres are not muscle volumes.

Do not spend further time polishing automatic shells or X-ray
ellipsoids unless someone asks for that look specifically.


### 7. minglanders1 — Male Mesh - Bodybuilder

- Source: <https://sketchfab.com/3d-models/male-mesh-bodybuilder-a2f2a1642e074abd8acaf76b1f521aba>
- Licence: **CC-BY-4.0** (`license.txt` in the glTF zip). Commercial use
  allowed; author must be credited. Required credit line:

  > This work is based on "Male Mesh - Bodybuilder"
  > (https://sketchfab.com/3d-models/male-mesh-bodybuilder-a2f2a1642e074abd8acaf76b1f521aba)
  > by minglanders1 (https://sketchfab.com/minglanders1) licensed
  > under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

- File: `output/_basemodels/sketchfab/male_mesh_bodybuilder/scene.gltf`
  (3 body chunks on the 65 k glTF split + a detached 80-tri `Icosphere`
  helper that was hidden and excluded from measure and paint). No
  packed images, no textures (`textureCount` 0). One principled
  material, roughness 0.6, metallic 0.
- Listing: “Mesh and rig and walk animation - no face feature.”
  Tags include `rigged` and `animated`. Not listed as AI-made.
- Import: Y-up, native height **1.70 m**. After rotating roots so Y→Z
  and flipping 180° around X (the first rotation left the head below
  the feet — detected from the `Head_021` vs foot bones), normalized
  ×1.029 to 1.75 m. Rest-pose bbox ≈ 0.92 × 0.35 × 1.75 m (standing,
  arms hanging, not a T-pose).
- Pose: walk action `Armature|Armature|walking_man|baselayer` is
  cleared to the armature rest pose before measure and paint.
- Armature: **25 bones** (`Hips_00` … `Head_021`, Mixamo-like
  `LeftArm` / `LeftForeArm` / `LeftShoulder` names). All three body
  meshes carry the 25 vertex groups. Custom split normals on all
  three. No shape keys.
- Face: smooth bald head, no features. Confirmed in the stills.
- Clothing: sculpted shorts (geometry, not a baked texture).

Measured (3 body meshes, Icosphere excluded, 1.75 m):

| | |
|---|---|
| Mesh objects | 3 (65 k exporter splits of one skin) |
| Verts / tris | 144,330 / 243,383 |
| After 0.1 mm weld at **1.75 m** | **1 loose part, 33 boundary, 20 non-manifold**, 121,686 verts / 243,383 tris |
| UV islands | 2,099 before weld / **1 after** |
| Armature | 25 bones, 1 walk animation (cleared) |
| Source textures | 0 |

The still (viewed at full size after the head-flip fix) shows a
faceless muscular figure with real lat, teres and rear-delt form in
the sculpture — closer to demic’s clay read than the mannequin.
Blob v2: peach lats on both sides and a centred pink mid-back read
in rear 3/4 and straight back. Rear delts and biceps are weak on
the blob. Grey V p50 = 80 (rear 3/4) and 88 (back), both in range.
Paint: lats 13,890 / mid-back 6,250 / rear delt 2,159 / biceps 1,197.

A best-effort FK one-arm row (`Hips_00` +30°, `Spine_011` +12°,
`RightArm_017` / `RightForeArm_018` guesses) produced a falling lean,
not a textbook dumbbell row. That still is not evidence the rig can
row; it only proves the bones move.

**Fails:** watertight-enough skin (20 non-manifold, 33 boundary), no
clothing, proven row pose, recomputable normals.
**Keeps as:** the first rigged, faceless, untextured, muscular
candidate. Stronger sculpt than the mannequin; dirtier topology.
**Recommendation: do not adopt yet.** The honest next experiment is a
hand-keyed one-arm DB row on this 25-bone rig (not the failed FK
guess), plus a decision on the shorts and the 20 non-manifold edges.

## Symmetric blob v2 (viewed at full size)

Root cause of the v1 “split this way and that” look: `estimate_torso_frame`
set `back` from the camera direction, so mirroring across `side` was a
camera flip, not a left-right flip. A rear 3/4 view therefore painted
only the near flank.

v2 (`tools/rig3d/blob_v2.py`) builds left-right from the shoulder-band
PCA, orients `back` toward the camera afterwards, recentres the hip on
that midline, and paints large smooth spheres along each muscle (the
v1 radii that actually reach this mannequin’s skin) with soft
overlapping blend instead of winner-takes-most. Hands/forearms (low
and far lateral) and the hip band are gated out. Colours are warmer
and more saturated than v1; emission 0.55; a slight chroma bloom is
applied after the still.

Sheet: `output/_basemodels/eval/blob_glow_v2.png`.

What improved: both lats light up; the pink mid-back sits on the
midline; the mannequin’s straight-back view is the cleanest symmetry
proof. Grey V stays in the 80–110 window (mannequin 81 / 90,
bodybuilder 80 / 88).

What is still short of demic (coldly):

- Colour is a soft cloud, not a sharp muscle silhouette. Saturation
  and edge discipline are below the demic clay.
- Lower lats meet across the lumbar as an orange belt and still kiss
  the waistband of the briefs/shorts.
- Rear delts are clear on the mannequin’s back view and weak on the
  bodybuilder blob. Yellow biceps almost never read from the rear.
- The bodybuilder figure had to be flipped 180° after the first
  Y→Z rotation (head was below the feet). Any later pose work must
  keep that correction.

Production colour method remains the **3D blob**, now the symmetric
v2 version.

## Simpler alternatives (same bodybuilder, same rear 3/4)

Sheet: `output/_basemodels/eval/blob_glow_v2_alts.png`. Judged against
the symmetric blob and against demic.

| Method | What it is | What it reads as |
|---|---|---|
| Blob (baseline) | Anatomic-mirrored spheres, soft blend, emission over grey | Clearest lat + mid-back identity |
| Painted strokes | Thinner mirrored polylines, one-time vertex paint | Near rear delt pops; lat fan is only a lower patch |
| Fresnel rim | Same blob mask; fresnel rim + light interior tint | A slightly softer copy of the blob, not a new language |
| Coloured lights | 14 spots, no mesh edit, white key dimmed | Best purple delts; peach+pink blow to a yellow-white core |
| Rig-weight tint | Shoulder → purple, upper arm → yellow, spine × gates → lat/mid | Best simpler alt: both delts + a visible biceps; lat wings stay thin |

**Winner overall vs the symmetric blob: the blob still wins** for
muscle identity and for a filled lat fan. **Best simpler alt:
rig-weight tint** — zero hand-placed anchors, follows any pose, and
is the only alt that makes rear delts and biceps obvious. It does
not replace the blob for lats and mid-back. Coloured lights are the
prettiest accident and the worst teacher. Do not spend more time on
the fresnel rim unless someone wants that specific look.


## Hybrid colour + barbell bent-over row v3 (viewed at full size)

Sheet: `output/_basemodels/eval/hybrid_barbell_row_v3.png`
(demic + standing back + barbell top + barbell bottom). Scripts:
`tools/rig3d/blob_v2.py` (`paint_hybrid`), `tools/rig3d/bodybuilder_pose.py`,
`tools/rig3d/eval_hybrid_row.py`. Reuses the procedural Olympic barbell
(`tools/rig3d/props/barbell_olympic.py`, `clear=False`) and the flat bench
(`tools/rig3d/props/bench_flat.py`).

Hybrid colour, painted in rest pose so it follows the rig:

- Lats: tapered capsules along a V-shaped fan (wide at the armpit,
  narrowing down the flank). A t-dependent lateral gate keeps the
  lumbar midline dark, and colour fades out ~8–10 cm above the shorts
  waistband (`smoothstep(0.32, 0.40, t)`). No orange belt.
- Mid-back: a soft diamond mask centred between the scapulae (still
  reads as a rounded rectangle more than a pointed diamond).
- Rear delts + biceps (+ a little posterior upper-arm wrap): smoothed
  rig weights, the v2 alt that actually made those muscles obvious.
- Colours more saturated than v2; emission 0.82.

Paint counts on the bodybuilder: lats 8,398 / mid-back 3,319 /
rear delt 44,648 / biceps 27,776. Grey V p50 = 98 (standing).

Barbell bent-over row (Mixamo-like 25-bone IK: feet planted, Hips
rotated about the character’s right so the head travels forward,
hands IK’d to the bar). QC against a textbook hinge, not the
bench-row spec:

| | top | bottom |
|---|---|---|
| torso from vertical | 39.7° | 39.7° |
| torso from horizontal | 50.3° | 50.3° |
| head–spine | 14.1° | 14.1° |
| working elbows | 97° / 97° | ~180° / ~180° |
| knee flexion | 22° | 22° |
| feet on floor | yes (ankle z ≈ 0.11 m) | yes |
| grip half / shoulder half | 0.29 / 0.25 m | same |
| bar | lower ribs / upper abdomen | below the knees (z ≈ 0.44 m) |

Camera: elevated back, 30° elev / 18° side, framed on the body (not
the 2.2 m bar) so the still can sit next to the demic panel.

What is still short of demic (coldly):

- The glow is still separate stickers, not a painted-on muscle
  silhouette. The pink mid-back is blocky; dark gutters sit between
  it and the lat fans. Saturation and edge discipline stay below the
  demic clay.
- The lat V is real now but thinner and shorter than demic’s upper-back
  glow.
- Torso is ~40° from vertical (a touch more upright than a 45° hinge).
- Hands approximate the bar; there is no finger wrap. The near hand at
  the bottom is an imperfect contact.
- Sculpted shorts, 33 boundary / 20 non-manifold edges, and some hip
  stiffness on this 25-bone rig remain.

Bench-supported one-arm DB row (same script, `pose_bench_db_row` via
`bench_row_pose.build_landmarks`): the torso did reach ~3° from
horizontal, and the procedural bench is in the scene, but the attempt
fails the spec. Support hand and knee sat ~10 cm above the pad
(`lwr_z` 0.53 / `lkn_z` 0.54 vs pad 0.43); the standing foot was not
flat (`rank_z` 0.18); the working elbow stayed ~180° at the advertised
peak (hand–hip 0.53 m, gate ≤ 0.34 m); the support shin kicked up.
Stills `bodybuilder_v3_db_top.png` / `bodybuilder_v3_db_bottom.png`
exist for the record and are not on the comparison sheet. Do not treat
them as pose proof.

## Recommendation

**Adopt none of the measured bodies yet** — not the first-round three,
not Diego, Beatriz, or Gridiron Jumpman, not the three earlier API
downloads as a shipped skin, and not the bodybuilder yet either.

The deepkexz mannequin is still the only body that *survives the weld
test* (1 / 0 / 0). The minglanders1 bodybuilder is the more complete
*candidate*: rigged, faceless, untextured, standing rest pose, real
muscle form, 243 k tris. It fails the weld on 20 non-manifold edges
and 33 boundary edges, wears sculpted shorts, and has no proven row.
Do not ship either one as the base tonight.

For muscle colour, use the **symmetric 3D blob (v2)** going forward.
The best simpler fallback on a rigged body is **rig-weight tint**.
Do not use coloured lights, the fresnel rim, cheap overlay shells,
or glass X-ray as the production method.

What would actually have a chance, in order:

1. **Tighten the hybrid glow** (connect the pink diamond to the lat
   fans, sharpen it into a real diamond, push saturation toward
   demic) and **nudge the barbell hinge** a few degrees deeper so
   the back faces the camera more like the demic panel. The barbell
   row on this rig is now a real candidate motion, not a guess.
2. **Re-do the bench-supported one-arm DB row** with contacts that
   work by construction (project the support hand/knee onto the pad,
   then IK). The first landmark pass got the torso flat and put a
   bench in the scene, then missed the pad by ~10 cm and never
   pulled the working arm. Do not ship that attempt.
3. **Rigify-test the mannequin** only if the bodybuilder row fails
   or the shorts become a hard no. It is still the cleaner skin and
   still a day of bind work.
4. If both binds fail, **there is still no free drop-in.** The
   method that works by construction is to author our own continuous,
   welded, unstrapped clay body (CC0), using Diego and/or the
   Małopolska sculpt as *reference only*, bind it once, and keep
   per-muscle glow on vertex attributes (the v2 blob). That is
   modeling work, not a download.
5. A paid athletic écorché would also need 중현 님’s explicit approval
   (standing rule: no paid assets).

Do not go back to Grab3D char1. Do not ship Z-Anatomy or Charleeh
meshes. Do not treat BodyParts3D’s missing latissimus as a small fix.
Do not use the excluded Gladiator (NoAI + For Honor-inspired).

## Boxing bodies (Tishkok + Quaternius jab test)

Access date: 2026-11-11 (Asia/Seoul). After the domain pivot to boxing
(t839u), two shortlisted bodies were downloaded and given the same jab.

Sheet (gitignored, viewed at full size before this write-up):
`output/_basemodels/eval/boxer_jab_test.png`.

Replay:

```
nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \
  --python tools/rig3d/eval_boxer_jab.py -- \
  --out-dir output/_basemodels/eval --name boxer_jab
python3 tools/rig3d/build_boxer_jab_sheet.py \
  --out output/_basemodels/eval/boxer_jab_test.png \
  --dir output/_basemodels/eval --name boxer_jab \
  --ref output/_basemodels/eval/ref_jab_usarmy.jpg \
  --tishkok-rest output/_basemodels/eval/tishkok_rest_y40_rear34.png \
  --facts output/_basemodels/eval/boxer_jab_facts.json
```

Motion: Quaternius Universal Animation Library **Standard**
`Punch_Jab_Rig` (CC0, 21 frames, downloaded from OpenGameArt
`universal_animation_librarystandard.zip`). It is in the free Standard
set, not only the paid Pro zip. The clip is a left-hand jab
(hand-travel L 0.24 m / R 0.06 m; frames guard/mid/full = 3 / 4 / 5).
Hands stay open — it is closer to a martial-arts palm strike than a
boxing jab with a fist. ACCAD Male2 `JabRight.bvh` (CC BY 3.0) was
downloaded as a fallback and not used. Mixamo was not used.

### 1. Tishkok — Stylized Boxer Base Mesh Character

- Source: Sketchfab uid `537003fb4788494fbb48c66412eb5dda`
- Licence: **CC BY 4.0**. Commercial use allowed; author must be credited:
  Tishkok, “Stylized Boxer Base Mesh Character”.
- Unrigged (0 bones, no skins). Rest pose is an **A-pose**, height 1.79 m.
- 57,905 verts / 83,048 tris. After 0.1 mm weld: **70 parts, 3,106
  boundary, 66 non-manifold**. The extras are sculpted boxing gloves,
  shorts and shoes (separate shells), not a single continuous skin.
- Has a simplified face (not faceless). Lean boxer proportions, real
  muscle form in the clay. That rest-pose still
  (`tishkok_rest_y40_rear34.png`) is the better-looking dedicated
  *boxer* body of the two.
- Auto-rig: vertex-group transfer from the T-pose Quaternius Superhero
  (65 groups copied) tore the mesh as soon as the jab posed the arms —
  the donor T-pose arms do not overlap this A-pose. Automatic weights
  onto the same T-pose armature would fail for the same reason. A
  Rigify metarig folded to this A-pose was not done this pass. Do not
  read the exploded `boxer_jab_tishkok_*.png` stills as the body.

### 2. Quaternius — Superhero Male (Universal Base Characters Standard)

- Source: itch.io Standard zip (122 MB, min price $0, CC0). The free
  Standard set is only Superhero male + female plus hairstyles — not
  Regular or Teen.
- Licence: **CC0 1.0**. Credit not required.
- Rigged: 65-bone UE-style humanoid (`pelvis`, `spine_01..03`,
  `upperarm_l/r`, finger bones…). Native weights.
- Body mesh only (eyes / eyebrows / helper icosphere hidden): after
  0.1 mm weld **1 part, 0 boundary, 0 non-manifold**, 6,285 verts /
  12,566 tris. Body height ~1.82 m.
- Has a stylized face. No gloves. Superhero proportions, not a boxer
  silhouette. Clay hide of the separate eye/brow meshes leaves empty
  sockets if you look closely; the body mesh itself still has a face.
- The same UAL jab retargeted with **rotation-only** `matrix_basis`
  copies (DEF-* → UE names). Copying locations as well stretched limbs
  to ~14 m; that was a retarget bug, not the clip.
- Glow (painted in rest, follows the rig) on the punching-side front
  delt (purple, 249 verts), triceps (yellow, 148), obliques (orange,
  251). Viewed at full size: the purple sits on the front shoulder and
  kisses the pec; the yellow triceps is faint; the orange reads as a
  hip stripe more than a clean external oblique.
- Against the US Army public-domain jab photo (Wikimedia,
  `Flickr_-_The_U.S._Army_-_All-Army_Boxing_jab.jpg`): real boxers
  keep fists and a tighter guard. This clip’s open palms and
  palm-forward guard are the bigger teaching miss, not the body.

### Verdict

**Adopt neither as the shipping boxing body tonight.**

Tishkok is the body that *looks like a boxer* and is legally usable
with credit, but it has no rig, fails the weld on clothing/glove
shells, and did not take the jab. Quaternius Superhero Male is the
only body that actually threw the jab (and it survives the weld), but
it is not a boxer character, the free Standard pack has no more
athletic male, and the UAL jab’s open hands are a problem next to a
real jab.

What would actually have a chance, in order:

1. **Fold a Rigify metarig to Tishkok’s A-pose**, bind with automatic
   weights, and retarget the same UAL jab by world-space bone aim
   (rest poses will not match). If that holds together, Tishkok plus
   credit is the stronger boxing silhouette.
2. Keep Quaternius Superhero Male as the **motion guinea pig** only —
   it already proves UAL Standard punches retarget onto the UE
   humanoid. Do not treat it as the show body without a glove/fist
   solution and a tighter jab clip.
3. Do not buy the Quaternius Source pack or UAL Pro just to get
   Regular/Teen bodies or extra punches (standing rule: no paid
   assets). The jab we needed was already in the free Standard UAL.


## Boxing bodies v2 (Tishkok A-pose Rigify + shortlist)

t845u: fit Rigify to Tishkok’s A-pose and search for a better free-commercial body.

### A — Tishkok bind

After a 0.1 mm weld the mesh is usable shells: body 30424 tris (plus two ears
joined in), shorts 5508, gloves 3882 each, shoes many islands joined per
side (~19676 tris). Landmark placement of the human metarig put
`upper_arm` heads on the delts and `hand` tails on the sculpted-glove
centroids (0 mm error). Automatic weights on body + shorts, rigid parent of
gloves → `hand.L/R` and shoes → `foot.L/R`.

Rigify *generate* works once the add-on is enabled with
`bpy.ops.preferences.addon_enable(module="rigify")` (plain
`addon_utils.enable` fails to register preferences in background mode, and
then generate dies on `make_custom_pivot`). The generated DEF bones are
constraint-driven (ORG/MCH), so keyframing them does nothing. Deformation
for this test uses the **fitted metarig** itself.

The rest still is a standing A-pose boxer: mesh intact, gloves on the
wrists, shoes on the feet, sculpted fists already closed. That bind is the
real progress versus v1.

Glow (em 1.30, punching-side only) at rest: purple front-delt is clearly
visible but still spills toward the pec as a band; yellow triceps was moved
onto the upper-arm bone (posterior — weak/unreadable from this 3/4 front);
orange “obliques” still sit too high (serratus/armpit, not the waist flank
between ribs and iliac crest).

### A — ACCAD JabRight retarget: stopped

`Male2_E2_JabRight.bvh` (CC BY 3.0, already on disk) does not become a
boxing jab on this rig with the retargets tried:

1. Rest-relative swing on the full skeleton tips the character onto its
   back. ACCAD’s bind OFFSETS are not an anatomical T-pose (spine bones
   point sideways in the file), so “rest → pose” on hips/spine is a huge
   stand-up swing applied to an already-standing metarig.
2. Arm-only world-aim keeps the body standing and the bind held together
   (no spaghetti, gloves stay on) but the arms go **overhead**, not
   forward at chin/shoulder height. Peak “reach” 1.15 m at frame 2/68,
   punching hand above the head (hand_z 1.85 m on a 1.79 m figure). Guard
   and mid collapsed to the same frame. That is not a jab against the
   U.S. Army public-domain reference (tight guard, horizontal extension,
   closed fists).

Stopped there rather than keep guessing ACCAD axis conversions. Next
motion path, if we stay on Tishkok: drive only the punching arm from a
clip whose rest is a real T/A-pose (UAL Punch_Jab is proven on Quaternius
but is an open-hand left jab), or author a short jab on the fitted
metarig. Do not treat the overhead stills as a boxing jab.

### B — Better body search

Sketchfab public search (CC Attribution / CC0 only) plus OpenGameArt,
itch listings, Quaternius, Kenney. Blend Swap search page was blocked;
Poly Pizza search API needs a key (not used). OGA “Boxer Game Character”
and “Mini Boxing Character” are **2D sprite sheets** (CC0), not 3D bodies.

Nothing clearly beats Tishkok under the rules (free commercial, no NC,
no NoAI, no real-person / game / film IP, lean athletic boxer, ideally
faceless and rigged, one continuous surface). Excluded on sight: Manny
Pacquiao likeness, arcade/Street-Fighter-like “boxer”, UFC-branded female
MMA, photo-derived CrazyTalk head, NSFW “Mr Sexy Man”, cartoon kid, and
jerlymans’ rigged boxer (closest *look* among rejects — CC BY-SA, internal
ref only). No extra full-body download and no second jab test.

Sheet: `output/_basemodels/boxer_thumbs/boxer_shortlist_v2.png`.

### Verdict

**Still not a production boxing body+motion pair.**

Tishkok remains the look leader (credit required, CC BY 4.0, Tishkok) and
now *has* an A-pose Rigify metarig bind that survives posing. It is not
production-ready: simplified face, glow placement still wrong on
obliques/triceps/pec spill, and we do not yet have a closed-fist jab that
matches a real jab. Quaternius Superhero Male stays the motion guinea pig
only. No new body adopted.
