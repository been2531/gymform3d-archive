# Base-model search — replacement for Grab3D char1

Access date: 2026-11-11 (Asia/Seoul). Grab3D Human Muscles (`char1`) is a
declared failure (834 UV islands, baked ankle retinaculum, baked
`rig_normals`). This search evaluates legally reusable replacements
*before* adopting one. No Grab3D patching and no `db_row` render was done.

**Verdict: do not adopt a production clay body yet.** First-round
bodies, Diego, Beatriz, and Gridiron Jumpman stay rejected. A later
Sketchfab API round (three more CC BY downloads) produced the first
mesh that survives the weld test: deepkexz’s *Muscular human mannequin
figure* joins to **1 loose part, 0 boundary, 0 non-manifold** at 1.75 m
× 0.1 mm. That is not the same as adopting it. It is unrigged, one
million triangles, has sculpted briefs, carries custom normals, is
listed as AI-made (Rodin), and has not been posed into a textbook
row. The shirtless athlete is a lighter runner-up; the basketball
dummy is a jump-shot and is rejected as a base. *Gladiator* was
excluded before evaluation (For Honor-inspired design and a NoAI tag).
Its zip and unpacked folder were deleted and it is not in any sheet.

Large smooth 3D muscle blobs (peach lats, pink mid-back, purple rear
delt, yellow biceps) read on Diego, the mannequin, and the athlete.
On the basketball dummy only the jersey lat and upper-back patches
read. Of four glow methods tested on the mannequin, the 3D blob is
the clearest for muscle identity and the most reusable; 2D compositing
is the softest still overlay; cheap overlay shells and glass X-ray
are not usable.

Comparison sheets (gitignored stills, viewed at full size before this
write-up):

- First round: `output/_basemodels/eval/basemodels_vs_demic.png`
- Sketchfab clay round: `output/_basemodels/eval/sketchfab_vs_demic.png`
- Precise-mask colour test (superseded): `output/_basemodels/eval/muscle_colour_test.png`
- Blob glow round: `output/_basemodels/eval/blob_glow_test.png`
- Glow methods on the mannequin: `output/_basemodels/eval/blob_glow_methods.png`

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

## Recommendation

**Adopt none of the measured bodies yet** — not the first-round three,
not Diego, Beatriz, or Gridiron Jumpman, and not the three new API
downloads as a shipped skin. A muscle atlas, a Renaissance statuette,
a 16-subtool écorché kit, a fragmented anatomy study, and two jumping
sports characters are not a production gym mannequin.

The deepkexz mannequin is the first body that *survives the weld
test*. That is progress, not a green light. Do not ship a 1 M-tri
unrigged AI figure with sculpted briefs just because the glow reads.
If we continue with a download rather than authoring a body, the
next scoped experiment is: join the 10 chunks, weld, decimate,
rebuild normals, Rigify-weight the standing pose, and render a
one-arm DB row. Until that pose still exists, treat it as a
candidate, not a base. The shirtless athlete is the fallback for
that same experiment (lighter, more leftover topology, Meshy).

Diego is still the closest *look* (real geometric muscle, no clothes,
standing, no plinth) and still fails the skin test that killed
Grab3D. Do not weld the 16 subtools together and call it a base.

For muscle colour, use the **3D blob** method going forward. Optionally
grade a still with (A) 2D compositing. Do not use the cheap (B) shells
or (C) X-ray primitives.

What would actually have a chance, in order:

1. **Rigify-test the mannequin** (or the athlete if the briefs or the
   triangle count get in the way) before downloading more look-alikes.
   CharacterZone’s muscular base (`0954aa04…`, CC BY 4.0, claims
   “ready for rig”) is still unmeasured if it can be fetched legally.
2. If that bind fails, **there is still no free drop-in.** The method
   that works by construction is to author our own continuous, welded,
   unstrapped clay body (CC0), using Diego and/or the Małopolska sculpt
   as *reference only* (credit Diego under CC BY if we derive from that
   mesh; looking is not deriving), bind it once, and keep per-muscle
   glow on vertex attributes (the blob method). That is modeling work,
   not a download, and it should be scoped honestly before anyone
   starts it.
3. A paid athletic écorché would also need 중현 님’s explicit approval
   (standing rule: no paid assets).

Do not go back to Grab3D char1. Do not ship Z-Anatomy or Charleeh
meshes. Do not treat BodyParts3D’s missing latissimus as a small fix.
Do not use the excluded Gladiator (NoAI + For Honor-inspired).
