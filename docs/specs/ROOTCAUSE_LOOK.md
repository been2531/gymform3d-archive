# Why the look keeps missing demic — and the base asset kit that fixes it

The drawers, the pale cracked glow, and the white spot in the seat are not
QC misses. They come from the meshes and the methods we start with. Patching
the current glow projection, or the current body-offset drawers, cannot make
this look like [demicstory](https://www.instagram.com/demicstory/). Look
defects should never exist, because we start from well-made base assets.

This note is the write-up of that cause, plus a reusable kit whose pieces are
free and commercial-OK. It is not a request to re-render, and it does not add
gates that “catch” a bad look after the fact.

Compared on 10 Nov 2026 (Asia/Seoul) against the approved `c026ff5` stills
(`output/_rig3d/db_row/render/f_0025.png`, `f_0001.png`, `glute_closeup.png`)
and the demic bench-row frames already on the box (left panel of
`output/_rig3d/v4_authored/peach34_air/ours_vs_demic_peak.png`, also
`output/_refs_color/demic_row_variants.jpg` and `/tmp/demic_clay_preview.png`).
The side-by-side sheet is built by `tools/rig3d/compare_demic_sheet.py`.
Demicstory is a quality bar, not a source to copy (see `SOURCES.md`).

---

## 1. Drawers, and the white spot

### What demic does

Demic’s figure wears fitted black athletic shorts, not painted-on anatomy.

- A wide, separate elastic waistband (their stills show a “DEMIC” mark —
  we must not copy that mark).
- The hem sits at upper-to-mid thigh, with a little fabric ease at the
  opening.
- The cloth is smooth and matte. It follows the overall glute shape, but
  it does **not** pick up every sculpted muscle groove.
- The fabric **bridges** the gluteal cleft. The seat is a continuous piece
  of cloth. No clay shows through.

### What ours does (`c026ff5`)

Ours read as a black shrink-wrap of the Grab3D pelvis.

- No real waistband — only a 2–3 cm displace lip on copied body verts.
- The hem is high-cut, closer to a brief than to demic’s short.
- Every glute striation of the mannequin shows through the “fabric”,
  because the fabric **is** those faces, smoothed a little and pushed out
  a few millimetres.
- A pale, clay-coloured spot sits at the back-centre of the seat / crack.
- A thin clay gap also shows at the waist, where the copied shell ends.

### How they are made

`tools/rig3d/retarget_render_v2.py` (the block labelled “black boxer briefs”)
does not load a garment. It copies the body’s own pelvis and upper-thigh
faces, bisects them on two horizontal planes, smooths the copied muscle
grooves, and offsets the verts along their normals (`garment.offset_m: 0.006`
in `exercises/db_row.yaml`). A later shrink-wrap was turned off because it
tore the crotch. The material is a matte black Principled BSDF.

The Grab3D Human Muscles mesh (CC0) is a single sculpted ecorché. At the
pelvis it has a geometric groove for the intergluteal cleft, and the
vertex normals there point **into** the crack, toward each other.

### Why the white spot exists

The offset cannot invent cloth that jumps across a valley. Along the cleft
the copied faces either:

1. dive into the groove with the body, leaving a hole or a paper-thin
   grazing surface where the lighter clay body shows through, or
2. flip, because the normal points the wrong way — the same file already
   comments “back-face = white body shows through” and tries to reverse
   those faces after the fact.

Either way the eye sees a pale clay dot in the seat. That is the body, not
a specular on real cloth.

### Why every drawers round failed

The rounds were the same method, retuned:

| Round | What we changed | What the method still was | What the eye saw next |
|---|---|---|---|
| Bulky shorts | More offset / solidify, longer hem | Body faces, pushed out | Puffy, not cloth |
| Torn | Shrink-wrap to the posed body | Same faces, now glued tighter | Crotch and glute edges tore |
| Shrink-wrap / fitted shell | Drop the wrap, smooth, 6 mm rest offset | Same faces, smoother | Muscle grooves became “fabric”; white spot at the cleft |
| Gates (`briefs_gap_max`, `briefs_no_zfight`, `briefs_no_holes`) | Measure poke, tear, gap | Still an offset of the body | Numbers could pass while the garment still did not look like demic |

We also never put a demic frame next to ours in the stills loop, so “skin-tight
boxer briefs” was judged against itself, not against the reference. The
root cause is the **method** (an offset of the sculpted pelvis) on this
**topology** (a grooved single mesh), not a missing pixel test.

A real boxer-brief has its own vertices: a waistband tube, a back panel
that spans the cleft, a crotch gusset, and leg openings. Those vertices do
not exist in the Grab3D pelvis, and no amount of smoothing will create them.

---

## 2. Crack-like lines on the back glow

They are still obvious in the `c026ff5` peak still, even though
`glow_dark_seams` passed (`glow_dark_seam_frac` 0.0209 against a max of
0.028 on that commit).

### What the eye sees

Thin dark lines running through the orange lat sheet and along the
orange/purple boundary: a long crack across the lower lat, finer lines in
the mid-trap, and short breaks near the rear delt. Demic’s lat sheet is a
single connected coral shape with painted fibre strokes and a soft halo.
It has no cracks.

### What produces each kind of line

Three things stack on one mesh:

1. **UV islands and CLIP seams.** Glow colour is a 2D drawing projected
   onto the Grab3D UVs (`tools/rig3d/project_paint.py`). Where a UV is
   invalid `(-1, -1)`, or two neighbouring faces sit on different islands,
   the texture clips and a dark hairline appears. Filling neighbours and
   lowering the Y-cut (`ycut - 0.06`) closed many true holes, but not the
   seams that already have valid, discontinuous UVs.
2. **Groove geometry.** The Grab3D back is sculpted with deep muscle
   valleys. Those valleys are real mesh concavities. They catch less light
   from the clay BSDF, so they stay darker even when the projection has
   colour there.
3. **AO / cavity, only partly removed.** The current shader excludes the
   asset AO/cavity maps from the glow weight, so baked cavity is no longer
   the main culprit. Geometric self-shadowing and the normal map remain,
   and they still cut dark lines through a multiplied tint.

The lines the eye still flags are mostly (1) plus (2): a UV or island
boundary sitting in a sculpted valley, showing up as a dark crack in the
tint.

### Why every glow fix only partly removed them

| Fix | What it did | What it could not do |
|---|---|---|
| Fill projection-space holes | Closed CLIP texels at invalid UVs | Does not merge UV islands or flatten sculpted valleys |
| Exclude AO/cavity from the glow | Stopped baked cavity from multiplying extra dirt | Leaves live shading of the grooves |
| Restore clay-under-glow (`c026ff5`, after `8e1b61c`) | Brought fibre/groove grain back, so the sheet was not a flat sticker | The same grooves are exactly what read as cracks |
| Raise emission / harden `glow_w` (`8e1b61c`) | Covered cracks by painting over them | Produced the flat opaque sheet 중현 님 rejected |
| Loosen `glow_dark_seams` so “intended grain” is ignored | The number went under 0.028 | The eye still sees cracks; a passing gate hid the defect |

You cannot have sculpted-clay grain *and* a seamless painted muscle on
the same single mesh by tweaking one texture. Demic’s coloured muscles
are separate, smooth sheets with their own material. Ours are a multiply
(and a little emission) on the clay.

The passing gate was itself a defect: it was taught to ignore the thing
the eye was pointing at.

---

## 3. Paler glow

On the `c026ff5` peak still, glow pixels average about value **0.74**.
On the demic bench-row crop they average about **0.95**, with a pale
blue-white bloom around the silhouette. Saturation is similar (~0.47 vs
~0.49); the gap is brightness and cleanliness, not hue alone. Ours reads
as salmon stained onto grey clay. Demic reads as a lit, opaque coral
paint.

That follows the shader. After `c026ff5` the glow is the clay Principled
BSDF, multiplied toward the projected colour (`glow_tint: 0.65`), plus a
modest additive emission (`glow_opacity` 0.40). Mid-grey clay in a
multiply always pulls the colour down. Turning the mix into a hard
emission (`8e1b61c`) made the value high enough — and also erased the
per-muscle shapes and the clay grain underneath.

So “make it brighter” and “keep clay grain” fight each other **inside this
method**. A separate muscle mesh can be a saturated emissive object. The
clay stays clay beside it. That is how demic gets both porcelain body tone
and a vivid sheet.

Body tone is part of the same gap: demic clay is a cooler, softer
porcelain; ours is a warmer, more contrasted grey because the sculpt and
the lighting are doing the work of an ecorché study, not a studio
mannequin.

---

## 4. Recommended base asset kit

Every candidate below is free and commercial-OK. Licences were checked on
10 Nov 2026 (Asia/Seoul). Prefer CC0 when it exists. Do not commit
third-party binaries to the repo; link them, as `SOURCES.md` already
requires for reference photos.

### 4.1 Mannequin and target muscles

We need two things, and they are not the same asset:

- a clay body we can pose (we already have one);
- **separate, seamless meshes** for the muscles we colour (lats, mid-trap,
  rhomboids, rear delt).

**Keep, for the clay body: [Grab3D Human Muscles (rigged)](https://grab3d.com/models/anatomy/human-muscles/).**
CC0 1.0. Already in `exercises/db_row.yaml` (`_scratch/assets_probe/human-muscles-rigged.glb`).
Fit to demic: the sculpted ecorché *body* is close enough; the problem is
that the muscles are not separate objects. Effort: none — already bound.

**Use, for the coloured muscles: [BodyParts3D](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html)
(Database Center for Life Science).**
The official archive licence page
([`lic.html`](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html),
last updated 2025-02-27) specifies **Creative Commons Attribution 4.0
International**. Required credit:

> BodyParts3D, © The Database Center for Life Science licensed under CC
> Attribution 4.0 International

Also cite Mitsuhashi N. et al., *Nucleic Acids Res.* 2008, PMID 18835852.
Commercial use is allowed; attribution is required; there is no
share-alike on this current page. The dataset is FMA-indexed and includes
on the order of 400 named skeletal-muscle meshes, so latissimus dorsi,
trapezius, rhomboideus and deltoid arrive as their own objects.

Caveats, which matter before we ship:

- Older 2013 OBJ headers, and the Anatomography info pages, still mention
  CC BY-SA 2.1 JP. The headers point at the same `lic.html`, which now
  says CC BY 4.0. That is a defensible reading (the rights holder
  relicensed), not a certainty. Email DBCLS for written confirmation
  before a commercial launch, and do not strip the embedded headers until
  then.
- Raw BodyParts3D meshes are known to be non-manifold and holey. Cleaning
  and retopologising the four (or so) target muscles is the real
  integration cost — about two to four days, once.
- **Do not download the muscles via Z-Anatomy** if we want to stay clear
  of share-alike. [Z-Anatomy / Models-of-human-anatomy](https://github.com/Z-Anatomy/Models-of-human-anatomy)
  is CC BY-SA 4.0 for their own atlas and retopology, even though they
  started from BodyParts3D. Same for [Charleeh/ecorche](https://github.com/Charleeh/ecorche)
  (CC BY-SA 4.0, a cleaned “one named mesh per structure” glTF built from
  BodyParts3D + Z-Anatomy). Those are excellent study files; shipping them
  would put share-alike on the muscle overlays.

**CC0 fallback, if we want zero attribution and no header ambiguity:**
sculpt four thin overlay shells ourselves on the Grab3D back (lats,
mid-trap, rhomboids, rear delt), unwrap each as one island, and release
them as our own CC0 kit. No public CC0 atlas with *separate named
superficial muscles* turned up in this search. Grab3D itself is CC0, but
it is one mesh. [Innerscene’s CC0 deep-layer muscular system](https://www.innerscene.com/tools/library/3d-parts/full-body-muscular-system-selected-deep-layer-959a514b)
is a single educational figure, not a demic-style set of labelled
superficial sheets. Effort for our own shells: two to four days of
modelling, once.

Fit to demic: separate seamless muscle objects with their own emissive
material are exactly how the coloured shapes stay connected, saturated
and crack-free.

### 4.2 Boxer-brief garment

A search for a redistributable CC0 fitted boxer-brief came up empty.

- Sketchfab “[Men’s Boxer Briefs](https://sketchfab.com/3d-models/free-mens-boxer-briefs-low-poly-3d-model-05fc0011a9614a3c871e8a2c171b239b)”
  (Leo_3D) allows commercial use and modification but **forbids
  redistribution**, so it cannot live in our kit.
- The Hanes “Boxers” model is CC BY-ND — we could not even edit it.
- Some Meshy gallery items are labelled CC0, but they are AI-generated,
  the topology is unreliable, and they are a last resort only.

**Pick: author one garment ourselves and release it CC0.** A boxer-brief
is a small pattern: waistband tube, front panel, back panel that *spans*
the cleft, crotch gusset, left and right legs to upper-mid thigh. Bind it
with Data Transfer or Mesh Deform from the posed body, not by copying
body faces. Matte black fabric BSDF, roughness high, no coat. Do not put
anyone else’s wordmark on the waistband.

Fit to demic: this is the piece that makes the seat, the waist and the
hem look like clothing. Effort: one to two days, once, including a
bind-pose weight check on the bench-row hip fold.

### 4.3 Bench and dumbbell

Demic’s equipment reads as a real padded bench and a bar or dumbbell with
metal texture. Ours are serviceable primitives.

**Bench — pick: refine the commercial flat bench we already build
(black pad, gunmetal frame, no floating parts), and keep that mesh as
our CC0 kit piece.**
[Grab3D Weight Bench](https://grab3d.com/models/sports/weight-bench/) is
genuinely CC0 1.0, but it is a pastel stylised home-gym toy (coral pad,
mint plates). Poor fit to demic. Use it only if we strip it for
topology and retexture it; authoring bevels, a thick pad and simple
seam stitching on our existing bench is less work and a better match.
Effort: about a day.

**Dumbbell — pick: [Innerscene Hex Weight – Dumbbell](https://www.innerscene.com/tools/library/3d-parts/hex-weight-dumbbell-32966c1b),
claimed CC0**, “dark rubber heads and knurled steel grip”, about 5k
triangles, modelled in metres. Verify the library terms page at download
time (the listing itself says “Original Innerscene model, released CC0”).
Alternatively, add a knurl (geometry or a CC0 metal texture from
[Poly Haven](https://polyhaven.com/textures)) to the procedural dumbbell
we already grip. Effort: half a day to swap and re-parent; a day if we
model the knurl ourselves.

### 4.4 Studio light rig with contact shadows

This is mostly a rig we author, plus one HDRI.

- **HDRI:** [Poly Haven Photo Studio 01](https://polyhaven.com/a/photo_studio_01)
  or [Studio Small 01](https://polyhaven.com/a/studio_small_01). Both are
  **CC0**. Use them at low strength for reflections and a soft wrap, not
  as the only light.
- **Lights we author (CC0):** a three-point area-light rig — a large upper
  key, a wide fill, a cooler rim — matching the kind of studio wrap demic
  uses. Keep them as a reusable `studio_light_rig.blend`.
- **Contact shadows:** a slightly rough cyclorama or ground plane that
  receives ray-traced shadows (Cycles already can). An HDRI alone will not
  plant the feet and the bench; the area key and a shadow-catching floor
  will. Optional: a very weak AO term on the floor shader only, never on
  the glow meshes.

Fit to demic: this is how the clay turns porcelain and how the muscle
sheets pick up a pale edge bloom without us faking it in grade. Effort:
half a day to build and lock exposure against a reference still.

---

## 5. Build the kit once

Do this as a single asset job, not as another stills patch on `db_row`.

1. Create `assets/kit/` (gitignored binaries, tracked `LICENSE.md` that
   lists each source, licence, URL and the date we checked it).
2. Keep the Grab3D clay body and its armature.
3. From the BodyParts3D archive (not from Z-Anatomy) pull only the target
   muscles. Clean non-manifold geometry, retopo each to a few thousand
   triangles, unwrap each as one island. Or, if we skip BodyParts3D,
   sculpt the four overlay shells ourselves.
4. Fit those shells to the Grab3D back, inflate them a millimetre off the
   clay, and transfer weights from the body so they follow the same pose.
   Give each an opaque emissive material (per-muscle hue from
   `exercises/db_row.spec.yaml`) with a soft geometric falloff. Delete the
   projection-paint glow path from the hot path.
5. Model the boxer-brief. Confirm in the rest pose and in the bench-row
   hip fold that the back panel bridges the cleft (no body visible) and
   that the waistband and hem are clean. Bind it. Delete the body-copy
   offset briefs path from the hot path.
6. Finish the bench pad and frame, and swap in the knurled dumbbell.
   Parent the dumbbell with the existing grip code.
7. Save the studio light rig (HDRI world + three area lights + floor) and
   use it instead of the ad-hoc key/fill/rim numbers in
   `exercises/db_row.yaml`.
8. Rebuild the demic comparison sheet from new stills with
   `tools/rig3d/compare_demic_sheet.py`. Judge the look against demic
   *before* any beauty QC, not after.

After that, a stills render either looks like the kit or it does not.
There is nothing left to “catch” with a glow-crack gate or a clay-in-drawers
gate, because those pixels should not be possible.

---

## 6. What we will stop doing

- No more UV-fill / AO-exclude / emission-harden cycles on the projected
  glow.
- No more offset, shrink-wrap, or smooth-iter retunes of copied body
  faces for the drawers.
- No more look gates whose job is to notice that the base method failed.
  Form gates (torso angle, ROM, contact) stay. Look is a property of the
  kit.
