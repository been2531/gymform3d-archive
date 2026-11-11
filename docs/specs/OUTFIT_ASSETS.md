# Boxer outfit assets (free, commercial-safe)

Ready-made meshes to fit onto a Mixamo-rigged character in Blender 4.2,
instead of sculpting gloves, beanie, trunks and a tank in code.

Searched 11 Nov 2026 (Asia/Seoul). Sources: Sketchfab (downloadable, CC0 or
CC BY only; NoAI / NC / ND / SA / editorial excluded), Poly Pizza, Poly Haven,
BlenderKit free CC0, OpenGameArt, Smithsonian, Grab3D, MakeHuman Community.
Real brand logos and game rips were dropped (Everlast scan, Argentina "ARG"
gloves, Karhu 1952 museum glove, C.P. Company beanie, Hanes boxers).

Binaries live under gitignored `output/_assets/outfit_candidates/<item>/` and
must not be committed. A clay + textured contact sheet under v2 lighting is at
`output/outfit_candidates_v1.png` (also copied next to the downloads).

**Next step (not done here):** shrinkwrap / surface-deform the clothes onto the
chosen body, data-transfer bone weights from the body, and parent the gloves
and beanie to the hand and head bones. Recolor as needed (glossy black gloves,
charcoal beanie and tank, burgundy trunks). Remove the trunks wordmark first.

## Best pick per item

| Item | Best pick | Licence | Why, and what is still wrong |
|---|---|---|---|
| Gloves | [Classic Red Boxing Glove (Low Poly)](https://sketchfab.com/3d-models/classic-red-boxing-glove-low-poly-7faefb99c91d484995899b0e4aa6921c) by ARTEL_3D | **CC BY 4.0** — credit required | Best classic 10–12 oz silhouette we could actually download: attached thumb, cuff, wrist strap, finger grooves, welt stitching. One mesh, 16,880 tris, wrist opening (not a closed volume). **Shortcomings:** it is a single glove (mirror for the pair); the downloaded GLB has only a 1K colour map and renders as satin salmon-red, not the glossy crimson of the Sketchfab preview, and not the black leather we want; the thumb is still a rounded blob. |
| Beanie | [Black Knit Beanie](https://sketchfab.com/3d-models/black-knit-beanie-edb7f2c044b64d5da40484495239fee3) by Arkify 3D | **CC BY 4.0** — credit required | Real rib-knit photogrammetry with a folded cuff. The only candidate that already looks like knitted yarn, not a smooth helmet. **Shortcomings:** 506,517 tris split into five meshes (16-bit vertex islands); the official GLB packs a 1K texture, not the 8K claimed on the listing; in v2 lighting it reads mid-grey, not black; imported at a huge unit scale (~218 m) and must be scaled; heavy, so remesh before production. Runner-up below is easier to fit. |
| Trunks | [Shorts Boxer](https://sketchfab.com/3d-models/shorts-boxer-4029542180bd4bb68a039fc00a200b3c) by Ginny Sutton | **CC BY 4.0** — credit required | The only downloadable CC BY model that is actually boxing trunks (high white rib waistband, side stripe, mid-thigh, leg slits). One mesh, 73,736 tris, three 1K PBR maps. **Shortcomings:** the waistband has a **BØCK** wordmark that must be painted out or retextured before publish (not a famous boxing brand we could confirm, but it is still a logo); the fabric reads a bit like plastic satin; imported in millimetre-scale units (~564 m). No second trunks candidate passed the licence and type filters. |
| Tank | [sleeveless shirt](https://sketchfab.com/3d-models/sleeveless-shirt-77de2bf888f5415d85f8fb672227fa20) by zyadelbrdan74 | **CC BY 4.0** — credit required | Male athletic tank, waist length, heather knit texture, no logo. One mesh, 19,815 tris, 1K colour + a non-colour map. **Shortcomings:** colour is slate-blue, not charcoal; the male torso (pecs, waist) is sculpted into the cloth, so it will not automatically fit a different Mixamo body; armholes and neck are open shells, as a garment should be. |

## Downloads (best 1–2 per item)

All paths are under `output/_assets/outfit_candidates/`.

### Gloves

**1. Classic Red Boxing Glove (Low Poly)** — best downloaded pick.
- Author: ARTEL_3D (https://sketchfab.com/artel_3D)
- URL: https://sketchfab.com/3d-models/classic-red-boxing-glove-low-poly-7faefb99c91d484995899b0e4aa6921c
- Licence: CC BY 4.0
- File: `gloves/artel_classic_red_glove/model.glb` (1.6 MB)
- 16,880 tris, 10,487 verts, **one mesh, not closed** (3,926 open edges at the cuff)
- Textures: yes, one 1024² sRGB colour map
- Imported span: 0.16 m (already near real 10–12 oz size)

**2. Grab3D Boxing Gloves** — best licence and a true pair, GLB not on disk this session.
- Author: Grab3D (https://grab3d.com/)
- URL: https://grab3d.com/models/sports/boxing-gloves/
- Licence: **CC0 1.0** (no attribution required)
- Claimed: 18,640 tris, 2K PBR (base colour, roughness, metallic, normal), mirrored 12 oz pair with attached thumbs, white cuff, hook-and-loop strap, **no logo**, coral padding
- Origin (from their manifest): procedural Blender scripts, AI-assisted code, no generative 3D
- The box could not fetch `https://grab3d.com/dl/boxing-gloves/boxing-gloves.glb` (TLS EOF; local DNS for grab3d.com points at 198.18.0.1). The public manifest and CC0 licence text were confirmed. Licence copy: `gloves/grab3d_boxing_gloves/LICENSE`. Retry the GLB later; it is the one we would rather use as a pair and recolor to black.

Shortlisted, not downloaded:
- [Vintage / Old boxing gloves](https://sketchfab.com/3d-models/vintage-old-boxing-gloves-b41079124d1e420d8549d088917b1dff) by maxsbond.work, CC BY 4.0, pair, 115,264 tris. Good leather, but lace-up vintage, not a modern 10–12 oz glove.
- [Boxing gloves](https://sketchfab.com/3d-models/boxing-gloves-c1572e87f6f2467caf1ece6cf39d10e6) by Nixak, CC BY 4.0, pair, 49,352 tris. Weak / missing thumbs.
- Poly Pizza [Boxing Gloves by Zsky](https://poly.pizza/m/l2S6HNEpey) and [Boxing Glove by J-Toastie](https://poly.pizza/m/35PCBHLrd4), CC BY 3.0, low-poly game props.

Rejected: Santa Cruz MAH Everlast scan; al1905 "ARG" logo; urheilumuseo Karhu 1952 signed glove; several NoAI-tagged gloves (incg5764, Gohar.Munir, alban).

### Beanie

**1. Black Knit Beanie** — best look.
- Author: Arkify 3D (https://sketchfab.com/arkify)
- URL: https://sketchfab.com/3d-models/black-knit-beanie-edb7f2c044b64d5da40484495239fee3
- Licence: CC BY 4.0
- File: `beanie/arkify_black_knit_beanie/model.glb` (16.3 MB)
- 506,517 tris, **five meshes**, not closed, one 1024² sRGB map
- Imported span: ~218 m (scale by ~0.0018)

**2. Beanie** by oluwaphemi — best topology for fitting.
- Author: oluwaphemi (https://sketchfab.com/oluwaphemi)
- URL: https://sketchfab.com/3d-models/beanie-ff6dc83c818c4e2aa722cb89e6ada577
- Licence: CC BY 4.0
- File: `beanie/oluwaphemi_beanie/model.glb` (3.8 MB)
- 38,844 tris, **one mesh**, not closed (cuff opening), three 1024² maps (colour + two non-colour, likely normal / roughness). Material name is `Rib_2X2_468gsm_FRONT_1948`.
- Light grey / off-white, clear 2×2 rib and a folded cuff. Crown gathers into a slightly pinched bun. Recolor to charcoal. Imported span already ~0.21 m.

Shortlisted, not downloaded:
- [Knit Hat - Cap Grey - Beanie](https://sketchfab.com/3d-models/knit-hat-cap-grey-beanie-4493602bc41e482a975482c1830f77fb) by Marc Sawyer / whitewashstudio, CC BY 4.0, 24,746 tris, made for Character Creator / iClone, already shown on a head. Tighter, less rib relief.
- [Grab3D Beanie](https://grab3d.com/models/props/beanie/), CC0, 8,688 tris, 2K PBR, procedural. Mustard cable knit with a cream pompom — not a boxer watch-cap unless the pompom is deleted and it is recolored. Same fetch problem as the Grab3D gloves.

Rejected: C.P. Company Beanie (brand); M1941 wool cap with goggles; cat-ear beanie; NoAI cartoon cap.

### Trunks

**1. Shorts Boxer** — only usable boxing-trunks mesh found.
- Author: Ginny Sutton (https://sketchfab.com/suttonj94)
- URL: https://sketchfab.com/3d-models/shorts-boxer-4029542180bd4bb68a039fc00a200b3c
- Licence: CC BY 4.0
- File: `trunks/sutton_shorts_boxer/model.glb` (5.9 MB)
- 73,736 tris, **one mesh**, not closed (waist and legs), three 1024² PBR maps
- Imported span: ~564 m (scale by ~0.00071)
- Tags on the listing include boxing and Marvelous Designer / Substance.

No second download. Casual gym shorts, soccer shorts, and female cycling shorts were the wrong garment. Other boxing-trunks listings were not downloadable or were NC / ND / brand (Hanes). MakeHuman "Boxer Shorts" looks like underwear and was not fetched. BlenderKit "Thick Strap Tank Tops" and similar clothes are Royalty Free / Full Plan, not free CC0.

Before use: replace the **BØCK** waistband badge (and check the left hem for a second mark). Recolor the body to burgundy if that is still the chosen palette; the white waistband can stay or shift to a charcoal stripe.

### Tank top

**1. sleeveless shirt** — best downloaded pick (see table).
- File: `tank/zyadelbrdan74_sleeveless_shirt/model.glb` (2.0 MB)
- Imported span: ~598 m (scale by ~0.00067)

**2. nycfashion Sleeveless Shirt** by mweisburgh
- URL: https://sketchfab.com/3d-models/nycfashion-sleeveless-shirt-7f3c463c314848e0a01bc4a77b664f49
- Licence: CC BY 4.0
- File: `tank/nycfashion_sleeveless_shirt/model.glb` (461 KB)
- 16,084 tris, one mesh, **no image textures** (a plain "Cloth" material)
- Clean sleeveless cut, but the chest has a crude pouch-like bulge and the cloth has no knit. Backup shape only. Imported span ~2.9 m.

Shortlisted, not downloaded:
- [Black Sports Tank Top](https://sketchfab.com/3d-models/black-sports-tank-top-16b544e3185a474a9e8e5f2c0fb8eb7d) by Style3D CG, CC BY 4.0, 44,550 tris. Female racerback, cropped like a sports bra — too short for the charcoal tank we want.
- MakeHuman Community `toigo_keyhole_tank_top` (CC0, shirts01 pack) is a fashion keyhole top, not an athletic tank, and is authored for the MakeHuman mesh.

## Required credits (CC BY)

Use these lines wherever the meshes appear (reel description, in-film end card, or a credits file shipped with the render). CC0 items need no credit.

> "Classic Red Boxing Glove (Low Poly)" (https://sketchfab.com/3d-models/classic-red-boxing-glove-low-poly-7faefb99c91d484995899b0e4aa6921c) by ARTEL_3D (https://sketchfab.com/artel_3D) licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

> "Black Knit Beanie" (https://sketchfab.com/3d-models/black-knit-beanie-edb7f2c044b64d5da40484495239fee3) by Arkify 3D (https://sketchfab.com/arkify) licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

> "Beanie" (https://sketchfab.com/3d-models/beanie-ff6dc83c818c4e2aa722cb89e6ada577) by oluwaphemi (https://sketchfab.com/oluwaphemi) licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

> "Shorts Boxer" (https://sketchfab.com/3d-models/shorts-boxer-4029542180bd4bb68a039fc00a200b3c) by Ginny Sutton (https://sketchfab.com/suttonj94) licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

> "sleeveless shirt" (https://sketchfab.com/3d-models/sleeveless-shirt-77de2bf888f5415d85f8fb672227fa20) by zyadelbrdan74 (https://sketchfab.com/zyadelbrdan74) licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

> "nycfashion Sleeveless Shirt" (https://sketchfab.com/3d-models/nycfashion-sleeveless-shirt-7f3c463c314848e0a01bc4a77b664f49) by mweisburgh (https://sketchfab.com/mweisburgh) licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

Grab3D gloves and beanie are CC0 1.0; attribution is optional. If the Grab3D GLB is used, keep their licence text with the files.

## Contact sheet

`output/outfit_candidates_v1.png` — each downloaded candidate as clay (left) and its own textures (right), v2 lighting (AgX, exposure 0, key 78, fill 36%, cool rim 55%, world 0.018, shadow-catcher floor, 50 mm vertical / 24 mm sensor, yaw 35°, elevation 10°, about 72% fill, 96 spp + OIDN). Assets that imported at extreme scales were normalized to a 0.40 m span for the sheet only.

Helper (uncommitted): `tools/rig3d/build_outfit_candidates_sheet.py`.

## MPFB boxer v1 (character + first fit)

Pipeline: `tools/rig3d/build_mpfb_boxer.py` (Blender 4.2 headless, `--phase body|retarget|qc|outfit|render`). Intermediate blend and stills are gitignored under `output/_basemodels/mpfb/` and `output/_basemodels/eval/mpfb_boxer_v1.png`.

**Body.** MPFB 2.0.18 male, height 1.779 m after targets (macro height 0.532). Macros: gender 1, age 0.5, muscle 0.68, weight 0.48, proportions 0.58, firmness 0.45. Targets: v-taper, shoulder width, waist in, moderate pecs/dorsi/arm muscle, stomach tone, slight hip narrow. Hair off, no face detail, clay v2. Evaluated mesh (helper Mask kept live, first in the stack) is closed: 13,380 verts, 13,378 faces, 0 open edges. Applying the Mask or merging by distance after the Mixamo rig remaps weights and stretches the mesh — do not apply those.

**Rig + jab retarget.** Built-in MPFB `mixamo` rig (52 `mixamorig:*` bones). World-space rotation bake from the v4 jab (`alts/alt04_Jab Cross_BoxingJabCrossMedium.fbx`, guard 9 / extension 14 / return 22) onto the body bones, hips location in world XY/Z. Hand, finger, and toe bones are left at rest relative to the forearm/foot: MPFB's Left/RightHand bones are ~47 cm (X Bot's are ~8 cm) and a world or baked local copy of those channels stretches the hands into spikes. The hand bones are shortened to 8 cm after the rig is fitted. QC vs the same clip on X Bot, all within tolerance and not worse:

| metric | MPFB | X Bot | delta |
| --- | --- | --- | --- |
| fist_max_dev_m | 0.0027 | 0.0111 | −0.0084 (straighter) |
| elbow_above_shoulder_max_m | −0.0539 | −0.0601 | +0.0062 |
| foot_slide_l_m | 0.368 | 0.375 | −0.007 |
| root_drift_m | 0.4444 | 0.4444 | 0 |

Open, unsplayed-to-a-fist fingers are a leftover of skipping those bones; gloves are meant to hide them. The bare jab still reads as a boxer, not as X Bot's slimmer default.

**Outfit (honest v1).** Scaled by rewriting mesh data (Sketchfab trunks/tank import at 560–600 m). Logo albedo on the Sutton trunks is replaced with burgundy satin (0.18, 0.025, 0.04) plus a charcoal waistband vertex-color trim. Tank and beanie recolored charcoal; gloves recolored glossy black leather and mirrored. Fit is shrinkwrap + data-transfer weights (body-only deform bones) + live shrinkwrap + corrective smooth.

Known failures at this pass, judged at full size: the beanie sits on the back of the skull rather than as a watch cap; the mirrored Artel gloves do not stay on the fists through the jab (they drift onto the forearm or float); the tank flares into a back sail and is not a usable sleeveless shirt yet (baked-in muscle + shrinkwrap); the trunks read as tight briefs more than boxing trunks and still poke at return. Deltoids stay visible. Treat the outfit as a first contact, not a publishable kit.

## MPFB credits

MPFB 2.0.18 (MakeHuman Plugin For Blender) from [makehumancommunity/mpfb2](https://github.com/makehumancommunity/mpfb2). The add-on is GPL-3.0-or-later; the basemesh, targets, and built-in rig data used here are CC0 (see `LICENSE.ASSETS.md` in that repo). No extra credit is required for the body; keep the CC BY lines above for any outfit mesh that appears in a published still or reel.

## What is still missing

- A downloaded **pair** of glossy **black** 10–12 oz gloves. The best shape we have is a single salmon-red glove; the best pair (Grab3D, CC0, coral) did not arrive on the box this session.
- A second boxing-trunks mesh, and a trunks texture **without a wordmark**.
- A charcoal tank that is a loose garment, not a cloth already sculpted to one male torso.
- A beanie that is already charcoal/black at the texture level (both good beanies need a recolor; Arkify also needs a remesh).
- A second outfit pass: glove orientation + rigid follow on the fist, beanie seated on the crown, tank without a back sail, trunks with a real boxer cut and less poke-through.
