# Boxer options — character body, video-to-motion, other angles

Access date: 2026-11-11 (Asia/Seoul). Research only: no production renders.
Context: X Bot has odd proportions and a segmented waist that will not take
clothes; the MPFB retarget distorted the guard; third-party outfit pieces
kept misplacing (t876u–t884u). 중현 님 asked for solutions from several
angles (t884u).

Comparison sheet (gitignored, viewed at full size before this write-up):
`output/_basemodels/eval/boxer_options_v1.png` — columns X Bot | Tishkok
Handpainted | Kickboxer | Muscle Man | chempunk; rows rest / guard f9 / jab
full f14 (X Bot alt04). Stills are Cycles 32 spp, 360×640, glTF textures
as shipped, which makes them smoke-test stills, not v2-lighting renders.

Replay:

```
B=/home/box/opt/blender-4.2.3-linux-x64/blender
# inspect rig / bones / rest pose / weld
nice -n 19 $B -b --python tools/rig3d/inspect_boxer_candidate.py -- \
  --input output/_basemodels/boxer_v3/<slug>/scene.gltf --out <facts.json>
# X Bot reference + candidate jab smoke test
nice -n 19 $B -b --python tools/rig3d/boxer_candidate_smoke.py -- --input x --name xbot --xbot-only --samples 32
nice -n 19 $B -b --python tools/rig3d/boxer_candidate_smoke.py -- \
  --input output/_basemodels/boxer_v3/<slug>/scene.gltf --name <slug> --samples 32 [--bonemap chempunk]
python3 tools/rig3d/build_boxer_options_sheet.py
```

Downloads came from the Sketchfab download API (token in `~/.sketchfab/env`,
never printed or committed) into gitignored `output/_basemodels/boxer_v3/<slug>/`.

## Smoke-test method

- Source motion: X Bot `alt04_Jab Cross_BoxingJabCrossMedium.fbx`, left jab
  guard f9 / full f14 / return f22, the same jab the MPFB test used.
- Retarget (`boxer_candidate_smoke.py`): bones renamed to canonical Mixamo
  names; each target bone gets the **source bone's world direction** (swing)
  while keeping **its own rest twist** (swing–twist split of
  `src_rest⁻¹ · tgt_rest` about the bone Y axis). Hips translation is the
  source's displacement from rest × hip-height ratio. Facing is matched on
  the **rest** pose. The first attempt matched facing on X Bot frame 1, which
  is already a rotated stance, and put a 46° yaw into the spine twist. That
  bug was fixed before the numbers below were taken.
- Scoring: `boxing_qc.punch_path_metrics` on the bones, as in
  `build_mpfb_boxer.py`, plus three things the MPFB test missed:
  - guard hand-to-head distance / height, for lead and rear hands;
  - guard hand separation / shoulder width;
  - **mesh edge-stretch**: the ratio of skinned edge length to rest edge
    length at f9 and f14, which catches skin tearing and spikes that bone
    metrics cannot see.
- X Bot reproduces the stored reference exactly: fist dev 1.11 cm, elbow
  −6.0 cm, foot slide L 0.375 m.

## A — Rigged boxers with an outfit (Sketchfab, CC BY / CC0, rigged, downloadable)

The API search (`rigged=true&downloadable=true`, licence `by` and `cc0`)
covered the queries boxer, boxing, boxing character, fighter, mma fighter,
kickboxer and athlete rigged, and returned 109 unique hits. Excluded on
sight:

- **Real people:** Chrisean Rock, Apollo Creed, Borat, "Volodymyr K.".
- **Game or film IP:** Street Fighter, Mario, Pokémon, Toy Story, Kim
  Possible, Skibidi.
- **NSFW:** "Mr Sexy Man" and the Chun-li nsfw model.
- **Mixamo or Fuse stock characters re-uploaded as "CC BY":** "Strong
  Security – Boxing", which says "Adobe Fuse CC – Mixamo", plus the Y-bot
  and anonymous "Boxing" uploads. Adobe does not let third parties
  relicense these.
- **Animation-only uploads.**
- **`createdwithai`-tagged models.**
- **MMA Kick (3DDomino):** it has a Reallusion/iClone-style 74-bone rig and
  unclear provenance, so it was not tested.

No CC0 rigged boxer turned up on Sketchfab. Five were downloaded. Measured:

| Candidate | Licence (license.txt) | Rig | Outfit | Weld (0.1 mm) | Jab smoke |
|---|---|---|---|---|---|
| **Tishkok — Boxing Character – Handpainted** `4fddd902…` | CC-BY-4.0 | Mixamo 26 bones, near-T (upper arm −6.5°) | gloves (green), trunks (kente-pattern), boots | 1 mesh, 9,748 tris, 7 shells, 8 boundary | **PASS** |
| **OldTÜRK34 — KICKBOXER_WITH_MIXAMO_ANIMATION** `e868e8dd…` | CC-BY-4.0 | Mixamo 66 bones with fingers, A 17° | red shorts only; **bare hands, barefoot** | 5,660 tris, 5 shells, 0 boundary, 6 non-manifold | **PASS** |
| YuxiangCG — Muscle Man `9bf7f4c4…` | CC-BY-4.0 | Mixamo 26 bones, A 33° | yellow shorts; no textures in the glTF | 1,581 shells, ~35 k boundary edges | **FAIL** |
| finnvleuten — free simple chempunk boxer `17fd1ece…` | CC-BY-4.0 | custom 35 bones (ass/chest/upper_arm…) | gas-mask head, gloves, trunks | 26 k-tri shell has 156 boundary edges | **FAIL** |
| X Bot (reference) | Mixamo | Mixamo | none | segmented | reference |

Smoke numbers (alt04 jab, same frames):

| | fist dev | elbow vs shoulder | guard lead / rear hand–head (×H) | guard hand sep / shoulder | foot slide L | root drift | edge stretch f14 p99 / max |
|---|---|---|---|---|---|---|---|
| X Bot | 1.1 cm | −6.0 cm | 0.31 / 0.23 | 0.95 | 0.37 m | 0.44 m | 1.01 / 1.23 (rigid segments) |
| Tishkok HP | 0.5 cm | −6.2 cm | 0.36 / 0.22 | 1.00 | 0.32 m | 0.38 m | 1.62 / 2.94 |
| Kickboxer | 0.3 cm | −4.5 cm | 0.31 / 0.17 | 0.98 | 0.37 m | 0.43 m | 1.84 / 3.52 |
| Muscle Man | 0.8 cm | −7.2 cm | 0.36 / 0.24 | 1.06 | 0.37 m | 0.43 m | 1.52 / **502** |
| chempunk | 1.4 cm | −8.2 cm | 0.49 / 0.27 | 0.95 | 0.35 m | 0.23 m | **12.1 / 63** |

Edge-stretch p99 of 1.5–1.8 is normal for a continuous skin at the
shoulder and elbow. X Bot's 1.01 comes from rigid segments, not from a
better result. A max in the hundreds means torn or unweighted vertices.

### Tishkok — Boxing Character – Handpainted (best look, best fit)

- Source: <https://sketchfab.com/3d-models/boxing-character-handpainted-4fddd9021cd246f6a459efb112f5bb8b>.
  Credit line from license.txt: *This work is based on "Boxing Character –
  Handpainted" (…4fddd9021cd246f6a459efb112f5bb8b) by Tishkok
  (https://sketchfab.com/Tishkok) licensed under CC-BY-4.0.*
- It is the rigged, game-ready sibling of the unrigged Tishkok sculpt from
  the earlier round. It was already auto-rigged with Mixamo
  (`mixamorig:*`), so the X Bot clips map by name and no rest-pose
  surgery is needed.
- Flaws first:
  - Stylized hand-painted look, closer to a game than to the clay
    biomechanics look.
  - It has a face with a fixed scowl.
  - The kente-pattern trunks and green gloves are baked into one texture
    atlas, so recolouring to black and burgundy needs a UV-region mask
    (one material).
  - No tank top and no beanie.
  - The guard lead hand sits about 5 % of height further from the head
    than X Bot's.
  - Low poly (9.7 k tris), so muscle glow will be coarse.
- At full size the jab reads as a real jab: closed sculpted gloves, chin
  behind the rear glove, no tearing at the shoulder.

### OldTÜRK34 — Kickboxer (best body, no kit)

- Source: <https://sketchfab.com/3d-models/kickboxer-with-mixamo-animation-e868e8dda18b494f845068d23ad02ac1>.
  It is CC-BY-4.0, and the author credit in license.txt is "OldTÜRK34
  (https://sketchfab.com/mr.turkish34)".
- Flaws first:
  - No gloves, bare hands and barefoot.
  - Bodybuilder proportions (very wide lats), not a lean boxer.
  - Visible crease on the back and lat at full extension (p99 stretch
    1.84, the highest of the passing candidates).
  - Model provenance (hand-made or AI) is not stated; it carries no AI
    tag, but this is unverified.
  - It has a face.
- It has the closest guard (rear hand 0.17 H) and the cleanest continuous
  skin (0 boundary). Its full Mixamo finger rig closes the fist from the
  X Bot clip.

### Muscle Man / chempunk — rejected

- **Muscle Man:**
  - The glTF has no textures.
  - Hands and feet are blobs.
  - It has 1,581 shells and edge-stretch spikes up to 500×.
- **chempunk:**
  - The model explodes into a giant shell as soon as the animation is
    cleared to rest (the bind is not the rest pose).
  - The custom rig was not chased further.

### Other sources checked

- **Mixamo characters:** no boxer or muscular character (already
  established earlier today).
- **Quaternius:**
  - UBC Standard Superhero, already tested, is not a boxer.
  - *Animated Men Pack* (CC0, Poly Pizza) is low-poly generic men with a
    punch clip.
- **Poly Pizza:** *Male Fighter* by mastjie is CC0 per the listing, has no
  confirmed gloves and was not downloaded.
- **Kenney:** no boxer.
- **OpenGameArt:** boxers are 2D sprite sheets (earlier round).
- **Brojek "Blocky Boxer":** Roblox-style and sits behind a US$3 pass, so
  it is excluded (paid).
- **Meshy community "Boxing Warrior" and "Boxing Stance"** are marked CC0
  but are AI-generated and unrigged. They were not downloaded.

## B — Video to motion

| Tool | Licence (verified this round) | Output | Box CPU? | Fast punches / feet | Usable for a monetised account? |
|---|---|---|---|---|---|
| GVHMR (zju3dv) | LICENSE: "educational, research and non-profit purposes only … prohibited for commercial use" | SMPL(-X) params | needs a CUDA GPU in practice (not tested) | strong world-grounded results and less foot slide; it still blurs at punch speed | **No** (code licence and SMPL) |
| WHAM | MIT code | SMPL params plus foot-contact | GPU (not tested) | foot-contact head reduces sliding | **No**: needs the SMPL body model |
| 4D-Humans / HMR2 | MIT code | SMPL params | CPU possible but slow (not tested) | per-frame, so jitter; no world trajectory | **No**: SMPL |
| TRAM | needs SMPL and SMPLify registration; third-party licences | SMPL plus camera | GPU and DROID-SLAM CUDA build | good trajectory | **No**: SMPL |
| SMPL body model | "non-commercial scientific research, non-commercial education, or non-commercial artistic projects … any use for commercial purposes is prohibited"; commercial licence via Meshcapade (paid) | — | — | — | blocks all four tools above |
| Rokoko Vision (Starter, free) | Rokoko FAQ: "all generated data with text-to-motion and with video-to-motion can be used commercially"; Starter has 30 s/month of video, FBX only (BVH is paid) | FBX with Mixamo skeleton preset | runs in the cloud | single camera, so the depth axis of a straight jab toward the lens is weak; cleanup filters | **Yes**, within 30 s/month |
| DeepMotion Animate 3D (Freemium) | Terms: Freemium is a "Non-Commercial License"; commercial use needs a paid plan | FBX / BVH / GLB | cloud | has foot locking | **No** on free |
| FreeMoCap | AGPL-3.0 (running it locally to make data is fine; AGPL bites only if we distribute a modified build) | Blender scene / CSV / BVH via add-on | CPU OK; needs **2+ synced cameras** we record | multi-view triangulation fixes depth; MediaPipe joints, so no finger or fist detail | **Yes** (self-recorded) |
| MediaPipe Pose Landmarker | Apache-2.0 code and model card | 33 landmarks; "world" metres with the hip origin; z "not metric but up to scale" | CPU real-time | single view; z jitter and wrist dropouts on fast punches; no foot contact, so we would need our own foot lock | **Yes** |

Honest read:

- Every state-of-the-art monocular method with good foot handling (GVHMR,
  WHAM, TRAM, HMR2) depends on SMPL. SMPL is non-commercial, so they are
  out for @gymform.3d unless the SMPL licence is bought.
- The free and commercial-safe routes are Rokoko Vision (30 s/month, FBX
  Mixamo), MediaPipe and FreeMoCap. None is better than the existing
  Mixamo, CMU and ACCAD clips for a clean jab.
- Their real value would be clips we cannot get elsewhere: specific
  mistakes performed on camera, slips, rolls and footwork.
- Fast punches: at 30 fps a jab spans about 4–6 frames with motion blur.
  To work they need:
  - 60–120 fps;
  - a 45° or side camera, since a punch straight at the lens has the
    weakest axis;
  - fitted clothes and a plain background.
- **Not tested on the box this round.**

Legality (not legal advice):

- **Downloading other creators' reels or broadcast fights** copies their
  footage (copyright) and breaks Instagram's or the broadcaster's terms on
  collecting content. Do not do it, even when only the skeleton is kept.
- **The extracted motion** of ordinary athletic movement (a jab, a slip)
  is generally treated as functional movement, not protected choreography.
  The risk is the *footage* we process and any distinctive sequence that
  clearly copies one creator's reel.
- **Safe sources:**
  - self-recorded clips (중현 님 or a consenting athlete with a simple
    release);
  - CC0 or CC BY video (Wikimedia Commons), with credit when it is CC BY;
  - the mocap we already use: CMU (free), ACCAD (CC BY 3.0) and Mixamo.

## C — Other angles (rough effort)

1. **Tishkok Handpainted plus a UV-mask recolour** (gloves to glossy black,
   trunks to burgundy with charcoal trim). Optionally paint a charcoal tank
   and skull-cap beanie into the same atlas.
   - **Effort:** about 0.5 day.
   - **Pros:** keeps the Mixamo rig, so X Bot clips work by name, as
     proven above.
   - **Risks:** a painted tank has no fabric folds; stylized look; glow on
     dark painted skin needs testing.
2. **Kickboxer plus hand-derived gloves** (the v3 "inflate the hand" glove
   that stayed on in every frame) plus a burgundy shorts recolour.
   - **Effort:** about 0.5–1 day.
   - **Risks:** bodybuilder silhouette; back crease at full extension;
     unverified provenance.
3. **Own vinyl-toy boxer, one closed mesh** (gloves, trunks, tank and
   beanie sculpted as part of the body, so nothing can slip), Rigify or
   Mixamo-compatible bones, then the same direction retarget.
   - **Effort:** about 2–4 days of modelling.
   - **Pros:** we own it (CC0), with no credit and no provenance doubt.
   - **Risks:** time; quality depends on modelling skill.
4. **AI image-to-3D boxer** (Meshy or Tripo free tiers) followed by the
   Mixamo auto-rigger.
   - **Effort:** about 0.5 day if it works.
   - **Risks:**
     - free-tier output licence not verified this round;
     - AI provenance;
     - messy topology;
     - today's Mixamo upload of the Tishkok sculpt already failed to
       auto-rig.
   - Lowest confidence.
5. **Rokoko text-to-motion** (Starter: unlimited clips, 5 Studio
   imports/month) as a source of extra boxing variations.
   - **Effort:** about 1 h to test.
   - **Risk:** quality unknown.

## Recommendation

**Tishkok — Boxing Character – Handpainted (CC BY 4.0), plus a
texture-mask recolour, plus the X Bot Mixamo clips by name.**

- It is the only candidate that arrives as a boxer with gloves, trunks and
  boots on one skinned mesh.
- It is already on a Mixamo skeleton.
- The alt04 jab transferred with path metrics equal to X Bot's and no
  skin tearing.

Risks:

- the look is stylized and game-like, not clay;
- it has a face, and the CC BY credit is mandatory;
- recolour and any tank or beanie are texture paint, without real cloth;
- the low poly count limits muscle-glow detail;
- only one clip (the jab) was tested; cross, hook and uppercut still have
  to pass the same smoke test;
- the guard lead hand sits about 5 % of height wider than X Bot's.

Fallback: Kickboxer plus hand-derived gloves.
