# D7 character × composition concept v2 (t685u): 0 generations

Sheet (mobile, Korean): [`CONCEPT_V2_sheet.jpg`](CONCEPT_V2_sheet.jpg) · numbers: [`CONCEPT_V2_metrics.json`](CONCEPT_V2_metrics.json),
[`COMPOSITION_fill_estimate.json`](COMPOSITION_fill_estimate.json) · scripts: `tools/concept_v2_sheet.py`, `tools/composition_fill_estimate.py`.
The character panels are code mock-ups on the existing D7 right-peak frame: MediaPipe silhouette, real machine pixels,
and the real empty plate behind the glass body. They are not final render quality.

## 1. Reference research (public pages, 2026-11-04 KST)
I didn't embed any third-party frames in the repo, only links. I kept one public poster locally for study, not committed.

| Account / source | Character | Machine camera | Body fill (9:16 or 16:9) | Whole machine? | Text / rhythm |
|---|---|---|---|---|---|
| [@demicstory](https://www.instagram.com/demicstory/) (12.6 M followers, ~800 K plays/reel, ~3 posts/day, [PicMany](https://picmany.com/demicstory)) | muscular grey mannequin, X-ray glow on the target muscles | close, often rear-3/4; machine partly cropped | body ≈ 70–90 % of height (our Day 1/2 copy this) | **no**, mostly cropped | big bold callouts; 1–2 s cuts between moves |
| [Muscle and Motion](https://www.instagram.com/muscleandmotion/) | écorché / anatomical muscles and bones | studio, no machine or a minimal one | body fills frame | rarely | educational captions; slower |
| [MoveKit seated row](https://movekit.com/exercises/wide-grip-seated-cable-row) | light-grey clay mannequin, soft relief, red highlights | **front-3/4, slightly high** | body ≈ 85 % of a 16:9 height | **no**, the tower is cut | none; 7.2 s seamless loop |
| [Gym Animations](https://gym-animations.com/) | grey mannequin, muscle highlight, white bg | 3/4 | large | partly | 4K / 60 fps clean loops |
| [Easy Anatomy](https://www.instagram.com/easyanatomyappofficial/) | anatomical muscle body | studio | large | n/a | app promo |

Pattern: the top performers make the BODY big and crop the machine. Nobody keeps a 2.4 m tower and a big body in one 9:16 shot.

## 2. Geometry (Matrix Aura Low Row G3-MS53, spec 230 × 71 × 240 cm; seated user head ≈ 1.3 m)
With the whole machine in one 9:16 shot, the person can be at most **21–28 %** of the height, whatever the angle:

| Composition | Person | Machine |
|---|---|---|
| side (now) | 22 % | 54 % |
| **A** rear-3/4 high | 21 % | 89 % |
| **C** front-3/4 eye | 28 % | 76 % (back muscles hidden) |
| **B** 2-shot: wide rear-3/4 high → tight, same angle | 21 % → **47 %** (full body; upper body larger) | full machine every loop |

## 3. Character directions
1. **Écorché muscle body** (Day 4 body as is): real muscle cues, but generated detail crawls in motion (texture 3.3) and the dark head sinks into the backdrop (body 29 / backdrop 38).
2. **Smoked glass** (recommended): translucent navy body with a rim light. Only the target muscles glow neon (code), and the real machine shows faintly through the body. No surface detail to crawl (1.5); head = body tone (32/32).
3. **Light clay** (MoveKit-like): stable, with a uniform head (146/143). But neon on a light body loses punch, and the body merges with the white machine frame.

## Recommendation: **② smoked glass + Ⓑ 2-shot (wide rear-3/4 high → tight)**
- **Motion:** the body texture comes from code, so nothing crawls.
- **Fibre look:** the Day 4 neon is drawn 100 % in code inside the glass.
- **Head tone:** same material as the body by construction.
- **Machine:** shown whole in every loop, and visible through the body.
- **Feel:** dark glass + neon reads as premium.
- **Speed:** 2 plates + 2 clips; everything else is code.

## Next step (needs generation approval)
1. 2 × GenerateImage: a smooth opaque mannequin on the real machine, rear-3/4 high. One wide 9:16 shot with the whole machine, one tight shot at the same angle.
2. 1–2 × GenerateVideo, 3–4 s start→peak loop. The wide clip at 720p (first + last frame). The tight clip at 1080p (image-to-video, no last frame) for sharpness.
3. Code: glass shader, Day 4 neon fibres tracked with keypoints, neon-leader angle. The matte needs flow-EMA (DEFECT_PLAYBOOK D4) to stop the rim from flickering.
