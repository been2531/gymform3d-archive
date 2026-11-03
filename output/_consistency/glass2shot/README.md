# D7 smoked glass + 2-shot STILLS (t687u)

Owner-approved combo: ② smoked glass + Ⓑ 2-shot, rear-3/4 high. Stills only: no video, no render.

- Sheet: `G2S_STILLS_sheet.jpg`
- QC and metrics: `g2s_stills_qc.json`
- Script: `nice -n 19 python3 tools/glass_2shot_stills.py` (about 4–5 s)
- Look module (opt-in): `gymform/glass_look.py`

**Inputs (`gen/`)**
- 3 × Grok Bot GenerateImage, all native 720×1280:
  - `wide_peak_gen1`: an edit of the verified empty plate plus the real photo;
  - `tight_peak_gen2`: references gen1 and the plate;
  - `tight_start_gen3`: references gen2.
- MediaPipe keypoints: `kp.json`; silhouettes: `*_seg.png`.

**How the look is built**
- **Glass body:** 30 % see-through to the verified empty plate (`plate_empty_1024.jpg`), aligned by ORB similarity. Navy smoke tint, fresnel rim, the shorts stay matte black.
- **Neon:** Day 4 right HSV. The fibre strokes run along each muscle's direction: lats as a fan from spine/iliac crest to the armpit, rhomboids, mid trap, rear delt, faint biceps. They live in a canonical torso space that is affine-warped from the keypoints, so in video they track the body with no sliding.
- **Callout:** `joint_callout.draw`, number 84 px, label 26 px, one neon leader, no arc.

**Resolution:** the 1080 output is a Lanczos ×1.5 of the native 720 still. The glass body is smooth, so it hides the upscale; the fibres, rim and callout are drawn at 1080.

**Machine fidelity:** checked against the verified plate, an edit of the real Matrix photo that passed part by part (`docs/mission/D7_MACHINE_CONCEPT.md` §9). The `_align_*` images show each still warped onto the plate.

**Not committed:** `gen/ref_real_photo.jpg`, a third-party product photo kept for internal reference only.
