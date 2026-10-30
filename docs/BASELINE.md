# @gymform.3d BASELINE TEMPLATE (t530u–t532u)

Owner (t530u–t532u): Day 6 fell apart because too much changed at once. **Build from what already worked in the
posted Day 1–5 reels, then change ONE element per day** (each change = its own preflight + owner look).
Contact sheet of the chosen frames: `output/baseline_best_frames.jpg` (rows: Day 5 v5, Day 4, + Day 1 X-Ray and the Day 5 native bone clip).

Posted Day 1–5 (exercise_curriculum.json): D1 `output/day1_v36_master_reel.mp4` (De_Yl9rAYC4) · D2
`output/day2_v38_master_reel.mp4` (De_mCcFAVTO) · D3 `output/day3_barbell_squat_simple2_master.mp4` (DfCpKW4IIV8) ·
D4 `output/day4_romanian_deadlift_4step_master.mp4` (DfC8EifoxFa) · D5
`output/day5_dumbbell_lateral_raise_4step_master_v5.mp4` (DfFPS3YFLF3). Day 6 (DfGabTBjwt8) deleted, t529u.

**Primary base = Day 4 + Day 5 (t534u: "Day4, Day5가 진짜 잘나오고 있자나")**: pipeline, look, framing, clip
generation all come from Day 5 v5 (built on the Day 4 look). From Day 1–3 only clearly better single elements are
taken: **Day 1 v36 bright X-Ray bones** (never dark, subtle tint). Nothing else from Day 1–3.

| Element | Base (source day) | Source file / commit | Shared module now |
|---|---|---|---|
| **Pipeline** | **Day 5 v5** modular locked pipeline (gymform/*, hashed per-step caches, QC gate) | ca6e9f1, 66590f9, 3b584ad | `run_day.py`, `gymform/*` |
| **Studio / look** | **Day 4–5** minimal dark charcoal studio, no props, silver mannequin, crisp (0 bloom/blur) | D4 f63a353/aeeb806, D5 80d99d4 | `gymform/bg.py` |
| **Clip generation** | **Day 5 v5**: Grok Imagine i2v, 720p, 121 f @24 fps, one native clip per step, start still with the anatomy baked in (bone clips = thin glowing upper-body bones over OPAQUE silver skin); locked angle across steps as in Day 4 | `output/assets/day5_lat_step*_720p.mp4` (bone: `day5_lat_step3_bone_v3_720p.mp4`), D4 aeeb806 | config + assets; `xai_edit.acceptance` colour gate |
| **X-Ray look** (from Day 1) | **Day 1 v36** bright near-white luminous bones in a local see-through window, only a soft red (wrong) / soft green (right) tint; never dark saturated fills (t530u) | `output/day1_v36_master_reel.mp4` @1.5 s; module in 80d99d4 | `gymform/xray.py`; QC (g) + **(u)** |
| **Muscle fibre** | **Day 5 v5** localized intrinsic fibre colouring on the target muscle (Day 4 RDL grain as the texture reference) | D5 v5 step 2/4; D4 f63a353 | `gymform/muscle.py` |
| **Silver skin** | **Day 5 v5** deterministic silver lock p75 ~165 / p90 ~191 / max 226 every step | 66590f9 (curve 68d2fb3) | `render.skin_tone_lock`; QC (f) |
| **Shorts** | **Day 5 v5** native fabric only, clean hem/waistband (Day 4 jet-black drawers look) | 3b584ad | `mask.native_shorts`; QC (n)(k) |
| **Framing** | **Day 4–5** full body centred, same mannequin size every step, locked angle, equipment never cut. **No target zoom** in the base (Day 6 zoom = later single change) | D4 aeeb806, D5 66590f9 | `ctx.normalize_scale`; QC (c)(d) |
| **Layout / title / preview** | **Day 4** depth-behind Kaushan title + 4-slot translucent top preview + see-through 0.40 (as in Day 5 v5) | 70e292c, 328c7b4, c607b9d | `gymform/layout.py`; QC (b1)(b2)(h) |
| **Callout / laser** | **Day 5 v5** degree callout + single laser chain, numbers = TTS | ca6e9f1 | `layout`, `callout_lint` |
| **TTS** | **Day 4** lines fitted inside each 5 s step window + Day 6 owner rule (spoken experience question at the end) | e04e274 | `gymform/tts.py`; QC (l) |
| **Tempo** | **Day 5 V3** 1 rep, 1.3 s peak hold: 0.3/1.5/1.3/1.5/0.4 s | 165922d | `gymform/tempo.py` |
| **Caption / step order** | Owner rules from Day 6: 3 English lines, 0 hashtags; 1 wrong bone, 2 right bone, 3 wrong muscle, 4 right muscle | 1eba222, t515u | `caption.py`, `ctx.order_lint` |

## Pair matching (t521u) without video edits
Wrong steps 1/3 and right steps 2/4 must show the same motion. Video edits (Day 6, t522) changed limb shape and
colour, so the baseline uses **Day-5-style generated clips + a joint-tolerance check** instead of pixel identity:
1. Keep the existing native muscle clips as the motion base (Day 6: `day6_idp_step2_muscle_720p.mp4`,
   `day6_idp_step4_muscle_720p.mp4`).
2. Make the bone stills by **image edit** of the base clip's frames (first, 4 keyframes on the 1/3 s grid, last):
   "same opaque silver mannequin, add thin bright near-white upper-body bones with a soft red/green glow".
   Each still must pass the acceptance gate (silhouette IoU ≥ 0.90, bone V50 ≥ 190, S50 ≤ 0.50, white core ≥ 0.10,
   tint hue) **before** any video spend.
3. Generate each bone clip with grok-imagine-video-1.5 i2v, **first + last frame + 4 keyframes pinned** to those
   stills (same duration/fps as the base).
4. Accept only if the joint trajectories match the base: mean joint deviation ≤ 4% of skeletal size per sampled
   frame (QC (s) already measures this on per-clip keypoints when kps are extracted per clip) + `xai_edit.acceptance`.

Expected spend for Day 6 (xAI list prices, docs.x.ai/developers/models, 2026-10-30): image edits 12 × ~$0.08
(input+output, grok-imagine-image-2.0) ≈ $1.0; 2 videos × 5.04 s × $0.08 ≈ $0.81; **≈ $1.8 per attempt**, budget
one retake → **≤ $3.6**. (t522 edits were billed $0.40 per 5 s clip.) No SuperGrok/grok.com quota used on the API route.

## Bone brightness gates (t530u)
- Render QC **(u)**: bone pixels in the X-Ray window L50 ≥ 140, L90 ≥ 190, contrast vs surround ≥ 25,
  dark-saturated share ≤ 0.20. Day 1 v36 scores L50 150–153, L90 200–206, contrast 27–43, dark 0.08–0.14.
- Source acceptance (`xai_edit.ACCEPT`): bone V50 ≥ 190, S50 ≤ 0.50, white core ≥ 0.10, dark fill ≤ 0.12,
  tint hue red ≥168 or ≤12 / green 40–80.

## One change at a time
Day N = baseline + at most ONE new element, listed in the day config `change` field, with a before/after preflight
sheet for the owner. Order of pending improvements: (1) pair matching via keyframe clips, (2) target zoom.
