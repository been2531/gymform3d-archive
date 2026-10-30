# @gymform.3d BASELINE TEMPLATE (t530u–t532u)

Owner (t530u–t532u): Day 6 fell apart because too much changed at once. **Build from what already worked in the
posted Day 1–5 reels, then change ONE element per day** (each change = its own preflight + owner look).
Contact sheet of the chosen frames: `output/baseline_best_frames.jpg`.

Posted Day 1–5 (exercise_curriculum.json): D1 `output/day1_v36_master_reel.mp4` (De_Yl9rAYC4) · D2
`output/day2_v38_master_reel.mp4` (De_mCcFAVTO) · D3 `output/day3_barbell_squat_simple2_master.mp4` (DfCpKW4IIV8) ·
D4 `output/day4_romanian_deadlift_4step_master.mp4` (DfC8EifoxFa) · D5
`output/day5_dumbbell_lateral_raise_4step_master_v5.mp4` (DfFPS3YFLF3). Day 6 (DfGabTBjwt8) deleted, t529u.

| Element | Best version (source day) | Source file / commit | Shared module now |
|---|---|---|---|
| **X-Ray look** | **Day 1 v36**: local see-through window, **bright near-white luminous bones**, only a soft tint (warm red wrong / soft green right). Never dark saturated fills (t530u). | `output/day1_v36_master_reel.mp4` @1.5 s, `build_v36_publish_reels.py`; generalised in 80d99d4 | `gymform/xray.py`; QC (g) look + **(u) bone visibility** |
| **Bone clip source** | **Day 5 v5**: native Grok Imagine i2v clip whose mannequin already shows thin glowing upper-body bones over OPAQUE silver skin (no edit, no overpaint). | `output/assets/day5_lat_step3_bone_v3_720p.mp4`, 66590f9 / 3b584ad | `configs/dayNN.json` clips; `gymform/xai_edit.acceptance` colour gate |
| **Muscle fibre** | **Day 5 v5** localized intrinsic fibre colouring (only the target muscle, striations visible); glow strength ref = Day 2 v38 | `output/day5_..._v5.mp4` step 2; D2 `build_v38_high_contrast_hud.py` | `gymform/muscle.py` |
| **Silver skin** | **Day 1/2 bright silver-grey curve** locked p75 ~165 / p90 ~191 / max 226, applied deterministically every step (Day 5 v5) | 68d2fb3 → 66590f9 | `render.skin_tone_lock`, spec `mannequin.skin_lock_all_steps`; QC (f) |
| **Shorts** | **Day 5 v5** native fabric only, clean hem/waistband by construction | 3b584ad | `mask.native_shorts`; QC (n)(k) |
| **Framing** | **Day 5 v5** full body centred, same mannequin size every step (pose-invariant skeletal size), equipment never cut. **No target zoom** in the baseline (Day 6 zoom 1.35 = a later single change). | 66590f9 | `ctx.normalize_scale`; QC (c)(d) |
| **Layout / title / preview** | **Day 4 Option 3** depth-behind Kaushan title + 4-slot translucent top preview, Day 4 see-through 0.40 over body/equipment (as in Day 5 v5) | 70e292c, 328c7b4, c607b9d | `gymform/layout.py`; QC (b1)(b2)(h) |
| **Callout / laser** | **Day 5 v5** degree callout + single laser chain, numbers = TTS | ca6e9f1 | `layout.callout_for`, `callout_lint` |
| **TTS** | **Day 4** lines fitted inside each 5 s step window (e04e274) + Day 6 owner rule: spoken experience question at the end | e04e274; Day 6 rule | `gymform/tts.py`; QC (l) |
| **Tempo** | **Day 5 V3** controlled 1 rep with 1.3 s peak hold: 0.3/1.5/1.3/1.5/0.4 s | 165922d | `gymform/tempo.py` |
| **Caption** | Day 6 owner format: 3 English lines, 0 hashtags | 1eba222 | `gymform/caption.py` |
| **Step order** | Day 6 owner lock: 1 wrong bone, 2 right bone, 3 wrong muscle, 4 right muscle (cover = step 4, 3/4 view) | t515u | `ctx.order_lint` |
| **Clip generation** | **Day 5 v5 route**: Grok Imagine i2v, 720p, 121 frames @24 fps, one clip per step, start still with the anatomy rendering baked in, 3/4 view | 66590f9 | config + assets; QC native source |

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
