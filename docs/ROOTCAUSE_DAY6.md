# Day 6 v4 root cause (t593u–t599u): why it looks wrong, and why Day 4 looked right

Owner: "영상을 봐 / Day4도 완벽한건 아니야 / 근본적인걸 아직도 해결못했어". No render and no grok.com were used for this investigation.
Evidence was produced by `tools/rootcause_metrics.py` (numbers) and `tools/rootcause_crops.py` (stills), and is stored in
`output/qc/rootcause_day6/` (`metrics_v4.json`, `metrics_v3.json`, `metrics_day4.json`, and the crops).

**Method.** For every output frame the tool rebuilds the exact raw source frame the renderer used: the same tempo
pose index and the same 1080 transform. Farneback optical flow is computed on the RAW frames and applied to both the
raw and the output frames. Real motion cancels, and what remains is morph, boil or flicker. If the output residual is
higher than the raw residual in a region, our processing created or amplified it.

## TL;DR

The generated clips are mostly fine. Our cut-out pipeline breaks them.

Since Day 5, `gymform` stopped keeping the clip and instead **re-synthesises the picture from per-frame,
keypoint-driven, hard-thresholded masks**: shorts zone, body envelope, temporal-median plate diff for equipment,
metal re-shade, static bench layer, static_body, and the shin capsule. It then composites the result over a
replaced backdrop. On a dark mannequin + black bench + near-black backdrop clip, these masks make structural
errors, and the errors change every frame.

Day 4 had **one soft colour-difference matte and a global tone remap**, on a free-standing bright figure with
backdrop all around. That is why Day 4 looked right. Day 4 still shows the same, smaller temporal amplification,
so "Day 4도 완벽한건 아니야" is confirmed by the numbers below.

## 1. Day 4 vs Day 6 v4, end to end (code-level evidence)

| Stage | Day 4 (`build_day4_rdl_master.py`, commits 328c7b4 / f63a353 / aeeb806, 10-29 00:04–01:58) | Day 6 v4 (`gymform/*`, introduced ca6e9f1 10-29 15:16 + 24 commits) |
|---|---|---|
| Source clips | i2v 480×848 24 fps (`day4_rdl_stepN_i2v.mp4`) → video-edit "ecorche"/"locked" 720×1280 24 fps, 145 frames. Bright silver ecorche, standing, nothing between the legs, barbell outboard. Raw feet-vs-backdrop contrast **33–41 lum** | grok i2v 1088×1920 24 fps, 145 frames → `*_src720` (crop + lanczos). Dark charcoal mannequin, seated on a black incline bench, near-black backdrop (raw backdrop 18–22 lum). Raw feet contrast **22–29 lum** |
| Slice / retime | Own loader, **no frame cap**: slices of 106 / 40 / 73 / 136 frames → 38–48 unique poses, 108–138 pose changes per 150 output frames | `d5.load_native_slice_720p` **caps every slice at 36 frames** (wrong 45 → 36, skipping 9). Gives 36/35 unique poses and 102 changes per 150 frames (24 fps clip shown at 30 fps with uneven holds) |
| Frame blending | None: nearest pose index (`int(round(wave*(n-1)))`) | None: same nearest index (`tempo.pose_index`). **Retime/blending is not a cause of morph or boil** |
| Matte | ONE soft colour matte per frame: `w_fg = blur(clip((|lum-bg_border| + 0.95*act - 6)/14), σ2)`. No keypoint geometry | `mask.body_key` (kp envelope + texture + row backdrop) ∪ `equipment.footprint` (diff vs temporal-median plate) ∪ `mask.native_shorts` (kp zone) ∪ `static_equipment` (plate) ∪ `static_body` (+ kp shin capsule) − near-wrist plate zeroing. All hard per-frame decisions, **no temporal smoothing** |
| Shorts | kp band × dark & low-sat × **w_fg** (needs foreground evidence) | kp zone × dark & low-sat, **no foreground gate**, then forced opaque in `mask.finalize` |
| Backdrop | Clip backdrop is already plain. Replaced softly by the constant (38,40,44) through the same soft matte | Replaced by a plate (v1–v3: vignetted) or flat (v4). Everything outside the union mask becomes backdrop, and everything inside is opaque |
| Equipment | Recoloured with the body by the same remap (no separate layer) | Separate metal mask + `metal_shade` re-shading + native static bench layer |
| Glow | Native clip glow (red/green in the raw clip), recoloured by the fibre shader | Same native glow + `muscle.day4_fibre` shader (glow gain 1.0 on v4) |
| Callout anchor | MediaPipe kp, Gaussian σ = 2.5 | MediaPipe kp, Gaussian σ = 2.2, on a ≤36-frame slice |
| Skin tone | `grey_back_exact` gamma 0.78, floor 42 | `silver_skin` + `skin_tone_lock` (v4: `_flat_skin_lock`) |
| Step cut | Hard cut, each step a different clip/angle | Hard cut, same camera, different generated clip; step N ends at pose n−1 and step N+1 starts at pose 0 |

## 2. Measurements: raw clip vs our output

Values are the mean flow-compensated residual (lum) per region, Day 6 v4 step 1 / step 2. Day 4 is given for comparison.

| Defect (owner list) | Raw → output (Day 6 v4) | Ratio | Day 4 ratio | Where it comes from |
|---|---|---|---|---|
| Fibre "boil" (body interior, no glow, no dumbbells) | 1.28 → 2.03 / 0.76 → 1.08 | **1.59× / 1.42×** | 1.20–1.58× | Ours: `silver_skin` high-pass emphasis (`hf_ridge*0.70 − sulc*11`) on generation micro-noise. Fibre HF std +31% / +20% (Day 4 +21–62%) |
| Edge flicker / cut-out halo (silhouette XOR after flow warp, px/frame at 540p) | 2230 → 3511 / 2212 → 3667 | **1.57× / 1.66×** | 1.56–1.74× | Ours: per-frame binary masks without temporal coherence |
| Glow / colour sliding (glow region residual) | 4.74 → 9.08 / 4.56 → 10.93 | **1.92× / 2.40×** | 1.59–3.55× | Ours: threshold ramp `(act−10)/34 ^0.85` + core seed blur 7 is a contrast stretch of native glow flicker |
| Callout anchor sliding (kp anchor vs an LK-flow-tracked skin point, native slice) | max **30.9 px** / 9.0 px; kp anchor frame step p90 7.6 / 9.9 px | – | – | Ours: the kp track is not locked to the surface |
| Bench seat melting (box between the legs) | 0.30 → 0.48 / 0.29 → 0.44 | 1.6× / 1.5× | 1.2–3.8× | Ours: **38.6% of the shorts-mask pixels are raw backdrop** (and the black bench), forced opaque and repainted as shorts (v3 and v4 identical) |
| Ghost / translucent leg | – | – | – | Ours: **11.9% of the raw object area is raw-backdrop pixels rendered as opaque body**. It includes the straight-edged kp shin capsule (`static_body_shin_r` 40, added in v3), drawn where the keypoints are, not where the leg is (`s2_shin_src_base_wfg_static.jpg`, `s2_bench_between_legs_raw_vs_out.jpg`) |
| Dumbbell plate morph / dumbbell melts into hand | dumbbell box residual 1.89 → 1.70 / 2.06 → 2.16 | 0.90× / 1.05× | – | **Not generation.** The raw dumbbells are rigid in the crops (`s1_right_dumbbell_raw_vs_out.jpg`, `s2_left_dumbbell_raw_vs_out.jpg`). Ours: `metal_mask` (lum < 42) + `metal_shade` turn the shaded faces black, and the plate-diff footprint cuts holes where the metal matches the backdrop. This is the same mechanism as the (d) holes of 94–102 px |
| Forearm thickness / elbow warp | visual | – | – | Ours: `mask.finalize` zeroes the body key near the wrists wherever the frame equals the temporal plate (`d<6`). Where the arm overlaps its own resting position in the median plate it is eaten, and the bite moves with the pose (`s2_elbow_forearm_raw_vs_out.jpg`) |
| Shorts edge flicker | visual + silhouette flicker | – | – | Ours: the kp-zone shorts mask with hem search is re-decided per frame |
| Feet / bench base floating | feet contrast 22.3 → 20.8 / 29.4 → 21.8 (v3 16.8 / 19.2) | – | 33–41 → 17–23 | Both: a low-separation source, and we remove the floor while pasting native bench and floor-shadow pixels as opaque static → black cut shapes on a flat backdrop (`s1_feet_bench_base_raw_vs_out.jpg`) |
| Hard jump at the step cut | wrist **271 px**, shoulder 46 px, ankle 24 px; body lum 52.0 → 57.7; frame diff 6.5 vs typical p90 1.52 (**4.3×**) | – | – | Both: step 1 ends at pose n−1 and step 2 starts at pose 0 (a phase mismatch by construction), the clips are generated separately (bench drawn differently), and the per-step transform is aligned on the shoulders (scale 0.829 vs 0.822, dx 18 vs 29, dy 337 vs 329) |
| 3D form flattened | form std (σ8 low-pass) 25.5 → 18.6 / 16.7 → 13.0 (−27% / −22%) | – | −6 to −43% | Ours: `silver_skin` remap. **Not `flat_look`**: v3 gives 18.8 / 12.9 and v4 gives 18.6 / 13.0, and boil is 2.03 in both. The v4 skin lift did not create the boil or the flattening |

Caveats: the per-frame wrist-box area metric (raw 4–6% vs output 4.1–4.6%) is confounded by the hand moving
through the box, so it is not used as evidence. The visual raw-vs-output crops are decisive for the dumbbells. The
glow-centroid offset (raw ~60 px std) is dominated by the arm moving relative to the chest, so I use the residual
and the flow-tracked anchor instead.

## 3. Ranked root causes

1. **Geometry is re-synthesised by per-frame heuristic masks and composited over a replaced backdrop** (shared
   modules `mask`, `equipment`, `bg`, `render.shade`). This explains the ghost leg, bench melt, shorts slab, dumbbell
   holes and black faces, eaten forearm, edge flicker (+57–66%) and floating feet. Evidence: 38.6% of the shorts
   mask is backdrop, 11.9% of the object area is backdrop painted as body, and the crops. *Fixable in shared modules.*
2. **Source scene with low figure-ground separation**: a dark mannequin, black bench and near-black backdrop
   (feet contrast 22–29 vs Day 4 33–41), and a seated pose with the bench between the legs. This is why any
   colour-difference matte is ambiguous here. *Source decision.*
3. **The shader amplifies the clip's temporal noise**: glow ×1.9–2.4 and boil ×1.4–1.6 (fibre HF +20–31%).
   *Fixable in shared modules.*
4. **Step cut by construction**: a phase mismatch (wrist jump of 271 px), clips generated separately, and
   shoulder-only alignment. *Fixable in tempo/ctx, plus a source decision (both clips from one start frame/set).*
5. **Callout anchor not locked to the surface** (≤31 px drift vs flow). *Fixable (layout).*
6. **36-frame slice cap** (minor judder; not a morph cause). *Fixable (loader).*

## 4. Fix plan, in order of impact (not implemented, no render; each step needs its own preflight + owner look)

1. **Keep the clip; no cut-out.** New `look: "native_grade"` path. The whole frame is tone-mapped globally (Day 4
   style). The backdrop is flattened by a smooth gain field, `target_bg / blur(backdrop estimate, σ≈60)`, applied to
   every pixel, so no matte edge exists. Bench, dumbbells, shorts, feet and floor contact keep their own pixels.
   This drops shorts/static/static_body/shin capsule/metal re-shade/near-wrist zeroing from the look path.
   Expected: silhouette flicker ≤1.1× raw, and 0% backdrop painted as shorts or body.
2. **If a matte is still required** (glow gating, UI depth): compute it once per pose and make it temporally
   coherent (flow-warp the previous alpha + EMA 0.6). Every hard threshold becomes a soft ramp. The shorts mask must
   be gated by foreground evidence (Day 4's `× w_fg`).
3. **Shader without noise amplification**: drop the `hf_ridge`/`sulc` emphasis in `silver_skin`, use a linear glow
   recolour (no ^0.85 ramp or seed blur), and apply a flow-guided temporal EMA on the glow weight. Targets: boil
   ≤1.15× raw, glow ≤1.3× raw.
4. **Step cut**: phase-matched cut. The wave continues across steps, so step N+1 starts at the pose where step N
   ended. Align steps on hips/ankles/bench rather than shoulders, and match the body-mean gain across the cut.
   Target: wrist jump ≤40 px and cut frame diff ≤2× the typical p90.
5. **Anchor**: an LK-flow-tracked anchor seeded from the kp, with a kp re-seed when it is more than 12 px apart.
   Target: drift ≤8 px.
6. **Loader**: remove the 36-frame cap (Day 4 had none).
7. **Source (Day 7+)**: a brighter silver body and a backdrop clearly different from the bench. Generate both clips
   from the same start-frame set and camera.

QC checks to add for each item are defined in [`DEFECT_PLAYBOOK.md`](DEFECT_PLAYBOOK.md) and are measurable today with
`tools/rootcause_metrics.py`.
