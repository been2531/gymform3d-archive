# D7 Seated Cable Row: why the motion collapses, and the fix (t690u–t691u)

Sheet: [`ROOTCAUSE_D7_MOTION_sheet.jpg`](ROOTCAUSE_D7_MOTION_sheet.jpg)
- Frames are read from the existing clips. Builder: `tools/rootcause_d7_motion_sheet.py`.
- 0 generations, 0 renders.
- Lighting (blocker 1) is treated as solved: use the verified real-machine plate with flat light, as on the glass stills. This doc covers motion (blocker 2).

## 1. Root cause
1. **The video model redraws every pixel of every frame. It has no notion of the machine's rules:**
   - the cable runs from the low pulley to the hands;
   - the hands stay on the handle;
   - the feet stay on the plate and the seat stays still;
   - the row travels along the rail axis.
2. **The longer the unpinned stretch, the further it drifts.** v1 ran 6 s with no pins and drifted into a fly and then a pulldown. v3 ran 5 s with only the ends pinned and gained an extra half rep plus a torso rock in the middle.
3. **Pinning works:** the verified real plate as the pinned first and last frame, with a static camera (v3), held the machine at SSIM 0.99 and kept the cable on the low pulley. What collapses is the free middle of the clip, plus details baked into the pins, such as the rectangular handle.

## 2. Catalogue of every D7 attempt (evidence: the sheet, gate JSONs and docs)
| # | Attempt | Setup | What failed (frame evidence) | Likely cause |
|---|---|---|---|---|
| 1 | grok.com v1 (`output/assets/day7_{wrong,right}_raw_v1.mp4`, 6 s, 145 f) | i2v from a start frame; machine beside the mannequin; front-3/4 view | Wrong: f41 arm spread like a fly, f102 handle overhead like a pulldown. Right: f61 one-hand split handle and green over the whole torso (glow 2.4% → 18.9–22.5%). Backdrop SSIM min 0.87 | The cable direction isn't readable in the start frame (the machine sits beside the body, not in front). 6 s free with no pins. The prompt alone can't hold the exercise identity |
| 2 | grok.com v2 + raw (`*_raw_v2.mp4`, `*_raw.mp4`) | Front view, low-row station with a flywheel box | The pull runs along the camera's depth axis, so the motion barely shows (f0/f82/f102 nearly the same). Invented machine, not the Matrix. Shading sharpness Laplacian 52 | Camera along the movement axis. The machine is invented, not a real plate |
| 3 | GenerateVideo smoke t1/t2 + D7v2 (`output/_consistency/smoke_d7*/`) | first = last = start pin + contraction keyframe, 5 s, 480p then 720p | Machine and backdrop stable (G1/G5 pass), but the motion is shallow: elbow 43–59°, against ≥ 80° needed. The wrong smoke clip did 2 reps despite asking for "1 rep" (elbow series peaks at ~f20 and ~f90). The machine still doesn't match the real one | Pins hold appearance, not mechanics. 5 s for a 1.5 s pull leaves the model filler time, which it spends on extra reps. Invented machine |
| 4 | GenerateVideo v3 (`output/assets/day7_{wrong,right}_raw_v3.mp4`, 5 s, 121 f) | Verified real-machine plate, first = last pinned, rear-3/4 | Machine held (`gates_v3_extra.json`: mean\|diff\| ≤ 1.67, SSIM 0.99) and the cable comes from the low pulley. In the free middle: torso range 19–23° (G9 `torso_still` fail, wrong `lean_back_rock`), a half extra rep, and right f60–f80 arms extending past the start pose (elbow 48 → 17°). Foot track flagged (ankle displacement 0.37 / 1.74 shin, partly a keypoint artefact under the plate). Handle rectangular, not V (it came from the pin). 720p cap makes the figure small (2.3× upscale) | Same as #3: the free middle. The handle shape was wrong in the pin itself. 720p is the cap for first + last frame mode |
| 5 | Relight (`output/_consistency/d7m/relight/`, `*_raw_v3_relit.avi`) | Code relight of the v3 clips | Only the shaded side gets lifted. The motion is unchanged, and lifting in post can't add the missing form | Lighting belongs in the source (flat light), not in post |
| 6 | Blender pilots 1–6 (branch `cursor/blender-lat-pulldown-pilot-6c8f`, `BLENDER_FEASIBILITY_DIAGNOSIS.md`, `SMOKE_PASS_FAIL_R6.md`) | bpy 5.2.3 headless, MakeHuman CC0 + BodyParts3D, hand-written IK, lat pulldown | R1–R3: press-like path, smooth body, neck-sized glow, box machine. R5: arms floating (wrist chain broken by 0.287 m). R6: crossed forearms, a 0.21 m shoulder gap, the cable not on the bar | Hand-rolled IK on a MakeHuman rig; procedural box equipment. The look can't reach Day 4 natively. The owner rejected the smooth body (t569u) |

**Days 1–6 worked, and why:**
- **D4 RDL, D5 lateral raise, D6 incline press, D3 squat (no rack):** the equipment is a free weight that moves rigidly with the hands. There's no static structure that has to stay connected to moving hands.
- **D1 pushdown, D2 lat pulldown:** the machine is a backdrop or cropped, and nobody checked it against a real model. The motion is large, vertical, in the image plane, and very common in training data.

**D7 is the first day that needs all of these at once:**
- an exact real machine that is static;
- moving hands on a small V handle;
- a cable that must join the two;
- feet on a plate and a fixed seat;
- a motion along the rail;
- elbows passing behind the torso, which hides them behind the body.

## 3. Fundamental options
| | A. Person only, composited onto the static plate | **B. 2 s pinned start→peak clip, return = code reverse** | C. Code puppet from 2 sculpted stills | D. Blender skeleton driving C | E. Motion transfer (Kling 3.0 Motion Control via the connected Higgsfield plugin) |
|---|---|---|---|---|---|
| How | Generate the row on a flat backdrop (no machine), matte it once, composite onto the verified plate; handle and cable drawn by code | `GenerateVideo` image = start pin, last frame = peak pin, 2 s, 720p, no free time. Code plays forward, holds the peak 0.3 s, then plays reversed (the return of a row is the pull backwards). One rep by construction, seamless loop | Start + peak sculpted stills on the plate; per-part layers (upper arm, forearm, torso) warped along a 2D IK trajectory; cable and handle drawn by code from the low pulley to the hands | Rig in bpy to produce the joint trajectory for C | A real seated-row reference video drives the sculpted still |
| Motion realism | Good (generated), but nothing stops a drifting pose | Good: 2 s gives the model no time to add reps or rock; both ends are real poses | Medium–low: a 2D warp can't redo the elbow passing behind the torso or the scapula changing shape; risks the "slideshow/puppet" look the owner already rejected | Same as C. The R6 arm crossing shows the rig is the hard part, and a planar row only needs 2D IK | High (real human motion) |
| Machine fidelity | 100% plate pixels; but contact breaks: hands vs handle, feet vs plate, seat occlusion | v3 evidence: SSIM 0.99 with the plate as both pins; both ends are pins | 100% | 100% | Unknown: the model redraws the frame (scene from the image) |
| Muscle look (sculpted + Day 4 grade) | Day 4 grade on the person layer; matte edge = DEFECT_PLAYBOOK D4 halo/flicker (the no-cut-out rule); a green-screen backdrop clashes with the green glow, so it needs a flat grey and a learned matte | Unchanged native Day 4 grade on clip pixels (no cut-out); the look comes from the pins | Graded once per still; fibres get warped, which means stretching | Same as C | Re-rendered by the model, so fibres may boil |
| Generations / reel | 2 videos + 2–4 images | Per form: 1 video (+ 2 images when new pins are needed) → 2 videos + ≤ 4 images per reel | 0 videos, 2–4 images | 0 videos | 2 paid Kling runs + a driving clip |
| Time per reel | Dev 2–3 days; then about 15 min | Dev about 0.5 day (reverse/hold/loop + gates already exist); then generation wait + about 3 min grade | Dev 2–3 days; then about 2 min | Dev 3+ days | Minutes, but paid |
| Reuse | All machine days | All machine days: any exercise whose return is the pull reversed (rows, pulldowns, presses, curls) | All 2D days | Low | All, if the budget is approved |
| Risk | High (contact, matte) | Medium: the in-between may still twist the arms; the handle must be fixed in the pins | High (looks fake) | High | Needs owner approval for paid credits; the driving video's rights and camera match |

## 4. Recommendation: **B** (fallback: C with the same pins, zero generation)
**Why B:**
- It removes the measured failure (the free middle of a long clip).
- It keeps what already worked: a real plate as the pin, a static camera, the native Day 4 grade.
- It costs the least and is reusable for every machine day.

**Smallest test: 1 GenerateVideo, 0 images; needs 중현 님's approval.**
- **Input stills:** the existing RIGHT pins, which share the camera (tower SSIM 0.992, rail 0.986 between them):
  - start: `output/_consistency/d7m/pin_right_fibre_head_t2_720x1280.png`;
  - peak: `output/_consistency/still_motion/reel/src_right_peak_stackfix_720.png`.
- **Settings:** 2 s, 720p, `image_path` = start, `last_frame_path` = peak, static camera, brief prompt ("one smooth seated cable row pull, torso still, elbows straight back along the sides, feet on the plate, camera locked").
- **Code step:** forward → 0.3 s hold → reverse → loop; no grade in this test (motion only).

**Pass criteria, by eye, frame strip + loop:**
1. The machine doesn't change: tower, stack, rail, seat, foot plate. Also checked: machine SSIM vs the plate ≥ 0.98.
2. Both hands stay on the handle in every frame. The handle doesn't split.
3. The cable is one straight line from the low pulley to the handle.
4. A natural row: one pull, torso still (range ≤ 10°), elbows travel straight back close to the sides, no arm crossing or twisting, no fly or pulldown drift.
5. The loop seam is invisible.

**After it passes:**
- New pins with the sculpted Day 4 body + flat light + a V handle (2 GenerateImage per form).
- Then the wrong form, using the same method.

If it fails on point 2 or 4, switch to C with the same two pins (0 generations).

Still open, not part of this test: the V handle (fix in the pins) and the peak pose (elbows past the back line).
