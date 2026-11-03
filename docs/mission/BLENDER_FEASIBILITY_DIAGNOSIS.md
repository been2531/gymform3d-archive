# Blender feasibility diagnosis (Round 4)

**Scope:** Evidence-only answers to (1) whether the **Day 4 look** is achievable in Blender, and (2) **root causes** of three failed pilot rounds. No pilot look iteration in this round.

**Owner observation (v3 reel / attached sheet) is correct:** side view reads as a **horizontal press/row**, body reads **smooth**, glow is **neck-sized patches**, machine is **boxes**. The Round 3 “reel-usable” verdict was **wrong**.

---

## Executive answers

| Question | Answer |
|---|---|
| **(1) Is Day 4 look achievable in Blender (Cycles/Eevee) as shipped in this repo?** | **No** — not to match the locked Day 4 master without a **hybrid** path. |
| **(2) Root cause of three failed rounds?** | Chasing Day 4 **2D generative grade** with **wrong asset class** (BP3D part meshes + MH skin), **wrong pose model** (FK angles ≠ lat pulldown path), **procedural placeholders**, and **QC that does not measure look or pose semantics**. |

---

## What “Day 4 look” actually is (repo evidence)

Day 4 is **not** a native Blender render product in this codebase.

- **Gold master:** `output/day4_romanian_deadlift_4step_master.mp4`, built by `build_day4_rdl_master.py` (`gold.json`, id `day4_rdl_master`).
- **Fibres + glow:** `shade_day4_3d_frame_1080p()` in `build_day4_rdl_master.py` (from line 63) operates on **RGB video frames** — it derives foreground, shorts, and **fiber ridges from blurred channel differences** on **generative source pixels** (e.g. `fiber_fine = (r_ch - r_blur_mid) * 1.85 + …`, lines 129–137). That is **image-space stylisation** of an already-detailed source clip, not mesh UV striations.
- **Production muscle steps** reuse the same shader via `gymform/muscle.py` → `day4_fibre()` (“line for line from `shade_day4_3d_frame_1080p`”, lines 62–63).

So the benchmark look couples **high-frequency detail in source video** + **monotonic skin lock / activation grading**. The Blender pilot never had an equivalent source signal.

---

## 1. Root-cause analysis per defect

Classification: **bug** = implementation error; **wrong method** = approach cannot reach target; **asset limit** = data/model insufficient.

### Wrong motion (side view = horizontal press / row, not pulldown)

| Version | What you see | Root cause | Class |
|---|---|---|---|
| **v1** | Standing / generic lean; bar plates floating in side view | `pose_lat()` in `studio3d/pilot/build_scene.py` (legacy) aims bones from **scalar angles** only; `make_bar()` placed bar from wrists **after** pose without seated leg model or cable path; end **plates** on bar (`make_bar`, ~lines 227–255 in same file pre-v3). | Wrong method + bug |
| **v2.1** | Fragmented torso; arms not on bar; no seated machine read | `body.hide_render = True` (e.g. commit `fb767de`, `build_scene.py` ~399); muscles imported **before** seat/pose order issues; rigid **bone parenting** without body (`anatomy_ecorche.py`) → parts float. | Wrong method |
| **v3** | **Contract-only** side reel (`assemble_reel.py` lines 21–25: steps 3–4 are `*_contract_side` at **phase 1**); arms driven by `pose_arms_from_standards()` (`pose_seated_lat.py` 180–199) using `upper_arm_forward_deg` + `elbow_flexion_deg` from `wide_grip_lat_pulldown.json` — **no vertical bar path**, **no overhead stretch** in the 4-step reel. Bar anchored at **wrist height** (`build_scene.py` 406–412) with only 35% z nudge toward `bar_y_m`; cable attaches to **chest-high** point (`equipment.py` `make_lat_machine`, cable to `bar_attach`). Side camera + horizontal upper-arm component (`sin(fwd)` in line 191) → reads as **horizontal push**. | Wrong method |

**Evidence (meta):** v3 `right_contract_side` reports elbow ~92° and grip OK while visually mimicking a press — QC never checks bar vs head vertical offset or pulldown phase.

---

### Smooth body / no visible fibres

| Version | Cause | Class |
|---|---|---|
| **v1** | Visible MH body + `mat_fibre_skin()` (`build_scene.py` 88–118): **world-space Noise + Wave** on MH UVs; bump strength **0.22**, scale **48** (`set_pilot.json`). Reads smooth at 1080p. | Wrong method |
| **v2.1** | Body **hidden**; fibres only on écorché with procedural UV waves (`shaders.py` 30–73). BP3D parts **low sculpt** + faceted. | Asset limit + wrong method |
| **v3** | **Full MH body visible** again (`build_scene.py` 431–437) with same `mat_fibre_skin`; écorché overlay **minority of pixels** in front view; subsurf on mannequin (`studio3d/build_scene.py` 111) smooths silhouette. Day 4 fibres are **post-process on video**, not this shader. | Wrong method + asset limit |

---

### Small glow (neck patches only)

| Version | Cause | Class |
|---|---|---|
| **v1** | `make_glow_shell()` vertex-group **proxy** on MH body (`build_scene.py` 280–305), opaque emission mix — not muscle geometry. | Wrong method |
| **v2.1** | Glow only on matched **mesh names** (`anatomy_ecorche.py` `glow_targets`); lat meshes **small in screen space** frontally; ID pass `glow_px` threshold **350** (`qc_pilot.py` 68–71) passes tiny areas. | Asset limit + QC gap |
| **v3** | `muscle_glow_pattern()` (`anatomy_bind.py` 116–119) lights **latissimus / teres / infraspinatus** meshes only; on side view **trap/delt/neck** pieces dominate glow bbox; beauty pass: **`right_contract_side` green ≈ 2408 px** vs proposed **8000** min (`diagnosis/qc_gap_analysis.py` output). Glow strength on **few small materials**, body unlit. | Asset limit + wrong method |

---

### Box machine

| All | `equipment.py` builds **`primitive_cube_add` / `_box()`** only (lines 10–16, 19–63); no rigged CC0 lat station found (stated in `PILOT_BLENDER_LATPULLDOWN.md`). Cable is thin boxes; pulley a cylinder. | Wrong method (explicit placeholder) |

---

### Muscle seams

| v2–v3 | Écorché = **hundreds of separate OBJ-derived meshes** (634 in feasibility import); **global/regional scale** (`anatomy_bind.py` 22–74) + **bone parenting** (104–111) without continuous skin wrap → **inter-part gaps** at pose. Surface-deform experiment (Round 3 early) caused **spikes / missing limbs** — removed. | Asset limit + wrong method |

---

### Limp arms (v2)

| v2.1 | `import_upper_muscles()` **before** `pose_lat()` in `fb767de` `build_scene.py`; muscles parented to bones with **rest bind**; `pose_lat` uses `aim()` on arms but **no IK / grip constraints**; bar placed at wrist midpoint **after** pose (`equipment.py` 56–64) — arms not pulled to bar. | Bug + wrong method |

---

### QC passing broken frames

| Check | What it measures | Why v3 passes broken reels |
|---|---|---|
| `backdrop_flat` | Corner RGB span (`qc_pilot.py` 36–48) | Backdrop is fine even when subject is wrong. |
| `separation` | Luminance skin vs bg (`51–65`) | Smooth grey body still separates from bg. |
| `glow_area` | Magenta **ID pass** on glow-tagged **meshes** (`68–71`, `155–183`) | Counts **mesh projection**, not perceptual muscle area; side lat ID **>350 px** while beauty glow **tiny**. |
| `angles` | Trunk lean only from **3D meta** (`74–88`) | Never validates **pulldown semantics** (bar above head at stretch, vertical pull). |
| `hand_bar_clearance` | Wrist-to-bar segment distance in **rig space** (`140–147`, meta `grip_ok`) | Passes when constraints **snap** hands (~0–7 cm) even if pose looks like press. |
| `deterministic` / `temporal` | Pixel stability | Irrelevant to look/pose. |

**Diagnostic script:** `studio3d/pilot/diagnosis/qc_gap_analysis.py` → `output/pilot/diagnosis/qc_gap_metrics.json` shows v3 **would fail** proposed gates: green glow **<8000 px**, side frames **flag horizontal bar** (Hough lines in upper frame).

**Checks that would catch failures:**

1. **Pose vs standard (2D):** MediaPipe / OpenPose on render — wrist above shoulder at stretch; bar y decreases on concentric; elbow travels **down-back** not forward press (compare to `configs/motion_standards.json` `wide_grip_lat_pulldown`).
2. **Glow area vs Day 4:** Reference mask from Day 4 activation ROI (lat bbox / % frame area); fail if `<10%` of reference or glow centroid outside lat trapezoid.
3. **Fibre detail:** Laplacian variance on skin mask — pilot passes **~1800–2100** on smooth MH; Day 4 ROI typically higher **multi-scale** energy; combine with **high-frequency ratio** vs blurred self (mirroring `shade_day4` ridge extract).
4. **Silhouette / machine:** Depth-free heuristics — overhead cable segment visible above head; `% frame` occupied by machine mesh ID; fail if bar–shoulder Δy `< threshold` at stretch keyframe.
5. **Semantic reel audit:** 4-step reel must include **stretch phase 0** for side/front, not **contract-only** (`assemble_reel.py` design bug).

---

## 2. Feasibility proof

### 2a. Isolated asset render (this repo)

**Script:** `studio3d/pilot/diagnosis/render_feasibility_ecorche.py`  
**Output:** `output/pilot/diagnosis/feasibility_ecorche_static.png` (artifact: `/opt/cursor/artifacts/screenshots/feasibility_ecorche_static.png`)  
**Asset:** Charleeh **écorché** glTF (CC BY-SA 4.0), 634 meshes; pilot `mat_muscle_fibre` + emerald on **latissimus** only (2 meshes).

**Result (honest):** Static import shows **faceted, disjoint BP3D-derived parts**, weak striations, **no** Day-4 continuous écorché read, **no** gym pose. This is an **asset + shader ceiling** sample, not a success.

**Best free sources considered (unchanged from pilot research):**

| Source | Detail | Licence | Fit |
|---|---|---|---|
| Charleeh/écorché | Segmented muscles; filled lats | CC BY-SA 4.0 | Best **OSS** rigged topology we used; still not sculpted fibre |
| Z-Anatomy | High detail Blender template | CC BY-SA 4.0 (+ NC parts) | Better **if** owner approves SA; not tested in pilot |
| BodyParts3D partof | Per-muscle OBJ | CC BY-SA 2.1 JP | Same lineage as écorché; no continuous surface |
| Sketchfab / Rig Manual écorché | Textured ~66k tris, rigged (see [YouTube Rig Manual](https://www.youtube.com/watch?v=qrGYBR_iO4I)) | Often **CC-BY**, check per asset | **Commercial-style** reference; not CC0 in repo |

No CC0 écorché with **Day-4-grade albedo/normal** was integrated.

### 2b. Public Blender examples (écorché + fibres + glow)

These show **pieces** of the stack, not the locked Day 4 master:

1. **[Blender Artists – Muscle study](https://blenderartists.org/t/muscle-study/635637)** — Cycles, sculpted mesh + **3DCoat paint**; SSS + roughness tuning. **Technique:** hand-authored mesh/textures, not procedural BP3D.
2. **[Blender Manual – Subsurface Scattering](https://docs.blender.org/manual/en/4.2/render/shader_nodes/shader/sss.html)** — SSS for organic tissue; Day 4 uses **different** 2D grade, but SSS is relevant for **realistic** muscle (not the flat Day 4 studio mannequin).
3. **[Reddit r/blenderhelp – emission + transparency](https://www.reddit.com/r/blenderhelp/comments/1n7v3ii/how_do_you_make_renders_like_this/)** — **Fresnel-weighted emission** + compositor layers for anatomy under skin. **Technique:** dual-layer render, not single BP3D stack.
4. **[Rig Manual écorché rig (YouTube)](https://www.youtube.com/watch?v=qrGYBR_iO4I)** — Rigged low-poly écorché with **bitmap muscle textures** and shrink/slider; **technique:** authored asset + texture swap.
5. **MyoGeneratorRemix (GPL)** — Procedural muscle **geometry** + Voronoi fibres; cited in pilot doc; fibres are **not** Day-4 photographic.

**Conclusion on gap:**

| Layer | Limit |
|---|---|
| **Asset** | BP3D/écorché **part meshes** ≠ continuous sculpted écorché with painted fibres. |
| **Shader** | Procedural waves **≠** Day 4 `shade_day4` multi-scale ridge extraction on **generative** input. |
| **Rig/pose** | FK angle lerp **≠** lat pulldown **spatial** path (overhead cable, thigh lock). |
| **Fundamental** | Day 4 lock is **video + 2D grade**; Blender-only path lacks the **source detail** the grade assumes. |

---

## 3. Three rounds — systemic failure mode

| Round | Stated goal | Actual bet | Failure |
|---|---|---|---|
| **v1** | Deterministic Day 4 | MH + procedural fibre + **glow shells** | Optimised **QC/determinism**, not look; rejected same as earlier studio3d smooth mannequin. |
| **v2/v2.1** | Écorché + fibres | Hide body; bone-parent BP3D parts | **Asset topology** + wrong A/B reference; scale/fit bugs; still not motion-complete. |
| **v3** | “Reel-usable” | Visible body + seated pose + boxes | **Declared success on QC** without visual/pose gates; side reel **contract-only** encodes wrong exercise read. |

**Common thread:** Treating Day 4 as a **Blender materials problem** when production Day 4 is **generative video + `shade_day4_3d_frame_1080p` compositing** (`build_day4_rdl_master.py`, `gymform/muscle.py`).

---

## 4. Verdict and realistic paths

### Is Day 4 look achievable in Blender?

- **Match locked Day 4 master (fibres + glow + skin lock) in pure Blender from OSS assets:** **No**, with evidence above — unless “Day 4” is redefined as “similar anatomy illustration,” not pixel-matched master.
- **Achievable in Blender for anatomy education quality (not pixel Day 4):** **Yes, with different inputs** — e.g. purchased/rigged textured écorché (Rig Manual–class), hand-painted normals, mocap lat pulldown, modelled machine asset; **weeks** of artist time, not current pilot automation.
- **Achievable matching Day 4 for reels:** **Hybrid yes** — Blender (or mocap) for **silhouette, machine, motion**; **existing Day 4 grade** (`shade_day4` / `gymform/muscle.day4_fibre`) on **generative or high-res plate**; same as rest of product line.

### Effort (order-of-magnitude, engineering + art)

| Path | Buy? | Effort | Outcome |
|---|---|---|---|
| **A. Fix motion + machine only in Blender** (mocap/IK, CC-BY machine mesh, keep écorché) | No | **~40–80 h** eng + **~20–40 h** art | Correct **lat pulldown read**; look still **not** Day 4. |
| **B. Premium écorché + texture paint + Cycles** | Optional asset **~$50–200** | **~80–160 h** art-heavy | Strong **3D anatomy** reel; may approach but not match Day 4 lock without grade. |
| **C. Hybrid (recommended for Day 4 parity)** | No new asset if generative pipeline exists | **~16–40 h** pipeline | **Geometry/motion** deterministic; **look** from proven `shade_day4` on source frames. |

**Recommendation:** Stop iterating pilot shaders for Day 4 parity. Either **(C) hybrid** aligned with `gold.json`, or **(B)** with a new visual target **explicitly not** pixel-matched to Day 4.

---

## Artifacts (Round 4)

| File | Purpose |
|---|---|
| `studio3d/pilot/diagnosis/render_feasibility_ecorche.py` | Isolated écorché + shader frame |
| `studio3d/pilot/diagnosis/qc_gap_analysis.py` | Metrics QC should use |
| `output/pilot/diagnosis/feasibility_ecorche_static.png` | Feasibility render |
| `output/pilot/diagnosis/qc_gap_metrics.json` | v3 keyframe metric demo |
| This document | `docs/mission/BLENDER_FEASIBILITY_DIAGNOSIS.md` |

---

## References (code)

- Day 4 grade: `build_day4_rdl_master.py` (`shade_day4_3d_frame_1080p`, from line 63)
- Pilot pose: `studio3d/pilot/pose_seated_lat.py` (`pose_arms_from_standards`, lines 180–199)
- Pilot QC: `studio3d/pilot/qc_pilot.py`
- 4-step reel poses: `studio3d/pilot/assemble_reel.py` (lines 21–25)
- Exercise numbers: `studio3d/pilot/exercises/wide_grip_lat_pulldown.json`
