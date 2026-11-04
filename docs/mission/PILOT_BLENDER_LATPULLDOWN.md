# Pilot: Blender Day-4 set — Wide-Grip Lat Pulldown

## Summary (v3)

Deterministic headless-Blender pipeline for **wide-grip lat pulldown** (`studio3d/pilot/`, version **pilot-3**).

**v3** targets a **usable reel frame**: full **MakeHuman** body (hands, feet, faceless head, neck, matte shorts) as dark fibre under-layer; **écorché/BP3D-derived** muscle overlays (per-region scale, bone parenting); **seated** pose from motion standards; **bar + partial hand lock**; **complete procedural lat station** (seat, thigh pad, frame, pulley, cable, palette-A stack). A/B references use **cropped Day 4 filmstrip** panels (180px-wide), not the full 2700×320 strip.

Deliverables: `output/pilot/lat/` (keyframes, `ab_keypose_vs_day4_v2.jpg`, `test_reel_4step.mp4`, `qc_report.json`).

---

## Source shortlist (unchanged — see git history)

Primary anatomy: **[Charleeh/écorché](https://github.com/Charleeh/ecorche)** (CC BY-SA 4.0, owner approval pending), upstream **BodyParts3D** (CC BY-SA 2.1 Japan). Rig/mesh: **MakeHuman CC0**. Equipment: procedural (no CC0 lat station found). Shaders: custom `shaders.py`.

---

## v3 architecture

| Piece | Module |
|---|---|
| Rig + visible body + shorts + head | `studio3d/build_scene.py` (`make_mannequin`, `make_shorts`) |
| Seated pose + arm angles + grip | `pose_seated_lat.py` |
| Muscle import + region fit | `anatomy_bind.py` (+ filters in `anatomy_ecorche.py`) |
| Fibre + glow | `shaders.py` |
| Lat station + bar | `equipment.py` |
| QC / A/B / reel | `qc_pilot.py`, `make_ab_sheet.py`, `assemble_reel.py` |

---

## Round 3 self-review (visible defects addressed vs v2.1)

| Defect (v2.1) | v3 status |
|---|---|
| Arms limp / no grip | Motion-standard arm FK + bar at wrists; **COPY_LOCATION** on hand/wrist (~62% influence) |
| No hands/head/feet | **Body visible**; MH mesh includes limbs; faceless egg head |
| Muscle gaps | Body fills silhouette; muscles overlaid (some **shoulder/neck seams** remain) |
| Shorts missing | `make_shorts` rendered |
| Standing / no machine | **Seated** legs; seat, thigh pad, stack, pulley, cable |
| A/B stripes | **Fixed**: crop **180px** filmstrip panel, then 9:16 letterbox |

**Remaining visible gaps (honest):**

- Muscle parts still **separate meshes** at neck/deltoid (not Day-4 continuous écorché).
- Glow ** softer than Day 4** generative grade; no post skin-lock (p90=88).
- Wrong-form elbow meta **~128°** vs standard 68° (shrug + grip constraint interaction) — may need constraint tuning.
- Occasional **Blender SIGSEGV** at 1080p on some frames; re-run single keyframe if batch aborts.
- Hand/bar contact **good but not perfect** (grip distance 2–7 cm in meta).

**Verdict:** v3 is **reel-usable** for blocking anatomy + machine + motion; still **below Day 4** photographic fibre polish.

---

## Commands

```bash
python3 studio3d/fetch_assets.py
python3 studio3d/pilot/fetch_ecorche.py
python3 studio3d/pilot/render_keyposes.py
python3 studio3d/pilot/make_ab_sheet.py
python3 studio3d/pilot/assemble_reel.py
python3 studio3d/pilot/qc_pilot.py
```

Attribution: `studio3d/pilot/ATTRIBUTION.md`.

---

## Reuse

Add `exercises/<slug>.json`, extend `muscle_glow_pattern()` and seated pose module; keep cameras/QC from `set_pilot.json`.

---

## Round 5 — DEMIC / Muscle & Motion target (`pilot-demic-1`)

**Benchmark shift:** stop Day 4 pixel parity; target **true-3D reels** like DEMIC STORY and Muscle & Motion (owner reference sheets only — not redistributed).

### Architecture

| Piece | Module |
|---|---|
| Continuous MH body + shorts | `studio3d/build_scene.py` via `demic/build_scene_demic.py` |
| Glow masks (UV/vertex, not 634 parts) | `demic/paint_glow_masks.py` — écorché centroids + MH vertex-group fallback → `assets3d/demic_glow_masks.npz` |
| Surface fibre + emission | `demic/shaders_demic.py` |
| Full-ROM lat pulldown | `demic/pose_lat_pull.py` (overhead → chest/belly) |
| Machine + bar accent line | `demic/equipment_demic.py` |
| Loops + 4-step reel | `demic/render_loop_demic.py`, `demic/assemble_reel_demic.py` |
| QC (MediaPipe + meta) | `demic/qc_demic.py` |
| A/B vs DEMIC refs | `demic/make_ab_sheet_demic.py` |

Deliverables: `output/pilot/demic/` — `keyframes/`, `loop_*_*.mp4`, `test_reel_4step.mp4`, `ab_pilot_vs_demic.jpg`, `qc_report.json`.

### Free sources & licences (Round 5)

| Asset | Licence | Use |
|---|---|---|
| MakeHuman base + targets | **CC0 1.0** | Single skinned mannequin mesh |
| Charleeh écorché glTF | **CC BY-SA 4.0** | **Mask projection only** (isolated under `assets3d/`; not shown in beauty pass) |
| BodyParts3D (upstream) | **CC BY-SA 2.1 Japan** | Upstream of écorché |
| Lat station + bar | Procedural | No suitable CC0 lat station integrated |
| MediaPipe | Apache 2.0 | QC pose on rendered PNGs |
| DEMIC / M&M frame sheets | External benchmark | Comparison only |

### Honest gap list vs DEMIC

- **Silhouette:** MH morphs are less “heroic / cut” than DEMIC’s custom mannequin; no paid sculpt or 4K normal map yet.
- **Glow:** Centroid + rig weights ≠ hand-painted UV muscle masks; edges softer, no art-directed fibre orientation per muscle.
- **Shader:** Procedural UV bands + fresnel rim; not DEMIC’s multi-layer albedo + inner contour art.
- **Machine:** Beveled primitives; readable but not product-visualization quality.
- **Motion:** Analytic two-bone IK + partial hand constraint; not mocap (Mixamo/CMU not wired this round).
- **UI:** No floating muscle icons / target-muscle menu (DEMIC app chrome).

### Commands

```bash
python3 studio3d/pilot/demic/render_keyposes_demic.py
python3 studio3d/pilot/demic/render_loop_demic.py right front 96
python3 studio3d/pilot/demic/assemble_reel_demic.py
python3 studio3d/pilot/demic/make_ab_sheet_demic.py
python3 studio3d/pilot/demic/qc_demic.py
```

Regenerate glow cache after mask logic changes: `DEMIC_REGEN_MASKS=1` on any Blender build invocation (or delete `assets3d/demic_glow_masks.npz`).
