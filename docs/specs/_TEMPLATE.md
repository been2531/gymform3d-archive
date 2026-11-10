# Exercise standards spec (template)

Copy this file's structure into `exercises/<id>.spec.yaml`. QC (`tools/rig3d/qc3d.py`)
loads that YAML via `tools/rig3d/spec_load.py` and scores the machine-readable `qc`
block. Do not commit third-party reference images — link them, with licence and credit.

Status values: `pending` (not filled) · `filled` (ready to implement against) · `wired` (QC reads it).

---

## 1. Pose

- **Textbook reference:** a freely licensed photo of the correct setup (URL, licence, author, credit). Link only.
- **Keyframes:** at least `bottom` and `peak`. Each lists joint-angle ranges the posed rig must hit (degrees unless noted).
- **Contacts:** what must touch the bench or the floor (support hand, support knee/shin, standing foot).
- **Alignment:** back angle vs the floor, head-on-spine, hips square.
- **Common faults:** the mistakes this exercise is judged against (rounded back, shrugged shoulder, foot pointing at the camera, …).

## 2. Camera

A fixed preset the beauty renderer must use (`yaw_back_deg`, `elev_deg`, `dist_m`, `look`, `lens_mm`, shifts).
`yaw_back_deg` 0 = pure working-side view; positive yaw rotates behind the side (toward the feet).
Optional `proof` camera (usually pure side) for a silhouette check, not the hero still.

## 3. Look

- Demic clay value range (body V, no clip above the stated max except as noted).
- Groove shading (AO / cavity so separations read as soft shadows).
- Per-muscle glow hue and saturation (OpenCV HSV: H 0–179, S/V 0–255).
- Drawers (briefs) and equipment (bench, dumbbell) so those do not regress.

## 4. QC gates

- Overlay: silhouette IoU minimum and joint MAE maximum vs the linked reference (when a side-view overlay exists).
- `qc.checks`: the list QC iterates. Each item has `name`, optional `frames` (`bottom` / `peak`), a `metric` found on `bench_qc.json` or `overlay_qc.json`, and `min` / `max` / `equals`.
- `qc.eye_checklist`: human review prompts. QC records them but does not auto-pass them.

### Check item schema

```yaml
- name: torso_horizontal          # becomes spec_torso_horizontal[_bottom]
  frames: [bottom, peak]          # omit → once; overlay checks use source: overlay
  source: bench_qc                # bench_qc (default) | overlay
  metric: torso_from_horizontal_deg
  min: 0                          # optional
  max: 10                         # optional
  equals: true                    # optional, for booleans
  label: 등이 바닥에 수평이 아님   # optional; owner-facing Telegram text on failure
  autofix: spec_disable_ref_fit   # optional; pipeline rule name if this gate can self-heal
  blocking: false                 # optional; scored but ignored by spec_pass (e.g. overlay vs authored pose)
```

Also recognized on the spec (applied generically by `apply_spec`):

- `pose.ref_fit`: default false — authored world targets win over an overlay photo.
- `render.frames: keyframes` — beauty stills for the listed keyframes only (not the full clip).
- `notify.title` / `notify.task` — short owner-facing Telegram wording.

Missing metrics are reported as FAIL with value `missing` so a new gate cannot silently skip.
