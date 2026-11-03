# Round 6 — adopted sources & strategic correction

## What we stopped doing

Custom `pose_lat_pull` analytic IK + COPY_LOCATION grips + écorché centroid glow + bar-end plates + stack-in-front layout (Round 5 demic path). Root causes documented in `ROOT_CAUSE_FLOATING_ARMS.md`.

## What we adopted (verified licences)

| Component | Choice | Licence / terms | Notes |
|---|---|---|---|
| **Body + rig** | MakeHuman base mesh + `default.mhskel` rig (`studio3d/build_scene.py`) | **CC0 1.0** | Continuous skinned mesh; unchanged from pilot v3. |
| **Motion / pose** | `studio3d/pilot/pose_seated_lat.py` + `exercises/wide_grip_lat_pulldown.json` (motion standards FK) | CC0 mannequin; exercise config in repo | Stretch: `bar_grip_points` + `_solve_arm_to_wrist` (same module, used in pilot-3). Contract: `pose_arms_from_standards`. **Wrist bone no longer FK-aimed** (chain fix). |
| **Mixamo** | Not integrated | Adobe Mixamo FAQ: royalty-free commercial use of embedded renders; **no redistribution of raw FBX** | Requires Adobe ID in browser; not available headless in this agent (no secrets). Documented for next step if owner uploads FBX to `assets3d/`. |
| **Rigify / MPFB2** | Not integrated | Rigify GPL (Blender addon); MPFB assets CC0 per community docs | Addon present in Blender 4.0; full meta-rig + retarget deferred — would replace MH rig entirely. |
| **CMU mocap** | Not integrated | CMU FAQ: free use with attribution common practice | No seated lat pulldown clip wired; retarget pipeline not in repo. |
| **Glow masks** | `discrete_glow_masks.py` — MH **vertex groups** + facing tests | Same CC0 body | Discrete lat / bi / delt / trap; **no écorché** in beauty pass. Cache: `assets3d/demic_discrete_glow.npz`. |
| **Machine** | `equipment_r6.py` procedural | N/A (author-generated) | **No CC0/CC-BY lat GLB downloaded:** Sketchfab “Gym Lat Pulldown” (neilken) is free download but **licence not stated on page** — skipped per owner rule. Layout: stack **behind** seat (+Y), pulley above, cable to bar centre, palette-A on **stack pins only**, **no bar-end plates**. |
| **Shaders** | Reused `shaders_demic.py` | In-repo | Fibre + contour emission on `GLOW` R channel. |

## Root-cause carry-forward (Round 5 → 6)

1. **Wrist `aim()`** broke bone chain → fixed in `pose_seated_lat.py` (forearm only).
2. **COPY_LOCATION** to miscalculated empties → removed in R6 builder.
3. **Bodysuit glow** from écorché/rig weights → replaced by discrete vgroup masks (mean active weight ~1.6–2% of verts vs full torso).
4. **Plates at face (side)** → bar is handle-only; weights on stack behind user.

## Remaining gaps (honest)

- **MH shoulders:** mesh–bone gap at deltoid; arms can still read detached/crossed at wide overhead stretch despite `_solve_arm_to_wrist`.
- **Mixamo/Rigify** not yet replacing MH rig — fastest *licence-safe* path in-repo was motion-standards module.
- **Machine** still blockout quality; no verified CC machine asset.
- **Stretch meta** now passes wrist above head; **visual** arm cross/attach still fails on front stretch in review.

## Commands (smoke only)

```bash
python3 studio3d/pilot/demic/render_smoke_demic.py   # uses build_scene_r6.py
python3 studio3d/pilot/demic/make_smoke_contact_sheet.py
```
