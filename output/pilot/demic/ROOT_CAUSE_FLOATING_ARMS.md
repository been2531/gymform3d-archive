# Root cause: floating / severed arms (pilot-demic-1)

## Evidence (Blender diagnostics, `diagnose_arms.py`, wrong form phase 0.5)

### 1. Broken bone chain (primary visual “severed arm”)

| Check | Before fix | After fix |
|---|---|---|
| `lowerarm02.L` tail → `wrist.L` head gap | **0.287 m** | **0.026 m** |
| Other bone pairs | ~0 mm | ~0 mm |

**Cause:** `pose_lat_pull.solve_arm()` called `aim()` on `wrist.{L,R}` with the same forearm direction as `lowerarm02`, but the wrist bone’s rest pose is **not collinear** with the forearm segment. Overwriting the wrist pose matrix **disconnects** the wrist head from the lowerarm tail (~29 cm in world space). The skinned mesh follows the broken chain, so the forearm/hands read as floating beside the torso.

**Not the cause:** Duplicate body mesh (`body_dup: false`), separate écorché render meshes (not imported in beauty pass), or SUBSURF modifier order.

### 2. COPY_LOCATION pulling hands off the solved pose (secondary)

| Check | Value |
|---|---|
| Grip empty `matrix_world.translation` (parented to bar) | **(0, 0, 0)** in depsgraph read during diagnose |
| Hand → grip empty after constraint @ 0.72 | **~1.52 m** |
| Wrist → bar target before constraint | **~0.36 m** (IK too short) |

**Cause:** Partial **COPY_LOCATION** on `hand`/`wrist` (72% influence) toward bar-parented empties whose **world transform was wrong** (origin), blending the solved arm toward world zero and exaggerating detachment.

**Fix applied:** Removed all COPY_LOCATION grip constraints. Bar is placed at **actual wrist positions** after IK.

### 3. IK unreachable bar height (contributing)

Shoulder-to-target distance exceeded `L1+L2` for configured `grip_span_m` and `stretch_bar_z_m`, leaving ~36 cm wrist-to-target error before constraints.

**Fix applied:** Clamp grip half-span and slightly lower bar Z in `_grip_targets()` until both arms reach within `(L1+L2)*0.98`.

### 4. Shoulder mesh vs bone gap (~0.21 m)

Nearest body vertex to `upperarm01` head remains ~21 cm. This is **MakeHuman weighting / shoulder topology**, not a second arm mesh. Tracked separately from chain breaks.

---

## Code changes

- `demic/pose_lat_pull.py`: stop aiming `wrist` bone; reach-clamped targets; bar center from solved wrists.
- `demic/build_scene_demic.py`: remove COPY_LOCATION grip constraints.
