# Smoke frame review (540×960, 8 samples) — pilot-demic-1

After root-cause fix (remove COPY_LOCATION; stop aiming `wrist` bone; reach-clamped IK; elbow pole toward floor + lateral).

Meta: all four frames report `grip_distance_m: 0.0` (bar aligned to solved wrists).

---

## wrong · front · stretch (`wrong_front_stretch_beauty.png`)

| Criterion | Result | Notes |
|---|---|---|
| Arms attached | **FAIL** | Forearms/hands read detached at shoulders; upper arms do not meet torso cleanly (MH shoulder weights + ~21 cm bone–mesh offset at `upperarm01` head). |
| Hands on bar | **PARTIAL** | Bar at wrist height; grip meta 0 m but hands clip through / beside bar visually. |
| Bar overhead at stretch | **PASS** | Meta: wrist_z ≥ head_z; bar upper third of frame. |
| Cable to pulley | **FAIL** | Vertical cable passes through head; no clear path from bar center to top pulley. |
| Glow correct muscles / DEMIC area | **FAIL** | Full front torso saturated red (wrong-form mask still covers chest/abs via rig weights). |
| No bleed | **FAIL** | Torso-wide emissive wash, not delt/bi-only patches. |
| No floating parts | **FAIL** | Hands/forearms appear as separate floating pieces on front view. |
| Machine readable | **PARTIAL** | Seat + stack + towers identifiable; cable/pulley logic wrong. |

---

## wrong · side · contract (`wrong_side_contract_beauty.png`)

| Criterion | Result | Notes |
|---|---|---|
| Arms attached | **FAIL** | Side view gap at shoulder; upper arm pivot reads behind torso shell. |
| Hands on bar | **PARTIAL** | Hands near bar plane; fingers not wrapped. |
| Bar overhead at stretch | N/A | Contract phase — bar at chest; **PASS** for phase intent. |
| Cable to pulley | **FAIL** | Cable segment missing or misaligned to bar attach point. |
| Glow correct muscles / DEMIC area | **PARTIAL** | Red on upper back/traps; still too broad vs DEMIC patches. |
| No bleed | **FAIL** | Red spills across upper torso block. |
| No floating parts | **FAIL** | Shoulder discontinuity visible. |
| Machine readable | **PARTIAL** | Side stack + seat OK; pulley path unclear. |

---

## right · front · stretch (`right_front_stretch_beauty.png`)

| Criterion | Result | Notes |
|---|---|---|
| Arms attached | **FAIL** | Same shoulder gap; arms crossed X in front of face (elbow pole still folds arms inward on wide grip). |
| Hands on bar | **PARTIAL** | Meta grip 0 m; visual cross grip. |
| Bar overhead at stretch | **PASS** | Bar high, arms extended upward. |
| Cable to pulley | **FAIL** | Centerline cable through mesh. |
| Glow correct muscles / DEMIC area | **FAIL** | Emerald on **front** chest (lats should dominate **back**; front view should show little/no lat glow). |
| No bleed | **FAIL** | Large front chest glow patch. |
| No floating parts | **FAIL** | Visible center seam / split on torso mesh in render; detached arms. |
| Machine readable | **PARTIAL** | Same as above. |

---

## right · side · contract (`right_side_contract_beauty.png`)

| Criterion | Result | Notes |
|---|---|---|
| Arms attached | **FAIL** | Shoulder socket gap on side view. |
| Hands on bar | **PASS** | Side profile: hands on bar, meta grip 0 m. |
| Bar overhead at stretch | N/A | Contract — bar at upper chest; **PASS**. |
| Cable to pulley | **FAIL** | Drop cable not tied to moving bar center. |
| Glow correct muscles / DEMIC area | **PARTIAL** | Green on mid-back/lats region — closest to target; area still small/soft vs DEMIC. |
| No bleed | **PARTIAL** | Neck/trap spill; less than front views. |
| No floating parts | **FAIL** | Shoulder gap remains. |
| Machine readable | **PASS** | Lat station, stack pins (cobalt), gold rims, bar plates readable. |

---

## Summary

- **Root-cause fixes verified in data:** bone chain gap 287 mm → 26 mm; wrist–target 0 m; COPY_LOCATION removed; bar–wrist grip 0 m in meta.
- **Visual smoke still fails** owner criteria on arm attachment, glow placement/bleed, and cable routing — needs shoulder–mesh weighting or clavicle solve, view-dependent glow masking, and bar-attached cable rig before any batch reels.
