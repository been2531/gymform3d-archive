# db_row target muscles

Sourced list for the one-arm dumbbell row (hinged / bent-over). Written 2026-11-10 (Asia/Seoul). Glow policy follows this list; IDs are the rig’s 18 persistent muscle labels (`tools/rig3d/muscle_ids.py`).

## Right-form glow

| Role | Muscles | Glow |
|---|---|---|
| Primary | latissimus dorsi; middle trapezius + rhomboids; posterior deltoid; teres major | Green sheet (current `target_muscles`: `lats`, `rhomboids`, `mid_trap`, `rear_delt`) |
| Secondary | biceps brachii; brachialis | Synergists. Not painted on the right-form sheet (`target_only` QC). Over-recruitment is treated as a compensator (red) on the wrong take. |
| Stabilizer | erector spinae | Isometric anti-flexion of the hinged torso. Not a prime mover; not in the right-form glow list. |
| Not a target | upper trapezius | Must **not** glow green in right form. Red compensator only on the wrong take (shrug / yank). |

Teres major has no separate ID today — the inferior-scapula / lat-insertion island is classified with `lats`, so it glows with the primary sheet. Infraspinatus has no separate ID either (posterior scapula is split between `rhomboids` and `rear_delt`). Lower trapezius is not a separate ID (midline mid-back is `erectors` / `rhomboids` / `mid_trap`).

## Sources

- ExRx.net, *Dumbbell Bent-over Row*: target latissimus dorsi; synergists teres major, posterior deltoid, rhomboids, middle and lower trapezius, infraspinatus, teres minor, brachialis, biceps, brachioradialis; erector spinae as a stabilizer. https://exrx.net/WeightExercises/BackGeneral/DBBentOverRow
- NSCA, *Essentials of Strength Training and Conditioning* (bent-over / one-arm row progressions): latissimus dorsi as the prime mover; middle trapezius and rhomboids for scapular retraction; posterior deltoid assisting; spinal erectors stabilize the hinged torso.
- Lehman, Buchan, Lundy, Myers, Nalborczyk (2004), *J Electromyogr Kinesiol* — “Variations in muscle activation levels during traditional latissimus dorsi weight training exercises.” Rows produce high latissimus activation; scapular retractors (mid trap / rhomboids) contribute; grip and trunk position change the mix.
- Youdas et al. (related EMG on seated / standing row variations): mid-trap and rhomboid activity rise with intentional scapular retraction; upper trap rises when the shrug substitutes for retraction (the wrong-form compensator).
- Schoenfeld / Contreras reviews of row EMG: teres major tracks the lats as a humeral extensor / adductor; biceps are consistent synergists, larger when the row is yanked with elbow flexors rather than driven by the shoulder.

## Wrong-form compensators (red)

`COMPENSATORS = [upper_trap, biceps, erectors]` in `tools/rig3d/muscle_activation.py`.

- Upper trap: shrug instead of scapular retraction.
- Biceps: yanking the dumbbell with elbow flexors.
- Erectors: standing up out of the hinge / lumbar extension instead of keeping the torso isometric.

## Atlas

`output/_rig3d/v4_authored/muscle_atlas.png` — back / side / front of all 18 IDs, each a distinct colour with its name. Self-check notes live in the executor report for this turn (anatomy vs the painted islands).

## Atlas self-check (v5 classify, 2026-11-10)

`output/_rig3d/v4_authored/muscle_atlas.png` — rest-pose back / right-side / front.

What matches anatomy:

- **Lats** (red): wide lateral back from the lower thoracic / iliac-crest level up toward the axilla. Teres major / lat insertion is merged into this island (no separate ID) so it glows with the primary sheet.
- **Rhomboids** (green): between the spine and the scapular region.
- **Mid trap** (blue): upper mid-back, below upper trap, across the midline.
- **Upper trap** (cyan): neck to the top of the shoulder — present, and **not** in the right-form glow list.
- **Rear delt** (orange): posterior shoulder cap (was previously swallowing the whole scapula and proximal arm).
- **Side delt** (magenta): lateral cap. v4 classify left only 25 verts here; v5 restores ~1357.
- **Front delt, pecs, abs, obliques, biceps, triceps, forearms, glutes, quads, hams, calves, erectors**: expected seats on the matching view.

Known gaps (not invented as extra IDs this turn — the atlas still shows the existing 18):

- No separate **teres major**, **infraspinatus**, **lower trapezius**, or **brachialis** labels.
- Infraspinatus sits in the zone between rhomboids, lats, and rear delt.
- Lower trap overlaps the midline mid-back (`erectors` / `rhomboids` / `mid_trap`).
- Brachialis is not distinguished from biceps / forearm.
