# D7 character concept (t684u): simplify the CHARACTER so the full machine can stay

Sheet: [`CHARACTER_CONCEPT_sheet.jpg`](CHARACTER_CONCEPT_sheet.jpg) · numbers: [`CHARACTER_CONCEPT_metrics.json`](CHARACTER_CONCEPT_metrics.json) ·
script: `nice -n 19 python3 tools/character_concept_sheet.py` (≈4 s, **0 generations**, existing frames only).

The A/B panels are **rough code mock-ups** on the existing D7 right-peak frame (same pose, real machine pixels). The silhouette
comes from a mask, so the arm and hand edges are blobby, and the head is an ellipse fill. They show tone, flatness and
the code fibre look, not the final render quality. The fibres on A/B are synthesised in code: oriented stripes inside the
existing glow regions (lats along armpit→opposite hip, scapular group across the scapula), Day 4 right HSV, 10 px period and a 4 px ramp.

| | A smooth matte charcoal | B low-poly facets | C Day 4 body, plain head |
|---|---|---|---|
| Motion stability | **best**: no surface detail for the video model to re-render → no boil (body texture Laplacian std 0.36 vs 3.9 now) | good, but facet edges get re-drawn per frame → edge shimmer (std 1.8) | worst: generated muscle detail boils and slides (playbook D5/D6) |
| Day 4 code fibres | **fully ours**: fibres drawn in code on a blank surface, tracked by kp, ramped with contraction | two competing textures (facets + fibres) | glow rides on generated ridges we don't control |
| Head tone | head = body by construction (head L 74 / torso 78) | same (77/77) | needs a per-frame fix (44 vs 30 after fix; raw head 39 ≈ backdrop 38) |
| Readability with full machine | **mid-grey body clearly above the backdrop (78 vs 38)**, reads against the white frame and the blue stack | same tone, busier | dark body ≈ backdrop (30 vs 38), so it sinks into the background |
| Premium feel | clean studio mannequin (Day 1 look) | reads as a "game asset" | closest to Day 4 today |

Fibre mock vs Day 4 right (A; B is the same): hue 148.0/147.6, sat 162.5/163.5, val 184/179, contrast 16.3/20.3,
density 5.3/10.5 per 100 px, falloff 2.8/4.0 px, cover 0.097/0.222. Colour matches. Density, contrast and cover are still
low because the stripe period is set before the Day 4 rescale (×1.23) and the mock only fills the current glow mask. The
real build tunes the period to about 8 px and uses the full region polygons.

**Scale (all options):** in this side view the whole machine is wider than tall. At full width the person is about 36%
and machine plus person about 68% of the 9:16 height, so cropping or zooming cannot reach the Day 1/2 size without cutting
the machine. Reaching about 75–85% with the full machine needs a new plate: a closer, higher rear-3/4 camera that
foreshortens the rail, at native 1080×1920. The current plate is 720×1280, and the 1024² source would need an upscale, which shows as blur.

## Recommendation: **A, smooth matte charcoal mannequin + code fibres**
- It is the only option where motion can't boil, the head tone is right by construction, and the glow is 100% ours.
- Next step, after the owner says OK:
  1. one new high-res 9:16 plate/still at the closer rear-3/4 angle;
  2. one start→peak clip;
  3. kp-tracked code fibres on top, with the region polygons from `configs/day07_muscle2.json`.
