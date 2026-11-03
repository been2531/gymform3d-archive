# Root cause: why the glass muscle look fails, and the fundamental fix (t689u)

Sheet: [`ROOTCAUSE_MUSCLE_LOOK_sheet.jpg`](ROOTCAUSE_MUSCLE_LOOK_sheet.jpg)
- Same-scale back crops (torso normalised to Day 4's 356 px): Day 4; the D7 sculpted body + Day 4 grade (t676u still); glass before; glass after.
- Builder: `tools/rootcause_muscle_sheet.py`.
- No generation, no render, no new look code.

## 1. Root cause, judged by eye
**Day 4 looks good because the BODY is a sculpted muscle figure.** The generator drew an écorché-like mannequin: every muscle has real relief. The light rolls over each belly, borders sink into grooves, and striations curve with the surface. The Day 4 grade only **recolours** the target region of that relief, so the glow inherits real form.

**Ours paints 2D fibres onto a smooth body.** The glass body has no muscle relief, so the code has to invent form (belly gradient, grooves) as flat paint. The result reads as stickers or islands floating on a blank surface. The D7 sculpted still (t676u), graded with the same Day 4 code, already looks right: **the source body is the deciding factor, not the colour code.**

What sells "real muscle":

| Cue | Day 4 | D7 sculpted + Day 4 grade | Glass before | Glass after |
|---|---|---|---|---|
| 1. The body's own muscle relief catches the light | O | O | X | X |
| 2. Muscle borders continue across the whole body (glowing and non-glowing muscles share one anatomy) | O | O | X (islands) | X |
| 3. Fibres bend with the surface curvature, not straight 2D lines | O | △ | X | X |
| 4. Specular highlights, cast shadow and occlusion (arm over lat, scapula edge) | O | O | X | △ (fake belly light) |
| 5. Glow and body read as one material, not a layer on top | O | O | X | X |

## 2. Why the metrics did not predict quality
- **What they measure:** `tools/day4_look_metrics.py` only looks INSIDE the glow mask and only at 2D statistics: hue, saturation, value, band-pass std, sign changes, the edge ramp and the area.
- **What they miss:** they cannot see 3D form, whether texture follows the surface, continuity with the rest of the body, or the "island" effect. Any texture with the right colour histogram and frequency passes, which is what the tuning produced.
- **A relief proxy fails too.** Shading std on the non-glow body ([`ROOTCAUSE_MUSCLE_LOOK_relief.json`](ROOTCAUSE_MUSCLE_LOOK_relief.json)): Day 4 6.9 vs glass 18.3. The glass rim and edges fool it, so no single number captures the look.

**New acceptance check (replaces metrics-as-gate):**
1. **Side-by-side eye check:** same scale, same crop, next to Day 4, with the 5 cues above. It passes only if every cue is O. The owner judges; I pre-screen.
2. **Paired preference:** the owner picks A or B blind (Day 4 crop vs candidate, flipped order). The target is "can't tell which is Day 4".
3. **Metrics stay as guards only:** hue and saturation within 10 %, glow outside the target regions 0 px, machine SSIM. They are never a proof of quality.

## 3. Fundamental options
| | A. Generated sculpted muscle mannequin + Day 4 grade as is | B. Blender + Z-Anatomy 3D muscles | C. Hybrid: Blender muscle normal/mask pass fitted onto the generated body | D. Demic-style pro pipeline |
|---|---|---|---|---|
| Muscle form | From the source body (like Day 4 and the t676u D7 still) | True geometry | Real normals, but fitted to someone else's body | True geometry, rigged anatomy character + a muscle mesh in a second render pass |
| Motion stability | Day 4-proven with the native grade. The generator's relief can boil (playbook D5), handled as in Day 4 | Perfect once rigged | Drift between the fitted mesh and the video body | Perfect |
| Machine fidelity | As now: edit of the verified plate, part-by-part check | Needs a 3D Matrix model or a camera-matched comp on the plate | As A | Needs a 3D machine |
| Cost (generations) | 1–2 images + 1 video per form | 0 | As A | 0, but paid assets and/or outsourcing (≈ €80 per 10 s clip on freelancer listings) |
| Time per reel | Generation plus a 1–2 min render (existing pipeline) | Setup days: Z-Anatomy is an atlas, not rigged. CPU render only (no GPU; headless bpy 5.2.3 in /workspace/.bpyenv, no Blender GUI), Cycles ≈ 3.5 min per 1080p frame ≈ hours per reel | B's setup + per-reel fitting | Weeks to set up, or external |
| Risk | Generator anatomy errors or glow off-target, both handled by existing clamps | High: rigging, licence (CC BY-SA 4.0 attribution + share-alike), render time | High | Purchase approval, budget |
| Reusable | The whole Day 4 pipeline (native grade, region clamp, joint callout, 2-shot plan, machine check) | Long-term asset library | Little | Everything, if built |

Licence note for B: Z-Anatomy is CC BY-SA 4.0 (BodyParts3D upstream). Reels would need the attribution, and any derived mesh shares alike.

## 4. Recommendation: **A**
**Why A:**
- The evidence shows the deciding factor is a sculpted source body; the existing Day 4 grade already looks right on it (the D7 t676u crop).
- Fastest path, and it reuses the most code.
- Machine fidelity and the camera plan stay as they are.

**What it costs:**
- The smoked-glass body is retired, as a muscle carrier at least.
- The look returns to the Day 4 charcoal écorché body with neon only on the target muscles.

**Smallest validation test: 1 still, 1 GenerateImage, after owner approval.**
1. Edit `wide_peak_gen1.jpg`, keeping the machine, camera and pose. Replace only the body with a Day 4-style sculpted charcoal muscle mannequin: visible muscle relief and striations, flat even studio light, head the same tone, no glow. Day 4 frame as the style reference.
2. Run the existing Day 4 native grade (green: lats, rhomboids/mid trap, rear delt, faint biceps) with the joint callout.

**Pass criteria, by eye, side by side with the Day 4 crop at the same scale:**
- All 5 cues are O.
- No glow outside the target muscles.
- The machine passes the part check vs the verified plate.
- Head tone = body.
- The owner prefers it, or can't tell it from Day 4.

**Fix later, not part of this test:** the V handle (currently rectangular) and the peak pose (elbows past the back line).
