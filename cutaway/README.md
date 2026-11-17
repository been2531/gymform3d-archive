# Cutaway reels — "what's inside your joint" (GymForm 3D)

Silent, language-free 15 s 9:16 reel: skin → circular window → muscle → bone/cartilage →
macro → sourced load number → the window blooms into a full-screen registered bone shot.
Built from Pickle's F1 template (`legacy_pickle/`, kept for reference only, not maintained)
and rewritten around **registered layers**, continuous motion, phone-size type, procedural
audio and automatic QA gates.

## One command

```bash
nice -n 19 python3 cutaway/make_reel.py --topic knee
```

Outputs (never committed, `output/cutaway/` is git-ignored):
`output/cutaway/<id>/reel.mp4, cover.jpg, carousel_1.jpg, carousel_2.jpg, caption.txt,
contact_sheet.jpg, qa.json, report.md`. Exit 0 = every gate passed, 2 = stopped (see report).
Nothing is uploaded; publishing stays behind the existing approval-file flow.

Wall time on the box (8 cores, nice 19, single render process): ~50 s render + ~10 s gates
+ ~5 s packaging ≈ 1 min per reel.

## Files

| file | role |
|---|---|
| `topics/<id>.json` | everything per episode: hook, shots (layer, label ≤3 words, label target and window centre in canvas px, zoom), counter, **facts with source + url**, 3-line caption |
| `joints.json` | prompt parts per joint (knee, shoulder, hip, lower_back, ankle, elbow) + macro crop box |
| `assets_cli.py` | asset step: prompts → crop for the macro edit → register → coordinate grids |
| `register.py` | aligns outer/bone to the skin (constrained similarity search on silhouettes), macro to its crop (ECC); writes `registered/` + `registration.json` + `registration_debug.jpg` |
| `cutaway2.py` | renderer (cv2 warps, ~0.12 s/frame at 1080×1920) |
| `sound.py` | CC0 procedural drone + SFX, BS.1770 loudness to −14 LUFS, soft limiter |
| `qa.py` | gates (criteria fixed before rendering, listed in every report) |
| `make_reel.py` | orchestration + retry table (max 2 retries, then stop and report) |

## Asset step for a new joint (agent-driven)

Image generation is a `GenerateImage` tool call by the agent/routine (no xAI API, no
grok.com, no Higgsfield, no paid assets). Deeper layers are **edits of the skin image** so the
pose stays the same; registration then removes the residual shift/scale.

```bash
python3 cutaway/assets_cli.py prompts --joint hip     # prompts.json
#   GenerateImage(skin prompt, 9:16)                     -> output/cutaway/assets/hip/skin_raw.jpg
#   GenerateImage(outer prompt, ref skin_raw, 9:16)      -> outer_raw.jpg   } in parallel
#   GenerateImage(bone prompt,  ref skin_raw, 9:16)      -> bone_raw.jpg    }
python3 cutaway/assets_cli.py crop --joint hip        # macro_src_crop.png (square crop of bone)
#   GenerateImage(macro prompt, ref macro_src_crop, 1:1) -> macro_raw.jpg
python3 cutaway/assets_cli.py register --joint hip    # registered/ + debug overlay
python3 cutaway/assets_cli.py grids --joint hip       # grid_outer.jpg / grid_bone.jpg
```

Then write `topics/hip.json` (copy knee/shoulder): pick window centres and label targets
from the grids (canvas px), add facts **only with a real source + url** (the gate fails on any
number without one), 3 caption lines, 0 hashtags. Measured: 7 generations ≈ 3 min wall.

Pilot (2026-11-17): shoulder generated end to end this way. Knee reused Pickle's skin
photo as the base and regenerated outer/bone/macro as edits of it.

## QA gates (see `qa.CRITERIA`)

motion in first 0.5 s · no static hold > 1.0 s · registration (silhouette chamfer median ×
max full-frame zoom ≤ 40 px) · label words ≤ 3 · bottom 15 % clear of overlays · camera
never shows padding · 12–15 s · window holes ≤ 10 % · −16…−12 LUFS, TP ≤ −1 dBTP · label
cap ≥ 70 px, hook cap ≥ 110 px · WCAG contrast ≥ 4.5 · text never over the open window ·
every number sourced · caption 3 lines, 0 hashtags.

Retry table (`make_reel.FIXES`): motion → faster push / earlier open; static → more drift;
holes → smaller window; contrast → opaque pill; text over window → shift + smaller window;
registration → drop the full-screen bloom (end inside the window); loudness → audio-only
gain fix. Words, sizes, facts, caption, bottom zone → **not** auto-fixed: the run stops.

## Known limits (honest)

* The registration gate checks the **outline** only. Internal structures can still sit off
  (knee: the generated patella is ~90 canvas px lower than the skin kneecap bump).
* AI anatomy is not textbook-accurate. Shoulder pilot: the scapula is drawn as a fork and the
  humeral head as a cup-shaped cap — needs a human (or reference-guided) anatomy check before
  any post.
* Generated layers are 720×1280; at 1.6–2.4× zoom the window is soft.
* Topic authoring (targets, facts, caption) is still an agent step with a human-readable check.

Fonts: Anton, Archivo Black, Barlow Condensed, Inter, Oswald — SIL Open Font License.

## LOAD METER series (2026-11-17, current default)

Every reel: one verified joint cutaway + the LOAD METER panel counting up to "N× BODY WEIGHT"
for one action, with a cited source. Per episode only `queue.json` changes.

- `series/load_meter.json` – fixed timeline (hook 0–2.6 s, two labelled layer cuts, meter 7.9–15 s,
  full-frame bloom at 12.7 s), counter style, sound. Same look every episode (lime #B4FF39 panel, 0–8× bar).
- `rigs/<joint>.json` – framing per verified joint (window centres, labels, zooms, hook position).
- `queue.json` – topics: joint, action (≤3 words), hook lines, value, fact {claim, source, url, year}, caption.
- `build_topic.py` – series + rig + entry → topic dict for `make_reel.py --entry <id>`.
- `daily.py` – picks the next runnable entry (sourced, verified joint, not the previous reel's joint,
  not done, <2 fails), renders, writes `output/cutaway/daily/<date>_<id>/` (reel, cover, carousel,
  caption, qa.json, report.md, package.json, sources.md). Never publishes. `--lint` runs pre-render gates
  for the whole queue; `--dry-run` shows the pick.
- `upscale.py` – Real-ESRGAN x4plus (BSD-3, ONNX, CPU) used once per library layer (models/ not in git).
- New gates: `one_text_block`, `source_upscale` (≤1.5 output px per source px for every visible layer),
  `landmark` (knee patella vs skin bump ≤30 px).
