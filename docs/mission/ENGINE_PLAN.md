# gymform.3d Engine Plan

Access-date: 2026-11-10 (Asia/Seoul). Read-only research of sibling bots; gymform code not modified this turn.

Goal for 중현 님 / @gymform.3d: reach or beat @demicstory quality while raising **reusability** and cutting **wall time** — by building an engine the way GlobeFlip and Hidden Systems already do, not by rewriting one-off scripts per exercise.

Evidence labels: **FACT** (path / measured number) · **INFERENCE** (mapped to gymform) · **ASSUMPTION** (needs a timed smoke).

---

## 1. Sibling engines (what they actually do)

### 1.1 GlobeFlip / FlipCut (YouTube map shorts)

**Repos / live code (FACT)**
- Working tree: `/workspace/next6/eng_test/` (live) and archive `/workspace/flipcut-archive/next6/`
- Remote pattern: `been2531/flipcut-archive` (branches like `box/ggt-template-v2`)
- Map Shorts CLI: `geo_template/flipcut.py build <slug> [--release]`
- Newer factory: `ggt_template/` (`engine.py`, `anim.py`, `director.py`, `factory.py`)
- Docs: `geo_template/MODULES.md`, `eng_test/PLAYLIST_FACTORY.md`, `ggt_template/FACTORY.md`

**Architecture (FACT)**
```
episode.json → spec → tts → maps → assets → build → thumb → preflight → render → mix → qc → meta → package
```
- Declarative episode (`episode.json`) + thin shim; actors come from a **REGISTRY**, not per-episode copy-paste.
- Shared modules under `geo_template/`: `fc_spec.py`, `fc_actors.py`, `fc_render.py`, `fc_assets.py`, `fc_hero.py`, `fc_geo.py`, `fc_post.py`, `flipcut.py`.
- **Hero library** `assets/hero_lib/HERO_LIB.json`: reusable 3D sprites/FX; promote only after owner approval (`flipcut.py lib promote`).
- **Lessons ledger** `LESSONS.json` / `LESSONS.md`: every correction → rule + enforcement tag (`preflight:<gate>` / `actor:<module>` / `review:<item>`). Build fails if a lesson claims a gate that does not exist.
- **Playlist factory**: one map engine → 15 playlists / 5 engine groups (E01–E05). Data + story swap; World Map / Camera / Label kits stay fixed (`PLAYLIST_FACTORY.md`).
- **Factory modules** (additive, do not edit in-flight `engine.py`): `network_engine.py`, `route_engine.py`, `heatmap_engine.py`, `timeline_engine.py` (`FACTORY.md`).

**Caching (FACT, kourou-rocket measured in MODULES.md)**
| Layer | Key / skip | Gain |
|---|---|---|
| TTS | line text hash; never re-voice unless `--force tts` | 0.00 s when text unchanged |
| Maps | graded+feathered `.npy` LRU (`FC_LAYER_CACHE_GB=4`) | 2.6 s → 0.3–0.4 s / process |
| Render chunks | code_sig + camera + captions + actors (±1.5 s) per 120-frame chunk | 1 caption edit → 13.3 s (2/11 chunks); no-op → 0.5 s |
| Mix | timeline + sfx + takes | 4.1 s cold / 2.6 s hit |
| **Whole build** | stage stamps in `qc/flipcut_stages.json` | **~108 s manual → 75.7 s cold · 18.1 s incremental · 4.6 s no-op** |

**QC gates (FACT)**
- Hard **preflight** before render (`preflight.py [--release]`); FAIL stops the chain.
- Gates include licence, no-2d-procedural-art, hero-size, land-only, vo-noun-match, hero-photoreal, caption safe-zone, etc.
- Post: close-ups + 12-up sheet; meta prepared with `doNotUpload` until approval.

**Timing / quality philosophy (FACT → INFERENCE for us)**
- Quality rules are **code**, not chat memory: auto QC → private upload → schedule with no bot judgment step (transcript t886u / pos ~3623).
- New series = new data YAML + playlist slot, not a new renderer.

---

### 1.2 Hidden Systems (AMP / hidden-shorts)

**Repos (FACT)**
- Local: `/workspace/autonomous-media-pipeline`
- Remote: `been2531/hidden-shorts` (private); folder name stays `autonomous-media-pipeline`
- Docs: `MODULE_REGISTRY.md`, `PROJECT_STATE.md`, `docs/engine.md`, `README.md`

**Architecture (FACT)** — “storyboard picks shots, library renders them”
```
storyboard.json
  → amp/shots.resolve (template params → cached JPEG dirs → rewrite as footage)
  → TTS (cached) → align → captions → SFX/voice
  → visuals + encode → QA → contact sheet + summary.json + SUMMARY.md
```
One command:
```
nice -n 19 python3 -m amp.make_episode storyboards/<id>.json
```
Exit codes: 0 pass · 2 QA fail · 10 bad storyboard · 11 shot fail · 12 other.

**Module types (FACT)**
| Kind | Path | Role |
|---|---|---|
| Pipeline modules | `amp/*.py` | storyboard, tts, align, audio, captions, visuals, render, qa, shots, parallax, pen, sfx, make_episode |
| Shot templates | `shotlib/templates/*.js` | `path_follow`, `globe_routes`, `exploded_view`, `macro_tunnel`, `machine_action`, `prop_turntable`, **`assembly`** (main format) |
| Shot core | `shotlib/core/{stage,gltf,kit}.js` + `runner.mjs` | three.js / Chrome SwiftShader stage, bloom, Z-up kit |
| Product objects | `shotlib/objects/<name>.json` | reusable assemblies (pen, padlock, raptor_engine, …) from `tools/obj_*.py` |
| Looks | `amp/pen.py` + storyboard `"look"` | post-process flag; 3D cache reused |
| Asset ledger | `assets/3d/LICENSES.csv`, `tools/fetch_asset.py` | CC0/CC-BY only; raw assets gitignored |

**Caching (FACT, PROJECT_STATE / MODULE_REGISTRY)**
| Cache | Key | Measured |
|---|---|---|
| Shots | `sha1(template, params, seconds, size, fps, **shotlib code version**)` → `cache/shots/<template>_<key>/` | Cross-episode hit verified (`hs_demo_parallax` reused seabed) |
| TTS | `sha(text,voice,lang,speed)` | look variants reuse audio |
| Pen look | `cache/pen/<shot>_<ver>_<params>_1080x1920` | ~0.165 s/frame filter |
| Episode wall | — | **~4.4–4.5 min cold / ~1.1 min warm** (hs_001_v3); assembly hs_003 **3.9 min cold / 0.5 min warm** |
| Engine choice | three.js 720→1080 encode vs Blender EEVEE | 0.21 s/frame vs 1.8 s/frame; “cheapest setting that looks the same” is a standing rule |

**QC gates (FACT)**
- `amp/qa.py`: res/aspect/fps/codec/duration/black/loudness/peak/silence/captions/loop/coverage (16 checks on recent episodes).
- Key overlay auto-avoids subject (`visuals._place_key`).
- Upload only after topic approval file + QA pass summary (routinizer parity).

**Product / explode engine (FACT, `docs/engine.md`)**
- Own parametric products; external catalogs = shape reference only, never published mesh.
- Mechanism primitives (`tools/mechanisms/`) compose new objects without rewriting the renderer.

---

## 2. What gymform already has (FACT)

Repo: `/workspace/daily_info_reels` · remote `been2531/gymform3d-archive` · pipeline entry `tools/rig3d/pipeline.py`.

| Layer | Exists today | Paths |
|---|---|---|
| One-command pipeline + stamps | yes (resume / `--force` / `--next`) | `tools/rig3d/pipeline.py`, `_cache_rig3d/<slug>/stamps` |
| Shared defaults | yes | `modules/_shared.yaml` |
| Motion primitives | partial | `modules/*.yaml` (pull_row, squat_*, press_*, hinge, …) |
| Equipment props | started | `modules/equipment/{barbell_olympic,bench_flat}.yaml`, `tools/rig3d/props/` |
| Playlist queue | yes | `playlist.yaml` (GlobeFlip-style items after `db_row`) |
| Exercise configs | yes | `exercises/*.yaml` |
| Retarget + Cycles render | yes (monolith) | `tools/rig3d/retarget_render_v2.py` (~766 LOC) |
| Day 4 grade + looks | yes | `grade3d.py` · looks `clay` / `charcoal` in `_shared.yaml` |
| QC + posture + anatomy | yes | `qc3d.py`, `posture_qc.py`, `anatomy_qc.py` + RULE_MAP auto-fix |
| Grip / support hand | yes | `hand_grip.py`, motion keys in exercise YAML |
| Docs | yes | `docs/mission/RIG3D_PIPELINE.md`, `REUSABLE_ASSET_PLAN.md`, `QUALITY_UPGRADE_PLAN.md`, `DEMIC_UPDATE_*.md` |

**Gaps vs siblings (INFERENCE from above)**
1. No **MODULE_REGISTRY** — interfaces and cache keys are tribal knowledge in markdown, not one table the CLI prints.
2. No **motion library cache** of baked retarget actions (re-pose every cold run).
3. No first-class **scene/shot templates** (camera + lights + floor + look as a named kit).
4. No **overlay / callout layout solver** with keep-out (Hidden Systems `_place_key` / FlipCut keepout) — callouts still ad-hoc in grade.
5. No **segment composer** for the planned 4-segment reel (wrong bone → right bone → wrong muscle → right muscle).
6. QC is strong **after** render; constraint-first (prevent bad poses at retarget) is incomplete vs owner ask t736u.
7. Render is one Blender job; no **chunk / still / farm** split (FlipCut chunks; AMP 720 preview → 1080 encode).
8. `retarget_render_v2.py` is a single file — hard to reuse for barbell two-hand / supine bench without forking.

---

## 3. Proposed gymform engine: `gymform_engine`

Name the stack **gymform_engine** (folder `tools/gymform_engine/` or evolve `tools/rig3d/` in place). Mirror sibling contracts: **one command, declarative inputs, cached stages, hard gates, playlist-driven**.

### 3.1 Layers

```
playlist.yaml + exercises/<slug>.yaml
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ L0  Asset library                                         │
│     body (Grab3D) · garments · props · HDRI · looks       │
├───────────────────────────────────────────────────────────┤
│ L1  Motion library                                        │
│     YMove/MediaPipe → cached retarget action (.blend/.npz)│
│     + constraint bake (feet, grip, ROM, support)          │
├───────────────────────────────────────────────────────────┤
│ L2  Scene / shot templates                                │
│     studio_side_tight · bench_supine · …                  │
│     lights + camera + floor + look preset                 │
├───────────────────────────────────────────────────────────┤
│ L3  Overlay / callout layout solver                       │
│     keep-out vs figure+props; laser / leader / title      │
├───────────────────────────────────────────────────────────┤
│ L4  Segment composer                                      │
│     4× (form × reveal): wrong/right × bone/muscle         │
│     TTS + captions + mux                                  │
├───────────────────────────────────────────────────────────┤
│ L5  QC + constraint layer                                 │
│     preflight (cheap) → render → grade QC → sheet         │
│     every fail → permanent check + auto-fix rule         │
├───────────────────────────────────────────────────────────┤
│ L6  Render cache / farm                                   │
│     stills / short / full; stamp keys; nice 19 / loadgate │
└───────────────────────────────────────────────────────────┘
```

#### L0 — Asset library
- **Body:** one locked Grab3D mannequin + painted muscle groups + fist shape key + briefs mesh (already in pipeline docs).
- **Props:** procedural builders under `tools/rig3d/props/` registered like AMP `shotlib/objects` (dumbbell hex, `barbell_olympic`, `bench_flat`, future plates/rack).
- **Looks:** named presets `clay` / `charcoal` (keep both; clay default) with skin p50 bands and glow params in `_shared.yaml`.
- **Licence ledger:** one CSV/MD for Grab3D / Poly Haven / YMove pose-only (never commit third-party clips).

#### L1 — Motion library (highest leverage for time)
- Cache key = `sha1(clip_sha, motion_cfg, retarget_code_sig, body_asset_sig)`.
- Output: baked action + `bones.npz` + posture series (already dumped today) under `_cache_rig3d/<slug>/motion_<key>/`.
- **Constraint bake (t736u):** grip wrap, feet plant, ROM clamps, support hand, scapula/elbow path as **retarget constraints**, not post-QC hope.
- Wrong-form twin: same clip + fault module override (or second clip) → second cache entry.

#### L2 — Scene / shot templates
- YAML kits: camera mode, light energies, HDRI, floor, res/spp, mask spp.
- Exercise picks `shot: studio_side_tight` or `bench_supine` instead of duplicating `render:` blocks.
- Pattern = AMP `shot_defaults` + FlipCut base styles.

#### L3 — Overlay / callout layout solver
- Input: figure mask, prop AABBs, safe zone, measured angles.
- Output: title / preview pill / laser / leader / 3 English caption lines with 0 hashtags.
- QC: `overlay_clear_of_figure_px`, callout clear of prop (owner issue: line near dumbbell).

#### L4 — Segment composer
- Spec (from Demic plan): 0–5 s wrong bone · 5–10 s right bone · 10–15 s wrong muscle · 15–20 s right muscle.
- Reuses **one** motion cache × two grades (bone x-ray vs muscle fibre) × two form variants.
- Mux + edge-tts + caption lint; no per-segment re-retarget.

#### L5 — QC + constraint layer
- **Preflight (seconds):** pose gates, config schema, licence, asset presence — FlipCut-style hard stop.
- **Geometry QC:** posture_qc + anatomy_qc on `bones.npz` before full beauty (AMP-style cheap fail).
- **Pixel QC:** grade + glow + overlay after render.
- **Self-healing:** keep `qc3d.RULE_MAP` + `--max-iter`; unmapped → human queue item.
- Add `MODULE_REGISTRY.md` + `docs/mission/ENGINE_PLAN.md` (this file) as the contract the bot/routinizer reads.

#### L6 — Render cache / farm
| Mode | Use | Target wall (ASSUMPTION → measure) |
|---|---|---|
| `probe` | 1–2 stills, low spp | ≤ 1–2 min |
| `sheet` | start+peak+crop | ≤ 5–8 min |
| `short` | N frames / one rep | ≤ 10–15 min |
| `full` | 4-segment master | ≤ 20–30 min cold; ≤ 5–8 min warm if motion+beauty cached |
| Cloud agent | optional 1080 beauty when box loaded | owner-approved only |

Stamp keys per stage like FlipCut `flipcut_stages.json` / AMP `out/episode_timings.csv` → append gymform timings CSV.

### 3.2 Exists vs gap (checklist)

| Engine piece | Status | Next concrete step |
|---|---|---|
| Asset body + HDRI + garment | EXISTS | Freeze as `assets3d/master` pointer; licence row |
| Prop modules barbell/bench | PARTIAL | Wire two-hand grip + supine into retarget (other worker) |
| Look presets clay/charcoal | EXISTS | Keep both; glow QC bands |
| Motion primitive YAML | EXISTS | Split retarget into `motion_lib.bake()` + cache |
| Cached retarget actions | GAP | Implement L1 cache + key |
| Constraint-first retarget | PARTIAL | Promote grip/feet/ROM to hard bake; QC verifies fail-closed |
| Shot templates | GAP | Extract `render:` from `_shared` → `modules/shots/*.yaml` |
| Overlay solver | GAP | Port keep-out from grade into `layout_solve.py` |
| 4-segment composer | GAP | `compose_segments.py` after db_row quality lock |
| Preflight stage | GAP | Cheap gates before Cycles |
| MODULE_REGISTRY | GAP | Write table mirroring AMP |
| Timings CSV | GAP | Log stage seconds every run |
| Playlist | EXISTS | Drive `--next`; add segment-ready flag |

### 3.3 Priority order (do in this order)

1. **P0 — Constraint bake + preflight** (quality floor, fewer retries; matches t736u).  
2. **P1 — Motion library cache** (largest warm-path time win; siblings’ proven pattern).  
3. **P2 — Shot templates + look registry** (barbell/bench without forking render).  
4. **P3 — Overlay layout solver** (Demic-like clarity without manual nudge).  
5. **P4 — Segment composer + 4-segment smoke on db_row** (format expansion).  
6. **P5 — MODULE_REGISTRY + timings CSV + playlist flags** (operability / Routinizer).  
7. **P6 — Render modes + optional cloud farm** (only after warm path is fast).

Do **not** invent a second renderer (three.js etc.) while Cycles + Day 4 grade are the brand look — AMP switched engines only after measured parity; gymform’s brand *is* the Blender/Day 4 look (**INFERENCE**).

### 3.4 Expected time savings

| Change | Basis | Expected effect |
|---|---|---|
| Motion cache hit | AMP shot cache warm 4.5→1.1 min; FlipCut TTS/maps skip | Skip MediaPipe+retarget on regrade / look A/B → **~30–50%** wall cut on warm runs (**ASSUMPTION**, measure) |
| Preflight before Cycles | FlipCut fail-fast | Avoid full beauty on bad pose → save **full render** on fail cases |
| Shot templates | FlipCut/AMP “no per-episode code” | New exercise setup **hours → minutes** of YAML |
| Segment composer | 4 grades share one motion bake | 4-segment cold ≈ **1× retarget + 4× grade**, not 4× full pipeline |
| Constraint bake | Owner: prevent errors upstream | Fewer `--max-iter` loops (each loop ≈ full re-render today) |
| Probe/sheet modes | AMP `stills_only` / FlipCut stills | Review cycle **minutes** instead of full reel |

Instrument every run into `output/_rig3d/timings.csv` so P1–P4 claims become FACT.

---

## 4. Operating rules (carry from siblings + owner)

- **0 generations**; free/CC0/YMove-pose-only; never commit clips/GLB/HDRI/tokens.
- **nice 19** + loadgate; yield to bots in active chat.
- **Never publish** without approval file / explicit yes.
- **Clay default, charcoal kept** (t734u).
- **Constraint first, QC last** (t736u).
- **Reference bar:** @demicstory quality or better; keep raising (t737u).
- **Engine/modules required** for reuse + speed — same direction as GlobeFlip / Hidden Systems (t738u).
- Do not edit other bots’ folders; do not burn paid APIs / Higgsfield / `/workspace/bgm`.

---

## 5. Immediate next deliverables (after this doc)

1. `docs/mission/MODULE_REGISTRY.md` (gymform) — interfaces + cache keys only.  
2. Implement L1 motion cache behind existing `pipeline.py` stamps (no API break).  
3. Preflight stage: posture/anatomy on bake before beauty.  
4. When quality_v3 sheet is owner-OK: db_row 4-segment smoke in chat.

---

## 6. Source index

| Source | Path / id |
|---|---|
| GlobeFlip modules | `/workspace/next6/eng_test/geo_template/MODULES.md` |
| GlobeFlip playlist factory | `/workspace/next6/eng_test/PLAYLIST_FACTORY.md` |
| GlobeFlip factory engines | `/workspace/next6/eng_test/ggt_template/FACTORY.md` |
| GlobeFlip transcript | agent `19a2f5b0-fccd-4979-8cb2-f169850ce2f7` |
| Hidden Systems registry | `/workspace/autonomous-media-pipeline/MODULE_REGISTRY.md` |
| Hidden Systems state | `/workspace/autonomous-media-pipeline/PROJECT_STATE.md` |
| Hidden Systems explode engine | `/workspace/autonomous-media-pipeline/docs/engine.md` |
| Hidden Systems transcript | agent `d1b926c0-aad1-4468-b38c-3666b246ca90` |
| gymform pipeline | `docs/mission/RIG3D_PIPELINE.md`, `tools/rig3d/*`, `modules/`, `playlist.yaml` |
