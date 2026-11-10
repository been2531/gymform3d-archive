# DEMIC update + gymform 4-segment plan — 2026-11-10 (Asia/Seoul)

Owner ask (t731u): once quality settles, widen from 1 segment to ~4; also check how the reference account is doing now and borrow what fits.

Scope of this note: analysis only. No new third-party media downloaded into git. No Instagram login. No publish.

Sources used this turn:
- Local (box, not committed as media): `demic_bench/*.info.json` + contact sheets (fetched ~2026-10-29), `docs/mission/02_BENCHMARK.md` (2026-11-03), `docs/mission/QUALITY_UPGRADE_PLAN.md` (2026-11-10), `DAY4_TEMPLATE_SPEC.md`, `PIPELINE.md` (Day 6+ 4-step lock), `playlist.yaml`, `exercises/db_row.yaml`.
- Public web (no login): FAMA Jersey Demic page, WebSearch snippets of PicMany / HypeAuditor / Jolygram / Pictame / Snapchat Spotlights, EIN Presswire (2025-06-26 app launch).

## 1. Access limits (what we could NOT get live today)

| Source | Result 2026-11-10 ~09:14 KST |
|---|---|
| `instagram.com/demicstory/` | Login wall HTML only — no follower count, no live grid, no view counts. |
| PicMany `picmany.com/demicstory` | Cloudflare block (403 / "you have been blocked"). Counts below are from **WebSearch-cached PicMany text dated Nov 2, 2026**, not a fresh scrape this turn. |
| HypeAuditor | Cloudflare JS challenge — no live page body. Search snippet only. |
| YouTube `@demicstory/shorts` | 404 / unusable via curl this turn. |
| Instagram view counts | Never exposed in our `demic_bench/*.info.json` (likes only). Same limit as 02_BENCHMARK. |

Treat third-party aggregator follower numbers as ±~100k ballpark. Prefer relative trends over absolute precision.

## 2. Account status now (evidence)

### Followers / growth
| Estimate | Source | Date |
|---|---|---|
| **12,591,267** | PicMany via our `02_BENCHMARK.md` | 2026-11-02 |
| **12,588,680** (from 12,593,599 on Oct 29) | PicMany WebSearch cache | 2026-11-02 |
| **12.58M** IG; ~14.65M cross-platform; ranked #14 FAMA; **▼18.1K** recent loss | [FAMA Jersey Demic](https://www.famacharts.com/creators/jersey-demic) | fetched 2026-11-10 |
| ~12.3M, ER ~2.08%, avg likes ~12.7K, **30-day growth 0.00% ("Low")** | HypeAuditor search snippet | ~Nov 2026 |
| Bio / CTA line | "Save this and come back stronger" + Follow `@appyoucan` | Jolygram / Pictame snippets |

**Trend read:** plateau / slight net loss, not collapse. Still a mega account (~12.5M). Growth engine has shifted from "follower explosion" to **high-frequency reel views** (see below).

### Posting frequency (last ~1–2 months)
- PicMany cache (12 most recent posts, **Oct 28 – Nov 2, 2026**): **~3 reels/day**, all Reels.
- Those 12: **~9.59M plays total ≈ 799K views/reel**, **120K likes total**; top of that window **20,774 likes** (Oct 28).
- Local `demic_bench` Oct sample (with likes): Oct 8 / 20 / 26 — still posting multi-variant anatomy reels into late October.

### Recent topics that perform (local likes + aggregator titles)
**Local `demic_bench` (likes = IG like_count in info.json):**

| Date | ID | Likes | Dur | Topic |
|---|---|---|---|---|
| 2026-02-10 | DUlzpgekQRy | **63,674** | 34.3 s | 3 dumbbell moves → Front / Side / Rear delts |
| 2026-10-08 | DePzKDbFB0y | **20,033** | 32.5 s | Push-up elbow angle **90° vs 45° vs 20°** |
| 2026-10-20 | DetDsIDj2BJ | **17,018** | 39.1 s | Barbell row **grip-width variations** |
| 2026-10-26 | De9qPwyGFvK | **10,569** | 48.7 s | Strong Back = Dumbbell Edition (multi-move) |
| 2025-03-23 | DHj07hdPpMV | 15,668 | 20.0 s | Quads vs Glutes vs Adductor squat stance |

**PicMany recent titles (Oct 28 – Nov 2 window, views ~800K avg, likes not per-id in cache):** forearm training, pull-up form in 5 steps, 3-min abs, 3D shoulders DB-only, rear-delt face pull, home abs, shoulder-press mistakes, hack squat quads/glutes/adductors, lat-pulldown mistakes, back region splits (lats / upper back / traps), foot-position → muscle target.

**Pattern:** comparison / mistake / "which region" beats generic "do this workout". Barbell + dumbbell both in the mix. App promo still ends many anatomy reels.

### Other platforms (public)
- Threads: `@demicstory` (PicMany).
- Snapchat Spotlights under `@jerseydemic` (e.g. lateral raise, DB back — 2025 dates in Spotlight pages).
- FAMA lists additional platforms (~901.7K / 597.8K / 565.9K / 12.4K) without reliable labels in the HTML fetch — do not invent which is TikTok vs Facebook.
- Brand: DEMIC fitness app ("YOU CAN" / `@appyoucan`) — press release 2025-06-26; end-card App Store badge still in Oct 2026 contact sheets.

## 3. Current reel structure (from local sheets + ffprobe)

Measured on box (analysis only):

| Trait | Evidence |
|---|---|
| Length | Oct hits **32–49 s** (was often ~20 s in Mar 2025 squat edition). Feb shoulder reel 34.3 s. |
| Hard scene cuts (`scene>0.3`) | Only **~1–2** per reel → mostly continuous 3D camera / soft chip changes, not rapid jump-cuts. |
| Loudness | mean ≈ **−17.4 to −20.3 dB** → music bed, no spoken VO in these samples (same as 02_BENCHMARK). |
| Hook (≤1 s) | Muscle already glowing / overflow close-up **or** stacked variants with ✗/✓ + angle numbers at frame 0. |
| Body of reel | **3 stacked variants** (angles, grips, or FRONT/SIDE/REAR), each ~4–8 s, top **chip / torso icons** change colour with the active variant. |
| Wrong vs right | Present when the topic is form (✗ red / ✓ green on DePz… 90° vs 45°). Many top posts are **variant comparison**, not classic wrong→right. |
| Graphics | Coloured bar/handle path, sets×reps (`3x10`), tiny labels (FRONT/SIDE/REAR, 45°), muscle % callouts on some frames. |
| Character | One consistent **faceless light-grey clay** mannequin, black shorts, dark grey studio + floor mat. |
| Muscle look | Target only, saturated orange/purple/green, soft volumetric glow, fibre still readable. |
| End card | App UI screen → black **"YOU CAN"** + App Store badge (~last 3–5 s). |
| Caption / CTA | Emoji title claim; short teaching body; **"Save this"** / "Train smarter"; bio line "come back stronger". Older 2025 posts used many hashtags; Oct 2026 teaching reels often **0–few hashtags**. |
| Series naming | Soft series by body region ("Dumbbell Edition", "Squat Edition"), not a numbered Day 1…N curriculum. |

**Important:** Demic's *current* winning grammar is **multi-variant comparison + app CTA**, not our Day 4/6 **bone → muscle × wrong → right** four-box. Bone/muscle layer language is closer to Muscle & Motion / SynerMuscle / our own locked template. We borrow Demic's *hooks, chips, glow behaviour, length band, save CTA* — we keep gymform's 4-step identity.

## 4. What changed vs our first benchmark (2026-11-03 / demic_bench ~10-29)

| Then (`02_BENCHMARK`) | Now (this update) |
|---|---|
| Followers ~12.59M | Still ~12.5–12.6M; **flat / slight loss** confirmed by FAMA ▼18.1K + PicMany Oct 29→Nov 2 dip. |
| Top local likes 63.6K (Feb), 20K / 17K (Oct) | Same local set; aggregator adds ~3/day cadence and ~800K avg views on newest dozen. |
| Length 32–39 s cited for Demic | Oct 26 reel stretches to **48.7 s**; cadence of posting is higher than we documented. |
| Hook = glow close-up / ✗✓ variants | Unchanged and still dominant. |
| End = app screen + logo | Unchanged in Oct sheets (YOU CAN / App Store). |
| Audio = music only | Unchanged (−17 to −20 dB). |
| New vs notes | Mistake/foot-position/region-split titles flooding late Oct–early Nov feed (PicMany titles); growth is **view-driven**, not follower-driven. |

Quality gap vs our delivered `db_row` is unchanged in kind: lighting, SSS/maps, true 3D glow, floor — already ranked in `QUALITY_UPGRADE_PLAN.md`. Finish that before betting the 4-segment widen.

## 5. Borrow list for @gymform.3d (ranked by impact × free-rig3d feasibility)

### Borrow (do)
1. **Frame-0 hook** — target muscle already lit within 0.3 s; optional punch-in. *(High. Shared camera preset in `retarget_render_v2` / grade. No new assets.)*
2. **One measured number on screen** — elbow/torso/knee degrees we already compute in bones.npz. Demic's "90° vs 45° vs 20°" is the template. *(High. Callout already exists; extend to multi-slot.)*
3. **Top preview chips / 4-slot pill** — Day 4 capsule already coded; widen 1-slot → 4-slot when n_steps=4. Active slot border = red wrong / green right. *(High. Layout reuse.)*
4. **Wrong vs right colour grammar** — red/crimson vs emerald glow only on target muscles; no ✕/✓ badge stickers (Day 4 lock: zero noisy 2D badges). *(High. `grade.glow: wrong|right` already in grade3d.)*
5. **Variant / mistake topics** — elbow flare, grip width, stance, torso angle. Playlist `fault:` fields already sketch this. *(High for saves. Content, not shaders.)*
6. **Length band 18–36 s** — our locked 4×5 s = **20 s** sits under Demic's 32–49 s but matches our Day 4 masters and SynerMuscle's short-loop evidence. Optional later: 4×7 s = 28 s A/B. *(Medium.)*
7. **Music bed, English TTS coaching** — keep our edge-tts (Demic is music-only; our VO is a differentiator for form cues). Mean loudness target ≈ −18 dB. *(Medium. Already in pipeline.)*
8. **Caption CTA** — exactly 3 English lines, 0 hashtags (RULES): title / one question / "Save this…" Soft Demic echo without their hashtag spam or app handle. *(Medium. `caption.py` lint.)*
9. **Dark studio + contact shadow floor** — already in QUALITY_UPGRADE P0. *(High for look parity; shared light rig.)*
10. **Barbell + dumbbell mix** — Demic Oct row/back posts prove barbell anatomy reels work; our `playlist.yaml` already queues bb_back_squat / deadlift / bench after quality gate. *(Medium–high once prop wiring lands.)*

### Do NOT copy
- **App end cards / `@appyoucan` / "YOU CAN" logo** — their product funnel, not ours.
- **Light-clay mannequin + orange brand glow as identity** — gymform stays charcoal + neon green/red unless owner A/B asks (QUALITY_UPGRADE brand note).
- **Branded ribbed shorts / custom heroic sculpt** — paid art; out of free Grab3D scope.
- **Hashtag walls** from 2025 home-workout posts.
- **Chair / wall / "no gym needed" diversion** as the main series — our lane is gym biomechanics (DB + bodyweight + now barbell).
- **Third-party video, their motion files, or paid OpenGym3D** into git / pipeline without explicit yes.
- **xAI / grok.com generation** for bone twins — old Day 6 path; rig3d must render bone + muscle passes itself (0 generation).
- **Exact music tracks** or any commercial BGM under `/workspace/bgm`.

## 6. Recommended gymform 4-segment structure (mapped to rig3d)

Keep the **Day 6+ locked order** from `PIPELINE.md` (matches 중현 님's "wrong bone → right bone → wrong muscle → right muscle"):

| Seg | t (20.0 s master) | Show | Glow / look | Callout | TTS (one line) |
|---|---|---|---|---|---|
| **1** | 0.0–5.0 | **Wrong form · bone X-Ray** | Silver/cyan bones, no muscle paint; red chip active | Mistake angle (e.g. elbow flare 90°) | Name the fault |
| **2** | 5.0–10.0 | **Right form · bone X-Ray** | Same camera + tempo as seg 1; green chip | Correct angle (e.g. 45°) | Name the fix |
| **3** | 10.0–15.0 | **Wrong form · muscle** | Red/crimson target + compensators lit | Same number, muscle context | Why it misses the target |
| **4** | 15.0–20.0 | **Right form · muscle** (cover = peak) | Emerald primary muscles only | Confirm number | Save / question line |

Hook rule: seg 1 frame 0 already shows wrong bones lit (or a 0.4 s punch-in on the fault joint) so the feed thumbnail isn't a dead grey mannequin.

### Pipeline mapping (free, scriptable)
```
exercises/<slug>.yaml
  motions.wrong  ← YMove retarget + fault offsets (playlist.fault → posture/auto-fix inverse)
  motions.right  ← same clip, corrected targets (current db_row path)
  renders: beauty + bone_mask + muscle_mask (extend retarget_render_v2 passes)
grade3d.py
  n_steps: 4
  per-step: bone|muscle × wrong|right  (grade.glow + bone overlay mode)
layout
  preview_pill slots: 4 (reuse Day 4 capsule geometry; slots at x 303/461/619/777)
qc3d.py
  pair sync (wrong↔right tempo), order_lint, glow colour, overlay clear, posture on BOTH motions
pipeline.py
  --exercise <slug> [--segments 4]   # 1-seg remains default until quality GO
deliver.py
  sheet with 4 peaks + email (existing default)
```

Feasibility notes:
- **Bone pass:** render armature / bone capsules or a translucent body + bone emission from `bones.npz` — no xAI edit. Prototype on `db_row` only after P0 lighting/glow lands.
- **Wrong motion:** derive from `fault:` in `playlist.yaml` (knee cave, elbow flare, torso swing) as retarget offsets + QC must FAIL the "right" checks and PASS a "mistake signature" check (same spirit as `gymform.motion_check`).
- **Cost:** ~4× beauty frames if naive; mitigate with shared pose cache, lower spp on bone-only pass, and resume stamps. Run `nice -n 19`, detached.
- **Gate:** do **not** widen the public playlist to 4-seg until (a) quality charcoal/clay stills approved, (b) one `db_row` 4-seg smoke sheet emailed, (c) owner yes. Until then 1-seg production continues.

### Optional later A/B (not default)
- Demic-style **3-variant** reel (A/B/C angles, no bone layer) for topic days — only if 4-seg underperforms on watch time.
- 28 s (4×7) if 20 s avg-watch saturates.

## 7. Suggested next actions (no publish)

1. Finish QUALITY_UPGRADE P0 (light + maps + 3D glow + floor + skin_lock) on `db_row` stills → owner tone pick.
2. Smoke **one** 4-seg `db_row` sheet (wrong bone / right bone / wrong muscle / right muscle) via script; email `daily25310@gmail.com`; do not upload.
3. Only after yes: turn `n_steps: 4` on for playlist head (`db_goblet_squat` …), keep approval-file gate on IG publish.
4. Refresh this note monthly with aggregator snapshots (PicMany/FAMA) — Instagram login still not required.

## 8. One-line verdict

Demic is still ~12.5M, posting ~3×/day, winning with **glow-first hooks + measured variant comparisons + save CTA**, while follower growth is flat; gymform should steal that grammar into our **already-locked 4-step bone/muscle template**, after the shared lighting/glow upgrade, and never copy their app funnel or clay brand identity.
