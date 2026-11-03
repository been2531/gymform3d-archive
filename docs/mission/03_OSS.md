# 03 — Open source, assets, licences (checked 2026-11-03 KST; GitHub API `pushed_at`, raw licence files)

| Tool / asset | Licence & commercial use | Maintenance / fit |
|---|---|---|
| Blender + bpy (github.com/blender/blender, blender.org/about/license) | GPL-3.0+; renders/output are yours, commercial OK | pushed 2026-11-03; bpy 5.2.3 already on box (headless, CPU) |
| Rigify (bundled addon) | GPL (Blender); generated rigs usable commercially | ships with 5.2.3, present in addons_core; not used yet |
| FFmpeg (ffmpeg.org/legal.html) | LGPL-2.1+ (GPL with some codecs); using it is fine | pushed 2026-11-03; 7.1.5 on box, core of current pipeline |
| Remotion (github.com/remotion-dev/remotion LICENSE.md) | Free for individuals, for-profit orgs with **up to 3 employees**, non-profits; Company License required above that; v5 licence change announced | pushed 2026-11-03; 4.0.528 declared in shorts-labs kinetic (not installed). Good for captions/overlays, not needed for core |
| OpenCV (github.com/opencv/opencv) | Apache-2.0, commercial OK | pushed 2026-11-02; 5.0.0 on box |
| MediaPipe (github.com/google-ai-edge/mediapipe) | Apache-2.0; pose model file on box | pushed 2026-11-03; 1.1.1 on box, used for keypoints/callouts |
| FreeMoCap (github.com/freemocap/freemocap) | AGPL-3.0 (tool); captured data is ours | pushed 2026-11-03; needs 2+ webcams and a human performer → human step |
| CMU Mocap (mocap.cs.cmu.edu/faqs.php) | "may be copied, modified, or redistributed without permission" | Static archive (ASF/AMC, BVH conversions by 3rd parties); gym-lift coverage limited — check per clip |
| Mixamo (helpx.adobe.com Mixamo FAQ) | Royalty-free for personal/commercial/non-profit; needs Adobe ID | Free; few gym-specific lifts; Adobe ID = human sign-in |
| Bandai Namco Motion dataset | CC BY-NC 4.0 → **not for a monetised channel** | avoid |
| SMPL-X (github.com/vchoutas/smplx) | Non-commercial research licence | avoid; also blocks WHAM/4D-Humans (code MIT, but need SMPL) |
| GVHMR (video → 3D motion) | Non-commercial; commercial by email | avoid unless licensed |
| MakeHuman / MPFB2 (github.com/makehumancommunity/mpfb2) | Code GPL-3.0; bundled assets CC0 (LICENSE.ASSETS.md) | pushed 2026-10-24; Blender addon that adds Rigify-compatible rigs → best fit for our CC0 mannequin |
| Z-Anatomy (github.com/Z-Anatomy/Models-of-human-anatomy) | CC BY-SA 4.0 (License.txt): attribution + share-alike | pushed 2026-11-02; Blender anatomy atlas (muscles/bones) — SA terms need owner OK before use in published reels |
| BodyParts3D (already in `assets3d/bp3d`) | CC BY 4.0 (attribution in STUDIO3D.md) | static; already used for bones |
| Playwright (github.com/microsoft/playwright) | Apache-2.0 | pushed 2026-11-03; venv-pw on box; for IG/UI checks only |
| Vitest (github.com/vitest-dev/vitest) | MIT | pushed 2026-11-02; only if Remotion/TS layer is adopted; Python side should use pytest (MIT, not installed) |
| edge-tts (current VO) | Python lib free; uses Microsoft's online voice service without an official licence → ToS risk flag | works today; alternatives on box: venv-kokoro (Kokoro, Apache-2.0 model), venv-melo, venv-supertonic |
| Grok Bot GenerateVideo | Included in the bot; supports first/last frame + up to 4 pinned keyframes, ≤15 s | already used for Day 7; key lever for loops and ROM |

Cost: everything above is free. Paid/blocked items: xAI paid API (forbidden by owner), Remotion Company licence (only if the business has >3 employees), Muscle & Motion app animations (their terms: paid subscription + non-commercial only → not usable), cloud GPU render (not proposed).
