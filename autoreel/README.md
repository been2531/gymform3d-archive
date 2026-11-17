# autoreel: unattended daily @gymform.3d reel (style B2 pen stickman)

Scripts only (no AI, no paid APIs). One reel per day. Choosing a topic from the queue counts as approval
(the owner's standing rule). Only failures send an alert.

| time (KST) | command |
|---|---|
| 07:00 build | `nice -n 19 /workspace/venv-pilots/bin/python /workspace/daily_info_reels/daily_run.py` |
| 10:00 publish | `nice -n 19 /workspace/venv-pilots/bin/python /workspace/daily_info_reels/publish_run.py` |

Dry run of the publish step (reads Instagram only, sends nothing): add `--dry-run`. Add `--simulate-due` to treat the scheduled reel as due now.

## Modules
- `topics/queue.json` + `topics/episodes/*.json`: the topic queue, used in order. Each entry is used once (tracked in `output/autoreel/state.json`).
- `autoreel/validate.py`: batch gate for new episodes. Run `python -m autoreel.validate topics/episodes/*.json --render --add`.
  - Static checks first.
  - With `--render`, a full render plus all QA gates (including `ig_spec`).
  - With `--add`, episodes that pass are appended to the queue.
- `autoreel/ig_spec.py`: Instagram Reels encode + container spec (single source of truth).
  - Encode: 1080x1920 30 fps, H.264 High yuv420p, 7 Mbps CBR (VBV 14 Mbps; ABR undershoots on these scenes), AAC-LC 48 kHz stereo 192k, `+faststart`.
  - Loudness stays -14 LUFS (two-pass loudnorm in `pilots/audio.py`; stereo mux is -3 dB/ch).
  - Gate `ig_spec` (ffprobe of the final mp4) fails the build if any of: not 1080x1920 9:16, fps not 30, codec not h264/yuv420p, audio not AAC stereo 44.1/48 kHz, moov not before mdat, video bitrate under 4 Mbps, duration outside 3-90 s, file over 300 MB.
  - `publish_run.py` runs the same gate before upload and refuses (Telegram ❌) on fail.
- `autoreel/render.py`: render plus QA. If QA fails, it retries once with tweaks; if the retry also fails, the episode is marked failed and the next one is tried (at most 2 per day).
- `autoreel/caption.py`: builds the caption (3 English lines, no hashtags) and runs the `gymform/caption.py` lint.
- `autoreel/cover.py`: cover frame.
- `daily_run.py`: build step. Writes the approval file `approvals/<date>_autoreel_<id>.json` (format below).
- `publish_run.py`: publish step. Uses the `tools/ig_publish.py` functions:
  - archive push to get signed URLs
  - REELS container
  - `media_publish`
  - records `media_id` and `permalink`, then sets `status` to `published`
- Telegram goes through `/home/box/routinizer/notify_telegram.py`: ✅ with the link on success, ❌ on failure. If that script is missing, the alert is skipped and logged.

## Approval file (status "scheduled")
```json
{"pipeline": "autoreel", "episode": "topics/episodes/<id>.json", "exercise": "...", "title": "<caption line 1>",
 "status": "scheduled", "publish_at": "<date>T10:00+09:00",
 "mp4": "output/autoreel/<date>/<id>_B.mp4", "cover": "..._cover.jpg", "caption": "..._caption.txt", "sheet": "..._sheet.jpg",
 "qa": "26/26 PASS", "approved_by": "topic queue (owner standing rule: queue selection = approval)",
 "approved_at": "...", "created_at": "..."}
```
After posting, `publish_run.py` adds `media_id`, `permalink`, `published_at` and `container_id`, and sets `status` to `"published"`.
- A claim written before the upload means an upload that dies halfway is never retried; it raises ❌ instead.
- A reel more than 6 h past `publish_at` is marked `"missed"` and raises ❌ once.

The routinizer watcher (`gymform_publish_watch.py`) only takes `"approved"` files, so it ignores these.

## Style B2 modules (v4, applies to every episode)
- Target muscles (v5, simplified): each episode lists `"muscles"` (a scene may override it with its own `"muscles"`).
  - The worked limb or torso segments themselves are re-inked in orange, slightly thicker, with a gentle pulse on the lift. No extra shapes.
  - Orange appears only on right-form figures and on front views. Wrong-form panels show none.
  - Front views never draw glutes, colour thighs only from mid-thigh down, and keep chest/core on the torso line above the hips.
  - Gate `no_highlight_in_groin_zone` checks the groin exclusion zone.
  - Gate `named_muscle_is_highlighted`: any muscle named on screen (GLUTES, CORE, QUADS ...) must be highlighted in that frame. The validator checks the same thing statically.
- Faults: only the faulty segment is re-inked in red, with a thicker stroke (no glow band).
- Standing exercises (squat, hinge, lunge) use a tall layout.
  - The figure is at least 50% of the frame height (gate).
  - The cue goes under the floor, with no rep counter.
  - Close-ups keep the same full-body framing (no zoom without cropping).
- Rep timing: lower 40%, hold at the bottom 15%, lift 35%, top 10%, eased.
  - `rep_period` must be at least 2.2 s.
  - Every reps, split and close-up scene lasts at least one full rep.
  - Reels are 10–15 s long (currently 13.4 s).
- Framing: horizontal poses (push-up, plank, bridge) are framed hips-to-head, and the feet may run off the edge.
  - Close-ups never crop the head or body.
- Faces: side figures get profile faces that look where the chest faces; front scenes keep front faces.
- Fault highlight: a thick pulsing red band under the faulty segments, which are also re-inked in red.
- Wrong forms stay visible over the whole rep (`rig.fault()`: 55% at the top, 100% at the bottom).
- Think pose variants are scratch, shrug and hold. Ending variants are flex, victory and thumbs.
  - Set them with `"variant"` on the scene.
  - Neighbouring queue episodes must differ (the validator checks this).
- QA gates added in v4:
  - `target_muscle_visible`
  - `min_rep_duration_ge_2_2s`
  - `figure_fill`
  - `profile_face_matches_body`
  - `fault_highlight_strong`
  - `wrong_form_readable_all_rep`
  - `think_end_variant_valid`
  - plus the queue variety check in `validate.py`
- QA gate `ig_spec` (Reels container/encode): 1080x1920 9:16, 30 fps, h264/yuv420p, AAC stereo 44.1/48 kHz, faststart, video at least 4 Mbps, 3-90 s, at most 300 MB. Safe-zone gates (`bottom_15pct_clear`, `text_inside_frame`) are unchanged.
