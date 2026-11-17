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
  - With `--render`, a full render plus all 16 QA gates.
  - With `--add`, episodes that pass are appended to the queue.
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
 "qa": "16/16 PASS", "approved_by": "topic queue (owner standing rule: queue selection = approval)",
 "approved_at": "...", "created_at": "..."}
```
After posting, `publish_run.py` adds `media_id`, `permalink`, `published_at` and `container_id`, and sets `status` to `"published"`.
- A claim written before the upload means an upload that dies halfway is never retried; it raises ❌ instead.
- A reel more than 6 h past `publish_at` is marked `"missed"` and raises ❌ once.

The routinizer watcher (`gymform_publish_watch.py`) only takes `"approved"` files, so it ignores these.
