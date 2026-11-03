# Routinizer: GymForm 3D approved-reel publisher

Replaces Fitness Studio's daily "GymForm 3D 10:00 승인 릴스 업로드" routine. Reuses `tools/ig_publish.py`,
`gymform/caption.py` lint and the `approvals/*.json` format.

- Every 2 min: parse `approvals/*.json` only (no network). Due = `status: "approved"`, approved_by/at set,
  `publish_at <= now` (+ optional `--defer-min`), no media_id/published_at/permalink/deleted_at/routinizer_claim,
  and publish_at not older than 6 h.
- Pre-checks: mp4 (ffprobe 9:16, 5-90 s), caption lint, cover, no unrelated staged git changes, IG token,
  IG duplicate (same caption or same title line -> adopt existing media into json and skip; any other post since
  publish_at-3h -> alert, skip).
- `routinizer_claim` is written before upload; a claim without media_id is never retried (alert).
- Yields to heavy Shorts renders, never > 15 min past due.
- Alerts: `/home/box/routinizer/alerts.jsonl`; heartbeat: `/home/box/routinizer/state/gymform_publish.heartbeat.json`.
- Secrets: read only from `/home/box/.kairo/` by `tools/ig_publish.py` (never in git).

```bash
cd /home/box/routinizer
python3 repos/gymform3d-archive-routinizer/routinizer/gymform_publish_watch.py --once --dry-run
python3 repos/gymform3d-archive-routinizer/routinizer/gymform_publish_watch.py --once --dry-run --simulate-due 2026-10-30_day06.json
nohup nice -n 10 python3 -u repos/gymform3d-archive-routinizer/routinizer/gymform_publish_watch.py --interval 120 --defer-min 30 \
  >> logs/gymform_publish.log 2>&1 &     # LIVE; drop --defer-min once the old 10:00 routine is paused
```
