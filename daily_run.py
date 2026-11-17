#!/usr/bin/env python3
"""07:00 BUILD step of the unattended @gymform.3d pipeline (idempotent, one reel per day).

  nice -n 19 /workspace/venv-pilots/bin/python /workspace/daily_info_reels/daily_run.py

Takes the next unused episode from topics/queue.json, renders it (pilots/make_reel.py, style B2), runs all QA
gates (one retry with tweaks, then the next episode; max 2 episodes per day), writes caption + cover and an
approval file approvals/<date>_autoreel_<id>.json with status "scheduled" and publish_at <date>T10:00+09:00.
Queue selection is the approval (owner's standing rule). Only failures alert (Telegram ❌).
The 10:00 step (publish_run.py) posts it.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoreel import caption, common, cover  # noqa: E402
from autoreel.render import render_with_retry  # noqa: E402

NAME = "build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today KST)")
    ap.add_argument("--approvals-dir", default=common.APPROVALS)
    ap.add_argument("--force", action="store_true", help="rebuild even if today is already scheduled")
    ap.add_argument("--max-episodes", type=int, default=2)
    ap.add_argument("--notify-dry-run", action="store_true", help="tests: Telegram notifier runs with --dry-run")
    a = ap.parse_args()
    common.NOTIFY_DRY = a.notify_dry_run
    date = a.date or common.today()
    with common.Lock(NAME):
        st = common.state()
        day = st["days"].get(date)
        if day and day.get("status") in ("scheduled", "published") and not a.force:
            common.log(NAME, f"{date} already {day['status']} ({day.get('episode')}); nothing to do")
            return 0
        common.wait_for_box(NAME)
        q = common.load_json(common.QUEUE, {"episodes": []})
        todo = [e for e in q["episodes"] if e not in st["used"]]
        common.log(NAME, f"{date}: queue {len(q['episodes'])}, unused {len(todo)}")
        if not todo:
            st["days"][date] = {"status": "failed", "reason": "queue empty", "at": common.iso()}
            common.write_json(common.STATE, st)
            common.notify("❌", "gymform.3d 주제 큐 비었음", "오늘 만들 릴스 주제가 없어요. 새 주제 묶음이 필요해요.", name=NAME)
            return 2
        out_dir = os.path.join(common.WORK, date)
        chosen = qa = None
        for rel in todo[: a.max_episodes]:
            ep_path = os.path.join(common.REPO, rel)
            qa, tweak = render_with_retry(ep_path, out_dir, NAME)
            st["used"][rel] = {"date": date, "pass": bool(qa.get("pass")), "tweak": tweak, "qa": qa.get("passed")}
            common.write_json(common.STATE, st)
            if qa.get("pass"):
                chosen = rel
                break
            bad = [k for k, g in qa.get("gates", {}).items() if not g["ok"]] or ["render error"]
            common.log(NAME, f"{rel} FAILED after retry: {bad}")
            common.notify("❌", "gymform.3d 릴스 품질검사 실패", "오늘 주제 하나가 품질검사에서 떨어져서 다음 주제로 넘어가요.", name=NAME)
        if not chosen:
            st["days"][date] = {"status": "failed", "reason": "QA", "at": common.iso()}
            common.write_json(common.STATE, st)
            common.notify("❌", "gymform.3d 오늘 릴스 없음", "품질검사를 통과한 릴스가 없어서 오늘은 예약하지 않았어요.", name=NAME)
            return 1
        ep = json.load(open(os.path.join(common.REPO, chosen)))
        stem = f"{ep['id']}_B"
        mp4 = os.path.join(out_dir, stem + ".mp4")
        cap_txt = caption.build(ep)
        lint = caption.lint(cap_txt)
        if lint:
            st["days"][date] = {"status": "failed", "reason": "caption lint: " + "; ".join(lint), "at": common.iso()}
            common.write_json(common.STATE, st)
            common.notify("❌", "gymform.3d 오늘 릴스 없음", "캡션 점검에서 걸려서 오늘은 예약하지 않았어요.", name=NAME)
            return 1
        cap_path = os.path.join(out_dir, stem + "_caption.txt")
        open(cap_path, "w", encoding="utf-8").write(cap_txt)
        cov = cover.make(mp4, os.path.join(out_dir, stem + "_cover.jpg"), ep)
        rel_ = lambda p: os.path.relpath(p, common.REPO)  # noqa: E731
        appr = {
            "pipeline": "autoreel",
            "episode": chosen,
            "exercise": ep["exercise"],
            "title": cap_txt.split("\n", 1)[0],
            "status": "scheduled",
            "publish_at": f"{date}T10:00+09:00",
            "mp4": rel_(mp4),
            "cover": rel_(cov),
            "caption": rel_(cap_path),
            "sheet": rel_(os.path.join(out_dir, stem + "_sheet.jpg")),
            "qa": f"{qa['passed']} PASS",
            "approved_by": "topic queue (owner standing rule: queue selection = approval)",
            "approved_at": common.iso(),
            "created_at": common.iso(),
        }
        os.makedirs(a.approvals_dir, exist_ok=True)
        ap_path = os.path.join(a.approvals_dir, f"{date}_autoreel_{ep['id']}.json")
        common.write_json(ap_path, appr)
        st["days"][date] = {"status": "scheduled", "episode": chosen, "approval": ap_path, "at": common.iso(),
                            "render_s": qa.get("total_s")}
        common.write_json(common.STATE, st)
        common.log(NAME, f"{date}: scheduled {ep['id']} ({qa['passed']}, {qa.get('total_s')} s) -> {rel_(ap_path) if ap_path.startswith(common.REPO) else ap_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
