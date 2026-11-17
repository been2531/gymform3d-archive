#!/usr/bin/env python3
"""10:00 PUBLISH step of the unattended @gymform.3d pipeline (idempotent, max one post per day).

  nice -n 19 /workspace/venv-pilots/bin/python /workspace/daily_info_reels/publish_run.py            # live
  nice -n 19 /workspace/venv-pilots/bin/python /workspace/daily_info_reels/publish_run.py --dry-run  # read-only

Posts the approvals/*.json written by daily_run.py (pipeline "autoreel", status "scheduled", publish_at <= now,
not older than 6 h). Same prechecks and steps as the routinizer watcher: ffprobe 9:16 5-90 s, caption lint, cover,
no unrelated staged git changes, duplicate check against the last 25 IG posts, nothing else posted since
publish_at-3h, never anything referencing DfGabTBjwt8. A claim is written before upload; a claim without a
media_id is never retried. Publishing reuses tools/ig_publish.py (archive push -> signed URL -> REELS container
-> media_publish). Telegram: ✅ with the link on success, ❌ on failure.
--dry-run: no git, no container, no post; only IG reads (GET me, me/media).
"""
from __future__ import annotations
import argparse
import datetime as dt
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoreel import caption, common  # noqa: E402

NAME = "publish"
STALE_H = 6
DONE = ("media_id", "published_at", "permalink", "deleted_at")


def load_igp():
    spec = importlib.util.spec_from_file_location("ig_publish", os.path.join(common.REPO, "tools", "ig_publish.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def ig(igp, method, path, **kw):
    try:
        return igp.api_call(method, path, **kw)
    except SystemExit as e:  # ig_publish exits on HTTP errors
        raise RuntimeError(str(e)[:300])


def parse(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def prechecks(d):
    errs = []
    mp4 = os.path.join(common.REPO, d.get("mp4", ""))
    if not os.path.isfile(mp4):
        errs.append("mp4 missing")
    else:
        pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=width,height:format=duration", "-of", "json", mp4],
                                       capture_output=True, text=True).stdout or "{}")
        try:
            w, h = pr["streams"][0]["width"], pr["streams"][0]["height"]; dur = float(pr["format"]["duration"])
            if abs(w / h - 9 / 16) > 0.01: errs.append("not 9:16")
            if not 5 <= dur <= 90: errs.append("duration")
        except (KeyError, IndexError):
            errs.append("ffprobe failed")
    cap = os.path.join(common.REPO, d.get("caption", ""))
    if not os.path.isfile(cap):
        errs.append("caption missing")
    else:
        errs += ["caption: " + e for e in caption.lint(open(cap, encoding="utf-8").read())]
    if d.get("cover") and not os.path.isfile(os.path.join(common.REPO, d["cover"])):
        errs.append("cover missing")
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=common.REPO, capture_output=True, text=True).stdout.split()
    if [s for s in staged if s not in (d.get("mp4"), d.get("cover"))]:
        errs.append("repo has unrelated staged changes")
    if any(x in json.dumps(d) for x in common.NEVER_REPOST):
        errs.append("never-repost id referenced")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--approvals-dir", default=common.APPROVALS)
    ap.add_argument("--simulate-due", action="store_true", help="dry-run only: treat scheduled items as due now")
    a = ap.parse_args()
    if a.simulate_due and not a.dry_run:
        raise SystemExit("--simulate-due requires --dry-run")
    common.NOTIFY_DRY = a.dry_run
    now = common.now()
    with common.Lock(NAME + ("_dry" if a.dry_run else "")):
        files = sorted(glob.glob(os.path.join(a.approvals_dir, "*.json")))
        items = [(p, common.load_json(p, {})) for p in files]
        # one post per day: anything (any pipeline) already published today?
        today = now.strftime("%Y-%m-%d")
        for p, d in items:
            if str(d.get("published_at", "")).startswith(today) and not a.simulate_due:
                common.log(NAME, f"already published today ({os.path.basename(p)}); nothing to do")
                return 0
        due = []
        for p, d in items:
            if d.get("pipeline") != "autoreel" or d.get("status") != "scheduled" or any(d.get(k) for k in DONE):
                continue
            if d.get("claim"):
                common.log(NAME, f"{os.path.basename(p)} has an unresolved claim; not retrying")
                common.notify("❌", "gymform.3d 업로드 확인 필요", "이전 업로드가 중간에 끊겼어요. 인스타에 올라갔는지 확인해 주세요.", name=NAME)
                continue
            pa = now - dt.timedelta(minutes=1) if a.simulate_due else parse(d["publish_at"])
            if pa > now:
                continue
            if (now - pa).total_seconds() > STALE_H * 3600:
                common.log(NAME, f"{os.path.basename(p)} publish_at is >{STALE_H} h old; not auto-posting")
                if not a.dry_run:  # alert once, not every day
                    d.update(status="missed", missed_at=common.iso()); common.write_json(p, d)
                common.notify("❌", "gymform.3d 오늘 릴스 미게시", "게시 시간을 놓쳐서 자동으로 올리지 않았어요.", name=NAME)
                continue
            due.append((p, d, pa))
        if not due:
            common.log(NAME, "nothing due")
            return 0
        p, d, pa = sorted(due, key=lambda x: x[2])[0]
        name = os.path.basename(p)
        errs = prechecks(d)
        if errs:
            common.log(NAME, f"{name} BLOCKED: {errs}")
            common.notify("❌", "gymform.3d 오늘 릴스 미게시", "업로드 전 점검에서 걸려서 올리지 않았어요.", name=NAME)
            return 1
        igp = load_igp()
        cap_text = open(os.path.join(common.REPO, d["caption"]), encoding="utf-8").read().strip()
        title = " ".join(cap_text.split("\n", 1)[0].split())
        try:
            me = ig(igp, "GET", "me", fields="user_id,username")
            recent = ig(igp, "GET", "me/media", fields="id,caption,timestamp,permalink", limit=25).get("data", [])
        except RuntimeError as e:
            common.log(NAME, f"IG read failed: {str(e)[:120]}")
            common.notify("❌", "gymform.3d 오늘 릴스 미게시", "인스타그램 연결 확인에 실패해서 올리지 않았어요.", name=NAME)
            return 1
        norm = lambda s: " ".join((s or "").split())  # noqa: E731
        dup = next((m for m in recent if norm(m.get("caption")) == norm(cap_text)
                    or norm((m.get("caption") or "").split("\n", 1)[0]) == title), None)
        if dup:
            common.log(NAME, f"{name}: already on IG {dup.get('permalink')}; adopting")
            if not a.dry_run:
                d.update(status="published", media_id=dup["id"], permalink=dup.get("permalink"), published_at=dup.get("timestamp"),
                         published_by="detected_existing(autoreel)")
                common.write_json(p, d)
            return 0
        near = [m for m in recent if parse(m["timestamp"][:22] + ":" + m["timestamp"][22:]) >= pa - dt.timedelta(hours=3)]
        if near and not a.simulate_due:
            common.log(NAME, f"{name}: another post since publish_at-3h ({near[0].get('permalink')}); not posting")
            common.notify("❌", "gymform.3d 오늘 릴스 보류", "최근 3시간 안에 다른 게시물이 있어서 보류했어요.", name=NAME)
            return 1
        if a.dry_run:
            common.log(NAME, f"DRY-RUN: WOULD PUBLISH {d['mp4']} as @{me.get('username')} (title {title!r}); prechecks OK, no duplicate")
            return 0
        d["claim"] = {"at": common.iso(), "pid": os.getpid()}
        common.write_json(p, d)
        try:
            mp4 = Path(os.path.join(common.REPO, d["mp4"]))
            cov = Path(os.path.join(common.REPO, d["cover"])) if d.get("cover") else None
            urls = igp.host_in_archive_and_get_signed_urls(mp4, cov if cov and cov.exists() else None)
            kw = dict(media_type="REELS", video_url=urls["reel.mp4"], caption=cap_text, share_to_feed="true")
            if "cover.jpg" in urls:
                kw["cover_url"] = urls["cover.jpg"]
            cid = ig(igp, "POST", f"{igp.IG_ID}/media", **kw)["id"]
            t0 = time.time()
            while True:
                s = ig(igp, "GET", cid, fields="status_code,status")
                if s.get("status_code") == "FINISHED":
                    break
                if s.get("status_code") in ("ERROR", "EXPIRED") or time.time() - t0 > 600:
                    raise RuntimeError(f"container {s.get('status_code')}")
                time.sleep(8)
            mid = ig(igp, "POST", f"{igp.IG_ID}/media_publish", creation_id=cid)["id"]
            info = ig(igp, "GET", mid, fields="id,permalink,timestamp")
        except Exception as e:  # noqa: BLE001
            common.log(NAME, f"{name} publish FAILED: {type(e).__name__} {str(e)[:200]} (claim kept, no auto-retry)")
            common.notify("❌", "gymform.3d 업로드 실패", "오늘 릴스가 올라가지 않았어요.", name=NAME)
            return 1
        d.pop("claim", None)
        d.update(status="published", media_id=info["id"], permalink=info.get("permalink"), published_at=common.iso(),
                 container_id=cid, published_by="autoreel")
        common.write_json(p, d)
        st = common.state(); day = st["days"].setdefault(p.split("/")[-1][:10], {})
        day.update(status="published", permalink=info.get("permalink")); common.write_json(common.STATE, st)
        common.log(NAME, f"{name} PUBLISHED {info.get('permalink')}")
        common.notify("✅", "gymform.3d 릴스 게시 완료", title[:60], link=info.get("permalink") or "", name=NAME)
        return 0


if __name__ == "__main__":
    sys.exit(main())
