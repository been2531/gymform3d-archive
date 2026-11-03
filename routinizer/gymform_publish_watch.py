#!/usr/bin/env python3
"""GymForm 3D approved-reel publisher (replaces Fitness Studio's daily "GymForm 3D 10:00 승인 릴스 업로드" routine).

Cheap loop: every cycle only stats/parses /workspace/daily_info_reels/approvals/*.json. Network happens only
when an item is DUE:
  status == "approved", approved_by + approved_at set, publish_at <= now,
  and no media_id / published_at / permalink / deleted_at / routinizer_claim.
Anything else (pending, ready_for_approval, published, deleted_by_user, ...) is ignored. Items whose publish_at is
more than STALE_H hours old are never auto-posted (alert instead) so a late script can't surprise-post.

Before publishing (all must pass, else alert + no post):
  - mp4 exists, ffprobe ok (video stream, 9:16, 5-90 s); caption file exists and passes gymform.caption.lint
  - no other IG post since publish_at-3h (else alert possible_duplicate)
  - cover (if named) exists; daily_info_reels has no unrelated STAGED changes (ig_publish commits what is staged)
  - IG token works (GET /me); recent IG media (last 25) has no post with the same caption or same title line -> if found, the
    routine/human already posted: adopt its media_id/permalink into the json and skip.
Publishing reuses tools/ig_publish.py (archive push -> signed URL -> REELS container -> media_publish).
A `routinizer_claim` is written into the approval json BEFORE any upload; a claim without media_id is never
retried automatically (alert `claim_unresolved`) -> no double posts.

Yields to heavy Shorts renders (loadavg > cores or a render process), but never delays a due post > 15 min.

Usage:
  gymform_publish_watch.py --once --dry-run   # read-only (GET /me, GET /me/media only; no container, no git)
  gymform_publish_watch.py --interval 120     # LIVE loop
"""
import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rz_common as rz  # noqa: E402

NAME = "gymform_publish"
REPO = os.environ.get("GF_REPO", "/workspace/daily_info_reels")
APPROVALS = os.path.join(REPO, "approvals")
OVERDUE_S = 15 * 60
STALE_H = 6
DEFER_S = 0  # set by --defer-min
DONE_KEYS = ("media_id", "published_at", "permalink", "deleted_at")


def load_ig_publish():
    spec = importlib.util.spec_from_file_location("ig_publish", os.path.join(REPO, "tools", "ig_publish.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_caption_lint():
    spec = importlib.util.spec_from_file_location("gf_caption", os.path.join(REPO, "gymform", "caption.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.lint


def ig(igp, method, path, **params):
    """ig_publish.api_call raises SystemExit on HTTP errors; turn that into RuntimeError."""
    try:
        return igp.api_call(method, path, **params)
    except SystemExit as e:
        raise RuntimeError(str(e)[:400])


def scan(st):
    """Return list of (path, data) approvals. Uses mtime cache so unchanged files aren't re-parsed."""
    cache = st.data.setdefault("mtimes", {})
    out = []
    for p in sorted(glob.glob(os.path.join(APPROVALS, "*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except ValueError:
            m = os.stat(p).st_mtime
            if cache.get(p) != m:
                rz.alert(st, "approval_unreadable", os.path.basename(p), "approval json is not valid JSON")
            cache[p] = m
            continue
        cache[p] = os.stat(p).st_mtime
        out.append((p, d))
    return out


def is_candidate(d):
    return (d.get("status") == "approved" and d.get("approved_by") and d.get("approved_at")
            and d.get("publish_at") and not any(d.get(k) for k in DONE_KEYS))


def prechecks(d, lint):
    errs = []
    mp4 = os.path.join(REPO, d.get("mp4", ""))
    if not d.get("mp4") or not os.path.isfile(mp4) or os.path.getsize(mp4) == 0:
        errs.append(f"mp4 missing: {d.get('mp4')}")
    else:
        try:
            pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height:format=duration", "-of", "json", mp4],
                                           capture_output=True, text=True, timeout=60).stdout)
            w, h = pr["streams"][0]["width"], pr["streams"][0]["height"]
            dur = float(pr["format"]["duration"])
            if abs(w / h - 9 / 16) > 0.01:
                errs.append(f"not 9:16 ({w}x{h})")
            if not 5 <= dur <= 90:
                errs.append(f"duration {dur:.1f}s outside 5-90s")
        except Exception as e:  # noqa: BLE001
            errs.append(f"ffprobe failed: {e}")
    cap = os.path.join(REPO, d.get("caption", ""))
    if not d.get("caption") or not os.path.isfile(cap):
        errs.append(f"caption missing: {d.get('caption')}")
    else:
        le = lint(open(cap, encoding="utf-8").read())
        if le:
            errs.append("caption lint: " + "; ".join(le))
    if d.get("cover") and not os.path.isfile(os.path.join(REPO, d["cover"])):
        errs.append(f"cover missing: {d['cover']}")
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=REPO, capture_output=True, text=True).stdout.split()
    mine = {d.get("mp4"), d.get("cover")}
    other = [s for s in staged if s not in mine]
    if other:
        errs.append(f"repo has unrelated staged changes ({len(other)}), ig_publish would commit them")
    return errs


def norm(s):
    return " ".join((s or "").split())


def update_json(path, **kv):
    d = json.load(open(path, encoding="utf-8"))
    d.update(kv)
    rz.atomic_write_json(path, d)


def publish(igp, d, cap_text):
    """Same steps as tools/ig_publish.py main() with --publish --approved."""
    mp4 = __import__("pathlib").Path(os.path.join(REPO, d["mp4"]))
    cover = __import__("pathlib").Path(os.path.join(REPO, d["cover"])) if d.get("cover") else None
    urls = igp.host_in_archive_and_get_signed_urls(mp4, cover if cover and cover.exists() else None)
    kw = dict(media_type="REELS", video_url=urls["reel.mp4"], caption=cap_text, share_to_feed="true")
    if "cover.jpg" in urls:
        kw["cover_url"] = urls["cover.jpg"]
    cid = ig(igp, "POST", f"{igp.IG_ID}/media", **kw)["id"]
    t0 = time.time()
    while time.time() - t0 < 600:
        s = ig(igp, "GET", cid, fields="status_code,status")
        if s.get("status_code") == "FINISHED":
            break
        if s.get("status_code") in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"container {cid} {s}")
        time.sleep(8)
    else:
        raise RuntimeError(f"container {cid} not FINISHED after 600s")
    mid = ig(igp, "POST", f"{igp.IG_ID}/media_publish", creation_id=cid)["id"]
    info = ig(igp, "GET", mid, fields="id,permalink,timestamp")
    return cid, info


def cycle(st, dry):
    t = rz.now()
    items = scan(st)
    due = []
    for p, d in items:
        name = os.path.basename(p)
        if d.get("routinizer_claim") and not any(d.get(k) for k in DONE_KEYS):
            rz.alert(st, "claim_unresolved", name, "publish was started by routinizer but no media_id recorded; check IG manually, then clear routinizer_claim",
                     severity="error", repeat_hours=12, dry_run=dry)
            continue
        if not is_candidate(d):
            continue
        pa = rz.parse_time(d["publish_at"])
        if pa > t:
            continue
        age = (t - pa).total_seconds() - DEFER_S
        if age < 0:
            continue  # deferred: give the still-active routine the first chance
        if age > STALE_H * 3600:
            rz.alert(st, "stale_due", name, f"approved but publish_at {d['publish_at']} is >{STALE_H}h old; not auto-posting",
                     repeat_hours=24, dry_run=dry)
            continue
        due.append((p, d, age))
    if not due:
        st.heartbeat(result="nothing_due", approvals=len(items), dry_run=dry)
        return
    busy, why = rz.box_busy()
    if busy and all(age <= OVERDUE_S for _, _, age in due):
        rz.log(NAME, "yield:", why)
        st.heartbeat(result="yielded", reason=why, due=[os.path.basename(p) for p, _, _ in due])
        return

    igp = load_ig_publish()
    lint = load_caption_lint()
    try:
        me = ig(igp, "GET", "me", fields="user_id,username")
        recent = ig(igp, "GET", "me/media", fields="id,caption,timestamp,permalink", limit=25).get("data", [])
    except RuntimeError as e:
        kind = "token_expired" if ("190" in str(e) or "OAuth" in str(e)) else "ig_api_error"
        rz.alert(st, kind, "instagram", f"IG read failed: {str(e)[:200]}", severity="error", dry_run=dry)
        st.heartbeat(result=kind)
        return
    report = []
    for p, d, age in due:
        name = os.path.basename(p)
        cap_text = open(os.path.join(REPO, d["caption"]), encoding="utf-8").read().strip() if d.get("caption") and os.path.isfile(os.path.join(REPO, d["caption"])) else ""
        title = norm(cap_text.split("\n", 1)[0]) if cap_text else ""
        dup = next((m for m in recent if cap_text and (norm(m.get("caption")) == norm(cap_text)
                    or norm((m.get("caption") or "").split("\n", 1)[0]) == title)), None)
        pa = rz.parse_time(d["publish_at"])
        near = [m for m in recent if rz.parse_time(m["timestamp"][:22] + ":" + m["timestamp"][22:]) >= pa - __import__("datetime").timedelta(hours=3)]
        if not dup and near:
            rz.alert(st, "possible_duplicate", name, f"IG already has a post since publish_at-3h ({near[0].get('permalink')}) with a different title; not posting",
                     severity="error", repeat_hours=12, dry_run=dry)
            report.append((name, f"BLOCKED possible duplicate {near[0].get('permalink')}"))
            continue
        if dup:
            report.append((name, f"SKIP already on IG {dup['id']} {dup.get('permalink')}"))
            if not dry:
                update_json(p, status="published", media_id=dup["id"], permalink=dup.get("permalink"),
                            published_at=dup.get("timestamp"), published_by="detected_existing(routinizer)")
            continue
        errs = prechecks(d, lint)
        if errs:
            rz.alert(st, "precheck_failed", name, " | ".join(errs), severity="error", repeat_hours=12, dry_run=dry)
            report.append((name, "BLOCKED " + " | ".join(errs)))
            continue
        if dry:
            report.append((name, f"WOULD PUBLISH {d.get('mp4')} as @{me.get('username')} (due {int(age // 60)} min ago)"))
            continue
        update_json(p, routinizer_claim=dict(at=rz.iso(), pid=os.getpid()))
        try:
            cid, info = publish(igp, d, cap_text)
        except Exception as e:  # noqa: BLE001
            rz.alert(st, "publish_failed", name, f"{str(e)[:300]} (claim left in json; not retried)", severity="error")
            report.append((name, "FAIL"))
            continue
        update_json(p, status="published", media_id=info["id"], permalink=info.get("permalink"),
                    published_at=rz.iso(), container_id=cid, published_by="routinizer")
        rz.alert(st, "published", name, f"Published {info.get('permalink')}", severity="info", repeat_hours=24 * 365)
        report.append((name, f"PUBLISHED {info.get('permalink')}"))
    for n, what in report:
        rz.log(NAME, n, what)
    st.data["history"].append(dict(at=rz.iso(), dry=dry, report=report))
    st.heartbeat(result="ok", dry_run=dry, report=report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--defer-min", type=int, default=0,
                    help="act only N minutes after publish_at (use ~30 while the old 10:00 routine is still active)")
    ap.add_argument("--simulate-due", metavar="APPROVAL_JSON",
                    help="dry-run only: treat this approval as approved+due now (tests prechecks + IG reads)")
    a = ap.parse_args()
    global DEFER_S
    DEFER_S = a.defer_min * 60
    if a.simulate_due and not a.dry_run:
        raise SystemExit("--simulate-due requires --dry-run")
    with rz.SingleInstance(NAME + ("_dry" if a.dry_run else "")):
        st = rz.State(NAME + ("_dry" if a.dry_run else ""))
        if a.simulate_due:
            global scan, is_candidate
            real_scan = scan

            def scan(s):  # noqa: F811
                out = []
                for p, d in real_scan(s):
                    if os.path.basename(p) == os.path.basename(a.simulate_due):
                        d = {k: v for k, v in d.items() if k not in DONE_KEYS + ("routinizer_claim",)}
                        d.update(status="approved", publish_at=rz.iso(rz.now() - __import__("datetime").timedelta(minutes=20)))  # overdue: bypasses yield
                    out.append((p, d))
                return out
        while True:
            try:
                cycle(st, a.dry_run)
            except Exception as ex:  # noqa: BLE001
                rz.alert(st, "watcher_error", "cycle", f"{type(ex).__name__}: {str(ex)[:300]}", severity="error", dry_run=a.dry_run)
                st.heartbeat(result="error", error=str(ex)[:300])
            st.save()
            if a.once:
                break
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
