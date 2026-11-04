#!/usr/bin/env python3
"""Deliver a rig3d result sheet to 중현 님 with NO bot turn (Routinizer v2).
Backends (tried in this order when enabled; --dry-run never sends, pushes or writes the alert):
  ntfy        true phone push. Needs env GYMFORM_NTFY_TOPIC (a private topic he subscribes to in the ntfy app;
              optional GYMFORM_NTFY_SERVER, default https://ntfy.sh). Sends title + the sheet link (Click) + the PNG.
  smtp        e-mail with the PNG attached. Needs env GYMFORM_SMTP_USER + GYMFORM_SMTP_APP_PASSWORD (Gmail app password)
              + GYMFORM_NOTIFY_TO (optional GYMFORM_SMTP_HOST, default smtp.gmail.com:465).
  repo        (always available, default) commit sheet + qc + caption + 540p preview to
              results/rig3d/<slug>/<run_id>/ in this repo (public been2531/gymform3d-archive), push with the repo's own
              credential helper, then append one line to /home/box/routinizer/state/alerts_for_routinizer.jsonl with the
              phone-viewable GitHub link. Allow-listed paths only + secret scan; never `git add -A`.
  pages       DRY-RUN ONLY in this version: stages results into a local clone of been2531/been2531.github.io
              (gymform/<slug>/<run_id>/index.html) and prints the URL it would get. Publishing would make it public.
usage: python3 tools/rig3d/deliver.py RUN_DIR CFG.json [--backend auto|repo|ntfy|smtp|pages] [--dry-run]"""
import argparse, datetime as dt, json, os, re, shutil, smtplib, subprocess, sys, urllib.request
from email.message import EmailMessage
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALERTS = "/home/box/routinizer/state/alerts_for_routinizer.jsonl"
REPO_WEB = "https://github.com/been2531/gymform3d-archive"
KST = dt.timezone(dt.timedelta(hours=9))
SECRET = re.compile(rb"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xai-[A-Za-z0-9]{20,}|EAA[A-Za-z0-9]{40,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)")

def sh(cmd, cwd=ROOT, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError(f"{cmd[:3]} failed: {r.stderr.strip()[-300:]}")
    return r.stdout.strip()

def summary(run_dir):
    qc = json.load(open(os.path.join(run_dir, "qc.json"))); meta = json.load(open(os.path.join(run_dir, "run.json")))
    fails = [c["name"] for c in qc["checks"] if not c["pass"]]
    return qc, meta, f"{meta['slug']} rig3d v2: QC {'PASS' if qc['pass'] else 'FAIL'} {qc['n_pass']}/{qc['n']}" + (f" (fail: {', '.join(fails)})" if fails else "")

def stage_results(run_dir, cfg):
    qc, meta, line = summary(run_dir); rel = f"results/rig3d/{meta['slug']}/{meta['run_id']}"; dst = os.path.join(ROOT, rel); os.makedirs(dst, exist_ok=True)
    files = {"sheet.png": meta["sheet"], "qc.json": os.path.join(run_dir, "qc.json"), "caption.txt": os.path.join(run_dir, "reel/caption.txt"),
             "run.json": os.path.join(run_dir, "run.json")}
    for k, v in files.items():
        if os.path.exists(v): shutil.copy2(v, os.path.join(dst, k))
    prev = os.path.join(dst, "preview_540p.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-threads", "2", "-i", meta["reel"], "-vf", "scale=540:960", "-c:v", "libx264", "-crf", "28",
                    "-preset", "veryfast", "-c:a", "aac", "-b:a", "64k", prev], check=True)
    blob = f"{REPO_WEB}/blob/main/{rel}"
    md = [f"# {line}", "", f"![sheet](sheet.png)", "", f"- preview (our own render, 540p): [preview_540p.mp4](preview_540p.mp4)",
          f"- full reel on the box: `{os.path.relpath(meta['reel'], ROOT)}`", f"- run {meta['run_id']}, {meta.get('wall_s')} s wall, peak RAM {meta.get('peak_rss_mb')} MB",
          f"- 0 generations; source pose {cfg['source']['provider']} (clip not committed); body Grab3D CC0; HDRI Poly Haven CC0", ""]
    open(os.path.join(dst, "README.md"), "w").write("\n".join(md))
    return rel, [os.path.join(rel, f) for f in sorted(os.listdir(dst))], blob, line

def secret_scan(paths):
    bad = [p for p in paths if SECRET.search(open(os.path.join(ROOT, p), "rb").read())]
    if bad: raise RuntimeError(f"secret pattern in {bad}")

def alert(kind, msg, extra, dry, severity="info"):
    rec = {"at": dt.datetime.now(KST).isoformat(timespec="seconds"), "watcher": "gymform_rig3d", "kind": kind, "key": f"gymform3d:{extra.get('slug')}:{extra.get('run_id')}",
           "severity": severity, "message": msg, **extra}
    if dry: print("DRY alert ->", ALERTS, json.dumps(rec, ensure_ascii=False)); return rec
    os.makedirs(os.path.dirname(ALERTS), exist_ok=True)
    with open(ALERTS, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

def backend_repo(run_dir, cfg, dry, code_paths=()):
    rel, paths, blob, line = stage_results(run_dir, cfg); secret_scan(paths)
    qc, meta, _ = summary(run_dir)
    if dry:
        print("DRY repo: would commit", paths, "->", blob); res = {"backend": "repo", "url": blob, "dry_run": True}
    else:
        sh(["git", "add", "--"] + list(paths) + list(code_paths))
        staged = sh(["git", "diff", "--cached", "--name-only"]).split("\n"); allowed = set(paths) | set(code_paths)
        extra = [p for p in staged if p and p not in allowed and not any(p.startswith(c.rstrip("/") + "/") for c in code_paths)]
        if extra: raise RuntimeError(f"refusing: unrelated staged files {extra[:5]}")
        sh(["git", "commit", "-m", f"rig3d {meta['slug']} {meta['run_id']}: {line} (pipeline result, 0 generations)", "--", *paths, *code_paths])
        sh(["git", "push", "origin", "HEAD:main"]); res = {"backend": "repo", "url": blob, "commit": sh(["git", "rev-parse", "--short", "HEAD"])}
    alert("rig3d_ready_for_review", f"{line} -> {blob}", {"slug": meta["slug"], "run_id": meta["run_id"], "url": blob, "sheet": meta["sheet"],
          "notify_owner": bool(cfg.get("_notify_owner", False)), "note": "repo fallback: no push channel to the owner configured"}, dry)
    return res

def backend_ntfy(run_dir, cfg, dry, url=None):
    topic = os.environ.get("GYMFORM_NTFY_TOPIC"); server = os.environ.get("GYMFORM_NTFY_SERVER", "https://ntfy.sh")
    if not topic: return {"backend": "ntfy", "skipped": "env GYMFORM_NTFY_TOPIC not set"}
    qc, meta, line = summary(run_dir)
    req = urllib.request.Request(f"{server}/{topic}", data=open(meta["sheet"], "rb").read(), method="PUT",
                                 headers={"Title": line[:120], "Filename": "sheet.png", **({"Click": url} if url else {}), "Tags": "muscle"})
    if dry: print("DRY ntfy PUT", f"{server}/<topic>", len(req.data), "bytes", line); return {"backend": "ntfy", "dry_run": True}
    urllib.request.urlopen(req, timeout=60).read(); return {"backend": "ntfy", "sent": True}

def backend_smtp(run_dir, cfg, dry, url=None):
    u, pw, to = (os.environ.get(k) for k in ("GYMFORM_SMTP_USER", "GYMFORM_SMTP_APP_PASSWORD", "GYMFORM_NOTIFY_TO"))
    if not (u and pw and to): return {"backend": "smtp", "skipped": "env GYMFORM_SMTP_USER / GYMFORM_SMTP_APP_PASSWORD / GYMFORM_NOTIFY_TO not set"}
    qc, meta, line = summary(run_dir); m = EmailMessage(); m["Subject"] = line; m["From"] = u; m["To"] = to
    m.set_content(f"{line}\n\n{url or ''}\nmp4 on the box: {meta['reel']}\n"); m.add_attachment(open(meta["sheet"], "rb").read(), maintype="image", subtype="png", filename="sheet.png")
    if dry: print("DRY smtp to <GYMFORM_NOTIFY_TO>", len(m.as_bytes()), "bytes", line); return {"backend": "smtp", "dry_run": True}
    host, _, port = os.environ.get("GYMFORM_SMTP_HOST", "smtp.gmail.com:465").partition(":")
    with smtplib.SMTP_SSL(host, int(port or 465), timeout=60) as s: s.login(u, pw); s.send_message(m)
    return {"backend": "smtp", "sent": True}

def backend_pages(run_dir, cfg, dry=True):
    qc, meta, line = summary(run_dir); clone = os.path.join(ROOT, "_cache_rig3d/pages_clone")
    if not os.path.isdir(os.path.join(clone, ".git")):
        sh(["git", "clone", "--depth", "1", "https://github.com/been2531/been2531.github.io.git", clone])
    rel = f"gymform/{meta['slug']}/{meta['run_id']}"; d = os.path.join(clone, rel); os.makedirs(d, exist_ok=True)
    shutil.copy2(meta["sheet"], os.path.join(d, "sheet.png"))
    open(os.path.join(d, "index.html"), "w").write(f"<!doctype html><meta name=viewport content='width=device-width'><title>{line}</title><h3>{line}</h3><img src=sheet.png style='width:100%'>")
    url = f"https://been2531.github.io/{rel}/"
    print("DRY pages: staged", rel, "in", clone, "(not committed, not pushed) ->", url)
    return {"backend": "pages", "dry_run": True, "url": url}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("cfg"); ap.add_argument("--backend", default="auto"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); cfg = json.load(open(a.cfg)); out = []
    if a.backend in ("auto", "repo"): out.append(backend_repo(a.run_dir, cfg, a.dry_run))
    url = out[0].get("url") if out else None
    if a.backend in ("auto", "ntfy"): out.append(backend_ntfy(a.run_dir, cfg, a.dry_run, url))
    if a.backend in ("auto", "smtp"): out.append(backend_smtp(a.run_dir, cfg, a.dry_run, url))
    if a.backend == "pages": out.append(backend_pages(a.run_dir, cfg))
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
