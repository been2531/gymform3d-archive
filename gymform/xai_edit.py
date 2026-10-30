"""Grok Imagine video edit (xAI API) for pair renders: same motion, different anatomy rendering.

Day 6+ pair rule: steps 1/3 share one motion and steps 2/4 share one motion. When a motion's base clip
has its anatomy baked in (e.g. muscle), the other rendering (e.g. bone X-Ray) is made by a *video edit*
of the base, which keeps the duration, framing, camera and motion and changes only the anatomy overlay.

API (docs.x.ai > Video Editing, checked 2026-10-30):
  POST https://api.x.ai/v1/videos/edits  {model: grok-imagine-video, prompt, video: {url | data URI | file_id}}
  -> {request_id};  poll GET /v1/videos/{request_id} until status == done (video.url) | failed | expired.
  Source mp4 <= 8.7 s, output keeps duration/aspect/resolution (<= 720p). ~$0.07 per output second at 720p.

Safety:
  * Key only from env XAI_API_KEY (or /home/box/.kairo/xai_api_key, chmod 600). Never printed/logged/committed.
  * ONE TAKE per target: every request is recorded in _cache_gymform/xai_edits.jsonl; a target that already
    has a ledger entry is refused unless --retake is passed explicitly (needs owner approval: costs money).
  * Existing target files are never overwritten.

Usage:
  python3 -m gymform.xai_edit --day 6 [--dry-run] [--only wrong|right] [--retake]
Config (motions.<m>.renders.bone): {"file": ..., "edit_of": <base>, "edit": {"color": "crimson"}}
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "output", "assets")
LEDGER = os.path.join(ROOT, "_cache_gymform", "xai_edits.jsonl")
API = "https://api.x.ai/v1"
MODEL = "grok-imagine-video"
USD_PER_S_720P = 0.07
MAX_SRC_S = 8.7

PROMPTS = {
    "bone": ("Change only the anatomy overlay: remove the red muscle colouring and show a glowing {color} X-ray "
             "skeleton (rib cage, clavicles, scapulae, upper-arm and forearm bones) inside the same silver "
             "mannequin. Keep everything else exactly as in the original: camera angle, framing, mannequin, "
             "black shorts, dumbbells, bench, studio background, lighting, and the exact same motion and timing."),
    "muscle": ("Change only the anatomy overlay: remove the X-ray skeleton and show glowing {color} muscle fibres "
               "on the target muscles inside the same silver mannequin. Keep everything else exactly as in the "
               "original: camera angle, framing, mannequin, black shorts, equipment, studio background, lighting, "
               "and the exact same motion and timing."),
}
DEFAULT_COLOR = {"wrong": "crimson", "right": "emerald"}


def _key():
    k = os.environ.get("XAI_API_KEY", "").strip()
    if not k:
        p = "/home/box/.kairo/xai_api_key"
        if os.path.exists(p):
            k = open(p).read().strip()
    if not k:
        sys.exit("xai_edit: XAI_API_KEY not available")
    return k


def _probe_s(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def _ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def _log(rec):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def jobs_for(cfg, only=None):
    """(motion, kind, src, dst, prompt) for every render that is an edit of a base and does not exist yet."""
    out = []
    for m, mo in (cfg.get("motions") or {}).items():
        if only and m != only:
            continue
        for kind, r in (mo.get("renders") or {}).items():
            if not r.get("edit_of") or r.get("file") == r.get("edit_of"):
                continue
            color = (r.get("edit") or {}).get("color") or DEFAULT_COLOR.get(m, "crimson")
            prompt = (r.get("edit") or {}).get("prompt") or PROMPTS[kind].format(color=color)
            out.append((m, kind, os.path.join(ASSETS, r["edit_of"]), os.path.join(ASSETS, r["file"]), prompt))
    return out


def edit_one(src, dst, prompt, key, poll_s=6, timeout_s=900):
    dur = _probe_s(src)
    if dur > MAX_SRC_S:
        raise RuntimeError(f"source {dur:.2f}s > {MAX_SRC_S}s")
    data = "data:video/mp4;base64," + base64.b64encode(open(src, "rb").read()).decode()
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    t0 = time.time()
    r = requests.post(f"{API}/videos/edits", headers=h, timeout=120,
                      json={"model": MODEL, "prompt": prompt, "video": {"url": data}})
    if r.status_code >= 300:
        raise RuntimeError(f"edit submit HTTP {r.status_code}: {r.text[:300]}")
    rid = r.json()["request_id"]
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "src": os.path.basename(src), "dst": os.path.basename(dst),
           "request_id": rid, "model": MODEL, "prompt": prompt, "src_s": round(dur, 3),
           "est_usd": round(dur * USD_PER_S_720P, 3), "status": "submitted"}
    _log(rec)
    print(f"[xai_edit] submitted {rec['dst']} request_id={rid}", flush=True)
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        g = requests.get(f"{API}/videos/{rid}", headers={"Authorization": f"Bearer {key}"}, timeout=60)
        if g.status_code >= 300:
            continue
        j = g.json()
        st = j.get("status")
        if st == "done":
            url = (j.get("video") or {}).get("url")
            tmp = dst + ".part"
            with requests.get(url, stream=True, timeout=300) as d:
                d.raise_for_status()
                with open(tmp, "wb") as f:
                    for c in d.iter_content(1 << 20):
                        f.write(c)
            os.replace(tmp, dst)
            rec.update(status="done", secs=round(time.time() - t0, 1), out_s=round(_probe_s(dst), 3),
                       usage=j.get("usage"))
            _log(rec)
            print(f"[xai_edit] done {rec['dst']} in {rec['secs']}s", flush=True)
            return rec
        if st in ("failed", "expired"):
            rec.update(status=st, secs=round(time.time() - t0, 1), detail=str(j)[:300])
            _log(rec)
            raise RuntimeError(f"edit {st}: {rec['detail']}")
    rec.update(status="timeout")
    _log(rec)
    raise RuntimeError("edit poll timeout (request_id in ledger; fetch later, do not resubmit)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--only", choices=["wrong", "right"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retake", action="store_true", help="allow a 2nd paid take (owner approval required)")
    a = ap.parse_args(argv)
    cfg = json.load(open(os.path.join(ROOT, "configs", f"day{a.day:02d}.json")))
    jobs = jobs_for(cfg, a.only)
    done = {r["dst"] for r in _ledger()}
    todo = []
    for m, kind, src, dst, prompt in jobs:
        b = os.path.basename(dst)
        if os.path.exists(dst):
            print(f"[xai_edit] exists, skip: {b}")
        elif b in done and not a.retake:
            print(f"[xai_edit] REFUSED (one take; already requested, see ledger): {b}")
        elif not os.path.exists(src):
            print(f"[xai_edit] missing source: {src}")
        else:
            todo.append((m, kind, src, dst, prompt))
    est = sum(_probe_s(s) * USD_PER_S_720P for _, _, s, _, _ in todo)
    print(f"[xai_edit] {len(todo)} edit(s), est ${est:.2f}")
    if a.dry_run or not todo:
        for t in todo:
            print("  ", t[0], t[1], os.path.basename(t[2]), "->", os.path.basename(t[3]))
        return 0
    key = _key()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(edit_one, s, d, p, key) for _, _, s, d, p in todo]
        errs = 0
        for f in futs:
            try:
                f.result()
            except Exception as e:  # noqa
                errs += 1
                print(f"[xai_edit] ERROR {e}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
