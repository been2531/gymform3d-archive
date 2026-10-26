#!/usr/bin/env python3
"""
@gymform.lab Instagram Reels Publisher (Instagram API with Instagram Login, graph.instagram.com).
- IG User ID: 17841416467054351
- Token File: /home/box/.kairo/ig_token.json (or env KAIRO_IG_ACCESS_TOKEN)
- Public media hosting: GitHub Pages been2531/been2531.github.io under gymform-media/<slug>/
- Safety guard: NEVER calls /media_publish unless both `--publish` and `--approved` are passed.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

IG_ID = "17841416467054351"
API = "https://graph.instagram.com/v22.0"
TOKEN_FILE = Path("/home/box/.kairo/ig_token.json")
OWNER, REPO, SUB = "been2531", "been2531.github.io", "gymform-media"
BASE_URL = f"https://{OWNER}.github.io/{SUB}"
CLONE = Path("/workspace/.gymform-pages")


def get_token() -> str:
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        if data.get("access_token"):
            return data["access_token"].strip()
    env_tok = os.environ.get("KAIRO_IG_ACCESS_TOKEN", "").strip()
    if env_tok:
        return env_tok
    raise SystemExit("Missing token in /home/box/.kairo/ig_token.json and KAIRO_IG_ACCESS_TOKEN")


def api_call(method: str, path: str, **params):
    params["access_token"] = get_token()
    encoded = urllib.parse.urlencode(params).encode()
    url = f"{API}/{path}"
    req = urllib.request.Request(
        url + ("?" + encoded.decode() if method == "GET" else ""),
        data=None if method == "GET" else encoded,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace").replace(params["access_token"], "[REDACTED]")
        raise SystemExit(f"API {method} {path} -> HTTP {e.code}: {body[:800]}")


def sh(*args, check=True):
    r = subprocess.run(args, cwd=CLONE if CLONE.exists() and args[0] == "git" else None, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"cmd failed: {' '.join(args[:4])}\n{r.stderr.strip()[-600:]}")
    return r.stdout.strip()


def host_files(slug: str, files: dict) -> dict:
    if not (CLONE / ".git").exists():
        if CLONE.exists():
            shutil.rmtree(CLONE)
        subprocess.run(
            ["gh", "repo", "clone", f"{OWNER}/{REPO}", str(CLONE), "--", "--filter=blob:none", "--no-checkout", "-q"],
            check=True,
            capture_output=True,
        )
        sh("git", "sparse-checkout", "set", "--no-cone", f"/{SUB}/")
        sh("git", "checkout", "-q", "main")
    gh = shutil.which("gh") or "gh"
    sh("git", "config", "--local", "--replace-all", "credential.helper", f"!{gh} auth git-credential")
    sh("git", "fetch", "-q", "origin", "main")
    sh("git", "reset", "-q", "--keep", "origin/main")

    dst = CLONE / SUB / slug
    dst.mkdir(parents=True, exist_ok=True)
    for name, src_path in files.items():
        shutil.copy2(src_path, dst / name)

    sh("git", "add", "--sparse", "--", f"{SUB}/{slug}")
    staged = [p for p in sh("git", "diff", "--cached", "--name-only").splitlines() if p]
    if staged:
        sh("git", "-c", f"user.name={OWNER}", "-c", f"user.email={OWNER}@users.noreply.github.com", "commit", "-qm", f"{SUB}: add {slug}")
        for _ in range(3):
            if subprocess.run(["git", "push", "-q", "origin", "HEAD:main"], cwd=CLONE, capture_output=True).returncode == 0:
                break
            sh("git", "pull", "-q", "--rebase", "origin", "main")

    urls = {name: f"{BASE_URL}/{slug}/{name}" for name in files}
    for name, u in urls.items():
        t0 = time.time()
        while time.time() - t0 < 300:
            try:
                req = urllib.request.Request(u + f"?v={int(time.time())}", method="HEAD")
                with urllib.request.urlopen(req, timeout=15) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(6)
    return urls


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: ig_publish.py <slug> <mp4_path> <caption_file> [--dry-run | --publish --approved]")
        sys.exit(1)
    slug, mp4_path, caption_file = args[0], args[1], args[2]
    do_publish = ("--publish" in args) and ("--approved" in args) and ("--dry-run" not in args)
    if "--publish" in args and not do_publish:
        raise SystemExit("Refusing to publish without both --publish and --approved")

    me = api_call("GET", "me", fields="user_id,username,followers_count,media_count")
    print("Verified IG account:", json.dumps(me, ensure_ascii=False))
    if str(me.get("user_id")) != IG_ID:
        raise SystemExit(f"Unexpected IG_ID: {me.get('user_id')} != {IG_ID}")

    caption = Path(caption_file).read_text(encoding="utf-8").strip()
    urls = host_files(slug, {"reel.mp4": mp4_path})
    print("Hosted video URL:", urls["reel.mp4"])

    c = api_call(
        "POST",
        f"{IG_ID}/media",
        media_type="REELS",
        video_url=urls["reel.mp4"],
        caption=caption,
        share_to_feed="true",
    )
    cid = c["id"]
    print("Created REELS container:", cid)

    t0 = time.time()
    while time.time() - t0 < 300:
        st_resp = api_call("GET", cid, fields="status_code,status")
        st = st_resp.get("status_code")
        print("Container status:", st)
        if st == "FINISHED":
            break
        if st in ("ERROR", "EXPIRED"):
            raise SystemExit(f"Container failed: {st_resp}")
        time.sleep(10)

    if not do_publish:
        print("DRY-RUN complete. Container is FINISHED and ready to publish upon approval:", cid)
        return

    pub = api_call("POST", f"{IG_ID}/media_publish", creation_id=cid)
    mid = pub["id"]
    info = api_call("GET", mid, fields="id,permalink,media_type,media_product_type,timestamp")
    print("PUBLISHED:", json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
