#!/usr/bin/env python3
"""
@gymform.3d Instagram Reels Publisher (Instagram API with Instagram Login, graph.instagram.com).
- IG User ID: 17841416467054351
- Token File: /home/box/.kairo/ig_token.json (or env KAIRO_IG_ACCESS_TOKEN)
- Media hosting: Pushes to been2531/gymform3d-archive and obtains temporary signed download_url via GitHub REST API (zero GitHub Pages build delay).
- Safety guard: NEVER calls /media_publish unless both `--publish` and `--approved` are passed.
"""
import json
import os
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
GH_TOKEN_FILE = Path("/home/box/.kairo/github_token")
REPO_DIR = Path("/workspace/daily_info_reels")
OWNER_REPO = "been2531/gymform3d-archive"


def get_ig_token() -> str:
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        if data.get("access_token"):
            return data["access_token"].strip()
    env_tok = os.environ.get("KAIRO_IG_ACCESS_TOKEN", "").strip()
    if env_tok:
        return env_tok
    raise SystemExit("Missing token in /home/box/.kairo/ig_token.json")


def get_gh_token() -> str:
    if GH_TOKEN_FILE.exists():
        return GH_TOKEN_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit("Missing GitHub token in /home/box/.kairo/github_token")


def api_call(method: str, path: str, **params):
    params["access_token"] = get_ig_token()
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


def host_in_archive_and_get_signed_urls(mp4_path: Path, cover_path: Path | None) -> dict:
    gh_tok = get_gh_token()
    rel_mp4 = mp4_path.resolve().relative_to(REPO_DIR.resolve())
    to_add = [str(rel_mp4)]
    rel_cov = None
    if cover_path and cover_path.exists():
        rel_cov = cover_path.resolve().relative_to(REPO_DIR.resolve())
        to_add.append(str(rel_cov))

    env = os.environ.copy()
    env["GH_TOKEN"] = gh_tok
    subprocess.run(["git", "add", "--"] + to_add, cwd=REPO_DIR, check=True, env=env)
    diff = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=REPO_DIR, capture_output=True, text=True, env=env).stdout.strip()
    if diff:
        subprocess.run(["git", "commit", "-qm", f"Publish media: {', '.join(to_add)}"], cwd=REPO_DIR, check=True, env=env)
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=REPO_DIR, check=True, env=env)

    def fetch_signed(rel_p):
        req = urllib.request.Request(
            f"https://api.github.com/repos/{OWNER_REPO}/contents/{rel_p}",
            headers={"Authorization": f"Bearer {gh_tok}", "Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())["download_url"]

    urls = {"reel.mp4": fetch_signed(rel_mp4)}
    if rel_cov:
        urls["cover.jpg"] = fetch_signed(rel_cov)
    return urls


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: ig_publish.py <slug> <mp4_path> <caption_file> [--dry-run | --publish --approved]")
        sys.exit(1)
    slug, mp4_str, caption_file = args[0], args[1], args[2]
    mp4_path = Path(mp4_str)
    do_publish = ("--publish" in args) and ("--approved" in args) and ("--dry-run" not in args)
    if "--publish" in args and not do_publish:
        raise SystemExit("Refusing to publish without both --publish and --approved")

    me = api_call("GET", "me", fields="user_id,username,followers_count,media_count")
    print("Verified IG account:", json.dumps(me, ensure_ascii=False), flush=True)

    caption = Path(caption_file).read_text(encoding="utf-8").strip()
    cover_candidate = Path(mp4_str.replace("_master_reel.mp4", "_cover.jpg").replace(".mp4", "_cover.jpg"))
    urls = host_in_archive_and_get_signed_urls(mp4_path, cover_candidate if cover_candidate.exists() else None)
    print("Signed URLs ready", flush=True)

    media_kwargs = dict(
        media_type="REELS",
        video_url=urls["reel.mp4"],
        caption=caption,
        share_to_feed="true",
    )
    if "cover.jpg" in urls:
        media_kwargs["cover_url"] = urls["cover.jpg"]
    c = api_call("POST", f"{IG_ID}/media", **media_kwargs)
    cid = c["id"]
    print("Created REELS container:", cid, flush=True)

    t0 = time.time()
    while time.time() - t0 < 300:
        st_resp = api_call("GET", cid, fields="status_code,status")
        st = st_resp.get("status_code")
        print("Container status:", st, flush=True)
        if st == "FINISHED":
            break
        if st in ("ERROR", "EXPIRED"):
            raise SystemExit(f"Container failed: {st_resp}")
        time.sleep(6)

    if not do_publish:
        print("DRY-RUN complete. Container is FINISHED:", cid, flush=True)
        return

    pub = api_call("POST", f"{IG_ID}/media_publish", creation_id=cid)
    mid = pub["id"]
    info = api_call("GET", mid, fields="id,permalink,media_type,media_product_type,timestamp")
    print("PUBLISHED:", json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
