"""Render one episode with make_reel.py (style B2) and read its QA. One retry with
parameter tweaks chosen from the failing gates; otherwise the episode is marked failed."""
import json
import os
import subprocess
from .common import PY, REPO, log

TWEAKS = {  # failing gate -> episode override for the single retry
    "narration_fits_cues": {"speed": 1.2},
    "loudness_-14pm1_LUFS": {"B2": {"bed": False}},
    "title_max_2_lines": {"B2": {"title_font": "kalam"}},
    "text_inside_frame": {"B2": {"title_font": "kalam"}},
}


def render(ep_path, out_dir, tag="B", override=None):
    cmd = ["nice", "-n", "19", PY, os.path.join(REPO, "pilots", "make_reel.py"), "--style", "B2",
           "--episode", ep_path, "--out", out_dir, "--tag", tag]
    if override:
        cmd += ["--override", json.dumps(override)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    ep = json.load(open(ep_path))
    qa_path = os.path.join(out_dir, f"{ep['id']}_{tag}_qa.json")
    if not os.path.exists(qa_path):
        return {"pass": False, "error": (p.stderr or p.stdout)[-600:]}
    return json.load(open(qa_path))


def render_with_retry(ep_path, out_dir, name="render"):
    qa = render(ep_path, out_dir)
    if qa.get("pass"):
        return qa, None
    failed = [k for k, g in qa.get("gates", {}).items() if not g["ok"]]
    ov = {}
    for k in failed:
        for kk, vv in TWEAKS.get(k, {}).items():
            ov[kk] = {**ov.get(kk, {}), **vv} if isinstance(vv, dict) else vv
    log(name, f"QA fail {failed or qa.get('error')}; retry with {ov or 'no tweak'}")
    if not ov:
        return qa, None
    qa2 = render(ep_path, out_dir, override=ov)
    return qa2, ov
