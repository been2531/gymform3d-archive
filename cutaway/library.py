#!/usr/bin/env python3
"""Verified joint asset library + judge bookkeeping (weekly, agent-run batch).

Daily runs never generate images: they only use joints whose
output/cutaway/library/<joint>/verified.json says status == "verified".

  python3 cutaway/library.py sheet  --joint knee --layer bone --attempt 2   # judge sheet (candidate | refs)
  python3 cutaway/library.py record --joint knee --layer bone --attempt 2 --results '[true,true,...]' --notes '...'
  python3 cutaway/library.py promote --joint knee                           # copy accepted set + write verified.json
  python3 cutaway/library.py status

Candidates live in output/cutaway/assets/<joint>/ (raw + registered/). Judge log:
output/cutaway/library/judge_log.jsonl (one JSON line per judged layer attempt).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
LIB = os.path.join(REPO, "output", "cutaway", "library")
ASSETS = os.path.join(REPO, "output", "cutaway", "assets")
REFS = os.path.join(LIB, "_refs")
LOG = os.path.join(LIB, "judge_log.jsonl")
CHECK = json.load(open(os.path.join(HERE, "judge_checklists.json")))
MAX_REGEN = 2


def sheet(joint, layer, attempt):
    reg = os.path.join(ASSETS, joint, "registered")
    cand = cv2.imread(os.path.join(reg, f"{layer}.png"))
    skin = cv2.imread(os.path.join(reg, "skin.png"))
    H = 1100
    tiles = []
    for im in ([skin, cand] if layer != "skin" else [cand]):
        k = H / im.shape[0]
        tiles.append(cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
    for r in CHECK[joint]["refs"]:
        im = cv2.imread(os.path.join(REFS, r))
        k = H / im.shape[0]
        tiles.append(cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
    out = np.hstack(tiles)
    d = os.path.join(LIB, joint, "judge")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"attempt{attempt}_{layer}.jpg")
    cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(p)
    for i, it in enumerate(CHECK[joint][layer]):
        print(f"  [{i}] {it}")
    return p


def record(joint, layer, attempt, results, notes, auto=None):
    items = CHECK[joint][layer]
    assert len(results) == len(items), f"need {len(items)} results"
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "joint": joint, "layer": layer,
           "attempt": attempt, "items": [{"check": c, "pass": bool(r)} for c, r in zip(items, results)],
           "score": f"{sum(map(bool, results))}/{len(items)}", "full_pass": all(results),
           "judge": "agent vision vs public-domain Gray's Anatomy plates (Wikimedia Commons)",
           "notes": notes, "auto": auto or {}}
    os.makedirs(LIB, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


def latest(joint):
    res = {}
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            if r["joint"] == joint:
                res[r["layer"]] = r
    return res


def promote(joint):
    lat = latest(joint)
    need = ("skin", "outer", "bone", "macro")
    ok = all(lat.get(l, {}).get("full_pass") for l in need)
    d = os.path.join(LIB, joint)
    os.makedirs(d, exist_ok=True)
    if not ok:
        missing = [l for l in need if not lat.get(l, {}).get("full_pass")]
        json.dump({"joint": joint, "status": "dropped", "failed_layers": missing,
                   "date": dt.date.today().isoformat(), "judge": {l: lat.get(l) for l in need}},
                  open(os.path.join(d, "verified.json"), "w"), indent=1)
        print("DROPPED", joint, missing)
        return False
    src = os.path.join(ASSETS, joint)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p) and (f.endswith(("_raw.jpg", "_raw.png")) or f in ("macro_crop.json", "macro_src_crop.png")):
            shutil.copy2(p, d)
    shutil.copytree(os.path.join(src, "registered"), os.path.join(d, "registered"), dirs_exist_ok=True)
    reg = json.load(open(os.path.join(d, "registered", "registration.json")))
    lm = json.load(open(os.path.join(HERE, "joint_landmarks.json"))).get(joint, {})
    json.dump({"joint": joint, "status": "verified", "date": dt.date.today().isoformat(),
               "judge": {l: {"attempt": lat[l]["attempt"], "score": lat[l]["score"]} for l in need},
               "registration": reg, "landmarks": lm,
               "refs": CHECK[joint]["refs"]}, open(os.path.join(d, "verified.json"), "w"), indent=1)
    print("VERIFIED", joint)
    return True


def verified_joints():
    out = []
    if os.path.isdir(LIB):
        for j in sorted(os.listdir(LIB)):
            p = os.path.join(LIB, j, "verified.json")
            if os.path.exists(p) and json.load(open(p)).get("status") == "verified":
                out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "record", "promote", "status"])
    ap.add_argument("--joint")
    ap.add_argument("--layer")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--results")
    ap.add_argument("--notes", default="")
    ap.add_argument("--auto", default=None)
    a = ap.parse_args()
    if a.cmd == "sheet":
        sheet(a.joint, a.layer, a.attempt)
    elif a.cmd == "record":
        record(a.joint, a.layer, a.attempt, json.loads(a.results), a.notes, json.loads(a.auto) if a.auto else None)
    elif a.cmd == "promote":
        promote(a.joint)
    else:
        print("verified:", verified_joints())


if __name__ == "__main__":
    main()
