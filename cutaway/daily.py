#!/usr/bin/env python3
"""Daily LOAD METER run: pick the next queue topic, render it, write an approval-ready
package.  Never publishes and never writes an approval file.

    nice -n 19 python3 cutaway/daily.py            # today's package
    python3 cutaway/daily.py --dry-run             # show what would be picked
    python3 cutaway/daily.py --lint                # pre-render gates for every runnable topic

Pick rule (in queue order): fact has source + url; joint is in the verified library;
joint differs from the last packaged (passed) reel; not already packaged; fewer than
2 failed attempts.  Exit: 0 package ready, 2 render stopped by QA, 3 nothing runnable.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_topic  # noqa: E402

REPO = build_topic.REPO
DAILY = os.path.join(REPO, "output", "cutaway", "daily")
STATE = os.path.join(DAILY, "state.json")


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": []}


def pick(state):
    done = {r["entry"] for r in state["runs"] if r["status"] == "pass"}
    fails = {}
    for r in state["runs"]:
        if r["status"] != "pass":
            fails[r["entry"]] = fails.get(r["entry"], 0) + 1
    last_joint = next((r["joint"] for r in reversed(state["runs"]) if r["status"] == "pass"), None)
    skipped = []
    for e in build_topic.load_queue():
        why = None
        if e["id"] in done:
            continue
        if not (e.get("fact", {}).get("source") and e["fact"].get("url")):
            why = "no cited source"
        elif not build_topic.verified(e["joint"]):
            why = f"joint '{e['joint']}' not in verified library"
        elif e["joint"] == last_joint:
            why = f"same joint as last reel ({last_joint})"
        elif fails.get(e["id"], 0) >= 2:
            why = "failed QA twice"
        if why:
            skipped.append((e["id"], why))
            continue
        return e, skipped
    return None, skipped


def lint():
    import cutaway2 as C
    import make_reel
    import qa
    bad = 0
    for e in build_topic.load_queue():
        if not build_topic.verified(e["joint"]):
            print(f"SKIP {e['id']}: joint not verified")
            continue
        t = build_topic.build(e)
        adir = os.path.join(REPO, t["assets"])
        reg = make_reel.ensure_registered(adir)
        r = C.Reel(copy.deepcopy(t), adir, {})
        pre = qa.prerender(r, t, reg)
        failed = [g for g, v in pre.items() if not v["pass"]]
        bad += bool(failed)
        print(("OK  " if not failed else "FAIL"), e["id"], failed or "", r.layout_log or "")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lint", action="store_true")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    a = ap.parse_args(argv)
    if a.lint:
        return lint()
    os.makedirs(DAILY, exist_ok=True)
    state = load_state()
    e, skipped = pick(state)
    for sid, why in skipped:
        print(f"skip {sid}: {why}")
    if e is None:
        print("nothing runnable today")
        open(os.path.join(DAILY, f"{a.date}_NO_TOPIC.md"), "w").write(
            "# No runnable topic\n\n" + "\n".join(f"- {s}: {w}" for s, w in skipped) + "\n")
        return 3
    print("picked", e["id"], e["joint"])
    if a.dry_run:
        return 0
    import make_reel
    out = os.path.join(DAILY, f"{a.date}_{e['id']}")
    if os.path.exists(out):
        shutil.rmtree(out)
    rc = make_reel.main(["--entry", e["id"], "--out", out])
    status = "pass" if rc == 0 else "fail"
    f = e["fact"]
    pkg = {"date": a.date, "entry": e["id"], "joint": e["joint"], "action": e["action"], "value": e["value"],
           "series": "load_meter", "status": status, "published": False,
           "approval": "not requested by this script; a human approves before anything is posted",
           "source": f["source"], "url": f["url"], "claim": f["claim"]}
    json.dump(pkg, open(os.path.join(out, "package.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(out, "sources.md"), "w").write(
        f"# Source\n\n- On screen: {e['value']}x BODY WEIGHT - {e['action']}\n- Claim: {f['claim']}\n"
        f"- {f['source']}\n- {f['url']}\n")
    state["runs"].append({"date": a.date, "entry": e["id"], "joint": e["joint"], "status": status, "dir": out})
    json.dump(state, open(STATE, "w"), indent=1)
    print(status, out)
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
