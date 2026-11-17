#!/usr/bin/env python3
"""Series template + joint rig + one queue entry -> the topic dict make_reel.py renders.

Per episode only queue.json changes (action, number, source, caption); the look,
timeline, meter and sound come from series/load_meter.json, the framing from
rigs/<joint>.json, the pixels from the verified library."""
from __future__ import annotations

import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LIB = os.path.join("output", "cutaway", "library")


def load_queue():
    return json.load(open(os.path.join(HERE, "queue.json")))["topics"]


def verified(joint: str) -> bool:
    p = os.path.join(REPO, LIB, joint, "verified.json")
    return os.path.exists(p) and json.load(open(p)).get("status") == "verified"


def build(entry: dict, series: str = "load_meter") -> dict:
    S = json.load(open(os.path.join(HERE, "series", f"{series}.json")))
    rig = json.load(open(os.path.join(HERE, "rigs", f"{entry['joint']}.json")))
    T = S["times"]
    shots = []
    for role in ("hook", "s1", "s2", "counter", "bloom"):
        sh = copy.deepcopy(rig["shots"][role])
        sh["t"] = T[role]
        shots.append(sh)
    f = entry["fact"]
    facts = [{"id": "load", "value": entry["value"], "unit": "x body weight", "claim": f["claim"],
              "source": f["source"], "url": f["url"]},
             {"id": "year", "value": f["year"], "unit": "year", "claim": "publication year",
              "source": f["source"], "url": f["url"]}]
    for k, x in enumerate(entry.get("extra_facts", [])):
        facts.append({"id": f"extra{k}", "value": x["value"], "unit": x.get("unit", ""), "claim": x.get("claim", ""),
                      "source": f["source"], "url": f["url"]})
    c = S["counter"]
    return {
        "id": entry["id"], "series": series, "joint_name": entry["joint"],
        "assets": os.path.join(LIB, entry["joint"]),
        "duration": S["duration"], "hook": entry["hook"], "hook_out": S["hook_out"],
        "joint": rig["joint_xy"], "shots": shots,
        "counter": {"t0": T["counter"] + S["counter"].get("delay", 0.0), "t1": S["duration"] + 0.2, "to": entry["value"], "fmt": c["fmt"],
                    "count": c["count"], "caption": "BODY WEIGHT", "action": entry["action"],
                    "header": c["header"], "fact": "load"},
        "facts": facts, "caption": entry["caption"], "audio": S["audio"], "render": rig.get("render", {}),
    }


if __name__ == "__main__":
    import sys
    q = {e["id"]: e for e in load_queue()}
    print(json.dumps(build(q[sys.argv[1]]), indent=1, ensure_ascii=False))
