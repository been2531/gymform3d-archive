"""GOLD registry (t541u): approved / known-good references, loaded and printed before every run_day.py run."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_P = os.path.join(ROOT, "gold.json")


def load():
    if not os.path.exists(GOLD_P):
        return []
    items = sorted(json.load(open(GOLD_P)).get("items", []), key=lambda it: it.get("rank", 99))
    for it in items:
        ps = it["path"] if isinstance(it["path"], list) else [it["path"]]
        it["missing"] = [p for p in ps if not os.path.exists(os.path.join(ROOT, p))]
    return items


def summary(items):
    lines = [f"GOLD references ({len(items)}) - start from these (docs/GOLD.md, docs/BASELINE.md):"]
    for it in items:
        miss = f"  [MISSING: {', '.join(it['missing'])}]" if it.get("missing") else ""
        lines.append(f"  #{it.get('rank', '-')} {it['id']} ({it['kind']}, day {it.get('day', '-')}, {it.get('status', '')}): {it['why']}{miss}")
    return "\n".join(lines)
