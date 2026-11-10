"""Load exercises/<id>.spec.yaml — the machine-readable standards file QC reads.

The human template lives at docs/specs/_TEMPLATE.md. Missing specs are not an
error: older exercises stay on the values in exercises/<id>.yaml until filled.
"""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def spec_path(slug, root=None):
    root = root or ROOT
    return os.path.join(root, "exercises", f"{slug}.spec.yaml")


def load_exercise_spec(slug, root=None):
    """Return the parsed spec dict, or None if that exercise has no spec yet."""
    path = spec_path(slug, root)
    if not slug or not os.path.isfile(path):
        return None
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    data["_path"] = path
    return data


def find_run_json(run_dir, name):
    """bench_qc.json / overlay_qc.json may sit at the run root or under render/."""
    for p in (
        os.path.join(run_dir, name),
        os.path.join(run_dir, "render", name),
    ):
        if os.path.isfile(p):
            return p
    return None


def eval_check(item, bench_frames, overlay):
    """Evaluate one spec.qc.checks item. Returns a list of result dicts.

    Each result: {name, value, limit, pass, frame, missing}.
    """
    source = str(item.get("source") or "bench_qc").lower()
    metric = item.get("metric")
    frames = list(item.get("frames") or [None])
    out = []
    for fr in frames:
        name = item["name"] if fr is None else f"{item['name']}_{fr}"
        if source == "overlay":
            blob = overlay or {}
            raw = blob.get(metric) if metric else None
            missing = overlay is None or metric not in (overlay or {})
        else:
            blob = (bench_frames or {}).get(fr or "") or {}
            # also accept the nested gates{} flags
            if metric in blob:
                raw = blob.get(metric)
                missing = False
            elif metric in (blob.get("gates") or {}):
                raw = blob["gates"][metric]
                missing = False
            else:
                raw = None
                missing = True
        limit = {k: item[k] for k in ("min", "max", "equals") if k in item}
        if missing:
            out.append({"name": name, "value": "missing", "limit": limit,
                        "pass": False, "frame": fr, "missing": True})
            continue
        ok = True
        if "equals" in item:
            ok = ok and bool(raw) == bool(item["equals"]) if isinstance(item["equals"], bool) else raw == item["equals"]
        try:
            num = float(raw)
            if "min" in item:
                ok = ok and num >= float(item["min"])
            if "max" in item:
                ok = ok and num <= float(item["max"])
        except (TypeError, ValueError):
            if "min" in item or "max" in item:
                ok = False
        out.append({"name": name, "value": raw if not isinstance(raw, float)
                    else round(raw, 4), "limit": limit, "pass": bool(ok),
                    "frame": fr, "missing": False})
    return out
