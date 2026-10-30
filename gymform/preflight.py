"""Preflight stamp (RULES.md): a full render is impossible without a PASSING preflight for the exact same
config + overrides + spec + code. run_day.py writes the stamp after the sample-frame preflight passes ALL QC
checks; gymform.render --render-step refuses to render without it. There is no bypass flag."""
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "_cache_gymform")


def stamp_key(cfg_path, ovr_path=None):
    """Hash of everything that changes pixels or QC: day config, overrides, spec, every gymform module."""
    hh = hashlib.sha1()
    paths = [cfg_path] + ([ovr_path] if ovr_path and os.path.exists(ovr_path) else []) + [os.path.join(ROOT, "gymform_spec.json")]
    paths += sorted(p for p in glob.glob(os.path.join(ROOT, "gymform", "*.py")) if not p.endswith("preflight.py"))
    for p in paths:
        hh.update(os.path.basename(p).encode())
        with open(p, "rb") as fh:
            hh.update(fh.read())
    return hh.hexdigest()[:16]


def stamp_path(cfg_path, ovr_path=None):
    day = os.path.splitext(os.path.basename(cfg_path))[0]
    return os.path.join(CACHE, "preflight", f"{day}_{stamp_key(cfg_path, ovr_path)}.json")


def write_stamp(cfg_path, ovr_path, summary):
    p = stamp_path(cfg_path, ovr_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(summary, fh, indent=1)
    return p


def read_stamp(cfg_path, ovr_path=None):
    p = stamp_path(cfg_path, ovr_path)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def require(cfg_path, ovr_path=None):
    """Raise SystemExit(5) unless a passing preflight stamp exists for this exact config/code hash."""
    st = read_stamp(cfg_path, ovr_path)
    if not st or not st.get("ok"):
        why = "no preflight stamp for this config/code hash" if not st else "preflight FAILED: " + "; ".join(st.get("fails", []))[:300]
        raise SystemExit(json.dumps({"ok": False, "stage": "render_guard", "blocker": why,
                                     "fix": "run: python3 run_day.py --day N --preflight (RULES.md)"}))
    return st
