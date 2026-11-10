"""Merge exercise YAML with shared + motion-module defaults (GlobeFlip-style playlist/modules).

Resolution order (later wins on conflict; dicts deep-merge, lists replace):
  modules/_shared.yaml  ->  modules/<module>.yaml  ->  exercises/<slug>.yaml

Playlist:
  playlist.yaml at repo root lists upcoming exercises. --next picks the first item whose
  status is ready or ready_proxy and whose exercise file exists.
"""
from __future__ import annotations
import copy, os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOD_DIR = os.path.join(ROOT, "modules")
EX_DIR = os.path.join(ROOT, "exercises")
PLAYLIST = os.path.join(ROOT, "playlist.yaml")

READY = {"ready", "ready_proxy"}


def deep_merge(base, over):
    out = copy.deepcopy(base) if base is not None else {}
    if not isinstance(over, dict):
        return copy.deepcopy(over)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return data


def load_playlist(path=None):
    path = path or PLAYLIST
    if not os.path.exists(path):
        raise SystemExit(f"playlist missing: {path}")
    data = _load_yaml(path)
    items = data.get("items") or []
    if not isinstance(items, list):
        raise SystemExit("playlist.yaml: 'items' must be a list")
    return data


def next_from_playlist(path=None, after=None):
    """Return the next playable slug (ready/ready_proxy, exercise yaml present).
    If after is set, skip until after that slug, then take the next ready one.
    """
    data = load_playlist(path)
    items = data.get("items") or []
    seen_after = after is None
    for it in items:
        slug = it.get("slug")
        if not slug:
            continue
        if not seen_after:
            if slug == after:
                seen_after = True
            continue
        status = (it.get("status") or "").lower()
        if status not in READY:
            continue
        ex = os.path.join(EX_DIR, f"{slug}.yaml")
        if not os.path.exists(ex):
            continue
        if it.get("motion_status") == "blocked_needs_motion":
            continue
        return slug, it
    raise SystemExit(
        "playlist: no ready exercise left"
        + (f" after {after}" if after else "")
        + ". Fill a free motion source or mark an item ready."
    )


def load_exercise(slug, root=None):
    root = root or ROOT
    ex_path = os.path.join(root, "exercises", f"{slug}.yaml")
    if not os.path.exists(ex_path):
        raise SystemExit(f"exercise config missing: {ex_path}")
    ex = _load_yaml(ex_path)
    shared = _load_yaml(os.path.join(root, "modules", "_shared.yaml"))
    module_name = ex.get("module")
    if not module_name and isinstance(ex.get("modules"), list) and ex["modules"]:
        module_name = ex["modules"][0]
    mod = {}
    if module_name:
        mod_path = os.path.join(root, "modules", f"{module_name}.yaml")
        if not os.path.exists(mod_path):
            raise SystemExit(f"module missing: {mod_path} (exercise {slug})")
        mod = _load_yaml(mod_path)
    # strip loader-only keys from layers before merge into runtime cfg
    def strip_meta(d):
        d = copy.deepcopy(d)
        for k in ("_doc", "primitive", "fault", "risks", "motion_hint", "status"):
            d.pop(k, None)
        return d
    cfg = deep_merge(strip_meta(shared), strip_meta(mod))
    cfg = deep_merge(cfg, strip_meta(ex))
    cfg["slug"] = cfg.get("slug") or slug
    cfg["_module"] = module_name
    cfg["_paths"] = {
        "exercise": ex_path,
        "module": os.path.join(root, "modules", f"{module_name}.yaml") if module_name else None,
        "shared": os.path.join(root, "modules", "_shared.yaml"),
    }
    spec_p = os.path.join(root, "exercises", f"{slug}.spec.yaml")
    if os.path.isfile(spec_p):
        cfg["_paths"]["spec"] = spec_p
        try:
            from spec_load import load_exercise_spec
            cfg["spec"] = load_exercise_spec(slug, root)
        except Exception:
            cfg["spec"] = None
    return cfg
