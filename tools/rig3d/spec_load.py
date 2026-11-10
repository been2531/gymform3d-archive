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
            missing = overlay is None or metric not in (overlay or {}) or raw is None
        else:
            blob = (bench_frames or {}).get(fr or "") or {}
            # also accept the nested gates{} flags
            if metric in blob:
                raw = blob.get(metric)
                missing = raw is None
            elif metric in (blob.get("gates") or {}):
                raw = blob["gates"][metric]
                missing = raw is None
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
        rec = {"name": name, "value": raw if not isinstance(raw, float)
               else round(raw, 4), "limit": limit, "pass": bool(ok),
               "frame": fr, "missing": False}
        if item.get("label"):
            rec["label"] = item["label"]
        if item.get("autofix"):
            rec["autofix"] = item["autofix"]
        out.append(rec)
    return out


_CAM_KEYS = (
    "mode", "yaw_back_deg", "elev_deg", "dist_m", "look",
    "lens_mm", "shift_x", "shift_y", "up", "search", "height_m", "fill_w",
)


def apply_spec(cfg, spec=None):
    """Overlay exercises/<id>.spec.yaml onto a runtime cfg. No slug branches."""
    if not cfg:
        return cfg
    spec = spec if spec is not None else cfg.get("spec")
    if not spec:
        return cfg
    cam = spec.get("camera") or {}
    if cam:
        rc = cfg.setdefault("render", {}).setdefault("camera", {})
        for k in _CAM_KEYS:
            if k in cam:
                rc[k] = cam[k]
        if cam.get("proof"):
            cfg.setdefault("render", {})["camera_proof"] = dict(cam["proof"])
    look = spec.get("look") or {}
    mus = look.get("muscles")
    if isinstance(mus, dict):
        mh = cfg.setdefault("grade", {}).setdefault("muscle_hues", {})
        for name, vals in mus.items():
            if isinstance(vals, dict):
                cur = dict(mh.get(name) or {})
                cur.update(vals)
                mh[name] = cur
    if look.get("clay_clip_max") is not None:
        cfg.setdefault("qc", {})["clay_clip_max"] = look["clay_clip_max"]
        cfg.setdefault("grade", {})["clay_clip_max"] = look["clay_clip_max"]
    pose = spec.get("pose") or {}
    mo = cfg.setdefault("motion", {})
    if "ref_fit" in pose:
        mo["ref_fit"] = bool(pose["ref_fit"])
    elif "ref_fit" not in mo:
        mo["ref_fit"] = False
    mo["spec_pose"] = pose
    ov = (spec.get("qc") or {}).get("overlay") or {}
    if ov:
        br = cfg.setdefault("qc", {}).setdefault("bench_row", {})
        if "sil_iou_min" in ov:
            br["sil_iou_min"] = ov["sil_iou_min"]
        if "joint_mae_max" in ov:
            br["joint_mae_max"] = ov["joint_mae_max"]
    rend = spec.get("render") or {}
    if rend:
        rcfg = cfg.setdefault("render", {})
        for k, v in rend.items():
            if k in ("camera", "look"):
                continue
            rcfg[k] = v
    cfg["spec"] = spec
    return cfg
