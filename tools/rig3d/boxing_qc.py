#!/usr/bin/env python3
"""Reusable boxing-form + render QC metrics.

No Blender dependency. A later self-improving loop can:
  - call evaluate_render() / punch_path_metrics()
  - log the JSON
  - ratchet thresholds from passing averages
  - retry with parameter tweaks

This module only computes scores and writes JSON. It does not run the loop.

Typical use:
  qc = punch_path_metrics(frames, punch, side="R")
  report = evaluate_render(png_path, path_qc=qc, punch_type="cross")
  write_qc_json(out_json, report)
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable

import numpy as np

VERSION = 1

# Default pass/fail gates. A later loop may overwrite these via
# tools/rig3d/boxing_qc_thresholds.json (never required to exist).
DEFAULT_THRESHOLDS = {
    "jab": {
        "fist_max_dev_m": 0.06,
        "fist_arc_height_m": 0.05,
        "elbow_above_shoulder_max_m": 0.03,
        "retract_max_dev_m": 0.08,
        "start_to_chin_m": 0.34,
        "foot_slide_m": 0.12,
        "root_drift_m": 0.22,
        "frame_fill_min": 0.62,
        "frame_fill_max": 0.82,
        "luminance_p05_min": 40.0,
        "luminance_median_min": 120.0,
        "luminance_max_max": 245.0,
    },
    "cross": {
        "fist_max_dev_m": 0.06,
        "fist_arc_height_m": 0.05,
        "elbow_above_shoulder_max_m": 0.03,
        "retract_max_dev_m": 0.08,
        "start_to_chin_m": 0.36,
        "foot_slide_m": 0.16,
        "root_drift_m": 0.28,
        "frame_fill_min": 0.62,
        "frame_fill_max": 0.82,
        "luminance_p05_min": 40.0,
        "luminance_median_min": 120.0,
        "luminance_max_max": 245.0,
    },
    "hook": {
        "fist_max_dev_m": 0.22,
        "fist_arc_height_m": 0.14,
        "elbow_above_shoulder_max_m": 0.10,
        "retract_max_dev_m": 0.16,
        "start_to_chin_m": 0.40,
        "foot_slide_m": 0.18,
        "root_drift_m": 0.30,
        "frame_fill_min": 0.62,
        "frame_fill_max": 0.82,
        "luminance_p05_min": 40.0,
        "luminance_median_min": 120.0,
        "luminance_max_max": 245.0,
    },
    "uppercut": {
        "fist_max_dev_m": 0.10,
        "fist_arc_height_m": 0.12,
        "elbow_above_shoulder_max_m": 0.06,
        "retract_max_dev_m": 0.12,
        "start_to_chin_m": 0.55,
        "foot_slide_m": 0.16,
        "root_drift_m": 0.28,
        "frame_fill_min": 0.62,
        "frame_fill_max": 0.82,
        "luminance_p05_min": 40.0,
        "luminance_median_min": 120.0,
        "luminance_max_max": 245.0,
    },
}

# Expected glow hues (approx) matching eval_xbot_boxing.COL
_GLOW_RGB = {
    "front_delt": (0.55, 0.16, 1.00),
    "triceps": (1.00, 0.72, 0.04),
    "biceps": (1.00, 0.88, 0.18),
    "chest": (1.00, 0.22, 0.52),
    "obliques": (1.00, 0.34, 0.08),
    "calf": (0.12, 0.82, 0.88),
    "hip": (0.22, 0.95, 0.38),
    "quads": (0.48, 1.00, 0.16),
    "glutes": (1.00, 0.32, 0.58),
}


def load_thresholds(path: str | None = None) -> dict:
    """Return per-type thresholds, overlaying an optional JSON file."""
    here = os.path.dirname(os.path.abspath(__file__))
    default_path = path or os.path.join(here, "boxing_qc_thresholds.json")
    out = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    if default_path and os.path.isfile(default_path):
        with open(default_path) as fh:
            extra = json.load(fh)
        for ptype, vals in extra.items():
            out.setdefault(ptype, {}).update(vals)
    return out


def _v(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).reshape(-1)[:3]


def point_to_segment_dist(p, a, b) -> float:
    p, a, b = _v(p), _v(a), _v(b)
    ab = b - a
    l2 = float(np.dot(ab, ab))
    if l2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def project_t(p, a, b) -> float:
    p, a, b = _v(p), _v(a), _v(b)
    ab = b - a
    l2 = float(np.dot(ab, ab))
    if l2 < 1e-12:
        return 0.0
    return float(np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0))


def point_to_polyline_dist(p, poly: Iterable) -> float:
    pts = [_v(q) for q in poly]
    if not pts:
        return 0.0
    if len(pts) == 1:
        return float(np.linalg.norm(_v(p) - pts[0]))
    return min(point_to_segment_dist(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def lo_better(x, good, bad) -> float:
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return 1.0 - (x - good) / (bad - good)


def _get(frame: dict, key: str):
    return frame.get(key)


def _window(frames: list[dict], f0: int, f1: int) -> list[dict]:
    return [r for r in frames if f0 <= int(r.get("f", -1)) <= f1]


def punch_path_metrics(frames: list[dict], punch: dict, side: str | None = None) -> dict:
    """Path metrics over guard → extension → return.

    ``frames`` are per-frame dicts from analyze_rich / similar:
    hand_l/r, arm_l/r (shoulder), fore_l/r (elbow), head, hips, foot_l/r as [x,y,z].
    """
    side = side or punch.get("side") or "R"
    sl = "l" if side == "L" else "r"
    hkey = f"hand_{sl}"
    skey = f"arm_{sl}"
    ekey = f"fore_{sl}"
    f_g = int(punch["guard"])
    f_full = int(punch["full"])
    f_ret = int(punch.get("ret") or punch["full"])
    if f_ret < f_full:
        f_ret = f_full
    out_b = _window(frames, f_g, f_full)
    ret_b = _window(frames, f_full, f_ret)
    whole = _window(frames, f_g, f_ret)

    def series(rows, key):
        pts = []
        for r in rows:
            p = r.get(key)
            if p:
                pts.append(_v(p))
        return pts

    fists_out = series(out_b, hkey)
    fists_ret = series(ret_b, hkey)
    fists_all = series(whole, hkey)
    elbows = series(whole, ekey)
    shoulders = series(whole, skey)

    start = fists_out[0] if fists_out else np.zeros(3)
    end = fists_out[-1] if fists_out else start.copy()
    # chin proxy: head position at guard
    head_g = None
    for r in out_b:
        if r.get("head"):
            head_g = _v(r["head"])
            break
    chin = head_g if head_g is not None else start

    devs = [point_to_segment_dist(p, start, end) for p in fists_out] or [0.0]
    fist_max_dev = max(devs)
    # height above the start→end line (positive = looping overhead)
    arcs = []
    for p in fists_out:
        t = project_t(p, start, end)
        line_z = (1.0 - t) * start[2] + t * end[2]
        arcs.append(float(p[2] - line_z))
    fist_arc_height = max(arcs) if arcs else 0.0
    fist_arc_depth = min(arcs) if arcs else 0.0

    # chin → target (extension fist) line — the intended straight punch
    chin_devs = [point_to_segment_dist(p, chin, end) for p in fists_out] or [0.0]
    chin_line_dev = max(chin_devs)

    retract_dev = 0.0
    if fists_out and fists_ret:
        retract_dev = max(point_to_polyline_dist(p, fists_out) for p in fists_ret)

    start_to_chin = float(np.linalg.norm(start - chin)) if head_g is not None else None

    # How much outbound motion is along the chord vs sideways/up
    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    forward_frac = 0.0
    if len(fists_out) >= 2 and chord_len > 1e-6:
        step = np.diff(np.stack(fists_out), axis=0)
        along = np.clip((step @ (chord / chord_len)), 0.0, None).sum()
        total = float(np.linalg.norm(step, axis=1).sum())
        forward_frac = float(along / total) if total > 1e-8 else 0.0

    elbow_above = []
    for e, s in zip(elbows, shoulders):
        elbow_above.append(float(e[2] - s[2]))
    # align by frame if lengths differ
    if elbows and shoulders and len(elbows) != len(shoulders):
        n = min(len(elbows), len(shoulders))
        elbow_above = [float(elbows[i][2] - shoulders[i][2]) for i in range(n)]
    elb_max = max(elbow_above) if elbow_above else 0.0
    elb_above_n = int(sum(1 for v in elbow_above if v > 0.02))
    # at impact (last outbound frame)
    impact_elb = 0.0
    if out_b:
        last = out_b[-1]
        if last.get(ekey) and last.get(skey):
            impact_elb = float(_v(last[ekey])[2] - _v(last[skey])[2])

    # flags
    # Elbow above the shoulder is the swimming-stroke / overhand tell even
    # when the fist ends at chin height (the old end-frame metrics missed this).
    is_overhand = (elb_max > 0.06) or (fist_arc_height > 0.14)
    is_looping = fist_max_dev > 0.10 or fist_arc_height > 0.10
    elbow_rises = elb_max > 0.04
    starts_at_chin = start_to_chin is not None and start_to_chin <= 0.36
    retracts_ok = retract_dev <= 0.08

    traj = []
    for r in whole:
        rec = {"f": int(r["f"])}
        if r.get(hkey):
            rec["fist"] = [float(x) for x in _v(r[hkey])]
        if r.get(ekey):
            rec["elbow"] = [float(x) for x in _v(r[ekey])]
        if r.get(skey):
            rec["shoulder"] = [float(x) for x in _v(r[skey])]
        if r.get("head"):
            rec["head"] = [float(x) for x in _v(r["head"])]
        traj.append(rec)

    return {
        "version": VERSION,
        "side": side,
        "guard": f_g,
        "full": f_full,
        "ret": f_ret,
        "fist_max_dev_m": round(fist_max_dev, 4),
        "fist_arc_height_m": round(fist_arc_height, 4),
        "fist_arc_depth_m": round(fist_arc_depth, 4),
        "chin_line_dev_m": round(chin_line_dev, 4),
        "retract_max_dev_m": round(retract_dev, 4),
        "start_to_chin_m": None if start_to_chin is None else round(start_to_chin, 4),
        "chord_len_m": round(chord_len, 4),
        "forward_frac": round(forward_frac, 3),
        "elbow_above_shoulder_max_m": round(elb_max, 4),
        "elbow_above_frames": elb_above_n,
        "elbow_frames": len(elbow_above),
        "elbow_at_impact_vs_shoulder_m": round(impact_elb, 4),
        "is_overhand": bool(is_overhand),
        "is_looping": bool(is_looping),
        "elbow_rises_above_shoulder": bool(elbow_rises),
        "fist_starts_at_chin": bool(starts_at_chin),
        "retracts_along_line": bool(retracts_ok),
        "traj": traj,
    }


def path_score_parts(qc: dict, punch_type: str) -> dict[str, float]:
    """Score components (same units as pick_xbot_punches_v3 parts) from path QC."""
    parts: dict[str, float] = {}
    dev = float(qc.get("fist_max_dev_m") or 0.0)
    arc = float(qc.get("fist_arc_height_m") or 0.0)
    elb = float(qc.get("elbow_above_shoulder_max_m") or 0.0)
    ret = float(qc.get("retract_max_dev_m") or 0.0)
    chin = qc.get("start_to_chin_m")
    if punch_type in ("jab", "cross"):
        parts["path_straight"] = 16 * lo_better(dev, 0.04, 0.18)
        parts["no_arc"] = 14 * lo_better(max(0.0, arc), 0.03, 0.16)
        parts["elbow_below"] = 16 * lo_better(max(0.0, elb), 0.02, 0.12)
        parts["retract_line"] = 8 * lo_better(ret, 0.05, 0.20)
        if chin is not None:
            parts["start_chin"] = 6 * lo_better(float(chin), 0.28, 0.55)
        if elb > 0.08 or arc > 0.14 or (qc.get("is_overhand") and punch_type == "cross"):
            parts["overhand_reject"] = -45.0
        elif qc.get("is_looping"):
            parts["loop_pen"] = -20.0
    elif punch_type == "hook":
        # hooks are curved; only reject an overhead windup
        parts["elbow_not_overhead"] = 10 * lo_better(max(0.0, elb), 0.08, 0.22)
        parts["impact_elbow"] = 8 * math.exp(
            -0.5 * ((float(qc.get("elbow_at_impact_vs_shoulder_m") or 0.0)) / 0.08) ** 2
        )
        if arc > 0.22 or elb > 0.18:
            parts["overhead_hook_pen"] = -15.0
    elif punch_type == "uppercut":
        parts["side_dev"] = 10 * lo_better(dev, 0.06, 0.22)
        parts["elbow_not_overhead"] = 8 * lo_better(max(0.0, elb), 0.05, 0.18)
        if qc.get("is_overhand"):
            parts["overhand_reject"] = -20.0
    return parts


def _load_rgb255(png_path: str) -> np.ndarray:
    """HxWx3 float32 0..255. PIL if available, else Blender's image loader
    (Blender's bundled Python has no PIL)."""
    try:
        from PIL import Image

        return np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32)
    except ImportError:
        import bpy  # type: ignore

        img = bpy.data.images.load(png_path, check_existing=False)
        try:
            w, h = img.size
            px = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            arr = px.reshape(h, w, 4)[::-1, :, :3] * 255.0
            return np.ascontiguousarray(arr)
        finally:
            bpy.data.images.remove(img)


def image_metrics(png_path: str) -> dict:
    """Frame-fill and luminance stats for a rendered still."""
    im = _load_rgb255(png_path)
    y = 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]
    char = y > 32.0
    h, w = y.shape
    if not char.any():
        return {
            "frame_fill": 0.0,
            "luminance_p05": 0.0,
            "luminance_median": 0.0,
            "luminance_max": 0.0,
            "luminance_p99": 0.0,
            "rgb_max": int(im.max()),
            "n_ge250": int((im.max(axis=2) >= 250).sum()),
        }
    rows = np.where(char.any(axis=1))[0]
    fill = float((rows[-1] - rows[0] + 1) / h)
    yc = y[char]
    return {
        "frame_fill": round(fill, 3),
        "luminance_p05": round(float(np.percentile(yc, 5)), 1),
        "luminance_median": round(float(np.percentile(yc, 50)), 1),
        "luminance_max": round(float(yc.max()), 1),
        "luminance_p99": round(float(np.percentile(yc, 99)), 1),
        "rgb_max": int(im.max()),
        "n_ge250": int((im.max(axis=2) >= 250).sum()),
    }


def glow_metrics(png_path: str, expected_band=(0.20, 0.95)) -> dict:
    """Heuristic glow-region check from colour (no mask required).

    High-chroma pixels are treated as glow. region_score is the fraction of
    those pixels whose vertical position falls inside ``expected_band`` of
    the character bbox (0 = feet, 1 = head).
    """
    im = _load_rgb255(png_path) / 255.0
    mx = im.max(axis=2)
    mn = im.min(axis=2)
    chroma = mx - mn
    y = 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]
    char = y > (32.0 / 255.0)
    glow = (chroma > 0.18) & (mx > 0.25) & char
    n_glow = int(glow.sum())
    n_char = int(char.sum())
    if n_glow == 0 or not char.any():
        return {
            "glow_pixel_frac": 0.0,
            "glow_region_score": None,
            "n_glow": 0,
        }
    rows = np.where(char.any(axis=1))[0]
    lo, hi = int(rows[0]), int(rows[-1])
    span = max(1, hi - lo)
    yy = np.where(glow)[0]
    rel = (yy - lo) / span
    lo_b, hi_b = expected_band
    region = float(((rel >= lo_b) & (rel <= hi_b)).mean())
    return {
        "glow_pixel_frac": round(n_glow / max(n_char, 1), 4),
        "glow_region_score": round(region, 3),
        "n_glow": n_glow,
    }


def evaluate_render(
    png_path: str | None,
    path_qc: dict | None = None,
    punch_type: str = "jab",
    thresholds: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Combine path + image + glow into one JSON-serialisable score report."""
    th_all = thresholds or load_thresholds()
    th = th_all.get(punch_type) or th_all.get("jab") or {}
    img = image_metrics(png_path) if png_path and os.path.isfile(png_path) else {}
    glow = glow_metrics(png_path) if png_path and os.path.isfile(png_path) else {}
    pq = path_qc or {}

    def gate(key, value, mode="max"):
        lim = th.get(key)
        if lim is None or value is None:
            return None
        if mode == "max":
            return float(value) <= float(lim)
        if mode == "min":
            return float(value) >= float(lim)
        return None

    fill = img.get("frame_fill")
    passes = {
        "fist_path_straight": gate("fist_max_dev_m", pq.get("fist_max_dev_m")),
        "fist_no_overhead_arc": gate("fist_arc_height_m", pq.get("fist_arc_height_m")),
        "elbow_not_above_shoulder": gate(
            "elbow_above_shoulder_max_m", pq.get("elbow_above_shoulder_max_m")
        ),
        "retract_along_line": gate("retract_max_dev_m", pq.get("retract_max_dev_m")),
        "foot_slide": gate("foot_slide_m", extra.get("foot_slide_m") if extra else None),
        "root_drift": gate("root_drift_m", extra.get("root_drift_m") if extra else None),
        "frame_fill_lo": gate("frame_fill_min", fill, "min") if fill is not None else None,
        "frame_fill_hi": gate("frame_fill_max", fill, "max") if fill is not None else None,
        "luminance_p05": gate("luminance_p05_min", img.get("luminance_p05"), "min"),
        "luminance_median": gate("luminance_median_min", img.get("luminance_median"), "min"),
        "luminance_not_blown": gate("luminance_max_max", img.get("luminance_max")),
    }
    checked = [v for v in passes.values() if v is not None]
    report = {
        "version": VERSION,
        "punch_type": punch_type,
        "render": png_path,
        "path": {k: pq[k] for k in pq if k != "traj"} if pq else {},
        "motion": {
            "foot_slide_m": None if not extra else extra.get("foot_slide_m"),
            "root_drift_m": None if not extra else extra.get("root_drift_m"),
        },
        "image": img,
        "glow": glow,
        "thresholds_used": th,
        "pass": passes,
        "n_checked": len(checked),
        "n_passed": int(sum(1 for v in checked if v)),
        "all_pass": bool(checked) and all(checked),
        "is_overhand": bool(pq.get("is_overhand")),
        "is_looping": bool(pq.get("is_looping")),
    }
    if extra:
        report["extra"] = extra
    return report


def write_qc_json(path: str, report: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path


def draw_path_trace(traj: list[dict], size=(420, 280), title: str = "") -> Any:
    """Side (forward vs Z) + top (X vs Y) line drawing of a fist path. Returns a PIL image."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    im = Image.new("RGB", (w, h), (22, 22, 24))
    dr = ImageDraw.Draw(im)
    font = None
    fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.isfile(fp):
        font = ImageFont.truetype(fp, 12)
    fists = [p["fist"] for p in traj if p.get("fist")]
    if len(fists) < 2:
        dr.text((8, 8), "no traj", fill=(180, 180, 180), font=font)
        return im
    arr = np.asarray(fists, dtype=np.float64)
    # split halves: first half outbound-ish for colouring by index
    n = len(arr)
    mid = max(1, n // 2)

    def panel(xy, box, xlabel, ylabel, lbl):
        x0, y0, x1, y1 = box
        dr.rectangle((x0, y0, x1, y1), outline=(60, 60, 66))
        xs, ys = xy[:, 0], xy[:, 1]
        pad = 0.08
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        if abs(xmax - xmin) < 1e-3:
            xmin -= 0.05
            xmax += 0.05
        if abs(ymax - ymin) < 1e-3:
            ymin -= 0.05
            ymax += 0.05
        # equal-ish margin
        dx, dy = xmax - xmin, ymax - ymin
        xmin -= dx * pad
        xmax += dx * pad
        ymin -= dy * pad
        ymax += dy * pad

        def sx(x):
            return x0 + 8 + (x - xmin) / (xmax - xmin) * (x1 - x0 - 16)

        def sy(y):
            return y1 - 8 - (y - ymin) / (ymax - ymin) * (y1 - y0 - 16)

        pts = [(sx(xs[i]), sy(ys[i])) for i in range(n)]
        # outbound
        if mid >= 2:
            dr.line(pts[: mid + 1], fill=(80, 180, 255), width=2)
        # return
        if n - mid >= 2:
            dr.line(pts[mid:], fill=(255, 170, 70), width=2)
        dr.ellipse((pts[0][0] - 3, pts[0][1] - 3, pts[0][0] + 3, pts[0][1] + 3),
                   fill=(120, 220, 150))
        dr.ellipse((pts[mid][0] - 3, pts[mid][1] - 3, pts[mid][0] + 3, pts[mid][1] + 3),
                   fill=(230, 80, 80))
        # chord
        dr.line([pts[0], pts[mid] if mid < n else pts[-1]], fill=(140, 140, 148), width=1)
        dr.text((x0 + 6, y0 + 4), lbl, fill=(200, 200, 204), font=font)

    # side: forward (-Y) vs Z   /  top: X vs Y
    half = w // 2
    side_xy = np.stack([-arr[:, 1], arr[:, 2]], 1)
    top_xy = np.stack([arr[:, 0], arr[:, 1]], 1)
    if title:
        dr.text((8, 4), title, fill=(220, 220, 224), font=font)
        top_off = 20
    else:
        top_off = 4
    panel(side_xy, (6, top_off, half - 4, h - 6), "fwd", "z", "side  (−Y vs Z)")
    panel(top_xy, (half + 2, top_off, w - 6, h - 6), "x", "y", "top  (X vs Y)")
    return im


def compose_contact_strip(frame_paths: list[str], labels: list[str] | None = None,
                          thumb=(140, 248)) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    font = None
    fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.isfile(fp):
        font = ImageFont.truetype(fp, 11)
    n = len(frame_paths)
    if n == 0:
        return Image.new("RGB", (160, 80), (20, 20, 22))
    tw, th = thumb
    im = Image.new("RGB", (n * (tw + 4) + 4, th + 22), (16, 16, 18))
    dr = ImageDraw.Draw(im)
    for i, path in enumerate(frame_paths):
        x = 4 + i * (tw + 4)
        if path and os.path.isfile(path):
            fr = Image.open(path).convert("RGB")
            fr.thumbnail((tw, th))
            im.paste(fr, (x, 2))
        lab = labels[i] if labels and i < len(labels) else str(i)
        dr.text((x + 4, th + 4), lab, fill=(180, 180, 186), font=font)
    return im


# ---------------------------------------------------------------------------
# Ground-truth reference: height-normalised form features + distance scoring
# ---------------------------------------------------------------------------
# Body height H is estimated from rigid arm segments so it works on any rig
# (Mixamo, CMU, ACCAD) and in any pose:
#   (upper arm + forearm) ≈ 0.332 · H   (Drillis & Contini segment ratios)
ARM_TO_HEIGHT = 0.332
REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "boxing_reference.json")

# feature → (std floor, min |delta| before a fault can fire)
FEATURE_SPEC = {
    "fist_dev_H": (0.012, 0.025),
    "arc_H": (0.012, 0.025),
    "elbow_above_H": (0.015, 0.03),
    "retract_dev_H": (0.015, 0.03),
    "elbow_ext_deg": (6.0, 14.0),
    "fist_vs_head_H": (0.025, 0.05),
    "elbow_impact_H": (0.02, 0.04),
    "hip_rot_deg": (6.0, 12.0),
    "sh_rot_deg": (6.0, 12.0),
    "rear_foot_yaw_deg": (6.0, 12.0),
    "rear_guard_H": (0.02, 0.05),
    "foot_slide_H": (0.012, 0.03),
    "elbow_flare_H": (0.015, 0.035),
    "lean_delta_H": (0.015, 0.035),
    "head_tilt_fwd": (0.06, 0.12),
    "retract_drop_H": (0.015, 0.035),
    "ret_dist_H": (0.02, 0.05),
    "pre_dip_H": (0.012, 0.03),
    "guard_fist_vs_chin_H": (0.025, 0.05),
}

# fault name → list of (feature, bad direction). A fault fires if ANY of its
# features is beyond Z_FLAG std in the bad direction and beyond the abs gate.
FAULTS = {
    "elbow_flare": [("elbow_flare_H", +1)],
    "rear_guard_drop": [("rear_guard_H", +1)],
    "chin_up": [("head_tilt_fwd", -1)],
    "overextension": [("lean_delta_H", +1)],
    "no_retraction": [("retract_drop_H", +1), ("ret_dist_H", +1)],
    "looping_path": [("elbow_above_H", +1), ("arc_H", +1), ("fist_dev_H", +1)],
    "no_hip_rotation": [("hip_rot_deg", -1), ("rear_foot_yaw_deg", -1)],
    "elbow_low": [("elbow_impact_H", -1)],
    "arm_too_open": [("elbow_ext_deg", +1)],
    "telegraph": [("pre_dip_H", +1), ("guard_fist_vs_chin_H", -1)],
    "overswing": [("fist_vs_head_H", +1), ("elbow_ext_deg", +1)],
    "foot_slide": [("foot_slide_H", +1)],
}
TYPE_FAULTS = {
    "jab": ["elbow_flare", "rear_guard_drop", "chin_up", "overextension",
            "no_retraction", "looping_path", "foot_slide"],
    "cross": ["elbow_flare", "rear_guard_drop", "chin_up", "overextension",
              "no_retraction", "looping_path", "no_hip_rotation", "foot_slide"],
    "hook": ["elbow_low", "arm_too_open", "rear_guard_drop", "chin_up", "foot_slide"],
    "uppercut": ["telegraph", "overswing", "rear_guard_drop", "chin_up", "foot_slide"],
}
TYPE_FEATURES = {
    "jab": ["fist_dev_H", "arc_H", "elbow_above_H", "retract_dev_H", "elbow_ext_deg",
            "fist_vs_head_H", "rear_guard_H", "foot_slide_H", "elbow_flare_H",
            "lean_delta_H", "head_tilt_fwd", "retract_drop_H", "ret_dist_H", "sh_rot_deg"],
    "cross": ["fist_dev_H", "arc_H", "elbow_above_H", "retract_dev_H", "elbow_ext_deg",
              "fist_vs_head_H", "rear_guard_H", "foot_slide_H", "elbow_flare_H",
              "lean_delta_H", "head_tilt_fwd", "retract_drop_H", "ret_dist_H",
              "hip_rot_deg", "sh_rot_deg", "rear_foot_yaw_deg"],
    "hook": ["elbow_ext_deg", "elbow_impact_H", "fist_vs_head_H", "rear_guard_H",
             "foot_slide_H", "head_tilt_fwd", "hip_rot_deg", "sh_rot_deg"],
    "uppercut": ["pre_dip_H", "guard_fist_vs_chin_H", "fist_vs_head_H", "elbow_ext_deg",
                 "rear_guard_H", "foot_slide_H", "head_tilt_fwd", "hip_rot_deg"],
}
Z_FLAG = 2.5
CURVE_N = 21


def _angle(a, b, c) -> float:
    a, b, c = _v(a), _v(b), _v(c)
    u, w = a - b, c - b
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-9 or nw < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, w) / (nu * nw), -1, 1))))


def body_height_from_rows(rows: list[dict]) -> float:
    vals = []
    for r in rows:
        for s in ("l", "r"):
            a, e, h = r.get(f"arm_{s}"), r.get(f"fore_{s}"), r.get(f"hand_{s}")
            if a and e and h:
                vals.append(np.linalg.norm(_v(e) - _v(a)) + np.linalg.norm(_v(h) - _v(e)))
    if not vals:
        return 1.75
    return float(np.median(vals)) / ARM_TO_HEIGHT


def _yaw(p_l, p_r) -> float:
    return math.degrees(math.atan2(p_r[1] - p_l[1], p_r[0] - p_l[0]))


def _yaw_span(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    acc = [0.0]
    for i in range(1, len(vals)):
        d = vals[i] - vals[i - 1]
        while d > 180:
            d -= 360
        while d < -180:
            d += 360
        acc.append(acc[-1] + d)
    return max(acc) - min(acc)


def _facing(row: dict) -> np.ndarray:
    """Horizontal forward unit vector: hips → mid of both fists."""
    hl, hr, hp = row.get("hand_l"), row.get("hand_r"), row.get("hips")
    if hl and hr and hp:
        d = 0.5 * (_v(hl) + _v(hr)) - _v(hp)
        d[2] = 0.0
        n = np.linalg.norm(d)
        if n > 1e-6:
            return d / n
    return np.array([0.0, -1.0, 0.0])


def head_pitch_fwd(row: dict) -> float:
    """Forward tilt of the head axis measured against the shoulder line, so a
    whole-body yaw does not read as chin up/down. Lower = chin up."""
    ax = row.get("head_axis")
    al, ar = row.get("arm_l"), row.get("arm_r")
    if ax and al and ar:
        side = _v(ar) - _v(al)
        fwd = np.cross([0.0, 0.0, 1.0], side)
        fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        if n > 1e-6:
            return float(np.dot(_v(ax), fwd / n))
    return float(row.get("head_tilt_fwd") or 0.0)


def form_features(rows: list[dict], punch: dict, H: float | None = None,
                  lead: str | None = None) -> dict:
    """Height-normalised features for one punch, using the punch's own
    guard/full/ret frames (so edited clips are measured on the same window)."""
    side = punch.get("side") or "L"
    s = "l" if side == "L" else "r"
    o = "r" if s == "l" else "l"
    lead = lead or punch.get("lead_side") or ("L" if punch.get("is_lead_hand", True) == (side == "L") else "R")
    rear = "r" if lead == "L" else "l"
    by_f = {int(r["f"]): r for r in rows}
    g, f, rt = int(punch["guard"]), int(punch["full"]), int(punch.get("ret") or punch["full"])
    rt = max(rt, f)
    H = H or body_height_from_rows([by_f[k] for k in range(g, rt + 1) if k in by_f])
    G, F, R = by_f.get(g), by_f.get(f), by_f.get(rt)
    if not (G and F and R):
        return {"H": H, "error": "missing frames"}
    pq = punch_path_metrics(rows, punch, side)
    fwd = _facing(G)
    out = {"H": round(H, 4)}
    out["fist_dev_H"] = pq["fist_max_dev_m"] / H
    out["arc_H"] = max(0.0, pq["fist_arc_height_m"]) / H
    out["elbow_above_H"] = pq["elbow_above_shoulder_max_m"] / H
    out["retract_dev_H"] = pq["retract_max_dev_m"] / H
    out["elbow_impact_H"] = pq["elbow_at_impact_vs_shoulder_m"] / H
    out["elbow_ext_deg"] = _angle(F[f"arm_{s}"], F[f"fore_{s}"], F[f"hand_{s}"])
    out["fist_vs_head_H"] = (F[f"hand_{s}"][2] - F["head"][2]) / H
    win = [by_f[k] for k in range(g, f + 1) if k in by_f]
    allw = [by_f[k] for k in range(g, rt + 1) if k in by_f]
    hip_y = [_yaw(r["upleg_l"], r["upleg_r"]) for r in win if r.get("upleg_l") and r.get("upleg_r")]
    sh_y = [_yaw(r["arm_l"], r["arm_r"]) for r in win if r.get("arm_l") and r.get("arm_r")]
    out["hip_rot_deg"] = _yaw_span(hip_y)
    out["sh_rot_deg"] = _yaw_span(sh_y)
    fy = [r.get(f"foot_{rear}_yaw") for r in win if r.get(f"foot_{rear}_yaw") is not None]
    out["rear_foot_yaw_deg"] = _yaw_span(fy) if fy else 0.0
    out["rear_guard_H"] = float(np.linalg.norm(_v(F[f"hand_{o}"]) - _v(F["head"]))) / H
    slides = []
    for ft in ("foot_l", "foot_r"):
        pts = [(_v(r[ft])[0], _v(r[ft])[1]) for r in allw if r.get(ft)]
        if pts:
            xs, ys = zip(*pts)
            slides.append(math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    out["foot_slide_H"] = (max(slides) if slides else 0.0) / H
    # elbow flare: outward offset of the elbow from the shoulder→fist line
    flare = []
    for r in win:
        a, e, h = r.get(f"arm_{s}"), r.get(f"fore_{s}"), r.get(f"hand_{s}")
        ao = r.get(f"arm_{o}")
        if not (a and e and h and ao):
            continue
        a, e, h, ao = _v(a), _v(e), _v(h), _v(ao)
        outv = a - ao
        outv[2] = 0.0
        n = np.linalg.norm(outv)
        if n < 1e-6:
            continue
        outv /= n
        t = project_t(e, a, h)
        closest = a + t * (h - a)
        flare.append(float(np.dot(e - closest, outv)))
    out["elbow_flare_H"] = (max(flare) if flare else 0.0) / H

    def fwd_off(r):
        return float(np.dot(_v(r["head"]) - _v(r["hips"]), fwd))

    out["lean_delta_H"] = (fwd_off(F) - fwd_off(G)) / H
    out["head_tilt_fwd"] = head_pitch_fwd(F)
    out["retract_drop_H"] = (G[f"hand_{s}"][2] - R[f"hand_{s}"][2]) / H
    out["ret_dist_H"] = float(np.linalg.norm(_v(R[f"hand_{s}"]) - _v(R["head"]))) / H
    zs = [r[f"hand_{s}"][2] for r in win if r.get(f"hand_{s}")]
    out["pre_dip_H"] = (G[f"hand_{s}"][2] - min(zs)) / H if zs else 0.0
    out["guard_fist_vs_chin_H"] = (G[f"hand_{s}"][2] - G["head"][2]) / H
    # curves over normalised time (guard→full = 0..0.5, full→ret = 0.5..1)
    out["curves"] = _punch_curves(by_f, g, f, rt, s, o, fwd, H)
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 4)
    return out


def _punch_curves(by_f, g, f, rt, s, o, fwd, H):
    def sample(fa, fb, n):
        if fb <= fa:
            return [by_f.get(fa)] * n
        res = []
        for i in range(n):
            x = fa + (fb - fa) * i / (n - 1)
            k0 = int(math.floor(x))
            k1 = min(k0 + 1, fb)
            t = x - k0
            r0, r1 = by_f.get(k0), by_f.get(k1)
            res.append((r0, r1, t))
        return res

    half = CURVE_N // 2 + 1
    seq = sample(g, f, half) + sample(f, rt, half)[1:]
    fist, elb_ang, elb_up = [], [], []
    for item in seq:
        if not isinstance(item, tuple):
            item = (item, item, 0.0)
        r0, r1, t = item
        if not r0 or not r1:
            continue

        def lerp(key):
            return (1 - t) * _v(r0[key]) + t * _v(r1[key])

        a, e, h, ao = lerp(f"arm_{s}"), lerp(f"fore_{s}"), lerp(f"hand_{s}"), lerp(f"arm_{o}")
        lat = a - ao
        lat[2] = 0
        lat /= max(np.linalg.norm(lat), 1e-9)
        rel = h - a
        fist.append([float(np.dot(rel, fwd)) / H, float(np.dot(rel, lat)) / H, float(rel[2]) / H])
        elb_ang.append(_angle(a, e, h))
        elb_up.append(float(e[2] - a[2]) / H)
    return {"fist_rel_shoulder_H": [[round(c, 4) for c in p] for p in fist],
            "elbow_deg": [round(x, 2) for x in elb_ang],
            "elbow_above_H": [round(x, 4) for x in elb_up]}


def load_reference(path: str | None = None) -> dict | None:
    p = path or REFERENCE_PATH
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def reference_distance(features: dict, punch_type: str, ref: dict | None = None) -> dict:
    """Distance of one punch's features to the mocap reference.

    Returns RMS z-distance over the type's features (+ fist-curve term),
    closeness 0-100, per-feature z, and fault flags (named, from FAULTS)."""
    ref = ref if ref is not None else load_reference()
    if not ref or punch_type not in (ref.get("types") or {}):
        return {"available": False}
    tref = ref["types"][punch_type]
    stats = tref.get("features") or {}
    zs, deltas = {}, {}
    for k in TYPE_FEATURES.get(punch_type, []):
        st = stats.get(k)
        x = features.get(k)
        if not st or x is None:
            continue
        floor, _ = FEATURE_SPEC.get(k, (0.01, 0.02))
        sd = max(float(st["std"]), floor)
        zs[k] = (float(x) - float(st["mean"])) / sd
        deltas[k] = float(x) - float(st["mean"])
    curve_z = None
    cref = (tref.get("curves") or {}).get("fist_rel_shoulder_H")
    cc = (features.get("curves") or {}).get("fist_rel_shoulder_H")
    if cref and cc and len(cref["mean"]) == len(cc):
        m = np.asarray(cref["mean"])
        sd = np.maximum(np.asarray(cref["std"]), 0.02)
        diff = (np.asarray(cc) - m) / sd
        curve_z = float(np.sqrt(np.mean(diff ** 2)))
    terms = [min(abs(z), 4.0) ** 2 for z in zs.values()]
    if curve_z is not None:
        terms.append(min(curve_z, 4.0) ** 2)
    dist = float(math.sqrt(sum(terms) / len(terms))) if terms else None
    faults = {}
    for fname in TYPE_FAULTS.get(punch_type, []):
        hits = []
        for feat, sign in FAULTS[fname]:
            if feat not in zs:
                continue
            _, gate = FEATURE_SPEC.get(feat, (0.01, 0.02))
            if sign * zs[feat] > Z_FLAG and sign * deltas[feat] > gate:
                hits.append({"feature": feat, "z": round(zs[feat], 2),
                             "delta": round(deltas[feat], 4)})
        faults[fname] = hits
    flagged = sorted(k for k, v in faults.items() if v)
    worst = sorted(zs.items(), key=lambda kv: -abs(kv[1]))[:4]
    return {
        "available": True,
        "punch_type": punch_type,
        "n_ref": tref.get("n"),
        "distance": None if dist is None else round(dist, 3),
        "closeness": None if dist is None else round(100.0 * math.exp(-(dist ** 2) / 8.0), 1),
        "curve_z": None if curve_z is None else round(curve_z, 3),
        "z": {k: round(v, 2) for k, v in zs.items()},
        "worst": [[k, round(v, 2)] for k, v in worst],
        "faults": flagged,
        "fault_detail": {k: v for k, v in faults.items() if v},
    }


def reference_score_part(rd: dict) -> dict[str, float]:
    """Score contribution for pickers: up to +30 for closeness, -12 per fault."""
    if not rd or not rd.get("available") or rd.get("closeness") is None:
        return {}
    parts = {"ref_close": 30.0 * rd["closeness"] / 100.0}
    if rd.get("faults"):
        parts["ref_faults"] = -12.0 * len(rd["faults"])
    return parts
