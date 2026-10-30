"""Motion + tempo standards check (t549u/t550u): does a day's motion match what real lifters do?

Compares the cached 2D keypoints of each step (the same `_cache_dayN_*_kps.pkl` the pipeline already writes:
{step: [ {joint: (x, y)} per sampled frame ], ...}) and the pipeline tempo against configs/motion_standards.json
(every number there carries its source URL; `basis: heuristic` marks pipeline tolerances).

  right steps: every right_form item must pass (severity "warn" items only warn)
  wrong steps: at least ONE measurable mistake signature must fire, so the wrong demo shows a real, common mistake
  tempo:       eccentric / pause / concentric / rep seconds inside the exercise's sourced range

NOT wired into render.py or preflight yet (standalone, importable). CLI:
  python3 -m gymform.motion_check --slug romanian_deadlift --kp _cache_day4_rdl_native_kps.pkl --wrong-steps 1,2
  python3 -m gymform.motion_check --slug incline_dumbbell_press --kp _cache_day6_idp_720p_kps.pkl --wrong-steps 1,3 --input bench_incline_deg=30
Exit code 1 if any fail-severity item fails. Pure python (math only), no CPU-heavy work.
"""
import argparse
import json
import math
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDARDS_PATH = os.path.join(ROOT, "configs", "motion_standards.json")
STEP_SECONDS = 5.0  # LOCKED step length (gymform/tempo.py: 150 frames @ 30 fps)


def load_standards(path=None):
    with open(path or STANDARDS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------- geometry (2D image plane, y points down)
def _ang(a, b, c):
    """Angle at b (deg) between b->a and b->c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n < 1e-9:
        return float("nan")
    cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cosv))


def _mid(a, b):
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _from_vertical(top, bottom):
    """Angle (deg) of bottom->top from straight up."""
    dx, dy = top[0] - bottom[0], bottom[1] - top[1]
    return math.degrees(math.atan2(abs(dx), dy)) if (dx or dy) else float("nan")


NAN = float("nan")


def _j(kp, name, side):
    """Joint lookup; legacy Day 3/4 caches only have a mid-shoulder 'sh' and no arms -> fall back / NaN."""
    p = kp.get(f"{name}_{side}")
    if p is None and name == "sh":
        p = kp.get("sh")
    return p


def joint_angles(kp):
    """One frame of keypoints -> {metric: {"r": v, "l": v} or float}. Metric names match motion_standards.json.
    Missing joints give NaN (that item then reports n/a instead of failing)."""
    out = {k: {} for k in ("elbow_flexion", "knee_flexion", "hip_flexion", "arm_torso", "thigh_below_horizontal", "shin_from_vertical")}

    def put(metric, side, pts, fn):
        out[metric][side] = fn(*pts) if all(q is not None for q in pts) else NAN

    for s in ("r", "l"):
        sh, elb, wr, hip, kn, ank = (_j(kp, n, s) for n in ("sh", "elb", "wr", "hip", "kn", "ank"))
        put("elbow_flexion", s, (sh, elb, wr), lambda a, b, c: 180.0 - _ang(a, b, c))
        put("knee_flexion", s, (hip, kn, ank), lambda a, b, c: 180.0 - _ang(a, b, c))
        put("hip_flexion", s, (sh, hip, kn), lambda a, b, c: 180.0 - _ang(a, b, c))
        put("arm_torso", s, (hip, sh, elb), _ang)
        put("thigh_below_horizontal", s, (hip, kn),
            lambda h, k: math.degrees(math.atan2(h[1] - k[1], max(abs(k[0] - h[0]), 1e-9))))
        put("shin_from_vertical", s, (kn, ank), _from_vertical)
    shr, shl, hr, hl = _j(kp, "sh", "r"), _j(kp, "sh", "l"), _j(kp, "hip", "r"), _j(kp, "hip", "l")
    out["trunk_from_vertical"] = _from_vertical(_mid(shr, shl), _mid(hr, hl)) if None not in (shr, shl, hr, hl) else NAN
    kr, kl, ar, al = (kp.get(k) for k in ("kn_r", "kn_l", "ank_r", "ank_l"))
    if None in (kr, kl, ar, al) or abs(ar[0] - al[0]) < 1e-6:
        out["knee_ankle_width_ratio"] = NAN
    else:
        out["knee_ankle_width_ratio"] = abs(kr[0] - kl[0]) / abs(ar[0] - al[0])
    return out


def _agg(v, how="mean"):
    if not isinstance(v, dict):
        return v
    vals = [x for x in v.values() if x == x]
    if not vals:
        return float("nan")
    return {"max_side": max(vals), "min_side": min(vals)}.get(how, sum(vals) / len(vals))


def series(frames, metric, agg="mean"):
    return [_agg(joint_angles(f)[metric], agg) for f in frames]


def stat_value(frames, item):
    """Evaluate item["stat"] (start/end/min/max/range/value) of item["metric"] over a step's frames."""
    agg = item.get("agg", "mean")
    s = [x for x in series(frames, item["metric"], agg)]
    good = [x for x in s if x == x]
    if not good:
        return float("nan")
    st = item["stat"]
    if st == "start":
        return good[0]
    if st == "end":
        return good[-1]
    if st == "min":
        return min(good)
    if st == "max":
        return max(good)
    if st == "range":
        return max(good) - min(good)
    if st == "value":
        at = item["at"]
        ref = series(frames, at["metric"], at.get("agg", "mean"))
        idx = [i for i, x in enumerate(ref) if x == x and s[i] == s[i]]
        if not idx:
            return float("nan")
        pick = max if at.get("extreme", "max") == "max" else min
        i = pick(idx, key=lambda k: ref[k])
        return s[i]
    raise ValueError(f"unknown stat {st}")


# ---------------------------------------------------------------- checks
def _res(item, kind, value, expected, ok, extra=None):
    r = {"kind": kind, "id": item["id"], "desc": item["desc"], "value": None if value is None or value != value else round(value, 2),
         "expected": expected, "pass": ok, "severity": item.get("severity", "fail"),
         "basis": item.get("basis") or item.get("signature", {}).get("basis"), "sources": item.get("sources", [])}
    if extra:
        r.update(extra)
    return r


def _view_ok(item, view):
    need = item.get("view")
    return need is None or need == view


def check_right_step(std_ex, frames, view="3/4"):
    out = []
    for it in std_ex["right_form"]:
        if not _view_ok(it, view):
            out.append(_res(it, "right_form", None, f"needs {it['view']} view", None, {"severity": "info"}))
            continue
        v = stat_value(frames, it)
        if v != v:
            out.append(_res(it, "right_form", None, "joints not in keypoints", None, {"severity": "info"}))
            continue
        lo, hi = it["range"]
        tol = float(it.get("tol_deg", 0))
        ok = (v == v) and (lo - tol) <= v <= (hi + tol)
        out.append(_res(it, "right_form", v, f"{lo}..{hi} (+/-{tol})", ok))
    return out


def _fires(sig, value):
    op, th = sig["op"], sig["value"]
    return value == value and {">": value > th, "<": value < th, ">=": value >= th, "<=": value <= th}[op]


def check_wrong_step(std_ex, frames, inputs=None, view="3/4", mistake_id=None):
    """Wrong steps must show >=1 measurable real mistake (keypoint signature or config input). Returns (items, summary).
    If no keypoint signature is measurable (missing joints / wrong view) the summary is n/a, not a fail; if some are
    measurable but quiet while others are unmeasurable, it only WARNS (the intended mistake may be the unmeasurable one).
    mistake_id: the mistake the day's wrong steps are meant to show -> only that signature is judged."""
    out, fired, measurable, unmeasurable = [], [], 0, 0
    for it in std_ex["mistakes"]:
        if mistake_id and it["id"] != mistake_id:
            continue
        sig = it["signature"]
        if not sig.get("metric"):
            unmeasurable += 1
            out.append(_res(it, "mistake", None, "unmeasurable", None, {"severity": "info", "note": sig.get("note")}))
            continue
        if sig["metric"].startswith("input:"):
            key = sig["metric"].split(":", 1)[1]
            if not inputs or key not in inputs:
                out.append(_res(it, "mistake", None, f"input {key} not given", None, {"severity": "info"}))
                continue
            v = float(inputs[key])
        else:
            if not _view_ok(sig, view):
                unmeasurable += 1
                out.append(_res(it, "mistake", None, f"needs {sig['view']} view", None, {"severity": "info"}))
                continue
            v = stat_value(frames, sig)
            if v != v:
                unmeasurable += 1
                out.append(_res(it, "mistake", None, "joints not in keypoints", None, {"severity": "info"}))
                continue
            measurable += 1
        f = _fires(sig, v)
        if f:
            fired.append(it["id"])
        out.append(_res(it, "mistake", v, f"{sig['stat'] if 'stat' in sig else 'input'} {sig['op']} {sig['value']}", f, {"severity": "info"}))
    ok = True if fired else (False if measurable else None)
    summary = {"kind": "wrong_shows_mistake", "id": "wrong_shows_real_mistake", "desc": "wrong step shows >=1 sourced common mistake",
               "value": fired, "expected": ">=1 signature fires", "pass": ok, "severity": "info" if not (measurable or fired) else ("warn" if unmeasurable else "fail"),
               "basis": "source", "sources": []}
    return out, summary


def check_inputs(std_ex, inputs):
    out = []
    for key, spec in (std_ex.get("inputs") or {}).items():
        it = {"id": f"input_{key}", "desc": key, "sources": spec.get("sources", []), "basis": "source", "severity": "fail"}
        if not inputs or key not in inputs:
            out.append(_res(it, "input", None, f"{spec['range'][0]}..{spec['range'][1]}", None, {"severity": "info", "note": "not given"}))
            continue
        v = float(inputs[key])
        out.append(_res(it, "input", v, f"{spec['range'][0]}..{spec['range'][1]}", spec["range"][0] <= v <= spec["range"][1]))
    return out


def pipeline_tempo(style=None):
    """Rep tempo the renderer actually uses (gymform/tempo.py), as seconds per phase.
    day4_cosine: 1.5 cosine cycles per 5 s step -> rep period 3.33 s, eccentric = concentric = 1.67 s, no flat hold.
    day5_holds: 0.3 rest / 1.5 up / 1.3 hold / 1.5 down / 0.4 rest (one rep per 5 s step).
    TODO hook: measure from the rendered mp4 (tempo.measure_reversals) once motion_check is wired into preflight."""
    try:
        from gymform import tempo as T
        style = style or T.TEMPO.get("style", "day4_cosine")
        cycles = float(T.TEMPO.get("cycles_per_step", 1.5))
    except Exception:  # spec unreadable -> locked Day 4 defaults
        style, cycles = style or "day4_cosine", 1.5
    if style == "day5_holds":
        return {"style": style, "concentric_s": 1.5, "pause_s": 1.3, "eccentric_s": 1.5, "rep_s": 5.0}
    rep = STEP_SECONDS / cycles
    return {"style": style, "concentric_s": rep / 2, "pause_s": 0.0, "eccentric_s": rep / 2, "rep_s": rep}


def check_tempo(std_ex, measured=None, general=None):
    t = std_ex.get("tempo") or general
    m = measured or pipeline_tempo()
    out = []
    for key in ("eccentric_s", "pause_s", "concentric_s", "rep_s"):
        lo, hi = t[key]
        v = float(m[key])
        it = {"id": f"tempo_{key}", "desc": f"{key} vs real-lifter range", "sources": t["sources"], "basis": "source", "severity": "fail"}
        out.append(_res(it, "tempo", v, f"{lo}..{hi} s", lo - 1e-6 <= v <= hi + 1e-6))
    return out


def check_day(slug, steps, tempo_measured=None, inputs=None, standards=None, view="3/4", mistake_id=None):
    """view: camera view of the clips ('3/4' for every reel so far; 'top'/'front' unlock view-gated items).
    steps: {step_id: {"frames": [kp dict...], "is_wrong": bool}}. Returns {"ok", "slug", "items", "fails", "warns"}."""
    S = standards or load_standards()
    if slug not in S["exercises"]:
        raise KeyError(f"no motion standard for '{slug}' (have: {', '.join(sorted(S['exercises']))})")
    ex = S["exercises"][slug]
    items = []
    for sid in sorted(steps, key=str):
        st = steps[sid]
        if st["is_wrong"]:
            sub, summ = check_wrong_step(ex, st["frames"], inputs, view, mistake_id)
            for r in sub + [summ]:
                r["step"] = sid
            items += sub + [summ]
        else:
            for r in check_right_step(ex, st["frames"], view):
                r["step"] = sid
                items.append(r)
    items += [dict(r, step="all") for r in check_inputs(ex, inputs)]
    items += [dict(r, step="all") for r in check_tempo(ex, tempo_measured, S["_meta"].get("general_tempo"))]
    fails = [r for r in items if r["pass"] is False and r["severity"] == "fail"]
    warns = [r for r in items if r["pass"] is False and r["severity"] == "warn"]
    return {"ok": not fails, "slug": slug, "items": items, "fails": [f"{r['step']}:{r['id']}" for r in fails],
            "warns": [f"{r['step']}:{r['id']}" for r in warns]}


def steps_from_kp_cache(pkl_path, wrong_steps):
    """Pipeline kp cache {1..4: [frame kp dicts], ('sig', n): ...} -> check_day steps."""
    d = pickle.load(open(pkl_path, "rb"))
    return {k: {"frames": v, "is_wrong": k in wrong_steps} for k, v in d.items() if isinstance(k, int)}


def format_report(rep):
    lines = [f"motion_check {rep['slug']}: {'PASS' if rep['ok'] else 'FAIL'}"]
    for r in rep["items"]:
        mark = {True: "PASS", False: "FAIL" if r["severity"] == "fail" else ("WARN" if r["severity"] == "warn" else "no"), None: "--"}[r["pass"]]
        if r["kind"] == "mistake":
            mark = {True: "fires", False: "quiet", None: "n/a"}[r["pass"]]
        lines.append(f"  [{mark:5}] step {r['step']} {r['kind']}:{r['id']} value={r['value']} expected={r['expected']} ({r.get('basis')})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--kp", required=True, help="pipeline keypoint cache pkl")
    ap.add_argument("--wrong-steps", default="", help="comma list of wrong-form steps, e.g. 1,3")
    ap.add_argument("--input", action="append", default=[], help="key=value, e.g. bench_incline_deg=30")
    ap.add_argument("--tempo-style", default=None, help="day4_cosine (default from spec) or day5_holds")
    ap.add_argument("--view", default="3/4", help="camera view of the clips (3/4, top, front)")
    ap.add_argument("--mistake", default=None, help="mistake id the wrong steps are meant to show (judge only that one)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    wrong = {int(x) for x in a.wrong_steps.split(",") if x.strip()}
    inputs = dict(kv.split("=", 1) for kv in a.input)
    rep = check_day(a.slug, steps_from_kp_cache(a.kp, wrong), pipeline_tempo(a.tempo_style), inputs, view=a.view, mistake_id=a.mistake)
    print(json.dumps(rep) if a.json else format_report(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
