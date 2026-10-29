#!/usr/bin/env python3
"""
One-command @gymform.3d day build:  python3 run_day.py --day N

  1. loads exercise_curriculum.json (day entry) + gymform_spec.json (locked rules)
  2. runs the day's builder (registry below) with modest threads (nice 10, 2 threads)
  3. runs qc_gate_gymform.py on the master mp4
  4. on FAIL, applies spec-safe auto-fixes (JSON overrides read by the builder) and retries, up to
     spec.pipeline.max_autofix_loops (3) loops
  5. writes output/qc/<day>_status.json; the reel is marked READY only when every QC check PASSes

Auto-fixes never loosen QC thresholds; they only change render parameters:
  (c) height spread  -> rescale the tallest/shortest steps toward the median height
  (d) equipment      -> grow the equipment zone radius (+40px) so nothing is clipped by its zone
  (b1)/(b2)/(h) UI   -> raise fg_front / equipment-in-front strength
  (a) halo           -> tighter mask feather
Anything else (X-Ray look, skin lock, format) is reported as needs-human and the run stops as NOT READY.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(ROOT, "gymform_spec.json")
CURRIC = os.path.join(ROOT, "exercise_curriculum.json")

# day -> (builder script, default CFG output keys). A new day needs: source step clips + a DAY_CONFIGS entry.
BUILDERS = {
    5: "build_day5_gym_studio_v4.py",
}


def load_day(n):
    cur = json.load(open(CURRIC))
    days = cur.get("days", [])
    for d in days:
        if not isinstance(d, dict):
            continue
        if int(d.get("day", -1)) == n:
            return d
    raise SystemExit(f"day {n} not in exercise_curriculum.json")


def cfg_of(builder):
    """Read CFG (outputs, step_scale, equipment) from the builder without rendering."""
    code = (f"import sys,json; sys.argv=['x']; sys.path.insert(0,{ROOT!r}); import importlib; "
            f"m=importlib.import_module({os.path.splitext(builder)[0]!r}); "
            "c=dict(m.CFG); print('CFGJSON'+json.dumps(c, default=str))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=_env(None))
    m = re.search(r"CFGJSON(.*)", out.stdout)
    if not m:
        raise SystemExit("could not read builder CFG:\n" + out.stderr[-2000:])
    return json.loads(m.group(1))


def _env(ovr_path):
    e = dict(os.environ)
    e.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    if ovr_path:
        e["GYMFORM_OVERRIDES"] = ovr_path
    return e


def run_qc(mp4, qc_dir):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "qc_gate_gymform.py"), mp4, "--qc-dir", qc_dir],
                       cwd=ROOT, capture_output=True, text=True)
    rows = re.findall(r"^\[(PASS|FAIL)\]\s+(.*)$", p.stdout, flags=re.M)
    return p.stdout, rows


def autofix(rows, cfg, ovr, qc_text):
    """Return (changed, notes). Only render parameters are changed; QC thresholds never."""
    notes, changed = [], False
    fails = [r[1] for r in rows if r[0] == "FAIL"]
    for f in fails:
        if f.startswith("(c)"):
            hs = {int(k): float(v) for k, v in re.findall(r"step(\d)=(\d+)px", f)}
            if hs:
                med = sorted(hs.values())[len(hs) // 2]
                sc = ovr.setdefault("step_scale", dict(cfg.get("step_scale", {})))
                for s, h in hs.items():
                    k = str(s)
                    cur = float(sc.get(k, sc.get(s, 1.0)))
                    sc[k] = round(cur * med / h, 4)
                notes.append(f"(c) rescaled steps toward median height {med:.0f}px: {sc}")
                changed = True
        elif f.startswith("(d)"):
            eqs = ovr.setdefault("equipment", [dict(e) for e in cfg.get("equipment", [])])
            for e in eqs:
                if "radius" in e:
                    e["radius"] = int(e["radius"]) + 40
                if "pad" in e:
                    e["pad"] = int(e["pad"]) + 20
            notes.append("(d) grew equipment zones by 40px")
            changed = True
        elif f.startswith("(a)"):
            ovr["mask_feather"] = max(0.5, float(ovr.get("mask_feather", 0.85)) - 0.15)
            notes.append(f"(a) mask feather -> {ovr['mask_feather']}")
            changed = True
        else:
            notes.append(f"needs-human: {f[:160]}")
    return changed, notes


LOG = None


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg.rstrip() + "\n")


def summary(obj, code):
    print(json.dumps(obj, separators=(",", ":")))
    sys.exit(code)


def caption(day):
    t = day.get("title", "").title()
    return (f"{t}: wrong vs right in 20 seconds.\n"
            f"WRONG: {day.get('wrong_metric', '')}\nRIGHT: {day.get('right_metric', '')}\n"
            f"Save this for your next shoulder day.\n#gymform #formcheck #{day.get('slug', '').replace('_', '')} #anatomy #3d")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--skip-build", action="store_true", help="QC the existing master only")
    ap.add_argument("--check", action="store_true", help="dry run: validate spec, curriculum, builder, clips, tools; render nothing")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "output", "logs"), exist_ok=True)
    LOG = os.path.join(ROOT, "output", "logs", f"day{args.day}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    t0 = time.time()
    spec = json.load(open(SPEC))
    day = load_day(args.day)
    builder = BUILDERS.get(args.day)
    clips = sorted(glob.glob(os.path.join(ROOT, "output", "assets", f"day{args.day}_*step[1-4]_*_720p.mp4")))
    checks = {"spec": spec.get("spec_version"), "curriculum": day.get("title"), "builder": builder,
              "clips": len(clips), "ffmpeg": subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0,
              "qc_gate": os.path.exists(os.path.join(ROOT, "qc_gate_gymform.py")),
              "bg_plate": os.path.exists(os.path.join(ROOT, spec["background"]["plate"]))}
    ok_pre = bool(builder) and len(clips) >= 4 and checks["ffmpeg"] and checks["qc_gate"] and checks["bg_plate"]
    if args.check or not ok_pre:
        reason = None if ok_pre else ("no builder/DAY_CONFIGS entry" if not builder else
                                      "step clips missing: generate 4 clips (output/assets/dayN_*_step{1..4}_*_720p.mp4)" if len(clips) < 4 else "tooling missing")
        summary({"day": args.day, "mode": "check", "ok": ok_pre, "checks": checks, "blocker": reason, "log": LOG}, 0 if ok_pre else 2)
    loops = int(spec["pipeline"].get("max_autofix_loops", 3))
    ovr_path = os.path.join(ROOT, f"_tmp_overrides_day{args.day}.json")
    ovr = {}
    status = {"day": args.day, "title": day.get("title"), "spec": spec["spec_version"], "attempts": []}
    cfg = cfg_of(builder)
    status["checks"] = checks
    for attempt in range(1, loops + 2):
        json.dump(ovr, open(ovr_path, "w"), indent=1)
        if not (args.skip_build and attempt == 1):
            log(f"attempt {attempt}: building {builder}")
            tb = time.time()
            with open(LOG, "a") as lf:
                r = subprocess.run(["nice", "-n", str(spec["pipeline"].get("nice", 10)), sys.executable, builder],
                                   cwd=ROOT, env=_env(ovr_path), stdout=lf, stderr=lf)
            status.setdefault("timings_s", {})[f"build_{attempt}"] = round(time.time() - tb, 1)
            if r.returncode != 0:
                status["attempts"].append({"attempt": attempt, "build": "error"})
                break
        cfg = cfg_of(builder) if not ovr else {**cfg, **ovr}
        tq = time.time()
        text, rows = run_qc(cfg["out_mp4"], cfg["qc_dir"])
        status.setdefault("timings_s", {})[f"qc_{attempt}"] = round(time.time() - tq, 1)
        log(text)
        status["qc"] = {r[1].split(":")[0]: r[0] for r in rows}
        n_fail = sum(1 for r in rows if r[0] == "FAIL")
        status["attempts"].append({"attempt": attempt, "pass": len(rows) - n_fail, "fail": n_fail, "overrides": dict(ovr)})
        if rows and n_fail == 0:
            status.update({"ready": True, "mp4": cfg["out_mp4"], "sheet": cfg["out_sheet"], "cover": cfg["out_cover"]})
            break
        if attempt > loops:
            break
        changed, notes = autofix(rows, cfg, ovr, text)
        status["attempts"][-1]["autofix"] = notes
        log("auto-fix: " + "; ".join(notes))
        if not changed:
            break
    status.setdefault("ready", False)
    if status["ready"]:
        cap_p = os.path.join(ROOT, "output", f"day{args.day}_caption.txt")
        open(cap_p, "w").write(caption(day))
        status["caption"] = cap_p
    status["timings_s"] = {**status.get("timings_s", {}), "total": round(time.time() - t0, 1)}
    status["log"] = LOG
    os.makedirs(os.path.join(ROOT, "output", "qc"), exist_ok=True)
    sp = os.path.join(ROOT, "output", "qc", f"day{args.day}_status.json")
    json.dump(status, open(sp, "w"), indent=1)
    status["status_file"] = sp
    for a in status.get("attempts", []):
        a.pop("overrides", None)
    summary(status, 0 if status["ready"] else 1)


if __name__ == "__main__":
    main()
