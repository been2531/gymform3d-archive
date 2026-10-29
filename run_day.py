#!/usr/bin/env python3
"""
One-command @gymform.3d day build (stage-based, hashed caches):

    python3 run_day.py --day 5                       # full run (cached stages are reused)
    python3 run_day.py --day 5 --only step2 --from composite   # re-run one step's composite only
    python3 run_day.py --day 5 --from qc                       # QC the existing master only
    python3 run_day.py --day 5 --check                         # dry run, renders nothing

Stages: kp (keypoints + temporal plates, cached) -> icons (4-slot preview) -> composite+overlays (per step,
piped into x264) -> tts (cached on its lines) -> encode (concat + mux, stream copy) -> qc.
Every stage output lives in _cache_gymform/ under a hash of its inputs (day config subset, gymform_spec.json,
module sources), so an edit or an auto-fix re-runs only the affected stage and step.
Auto-fix is targeted: a failing QC item is mapped to its module + steps, a render parameter is changed
(never a QC threshold) and only those steps re-render. Prints one tiny JSON line; details go to the log.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SPEC_P = os.path.join(ROOT, "gymform_spec.json")
CACHE = os.path.join(ROOT, "_cache_gymform")
STAGES = ["kp", "icons", "composite", "overlays", "tts", "encode", "qc"]
LOOK_MODULES = ["ctx", "bg", "mask", "equipment", "xray", "muscle", "layout", "tempo", "render"]
LEGACY = ["build_day5_lateral_raise_master.py", "build_4step_bone_muscle_master.py",
          "preview_v62_seamless_day2_master.py", "build_v30_pure_graphic_numbers.py"]
LOG = None


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg.rstrip() + "\n")


def h(*objs):
    m = hashlib.sha1()
    for o in objs:
        m.update(json.dumps(o, sort_keys=True, default=str).encode() if not isinstance(o, bytes) else o)
    return m.hexdigest()[:12]


def src_hash(files):
    return h(*[open(f, "rb").read() for f in files])


def merged_cfg(day_cfg, ovr):
    c = json.loads(json.dumps(day_cfg))
    for k, v in ovr.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


GLOBAL_KEYS = ["exercise_name", "equipment", "bone_steps", "muscle_steps", "studio_bg", "xray_style",
               "mask_feather", "equipment_hull", "kp_cache", "assets_dir", "qc_dir"]


def step_key(c, s, code_h, spec_h):
    ss = str(s)
    sub = {k: c.get(k) for k in GLOBAL_KEYS}
    sub.update({"clip": c["clips"][ss], "scale": c["step_scale"].get(ss), "dy": c["step_dy"].get(ss),
                "callout": c["callouts"][ss], "ty": c.get("callout_ty", {}).get(ss)})
    return h(sub, code_h, spec_h)


def icons_key(c, code_h, spec_h):
    return h({k: c.get(k) for k in GLOBAL_KEYS}, c["clips"], c["step_scale"], c["step_dy"], code_h, spec_h)


def run(cmd, env=None):
    with open(LOG, "a") as lf:
        return subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode


def env_threads(n="1"):
    return dict(os.environ, OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, MKL_NUM_THREADS=n)


def render_steps(steps, keys, cfg_p, ovr_p, icons_p, workers):
    """Composite + overlays stage: one subprocess per step (nice 10, 1 thread each), `workers` at a time."""
    pending, running = list(steps), []
    base = [sys.executable, "-m", "gymform.render", "--config", cfg_p, "--overrides", ovr_p, "--icons", icons_p]
    while pending or running:
        while pending and len(running) < workers:
            s = pending.pop(0)
            out = os.path.join(CACHE, "steps", f"s{s}_{keys[s]}.mp4")
            lf = open(LOG, "a")
            running.append((s, out, subprocess.Popen(["nice", "-n", "10"] + base + ["--render-step", str(s), "--out", out + ".part.mp4",
                                                     "--peak", out[:-4] + "_peak.png"], cwd=ROOT, stdout=lf,
                                                    stderr=subprocess.STDOUT, env=env_threads()), lf))
        time.sleep(0.5)
        for it in list(running):
            s, out, pr, lf = it
            if pr.poll() is not None:
                lf.close()
                running.remove(it)
                if pr.returncode != 0:
                    raise RuntimeError(f"step {s} render failed (see log)")
                os.replace(out + ".part.mp4", out)
                # QC sidecar meta of this render, stored next to the cached step
                c_meta = json.load(open(os.path.join(qc_dir_g, f"meta_step{s}.json")))
                json.dump(c_meta, open(out[:-4] + "_meta.json", "w"))


def qc(mp4, qc_dir):
    p = subprocess.run([sys.executable, "-m", "gymform.qc", mp4, "--qc-dir", qc_dir], cwd=ROOT, capture_output=True, text=True)
    log(p.stdout + p.stderr[-3000:])
    return re.findall(r"^\[(PASS|FAIL)\]\s+(.*)$", p.stdout, flags=re.M)


def autofix(rows, c, ovr):
    """Targeted: map each failing item to (module, steps) and change one render parameter. Returns steps to redo."""
    redo, notes = set(), []
    for st, item in rows:
        if st != "FAIL":
            continue
        steps = {int(x) for x in re.findall(r"step(\d)_", item)} or {1, 2, 3, 4}
        if item.startswith("(c)") and c.get("auto_scale", True):
            notes.append("needs-human (scale is auto-normalised from keypoints; fix ctx.normalize_scale, not a param): " + item[:120])
        elif item.startswith("(c)"):
            hs = {int(k): float(v) for k, v in re.findall(r"step(\d)=(\d+)px", item)}
            med = sorted(hs.values())[len(hs) // 2]
            sc = ovr.setdefault("step_scale", {})
            for s, hh in hs.items():
                if abs(hh / med - 1) > 0.02:
                    sc[str(s)] = round(float(c["step_scale"][str(s)]) * med / hh, 4)
                    redo.add(s)
            notes.append(f"layout: rescale {sorted(redo)}")
        elif item.startswith("(d)") and "zone border" in item:
            ovr["equipment"] = [dict(e, radius=int(e.get("radius", 0)) + 40) if "radius" in e else e for e in c["equipment"]]
            redo |= steps
            notes.append(f"equipment: zone +40px on {sorted(steps)}")
        elif item.startswith("(a)"):
            ovr["mask_feather"] = max(0.5, float(c.get("mask_feather", 0.85)) - 0.15)
            redo |= steps
            notes.append(f"mask: feather {ovr['mask_feather']}")
        else:
            notes.append("needs-human: " + item[:140])
    return redo, notes


qc_dir_g = None


def main():
    global LOG, qc_dir_g
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--only", default="", help="e.g. step2 or step1,step3")
    ap.add_argument("--from", dest="frm", default="kp", choices=STAGES)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(ROOT, "output", "logs"), exist_ok=True)
    os.makedirs(os.path.join(CACHE, "steps"), exist_ok=True)
    LOG = os.path.join(ROOT, "output", "logs", f"day{a.day:02d}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    cfg_p = os.path.join(ROOT, "configs", f"day{a.day:02d}.json")
    if not os.path.exists(cfg_p):
        print(json.dumps({"day": a.day, "ok": False, "blocker": f"missing {os.path.relpath(cfg_p, ROOT)}"}))
        sys.exit(2)
    print(f"RULES: {os.path.join(ROOT, 'RULES.md')} (locked; native 3D only, 0% 2D overpaint, root cause before render)", file=sys.stderr)
    spec = json.load(open(SPEC_P))
    day_cfg = json.load(open(cfg_p))
    # HARD GUARD (RULES.md #1): native 3D source only, no 2D overpaint stage -> fail before any rendering
    sys.path.insert(0, ROOT)
    from gymform.qc import native_source_ok
    ok_src, why = native_source_ok(day_cfg.get("source", {}))
    if not ok_src:
        print(json.dumps({"day": a.day, "ok": False, "stage": "guard", "blocker": "native_3d_source: " + why, "rules": "RULES.md"}))
        sys.exit(3)
    clips = [os.path.join(ROOT, day_cfg["assets_dir"], v["file"]) for v in day_cfg["clips"].values()]
    missing = [os.path.basename(p) for p in clips if not os.path.exists(p)]
    sys.path.insert(0, ROOT)
    from gymform import caption as _cap
    lint = {"caption": _cap.lint(_cap.build(day_cfg.get("caption", ""))), "spoken_question": _cap.lint_spoken(day_cfg, spec)}
    lint = {k: v for k, v in lint.items() if v}
    if a.check or missing:
        sq = spec["caption_rules"].get("spoken_question", {})
        plan = _cap.spoken_plan(day_cfg, spec) if sq and a.day >= int(sq.get("from_day", 6)) else None
        print(json.dumps({"day": a.day, "mode": "check", "ok": not missing and not lint, "missing_clips": missing,
                          "lint": lint, "spoken_question": plan}))
        sys.exit(0 if not missing and not lint else 2)
    if lint.get("spoken_question"):
        print(json.dumps({"day": a.day, "ok": False, "stage": "lint", "lint": lint}))
        sys.exit(2)
    A = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    ovr_p = os.path.join(CACHE, f"overrides_day{a.day:02d}.json")
    ovr = json.load(open(ovr_p)) if os.path.exists(ovr_p) else {}
    only = {int(x) for x in re.findall(r"\d", a.only)} or {1, 2, 3, 4}
    si = STAGES.index(a.frm)
    code_h = src_hash([os.path.join(ROOT, "gymform", f"{m}.py") for m in LOOK_MODULES] + [os.path.join(ROOT, f) for f in LEGACY])
    spec_h = src_hash([SPEC_P])
    T, status = {}, {"day": a.day}
    loops = int(spec["pipeline"].get("max_autofix_loops", 3))
    for attempt in range(1, loops + 2):
        json.dump(ovr, open(ovr_p, "w"))
        c = merged_cfg(day_cfg, ovr)
        qc_dir_g = A(c["qc_dir"])
        os.makedirs(qc_dir_g, exist_ok=True)
        keys = {s: step_key(c, s, code_h, spec_h) for s in (1, 2, 3, 4)}
        # kp stage: keypoints are cached per clip slice in kp_cache (and plates per step in _cache_gymform)
        if si <= 0 and a.frm == "kp" and a.only:
            import pickle
            kp = pickle.load(open(A(c["kp_cache"]), "rb"))
            for s in only:
                kp.pop(s, None)
            pickle.dump(kp, open(A(c["kp_cache"]), "wb"))
            for f in glob.glob(os.path.join(CACHE, f"plate_{a.day:02d}_s[{''.join(map(str, only))}]_*.png")):
                os.remove(f)
        # icons stage (4-slot preview). A single-step re-run keeps the last icons (other steps untouched).
        ik = icons_key(c, code_h, spec_h)
        icons_p = os.path.join(CACHE, f"icons_day{a.day:02d}_{ik}.pkl")
        if not os.path.exists(icons_p):
            last = sorted(glob.glob(os.path.join(CACHE, f"icons_day{a.day:02d}_*.pkl")), key=os.path.getmtime)
            if a.only and last:
                icons_p = last[-1]
            else:
                tt = time.time()
                if run(["nice", "-n", "10", sys.executable, "-m", "gymform.render", "--config", cfg_p, "--overrides", ovr_p,
                        "--icons-out", icons_p], env_threads("2")):
                    raise SystemExit(json.dumps({"day": a.day, "ok": False, "stage": "icons", "log": LOG}))
                T["icons"] = round(time.time() - tt, 1)
        # composite + overlays stage
        step_mp4 = {s: os.path.join(CACHE, "steps", f"s{s}_{keys[s]}.mp4") for s in (1, 2, 3, 4)}
        force = si <= STAGES.index("overlays")
        todo = [s for s in (1, 2, 3, 4) if not os.path.exists(step_mp4[s]) or (force and s in only and attempt == 1 and a.only)]
        if todo:
            tt = time.time()
            render_steps(todo, keys, cfg_p, ovr_p, icons_p, a.workers)
            T[f"composite_{attempt}"] = round(time.time() - tt, 1)
            T.setdefault("steps_rendered", []).append(todo)
        # restore the QC sidecar meta of cached steps, then merge
        for s in (1, 2, 3, 4):
            shutil.copy(step_mp4[s][:-4] + "_meta.json", os.path.join(qc_dir_g, f"meta_step{s}.json"))
        sys.path.insert(0, ROOT)
        merge = {}
        for s in (1, 2, 3, 4):
            m = json.load(open(os.path.join(qc_dir_g, f"meta_step{s}.json")))
            fr = merge.get("frames", {})
            fr.update(m["frames"])
            merge.update(m)
            merge["frames"] = fr
        merge["day"] = a.day
        json.dump(merge, open(os.path.join(qc_dir_g, "meta.json"), "w"), indent=1)
        # tts stage (cached on lines + voice spec)
        sq_from = int(spec["caption_rules"].get("spoken_question", {}).get("from_day", 6))
        tk = (h(c["tts_lines"], spec["format"]) if a.day < sq_from else      # Day <= 5: legacy key, no spoken question
              h(c["tts_lines"], spec["format"], _cap.spoken_question(c, spec), spec["caption_rules"]["spoken_question"],
                src_hash([os.path.join(ROOT, "gymform", "tts.py")])))
        wav = os.path.join(CACHE, f"tts_day{a.day:02d}_{tk}.wav")
        if not os.path.exists(wav) or (si == STAGES.index("tts") and attempt == 1):
            tt = time.time()
            if run([sys.executable, "-c", "import sys,json;sys.argv=['x'];from gymform import ctx;"
                    f"ctx.configure(json.load(open({cfg_p!r})));from gymform import tts;tts.synth({wav!r})"]):
                raise SystemExit(json.dumps({"day": a.day, "ok": False, "stage": "tts", "log": LOG}))
            T["tts"] = round(time.time() - tt, 1)
        if os.path.exists(wav + ".timing.json"):
            shutil.copy(wav + ".timing.json", os.path.join(qc_dir_g, "tts_timing.json"))
        elif os.path.exists(os.path.join(qc_dir_g, "tts_timing.json")):
            os.remove(os.path.join(qc_dir_g, "tts_timing.json"))
        # encode stage: concat cached step streams (stream copy) + AAC mux
        if si <= STAGES.index("encode") or attempt > 1:
            tt = time.time()
            lst = os.path.join(CACHE, f"steps_day{a.day:02d}.txt")
            open(lst, "w").write("".join(f"file '{step_mp4[s]}'\n" for s in (1, 2, 3, 4)))
            if run(["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-f", "concat", "-safe", "0",
                    "-i", lst, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", A(c["out_mp4"])]):
                raise SystemExit(json.dumps({"day": a.day, "ok": False, "stage": "encode", "log": LOG}))
            run([sys.executable, "-m", "gymform.render", "--config", cfg_p, "--overrides", ovr_p, "--sheet", A(c["out_sheet"])]
                + [step_mp4[s][:-4] + "_peak.png" for s in (1, 2, 3, 4)])
            T[f"encode_{attempt}"] = round(time.time() - tt, 1)
        # qc stage
        tt = time.time()
        rows = qc(A(c["out_mp4"]), qc_dir_g)
        T[f"qc_{attempt}"] = round(time.time() - tt, 1)
        fails = [r for r in rows if r[0] == "FAIL"]
        status.update({"qc": f"{len(rows) - len(fails)}/{len(rows)} PASS"})
        if rows and not fails:
            break
        if attempt > loops:
            break
        redo, notes = autofix(rows, c, ovr)
        status.setdefault("autofix", []).extend(notes)
        log("autofix: " + "; ".join(notes))
        if not redo:
            status["fails"] = [r[1][:120] for r in fails]
            break
    ok = bool(rows) and not [r for r in rows if r[0] == "FAIL"]
    if ok:
        sys.path.insert(0, ROOT)
        from gymform.caption import build as cap_build, lint as cap_lint
        cap_txt = cap_build(c.get("caption", ""))
        cap_err = cap_lint(cap_txt)
        if cap_err:
            status["caption_lint"] = cap_err
            log("caption lint: " + "; ".join(cap_err))
        open(A(f"output/day{a.day}_caption.txt"), "w").write(cap_txt)
    T["total"] = round(time.time() - t0, 1)
    # production time is measured every run (RULES.md: each reel faster than the last)
    hist_p = os.path.join(ROOT, "output", "qc", "production_times.jsonl")
    with open(hist_p, "a") as hf:
        hf.write(json.dumps({"day": a.day, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "ok": ok, "t": T}) + "\n")
    status.update({"ok": ok, "mp4": os.path.relpath(A(c["out_mp4"]), ROOT), "t": T, "log": os.path.relpath(LOG, ROOT)})
    json.dump(status, open(os.path.join(ROOT, "output", "qc", f"day{a.day}_status.json"), "w"), indent=1)
    print(json.dumps(status, separators=(",", ":")))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
