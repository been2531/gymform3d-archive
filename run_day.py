#!/usr/bin/env python3
"""
One-command @gymform.3d day build (stage-based, hashed caches):

    python3 run_day.py --day 5                       # full run (cached stages are reused)
    python3 run_day.py --day 5 --only step2 --from composite   # re-run one step's composite only
    python3 run_day.py --day 5 --from qc                       # QC the existing master only
    python3 run_day.py --day 5 --check                         # dry run, renders nothing
    python3 run_day.py --day 6 --preflight                     # QC sample frames only + ALL QC checks (~1 min)
A full run always preflights first (cached on the step keys) and refuses to render if the preflight fails.

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
LOOK_MODULES = ["ctx", "bg", "mask", "equipment", "xray", "muscle", "layout", "tempo", "render", "native", "plates"]
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
               "mask_feather", "equipment_hull", "kp_cache", "assets_dir", "qc_dir",
               "n_steps", "muscle_shader", "clean_plate", "callout_axis", "target_zoom", "static_body_lum", "metal_lum_max", "static_body_shin_r", "flat_look", "look", "no_sharpen", "no_slice_cap", "phase_continuous_cut", "native_temporal_w", "native_skin_p5", "native_glow_max", "shorts_evidence_lum", "anchor_lock_px", "native_dark_floor", "native_skin_p90", "native_chroma_k", "native_steel_dx", "native_steel_lift", "native_dark_warm_rgb", "native_sep_dx", "native_sep_dy", "plate_recolour", "native_keep_chroma", "native_glow_hue_strict", "native_keep_chroma_sat", "native_highlight_rolloff", "shorts_evidence_strict"]   # t567u look keys


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


def ensure_tts(day, c, spec, cfg_p, T, force=False):
    """TTS stage (cached on lines + voice spec). Returns the wav path."""
    from gymform import caption as _cap
    sq_from = int(spec["caption_rules"].get("spoken_question", {}).get("from_day", 6))
    tk = (h(c["tts_lines"], spec["format"]) if day < sq_from else      # Day <= 5: legacy key, no spoken question
          h(c["tts_lines"], spec["format"], _cap.spoken_question(c, spec), spec["caption_rules"]["spoken_question"],
            src_hash([os.path.join(ROOT, "gymform", "tts.py")])))
    wav = os.path.join(CACHE, f"tts_day{day:02d}_{tk}.wav")
    if not os.path.exists(wav) or force:
        tt = time.time()
        if run([sys.executable, "-c", "import sys,json;sys.argv=['x'];from gymform import ctx;"
                f"ctx.configure(json.load(open({cfg_p!r})));from gymform import tts;tts.synth({wav!r})"]):
            raise SystemExit(json.dumps({"day": day, "ok": False, "stage": "tts", "log": LOG}))
        T["tts"] = round(time.time() - tt, 1)
    return wav


def preflight(day, c, keys, cfg_p, ovr_p, icons_p, wav, workers, steps=(1, 2, 3, 4)):
    """PREFLIGHT (default gate, RULES.md): render only the QC sample frames of every step (7 per step, zoom
    included) through the real composite path and run ALL QC checks on them. On PASS a stamp keyed by the
    config/overrides/spec/code hash is written (gymform/preflight.py); gymform.render --render-step refuses to
    render without it. A passing stamp for the same hash is reused (nothing to re-check)."""
    from gymform import preflight as _pf
    st = _pf.read_stamp(cfg_p, ovr_p)
    pdir = os.path.join(ROOT, c["qc_dir"].rstrip("/") + "_preflight")
    if st and st.get("ok"):
        return True, [], 0.0, pdir, st
    tt = time.time()
    os.makedirs(pdir, exist_ok=True)
    for f in glob.glob(os.path.join(pdir, "*")):
        os.remove(f)
    base = [sys.executable, "-m", "gymform.render", "--config", cfg_p, "--overrides", ovr_p, "--icons", icons_p, "--qc-dir", pdir]
    pending, running = [x for x in (3, 4, 1, 2) if x in steps], []   # motion bases first: edit steps reuse their cached geometry
    while pending or running:
        while pending and len(running) < workers:
            busy = set(pending) | {r[0] for r in running}
            ready = [x for x in pending if {1: 3, 2: 4}.get(x) not in busy]   # edit step waits for its base
            if not ready:
                break
            sn = ready[0]; pending.remove(sn)
            lf = open(LOG, "a")
            running.append((sn, subprocess.Popen(["nice", "-n", "10"] + base + ["--preflight-step", str(sn)], cwd=ROOT,
                                                 stdout=lf, stderr=subprocess.STDOUT, env=env_threads()), lf))
        time.sleep(0.3)
        for it in list(running):
            sn, pr, lf = it
            if pr.poll() is not None:
                lf.close()
                running.remove(it)
                if pr.returncode != 0:
                    raise SystemExit(json.dumps({"day": day, "ok": False, "stage": "preflight", "step": sn, "log": LOG}))
    t_frames = round(time.time() - tt, 1)
    merge = {}
    for sn in steps:
        m = json.load(open(os.path.join(pdir, f"meta_step{sn}.json")))
        fr = merge.get("frames", {})
        fr.update(m["frames"])
        merge.update(m)
        merge["frames"] = fr
    merge["day"] = day
    json.dump(merge, open(os.path.join(pdir, "meta.json"), "w"), indent=1)
    if os.path.exists(wav + ".timing.json"):
        shutil.copy(wav + ".timing.json", os.path.join(pdir, "tts_timing.json"))
    rows = qc("-", pdir, preflight=True)
    fails = [r[1][:160] for r in rows if r[0] == "FAIL"]
    # one-time audited owner override (gymform/preflight.py): only the named checks, only this hash, once
    ovr_o = _pf.load_override(day, cfg_p, ovr_p)
    fails, waived = _pf.waive(fails, ovr_o)
    if waived:
        log(f"OWNER OVERRIDE {ovr_o['authorized_by']}: {ovr_o['reason']} | waived: " + "; ".join(w[:120] for w in waived))
    ok = bool(rows) and not fails
    checks = {}
    for r_ in rows:
        m_ = re.match(r"(\([a-z0-9]+\))", r_[1])
        k_ = m_.group(1) if m_ else r_[1][:24]
        checks[k_] = "FAIL" if (r_[0] == "FAIL" or checks.get(k_) == "FAIL") else "PASS"
    summary = {"ok": ok, "day": day, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": checks, "fails": fails,
               "override": ({"authorized_by": ovr_o["authorized_by"], "reason": ovr_o["reason"], "waived": waived}
                            if waived else None),
               "t": {"frames": t_frames, "qc": round(time.time() - tt - t_frames, 1)}, "qc_dir": os.path.relpath(pdir, ROOT)}
    _pf.write_stamp(cfg_p, ovr_p, summary)   # a FAILED stamp is stored too (render.py refuses it)
    return ok, rows, round(time.time() - tt, 1), pdir, summary


def render_steps(steps, keys, cfg_p, ovr_p, icons_p, workers):
    """Composite + overlays stage: one subprocess per step (nice 10, 1 thread each), `workers` at a time."""
    pending, running = sorted(steps, key=lambda s: (s not in (3, 4), s)), []   # bases first (geometry cache)
    base = [sys.executable, "-m", "gymform.render", "--config", cfg_p, "--overrides", ovr_p, "--icons", icons_p]
    while pending or running:
        while pending and len(running) < workers:
            busy = set(pending) | {r[0] for r in running}
            ready = [x for x in pending if {1: 3, 2: 4}.get(x) not in busy]   # edit step waits for its base
            if not ready:
                break
            s = ready[0]; pending.remove(s)
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


def qc(mp4, qc_dir, preflight=False):
    p = subprocess.run([sys.executable, "-m", "gymform.qc", mp4, "--qc-dir", qc_dir] + (["--preflight"] if preflight else []),
                       cwd=ROOT, capture_output=True, text=True)
    log(p.stdout + p.stderr[-3000:])
    rows = re.findall(r"^\[(PASS|FAIL)\]\s+(.*)$", p.stdout, flags=re.M)
    if p.returncode != 0 and "Traceback" in p.stderr:   # a crashed QC never counts as a pass
        rows.append(("FAIL", "QC crashed: " + p.stderr.strip().splitlines()[-1][:200]))
    return rows


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
    ap.add_argument("--config", default="", help="variant day config (default configs/dayNN.json), e.g. configs/day06_muscle2.json")
    ap.add_argument("--preflight", action="store_true", help="preflight only: QC sample frames + all QC checks (~1 min), no full render")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(ROOT, "output", "logs"), exist_ok=True)
    os.makedirs(os.path.join(CACHE, "steps"), exist_ok=True)
    LOG = os.path.join(ROOT, "output", "logs", f"day{a.day:02d}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    cfg_p = os.path.join(ROOT, a.config) if a.config else os.path.join(ROOT, "configs", f"day{a.day:02d}.json")
    if not os.path.exists(cfg_p):
        print(json.dumps({"day": a.day, "ok": False, "blocker": f"missing {os.path.relpath(cfg_p, ROOT)}"}))
        sys.exit(2)
    print(f"RULES: {os.path.join(ROOT, 'RULES.md')} (locked; native 3D only, 0% 2D overpaint, root cause before render)", file=sys.stderr)
    spec = json.load(open(SPEC_P))
    day_cfg = json.load(open(cfg_p))
    # GOLD registry (t541u): load + print the approved / known-good references before every run
    sys.path.insert(0, ROOT)
    from gymform import gold as _gold
    GOLD = _gold.load()
    print(_gold.summary(GOLD), file=sys.stderr, flush=True)
    log(_gold.summary(GOLD))
    # HARD GUARD (RULES.md #1): native 3D source only, no 2D overpaint stage -> fail before any rendering
    sys.path.insert(0, ROOT)
    from gymform.qc import native_source_ok
    from gymform.layout import callout_lint
    from gymform.ctx import expand_motions as _xm
    co_err = callout_lint(_xm(json.loads(json.dumps(day_cfg))))   # callouts are derived from motions.*.callout
    if co_err:   # numbers / labels / TTS must agree before anything is rendered
        print(json.dumps({"day": a.day, "ok": False, "stage": "guard", "blocker": "callout: " + "; ".join(co_err)}))
        sys.exit(3)
    from gymform.ctx import order_lint, pair_lint, expand_motions
    p_err = pair_lint(day_cfg, spec)
    if p_err:   # LOCKED (t521u): wrong steps 1/3 = one motion, right steps 2/4 = one motion
        print(json.dumps({"day": a.day, "ok": False, "stage": "guard", "blocker": "pair_motion: " + "; ".join(p_err), "rules": "RULES.md"}))
        sys.exit(3)
    from gymform.ctx import camera_plan_lint, step_plan, mistake_type
    mt_ = mistake_type(day_cfg)
    plan_ = ", ".join(f"{s_} {f_}/{c_}/{k_}" for s_, (f_, c_, k_) in step_plan(mt_, int(day_cfg.get("n_steps", 4))).items())
    print(f"STEP PLAN ({mt_} mistake day, t554u/t556u): {plan_}", file=sys.stderr, flush=True)
    log(f"step plan ({mt_}): {plan_}")
    cam_err = camera_plan_lint(day_cfg)
    if cam_err:   # t556u: 2 motions x fixed front/side cameras; Day 6 predates it (warning), Day 7+ must comply
        msg = "camera_plan: " + "; ".join(cam_err)
        log("WARNING " + msg)
        print("WARNING " + msg, file=sys.stderr, flush=True)
        if a.day >= 7:
            print(json.dumps({"day": a.day, "ok": False, "stage": "guard", "blocker": msg, "rules": "RULES.md"}))
            sys.exit(3)
    expand_motions(day_cfg)
    o_err = order_lint(day_cfg, json.load(open(os.path.join(ROOT, "gymform_spec.json"))))
    if o_err:   # LOCKED step order (Day 6+): 1 wrong bone, 2 right bone, 3 wrong muscle, 4 right muscle
        print(json.dumps({"day": a.day, "ok": False, "stage": "guard", "blocker": "step_order: " + "; ".join(o_err), "rules": "RULES.md"}))
        sys.exit(3)
    ok_src, why = native_source_ok(day_cfg.get("source", {}))
    if not ok_src:
        print(json.dumps({"day": a.day, "ok": False, "stage": "guard", "blocker": "native_3d_source: " + why, "rules": "RULES.md"}))
        sys.exit(3)
    clips = [os.path.join(ROOT, day_cfg["assets_dir"], v["file"]) for v in day_cfg["clips"].values()]
    # EDIT ACCEPTANCE GATE (t529u): a motion-preserving edit (edit_of) must pass gymform.xai_edit.acceptance vs its
    # base (opaque silver limbs of the same thickness, same motion, no leg glow, true tint) before any render
    from gymform import xai_edit as _xe
    acc_bad = []
    for m_, k_, src_, dst_, _, col_ in _xe.jobs_for(day_cfg):
        if os.path.exists(dst_) and os.path.exists(src_):
            ok_, am_ = _xe.acceptance(src_, dst_, col_)
            if not ok_:
                acc_bad.append(f"{os.path.basename(dst_)}: " + "; ".join(am_["fails"]))
    if acc_bad and not a.check:
        print(json.dumps({"day": a.day, "ok": False, "stage": "guard", "blocker": "edit_acceptance: " + " | ".join(acc_bad),
                          "fix": "python3 -m gymform.xai_edit --day N (new take needs owner approval)"}))
        sys.exit(3)
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
    # t598u/t599u: problem -> fix definitions are decided up front in docs/DEFECT_PLAYBOOK.md. Printed at every
    # run; from Day 7 the config must carry the playbook's current sha (proof it was read after its last change).
    import hashlib as _hl
    _pb = os.path.join(ROOT, "docs", "DEFECT_PLAYBOOK.md")
    _pb_sha = _hl.sha1(open(_pb, "rb").read()).hexdigest()[:8]
    print(f"DEFECT PLAYBOOK (docs/DEFECT_PLAYBOOK.md, sha {_pb_sha}): fix every on-screen defect by its row:", flush=True)
    for _ln in open(_pb, encoding="utf-8"):
        if _ln.startswith("| D"):
            _c = [x.strip() for x in _ln.strip().strip("|").split("|")]
            print(f"  {_c[0]} {_c[1]} -> fix: {_c[4][:110]} | check: {_c[6]}", flush=True)
    if a.day >= 7 and day_cfg.get("defect_playbook_sha") != _pb_sha:
        print(json.dumps({"day": a.day, "ok": False, "stage": "playbook",
                          "error": f"read docs/DEFECT_PLAYBOOK.md, then set \"defect_playbook_sha\": \"{_pb_sha}\" in the day config"}))
        sys.exit(2)
    A = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    VAR = os.path.splitext(os.path.basename(cfg_p))[0]   # day06 | day06_muscle2 (variants keep their own overrides)
    ovr_p = os.path.join(CACHE, f"overrides_{VAR}.json")
    STEPS = tuple(range(1, int(day_cfg.get("n_steps", 4)) + 1))
    ovr = json.load(open(ovr_p)) if os.path.exists(ovr_p) else {}
    only = {int(x) for x in re.findall(r"\d", a.only)} or set(STEPS)
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
        keys = {s: step_key(c, s, code_h, spec_h) for s in STEPS}
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
        step_mp4 = {s: os.path.join(CACHE, "steps", f"s{s}_{keys[s]}.mp4") for s in STEPS}
        force = si <= STAGES.index("overlays")
        todo = [s for s in STEPS if not os.path.exists(step_mp4[s]) or (force and s in only and attempt == 1 and a.only)]
        # PREFLIGHT GATE (default, RULES.md): no full render before the QC sample frames pass ALL QC checks
        if todo or a.preflight:
            wav_pf = ensure_tts(a.day, c, spec, cfg_p, T)
            pf_ok, pf_rows, pf_t, pf_dir, pf_sum = preflight(a.day, c, keys, cfg_p, ovr_p, icons_p, wav_pf, a.workers, STEPS)
            T[f"preflight_{attempt}"] = pf_t
            pf_fail = pf_sum.get("fails", [])
            n_ck = len(pf_sum.get("checks", {}))
            status["preflight"] = ("PASS" if pf_ok else "FAIL") + f" {n_ck - len(pf_fail)}/{n_ck}" + ("" if pf_rows else " (stamp reused)")
            if not pf_ok or a.preflight:
                T["total"] = round(time.time() - t0, 1)
                status.update({"ok": pf_ok, "stage": "preflight", "checks": pf_sum.get("checks", {}), "fails": pf_fail,
                               "qc_dir": os.path.relpath(pf_dir, ROOT), "t": T, "log": os.path.relpath(LOG, ROOT)})
                with open(os.path.join(ROOT, "output", "qc", "production_times.jsonl"), "a") as hf:
                    hf.write(json.dumps({"day": a.day, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "ok": pf_ok, "stage": "preflight", "t": T}) + "\n")
                print(json.dumps(status, separators=(",", ":")))
                sys.exit(0 if pf_ok else 4)
        if todo:
            tt = time.time()
            render_steps(todo, keys, cfg_p, ovr_p, icons_p, a.workers)
            T[f"composite_{attempt}"] = round(time.time() - tt, 1)
            T.setdefault("steps_rendered", []).append(todo)
        # restore the QC sidecar meta of cached steps, then merge
        for s in STEPS:
            shutil.copy(step_mp4[s][:-4] + "_meta.json", os.path.join(qc_dir_g, f"meta_step{s}.json"))
        sys.path.insert(0, ROOT)
        merge = {}
        for s in STEPS:
            m = json.load(open(os.path.join(qc_dir_g, f"meta_step{s}.json")))
            fr = merge.get("frames", {})
            fr.update(m["frames"])
            merge.update(m)
            merge["frames"] = fr
        merge["day"] = a.day
        json.dump(merge, open(os.path.join(qc_dir_g, "meta.json"), "w"), indent=1)
        # tts stage (cached on lines + voice spec)
        wav = ensure_tts(a.day, c, spec, cfg_p, T, force=(si == STAGES.index("tts") and attempt == 1))
        if os.path.exists(wav + ".timing.json"):
            shutil.copy(wav + ".timing.json", os.path.join(qc_dir_g, "tts_timing.json"))
        elif os.path.exists(os.path.join(qc_dir_g, "tts_timing.json")):
            os.remove(os.path.join(qc_dir_g, "tts_timing.json"))
        # encode stage: concat cached step streams (stream copy) + AAC mux
        if si <= STAGES.index("encode") or attempt > 1:
            tt = time.time()
            lst = os.path.join(CACHE, f"steps_day{a.day:02d}.txt")
            open(lst, "w").write("".join(f"file '{step_mp4[s]}'\n" for s in STEPS))
            if run(["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-f", "concat", "-safe", "0",
                    "-i", lst, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", A(c["out_mp4"])]):
                raise SystemExit(json.dumps({"day": a.day, "ok": False, "stage": "encode", "log": LOG}))
            run([sys.executable, "-m", "gymform.render", "--config", cfg_p, "--overrides", ovr_p, "--sheet", A(c["out_sheet"])]
                + [step_mp4[s][:-4] + "_peak.png" for s in STEPS])
            T[f"encode_{attempt}"] = round(time.time() - tt, 1)
        # qc stage
        tt = time.time()
        rows = qc(A(c["out_mp4"]), qc_dir_g)
        T[f"qc_{attempt}"] = round(time.time() - tt, 1)
        from gymform import preflight as _pf
        ovr_o = _pf.load_override(a.day, cfg_p, ovr_p)
        rest, waived = _pf.waive([r[1] for r in rows if r[0] == "FAIL"], ovr_o)
        fails = [r for r in rows if r[0] == "FAIL" and r[1] in rest]
        status.update({"qc": f"{len(rows) - len(fails) - len(waived)}/{len(rows)} PASS" + (f" + {len(waived)} WAIVED" if waived else "")})
        if waived:
            status["override"] = {"authorized_by": ovr_o["authorized_by"], "reason": ovr_o["reason"], "waived": [w[:160] for w in waived]}
            log(f"OWNER OVERRIDE {ovr_o['authorized_by']}: {ovr_o['reason']} | final QC waived: " + "; ".join(w[:120] for w in waived))
        if ovr_o:
            _pf.mark_used(ovr_o, f"final QC {status['qc']}; remaining fails: {len(fails)}")
            if fails:
                status["fails"] = [r[1][:120] for r in fails]
                break   # override run: exactly ONE render, no autofix re-render
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
    ok = bool(rows) and not fails
    if ok:
        sys.path.insert(0, ROOT)
        from gymform.caption import build as cap_build, lint as cap_lint
        cap_txt = cap_build(c.get("caption", ""))
        cap_err = cap_lint(cap_txt)
        if cap_err:
            status["caption_lint"] = cap_err
            log("caption lint: " + "; ".join(cap_err))
        cap_p = A(c.get("out_caption") or f"output/day{a.day}_caption.txt")   # variants keep their own caption file
        open(cap_p, "w").write(cap_txt)
        status["caption"] = os.path.relpath(cap_p, ROOT)
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
