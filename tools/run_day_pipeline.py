#!/usr/bin/env python3
"""One-command @gymform.3d Day pipeline (Routinizer, 2026-11-03): one pipeline config -> master mp4 + result sheet +
JSON verdict. The bot only steps in for judgement (failures) and for video generation (GenerateVideo stays OUTSIDE).

    python3 tools/run_day_pipeline.py --config configs/pipeline/day07.json --plan     # validate + list steps, writes nothing
    python3 tools/run_day_pipeline.py --config configs/pipeline/day07.json            # full run (nice 19, yields to Shorts)
    python3 tools/run_day_pipeline.py --config configs/pipeline/day07.json --watch    # wait for the 2 clips, then run
    python3 tools/run_day_pipeline.py --config configs/pipeline/day07.json --manifest # (re)write the GenerateVideo handoff

Stages (each heavy one is a nice-19 subprocess that is SIGSTOP-paused while a Shorts render runs):
  1 validate   pipeline config + day config + DEFECT_PLAYBOOK sha + pinned stills + clip files
  2 handoff    clips missing -> clips manifest (GenerateVideo requests) + alert, exit 10 (or --watch waits)
  3 gates      raw-clip drift gate G1-G10 (docs/mission/2D_CONSISTENCY_PLAN.md 3d), REUSES
               tools/source_gate_smoke.clip_gate + consistency_metrics + startframe_gates + gymform.motion_check
  4 run_day    run_day.py on a per-run copy of the day config (clips/windows swapped in, every output path redirected
               into the run dir): Day 4 grade (native_grade + native_muscle_shader day4 + muscle_glow_clamp/target_regions
               via gymform/region_glow.py, as set in the day config) -> Day 4 template (title, preview pill, callout,
               see-through) -> TTS + SFX (gymform/tts.py) -> master mp4 -> gymform.qc + preflight gate + step sheet
  5 verdict    result sheet (jpg + md) + verdict.json; on failure: short summary + /home/box/routinizer/alerts.jsonl

Exit codes: 0 PASS | 1 PASS_WITH_WARNINGS is still 0 (warnings in verdict) | 2 config invalid | 3 input missing (stills)
            | 4 clip gate fail | 5 preflight fail | 6 render/QC fail | 7 run_day guard/lint | 8 box busy timeout
            | 9 internal error | 10 awaiting clips (manifest written) | 11 watch timeout
Rules kept: no xAI API (XAI_API_KEY is stripped from every child env, never read), no grok.com, no /workspace/bgm, no
per-frame cut-outs (the look is whatever the day config's native_grade does), nothing publishes, nothing in the repo is
deleted (only this run's own _work/ temp dir is removed at the end).
"""
import argparse
import datetime as _dt
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RZ_HOME = os.environ.get("ROUTINIZER_HOME", "/home/box/routinizer")
ALERTS = os.path.join(RZ_HOME, "alerts.jsonl")
KST = _dt.timezone(_dt.timedelta(hours=9))
SCHEMA = "gymform.day_pipeline/v1"
HARD_GATES_DEFAULT = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10"]
WARN_GATES_DEFAULT = ["G9"]   # motion_check: "existing module, not yet wired (D14)"; D7v2 t2 wrong did not fire it
# Shorts has box priority: these foreign processes pause us (rz_common.HEAVY_PATTERNS)
HEAVY_PATTERNS = ("flipcut.py", "geo_template", "blender", "ffmpeg")
EXIT = dict(ok=0, config=2, input=3, gate=4, preflight=5, render=6, guard=7, busy=8, internal=9, awaiting=10, watch_timeout=11)
EXIT_NAME = {v: k for k, v in EXIT.items()}


def iso():
    return _dt.datetime.now(KST).isoformat(timespec="seconds")


def A(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def rel(p):
    try:
        r = os.path.relpath(p, ROOT)
        return p if r.startswith("..") else r
    except ValueError:
        return p


def say(*a):
    print(f"[{iso()}]", *a, flush=True)


def sha8(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:8]


def child_env(threads="2"):
    e = {k: v for k, v in os.environ.items() if k != "XAI_API_KEY"}   # stripped, never read: no paid xAI possible
    e.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads, PYTHONUNBUFFERED="1")
    return e


# ---------------------------------------------------------------- box priority (Shorts first)
def _descendants(pid):
    kids, out = {}, set()
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            pp = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(pp, []).append(int(p))
    st = [pid]
    while st:
        x = st.pop()
        for k in kids.get(x, []):
            if k not in out:
                out.add(k); st.append(k)
    return out


def foreign_heavy():
    mine = _descendants(os.getpid()) | {os.getpid()}
    hits = []
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) in mine:
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if not cmd or "routinizer" in cmd or "run_day_pipeline" in cmd:
            continue
        if any(h in cmd for h in HEAVY_PATTERNS):
            hits.append(f"{p}:{cmd[:90]}")
    return hits


def box_busy():
    heavy = foreign_heavy()
    if heavy:
        return True, "Shorts/heavy render running: " + heavy[0]
    l1, cores = os.getloadavg()[0], os.cpu_count() or 1
    if l1 > cores:
        return True, f"load {l1:.2f} > {cores} cores"
    return False, f"load {l1:.2f}/{cores}"


def wait_idle(max_wait_s, poll=30):
    t0 = time.time()
    while True:
        busy, why = box_busy()
        if not busy:
            return True, why
        if time.time() - t0 > max_wait_s:
            return False, why
        say(f"box busy ({why}); waiting {poll}s (Shorts has priority)")
        time.sleep(poll)


def run_yielding(cmd, log_p, max_pause_s, cwd=None):
    """Run cmd at nice 19 in its own process group; SIGSTOP the group while a foreign heavy render runs."""
    with open(log_p, "a") as lf:
        lf.write(f"\n[{iso()}] $ {' '.join(cmd)}\n"); lf.flush()
        pr = subprocess.Popen(["nice", "-n", "19"] + cmd, cwd=cwd or ROOT, stdout=lf, stderr=subprocess.STDOUT,
                              env=child_env(), start_new_session=True)
        paused_at, paused_total, n_pauses = None, 0.0, 0
        while pr.poll() is None:
            time.sleep(5)
            heavy = foreign_heavy()
            if heavy and paused_at is None:
                os.killpg(pr.pid, signal.SIGSTOP); paused_at = time.time(); n_pauses += 1
                say(f"paused (yield to {heavy[0][:70]})"); lf.write(f"[{iso()}] PAUSE yield to {heavy[0]}\n"); lf.flush()
            elif not heavy and paused_at is not None:
                os.killpg(pr.pid, signal.SIGCONT); paused_total += time.time() - paused_at; paused_at = None
                say("resumed"); lf.write(f"[{iso()}] RESUME\n"); lf.flush()
            if paused_at is not None and paused_total + (time.time() - paused_at) > max_pause_s:
                os.killpg(pr.pid, signal.SIGCONT); os.killpg(pr.pid, signal.SIGTERM); pr.wait()
                return None, {"paused_s": round(paused_total + time.time() - paused_at), "pauses": n_pauses, "busy_timeout": True}
        return pr.returncode, {"paused_s": round(paused_total), "pauses": n_pauses}


# ---------------------------------------------------------------- config
def load_pipeline(path):
    p = A(path)
    cfg = json.load(open(p))
    if "pipeline" in cfg:   # pipeline block inside a day config
        pl = dict(cfg["pipeline"]); pl.setdefault("day_config", rel(p)); pl.setdefault("day", cfg.get("day"))
    else:
        pl = cfg
    pl["_path"] = p
    return pl


def validate(pl):
    """Returns (errors, warnings, info). errors = config/input problems; clips missing is NOT an error (handoff)."""
    err, warn, info = [], [], {}
    for k in ("day", "day_config", "forms"):
        if k not in pl:
            err.append(f"pipeline config: missing '{k}'")
    if err:
        return err, warn, info
    dc_p = A(pl["day_config"])
    if not os.path.exists(dc_p):
        return [f"day config not found: {pl['day_config']}"], warn, info
    dc = json.load(open(dc_p))
    info["day_config"] = dc
    if int(dc.get("day", -1)) != int(pl["day"]):
        err.append(f"day mismatch: pipeline day {pl['day']} vs day config day {dc.get('day')}")
    # DEFECT_PLAYBOOK read first (run_day refuses Day >= 7 without the current sha)
    pb = os.path.join(ROOT, "docs", "DEFECT_PLAYBOOK.md")
    pb_sha = sha8(pb)
    info["defect_playbook_sha"] = pb_sha
    info["playbook_rows"] = [ln.split("|")[1].strip() + " " + ln.split("|")[2].strip()
                             for ln in open(pb, encoding="utf-8") if ln.startswith("| D")]
    if int(pl["day"]) >= 7 and dc.get("defect_playbook_sha") != pb_sha:
        err.append(f"defect_playbook_sha: day config has {dc.get('defect_playbook_sha')!r}, docs/DEFECT_PLAYBOOK.md is {pb_sha!r} "
                   f"(read the playbook, then set it in {pl['day_config']}; this script never sets it for you)")
    # Day 4 grade flags (the look lives in the day config; this script only reports)
    look = {k: dc.get(k) for k in ("look", "native_muscle_shader", "muscle_glow_clamp", "muscle_shader")}
    info["look"] = look
    if dc.get("look") != "native_grade":
        err.append(f"look must be native_grade (no per-frame cut-outs, DEFECT_PLAYBOOK D4); got {dc.get('look')!r}")
    if dc.get("native_muscle_shader") != "day4":
        warn.append(f"native_muscle_shader is {dc.get('native_muscle_shader')!r}, not 'day4' (Day 4 fibre grade)")
    if not dc.get("muscle_glow_clamp") or not dc.get("target_regions"):
        warn.append("muscle_glow_clamp/target_regions not set: region_glow.py clamp is off")
    motions = dc.get("motions", {})
    forms = pl["forms"]
    for form in ("wrong", "right"):
        f = forms.get(form)
        if not f:
            err.append(f"forms.{form} missing"); continue
        if form not in motions:
            err.append(f"day config has no motions.{form}")
        for k in ("start", "clip"):
            if not f.get(k):
                err.append(f"forms.{form}.{k} missing")
        if f.get("start") and not os.path.exists(A(f["start"])):
            err.append(f"INPUT forms.{form}.start not found: {f['start']}")
        kfs = f.get("keyframes") or []
        if not kfs:
            err.append(f"forms.{form}.keyframes: at least one pinned contraction keyframe is required (G2/G3)")
        dur = float(pl.get("generate", {}).get("duration", 5))
        for i, k in enumerate(kfs):
            if not os.path.exists(A(k.get("path", ""))):
                err.append(f"INPUT forms.{form}.keyframes[{i}] not found: {k.get('path')}")
            if not (0 < float(k.get("t", -1)) < dur):
                err.append(f"forms.{form}.keyframes[{i}].t must be strictly between 0 and duration {dur}")
    if any("/workspace/bgm" in json.dumps(x) for x in (pl, dc)):
        err.append("config references /workspace/bgm (commercial music is off-limits)")
    if dc.get("motion_standard") is None:
        warn.append("day config has no motion_standard: G9 (motion_check) cannot run")
    return err, warn, info


def clip_status(pl):
    out = {}
    for form in ("wrong", "right"):
        p = A(pl["forms"][form]["clip"])
        st = {"path": p, "exists": os.path.exists(p)}
        if st["exists"]:
            try:
                import cv2
                cap = cv2.VideoCapture(p)
                n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                st.update(frames=n, fps=round(fps, 3), dur_s=round(n / fps, 2) if fps else None,
                          size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
                          readable=n > 0 and fps > 0, bytes=os.path.getsize(p))
            except Exception as e:   # noqa
                st.update(readable=False, error=str(e)[:120])
        out[form] = st
    return out


# ---------------------------------------------------------------- GenerateVideo handoff
def prompt_for(pl, form):
    f = pl["forms"][form]
    if f.get("prompt"):
        return f["prompt"]
    t = pl.get("prompt_template")
    if t:
        return t.format(form=form, **f.get("prompt_vars", {}))
    return ""


def manifest(pl):
    g = pl.get("generate", {})
    dur, res, ar = int(g.get("duration", 5)), g.get("resolution", "720p"), g.get("aspect_ratio", "9:16")
    cs = clip_status(pl)
    clips = []
    for form in ("wrong", "right"):
        f = pl["forms"][form]
        tgt = A(f["clip"])
        clips.append({
            "form": form,
            "target_path": tgt,
            "exists": cs[form]["exists"],
            "generate_video": {   # pass these keys 1:1 to Grok Bot GenerateVideo
                "prompt": prompt_for(pl, form),
                "image_path": A(f["start"]),
                "last_frame_path": A(f.get("last") or f["start"]),   # first = last frame: seamless loop (2D plan 3b)
                "keyframes": [{"image_path": A(k["path"]), "timestamp_s": float(k["t"])} for k in f["keyframes"]],
                "duration": dur, "resolution": res, "aspect_ratio": ar,
                "filename": os.path.basename(tgt),
            },
            "after_generation": f"save/copy the returned mp4 to target_path ({rel(tgt)}); do not re-grade or re-encode it",
        })
    return {
        "schema": "gymform.clips_manifest/v1", "day": int(pl["day"]), "written_at": iso(),
        "pipeline_config": rel(pl["_path"]),
        "status": "ready" if all(c["exists"] for c in clips) else "awaiting_clips",
        "rules": ["Grok Bot GenerateVideo only (2 calls). No grok.com, no xAI API.",
                  f"One take per clip (max_takes {g.get('max_takes', 1)}); a rejected take is not re-rolled blindly: read the gate report first.",
                  "Stills are the frozen pinned frames; never regrade them before passing them as pins."],
        "clips": clips,
        "resume": {"watch": f"nice -n 19 python3 tools/run_day_pipeline.py --config {rel(pl['_path'])} --watch",
                   "once": f"nice -n 19 python3 tools/run_day_pipeline.py --config {rel(pl['_path'])}"},
    }


def manifest_path(pl):
    return A(pl.get("manifest_path") or os.path.join(pl.get("out_root", "output/runs"), f"day{int(pl['day']):02d}_clips_manifest.json"))


def write_manifest(pl):
    m = manifest(pl)
    p = manifest_path(pl)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(m, open(p, "w"), indent=1)
    return p, m


# ---------------------------------------------------------------- alerts
def alert(kind, day, message, severity="error", **extra):
    os.makedirs(RZ_HOME, exist_ok=True)
    rec = dict(at=iso(), source="gymform_day", watcher=f"gymform_day_d{int(day):02d}", kind=kind, key=f"day{int(day):02d}",
               severity=severity, message=message[:600], **extra)
    with open(ALERTS, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(rec, ensure_ascii=False) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
    say("ALERT", kind, message[:200])


# ---------------------------------------------------------------- stage 3: G1-G10 (child process)
def gate_child(args):
    """Internal: python3 tools/run_day_pipeline.py _gate <spec.json> <out.json>. Reuses tools/source_gate_smoke.py."""
    spec = json.load(open(args[0]))
    sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, ROOT)
    import numpy as np
    import source_gate_smoke as sgs   # clip_gate = G1-G6, G8 per clip, G10, motion keypoints
    from gymform import motion_check as mc
    res = {}
    for form, c in spec["forms"].items():
        kp = ",".join(k["path"] for k in c["keyframes"]); kt = ",".join(str(k["t"]) for k in c["keyframes"])
        res[form] = sgs.clip_gate(form, c["clip"], c["start"], kp, kt)
    # between-clip G7/G8 + G9: same rules as source_gate_smoke.main(), exercise taken from the day config
    std = mc.load_standards(); ex = (std.get("exercises") or std).get(spec.get("exercise") or "", None)
    view = spec.get("view", "3/4")
    for nm in res:
        K = res[nm].pop("_K")
        if ex is None:
            res[nm]["G9"] = {"pass": False, "skipped": f"no motion standard {spec.get('exercise')!r}"}
        elif nm == "wrong":
            items, summ = mc.check_wrong_step(ex, K, view=view)
            res[nm]["G9"] = {"kind": "wrong", "fired": summ["value"], "pass": summ["pass"], "severity": summ["severity"],
                             "items": [(i["id"], i.get("value"), i.get("expected"), i.get("pass")) for i in items]}
        else:
            items = mc.check_right_step(ex, K, view=view)
            res[nm]["G9"] = {"kind": "right", "pass": all(i.get("pass") is not False for i in items if i.get("severity") == "fail"),
                             "items": [(i["id"], None if i.get("value") is None else round(i["value"], 1), i.get("expected"),
                                        i.get("pass"), i.get("severity")) for i in items]}
    names = list(res)
    v = [res[n]["view_mean"] for n in names]; cfs = [res[n]["G8_clip"]["cobalt_frac_mean"] for n in names]
    relsp = (max(cfs) - min(cfs)) / np.mean(cfs) * 100 if np.mean(cfs) else 999
    for n in names:
        c = res[n].pop("G8_clip")
        res[n]["G7"] = {"view_spread_between_clips": round(max(v) - min(v), 3), "pass": (max(v) - min(v)) <= 0.05}
        res[n]["G8"] = dict(c, between_rel_pct=round(relsp, 1),
                            **{"pass": bool(c["cobalt_cv_pct"] <= 10 and relsp <= 20 and c["cobalt_rgb_dist_vs_start"] is not None
                                            and c["cobalt_rgb_dist_vs_start"] <= 20)})
    json.dump(res, open(args[1], "w"), indent=1, default=float)


def summarize_gates(g, hard, warnl):
    fails, warns, table = [], [], {}
    for form, r in g.items():
        table[form] = {k: bool(r[k]["pass"]) for k in sorted(r, key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)
                       if re.fullmatch(r"G\d+", k) and isinstance(r[k], dict)}
        for k, ok in table[form].items():
            if not ok:
                d = {kk: vv for kk, vv in r[k].items() if kk != "pass" and not isinstance(vv, list)}
                msg = f"{form} {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in list(d.items())[:4])
                (fails if k in hard else warns if k in warnl else warns).append(msg)
    return table, fails, warns


# ---------------------------------------------------------------- stage 4: run config for run_day.py
def build_run_config(pl, dc, cs, run_dir, work):
    c = json.loads(json.dumps(dc))
    assets = os.path.join(work, "assets"); os.makedirs(assets, exist_ok=True)
    windows = {}
    for form in ("wrong", "right"):
        f = pl["forms"][form]; st = cs[form]
        fn = f"day{int(pl['day']):02d}_{form}_clip.mp4"
        dst = os.path.join(assets, fn)
        if not os.path.exists(dst):
            shutil.copy2(st["path"], dst)   # copy (not link) so the run is reproducible; removed with _work/
        w = f.get("window") or {}
        i0 = int(w.get("i0", 0))
        i1 = w.get("i1")
        if i1 is None:   # stretch (frame 0) -> first pinned contraction keyframe; run_day's cosine wave loops it
            i1 = int(round(float(f["keyframes"][0]["t"]) * st["fps"]))
        i1 = min(int(i1), st["frames"] - 1)
        m = c["motions"][form]
        m["base"] = fn; m["i0"] = i0; m["i1"] = i1
        for r in (m.get("renders") or {}).values():
            r["file"] = fn
        windows[form] = [i0, i1]
    tag = f"day{int(pl['day']):02d}"
    c.update(assets_dir=assets, kp_cache=os.path.join(work, f"_kps_{tag}.pkl"), tmp_dir=os.path.join(work, "tmp"),
             qc_dir=os.path.join(run_dir, "qc"),
             out_mp4=os.path.join(run_dir, f"{tag}_master.mp4"), out_sheet=os.path.join(run_dir, f"{tag}_step_sheet.jpg"),
             out_cover=os.path.join(run_dir, f"{tag}_cover.jpg"), out_caption=os.path.join(run_dir, f"{tag}_caption.txt"))
    c["source_clips_note"] = (f"run_day_pipeline {os.path.basename(run_dir)}: clips from {pl['forms']['wrong']['clip']} / "
                              f"{pl['forms']['right']['clip']}, windows {windows}")
    c.update(pl.get("day_config_overrides", {}))
    p = os.path.join(run_dir, f"{tag}__{os.path.basename(run_dir)}.json")   # unique name -> own overrides + preflight stamp
    json.dump(c, open(p, "w"), indent=1)
    return p, windows


def parse_run_day(log_p):
    last = None
    for ln in open(log_p, errors="replace"):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                last = json.loads(ln)
            except ValueError:
                pass
    return last


# ---------------------------------------------------------------- stage 5: result sheet
def result_sheet(run_dir, mp4, out_jpg, log_p):
    if not os.path.exists(mp4):
        return None
    vf = "fps=1,scale=216:384,tile=5x2:padding=6:color=0x23272B"
    rc = subprocess.run(["nice", "-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-i", mp4, "-vf", vf,
                         "-frames:v", "1", out_jpg], stdout=open(log_p, "a"), stderr=subprocess.STDOUT, env=child_env("1")).returncode
    return out_jpg if rc == 0 and os.path.exists(out_jpg) else None


def write_result(run_dir, v):
    json.dump(v, open(os.path.join(run_dir, "verdict.json"), "w"), indent=1, ensure_ascii=False)
    L = [f"# Day {v['day']} pipeline run {v['run_id']}: **{v['verdict']}** (exit {v['exit_code']} {EXIT_NAME.get(v['exit_code'], '')})", "",
         f"- at: {v['finished_at']} (KST)  stage reached: {v['stage']}", f"- pipeline config: `{v['pipeline_config']}`",
         f"- day config: `{v['day_config']}` (playbook sha {v.get('defect_playbook_sha')})"]
    if v.get("failure_summary"):
        L += ["", "## Failure summary (for the bot)", "", v["failure_summary"]]
    if v.get("gates"):
        L += ["", "## Raw-clip gates G1-G10", "", "| form | " + " | ".join(next(iter(v["gates"].values())).keys()) + " |",
              "|---|" + "---|" * len(next(iter(v["gates"].values())))]
        for form, t in v["gates"].items():
            L.append(f"| {form} | " + " | ".join("✓" if x else "✗" for x in t.values()) + " |")
    if v.get("run_day"):
        rd = v["run_day"]
        L += ["", "## run_day.py", "", f"- ok: {rd.get('ok')}  preflight: {rd.get('preflight')}  qc: {rd.get('qc')}",
              f"- fails: {rd.get('fails')}"]
    if v.get("warnings"):
        L += ["", "## Warnings (judgement, not failures)", ""] + [f"- {w}" for w in v["warnings"]]
    L += ["", "## Outputs", ""] + [f"- {k}: `{x}`" for k, x in (v.get("outputs") or {}).items() if x]
    L += ["", f"Timings (s): {json.dumps(v.get('timings', {}))}", "", "Nothing was published."]
    open(os.path.join(run_dir, "result.md"), "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main run
def full_run(pl, a):
    t0 = time.time()
    day = int(pl["day"])
    err, warn, info = validate(pl)
    if err:
        code = EXIT["input"] if all(e.startswith("INPUT") for e in err) else EXIT["config"]
        msg = "; ".join(err)
        say("INVALID:", msg)
        alert("config_invalid" if code == 2 else "input_missing", day, msg, pipeline_config=rel(pl["_path"]), exit_code=code)
        return code
    cs = clip_status(pl)
    if not all(s["exists"] for s in cs.values()):
        mp, _ = write_manifest(pl)
        missing = [f for f, s in cs.items() if not s["exists"]]
        say(f"clips missing {missing}: manifest -> {rel(mp)}")
        if not a.no_alert_awaiting:
            alert("needs_clips", day, f"Day {day}: generate {len(missing)} clip(s) {missing} with GenerateVideo per {rel(mp)}, "
                  "then the watcher resumes", severity="action", manifest=mp, exit_code=EXIT["awaiting"])
        return EXIT["awaiting"]
    bad = [f for f, s in cs.items() if not s.get("readable")]
    if bad:
        msg = f"clip(s) unreadable: {bad}"
        alert("input_missing", day, msg, exit_code=EXIT["input"]); say(msg)
        return EXIT["input"]
    out_root = A(a.out_root or pl.get("out_root", "output/runs"))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"day{day:02d}_{run_id}")
    while os.path.exists(run_dir):
        run_id += "b"; run_dir = os.path.join(out_root, f"day{day:02d}_{run_id}")
    work = os.path.join(run_dir, "_work")
    os.makedirs(work)
    open(os.path.join(run_dir, ".gitignore"), "w").write("_work/\nqc/\nqc_preflight/\n*.log\n")
    log_p = os.path.join(run_dir, "pipeline.log")
    hard = pl.get("gates", {}).get("hard", HARD_GATES_DEFAULT); warnl = pl.get("gates", {}).get("warn", WARN_GATES_DEFAULT)
    V = {"schema": SCHEMA, "day": day, "run_id": run_id, "run_dir": run_dir, "pipeline_config": rel(pl["_path"]),
         "day_config": pl["day_config"], "defect_playbook_sha": info["defect_playbook_sha"], "look": info["look"],
         "started_at": iso(), "clips": cs, "warnings": list(warn), "timings": {}, "box": {}, "outputs": {}}
    max_wait, max_pause = float(a.max_wait_min) * 60, float(a.max_pause_min) * 60

    def finish(code, stage, summary=None):
        V.update(exit_code=code, stage=stage, finished_at=iso(), failure_summary=summary,
                 verdict=("PASS_WITH_WARNINGS" if V["warnings"] else "PASS") if code == 0 else "FAIL",
                 ok=code == 0)
        V["timings"]["total"] = round(time.time() - t0, 1)
        if code != 0:
            V["next_action_for_bot"] = {
                EXIT["gate"]: "Read gates.json: name the failed gate/metric first, then decide (new pins? 1 retake?). No blind re-roll.",
                EXIT["preflight"]: "Read run_day.json fails + qc_preflight/: fix by DEFECT_PLAYBOOK row (render param, never a threshold).",
                EXIT["render"]: "Read run_day.json fails + the run_day log; fix by DEFECT_PLAYBOOK row.",
                EXIT["guard"]: "run_day.py guard/lint refused (callouts, step order, camera plan, caption, playbook sha): fix the day config.",
                EXIT["busy"]: "Box stayed busy with Shorts; re-run later (nothing is wrong with the reel).",
            }.get(code, "Read pipeline.log.")
        sheet = result_sheet(run_dir, V["outputs"].get("master_mp4") or "", os.path.join(run_dir, f"day{day:02d}_result_sheet.jpg"), log_p)
        V["outputs"]["result_sheet"] = sheet
        V["outputs"]["result_md"] = os.path.join(run_dir, "result.md")
        V["outputs"]["verdict_json"] = os.path.join(run_dir, "verdict.json")
        write_result(run_dir, V)
        if not a.keep_work and os.path.isdir(work):
            shutil.rmtree(work, ignore_errors=True)   # only this run's own temp dir
            V["work_removed"] = True
            json.dump(V, open(os.path.join(run_dir, "verdict.json"), "w"), indent=1, ensure_ascii=False)
        if code != 0:
            alert({4: "gate_fail", 5: "preflight_fail", 6: "render_fail", 7: "run_day_guard", 8: "box_busy"}.get(code, "pipeline_error"),
                  day, f"Day {day} {EXIT_NAME.get(code)}: {summary or ''}", severity="warn" if code == 8 else "error",
                  run_dir=run_dir, verdict=os.path.join(run_dir, "verdict.json"), exit_code=code)
        say(f"DONE {V['verdict']} exit {code} -> {rel(run_dir)}")
        print(json.dumps({"day": day, "verdict": V["verdict"], "exit": code, "stage": stage, "run_dir": rel(run_dir),
                          "summary": summary}, ensure_ascii=False))
        return code

    try:
        ok, why = wait_idle(max_wait)
        V["box"]["start"] = why
        if not ok:
            return finish(EXIT["busy"], "wait_idle", f"box busy > {a.max_wait_min} min: {why}")
        # ---- stage 3: gates
        tt = time.time()
        gspec = {"exercise": info["day_config"].get("motion_standard"), "view": pl.get("gate_view", "3/4"),
                 "forms": {f: {"clip": cs[f]["path"], "start": A(pl["forms"][f]["start"]),
                               "keyframes": [{"path": A(k["path"]), "t": k["t"]} for k in pl["forms"][f]["keyframes"]]}
                           for f in ("wrong", "right")}}
        gs_p = os.path.join(work, "gate_spec.json"); json.dump(gspec, open(gs_p, "w"))
        g_out = os.path.join(run_dir, "gates.json")
        if a.skip_gates:
            V["warnings"].append("gates skipped (--skip-gates)")
        else:
            rc, yl = run_yielding([sys.executable, os.path.abspath(__file__), "_gate", gs_p, g_out], log_p, max_pause)
            V["box"]["gates"] = yl
            V["timings"]["gates"] = round(time.time() - tt, 1)
            if rc is None:
                return finish(EXIT["busy"], "gates", "paused too long for Shorts during the gates")
            if rc != 0 or not os.path.exists(g_out):
                return finish(EXIT["internal"], "gates", "gate tool crashed (see pipeline.log)")
            table, gfails, gwarns = summarize_gates(json.load(open(g_out)), hard, warnl)
            V["gates"] = table; V["outputs"]["gates_json"] = g_out
            V["warnings"] += [f"{w} (warn-only gate)" for w in gwarns]
            if gfails:
                return finish(EXIT["gate"], "gates", "Raw-clip gate FAIL (reject the clip, not the render): " + " | ".join(gfails))
        if a.gates_only:
            return finish(0, "gates", None)
        # ---- stage 4: run_day.py
        rc_p, windows = build_run_config(pl, info["day_config"], cs, run_dir, work)
        V["run_config"] = rc_p; V["windows"] = windows
        ok, why = wait_idle(max_wait)
        if not ok:
            return finish(EXIT["busy"], "wait_idle", f"box busy > {a.max_wait_min} min before render: {why}")
        tt = time.time()
        rd_log = os.path.join(run_dir, "run_day.log")
        cmd = [sys.executable, os.path.join(ROOT, "run_day.py"), "--day", str(day), "--config", rc_p,
               "--workers", str(pl.get("workers", 2))]
        rc, yl = run_yielding(cmd, rd_log, max_pause)
        V["box"]["run_day"] = yl
        V["timings"]["run_day"] = round(time.time() - tt, 1)
        if rc is None:
            return finish(EXIT["busy"], "run_day", "paused too long for Shorts during run_day")
        st = parse_run_day(rd_log) or {}
        json.dump(st, open(os.path.join(run_dir, "run_day.json"), "w"), indent=1)
        V["run_day"] = {k: st.get(k) for k in ("ok", "stage", "preflight", "qc", "fails", "blocker", "error", "lint", "checks", "autofix", "t")}
        V["outputs"]["run_day_log"] = rd_log
        mp4 = json.load(open(rc_p))["out_mp4"]
        for k, p in (("master_mp4", mp4), ("step_sheet", json.load(open(rc_p))["out_sheet"]),
                     ("caption", json.load(open(rc_p))["out_caption"]), ("qc_dir", os.path.join(run_dir, "qc")),
                     ("preflight_dir", os.path.join(run_dir, "qc_preflight"))):
            V["outputs"][k] = p if os.path.exists(p) else None
        if st.get("caption_lint"):
            V["warnings"].append("caption lint: " + "; ".join(st["caption_lint"]))
        if rc == 0 and st.get("ok"):
            return finish(0, "verdict", None)
        if rc == 4 or st.get("stage") == "preflight":
            return finish(EXIT["preflight"], "preflight", "Preflight FAIL: " + " | ".join(str(x)[:140] for x in (st.get("fails") or [])[:6]))
        if rc in (2, 3) or st.get("stage") in ("guard", "lint", "playbook"):
            return finish(EXIT["guard"], st.get("stage") or "run_day", "run_day refused: " +
                          str(st.get("blocker") or st.get("error") or st.get("lint") or st.get("missing_clips") or f"exit {rc}")[:400])
        return finish(EXIT["render"], st.get("stage") or "render", f"run_day exit {rc}, stage {st.get('stage')}: " +
                      " | ".join(str(x)[:140] for x in (st.get("fails") or [])[:6]))
    except Exception as e:   # noqa
        import traceback
        open(log_p, "a").write(traceback.format_exc())
        return finish(EXIT["internal"], "internal", f"{type(e).__name__}: {str(e)[:200]}")


def plan(pl, a):
    err, warn, info = validate(pl)
    cs = clip_status(pl) if "forms" in pl else {}
    busy, why = box_busy()
    dc = info.get("day_config") or {}
    steps = [
        "1 validate: pipeline + day config, DEFECT_PLAYBOOK sha, pinned stills, clips",
        "2 handoff: if a clip is missing -> write clips manifest (GenerateVideo) + alert needs_clips, exit 10 (--watch waits)",
        f"3 gates: G1-G10 raw-clip drift gate on both clips (hard: {','.join(pl.get('gates', {}).get('hard', HARD_GATES_DEFAULT))}; "
        f"warn: {','.join(pl.get('gates', {}).get('warn', WARN_GATES_DEFAULT))}), exercise {dc.get('motion_standard')!r}",
        f"4 run_day.py --day {pl.get('day')} --config <run copy of {pl.get('day_config')}> (clips swapped, outputs -> run dir): "
        f"look {info.get('look')} -> Day 4 template -> TTS+SFX -> master mp4 -> preflight + QC + step sheet",
        "5 verdict: result sheet jpg + result.md + verdict.json; on failure: alert -> /home/box/routinizer/alerts.jsonl",
    ]
    out = {"mode": "plan", "day": pl.get("day"), "ok": not err, "errors": err, "warnings": warn,
           "defect_playbook_sha": info.get("defect_playbook_sha"), "playbook_rows": info.get("playbook_rows"),
           "clips": cs, "clips_ready": bool(cs) and all(s["exists"] for s in cs.values()),
           "box_busy": busy, "box": why, "steps": steps,
           "manifest_preview": manifest(pl) if not err else None,
           "would_exit": (EXIT["config"] if err else EXIT["awaiting"] if not all(s["exists"] for s in cs.values()) else 0)}
    # run_day.py's own --check on a scratch copy of the run config (lint + guards, renders nothing)
    if not err and out["clips_ready"]:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="rdp_plan_") as td:
            p, windows = build_run_config(pl, dc, cs, td, os.path.join(td, "_work"))
            r = subprocess.run(["nice", "-n", "19", sys.executable, os.path.join(ROOT, "run_day.py"), "--day", str(pl["day"]),
                                "--config", p, "--check"], cwd=ROOT, capture_output=True, text=True, env=child_env("1"))
            last = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            out["windows"] = windows
            out["run_day_check"] = {"exit": r.returncode, "result": json.loads(last[-1]) if last else r.stdout[-400:]}
            if r.returncode != 0:
                out["ok"] = False
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return 0 if out["ok"] else EXIT["config"]


def watch(pl, a):
    day = int(pl["day"])
    os.makedirs(os.path.join(RZ_HOME, "locks"), exist_ok=True); os.makedirs(os.path.join(RZ_HOME, "state"), exist_ok=True)
    lock = open(os.path.join(RZ_HOME, "locks", f"gymform_day_d{day:02d}.lock"), "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        say("another watcher for this day is running; exit"); return 0
    hb = os.path.join(RZ_HOME, "state", f"gymform_day_d{day:02d}.heartbeat.json")
    mp, _ = write_manifest(pl)
    say(f"watching for clips (manifest {rel(mp)})")
    t0, last_sizes, alerted = time.time(), None, False
    while True:
        cs = clip_status(pl)
        sizes = {f: s.get("bytes") for f, s in cs.items()}
        json.dump({"watcher": f"gymform_day_d{day:02d}", "at": iso(), "pid": os.getpid(), "clips": {f: s["exists"] for f, s in cs.items()}},
                  open(hb, "w"))
        if all(s["exists"] and s.get("readable") for s in cs.values()) and sizes == last_sizes:   # stable for one poll
            say("clips present and stable -> running the pipeline")
            write_manifest(pl)
            return full_run(pl, a)
        if not alerted and not all(s["exists"] for s in cs.values()) and not a.no_alert_awaiting:
            alert("needs_clips", day, f"Day {day}: waiting for clips; generate per {rel(mp)}", severity="action", manifest=mp,
                  exit_code=EXIT["awaiting"])
            alerted = True
        last_sizes = sizes
        if time.time() - t0 > float(a.watch_timeout_h) * 3600:
            alert("watch_timeout", day, f"Day {day}: clips not delivered within {a.watch_timeout_h} h", severity="warn", manifest=mp,
                  exit_code=EXIT["watch_timeout"])
            return EXIT["watch_timeout"]
        time.sleep(float(a.interval))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_gate":
        return gate_child(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="configs/pipeline/dayNN.json (or a day config with a 'pipeline' block)")
    ap.add_argument("--plan", "--dry-run", dest="plan", action="store_true", help="validate + list steps; renders nothing")
    ap.add_argument("--manifest", action="store_true", help="write the GenerateVideo clips manifest and exit")
    ap.add_argument("--watch", action="store_true", help="wait until both clips exist (stable), then run")
    ap.add_argument("--interval", default=60, help="watch poll seconds")
    ap.add_argument("--watch-timeout-h", default=24)
    ap.add_argument("--gates-only", action="store_true", help="stop after G1-G10 (no render)")
    ap.add_argument("--skip-gates", action="store_true", help="debug only: render without G1-G10")
    ap.add_argument("--out-root", default="", help="default output/runs (each run gets its own new dir)")
    ap.add_argument("--max-wait-min", default=120, help="max wait for an idle box before starting a stage")
    ap.add_argument("--max-pause-min", default=180, help="max total pause (yielding to Shorts) per stage")
    ap.add_argument("--keep-work", action="store_true", help="keep _work/ (copied clips, frames, kp cache)")
    ap.add_argument("--no-alert-awaiting", action="store_true", help="do not alert when clips are missing")
    a = ap.parse_args()
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    pl = load_pipeline(a.config)
    if a.plan:
        return plan(pl, a)
    if a.manifest:
        p, m = write_manifest(pl); print(json.dumps({"manifest": p, "status": m["status"]})); return 0
    if a.watch:
        return watch(pl, a)
    return full_run(pl, a)


if __name__ == "__main__":
    sys.exit(main() or 0)
