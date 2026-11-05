#!/usr/bin/env python3
"""rig3d end-to-end pipeline (Routinizer v2, 2026-11-04): free motion source -> Grab3D CC0 rig -> Blender -> Day 4 grade
-> TTS/caption/reel -> auto-QC (+1 rule-based retry per fix) -> result sheet -> delivery. 0 generations, no xAI, no
paid APIs, never posts anywhere (ig_publish.py is not imported or called).

  nice -n 19 python3 tools/rig3d/pipeline.py --exercise db_row            # resumable: finished stages are skipped
  ... --force grade        re-run one stage (and everything after it)
  ... --short 12           render only the first N frames (cost probe: time + RAM), stops after the render stage
  ... --deliver-dry-run    build the delivery but do not commit/push/alert
  (default)                e-mail the result (sheet + QC summary) to the owner's Gmail (rz_notify_mail notify_to) via
                           Routinizer rz_mail_digest (batched, <= 30 min) -- 중현 님 t719u "Gmail로 결과보내주면 되자나"
  ... --mail-direct        send that e-mail now through the Gmail API (message id in run.json deliver results)
  ... --no-email           opt-out: commit/push/alert for real, but no owner e-mail
  ... --no-gate            skip the box load gate (default: wait in /home/box/routinizer/rz_loadgate.py)

Stages (each cached by a hash of its config slice + code + upstream hashes; stamps in _cache_rig3d/<slug>/stamps):
  1 config  2 fetch  3 pose (MediaPipe + source check)  4/5 render (retarget + Blender beauty + muscle mask)
  6 grade (Day 4)  7 reel (TTS, caption, encode)  8 qc  9 sheet  10 deliver
QC failures with a rule (qc.json `fix`) are retried once each; anything else -> one AI-queue item for Fitness Studio
(/home/box/routinizer/ai_queue/fitness_studio.jsonl) + a warn alert. Outputs: output/_rig3d/<slug>/ (gitignored)."""
import argparse, asyncio, copy, datetime as dt, hashlib, json, math, os, shutil, subprocess, sys, threading, time, urllib.request, wave
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, "/home/box/routinizer")
HERE = os.path.join(ROOT, "tools/rig3d"); BPY = os.environ.get("RIG3D_BPY", "/workspace/.bpyenv/bin/python")
KST = dt.timezone(dt.timedelta(hours=9))
AIQ = "/home/box/routinizer/ai_queue/fitness_studio.jsonl"
FITNESS_STUDIO_AGENT = "7c203a21-2233-41e0-92f1-189a01945890"
STAGES = ["config", "fetch", "pose", "render", "grade", "reel", "qc", "sheet", "deliver"]
CODE = {"pose": ["tools/rig3d/clip_pose.py"], "render": ["tools/rig3d/retarget_render_v2.py", "tools/rig3d/hand_grip.py"], "grade": ["tools/rig3d/grade3d.py", "tools/rig3d/sharp.py", "gymform/muscle.py", "gymform/layout.py"],
        "reel": ["gymform/caption.py", "gymform/tts.py"], "qc": ["tools/rig3d/qc3d.py", "tools/rig3d/sharp.py", "tools/rig3d/posture_qc.py", "tools/rig3d/anatomy_qc.py", "tools/rig3d/hand_grip.py"], "sheet": ["tools/rig3d/sheet.py"], "deliver": ["tools/rig3d/deliver.py"]}
PEAK = {"rss_mb": 0}

def log(m): print(f"[{dt.datetime.now(KST):%H:%M:%S}] {m}", flush=True)
def jdump(o, p): os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(o, open(p, "w"), indent=1, ensure_ascii=False)
def fsha(p): return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()[:16] if os.path.exists(os.path.join(ROOT, p)) else "missing"
def hsh(*parts): return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]

def rss_tree(pid):
    kids, tot = {pid}, 0
    try:
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    st = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
                    if int(st[1]) in kids: kids.add(int(d))
                except OSError: pass
        for k in kids:
            for line in open(f"/proc/{k}/status"):
                if line.startswith("VmRSS:"): tot += int(line.split()[1]) // 1024
    except OSError: pass
    return tot

def run_child(cmd, env=None, cwd=ROOT, logf=None):
    env = dict(os.environ if env is None else env); env["RZ_GATED"] = "1"
    out = open(logf, "w") if logf else subprocess.DEVNULL
    p = subprocess.Popen(["nice", "-n", "19"] + cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT); peak = 0
    while p.poll() is None:
        peak = max(peak, rss_tree(p.pid)); time.sleep(0.5)
    PEAK["rss_mb"] = max(PEAK["rss_mb"], peak); PEAK["stage"] = max(PEAK.get("stage", 0), peak)
    if p.returncode: raise RuntimeError(f"{os.path.basename(cmd[1] if len(cmd) > 1 else cmd[0])} exit {p.returncode} (log {logf})")
    return peak

def gate(args, what):
    if args.no_gate: return
    try:
        import rz_loadgate as g
    except ImportError:
        log("rz_loadgate not found: running without the gate"); return
    ok, why = g.check()
    if not ok:
        log(f"box busy ({why}): waiting before {what}")
        ok, why = g.wait(timeout=args.gate_timeout, poll=60)
        if not ok: raise SystemExit(f"load gate timeout before {what}: {why}")

class Run:
    def __init__(self, args):
        self.args = args; self.slug = args.exercise
        self.cfg_path = os.path.join(ROOT, "exercises", f"{self.slug}.yaml")
        self.work = os.path.join(ROOT, "output/_rig3d", self.slug); self.stamps = os.path.join(ROOT, "_cache_rig3d", self.slug, "stamps")
        os.makedirs(self.work, exist_ok=True); os.makedirs(self.stamps, exist_ok=True)
        self.h = {}; self.times = {}; self.forced = set()
        if args.force:
            i = STAGES.index(args.force); self.forced = set(STAGES[i:])
        self.overrides = {}
    def p(self, *a): return os.path.join(self.work, *a)
    def stage(self, name, key, fn, outputs):
        h = hsh(name, key, [fsha(c) for c in CODE.get(name, [])], [self.h.get(s) for s in STAGES[:STAGES.index(name)]])
        sp = os.path.join(self.stamps, name + ".json"); st = json.load(open(sp)) if os.path.exists(sp) else {}
        if name not in self.forced and st.get("hash") == h and all(os.path.exists(o) for o in outputs):
            self.h[name] = h; log(f"{name}: cached"); self.times[name] = {"cached": True, "computed_s": st.get("s"), "computed_at": st.get("at")}; return st.get("info")
        t = time.time(); log(f"{name}: run"); PEAK["stage"] = 0; info = fn()
        self.h[name] = h; self.times[name] = {"s": round(time.time() - t, 1)}
        jdump({"hash": h, "at": dt.datetime.now(KST).isoformat(timespec="seconds"), "s": round(time.time() - t, 1), "peak_rss_mb": PEAK.get("stage", 0), "info": info}, sp)
        self.forced.discard(name); return info

    # 1 ------------------------------------------------------------------------------------------------------------------
    def s_config(self):
        c = yaml.safe_load(open(self.cfg_path))
        for k in ("slug", "name", "source", "asset", "motion", "render", "grade", "tts", "caption", "qc"):
            if k not in c: raise SystemExit(f"exercise config missing '{k}'")
        for path, val in self.overrides.items():
            d = c
            for k in path.split(".")[:-1]: d = d[k]
            d[path.split(".")[-1]] = val
        c["_overrides"] = dict(self.overrides); self.cfg = c; return c

    # 2 ------------------------------------------------------------------------------------------------------------------
    def fetch(self, url, dst, min_bytes):
        dst = os.path.join(ROOT, dst)
        if os.path.exists(dst) and os.path.getsize(dst) >= min_bytes: return {"file": dst, "bytes": os.path.getsize(dst), "downloaded": False}
        os.makedirs(os.path.dirname(dst), exist_ok=True); tmp = dst + ".part"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (gymform3d rig3d pipeline)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f: shutil.copyfileobj(r, f)
        if os.path.getsize(tmp) < min_bytes: raise SystemExit(f"download too small: {url}")
        os.replace(tmp, dst); return {"file": dst, "bytes": os.path.getsize(dst), "downloaded": True}
    def s_fetch(self):
        S, Hd = self.cfg["source"], self.cfg["asset"]["hdri"]
        if not os.path.exists(os.path.join(ROOT, self.cfg["asset"]["glb"])): raise SystemExit("Grab3D GLB missing (see docs/mission/REUSABLE_ASSET_PLAN.md)")
        return {"clip": self.fetch(S["url"], S["file"], int(S.get("min_bytes", 100000))), "hdri": self.fetch(Hd["url"], Hd["file"], 200000)}

    # 3 ------------------------------------------------------------------------------------------------------------------
    def s_pose(self):
        gate(self.args, "pose"); out = self.p("pose", "pose.npz"); os.makedirs(self.p("pose"), exist_ok=True)
        run_child(["python3", os.path.join(HERE, "clip_pose.py"), os.path.join(ROOT, self.cfg["source"]["file"]), out], logf=self.p("pose", "log.txt"))
        chk = json.load(open(out.replace(".npz", "_check.json")))
        hard = chk["full_body"]["pass"] and chk["reps"]["pass"]
        chk["v2_gate"] = {"hard_pass": hard, "note": "hinge/jitter are corrected downstream (torso clamp + smoothing); full body + >=2 reps are hard gates"}
        jdump(chk, out.replace(".npz", "_check.json"))
        if not hard: raise SystemExit("source check failed (full body / reps): " + json.dumps(chk)[:300])
        b = chk["reps"]["bottoms_f"]; i = int(self.cfg["motion"].get("rep_index", 0))
        if len(b) < i + 2: raise SystemExit("not enough reps for rep_index")
        return {"rep": [b[i], b[i + 1]], "working": chk["working_side"], "source_check": {k: chk[k]["pass"] for k in ("full_body", "hinge", "reps", "jitter")}}

    # 4/5 ----------------------------------------------------------------------------------------------------------------
    def s_render(self, rep):
        gate(self.args, "render"); d = self.p("render"); os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(".png"): os.remove(os.path.join(d, f))   # stale frames of a previous config (own scratch output)
        rc = dict(self.cfg); rc["_rep"] = rep; rc["asset"] = dict(rc["asset"]); rc["asset"]["glb"] = os.path.join(ROOT, rc["asset"]["glb"])
        rc["asset"]["hdri"] = dict(rc["asset"]["hdri"]); rc["asset"]["hdri"]["file"] = os.path.join(ROOT, rc["asset"]["hdri"]["file"]); jdump(rc, os.path.join(d, "cfg.json"))
        frames = "all" if not self.args.short else ",".join(str(i) for i in range(1, self.args.short + 1))
        t = time.time()
        peak = run_child([BPY, os.path.join(HERE, "retarget_render_v2.py"), "--npz", self.p("pose", "pose.npz"), "--cfg", os.path.join(d, "cfg.json"), "--out", d, "--frames", frames], logf=os.path.join(d, "log.txt"))
        rt = json.load(open(os.path.join(d, "render_time.json"))); rt.update({"wall_s": round(time.time() - t, 1), "peak_rss_mb": peak})
        jdump(rt, os.path.join(d, "render_time.json")); return rt

    # 6 ------------------------------------------------------------------------------------------------------------------
    def s_grade(self):
        gate(self.args, "grade"); d = self.p("grade")
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".png"): os.remove(os.path.join(d, f))
        jdump(self.cfg, self.p("cfg.json"))
        peak = run_child(["python3", os.path.join(HERE, "grade3d.py"), self.p("render"), d, self.p("cfg.json")], logf=self.p("grade_log.txt"))
        return {"peak_rss_mb": peak, "frames": len([f for f in os.listdir(d) if f.startswith("g_")])}

    # 7 ------------------------------------------------------------------------------------------------------------------
    def tts(self, d):
        from gymform.tts import VOICE, CLEAN
        from gymform import caption
        import edge_tts
        T = self.cfg["tts"]; texts = [T["line"], T.get("question_spoken") or caption.SPOKEN_DEFAULT]
        cache = os.path.join(ROOT, "_cache_rig3d", "tts"); os.makedirs(cache, exist_ok=True); wavs = []
        for tx in texts:
            k = hsh(tx, VOICE, CLEAN); wp = os.path.join(cache, k + ".wav")
            if not os.path.exists(wp):
                mp3 = wp[:-4] + ".mp3"; asyncio.run(edge_tts.Communicate(tx, **VOICE).save(mp3))
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-threads", "2", "-i", mp3, "-af", CLEAN, "-ac", "1", "-ar", "44100", wp], check=True)
            with wave.open(wp, "rb") as wf: wavs.append(np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768.0)
        segs, t0 = [], 0.25
        for tx, w in zip(texts, wavs):
            segs.append({"text": tx, "start": round(t0, 3), "dur": round(len(w) / 44100, 3)}); t0 += len(w) / 44100 + 0.3
        return segs, wavs
    def s_reel(self):
        from gymform import caption
        gate(self.args, "reel"); d = self.p("reel"); os.makedirs(d, exist_ok=True)
        cap = caption.build(self.cfg["caption"]); errs = caption.lint(cap); open(os.path.join(d, "caption.txt"), "w").write(cap)
        jdump({"errors": errs}, os.path.join(d, "caption_lint.json"))
        segs, wavs = self.tts(d); end = segs[-1]["start"] + segs[-1]["dur"]
        G = sorted(f for f in os.listdir(self.p("grade")) if f.startswith("g_")); fps_src = float(json.load(open(self.p("pose", "pose_check.json")))["fps"])
        rep_s = len(G) / fps_src; loops = max(int(self.cfg["grade"].get("loops", 2)), math.ceil((end + 0.6) / rep_s)); dur = loops * rep_s
        sr = 44100; n = int(sr * dur); t = np.arange(n, dtype=np.float32) / sr
        beat = 0.45 + 0.55 * (np.sin(2 * math.pi * (100 / 60) * t) > 0.72).astype(np.float32); mix = 0.04 * np.sin(2 * math.pi * 55 * t) * beat
        dc = int(0.24 * sr); tc = np.linspace(0, 0.24, dc, endpoint=False); mix[int(0.05 * sr):int(0.05 * sr) + dc] += 0.085 * np.sin(2 * math.pi * 528 * tc) * np.exp(-tc * 13)
        for s, w in zip(segs, wavs):
            a = int(s["start"] * sr); b = min(n, a + len(w)); mix[a:b] += w[:b - a] * 0.92
        wav = os.path.join(d, "audio.wav")
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(np.repeat((np.clip(mix, -0.96, 0.96) * 32767).astype(np.int16), 2).tobytes())
        jdump({"segments": segs, "end_s": round(end, 3), "loops": loops, "dur_s": round(dur, 3), "voice": "en-US-ChristopherNeural"}, os.path.join(d, "tts.json"))
        seq = os.path.join(d, "seq"); shutil.rmtree(seq, ignore_errors=True); os.makedirs(seq)
        k = 0
        for _ in range(loops):
            for g in G: os.symlink(self.p("grade", g), os.path.join(seq, f"s_{k:05d}.png")); k += 1
        mp4 = os.path.join(d, f"{self.slug}_rig3d_v2.mp4")
        run_child(["ffmpeg", "-loglevel", "error", "-y", "-threads", "2", "-framerate", str(fps_src), "-i", os.path.join(seq, "s_%05d.png"), "-i", wav,
                   "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-threads", "2", "-c:a", "aac", "-b:a", "160k",
                   "-shortest", "-movflags", "+faststart", mp4], logf=os.path.join(d, "ffmpeg.log"))
        cover = os.path.join(d, "cover.jpg"); shutil.copy2(self.p("grade", G[len(G) // 2]), cover.replace(".jpg", ".png"))
        return {"mp4": mp4, "dur_s": round(dur, 2), "loops": loops, "caption_errors": errs}

    def rig_precheck(self, rep):
        """retarget --checks-only into render_pre/ + qc3d.rig_checks (no render, no grade)."""
        d = self.p("render_pre"); os.makedirs(d, exist_ok=True)
        rc = dict(self.cfg); rc["_rep"] = rep; rc["asset"] = dict(rc["asset"]); rc["asset"]["glb"] = os.path.join(ROOT, rc["asset"]["glb"])
        rc["asset"]["hdri"] = dict(rc["asset"]["hdri"]); rc["asset"]["hdri"]["file"] = os.path.join(ROOT, rc["asset"]["hdri"]["file"]); jdump(rc, os.path.join(d, "cfg.json"))
        run_child([BPY, os.path.join(HERE, "retarget_render_v2.py"), "--npz", self.p("pose", "pose.npz"), "--cfg", os.path.join(d, "cfg.json"), "--out", d, "--checks-only"], logf=os.path.join(d, "log.txt"))
        import importlib; sys.path.insert(0, HERE); import qc3d; importlib.reload(qc3d)
        out = qc3d.apply_rule_map(qc3d.rig_checks(d, self.cfg)[0])
        res = {"pass": all(c["pass"] for c in out), "checks": out, "auto_fixable": sorted({c["fix"] for c in out if not c["pass"] and c["fix"]})}
        log(f"rig pre-check: {sum(c['pass'] for c in out)}/{len(out)}" + ("" if res["pass"] else f" fail {[c['name'] for c in out if not c['pass']]}"))
        jdump(res, os.path.join(d, "precheck.json")); return res

    # 8 ------------------------------------------------------------------------------------------------------------------
    def s_qc(self):
        import importlib; sys.path.insert(0, HERE); import qc3d; importlib.reload(qc3d)
        return qc3d.run(self.work, self.cfg)

RULES = {   # qc fix name -> (config path, function old_value -> new value). v6: a rule may be applied again in a later
            # iteration (it steps the value further); a rule whose value no longer changes is exhausted.
    "torso_smooth": ("motion.torso_smooth_sigma_f", lambda v: float(v) * 2),
    "grip_offset": ("motion.grip_offset_m", lambda v: 0.06),
    "spine_spread": ("motion.spine_influence", lambda v: max(0.3, float(v) - 0.2)),
    "camera_margin": ("render.camera.fill_w", lambda v: float(v) - 0.08),
    "camera_tighter": ("render.camera.fill_w", lambda v: min(0.95, float(v) + 0.06)),
    "neck_follow_spine": ("motion.neck_mode", lambda v: "spine_follow"),
    "plant_legs": ("motion.legs", lambda v: "planted"),
    "balance_knees": ("motion.balance_knees", lambda v: True),
    "grip_palm": ("motion.grip_mode", lambda v: "palm_neutral"),
    "support_hand": ("motion.support_hand", lambda v: "front_thigh"),
    "camera_top_margin": ("render.camera.top_y", lambda v: min(0.32, float(v or 0.24) + 0.02)),
    # v6
    "grip_wrap": ("motion.grip_wrap.mcp_back_m", lambda v: min(0.025, float(v or 0.015) + 0.005)),
    "limb_clearance": ("motion.support_hand_lift_m", lambda v: min(0.13, float(v or 0.075) + 0.02)),
    "joint_range": ("motion.support_elbow_out", lambda v: min(1.5, float(v if v is not None else 0.5) + 0.5)),
    "com_balance": ("motion.stance_mode", lambda v: "feet_forward"),
    "camera_yaw": ("render.camera.yaw_back_deg", lambda v: max(0.0, float(v or 35) - 10)),
    "glow_cap": ("render.camera.yaw_back_deg", lambda v: max(0.0, float(v or 35) - 10)),
    "camera_bigger": ("render.camera.feet_y", lambda v: min(0.95, float(v or 0.9) + 0.015)),
    "torso_target": ("motion.torso_target_deg", lambda v: 47.0),
}

def getp(c, path):
    for k in path.split("."): c = c.get(k) if isinstance(c, dict) else None
    return c

def enqueue_ai(run, qc, meta):
    os.makedirs(os.path.dirname(AIQ), exist_ok=True)
    iid = f"gymform3d:rig3d:{run.slug}:{hsh(sorted(qc['needs_ai'] + [c['name'] for c in qc['checks'] if not c['pass']]))}"
    have = set()
    if os.path.exists(AIQ):
        for line in open(AIQ):
            try: have.add(json.loads(line)["id"])
            except ValueError: pass
    if iid in have: log(f"ai_queue: {iid} already queued"); return iid
    fails = {c["name"]: {"value": c["value"], "limit": c["limit"], "rule": c["fix"]} for c in qc["checks"] if not c["pass"]}
    item = {"id": iid, "bot": "fitness_studio", "agent_id": FITNESS_STUDIO_AGENT, "channel": "gymform3d", "topic_id": run.slug, "step": "rig3d_qc_review",
            "needs": sorted(fails), "why": fails, "rules_tried": meta.get("fixes_applied", []),
            "paths": {"sheet": meta.get("sheet"), "qc": os.path.join(run.work, "qc.json"), "reel": meta.get("reel"), "config": run.cfg_path},
            "brief": f"rig3d {run.slug}: QC failed after rule retries ({', '.join(sorted(fails))}). Look at the sheet, fix config/code, then re-run: nice -n 19 python3 tools/rig3d/pipeline.py --exercise {run.slug}",
            "created_at": dt.datetime.now(KST).isoformat(timespec="seconds"), "done_marker": f"qc.json pass in {run.work}", "status": "pending"}
    with open(AIQ, "a", encoding="utf-8") as f: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"ai_queue: queued {iid} -> {AIQ}"); return iid

def resolve_ai(slug, run_id, qc):
    """QC passed: mark this exercise's pending Fitness Studio items done (rewritten in place, never deleted)."""
    if not os.path.exists(AIQ): return []
    lines, done = open(AIQ, encoding="utf-8").read().splitlines(), []
    for i, line in enumerate(lines):
        try: it = json.loads(line)
        except ValueError: continue
        if it.get("topic_id") == slug and it.get("step") == "rig3d_qc_review" and it.get("status") == "pending":
            it.update({"status": "done", "done_at": dt.datetime.now(KST).isoformat(timespec="seconds"),
                       "resolution": f"auto: rig3d run {run_id} QC PASS {qc['n_pass']}/{qc['n']}"}); lines[i] = json.dumps(it, ensure_ascii=False); done.append(it["id"])
    if done:
        tmp = AIQ + ".tmp"; open(tmp, "w", encoding="utf-8").write("\n".join(lines) + "\n"); os.replace(tmp, AIQ)
    return done

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exercise", required=True); ap.add_argument("--force", choices=STAGES)
    ap.add_argument("--short", type=int, default=0); ap.add_argument("--deliver-dry-run", action="store_true"); ap.add_argument("--no-email", action="store_true", help="real repo commit + alert, but build the owner e-mail without sending it (one e-mail per run budget)"); ap.add_argument("--no-gate", action="store_true"); ap.add_argument("--mail-direct", action="store_true", help="send the owner e-mail now via the Gmail API instead of the batched digest")
    ap.add_argument("--gate-timeout", type=int, default=6 * 3600); ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--max-iter", type=int, default=3, help="self-healing iterations (rig pre-check loop and full render loop, each)")
    args = ap.parse_args()
    try: os.nice(max(0, 19 - os.nice(0)))
    except OSError: pass
    if "xai_edit" in sys.modules: raise SystemExit("xai module loaded: refused")
    T0 = time.time(); run = Run(args); run_id = dt.datetime.now(KST).strftime("%Y%m%d_%H%M")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    fixes = []; max_iter = args.max_iter
    def apply_rules(qc_, it, phase):
        did = []
        for f in qc_["auto_fixable"]:
            if f not in RULES: continue
            path, fn = RULES[f]
            if path in [d["path"] for d in did]: continue                       # one step per path per iteration
            old = run.overrides.get(path, getp(run.cfg, path)); new = fn(old)
            if new == old: continue                                              # exhausted
            run.overrides[path] = new; d = {"iter": it, "phase": phase, "rule": f, "path": path, "old": old, "new": new,
                                            "for": [c["name"] for c in qc_["checks"] if not c["pass"] and c.get("fix") == f]}
            did.append(d); fixes.append(d); log(f"auto-fix [{phase} {it}] {f}: {path} {old} -> {new} (for {', '.join(d['for'])})")
        return did
    iters = {"rig_precheck": 0, "full": 0}
    for attempt in range(max_iter + 1):
        cfg = run.stage("config", [open(run.cfg_path).read(), run.overrides], run.s_config, [])
        run.cfg = cfg
        run.stage("fetch", [cfg["source"], cfg["asset"]], run.s_fetch, [os.path.join(ROOT, cfg["source"]["file"]), os.path.join(ROOT, cfg["asset"]["hdri"]["file"])])
        pinfo = run.stage("pose", [cfg["source"]["file"], fsha(cfg["source"]["file"])], run.s_pose, [run.p("pose", "pose.npz")])
        # v6 cheap rig pre-check loop (retarget --checks-only, ~5 s): fix every rig/posture/anatomy/grip/prop failure
        # with its rule BEFORE paying for a full render
        if not args.no_retry:
            for pit in range(max_iter):
                pq = run.rig_precheck(pinfo["rep"])
                if pq["pass"]: break
                if not apply_rules(pq, pit + 1, "rig_precheck"): break
                iters["rig_precheck"] += 1
                cfg = run.stage("config", [open(run.cfg_path).read(), run.overrides], run.s_config, []); run.cfg = cfg
        rinfo = run.stage("render", [cfg["motion"], cfg["render"], cfg["asset"], pinfo["rep"], args.short], lambda: run.s_render(pinfo["rep"]), [run.p("render", "checks.json"), run.p("render", "render_time.json")])
        if args.short:
            log(f"short probe done: {json.dumps(rinfo)}"); return
        run.stage("grade", [cfg["grade"], cfg["name"]], run.s_grade, [run.p("grade", "look.json")])
        reel = run.stage("reel", [cfg["tts"], cfg["caption"], cfg["grade"]], run.s_reel, [run.p("reel", f"{run.slug}_rig3d_v2.mp4")])
        qc = run.stage("qc", [cfg["qc"]], run.s_qc, [run.p("qc.json")])
        if qc["pass"] or args.no_retry or attempt == max_iter: break
        if not apply_rules(qc, attempt + 1, "full"): break
        iters["full"] += 1
    meta = {"run_id": run_id, "slug": run.slug, "code_commit": commit, "stages": run.times, "fixes_applied": fixes, "rep": pinfo["rep"],
            "render": rinfo, "reel": reel["mp4"], "sheet": run.p("sheet.png"), "qc_pass": qc["pass"],
            "auto_iterations": iters, "max_iter": max_iter, "needs_human": [c["name"] for c in qc["checks"] if not c["pass"] and not c.get("fix")]}
    def production():   # real cost of producing THIS output = latest computed run of every stage (cached re-runs don't reset it)
        st = [json.load(open(os.path.join(run.stamps, f + ".json"))) for f in STAGES if os.path.exists(os.path.join(run.stamps, f + ".json"))]
        return {"chain_s": round(sum(x.get("s") or 0 for x in st), 1), "peak_rss_mb": max([x.get("peak_rss_mb") or 0 for x in st] + [rinfo.get("peak_rss_mb") or 0]),
                "render_s": rinfo.get("wall_s"), "render_peak_rss_mb": rinfo.get("peak_rss_mb")}
    meta["production"] = production(); meta["wall_s_this_invocation"] = round(time.time() - T0, 1)
    meta["wall_s"] = meta["production"]["chain_s"]; meta["peak_rss_mb"] = meta["production"]["peak_rss_mb"]
    jdump(meta, run.p("run.json"))
    sys.path.insert(0, HERE); import sheet
    run.stage("sheet", [run_id, qc], lambda: sheet.build(run.work, cfg, run.p("sheet.png")), [run.p("sheet.png")])
    if not qc["pass"]:
        meta["ai_queue"] = enqueue_ai(run, qc, meta)
    else:
        meta["ai_queue_resolved"] = resolve_ai(run.slug, run_id, qc)
    meta["production"] = production(); meta["wall_s"] = meta["production"]["chain_s"]; meta["wall_s_this_invocation"] = round(time.time() - T0, 1); jdump(meta, run.p("run.json"))
    import deliver
    cfg_d = dict(cfg); cfg_d["_no_email"] = bool(args.no_email); cfg_d["_mail_direct"] = bool(args.mail_direct) and not args.no_email; jdump(cfg_d, run.p("cfg.json"))
    res = [deliver.backend_repo(run.work, cfg_d, args.deliver_dry_run)]
    url = res[0].get("url")
    res.append(deliver.backend_ntfy(run.work, cfg_d, args.deliver_dry_run, url)); res.append(deliver.backend_smtp(run.work, cfg_d, args.deliver_dry_run or args.no_email, url))
    meta["delivery"] = res; jdump(meta, run.p("run.json"))
    if not qc["pass"]:
        deliver.alert("rig3d_needs_ai", f"{run.slug} rig3d QC FAIL -> AI queue {meta.get('ai_queue')}", {"slug": run.slug, "run_id": run_id}, args.deliver_dry_run, severity="warn")
    log(f"DONE qc={'PASS' if qc['pass'] else 'FAIL'} {qc['n_pass']}/{qc['n']} chain {meta['wall_s']} s (this invocation {meta['wall_s_this_invocation']} s) peak RSS {meta['peak_rss_mb']} MB sheet {meta['sheet']}")
    print(json.dumps({"qc_pass": qc["pass"], "sheet": meta["sheet"], "delivery": res}, ensure_ascii=False))

if __name__ == "__main__":
    main()
