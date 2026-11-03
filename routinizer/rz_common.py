"""Shared helpers for Routinizer watchers (stdlib only, no secrets).

Runtime files (outside git):
  /home/box/routinizer/alerts.jsonl          structured "wake AI" alerts (append-only)
  /home/box/routinizer/state/<name>.json     per-watcher state (dedupe, history)
  /home/box/routinizer/state/<name>.heartbeat.json
  /home/box/routinizer/locks/<name>.lock     single-instance lock
"""
import datetime as _dt
import fcntl
import json
import os
import tempfile
import time

ROOT = os.environ.get("ROUTINIZER_HOME", "/home/box/routinizer")
ALERTS = os.path.join(ROOT, "alerts.jsonl")
KST = _dt.timezone(_dt.timedelta(hours=9))

# processes that mean "a heavy Shorts render is running" (Shorts work has top box priority)
HEAVY_PATTERNS = ("flipcut.py", "geo_template/", "blender", "ffmpeg")


def now():
    return _dt.datetime.now(_dt.timezone.utc)


def iso(t=None):
    return (t or now()).astimezone(KST).isoformat(timespec="seconds")


def parse_time(s):
    """Parse '2026-11-02T01:00:00Z', '2026-10-30T10:00+09:00', '2026-10-30T10:00:00+09:00'."""
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    t = _dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=KST)
    return t


def log(name, *a):
    print(f"[{iso()}] [{name}]", *a, flush=True)


def atomic_write_json(path, data):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".rz_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.write("\n")
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o777)
        except FileNotFoundError:
            pass
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class State:
    def __init__(self, name):
        os.makedirs(os.path.join(ROOT, "state"), exist_ok=True)
        self.name = name
        self.path = os.path.join(ROOT, "state", f"{name}.json")
        self.hb = os.path.join(ROOT, "state", f"{name}.heartbeat.json")
        try:
            self.data = json.load(open(self.path, encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            self.data = {}
        self.data.setdefault("alerts_sent", {})
        self.data.setdefault("history", [])

    def save(self):
        self.data["history"] = self.data["history"][-200:]
        atomic_write_json(self.path, self.data)

    def heartbeat(self, **kw):
        atomic_write_json(self.hb, dict(watcher=self.name, at=iso(), pid=os.getpid(), **kw))


def alert(state, kind, key, message, severity="warn", repeat_hours=6, dry_run=False, **extra):
    """Append a structured alert once per (kind,key) per repeat_hours. Never emails/messages anyone."""
    k = f"{kind}:{key}"
    last = state.data["alerts_sent"].get(k)
    if last and (time.time() - last) < repeat_hours * 3600:
        return False
    rec = dict(at=iso(), watcher=state.name, kind=kind, key=key, severity=severity,
               message=message, dry_run=dry_run, **extra)
    if dry_run:
        log(state.name, "ALERT(dry-run, not written):", json.dumps(rec, ensure_ascii=False))
        return True
    os.makedirs(ROOT, exist_ok=True)
    with open(ALERTS, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    state.data["alerts_sent"][k] = time.time()
    log(state.name, "ALERT:", kind, key, message)
    return True


class SingleInstance:
    """Non-blocking exclusive lock; a second copy of the same watcher exits."""

    def __init__(self, name):
        os.makedirs(os.path.join(ROOT, "locks"), exist_ok=True)
        self.path = os.path.join(ROOT, "locks", f"{name}.lock")
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "a+")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another instance holds {self.path}")
        self.f.seek(0); self.f.truncate(); self.f.write(str(os.getpid())); self.f.flush()
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def heavy_processes():
    me = os.getpid()
    hits = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "routinizer" in cmd:
            continue
        if any(p in cmd for p in HEAVY_PATTERNS):
            hits.append(f"{pid}:{cmd[:80]}")
    return hits


def box_busy():
    """(busy, reason). Busy = 1-min loadavg > cores, or a heavy render process is running."""
    load1 = os.getloadavg()[0]
    cores = os.cpu_count() or 1
    heavy = heavy_processes()
    if load1 > cores:
        return True, f"load {load1:.2f} > {cores} cores"
    if heavy:
        return True, f"heavy render running ({heavy[0]})"
    return False, f"load {load1:.2f}/{cores}"
