"""Paths, KST time, state, logging, Telegram, box-yield. No secrets here."""
from __future__ import annotations
import datetime as dt
import json
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/workspace/venv-pilots/bin/python"            # skia/numpy venv used for render + validate
QUEUE = os.path.join(REPO, "topics", "queue.json")
WORK = os.environ.get("AUTOREEL_WORK") or os.path.join(REPO, "output", "autoreel")  # not committed; env override for tests
STATE = os.path.join(WORK, "state.json")
LOGDIR = os.path.join(WORK, "logs")
APPROVALS = os.path.join(REPO, "approvals")
NOTIFY = "/home/box/routinizer/notify_telegram.py"
KST = dt.timezone(dt.timedelta(hours=9))
NEVER_REPOST = ("DfGabTBjwt8",)


def now():
    return dt.datetime.now(KST)


def today():
    return now().strftime("%Y-%m-%d")


def iso(t=None):
    return (t or now()).isoformat(timespec="seconds")


def log(name, *msg):
    os.makedirs(LOGDIR, exist_ok=True)
    line = f"{iso()} [{name}] " + " ".join(str(m) for m in msg)
    print(line, flush=True)
    with open(os.path.join(LOGDIR, f"{today()}.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return default


def write_json(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def state():
    return load_json(STATE, {"days": {}, "used": {}})


NOTIFY_DRY = False  # set by publish_run --dry-run / tests: notifier runs with --dry-run (nothing sent)
BOT = "GymForm/Fitness Studio"


def notify(status, title, result="", link="", name="autoreel"):
    """Telegram via the shared Routinizer notifier (owner wording: short Korean, no paths/logs).
    status: "✅" | "❌". Skips (and logs) if the notifier is missing."""
    if not os.path.isfile(NOTIFY):
        log(name, "telegram skipped: notifier script missing")
        return False
    st = {"✅": "✅ 완료", "❌": "❌ 실패"}.get(status, status)
    cmd = ["python3", NOTIFY, "--status", st, "--title", title, "--bot", BOT, "--task", "gymform.3d 자동 릴스",
           "--result", result, "--immediate"]
    if link:
        cmd += ["--link", link]
    if NOTIFY_DRY:
        cmd.append("--dry-run")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        log(name, ("telegram dry-run ok" if NOTIFY_DRY else "telegram sent") if r.returncode == 0 else f"telegram failed rc={r.returncode}")
        return r.returncode == 0
    except Exception as e:  # noqa: BLE001
        log(name, f"telegram error {type(e).__name__}")
        return False


def wait_for_box(name, max_wait_s=1200):
    """Yield to bots the user is actively using: wait while the box is saturated."""
    ncpu = os.cpu_count() or 1
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        la = os.getloadavg()[0]
        if la < ncpu * 0.9:
            return True
        log(name, f"box busy (load {la:.1f}/{ncpu}), waiting")
        time.sleep(60)
    log(name, "box still busy after wait; continuing at nice 19")
    return False


class Lock:
    def __init__(self, name):
        self.p = os.path.join(WORK, f".{name}.lock")

    def __enter__(self):
        os.makedirs(WORK, exist_ok=True)
        try:
            self.fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode())
        except FileExistsError:
            pid = open(self.p).read().strip()
            if pid.isdigit() and os.path.exists(f"/proc/{pid}"):
                raise SystemExit(f"another run is active (pid {pid})")
            os.remove(self.p)
            return self.__enter__()
        return self

    def __exit__(self, *a):
        os.close(self.fd); os.remove(self.p)
