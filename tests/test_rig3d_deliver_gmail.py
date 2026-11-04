"""Offline tests: rig3d delivery prefers the Gmail API (rz_notify_mail) and falls back to SMTP. No network, nothing sent."""
import json, os, sys, tempfile, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools/rig3d"))
import deliver


def run_dir():
    d = tempfile.mkdtemp(); sheet = os.path.join(d, "sheet.png"); open(sheet, "wb").write(b"\x89PNG fake")
    json.dump({"pass": False, "n_pass": 21, "n": 22, "checks": [{"name": "sharpness_vs_day4", "value": 0.012, "limit": ">= 0.35", "pass": False},
                                                              {"name": "arm_crossing_frames", "value": 0, "limit": 0, "pass": True}]}, open(os.path.join(d, "qc.json"), "w"))
    json.dump({"slug": "db_row", "run_id": "T1", "sheet": sheet, "reel": "/x/reel.mp4", "wall_s": 1}, open(os.path.join(d, "run.json"), "w"))
    return d, sheet


def fake_mailer(result):
    m = types.ModuleType("rz_notify_mail"); m.calls = []
    def send(subject, body, attachments=(), dry_run=False):
        m.calls.append((subject, body, list(attachments), dry_run)); return dict(result, dry_run=dry_run) if dry_run else dict(result)
    m.send = send; sys.modules["rz_notify_mail"] = m; return m


def test_gmail_preferred_and_no_smtp_when_sent():
    d, sheet = run_dir(); m = fake_mailer({"backend": "gmail_api", "sent": True, "id": "x"})
    deliver._smtp = (lambda *a, **k: (_ for _ in ()).throw(AssertionError("smtp must not run")))
    r = deliver.backend_smtp(d, {}, False, "https://github.com/been2531/gymform3d-archive/blob/main/results/rig3d/db_row/T1")
    assert r["backend"] == "gmail_api" and r["sent"]
    subj, body, att, dry = m.calls[0]
    assert "QC FAIL 21/22" in subj and "sharpness_vs_day4 (0.012, limit >= 0.35)" in body and att[0] == sheet and dry is False


def test_falls_back_to_smtp_when_token_missing():
    d, _ = run_dir(); fake_mailer({"backend": "gmail_api", "sent": False, "error": "gmail_token_missing", "alerted": True})
    deliver._smtp = (lambda *a, **k: {"backend": "smtp", "skipped": "env not set"})
    r = deliver.backend_smtp(d, {}, False, None)
    assert r["backend"] == "smtp" and r["gmail_api"]["error"] == "gmail_token_missing"


def test_dry_run_passes_through():
    d, _ = run_dir(); m = fake_mailer({"backend": "gmail_api"})
    r = deliver.backend_smtp(d, {}, True, None); assert r["dry_run"] and m.calls[0][3] is True


def test_mailer_exception_never_breaks_delivery():
    d, _ = run_dir(); m = fake_mailer({}); m.send = lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    deliver._smtp = (lambda *a, **k: {"backend": "smtp", "skipped": "env not set"})
    r = deliver.backend_smtp(d, {}, False, None); assert r["gmail_api"]["sent"] is False and "boom" in r["gmail_api"]["error"]


if __name__ == "__main__":
    orig = deliver._smtp
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); deliver._smtp = orig
    print("test_rig3d_deliver_gmail OK")
