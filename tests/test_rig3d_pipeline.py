"""Offline tests for the rig3d v2 pipeline plumbing (no Blender, no network)."""
import json, os, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools/rig3d"))
import yaml
import pipeline, deliver, qc3d
from gymform import caption


def cfg():
    return yaml.safe_load(open(os.path.join(ROOT, "exercises/db_row.yaml")))


def test_config_has_all_stages():
    c = cfg()
    for k in ("slug", "name", "source", "asset", "motion", "render", "grade", "tts", "caption", "qc"):
        assert k in c
    assert c["motion"]["torso_target_deg"] == 45 and "lats" in c["grade"]["target_muscles"]


def test_caption_lints_clean():
    assert caption.lint(caption.build(cfg()["caption"])) == []


def test_rules_change_config_once():
    c = cfg()
    for name, (path, fn) in pipeline.RULES.items():
        old = pipeline.getp(c, path)
        assert old is not None or path.endswith("grip_offset_m"), path
        assert fn(old if old is not None else 0.075) != old


def test_hash_stable_and_sensitive():
    assert pipeline.hsh("a", {"x": 1}) == pipeline.hsh("a", {"x": 1})
    assert pipeline.hsh("a", {"x": 1}) != pipeline.hsh("a", {"x": 2})


def test_secret_scan_blocks_tokens():
    assert deliver.SECRET.search(b"token=ghp_" + b"A" * 36)
    assert not deliver.SECRET.search(b"plain sheet text")


def test_alert_dry_run_writes_nothing():
    old = deliver.ALERTS
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "alerts.jsonl"); deliver.ALERTS = p
        try:
            deliver.alert("k", "m", {"slug": "s", "run_id": "r"}, True)
            assert not os.path.exists(p)
            deliver.alert("k", "m", {"slug": "s", "run_id": "r"}, False, severity="warn")
            rec = json.loads(open(p).read().strip()); assert rec["severity"] == "warn" and rec["watcher"] == "gymform_rig3d"
        finally:
            deliver.ALERTS = old


def test_push_backends_skip_without_env():
    for k in ("GYMFORM_NTFY_TOPIC", "GYMFORM_SMTP_USER", "GYMFORM_SMTP_APP_PASSWORD", "GYMFORM_NOTIFY_TO"):
        os.environ.pop(k, None)
    assert "skipped" in deliver.backend_ntfy("/nonexistent", {}, True)
    assert "skipped" in deliver.backend_smtp("/nonexistent", {}, True)


def test_qc_check_record():
    c = qc3d.ck("x", 1, 0, False, "spine_spread"); assert c["fix"] == "spine_spread" and not c["pass"]
    assert qc3d.ck("x", 0, 0, True, "spine_spread")["fix"] is None


def test_no_publish_or_xai_in_pipeline_code():
    for f in ("pipeline.py", "deliver.py", "grade3d.py", "qc3d.py", "sheet.py", "retarget_render_v2.py"):
        s = open(os.path.join(ROOT, "tools/rig3d", f)).read()
        assert "import ig_publish" not in s and "xai_edit" not in s.replace('"xai_edit" in sys.modules', "")


class T(unittest.TestCase):
    pass


for _n, _f in list(globals().items()):
    if _n.startswith("test_") and callable(_f):
        setattr(T, _n, staticmethod(_f))

if __name__ == "__main__":
    unittest.main()
