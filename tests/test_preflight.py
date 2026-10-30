"""Preflight gate tests (RULES.md): no full render without a PASSING preflight stamp for the same hash.
Fast: nothing is rendered (the guard fires before any clip is loaded). Run: python3 -m unittest tests.test_preflight"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import preflight as pf  # noqa: E402


class PreflightGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf_test_", dir=os.path.join(ROOT, "_scratch") if os.path.isdir(os.path.join(ROOT, "_scratch")) else None)
        self.cfg = os.path.join(self.tmp, "day99.json")
        c = json.load(open(os.path.join(ROOT, "configs", "day06.json")))
        c["day"] = 99
        json.dump(c, open(self.cfg, "w"))
        self.ovr = os.path.join(self.tmp, "ovr.json")
        json.dump({}, open(self.ovr, "w"))
        self.stamps = []

    def tearDown(self):
        for p in self.stamps + [pf.stamp_path(self.cfg, self.ovr)]:
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _render(self):
        return subprocess.run([sys.executable, "-m", "gymform.render", "--config", self.cfg, "--overrides", self.ovr,
                               "--icons", os.path.join(self.tmp, "none.pkl"), "--render-step", "1",
                               "--out", os.path.join(self.tmp, "o.mp4"), "--peak", os.path.join(self.tmp, "p.png")],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def _stamp(self, ok):
        self.stamps.append(pf.write_stamp(self.cfg, self.ovr, {"ok": ok, "fails": [] if ok else ["(q) stray line"]}))

    def test_no_stamp_blocks_render(self):
        r = self._render()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("render_guard", r.stdout + r.stderr)
        self.assertIn("no preflight stamp", r.stdout + r.stderr)

    def test_failing_preflight_blocks_render(self):
        self._stamp(False)
        r = self._render()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("preflight FAILED", r.stdout + r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o.mp4.part.mp4")))

    def test_passing_stamp_is_accepted(self):
        self._stamp(True)
        self.assertTrue(pf.require(self.cfg, self.ovr)["ok"])

    def test_changed_config_invalidates_stamp(self):
        self._stamp(True)
        k0 = pf.stamp_key(self.cfg, self.ovr)
        c = json.load(open(self.cfg))
        c["target_zoom"]["zoom"] = 1.30
        json.dump(c, open(self.cfg, "w"))
        self.assertNotEqual(pf.stamp_key(self.cfg, self.ovr), k0)
        with self.assertRaises(SystemExit):
            pf.require(self.cfg, self.ovr)

    def test_changed_overrides_invalidate_stamp(self):
        self._stamp(True)
        json.dump({"autofix": {"x": 1}}, open(self.ovr, "w"))
        with self.assertRaises(SystemExit):
            pf.require(self.cfg, self.ovr)


if __name__ == "__main__":
    unittest.main()
