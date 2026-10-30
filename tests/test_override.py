"""One-time owner override: only named checks, only one day + exact hash, only once.
Run: python3 -m unittest tests.test_override"""
import json, os, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import preflight as pf  # noqa: E402


class Override(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); self.old = pf.OVR_DIR; pf.OVR_DIR = self.d
        self.cfg = os.path.join(ROOT, "configs", "day06.json")
        self.o = {"day": 6, "authorized_by": "tX", "reason": "r", "waive": ["(q)"], "stamp_key": pf.stamp_key(self.cfg)}
        json.dump(self.o, open(os.path.join(self.d, "day06_tX.json"), "w"))

    def tearDown(self):
        pf.OVR_DIR = self.old

    def test_waives_only_named_check(self):
        o = pf.load_override(6, self.cfg)
        rest, w = pf.waive(["(q) stray", "(d) hole", "(s) iou"], o)
        self.assertEqual(w, ["(q) stray"]); self.assertEqual(rest, ["(d) hole", "(s) iou"])

    def test_other_day_or_hash_or_used_never_applies(self):
        self.assertIsNone(pf.load_override(7, self.cfg))
        self.assertIsNone(pf.load_override(6, os.path.join(ROOT, "configs", "day05.json")))
        o = pf.load_override(6, self.cfg); pf.mark_used(o, "test")
        self.assertIsNone(pf.load_override(6, self.cfg))

    def test_no_override_waives_nothing(self):
        self.assertEqual(pf.waive(["(q) x"], None), (["(q) x"], []))


if __name__ == "__main__":
    unittest.main()
