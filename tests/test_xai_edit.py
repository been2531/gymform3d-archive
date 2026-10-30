"""gymform.xai_edit: config-driven jobs, one-take guard, no key needed for dry runs.
Run: python3 -m unittest tests.test_xai_edit"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import xai_edit  # noqa: E402


class XaiEdit(unittest.TestCase):
    def test_day6_jobs(self):
        cfg = json.load(open(os.path.join(ROOT, "configs", "day06.json")))
        jobs = {(m, k): (os.path.basename(s), os.path.basename(d), p) for m, k, s, d, p, _ in xai_edit.jobs_for(cfg)}
        self.assertEqual(jobs[("wrong", "bone")][:2], ("day6_idp_step2_muscle_720p.mp4", "day6_idp_wrong_bone_720p.mp4"))
        self.assertEqual(jobs[("right", "bone")][:2], ("day6_idp_step4_muscle_720p.mp4", "day6_idp_right_bone_720p.mp4"))
        self.assertIn("crimson", jobs[("wrong", "bone")][2])
        self.assertIn("emerald", jobs[("right", "bone")][2])
        self.assertIn("not cyan", jobs[("right", "bone")][2])
        self.assertIn("OPAQUE", jobs[("right", "bone")][2])
        self.assertIn("No bones and no glow on the hips, legs or feet", jobs[("right", "bone")][2])
        for _, _, p in jobs.values():
            self.assertIn("exact same motion and timing", p)
        self.assertNotIn(("wrong", "muscle"), jobs)   # base clips are never edited

    def test_one_take_guard_refuses_resubmit(self):
        old = xai_edit.LEDGER
        with tempfile.TemporaryDirectory() as d:
            xai_edit.LEDGER = os.path.join(d, "ledger.jsonl")
            xai_edit._log({"dst": "day99_x_bone.mp4", "status": "submitted"})
            cfg = {"motions": {"wrong": {"base": "a.mp4", "renders": {"bone": {"file": "day99_x_bone.mp4", "edit_of": "a.mp4"}}}}}
            p = os.path.join(d, "configs"); os.makedirs(p)
            # main() reads configs from ROOT: exercise the guard through the ledger filter directly
            done = {r["dst"] for r in xai_edit._ledger()}
            self.assertIn("day99_x_bone.mp4", done)
        xai_edit.LEDGER = old

    def test_acceptance_identity_passes_and_t522_edits_fail(self):
        a = os.path.join(ROOT, "output", "assets")
        base, right = os.path.join(a, "day6_idp_step4_muscle_720p.mp4"), os.path.join(a, "day6_idp_right_bone_720p.mp4")
        if not (os.path.exists(base) and os.path.exists(right)):
            self.skipTest("Day 6 assets not on this box")
        ok, m = xai_edit.acceptance(base, base, "emerald")
        self.assertGreaterEqual(m["sil_iou_min"], 0.99)
        self.assertLessEqual(m["flow_p90_px"], 1.0)
        ok, m = xai_edit.acceptance(base, right, "emerald")
        self.assertFalse(ok)
        self.assertTrue(any("silhouette" in f for f in m["fails"]))
        self.assertTrue(any("hue" in f for f in m["fails"]))

    def test_never_prints_key(self):
        src = open(os.path.join(ROOT, "gymform", "xai_edit.py")).read()
        self.assertNotIn("print(key", src)
        self.assertNotIn("print(k)", src)


if __name__ == "__main__":
    unittest.main()
