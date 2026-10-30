"""(u) bone visibility QC (t530u): Day 1 v36 passes, a dark saturated fill fails.
Run: python3 -m unittest tests.test_bone_visibility"""
import os, sys, unittest
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import qc  # noqa: E402


class BoneVisibility(unittest.TestCase):
    def frame(self):
        p = os.path.join(ROOT, "output", "day1_v36_master_reel.mp4")
        if not os.path.exists(p):
            self.skipTest("Day 1 v36 reel missing")
        c = cv2.VideoCapture(p); c.set(cv2.CAP_PROP_POS_MSEC, 1500); _, x = c.read()
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    def test_day1_v36_passes(self):
        f = self.frame(); r = np.zeros(f.shape[:2], bool); r[600:1500, 600:1000] = True
        self.assertTrue(qc.bone_visible_ok(qc.bone_visibility(f, r)))

    def test_dark_saturated_bones_fail(self):
        f = self.frame(); r = np.zeros(f.shape[:2], bool); r[600:1500, 600:1000] = True
        g = f.copy(); L = f.astype(np.float32).mean(-1); g[(L > 140) & r] = (30, 158, 90)
        self.assertFalse(qc.bone_visible_ok(qc.bone_visibility(g, r)))


if __name__ == "__main__":
    unittest.main()
