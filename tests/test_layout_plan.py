"""t552u/t554u/t556u: target bones, mistake-type step layout, fixed front/side camera plan.
Run: python3 -m unittest tests.test_layout_plan"""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import bones, ctx  # noqa: E402


def day6():
    return json.load(open(os.path.join(ROOT, "configs", "day06.json")))


def camera_cfg(mt):
    """A future day: 2 motions x front/side native renders."""
    c = day6()
    c["day"], c["mistake_type"] = 7, mt
    for f in ("wrong", "right"):
        m = c["motions"][f]
        m["renders"] = {"front": {"file": f"{f}_front.mp4", "camera": "front"},
                        "side": {"file": f"{f}_side.mp4", "camera": "side", "pair_match": {"joint_dev_pct": 0, "elbow_deg": 0,
                                 "shoulder_deg": 0, "view_dev_pct": 0}, "i0": 2, "i1": 60}}
        m["base"] = f"{f}_front.mp4"
    return c


class Plan(unittest.TestCase):
    def test_every_exercise_has_type_bones_cameras(self):
        ex = json.load(open(os.path.join(ROOT, "configs", "motion_standards.json")))["exercises"]
        for k, e in ex.items():
            self.assertIn(e["mistake_type"], ("joint", "muscle"), k)
            self.assertTrue(e["target_bones"], k)
            self.assertEqual(set(e["camera_presets"]), {"front", "side"}, k)
            bones.region({j: (500.0 + 10 * i, 900.0 + 40 * i) for i, j in enumerate(
                ["sh", "sh_r", "sh_l", "elb_r", "elb_l", "wr_r", "wr_l", "hip_r", "hip_l", "kn_r", "kn_l", "ank_r", "ank_l"])},
                e["target_bones"], (64, 64))

    def test_joint_day_plan(self):
        c = ctx.expand_motions(camera_cfg("joint"))
        self.assertEqual((c["bone_steps"], c["muscle_steps"]), ([1, 2], [3, 4]))
        self.assertEqual([c["clips"][s]["camera"] for s in "1234"], ["front", "front", "side", "side"])
        self.assertEqual(ctx.order_lint(c), [])
        self.assertEqual(ctx.camera_plan_lint(camera_cfg("joint")), [])

    def test_muscle_day_plan(self):
        c = ctx.expand_motions(camera_cfg("muscle"))
        self.assertEqual((c["bone_steps"], c["muscle_steps"]), ([], [1, 2, 3, 4]))
        self.assertEqual([c["clips"][s]["mode"] for s in "1234"], ["muscle_blood"] * 4)
        self.assertEqual([c["clips"][s]["is_wrong"] for s in "1234"], [True, False, True, False])
        self.assertEqual(ctx.order_lint(c), [])

    def test_day6_is_joint_and_flags_camera_plan(self):
        self.assertEqual(ctx.mistake_type(day6()), "joint")
        self.assertEqual(len(ctx.camera_plan_lint(day6())), 4)   # one 3/4 camera for all clips (pre-t556u)

    def test_target_region_is_local(self):
        kp = json.load(open(os.path.join(ROOT, "tests", "data", "day6_step1_peak_kp.json")))
        r = bones.region(kp, ["humerus", "scapula", "clavicle", "sternum", "upper_ribs"], (1920, 1080))
        self.assertGreater(r[int(kp["sh_r"][1]), int(kp["sh_r"][0])], 0.9)
        self.assertLess(r[int(kp["kn_r"][1]), int(kp["kn_r"][0])], 0.01)     # no legs
        self.assertLess(r[int(kp["wr_r"][1]), int(kp["wr_r"][0])], 0.01)     # no forearm/hand
        self.assertLess(float((r > 0.5).mean()), 0.15)


if __name__ == "__main__":
    unittest.main()
