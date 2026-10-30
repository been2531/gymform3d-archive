"""t568u 2-step muscle-only day. Run: python3 -m unittest tests.test_two_step"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import ctx  # noqa: E402
from gymform.layout import callout_lint  # noqa: E402


class TwoStep(unittest.TestCase):
    def cfg(self):
        return json.load(open(os.path.join(ROOT, "configs", "day06_muscle2.json")))

    def test_plan_and_expand(self):
        c = ctx.expand_motions(self.cfg())
        self.assertEqual(sorted(c["clips"]), ["1", "2"])
        self.assertEqual(c["bone_steps"], [])
        self.assertEqual(c["muscle_steps"], [1, 2])
        self.assertTrue(c["clips"]["1"]["is_wrong"])
        self.assertFalse(c["clips"]["2"]["is_wrong"])
        self.assertEqual(c["callouts"]["1"]["deg"], "90°")
        self.assertEqual(c["callouts"]["2"]["deg"], "45°")
        self.assertEqual(ctx.order_lint(c), [])
        self.assertEqual(ctx.pair_lint(c), [])
        self.assertEqual(callout_lint(c), [])
        self.assertEqual(ctx.duration(c), 10.0)

    def test_four_step_unchanged(self):
        c = ctx.expand_motions(json.load(open(os.path.join(ROOT, "configs", "day06.json"))))
        self.assertEqual(sorted(c["clips"]), ["1", "2", "3", "4"])
        self.assertEqual(ctx.duration(c), 20.0)


if __name__ == "__main__":
    unittest.main()
