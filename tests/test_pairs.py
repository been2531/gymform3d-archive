"""Pair-motion rule (t521u): wrong steps 1/3 share one motion, right steps 2/4 share one motion.
Run: python3 -m unittest tests.test_pairs"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import ctx  # noqa: E402


def day6():
    return json.load(open(os.path.join(ROOT, "configs", "day06.json")))


class PairMotion(unittest.TestCase):
    def test_steps_are_derived_from_two_motions(self):
        c = ctx.expand_motions(day6())
        cl = c["clips"]
        self.assertEqual(cl["1"]["motion"], cl["3"]["motion"])
        self.assertEqual(cl["2"]["motion"], cl["4"]["motion"])
        self.assertEqual(cl["1"]["kp_file"], cl["3"]["kp_file"])
        self.assertEqual((cl["1"]["i0"], cl["1"]["i1"]), (cl["3"]["i0"], cl["3"]["i1"]))
        self.assertEqual([cl[s]["is_wrong"] for s in "1234"], [True, False, True, False])
        self.assertEqual([cl[s]["mode"] for s in "1234"], ["bone_blood", "bone_blood", "muscle_blood", "muscle_blood"])
        self.assertEqual(ctx.order_lint(c), [])

    def test_lint_accepts_day6(self):
        self.assertEqual(ctx.pair_lint(day6()), [])

    def test_lint_rejects_unrelated_clip_in_pair(self):
        c = day6()
        c["motions"]["wrong"]["renders"]["bone"] = {"file": "day6_idp_step1_bone_720p.mp4"}
        self.assertTrue(ctx.pair_lint(c))

    def test_lint_rejects_per_step_clips_from_day6(self):
        c = day6()
        c.pop("motions")
        c["clips"] = {str(s): {"file": f"c{s}.mp4"} for s in (1, 2, 3, 4)}
        self.assertTrue(ctx.pair_lint(c))

    def test_published_days_untouched(self):
        self.assertEqual(ctx.pair_lint(json.load(open(os.path.join(ROOT, "configs", "day05.json")))), [])


if __name__ == "__main__":
    unittest.main()
