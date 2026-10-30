"""motion_check tests (t549u/t550u): standards file integrity + angle maths + pass/fail logic on synthetic keypoints.
Fast, pure python. Run: python3 -m unittest tests.test_motion_check"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import motion_check as mc  # noqa: E402


def body(arm_deg=0.0, elbow_deg=0.0, trunk_deg=0.0, knee_deg=0.0):
    """Synthetic front-view skeleton (y down). arm_deg = arm-torso angle, elbow_deg = elbow flexion."""
    kp = {}
    for s, sx in (("r", -1), ("l", 1)):
        hip = (500 + sx * 60, 1000)
        t = math.radians(trunk_deg)
        sh = (hip[0] + 400 * math.sin(t), hip[1] - 400 * math.cos(t))
        a = math.radians(arm_deg)  # rotate the arm outward from straight down
        elb = (sh[0] + sx * 250 * math.sin(a), sh[1] + 250 * math.cos(a))
        b = a + math.radians(elbow_deg)
        wr = (elb[0] + sx * 220 * math.sin(b), elb[1] + 220 * math.cos(b))
        k = math.radians(knee_deg)
        kn = (hip[0], hip[1] + 400)
        ank = (kn[0] - 380 * math.sin(k), kn[1] + 380 * math.cos(k))
        kp.update({f"sh_{s}": sh, f"elb_{s}": elb, f"wr_{s}": wr, f"hip_{s}": hip, f"kn_{s}": kn, f"ank_{s}": ank})
    kp["sh"] = mc._mid(kp["sh_r"], kp["sh_l"])
    return kp


def rep(peak_arm, n=20, **kw):
    return [body(arm_deg=peak_arm * math.sin(math.pi * i / (n - 1)), **kw) for i in range(n)]


class Standards(unittest.TestCase):
    def setUp(self):
        self.S = mc.load_standards()

    def test_covers_done_and_queued_exercises(self):
        for slug in ("cable_rope_pushdown", "wide_grip_lat_pulldown", "barbell_back_squat", "romanian_deadlift",
                     "dumbbell_lateral_raise", "incline_dumbbell_press", "seated_cable_row", "bulgarian_split_squat"):
            self.assertIn(slug, self.S["exercises"])
        self.assertGreaterEqual(len(self.S["exercises"]), 12)

    def test_every_item_is_sourced(self):
        src = self.S["sources"]
        for k, v in src.items():
            self.assertTrue(v["url"].startswith("https://"), k)
        for slug, ex in self.S["exercises"].items():
            self.assertTrue(ex["target_muscles"], slug)
            for it in ex["right_form"] + ex["mistakes"]:
                self.assertTrue(it["sources"], f"{slug}:{it['id']}")
                for s in it["sources"]:
                    self.assertIn(s, src, f"{slug}:{it['id']}")
            t = ex["tempo"]
            for key in ("eccentric_s", "pause_s", "concentric_s", "rep_s"):
                lo, hi = t[key]
                self.assertLessEqual(lo, hi)
            self.assertTrue(t["sources"])
            self.assertLessEqual(t["rep_s"][1], self.S["_meta"]["rep_duration_effective_s"]["range"][1])


class Angles(unittest.TestCase):
    def test_joint_angles(self):
        a = mc.joint_angles(body(arm_deg=90, elbow_deg=0, trunk_deg=0, knee_deg=40))
        self.assertAlmostEqual(mc._agg(a["arm_torso"]), 90, delta=0.5)
        self.assertAlmostEqual(mc._agg(a["elbow_flexion"]), 0, delta=0.5)
        self.assertAlmostEqual(a["trunk_from_vertical"], 0, delta=0.5)
        self.assertAlmostEqual(mc._agg(a["knee_flexion"]), 40, delta=0.5)
        self.assertAlmostEqual(mc.joint_angles(body(trunk_deg=30))["trunk_from_vertical"], 30, delta=0.5)

    def test_legacy_cache_without_arms_is_na_not_fail(self):
        kp = {k: v for k, v in body().items() if k in ("sh", "hip_r", "hip_l", "kn_r", "kn_l", "ank_r", "ank_l")}
        a = mc.joint_angles(kp)
        self.assertTrue(math.isnan(mc._agg(a["arm_torso"])))
        self.assertFalse(math.isnan(mc._agg(a["knee_flexion"])))


class Checks(unittest.TestCase):
    def steps(self, right_peak, wrong_peak):
        return {1: {"frames": rep(wrong_peak), "is_wrong": True}, 2: {"frames": rep(right_peak), "is_wrong": False}}

    def test_lateral_raise_right_passes_wrong_shows_mistake(self):
        r = mc.check_day("dumbbell_lateral_raise", self.steps(90, 140))
        self.assertTrue(r["ok"], r["fails"])
        summ = [i for i in r["items"] if i["id"] == "wrong_shows_real_mistake"][0]
        self.assertIn("raise_too_high", summ["value"])

    def test_lateral_raise_too_low_fails(self):
        r = mc.check_day("dumbbell_lateral_raise", self.steps(45, 140))
        self.assertFalse(r["ok"])
        self.assertIn("2:raise_to_shoulder_height", r["fails"])

    def test_wrong_step_without_measurable_mistake_fails(self):
        r = mc.check_day("dumbbell_lateral_raise", self.steps(90, 90), mistake_id="raise_too_high")
        self.assertIn("1:wrong_shows_real_mistake", r["fails"])

    def test_view_gated_items_are_na_in_3_4_view(self):
        r = mc.check_day("incline_dumbbell_press", self.steps(50, 90), inputs={"bench_incline_deg": 30})
        tuck = [i for i in r["items"] if i["id"] == "elbow_tuck_bottom"][0]
        self.assertIsNone(tuck["pass"])
        inc = [i for i in r["items"] if i["id"] == "input_bench_incline_deg"][0]
        self.assertTrue(inc["pass"])
        r45 = mc.check_day("incline_dumbbell_press", self.steps(50, 90), inputs={"bench_incline_deg": 45})
        self.assertIn("all:input_bench_incline_deg", r45["fails"])

    def test_tempo(self):
        ex = mc.load_standards()["exercises"]["dumbbell_lateral_raise"]
        day4 = mc.pipeline_tempo("day4_cosine")
        self.assertAlmostEqual(day4["rep_s"], 5.0 / 1.5, places=3)
        self.assertTrue(all(i["pass"] for i in mc.check_tempo(ex, day4)))
        too_fast = {"eccentric_s": 0.3, "pause_s": 0, "concentric_s": 0.3, "rep_s": 0.6}
        self.assertFalse(all(i["pass"] for i in mc.check_tempo(ex, too_fast)))

    def test_unknown_slug(self):
        with self.assertRaises(KeyError):
            mc.check_day("no_such_lift", {})


if __name__ == "__main__":
    unittest.main()
