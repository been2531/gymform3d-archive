"""The B2 quality gates that keep a caption on the pose it describes and keep limbs in frame.

    python3 -m unittest tests.test_pose_caption_gates -v
"""
import unittest

from pilots import claims, qa
from pilots.rig import Body, squat_pose

FLOOR = 1250.0


def squat_pose_at(phase, form):
    return squat_pose(Body(H=760), phase, form, FLOOR, 520.0)


GOOD = {"depth": 1.0, "lean": 0, "head": 0}
HEELS_UP = {"depth": 0.85, "shin": 62, "thigh": 170, "lean": 14, "heel": 22, "head": 18}


class ClaimTest(unittest.TestCase):
    def test_hips_below_knees_only_at_the_bottom(self):
        bottom, top = squat_pose_at(1.0, GOOD), squat_pose_at(0.0, GOOD)
        self.assertTrue(claims.holds("HIPS BELOW KNEES", bottom, 760))
        self.assertFalse(claims.holds("HIPS BELOW KNEES", top, 760))

    def test_heels_up_is_a_pose_fact(self):
        fault, good = squat_pose_at(1.0, HEELS_UP), squat_pose_at(1.0, GOOD)
        self.assertTrue(claims.holds("HEELS UP", fault, 760))
        self.assertFalse(claims.holds("HEELS UP", good, 760))
        self.assertTrue(claims.holds("HEELS DOWN", good, 760))
        self.assertFalse(claims.holds("HEELS DOWN", fault, 760))

    def test_chest_to_floor_needs_a_body_on_the_floor(self):
        standing = squat_pose_at(0.0, GOOD)   # a standing pose cannot show a floor press
        self.assertFalse(claims.holds("CHEST TO FLOOR", standing, 760))

    def test_unknown_text_makes_no_claim(self):
        self.assertEqual(claims.claims_in("SQUEEZE GLUTES"), [])
        self.assertTrue(claims.holds("SQUEEZE GLUTES", squat_pose_at(0.0, GOOD), 760))


class PoseCaptionSyncTest(unittest.TestCase):
    def meta(self, joints, texts):
        return {"t": 0.0, "joints": joints, "text": [{"role": "title", "text": t} for t in texts]}

    def test_caption_over_a_contradicting_pose_fails(self):
        m = self.meta([squat_pose_at(0.0, GOOD)], ["HIPS BELOW KNEES"])
        ok, detail = qa.pose_caption_sync([m], Body(H=760))
        self.assertFalse(ok, detail)

    def test_caption_over_the_matching_pose_passes(self):
        m = self.meta([squat_pose_at(1.0, GOOD)], ["HIPS BELOW KNEES"])
        ok, detail = qa.pose_caption_sync([m], Body(H=760))
        self.assertTrue(ok, detail)

    def test_floor_claim_over_a_standing_ending_fails(self):
        m = self.meta([squat_pose_at(1.0, GOOD)], ["CHEST TO FLOOR"])
        ok, _ = qa.pose_caption_sync([m], Body(H=760))
        self.assertFalse(ok)

    def test_caption_without_a_posture_claim_passes(self):
        ok, detail = qa.pose_caption_sync([self.meta([], ["OWN THE DEPTH"])], Body(H=760))
        self.assertTrue(ok, detail)
        self.assertIn("0 posture captions", detail)

    def test_a_claim_that_never_holds_is_reported(self):
        # the episode asks for HIPS BELOW KNEES but the scene only ever shows a standing pose
        ep = {"B2": {"scenes": [{"t0": 0.0, "t1": 1.0, "type": "reps", "form": "good",
                                 "title": "HIPS BELOW KNEES"}]}}
        m = self.meta([squat_pose_at(0.0, GOOD)], ["OWN THE DEPTH"])
        ok, detail = qa.pose_caption_sync([m], Body(H=760), ep)
        self.assertFalse(ok, detail)
        self.assertIn("never holds", detail)


class LimbsInBoundsTest(unittest.TestCase):
    def test_inside_points_pass(self):
        ok, detail = qa.limbs_in_bounds([{"scr": [(40, 40), (1040, 1880)]}])
        self.assertTrue(ok, detail)

    def test_a_limb_on_the_edge_fails(self):
        ok, detail = qa.limbs_in_bounds([{"scr": [(40, 40), (3, 900)]}])
        self.assertFalse(ok, detail)
        self.assertIn("frames", detail)


class FaultJointDeltaTest(unittest.TestCase):
    ep = {"forms": {"good": GOOD, "heel": HEELS_UP}}

    def metas(self, wrong):
        poses = {name: {"top": squat_pose_at(0.0, f), "bottom": squat_pose_at(1.0, f)}
                 for name, f in {"good": GOOD, "heel": wrong}.items()}
        return [{"t": 0.0, "form_poses": poses}]

    def test_a_real_pose_change_passes(self):
        ok, detail = qa._fault_joint_delta(self.ep, self.metas(HEELS_UP), 760)
        self.assertTrue(ok, detail)

    def test_a_colour_only_change_fails(self):
        colour_only = dict(GOOD, highlight=[["hip", "shoulder"]])
        ok, detail = qa._fault_joint_delta(self.ep, self.metas(colour_only), 760)
        self.assertFalse(ok, detail)


if __name__ == "__main__":
    unittest.main()
