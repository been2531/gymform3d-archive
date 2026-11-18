"""Posture claims: what a caption asserts about the body, and how it is measured.

A caption that names a posture ("HIPS BELOW KNEES", "HEELS UP", "CHEST TO FLOOR")
is only true while the drawn joints actually show it. Two users of this module:

- style_b2 draws such a caption only while its claim holds on the current pose, so
  a line cannot sit on screen over a body that contradicts it (the v5 defect: the
  close-up title stayed up while the figure stood back up, and the ending caption
  said "CHEST TO FLOOR" over a standing victory pose);
- qa's `pose_caption_sync` gate re-measures every logged frame from the joints the
  renderer recorded, so a caption that drifts from the pose fails the reel.

Angles are measured on the world-space rig joints (screen y grows downward, the
figure faces right), scaled by the character height H so thresholds are device-free.
"""
from __future__ import annotations
import math

from .rig import FLAT_FOOT

# phrase as it appears on screen -> claim id
CLAIMS = {
    "HIPS BELOW KNEES": "hips_below_knees",
    "HIPS SAG": "hips_sag",
    "HIPS DROP": "hips_sag",
    "HEELS UP": "heels_up",
    "HEELS DOWN": "heels_down",
    "HEEL DOWN": "heels_down",
    "CHEST TO FLOOR": "chest_to_floor",
    "ONE LINE": "straight_line",
    "STRAIGHT LINE": "straight_line",
    "SIT BACK": "sit_back",
}

# joints each claim is measured from (a caption naming a posture the frame cannot
# show - e.g. a floor claim over the standing ending - is a mismatch, not a pass)
NEEDS = {
    "hips_below_knees": ("hip", "knee"),
    "hips_sag": ("hip", "ankle", "shoulder"),
    "heels_up": ("ball", "ankle"),
    "heels_down": ("ball", "ankle"),
    "chest_to_floor": ("ball", "shoulder"),
    "straight_line": ("hip", "ankle", "shoulder"),
    "sit_back": ("hip", "ankle"),
}

HEEL_UP_MIN = 18.0      # foot tilted at least this far above flat = heel is off the floor
HEEL_DOWN_MAX = 6.0     # ... within this much of flat = heel is planted
BODY_EPS = 0.01         # 1% of standing height: "level with" tolerance


def claims_in(text):
    """Claim ids named by one piece of on-screen text (empty when it names none)."""
    up = " ".join((text or "").upper().split())
    return [c for phrase, c in CLAIMS.items() if phrase in up]


def foot_angle(J):
    """Degrees of the foot line from the floor, ball -> ankle (rig.FLAT_FOOT when flat)."""
    return math.degrees(math.atan2(-(J["ankle"][1] - J["ball"][1]), J["ankle"][0] - J["ball"][0]))


def _line_dist(J):
    """Signed distance of the hip above (+)/below (-) the ankle->shoulder line."""
    ax, ay = J["ankle"]; sx, sy = J["shoulder"]; hx, hy = J["hip"]
    dx, dy = sx - ax, sy - ay
    n = math.hypot(dx, dy) or 1e-6
    return -((hy - ay) * dx - (hx - ax) * dy) / n


def measure(claim, J, H):
    """True when the pose in J (world joints) shows the claim; None when unmeasurable."""
    if J is None or any(k not in J for k in NEEDS.get(claim, ())):
        return None
    if claim == "hips_below_knees":
        return J["hip"][1] >= J["knee"][1] + BODY_EPS * H
    if claim == "heels_up":
        return foot_angle(J) <= FLAT_FOOT - HEEL_UP_MIN
    if claim == "heels_down":
        return foot_angle(J) >= FLAT_FOOT - HEEL_DOWN_MAX
    if claim == "chest_to_floor":
        return (J["ball"][1] - J["shoulder"][1]) <= 0.22 * H
    if claim == "straight_line":
        return abs(_line_dist(J)) <= 0.035 * H
    if claim == "hips_sag":
        return _line_dist(J) <= -0.025 * H
    if claim == "sit_back":
        return J["hip"][0] <= J["ankle"][0] - 0.05 * H
    raise KeyError(claim)


def holds(text, J, H):
    """True when every posture claim in the text holds on this pose (no claim = True)."""
    cs = claims_in(text)
    return all(measure(c, J, H) for c in cs)


def synced(text, J, H):
    """(ok, claim, measured) for the gate: claim that does not hold, if any."""
    for c in claims_in(text):
        m = measure(c, J, H)
        if not m:
            return False, c, m
    return True, None, True
