"""2D side-view rig for the push-up. Bone lengths are fixed; every pose is
built by forward kinematics plus exact two-bone IK, so limbs cannot drift.

Angles are in degrees, measured counter-clockwise on screen (0 = right,
90 = up). Screen y grows downward.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

BONES = [  # (parent joint, child joint) pairs whose length must stay constant
    ("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "fingers"),
    ("hip", "knee"), ("knee", "ankle"), ("ankle", "ball"),
    ("hip", "shoulder"), ("shoulder", "neck"), ("neck", "head"),
]


def d(a):
    r = math.radians(a)
    return (math.cos(r), -math.sin(r))


def add(p, v, s=1.0):
    return (p[0] + v[0] * s, p[1] + v[1] * s)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ang(a, b):
    return math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))


@dataclass
class Body:
    H: float = 800.0  # standing height in px (world units)

    def __post_init__(self):
        H = self.H
        self.L = {
            "upper_arm": 0.180 * H, "forearm": 0.150 * H, "hand": 0.085 * H,
            "thigh": 0.245 * H, "shin": 0.246 * H, "foot": 0.105 * H,
            "torso": 0.290 * H, "neck": 0.055 * H, "head": 0.075 * H,
        }

    def nominal(self):
        L = self.L
        return {
            ("shoulder", "elbow"): L["upper_arm"], ("elbow", "wrist"): L["forearm"],
            ("wrist", "fingers"): L["hand"], ("hip", "knee"): L["thigh"],
            ("knee", "ankle"): L["shin"], ("ankle", "ball"): L["foot"],
            ("hip", "shoulder"): L["torso"], ("shoulder", "neck"): L["neck"],
            ("neck", "head"): L["head"],
        }


def two_bone(S, W, l1, l2, prefer_left=True, pick=None):
    """Elbow position for shoulder S, wrist W. Exact lengths; W is pulled in
    along S->W if out of reach (keeps lengths constant)."""
    dd = dist(S, W)
    dd = max(min(dd, l1 + l2 - 1e-3), abs(l1 - l2) + 1e-3)
    ux, uy = ((W[0] - S[0]) / max(dist(S, W), 1e-6), (W[1] - S[1]) / max(dist(S, W), 1e-6))
    a = (l1 * l1 - l2 * l2 + dd * dd) / (2 * dd)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    mx, my = S[0] + ux * a, S[1] + uy * a
    c1 = (mx - uy * h, my + ux * h)
    c2 = (mx + uy * h, my - ux * h)
    score = lambda c: -c[1] - (0.5 * c[0] if prefer_left else -0.5 * c[0])
    E = c1 if score(c1) > score(c2) else c2
    if pick is not None:
        E = max((c1, c2), key=pick)
    W2 = (S[0] + ux * dd, S[1] + uy * dd)
    # re-derive wrist so |E-W| == l2 exactly
    v = (W2[0] - E[0], W2[1] - E[1]); n = math.hypot(*v)
    W3 = (E[0] + v[0] / n * l2, E[1] + v[1] / n * l2)
    return E, W3


def pushup_pose(body: Body, phase: float, form: dict, floor_y: float, ball_x: float, hand_x: float):
    """phase 0 = top, 1 = bottom. form keys: sag (deg hips drop), pike (deg),
    depth (0..1 of full range), head (deg head drop vs torso)."""
    L = body.L
    depth = form.get("depth", 1.0)
    sag = form.get("sag", 0.0) - form.get("pike", 0.0)
    head_drop = form.get("head", 0.0)
    wrist_h = 0.035 * body.H
    W = (hand_x, floor_y - wrist_h)
    arm = L["upper_arm"] + L["forearm"]
    h_top = arm * 0.985 + wrist_h
    h_bot = 0.165 * body.H
    h = h_top + (h_bot - h_top) * depth * phase
    target_y = floor_y - h
    B = (ball_x, floor_y - 0.012 * body.H)

    def chain(theta):
        ankle = add(B, d(theta + 100.0), L["foot"])
        knee = add(ankle, d(theta), L["shin"])
        hip = add(knee, d(theta - sag * 0.5), L["thigh"])
        sh = add(hip, d(theta + sag * 0.9), L["torso"])
        return ankle, knee, hip, sh

    lo, hi = -10.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if chain(mid)[3][1] > target_y:
            lo = mid
        else:
            hi = mid
    theta = (lo + hi) / 2
    ankle, knee, hip, sh = chain(theta)
    t_ang = theta + sag * 0.9
    neck = add(sh, d(t_ang + 8.0), L["neck"])
    head = add(neck, d(t_ang + 4.0 - head_drop), L["head"])
    E, W2 = two_bone(sh, W, L["upper_arm"], L["forearm"], prefer_left=True)
    fingers = add(W2, d(0.0), L["hand"])
    return {
        "ball": B, "ankle": ankle, "knee": knee, "hip": hip, "shoulder": sh,
        "neck": neck, "head": head, "elbow": E, "wrist": W2, "fingers": fingers,
        "_torso_ang": t_ang, "_head_ang": t_ang + 4.0 - head_drop, "_theta": theta,
    }


def standing_pose(body: Body, base, t: float, kind: str):
    """Simple FK standing poses for cut-aways (think / flex / point)."""
    L = body.L
    x, fy = base
    bob = math.sin(t * 2 * math.pi * 1.2)
    ball = (x + 0.06 * body.H, fy)
    ankle = (x, fy - 0.04 * body.H)
    knee = add(ankle, d(92 + 2 * bob), L["shin"])
    hip = add(knee, d(88 - 2 * bob), L["thigh"])
    sh = add(hip, d(91 + 1.5 * bob), L["torso"])
    neck = add(sh, d(90), L["neck"])
    if kind == "think":
        tilt = 98 + 8 * math.sin(t * 2 * math.pi * 0.9)
        head = add(neck, d(tilt), L["head"])
        ua = 150 + 6 * math.sin(t * 2 * math.pi * 3.0)
        elbow = add(sh, d(ua), L["upper_arm"])
        wrist = add(elbow, d(ua - 140 + 25 * math.sin(t * 2 * math.pi * 3.0)), L["forearm"])
    elif kind == "flex":
        head = add(neck, d(90), L["head"])
        pump = math.sin(t * 2 * math.pi * 1.6)
        elbow = add(sh, d(10 + 8 * pump), L["upper_arm"])
        wrist = add(elbow, d(100 + 20 * pump), L["forearm"])
    else:  # point / thumbs up
        head = add(neck, d(90 + 4 * bob), L["head"])
        elbow = add(sh, d(-60 + 5 * bob), L["upper_arm"])
        wrist = add(elbow, d(30 + 20 * math.sin(t * 2 * math.pi * 1.5)), L["forearm"])
    fingers = add(wrist, d(ang(elbow, wrist)), L["hand"])
    return {"ball": ball, "ankle": ankle, "knee": knee, "hip": hip, "shoulder": sh,
            "neck": neck, "head": head, "elbow": elbow, "wrist": wrist, "fingers": fingers,
            "_torso_ang": ang(hip, sh), "_head_ang": ang(neck, head)}


def bone_errors(body: Body, J: dict):
    nom = body.nominal()
    return max(abs(dist(J[a], J[b]) - L) for (a, b), L in nom.items() if a in J and b in J)


def rep_phase(t: float, period: float, offset: float = 0.0):
    """Smooth 0..1..0 rep curve; slightly slower lowering than pressing."""
    u = ((t + offset) / period) % 1.0
    if u < 0.55:
        x = u / 0.55
    else:
        x = 1.0 - (u - 0.55) / 0.45
    return 0.5 - 0.5 * math.cos(math.pi * x)


def hand_x_for(body: Body, floor_y: float, ball_x: float, lead: float = 0.0):
    """Hands sit just ahead of where the shoulder lands at the bottom of a
    clean rep (forearm near vertical at depth)."""
    J = pushup_pose(body, 1.0, {}, floor_y, ball_x, ball_x + 9999)
    return J["shoulder"][0] + lead * body.H


def squat_pose(body: Body, phase: float, form: dict, floor_y: float, ankle_x: float):
    """Side-view bodyweight squat, facing right. form: depth (0..1 of a full
    squat), lean (extra forward torso lean in deg at the bottom)."""
    L = body.L
    dpt = form.get("depth", 1.0) * phase
    ankle = (ankle_x, floor_y - 0.04 * body.H)
    ball = add(ankle, d(-22.0), L["foot"])
    knee = add(ankle, d(90.0 - 38.0 * dpt), L["shin"])
    hip = add(knee, d(90.0 + 106.0 * dpt), L["thigh"])
    lean = 42.0 * dpt + form.get("lean", 0.0) * phase
    t_ang = 90.0 - lean
    sh = add(hip, d(t_ang), L["torso"])
    neck = add(sh, d(t_ang + 6.0), L["neck"])
    h_ang = t_ang + 0.55 * lean - form.get("head", 0.0) * phase
    head = add(neck, d(h_ang), L["head"])
    arm = -70.0 + 72.0 * min(dpt * 1.4, 1.0)
    elbow = add(sh, d(arm), L["upper_arm"])
    wrist = add(elbow, d(arm + 4.0), L["forearm"])
    fingers = add(wrist, d(arm + 4.0), L["hand"])
    return {"ball": ball, "ankle": ankle, "knee": knee, "hip": hip, "shoulder": sh, "neck": neck, "head": head,
            "elbow": elbow, "wrist": wrist, "fingers": fingers, "_torso_ang": t_ang, "_head_ang": h_ang}
