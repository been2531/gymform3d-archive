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
            ("hip", "knee2"): L["thigh"], ("knee2", "ankle2"): L["shin"], ("ankle2", "ball2"): L["foot"],
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
    ball0 = add((ankle_x, floor_y - 0.04 * body.H), d(-22.0), L["foot"])
    ball = (ball0[0], ball0[1])
    ankle = add(ball, d(158.0 - form.get("heel", 0.0) * phase), L["foot"])  # heel lift pivots on the ball
    knee = add(ankle, d(90.0 - (38.0 + form.get("knee_fwd", 0.0)) * dpt), L["shin"])
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



def _arms_down(J, L, ang_deg=-90.0):
    J["elbow"] = add(J["shoulder"], d(ang_deg), L["upper_arm"])
    J["wrist"] = add(J["elbow"], d(ang_deg), L["forearm"])
    J["fingers"] = add(J["wrist"], d(ang_deg), L["hand"])
    return J


def plank_pose(body: Body, phase: float, form: dict, floor_y: float, ball_x: float):
    """Forearm plank, facing right. A hold: phase drives a slow breath/fatigue
    drift (sag/pike grow with phase for faulty forms)."""
    L = body.L
    sag = form.get("sag", 0.0) * (0.6 + 0.4 * phase) - form.get("pike", 0.0) * (0.6 + 0.4 * phase)
    breath = 1.5 * phase
    B = (ball_x, floor_y - 0.012 * body.H)
    target_y = floor_y - 0.03 * body.H - L["upper_arm"]

    def chain(theta):
        ankle = add(B, d(theta + 100.0), L["foot"])
        knee = add(ankle, d(theta), L["shin"])
        hip = add(knee, d(theta - sag * 0.5 + breath), L["thigh"])
        sh = add(hip, d(theta + sag * 0.9 - breath), L["torso"])
        return ankle, knee, hip, sh
    lo, hi = -20.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if chain(mid)[3][1] > target_y: lo = mid
        else: hi = mid
    theta = (lo + hi) / 2
    ankle, knee, hip, sh = chain(theta)
    t_ang = theta + sag * 0.9 - breath
    neck = add(sh, d(t_ang + 8.0), L["neck"])
    h_ang = t_ang + 4.0 - form.get("head", 0.0)
    head = add(neck, d(h_ang), L["head"])
    elbow = add(sh, d(-90.0), L["upper_arm"])
    wrist = add(elbow, d(0.0), L["forearm"])
    fingers = add(wrist, d(0.0), L["hand"])
    return {"ball": B, "ankle": ankle, "knee": knee, "hip": hip, "shoulder": sh, "neck": neck, "head": head,
            "elbow": elbow, "wrist": wrist, "fingers": fingers, "_torso_ang": t_ang, "_head_ang": h_ang}


def lunge_pose(body: Body, phase: float, form: dict, floor_y: float, mid_x: float):
    """Static split-stance lunge, facing right. Front leg = knee/ankle/ball,
    back leg = knee2/ankle2/ball2. form: depth, lean, fwd (hip shift fwd, xH), heel."""
    L = body.L; Hh = body.H
    dpt = form.get("depth", 1.0) * phase
    ankle_f = (mid_x + 0.26 * Hh, floor_y - 0.04 * Hh)
    ball_f = add(ankle_f, d(-22.0), L["foot"])
    ball_b = (mid_x - 0.36 * Hh, floor_y - 0.01 * Hh)
    ankle_b = add(ball_b, d(130.0), L["foot"])
    leg = L["thigh"] + L["shin"]
    hip_y0 = floor_y - 0.04 * Hh - math.sqrt(max(leg ** 2 - (0.28 * Hh) ** 2, 1)) * 0.97
    hip = (mid_x - 0.03 * Hh + form.get("fwd", 0.0) * Hh * phase, hip_y0 + 0.22 * Hh * dpt)
    kf, af = two_bone(hip, ankle_f, L["thigh"], L["shin"], pick=lambda c: c[0])
    kb, ab = two_bone(hip, ankle_b, L["thigh"], L["shin"], pick=lambda c: c[1])
    ball_f = add(af, d(-22.0), L["foot"]); ball_b2 = add(ab, d(-50.0), L["foot"])
    lean = 4.0 + form.get("lean", 0.0) * phase
    t_ang = 90.0 - lean
    sh = add(hip, d(t_ang), L["torso"])
    neck = add(sh, d(t_ang + 4.0), L["neck"])
    h_ang = 90.0 - lean * 0.4
    head = add(neck, d(h_ang), L["head"])
    J = {"ball": ball_f, "ankle": af, "knee": kf, "hip": hip, "shoulder": sh, "neck": neck, "head": head,
         "knee2": kb, "ankle2": ab, "ball2": ball_b2, "_torso_ang": t_ang, "_head_ang": h_ang}
    return _arms_down(J, L, -86.0)


def bridge_pose(body: Body, phase: float, form: dict, floor_y: float, sh_x: float):
    """Glute bridge, lying on the back, head left, feet right. form: lift (deg of
    torso rise at the top), head (deg chin tuck)."""
    L = body.L; Hh = body.H
    R = 0.105 * Hh
    sh = (sh_x, floor_y - R * 0.95)
    a = 4.0 + form.get("lift", 28.0) * phase
    hip = add(sh, d(a), L["torso"])
    ankle = (sh_x + L["torso"] + 0.30 * Hh, floor_y - 0.04 * Hh)
    knee, ankle = two_bone(hip, ankle, L["thigh"], L["shin"], pick=lambda c: -c[1])
    ball = add(ankle, d(-10.0), L["foot"])
    neck = add(sh, d(180.0), L["neck"])
    h_ang = 180.0 - form.get("head", 0.0) * phase
    head = add(neck, d(h_ang), L["head"])
    elbow = add(sh, d(-4.0), L["upper_arm"])
    wrist = add(elbow, d(0.0), L["forearm"])
    fingers = add(wrist, d(0.0), L["hand"])
    return {"ball": ball, "ankle": ankle, "knee": knee, "hip": hip, "shoulder": sh, "neck": neck, "head": head,
            "elbow": elbow, "wrist": wrist, "fingers": fingers, "_torso_ang": a, "_head_ang": h_ang}


def hinge_pose(body: Body, phase: float, form: dict, floor_y: float, ankle_x: float):
    """Bodyweight hip hinge (RDL pattern), facing right. form: knee (deg extra
    knee bend = 'squatting the hinge'), lean (max torso lean from vertical), head."""
    L = body.L
    p = phase
    ankle = (ankle_x, floor_y - 0.04 * body.H)
    ball = add(ankle, d(-22.0), L["foot"])
    kb = form.get("knee", 0.0)
    knee = add(ankle, d(90.0 - (9.0 + kb * 0.45) * p), L["shin"])
    hip = add(knee, d(90.0 + (22.0 + kb) * p), L["thigh"])
    lean = form.get("lean", 78.0) * p
    t_ang = 90.0 - lean
    sh = add(hip, d(t_ang), L["torso"])
    neck = add(sh, d(t_ang + 4.0), L["neck"])
    h_ang = t_ang + 10.0 - form.get("head", 0.0) * p
    head = add(neck, d(h_ang), L["head"])
    J = {"ball": ball, "ankle": ankle, "knee": knee, "hip": hip, "shoulder": sh, "neck": neck, "head": head,
         "_torso_ang": t_ang, "_head_ang": h_ang}
    return _arms_down(J, L, -90.0 + 6.0 * p)
