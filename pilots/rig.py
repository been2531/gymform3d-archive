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
    # Foot length (ankle -> ball, i.e. the sole) as a fraction of standing height.
    # 0.105 was a stub that made a lifted heel read as nothing on a phone; 0.152 is the
    # anthropometric norm (~15% of stature) and is what makes the 45-degree heel lift in
    # the squat read as air under the sole at phone size (style B2 renders at 0.152).
    foot: float = 0.105

    def __post_init__(self):
        H = self.H
        self.L = {
            "upper_arm": 0.180 * H, "forearm": 0.150 * H, "hand": 0.085 * H,
            "thigh": 0.245 * H, "shin": 0.246 * H, "foot": self.foot * H,
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


TOE_DROP = 15.0     # a flat foot: the sole sits this far off the mat at the heel end
FLAT_FOOT = 180.0 - TOE_DROP   # foot line (ball -> ankle) of a flat foot on the mat
SOLE_Y = 0.012      # the ball rests this far (body heights) above the floor, on the mat's top


def ankle_from_ball(body: Body, ball, heel=0.0):
    """Ankle of a foot whose ball is planted on the mat. A lifted heel pivots on the ball,
    so the heel rises by foot_len * sin(TOE_DROP + heel) - the reason foot length sets how big
    a heel lift looks. Standing poses anchor on the ball this way (not on the ankle) so a
    foot longer than the old stub cannot push the toe through the floor."""
    return add(ball, d(FLAT_FOOT - heel), body.L["foot"])


def ball_x_from_ankle(body: Body, ankle_x: float):
    """Ball x of a flat-footed sole whose heel end sits at ankle_x."""
    return ankle_x + math.cos(math.radians(TOE_DROP)) * body.L["foot"]


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


# Rep timing (fractions of the period): lower 0-0.40, hold at the bottom 0.40-0.55,
# press up 0.55-0.90 (concentric), short hold at the top 0.90-1.0. Eased both ways.
REP_DOWN, REP_HOLD, REP_UP = 0.40, 0.55, 0.90
REP_DONE = REP_UP  # a rep is counted when the lift finishes


def _ease(x):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0.0), 1.0))


def rep_phase(t: float, period: float, offset: float = 0.0):
    """0 = top/start, 1 = bottom. Slow eased lowering, brief bottom hold, eased lift, short top hold."""
    u = ((t + offset) / period) % 1.0
    if u < REP_DOWN:
        return _ease(u / REP_DOWN)
    if u < REP_HOLD:
        return 1.0
    if u < REP_UP:
        return 1.0 - _ease((u - REP_HOLD) / (REP_UP - REP_HOLD))
    return 0.0


def rep_concentric(t: float, period: float, offset: float = 0.0):
    """0..1 pulse while the working muscles shorten (the lift), 0 elsewhere."""
    u = ((t + offset) / period) % 1.0
    if REP_HOLD <= u < REP_UP:
        return math.sin(math.pi * (u - REP_HOLD) / (REP_UP - REP_HOLD))
    return 0.0


def fault(phase: float):
    """Faults stay visible over the whole rep (55% at the top, 100% at the bottom)."""
    return 0.55 + 0.45 * phase


def hand_x_for(body: Body, floor_y: float, ball_x: float, lead: float = 0.0):
    """Hands sit just ahead of where the shoulder lands at the bottom of a
    clean rep (forearm near vertical at depth)."""
    J = pushup_pose(body, 1.0, {}, floor_y, ball_x, ball_x + 9999)
    return J["shoulder"][0] + lead * body.H


def squat_pose(body: Body, phase: float, form: dict, floor_y: float, ankle_x: float):
    """Side-view bodyweight squat, facing right. form: depth (0..1 of a full
    squat), plus the bottom-of-rep joint angles that make a fault read as a real
    pose instead of a colour change:
      shin   deg of the shin from horizontal at the bottom (90 = vertical, 52 = good)
      thigh  deg of the thigh from horizontal at the bottom (196 = hips below knees)
      lean   deg the torso leans forward from vertical at the bottom
      heel   deg the heel lifts, pivoting on the ball (0 = flat foot)
    Defaults are the good squat, so a correct form needs no angle keys; a fault
    keeps fault(phase) of its offset at the top of the rep so it stays readable.
    """
    L = body.L
    k = float(form.get("depth", 1.0)) * phase
    fw = fault(phase)
    heel = form.get("heel", 0.0) * fw
    shin_b = float(form.get("shin", 52.0 - form.get("knee_fwd", 0.0)))
    thigh_b = float(form.get("thigh", 196.0))
    lean = 42.0 * k + float(form.get("lean", 0.0)) * fw   # lean = extra forward torso lean
    ball = (ball_x_from_ankle(body, ankle_x), floor_y - SOLE_Y * body.H)  # sole planted on the mat
    ankle = ankle_from_ball(body, ball, heel)  # heel lift pivots on the ball
    knee = add(ankle, d(90.0 - (38.0 * k) + (shin_b - 52.0) * fw), L["shin"])
    hip = add(knee, d(90.0 + 106.0 * k + (thigh_b - 196.0) * fw), L["thigh"])
    t_ang = 90.0 - lean
    sh = add(hip, d(t_ang), L["torso"])
    neck = add(sh, d(t_ang + 6.0), L["neck"])
    h_ang = t_ang + 0.55 * lean - form.get("head", 0.0) * fw
    head = add(neck, d(h_ang), L["head"])
    arm = -70.0 + 72.0 * min(k * 1.4, 1.0)
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
    ball_f = (ball_x_from_ankle(body, mid_x + 0.26 * Hh), floor_y - SOLE_Y * Hh)
    ankle_f = ankle_from_ball(body, ball_f)
    ball_b = (mid_x - 0.36 * Hh, floor_y - 0.01 * Hh)
    ankle_b = add(ball_b, d(130.0), L["foot"])
    leg = L["thigh"] + L["shin"]
    hip_y0 = floor_y - 0.04 * Hh - math.sqrt(max(leg ** 2 - (0.28 * Hh) ** 2, 1)) * 0.97
    hip = (mid_x - 0.03 * Hh + form.get("fwd", 0.0) * Hh * fault(phase), hip_y0 + 0.22 * Hh * dpt)
    kf, af = two_bone(hip, ankle_f, L["thigh"], L["shin"], pick=lambda c: c[0])
    kb, ab = two_bone(hip, ankle_b, L["thigh"], L["shin"], pick=lambda c: c[1])
    ball_f = add(af, d(-22.0), L["foot"]); ball_b2 = add(ab, d(-50.0), L["foot"])
    lean = 4.0 + form.get("lean", 0.0) * fault(phase)
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
    ball = (ball_x_from_ankle(body, ankle_x), floor_y - SOLE_Y * body.H)
    ankle = ankle_from_ball(body, ball)
    kb = form.get("knee", 0.0)
    kf = kb * fault(p)  # a 'squatted' hinge keeps bent knees all rep
    knee = add(ankle, d(90.0 - 9.0 * p - kf * 0.45), L["shin"])
    hip = add(knee, d(90.0 + 22.0 * p + kf), L["thigh"])
    lean = form.get("lean", 78.0) * p
    t_ang = 90.0 - lean
    sh = add(hip, d(t_ang), L["torso"])
    neck = add(sh, d(t_ang + 4.0), L["neck"])
    h_ang = t_ang + 10.0 - form.get("head", 0.0) * p
    head = add(neck, d(h_ang), L["head"])
    J = {"ball": ball, "ankle": ankle, "knee": knee, "hip": hip, "shoulder": sh, "neck": neck, "head": head,
         "_torso_ang": t_ang, "_head_ang": h_ang}
    return _arms_down(J, L, -90.0 + 6.0 * p)
