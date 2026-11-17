"""Shared 2D joint skeleton and barbell-squat solver.

All styles use the same joint names and the same pose. Bone lengths stay
fixed via two-bone IK, so the character cannot drift between frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

Point = Tuple[float, float]
Joints = Dict[str, Point]

CANVAS = (1080, 1920)
# Bottom 15% of a 1920 frame starts here; keep the figure above it.
SAFE_BOTTOM = int(1920 * 0.85)


@dataclass(frozen=True)
class Proportions:
    head_r: float = 80.0
    neck: float = 58.0
    spine: float = 340.0
    shoulder_half: float = 136.0
    hip_half: float = 70.0
    upper_arm: float = 122.0
    forearm: float = 114.0
    thigh: float = 318.0
    shin: float = 300.0
    foot: float = 92.0
    near_down: float = 14.0
    far_up: float = 12.0
    squat_drop: float = 300.0
    squat_back: float = 48.0
    torso_lean: float = 12.0  # degrees, lean toward the facing direction
    origin_x: float = 540.0
    stand_hip_y: float = 900.0
    bar_len: float = 700.0
    bar_lift: float = 28.0  # bar sits clearly on top of the shoulder line
    bar_behind: float = 6.0
    grip_out: float = 36.0  # hands just outside the shoulders


DEFAULT = Proportions()


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def norm(a: Point) -> Point:
    l = math.hypot(a[0], a[1]) or 1.0
    return (a[0] / l, a[1] / l)


def perp(a: Point) -> Point:
    return (-a[1], a[0])


def ik_2bone(
    origin: Point, target: Point, l1: float, l2: float, bend: Point
) -> Tuple[Point, Point]:
    """Place a mid joint (knee/elbow) with a preferred bend direction."""
    ox, oy = origin
    tx, ty = target
    dx, dy = tx - ox, ty - oy
    d = math.hypot(dx, dy)
    max_reach = l1 + l2 - 1.5
    min_reach = abs(l1 - l2) + 1.5
    if d < 1e-3:
        mid = (ox + bend[0] * l1, oy + bend[1] * l1)
        return mid, target
    if d > max_reach:
        s = max_reach / d
        tx, ty = ox + dx * s, oy + dy * s
        dx, dy, d = tx - ox, ty - oy, max_reach
    elif d < min_reach:
        s = min_reach / d
        tx, ty = ox + dx * s, oy + dy * s
        dx, dy, d = tx - ox, ty - oy, min_reach
    a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    if px * bend[0] + py * bend[1] < 0:
        px, py = -px, -py
    mid = (ox + ux * a + px * h, oy + uy * a + py * h)
    return mid, (tx, ty)


def smoothstep(t: float) -> float:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return t * t * (3.0 - 2.0 * t)


def squat_depth(t: float, duration: float = 7.0, reps: int = 2) -> float:
    """0 at the top, 1 at the bottom. Two eased reps over `duration` seconds."""
    if duration <= 0:
        return 1.0
    # Small holds at the top and bottom so the pulse can read.
    cycle = (t / duration) * reps
    phase = cycle - math.floor(cycle)
    # 0..0.08 hold top, 0.08..0.45 down, 0.45..0.55 hold bottom, 0.55..0.92 up, 0.92..1 hold top
    if phase < 0.08:
        return 0.0
    if phase < 0.45:
        return smoothstep((phase - 0.08) / 0.37)
    if phase < 0.55:
        return 1.0
    if phase < 0.92:
        return 1.0 - smoothstep((phase - 0.55) / 0.37)
    return 0.0


def solve_squat(depth: float, prop: Proportions = DEFAULT) -> Joints:
    """Return joint positions for a 3/4, facing-left barbell back squat.

    depth=0 is standing; depth=1 is the bottom (thighs near parallel).
    """
    d = 0.0 if depth < 0 else 1.0 if depth > 1 else depth
    p = prop

    hip_mid: Point = (
        p.origin_x + p.squat_back * d,
        p.stand_hip_y + p.squat_drop * d,
    )
    hip_far = (hip_mid[0] - p.hip_half, hip_mid[1] - p.far_up)
    hip_near = (hip_mid[0] + p.hip_half, hip_mid[1] + p.near_down)

    # Planted feet. Near foot is slightly lower and further back for 3/4.
    stand_leg = p.thigh + p.shin - 16.0
    ankle_y_far = p.stand_hip_y - p.far_up + stand_leg
    ankle_y_near = p.stand_hip_y + p.near_down + stand_leg + 4.0
    ankle_far: Point = (p.origin_x - p.hip_half - 8.0, ankle_y_far)
    ankle_near: Point = (p.origin_x + p.hip_half + 16.0, ankle_y_near)
    toe_far = (ankle_far[0] - p.foot, ankle_far[1] + 8.0)
    toe_near = (ankle_near[0] - p.foot, ankle_near[1] + 10.0)

    # Knees bend forward (toward -x, the facing direction).
    knee_far, _ = ik_2bone(hip_far, ankle_far, p.thigh, p.shin, (-1.0, -0.05))
    knee_near, _ = ik_2bone(hip_near, ankle_near, p.thigh, p.shin, (-1.0, 0.0))

    lean = math.radians(p.torso_lean * (0.25 + 0.75 * d))
    # Spine goes up and slightly forward as the torso leans.
    shoulder_mid: Point = (
        hip_mid[0] - p.spine * math.sin(lean),
        hip_mid[1] - p.spine * math.cos(lean),
    )
    shoulder_far = (
        shoulder_mid[0] - p.shoulder_half,
        shoulder_mid[1] - p.far_up * 0.7,
    )
    shoulder_near = (
        shoulder_mid[0] + p.shoulder_half,
        shoulder_mid[1] + p.near_down * 0.7,
    )

    bar_mid: Point = (
        shoulder_mid[0] + p.bar_behind,
        shoulder_mid[1] - p.bar_lift,
    )
    # Head sits fully above the bar (back squat: bar behind the neck, not
    # through the face). Neck is the short link from shoulders to chin.
    head: Point = (
        shoulder_mid[0] - 16.0 - 4.0 * d,
        bar_mid[1] - p.head_r - 22.0,
    )
    neck: Point = (
        (head[0] + shoulder_mid[0]) * 0.5,
        (head[1] + p.head_r * 0.85 + shoulder_mid[1]) * 0.5,
    )
    bar_l = (bar_mid[0] - p.bar_len * 0.5, bar_mid[1])
    bar_r = (bar_mid[0] + p.bar_len * 0.5, bar_mid[1])
    # Plate centres sit on the bar, inboard of the sleeves so the bar
    # clearly passes through them and a short collar sticks out.
    plate_inset = 36.0
    plate_l = (bar_l[0] + plate_inset, bar_l[1])
    plate_r = (bar_r[0] - plate_inset, bar_r[1])

    grip_far = (shoulder_far[0] - p.grip_out, bar_mid[1])
    grip_near = (shoulder_near[0] + p.grip_out, bar_mid[1])

    # Elbows hang down and slightly back of the bar.
    elbow_far, hand_far = ik_2bone(
        shoulder_far, grip_far, p.upper_arm, p.forearm, (0.0, 1.0)
    )
    elbow_near, hand_near = ik_2bone(
        shoulder_near, grip_near, p.upper_arm, p.forearm, (0.05, 1.0)
    )

    return {
        "head": head,
        "neck": neck,
        "shoulder_mid": shoulder_mid,
        "shoulder_far": shoulder_far,
        "shoulder_near": shoulder_near,
        "elbow_far": elbow_far,
        "elbow_near": elbow_near,
        "hand_far": hand_far,
        "hand_near": hand_near,
        "hip_mid": hip_mid,
        "hip_far": hip_far,
        "hip_near": hip_near,
        "knee_far": knee_far,
        "knee_near": knee_near,
        "ankle_far": ankle_far,
        "ankle_near": ankle_near,
        "toe_far": toe_far,
        "toe_near": toe_near,
        "bar_l": bar_l,
        "bar_r": bar_r,
        "bar_mid": bar_mid,
        "plate_l": plate_l,
        "plate_r": plate_r,
        "grip_far": grip_far,
        "grip_near": grip_near,
    }


def anterior(hip: Point, knee: Point, scale: float = 1.0) -> Point:
    """Unit vector pointing to the front of the thigh (facing left)."""
    u = norm(_sub(knee, hip))
    # Perp that points roughly toward -x (forward).
    n = perp(u)
    if n[0] > 0:
        n = (-n[0], -n[1])
    return _mul(n, scale)


def muscle_center(
    hip: Point, knee: Point, along: float = 0.40, out: float = 18.0
) -> Point:
    u = norm(_sub(knee, hip))
    length = dist(hip, knee)
    ant = anterior(hip, knee)
    return _add(hip, _add(_mul(u, length * along), _mul(ant, out)))
