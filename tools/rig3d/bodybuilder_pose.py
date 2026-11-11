"""Hand-keyed poses for the minglanders1 25-bone Mixamo-like bodybuilder.

World +Z up. Character facing is measured from the rest-pose hips/spine.
Feet stay planted with 2-bone IK; the torso is hinged by rotating Hips.
"""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _n(v):
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def find_pbone(arm, *needles, exclude=("end", "front", "twist", "ik")):
    lower = {b.name.lower().replace(" ", ""): b for b in arm.pose.bones}
    for name, b in lower.items():
        if any(x in name for x in exclude):
            continue
        if all(n.replace(" ", "") in name for n in needles):
            return b
    return None


def map_bones(arm):
    """Resolve Mixamo-like names on this 25-bone rig."""
    def f(*n, exclude=("end", "front", "twist", "ik")):
        return find_pbone(arm, *n, exclude=exclude)

    bones = {
        "hips": f("hips") or f("hip"),
        "spine_lo": f("spine02") or f("spine2"),
        "spine_mid": f("spine01") or f("spine1"),
        "spine_hi": None,
        "neck": f("neck"),
        "head": f("head", exclude=("end", "front")),
        "l_upleg": f("left", "upleg") or f("left", "up_leg") or f("left", "thigh"),
        "r_upleg": f("right", "upleg") or f("right", "up_leg") or f("right", "thigh"),
        "l_leg": f("left", "leg", exclude=("up", "thigh", "foot", "toe")),
        "r_leg": f("right", "leg", exclude=("up", "thigh", "foot", "toe")),
        "l_foot": f("left", "foot"),
        "r_foot": f("right", "foot"),
        "l_toe": f("left", "toe"),
        "r_toe": f("right", "toe"),
        "l_shoulder": f("left", "shoulder"),
        "r_shoulder": f("right", "shoulder"),
        "l_arm": f("left", "arm", exclude=("fore", "hand", "shoulder")),
        "r_arm": f("right", "arm", exclude=("fore", "hand", "shoulder")),
        "l_fore": find_pbone(arm, "left", "fore"),
        "r_fore": find_pbone(arm, "right", "fore"),
        "l_hand": f("left", "hand"),
        "r_hand": f("right", "hand"),
    }
    # Upper-most spine: the spine bone whose head is farthest from the hips.
    hip_pb = bones["hips"]
    spines = [b for b in arm.pose.bones if "spine" in b.name.lower()]
    if spines and hip_pb is not None:
        hips_h = hip_pb.head.copy()
        spines.sort(key=lambda b: (b.head - hips_h).length, reverse=True)
        bones["spine_hi"] = spines[0]
    elif bones["spine_hi"] is None:
        bones["spine_hi"] = bones["spine_mid"]
    return bones


def world_head(arm, pb):
    if pb is None:
        return None
    return np.asarray(arm.matrix_world @ pb.matrix.to_translation(), np.float64)


def world_tail(arm, pb):
    if pb is None:
        return None
    loc = pb.matrix @ Vector((0.0, pb.length, 0.0))
    return np.asarray(arm.matrix_world @ loc, np.float64)


def world_len(arm, pb):
    if pb is None:
        return 0.0
    return float(np.linalg.norm(world_tail(arm, pb) - world_head(arm, pb)))


def rest_axes(arm, bones):
    """Character right / forward / up from the rest-pose skeleton."""
    lhip = world_head(arm, bones["l_upleg"])
    rhip = world_head(arm, bones["r_upleg"])
    hip = 0.5 * (lhip + rhip)
    chest = world_head(arm, bones["spine_hi"]) if bones["spine_hi"] is not None else None
    if chest is None:
        chest = world_head(arm, bones["neck"]) or (hip + np.array([0, 0, 0.5]))
    up = _n(chest - hip)
    if up[2] < 0:
        up = -up
    right = _n(rhip - lhip)
    right = _n(right - (right @ up) * up)
    forward = _n(np.cross(up, right))
    # Prefer a forward that points the way the chest faces: if a head-front
    # bone exists we ignore it; otherwise assume the rest facing we measured.
    return {
        "hip": hip, "up": up, "right": right, "forward": forward,
        "lhip": lhip, "rhip": rhip, "chest": chest,
        "torso_len": float(np.linalg.norm(chest - hip)),
    }


def ensure_empty(name, loc, size=0.06):
    ob = bpy.data.objects.get(name)
    if ob is None:
        ob = bpy.data.objects.new(name, None)
        ob.empty_display_size = size
        bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector((float(loc[0]), float(loc[1]), float(loc[2])))
    return ob


def clear_bone_constraints(arm, types=("IK", "COPY_LOCATION", "DAMPED_TRACK", "TRACK_TO")):
    for pb in arm.pose.bones:
        for c in list(pb.constraints):
            if c.type in types:
                pb.constraints.remove(c)


def add_ik(pb, target, chain=2, pole=None, pole_angle=0.0, use_tail=True):
    for c in list(pb.constraints):
        if c.type == "IK":
            pb.constraints.remove(c)
    ik = pb.constraints.new("IK")
    ik.target = target
    ik.chain_count = int(chain)
    ik.use_tail = bool(use_tail)
    ik.iterations = 100
    if pole is not None:
        ik.pole_target = pole
        ik.pole_angle = float(pole_angle)
    return ik


def set_hips_world(arm, hips, R3, translate):
    """Rotate Hips in world space about its head, then translate."""
    mw = arm.matrix_world
    wmat = mw @ hips.matrix
    head = wmat.to_translation()
    R4 = Matrix.Rotation(0.0, 4, "Z")
    for i in range(3):
        for j in range(3):
            R4[i][j] = float(R3[i, j]) if hasattr(R3, "__getitem__") else R3[i][j]
    # mathutils from numpy 3x3
    R4 = Matrix((
        (float(R3[0, 0]), float(R3[0, 1]), float(R3[0, 2]), 0.0),
        (float(R3[1, 0]), float(R3[1, 1]), float(R3[1, 2]), 0.0),
        (float(R3[2, 0]), float(R3[2, 1]), float(R3[2, 2]), 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    # Blender matrices are row-major; apply as R @ (p - head) + head
    T1 = Matrix.Translation(-head)
    T2 = Matrix.Translation(head + Vector((float(translate[0]), float(translate[1]), float(translate[2]))))
    new_w = T2 @ R4 @ T1 @ wmat
    hips.matrix = mw.inverted() @ new_w


def rotation_about(axis, angle_rad):
    axis = _n(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ], np.float64)


def _enter_pose(arm):
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    if bpy.context.object.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")


def _leave_pose(arm):
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    bpy.context.view_layer.update()


def measure_lengths(arm, bones):
    return {
        "thigh": world_len(arm, bones["l_upleg"]) or world_len(arm, bones["r_upleg"]),
        "shin": world_len(arm, bones["l_leg"]) or world_len(arm, bones["r_leg"]),
        "upper_arm": world_len(arm, bones["l_arm"]) or world_len(arm, bones["r_arm"]),
        "forearm": world_len(arm, bones["l_fore"]) or world_len(arm, bones["r_fore"]),
    }


def angle_between(a, b):
    a, b = _n(a), _n(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def joint_angle(a, b, c):
    """Interior angle ABC in degrees."""
    return angle_between(a - b, c - b)


def pose_barbell_row(arm, phase="top", hinge_deg=45.0, grip_extra_m=0.04):
    """Textbook standing barbell bent-over row (top = bar to lower ribs).

    phase: 'top' | 'bottom'
    """
    _enter_pose(arm)
    clear_bone_constraints(arm)
    bones = map_bones(arm)
    missing = [k for k, v in bones.items() if v is None and k not in ("l_toe", "r_toe", "spine_lo", "spine_mid")]
    axes = rest_axes(arm, bones)
    fwd, right, up = axes["forward"], axes["right"], np.array([0.0, 0.0, 1.0])
    L = measure_lengths(arm, bones)
    # Record rest ankles / toes before we move anything.
    lank = world_head(arm, bones["l_foot"])
    rank = world_head(arm, bones["r_foot"])
    ltoe = world_head(arm, bones["l_toe"]) if bones["l_toe"] is not None else lank + fwd * 0.18
    rtoe = world_head(arm, bones["r_toe"]) if bones["r_toe"] is not None else rank + fwd * 0.18
    lkn_rest = world_head(arm, bones["l_leg"])
    rkn_rest = world_head(arm, bones["r_leg"])

    # Foot IK (shin bones) so rotating Hips cannot lift the feet.
    e_lank = ensure_empty("ik_lank", lank)
    e_rank = ensure_empty("ik_rank", rank)
    e_lpole = ensure_empty("ik_lpole_leg", lkn_rest + fwd * 0.35 + np.array([0, 0, 0.05]))
    e_rpole = ensure_empty("ik_rpole_leg", rkn_rest + fwd * 0.35 + np.array([0, 0, 0.05]))
    if bones["l_leg"] is not None:
        add_ik(bones["l_leg"], e_lank, chain=2, pole=e_lpole, pole_angle=0.0, use_tail=True)
    if bones["r_leg"] is not None:
        add_ik(bones["r_leg"], e_rank, chain=2, pole=e_rpole, pole_angle=0.0, use_tail=True)

    # Hinge: rotate Hips about the character's right so the spine tips toward forward.
    hinge = math.radians(float(hinge_deg))
    # Rotate about the character's right so the head travels *forward*
    # (right-hand rule around +right would tip the spine backward).
    R = rotation_about(right, -hinge)
    # Hips travel back and slightly down so the COM stays over the midfoot.
    # Small shift only — a 20° soft knee, not a squat. Feet stay IK-locked.
    back_m = 0.09 + 0.04 * math.sin(hinge)
    down_m = 0.015 + 0.015 * math.sin(hinge)
    translate = -fwd * back_m + np.array([0.0, 0.0, -down_m])
    set_hips_world(arm, bones["hips"], R, translate)
    bpy.context.view_layer.update()

    # After the hinge, read posed shoulders / hips.
    lsh = world_head(arm, bones["l_arm"])
    rsh = world_head(arm, bones["r_arm"])
    lhip = world_head(arm, bones["l_upleg"])
    rhip = world_head(arm, bones["r_upleg"])
    hip = 0.5 * (lhip + rhip)
    sh_mid = 0.5 * (lsh + rsh)
    spine_dir = _n(sh_mid - hip)
    torso_len = float(np.linalg.norm(sh_mid - hip)) or axes["torso_len"]
    La, Lf = L["upper_arm"], L["forearm"]
    shoulder_half = 0.5 * float(np.linalg.norm(rsh - lsh))
    grip = shoulder_half + float(grip_extra_m)
    torso_front = _n(fwd * math.cos(hinge) - up * math.sin(hinge))

    rkn = world_head(arm, bones["r_leg"])
    lkn = world_head(arm, bones["l_leg"])

    if phase == "bottom":
        hang_z = min(float(sh_mid[2] - 0.96 * (La + Lf)), float(rkn[2]) - 0.07)
        hang_z = max(hang_z, 0.27)  # keep 45 cm plates off the floor
        l_hand = lsh + (-right) * (grip - shoulder_half) + fwd * 0.05
        r_hand = rsh + right * (grip - shoulder_half) + fwd * 0.05
        l_hand = np.array([l_hand[0], l_hand[1], hang_z])
        r_hand = np.array([r_hand[0], r_hand[1], hang_z])
        l_pole = lsh + (-right) * 0.16 + np.array([0.0, 0.0, -0.28])
        r_pole = rsh + right * 0.16 + np.array([0.0, 0.0, -0.28])
    else:
        ribs = hip + spine_dir * (0.28 * torso_len)
        l_hand = ribs + torso_front * 0.09 + (-right) * grip
        r_hand = ribs + torso_front * 0.09 + right * grip
        l_pole = lsh - torso_front * 0.38 + (-right) * 0.10
        r_pole = rsh - torso_front * 0.38 + right * 0.10

    e_lhand = ensure_empty("ik_lhand", l_hand)
    e_rhand = ensure_empty("ik_rhand", r_hand)
    e_lpole_a = ensure_empty("ik_lpole_arm", l_pole)
    e_rpole_a = ensure_empty("ik_rpole_arm", r_pole)
    if bones["l_fore"] is not None:
        add_ik(bones["l_fore"], e_lhand, chain=2, pole=e_lpole_a, pole_angle=math.radians(180), use_tail=True)
    if bones["r_fore"] is not None:
        add_ik(bones["r_fore"], e_rhand, chain=2, pole=e_rpole_a, pole_angle=math.radians(180), use_tail=True)

    bpy.context.view_layer.update()
    _leave_pose(arm)

    # Re-read after IK solve.
    pts = {
        "lsh": world_head(arm, bones["l_arm"]),
        "rsh": world_head(arm, bones["r_arm"]),
        "lel": world_head(arm, bones["l_fore"]),
        "rel": world_head(arm, bones["r_fore"]),
        "lwr": world_tail(arm, bones["l_fore"]),
        "rwr": world_tail(arm, bones["r_fore"]),
        "lhip": world_head(arm, bones["l_upleg"]),
        "rhip": world_head(arm, bones["r_upleg"]),
        "lkn": world_head(arm, bones["l_leg"]),
        "rkn": world_head(arm, bones["r_leg"]),
        "lank": world_head(arm, bones["l_foot"]),
        "rank": world_head(arm, bones["r_foot"]),
        "head": world_head(arm, bones["head"]),
    }
    hip = 0.5 * (pts["lhip"] + pts["rhip"])
    sh = 0.5 * (pts["lsh"] + pts["rsh"])
    spine_dir = _n(sh - hip)
    torso_from_vert = angle_between(spine_dir, np.array([0.0, 0.0, 1.0]))
    torso_from_horiz = abs(90.0 - torso_from_vert)
    head_spine = angle_between(spine_dir, _n(pts["head"] - sh)) if pts["head"] is not None else None
    r_elbow = joint_angle(pts["rsh"], pts["rel"], pts["rwr"])
    l_elbow = joint_angle(pts["lsh"], pts["lel"], pts["lwr"])
    r_knee = joint_angle(pts["rhip"], pts["rkn"], pts["rank"])
    l_knee = joint_angle(pts["lhip"], pts["lkn"], pts["lank"])
    bar_mid = 0.5 * (np.asarray(l_hand) + np.asarray(r_hand))
    qc = {
        "phase": phase,
        "hinge_deg_requested": float(hinge_deg),
        "torso_from_vertical_deg": round(torso_from_vert, 2),
        "torso_from_horizontal_deg": round(torso_from_horiz, 2),
        "head_spine_deg": None if head_spine is None else round(head_spine, 2),
        "r_elbow_deg": round(r_elbow, 2),
        "l_elbow_deg": round(l_elbow, 2),
        "r_knee_deg": round(r_knee, 2),
        "l_knee_deg": round(l_knee, 2),
        "r_knee_flexion_deg": round(180.0 - r_knee, 2),
        "l_knee_flexion_deg": round(180.0 - l_knee, 2),
        "rank_z": round(float(pts["rank"][2]), 3),
        "lank_z": round(float(pts["lank"][2]), 3),
        "bar_mid": [round(float(x), 3) for x in bar_mid],
        "l_hand": [round(float(x), 3) for x in l_hand],
        "r_hand": [round(float(x), 3) for x in r_hand],
        "grip_half_m": round(float(grip), 3),
        "shoulder_half_m": round(float(shoulder_half), 3),
        "foot_on_floor": bool(pts["rank"][2] < 0.16 and pts["lank"][2] < 0.16),
        "found": {k: (None if v is None else v.name) for k, v in bones.items()},
        "missing": missing,
        "lengths": {k: round(v, 3) for k, v in L.items()},
        "forward": [round(float(x), 3) for x in fwd],
        "right": [round(float(x), 3) for x in right],
    }
    return {
        "ok": bones["hips"] is not None and bones["r_fore"] is not None,
        "qc": qc,
        "bar_mid": bar_mid,
        "l_hand": np.asarray(l_hand, np.float64),
        "r_hand": np.asarray(r_hand, np.float64),
        "fwd": fwd, "right": right, "up": up,
        "spine_dir": spine_dir, "hip": hip, "sh_mid": sh,
        "axes": axes, "bones": bones, "pts": pts,
    }


def pose_bench_db_row(arm, phase="top", pad_z=0.43):
    """Apply textbook bench-supported one-arm DB row landmarks via IK."""
    from bench_row_pose import build_landmarks

    _enter_pose(arm)
    clear_bone_constraints(arm)
    bones = map_bones(arm)
    axes = rest_axes(arm, bones)
    Lmeas = measure_lengths(arm, bones)
    lsh0 = world_head(arm, bones["l_arm"])
    rsh0 = world_head(arm, bones["r_arm"])
    lhip0 = world_head(arm, bones["l_upleg"])
    rhip0 = world_head(arm, bones["r_upleg"])
    L = {
        "thigh": Lmeas["thigh"],
        "shin": Lmeas["shin"],
        "upper_arm": Lmeas["upper_arm"],
        "forearm": Lmeas["forearm"],
        "torso": float(np.linalg.norm(0.5 * (lsh0 + rsh0) - 0.5 * (lhip0 + rhip0))),
        "shoulder_half": 0.5 * float(np.linalg.norm(rsh0 - lsh0)),
        "hip_half": 0.5 * float(np.linalg.norm(rhip0 - lhip0)),
        "pad_z": float(pad_z),
        "hand_ahead_m": 0.55,
    }
    u = np.array([1.0 if phase == "top" else 0.0], np.float64)
    B, spec = build_landmarks(u, L, side=12)
    rec = B[0]
    # Landmark index map (MediaPipe-ish, as written in bench_row_pose).
    tgt = {
        "lhip": rec[23], "rhip": rec[24],
        "lsh": rec[11], "rsh": rec[12],
        "lel": rec[13], "rel": rec[14],
        "lwr": rec[15], "rwr": rec[16],
        "lkn": rec[25], "rkn": rec[26],
        "lank": rec[27], "rank": rec[28],
        "lheel": rec[29], "rheel": rec[30],
        "ltoe": rec[31], "rtoe": rec[32],
        "head": rec[0],
    }

    # Orient Hips so hip_mid -> sh_mid matches the authored torso, then IK limbs.
    hip_t = 0.5 * (tgt["lhip"] + tgt["rhip"])
    sh_t = 0.5 * (tgt["lsh"] + tgt["rsh"])
    want_spine = _n(sh_t - hip_t)
    rest_spine = _n(axes["chest"] - axes["hip"])
    # rotation that takes rest_spine to want_spine, around their cross product
    axis = np.cross(rest_spine, want_spine)
    if np.linalg.norm(axis) < 1e-8:
        R = np.eye(3)
    else:
        ang = math.acos(float(np.clip(np.dot(_n(rest_spine), want_spine), -1, 1)))
        R = rotation_about(axis, ang)
    translate = hip_t - axes["hip"]
    set_hips_world(arm, bones["hips"], R, translate)
    bpy.context.view_layer.update()

    # Limb IK targets
    e_lank = ensure_empty("ik_lank", tgt["lank"])
    e_rank = ensure_empty("ik_rank", tgt["rank"])
    e_lhand = ensure_empty("ik_lhand", tgt["lwr"])
    e_rhand = ensure_empty("ik_rhand", tgt["rwr"])
    e_lpole_l = ensure_empty("ik_lpole_leg", tgt["lkn"] + np.array([0.0, 0.0, 0.15]))
    e_rpole_l = ensure_empty("ik_rpole_leg", tgt["rkn"] + np.array([0.14, 0.0, 0.05]))
    e_lpole_a = ensure_empty("ik_lpole_arm", tgt["lel"] + np.array([0.0, -0.05, -0.10]))
    e_rpole_a = ensure_empty("ik_rpole_arm", tgt["rel"] + np.array([0.08, 0.0, 0.05]))
    if bones["l_leg"] is not None:
        add_ik(bones["l_leg"], e_lank, chain=2, pole=e_lpole_l, use_tail=True)
    if bones["r_leg"] is not None:
        add_ik(bones["r_leg"], e_rank, chain=2, pole=e_rpole_l, use_tail=True)
    if bones["l_fore"] is not None:
        add_ik(bones["l_fore"], e_lhand, chain=2, pole=e_lpole_a, pole_angle=math.radians(180), use_tail=True)
    if bones["r_fore"] is not None:
        add_ik(bones["r_fore"], e_rhand, chain=2, pole=e_rpole_a, pole_angle=math.radians(180), use_tail=True)
    bpy.context.view_layer.update()
    _leave_pose(arm)

    pts = {
        "lsh": world_head(arm, bones["l_arm"]),
        "rsh": world_head(arm, bones["r_arm"]),
        "lel": world_head(arm, bones["l_fore"]),
        "rel": world_head(arm, bones["r_fore"]),
        "lwr": world_tail(arm, bones["l_fore"]),
        "rwr": world_tail(arm, bones["r_fore"]),
        "lhip": world_head(arm, bones["l_upleg"]),
        "rhip": world_head(arm, bones["r_upleg"]),
        "lkn": world_head(arm, bones["l_leg"]),
        "rkn": world_head(arm, bones["r_leg"]),
        "lank": world_head(arm, bones["l_foot"]),
        "rank": world_head(arm, bones["r_foot"]),
        "head": world_head(arm, bones["head"]),
    }
    hip = 0.5 * (pts["lhip"] + pts["rhip"])
    sh = 0.5 * (pts["lsh"] + pts["rsh"])
    spine_dir = _n(sh - hip)
    torso_from_vert = angle_between(spine_dir, np.array([0.0, 0.0, 1.0]))
    qc = {
        "phase": phase,
        "torso_from_vertical_deg": round(torso_from_vert, 2),
        "torso_from_horizontal_deg": round(abs(90.0 - torso_from_vert), 2),
        "head_spine_deg": round(angle_between(spine_dir, _n(pts["head"] - sh)), 2) if pts["head"] is not None else None,
        "r_elbow_deg": round(joint_angle(pts["rsh"], pts["rel"], pts["rwr"]), 2),
        "l_elbow_deg": round(joint_angle(pts["lsh"], pts["lel"], pts["lwr"]), 2),
        "r_knee_deg": round(joint_angle(pts["rhip"], pts["rkn"], pts["rank"]), 2),
        "l_knee_deg": round(joint_angle(pts["lhip"], pts["lkn"], pts["lank"]), 2),
        "lwr_z": round(float(pts["lwr"][2]), 3),
        "lkn_z": round(float(pts["lkn"][2]), 3),
        "rank_z": round(float(pts["rank"][2]), 3),
        "pad_z": float(pad_z),
        "support_hand_on_pad": bool(abs(pts["lwr"][2] - pad_z) < 0.07),
        "support_knee_on_pad": bool(abs(pts["lkn"][2] - pad_z) < 0.09),
        "foot_on_floor": bool(pts["rank"][2] < 0.16),
        "working_hand_drop_m": round(float(pts["rsh"][2] - pts["rwr"][2]), 3),
        "working_hand_hip_m": round(float(np.linalg.norm(pts["rwr"] - pts["rhip"])), 3),
        "found": {k: (None if v is None else v.name) for k, v in bones.items()},
        "lengths": {k: round(v, 3) for k, v in Lmeas.items()},
        "spec_origin": spec.get("origin_xy"),
        "spec_yaw_z": spec.get("yaw_z"),
    }
    return {
        "ok": True,
        "qc": qc,
        "spec": spec,
        "targets": {k: [round(float(x), 3) for x in v] for k, v in tgt.items()},
        "r_hand": pts["rwr"],
        "l_hand": pts["lwr"],
        "bones": bones,
        "pts": pts,
    }
