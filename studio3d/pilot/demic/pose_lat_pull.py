"""Seated lat pulldown ROM: bar overhead (phase 0) → chest (phase 1). FK/IK without breaking bone chain."""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Vector


def upd():
    bpy.context.view_layer.update()


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def aim(arm, bn, d):
    pb = arm.pose.bones[bn]
    upd()
    M = pb.matrix.copy()
    y = (M.to_3x3() @ Vector((0, 1, 0))).normalized()
    q = y.rotation_difference(Vector(d).normalized())
    R = q.to_matrix() @ M.to_3x3()
    N = R.to_4x4()
    N.translation = M.translation
    pb.matrix = N
    upd()


def pose_legs(arm):
    for sd, sx in (("L", 1), ("R", -1)):
        thigh = unit((0.05 * sx, -0.9, -0.15))
        shin = unit((0.04 * sx, -0.12, -0.98))
        for bn, dr in (
            (f"upperleg01.{sd}", thigh),
            (f"upperleg02.{sd}", thigh),
            (f"lowerleg01.{sd}", shin),
            (f"lowerleg02.{sd}", shin),
            (f"foot.{sd}", unit((0.05 * sx, -0.3, -0.95))),
        ):
            aim(arm, bn, dr)
    upd()


def _bone_len(arm, name):
    b = arm.data.bones[name]
    return float((Vector(b.tail_local) - Vector(b.head_local)).length)


def _arm_lengths(arm, side):
    sd = side
    l1 = sum(_bone_len(arm, b) for b in (f"upperarm01.{sd}", f"upperarm02.{sd}") if b in arm.data.bones)
    l2 = sum(_bone_len(arm, b) for b in (f"lowerarm01.{sd}", f"lowerarm02.{sd}") if b in arm.data.bones)
    return l1, l2


def _wrist_world(arm, sd):
    if f"hand.{sd}" in arm.pose.bones:
        return np.array(arm.matrix_world @ arm.pose.bones[f"hand.{sd}"].tail)
    return np.array(arm.matrix_world @ arm.pose.bones[f"wrist.{sd}"].tail)


def solve_arm(arm, side, wrist_w, pole_bias):
    """Two-bone IK via aim(); do NOT aim wrist — keeps lowerarm02→wrist chain connected."""
    sd = side
    S = np.array(arm.matrix_world @ arm.pose.bones[f"upperarm01.{sd}"].head)
    W = np.asarray(wrist_w, float)
    L1, L2 = _arm_lengths(arm, sd)
    d = float(np.linalg.norm(W - S))
    d = np.clip(d, abs(L1 - L2) + 0.008, L1 + L2 - 0.008)
    dir_sw = unit(W - S)
    down = np.array((0.0, 0.0, -1.0))
    pole = np.cross(dir_sw, np.cross(dir_sw, down))
    if np.linalg.norm(pole) < 1e-6:
        pole = np.array((1.0 if sd == "R" else -1.0, 0.0, 0.0))
    pole = (unit(pole) + np.array((0.35 * (1 if sd == "R" else -1), 0.0, 0.0))) * pole_bias
    pole = unit(pole)
    cos_a = np.clip((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d + 1e-12), -1, 1)
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    upper = unit(dir_sw * cos_a + pole * sin_a)
    aim(arm, f"upperarm01.{sd}", upper)
    aim(arm, f"upperarm02.{sd}", upper)
    upd()
    E = S + upper * L1
    fore = unit(W - E)
    for bn in (f"lowerarm01.{sd}", f"lowerarm02.{sd}"):
        if bn in arm.pose.bones:
            aim(arm, bn, fore)
    upd()


def grip_fingers(arm):
    for sd in ("L", "R"):
        for fn in range(1, 6):
            for seg in (1, 2, 3):
                bn = f"finger{fn}-{seg}.{sd}"
                if bn in arm.pose.bones:
                    pb = arm.pose.bones[bn]
                    pb.rotation_mode = "XYZ"
                    pb.rotation_euler = (math.radians(70), 0, 0)


def _grip_targets(arm, motion_cfg, form, phase):
    z0 = motion_cfg["stretch_bar_z_m"]
    z1 = motion_cfg["right_contract_bar_z_m"] if form == "right" else motion_cfg["wrong_contract_bar_z_m"]
    bar_z = z0 + (z1 - z0) * phase
    y = motion_cfg["bar_front_y_m"]
    half = motion_cfg["grip_span_m"] / 2
    sh_l = np.array(arm.matrix_world @ arm.pose.bones["upperarm01.L"].head)
    sh_r = np.array(arm.matrix_world @ arm.pose.bones["upperarm01.R"].head)
    cx = float((sh_l[0] + sh_r[0]) / 2)
    cy = float((sh_l[1] + sh_r[1]) / 2) + y
    L1, L2 = _arm_lengths(arm, "L")
    reach = (L1 + L2) * 0.98
    for _ in range(8):
        ok = True
        for S, sx in ((sh_l, -1), (sh_r, 1)):
            W = np.array((cx + sx * half, cy, bar_z))
            if float(np.linalg.norm(W - S)) > reach:
                ok = False
        if ok:
            break
        half *= 0.94
        bar_z -= 0.02
    wl = np.array((cx - half, cy, bar_z))
    wr = np.array((cx + half, cy, bar_z))
    return wl, wr, bar_z


def pose_lat_pull(arm, EX, form, phase, motion_cfg):
    f0, f1 = EX["forms"][form]["stretch"], EX["forms"][form]["contract"]
    lerp = lambda k: f0[k] + (f1[k] - f0[k]) * phase
    lean = lerp("trunk_lean_deg")
    if form == "wrong":
        lean = lerp("trunk_lean_deg") + 4 * phase
    lean_r = math.radians(lean)
    Tt = unit((0, -math.sin(lean_r), math.cos(lean_r)))
    N = unit((0, math.cos(lean_r), math.sin(lean_r)))
    pose_legs(arm)
    for bn in ("spine05", "spine04", "spine03", "spine02", "spine01"):
        aim(arm, bn, Tt)
    for bn in ("neck01", "neck02", "neck03", "head"):
        aim(arm, bn, unit(Tt * 0.9 + N * 0.15))
    if form == "wrong" and phase > 0.4:
        for sd in ("L", "R"):
            for bn in (f"shoulder01.{sd}", f"clavicle.{sd}"):
                if bn in arm.pose.bones:
                    pb = arm.pose.bones[bn]
                    pb.rotation_euler[0] += math.radians(12 * phase)

    wl_tgt, wr_tgt, bar_z = _grip_targets(arm, motion_cfg, form, phase)
    pole = 0.35 if form == "right" else 0.55
    solve_arm(arm, "L", wl_tgt, pole)
    solve_arm(arm, "R", wr_tgt, pole)
    grip_fingers(arm)
    wl = _wrist_world(arm, "L")
    wr = _wrist_world(arm, "R")
    center = (wl + wr) / 2
    axis = unit(wr - wl)
    return center, axis, wl, wr, bar_z
