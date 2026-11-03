"""Seated wide-grip lat pulldown pose: legs on pads, feet flat, hands locked to bar via IK."""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

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


def _bone_len(arm, name):
    b = arm.data.bones[name]
    return float((Vector(b.tail_local) - Vector(b.head_local)).length)


def _world_head(arm, bn):
    return np.array(arm.matrix_world @ arm.pose.bones[bn].head)


def _world_tail(arm, bn):
    return np.array(arm.matrix_world @ arm.pose.bones[bn].tail)


def pose_seated_legs(arm, seat_h=0.42, thigh_clear=0.08):
    """Thighs horizontal under pad, shins vertical, feet flat on z≈0."""
    upd()
    for sd, sx in (("L", 1), ("R", -1)):
        hip = f"upperleg01.{sd}"
        knee = f"upperleg02.{sd}"
        shin = f"lowerleg01.{sd}"
        ankle = f"lowerleg02.{sd}"
        foot = f"foot.{sd}"
        # Femur: forward (-Y) and slightly down from hip
        thigh_dir = unit((0.06 * sx, -0.92, -0.12))
        aim(arm, hip, thigh_dir)
        aim(arm, knee, thigh_dir)
        shin_dir = unit((0.04 * sx, -0.15, -0.98))
        aim(arm, shin, shin_dir)
        aim(arm, ankle, shin_dir)
        foot_dir = unit((0.05 * sx, -0.35, -0.94))
        aim(arm, foot, foot_dir)
    upd()


def _arm_lengths(arm, side):
    sd = side
    l1 = sum(_bone_len(arm, b) for b in (f"upperarm01.{sd}", f"upperarm02.{sd}") if b in arm.data.bones)
    l2 = sum(_bone_len(arm, b) for b in (f"lowerarm01.{sd}", f"lowerarm02.{sd}", f"wrist.{sd}") if b in arm.data.bones)
    return l1, l2


def _solve_arm_to_wrist(arm, side, wrist_target, elbow_pole_bias=0.55):
    """Two-bone analytic IK using aim(); elbow biased forward (-Y) for lat pulldown."""
    sd = side
    sh_bn = f"upperarm01.{sd}"
    S = np.array(arm.matrix_world @ arm.pose.bones[sh_bn].head)
    W = np.asarray(wrist_target, float)
    L1, L2 = _arm_lengths(arm, sd)
    d_vec = W - S
    d = float(np.linalg.norm(d_vec))
    d = min(d, L1 + L2 - 0.008)
    d = max(d, abs(L1 - L2) + 0.008)
    dir_sw = unit(d_vec)
    forward = np.array((0.0, -1.0, 0.0))
    pole = np.cross(dir_sw, np.cross(dir_sw, forward))
    if np.linalg.norm(pole) < 1e-6:
        pole = np.array((1.0 if sd == "R" else -1.0, 0.0, 0.0))
    pole = unit(pole) * elbow_pole_bias
    cos_a = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d + 1e-12)
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    upper_dir = unit(dir_sw * cos_a + pole * sin_a)
    aim(arm, f"upperarm01.{sd}", upper_dir)
    aim(arm, f"upperarm02.{sd}", upper_dir)
    upd()
    E = S + upper_dir * L1
    fore = unit(W - E)
    for bn in (f"lowerarm01.{sd}", f"lowerarm02.{sd}", f"wrist.{sd}"):
        if bn in arm.pose.bones:
            aim(arm, bn, fore)
    upd()


def grip_fingers(arm, side, closed=True):
    curl = math.radians(72 if closed else 25)
    spread = math.radians(8)
    for fn in range(1, 6):
        for seg in (1, 2, 3):
            bn = f"finger{fn}-{seg}.{side}"
            if bn not in arm.pose.bones:
                continue
            pb = arm.pose.bones[bn]
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = (curl, spread * (fn - 3), 0)


def shrug_shoulders(arm, amount=0.0):
    """amount 0..1 for wrong-form trap shrug."""
    if amount <= 0:
        return
    upd()
    for sd in ("L", "R"):
        for bn in (f"shoulder01.{sd}", f"clavicle.{sd}"):
            if bn not in arm.pose.bones:
                continue
            pb = arm.pose.bones[bn]
            M = pb.matrix.copy()
            R = Matrix.Rotation(math.radians(18 * amount), 4, "X")
            pb.matrix = M @ R
    upd()


def pose_torso(arm, lean_deg):
    lean = math.radians(lean_deg)
    Tt = unit((0, -math.sin(lean), math.cos(lean)))
    N = unit((0, math.cos(lean), math.sin(lean)))
    for bn in ("spine05", "spine04", "spine03", "spine02", "spine01"):
        aim(arm, bn, Tt)
    for bn in ("neck01", "neck02", "neck03", "head"):
        aim(arm, bn, unit(Tt * 0.88 + N * 0.22))
    upd()


def bar_grip_points(arm, EX, form, phase, bar_width, seat_z_offset=0.0):
    f0, f1 = EX["forms"][form]["stretch"], EX["forms"][form]["contract"]
    lerp = lambda k: f0[k] + (f1[k] - f0[k]) * phase
    bar_z = lerp("bar_y_m") + seat_z_offset
    lean = math.radians(lerp("trunk_lean_deg"))
    sh_l = _world_head(arm, "upperarm01.L")
    sh_r = _world_head(arm, "upperarm01.R")
    chest = (sh_l + sh_r) / 2
    span = float(np.linalg.norm(sh_r - sh_l))
    L1, L2 = _arm_lengths(arm, "L")
    reach = (L1 + L2) * 0.92
    # Wide grip ≈ 2.1× shoulder span, clamped to arm reach at this bar height
    grip_span = min(bar_width, span * 2.15)
    y_off = -0.22 - 0.14 * phase - 0.05 * math.sin(lean)
    cx = float(chest[0])
    cy = float(chest[1]) + y_off
    cz = bar_z
    half = grip_span / 2
    # If horizontal reach exceeds arm length, pull hands closer
    for _ in range(3):
        ok = True
        for side, sx in (("L", -1), ("R", 1)):
            S = sh_l if side == "L" else sh_r
            W = np.array((cx + sx * half, cy, cz))
            if float(np.linalg.norm(W - S)) > reach:
                ok = False
        if ok:
            break
        half *= 0.92
    left = np.array((cx - half, cy, cz))
    right = np.array((cx + half, cy, cz))
    return left, right, np.array((cx, cy, cz))


def pose_arms_from_standards(arm, EX, form, phase):
    """Elbow + upper-arm angles from motion standards (lat pulldown)."""
    f0, f1 = EX["forms"][form]["stretch"], EX["forms"][form]["contract"]
    lerp = lambda k: f0[k] + (f1[k] - f0[k]) * phase
    lean = math.radians(lerp("trunk_lean_deg"))
    ef = math.radians(lerp("elbow_flexion_deg"))
    fwd = math.radians(lerp("upper_arm_forward_deg"))
    Tt = unit((0, -math.sin(lean), math.cos(lean)))
    N = unit((0, math.cos(lean), math.sin(lean)))
    for sd, sx in (("L", 1), ("R", -1)):
        X = np.array((float(sx), 0.0, 0.0))
        u = unit(-Tt * math.cos(fwd) + X * math.sin(fwd) + N * 0.06)
        w = unit(N - u * np.dot(N, u))
        fa = unit(u * math.cos(ef) + w * math.sin(ef))
        for bn in (f"upperarm01.{sd}", f"upperarm02.{sd}"):
            aim(arm, bn, u)
        for bn in (f"lowerarm01.{sd}", f"lowerarm02.{sd}"):
            if bn in arm.pose.bones:
                aim(arm, bn, fa)
        # wrist follows lowerarm02 (do not aim wrist — breaks MH chain)
    upd()


def pose_seated_lat_pulldown(arm, EX, form, phase, bar_width, wrong_shrug=0.35):
    """Full seated lat pose; returns (bar_center, bar_axis_unit, wl, wr)."""
    f0, f1 = EX["forms"][form]["stretch"], EX["forms"][form]["contract"]
    lerp = lambda k: f0[k] + (f1[k] - f0[k]) * phase
    lean = lerp("trunk_lean_deg")
    pose_seated_legs(arm)
    pose_torso(arm, lean)
    if form == "wrong" and phase > 0.5:
        shrug_shoulders(arm, wrong_shrug)
    elif form == "wrong":
        shrug_shoulders(arm, wrong_shrug * 0.4)
    pose_arms_from_standards(arm, EX, form, phase)
    grip_fingers(arm, "L")
    grip_fingers(arm, "R")
    upd()
    def wrist_pt(sd):
        if f"hand.{sd}" in arm.pose.bones:
            return _world_head(arm, f"hand.{sd}")
        return _world_tail(arm, f"wrist.{sd}")

    wl = wrist_pt("L")
    wr = wrist_pt("R")
    center = (wl + wr) / 2
    axis = wr - wl
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array((1.0, 0.0, 0.0))
    else:
        axis = axis / np.linalg.norm(axis)
    return center, axis, wl, wr


def align_bar_to_grip(bar_root, center, axis, width):
    bar_root.location = Vector(center)
    ax = Vector(axis.tolist()).normalized()
    bar_root.rotation_euler = Vector((1.0, 0.0, 0.0)).rotation_difference(ax).to_euler()
    upd()
