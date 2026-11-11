"""Lat station layout: stack + pulley BEHIND seat; cable to bar centre; palette-A on stack only."""
from __future__ import annotations

import math

import bpy
from mathutils import Vector


def _box(name, size, loc, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    o.rotation_euler = rot
    return o


def _bevel(ob, w=0.006):
    m = ob.modifiers.new("bevel", "BEVEL")
    m.width = w
    m.segments = 2


def make_lat_station(S, seat_y, bar_center):
    """User faces -Y; stack on +Y (behind back). Pulley above head."""
    e = S["equipment"]
    sh = e["seat_h"]
    parts = []
    back_y = seat_y + 0.52
    stack_h = e["stack_h"]
    top_z = sh + stack_h + 0.15

    parts.append(_box("machine_base", (1.0, 0.85, 0.07), (0, seat_y + 0.05, 0.035)))
    parts.append(_box("machine_seat", (0.38, 0.34, 0.07), (0, seat_y + 0.02, sh)))
    parts.append(_box("machine_seat_back", (0.36, 0.07, 0.45), (0, seat_y + 0.18, sh + 0.2)))
    parts.append(
        _box("machine_thigh", (0.38, 0.22, e["thigh_pad_h"]), (0, seat_y - 0.22, sh - e["thigh_pad_h"] / 2 + 0.06))
    )
    parts.append(_box("machine_stack", (e["stack_w"], 0.32, stack_h), (0, back_y, sh + stack_h / 2 - 0.05)))
    for side, sx in (("L", -0.38), ("R", 0.38)):
        parts.append(_box(f"tower_{side}", (0.08, 0.08, stack_h + 0.55), (sx, back_y - 0.04, sh + (stack_h + 0.55) / 2)))
    parts.append(_box("machine_top_beam", (0.85, 0.1, 0.07), (0, back_y - 0.04, top_z)))

    pulley = _box("machine_pulley", (0.12, 0.12, 0.06), (0, back_y - 0.04, top_z - 0.04))
    attach = Vector(bar_center)
    pulley_z = top_z - 0.06
    cable_len = max(pulley_z - attach.z, 0.25)
    mid_z = attach.z + cable_len / 2
    parts.append(_box("machine_cable", (0.012, 0.012, cable_len), (attach.x, back_y - 0.04, mid_z)))

    for i in range(7):
        z = sh + 0.28 + i * 0.22
        parts.append(_box(f"stack_pin_{i}", (0.42, 0.08, 0.04), (0, back_y, z)))
        parts.append(_box(f"stack_rim_{i}", (0.44, 0.022, 0.02), (0, back_y + 0.18, z)))

    for o in parts:
        if "cable" not in o.name:
            _bevel(o)
    return parts


def make_grip_bar(S, center, axis, grip_span_m):
    """Steel handle only (plates on stack, not floating at face in side view)."""
    b = S["equipment"]["bar"]
    root = bpy.data.objects.new("bar_root", None)
    bpy.context.collection.objects.link(root)
    c = Vector(center.tolist() if hasattr(center, "tolist") else center)
    ax = Vector(axis.tolist() if hasattr(axis, "tolist") else axis).normalized()
    root.location = c
    root.rotation_euler = Vector((1.0, 0.0, 0.0)).rotation_difference(ax).to_euler()
    width = min(grip_span_m, b["width"] * 0.85)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=b["handle_r"], depth=width, location=(0, 0, 0), rotation=(0, math.pi / 2, 0))
    hnd = bpy.context.object
    hnd.name = "bar_handle"
    hnd.parent = root
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=b["handle_r"] * 0.45, depth=width * 0.92, location=(0, 0, 0), rotation=(0, math.pi / 2, 0))
    accent = bpy.context.object
    accent.name = "bar_accent_line"
    accent.parent = root
    bpy.context.view_layer.update()
    return root, [hnd, accent]
