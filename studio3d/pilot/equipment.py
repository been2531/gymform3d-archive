"""Lat pulldown machine — stack behind the athlete, palette-A weight pins, steel bar (no floating end discs)."""
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


def make_lat_machine(S, seat_y=0.0):
    e = S["equipment"]
    sh = e["seat_h"]
    parts = []
    # Athlete faces -Y; stack sits behind the back (+Y)
    sy = seat_y + 0.62
    parts.append(_box("machine_base", (0.55, 0.42, 0.06), (0, sy - 0.05, 0.03)))
    stack_h = e["stack_h"]
    parts.append(_box("machine_stack", (e["stack_w"], 0.36, stack_h), (0, sy, sh + stack_h / 2 - 0.12)))
    # Palette-A weight pins (cobalt + gold cap) — only on stack, not on bar
    bar_cfg = e["bar"]
    n_pins = 5
    for i in range(n_pins):
        z = sh + 0.35 + i * 0.22
        parts.append(_box(f"stack_pin_{i}", (0.42, 0.08, 0.04), (0, sy, z)))
        parts.append(_box(f"stack_rim_{i}", (0.44, 0.02, 0.02), (0, sy + 0.19, z)))
    parts.append(_box("machine_seat", (0.36, 0.34, 0.06), (0, seat_y, sh)))
    parts.append(_box("machine_thigh", (0.36, 0.22, e["thigh_pad_h"]), (0, seat_y - 0.22, sh - e["thigh_pad_h"] / 2 + 0.04)))
    parts.append(_box("machine_post", (0.08, 0.08, 1.15), (0, sy - 0.08, sh + 0.55)))
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.05, depth=0.1, location=(0, sy - 0.02, sh + stack_h - 0.15))
    parts.append(bpy.context.object)
    parts[-1].name = "machine_pulley"
    # Cable (thin box) from pulley toward bar line
    parts.append(_box("machine_cable", (0.015, 0.015, 0.85), (0, sy - 0.12, sh + stack_h - 0.55)))
    return parts


def make_bar(S, arm, bar_root_name="bar_root"):
    b = S["equipment"]["bar"]
    root = bpy.data.objects.new(bar_root_name, None)
    bpy.context.collection.objects.link(root)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=b["handle_r"], depth=b["width"], location=(0, 0, 0), rotation=(0, math.pi / 2, 0))
    hnd = bpy.context.object
    hnd.name = "bar_handle"
    hnd.parent = root
    upd = bpy.context.view_layer.update
    upd()
    wr_l = arm.matrix_world @ arm.pose.bones["wrist.L"].tail
    wr_r = arm.matrix_world @ arm.pose.bones["wrist.R"].tail
    center = (wr_l + wr_r) / 2
    root.location = center
    axis = wr_r - wr_l
    if axis.length > 1e-6:
        axis.normalize()
        root.rotation_euler = Vector((1.0, 0, 0)).rotation_difference(axis).to_euler()
    upd()
    return root, [hnd]
