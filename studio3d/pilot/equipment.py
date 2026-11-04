"""Complete lat pulldown station: seat, thigh pad, frame, pulley, cable, palette-A stack."""
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


def make_lat_machine(S, seat_y=0.0, bar_attach=None):
    e = S["equipment"]
    sh = e["seat_h"]
    parts = []
    sy = seat_y + 0.58
    stack_h = e["stack_h"]
    top_z = sh + stack_h - 0.08

    parts.append(_box("machine_base", (0.95, 0.72, 0.07), (0, sy - 0.08, 0.035)))
    parts.append(_box("machine_stack", (e["stack_w"], 0.38, stack_h), (0, sy + 0.02, sh + stack_h / 2 - 0.12)))
    for side, sx in (("L", -0.42), ("R", 0.42)):
        parts.append(_box(f"machine_tower_{side}", (0.09, 0.09, stack_h + 0.35), (sx, sy - 0.05, sh + (stack_h + 0.35) / 2 - 0.1)))
        parts.append(_box(f"machine_foot_{side}", (0.14, 0.14, 0.05), (sx, sy - 0.05, 0.025)))
    parts.append(_box("machine_top_beam", (0.92, 0.12, 0.07), (0, sy - 0.05, top_z)))
    parts.append(_box("machine_seat", (0.38, 0.36, 0.07), (0, seat_y + 0.02, sh)))
    parts.append(_box("machine_seat_back", (0.38, 0.06, 0.42), (0, seat_y + 0.2, sh + 0.18)))
    parts.append(
        _box(
            "machine_thigh",
            (0.38, 0.24, e["thigh_pad_h"]),
            (0, seat_y - 0.24, sh - e["thigh_pad_h"] / 2 + 0.05),
        )
    )
    n_pins = 6
    for i in range(n_pins):
        z = sh + 0.32 + i * 0.21
        parts.append(_box(f"stack_pin_{i}", (0.44, 0.09, 0.045), (0, sy + 0.02, z)))
        parts.append(_box(f"stack_rim_{i}", (0.46, 0.025, 0.022), (0, sy + 0.21, z)))

    pulley_y = sy - 0.06
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.055, depth=0.11, location=(0, pulley_y, top_z - 0.04))
    pulley = bpy.context.object
    pulley.name = "machine_pulley"
    pulley.rotation_euler = (math.pi / 2, 0, 0)
    parts.append(pulley)

    if bar_attach is None:
        bar_attach = Vector((0, seat_y - 0.32, sh + 1.05))
    attach = Vector(bar_attach)
    mid_z = (top_z - 0.15 + attach.z) / 2
    parts.append(_box("machine_cable_v", (0.018, 0.018, top_z - 0.15 - mid_z), (0, pulley_y, mid_z + (top_z - 0.15 - mid_z) / 2)))
    cy = (pulley_y + attach.y) / 2
    parts.append(_box("machine_cable_h", (0.018, abs(pulley_y - attach.y), 0.018), (0, cy, attach.z)))
    parts.append(_box("machine_cable_drop", (0.018, 0.018, 0.12), (attach.x, attach.y, attach.z + 0.06)))
    return parts


def make_bar(S, center, axis, bar_root_name="bar_root"):
    b = S["equipment"]["bar"]
    root = bpy.data.objects.new(bar_root_name, None)
    bpy.context.collection.objects.link(root)
    align_bar_to_grip(root, center, axis, b["width"])
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=b["handle_r"], depth=b["width"], location=(0, 0, 0), rotation=(0, math.pi / 2, 0))
    hnd = bpy.context.object
    hnd.name = "bar_handle"
    hnd.parent = root
    bpy.context.view_layer.update()
    return root, [hnd]


def align_bar_to_grip(bar_root, center, axis, width):
    bar_root.location = Vector(center) if not isinstance(center, Vector) else center
    ax = Vector(axis.tolist() if hasattr(axis, "tolist") else axis).normalized()
    bar_root.rotation_euler = Vector((1.0, 0.0, 0.0)).rotation_difference(ax).to_euler()
    bpy.context.view_layer.update()
