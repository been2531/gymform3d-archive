#!/usr/bin/env python3
"""Procedural flat bench (bpy primitives, no third-party mesh).

Looks:
  warm_grey       — original bb_bench_press pad (distinct from mannequin charcoal)
  gunmetal_black  — dark gunmetal frame + black pad (one-arm row / shared default for db_row)

  nice -n 19 blender -b -P tools/rig3d/props/bench_flat.py -- --no-still
"""
from __future__ import annotations
import argparse, math, os, json
import bpy
import mathutils as mu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOOKS = {
    "warm_grey": {
        "pad": (0.42, 0.40, 0.38),
        "frame": (0.55, 0.56, 0.58),
        "feet": (0.25, 0.25, 0.26),
        "pad_metal": 0.0, "pad_rough": 0.78,
        "frame_metal": 1.0, "frame_rough": 0.40,
    },
    "gunmetal_black": {
        "pad": (0.028, 0.023, 0.021),
        "frame": (0.11, 0.12, 0.135),
        "feet": (0.045, 0.045, 0.048),
        "pad_metal": 0.0, "pad_rough": 0.70,
        "frame_metal": 0.88, "frame_rough": 0.36,
    },
}
# backwards-compat aliases used by the original builder
PAD = LOOKS["warm_grey"]["pad"]
FRAME = LOOKS["warm_grey"]["frame"]
FEET = LOOKS["warm_grey"]["feet"]


def log(m):
    print(f"[bench] {m}", flush=True)


def mat(name, rgb, metal=0.0, rough=0.6):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bs = m.node_tree.nodes.get("Principled BSDF")
    if bs is None:
        m.node_tree.nodes.clear()
        out = m.node_tree.nodes.new("ShaderNodeOutputMaterial")
        bs = m.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        m.node_tree.links.new(bs.outputs["BSDF"], out.inputs["Surface"])
    bs.inputs["Base Color"].default_value = (*rgb, 1)
    bs.inputs["Metallic"].default_value = metal
    bs.inputs["Roughness"].default_value = rough
    if "pad" in name and metal < 0.05:
        if "Specular IOR Level" in bs.inputs:
            bs.inputs["Specular IOR Level"].default_value = 0.42
        if "Sheen Weight" in bs.inputs:
            bs.inputs["Sheen Weight"].default_value = 0.28
            if "Sheen Roughness" in bs.inputs:
                bs.inputs["Sheen Roughness"].default_value = 0.45
    return m


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def box(name, sx, sy, sz, loc, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    ob = bpy.context.object
    ob.name = name
    ob.scale = (sx / 2, sy / 2, sz / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ob.data.materials.append(material)
    return ob


def empty(name, loc):
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=loc)
    e = bpy.context.object
    e.name = name
    e.empty_display_size = 0.06
    return e


def build_bench(look="warm_grey", clear=True):
    """Build (and optionally join) a flat bench. `clear=False` to spawn into an existing scene."""
    if clear:
        clear_scene()
    look = (look or "warm_grey").lower()
    if look not in LOOKS:
        look = "warm_grey"
    L = LOOKS[look]
    pad_m = mat(f"bench_pad_{look}", L["pad"], metal=L["pad_metal"], rough=L["pad_rough"])
    frame_m = mat(f"bench_frame_{look}", L["frame"], metal=L["frame_metal"], rough=L["frame_rough"])
    feet_m = mat(f"bench_feet_{look}", L["feet"], metal=0.25, rough=0.62)

    # dimensions (m): flat bench, pad ~1.25 x 0.30, height ~0.45, pad top 0.49
    # Commercial flat bench, local +X = length.
    # Pad 120 x 30 x 8 cm, top at z=0.43 (origin = pad centre z=0.39).
    PAD_L, PAD_W, PAD_T = 1.20, 0.30, 0.08
    PAD_TOP = 0.43
    PAD_CZ = PAD_TOP - PAD_T / 2.0          # 0.39
    PAD_BOT = PAD_TOP - PAD_T               # 0.35

    pad = box("bench_pad", PAD_L, PAD_W, PAD_T, (0, 0, PAD_CZ), pad_m)
    bpy.context.view_layer.objects.active = pad
    bev = pad.modifiers.new("pad_bevel", "BEVEL")
    bev.width = 0.016
    bev.segments = 4
    bev.limit_method = "ANGLE"
    bev.angle_limit = math.radians(40)
    bpy.ops.object.modifier_apply(modifier="pad_bevel")
    for poly in pad.data.polygons:
        poly.use_smooth = True

    def tube(name, radius, length, loc):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius, depth=length, location=loc,
            rotation=(0.0, math.pi / 2.0, 0.0), vertices=24)
        ob = bpy.context.object
        ob.name = name
        ob.data.materials.append(frame_m)
        for poly in ob.data.polygons:
            poly.use_smooth = True
        return ob

    # Central spine tube under the pad, overlapping the pad bottom and both uprights.
    spine = tube("bench_spine", 0.026, 1.02, (0.0, 0.0, PAD_BOT - 0.004))
    # End cross-members (T into the spine) so the under-pad frame is one piece.
    xbar_h = box("xbar_H", 0.08, 0.22, 0.045, (0.48, 0.0, PAD_BOT - 0.01), frame_m)
    xbar_f = box("xbar_F", 0.08, 0.22, 0.045, (-0.48, 0.0, PAD_BOT - 0.01), frame_m)
    # Mounting plates between pad and frame.
    plate_h = box("plate_H", 0.12, 0.16, 0.016, (0.48, 0.0, PAD_BOT + 0.002), frame_m)
    plate_f = box("plate_F", 0.12, 0.16, 0.016, (-0.48, 0.0, PAD_BOT + 0.002), frame_m)
    # T-uprights: thick along the length so they read from a side/3/4 camera.
    # They run from the floor feet up into the cross-members (no gap).
    up_h = 0.34
    up_cz = up_h / 2.0                       # 0.17, top = 0.34 (into the xbar)
    upright_h = box("upright_H", 0.09, 0.05, up_h, (0.48, 0.0, up_cz), frame_m)
    upright_f = box("upright_F", 0.09, 0.05, up_h, (-0.48, 0.0, up_cz), frame_m)
    # Lower stretcher ties the two T-frames into one solid.
    stretch = box("stretcher", 0.90, 0.04, 0.04, (0.0, 0.0, 0.10), frame_m)
    # T-feet on the floor (visible length + width) + rubber end caps.
    foot_h = box("foot_H", 0.11, 0.40, 0.036, (0.48, 0.0, 0.018), frame_m)
    foot_f = box("foot_F", 0.11, 0.40, 0.036, (-0.48, 0.0, 0.018), frame_m)
    cap_parts = []
    for sx, sy in ((0.48, 0.17), (0.48, -0.17), (-0.48, 0.17), (-0.48, -0.17)):
        cap_parts.append(box(f"cap_{sx:.2f}_{sy:.2f}", 0.08, 0.08, 0.024, (sx, sy, 0.012), feet_m))

    parts = [pad, spine, xbar_h, xbar_f, plate_h, plate_f,
             upright_h, upright_f, stretch, foot_h, foot_f] + cap_parts
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = pad
    bpy.ops.object.join()
    bench = bpy.context.object
    bench.name = "bench_flat"

    anchors = {}
    for name, loc in (
        ("pad_center", (0, 0, PAD_TOP)),
        ("head_end", (0.52, 0, PAD_TOP)),
        ("foot_end", (-0.52, 0, PAD_TOP)),
        ("support_knee", (-0.12, 0.0, PAD_TOP)),
        ("support_hand", (0.32, 0.0, PAD_TOP)),
        ("floor", (0, 0, 0)),
    ):
        e = empty(name, loc)
        e.parent = bench
        anchors[name] = e

    meta = {
        "name": "bench_flat",
        "look": look,
        "pad_size_m": [1.20, 0.30, 0.08],
        "pad_top_z_m": 0.43,
        "pad_rgb": list(L["pad"]),
        "frame_rgb": list(L["frame"]),
        "note": ("Black pad + dark gunmetal frame." if look == "gunmetal_black"
                 else "Warm grey pad — keep distinct from mannequin charcoal (Day 4 skin lock)."),
        "anchors": {k: {"empty": k, "local": list(anchors[k].location)} for k in anchors},
        "licence": "Procedural bpy primitives. No third-party mesh.",
    }
    bench["gymform_prop"] = "bench_flat"
    bench["gymform_look"] = look
    log(f"built {bench.name} look={look}")
    return bench, meta


def place_bench(bench, origin_xy, yaw_z, origin_z=0.39):
    """Put the bench so its object origin (pad centre, z=0.39) sits at origin_xy, z=origin_z.
    Local +X (pad length) aligns with world yaw_z (radians, atan2(fwd.y, fwd.x))."""
    bench.location = (float(origin_xy[0]), float(origin_xy[1]), float(origin_z))
    bench.rotation_euler = (0.0, 0.0, float(yaw_z))
    return bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-still", action="store_true")
    ap.add_argument("--look", default="warm_grey")
    ap.add_argument("--still", default=os.path.join(ROOT, "docs/mission/bench_flat_prop_still.png"))
    args = ap.parse_args()
    bench, meta = build_bench(look=args.look, clear=True)
    meta_path = os.path.join(ROOT, "modules/equipment/bench_flat_build_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    json.dump(meta, open(meta_path, "w"), indent=2)
    if not args.no_still:
        # reuse a tiny still setup
        sc = bpy.context.scene
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = 16
        sc.cycles.use_denoising = True
        sc.render.resolution_x = 900
        sc.render.resolution_y = 600
        world = bpy.data.worlds.new("w") if "w" not in bpy.data.worlds else bpy.data.worlds["w"]
        sc.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        bg.inputs["Color"].default_value = (0.085, 0.095, 0.11, 1)
        bg.inputs["Strength"].default_value = 0.7
        bpy.ops.object.camera_add(location=(1.4, -1.8, 1.0))
        cam = bpy.context.object
        cam.rotation_euler = (mu.Vector((0, 0, 0.3)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.camera = cam
        bpy.ops.object.light_add(type="AREA", location=(1, -1, 2))
        bpy.context.object.data.energy = 200
        bpy.context.object.data.size = 1.2
        sc.render.filepath = os.path.abspath(args.still)
        bpy.ops.render.render(write_still=True)
        log(f"still -> {args.still}")
    print("OK", meta["name"])


if __name__ == "__main__":
    main()
