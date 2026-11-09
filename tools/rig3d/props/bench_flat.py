#!/usr/bin/env python3
"""Procedural flat bench (warm grey) for barbell bench press. Separate from mannequin charcoal.

  nice -n 19 /workspace/.bpyenv/bin/python tools/rig3d/props/bench_flat.py --no-still
"""
from __future__ import annotations
import argparse, math, os, json
import bpy
import mathutils as mu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# warm grey pad (palette A bench), cooler steel frame — NOT mannequin charcoal
PAD = (0.42, 0.40, 0.38)
FRAME = (0.55, 0.56, 0.58)
FEET = (0.25, 0.25, 0.26)


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


def build_bench():
    clear_scene()
    pad_m = mat("bench_pad_warm_grey", PAD, metal=0.0, rough=0.78)
    frame_m = mat("bench_frame_steel", FRAME, metal=1.0, rough=0.4)
    feet_m = mat("bench_feet", FEET, metal=0.2, rough=0.7)

    # dimensions (m): flat bench, pad ~1.25 x 0.30, height ~0.45
    pad = box("bench_pad", 1.25, 0.30, 0.08, (0, 0, 0.45), pad_m)
    rail = box("bench_rail", 1.10, 0.06, 0.06, (0, 0, 0.38), frame_m)
    leg_f = box("leg_F", 0.06, 0.28, 0.38, (0.45, 0, 0.19), frame_m)
    leg_b = box("leg_B", 0.06, 0.28, 0.38, (-0.45, 0, 0.19), frame_m)
    foot_f = box("foot_F", 0.10, 0.40, 0.04, (0.45, 0, 0.02), feet_m)
    foot_b = box("foot_B", 0.10, 0.40, 0.04, (-0.45, 0, 0.02), feet_m)

    parts = [pad, rail, leg_f, leg_b, foot_f, foot_b]
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = pad
    bpy.ops.object.join()
    bench = bpy.context.object
    bench.name = "bench_flat"

    anchors = {}
    for name, loc in (
        ("pad_center", (0, 0, 0.49)),
        ("head_end", (0.55, 0, 0.49)),
        ("foot_end", (-0.55, 0, 0.49)),
        ("floor", (0, 0, 0)),
    ):
        e = empty(name, loc)
        e.parent = bench
        anchors[name] = e

    meta = {
        "name": "bench_flat",
        "pad_size_m": [1.25, 0.30, 0.08],
        "pad_top_z_m": 0.49,
        "pad_rgb_warm_grey": list(PAD),
        "note": "Warm grey pad — keep distinct from mannequin charcoal (Day 4 skin lock).",
        "anchors": {k: {"empty": k} for k in anchors},
        "licence": "Procedural bpy primitives. No third-party mesh.",
    }
    bench["gymform_prop"] = "bench_flat"
    log(f"built {bench.name}")
    return bench, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-still", action="store_true")
    ap.add_argument("--still", default=os.path.join(ROOT, "docs/mission/bench_flat_prop_still.png"))
    args = ap.parse_args()
    bench, meta = build_bench()
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
