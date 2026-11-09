#!/usr/bin/env python3
"""Procedural Olympic barbell (palette A) for @gymform.3d rig3d.

Build only — does NOT touch retarget_render_v2 / grade3d. Wire-in later.
0 generations. No paid assets. CC0 geometry from bpy primitives.

  nice -n 19 /workspace/.bpyenv/bin/python tools/rig3d/props/barbell_olympic.py
  ... --still docs/mission/barbell_olympic_prop_still.png
  ... --export _scratch/props/barbell_olympic.glb   # gitignored

Units: metres. Local +X = bar long axis. Empties = two-hand grip attachments.
"""
from __future__ import annotations
import argparse, math, os, sys

import bpy
import mathutils as mu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Olympic-ish visual proportions (not a CAD replica)
BAR_LEN = 2.20
SHAFT_R = 0.014
SLEEVE_R = 0.025
SLEEVE_LEN = 0.40
COLLAR_R = 0.032
COLLAR_W = 0.028
# plates sit on sleeves; outer face of plate ~ bar end
PLATE_R = 0.225          # ~45 cm diameter look
PLATE_T = 0.045
GOLD_RING_T = 0.004
GOLD_RING_W = 0.010
# dual-hand grips on the shaft (inside collars); reusable by any barbell exercise
GRIP_INBOARD = 0.40      # |x| from centre — default squat/deadlift feel
GRIP_WIDE = 0.55         # bench-ish
KNURL_BANDS = ((0.18, 0.34), (0.46, 0.62))  # |x| ranges with knurl (both sides mirrored)

COBALT = (0.02, 0.05, 0.45)
GOLD = (0.72, 0.55, 0.18)
STEEL = (0.55, 0.55, 0.58)
STEEL_DARK = (0.35, 0.35, 0.38)


def log(m):
    print(f"[barbell] {m}", flush=True)


def mat(name, rgb, metal=0.0, rough=0.55):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bs.inputs["Base Color"].default_value = (*rgb, 1.0)
    bs.inputs["Metallic"].default_value = metal
    bs.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in bs.inputs:
        bs.inputs["Specular IOR Level"].default_value = 0.5
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])
    return m


def mat_knurled_steel(name="bar_steel_knurl"):
    """Brushed steel + procedural cross-hatch bump for knurl bands (applied mesh-wide; knurl is visual)."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bs.location = (400, 0)
    out.location = (650, 0)
    bs.inputs["Base Color"].default_value = (*STEEL, 1.0)
    bs.inputs["Metallic"].default_value = 1.0
    bs.inputs["Roughness"].default_value = 0.38
    # procedural knurl: two rotated wave textures -> bump
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    map1 = nt.nodes.new("ShaderNodeMapping")
    map2 = nt.nodes.new("ShaderNodeMapping")
    map1.inputs["Scale"].default_value = (1, 90, 1)
    map2.inputs["Scale"].default_value = (1, 90, 1)
    map2.inputs["Rotation"].default_value = (0, 0, math.radians(45))
    w1 = nt.nodes.new("ShaderNodeTexWave")
    w2 = nt.nodes.new("ShaderNodeTexWave")
    for w in (w1, w2):
        w.wave_type = "BANDS"
        w.bands_direction = "Y"
        w.inputs["Scale"].default_value = 18.0
        w.inputs["Distortion"].default_value = 0.0
        w.inputs["Detail"].default_value = 0.0
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    mix.inputs["Factor"].default_value = 0.5
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0015
    nt.links.new(tex_coord.outputs["Object"], map1.inputs["Vector"])
    nt.links.new(tex_coord.outputs["Object"], map2.inputs["Vector"])
    nt.links.new(map1.outputs["Vector"], w1.inputs["Vector"])
    nt.links.new(map2.outputs["Vector"], w2.inputs["Vector"])
    nt.links.new(w1.outputs["Fac"], mix.inputs["A"])
    nt.links.new(w2.outputs["Fac"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bs.inputs["Normal"])
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])
    return m


def cylinder(name, radius, depth, loc=(0, 0, 0), rot=(0, math.pi / 2, 0), verts=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    ob = bpy.context.object
    ob.name = name
    return ob


def empty(name, loc):
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=loc)
    e = bpy.context.object
    e.name = name
    e.empty_display_size = 0.05
    return e


def join_selected(active, others, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in others:
        o.select_set(True)
    active.select_set(True)
    bpy.context.view_layer.objects.active = active
    bpy.ops.object.join()
    active.name = name
    return active


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for b in list(bpy.data.meshes):
        bpy.data.meshes.remove(b)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)


def build_barbell(n_plates_each=2):
    clear_scene()
    cobalt = mat("paletteA_cobalt", COBALT, metal=0.05, rough=0.72)
    gold = mat("paletteA_gold_ring", GOLD, metal=1.0, rough=0.28)
    steel = mat_knurled_steel()
    steel_sleeve = mat("bar_sleeve_steel", STEEL_DARK, metal=1.0, rough=0.42)
    collar_m = mat("bar_collar_steel", STEEL, metal=1.0, rough=0.32)

    # shaft (knurled steel look)
    shaft = cylinder("bar_shaft", SHAFT_R, BAR_LEN - 2 * SLEEVE_LEN, verts=64)
    shaft.data.materials.append(steel)

    # sleeves (left / right)
    sleeve_x = (BAR_LEN / 2 - SLEEVE_LEN / 2)
    sl = cylinder("sleeve_L", SLEEVE_R, SLEEVE_LEN, loc=(-sleeve_x, 0, 0))
    sr = cylinder("sleeve_R", SLEEVE_R, SLEEVE_LEN, loc=(sleeve_x, 0, 0))
    for s in (sl, sr):
        s.data.materials.append(steel_sleeve)

    # collars just inside sleeves
    collar_x = BAR_LEN / 2 - SLEEVE_LEN - COLLAR_W / 2
    cl = cylinder("collar_L", COLLAR_R, COLLAR_W, loc=(-collar_x, 0, 0), verts=48)
    cr = cylinder("collar_R", COLLAR_R, COLLAR_W, loc=(collar_x, 0, 0), verts=48)
    for c in (cl, cr):
        c.data.materials.append(collar_m)

    plate_parts = []
    for side, sx in (("L", -1.0), ("R", 1.0)):
        # stack plates from collar outward
        base = collar_x + COLLAR_W / 2 + PLATE_T / 2
        for i in range(n_plates_each):
            x = sx * (base + i * (PLATE_T + 0.004))
            p = cylinder(f"plate_{side}_{i}", PLATE_R, PLATE_T, loc=(x, 0, 0), verts=64)
            p.data.materials.append(cobalt)
            # thin brushed-gold ring on the OUTER face of the outermost plate; inner plates get a thin face ring too
            ring_x = x + sx * (PLATE_T / 2 + GOLD_RING_T / 2)
            # ring as a thin torus flattened ≈ annulus on plate face
            bpy.ops.mesh.primitive_torus_add(
                major_radius=PLATE_R - GOLD_RING_W,
                minor_radius=GOLD_RING_W * 0.45,
                major_segments=64,
                minor_segments=12,
                location=(ring_x, 0, 0),
                rotation=(0, math.pi / 2, 0),
            )
            ring = bpy.context.object
            ring.name = f"gold_ring_{side}_{i}"
            ring.data.materials.append(gold)
            # squash minor axis slightly so it reads as a flat face ring
            ring.scale = (1.0, 1.0, 0.35)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            plate_parts.extend([p, ring])

    # join rigid bar assembly (shaft+sleeves+collars+plates+rings)
    parts = [shaft, sl, sr, cl, cr] + plate_parts
    barbell = join_selected(shaft, [p for p in parts if p != shaft], "barbell_olympic")
    # apply rotation so local +X is the long axis (cylinder was rotated 90° about Y)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # attachment empties (parented to barbell) — two-hand grips + centre
    grips = {}
    for name, x in (
        ("grip.L", -GRIP_INBOARD),
        ("grip.R", GRIP_INBOARD),
        ("grip_wide.L", -GRIP_WIDE),
        ("grip_wide.R", GRIP_WIDE),
        ("bar_center", 0.0),
        ("collar.L", -collar_x),
        ("collar.R", collar_x),
    ):
        e = empty(name, (x, 0, 0))
        e.parent = barbell
        grips[name] = e

    # collection
    col = bpy.data.collections.get("prop_barbell_olympic") or bpy.data.collections.new("prop_barbell_olympic")
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    for ob in [barbell] + list(grips.values()):
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        col.objects.link(ob)

    meta = {
        "name": "barbell_olympic",
        "bar_len_m": BAR_LEN,
        "shaft_r_m": SHAFT_R,
        "plate_r_m": PLATE_R,
        "plate_t_m": PLATE_T,
        "n_plates_each": n_plates_each,
        "handle_axis_local": [1.0, 0.0, 0.0],
        "grips": {
            "left": {"empty": "grip.L", "local": [-GRIP_INBOARD, 0.0, 0.0]},
            "right": {"empty": "grip.R", "local": [GRIP_INBOARD, 0.0, 0.0]},
            "wide_left": {"empty": "grip_wide.L", "local": [-GRIP_WIDE, 0.0, 0.0]},
            "wide_right": {"empty": "grip_wide.R", "local": [GRIP_WIDE, 0.0, 0.0]},
            "center": {"empty": "bar_center", "local": [0.0, 0.0, 0.0]},
        },
        "materials": {
            "plates": "paletteA_cobalt matte",
            "ring": "paletteA thin brushed gold",
            "bar": "brushed steel + procedural knurl bump",
        },
        "licence": "Procedural bpy primitives; palette A colours match docs/plate_palette_A_reference.jpg. No third-party mesh.",
    }
    barbell["gymform_prop"] = "barbell_olympic"
    log(f"built {barbell.name}: {len(barbell.data.vertices)} verts, grips={list(grips)}")
    return barbell, grips, meta


def setup_still_camera(target_len=2.2):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 1080
    sc.render.resolution_y = 720
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    # world
    world = bpy.data.worlds.new("prop_world") if "prop_world" not in bpy.data.worlds else bpy.data.worlds["prop_world"]
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.085, 0.095, 0.11, 1)
    bg.inputs["Strength"].default_value = 0.6
    # camera 3/4
    bpy.ops.object.camera_add(location=(1.6, -2.4, 0.55))
    cam = bpy.context.object
    cam.name = "prop_cam"
    cam.data.lens = 50
    # aim at origin
    direction = mu.Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    # soft key + fill
    bpy.ops.object.light_add(type="AREA", location=(1.2, -1.0, 1.8))
    key = bpy.context.object
    key.data.energy = 250
    key.data.size = 1.5
    bpy.ops.object.light_add(type="AREA", location=(-1.5, 1.2, 0.8))
    fill = bpy.context.object
    fill.data.energy = 80
    fill.data.size = 2.0
    # subtle floor for contact shadow
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0, 0, -PLATE_R - 0.002))
    floor = bpy.context.object
    floor.name = "prop_floor"
    fm = mat("prop_floor", (0.06, 0.07, 0.08), metal=0.0, rough=0.85)
    floor.data.materials.append(fm)
    return cam


def render_still(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    setup_still_camera()
    sc = bpy.context.scene
    sc.render.filepath = os.path.abspath(path)
    log(f"render still -> {sc.render.filepath}")
    bpy.ops.render.render(write_still=True)
    log("still done")


def export_glb(path, barbell):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    barbell.select_set(True)
    for ch in barbell.children:
        ch.select_set(True)
    bpy.context.view_layer.objects.active = barbell
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(path), use_selection=True, export_format="GLB")
    log(f"exported {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default=os.path.join(ROOT, "docs/mission/barbell_olympic_prop_still.png"))
    ap.add_argument("--export", default="")
    ap.add_argument("--plates", type=int, default=2, help="plates per side")
    ap.add_argument("--no-still", action="store_true")
    args = ap.parse_args()
    barbell, grips, meta = build_barbell(n_plates_each=max(1, args.plates))
    meta_path = os.path.join(ROOT, "modules/equipment/barbell_olympic_build_meta.json")
    import json
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    json.dump(meta, open(meta_path, "w"), indent=2)
    log(f"wrote {meta_path}")
    if args.export:
        export_glb(args.export, barbell)
    if not args.no_still:
        render_still(args.still)
    print("OK", meta["name"], "grips", list(meta["grips"]))


if __name__ == "__main__":
    main()
