#!/usr/bin/env python3
"""Isolated feasibility frame: écorché asset + pilot fibre/glow shader, no MH rig/pose pipeline.

Run: studio3d/pilot/blender_cli.sh studio3d/pilot/diagnosis/render_feasibility_ecorche.py
Output: output/pilot/diagnosis/feasibility_ecorche_static.png
"""
from __future__ import annotations

import json
import os
import sys

import bpy
from mathutils import Vector

PILOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(PILOT))
sys.path.insert(0, PILOT)

from shaders import assign, mat_muscle_fibre  # noqa: E402

ECORCHE = os.path.join(ROOT, "assets3d", "ecorche.glb")
OUT = os.path.join(ROOT, "output", "pilot", "diagnosis", "feasibility_ecorche_static.png")
META = os.path.join(ROOT, "output", "pilot", "diagnosis", "feasibility_ecorche_meta.json")

BG = (35 / 255, 39 / 255, 43 / 255)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 32
    sc.cycles.use_denoising = False
    sc.render.resolution_x = 1080
    sc.render.resolution_y = 1920
    sc.render.film_transparent = False

    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*BG, 1)

    import addon_utils

    addon_utils.enable("io_scene_gltf2", default_set=True, persistent=True)
    bpy.ops.import_scene.gltf(filepath=ECORCHE)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    # Center and scale to ~1.7m figure
    coords = []
    for o in meshes:
        for v in o.data.vertices:
            coords.append(o.matrix_world @ v.co)
    if coords:
        import numpy as np

        c = np.mean(coords, axis=0)
        for o in meshes:
            o.location -= Vector(c)
        bpy.context.view_layer.update()
        zs = [float((o.matrix_world @ o.data.vertices[0].co).z) for o in meshes if o.data.vertices]
        height = max(zs) - min(zs) if zs else 1.0
        s = 1.65 / max(height, 1e-6)
        root = bpy.data.objects.new("scale", None)
        bpy.context.collection.objects.link(root)
        for o in meshes:
            o.parent = root
        root.scale = (s, s, s)
    bpy.context.view_layer.update()

    base = [48, 50, 54]
    m_base = mat_muscle_fibre("base", base)
    m_glow = mat_muscle_fibre("glow", base, glow_srgb=[32, 210, 120], glow_strength=3.0, mix_glow=0.42)
    glow_n = 0
    for o in meshes:
        if "latissimus" in o.name.lower():
            assign(o, m_glow)
            glow_n += 1
        elif "bone" not in o.name.lower() and "vertebra" not in o.name.lower():
            assign(o, m_base)

    for i, (loc, en) in enumerate(
        [
            ((-3, -3, 4), 120),
            ((3, -3, 4), 120),
            ((0, -4, 3), 100),
            ((0, 3, 3), 80),
        ]
    ):
        ld = bpy.data.lights.new(f"L{i}", "AREA")
        ld.energy = en
        ld.size = 4
        ob = bpy.data.objects.new(f"L{i}", ld)
        bpy.context.collection.objects.link(ob)
        ob.location = Vector(loc)

    cam = bpy.data.cameras.new("cam")
    cam_obj = bpy.data.objects.new("cam", cam)
    bpy.context.collection.objects.link(cam_obj)
    import numpy as np

    coords2 = []
    for o in meshes:
        for v in o.data.vertices:
            coords2.append(o.matrix_world @ v.co)
    arr = np.array(coords2)
    mn, mx = arr.min(0), arr.max(0)
    cen = (mn + mx) / 2
    cam_obj.location = Vector((float(cen[0]), float(cen[1] - 2.8), float(cen[2] + 0.15)))
    tgt = Vector((float(cen[0]), float(cen[1]), float(cen[2])))
    cam_obj.rotation_euler = (tgt - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
    cam.lens = 50
    sc.camera = cam_obj

    sc.render.filepath = OUT
    bpy.ops.render.render(write_still=True)
    meta = {
        "asset": "Charleeh/ecorche CC BY-SA 4.0",
        "path": ECORCHE,
        "mesh_count": len(meshes),
        "glow_meshes": glow_n,
        "shader": "studio3d/pilot/shaders.py mat_muscle_fibre",
        "note": "Static neutral import; no rig, no MH body, no gym pose",
    }
    json.dump(meta, open(META, "w"), indent=2)
    print("WROTE", OUT, meta)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    main()
