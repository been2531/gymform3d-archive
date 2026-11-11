#!/usr/bin/env python3
"""Inspect a downloaded rigged character (glTF/FBX): rig, bones, rest pose, mesh closure.

  nice -n 19 blender -b --python tools/rig3d/inspect_boxer_candidate.py -- \
      --input <scene.gltf|.fbx> --out <facts.json>

Read-only research helper (BOXER_OPTIONS.md). Writes only the facts JSON.
"""
import bpy, bmesh, json, sys, math, argparse
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
if a.input.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=a.input)
else:
    bpy.ops.import_scene.gltf(filepath=a.input)
bpy.context.view_layer.update()

arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
facts = {"input": a.input, "armatures": [], "meshes": []}

def world_bbox(objs):
    mn = Vector((1e9,) * 3); mx = Vector((-1e9,) * 3)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        for v in me.vertices:
            w = o.matrix_world @ v.co
            mn = Vector(map(min, mn, w)); mx = Vector(map(max, mx, w))
        oe.to_mesh_clear()
    return mn, mx

# rest pose: clear animation so we see the bind
for arm in arms:
    if arm.animation_data:
        arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
bpy.context.view_layer.update()

for arm in arms:
    bones = [b.name for b in arm.data.bones]
    rec = {"name": arm.name, "n_bones": len(bones), "bones_sample": bones[:80],
           "deform": sum(1 for b in arm.data.bones if b.use_deform),
           "actions": [ac.name for ac in bpy.data.actions]}
    def find(keys):
        for b in arm.data.bones:
            n = b.name.lower().replace("_", "").replace(".", "").replace(":", "")
            if all(k in n for k in keys) and "fore" not in n and "twist" not in n:
                return b
    # upper-arm angle vs horizontal (T ≈ 0°, A ≈ 30–50°)
    for side, keys in (("L", ["left", "arm"]), ("R", ["right", "arm"])):
        b = find(keys) or find(["upperarm", side.lower()]) or find(["arm" + side.lower()])
        if b:
            d = (arm.matrix_world.to_3x3() @ (b.tail_local - b.head_local)).normalized()
            rec[f"upperarm_{side}"] = b.name
            rec[f"upperarm_{side}_drop_deg"] = round(math.degrees(math.asin(max(-1, min(1, -d.z)))), 1)
    rec["mixamo_names"] = any("mixamorig" in n for n in bones)
    facts["armatures"].append(rec)

tot_parts = 0
for m in meshes:
    bm = bmesh.new(); bm.from_mesh(m.data)
    bm.transform(m.matrix_world)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bnd = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    nonman = sum(1 for e in bm.edges if not e.is_manifold)
    # loose parts
    seen = set(); parts = 0
    for f in bm.faces:
        if f.index in seen: continue
        parts += 1; stack = [f]; seen.add(f.index)
        while stack:
            g = stack.pop()
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index); stack.append(h)
    tot_parts += parts
    skinned = any(md.type == "ARMATURE" for md in m.modifiers) or m.parent_type == "ARMATURE"
    facts["meshes"].append({"name": m.name, "verts": len(m.data.vertices), "tris": sum(len(p.vertices) - 2 for p in m.data.polygons),
        "parts_after_weld": parts, "boundary": bnd, "non_manifold": nonman,
        "vgroups": len(m.vertex_groups), "skinned": skinned,
        "materials": [s.material.name for s in m.material_slots if s.material]})
    bm.free()
mn, mx = world_bbox(meshes)
facts["bbox_min"] = list(mn); facts["bbox_max"] = list(mx)
facts["height_z"] = mx.z - mn.z; facts["height_y"] = mx.y - mn.y
facts["total_parts"] = tot_parts
json.dump(facts, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in facts.items() if k != "meshes"}, indent=1)[:3000])
for m in facts["meshes"]:
    print(m)
