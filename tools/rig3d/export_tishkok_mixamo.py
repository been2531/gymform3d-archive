#!/usr/bin/env python3
"""Prepare Tishkok Stylized Boxer as a Mixamo auto-rigger upload mesh.

  nice -n 19 blender -b --python tools/rig3d/export_tishkok_mixamo.py

Writes (gitignored):
  output/_basemodels/boxer/mixamo_upload/tishkok_boxer_mixamo.fbx
  output/_basemodels/boxer/mixamo_upload/tishkok_boxer_mixamo.obj
  output/_basemodels/boxer/mixamo_upload/tishkok_boxer_mixamo.zip
"""
from __future__ import annotations

import json
import math
import os
import sys
import zipfile

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "output/_basemodels/boxer/tishkok_stylized_boxer/scene.gltf")
OUT_DIR = os.path.join(ROOT, "output/_basemodels/boxer/mixamo_upload")
STEM = "tishkok_boxer_mixamo"
WELD = 0.0001  # 0.1 mm
CLAY = (0.42, 0.428, 0.442)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def apply_tf(objs, loc=True, rot=True, scale=True):
    if not objs:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=loc, rotation=rot, scale=scale)


def world_dims_verts(objs):
    xs, ys, zs = [], [], []
    for o in objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
    if not xs:
        return Vector((0, 0, 0)), Vector((0, 0, 0)), Vector((0, 0, 0))
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    return mn, mx, mx - mn


def detect_forward(obj):
    """Estimate face direction in XY from the head-band outlier."""
    mw = obj.matrix_world
    zs = [(mw @ v.co).z for v in obj.data.vertices]
    zmin, zmax = min(zs), max(zs)
    cut = zmin + 0.82 * (zmax - zmin)
    pts = [mw @ v.co for v in obj.data.vertices if (mw @ v.co).z >= cut]
    if len(pts) < 20:
        return Vector((0, -1, 0)), "fallback_-Y"
    cx = sum(p.x for p in pts) / len(pts)
    cy = sum(p.y for p in pts) / len(pts)
    farthest = max(pts, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)
    face = Vector((farthest.x - cx, farthest.y - cy, 0.0))
    if face.length < 1e-6:
        return Vector((0, -1, 0)), "fallback_-Y"
    face.normalize()
    return face, "head_outlier"


def clay_mat():
    mat = bpy.data.materials.new("PorcelainClay")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*CLAY, 1)
    bsdf.inputs["Roughness"].default_value = 0.93
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.06
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def mesh_stats(obj):
    me = obj.data
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    ngons = sum(1 for p in me.polygons if len(p.vertices) > 4)
    return {
        "name": obj.name,
        "verts": len(me.vertices),
        "faces": len(me.polygons),
        "tris": int(tris),
        "quads": sum(1 for p in me.polygons if len(p.vertices) == 4),
        "ngons": int(ngons),
        "modifiers": [m.type for m in obj.modifiers],
        "vertex_groups": len(obj.vertex_groups),
        "shape_keys": bool(me.shape_keys),
        "location": list(obj.location),
        "rotation_euler": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "parent": obj.parent.name if obj.parent else None,
    }


def prepare(src):
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    extras = [o for o in bpy.data.objects if o.type not in {"MESH"}]
    if not meshes:
        raise RuntimeError("no mesh imported")
    # Bake parent (glTF empties often hold the +90 X that stands the mesh up)
    # into each mesh BEFORE deleting extras, or the body falls onto Y.
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    apply_tf(meshes)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "tishkok_boxer_mixamo"

    # 0.1 mm weld, keep as ONE object (do not separate)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=WELD)
    bpy.ops.object.mode_set(mode="OBJECT")

    # drop leftover non-mesh (empties, cameras, armatures)
    for o in list(bpy.data.objects):
        if o != obj:
            bpy.data.objects.remove(o, do_unlink=True)

    # no modifiers, no vgroups, no shape keys, no animation
    for m in list(obj.modifiers):
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.modifier_apply(modifier=m.name)
        except Exception:
            obj.modifiers.remove(m)
    obj.vertex_groups.clear()
    if obj.data.shape_keys:
        obj.shape_key_clear()
    if obj.animation_data:
        obj.animation_data_clear()
    obj.parent = None

    # clay (no packed textures — Mixamo only needs geometry)
    mat = clay_mat()
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    try:
        obj.data.use_auto_smooth = False
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    apply_tf([obj])
    mn, mx, dim = world_dims_verts([obj])
    # If the mesh is still Y-up or X-up, stand it onto Z.
    longest = max(("x", dim.x), ("y", dim.y), ("z", dim.z), key=lambda t: t[1])[0]
    if longest != "z":
        print(f"STAND {longest}->Z dim={list(dim)}", flush=True)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if longest == "y":
            bpy.ops.transform.rotate(value=math.radians(-90), orient_axis="X",
                                     orient_type="GLOBAL")
        elif longest == "x":
            bpy.ops.transform.rotate(value=math.radians(90), orient_axis="Y",
                                     orient_type="GLOBAL")
        apply_tf([obj])
        mn, mx, dim = world_dims_verts([obj])
    height = float(dim.z)
    if height < 1.5 or height > 2.2:
        raise RuntimeError(f"unexpected height {height:.3f} m dim={list(dim)}")

    # feet on ground, origin at ground-center between the feet
    obj.location.z -= mn.z
    obj.location.x -= (mn.x + mx.x) * 0.5
    obj.location.y -= (mn.y + mx.y) * 0.5
    apply_tf([obj])

    face, how = detect_forward(obj)
    # rotate around Z so face points -Y
    want = Vector((0.0, -1.0, 0.0))
    ang = math.atan2(want.x, want.y) - math.atan2(face.x, face.y)
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    if abs(ang) > math.radians(3):
        obj.rotation_euler[2] += ang
        apply_tf([obj])
        face2, how2 = detect_forward(obj)
    else:
        face2, how2 = face, how
        ang = 0.0

    mn, mx, dim = world_dims_verts([obj])
    notes = {
        "source": src,
        "weld_m": WELD,
        "yaw_correction_deg": float(math.degrees(ang)),
        "facing_method": how2,
        "facing": [float(face2.x), float(face2.y), float(face2.z)],
        "height_m": float(dim.z),
        "bbox_min": list(mn),
        "bbox_max": list(mx),
        "stats": mesh_stats(obj),
        "armatures_removed": len(arms),
        "extras_removed": len(extras),
    }
    return obj, notes


def export_fbx(obj, path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=True,
        object_types={"MESH"},
        use_mesh_modifiers=False,
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        bake_anim=False,
        path_mode="AUTO",
        embed_textures=False,
        axis_forward="-Z",
        axis_up="Y",
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_ALL",
        bake_space_transform=True,
        global_scale=1.0,
        use_tspace=False,
        use_custom_props=False,
    )


def export_obj(obj, path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=path,
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
        export_triangulated_mesh=False,
        apply_modifiers=False,
        export_eval_mode="DAG_EVAL_RENDER",
        global_scale=1.0,
    )


def verify_import(path, kind):
    reset()
    before = set(bpy.data.objects)
    if kind == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif kind == "obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(kind)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    arms = [o for o in new if o.type == "ARMATURE"]
    info = {
        "kind": kind,
        "n_mesh": len(meshes),
        "n_armature": len(arms),
        "other": [o.type for o in new if o.type not in {"MESH", "ARMATURE"}],
    }
    if not meshes:
        info["ok"] = False
        info["error"] = "no mesh on re-import"
        return info
    raw_stats = [mesh_stats(o) for o in meshes]
    apply_tf(meshes)
    obj = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
        meshes = [obj]
    mn, mx, dim = world_dims_verts(meshes)
    # stand up if the baked Mixamo Y-up survived into world Z as a side axis
    longest = max(("x", dim.x), ("y", dim.y), ("z", dim.z), key=lambda t: t[1])[0]
    stood = False
    if longest != "z":
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if longest == "y":
            bpy.ops.transform.rotate(value=math.radians(-90), orient_axis="X",
                                     orient_type="GLOBAL")
        elif longest == "x":
            bpy.ops.transform.rotate(value=math.radians(90), orient_axis="Y",
                                     orient_type="GLOBAL")
        apply_tf([obj])
        mn, mx, dim = world_dims_verts([obj])
        stood = True
    face, how = detect_forward(obj)
    info.update({
        "ok": True,
        "raw_stats": raw_stats,
        "stats": [mesh_stats(o) for o in meshes],
        "height_m": float(dim.z),
        "min_z": float(mn.z),
        "bbox_min": list(mn),
        "bbox_max": list(mx),
        "facing": [float(face.x), float(face.y), float(face.z)],
        "facing_method": how,
        "facing_dot_-Y": float(face.dot(Vector((0, -1, 0)))),
        "stood_up_after_import": stood,
        "identity_tf": all(
            abs(c) < 1e-4 for o in meshes
            for c in list(o.location) + list(o.rotation_euler)
        ) and all(
            abs(s - 1.0) < 1e-4 for o in meshes for s in o.scale
        ),
    })
    return info


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fbx = os.path.join(OUT_DIR, STEM + ".fbx")
    objp = os.path.join(OUT_DIR, STEM + ".obj")
    zipp = os.path.join(OUT_DIR, STEM + ".zip")
    facts_path = os.path.join(OUT_DIR, STEM + "_export.json")

    reset()
    obj, notes = prepare(SRC)
    export_fbx(obj, fbx)
    export_obj(obj, objp)

    # zip OBJ (+ mtl if written) as the fallback package
    mtl = os.path.splitext(objp)[0] + ".mtl"
    with zipfile.ZipFile(zipp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(objp, arcname=os.path.basename(objp))
        if os.path.isfile(mtl):
            zf.write(mtl, arcname=os.path.basename(mtl))

    report = {
        "prepare": notes,
        "files": {
            "fbx": fbx,
            "fbx_bytes": os.path.getsize(fbx),
            "obj": objp,
            "obj_bytes": os.path.getsize(objp),
            "zip": zipp,
            "zip_bytes": os.path.getsize(zipp),
            "mtl": mtl if os.path.isfile(mtl) else None,
        },
    }
    report["verify_fbx"] = verify_import(fbx, "fbx")
    report["verify_obj"] = verify_import(objp, "obj")

    # caveats for the operator
    tris = notes["stats"]["tris"]
    report["caveats"] = [
        "Sculpted boxing gloves are fused into the same mesh as the hands. "
        "Mixamo's auto-rigger may fail to find fingers inside the glove volume "
        "or place finger bones badly. Use skeleton LOD 'No Fingers' "
        "(the gloves are already closed fists).",
        "Shorts and shoes are fused shells, not separate meshes — expected; "
        "auto-weights should still land on hips/feet if markers are sane.",
        "Rest pose is A-pose, not T-pose. Mixamo accepts A-pose; do not "
        "force a T-pose on this unrigged sculpt.",
        f"Triangle count after 0.1 mm weld is {tris} (limit 100k).",
        "Original textures replaced with a single clay material so the "
        "upload stays geometry-only. Mixamo auto-rig does not need textures.",
        "Credit Tishkok (CC BY 4.0) if this mesh ships.",
    ]

    with open(facts_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print("PREPARE", json.dumps(notes, indent=2))
    print("FILES", json.dumps(report["files"], indent=2))
    print("VERIFY_FBX", json.dumps(report["verify_fbx"], indent=2, default=str))
    print("VERIFY_OBJ", json.dumps(report["verify_obj"], indent=2, default=str))
    print("CAVEATS")
    for c in report["caveats"]:
        print("-", c)
    print("FACTS", facts_path)

    # fail the process if verification is clearly wrong
    vf = report["verify_fbx"]
    if not vf.get("ok"):
        raise SystemExit("FBX re-import failed")
    if vf.get("n_armature", 0) != 0:
        raise SystemExit("FBX re-import still has an armature")
    if vf.get("n_mesh") != 1:
        raise SystemExit(f"FBX re-import mesh count {vf.get('n_mesh')}")
    if vf["stats"][0]["modifiers"]:
        raise SystemExit(f"FBX re-import modifiers {vf['stats'][0]['modifiers']}")
    if vf["height_m"] < 1.5 or vf["height_m"] > 2.1:
        raise SystemExit(f"FBX re-import height {vf['height_m']}")
    if abs(vf["min_z"]) > 0.05:
        print("WARN feet not on ground, min_z", vf["min_z"])
    if vf.get("facing_dot_-Y", 0) < 0.5:
        print("WARN facing may not be -Y, dot", vf.get("facing_dot_-Y"))
    if tris >= 100000:
        raise SystemExit(f"tri count {tris} exceeds 100k")


if __name__ == "__main__":
    main()
