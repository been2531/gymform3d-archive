#!/usr/bin/env python3
"""Import a candidate body, measure topology, render one porcelain-clay still.

Run with Blender 4.2.3:
  nice -n 19 blender -b --python tools/rig3d/eval_base_model.py -- \
      --input PATH --out-dir DIR --name ID [--yaw 135] [--objs-glob GLOB]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

import bpy
import bmesh
from mathutils import Vector

CLAY = (0.42, 0.428, 0.442)
ROUGH = 0.93
SPEC = 0.06
WELD_M = 0.0001  # 0.1 mm in metres
RES = (540, 960)
SAMPLES = 24


def _argv():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import(path, objs_glob=None):
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif os.path.isdir(path):
        import glob
        files = sorted(glob.glob(os.path.join(path, objs_glob or "*.obj")))
        if not files:
            raise SystemExit(f"no obj files in {path}")
        for i, f in enumerate(files):
            bpy.ops.wm.obj_import(filepath=f)
            if i % 50 == 0:
                print(f"imported {i}/{len(files)}", flush=True)
        print(f"imported {len(files)} objs", flush=True)
    else:
        raise SystemExit(f"unsupported: {path}")
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    arms = [o for o in new if o.type == "ARMATURE"]
    return meshes, arms


def _world_dims(objs):
    pts = []
    for o in objs:
        for c in o.bound_box:
            pts.append(o.matrix_world @ Vector(c))
    if not pts:
        return Vector((0, 0, 0)), Vector((0, 0, 0)), Vector((0, 0, 0))
    xs, ys, zs = zip(*[(p.x, p.y, p.z) for p in pts])
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    return mn, mx, mx - mn


def _apply_scale_to_metres(meshes):
    """If the figure is clearly in mm or cm, scale to ~1.7 m and apply."""
    mn, mx, dim = _world_dims(meshes)
    height = max(dim.x, dim.y, dim.z)
    scale = 1.0
    note = f"height_before={height:.4f}"
    if height > 50:  # mm (or huge cm)
        scale = 0.001
        note += " treated as millimetres"
    elif 8 < height < 40:  # cm? unusual; if ~17-20 treat as cm
        scale = 0.01
        note += " treated as centimetres"
    else:
        note += " treated as metres"
    if scale != 1.0:
        # scale about origin
        for o in meshes:
            o.scale = o.scale * scale
            bpy.context.view_layer.objects.active = o
            o.select_set(True)
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mn, mx, dim = _world_dims(meshes)
    info = {"scale_applied": scale, "note": note,
            "height_m": max(dim.x, dim.y, dim.z),
            "bbox_min": list(mn), "bbox_max": list(mx), "bbox_dim": list(dim)}
    info.update(_stand_upright(meshes))
    mn, mx, dim = _world_dims(meshes)
    info["height_m"] = max(dim.z, dim.y, dim.x)
    info["bbox_min"] = list(mn)
    info["bbox_max"] = list(mx)
    info["bbox_dim"] = list(dim)
    return info


def _stand_upright(meshes):
    """If the longest axis is not Z, rotate so the figure stands on Z and sits on z=0."""
    mn, mx, dim = _world_dims(meshes)
    axes = [(dim.x, "X"), (dim.y, "Y"), (dim.z, "Z")]
    longest = max(axes, key=lambda t: t[0])[1]
    if longest == "Z":
        return {"upright_rotate": None}
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if longest == "Y":
        bpy.ops.transform.rotate(value=math.radians(90), orient_axis="X", orient_type="GLOBAL")
    elif longest == "X":
        bpy.ops.transform.rotate(value=math.radians(-90), orient_axis="Y", orient_type="GLOBAL")
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    mn, mx, dim = _world_dims(meshes)
    dz = mn.z
    for o in meshes:
        o.location.z -= dz
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    return {"upright_rotate": f"{longest}->Z"}


def _boundary_and_islands(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    boundary = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    loose = sum(1 for e in bm.edges if len(e.link_faces) == 0)
    nonman = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    uv = bm.loops.layers.uv.active
    islands = 0
    if uv is not None and bm.faces:
        # faces connected if they share an edge whose UV endpoints match
        parent = list(range(len(bm.faces)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # map edge -> list of (face_index, uv_a, uv_b)
        from collections import defaultdict
        efaces = defaultdict(list)
        for fi, f in enumerate(bm.faces):
            loops = list(f.loops)
            n = len(loops)
            for i in range(n):
                a, b = loops[i], loops[(i + 1) % n]
                key = a.edge.index
                ua, ub = a[uv].uv.copy(), b[uv].uv.copy()
                efaces[key].append((fi, ua, ub))
        def close(p, q, eps=1e-6):
            return (p - q).length < eps
        for recs in efaces.values():
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    fi, a1, b1 = recs[i]
                    fj, a2, b2 = recs[j]
                    if (close(a1, a2) and close(b1, b2)) or (close(a1, b2) and close(b1, a2)):
                        union(fi, fj)
        islands = len({find(i) for i in range(len(bm.faces))})
    # geometric loose parts: faces connected by any shared edge
    loose_parts = 0
    isolated_verts = 0
    if bm.faces:
        gparent = list(range(len(bm.faces)))

        def gfind(i):
            while gparent[i] != i:
                gparent[i] = gparent[gparent[i]]
                i = gparent[i]
            return i

        def gunion(a, b):
            ra, rb = gfind(a), gfind(b)
            if ra != rb:
                gparent[rb] = ra

        for e in bm.edges:
            linked = e.link_faces
            if len(linked) >= 2:
                i0 = linked[0].index
                for f in linked[1:]:
                    gunion(i0, f.index)
        loose_parts = len({gfind(i) for i in range(len(bm.faces))})
    isolated_verts = sum(1 for v in bm.verts if not v.link_faces)
    bm.free()
    return {
        "verts": len(me.vertices),
        "faces": len(me.polygons),
        "tris": sum(len(p.vertices) - 2 for p in me.polygons),
        "edges": len(me.edges),
        "boundary_edges": int(boundary),
        "loose_edges": int(loose),
        "nonmanifold_edges": int(nonman),
        "loose_parts": int(loose_parts),
        "isolated_verts": int(isolated_verts),
        "uv_islands": int(islands),
        "has_uv": bool(me.uv_layers),
        "uv_layers": [l.name for l in me.uv_layers],
    }


def _weld_copy_metrics(obj):
    """Duplicate, apply 0.1 mm merge-by-distance, measure, delete the copy."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    dup = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=WELD_M)
    bpy.ops.object.mode_set(mode="OBJECT")
    # also apply object-level... already in object mode on evaluated mesh
    met = _boundary_and_islands(dup.data)
    n = dup.name
    me = dup.data
    bpy.data.objects.remove(dup, do_unlink=True)
    if me.users == 0:
        bpy.data.meshes.remove(me)
    return met


def _clay_material():
    mat = bpy.data.materials.new("PorcelainClay")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*CLAY, 1)
    bsdf.inputs["Roughness"].default_value = ROUGH
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = SPEC
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = SPEC
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _assign_clay(meshes):
    mat = _clay_material()
    packed_images = []
    for o in meshes:
        # record original materials / images before replacing
        for slot in o.material_slots:
            m = slot.material
            if not m or not m.use_nodes:
                continue
            for n in m.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image:
                    packed_images.append({
                        "object": o.name, "image": n.image.name,
                        "size": list(n.image.size) if n.image.size else None,
                    })
        o.data.materials.clear()
        o.data.materials.append(mat)
        # shade smooth, drop custom split normals so baked normal data cannot leak
        try:
            if hasattr(o.data, "attributes") and "custom_normal" in o.data.attributes:
                o.data.attributes.remove(o.data.attributes["custom_normal"])
        except Exception:
            pass
        try:
            o.data.use_auto_smooth = False
        except Exception:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            pass
    return packed_images


def _hide_by_name(meshes, substrings):
    """Hide reference/plinth objects by name substring (case-insensitive)."""
    keys = [s.lower() for s in (substrings or []) if s]
    hidden, kept = [], []
    for o in meshes:
        name = (o.name or "").lower()
        if keys and any(k in name for k in keys):
            o.hide_render = True
            o.hide_set(True)
            hidden.append(o.name)
        else:
            kept.append(o)
    return kept, hidden


def _hide_plinths(meshes):
    """Hide a wide, flat, low object (museum stand), never foot tendons."""
    mn, mx, dim = _world_dims(meshes)
    if dim.z < 0.5:
        return meshes, []
    hidden, kept = [], []
    for o in meshes:
        pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
        top = max(p.z for p in pts)
        bot = min(p.z for p in pts)
        width = max(p.x for p in pts) - min(p.x for p in pts)
        depth = max(p.y for p in pts) - min(p.y for p in pts)
        height = top - bot
        low = top < mn.z + 0.15 * dim.z
        wide = width > 0.35 * dim.x and depth > 0.35 * dim.y
        flat = height < 0.18 * dim.z
        name = (o.name or "").lower()
        named = any(k in name for k in ("plinth", "stand", "pedestal", "baseboard", "socle"))
        if (low and wide and flat) or named:
            o.hide_render = True
            o.hide_set(True)
            hidden.append(o.name)
        else:
            kept.append(o)
    return kept, hidden


def _setup_camera(meshes, yaw_deg, elev_deg=8.0, margin=0.08):
    mn, mx, dim = _world_dims(meshes)
    center = (mn + mx) / 2
    # prefer putting feet near the bottom: aim slightly above mid
    look = Vector((center.x, center.y, mn.z + 0.52 * dim.z))
    height = max(dim.z, 0.01)
    # distance to fit height in 9:16 with a 50mm lens
    # vertical sensor 24mm, angle v = 2*atan(12/50)
    v_ang = 2 * math.atan(12.0 / 50.0)
    dist = (height * (1 + 2 * margin) / 2) / math.tan(v_ang / 2)
    # 9:16: vertical FOV is the long side. Keep a small margin only.
    dist *= 0.92
    yaw = math.radians(yaw_deg)
    elev = math.radians(elev_deg)
    # rear 3/4: yaw around Z, from +Y as 0 (front) — we treat -Y as back default
    # camera sits at back-right when yaw=135 if +Y is front.
    direction = Vector((
        math.sin(yaw) * math.cos(elev),
        -math.cos(yaw) * math.cos(elev),
        math.sin(elev),
    ))
    loc = look + direction.normalized() * dist
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50
    cam_data.clip_start = 0.05
    cam_data.clip_end = 50
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    # aim
    direction = look - loc
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = cam
    return cam


def _setup_world_lights(cam):
    world = bpy.data.worlds.new("Charcoal")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.02, 0.02, 0.022, 1)
    bg.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world

    def area(name, loc, size, energy, look_at):
        d = bpy.data.lights.new(name, "AREA")
        d.size = size
        d.energy = energy
        d.spread = 150
        o = bpy.data.objects.new(name, d)
        bpy.context.scene.collection.objects.link(o)
        o.location = loc
        direction = look_at - loc
        o.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return o

    mn, mx, dim = _world_dims([o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render])
    look = (mn + mx) / 2
    # key slightly above camera-right, fill opposite, soft rim behind
    right = cam.matrix_world.to_quaternion() @ Vector((1, 0, 0))
    back = cam.matrix_world.to_quaternion() @ Vector((0, 0, 1))
    up = Vector((0, 0, 1))
    key = cam.location + right * 0.6 + up * 0.4
    fill = cam.location - right * 1.0 + up * 0.2
    rim = look - back * 1.2 + up * 0.8
    area("Key", key, 1.8, 40, look)
    area("Fill", fill, 2.4, 12, look)
    area("Rim", rim, 1.2, 16, look)


def _render(out_png, exposure=-0.20):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.filepath = out_png
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = exposure
    sc.view_settings.gamma = 1.0
    sc.cycles.filter_width = 1.5
    bpy.ops.render.render(write_still=True)
    return out_png


def measure(meshes, arms):
    per = []
    tot = {
        "objects": len(meshes),
        "armatures": [a.name for a in arms],
        "armature_bones": sum(len(a.data.bones) for a in arms),
        "verts": 0, "faces": 0, "tris": 0,
        "boundary_edges": 0, "uv_islands": 0,
        "nonmanifold_edges": 0,
        "objects_with_uv": 0,
    }
    for o in meshes:
        m = _boundary_and_islands(o.data)
        m["name"] = o.name
        per.append(m)
        for k in ("verts", "faces", "tris", "boundary_edges", "uv_islands", "nonmanifold_edges", "loose_parts"):
            tot[k] = tot.get(k, 0) + m[k]
        if m["has_uv"]:
            tot["objects_with_uv"] += 1
    # weld: join a temp copy of all, merge 0.1 mm, measure once
    weld = None
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.duplicate()
        dups = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        bpy.context.view_layer.objects.active = dups[0]
        if len(dups) > 1:
            bpy.ops.object.join()
        joined = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=WELD_M)
        bpy.ops.object.mode_set(mode="OBJECT")
        weld = _boundary_and_islands(joined.data)
        me = joined.data
        bpy.data.objects.remove(joined, do_unlink=True)
        if me.users == 0:
            bpy.data.meshes.remove(me)
    except Exception as e:
        weld = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    tot["after_0.1mm_weld"] = weld
    tot["per_object_top10"] = sorted(per, key=lambda x: -x["tris"])[:10]
    tot["separate_muscle_meshes"] = len(meshes) > 1
    tot["mesh_object_count"] = len(meshes)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--yaw", type=float, default=135.0,
                    help="camera yaw degrees; 135 = rear 3/4 if +Y is front")
    ap.add_argument("--elev", type=float, default=8.0)
    ap.add_argument("--objs-glob", default="*.obj")
    ap.add_argument("--no-hide-plinth", action="store_true")
    ap.add_argument("--hide-name", action="append", default=[],
                    help="hide objects whose name contains this substring")
    ap.add_argument("--exposure", type=float, default=-0.20,
                    help="view-transform exposure; proven porcelain clay is -0.20")
    ap.add_argument("--target-height", type=float, default=0.0,
                    help="if >0, rescale so current height becomes this many metres before weld")
    ap.add_argument("--no-render", action="store_true",
                    help="measure only; do not write a still")
    args = ap.parse_args(_argv())
    os.makedirs(args.out_dir, exist_ok=True)
    _reset()
    print(f"blender {bpy.app.version_string} import {args.input}", flush=True)
    meshes, arms = _import(args.input, args.objs_glob)
    print(f"meshes={len(meshes)} armatures={len(arms)}", flush=True)
    if not meshes:
        raise SystemExit("no meshes imported")
    scale_info = _apply_scale_to_metres(meshes)
    print("scale", scale_info, flush=True)
    if args.target_height and args.target_height > 0:
        current = max(scale_info.get("height_m") or 0.0, 1e-9)
        factor = args.target_height / current
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
            o.scale = o.scale * factor
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        mn, mx, dim = _world_dims(meshes)
        scale_info["normalized_from_m"] = current
        scale_info["normalize_factor"] = factor
        scale_info["target_height_m"] = args.target_height
        scale_info["height_m"] = max(dim.x, dim.y, dim.z)
        scale_info["bbox_min"] = list(mn)
        scale_info["bbox_max"] = list(mx)
        scale_info["bbox_dim"] = list(dim)
        print("normalized", scale_info, flush=True)
    metrics = measure(meshes, arms)
    metrics["scale"] = scale_info
    metrics["name"] = args.name
    metrics["input"] = args.input
    packed = _assign_clay(meshes)
    metrics["source_images"] = packed[:40]
    metrics["source_image_count"] = len(packed)
    extras = []
    for o in meshes:
        me = o.data
        extras.append({
            "name": o.name,
            "has_custom_normals": bool(getattr(me, "has_custom_normals", False)),
            "vertex_groups": [g.name for g in o.vertex_groups],
            "shape_keys": ([k.name for k in me.shape_keys.key_blocks] if me.shape_keys else []),
            "color_attributes": [a.name for a in getattr(me, "color_attributes", [])],
            "attribute_names": [a.name for a in me.attributes][:40] if hasattr(me, "attributes") else [],
        })
    metrics["mesh_extras"] = extras
    metrics["object_types"] = sorted({o.type for o in bpy.data.objects})
    metrics["non_mesh_objects"] = [
        {"name": o.name, "type": o.type} for o in bpy.data.objects if o.type != "MESH"
    ]
    # clay render does not need source textures; drop them to free RAM
    for img in list(bpy.data.images):
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass
    kept = meshes
    hidden = []
    if args.hide_name:
        kept, hid = _hide_by_name(kept, args.hide_name)
        hidden.extend(hid)
    if not args.no_hide_plinth:
        kept, hid = _hide_plinths(kept)
        hidden.extend(hid)
        if not kept:
            kept = meshes
    metrics["hidden_objects"] = hidden
    metrics["mesh_names"] = [o.name for o in meshes]
    still = os.path.join(args.out_dir, f"{args.name}_rear34.png")
    if args.no_render:
        metrics["still"] = None
        metrics["exposure"] = args.exposure
        metrics["camera"] = None
    else:
        cam = _setup_camera(kept, args.yaw, args.elev)
        _setup_world_lights(cam)
        _render(still, exposure=args.exposure)
        metrics["still"] = still
        metrics["exposure"] = args.exposure
        metrics["camera"] = {"yaw": args.yaw, "elev": args.elev, "location": list(cam.location)}
    out_json = os.path.join(args.out_dir, f"{args.name}_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("WROTE", still, out_json, flush=True)
    weld = metrics.get("after_0.1mm_weld") or {}
    print("SUMMARY", json.dumps({
        "name": args.name,
        "objects": metrics["objects"],
        "verts": metrics["verts"],
        "tris": metrics["tris"],
        "boundary_before": metrics["boundary_edges"],
        "boundary_after_weld": weld.get("boundary_edges"),
        "uv_islands": metrics["uv_islands"],
        "loose_parts_sum": metrics.get("loose_parts"),
        "weld_loose_parts": weld.get("loose_parts"),
        "weld_nonmanifold": weld.get("nonmanifold_edges"),
        "armature_bones": metrics["armature_bones"],
        "height_m": scale_info["height_m"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
