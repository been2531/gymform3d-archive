#!/usr/bin/env python3
"""Build a watertight Mixamo proxy from Tishkok (voxel-remesh + decimate).

Mixamo's auto-rigger failed on the multi-shell original (gloves/shorts/shoes
as disconnected overlapping islands). This fuses everything into one
manifold component.

  nice -n 19 blender -b --python tools/rig3d/export_tishkok_mixamo_proxy.py

Writes (gitignored):
  output/_basemodels/boxer/mixamo_upload/tishkok_proxy_mixamo.fbx
  output/_basemodels/boxer/mixamo_upload/tishkok_proxy_mixamo.obj
  output/_basemodels/boxer/mixamo_upload/tishkok_proxy_mixamo.zip
"""
from __future__ import annotations

import json
import os
import sys
import zipfile

import bmesh
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from export_tishkok_mixamo import (  # noqa: E402
    SRC,
    OUT_DIR,
    apply_tf,
    clay_mat,
    export_fbx,
    export_obj,
    mesh_stats,
    prepare,
    reset,
    verify_import,
    world_dims_verts,
)

STEM = "tishkok_proxy_mixamo"
TARGET_TRIS = (15000, 25000)
# 6–8 mm as requested; larger sizes are fallbacks if components don't fuse.
VOXEL_TRIES_M = (0.007, 0.006, 0.008, 0.010, 0.012)


def count_tris(obj):
    return int(sum(len(p.vertices) - 2 for p in obj.data.polygons))


def manifold_stats(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    loose_e = sum(1 for e in bm.edges if len(e.link_faces) == 0)
    over = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    nonman_e = sum(1 for e in bm.edges if not e.is_manifold)
    nonman_v = sum(1 for v in bm.verts if not v.is_manifold)
    isolated_v = sum(1 for v in bm.verts if not v.link_faces)

    parent = list(range(len(bm.faces))) if bm.faces else []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in bm.edges:
        linked = e.link_faces
        if len(linked) >= 2:
            i0 = linked[0].index
            for f in linked[1:]:
                union(i0, f.index)
    components = len({find(i) for i in range(len(bm.faces))}) if bm.faces else 0
    bm.free()
    return {
        "boundary_edges": int(boundary),
        "loose_edges": int(loose_e),
        "overconnected_edges": int(over),
        "nonmanifold_edges": int(nonman_e),
        "nonmanifold_verts": int(nonman_v),
        "isolated_verts": int(isolated_v),
        "components": int(components),
        "manifold": (
            boundary == 0 and over == 0 and nonman_e == 0
            and nonman_v == 0 and components == 1 and isolated_v == 0
        ),
    }


def select_active(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def voxel_remesh(obj, voxel_size):
    select_active(obj)
    me = obj.data
    me.remesh_voxel_size = float(voxel_size)
    me.remesh_voxel_adaptivity = 0.0
    for attr, val in (
        ("use_remesh_fix_poles", True),
        ("use_remesh_preserve_volume", True),
        ("use_remesh_preserve_paint_mask", False),
        ("use_remesh_preserve_sculpt_face_sets", False),
        ("use_remesh_preserve_vertex_colors", False),
    ):
        if hasattr(me, attr):
            setattr(me, attr, val)
    bpy.ops.object.voxel_remesh()
    print(f"REMESH voxel={voxel_size:.4f} m -> tris={count_tris(obj)}", flush=True)


def clean_mesh(obj):
    select_active(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=True)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.select_all(action="DESELECT")
    try:
        bpy.ops.mesh.select_interior_faces()
        bpy.ops.mesh.delete(type="FACE")
    except Exception as e:
        print("interior_faces", type(e).__name__, e, flush=True)
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.mesh.fill_holes(sides=8)
    except Exception:
        pass
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    try:
        obj.data.use_auto_smooth = False
    except Exception:
        pass
    bpy.ops.object.shade_smooth()


def decimate_to(obj, target_tris):
    tris = count_tris(obj)
    if tris <= target_tris * 1.02:
        print(f"DECIMATE skip ({tris} <= {target_tris})", flush=True)
        return tris
    select_active(obj)
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = max(0.05, min(1.0, float(target_tris) / float(max(tris, 1))))
    try:
        mod.use_collapse_triangulate = True
    except Exception:
        pass
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"DECIMATE {tris} -> {count_tris(obj)} (target {target_tris})",
          flush=True)
    return count_tris(obj)


def drop_tiny_components(obj, min_face_frac=0.02):
    """If a few dust islands remain, keep only components above the cutoff."""
    select_active(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    # separate, inspect, rejoin keepers
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    parts = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if len(parts) <= 1:
        return parts[0] if parts else obj
    parts = sorted(parts, key=lambda o: len(o.data.polygons), reverse=True)
    total = sum(len(o.data.polygons) for o in parts)
    keep, drop = [], []
    for o in parts:
        if len(o.data.polygons) >= min_face_frac * total or o is parts[0]:
            keep.append(o)
        else:
            drop.append(o)
    print(f"COMPONENTS {len(parts)} keep={len(keep)} drop={len(drop)} "
          f"sizes={[len(o.data.polygons) for o in parts[:8]]}", flush=True)
    for o in drop:
        me = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if me.users == 0:
            bpy.data.meshes.remove(me)
    select_active(keep[0])
    for o in keep[1:]:
        o.select_set(True)
    if len(keep) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def fuse_and_simplify(obj):
    attempts = []
    best = None
    for voxel in VOXEL_TRIES_M:
        # work on a copy so we can retry
        select_active(obj)
        bpy.ops.object.duplicate()
        cand = bpy.context.view_layer.objects.active
        cand.name = f"proxy_try_{int(voxel * 1000)}mm"
        voxel_remesh(cand, voxel)
        clean_mesh(cand)
        pre = {"voxel_m": voxel, "after_remesh": manifold_stats(cand),
               "tris_after_remesh": count_tris(cand)}
        # fuse leftover dust only; if gloves didn't fuse they are large — don't drop
        ms = pre["after_remesh"]
        if ms["components"] > 1:
            cand = drop_tiny_components(cand, min_face_frac=0.05)
            clean_mesh(cand)
            ms = manifold_stats(cand)
            pre["after_drop"] = ms
            pre["tris_after_drop"] = count_tris(cand)

        tris = count_tris(cand)
        lo, hi = TARGET_TRIS
        mid = (lo + hi) // 2
        if tris > hi:
            decimate_to(cand, mid)
            clean_mesh(cand)
        pre["tris_final"] = count_tris(cand)
        pre["manifold_final"] = manifold_stats(cand)
        attempts.append(pre)
        print("TRY", json.dumps(pre, default=str), flush=True)

        mf = pre["manifold_final"]
        tcount = pre["tris_final"]
        in_range = lo <= tcount <= hi
        score = (
            (0 if mf["manifold"] else 1000)
            + (0 if mf["components"] == 1 else 200)
            + mf["boundary_edges"] + mf["nonmanifold_edges"]
            + (0 if in_range else 50)
            + abs(tcount - mid) / 1000.0
        )
        rec = {"score": score, "obj": cand, "meta": pre}
        if best is None or score < best["score"]:
            if best is not None:
                o = best["obj"]
                me = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                if me.users == 0:
                    bpy.data.meshes.remove(me)
            best = rec
        else:
            me = cand.data
            bpy.data.objects.remove(cand, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)

        if mf["manifold"] and in_range:
            print("ACCEPTED", voxel, tcount, flush=True)
            break

    # drop the unfused original working mesh
    me = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if me.users == 0:
        bpy.data.meshes.remove(me)
    out = best["obj"]
    out.name = "tishkok_proxy_mixamo"
    mat = clay_mat()
    out.data.materials.clear()
    out.data.materials.append(mat)
    apply_tf([out])
    mn, mx, dim = world_dims_verts([out])
    out.location.z -= mn.z
    out.location.x -= (mn.x + mx.x) * 0.5
    out.location.y -= (mn.y + mx.y) * 0.5
    apply_tf([out])
    return out, attempts, best["meta"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fbx = os.path.join(OUT_DIR, STEM + ".fbx")
    objp = os.path.join(OUT_DIR, STEM + ".obj")
    zipp = os.path.join(OUT_DIR, STEM + ".zip")
    facts_path = os.path.join(OUT_DIR, STEM + "_export.json")

    reset()
    base, prep = prepare(SRC)
    base.name = "tishkok_proxy_src"
    obj, attempts, chosen = fuse_and_simplify(base)
    mn, mx, dim = world_dims_verts([obj])
    notes = {
        "source": SRC,
        "prepare": prep,
        "attempts": attempts,
        "chosen": chosen,
        "height_m": float(dim.z),
        "bbox_min": list(mn),
        "bbox_max": list(mx),
        "stats": mesh_stats(obj),
        "manifold": manifold_stats(obj),
    }
    print("CHOSEN", json.dumps(notes["manifold"], indent=2), flush=True)
    print("STATS", json.dumps(notes["stats"], indent=2), flush=True)

    export_fbx(obj, fbx)
    export_obj(obj, objp)
    mtl = os.path.splitext(objp)[0] + ".mtl"
    with zipfile.ZipFile(zipp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(objp, arcname=os.path.basename(objp))
        if os.path.isfile(mtl):
            zf.write(mtl, arcname=os.path.basename(mtl))

    report = {
        "build": notes,
        "files": {
            "fbx": fbx,
            "fbx_bytes": os.path.getsize(fbx),
            "obj": objp,
            "obj_bytes": os.path.getsize(objp),
            "zip": zipp,
            "zip_bytes": os.path.getsize(zipp),
            "mtl": mtl if os.path.isfile(mtl) else None,
        },
        "verify_fbx": verify_import(fbx, "fbx"),
        "verify_obj": verify_import(objp, "obj"),
    }
    with open(facts_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("FILES", json.dumps(report["files"], indent=2), flush=True)
    print("VERIFY_FBX height", report["verify_fbx"].get("height_m"),
          "facing_dot", report["verify_fbx"].get("facing_dot_-Y"),
          "n_mesh", report["verify_fbx"].get("n_mesh"),
          "n_arm", report["verify_fbx"].get("n_armature"), flush=True)
    print("VERIFY_OBJ height", report["verify_obj"].get("height_m"),
          "facing_dot", report["verify_obj"].get("facing_dot_-Y"), flush=True)
    print("FACTS", facts_path, flush=True)

    mf = notes["manifold"]
    if not mf["manifold"]:
        raise SystemExit(
            f"proxy is not manifold: {json.dumps(mf)}"
        )
    lo, hi = TARGET_TRIS
    tris = notes["stats"]["tris"]
    if tris < lo or tris > hi:
        print(f"WARN tris {tris} outside {lo}-{hi}", flush=True)
    vf = report["verify_fbx"]
    if vf.get("n_mesh") != 1 or vf.get("n_armature", 0) != 0:
        raise SystemExit(f"FBX re-import unexpected: {vf}")
    if vf.get("height_m", 0) < 1.5 or vf.get("height_m", 0) > 2.2:
        raise SystemExit(f"FBX re-import height {vf.get('height_m')}")


if __name__ == "__main__":
    main()
