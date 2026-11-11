#!/usr/bin/env python3
"""Transfer a Mixamo-rigged Tishkok *proxy* onto the original Tishkok shells.

Run AFTER Mixamo auto-rig succeeds on tishkok_proxy_mixamo.fbx and you have
downloaded the rigged FBX (any animation clip is fine; we only need the
armature rest + weights on the proxy mesh).

  nice -n 19 blender -b --python tools/rig3d/transfer_mixamo_to_tishkok.py -- \
      --proxy-fbx output/_basemodels/boxer/mixamo_upload/tishkok_proxy_mixamo_rigged.fbx \
      --out-dir output/_basemodels/boxer/mixamo_upload \
      --name tishkok_from_mixamo

Steps:
  1. Import the Mixamo FBX, stand it up, scale to ~1.79 m, feet on the ground.
  2. Import original Tishkok, 0.1 mm weld, separate shells, classify
     (body / shorts / glove_l/r / shoe_l/r) — same rules as eval_tishkok_jab.
  3. Align the original to the proxy (height + feet + hip XY).
  4. Data Transfer weights from the proxy mesh onto body and shorts
     (POLYINTERP_NEAREST — nearest face interpolated). Bind those shells
     to the Mixamo armature (modifier, no automatic re-weight).
  5. Rigid-parent gloves to hand bones and shoes to foot bones.
  6. Hide the proxy mesh. Export an FBX of armature + original shells and
     write a facts JSON. Does not download or call Mixamo.

Gloves stay closed fists (sculpted); Mixamo 'No Fingers' skeletons are
expected and preferred.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from export_tishkok_mixamo import (  # noqa: E402
    SRC as TISHKOK_GLTF,
    apply_tf,
    clay_mat,
    detect_forward,
    mesh_stats,
    reset,
    world_dims_verts,
)

WELD = 0.0001

HAND_L = (
    "mixamorig:LeftHand", "mixamorig_LeftHand", "LeftHand",
    "hand_l", "DEF-hand.L", "hand.L",
)
HAND_R = (
    "mixamorig:RightHand", "mixamorig_RightHand", "RightHand",
    "hand_r", "DEF-hand.R", "hand.R",
)
FOOT_L = (
    "mixamorig:LeftFoot", "mixamorig_LeftFoot", "LeftFoot",
    "foot_l", "DEF-foot.L", "foot.L",
)
FOOT_R = (
    "mixamorig:RightFoot", "mixamorig_RightFoot", "RightFoot",
    "foot_r", "DEF-foot.R", "foot.R",
)


def argv():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def pick_bone(arm, names):
    pose = arm.pose.bones
    for n in names:
        if pose.get(n):
            return n
    lower = {b.name.lower(): b.name for b in pose}
    for n in names:
        key = n.lower().split(":")[-1]
        for lk, orig in lower.items():
            if lk.endswith(key) or key in lk.replace("mixamorig:", ""):
                return orig
    return None


def centroid(obj):
    ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
    acc = Vector((0, 0, 0))
    for p in ws:
        acc += p
    return acc / max(len(ws), 1)


def classify_part(p):
    ws = [p.matrix_world @ v.co for v in p.data.vertices]
    xs = [q.x for q in ws]
    zs = [q.z for q in ws]
    cen = Vector((sum(xs) / len(xs), 0.0, sum(zs) / len(zs)))
    dimx = max(xs) - min(xs)
    dimz = max(zs) - min(zs)
    tris = len(p.data.polygons)
    if cen.z < 0.25 and abs(cen.x) > 0.05:
        return "shoe_r" if cen.x > 0 else "shoe_l"
    if 0.68 < cen.z < 1.05 and abs(cen.x) > 0.28:
        return "glove_r" if cen.x > 0 else "glove_l"
    if (tris > 2000 and abs(cen.x) < 0.12 and 0.65 < cen.z < 1.05
            and dimx < 0.50 and dimz < 0.55):
        return "shorts"
    return "body"


def join_group(objs, name):
    if not objs:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    out = bpy.context.view_layer.objects.active
    out.name = name
    return out


def stand_and_ground(objs):
    """Stand the longest axis onto Z, feet at z=0, center XY, face -Y."""
    roots = [o for o in objs if o.parent is None] or list(objs)
    mn, mx, dim = world_dims_verts(objs)
    longest = max(("x", dim.x), ("y", dim.y), ("z", dim.z), key=lambda t: t[1])[0]
    if longest != "z":
        bpy.ops.object.select_all(action="DESELECT")
        for o in roots:
            o.select_set(True)
        bpy.context.view_layer.objects.active = roots[0]
        axis = "X" if longest == "y" else "Y"
        value = math.radians(-90 if longest == "y" else 90)
        bpy.ops.transform.rotate(value=value, orient_axis=axis,
                                 orient_type="GLOBAL")
        apply_tf(roots, loc=False, rot=True, scale=False)
        mn, mx, dim = world_dims_verts(objs)
    # scale to ~1.79 m if Mixamo handed back centimetres
    height = float(dim.z)
    if height > 10:  # almost certainly cm
        factor = 0.01
        for o in roots:
            o.scale *= factor
        apply_tf(roots, loc=False, rot=False, scale=True)
        mn, mx, dim = world_dims_verts(objs)
        height = float(dim.z)
    elif height < 0.4:
        raise RuntimeError(f"character too small after import: {height:.3f} m")
    for o in roots:
        o.location.z -= mn.z
        o.location.x -= (mn.x + mx.x) * 0.5
        o.location.y -= (mn.y + mx.y) * 0.5
    apply_tf(roots, loc=True, rot=False, scale=False)
    # facing: use the biggest mesh
    meshes = [o for o in objs if o.type == "MESH"]
    if meshes:
        body = max(meshes, key=lambda o: len(o.data.polygons))
        face, _ = detect_forward(body)
        want = Vector((0.0, -1.0, 0.0))
        ang = math.atan2(want.x, want.y) - math.atan2(face.x, face.y)
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        if abs(ang) > math.radians(3):
            bpy.ops.object.select_all(action="DESELECT")
            for o in roots:
                o.select_set(True)
            bpy.context.view_layer.objects.active = roots[0]
            bpy.ops.transform.rotate(value=ang, orient_axis="Z",
                                     orient_type="GLOBAL")
            apply_tf(roots, loc=False, rot=True, scale=False)
    mn, mx, dim = world_dims_verts(objs)
    return {"height_m": float(dim.z), "bbox_min": list(mn), "bbox_max": list(mx)}


def prepare_tishkok_shells(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("Tishkok import produced no meshes")
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    apply_tf(meshes)
    extras = [o for o in new if o not in meshes]
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    raw = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=WELD)
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    parts = [o for o in bpy.data.objects
             if o.type == "MESH" and o not in before]
    buckets = {k: [] for k in (
        "body", "shorts", "glove_l", "glove_r", "shoe_l", "shoe_r")}
    for p in parts:
        buckets[classify_part(p)].append(p)
    groups = {}
    for key, objs in buckets.items():
        groups[key] = join_group(objs, f"tishkok_{key}") if objs else None
        print(f"SHELL {key}: {len(objs)} -> "
              f"{None if groups[key] is None else groups[key].name}",
              flush=True)
    visible = [o for o in groups.values() if o is not None]
    apply_tf(visible)
    for o in extras:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    return groups


def import_mixamo_proxy(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError(
            f"Mixamo FBX has no armature — is this the *rigged* download? "
            f"objects={[o.name+':'+o.type for o in new]}"
        )
    if not meshes:
        raise RuntimeError("Mixamo FBX has no mesh to transfer weights from")
    # bake any import parents so stand_and_ground can move roots together
    for o in new:
        if o.parent is not None and o.parent not in new:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
    info = stand_and_ground(new)
    # prefer the largest mesh as the donor
    donor = max(meshes, key=lambda o: len(o.data.polygons))
    arm = arms[0]
    if donor.find_armature() is None:
        # ensure the armature modifier points at arm
        mods = [m for m in donor.modifiers if m.type == "ARMATURE"]
        if mods:
            mods[0].object = arm
        else:
            am = donor.modifiers.new("Armature", "ARMATURE")
            am.object = arm
    info.update({
        "armature": arm.name,
        "bones": [b.name for b in arm.data.bones],
        "n_bones": len(arm.data.bones),
        "donor": donor.name,
        "donor_groups": [vg.name for vg in donor.vertex_groups],
        "donor_tris": len(donor.data.polygons),
    })
    return arm, donor, info


def transfer_weights(donor, target, arm):
    """Nearest-face interpolated vertex-group copy, then armature-bind."""
    for vg in donor.vertex_groups:
        if target.vertex_groups.get(vg.name) is None:
            target.vertex_groups.new(name=vg.name)
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    mod = target.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    mod.object = donor
    mod.use_vert_data = True
    try:
        mod.data_types_verts = {"VGROUP_WEIGHTS"}
    except Exception:
        pass
    try:
        mod.vert_mapping = "POLYINTERP_NEAREST"
    except Exception:
        mod.vert_mapping = "NEAREST_FACE"
    try:
        mod.layers_vgroup_select_src = "ALL"
        mod.layers_vgroup_select_dst = "NAME"
    except Exception:
        pass
    notes = {"target": target.name}
    try:
        bpy.ops.object.datalayout_transfer(modifier=mod.name)
    except Exception as e:
        notes["datalayout_error"] = f"{type(e).__name__}: {e}"
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        notes["transfer"] = "data_transfer_polyinterp_nearest"
    except Exception as e:
        notes["transfer_apply_error"] = f"{type(e).__name__}: {e}"
        try:
            target.modifiers.remove(mod)
        except Exception:
            pass
        notes["transfer"] = "failed"
        return notes
    # bind without re-weighting
    existing = [m for m in target.modifiers if m.type == "ARMATURE"]
    if existing:
        existing[0].object = arm
    else:
        am = target.modifiers.new("Armature", "ARMATURE")
        am.object = arm
    target.parent = arm
    notes["groups"] = len(target.vertex_groups)
    return notes


def rigid_parent_bone(mesh, arm, bone_name):
    if arm.pose.bones.get(bone_name) is None:
        raise KeyError(f"missing bone {bone_name}")
    before = centroid(mesh)
    mw = mesh.matrix_world.copy()
    bone = arm.pose.bones[bone_name]
    tail_space = arm.matrix_world @ bone.matrix @ Matrix.Translation(
        (0.0, bone.length, 0.0))
    mesh.parent = arm
    mesh.parent_type = "BONE"
    mesh.parent_bone = bone_name
    mesh.matrix_parent_inverse = tail_space.inverted() @ mw
    bpy.context.view_layer.update()
    mesh.matrix_world = mw
    bpy.context.view_layer.update()
    after = centroid(mesh)
    err = (after - before).length
    print(f"RIGID {mesh.name} -> {bone_name} jump={err:.4f}m", flush=True)
    return err


def assign_clay(objs):
    mat = clay_mat()
    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)


def export_bound_fbx(objs, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=True,
        object_types={"MESH", "ARMATURE"},
        use_mesh_modifiers=True,
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
        use_armature_deform_only=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-fbx", required=True,
                    help="Mixamo-rigged proxy FBX (downloaded from Mixamo)")
    ap.add_argument("--source", default=TISHKOK_GLTF,
                    help="Original unrigged Tishkok glTF")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="tishkok_from_mixamo")
    args = ap.parse_args(argv())
    os.makedirs(args.out_dir, exist_ok=True)
    facts = {"ok": False, "proxy_fbx": args.proxy_fbx, "source": args.source}

    reset()
    try:
        arm, donor, proxy_info = import_mixamo_proxy(args.proxy_fbx)
        facts["proxy"] = proxy_info
        groups = prepare_tishkok_shells(args.source)
        vis = [o for o in groups.values() if o is not None]
        facts["tishkok_align"] = stand_and_ground(vis)
        facts["shells"] = {
            k: (None if v is None else {
                "name": v.name,
                "verts": len(v.data.vertices),
                "tris": len(v.data.polygons),
            })
            for k, v in groups.items()
        }

        transfers = {}
        for key in ("body", "shorts"):
            if groups.get(key) is not None:
                transfers[key] = transfer_weights(donor, groups[key], arm)
        facts["transfers"] = transfers

        bones = {
            "glove_l": pick_bone(arm, HAND_L),
            "glove_r": pick_bone(arm, HAND_R),
            "shoe_l": pick_bone(arm, FOOT_L),
            "shoe_r": pick_bone(arm, FOOT_R),
        }
        facts["parent_bones"] = bones
        jumps = {}
        for key, bname in bones.items():
            if groups.get(key) is None:
                continue
            if not bname:
                jumps[key] = f"missing bone for {key}"
                continue
            jumps[key] = rigid_parent_bone(groups[key], arm, bname)
        facts["parent_jumps_m"] = jumps

        assign_clay(vis)
        # hide the proxy donor so the original shells are what we export
        donor.hide_render = True
        try:
            donor.hide_set(True)
        except Exception:
            pass

        out_objs = [arm] + vis
        out_fbx = os.path.join(args.out_dir, args.name + ".fbx")
        export_bound_fbx(out_objs, out_fbx)
        facts["out_fbx"] = out_fbx
        facts["out_bytes"] = os.path.getsize(out_fbx)
        facts["ok"] = True
        print("WROTE", out_fbx, flush=True)
    except Exception:
        facts["ok"] = False
        facts["error"] = traceback.format_exc()
        print("ERROR", facts["error"], flush=True)

    fact_path = os.path.join(args.out_dir, args.name + "_facts.json")
    with open(fact_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("FACTS", fact_path, flush=True)
    if not facts.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
