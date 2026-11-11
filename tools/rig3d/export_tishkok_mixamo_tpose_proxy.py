#!/usr/bin/env python3
"""Build a T-pose Mixamo proxy of Tishkok + a Quaternius control FBX.

A-pose voxel remesh may fuse hanging gloves to the thighs / arms to the
ribs. This script:

  1. Measures glove-to-thigh and arm-to-torso gaps on the original A-pose
     shells and on the existing A-pose remesh proxy.
  2. Fits the known Rigify A-pose bind, poses arms to a horizontal T-pose
     (palms down) and slightly spreads the legs, applies the pose as the
     new rest, then voxel-remeshes / decimates like the A-pose proxy.
  3. Exports an unrigged Quaternius Superhero Male T-pose as a Mixamo
     control file (should auto-rig easily if the server itself is healthy).

  nice -n 19 blender -b --python tools/rig3d/export_tishkok_mixamo_tpose_proxy.py

Writes (gitignored):
  output/_basemodels/boxer/mixamo_upload/tishkok_tpose_proxy_mixamo.fbx
  output/_basemodels/boxer/mixamo_upload/tishkok_tpose_proxy_mixamo.obj
  output/_basemodels/boxer/mixamo_upload/tishkok_tpose_proxy_mixamo.zip
  output/_basemodels/boxer/mixamo_upload/tishkok_tpose_proxy_preview.png
  output/_basemodels/boxer/mixamo_upload/control_quaternius_tpose.fbx
"""
from __future__ import annotations

import json
import math
import os
import sys
import zipfile

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval_tishkok_jab import (  # noqa: E402
    auto_weight,
    centroid,
    fit_metarig,
    prepare_tishkok,
    rigid_parent_bone,
)
from export_tishkok_mixamo import (  # noqa: E402
    OUT_DIR,
    SRC,
    apply_tf,
    clay_mat,
    detect_forward,
    export_fbx,
    export_obj,
    mesh_stats,
    reset,
    verify_import,
    world_dims_verts,
)
from export_tishkok_mixamo_proxy import (  # noqa: E402
    count_tris,
    fuse_and_simplify,
    manifold_stats,
)

STEM = "tishkok_tpose_proxy_mixamo"
APOSE_PROXY = os.path.join(OUT_DIR, "tishkok_proxy_mixamo.fbx")
PREVIEW = os.path.join(OUT_DIR, "tishkok_tpose_proxy_preview.png")
CONTROL = os.path.join(OUT_DIR, "control_quaternius_tpose.fbx")
QUATERNIUS = os.path.join(
    os.path.dirname(OUT_DIR),
    "quaternius_ubc",
    "Universal Base Characters[Standard]",
    "Base Characters",
    "Godot - UE",
    "Superhero_Male_FullBody.gltf",
)
HIDE_KEYS = ("eyebrow", "eye", "hair", "lash", "icosphere", "sphere", "brow")


def world_verts(obj):
    """World-space verts from the evaluated mesh (includes armature pose)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = obj.matrix_world
    pts = [mw @ v.co.copy() for v in me.vertices]
    ev.to_mesh_clear()
    return pts


def min_dist(pts_a, pts_b):
    if not pts_a or not pts_b:
        return None, None
    tree = KDTree(len(pts_b))
    for i, p in enumerate(pts_b):
        tree.insert(p, i)
    tree.balance()
    best = 1e9
    pair = None
    for p in pts_a:
        co, _idx, dist = tree.find(p)
        if dist < best:
            best = dist
            pair = (list(p), list(co))
    return float(best), pair


def aabb(pts):
    if not pts:
        return None
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    return {
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
    }


def count_in_box(pts, mn, mx):
    n = 0
    for p in pts:
        if (mn[0] <= p.x <= mx[0] and mn[1] <= p.y <= mx[1]
                and mn[2] <= p.z <= mx[2]):
            n += 1
    return n


def _pair_info(dist, pair):
    return {
        "min_m": None if dist is None else round(dist, 4),
        "min_cm": None if dist is None else round(dist * 100.0, 2),
        "at": pair,
    }


def measure_shell_gaps(groups, label):
    """Gaps between separate shells (or a joined mesh treated as body)."""
    body = groups.get("body")
    shorts = groups.get("shorts")
    glove_l = groups.get("glove_l")
    glove_r = groups.get("glove_r")
    report = {"label": label}

    thigh_src = []
    if body is not None:
        thigh_src.extend(world_verts(body))
    if shorts is not None:
        thigh_src.extend(world_verts(shorts))
    torso_pts, arm_pts, thigh_l, thigh_r = [], [], [], []
    for p in thigh_src:
        if 0.28 <= p.z <= 0.85 and p.x <= -0.04 and abs(p.y) < 0.18:
            thigh_l.append(p)
        if 0.28 <= p.z <= 0.85 and p.x >= 0.04 and abs(p.y) < 0.18:
            thigh_r.append(p)
        if 0.92 <= p.z <= 1.38 and abs(p.x) <= 0.16:
            torso_pts.append(p)
        if 1.00 <= p.z <= 1.42 and abs(p.x) >= 0.17:
            arm_pts.append(p)

    gl = world_verts(glove_l) if glove_l else []
    gr = world_verts(glove_r) if glove_r else []

    d_l, p_l = min_dist(gl, thigh_l)
    d_r, p_r = min_dist(gr, thigh_r)
    d_arm, p_arm = min_dist(arm_pts, torso_pts)
    report["glove_l_to_thigh"] = _pair_info(d_l, p_l)
    report["glove_r_to_thigh"] = _pair_info(d_r, p_r)
    report["arm_to_torso"] = _pair_info(d_arm, p_arm)
    report["counts"] = {
        "glove_l": len(gl), "glove_r": len(gr),
        "thigh_l": len(thigh_l), "thigh_r": len(thigh_r),
        "arm": len(arm_pts), "torso": len(torso_pts),
    }

    # A-pose air corridor between inner glove and outer thigh (each side).
    corridors = {}
    for side, gpts, tpts in (("L", gl, thigh_l), ("R", gr, thigh_r)):
        if not gpts or not tpts:
            corridors[side] = None
            continue
        ga, ta = aabb(gpts), aabb(tpts)
        # overlap in Z / Y, gap along X
        z0 = max(ga["min"][2], ta["min"][2])
        z1 = min(ga["max"][2], ta["max"][2])
        y0 = min(ga["min"][1], ta["min"][1]) - 0.02
        y1 = max(ga["max"][1], ta["max"][1]) + 0.02
        if side == "L":
            # glove more negative X; inner glove = max x, outer thigh = min x
            x0, x1 = ga["max"][0], ta["min"][0]
        else:
            x0, x1 = ta["max"][0], ga["min"][0]
        if x1 < x0:
            x0, x1 = x1, x0
        box = {"min": [x0, y0, z0], "max": [x1, y1, z1],
               "z_overlap_m": round(z1 - z0, 4) if z1 > z0 else 0.0,
               "x_gap_m": round(x1 - x0, 4)}
        all_pts = thigh_src + gpts
        box["verts_in_corridor"] = count_in_box(all_pts, box["min"], box["max"])
        corridors[side] = box
    report["glove_thigh_corridor"] = corridors

    print("GAPS", label, json.dumps(report, default=str), flush=True)
    return report


def measure_mesh_regions(obj, label, glove_pred, thigh_pred, arm_pred, torso_pred):
    pts = world_verts(obj)
    gloves = [p for p in pts if glove_pred(p)]
    thighs = [p for p in pts if thigh_pred(p)]
    arms = [p for p in pts if arm_pred(p)]
    torso = [p for p in pts if torso_pred(p)]
    d_g, p_g = min_dist(gloves, thighs)
    d_a, p_a = min_dist(arms, torso)
    report = {
        "label": label,
        "glove_to_thigh": _pair_info(d_g, p_g),
        "arm_to_torso": _pair_info(d_a, p_a),
        "counts": {
            "glove": len(gloves), "thigh": len(thighs),
            "arm": len(arms), "torso": len(torso), "total": len(pts),
        },
    }
    print("GAPS", label, json.dumps(report, default=str), flush=True)
    return report


def _whead(arm, name):
    pb = arm.pose.bones[name]
    return arm.matrix_world @ pb.head


def _wtail(arm, name):
    pb = arm.pose.bones[name]
    return arm.matrix_world @ pb.tail


def arm_elevation_deg(arm, side):
    h = _whead(arm, f"upper_arm.{side}")
    t = _wtail(arm, f"upper_arm.{side}")
    d = t - h
    horiz = math.hypot(d.x, d.y)
    return math.degrees(math.atan2(d.z, horiz))


def _aim_pose_bone(arm, name, y_dir, roll_hint=Vector((0.0, 0.0, 1.0))):
    """Rotate a pose bone so its +Y aims along y_dir; keep the head fixed."""
    pb = arm.pose.bones[name]
    bpy.context.view_layer.update()
    y = y_dir.normalized()
    x = y.cross(roll_hint)
    if x.length < 1e-5:
        x = y.cross(Vector((0.0, 1.0, 0.0)))
    x.normalize()
    z = x.cross(y)
    z.normalize()
    x = y.cross(z)
    x.normalize()
    mat = Matrix((x, y, z)).transposed().to_4x4()
    head_w = arm.matrix_world @ pb.head
    mat.translation = head_w
    pb.matrix = arm.matrix_world.inverted() @ mat
    bpy.context.view_layer.update()


def _spread_thighs(meta, groups, target_half):
    """Small hip swing so the shoes move out toward shoulder width.

    Picks the local Euler axis that actually increases |shoe.x| — no IK, so
    a misplaced foot bone cannot yank the pelvis.
    """
    bpy.ops.object.select_all(action="DESELECT")
    meta.select_set(True)
    bpy.context.view_layer.objects.active = meta
    bpy.ops.object.mode_set(mode="POSE")
    result = {}
    joined = groups.get("joined")

    def _shoe_x(side_sign):
        if joined is not None:
            pts = [p for p in world_verts(joined)
                   if p.z < 0.16 and side_sign * p.x > 0.04]
            if pts:
                return sum(p.x for p in pts) / len(pts)
        key = "shoe_l" if side_sign < 0 else "shoe_r"
        shoe = groups.get(key)
        if shoe is None:
            return None
        return centroid(shoe).x

    def _shoe_z(side_sign):
        if joined is not None:
            pts = [p for p in world_verts(joined)
                   if p.z < 0.16 and side_sign * p.x > 0.04]
            if pts:
                return sum(p.z for p in pts) / len(pts)
        key = "shoe_l" if side_sign < 0 else "shoe_r"
        shoe = groups.get(key)
        return None if shoe is None else centroid(shoe).z

    for side, sign in (("L", -1.0), ("R", 1.0)):
        pb = meta.pose.bones.get(f"thigh.{side}")
        if pb is None:
            continue
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        x0 = _shoe_x(sign)
        if x0 is None:
            continue
        if abs(x0) >= target_half - 0.02:
            result[side] = {"skipped": True, "x": x0}
            print(f"TPOSE thigh {side} already wide shoe_x={x0:.3f}", flush=True)
            continue
        # Probe each local axis / sign at +8 deg and pick the one that
        # moves the shoe further outward (more |x| in the sign direction).
        best = None
        for axis in (0, 1, 2):
            for sgn in (1.0, -1.0):
                eul = [0.0, 0.0, 0.0]
                eul[axis] = math.radians(8.0 * sgn)
                pb.rotation_euler = eul
                bpy.context.view_layer.update()
                x1 = _shoe_x(sign)
                score = sign * (x1 - x0)  # positive = moved outward
                z1 = _shoe_z(sign)
                rec = (score, axis, sgn, x1, z1)
                if best is None or rec[0] > best[0]:
                    best = rec
        pb.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        if best is None or best[0] < 0.005:
            result[side] = {"skipped": True, "x": x0, "best": best}
            print(f"TPOSE thigh {side} no useful axis shoe_x={x0:.3f}",
                  flush=True)
            continue
        _score, axis, sgn, _x1, _z1 = best
        # Scale 8 deg by how far we still need to go, cap at 16 deg.
        need = target_half - abs(x0)
        gained = abs(best[0])  # |dx| at 8 deg
        deg = 8.0 * (need / max(gained, 1e-4))
        deg = max(3.0, min(16.0, deg))
        eul = [0.0, 0.0, 0.0]
        eul[axis] = math.radians(deg * sgn)
        pb.rotation_euler = eul
        bpy.context.view_layer.update()
        x1 = _shoe_x(sign)
        z1 = _shoe_z(sign)
        result[side] = {
            "axis": axis, "sign": sgn, "deg": round(deg, 2),
            "x0": round(x0, 4), "x1": round(x1, 4), "z1": round(z1, 4),
        }
        print(f"TPOSE thigh {side} axis={axis} sgn={sgn} deg={deg:.1f} "
              f"shoe {x0:.3f} -> {x1:.3f} z={z1:.3f}", flush=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    return result


def pose_metarig_to_tpose(meta, groups=None):
    """Aim A-pose arms into a horizontal T-pose (palms down-ish).

    Does not apply as rest — the bind stays valid against the A-pose rest
    so the armature modifier can bake the posed mesh afterwards.
    Direct bone aiming is used instead of IK so a bad pole/foot target
    cannot yank the spine. Legs get a small thigh swing, not foot IK.
    """
    bpy.context.view_layer.update()
    print("TPOSE before elev L/R",
          round(arm_elevation_deg(meta, "L"), 1),
          round(arm_elevation_deg(meta, "R"), 1), flush=True)

    bpy.ops.object.select_all(action="DESELECT")
    meta.select_set(True)
    bpy.context.view_layer.objects.active = meta
    bpy.ops.object.mode_set(mode="POSE")
    for side, sign in (("L", -1.0), ("R", 1.0)):
        aim = Vector((sign, 0.0, 0.0))
        for bname in (f"upper_arm.{side}", f"forearm.{side}", f"hand.{side}"):
            _aim_pose_bone(meta, bname, aim, roll_hint=Vector((0.0, 0.0, 1.0)))
        print(f"TPOSE aimed {side} elev={arm_elevation_deg(meta, side):.1f} "
              f"hand={list(_wtail(meta, f'hand.{side}'))} "
              f"sh={list(_whead(meta, f'upper_arm.{side}'))}", flush=True)
    bpy.ops.object.mode_set(mode="OBJECT")

    thigh_info = {}
    if groups:
        sh_l = _whead(meta, "upper_arm.L")
        sh_r = _whead(meta, "upper_arm.R")
        half = 0.5 * abs(sh_r.x - sh_l.x)
        thigh_info = _spread_thighs(meta, groups, half)

    bpy.context.view_layer.update()
    print("TPOSE after elev L/R",
          round(arm_elevation_deg(meta, "L"), 1),
          round(arm_elevation_deg(meta, "R"), 1),
          "handL", list(_wtail(meta, "hand.L")),
          "handR", list(_wtail(meta, "hand.R")),
          flush=True)
    return {
        "elev_L_deg": round(arm_elevation_deg(meta, "L"), 2),
        "elev_R_deg": round(arm_elevation_deg(meta, "R"), 2),
        "hand_L": list(_wtail(meta, "hand.L")),
        "hand_R": list(_wtail(meta, "hand.R")),
        "thighs": thigh_info,
    }


def apply_visual_shells(groups, meta):
    """Bake the current pose into mesh data and detach from the metarig."""
    bpy.context.view_layer.update()
    for key, obj in list(groups.items()):
        if obj is None:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        for m in list(obj.modifiers):
            try:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier=m.name)
            except Exception as e:
                print(f"APPLY_MOD_FAIL {obj.name} {m.name} {e}", flush=True)
                try:
                    obj.modifiers.remove(m)
                except Exception:
                    pass
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw
        apply_tf([obj])
        print(f"BAKED {key} {obj.name} c={list(centroid(obj))}", flush=True)
    bpy.context.view_layer.update()
    if meta is not None:
        bpy.data.objects.remove(meta, do_unlink=True)


def join_groups(groups, name):
    objs = [o for o in groups.values() if o is not None]
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    mat = clay_mat()
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    apply_tf([obj])
    return obj


def face_minus_y(obj):
    face, how = detect_forward(obj)
    target = Vector((0.0, -1.0, 0.0))
    ang = math.atan2(face.x, -face.y)
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    if abs(ang) > math.radians(3):
        obj.rotation_euler[2] += ang
        apply_tf([obj])
        print(f"YAW {math.degrees(ang):.1f} deg via {how} face={list(face)}",
              flush=True)
    return how, list(face)


def render_front_preview(obj, path, size=(540, 960)):
    scene = bpy.context.scene
    # EEVEE name differs across Blender versions.
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue
    if hasattr(scene, "eevee"):
        try:
            scene.eevee.taa_render_samples = 16
        except Exception:
            pass
        try:
            scene.eevee.use_gtao = True
            scene.eevee.use_bloom = True
        except Exception:
            pass
    scene.render.resolution_x = size[0]
    scene.render.resolution_y = size[1]
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = path

    world = bpy.data.worlds.new("preview_world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.10, 0.105, 0.115, 1.0)
        bg.inputs[1].default_value = 1.0

    mn, mx, dim = world_dims_verts([obj])
    cx = 0.5 * (mn.x + mx.x)
    cy = 0.5 * (mn.y + mx.y)
    cz = 0.5 * (mn.z + mx.z)
    cam_d = max(dim.x * 1.55, dim.z * 1.45)
    cam_data = bpy.data.cameras.new("preview_cam")
    cam_data.lens = 40
    cam = bpy.data.objects.new("preview_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (cx, cy - cam_d, cz + dim.z * 0.04)
    direction = Vector((cx, cy, cz + dim.z * 0.02)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    def add_light(name, loc, energy, size=0.6):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.size = size
        obj_l = bpy.data.objects.new(name, data)
        obj_l.location = loc
        scene.collection.objects.link(obj_l)
        look = Vector((cx, cy, cz)) - Vector(loc)
        obj_l.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
        return obj_l

    add_light("key", (cx - 1.4, cy - 2.2, cz + 1.6), 450.0, 1.2)
    add_light("fill", (cx + 1.6, cy - 1.4, cz + 0.8), 180.0, 1.4)
    add_light("rim", (cx, cy + 2.0, cz + 1.4), 220.0, 0.8)

    # clay already assigned
    bpy.ops.render.render(write_still=True)
    print("PREVIEW", path, os.path.getsize(path) if os.path.isfile(path) else 0,
          flush=True)
    return path


def bind_and_pose_tishkok(groups):
    """Fit the metarig to A-pose landmarks, then join every shell and
    auto-weight the *single* multi-island mesh.

    Rigid-parenting gloves/shoes while auto-weighting the body lets the two
    deform differently; after a 70 deg arm raise the gloves then float off
    the wrists and voxel remesh cannot fuse them (even at 20 mm). Joining
    first keeps the A-pose glove/hand overlap so remesh fuses like it did
    on the A-pose proxy.
    """
    meta = fit_metarig(groups)
    joined = join_groups(groups, "tishkok_tpose_src")
    auto_weight(joined, meta)
    # Reconstruct a groups dict so thigh-spread can still probe shoes.
    # After join there is only one object; probe via spatial filters.
    probe = {"joined": joined}
    info = pose_metarig_to_tpose(meta, probe)
    info["joined"] = joined.name
    return meta, info, joined


def export_quaternius_control():
    if not os.path.isfile(QUATERNIUS):
        raise FileNotFoundError(QUATERNIUS)
    reset()
    bpy.ops.import_scene.gltf(filepath=QUATERNIUS)
    bpy.context.view_layer.update()

    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        low = o.name.lower()
        if any(k in low for k in HIDE_KEYS):
            o.hide_set(True)
            o.hide_render = True

    meshes = [o for o in bpy.data.objects
              if o.type == "MESH" and not o.hide_get()]
    # Face meshes sit on the front of the head — use that to yaw to -Y
    # *before* joining, because a T-pose arm span fools detect_forward.
    face_ms = [o for o in meshes if "face" in o.name.lower()]
    yaw_from_face = None
    if face_ms:
        fc = centroid(face_ms[0]) if len(face_ms) == 1 else (
            sum((centroid(o) for o in face_ms), Vector()) / len(face_ms))
        xs, ys = [], []
        for o in meshes:
            for v in o.data.vertices:
                p = o.matrix_world @ v.co
                xs.append(p.x)
                ys.append(p.y)
        body_c = Vector((sum(xs) / len(xs), sum(ys) / len(ys), 0.0))
        face_xy = Vector((fc.x - body_c.x, fc.y - body_c.y, 0.0))
        if face_xy.length > 1e-6:
            face_xy.normalize()
            yaw_from_face = math.atan2(face_xy.x, -face_xy.y)
            yaw_from_face = (yaw_from_face + math.pi) % (2 * math.pi) - math.pi
            print(f"QUATERNIUS face-dir {list(face_xy)} yaw={math.degrees(yaw_from_face):.1f}",
                  flush=True)
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    apply_tf(meshes)

    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" or (o.type == "MESH" and o.hide_get()):
            me = getattr(o, "data", None)
            bpy.data.objects.remove(o, do_unlink=True)
            if me is not None and getattr(me, "users", 1) == 0:
                try:
                    if o.type == "MESH":
                        bpy.data.meshes.remove(me)
                    else:
                        bpy.data.armatures.remove(me)
                except Exception:
                    pass

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no Quaternius meshes left after hiding extras")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "control_quaternius_tpose"
    obj.modifiers.clear()
    obj.vertex_groups.clear()
    if obj.data.shape_keys:
        obj.shape_key_clear()
    obj.animation_data_clear()
    mat = clay_mat()
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    apply_tf([obj])
    if yaw_from_face is not None and abs(yaw_from_face) > math.radians(3):
        obj.rotation_euler[2] += yaw_from_face
        apply_tf([obj])
        print(f"QUATERNIUS applied yaw {math.degrees(yaw_from_face):.1f}",
              flush=True)

    mn, mx, dim = world_dims_verts([obj])
    # Stand up if the import left it on its back (longest axis not Z).
    if dim.y > dim.z * 1.15 and dim.y > dim.x:
        obj.rotation_euler[0] = -math.pi / 2.0
        apply_tf([obj])
        mn, mx, dim = world_dims_verts([obj])
        print("STOOD_UP Quaternius (was Y-long)", flush=True)
    obj.location.z -= mn.z
    obj.location.x -= 0.5 * (mn.x + mx.x)
    obj.location.y -= 0.5 * (mn.y + mx.y)
    apply_tf([obj])
    mn, mx, dim = world_dims_verts([obj])
    # Real-world adult ~1.82 m; don't rescale if already close.
    if not (1.6 <= dim.z <= 2.1):
        factor = 1.82 / max(dim.z, 1e-6)
        obj.scale *= factor
        apply_tf([obj])
        mn, mx, dim = world_dims_verts([obj])
        obj.location.z -= mn.z
        apply_tf([obj])
        print(f"RESCALE Quaternius x{factor:.3f} -> {dim.z:.3f}m", flush=True)

    export_fbx(obj, CONTROL)
    stats = mesh_stats(obj)
    mn, mx, dim = world_dims_verts([obj])
    info = {
        "path": CONTROL,
        "bytes": os.path.getsize(CONTROL),
        "height_m": round(dim.z, 4),
        "width_m": round(dim.x, 4),
        "stats": stats,
        "identity": [list(obj.location), list(obj.rotation_euler),
                     list(obj.scale)],
        "armatures": 0,
    }
    print("CONTROL", json.dumps(info, default=str), flush=True)
    return info


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    facts = {"ok": False}
    fbx = os.path.join(OUT_DIR, STEM + ".fbx")
    objp = os.path.join(OUT_DIR, STEM + ".obj")
    zipp = os.path.join(OUT_DIR, STEM + ".zip")
    facts_path = os.path.join(OUT_DIR, STEM + "_export.json")

    # --- 1a. Original A-pose shell gaps ---
    reset()
    groups = prepare_tishkok(SRC)
    facts["gap_original_apose"] = measure_shell_gaps(groups, "original_apose")

    # --- 2. Bind + pose to T-pose, measure clearance, remesh ---
    meta, pose_info, joined = bind_and_pose_tishkok(groups)
    facts["tpose"] = pose_info
    bpy.context.view_layer.update()
    # After join there is a single mesh; measure with T-pose spatial filters
    # on the evaluated (posed) mesh.
    facts["gap_posed_tpose"] = measure_mesh_regions(
        joined, "posed_tpose_eval",
        glove_pred=lambda p: p.z >= 1.12 and abs(p.x) >= 0.45,
        thigh_pred=lambda p: 0.28 <= p.z <= 0.85 and 0.04 <= abs(p.x) <= 0.30,
        arm_pred=lambda p: 1.15 <= p.z <= 1.50 and 0.28 <= abs(p.x) <= 0.70,
        torso_pred=lambda p: 0.92 <= p.z <= 1.40 and abs(p.x) <= 0.16,
    )

    apply_visual_shells({"body": joined}, meta)
    joined = bpy.data.objects.get("tishkok_tpose_src") or joined
    # Do NOT yaw: detect_forward uses the head-band XY outlier, and in T-pose
    # the outstretched gloves win, producing a ~90 deg spin.
    mn, mx, dim = world_dims_verts([joined])
    joined.location.z -= mn.z
    joined.location.x -= 0.5 * (mn.x + mx.x)
    joined.location.y -= 0.5 * (mn.y + mx.y)
    apply_tf([joined])
    facts["joined_before_remesh"] = {
        "height_m": round(world_dims_verts([joined])[2].z, 4),
        "tris": count_tris(joined),
        "stats": mesh_stats(joined),
    }

    import export_tishkok_mixamo_proxy as proxylib
    # T-pose gloves sit on the fists; if a wrist gap survives 12 mm, step up.
    # Arms are ~10 cm+ off the ribs so 20 mm will not weld them to the torso.
    proxylib.VOXEL_TRIES_M = (0.007, 0.008, 0.010, 0.012, 0.016, 0.020)
    proxy, attempts, best = fuse_and_simplify(joined)
    proxy.name = STEM
    mn, mx, dim = world_dims_verts([proxy])
    proxy.location.z -= mn.z
    proxy.location.x -= 0.5 * (mn.x + mx.x)
    proxy.location.y -= 0.5 * (mn.y + mx.y)
    apply_tf([proxy])

    facts["gap_tpose_proxy"] = measure_mesh_regions(
        proxy, "tpose_proxy",
        glove_pred=lambda p: p.z >= 1.12 and abs(p.x) >= 0.45,
        thigh_pred=lambda p: 0.28 <= p.z <= 0.85 and 0.04 <= abs(p.x) <= 0.30,
        arm_pred=lambda p: 1.15 <= p.z <= 1.50 and 0.28 <= abs(p.x) <= 0.70,
        torso_pred=lambda p: 0.92 <= p.z <= 1.40 and abs(p.x) <= 0.16,
    )
    facts["remesh_attempts"] = attempts
    facts["remesh_best"] = best
    mf = manifold_stats(proxy)
    facts["manifold"] = mf
    mn, mx, dim = world_dims_verts([proxy])
    facts["dims"] = {"min": list(mn), "max": list(mx), "dim": list(dim)}
    facts["identity"] = [list(proxy.location), list(proxy.rotation_euler),
                         list(proxy.scale)]
    facts["stats"] = mesh_stats(proxy)

    render_front_preview(proxy, PREVIEW)
    facts["preview"] = PREVIEW
    facts["preview_bytes"] = (os.path.getsize(PREVIEW)
                              if os.path.isfile(PREVIEW) else 0)

    export_fbx(proxy, fbx)
    export_obj(proxy, objp)
    with zipfile.ZipFile(zipp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(objp, arcname=os.path.basename(objp))
        mtl = objp[:-4] + ".mtl"
        if os.path.isfile(mtl):
            zf.write(mtl, arcname=os.path.basename(mtl))
    facts["exports"] = {
        "fbx": fbx, "fbx_bytes": os.path.getsize(fbx),
        "obj": objp, "obj_bytes": os.path.getsize(objp),
        "zip": zipp, "zip_bytes": os.path.getsize(zipp),
    }
    print("WROTE", fbx, os.path.getsize(fbx), flush=True)

    # verify by re-import
    facts["verify_fbx"] = verify_import(fbx, "fbx")

    # --- 1b. Existing A-pose remesh proxy gaps ---
    if os.path.isfile(APOSE_PROXY):
        reset()
        bpy.ops.import_scene.fbx(filepath=APOSE_PROXY)
        bpy.context.view_layer.update()
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        if meshes:
            # If more than one, join for measurement (should be one).
            bpy.ops.object.select_all(action="DESELECT")
            for o in meshes:
                o.select_set(True)
            bpy.context.view_layer.objects.active = meshes[0]
            if len(meshes) > 1:
                bpy.ops.object.join()
            ap = bpy.context.view_layer.objects.active
            facts["gap_apose_proxy"] = measure_mesh_regions(
                ap, "apose_proxy",
                glove_pred=lambda p: 0.66 <= p.z <= 1.06 and abs(p.x) >= 0.26,
                thigh_pred=lambda p: 0.28 <= p.z <= 0.82 and 0.04 <= abs(p.x) <= 0.24,
                arm_pred=lambda p: 1.00 <= p.z <= 1.42 and 0.17 <= abs(p.x) <= 0.40,
                torso_pred=lambda p: 0.92 <= p.z <= 1.38 and abs(p.x) <= 0.16,
            )
            # Corridor occupancy using original A-pose corridor if we have it.
            orig = facts.get("gap_original_apose", {})
            corridors = orig.get("glove_thigh_corridor") or {}
            pts = world_verts(ap)
            occ = {}
            for side, box in corridors.items():
                if not box:
                    continue
                occ[side] = {
                    "original_verts": box.get("verts_in_corridor"),
                    "proxy_verts": count_in_box(pts, box["min"], box["max"]),
                    "box": box,
                }
            facts["apose_proxy_corridor_fill"] = occ
            print("CORRIDOR_FILL", json.dumps(occ, default=str), flush=True)
    else:
        facts["gap_apose_proxy"] = f"missing {APOSE_PROXY}"

    # --- 3. Quaternius control ---
    facts["quaternius"] = export_quaternius_control()
    facts["verify_control"] = verify_import(CONTROL, "fbx")

    facts["ok"] = bool(
        facts.get("manifold", {}).get("manifold")
        and facts.get("manifold", {}).get("components") == 1
        and facts.get("verify_fbx", {}).get("ok", True)
        if isinstance(facts.get("verify_fbx"), dict) else True
    )
    # verify_import returns a dict; tolerate missing 'ok'
    if isinstance(facts.get("verify_fbx"), dict) and "ok" not in facts["verify_fbx"]:
        vf = facts["verify_fbx"]
        facts["ok"] = bool(
            facts.get("manifold", {}).get("manifold")
            and facts.get("manifold", {}).get("components") == 1
            and vf.get("n_mesh", 1) == 1
            and vf.get("n_arm", 0) == 0
        )

    with open(facts_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("FACTS", facts_path, flush=True)
    print("OK", facts["ok"], flush=True)
    if not facts["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
