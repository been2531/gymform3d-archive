#!/usr/bin/env python3
"""Procedural vinyl muscle mascot for @gymform.3d — Concept 3.

Built from scratch in Blender 4.2.3 (metaballs + a 6-bone armature +
swappable face pieces). No paid assets. No image generation.

Pass criteria (set before the render, judged once):
  1. Reads like concept3_sheet.jpg: glossy pink-red vinyl, chubby, cute,
     figure-like, big thighs, short arms, little feet, dark charcoal studio.
  2. Face is fixed. Same sockets every time. Three expression kits
     (smug / scream / faint) are swappable mesh pieces, not regenerated.
  3. Readable at 1080x1920 on a phone — the expression is clear at a glance.
  4. Simple: one smooth body, a few motions (bounce, shake, fall). No
     complex human rig.

Run (from the repo root):
  nice -n 19 /home/box/opt/blender-4.2.3-linux-x64/blender -b \\
    -P mascot/build_mascot.py -- --out output/mascot/2026-11-17 --mode stills
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from mathutils import Vector

import bpy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PINK_SRGB = (200, 53, 67)          # sampled mid-tone from the concept still
FLOOR_SRGB = (32, 30, 32)
BACKDROP_SRGB = (22, 20, 22)
WORLD_SRGB = (16, 15, 17)

# Timeline for the optional test reel (24 fps, 13 s = 312 frames).
FPS = 24
REEL_SECONDS = 13.0
# 0.0–4.5 smug bounce, 4.5–9.0 scream squat+shake, 9.0–13.0 fall+faint
SEG_A = 4.5
SEG_B = 9.0


def srgb_to_linear(c, a=1.0):
    def ch(x):
        x = x / 255.0 if x > 1 else float(x)
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    return (ch(c[0]), ch(c[1]), ch(c[2]), a)


def parse_args():
    argv = sys.argv
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/daily_info_reels/output/mascot/2026-11-17")
    p.add_argument("--mode", choices=("preview", "stills", "reel", "all"), default="stills")
    p.add_argument("--samples", type=int, default=0, help="0 = mode default")
    return p.parse_args(extra)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.metaballs, bpy.data.materials,
                  bpy.data.armatures, bpy.data.curves, bpy.data.cameras,
                  bpy.data.lights, bpy.data.collections):
        for item in list(block):
            try:
                block.remove(item)
            except Exception:
                pass
    # Keep the default collection if it survived; otherwise make one.
    if not bpy.context.scene.collection:
        pass


def collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def link(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def active(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def look_at(obj, target, up="Y"):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", up).to_euler()


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def principled_mat(name, color, roughness=0.18, coat=0.45, coat_rough=0.08,
                   specular=0.55, subsurface=0.12, metallic=0.0, emission=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Specular IOR Level"].default_value = specular
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = coat_rough
    if subsurface > 0 and "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = subsurface
        bsdf.inputs["Subsurface Radius"].default_value = (0.5, 0.15, 0.15)
        bsdf.inputs["Subsurface Scale"].default_value = 0.15
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = emission[0]
        bsdf.inputs["Emission Strength"].default_value = emission[1]
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


# ---------------------------------------------------------------------------
# Body (metaballs → one smooth mesh)
# ---------------------------------------------------------------------------
def _ball(mb, co, radius, stiffness=2.0, typ="BALL", size=None):
    e = mb.elements.new(type=typ)
    e.co = co
    e.radius = radius
    e.stiffness = stiffness
    if size is not None:
        e.size_x, e.size_y, e.size_z = size
    return e


def build_body(col):
    """One fused vinyl body, arms slightly out so they are not welded to the hips."""
    mb = bpy.data.metaballs.new("BodyMB")
    mb.resolution = 0.06
    mb.render_resolution = 0.045
    mb.threshold = 0.48
    obj = bpy.data.objects.new("BodyMB", mb)
    link(obj, col)

    # Compact pear: round head, wide belly, huge thighs, short akimbo arms.
    # Head — one round dome, no pointy bump.
    _ball(mb, (0.0, 0.05, 1.12), 0.40)
    _ball(mb, (-0.16, -0.08, 1.06), 0.16, stiffness=1.5)
    _ball(mb, (0.16, -0.08, 1.06), 0.16, stiffness=1.5)
    # Chest blends into the head (no neck)
    _ball(mb, (0.0, 0.00, 0.82), 0.40)
    # Big belly, slightly toward the camera (−Y)
    _ball(mb, (0.0, -0.04, 0.55), 0.50)
    # Pelvis
    _ball(mb, (0.0, 0.00, 0.38), 0.36)
    # Huge thighs — widest part of the silhouette
    _ball(mb, (-0.34, 0.07, 0.28), 0.38, typ="ELLIPSOID", size=(0.95, 0.95, 1.05))
    _ball(mb, (0.34, 0.07, 0.28), 0.38, typ="ELLIPSOID", size=(0.95, 0.95, 1.05))
    _ball(mb, (-0.32, 0.05, 0.12), 0.24)
    _ball(mb, (0.32, 0.05, 0.12), 0.24)
    # Little feet
    _ball(mb, (-0.30, 0.09, 0.04), 0.10, typ="ELLIPSOID", size=(0.85, 1.35, 0.55))
    _ball(mb, (0.30, 0.09, 0.04), 0.10, typ="ELLIPSOID", size=(0.85, 1.35, 0.55))
    _ball(mb, (-0.30, 0.16, 0.03), 0.04, stiffness=1.4)
    _ball(mb, (0.30, 0.16, 0.03), 0.04, stiffness=1.4)
    # Shoulders rooted in the chest, then a short overlapping arm chain
    # to the hips, so the metaballs fuse into one body (no floating lobes).
    _ball(mb, (-0.28, 0.00, 0.86), 0.22)
    _ball(mb, (0.28, 0.00, 0.86), 0.22)
    _ball(mb, (-0.42, 0.02, 0.72), 0.16)
    _ball(mb, (0.42, 0.02, 0.72), 0.16)
    _ball(mb, (-0.48, 0.04, 0.58), 0.13)
    _ball(mb, (0.48, 0.04, 0.58), 0.13)
    _ball(mb, (-0.44, 0.07, 0.47), 0.11)
    _ball(mb, (0.44, 0.07, 0.47), 0.11)
    _ball(mb, (-0.38, 0.01, 0.46), 0.06, stiffness=1.5)
    _ball(mb, (0.38, 0.01, 0.46), 0.06, stiffness=1.5)

    bpy.context.view_layer.update()
    active(obj)
    bpy.ops.object.convert(target="MESH")
    body = bpy.context.active_object
    body.name = "MascotBody"

    # Voxel remesh keeps one closed, even vinyl surface.
    rem = body.modifiers.new("Remesh", "REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = 0.016
    rem.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier="Remesh")
    sm = body.modifiers.new("LSmooth", "LAPLACIANSMOOTH")
    sm.iterations = 8
    sm.lambda_factor = 0.35
    bpy.ops.object.modifier_apply(modifier="LSmooth")
    for poly in body.data.polygons:
        poly.use_smooth = True

    mat = principled_mat(
        "VinylPink",
        srgb_to_linear(PINK_SRGB),
        roughness=0.22,
        coat=0.35,
        coat_rough=0.12,
        specular=0.6,
        subsurface=0.18,
    )
    assign(body, mat)
    return body


# ---------------------------------------------------------------------------
# Simple 6-bone armature + spatial weights
# ---------------------------------------------------------------------------
BONES = {
    "Root":  ((0.0, 0.0, 0.00), (0.0, 0.0, 0.22)),
    "Torso": ((0.0, 0.0, 0.22), (0.0, 0.0, 0.80)),
    "Head":  ((0.0, 0.0, 0.80), (0.0, 0.0, 1.30)),
    "Arm_L": ((-0.40, 0.02, 0.86), (-0.48, 0.07, 0.48)),
    "Arm_R": ((0.40, 0.02, 0.86), (0.48, 0.07, 0.48)),
    "Thigh_L": ((-0.18, 0.04, 0.42), (-0.30, 0.06, 0.04)),
    "Thigh_R": ((0.18, 0.04, 0.42), (0.30, 0.06, 0.04)),
}
PARENTS = {
    "Torso": "Root",
    "Head": "Torso",
    "Arm_L": "Torso",
    "Arm_R": "Torso",
    "Thigh_L": "Root",
    "Thigh_R": "Root",
}


def build_armature(col, body):
    data = bpy.data.armatures.new("MascotArmature")
    data.display_type = "OCTAHEDRAL"
    arm = bpy.data.objects.new("MascotArmature", data)
    link(arm, col)
    active(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = {}
    for name, (head, tail) in BONES.items():
        eb = data.edit_bones.new(name)
        eb.head = head
        eb.tail = tail
        eb.roll = 0.0
        ebs[name] = eb
    for name, parent in PARENTS.items():
        ebs[name].parent = ebs[parent]
        ebs[name].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    _assign_spatial_weights(body)
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    mod.use_vertex_groups = True
    mod.use_deform_preserve_volume = True
    body.parent = arm
    return arm


def _assign_spatial_weights(body):
    mesh = body.data
    groups = {name: body.vertex_groups.new(name=name) for name in BONES}
    # Pre-gather coordinates in object space.
    for i, v in enumerate(mesh.vertices):
        x, y, z = v.co
        w = {name: 0.0 for name in BONES}
        # Head
        if z >= 0.90:
            w["Head"] = min(1.0, (z - 0.90) / 0.16)
        # Arms: outside the shoulder and below the head
        if z > 0.40 and z < 1.05:
            if x < -0.30:
                t = min(1.0, (-0.30 - x) / 0.22)
                w["Arm_L"] = t * t * (3 - 2 * t)
            elif x > 0.30:
                t = min(1.0, (x - 0.30) / 0.22)
                w["Arm_R"] = t * t * (3 - 2 * t)
        # Thighs / feet
        if z < 0.52:
            side = "Thigh_L" if x < 0 else "Thigh_R"
            t = min(1.0, (0.52 - z) / 0.20)
            w[side] = t * t * (3 - 2 * t)
        # Torso fills the remainder, Root gets a little everywhere
        used = w["Head"] + w["Arm_L"] + w["Arm_R"] + w["Thigh_L"] + w["Thigh_R"]
        w["Torso"] = max(0.0, 1.0 - used)
        w["Root"] = 0.08
        # Normalise
        total = sum(w.values())
        for name, val in w.items():
            if val > 0.001 and total > 0:
                groups[name].add([i], val / total, "REPLACE")


def pose_bone(arm, name, rot_xyz_deg=None, scale=None, location=None, frame=None):
    bone = arm.pose.bones[name]
    bone.rotation_mode = "XYZ"
    if rot_xyz_deg is not None:
        bone.rotation_euler = tuple(math.radians(a) for a in rot_xyz_deg)
        if frame is not None:
            bone.keyframe_insert("rotation_euler", frame=frame)
    if scale is not None:
        bone.scale = scale
        if frame is not None:
            bone.keyframe_insert("scale", frame=frame)
    if location is not None:
        bone.location = location
        if frame is not None:
            bone.keyframe_insert("location", frame=frame)


def clear_pose(arm):
    for b in arm.pose.bones:
        b.rotation_mode = "XYZ"
        b.rotation_euler = (0, 0, 0)
        b.scale = (1, 1, 1)
        b.location = (0, 0, 0)


# ---------------------------------------------------------------------------
# Face pieces — authored once, swapped per expression
# ---------------------------------------------------------------------------
def _icosphere(col, name, loc, radius, subdivisions=2):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, subdivisions=subdivisions, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    link(obj, col)
    return obj


def _uvsphere(col, name, loc, radius, scale=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=32, ring_count=16, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    if scale:
        obj.scale = scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    link(obj, col)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def make_smile(col, name, width=0.16, depth=0.04, thick=0.012):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = thick
    curve.bevel_resolution = 4
    curve.fill_mode = "FULL"
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(2)
    pts = [(-width, 0, 0.008), (0, 0, -depth), (width, 0, 0.008)]
    handles = [(-width * 0.4, 0, -0.004), (-width * 0.35, 0, -depth), (width * 0.4, 0, -0.004)]
    rhandles = [(-width * 0.15, 0, -depth * 0.3), (width * 0.35, 0, -depth), (width * 0.15, 0, 0.0)]
    for i, (p, hl, hr) in enumerate(zip(pts, handles, [
        (-width * 0.55, 0, 0.01),
        (width * 0.0, 0, -depth),
        (width * 0.55, 0, 0.01),
    ])):
        bp = spline.bezier_points[i]
        bp.co = (*p, )
        bp.handle_left_type = "ALIGNED"
        bp.handle_right_type = "ALIGNED"
        # simpler auto handles
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
        bp.co = p
    obj = bpy.data.objects.new(name, curve)
    link(obj, col)
    # Face the camera (−Y): the smile lives in XZ, so rotate into XY facing −Y
    # Points are in XZ; rotate +90° around X so they sit in a plane facing -Y? 
    # Bezier is already in X (width) and Z (curve). To face -Y, the curve
    # should be in the XZ plane, which it already is. Good.
    active(obj)
    bpy.ops.object.convert(target="MESH")
    obj = bpy.context.active_object
    obj.name = name
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def make_open_mouth(col, name, rx=0.09, rz=0.11, depth=0.08):
    """A dark oval bowl with a pink tongue — a toy mouth, not a boolean hole."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=28, ring_count=16, location=(0, 0, 0))
    bowl = bpy.context.active_object
    bowl.name = name
    # Flatten and cut conceptually by scaling, then squash in Y and keep front.
    bowl.scale = (rx, depth, rz)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # Keep only the back half (positive Y is into the head) by shrinking
    # vertices on the camera side slightly inward, forming a cavity.
    for v in bowl.data.vertices:
        if v.co.y < 0:
            v.co.y *= 0.15
            v.co.x *= 0.85
            v.co.z *= 0.85
    for p in bowl.data.polygons:
        p.use_smooth = True
    link(bowl, col)
    return bowl


def make_tongue(col, name, size=(0.07, 0.08, 0.045)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=20, ring_count=12, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for p in obj.data.polygons:
        p.use_smooth = True
    link(obj, col)
    return obj


def make_brow(col, name, length=0.11, thick=0.012, arch=0.018):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = thick
    curve.bevel_resolution = 3
    curve.fill_mode = "FULL"
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(2)
    pts = [(-length, 0, 0), (0, 0, arch), (length, 0, 0)]
    for i, p in enumerate(pts):
        spline.bezier_points[i].co = p
        spline.bezier_points[i].handle_left_type = "AUTO"
        spline.bezier_points[i].handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    link(obj, col)
    active(obj)
    bpy.ops.object.convert(target="MESH")
    obj = bpy.context.active_object
    obj.name = name
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def make_spiral(col, name, radius=0.07, turns=2.6, tube=0.0075, n=100):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = tube
    curve.bevel_resolution = 3
    curve.fill_mode = "FULL"
    spline = curve.splines.new("POLY")
    spline.points.add(n - 1)
    for i in range(n):
        t = i / (n - 1)
        ang = t * turns * 2 * math.pi
        r = radius * (0.04 + 0.96 * t)
        spline.points[i].co = (r * math.cos(ang), r * math.sin(ang), 0.0, 1.0)
    obj = bpy.data.objects.new(name, curve)
    link(obj, col)
    # Lie in a plane facing the camera (XZ): rotate 90° around X
    obj.rotation_euler = (math.pi / 2, 0, 0)
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.ops.object.convert(target="MESH")
    obj = bpy.context.active_object
    obj.name = name
    return obj


def make_teardrop(col, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.025, segments=16, ring_count=10, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    # Pull the bottom vertex down to a point.
    for v in obj.data.vertices:
        if v.co.z < -0.01:
            v.co.z -= 0.02 * ((-v.co.z) / 0.025)
            v.co.x *= 0.8
            v.co.y *= 0.8
    for p in obj.data.polygons:
        p.use_smooth = True
    link(obj, col)
    return obj


def make_star(col, name, size=0.035):
    # Two flattened cubes forming a plus, plus a small sphere — a sparkle.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size * 0.35, segments=12, ring_count=8, location=(0, 0, 0))
    core = bpy.context.active_object
    core.name = name
    link(core, col)
    for dx, dy, dz, sc in (
        (0, 0, 0, (0.18, 0.18, 1.0)),
        (0, 0, 0, (1.0, 0.18, 0.18)),
        (0, 0, 0, (0.18, 1.0, 0.18)),
    ):
        bpy.ops.mesh.primitive_cube_add(size=size * 2, location=(0, 0, 0))
        arm = bpy.context.active_object
        arm.scale = sc
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # Join later
        arm.parent = core
        arm.parent_type = "OBJECT"
        link(arm, col)
        arm.name = name + "_arm"
        # We'll join
        arm.select_set(True)
    active(core)
    # select children too
    for c in list(core.children):
        c.select_set(True)
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = name
    link(obj, col)
    return obj


def build_face_library(col, body):
    """Create every expression piece once, parented to the body, hidden by default."""
    mats = {
        "black": principled_mat("FaceBlack", srgb_to_linear((12, 10, 12)), roughness=0.12, coat=0.7, coat_rough=0.05, subsurface=0.0, specular=0.7),
        "white": principled_mat("FaceWhite", srgb_to_linear((245, 245, 248)), roughness=0.18, coat=0.4, subsurface=0.0),
        "highlight": principled_mat("FaceHighlight", srgb_to_linear((255, 255, 255)), roughness=0.08, coat=0.2, subsurface=0.0,
                                    emission=(srgb_to_linear((255, 255, 255)), 1.4)),
        "mouth_dark": principled_mat("MouthDark", srgb_to_linear((40, 10, 16)), roughness=0.45, coat=0.1, subsurface=0.0),
        "tongue": principled_mat("Tongue", srgb_to_linear((214, 70, 90)), roughness=0.28, coat=0.25, subsurface=0.25),
        "tear": principled_mat("Tear", srgb_to_linear((210, 230, 240), a=0.85), roughness=0.08, coat=0.6, subsurface=0.0, specular=0.8),
        "star": principled_mat("Sparkle", srgb_to_linear((255, 255, 255)), roughness=0.15, coat=0.2, subsurface=0.0,
                               emission=(srgb_to_linear((255, 255, 255)), 6.0)),
    }
    # Tear material alpha: also set blend if needed
    mats["tear"].blend_method = "BLEND"
    mats["tear"].use_screen_refraction = False

    # Face sockets — measured in body space, slightly in front of the head.
    # Eyes sit on the lower half of the dome, close together, facing -Y.
    eye_l = Vector((-0.105, -0.30, 1.08))
    eye_r = Vector((0.105, -0.30, 1.08))
    # Snap both to the body surface along -Y from inside the head, then pull out.
    eye_l = _snap_to_front(body, eye_l, pull=0.022)
    eye_r = _snap_to_front(body, eye_r, pull=0.022)
    print(f"EYE SOCKETS L={tuple(round(c,3) for c in eye_l)} R={tuple(round(c,3) for c in eye_r)}")

    highlight_off = Vector((0.022, -0.018, 0.020))  # same world side on both eyes
    pieces = {"smug": [], "scream": [], "faint": [], "fx_scream": [], "fx_faint": []}

    def add(expr, obj, loc, rot=(0, 0, 0), mat=None, scale=None):
        obj.location = loc
        # rot is in degrees
        obj.rotation_euler = (math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2]))
        if scale is not None:
            obj.scale = scale
        if mat is not None:
            assign(obj, mat)
        obj.parent = body
        obj.hide_render = True
        obj.hide_viewport = True
        pieces[expr].append(obj)
        return obj

    # ----- SMUG: glossy button eyes, soft brows, closed smile -----
    for side, pos in (("L", eye_l), ("R", eye_r)):
        add("smug", _uvsphere(col, f"SmugEye_{side}", (0, 0, 0), 0.085), pos, mat=mats["black"])
        add("smug", _icosphere(col, f"SmugHi_{side}", (0, 0, 0), 0.022, 2), pos + highlight_off, mat=mats["highlight"])
    add("smug", make_brow(col, "SmugBrow_L", length=0.055, thick=0.011, arch=0.012),
        eye_l + Vector((-0.01, -0.01, 0.11)), rot=(8, 0, 14), mat=mats["black"])
    add("smug", make_brow(col, "SmugBrow_R", length=0.055, thick=0.011, arch=0.012),
        eye_r + Vector((0.01, -0.01, 0.11)), rot=(8, 0, -14), mat=mats["black"])
    add("smug", make_smile(col, "SmugSmile", width=0.085, depth=0.028, thick=0.011),
        Vector((0.0, eye_l.y - 0.02, eye_l.z - 0.145)), mat=mats["black"])

    # ----- SCREAM: white sclera + pupil, worried brows, open mouth -----
    for side, pos in (("L", eye_l), ("R", eye_r)):
        add("scream", _uvsphere(col, f"ScreamSclera_{side}", (0, 0, 0), 0.095, scale=(1.0, 0.7, 1.15)),
            pos + Vector((0, 0.01, 0.0)), mat=mats["white"])
        add("scream", _uvsphere(col, f"ScreamPupil_{side}", (0, 0, 0), 0.048),
            pos + Vector((0.0, -0.02, -0.01)), mat=mats["black"])
        add("scream", _icosphere(col, f"ScreamHi_{side}", (0, 0, 0), 0.016, 2),
            pos + highlight_off + Vector((0, -0.01, 0)), mat=mats["highlight"])
    # Worried brows: inner ends up (inverted-V)
    add("scream", make_brow(col, "ScreamBrow_L", length=0.06, thick=0.012, arch=0.006),
        eye_l + Vector((-0.01, -0.01, 0.13)), rot=(6, 0, -28), mat=mats["black"])
    add("scream", make_brow(col, "ScreamBrow_R", length=0.06, thick=0.012, arch=0.006),
        eye_r + Vector((0.01, -0.01, 0.13)), rot=(6, 0, 28), mat=mats["black"])
    mouth_pos = Vector((0.0, eye_l.y + 0.01, eye_l.z - 0.16))
    add("scream", make_open_mouth(col, "ScreamMouth", rx=0.09, rz=0.11, depth=0.09),
        mouth_pos, mat=mats["mouth_dark"])
    add("scream", make_tongue(col, "ScreamTongue", size=(0.055, 0.07, 0.04)),
        mouth_pos + Vector((0.0, -0.05, -0.035)), mat=mats["tongue"])

    # Tears + sweat (scream FX)
    add("fx_scream", make_teardrop(col, "Tear_L"),
        eye_l + Vector((-0.02, -0.02, -0.08)), rot=(0, 0, 18), mat=mats["tear"])
    add("fx_scream", make_teardrop(col, "Tear_R"),
        eye_r + Vector((0.02, -0.02, -0.08)), rot=(0, 0, -18), mat=mats["tear"])
    add("fx_scream", _uvsphere(col, "Sweat_1", (0, 0, 0), 0.028),
        Vector((0.22, -0.22, 1.42)), mat=mats["tear"])
    add("fx_scream", _uvsphere(col, "Sweat_2", (0, 0, 0), 0.018),
        Vector((0.28, -0.18, 1.32)), mat=mats["tear"])

    # ----- FAINT: spiral eyes, tiny tongue-out, floppy brows -----
    for side, pos in (("L", eye_l), ("R", eye_r)):
        add("faint", make_spiral(col, f"FaintSpiral_{side}", radius=0.068, turns=2.5, tube=0.007),
            pos + Vector((0, -0.01, 0)), mat=mats["black"])
    add("faint", make_brow(col, "FaintBrow_L", length=0.05, thick=0.010, arch=0.004),
        eye_l + Vector((-0.01, -0.01, 0.12)), rot=(8, 0, 8), mat=mats["black"])
    add("faint", make_brow(col, "FaintBrow_R", length=0.05, thick=0.010, arch=0.004),
        eye_r + Vector((0.01, -0.01, 0.12)), rot=(8, 0, -8), mat=mats["black"])
    add("faint", make_tongue(col, "FaintTongue", size=(0.035, 0.05, 0.028)),
        Vector((0.0, eye_l.y - 0.05, eye_l.z - 0.145)), mat=mats["tongue"])
    add("faint", _uvsphere(col, "FaintMouth", (0, 0, 0), 0.028, scale=(1.15, 0.5, 0.85)),
        Vector((0.0, eye_l.y - 0.01, eye_l.z - 0.13)), mat=mats["black"])

    # Dizzy sparkles — positions in body space above the head; will still
    # read when the character is lying down because they are parented to body.
    sparkle_pts = [
        Vector((-0.08, -0.15, 1.50)),
        Vector((0.12, -0.20, 1.58)),
        Vector((0.00, -0.05, 1.64)),
        Vector((0.22, -0.10, 1.46)),
        Vector((-0.18, -0.08, 1.42)),
    ]
    for i, p in enumerate(sparkle_pts):
        add("fx_faint", make_star(col, f"Sparkle_{i}", size=0.028), p, rot=(20 * i, 15, 10 * i), mat=mats["star"])

    return pieces, (eye_l, eye_r)


def _snap_to_front(body, point, pull=0.01):
    """Ray-cast from inside the head toward the camera (−Y) and sit on the surface."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    origin = Vector((point.x, 0.15, point.z))  # inside
    direction = Vector((0, -1, 0))
    ok, loc, normal, _idx = ev.ray_cast(origin, direction)
    if ok:
        return Vector(loc) + Vector((0, -pull, 0))
    return point


def set_expression(pieces, name, fx=True):
    """Show one expression kit (and its FX). Hide the rest."""
    visible = {name}
    if fx and name == "scream":
        visible.add("fx_scream")
    if fx and name == "faint":
        visible.add("fx_faint")
    for key, objs in pieces.items():
        hide = key not in visible
        for obj in objs:
            obj.hide_render = hide
            obj.hide_viewport = hide


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------
def pose_smug(arm):
    """Rest pose already has hands on the hips; no bone deform (keeps the face stuck)."""
    clear_pose(arm)
    arm.scale = (1.0, 1.0, 1.0)
    arm.location = (0.0, 0.0, 0.0)
    arm.rotation_euler = (0.0, 0.0, 0.0)


def pose_scream(arm):
    """Squash about the feet. No bone deform, so the face stays on the head."""
    clear_pose(arm)
    arm.scale = (1.22, 1.14, 0.74)
    arm.location = (0.0, 0.0, 0.0)
    arm.rotation_euler = (0.0, 0.0, 0.0)


def pose_faint(arm):
    """Bones stay at rest. The fall is a rigid transform of the armature object."""
    clear_pose(arm)


def place_standing(arm):
    arm.location = (0, 0, 0)
    arm.rotation_euler = (0, 0, 0)
    arm.scale = (1, 1, 1)


def place_faint(arm):
    """Roll onto the back, slightly 3/4, resting on the floor."""
    arm.scale = (1.0, 1.0, 1.0)
    # Rx -90 maps the back (+Y) to the floor and the face (−Y) up.
    arm.rotation_euler = (math.radians(-85), math.radians(12), math.radians(32))
    arm.location = (0.0, 0.10, 0.40)


# ---------------------------------------------------------------------------
# Studio, camera, lights
# ---------------------------------------------------------------------------
def build_studio(col):
    # Floor
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    link(floor, col)
    assign(floor, principled_mat("Floor", srgb_to_linear(FLOOR_SRGB), roughness=0.62, coat=0.0, subsurface=0.0, specular=0.25))

    # Backdrop
    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 4.0, 3.0))
    back = bpy.context.active_object
    back.name = "Backdrop"
    back.rotation_euler = (math.radians(90), 0, 0)
    link(back, col)
    assign(back, principled_mat("Backdrop", srgb_to_linear(BACKDROP_SRGB), roughness=1.0, coat=0.0, subsurface=0.0, specular=0.0))

    world = bpy.data.worlds.new("StudioWorld")
    world.use_nodes = True
    bpy.context.scene.world = world
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = srgb_to_linear(WORLD_SRGB)
    bg.inputs["Strength"].default_value = 0.45

    def area(name, loc, energy, size, target, col_temp=None):
        bpy.ops.object.light_add(type="AREA", location=loc)
        lamp = bpy.context.active_object
        lamp.name = name
        lamp.data.energy = energy
        lamp.data.size = size
        lamp.data.shape = "DISK"
        look_at(lamp, target)
        link(lamp, col)
        return lamp

    # Key — upper left of camera (front-left-up)
    area("Key", (-2.2, -3.0, 3.2), 550, 2.4, (0, 0, 0.9))
    # Fill
    area("Fill", (2.4, -2.2, 1.6), 90, 3.5, (0, 0, 0.8))
    # Rim — back right
    area("Rim", (1.8, 2.6, 2.8), 220, 1.6, (0, 0, 1.0))
    # Gentle top
    area("Top", (0.0, -0.4, 4.2), 80, 3.0, (0, 0, 0.8))
    return floor


def build_cameras(col):
    def cam(name, loc, target, lens=85):
        data = bpy.data.cameras.new(name)
        data.lens = lens
        data.clip_start = 0.05
        data.clip_end = 40
        data.dof.use_dof = False
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        look_at(obj, target)
        link(obj, col)
        return obj

    # Standing: 50 mm, full body, room for the top pill and 15% floor.
    stand = cam("CamStand", (0.0, -4.6, 0.72), (0.0, 0.05, 0.62), lens=50)
    faint = cam("CamFaint", (1.1, -3.4, 2.8), (0.0, 0.15, 0.40), lens=50)
    return stand, faint


def setup_render(scene, res=(1080, 1920), samples=32, raytrace=True):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = res[0]
    scene.render.resolution_y = res[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.fps = FPS
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = 0.15
    e = scene.eevee
    e.taa_render_samples = samples
    e.use_shadows = True
    e.use_soft_shadows = True
    if hasattr(e, "use_bloom"):
        e.use_bloom = False
    if hasattr(e, "use_raytracing"):
        e.use_raytracing = raytrace
    if hasattr(e, "use_gtao"):
        e.use_gtao = True


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def key_obj(obj, data_path, frame, value=None):
    if value is not None:
        if data_path == "location":
            obj.location = value
        elif data_path == "rotation_euler":
            obj.rotation_euler = value
        elif data_path == "scale":
            obj.scale = value
        elif data_path == "hide_render":
            obj.hide_render = value
            obj.hide_viewport = value
    obj.keyframe_insert(data_path, frame=frame)


def build_reel_animation(arm, pieces, cam_stand, cam_faint, scene):
    total = int(round(REEL_SECONDS * FPS))  # 312
    a_end = int(SEG_A * FPS)                # 108
    b_end = int(SEG_B * FPS)                # 216
    scene.frame_start = 1
    scene.frame_end = total
    scene.frame_current = 1

    # Camera marker: standing through the squat, cut/lerp to faint cam on the fall.
    scene.camera = cam_stand
    cam_stand.data.keyframe_insert  # noqa: cameras switched by markers or by location
    # We'll animate CamStand to the faint viewpoint during the fall so there is
    # no hard cut. Start pose is standing, end pose copies CamFaint.
    start_loc = cam_stand.location.copy()
    start_rot = cam_stand.rotation_euler.copy()
    end_loc = cam_faint.location.copy()
    end_rot = cam_faint.rotation_euler.copy()
    key_obj(cam_stand, "location", 1, start_loc)
    key_obj(cam_stand, "rotation_euler", 1, start_rot)
    key_obj(cam_stand, "location", b_end, start_loc)
    key_obj(cam_stand, "rotation_euler", b_end, start_rot)
    key_obj(cam_stand, "location", b_end + int(0.9 * FPS), end_loc)
    key_obj(cam_stand, "rotation_euler", b_end + int(0.9 * FPS), end_rot)
    key_obj(cam_stand, "location", total, end_loc)
    key_obj(cam_stand, "rotation_euler", total, end_rot)
    for fc in cam_stand.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            k.interpolation = "BEZIER"

    # Expression visibility
    def key_pieces(frame, visible_keys):
        for key, objs in pieces.items():
            hide = key not in visible_keys
            for obj in objs:
                key_obj(obj, "hide_render", frame, hide)
                # also hide_viewport so depsgraph stays consistent
                obj.hide_viewport = hide
                obj.keyframe_insert("hide_viewport", frame=frame)

    key_pieces(1, {"smug"})
    key_pieces(a_end, {"smug"})
    key_pieces(a_end + 1, {"scream", "fx_scream"})
    key_pieces(b_end, {"scream", "fx_scream"})
    key_pieces(b_end + 1, {"faint", "fx_faint"})
    key_pieces(total, {"faint", "fx_faint"})

    # Bone / object poses
    # Section A: smug + bounce
    pose_smug(arm)
    for name in BONES:
        pose_bone(arm, name, rot_xyz_deg=tuple(math.degrees(a) for a in arm.pose.bones[name].rotation_euler),
                  scale=tuple(arm.pose.bones[name].scale),
                  location=tuple(arm.pose.bones[name].location),
                  frame=1)
        pose_bone(arm, name, rot_xyz_deg=tuple(math.degrees(a) for a in arm.pose.bones[name].rotation_euler),
                  scale=tuple(arm.pose.bones[name].scale),
                  location=tuple(arm.pose.bones[name].location),
                  frame=a_end)

    place_standing(arm)
    # Bounce on the armature object: 1.15 Hz, 3 cm
    for f in range(1, a_end + 1):
        t = (f - 1) / FPS
        z = 0.025 * (1 - math.cos(2 * math.pi * 1.15 * t))  # up and down, starts at 0
        sx = 1.0 + 0.02 * math.sin(2 * math.pi * 1.15 * t)
        sz = 1.0 - 0.03 * math.sin(2 * math.pi * 1.15 * t)
        key_obj(arm, "location", f, (0, 0, z))
        key_obj(arm, "scale", f, (sx, sx, sz))
        key_obj(arm, "rotation_euler", f, (0, 0, 0))

    # Section B: squat scream + shake. Transition 0.4 s.
    trans = int(0.4 * FPS)
    pose_scream(arm)
    for name in BONES:
        b = arm.pose.bones[name]
        pose_bone(arm, name,
                  rot_xyz_deg=tuple(math.degrees(a) for a in b.rotation_euler),
                  scale=tuple(b.scale),
                  location=tuple(b.location),
                  frame=a_end + trans)
        pose_bone(arm, name,
                  rot_xyz_deg=tuple(math.degrees(a) for a in b.rotation_euler),
                  scale=tuple(b.scale),
                  location=tuple(b.location),
                  frame=b_end)

    squat_scale = (1.22, 1.14, 0.74)
    key_obj(arm, "scale", a_end, (1, 1, 1))
    key_obj(arm, "scale", a_end + trans, squat_scale)
    key_obj(arm, "scale", b_end, squat_scale)
    for f in range(a_end + 1, b_end + 1):
        t = (f - (a_end + 1)) / FPS
        amp = 0.03 if f > a_end + trans else 0.03 * ((f - a_end) / max(1, trans))
        x = amp * math.sin(2 * math.pi * 10 * t)
        z = 0.01 * math.sin(2 * math.pi * 8 * t)
        key_obj(arm, "location", f, (x, 0, z))
        key_obj(arm, "rotation_euler", f, (0, 0, 0))

    # Section C: fall then settle
    fall = int(0.9 * FPS)
    pose_faint(arm)
    for name in BONES:
        b = arm.pose.bones[name]
        pose_bone(arm, name,
                  rot_xyz_deg=tuple(math.degrees(a) for a in b.rotation_euler),
                  scale=tuple(b.scale),
                  location=tuple(b.location),
                  frame=b_end + fall)
        pose_bone(arm, name,
                  rot_xyz_deg=tuple(math.degrees(a) for a in b.rotation_euler),
                  scale=tuple(b.scale),
                  location=tuple(b.location),
                  frame=total)

    # Interpolate armature object from standing to faint placement
    stand_loc, stand_rot, stand_s = (0, 0, 0), (0, 0, 0), (1, 1, 1)
    faint_loc = (0.0, 0.10, 0.40)
    faint_rot = (math.radians(-85), math.radians(12), math.radians(32))
    key_obj(arm, "location", b_end, stand_loc)
    key_obj(arm, "rotation_euler", b_end, stand_rot)
    key_obj(arm, "scale", b_end, stand_s)
    key_obj(arm, "location", b_end + fall, faint_loc)
    key_obj(arm, "rotation_euler", b_end + fall, faint_rot)
    key_obj(arm, "scale", b_end + fall, stand_s)
    # A tiny settle bounce after the fall
    settle = b_end + fall + int(0.3 * FPS)
    key_obj(arm, "location", min(settle, total), (faint_loc[0], faint_loc[1], faint_loc[2] + 0.04))
    key_obj(arm, "location", total, faint_loc)
    key_obj(arm, "rotation_euler", total, faint_rot)

    return total


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def render_still(scene, camera, path, samples=None):
    scene.camera = camera
    if samples:
        scene.eevee.taa_render_samples = samples
    scene.render.filepath = str(path)
    os.makedirs(path.parent, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print(f"WROTE {path}")


def render_animation(scene, camera, dirpath):
    scene.camera = camera
    dirpath.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(dirpath / "frame_")
    scene.render.use_file_extension = True
    bpy.ops.render.render(animation=True)
    print(f"WROTE ANIM {dirpath}")



def _world_bbox(obj):
    mat = obj.matrix_world
    corners = [mat @ Vector(c) for c in obj.bound_box]
    xs, ys, zs = zip(*[(c.x, c.y, c.z) for c in corners])
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _report_bbox(obj, label):
    lo, hi = _world_bbox(obj)
    size = hi - lo
    mid = (lo + hi) * 0.5
    print(f"BBOX {label} lo={tuple(round(v,3) for v in lo)} hi={tuple(round(v,3) for v in hi)} size={tuple(round(v,3) for v in size)} mid={tuple(round(v,3) for v in mid)}")


def _frame_standing_camera(cam, body):
    """Pull the camera back so the full body sits in the middle 62% of a 9:16 frame."""
    lo, hi = _world_bbox(body)
    mid = (lo + hi) * 0.5
    height = max(0.8, hi.z - lo.z)
    # 50 mm, sensor 36 mm vertical fit: v_fov = 2*atan(18/50)
    cam.data.lens = 50
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 36
    v_fov = 2 * math.atan((cam.data.sensor_height * 0.5) / cam.data.lens)
    # Character should occupy ~62% of the frame height (22% headroom, 16% floor).
    frame_h = height / 0.62
    dist = (frame_h * 0.5) / math.tan(v_fov * 0.5)
    cam.location = (mid.x, lo.y - dist, mid.z + 0.02)
    look_at(cam, (mid.x, mid.y, mid.z))
    print(f"CAM STAND loc={tuple(round(v,3) for v in cam.location)} dist={dist:.3f} height={height:.3f}")


def _frame_faint_camera(cam, body):
    lo, hi = _world_bbox(body)
    mid = (lo + hi) * 0.5
    size = hi - lo
    span = max(size.x, size.y, size.z, 0.8)
    cam.data.lens = 50
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 36
    v_fov = 2 * math.atan((cam.data.sensor_height * 0.5) / cam.data.lens)
    frame_h = span / 0.62
    dist = (frame_h * 0.5) / math.tan(v_fov * 0.5)
    # 3/4 elevated view, similar to the concept sheet.
    offset = Vector((dist * 0.35, -dist * 0.85, dist * 0.55))
    cam.location = mid + offset
    look_at(cam, mid)
    print(f"CAM FAINT loc={tuple(round(v,3) for v in cam.location)} mid={tuple(round(v,3) for v in mid)} dist={dist:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out = __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preview = args.mode == "preview"
    res = (540, 960) if preview else (1080, 1920)
    if args.samples:
        samples = args.samples
    elif preview:
        samples = 8
    elif args.mode == "reel":
        samples = 8
    else:
        samples = 32

    reset_scene()
    studio_col = collection("Studio")
    char_col = collection("Mascot")
    face_col = collection("Face")

    body = build_body(char_col)
    arm = build_armature(char_col, body)
    pieces, _eyes = build_face_library(face_col, body)
    build_studio(studio_col)
    cam_stand, cam_faint = build_cameras(studio_col)
    _report_bbox(body, "rest")
    _frame_standing_camera(cam_stand, body)

    scene = bpy.context.scene
    setup_render(scene, res=res, samples=samples, raytrace=not preview and args.mode != "reel")

    # Save a rebuildable .blend next to the stills (not committed).
    blend_path = out / "mascot.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"WROTE {blend_path}")

    if args.mode in ("preview", "stills", "all"):
        # SMUG
        place_standing(arm)
        pose_smug(arm)
        set_expression(pieces, "smug")
        bpy.context.view_layer.update()
        render_still(scene, cam_stand, out / "raw_smug.png")

        # SCREAM
        place_standing(arm)
        pose_scream(arm)
        set_expression(pieces, "scream")
        bpy.context.view_layer.update()
        render_still(scene, cam_stand, out / "raw_scream.png")

        # FAINT
        place_faint(arm)
        pose_faint(arm)
        set_expression(pieces, "faint")
        bpy.context.view_layer.update()
        _report_bbox(body, "faint")
        _frame_faint_camera(cam_faint, body)
        render_still(scene, cam_faint, out / "raw_faint.png")

    if args.mode in ("reel", "all"):
        # Rebuild animation from a clean standing start
        clear_pose(arm)
        place_standing(arm)
        set_expression(pieces, "smug")
        total = build_reel_animation(arm, pieces, cam_stand, cam_faint, scene)
        print(f"REEL frames 1-{total} at {FPS} fps")
        # Use the (animated) standing camera
        setup_render(scene, res=res, samples=samples if args.mode == "reel" else 8, raytrace=False)
        render_animation(scene, cam_stand, out / "frames")

    # Always save the posed file last
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print("DONE")


if __name__ == "__main__":
    main()
