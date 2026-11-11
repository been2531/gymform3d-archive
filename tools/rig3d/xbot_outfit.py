#!/usr/bin/env python3
"""Procedural boxing outfit kit for Mixamo X Bot.

All meshes and materials are generated in Blender — no third-party assets,
so the kit is commercially free (public-domain geometry we authored).

Import from any render script (do not require eval_xbot_boxing.py):

    from xbot_outfit import apply_outfit, COLORWAYS, clear_outfit
    handle = apply_outfit(arm, meshes, items=("gloves", "beanie"), colorway="A")

Items:
  gloves    — rigid-parented to mixamorig:LeftHand / RightHand
  beanie    — knit cap, rigid-parented to mixamorig:Head
  headgear  — open-face amateur pads + chin strap, parented to Head

Gloves and beanie leave the shoulder / chest glow area uncovered.
Headgear is an alternative to the beanie; do not wear both at once.

Mixamo FBX files themselves must not be redistributed; finished renders are fine.
"""
from __future__ import annotations

import math
from typing import Iterable

import bpy
import bmesh
from mathutils import Vector, Matrix, kdtree
from mathutils.bvhtree import BVHTree

# ---------------------------------------------------------------------------
# Colourways (linear RGB). Black is lifted so form still reads in the
# charcoal studio; pure 0.02 leather crushed to a silhouette under AgX.
# ---------------------------------------------------------------------------
COLORWAYS = {
    "A": {
        "name": "crimson_charcoal",
        "label": "A  ·  crimson gloves + charcoal beanie",
        "glove": (0.31, 0.040, 0.045),
        "glove_cuff": (0.07, 0.045, 0.045),
        "glove_strap": (0.04, 0.035, 0.035),
        "beanie": (0.085, 0.085, 0.090),
        "headgear": (0.10, 0.10, 0.105),
        "strap": (0.045, 0.045, 0.048),
    },
    "B": {
        "name": "black_matte",
        "label": "B  ·  black gloves + matte headgear",
        "glove": (0.058, 0.055, 0.052),
        "glove_cuff": (0.085, 0.082, 0.078),
        "glove_strap": (0.04, 0.038, 0.036),
        "beanie": (0.12, 0.13, 0.145),
        "headgear": (0.072, 0.070, 0.068),
        "strap": (0.040, 0.038, 0.036),
    },
}

# Conservative union of Fighting Idle + Lead Jab hand bounds, both sides,
# in hand-bone local centimetres (armature-local units). Expanded slightly
# so a glove sized once stays consistent across clips.
DEFAULT_HAND_ENV = {
    "min": (-6.60, -3.05, -3.45),
    "max": (6.60, 13.50, 6.40),
}

# Padding added outside that envelope, centimetres.
GLOVE_PAD = {
    "x": 1.35,
    "y_knuckle": 4.6,
    "y_cuff": 5.0,   # overlap the wrist joint so the masked fist leaves no gap
    "z_pinky": 1.35,
    "z_thumb": 2.15,
}

BEANIE_CLEARANCE_CM = 0.55
HEADGEAR_CLEARANCE_CM = 0.70
BROW_Y_CM = 6.0  # cuff line in head-bone local cm (mid-head, watch-cap)


# ---------------------------------------------------------------------------
# Bone / mesh helpers
# ---------------------------------------------------------------------------
def _clean(name: str) -> str:
    return name.lower().replace("mixamorig:", "").replace("mixamorig_", "")


def find_bone(arm, *cands):
    lower = {b.name.lower(): b.name for b in arm.pose.bones}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in cands:
        cl = _clean(c)
        for bn, raw in lower.items():
            if _clean(bn) == cl:
                return raw
    return None


def bone_world_matrix(arm, bone_name) -> Matrix:
    pb = arm.pose.bones[bone_name]
    return arm.matrix_world @ pb.matrix


def parent_to_bone(obj, arm, bone_name, keep_world=True):
    """Rigid-parent an object to a pose bone, preserving world transform.

    Blender's BONE parent space is the bone *tail*, so the parent-inverse
    is built from pose_matrix @ Translation((0, length, 0)).
    """
    pb = arm.pose.bones[bone_name]
    mw = obj.matrix_world.copy()
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    parent_space = arm.matrix_world @ pb.matrix @ Matrix.Translation((0.0, pb.length, 0.0))
    obj.matrix_parent_inverse = parent_space.inverted()
    if keep_world:
        obj.matrix_world = mw
    return obj


def _vg_max_weights(obj, pred) -> list[float]:
    nv = len(obj.data.vertices)
    weights = [0.0] * nv
    for vg in obj.vertex_groups:
        if not pred(vg.name):
            continue
        idx = vg.index
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == idx and g.weight > weights[v.index]:
                    weights[v.index] = g.weight
    return weights


def iter_weighted_world_verts(meshes, pred, thresh=0.25):
    deps = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        if o.type != "MESH" or not o.data.vertices:
            continue
        wmap = _vg_max_weights(o, pred)
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        try:
            mw = ev.matrix_world
            for i, v in enumerate(me.vertices):
                if i < len(wmap) and wmap[i] < thresh:
                    continue
                yield mw @ v.co
        finally:
            ev.to_mesh_clear()


def aabb_in_bone(arm, bone_name, world_pts):
    if not world_pts:
        return None
    inv = bone_world_matrix(arm, bone_name).inverted()
    xs, ys, zs = [], [], []
    for p in world_pts:
        loc = inv @ p
        xs.append(loc.x)
        ys.append(loc.y)
        zs.append(loc.z)
    return {
        "min": (min(xs), min(ys), min(zs)),
        "max": (max(xs), max(ys), max(zs)),
        "n": len(xs),
    }


def union_aabb(*aabbs):
    aabbs = [a for a in aabbs if a]
    if not aabbs:
        return None
    mins = [min(a["min"][i] for a in aabbs) for i in range(3)]
    maxs = [max(a["max"][i] for a in aabbs) for i in range(3)]
    return {"min": tuple(mins), "max": tuple(maxs), "n": sum(a.get("n", 0) for a in aabbs)}


def expand_aabb(env, pad_min, pad_max):
    mn = env["min"]
    mx = env["max"]
    return {
        "min": (mn[0] - pad_min[0], mn[1] - pad_min[1], mn[2] - pad_min[2]),
        "max": (mx[0] + pad_max[0], mx[1] + pad_max[1], mx[2] + pad_max[2]),
    }


def scan_hand_envelope(arm, meshes, bone_name, pred, frame_start, frame_end, step=2):
    """Max AABB of hand verts in that hand-bone's local space across frames."""
    sc = bpy.context.scene
    saved = sc.frame_current
    acc = None
    f = int(frame_start)
    while f <= int(frame_end):
        sc.frame_set(f)
        bpy.context.view_layer.update()
        pts = list(iter_weighted_world_verts(meshes, pred))
        box = aabb_in_bone(arm, bone_name, pts)
        acc = union_aabb(acc, box)
        f += step
    sc.frame_set(saved)
    bpy.context.view_layer.update()
    return acc


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def _principled(mat):
    nt = mat.node_tree
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    return None


def _set(bsdf, name, value):
    if bsdf and name in bsdf.inputs:
        bsdf.inputs[name].default_value = value


def make_leather_material(name, color, roughness=0.64):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.location = (-300, -80)
    tex.inputs["Scale"].default_value = 55.0
    tex.inputs["Detail"].default_value = 8.0
    tex.inputs["Roughness"].default_value = 0.55
    ramp = nt.nodes.new("ShaderNodeRamp") if False else nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-40, -80)
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (80, -160)
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.04
    mix = nt.nodes.new("ShaderNodeMix")
    mix.location = (80, 80)
    mix.data_type = "FLOAT"
    if "Factor" in mix.inputs:
        mix.inputs["Factor"].default_value = 0.22
    # roughness variation: base ± grain
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "MULTIPLY_ADD"
    add.location = (80, -20)
    add.inputs[1].default_value = 0.10
    add.inputs[2].default_value = roughness
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(tex.outputs["Fac"], add.inputs[0])
    nt.links.new(add.outputs["Value"], bsdf.inputs["Roughness"])
    if "Normal" in bsdf.inputs:
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    _set(bsdf, "Base Color", (*color, 1.0))
    _set(bsdf, "Specular IOR Level", 0.32)
    _set(bsdf, "Specular", 0.32)
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Coat Weight", 0.04)
    _set(bsdf, "Coat Roughness", 0.35)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_knit_material(name, color):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (820, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (520, 0)
    # Object coords so ribs follow the cap (Y is bone-up in our mesh space).
    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-520, 0)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-340, 40)
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.location = (-160, 80)
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.wave_profile = "SIN"
    wave.inputs["Scale"].default_value = 9.5
    wave.inputs["Distortion"].default_value = 0.6
    wave.inputs["Detail"].default_value = 3.0
    # rotate generated X around the cap's Y axis ≈ circumference ribs
    # Using object X (we will also mix a second wave on Z).
    wave2 = nt.nodes.new("ShaderNodeTexWave")
    wave2.location = (-160, -120)
    wave2.wave_type = "BANDS"
    wave2.bands_direction = "Z"
    wave2.wave_profile = "SIN"
    wave2.inputs["Scale"].default_value = 9.5
    wave2.inputs["Distortion"].default_value = 0.5
    wave2.inputs["Detail"].default_value = 3.0
    mixw = nt.nodes.new("ShaderNodeMix")
    mixw.location = (40, 0)
    mixw.data_type = "FLOAT"
    if "Factor" in mixw.inputs:
        mixw.inputs["Factor"].default_value = 0.5
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-160, -280)
    noise.inputs["Scale"].default_value = 40.0
    noise.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (280, -80)
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.08
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.location = (40, -180)
    add.use_clamp = True
    nt.links.new(texcoord.outputs["Object"], sep.inputs["Vector"])
    nt.links.new(texcoord.outputs["Object"], wave.inputs["Vector"])
    nt.links.new(texcoord.outputs["Object"], wave2.inputs["Vector"])
    nt.links.new(texcoord.outputs["Object"], noise.inputs["Vector"])
    # Mix.outputs may be "Result" (4.2 float mix) or "Color"
    def _mix_out(n):
        return n.outputs.get("Result") or n.outputs.get("Color") or n.outputs[0]

    def _mix_a(n):
        return n.inputs.get("A") or n.inputs.get("A_Float") or n.inputs[2]

    def _mix_b(n):
        return n.inputs.get("B") or n.inputs.get("B_Float") or n.inputs[3]

    nt.links.new(wave.outputs["Fac"], _mix_a(mixw))
    nt.links.new(wave2.outputs["Fac"], _mix_b(mixw))
    nt.links.new(_mix_out(mixw), add.inputs[0])
    # scale noise down
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.location = (40, -300)
    mul.inputs[1].default_value = 0.35
    nt.links.new(noise.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], add.inputs[1])
    nt.links.new(add.outputs["Value"], bump.inputs["Height"])
    if "Normal" in bsdf.inputs:
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    _set(bsdf, "Base Color", (*color, 1.0))
    _set(bsdf, "Roughness", 0.90)
    _set(bsdf, "Specular IOR Level", 0.12)
    _set(bsdf, "Specular", 0.12)
    _set(bsdf, "Sheen Weight", 0.45)
    _set(bsdf, "Sheen Roughness", 0.55)
    _set(bsdf, "Sheen", 0.45)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_foam_leather_material(name, color, roughness=0.70):
    """Duller padded-leather for amateur headgear (less sheen than gloves)."""
    mat = make_leather_material(name, color, roughness=roughness)
    bsdf = _principled(mat)
    _set(bsdf, "Coat Weight", 0.0)
    _set(bsdf, "Specular IOR Level", 0.22)
    _set(bsdf, "Specular", 0.22)
    return mat


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------
def _new_mesh_object(name, mesh, collection):
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _shade_smooth(obj):
    mesh = obj.data
    if hasattr(mesh, "shade_smooth"):
        try:
            mesh.shade_smooth()
            return
        except Exception:
            pass
    for p in mesh.polygons:
        p.use_smooth = True


def _apply_modifier(obj, mod):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    finally:
        obj.select_set(False)


def _ensure_active(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _boolean_union(target, *others, cleanup=True):
    for other in others:
        if other is None:
            continue
        mod = target.modifiers.new("BoolUnion", "BOOLEAN")
        mod.operation = "UNION"
        try:
            mod.solver = "EXACT"
        except Exception:
            pass
        mod.object = other
        _apply_modifier(target, mod)
        if cleanup:
            bpy.data.objects.remove(other, do_unlink=True)
    return target


def _boolean_difference(target, cutter, cleanup=True):
    mod = target.modifiers.new("BoolDiff", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    try:
        mod.solver = "EXACT"
    except Exception:
        pass
    mod.object = cutter
    _apply_modifier(target, mod)
    if cleanup:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def _voxel_remesh(obj, voxel=0.28):
    mod = obj.modifiers.new("Voxel", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    try:
        mod.use_smooth_shade = True
    except Exception:
        pass
    _apply_modifier(obj, mod)
    _shade_smooth(obj)
    return obj


def _smooth_mesh(obj, iterations=12, factor=0.55):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for _ in range(iterations):
        bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=factor, use_axis_x=True,
                              use_axis_y=True, use_axis_z=True)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _icosphere(name, radius, subdivisions, collection, location=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius)
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(location))
    bm.to_mesh(mesh)
    bm.free()
    return obj


def _ellipsoid(name, radii, subdivisions, collection, location=(0, 0, 0)):
    obj = _icosphere(name, 1.0, subdivisions, collection, location=(0, 0, 0))
    sx, sy, sz = radii
    for v in obj.data.vertices:
        v.co.x *= sx
        v.co.y *= sy
        v.co.z *= sz
        v.co += Vector(location)
    obj.data.update()
    return obj


def _cylinder(name, radius, depth, collection, location, axis="Y", segments=24):
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False,
        segments=segments, radius1=radius, radius2=radius, depth=depth,
    )
    # created along Z; rotate into axis
    if axis == "Y":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=Vector((0, 0, 0)),
                         matrix=Matrix.Rotation(math.radians(-90), 3, "X"))
    elif axis == "X":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=Vector((0, 0, 0)),
                         matrix=Matrix.Rotation(math.radians(90), 3, "Y"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(location))
    bm.to_mesh(mesh)
    bm.free()
    return obj


def _torus_segment_approx(name, collection, major, minor, center, normal, start_deg, end_deg, segs=20, sides=10):
    """A short bent tube (for chin strap / brow pad). Normal is the torus axis."""
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    # build in XY (axis Z), then rotate
    verts_rows = []
    nseg = max(4, segs)
    ns = max(6, sides)
    a0 = math.radians(start_deg)
    a1 = math.radians(end_deg)
    for i in range(nseg + 1):
        a = a0 + (a1 - a0) * (i / nseg)
        cx = major * math.cos(a)
        cy = major * math.sin(a)
        row = []
        for j in range(ns):
            b = 2 * math.pi * j / ns
            lx = (major + minor * math.cos(b)) * math.cos(a)
            ly = (major + minor * math.cos(b)) * math.sin(a)
            lz = minor * math.sin(b)
            row.append(bm.verts.new((lx, ly, lz)))
        verts_rows.append(row)
    bm.verts.ensure_lookup_table()
    for i in range(nseg):
        r0 = verts_rows[i]
        r1 = verts_rows[i + 1]
        for j in range(ns):
            j2 = (j + 1) % ns
            bm.faces.new((r0[j], r0[j2], r1[j2], r1[j]))
    # cap ends
    try:
        bm.faces.new(list(reversed(verts_rows[0])))
        bm.faces.new(verts_rows[-1])
    except Exception:
        pass
    # rotate Z-axis torus so its axis matches `normal`, and its +X points toward a helper
    n = Vector(normal).normalized()
    quat = Vector((0, 0, 1)).rotation_difference(n)
    rot = quat.to_matrix()
    bmesh.ops.rotate(bm, verts=bm.verts, cent=Vector((0, 0, 0)), matrix=rot)
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(center))
    bm.to_mesh(mesh)
    bm.free()
    _shade_smooth(obj)
    return obj


# ---------------------------------------------------------------------------
# Gloves
# ---------------------------------------------------------------------------
def _padded_glove_env(hand_env):
    return expand_aabb(
        hand_env,
        (GLOVE_PAD["x"], GLOVE_PAD["y_cuff"], GLOVE_PAD["z_pinky"]),
        (GLOVE_PAD["x"], GLOVE_PAD["y_knuckle"], GLOVE_PAD["z_thumb"]),
    )




def build_glove_mesh(name, collection, hand_env, wrist_radius=None):
    """Solid mitten in hand-bone local centimetres, origin at the bone head.

    +Y = knuckles, +Z = thumb side, cuff opening toward -Y. The body ellipsoid
    is centred on the full cuff-to-knuckle span so the wrist ring stays
    attached (a knuckle-biased body previously left a detached band mid-forearm).
    """
    env = _padded_glove_env(hand_env or DEFAULT_HAND_ENV)
    mn, mx = Vector(env["min"]), Vector(env["max"])
    center = (mn + mx) * 0.5
    # Slight knuckle bias (55%), not 88% — the body must overlap the cuff.
    body_c = Vector((center.x, mn.y * 0.42 + mx.y * 0.58, center.z))
    rx = max(4.5, (mx.x - mn.x) * 0.50)
    ry = max(6.0, (mx.y - mn.y) * 0.52)
    rz = max(4.0, (mx.z - mn.z) * 0.50)

    body = _ellipsoid(name + "_body", (rx * 0.96, ry, rz * 0.88), 3, collection, location=body_c)
    knuckle_c = Vector((center.x, mx.y - 1.4, center.z))
    knuckle = _ellipsoid(name + "_knuckle", (rx * 0.88, 3.0, rz * 0.90), 2, collection, location=knuckle_c)
    # Thumb is an elongated cap that heavily overlaps the body (not a detached cherry).
    thumb_c = Vector((center.x, center.y * 0.35 + mx.y * 0.40, mx.z * 0.55))
    thumb = _ellipsoid(name + "_thumb", (rx * 0.55, ry * 0.38, rz * 0.48), 3, collection, location=thumb_c)
    cuff_y = mn.y + 2.0
    cuff_r = min(rx * 0.86, 4.4)
    cuff = _ellipsoid(name + "_cuff", (cuff_r, 2.2, cuff_r * 0.92), 2, collection,
                      location=(center.x, cuff_y, center.z))

    _boolean_union(body, knuckle, thumb, cuff, cleanup=True)
    _voxel_remesh(body, voxel=0.28)
    _smooth_mesh(body, iterations=8, factor=0.42)
    _shade_smooth(body)
    body.name = name
    # Solid glove: no cuff cavity. A boolean hole previously punched through
    # the mid-glove wall and showed the clay hand. The wrist is covered by
    # the cuff volume and the forearm disappears into it.
    return body, {"env": env, "cavity_r": None, "cavity_y": None}


def _full_torus(name, collection, major, minor, location, axis="Y", major_seg=28, minor_seg=12):
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    for i in range(major_seg):
        a0 = 2 * math.pi * i / major_seg
        a1 = 2 * math.pi * (i + 1) / major_seg
        ring0, ring1 = [], []
        for j in range(minor_seg):
            b = 2 * math.pi * j / minor_seg
            def pt(a, b=b):
                return Vector((
                    (major + minor * math.cos(b)) * math.cos(a),
                    (major + minor * math.cos(b)) * math.sin(a),
                    minor * math.sin(b),
                ))
            ring0.append(pt(a0))
            ring1.append(pt(a1))
        v0 = [bm.verts.new(p) for p in ring0]
        v1 = [bm.verts.new(p) for p in ring1]
        for j in range(minor_seg):
            j2 = (j + 1) % minor_seg
            bm.faces.new((v0[j], v0[j2], v1[j2], v1[j]))
    if axis == "Y":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=Vector((0, 0, 0)),
                         matrix=Matrix.Rotation(math.radians(-90), 3, "X"))
    elif axis == "X":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=Vector((0, 0, 0)),
                         matrix=Matrix.Rotation(math.radians(90), 3, "Y"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(location))
    bm.to_mesh(mesh)
    bm.free()
    _shade_smooth(obj)
    return obj



def _head_points(meshes, arm, head_bone, thresh=0.30):
    pred = lambda n: _clean(n) == "head"
    pts = []
    inv = bone_world_matrix(arm, head_bone).inverted()
    for wp in iter_weighted_world_verts(meshes, pred, thresh=thresh):
        pts.append(inv @ wp)
    return pts


def _head_kdtree(pts):
    kd = kdtree.KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()
    return kd


def _head_origin(pts):
    o = Vector((0.0, 0.0, 0.0))
    for p in pts:
        o += p
    o /= max(1, len(pts))
    # bias slightly toward the skull centre (back from the face, up from the jaw)
    o.z = o.z * 0.45
    return o


def _hit_head(pts, origin, direction):
    """Approximate surface hit: head vert most aligned with a ray from origin."""
    d = direction.normalized()
    best, best_score = None, -1e9
    for p in pts:
        v = p - origin
        if v.length < 1e-8:
            continue
        cos = v.normalized().dot(d)
        if cos < 0.25:
            continue
        score = v.dot(d) + 0.8 * cos
        if score > best_score:
            best_score, best = score, p
    if best is None:
        best = max(pts, key=lambda p: (p - origin).dot(d))
    n = best - origin
    if n.length < 1e-8:
        n = d
    else:
        n.normalize()
    return best, n


def _push_off_surface(obj, pts, clearance, only_inside=True):
    """Move verts that have sunk *into* the skull back out to `clearance`.

    Only the inside case is corrected. Outer pad / cuff volume is left alone
    — a global push flattened the headgear into a skull-cap.
    """
    if not pts or not obj.data.vertices:
        return 0
    kd = _head_kdtree(pts)
    centre = _head_origin(pts)
    moved = 0
    for v in obj.data.vertices:
        _co, idx, dist = kd.find(v.co)
        if idx is None:
            continue
        surface = pts[idx]
        outward = surface - Vector((centre.x, centre.y - 1.5, centre.z))
        if outward.length < 1e-6:
            outward = v.co - centre
        outward.normalize()
        signed = (v.co - surface).dot(outward)
        if only_inside and signed >= 0.0:
            continue
        if signed < clearance:
            v.co = surface + outward * clearance
            moved += 1
    obj.data.update()
    return moved


def build_beanie(name, collection, head_pts, brow_y=BROW_Y_CM, clearance=BEANIE_CLEARANCE_CM):
    """Fitted watch-cap with a clean folded cuff, in head-bone local cm."""
    upper = [p for p in head_pts if p.y >= brow_y - 2.0] or list(head_pts)
    xs = [p.x for p in upper]
    ys = [p.y for p in upper]
    zs = [p.z for p in upper]
    cx = 0.0
    cy = (min(ys) + max(ys)) * 0.50
    cz = (min(zs) * 0.55 + max(zs) * 0.45)
    rx = max(abs(min(xs)), abs(max(xs))) + clearance + 0.20
    ry = (max(ys) - cy) + clearance + 0.35
    rz_front = (max(zs) - cz) * 0.95 + clearance
    rz_back = (cz - min(zs)) + clearance + 0.25

    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co.x, v.co.y, v.co.z
        zscale = rz_front if z >= 0.0 else rz_back
        v.co = Vector((cx + x * rx, cy + y * ry, cz + z * zscale))
    # Clean planar cut at the cuff line (avoids the jagged deleted-vert rim).
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bmesh.ops.bisect_plane(
        bm,
        geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
        dist=0.0001,
        plane_co=Vector((0.0, brow_y, 0.0)),
        plane_no=Vector((0.0, 1.0, 0.0)),
        clear_inner=True,
    )
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    # Single clean cuff: extrude the rim down and slightly out, then stop.
    # A multi-fold previously frayed into a jagged hairline.
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        res = bmesh.ops.extrude_edge_only(bm, edges=boundary)
        verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
        for v in verts:
            radial = Vector((v.co.x - cx, 0.0, v.co.z - cz))
            if radial.length > 1e-6:
                radial.normalize()
            v.co += radial * 0.45
            v.co.y = brow_y - 2.8
        # a short second extrusion straight down to give the cuff a flat hem
        bm.edges.ensure_lookup_table()
        boundary2 = [e for e in bm.edges if e.is_boundary]
        res2 = bmesh.ops.extrude_edge_only(bm, edges=boundary2)
        verts2 = [g for g in res2["geom"] if isinstance(g, bmesh.types.BMVert)]
        for v in verts2:
            v.co.y = brow_y - 3.2
        try:
            bm.faces.new(verts2)
        except Exception:
            pass
    # Flare the cuff so the hem sits outside the forehead, not cut into it.
    bm.verts.ensure_lookup_table()
    def _head_r(y, direc_xz):
        d = direc_xz.normalized()
        best = 0.0
        for p in head_pts:
            if abs(p.y - y) > 2.8:
                continue
            xz = Vector((p.x - cx, 0.0, p.z - cz))
            if xz.length < 1e-8:
                continue
            if xz.normalized().dot(d) < 0.5:
                continue
            best = max(best, xz.dot(d))
        return best
    for v in bm.verts:
        if v.co.y > brow_y + 0.6:
            continue
        xz = Vector((v.co.x - cx, 0.0, v.co.z - cz))
        if xz.length < 1e-8:
            continue
        need = _head_r(v.co.y, xz) + clearance + 0.55
        if xz.length < need:
            off = xz.normalized() * need
            v.co.x = cx + off.x
            v.co.z = cz + off.z
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    sol = obj.modifiers.new("Thick", "SOLIDIFY")
    sol.thickness = 0.40
    sol.offset = 1.0
    _apply_modifier(obj, sol)
    # Only un-sink verts that actually entered the skull; keep cuff volume.
    moved = _push_off_surface(obj, head_pts, clearance, only_inside=True)
    _smooth_mesh(obj, iterations=3, factor=0.20)
    _shade_smooth(obj)
    return obj, {"moved": moved, "brow_y": brow_y, "rx": rx, "ry": ry}



def _place_pad(name, collection, pts, origin, direction, radii, clearance, normal_scale=None):
    """Flattened ellipsoid pad sitting just outside the skull along `direction`.

    `radii` is (tangent_a, tangent_b, thickness). The thin axis is aligned
    with the surface normal so the pad hugs instead of standing off as a ball.
    """
    hit, n = _hit_head(pts, origin, direction)
    thick = radii[2]
    loc = hit + n * (clearance + thick * 0.55)
    obj = _ellipsoid(name, radii, 3, collection, location=(0, 0, 0))
    # Rotate local +Z (the thin axis of our radii if we pass (a,b,thick) and
    # then swap) — easier: scale in a basis where Z = normal.
    # Rebuild orientation: object Y up-ish, Z = normal.
    z = n
    up = Vector((0.0, 1.0, 0.0))
    if abs(z.dot(up)) > 0.9:
        up = Vector((1.0, 0.0, 0.0))
    x = up.cross(z).normalized()
    y = z.cross(x).normalized()
    basis = Matrix((
        (x.x, y.x, z.x),
        (x.y, y.y, z.y),
        (x.z, y.z, z.z),
    ))
    # verts currently along XYZ with radii; map X->x, Y->y, Z->z, then translate
    # _ellipsoid already scaled verts to radii around origin.
    for v in obj.data.vertices:
        v.co = loc + basis @ v.co
    obj.data.update()
    _shade_smooth(obj)
    return obj, loc, n


def _loft_tube(name, collection, path, radius, closed=True, segs=8):
    """Loft a circular tube through `path` points (bone-local cm)."""
    if len(path) < 2:
        raise ValueError("tube needs at least 2 points")
    pts = [Vector(p) for p in path]
    if closed:
        pts = pts + [pts[0]]
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    rings = []
    nseg = len(pts) if not closed else len(pts) - 1
    for i, p in enumerate(pts):
        prev = pts[i - 1] if i > 0 else (pts[-2] if closed else pts[i + 1])
        nxt = pts[i + 1] if i < len(pts) - 1 else (pts[1] if closed else pts[i - 1])
        tangent = (nxt - prev)
        if tangent.length < 1e-6:
            tangent = nxt - p if (nxt - p).length > 1e-6 else Vector((0, 1, 0))
        tangent.normalize()
        up = Vector((0, 1, 0)) if abs(tangent.dot(Vector((0, 1, 0)))) < 0.9 else Vector((1, 0, 0))
        x = up.cross(tangent).normalized()
        y = tangent.cross(x).normalized()
        ring = []
        for j in range(segs):
            a = 2 * math.pi * j / segs
            ring.append(bm.verts.new(p + (x * math.cos(a) + y * math.sin(a)) * radius))
        rings.append(ring)
        if closed and i == len(pts) - 1:
            break
    bm.verts.ensure_lookup_table()
    n_rings = len(rings)
    for i in range(n_rings - 1):
        r0, r1 = rings[i], rings[i + 1]
        for j in range(segs):
            j2 = (j + 1) % segs
            try:
                bm.faces.new((r0[j], r0[j2], r1[j2], r1[j]))
            except Exception:
                pass
    if closed and n_rings > 2:
        r0, r1 = rings[-1], rings[0]
        for j in range(segs):
            j2 = (j + 1) % segs
            try:
                bm.faces.new((r0[j], r0[j2], r1[j2], r1[j]))
            except Exception:
                pass
    else:
        try:
            bm.faces.new(list(reversed(rings[0])))
            bm.faces.new(rings[-1])
        except Exception:
            pass
    bm.to_mesh(mesh)
    bm.free()
    _shade_smooth(obj)
    return obj


def _head_ring(pts, origin, y_target, offset, n=24, y_window=3.5):
    """Offset loop around the skull at a given Y (forehead / hat-band latitude)."""
    ring = []
    for i in range(n):
        a = 2 * math.pi * i / n
        direc = Vector((math.cos(a), 0.15, math.sin(a)))
        cand = []
        d_xz = Vector((math.cos(a), 0.0, math.sin(a)))
        for p in pts:
            if abs(p.y - y_target) > y_window:
                continue
            xz = Vector((p.x - origin.x, 0.0, p.z - origin.z))
            if xz.length < 1e-6:
                continue
            if xz.normalized().dot(d_xz) < 0.55:
                continue
            cand.append(p)
        if cand:
            hit = max(cand, key=lambda p: Vector((p.x - origin.x, 0, p.z - origin.z)).length)
            nrm = Vector((hit.x - origin.x, 0.0, hit.z - origin.z))
            if nrm.length < 1e-6:
                nrm = d_xz
            nrm.normalize()
            # keep the ring at roughly y_target
            hit = Vector((hit.x, y_target * 0.7 + hit.y * 0.3, hit.z))
        else:
            hit, nrm = _hit_head(pts, origin, direc)
        ring.append(hit + nrm * offset)
    return ring


def build_headgear(name, collection, head_pts, clearance=HEADGEAR_CLEARANCE_CM):
    """Open-face amateur headgear: a padded hoop, flattened pads, chin strap.

    Built as separate objects (no union remesh) so thin straps stay intact
    and pads keep their volume instead of shrinking onto the skull.
    """
    if not head_pts:
        raise RuntimeError("no head verts to fit headgear")
    origin = _head_origin(head_pts)
    ys = [p.y for p in head_pts]
    y0, y1 = min(ys), max(ys)
    brow_y = y0 + (y1 - y0) * 0.60
    objs = []

    hoop = _loft_tube(
        name + "_hoop", collection,
        _head_ring(head_pts, origin, brow_y, offset=clearance + 1.15, n=28),
        radius=1.35, closed=True, segs=8,
    )
    objs.append(hoop)

    # Flattened pads: (tangent, tangent, thickness) with thickness along the normal
    brow, _, _ = _place_pad(
        name + "_brow", collection, head_pts, origin,
        Vector((0.0, 0.40, 1.0)), radii=(6.8, 2.0, 1.35), clearance=clearance)
    objs.append(brow)
    for side, sx in (("L", 1.0), ("R", -1.0)):
        cheek, _, _ = _place_pad(
            name + f"_cheek_{side}", collection, head_pts, origin,
            Vector((sx * 0.95, -0.20, 0.55)), radii=(2.15, 3.10, 1.25),
            clearance=clearance)
        objs.append(cheek)

    # Crown strap: left temple over the top to the right temple
    crown_path = []
    for direc in (
        Vector((-1.0, 0.35, 0.05)),
        Vector((-0.55, 0.85, 0.0)),
        Vector((0.0, 1.0, -0.05)),
        Vector((0.55, 0.85, 0.0)),
        Vector((1.0, 0.35, 0.05)),
    ):
        hit, n = _hit_head(head_pts, origin, direc)
        crown_path.append(hit + n * (clearance + 1.05))
    crown = _loft_tube(name + "_crown", collection, crown_path, radius=1.15, closed=False, segs=8)
    objs.append(crown)

    # Rear connector: a short hoop arc at the occiput (already partly in the hoop)
    rear, _, _ = _place_pad(
        name + "_rear", collection, head_pts, origin,
        Vector((0.0, 0.20, -1.0)), radii=(4.2, 2.2, 1.30), clearance=clearance)
    objs.append(rear)

    # Chin strap: left jaw -> under chin -> right jaw, standing off the surface
    strap_path = []
    for direc in (
        Vector((-0.9, -0.45, 0.35)),
        Vector((-0.45, -0.85, 0.45)),
        Vector((0.0, -1.0, 0.40)),
        Vector((0.45, -0.85, 0.45)),
        Vector((0.9, -0.45, 0.35)),
    ):
        hit, n = _hit_head(head_pts, origin, direc)
        strap_path.append(hit + n * (clearance + 0.55))
    strap = _loft_tube(name + "_strap", collection, strap_path, radius=0.36, closed=False, segs=7)
    objs.append(strap)

    info = {
        "moved": 0,
        "n_parts": len(objs),
        "note": (
            "Open-face amateur headgear: padded hoop, forehead and cheek pads, "
            "crown strap and a thin chin strap. Cheek pads and the strap hide "
            "part of the jaw, so chin-tuck / guard is slightly harder to read "
            "than a beanie or bare head."
        ),
    }
    return objs, info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _hand_pred(side):
    token = "lefthand" if side == "L" else "righthand"

    def _p(n):
        return token in _clean(n).replace("_", "")

    return _p


def _frame_range(arm):
    frames = []
    if arm.animation_data and arm.animation_data.action:
        act = arm.animation_data.action
        frames = [int(act.frame_range[0]), int(act.frame_range[1])]
    if not frames or frames[1] <= frames[0]:
        sc = bpy.context.scene
        frames = [sc.frame_start, sc.frame_end]
    return frames


def measure_hand_envelopes(arm, meshes, scan_frames=True, step=2):
    names = {
        "L": find_bone(arm, "mixamorig:LeftHand", "LeftHand", "hand_l"),
        "R": find_bone(arm, "mixamorig:RightHand", "RightHand", "hand_r"),
    }
    f0, f1 = _frame_range(arm)
    out = {}
    for side in ("L", "R"):
        bn = names[side]
        if not bn:
            continue
        measured = None
        if scan_frames:
            measured = scan_hand_envelope(arm, meshes, bn, _hand_pred(side), f0, f1, step=step)
        env = union_aabb(DEFAULT_HAND_ENV, measured) if measured else dict(DEFAULT_HAND_ENV)
        # keep X symmetric so one construction serves both hands
        ax = max(abs(env["min"][0]), abs(env["max"][0]))
        env = {
            "min": (-ax, env["min"][1], env["min"][2]),
            "max": (ax, env["max"][1], env["max"][2]),
            "n": env.get("n", 0),
            "bone": bn,
            "measured": measured,
        }
        out[side] = env
    return out


def _wrist_radius(arm, meshes, bone_name, side):
    pts = []
    inv = bone_world_matrix(arm, bone_name).inverted()
    for wp in iter_weighted_world_verts(meshes, _hand_pred(side), thresh=0.25):
        loc = inv @ wp
        if -3.2 <= loc.y <= 0.4:
            pts.append(math.hypot(loc.x, loc.z))
    if not pts:
        return 3.35
    pts.sort()
    return float(pts[int(len(pts) * 0.85)])


def _assign_mat(obj, mat, slot_mat=None):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.active_material = mat



def _glove_bvh_world(glove):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = glove.evaluated_get(deps)
    me = ev.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
    finally:
        ev.to_mesh_clear()
    bmesh.ops.transform(bm, verts=bm.verts, matrix=glove.matrix_world)
    bvh = BVHTree.FromBMesh(bm, epsilon=0.0001)
    bm.free()
    return bvh



def mask_inside_gloves(meshes, handle, margin_cm=0.45):
    """Hide character verts that sit in the glove along the hand-bone axis.

    A signed-distance mask followed the bumpy cuff and left a ragged clay
    ring. A plane cut in hand-bone space (Y >= cuff lip, inside the cuff
    radius) hides the fist cleanly so the leather can cover the join.
    """
    hidden = {}
    for side, info in (handle.get("glove_info") or {}).items():
        gname = next((n for n in handle["objects"] if n.startswith(f"XBotGlove_{side}_")), None)
        if not gname or gname not in bpy.data.objects:
            continue
        bone_name = handle["bones"][gname]
        env = info["env"]
        # Cut a little inside the cuff so the leather overlaps the remaining arm.
        plane_y = env["min"][1] + 1.4
        rx = max(abs(env["min"][0]), abs(env["max"][0])) + 1.5
        rz = max(abs(env["min"][2]), abs(env["max"][2])) + 1.5
        inv = bone_world_matrix(bpy.data.objects[gname].parent, bone_name).inverted()
        # parent is the armature
        pred = _hand_pred(side)
        deps = bpy.context.evaluated_depsgraph_get()
        for o in meshes:
            if o.type != "MESH" or not o.data.vertices:
                continue
            weights = _vg_max_weights(o, pred)
            ev = o.evaluated_get(deps)
            me = ev.to_mesh()
            idxs = []
            try:
                mw = ev.matrix_world
                for i, v in enumerate(me.vertices):
                    # Always hide hand-weighted verts — they are the fist.
                    if i < len(weights) and weights[i] >= 0.20:
                        idxs.append(i)
                        continue
                    loc = inv @ (mw @ v.co)
                    if loc.y < plane_y:
                        continue
                    # elliptical cap in XZ
                    if (loc.x / rx) ** 2 + (loc.z / rz) ** 2 <= 1.05:
                        idxs.append(i)
            finally:
                ev.to_mesh_clear()
            if not idxs:
                continue
            vg_name = "xbot_glove_hide"
            vg = o.vertex_groups.get(vg_name) or o.vertex_groups.new(name=vg_name)
            vg.add(sorted(set(idxs)), 1.0, "ADD")
            hidden[o.name] = hidden.get(o.name, 0) + len(set(idxs))
    for o in meshes:
        if o.name not in hidden:
            continue
        mod_name = "XBotGloveMask"
        if mod_name not in o.modifiers:
            mod = o.modifiers.new(mod_name, "MASK")
        else:
            mod = o.modifiers[mod_name]
        mod.vertex_group = "xbot_glove_hide"
        try:
            mod.invert_vertex_group = True
        except Exception:
            pass
    handle["masked"] = hidden
    return hidden


def apply_outfit(arm, meshes, *, items=("gloves", "beanie"), colorway="A",
                 scan_frames=True, collection_name="XBotOutfit"):
    """Build and bone-parent the requested items. Returns a handle dict.

    `items` is a subset of {"gloves", "beanie", "headgear", "strap"} ("strap"
    is a wrist band and is implied by gloves). Prefer beanie XOR headgear.
    """
    if colorway not in COLORWAYS:
        raise ValueError(f"unknown colorway {colorway!r}; expected one of {list(COLORWAYS)}")
    colors = COLORWAYS[colorway]
    items = tuple(items)
    if "beanie" in items and "headgear" in items:
        # Allowed technically, but looks wrong; keep both only if the caller
        # really asked — still warn in the handle.
        both = True
    else:
        both = False

    col = bpy.data.collections.get(collection_name)
    if col is None:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)

    handle = {
        "collection": collection_name,
        "colorway": colorway,
        "colorway_name": colors["name"],
        "items": list(items),
        "objects": [],
        "bones": {},
        "notes": [],
        "source": "procedural (authored in Blender, no third-party mesh)",
        "license": "original geometry; commercially free",
    }
    if both:
        handle["notes"].append("beanie and headgear both requested; they will overlap")

    leather = make_leather_material(f"XBotGloveLeather_{colorway}", colors["glove"])
    leather_cuff = make_leather_material(f"XBotGloveCuff_{colorway}", colors["glove_cuff"], roughness=0.72)
    leather_strap = make_leather_material(f"XBotGloveStrap_{colorway}", colors["glove_strap"], roughness=0.75)
    knit = make_knit_material(f"XBotBeanieKnit_{colorway}", colors["beanie"])
    foam = make_foam_leather_material(f"XBotHeadgear_{colorway}", colors["headgear"], roughness=0.74)
    strap_mat = make_leather_material(f"XBotStrap_{colorway}", colors["strap"], roughness=0.78)

    if "gloves" in items:
        envs = measure_hand_envelopes(arm, meshes, scan_frames=scan_frames, step=3)
        handle["hand_envelopes"] = {
            s: {"min": e["min"], "max": e["max"], "n": e.get("n", 0), "bone": e["bone"]}
            for s, e in envs.items()
        }
        for side, env in envs.items():
            bn = env["bone"]
            wr = _wrist_radius(arm, meshes, bn, side)
            gname = f"XBotGlove_{side}_{colorway}"
            glove, ginfo = build_glove_mesh(gname, col, env, wrist_radius=wr)
            _assign_mat(glove, leather)
            glove.matrix_world = bone_world_matrix(arm, bn)
            parent_to_bone(glove, arm, bn)
            handle["objects"].append(glove.name)
            handle["bones"][glove.name] = bn
            handle.setdefault("glove_info", {})[side] = {**ginfo, "wrist_r": wr}
        handle["notes"].append(
            "Gloves are rigid to the Hand bone (they do not squash with finger curl). "
            "Sized to a union envelope of the clip plus a stored Idle+Jab envelope "
            "so fingertips should stay inside across the jab."
        )

    head_bone = find_bone(arm, "mixamorig:Head", "Head", "head")
    handle["head_bone"] = head_bone
    head_pts = _head_points(meshes, arm, head_bone) if head_bone else []

    if "beanie" in items:
        if not head_bone or not head_pts:
            handle["notes"].append("beanie skipped: no Head bone / verts")
        else:
            cap, binfo = build_beanie(f"XBotBeanie_{colorway}", col, head_pts)
            _assign_mat(cap, knit)
            cap.matrix_world = bone_world_matrix(arm, head_bone)
            parent_to_bone(cap, arm, head_bone)
            handle["objects"].append(cap.name)
            handle["bones"][cap.name] = head_bone
            handle["beanie_info"] = binfo
            handle["notes"].append(
                f"Beanie hugged to the skull ({binfo['moved']} verts pushed out to "
                f"{BEANIE_CLEARANCE_CM:.2f} cm clearance)."
            )

    if "headgear" in items:
        if not head_bone or not head_pts:
            handle["notes"].append("headgear skipped: no Head bone / verts")
        else:
            parts, hinfo = build_headgear(f"XBotHeadgear_{colorway}", col, head_pts)
            for hg in parts:
                _assign_mat(hg, foam)
                hg.matrix_world = bone_world_matrix(arm, head_bone)
                parent_to_bone(hg, arm, head_bone)
                handle["objects"].append(hg.name)
                handle["bones"][hg.name] = head_bone
            handle["headgear_info"] = hinfo
            handle["notes"].append(hinfo["note"])

    # Intersection diagnostic at the current frame (gloves).
    if "gloves" in items:
        handle["containment"] = _containment_report(arm, meshes, handle)
        handle["masked"] = mask_inside_gloves(meshes, handle)
        if handle["masked"]:
            handle["notes"].append(
                "Hand / wrist-joint verts inside the gloves are masked so the "
                "clay fist cannot z-fight through the leather."
            )

    bpy.context.view_layer.update()
    return handle


def _containment_report(arm, meshes, handle, thresh=0.25):
    """Count hand verts that sit outside the glove solid (except those in the
    cuff opening half-space, which are allowed to leave through the wrist)."""
    report = {}
    deps = bpy.context.evaluated_depsgraph_get()
    for side, info in (handle.get("glove_info") or {}).items():
        gname = next((n for n in handle["objects"] if n.startswith(f"XBotGlove_{side}_")), None)
        if not gname or gname not in bpy.data.objects:
            continue
        glove = bpy.data.objects[gname]
        ev = glove.evaluated_get(deps)
        try:
            gme = ev.to_mesh()
        except Exception:
            continue
        # Build a simple inside test: glove object-space AABB of the SOLID
        # minus the cavity Y-range at the opening is too coarse. Use nearest
        # glove vert + face normal? Expensive. Instead: count hand verts whose
        # nearest glove surface point has the hand on the outward side AND
        # whose bone-local Y is in front of the cuff lip (i.e. should be inside).
        try:
            kd = kdtree.KDTree(len(gme.vertices))
            gmw = ev.matrix_world
            gverts = [gmw @ v.co for v in gme.vertices]
            for i, p in enumerate(gverts):
                kd.insert(p, i)
            kd.balance()
        finally:
            ev.to_mesh_clear()
        bn = handle["bones"][gname]
        inv = bone_world_matrix(arm, bn).inverted()
        cuff_y = info["env"]["min"][1] + 0.6
        outside = 0
        total = 0
        for wp in iter_weighted_world_verts(meshes, _hand_pred(side), thresh=thresh):
            loc = inv @ wp
            if loc.y < cuff_y:
                continue  # leaving through the cuff is expected
            total += 1
            _co, idx, dist = kd.find(wp)
            # dist is in world metres. A vert clearly outside the 0.01-scaled
            # mesh: more than ~3 mm from the surface is only a poke if it is
            # also outside the padded AABB (kd finds surface, not inside/out).
            # Use bone-local padded AABB as the containment volume.
            env = info["env"]
            inside_aabb = (
                env["min"][0] - 0.15 <= loc.x <= env["max"][0] + 0.15
                and env["min"][1] - 0.15 <= loc.y <= env["max"][1] + 0.15
                and env["min"][2] - 0.15 <= loc.z <= env["max"][2] + 0.15
            )
            if not inside_aabb:
                outside += 1
        report[side] = {"tested": total, "outside_padded_aabb": outside}
    return report


def clear_outfit(handle, meshes=None):
    """Remove objects created by apply_outfit and restore any glove masks."""
    if not handle:
        return
    if handle.get("masked"):
        names = list(handle["masked"])
        if meshes:
            names = list({*names, *[o.name for o in meshes]})
        for name in names:
            obj = bpy.data.objects.get(name)
            if not obj:
                continue
            if "XBotGloveMask" in obj.modifiers:
                obj.modifiers.remove(obj.modifiers["XBotGloveMask"])
            vg = obj.vertex_groups.get("xbot_glove_hide")
            if vg:
                obj.vertex_groups.remove(vg)
    for name in list(handle.get("objects") or []):
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
    colname = handle.get("collection")
    col = bpy.data.collections.get(colname) if colname else None
    if col and len(col.objects) == 0 and colname not in ("Collection", "Master Collection"):
        bpy.data.collections.remove(col)
    handle["objects"] = []
    handle["masked"] = {}
