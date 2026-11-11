#!/usr/bin/env python3
"""Procedural boxing outfit kit for Mixamo X Bot.

All meshes and materials are generated in Blender — no third-party assets,
so the kit is commercially free (public-domain geometry we authored).

Import from any render script (do not require eval_xbot_boxing.py):

    from xbot_outfit import apply_outfit, COLORWAYS, clear_outfit
    handle = apply_outfit(arm, meshes, items=("gloves", "beanie", "trunks", "tank"), colorway="A", version="v4")

Items:
  gloves    — rigid-parented to mixamorig:LeftHand / RightHand
  beanie    — knit cap, rigid-parented to mixamorig:Head
  headgear  — open-face amateur pads + chin strap, parented to Head
              (kept for v1 / explicit requests; v2 outfit A drops it)

`version` selects the mesh: "v4" (default) is a glossy black 10–12 oz
glove matched to a CC0 reference silhouette, a charcoal rib-knit beanie,
charcoal satin trunks and a fitted charcoal tank; "v3" is the crimson
0.8×-head kit; "v2" the earlier compact kit; "v1" the long mitten.

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

# v2 10–12 oz amateur glove, centimetres, hand-bone local. Tuned against
# side/top photos of a 10 oz hook-and-loop glove (compact rounded fist,
# distinct cuff, attached thumb). Front face ≈ 65% of a typical X Bot
# head width (~17.6 cm). Body length ≈ 1.55× a ~10 cm knuckle width.
GLOVE_V2 = {
    "front_width": 11.5,
    "front_thick": 10.2,
    "body_len": 15.6,
    "body_y0": -0.8,       # cuff junction; slightly overlaps the wrist
    "cuff_len": 9.0,
    "cuff_r_near": 4.55,   # at the fist, swallows the ~3.06 cm wrist
    "cuff_r_far": 4.10,    # slight inward taper, still > forearm
    "thumb_len": 6.4,
    "thumb_rx": 2.25,
    "thumb_rz": 2.35,
}

# v3 10–12 oz amateur glove. v2's 11.5 cm fist (~0.65× head) read as a
# toy ball; a real 10–12 oz glove's padded fist is about 0.8× a head.
# Typical X Bot head width is ~17.61 cm → front_width 14.1. Body is a
# longer padded knuckle (not a v1 sausage): length/width ≈ 1.30.
# Cuff stays ~9 cm and is the same crimson as the fist.
HEAD_WIDTH_REF_CM = 17.61
GLOVE_V3 = {
    "front_width": 14.1,
    "front_thick": 12.4,
    "body_len": 18.4,
    "body_y0": -0.8,
    "cuff_len": 9.0,
    "cuff_r_near": 5.20,
    "cuff_r_far": 4.60,
    "thumb_len": 8.4,
    "thumb_rx": 3.05,
    "thumb_rz": 3.45,
}

# v4: silhouette matched to the CC0 Wikimedia pair (Petey21, 2014) —
# one plump bulbous body that tapers smoothly into a same-colour cuff,
# thumb as a padded lobe along the side (not a separate knob). Fist
# front stays ~0.8× head width. Authored in hand-bone local centimetres.
GLOVE_V4 = {
    "front_width": 14.1,
    "front_thick": 12.8,
    "body_len": 16.2,
    "cuff_len": 8.0,
    "cuff_r": 5.15,
    "thumb_len": 9.2,
    "thumb_rx": 2.55,
    "thumb_rz": 3.15,
}

# Lifted glossy black — AgX crushes anything near 0.02 to a silhouette.
GLOVE_V4_BLACK = (0.072, 0.066, 0.060)
TRIM_V4_BLACK = (0.040, 0.037, 0.034)

BEANIE_CLEARANCE_CM = 0.55
BEANIE_V2_CLEARANCE_CM = 0.65
HEADGEAR_CLEARANCE_CM = 0.70
BROW_Y_CM = 6.0  # cuff line in head-bone local cm (mid-head, watch-cap)
BROW_Y_V2_CM = 6.2  # top of the folded cuff; hem sits lower, just above the brow


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


def make_leather_material(name, color, roughness=0.64, wrinkles=False):
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
    height_out = ramp.outputs["Color"]
    if wrinkles:
        # Object-space creases, stronger at the cuff (low Y) and thumb (high Z).
        texcoord = nt.nodes.new("ShaderNodeTexCoord")
        texcoord.location = (-720, -280)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        sep.location = (-540, -320)
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.location = (-540, -140)
        wave.wave_type = "BANDS"
        wave.bands_direction = "Y"
        wave.wave_profile = "SIN"
        wave.inputs["Scale"].default_value = 6.5
        wave.inputs["Distortion"].default_value = 2.4
        wave.inputs["Detail"].default_value = 4.0
        n2 = nt.nodes.new("ShaderNodeTexNoise")
        n2.location = (-540, -480)
        n2.inputs["Scale"].default_value = 9.0
        n2.inputs["Detail"].default_value = 8.0
        n2.inputs["Roughness"].default_value = 0.62
        wr_add = nt.nodes.new("ShaderNodeMath")
        wr_add.operation = "MULTIPLY_ADD"
        wr_add.location = (-300, -300)
        wr_add.inputs[1].default_value = 0.55
        wr_add.inputs[2].default_value = 0.0
        cuff_map = nt.nodes.new("ShaderNodeMapRange")
        cuff_map.location = (-300, -460)
        cuff_map.inputs["From Min"].default_value = 4.0
        cuff_map.inputs["From Max"].default_value = -3.0
        cuff_map.inputs["To Min"].default_value = 0.08
        cuff_map.inputs["To Max"].default_value = 1.0
        thumb_map = nt.nodes.new("ShaderNodeMapRange")
        thumb_map.location = (-300, -640)
        thumb_map.inputs["From Min"].default_value = 2.0
        thumb_map.inputs["From Max"].default_value = 6.5
        thumb_map.inputs["To Min"].default_value = 0.0
        thumb_map.inputs["To Max"].default_value = 0.85
        mask_max = nt.nodes.new("ShaderNodeMath")
        mask_max.operation = "MAXIMUM"
        mask_max.location = (-80, -520)
        wr_mul = nt.nodes.new("ShaderNodeMath")
        wr_mul.operation = "MULTIPLY"
        wr_mul.location = (-80, -300)
        height_add = nt.nodes.new("ShaderNodeMath")
        height_add.operation = "ADD"
        height_add.location = (80, -300)
        nt.links.new(texcoord.outputs["Object"], sep.inputs["Vector"])
        nt.links.new(texcoord.outputs["Object"], wave.inputs["Vector"])
        nt.links.new(texcoord.outputs["Object"], n2.inputs["Vector"])
        nt.links.new(wave.outputs["Fac"], wr_add.inputs[0])
        nt.links.new(n2.outputs["Fac"], wr_add.inputs[2])
        nt.links.new(sep.outputs["Y"], cuff_map.inputs["Value"])
        nt.links.new(sep.outputs["Z"], thumb_map.inputs["Value"])
        nt.links.new(cuff_map.outputs["Result"], mask_max.inputs[0])
        nt.links.new(thumb_map.outputs["Result"], mask_max.inputs[1])
        nt.links.new(wr_add.outputs["Value"], wr_mul.inputs[0])
        nt.links.new(mask_max.outputs["Value"], wr_mul.inputs[1])
        nt.links.new(ramp.outputs["Color"], height_add.inputs[0])
        nt.links.new(wr_mul.outputs["Value"], height_add.inputs[1])
        # reconnect bump to the combined height; stronger on v2 leather
        for link in list(nt.links):
            if link.to_socket == bump.inputs["Height"]:
                nt.links.remove(link)
        nt.links.new(height_add.outputs["Value"], bump.inputs["Height"])
        bump.inputs["Strength"].default_value = 0.16
        bump.inputs["Distance"].default_value = 0.07
    if "Normal" in bsdf.inputs:
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    _set(bsdf, "Base Color", (*color, 1.0))
    _set(bsdf, "Specular IOR Level", 0.32 if not wrinkles else 0.38)
    _set(bsdf, "Specular", 0.32 if not wrinkles else 0.38)
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Coat Weight", 0.04 if not wrinkles else 0.08)
    _set(bsdf, "Coat Roughness", 0.35 if not wrinkles else 0.28)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat



def make_glossy_leather_material(name, color, roughness=0.30):
    """AgX-safe glossy boxing-glove leather with a slight Fresnel rim pickup."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (920, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (560, 0)
    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-620, -40)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-400, -40)
    noise.inputs["Scale"].default_value = 48.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.52
    # Fine leather grain bump
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (280, -200)
    bump.inputs["Strength"].default_value = 0.10
    bump.inputs["Distance"].default_value = 0.035
    # Creases near the thumb (+Z) and cuff (low Y), object space.
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.location = (-400, -280)
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.wave_profile = "SIN"
    wave.inputs["Scale"].default_value = 5.8
    wave.inputs["Distortion"].default_value = 2.6
    wave.inputs["Detail"].default_value = 5.0
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.location = (-400, -500)
    n2.inputs["Scale"].default_value = 11.0
    n2.inputs["Detail"].default_value = 8.0
    crease = nt.nodes.new("ShaderNodeMath")
    crease.operation = "MULTIPLY_ADD"
    crease.location = (-180, -320)
    crease.inputs[1].default_value = 0.45
    crease.inputs[2].default_value = 0.0
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-400, 160)
    thumb_map = nt.nodes.new("ShaderNodeMapRange")
    thumb_map.location = (-180, -520)
    thumb_map.inputs["From Min"].default_value = 3.0
    thumb_map.inputs["From Max"].default_value = 8.0
    thumb_map.inputs["To Min"].default_value = 0.0
    thumb_map.inputs["To Max"].default_value = 0.9
    cuff_map = nt.nodes.new("ShaderNodeMapRange")
    cuff_map.location = (-180, -700)
    cuff_map.inputs["From Min"].default_value = 4.0
    cuff_map.inputs["From Max"].default_value = -4.0
    cuff_map.inputs["To Min"].default_value = 0.05
    cuff_map.inputs["To Max"].default_value = 0.85
    mask = nt.nodes.new("ShaderNodeMath")
    mask.operation = "MAXIMUM"
    mask.location = (20, -560)
    wr = nt.nodes.new("ShaderNodeMath")
    wr.operation = "MULTIPLY"
    wr.location = (200, -400)
    height = nt.nodes.new("ShaderNodeMath")
    height.operation = "ADD"
    height.location = (280, -80)
    # Roughness: glossier with a little grain, never flat black
    rough = nt.nodes.new("ShaderNodeMath")
    rough.operation = "MULTIPLY_ADD"
    rough.location = (280, 80)
    rough.inputs[1].default_value = 0.10
    rough.inputs[2].default_value = float(roughness)
    # Fresnel rim: lift the base a little at glancing angles so the form
    # still reads under AgX in a dark studio.
    layer = nt.nodes.new("ShaderNodeLayerWeight")
    layer.location = (20, 220)
    layer.inputs["Blend"].default_value = 0.42
    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.location = (280, 280)
    mix_col.data_type = "RGBA"
    rim = tuple(min(1.0, c * 1.85 + 0.025) for c in color)
    if "A" in mix_col.inputs:
        mix_col.inputs["A"].default_value = (*color, 1.0)
        mix_col.inputs["B"].default_value = (*rim, 1.0)
    else:
        mix_col.inputs[6].default_value = (*color, 1.0)
        mix_col.inputs[7].default_value = (*rim, 1.0)
    nt.links.new(texcoord.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(texcoord.outputs["Object"], wave.inputs["Vector"])
    nt.links.new(texcoord.outputs["Object"], n2.inputs["Vector"])
    nt.links.new(texcoord.outputs["Object"], sep.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], height.inputs[0])
    nt.links.new(wave.outputs["Fac"], crease.inputs[0])
    nt.links.new(n2.outputs["Fac"], crease.inputs[2])
    nt.links.new(sep.outputs["Z"], thumb_map.inputs["Value"])
    nt.links.new(sep.outputs["Y"], cuff_map.inputs["Value"])
    nt.links.new(thumb_map.outputs["Result"], mask.inputs[0])
    nt.links.new(cuff_map.outputs["Result"], mask.inputs[1])
    nt.links.new(crease.outputs["Value"], wr.inputs[0])
    nt.links.new(mask.outputs["Value"], wr.inputs[1])
    nt.links.new(wr.outputs["Value"], height.inputs[1])
    nt.links.new(height.outputs["Value"], bump.inputs["Height"])
    nt.links.new(noise.outputs["Fac"], rough.inputs[0])
    nt.links.new(layer.outputs["Fresnel"], mix_col.inputs["Factor"] if "Factor" in mix_col.inputs else mix_col.inputs[0])
    def _mix_out(n):
        return n.outputs.get("Result") or n.outputs.get("Color") or n.outputs[0]
    col_out = _mix_out(mix_col)
    if "Base Color" in bsdf.inputs:
        nt.links.new(col_out, bsdf.inputs["Base Color"])
    if "Normal" in bsdf.inputs:
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(rough.outputs["Value"], bsdf.inputs["Roughness"])
    _set(bsdf, "Specular IOR Level", 0.58)
    _set(bsdf, "Specular", 0.58)
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Coat Weight", 0.48)
    _set(bsdf, "Coat Roughness", 0.12)
    _set(bsdf, "Coat IOR", 1.52)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_satin_material(name, color, roughness=0.20):
    """Glossy athletic satin, AgX-safe (lifted blacks, soft sheen)."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (360, 0)
    layer = nt.nodes.new("ShaderNodeLayerWeight")
    layer.location = (40, 180)
    layer.inputs["Blend"].default_value = 0.38
    mix_col = nt.nodes.new("ShaderNodeMix")
    mix_col.location = (200, 180)
    mix_col.data_type = "RGBA"
    rim = tuple(min(1.0, c * 1.65 + 0.02) for c in color)
    if "A" in mix_col.inputs:
        mix_col.inputs["A"].default_value = (*color, 1.0)
        mix_col.inputs["B"].default_value = (*rim, 1.0)
    else:
        mix_col.inputs[6].default_value = (*color, 1.0)
        mix_col.inputs[7].default_value = (*rim, 1.0)
    nt.links.new(layer.outputs["Fresnel"], mix_col.inputs["Factor"] if "Factor" in mix_col.inputs else mix_col.inputs[0])
    def _mix_out(n):
        return n.outputs.get("Result") or n.outputs.get("Color") or n.outputs[0]
    nt.links.new(_mix_out(mix_col), bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", float(roughness))
    _set(bsdf, "Specular IOR Level", 0.52)
    _set(bsdf, "Specular", 0.52)
    _set(bsdf, "Sheen Weight", 0.35)
    _set(bsdf, "Sheen Roughness", 0.22)
    _set(bsdf, "Anisotropic", 0.35)
    _set(bsdf, "Coat Weight", 0.22)
    _set(bsdf, "Coat Roughness", 0.18)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_knit_material(name, color, strength=0.22):
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
    bump.inputs["Strength"].default_value = float(strength)
    bump.inputs["Distance"].default_value = 0.10 if strength > 0.3 else 0.08
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


def _join_objects(target, *others):
    """Join mesh objects into `target` (more reliable than EXACT boolean)."""
    others = [o for o in others if o is not None]
    if not others:
        return target
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    for o in others:
        if o.name not in bpy.context.view_layer.objects:
            # ensure it's in the view layer so join can see it
            try:
                bpy.context.scene.collection.objects.link(o)
            except Exception:
                pass
        o.select_set(True)
    bpy.ops.object.join()
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




def _tapered_tube(name, collection, y_near, y_far, r_near, r_far, segments=28, rings=6):
    """Closed tapered tube along +Y, r_near at y_near and r_far at y_far."""
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    n = max(8, int(segments))
    steps = max(2, int(rings))
    rows = []
    for s in range(steps + 1):
        t = s / steps
        y = y_near * (1.0 - t) + y_far * t
        r = r_near * (1.0 - t) + r_far * t
        row = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            row.append(bm.verts.new((r * math.cos(a), y, r * math.sin(a))))
        rows.append(row)
    bm.verts.ensure_lookup_table()
    for s in range(steps):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((rows[s][i], rows[s][j], rows[s + 1][j], rows[s + 1][i]))
    try:
        bm.faces.new(rows[0])
        bm.faces.new(list(reversed(rows[-1])))
    except Exception:
        pass
    bm.to_mesh(mesh)
    bm.free()
    _shade_smooth(obj)
    return obj


def _shape_glove_palm_v2(obj, y0, y1, rx):
    """Shallow palm cavity and a slight grip-bar ridge on the -X side."""
    mid_y = 0.45 * y0 + 0.55 * y1
    span = max(4.0, (y1 - y0) * 0.38)
    for v in obj.data.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        if x > 0.15 or y < y0 + 2.0 or y > y1 - 2.0:
            continue
        ty = max(0.0, 1.0 - abs(y - mid_y) / span)
        palm = max(0.0, min(1.0, -x / max(0.2, rx)))
        cavity = 0.85 * ty * palm * palm
        bar = math.exp(-((y - mid_y) / 1.15) ** 2) * palm
        v.co.x += cavity - 0.35 * bar
    obj.data.update()


def _assign_glove_mats_v2(obj, leather, cuff_mat, strap_mat, info):
    """Crimson fist, darker cuff, subtle velcro band by bone-local Y."""
    me = obj.data
    me.materials.clear()
    me.materials.append(leather)
    me.materials.append(cuff_mat)
    me.materials.append(strap_mat)
    y_near = float(info.get("cuff_y_near", -0.5))
    y_far = float(info.get("cuff_y_far", -10.0))
    # strap sits on the lower half of the cuff
    strap_lo = y_far + 1.4
    strap_hi = y_far + 5.2
    for poly in me.polygons:
        y = sum((me.vertices[i].co.y for i in poly.vertices), 0.0) / max(1, len(poly.vertices))
        if strap_lo <= y <= strap_hi:
            poly.material_index = 2
        elif y < y_near - 0.15:
            poly.material_index = 1
        else:
            poly.material_index = 0
    me.update()


def build_glove_mesh_v2(name, collection, hand_env, wrist_radius=None):
    """Compact 10–12 oz glove: rounded fist, attached thumb, distinct cuff.

    Built in hand-bone local centimetres, origin at the bone head.
    +Y = knuckles, +Z = thumb side, cuff opening toward -Y.
    """
    wr = float(wrist_radius or 3.35)
    spec = GLOVE_V2
    cuff_r_near = max(spec["cuff_r_near"], wr + 1.25)
    cuff_r_far = max(spec["cuff_r_far"], wr + 0.90)
    fw, ft, bl = spec["front_width"], spec["front_thick"], spec["body_len"]
    y0 = spec["body_y0"]
    y1 = y0 + bl
    body_c = Vector((0.0, 0.5 * (y0 + y1), 0.10))
    rx, ry, rz = fw * 0.50, bl * 0.50, ft * 0.50

    body = _ellipsoid(name + "_body", (rx, ry, rz * 0.96), 3, collection, location=body_c)
    # Shape the fist as one bean: blunt knuckle bulge, taper into the cuff.
    # A second knuckle ellipsoid previously fused into a peanut / snowman tip.
    for v in body.data.vertices:
        t = (v.co.y - body_c.y) / max(0.2, ry)  # -1 cuff pole, +1 knuckle pole
        xz = Vector((v.co.x, 0.0, v.co.z - body_c.z))
        if t > 0.20:
            k = (t - 0.20) / 0.80
            # Widen slightly into a rounded knuckle face, do not grow another lobe.
            scale = 1.0 + 0.12 * k
            v.co.x *= scale
            v.co.z = body_c.z + (v.co.z - body_c.z) * scale
            # Flatten the tip a little so it stays a cap, not a ball.
            if t > 0.72:
                v.co.y = body_c.y + (v.co.y - body_c.y) * (1.0 - 0.18 * (t - 0.72) / 0.28)
        elif t < -0.10:
            k = min(1.0, (-0.10 - t) / 0.90)
            scale = 1.0 - 0.26 * k  # waist into the cuff
            v.co.x *= scale
            v.co.z = body_c.z + (v.co.z - body_c.z) * scale
    body.data.update()

    # Attached thumb lobe along +Z, mid-fist (not at the knuckle tip).
    # Sit it far enough out that voxel remesh keeps a readable side lobe.
    thumb_c = Vector((0.20, body_c.y - 0.15, rz * 0.88 + 1.20))
    thumb = _ellipsoid(
        name + "_thumb",
        (spec["thumb_rx"] * 1.05, spec["thumb_len"] * 0.50, spec["thumb_rz"] * 1.08),
        3, collection, location=(0.0, 0.0, 0.0),
    )
    ang = math.radians(14.0)
    ca, sa = math.cos(ang), math.sin(ang)
    for v in thumb.data.vertices:
        y, z = v.co.y, v.co.z
        v.co.y = y * ca - z * sa
        v.co.z = y * sa + z * ca
        v.co += thumb_c
    thumb.data.update()

    # Cuff overlaps the fist by ~1.6 cm, then runs 9 cm toward the forearm.
    cuff_y_near = y0 + 1.6
    cuff_y_far = y0 - spec["cuff_len"]
    cuff = _tapered_tube(
        name + "_cuff", collection, cuff_y_near, cuff_y_far,
        cuff_r_near, cuff_r_far, segments=28, rings=6,
    )
    # Subtle velcro strap: a slightly proud band on the lower cuff.
    strap_y_near = cuff_y_far + 5.2
    strap_y_far = cuff_y_far + 1.4
    strap_r = 0.5 * (cuff_r_near + cuff_r_far) + 0.22
    strap = _tapered_tube(
        name + "_strap", collection, strap_y_near, strap_y_far,
        strap_r, strap_r * 0.98, segments=28, rings=3,
    )

    _boolean_union(body, thumb, cuff, strap, cleanup=True)
    _voxel_remesh(body, voxel=0.18)
    _smooth_mesh(body, iterations=7, factor=0.32)
    _shape_glove_palm_v2(body, y0, y1, rx)
    _smooth_mesh(body, iterations=2, factor=0.16)
    _shade_smooth(body)
    body.name = name

    env = {
        "min": (-rx - 0.55, cuff_y_far - 0.35, -rz - 0.55),
        "max": (rx + 0.55, y1 + 0.85, thumb_c.z + spec["thumb_rz"] + 0.65),
    }
    return body, {
        "env": env,
        "cavity_r": None,
        "cavity_y": None,
        "version": "v2",
        "mask_plane_y": cuff_y_far + 2.6,
        "cuff_rx": cuff_r_near + 0.45,
        "cuff_rz": cuff_r_near + 0.45,
        "cuff_y_far": cuff_y_far,
        "cuff_y_near": cuff_y_near,
        "body_len": bl,
        "front_width": fw,
        "front_thick": ft,
    }



def _glove_v3_spec(head_width=None):
    """Scale GLOVE_V3 so the fist front is 0.8× the actual head width."""
    hw = float(head_width) if head_width and head_width > 4.0 else HEAD_WIDTH_REF_CM
    fw = 0.80 * hw
    s = fw / GLOVE_V3["front_width"]
    return {
        "front_width": fw,
        "front_thick": GLOVE_V3["front_thick"] * s,
        "body_len": GLOVE_V3["body_len"] * s,
        "body_y0": GLOVE_V3["body_y0"],
        "cuff_len": GLOVE_V3["cuff_len"],  # cuff length stays a real ~9 cm
        "cuff_r_near": GLOVE_V3["cuff_r_near"] * s,
        "cuff_r_far": GLOVE_V3["cuff_r_far"] * s,
        "thumb_len": GLOVE_V3["thumb_len"] * s,
        "thumb_rx": GLOVE_V3["thumb_rx"] * s,
        "thumb_rz": GLOVE_V3["thumb_rz"] * s,
        "head_width": hw,
        "scale": s,
    }


def _assign_glove_mats_v3(obj, leather, strap_mat, info):
    """Same-crimson fist and cuff; only a thin slightly darker hook-and-loop seam."""
    me = obj.data
    me.materials.clear()
    me.materials.append(leather)
    me.materials.append(strap_mat)
    y_far = float(info.get("cuff_y_far", -10.0))
    strap_lo = y_far + 2.15
    strap_hi = y_far + 3.35  # ~1.2 cm band, not a black wrist
    for poly in me.polygons:
        y = sum((me.vertices[i].co.y for i in poly.vertices), 0.0) / max(1, len(poly.vertices))
        poly.material_index = 1 if strap_lo <= y <= strap_hi else 0
    me.update()


def build_glove_mesh_v3(name, collection, hand_env, wrist_radius=None, head_width=None):
    """10–12 oz glove at real scale: fist ~0.8× head, long padded knuckle, visible thumb.

    Built in hand-bone local centimetres, origin at the bone head.
    +Y = knuckles, +Z = thumb side, cuff opening toward -Y.
    The thumb is unioned after the body remesh so the side lobe survives.
    """
    wr = float(wrist_radius or 3.35)
    spec = _glove_v3_spec(head_width)
    fw, ft, bl = spec["front_width"], spec["front_thick"], spec["body_len"]
    # Meet the fist waist so the cuff does not form a stepped cylinder.
    cuff_r_near = max(spec["cuff_r_near"] * 1.12, wr + 1.55, fw * 0.42)
    cuff_r_far = max(spec["cuff_r_far"], wr + 1.00)
    y0 = spec["body_y0"]
    y1 = y0 + bl
    body_c = Vector((0.0, 0.5 * (y0 + y1), 0.12))
    rx, ry, rz = fw * 0.50, bl * 0.50, ft * 0.50

    body = _ellipsoid(name + "_body", (rx, ry, rz * 0.97), 3, collection, location=body_c)
    # Bean: blunt padded knuckle face, slight waist into the cuff. Longer
    # knuckle pad than v2 (more of the front of the fist is padding).
    for v in body.data.vertices:
        t = (v.co.y - body_c.y) / max(0.2, ry)
        if t > 0.12:
            k = (t - 0.12) / 0.88
            # Widen into a rounded, slightly flattened knuckle pad.
            scale = 1.0 + 0.16 * k
            v.co.x *= scale
            v.co.z = body_c.z + (v.co.z - body_c.z) * scale
            if t > 0.62:
                # Flatten the tip into a padded face, not a ball or sausage.
                v.co.y = body_c.y + (v.co.y - body_c.y) * (1.0 - 0.14 * (t - 0.62) / 0.38)
        elif t < -0.08:
            k = min(1.0, (-0.08 - t) / 0.92)
            scale = 1.0 - 0.22 * k
            v.co.x *= scale
            v.co.z = body_c.z + (v.co.z - body_c.z) * scale
    body.data.update()

    cuff_y_near = y0 + 3.2
    cuff_y_far = y0 - spec["cuff_len"]
    cuff = _tapered_tube(
        name + "_cuff", collection, cuff_y_near, cuff_y_far,
        cuff_r_near, cuff_r_far, segments=28, rings=6,
    )
    # Thin same-family seam (geometry only slightly proud), not a dark wrist.
    strap_y_near = cuff_y_far + 3.35
    strap_y_far = cuff_y_far + 2.15
    strap_r = 0.5 * (cuff_r_near + cuff_r_far) + 0.16
    strap = _tapered_tube(
        name + "_strap", collection, strap_y_near, strap_y_far,
        strap_r, strap_r * 0.99, segments=28, rings=2,
    )

    # Elongated attached thumb (not a cherry sphere). Long axis starts mid-fist
    # and runs along the side toward the knuckles, base buried in the body.
    thumb_c = Vector((0.25, body_c.y + 0.10, rz * 0.62 + 1.35))
    thumb = _ellipsoid(
        name + "_thumb",
        (spec["thumb_rx"] * 0.82, spec["thumb_len"] * 0.62, spec["thumb_rz"] * 0.78),
        3, collection, location=(0.0, 0.0, 0.0),
    )
    ang = math.radians(28.0)
    ca, sa = math.cos(ang), math.sin(ang)
    for v in thumb.data.vertices:
        y, z = v.co.y, v.co.z
        v.co.y = y * ca - z * sa
        v.co.z = y * sa + z * ca
        v.co += thumb_c
    thumb.data.update()

    # Join then voxel-remesh: EXACT boolean silently dropped the cuff in v3,
    # leaving a floating fist egg and a bare clay wrist.
    _join_objects(body, cuff, strap, thumb)
    _voxel_remesh(body, voxel=0.14)
    _smooth_mesh(body, iterations=4, factor=0.22)
    _shape_glove_palm_v2(body, y0, y1, rx)
    _smooth_mesh(body, iterations=2, factor=0.14)
    _shade_smooth(body)
    body.name = name

    env = {
        "min": (-rx - 0.65, cuff_y_far - 0.35, -rz - 0.65),
        "max": (rx + 0.65, y1 + 0.95, thumb_c.z + spec["thumb_rz"] + 0.85),
    }
    return body, {
        "env": env,
        "cavity_r": None,
        "cavity_y": None,
        "version": "v3",
        "mask_plane_y": cuff_y_far + 2.6,
        "cuff_rx": cuff_r_near + 0.50,
        "cuff_rz": cuff_r_near + 0.50,
        "cuff_y_far": cuff_y_far,
        "cuff_y_near": cuff_y_near,
        "body_len": bl,
        "front_width": fw,
        "front_thick": ft,
        "head_width": spec["head_width"],
        "fist_to_head": fw / max(1e-6, spec["head_width"]),
    }


def _smoothstep(a, b, x):
    if b == a:
        return 0.0 if x < a else 1.0
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def _glove_v4_spec(head_width=None):
    hw = float(head_width) if head_width and head_width > 4.0 else HEAD_WIDTH_REF_CM
    fw = 0.80 * hw
    s = fw / GLOVE_V4["front_width"]
    return {
        "front_width": fw,
        "front_thick": GLOVE_V4["front_thick"] * s,
        "body_len": GLOVE_V4["body_len"] * s,
        "cuff_len": GLOVE_V4["cuff_len"],
        "cuff_r": GLOVE_V4["cuff_r"] * s,
        "thumb_len": GLOVE_V4["thumb_len"] * s,
        "thumb_rx": GLOVE_V4["thumb_rx"] * s,
        "thumb_rz": GLOVE_V4["thumb_rz"] * s,
        "head_width": hw,
        "scale": s,
    }


def _v4_rxz(t, rx, rz, cuff_r):
    """Radius profile of the CC0 reference: plump bulb, smooth cuff taper."""
    widen = _smoothstep(0.30, 0.50, t)
    # Slight proud hook-and-loop band on the lower cuff, same colour.
    strap = 0.12 * math.exp(-((t - 0.16) / 0.038) ** 2)
    sx = cuff_r * (1.0 + strap) + (rx - cuff_r) * widen
    sz = cuff_r * 0.97 * (1.0 + strap) + (rz - cuff_r * 0.97) * widen
    if t > 0.80:
        s = (t - 0.80) / 0.20
        cap = math.sqrt(max(0.0, 1.0 - s * s * 0.96))
        sx *= cap
        sz *= cap
    return sx, sz


def _add_stitch_grooves_v4(obj, y_open, y_tip, rx, rz, cuff_r, thumb_c, thumb_len):
    """Shallow stitch lines: perimeter, thumb crotch, cuff band."""
    me = obj.data
    # Precompute a simple outward normal approx from centroid.
    for v in me.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        # 1) dorsal / ventral perimeter seams (near the X silhouette)
        t = (y - y_open) / max(1.0, y_tip - y_open)
        sx, sz = _v4_rxz(max(0.0, min(1.0, t)), rx, rz, cuff_r)
        if 0.08 < t < 0.96 and sx > 0.4:
            edge = abs(abs(x) / sx - 0.92)
            if edge < 0.10:
                w = 1.0 - edge / 0.10
                # pull slightly inward
                v.co.x *= 1.0 - 0.035 * w * w
                v.co.z *= 1.0 - 0.012 * w * w
        # 2) thumb-attachment oval around the lobe base
        tc = thumb_c
        d = math.sqrt((x - tc.x) ** 2 * 1.4 + (y - tc.y) ** 2 * 0.45 + (z - tc.z * 0.35) ** 2)
        # ring around the crotch
        ring = abs(d - (thumb_len * 0.22))
        if ring < 0.45 and z > 0.5:
            w = 1.0 - ring / 0.45
            n = Vector((x, y, z)) - tc
            if n.length > 1e-6:
                v.co -= n.normalized() * (0.10 * w * w)
        # 3) cuff band seam (two thin rings)
        for cy in (y_open + 1.6, y_open + 3.2):
            if abs(y - cy) < 0.18:
                w = 1.0 - abs(y - cy) / 0.18
                radial = Vector((x, 0.0, z))
                if radial.length > 1e-6:
                    v.co -= radial.normalized() * (0.07 * w)
    me.update()


def build_glove_mesh_v4(name, collection, hand_env, wrist_radius=None, head_width=None):
    """Glossy 10–12 oz glove: one bulbous body, side-lobe thumb, smooth cuff.

    Hand-bone local centimetres. +Y knuckles, +Z thumb, cuff toward -Y.
    The body is a single lathed surface (no joined cuff tube) so the waist
    cannot step. The thumb is a heavily overlapping elongated lobe that
    remesh blends into the side — not a cherry knob.
    """
    wr = float(wrist_radius or 3.35)
    spec = _glove_v4_spec(head_width)
    fw, ft, bl = spec["front_width"], spec["front_thick"], spec["body_len"]
    cuff_len = spec["cuff_len"]
    cuff_r = max(spec["cuff_r"], wr + 1.45)
    rx, rz = fw * 0.50, ft * 0.50
    # Wrist at y=0; cuff overlaps the forearm, bulb covers the fist.
    y_open = -cuff_len
    y_tip = bl
    y_span = y_tip - y_open

    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name + "_body", mesh, collection)
    bm = bmesh.new()
    n_seg = 36
    n_rings = 32
    rows = []
    for i in range(n_rings + 1):
        t = i / n_rings
        # t=0 is a slightly inset opening cap; t≈0.06 is the cuff lip.
        if t < 0.06:
            y = y_open + 1.15 * (1.0 - t / 0.06)
            lip_s = t / 0.06
            sx, sz = cuff_r * lip_s * 0.92, cuff_r * 0.97 * lip_s * 0.92
        else:
            tt = (t - 0.06) / 0.94
            y = y_open + y_span * tt
            sx, sz = _v4_rxz(tt, rx, rz, cuff_r)
        row = []
        for j in range(n_seg):
            a = 2.0 * math.pi * j / n_seg
            # Palm (−X) is slightly flatter on the bulb, like packed leather.
            pal = 0.90 if (math.cos(a) < 0.0 and t > 0.35) else 1.0
            x = sx * pal * math.cos(a)
            z = sz * math.sin(a) + 0.08  # slight +Z bias, thumb side
            row.append(bm.verts.new((x, y, z)))
        rows.append(row)
    bm.verts.ensure_lookup_table()
    for i in range(n_rings):
        for j in range(n_seg):
            k = (j + 1) % n_seg
            bm.faces.new((rows[i][j], rows[i][k], rows[i + 1][k], rows[i + 1][j]))
    # Caps
    try:
        bm.faces.new(list(reversed(rows[0])))
        bm.faces.new(rows[-1])
    except Exception:
        pass
    bm.to_mesh(mesh)
    bm.free()

    # Elongated thumb lobe along +Z, mid-body, base buried in the bulb.
    body_mid_y = 0.38 * y_open + 0.62 * y_tip  # toward the bulb, not the cuff
    thumb_c = Vector((0.20, body_mid_y - 0.4, rz * 0.42 + 1.15))
    thumb = _ellipsoid(
        name + "_thumb",
        (spec["thumb_rx"], spec["thumb_len"] * 0.55, spec["thumb_rz"]),
        3, collection, location=(0.0, 0.0, 0.0),
    )
    ang = math.radians(32.0)
    ca, sa = math.cos(ang), math.sin(ang)
    for v in thumb.data.vertices:
        y, z = v.co.y, v.co.z
        v.co.y = y * ca - z * sa
        v.co.z = y * sa + z * ca
        v.co += thumb_c
    thumb.data.update()

    _join_objects(obj, thumb)
    _voxel_remesh(obj, voxel=0.13)
    _smooth_mesh(obj, iterations=5, factor=0.24)
    _add_stitch_grooves_v4(obj, y_open, y_tip, rx, rz, cuff_r, thumb_c, spec["thumb_len"])
    _smooth_mesh(obj, iterations=1, factor=0.10)
    _shade_smooth(obj)
    obj.name = name

    env = {
        "min": (-rx - 0.8, y_open - 0.4, -rz - 0.8),
        "max": (rx + 0.8, y_tip + 0.6, thumb_c.z + spec["thumb_rz"] + 1.0),
    }
    return obj, {
        "env": env,
        "cavity_r": None,
        "cavity_y": None,
        "version": "v4",
        "mask_plane_y": y_open + 2.4,
        "cuff_rx": cuff_r + 0.55,
        "cuff_rz": cuff_r + 0.55,
        "cuff_y_far": y_open,
        "cuff_y_near": y_open + cuff_len * 0.85,
        "body_len": bl,
        "front_width": fw,
        "front_thick": ft,
        "head_width": spec["head_width"],
        "fist_to_head": fw / max(1e-6, spec["head_width"]),
    }


def build_glove_mesh(name, collection, hand_env, wrist_radius=None, version="v4",
                     head_width=None):
    """Solid mitten in hand-bone local centimetres, origin at the bone head.

    +Y = knuckles, +Z = thumb side, cuff opening toward -Y. The body ellipsoid
    is centred on the full cuff-to-knuckle span so the wrist ring stays
    attached (a knuckle-biased body previously left a detached band mid-forearm).
    Pass version="v4" (default) for the glossy black CC0-matched glove,
    "v3" for the crimson 0.8×-head kit, "v2" compact, "v1" long mitten.
    """
    if version == "v4":
        return build_glove_mesh_v4(name, collection, hand_env, wrist_radius=wrist_radius,
                                   head_width=head_width)
    if version == "v3":
        return build_glove_mesh_v3(name, collection, hand_env, wrist_radius=wrist_radius,
                                   head_width=head_width)
    if version == "v2":
        return build_glove_mesh_v2(name, collection, hand_env, wrist_radius=wrist_radius)
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


def _push_off_surface(obj, pts, clearance, only_inside=True, y_max=None, y_min=None):
    """Move verts that have sunk *into* the skull back out to `clearance`.

    Only the inside case is corrected. Outer pad / cuff volume is left alone
    — a global push flattened the headgear into a skull-cap.
    `y_max` / `y_min` limit the correction in bone-local Y (so a beanie crown
    is not crumpled, and a locked hem is not pushed into a sawtooth).
    """
    if not pts or not obj.data.vertices:
        return 0
    kd = _head_kdtree(pts)
    centre = _head_origin(pts)
    moved = 0
    for v in obj.data.vertices:
        if y_max is not None and v.co.y > y_max:
            continue
        if y_min is not None and v.co.y < y_min:
            continue
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


def _lock_beanie_hem(obj, cx, cz, hem_y, band=0.22):
    """Force the cuff rim to planar circle(s) so ribs/push cannot zigzag it.

    After Solidify there is an inner and outer rim; each is locked to its
    own mean radius so the cuff thickness is preserved.
    """
    items = []
    for v in obj.data.vertices:
        if v.co.y > hem_y + band:
            continue
        r = math.hypot(v.co.x - cx, v.co.z - cz)
        if r > 1e-6:
            items.append((v, r))
    if not items:
        return
    mid = sorted(r for _, r in items)[len(items) // 2]
    groups = [[], []]
    for v, r in items:
        groups[0 if r >= mid else 1].append((v, r))
    for group in groups:
        if not group:
            continue
        mean_r = sum(r for _, r in group) / len(group)
        for v, _ in group:
            xz = Vector((v.co.x - cx, 0.0, v.co.z - cz))
            if xz.length < 1e-6:
                continue
            off = xz.normalized() * mean_r
            v.co.x = cx + off.x
            v.co.z = cz + off.z
            v.co.y = hem_y
    obj.data.update()


def _apply_beanie_ribs(obj, cx, cz, fold_y, crown_y, n_ribs=30, cuff_amp=0.42,
                       crown_amp=0.20, hem_y=None, crown_to_top=False):
    """Vertical knit ridges in real geometry, stronger on the folded cuff.

    Amplitude fades to zero at the hem so the rim stays a clean fold, not
    a sawtooth of displaced boundary verts. With crown_to_top=True the ribs
    keep a readable amplitude all the way to the crown (v3).
    """
    for v in obj.data.vertices:
        xz = Vector((v.co.x - cx, 0.0, v.co.z - cz))
        if xz.length < 1e-6:
            continue
        theta = math.atan2(v.co.z - cz, v.co.x - cx)
        rib = 0.5 + 0.5 * math.cos(n_ribs * theta)
        rib = rib ** 0.85
        if v.co.y <= fold_y + 0.35:
            amp = cuff_amp
            if hem_y is not None:
                dist = v.co.y - hem_y
                if dist < 0.70:
                    amp *= max(0.0, dist / 0.70)
        else:
            span = max(1.0, crown_y - fold_y)
            t = min(1.0, max(0.0, (v.co.y - fold_y) / span))
            amp = cuff_amp * (1.0 - t) + crown_amp * t
            if crown_to_top:
                # Keep a visible ridge at the crown; only ease the last 12%
                # so the pole does not burst into a star.
                if t > 0.88:
                    amp *= 1.0 - 0.40 * (t - 0.88) / 0.12
            else:
                if t > 0.55:
                    amp *= 1.0 - (t - 0.55) / 0.45 * 0.85
                if t > 0.82:
                    amp = 0.0
        # Only push outward so rib valleys cannot open gaps to the skull.
        v.co += xz.normalized() * (amp * max(0.0, rib - 0.30))
    if hem_y is not None:
        _lock_beanie_hem(obj, cx, cz, hem_y)
    obj.data.update()


def build_beanie_v2(name, collection, head_pts, brow_y=BROW_Y_V2_CM,
                    clearance=BEANIE_V2_CLEARANCE_CM):
    """Watch-cap with a puffy fold-up cuff, geometric rib knit, slight slouch."""
    fold_y = brow_y
    cuff_h = 3.7
    hem_y = fold_y - cuff_h
    upper = [p for p in head_pts if p.y >= hem_y - 1.5] or list(head_pts)
    xs = [p.x for p in upper]
    ys = [p.y for p in upper]
    zs = [p.z for p in upper]
    cx = 0.0
    cy = (min(ys) + max(ys)) * 0.50
    # Same ellipsoid fit as v1, which covered the skull; v2 adds cuff/ribs/slouch
    # on top. Tight front-radius clamps previously clipped the beanie off the face.
    cz = (min(zs) * 0.55 + max(zs) * 0.45)
    rx = max(abs(min(xs)), abs(max(xs))) + clearance + 0.22
    ry = (max(ys) - cy) + clearance + 0.45
    rz_front = (max(zs) - cz) * 0.95 + clearance
    rz_back = (cz - min(zs)) + clearance + 0.25
    crown_y = cy + ry

    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=5, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co.x, v.co.y, v.co.z
        zscale = rz_front if z >= 0.0 else rz_back
        v.co = Vector((cx + x * rx, cy + y * ry, cz + z * zscale))
        # Slight slouch: crown drops back and down, more on the rear half.
        # Keep it gentle — a strong squash previously dented the side wall.
        t = max(0.0, (v.co.y - cy) / max(0.2, ry))
        back = max(0.0, -(v.co.z - cz) / max(0.2, rz_back))
        slouch = (t ** 1.35) * (0.35 + 0.65 * back)
        v.co.z -= 1.15 * slouch
        v.co.y -= 0.45 * slouch

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bmesh.ops.bisect_plane(
        bm,
        geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
        dist=0.0001,
        plane_co=Vector((0.0, fold_y, 0.0)),
        plane_no=Vector((0.0, 1.0, 0.0)),
        clear_inner=True,
    )
    # Fold-up cuff: flare out, drop to mid, then a slightly in-turned hem.
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        mid_y = fold_y - cuff_h * 0.52
        res = bmesh.ops.extrude_edge_only(bm, edges=boundary)
        verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
        for v in verts:
            radial = Vector((v.co.x - cx, 0.0, v.co.z - cz))
            if radial.length > 1e-6:
                radial.normalize()
            v.co += radial * 0.32
            v.co.y = mid_y
        bm.edges.ensure_lookup_table()
        boundary2 = [e for e in bm.edges if e.is_boundary]
        res2 = bmesh.ops.extrude_edge_only(bm, edges=boundary2)
        verts2 = [g for g in res2["geom"] if isinstance(g, bmesh.types.BMVert)]
        for v in verts2:
            radial = Vector((v.co.x - cx, 0.0, v.co.z - cz))
            if radial.length > 1e-6:
                radial.normalize()
            v.co += radial * (-0.04)  # slight in-turn at the hem
            v.co.y = hem_y
        try:
            bm.faces.new(verts2)
        except Exception:
            pass

    # Puffy doubled-fabric profile on the cuff band.
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        if v.co.y > fold_y + 0.25:
            continue
        t = max(0.0, min(1.0, (fold_y - v.co.y) / max(0.2, cuff_h)))
        puff = math.sin(math.pi * t) * 0.32
        xz = Vector((v.co.x - cx, 0.0, v.co.z - cz))
        if xz.length < 1e-6:
            continue
        off = xz.normalized() * puff
        v.co.x += off.x
        v.co.z += off.z

    # Keep the cuff outside the forehead.
    def _head_r(y, direc_xz):
        d = direc_xz.normalized()
        best = 0.0
        for p in head_pts:
            if abs(p.y - y) > 3.0:
                continue
            xz = Vector((p.x - cx, 0.0, p.z - cz))
            if xz.length < 1e-8:
                continue
            if xz.normalized().dot(d) < 0.5:
                continue
            best = max(best, xz.dot(d))
        return best

    for v in bm.verts:
        if v.co.y > fold_y + 0.8:
            continue
        xz = Vector((v.co.x - cx, 0.0, v.co.z - cz))
        if xz.length < 1e-8:
            continue
        # Only lift a sunken cuff; do not keep expanding the crown.
        if v.co.y > fold_y + 0.2:
            continue
        need = _head_r(v.co.y, xz) + clearance + 0.30
        if xz.length < need:
            off = xz.normalized() * need
            v.co.x = cx + off.x
            v.co.z = cz + off.z

    # Ribs in the raw shell, then solidify so both surfaces keep the knit.
    bm.to_mesh(mesh)
    bm.free()
    _apply_beanie_ribs(obj, cx, cz, fold_y, crown_y,
                       n_ribs=32, cuff_amp=0.40, crown_amp=0.10, hem_y=hem_y)
    sol = obj.modifiers.new("Thick", "SOLIDIFY")
    sol.thickness = 0.55
    sol.offset = 1.0
    _apply_modifier(obj, sol)
    # Push only the cuff / lower cap off the skull. Pushing the crown made
    # hexagonal dents where the icosphere met the head mesh.
    moved = _push_off_surface(obj, head_pts, clearance, only_inside=True,
                             y_max=fold_y + 1.0, y_min=hem_y + 0.55)
    # Re-lock the hem after solidify/push — a forehead push previously
    # zigzagged the rim into sawteeth.
    _lock_beanie_hem(obj, cx, cz, hem_y, band=0.35)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    crown_verts = [v for v in bm.verts if v.co.y > fold_y + 1.0]
    if crown_verts:
        for _ in range(2):
            bmesh.ops.smooth_vert(bm, verts=crown_verts, factor=0.14,
                                  use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    _shade_smooth(obj)
    return obj, {
        "moved": moved,
        "brow_y": brow_y,
        "hem_y": hem_y,
        "rx": rx,
        "ry": ry,
        "version": "v2",
        "slouch": True,
        "ribs": 32,
    }



def build_beanie_v3(name, collection, head_pts, brow_y=BROW_Y_V2_CM,
                    clearance=BEANIE_V2_CLEARANCE_CM):
    """Watch-cap with fold-up cuff, rib knit continuing to the crown, slight slouch/fold.

    Same coverage / hem-lock lessons as v2 (no visor, no face-clip clamp). The
    crown is no longer smoothed flat — ribs stay readable at a distance.
    """
    fold_y = brow_y
    cuff_h = 3.7
    hem_y = fold_y - cuff_h
    upper = [p for p in head_pts if p.y >= hem_y - 1.5] or list(head_pts)
    xs = [p.x for p in upper]
    ys = [p.y for p in upper]
    zs = [p.z for p in upper]
    cx = 0.0
    cy = (min(ys) + max(ys)) * 0.50
    cz = (min(zs) * 0.55 + max(zs) * 0.45)
    rx = max(abs(min(xs)), abs(max(xs))) + clearance + 0.22
    ry = (max(ys) - cy) + clearance + 0.45
    rz_front = (max(zs) - cz) * 0.95 + clearance
    rz_back = (cz - min(zs)) + clearance + 0.25
    crown_y = cy + ry

    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = _new_mesh_object(name, mesh, collection)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=5, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co.x, v.co.y, v.co.z
        zscale = rz_front if z >= 0.0 else rz_back
        v.co = Vector((cx + x * rx, cy + y * ry, cz + z * zscale))
        t = max(0.0, (v.co.y - cy) / max(0.2, ry))
        back = max(0.0, -(v.co.z - cz) / max(0.2, rz_back))
        # Slightly stronger slouch than v2, plus a soft back-crown fold.
        slouch = (t ** 1.25) * (0.40 + 0.75 * back)
        v.co.z -= 1.50 * slouch
        v.co.y -= 0.70 * slouch
        crease_y = crown_y - 1.7
        band = abs(v.co.y - crease_y)
        if band < 1.35 and back > 0.18:
            k = (1.0 - band / 1.35) * back
            v.co.y -= 0.50 * k
            v.co.z -= 0.38 * k

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bmesh.ops.bisect_plane(
        bm,
        geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
        dist=0.0001,
        plane_co=Vector((0.0, fold_y, 0.0)),
        plane_no=Vector((0.0, 1.0, 0.0)),
        clear_inner=True,
    )
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        mid_y = fold_y - cuff_h * 0.52
        res = bmesh.ops.extrude_edge_only(bm, edges=boundary)
        verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
        for v in verts:
            radial = Vector((v.co.x - cx, 0.0, v.co.z - cz))
            if radial.length > 1e-6:
                radial.normalize()
            v.co += radial * 0.32
            v.co.y = mid_y
        bm.edges.ensure_lookup_table()
        boundary2 = [e for e in bm.edges if e.is_boundary]
        res2 = bmesh.ops.extrude_edge_only(bm, edges=boundary2)
        verts2 = [g for g in res2["geom"] if isinstance(g, bmesh.types.BMVert)]
        for v in verts2:
            radial = Vector((v.co.x - cx, 0.0, v.co.z - cz))
            if radial.length > 1e-6:
                radial.normalize()
            v.co += radial * (-0.04)
            v.co.y = hem_y
        try:
            bm.faces.new(verts2)
        except Exception:
            pass

    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        if v.co.y > fold_y + 0.25:
            continue
        t = max(0.0, min(1.0, (fold_y - v.co.y) / max(0.2, cuff_h)))
        puff = math.sin(math.pi * t) * 0.32
        xz = Vector((v.co.x - cx, 0.0, v.co.z - cz))
        if xz.length < 1e-6:
            continue
        off = xz.normalized() * puff
        v.co.x += off.x
        v.co.z += off.z

    def _head_r(y, direc_xz):
        d = direc_xz.normalized()
        best = 0.0
        for p in head_pts:
            if abs(p.y - y) > 3.0:
                continue
            xz = Vector((p.x - cx, 0.0, p.z - cz))
            if xz.length < 1e-8:
                continue
            if xz.normalized().dot(d) < 0.5:
                continue
            best = max(best, xz.dot(d))
        return best

    for v in bm.verts:
        if v.co.y > fold_y + 0.2:
            continue
        xz = Vector((v.co.x - cx, 0.0, v.co.z - cz))
        if xz.length < 1e-8:
            continue
        need = _head_r(v.co.y, xz) + clearance + 0.30
        if xz.length < need:
            off = xz.normalized() * need
            v.co.x = cx + off.x
            v.co.z = cz + off.z

    bm.to_mesh(mesh)
    bm.free()
    # Ribs after the shell exists; stronger crown, continuing to the top.
    _apply_beanie_ribs(obj, cx, cz, fold_y, crown_y,
                       n_ribs=32, cuff_amp=0.42, crown_amp=0.26, hem_y=hem_y,
                       crown_to_top=True)
    sol = obj.modifiers.new("Thick", "SOLIDIFY")
    sol.thickness = 0.55
    sol.offset = 1.0
    _apply_modifier(obj, sol)
    moved = _push_off_surface(obj, head_pts, clearance, only_inside=True,
                             y_max=fold_y + 1.0, y_min=hem_y + 0.55)
    _lock_beanie_hem(obj, cx, cz, hem_y, band=0.35)
    # Re-apply ribs after solidify/push so the crown knit is not lost. Do
    # not smooth the crown (v2's post-smooth flattened it into a helmet).
    _apply_beanie_ribs(obj, cx, cz, fold_y, crown_y,
                       n_ribs=32, cuff_amp=0.0, crown_amp=0.18, hem_y=hem_y,
                       crown_to_top=True)
    _lock_beanie_hem(obj, cx, cz, hem_y, band=0.35)
    obj.data.update()
    _shade_smooth(obj)
    return obj, {
        "moved": moved,
        "brow_y": brow_y,
        "hem_y": hem_y,
        "rx": rx,
        "ry": ry,
        "version": "v3",
        "slouch": True,
        "ribs": 32,
        "crown_to_top": True,
    }


def build_beanie(name, collection, head_pts, brow_y=None, clearance=None, version="v4"):
    """Fitted watch-cap with a clean folded cuff, in head-bone local cm."""
    if version in ("v3", "v4"):
        return build_beanie_v3(
            name, collection, head_pts,
            brow_y=BROW_Y_V2_CM if brow_y is None else brow_y,
            clearance=BEANIE_V2_CLEARANCE_CM if clearance is None else clearance,
        )
    if version == "v2":
        return build_beanie_v2(
            name, collection, head_pts,
            brow_y=BROW_Y_V2_CM if brow_y is None else brow_y,
            clearance=BEANIE_V2_CLEARANCE_CM if clearance is None else clearance,
        )
    if brow_y is None:
        brow_y = BROW_Y_CM
    if clearance is None:
        clearance = BEANIE_CLEARANCE_CM
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
        # v2 stores an explicit plane well inside the 9 cm cuff.
        plane_y = info.get("mask_plane_y", env["min"][1] + 1.4)
        rx = info.get("cuff_rx", max(abs(env["min"][0]), abs(env["max"][0])) + 1.5)
        rz = info.get("cuff_rz", max(abs(env["min"][2]), abs(env["max"][2])) + 1.5)
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


def _primary_body_meshes(meshes):
    """Skip joint helper meshes; prefer the dense body surface."""
    out = []
    for o in meshes:
        if o.type != "MESH" or not o.data.vertices:
            continue
        if "joint" in o.name.lower():
            continue
        out.append(o)
    out.sort(key=lambda o: len(o.data.vertices), reverse=True)
    return out


def _rest_context(arm):
    prev = arm.data.pose_position
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    return prev


def _restore_pose(arm, prev):
    arm.data.pose_position = prev
    bpy.context.view_layer.update()


def _world_body_samples(meshes, arm):
    """World-space rest samples: co, normal, max weight name (cleaned)."""
    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in _primary_body_meshes(meshes):
        # Build a name map of vertex groups
        gnames = {vg.index: _clean(vg.name) for vg in o.vertex_groups}
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        try:
            mw = ev.matrix_world
            # normals
            me.calc_normals()
            for i, v in enumerate(me.vertices):
                wmax, wname = 0.0, ""
                for g in o.data.vertices[i].groups if i < len(o.data.vertices) else []:
                    if g.weight > wmax:
                        wmax = g.weight
                        wname = gnames.get(g.group, "")
                co = mw @ v.co
                no = (mw.to_3x3() @ v.normal).normalized()
                pts.append((co, no, wname, wmax))
        finally:
            ev.to_mesh_clear()
    return pts


def _loop_at_z(samples, z, half_band, inflate, pred, n=28, min_r=0.04):
    """Best-fit ring at height z from samples passing `pred(co, no, name, w)`."""
    picked = [s for s in samples if abs(s[0].z - z) <= half_band and pred(*s)]
    if len(picked) < 8:
        picked = [s for s in samples if abs(s[0].z - z) <= half_band * 2.2 and pred(*s)]
    if not picked:
        return None
    cx = sum(s[0].x for s in picked) / len(picked)
    cy = sum(s[0].y for s in picked) / len(picked)
    # For each angle, take the farthest radial sample (silhouette), plus inflate.
    ring = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        d = Vector((math.cos(a), math.sin(a), 0.0))
        best = min_r
        best_z = z
        for co, no, name, w in picked:
            rel = Vector((co.x - cx, co.y - cy, 0.0))
            if rel.length < 1e-8:
                continue
            if rel.normalized().dot(d) < 0.55:
                continue
            rad = rel.dot(d)
            if rad > best:
                best = rad
                best_z = co.z
        ring.append(Vector((cx + d.x * (best + inflate), cy + d.y * (best + inflate), best_z)))
    return ring


def _loft_rings(bm, rings, close_caps=False):
    if not rings:
        return None
    n = len(rings[0])
    rows = []
    for ring in rings:
        rows.append([bm.verts.new(p) for p in ring])
    bm.verts.ensure_lookup_table()
    for i in range(len(rows) - 1):
        if len(rows[i]) != n or len(rows[i + 1]) != n:
            continue
        for j in range(n):
            k = (j + 1) % n
            try:
                bm.faces.new((rows[i][j], rows[i][k], rows[i + 1][k], rows[i + 1][j]))
            except Exception:
                pass
    if close_caps:
        try:
            bm.faces.new(list(reversed(rows[0])))
            bm.faces.new(rows[-1])
        except Exception:
            pass
    return rows


def _copy_weights_nearest(dest, src):
    """Fallback: copy vertex groups from the nearest source vert (world space)."""
    from mathutils import kdtree as _kd
    sw = src.matrix_world
    kd = _kd.KDTree(len(src.data.vertices))
    for i, v in enumerate(src.data.vertices):
        kd.insert(sw @ v.co, i)
    kd.balance()
    # Ensure dest has the same groups
    for vg in src.vertex_groups:
        if dest.vertex_groups.get(vg.name) is None:
            dest.vertex_groups.new(name=vg.name)
    name_of = {vg.index: vg.name for vg in src.vertex_groups}
    src_groups = src.data.vertices
    dw = dest.matrix_world
    for v in dest.data.vertices:
        _co, idx, _d = kd.find(dw @ v.co)
        if idx is None:
            continue
        for g in src_groups[idx].groups:
            gname = name_of.get(g.group)
            if not gname:
                continue
            dest.vertex_groups[gname].add([v.index], g.weight, "REPLACE")


def _transfer_weights_and_bind(dest, meshes, arm):
    """Copy body vertex groups, add an armature modifier, parent in bind space."""
    srcs = _primary_body_meshes(meshes)
    if not srcs:
        return False
    src = srcs[0]
    # Put dest verts into armature-local bind space.
    inv = arm.matrix_world.inverted()
    for v in dest.data.vertices:
        v.co = inv @ v.co
    dest.data.update()
    dest.parent = arm
    dest.parent_type = "OBJECT"
    dest.matrix_parent_inverse = Matrix.Identity(4)
    dest.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()

    ok = False
    try:
        mod = dest.modifiers.new("WTransfer", "DATA_TRANSFER")
        mod.object = src
        mod.use_vert_data = True
        try:
            mod.data_types_verts = {"VGROUP_WEIGHTS"}
        except Exception:
            pass
        try:
            mod.vert_mapping = "POLYINTERP_NEAREST"
        except Exception:
            try:
                mod.vert_mapping = "NEAREST_FACE_INTERPOLATED"
            except Exception:
                pass
        try:
            mod.layers_vgroup_select_src = "ALL"
            mod.layers_vgroup_select_dst = "NAME"
        except Exception:
            pass
        _apply_modifier(dest, mod)
        ok = any(len(v.groups) for v in dest.data.vertices)
    except Exception:
        ok = False
    if not ok:
        _copy_weights_nearest(dest, src)

    arm_mod = dest.modifiers.new("Armature", "ARMATURE")
    arm_mod.object = arm
    arm_mod.use_vertex_groups = True
    try:
        while dest.modifiers[0] != arm_mod:
            bpy.context.view_layer.objects.active = dest
            bpy.ops.object.modifier_move_up(modifier=arm_mod.name)
    except Exception:
        pass
    return True


def _assign_region_mats(obj, satin, trim, regions):
    """regions: list of (mat_index, pred(x,y,z)->bool) after materials set."""
    me = obj.data
    me.materials.clear()
    me.materials.append(satin)
    me.materials.append(trim)
    for poly in me.polygons:
        c = Vector((0, 0, 0))
        for i in poly.vertices:
            c += me.vertices[i].co
        c /= max(1, len(poly.vertices))
        poly.material_index = 1 if regions(c.x, c.y, c.z) else 0
    me.update()



def _stitch_loops(bm, loop_a, loop_b):
    """Bridge two BMVert loops of equal or similar length."""
    a, b = list(loop_a), list(loop_b)
    if not a or not b:
        return
    # Align b so b[0] is closest to a[0]
    off = min(range(len(b)), key=lambda i: (b[i].co - a[0].co).length)
    b = b[off:] + b[:off]
    # Flip b if the next step would cross
    if len(b) > 1 and len(a) > 1:
        d_fwd = (b[1].co - a[1].co).length if len(b) > 1 else 1e9
        d_rev = (b[-1].co - a[1].co).length
        if d_rev < d_fwd:
            b = [b[0]] + list(reversed(b[1:]))
    n = min(len(a), len(b))
    if n < 2:
        return
    # If counts differ, walk the longer at a proportional index
    for i in range(n):
        j = (i + 1) % n
        ia, ja = int(i * len(a) / n), int(j * len(a) / n)
        ib, jb = int(i * len(b) / n), int(j * len(b) / n)
        try:
            bm.faces.new((a[ia], a[ja], b[jb], b[ib]))
        except Exception:
            try:
                bm.faces.new((a[ia], a[ja], b[ib]))
            except Exception:
                pass


def _finalize_garment_shell(obj, voxel=0.010, solid=0.005):
    """Solidify a closed (or nearly closed) shell, remesh, smooth."""
    sol = obj.modifiers.new("Thick", "SOLIDIFY")
    sol.thickness = float(solid)
    sol.offset = 1.0
    try:
        sol.use_quality_normals = True
    except Exception:
        pass
    _apply_modifier(obj, sol)
    _voxel_remesh(obj, voxel=voxel)
    _smooth_mesh(obj, iterations=3, factor=0.22)
    _shade_smooth(obj)
    return obj


def garment_poke_report(meshes, garment, pred, near=0.045, margin=0.001):
    """Count region verts that poke through a nearby garment surface."""
    if garment is None:
        return {"tested": 0, "pokes": 0}
    deps = bpy.context.evaluated_depsgraph_get()
    ev = garment.evaluated_get(deps)
    me = ev.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.transform(bm, verts=bm.verts, matrix=ev.matrix_world)
        bvh = BVHTree.FromBMesh(bm, epsilon=0.0002)
    finally:
        ev.to_mesh_clear()
        bm.free()
    tested = pokes = 0
    for o in meshes:
        if o.type != "MESH" or not o.data.vertices or "joint" in o.name.lower():
            continue
        weights = _vg_max_weights(o, pred)
        evb = o.evaluated_get(deps)
        meb = evb.to_mesh()
        try:
            mw = evb.matrix_world
            for i, v in enumerate(meb.vertices):
                if i >= len(weights) or weights[i] < 0.35:
                    continue
                wp = mw @ v.co
                hit = bvh.find_nearest(wp)
                if not hit or hit[0] is None:
                    continue
                loc, normal, _idx, dist = hit
                if dist > near:
                    continue
                tested += 1
                if (wp - loc).dot(normal) > margin:
                    pokes += 1
        finally:
            evb.to_mesh_clear()
    return {"tested": tested, "pokes": pokes}


def build_trunks_v4(name, collection, meshes, arm, colors, clearance=0.016):
    """Classic satin boxing trunks, closed shell, bound to the armature.

    Wide waistband high at the navel, loose mid-thigh legs, side stripe,
    thin black trim on waistband and hems. Fitted through the seat so hip
    rotation still reads. Dimensions are world metres at rest.
    """
    prev = _rest_context(arm)
    try:
        samples = _world_body_samples(meshes, arm)
        hips_bn = find_bone(arm, "mixamorig:Hips", "Hips", "hips")
        spine_bn = find_bone(arm, "mixamorig:Spine", "Spine", "spine")
        spine1_bn = find_bone(arm, "mixamorig:Spine1", "Spine1", "spine1")
        upleg_l = find_bone(arm, "mixamorig:LeftUpLeg", "LeftUpLeg")
        leg_l = find_bone(arm, "mixamorig:LeftLeg", "LeftLeg")
        hips_w = bone_world_matrix(arm, hips_bn).to_translation() if hips_bn else Vector((0, 0, 0.9))
        spine_w = bone_world_matrix(arm, spine_bn).to_translation() if spine_bn else hips_w + Vector((0, 0, 0.08))
        spine1_w = bone_world_matrix(arm, spine1_bn).to_translation() if spine1_bn else spine_w + Vector((0, 0, 0.08))
        # Navel is between hips and spine, slightly toward spine.
        navel_z = hips_w.z * 0.25 + spine_w.z * 0.75
        waist_top = navel_z + 0.015
        band_h = 0.062
        waist_bot = waist_top - band_h
        # Mid-thigh: halfway along the upleg (hip to knee).
        if upleg_l and leg_l:
            a = bone_world_matrix(arm, upleg_l).to_translation()
            b = bone_world_matrix(arm, leg_l).to_translation()
            mid_thigh_z = (a.z + b.z) * 0.50
        else:
            mid_thigh_z = hips_w.z - 0.22
        hem_z = mid_thigh_z

        def torso_pred(co, no, name, w):
            # Hips / pelvis / waist, exclude legs far from midline at low z
            if any(k in name for k in ("hand", "arm", "fore", "head", "neck", "foot", "toe")):
                return False
            return True

        n = 32
        zs = [
            waist_top + 0.002,
            waist_top - 0.018,
            waist_bot + 0.004,
            waist_bot - 0.012,          # upper hip
            hips_w.z - 0.01,            # seat
            hips_w.z - 0.06,
            (hips_w.z + hem_z) * 0.55,  # just below crotch branch — still one ring; we'll split later
        ]
        # Until we split into legs, only go down to crotch.
        crotch_z = hips_w.z - 0.09
        zs_torso = [
            waist_top + 0.002,
            waist_top - 0.020,
            waist_bot + 0.006,
            waist_bot - 0.010,
            hips_w.z + 0.01,
            hips_w.z - 0.04,
            crotch_z,
        ]
        infl = [
            clearance * 0.7,  # snug waistband
            clearance * 0.7,
            clearance * 0.85,
            clearance * 1.1,
            clearance * 1.4,  # seat ease, still follows the cheek
            clearance * 1.6,
            clearance * 1.5,
        ]
        rings = []
        for z, inf in zip(zs_torso, infl):
            band = 0.035 if z > waist_bot else 0.05
            r = _loop_at_z(samples, z, band, inf, torso_pred, n=n, min_r=0.07)
            if r is None:
                raise RuntimeError(f"trunks: no ring at z={z:.3f}")
            # flatten z for a cleaner waistband
            if z >= waist_bot - 0.002:
                for p in r:
                    p.z = z
            rings.append(r)

        # Two legs from crotch to hem. Use left/right body samples.
        def leg_pred(side):
            token = "left" if side == "L" else "right"
            def pred(co, no, name, w):
                if token not in name and "upleg" not in name and "thigh" not in name and "leg" not in name:
                    # still allow unlabelled verts on that side of midline
                    return (co.x >= 0.01) if side == "L" else (co.x <= -0.01)
                if side == "L" and "right" in name:
                    return False
                if side == "R" and "left" in name:
                    return False
                return "foot" not in name and "hand" not in name and "arm" not in name
            return pred

        # Split the crotch ring into left/right halves for leg starts.
        crotch = rings[-1]
        # Character: +X is left in Mixamo? Typically +X is left. We'll split by x.
        mid_x = sum(p.x for p in crotch) / len(crotch)

        def side_of(p, side):
            return p.x >= mid_x if side == "L" else p.x <= mid_x

        # Build each leg as its own loft, then join.
        mesh = bpy.data.meshes.new(name + "Mesh")
        obj = _new_mesh_object(name, mesh, collection)
        bm = bmesh.new()
        rows = _loft_rings(bm, rings, close_caps=False)
        # Cap the waist opening with an inset (closed garment: inner waist lip)
        # We'll solidify later; cap the top and each hem after legs.

        # Leg rings
        leg_zs = [crotch_z - 0.01, (crotch_z + hem_z) * 0.5, hem_z + 0.012, hem_z]
        leg_inf = [clearance * 1.7, clearance * 2.2, clearance * 2.3, clearance * 2.1]
        leg_rows = {}
        for side in ("L", "R"):
            lrings = []
            for z, inf in zip(leg_zs, leg_inf):
                r = _loop_at_z(samples, z, 0.05, inf, leg_pred(side), n=n // 2, min_r=0.05)
                if r is None:
                    # fallback: take crotch half and scale down toward hem
                    half = [p.copy() for p in crotch if side_of(p, side)]
                    if len(half) < 6:
                        raise RuntimeError(f"trunks: no {side} leg at z={z:.3f}")
                    # resample
                    cx = sum(p.x for p in half) / len(half)
                    cy = sum(p.y for p in half) / len(half)
                    shrink = 0.92 if z > hem_z + 0.01 else 0.90
                    r = [Vector((cx + (p.x - cx) * shrink, cy + (p.y - cy) * shrink, z)) for p in half]
                for p in r:
                    p.z = z
                lrings.append(r)
            # Match vert count
            m = len(lrings[0])
            lrings = [r if len(r) == m else _resample_ring(r, m) for r in lrings]
            leg_rows[side] = _loft_rings(bm, lrings, close_caps=False)
            # Close hem
            try:
                bm.faces.new(list(reversed(leg_rows[side][-1])))
            except Exception:
                pass
            # Bridge the last torso (crotch) ring half to the top of this leg.
            half = [v for v in rows[-1] if side_of(v.co, side)]
            if half and leg_rows[side]:
                _stitch_loops(bm, half, leg_rows[side][0])

        # Close the waist: inset cap so it's a closed volume after solidify too.
        try:
            bm.faces.new(rows[0])
        except Exception:
            pass

        bm.to_mesh(mesh)
        bm.free()
        _finalize_garment_shell(obj, voxel=0.011, solid=0.006)

        # Bind in rest, then restore pose
        _transfer_weights_and_bind(obj, meshes, arm)

        # Region flags are in bind (armature local) space now — centimetres
        # if the armature was scaled 0.01. Use world bounds before bind? We
        # already bound. Re-read world after update for material assignment
        # is messy. Assign by object-local z: after inv @ world, z is armature
        # local. Safer to assign BEFORE bind, when verts are still world metres.
        # Too late. Convert thresholds into current local by using the object's
        # bounding box percentiles.
        zs_local = [v.co.z for v in obj.data.vertices]
        zmax, zmin = max(zs_local), min(zs_local)
        zspan = max(1e-6, zmax - zmin)
        # waistband = top ~18% of height; hem = bottom ~10%; side stripe = large |x|
        xs = [abs(v.co.x) for v in obj.data.vertices]
        xside = sorted(xs)[int(0.82 * len(xs))] if xs else 0.1

        def trim_pred(x, y, z):
            top = (zmax - z) / zspan < 0.18 and (zmax - z) / zspan > 0.0
            # thin edges of the waistband only
            waist_edge = (zmax - z) / zspan <= 0.035 or (0.145 <= (zmax - z) / zspan <= 0.18)
            hem = (z - zmin) / zspan <= 0.07
            stripe = abs(x) >= xside * 0.92 and (z - zmin) / zspan > 0.08 and (zmax - z) / zspan > 0.18
            return waist_edge or hem  # side stripe painted with a slightly darker satin later? user asked thin black trim on waistband and hems. Stripe can be same charcoal or thin black. I'll use trim (black) for waist/hem only, stripe stays charcoal — or a thin black side stripe.
            # include a thin side stripe as trim
            # return waist_edge or hem or (abs(x) >= xside * 0.96 and 0.10 < (z-zmin)/zspan < 0.82)

        # redefine cleanly
        def trim_pred(x, y, z, _zmax=zmax, _zmin=zmin, _zspan=zspan, _xside=xside):
            ttop = (_zmax - z) / _zspan
            tbot = (z - _zmin) / _zspan
            waist_edge = ttop <= 0.035 or (0.145 <= ttop <= 0.185)
            hem = tbot <= 0.07
            stripe = abs(x) >= _xside * 0.96 and 0.10 < tbot and ttop > 0.20
            return waist_edge or hem or stripe

        satin = make_satin_material(name + "Satin", colors["beanie"], roughness=0.20)
        trim = make_satin_material(name + "Trim", TRIM_V4_BLACK, roughness=0.28)
        _assign_region_mats(obj, satin, trim, trim_pred)

        info = {
            "version": "v4",
            "waist_top_m": waist_top,
            "hem_z_m": hem_z,
            "navel_z_m": navel_z,
            "clearance_m": clearance,
        }
        return obj, info
    finally:
        _restore_pose(arm, prev)


def _resample_ring(ring, n):
    if len(ring) == n:
        return ring
    # cumulative length
    pts = list(ring) + [ring[0]]
    segl = [ (pts[i+1] - pts[i]).length for i in range(len(ring)) ]
    total = sum(segl) or 1.0
    out = []
    for k in range(n):
        target = total * k / n
        acc = 0.0
        for i, L in enumerate(segl):
            if acc + L >= target or i == len(segl) - 1:
                t = 0.0 if L < 1e-8 else (target - acc) / L
                out.append(pts[i].lerp(pts[i + 1], max(0.0, min(1.0, t))))
                break
            acc += L
    return out


def build_tank_v4(name, collection, meshes, arm, colors, clearance=0.008):
    """Fitted sleeveless athletic tank. Charcoal satin, closed shell.

    Wide armholes and thin medial straps so deltoids, arms and scapulae
    stay uncovered (shoulder glow remains visible). Bound to the armature.
    """
    prev = _rest_context(arm)
    try:
        samples = _world_body_samples(meshes, arm)
        hips_bn = find_bone(arm, "mixamorig:Hips", "Hips", "hips")
        spine_bn = find_bone(arm, "mixamorig:Spine", "Spine", "spine")
        spine1_bn = find_bone(arm, "mixamorig:Spine1", "Spine1", "spine1")
        spine2_bn = find_bone(arm, "mixamorig:Spine2", "Spine2", "spine2")
        sh_l = find_bone(arm, "mixamorig:LeftShoulder", "LeftShoulder")
        sh_r = find_bone(arm, "mixamorig:RightShoulder", "RightShoulder")
        neck_bn = find_bone(arm, "mixamorig:Neck", "Neck", "neck")
        hips_w = bone_world_matrix(arm, hips_bn).to_translation() if hips_bn else Vector((0, 0, 0.9))
        spine_w = bone_world_matrix(arm, spine_bn).to_translation() if spine_bn else hips_w
        spine1_w = bone_world_matrix(arm, spine1_bn).to_translation() if spine1_bn else spine_w
        spine2_w = bone_world_matrix(arm, spine2_bn).to_translation() if spine2_bn else spine1_w
        neck_w = bone_world_matrix(arm, neck_bn).to_translation() if neck_bn else spine2_w + Vector((0, 0, 0.08))
        sh_lw = bone_world_matrix(arm, sh_l).to_translation() if sh_l else Vector((0.12, 0, 1.3))
        sh_rw = bone_world_matrix(arm, sh_r).to_translation() if sh_r else Vector((-0.12, 0, 1.3))
        # Armhole sits at the armpit crease, clearly below the deltoid mass
        # so the glow on shoulders / lateral delts stays uncovered.
        armpit_z = min(sh_lw.z, sh_rw.z) - 0.055
        # Hem overlaps the trunks waistband slightly.
        navel_z = hips_w.z * 0.25 + spine_w.z * 0.75
        hem_z = navel_z - 0.025
        # Torso of the tank stops at armpit; straps go up from there.
        chest_z = (spine1_w.z + spine2_w.z) * 0.5

        def torso_pred(co, no, name, w):
            # No arms, no shoulders, no head — this keeps the shell off deltoids.
            if any(k in name for k in ("arm", "fore", "hand", "shoulder", "head", "neck", "leg", "foot", "upleg", "thigh")):
                return False
            return True

        n = 36
        zs = [
            hem_z,
            hem_z + 0.03,
            navel_z + 0.02,
            spine_w.z + 0.02,
            chest_z,
            armpit_z - 0.01,
        ]
        rings = []
        for z in zs:
            r = _loop_at_z(samples, z, 0.04, clearance, torso_pred, n=n, min_r=0.08)
            if r is None:
                raise RuntimeError(f"tank: no torso ring at z={z:.3f}")
            # Cut the sides above mid-chest so the armhole is already opening
            # (pull side verts inward toward the ribs, not the deltoid).
            if z >= chest_z - 0.01:
                cx = sum(p.x for p in r) / len(r)
                cy = sum(p.y for p in r) / len(r)
                # Identify leftmost/rightmost verts and pinch them in, more at the top.
                pinch = _smoothstep(chest_z - 0.01, armpit_z, z) * 0.55
                xs = [p.x for p in r]
                xmin, xmax = min(xs), max(xs)
                for p in r:
                    # side = near max |x|
                    side = 0.0
                    if p.x > 0:
                        side = _smoothstep(xmax * 0.45, xmax * 0.85, p.x)
                    else:
                        side = _smoothstep(abs(xmin) * 0.45, abs(xmin) * 0.85, abs(p.x))
                    if side > 0:
                        mid = Vector((cx, cy, p.z))
                        q = mid.lerp(p, 1.0 - pinch * side * 0.85)
                        p.x, p.y, p.z = q.x, q.y, q.z
            rings.append(r)

        # Thin medial straps: from the front upper-chest (near sternum/clavicle)
        # over the inner shoulder (next to the neck, NOT the deltoid) to the
        # upper back, meeting as a racerback so the scapulae stay visible.
        mid_sh = (sh_lw + sh_rw) * 0.5
        # neck-adjacent shoulder points (medial)
        def medial_shoulder(sh, k=0.72):
            return mid_sh.lerp(sh, 1.0 - k) + Vector((0, 0, 0.015))

        strap_l_top = medial_shoulder(sh_lw) + Vector((0.0, 0.0, 0.02))
        strap_r_top = medial_shoulder(sh_rw) + Vector((0.0, 0.0, 0.02))
        # Front attach: upper chest, slightly medial
        front_z = armpit_z - 0.005
        # Use the top torso ring to pick front/back attach points
        top = rings[-1]
        def pick(ring, want_front, want_left):
            best, score = None, None
            cy = sum(p.y for p in ring) / len(ring)
            cx = sum(p.x for p in ring) / len(ring)
            for p in ring:
                front = (p.y - cy)  # Mixamo faces -Y, so front is smaller y
                # after face -Y, front of body is -Y (negative y)
                front_s = -p.y
                left_s = p.x
                sc = (front_s if want_front else -front_s) + 0.15 * (left_s if want_left else -left_s)
                # prefer not too lateral
                sc -= 0.8 * abs(p.x)
                if want_left and p.x < 0:
                    sc -= 1.0
                if (not want_left) and p.x > 0:
                    sc -= 1.0
                if score is None or sc > score:
                    best, score = p, sc
            return best.copy()

        fl = pick(top, True, True)
        fr = pick(top, True, False)
        bl = pick(top, False, True)
        br = pick(top, False, False)
        # Pull attach points slightly medial / up
        def medialize(p, k=0.35):
            c = Vector((0.0, p.y, p.z))
            return c.lerp(p, 1.0 - k)

        fl, fr, bl, br = medialize(fl), medialize(fr), medialize(bl, 0.45), medialize(br, 0.45)
        # Racerback: back straps meet near the spine, a bit below the neck.
        back_meet = Vector((0.0, (bl.y + br.y) * 0.5, (bl.z + neck_w.z) * 0.45 + 0.02))
        # Nudge back_meet onto the back surface
        back_hits = [s[0] for s in samples if abs(s[0].z - back_meet.z) < 0.04 and s[0].y > 0]
        # facing -Y, back is +Y
        back_hits = [s[0] for s in samples if abs(s[0].z - back_meet.z) < 0.05 and s[0].y > (bl.y + br.y) * 0.5]
        if back_hits:
            # furthest back (max y)
            hit = max(back_hits, key=lambda p: p.y)
            back_meet = Vector((0.0, hit.y + clearance + 0.003, hit.z))

        def strap_tube(a, b, c, r=0.016, segs=8, steps=7):
            """Lofted tube through a-b-c (quadratic)."""
            pts_path = []
            for i in range(steps + 1):
                t = i / steps
                # quadratic bezier
                p = (1-t)*(1-t)*a + 2*(1-t)*t*b + t*t*c
                pts_path.append(p)
            rings_s = []
            for p, nxt in zip(pts_path, pts_path[1:] + pts_path[-1:]):
                d = (nxt - p)
                if d.length < 1e-8:
                    d = Vector((0, 0, 1))
                d.normalize()
                # arbitrary perpendicular
                aux = Vector((0, 0, 1)) if abs(d.z) < 0.9 else Vector((1, 0, 0))
                u = d.cross(aux).normalized()
                v = d.cross(u).normalized()
                ring = [p + (u * math.cos(2*math.pi*j/segs) + v * math.sin(2*math.pi*j/segs)) * r
                        for j in range(segs)]
                rings_s.append(ring)
            return rings_s

        mesh = bpy.data.meshes.new(name + "Mesh")
        obj = _new_mesh_object(name, mesh, collection)
        bm = bmesh.new()
        torso_rows = _loft_rings(bm, rings, close_caps=False)
        # Cap the hem (first ring is the waist hem).
        if torso_rows:
            try:
                bm.faces.new(list(reversed(torso_rows[0])))
            except Exception:
                pass
        # straps: front attach -> top (medial shoulder) -> back meet (racer)
        for front, top, back in (
            (fl, strap_l_top, back_meet),
            (fr, strap_r_top, back_meet),
        ):
            # Keep straps medial (next to the neck, not over the deltoid).
            top = Vector((top.x * 0.50, top.y, max(top.z, neck_w.z - 0.035)))
            sr = strap_tube(front, top, back, r=0.014, segs=8, steps=8)
            _loft_rings(bm, sr, close_caps=True)
        bm.to_mesh(mesh)
        bm.free()
        _finalize_garment_shell(obj, voxel=0.007, solid=0.0045)
        _transfer_weights_and_bind(obj, meshes, arm)
        satin = make_satin_material(name + "Satin", colors["beanie"], roughness=0.20)
        _assign_mat(obj, satin)
        info = {
            "version": "v4",
            "hem_z_m": hem_z,
            "armpit_z_m": armpit_z,
            "clearance_m": clearance,
            "racerback": True,
        }
        return obj, info
    finally:
        _restore_pose(arm, prev)


def _inside_bvh(bvh, point, margin=0.0):
    hit = bvh.find_nearest(point)
    if not hit or hit[0] is None:
        return False
    loc, normal, *_ = hit
    # inside if we are behind the outward normal, plus a small margin
    return (point - loc).dot(normal) <= margin


def mask_inside_garment(meshes, garment, vg_name, mod_name, pred, margin=0.002):
    """Hide body verts that sit inside a closed garment and match `pred`."""
    if garment is None:
        return 0
    deps = bpy.context.evaluated_depsgraph_get()
    ev = garment.evaluated_get(deps)
    me = ev.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.transform(bm, verts=bm.verts, matrix=ev.matrix_world)
        bvh = BVHTree.FromBMesh(bm, epsilon=0.0002)
    finally:
        ev.to_mesh_clear()
        bm.free()
    hidden = 0
    for o in meshes:
        if o.type != "MESH" or not o.data.vertices:
            continue
        if "joint" in o.name.lower():
            continue
        weights_ok = None
        if pred is not None:
            weights_ok = _vg_max_weights(o, pred)
        evb = o.evaluated_get(deps)
        meb = evb.to_mesh()
        idxs = []
        try:
            mw = evb.matrix_world
            for i, v in enumerate(meb.vertices):
                if weights_ok is not None and (i >= len(weights_ok) or weights_ok[i] < 0.20):
                    continue
                wp = mw @ v.co
                if _inside_bvh(bvh, wp, margin=margin):
                    idxs.append(i)
        finally:
            evb.to_mesh_clear()
        if not idxs:
            continue
        vg = o.vertex_groups.get(vg_name) or o.vertex_groups.new(name=vg_name)
        vg.add(sorted(set(idxs)), 1.0, "ADD")
        if mod_name not in o.modifiers:
            mod = o.modifiers.new(mod_name, "MASK")
            try:
                mod.vertex_group = vg_name
                mod.invert_vertex_group = True
            except Exception:
                pass
        hidden += len(set(idxs))
    return hidden



def apply_outfit(arm, meshes, *, items=("gloves", "beanie"), colorway="A",
                 scan_frames=True, collection_name="XBotOutfit", version="v4"):
    """Build and attach the requested items. Returns a handle dict.

    `items` is a subset of {"gloves", "beanie", "headgear", "strap",
    "trunks", "tank"} ("strap" is implied by gloves). Prefer beanie XOR
    headgear. Gloves and beanie are rigid-parented to Hand / Head;
    trunks and tank are closed garment shells weight-bound to the armature.

    `version` is "v4" (default: glossy black CC0-matched gloves, charcoal
    rib-knit beanie, optional charcoal satin trunks + fitted tank), "v3"
    (crimson 0.8×-head kit), "v2" (compact), or "v1" (long mitten + smooth cap).
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
        "version": version,
    }
    if both:
        handle["notes"].append("beanie and headgear both requested; they will overlap")

    _v23 = version in ("v2", "v3")
    if version == "v4":
        leather = make_glossy_leather_material(
            f"XBotGloveLeather_{colorway}_{version}", GLOVE_V4_BLACK, roughness=0.30,
        )
        leather_cuff = leather
        leather_strap = leather
    else:
        leather = make_leather_material(
            f"XBotGloveLeather_{colorway}_{version}", colors["glove"],
            roughness=0.55 if _v23 else 0.64,
            wrinkles=_v23,
        )
    if version == "v3":
        # Cuff is the same crimson as the fist; only a thin slightly
        # darker-red hook-and-loop seam (never a black wrist band).
        cuff_col = colors["glove"]
        strap_col = tuple(max(0.018, c * 0.78) for c in colors["glove"])
        leather_cuff = make_leather_material(
            f"XBotGloveCuff_{colorway}_{version}", cuff_col,
            roughness=0.58, wrinkles=True,
        )
        leather_strap = make_leather_material(
            f"XBotGloveStrap_{colorway}_{version}", strap_col,
            roughness=0.64, wrinkles=False,
        )
    else:
        leather_cuff = make_leather_material(
            f"XBotGloveCuff_{colorway}_{version}", colors["glove_cuff"],
            roughness=0.62 if version == "v2" else 0.72,
            wrinkles=(version == "v2"),
        )
        leather_strap = make_leather_material(
            f"XBotGloveStrap_{colorway}_{version}", colors["glove_strap"],
            roughness=0.70 if version == "v2" else 0.75,
            wrinkles=False,
        )
    knit = make_knit_material(
        f"XBotBeanieKnit_{colorway}_{version}", colors["beanie"],
        strength=(0.52 if version in ("v3", "v4") else 0.45 if version == "v2" else 0.22),
    )
    foam = make_foam_leather_material(f"XBotHeadgear_{colorway}", colors["headgear"], roughness=0.74)
    strap_mat = make_leather_material(f"XBotStrap_{colorway}", colors["strap"], roughness=0.78)

    head_bone = find_bone(arm, "mixamorig:Head", "Head", "head")
    handle["head_bone"] = head_bone
    head_pts = _head_points(meshes, arm, head_bone) if head_bone else []
    head_width = None
    if head_pts:
        xs = [p.x for p in head_pts]
        head_width = max(xs) - min(xs)
        handle["head_width"] = head_width

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
            glove, ginfo = build_glove_mesh(
                gname, col, env, wrist_radius=wr, version=version,
                head_width=head_width,
            )
            if version == "v4":
                _assign_mat(glove, leather)
            elif version == "v3":
                _assign_glove_mats_v3(glove, leather, leather_strap, ginfo)
            elif version == "v2":
                _assign_glove_mats_v2(glove, leather, leather_cuff, leather_strap, ginfo)
            else:
                _assign_mat(glove, leather)
            glove.matrix_world = bone_world_matrix(arm, bn)
            parent_to_bone(glove, arm, bn)
            handle["objects"].append(glove.name)
            handle["bones"][glove.name] = bn
            handle.setdefault("glove_info", {})[side] = {**ginfo, "wrist_r": wr}
        if version == "v4":
            fw = (handle.get("glove_info") or {}).get("R", {}).get("front_width") or                  (handle.get("glove_info") or {}).get("L", {}).get("front_width")
            handle["notes"].append(
                "v4 gloves are rigid to the Hand bone (no finger-curl squash). "
                "Glossy black leather matched to the CC0 reference silhouette: "
                "bulbous body, side-lobe thumb, smooth same-black cuff, subtle stitch grooves"
                + (f"; fist front {fw:.1f} cm vs head {head_width:.1f} cm"
                   if fw and head_width else "")
                + "."
            )
        elif version == "v3":
            fw = (handle.get("glove_info") or {}).get("R", {}).get("front_width") or                  (handle.get("glove_info") or {}).get("L", {}).get("front_width")
            handle["notes"].append(
                "v3 gloves are rigid to the Hand bone (no finger-curl squash). "
                f"10–12 oz proportions: fist front ~0.8× head width"
                + (f" ({fw:.1f} cm vs head {head_width:.1f} cm)" if fw and head_width else "")
                + ", longer padded knuckle, visible attached thumb, same-red 9 cm cuff."
            )
        elif version == "v2":
            handle["notes"].append(
                "v2 gloves are rigid to the Hand bone (no finger-curl squash). "
                "Compact 10–12 oz proportions: rounded fist ~1.55× knuckle width, "
                "distinct 9 cm cuff overlapping the forearm, attached thumb."
            )
        else:
            handle["notes"].append(
                "Gloves are rigid to the Hand bone (they do not squash with finger curl). "
                "Sized to a union envelope of the clip plus a stored Idle+Jab envelope "
                "so fingertips should stay inside across the jab."
            )

    if "beanie" in items:
        if not head_bone or not head_pts:
            handle["notes"].append("beanie skipped: no Head bone / verts")
        else:
            cap, binfo = build_beanie(f"XBotBeanie_{colorway}", col, head_pts, version=version)
            _assign_mat(cap, knit)
            cap.matrix_world = bone_world_matrix(arm, head_bone)
            parent_to_bone(cap, arm, head_bone)
            handle["objects"].append(cap.name)
            handle["bones"][cap.name] = head_bone
            handle["beanie_info"] = binfo
            clearance = (BEANIE_V2_CLEARANCE_CM if version in ("v2", "v3", "v4")
                         else BEANIE_CLEARANCE_CM)
            if version in ("v3", "v4"):
                extra = (f"; {version} has a fold-up cuff, rib knit continuing to the "
                         "crown, slight slouch/fold).")
            elif version == "v2":
                extra = ("; v2 has a fold-up cuff, geometric rib knit, slight slouch).")
            else:
                extra = ")."
            handle["notes"].append(
                f"Beanie hugged to the skull ({binfo['moved']} verts pushed out to "
                f"{clearance:.2f} cm clearance" + extra
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

    if "trunks" in items:
        trunks, tinfo = build_trunks_v4(
            f"XBotTrunks_{colorway}", col, meshes, arm, colors,
        )
        handle["objects"].append(trunks.name)
        handle["trunks_info"] = tinfo
        hidden = mask_inside_garment(
            meshes, trunks, "xbot_trunks_hide", "XBotTrunksMask",
            pred=lambda n: any(k in _clean(n) for k in
                               ("hip", "pelvis", "spine", "upleg", "thigh",
                                "butt", "glute")),
            margin=0.002,
        )
        handle.setdefault("garment_masked", {})["trunks"] = hidden
        handle["notes"].append(
            "v4 trunks: charcoal satin (same as the beanie base), wide navel "
            "waistband, mid-thigh legs, thin black waist/hem/side trim; "
            f"armature-bound, {hidden} body verts masked inside."
        )

    if "tank" in items:
        tank, tank_info = build_tank_v4(
            f"XBotTank_{colorway}", col, meshes, arm, colors,
        )
        handle["objects"].append(tank.name)
        handle["tank_info"] = tank_info
        hidden = mask_inside_garment(
            meshes, tank, "xbot_tank_hide", "XBotTankMask",
            pred=lambda n: (
                any(k in _clean(n) for k in ("spine", "chest", "abdomen", "torso", "hip"))
                and not any(k in _clean(n) for k in
                            ("shoulder", "arm", "hand", "neck", "head",
                             "leg", "foot", "upleg"))
            ),
            margin=0.0015,
        )
        handle.setdefault("garment_masked", {})["tank"] = hidden
        handle["notes"].append(
            "v4 tank: fitted charcoal satin, sleeveless racerback, deltoids "
            f"and arms left uncovered so shoulder glow stays visible; "
            f"armature-bound, {hidden} torso verts masked inside."
        )

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
        cuff_y = info.get("cuff_y_far", info["env"]["min"][1] + 0.6) + 0.4
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
    mask_targets = []
    if handle.get("masked") or handle.get("garment_masked"):
        mask_targets = list(handle.get("masked") or [])
        if meshes:
            mask_targets = list({*mask_targets, *[o.name for o in meshes]})
        for name in mask_targets:
            obj = bpy.data.objects.get(name)
            if not obj:
                continue
            for mod_name, vg_name in (
                ("XBotGloveMask", "xbot_glove_hide"),
                ("XBotTrunksMask", "xbot_trunks_hide"),
                ("XBotTankMask", "xbot_tank_hide"),
            ):
                if mod_name in obj.modifiers:
                    obj.modifiers.remove(obj.modifiers[mod_name])
                vg = obj.vertex_groups.get(vg_name)
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
