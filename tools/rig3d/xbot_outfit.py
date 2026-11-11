#!/usr/bin/env python3
"""Procedural boxing outfit kit for Mixamo X Bot.

All meshes and materials are generated in Blender — no third-party assets,
so the kit is commercially free (public-domain geometry we authored).

Import from any render script (do not require eval_xbot_boxing.py):

    from xbot_outfit import apply_outfit, COLORWAYS, clear_outfit
    handle = apply_outfit(arm, meshes, items=("gloves", "beanie", "trunks", "tank"), colorway="A", version="v5")

Items:
  gloves    — rigid-parented to mixamorig:LeftHand / RightHand
  beanie    — knit cap, rigid-parented to mixamorig:Head
  headgear  — open-face amateur pads + chin strap, parented to Head
              (kept for v1 / explicit requests; v2 outfit A drops it)

`version` selects the mesh: "v5" (default) is a glossy black 10–12 oz
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
    "front_thick": 13.6,
    "body_len": 13.6,
    "cuff_len": 7.0,
    "cuff_r": 4.55,
    "thumb_len": 11.0,
    "thumb_rx": 3.50,
    "thumb_rz": 4.40,
}

# Lifted glossy black — AgX crushes anything near 0.02 to a silhouette.
GLOVE_V4_BLACK = (0.048, 0.044, 0.040)
TRIM_V4_BLACK = (0.036, 0.033, 0.030)

# v5: slightly longer cuff to cover the wrist on jab extension; thumb is a
# flattened kidney lobe that hugs the bulb (not a separate egg).
GLOVE_V5 = {
    "front_width": 14.1,
    "front_thick": 13.6,
    "body_len": 13.6,
    "cuff_len": 8.6,
    "cuff_r": 4.35,
    "thumb_len": 11.6,
    "thumb_rx": 2.15,
    "thumb_rz": 3.55,
}

# v5 trunks: deep burgundy/wine satin (linear), clearly darker and redder than
# the jab glow (front-delt purple, triceps gold). Thin black or charcoal trim.
# Tank stays the beanie charcoal.
TRUNKS_V5_BURGUNDY = (0.18, 0.025, 0.04)
TRIM_V5_CHARCOAL = (0.055, 0.052, 0.055)

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
    rim = tuple(min(1.0, c * 1.35 + 0.008) for c in color)
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
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    _set(bsdf, "Specular IOR Level", 0.50)
    _set(bsdf, "Specular", 0.50)
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Coat Weight", 0.22)
    _set(bsdf, "Coat Roughness", 0.16)
    _set(bsdf, "Coat IOR", 1.48)
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
    """Radius profile of the CC0 reference: plump bulb, short smooth cuff taper."""
    widen = _smoothstep(0.18, 0.40, t)
    plump = 0.07 * math.sin(math.pi * _smoothstep(0.42, 0.88, t))
    strap = 0.055 * math.exp(-((t - 0.13) / 0.036) ** 2)
    sx = cuff_r * (1.0 + strap) + (rx - cuff_r) * widen
    sz = cuff_r * 0.95 * (1.0 + strap) + (rz - cuff_r * 0.95) * widen
    sx *= (1.0 + plump)
    sz *= (1.0 + plump)
    if t > 0.84:
        s = (t - 0.84) / 0.16
        cap = math.sqrt(max(0.0, 1.0 - s * s * 0.90))
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
                v.co.x *= 1.0 - 0.012 * w * w
                v.co.z *= 1.0 - 0.005 * w * w
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
                    v.co -= radial.normalized() * (0.035 * w)
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
            # Shallow dished cap so the cuff reads as a closed hook-and-loop
            # strap, not an open jar mouth, in a product shot.
            y = y_open + 0.45 * (1.0 - t / 0.06)
            lip_s = 0.55 + 0.45 * (t / 0.06)
            sx, sz = cuff_r * lip_s, cuff_r * 0.97 * lip_s
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

    # Elongated thumb lobe along +Z, mid-bulb, base heavily buried so remesh
    # blends it into a side swell (not a cherry knob).
    body_mid_y = 0.22 * y_open + 0.78 * y_tip - spec["thumb_len"] * 0.18
    thumb_c = Vector((0.30, body_mid_y, rz * 0.62 + 0.15))
    thumb = _ellipsoid(
        name + "_thumb",
        (spec["thumb_rx"], spec["thumb_len"] * 0.62, spec["thumb_rz"]),
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

    _join_objects(obj, thumb)
    _voxel_remesh(obj, voxel=0.16)
    _smooth_mesh(obj, iterations=6, factor=0.28)
    _add_stitch_grooves_v4(obj, y_open, y_tip, rx, rz, cuff_r, thumb_c, spec["thumb_len"])
    _smooth_mesh(obj, iterations=2, factor=0.14)
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


def build_glove_mesh(name, collection, hand_env, wrist_radius=None, version="v5",
                     head_width=None):
    """Solid mitten in hand-bone local centimetres, origin at the bone head.

    +Y = knuckles, +Z = thumb side, cuff opening toward -Y. The body ellipsoid
    is centred on the full cuff-to-knuckle span so the wrist ring stays
    attached (a knuckle-biased body previously left a detached band mid-forearm).
    Pass version="v5" (default) for the kidney-thumb glossy black glove,
    "v3" for the crimson 0.8×-head kit, "v2" compact, "v1" long mitten.
    """
    if version == "v5":
        return build_glove_mesh_v5(name, collection, hand_env, wrist_radius=wrist_radius,
                                   head_width=head_width)
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


def build_beanie(name, collection, head_pts, brow_y=None, clearance=None, version="v5"):
    """Fitted watch-cap with a clean folded cuff, in head-bone local cm."""
    if version in ("v3", "v4", "v5"):
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
            # Blender 4.x dropped Mesh.calc_normals(); evaluated normals are fine.
            try:
                me.calc_normals()
            except AttributeError:
                pass
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


def _drop_tiny_islands(bm, min_frac=0.02, min_verts=40):
    """Drop only tiny loose scraps; keep every substantial island (legs, straps)."""
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    seen = set()
    islands = []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack = [v]
        island = []
        seen.add(v.index)
        while stack:
            cur = stack.pop()
            island.append(cur)
            for e in cur.link_edges:
                oth = e.other_vert(cur)
                if oth.index not in seen:
                    seen.add(oth.index)
                    stack.append(oth)
        islands.append(island)
    if not islands:
        return
    n = max(1, len(bm.verts))
    thresh = max(int(min_verts), int(min_frac * n))
    drop = [v for isl in islands if len(isl) < thresh for v in isl]
    if drop:
        bmesh.ops.delete(bm, geom=drop, context="VERTS")


def _world_inflate(obj, inflate_fn):
    """Move verts along world-space normals. inflate_fn(world_co) -> metres."""
    mw = obj.matrix_world
    imw = mw.inverted()
    rot = mw.to_3x3()
    me = obj.data
    me.calc_loop_triangles()
    # Use bmesh for reliable normals
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        wco = mw @ v.co
        wn = rot @ v.normal
        if wn.length < 1e-8:
            continue
        wn.normalize()
        off = float(inflate_fn(wco))
        v.co = imw @ (wco + wn * off)
    bm.to_mesh(me)
    bm.free()
    me.update()


def _bisect_world_z(obj, z, keep_below=True):
    """Cut a horizontal plane in world space and discard one half."""
    mw = obj.matrix_world
    imw = mw.inverted()
    plane_co = imw @ Vector((0.0, 0.0, z))
    # normal in object space: transform a world +Z direction
    plane_no = (imw.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(
        bm,
        geom=geom,
        dist=0.0002,
        plane_co=plane_co,
        plane_no=plane_no,
        clear_inner=not keep_below,  # clear below the plane if keep_below is False
        clear_outer=keep_below,      # clear above if we keep below
    )
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()



def _relax_boundaries(obj, iterations=8, factor=0.45):
    """Even out jagged bisect rims (hem, waist, armholes, neck)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    for _ in range(iterations):
        boundary = [v for v in bm.verts if v.is_boundary]
        news = {}
        for v in boundary:
            pts = [e.other_vert(v).co for e in v.link_edges if e.other_vert(v).is_boundary]
            if len(pts) < 2:
                continue
            mid = sum(pts, Vector()) / len(pts)
            news[v] = v.co.lerp(mid, factor)
        for v, co in news.items():
            v.co = co
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _delete_verts(obj, keep_flags):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    drop = [v for v in bm.verts if v.index >= len(keep_flags) or not keep_flags[v.index]]
    if drop:
        bmesh.ops.delete(bm, geom=drop, context="VERTS")
    _drop_tiny_islands(bm)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _bind_like_body(dest, src, arm):
    """Share the body's parent/transform and deform with the same groups."""
    dest.parent = src.parent
    dest.parent_type = src.parent_type
    if getattr(src, "parent_bone", ""):
        try:
            dest.parent_bone = src.parent_bone
        except Exception:
            pass
    dest.matrix_parent_inverse = src.matrix_parent_inverse.copy()
    dest.matrix_basis = src.matrix_basis.copy()
    # matrix_world is stale until the view layer updates; nearest-weight
    # copy is in world space and will explode a newly parented loft.
    bpy.context.view_layer.update()
    # Solidify (and any new cut verts) need weights; fill from the body.
    _copy_weights_nearest(dest, src)
    # Drop any leftover armature first
    for mod in list(dest.modifiers):
        dest.modifiers.remove(mod)
    arm_mod = dest.modifiers.new("Armature", "ARMATURE")
    arm_mod.object = arm
    arm_mod.use_vertex_groups = True
    return True


def _duplicate_body(name, collection, src):
    me = src.data.copy()
    me.name = name + "Mesh"
    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)
    obj.matrix_world = src.matrix_world.copy()
    return obj


def _vert_group_names(obj, index):
    return {vg.index: _clean(vg.name) for vg in obj.vertex_groups}


def build_trunks_v4(name, collection, meshes, arm, colors, clearance=0.016):
    """Classic satin boxing trunks, closed shell, bound to the armature.

    Built from the bind-pose body (region cut + inflate) so topology already
    has two legs and weights, then bisected at a high navel waistband and
    mid-thigh hems. Thin black trim on the waist, hems and a side stripe.
    """
    prev = _rest_context(arm)
    try:
        srcs = _primary_body_meshes(meshes)
        if not srcs:
            raise RuntimeError("trunks: no body mesh")
        src = srcs[0]
        hips_bn = find_bone(arm, "mixamorig:Hips", "Hips", "hips")
        spine_bn = find_bone(arm, "mixamorig:Spine", "Spine", "spine")
        upleg_l = find_bone(arm, "mixamorig:LeftUpLeg", "LeftUpLeg")
        leg_l = find_bone(arm, "mixamorig:LeftLeg", "LeftLeg")
        hips_w = bone_world_matrix(arm, hips_bn).to_translation() if hips_bn else Vector((0, 0, 0.9))
        spine_w = bone_world_matrix(arm, spine_bn).to_translation() if spine_bn else hips_w + Vector((0, 0, 0.08))
        navel_z = hips_w.z * 0.25 + spine_w.z * 0.75
        waist_top = navel_z + 0.018
        if upleg_l and leg_l:
            a = bone_world_matrix(arm, upleg_l).to_translation()
            b = bone_world_matrix(arm, leg_l).to_translation()
            hem_z = (a.z + b.z) * 0.50
        else:
            hem_z = hips_w.z - 0.22

        obj = _duplicate_body(name, collection, src)
        gnames = _vert_group_names(obj, 0)
        mw = obj.matrix_world
        keep = [False] * len(obj.data.vertices)
        deny = ("hand", "forearm", "upperarm", "lowerarm", "head", "neck", "foot", "toe", "shoulder")
        # Also reject plain "arm" but not "upleg".
        for i, v in enumerate(obj.data.vertices):
            wco = mw @ v.co
            if wco.z < hem_z - 0.05 or wco.z > waist_top + 0.06:
                continue
            names = [gnames.get(g.group, "") for g in v.groups if g.weight >= 0.15]
            if any(
                (any(k in n for k in deny) or (n.endswith("arm") or n == "arm" or "leftarm" in n or "rightarm" in n))
                and "upleg" not in n and "thigh" not in n
                for n in names
            ):
                continue
            keep[i] = True
        _delete_verts(obj, keep)
        _bisect_world_z(obj, waist_top, keep_below=True)
        _bisect_world_z(obj, hem_z, keep_below=False)
        _relax_boundaries(obj, iterations=10, factor=0.5)

        def infl(wco):
            # Snug waist, a little ease at the seat, looser legs.
            t = 0.0
            if wco.z >= waist_top - 0.065:
                return clearance * 0.70  # waistband slightly proud but fitted
            if wco.z >= hips_w.z - 0.03:
                return clearance * 1.15  # seat: follow the glute
            # legs
            return clearance * 2.05

        _world_inflate(obj, infl)
        # Fatten the waistband a little (wide elastic).
        mw = obj.matrix_world
        imw = mw.inverted()
        for v in obj.data.vertices:
            wco = mw @ v.co
            if wco.z >= waist_top - 0.062:
                # extra 4 mm out, slightly up at the top edge
                radial = Vector((wco.x, wco.y, 0.0))
                mid = Vector((0.0, 0.0, wco.z))  # will use hips xy
                # push horizontally
                xy = Vector((wco.x, wco.y, 0.0))
                if xy.length > 1e-6:
                    wco = wco + xy.normalized() * 0.006
                if wco.z > waist_top - 0.012:
                    wco = wco + Vector((0, 0, 0.004))
                v.co = imw @ wco
        obj.data.update()

        sol = obj.modifiers.new("Thick", "SOLIDIFY")
        sol.thickness = 0.0055
        sol.offset = 1.0
        try:
            sol.use_even_offset = True
            sol.use_quality_normals = True
            sol.use_rim = True
        except Exception:
            pass
        _apply_modifier(obj, sol)
        _smooth_mesh(obj, iterations=2, factor=0.18)
        _shade_smooth(obj)
        _bind_like_body(obj, src, arm)

        zs_local = [v.co.z for v in obj.data.vertices]
        zmax, zmin = max(zs_local), min(zs_local)
        zspan = max(1e-6, zmax - zmin)
        xs = [abs(v.co.x) for v in obj.data.vertices]
        xside = sorted(xs)[int(0.88 * len(xs))] if xs else 0.1

        def trim_pred(x, y, z, _zmax=zmax, _zmin=zmin, _zspan=zspan, _xside=xside):
            ttop = (_zmax - z) / _zspan
            tbot = (z - _zmin) / _zspan
            waist_edge = ttop <= 0.04 or (0.14 <= ttop <= 0.19)
            hem = tbot <= 0.07
            stripe = abs(x) >= _xside * 0.96 and 0.10 < tbot and ttop > 0.20
            return waist_edge or hem or stripe

        satin = make_satin_material(name + "Satin", colors["beanie"], roughness=0.20)
        trim = make_satin_material(name + "Trim", TRIM_V4_BLACK, roughness=0.28)
        _assign_region_mats(obj, satin, trim, trim_pred)
        return obj, {
            "version": "v4",
            "waist_top_m": waist_top,
            "hem_z_m": hem_z,
            "navel_z_m": navel_z,
            "clearance_m": clearance,
            "source": "body-shell",
        }
    finally:
        _restore_pose(arm, prev)


def build_tank_v4(name, collection, meshes, arm, colors, clearance=0.008):
    """Fitted sleeveless athletic tank. Charcoal satin, closed shell.

    Derived from the bind-pose torso so it follows the ribcage. Armholes
    sit at the armpit, below the deltoid mass; only a thin medial strap
    (neck-adjacent torso verts) crosses the shoulder so glow stays visible.
    """
    prev = _rest_context(arm)
    try:
        srcs = _primary_body_meshes(meshes)
        if not srcs:
            raise RuntimeError("tank: no body mesh")
        src = srcs[0]
        hips_bn = find_bone(arm, "mixamorig:Hips", "Hips", "hips")
        spine_bn = find_bone(arm, "mixamorig:Spine", "Spine", "spine")
        sh_l = find_bone(arm, "mixamorig:LeftShoulder", "LeftShoulder")
        sh_r = find_bone(arm, "mixamorig:RightShoulder", "RightShoulder")
        neck_bn = find_bone(arm, "mixamorig:Neck", "Neck", "neck")
        hips_w = bone_world_matrix(arm, hips_bn).to_translation() if hips_bn else Vector((0, 0, 0.9))
        spine_w = bone_world_matrix(arm, spine_bn).to_translation() if spine_bn else hips_w
        neck_w = bone_world_matrix(arm, neck_bn).to_translation() if neck_bn else Vector((0, 0, 1.45))
        sh_lw = bone_world_matrix(arm, sh_l).to_translation() if sh_l else Vector((0.12, 0, 1.3))
        sh_rw = bone_world_matrix(arm, sh_r).to_translation() if sh_r else Vector((-0.12, 0, 1.3))
        armpit_z = min(sh_lw.z, sh_rw.z) - 0.050
        navel_z = hips_w.z * 0.25 + spine_w.z * 0.75
        hem_z = navel_z - 0.050
        # Strap band: a tight corridor next to the neck, not the deltoid.
        mid_x = (sh_lw.x + sh_rw.x) * 0.5
        strap_half = 0.042  # metres from the neck/midline, world

        obj = _duplicate_body(name, collection, src)
        gnames = _vert_group_names(obj, 0)
        mw = obj.matrix_world
        keep = [False] * len(obj.data.vertices)
        torso_ok = ("spine", "chest", "abdomen", "torso", "hip", "pelvis")
        deny = ("arm", "fore", "hand", "head", "leg", "foot", "upleg", "thigh", "toe")
        for i, v in enumerate(obj.data.vertices):
            wco = mw @ v.co
            if wco.z < hem_z - 0.03 or wco.z > neck_w.z + 0.01:
                continue
            names = [gnames.get(g.group, "") for g in v.groups if g.weight >= 0.18]
            if any(any(k in n for k in deny) for n in names):
                continue
            is_shoulder = any("shoulder" in n or "clav" in n for n in names)
            medial = abs(wco.x - mid_x) <= strap_half
            if is_shoulder:
                # only the medial strip, and not below the armpit (that's side fabric we don't want over the delt)
                if medial and wco.z >= armpit_z - 0.01:
                    keep[i] = True
                continue
            # torso verts
            if wco.z <= armpit_z + 0.01:
                # below the armhole: full torso width (rib cage)
                if names and not any(any(k in n for k in torso_ok) for n in names):
                    # still allow unlabelled chest verts in this band
                    if wco.z > navel_z - 0.05:
                        keep[i] = True
                    continue
                keep[i] = True
            else:
                # above the armhole: only the chest/neck panel and medial straps
                # pinch sides — drop anything heading out toward the deltoid
                if medial or abs(wco.x - mid_x) <= strap_half * 1.15:
                    keep[i] = True
        _delete_verts(obj, keep)
        _bisect_world_z(obj, hem_z, keep_below=False)
        _bisect_world_z(obj, neck_w.z - 0.015, keep_below=True)
        _relax_boundaries(obj, iterations=10, factor=0.5)

        def infl(wco):
            # slightly more ease on the back for scapula motion
            return clearance * (1.15 if wco.y > 0.0 else 1.0)

        _world_inflate(obj, infl)
        sol = obj.modifiers.new("Thick", "SOLIDIFY")
        sol.thickness = 0.004
        sol.offset = 1.0
        try:
            sol.use_even_offset = True
            sol.use_quality_normals = True
            sol.use_rim = True
        except Exception:
            pass
        _apply_modifier(obj, sol)
        _smooth_mesh(obj, iterations=2, factor=0.16)
        _shade_smooth(obj)
        _bind_like_body(obj, src, arm)
        satin = make_satin_material(name + "Satin", colors["beanie"], roughness=0.20)
        _assign_mat(obj, satin)
        return obj, {
            "version": "v4",
            "hem_z_m": hem_z,
            "armpit_z_m": armpit_z,
            "clearance_m": clearance,
            "racerback": True,
            "source": "body-shell",
        }
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
                if weights_ok is not None and (i >= len(weights_ok) or weights_ok[i] < 0.50):
                    continue
                wp = mw @ v.co
                if _inside_bvh(bvh, wp, margin=margin - 0.004):
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



def _torso_landmarks(arm):
    """World-metre landmarks at the current pose (call while REST)."""
    hips_bn = find_bone(arm, "mixamorig:Hips", "Hips", "hips")
    spine_bn = find_bone(arm, "mixamorig:Spine", "Spine", "spine")
    spine1_bn = find_bone(arm, "mixamorig:Spine1", "Spine1", "spine1")
    sh_l = find_bone(arm, "mixamorig:LeftShoulder", "LeftShoulder")
    sh_r = find_bone(arm, "mixamorig:RightShoulder", "RightShoulder")
    neck_bn = find_bone(arm, "mixamorig:Neck", "Neck", "neck")
    upleg_l = find_bone(arm, "mixamorig:LeftUpLeg", "LeftUpLeg")
    leg_l = find_bone(arm, "mixamorig:LeftLeg", "LeftLeg")
    hips_w = bone_world_matrix(arm, hips_bn).to_translation() if hips_bn else Vector((0, 0, 0.9))
    spine_w = bone_world_matrix(arm, spine_bn).to_translation() if spine_bn else hips_w + Vector((0, 0, 0.08))
    spine1_w = bone_world_matrix(arm, spine1_bn).to_translation() if spine1_bn else spine_w + Vector((0, 0, 0.08))
    neck_w = bone_world_matrix(arm, neck_bn).to_translation() if neck_bn else Vector((0, 0, 1.45))
    sh_lw = bone_world_matrix(arm, sh_l).to_translation() if sh_l else Vector((0.12, 0, 1.3))
    sh_rw = bone_world_matrix(arm, sh_r).to_translation() if sh_r else Vector((-0.12, 0, 1.3))
    navel_z = hips_w.z * 0.25 + spine_w.z * 0.75
    waist_top = navel_z + 0.018
    if upleg_l and leg_l:
        a = bone_world_matrix(arm, upleg_l).to_translation()
        b = bone_world_matrix(arm, leg_l).to_translation()
        mid_thigh_z = (a.z + b.z) * 0.50
    else:
        mid_thigh_z = hips_w.z - 0.22
    return {
        "hips_w": hips_w, "spine_w": spine_w, "spine1_w": spine1_w,
        "neck_w": neck_w, "sh_lw": sh_lw, "sh_rw": sh_rw,
        "navel_z": navel_z, "waist_top": waist_top,
        "band_h": 0.055, "mid_thigh_z": mid_thigh_z,
        "crotch_z": hips_w.z - 0.09,
        "armpit_z": min(sh_lw.z, sh_rw.z) - 0.052,
    }


def _resample_ring(ring, n):
    if len(ring) == n:
        return list(ring)
    pts = list(ring) + [ring[0]]
    segl = [(pts[i + 1] - pts[i]).length for i in range(len(ring))]
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


def _cap_ring_inset(bm, row, inward=0.75, drop_z=0.0):
    """Close a BMVert loop with a slightly inset cap so the hem stays clean."""
    if not row:
        return
    c = sum((v.co for v in row), Vector()) / len(row)
    inner = []
    for v in row:
        p = c.lerp(v.co, inward)
        if drop_z:
            p = Vector((p.x, p.y, p.z + drop_z))
        inner.append(bm.verts.new(p))
    bm.verts.ensure_lookup_table()
    n = min(len(row), len(inner))
    for i in range(n):
        j = (i + 1) % n
        try:
            bm.faces.new((row[i], row[j], inner[j], inner[i]))
        except Exception:
            pass
    try:
        bm.faces.new(list(reversed(inner)))
    except Exception:
        try:
            bm.faces.new(inner)
        except Exception:
            pass


def _circularize_boundaries(obj, iterations=2, min_world_z=None):
    """Even, rounded boundary loops (hem / armhole / neck) in object space.

    If min_world_z is set, loops whose centroid sits below that world height
    (typically the hem) are left alone so circularize cannot open a crotch
    hole or ovalize a waist-shaped hem.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    visited = set()
    loops = []
    bverts = [v for v in bm.verts if v.is_boundary]
    for start in bverts:
        if start.index in visited:
            continue
        loop = []
        cur, prev = start, None
        while cur is not None and cur.index not in visited:
            visited.add(cur.index)
            loop.append(cur)
            nxts = [e.other_vert(cur) for e in cur.link_edges if e.other_vert(cur).is_boundary]
            nxt = None
            for cand in nxts:
                if prev is None or cand != prev:
                    nxt = cand
                    if cand.index not in visited:
                        break
            prev, cur = cur, (None if nxt is None or nxt.index in visited else nxt)
        if len(loop) >= 8:
            loops.append(loop)
    mw = obj.matrix_world
    for loop in loops:
        c = sum((v.co for v in loop), Vector()) / len(loop)
        if min_world_z is not None and (mw @ c).z < min_world_z:
            continue
        # fit a plane normal from Newell's method
        nrm = Vector((0, 0, 0))
        for i, v in enumerate(loop):
            a, b = v.co, loop[(i + 1) % len(loop)].co
            nrm += (a - c).cross(b - c)
        if nrm.length < 1e-8:
            nrm = Vector((0, 0, 1))
        nrm.normalize()
        # mean radius in plane
        rs = []
        for v in loop:
            d = v.co - c
            d = d - nrm * d.dot(nrm)
            rs.append(d.length)
        r = sum(rs) / max(1, len(rs))
        # even angles around an in-plane axis
        ref = (loop[0].co - c)
        ref = ref - nrm * ref.dot(nrm)
        if ref.length < 1e-8:
            ref = nrm.cross(Vector((0, 0, 1)))
            if ref.length < 1e-8:
                ref = Vector((1, 0, 0))
        ref.normalize()
        bit = nrm.cross(ref).normalized()
        # keep existing angular order
        def ang(v):
            d = v.co - c
            d = d - nrm * d.dot(nrm)
            return math.atan2(d.dot(bit), d.dot(ref))
        loop_sorted = loop  # already walking order
        for i, v in enumerate(loop_sorted):
            a = ang(v)
            # blend toward the even circle to clean jags but keep oval-ish shape
            target = c + ref * (r * math.cos(a)) + bit * (r * math.sin(a))
            # mix with current radial length to keep armholes slightly oval
            cur_r = (v.co - c - nrm * (v.co - c).dot(nrm)).length
            use_r = 0.45 * r + 0.55 * cur_r
            target = c + (target - c).normalized() * use_r if (target - c).length > 1e-8 else target
            v.co = v.co.lerp(target, 0.85)
    for _ in range(iterations):
        bm.verts.ensure_lookup_table()
        news = {}
        for v in bm.verts:
            if not v.is_boundary:
                continue
            pts = [e.other_vert(v).co for e in v.link_edges if e.other_vert(v).is_boundary]
            if len(pts) >= 2:
                mid = sum(pts, Vector()) / len(pts)
                news[v] = v.co.lerp(mid, 0.35)
        for v, co in news.items():
            v.co = co
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _glove_v5_spec(head_width=None):
    hw = float(head_width) if head_width and head_width > 4.0 else HEAD_WIDTH_REF_CM
    fw = 0.80 * hw
    s = fw / GLOVE_V5["front_width"]
    return {
        "front_width": fw,
        "front_thick": GLOVE_V5["front_thick"] * s,
        "body_len": GLOVE_V5["body_len"] * s,
        "cuff_len": GLOVE_V5["cuff_len"],
        "cuff_r": GLOVE_V5["cuff_r"] * s,
        "thumb_len": GLOVE_V5["thumb_len"] * s,
        "thumb_rx": GLOVE_V5["thumb_rx"] * s,
        "thumb_rz": GLOVE_V5["thumb_rz"] * s,
        "head_width": hw,
        "scale": s,
    }


def build_glove_mesh_v5(name, collection, hand_env, wrist_radius=None, head_width=None):
    """v5 glove: kidney-hugging thumb, longer cuff, same glossy black."""
    wr = float(wrist_radius or 3.35)
    spec = _glove_v5_spec(head_width)
    fw, ft, bl = spec["front_width"], spec["front_thick"], spec["body_len"]
    cuff_len = spec["cuff_len"]
    # Tighter cuff so it sits on the wrist instead of hovering off it.
    cuff_r = max(spec["cuff_r"], wr + 1.20)
    rx, rz = fw * 0.50, ft * 0.50
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
        if t < 0.06:
            y = y_open + 0.40 * (1.0 - t / 0.06)
            lip_s = 0.62 + 0.38 * (t / 0.06)
            sx, sz = cuff_r * lip_s, cuff_r * 0.97 * lip_s
        else:
            tt = (t - 0.06) / 0.94
            y = y_open + y_span * tt
            sx, sz = _v4_rxz(tt, rx, rz, cuff_r)
        row = []
        for j in range(n_seg):
            a = 2.0 * math.pi * j / n_seg
            pal = 0.90 if (math.cos(a) < 0.0 and t > 0.35) else 1.0
            x = sx * pal * math.cos(a)
            z = sz * math.sin(a) + 0.06
            row.append(bm.verts.new((x, y, z)))
        rows.append(row)
    bm.verts.ensure_lookup_table()
    for i in range(n_rings):
        for j in range(n_seg):
            k = (j + 1) % n_seg
            bm.faces.new((rows[i][j], rows[i][k], rows[i + 1][k], rows[i + 1][j]))
    try:
        bm.faces.new(list(reversed(rows[0])))
        bm.faces.new(rows[-1])
    except Exception:
        pass
    bm.to_mesh(mesh)
    bm.free()

    # Kidney thumb: flattened lobe that hugs the +Z side of the bulb.
    # Centre is buried inside the body so remesh fuses it as a swell, not a knob.
    body_mid_y = 0.20 * y_open + 0.80 * y_tip - spec["thumb_len"] * 0.28
    # Sit the lobe centre just inside the +Z skin so a rounded kidney bulges out.
    thumb_c = Vector((0.15, body_mid_y, rz * 0.62))
    thumb = _ellipsoid(
        name + "_thumb",
        (spec["thumb_rx"] * 1.15, spec["thumb_len"] * 0.48, spec["thumb_rz"] * 0.72),
        3, collection, location=(0.0, 0.0, 0.0),
    )
    ang = math.radians(28.0)
    ca, sa = math.cos(ang), math.sin(ang)
    for v in thumb.data.vertices:
        y, z = v.co.y, v.co.z
        v.co.y = y * ca - z * sa
        v.co.z = y * sa + z * ca
        # Bend the lobe around the bulb (kidney wrap).
        wrap = max(0.0, v.co.z)
        v.co.x -= wrap * wrap * 0.08
        v.co += thumb_c
    thumb.data.update()
    # A second, shorter blend cap at the crotch so the join is a swell, not a seam.
    crotch = _ellipsoid(
        name + "_crotch",
        (spec["thumb_rx"] * 1.05, spec["thumb_len"] * 0.38, spec["thumb_rz"] * 0.85),
        2, collection, location=(0.0, 0.0, 0.0),
    )
    crotch_c = Vector((0.0, body_mid_y - spec["thumb_len"] * 0.16, rz * 0.02))
    for v in crotch.data.vertices:
        v.co += crotch_c
    crotch.data.update()

    _join_objects(obj, thumb, crotch)
    _voxel_remesh(obj, voxel=0.18)
    _smooth_mesh(obj, iterations=7, factor=0.30)
    _add_stitch_grooves_v4(obj, y_open, y_tip, rx, rz, cuff_r, thumb_c, spec["thumb_len"])
    _smooth_mesh(obj, iterations=2, factor=0.16)
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
        "version": "v5",
        "mask_plane_y": y_open + 0.45,
        "cuff_rx": cuff_r + 0.25,
        "cuff_rz": cuff_r + 0.25,
        "cuff_y_far": y_open,
        "cuff_y_near": y_open + cuff_len * 0.88,
        "body_len": bl,
        "front_width": fw,
        "front_thick": ft,
        "head_width": spec["head_width"],
        "fist_to_head": fw / max(1e-6, spec["head_width"]),
    }



def _flatten_boundary_world_z(obj, z, z_tol=0.05):
    """Snap near-height boundary verts onto a world-Z plane (clean hem / waist)."""
    mw = obj.matrix_world
    imw = mw.inverted()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        if not v.is_boundary:
            continue
        w = mw @ v.co
        if abs(w.z - z) <= z_tol:
            w = Vector((w.x, w.y, z))
            v.co = imw @ w
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _fill_interior_holes(obj, rim_zs, z_tol=0.04):
    """Fill boundary loops that are not the waist / hem rims."""
    mw = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    visited = set()
    for start in [v for v in bm.verts if v.is_boundary]:
        if start.index in visited:
            continue
        loop = []
        cur, prev = start, None
        while cur is not None and cur.index not in visited:
            visited.add(cur.index)
            loop.append(cur)
            nxts = [e.other_vert(cur) for e in cur.link_edges if e.other_vert(cur).is_boundary]
            nxt = None
            for cand in nxts:
                if prev is None or cand != prev:
                    nxt = cand
                    if cand.index not in visited:
                        break
            prev, cur = cur, (None if nxt is None or nxt.index in visited else nxt)
        if len(loop) < 3:
            continue
        c = sum((v.co for v in loop), Vector()) / len(loop)
        wz = (mw @ c).z
        if any(abs(wz - z) <= z_tol for z in rim_zs):
            continue
        try:
            bm.faces.new(loop)
        except Exception:
            try:
                bm.faces.new(list(reversed(loop)))
            except Exception:
                pass
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _paint_torso_chain(obj, arm):
    """Replace all weights with a hips–spine chain blend by world height.

    Nearest-body weights and leftover arm groups shred a lofted tank on jab.
    A two-bone height blend on the torso chain keeps the midriff intact.
    """
    chain = []
    for cands in (("mixamorig:Hips", "Hips", "hips"),
                  ("mixamorig:Spine", "Spine", "spine"),
                  ("mixamorig:Spine1", "Spine1", "spine1"),
                  ("mixamorig:Spine2", "Spine2", "spine2")):
        bn = find_bone(arm, *cands)
        if not bn:
            continue
        if obj.vertex_groups.get(bn) is None:
            obj.vertex_groups.new(name=bn)
        z = bone_world_matrix(arm, bn).to_translation().z
        chain.append((z, bn))
    if len(chain) < 2:
        return False
    chain.sort()
    keep = {n for _, n in chain}
    mw = obj.matrix_world
    for v in obj.data.vertices:
        for vg in obj.vertex_groups:
            try:
                vg.remove([v.index])
            except Exception:
                pass
        z = (mw @ v.co).z
        if z <= chain[0][0]:
            obj.vertex_groups[chain[0][1]].add([v.index], 1.0, "REPLACE")
            continue
        if z >= chain[-1][0]:
            obj.vertex_groups[chain[-1][1]].add([v.index], 1.0, "REPLACE")
            continue
        for i in range(len(chain) - 1):
            z0, n0 = chain[i]
            z1, n1 = chain[i + 1]
            if z0 <= z <= z1:
                t = (z - z0) / max(1e-6, z1 - z0)
                obj.vertex_groups[n0].add([v.index], 1.0 - t, "REPLACE")
                obj.vertex_groups[n1].add([v.index], t, "REPLACE")
                break
    return True


def _strip_limb_weights(obj):
    """Keep only torso groups on a tank so a jab cannot shred the midriff."""
    drop_keys = ("arm", "hand", "fore", "shoulder", "clav", "neck", "head",
                 "leg", "upleg", "thigh", "foot", "toe")
    keep_keys = ("spine", "chest", "abdomen", "torso", "hip", "pelvis", "belly")
    drop, keep = [], []
    for vg in obj.vertex_groups:
        n = vg.name.lower()
        if any(k in n for k in drop_keys) and not any(k in n for k in keep_keys):
            drop.append(vg)
        elif any(k in n for k in keep_keys):
            keep.append(vg)
    for v in obj.data.vertices:
        for vg in drop:
            try:
                vg.remove([v.index])
            except Exception:
                pass
    return True


def _blend_weights_across_gap(dest, src, z_lo, z_hi):
    """Lerp vertex groups across a vertical hole in the source body.

    Lofted midriff/waist verts otherwise snap to the nearest chest *or* abdomen
    vert and tear when those bones move apart in a fighting pose.
    """
    from mathutils import kdtree as _kd
    sw, dw = src.matrix_world, dest.matrix_world
    below, above = [], []
    for i, v in enumerate(src.data.vertices):
        z = (sw @ v.co).z
        if z <= z_lo + 0.018:
            below.append(i)
        if z >= z_hi - 0.018:
            above.append(i)
    if not below or not above:
        return False

    def _kd_of(idxs):
        tree = _kd.KDTree(len(idxs))
        for i in idxs:
            tree.insert(sw @ src.data.vertices[i].co, i)
        tree.balance()
        return tree

    kd_lo, kd_hi = _kd_of(below), _kd_of(above)
    for vg in src.vertex_groups:
        if dest.vertex_groups.get(vg.name) is None:
            dest.vertex_groups.new(name=vg.name)
    name_of = {vg.index: vg.name for vg in src.vertex_groups}
    zspan = max(1e-6, z_hi - z_lo)
    for v in dest.data.vertices:
        wco = dw @ v.co
        if wco.z < z_lo - 0.02 or wco.z > z_hi + 0.02:
            continue
        t = max(0.0, min(1.0, (wco.z - z_lo) / zspan))
        _p, i_lo, _d = kd_lo.find(wco)
        _p, i_hi, _d = kd_hi.find(wco)
        for vg in dest.vertex_groups:
            try:
                vg.remove([v.index])
            except Exception:
                pass
        weights = {}

        def acc(idx, scale):
            if idx is None:
                return
            for g in src.data.vertices[idx].groups:
                n = name_of.get(g.group)
                if n:
                    weights[n] = weights.get(n, 0.0) + g.weight * scale

        acc(i_lo, 1.0 - t)
        acc(i_hi, t)
        tot = sum(weights.values()) or 1.0
        for n, wgt in weights.items():
            dest.vertex_groups[n].add([v.index], wgt / tot, "REPLACE")
    return True


def _boundary_loops(obj):
    """Return boundary loops as lists of BMVert indices (mesh domain)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    visited = set()
    loops = []
    for start in [v for v in bm.verts if v.is_boundary]:
        if start.index in visited:
            continue
        loop = []
        cur, prev = start, None
        while cur is not None and cur.index not in visited:
            visited.add(cur.index)
            loop.append(cur.index)
            nxts = [e.other_vert(cur) for e in cur.link_edges if e.other_vert(cur).is_boundary]
            nxt = None
            for cand in nxts:
                if prev is None or cand != prev:
                    nxt = cand
                    if cand.index not in visited:
                        break
            prev, cur = cur, (None if nxt is None or nxt.index in visited else nxt)
        if len(loop) >= 6:
            loops.append(loop)
    bm.free()
    return loops


def _loft_boundary_to_world_z(obj, target_z, which="lowest", n_rings=5,
                              end_scale=1.0, mid_scale=1.03):
    """Grow the lowest (or highest) boundary to a new world-Z ring, filling a gap.

    Mixamo's Beta_Surface has a ~13 cm midriff hole (no verts/faces between
    ~1.10 m and ~1.23 m), so a body-shell tank cannot reach the waistband
    without this loft.
    """
    mw = obj.matrix_world
    imw = mw.inverted()
    loops = _boundary_loops(obj)
    if not loops:
        return False
    scored = []
    for loop in loops:
        pts = [mw @ obj.data.vertices[i].co for i in loop]
        cz = sum(p.z for p in pts) / len(pts)
        scored.append((cz, loop, pts))
    if which == "lowest":
        cz, loop, pts = min(scored, key=lambda t: t[0])
    else:
        cz, loop, pts = max(scored, key=lambda t: t[0])
    if abs(cz - target_z) < 0.004:
        return False
    c = sum(pts, Vector()) / len(pts)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    row = [bm.verts[i] for i in loop]
    n = len(row)
    for k in range(1, n_rings + 1):
        t = k / n_rings
        z = cz + (target_z - cz) * t
        if t < 0.5:
            sc = 1.0 + (mid_scale - 1.0) * (t / 0.5)
        else:
            sc = mid_scale + (end_scale - mid_scale) * ((t - 0.5) / 0.5)
        new_row = []
        for p in pts:
            w = Vector((c.x + (p.x - c.x) * sc,
                        c.y + (p.y - c.y) * sc,
                        z))
            new_row.append(bm.verts.new(imw @ w))
        bm.verts.ensure_lookup_table()
        for i in range(n):
            j = (i + 1) % n
            try:
                bm.faces.new((row[i], row[j], new_row[j], new_row[i]))
            except Exception:
                pass
        row = new_row
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return True


def build_trunks_v5(name, collection, meshes, arm, colors):
    """Classic boxing trunks: snug navel band, follows seat, gentle mid-thigh flare.

    Body-shell (same bind space as the character) so the armature deform stays
    stable. Inflate is tight at the waist, follows the glutes, and only flares
    in the lower third to ~3.5 cm at a clean mid-thigh hem. Deep burgundy
    satin with thin charcoal/black trim; tank stays beanie charcoal.
    """
    prev = _rest_context(arm)
    try:
        srcs = _primary_body_meshes(meshes)
        if not srcs:
            raise RuntimeError("trunks v5: no body mesh")
        src = srcs[0]
        lm = _torso_landmarks(arm)
        waist_top = lm["waist_top"]
        band_h = lm["band_h"]
        hips_z = lm["hips_w"].z
        crotch_z = lm["crotch_z"]
        hem_z = lm["mid_thigh_z"]

        obj = _duplicate_body(name, collection, src)
        gnames = _vert_group_names(obj, 0)
        mw = obj.matrix_world
        keep = [False] * len(obj.data.vertices)
        deny = ("hand", "forearm", "upperarm", "lowerarm", "head", "neck", "foot", "toe", "shoulder")
        for i, v in enumerate(obj.data.vertices):
            wco = mw @ v.co
            if wco.z < hem_z - 0.05 or wco.z > waist_top + 0.06:
                continue
            names = [gnames.get(g.group, "") for g in v.groups if g.weight >= 0.15]
            if any(
                (any(k in n for k in deny) or (n.endswith("arm") or n == "arm"
                                               or "leftarm" in n or "rightarm" in n))
                and "upleg" not in n and "thigh" not in n
                for n in names
            ):
                continue
            keep[i] = True
        _delete_verts(obj, keep)
        _bisect_world_z(obj, waist_top, keep_below=True)
        _bisect_world_z(obj, hem_z, keep_below=False)
        _flatten_boundary_world_z(obj, waist_top, z_tol=0.06)
        _flatten_boundary_world_z(obj, hem_z, z_tol=0.06)
        _relax_boundaries(obj, iterations=8, factor=0.45)
        # Do not circularize trunks — a crotch seam would be blown into a hole.
        _flatten_boundary_world_z(obj, waist_top, z_tol=0.04)
        _flatten_boundary_world_z(obj, hem_z, z_tol=0.04)
        _fill_interior_holes(obj, (waist_top, hem_z), z_tol=0.05)
        # The Mixamo abdomen island stops ~4 cm short of the navel; loft a
        # snug waistband up to waist_top so the band actually sits at the navel.
        _loft_boundary_to_world_z(obj, waist_top, which="highest", n_rings=4,
                                  end_scale=0.98, mid_scale=0.99)
        _flatten_boundary_world_z(obj, waist_top, z_tol=0.02)
        _relax_boundaries(obj, iterations=4, factor=0.30)

        def infl(wco):
            z = wco.z
            if z >= waist_top - band_h:
                off = 0.006
            elif z >= hips_z:
                u = (waist_top - band_h - z) / max(1e-6, (waist_top - band_h - hips_z))
                off = 0.006 + 0.004 * max(0.0, min(1.0, u))
            elif z >= crotch_z:
                u = (hips_z - z) / max(1e-6, (hips_z - crotch_z))
                off = 0.010 + 0.003 * max(0.0, min(1.0, u))
            else:
                # Flare late: stay close through the upper thigh, open ~3.5 cm at the hem.
                t = (crotch_z - z) / max(1e-6, (crotch_z - hem_z))
                t = max(0.0, min(1.0, t))
                t = t * t
                off = 0.013 + 0.022 * t
            # Extra groin ease so a jab-stance hip open doesn't tear the crotch.
            if z <= hips_z + 0.02 and abs(wco.x) < 0.07:
                off += 0.014 * (1.0 - abs(wco.x) / 0.07)
            return off

        _world_inflate(obj, infl)
        # Subtle elastic lip — a couple of millimetres, not a v4 bulge.
        mw = obj.matrix_world
        imw = mw.inverted()
        for v in obj.data.vertices:
            wco = mw @ v.co
            if wco.z >= waist_top - band_h:
                xy = Vector((wco.x, wco.y, 0.0))
                if xy.length > 1e-6:
                    wco = wco + xy.normalized() * 0.002
                if wco.z > waist_top - 0.010:
                    wco = wco + Vector((0.0, 0.0, 0.0015))
                v.co = imw @ wco
        obj.data.update()
        _flatten_boundary_world_z(obj, waist_top + 0.0015, z_tol=0.02)

        sol = obj.modifiers.new("Thick", "SOLIDIFY")
        sol.thickness = 0.0055
        sol.offset = 1.0
        try:
            sol.use_even_offset = True
            sol.use_quality_normals = True
            sol.use_rim = True
        except Exception:
            pass
        _apply_modifier(obj, sol)
        _smooth_mesh(obj, iterations=1, factor=0.10)
        _shade_smooth(obj)
        _bind_like_body(obj, src, arm)
        _blend_weights_across_gap(obj, src, z_lo=1.095, z_hi=waist_top + 0.01)

        # Trim in world space (local Z is Mixamo forward, not height).
        mw = obj.matrix_world
        world = [mw @ v.co for v in obj.data.vertices]
        zs = [w.z for w in world]
        xs = [abs(w.x) for w in world]
        zmax, zmin = max(zs), min(zs)
        zspan = max(1e-6, zmax - zmin)
        xside = sorted(xs)[int(0.992 * (len(xs) - 1))] if xs else 0.1

        def trim_pred(x, y, z, _mw=mw, _zmax=zmax, _zmin=zmin, _zspan=zspan, _xside=xside):
            w = _mw @ Vector((x, y, z))
            ttop = (_zmax - w.z) / _zspan
            tbot = (w.z - _zmin) / _zspan
            waist_edge = ttop <= 0.030 or (0.125 <= ttop <= 0.155)
            hem = tbot <= 0.040
            return waist_edge or hem

        satin = make_satin_material(name + "Satin", TRUNKS_V5_BURGUNDY, roughness=0.28)
        trim = make_satin_material(name + "Trim", TRIM_V5_CHARCOAL, roughness=0.34)
        _assign_region_mats(obj, satin, trim, trim_pred)
        return obj, {
            "version": "v5",
            "waist_top_m": waist_top,
            "hem_z_m": hem_z,
            "navel_z_m": lm["navel_z"],
            "hem_clearance_m": 0.035,
            "color": list(TRUNKS_V5_BURGUNDY),
            "source": "body-shell-snug",
        }
    finally:
        _restore_pose(arm, prev)


def _lerp_world_rings(r0, z0, r1, z1, z, n=36):
    """Interpolate two same-count world rings onto height z."""
    if r0 is None and r1 is None:
        return None
    if r0 is None:
        r0, z0 = r1, z1
    if r1 is None:
        r1, z1 = r0, z0
    a = _resample_ring(r0, n)
    b = _resample_ring(r1, n)
    # Align b so b[0] is closest to a[0]
    off = min(range(n), key=lambda i: (b[i] - a[0]).length_squared)
    b = b[off:] + b[:off]
    if n > 1 and (b[1] - a[1]).length > (b[-1] - a[1]).length:
        b = [b[0]] + list(reversed(b[1:]))
    t = 0.0 if abs(z1 - z0) < 1e-8 else (z - z0) / (z1 - z0)
    t = max(0.0, min(1.0, t))
    out = []
    for p, q in zip(a, b):
        w = p.lerp(q, t)
        w.z = z
        out.append(w)
    return out


def build_tank_v5(name, collection, meshes, arm, colors, clearance=0.007):
    """Fitted sleeveless tank whose hem covers the trunks waistband top.

    Built as a clean loft in the body's local space (not a body-shell cut).
    Mixamo's Beta_Surface has a 13 cm midriff hole, so rings in that gap are
    interpolated between the abdomen and chest silhouettes; vertex weights
    in the gap are blended so the midriff does not tear on a jab.
    """
    prev = _rest_context(arm)
    try:
        srcs = _primary_body_meshes(meshes)
        if not srcs:
            raise RuntimeError("tank v5: no body mesh")
        src = srcs[0]
        lm = _torso_landmarks(arm)
        waist_top = lm["waist_top"]
        hem_z = waist_top - 0.025
        armpit_z = lm["armpit_z"]
        neck_z = lm["neck_w"].z - 0.028
        mid_x = (lm["sh_lw"].x + lm["sh_rw"].x) * 0.5
        strap_half = 0.038
        samples = _world_body_samples(meshes, arm)

        def torso_pred(co, no, name, w):
            if any(k in name for k in ("hand", "fore", "head", "foot", "toe",
                                       "leg", "upleg", "thigh")):
                return False
            if "arm" in name and "shoulder" not in name:
                return False
            return True

        n = 36
        abd = _loop_at_z(samples, 1.090, 0.030, clearance, torso_pred, n=n, min_r=0.08)
        cht = _loop_at_z(samples, 1.245, 0.030, clearance, torso_pred, n=n, min_r=0.08)
        if abd is None or cht is None:
            raise RuntimeError("tank v5: missing abdomen/chest silhouette")
        abd = _resample_ring(abd, n)
        cht = _resample_ring(cht, n)

        def ring_at(z):
            if z <= 1.100:
                r = _loop_at_z(samples, z, 0.035, clearance, torso_pred, n=n, min_r=0.08)
                if r is not None:
                    r = _resample_ring(r, n)
                    for p in r:
                        p.z = z
                    return r
                return _lerp_world_rings(abd, 1.090, cht, 1.245, z, n=n)
            if z >= 1.230:
                r = _loop_at_z(samples, z, 0.035, clearance, torso_pred, n=n, min_r=0.08)
                if r is not None:
                    r = _resample_ring(r, n)
                    for p in r:
                        p.z = z
                    return r
                return _lerp_world_rings(abd, 1.090, cht, 1.245, z, n=n)
            return _lerp_world_rings(abd, 1.090, cht, 1.245, z, n=n)

        # Full-torso rings from hem up to the armpit (armhole baseline).
        zs = []
        steps = 8
        for i in range(steps + 1):
            zs.append(hem_z + (armpit_z - hem_z) * (i / steps))
        rings_w = [ring_at(z) for z in zs]
        if any(r is None for r in rings_w):
            raise RuntimeError("tank v5: failed to build a torso ring")

        imw = src.matrix_world.inverted()
        mesh = bpy.data.meshes.new(name + "Mesh")
        obj = _new_mesh_object(name, mesh, collection)
        # Verts are in Beta_Surface local (cm, Y-up). Do not copy matrix_world
        # here — _bind_like_body will parent to the armature like the body, and
        # a pre-set world matrix would double-apply the 0.01 scale and explode.
        bm = bmesh.new()
        local_rings = [[imw @ p for p in ring] for ring in rings_w]
        rows = _loft_rings(bm, local_rings, close_caps=False)
        _cap_ring_inset(bm, rows[0], inward=0.80, drop_z=0.0)

        # Two medial straps from the armpit ring up to the neck, leaving delts open.
        arm_ring = rings_w[-1]
        def strap_pts(side_sign):
            # Pick verts in a medial corridor on this side, front and back combined
            # into one loop later as a small oval at each height.
            picked = []
            for p in arm_ring:
                dx = p.x - mid_x
                if side_sign > 0 and 0.0 <= dx <= strap_half * 1.55:
                    picked.append(p)
                if side_sign < 0 and -strap_half * 1.55 <= dx <= 0.0:
                    picked.append(p)
            if len(picked) < 4:
                # fallback: tiny oval next to the neck
                sx = mid_x + side_sign * strap_half * 0.70
                cz, cy = armpit_z, sum(p.y for p in arm_ring) / len(arm_ring)
                picked = [
                    Vector((sx + 0.012 * math.cos(a), cy + 0.016 * math.sin(a), armpit_z))
                    for a in [i * 2 * math.pi / 8 for i in range(8)]
                ]
            return picked

        for sign in (1.0, -1.0):
            base = strap_pts(sign)
            c0 = sum(base, Vector()) / len(base)
            strap_zs = [armpit_z + (neck_z - armpit_z) * t for t in (0.0, 0.45, 0.80, 1.0)]
            srings = []
            for z in strap_zs:
                t = 0.0 if neck_z == armpit_z else (z - armpit_z) / (neck_z - armpit_z)
                # Drift slightly toward the neck (medial) as we go up.
                cx = c0.x + (mid_x - c0.x) * 0.35 * t
                cy = c0.y
                rx, ry = 0.016 * (1.0 - 0.15 * t), 0.020 * (1.0 - 0.10 * t)
                srings.append([
                    Vector((cx + rx * math.cos(a), cy + ry * math.sin(a), z))
                    for a in [i * 2 * math.pi / 8 for i in range(8)]
                ])
            local_s = [[imw @ p for p in ring] for ring in srings]
            srows = _loft_rings(bm, local_s, close_caps=False)
            _cap_ring_inset(bm, list(reversed(srows[-1])), inward=0.70, drop_z=0.0)

        try:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        except Exception:
            pass
        bm.to_mesh(mesh)
        bm.free()
        _relax_boundaries(obj, iterations=6, factor=0.40)
        # Skip Solidify — even-offset on any leftover non-manifold scrap
        # (strap caps, open hems) flew tens of kilometres. The loft is already
        # a closed-enough shell for Cycles; a 2 mm thickness is not worth it.
        _smooth_mesh(obj, iterations=1, factor=0.10)
        _shade_smooth(obj)
        _bind_like_body(obj, src, arm)
        _paint_torso_chain(obj, arm)
        satin = make_satin_material(name + "Satin", colors["beanie"], roughness=0.20)
        _assign_mat(obj, satin)
        return obj, {
            "version": "v5",
            "hem_z_m": hem_z,
            "armpit_z_m": armpit_z,
            "waist_top_m": waist_top,
            "clearance_m": clearance,
            "racerback": False,
            "source": "clean-loft-local",
        }
    finally:
        _restore_pose(arm, prev)


def apply_outfit(arm, meshes, *, items=("gloves", "beanie"), colorway="A",
                 scan_frames=True, collection_name="XBotOutfit", version="v5"):
    """Build and attach the requested items. Returns a handle dict.

    `items` is a subset of {"gloves", "beanie", "headgear", "strap",
    "trunks", "tank"} ("strap" is implied by gloves). Prefer beanie XOR
    headgear. Gloves and beanie are rigid-parented to Hand / Head;
    trunks and tank are closed garment shells weight-bound to the armature.

    `version` is "v5" (default: snug boxing trunks, waist-length tank,
    kidney-thumb glossy black gloves), "v4" (earlier kit), "v3"
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
    if version in ("v4", "v5"):
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
        strength=(0.52 if version in ("v3", "v4", "v5") else 0.45 if version == "v2" else 0.22),
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
            if version in ("v4", "v5"):
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
        if version == "v5":
            fw = (handle.get("glove_info") or {}).get("R", {}).get("front_width") or                  (handle.get("glove_info") or {}).get("L", {}).get("front_width")
            handle["notes"].append(
                "v5 gloves are rigid to the Hand bone (no finger-curl squash). "
                "Glossy black leather, kidney-shaped thumb hugging the bulb, "
                "longer cuff to cover the wrist on jab extension"
                + (f"; fist front {fw:.1f} cm vs head {head_width:.1f} cm"
                   if fw and head_width else "")
                + "."
            )
        elif version == "v4":
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
            clearance = (BEANIE_V2_CLEARANCE_CM if version in ("v2", "v3", "v4", "v5")
                         else BEANIE_CLEARANCE_CM)
            if version in ("v3", "v4", "v5"):
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
        tbuild = build_trunks_v5 if version == "v5" else build_trunks_v4
        trunks, tinfo = tbuild(
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
        if version == "v5":
            handle["notes"].append(
                "v5 trunks: deep burgundy/wine satin with thin charcoal trim; "
                "snug navel waistband, fabric follows hips/glutes, mid-thigh hem "
                "with a late ~3.5 cm flare; "
                f"armature-bound, {hidden} body verts masked inside."
            )
        else:
            handle["notes"].append(
                f"{version} trunks: charcoal satin (same as the beanie base), wide navel "
                "waistband, mid-thigh legs, thin black waist/hem/side trim; "
                f"armature-bound, {hidden} body verts masked inside."
            )

    if "tank" in items:
        kbuild = build_tank_v5 if version == "v5" else build_tank_v4
        tank, tank_info = kbuild(
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
            f"{version} tank: fitted charcoal satin, sleeveless racerback, deltoids "
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
