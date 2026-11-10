"""Bind Charleeh/ecorche as the FULL visible body on a clean armature.

A scale-1 copy of the Grab3D rig is built with anatomical bone tails
(head → child head, 8 cm leaves) so heat weighting does not explode.
That armature is driven from the original posed rig (Copy Location +
Damped Track to the source joint positions). Superficial muscles are
the body: targets glow, the rest join into one clay shell. Deep muscles
and the skeleton are discarded. The Grab3D mesh is hidden (garment and
dumbbell stay).
"""
from __future__ import annotations

import os
import time

import bpy
from mathutils import Vector, Matrix

import glow_shader as GSH
import muscle_ids as MIDS

# Exact extras.name values (lowercased) from ecorche.structures.json.
TARGET_MATCH = {
    "lats": (
        "left latissimus dorsi",
        "right latissimus dorsi",
    ),
    "rhomboids": (
        "left rhomboid major",
        "left rhomboid minor",
        "right rhomboid major",
        "right rhomboid minor",
    ),
    "mid_trap": (
        "transverse part of left trapezius",
        "transverse part of right trapezius",
        "ascending part of left trapezius",
        "ascending part of right trapezius",
    ),
    "upper_trap": (
        "descending part of left trapezius",
        "descending part of right trapezius",
    ),
    "rear_delt": (
        "spinal part of left deltoid",
        "spinal part of right deltoid",
    ),
    "side_delt": (
        "acromial part of left deltoid",
        "acromial part of right deltoid",
    ),
    "front_delt": (
        "clavicular part of left deltoid",
        "clavicular part of right deltoid",
    ),
    "teres_major": (
        "left teres major",
        "right teres major",
    ),
    "infraspinatus": (
        "left infraspinatus muscle",
        "right infraspinatus muscle",
    ),
    "biceps": (
        "long head of left biceps brachii",
        "short head of left biceps brachii",
        "long head of right biceps brachii",
        "short head of right biceps brachii",
    ),
    "triceps": (
        "long head of left triceps brachii",
        "lateral head of left triceps brachii",
        "medial head of left triceps brachii",
        "long head of right triceps brachii",
        "lateral head of right triceps brachii",
        "medial head of right triceps brachii",
    ),
}

FIBRE = {
    "lats": {"rot": (0.0, 0.85, 0.0), "scale": 36.0},
    "rhomboids": {"rot": (0.0, 0.0, 0.15), "scale": 52.0},
    "mid_trap": {"rot": (0.0, 0.0, 0.0), "scale": 44.0},
    "upper_trap": {"rot": (0.15, 0.0, 0.0), "scale": 40.0},
    "rear_delt": {"rot": (0.55, 0.15, 0.0), "scale": 56.0},
    "side_delt": {"rot": (0.35, 0.0, 0.0), "scale": 48.0},
    "front_delt": {"rot": (0.25, 0.2, 0.0), "scale": 48.0},
    "teres_major": {"rot": (0.0, 0.7, 0.0), "scale": 42.0},
    "infraspinatus": {"rot": (0.0, 0.35, 0.0), "scale": 46.0},
    "biceps": {"rot": (0.0, 0.05, 0.0), "scale": 40.0},
    "triceps": {"rot": (0.0, 0.08, 0.0), "scale": 40.0},
}


def _log(fn, msg):
    if fn:
        fn(msg)
    else:
        print(msg, flush=True)


def _aname(o):
    n = o.get("name") or o.name
    return str(n).split(".", 1)[0].strip().lower()


def _world_bbox_obj(obj):
    mw = obj.matrix_world
    xs, ys, zs = [], [], []
    for c in obj.bound_box:
        v = mw @ Vector(c)
        xs.append(v.x)
        ys.append(v.y)
        zs.append(v.z)
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _world_bbox(objs):
    xs = ys = zs = None
    for o in objs:
        if o.type != "MESH" or not o.data or not o.data.vertices:
            continue
        b = _world_bbox_obj(o)
        if xs is None:
            xs, ys, zs = [b[0], b[1]], [b[2], b[3]], [b[4], b[5]]
        else:
            xs[0] = min(xs[0], b[0])
            xs[1] = max(xs[1], b[1])
            ys[0] = min(ys[0], b[2])
            ys[1] = max(ys[1], b[3])
            zs[0] = min(zs[0], b[4])
            zs[1] = max(zs[1], b[5])
    if xs is None:
        raise RuntimeError("no mesh bbox")
    return (xs[0], xs[1], ys[0], ys[1], zs[0], zs[1])


def _unparent_keep(o):
    mw = o.matrix_world.copy()
    o.parent = None
    o.matrix_world = mw


def _apply_visual(objs):
    if not objs:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.hide_set(False)
        o.hide_select = False
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _join(objs, name):
    if not objs:
        return None
    if len(objs) == 1:
        objs[0].name = name
        return objs[0]
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.hide_set(False)
        o.hide_select = False
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    objs[0].name = name
    return objs[0]


def _transfer_weights(src, dst):
    mod = dst.modifiers.new("GF_WT", "DATA_TRANSFER")
    mod.object = src
    mod.use_vert_data = True
    try:
        mod.data_types_verts = {"VGROUP_WEIGHTS"}
    except TypeError:
        mod.data_types_verts.add("VGROUP_WEIGHTS")
    try:
        mod.vert_mapping = "POLYINTERP_NEAREST"
    except TypeError:
        mod.vert_mapping = "NEAREST"
    try:
        mod.layers_vgroup_select_src = "ALL"
        mod.layers_vgroup_select_dst = "NAME"
    except Exception:
        pass
    try:
        mod.mix_mode = "REPLACE"
    except Exception:
        pass
    bpy.context.view_layer.objects.active = dst
    dst.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _armature_mod(dst, arm):
    # Remove leftovers from a previous parent_set, then add a clean modifier.
    for mod in list(dst.modifiers):
        if mod.type == "ARMATURE":
            dst.modifiers.remove(mod)
    mod = dst.modifiers.new("GF_ARM", "ARMATURE")
    mod.object = arm
    mod.use_vertex_groups = True


def _inflate(dst, mm=4.0):
    # Sit the glow mesh just outside the Grab3D skin to reduce z-fighting.
    try:
        mod = dst.modifiers.new("GF_INFLATE", "DISPLACE")
        mod.strength = float(mm) * 0.001
        mod.mid_level = 0.0
        try:
            mod.direction = "NORMAL"
        except Exception:
            pass
    except Exception:
        pass



def _push_verts(obj, dist):
    """Expand a hole-fill mesh slightly along its rest normals."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.normal_update()
    d = float(dist)
    for v in bm.verts:
        v.co += v.normal * d
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()


def _clip_lat_wing(obj):
    """Drop the lumbar aponeurosis and humeral wrap.

    The ecorche latissimus mesh includes the thoracolumbar origin that sits
    over the erectors (reads as 'the whole lower back is orange') and a
    humeral insertion that slashes the posed arm. Demic-style lats are the
    lateral V-wing only.
    """
    import bmesh
    mw = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    n0 = len(bm.verts)
    kill = []
    for v in bm.verts:
        w = mw @ v.co
        ax, z = abs(float(w.x)), float(w.z)
        drop = (
            (z < 1.22 and ax < 0.125) or
            (z < 1.15 and ax < 0.17) or
            (z < 1.08 and ax < 0.20) or
            (z < 1.00 and ax < 0.22) or
            (z > 1.22 and ax < 0.13) or   # leave the mid-trap diamond clear
            (z > 1.32 and ax > 0.23)      # onto the working arm (keep axillary wing)
        )
        if drop:
            kill.append(v)
    if kill:
        bmesh.ops.delete(bm, geom=kill, context="VERTS")
    try:
        bmesh.ops.dissolve_degenerate(bm, dist=1e-5)
    except Exception:
        pass
    # smooth the new cut so the silhouette is not a sawtooth
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    boundary = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]
    if boundary:
        try:
            bmesh.ops.smooth_vert(
                bm, verts=boundary, factor=0.6, repeat=8,
                use_axis_x=True, use_axis_y=True, use_axis_z=True,
            )
        except Exception:
            pass
    bm.to_mesh(obj.data)
    obj.data.update()
    n1 = len(obj.data.vertices)
    bm.free()
    return n0, n1


def _boundary_fade(obj, rings=4):
    """Write vertex color `glow_w` = 1 inside, fading over `rings` to ~0.12."""
    import bmesh
    from collections import deque
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    nv = len(bm.verts)
    dist = [rings] * nv
    q = deque()
    for v in bm.verts:
        if any(e.is_boundary for e in v.link_edges):
            dist[v.index] = 0
            q.append(v.index)
    while q:
        i = q.popleft()
        d = dist[i]
        if d >= rings:
            continue
        for e in bm.verts[i].link_edges:
            j = e.other_vert(bm.verts[i]).index
            if dist[j] > d + 1:
                dist[j] = d + 1
                q.append(j)
    bm.free()
    w = [(0.12 + 0.88 * (d / float(rings))) for d in dist]
    attr = me.color_attributes.get("glow_w")
    if attr is None:
        attr = me.color_attributes.new("glow_w", "FLOAT_COLOR", "POINT")
    # foreach_set needs a flat rgba sequence
    cols = []
    for val in w:
        cols.extend((float(val), float(val), float(val), 1.0))
    attr.data.foreach_set("color", cols)
    me.update()
    return nv

def _heat_bind(dst, arm):
    """Skin `dst` to `arm` with automatic (heat) weights.

    Nearest-face transfer from the Grab3D body fails when rest poses differ
    (ecorche arms hang closer to the torso): arm verts pick up spine weights
    and the working limb stays in rest pose. Heat weights use bone proximity
    instead. keep_transform avoids inheriting the armature's 0.01 glTF scale.
    """
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    dst.hide_set(False)
    dst.hide_select = False
    dst.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)
    # Fold the parent-inverse into the mesh and unparent. The armature's glTF
    # scale is 0.01; joining hundreds of differently-compensated children
    # explodes the shell. Vertex groups and the Armature modifier stay.
    mw = dst.matrix_world.copy()
    dst.parent = None
    dst.matrix_world = mw
    has_arm = False
    for mod in dst.modifiers:
        if mod.type == "ARMATURE":
            mod.object = arm
            has_arm = True
    if not has_arm:
        mod = dst.modifiers.new("GF_ARM", "ARMATURE")
        mod.object = arm
        mod.use_vertex_groups = True


def _make_clay(name, base_rgb, ao_dist, ao_amt):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"][0]
    bsdf.inputs["Base Color"].default_value = (float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]), 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.18
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.0
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.0
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = float(ao_dist)
    aom = nt.nodes.new("ShaderNodeMix")
    aom.data_type = "RGBA"
    aom.blend_type = "MULTIPLY"
    aom.inputs[0].default_value = float(ao_amt)
    aom.inputs[6].default_value = (float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]), 1.0)
    nt.links.new(ao.outputs["Color"], aom.inputs[7])
    for l in list(bsdf.inputs["Base Color"].links):
        nt.links.remove(l)
    nt.links.new(aom.outputs[2], bsdf.inputs["Base Color"])
    return m


def _make_glow(name, rgb, em_str, flex_min, tint, fibre_bump, rim, fibre_rot, fibre_scale, base_rgb, ao_dist, ao_amt):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"][0]
    mout = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
    bsdf.inputs["Base Color"].default_value = (float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]), 1.0)
    bsdf.inputs["Roughness"].default_value = 0.58
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.20
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.0
    # Soft AO on the clay albedo before the glow tint hooks it.
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = float(ao_dist)
    aom = nt.nodes.new("ShaderNodeMix")
    aom.data_type = "RGBA"
    aom.blend_type = "MULTIPLY"
    aom.inputs[0].default_value = float(ao_amt) * 0.55
    aom.inputs[6].default_value = (float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]), 1.0)
    nt.links.new(ao.outputs["Color"], aom.inputs[7])
    for l in list(bsdf.inputs["Base Color"].links):
        nt.links.remove(l)
    nt.links.new(aom.outputs[2], bsdf.inputs["Base Color"])
    em_col = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    flex = GSH.attach_mesh_glow(
        nt, bsdf, mout, em_col, float(em_str), flex_min=float(flex_min),
        tint_mix=float(tint), fibre_bump=float(fibre_bump), rim=float(rim),
        fibre_rot=tuple(fibre_rot), fibre_scale=float(fibre_scale),
    )
    return m, flex


def _classify(new_objs, targets):
    want = {}
    for t in targets:
        keys = TARGET_MATCH.get(t)
        if not keys:
            continue
        want[t] = set(keys)
    hit = {t: [] for t in want}
    clay, discard = [], []
    for o in new_objs:
        if o.type != "MESH":
            continue
        layer = str(o.get("layer") or "").lower()
        name = _aname(o)
        matched = None
        for t, keys in want.items():
            if name in keys:
                matched = t
                break
        if matched:
            hit[matched].append(o)
            continue
        # Full-body path: unused superficial muscles are clay. A small
        # allowlist of deep back muscles fills holes the camera can see
        # through (triangle of auscultation, lumbar under the lat sheet).
        # Everything else deep, plus the skeleton, stays hidden.
        if layer.startswith("superficial") or name in FILL_DEEP:
            clay.append(o)
        else:
            discard.append(o)
    return hit, clay, discard



CONTINUE_CHILD = {
    "Hips": "Spine02",
    "Spine": "neck",
    "Head": "head_end",
}

# Deep muscles kept only to plug camera-visible holes (not glowed).
FILL_DEEP = {
    "left rhomboid major", "left rhomboid minor",
    "right rhomboid major", "right rhomboid minor",
    "left scapula", "right scapula",
    "left iliocostalis lumborum", "left iliocostalis thoracis",
    "right iliocostalis lumborum", "right iliocostalis thoracis",
    "left longissimus thoracis", "right longissimus thoracis",
    "left spinalis thoracis", "right spinalis thoracis",
    "left supraspinatus", "right supraspinatus",
    "left serratus posterior inferior", "right serratus posterior inferior",
}

# No scapula bone: heat splits these between spine and arm and tears a hole
# in the scapular fossa. Pin each mesh to the shoulder girdle.
FORCE_RIGID = {
    "left infraspinatus muscle": "Spine01",
    "right infraspinatus muscle": "Spine01",
    "left teres minor": "Spine01",
    "right teres minor": "Spine01",
    "left teres major": "Spine01",
    "right teres major": "Spine01",
    "left rhomboid major": "Spine01",
    "left rhomboid minor": "Spine01",
    "right rhomboid major": "Spine01",
    "right rhomboid minor": "Spine01",
    "left scapula": "Spine01",
    "right scapula": "Spine01",
    "left levator scapulae": "Spine01",
    "right levator scapulae": "Spine01",
    "left supraspinatus": "Spine01",
    "right supraspinatus": "Spine01",
    "left serratus anterior": "LeftShoulder",
    "right serratus anterior": "RightShoulder",
    "left posterior layer of thoracolumbar fascia": "Spine01",
    "right posterior layer of thoracolumbar fascia": "Spine01",
    "left iliocostalis lumborum": "Spine02",
    "right iliocostalis lumborum": "Spine02",
    "left iliocostalis thoracis": "Spine01",
    "right iliocostalis thoracis": "Spine01",
    "left longissimus thoracis": "Spine01",
    "right longissimus thoracis": "Spine01",
    "left spinalis thoracis": "Spine",
    "right spinalis thoracis": "Spine",
    "left serratus posterior inferior": "Spine02",
    "right serratus posterior inferior": "Spine02",
}


def _pick_child(bone):
    kids = list(bone.children)
    if not kids:
        return None
    prefer = CONTINUE_CHILD.get(bone.name)
    if prefer:
        for k in kids:
            if k.name == prefer:
                return k
    return min(kids, key=lambda c: (c.head_local - bone.head_local).length)


def make_clean_armature(src_arm, name="EcoArm"):
    """Scale-1 copy of `src_arm` with tails pointing at the continuation child."""
    prev = src_arm.data.pose_position
    src_arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    data = bpy.data.armatures.new(name + "Data")
    obj = bpy.data.objects.new(name, data)
    try:
        bpy.context.scene.collection.objects.link(obj)
    except RuntimeError:
        bpy.context.collection.objects.link(obj)
    obj.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = data.edit_bones
    src_mw = src_arm.matrix_world
    info = {}
    for b in src_arm.data.bones:
        h = src_mw @ b.head_local
        child = _pick_child(b)
        if child is not None:
            t = src_mw @ child.head_local
        else:
            if b.parent:
                d = h - (src_mw @ b.parent.head_local)
            else:
                d = Vector((0.0, 0.0, 0.08))
            if d.length < 1e-5:
                d = Vector((0.0, 0.0, 0.08))
            t = h + d.normalized() * 0.08
        if (t - h).length < 0.015:
            t = h + Vector((0.0, 0.0, 0.08))
        info[b.name] = (h, t, b.parent.name if b.parent else None)
    created = {}

    def add(n):
        if n in created:
            return
        h, t, pname = info[n]
        if pname:
            add(pname)
        eb = ebones.new(n)
        eb.head = h
        eb.tail = t
        eb.use_connect = False
        eb.use_deform = True
        if pname:
            eb.parent = created[pname]
        created[n] = eb

    for n in info:
        add(n)
    bpy.ops.object.mode_set(mode="OBJECT")
    src_arm.data.pose_position = prev
    lengths = [(n, (info[n][1] - info[n][0]).length) for n in info]
    return obj, lengths


def constrain_clean_to_src(clean_arm, src_arm):
    """Drive the clean armature from Grab3D joint positions, not local poses."""
    for pb in clean_arm.pose.bones:
        if pb.name not in src_arm.pose.bones:
            continue
        cl = pb.constraints.new("COPY_LOCATION")
        cl.target = src_arm
        cl.subtarget = pb.name
        cl.target_space = "WORLD"
        cl.owner_space = "WORLD"
        src_b = src_arm.data.bones[pb.name]
        child = _pick_child(src_b)
        if child is not None:
            dt = pb.constraints.new("DAMPED_TRACK")
            dt.target = src_arm
            dt.subtarget = child.name
            dt.track_axis = "TRACK_Y"
        else:
            cr = pb.constraints.new("COPY_ROTATION")
            cr.target = src_arm
            cr.subtarget = pb.name
            cr.target_space = "WORLD"
            cr.owner_space = "WORLD"


def _rigid_bind(dst, arm, bone=None):
    """Fallback: the whole mesh follows one bone (no intra-muscle tear)."""
    import numpy as np
    _clear_bind(dst)
    n = len(dst.data.vertices)
    if n == 0:
        return None
    best = bone if bone and bone in arm.data.bones else None
    if best is None:
        mw = dst.matrix_world
        co = np.empty(n * 3, dtype=np.float64)
        dst.data.vertices.foreach_get("co", co)
        c = co.reshape(-1, 3).mean(0)
        cw = mw @ Vector((float(c[0]), float(c[1]), float(c[2])))
        best, best_d = None, 1e9
        for b in arm.data.bones:
            h = arm.matrix_world @ b.head_local
            d = (h - cw).length
            if d < best_d:
                best, best_d = b.name, d
    if not best:
        return None
    vg = dst.vertex_groups.new(name=best)
    vg.add(list(range(n)), 1.0, "REPLACE")
    _armature_mod(dst, arm)
    return best


def _weight_coverage(obj):
    n = len(obj.data.vertices)
    if n == 0:
        return 0.0
    hit = 0
    for v in obj.data.vertices:
        if v.groups:
            hit += 1
    return hit / float(n)


def _clear_bind(obj):
    for vg in list(obj.vertex_groups):
        obj.vertex_groups.remove(vg)
    for mod in list(obj.modifiers):
        if mod.type == "ARMATURE":
            obj.modifiers.remove(mod)
    if obj.parent is not None:
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw


def _bind_one(obj, arm, log=None):
    """Heat-bind; if coverage is poor, rigidly parent to the nearest bone."""
    try:
        _heat_bind(obj, arm)
        cov = _weight_coverage(obj)
        if cov >= 0.90:
            return "heat", cov, None
        _log(log, f"heat coverage {cov:.2f} {obj.name} -> rigid")
    except Exception as e:
        _log(log, f"heat fail {obj.name}: {e}")
        cov = 0.0
    _clear_bind(obj)
    bn = _rigid_bind(obj, arm)
    return "rigid", cov, bn


def _bind_via_proxy(meshes, arm, log=None):
    """Transfer weights from a voxel hull so neighbouring muscles share a bind."""
    if not meshes:
        return False
    copies = []
    for o in meshes:
        c = o.copy()
        c.data = o.data.copy()
        try:
            bpy.context.scene.collection.objects.link(c)
        except RuntimeError:
            bpy.context.collection.objects.link(c)
        copies.append(c)
    proxy = _join(copies, "EcoProxy")
    if proxy is None:
        return False
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    proxy.hide_set(False)
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy
    rmod = proxy.modifiers.new("GF_VOX", "REMESH")
    rmod.mode = "VOXEL"
    rmod.voxel_size = 0.016
    try:
        bpy.ops.object.modifier_apply(modifier=rmod.name)
    except Exception as e:
        _log(log, f"proxy remesh fail: {e}")
        bpy.data.objects.remove(proxy, do_unlink=True)
        return False
    nv = len(proxy.data.vertices)
    _log(log, f"ecorche proxy verts={nv}")
    if nv < 500:
        _log(log, "proxy too empty, skip")
        mesh = proxy.data
        bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        return False
    try:
        _heat_bind(proxy, arm)
        cov = _weight_coverage(proxy)
        _log(log, f"ecorche proxy heat coverage={cov:.3f}")
        if cov < 0.90:
            raise RuntimeError(f"proxy heat coverage {cov:.3f}")
    except Exception as e:
        _log(log, f"proxy heat fail: {e}")
        mesh = proxy.data
        bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        return False
    n_ok = n_rigid = 0
    for o in meshes:
        try:
            _clear_bind(o)
            _transfer_weights(proxy, o)
            _armature_mod(o, arm)
            cov = _weight_coverage(o)
            if cov < 0.85:
                _log(log, f"transfer coverage {cov:.2f} {o.name} -> rigid")
                _clear_bind(o)
                bn = _rigid_bind(o, arm)
                _log(log, f"rigid fallback {o.name} -> {bn}")
                n_rigid += 1
            else:
                n_ok += 1
        except Exception as e:
            _log(log, f"transfer fail {o.name}: {e}")
            try:
                _clear_bind(o)
                bn = _rigid_bind(o, arm)
                _log(log, f"rigid fallback {o.name} -> {bn}")
                n_rigid += 1
            except Exception as e2:
                _log(log, f"bind fail {o.name}: {e2}")
    _log(log, f"ecorche proxy transfer ok={n_ok} rigid={n_rigid}")
    mesh = proxy.data
    bpy.data.objects.remove(proxy, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    return True


def _write_glow_w_ones(obj):
    me = obj.data
    n = len(me.vertices)
    attr = me.color_attributes.get("glow_w")
    if attr is None:
        attr = me.color_attributes.new("glow_w", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", [1.0, 1.0, 1.0, 1.0] * n)
    me.update()


def bind_and_shade(body, arm, path, targets, hues, hue_strength, base_rgb,
                   em_str, flex_min, tint, fibre_bump, rim,
                   ao_distance=0.14, ao_amount=0.38, hide_body=True, log=None):
    """Import ecorche, align + weight-transfer onto `arm`, assign glow/clay mats.

    Returns a list of GF_FLEX Value nodes (may be empty on failure).
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    t0 = time.time()
    prev_pose = arm.data.pose_position
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.get("Ecorche")
    if root is None:
        for o in new:
            if o.type == "EMPTY" and o.parent is None and o.children:
                root = o
                break
    if root is None:
        raise RuntimeError("ecorche root empty not found")

    # Bake the documented 0.0001 root scale into children.
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    hit, clay, discard = _classify(new, list(targets))
    n_hit = {k: [o.name for o in vs] for k, vs in hit.items()}
    missing = [t for t, vs in hit.items() if not vs]
    _log(log, f"ecorche import meshes={sum(1 for o in new if o.type=='MESH')} hit={n_hit} missing={missing}")

    keep = list(clay)
    for vs in hit.values():
        keep.extend(vs)
    # Discard skeleton + unused deep + unused empties later.

    # Align keep-set to the rest-pose Grab3D body (both ~1.7 m, arms down).
    # Scale by HEIGHT only: full-mesh width includes Grab3D's more abducted
    # arms and would inflate the ecorche into a giant.
    bb = _world_bbox_obj(body)
    eb = _world_bbox(keep)
    body_h = max(bb[5] - bb[4], 1e-6)
    eco_h = max(eb[5] - eb[4], 1e-6)
    body_w = max(bb[1] - bb[0], 1e-6)
    eco_w = max(eb[1] - eb[0], 1e-6)
    s = body_h / eco_h
    body_cx = 0.5 * (bb[0] + bb[1])
    body_cy = 0.5 * (bb[2] + bb[3])
    root.scale = (s, s, s)
    bpy.context.view_layer.update()
    eb2 = _world_bbox(keep)
    tx = body_cx - 0.5 * (eb2[0] + eb2[1])
    ty = body_cy - 0.5 * (eb2[2] + eb2[3])
    tz = bb[4] - eb2[4]
    root.location = (tx, ty, tz)
    bpy.context.view_layer.update()
    _log(log, f"ecorche align s={s:.4f} (h {body_h:.3f}/{eco_h:.3f} w {body_w:.3f}/{eco_w:.3f}) t=({tx:.3f},{ty:.3f},{tz:.3f})")

    # Capture world matrices (including parent scale+loc) then unparent.
    # Applying loc/scale on the root empty itself can invalidate child Object
    # RNA ("StructRNA of type Object has been removed").
    alive = []
    for o in keep:
        try:
            if o.name not in bpy.data.objects:
                continue
            _unparent_keep(o)
            alive.append(o)
        except ReferenceError:
            continue
    keep[:] = alive
    clay[:] = [o for o in clay if o in set(alive)]
    for t in list(hit):
        hit[t] = [o for o in hit[t] if o in set(alive)]
    _apply_visual(keep)
    _log(log, f"ecorche unparented {len(keep)} meshes")

    # Rhomboids glow-target is suppressed when mid_trap is the purple surface,
    # but the meshes stay as clay so they plug the triangle of auscultation.
    if hit.get("mid_trap") and hit.get("rhomboids"):
        for o in hit["rhomboids"]:
            if o not in clay:
                clay.append(o)
        hit["rhomboids"] = []
        _log(log, "ecorche: rhomboids as clay fill (mid_trap is the glow surface)")

    # Superficial clay + visible targets are the full body. Everything else goes.
    keep[:] = [o for o in clay if o is not None] + [o for vs in hit.values() for o in vs]
    _log(log, f"ecorche full-body keep {len(keep)} (clay={len(clay)} targets={sum(len(v) for v in hit.values())})")

    drop = []
    keep_set = set(keep)
    for o in new:
        try:
            if o in keep_set:
                continue
            if o.type == "MESH":
                drop.append(o)
        except ReferenceError:
            continue
    for o in drop:
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass
    for o in list(new):
        try:
            if o.name in bpy.data.objects and o.type == "EMPTY":
                bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass

    # Clean bind armature: same joint positions, sane tails.
    clean_arm, blens = make_clean_armature(arm, name="EcoArm")
    _log(log, "ecorche clean arm tails " + ", ".join(f"{n}={l:.3f}m" for n, l in blens if l > 0.6 or n in ("LeftArm", "Hips", "Spine", "LeftHand")))

    t1 = time.time()
    bind_list = list(dict.fromkeys(keep))
    used_proxy = _bind_via_proxy(bind_list, clean_arm, log=log)
    if used_proxy:
        _log(log, f"ecorche bound via proxy in {time.time()-t1:.1f}s")
    else:
        n_heat = n_rigid = n_fail = 0
        for o in bind_list:
            try:
                force = FORCE_RIGID.get(_aname(o))
                if force:
                    _clear_bind(o)
                    bn = _rigid_bind(o, clean_arm, bone=force)
                    n_rigid += 1
                    _log(log, f"force rigid {o.name} -> {bn}")
                    continue
                mode, cov, bn = _bind_one(o, clean_arm, log=log)
                if mode == "heat":
                    n_heat += 1
                else:
                    n_rigid += 1
                    _log(log, f"rigid fallback {o.name} -> {bn} (cov={cov:.2f})")
            except Exception as e2:
                _log(log, f"bind fail {o.name}: {e2}")
                n_fail += 1
        _log(log, f"ecorche bound heat={n_heat} rigid={n_rigid} fail={n_fail} in {time.time()-t1:.1f}s")

    glow_set = {o for vs in hit.values() for o in vs}
    for o in bind_list:
        if o in glow_set:
            continue
        n = _aname(o)
        if n in FILL_DEEP or n in FORCE_RIGID:
            try:
                _push_verts(o, 0.008)
            except Exception as e:
                _log(log, f"push fail {o.name}: {e}")

    t2 = time.time()
    clay_obj = _join(clay, "EcoClay") if clay else None
    _log(log, f"ecorche clay joined n={len(clay)} obj={getattr(clay_obj, 'name', None)} in {time.time()-t2:.1f}s")

    constrain_clean_to_src(clean_arm, arm)
    _log(log, "ecorche EcoArm constrained to Grab3D joint positions")

    # Full-muscle glow: attribute = 1 on every target vert (shader reads glow_w).
    for objs in hit.values():
        for o in objs:
            try:
                _write_glow_w_ones(o)
            except Exception as e:
                _log(log, f"glow_w fail {o.name}: {e}")

    # Materials: one clay, one glow mat per target id (L/R share).
    clay_mat = _make_clay("EcoClayMat", base_rgb, ao_distance, ao_amount)
    if clay_obj is not None:
        clay_obj.data.materials.clear()
        clay_obj.data.materials.append(clay_mat)

    flex_nodes = []
    mat_cache = {}
    for tid, objs in hit.items():
        if not objs:
            continue
        spec = (hues or {}).get(tid) or MIDS.DEFAULT_HUES.get(tid) or {"h": 10, "s": 200, "v": 242}
        rgb = MIDS.hsv_to_lin_rgb(spec["h"], spec["s"], spec["v"])
        strength = 1.0
        if isinstance(hue_strength, dict) and tid in hue_strength:
            try:
                strength = float(hue_strength[tid])
            except (TypeError, ValueError):
                strength = 1.0
        fib = FIBRE.get(tid) or {"rot": (0.0, 0.4, 0.0), "scale": 48.0}
        key = (tid, round(strength, 4))
        if key not in mat_cache:
            mat, flex = _make_glow(
                f"EcoGlow_{tid}", rgb, float(em_str) * strength, flex_min, tint,
                fibre_bump, rim, fib["rot"], fib["scale"], base_rgb, ao_distance, ao_amount,
            )
            mat_cache[key] = mat
            flex_nodes.append(flex)
        mat = mat_cache[key]
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(mat)

    body.hide_render = True
    body.hide_viewport = True

    arm.data.pose_position = prev_pose
    bpy.context.view_layer.update()
    _log(log, f"ecorche bind done in {time.time()-t0:.1f}s glow_mats={list(mat_cache)} flex={len(flex_nodes)}")
    return flex_nodes
