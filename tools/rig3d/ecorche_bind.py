"""Bind Charleeh/ecorche's separated muscle meshes to the Grab3D armature.

Route (a) of the mesh-glow look: each target muscle is its own mesh, so the
silhouette is anatomical rather than a painted region on one-piece skin.
Non-target superficial muscles are joined into one clay shell and weighted
the same way. Skeleton and unused deep muscles are discarded.

The Grab3D body stays in the scene as the weight source (and garment/grip
wrap target) but is hidden from render.
"""
from __future__ import annotations

import os
import time

import bpy
from mathutils import Vector

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
        # Tiny skeleton pieces (ribs, vertebrae, phalanges) pick up this
        # rig's multi-metre garbage bone tails and shred the joined clay.
        # Deep + unused superficial muscles stay as clay; Grab3D is the
        # hole-filling underlayer.
        if layer.startswith("skeleton"):
            discard.append(o)
        else:
            clay.append(o)
    return hit, clay, discard


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

    # Only the glowing target muscles are bound and shown. Heat-binding
    # every other ecorche piece (face, hands, deep slivers) against this
    # rig's multi-metre bone tails shreds the shell. Grab3D, already
    # weighted, is the posed clay figure; targets overlay it.
    clay.clear()
    keep[:] = [o for vs in hit.values() for o in vs]
    _log(log, f"ecorche keep {len(keep)} target meshes (clay underlayer = Grab3D)")

    # Drop unused geometry now (saves Cycles memory).
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

    # Rhomboids lie under trapezius; drawing both tears the purple diamond.
    # mid_trap (transverse + ascending) is the surface stand-in.
    if hit.get("mid_trap") and hit.get("rhomboids"):
        for o in hit["rhomboids"]:
            try:
                o.hide_render = True
                o.hide_viewport = True
            except Exception:
                pass
        hit["rhomboids"] = []
        _log(log, "ecorche: hide rhomboids (under trapezius; mid_trap is the surface)")

    # Anatomical clip + soft edge before binding (rest pose).
    for o in list(hit.get("lats") or []):
        n0, n1 = _clip_lat_wing(o)
        _log(log, f"lat clip {o.name}: {n0} -> {n1} verts")
    for tid, objs in hit.items():
        for o in objs:
            try:
                _boundary_fade(o, rings=4)
            except Exception as e:
                _log(log, f"fade fail {o.name}: {e}")

    t1 = time.time()
    # After clipping the arm insertion, lats sit on the torso wall and can
    # copy Grab3D weights (smoother than heat, same deform as the clay body
    # so purple/orange no longer tear apart). Heat only the rear delt, which
    # rides the moving shoulder.
    HEAT = {"lats", "rear_delt", "mid_trap"}
    n_heat = n_xfer = n_fail = 0
    for tid, objs in hit.items():
        for o in objs:
            try:
                if tid in HEAT:
                    _heat_bind(o, arm)
                    n_heat += 1
                else:
                    _transfer_weights(body, o)
                    _armature_mod(o, arm)
                    n_xfer += 1
                _inflate(o, 2.0)
            except Exception as e:
                _log(log, f"bind fail {tid}/{o.name}: {e}")
                n_fail += 1
    _log(log, f"ecorche bound heat={n_heat} xfer={n_xfer} fail={n_fail} in {time.time()-t1:.1f}s")
    clay_obj = None

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

    if hide_body:
        body.hide_render = True
    else:
        # Clay underlayer: strip the painted glow_w emission so the donor
        # body only plugs holes, it does not paint atlas stickers.
        body.hide_render = False
        try:
            for slot in body.material_slots:
                mat = slot.material
                if not mat or not mat.node_tree:
                    continue
                for n in mat.node_tree.nodes:
                    if n.name.startswith("GF_") and n.type == "VALUE" and "FLEX" in n.name:
                        n.outputs[0].default_value = 0.0
        except Exception:
            pass

    arm.data.pose_position = prev_pose
    bpy.context.view_layer.update()
    _log(log, f"ecorche bind done in {time.time()-t0:.1f}s glow_mats={list(mat_cache)} flex={len(flex_nodes)}")
    return flex_nodes
