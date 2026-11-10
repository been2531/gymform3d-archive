"""Smooth overlay muscle shells authored from the Grab3D clay (CC0).

Each target group is copied from the body's faces, voxel-remeshed so sculpted
grooves and UV islands disappear, inflated, and bound to the same armature.
Glow is an opaque emission on the shell, not a projection on the clay.
"""
from __future__ import annotations
import colorsys
import numpy as np


def _srgb_to_lin(c):
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hsv_to_lin(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(float(h) / 180.0, float(s) / 255.0, float(v) / 255.0)
    return (_srgb_to_lin(r), _srgb_to_lin(g), _srgb_to_lin(b), 1.0)


def _dilate(mask, adj, rings):
    m = np.asarray(mask, dtype=bool).copy()
    if adj is None or rings <= 0:
        return m
    for _ in range(int(rings)):
        nxt = m.copy()
        for i in np.nonzero(m)[0]:
            nbr = adj[i] if i < len(adj) else None
            if nbr is None:
                continue
            nxt[np.asarray(nbr, dtype=np.int64)] = True
        m = nxt
    return m


def _shade_shell(obj, rgba, emission, flex_min, edge_min, w):
    import bpy
    mat = bpy.data.materials.new(obj.name + "_em")
    mat.use_nodes = True
    mat.blend_method = "OPAQUE"
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = tuple(rgba)
    flex = nt.nodes.new("ShaderNodeValue")
    flex.name = "GF_FLEX"
    flex.label = "GF_FLEX"
    flex.outputs[0].default_value = 0.0
    # strength = emission * (flex_min + (1-flex_min)*flex) * mix(edge_min, 1, shell_w)
    one_m = nt.nodes.new("ShaderNodeMath"); one_m.operation = "SUBTRACT"
    one_m.inputs[0].default_value = 1.0
    one_m.inputs[1].default_value = float(flex_min)
    mul_f = nt.nodes.new("ShaderNodeMath"); mul_f.operation = "MULTIPLY"
    nt.links.new(one_m.outputs[0], mul_f.inputs[0])
    nt.links.new(flex.outputs[0], mul_f.inputs[1])
    add_f = nt.nodes.new("ShaderNodeMath"); add_f.operation = "ADD"
    add_f.inputs[0].default_value = float(flex_min)
    nt.links.new(mul_f.outputs[0], add_f.inputs[1])
    stm = nt.nodes.new("ShaderNodeMath"); stm.operation = "MULTIPLY"
    stm.inputs[1].default_value = float(emission)
    nt.links.new(add_f.outputs[0], stm.inputs[0])
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "shell_w"
    mix = nt.nodes.new("ShaderNodeMapRange")
    mix.inputs["From Min"].default_value = 0.0
    mix.inputs["From Max"].default_value = 1.0
    mix.inputs["To Min"].default_value = float(edge_min)
    mix.inputs["To Max"].default_value = 1.0
    nt.links.new(attr.outputs["Fac"], mix.inputs["Value"])
    stm2 = nt.nodes.new("ShaderNodeMath"); stm2.operation = "MULTIPLY"
    nt.links.new(stm.outputs[0], stm2.inputs[0])
    nt.links.new(mix.outputs[0], stm2.inputs[1])
    nt.links.new(stm2.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return flex



def _parent_to_rest_bone(obj, arm, bone_name, rest_world):
    """Keep obj's rest world transform, then follow that bone's pose."""
    from mathutils import Matrix
    bone = arm.data.bones.get(bone_name)
    if bone is None:
        return False
    rest_bone_mw = arm.matrix_world @ bone.matrix_local
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = rest_bone_mw.inverted() @ rest_world
    obj.matrix_basis = Matrix.Identity(4)
    return True

def _face_mask(me, vmask):
    vmask = np.asarray(vmask, dtype=bool)
    keep = []
    for i, p in enumerate(me.polygons):
        vs = p.vertices
        if sum(1 for v in vs if v < len(vmask) and vmask[v]) >= max(1, len(vs) - 1):
            keep.append(i)
    return keep


def _apply_mods(obj, names):
    import bpy
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for nm in names:
        if nm in obj.modifiers:
            try:
                bpy.ops.object.modifier_apply(modifier=nm)
            except Exception:
                pass


def build_overlay_shells(body, arm, tid, names, hues, cfg, adj=None, log=print, col=None):
    """Return (objects, flex_nodes, info). Built in the body's rest mesh."""
    import bmesh
    import bpy
    import mathutils as mu

    mus = (cfg.get("kit") or {}).get("muscles") or {}
    groups = list(mus.get("groups") or [
        {"name": "lats", "ids": ["lats"]},
        {"name": "mid_back", "ids": ["rhomboids", "mid_trap"]},
        {"name": "rear_delt", "ids": ["rear_delt"]},
    ])
    voxel = float(mus.get("voxel_m", 0.011))
    inflate = float(mus.get("inflate_m", 0.007))
    smooth_n = int(mus.get("smooth_iters", 18))
    rings = int(mus.get("dilate_rings", 2))
    edge_min = float(mus.get("edge_min", 0.38))
    emission = float(mus.get("emission", 1.15))
    flex_min = float(mus.get("flex_min", 0.88))
    # QC identity hues (rhomboids 141, rear_delt 162) are not the look.
    # Overlay shells are vivid peach-orange unless kit.muscles.hue says otherwise.
    peach = mus.get("hue") or {"h": 12, "s": 200, "v": 244}
    name_i = {n: i for i, n in enumerate(names or [])}
    _scale = float(np.mean(np.abs(np.array(body.matrix_world.to_scale()))))
    _scale = _scale if _scale > 1e-6 else 1.0
    voxel_local = voxel / _scale
    inflate_local = inflate / _scale
    tid = np.asarray(tid) if tid is not None else None
    nv = len(body.data.vertices)
    objs, flexes, info = [], [], {"groups": []}
    if col is None:
        col = body.users_collection[0] if body.users_collection else bpy.context.scene.collection

    # store original selection
    prev_active = bpy.context.view_layer.objects.active
    for g in groups:
        ids = [s for s in (g.get("ids") or [g.get("name")]) if s]
        if tid is None or not name_i:
            info["groups"].append({"name": g.get("name"), "skipped": "no muscle ids"})
            continue
        mask = np.zeros(nv, dtype=bool)
        for nm in ids:
            if nm in name_i:
                mask |= tid == name_i[nm]
        mask = _dilate(mask, adj, rings)
        nsel = int(mask.sum())
        if nsel < 80:
            info["groups"].append({"name": g.get("name"), "skipped": f"few verts {nsel}"})
            continue
        # duplicate body, delete unselected faces
        shell = body.copy()
        shell.data = body.data.copy()
        shell.name = "shell_" + str(g.get("name") or "mus")
        col.objects.link(shell)
        # drop unused modifiers; we'll rebind
        for m in list(shell.modifiers):
            shell.modifiers.remove(m)
        bm = bmesh.new()
        bm.from_mesh(shell.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        drop = [f for f in bm.faces if sum(1 for v in f.verts if v.index < nv and mask[v.index]) < 2]
        bmesh.ops.delete(bm, geom=drop, context="FACES")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
        if len(bm.faces) < 20:
            bm.free()
            bpy.data.objects.remove(shell, do_unlink=True)
            info["groups"].append({"name": g.get("name"), "skipped": "few faces"})
            continue
        bm.to_mesh(shell.data)
        bm.free()
        shell.data.update()
        # remesh to a seamless shell (kills grooves + UV islands)
        def _remesh_wrap(vsize):
            for nm in list(shell.modifiers):
                shell.modifiers.remove(nm)
            rm = shell.modifiers.new("kit_remesh", "REMESH")
            rm.mode = "VOXEL"
            rm.voxel_size = float(vsize)
            try:
                rm.adaptivity = 0.0
            except Exception:
                pass
            sm = shell.modifiers.new("kit_smooth", "SMOOTH")
            sm.factor = 0.6
            sm.iterations = smooth_n
            _apply_mods(shell, ["kit_remesh", "kit_smooth"])
            return len(shell.data.vertices)

        src_me = shell.data.copy()
        nsv = _remesh_wrap(voxel_local)
        # small muscles (rear delt) collapse at 11 mm; retry from the unremeshed copy
        if nsv < 80:
            shell.data = src_me.copy()
            nsv = _remesh_wrap(max(voxel_local * 0.45, 0.004 / _scale))
        if nsv < 40:
            shell.data = src_me.copy()
            nsv = _remesh_wrap(max(voxel_local * 0.28, 0.003 / _scale))
        me = shell.data
        nsv = len(me.vertices)
        sco = np.empty(nsv * 3); me.vertices.foreach_get("co", sco)
        sco = sco.reshape(-1, 3)
        # Inflate along normals flipped away from the shell centroid. Grab3D
        # has inward faces, so shrinkwrap ABOVE_SURFACE hid the glow inside.
        try:
            me.calc_normals()
        except Exception:
            import bmesh as _bm
            _b = _bm.new(); _b.from_mesh(me); _b.normal_update(); _b.to_mesh(me); _b.free()
        nor = np.empty(nsv * 3); me.vertices.foreach_get("normal", nor)
        nor = nor.reshape(-1, 3)
        ctr = sco.mean(0)
        flip = ((sco - ctr) * nor).sum(1) < 0
        nor = nor.copy(); nor[flip] *= -1
        sco = sco + nor * inflate_local
        me.vertices.foreach_set("co", sco.ravel())
        me.update()
        ctr = sco.mean(0)
        d = np.linalg.norm(sco - ctr, axis=1)
        w = np.clip(1.0 - (d / (np.percentile(d, 92) + 1e-9)), 0.0, 1.0).astype(np.float32)
        attr = me.attributes.new("shell_w", "FLOAT", "POINT")
        attr.data.foreach_set("value", w)
        # Rigid follow of a torso/shoulder bone. Weight-transfer on a voxel
        # remesh exploded into orange slivers; the Grab3D scale made it worse.
        bname = {"lats": "Spine02", "mid_back": "Spine02", "rear_delt": "RightShoulder"}.get(
            str(g.get("name") or ""), "Spine02")
        if bname not in arm.data.bones:
            bname = "Spine01" if "Spine01" in arm.data.bones else "Spine"
        if bname not in arm.data.bones and "RightArm" in arm.data.bones and str(g.get("name")) == "rear_delt":
            bname = "RightArm"
        _parent_to_rest_bone(shell, arm, bname, body.matrix_world.copy())
        # shade
        hue = peach if isinstance(peach, dict) else {"h": 12, "s": 200, "v": 244}
        # bump V toward ~0.95 if the spec is a bit dim
        hv = dict(hue)
        hv["v"] = max(int(hv.get("v") or 236), 242)
        rgba = hsv_to_lin(hv.get("h", 8), hv.get("s", 205), hv.get("v", 242))
        flex = _shade_shell(shell, rgba, emission, flex_min, edge_min, w)
        for p in me.polygons:
            p.use_smooth = True
        objs.append(shell)
        flexes.append(flex)
        info["groups"].append({
            "name": shell.name, "src_verts": nsel, "shell_verts": nsv,
            "rgba": [round(c, 3) for c in rgba[:3]], "parent": getattr(shell, "parent_bone", ""),
        })
    if prev_active is not None:
        try:
            bpy.context.view_layer.objects.active = prev_active
        except Exception:
            pass
    return objs, flexes, info
