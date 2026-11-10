"""Weld a render copy of the Grab3D body so UV-seam duplicates disappear.

glTF import splits vertices at UV seams. Any mesh copied or lofted from those
faces inherits the splits and tears under deform. A merge-by-distance of
~0.1 mm makes the render body one manifold (real orifices may remain).
"""
from __future__ import annotations
import numpy as np


def topology_stats(me):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    n_bound = sum(1 for e in bm.edges if e.is_boundary)
    n_nonman = sum(1 for e in bm.edges if not e.is_manifold)
    n_wire = sum(1 for e in bm.edges if e.is_wire)
    out = {
        "nv": len(bm.verts),
        "ne": len(bm.edges),
        "nf": len(bm.faces),
        "boundary_edges": int(n_bound),
        "nonmanifold_edges": int(n_nonman),
        "wire_edges": int(n_wire),
    }
    bm.free()
    return out


def _mean_scale(ob):
    s = np.array(ob.matrix_world.to_scale(), dtype=np.float64)
    m = float(np.mean(np.abs(s)))
    return m if m > 1e-8 else 1.0


def world_verts_basis(ob):
    co = np.empty(len(ob.data.vertices) * 3, np.float64)
    ob.data.vertices.foreach_get("co", co)
    Mw = np.array(ob.matrix_world, dtype=np.float64)
    return co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]


def _kdtree(pts):
    from mathutils.kdtree import KDTree
    kd = KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert((float(p[0]), float(p[1]), float(p[2])), int(i))
    kd.balance()
    return kd


def nearest_map(src_pts, dst_pts):
    kd = _kdtree(src_pts)
    out = np.empty(len(dst_pts), np.int32)
    for i, p in enumerate(dst_pts):
        _co, idx, _d = kd.find((float(p[0]), float(p[1]), float(p[2])))
        out[i] = int(idx)
    return out


def transfer_vgroups(src, dst, nearest, log=print):
    """Copy vertex-group weights along a nearest-index map. More reliable than
    DATA_TRANSFER after a bmesh weld, which often drops the deform layer."""
    src_names = {vg.index: vg.name for vg in src.vertex_groups}
    for vg in src.vertex_groups:
        if vg.name not in dst.vertex_groups:
            dst.vertex_groups.new(name=vg.name)
    dst_vg = {vg.name: vg for vg in dst.vertex_groups}
    src_w = [[] for _ in range(len(src.data.vertices))]
    for v in src.data.vertices:
        src_w[v.index] = [(src_names[g.group], g.weight) for g in v.groups if g.group in src_names]
    n_zero = 0
    n_copy = 0
    for i, src_i in enumerate(nearest):
        pairs = src_w[int(src_i)]
        if not pairs:
            n_zero += 1
            continue
        for name, w in pairs:
            dst_vg[name].add([int(i)], float(w), "REPLACE")
        n_copy += 1
    log(f"kit weld vgroups copied={n_copy} zero={n_zero} groups={len(dst_vg)}")
    return {"copied": n_copy, "zero": n_zero, "groups": len(dst_vg)}


def _read_attr(attr, n):
    dt = attr.data_type
    if dt == "FLOAT":
        buf = np.zeros(n, np.float32)
        attr.data.foreach_get("value", buf)
        return "FLOAT", buf
    if dt == "INT":
        buf = np.zeros(n, np.int32)
        attr.data.foreach_get("value", buf)
        return "INT", buf
    if dt == "BOOLEAN":
        buf = np.zeros(n, np.int8)
        try:
            attr.data.foreach_get("value", buf)
        except Exception:
            buf = np.array([int(bool(d.value)) for d in attr.data], np.int8)
        return "BOOLEAN", buf
    if dt in ("FLOAT_COLOR", "BYTE_COLOR"):
        buf = np.zeros(n * 4, np.float32)
        try:
            attr.data.foreach_get("color", buf)
        except Exception:
            buf = np.array([list(d.color) for d in attr.data], np.float32).ravel()
        return dt, buf.reshape(n, 4)
    if dt == "FLOAT_VECTOR":
        buf = np.zeros(n * 3, np.float32)
        try:
            attr.data.foreach_get("vector", buf)
        except Exception:
            buf = np.array([list(d.vector) for d in attr.data], np.float32).ravel()
        return dt, buf.reshape(n, 3)
    return None, None


def _write_attr(me, name, dt, data):
    if name in me.attributes:
        try:
            me.attributes.remove(me.attributes[name])
        except Exception:
            pass
    n = len(me.vertices)
    attr = me.attributes.new(name, dt, "POINT")
    if dt == "FLOAT":
        attr.data.foreach_set("value", np.asarray(data, np.float32).ravel())
    elif dt == "INT":
        attr.data.foreach_set("value", np.asarray(data, np.int32).ravel())
    elif dt == "BOOLEAN":
        try:
            attr.data.foreach_set("value", np.asarray(data, np.int8).ravel())
        except Exception:
            for i, v in enumerate(np.asarray(data).ravel()):
                attr.data[i].value = bool(v)
    elif dt in ("FLOAT_COLOR", "BYTE_COLOR"):
        attr.data.foreach_set("color", np.asarray(data, np.float32).ravel())
    elif dt == "FLOAT_VECTOR":
        try:
            attr.data.foreach_set("vector", np.asarray(data, np.float32).ravel())
        except Exception:
            for i, v in enumerate(np.asarray(data).reshape(n, 3)):
                attr.data[i].vector = tuple(v)
    return attr


def transfer_point_attrs(src_me, dst_me, nearest, skip=()):
    n_src = len(src_me.vertices)
    n_dst = len(dst_me.vertices)
    skip = set(skip)
    transferred = []
    for attr in list(src_me.attributes):
        if getattr(attr, "domain", None) != "POINT":
            continue
        if attr.name in skip or attr.name.startswith("."):
            continue
        dt, data = _read_attr(attr, n_src)
        if dt is None:
            continue
        if data.ndim == 1:
            mapped = data[nearest]
        else:
            mapped = data[nearest]
        try:
            _write_attr(dst_me, attr.name, dt, mapped)
            transferred.append(attr.name)
        except Exception:
            pass
    _write_attr(dst_me, "src_vi", "INT", nearest)
    return transferred


def apply_source_shapekey_mix(src, welded, log=print):
    """Add src's current shapekey mix (no armature) onto the welded basis."""
    import bpy
    if not src.data.shape_keys:
        return {"applied": False, "reason": "no shapekeys"}
    if "src_vi" not in welded.data.attributes:
        return {"applied": False, "reason": "no src_vi"}
    nearest = np.zeros(len(welded.data.vertices), np.int32)
    welded.data.attributes["src_vi"].data.foreach_get("value", nearest)
    basis = np.empty(len(src.data.vertices) * 3, np.float64)
    src.data.vertices.foreach_get("co", basis)
    basis = basis.reshape(-1, 3)
    muted = []
    for m in src.modifiers:
        if m.type == "ARMATURE" and m.show_viewport:
            m.show_viewport = False
            muted.append(m)
    was_hide = bool(getattr(src, "hide_viewport", False))
    src.hide_viewport = False
    try:
        bpy.context.view_layer.update()
        ev = src.to_mesh()
        mix = np.empty(len(ev.vertices) * 3, np.float64)
        ev.vertices.foreach_get("co", mix)
        mix = mix.reshape(-1, 3)
        src.to_mesh_clear()
    finally:
        src.hide_viewport = was_hide
        for m in muted:
            m.show_viewport = True
    if len(mix) != len(basis):
        return {"applied": False, "reason": f"mix nv {len(mix)} != basis {len(basis)}"}
    delta = mix - basis
    wco = np.empty(len(welded.data.vertices) * 3, np.float64)
    welded.data.vertices.foreach_get("co", wco)
    wco = wco.reshape(-1, 3)
    wco = wco + delta[nearest]
    welded.data.vertices.foreach_set("co", wco.ravel())
    welded.data.update()
    moved = float(np.linalg.norm(delta[nearest], axis=1).max())
    log(f"kit weld shapekey transfer max_delta={moved:.5f} m")
    return {"applied": True, "max_delta_m": moved}


def make_render_copy(body, arm, cfg, log=print, col=None, dist_world=None):
    """Copy `body` (basis), weld, rebind weights + armature, hide original.

    Shapekeys on the source are NOT baked here (they are often added later).
    Call `apply_source_shapekey_mix` after the source mix is ready.
    """
    import bmesh
    import bpy

    kit = (cfg.get("kit") or {})
    if dist_world is None:
        dist_world = float((kit.get("weld") or {}).get("dist_m", 0.0001))
    scale = _mean_scale(body)
    dist_local = dist_world / scale
    before = topology_stats(body.data)

    if col is None:
        col = body.users_collection[0] if body.users_collection else bpy.context.scene.collection

    welded = body.copy()
    welded.data = body.data.copy()
    welded.name = "body_weld"
    try:
        col.objects.link(welded)
    except RuntimeError:
        pass
    for m in list(welded.modifiers):
        welded.modifiers.remove(m)
    if welded.data.shape_keys:
        try:
            welded.shape_key_clear()
        except Exception:
            pass

    bm = bmesh.new()
    bm.from_mesh(welded.data)
    try:
        bm.verts.layers.deform.verify()
    except Exception:
        pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(dist_local))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(welded.data)
    bm.free()
    welded.data.update()
    after = topology_stats(welded.data)

    src_w = world_verts_basis(body)
    dst_w = world_verts_basis(welded)
    nearest = nearest_map(src_w, dst_w)
    transferred = transfer_point_attrs(body.data, welded.data, nearest)
    transfer_vgroups(body, welded, nearest, log=log)

    am = welded.modifiers.new("kit_weld_arm", "ARMATURE")
    am.object = arm
    # Hide from render only. hide_viewport breaks depsgraph evaluation of
    # any modifier that still points at the original (and shapekey to_mesh).
    body.hide_render = True
    try:
        body.hide_viewport = False
    except Exception:
        pass

    stats = {
        "dist_world_m": dist_world,
        "dist_local": dist_local,
        "scale": scale,
        "before": before,
        "after": after,
        "attrs": transferred,
    }
    log(f"kit weld nv {before['nv']}->{after['nv']} "
        f"boundary {before['boundary_edges']}->{after['boundary_edges']} "
        f"nonmanifold {before['nonmanifold_edges']}->{after['nonmanifold_edges']} "
        f"dist_world={dist_world} attrs={len(transferred)}")
    return welded, stats
