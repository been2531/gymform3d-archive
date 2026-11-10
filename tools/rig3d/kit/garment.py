"""Authored boxer-short garment (CC0): convex-hull loft, not a body-face offset.

A convex slice at the seat cannot dive into the gluteal cleft, so clay cannot
show through there. A wide waistband and a mid-thigh hem are taken from the
hip and knee bones, not copied from sculpted topology.
"""
from __future__ import annotations
import numpy as np


def _world_verts(ob):
    """Rest-pose world verts (not the armature-evaluated mesh)."""
    co = np.empty(len(ob.data.vertices) * 3, np.float64)
    ob.data.vertices.foreach_get("co", co)
    Mw = np.array(ob.matrix_world, dtype=np.float64)
    return co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]


def _bone_z(arm, pb, *names):
    Mw = arm.matrix_world
    for n in names:
        if n in arm.data.bones:
            return float((Mw @ arm.data.bones[n].head_local).z)
    for n in names:
        if n in pb:
            return float((Mw @ pb[n].head).z)
    return None


def _slice_hull(pts, z, half, n=28, side=None, ease=0.012):
    """Convex ring of `n` points at height z, offset outward by ease."""
    sel = np.abs(pts[:, 2] - z) <= half
    xy = pts[sel][:, :2]
    if side == "L":
        xy = xy[xy[:, 0] <= 0.03]
    elif side == "R":
        xy = xy[xy[:, 0] >= -0.03]
    if len(xy) < 8:
        return None
    # Andrew's monotone chain
    xy = xy[np.lexsort((xy[:, 1], xy[:, 0]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in xy:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in xy[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return None
    c = hull.mean(0)
    ang = np.arctan2(hull[:, 1] - c[1], hull[:, 0] - c[0])
    hull = hull[np.argsort(ang)]
    # resample at equal angle so rings share an index
    ang = np.arctan2(hull[:, 1] - c[1], hull[:, 0] - c[0])
    # unwrap so it's increasing
    order = np.argsort(ang)
    hull, ang = hull[order], ang[order]
    # close
    ang_c = np.r_[ang, ang[0] + 2 * np.pi]
    hull_c = np.vstack([hull, hull[0]])
    targets = np.linspace(-np.pi, np.pi, n, endpoint=False)
    out = np.zeros((n, 3), np.float64)
    for i, t in enumerate(targets):
        # find segment
        tt = t
        if tt < ang_c[0]:
            tt += 2 * np.pi
        j = int(np.searchsorted(ang_c, tt) - 1) % len(hull)
        jn = (j + 1) % len(hull)
        a0, a1 = ang_c[j], ang_c[j + 1]
        u = 0.0 if a1 <= a0 else np.clip((tt - a0) / (a1 - a0), 0, 1)
        p = (1 - u) * hull_c[j] + u * hull_c[j + 1]
        rad = p - c
        rn = rad / (np.linalg.norm(rad) + 1e-9)
        p = c + rad + rn * float(ease)
        out[i] = (p[0], p[1], z)
    return out


def _loft_rings(rings):
    """Build a bmesh tube from a list of (n,3) rings (same n, same winding)."""
    import bmesh
    bm = bmesh.new()
    loops = []
    for ring in rings:
        loops.append([bm.verts.new(tuple(p)) for p in ring])
    n = len(loops[0])
    for a, b in zip(loops, loops[1:]):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
    return bm



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

def _join_bmeshes(bms):
    import bmesh
    out = bmesh.new()
    for bm in bms:
        verts = {v: out.verts.new(v.co) for v in bm.verts}
        for f in bm.faces:
            try:
                out.faces.new([verts[v] for v in f.verts])
            except ValueError:
                pass
        bm.free()
    return out


def build_boxer_shorts(body, arm, cfg, log=print, col=None):
    """Return (object, meta). The garment is skinned via data-transfer + armature."""
    import bmesh
    import bpy

    gcfg = ((cfg.get("kit") or {}).get("garment") or {})
    G = cfg.get("garment") or {}
    n = int(gcfg.get("rings", 28))
    ease = float(gcfg.get("ease_m", 0.012))
    wb_h = float(gcfg.get("waistband_h_m", 0.07))
    waist_above = float(gcfg.get("waist_above_hip_m", 0.10))
    hem_blend = float(gcfg.get("hem_blend", 0.48))
    wb_extra = float(gcfg.get("waistband_extra_m", 0.004))
    thick = float(gcfg.get("solidify_m", 0.003))
    smooth_n = int(gcfg.get("smooth_iters", 8))
    rgb = tuple(gcfg.get("matte_rgb") or (0.012, 0.012, 0.014))
    rough = float(gcfg.get("roughness", 0.95))

    pb = arm.pose.bones
    hip_z = _bone_z(arm, pb, "Hips") or 1.0
    # Use the lower (standing) knee. The support knee sits on the pad and is
    # a bad hem reference for a bench row.
    _kz = [z for z in (_bone_z(arm, pb, "RightLeg"), _bone_z(arm, pb, "LeftLeg")) if z is not None]
    knee_z = min(_kz) if _kz else (hip_z - 0.45)
    # UpLeg heads are hip sockets, not the perineum. Drop a fixed anatomical
    # offset so the torso loft stops at the seat and the thighs can split.
    crotch_z = hip_z - 0.16
    z_wb_top = hip_z + waist_above
    z_wb_bot = z_wb_top - wb_h
    z_hem = (1.0 - hem_blend) * knee_z + hem_blend * hip_z
    # keep hem below the crotch and well above the knee (mid-thigh)
    z_hem = min(z_hem, crotch_z - 0.06)
    z_hem = max(z_hem, knee_z + 0.10)

    # rest-pose world verts of the clay
    # temporarily jump to frame 1 / rest: caller should be at rest-ish.
    pts = _world_verts(body)
    half = 0.028
    z_hip = 0.5 * (z_wb_bot + crotch_z)
    torso_zs = [z_wb_top, z_wb_bot, z_hip, crotch_z + 0.01]
    torso_rings = []
    for z in torso_zs:
        e = ease + (wb_extra if z >= z_wb_bot - 1e-4 else 0.0)
        r = _slice_hull(pts, z, half, n=n, ease=e)
        if r is None:
            raise RuntimeError(f"garment: empty hull at z={z:.3f}")
        torso_rings.append(r)
    # thighs: left / right from just below crotch to hem
    thigh_zs = [crotch_z - 0.01, 0.5 * (crotch_z + z_hem), z_hem]
    left_rings, right_rings = [], []
    for z in thigh_zs:
        rl = _slice_hull(pts, z, half, n=n, side="L", ease=ease)
        rr = _slice_hull(pts, z, half, n=n, side="R", ease=ease)
        if rl is None or rr is None:
            raise RuntimeError(f"garment: empty thigh hull at z={z:.3f}")
        left_rings.append(rl)
        right_rings.append(rr)
    # stitch last torso ring to the top thigh rings: we don't loft across
    # the split; voxel remesh after join fuses the crotch.
    bm_t = _loft_rings(torso_rings)
    bm_l = _loft_rings(left_rings)
    bm_r = _loft_rings(right_rings)
    bm = _join_bmeshes([bm_t, bm_l, bm_r])
    # cap nothing; remesh will close small gaps
    me = bpy.data.meshes.new("boxer_shorts")
    bm.to_mesh(me)
    bm.free()
    garm = bpy.data.objects.new("briefs", me)
    if col is None:
        col = body.users_collection[0] if body.users_collection else bpy.context.scene.collection
    try:
        col.objects.link(garm)
    except RuntimeError:
        pass
    bpy.context.view_layer.objects.active = garm
    garm.select_set(True)
    # fuse torso + thighs and erase any remaining cleft by remeshing a closed-ish cage
    rm = garm.modifiers.new("kit_remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = 0.014
    sm = garm.modifiers.new("kit_smooth", "SMOOTH")
    sm.factor = 0.55
    sm.iterations = smooth_n
    sol = garm.modifiers.new("kit_solidify", "SOLIDIFY")
    sol.thickness = thick
    sol.offset = 1.0
    try:
        sol.use_quality_normals = True
    except Exception:
        pass
    for nm in ("kit_remesh", "kit_smooth", "kit_solidify"):
        try:
            bpy.ops.object.modifier_apply(modifier=nm)
        except Exception as e:
            log(f"garment modifier {nm}: {e}")

    # proud waistband: displace top verts outward in XY
    vco = np.empty(len(garm.data.vertices) * 3)
    garm.data.vertices.foreach_get("co", vco)
    vco = vco.reshape(-1, 3)
    wb = vco[:, 2] >= (z_wb_bot - 0.008)
    if np.any(wb):
        xy = vco[wb][:, :2]
        cxy = xy.mean(0)
        rad = xy - cxy
        rn = rad / (np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9)
        vco[wb, 0:2] = xy + rn * wb_extra
        garm.data.vertices.foreach_set("co", vco.ravel())
        garm.data.update()
        vg = garm.vertex_groups.new(name="waistband")
        for i in np.nonzero(wb)[0]:
            vg.add([int(i)], 1.0, "REPLACE")

    # Verts stay in world metres (object at identity). Parent to the REST
    # hips matrix so the shorts follow the hinge instead of staying at the
    # standing location of the current (already posed) frame.
    from mathutils import Matrix
    garm.matrix_world = Matrix.Identity(4)
    if not _parent_to_rest_bone(garm, arm, "Hips", Matrix.Identity(4)):
        log("kit garment: Hips bone missing, left unparented")

    mat = bpy.data.materials.new("briefs_black")
    mat.use_nodes = True
    bs = mat.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*rgb, 1)
    bs.inputs["Roughness"].default_value = rough
    for k, val in (("Specular IOR Level", 0.12), ("Coat Weight", 0.0), ("Sheen Weight", 0.04)):
        if k in bs.inputs:
            bs.inputs[k].default_value = val
    garm.data.materials.clear()
    garm.data.materials.append(mat)
    for p in garm.data.polygons:
        p.use_smooth = True

    meta = {
        "verts": len(garm.data.vertices),
        "faces": len(garm.data.polygons),
        "z_wb_top": round(z_wb_top, 3),
        "z_wb_bot": round(z_wb_bot, 3),
        "z_crotch": round(crotch_z, 3),
        "z_hem": round(z_hem, 3),
        "ease_m": ease,
        "method": "loft",
        "parent": "Hips",
    }
    log(f"kit garment loft verts={meta['verts']} hem_z={meta['z_hem']} wb=[{meta['z_wb_bot']},{meta['z_wb_top']}]")
    return garm, meta
