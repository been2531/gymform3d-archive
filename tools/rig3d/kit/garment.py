"""Closed boxer-short garment (CC0): authored watertight tube, optional cloth fit.

The tube is built from convex waist/seat rings plus two thigh tubes and a
crotch bridge — never copied from Grab3D faces — so UV-seam tears cannot
appear. Cloth (if it actually bakes headless) shrinks the tube onto the
welded rest body with the waistband pinned. A failed cloth bake is reported
and the authored tube is kept; we do not pretend the sim ran.
"""
from __future__ import annotations
import numpy as np


def _world_verts(ob):
    """Rest-pose world verts. World-Z slices are only valid on a standing figure."""
    co = np.empty(len(ob.data.vertices) * 3, np.float64)
    ob.data.vertices.foreach_get("co", co)
    Mw = np.array(ob.matrix_world, dtype=np.float64)
    return co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]


def _torso_world_verts(ob):
    """World verts with arm/hand groups removed so hanging arms do not widen hip hulls."""
    pts = _world_verts(ob)
    gis = []
    for vg in ob.vertex_groups:
        ln = vg.name.lower()
        if "hand" in ln or "arm" in ln:
            gis.append(vg.index)
    if not gis:
        return pts
    gi = set(gis)
    keep = np.ones(len(pts), dtype=bool)
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group in gi and g.weight > 0.35:
                keep[v.index] = False
                break
    out = pts[keep]
    return out if len(out) > 32 else pts


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
    order = np.argsort(ang)
    hull, ang = hull[order], ang[order]
    ang_c = np.r_[ang, ang[0] + 2 * np.pi]
    hull_c = np.vstack([hull, hull[0]])
    targets = np.linspace(-np.pi, np.pi, n, endpoint=False)
    out = np.zeros((n, 3), np.float64)
    for i, t in enumerate(targets):
        tt = t
        if tt < ang_c[0]:
            tt += 2 * np.pi
        j = int(np.searchsorted(ang_c, tt) - 1) % len(hull)
        a0, a1 = ang_c[j], ang_c[j + 1]
        u = 0.0 if a1 <= a0 else np.clip((tt - a0) / (a1 - a0), 0, 1)
        p = (1 - u) * hull_c[j] + u * hull_c[j + 1]
        rad = p - c
        rn = rad / (np.linalg.norm(rad) + 1e-9)
        p = c + rad + rn * float(ease)
        out[i] = (p[0], p[1], z)
    return out


def _loft_into(bm, rings):
    """Append a lofted tube of `rings` to bm. Returns list of BMVert loops."""
    loops = []
    for ring in rings:
        loops.append([bm.verts.new(tuple(p)) for p in ring])
    n = len(loops[0])
    for a, b in zip(loops, loops[1:]):
        for i in range(n):
            j = (i + 1) % n
            try:
                bm.faces.new((a[i], a[j], b[j], b[i]))
            except ValueError:
                pass
    return loops


def _add_flange(bm, S_verts, L_verts, R_verts):
    cloud_v = list(L_verts) + list(R_verts)
    cloud = np.array([tuple(v.co) for v in cloud_v], dtype=np.float64)
    S = np.array([tuple(v.co) for v in S_verts], dtype=np.float64)
    d = ((S[:, None, :] - cloud[None, :, :]) ** 2).sum(-1)
    idx = d.argmin(1)
    n = len(S_verts)
    nfaces = 0
    for i in range(n):
        j = (i + 1) % n
        a, b = S_verts[i], S_verts[j]
        c, dlt = cloud_v[int(idx[j])], cloud_v[int(idx[i])]
        try:
            if c == dlt:
                bm.faces.new((a, b, c))
            else:
                bm.faces.new((a, b, c, dlt))
            nfaces += 1
        except ValueError:
            try:
                bm.faces.new((a, b, c))
                nfaces += 1
            except ValueError:
                pass
    return nfaces


def _crotch_bridge(bm, L_verts, R_verts, drop=0.028):
    L = np.array([tuple(v.co) for v in L_verts], dtype=np.float64)
    R = np.array([tuple(v.co) for v in R_verts], dtype=np.float64)
    Lc, Rc = L.mean(0), R.mean(0)
    mid_dir = Rc - Lc
    L_in = (L - Lc)[:, :2] @ mid_dir[:2] > 0
    R_in = (R - Rc)[:, :2] @ (-mid_dir[:2]) > 0

    def chain(verts, mask):
        idx = np.nonzero(mask)[0]
        if len(idx) < 2:
            return []
        ys = np.array([verts[i].co[1] for i in idx])
        order = idx[np.argsort(-ys)]
        return [verts[int(i)] for i in order]

    def take(seq, k):
        if len(seq) <= k:
            return seq
        sel = np.linspace(0, len(seq) - 1, k).round().astype(int)
        return [seq[i] for i in sel]

    Lc_v, Rc_v = chain(L_verts, L_in), chain(R_verts, R_in)
    k = min(len(Lc_v), len(Rc_v), 10)
    if k < 2:
        return 0
    Lc_v, Rc_v = take(Lc_v, k), take(Rc_v, k)
    mid = []
    for a, b in zip(Lc_v, Rc_v):
        p = 0.5 * (np.array(a.co) + np.array(b.co))
        p[2] -= drop
        mid.append(bm.verts.new(tuple(p)))
    nfaces = 0
    for i in range(k - 1):
        for strip in (
            (Lc_v[i], Lc_v[i + 1], mid[i + 1], mid[i]),
            (Rc_v[i], mid[i], mid[i + 1], Rc_v[i + 1]),
        ):
            try:
                bm.faces.new(strip)
                nfaces += 1
            except ValueError:
                pass
    return nfaces


def _boundary_loops(bm):
    used = set()
    loops = []
    bset = {e for e in bm.edges if e.is_boundary}
    for e0 in list(bset):
        if e0 in used:
            continue
        loop = []
        e = e0
        v = e.verts[0]
        guard = 0
        while e not in used and guard < 10000:
            used.add(e)
            loop.append(e)
            v = e.other_vert(v)
            nxt = None
            for ee in v.link_edges:
                if ee in bset and ee not in used:
                    nxt = ee
                    break
            if nxt is None:
                break
            e = nxt
            guard += 1
        if loop:
            loops.append(loop)
    return loops


def _matte_black(rgb, rough):
    import bpy
    mat = bpy.data.materials.new("briefs_black")
    mat.use_nodes = True
    bs = mat.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*rgb, 1)
    bs.inputs["Roughness"].default_value = rough
    for k, val in (("Specular IOR Level", 0.12), ("Coat Weight", 0.0), ("Sheen Weight", 0.04)):
        if k in bs.inputs:
            bs.inputs[k].default_value = val
    return mat




def _widest_z(pts, z0, z1, step=0.012):
    best_z, best_r = 0.5 * (z0 + z1), -1.0
    z = float(min(z0, z1))
    z_hi = float(max(z0, z1))
    while z <= z_hi + 1e-9:
        sl = pts[np.abs(pts[:, 2] - z) <= 0.03]
        if len(sl) >= 8:
            xy = sl[:, :2]
            r = float(np.linalg.norm(xy - xy.mean(0), axis=1).max())
            if r > best_r:
                best_r, best_z = r, z
        z += step
    return best_z, best_r


def _push_clearance(garm, body_pts, min_clear=0.010, log=print):
    """Push any garment vert closer than min_clear off the rest body."""
    from mathutils.kdtree import KDTree
    if body_pts is None or len(body_pts) < 8:
        return 0
    kd = KDTree(len(body_pts))
    for i, p in enumerate(body_pts):
        kd.insert((float(p[0]), float(p[1]), float(p[2])), int(i))
    kd.balance()
    gco = np.empty(len(garm.data.vertices) * 3, np.float64)
    garm.data.vertices.foreach_get("co", gco)
    gco = gco.reshape(-1, 3)
    n_push = 0
    for i, p in enumerate(gco):
        _co, idx, dist = kd.find((float(p[0]), float(p[1]), float(p[2])))
        bp = body_pts[int(idx)]
        delta = p - bp
        d = float(np.linalg.norm(delta))
        if d >= min_clear:
            continue
        if d < 1e-6:
            nrm = np.array([p[0], p[1], 0.0], dtype=np.float64)
            ln = float(np.linalg.norm(nrm))
            nrm = nrm / ln if ln > 1e-9 else np.array([0.0, 0.0, 1.0])
        else:
            nrm = delta / d
        gco[i] = bp + nrm * float(min_clear)
        n_push += 1
    if n_push:
        garm.data.vertices.foreach_set("co", gco.ravel())
        garm.data.update()
    log(f"garment clearance min={min_clear:.3f} m pushed={n_push}/{len(gco)}")
    return n_push


def _fill_loop(bm, loop):
    if len(loop) < 3:
        return False
    used = set()
    e = loop[0]
    v = e.verts[0]
    verts = []
    guard = 0
    while e not in used and guard < 10000:
        used.add(e)
        verts.append(v)
        v = e.other_vert(v)
        nxt = None
        for ee in v.link_edges:
            if ee.is_boundary and ee not in used:
                nxt = ee
                break
        if nxt is None:
            break
        e = nxt
        guard += 1
    if len(verts) < 3:
        return False
    try:
        bm.faces.new(verts)
        return True
    except ValueError:
        pass
    c = np.mean([np.array(vv.co) for vv in verts], axis=0)
    cv = bm.verts.new(tuple(c))
    ok = 0
    n = len(verts)
    for i in range(n):
        try:
            bm.faces.new((verts[i], verts[(i + 1) % n], cv))
            ok += 1
        except ValueError:
            pass
    return ok > 0


def _close_mesh_holes(obj, keep_open=3, weld_dist=0.0002, log=print):
    """Fill extra holes and collapse 1–2-edge stubs, keeping the 3 openings."""
    import bmesh
    from .weld import topology_stats
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(weld_dist))
    loops = sorted(_boundary_loops(bm), key=len, reverse=True)
    log(f"garment close_holes loops={[len(L) for L in loops]}")
    filled = 0
    for loop in loops[int(keep_open):]:
        if len(loop) >= 3 and _fill_loop(bm, loop):
            filled += 1
    loops = sorted(_boundary_loops(bm), key=len, reverse=True)
    keep = {e for loop in loops[:int(keep_open)] for e in loop}
    stubs = [e for e in bm.edges if e.is_boundary and e not in keep]
    collapsed = 0
    if stubs:
        try:
            bmesh.ops.collapse(bm, edges=stubs, uvs=False)
            collapsed = len(stubs)
        except Exception:
            for e in list(stubs):
                try:
                    vs = list(e.verts)
                    mid = tuple(0.5 * (np.array(vs[0].co) + np.array(vs[1].co)))
                    bmesh.ops.pointmerge(bm, verts=vs, merge_co=mid)
                    collapsed += 1
                except Exception:
                    pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(weld_dist) * 2.0)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    st = topology_stats(obj.data)
    log(f"garment close_holes filled={filled} collapsed={collapsed} "
        f"bound={st['boundary_edges']} nonman={st['nonmanifold_edges']}")
    return st


def _seal_boundary(obj, log):
    """Cap leftover boundary loops after solidify (fabric rims, not wear-holes)."""
    import bmesh
    from .weld import topology_stats
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    loops = _boundary_loops(bm)
    log(f"garment seal start bound={sum(1 for e in bm.edges if e.is_boundary)} loops={[len(L) for L in loops]}")
    filled = 0
    for loop in list(_boundary_loops(bm)):
        if len(loop) < 3:
            continue
        used = set()
        e = loop[0]
        v = e.verts[0]
        verts = []
        guard = 0
        while e not in used and guard < 10000:
            used.add(e)
            verts.append(v)
            v = e.other_vert(v)
            nxt = None
            for ee in v.link_edges:
                if ee.is_boundary and ee not in used:
                    nxt = ee
                    break
            if nxt is None:
                break
            e = nxt
            guard += 1
        if len(verts) < 3:
            continue
        try:
            bm.faces.new(verts)
            filled += 1
            continue
        except ValueError:
            pass
        c = np.mean([np.array(vv.co) for vv in verts], axis=0)
        cv = bm.verts.new(tuple(c))
        ok = 0
        n = len(verts)
        for i in range(n):
            try:
                bm.faces.new((verts[i], verts[(i + 1) % n], cv))
                ok += 1
            except ValueError:
                pass
        filled += int(ok > 0)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    stubs = [e for e in bm.edges if e.is_boundary]
    if stubs:
        try:
            bmesh.ops.collapse(bm, edges=stubs, uvs=False)
        except Exception:
            for e in list(stubs):
                try:
                    vs = list(e.verts)
                    mid = tuple(0.5 * (np.array(vs[0].co) + np.array(vs[1].co)))
                    bmesh.ops.pointmerge(bm, verts=vs, merge_co=mid)
                except Exception:
                    pass
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0004)
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    st = topology_stats(obj.data)
    log(f"garment seal filled={filled} bound={st['boundary_edges']} nonman={st['nonmanifold_edges']}")
    return st


def _solidify_close(obj, thick, smooth_n, log):
    import bpy
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if smooth_n > 0:
        sm = obj.modifiers.new("kit_smooth", "SMOOTH")
        sm.factor = 0.55
        sm.iterations = int(smooth_n)
        try:
            bpy.ops.object.modifier_apply(modifier="kit_smooth")
        except Exception as e:
            log(f"garment smooth: {e}")
    sol = obj.modifiers.new("kit_solidify", "SOLIDIFY")
    sol.thickness = float(thick)
    sol.offset = 1.0
    try:
        sol.use_rim = True
    except Exception:
        pass
    try:
        sol.use_quality_normals = True
    except Exception:
        pass
    try:
        bpy.ops.object.modifier_apply(modifier="kit_solidify")
    except Exception as e:
        log(f"garment solidify: {e}")


def _bind_to_body(obj, donor, arm, log):
    """Put world-metre mesh into donor local space, transfer weights, armature."""
    import bpy
    Mw = np.array(donor.matrix_world, dtype=np.float64)
    Minv = np.linalg.inv(Mw)
    co = np.empty(len(obj.data.vertices) * 3, np.float64)
    obj.data.vertices.foreach_get("co", co)
    pts = co.reshape(-1, 3)
    # treat current coords as world (object mw should be I)
    Om = np.array(obj.matrix_world, dtype=np.float64)
    world = pts @ Om[:3, :3].T + Om[:3, 3]
    local = world @ Minv[:3, :3].T + Minv[:3, 3]
    obj.data.vertices.foreach_set("co", local.ravel())
    obj.data.update()
    obj.matrix_world = donor.matrix_world.copy()

    from .weld import nearest_map, transfer_vgroups
    src_pts = np.empty(len(donor.data.vertices) * 3, np.float64)
    donor.data.vertices.foreach_get("co", src_pts)
    src_pts = src_pts.reshape(-1, 3)
    dst_pts = np.empty(len(obj.data.vertices) * 3, np.float64)
    obj.data.vertices.foreach_get("co", dst_pts)
    dst_pts = dst_pts.reshape(-1, 3)
    nearest = nearest_map(src_pts, dst_pts)
    transfer_vgroups(donor, obj, nearest, log=log)
    am = obj.modifiers.new("kit_garm_arm", "ARMATURE")
    am.object = arm
    # After skinning, push the cloth off the posed body so a rest-fit tube
    # cannot be swallowed by the hinged hip. Strength is world metres / scale.
    inflate = float(getattr(_bind_to_body, "_inflate_m", 0.0) or 0.0)
    if inflate > 1e-5:
        scale = float(np.mean(np.abs(np.array(donor.matrix_world.to_scale())))) or 1.0
        disp = obj.modifiers.new("kit_garm_inflate", "DISPLACE")
        disp.mid_level = 0.0
        disp.strength = inflate / scale
        try:
            disp.direction = "NORMAL"
        except Exception:
            pass
        if "pose_inflate" in obj.vertex_groups:
            disp.vertex_group = "pose_inflate"
        log(f"garment inflate {inflate:.3f} m (local {disp.strength:.3f}) "
            f"vg={'pose_inflate' if 'pose_inflate' in obj.vertex_groups else 'all'}")
    return True


def _try_cloth(garm, collider, pin_name, frames, shrink, log, collide_dist=0.005):
    """Attempt a headless cloth bake. Returns a report dict; never fakes success."""
    import bpy
    from .weld import topology_stats

    report = {
        "attempted": True,
        "ok": False,
        "applied": False,
        "method": None,
        "error": None,
        "frames": int(frames),
        "shrink": float(shrink),
        "max_disp_m": 0.0,
        "mean_disp_m": 0.0,
        "bbox_before": None,
        "bbox_after": None,
        "exploded": False,
    }
    rest = np.empty(len(garm.data.vertices) * 3, np.float64)
    garm.data.vertices.foreach_get("co", rest)
    rest = rest.reshape(-1, 3).copy()
    report["bbox_before"] = [rest.min(0).round(4).tolist(), rest.max(0).round(4).tolist()]

    colm = collider.modifiers.new("kit_collide", "COLLISION")
    try:
        collider.collision.thickness_outer = float(collide_dist)
        collider.collision.thickness_inner = float(collide_dist)
        collider.collision.use_culling = False
    except Exception as e:
        log(f"cloth collision settings: {e}")

    cl = garm.modifiers.new("kit_cloth", "CLOTH")
    s = cl.settings
    try:
        s.quality = 10
        s.mass = 0.35
        s.time_scale = 1.0
        s.tension_stiffness = 18.0
        s.compression_stiffness = 18.0
        s.shear_stiffness = 6.0
        s.bending_stiffness = 0.6
        s.shrink_min = float(shrink)
        if hasattr(s, "shrink_max"):
            s.shrink_max = float(shrink)
    except Exception as e:
        log(f"cloth stiffness settings: {e}")
    try:
        s.vertex_group_mass = pin_name
        if hasattr(s, "pin_stiffness"):
            s.pin_stiffness = 1.0
    except Exception as e:
        log(f"cloth pin group: {e}")
    try:
        if hasattr(s, "effector_weights") and hasattr(s.effector_weights, "gravity"):
            s.effector_weights.gravity = 0.2
    except Exception as e:
        log(f"cloth gravity weight: {e}")
    cc = cl.collision_settings
    try:
        cc.use_collision = True
        cc.distance_min = float(collide_dist)
        cc.use_self_collision = True
        if hasattr(cc, "self_distance_min"):
            cc.self_distance_min = max(0.002, float(collide_dist) * 0.5)
        if hasattr(cc, "impulse_clamp"):
            cc.impulse_clamp = 8.0
    except Exception as e:
        log(f"cloth collision_settings: {e}")

    sc = bpy.context.scene
    old = (sc.frame_start, sc.frame_end, sc.frame_current)
    frames = int(frames)
    sc.frame_start, sc.frame_end = 1, frames
    sc.frame_set(1)
    bpy.context.view_layer.update()
    pc = cl.point_cache
    try:
        pc.frame_start = 1
        pc.frame_end = frames
    except Exception:
        pass

    bpy.context.view_layer.objects.active = garm
    garm.select_set(True)

    def _verts_now():
        dg = bpy.context.evaluated_depsgraph_get()
        eg = garm.evaluated_get(dg)
        m = eg.to_mesh()
        buf = np.empty(len(m.vertices) * 3, np.float64)
        m.vertices.foreach_get("co", buf)
        eg.to_mesh_clear()
        return buf.reshape(-1, 3)

    try:
        with bpy.context.temp_override(scene=sc, active_object=garm, object=garm, point_cache=pc):
            bpy.ops.ptcache.bake(bake=True)
        report["method"] = "ptcache.bake"
        report["ok"] = True
    except Exception as e1:
        try:
            bpy.ops.ptcache.bake(bake=True)
            report["method"] = "ptcache.bake_active"
            report["ok"] = True
        except Exception as e2:
            try:
                for f in range(1, frames + 1):
                    sc.frame_set(f)
                    bpy.context.view_layer.update()
                report["method"] = "frame_step"
                report["ok"] = True
                report["error"] = f"ptcache failed: {e1} | {e2}"
            except Exception as e3:
                report["method"] = "failed"
                report["error"] = f"{e1} | {e2} | {e3}"
                report["ok"] = False

    deformed = None
    try:
        sc.frame_set(frames)
        bpy.context.view_layer.update()
        deformed = _verts_now()
    except Exception as e:
        report["error"] = (report.get("error") or "") + f" eval:{e}"

    if deformed is not None and len(deformed) == len(rest):
        if not np.isfinite(deformed).all():
            report["error"] = (report.get("error") or "") + " NaNs in cloth verts"
            report["ok"] = False
        else:
            disp = np.linalg.norm(deformed - rest, axis=1)
            report["max_disp_m"] = float(disp.max())
            report["mean_disp_m"] = float(disp.mean())
            report["bbox_after"] = [deformed.min(0).round(4).tolist(),
                                    deformed.max(0).round(4).tolist()]
            extent = float((deformed.max(0) - deformed.min(0)).max())
            report["exploded"] = bool(extent > 3.0 or report["max_disp_m"] > 0.6)
            if report["exploded"]:
                report["ok"] = False
                report["error"] = (report.get("error") or "") + f" exploded extent={extent:.3f}"

    applied = False
    if report["ok"] and not report["exploded"] and report["max_disp_m"] > 0.0005:
        try:
            bpy.ops.object.modifier_apply(modifier="kit_cloth")
            applied = True
        except Exception as e:
            try:
                if deformed is not None and len(deformed) == len(garm.data.vertices):
                    garm.data.vertices.foreach_set("co", deformed.ravel())
                    garm.data.update()
                    garm.modifiers.remove(cl)
                    applied = True
                    report["method"] = (report["method"] or "") + "+to_mesh"
                else:
                    report["error"] = (report.get("error") or "") + f" apply:{e}"
            except Exception as e2:
                report["error"] = (report.get("error") or "") + f" apply:{e} | {e2}"
    else:
        # revert / drop cloth — keep authored tube
        try:
            if "kit_cloth" in garm.modifiers:
                garm.modifiers.remove(garm.modifiers["kit_cloth"])
        except Exception:
            pass
        garm.data.vertices.foreach_set("co", rest.ravel())
        garm.data.update()
        if report["ok"] and report["max_disp_m"] <= 0.0005:
            report["error"] = (report.get("error") or "") + " cloth did not move verts (>0.5 mm)"
            report["ok"] = False

    report["applied"] = applied
    sc.frame_start, sc.frame_end, _cf = old[0], old[1], old[2]
    try:
        sc.frame_set(old[2])
    except Exception:
        pass
    if "kit_collide" in collider.modifiers:
        try:
            collider.modifiers.remove(collider.modifiers["kit_collide"])
        except Exception:
            pass
    report["topo_after"] = topology_stats(garm.data)
    log(f"kit cloth {report['method']} ok={report['ok']} applied={applied} "
        f"max_disp={report['max_disp_m']*1000:.2f}mm mean={report['mean_disp_m']*1000:.2f}mm "
        f"err={report['error']}")
    return report


def build_boxer_shorts(body, arm, cfg, log=print, col=None, collider=None):
    """Return (object, meta). Closed authored tube, cloth if it really bakes."""
    import bmesh
    import bpy
    from .weld import topology_stats

    gcfg = ((cfg.get("kit") or {}).get("garment") or {})
    n = int(gcfg.get("rings", 32))
    n_leg = int(gcfg.get("leg_rings", max(16, n // 2)))
    ease = float(gcfg.get("ease_m", 0.012))
    ease_waist = float(gcfg.get("ease_waist_m", 0.010))
    ease_hip = float(gcfg.get("ease_hip_m", 0.015))
    ease_thigh = float(gcfg.get("ease_thigh_m", 0.014))
    wb_h = float(gcfg.get("waistband_h_m", 0.07))
    waist_above = float(gcfg.get("waist_above_hip_m", 0.10))
    hem_blend = float(gcfg.get("hem_blend", 0.36))
    wb_extra = float(gcfg.get("waistband_extra_m", 0.006))
    thick = float(gcfg.get("solidify_m", 0.003))
    thick_wb = float(gcfg.get("waistband_solidify_m", 0.008))
    smooth_n = int(gcfg.get("smooth_iters", 6))
    rgb = tuple(gcfg.get("matte_rgb") or (0.012, 0.012, 0.014))
    rough = float(gcfg.get("roughness", 0.95))
    cloth_on = str(gcfg.get("method") or "cloth").lower() in ("cloth", "closed", "tube")
    cloth_frames = int(gcfg.get("cloth_frames", 28))
    cloth_shrink = float(gcfg.get("cloth_shrink", 0.01))
    collide_dist = float(gcfg.get("cloth_collide_m", 0.005))
    min_clear = float(gcfg.get("clearance_m", 0.010))
    want_cloth = bool(gcfg.get("cloth", cloth_on))

    pb = arm.pose.bones
    hip_z = _bone_z(arm, pb, "Hips") or 1.0
    _kz = [z for z in (_bone_z(arm, pb, "RightLeg"), _bone_z(arm, pb, "LeftLeg")) if z is not None]
    knee_z = min(_kz) if _kz else (hip_z - 0.45)
    crotch_z = hip_z - 0.16
    z_wb_top = hip_z + waist_above
    z_wb_bot = z_wb_top - wb_h
    z_hem = (1.0 - hem_blend) * knee_z + hem_blend * hip_z
    z_hem = min(z_hem, crotch_z - 0.14)
    z_hem = max(z_hem, knee_z + 0.10)

    # REST pose world verts of the collision/render body
    prev_pp = arm.data.pose_position
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    try:
        donor = collider if collider is not None else body
        pts = _torso_world_verts(body)
        # Do not clip the body's own hip width; only drop obvious arm outliers.
        max_abs_x = float(gcfg.get("max_abs_x", 0.28))
        pts = pts[np.abs(pts[:, 0]) <= max_abs_x]
        if len(pts) < 64:
            pts = _torso_world_verts(body)
        half = 0.036
        z_wide, hip_r = _widest_z(pts, crotch_z + 0.02, z_wb_bot - 0.01)
        z_mid = 0.5 * (z_wb_bot + crotch_z)
        z_seat = crotch_z + 0.018
        torso_spec = [
            (z_wb_top, ease_waist + wb_extra),
            (z_wb_bot, ease_waist + wb_extra),
        ]
        for z, e in ((z_mid, ease_hip), (z_wide, ease_hip), (z_seat, ease_hip)):
            if all(abs(z - z0) > 0.018 for z0, _e in torso_spec):
                torso_spec.append((z, e))
        z_hip = z_wide
        torso_rings = []
        for z, e in torso_spec:
            r = _slice_hull(pts, z, half, n=n, ease=e)
            if r is None:
                raise RuntimeError(f"garment: empty hull at z={z:.3f}")
            torso_rings.append(r)
        thigh_zs = [crotch_z - 0.008, 0.5 * (crotch_z + z_hem), z_hem]
        left_rings, right_rings = [], []
        for z in thigh_zs:
            rl = _slice_hull(pts, z, half, n=n_leg, side="L", ease=ease_thigh)
            rr = _slice_hull(pts, z, half, n=n_leg, side="R", ease=ease_thigh)
            if rl is None or rr is None:
                raise RuntimeError(f"garment: empty thigh hull at z={z:.3f}")
            left_rings.append(rl)
            right_rings.append(rr)
        log(f"garment rings waist={z_wb_top:.3f}/{z_wb_bot:.3f} hip={z_hip:.3f} "
            f"(r={hip_r:.3f}) seat={z_seat:.3f} hem={z_hem:.3f} "
            f"ease_w/h/t={ease_waist}/{ease_hip}/{ease_thigh}")

        bm = bmesh.new()
        t_loops = _loft_into(bm, torso_rings)
        l_loops = _loft_into(bm, left_rings)
        r_loops = _loft_into(bm, right_rings)
        n_flange = _add_flange(bm, t_loops[-1], l_loops[0], r_loops[0])
        n_bridge = _crotch_bridge(bm, l_loops[0], r_loops[0], drop=0.026)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        loops = _boundary_loops(bm)
        loops_sorted = sorted(loops, key=len, reverse=True)
        filled = 0
        for loop in loops_sorted[3:]:
            if 3 <= len(loop) < 24:
                try:
                    bmesh.ops.holes_fill(bm, edges=loop, sides=len(loop) + 2)
                    filled += 1
                except Exception:
                    pass
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        open_stats = {
            "boundary_edges": sum(1 for e in bm.edges if e.is_boundary),
            "boundary_loops": len(_boundary_loops(bm)),
            "flange_faces": n_flange,
            "bridge_faces": n_bridge,
            "filled_small_holes": filled,
        }

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
        garm.matrix_world = __import__("mathutils").Matrix.Identity(4)

        # pin group = waistband ring
        vco = np.empty(len(garm.data.vertices) * 3, np.float64)
        garm.data.vertices.foreach_get("co", vco)
        vco = vco.reshape(-1, 3)
        pin = garm.vertex_groups.new(name="waistband")
        n_pin = 0
        for i, z in enumerate(vco[:, 2]):
            if z >= (z_wb_bot - 0.012):
                pin.add([int(i)], 1.0, "REPLACE")
                n_pin += 1

        # slight outward hem lip
        hem = vco[:, 2] <= (z_hem + 0.014)
        if np.any(hem):
            xy = vco[hem][:, :2]
            cxy = xy.mean(0)
            rad = xy - cxy
            rn = rad / (np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9)
            vco[hem, 0:2] = xy + rn * 0.003
            garm.data.vertices.foreach_set("co", vco.ravel())
            garm.data.update()

        cloth_report = {"attempted": False, "ok": False, "applied": False}
        if want_cloth:
            cloth_report = _try_cloth(garm, donor, "waistband", cloth_frames, cloth_shrink, log,
                                      collide_dist=collide_dist)
        else:
            log("kit cloth skipped (method does not request it)")

        _push_clearance(garm, pts, min_clear=min_clear, log=log)
        _close_mesh_holes(garm, keep_open=3, weld_dist=0.00025, log=log)
        _solidify_close(garm, thick, smooth_n, log)
        closed = _seal_boundary(garm, log)

        mat = _matte_black(rgb, rough)
        garm.data.materials.clear()
        garm.data.materials.append(mat)
        for p in garm.data.polygons:
            p.use_smooth = True

        # separate thicker elastic waistband ring
        waist = None
        try:
            wb_rings = [
                _slice_hull(pts, z_wb_top, half, n=n, ease=ease + wb_extra),
                _slice_hull(pts, z_wb_bot, half, n=n, ease=ease + wb_extra),
            ]
            if any(r is None for r in wb_rings):
                raise RuntimeError("empty waistband hull")
            bm_w = bmesh.new()
            _loft_into(bm_w, wb_rings)
            bmesh.ops.recalc_face_normals(bm_w, faces=bm_w.faces)
            wme = bpy.data.meshes.new("boxer_waistband")
            bm_w.to_mesh(wme)
            bm_w.free()
            waist = bpy.data.objects.new("briefs_waistband", wme)
            try:
                col.objects.link(waist)
            except RuntimeError:
                pass
            waist.matrix_world = __import__("mathutils").Matrix.Identity(4)
            _solidify_close(waist, thick_wb, max(2, smooth_n // 2), log)
            wmat = _matte_black(rgb, min(0.9, rough))
            wmat.name = "briefs_waistband_black"
            if "Sheen Weight" in wmat.node_tree.nodes["Principled BSDF"].inputs:
                wmat.node_tree.nodes["Principled BSDF"].inputs["Sheen Weight"].default_value = 0.12
            waist.data.materials.clear()
            waist.data.materials.append(wmat)
            for p in waist.data.polygons:
                p.use_smooth = True
            _bind_to_body(waist, body, arm, log)
        except Exception as e:
            log(f"kit waistband failed: {e}")
            waist = None

        inf_grp = garm.vertex_groups.new(name="pose_inflate")
        v_inf = np.empty(len(garm.data.vertices) * 3, np.float64)
        garm.data.vertices.foreach_get("co", v_inf)
        v_inf = v_inf.reshape(-1, 3)
        for i, zi in enumerate(v_inf[:, 2]):
            if zi >= (z_hem + 0.05):
                w = 1.0
            elif zi <= z_hem:
                w = 0.0
            else:
                w = float((zi - z_hem) / 0.05)
            inf_grp.add([int(i)], w, "REPLACE")
        _bind_to_body._inflate_m = float(gcfg.get("pose_inflate_m", 0.010))
        _bind_to_body(garm, body, arm, log)
    finally:
        arm.data.pose_position = prev_pp
        bpy.context.view_layer.update()

    meta = {
        "verts": len(garm.data.vertices),
        "faces": len(garm.data.polygons),
        "z_wb_top": round(z_wb_top, 3),
        "z_wb_bot": round(z_wb_bot, 3),
        "z_crotch": round(crotch_z, 3),
        "z_hem": round(z_hem, 3),
        "ease_m": ease,
        "method": "cloth" if cloth_report.get("applied") else "closed_tube",
        "open_stats": open_stats,
        "closed_topo": closed,
        "cloth": cloth_report,
        "pin_verts": n_pin,
        "waistband": waist.name if waist is not None else None,
        "waistband_topo": topology_stats(waist.data) if waist is not None else None,
        "parent": "data_transfer+armature",
    }
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        eg = garm.evaluated_get(dg)
        m = eg.to_mesh()
        _co = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", _co)
        _Mw = np.array(eg.matrix_world)
        _pts = _co.reshape(-1, 3) @ _Mw[:3, :3].T + _Mw[:3, 3]
        eg.to_mesh_clear()
        meta["world_bbox"] = [_pts.min(0).round(3).tolist(), _pts.max(0).round(3).tolist()]
    except Exception as e:
        meta["world_bbox"] = str(e)
    log(f"kit garment closed verts={meta['verts']} bound={closed['boundary_edges']} "
        f"hem_z={meta['z_hem']} cloth_applied={cloth_report.get('applied')} "
        f"bbox={meta.get('world_bbox')}")
    meta["waistband_obj"] = waist
    return garm, meta
