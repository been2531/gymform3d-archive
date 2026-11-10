"""Flatten the raised wristband/cuff modelled into the Grab3D char1 mesh.

The asset has no separate wristband object or vertex group. Each hand-forearm
junction is a ring of verts whose radius jumps above the forearm neck, and
the same ridge is baked into `rig_normal`. Together they read as a bright
white band under the clay shader.
"""
from __future__ import annotations

import numpy as np


def _weights(me, vgi, name, nv):
    w = np.zeros(nv, np.float32)
    gi = vgi.get(name)
    if gi is None:
        return w
    for v in me.vertices:
        for g in v.groups:
            if g.group == gi:
                w[v.index] = g.weight
    return w


def _bone_axis(body, name):
    """World-space (head, tail, unit axis toward tail) of a rest bone."""
    arm = body.find_armature()
    if arm is None:
        return None
    bone = arm.data.bones.get(name)
    if bone is None:
        return None
    # Grab3D bones and mesh share rest-pose local space; the mesh object
    # carries the 0.01 world scale, the armature often does not.
    M = np.array(body.matrix_world, dtype=np.float64)
    h = np.array(bone.head_local, dtype=np.float64)
    t = np.array(bone.tail_local, dtype=np.float64)
    h = M[:3, :3] @ h + M[:3, 3]
    t = M[:3, :3] @ t + M[:3, 3]
    axis = t - h
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        return None
    return h, t, axis / n, n


def _write_mask(me, w):
    col = np.zeros((len(w), 4), np.float32)
    col[:, 0] = w
    col[:, 1] = w
    col[:, 2] = w
    col[:, 3] = 1.0
    va = me.color_attributes.get("cuff_mask")
    if va is None:
        va = me.color_attributes.new("cuff_mask", "FLOAT_COLOR", "POINT")
    va.data.foreach_set("color", col.ravel())


def flatten_wrist_cuffs(body, log=print):
    """Shrink the cuff to the forearm-neck/hand valley profile and write `cuff_mask`.

    `cuff_mask` is 1 on the raised ring so the clay shader can drop the baked
    normal-map ridge. Returns the number of verts moved.
    """
    me = body.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3).copy()
    mw = np.array(body.matrix_world, dtype=np.float64)
    R, T = mw[:3, :3], mw[:3, 3]
    cw = co @ R.T + T

    vgi = {vg.name: vg.index for vg in body.vertex_groups}
    cuff = np.zeros(nv, np.float32)
    n_moved = 0

    for side in ("Left", "Right"):
        axis_info = _bone_axis(body, side + "ForeArm")
        wh = _weights(me, vgi, side + "Hand", nv)
        wf = _weights(me, vgi, side + "ForeArm", nv)
        arm = (wh > 0.12) | (wf > 0.12)
        if int(arm.sum()) < 50:
            continue
        fore = cw[wf > 0.35]
        handp = cw[wh > 0.35]
        if len(fore) < 10 or len(handp) < 10:
            continue
        origin = fore.mean(0)
        if axis_info is not None:
            u = axis_info[2]          # direction only (bone space scale differs from mesh)
        else:
            axis = handp.mean(0) - origin
            u = axis / max(float(np.linalg.norm(axis)), 1e-8)
        if float((handp.mean(0) - origin) @ u) < 0:
            u = -u
        t = (cw - origin) @ u
        radial = (cw - origin) - np.outer(t, u)
        r = np.linalg.norm(radial, axis=1)

        # Junction = verts skinned to both forearm and hand.
        both = (wh > 0.20) & (wf > 0.20)
        if int(both.sum()) < 8:
            log(f"wrist cuff {side}: no junction verts")
            continue
        t_j = float(np.median(t[both]))
        t_lo, t_hi = t_j - 0.040, t_j + 0.035
        win = arm & (t >= t_lo) & (t <= t_hi)
        if int(win.sum()) < 20:
            continue
        blen = t_j

        # Binned median radius: valley (forearm neck) — peak (cuff) — valley (hand).
        edges = np.linspace(t_lo, t_hi, 13)
        centres, meds = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = win & (t >= a) & (t < b)
            if int(m.sum()) < 6:
                continue
            centres.append(0.5 * (a + b))
            meds.append(float(np.median(r[m])))
        if len(meds) < 4:
            log(f"wrist cuff {side}: too few radius bins ({len(meds)})")
            continue
        centres = np.asarray(centres)
        meds = np.asarray(meds)
        peak_i = int(np.argmax(meds))
        left = meds[: max(peak_i, 1)]
        right = meds[peak_i + 1:] if peak_i + 1 < len(meds) else meds[-1:]
        r_left = float(np.min(left))
        r_right = float(np.min(right))
        t_left = float(centres[int(np.argmin(left))])
        t_right = float(centres[peak_i + 1 + int(np.argmin(right))] if peak_i + 1 < len(meds)
                        else centres[-1])
        if t_right <= t_left:
            t_right = t_left + 1e-3
        span = t_right - t_left
        alpha = np.clip((t - t_left) / span, 0.0, 1.0)
        r_tgt = (1.0 - alpha) * r_left + alpha * r_right
        r_tgt = np.minimum(r_tgt, min(r_left, r_right) + 0.001)
        # Anything clearly proud of the valley profile is the modelled cuff.
        bulge = win & (r > r_tgt + 0.0012)
        if not bulge.any():
            log(f"wrist cuff {side}: no bulge (r_left={r_left:.4f} r_right={r_right:.4f} "
                f"peak={float(meds[peak_i]):.4f})")
            continue
        scale = r_tgt[bulge] / np.maximum(r[bulge], 1e-8)
        cw[bulge] = origin + np.outer(t[bulge], u) + radial[bulge] * scale[:, None]
        n_moved += int(bulge.sum())

        # Broad mask over the whole junction so the visible band is covered,
        # not only the verts that happened to exceed the radius threshold.
        excess = np.clip((r - r_tgt) / 0.008, 0.0, 1.0)
        near = win & (np.abs(t - t_j) < 0.032)
        cuff[near] = np.maximum(cuff[near], np.maximum(excess[near], 0.85))
        cuff[bulge] = 1.0
        log(f"wrist cuff {side}: moved {int(bulge.sum())} verts "
            f"t=[{t_lo:.3f},{t_hi:.3f}] bone_len={blen:.3f} "
            f"r_left={r_left:.4f} peak={float(meds[peak_i]):.4f} r_right={r_right:.4f}")

    if n_moved:
        Ri = np.linalg.inv(R)
        new_co = (cw - T) @ Ri.T
        me.vertices.foreach_set("co", new_co.ravel())
        me.update()

    # Small dilate of the mask across mesh edges so the blend covers the band lip.
    if float(cuff.max()) > 0:
        try:
            ev = np.empty(len(me.edges) * 2, np.int64)
            me.edges.foreach_get("vertices", ev)
            ev = ev.reshape(-1, 2)
            for _ in range(5):
                c2 = cuff.copy()
                np.maximum.at(c2, ev[:, 0], cuff[ev[:, 1]] * 0.75)
                np.maximum.at(c2, ev[:, 1], cuff[ev[:, 0]] * 0.75)
                cuff = np.maximum(cuff, c2)
        except Exception:
            pass
    _write_mask(me, np.clip(cuff, 0.0, 1.0))

    try:
        import bpy
        prev = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = body
        try:
            if hasattr(bpy.ops.object, "mode_set") and bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except Exception as e:
            log(f"wrist cuff custom-normal clear skipped: {e}")
        for p in me.polygons:
            p.use_smooth = True
        if prev is not None:
            try:
                bpy.context.view_layer.objects.active = prev
            except Exception:
                pass
        log(f"wrist cuff: flattened {n_moved} verts total, mask={(cuff > 0.5).sum()} "
            f"shade smooth")
    except Exception as e:
        log(f"wrist cuff finish skipped: {e}")
    return n_moved


def _boundary_edge_count(me):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    n = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()
    return n


def _ankle_side_masks(body, me, nv, cw, vgi):
    """Per-side ankle-band / heel / bulge masks in rest-pose world space."""
    region = np.zeros(nv, np.bool_)
    heel = np.zeros(nv, np.bool_)
    bulge = np.zeros(nv, np.bool_)
    axes = np.zeros((nv, 3), np.float64)
    info = []
    for side in ("Left", "Right"):
        axis_info = _bone_axis(body, side + "Leg")
        wf = _weights(me, vgi, side + "Foot", nv)
        ws = _weights(me, vgi, side + "Leg", nv)
        limb = (wf > 0.12) | (ws > 0.12)
        if int(limb.sum()) < 50:
            continue
        shin = cw[ws > 0.35]
        footp = cw[wf > 0.35]
        if len(shin) < 10 or len(footp) < 10:
            continue
        origin = shin.mean(0)
        if axis_info is not None:
            u = axis_info[2]
        else:
            axis = footp.mean(0) - origin
            u = axis / max(float(np.linalg.norm(axis)), 1e-8)
        if float((footp.mean(0) - origin) @ u) < 0:
            u = -u
        t = (cw - origin) @ u
        radial = (cw - origin) - np.outer(t, u)
        r = np.linalg.norm(radial, axis=1)
        both = (wf > 0.20) & (ws > 0.20)
        if int(both.sum()) < 8:
            continue
        t_j = float(np.median(t[both]))
        shin_win = limb & (t >= t_j - 0.055) & (t <= t_j - 0.022) & (r < 0.050)
        if int(shin_win.sum()) < 12:
            continue
        r_shin = float(np.median(r[shin_win]))
        if not (0.018 <= r_shin <= 0.048):
            continue
        foot_axis = _bone_axis(body, side + "Foot")
        if foot_axis is not None:
            fwd = np.asarray(foot_axis[2], dtype=np.float64)
        else:
            fwd = footp.mean(0) - origin
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / max(fn, 1e-8)
        if float((footp.mean(0) - origin) @ fwd) < 0:
            fwd = -fwd
        forwardness = radial @ fwd
        heel_s = (t > t_j - 0.006) & (forwardness < -0.006) & (r > 0.042)
        # True foot beyond the junction (toes / dorsum far from the band).
        far_foot = (wf > 0.70) & (ws < 0.18) & (t > t_j + 0.012)
        win = limb & (t >= t_j - 0.048) & (t <= t_j + 0.016) & (~heel_s) & (~far_foot) & (r < 0.082)
        bul = win & (r > r_shin + 0.003)
        region[win] = True
        heel[heel_s] = True
        bulge[bul] = True
        axes[win | heel_s] = u
        info.append((side, int(win.sum()), int(bul.sum()), int(heel_s.sum()), r_shin, t_j))
    return region, heel, bulge, axes, info


def flatten_ankle_cuffs(body, log=print, dissolve=True):
    """Remove the modelled extensor-retinaculum shelf by construction.

    The Grab3D char1 ankle band is a raised crease-loop cylinder, not a
    leftover garment. Radius-flattening could not collapse those loops
    without tearing the heel. Instead: dissolve the circumferential crease
    (and bulge-ring) edges on both ankles, relax the band with a corrective
    smooth, and fade the baked `rig_normal` via `cuff_mask` so the shader
    uses computed normals. Heel/foot verts stay out of the dissolve set;
    the mesh must gain zero new boundary edges.
    """
    import math
    import bmesh
    try:
        import bpy
    except Exception as e:
        log(f"ankle retinaculum: bpy missing ({e})")
        return 0

    me = body.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3).copy()
    mw = np.array(body.matrix_world, dtype=np.float64)
    R, T = mw[:3, :3], mw[:3, 3]
    cw = co @ R.T + T

    vgi = {vg.name: vg.index for vg in body.vertex_groups}
    cuff = np.zeros(nv, np.float32)
    va = me.color_attributes.get("cuff_mask")
    if va is not None:
        raw = np.empty(nv * 4, np.float32)
        va.data.foreach_get("color", raw)
        cuff = raw.reshape(nv, 4)[:, 0].copy()

    region, heel, bulge, axes, info = _ankle_side_masks(body, me, nv, cw, vgi)
    for rec in info:
        log(f"ankle retinaculum {rec[0]}: win={rec[1]} bulge={rec[2]} "
            f"heel={rec[3]} r_shin={rec[4]:.4f} t_j={rec[5]:.3f}")
    if int(region.sum()) < 20:
        log("ankle retinaculum: no band verts")
        return 0

    if not dissolve:
        cuff[region] = np.maximum(cuff[region], 1.0)
        try:
            ev = np.empty(len(me.edges) * 2, np.int64)
            me.edges.foreach_get("vertices", ev)
            ev = ev.reshape(-1, 2)
            for _ in range(4):
                c2 = cuff.copy()
                np.maximum.at(c2, ev[:, 0], cuff[ev[:, 1]] * 0.78)
                np.maximum.at(c2, ev[:, 1], cuff[ev[:, 0]] * 0.78)
                cuff = np.maximum(cuff, c2)
        except Exception:
            pass
        _write_mask(me, np.clip(cuff, 0.0, 1.0))
        log(f"ankle retinaculum: mask-only n={int(region.sum())} "
            f"(topology left intact)")
        return 0

    bound0 = _boundary_edge_count(me)
    heel_idx = np.nonzero(heel)[0]
    heel_w0 = cw[heel_idx].copy() if len(heel_idx) else np.zeros((0, 3))
    nv0 = nv

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    crease_min = math.radians(20.0)
    picked = []
    n_crease = n_ring = 0
    for e in bm.edges:
        if not e.is_manifold or len(e.link_faces) != 2:
            continue
        i0, i1 = e.verts[0].index, e.verts[1].index
        if i0 >= nv or i1 >= nv:
            continue
        if heel[i0] or heel[i1]:
            continue
        if not (region[i0] and region[i1]):
            continue
        dloc = np.array(e.verts[1].co, dtype=np.float64) - np.array(e.verts[0].co, dtype=np.float64)
        dw = R @ dloc
        dn = float(np.linalg.norm(dw))
        if dn < 1e-10:
            continue
        dw /= dn
        u = axes[i0] if np.linalg.norm(axes[i0]) > 0.5 else axes[i1]
        if np.linalg.norm(u) < 0.5:
            continue
        # Keep longitudinal edges; only dissolve loops that go around the shin.
        if abs(float(dw @ u)) > 0.42:
            continue
        ang = float(e.calc_face_angle())
        is_crease = ang >= crease_min
        is_ring = bool(bulge[i0] and bulge[i1])
        if is_crease or is_ring:
            picked.append(e)
            n_crease += int(is_crease)
            n_ring += int(is_ring and not is_crease)

    n_dissolved = 0
    if picked:
        ret = bmesh.ops.dissolve_edges(
            bm, edges=picked, use_verts=True, use_face_split=False)
        n_dissolved = len(picked)
    bm.normal_update()
    bm.to_mesh(me)
    try:
        me.validate(verbose=False, clean_customdata=True)
    except Exception:
        pass
    me.update()
    bm.free()
    log(f"ankle retinaculum: dissolved {n_dissolved} edges "
        f"(crease={n_crease} ring={n_ring}) nv {nv0}->{len(me.vertices)}")

    bound1 = _boundary_edge_count(me)
    if bound1 > bound0:
        log(f"ankle retinaculum: NEW boundary edges {bound0}->{bound1} "
            f"(will still try relax; inspect heel/foot)")
    else:
        log(f"ankle retinaculum: boundary edges {bound0}->{bound1} (no new)")

    # Re-read geometry and rebuild the relax / normal-fade mask.
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3)
    cw = co @ R.T + T
    vgi = {vg.name: vg.index for vg in body.vertex_groups}
    region2, heel2, bulge2, _axes2, _info2 = _ankle_side_masks(body, me, nv, cw, vgi)
    relax = region2 | bulge2
    # Dilate the relax set one ring so corrective smooth can blend into shin.
    try:
        ev = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        dil = relax.copy()
        dil[ev[:, 0]] |= relax[ev[:, 1]]
        dil[ev[:, 1]] |= relax[ev[:, 0]]
        relax = dil & (~heel2)
    except Exception:
        relax = relax & (~heel2)

    # Heel integrity: surviving heel verts should stay near their old positions.
    heel_idx2 = np.nonzero(heel2)[0]
    heel_move = 0.0
    heel_lost = max(0, int(len(heel_idx) - len(heel_idx2)))
    if len(heel_w0) and len(heel_idx2):
        hw = cw[heel_idx2]
        # nearest-old distance for each surviving heel vert
        dmin = np.empty(len(hw))
        for i, p in enumerate(hw):
            dmin[i] = float(np.linalg.norm(heel_w0 - p, axis=1).min())
        heel_move = float(dmin.max()) if len(dmin) else 0.0
    log(f"ankle retinaculum: heel verts {len(heel_idx)}->{len(heel_idx2)} "
        f"lost={heel_lost} max_move={heel_move*1000:.2f} mm")

    # Relax the remaining shelf with a pinned-boundary Laplacian smooth
    # (corrective-smooth equivalent that does not apply a modifier on top
    # of the armature, which segfaulted).
    n_relax = int(relax.sum())
    if n_relax >= 8:
        try:
            ev = np.empty(len(me.edges) * 2, np.int64)
            me.edges.foreach_get("vertices", ev)
            ev = ev.reshape(-1, 2)
            boundary = np.zeros(nv, np.bool_)
            boundary[ev[:, 0]] |= ~relax[ev[:, 1]]
            boundary[ev[:, 1]] |= ~relax[ev[:, 0]]
            interior = relax & (~boundary) & (~heel2)
            bm = bmesh.new()
            bm.from_mesh(me)
            bm.verts.ensure_lookup_table()
            verts = [bm.verts[i] for i in np.nonzero(interior)[0] if i < len(bm.verts)]
            if verts:
                for _ in range(12):
                    bmesh.ops.smooth_vert(
                        bm, verts=verts, factor=0.55,
                        use_axis_x=True, use_axis_y=True, use_axis_z=True)
            bm.to_mesh(me)
            try:
                me.validate(verbose=False, clean_customdata=True)
            except Exception:
                pass
            me.update()
            bm.free()
            log(f"ankle retinaculum: laplacian-relax n={len(verts)} "
                f"(pinned_boundary={int((boundary & relax).sum())})")
        except Exception as e:
            log(f"ankle retinaculum: laplacian-relax skipped: {e}")

    bound2 = _boundary_edge_count(me)
    log(f"ankle retinaculum: boundary after relax {bound1}->{bound2} "
        f"(start {bound0})")

    # Fade baked rig_normal on the (dilated) band; merge with wrist cuff_mask.
    nv = len(me.vertices)
    cuff = np.zeros(nv, np.float32)
    va = me.color_attributes.get("cuff_mask")
    if va is not None and len(va.data) == nv:
        raw = np.empty(nv * 4, np.float32)
        va.data.foreach_get("color", raw)
        cuff = raw.reshape(nv, 4)[:, 0].copy()
    # Recompute region after smooth (vert count unchanged by the modifier).
    co = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", co)
    cw = co.reshape(nv, 3) @ R.T + T
    vgi = {vg.name: vg.index for vg in body.vertex_groups}
    region3, heel3, bulge3, _a3, _i3 = _ankle_side_masks(body, me, nv, cw, vgi)
    mask = (region3 | bulge3) & (~heel3)
    cuff[mask] = np.maximum(cuff[mask], 1.0)
    try:
        ev = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        for _ in range(4):
            c2 = cuff.copy()
            np.maximum.at(c2, ev[:, 0], cuff[ev[:, 1]] * 0.78)
            np.maximum.at(c2, ev[:, 1], cuff[ev[:, 0]] * 0.78)
            cuff = np.maximum(cuff, c2)
    except Exception:
        pass
    _write_mask(me, np.clip(cuff, 0.0, 1.0))

    try:
        prev = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = body
        try:
            if hasattr(bpy.ops.object, "mode_set") and bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except Exception as e:
            log(f"ankle retinaculum: custom-normal clear skipped: {e}")
        for poly in me.polygons:
            poly.use_smooth = True
        if prev is not None:
            try:
                bpy.context.view_layer.objects.active = prev
            except Exception:
                pass
    except Exception as e:
        log(f"ankle retinaculum: shade-smooth skipped: {e}")

    log(f"ankle retinaculum: done dissolved={n_dissolved} relax={n_relax} "
        f"mask={(cuff > 0.5).sum()} bound {bound0}->{bound2} "
        f"heel_max_move_mm={heel_move*1000:.2f}")
    return n_dissolved
