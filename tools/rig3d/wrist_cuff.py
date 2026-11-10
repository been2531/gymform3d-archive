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


def flatten_ankle_cuffs(body, log=print):
    """Shrink the modelled ankle retinaculum/band the same way as the wrist cuff.

    Does not change wrist verts or the wrist-flattening logic. Merges any new
    mask into the existing `cuff_mask` so the clay shader also drops the
    baked ankle ridge.
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
    va = me.color_attributes.get("cuff_mask")
    if va is not None:
        raw = np.empty(nv * 4, np.float32)
        va.data.foreach_get("color", raw)
        cuff = raw.reshape(nv, 4)[:, 0].copy()
    n_moved = 0

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
            log(f"ankle cuff {side}: no junction verts")
            continue
        t_j = float(np.median(t[both]))
        # Shin radius well above the ankle. A window that crosses the joint
        # treats the heel as the cuff and tears the standing foot off.
        shin_win = limb & (t >= t_j - 0.055) & (t <= t_j - 0.022) & (r < 0.050)
        if int(shin_win.sum()) < 12:
            log(f"ankle cuff {side}: no shin sample")
            continue
        r_shin = float(np.median(r[shin_win]))
        if not (0.018 <= r_shin <= 0.048):
            log(f"ankle cuff {side}: skip, shin radius looks wrong ({r_shin:.4f})")
            continue
        # Toe/heel axis: flatten the anterior + side retinaculum further
        # onto the foot dorsum, but never the heel (that tears the foot).
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
        heel = (t > t_j - 0.006) & (forwardness < -0.006) & (r > 0.042)
        t_lo, t_hi = t_j - 0.050, t_j + 0.022
        win = limb & (t >= t_lo) & (t <= t_hi) & (~heel) & (r < 0.078)
        if int(win.sum()) < 20:
            continue
        r_tgt = np.full(nv, r_shin)
        bulge = win & (r > r_shin + 0.0006) & (r < 0.078)
        if int(bulge.sum()) < 8:
            log(f"ankle cuff {side}: no bulge (r_shin={r_shin:.4f} win_n={int(win.sum())})")
            continue
        scale = np.clip(r_shin / np.maximum(r[bulge], 1e-8), 0.42, 1.0)
        disp = r[bulge] * (1.0 - scale)
        cw[bulge] = origin + np.outer(t[bulge], u) + radial[bulge] * scale[:, None]
        n_moved += int(bulge.sum())
        excess = np.clip((r - r_tgt) / 0.008, 0.0, 1.0)
        near = win & (np.abs(t - t_j) < 0.038)
        cuff[near] = np.maximum(cuff[near], np.maximum(excess[near], 0.88))
        cuff[bulge] = 1.0
        log(f"ankle cuff {side}: moved {int(bulge.sum())} verts "
            f"t=[{t_lo:.3f},{t_hi:.3f}] r_shin={r_shin:.4f} "
            f"peak={float(r[bulge].max()):.4f} heel_excl={int(heel.sum())} "
            f"mean_disp={float(disp.mean()):.4f} max_disp={float(disp.max()):.4f}")

    if n_moved:
        Ri = np.linalg.inv(R)
        new_co = (cw - T) @ Ri.T
        me.vertices.foreach_set("co", new_co.ravel())
        me.update()
        try:
            import bpy
            prev = bpy.context.view_layer.objects.active
            bpy.context.view_layer.objects.active = body
            try:
                if hasattr(bpy.ops.object, "mode_set") and bpy.ops.object.mode_set.poll():
                    bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
            except Exception as e:
                log(f"ankle cuff custom-normal clear skipped: {e}")
            for poly in me.polygons:
                poly.use_smooth = True
            if prev is not None:
                try:
                    bpy.context.view_layer.objects.active = prev
                except Exception:
                    pass
        except Exception as e:
            log(f"ankle cuff shade-smooth skipped: {e}")
    if float(cuff.max()) > 0:
        try:
            ev = np.empty(len(me.edges) * 2, np.int64)
            me.edges.foreach_get("vertices", ev)
            ev = ev.reshape(-1, 2)
            for _ in range(6):
                c2 = cuff.copy()
                np.maximum.at(c2, ev[:, 0], cuff[ev[:, 1]] * 0.78)
                np.maximum.at(c2, ev[:, 1], cuff[ev[:, 0]] * 0.78)
                cuff = np.maximum(cuff, c2)
        except Exception:
            pass
        _write_mask(me, np.clip(cuff, 0.0, 1.0))
    log(f"ankle cuff: flattened {n_moved} verts total")
    return n_moved
