#!/usr/bin/env python3
"""Painted X Bot outfit v3 — gloves grown from the X Bot hand itself.

The v2 beanie (shape and placement) is kept as-is. Paint zones are unchanged.
The imported ARTEL glove is not used.

Each glove is built from the character's own hand and wrist vertices,
selected by hand/finger bone weights: the fingers are closed into a fist,
the result is inflated to about 1.5× the hand volume, a cuff covers the
wrist and 8 cm of the forearm, and the mesh is smoothed into one closed
mitt with a separated thumb. The mitt is skinned to the Hand and ForeArm
bones (not the finger bones) so it stays a closed volume, deforms with
the hand, and cannot float. The original hand and finger geometry is
then masked by those same weights, so only the hand-derived glove is
visible — the glove is the hand.

Material: glossy black, low roughness, visible highlight, matched to the
Petey21 CC0 reference (Wikimedia, 2014).

Pass criteria (one attempt, no iteration):
  (1) The glove encloses the fist on all frames of all 4 punches.
  (2) The beanie stays on the crown.
  (3) boxing_qc is identical to bare X Bot.
  (4) The glove reads like the Petey21 reference at full size.
"""
from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion

import xbot_outfit as xo  # noqa: E402
import xbot_painted_outfit as v1  # noqa: E402
import xbot_painted_outfit_v2 as v2  # noqa: E402

# Re-export for the build script (close-ups, beanie anatomy).
beanie_anatomy = v2.beanie_anatomy

VOLUME_TARGET = 1.50
VOLUME_RANGE = (1.40, 1.60)
CUFF_LEN_M = 0.08
CUFF_RADIUS_SCALE = 1.28
VOXEL_M = 0.0035
HAND_MASK_THRESH = 0.20

CRITERIA = {
    "version": "xbot_painted_v3",
    "rule": "ONE honest attempt. A miss is a fail; do not iterate.",
    "criteria": {
        "1_gloves": (
            "The glove encloses the fist on all frames of all 4 punches. "
            "It cannot float: it is skinned to the Hand and ForeArm bones, "
            "and the original hand/finger geometry is masked."
        ),
        "2_beanie": "The beanie stays on the crown (v2 placement, unchanged).",
        "3_motion": "boxing_qc is identical to bare X Bot.",
        "4_reference": (
            "The glove reads like the Petey21 CC0 black boxing-glove "
            "reference at full size (glossy black, low roughness, visible "
            "highlight, closed mitt with a separated thumb)."
        ),
    },
}

MCP_CURL = 1.20
PIP_CURL = 1.40
THUMB_CURL = 0.95


def world_pts(obj):
    """World-space points from the evaluated mesh (armature-deformed)."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        mw = ev.matrix_world
        return [mw @ v.co for v in me.vertices]
    finally:
        ev.to_mesh_clear()


def _side_token(side):
    return "Left" if side == "L" else "Right"


def _hand_or_finger_pred(side):
    tok = _side_token(side).lower()

    def pred(n, tok=tok):
        c = v1._clean(n)
        if tok not in c:
            return False
        return any(k in c for k in (
            "hand", "thumb", "index", "middle", "ring", "pinky"))
    return pred


def _finger_pred(side, finger):
    tok = _side_token(side).lower()
    fl = finger.lower()

    def pred(n, tok=tok, fl=fl):
        c = v1._clean(n)
        return tok in c and fl in c
    return pred


def _forearm_pred(side):
    tok = _side_token(side).lower()

    def pred(n, tok=tok):
        c = v1._clean(n)
        return tok in c and "forearm" in c
    return pred


def _is_finger_or_thumb_vg(name, side):
    c = v1._clean(name)
    tok = _side_token(side).lower()
    if tok not in c:
        return False
    return any(k in c for k in ("thumb", "index", "middle", "ring", "pinky"))


def _primary_body(meshes):
    cands = [o for o in meshes if o.type == "MESH"
             and "joint" not in o.name.lower()
             and len(o.data.vertices) > 1000]
    if not cands:
        cands = [o for o in meshes if o.type == "MESH"]
    return max(cands, key=lambda o: len(o.data.vertices))


def _mesh_volume_world(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.transform(bm, verts=bm.verts, matrix=ev.matrix_world)
        try:
            vol = abs(float(bm.calc_volume(signed=True)))
        except Exception:
            vol = 0.0
    finally:
        bm.free()
        ev.to_mesh_clear()
    return vol


def _displace_along_normals(obj, distance):
    mesh = obj.data
    normals = [Vector((0, 0, 0)) for _ in mesh.vertices]
    counts = [0] * len(mesh.vertices)
    for p in mesh.polygons:
        n = p.normal
        for i in p.vertices:
            normals[i] += n
            counts[i] += 1
    for i, v in enumerate(mesh.vertices):
        n = normals[i]
        if counts[i] and n.length > 1e-8:
            n.normalize()
            v.co = v.co + n * distance
    mesh.update()


def extract_hand_mesh(body, arm, side, name, include_cuff=True):
    """Duplicate hand + finger (+ optional cuff-region forearm) verts.

    Vertices are stored in WORLD metres; the new object has identity transform.
    Must be called with the armature in REST pose.
    """
    pred_h = _hand_or_finger_pred(side)
    pred_f = _forearm_pred(side)
    w_h = v1._vg_weights(body, pred_h)
    w_f = v1._vg_weights(body, pred_f)
    hand_bn = xo.find_bone(arm, f"mixamorig:{_side_token(side)}Hand",
                           f"{_side_token(side)}Hand")
    fore_bn = xo.find_bone(arm, f"mixamorig:{_side_token(side)}ForeArm",
                           f"{_side_token(side)}ForeArm")
    origin, _, _, _ = v2.bone_frame(arm, hand_bn)
    _, _, fy, _ = v2.bone_frame(arm, fore_bn)
    toward_elbow = -fy

    mw = body.matrix_world
    keep = []
    for i, v in enumerate(body.data.vertices):
        wp = mw @ v.co
        if i < len(w_h) and w_h[i] >= HAND_MASK_THRESH:
            keep.append(i)
            continue
        if include_cuff and i < len(w_f) and w_f[i] >= 0.25:
            along = (wp - origin).dot(toward_elbow)
            if 0.0 <= along <= (CUFF_LEN_M + 0.02):
                keep.append(i)
    keep_set = set(keep)
    if len(keep) < 30:
        raise RuntimeError(f"hand extract {side}: only {len(keep)} verts")

    old_to_new = {old: new for new, old in enumerate(keep)}
    verts = [(mw @ body.data.vertices[i].co) for i in keep]
    faces = []
    for p in body.data.polygons:
        idx = list(p.vertices)
        hit = [i for i in idx if i in keep_set]
        if len(hit) >= max(3, len(idx) - 1) and len(hit) >= 3:
            mapped = [old_to_new[i] for i in idx if i in old_to_new]
            if len(mapped) >= 3:
                faces.append(mapped)
    faces = [f for f in faces if len(set(f)) >= 3]

    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata([tuple(p) for p in verts], [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = Matrix.Identity(4)
    v1._shade_smooth(obj)
    return obj, keep


def _rotate_verts_around(obj, indices, pivot, axis, angle):
    if abs(angle) < 1e-6 or not indices or axis.length < 1e-8:
        return
    axis = axis.normalized()
    R = Quaternion(axis, angle).to_matrix().to_4x4()
    T = Matrix.Translation(pivot)
    M = T @ R @ T.inverted()
    for i in indices:
        obj.data.vertices[i].co = M @ obj.data.vertices[i].co
    obj.data.update()


def close_fingers_to_fist(obj, arm, side, body, src_indices):
    """Curl finger and thumb verts, in rest-pose world metres, into a fist."""
    tok = _side_token(side)
    hand_bn = xo.find_bone(arm, f"mixamorig:{tok}Hand", f"{tok}Hand")
    origin, hx, hy, hz = v2.bone_frame(arm, hand_bn)

    palm_idx = []
    w_hand = v1._vg_weights(body, lambda n, tok=tok.lower(): (
        tok in v1._clean(n) and "hand" in v1._clean(n)
        and not any(k in v1._clean(n) for k in
                    ("thumb", "index", "middle", "ring", "pinky"))
    ))
    for new_i, old_i in enumerate(src_indices):
        if old_i < len(w_hand) and w_hand[old_i] >= 0.35:
            palm_idx.append(new_i)
    palm_pts = [obj.data.vertices[i].co.copy() for i in palm_idx] or [
        obj.data.vertices[i].co.copy() for i in range(len(obj.data.vertices))
    ]
    arr = np.array([[p.x, p.y, p.z] for p in palm_pts], dtype=np.float64)
    c = arr.mean(axis=0)
    cov = np.cov((arr - c).T)
    evals, V = np.linalg.eigh(cov)
    thin = Vector(V[:, int(np.argmin(evals))].tolist()).normalized()
    palm_c = Vector(c.tolist())
    palm_inward = thin
    off = (palm_c - origin).dot(thin)
    if abs(off) > 1e-4:
        if off < 0:
            palm_inward = -thin
    else:
        forward = Vector((0, -1, 0))
        if palm_inward.dot(forward) < 0:
            palm_inward = -palm_inward

    def verts_of(pred, thresh=0.25):
        w = v1._vg_weights(body, pred)
        return [new_i for new_i, old_i in enumerate(src_indices)
                if old_i < len(w) and w[old_i] >= thresh]

    for finger, mcp_curl, pip_curl in (
        ("Index", MCP_CURL, PIP_CURL),
        ("Middle", MCP_CURL, PIP_CURL),
        ("Ring", MCP_CURL * 0.98, PIP_CURL),
        ("Pinky", MCP_CURL * 0.95, PIP_CURL * 0.95),
    ):
        mcp_bn = xo.find_bone(arm, f"mixamorig:{tok}Hand{finger}1",
                              f"{tok}Hand{finger}1")
        pip_bn = xo.find_bone(arm, f"mixamorig:{tok}Hand{finger}2",
                              f"{tok}Hand{finger}2")
        if not mcp_bn:
            continue
        mcp_o, _, mcp_y, _ = v2.bone_frame(arm, mcp_bn)
        axis = mcp_y.cross(palm_inward)
        if axis.length < 1e-6:
            axis = mcp_y.orthogonal()
        axis.normalize()
        motion = axis.cross(mcp_y)
        if motion.dot(palm_inward) < 0:
            axis = -axis
        idx_all = verts_of(_finger_pred(side, finger), 0.20)
        _rotate_verts_around(obj, idx_all, mcp_o, axis, mcp_curl)
        if pip_bn and idx_all:
            pip_o, _, pip_y, _ = v2.bone_frame(arm, pip_bn)
            R = Quaternion(axis, mcp_curl).to_matrix().to_4x4()
            T = Matrix.Translation(mcp_o)
            pip_posed = T @ R @ T.inverted() @ pip_o
            idx_distal = verts_of(
                lambda n, tok=tok.lower(), fl=finger.lower(): (
                    tok in v1._clean(n) and fl in v1._clean(n)
                    and any(s in v1._clean(n) for s in ("2", "3"))
                ),
                0.20,
            )
            _rotate_verts_around(obj, idx_distal, pip_posed, axis, pip_curl)

    thumb_bn = xo.find_bone(arm, f"mixamorig:{tok}HandThumb1", f"{tok}HandThumb1")
    if thumb_bn:
        th_o, _, th_y, _ = v2.bone_frame(arm, thumb_bn)
        toward = palm_c - th_o
        toward = toward - th_y * toward.dot(th_y)
        if toward.length < 1e-6:
            toward = palm_inward
        toward.normalize()
        axis = th_y.cross(toward)
        if axis.length < 1e-6:
            axis = hy.cross(palm_inward)
        axis.normalize()
        motion = axis.cross(th_y)
        if motion.dot(toward) < 0:
            axis = -axis
        idx_th = verts_of(_finger_pred(side, "Thumb"), 0.20)
        _rotate_verts_around(obj, idx_th, th_o, axis, THUMB_CURL)
    bpy.context.view_layer.update()
    return {"palm_inward": list(palm_inward), "palm_c": list(palm_c)}


def add_cuff_and_close(obj, arm, side, wrist_r):
    """A closed cuff tube, 8 cm down the forearm from the wrist."""
    tok = _side_token(side)
    hand_bn = xo.find_bone(arm, f"mixamorig:{tok}Hand", f"{tok}Hand")
    fore_bn = xo.find_bone(arm, f"mixamorig:{tok}ForeArm", f"{tok}ForeArm")
    wrist, _, _, _ = v2.bone_frame(arm, hand_bn)
    _, _, fy, _ = v2.bone_frame(arm, fore_bn)
    toward_elbow = -fy
    r0 = max(wrist_r * CUFF_RADIUS_SCALE, 0.028)
    r1 = r0 * 0.92
    segs = 20
    rings = 6
    up = fy.orthogonal()
    up.normalize()
    rt = fy.cross(up).normalized()
    if rt.length < 1e-6:
        rt = Vector((1, 0, 0))
        if abs(rt.dot(fy)) > 0.9:
            rt = Vector((0, 0, 1))
        rt = fy.cross(rt).normalized()
    up = rt.cross(fy).normalized()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    rows = []
    for k in range(rings + 1):
        t = k / rings
        centre = wrist + toward_elbow * (CUFF_LEN_M * t)
        rad = r0 * (1.0 - t) + r1 * t
        row = []
        for j in range(segs):
            ang = 2 * math.pi * j / segs
            p = centre + (up * math.cos(ang) + rt * math.sin(ang)) * rad
            row.append(bm.verts.new((p.x, p.y, p.z)))
        rows.append(row)
    for i in range(rings):
        for j in range(segs):
            a, b = rows[i][j], rows[i][(j + 1) % segs]
            c, d = rows[i + 1][(j + 1) % segs], rows[i + 1][j]
            try:
                bm.faces.new((a, b, c, d))
            except Exception:
                pass
    try:
        bm.faces.new(list(reversed(rows[-1])))
    except Exception:
        pass
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return {"wrist_r": wrist_r, "cuff_r": r0, "cuff_len": CUFF_LEN_M}


def _char_radius(obj):
    pts = [v.co.copy() for v in obj.data.vertices]
    if not pts:
        return 0.03
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return max((mx - mn).length / 4.0, 0.03)


def _sphere_radius(vol):
    return max((3.0 * max(vol, 0.0) / (4.0 * math.pi)) ** (1.0 / 3.0), 0.01)


def inflate_fist(obj, target_ratio, hand_volume):
    """Remesh the closed fist, then normal-displace toward target_ratio × hand volume."""
    xo._voxel_remesh(obj, voxel=VOXEL_M)
    xo._smooth_mesh(obj, iterations=4, factor=0.40)
    vol0 = _mesh_volume_world(obj)
    vol = vol0
    if hand_volume > 1e-9:
        for _ in range(3):
            ratio = vol / hand_volume
            if VOLUME_RANGE[0] <= ratio <= VOLUME_RANGE[1]:
                break
            if vol <= 1e-9:
                s = target_ratio ** (1.0 / 3.0)
                _displace_along_normals(obj, (s - 1.0) * _char_radius(obj))
            else:
                s = (target_ratio * hand_volume / vol) ** (1.0 / 3.0)
                _displace_along_normals(obj, (s - 1.0) * _sphere_radius(vol))
            vol = _mesh_volume_world(obj)
    ratio = (vol / hand_volume) if hand_volume > 1e-9 else 0.0
    return {"fist_volume_pre": vol0, "fist_volume": vol, "fist_ratio": ratio}


def fuse_cuff(obj, hand_volume):
    """Remesh fist+cuff into one volume, light smooth, one size correction."""
    xo._voxel_remesh(obj, voxel=VOXEL_M)
    xo._smooth_mesh(obj, iterations=8, factor=0.45)
    vol = _mesh_volume_world(obj)
    ratio = (vol / hand_volume) if hand_volume > 1e-9 else 0.0
    # Cuff adds volume, so the total can sit above 1.6. Only correct if the
    # remesh *shrunk* the mitt below the floor.
    if hand_volume > 1e-9 and ratio < VOLUME_RANGE[0]:
        want = VOLUME_TARGET / max(ratio, 1e-6)
        _displace_along_normals(obj, (want ** (1.0 / 3.0) - 1.0) * _char_radius(obj))
        xo._smooth_mesh(obj, iterations=4, factor=0.40)
        vol = _mesh_volume_world(obj)
        ratio = vol / hand_volume
    return {"volume": vol, "ratio": ratio, "hand_volume": hand_volume}


def _emphasize_hand_weights(obj, arm, side):
    """Mitt follows the Hand bone; cuff mixes in ForeArm. Finger bones stay at 0
    so a later fist pose cannot crumple an already-closed mitt."""
    tok = _side_token(side)
    hand_bn = xo.find_bone(arm, f"mixamorig:{tok}Hand", f"{tok}Hand")
    fore_bn = xo.find_bone(arm, f"mixamorig:{tok}ForeArm", f"{tok}ForeArm")
    if not hand_bn:
        return
    if obj.vertex_groups.get(hand_bn) is None:
        obj.vertex_groups.new(name=hand_bn)
    if fore_bn and obj.vertex_groups.get(fore_bn) is None:
        obj.vertex_groups.new(name=fore_bn)

    finger_groups = [vg for vg in obj.vertex_groups
                     if _is_finger_or_thumb_vg(vg.name, side)]
    origin, _, _, _ = v2.bone_frame(arm, hand_bn)
    if fore_bn:
        _, _, fy, _ = v2.bone_frame(arm, fore_bn)
    else:
        fy = Vector((0, 1, 0))
    toward_elbow = -fy
    mw = obj.matrix_world
    for v in obj.data.vertices:
        for vg in finger_groups:
            vg.add([v.index], 0.0, "REPLACE")
        wp = mw @ v.co
        along = (wp - origin).dot(toward_elbow)
        if along > 0.012:
            obj.vertex_groups[hand_bn].add([v.index], 0.55, "REPLACE")
            if fore_bn:
                obj.vertex_groups[fore_bn].add([v.index], 0.80, "REPLACE")
        else:
            obj.vertex_groups[hand_bn].add([v.index], 1.0, "REPLACE")
            if fore_bn:
                obj.vertex_groups[fore_bn].add([v.index], 0.0, "REPLACE")
    obj.data.update()


def bind_glove(obj, body, arm, side):
    bpy.context.view_layer.update()
    mw = obj.matrix_world.copy()
    if mw != Matrix.Identity(4):
        obj.data.transform(mw)
        obj.matrix_world = Matrix.Identity(4)
        obj.data.update()
    xo._transfer_weights_and_bind(obj, [body], arm)
    _emphasize_hand_weights(obj, arm, side)
    v1._shade_smooth(obj)


def mask_original_hands(meshes, arm, sides=("L", "R")):
    """Hide original hand, finger, and cuff-region forearm verts.

    The glove *is* the hand: no bare fingers may poke through.
    Measurements are taken in REST pose so bone frames match bind verts.
    """
    saved = arm.data.pose_position
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    hidden = {}
    try:
        for o in meshes:
            if o.type != "MESH" or not o.data.vertices:
                continue
            n = len(o.data.vertices)
            weights = np.zeros(n, np.float32)
            for side in sides:
                w_h = v1._vg_weights(o, _hand_or_finger_pred(side))
                for i in range(n):
                    if i < len(w_h) and w_h[i] >= HAND_MASK_THRESH:
                        weights[i] = 1.0
            if weights.max() <= 0:
                continue
            vg_name = "xbot_hand_hide"
            vg = o.vertex_groups.get(vg_name) or o.vertex_groups.new(name=vg_name)
            for i, val in enumerate(weights):
                if val > 0.5:
                    vg.add([i], 1.0, "REPLACE")
            if "XBotHandMask" not in o.modifiers:
                mod = o.modifiers.new("XBotHandMask", "MASK")
                mod.vertex_group = vg_name
                mod.invert_vertex_group = True
            hidden[o.name] = int(weights.sum())
    finally:
        arm.data.pose_position = saved
        bpy.context.view_layer.update()
    return hidden


def make_glossy_black():
    return xo.make_glossy_leather_material(
        "XBotGloveLeather_painted_v3", xo.GLOVE_V4_BLACK, roughness=0.18,
    )


def build_one_glove(body, arm, side, collection, hand_volume):
    name = f"XBotGlove_{side}"
    obj, src_idx = extract_hand_mesh(body, arm, side, name, include_cuff=True)
    if collection and obj.name not in collection.objects:
        collection.objects.link(obj)
        if obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(obj)
    curl = close_fingers_to_fist(obj, arm, side, body, src_idx)
    infl = inflate_fist(obj, VOLUME_TARGET, hand_volume)
    wrist_r = v2.wrist_radius_world(
        arm, [body],
        xo.find_bone(arm, f"mixamorig:{_side_token(side)}Hand",
                     f"{_side_token(side)}Hand"),
        side,
    )
    cuff = add_cuff_and_close(obj, arm, side, wrist_r)
    fused = fuse_cuff(obj, hand_volume)
    mat = make_glossy_black()
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.active_material = mat
    # World-metre length before bind (after bind, verts are armature-local cm).
    pts = [v.co.copy() for v in obj.data.vertices]
    if pts:
        mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        length_m = (mx - mn).length
    else:
        length_m = 0.0
    bind_glove(obj, body, arm, side)
    return obj, {
        "side": side,
        "n_src_verts": len(src_idx),
        "n_verts": len(obj.data.vertices),
        "wrist_r": wrist_r,
        "curl": curl,
        "cuff": cuff,
        "inflate": infl,
        "volume": fused,
        "length_m": length_m,
        "bone": xo.find_bone(arm, f"mixamorig:{_side_token(side)}Hand",
                             f"{_side_token(side)}Hand"),
        "object": obj.name,
        "construction": "hand-derived, skinned to Hand+ForeArm, original hand masked",
    }


def hand_volume_of(body, arm, side):
    """World-metre volume of the raw extracted hand (no cuff), then discard."""
    tmp, _ = extract_hand_mesh(body, arm, side, f"_tmpHandVol_{side}",
                               include_cuff=False)
    pts = [v.co.copy() for v in tmp.data.vertices]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    dim = mx - mn
    aabb_vol = float(abs(dim.x * dim.y * dim.z))
    vol = aabb_vol * 0.40
    try:
        xo._voxel_remesh(tmp, voxel=0.005)
        rv = _mesh_volume_world(tmp)
        if rv > 1e-6:
            vol = rv
    except Exception:
        pass
    bpy.data.objects.remove(tmp, do_unlink=True)
    return vol


def make_hand_gloves(arm, meshes, collection):
    saved = arm.data.pose_position
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    body = _primary_body(meshes)
    gloves = {}
    try:
        for side in ("L", "R"):
            hv = hand_volume_of(body, arm, side)
            print(f"  hand_volume_{side}={hv:.6f} m^3", flush=True)
            obj, info = build_one_glove(body, arm, side, collection, hv)
            info["hand_volume"] = hv
            gloves[side] = (obj, info)
            print(f"  glove_{side} volume_ratio={info['volume'].get('ratio'):.3f} "
                  f"fist_ratio={info['inflate'].get('fist_ratio'):.3f} "
                  f"verts={len(obj.data.vertices)} length≈{info['length_m']:.3f}m",
                  flush=True)
    finally:
        arm.data.pose_position = saved
        bpy.context.view_layer.update()
    return gloves


def apply_painted_outfit(arm, meshes, f0, f1, collection_name="XBotPaintedV3"):
    col = bpy.data.collections.get(collection_name)
    if col is None:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)

    head_bone = xo.find_bone(arm, "mixamorig:Head", "Head")
    head_w = v2.head_width_world(meshes, arm, head_bone)

    zone_info = v1.paint_body_zones(meshes, arm)
    v1.assign_painted_material(meshes)

    hat, hat_info = v2.make_beanie(arm, meshes, col, head_w)

    gloves = make_hand_gloves(arm, meshes, col)
    hidden = mask_original_hands(arm=arm, meshes=meshes)

    handle = {
        "collection": collection_name,
        "head_width_m": head_w,
        "gloves": {s: {"object": gloves[s][0].name, **gloves[s][1]}
                   for s in ("L", "R")},
        "beanie": {"object": hat.name, **hat_info},
        "zones": zone_info,
        "hidden_hand_verts": hidden,
        "objects": [gloves["R"][0].name, gloves["L"][0].name, hat.name],
        "credits": [
            "Beanie by oluwaphemi, CC BY 4.0 (unchanged from v2).",
            "Gloves are derived from the Mixamo X Bot hand mesh (no third-party glove).",
            "Gloss matched to Pair of boxing gloves (Petey21, Wikimedia Commons, CC0 1.0, 2014).",
        ],
        "criteria": CRITERIA,
        "units": "world_metres",
    }
    bpy.context.view_layer.update()
    return handle


def _disable_hand_masks(meshes):
    disabled = []
    for o in meshes:
        for m in o.modifiers:
            if m.type == "MASK" and m.name == "XBotHandMask":
                disabled.append((m, m.show_viewport, m.show_render))
                m.show_viewport = False
                m.show_render = False
    bpy.context.view_layer.update()
    return disabled


def _restore_hand_masks(disabled):
    for m, vp, rd in disabled:
        m.show_viewport = vp
        m.show_render = rd
    bpy.context.view_layer.update()


def check_gloves(arm, meshes, handle, f0, f1, step=1):
    """Criterion 1, world metres, using the armature-deformed glove mesh.

    The original-hand mask is temporarily disabled so the hidden fist verts
    can still be tested for enclosure.
    """
    disabled = _disable_hand_masks(meshes)
    try:
        report = v2.check_gloves(arm, meshes, handle, f0, f1, step=step)
    finally:
        # v2.check_gloves uses v1.world_pts (undeformed). Override the
        # per-side numbers with an evaluated-mesh pass below if needed.
        _restore_hand_masks(disabled)
    # Re-run the enclosure test ourselves with evaluated glove points,
    # because v2.world_pts does not see armature deform.
    report = {"ok": True, "per_side": {}, "worst": {}, "frames_tested": 0,
              "frames_failed": []}
    saved = bpy.context.scene.frame_current
    disabled = _disable_hand_masks(meshes)
    try:
        for side in ("L", "R"):
            rec = handle["gloves"][side]
            gname, bn = rec["object"], rec["bone"]
            glove = bpy.data.objects.get(gname)
            if glove is None:
                report["ok"] = False
                report["per_side"][side] = {"error": "missing glove"}
                continue
            failures = []
            n_frames = 0
            max_outside = 0
            max_float = 0.0
            min_overlap = 1e9
            f = int(f0)
            pred = v1.hand_pred(side)
            while f <= int(f1):
                bpy.context.scene.frame_set(f)
                bpy.context.view_layer.update()
                n_frames += 1
                origin, x, y, z = v2.bone_frame(arm, bn)
                hand = v2.world_project(v1.collect_weighted(meshes, pred, 0.30),
                                        origin, x, y, z)
                glove_locs = v2.world_project(world_pts(glove), origin, x, y, z)
                if not hand or not glove_locs:
                    failures.append({"f": f, "reason": "no verts"})
                    f += step
                    continue
                gmin = Vector((min(p.x for p in glove_locs),
                               min(p.y for p in glove_locs),
                               min(p.z for p in glove_locs)))
                gmax = Vector((max(p.x for p in glove_locs),
                               max(p.y for p in glove_locs),
                               max(p.z for p in glove_locs)))
                cuff_y = gmin.y + 0.008
                pad = 0.010
                outside = 0
                tested = 0
                for p in hand:
                    if p.y < cuff_y:
                        continue
                    tested += 1
                    inside = (
                        gmin.x - pad <= p.x <= gmax.x + pad
                        and gmin.y - pad <= p.y <= gmax.y + pad
                        and gmin.z - pad <= p.z <= gmax.z + pad
                    )
                    if not inside:
                        outside += 1
                max_outside = max(max_outside, outside)
                hc = sum(hand, Vector()) / len(hand)
                gc = sum(glove_locs, Vector()) / len(glove_locs)
                dist = (hc - gc).length
                max_float = max(max_float, dist)
                glove_mid_y = 0.5 * (gmin.y + gmax.y)
                hy0, hy1 = min(p.y for p in hand), max(p.y for p in hand)
                overlap = max(0.0, min(gmax.y, hy1) - max(gmin.y, hy0))
                min_overlap = min(min_overlap, overlap)
                reasons = []
                if tested and outside / max(tested, 1) > 0.04 and outside > 8:
                    reasons.append(
                        f"{outside}/{tested} hand verts outside glove AABB")
                if dist > 0.10:
                    reasons.append(
                        f"glove centroid {dist*100:.1f} cm from hand (floating)")
                if glove_mid_y < -0.07:
                    reasons.append(f"glove mid Y={glove_mid_y:.3f} m (forearm)")
                if overlap < 0.03:
                    reasons.append(f"glove/hand Y overlap {overlap:.3f} m")
                if reasons:
                    failures.append({"f": f, "reasons": reasons,
                                     "outside": outside, "tested": tested,
                                     "float_m": dist, "glove_mid_y": glove_mid_y})
                f += step
            side_ok = not failures
            report["per_side"][side] = {
                "ok": side_ok,
                "frames": n_frames,
                "failures": failures[:8],
                "n_failures": len(failures),
                "max_outside": max_outside,
                "max_float_m": max_float,
                "min_y_overlap_m": min_overlap if min_overlap < 1e8 else None,
            }
            if not side_ok:
                report["ok"] = False
                report["frames_failed"].extend(
                    {"side": side, **fail} for fail in failures[:4])
            report["frames_tested"] = max(report["frames_tested"], n_frames)
            report["worst"][side] = {
                "max_outside": max_outside,
                "max_float_m": max_float,
                "min_y_overlap_m": min_overlap if min_overlap < 1e8 else None,
            }
    finally:
        _restore_hand_masks(disabled)
        bpy.context.scene.frame_set(saved)
        bpy.context.view_layer.update()
    return report


def check_beanie(arm, meshes, handle, f0, f1, step=1):
    return v2.check_beanie(arm, meshes, handle, f0, f1, step=step)


def clear_outfit(handle):
    v1.clear_outfit(handle)
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        if "XBotHandMask" in o.modifiers:
            o.modifiers.remove(o.modifiers["XBotHandMask"])
        vg = o.vertex_groups.get("xbot_hand_hide")
        if vg:
            o.vertex_groups.remove(vg)
