#!/usr/bin/env python3
"""Soft 3D blob glow (and cheaper alternate methods) on an unrigged body.

Methods:
  blob   — a few anatomical anchors per muscle, smoothstep radial falloff in 3D
  idpass — plain grey render + 2D soft elliptical masks at projected anchors
  shells — surface patches displaced 2.5 mm along normals; only shells emit
  xray   — translucent glass skin with simple inner emission ellipsoids

Run with Blender 4.2.3:
  nice -n 19 blender -b --python tools/rig3d/eval_blob_glow.py -- \\
      --input PATH --out-dir DIR --name ID [--method blob]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval_base_model import (  # noqa: E402
    _apply_scale_to_metres,
    _hide_plinths,
    _import,
    _render,
    _reset,
    _setup_camera,
    _setup_world_lights,
    _world_dims,
    _argv,
)
from eval_muscle_colour import (  # noqa: E402
    _assign_colour_mat,
    _normalize_height,
    _world_co_no,
    estimate_torso_frame,
    measure_grey_v,
    GREY,
)
import blob_v2  # noqa: E402

# Distinct warm colours
COL = {
    "lats": np.array([1.00, 0.46, 0.20], np.float32),
    "mid_back": np.array([0.96, 0.36, 0.62], np.float32),
    "rear_delt": np.array([0.52, 0.28, 0.90], np.float32),
    "biceps": np.array([1.00, 0.84, 0.12], np.float32),
}
MUSCLE_ORDER = ("lats", "mid_back", "rear_delt", "biceps")


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def muscle_anchors(hip, spine, side, back, slen):
    """A few landmarks per muscle, in world metres, for one big smooth blob each."""
    s = float(slen / 0.55)

    def pt(t, lat, bck):
        return hip + (t * slen) * spine + (lat * s) * side + (bck * s) * back

    # Bilateral where the exercise reads on both sides; biceps on the FRONT of each upper arm.
    return {
        "lats": {
            "color": COL["lats"],
            "radius": 0.20 * s,
            "points": [
                pt(0.34, 0.13, 0.07), pt(0.48, 0.15, 0.06),
                pt(0.34, -0.13, 0.07), pt(0.48, -0.15, 0.06),
            ],
        },
        "mid_back": {
            "color": COL["mid_back"],
            "radius": 0.16 * s,
            "points": [
                pt(0.62, 0.00, 0.08),
                pt(0.74, 0.045, 0.07), pt(0.74, -0.045, 0.07),
            ],
        },
        "rear_delt": {
            "color": COL["rear_delt"],
            "radius": 0.095 * s,
            "points": [pt(0.82, 0.20, 0.055), pt(0.82, -0.20, 0.055)],
        },
        "biceps": {
            "color": COL["biceps"],
            "radius": 0.09 * s,
            "points": [pt(0.54, 0.25, -0.065), pt(0.54, -0.25, -0.065)],
        },
    }


def blob_weights(world, groups):
    """(N, 4) weights in MUSCLE_ORDER; each channel is max-of-anchors smoothstep falloff."""
    n = len(world)
    w = np.zeros((n, len(MUSCLE_ORDER)), np.float32)
    for i, name in enumerate(MUSCLE_ORDER):
        g = groups[name]
        radius = float(g["radius"])
        inner = 0.40 * radius
        acc = np.zeros(n, np.float32)
        for p in g["points"]:
            dist = np.linalg.norm(world - np.asarray(p, np.float64), axis=1)
            acc = np.maximum(acc, (1.0 - smoothstep(inner, radius, dist)).astype(np.float32))
        w[:, i] = acc
    return w


def write_mcol_from_weights(obj, world, groups):
    w = blob_weights(world, groups)
    # winner-takes-most: keep the strongest channel so colours stay distinct
    winner = np.argmax(w, axis=1)
    strength = w.max(axis=1)
    col = np.zeros((len(world), 4), np.float32)
    col[:, 3] = 1.0
    for i, name in enumerate(MUSCLE_ORDER):
        pick = (winner == i) & (strength > 1e-4)
        if pick.any():
            col[pick, :3] = groups[name]["color"] * strength[pick, None]
    me = obj.data
    va = me.color_attributes.get("mcol")
    if va is None:
        va = me.color_attributes.new("mcol", "FLOAT_COLOR", "POINT")
    va.data.foreach_set("color", col.ravel())
    me.update()
    return {name: int(((winner == i) & (strength > 1e-4)).sum()) for i, name in enumerate(MUSCLE_ORDER)}


def paint_all(meshes, groups):
    totals = {n: 0 for n in MUSCLE_ORDER}
    per = []
    for o in meshes:
        world, _ = _world_co_no(o)
        if not len(world):
            continue
        counts = write_mcol_from_weights(o, world, groups)
        for k, v in counts.items():
            totals[k] += v
        per.append({"name": o.name, **counts, "verts": int(len(world))})
    return totals, per


def grey_only_material():
    mat = bpy.data.materials.new("MatteGrey")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*GREY, 1)
    bsdf.inputs["Roughness"].default_value = 0.93
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.06
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def assign_mat(meshes, mat):
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            pass
    for img in list(bpy.data.images):
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass


def emission_mat(color, strength):
    mat = bpy.data.materials.new("EmitBlob")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*[float(c) for c in color], 1)
    em.inputs["Strength"].default_value = float(strength)
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def glass_mat():
    mat = bpy.data.materials.new("GlassSkin")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.38, 0.40, 0.43, 1)
    bsdf.inputs["Roughness"].default_value = 0.14
    key = "Transmission Weight" if "Transmission Weight" in bsdf.inputs else "Transmission"
    if key in bsdf.inputs:
        bsdf.inputs[key].default_value = 0.88
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.45
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 0.22
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    try:
        mat.blend_method = "HASHED"
        mat.shadow_method = "HASHED"
    except Exception:
        pass
    return mat


def basis_from_frame(spine, side, back):
    """3x3 whose columns are side, back, spine (local X/Y/Z)."""
    return np.stack([side, back, spine], 1)


def add_ellipsoid(center, radii, basis, color, strength, name):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=1.0, location=(0, 0, 0))
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    # scale in local then rotate to frame
    obj.scale = (float(radii[0]), float(radii[1]), float(radii[2]))
    # build rotation matrix from basis columns
    R = np.asarray(basis, dtype=np.float64)
    # orthonormalize defensively
    from mathutils import Matrix
    m = Matrix((
        (float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(center[0])),
        (float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(center[1])),
        (float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(center[2])),
        (0.0, 0.0, 0.0, 1.0),
    ))
    # apply scale first via object.scale, then world matrix for rotation+translation
    loc, rot, _ = m.decompose()
    obj.location = loc
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = rot
    mat = emission_mat(color, strength)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return obj


def make_shells(meshes, groups):
    """Duplicate surface where blob weight is high; push 2.5 mm along normals."""
    created = []
    for name in MUSCLE_ORDER:
        g = groups[name]
        mat = emission_mat(g["color"], 1.4)
        pieces = 0
        for src in meshes:
            world, no = _world_co_no(src)
            if not len(world):
                continue
            w = blob_weights(world, {name: g, **{k: groups[k] for k in MUSCLE_ORDER}})
            idx = MUSCLE_ORDER.index(name)
            mask = w[:, idx] > 0.32
            if int(mask.sum()) < 30:
                continue
            dup = src.copy()
            dup.data = src.data.copy()
            dup.name = f"shell_{name}_{src.name}"[:60]
            bpy.context.scene.collection.objects.link(dup)
            import bmesh
            bm = bmesh.new()
            bm.from_mesh(dup.data)
            bm.verts.ensure_lookup_table()
            # bmesh verts are created in index order from the mesh
            kill = [v for v in bm.verts if v.index >= len(mask) or not mask[v.index]]
            for v in kill:
                bm.verts.remove(v)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            for v in bm.verts:
                v.co = v.co + v.normal.normalized() * 0.0025
            bm.to_mesh(dup.data)
            bm.free()
            dup.data.materials.clear()
            dup.data.materials.append(mat)
            created.append(dup)
            pieces += 1
        print(f"shell {name} pieces={pieces}", flush=True)
    return created


def project_anchors(cam, groups, res):
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene
    out = {}
    for name, g in groups.items():
        pts = []
        for p in g["points"]:
            co = world_to_camera_view(sc, cam, Vector((float(p[0]), float(p[1]), float(p[2]))))
            pts.append({
                "ndc": [float(co.x), float(co.y), float(co.z)],
                "px": int(round(co.x * res[0])),
                "py": int(round((1.0 - co.y) * res[1])),
                "visible": bool(0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0 and co.z > 0),
            })
        out[name] = {"radius_m": float(g["radius"]), "color": g["color"].tolist(), "points": pts}
    return out


def compose_idpass(grey_png, proj, res, out_png):
    """Soft 2D elliptical glows at projected anchors, clipped to the body silhouette."""
    img = bpy.data.images.load(grey_png)
    w, h = img.size
    ch = img.channels
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)
    rgb = px[..., :3].copy()
    # silhouette: anything clearly brighter than charcoal
    sil = rgb.max(axis=2) > (40.0 / 255.0)
    # Blender image pixels are bottom-up; our py was computed top-down like PNG.
    # After render.write_still the saved PNG is top-down; bpy load of that PNG
    # is bottom-up. Flip sil to top-down to match px/py, then flip result back.
    sil_td = np.flipud(sil)
    glow = np.zeros((h, w, 3), np.float32)
    # approximate pixel radius from 540-wide frame of a ~1.7 m body filling ~80% height
    body_px = 0.78 * h
    metres_to_px = body_px / 1.55
    yy, xx = np.mgrid[0:h, 0:w]
    for name in MUSCLE_ORDER:
        rec = proj[name]
        color = np.asarray(rec["color"], np.float32)
        radius_px = max(18.0, rec["radius_m"] * metres_to_px)
        # muscle-shaped: lats/biceps taller, mid-back wider, delt round
        if name == "lats":
            sx, sy = radius_px * 0.72, radius_px * 1.15
        elif name == "mid_back":
            sx, sy = radius_px * 1.05, radius_px * 0.95
        elif name == "rear_delt":
            sx, sy = radius_px * 0.95, radius_px * 0.85
        else:
            sx, sy = radius_px * 0.70, radius_px * 1.05
        acc = np.zeros((h, w), np.float32)
        for p in rec["points"]:
            if not p["visible"]:
                continue
            cx, cy = p["px"], p["py"]
            nx = (xx - cx) / max(sx, 1.0)
            ny = (yy - cy) / max(sy, 1.0)
            r = np.sqrt(nx * nx + ny * ny)
            acc = np.maximum(acc, (1.0 - smoothstep(0.35, 1.0, r)).astype(np.float32))
        acc *= sil_td.astype(np.float32)
        glow += color[None, None, :] * acc[..., None]
    # soft bloom
    try:
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            glow[..., c] = gaussian_filter(glow[..., c], sigma=1.6)
    except Exception:
        pass
    glow *= sil_td[..., None]
    out = np.clip(np.flipud(rgb) + glow * 0.85, 0, 1)
    # write via bpy image
    flat = np.dstack([np.flipud(out), np.ones((h, w), np.float32)])
    img.pixels.foreach_set(flat.astype(np.float32).ravel())
    img.filepath_raw = out_png
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return out_png


def setup_and_import(args):
    _reset()
    print(f"blender {bpy.app.version_string} import {args.input} method={args.method}", flush=True)
    meshes, arms = _import(args.input)
    print(f"meshes={len(meshes)} armatures={len(arms)}", flush=True)
    if not meshes:
        raise SystemExit("no meshes imported")
    if arms:
        blob_v2.clear_to_rest_pose(arms)
        print("cleared to rest pose", [a.name for a in arms],
              "bones", sum(len(a.data.bones) for a in arms), flush=True)
    helpers = []
    body = []
    for o in meshes:
        n = (o.name or "").lower()
        if n.startswith("icosphere"):
            o.hide_render = True
            try:
                o.hide_set(True)
            except Exception:
                pass
            helpers.append(o.name)
        else:
            body.append(o)
    if body:
        meshes = body
        print("hidden helpers", helpers, "body_meshes", len(meshes), flush=True)
    scale_info = _apply_scale_to_metres(meshes)
    if args.target_height > 0:
        scale_info = _normalize_height(meshes, args.target_height, scale_info)
    print("scale", {k: scale_info.get(k) for k in ("height_m", "note", "normalize_factor")}, flush=True)
    pose_info = {"pose": getattr(args, "pose", "rest")}
    if getattr(args, "pose", "rest") == "row":
        pose_info = blob_v2.try_one_arm_row(arms)
        print("ROW_POSE", pose_info, flush=True)
    yaw = math.radians(args.yaw)
    elev = math.radians(args.elev)
    cam_dir = (
        math.sin(yaw) * math.cos(elev),
        -math.cos(yaw) * math.cos(elev),
        math.sin(elev),
    )
    # v2 anatomic frame (PCA shoulders) so mirroring is left-right, not camera-side.
    hip, spine, side, back, slen, frame = blob_v2.estimate_anatomic_frame(meshes, cam_dir)
    groups = muscle_anchors(hip, spine, side, back, slen)
    kept, hidden = _hide_plinths(meshes)
    if not kept:
        kept = meshes
    # Drop exporter helpers (this bodybuilder ships a 80-tri Icosphere).
    still_kept = []
    for o in kept:
        n = (o.name or "").lower()
        if n.startswith("icosphere"):
            o.hide_render = True
            try:
                o.hide_set(True)
            except Exception:
                pass
            hidden.append(o.name)
        else:
            still_kept.append(o)
    if still_kept:
        kept = still_kept
    return kept, hidden, scale_info, frame, groups, cam_dir, {
        "hip": hip, "spine": spine, "side": side, "back": back, "slen": slen,
        "arms": arms, "pose_info": pose_info,
    }


def render_with_exposure(still, exposure, auto):
    _render(still, exposure=exposure)
    stats = measure_grey_v(still)
    print("GREY", stats, flush=True)
    if auto and stats.get("grey_v_p50"):
        v = stats["grey_v_p50"]
        if v < 80 or v > 110:
            adj = math.log2(max(95.0 / max(v, 1.0), 1e-3))
            adj = float(max(-1.2, min(1.2, adj)))
            exposure = exposure + adj
            print(f"auto-exposure {v} -> {exposure:.3f}", flush=True)
            _render(still, exposure=exposure)
            stats = measure_grey_v(still)
            print("GREY2", stats, flush=True)
    return exposure, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", choices=(
        "blob", "idpass", "shells", "xray",
        "rigweight", "lights", "painted", "rim",
    ), default="blob")
    ap.add_argument("--yaw", type=float, default=135.0)
    ap.add_argument("--elev", type=float, default=14.0)
    ap.add_argument("--exposure", type=float, default=-0.15)
    ap.add_argument("--target-height", type=float, default=1.75)
    ap.add_argument("--no-auto-exposure", action="store_true")
    ap.add_argument("--pose", choices=("rest", "row"), default="rest",
                    help="rest = bind pose (cleared). row = best-effort one-arm DB row.")
    ap.add_argument("--no-bloom", action="store_true")
    args = ap.parse_args(_argv())
    os.makedirs(args.out_dir, exist_ok=True)
    meshes, hidden, scale_info, frame, groups, cam_dir, ctx = setup_and_import(args)
    hip, spine, side, back, slen = ctx["hip"], ctx["spine"], ctx["side"], ctx["back"], ctx["slen"]

    still = os.path.join(args.out_dir, f"{args.name}_{args.method}.png")
    extra = {"pose": ctx.get("pose_info"), "frame_mode": frame.get("frame_mode")}


    if args.method == "blob":
        totals, per = blob_v2.paint_blob(meshes, hip, spine, side, back, slen)
        print("PAINT_V2", totals, flush=True)
        extra.update({"paint_totals": totals, "paint_per_object": per, "colour_method": "capsule_v2"})
        blob_v2.assign_mat(meshes, blob_v2.punch_material())
        cam = _setup_camera(meshes, args.yaw, args.elev)
        _setup_world_lights(cam)
        exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)
        if not args.no_bloom:
            blob_v2.slight_bloom(still)
            stats = measure_grey_v(still)
            print("GREY_BLOOM", stats, flush=True)

    elif args.method == "painted":
        totals, per = blob_v2.paint_strokes(meshes, hip, spine, side, back, slen)
        print("PAINT_STROKES", totals, flush=True)
        extra.update({"paint_totals": totals, "paint_per_object": per, "colour_method": "strokes"})
        blob_v2.assign_mat(meshes, blob_v2.punch_material())
        cam = _setup_camera(meshes, args.yaw, args.elev)
        _setup_world_lights(cam)
        exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)
        if not args.no_bloom:
            blob_v2.slight_bloom(still)
            stats = measure_grey_v(still)
            print("GREY_BLOOM", stats, flush=True)

    elif args.method == "rim":
        totals, per = blob_v2.paint_blob(meshes, hip, spine, side, back, slen)
        print("PAINT_RIMMASK", totals, flush=True)
        extra.update({"paint_totals": totals, "paint_per_object": per, "colour_method": "rim"})
        blob_v2.assign_mat(meshes, blob_v2.rim_material())
        cam = _setup_camera(meshes, args.yaw, args.elev)
        _setup_world_lights(cam)
        exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)
        if not args.no_bloom:
            blob_v2.slight_bloom(still, amount=0.16)
            stats = measure_grey_v(still)
            print("GREY_BLOOM", stats, flush=True)

    elif args.method == "rigweight":
        totals, per, used = blob_v2.paint_rig_weights(meshes, hip, spine, side, back, slen)
        print("PAINT_RIG", totals, "groups", len(used), flush=True)
        extra.update({"paint_totals": totals, "paint_per_object": per,
                      "used_groups": used[:40], "colour_method": "rigweight"})
        if sum(totals.values()) == 0:
            raise SystemExit("rigweight: no classified vertex groups on any mesh")
        blob_v2.assign_mat(meshes, blob_v2.punch_material())
        cam = _setup_camera(meshes, args.yaw, args.elev)
        _setup_world_lights(cam)
        exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)
        if not args.no_bloom:
            blob_v2.slight_bloom(still)
            stats = measure_grey_v(still)
            print("GREY_BLOOM", stats, flush=True)

    elif args.method == "lights":
        assign_mat(meshes, grey_only_material())
        cam = _setup_camera(meshes, args.yaw, args.elev)
        _setup_world_lights(cam)
        blob_v2.dim_white_lights()
        spots = blob_v2.add_colored_spots(meshes, hip, spine, side, back, slen)
        extra.update({"spot_lights": spots, "colour_method": "projected_spots"})
        print("LIGHTS", len(spots), flush=True)
        exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)

    elif args.method == "idpass":
        assign_mat(meshes, grey_only_material())
        cam = _setup_camera(meshes, args.yaw, args.elev)
        _setup_world_lights(cam)
        grey_png = os.path.join(args.out_dir, f"{args.name}_idpass_grey.png")
        exposure, stats = render_with_exposure(grey_png, args.exposure, not args.no_auto_exposure)
        proj = project_anchors(cam, groups, (540, 960))
        compose_idpass(grey_png, proj, (540, 960), still)
        extra = {"projected": {k: v["points"] for k, v in proj.items()}}
        # re-measure the composited image
        stats = measure_grey_v(still)
        print("GREY_COMPOSITE", stats, flush=True)

    elif args.method == "shells":
        assign_mat(meshes, grey_only_material())
        shells = make_shells(meshes, groups)
        extra = {"shell_count": len(shells)}
        cam = _setup_camera(meshes, args.yaw, args.elev)
        _setup_world_lights(cam)
        exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)

    elif args.method == "xray":
        assign_mat(meshes, glass_mat())
        basis = basis_from_frame(
            np.asarray(frame["spine"], np.float64),
            np.asarray(frame["side"], np.float64),
            np.asarray(frame["back"], np.float64),
        )
        inners = []
        for name in MUSCLE_ORDER:
            g = groups[name]
            # elongate along spine for lats / mid-back; rounder for delt / biceps
            if name == "lats":
                rad = (0.55 * g["radius"], 0.40 * g["radius"], 0.70 * g["radius"])
            elif name == "mid_back":
                rad = (0.70 * g["radius"], 0.35 * g["radius"], 0.60 * g["radius"])
            elif name == "rear_delt":
                rad = (0.70 * g["radius"], 0.50 * g["radius"], 0.55 * g["radius"])
            else:
                rad = (0.50 * g["radius"], 0.45 * g["radius"], 0.70 * g["radius"])
            for i, p in enumerate(g["points"]):
                inners.append(add_ellipsoid(p, rad, basis, g["color"], 3.2, f"inner_{name}_{i}"))
        extra = {"inner_count": len(inners)}
        cam = _setup_camera(meshes, args.yaw, args.elev)
        _setup_world_lights(cam)
        # glass reads darker; don't auto-expose off the grey-body heuristic alone
        exposure, stats = render_with_exposure(still, args.exposure, False)

    else:
        raise SystemExit(args.method)

    out = {
        "name": args.name,
        "method": args.method,
        "input": args.input,
        "still": still,
        "exposure": exposure if args.method != "idpass" else args.exposure,
        "scale": scale_info,
        "frame": {k: frame.get(k) for k in ("body_len", "spine_len", "used_named_head", "used_named_feet", "back_dot_cam")},
        "hidden": hidden,
        "grey": stats,
        **extra,
    }
    jpath = os.path.join(args.out_dir, f"{args.name}_{args.method}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("WROTE", still, jpath, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
