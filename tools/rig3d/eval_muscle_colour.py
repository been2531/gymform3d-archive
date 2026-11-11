#!/usr/bin/env python3
"""Paint db-row target muscles on an unrigged candidate and render one still.

Geometric / vertex-attribute mask (no UVs). Torso frame is estimated from
the mesh (named head/feet/arms when present, otherwise height bands).
Regions: lats, rhomboids+mid-traps, rear delts, biceps.

Run with Blender 4.2.3:
  nice -n 19 blender -b --python tools/rig3d/eval_muscle_colour.py -- \\
      --input PATH --out-dir DIR --name ID
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
    _assign_clay,
    _hide_plinths,
    _import,
    _render,
    _setup_camera,
    _setup_world_lights,
    _world_dims,
    _reset,
    _argv,
)

RES = (540, 960)
GREY = (0.22, 0.224, 0.228)
ROUGH = 0.93
SPEC = 0.06
EM_STR = 0.30

# Distinct warm colours (linear-ish display targets)
COL = {
    "lats": np.array([1.00, 0.46, 0.20], np.float32),          # peach-orange
    "rhomb_traps": np.array([0.96, 0.36, 0.62], np.float32),   # pink
    "rear_delt": np.array([0.52, 0.28, 0.90], np.float32),     # purple
    "biceps": np.array([1.00, 0.84, 0.12], np.float32),        # yellow
}


def _foreach_co(me):
    n = len(me.vertices)
    buf = np.empty(n * 3, np.float64)
    me.vertices.foreach_get("co", buf)
    return buf.reshape(n, 3)


def _foreach_no(me):
    n = len(me.vertices)
    buf = np.empty(n * 3, np.float64)
    me.vertices.foreach_get("normal", buf)
    return buf.reshape(n, 3)


def _world_co_no(obj):
    mw = np.array(obj.matrix_world, dtype=np.float64)
    co = _foreach_co(obj.data)
    h = np.c_[co, np.ones(len(co))]
    world = (mw @ h.T).T[:, :3]
    R = mw[:3, :3]
    invT = np.linalg.inv(R).T
    no = _foreach_no(obj.data) @ invT
    no /= np.linalg.norm(no, axis=1, keepdims=True) + 1e-12
    return world, no


def _named_centroid(meshes, keys):
    keys = [k.lower() for k in keys]
    pts = []
    for o in meshes:
        name = (o.name or "").lower()
        if any(k in name for k in keys):
            world, _ = _world_co_no(o)
            if len(world):
                pts.append(world.mean(axis=0))
    if not pts:
        return None
    return np.mean(np.stack(pts, 0), axis=0)


def _all_world(meshes):
    chunks = []
    for o in meshes:
        world, no = _world_co_no(o)
        if len(world):
            chunks.append((o, world, no))
    return chunks


def estimate_torso_frame(meshes, cam_dir):
    """Build hip / spine / side / back without an armature.

    Named parts (head, helmet, shoe, shirt, arm) win when present so a
    jumping football player is not framed as if it were standing on Z.
    """
    chunks = _all_world(meshes)
    all_co = np.concatenate([w for _, w, _ in chunks], 0)
    zmin, zmax = float(all_co[:, 2].min()), float(all_co[:, 2].max())
    height = max(zmax - zmin, 1e-6)

    head = _named_centroid(meshes, ("helmet", "head_head", "head"))
    feet = _named_centroid(meshes, ("shoe", "shoes"))
    if head is None:
        top = all_co[all_co[:, 2] >= zmin + 0.92 * height]
        head = top.mean(axis=0)
    if feet is None:
        bot = all_co[all_co[:, 2] <= zmin + 0.08 * height]
        feet = bot.mean(axis=0)

    axis = head - feet
    body_len = float(np.linalg.norm(axis)) or 1.0
    up = axis / body_len
    hip = feet + 0.50 * body_len * up
    neck = feet + 0.85 * body_len * up
    spine = neck - hip
    slen = float(np.linalg.norm(spine)) or 1e-6
    spine = spine / slen

    # Back = camera direction in the plane perpendicular to the spine.
    # PCA of a jumping / one-arm-out shoulder band can pick the player's
    # left-right axis as "back" and paint the flank instead of the jersey.
    cam = np.asarray(cam_dir, dtype=np.float64)
    back = cam - cam.dot(spine) * spine
    bn = float(np.linalg.norm(back))
    if bn < 1e-8:
        back = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        back = back - back.dot(spine) * spine
        bn = float(np.linalg.norm(back)) or 1.0
    back = back / bn
    side = np.cross(spine, back)
    side = side / (np.linalg.norm(side) + 1e-12)

    info = {
        "body_len": body_len,
        "spine_len": slen,
        "hip": hip.tolist(),
        "neck": neck.tolist(),
        "head": head.tolist(),
        "feet": feet.tolist(),
        "spine": spine.tolist(),
        "side": side.tolist(),
        "back": back.tolist(),
        "used_named_head": bool(_named_centroid(meshes, ("helmet", "head_head", "head")) is not None),
        "used_named_feet": bool(_named_centroid(meshes, ("shoe", "shoes")) is not None),
        "back_dot_cam": float(back.dot(np.asarray(cam_dir, dtype=np.float64))),
    }
    return hip, spine, side, back, slen, info


def _edges_np(me):
    n = len(me.edges)
    buf = np.empty(n * 2, np.int32)
    me.edges.foreach_get("vertices", buf)
    return buf.reshape(n, 2)


def _dilate(mask, edges, steps, allowed):
    m = mask.copy()
    for _ in range(int(steps)):
        grow = np.zeros_like(m)
        hit = m[edges[:, 0]] | m[edges[:, 1]]
        e = edges[hit]
        if len(e):
            grow[e[:, 0]] = True
            grow[e[:, 1]] = True
        m = m | (grow & allowed)
    return m


def paint_object(obj, hip, spine, side, back, slen):
    me = obj.data
    world, no = _world_co_no(obj)
    nv = len(world)
    if nv == 0:
        return {"verts": 0}
    rel = world - hip
    t = (rel @ spine) / slen
    lat = rel @ side
    bck = rel @ back
    nb = no @ back
    abs_lat = np.abs(lat)
    # Scale metre windows to this figure's hip→neck length (ref ~0.55 m).
    s = float(slen / 0.55)
    name = (obj.name or "").lower()
    is_arm_mesh = any(k in name for k in ("arm", "hand")) and "under" not in name
    is_head_mesh = any(k in name for k in ("helmet", "head", "eye", "lash"))
    is_leg_mesh = any(k in name for k in ("shoe", "sock", "leg", "pant"))

    backish = (nb > -0.18) & (bck > -0.16)
    sideish = np.abs(no @ side) > 0.45

    # Mid-traps: upper medial sheet, just below the neck, not the whole back.
    mid_trap = backish & (t >= 0.55) & (t <= 0.96) & (abs_lat <= 0.10 * s)
    # Rhomboids: between the scapulae, slightly lower and just off the midline.
    rhomb = backish & (t >= 0.44) & (t < 0.74) & (abs_lat >= 0.012 * s) & (abs_lat <= 0.10 * s) & ~mid_trap
    # Lats: lateral wings of the back, under the armpit down toward the pelvis.
    lats = (t >= 0.18) & (t <= 0.64) & (abs_lat >= 0.07 * s) & (abs_lat <= 0.22 * s)
    lats &= ((nb > 0.00) | sideish) & (nb > -0.28) & ~mid_trap & ~rhomb
    # Rear delts: posterior shoulder cap.
    rear_delt = (nb > -0.28) & (bck > -0.12) & (t >= 0.64) & (t <= 0.95) & (abs_lat >= 0.10 * s) & (abs_lat <= 0.28 * s)
    rear_delt &= ~mid_trap & ~rhomb
    # Biceps: anterior upper arm only — not forearm (lower t when the arm hangs).
    biceps = (t >= 0.40) & (t <= 0.74) & (abs_lat >= 0.16 * s) & (abs_lat <= 0.38 * s)
    biceps &= (bck < 0.05)
    if is_arm_mesh:
        biceps = (t >= 0.40) & (t <= 0.74) & (abs_lat >= 0.13 * s) & (nb < 0.32) & (bck < 0.10)
    if is_arm_mesh:
        # jersey-adjacent arms must not inherit lat / trap / rear-delt colours
        mid_trap[:] = False
        rhomb[:] = False
        lats[:] = False
        rear_delt[:] = False
    if is_head_mesh or is_leg_mesh:
        mid_trap[:] = False
        rhomb[:] = False
        lats[:] = False
        rear_delt[:] = False
        biceps[:] = False

    far_limb = (t < -0.10) | (t > 1.12) | (abs_lat > 0.42 * s)
    mid_trap &= ~far_limb
    rhomb &= ~far_limb
    lats &= ~far_limb
    rear_delt &= ~far_limb
    biceps &= ~far_limb & ~lats

    allowed = (t >= 0.06) & (t <= 0.98) & (abs_lat < 0.30 * s) & (nb > -0.34)
    edges = _edges_np(me)
    mid_trap = _dilate(mid_trap, edges, 2, allowed & (t >= 0.54))
    rhomb = _dilate(rhomb, edges, 2, allowed & (t >= 0.40) & (t <= 0.78) & ~mid_trap)
    lats = _dilate(lats, edges, 2, allowed & (t <= 0.68) & ~mid_trap & ~rhomb)
    rear_delt = _dilate(rear_delt, edges, 2, allowed & (t >= 0.62) & ~mid_trap & ~rhomb)
    biceps = _dilate(biceps, edges, 2, (abs_lat >= 0.13 * s) & (t >= 0.38) & (t <= 0.76) & ~lats)

    # Exclusive by specificity: biceps > rear_delt > rhomb/traps > lats
    rhomb_traps = (rhomb | mid_trap) & ~biceps
    rear_delt = rear_delt & ~biceps & ~rhomb_traps
    lats = lats & ~biceps & ~rear_delt & ~rhomb_traps

    col = np.zeros((nv, 4), np.float32)
    col[:, 3] = 1.0
    col[lats, :3] = COL["lats"]
    col[rhomb_traps, :3] = COL["rhomb_traps"]
    col[rear_delt, :3] = COL["rear_delt"]
    col[biceps, :3] = COL["biceps"]

    va = me.color_attributes.get("mcol")
    if va is None:
        va = me.color_attributes.new("mcol", "FLOAT_COLOR", "POINT")
    va.data.foreach_set("color", col.ravel())
    me.update()
    return {
        "name": obj.name,
        "verts": int(nv),
        "lats": int(lats.sum()),
        "rhomb_traps": int(rhomb_traps.sum()),
        "rear_delt": int(rear_delt.sum()),
        "biceps": int(biceps.sum()),
    }


def _colour_material():
    mat = bpy.data.materials.new("MuscleColourClay")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    grey = nt.nodes.new("ShaderNodeBsdfPrincipled")
    grey.inputs["Base Color"].default_value = (*GREY, 1)
    grey.inputs["Roughness"].default_value = ROUGH
    if "Specular IOR Level" in grey.inputs:
        grey.inputs["Specular IOR Level"].default_value = SPEC
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "mcol"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(attr.outputs["Color"], sep.inputs["Color"])
    mx1 = nt.nodes.new("ShaderNodeMath")
    mx1.operation = "MAXIMUM"
    nt.links.new(sep.outputs["Red"], mx1.inputs[0])
    nt.links.new(sep.outputs["Green"], mx1.inputs[1])
    mx2 = nt.nodes.new("ShaderNodeMath")
    mx2.operation = "MAXIMUM"
    nt.links.new(mx1.outputs[0], mx2.inputs[0])
    nt.links.new(sep.outputs["Blue"], mx2.inputs[1])
    # tinted principled so sculpt form remains under the colour
    tint = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(attr.outputs["Color"], tint.inputs["Base Color"])
    tint.inputs["Roughness"].default_value = 0.72
    if "Specular IOR Level" in tint.inputs:
        tint.inputs["Specular IOR Level"].default_value = 0.08
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(attr.outputs["Color"], em.inputs["Color"])
    stm = nt.nodes.new("ShaderNodeMath")
    stm.operation = "MULTIPLY"
    nt.links.new(mx2.outputs[0], stm.inputs[0])
    stm.inputs[1].default_value = EM_STR
    nt.links.new(stm.outputs[0], em.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tint.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mx2.outputs[0], mix.inputs["Fac"])
    nt.links.new(grey.outputs[0], mix.inputs[1])
    nt.links.new(add.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def _assign_colour_mat(meshes):
    mat = _colour_material()
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        try:
            if hasattr(o.data, "attributes") and "custom_normal" in o.data.attributes:
                o.data.attributes.remove(o.data.attributes["custom_normal"])
        except Exception:
            pass
        try:
            o.data.use_auto_smooth = False
        except Exception:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            pass
        # drop packed images after clay swap to free RAM
    for img in list(bpy.data.images):
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass


def _normalize_height(meshes, target, scale_info):
    """Scale the figure about its roots so parented pieces (helmet, shoes) stay assembled."""
    current = max(scale_info.get("height_m") or 0.0, 1e-9)
    factor = target / current
    roots = [o for o in bpy.data.objects if o.parent is None]
    for o in roots:
        o.scale = o.scale * factor
    bpy.context.view_layer.update()
    mn, mx, dim = _world_dims(meshes)
    scale_info["normalized_from_m"] = current
    scale_info["normalize_factor"] = factor
    scale_info["target_height_m"] = target
    scale_info["height_m"] = max(dim.x, dim.y, dim.z)
    scale_info["bbox_min"] = list(mn)
    scale_info["bbox_max"] = list(mx)
    scale_info["bbox_dim"] = list(dim)
    return scale_info


def measure_grey_v(png):
    """Sample 8-bit V without PIL (Blender's Python has no Pillow)."""
    img = bpy.data.images.load(png)
    w, h = img.size
    ch = img.channels
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)
    rgb = px[..., :3] * 255.0
    mx = rgb.max(axis=2)
    chroma = mx - rgb.min(axis=2)
    # exclude charcoal bg (V~30-45 after denoise) and saturated muscle glow
    grey = (mx >= 55) & (mx <= 200) & (chroma < 28)
    color = (chroma >= 40) & (mx >= 40)
    bg = mx < 45
    stats = {
        "grey_n": int(grey.sum()),
        "color_n": int(color.sum()),
        "bg_n": int(bg.sum()),
    }
    if grey.any():
        pr = np.percentile(mx[grey], [10, 50, 90])
        stats.update({
            "grey_v_p10": round(float(pr[0]), 1),
            "grey_v_p50": round(float(pr[1]), 1),
            "grey_v_p90": round(float(pr[2]), 1),
            "grey_v_mean": round(float(mx[grey].mean()), 1),
        })
    if color.any():
        stats["color_v_mean"] = round(float(mx[color].mean()), 1)
    bpy.data.images.remove(img)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--yaw", type=float, default=135.0)
    ap.add_argument("--elev", type=float, default=8.0)
    ap.add_argument("--exposure", type=float, default=-0.15)
    ap.add_argument("--target-height", type=float, default=1.75)
    ap.add_argument("--no-auto-exposure", action="store_true")
    args = ap.parse_args(_argv())
    os.makedirs(args.out_dir, exist_ok=True)
    _reset()
    print(f"blender {bpy.app.version_string} import {args.input}", flush=True)
    meshes, arms = _import(args.input)
    print(f"meshes={len(meshes)} armatures={len(arms)}", flush=True)
    if not meshes:
        raise SystemExit("no meshes imported")
    scale_info = _apply_scale_to_metres(meshes)
    if args.target_height > 0:
        scale_info = _normalize_height(meshes, args.target_height, scale_info)
    print("scale", scale_info, flush=True)

    yaw = math.radians(args.yaw)
    elev = math.radians(args.elev)
    cam_dir = (
        math.sin(yaw) * math.cos(elev),
        -math.cos(yaw) * math.cos(elev),
        math.sin(elev),
    )
    hip, spine, side, back, slen, frame = estimate_torso_frame(meshes, cam_dir)
    print("frame", {k: frame[k] for k in ("body_len", "spine_len", "used_named_head", "used_named_feet")}, flush=True)

    paints = []
    for o in meshes:
        paints.append(paint_object(o, hip, spine, side, back, slen))
    totals = {k: sum(p.get(k, 0) for p in paints) for k in ("verts", "lats", "rhomb_traps", "rear_delt", "biceps")}
    print("PAINT", totals, flush=True)

    _assign_clay(meshes)  # records/strips source images; we replace next
    _assign_colour_mat(meshes)
    kept, hidden = _hide_plinths(meshes)
    if not kept:
        kept = meshes

    cam = _setup_camera(kept, args.yaw, args.elev)
    _setup_world_lights(cam)
    still = os.path.join(args.out_dir, f"{args.name}_muscle.png")
    exposure = args.exposure
    _render(still, exposure=exposure)
    stats = measure_grey_v(still)
    print("GREY", stats, flush=True)
    if (not args.no_auto_exposure) and stats.get("grey_v_p50"):
        v = stats["grey_v_p50"]
        target = 95.0
        if v < 80 or v > 110:
            adj = math.log2(max(target / max(v, 1.0), 1e-3))
            # clamp so we do not invert the picture
            adj = float(max(-1.2, min(1.2, adj)))
            exposure = exposure + adj
            print(f"auto-exposure {stats['grey_v_p50']} -> retry exposure={exposure:.3f}", flush=True)
            _render(still, exposure=exposure)
            stats = measure_grey_v(still)
            print("GREY2", stats, flush=True)

    out = {
        "name": args.name,
        "input": args.input,
        "still": still,
        "exposure": exposure,
        "scale": scale_info,
        "frame": frame,
        "paint_totals": totals,
        "paint_per_object": [p for p in paints if p.get("verts")],
        "hidden": hidden,
        "grey": stats,
        "colours": {k: v.tolist() for k, v in COL.items()},
    }
    jpath = os.path.join(args.out_dir, f"{args.name}_muscle.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", still, jpath, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
