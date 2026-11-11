#!/usr/bin/env python3
"""Hybrid muscle colour + hand-keyed row stills on the minglanders1 bodybuilder.

Shots:
  standing  — rest pose, straight back view (hybrid colour)
  bb_top / bb_bottom — barbell bent-over row, elevated back camera (demic match)
  db_top / db_bottom — bench-supported one-arm DB row, side-up-and-back camera

  nice -n 19 blender -b --python tools/rig3d/eval_hybrid_row.py -- \
      --input output/_basemodels/sketchfab/male_mesh_bodybuilder/scene.gltf \
      --out-dir output/_basemodels/eval --name bodybuilder_v3
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
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROP = os.path.join(HERE, "props")
if PROP not in sys.path:
    sys.path.insert(0, PROP)

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
    _normalize_height,
    measure_grey_v,
)
import blob_v2  # noqa: E402
import bodybuilder_pose as bbpose  # noqa: E402
from barbell_olympic import build_barbell  # noqa: E402
from bench_flat import build_bench, place_bench  # noqa: E402


def hide_helpers(meshes):
    body, hidden = [], []
    for o in meshes:
        n = (o.name or "").lower()
        if n.startswith("icosphere"):
            o.hide_render = True
            try:
                o.hide_set(True)
            except Exception:
                pass
            hidden.append(o.name)
        else:
            body.append(o)
    return body or meshes, hidden


def wipe_named(*names):
    for n in names:
        o = bpy.data.objects.get(n)
        if o is None:
            continue
        bpy.data.objects.remove(o, do_unlink=True)


def add_floor():
    if bpy.data.objects.get("hybrid_floor"):
        return bpy.data.objects["hybrid_floor"]
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
    fl = bpy.context.object
    fl.name = "hybrid_floor"
    mat = bpy.data.materials.new("hybrid_floor_mat")
    mat.use_nodes = True
    bs = mat.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value = (0.022, 0.022, 0.024, 1)
    bs.inputs["Roughness"].default_value = 0.92
    fl.data.materials.append(mat)
    fl.hide_select = True
    fl.is_shadow_catcher = True
    return fl


def place_barbell_at_hands(bar, l_hand, r_hand, right):
    mid = 0.5 * (np.asarray(l_hand, np.float64) + np.asarray(r_hand, np.float64))
    x = np.asarray(r_hand, np.float64) - np.asarray(l_hand, np.float64)
    if float(np.linalg.norm(x)) < 0.04:
        x = np.asarray(right, np.float64)
    x = x / (float(np.linalg.norm(x)) or 1.0)
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    yn = float(np.linalg.norm(y))
    if yn < 1e-6:
        y = np.cross(np.array([0.0, 1.0, 0.0]), x)
        yn = float(np.linalg.norm(y))
    y = y / max(yn, 1e-9)
    z = np.cross(x, y)
    z = z / (float(np.linalg.norm(z)) or 1.0)
    mat = Matrix.Identity(4)
    mat.col[0][:3] = Vector(x)
    mat.col[1][:3] = Vector(y)
    mat.col[2][:3] = Vector(z)
    mat.col[3][:3] = Vector((float(mid[0]), float(mid[1]), float(mid[2])))
    bar.matrix_world = mat
    return [round(float(v), 3) for v in mid]


def build_simple_dumbbell():
    """Small procedural hex-ish dumbbell (handle + two plates)."""
    steel = bpy.data.materials.new("db_steel")
    steel.use_nodes = True
    bs = steel.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value = (0.45, 0.45, 0.48, 1)
    bs.inputs["Metallic"].default_value = 1.0
    bs.inputs["Roughness"].default_value = 0.38
    dark = bpy.data.materials.new("db_plate")
    dark.use_nodes = True
    bs2 = dark.node_tree.nodes.get("Principled BSDF")
    bs2.inputs["Base Color"].default_value = (0.04, 0.04, 0.045, 1)
    bs2.inputs["Metallic"].default_value = 0.6
    bs2.inputs["Roughness"].default_value = 0.45
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.016, depth=0.15, location=(0, 0, 0),
        rotation=(0.0, math.pi / 2.0, 0.0), vertices=24)
    handle = bpy.context.object
    handle.name = "db_handle"
    handle.data.materials.append(steel)
    parts = [handle]
    for sx in (-1.0, 1.0):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.055, depth=0.032, location=(sx * 0.095, 0, 0),
            rotation=(0.0, math.pi / 2.0, 0.0), vertices=20)
        p = bpy.context.object
        p.name = f"db_plate_{sx:.0f}"
        p.data.materials.append(dark)
        parts.append(p)
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = handle
    bpy.ops.object.join()
    db = bpy.context.object
    db.name = "dumbbell_simple"
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return db


def setup_elevated_back_camera(look, back_dir, right_dir, body_height, elev_deg=42.0,
                               side_deg=16.0, margin=0.10):
    """Demic-like elevated back view: behind the athlete, slightly off to one side."""
    wipe_named("Cam", "Key", "Fill", "Rim")
    v_ang = 2 * math.atan(12.0 / 50.0)
    dist = (body_height * (1 + 2 * margin) / 2) / math.tan(v_ang / 2)
    dist *= 0.95
    elev = math.radians(elev_deg)
    side = math.radians(side_deg)
    back = np.asarray(back_dir, np.float64)
    right = np.asarray(right_dir, np.float64)
    horiz = back * math.cos(side) - right * math.sin(side)
    hn = float(np.linalg.norm(horiz)) or 1.0
    horiz = horiz / hn
    direction = horiz * math.cos(elev) + np.array([0.0, 0.0, 1.0]) * math.sin(elev)
    dn = float(np.linalg.norm(direction)) or 1.0
    direction = direction / dn
    look_v = Vector((float(look[0]), float(look[1]), float(look[2])))
    loc = look_v + Vector(direction) * dist
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50
    cam_data.clip_start = 0.05
    cam_data.clip_end = 50
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    aim = look_v - loc
    cam.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam, {"dist": dist, "elev_deg": elev_deg, "side_deg": side_deg,
                 "look": [float(x) for x in look], "loc": list(loc)}


def setup_db_side_camera(look, right_dir, back_dir, body_height,
                         yaw_back_deg=18.0, elev_deg=25.0, margin=0.12):
    """Spec camera: pure working-side, then raise and rotate slightly toward the back."""
    wipe_named("Cam", "Key", "Fill", "Rim")
    v_ang = 2 * math.atan(12.0 / 50.0)
    dist = (body_height * (1 + 2 * margin) / 2) / math.tan(v_ang / 2)
    dist *= 1.05
    elev = math.radians(elev_deg)
    yaw = math.radians(yaw_back_deg)
    right = np.asarray(right_dir, np.float64)
    back = np.asarray(back_dir, np.float64)
    horiz = right * math.cos(yaw) + back * math.sin(yaw)
    hn = float(np.linalg.norm(horiz)) or 1.0
    horiz = horiz / hn
    direction = horiz * math.cos(elev) + np.array([0.0, 0.0, 1.0]) * math.sin(elev)
    direction = direction / (float(np.linalg.norm(direction)) or 1.0)
    look_v = Vector((float(look[0]), float(look[1]), float(look[2])))
    loc = look_v + Vector(direction) * dist
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50
    cam_data.clip_start = 0.05
    cam_data.clip_end = 50
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (look_v - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam, {"dist": dist, "elev_deg": elev_deg, "yaw_back_deg": yaw_back_deg,
                 "look": [float(x) for x in look], "loc": list(loc)}


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


def reset_pose(arms):
    for arm in arms:
        bbpose.clear_bone_constraints(arm)
    blob_v2.clear_to_rest_pose(arms)


def import_body(path, target_height):
    _reset()
    meshes, arms = _import(path)
    print(f"meshes={len(meshes)} armatures={len(arms)}", flush=True)
    if arms:
        blob_v2.clear_to_rest_pose(arms)
        print("cleared to rest pose", [a.name for a in arms],
              "bones", sum(len(a.data.bones) for a in arms), flush=True)
    meshes, hidden = hide_helpers(meshes)
    scale_info = _apply_scale_to_metres(meshes)
    if target_height > 0:
        scale_info = _normalize_height(meshes, target_height, scale_info)
    print("scale", {k: scale_info.get(k) for k in ("height_m", "note", "normalize_factor", "head_flip")},
          flush=True)
    kept, hidden2 = _hide_plinths(meshes)
    hidden.extend(hidden2)
    kept, hidden3 = hide_helpers(kept)
    hidden.extend(hidden3)
    return kept, arms, scale_info, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="bodybuilder_v3")
    ap.add_argument("--target-height", type=float, default=1.75)
    ap.add_argument("--exposure", type=float, default=-0.15)
    ap.add_argument("--no-auto-exposure", action="store_true")
    ap.add_argument("--no-bloom", action="store_true")
    ap.add_argument("--shots", default="standing,bb_top,bb_bottom,db_top,db_bottom")
    args = ap.parse_args(_argv())
    os.makedirs(args.out_dir, exist_ok=True)
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]

    meshes, arms, scale_info, hidden = import_body(args.input, args.target_height)
    if not meshes:
        raise SystemExit("no body meshes")
    # Anatomic frame from a dummy back-view cam_dir (yaw 180, elev 8).
    cam_dir = (0.0, math.cos(math.radians(8.0)), math.sin(math.radians(8.0)))
    hip, spine, side, back, slen, frame = blob_v2.estimate_anatomic_frame(meshes, cam_dir)
    totals, per, used = blob_v2.paint_hybrid(meshes, hip, spine, side, back, slen, working_side=None)
    print("PAINT_HYBRID", totals, "groups", len(used), flush=True)
    blob_v2.assign_mat(meshes, blob_v2.punch_material(em_str=blob_v2.EM_STR_HYBRID))
    add_floor()

    reports = {
        "name": args.name,
        "input": args.input,
        "scale": scale_info,
        "paint_totals": totals,
        "paint_per_object": per,
        "frame_mode": frame.get("frame_mode"),
        "hidden": hidden,
        "shots": {},
    }
    arm = arms[0] if arms else None

    if "standing" in shots:
        wipe_named("Cam", "Key", "Fill", "Rim")
        cam = _setup_camera(meshes, 180.0, 8.0)
        _setup_world_lights(cam)
        still = os.path.join(args.out_dir, f"{args.name}_standing_hybrid.png")
        exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)
        if not args.no_bloom:
            blob_v2.slight_bloom(still)
            stats = measure_grey_v(still)
        print("WROTE", still, flush=True)
        reports["shots"]["standing"] = {"still": still, "grey": stats, "exposure": exposure}

    barbell = None
    if arm is not None and any(s.startswith("bb_") for s in shots):
        barbell, grips, bmeta = build_barbell(n_plates_each=1, clear=False)
        print("BARBELL", barbell.name, "verts", len(barbell.data.vertices), flush=True)
        reports["barbell"] = {"name": barbell.name, "meta": {k: bmeta.get(k) for k in
                             ("bar_len_m", "plate_r_m", "n_plates_each", "licence")}}

        for phase, key in (("top", "bb_top"), ("bottom", "bb_bottom")):
            if key not in shots:
                continue
            reset_pose(arms)
            info = bbpose.pose_barbell_row(arm, phase=phase, hinge_deg=45.0)
            print("BB_ROW", key, json.dumps(info["qc"], default=str), flush=True)
            # Sit the bar on the solved wrists, not the IK targets.
            mid = place_barbell_at_hands(
                barbell, info["pts"]["lwr"], info["pts"]["rwr"], info["right"])
            look = info["hip"] + info["spine_dir"] * (0.48 * float(np.linalg.norm(info["sh_mid"] - info["hip"])))
            mn, mx, dim = _world_dims(meshes)
            back_dir = -info["fwd"]
            body_h = float(max(dim.z, 0.90))
            cam, cam_info = setup_elevated_back_camera(
                look, back_dir, info["right"], body_h,
                elev_deg=30.0, side_deg=18.0, margin=0.12)
            _setup_world_lights(cam)
            still = os.path.join(args.out_dir, f"{args.name}_{key}.png")
            exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)
            if not args.no_bloom:
                blob_v2.slight_bloom(still)
                stats = measure_grey_v(still)
            print("WROTE", still, flush=True)
            reports["shots"][key] = {
                "still": still, "grey": stats, "exposure": exposure,
                "qc": info["qc"], "cam": cam_info, "bar_mid": mid,
            }

    bench = None
    dumbbell = None
    if arm is not None and any(s.startswith("db_") for s in shots):
        if barbell is not None:
            barbell.hide_render = True
            try:
                barbell.hide_set(True)
            except Exception:
                pass
        bench, bmeta = build_bench(look="gunmetal_black", clear=False, padded=True)
        # Landmark frame: pad centre at origin, long axis +Y (place_bench yaw = pi/2).
        place_bench(bench, (0.0, 0.0), math.pi / 2.0, origin_z=0.39)
        print("BENCH", bench.name, flush=True)
        dumbbell = build_simple_dumbbell()
        reports["bench"] = {"name": bench.name, "pad_top_z_m": 0.43}

        for phase, key in (("top", "db_top"), ("bottom", "db_bottom")):
            if key not in shots:
                continue
            reset_pose(arms)
            info = bbpose.pose_bench_db_row(arm, phase=phase, pad_z=0.43)
            print("DB_ROW", key, json.dumps(info["qc"], default=str), flush=True)
            # Align dumbbell with the working (right) hand; handle along world X (working side).
            rh = info["r_hand"]
            dumbbell.location = Vector((float(rh[0]), float(rh[1]), float(rh[2]) - 0.03))
            dumbbell.rotation_euler = (0.0, 0.0, 0.0)
            look = 0.5 * (np.asarray(info["pts"]["lhip"]) + np.asarray(info["pts"]["rsh"]))
            mn, mx, dim = _world_dims(meshes)
            # Landmark frame: forward = +Y, right = +X, back = -Y
            cam, cam_info = setup_db_side_camera(
                look, np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]),
                max(dim.y, dim.z, 0.8), yaw_back_deg=18.0, elev_deg=25.0)
            _setup_world_lights(cam)
            still = os.path.join(args.out_dir, f"{args.name}_{key}.png")
            exposure, stats = render_with_exposure(still, args.exposure, not args.no_auto_exposure)
            if not args.no_bloom:
                blob_v2.slight_bloom(still)
                stats = measure_grey_v(still)
            print("WROTE", still, flush=True)
            reports["shots"][key] = {
                "still": still, "grey": stats, "exposure": exposure,
                "qc": info["qc"], "cam": cam_info, "targets": info.get("targets"),
            }

    jpath = os.path.join(args.out_dir, f"{args.name}_hybrid.json")
    with open(jpath, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print("WROTE", jpath, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
