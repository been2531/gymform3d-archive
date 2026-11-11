"""Round 6 smoke builder: MH CC0 + motion-standards FK pose, discrete glow, R6 machine layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

DEMIC = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(DEMIC)
ROOT = os.path.dirname(os.path.dirname(PILOT))
STUDIO = os.path.join(ROOT, "studio3d")
sys.path.insert(0, DEMIC)
sys.path.insert(0, PILOT)
sys.path.insert(0, STUDIO)

import importlib.util

_spec = importlib.util.spec_from_file_location("pilot_build_scene", os.path.join(PILOT, "build_scene.py"))
pilot_scene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot_scene)

from discrete_glow_masks import apply_to_mesh  # noqa: E402
from equipment_r6 import make_grip_bar, make_lat_station  # noqa: E402
from pose_seated_lat import (  # noqa: E402
    _solve_arm_to_wrist,
    bar_grip_points,
    pose_seated_lat_pulldown,
    pose_seated_legs,
    pose_torso,
    shrug_shoulders,
)
from shaders_demic import (  # noqa: E402
    mat_bar_accent,
    mat_bar_steel,
    mat_body_demic,
    mat_machine_demic,
    mat_shorts_demic,
)


def load_settings():
    S = json.load(open(os.path.join(PILOT, "set_pilot.json")))
    D = json.load(open(os.path.join(DEMIC, "set_pilot_demic.json")))
    R6 = json.load(open(os.path.join(DEMIC, "set_pilot_r6.json")))
    base = json.load(open(os.path.join(STUDIO, "set_v1.json")))
    merged = {**S, **D, **R6}
    merged["version"] = R6.get("version", "pilot-demic-r6")
    merged["render"] = {**S.get("render", {}), **D.get("render", {})}
    merged["equipment"] = S.get("equipment", {})
    merged["cameras"] = S.get("cameras", {})
    merged["world_srgb"] = D.get("world_srgb", S.get("world_srgb"))
    merged["glow"] = {**S.get("glow", {}), **D.get("glow", {})}
    S_mh = {**base, **S}
    S_mh["asset_dir"] = S["asset_dir"]
    S_mh["mannequin"] = {**base.get("mannequin", {}), **S["mannequin"], **D.get("mannequin", {})}
    merged["mannequin_mh"] = S_mh
    merged["mannequin"] = {**S.get("mannequin", {}), **D.get("mannequin", {})}
    merged["attribution"] = {**S.get("attribution", {}), **D.get("attribution", {}), **R6.get("attribution", {})}
    return merged


def assign(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)


def add_contact_shadow(seat_y):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, seat_y - 0.05, 0.002))
    pl = bpy.context.object
    pl.name = "contact_shadow"
    pl.scale = (0.5, 0.38, 1)
    m = bpy.data.materials.new("contact")
    m.use_nodes = True
    bs = m.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (0.08, 0.085, 0.09, 1)
    bs.inputs["Roughness"].default_value = 1.0
    assign(pl, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=("wrong", "right"), default="wrong")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--camera", choices=("front", "side"), default="front")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])

    S = load_settings()
    S_mh = S["mannequin_mh"]
    EX = json.load(open(os.path.join(PILOT, "exercises", "wide_grip_lat_pulldown.json")))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = pilot_scene.setup_render(S, a.scale, a.samples, a.seed or S["render"]["seed"])

    body, arm, _ = pilot_scene.make_mannequin(S_mh)
    body.name = "mannequin"
    shorts = pilot_scene.make_shorts(S_mh, body)
    egg = bpy.data.objects.get("head_egg")

    pose_seated_legs(arm, seat_h=S["equipment"]["seat_h"])
    pilot_scene.seat_character(arm, body, S)
    seat_y = float(arm.location.y)
    add_contact_shadow(seat_y)

    bar_w = S["equipment"]["bar"]["width"]
    f0, f1 = EX["forms"][a.form]["stretch"], EX["forms"][a.form]["contract"]
    lean = f0["trunk_lean_deg"] + (f1["trunk_lean_deg"] - f0["trunk_lean_deg"]) * a.phase
    pose_seated_legs(arm, seat_h=S["equipment"]["seat_h"])
    pose_torso(arm, lean)
    if a.form == "wrong":
        shrug_shoulders(arm, 0.35 * (0.4 + 0.6 * a.phase))
    EX_arm = json.loads(json.dumps(EX))
    if a.phase < 0.15:
        EX_arm["forms"][a.form]["stretch"]["bar_y_m"] = S["motion"]["stretch_bar_z_m"]
        EX_arm["forms"][a.form]["stretch"]["elbow_flexion_deg"] = 168
        EX_arm["forms"][a.form]["stretch"]["upper_arm_forward_deg"] = 14 if a.form == "right" else 22
    wl, wr, _c = bar_grip_points(arm, EX_arm, a.form, min(a.phase, 0.001) if a.phase < 0.15 else a.phase, bar_w)
    pole = 0.42 if a.form == "right" else 0.5
    _solve_arm_to_wrist(arm, "L", wl, pole)
    _solve_arm_to_wrist(arm, "R", wr, pole)
    from pose_seated_lat import grip_fingers

    grip_fingers(arm, "L")
    grip_fingers(arm, "R")
    pilot_scene.upd()
    if a.phase >= 0.15:
        pose_torso(arm, lean)
        if a.form == "wrong":
            shrug_shoulders(arm, 0.35 * (0.4 + 0.6 * a.phase))
        from pose_seated_lat import pose_arms_from_standards

        pose_arms_from_standards(arm, EX, a.form, a.phase)
        grip_fingers(arm, "L")
        grip_fingers(arm, "R")
        pilot_scene.upd()
        wl = np.array(arm.matrix_world @ (arm.pose.bones["hand.L"].tail if "hand.L" in arm.pose.bones else arm.pose.bones["wrist.L"].tail))
        wr = np.array(arm.matrix_world @ (arm.pose.bones["hand.R"].tail if "hand.R" in arm.pose.bones else arm.pose.bones["wrist.R"].tail))
        center = (wl + wr) / 2
        axis = (wr - wl) / (np.linalg.norm(wr - wl) + 1e-9)
    else:
        wl = np.array(arm.matrix_world @ arm.pose.bones["hand.L"].tail if "hand.L" in arm.pose.bones else arm.pose.bones["wrist.L"].tail)
        wr = np.array(arm.matrix_world @ arm.pose.bones["hand.R"].tail if "hand.R" in arm.pose.bones else arm.pose.bones["wrist.R"].tail)
        center = (wl + wr) / 2
        axis = (wr - wl) / (np.linalg.norm(wr - wl) + 1e-9)
    grip_span = float(np.linalg.norm(wr - wl))
    bar_root, bar_parts = make_grip_bar(S, center, axis, grip_span)
    machine = make_lat_station(S, seat_y, center)
    pilot_scene.upd()

    mask_w, lat, bi, delt, trap = apply_to_mesh(body, a.form)

    mn = S["mannequin"]
    glow_cfg = S["glow"]
    glow_srgb = glow_cfg["wrong_srgb"] if a.form == "wrong" else glow_cfg["right_srgb"]
    assign(
        body,
        mat_body_demic(
            "body_r6",
            mn.get("skin_srgb", [48, 50, 54]),
            "GLOW",
            glow_srgb,
            glow_cfg["emission_strength"],
            glow_cfg["mix"],
            mn.get("fibre_scale_uv", 28.0),
            glow_cfg.get("contour_boost", 1.35),
        ),
    )
    if egg:
        assign(egg, mat_shorts_demic("head", mn.get("skin_srgb", [48, 50, 54]), 0.68))
    assign(shorts, mat_shorts_demic("shorts", mn.get("shorts_srgb", [8, 8, 9]), 0.88))

    mm = mat_machine_demic("machine", S["equipment"]["machine_srgb"], S["equipment"]["machine_roughness"])
    m_cob = mat_machine_demic("cobalt", S["equipment"]["bar"]["cobalt_srgb"], 0.55)
    m_gold = mat_machine_demic("gold", S["equipment"]["bar"]["gold_srgb"], 0.4)
    m_steel = mat_bar_steel("steel", S["equipment"]["bar"]["steel_srgb"])
    m_accent = mat_bar_accent("accent", glow_srgb, 9.0)
    for o in machine:
        assign(o, mm)
        if o.name.startswith("stack_pin"):
            assign(o, m_cob)
        elif o.name.startswith("stack_rim"):
            assign(o, m_gold)
    for p in bar_parts:
        assign(p, m_accent if p.name == "bar_accent_line" else m_steel)

    cam_key = a.camera
    if cam_key == "side" and "side" in S.get("cameras_r6", {}):
        cam_cfg = {**S["cameras"]["side"], **S["cameras_r6"]["side"]}
        S = {**S, "cameras": {**S["cameras"], "side": cam_cfg}}
    chest = arm.matrix_world @ arm.pose.bones["spine02"].head
    to = S["cameras"][cam_key].get("target_offset_m", [0, 0, 0])
    tgt = (to[0], float(chest.y) + to[1], float(chest.z) + to[2])
    cam = pilot_scene.add_camera(S, cam_key, tgt)
    pilot_scene.add_flat_lights((0, float(chest.y) - 0.2, 0))

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sc.camera = cam
    sc.render.filepath = out + "_beauty.png"
    bpy.ops.render.render(write_still=True)

    def P(bn, tail=False):
        return arm.matrix_world @ (arm.pose.bones[bn].tail if tail else arm.pose.bones[bn].head)

    wr_l = np.array(P("wrist.L", tail=True))
    wr_r = np.array(P("wrist.R", tail=True))
    head_z = float(P("head")[2])
    meta = {
        "pilot_version": S["version"],
        "exercise": "wide_grip_lat_pulldown",
        "form": a.form,
        "phase": a.phase,
        "camera": a.camera,
        "trunk_from_vertical_deg": round(pilot_scene.trunk_from_vertical(arm), 1),
        "wrist_z_m": round(max(float(wr_l[2]), float(wr_r[2])), 3),
        "head_z_m": round(head_z, 3),
        "glow_active_mean": round(float(mask_w.mean()), 4),
        "glow_active_max": round(float(mask_w.max()), 4),
        "attribution": S["attribution"],
    }
    json.dump(meta, open(out + "_meta.json", "w"), indent=1)
    print("META", json.dumps(meta))


if __name__ == "__main__":
    main()
