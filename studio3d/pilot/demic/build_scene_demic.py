"""DEMIC-target pilot: continuous MH body + projected glow masks, full ROM lat pulldown.

Run: studio3d/pilot/blender_cli.sh studio3d/pilot/demic/build_scene_demic.py \\
       --form right --phase 0 --camera front --out output/pilot/demic/_frame
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

DEMIC = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(DEMIC)
ROOT = os.path.dirname(os.path.dirname(PILOT))
STUDIO = os.path.join(ROOT, "studio3d")
sys.path.insert(0, DEMIC)
sys.path.insert(0, PILOT)
sys.path.insert(0, STUDIO)

import importlib.util

_pilot_spec = importlib.util.spec_from_file_location("pilot_build_scene", os.path.join(PILOT, "build_scene.py"))
pilot_scene = importlib.util.module_from_spec(_pilot_spec)
_pilot_spec.loader.exec_module(pilot_scene)

from equipment_demic import make_bar, make_lat_machine  # noqa: E402
from paint_glow_masks import apply_vertex_glow, set_active_glow_channel  # noqa: E402
from pose_lat_pull import pose_lat_pull  # noqa: E402
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
    base = json.load(open(os.path.join(ROOT, "studio3d", "set_v1.json")))
    merged = {**S, **{k: v for k, v in D.items() if not k.startswith("_") and k != "inherits"}}
    merged["version"] = D.get("version", "pilot-demic-1")
    S_mh = {**base, **S}
    S_mh["asset_dir"] = S["asset_dir"]
    S_mh["mannequin"] = {**base.get("mannequin", {}), **S["mannequin"], **D.get("mannequin", {})}
    if D.get("mannequin", {}).get("definition_subsurf"):
        S_mh["mannequin"]["definition_subsurf"] = D["mannequin"]["definition_subsurf"]
    merged["mannequin_mh"] = S_mh
    merged["mannequin"] = {**S.get("mannequin", {}), **D.get("mannequin", {})}
    merged["glow"] = {**S.get("glow", {}), **D.get("glow", {})}
    merged["motion"] = D.get("motion", {})
    merged["render"] = {**S.get("render", {}), **D.get("render", {})}
    merged["attribution"] = {**S.get("attribution", {}), **D.get("attribution", {})}
    return merged


def set_hash(S):
    h = hashlib.sha256(json.dumps(S, sort_keys=True, default=str).encode())
    for rel in (
        "demic/build_scene_demic.py",
        "demic/set_pilot_demic.json",
        "demic/shaders_demic.py",
        "demic/paint_glow_masks.py",
        "demic/pose_lat_pull.py",
        "demic/equipment_demic.py",
        "set_pilot.json",
    ):
        p = os.path.join(PILOT, rel)
        if os.path.isfile(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def assign(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)


def add_contact_shadow(seat_y, seat_h):
    """Subtle dark patch under seat/feet — no vignette."""
    bpy.ops.mesh.primitive_plane_add(size=1.2, location=(0, seat_y - 0.05, 0.002))
    pl = bpy.context.object
    pl.name = "contact_shadow"
    pl.scale = (0.55, 0.42, 1)
    m = bpy.data.materials.new("contact")
    m.use_nodes = True
    bs = m.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (0.08, 0.085, 0.09, 1)
    bs.inputs["Roughness"].default_value = 1.0
    if "Specular IOR Level" in bs.inputs:
        bs.inputs["Specular IOR Level"].default_value = 0.0
    assign(pl, m)
    return pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=("wrong", "right"), default="wrong")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--camera", choices=("front", "side"), default="front")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])

    S = load_settings()
    S_mh = S["mannequin_mh"]
    EX = json.load(open(os.path.join(PILOT, "exercises", "wide_grip_lat_pulldown.json")))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = pilot_scene.setup_render(S, a.scale, a.samples or S["render"]["samples"], a.seed or S["render"]["seed"])

    body, arm, _J = pilot_scene.make_mannequin(S_mh)
    body.name = "mannequin"
    sub = body.modifiers.get("sub")
    if sub and S_mh["mannequin"].get("definition_subsurf", 0):
        sub.levels = int(S_mh["mannequin"]["definition_subsurf"])
        sub.render_levels = sub.levels
    shorts = pilot_scene.make_shorts(S_mh, body)
    egg = bpy.data.objects.get("head_egg")

    from pose_seated_lat import pose_seated_legs  # noqa: E402

    pose_seated_legs(arm, seat_h=S["equipment"]["seat_h"])
    pilot_scene.seat_character(arm, body, S)
    seat_y = float(arm.location.y)
    add_contact_shadow(seat_y, S["equipment"]["seat_h"])

    center, axis, wl, wr, bar_z = pose_lat_pull(arm, EX, a.form, a.phase, S["motion"])
    bar_root, bar_parts = make_bar(S, center, axis, a.form)
    half_w = S["equipment"]["bar"]["width"] / 2
    pilot_scene.upd()
    machine = make_lat_machine(S, seat_y=seat_y, bar_attach=Vector(center))
    pilot_scene.upd()

    lat, bi, delt, trap = apply_vertex_glow(body, arm)
    mask_w = set_active_glow_channel(body, a.form, lat, bi, delt, trap)

    mn = S["mannequin"]
    glow_cfg = S["glow"]
    glow_srgb = glow_cfg["wrong_srgb"] if a.form == "wrong" else glow_cfg["right_srgb"]
    m_body = mat_body_demic(
        "body_demic",
        mn.get("skin_srgb", [48, 50, 54]),
        "GLOW",
        glow_srgb,
        glow_cfg["emission_strength"],
        glow_cfg["mix"],
        mn.get("fibre_scale_uv", 28.0),
        glow_cfg.get("contour_boost", 1.35),
    )
    assign(body, m_body)
    if egg:
        assign(egg, mat_shorts_demic("head_demic", mn.get("skin_srgb", [48, 50, 54]), mn.get("skin_roughness", 0.68)))
    assign(shorts, mat_shorts_demic("shorts", mn.get("shorts_srgb", [8, 8, 9]), S_mh["mannequin"].get("shorts_roughness", 0.88)))

    mm = mat_machine_demic("machine", S["equipment"]["machine_srgb"], S["equipment"]["machine_roughness"])
    for o in machine:
        assign(o, mm)

    b = S["equipment"]["bar"]
    m_steel = mat_bar_steel("steel", b["steel_srgb"])
    m_cob = mat_machine_demic("cobalt", b["cobalt_srgb"], 0.55)
    m_gold = mat_machine_demic("gold", b["gold_srgb"], 0.4)
    accent_srgb = glow_cfg["wrong_srgb"] if a.form == "wrong" else glow_cfg["right_srgb"]
    m_accent = mat_bar_accent("bar_line", accent_srgb, strength=10.0)
    for p in bar_parts:
        if p.name == "bar_accent_line":
            assign(p, m_accent)
        elif p.name.startswith("bar_plate"):
            assign(p, m_cob)
        elif p.name.startswith("bar_rim"):
            assign(p, m_gold)
        else:
            assign(p, m_steel)
    for o in machine:
        if o.name.startswith("stack_pin"):
            assign(o, m_cob)
        elif o.name.startswith("stack_rim"):
            assign(o, m_gold)

    chest = arm.matrix_world @ arm.pose.bones["spine02"].head
    to = S["cameras"][a.camera].get("target_offset_m", [0, 0, 0])
    tgt = (to[0], float(chest.y) + to[1], float(chest.z) + to[2])
    cam = pilot_scene.add_camera(S, a.camera, tgt)
    pilot_scene.add_flat_lights((0, float(chest.y) - 0.2, 0))

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sc.camera = cam
    sc.render.filepath = out + "_beauty.png"
    bpy.ops.render.render(write_still=True)

    rx, ry = S["render"]["res"]

    def P(bn, tail=False):
        return arm.matrix_world @ (arm.pose.bones[bn].tail if tail else arm.pose.bones[bn].head)

    wr_l = np.array(P("wrist.L", tail=True))
    wr_r = np.array(P("wrist.R", tail=True))
    head_z = float(P("head")[2])
    bar_c = np.array(bar_root.location)
    axis_n = np.array(bar_root.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0)))
    axis_n = axis_n / (np.linalg.norm(axis_n) + 1e-12)

    def grip_err(wr):
        a0 = bar_c - axis_n * half_w
        b0 = bar_c + axis_n * half_w
        ab = b0 - a0
        t = np.clip(float(np.dot(wr - a0, ab) / (np.dot(ab, ab) + 1e-12)), 0.0, 1.0)
        return float(np.linalg.norm(wr - (a0 + t * ab)))

    grip_d = max(grip_err(wr_l), grip_err(wr_r))
    wrist_z = max(float(wr_l[2]), float(wr_r[2]))
    shc = (np.array(P("upperarm01.L")) + np.array(P("upperarm01.R"))) / 2
    hip = (np.array(P("upperleg01.L")) + np.array(P("upperleg01.R"))) / 2
    mw = body.matrix_world
    torso_idx = []
    for i, v in enumerate(body.data.vertices):
        p = np.array(mw @ v.co)
        if hip[2] - 0.05 < p[2] < shc[2] + 0.15 and abs(p[0]) < 0.35:
            torso_idx.append(i)
    glow_px = float(mask_w[torso_idx].mean()) if torso_idx else 0.0

    meta = {
        "pilot_version": S.get("version", "pilot-demic-1"),
        "set_hash": set_hash(S),
        "exercise": "wide_grip_lat_pulldown",
        "form": a.form,
        "phase": a.phase,
        "camera": a.camera,
        "trunk_from_vertical_deg": round(pilot_scene.trunk_from_vertical(arm), 1),
        "bar_z_m": round(float(bar_z), 3),
        "wrist_z_m": round(wrist_z, 3),
        "head_z_m": round(head_z, 3),
        "wrist_above_head_at_phase": wrist_z >= head_z - 0.05 if a.phase < 0.15 else None,
        "glow_torso_mean": round(glow_px, 4),
        "grip_distance_m": round(grip_d, 4),
        "grip_ok": grip_d < 0.06,
        "render_seed": sc.cycles.seed,
        "attribution": S["attribution"],
    }
    json.dump(meta, open(out + "_meta.json", "w"), indent=1)
    print("META", json.dumps(meta))


if __name__ == "__main__":
    main()
