"""Day-4-look pilot: locked lat-pulldown set, one frame or animation strip.

Run: studio3d/pilot/blender_cli.sh studio3d/pilot/build_scene.py \\
       --form wrong --phase 0 --camera front --out output/pilot/lat/_frame
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

PILOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(PILOT))
sys.path.insert(0, PILOT)
sys.path.insert(0, os.path.join(ROOT, "studio3d"))

from anatomy_bind import import_and_bind_muscles, muscle_glow_pattern  # noqa: E402
from equipment import make_bar as make_bar_v2, make_lat_machine as make_lat_machine_v2  # noqa: E402
from pose_seated_lat import pose_seated_lat_pulldown, pose_seated_legs  # noqa: E402
from shaders import assign as assign_mat, mat_muscle_fibre  # noqa: E402

from build_scene import (  # noqa: E402
    load_mh,
    make_mannequin,
    make_shorts,
    aim,
    unit,
    upd,
    add_camera,
)

ID_COL = {
    "skin": (1, 0, 0),
    "shorts": (0, 1, 0),
    "machine": (0, 0, 1),
    "bar": (1, 1, 0),
    "glow": (1, 0, 1),
}


def srgb_to_linear(c):
    out = []
    for u in c:
        u = u / 255.0
        out.append(u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def set_hash(S):
    h = hashlib.sha256(json.dumps(S, sort_keys=True).encode())
    for fn in (
        "build_scene.py",
        "set_pilot.json",
        "anatomy_ecorche.py",
        "anatomy_bind.py",
        "pose_seated_lat.py",
        "shaders.py",
        "equipment.py",
    ):
        p = os.path.join(PILOT, fn)
        if os.path.isfile(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def mat_emit(name, rgb, strength=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*rgb, 1)
    e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def mat_fibre_skin(name, rgb_lin, rough, spec, bump_strength, scale):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Detail"].default_value = 8.0
    tex.inputs["Roughness"].default_value = 0.55
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.inputs["Scale"].default_value = scale * 1.7
    wave.inputs["Distortion"].default_value = 2.5
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.65
    bump_n = nt.nodes.new("ShaderNodeBump")
    bump_n.inputs["Strength"].default_value = bump_strength
    bump_n.inputs["Distance"].default_value = 0.015
    nt.links.new(tex.outputs["Fac"], mix.inputs[1])
    nt.links.new(wave.outputs["Color"], mix.inputs[2])
    nt.links.new(mix.outputs["Color"], bump_n.inputs["Height"])
    bsdf.inputs["Base Color"].default_value = (*rgb_lin, 1)
    bsdf.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = spec
    nt.links.new(bump_n.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def mat_glow_shell(name, rgb_lin, strength, mix_fac):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb_lin, 1)
    em.inputs["Strength"].default_value = strength
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = mix_fac
    bsdf.inputs["Base Color"].default_value = (*rgb_lin, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    m.blend_method = "BLEND"
    return m


def assign(ob, m):
    ob.data.materials.clear()
    ob.data.materials.append(m)


def setup_render(S, scale, samples, seed):
    sc = bpy.context.scene
    r = S["render"]
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples or r["samples"]
    sc.cycles.use_denoising = False  # apt Blender build lacks OIDN; keep deterministic
    sc.cycles.use_animated_seed = False
    sc.cycles.seed = seed or r["seed"]
    sc.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    sc.cycles.filter_width = 1.5
    sc.render.threads_mode = "FIXED"
    sc.render.threads = r["threads"]
    sc.render.resolution_x, sc.render.resolution_y = r["res"]
    sc.render.resolution_percentage = int(scale * 100)
    sc.render.film_transparent = r["film_transparent"]
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    w = bpy.data.worlds.new("world")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*srgb_to_linear(S["world_srgb"]), 1)
    bg.inputs[1].default_value = 1.0
    return sc


def add_flat_lights(anchor):
    """Even, shadowless studio: many equal soft boxes — no dark side on the figure."""
    energies = [85, 85, 85, 85, 70, 70]
    locs = [
        (-2.2, -2.0, 2.4),
        (2.2, -2.0, 2.4),
        (0, -3.0, 2.0),
        (0, 2.5, 2.2),
        (-2.0, 0.5, 1.2),
        (2.0, 0.5, 1.2),
    ]
    for i, (loc, en) in enumerate(zip(locs, energies)):
        ld = bpy.data.lights.new(f"flat_{i}", "AREA")
        ld.energy = en
        ld.size = 3.5
        ld.shape = "SQUARE"
        o = bpy.data.objects.new(f"flat_{i}", ld)
        bpy.context.collection.objects.link(o)
        o.location = Vector(loc) + Vector(anchor)
        tgt = Vector(anchor) + Vector((0, 0, 0.9))
        o.rotation_euler = (tgt - o.location).to_track_quat("-Z", "Y").to_euler()


def box(name, size, loc, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    o.rotation_euler = rot
    return o


def make_lat_machine(S):
    e = S["equipment"]
    parts = []
    sh = e["seat_h"]
    parts.append(box("machine_base", (e["stack_w"] + 0.35, e["stack_d"] + 0.25, 0.06), (0, 0.15, 0.03)))
    parts.append(
        box(
            "machine_stack",
            (e["stack_w"], e["stack_d"], e["stack_h"]),
            (e["stack_x"], 0.05, sh + e["stack_h"] / 2 - 0.15),
        )
    )
    parts.append(box("machine_seat", (0.36, 0.34, 0.06), (0, 0.05, sh)))
    parts.append(box("machine_thigh", (0.36, 0.22, e["thigh_pad_h"]), (0, -0.22, sh - e["thigh_pad_h"] / 2 + 0.04)))
    parts.append(box("machine_post", (0.08, 0.08, 1.2), (e["stack_x"] * 0.55, 0.05, sh + 0.55)))
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.06, depth=0.12, location=(e["stack_x"], 0.05, sh + e["stack_h"] - 0.2))
    parts.append(bpy.context.object)
    parts[-1].name = "machine_pulley"
    return parts


def make_bar(S):
    b = S["equipment"]["bar"]
    root = bpy.data.objects.new("bar_root", None)
    bpy.context.collection.objects.link(root)
    parts = []
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32, radius=b["handle_r"], depth=b["width"], location=(0, 0, 0), rotation=(0, math.pi / 2, 0)
    )
    hnd = bpy.context.object
    hnd.name = "bar_handle"
    parts.append(hnd)
    for sgn in (-1, 1):
        cx = sgn * (b["width"] / 2 + b["plate_r"] * 0.55)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=b["plate_r"], depth=b["plate_t"], location=(cx, 0, 0), rotation=(0, math.pi / 2, 0)
        )
        pl = bpy.context.object
        pl.name = f"bar_plate_{sgn}"
        parts.append(pl)
        bpy.ops.mesh.primitive_torus_add(
            major_radius=b["plate_r"] * 0.92,
            minor_radius=0.006,
            location=(cx, 0, sgn * b["plate_t"] * 0.55),
            rotation=(math.pi / 2, 0, 0),
        )
        rim = bpy.context.object
        rim.name = f"bar_rim_{sgn}"
        parts.append(rim)
    for p in parts:
        p.parent = root
    root.location = (0, 0, 0)
    return root, parts


def dominant_group(ob, prefixes):
    gi = {g.index: g.name for g in ob.vertex_groups}
    keep = set()
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    dl = bm.verts.layers.deform.verify()
    for v in bm.verts:
        d = v[dl]
        if not d:
            continue
        g, w = max(d.items(), key=lambda t: t[1])
        name = gi.get(g, "")
        if any(name.startswith(p) or p in name for p in prefixes):
            keep.add(v.index)
    bm.free()
    return keep


def make_glow_shell(body, arm, region, name):
    sh = body.copy()
    sh.data = body.data.copy()
    sh.name = name
    bpy.context.collection.objects.link(sh)
    if region == "lat":
        prefs = ("spine03", "spine04", "spine05", "upperarm01", "shoulder01")
    elif region == "biceps":
        prefs = ("upperarm02", "lowerarm01", "lowerarm02")
    else:
        prefs = ("shoulder01", "upperarm01", "clavicle")
    keep = dominant_group(sh, prefs)
    me = sh.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not all(v.index in keep for v in f.verts)], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    sol = sh.modifiers.new("off", "SOLIDIFY")
    sol.thickness = 0.003
    sol.offset = 1.0
    sh.parent = arm
    sh.modifiers.new("arm", "ARMATURE").object = arm
    return sh


def pose_lat(arm, EX, form, phase):
    f0, f1 = EX["forms"][form]["stretch"], EX["forms"][form]["contract"]
    lerp = lambda k: f0[k] + (f1[k] - f0[k]) * phase
    lean = math.radians(lerp("trunk_lean_deg"))
    ef = math.radians(lerp("elbow_flexion_deg"))
    fwd = math.radians(lerp("upper_arm_forward_deg"))
    bar_z = lerp("bar_y_m")
    bar_y = -0.05
    Tt = unit((0, -math.sin(lean), math.cos(lean)))
    N = unit((0, math.cos(lean), math.sin(lean)))
    for bn in ("spine05", "spine04", "spine03", "spine02", "spine01"):
        aim(arm, bn, Tt)
    for bn in ("neck01", "neck02", "neck03", "head"):
        aim(arm, bn, unit(Tt * 0.9 + N * 0.15))
    for sd, sx in (("L", 1), ("R", -1)):
        X = np.array((sx, 0.0, 0.0))
        u = unit(-Tt * math.cos(fwd) + X * math.sin(fwd) + N * 0.08)
        w = unit(N - u * np.dot(N, u))
        fa = unit(u * math.cos(ef) + w * math.sin(ef))
        aim(arm, f"upperarm01.{sd}", u)
        aim(arm, f"upperarm02.{sd}", u)
        aim(arm, f"lowerarm01.{sd}", fa)
        aim(arm, f"lowerarm02.{sd}", fa)
        aim(arm, f"wrist.{sd}", fa)
        for fn in range(2, 6):
            for seg in (1, 2, 3):
                bn = f"finger{fn}-{seg}.{sd}"
                if bn in arm.pose.bones:
                    pb = arm.pose.bones[bn]
                    pb.rotation_mode = "XYZ"
                    pb.rotation_euler = (math.radians(35), 0, 0)
    upd()
    return bar_y, bar_z


def seat_character(arm, body, S):
    gap = S["mannequin"]["contact_seat_gap_m"]
    sh = S["equipment"]["seat_h"]
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    V = np.array([ev.matrix_world @ v.co for v in ev.to_mesh().vertices])
    ev.to_mesh_clear()
    pelvis = (np.array(arm.matrix_world @ arm.pose.bones["upperleg01.L"].head) + np.array(arm.matrix_world @ arm.pose.bones["upperleg01.R"].head)) / 2
    seat_sel = (np.abs(V[:, 0] - pelvis[0]) < 0.28) & (V[:, 1] > pelvis[1] - 0.35) & (V[:, 1] < pelvis[1] + 0.08) & (V[:, 2] < pelvis[2] + 0.05)
    zs = V[seat_sel, 2].min() if seat_sel.any() else pelvis[2] - 0.12
    arm.location = arm.location + Vector((-pelvis[0], -pelvis[1], sh + gap - zs))
    upd()


def ang(a_, b_, c_):
    u, v = np.array(a_) - np.array(b_), np.array(c_) - np.array(b_)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(u @ v / nu / nv, -1, 1))))


def trunk_from_vertical(arm):
    shc = (np.array(arm.matrix_world @ arm.pose.bones["upperarm01.L"].head) + np.array(arm.matrix_world @ arm.pose.bones["upperarm01.R"].head)) / 2
    hip = (np.array(arm.matrix_world @ arm.pose.bones["upperleg01.L"].head) + np.array(arm.matrix_world @ arm.pose.bones["upperleg01.R"].head)) / 2
    t = shc - hip
    t = t / np.linalg.norm(t)
    vert = np.array((0, 0, 1.0))
    return ang(hip + vert, hip, hip + t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=("wrong", "right"), default="wrong")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--camera", choices=("front", "side"), default="front")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-id", action="store_true")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])

    S = json.load(open(os.path.join(PILOT, "set_pilot.json")))
    base = json.load(open(os.path.join(ROOT, "studio3d", "set_v1.json")))
    S_mh = {**base, **S}
    S_mh["asset_dir"] = S["asset_dir"]
    S_mh["mannequin"] = {**base["mannequin"], **S["mannequin"]}

    EX = json.load(open(os.path.join(PILOT, "exercises", "wide_grip_lat_pulldown.json")))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = setup_render(S, a.scale, a.samples, a.seed or None)

    body, arm, _J = make_mannequin(S_mh)
    shorts = make_shorts(S_mh, body)
    egg = bpy.data.objects.get("head_egg")
    pose_seated_legs(arm, seat_h=S["equipment"]["seat_h"])
    seat_character(arm, body, S)
    seat_y = float(arm.location.y)
    bar_w = S["equipment"]["bar"]["width"]
    center, axis, wl, wr = pose_seated_lat_pulldown(arm, EX, a.form, a.phase, bar_w)
    # Nudge bar toward motion-standard height while keeping grip on wrists
    f0, f1 = EX["forms"][a.form]["stretch"], EX["forms"][a.form]["contract"]
    target_z = f0["bar_y_m"] + (f1["bar_y_m"] - f0["bar_y_m"]) * a.phase
    center = np.array(center, float)
    center[2] += (target_z - center[2]) * 0.35
    bar_root, bar_parts = make_bar_v2(S, center, axis)
    axv = Vector(axis.tolist())
    for sd, w in (("L", wl), ("R", wr)):
        g = bpy.data.objects.new(f"grip_{sd}", None)
        bpy.context.collection.objects.link(g)
        g.parent = bar_root
        off = float(np.dot(np.array(w) - center, axis))
        g.location = axv * off
        pb = arm.pose.bones.get(f"hand.{sd}") or arm.pose.bones.get(f"wrist.{sd}")
        if pb:
            c = pb.constraints.new("COPY_LOCATION")
            c.target = g
            c.influence = 0.62
    upd()
    machine = make_lat_machine_v2(S, seat_y=seat_y, bar_attach=Vector(center))
    upd()
    muscles = import_and_bind_muscles(arm, body)
    glow_cfg = S["glow"]
    gt = muscle_glow_pattern(a.form)
    assign(body, mat_fibre_skin(
        "body_under",
        skin_lin := srgb_to_linear(S["mannequin"]["skin_srgb"]),
        S["mannequin"]["skin_roughness"],
        S["mannequin"]["skin_specular"],
        S["mannequin"]["fibre_bump_strength"],
        S["mannequin"]["fibre_scale"],
    ))
    m_charcoal = mat_muscle_fibre("muscle_charcoal", S["mannequin"]["skin_srgb"])
    glow_srgb = glow_cfg["wrong_srgb" if a.form == "wrong" else "right_srgb"]
    m_glow = mat_muscle_fibre(
        "muscle_glow",
        S["mannequin"]["skin_srgb"],
        glow_srgb=glow_srgb,
        glow_strength=glow_cfg["emission_strength"],
        mix_glow=glow_cfg["mix"],
    )
    glow_objs = []
    for o in muscles:
        if gt.search(o.name):
            assign_mat(o, m_glow)
            glow_objs.append(o)
        else:
            assign_mat(o, m_charcoal)

    m_skin = mat_fibre_skin(
        "skin",
        skin_lin,
        S["mannequin"]["skin_roughness"],
        S["mannequin"]["skin_specular"],
        S["mannequin"]["fibre_bump_strength"],
        S["mannequin"]["fibre_scale"],
    )
    if egg:
        assign(egg, m_skin)
    sm = mat_fibre_skin("shorts_m", srgb_to_linear(S["mannequin"]["shorts_srgb"]), S["mannequin"]["shorts_roughness"], 0.05, 0.05, 20)
    assign(shorts, sm)

    mm = mat_fibre_skin("machine", srgb_to_linear(S["equipment"]["machine_srgb"]), S["equipment"]["machine_roughness"], 0.12, 0.04, 12)
    for o in machine:
        assign(o, mm)

    b = S["equipment"]["bar"]
    m_steel = mat_fibre_skin("steel", srgb_to_linear(b["steel_srgb"]), 0.35, 0.45, 0.08, 80)
    m_cob = mat_fibre_skin("cobalt", srgb_to_linear(b["cobalt_srgb"]), 0.55, 0.15, 0.02, 40)
    m_gold = mat_fibre_skin("gold", srgb_to_linear(b["gold_srgb"]), 0.4, 0.35, 0.03, 60)
    for p in bar_parts:
        assign(p, m_steel)
    for o in machine:
        if o.name.startswith("stack_pin"):
            assign(o, m_cob)
        elif o.name.startswith("stack_rim"):
            assign(o, m_gold)

    chest = arm.matrix_world @ arm.pose.bones["spine02"].head
    to = S["cameras"][a.camera].get("target_offset_m", [0, 0, 0])
    tgt = (to[0], float(chest.y) + to[1], float(chest.z) + to[2])
    cam = add_camera(S, a.camera, tgt)
    add_flat_lights((0, float(chest.y) - 0.2, 0))

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    sc.camera = cam
    sc.render.filepath = out + "_beauty.png"
    bpy.ops.render.render(write_still=True)

    if not a.skip_id:
        sc.cycles.samples = 1
        sc.cycles.use_denoising = False
        sc.cycles.pixel_filter_type = "BOX"
        sc.cycles.filter_width = 0.01
        ids = {
            "skin": [body] + muscles + ([egg] if egg else []),
            "shorts": [shorts],
            "machine": machine,
            "bar": bar_parts,
            "glow": glow_objs,
        }
        saved = {}
        for k, obs in ids.items():
            mi = mat_emit("id_" + k, ID_COL[k])
            for o in obs:
                saved[o.name] = list(o.data.materials)
                assign(o, mi)
        sc.render.filepath = out + "_id.png"
        bpy.ops.render.render(write_still=True)
        for oname, mats in saved.items():
            ob = bpy.data.objects.get(oname)
            if ob:
                ob.data.materials.clear()
                for m in mats:
                    ob.data.materials.append(m)

    rx, ry = S["render"]["res"]

    def px(p):
        c = world_to_camera_view(sc, cam, Vector(p))
        return [round(c.x * rx, 1), round((1 - c.y) * ry, 1)]

    P = lambda bn, tail=False: arm.matrix_world @ (arm.pose.bones[bn].tail if tail else arm.pose.bones[bn].head)
    hipc = (np.array(P("upperleg01.L")) + np.array(P("upperleg01.R"))) / 2
    shc = (np.array(P("upperarm01.L")) + np.array(P("upperarm01.R"))) / 2
    angles = {}
    for sd in "LR":
        angles[sd] = {
            "elbow_flexion": round(180 - ang(P(f"upperarm01.{sd}"), P(f"lowerarm01.{sd}"), P(f"wrist.{sd}")), 1),
            "arm_torso": round(ang(P(f"lowerarm01.{sd}"), P(f"upperarm01.{sd}"), np.array(P(f"upperarm01.{sd}")) + (hipc - shc)), 1),
        }
    wr_l = np.array(P("wrist.L", tail=True))
    wr_r = np.array(P("wrist.R", tail=True))
    bar_c = np.array(bar_root.location)
    half_w = S["equipment"]["bar"]["width"] / 2

    axis = np.array(bar_root.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0)))
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    def grip_err(wr):
        a = bar_c - axis * half_w
        b = bar_c + axis * half_w
        ab = b - a
        t = np.clip(float(np.dot(wr - a, ab) / (np.dot(ab, ab) + 1e-12)), 0.0, 1.0)
        closest = a + t * ab
        return float(np.linalg.norm(wr - closest))

    grip_d = float(max(grip_err(wr_l), grip_err(wr_r)))
    meta = {
        "pilot_version": S["version"],
        "set_hash": set_hash(S),
        "exercise": "wide_grip_lat_pulldown",
        "form": a.form,
        "phase": a.phase,
        "camera": a.camera,
        "trunk_from_vertical_deg": round(trunk_from_vertical(arm), 1),
        "angles_3d": angles,
        "bar_world": [float(center[1]), float(center[2])],
        "grip_distance_m": round(grip_d, 4),
        "grip_ok": grip_d < 0.08,
        "render_seed": sc.cycles.seed,
        "attribution": S["attribution"],
    }
    json.dump(meta, open(out + "_meta.json", "w"), indent=1)
    print("META", json.dumps(meta))


if __name__ == "__main__":
    main()
