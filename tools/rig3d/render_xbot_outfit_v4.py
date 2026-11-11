#!/usr/bin/env python3
"""Render the X Bot outfit v4 comparison sheet and a short jab preview.

  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v4.py
  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v4.py -- --eevee-stills
  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v4.py -- --eevee-stills --no-preview

Outfit A v4: glossy black CC0-matched gloves + charcoal beanie + charcoal
satin trunks + fitted charcoal tank. Columns: plain | v3 A | v4 A.
Rows: Idle guard, Jab full, Jab return. Close-ups: v4 glove next to the
CC0 reference at the same long-axis scale, plus a garments close-up.

Writes (gitignored):
  output/_basemodels/eval/xbot_outfit_v4.png
  output/_basemodels/eval/xbot_outfit_v4_jab_preview.mp4
  output/_basemodels/eval/xbot_outfit_v4_facts.json
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import eval_xbot_boxing as xbot  # noqa: E402
import render_xbot_outfit_v1 as v1  # noqa: E402
from xbot_outfit import (  # noqa: E402
    COLORWAYS, apply_outfit, clear_outfit, garment_poke_report, _clean,
)

ROOT = xbot.ROOT
SRC_DIR = xbot.SRC_DIR
OUT_DIR = xbot.OUT_DIR
RES = xbot.RES
CLOSE_RES = (540, 720)
MARGIN = 0.06
CC0_SRC = "/tmp/glove_refs/pair_cc0.jpg"
V4_ITEMS = ("gloves", "beanie", "trunks", "tank")

VARIANTS = (
    {"key": "plain", "title": "plain X Bot", "items": (), "colorway": "A", "version": "v4"},
    {"key": "v3A", "title": "v3 A  ·  crimson 0.8× head + crown-rib beanie",
     "items": ("gloves", "beanie"), "colorway": "A", "version": "v3"},
    {"key": "v4A", "title": "v4 A  ·  glossy black gloves + charcoal kit",
     "items": V4_ITEMS, "colorway": "A", "version": "v4"},
)

POSES = (
    {"clip": "idle", "file": "Fighting Idle.fbx", "stage": "guard", "title": "Fighting Idle — guard"},
    {"clip": "jab", "file": "Lead Jab.fbx", "stage": "full", "title": "Lead Jab — full extension"},
    {"clip": "jab", "file": "Lead Jab.fbx", "stage": "return", "title": "Lead Jab — return"},
)


def _cli():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    eevee_stills = "--eevee-stills" in argv
    no_preview = "--no-preview" in argv
    samples = None
    if "--samples" in argv:
        samples = int(argv[argv.index("--samples") + 1])
    return eevee_stills, no_preview, samples


def _objects(handle, prefix):
    out = []
    for n in handle.get("objects") or []:
        obj = bpy.data.objects.get(n)
        if obj and n.startswith(prefix):
            out.append(obj)
    return out


def _return_frame(state):
    keys = state["keys"]
    full = int(keys["full"])
    f1 = int(state["f1"])
    if f1 > full + 2:
        return f1
    # Clip ends at extension; step a couple of frames back toward guard if possible.
    guard = int(keys.get("guard", state["f0"]))
    return max(full, min(f1, full + max(1, (f1 - guard) // 8)))


def set_pose_ex(state, stage):
    if stage == "return":
        f = _return_frame(state)
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        if state["lock"]:
            xbot.lock_root_xy(state["arm"], state["bones"], state["start_xy"])
            bpy.context.view_layer.update()
        return f
    return v1.set_pose(state, stage)


def _poke(handle, meshes, kind):
    if kind == "trunks":
        objs = _objects(handle, "XBotTrunks_")
        pred = lambda n: any(k in _clean(n) for k in ("hip", "pelvis", "spine", "upleg", "thigh"))
    else:
        objs = _objects(handle, "XBotTank_")
        pred = lambda n: (
            any(k in _clean(n) for k in ("spine", "chest", "abdomen", "torso"))
            and not any(k in _clean(n) for k in ("shoulder", "arm", "hand", "neck", "head"))
        )
    if not objs:
        return None
    return garment_poke_report(meshes, objs[0], pred)


def setup_oriented_camera(points, direction, up, margin=0.10, res=None, look=None):
    """50 mm vertical-fit camera with an arbitrary view direction and up."""
    pts = list(points)
    if not pts:
        raise RuntimeError("no points for camera")
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    if look is None:
        look = Vector((sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)))
    direction = direction.normalized()
    cam_z = direction
    cam_x = cam_z.cross(up)
    if cam_x.length < 1e-6:
        cam_x = cam_z.cross(Vector((0.0, 0.0, 1.0)))
    cam_x.normalize()
    cam_y = cam_z.cross(cam_x).normalized()

    sensor_h, lens = 24.0, 50.0
    res_wh = res or RES
    aspect = res_wh[0] / res_wh[1]
    v_ang = 2 * math.atan((sensor_h * 0.5) / lens)
    h_ang = 2 * math.atan(math.tan(v_ang * 0.5) * aspect)
    lim = 1.0 - 2.0 * margin
    tan_h = math.tan(h_ang * 0.5) * lim
    tan_v = math.tan(v_ang * 0.5) * lim

    def cam_xy(p, look_at):
        d = p - look_at
        return d.dot(cam_x), d.dot(cam_y), d.dot(cam_z)

    xys = [cam_xy(p, look) for p in pts]
    mid_x = 0.5 * (min(c[0] for c in xys) + max(c[0] for c in xys))
    mid_y = 0.5 * (min(c[1] for c in xys) + max(c[1] for c in xys))
    look = look + cam_x * mid_x + cam_y * mid_y

    dist = 0.4
    for p in pts:
        x, y, zoff = cam_xy(p, look)
        if tan_h > 1e-8:
            dist = max(dist, abs(x) / tan_h + zoff)
        if tan_v > 1e-8:
            dist = max(dist, abs(y) / tan_v + zoff)
        dist = max(dist, zoff + 0.35)
    dist *= 1.02
    loc = look + direction * dist

    # Drop previous helper cams so we don't leak objects.
    for name in list(bpy.data.objects.keys()):
        if name.startswith("Cam") and bpy.data.objects[name].type == "CAMERA":
            if bpy.context.scene.camera == bpy.data.objects[name]:
                bpy.context.scene.camera = None
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = lens
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = sensor_h
    cam_data.clip_start = 0.05
    cam_data.clip_end = 50
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    rot = Matrix((
        (cam_x.x, cam_y.x, cam_z.x),
        (cam_x.y, cam_y.y, cam_z.y),
        (cam_x.z, cam_y.z, cam_z.z),
    ))
    mw = rot.to_4x4()
    mw.translation = loc
    cam.matrix_world = mw
    bpy.context.scene.camera = cam
    xbot.setup_studio(cam, look)
    return cam, look, direction, dist


def _hide_for_glove_close(handle, meshes, hide=True):
    changed = []
    keep = set(_objects(handle, "XBotGlove_"))
    for o in list(meshes) + [_objects(handle, p) for p in
                             ("XBotBeanie_", "XBotTrunks_", "XBotTank_", "XBotHeadgear_")]:
        if isinstance(o, list):
            group = o
        else:
            group = [o]
        for obj in group:
            if obj is None or obj in keep:
                continue
            if hide:
                if not obj.hide_render:
                    obj.hide_render = True
                    changed.append(obj)
            # show path handled by caller restore
    floor = bpy.data.objects.get("StudioFloor")
    if hide and floor and not floor.hide_render:
        floor.hide_render = True
        changed.append(floor)
    return changed


def _add_glove_card(look, direction, dist, size=0.28):
    mesh = bpy.data.meshes.new("XBotGloveCardMesh")
    obj = bpy.data.objects.new("XBotGloveCard", mesh)
    bpy.context.scene.collection.objects.link(obj)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size)
    bm.to_mesh(mesh)
    bm.free()
    # Sit the card just behind the glove (further from camera than `look`).
    loc = look - direction * 0.06
    # Face the camera: plane +Z toward camera = -direction? grid is XY, +Z normal.
    z = direction  # card normal toward camera? camera is at look+direction*dist, so toward camera is +direction
    # We want the visible face pointing at the camera, i.e. normal = direction
    # (from card to camera is roughly +direction).
    up = Vector((0, 0, 1))
    if abs(z.dot(up)) > 0.9:
        up = Vector((0, 1, 0))
    x = z.cross(up).normalized()
    y = z.cross(x).normalized()
    rot = Matrix((
        (x.x, y.x, z.x),
        (x.y, y.y, z.y),
        (x.z, y.z, z.z),
    ))
    mw = rot.to_4x4()
    mw.translation = loc
    obj.matrix_world = mw
    mat = bpy.data.materials.new("XBotGloveCardMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.42, 0.42, 0.44, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.72
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj


def stage_closeups(state, handle, stills, eevee, samples):
    """Product-style v4 glove close-up (CC0 angle) + garments close-up."""
    gloves = _objects(handle, "XBotGlove_")
    punch = state.get("punch_side") or "R"
    prefer = [o for o in gloves if f"XBotGlove_{punch}_" in o.name]
    glove = (prefer or gloves or [None])[0]
    shots = []

    if glove is not None:
        pts = v1.evaluated_world_points([glove])
        look = Vector(sum((Vector(p) for p in pts), Vector()) / max(1, len(pts)))
        axes = glove.matrix_world.to_3x3()
        dorsal = (axes @ Vector((1.0, 0.0, 0.0))).normalized()
        knuckle = (axes @ Vector((0.0, 1.0, 0.0))).normalized()
        thumb = (axes @ Vector((0.0, 0.0, 1.0))).normalized()
        # Dorsal / slightly above, thumb as a side lobe; cuff toward the top.
        direction = (dorsal * 0.80 + knuckle * 0.10 + thumb * 0.42).normalized()
        up = (-knuckle + 0.04 * dorsal).normalized()
        hidden = _hide_for_glove_close(handle, state["meshes"], hide=True)
        cam, look, direction, dist = setup_oriented_camera(
            pts, direction, up, margin=0.12, res=CLOSE_RES, look=look,
        )
        card = _add_glove_card(look, direction, dist)
        path = os.path.join(OUT_DIR, "xbot_outfit_v4_close_glove.png")
        v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
        stills["close_glove"] = path
        shots.append(path)
        print(f"WROTE {path} ({os.path.getsize(path)} bytes) product-close", flush=True)
        bpy.data.objects.remove(card, do_unlink=True)
        for obj in hidden:
            obj.hide_render = False

    # Garments: torso 3/4 so tank armholes, delts, trunks seat all read.
    gar = _objects(handle, "XBotTank_") + _objects(handle, "XBotTrunks_") + list(state["meshes"])
    if gar:
        pts = v1.evaluated_world_points(gar)
        # Focus on the torso / hips rather than the full figure.
        if pts:
            zs = [p.z for p in pts]
            zlo = min(zs) + 0.22 * (max(zs) - min(zs))
            zhi = min(zs) + 0.78 * (max(zs) - min(zs))
            focus = [p for p in pts if zlo <= p.z <= zhi] or pts
            v1.setup_covering_camera(focus, margin=0.10, yaw=-22.0, elev=8.0, res=CLOSE_RES)
            path = os.path.join(OUT_DIR, "xbot_outfit_v4_close_garments.png")
            v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
            stills["close_garments"] = path
            shots.append(path)
            print(f"WROTE {path} ({os.path.getsize(path)} bytes)", flush=True)
    return shots


def _stage_cc0_crop():
    dest = os.path.join(OUT_DIR, "xbot_outfit_v4_ref_cc0.jpg")
    src = CC0_SRC
    if not os.path.isfile(src):
        # fall back to the v3 copy if the /tmp ref is gone
        alt = os.path.join(OUT_DIR, "xbot_outfit_v3_ref_cc0.jpg")
        if os.path.isfile(alt):
            src = alt
        else:
            return None
    shutil.copy2(src, dest)
    return dest


def main():
    eevee_stills, no_preview, samples = _cli()
    os.makedirs(OUT_DIR, exist_ok=True)
    stills = {}
    facts = {
        "ok": False,
        "eevee_stills": eevee_stills,
        "samples": 16 if eevee_stills else (samples or xbot.SAMPLES),
        "margin": MARGIN,
        "colorway": "A",
        "colorway_name": COLORWAYS["A"]["name"],
        "source": "procedural (authored in Blender, no third-party mesh)",
        "license": "original geometry; commercially free",
        "ref_photo": {
            "file": "Pair of boxing gloves.jpg",
            "license": "CC0 1.0 (Wikimedia Commons, Petey21, 2014)",
            "url": "https://commons.wikimedia.org/wiki/File:Pair_of_boxing_gloves.jpg",
            "note": "Cropped and scaled to the v4 glove close-up long axis; logo/text not copied into the mesh.",
        },
        "clips": [],
        "handles": {},
        "poke": {},
    }
    t0 = time.time()
    honest = []
    cc0 = _stage_cc0_crop()
    if cc0:
        stills["close_ref"] = cc0
    else:
        honest.append("CC0 reference photo was not on disk; sheet will describe proportions instead.")

    # Reuse the Lead Jab clip across full + return so we don't reload FBX twice.
    clip_state = {}

    for pose in POSES:
        path = os.path.join(SRC_DIR, pose["file"])
        cache_key = pose["file"]
        if cache_key not in clip_state:
            clip_state[cache_key] = v1.prepare_clip(path, "idle" if pose["clip"] == "idle" else "punch")
            # Drop previous clip's armature when we move to a new file; prepare_clip
            # already replaces the scene in existing scripts, so one-at-a-time is fine.
            # If prepare_clip always loads fresh, forget the other cache entry.
            if len(clip_state) > 1:
                # Keep only the current file — previous FBX is gone.
                clip_state = {cache_key: clip_state[cache_key]}
        state = clip_state[cache_key]
        state["clip_kind"] = pose["clip"]
        frame = set_pose_ex(state, pose["stage"])
        rec = {
            "clip": pose["clip"], "stage": pose["stage"], "frame": frame,
            "lock": state["lock"], "drift": state["drift"],
            "punch_side": state["punch_side"], "keys": state["keys"],
            "return_frame": _return_frame(state) if pose["clip"] == "jab" else None,
        }

        probe = apply_outfit(state["arm"], state["meshes"],
                             items=V4_ITEMS, colorway="A",
                             scan_frames=True, version="v4")
        objs = list(state["meshes"]) + [bpy.data.objects[n] for n in probe["objects"]
                                        if n in bpy.data.objects]
        pts = v1.evaluated_world_points(objs)
        v1.setup_covering_camera(pts, margin=MARGIN)
        rec["containment_probe"] = probe.get("containment")
        rec["notes"] = list(probe.get("notes") or [])
        rec["head_width"] = probe.get("head_width")
        rec["poke_probe"] = {
            "trunks": _poke(probe, state["meshes"], "trunks"),
            "tank": _poke(probe, state["meshes"], "tank"),
        }
        facts["poke"][f"{pose['clip']}_{pose['stage']}"] = rec["poke_probe"]
        for kind, report in rec["poke_probe"].items():
            if report and report.get("pokes"):
                honest.append(
                    f"{pose['clip']} {pose['stage']} {kind}: "
                    f"{report['pokes']}/{report['tested']} region verts poke through"
                )
        clear_outfit(probe, state["meshes"])

        if pose["clip"] == "jab" and pose["stage"] == "full":
            totals, per, _info = xbot.paint_glow(
                state["meshes"], state["arm"], state["bones"],
                state["punch_side"], ("front_delt", "triceps"))
            rec["glow_totals"] = totals
            print(f"GLOW totals={totals} punch_side={state['punch_side']}", flush=True)
        elif pose["stage"] != "return":
            # keep glow through the return row of the same jab clip so the
            # tank-vs-delt relationship stays visible; clear on idle.
            xbot.clear_glow(state["meshes"])

        kept_v4 = None
        for var in VARIANTS:
            handle = None
            if var["items"]:
                handle = apply_outfit(state["arm"], state["meshes"],
                                      items=var["items"], colorway=var["colorway"],
                                      scan_frames=True, version=var["version"])
                facts["handles"][f"{pose['clip']}_{pose['stage']}_{var['key']}"] = v1._handle_public(handle)
                if handle.get("containment"):
                    for side, c in handle["containment"].items():
                        if c.get("outside_padded_aabb"):
                            honest.append(
                                f"{pose['clip']} {pose['stage']} {var['key']} glove {side}: "
                                f"{c['outside_padded_aabb']} hand verts outside padded envelope"
                            )
            out = os.path.join(
                OUT_DIR,
                f"xbot_outfit_v4_{pose['clip']}_{pose['stage']}_{var['key']}.png",
            )
            v1.render_still(out, eevee=eevee_stills, samples=samples)
            stills[f"{pose['clip']}_{pose['stage']}_{var['key']}"] = out
            print(f"WROTE {out} ({os.path.getsize(out)} bytes)", flush=True)
            if var["key"] == "v4A" and pose["clip"] == "jab" and pose["stage"] == "full":
                kept_v4 = handle
            elif handle:
                clear_outfit(handle, state["meshes"])

        if kept_v4 is not None:
            stage_closeups(state, kept_v4, stills, eevee_stills, samples)
            ginfo = (kept_v4.get("glove_info") or {}).get(state["punch_side"] or "R") or {}
            rec["v4_glove"] = {k: ginfo.get(k) for k in
                               ("body_len", "front_width", "front_thick",
                                "cuff_y_near", "cuff_y_far", "mask_plane_y",
                                "head_width", "fist_to_head")}
            rec["v4_beanie"] = kept_v4.get("beanie_info")
            rec["v4_trunks"] = kept_v4.get("trunks_info")
            rec["v4_tank"] = kept_v4.get("tank_info")
            clear_outfit(kept_v4, state["meshes"])

        facts["clips"].append(rec)

    facts["stills"] = stills
    facts["stills_elapsed_s"] = time.time() - t0

    honest.append("Assets: all outfit meshes are procedural (no third-party / paid / logo files).")
    honest.append("v4 gloves: glossy black leather, bulbous body, side-lobe thumb, smooth same-black cuff, subtle stitch grooves; fist ~0.8× head width.")
    honest.append("The CC0 photo's PRO strap lettering is not reproduced (trademark).")
    honest.append("Trunks and tank are charcoal satin (beanie base color) with thin black trim on the trunks, armature-bound.")
    honest.append("Tank is sleeveless / racerback so deltoids, arms and scapulae stay visible for the jab glow.")
    facts["honest_notes"] = honest

    sheet = os.path.join(OUT_DIR, "xbot_outfit_v4.png")
    facts_path = os.path.join(OUT_DIR, "xbot_outfit_v4_facts.json")
    with open(facts_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    import subprocess
    subprocess.check_call([
        "/usr/bin/python3",
        os.path.join(HERE, "build_xbot_outfit_sheet_v4.py"),
        "--facts", facts_path,
    ])
    facts["sheet"] = sheet

    if not no_preview:
        facts["jab_preview"] = v1.render_jab_preview(
            "xbot_outfit_v4_jab_preview.mp4",
            colorway="A", items=V4_ITEMS, version="v4",
        )
    else:
        facts["jab_preview"] = {"skipped": True, "reason": "preview not requested"}

    facts["elapsed_s"] = time.time() - t0
    facts["ok"] = os.path.isfile(sheet) and all(
        os.path.isfile(p) for k, p in stills.items() if k != "close_ref" or p
    )
    with open(facts_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("FACTS", facts_path, "ok", facts["ok"], "elapsed", round(facts["elapsed_s"], 1), flush=True)
    if not facts["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
