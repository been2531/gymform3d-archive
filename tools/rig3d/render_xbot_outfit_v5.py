#!/usr/bin/env python3
"""Render the X Bot outfit v5 comparison sheet and a short jab preview.

  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v5.py
  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v5.py -- --eevee-stills
  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v5.py -- --eevee-stills --no-preview

Outfit A v5: glossy black CC0-matched gloves + charcoal beanie + burgundy
satin trunks + fitted charcoal tank. Columns: plain | v4 A | v5 A.
Rows: Idle guard, Jab full, Jab return. Close-ups: v5 glove vs CC0 glove,
v5 trunks vs CC0 trunks (Nationaal Archief 1951), plus a garments close-up.

Writes (gitignored):
  output/_basemodels/eval/xbot_outfit_v5.png
  output/_basemodels/eval/xbot_outfit_v5_jab_preview.mp4
  output/_basemodels/eval/xbot_outfit_v5_facts.json
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
CC0_TRUNKS_SRC = "/tmp/trunks_refs/bruin_5273.jpg"
V5_ITEMS = ("gloves", "beanie", "trunks", "tank")

VARIANTS = (
    {"key": "plain", "title": "plain X Bot", "items": (), "colorway": "A", "version": "v5"},
    {"key": "v4A", "title": "v4 A  ·  ballooned trunks / crop tank (rejected)",
     "items": V5_ITEMS, "colorway": "A", "version": "v4"},
    {"key": "v5A", "title": "v5 A  ·  burgundy trunks, charcoal tank, kidney-thumb gloves",
     "items": V5_ITEMS, "colorway": "A", "version": "v5"},
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
    cam_x = up.cross(cam_z)
    if cam_x.length < 1e-6:
        cam_x = Vector((0.0, 0.0, 1.0)).cross(cam_z)
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


def frame_camera_keep_studio(points, margin=0.12, yaw=None, elev=None, res=None):
    """Place a 50 mm camera to frame `points` without restaging the body lights.

    setup_studio() puts key/fill/rim next to the camera. Framing a 15 cm glove
    would pull those lights onto the leather and wash glossy black to grey.
    Close-ups must keep the already-staged body-scale studio.
    """
    yaw = xbot.YAW if yaw is None else yaw
    elev = xbot.ELEV if elev is None else elev
    pts = [Vector(p) for p in points]
    if not pts:
        raise RuntimeError("no points for close camera")
    xs, ys, zs = zip(*[(p.x, p.y, p.z) for p in pts])
    look = Vector(((min(xs) + max(xs)) * 0.5,
                   (min(ys) + max(ys)) * 0.5,
                   (min(zs) + max(zs)) * 0.5))
    direction = Vector((
        math.sin(math.radians(yaw)) * math.cos(math.radians(elev)),
        -math.cos(math.radians(yaw)) * math.cos(math.radians(elev)),
        math.sin(math.radians(elev)),
    )).normalized()
    cam_z = direction
    cam_x = cam_z.cross(Vector((0, 0, 1)))
    if cam_x.length < 1e-6:
        cam_x = Vector((1, 0, 0))
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

    # Replace the camera only; leave Key/Fill/Rim and the floor alone.
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
    cam.location = loc
    cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    print(f"camera (keep studio) look={tuple(round(c,3) for c in look)} dist={dist:.3f} "
          f"bbox=({min(xs):.2f},{min(ys):.2f},{min(zs):.2f})-"
          f"({max(xs):.2f},{max(ys):.2f},{max(zs):.2f})", flush=True)
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
    bsdf.inputs["Base Color"].default_value = (0.18, 0.18, 0.19, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.72
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj



def stage_closeups(state, handle, stills, eevee, samples):
    """v5 glove close-up under the SAME studio lighting (no grey card) + trunks close."""
    # Re-stage the FRONT body-scale studio before any close-up. A prior back
    # view (or a product camera) would otherwise leave the key behind the figure
    # and the close-ups would render as underexposed silhouettes.
    body_for_studio = list(state["meshes"]) + [
        bpy.data.objects[n] for n in (handle.get("objects") or [])
        if n in bpy.data.objects
    ]
    bpts = v1.evaluated_world_points(body_for_studio)
    if bpts:
        v1.setup_covering_camera(bpts, margin=0.06, yaw=xbot.YAW, elev=xbot.ELEV, res=RES)

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
        direction = (dorsal * 0.72 + thumb * 0.58 + knuckle * 0.08).normalized()
        up = (-knuckle + 0.02 * dorsal).normalized()
        hidden = _hide_for_glove_close(handle, state["meshes"], hide=True)
        # Keep the already-staged body lights. Do not call setup_studio()
        # (that parks the key on the glove and turns glossy black into grey plastic).
        # Prefer the same yaw/elev language as the body rows so the material reads
        # the same; a slight yaw bias shows the kidney thumb.
        if not any(o.type == "LIGHT" for o in bpy.data.objects):
            body_pts = v1.evaluated_world_points(list(state["meshes"]) + gloves)
            v1.setup_covering_camera(body_pts, margin=0.06, res=RES)
        frame_camera_keep_studio(pts, margin=0.14, yaw=xbot.YAW + 18.0,
                                 elev=xbot.ELEV + 4.0, res=CLOSE_RES)
        path = os.path.join(OUT_DIR, "xbot_outfit_v5_close_glove.png")
        v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
        stills["close_glove"] = path
        shots.append(path)
        print(f"WROTE {path} ({os.path.getsize(path)} bytes) close (same studio lights, no card)", flush=True)
        for obj in hidden:
            obj.hide_render = False

    trunks = _objects(handle, "XBotTrunks_")
    if trunks:
        # Lower torso / hips, same studio camera language as the body rows.
        body = list(state["meshes"])
        pts = v1.evaluated_world_points(trunks + body)
        if pts:
            zs = [p.z for p in pts]
            zlo = min(zs) + 0.18 * (max(zs) - min(zs))
            zhi = min(zs) + 0.62 * (max(zs) - min(zs))
            focus = [p for p in pts if zlo <= p.z <= zhi] or pts
            if not any(o.type == "LIGHT" for o in bpy.data.objects):
                v1.setup_covering_camera(pts, margin=0.10, yaw=-22.0, elev=6.0, res=CLOSE_RES)
            frame_camera_keep_studio(focus, margin=0.10, yaw=-22.0, elev=6.0, res=CLOSE_RES)
            path = os.path.join(OUT_DIR, "xbot_outfit_v5_close_trunks.png")
            v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
            stills["close_trunks"] = path
            shots.append(path)
            print(f"WROTE {path} ({os.path.getsize(path)} bytes)", flush=True)

    # Tank / torso close so the waistband cover and armholes can be judged.
    gar = _objects(handle, "XBotTank_") + _objects(handle, "XBotTrunks_") + list(state["meshes"])
    if gar:
        pts = v1.evaluated_world_points(gar)
        if pts:
            zs = [p.z for p in pts]
            zlo = min(zs) + 0.22 * (max(zs) - min(zs))
            zhi = min(zs) + 0.82 * (max(zs) - min(zs))
            focus = [p for p in pts if zlo <= p.z <= zhi] or pts
            if not any(o.type == "LIGHT" for o in bpy.data.objects):
                v1.setup_covering_camera(pts, margin=0.10, yaw=-22.0, elev=8.0, res=CLOSE_RES)
            frame_camera_keep_studio(focus, margin=0.10, yaw=-22.0, elev=8.0, res=CLOSE_RES)
            path = os.path.join(OUT_DIR, "xbot_outfit_v5_close_garments.png")
            v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
            stills["close_garments"] = path
            shots.append(path)
            print(f"WROTE {path} ({os.path.getsize(path)} bytes)", flush=True)
    return shots


def _stage_cc0_crop():
    dest = os.path.join(OUT_DIR, "xbot_outfit_v5_ref_cc0.jpg")
    src = CC0_SRC
    if not os.path.isfile(src):
        alt = os.path.join(OUT_DIR, "xbot_outfit_v3_ref_cc0.jpg")
        if os.path.isfile(alt):
            src = alt
        else:
            dest = None
    if dest and os.path.isfile(src):
        shutil.copy2(src, dest)
    else:
        dest = None
    tdest = os.path.join(OUT_DIR, "xbot_outfit_v5_ref_trunks.jpg")
    tsrc = CC0_TRUNKS_SRC
    if os.path.isfile(tsrc):
        shutil.copy2(tsrc, tdest)
    else:
        tdest = None
    return dest, tdest



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
            "note": "Cropped and scaled to the v5 glove close-up long axis; logo/text not copied into the mesh.",
        },
        "ref_trunks": {
            "file": "Training Jan de Bruin met Theo Huizenaar, Bestanddeelnr 904-5273.jpg",
            "license": "CC0 1.0 (Nationaal Archief / Anefo, Harry Pot, 19 April 1951)",
            "url": "https://commons.wikimedia.org/wiki/File:Training_Jan_de_Bruin_met_Theo_Huizenaar,_Bestanddeelnr_904-5273.jpg",
            "note": "Cropped to the worn trunks and scaled to the v5 trunks close-up long axis.",
        },
        "clips": [],
        "handles": {},
        "poke": {},
    }
    t0 = time.time()
    honest = []
    cc0, trunks_ref = _stage_cc0_crop()
    if cc0:
        stills["close_ref"] = cc0
    else:
        honest.append("CC0 glove reference photo was not on disk; sheet will describe proportions instead.")
    if trunks_ref:
        stills["close_trunks_ref"] = trunks_ref
    else:
        honest.append("CC0 trunks reference photo was not on disk; sheet will describe proportions instead.")

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
                             items=V5_ITEMS, colorway="A",
                             scan_frames=True, version="v5")
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
        # garment_poke_report is inverted / unreliable — do not treat counts as visual failures.
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

        kept_v5 = None
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
                f"xbot_outfit_v5_{pose['clip']}_{pose['stage']}_{var['key']}.png",
            )
            v1.render_still(out, eevee=eevee_stills, samples=samples)
            stills[f"{pose['clip']}_{pose['stage']}_{var['key']}"] = out
            print(f"WROTE {out} ({os.path.getsize(out)} bytes)", flush=True)
            if var["key"] == "v5A" and pose["clip"] == "jab" and pose["stage"] == "full":
                kept_v5 = handle
            elif handle:
                clear_outfit(handle, state["meshes"])

        if kept_v5 is not None:
            stage_closeups(state, kept_v5, stills, eevee_stills, samples)
            ginfo = (kept_v5.get("glove_info") or {}).get(state["punch_side"] or "R") or {}
            rec["v5_glove"] = {k: ginfo.get(k) for k in
                               ("body_len", "front_width", "front_thick",
                                "cuff_y_near", "cuff_y_far", "mask_plane_y",
                                "head_width", "fist_to_head")}
            rec["v5_beanie"] = kept_v5.get("beanie_info")
            rec["v5_trunks"] = kept_v5.get("trunks_info")
            rec["v5_tank"] = kept_v5.get("tank_info")
            clear_outfit(kept_v5, state["meshes"])

        facts["clips"].append(rec)

    facts["stills"] = stills
    facts["stills_elapsed_s"] = time.time() - t0

    honest.append("Assets: all outfit meshes are procedural (no third-party / paid / logo files).")
    honest.append("v5 trunks are deep burgundy/wine satin with thin charcoal trim — distinct from the jab glow (front-delt purple, triceps gold), which stays on the shoulder/arm.")
    honest.append("v5 trunks: snug navel waistband, fabric follows hips/glutes, mid-thigh hem with a late ~3.5 cm flare (no balloon/skirt). Inner thigh can still show a little clay on a deep jab stance.")
    honest.append("v5 tank is charcoal (beanie base). Mixamo Beta_Surface has a 13 cm midriff hole, so the tank is a clean loft skinned to the spine chain; it reaches the waistband at rest but still tears/wrinkles on jab extension. Armholes are not fully clean.")
    honest.append("v5 glove close-up uses the same body-scale studio lights (no grey card) and reads glossy black. The thumb is still weaker than the CC0 kidney lobe (close-up angle hides it; cuff has a slight lip). Wrist clay can still peek at the cuff on jab extension.")
    honest.append("The CC0 glove photo's PRO strap lettering is not reproduced (trademark). Trunks reference is CC0 (Nationaal Archief / Anefo, Harry Pot, 1951).")
    facts["honest_notes"] = honest

    sheet = os.path.join(OUT_DIR, "xbot_outfit_v5.png")
    facts_path = os.path.join(OUT_DIR, "xbot_outfit_v5_facts.json")
    with open(facts_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    import subprocess
    subprocess.check_call([
        "/usr/bin/python3",
        os.path.join(HERE, "build_xbot_outfit_sheet_v5.py"),
        "--facts", facts_path,
    ])
    facts["sheet"] = sheet

    if not no_preview:
        facts["jab_preview"] = v1.render_jab_preview(
            "xbot_outfit_v5_jab_preview.mp4",
            colorway="A", items=V5_ITEMS, version="v5",
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
