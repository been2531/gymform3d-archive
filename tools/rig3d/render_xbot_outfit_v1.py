#!/usr/bin/env python3
"""Render the X Bot outfit v1 comparison sheet and a short jab preview.

  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v1.py
  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v1.py -- --eevee-stills

Does not modify eval_xbot_boxing.py. Imports its clay / studio / clip helpers
and applies tools/rig3d/xbot_outfit.py on top.

Writes (gitignored):
  output/_basemodels/eval/xbot_outfit_v1.png
  output/_basemodels/eval/xbot_outfit_a_jab_preview.mp4
  output/_basemodels/eval/xbot_outfit_v1_facts.json
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import eval_xbot_boxing as xbot  # noqa: E402
from xbot_outfit import (  # noqa: E402
    COLORWAYS, apply_outfit, clear_outfit,
)

ROOT = xbot.ROOT
SRC_DIR = xbot.SRC_DIR
OUT_DIR = xbot.OUT_DIR
RES = xbot.RES
MARGIN = 0.06  # ≥5% frame margin; keep the extended fist uncropped

VARIANTS = (
    {"key": "plain", "title": "plain X Bot", "items": (), "colorway": "A"},
    {"key": "A", "title": COLORWAYS["A"]["label"], "items": ("gloves", "beanie"), "colorway": "A"},
    {"key": "B", "title": COLORWAYS["B"]["label"], "items": ("gloves", "headgear"), "colorway": "B"},
)

POSES = (
    {"clip": "idle", "file": "Fighting Idle.fbx", "stage": "guard", "title": "Fighting Idle — guard"},
    {"clip": "jab", "file": "Lead Jab.fbx", "stage": "full", "title": "Lead Jab — full extension"},
)


def _cli():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    eevee_stills = "--eevee-stills" in argv
    no_preview = "--no-preview" in argv
    samples = None
    if "--samples" in argv:
        samples = int(argv[argv.index("--samples") + 1])
    return eevee_stills, no_preview, samples


def prepare_clip(path, pick):
    meshes, arm = xbot.import_clip(path)
    bones = xbot.bone_map(arm)
    f0, f1 = xbot.frame_range(arm)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    xbot.scale_ground_center(arm, meshes, bones)
    xbot.face_minus_y(arm, bones)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    start_xy = xbot.hips_xy(arm, bones)
    rows = xbot.analyze_motion(arm, bones, f0, f1)
    xs = [r["hips"][0] for r in rows if "hips" in r]
    ys = [r["hips"][1] for r in rows if "hips" in r]
    drift = 0.0
    if xs:
        drift = float(math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    lock = drift > xbot.ROOT_DRIFT_LOCK_M
    punch_side, span = xbot.pick_punch_hand(rows)
    if pick == "idle":
        keys = xbot.pick_idle_frame(rows)
    else:
        keys = xbot.pick_keyframes(rows, punch_side)
    xbot.assign_clay(meshes)
    return {
        "meshes": meshes,
        "arm": arm,
        "bones": bones,
        "f0": f0,
        "f1": f1,
        "keys": keys,
        "punch_side": punch_side,
        "span": span,
        "start_xy": start_xy,
        "lock": lock,
        "drift": drift,
    }


def set_pose(state, stage):
    f = int(state["keys"][stage])
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    if state["lock"]:
        xbot.lock_root_xy(state["arm"], state["bones"], state["start_xy"])
        bpy.context.view_layer.update()
    return f


def evaluated_world_points(objects):
    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in objects:
        if o.type != "MESH" or o.name == "StudioFloor":
            continue
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        try:
            mw = ev.matrix_world
            # subsample dense meshes
            step = 1 if len(me.vertices) < 4000 else max(1, len(me.vertices) // 2500)
            for i, v in enumerate(me.vertices):
                if i % step:
                    continue
                pts.append(mw @ v.co)
        finally:
            ev.to_mesh_clear()
    return pts


def setup_covering_camera(points, margin=MARGIN, yaw=None, elev=None, res=None):
    """50 mm vertical-fit 9:16 camera that keeps every point inside `margin`.

    Closed-form: x/y in camera space do not depend on distance (the camera
    only travels along its view axis), so we can solve the minimum distance
    that fits the projected AABB without an iterative look-shift — the
    previous iterator could run away when a close fist inflated NDC.
    """
    for o in list(bpy.data.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)
    yaw = xbot.YAW if yaw is None else yaw
    elev = xbot.ELEV if elev is None else elev
    pts = [Vector(p) for p in points]
    if not pts:
        raise RuntimeError("no points to frame")
    xs, ys, zs = zip(*[(p.x, p.y, p.z) for p in pts])
    look = Vector(((min(xs) + max(xs)) * 0.5,
                   (min(ys) + max(ys)) * 0.5,
                   (min(zs) + max(zs)) * 0.5))
    direction = Vector((
        math.sin(math.radians(yaw)) * math.cos(math.radians(elev)),
        -math.cos(math.radians(yaw)) * math.cos(math.radians(elev)),
        math.sin(math.radians(elev)),
    )).normalized()
    # camera axes: look along -Z, Y up
    cam_z = direction                          # away from subject
    cam_x = cam_z.cross(Vector((0, 0, 1)))
    if cam_x.length < 1e-6:
        cam_x = Vector((1, 0, 0))
    cam_x.normalize()
    cam_y = cam_z.cross(cam_x).normalized()    # true camera +Y

    sensor_h = 24.0
    lens = 50.0
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

    # one stable centering shift: put the mid of the camera-XY AABB on the
    # optical axis (does not depend on distance)
    xys = [cam_xy(p, look) for p in pts]
    mid_x = 0.5 * (min(c[0] for c in xys) + max(c[0] for c in xys))
    mid_y = 0.5 * (min(c[1] for c in xys) + max(c[1] for c in xys))
    look = look + cam_x * mid_x + cam_y * mid_y

    dist = 0.4
    for p in pts:
        x, y, zoff = cam_xy(p, look)
        # z_cam = zoff - dist  (must stay < 0, in front of camera)
        # |x| <= -z_cam * tan_h  =>  dist >= |x|/tan_h + zoff
        if tan_h > 1e-8:
            dist = max(dist, abs(x) / tan_h + zoff)
        if tan_v > 1e-8:
            dist = max(dist, abs(y) / tan_v + zoff)
        dist = max(dist, zoff + 0.35)
    dist *= 1.02  # tiny slack so the solved bound is not exactly on the pixel edge

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = lens
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = sensor_h
    cam_data.clip_start = 0.05
    cam_data.clip_end = 50
    loc = look + direction * dist
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    xbot.setup_studio(cam, look)
    print(f"camera look={tuple(round(c,3) for c in look)} dist={dist:.3f} "
          f"margin={margin:.2f} bbox=({min(xs):.2f},{min(ys):.2f},{min(zs):.2f})-"
          f"({max(xs):.2f},{max(ys):.2f},{max(zs):.2f})", flush=True)
    return cam

def render_still(path, eevee=False, samples=None, res=None):
    sc = bpy.context.scene
    if eevee:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
        if hasattr(sc, "eevee"):
            try:
                sc.eevee.taa_render_samples = 16
            except Exception:
                pass
    else:
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = int(samples or xbot.SAMPLES)
        sc.cycles.use_denoising = True
        try:
            sc.cycles.denoiser = "OPENIMAGEDENOISE"
        except Exception:
            pass
        sc.cycles.filter_width = 1.5
    sc.render.resolution_x, sc.render.resolution_y = res or RES
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.filepath = path
    xbot.apply_color_mgmt(sc)
    bpy.ops.render.render(write_still=True)
    return path



def compose_sheet(stills, facts, out_path):
    """Delegate to system Python — Blender's interpreter has no PIL."""
    facts_path = os.path.join(OUT_DIR, "xbot_outfit_v1_facts.json")
    # facts may not have been written yet; write a stub the builder can read
    payload = dict(facts)
    payload["stills"] = stills
    with open(facts_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    cmd = [
        "/usr/bin/python3",
        os.path.join(HERE, "build_xbot_outfit_sheet.py"),
        "--facts", facts_path,
    ]
    subprocess.check_call(cmd)
    if not os.path.isfile(out_path):
        # builder always writes the canonical path
        canonical = os.path.join(OUT_DIR, "xbot_outfit_v1.png")
        if canonical != out_path and os.path.isfile(canonical):
            return canonical
    print("WROTE", out_path, os.path.getsize(out_path) if os.path.isfile(out_path) else 0, flush=True)
    return out_path



def render_jab_preview(preview_name, colorway="A", items=("gloves", "beanie"), version="v1"):
    t0 = time.time()
    path = os.path.join(SRC_DIR, "Lead Jab.fbx")
    state = prepare_clip(path, "punch")
    f = set_pose(state, "guard")
    handle = apply_outfit(state["arm"], state["meshes"], items=items,
                          colorway=colorway, scan_frames=True, version=version)
    # cover the whole clip, including gloves, so the fist stays in frame
    frames_span = list(range(state["f0"], state["f1"] + 1, max(1, (state["f1"] - state["f0"]) // 6)))
    cover_pts = []
    saved = bpy.context.scene.frame_current
    for fr in frames_span:
        bpy.context.scene.frame_set(fr)
        bpy.context.view_layer.update()
        if state["lock"]:
            xbot.lock_root_xy(state["arm"], state["bones"], state["start_xy"])
        objs = list(state["meshes"]) + [bpy.data.objects[n] for n in handle["objects"] if n in bpy.data.objects]
        cover_pts.extend(evaluated_world_points(objs))
    bpy.context.scene.frame_set(saved)
    bpy.context.view_layer.update()
    setup_covering_camera(cover_pts, margin=MARGIN)

    n = state["f1"] - state["f0"] + 1
    fps = 15
    if n <= 40:
        frames = list(range(state["f0"], state["f1"] + 1))
        fps = max(12, min(30, int(round(n / 2.0)))) or 15
    else:
        a, b = state["keys"]["guard"], min(state["f1"], state["keys"]["full"] + 3)
        raw = list(range(a, b + 1))
        step = max(1, int(math.ceil(len(raw) / 30)))
        frames = raw[::step]
    if len(frames) > 50:
        return {"skipped": True, "reason": f"too many frames ({len(frames)})"}

    seq_dir = os.path.join(OUT_DIR, f"_xbot_outfit_{version}_jab_seq")
    os.makedirs(seq_dir, exist_ok=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.render.resolution_x, sc.render.resolution_y = 360, 640
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    xbot.apply_color_mgmt(sc)
    if hasattr(sc, "eevee"):
        try:
            sc.eevee.taa_render_samples = 8
        except Exception:
            pass

    written = []
    for i, fr in enumerate(frames):
        if time.time() - t0 > 150 and i > 4:
            return {"skipped": True, "reason": "pacing would exceed 3 min",
                    "elapsed_s": time.time() - t0, "done": i}
        bpy.context.scene.frame_set(fr)
        bpy.context.view_layer.update()
        if state["lock"]:
            xbot.lock_root_xy(state["arm"], state["bones"], state["start_xy"])
        xbot.paint_glow(state["meshes"], state["arm"], state["bones"],
                        state["punch_side"], ("front_delt", "triceps"))
        fp = os.path.join(seq_dir, f"f_{i:04d}.png")
        sc.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        written.append(fp)
    if not written:
        return {"skipped": True, "reason": "no frames"}
    mp4 = preview_name if os.path.isabs(preview_name) else os.path.join(OUT_DIR, preview_name)
    pattern = os.path.join(seq_dir, "f_%04d.png")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-i", pattern,
        "-vf", "loop=loop=1:size=" + str(len(written)) + ":start=0",
        "-frames:v", str(max(len(written), int(2 * fps))),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4,
    ]
    try:
        subprocess.check_call(cmd)
    except Exception:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps), "-i", pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4,
        ]
        subprocess.check_call(cmd)
    return {
        "skipped": False,
        "path": mp4,
        "bytes": os.path.getsize(mp4) if os.path.isfile(mp4) else 0,
        "n_src": len(written),
        "fps": fps,
        "elapsed_s": time.time() - t0,
        "outfit": handle,
    }


def _handle_public(handle):
    if not handle:
        return None
    keep = ("colorway", "colorway_name", "items", "objects", "notes",
            "source", "license", "containment", "glove_info", "beanie_info",
            "headgear_info", "hand_envelopes")
    out = {k: handle.get(k) for k in keep if k in handle}
    # glove_info env may contain tuples
    return out


def main():
    eevee_stills, no_preview, samples = _cli()
    os.makedirs(OUT_DIR, exist_ok=True)
    stills = {}
    facts = {
        "ok": False,
        "eevee_stills": eevee_stills,
        "samples": 16 if eevee_stills else (samples or xbot.SAMPLES),
        "margin": MARGIN,
        "colorways": {k: {kk: vv for kk, vv in v.items() if kk != "glove"} | {"glove": list(v["glove"])}
                      for k, v in COLORWAYS.items()},
        "source": "procedural (authored in Blender, no third-party mesh)",
        "license": "original geometry; commercially free",
        "clips": [],
        "handles": {},
    }
    t0 = time.time()
    honest = []

    for pose in POSES:
        path = os.path.join(SRC_DIR, pose["file"])
        state = prepare_clip(path, "idle" if pose["clip"] == "idle" else "punch")
        frame = set_pose(state, pose["stage"])
        rec = {
            "clip": pose["clip"], "stage": pose["stage"], "frame": frame,
            "lock": state["lock"], "drift": state["drift"],
            "punch_side": state["punch_side"], "keys": state["keys"],
        }
        # Apply the largest outfit (gloves + headgear, similar bounds to beanie)
        # once, to compute a shared camera, then clear and render each variant.
        probe = apply_outfit(state["arm"], state["meshes"],
                             items=("gloves", "headgear"), colorway="B",
                             scan_frames=True, version="v1")
        objs = list(state["meshes"]) + [bpy.data.objects[n] for n in probe["objects"]
                                        if n in bpy.data.objects]
        pts = evaluated_world_points(objs)
        setup_covering_camera(pts, margin=MARGIN)
        rec["containment_probe"] = probe.get("containment")
        rec["notes"] = list(probe.get("notes") or [])
        clear_outfit(probe, state["meshes"])

        if pose["clip"] == "jab" and pose["stage"] == "full":
            xbot.paint_glow(state["meshes"], state["arm"], state["bones"],
                            state["punch_side"], ("front_delt", "triceps"))
        else:
            xbot.clear_glow(state["meshes"])

        for var in VARIANTS:
            handle = None
            if var["items"]:
                handle = apply_outfit(state["arm"], state["meshes"],
                                      items=var["items"], colorway=var["colorway"],
                                      scan_frames=True, version="v1")
                facts["handles"][f"{pose['clip']}_{var['key']}"] = _handle_public(handle)
                if handle.get("containment"):
                    for side, c in handle["containment"].items():
                        if c.get("outside_padded_aabb"):
                            honest.append(
                                f"{pose['clip']} outfit {var['key']} glove {side}: "
                                f"{c['outside_padded_aabb']} hand verts outside padded envelope"
                            )
            out = os.path.join(OUT_DIR, f"xbot_outfit_{pose['clip']}_{pose['stage']}_{var['key']}.png")
            render_still(out, eevee=eevee_stills, samples=samples)
            stills[f"{pose['clip']}_{pose['stage']}_{var['key']}"] = out
            print(f"WROTE {out} ({os.path.getsize(out)} bytes)", flush=True)
            if handle:
                clear_outfit(handle, state["meshes"])
        facts["clips"].append(rec)

    facts["stills"] = stills
    facts["stills_elapsed_s"] = time.time() - t0

    honest.append("Assets: all procedural (no third-party / paid / logo files).")
    honest.append("Gloves are rigid mittens (no finger-curl squash). A slight seam remains where the forearm meets the cuff.")
    honest.append("Outfit B is open-face amateur headgear; cheek pads + chin strap hide part of the jaw, so chin-tuck is slightly harder to read than with a beanie or a bare head.")
    facts["honest_notes"] = honest

    sheet = os.path.join(OUT_DIR, "xbot_outfit_v1.png")
    compose_sheet(stills, facts, sheet)
    facts["sheet"] = sheet

    if not no_preview and not eevee_stills:
        remaining = 240.0 - (time.time() - t0)
        if remaining > 40:
            facts["jab_preview"] = render_jab_preview("xbot_outfit_a_jab_preview.mp4")
        else:
            facts["jab_preview"] = {
                "skipped": True,
                "reason": f"stills used {facts['stills_elapsed_s']:.0f}s; no budget left",
            }
    elif eevee_stills and not no_preview:
        # still do a short preview after fast stills
        facts["jab_preview"] = render_jab_preview("xbot_outfit_a_jab_preview.mp4")
    else:
        facts["jab_preview"] = {"skipped": True, "reason": "preview not requested"}

    facts["elapsed_s"] = time.time() - t0
    facts["ok"] = os.path.isfile(sheet) and all(os.path.isfile(p) for p in stills.values())
    out_json = os.path.join(OUT_DIR, "xbot_outfit_v1_facts.json")
    with open(out_json, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("FACTS", out_json, "ok", facts["ok"], "elapsed", round(facts["elapsed_s"], 1), flush=True)
    if not facts["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
