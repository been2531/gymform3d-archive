#!/usr/bin/env python3
"""Render the X Bot outfit v3 comparison sheet and a short jab preview.

  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v3.py
  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v3.py -- --eevee-stills

Outfit A only (crimson gloves + charcoal beanie). Headgear is dropped.
Columns: plain | v2 A | v3 A. Rows: Idle guard, Jab full, close-ups + CC0 ref.

Writes (gitignored):
  output/_basemodels/eval/xbot_outfit_v3.png
  output/_basemodels/eval/xbot_outfit_v3_jab_preview.mp4
  output/_basemodels/eval/xbot_outfit_v3_facts.json
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import eval_xbot_boxing as xbot  # noqa: E402
import render_xbot_outfit_v1 as v1  # noqa: E402
from xbot_outfit import (  # noqa: E402
    COLORWAYS, apply_outfit, clear_outfit,
)

ROOT = xbot.ROOT
SRC_DIR = xbot.SRC_DIR
OUT_DIR = xbot.OUT_DIR
RES = xbot.RES
CLOSE_RES = (540, 720)
MARGIN = 0.06
CC0_SRC = "/tmp/glove_refs/pair_cc0.jpg"

VARIANTS = (
    {"key": "plain", "title": "plain X Bot", "items": (), "colorway": "A", "version": "v3"},
    {"key": "v2A", "title": "v2 A  ·  compact (too small) + ribbed beanie",
     "items": ("gloves", "beanie"), "colorway": "A", "version": "v2"},
    {"key": "v3A", "title": "v3 A  ·  10–12 oz 0.8× head + crown-rib beanie",
     "items": ("gloves", "beanie"), "colorway": "A", "version": "v3"},
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


def _glove_objects(handle, side=None):
    out = []
    for n in handle.get("objects") or []:
        obj = bpy.data.objects.get(n)
        if not obj:
            continue
        if not n.startswith("XBotGlove_"):
            continue
        if side and f"XBotGlove_{side}_" not in n:
            continue
        out.append(obj)
    return out


def _beanie_objects(handle):
    out = []
    for n in handle.get("objects") or []:
        obj = bpy.data.objects.get(n)
        if obj and n.startswith("XBotBeanie_"):
            out.append(obj)
    return out


def _lower_or_punch_glove(handle, punch_side=None, prefer_lower=True):
    gloves = _glove_objects(handle)
    if punch_side and not prefer_lower:
        named = _glove_objects(handle, punch_side)
        if named:
            return named
    if not gloves:
        return []

    def _cz(o):
        pts = v1.evaluated_world_points([o])
        if not pts:
            return 0.0
        return sum(p.z for p in pts) / len(pts)

    return [min(gloves, key=_cz)]


def _thumb_side_yaw_elev(glove, body_center=None):
    """3/4 yaw from OUTSIDE the body so the torso does not fill the frame.

    A purely local +Z (thumb) direction previously parked the camera against
    the hip, leaving only a sliver of glove in the shot.
    """
    mw = glove.matrix_world.to_3x3()
    thumb = (mw @ Vector((0.0, 0.0, 1.0))).normalized()
    pts_guess = glove.location
    away = None
    if body_center is not None:
        away = (pts_guess - body_center)
        away.z = 0.0
        if away.length > 1e-6:
            away.normalize()
    # Prefer the sheet's outside 3/4 (yaw ~48), biased toward the thumb
    # and away from the torso.
    base = Vector((
        math.sin(math.radians(48.0)) * math.cos(math.radians(16.0)),
        -math.cos(math.radians(48.0)) * math.cos(math.radians(16.0)),
        math.sin(math.radians(16.0)),
    )).normalized()
    view = base * 0.70 + thumb * 0.25
    if away is not None:
        view = view + away * 0.35
    view.normalize()
    yaw = math.degrees(math.atan2(view.x, -view.y))
    elev = 16.0
    return yaw, elev


def stage_closeups(state, handle, stills, eevee, samples):
    """Dedicated tight cameras of the v3 glove (thumb side) and beanie."""
    punch = state.get("punch_side")
    prefer_lower = state.get("clip_kind") == "idle"
    gloves = _lower_or_punch_glove(handle, punch_side=punch, prefer_lower=prefer_lower)
    if not gloves and punch:
        gloves = _glove_objects(handle, punch)
    if not gloves:
        gloves = _glove_objects(handle)
    beanie = _beanie_objects(handle)

    shots = []
    if gloves:
        pts = v1.evaluated_world_points(gloves)
        body = v1.evaluated_world_points(list(state["meshes"]))
        if body:
            bc = Vector(sum((Vector(p) for p in body), Vector()) / len(body))
        else:
            bc = None
        yaw, elev = _thumb_side_yaw_elev(gloves[0], body_center=bc)
        v1.setup_covering_camera(pts, margin=0.10, yaw=yaw, elev=elev, res=CLOSE_RES)
        path = os.path.join(OUT_DIR, "xbot_outfit_v3_close_glove.png")
        v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
        stills["close_glove"] = path
        shots.append(path)
        print(f"WROTE {path} ({os.path.getsize(path)} bytes) yaw={yaw:.1f} elev={elev:.1f}",
              flush=True)
    if beanie:
        pts = v1.evaluated_world_points(beanie + list(state["meshes"]))
        bpts = v1.evaluated_world_points(beanie)
        if bpts:
            bc = Vector(sum((Vector(p) for p in bpts), Vector()) / len(bpts))
            nearby = [p for p in pts if (Vector(p) - bc).length < 0.22]
            pts = nearby or bpts
        # Slightly higher so the crown ribs / slouch fold read.
        v1.setup_covering_camera(pts, margin=0.12, yaw=28.0, elev=22.0, res=CLOSE_RES)
        path = os.path.join(OUT_DIR, "xbot_outfit_v3_close_beanie.png")
        v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
        stills["close_beanie"] = path
        shots.append(path)
        print(f"WROTE {path} ({os.path.getsize(path)} bytes)", flush=True)
    return shots


def _stage_cc0_crop():
    dest = os.path.join(OUT_DIR, "xbot_outfit_v3_ref_cc0.jpg")
    src = CC0_SRC
    if not os.path.isfile(src):
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
            "note": "Cropped and scaled to match the v3 glove close-up long axis; not in the mesh.",
        },
        "clips": [],
        "handles": {},
    }
    t0 = time.time()
    honest = []
    cc0 = _stage_cc0_crop()
    if cc0:
        stills["close_ref"] = cc0
    else:
        honest.append("CC0 reference photo was not on disk; sheet will describe proportions instead.")

    for pose in POSES:
        path = os.path.join(SRC_DIR, pose["file"])
        state = v1.prepare_clip(path, "idle" if pose["clip"] == "idle" else "punch")
        state["clip_kind"] = pose["clip"]
        frame = v1.set_pose(state, pose["stage"])
        rec = {
            "clip": pose["clip"], "stage": pose["stage"], "frame": frame,
            "lock": state["lock"], "drift": state["drift"],
            "punch_side": state["punch_side"], "keys": state["keys"],
        }
        # Shared camera per row: probe v3 (larger gloves) so v3 is not cropped.
        probe = apply_outfit(state["arm"], state["meshes"],
                             items=("gloves", "beanie"), colorway="A",
                             scan_frames=True, version="v3")
        objs = list(state["meshes"]) + [bpy.data.objects[n] for n in probe["objects"]
                                        if n in bpy.data.objects]
        pts = v1.evaluated_world_points(objs)
        v1.setup_covering_camera(pts, margin=MARGIN)
        rec["containment_probe"] = probe.get("containment")
        rec["notes"] = list(probe.get("notes") or [])
        rec["head_width"] = probe.get("head_width")
        clear_outfit(probe, state["meshes"])

        if pose["clip"] == "jab" and pose["stage"] == "full":
            totals, per, _info = xbot.paint_glow(
                state["meshes"], state["arm"], state["bones"],
                state["punch_side"], ("front_delt", "triceps"))
            rec["glow_totals"] = totals
            print(f"GLOW totals={totals} punch_side={state['punch_side']}", flush=True)
        else:
            xbot.clear_glow(state["meshes"])

        kept_v3 = None
        for var in VARIANTS:
            handle = None
            if var["items"]:
                handle = apply_outfit(state["arm"], state["meshes"],
                                      items=var["items"], colorway=var["colorway"],
                                      scan_frames=True, version=var["version"])
                facts["handles"][f"{pose['clip']}_{var['key']}"] = v1._handle_public(handle)
                if handle.get("containment"):
                    for side, c in handle["containment"].items():
                        if c.get("outside_padded_aabb"):
                            honest.append(
                                f"{pose['clip']} {var['key']} glove {side}: "
                                f"{c['outside_padded_aabb']} hand verts outside padded envelope"
                            )
            out = os.path.join(OUT_DIR, f"xbot_outfit_v3_{pose['clip']}_{pose['stage']}_{var['key']}.png")
            v1.render_still(out, eevee=eevee_stills, samples=samples)
            stills[f"{pose['clip']}_{pose['stage']}_{var['key']}"] = out
            print(f"WROTE {out} ({os.path.getsize(out)} bytes)", flush=True)
            if var["key"] == "v3A" and pose["clip"] == "jab":
                kept_v3 = handle
            elif handle:
                clear_outfit(handle, state["meshes"])

        if kept_v3 is not None:
            stage_closeups(state, kept_v3, stills, eevee_stills, samples)
            ginfo = (kept_v3.get("glove_info") or {}).get(state["punch_side"] or "R") or {}
            rec["v3_glove"] = {k: ginfo.get(k) for k in
                               ("body_len", "front_width", "front_thick",
                                "cuff_y_near", "cuff_y_far", "mask_plane_y",
                                "head_width", "fist_to_head")}
            rec["v3_beanie"] = kept_v3.get("beanie_info")
            clear_outfit(kept_v3, state["meshes"])

        facts["clips"].append(rec)

    facts["stills"] = stills
    facts["stills_elapsed_s"] = time.time() - t0

    honest.append("Assets: all outfit meshes are procedural (no third-party / paid / logo files).")
    honest.append("v3 gloves: fist front ~0.8× head width, longer padded knuckle, visible side thumb, same-red cuff.")
    honest.append("Beanie crown ribs are real geometry continuing to the top, plus a slight slouch/fold.")
    honest.append("Palm cavity / grip bar is a mild procedural indent — not a fully sculpted inner glove.")
    honest.append("Jab glow: anterior delt (purple) and triceps (yellow) on the punching side only.")
    facts["honest_notes"] = honest

    sheet = os.path.join(OUT_DIR, "xbot_outfit_v3.png")
    facts_path = os.path.join(OUT_DIR, "xbot_outfit_v3_facts.json")
    with open(facts_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    import subprocess
    subprocess.check_call([
        "/usr/bin/python3",
        os.path.join(HERE, "build_xbot_outfit_sheet_v3.py"),
        "--facts", facts_path,
    ])
    facts["sheet"] = sheet

    if not no_preview:
        facts["jab_preview"] = v1.render_jab_preview(
            "xbot_outfit_v3_jab_preview.mp4",
            colorway="A", items=("gloves", "beanie"), version="v3",
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
