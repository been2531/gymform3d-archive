#!/usr/bin/env python3
"""Render the X Bot outfit v2 comparison sheet and a short jab preview.

  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v2.py
  nice -n 19 blender -b --python tools/rig3d/render_xbot_outfit_v2.py -- --eevee-stills

Outfit A only (crimson gloves + charcoal beanie). Headgear is dropped.
Columns: plain | v1 A | v2 A. Rows: Idle guard, Jab full, close-ups.

Does not modify eval_xbot_boxing.py. Reuses clip / studio helpers from
render_xbot_outfit_v1.py and apply_outfit(..., version=).

Writes (gitignored):
  output/_basemodels/eval/xbot_outfit_v2.png
  output/_basemodels/eval/xbot_outfit_v2_jab_preview.mp4
  output/_basemodels/eval/xbot_outfit_v2_facts.json
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
    {"key": "plain", "title": "plain X Bot", "items": (), "colorway": "A", "version": "v2"},
    {"key": "v1A", "title": "v1 A  ·  crimson + charcoal", "items": ("gloves", "beanie"),
     "colorway": "A", "version": "v1"},
    {"key": "v2A", "title": "v2 A  ·  compact 10–12 oz + ribbed beanie",
     "items": ("gloves", "beanie"), "colorway": "A", "version": "v2"},
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
    # pick the glove with the lower world-space centroid (idle lead hand)
    def _cz(o):
        pts = v1.evaluated_world_points([o])
        if not pts:
            return 0.0
        return sum(p.z for p in pts) / len(pts)
    return [min(gloves, key=_cz)]


def stage_closeups(state, handle, stills, eevee, samples):
    """Dedicated tight cameras of the v2 glove and beanie."""
    punch = state.get("punch_side")
    # Glove: prefer the punch hand on the jab; otherwise the lower glove.
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
        # 3/4 side, close to the reference photo angle
        v1.setup_covering_camera(pts, margin=0.10, yaw=48.0, elev=22.0, res=CLOSE_RES)
        path = os.path.join(OUT_DIR, "xbot_outfit_v2_close_glove.png")
        v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
        stills["close_glove"] = path
        shots.append(path)
        print(f"WROTE {path} ({os.path.getsize(path)} bytes)", flush=True)
    if beanie:
        # include a bit of the skull so the cuff-on-forehead read is clear
        pts = v1.evaluated_world_points(beanie + list(state["meshes"]))
        # tighter: just beanie + nearby head verts (filter by distance to beanie)
        bpts = v1.evaluated_world_points(beanie)
        if bpts:
            bc = Vector(sum((Vector(p) for p in bpts), Vector()) / len(bpts))
            nearby = [p for p in pts if (Vector(p) - bc).length < 0.22]
            pts = nearby or bpts
        v1.setup_covering_camera(pts, margin=0.12, yaw=28.0, elev=16.0, res=CLOSE_RES)
        path = os.path.join(OUT_DIR, "xbot_outfit_v2_close_beanie.png")
        v1.render_still(path, eevee=eevee, samples=samples, res=CLOSE_RES)
        stills["close_beanie"] = path
        shots.append(path)
        print(f"WROTE {path} ({os.path.getsize(path)} bytes)", flush=True)
    return shots


def _stage_cc0_crop():
    """Copy the CC0 reference into OUT_DIR for the sheet builder."""
    dest = os.path.join(OUT_DIR, "xbot_outfit_v2_ref_cc0.jpg")
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
            "note": "Used only as a small labeled crop on the sheet; not in the mesh.",
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

    jab_state = None
    jab_v2_handle = None

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
        # Shared camera per row: probe v1 (longer gloves) so v1 is not cropped.
        probe = apply_outfit(state["arm"], state["meshes"],
                             items=("gloves", "beanie"), colorway="A",
                             scan_frames=True, version="v1")
        objs = list(state["meshes"]) + [bpy.data.objects[n] for n in probe["objects"]
                                        if n in bpy.data.objects]
        pts = v1.evaluated_world_points(objs)
        v1.setup_covering_camera(pts, margin=MARGIN)
        rec["containment_probe"] = probe.get("containment")
        rec["notes"] = list(probe.get("notes") or [])
        clear_outfit(probe, state["meshes"])

        if pose["clip"] == "jab" and pose["stage"] == "full":
            xbot.paint_glow(state["meshes"], state["arm"], state["bones"],
                            state["punch_side"], ("front_delt", "triceps"))
        else:
            xbot.clear_glow(state["meshes"])

        kept_v2 = None
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
            out = os.path.join(OUT_DIR, f"xbot_outfit_v2_{pose['clip']}_{pose['stage']}_{var['key']}.png")
            v1.render_still(out, eevee=eevee_stills, samples=samples)
            stills[f"{pose['clip']}_{pose['stage']}_{var['key']}"] = out
            print(f"WROTE {out} ({os.path.getsize(out)} bytes)", flush=True)
            if var["key"] == "v2A" and pose["clip"] == "jab":
                kept_v2 = handle
                jab_state = state
                jab_v2_handle = handle
            elif handle:
                clear_outfit(handle, state["meshes"])

        if kept_v2 is not None:
            # close-ups from the jab, v2 still applied
            stage_closeups(state, kept_v2, stills, eevee_stills, samples)
            ginfo = (kept_v2.get("glove_info") or {}).get(state["punch_side"] or "R") or {}
            rec["v2_glove"] = {k: ginfo.get(k) for k in
                               ("body_len", "front_width", "front_thick",
                                "cuff_y_near", "cuff_y_far", "mask_plane_y") if k in ginfo or True}
            rec["v2_beanie"] = kept_v2.get("beanie_info")
            clear_outfit(kept_v2, state["meshes"])
            jab_v2_handle = None

        facts["clips"].append(rec)

    facts["stills"] = stills
    facts["stills_elapsed_s"] = time.time() - t0

    honest.append("Assets: all outfit meshes are procedural (no third-party / paid / logo files).")
    honest.append("v2 gloves: compact rounded fist + distinct 9 cm cuff; rigid to the Hand bone (no finger-curl squash).")
    honest.append("Knit ribs are real displaced geometry, but a 9:16 phone still may soften the finest crown ridges.")
    honest.append("Palm cavity / grip bar is a mild procedural indent — not a fully sculpted inner glove.")
    facts["honest_notes"] = honest

    sheet = os.path.join(OUT_DIR, "xbot_outfit_v2.png")
    facts_path = os.path.join(OUT_DIR, "xbot_outfit_v2_facts.json")
    with open(facts_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    import subprocess
    subprocess.check_call([
        "/usr/bin/python3",
        os.path.join(HERE, "build_xbot_outfit_sheet_v2.py"),
        "--facts", facts_path,
    ])
    facts["sheet"] = sheet

    if not no_preview:
        remaining = 420.0 - (time.time() - t0)
        if remaining > 40 or eevee_stills:
            facts["jab_preview"] = v1.render_jab_preview(
                "xbot_outfit_v2_jab_preview.mp4",
                colorway="A", items=("gloves", "beanie"), version="v2",
            )
        else:
            facts["jab_preview"] = {
                "skipped": True,
                "reason": f"stills used {facts['stills_elapsed_s']:.0f}s; no budget left",
            }
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
