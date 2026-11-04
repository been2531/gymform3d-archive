"""Evidence dump: bone vs mesh vs grip targets (no render)."""
from __future__ import annotations

import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

DEMIC = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(DEMIC)
ROOT = os.path.dirname(os.path.dirname(PILOT))
sys.path.insert(0, DEMIC)
sys.path.insert(0, PILOT)
sys.path.insert(0, os.path.join(ROOT, "studio3d"))

import importlib.util

spec = importlib.util.spec_from_file_location("pilot_build_scene", os.path.join(PILOT, "build_scene.py"))
pilot_scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot_scene)

from build_scene_demic import load_settings  # noqa: E402
from pose_lat_pull import pose_lat_pull  # noqa: E402
from pose_seated_lat import pose_seated_legs  # noqa: E402


def wpos(arm, bn, tail=False):
    pb = arm.pose.bones[bn]
    p = pb.tail if tail else pb.head
    return np.array(arm.matrix_world @ p)


def mesh_nearest(body, p_world):
    mw = body.matrix_world
    imw = mw.inverted()
    pl = imw @ Vector(p_world)
    best = 1e9
    for v in body.data.vertices:
        d = (Vector(v.co) - pl).length
        if d < best:
            best = d
    return float(best)


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    form = argv[0] if len(argv) > 0 else "wrong"
    phase = float(argv[1]) if len(argv) > 1 else 0.5
    camera = argv[2] if len(argv) > 2 else "front"

    S = load_settings()
    S_mh = S["mannequin_mh"]
    EX = json.load(open(os.path.join(PILOT, "exercises", "wide_grip_lat_pulldown.json")))
    bpy.ops.wm.read_factory_settings(use_empty=True)

    body, arm, _ = pilot_scene.make_mannequin(S_mh)
    body.name = "mannequin"
    shorts = pilot_scene.make_shorts(S_mh, body)
    pose_seated_legs(arm, seat_h=S["equipment"]["seat_h"])
    pilot_scene.seat_character(arm, body, S)

    center, axis, wl, wr, _ = pose_lat_pull(arm, EX, form, phase, S["motion"])

    report = {
        "form": form,
        "phase": phase,
        "camera": camera,
        "objects": sorted(bpy.data.objects.keys()),
        "body_modifiers": [m.type for m in body.modifiers],
        "shorts_modifiers": [m.type for m in shorts.modifiers],
        "body_dup": body.data.name == shorts.data.name,
        "armature": arm.name,
    }

    side_data = {}
    for sd, target, label in (("L", wl, "left"), ("R", wr, "right")):
        sh = wpos(arm, f"shoulder01.{sd}")
        ua = wpos(arm, f"upperarm01.{sd}")
        ua_t = wpos(arm, f"upperarm01.{sd}", tail=True)
        la = wpos(arm, f"lowerarm01.{sd}")
        wr_b = wpos(arm, f"wrist.{sd}", tail=True)
        hand = wpos(arm, f"hand.{sd}", tail=True) if f"hand.{sd}" in arm.pose.bones else wr_b
        target = np.asarray(target, float)
        side_data[label] = {
            "grip_target_world": target.tolist(),
            "shoulder_world": sh.tolist(),
            "upperarm01_head_world": ua.tolist(),
            "upperarm01_tail_world": ua_t.tolist(),
            "lowerarm01_head_world": la.tolist(),
            "wrist_tail_world": wr_b.tolist(),
            "hand_tail_world": hand.tolist(),
            "shoulder_to_upperarm_head_m": float(np.linalg.norm(ua - sh)),
            "upperarm_bone_length_m": float(np.linalg.norm(ua_t - ua)),
            "wrist_to_target_m": float(np.linalg.norm(wr_b - target)),
            "hand_to_target_m": float(np.linalg.norm(hand - target)),
            "mesh_gap_at_shoulder_m": mesh_nearest(body, ua),
            "mesh_gap_at_wrist_m": mesh_nearest(body, wr_b),
        }

    report["arms"] = side_data

    # Apply same grip setup as build_scene_demic (pre-constraint)
    from equipment_demic import make_bar  # noqa: E402

    bar_root, _ = make_bar(S, center, axis, form)
    axv = Vector(axis.tolist())
    half_w = S["equipment"]["bar"]["width"] / 2
    grip_world = {}
    for sd, w in (("L", wl), ("R", wr)):
        g = bpy.data.objects.new(f"grip_{sd}", None)
        bpy.context.collection.objects.link(g)
        g.parent = bar_root
        off = float(np.dot(np.array(w) - center, axis))
        g.location = axv * off
        g_world = np.array(g.matrix_world.translation)
        grip_world[sd] = g_world.tolist()
        pb = arm.pose.bones.get(f"hand.{sd}") or arm.pose.bones.get(f"wrist.{sd}")
        if pb:
            c = pb.constraints.new("COPY_LOCATION")
            c.target = g
            c.influence = 0.72
    pilot_scene.upd()

    post = {}
    for sd, label in (("L", "left"), ("R", "right")):
        wr_b = wpos(arm, f"wrist.{sd}", tail=True)
        hand = wpos(arm, f"hand.{sd}", tail=True) if f"hand.{sd}" in arm.pose.bones else wr_b
        gw = np.array(grip_world[sd])
        post[label] = {
            "grip_empty_world": gw.tolist(),
            "wrist_tail_after_constraint": wr_b.tolist(),
            "hand_tail_after_constraint": hand.tolist(),
            "hand_to_grip_empty_m": float(np.linalg.norm(hand - gw)),
            "wrist_to_grip_empty_m": float(np.linalg.norm(wr_b - gw)),
            "shoulder_to_hand_m": float(np.linalg.norm(hand - wpos(arm, f"shoulder01.{sd}"))),
        }
    report["after_copy_location_0p72"] = post

    # Check bone chain continuity
    chain = {}
    for sd in ("L", "R"):
        names = [f"shoulder01.{sd}", f"upperarm01.{sd}", f"upperarm02.{sd}", f"lowerarm01.{sd}", f"lowerarm02.{sd}", f"wrist.{sd}"]
        gaps = []
        for a, b in zip(names[:-1], names[1:]):
            if a in arm.pose.bones and b in arm.pose.bones:
                ta = wpos(arm, a, tail=True)
                hb = wpos(arm, b)
                gaps.append({"from": a, "to": b, "head_tail_gap_m": float(np.linalg.norm(ta - hb))})
        chain[sd] = gaps
    report["bone_chain_gaps"] = chain

    out = os.path.join(ROOT, "output/pilot/demic/diagnose_arms.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
