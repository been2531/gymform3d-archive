"""Debug muscle visibility after full pose pipeline."""
import json, os, sys
import bpy
import numpy as np

PILOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(PILOT))
sys.path.insert(0, PILOT)
sys.path.insert(0, os.path.join(ROOT, "studio3d"))

import importlib.util

spec = importlib.util.spec_from_file_location("pilot_build", os.path.join(PILOT, "build_scene.py"))
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)
from anatomy_ecorche import import_upper_muscles

S = json.load(open(os.path.join(PILOT, "set_pilot.json")))
base = json.load(open(os.path.join(ROOT, "studio3d", "set_v1.json")))
S_mh = {**base, **S}
S_mh["asset_dir"] = S["asset_dir"]
S_mh["mannequin"] = {**base["mannequin"], **S["mannequin"]}
EX = json.load(open(os.path.join(PILOT, "exercises", "wide_grip_lat_pulldown.json")))

bpy.ops.wm.read_factory_settings(use_empty=True)
body, arm, _ = pb.make_mannequin(S_mh)
muscles = import_upper_muscles(arm)
pb.pose_lat(arm, EX, "right", 1.0)
pb.seat_character(arm, body, S)
pb.upd()
print("muscles", len(muscles))
for name in ("Right latissimus dorsi", "Long head of right biceps brachii"):
    o = bpy.data.objects.get(name)
    if not o:
        print("missing", name)
        continue
    coords = np.array([o.matrix_world @ v.co for v in o.data.vertices])
    print(name, "verts", len(coords), "bbox z", coords[:, 2].min(), coords[:, 2].max(), "y", coords[:, 1].min(), coords[:, 1].max())
