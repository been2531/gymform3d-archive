"""List muscle mesh names in ecorche.glb (run via blender_cli.sh)."""
import bpy
import sys

import os

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
path = argv[0] if argv else "/agent/repos/gymform3d-archive/assets3d/ecorche.glb"
print("path", path, "exists", os.path.isfile(path))
bpy.ops.wm.read_factory_settings(use_empty=True)
import addon_utils

addon_utils.enable("io_scene_gltf2", default_set=True, persistent=True)
bpy.ops.import_scene.gltf(filepath=os.path.abspath(path))
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
print("total meshes", len(meshes))
skip = ("bone", "vertebra", "rib", "cartilage", "tooth", "sesamoid", "femur", "tibia", "fibula", "patella", "metacarpal", "metatarsal", "phalanx", "carpal", "tarsal", "skull", "mandible", "clavicle", "scapula", "humerus", "radius", "ulna", "pelvis", "sacrum", "sternum", "hyoid")
upper = [o for o in meshes if not any(s in o.name.lower() for s in skip)]
print("non-bone meshes", len(upper))
for o in sorted(upper, key=lambda x: x.name.lower())[:40]:
    print(o.name, len(o.data.vertices))
