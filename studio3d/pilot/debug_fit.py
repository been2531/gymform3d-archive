import bpy, sys, addon_utils
import numpy as np
from mathutils import Vector
addon_utils.enable("io_scene_gltf2", default_set=True, persistent=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
path = "/agent/repos/gymform3d-archive/assets3d/ecorche.glb"
bpy.ops.import_scene.gltf(filepath=path)
coords = []
for o in bpy.data.objects:
    if o.type == "MESH" and "latissimus" in o.name.lower():
        for v in o.data.vertices:
            coords.append(o.matrix_world @ v.co)
c = np.array(coords)
print("lat bbox", c.min(0), c.max(0), "size", c.max(0)-c.min(0))
