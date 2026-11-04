"""Headless bpy (t701u): MediaPipe world landmarks (npz from clip_pose.py) -> Grab3D CC0 muscle rig, code-made hex dumbbell
in the working hand, feet pinned, green target-muscle region painted by geometry, flat Day 4 studio, Cycles CPU low res.
Writes frames + per-frame checks JSON.
usage: /workspace/.bpyenv/bin/python tools/rig3d/retarget_render.py POSE.npz GLB OUTDIR [--f0 N --f1 N --res 540x960 --spp 16]"""
import sys, json, math, time, argparse, numpy as np, bpy, mathutils as mu
T0 = time.time()
def log(m): print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("glb"); ap.add_argument("out")
ap.add_argument("--f0", type=int, default=0); ap.add_argument("--f1", type=int, default=75); ap.add_argument("--res", default="540x960"); ap.add_argument("--spp", type=int, default=16)
ap.add_argument("--cam", default="2.6,-2.9,1.0"); ap.add_argument("--still-only", action="store_true")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
D = np.load(a.npz); P3 = D["P3"].copy(); side = int(D["side"])
# fill gaps + smooth (Gaussian in time) -> removes MediaPipe jitter
for j in range(33):
    for k in range(3):
        v = P3[:, j, k]; m = np.isnan(v)
        if m.any(): v[m] = np.interp(np.nonzero(m)[0], np.nonzero(~m)[0], v[~m])
k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2); k /= k.sum()   # Gaussian sigma 2.5 frames (bpy env has no scipy)
P3 = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 6, mode="edge"), k, "valid"), 0, P3)
B = np.stack([P3[..., 0], P3[..., 2], -P3[..., 1]], -1)          # MP (x right, y down, z depth) -> Blender (X, Y depth, Z up)
ank = (B[:, 27] + B[:, 28]) / 2; B = B - (ank - ank[a.f0])[:, None, :]   # pin the feet: cancel ankle drift
B[..., 2] -= min(B[a.f0, 27, 2], B[a.f0, 28, 2]) - 0.07               # ankles ~7 cm above the floor
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=a.glb)
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]; body = bpy.data.objects["char1"]
for o in bpy.data.objects:
    if o.name == "Icosphere": o.hide_render = True
if arm.animation_data:
    arm.animation_data.action = None
    for t in list(arm.animation_data.nla_tracks): arm.animation_data.nla_tracks.remove(t)
sc = bpy.context.scene; col = sc.collection
def empty(n):
    e = bpy.data.objects.new(n, None); col.objects.link(e); return e
L = {"hip": None, "sh": None}
names = {"hipmid": lambda f: (B[f, 23] + B[f, 24]) / 2, "shmid": lambda f: (B[f, 11] + B[f, 12]) / 2, "nose": lambda f: (B[f, 7] + B[f, 8]) / 2}
for j, n in [(11, "lsh"), (12, "rsh"), (13, "lel"), (14, "rel"), (15, "lwr"), (16, "rwr"), (23, "lhip"), (24, "rhip"), (25, "lkn"), (26, "rkn"), (27, "lank"), (28, "rank"), (31, "lft"), (32, "rft"), (19, "lix"), (20, "rix")]:
    names[n] = (lambda jj: (lambda f: B[f, jj]))(j)
E = {n: empty("T_" + n) for n in names}
for f in range(a.f0, a.f1):
    for n, fn in names.items(): E[n].location = mu.Vector(fn(f).tolist()); E[n].keyframe_insert("location", frame=f - a.f0 + 1)
log("empties keyed")
pb = arm.pose.bones
for b in pb: b.rotation_mode = "QUATERNION"; b.rotation_quaternion = (1, 0, 0, 0); b.location = (0, 0, 0)
def track(bone, tgt):
    c = pb[bone].constraints.new("DAMPED_TRACK"); c.target = E[tgt]; c.track_axis = "TRACK_Y"
c = pb["Hips"].constraints.new("COPY_LOCATION"); c.target = E["hipmid"]
track("Hips", "shmid"); track("Spine02", "shmid"); track("Spine01", "shmid"); track("Spine", "shmid"); track("neck", "nose"); track("Head", "nose")
# MediaPipe "left/right" = the subject's sides, same convention as the rig bone names
for s, p in (("Left", "l"), ("Right", "r")):
    track(s + "Arm", p + "el"); track(s + "ForeArm", p + "wr"); track(s + "Hand", p + "ix")
    track(s + "UpLeg", p + "kn"); track(s + "Leg", p + "ank"); track(s + "Foot", p + "ft")
# rig rest proportions differ from the person: scale the armature so its hip->shoulder length matches the clip
bpy.context.view_layer.update()
rest_t = (arm.matrix_world @ arm.data.bones["Spine"].tail_local - arm.matrix_world @ arm.data.bones["Hips"].head_local).length
clip_t = float(np.median(np.linalg.norm(B[:, 11] / 2 + B[:, 12] / 2 - B[:, 23] / 2 - B[:, 24] / 2, axis=1)))
log("constraints set")
# --- code-made hex dumbbell (palette A cobalt + steel), parented to the working hand
def hexhead(x):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.06, depth=0.07, location=(x, 0, 0), rotation=(0, math.pi / 2, 0)); return bpy.context.object
h1, h2 = hexhead(-0.12), hexhead(0.12); bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.016, depth=0.2, rotation=(0, math.pi / 2, 0)); hd = bpy.context.object
blue = bpy.data.materials.new("cobalt"); blue.use_nodes = True; bs = blue.node_tree.nodes["Principled BSDF"]; bs.inputs["Base Color"].default_value = (0.02, 0.05, 0.45, 1); bs.inputs["Roughness"].default_value = 0.6
steel = bpy.data.materials.new("steel"); steel.use_nodes = True; ss = steel.node_tree.nodes["Principled BSDF"]; ss.inputs["Base Color"].default_value = (0.5, 0.5, 0.52, 1); ss.inputs["Metallic"].default_value = 1; ss.inputs["Roughness"].default_value = 0.35
for o in (h1, h2): o.data.materials.append(blue)
hd.data.materials.append(steel)
for o in (h1, h2): o.select_set(True)
hd.select_set(True); bpy.context.view_layer.objects.active = hd; bpy.ops.object.join(); db = bpy.context.object; db.name = "dumbbell"
hand = ("Left" if side == 11 else "Right") + "Hand"; wr_e, ix_e = (("lwr", "lix") if side == 11 else ("rwr", "rix"))
# dumbbell centre = between wrist and index MCP, handle perpendicular to the hand->index line and to the body's side axis
cdb = db.constraints.new("COPY_LOCATION"); cdb.target = E[ix_e]
log("dumbbell made")
# --- materials: charcoal matte, keep the sculpt normal map; green target region (lats / rhomboids / mid trap) by rest geometry
me = body.data; vcol = me.color_attributes.new("tgt", "FLOAT_COLOR", "POINT")
nv = len(me.vertices); co = np.empty(nv * 3); me.vertices.foreach_get("co", co); nr = np.empty(nv * 3); me.vertices.foreach_get("normal", nr)
M4 = np.array(body.matrix_world); co = co.reshape(-1, 3) @ M4[:3, :3].T + M4[:3, 3]; nr = nr.reshape(-1, 3) @ M4[:3, :3].T
nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9; zr = (co[:, 2] - co[:, 2].min()) / np.ptp(co[:, 2])
# glTF import: the figure faces -Y, so the back faces +Y
t = ((nr[:, 1] > 0.25) & (zr >= 0.55) & (zr <= 0.80) & (np.abs(co[:, 0]) <= 0.17 + 0.25 * (zr - 0.55))).astype(np.float32)
cols = np.stack([t, t, t, np.ones_like(t)], 1).ravel(); vcol.data.foreach_set("color", cols)
for s in body.material_slots:
    nt = s.material.node_tree; bsdf = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"][0]
    for l in list(bsdf.inputs["Base Color"].links): nt.links.remove(l)
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "tgt"; mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs[6].default_value = (0.06, 0.065, 0.07, 1); mix.inputs[7].default_value = (0.03, 0.55, 0.18, 1)
    nt.links.new(attr.outputs["Fac"], mix.inputs[0]); nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"]); bsdf.inputs["Roughness"].default_value = 0.75
log("materials done")
# --- studio: flat light, Day 4 backdrop
sc.render.engine = "CYCLES"; sc.cycles.samples = a.spp; sc.cycles.device = "CPU"; rx, ry = map(int, a.res.split("x")); sc.render.resolution_x, sc.render.resolution_y = rx, ry
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True; bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.0168, 0.0203, 0.0242, 1); bg.inputs[1].default_value = 1.0
for loc, e in (((3, -3, 3), 3.0), ((-3, -2, 2), 1.8), ((0, 4, 3), 2.0), ((0, -4, 0.5), 1.2)):
    Lt = bpy.data.lights.new("l", "AREA"); Lt.energy = e * 150; Lt.size = 4; o = bpy.data.objects.new("l", Lt); col.objects.link(o); o.location = loc
    o.rotation_euler = (mu.Vector((0, 0, 0.9)) - mu.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
cam = bpy.data.cameras.new("c"); C = bpy.data.objects.new("c", cam); col.objects.link(C); sc.camera = C
C.location = tuple(map(float, a.cam.split(","))); cam.lens = 32
tgt = mu.Vector(((B[a.f0, 23] + B[a.f0, 24]) / 2).tolist()); tgt.z = 0.75; C.rotation_euler = (tgt - C.location).to_track_quat("-Z", "Y").to_euler()
sc.frame_start, sc.frame_end = 1, a.f1 - a.f0; sc.render.fps = int(D["fps"])
log("scene ready")
# --- per-frame checks on the evaluated rig
chk = []
for f in range(1, a.f1 - a.f0 + 1):
    sc.frame_set(f); dg = bpy.context.evaluated_depsgraph_get(); M = arm.matrix_world
    P = lambda b, t="head": M @ (getattr(pb[b], t))
    lsh, rsh, lwr, rwr = P("LeftArm"), P("RightArm"), P("LeftHand"), P("RightHand")
    hip, shc = P("Hips"), (lsh + rsh) / 2
    # arm crossing: wrists swap their order along the body's left-right axis vs the shoulders
    lr = (rsh - lsh).normalized(); cross = ((rwr - lwr).dot(lr) < 0)
    torso = math.degrees(math.atan2(math.hypot((shc - hip).x, (shc - hip).y), (shc - hip).z))
    hpos = P(hand, "tail"); dbpos = db.matrix_world.translation; d_hand = (dbpos - hpos).length
    chk.append({"f": f, "arm_cross": bool(cross), "torso_deg": round(torso, 1), "db_hand_m": round(d_hand, 3)})
json.dump({"frames": chk, "rest_torso_m": round(rest_t, 3), "clip_torso_m": round(clip_t, 3), "working": hand}, open(a.out + "/checks.json", "w"), indent=0)
log("checks done")
sc.render.filepath = a.out + "/f_"; sc.render.image_settings.file_format = "PNG"
t = time.time()
if a.still_only:
    for f in (1, (a.f1 - a.f0) // 2):
        sc.frame_set(f); sc.render.filepath = a.out + f"/still_{f:03d}.png"; bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)
print("RENDER_S", round(time.time() - t, 1))
