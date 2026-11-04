"""Headless bpy, rig3d v2 (Routinizer, 2026-11-04). MediaPipe world landmarks (clip_pose.py npz) -> Grab3D CC0 muscle rig.
v2 vs v1 (retarget_render.py, kept unchanged):
  - torso clamp: hip->shoulder vector re-aimed every frame to cfg.motion.torso_target_deg from vertical (azimuth kept,
    smoothed); the head follows the torso rigidly, each arm is TRANSLATED with its shoulder (keeps its world direction,
    so a hanging arm still hangs) -> torso steady instead of swinging 64-91 deg
  - both feet locked to their first-frame position (ankle, heel, toe)
  - dumbbell baked to a grip point on the HAND bone (wrist + 7.5 cm along the bone axis), handle across the body;
    the v1 check used pose_bone.tail, which is garbage on this GLB (leaf tails are ~13 m long) = the "unit bug"
  - tight side camera on the working-arm side turned toward the back, auto-framed from the deformed mesh
  - Poly Haven CC0 HDRI fill + key lights, transparent film (backdrop comes from the Day 4 grade)
  - second render pass = muscle-region mask (R lats, G rhomboids, B mid trap; dumbbell = holdout)
  - per-frame checks: arm crossing, torso angle, grip distance (bone + nearest hand-mesh vertex), mesh tearing
    (deformed/rest edge-length ratio), working-elbow range
usage: python -m bpy-env tools/rig3d/retarget_render_v2.py --npz P.npz --cfg run_cfg.json --out DIR
       [--frames all|N,M (render only those 1-based frames)] [--checks-only] [--no-mask] [--mask-only]"""
import sys, json, math, time, argparse, numpy as np, bpy, mathutils as mu
from bpy_extras.object_utils import world_to_camera_view
T0 = time.time()
def log(m): print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="all"); ap.add_argument("--checks-only", action="store_true"); ap.add_argument("--no-mask", action="store_true")
ap.add_argument("--mask-only", action="store_true"); ap.add_argument("--res", default=None)
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
C = json.load(open(a.cfg)); MO, RC = C["motion"], C["render"]
D = np.load(a.npz); P3 = D["P3"].copy(); side = int(D["side"]); f0, f1 = int(C["_rep"][0]), int(C["_rep"][1])
NF = f1 - f0

def gsmooth(X, sig):
    r = int(3 * sig) + 1; k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 0, X)
for j in range(33):
    for k in range(3):
        v = P3[:, j, k]; m = np.isnan(v)
        if m.any() and (~m).any(): v[m] = np.interp(np.nonzero(m)[0], np.nonzero(~m)[0], v[~m])
P3 = gsmooth(P3, float(MO.get("smooth_sigma_f", 2.5)))
B = np.stack([P3[..., 0], P3[..., 2], -P3[..., 1]], -1)          # MP (x right, y down, z depth) -> Blender (X, Y depth, Z up)
B = B[f0:f1].copy()
# level the floor: MediaPipe world axes follow the source camera (pitched), so fit a plane through both heels + toes
# (median over the rep) and rotate everything so its normal is +Z
FP = np.nanmedian(B[:, [29, 30, 31, 32]], axis=0); cen = FP.mean(0)
# ridge fit z = a x + b y + c (a staggered stance makes the 4 points nearly collinear: the tilt across the foot line is
# undetermined, so it is pulled to 0 instead of letting an SVD pick an arbitrary vertical plane)
Xf = FP[:, :2] - cen[:2]; lam = 0.002; ab = np.linalg.solve(Xf.T @ Xf + lam * np.eye(2), Xf.T @ (FP[:, 2] - cen[2]))
nrm = np.array([-ab[0], -ab[1], 1.0]); nrm /= np.linalg.norm(nrm)
hipc = np.nanmedian((B[:, 23] + B[:, 24]) / 2, axis=0)
if np.dot(nrm, hipc - cen) < 0: nrm = -nrm
floor_tilt = math.degrees(math.acos(np.clip(nrm[2], -1, 1)))
axl = np.cross(nrm, [0.0, 0.0, 1.0])
if np.linalg.norm(axl) > 1e-9:
    angl = math.acos(np.clip(nrm[2], -1, 1)); axl /= np.linalg.norm(axl)
    K = np.array([[0, -axl[2], axl[1]], [axl[2], 0, -axl[0]], [-axl[1], axl[0], 0]]); Rl = np.eye(3) + math.sin(angl) * K + (1 - math.cos(angl)) * K @ K
    B = (B - cen) @ Rl.T + cen

# ---- torso clamp -------------------------------------------------------------------------------------------------
def rodrigues(v, axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang))
tgt = math.radians(float(MO.get("torso_target_deg", 45)))
hipm = (B[:, 23] + B[:, 24]) / 2; shm = (B[:, 11] + B[:, 12]) / 2; V = shm - hipm
az = np.unwrap(np.arctan2(V[:, 1], V[:, 0])); az = gsmooth(az[:, None], float(MO.get("torso_smooth_sigma_f", 4.0)))[:, 0]
raw_torso = np.degrees(np.arctan2(np.hypot(V[:, 0], V[:, 1]), V[:, 2]))
for f in range(NF):
    v = V[f]; L = np.linalg.norm(v)
    newv = L * np.array([math.sin(tgt) * math.cos(az[f]), math.sin(tgt) * math.sin(az[f]), math.cos(tgt)])
    # rotation taking v -> newv, applied to shoulders + head about the hip midpoint
    ax = np.cross(v, newv); s = np.linalg.norm(ax)
    ang = math.atan2(s, float(np.dot(v, newv)))
    old_sh = {11: B[f, 11].copy(), 12: B[f, 12].copy()}
    if s > 1e-9:
        for j in list(range(0, 11)) + [11, 12]:
            B[f, j] = hipm[f] + rodrigues(B[f, j] - hipm[f], ax, ang)
    for sj, arm_js in ((11, (13, 15, 17, 19, 21)), (12, (14, 16, 18, 20, 22))):
        d = B[f, sj] - old_sh[sj]
        if sj == side:                       # working arm: rigid shift with its shoulder (keeps the row path)
            for j in arm_js: B[f, j] += d
        else:                                # support arm: the hand stays planted (knee/bench), the elbow moves half way
            B[f, arm_js[0]] += 0.5 * d
if MO.get("lock_feet", True):
    for j in (27, 28, 29, 30, 31, 32): B[:, j] = B[0, j]
B[..., 2] -= min(B[0, 27, 2], B[0, 28, 2]) - 0.07               # ankles ~7 cm above the floor

np.save(a.out + "/landmarks_processed.npy", B)
# ---- scene ---------------------------------------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=C["asset"]["glb"])
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]; body = bpy.data.objects["char1"]
for o in bpy.data.objects:
    if o.name == "Icosphere": o.hide_render = True; o.hide_viewport = True
if arm.animation_data:
    arm.animation_data.action = None
    for t in list(arm.animation_data.nla_tracks): arm.animation_data.nla_tracks.remove(t)
sc = bpy.context.scene; col = sc.collection
# scale the person to the rig: the targets only give directions, but bone heads sit at RIG lengths, so a smaller
# person (MediaPipe world scale) folds the legs (rig knee below the clip ankle). Uniform scale about the feet.
bpy.context.view_layer.update(); Wr = arm.matrix_world; bl = arm.data.bones
def rl(a_, b_): return (Wr @ bl[b_].head_local - Wr @ bl[a_].head_local).length
rig_leg = (rl("LeftUpLeg", "LeftLeg") + rl("LeftLeg", "LeftFoot") + rl("RightUpLeg", "RightLeg") + rl("RightLeg", "RightFoot")) / 2
rig_tor = ((Wr @ bl["LeftArm"].head_local + Wr @ bl["RightArm"].head_local) / 2 - Wr @ bl["Hips"].head_local).length
nl = lambda i, j: np.median(np.linalg.norm(B[:, i] - B[:, j], axis=1))
clip_leg = (nl(23, 25) + nl(25, 27) + nl(24, 26) + nl(26, 28)) / 2
clip_tor = float(np.median(np.linalg.norm((B[:, 11] + B[:, 12]) / 2 - (B[:, 23] + B[:, 24]) / 2, axis=1)))
SCALE = float(0.5 * rig_leg / clip_leg + 0.5 * rig_tor / clip_tor)
fc_ = (B[0, 27] + B[0, 28]) / 2; fc_[2] = 0.0; B = fc_ + SCALE * (B - fc_)
B[..., 2] -= min(B[0, 27, 2], B[0, 28, 2]) - (Wr @ bl["LeftFoot"].head_local).z        # ankles at the rig's rest ankle height
log(f"person->rig scale {SCALE:.3f} (leg {rig_leg:.2f}/{clip_leg:.2f}, torso {rig_tor:.2f}/{clip_tor:.2f})")
def empty(n):
    e = bpy.data.objects.new(n, None); col.objects.link(e); return e
names = {"hipmid": lambda f: (B[f, 23] + B[f, 24]) / 2, "shmid": lambda f: (B[f, 11] + B[f, 12]) / 2, "nose": lambda f: (B[f, 7] + B[f, 8]) / 2}
for j, n in [(11, "lsh"), (12, "rsh"), (13, "lel"), (14, "rel"), (15, "lwr"), (16, "rwr"), (23, "lhip"), (24, "rhip"), (25, "lkn"), (26, "rkn"),
             (27, "lank"), (28, "rank"), (31, "lft"), (32, "rft"), (19, "lix"), (20, "rix")]:
    names[n] = (lambda jj: (lambda f: B[f, jj]))(j)
E = {n: empty("T_" + n) for n in names}
for f in range(NF):
    for n, fn in names.items(): E[n].location = mu.Vector(fn(f).tolist()); E[n].keyframe_insert("location", frame=f + 1)
pb = arm.pose.bones
for b in pb: b.rotation_mode = "QUATERNION"; b.rotation_quaternion = (1, 0, 0, 0); b.location = (0, 0, 0)
def track(bone, tgt_):
    c = pb[bone].constraints.new("DAMPED_TRACK"); c.target = E[tgt_]; c.track_axis = "TRACK_Y"
c = pb["Hips"].constraints.new("COPY_LOCATION"); c.target = E["hipmid"]
for bn in ("Hips", "Spine02", "Spine01", "Spine"): track(bn, "shmid")
for bn in ("Spine02", "Spine01", "Spine"):          # < 1 spreads the bend over the chain (less skin stretch at one joint)
    pb[bn].constraints[-1].influence = float(MO.get("spine_influence", 1.0))
track("neck", "nose"); track("Head", "nose")
for s, p in (("Left", "l"), ("Right", "r")):
    track(s + "Arm", p + "el"); track(s + "ForeArm", p + "wr"); track(s + "Hand", p + "ix")
    track(s + "UpLeg", p + "kn"); track(s + "Leg", p + "ank"); track(s + "Foot", p + "ft")
hand = ("Left" if side == 11 else "Right") + "Hand"
log("rig constrained")

# ---- rest geometry: muscle regions, shorts, hand vertices, edges ----------------------------------------------------
me = body.data; nv = len(me.vertices)
co = np.empty(nv * 3); me.vertices.foreach_get("co", co); nr = np.empty(nv * 3); me.vertices.foreach_get("normal", nr)
M4 = np.array(body.matrix_world); co = co.reshape(-1, 3) @ M4[:3, :3].T + M4[:3, 3]; nr = nr.reshape(-1, 3) @ M4[:3, :3].T
nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
x, y, z = co[:, 0], co[:, 1], co[:, 2]; ax_ = np.abs(x)
back = nr[:, 1] > 0.15                                   # glTF import: figure faces -Y, back faces +Y
trunk = ax_ <= 0.19
mid_trap = back & (z >= 1.30) & (z <= 1.45) & (ax_ <= 0.13) & (y > 0.0)
rhomb = back & (z >= 1.20) & (z < 1.34) & (ax_ >= 0.012) & (ax_ <= 0.085) & (y > 0.02) & ~mid_trap
zl = np.clip(z - 1.0, 0, 0.31)                            # lats: V shape, narrow at the lumbar, wide at the armpit
lats = trunk & (z >= 1.00) & (z <= 1.31) & (ax_ >= 0.035 + 0.12 * zl) & (ax_ <= 0.085 + 0.33 * zl) & ((nr[:, 1] > 0.05) | (np.abs(nr[:, 0]) > 0.55)) & (nr[:, 1] > -0.25) & ~mid_trap & ~rhomb
shorts = (z >= 0.80) & (z <= 1.02) & (ax_ <= 0.22)
reg = np.stack([lats, rhomb, mid_trap], 1).astype(np.float32)
va = me.color_attributes.new("mregion", "FLOAT_COLOR", "POINT"); va.data.foreach_set("color", np.concatenate([reg, np.ones((nv, 1), np.float32)], 1).ravel())
vs = me.color_attributes.new("shorts", "FLOAT_COLOR", "POINT"); s4 = np.repeat(shorts.astype(np.float32)[:, None], 4, 1); s4[:, 3] = 1; vs.data.foreach_set("color", s4.ravel())
gi = body.vertex_groups[hand].index; hand_vi = []; dom = np.full(nv, -1)
for v in me.vertices:
    best = (0.0, -1)
    for g in v.groups:
        if g.weight > best[0]: best = (g.weight, g.group)
        if g.group == gi and g.weight > 0.5: hand_vi.append(v.index)
    dom[v.index] = best[1]
hand_vi = np.array(hand_vi); GN = {g.index: g.name for g in body.vertex_groups}
ev = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ev); ev = ev.reshape(-1, 2)
rest_len = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1); ok_e = rest_len > 1e-4
region_counts = {"lats": int(lats.sum()), "rhomboids": int(rhomb.sum()), "mid_trap": int(mid_trap.sum()), "shorts": int(shorts.sum()), "hand_verts": int(len(hand_vi))}
log(f"regions {region_counts}")

# ---- black boxer briefs (드로즈): a separate garment mesh copied from the body's own pelvis/upper-thigh faces, pushed out
# along the normals, keeping the body's vertex groups + armature modifier -> skinned exactly like the body (no sim).
GC = C.get("garment", {"enabled": False})
garm, g_src = None, None
if GC.get("enabled"):
    import bmesh
    gz0, gz1, gx = float(GC.get("z_hem", 0.78)), float(GC.get("z_waist", 1.03)), float(GC.get("max_abs_x", 0.22))
    sel = (z >= gz0) & (z <= gz1) & (ax_ <= gx)
    garm = body.copy(); garm.data = body.data.copy(); garm.name = "briefs"; col.objects.link(garm)
    gm_ = garm.data
    for at_ in list(gm_.color_attributes): gm_.color_attributes.remove(at_)
    src = gm_.attributes.new("src_vi", "INT", "POINT"); src.data.foreach_set("value", np.arange(nv, dtype=np.int32))
    bm = bmesh.new(); bm.from_mesh(gm_)
    # straight waistband + leg hems: cut the body along the two horizontal planes (rest pose, world z) instead of
    # keeping whole triangles (that left a saw-tooth fringe), then keep the faces whose centre lies in the band
    Mi = np.linalg.inv(M4)
    for zc in (gz0, gz1):
        co_l = Mi[:3, :3] @ np.array([0, 0, zc]) + Mi[:3, 3]; no_l = M4[:3, :3].T @ np.array([0, 0, 1.0])   # plane normal: local = M^T world
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-6, plane_co=tuple(co_l), plane_no=tuple(no_l / np.linalg.norm(no_l)))
    def wz(v_): return float(M4[2, :3] @ np.array(v_.co) + M4[2, 3])
    def wx(v_): return float(M4[0, :3] @ np.array(v_.co) + M4[0, 3])
    drop = [f_ for f_ in bm.faces if not (gz0 - 1e-4 <= sum(wz(v_) for v_ in f_.verts) / len(f_.verts) <= gz1 + 1e-4 and all(abs(wx(v_)) <= gx for v_ in f_.verts))]
    bmesh.ops.delete(bm, geom=drop, context="FACES_ONLY")
    bmesh.ops.delete(bm, geom=[v_ for v_ in bm.verts if not v_.link_faces], context="VERTS")
    # weld the asset's UV-seam duplicates (else the offset opens hairline cracks)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.02)
    bm.verts.ensure_lookup_table(); bm.normal_update()
    # offset outward (the Meshy-made body has some inward-facing normals: orient each normal away from its limb axis,
    # pelvis = vertical axis through x=0 above the crotch, thighs = x=+-0.09 below it)
    off_local = float(GC.get("offset_m", 0.008)) / float(np.array(body.matrix_world.to_scale()).mean())
    gco = np.array([v_.co for v_ in bm.verts]); lnr_g = np.array([v_.normal for v_ in bm.verts])
    vw = gco @ M4[:3, :3].T + M4[:3, 3]; nw_g = lnr_g @ M4[:3, :3].T; cx_ = np.where((vw[:, 2] < 0.92) & (np.abs(vw[:, 0]) > 0.03), np.sign(vw[:, 0]) * 0.09, 0.0)
    outw = np.stack([vw[:, 0] - cx_, vw[:, 1], np.zeros(len(vw))], 1)
    g_sign = np.where((nw_g * outw).sum(1) < 0, -1.0, 1.0); dn = lnr_g * g_sign[:, None]; n_flip = int((g_sign < 0).sum())
    for i_, v_ in enumerate(bm.verts): v_.co = tuple(gco[i_] + off_local * dn[i_])
    # hem/waist lip: fold every open edge back in to ~1.5 mm off the skin so no skin shows under the band edge
    bnd_e = [e_ for e_ in bm.edges if e_.is_boundary]; bvi = {v_ for e_ in bnd_e for v_ in e_.verts}
    dn_of = {v_: dn[v_.index] for v_ in bvi}
    ret = bmesh.ops.extrude_edge_only(bm, edges=bnd_e)
    newv = [g_ for g_ in ret["geom"] if isinstance(g_, bmesh.types.BMVert)]
    lip_back = off_local * (1.0 - float(GC.get("lip_offset_frac", 0.12)))
    # each new vert was extruded from a boundary vert: find it through the connecting (non-boundary) edge
    for v_ in newv:
        src_v = next((o_ for e_ in v_.link_edges for o_ in e_.verts if o_ is not v_ and o_ in dn_of), None)
        if src_v is not None: v_.co = tuple(np.array(v_.co) - lip_back * dn_of[src_v])
    bm.verts.ensure_lookup_table()
    for v_ in bm.verts: v_.select_set(False)
    for v_ in bvi: v_.select_set(True)                                   # waist/hem rim (for the QC dump)
    bm.normal_update(); bm.to_mesh(gm_); bm.free(); gm_.update()
    gnv = len(gm_.vertices); g_bnd = np.array([v_.select for v_ in gm_.vertices]); g_src = np.empty(gnv, np.int32); gm_.attributes["src_vi"].data.foreach_get("value", g_src)
    g_src = np.clip(g_src, 0, nv - 1)
    gco = np.empty(gnv * 3); gm_.vertices.foreach_get("co", gco); gco = gco.reshape(-1, 3)
    vw = gco @ M4[:3, :3].T + M4[:3, 3]
    cx_ = np.where((vw[:, 2] < 0.92) & (np.abs(vw[:, 0]) > 0.03), np.sign(vw[:, 0]) * 0.09, 0.0); outw = np.stack([vw[:, 0] - cx_, vw[:, 1], np.zeros(len(vw))], 1)
    region_counts["briefs_flipped_normals"] = n_flip
    b_sign = np.where((nr[g_src] * outw).sum(1) < 0, -1.0, 1.0)   # same outward rule for the body's own normals
    gmat = bpy.data.materials.new("briefs_black"); gmat.use_nodes = True; gb = gmat.node_tree.nodes["Principled BSDF"]
    gb.inputs["Base Color"].default_value = (0.012, 0.012, 0.014, 1); gb.inputs["Roughness"].default_value = 0.95
    for k_, v_ in (("Specular IOR Level", 0.15), ("Coat Weight", 0.0), ("Sheen Weight", 0.0)):
        if k_ in gb.inputs: gb.inputs[k_].default_value = v_
    gm_.materials.clear(); gm_.materials.append(gmat)
    gev = np.empty(len(gm_.edges) * 2, np.int64); gm_.edges.foreach_get("vertices", gev); gev = gev.reshape(-1, 2)
    gco_w = gco @ M4[:3, :3].T + M4[:3, 3]; g_rest = np.linalg.norm(gco_w[gev[:, 0]] - gco_w[gev[:, 1]], axis=1); g_ok = g_rest > 1e-4
    region_counts.update({"briefs_verts": int(gnv), "briefs_faces": len(gm_.polygons)})
    log(f"briefs garment: {gnv} verts, {len(gm_.polygons)} faces, offset {GC.get('offset_m', 0.008)} m")

# ---- dumbbell ---------------------------------------------------------------------------------------------------------
def hexhead(xx):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.06, depth=0.07, location=(xx, 0, 0), rotation=(0, math.pi / 2, 0)); return bpy.context.object
h1, h2 = hexhead(-0.12), hexhead(0.12); bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.016, depth=0.2, rotation=(0, math.pi / 2, 0)); hd = bpy.context.object
def mat(n, rgb, metal=0.0, rough=0.6):
    m_ = bpy.data.materials.new(n); m_.use_nodes = True; bs = m_.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*rgb, 1); bs.inputs["Metallic"].default_value = metal; bs.inputs["Roughness"].default_value = rough; return m_
blue, steel = mat("cobalt", (0.02, 0.05, 0.45)), mat("steel", (0.5, 0.5, 0.52), 1.0, 0.35)
for o in (h1, h2): o.data.materials.append(blue)
hd.data.materials.append(steel)
bpy.ops.object.select_all(action="DESELECT")
for o in (h1, h2, hd): o.select_set(True)
bpy.context.view_layer.objects.active = hd; bpy.ops.object.join(); db = bpy.context.object; db.name = "dumbbell"; db.rotation_mode = "QUATERNION"
GRIP = float(MO.get("grip_offset_m", 0.075))
def grip_frame():
    Mb = arm.matrix_world @ pb[hand].matrix; R = Mb.to_3x3().normalized()
    u = (R @ mu.Vector((0, 1, 0))).normalized(); w = Mb.translation          # bone head = wrist, Y axis = bone direction
    lr = (arm.matrix_world @ pb["RightArm"].head - arm.matrix_world @ pb["LeftArm"].head).normalized()
    ha = (lr - lr.dot(u) * u).normalized(); third = ha.cross(u).normalized()
    return w + GRIP * u, mu.Matrix((ha, u, third)).transposed().to_quaternion()
# pass 1: evaluate the rig per frame, bake the dumbbell, collect framing points
dg = bpy.context.evaluated_depsgraph_get(); samp = np.arange(0, nv, 23)
frame_pts = []
for f in range(1, NF + 1):
    sc.frame_set(f); g, q = grip_frame(); db.location = g; db.rotation_quaternion = q
    db.keyframe_insert("location", frame=f); db.keyframe_insert("rotation_quaternion", frame=f)
    if f in (1, NF // 4, NF // 2, 3 * NF // 4, NF):
        dg = bpy.context.evaluated_depsgraph_get(); eb = body.evaluated_get(dg); m_ = eb.to_mesh()
        c2 = np.empty(len(m_.vertices) * 3); m_.vertices.foreach_get("co", c2); c2 = c2.reshape(-1, 3)[samp]
        Mw = np.array(eb.matrix_world); frame_pts.append(c2 @ Mw[:3, :3].T + Mw[:3, 3]); eb.to_mesh_clear()
        frame_pts.append(np.array([g + mu.Vector(d_) for d_ in ((0.15, 0, 0), (-0.15, 0, 0), (0, 0, 0.07), (0, 0, -0.07))]))
for fc in (db.animation_data.action.fcurves if hasattr(db.animation_data.action, "fcurves") else []):
    for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
log("dumbbell baked")

# ---- camera: tight side, working-arm side, turned toward the back -----------------------------------------------------
CAMC = RC["camera"]; sc.frame_set(1)
Wm = arm.matrix_world
# body axes from the (processed) landmarks, averaged over the rep: forward = horizontal torso lean, lateral = hips +
# shoulders left-right made perpendicular to forward (the rig's pelvis twists in a staggered stance, so not the rig)
tv_ = np.mean((B[:, 11] + B[:, 12]) / 2 - (B[:, 23] + B[:, 24]) / 2, axis=0); fw = mu.Vector((tv_[0], tv_[1], 0)).normalized()
lv_ = np.mean((B[:, 23] - B[:, 24]) + (B[:, 11] - B[:, 12]), axis=0); lat = mu.Vector((lv_[0], lv_[1], 0)); lat = (lat - lat.dot(fw) * fw).normalized()
side_dir = -lat if side == 12 else lat
yaw, el = math.radians(CAMC.get("yaw_back_deg", 35)), math.radians(CAMC.get("elev_deg", 14))
dh = (side_dir * math.cos(yaw) - fw * math.sin(yaw)).normalized(); dirv = (dh * math.cos(el) + mu.Vector((0, 0, 1)) * math.sin(el)).normalized()
P = np.concatenate(frame_pts); ctr = mu.Vector(((P.min(0) + P.max(0)) / 2).tolist())
rx, ry = map(int, a.res.split("x")) if a.res else map(int, RC["res"]); sc.render.resolution_x, sc.render.resolution_y = rx, ry
camd = bpy.data.cameras.new("c"); CAM = bpy.data.objects.new("c", camd); col.objects.link(CAM); sc.camera = CAM
camd.lens = float(CAMC.get("lens_mm", 50)); camd.sensor_fit = "VERTICAL"; camd.sensor_height = 36.0
fwd = -dirv; right = fwd.cross(mu.Vector((0, 0, 1))).normalized(); up = right.cross(fwd)
rel = P - np.array(ctr); px, py, pz = rel @ np.array(right), rel @ np.array(up), rel @ np.array(fwd)
kf = camd.lens / 18.0; asp = ry / rx
def ext(Dd):
    nx = px / (Dd + pz) * kf * asp; ny = py / (Dd + pz) * kf
    return (nx.max() - nx.min()) / 2, (ny.max() - ny.min()) / 2
lo, hi = 0.5, 30.0
fill_w = float(CAMC.get("fill_w", 0.86)); top_y = 0.24; feet_y = float(CAMC.get("feet_y", 0.86))
for _ in range(60):
    mid = (lo + hi) / 2; wx, wy = ext(mid)
    if wx > fill_w or wy > (feet_y - top_y): lo = mid
    else: hi = mid
Dd = hi; CAM.location = ctr - fwd * Dd; CAM.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
ny = py / (Dd + pz) * kf; nx = px / (Dd + pz) * kf * asp
camd.shift_x = float((nx.max() + nx.min()) / 2) / 2 / asp       # shift is in units of the larger (vertical) dimension
want_low = 1 - 2 * feet_y; camd.shift_y = float((ny.min() - want_low) / 2)
bpy.context.view_layer.update()
pr = np.array([world_to_camera_view(sc, CAM, mu.Vector(p_.tolist()))[:2] for p_ in P[::5]])
framing = {"x_min": round(float(pr[:, 0].min()), 3), "x_max": round(float(pr[:, 0].max()), 3), "y_top": round(float(1 - pr[:, 1].max()), 3),
           "y_feet": round(float(1 - pr[:, 1].min()), 3), "dist_m": round(Dd, 2), "lens": camd.lens}
log(f"camera {framing}")

# ---- per-frame checks (evaluated rig + deformed mesh) -----------------------------------------------------------------
chk = []
for f in range(1, NF + 1):
    sc.frame_set(f); Wm = arm.matrix_world; Pp = lambda b: Wm @ pb[b].head
    lsh, rsh, lwr, rwr, hip = Pp("LeftArm"), Pp("RightArm"), Pp("LeftHand"), Pp("RightHand"), Pp("Hips"); shc = (lsh + rsh) / 2
    lr = (rsh - lsh).normalized(); cross = (rwr - lwr).dot(lr) < 0
    tv = shc - hip; torso = math.degrees(math.atan2(math.hypot(tv.x, tv.y), tv.z))
    g, _ = grip_frame(); dbp = db.matrix_world.translation; d_bone = (dbp - g).length
    sh_, el_, wr_ = (Pp("LeftArm"), Pp("LeftForeArm"), Pp("LeftHand")) if side == 11 else (Pp("RightArm"), Pp("RightForeArm"), Pp("RightHand"))
    elbow = math.degrees((sh_ - el_).angle(wr_ - el_))
    rec = {"f": f, "arm_cross": bool(cross), "torso_deg": round(torso, 2), "db_grip_m": round(d_bone, 4), "elbow_deg": round(elbow, 1)}
    def p2(v_):
        q_ = world_to_camera_view(sc, CAM, v_); return [round(q_.x * rx, 1), round((1 - q_.y) * ry, 1)]
    kn_ = Pp("LeftLeg") if side == 11 else Pp("RightLeg")
    rec["kp2d"] = {"hip": p2(hip), "sh": p2(shc), "sh_w": p2(sh_), "el": p2(el_), "wr": p2(wr_), "kn": p2(kn_), "db": p2(dbp)}
    if f % 3 == 1 or f == NF:
        dg = bpy.context.evaluated_depsgraph_get(); eb = body.evaluated_get(dg); m_ = eb.to_mesh()
        c2 = np.empty(len(m_.vertices) * 3); m_.vertices.foreach_get("co", c2); Mw = np.array(eb.matrix_world)
        c2 = c2.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]; eb.to_mesh_clear()
        ratio = np.linalg.norm(c2[ev[ok_e, 0]] - c2[ev[ok_e, 1]], axis=1) / rest_len[ok_e]
        rec.update({"tear_max_ratio": round(float(ratio.max()), 3), "tear_frac2x": round(float((ratio > 2.0).mean()), 6),
                    "grip_mesh_m": round(float(np.linalg.norm(c2[hand_vi] - np.array(dbp), axis=1).min()), 4)})
        bad = ev[ok_e][ratio > 2.0][:, 0]
        if len(bad):
            u_, n_ = np.unique(dom[bad], return_counts=True); o_ = np.argsort(-n_)[:3]
            rec["tear_where"] = {GN.get(int(u_[i]), "?"): int(n_[i]) for i in o_}
    if garm is not None and (f % 3 == 1 or f == NF):
        from mathutils.bvhtree import BVHTree
        dg = bpy.context.evaluated_depsgraph_get(); eb = body.evaluated_get(dg); mb = eb.to_mesh(); Mw = eb.matrix_world.copy()
        bc = np.empty(len(mb.vertices) * 3); mb.vertices.foreach_get("co", bc); bc = bc.reshape(-1, 3) @ np.array(Mw)[:3, :3].T + np.array(Mw)[:3, 3]
        tri = [tuple(pl.vertices) for pl in mb.polygons]; eb.to_mesh_clear()
        tree = BVHTree.FromPolygons([tuple(v_) for v_ in bc], tri)
        eg = garm.evaluated_get(dg); mg = eg.to_mesh(); gc_ = np.empty(len(mg.vertices) * 3); mg.vertices.foreach_get("co", gc_)
        Mg = np.array(eg.matrix_world); gc_ = gc_.reshape(-1, 3) @ Mg[:3, :3].T + Mg[:3, 3]; eg.to_mesh_clear()
        # clearance = distance from each garment vertex to the deformed body surface (normal-free: the asset's normals
        # are inconsistent). The body folding through the briefs shows up as clearance ~0.
        gap = np.array([tree.find_nearest(mu.Vector(q_))[3] for q_ in gc_.tolist()])
        gr = np.linalg.norm(gc_[gev[g_ok, 0]] - gc_[gev[g_ok, 1]], axis=1) / g_rest[g_ok]
        cmin = float(GC.get("clearance_min_m", 0.002))
        rec.update({"briefs_gap_min_m": round(float(gap.min()), 4), "briefs_poke_frac": round(float((gap < cmin).mean()), 6),
                    "briefs_tear_frac2x": round(float((gr > 2.0).mean()), 6), "briefs_tear_max_ratio": round(float(gr.max()), 3),
                    "briefs_cover_frac": round(float((gap >= cmin).mean()), 6), "briefs_verts": int(len(gc_))})
    chk.append(rec)
json.dump({"frames": chk, "working": hand, "floor_tilt_fixed_deg": round(floor_tilt, 1), "rep": [f0, f1], "raw_torso_deg": [round(float(raw_torso.min()), 1), round(float(raw_torso.max()), 1)],
           "framing": framing, "regions": region_counts, "grip_offset_m": GRIP, "person_to_rig_scale": round(SCALE, 3), "kp_res": [rx, ry]}, open(a.out + "/checks.json", "w"), indent=0)
log("checks done")
if a.checks_only: sys.exit(0)

# ---- materials + lights ------------------------------------------------------------------------------------------------
body_mats = [s.material for s in body.material_slots]
for m_ in body_mats:
    nt = m_.node_tree; bsdf = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"][0]
    for l in list(bsdf.inputs["Base Color"].links): nt.links.remove(l)
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "shorts"; mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs[6].default_value = (0.20, 0.205, 0.215, 1); mix.inputs[7].default_value = (0.20, 0.205, 0.215, 1) if garm is not None else (0.012, 0.012, 0.014, 1)
    nt.links.new(at.outputs["Fac"], mix.inputs[0]); bsdf.inputs["Roughness"].default_value = 0.62
    tex = [n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image and "basecolor" in n.image.name.lower()]
    ad = float(RC.get("albedo_detail", 0.0))
    if tex and ad > 0:   # keep the asset's painted fibre detail: grey x (albedo luminance, contrast-boosted)
        bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(tex[0].outputs["Color"], bw.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["From Min"].default_value = 0.15; mr.inputs["From Max"].default_value = 0.8
        mr.inputs["To Min"].default_value = 1.0 - ad; mr.inputs["To Max"].default_value = 1.0 + ad; nt.links.new(bw.outputs[0], mr.inputs["Value"])
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"; mul.inputs[0].default_value = 1.0
        nt.links.new(mix.outputs[2], mul.inputs[6]); nt.links.new(mr.outputs[0], mul.inputs[7]); nt.links.new(mul.outputs[2], bsdf.inputs["Base Color"])
    else:
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    for nm in [n for n in nt.nodes if n.type == "NORMAL_MAP"]:   # sculpt detail (fibres) from the asset's normal map
        nm.inputs["Strength"].default_value = float(RC.get("normal_strength", 1.0))
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.render.film_transparent = True
wd = bpy.data.worlds.new("w"); sc.world = wd; wd.use_nodes = True; bg = wd.node_tree.nodes["Background"]
hdri = C["asset"]["hdri"]; envn = wd.node_tree.nodes.new("ShaderNodeTexEnvironment"); envn.image = bpy.data.images.load(hdri["file"])
wd.node_tree.links.new(envn.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = float(hdri.get("strength", 0.55))
key_from = (CAM.location - ctr); key_from.z = 0; key_from.normalize()
for off, e in (((key_from * 3.0).to_tuple(), 2.4), ((-fw * 2.5 + mu.Vector((0, 0, 2.5))).to_tuple(), 2.2), ((fw * 3 + mu.Vector((0, 0, 1.5))).to_tuple(), 1.0)):
    Lt = bpy.data.lights.new("l", "AREA"); Lt.energy = e * 150; Lt.size = 3; o = bpy.data.objects.new("l", Lt); col.objects.link(o)
    o.location = ctr + mu.Vector(off) + mu.Vector((0, 0, 1.2)); o.rotation_euler = (ctr - o.location).to_track_quat("-Z", "Y").to_euler()
sc.frame_start, sc.frame_end = 1, NF; sc.render.fps = int(D["fps"]); sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
frames = list(range(1, NF + 1)) if a.frames == "all" else [int(v) for v in a.frames.split(",")]
def render_set(prefix):
    for f in frames:
        sc.frame_set(f); sc.render.filepath = f"{a.out}/{prefix}_{f:04d}.png"; bpy.ops.render.render(write_still=True)
tim = {}
if not a.mask_only:
    sc.cycles.samples = int(RC.get("spp", 12)); sc.cycles.use_denoising = bool(RC.get("denoise", True))
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"; sc.cycles.denoising_prefilter = RC.get("denoise_prefilter", "ACCURATE")   # albedo+normal guided: keeps texture
        sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception as e: log(f"denoise option: {e}")
    sc.view_settings.view_transform = "AgX"
    t = time.time(); render_set("f"); tim["beauty_s"] = round(time.time() - t, 1); log(f"beauty {tim['beauty_s']} s")
if not a.no_mask:
    for m_ in body_mats:
        nt = m_.node_tree; out = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "mregion"; em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(at.outputs["Color"], em.inputs["Color"]); nt.links.new(em.outputs[0], out.inputs["Surface"])
    ho = bpy.data.materials.new("hold"); ho.use_nodes = True; hn = ho.node_tree.nodes.new("ShaderNodeHoldout")
    ho.node_tree.links.new(hn.outputs[0], [n for n in ho.node_tree.nodes if n.type == "OUTPUT_MATERIAL"][0].inputs["Surface"])
    db.data.materials.clear(); db.data.materials.append(ho)
    if garm is not None:
        gw = bpy.data.materials.new("briefs_mask"); gw.use_nodes = True; ge = gw.node_tree.nodes.new("ShaderNodeEmission"); ge.inputs["Color"].default_value = (1, 1, 1, 1)
        gw.node_tree.links.new(ge.outputs[0], [n for n in gw.node_tree.nodes if n.type == "OUTPUT_MATERIAL"][0].inputs["Surface"])
        garm.data.materials.clear(); garm.data.materials.append(gw)
    sc.cycles.samples = int(RC.get("mask_spp", 4)); sc.cycles.use_denoising = False; sc.cycles.max_bounces = 0
    sc.view_settings.view_transform = "Standard"; bg.inputs["Strength"].default_value = 0.0
    for o in bpy.data.objects:
        if o.type == "LIGHT": o.hide_render = True
    t = time.time(); render_set("m"); tim["mask_s"] = round(time.time() - t, 1); log(f"mask {tim['mask_s']} s")
json.dump({**tim, "frames": len(frames), "res": [rx, ry]}, open(a.out + "/render_time.json", "w"))
print("RENDER_DONE", json.dumps(tim))
