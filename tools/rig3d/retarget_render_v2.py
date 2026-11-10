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
import sys, os, json, math, time, argparse, numpy as np, bpy, mathutils as mu
from bpy_extras.object_utils import world_to_camera_view
T0 = time.time()
def log(m): print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="all"); ap.add_argument("--checks-only", action="store_true"); ap.add_argument("--no-mask", action="store_true")
ap.add_argument("--mask-only", action="store_true"); ap.add_argument("--res", default=None)
ap.add_argument("--skip-existing", action="store_true", help="skip frames whose f_/m_ png already exists (crash recovery)")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
C = json.load(open(a.cfg)); MO, RC = C["motion"], C["render"]
if a.skip_existing: RC["skip_existing_frames"] = True
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
# legs (v4). legacy "source": ankles/heels/toes locked to frame 1 of the clip, knees from MediaPipe -> long staggered
# stride, rear heel up on its toes, knees drifting = a running-stride look. "planted": both feet flat on the floor at the
# rig's rest foot pitch, fore-aft stagger scaled by stance_scale toward the hips, constant all rep; knees are solved per
# frame (2-bone, rig bone lengths, bending toward the toes) from the rig's actual hip joints, so feet do not slide.
LEGS = MO.get("legs", "source")
if LEGS == "planted":
    tv0 = np.mean((B[:, 11] + B[:, 12]) / 2 - (B[:, 23] + B[:, 24]) / 2, axis=0); fwd_h = np.array([tv0[0], tv0[1], 0.0]); fwd_h /= np.linalg.norm(fwd_h)
    hg = np.median((B[:, 23] + B[:, 24]) / 2, axis=0); ss = float(MO.get("stance_scale", 0.5)); sw = float(MO.get("stance_width_m", 0.0))
    lat_h = np.cross([0.0, 0.0, 1.0], fwd_h)
    for aj, tj, hj, s_ in ((27, 31, 29, "Left"), (28, 32, 30, "Right")):
        a0 = B[0, aj].copy(); dv = a0 - hg; fa = float(dv @ fwd_h); la = float(dv @ lat_h)
        if sw > 0: la = math.copysign(sw / 2, la if abs(la) > 1e-6 else (1 if s_ == "Left" else -1))
        an = hg + ss * fa * fwd_h + la * lat_h; an[2] = (Wr @ bl[s_ + "Foot"].head_local).z
        fr = Wr @ bl[s_ + "Foot"].head_local; tr = Wr @ bl[s_ + "ToeBase"].head_local; hl = math.hypot(tr.x - fr.x, tr.y - fr.y)
        td = B[0, tj] - a0; td[2] = 0; td = td / (np.linalg.norm(td) + 1e-9) if np.linalg.norm(td) > 1e-6 else fwd_h
        td = td * 0.5 + fwd_h * 0.5; td /= np.linalg.norm(td)                # toes mostly forward (no splayed / turned-in foot)
        to = an + hl * td; to[2] = tr.z
        B[:, aj] = an; B[:, tj] = to; B[:, hj] = an - 0.05 * td
    if MO.get("lock_hips", False):   # v5: static stance = static pelvis; the clip's hip sway moved the knees 12 deg per rep
        hm_ = (B[:, 23] + B[:, 24]) / 2; dh_ = np.median(hm_, axis=0) - hm_
        upj_ = [j for j in range(33) if j not in (27, 28, 29, 30, 31, 32)]; B[:, upj_] += dh_[:, None, :]
    kt = math.radians(float(MO.get("knee_target_deg", 155)))      # soft knee: raise hips + upper body (rigid) to get it
    rq = []
    for s_, hj, aj in (("Left", 23, 27), ("Right", 24, 28)):
        Lt, Ls = rl(s_ + "UpLeg", s_ + "Leg"), rl(s_ + "Leg", s_ + "Foot"); dt_ = math.sqrt(Lt * Lt + Ls * Ls - 2 * Lt * Ls * math.cos(kt))
        hh = np.median(np.hypot(B[:, hj, 0] - B[0, aj, 0], B[:, hj, 1] - B[0, aj, 1]))
        rq.append(B[0, aj, 2] + math.sqrt(max(dt_ * dt_ - hh * hh, 0.0)) - float(np.median(B[:, hj, 2])))
    raise_z = float(np.mean(rq)) if MO.get("raise_hips", True) else 0.0
    up_j = [j for j in range(33) if j not in (27, 28, 29, 30, 31, 32)]; B[:, up_j, 2] += raise_z
    log(f"legs planted: hips raised {raise_z:+.3f} m for knee {math.degrees(kt):.0f}, fwd {fwd_h.round(2).tolist()}")
    log(f"legs planted: stance_scale {ss}, ankles {B[0, 27].round(3).tolist()} / {B[0, 28].round(3).tolist()}")
def empty(n):
    e = bpy.data.objects.new(n, None); col.objects.link(e); return e
names = {"hipmid": lambda f: (B[f, 23] + B[f, 24]) / 2, "shmid": lambda f: (B[f, 11] + B[f, 12]) / 2, "nose": lambda f: (B[f, 7] + B[f, 8]) / 2}
# neck/head (v4). legacy "ears_track": neck AND Head damped-tracked the ear midpoint (landmarks 7/8, mislabelled "nose").
# The Head bone starts at the top of the neck, almost level with the ears, and its rest axis leans 34 deg forward, so
# aiming it at the ears folded the head down toward the chest (neck flexion 30+ deg, QC neck_in_line_with_spine).
# "spine_follow": neither bone is aimed at landmarks; the neck inherits the chest bone (Spine) orientation, so it
# continues the spine line by construction (0 roll, 0 yaw, no face-landmark jitter), tipped back by a fixed
# gaze_ext_deg about its lateral axis (natural slight forward gaze); the Head keeps its rest relation to the neck.
NECK_MODE = MO.get("neck_mode", "ears_track")
# spine aim (v4): legacy "shmid" damped-tracked the shoulder midpoint from every spine bone; the top bone (Spine) starts
# almost AT shoulder height, so its aim was ill-conditioned and swung 7-65 deg off the thoracic line frame to frame
# (the chest the neck/head hang from). "far": the same torso direction, but a point 3 torso lengths out.
if MO.get("spine_aim", "shmid") == "far":
    names["shfar"] = lambda f: (B[f, 23] + B[f, 24]) / 2 + 3.0 * ((B[f, 11] + B[f, 12]) / 2 - (B[f, 23] + B[f, 24]) / 2)
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
for bn in ("Hips", "Spine02", "Spine01", "Spine"): track(bn, "shfar" if "shfar" in E else "shmid")
if MO.get("lock_hips", False) and "shfar" in E:     # v5: the pelvis aims at the rep-median torso direction (static), the
    _hv = np.median(np.stack([names["shfar"](f) - names["hipmid"](f) for f in range(NF)]), axis=0)   # spine takes the rest
    eh = bpy.data.objects.new("T_hipaim", None); col.objects.link(eh)
    for f in range(NF): eh.location = mu.Vector((names["hipmid"](f) + _hv).tolist()); eh.keyframe_insert("location", frame=f + 1)
    pb["Hips"].constraints[-1].target = eh
for bn in ("Spine02", "Spine01", "Spine"):          # < 1 spreads the bend over the chain (less skin stretch at one joint)
    pb[bn].constraints[-1].influence = float(MO.get("spine_influence", 1.0))
if NECK_MODE == "spine_follow":
    pb["neck"].rotation_quaternion = mu.Quaternion((1, 0, 0), -math.radians(float(MO.get("gaze_ext_deg", 8.0))))   # local X = lateral, - = extension
else:
    track("neck", "nose"); track("Head", "nose")
if MO.get("pelvis_yaw", False):
    # v5: face the pelvis along the torso's horizontal forward BEFORE the hinge track. Without it the minimal-rotation
    # DAMPED_TRACK tipped the (rig-forward) pelvis diagonally toward the person's lean = a 9 cm sideways pelvis roll,
    # one hip joint high enough to lock that leg straight (175 deg) whatever the stance.
    tvy = np.mean((B[:, 11] + B[:, 12]) / 2 - (B[:, 23] + B[:, 24]) / 2, axis=0); fy_ = np.array([tvy[0], tvy[1]]); fy_ /= np.linalg.norm(fy_)
    yaw_ = math.atan2(0.0 * fy_[1] - (-1.0) * fy_[0], 0.0 * fy_[0] + (-1.0) * fy_[1])     # rig forward (0,-1) -> fy_, about +Z
    pb["Hips"].rotation_quaternion = mu.Quaternion((0, 1, 0), yaw_); log(f"pelvis yaw {math.degrees(yaw_):.1f} deg")
WRIST_NEUTRAL = MO.get("grip_mode", "shoulder_line") in ("palm", "palm_neutral") and MO.get("wrist_neutral", True)
for s, p in (("Left", "l"), ("Right", "r")):
    track(s + "Arm", p + "el"); track(s + "ForeArm", p + "wr")
    if not (WRIST_NEUTRAL and s + "Hand" == ("Left" if side == 11 else "Right") + "Hand"):
        track(s + "Hand", p + "ix")   # v5: the working hand keeps its rest line with the forearm (neutral wrist, no 77 deg wrist curl)
    track(s + "UpLeg", p + "kn"); track(s + "Leg", p + "ank"); track(s + "Foot", p + "ft")
hand = ("Left" if side == 11 else "Right") + "Hand"
if LEGS == "planted":       # per-frame 2-bone knee solve from the evaluated hip joints (hips/spine do not depend on knees)
    # refine the hip raise on the evaluated rig (rig hip joints sit off the landmark hips once the pelvis hinges)
    upn = [n for n in names if n not in ("lank", "rank", "lft", "rft", "lkn", "rkn")]
    for it in range(4):
        req = []
        for s_, aj in (("Left", 27), ("Right", 28)):
            Lt, Ls = rl(s_ + "UpLeg", s_ + "Leg"), rl(s_ + "Leg", s_ + "Foot"); dt_ = math.sqrt(Lt * Lt + Ls * Ls - 2 * Lt * Ls * math.cos(kt))
            for f in (1, NF // 2, NF):
                sc.frame_set(f); H = arm.matrix_world @ pb[s_ + "UpLeg"].head; hh = math.hypot(H.x - B[0, aj, 0], H.y - B[0, aj, 1])
                req.append(B[0, aj, 2] + math.sqrt(max(dt_ * dt_ - hh * hh, 0.0)) - H.z)
        dz = float(np.mean(req))
        if abs(dz) < 0.002: break
        B[:, up_j, 2] += dz; raise_z += dz
        for f in range(NF):
            for n in upn: E[n].location = mu.Vector(names[n](f).tolist()); E[n].keyframe_insert("location", frame=f + 1)
    log(f"legs: rig-refined hip raise {raise_z:+.3f} m")
    if MO.get("balance_knees", False):
        # v5: one shared hip height cannot give both legs the same knee (the rear ankle sits farther from its hip), so
        # each ankle (+ heel/toe) slides along the fore-aft axis until ITS hip-ankle distance gives knee_target_deg
        # on the evaluated rig (lateral width kept). Removes the locked-straight rear leg / standing look.
        sc.frame_set(NF // 2 + 1); moved = []
        for s_, aj, tj, hj in (("Left", 27, 31, 29), ("Right", 28, 32, 30)):
            Lt, Ls = rl(s_ + "UpLeg", s_ + "Leg"), rl(s_ + "Leg", s_ + "Foot"); dt_ = math.sqrt(Lt * Lt + Ls * Ls - 2 * Lt * Ls * math.cos(kt))
            H = np.array(arm.matrix_world @ pb[s_ + "UpLeg"].head); A = B[0, aj].copy(); hv = A - H; hv[2] = 0
            fa, la = float(hv @ fwd_h), float(hv @ lat_h); dz = float(H[2] - A[2]); hreq = math.sqrt(max(dt_ * dt_ - dz * dz, 0.0))
            sgn_ = 1.0 if MO.get("stance_mode", "source") == "feet_forward" else (fa if abs(fa) > 1e-6 else 1.0)   # v6: hips back over the feet
            fa_new = math.copysign(math.sqrt(max(hreq * hreq - la * la, 1e-4)), sgn_)
            dv = (fa_new - fa) * fwd_h
            for j in (aj, tj, hj): B[:, j] += dv
            moved.append(round(fa_new - fa, 3))
        for f in range(NF):
            for n in ("lank", "rank", "lft", "rft"): E[n].location = mu.Vector(names[n](f).tolist()); E[n].keyframe_insert("location", frame=f + 1)
        log(f"legs: knees balanced, ankles moved fore-aft {moved} m")
    for s_, p_, aj in (("Left", "l", 27), ("Right", "r", 28)):
        Lt, Ls = rl(s_ + "UpLeg", s_ + "Leg"), rl(s_ + "Leg", s_ + "Foot"); A = mu.Vector(B[0, aj].tolist())
        pole = mu.Vector((B[0, 31 if aj == 27 else 32] - B[0, aj]).tolist()); pole.z = 0; pole.normalize()
        for f in range(NF):
            sc.frame_set(f + 1); H = arm.matrix_world @ pb[s_ + "UpLeg"].head; dv = A - H; d = min(dv.length, 0.999 * (Lt + Ls)); u_ = dv.normalized()
            x_ = (Lt * Lt - Ls * Ls + d * d) / (2 * d); h_ = math.sqrt(max(Lt * Lt - x_ * x_, 0.0)); pp = (pole - pole.dot(u_) * u_).normalized()
            E[p_ + "kn"].location = H + u_ * x_ + pp * h_; E[p_ + "kn"].keyframe_insert("location", frame=f + 1)
if LEGS == "planted" and MO.get("support_hand", "") == "front_thigh":
    # v5: the free hand rests on the front thigh just above the knee (classic split-stance row brace) instead of
    # floating ~30 cm in front of it after the stance/hip changes. Elbow from a 2-bone solve (rig arm lengths), out to the side.
    sp_ = "Left" if side == 12 else "Right"; pp_ = sp_[0].lower()
    fl = max((("Left", 27), ("Right", 28)), key=lambda t: float(B[0, t[1]] @ np.append(fwd_h[:2], 0.0)))[0]
    if MO.get("stance_mode", "source") == "feet_forward": fl = sp_                   # v6: both feet ahead of the hips: brace on the same-side thigh
    La, Lf = rl(sp_ + "Arm", sp_ + "ForeArm"), rl(sp_ + "ForeArm", sp_ + "Hand"); latv = mu.Vector(lat_h.tolist()) * (1 if sp_ == "Left" else -1)
    if float(np.dot(np.array(arm.matrix_world @ pb[sp_ + "Arm"].head)[:2] - np.array(arm.matrix_world @ pb[("Right" if sp_ == "Left" else "Left") + "Arm"].head)[:2], lat_h[:2])) * (1 if sp_ == "Left" else -1) < 0: latv = -latv
    for f in range(NF):
        sc.frame_set(f + 1); Hh = arm.matrix_world @ pb[fl + "UpLeg"].head; Kk = E[fl[0].lower() + "kn"].location.copy()
        td_ = (Kk - Hh).normalized(); upn_ = mu.Vector((0, 0, 1)); upn_ = (upn_ - upn_.dot(td_) * td_).normalized()
        Sh = arm.matrix_world @ pb[sp_ + "Arm"].head
        lift = float(MO.get("support_hand_lift_m", 0.075))
        t0 = float(MO.get("support_hand_thigh_t0", 0.35)); t1 = float(MO.get("support_hand_thigh_frac", 0.75))
        if t1 < t0: t0, t1 = t1, t0
        lat_off = latv.normalized() * float(MO.get("support_hand_lateral_m", 0.0))
        back_off = -mu.Vector(fwd_h.tolist()) * float(MO.get("support_hand_back_m", 0.0))
        reach_s = float(MO.get("support_hand_reach_scale", 0.97))
        cand = [Hh + (Kk - Hh) * t_ + upn_ * lift + lat_off + back_off for t_ in np.linspace(t0, t1, 9)]
        if MO.get("support_hand_shoulder_reach", True):
            Wt = cand[-1]   # commit to the distal target; the support Shoulder then reaches so the arm can land on it
        else:
            Wt = min(cand, key=lambda c_: max((c_ - Sh).length - reach_s * (La + Lf), 0.0) - 0.05 * (c_ - Hh).length)
        Sh = arm.matrix_world @ pb[sp_ + "Arm"].head
        if MO.get("support_hand_shoulder_reach", True) and (sp_ + "Shoulder") in pb:
            sb = pb[sp_ + "Shoulder"]; base_q = mu.Quaternion((1, 0, 0, 0))
            toward = np.array(Wt - Sh); toward = toward / max(float(np.linalg.norm(toward)), 1e-9)
            arm0 = np.array(Sh); best_axis, best_score = (1, 0, 0), -1e9
            for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                sb.rotation_quaternion = mu.Quaternion(axis, math.radians(6.0)); bpy.context.view_layer.update()
                arm1 = np.array(arm.matrix_world @ pb[sp_ + "Arm"].head); score = float((arm1 - arm0) @ toward)
                if score > best_score: best_axis, best_score = axis, score
            sb.rotation_quaternion = base_q
            if best_score < 0: best_axis = tuple(-x for x in best_axis)
            lo, hi = 0.0, float(MO.get("support_hand_shoulder_reach_max_deg", 50.0))
            if (Wt - Sh).length <= 0.985 * (La + Lf): hi = 0.0
            else:
                for _ in range(9):
                    mid = 0.5 * (lo + hi); sb.rotation_quaternion = mu.Quaternion(best_axis, math.radians(mid)); bpy.context.view_layer.update()
                    if (Wt - (arm.matrix_world @ pb[sp_ + "Arm"].head)).length <= 0.985 * (La + Lf): hi = mid
                    else: lo = mid
            sb.rotation_quaternion = mu.Quaternion(best_axis, math.radians(hi))
            bpy.context.view_layer.update(); Sh2 = arm.matrix_world @ pb[sp_ + "Arm"].head
            if (Wt - Sh2).length > 0.985 * (La + Lf):
                sb.rotation_quaternion = base_q   # unused max wind-up jams the support arm into the ribs
            sb.keyframe_insert("rotation_quaternion", frame=f + 1)
            bpy.context.view_layer.update(); Sh = arm.matrix_world @ pb[sp_ + "Arm"].head
            if (Wt - Sh).length > 0.985 * (La + Lf):
                walk = [Hh + (Kk - Hh) * t_ + upn_ * lift + lat_off + back_off for t_ in np.linspace(0.35, t1, 13)]
                landed = False
                for Wt2 in reversed(walk):
                    if (Wt2 - Sh).length <= 0.985 * (La + Lf):
                        Wt = Wt2; landed = True; break
                if not landed: Wt = walk[0]
        dv = Wt - Sh; u_ = dv.normalized(); d_wr = min(max(dv.length, 1e-4), 0.999 * (La + Lf)); d = d_wr
        x_ = (La * La - Lf * Lf + d * d) / (2 * d); h_ = math.sqrt(max(La * La - x_ * x_, 0.0))
        if MO.get("support_elbow_mode", "lateral") == "back_out":                 # v6: elbow points back/out (no hyperextension)
            pv_ = latv * float(MO.get("support_elbow_out", 0.5)) - mu.Vector(fwd_h.tolist()) * float(MO.get("support_elbow_back", 1.0))
        else: pv_ = latv                                                            # v5 legacy
        pole = (pv_ - pv_.dot(u_) * u_).normalized()
        E[pp_ + "el"].location = Sh + u_ * x_ + pole * h_; E[pp_ + "wr"].location = Sh + u_ * d_wr; E[pp_ + "ix"].location = Sh + u_ * d_wr + ((u_ * 0.6 + td_ * 0.4).normalized() if MO.get("support_wrist_neutral") else td_) * 0.09   # v6: wrist ~neutral
        for n_ in ("el", "wr", "ix"): E[pp_ + n_].keyframe_insert("location", frame=f + 1)
    log(f"support hand on the front ({fl}) thigh")
if MO.get("elbow_path", "source") == "tuck_torso":
    # Shared construction: working elbow rebuilt on the 2-bone circle with the pole toward Spine01, so
    # MediaPipe flare cannot occur — the elbow tracks close to the torso toward the hip.
    wh_ = "Left" if side == 11 else "Right"; ppw = wh_[0].lower()
    La_w, Lf_w = rl(wh_ + "Arm", wh_ + "ForeArm"), rl(wh_ + "ForeArm", wh_ + "Hand")
    min_torso = float(MO.get("elbow_torso_min_m", 0.25))
    for f in range(NF):
        sc.frame_set(f + 1); Sh = arm.matrix_world @ pb[wh_ + "Arm"].head; Wt = E[ppw + "wr"].location.copy()
        dv = Wt - Sh; d = min(max(dv.length, 1e-4), 0.995 * (La_w + Lf_w)); u_ = dv.normalized()
        x_ = (La_w * La_w - Lf_w * Lf_w + d * d) / (2 * d); h_ = math.sqrt(max(La_w * La_w - x_ * x_, 0.0))
        mid = Sh + u_ * x_; Sp = arm.matrix_world @ pb["Spine01"].head
        raw = Sp - mid; pole_tuck = raw - raw.dot(u_) * u_
        if pole_tuck.length < 1e-5: pole_tuck = mu.Vector((0.0, 0.0, 1.0))
        pole_tuck.normalize()
        El_src = E[ppw + "el"].location.copy(); src_pole = El_src - mid; src_pole = src_pole - src_pole.dot(u_) * u_
        if src_pole.length < 1e-5: src_pole = pole_tuck
        src_pole.normalize()
        # walk source -> full-tuck and keep the most tucked elbow that stays outside the torso
        best = mid + src_pole * h_
        for a_ in np.linspace(0.0, 1.0, 12):
            p_ = (src_pole * (1.0 - a_) + pole_tuck * a_); p_ = p_ - p_.dot(u_) * u_
            if p_.length < 1e-5: continue
            p_.normalize(); El = mid + p_ * h_
            if (El - Sp).length >= min_torso: best = El
            else: break
        E[ppw + "el"].location = best; E[ppw + "el"].keyframe_insert("location", frame=f + 1)
    log(f"elbow_path tuck_torso on {wh_}")
if MO.get("scapula_retract", False) and (("LeftShoulder" if side == 11 else "RightShoulder") in pb):
    # Shared construction: rotate the working Shoulder so the arm-head moves toward Spine01 in XY,
    # proportional to working-arm flexion (retracts at the peak, rest at the start). Axis chosen by a
    # 6 deg probe so we never guess Mixamo rest Euler axes.
    wh_ = "Left" if side == 11 else "Right"; sb = pb[wh_ + "Shoulder"]; ppw = wh_[0].lower()
    sc.frame_set(1); bpy.context.view_layer.update()
    arm0 = np.array(arm.matrix_world @ pb[wh_ + "Arm"].head); sp0 = np.array(arm.matrix_world @ pb["Spine01"].head)
    toward = sp0 - arm0; toward[2] = 0.0; toward = toward / max(float(np.linalg.norm(toward)), 1e-9)
    base_q = sb.rotation_quaternion.copy(); best_axis, best_score = (0, 0, 1), -1e9
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        sb.rotation_quaternion = mu.Quaternion(axis, math.radians(6.0)) @ base_q; bpy.context.view_layer.update()
        arm1 = np.array(arm.matrix_world @ pb[wh_ + "Arm"].head); sb.rotation_quaternion = base_q
        score = float((arm1 - arm0) @ toward)
        if score > best_score: best_axis, best_score = axis, score
    if best_score < 0: best_axis = tuple(-x for x in best_axis); best_score = -best_score
    angs = []
    for f in range(NF):
        sc.frame_set(f + 1); S = arm.matrix_world @ pb[wh_ + "Arm"].head; El = E[ppw + "el"].location; Wr = E[ppw + "wr"].location
        angs.append(math.degrees((S - El).angle(Wr - El)) if (S - El).length > 1e-6 and (Wr - El).length > 1e-6 else 180.0)
    amin, amax = min(angs), max(angs); max_deg = float(MO.get("scapula_retract_deg", 9.0))
    for f, ang in enumerate(angs):
        amt = 0.0 if amax - amin < 1e-3 else (amax - ang) / (amax - amin)
        sb.rotation_quaternion = mu.Quaternion(best_axis, math.radians(max_deg * amt)); sb.keyframe_insert("rotation_quaternion", frame=f + 1)
    log(f"scapula_retract {wh_} axis {best_axis} score {best_score:.3f} max {max_deg:.1f} deg")
if LEGS == "planted" and MO.get("foot_level", True):
    # Shared construction: DAMPED_TRACK aims the foot Y at the toes (unconstrained roll = sickle). LOCKED_TRACK
    # the sole-up axis (whichever of local X/Z is closest to world +Z) at a world-up empty, lock Y.
    up_e = empty("T_world_up"); up_e.location = mu.Vector((0.0, 0.0, 80.0))
    sc.frame_set(1); bpy.context.view_layer.update(); used = None
    for s_ in ("Left", "Right"):
        R = np.array((arm.matrix_world @ pb[s_ + "Foot"].matrix).to_3x3()); R = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
        upness = [abs(R[2, 0]), 0.0, abs(R[2, 2])]; ai = int(np.argmax(upness))
        if upness[ai] < 0.15: continue
        names_pos, names_neg = ("TRACK_X", "TRACK_Y", "TRACK_Z"), ("TRACK_NEGATIVE_X", "TRACK_NEGATIVE_Y", "TRACK_NEGATIVE_Z")
        track_ax = names_pos[ai] if R[2, ai] >= 0 else names_neg[ai]
        c = pb[s_ + "Foot"].constraints.new("LOCKED_TRACK"); c.target = up_e; c.track_axis = track_ax; c.lock_axis = "LOCK_Y"; used = track_ax
    log(f"foot_level LOCKED_TRACK {used} lock Y" if used else "foot_level: no clear sole-up axis, skip")
log("rig constrained")

# ---- rest geometry: muscle regions, shorts, hand vertices, edges ----------------------------------------------------
me = body.data; nv = len(me.vertices)
co = np.empty(nv * 3); me.vertices.foreach_get("co", co); nr = np.empty(nv * 3); me.vertices.foreach_get("normal", nr)
M4 = np.array(body.matrix_world); co = co.reshape(-1, 3) @ M4[:3, :3].T + M4[:3, 3]; nr = nr.reshape(-1, 3) @ M4[:3, :3].T
nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
x, y, z = co[:, 0], co[:, 1], co[:, 2]; ax_ = np.abs(x)
shorts = (z >= 0.80) & (z <= 1.02) & (ax_ <= 0.22)
vs = me.color_attributes.new("shorts", "FLOAT_COLOR", "POINT"); s4 = np.repeat(shorts.astype(np.float32)[:, None], 4, 1); s4[:, 3] = 1; vs.data.foreach_set("color", s4.ravel())
gi = body.vertex_groups[hand].index; hand_vi = []; dom = np.full(nv, -1)
for v in me.vertices:
    best = (0.0, -1)
    for g in v.groups:
        if g.weight > best[0]: best = (g.weight, g.group)
        if g.group == gi and g.weight > 0.5: hand_vi.append(v.index)
    dom[v.index] = best[1]
hand_vi = np.array(hand_vi); GN = {g.index: g.name for g in body.vertex_groups}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import muscle_regions as MREG
ev0 = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ev0)
ADJ = MREG.adjacency(nv, ev0.reshape(-1, 2))
if RC.get("crease_smooth", True):
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _cdir = RC.get("weight_cache_dir") or os.path.join(_root, "output/_rig3d/cache")
    _gn = ("Spine", "Spine01", "Spine02", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm")
    did, hit, cpath = MREG.cached_smooth_vertex_groups(
        body, _gn, ADJ,
        factor=float(RC.get("crease_weight_smooth", 0.5)),
        repeat=int(RC.get("crease_weight_smooth_iter", 6)),
        cache_dir=_cdir, glb_path=C["asset"]["glb"])
    for v in me.vertices:
        best = (0.0, -1)
        for g in v.groups:
            if g.weight > best[0]: best = (g.weight, g.group)
        dom[v.index] = best[1]
    log(f"crease weight-smooth {did} cache={hit} {cpath}")
EXCLUDE = MREG.exclude_mask(dom, GN)
PRIOR = np.zeros(nv, dtype=bool)
for _pn in ("Spine", "Spine01", "Spine02", "LeftShoulder", "RightShoulder"):
    if _pn in body.vertex_groups:
        _gi = body.vertex_groups[_pn].index
        for v in me.vertices:
            for g in v.groups:
                if g.group == _gi and g.weight > 0.20:
                    PRIOR[v.index] = True
def _bone_heads(posed=False):
    out = {}
    for n_ in ("Hips", "Spine", "Spine01", "Spine02", "Neck", "neck", "Head", "head_end", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"):
        if posed and n_ in pb: out[n_] = np.array(arm.matrix_world @ pb[n_].head)
        elif n_ in arm.data.bones: out[n_] = np.array(arm.matrix_world @ arm.data.bones[n_].head_local)
    return out
_peak_dilate = int(RC.get("glow_peak_dilate", 4))
reg, rinfo = MREG.assign(co, nr, _bone_heads(False), ADJ, shorts, EXCLUDE, flex=0.0, peak_dilate=_peak_dilate, prior=PRIOR)
REST_REG = reg.copy()
MREG.write_mregion(me, reg)
lats = reg[:, 0] > 0.5; rhomb = reg[:, 1] > 0.5; mid_trap = reg[:, 2] > 0.5
# palm axis (v5): the across-palm direction (index knuckle -> little finger) in the HAND bone's local frame = the
# middle principal axis of the rest hand-mesh vertices (largest = fingers, smallest = palm normal). On this GLB the
# bone X/Z axes sit ~45 deg off the palm, so the legacy handle (shoulder line projected off the hand axis) was tilted.
def palm_local(bn):
    gi_ = body.vertex_groups[bn].index; vi_ = [v.index for v in me.vertices if any(g.group == gi_ and g.weight > 0.8 for g in v.groups)]
    P_ = co[vi_] - co[vi_].mean(0); w_, V_ = np.linalg.eigh(np.cov(P_.T)); R_ = np.array((arm.matrix_world @ arm.data.bones[bn].matrix_local).to_3x3().normalized())
    a_ = R_.T @ V_[:, 1]; a_[1] = 0.0; return a_ / np.linalg.norm(a_)           # perpendicular to the bone (Y) axis
PALM = {bn: palm_local(bn) for bn in ("LeftHand", "RightHand")}
# power grip (v6, tools/rig3d/hand_grip.py): the rig has no finger bones, so the fist is a geometric shape key on the
# working hand (fingers bent around the handle axis, thumb flexed to oppose). Its refined across-palm axis replaces the
# PCA axis for the handle + neutral-grip twist; the handle centre sits in the palm instead of on the bone line.
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); import hand_grip as HGR
GW = MO.get("grip_wrap", {}) or {}; HG = None
_gi = body.vertex_groups[hand].index; _W = np.zeros(nv)
for v in me.vertices:
    for g in v.groups:
        if g.group == _gi: _W[v.index] = g.weight
HIDX = np.nonzero(_W > 0.5)[0]; _Mh = np.array(arm.matrix_world @ arm.data.bones[hand].matrix_local); _Mh[:3, :3] /= np.linalg.norm(_Mh[:3, :3], axis=0)
HG = HGR.build(co, HIDX, _Mh, hand.startswith("Left"), r=float(GW.get("handle_r_m", 0.016)), clear=float(GW.get("clear_m", 0.002)),
               max_wrap_deg=float(GW.get("max_wrap_deg", 250)), mcp_back=float(GW.get("mcp_back_m", 0.015)))
if GW.get("enabled"):
    PALM[hand] = np.array(HG["handle_local"]["axis"]); log(f"grip wrap: {HG['info']}")
GRIP_MODE = MO.get("grip_mode", "shoulder_line")
tvg = np.mean((B[:, 11] + B[:, 12]) / 2 - (B[:, 23] + B[:, 24]) / 2, axis=0); FWH = mu.Vector((tvg[0], tvg[1], 0.0)).normalized()
def palm_world(bn):
    R_ = (arm.matrix_world @ pb[bn].matrix).to_3x3().normalized(); return (R_ @ mu.Vector(PALM[bn].tolist())).normalized(), (R_ @ mu.Vector((0, 1, 0))).normalized()
if GRIP_MODE == "palm_neutral":
    # neutral grip: twist the working forearm (share) + hand about their own axes per frame so the across-palm axis
    # lines up with the body's horizontal forward axis (palm toward the torso, handle front-to-back)
    fa_b = hand.replace("Hand", "ForeArm"); share = float(MO.get("grip_twist_forearm_share", 0.6)); TH = np.zeros(NF)
    def _resid(f):
        sc.frame_set(f); pa, u_ = palm_world(hand); want = FWH - FWH.dot(u_) * u_
        if want.length < 1e-3: return 0.0
        want.normalize()
        if pa.dot(want) < 0: want = -want                                         # the handle has no direction: nearest of +-
        return math.atan2(pa.cross(want).dot(u_), pa.dot(want))
    # a twist about the bone's own axis survives DAMPED_TRACK, so the palm turns about the hand axis by ~theta; pass 1
    # with no twist keys (frame_set would otherwise re-apply the keys), then 2 residual passes on the keyed rig
    for ps in range(3):
        for f in range(1, NF + 1): TH[f - 1] += _resid(f)
        if ps == 0:
            TH = np.unwrap(2 * TH) / 2                                             # the handle has no direction: period pi, continuous
            TH -= np.pi * np.round(np.median(TH) / np.pi)                          # branch nearest zero twist
            TH = gsmooth(TH[:, None], 1.5)[:, 0]                                   # radians, smooth over the rep
        for f in range(1, NF + 1):
            pb[fa_b].rotation_quaternion = mu.Quaternion((0, 1, 0), share * TH[f - 1]); pb[hand].rotation_quaternion = mu.Quaternion((0, 1, 0), (1 - share) * TH[f - 1])
            pb[fa_b].keyframe_insert("rotation_quaternion", frame=f); pb[hand].keyframe_insert("rotation_quaternion", frame=f)
    TW = np.degrees(TH).tolist()
    region_counts_twist = [round(min(TW), 1), round(max(TW), 1)]; log(f"neutral grip twist {region_counts_twist} deg (forearm share {share})")
ev = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ev); ev = ev.reshape(-1, 2)
rest_len = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1); ok_e = rest_len > 1e-4; MED_E = float(np.median(rest_len[ok_e]))
region_counts = {"lats": int(lats.sum()), "rhomboids": int(rhomb.sum()), "mid_trap": int(mid_trap.sum()), "rear_delt": int(rinfo.get("rear_delt", 0)), "shorts": int(shorts.sum()), "hand_verts": int(len(hand_vi))}
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

if GW.get("enabled"):      # after the garment copy (the briefs never carry the fist key)
    if not me.shape_keys: body.shape_key_add(name="Basis", from_mix=False)
    sk = body.shape_key_add(name="fist", from_mix=False); lc = np.empty(nv * 3); sk.data.foreach_get("co", lc); lc = lc.reshape(-1, 3)
    Minv = np.linalg.inv(M4); lc[HIDX] = HG["new_world"] @ Minv[:3, :3].T + Minv[:3, 3]; sk.data.foreach_set("co", lc.ravel()); sk.value = 1.0
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
bpy.context.view_layer.objects.active = hd; bpy.ops.object.join(); db = bpy.context.object; db.name = "dumbbell"
# v6 bug fix: the join keeps the handle cylinder's (0, 90 deg, 0) object rotation, so the long axis is LOCAL Z, while
# grip_frame() puts the intended handle direction on local X -> through v5 the handle actually pointed along the
# third axis (and the v5 handle QC measured local X). Apply the rotation so local X = handle axis.
DB_FIX = bool(RC.get("dumbbell_axis_fix", False))
if DB_FIX: bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
DB_AX = (1.0, 0.0, 0.0) if DB_FIX else (0.0, 0.0, 1.0)
db.rotation_mode = "QUATERNION"
GRIP = float(MO.get("grip_offset_m", 0.075))
def grip_frame():
    Mb = arm.matrix_world @ pb[hand].matrix; R = Mb.to_3x3().normalized()
    u = (R @ mu.Vector((0, 1, 0))).normalized(); w = Mb.translation          # bone head = wrist, Y axis = bone direction
    if GW.get("enabled"):                                                           # handle in the palm, across it
        ha = (R @ mu.Vector(HG["handle_local"]["axis"])).normalized(); third = ha.cross(u).normalized()
        return Mb.translation + R @ mu.Vector(HG["handle_local"]["center"]), mu.Matrix((ha, u, third)).transposed().to_quaternion()
    if GRIP_MODE in ("palm", "palm_neutral"):
        ha, _ = palm_world(hand); ha = (ha - ha.dot(u) * u).normalized()                # handle across the palm
    else:
        lr = (arm.matrix_world @ pb["RightArm"].head - arm.matrix_world @ pb["LeftArm"].head).normalized()
        ha = (lr - lr.dot(u) * u).normalized()
    third = ha.cross(u).normalized()
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

# Shared construction: mild Corrective Smooth on a tight Spine01 / armpit group.
# v3.1 used factor 0.7 / 12 iters / LENGTH_WEIGHTED over a broad z-band and collapsed
# some edges to ratio 0.0. Weight-smooth (cached) does the bulk of the crease fix.
if RC.get("crease_smooth", True) and float(RC.get("crease_smooth_factor", 0.35)) > 0.01:
    def _vg_w(name):
        w = np.zeros(nv)
        if name in body.vertex_groups:
            gi_ = body.vertex_groups[name].index
            for v in me.vertices:
                for g in v.groups:
                    if g.group == gi_: w[v.index] = g.weight
        return w
    w_sp = _vg_w("Spine01")
    w_sh = np.maximum(_vg_w("LeftShoulder"), _vg_w("RightShoulder"))
    w_arm = np.maximum(_vg_w("LeftArm"), _vg_w("RightArm"))
    crease = (w_sp > 0.15) & (z >= 1.15) & (z <= 1.45)
    crease |= ((z >= 1.18) & (z <= 1.42) & (ax_ >= 0.06) & (ax_ <= 0.20)
               & ((w_sp > 0.08) | (w_sh > 0.12) | (w_arm > 0.12)))
    if "crease_smooth" not in body.vertex_groups:
        vg = body.vertex_groups.new(name="crease_smooth")
        idx = np.where(crease)[0]
        if len(idx): vg.add(idx.tolist(), 1.0, "REPLACE")
    if "crease_corrective" not in body.modifiers:
        mod = body.modifiers.new("crease_corrective", "CORRECTIVE_SMOOTH")
        mod.factor = float(RC.get("crease_smooth_factor", 0.35))
        mod.iterations = int(RC.get("crease_smooth_iter", 6))
        try: mod.smooth_type = "SIMPLE"
        except Exception: pass
        try: mod.use_pin_boundary = True
        except Exception: pass
        mod.use_only_smooth = False
        mod.vertex_group = "crease_smooth"
    log(f"crease corrective-smooth verts={int(crease.sum())} factor={RC.get('crease_smooth_factor', 0.35)}")

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
fill_w = float(CAMC.get("fill_w", 0.86)); top_y = float(CAMC.get("top_y", 0.24)); feet_y = float(CAMC.get("feet_y", 0.86))
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
chk = []; BN = [b.name for b in arm.data.bones]; BPOSE = np.zeros((NF, len(BN), 4, 4))
def _nm(M):       # world 4x4 with an orthonormal rotation part (the armature object carries the glTF scale)
    M = np.array(M); R = M[:3, :3]; M[:3, :3] = R / np.linalg.norm(R, axis=0, keepdims=True); return M
HANDF, HANDP = [], []      # v6: deformed working-hand vertices (every 3rd frame) for the grip QC
# v6 anatomy: capsule radius per bone (rest): median distance of its dominant vertices to the bone segment
_kids = {b.name: [c.name for c in b.children] for b in arm.data.bones}
def _seg(bn):
    b = arm.data.bones[bn]; h = np.array(arm.matrix_world @ b.head_local)
    if _kids[bn]: t = np.mean([np.array(arm.matrix_world @ arm.data.bones[c].head_local) for c in _kids[bn]], axis=0)
    else: t = h + np.array((arm.matrix_world.to_3x3().normalized() @ (b.matrix_local.to_3x3().normalized() @ mu.Vector((0, 1, 0))))) * 0.12
    return h, t
CAPR = {}
for bn in [b.name for b in arm.data.bones]:
    g_ = body.vertex_groups.get(bn)
    if g_ is None: continue
    vv = np.nonzero(dom == g_.index)[0]
    if len(vv) < 30: continue
    h_, t_ = _seg(bn); ab = t_ - h_; tt = np.clip(((co[vv] - h_) @ ab) / max(ab @ ab, 1e-9), 0, 1); dd = np.linalg.norm(co[vv] - (h_ + np.outer(tt, ab)), axis=1)
    CAPR[bn] = float(np.percentile(dd, 30))     # ~ the minor radius (the torso is flat, not round)
DBH = [mu.Vector(tuple(sx * c_ for c_ in DB_AX)) for sx in (-0.12, 0.12)]   # dumbbell head centres (object local)
def _visible_frac(pts, dist_eps=0.004):
    dg_ = bpy.context.evaluated_depsgraph_get(); ok_ = 0
    for p_ in pts:
        d_ = p_ - CAM.location; L_ = d_.length; hit, loc_, nrm_, ix_, ob_, mt_ = sc.ray_cast(dg_, CAM.location, d_.normalized(), distance=L_ - dist_eps)
        ok_ += (not hit) or ob_.name == "dumbbell"
    return ok_ / max(len(pts), 1)
DBM = np.zeros((NF, 4, 4)); BREST = np.array([_nm(arm.matrix_world @ arm.data.bones[n].matrix_local) for n in BN])
for f in range(1, NF + 1):
    sc.frame_set(f); Wm = arm.matrix_world; Pp = lambda b: Wm @ pb[b].head
    for bi, n in enumerate(BN): BPOSE[f - 1, bi] = _nm(Wm @ pb[n].matrix)
    DBM[f - 1] = _nm(db.matrix_world)
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
    _ex = {"head_top": Wm @ pb["head_end"].head if "head_end" in pb else Wm @ pb["Head"].head}
    for s_ in ("Left", "Right"):
        _ex[s_ + "_fingers"] = Wm @ pb[s_ + "Hand"].head + (Wm.to_3x3().normalized() @ pb[s_ + "Hand"].matrix.to_3x3().normalized() @ mu.Vector((0, 0.19, 0)))
        _ex[s_ + "_toe"] = Wm @ pb[s_ + "ToeBase"].head
    for k_, hc_ in zip(("db_head_a", "db_head_b"), DBH): _ex[k_] = db.matrix_world @ hc_
    rec["extrem2d"] = {k_: p2(v_) for k_, v_ in _ex.items()}
    rec["kp2d"] = {"hip": p2(hip), "sh": p2(shc), "sh_w": p2(sh_), "el": p2(el_), "wr": p2(wr_), "wr_s": p2(lwr if side == 12 else rwr), "kn": p2(kn_), "db": p2(dbp)}
    if f % 3 == 1 or f == NF:
        dg = bpy.context.evaluated_depsgraph_get(); eb = body.evaluated_get(dg); m_ = eb.to_mesh()
        c2 = np.empty(len(m_.vertices) * 3); m_.vertices.foreach_get("co", c2); Mw = np.array(eb.matrix_world)
        c2 = c2.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]; eb.to_mesh_clear()
        ratio = np.linalg.norm(c2[ev[ok_e, 0]] - c2[ev[ok_e, 1]], axis=1) / rest_len[ok_e]
        HANDF.append(f); HANDP.append(c2[HIDX].astype(np.float32))
        # v6 prop readability: share of each dumbbell head's sample points not hidden by the body/briefs, and the
        # on-screen separation of the two heads (centre distance / head diameter)
        Mdb = db.matrix_world; vis_ = []
        for hc_ in DBH:
            e1_ = mu.Vector((0, 1, 0)); e2_ = mu.Vector(DB_AX).cross(e1_)
            pts_ = [Mdb @ (hc_ + mu.Vector(DB_AX) * dx + 0.055 * math.cos(t_) * e1_ + 0.055 * math.sin(t_) * e2_) for dx in (-0.03, 0.0, 0.03) for t_ in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
            vis_.append(round(_visible_frac(pts_), 3))
        hp_ = [np.array(p2(Mdb @ hc_)) for hc_ in DBH]; rim_ = [np.array(p2(Mdb @ (hc_ + mu.Vector((0, 0.06, 0))))) for hc_ in DBH]
        hd_ = 2 * max(np.linalg.norm(rim_[0] - hp_[0]), np.linalg.norm(rim_[1] - hp_[1]), 1.0)
        for key_, sel_ in (("spike_where", np.argsort(-ratio)[:40]), ("collapse_where", np.argsort(ratio)[:40])):
            u_, n_ = np.unique(dom[ev[ok_e][sel_][:, 0]], return_counts=True); rec[key_] = {GN.get(int(u_[i]), "?"): int(n_[i]) for i in np.argsort(-n_)[:3]}
        e2_ = ev[ok_e]; vis_e = ~(shorts[e2_[:, 0]] | shorts[e2_[:, 1]])         # edges not under the shorts/briefs
        # visible spike = stretch relative to the local mesh scale: a 1 mm edge pulled to 4 mm is still smaller than a
        # median edge (3.6 mm) and invisible; L_deformed / max(L_rest, median rest edge)
        sr_ = ratio * rest_len[ok_e] / np.maximum(rest_len[ok_e], MED_E)
        rec["tear_max_visible"] = round(float(sr_[vis_e].max()), 3); rec["tear_max_raw_visible"] = round(float(ratio[vis_e].max()), 3)
        we_ = e2_[vis_e][int(np.argmax(sr_[vis_e]))]; rec["spike_at_rest"] = [round(float(v), 3) for v in (co[we_[0]] + co[we_[1]]) / 2]
        gm_ = {}
        for gi_ in np.unique(dom[e2_[:, 0]]):
            mm_ = (dom[e2_[:, 0]] == gi_) & vis_e; gm_[GN.get(int(gi_), "?")] = round(float(sr_[mm_].max()), 2) if mm_.any() else 0.0
        rec["spike_max_by_group"] = dict(sorted(gm_.items(), key=lambda kv: -kv[1])[:5])
        # min-ratio on visible, non-sliver rest edges (same visibility as tear_max_visible).
        # Raw ratio.min() was dominated by sub-median UV-seam slivers whose rest length is already << MED_E.
        long_e = rest_len[ok_e] >= 0.35 * MED_E
        pick = vis_e & long_e
        rec.update({"db_head_visible": vis_, "db_head_sep_rel": round(float(np.linalg.norm(hp_[0] - hp_[1]) / hd_), 3),
                    "tear_min_ratio": round(float((ratio[pick] if pick.any() else ratio[vis_e]).min()), 3)})
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
np.savez_compressed(a.out + "/hand.npz", frames=np.array(HANDF), P=np.array(HANDP), labels=HG["labels"], s_finger=HG["s_finger"],
                    handle_r=float(GW.get("handle_r_m", 0.016)), wrap=bool(GW.get("enabled", False)), info=json.dumps(HG["info"]))
np.savez_compressed(a.out + "/bones.npz", names=np.array(BN), rest=BREST, pose=BPOSE, neck_mode=NECK_MODE, legs=LEGS, dumbbell=DBM,
                    capsule_names=np.array(list(CAPR)), capsule_r=np.array(list(CAPR.values())),
                    handle_axis_local=np.array(DB_AX), working_hand=hand, palm_left=PALM["LeftHand"], palm_right=PALM["RightHand"], grip_mode=GRIP_MODE)
log("checks done")
if a.checks_only: sys.exit(0)

# ---- materials + lights (P0 quality upgrade) -----------------------------------------------------------------------
body_mats = [s.material for s in body.material_slots]
ad = float(RC.get("albedo_detail", 0.50))
sss_w = float(RC.get("sss_weight", 0.16))
em_str = float(RC.get("glow_emission", 1.35))
glow_wrong = C.get("grade", {}).get("glow") == "wrong"
em_col = (1.0, 0.28, 0.10, 1) if glow_wrong else (0.16, 1.0, 0.42, 1)
base_rgb = tuple(float(x) for x in RC.get("base_color", [0.22, 0.225, 0.232]))

for m_ in body_mats:
    nt = m_.node_tree; bsdf = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"][0]
    for l in list(bsdf.inputs["Base Color"].links): nt.links.remove(l)
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "shorts"; mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs[6].default_value = (*base_rgb, 1); mix.inputs[7].default_value = (*base_rgb, 1) if garm is not None else (0.012, 0.012, 0.014, 1)
    nt.links.new(at.outputs["Fac"], mix.inputs[0])
    tex = [n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image and "basecolor" in n.image.name.lower()]
    if tex and ad > 0:   # keep the asset's painted fibre detail: grey x (albedo luminance, contrast-boosted)
        bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(tex[0].outputs["Color"], bw.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["From Min"].default_value = 0.15; mr.inputs["From Max"].default_value = 0.8
        mr.inputs["To Min"].default_value = 1.0 - ad; mr.inputs["To Max"].default_value = 1.0 + ad; nt.links.new(bw.outputs[0], mr.inputs["Value"])
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"; mul.inputs[0].default_value = 1.0
        nt.links.new(mix.outputs[2], mul.inputs[6]); nt.links.new(mr.outputs[0], mul.inputs[7]); nt.links.new(mul.outputs[2], bsdf.inputs["Base Color"])
    else:
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    if not bsdf.inputs["Roughness"].links:          # keep the packed roughness map when the GLB already wired it
        bsdf.inputs["Roughness"].default_value = 0.50
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = sss_w
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.55, 0.58, 0.62)
    if "Subsurface Scale" in bsdf.inputs:
        bsdf.inputs["Subsurface Scale"].default_value = float(RC.get("sss_scale", 0.05))
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    for nm in [n for n in nt.nodes if n.type == "NORMAL_MAP"]:
        nm.inputs["Strength"].default_value = float(RC.get("normal_strength", 2.0))
    # 3D glow: shared tools/rig3d/glow_shader.py (simple emission for the 2D-grade path).
    mout = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import glow_shader as GSH
_flex_min = float(RC.get("glow_flex_min", 0.58))
flex_nodes = GSH.switch_materials(body_mats, "post", em_col, em_str, _flex_min, float(RC.get("glow_fibre_scale", 42.0)))

sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.render.film_transparent = True
wd = bpy.data.worlds.new("w"); sc.world = wd; wd.use_nodes = True; bg = wd.node_tree.nodes["Background"]
hdri = C["asset"]["hdri"]; envn = wd.node_tree.nodes.new("ShaderNodeTexEnvironment"); envn.image = bpy.data.images.load(hdri["file"])
wd.node_tree.links.new(envn.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = float(RC.get("hdri_strength", hdri.get("strength", 0.35)))
if hasattr(wd, "cycles_visibility") and hasattr(wd.cycles_visibility, "camera"):
    wd.cycles_visibility.camera = False                                          # HDRI lights the figure, never shows as a photographed studio
key_from = (CAM.location - ctr); key_from.z = 0; key_from.normalize()
upv = mu.Vector((0.0, 0.0, 1.0)); sidev = key_from.cross(upv)
sidev = (sidev if sidev.length > 1e-4 else mu.Vector((1.0, 0.0, 0.0))).normalized()

def add_area(name, loc, energy, size, color):
    Lt = bpy.data.lights.new(name, "AREA"); Lt.energy = float(energy); Lt.size = float(size); Lt.color = color
    o = bpy.data.objects.new(name, Lt); col.objects.link(o)
    o.location = loc; o.rotation_euler = (ctr - o.location).to_track_quat("-Z", "Y").to_euler(); return o

add_area("key", ctr + key_from * 2.8 + upv * 1.8, RC.get("key_energy", 380), 2.2, (1.00, 0.98, 0.94))
add_area("fill", ctr - key_from * 0.5 + sidev * 2.4 + upv * 2.2, RC.get("fill_energy", 150), 3.4, (0.90, 0.94, 1.00))
add_area("rim", ctr - key_from * 2.6 + upv * 1.9, RC.get("rim_energy", 270), 1.6, (0.65, 0.82, 1.00))
# Shared construction: a lower key + rim aimed at the legs so charcoal grade cannot crush
# the lower body into the backdrop (silhouette separation, charcoal identity kept).
leg_aim = mu.Vector((ctr.x, ctr.y, 0.48))
def add_area_aim(name, loc, energy, size, color, aim):
    Lt = bpy.data.lights.new(name, "AREA"); Lt.energy = float(energy); Lt.size = float(size); Lt.color = color
    o = bpy.data.objects.new(name, Lt); col.objects.link(o)
    o.location = loc; o.rotation_euler = (aim - o.location).to_track_quat("-Z", "Y").to_euler(); return o
add_area_aim("leg_key", leg_aim + key_from * 2.4 + upv * 0.55, RC.get("leg_key_energy", 190), 1.8, (1.00, 0.98, 0.94), leg_aim)
add_area_aim("leg_rim", leg_aim - key_from * 2.3 + upv * 0.70, RC.get("leg_rim_energy", 170), 1.5, (0.70, 0.84, 1.00), leg_aim)
floor = None
if RC.get("floor", True):
    import bmesh
    flz = float(RC.get("floor_z", 0.0))
    flsz = float(RC.get("floor_size_m", 8.0))
    bpy.ops.mesh.primitive_plane_add(size=flsz, location=(ctr.x, ctr.y, flz))
    floor = bpy.context.object; floor.name = "studio_floor"
    me = floor.data
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=22)
    bm.to_mesh(me); bm.free(); me.update()
    away = (ctr - CAM.location); away.z = 0
    away = (away if away.length > 1e-6 else mu.Vector((0.0, 1.0, 0.0))).normalized()
    for v in me.vertices:                                          # cyclorama: far half curves up into a wall
        w = floor.matrix_world @ v.co
        d = (w.x - ctr.x) * away.x + (w.y - ctr.y) * away.y
        t = min(max((d - 0.45) / 2.6, 0.0), 1.0)
        v.co.z = (t * t) * 3.4
    me.update()
    fm = bpy.data.materials.new("studio_floor"); fm.use_nodes = True
    nt = fm.node_tree; fb = nt.nodes["Principled BSDF"]; mout = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
    fb.inputs["Base Color"].default_value = (0.008, 0.009, 0.010, 1)
    fb.inputs["Roughness"].default_value = 0.55
    if "Specular IOR Level" in fb.inputs:
        fb.inputs["Specular IOR Level"].default_value = 0.28
    if "Emission Color" in fb.inputs:
        fb.inputs["Emission Color"].default_value = (0.011, 0.012, 0.014, 1)
        if "Emission Strength" in fb.inputs:
            fb.inputs["Emission Strength"].default_value = 0.12
    tex = nt.nodes.new("ShaderNodeTexCoord")
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
    nt.links.new(tex.outputs["Object"], ln.inputs[0])
    fade_xy = nt.nodes.new("ShaderNodeMapRange")
    fade_xy.clamp = True
    fade_xy.inputs["From Min"].default_value = 1.05
    fade_xy.inputs["From Max"].default_value = 2.85
    fade_xy.inputs["To Min"].default_value = 1.0
    fade_xy.inputs["To Max"].default_value = 0.0
    nt.links.new(ln.outputs["Value"], fade_xy.inputs["Value"])
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(tex.outputs["Object"], sep.inputs[0])
    fade_z = nt.nodes.new("ShaderNodeMapRange")
    fade_z.clamp = True
    fade_z.inputs["From Min"].default_value = 0.25
    fade_z.inputs["From Max"].default_value = 2.2
    fade_z.inputs["To Min"].default_value = 1.0
    fade_z.inputs["To Max"].default_value = 0.0
    nt.links.new(sep.outputs["Z"], fade_z.inputs["Value"])
    fac_a = nt.nodes.new("ShaderNodeMath"); fac_a.operation = "MULTIPLY"
    nt.links.new(fade_xy.outputs["Result"], fac_a.inputs[0]); nt.links.new(fade_z.outputs["Result"], fac_a.inputs[1])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
    nt.links.new(fac_a.outputs[0], inv.inputs[1])
    nt.links.new(inv.outputs[0], mix.inputs["Fac"])
    nt.links.new(fb.outputs[0], mix.inputs[1]); nt.links.new(tr.outputs[0], mix.inputs[2])
    for l in list(mout.inputs["Surface"].links): nt.links.remove(l)
    nt.links.new(mix.outputs[0], mout.inputs["Surface"])
    floor.data.materials.clear(); floor.data.materials.append(fm)
    try: col.objects.link(floor)
    except RuntimeError: pass
sc.frame_start, sc.frame_end = 1, NF; sc.render.fps = int(D["fps"]); sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
frames = list(range(1, NF + 1)) if a.frames == "all" else [int(v) for v in a.frames.split(",")]
def flex_at(f):
    els = [c["elbow_deg"] for c in chk]
    emin, emax = min(els), max(els)
    if emax - emin < 1e-3: return 0.0
    return float((emax - els[f - 1]) / (emax - emin))

def update_regions(f):
    sc.frame_set(f); bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get(); eb = body.evaluated_get(dg); m_ = eb.to_mesh()
    c2 = np.empty(len(m_.vertices) * 3); m_.vertices.foreach_get("co", c2)
    n2 = np.empty(len(m_.vertices) * 3); m_.vertices.foreach_get("normal", n2)
    Mw = np.array(eb.matrix_world)
    c2 = c2.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
    n2 = n2.reshape(-1, 3) @ Mw[:3, :3].T
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-9
    eb.to_mesh_clear()
    nuse = min(len(c2), nv)
    fl = flex_at(f)
    rg, info = MREG.assign(c2[:nuse], n2[:nuse], _bone_heads(True), ADJ, shorts, EXCLUDE, flex=fl, peak_dilate=_peak_dilate, prior=PRIOR[:nuse])
    # Union rest-pose paint (travels with verts, continuous at the hinged start) with
    # the per-frame assign (paints surface that scapula retract newly exposes at peak).
    if bool(RC.get("glow_union_rest", True)):
        rest = REST_REG[:nuse]
        rg = np.maximum(rg, rest)
        info = dict(info, union_rest=True, rest_px=int((rest.max(1) > 0.5).sum()))
    if nuse < nv:
        pad = np.zeros((nv, 3), np.float32); pad[:nuse] = rg; rg = pad
    MREG.write_mregion(body.data, rg)
    GSH.set_flex(flex_nodes, fl)
    return fl, info

def render_set(prefix):
    skipped = 0
    for f in frames:
        path = f"{a.out}/{prefix}_{f:04d}.png"
        if RC.get("skip_existing_frames") and os.path.exists(path):
            skipped += 1; continue
        update_regions(f)
        sc.frame_set(f); sc.render.filepath = path; bpy.ops.render.render(write_still=True)
    if skipped: log(f"{prefix}: skipped {skipped} existing frames")
tim = {"retarget_s": round(time.time() - T0, 1)}
if not a.mask_only:
    sc.cycles.samples = int(RC.get("spp", 12)); sc.cycles.use_denoising = bool(RC.get("denoise", True))
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"; sc.cycles.denoising_prefilter = RC.get("denoise_prefilter", "ACCURATE")
        sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception as e: log(f"denoise option: {e}")
    sc.view_settings.view_transform = "AgX"
    t = time.time(); render_set("f"); tim["beauty_s"] = round(time.time() - t, 1); log(f"beauty {tim['beauty_s']} s")
    want_ab = bool(RC.get("glow_ab") or str(RC.get("glow_backend", "post")).lower() in ("blender", "both"))
    if want_ab:
        flex_nodes[:] = GSH.switch_materials(body_mats, "blender", em_col, em_str, _flex_min, float(RC.get("glow_fibre_scale", 42.0)))
        GSH.setup_compositor(sc, float(RC.get("blender_bloom", 0.60)))
        t = time.time(); render_set("b"); tim["beauty_blender_s"] = round(time.time() - t, 1); log(f"beauty blender-glow {tim['beauty_blender_s']} s")
        GSH.teardown_compositor(sc)
if not a.no_mask:
    if floor is not None:
        floor.hide_render = True                                                 # floor must not leak into the muscle mask
    GSH.teardown_compositor(sc)
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
json.dump({**tim, "frames": len(frames), "res": [rx, ry], "skipped_existing": bool(RC.get("skip_existing_frames"))}, open(a.out + "/render_time.json", "w"))
print("RENDER_DONE", json.dumps(tim))
