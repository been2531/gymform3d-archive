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
ap.add_argument("--camera-sheet", default=None, help="render an 8-camera contact sheet of the peak pose and exit")
ap.add_argument("--cam-preview", action="store_true",
                help="fit 4 yaw/elev candidates, render cheap peak stills, write metrics, exit")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
C = json.load(open(a.cfg)); MO, RC = C["motion"], C["render"]
def apply_form_offsets(mo):
    """Construct a common-mistake take from the same free motion via rig offsets (v3.4).
    Shared: not a generated clip and not a per-exercise hack. `form: wrong` overlays motion.wrong_form."""
    if str(mo.get("form") or "right").lower() != "wrong":
        return mo
    mo = dict(mo); WF = dict(mo.get("wrong_form") or {})
    if "torso_target_deg" in WF:
        mo["torso_target_deg"] = float(WF["torso_target_deg"])
    elif "torso_rise_deg" in WF:
        mo["torso_target_deg"] = float(mo.get("torso_target_deg", 45)) - float(WF["torso_rise_deg"])
    for k in ("elbow_path", "shrug_m", "round_back_deg", "scapula_retract", "elbow_flare_extra_m", "elbow_flare_max_m"):
        if k in WF:
            mo[k] = WF[k]
    mo.setdefault("elbow_path", "flare")
    mo.setdefault("shrug_m", 0.0)
    mo.setdefault("round_back_deg", 0.0)
    mo.setdefault("scapula_retract", False)
    return mo
MO = apply_form_offsets(MO); C["motion"] = MO
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
BENCH_OBJ = None; BENCH_META = None; BENCH_SPEC = None
AUTH_META = None; AUTH_SPEC = None; AUTH_U = None
if LEGS == "bench_row":
    # Explicit world targets BEFORE any IK. Bench along +Y, head toward +Y, +X = working right.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import authored_motion as AM
    import bench_row_pose as BRP
    AUTH_SPEC = AM.load_spec(C)
    _form = str(MO.get("form") or "right").lower()
    _u, _fps, _dur = AM.phase_u(AUTH_SPEC, _form)
    _wh = "Left" if side == 11 else "Right"
    _Lt = 0.5 * (rl("LeftUpLeg", "LeftLeg") + rl("RightUpLeg", "RightLeg"))
    _Ls = 0.5 * (rl("LeftLeg", "LeftFoot") + rl("RightLeg", "RightFoot"))
    _La = rl(_wh + "Arm", _wh + "ForeArm"); _Lf = rl(_wh + "ForeArm", _wh + "Hand")
    _tor = ((Wr @ bl["LeftArm"].head_local + Wr @ bl["RightArm"].head_local) / 2 - Wr @ bl["Hips"].head_local).length
    _shw = 0.5 * (Wr @ bl["LeftArm"].head_local - Wr @ bl["RightArm"].head_local).length
    _hpw = 0.5 * (Wr @ bl["LeftUpLeg"].head_local - Wr @ bl["RightUpLeg"].head_local).length
    _sp = (C.get("spec") or {}).get("pose") or MO.get("spec_pose") or {}
    B, BENCH_SPEC = BRP.build_landmarks(_u, {
        "thigh": _Lt, "shin": _Ls, "upper_arm": _La, "forearm": _Lf,
        "torso": float(_tor), "shoulder_half": float(_shw), "hip_half": float(_hpw),
        "pad_z": float(MO.get("bench_pad_z_m", 0.43)),
        "hand_ahead_m": float(MO.get("support_hand_ahead_m", 0.55)),
        "spec_pose": _sp,
    }, side=side)
    NF = int(B.shape[0]); AUTH_U = _u
    AUTH_META = {"u": _u, "fps": _fps, "dur_s": _dur, "n": NF, "form": _form,
                 "torso_start_deg": 90.0, "torso_peak_deg": 90.0, "world_targets": True}
    np.save(a.out + "/authored_landmarks_prior.npy", B)
    _want_fit = bool(MO.get("ref_fit", False))
    if isinstance(_sp, dict) and "ref_fit" in _sp:
        _want_fit = bool(_sp["ref_fit"])
    try:
        import ref_match as RM
        _here = os.path.dirname(os.path.abspath(__file__))
        _repo = os.path.dirname(os.path.dirname(_here))
        _ref = os.path.join(_repo, "_scratch", "refs", "db_row")
        if not _want_fit:
            log("ref-fit skipped (authored world targets win unless pose.ref_fit is true)")
        elif os.path.isfile(os.path.join(_ref, "lm2d_peak.npy")):
            _res = tuple(RC.get("res") or [540, 960])
            _cam = RM.camera_from_cfg(RC.get("camera_fit") or RC.get("camera") or {})
            _Lfit = {"thigh": float(_Lt), "shin": float(_Ls), "upper_arm": float(_La),
                     "forearm": float(_Lf), "torso": float(_tor)}
            B, _fitinfo = RM.fit_cycle(B, AUTH_U, _ref, _cam, _res, BENCH_SPEC, _Lfit)
            AUTH_META["ref_fit"] = _fitinfo
            json.dump(_fitinfo, open(a.out + "/ref_fit.json", "w"), indent=2)
            log(f"ref-fit bottom mae={_fitinfo['bottom']['joint_mae_ndc']} peak mae={_fitinfo['peak']['joint_mae_ndc']}")
        else:
            log(f"ref-fit skipped: no prepared landmarks at {_ref}")
    except Exception as _fe:
        import traceback
        log(f"ref-fit failed: {_fe}")
        traceback.print_exc()
    np.save(a.out + "/authored_landmarks.npy", B)
    np.save(a.out + "/authored_u.npy", AUTH_U)
    json.dump({k: AUTH_META[k] for k in AUTH_META if k != "u"}, open(a.out + "/authored_meta.json", "w"), default=float)
    MO["elbow_path"] = "source"
    MO["shrug_m"] = 0.0
    MO["grip_twist_forearm_share"] = 1.0
    MO["scapula_retract"] = False
    try:
        import importlib.util
        _bp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "props", "bench_flat.py")
        _spec = importlib.util.spec_from_file_location("bench_flat_prop", _bp)
        _BF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_BF)
        _bdict = RC.get("bench") if isinstance(RC.get("bench"), dict) else {}
        _look = str(_bdict.get("look") or MO.get("bench_look") or "gunmetal_black")
        _padded = bool(((C.get("kit") or {}).get("bench") or {}).get("padded"))
        BENCH_OBJ, BENCH_META = _BF.build_bench(look=_look, clear=False, padded=_padded)
        _BF.place_bench(BENCH_OBJ, BENCH_SPEC["origin_xy"], BENCH_SPEC["yaw_z"], BENCH_SPEC["origin_z"])
        log(f"bench_row WORLD +Y hip_z={BENCH_SPEC['hip_z']:.3f} kn={np.asarray(BENCH_SPEC['support_knee']).round(3).tolist()} hand={np.asarray(BENCH_SPEC['support_hand']).round(3).tolist()}")
    except Exception as _be:
        log(f"bench spawn failed: {_be}")
    log(f"authored {_form} {NF} frames {_dur:.2f}s fps {_fps} (explicit world targets)")
if (MO.get("authored") or {}).get("enabled") and AUTH_META is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import authored_motion as AM
    AUTH_SPEC = AM.load_spec(C)
    _form = str(MO.get("form") or "right").lower()
    _wh = "Left" if side == 11 else "Right"
    _la, _lf = rl(_wh + "Arm", _wh + "ForeArm"), rl(_wh + "ForeArm", _wh + "Hand")
    B, AUTH_META = AM.expand_landmarks(B, AUTH_SPEC, side=side, form=_form, arm_len=(_la, _lf))
    NF = int(B.shape[0]); AUTH_U = AUTH_META["u"]
    np.save(a.out + "/authored_landmarks.npy", B)
    # authored landmarks already place the working elbow; don't let tuck/flare rebuild it
    MO["elbow_path"] = "source"
    MO["shrug_m"] = 0.0   # clavicle offset applied below from AUTH_U (not the reserved matrix shrug)
    MO["grip_twist_forearm_share"] = 1.0  # radioulnar (forearm) sets palm facing; hand stays rest-relative so the fist shapekey + handle stay aligned (share=0 sheared the wrap ~100 deg at the wrist)
    _prof = AM.deep_merge(AUTH_SPEC.get("right") or {}, AUTH_SPEC.get(_form) or {} if _form != "right" else {})
    # Real scapula retract is applied in the authored clavicle block (medial translation +
    # slight upward rotation). Skip the old 6° local-axis probe (score ~0, invisible).
    MO["scapula_retract"] = False
    np.save(a.out + "/authored_u.npy", AUTH_U)
    json.dump({k: AUTH_META[k] for k in AUTH_META if k != "u"}, open(a.out + "/authored_meta.json", "w"), default=float)
    log(f"authored {_form} {NF} frames {AUTH_META['dur_s']:.2f}s fps {AUTH_META['fps']}")
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
if LEGS == "bench_row" and BENCH_SPEC is not None:
    # Lock support knee to the pad; solve the working knee toward the floor ankle.
    support_left = bool(BENCH_SPEC["support_left"])
    sp_ = "Left" if support_left else "Right"
    wk_ = "Right" if support_left else "Left"
    kn_tgt = mu.Vector(BENCH_SPEC["support_knee"])
    ekn = E[sp_[0].lower() + "kn"]
    eak = E[sp_[0].lower() + "ank"]
    eft = E[sp_[0].lower() + "ft"]
    ank_tgt = mu.Vector(BENCH_SPEC.get("support_ankle") or BENCH_SPEC["support_knee"])
    for f in range(NF):
        ekn.location = kn_tgt
        ekn.keyframe_insert("location", frame=f + 1)
        eak.location = ank_tgt
        eak.keyframe_insert("location", frame=f + 1)
        eft.location = ank_tgt + mu.Vector((0.0, -0.08, -0.01))
        eft.keyframe_insert("location", frame=f + 1)
    # working leg 2-bone IK
    Lt, Ls = rl(wk_ + "UpLeg", wk_ + "Leg"), rl(wk_ + "Leg", wk_ + "Foot")
    aj = 28 if wk_ == "Right" else 27
    A = mu.Vector(B[0, aj].tolist())
    pole = mu.Vector(BENCH_SPEC["fwd"]); pole.z = 0
    if pole.length < 1e-6: pole = mu.Vector((0.0, -1.0, 0.0))
    pole.normalize()
    p_ = wk_[0].lower()
    for f in range(NF):
        sc.frame_set(f + 1)
        H = arm.matrix_world @ pb[wk_ + "UpLeg"].head
        dv = A - H; d = min(max(dv.length, 1e-4), 0.999 * (Lt + Ls)); u_ = dv.normalized()
        x_ = (Lt * Lt - Ls * Ls + d * d) / (2 * d); h_ = math.sqrt(max(Lt * Lt - x_ * x_, 0.0))
        pp = (pole - pole.dot(u_) * u_)
        if pp.length < 1e-6: pp = mu.Vector((0.0, 0.0, 1.0))
        pp.normalize()
        E[p_ + "kn"].location = H + u_ * x_ + pp * h_
        E[p_ + "kn"].keyframe_insert("location", frame=f + 1)
        E[p_ + "ank"].location = A
        E[p_ + "ank"].keyframe_insert("location", frame=f + 1)
        toe = mu.Vector(B[0, 32 if wk_ == "Right" else 31].tolist())
        E[p_ + "ft"].location = toe
        E[p_ + "ft"].keyframe_insert("location", frame=f + 1)
    log(f"bench_row IK: {sp_} knee locked to pad, {wk_} foot on floor z={A.z:.3f}")
shrug_m = float(MO.get("shrug_m", 0.0) or 0.0)
if shrug_m > 0 and (("LeftShoulder" if side == 11 else "RightShoulder") in pb):
    # World-space translate the working Shoulder toward the head (scapular elevation),
    # composed onto the already-keyed pose. Replacing the shoulder quaternion with a
    # probed-axis rotation wiped the retarget and did not measure as a rise.
    wh_ = "Left" if side == 11 else "Right"; sb = pb[wh_ + "Shoulder"]; ppw = wh_[0].lower()
    angs = []
    for f in range(NF):
        sc.frame_set(f + 1); S = arm.matrix_world @ pb[wh_ + "Arm"].head; El = E[ppw + "el"].location; Wri = E[ppw + "wr"].location
        angs.append(math.degrees((S - El).angle(Wri - El)) if (S - El).length > 1e-6 and (Wri - El).length > 1e-6 else 180.0)
    amin, amax = min(angs), max(angs)
    for f, ang in enumerate(angs):
        amt = 0.0 if amax - amin < 1e-3 else (amax - ang) / (amax - amin)
        sc.frame_set(f + 1); bpy.context.view_layer.update()
        sh = arm.matrix_world @ pb[wh_ + "Arm"].head
        hd = arm.matrix_world @ pb["Head"].head
        toward = hd - sh
        if toward.length < 1e-6:
            continue
        toward.normalize()
        x = shrug_m * amt
        # keep the working-arm IK reachable after the shoulder moves
        if ppw + "wr" in E:
            Wt = E[ppw + "wr"].location
            La, Lf = rl(wh_ + "Arm", wh_ + "ForeArm"), rl(wh_ + "ForeArm", wh_ + "Hand")
            reach = 0.98 * (La + Lf)
            if (Wt - (sh + toward * x)).length > reach:
                lo, hi = 0.0, x
                for _ in range(10):
                    mid = 0.5 * (lo + hi)
                    if (Wt - (sh + toward * mid)).length > reach: hi = mid
                    else: lo = mid
                x = lo
        # don't drive the arm capsule through the head (~8 cm combined radius)
        x = min(x, max(0.0, toward.length - 0.08))
        delta = toward * x
        if sb.parent:
            parent_mw = arm.matrix_world @ sb.parent.matrix
            delta_local = parent_mw.to_3x3().inverted() @ delta
        else:
            delta_local = arm.matrix_world.to_3x3().inverted() @ delta
        sb.location = sb.location.copy() + delta_local
        bpy.context.view_layer.update()
        sb.keyframe_insert("location", frame=f + 1)
    log(f"shrug {wh_} translate toward head max {shrug_m:.3f} m")
round_back_deg = float(MO.get("round_back_deg", 0.0) or 0.0)
if round_back_deg > 0:
    # World-matrix kyphosis about each bone head (not a local-axis quaternion multiply,
    # which turned an 8 deg request into ~50 deg and yanked the neck). Neck/Spine02 stay
    # as keyed so the head continues the rounded thoracic line.
    sc.frame_set(1); bpy.context.view_layer.update()
    lat = (arm.matrix_world @ pb["LeftUpLeg"].head) - (arm.matrix_world @ pb["RightUpLeg"].head)
    lat = mu.Vector((lat.x, lat.y, 0.0))
    if lat.length < 1e-6: lat = mu.Vector((1.0, 0.0, 0.0))
    lat.normalize()
    def _thor_ang():
        hips = arm.matrix_world @ pb["Hips"].head
        s1 = arm.matrix_world @ pb["Spine01"].head
        nk = arm.matrix_world @ (pb["neck"].head if "neck" in pb else pb["Head"].head)
        return math.degrees((s1 - hips).angle(nk - s1)) if (s1 - hips).length > 1e-6 and (nk - s1).length > 1e-6 else 0.0
    def _rot_about_head(sb, axis, deg):
        mw = (arm.matrix_world @ sb.matrix).copy()
        hd = mw.to_translation().copy()
        q = mu.Quaternion(axis.normalized(), math.radians(deg))
        mw_new = mu.Matrix.Translation(hd) @ q.to_matrix().to_4x4() @ mu.Matrix.Translation(-hd) @ mw
        sb.matrix = arm.matrix_world.inverted() @ mw_new
    base = _thor_ang()
    _rot_about_head(pb["Spine01"], lat, 3.0); bpy.context.view_layer.update()
    if _thor_ang() < base:
        lat = -lat
    sc.frame_set(1); bpy.context.view_layer.update()  # drop the 3 deg probe
    for bname, share in (("Spine", 0.35), ("Spine01", 0.65)):
        if bname not in pb: continue
        sb = pb[bname]
        for f in range(NF):
            sc.frame_set(f + 1); bpy.context.view_layer.update()
            _rot_about_head(sb, lat, round_back_deg * share)
            bpy.context.view_layer.update()
            sb.keyframe_insert("rotation_quaternion", frame=f + 1)
    log(f"round_back {round_back_deg:.1f} deg world-lateral about bone heads")
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
if MO.get("support_hand", "") == "bench_pad" and BENCH_SPEC is not None:
    # Support hand locked to the pad (same side as the support knee).
    sp_ = "Left" if int(side) == 12 else "Right"; pp_ = sp_[0].lower()
    La, Lf = rl(sp_ + "Arm", sp_ + "ForeArm"), rl(sp_ + "ForeArm", sp_ + "Hand")
    Wt0 = mu.Vector(BENCH_SPEC["support_hand"])
    latv = mu.Vector((lat_h if 'lat_h' in dir() else (1.0, 0.0, 0.0)))
    if 'lat_h' in dir():
        latv = mu.Vector(lat_h.tolist()) * (1 if sp_ == "Left" else -1)
    fwdv = mu.Vector(BENCH_SPEC["fwd"])
    for f in range(NF):
        sc.frame_set(f + 1); bpy.context.view_layer.update()
        Sh = arm.matrix_world @ pb[sp_ + "Arm"].head
        Wt = Wt0.copy()
        dv = Wt - Sh; u_ = dv.normalized(); d_wr = min(max(dv.length, 1e-4), 0.999 * (La + Lf)); d = d_wr
        x_ = (La * La - Lf * Lf + d * d) / (2 * d); h_ = math.sqrt(max(La * La - x_ * x_, 0.0))
        pv_ = latv * 0.55 - fwdv * 0.35 + mu.Vector((0.0, 0.0, 0.25))
        pole = (pv_ - pv_.dot(u_) * u_)
        if pole.length < 1e-6: pole = mu.Vector((0.0, 0.0, 1.0))
        pole.normalize()
        E[pp_ + "el"].location = Sh + u_ * x_ + pole * h_
        E[pp_ + "wr"].location = Wt
        E[pp_ + "ix"].location = Wt + fwdv.normalized() * 0.035
        for n_ in ("el", "wr", "ix"):
            E[pp_ + n_].keyframe_insert("location", frame=f + 1)
    # Flat palm on the pad: keep finger aim (DAMPED_TRACK Y -> ix along the bench)
    # and roll so a palmar axis points at world down.
    down_e = empty("T_palm_down"); down_e.location = mu.Vector((0.0, 0.0, -80.0))
    sc.frame_set(1); bpy.context.view_layer.update()
    Rhm = np.array((arm.matrix_world @ pb[sp_ + "Hand"].matrix).to_3x3())
    Rhm = Rhm / (np.linalg.norm(Rhm, axis=0, keepdims=True) + 1e-12)
    down = np.array([0.0, 0.0, -1.0])
    best_ax, best_name, best_dot = 0, "TRACK_NEGATIVE_Z", -1e9
    for ai, pos, neg in ((0, "TRACK_X", "TRACK_NEGATIVE_X"), (2, "TRACK_Z", "TRACK_NEGATIVE_Z")):
        d = float(np.dot(Rhm[:, ai], down))
        name = pos if d >= 0 else neg
        if abs(d) > best_dot:
            best_dot, best_ax, best_name = abs(d), ai, name
    cpalm = pb[sp_ + "Hand"].constraints.new("LOCKED_TRACK")
    cpalm.target = down_e
    cpalm.track_axis = best_name
    cpalm.lock_axis = "LOCK_Y"
    log(f"support hand locked to bench pad ({sp_}), palm-down {best_name} (align {best_dot:.2f})")
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
if MO.get("elbow_path", "source") == "flare":
    # Opposite of tuck_torso: walk the 2-bone circle AWAY from Spine01 (elbow flare),
    # still constrained to the IK circle and an anatomical max torso distance.
    wh_ = "Left" if side == 11 else "Right"; ppw = wh_[0].lower()
    La_w, Lf_w = rl(wh_ + "Arm", wh_ + "ForeArm"), rl(wh_ + "ForeArm", wh_ + "Hand")
    max_torso = float(MO.get("elbow_flare_max_m", 0.48))
    for f in range(NF):
        sc.frame_set(f + 1); Sh = arm.matrix_world @ pb[wh_ + "Arm"].head; Wt = E[ppw + "wr"].location.copy()
        dv = Wt - Sh; d = min(max(dv.length, 1e-4), 0.995 * (La_w + Lf_w)); u_ = dv.normalized()
        x_ = (La_w * La_w - Lf_w * Lf_w + d * d) / (2 * d); h_ = math.sqrt(max(La_w * La_w - x_ * x_, 0.0))
        mid = Sh + u_ * x_; Sp = arm.matrix_world @ pb["Spine01"].head
        # Wing the elbow out along the hip-lateral (same-side), not "straight away from
        # Spine01" — that opposite pole crosses the arm line and inverts signed flex.
        if wh_ == "Right":
            out_dir = (arm.matrix_world @ pb["RightUpLeg"].head) - (arm.matrix_world @ pb["LeftUpLeg"].head)
        else:
            out_dir = (arm.matrix_world @ pb["LeftUpLeg"].head) - (arm.matrix_world @ pb["RightUpLeg"].head)
        out_dir = mu.Vector((out_dir.x, out_dir.y, 0.0))
        if out_dir.length < 1e-6: out_dir = mu.Vector((1.0 if wh_ == "Left" else -1.0, 0.0, 0.0))
        pole_flare = out_dir - out_dir.dot(u_) * u_
        if pole_flare.length < 1e-5: pole_flare = mu.Vector((0.0, 0.0, 1.0))
        pole_flare.normalize()
        El_src = E[ppw + "el"].location.copy(); src_pole = El_src - mid; src_pole = src_pole - src_pole.dot(u_) * u_
        if src_pole.length < 1e-5: src_pole = pole_flare
        src_pole.normalize()
        best = mid + src_pole * h_; best_d = (best - Sp).length
        for a_ in np.linspace(0.0, 1.0, 16):
            p_ = (src_pole * (1.0 - a_) + pole_flare * a_); p_ = p_ - p_.dot(u_) * u_
            if p_.length < 1e-5: continue
            p_.normalize()
            if p_.dot(src_pole) < 0.0:
                continue  # far side inverts signed elbow flex
            El = mid + p_ * h_; dd = (El - Sp).length
            el_ang = math.degrees((Sh - El).angle(Wt - El)) if (Sh - El).length > 1e-6 and (Wt - El).length > 1e-6 else 180.0
            if dd <= max_torso and dd >= best_d and 15.0 <= el_ang <= 175.0:
                best = El; best_d = dd
        E[ppw + "el"].location = best; E[ppw + "el"].keyframe_insert("location", frame=f + 1)
    log(f"elbow_path flare on {wh_} max {max_torso:.2f} m")
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
        sc.frame_set(f + 1); S = arm.matrix_world @ pb[wh_ + "Arm"].head; El = E[ppw + "el"].location; Wri = E[ppw + "wr"].location
        angs.append(math.degrees((S - El).angle(Wri - El)) if (S - El).length > 1e-6 and (Wri - El).length > 1e-6 else 180.0)
    amin, amax = min(angs), max(angs); max_deg = float(MO.get("scapula_retract_deg", 9.0))
    for f, ang in enumerate(angs):
        amt = 0.0 if amax - amin < 1e-3 else (amax - ang) / (amax - amin)
        sb.rotation_quaternion = mu.Quaternion(best_axis, math.radians(max_deg * amt)); sb.keyframe_insert("rotation_quaternion", frame=f + 1)
    log(f"scapula_retract {wh_} axis {best_axis} score {best_score:.3f} max {max_deg:.1f} deg")
# Authored clavicle / scapula: apply from rest each frame (adding to sb.location accumulates).
# Retraction is a real medial translation toward the spine (15-25 mm) plus a slight upward
# rotation at peak — visible from a 3/4 camera. Depression/shrug stay along chest-up.
if AUTH_U is not None and AUTH_SPEC is not None:
    _form = AUTH_META["form"]
    _prof = __import__("authored_motion", fromlist=["x"]).deep_merge(
        AUTH_SPEC.get("right") or {}, AUTH_SPEC.get(_form) or {} if _form != "right" else {})
    shrug_m = float(_prof.get("shrug_m") or 0)
    depress_m = float(_prof.get("depress_m") or 0)
    do_retract = bool(_prof.get("scapula_retract", False))
    retract_m = float(_prof.get("scapula_retract_m") or 0.020) if do_retract else 0.0
    retract_back = float(_prof.get("scapula_retract_back_m") or 0.006) if do_retract else 0.0
    up_rot = float(_prof.get("scapula_up_rot_deg") or 8.0) if do_retract else 0.0
    whc = "Left" if side == 11 else "Right"
    if (whc + "Shoulder") in pb and (shrug_m or depress_m or retract_m or retract_back or up_rot):
        sb = pb[whc + "Shoulder"]
        sc.frame_set(1); bpy.context.view_layer.update()
        base_loc = sb.location.copy()
        base_q = sb.rotation_quaternion.copy()
        sh0 = arm.matrix_world @ pb[whc + "Arm"].head
        sp0 = arm.matrix_world @ pb["Spine01"].head
        toward = sp0 - sh0; toward.z = 0.0
        if toward.length < 1e-6: toward = mu.Vector((1.0, 0.0, 0.0))
        toward.normalize()
        tv = np.mean((B[:, 11] + B[:, 12]) / 2 - (B[:, 23] + B[:, 24]) / 2, axis=0)
        fw = mu.Vector((float(tv[0]), float(tv[1]), 0.0))
        if fw.length < 1e-6: fw = mu.Vector((0.0, -1.0, 0.0))
        fw.normalize()
        # pick the forward-axis rotation sign that RAISES Arm.head and prefers moving it toward the spine
        rot_sign = 1.0
        trans_m = retract_m
        if up_rot:
            z0 = float(sh0.z); dxy0 = float(math.hypot(sh0.x - sp0.x, sh0.y - sp0.y))
            best, rot_sign = -1e9, 1.0
            rot_medial = 0.0
            for sgn in (1.0, -1.0):
                sb.rotation_quaternion = mu.Quaternion(fw, math.radians(sgn * up_rot)) @ base_q
                bpy.context.view_layer.update()
                p1 = arm.matrix_world @ pb[whc + "Arm"].head
                dz = float(p1.z) - z0
                dxy1 = float(math.hypot(p1.x - sp0.x, p1.y - sp0.y))
                med = dxy0 - dxy1
                score = dz + 2.0 * med
                if score > best: best, rot_sign, rot_medial = score, sgn, med
            sb.rotation_quaternion = base_q; bpy.context.view_layer.update()
            # keep net medial travel ≈ retract_m after the rotation's own XY swing
            trans_m = retract_m - rot_medial
        log(f"authored scapula rot_sign {rot_sign} trans {trans_m*1000:.1f}mm (target retract {retract_m*1000:.0f}mm)")
        for fi, uu in enumerate(AUTH_U):
            sc.frame_set(fi + 1); bpy.context.view_layer.update()
            uu = float(uu)
            cu = (arm.matrix_world @ pb["neck"].head) - (arm.matrix_world @ pb["Spine01"].head)
            if cu.length < 1e-6: cu = mu.Vector((0.0, 0.0, 1.0))
            else: cu.normalize()
            delta = toward * (trans_m * uu) - fw * (retract_back * uu) + cu * (shrug_m * uu - depress_m * uu)
            if sb.parent:
                parent_mw = arm.matrix_world @ sb.parent.matrix
                delta_local = parent_mw.to_3x3().inverted() @ delta
            else:
                delta_local = arm.matrix_world.to_3x3().inverted() @ delta
            sb.location = base_loc + delta_local
            sb.rotation_quaternion = mu.Quaternion(fw, math.radians(rot_sign * up_rot * uu)) @ base_q
            sb.keyframe_insert("location", frame=fi + 1)
            sb.keyframe_insert("rotation_quaternion", frame=fi + 1)
        log(f"authored scapula retract {retract_m*1000:.0f}mm medial, back {retract_back*1000:.0f}mm, up-rot {up_rot:.1f}deg, depress {depress_m} shrug {shrug_m} on {whc}Shoulder")
    # Snap working-arm empties to the EVALUATED shoulder so DAMPED_TRACK preserves authored
    # angles (the rig shoulder is placed by the spine chain, not by landmark 11/12).
    ppw = whc[0].lower(); La_a, Lf_a = rl(whc + "Arm", whc + "ForeArm"), rl(whc + "ForeArm", whc + "Hand")
    w_sh_i, w_el_i, w_wr_i = (11, 13, 15) if side == 11 else (12, 14, 16)
    for fi in range(NF):
        sc.frame_set(fi + 1); bpy.context.view_layer.update()
        sh_w = np.array(arm.matrix_world @ pb[whc + "Arm"].head)
        hum = B[fi, w_el_i] - B[fi, w_sh_i]
        hn = np.linalg.norm(hum); hum = hum / hn if hn > 1e-9 else np.array([0.0, 0.0, -1.0])
        fore = B[fi, w_wr_i] - B[fi, w_el_i]
        fn = np.linalg.norm(fore); fore = fore / fn if fn > 1e-9 else np.array([0.0, 0.0, -1.0])
        el_w = sh_w + hum * La_a
        wr_w = el_w + fore * Lf_a
        E[ppw + "el"].location = mu.Vector(el_w.tolist()); E[ppw + "el"].keyframe_insert("location", frame=fi + 1)
        E[ppw + "wr"].location = mu.Vector(wr_w.tolist()); E[ppw + "wr"].keyframe_insert("location", frame=fi + 1)
    for _en in (ppw + "el", ppw + "wr"):
        act = E[_en].animation_data.action if E[_en].animation_data else None
        for fc in (act.fcurves if act is not None and hasattr(act, "fcurves") else []):
            for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    log(f"authored arm snapped to evaluated {whc}Arm head, La {La_a:.3f} Lf {Lf_a:.3f}")
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
if LEGS == "bench_row":
    # DAMPED_TRACK of a standing hip onto a +Y torso can leave the figure belly-up.
    # Compare posed height of rest-back vs rest-front verts and roll the hips if needed.
    sc.frame_set(max(1, NF // 2)); bpy.context.view_layer.update()
    _rest = np.empty(len(body.data.vertices) * 3); body.data.vertices.foreach_get("co", _rest)
    _rest = _rest.reshape(-1, 3)
    _dg = bpy.context.evaluated_depsgraph_get(); _eb = body.evaluated_get(_dg); _mm = _eb.to_mesh()
    _posed = np.empty(len(_mm.vertices) * 3); _mm.vertices.foreach_get("co", _posed)
    _Mw = np.array(_eb.matrix_world); _posed = _posed.reshape(-1, 3) @ _Mw[:3, :3].T + _Mw[:3, 3]
    _eb.to_mesh_clear()
    _back = _rest[:, 1] >= np.percentile(_rest[:, 1], 80)
    _front = _rest[:, 1] <= np.percentile(_rest[:, 1], 20)
    _bz, _fz = float(_posed[_back, 2].mean()), float(_posed[_front, 2].mean())
    log(f"bench_row facing probe back_z={_bz:.3f} front_z={_fz:.3f}")
    if _bz < _fz + 0.02:
        pb["Hips"].rotation_quaternion = mu.Quaternion((0, 1, 0), math.pi) @ pb["Hips"].rotation_quaternion.copy()
        bpy.context.view_layer.update()
        log("bench_row: rolled hips 180 deg so the anatomical back faces up")

# ---- rest geometry: muscle regions, shorts, hand vertices, edges ----------------------------------------------------
# Grab3D models a raised cuff into char1 at each wrist (not a separate object).
# Flatten it on the rest mesh before we capture `co` / bake glow / add shapekeys.
try:
    import wrist_cuff as WCUFF
    WCUFF.flatten_wrist_cuffs(body, log)
    if hasattr(WCUFF, "flatten_ankle_cuffs"):
        # Mask only on the source: dissolving here changes vert count and
        # breaks the crease-weight cache / later shapekeys.
        WCUFF.flatten_ankle_cuffs(body, log, dissolve=False)
except Exception as _we:
    log(f"wrist cuff flatten skipped: {_we}")
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
import muscle_ids as MIDS
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_cdir = RC.get("weight_cache_dir") or os.path.join(_root, "output/_rig3d/cache")
try:
    MID = MIDS.load_or_build(C["asset"]["glb"], _cdir)
except Exception as _e:
    log(f"muscle_ids build failed: {_e}"); MID = None
USE_MID = bool(MID) and int(MID["nv"]) == int(nv)
if USE_MID:
    MIDS.write_mid_attribute(me, MID["vert_id"])
    _tgts = list((C.get("grade") or {}).get("target_muscles") or ["lats", "rhomboids", "mid_trap", "rear_delt"])
    ntgt = MIDS.write_mid_target(me, MID["vert_id"], MID["names"], _tgts)
    tid = MID["vert_id"]; name_i = {n: i for i, n in enumerate(MID["names"])}
    reg = np.zeros((nv, 3), np.float32)
    if "lats" in name_i: reg[tid == name_i["lats"], 0] = 1
    if "rhomboids" in name_i: reg[tid == name_i["rhomboids"], 1] = 1
    for n_ in ("mid_trap", "rear_delt"):
        if n_ in name_i: reg[tid == name_i[n_], 2] = 1.0
    reg[shorts] = 0
    MREG.write_mregion(me, reg); REST_REG = reg.copy()
    lats = reg[:, 0] > 0.5; rhomb = reg[:, 1] > 0.5; mid_trap = reg[:, 2] > 0.5
    log(f"muscle IDs hit={MID.get('hit')} tgt_verts={ntgt} { {k: MID['counts'][k] for k in _tgts if k in MID['counts']} }")
    _gw_tgts = list(_tgts)
    if str((C.get("grade") or {}).get("glow") or "right") == "wrong":
        try:
            from muscle_activation import COMPENSATORS as _COMP
            _gw_tgts = list(_COMP)
        except Exception:
            pass
    _hues = dict(MIDS.DEFAULT_HUES)
    _yh = (C.get("grade") or {}).get("muscle_hues") or {}
    if isinstance(_yh, dict):
        _hues.update(_yh)
    _hstr = (C.get("grade") or {}).get("muscle_hue_strength") or {}
    try:
        try:
            _rest = MIDS._read_glb(C["asset"]["glb"])["pos"]
            if len(_rest) != nv:
                raise ValueError(f"rest nv {len(_rest)} != {nv}")
        except Exception as _re:
            _rest = co
            log(f"glow rest-pos fallback: {_re}")
        _shape = str(RC.get("glow_shape") or "hull").lower()
        if _shape in ("shells", "attr", "weld", "weld_attr"):
            log(f"glow_shape={_shape}: skipping projection paint")
            _gw = np.zeros(nv, np.float32)
            _grgb = np.zeros((nv, 3), np.float32)
            _gst = {"shape": _shape}
            try:
                MIDS.write_glow_attrs(me, _gw, _grgb)
            except Exception:
                pass
        elif _shape == "project":
            import project_paint as PPAINT
            _tex = RC.get("glow_proj_tex") or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "masks", "db_row_back_paint.png")
            _tex = _tex if os.path.isabs(_tex) else os.path.join(os.getcwd(), _tex)
            if not os.path.isfile(_tex):
                raise FileNotFoundError("glow_proj_tex missing: " + _tex)
            _ex = np.zeros(nv, dtype=bool)
            if USE_MID and "forearm" in name_i:
                _ex |= (tid == name_i["forearm"])
            _st = PPAINT.write_rest_proj_uvs_obj(body, exclude_mask=_ex)
            log(f"glow rest-proj UVs {_st}")
            _img = bpy.data.images.load(_tex, check_existing=True)
            _img.name = "glow_proj"
            try:
                _img.colorspace_settings.name = "sRGB"
            except Exception:
                pass
            PPAINT.close_proj_image(_img, ksize=int(RC.get("glow_proj_close_px", 5)))
            _gw, _grgb, _gst = PPAINT.sample_uv_texture(me, _img, shorts, EXCLUDE, uv_name="proj")
            log(f"glow proj tex={_tex} (rest-camera UVs, no atlas bake)")
        else:
            _gw, _grgb, _gst = MIDS.bake_target_glow(
                tid, MID["names"], _gw_tgts, _hues, _hstr, ADJ,
                exclude=EXCLUDE, shorts=shorts,
                erode_rings=int(RC.get("glow_erode_rings", 0)),
                power=float(RC.get("glow_center_power", 1.55)),
                fade_rings=int(RC.get("glow_fade_rings", 3)),
                pos=_rest,
                smooth_iters=int(RC.get("glow_smooth_iters", 28)),
                solid_thresh=float(RC.get("glow_solid_thresh", 0.32)),
                close_rings=int(RC.get("glow_close_rings", 3)),
                radial=bool(RC.get("glow_radial", False)),
            )
        _gw_mean, _gw_n = MIDS.write_glow_attrs(me, _gw, _grgb)
        log(f"glow bake verts>0.5={_gw_n} mean={_gw_mean:.4f} per_id={_gst}")
    except Exception as _e:
        log(f"glow bake failed: {_e}")
else:
    log(f"muscle IDs unused nv={nv} built={None if not MID else MID.get('nv')}")
    if str(RC.get("glow_shape") or "hull").lower() in ("shells", "attr", "weld", "weld_attr"):
        log(f"glow_shape={RC.get('glow_shape')} (no MID): projection paint not used")
    elif str(RC.get("glow_shape") or "hull").lower() == "project":
        import project_paint as PPAINT
        _tex = RC.get("glow_proj_tex") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "masks", "db_row_back_paint.png")
        _tex = _tex if os.path.isabs(_tex) else os.path.join(os.getcwd(), _tex)
        _ex = np.zeros(nv, dtype=bool)
        _st = PPAINT.write_rest_proj_uvs_obj(body, exclude_mask=_ex)
        log(f"glow rest-proj UVs (no MID) {_st}")
        _img = bpy.data.images.load(_tex, check_existing=True)
        _img.name = "glow_proj"
        try:
            _img.colorspace_settings.name = "sRGB"
        except Exception:
            pass
        PPAINT.close_proj_image(_img, ksize=int(RC.get("glow_proj_close_px", 5)))
        _gw, _grgb, _gst = PPAINT.sample_uv_texture(me, _img, shorts, EXCLUDE, uv_name="proj")
        _gw_mean, _gw_n = MIDS.write_glow_attrs(me, _gw, _grgb)
        log(f"glow proj tex={_tex} verts>0.5={_gw_n} { _gst }")
    else:
        try:
            _gw = (reg.max(axis=1) > 0.5).astype(np.float32)
            _gw = MIDS.center_weight(_gw > 0.5, ADJ, erode_rings=int(RC.get("glow_erode_rings", 0)),
                                     power=float(RC.get("glow_center_power", 1.2)),
                                     fade_rings=int(RC.get("glow_fade_rings", 4)))
            _gw[shorts] = 0.0
            _gw[EXCLUDE] = 0.0
            _gw_mean, _gw_n = MIDS.write_glow_w(me, _gw)
            log(f"glow_w fallback verts>0.5={_gw_n} mean={_gw_mean:.4f}")
        except Exception as _e:
            log(f"glow_w fallback failed: {_e}")
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
    # Neutral one-arm-row grip from rig axes + the mesh palmar normal (not nearest-of-± handle).
    # The old period-π "handle has no direction" branch flipped the palm 180° whenever the rest
    # across-palm axis was closer to −forward than +forward: wrist looked inverted on the handle.
    # Construction: palmar normal (HG, out of the palm) points MEDIAL (toward Spine01, perp to the
    # forearm). That uniquely sets roll so the palm faces the torso, thumb/radial side points
    # forward, fingers wrap from the outside, knuckles face away from the body. Wrist stays in
    # line with the forearm (WRIST_NEUTRAL already skips Hand DAMPED_TRACK).
    fa_b = hand.replace("Hand", "ForeArm"); share = float(MO.get("grip_twist_forearm_share", 0.6)); TH = np.zeros(NF)
    n_loc = mu.Vector(list(HG["handle_local"]["n"]))
    def _resid(f):
        sc.frame_set(f)
        R_ = (arm.matrix_world @ pb[hand].matrix).to_3x3().normalized()
        u_ = (R_ @ mu.Vector((0, 1, 0))).normalized()
        pn = (R_ @ n_loc).normalized()
        Sp = arm.matrix_world @ pb["Spine01"].head
        Hd = arm.matrix_world @ pb[hand].head
        medial = Sp - Hd
        medial = medial - medial.dot(u_) * u_
        if medial.length < 1e-4:
            other = "LeftArm" if hand.startswith("Right") else "RightArm"
            medial = (arm.matrix_world @ pb[other].head) - (arm.matrix_world @ pb[hand.replace("Hand", "Arm")].head)
            medial = medial - medial.dot(u_) * u_
        if medial.length < 1e-4:
            return 0.0
        medial.normalize()
        return math.atan2(pn.cross(medial).dot(u_), pn.dot(medial))
    for ps in range(3):
        for f in range(1, NF + 1): TH[f - 1] += _resid(f)
        if ps == 0:
            TH = np.unwrap(TH)                                                     # palm normal has a direction: period 2π
            TH = gsmooth(TH[:, None], 1.5)[:, 0]
        for f in range(1, NF + 1):
            pb[fa_b].rotation_quaternion = mu.Quaternion((0, 1, 0), share * TH[f - 1]); pb[hand].rotation_quaternion = mu.Quaternion((0, 1, 0), (1 - share) * TH[f - 1])
            pb[fa_b].keyframe_insert("rotation_quaternion", frame=f); pb[hand].keyframe_insert("rotation_quaternion", frame=f)
    TW = np.degrees(TH).tolist()
    region_counts_twist = [round(min(TW), 1), round(max(TW), 1)]; log(f"neutral grip twist {region_counts_twist} deg (forearm share {share}, palm->medial)")
ev = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get("vertices", ev); ev = ev.reshape(-1, 2)
rest_len = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1); ok_e = rest_len > 1e-4; MED_E = float(np.median(rest_len[ok_e]))
region_counts = {"lats": int(lats.sum()), "rhomboids": int(rhomb.sum()), "mid_trap": int(mid_trap.sum()), "rear_delt": int(rinfo.get("rear_delt", 0)), "shorts": int(shorts.sum()), "hand_verts": int(len(hand_vi))}
log(f"regions {region_counts}")

# ---- black boxer briefs (드로즈): a separate garment mesh copied from the body's own pelvis/upper-thigh faces, pushed out
# along the normals, keeping the body's vertex groups + armature modifier -> skinned exactly like the body (no sim).
GC = C.get("garment", {"enabled": False})
garm, g_src = None, None
SHELL_OBJS, SHELL_FLEX = [], []
WELDED, WELD_STATS, WAISTBAND = None, None, None
GLOW_ATTR_INFO, GARM_META = None, None
_kit = C.get("kit") or {}
_kit_gmethod = str((( _kit.get("garment") or {}).get("method")) or GC.get("method") or "").lower()
_kit_prev_f = sc.frame_current
try:
    if AUTH_U is not None and str(RC.get("glow_shape") or "").lower() == "shells":
        _kit_peak = int(np.argmax(AUTH_U)) + 1
        sc.frame_set(_kit_peak)
        bpy.context.view_layer.update()
        log(f"kit overlays built at peak frame {_kit_peak}")
except Exception:
    pass
if _kit.get("enabled"):
    try:
        import kit.weld as KITW
        WELDED, WELD_STATS = KITW.make_render_copy(body, arm, C, log=log, col=col)
        import kit.glow_attr as KITGA
        _tid = tid if "tid" in dir() else None
        _names = list((MID or {}).get("names") or [])
        GLOW_ATTR_INFO = KITGA.write_weights(
            WELDED, body, _tid, _names, C,
            shorts=shorts if "shorts" in dir() else None,
            exclude=EXCLUDE if "EXCLUDE" in dir() else None,
            log=log)
    except Exception as _we:
        import traceback
        log(f"kit weld/glow_attr failed: {_we}")
        log(traceback.format_exc())
        WELDED = None
if GC.get("enabled") and _kit.get("enabled") and _kit_gmethod in ("loft", "cloth", "closed", "tube"):
    try:
        import kit.garment as KITG
        _collider = WELDED if WELDED is not None else body
        garm, _gmeta = KITG.build_boxer_shorts(body, arm, C, log=log, col=col, collider=_collider)
        WAISTBAND = _gmeta.pop("waistband_obj", None)
        GARM_META = {k: v for k, v in _gmeta.items() if k != "waistband_obj"}
        g_src = None
        gm_ = garm.data
        gev = np.empty(len(gm_.edges) * 2, np.int64)
        gm_.edges.foreach_get("vertices", gev)
        gev = gev.reshape(-1, 2)
        _gco = np.empty(len(gm_.vertices) * 3)
        gm_.vertices.foreach_get("co", _gco)
        _gco = _gco.reshape(-1, 3)
        _Mw = np.array(garm.matrix_world)
        _gw = _gco @ _Mw[:3, :3].T + _Mw[:3, 3]
        g_rest = np.linalg.norm(_gw[gev[:, 0]] - _gw[gev[:, 1]], axis=1)
        g_ok = g_rest > 1e-4
        region_counts.update({"briefs_verts": int(len(gm_.vertices)), "briefs_faces": len(gm_.polygons)})
        log(f"kit garment ok {_gmeta}")
    except Exception as _ge:
        import traceback
        log(f"kit garment failed: {_ge}")
        log(traceback.format_exc())
        garm = None
elif GC.get("enabled"):
    import bmesh
    gz0, gz1, gx = float(GC.get("z_hem", 0.78)), float(GC.get("z_waist", 1.03)), float(GC.get("max_abs_x", 0.22))
    sel = (z >= gz0) & (z <= gz1) & (ax_ <= gx)
    garm = body.copy(); garm.data = body.data.copy(); garm.name = "briefs"; col.objects.link(garm)
    gm_ = garm.data
    for at_ in list(gm_.color_attributes):
        try:
            gm_.color_attributes.remove(at_)
        except RuntimeError:
            pass  # Blender 4.x refuses to drop the last/default colour attribute
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
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.006)
    bmesh.ops.holes_fill(bm, edges=[e_ for e_ in bm.edges if e_.is_boundary], sides=28)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.ensure_lookup_table(); bm.normal_update()
    # Smooth copied muscle grooves into a fabric surface BEFORE the 5 mm offset
    # so rest edge lengths already match the fitted shell (posed Smooth tore edges).
    for _ in range(int(GC.get("smooth_iters", 6))):
        bmesh.ops.smooth_vert(bm, verts=bm.verts[:], factor=float(GC.get("smooth_factor", 0.5)))
    bm.verts.ensure_lookup_table(); bm.normal_update()
    # offset outward (the Meshy-made body has some inward-facing normals: orient each normal away from its limb axis,
    # pelvis = vertical axis through x=0 above the crotch, thighs = x=+-0.09 below it)
    off_local = float(GC.get("offset_m", 0.001)) / float(np.array(body.matrix_world.to_scale()).mean())
    gco = np.array([v_.co for v_ in bm.verts]); lnr_g = np.array([v_.normal for v_ in bm.verts])
    vw = gco @ M4[:3, :3].T + M4[:3, 3]; nw_g = lnr_g @ M4[:3, :3].T; cx_ = np.where((vw[:, 2] < 0.92) & (np.abs(vw[:, 0]) > 0.03), np.sign(vw[:, 0]) * 0.09, 0.0)
    outw = np.stack([vw[:, 0] - cx_, vw[:, 1], np.zeros(len(vw))], 1)
    g_sign = np.where((nw_g * outw).sum(1) < 0, -1.0, 1.0); dn = lnr_g * g_sign[:, None]; n_flip = int((g_sign < 0).sum())
    for i_, v_ in enumerate(bm.verts): v_.co = tuple(gco[i_] + off_local * dn[i_])
    bm.faces.ensure_lookup_table(); bm.normal_update()
    # flip any face whose world normal points toward the pelvis (back-face = white body shows through)
    pelvis_w = np.array([0.0, 0.0, 0.92])
    flip_f = []
    for f_ in bm.faces:
        c_ = np.mean([np.array(v_.co) for v_ in f_.verts], 0)
        cw = c_ @ M4[:3, :3].T + M4[:3, 3]
        nw = np.array(f_.normal) @ M4[:3, :3].T
        if np.dot(nw, cw - pelvis_w) < 0:
            flip_f.append(f_)
    if flip_f:
        bmesh.ops.reverse_faces(bm, faces=flip_f)
    bm.normal_update()
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
    bpy.context.view_layer.objects.active = garm
    # The glTF armature carries a ~0.025 object scale. Shrinkwrap/Displace offsets
    # are in the garment's local units, so convert world metres before assigning.
    _gscale = float(np.array(garm.matrix_world.to_scale()).mean())
    _gscale = _gscale if _gscale > 1e-6 else 1.0
    # Keep the wrap after the armature so the offset sits on the posed surface
    # (imported Armature modifiers are often pinned last).
    for _am in garm.modifiers:
        if _am.type == "ARMATURE" and getattr(_am, "use_pin_to_last", False):
            _am.use_pin_to_last = False
    # Slim visible waistband: a 2–3 cm top strip sits ~2 mm further out than the body.
    try:
        vg_w = garm.vertex_groups.new(name="waistband")
        z_cut = float(GC.get("z_waist", 1.03)) - 0.028
        n_wb = 0
        for i_, z_ in enumerate(vw[:, 2]):
            if z_ >= z_cut:
                vg_w.add([i_], 1.0, "REPLACE"); n_wb += 1
        log(f"waistband verts {n_wb}/{gnv} z>={z_cut:.3f}")
    except Exception as _ve:
        log(f"waistband group skipped: {_ve}")
    # Fitted shell is the rest-pose offset of the pre-smoothed pelvis faces.
    try:
        disp = garm.modifiers.new("briefs_waistband", "DISPLACE")
        disp.mid_level = 0.0
        disp.strength = 0.002 / _gscale
        if "waistband" in garm.vertex_groups:
            disp.vertex_group = "waistband"
        if hasattr(disp, "direction"):
            disp.direction = "NORMAL"
    except Exception as _de:
        log(f"briefs waistband skipped: {_de}")
    sol_t = float(GC.get("solidify_m", 0.0) or 0.0)
    if sol_t > 1e-5:
        try:
            sol = garm.modifiers.new("briefs_solidify", "SOLIDIFY")
            sol.thickness = sol_t / _gscale
            sol.offset = 1.0
            if hasattr(sol, "use_quality_normals"):
                sol.use_quality_normals = True
        except Exception as _se:
            log(f"briefs solidify skipped: {_se}")
    _mod_order = [m.name + ":" + m.type for m in garm.modifiers]
    log(f"briefs garment: {gnv} verts, {len(gm_.polygons)} faces, offset {GC.get('offset_m', 0.001)} m "
        f"wrap {GC.get('wrap_offset_m', 0.004)} m local_scale={_gscale:.4f} mods={_mod_order}")

if _kit.get("enabled") and str(RC.get("glow_shape") or "").lower() == "shells":
    try:
        import kit.muscle_shells as KITS
        _hues = dict(MIDS.DEFAULT_HUES)
        _yh = (C.get("grade") or {}).get("muscle_hues") or {}
        if isinstance(_yh, dict):
            _hues.update(_yh)
        _tid = tid if "tid" in dir() else None
        _names = list((MID or {}).get("names") or [])
        SHELL_OBJS, SHELL_FLEX, _sinfo = KITS.build_overlay_shells(
            body, arm, _tid, _names, _hues, C, adj=ADJ if "ADJ" in dir() else None,
            log=log, col=col)
        log(f"kit muscle shells {_sinfo}")
    except Exception as _se:
        import traceback
        log(f"kit muscle shells failed: {_se}")
        log(traceback.format_exc())
try:
    sc.frame_set(_kit_prev_f)
    bpy.context.view_layer.update()
except Exception:
    pass

if GW.get("enabled"):      # after the garment copy (the briefs never carry the fist key)
    if not me.shape_keys: body.shape_key_add(name="Basis", from_mix=False)
    sk = body.shape_key_add(name="fist", from_mix=False); lc = np.empty(nv * 3); sk.data.foreach_get("co", lc); lc = lc.reshape(-1, 3)
    new_w = HG["new_world"]
    if GRIP_MODE == "palm_neutral":
        # Rebuild the wrap AFTER the palm→medial roll. The rest-pose wrap gave n_loc for the
        # twist, but baking it as a shapekey and then rolling the forearm 55–118° sheared the
        # fist through the handle (~170 intersecting verts). Evaluate the open posed hand,
        # wrap in that frame, convert bone-local wrapped verts back through the REST bone
        # matrix into mesh-local shapekey deltas (share=1.0 ⇒ bone-local shape is nearly constant).
        sc.frame_set(1); bpy.context.view_layer.update()
        _dg = bpy.context.evaluated_depsgraph_get()
        _eb = body.evaluated_get(_dg); _m = _eb.to_mesh()
        _p = np.empty(len(_m.vertices) * 3); _m.vertices.foreach_get("co", _p); _p = _p.reshape(-1, 3)
        _Mw = np.array(_eb.matrix_world)
        _pw = _p @ _Mw[:3, :3].T + _Mw[:3, 3]
        _eb.to_mesh_clear()
        Mh_p = np.array(arm.matrix_world @ pb[hand].matrix)
        Mh_p[:3, :3] /= np.linalg.norm(Mh_p[:3, :3], axis=0)
        HG = HGR.build(_pw, HIDX, Mh_p, hand.startswith("Left"),
                       r=float(GW.get("handle_r_m", 0.016)), clear=float(GW.get("clear_m", 0.002)),
                       max_wrap_deg=float(GW.get("max_wrap_deg", 250)), mcp_back=float(GW.get("mcp_back_m", 0.015)))
        PALM[hand] = np.array(HG["handle_local"]["axis"])
        log(f"grip wrap after roll: {HG['info']}")
        Rp, op = Mh_p[:3, :3], Mh_p[:3, 3]
        newL = (HG["new_world"] - op) @ Rp
        new_w = newL @ _Mh[:3, :3].T + _Mh[:3, 3]
    Minv = np.linalg.inv(M4)
    new_w = np.asarray(new_w, float)
    y_loc = HG.get("y_local")
    if y_loc is not None and len(y_loc) == len(HIDX):
        y_lo = float(np.percentile(y_loc, 6)); y_hi = y_lo + 0.060
        ramp = np.clip((y_loc - y_lo) / max(y_hi - y_lo, 1e-4), 0.0, 1.0)
        new_w = co[HIDX] + (new_w - co[HIDX]) * ramp[:, None]
    lc[HIDX] = new_w @ Minv[:3, :3].T + Minv[:3, 3]
    # UV-split twins at the wrist share a rest location but not an index; copy
    # the faded fist so one loop cannot peel off and read as a white ring.
    key = np.round(co, 5); pos = {tuple(key[vi]): lc[vi].copy() for vi in HIDX}
    n_prop = 0
    for i in range(nv):
        hit = pos.get(tuple(key[i]))
        if hit is not None and not np.allclose(lc[i], hit, atol=1e-7):
            lc[i] = hit; n_prop += 1
    log(f"fist wrist fade, synced {n_prop} coincident verts")
    sk.data.foreach_set("co", lc.ravel()); sk.value = 1.0
    # The white wrist ring is a split-normal / sharp-edge highlight at the
    # hand–forearm UV seam. Clear sharp edges so it does not catch a white rim.
    try:
        for e in me.edges:
            e.use_edge_sharp = False
        if hasattr(me, "shade_smooth"):
            me.shade_smooth()
        elif hasattr(body, "shade_smooth"):
            body.data.polygons.foreach_set("use_smooth", [True] * len(body.data.polygons))
        log("wrist: cleared sharp edges, shade smooth")
    except Exception as _se:
        log(f"shade smooth skipped: {_se}")
# Support hand on the bench pad: no finger bones, so flatten via a geometric shape key
# (fingers adducted, palm planar) rather than leaving the rest-pose splay.
SUPPORT_BN = "LeftHand" if not str(hand).startswith("Left") else "RightHand"
SIDX = None
if LEGS == "bench_row":
    try:
        _sgi = body.vertex_groups[SUPPORT_BN].index
        _sW = np.zeros(nv)
        for v in me.vertices:
            for g in v.groups:
                if g.group == _sgi:
                    _sW[v.index] = g.weight
        SIDX = np.nonzero(_sW > 0.5)[0]
        _Ms = np.array(arm.matrix_world @ arm.data.bones[SUPPORT_BN].matrix_local)
        _Ms[:3, :3] /= np.linalg.norm(_Ms[:3, :3], axis=0)
        FP = HGR.flatten_palm(co, SIDX, _Ms, SUPPORT_BN.startswith("Left"),
                              adduct=0.78, thumb_adduct=0.50, flatten=0.55)
        if not me.shape_keys:
            body.shape_key_add(name="Basis", from_mix=False)
        sks = body.shape_key_add(name="support_flat", from_mix=False)
        lcs = np.empty(nv * 3); sks.data.foreach_get("co", lcs); lcs = lcs.reshape(-1, 3)
        Minv = np.linalg.inv(M4)
        lcs[SIDX] = FP["new_world"] @ Minv[:3, :3].T + Minv[:3, 3]
        sks.data.foreach_set("co", lcs.ravel()); sks.value = 1.0
        log(f"support palm flatten: {FP['info']}")
    except Exception as _fe:
        log(f"support palm flatten skipped: {_fe}")
if WELDED is not None:
    try:
        import kit.weld as KITW
        KITW.apply_source_shapekey_mix(body, WELDED, log=log)
    except Exception as _se:
        log(f"kit weld shapekey transfer: {_se}")
    # Re-apply ankle flatten on the mesh that actually renders. The source
    # flatten can be lost when shapekeys / weld reset the basis.
    try:
        import wrist_cuff as WCUFF
        if hasattr(WCUFF, "flatten_ankle_cuffs"):
            log("ankle cuff: re-flatten on welded render mesh")
            WCUFF.flatten_ankle_cuffs(WELDED, log, dissolve=True)
    except Exception as _ae:
        log(f"welded ankle flatten skipped: {_ae}")
# ---- dumbbell ---------------------------------------------------------------------------------------------------------
def db_plate(xx):
    # Round gunmetal plates (32-gon) so the weight reads as a metal disc, not a blue cube.
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.055, depth=0.034, location=(xx, 0, 0),
                                        rotation=(0, math.pi / 2, 0))
    return bpy.context.object
h1, h2 = db_plate(-0.118), db_plate(0.118)
_knurl = bool(((_kit if "_kit" in dir() else {}).get("dumbbell") or {}).get("knurl") or (C.get("kit") or {}).get("dumbbell", {}).get("knurl"))
_hv = 48 if _knurl else 24
bpy.ops.mesh.primitive_cylinder_add(vertices=_hv, radius=0.016, depth=0.22, rotation=(0, math.pi / 2, 0)); hd = bpy.context.object
if _knurl:
    try:
        import kit.dumbbell as KITD
        _kd = (C.get("kit") or {}).get("dumbbell") or {}
        _kinfo = KITD.add_handle_knurl(
            hd, axis=(0.0, 0.0, 1.0),
            amp=float(_kd.get("knurl_amp_m", 0.0007)),
            around=int(_kd.get("knurl_around", 14)),
            along=int(_kd.get("knurl_along", 22)))
        log(f"kit dumbbell knurl {_kinfo}")
    except Exception as _ke:
        log(f"kit knurl failed: {_ke}")
# slim collars so the plate-to-handle junction reads as a real hex/round dumbbell
def db_collar(xx):
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.020, depth=0.012, location=(xx, 0, 0),
                                        rotation=(0, math.pi / 2, 0))
    return bpy.context.object
c1, c2 = db_collar(-0.098), db_collar(0.098)
def mat(n, rgb, metal=0.0, rough=0.6):
    m_ = bpy.data.materials.new(n); m_.use_nodes = True; bs = m_.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*rgb, 1); bs.inputs["Metallic"].default_value = metal; bs.inputs["Roughness"].default_value = rough; return m_
gun, steel = mat("gunmetal", (0.12, 0.125, 0.13), 1.0, 0.52), mat("steel", (0.50, 0.51, 0.53), 1.0, 0.45)
for o in (h1, h2): o.data.materials.append(gun)
for o in (c1, c2, hd): o.data.materials.append(steel)
bpy.ops.object.select_all(action="DESELECT")
for o in (h1, h2, c1, c2, hd): o.select_set(True)
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
interp = "BEZIER" if AUTH_SPEC is not None else "LINEAR"
for fc in (db.animation_data.action.fcurves if hasattr(db.animation_data.action, "fcurves") else []):
    for kp in fc.keyframe_points: kp.interpolation = interp
if AUTH_SPEC is not None:
    lag = float(AUTH_SPEC.get("db_pendulum_lag_s") or 0)
    tilt = float(AUTH_SPEC.get("db_pendulum_tilt_deg") or 0)
    if lag > 0 or tilt > 0:
        delay = int(round(lag * float(AUTH_META["fps"])))
        locs, rots = [], []
        for f in range(1, NF + 1):
            sc.frame_set(f)
            locs.append(db.location.copy()); rots.append(db.rotation_quaternion.copy())
        Lz = np.array([p.z for p in locs], dtype=np.float64)
        az = np.gradient(np.gradient(Lz)) if len(Lz) > 2 else np.zeros(len(Lz))
        scale = float(np.percentile(np.abs(az), 90) or 1.0)
        az = np.clip(az / scale, -1.0, 1.0)
        for i in range(NF):
            # A delayed world rotation tilted the handle axis up to ~15° through a
            # static fist shapekey (170 intersecting verts at mid-concentric). The
            # handle centre already follows the palm; keep the handle AXIS locked to
            # the current grip frame and apply any pendulum only as a spin about
            # that axis (circular handle occupancy unchanged).
            ha = rots[i] @ mu.Vector(DB_AX)
            if ha.length < 1e-8:
                ha = mu.Vector((1.0, 0.0, 0.0))
            extra = mu.Quaternion(ha.normalized(), math.radians(tilt * float(az[i])))
            db.location = locs[i]
            db.rotation_quaternion = extra @ rots[i]
            db.keyframe_insert("location", frame=i + 1)
            db.keyframe_insert("rotation_quaternion", frame=i + 1)
        for fc in (db.animation_data.action.fcurves if hasattr(db.animation_data.action, "fcurves") else []):
            for kp in fc.keyframe_points: kp.interpolation = "BEZIER"
        log(f"db pendulum lag {lag}s tilt {tilt}deg")
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
# yaw_back_deg is "how far from a pure working-side view toward the back":
# 0 = true lateral (camera on the working side, perpendicular to the sagittal plane),
# 90 = straight behind the figure. Production 3/4 is ~48. elev 0 = horizon level.
_view = str(CAMC.get("view") or "").lower()
if _view in ("lateral", "side", "side_profile", "true_side"):
    yaw = el = 0.0
    log("camera view=lateral: yaw_back 0 elev 0 (perp to sagittal, working side, horizon level)")
else:
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
Dd = hi
_h = float(CAMC.get("height_m", 0.0)); _s = float(CAMC.get("shift_m", 0.0))
if _view in ("lateral", "side", "side_profile", "true_side"):
    _h = _s = 0.0
CAM.location = ctr - fwd * Dd + side_dir * _s + mu.Vector((0.0, 0.0, _h))
CAM.rotation_euler = (ctr - CAM.location).to_track_quat("-Z", "Y").to_euler()
ny = py / (Dd + pz) * kf; nx = px / (Dd + pz) * kf * asp
camd.shift_x = float((nx.max() + nx.min()) / 2) / 2 / asp       # shift is in units of the larger (vertical) dimension
want_low = 1 - 2 * feet_y; camd.shift_y = float((ny.min() - want_low) / 2)
bpy.context.view_layer.update()
pr = np.array([world_to_camera_view(sc, CAM, mu.Vector(p_.tolist()))[:2] for p_ in P[::5]])
framing = {"x_min": round(float(pr[:, 0].min()), 3), "x_max": round(float(pr[:, 0].max()), 3), "y_top": round(float(1 - pr[:, 1].max()), 3),
           "y_feet": round(float(1 - pr[:, 1].min()), 3), "dist_m": round(Dd, 2), "lens": camd.lens}
log(f"camera {framing}")

def _eval_world_verts(obj, step=1):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    mm = ev.to_mesh()
    n = len(mm.vertices)
    co = np.empty(n * 3, np.float64)
    mm.vertices.foreach_get("co", co)
    Mw = np.array(ev.matrix_world, dtype=np.float64)
    ev.to_mesh_clear()
    pts = co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
    return pts[::max(1, int(step))]


def _framing_point_sets():
    """World-space must-include points and figure+bench fill points at the current frame."""
    src = WELDED if WELDED is not None else body
    body_pts = _eval_world_verts(src, step=6)
    fill_bits = [body_pts]
    for o in list(bpy.data.objects):
        if o.type != "MESH" or o == src or o == db:
            continue
        nm = (o.name or "").lower()
        if any(k in nm for k in ("bench", "pad", "seat", "frame")):
            try:
                fill_bits.append(_eval_world_verts(o, step=3))
            except Exception:
                pass
    try:
        db_pts = _eval_world_verts(db, step=1)
    except Exception:
        db_pts = np.array(db.matrix_world.translation).reshape(1, 3)
    fill_bits.append(db_pts)

    def _bone(name):
        if name not in pb:
            return None
        return np.array(arm.matrix_world @ pb[name].head, dtype=np.float64)

    extras = []
    head = _bone("Head")
    if head is not None:
        extras.append(head)
        for d in ((0, 0, 0.09), (0, 0.08, 0.02), (0, -0.05, 0.02),
                  (0.06, 0, 0.02), (-0.06, 0, 0.02)):
            extras.append(head + np.array(d, dtype=np.float64))
    kn = _bone("LeftFoot")
    if kn is not None:
        extras.append(kn)
    for tn in ("LeftToeBase", "LeftToe", "LeftFootToeBase"):
        t = _bone(tn)
        if t is not None:
            extras.append(t)
            extras.append(t + np.array((0.0, -0.04, 0.0)))
            break
    if kn is not None:
        for d in ((0, -0.05, 0.0), (0.04, 0, 0.02), (-0.04, 0, 0.02), (0, 0, 0.04)):
            extras.append(kn + np.array(d, dtype=np.float64))
    must = np.vstack([np.stack(extras, 0), db_pts]) if extras else db_pts
    fill = np.vstack(fill_bits)
    return must, fill


def _project_pts(pts):
    ndc = np.array([world_to_camera_view(sc, CAM, mu.Vector(p.tolist()))
                    for p in pts], dtype=np.float64)
    return ndc


def fit_yaw_elev_cam(yaw_deg, elev_deg, look, margin=0.03, lens=50.0):
    """Smallest dist + centering shift that keeps figure+bench inside `margin`."""
    look_v = [float(x) for x in look]
    must, fill = _framing_point_sets()
    asp = float(sc.render.resolution_y) / max(float(sc.render.resolution_x), 1.0)

    def _place(dist, sx, sy):
        apply_fixed_cam({
            "yaw_back_deg": float(yaw_deg),
            "elev_deg": float(elev_deg),
            "dist_m": float(dist),
            "look": look_v,
            "lens_mm": float(lens),
            "shift_x": float(sx),
            "shift_y": float(sy),
            "up": "world",
        })

    def _center_shift(dist):
        _place(dist, 0.0, 0.0)
        ndc = _project_pts(fill)
        front = ndc[:, 2] > 1e-4
        if int(front.sum()) < 8:
            front = np.ones(len(ndc), dtype=bool)
        xs, ys = ndc[front, 0], ndc[front, 1]
        cx = 0.5 * (float(xs.min()) + float(xs.max()))
        cy = 0.5 * (float(ys.min()) + float(ys.max()))
        # VERTICAL sensor_fit: shift is in units of sensor height.
        sx = (cx - 0.5) / max(asp, 1e-6)
        sy = (cy - 0.5)
        return sx, sy

    def _metrics(dist, sx, sy):
        _place(dist, sx, sy)
        ndc_f = _project_pts(fill)
        ndc_m = _project_pts(must)
        ff = ndc_f[:, 2] > 1e-4
        mm = ndc_m[:, 2] > 1e-4
        if int(ff.sum()) < 8:
            ff = np.ones(len(ndc_f), dtype=bool)
        if int(mm.sum()) < 1:
            mm = np.ones(len(ndc_m), dtype=bool)
        xf, yf = ndc_f[ff, 0], ndc_f[ff, 1]
        xm, ym = ndc_m[mm, 0], ndc_m[mm, 1]
        info = {
            "x_min": float(xf.min()), "x_max": float(xf.max()),
            "y_min": float(yf.min()), "y_max": float(yf.max()),
            "must_x_min": float(xm.min()), "must_x_max": float(xm.max()),
            "must_y_min": float(ym.min()), "must_y_max": float(ym.max()),
        }
        info["fill_w"] = info["x_max"] - info["x_min"]
        info["fill_h"] = info["y_max"] - info["y_min"]
        info["margin_l"] = info["must_x_min"]
        info["margin_r"] = 1.0 - info["must_x_max"]
        info["margin_b"] = info["must_y_min"]
        info["margin_t"] = 1.0 - info["must_y_max"]
        # Also require the whole figure+bench to stay on frame.
        info["ok"] = (
            info["must_x_min"] >= margin - 1e-4 and
            info["must_x_max"] <= 1.0 - margin + 1e-4 and
            info["must_y_min"] >= margin - 1e-4 and
            info["must_y_max"] <= 1.0 - margin + 1e-4 and
            info["x_min"] >= 0.01 and info["x_max"] <= 0.99 and
            info["y_min"] >= 0.01 and info["y_max"] <= 0.99
        )
        return info

    lo, hi = 1.15, 5.20
    best = None
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        sx, sy = _center_shift(mid)
        met = _metrics(mid, sx, sy)
        met.update({"dist_m": mid, "shift_x": sx, "shift_y": sy,
                    "yaw_back_deg": float(yaw_deg), "elev_deg": float(elev_deg)})
        if met["ok"]:
            best = met
            hi = mid
        else:
            lo = mid
    if best is None:
        sx, sy = _center_shift(hi)
        best = _metrics(hi, sx, sy)
        best.update({"dist_m": hi, "shift_x": sx, "shift_y": sy,
                     "yaw_back_deg": float(yaw_deg), "elev_deg": float(elev_deg)})
    # Snap onto the winning camera.
    apply_fixed_cam({
        "yaw_back_deg": float(yaw_deg), "elev_deg": float(elev_deg),
        "dist_m": float(best["dist_m"]), "look": look_v, "lens_mm": float(lens),
        "shift_x": float(best["shift_x"]), "shift_y": float(best["shift_y"]),
        "up": "world",
    })
    best["look"] = [round(float(x), 3) for x in look_v]
    best["dist_m"] = round(float(best["dist_m"]), 3)
    best["shift_x"] = round(float(best["shift_x"]), 4)
    best["shift_y"] = round(float(best["shift_y"]), 4)
    for k in ("fill_w", "fill_h", "margin_l", "margin_r", "margin_b", "margin_t",
              "x_min", "x_max", "y_min", "y_max"):
        best[k] = round(float(best[k]), 4)
    return best


def dump_bone_proj(path):
    """Projected peak-frame bone positions for sheet crops (NDC + pixels)."""
    names = [
        "Hips", "Spine", "Spine01", "Spine02", "Neck", "Head",
        "LeftShoulder", "RightShoulder", "LeftArm", "RightArm",
        "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg",
        "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase",
        "LeftHand", "RightHand", "LeftForeArm", "RightForeArm",
    ]
    out = {"res": [int(sc.render.resolution_x), int(sc.render.resolution_y)],
           "ndc": {}, "px": {}}
    W, H = out["res"]
    for n in names:
        if n not in pb:
            continue
        w = arm.matrix_world @ pb[n].head
        ndc = world_to_camera_view(sc, CAM, w)
        out["ndc"][n] = [round(float(ndc.x), 4), round(float(ndc.y), 4),
                         round(float(ndc.z), 4)]
        out["px"][n] = [int(round(float(ndc.x) * W)),
                        int(round((1.0 - float(ndc.y)) * H))]
    try:
        w = db.matrix_world.translation
        ndc = world_to_camera_view(sc, CAM, w)
        out["ndc"]["Dumbbell"] = [round(float(ndc.x), 4), round(float(ndc.y), 4),
                                  round(float(ndc.z), 4)]
        out["px"]["Dumbbell"] = [int(round(float(ndc.x) * W)),
                                 int(round((1.0 - float(ndc.y)) * H))]
    except Exception:
        pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    log(f"bone proj -> {path} n={len(out['px'])}")
    return out


def render_fitted_cam_previews(out_dir):
    """Cheap peak stills of 4 foreshortening candidates, each fitted for fill."""
    cands = [
        {"yaw": 32, "elev": 32, "tag": "moderate 3/4"},
        {"yaw": 40, "elev": 36, "tag": "stronger 3/4"},
        {"yaw": 48, "elev": 38, "tag": "production 3/4"},
        {"yaw": 42, "elev": 44, "tag": "higher elev"},
    ]
    look = list(CAMC.get("look") or [0.18, 0.20, 0.70])
    peak_f = 25
    try:
        if AUTH_U is not None:
            peak_f = int(np.argmax(AUTH_U)) + 1
    except Exception:
        pass
    sc.frame_set(peak_f)
    bpy.context.view_layer.update()
    try:
        update_regions(peak_f)
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    _orx, _ory = sc.render.resolution_x, sc.render.resolution_y
    _ospp = int(getattr(sc.cycles, "samples", 8) or 8)
    _oden = bool(getattr(sc.cycles, "use_denoising", False))
    sc.render.resolution_x, sc.render.resolution_y = 540, 960
    sc.cycles.samples = 8
    sc.cycles.use_denoising = False
    results = []
    for c in cands:
        met = fit_yaw_elev_cam(c["yaw"], c["elev"], look, margin=0.03)
        met["tag"] = c["tag"]
        path = os.path.join(out_dir, f"yaw{c['yaw']}_el{c['elev']}.png")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        met["path"] = path
        results.append(met)
        log(f"cam-preview yaw={c['yaw']} elev={c['elev']} dist={met['dist_m']} "
            f"shift=({met['shift_x']},{met['shift_y']}) "
            f"fill_w={met['fill_w']} fill_h={met['fill_h']} ok={met['ok']} "
            f"must_m=({met['margin_l']},{met['margin_r']},{met['margin_b']},{met['margin_t']})")
    sc.render.resolution_x, sc.render.resolution_y = _orx, _ory
    sc.cycles.samples = _ospp
    sc.cycles.use_denoising = _oden
    json.dump(results, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    return results

def apply_fixed_cam(cfg_cam):
    look_v = mu.Vector([float(x) for x in (cfg_cam.get("look") or [0.18, 0.18, 0.78])])
    dist = float(cfg_cam.get("dist_m", 2.65))
    yaw = math.radians(float(cfg_cam.get("yaw_back_deg", 8)))
    el = math.radians(float(cfg_cam.get("elev_deg", 8)))
    dirv = mu.Vector((math.cos(yaw) * math.cos(el),
                      -math.sin(yaw) * math.cos(el),
                      math.sin(el))).normalized()
    CAM.location = look_v + dirv * dist
    aim = look_v - CAM.location
    up_mode = str(cfg_cam.get("up") or "world").lower()
    if up_mode in ("head", "head_up", "spine"):
        # Roll so image-up points toward world +Y (the head). Default look-at
        # keeps world +Z as up, which lays a face-down rower sideways in 9:16.
        fwd = aim.normalized()
        prefer = mu.Vector((0.0, 1.0, 0.0))
        cy = prefer - fwd * prefer.dot(fwd)
        if cy.length < 1e-4:
            cy = mu.Vector((0.0, 0.0, 1.0))
        cy.normalize()
        cx = fwd.cross(cy).normalized()
        cy = (-fwd).cross(cx).normalized()
        R = mu.Matrix((cx, cy, -fwd)).transposed()
        CAM.matrix_world = mu.Matrix.Translation(CAM.location) @ R.to_4x4()
    else:
        CAM.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
    camd.lens = float(cfg_cam.get("lens_mm", 50))
    camd.shift_x = float(cfg_cam.get("shift_x", 0.0))
    camd.shift_y = float(cfg_cam.get("shift_y", 0.02))
    bpy.context.view_layer.update()
    info = {"mode": "fixed",
            "yaw_back_deg": float(cfg_cam.get("yaw_back_deg", 8)),
            "elev_deg": float(cfg_cam.get("elev_deg", 8)),
            "dist_m": round(dist, 2),
            "look": [round(float(x), 3) for x in look_v],
            "shift_x": camd.shift_x, "shift_y": camd.shift_y, "up": up_mode}
    if len(P):
        pr_ = np.array([world_to_camera_view(sc, CAM, mu.Vector(p_.tolist()))[:2] for p_ in P[::5]])
        info.update({"x_min": round(float(pr_[:, 0].min()), 3), "x_max": round(float(pr_[:, 0].max()), 3),
                     "y_top": round(float(1 - pr_[:, 1].max()), 3),
                     "y_feet": round(float(1 - pr_[:, 1].min()), 3)})
    return info

if str(CAMC.get("mode") or "").lower() == "fixed":
    # Locked per-exercise world preset (no auto-frame, no binary-search).
    framing.update(apply_fixed_cam(CAMC))
    try:
        json.dump(framing, open(a.out + "/camera_fixed.json", "w"), indent=2)
    except Exception:
        pass
    log(f"camera FIXED yaw={framing['yaw_back_deg']} elev={framing['elev_deg']} dist={framing['dist_m']:.2f} look={framing['look']} {framing}")
elif LEGS == "bench_row":
    # Back 3/4 from the working side and slightly above, from the HEAD end
    # so the glowing back fills the upper middle of the 9:16 frame.
    peak_f = 25 if NF >= 25 else max(1, NF // 2)
    if AUTH_U is not None:
        peak_f = int(np.argmax(AUTH_U)) + 1
    sc.frame_set(peak_f); bpy.context.view_layer.update()
    hipc = (arm.matrix_world @ pb["Hips"].head + arm.matrix_world @ pb["Spine01"].head) / 2
    shc = (arm.matrix_world @ pb["LeftArm"].head + arm.matrix_world @ pb["RightArm"].head) / 2
    hdc = arm.matrix_world @ pb["Head"].head
    extra = [hipc, shc, hdc,
             arm.matrix_world @ pb["LeftHand"].head,
             arm.matrix_world @ pb["RightHand"].head,
             arm.matrix_world @ pb["LeftFoot"].head,
             arm.matrix_world @ pb["RightFoot"].head,
             arm.matrix_world @ pb["LeftLeg"].head,
             arm.matrix_world @ pb["RightLeg"].head,
             arm.matrix_world @ pb["LeftUpLeg"].head,
             arm.matrix_world @ pb["RightUpLeg"].head,
             db.matrix_world.translation]
    for wy, wz in ((-0.62, 0.0), (-0.62, 0.49), (-0.10, 0.49),
                   (0.45, 0.49), (0.62, 0.0), (0.62, 0.49)):
        extra.append(mu.Vector((-0.15, wy, wz)))
        extra.append(mu.Vector((0.15, wy, wz)))
    pts = np.array([tuple(v) for v in extra], np.float64)
    ctr = mu.Vector(((pts.min(0) + pts.max(0)) / 2).tolist())
    # Blend cloud centre with the upper back. Aiming only at the back forces
    # a zoom-out to keep the floor corners in frame and shrinks the figure.
    look_v = ctr * 0.45 + (shc * 0.70 + hipc * 0.30) * 0.55
    dirv = mu.Vector((1.05, 0.90, 0.95)).normalized()
    def _place(Dd):
        CAM.location = look_v + dirv * Dd
        CAM.rotation_euler = (look_v - CAM.location).to_track_quat("-Z", "Y").to_euler()
        camd.shift_x = 0.0
        camd.shift_y = 0.02
        bpy.context.view_layer.update()
        return np.array([list(world_to_camera_view(sc, CAM, mu.Vector(p_.tolist()))) for p_ in pts])
    lo, hi = 1.8, 4.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        nd = _place(mid)
        xs, ys, zs = nd[:, 0], nd[:, 1], nd[:, 2]
        ok = bool(np.all(zs > 0.08) and xs.min() > 0.07 and xs.max() < 0.93
                  and ys.min() > 0.08 and ys.max() < 0.90)
        if ok: hi = mid
        else: lo = mid
    nd = _place(hi)
    qsh = world_to_camera_view(sc, CAM, shc)
    qdb = world_to_camera_view(sc, CAM, db.matrix_world.translation)
    framing.update({"yaw_back_deg": 41, "elev_deg": 36, "dist_m": round(float(hi), 2),
                    "bench_row_cam": True, "from": "head_end_working_side",
                    "x_min": round(float(nd[:, 0].min()), 3), "x_max": round(float(nd[:, 0].max()), 3),
                    "y_top": round(float(1 - nd[:, 1].max()), 3), "y_feet": round(float(1 - nd[:, 1].min()), 3),
                    "back_ndc": [round(qsh.x, 3), round(qsh.y, 3)],
                    "db_ndc": [round(qdb.x, 3), round(qdb.y, 3)]})
    log(f"camera bench_row 3/4 dist={hi:.2f} {framing}")

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

# ---- v3.4 camera search: maximise peak back-sheet connectivity under db-head + support-hand + callout constraints
if bool(CAMC.get("search", False)) and (USE_MID if "USE_MID" in dir() else False):
    wh_ = "Left" if side == 11 else "Right"; ppw = wh_[0].lower(); sp_ = "Right" if side == 11 else "Left"
    peak_f, best_el = 1, 1e9
    for f in range(NF):
        sc.frame_set(f + 1)
        S = arm.matrix_world @ pb[wh_ + "Arm"].head; El = E[ppw + "el"].location; Wr = E[ppw + "wr"].location
        ang = math.degrees((S - El).angle(Wr - El)) if (S - El).length > 1e-6 and (Wr - El).length > 1e-6 else 180.0
        if ang < best_el:
            best_el, peak_f = ang, f + 1
    sc.frame_set(peak_f); bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get(); eb = body.evaluated_get(dg); m_ = eb.to_mesh()
    cpeak = np.empty(len(m_.vertices) * 3); m_.vertices.foreach_get("co", cpeak)
    Mw = np.array(eb.matrix_world); cpeak = cpeak.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]; eb.to_mesh_clear()
    name_i = {n: i for i, n in enumerate(MID["names"])}
    back = np.isin(MID["vert_id"], [name_i[n] for n in ("lats", "rhomboids", "mid_trap") if n in name_i])
    bidx = np.nonzero(back)[0]
    if len(bidx) > 1400:
        bidx = bidx[::max(1, len(bidx) // 1400)]
    sheet_pts = cpeak[bidx]
    def _cc_frac(flags, ndc):
        gh, gw = 140, 80
        grid = np.zeros((gh, gw), np.uint8)
        for (x, y), v in zip(ndc, flags):
            if not v: continue
            ix = int(np.clip(x * gw, 0, gw - 1)); iy = int(np.clip((1.0 - y) * gh, 0, gh - 1))
            grid[iy, ix] = 1
        total = int(grid.sum())
        if total == 0: return 0.0, 0
        seen = np.zeros_like(grid); best = 0
        for y in range(gh):
            for x in range(gw):
                if grid[y, x] == 0 or seen[y, x]: continue
                stack = [(y, x)]; seen[y, x] = 1; n = 0
                while stack:
                    cy, cx = stack.pop(); n += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = 1; stack.append((ny, nx))
                if n > best: best = n
        return best / total, total
    def _place(yaw_deg, elev_deg, height_m, shift_m):
        yaw, el = math.radians(yaw_deg), math.radians(elev_deg)
        dh = (side_dir * math.cos(yaw) - fw * math.sin(yaw)).normalized()
        dirv = (dh * math.cos(el) + mu.Vector((0, 0, 1)) * math.sin(el)).normalized()
        look = ctr + side_dir * float(shift_m)
        fwd_ = -dirv; right_ = fwd_.cross(mu.Vector((0, 0, 1))).normalized(); up_ = right_.cross(fwd_)
        rel = P - np.array(look); px_ = rel @ np.array(right_); py_ = rel @ np.array(up_); pz_ = rel @ np.array(fwd_)
        def ext(Dd):
            nx = px_ / (Dd + pz_) * kf * asp; ny = py_ / (Dd + pz_) * kf
            return (nx.max() - nx.min()) / 2, (ny.max() - ny.min()) / 2
        lo, hi = 0.5, 30.0
        for _ in range(50):
            mid = (lo + hi) / 2; wx, wy = ext(mid)
            if wx > fill_w or wy > (feet_y - top_y): lo = mid
            else: hi = mid
        Dd = hi
        CAM.location = look - fwd_ * Dd + side_dir * float(shift_m) * 0.15 + mu.Vector((0.0, 0.0, float(height_m)))
        aim = (look - CAM.location).normalized()
        CAM.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
        ny = py_ / (Dd + pz_) * kf; nx = px_ / (Dd + pz_) * kf * asp
        camd.shift_x = float((nx.max() + nx.min()) / 2) / 2 / asp
        want_low = 1 - 2 * feet_y; camd.shift_y = float((ny.min() - want_low) / 2)
        bpy.context.view_layer.update()
        return Dd
    yaws = list(CAMC.get("search_yaw_deg") or [32, 44, 56, 68, 80])
    elevs = list(CAMC.get("search_elev_deg") or [14, 22, 30, 38])
    heights = list(CAMC.get("search_height_m") or [0.0, 0.12])
    shifts = list(CAMC.get("search_shift_m") or [0.0, 0.08])
    # seed first so a failed search still keeps the yaml camera
    seed = (float(CAMC.get("yaw_back_deg", 35)), float(CAMC.get("elev_deg", 16)), 0.0, 0.0)
    cands = [seed] + [(y, e, h, s) for y in yaws for e in elevs for h in heights for s in shifts]
    # unique
    seen_c, ordered = set(), []
    for c in cands:
        key = tuple(round(float(x), 4) for x in c)
        if key not in seen_c:
            seen_c.add(key); ordered.append(c)
    results = []
    sc.frame_set(peak_f); bpy.context.view_layer.update()
    for (yaw_d, el_d, ht, sh) in ordered:
        _place(yaw_d, el_d, ht, sh)
        ndc, flags = [], []
        dg = bpy.context.evaluated_depsgraph_get()
        for p_ in sheet_pts:
            v = mu.Vector(p_.tolist()); q = world_to_camera_view(sc, CAM, v)
            inf = (0.02 < q.x < 0.98) and (0.02 < q.y < 0.98) and q.z > 0
            if not inf:
                flags.append(False); ndc.append((q.x, q.y)); continue
            d_ = v - CAM.location; L_ = d_.length
            hit, loc_, nrm_, ix_, ob_, mt_ = sc.ray_cast(dg, CAM.location, d_.normalized(), distance=max(L_ - 0.004, 0.0))
            ok = (not hit) or (ob_ is not None and ob_.name == "dumbbell")
            flags.append(bool(ok)); ndc.append((q.x, q.y))
        cc, nvis = _cc_frac(flags, ndc)
        Mdb = db.matrix_world; vis_db = []
        for hc_ in DBH:
            e1_ = mu.Vector((0, 1, 0)); e2_ = mu.Vector(DB_AX).cross(e1_)
            pts_ = [Mdb @ (hc_ + mu.Vector(DB_AX) * dx + 0.055 * math.cos(t_) * e1_ + 0.055 * math.sin(t_) * e2_)
                    for dx in (-0.03, 0.0, 0.03) for t_ in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
            vis_db.append(round(_visible_frac(pts_), 3))
        # start-frame db visibility: a back camera that sees a connected peak sheet
        # often hides the hanging dumbbell in front of the hips
        sc.frame_set(1); bpy.context.view_layer.update()
        Mdb0 = db.matrix_world; vis_db0 = []
        for hc_ in DBH:
            e1_ = mu.Vector((0, 1, 0)); e2_ = mu.Vector(DB_AX).cross(e1_)
            pts_ = [Mdb0 @ (hc_ + mu.Vector(DB_AX) * dx + 0.055 * math.cos(t_) * e1_ + 0.055 * math.sin(t_) * e2_)
                    for dx in (-0.03, 0.0, 0.03) for t_ in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
            vis_db0.append(round(_visible_frac(pts_), 3))
        sc.frame_set(peak_f); bpy.context.view_layer.update()
        Wm = arm.matrix_world
        raw_s = [Wm @ pb[sp_ + "Hand"].head, Wm @ pb[sp_ + "ForeArm"].head]
        if sp_ + "Hand" in pb:
            raw_s.append(Wm @ pb[sp_ + "Hand"].head + (Wm.to_3x3().normalized() @ pb[sp_ + "Hand"].matrix.to_3x3().normalized() @ mu.Vector((0, 0.12, 0))))
        # bone heads sit inside the mesh; nudge toward the camera so a skin hit is not counted as occlusion
        s_pts = []
        for p0 in raw_s:
            to_cam = CAM.location - p0
            if to_cam.length > 1e-6:
                s_pts.append(p0 + to_cam.normalized() * 0.045)
            else:
                s_pts.append(p0)
        hand_vis = _visible_frac(s_pts)
        # callout text box in NDC (1080x1920 layout: x 60-320, y 600-900)
        call_hit = 0; call_n = 0
        for (x, y), v in zip(ndc, flags):
            if not v: continue
            iy = 1.0 - y
            if 0.055 < x < 0.30 and 0.31 < iy < 0.48:
                call_hit += 1
            call_n += 1
        call_frac = call_hit / max(call_n, 1)
        dbmin = min(vis_db) if vis_db else 0.0
        dbmin0 = min(vis_db0) if vis_db0 else 0.0
        score = cc * 2.0 + dbmin * 1.3 + dbmin0 * 1.3 + float(hand_vis) * 0.55 - call_frac * 1.6
        if dbmin < 0.5: score -= 100.0
        if dbmin0 < 0.5: score -= 100.0
        if hand_vis < 0.15: score -= 2.0
        results.append({"yaw": yaw_d, "elev": el_d, "height_m": ht, "shift_m": sh, "cc": round(cc, 4),
                        "nvis": int(nvis), "db": vis_db, "db_start": vis_db0, "hand": round(float(hand_vis), 3),
                        "callout_frac": round(call_frac, 3), "score": round(score, 4)})
    results.sort(key=lambda r: -r["score"])
    win = results[0]
    _place(win["yaw"], win["elev"], win["height_m"], win["shift_m"])
    pr = np.array([world_to_camera_view(sc, CAM, mu.Vector(p_.tolist()))[:2] for p_ in P[::5]])
    framing.update({"x_min": round(float(pr[:, 0].min()), 3), "x_max": round(float(pr[:, 0].max()), 3),
                    "y_top": round(float(1 - pr[:, 1].max()), 3), "y_feet": round(float(1 - pr[:, 1].min()), 3),
                    "yaw_back_deg": win["yaw"], "elev_deg": win["elev"], "height_m": win["height_m"],
                    "shift_m": win["shift_m"], "search_peak_f": peak_f, "search_cc": win["cc"],
                    "search_db": win["db"], "search_hand": win["hand"], "search_score": win["score"]})
    try:
        json.dump({"winner": win, "peak_f": peak_f, "n": len(results), "top": results[:8], "all": results},
                  open(a.out + "/camera_search.json", "w"), indent=1)
    except Exception:
        pass
    log(f"camera search peak f{peak_f} winner yaw={win['yaw']} elev={win['elev']} h={win['height_m']} sh={win['shift_m']} cc={win['cc']} db={win['db']} hand={win['hand']} score={win['score']} ({len(results)} cands)")

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
        long_e = rest_len[ok_e] >= float(RC.get("sliver_rest_frac", 1.0)) * MED_E
        hips_gi = next((i for i, n in GN.items() if n == "Hips"), None)
        hips_sliver = np.zeros(vis_e.shape, dtype=bool)
        if hips_gi is not None:
            hips_sliver = ((dom[e2_[:, 0]] == hips_gi) | (dom[e2_[:, 1]] == hips_gi)) & (ratio < 0.05)
        defl = rest_len[ok_e] * ratio
        pick = vis_e & long_e & ~hips_sliver & (defl >= 5e-4)  # ignore sub-mm collapsed slivers
        rec.update({"db_head_visible": vis_, "db_head_sep_rel": round(float(np.linalg.norm(hp_[0] - hp_[1]) / hd_), 3),
                    "tear_min_ratio": round(float((ratio[pick] if pick.any() else ratio[vis_e]).min()), 3)})
        rec.update({"tear_max_ratio": round(float(ratio.max()), 3), "tear_frac2x": round(float((ratio > 2.0).mean()), 6),
                    "grip_mesh_m": round(float(np.linalg.norm(c2[hand_vi] - np.array(dbp), axis=1).min()), 4)})
        bad = ev[ok_e][ratio > 2.0][:, 0]
        if len(bad):
            u_, n_ = np.unique(dom[bad], return_counts=True); o_ = np.argsort(-n_)[:3]
            rec["tear_where"] = {GN.get(int(u_[i]), "?"): int(n_[i]) for i in o_}
    if garm is not None and (f % 3 == 1 or f == NF) and "gev" in dir() and "g_ok" in dir():
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
        rec.update({"briefs_gap_min_m": round(float(gap.min()), 4),
                    "briefs_gap_max_m": round(float(gap.max()), 4),
                    "briefs_poke_frac": round(float((gap < cmin).mean()), 6),
                    "briefs_tear_frac2x": round(float((gr > 2.0).mean()), 6), "briefs_tear_max_ratio": round(float(gr.max()), 3),
                    "briefs_cover_frac": round(float((gap >= cmin).mean()), 6), "briefs_verts": int(len(gc_))})
    chk.append(rec)
json.dump({"frames": chk, "working": hand, "floor_tilt_fixed_deg": round(floor_tilt, 1), "rep": [f0, f1], "raw_torso_deg": [round(float(raw_torso.min()), 1), round(float(raw_torso.max()), 1)],
           "framing": framing, "regions": region_counts, "grip_offset_m": GRIP, "person_to_rig_scale": round(SCALE, 3), "kp_res": [rx, ry]}, open(a.out + "/checks.json", "w"), indent=0)
np.savez_compressed(a.out + "/hand.npz", frames=np.array(HANDF), P=np.array(HANDP), labels=HG["labels"], s_finger=HG["s_finger"],
                    handle_r=float(GW.get("handle_r_m", 0.016)), wrap=bool(GW.get("enabled", False)), info=json.dumps(HG["info"]))
np.savez_compressed(a.out + "/bones.npz", names=np.array(BN), rest=BREST, pose=BPOSE, neck_mode=NECK_MODE, legs=LEGS, dumbbell=DBM,
                    capsule_names=np.array(list(CAPR)), capsule_r=np.array(list(CAPR.values())),
                    handle_axis_local=np.array(DB_AX), working_hand=hand, palm_left=PALM["LeftHand"], palm_right=PALM["RightHand"], grip_mode=GRIP_MODE,
                    handle_n_local=np.array(HG["handle_local"]["n"]))
log("checks done")
if LEGS == "bench_row" and BENCH_SPEC is not None:
    import bench_row_pose as _BRP
    def _bp(n_):
        return np.array(arm.matrix_world @ pb[n_].head)
    qc_frames = {}
    for _tag, _fi in (("bottom", 1), ("peak", int(np.argmax(AUTH_U)) + 1 if AUTH_U is not None else max(1, NF // 3))):
        sc.frame_set(_fi); bpy.context.view_layer.update()
        rec_pts = {
            "lkn": _bp("LeftLeg"), "lwr": _bp("LeftHand"), "rank": _bp("RightFoot"),
            "rsh": _bp("RightArm"), "rwr": _bp("RightHand"), "rhip": _bp("RightUpLeg"),
            "rel": _bp("RightForeArm"),
            "lsh": _bp("LeftArm"), "lel": _bp("LeftForeArm"), "lank": _bp("LeftFoot"),
            "rkn": _bp("RightLeg"),
            "lhip": _bp("LeftUpLeg"),
            "rtoe": _bp("RightToeBase") if "RightToeBase" in pb else _bp("RightFoot"),
            "hip": _bp("Hips"), "sh": 0.5 * (_bp("LeftArm") + _bp("RightArm")),
            "head": _bp("Head") if "Head" in pb else _bp("head_end"),
        }
        try:
            rec_pts["db"] = np.array(db.matrix_world.translation)
        except Exception:
            rec_pts["db"] = rec_pts["rwr"]
        if SIDX is not None and len(SIDX):
            try:
                _dg_h = bpy.context.evaluated_depsgraph_get()
                _eb_h = body.evaluated_get(_dg_h); _mh = _eb_h.to_mesh()
                _ph = np.empty(len(_mh.vertices) * 3); _mh.vertices.foreach_get("co", _ph)
                _ph = _ph.reshape(-1, 3)
                _Mwh = np.array(_eb_h.matrix_world)
                _pwh = _ph @ _Mwh[:3, :3].T + _Mwh[:3, 3]
                _eb_h.to_mesh_clear()
                _Mhs = np.array(arm.matrix_world @ pb[SUPPORT_BN].matrix)
                _Mhs[:3, :3] /= np.linalg.norm(_Mhs[:3, :3], axis=0)
                _pm = HGR.measure_palm(_pwh, SIDX, _Mhs, pad_normal=np.array([0.0, 0.0, 1.0]))
                rec_pts["support_finger_spread_m"] = _pm["finger_spread_m"]
                rec_pts["support_palm_from_pad_deg"] = _pm["palm_from_pad_deg"]
                _hp = _pwh[SIDX]
                _pz = float(BENCH_SPEC["pad_z"])
                _ox, _oy = BENCH_SPEC["origin_xy"]
                _hx, _hy = BENCH_SPEC["pad_half_xy"]
                rec_pts["support_palm_pad_dz_m"] = float(np.median(np.abs(_hp[:, 2] - _pz)))
                _inside = ((np.abs(_hp[:, 0] - _ox) <= (_hx + 0.01))
                           & (np.abs(_hp[:, 1] - _oy) <= (_hy + 0.01)))
                rec_pts["support_palm_in_pad_xy"] = bool(float(_inside.mean()) >= 0.90)
                log(f"bench_qc {_tag} support_hand spread={_pm['finger_spread_m']:.3f}m "
                    f"palm_from_pad={_pm['palm_from_pad_deg']:.1f}deg "
                    f"pad_dz={rec_pts['support_palm_pad_dz_m']:.3f} in_xy={rec_pts['support_palm_in_pad_xy']}")
            except Exception as _he:
                log(f"bench_qc {_tag} support_hand measure failed: {_he}")
        _mp_bones = {0: "Head", 11: "LeftArm", 12: "RightArm", 13: "LeftForeArm",
                     14: "RightForeArm", 15: "LeftHand", 16: "RightHand",
                     23: "LeftUpLeg", 24: "RightUpLeg", 25: "LeftLeg", 26: "RightLeg",
                     27: "LeftFoot", 28: "RightFoot",
                     31: "LeftToeBase" if "LeftToeBase" in pb else None,
                     32: "RightToeBase" if "RightToeBase" in pb else None}
        _xy = np.zeros((33, 2), np.float64)
        for _ji, _bn in _mp_bones.items():
            if not _bn or _bn not in pb:
                continue
            _q = world_to_camera_view(sc, CAM, arm.matrix_world @ pb[_bn].head)
            _xy[_ji] = (_q.x, 1.0 - _q.y)
        np.save(a.out + f"/ours_lm2d_{_tag}_render.npy", _xy)
        # Overlay QC stays in the side-fit camera (Commons photo), not the 3/4 beauty camera.
        try:
            import ref_match as _RMo
            _fitc = _RMo.camera_from_cfg(RC.get("camera_fit") or RC.get("camera") or {})
            _reswh = (int(sc.render.resolution_x), int(sc.render.resolution_y))
            _sxy = np.zeros((33, 2), np.float64)
            for _ji, _bn in _mp_bones.items():
                if not _bn or _bn not in pb:
                    continue
                _pw = np.array(arm.matrix_world @ pb[_bn].head)
                _ndc, _ = _RMo.project_ndc(_pw, _fitc, _reswh)
                _sxy[_ji] = (float(_ndc[0]), 1.0 - float(_ndc[1]))
            np.save(a.out + f"/ours_lm2d_{_tag}.npy", _sxy)
            # Rasterize the posed body (+briefs) through the side camera for silhouette IoU.
            _dg = bpy.context.evaluated_depsgraph_get()
            _ww, _hh = _reswh
            _sil = np.zeros((_hh, _ww), np.uint8)
            for _obn in (body, garm if "garm" in dir() and garm is not None else body):
                if _obn is None:
                    continue
                _eb = _obn.evaluated_get(_dg); _mm = _eb.to_mesh()
                _co = np.empty(len(_mm.vertices) * 3); _mm.vertices.foreach_get("co", _co)
                _Mw = np.array(_eb.matrix_world)
                _pts = _co.reshape(-1, 3) @ _Mw[:3, :3].T + _Mw[:3, 3]
                _eb.to_mesh_clear()
                _nd, _zz = _RMo.project_ndc(_pts[::3], _fitc, _reswh)
                for (_u, _v), _z in zip(_nd, _zz):
                    if _z <= 0.05 or not (0.0 <= _u <= 1.0 and 0.0 <= _v <= 1.0):
                        continue
                    _ix = int(round(_u * (_ww - 1)))
                    _iy = int(round((1.0 - _v) * (_hh - 1)))
                    _sil[max(0, _iy - 1):_iy + 2, max(0, _ix - 1):_ix + 2] = 255
            np.save(a.out + f"/ours_sil_{_tag}.npy", _sil)
        except Exception as _se:
            log(f"side-overlay dump failed ({_tag}): {_se}")
            np.save(a.out + f"/ours_lm2d_{_tag}.npy", _xy)
        uu = float(AUTH_U[_fi - 1]) if AUTH_U is not None else (1.0 if _tag == "peak" else 0.0)
        db_ndc = db_vis = db_con = None
        if "db" in dir() or True:
            try:
                dbp = db.matrix_world.translation
                q = world_to_camera_view(sc, CAM, dbp)
                db_ndc = (q.x, q.y)
                # visibility of both heads
                vis = []
                samples = [mu.Vector((0.0, 0.0, 0.0))] + list(DBH)
                for hc_ in samples:
                    wp_ = db.matrix_world @ hc_
                    toward_ = (CAM.location - wp_)
                    if toward_.length > 1e-6:
                        wp_ = wp_ + toward_.normalized() * 0.04
                    vis.append(_visible_frac([wp_]))
                db_vis = float(max(vis)) if vis else 0.0
                log(f"bench_qc {_tag} db world={tuple(round(c,3) for c in dbp)} ndc=({q.x:.3f},{q.y:.3f}) vis={db_vis:.2f}")
            except Exception as _qe:
                log(f"bench_qc {_tag} db probe failed: {_qe}")
        qc_frames[_tag] = _BRP.qc_from_bones(rec_pts, BENCH_SPEC, uu, db_ndc, db_vis, db_con)
        if 0 <= _fi - 1 < len(chk):
            for _bk in ("briefs_gap_min_m", "briefs_gap_max_m", "briefs_poke_frac",
                        "briefs_tear_frac2x", "briefs_cover_frac"):
                if _bk in chk[_fi - 1]:
                    qc_frames[_tag][_bk] = chk[_fi - 1][_bk]
        log(f"bench_qc {_tag} f{_fi} pass={qc_frames[_tag]['pass']} {qc_frames[_tag]['gates']}")
    def _jsonify(o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o).__name__)
    json.dump({"frames": qc_frames, "spec": {k: BENCH_SPEC[k] for k in BENCH_SPEC if k != "u"}},
              open(a.out + "/bench_qc.json", "w"), indent=2, default=_jsonify)
if a.checks_only: sys.exit(0)

# ---- materials + lights (P0 quality upgrade) -----------------------------------------------------------------------
body_mats = [s.material for s in body.material_slots]
look_inscene = (
    str(RC.get("look_backend") or "").lower() in ("inscene", "native", "clay")
    or str((C.get("grade") or {}).get("glow_method") or "").upper() in ("NATIVE", "INSCENE", "CLAY")
)
ad = float(RC.get("albedo_detail", 0.0 if look_inscene else 0.50))
sss_w = min(float(RC.get("sss_weight", 0.0 if look_inscene else 0.16)), 0.04 if look_inscene else 1.0)
em_str = float(RC.get("glow_emission", 0.95 if look_inscene else 1.35))
glow_wrong = C.get("grade", {}).get("glow") == "wrong"
_pal = str((C.get("grade") or {}).get("palette") or "demic_peach").strip().lower()

def _srgb_to_lin(c):
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _hsv_to_lin_rgba(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(float(h) / 180.0, float(s) / 255.0, float(v) / 255.0)
    return (_srgb_to_lin(r), _srgb_to_lin(g), _srgb_to_lin(b), 1.0)

if glow_wrong:
    em_col = (0.82, 0.07, 0.07, 1) if look_inscene else (1.0, 0.28, 0.10, 1)
elif _pal in ("green", "gymform_green", "mint", "neon"):
    em_col = (0.16, 1.0, 0.42, 1)
elif look_inscene:
    em_col = _hsv_to_lin_rgba(16, 62, 225)   # demic core: display H16 S62 V225 under Standard
else:
    em_col = (1.0, 0.62, 0.38, 1)            # demic peach (H~16); COMBO snap finishes H/S/V
base_rgb = tuple(float(x) for x in RC.get("base_color", [0.325, 0.327, 0.332] if look_inscene else [0.22, 0.225, 0.232]))
_clay = (_kit.get("clay") or {}) if "_kit" in dir() else {}
if _clay.get("base_color"):
    base_rgb = tuple(float(x) for x in _clay["base_color"])
if look_inscene:
    log(f"look_inscene clay albedo={tuple(round(c,3) for c in base_rgb)} em={tuple(round(c,3) for c in em_col[:3])} ad=0 sss={sss_w}")

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
    if look_inscene:
        for l in list(bsdf.inputs["Roughness"].links):
            nt.links.remove(l)
        bsdf.inputs["Roughness"].default_value = float(_clay.get("roughness") or 0.78)
        # Soft AO into the clay albedo so grooves and separations read as shadows.
        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao.inputs["Distance"].default_value = float(RC.get("ao_distance", 0.14))
        aom = nt.nodes.new("ShaderNodeMix")
        aom.name = "CLAY_AO_MIX"
        aom.data_type = "RGBA"
        aom.blend_type = "MULTIPLY"
        aom.inputs[0].default_value = float(RC.get("ao_amount", 0.38))
        if body.data.color_attributes.get("cuff_mask") is not None:
            _ca = nt.nodes.new("ShaderNodeAttribute"); _ca.attribute_name = "cuff_mask"
            _sp = nt.nodes.new("ShaderNodeSeparateColor")
            nt.links.new(_ca.outputs["Color"], _sp.inputs["Color"])
            _iv = nt.nodes.new("ShaderNodeMath"); _iv.operation = "SUBTRACT"
            _iv.inputs[0].default_value = 1.0
            nt.links.new(_sp.outputs["Red"], _iv.inputs[1])
            _am = nt.nodes.new("ShaderNodeMath"); _am.operation = "MULTIPLY"
            _am.inputs[0].default_value = float(RC.get("ao_amount", 0.38))
            nt.links.new(_iv.outputs[0], _am.inputs[1])
            nt.links.new(_am.outputs[0], aom.inputs[0])
        base_src = bsdf.inputs["Base Color"].links[0].from_socket
        nt.links.new(base_src, aom.inputs[6])
        nt.links.new(ao.outputs["Color"], aom.inputs[7])
        clay_src = aom.outputs[2]
        cav = float(RC.get("cavity_amount", 0.0))
        if cav > 0:
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            ramp = nt.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].position = 0.46
            ramp.color_ramp.elements[0].color = (0.58, 0.58, 0.58, 1)
            ramp.color_ramp.elements[1].position = 0.56
            ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
            nt.links.new(geo.outputs["Pointiness"], ramp.inputs["Fac"])
            cvm = nt.nodes.new("ShaderNodeMix")
            cvm.name = "CLAY_CAV_MIX"
            cvm.data_type = "RGBA"
            cvm.blend_type = "MULTIPLY"
            cvm.inputs[0].default_value = cav
            nt.links.new(clay_src, cvm.inputs[6])
            nt.links.new(ramp.outputs["Color"], cvm.inputs[7])
            clay_src = cvm.outputs[2]
        for l in list(bsdf.inputs["Base Color"].links):
            nt.links.remove(l)
        nt.links.new(clay_src, bsdf.inputs["Base Color"])
    elif not bsdf.inputs["Roughness"].links:          # keep the packed roughness map when the GLB already wired it
        bsdf.inputs["Roughness"].default_value = 0.50
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = sss_w
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.55, 0.58, 0.62)
    if "Subsurface Scale" in bsdf.inputs:
        bsdf.inputs["Subsurface Scale"].default_value = float(RC.get("sss_scale", 0.05))
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.06 if look_inscene else 0.35
    nstr = float(RC.get("normal_strength", 1.0 if look_inscene else 2.0))
    if look_inscene and nstr > 1.35:
        nstr = 1.0
    cuff_ok = body.data.color_attributes.get("cuff_mask") is not None
    for nm in [n for n in nt.nodes if n.type == "NORMAL_MAP"]:
        nm.inputs["Strength"].default_value = nstr
        if cuff_ok:
            # Strength 0 on the cuff drops the baked wristband ridge in rig_normal.
            ca = nt.nodes.new("ShaderNodeAttribute"); ca.attribute_name = "cuff_mask"
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            nt.links.new(ca.outputs["Color"], sep.inputs["Color"])
            inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"
            inv.inputs[0].default_value = 1.0
            nt.links.new(sep.outputs["Red"], inv.inputs[1])
            mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
            mul.inputs[0].default_value = nstr
            nt.links.new(inv.outputs[0], mul.inputs[1])
            nt.links.new(mul.outputs[0], nm.inputs["Strength"])
    if cuff_ok:
        log("cuff_mask: normal-map strength faded on wrist cuff")
    # 3D glow: shared tools/rig3d/glow_shader.py
    mout = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import glow_shader as GSH
_flex_min = float(RC.get("glow_flex_min", 0.88 if look_inscene else 0.58))
_backend = "inscene" if look_inscene else "post"
if str(RC.get("glow_shape") or "").lower() == "shells":
    flex_nodes = list(SHELL_FLEX) if "SHELL_FLEX" in dir() else []
    log(f"kit: porcelain clay on body, {len(flex_nodes)} overlay shell flex nodes")
elif str(RC.get("glow_shape") or "").lower() in ("attr", "weld", "weld_attr"):
    import kit.glow_attr as KITGA
    _hrgb = KITGA.hues_rgb_from_cfg(C)
    flex_nodes = []
    for m_ in body_mats:
        _nt = m_.node_tree
        _bsdf = [n for n in _nt.nodes if n.type == "BSDF_PRINCIPLED"][0]
        _mout = [n for n in _nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
        flex_nodes.append(GSH.attach_attr_sheets(
            _nt, _bsdf, _mout, _hrgb, em_str, _flex_min,
            tint_mix=float(RC.get("glow_tint", 0.18)),
            rim=float(RC.get("glow_rim", 0.28))))
    log(f"kit: attr sheets flex={len(flex_nodes)} hues={ {k: [round(x,3) for x in v] for k,v in _hrgb.items()} }")
else:
    flex_nodes = GSH.switch_materials(
        body_mats, _backend, em_col, em_str, _flex_min, float(RC.get("glow_fibre_scale", 72.0)),
        tint_mix=float(RC.get("glow_tint", 0.12)),
        fibre_bump=float(RC.get("glow_fibre_bump", 0.22)),
        rim=float(RC.get("glow_rim", 0.28)),
        body_rim=float(RC.get("body_rim", 0.28)),
        fibre_line=float(RC.get("glow_fibre_line", 0.45)),
        opacity=float(RC.get("glow_opacity", 0.94)),
    )
if look_inscene:
    log(f"glow backend=inscene flex_min={_flex_min} em_str={em_str} shape={RC.get('glow_shape')}")

_eco_path = str((C.get("asset") or {}).get("ecorche_glb") or "")
_body_src = str(RC.get("body_source") or "grab3d").lower()
if look_inscene and _body_src in ("ecorche", "eco", "mesh") and _eco_path:
    try:
        import ecorche_bind as ECO
        _eco_tgts = list((C.get("grade") or {}).get("target_muscles") or ["lats", "rhomboids", "mid_trap", "rear_delt"])
        _eco_hues = dict(MIDS.DEFAULT_HUES)
        _yh2 = (C.get("grade") or {}).get("muscle_hues") or {}
        if isinstance(_yh2, dict):
            _eco_hues.update(_yh2)
        _eco_hstr = (C.get("grade") or {}).get("muscle_hue_strength") or {}
        _eco_flex = ECO.bind_and_shade(
            body=body, arm=arm, path=_eco_path, targets=_eco_tgts,
            hues=_eco_hues, hue_strength=_eco_hstr, base_rgb=base_rgb,
            em_str=em_str, flex_min=_flex_min,
            tint=float(RC.get("glow_tint", 0.22)),
            fibre_bump=float(RC.get("glow_fibre_bump", 0.42)),
            rim=float(RC.get("glow_rim", 0.35)),
            ao_distance=float(RC.get("ao_distance", 0.14)),
            ao_amount=float(RC.get("ao_amount", 0.38)),
            hide_body=True, log=log,
        )
        if _eco_flex:
            # Grab3D flex stays at 0 (underlayer). Only ecorche meshes glow.
            GSH.set_flex(flex_nodes, 0.0)
            flex_nodes[:] = list(_eco_flex)
        log(f"body_source=ecorche targets={_eco_tgts} glow_flex={len(_eco_flex or [])}")
    except Exception as _ee:
        import traceback
        log(f"ecorche bind failed (keeping Grab3D body): {_ee}")
        log(traceback.format_exc())



sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.render.film_transparent = False
wd = bpy.data.worlds.new("w"); sc.world = wd; wd.use_nodes = True; bg = wd.node_tree.nodes["Background"]
hdri = C["asset"]["hdri"]
if "_kit" in dir() and _kit.get("enabled") and (_kit.get("lights") or {}):
    import kit.lights as KITL
    KITL.apply_studio_world(sc, hdri, C, log=log)
    bg = sc.world.node_tree.nodes.get("KIT_HDRI_BG") or bg
elif look_inscene:
    # Neutral even ambient — no photographed HDRI hotspots (those dirtied non-targets).
    bg.inputs["Color"].default_value = (0.82, 0.83, 0.85, 1)
    bg.inputs["Strength"].default_value = float(RC.get("ambient_strength", 0.28))
    log(f"world lighting: neutral ambient {bg.inputs['Strength'].default_value}")
else:
    envn = wd.node_tree.nodes.new("ShaderNodeTexEnvironment"); envn.image = bpy.data.images.load(hdri["file"])
    wd.node_tree.links.new(envn.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = float(RC.get("hdri_strength", hdri.get("strength", 0.35)))
# Camera rays see dark charcoal (never a black void / photographed HDRI). Lighting rays still use HDRI/ambient.
# Kit world already built this mix; do not rewire a cleared node tree.
_kit_world = "_kit" in dir() and _kit.get("enabled") and (_kit.get("lights") or {})
try:
    if _kit_world:
        raise RuntimeError("kit-world-already-wired")
    nt_w = wd.node_tree
    lp = nt_w.nodes.new("ShaderNodeLightPath")
    bg_cam = nt_w.nodes.new("ShaderNodeBackground")
    _cam_lin = 0.0185 if look_inscene else 0.028   # Standard display V~38
    bg_cam.inputs["Color"].default_value = (_cam_lin, _cam_lin, _cam_lin + 0.001, 1)
    bg_cam.inputs["Strength"].default_value = 1.0
    mx = nt_w.nodes.new("ShaderNodeMixShader")
    out_w = [n for n in nt_w.nodes if n.type == "OUTPUT_WORLD"][0]
    nt_w.links.new(lp.outputs["Is Camera Ray"], mx.inputs["Fac"])
    nt_w.links.new(bg.outputs["Background"], mx.inputs[1])
    nt_w.links.new(bg_cam.outputs["Background"], mx.inputs[2])
    for _l in list(out_w.inputs["Surface"].links):
        nt_w.links.remove(_l)
    nt_w.links.new(mx.outputs[0], out_w.inputs["Surface"])
    if hasattr(wd, "cycles_visibility") and hasattr(wd.cycles_visibility, "camera"):
        wd.cycles_visibility.camera = True
    log("world: HDRI lights, camera rays charcoal")
except Exception as e:
    if str(e) != "kit-world-already-wired":
        if hasattr(wd, "cycles_visibility") and hasattr(wd.cycles_visibility, "camera"):
            wd.cycles_visibility.camera = False
        log(f"world charcoal fallback: {e}")
key_from = (CAM.location - ctr); key_from.z = 0; key_from.normalize()
upv = mu.Vector((0.0, 0.0, 1.0)); sidev = key_from.cross(upv)
sidev = (sidev if sidev.length > 1e-4 else mu.Vector((1.0, 0.0, 0.0))).normalized()

def add_area(name, loc, energy, size, color):
    Lt = bpy.data.lights.new(name, "AREA"); Lt.energy = float(energy); Lt.size = float(size); Lt.color = color
    o = bpy.data.objects.new(name, Lt); col.objects.link(o)
    o.location = loc; o.rotation_euler = (ctr - o.location).to_track_quat("-Z", "Y").to_euler(); return o

if "_kit" in dir() and _kit.get("enabled") and (_kit.get("lights") or {}):
    import kit.lights as KITL
    KITL.add_three_point(col, ctr, CAM.location, C, log=log)
elif look_inscene:
    # Large soft studio, front-biased toward the camera (the visible 3/4 surface),
    # even lighting, soft shadows, no cool-tinted rim. Matches demic's even clay.
    add_area("key", ctr + key_from * 2.5 + upv * 2.4 + sidev * 1.15,
             RC.get("key_energy", 100), float(RC.get("key_size", 3.2)), (1.00, 0.99, 0.97))
    add_area("fill", ctr + key_from * 1.6 + upv * 0.35 - sidev * 2.6,
             RC.get("fill_energy", 80), float(RC.get("fill_size", 5.0)), (0.99, 0.99, 1.00))
    add_area("rim", ctr - key_from * 2.2 + upv * 1.4 + sidev * 0.5, RC.get("rim_energy", 22), 3.2, (1.00, 1.00, 1.00))
    add_area("top", ctr + upv * 3.2 + key_from * 0.3 + sidev * 0.7, RC.get("top_energy", 28), 3.6, (1.00, 0.99, 0.97))
    log("studio: directional upper-side key + weaker fill (demic clay)")
else:
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
if look_inscene:
    _lk = float(RC.get("leg_key_energy", 0))
    if _lk > 1.0:
        add_area_aim("leg_key", leg_aim + key_from * 2.2 + upv * 0.7, _lk, 4.6, (1.00, 0.99, 0.97), leg_aim)
    if LEGS == "bench_row":
        _dbp = db.matrix_world.translation
        add_area_aim("db_kicker", _dbp + mu.Vector((0.55, -0.30, 0.28)), float(RC.get("db_kicker_energy", 10)), 0.40, (1.00, 0.96, 0.88), _dbp)
        log("studio: working-side DB kicker")
else:
    add_area_aim("leg_key", leg_aim + key_from * 2.4 + upv * 0.55, RC.get("leg_key_energy", 190), 1.8, (1.00, 0.98, 0.94), leg_aim)
    add_area_aim("leg_rim", leg_aim - key_from * 2.3 + upv * 0.70, RC.get("leg_rim_energy", 170), 1.5, (0.70, 0.84, 1.00), leg_aim)
floor = None
if RC.get("floor", True):
    import bmesh
    flz = float(RC.get("floor_z") or 0.0)
    try:
        sc.frame_set(1); bpy.context.view_layer.update()
        _ft = [(arm.matrix_world @ pb[n].head).z for n in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase") if n in pb]
        if _ft and RC.get("floor_z") in (None, ""):
            flz = float(min(_ft)) - 0.008
    except Exception:
        pass
    # Enclosing cyclorama: rotationally symmetric bowl (flat floor + quadratic cove) so lateral,
    # 3/4, and front cameras all see the same seamless charcoal interior — no camera-relative
    # "far-half" ramp, no hard slab edges, no transparent fade (that ghosted the legs and left a
    # black void at the top of the true-side view).
    R_max = float(RC.get("cyc_radius_m", 12.0))
    R_flat = float(RC.get("cyc_flat_radius_m", R_max))
    H_wall = float(RC.get("cyc_height_m", 10.0))
    segs = int(RC.get("cyc_segments", 72))
    rings = int(RC.get("cyc_rings", 28))
    bm = bmesh.new()
    v0 = bm.verts.new((0.0, 0.0, 0.0))
    rings_v = []
    for i in range(1, rings + 1):
        r = R_max * (i / rings)
        tt = 0.0 if r <= R_flat else (r - R_flat) / max(R_max - R_flat, 1e-6)
        z = H_wall * (tt * tt)
        ring = [bm.verts.new((r * math.cos(2.0 * math.pi * j / segs), r * math.sin(2.0 * math.pi * j / segs), z)) for j in range(segs)]
        rings_v.append(ring)
    bm.verts.ensure_lookup_table()
    r1 = rings_v[0]
    for j in range(segs):
        bm.faces.new((v0, r1[j], r1[(j + 1) % segs]))
    for i in range(len(rings_v) - 1):
        ra, rb = rings_v[i], rings_v[i + 1]
        for j in range(segs):
            jn = (j + 1) % segs
            bm.faces.new((ra[j], rb[j], rb[jn], ra[jn]))
    me = bpy.data.meshes.new("studio_cyc")
    bm.to_mesh(me); bm.free(); me.update()
    floor = bpy.data.objects.new("studio_floor", me)
    floor.location = (ctr.x, ctr.y, flz)
    try: col.objects.link(floor)
    except RuntimeError: pass
    fm = bpy.data.materials.new("studio_floor"); fm.use_nodes = True
    nt = fm.node_tree; fb = nt.nodes["Principled BSDF"]
    if look_inscene:
        # Emission-only charcoal so area lights cannot lift the bowl. Standard display V~38.
        _cyc = 0.0185
        emc = nt.nodes.new("ShaderNodeEmission")
        emc.inputs["Color"].default_value = (_cyc, _cyc, _cyc + 0.0004, 1)
        emc.inputs["Strength"].default_value = 1.0
        out_c = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
        for _l in list(out_c.inputs["Surface"].links):
            nt.links.remove(_l)
        nt.links.new(emc.outputs[0], out_c.inputs["Surface"])
    else:
        fb.inputs["Base Color"].default_value = (0.018, 0.018, 0.019, 1)
        fb.inputs["Roughness"].default_value = 0.62
        if "Specular IOR Level" in fb.inputs:
            fb.inputs["Specular IOR Level"].default_value = 0.22
        if "Emission Color" in fb.inputs:
            fb.inputs["Emission Color"].default_value = (0.028, 0.028, 0.030, 1)
            if "Emission Strength" in fb.inputs:
                fb.inputs["Emission Strength"].default_value = 0.85
    floor.data.materials.clear(); floor.data.materials.append(fm)
    log(f"cyclorama bowl r_flat={R_flat} r_max={R_max} h={H_wall} z={flz:.3f} (opaque, all views)")
    if "_kit" in dir() and _kit.get("enabled"):
        # Keep the emission cyclorama and a flat floor (R_flat=R_max) so the
        # cove horizon cannot photograph as a hoop around the figure.
        # Replace the tessellated bowl with a single quad so no ring can show.
        try:
            import bmesh as _bm
            _b = _bm.new()
            _s = 16.0
            _vs = [_b.verts.new(p) for p in ((-_s, -_s, 0), (_s, -_s, 0), (_s, _s, 0), (-_s, _s, 0))]
            _b.faces.new(_vs)
            _me = bpy.data.meshes.new("studio_floor_flat")
            _b.to_mesh(_me); _b.free()
            plane = bpy.data.objects.new("studio_floor_flat", _me)
            plane.location = floor.location.copy()
            col.objects.link(plane)
            if floor.data.materials:
                plane.data.materials.append(floor.data.materials[0])
            floor.hide_render = True
            floor.hide_viewport = True
            floor = plane
            log("kit floor: replaced bowl with flat emission plane 32 m")
        except Exception as _fe:
            log(f"kit floor plane replace: {_fe}")
        for o in list(col.objects):
            n = (o.name or "").lower()
            if o.type in ("CURVE",) or (o is not floor and ("circle" in n or "hoop" in n)):
                o.hide_render = True
                log(f"kit floor: hid leftover {o.name}")
sc.frame_start, sc.frame_end = 1, NF; sc.render.fps = int(AUTH_META["fps"]) if AUTH_META else int(D["fps"]); sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
if a.frames == "all":
    frames = list(range(1, NF + 1))
elif str(a.frames) in ("keyframes", "key", "spec"):
    _peak = int(np.argmax(AUTH_U)) + 1 if AUTH_U is not None else max(1, NF // 3)
    frames = sorted({1, _peak})
    log(f"frames=keyframes -> {frames}")
else:
    frames = [int(v) for v in a.frames.split(",")]
def flex_at(f):
    els = [c["elbow_deg"] for c in chk]
    emin, emax = min(els), max(els)
    if emax - emin < 1e-3: return 0.0
    return float((emax - els[f - 1]) / (emax - emin))

def update_regions(f):
    if USE_MID:
        fl = flex_at(f); GSH.set_flex(flex_nodes, fl)
        return fl, {"mid": True, "flex": round(fl, 3)}
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


def _place_sheet_cam(yaw_deg, elev_deg, look, dist, lens=50.0, shift_x=0.0, shift_y=-0.04):
    """Working-side camera: yaw 0 = pure +X side; +yaw rotates toward the feet (behind the side).
    Default Z-up roll so a horizontal bench stays horizontal-ish — no head-up roll."""
    look_v = mu.Vector([float(x) for x in look])
    yaw, el = math.radians(float(yaw_deg)), math.radians(float(elev_deg))
    dirv = mu.Vector((math.cos(yaw) * math.cos(el),
                      -math.sin(yaw) * math.cos(el),
                      math.sin(el))).normalized()
    CAM.location = look_v + dirv * float(dist)
    CAM.rotation_euler = (look_v - CAM.location).to_track_quat("-Z", "Y").to_euler()
    camd.lens = float(lens)
    camd.shift_x = float(shift_x)
    camd.shift_y = float(shift_y)
    bpy.context.view_layer.update()


def render_camera_contact_sheet(out_path):
    """Cheap 8-view sheet of the peak pose. Does not overwrite beauty frames."""
    cands = [
        {"yaw": 35, "elev": 20, "tag": "3/4 back"},
        {"yaw": 40, "elev": 30, "tag": "3/4 back"},
        {"yaw": 48, "elev": 40, "tag": "3/4 back"},
        {"yaw":  0, "elev": 15, "tag": "pure side"},
        {"yaw": 70, "elev": 35, "tag": "hip-end 3/4"},
        {"yaw": 50, "elev": 28, "tag": "demic-ish A"},
        {"yaw": 33, "elev": 36, "tag": "demic-ish B"},
        {"yaw": 58, "elev": 24, "tag": "wide 3/4"},
    ]
    look = [0.18, 0.22, 0.74]
    dist = 2.50
    peak_f = int(np.argmax(AUTH_U)) + 1 if AUTH_U is not None else (frames[0] if frames else 1)
    update_regions(peak_f)
    sc.frame_set(peak_f)
    bpy.context.view_layer.update()
    try:
        sc.render.use_compositing = False
    except Exception:
        pass
    eng = None
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = name
            eng = name
            break
        except Exception:
            continue
    if eng is None:
        sc.render.engine = "CYCLES"
        eng = "CYCLES"
        sc.cycles.samples = 8
        sc.cycles.use_denoising = False
    else:
        try:
            sc.eevee.taa_render_samples = 8
        except Exception:
            pass
        try:
            sc.eevee.use_raytracing = True
        except Exception:
            pass
    log(f"camera sheet engine={eng} peak_f={peak_f}")
    tw, th = 270, 480
    sc.render.resolution_x, sc.render.resolution_y = tw, th
    sc.render.resolution_percentage = 100
    tile_dir = os.path.join(a.out, "_cam_sheet")
    os.makedirs(tile_dir, exist_ok=True)
    tiles = []
    for i, c in enumerate(cands):
        _place_sheet_cam(c["yaw"], c["elev"], look, dist)
        path = os.path.join(tile_dir, f"c{i:02d}_yaw{c['yaw']}_el{c['elev']}.png")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        tiles.append((path, c))
        log(f"sheet tile {i+1}/8 yaw={c['yaw']} elev={c['elev']} -> {path}")
    # composite with a caption bar under each tile
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        log(f"PIL missing, tiles only: {e}")
        return tile_dir
    label_h, title_h, pad = 44, 40, 8
    cols, rows = 4, 2
    W = pad + cols * (tw + pad)
    H = title_h + pad + rows * (th + label_h + pad)
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = font_b = ImageFont.load_default()
    draw.text((pad, 10),
              "peak pose camera options   |   yaw 0 = working-side  +yaw = behind the side (toward feet)",
              font=font_b, fill=(220, 220, 224))
    for i, (path, c) in enumerate(tiles):
        r, col = divmod(i, cols)
        x = pad + col * (tw + pad)
        y = title_h + pad + r * (th + label_h + pad)
        im = Image.open(path).convert("RGB")
        if im.size != (tw, th):
            im = im.resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(im, (x, y))
        cap = f"yaw {c['yaw']}°   elev {c['elev']}°"
        sub = c["tag"]
        draw.rectangle((x, y + th, x + tw, y + th + label_h), fill=(24, 24, 26))
        draw.text((x + 8, y + th + 4), cap, font=font_b, fill=(240, 240, 242))
        draw.text((x + 8, y + th + 24), sub, font=font, fill=(170, 170, 174))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    log(f"camera sheet -> {out_path} ({sheet.size[0]}x{sheet.size[1]})")
    return out_path

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
    sc.view_settings.view_transform = str(RC.get("view_transform") or ("Standard" if look_inscene else "AgX"))
    try:
        sc.view_settings.look = "None"
    except Exception:
        pass
    if look_inscene:
        try:
            sc.view_settings.exposure = float(RC.get("exposure", 0.0))
        except Exception:
            pass
        log(f"view_transform={sc.view_settings.view_transform} exposure={getattr(sc.view_settings, 'exposure', 0)}")
        if bool(RC.get("glow_bloom", True)) and not a.camera_sheet:
            try:
                GSH.setup_compositor(sc, float(RC.get("blender_bloom", 0.22)))
                log(f"inscene bloom={float(RC.get('blender_bloom', 0.22))}")
            except Exception as _e:
                log(f"inscene bloom skipped: {_e}")
                try:
                    sc.render.use_compositing = False
                except Exception:
                    pass
    if a.camera_sheet:
        t = time.time()
        render_camera_contact_sheet(a.camera_sheet)
        tim["camera_sheet_s"] = round(time.time() - t, 1)
        log(f"camera sheet {tim['camera_sheet_s']} s")
        json.dump({**tim, "frames": 0, "res": [270, 480], "camera_sheet": a.camera_sheet},
                  open(a.out + "/render_time.json", "w"))
        print("CAMERA_SHEET_DONE", a.camera_sheet)
        sys.exit(0)
    if getattr(a, "cam_preview", False):
        t = time.time()
        prev_dir = os.path.join(a.out, "_cam_preview")
        results = render_fitted_cam_previews(prev_dir)
        tim["cam_preview_s"] = round(time.time() - t, 1)
        log(f"cam-preview {tim['cam_preview_s']} s -> {prev_dir}")
        json.dump({**tim, "frames": 0, "res": [540, 960], "cam_preview": results},
                  open(a.out + "/render_time.json", "w"))
        print("CAM_PREVIEW_DONE", prev_dir)
        sys.exit(0)
    t = time.time(); render_set("f"); tim["beauty_s"] = round(time.time() - t, 1); log(f"beauty {tim['beauty_s']} s")
    try:
        _pf = int(np.argmax(AUTH_U)) + 1 if AUTH_U is not None else (25 if 25 in frames else (frames[-1] if frames else 1))
        sc.frame_set(_pf)
        bpy.context.view_layer.update()
        dump_bone_proj(os.path.join(a.out, "bone_proj.json"))
    except Exception as _be:
        log(f"bone proj dump skipped: {_be}")
    _proof = RC.get("camera_proof")
    if _proof:
        try:
            _old_loc, _old_rot = CAM.location.copy(), CAM.rotation_euler.copy()
            _old_sx, _old_sy, _old_lens = camd.shift_x, camd.shift_y, camd.lens
            _pinfo = apply_fixed_cam(_proof)
            _pf = int(np.argmax(AUTH_U)) + 1 if AUTH_U is not None else (frames[-1] if frames else 1)
            sc.frame_set(_pf); bpy.context.view_layer.update()
            try:
                update_regions(_pf)
            except Exception:
                pass
            sc.render.filepath = a.out + "/proof_peak.png"
            bpy.ops.render.render(write_still=True)
            json.dump(_pinfo, open(a.out + "/camera_proof.json", "w"), indent=2)
            CAM.location, CAM.rotation_euler = _old_loc, _old_rot
            camd.shift_x, camd.shift_y, camd.lens = _old_sx, _old_sy, _old_lens
            apply_fixed_cam(CAMC) if str(CAMC.get("mode") or "").lower() == "fixed" else None
            log(f"proof peak f{_pf} yaw={_pinfo.get('yaw_back_deg')} elev={_pinfo.get('elev_deg')}")
            tim["proof_s"] = round(time.time() - t, 1)
        except Exception as _pe:
            log(f"proof camera failed: {_pe}")
    if look_inscene and LEGS == "bench_row" and garm is not None:
        try:
            _pf = int(np.argmax(AUTH_U)) + 1 if AUTH_U is not None else frames[0]
            sc.frame_set(_pf); bpy.context.view_layer.update()
            _hip = arm.matrix_world @ pb["Hips"].head
            # posed back faces +Z; inspect the camera-side glute (working +X, slightly up)
            _lookg = mu.Vector((_hip.x + 0.06, _hip.y - 0.03, _hip.z + 0.07))
            _old_loc, _old_rot = CAM.location.copy(), CAM.rotation_euler.copy()
            _old_sx, _old_sy = camd.shift_x, camd.shift_y
            CAM.location = _lookg + mu.Vector((0.28, -0.16, 0.20))
            CAM.rotation_euler = (_lookg - CAM.location).to_track_quat("-Z", "Y").to_euler()
            camd.shift_x = 0.0; camd.shift_y = 0.0
            _orx, _ory = sc.render.resolution_x, sc.render.resolution_y
            sc.render.resolution_x = 640; sc.render.resolution_y = 640
            sc.render.filepath = a.out + "/glute_closeup.png"
            bpy.ops.render.render(write_still=True)
            sc.render.resolution_x, sc.render.resolution_y = _orx, _ory
            CAM.location, CAM.rotation_euler = _old_loc, _old_rot
            camd.shift_x, camd.shift_y = _old_sx, _old_sy
            log(f"glute close-up -> {a.out}/glute_closeup.png")
        except Exception as _ge:
            log(f"glute close-up failed: {_ge}")
    if look_inscene:
        GSH.teardown_compositor(sc)
    want_ab = (not look_inscene) and bool(RC.get("glow_ab") or str(RC.get("glow_backend", "post")).lower() in ("blender", "both"))
    if want_ab:
        flex_nodes[:] = GSH.switch_materials(body_mats, "blender", em_col, em_str, _flex_min, float(RC.get("glow_fibre_scale", 42.0)))
        GSH.setup_compositor(sc, float(RC.get("blender_bloom", 0.60)))
        t = time.time(); render_set("b"); tim["beauty_blender_s"] = round(time.time() - t, 1); log(f"beauty blender-glow {tim['beauty_blender_s']} s")
        GSH.teardown_compositor(sc)
    if (not look_inscene) and bool(RC.get("glow_method_d", True)):
        flex_nodes[:] = GSH.switch_materials(body_mats, "tint", em_col, em_str, _flex_min, float(RC.get("glow_fibre_scale", 42.0)))
        t = time.time(); render_set("d"); tim["beauty_tint_s"] = round(time.time() - t, 1); log(f"beauty tint {tim['beauty_tint_s']} s")
        flex_nodes[:] = GSH.switch_materials(body_mats, "post", em_col, em_str, _flex_min, float(RC.get("glow_fibre_scale", 42.0)))
if not a.no_mask:
    # Beauty used an opaque cyclorama (film_transparent=False) so every view is charcoal.
    # Mask + ID must be transparent-film: otherwise alpha=255 everywhere (body reads as
    # the whole frame) and the charcoal world decodes as muscle IDs 7/8 (front_delt/biceps).
    sc.render.film_transparent = True
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
        if WAISTBAND is not None:
            WAISTBAND.data.materials.clear(); WAISTBAND.data.materials.append(gw)
    sc.cycles.samples = int(RC.get("mask_spp", 4)); sc.cycles.use_denoising = False; sc.cycles.max_bounces = 0
    sc.view_settings.view_transform = "Standard"; bg.inputs["Strength"].default_value = 0.0
    for o in bpy.data.objects:
        if o.type == "LIGHT": o.hide_render = True
    t = time.time(); render_set("m"); tim["mask_s"] = round(time.time() - t, 1); log(f"mask {tim['mask_s']} s")
    if bool(RC.get("glow_id_pass", True)) and USE_MID:
        for m_ in body_mats:
            nt = m_.node_tree; out = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
            # Emit mid.R = id/255 (G=B=0). Raw view-transform writes R_png == muscle index.
            at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "mid"; em = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(at.outputs["Color"], em.inputs["Color"]); em.inputs["Strength"].default_value = 1.0
            nt.links.new(em.outputs[0], out.inputs["Surface"])
        try:
            sc.view_settings.view_transform = "Raw"
        except Exception:
            sc.view_settings.view_transform = "Standard"
        t = time.time(); render_set("id"); tim["id_s"] = round(time.time() - t, 1); log(f"id pass {tim['id_s']} s")
json.dump({**tim, "frames": len(frames), "res": [rx, ry], "skipped_existing": bool(RC.get("skip_existing_frames"))}, open(a.out + "/render_time.json", "w"))
try:
    import kit.gates as KITGT
    KITGT.write_topo(a.out + "/kit_topo.json", {
        "weld": WELD_STATS,
        "glow_attr": GLOW_ATTR_INFO,
        "garment": GARM_META,
        "garment_bound": None if garm is None else KITGT.mesh_report(garm),
        "waistband_bound": None if WAISTBAND is None else KITGT.mesh_report(WAISTBAND),
        "welded_bound": None if WELDED is None else KITGT.mesh_report(WELDED),
    })
    log(f"kit_topo written {a.out}/kit_topo.json")
except Exception as _te:
    log(f"kit_topo write failed: {_te}")
print("RENDER_DONE", json.dumps(tim))
