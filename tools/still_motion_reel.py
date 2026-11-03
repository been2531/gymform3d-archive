#!/usr/bin/env python3
"""t677u still + code-motion reel (one config per exercise, 0 generations).

  nice -n 19 python3 tools/still_motion_reel.py configs/still_motion/<exercise>.json [--stills-only]

Stage 1 (stills, once): per form take the START and PEAK frames of the existing clip; stack region of the peak taken from
the empty plate (no split stack, no warp); Lanczos + mild unsharp; glow-groove compression; a DE-GLOWED copy (glow
pixels -> grey at the surrounding skin level) for the start and the glow ramp. The three stills of a form (start
no-glow, peak no-glow, peak glow) go into ONE lossless static clip so a single native.prepare gives them the same tone
curve / flat field, then the UNCHANGED production grade (native_grade + muscle.day4_fibre + region_glow clamp + the
Day 4 calibrated knobs of the base config) runs once per still.
Stage 2 (motion, per frame, cheap): crossfades, glow ramp + pulse (blend of the peak no-glow / glow stills), slow
push-in (camera only), joint-attached neon-leader angle counter (gymform/joint_callout.blend_tile), Day 4 title +
2-thumb preview (active highlighted) + X / check badge, then voice (gymform.tts voice, one line per form) + bed/chimes.
Timeline: wrong peak hook -> wrong loop (start -> peak) -> right (start -> peak) -> back to the wrong peak (loops)."""
import sys, os, json, time, copy, gc, math, subprocess, asyncio, wave, argparse
import cv2, numpy as np
ROOT = "/workspace/daily_info_reels"
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--stills-only", action="store_true")
A = ap.parse_args()
T0 = time.time()
C = json.load(open(A.config))
R = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
OUT = R(C["out_dir"]); os.makedirs(OUT, exist_ok=True)
SP = C["still_prep"]
W, H = 1080, 1920


def read_frame(path, i):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read(); cap.release()
    assert ok, (path, i)
    return f


def stack_from_empty(fr, em):
    """weight-stack column (blue plates of the EMPTY plate, extended upward for lifted plates) taken from the empty
    plate where the frame differs: the stack reads at rest and whole; nothing is warped."""
    b, g, r = [em[..., i].astype(int) for i in range(3)]
    blue = (b - r > 40) & (b - g > 20)
    ys, xs = np.where(blue)
    if len(xs) < 200:
        return fr, 0
    x0, x1 = np.percentile(xs, 1) - 8, np.percentile(xs, 99) + 8
    y0, y1 = ys.min() - 260, ys.max() + 12
    col = np.zeros(fr.shape[:2], np.uint8); col[int(max(0, y0)):int(y1), int(max(0, x0)):int(x1)] = 1
    d = np.abs(fr.astype(int) - em.astype(int)).max(2) > 10
    m = cv2.dilate((d & (col > 0)).astype(np.uint8), np.ones((7, 7), np.uint8)) & col
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)[..., None]
    return np.clip(fr * (1 - mf) + em * mf, 0, 255).astype(np.uint8), int(m.sum())


def prep(im):
    im = cv2.resize(im, (720, 1280), interpolation=cv2.INTER_LANCZOS4)
    if SP.get("sharpen", 0) > 0:
        bl = cv2.GaussianBlur(im, (0, 0), 1.0); im = cv2.addWeighted(im, 1 + SP["sharpen"], bl, -SP["sharpen"], 0)
    gd = SP.get("glow_detail", 1.0)
    if gd != 1.0:
        f = im.astype(np.float32)
        dom = np.maximum(f[..., 1] - np.maximum(f[..., 0], f[..., 2]), f[..., 2] - np.maximum(f[..., 0], f[..., 1]))
        g = cv2.GaussianBlur(cv2.dilate((dom > 18).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3.0)
        base = cv2.GaussianBlur(f, (0, 0), 2.5)
        im = np.clip(f * (1 - g[..., None]) + (base + gd * (f - base)) * g[..., None], 0, 255).astype(np.uint8)
    return im


def deglow(im):
    """glow pixels -> neutral grey with their own relief, levelled to the surrounding (unlit) skin: the no-glow still."""
    f = im.astype(np.float32)
    dom = np.maximum(f[..., 1] - np.maximum(f[..., 0], f[..., 2]), f[..., 2] - np.maximum(f[..., 0], f[..., 1]))
    m = cv2.GaussianBlur(cv2.dilate((dom > 12).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 2.0)
    L = f @ np.array([0.114, 0.587, 0.299], np.float32)
    core = m > 0.5
    ring = (cv2.dilate(core.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~core & (L > 25) & (dom < 8)
    k = (float(np.median(L[ring])) / max(float(np.median(L[core])), 1.0)) if core.sum() > 100 and ring.sum() > 100 else 1.0
    grey = np.repeat((L * k)[..., None], 3, -1)
    return np.clip(f * (1 - m[..., None]) + grey * m[..., None], 0, 255).astype(np.uint8), round(k, 3)


# ---------------- stage 1: stills ----------------
EM = cv2.imread(R(C["empty_plate"]))
assets = os.path.join(OUT, "_assets"); os.makedirs(assets, exist_ok=True)
prep_info = {}
BLK = 8   # frames per still in the static clip; graded at the block centre
for form, F in C["forms"].items():
    st = read_frame(R(F["clip"]), F["start"]); pk = read_frame(R(F["clip"]), F["peak"])
    nfix = 0
    if SP.get("stack_from_empty"):
        pk, nfix = stack_from_empty(pk, EM); st, _ = stack_from_empty(st, EM)
    stp, pkp = prep(st), prep(pk)
    st0, ks = deglow(stp); pk0, kp_ = deglow(pkp)
    cv2.imwrite(os.path.join(OUT, f"src_{form}_peak_stackfix_720.png"), pk)
    vw = cv2.VideoWriter(os.path.join(assets, f"{form}_stills.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 24.0, (720, 1280))
    for im in (st0, pk0, pkp):
        for _ in range(BLK):
            vw.write(im)
    vw.release()
    prep_info[form] = {"stack_px_from_empty": nfix, "deglow_level_k": [ks, kp_]}
from gymform import ctx, render, native, layout, bg as gbg, region_glow as rg, joint_callout as jc
import build_4step_bone_muscle_master as b4
cfg = copy.deepcopy(json.load(open(R(C["base_config"]))))
cfg["assets_dir"] = assets; cfg["kp_cache"] = os.path.join(OUT, "_kps_stills.pkl"); cfg["tmp_dir"] = os.path.join(ROOT, "_tmp_still_motion")
for form in ("wrong", "right"):
    m = cfg["motions"][form]; fn = f"{form}_stills.avi"
    m["base"] = fn; m["i0"] = 0; m["i1"] = 3 * BLK - 1; m["renders"]["front"]["file"] = fn
ctx.configure(cfg)
os.makedirs(ctx.TMP["dir"], exist_ok=True)
_orig_cm = rg.clamp_mask
ARM = SP.get("arm_limit_right")


def _clamp_arm(kp, is_wrong, shape):   # right form: visible upper arm (triceps) capped - see tools/still_motion_mock.py
    m, ms = _orig_cm(kp, is_wrong, shape)
    if is_wrong or ARM is None:
        return m, ms
    tq = np.zeros(shape, np.uint8)
    cv2.fillPoly(tq, [np.array([kp[k] for k in ("sh_r", "sh_l", "hip_l", "hip_r")], np.int32)], 1)
    cap = np.zeros(shape, np.float32)
    for sd in ("r", "l"):
        sh, el = np.array(kp[f"sh_{sd}"], np.float32), np.array(kp[f"elb_{sd}"], np.float32)
        ln = np.zeros(shape, np.uint8); cv2.line(ln, tuple(int(v) for v in sh), tuple(int(v) for v in el), 1, 3)
        if (ln & tq).sum() > 0.5 * max(ln.sum(), 1):
            continue
        q0 = sh + 0.38 * (el - sh); q1 = el + 0.08 * (el - sh)
        c = np.zeros(shape, np.uint8)
        cv2.line(c, tuple(int(v) for v in q0), tuple(int(v) for v in q1), 1, max(4, int(0.55 * np.linalg.norm(el - sh))))
        cap = np.maximum(cap, c.astype(np.float32))
    return m * (1.0 - cv2.GaussianBlur(cap, (0, 0), 6.0) * (1.0 - ARM)), ms


rg.clamp_mask = _clamp_arm
raw, kps = render.load_inputs()
G = {}
t_grade = time.time()
for s in ctx.steps():
    fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(s)
    form = "wrong" if is_w else "right"
    ctx.CFG["_cur_step"] = s
    gbg.set_step_plate(raw[s], kps[s], s)
    native.prepare(s, raw[s], kps[s], y_sh)
    for tag, idx in (("start", BLK // 2), ("peak0", BLK + BLK // 2), ("peak1", 2 * BLK + BLK // 2)):
        base, sh_m, w_fg, src, kp = native.grade(s, idx, is_w)
        G[(form, tag)] = {"rgb": base.copy(), "w_fg": w_fg.copy(), "kp": copy.deepcopy(kp)}
    native._ST.clear(); gc.collect()
grade_s = time.time() - t_grade
# callout placement on each still (text on clean backdrop, never on the target regions / machine)
META = {}
for form, F in C["forms"].items():
    isw = form == "wrong"
    for tag in ("start", "peak1"):
        g = G[(form, tag)]
        cm, ms = rg.clamp_mask(g["kp"], isw, (H, W))
        g["allowed"] = cm
        _, info = jc.draw(g["rgb"], g["kp"], protect=cm, is_wrong=isw, label=F["label"])
        g["callout"] = info
    G[(form, "peak0")]["callout"] = G[(form, "peak1")]["callout"]
    cv2.imwrite(os.path.join(OUT, f"still_{form}_start_1080.png"), cv2.cvtColor(G[(form, "start")]["rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT, f"still_{form}_peak_1080.png"), cv2.cvtColor(G[(form, "peak1")]["rgb"], cv2.COLOR_RGB2BGR))
# QC on stills: glow outside the allowed regions; start glow amount; head vs body tone; Day 4 look metrics
QC = {"prep": prep_info, "grade_s": round(grade_s, 1)}
for form in C["forms"]:
    isw = form == "wrong"
    for tag in ("start", "peak1"):
        f32 = G[(form, tag)]["rgb"].astype(np.float32)
        dom = (f32[..., 0] - np.maximum(f32[..., 1], f32[..., 2])) if isw else (f32[..., 1] - np.maximum(f32[..., 0], f32[..., 2]))
        al = cv2.dilate((G[(form, "peak1")]["allowed"] > 0.02).astype(np.uint8), np.ones((21, 21), np.uint8))
        QC[f"{form}_{tag}"] = {"glow_px": int((dom > 25).sum()), "spill_px": int(((dom > 25) & (al == 0)).sum()),
                              "deg": G[(form, tag)]["callout"]["deg"], "text_box_obstacle_frac": G[(form, tag)]["callout"]["text_box_obstacle_frac"],
                              "leader_on_target_frac": G[(form, tag)]["callout"]["leader_on_target_muscle_frac"]}
    kp = G[(form, "peak1")]["kp"]
    sh = np.array(kp["sh"]); hp = 0.5 * (np.array(kp["hip_r"]) + np.array(kp["hip_l"]))
    QC[f"{form}_torso_px"] = round(float(np.linalg.norm(sh - hp)), 1)
ref = {"wrong": "output/_consistency/d7m/day4_master_t7.5_wrong.jpg", "right": "output/_consistency/d7m/day4_master_t17.5_right.jpg"}
for form in C["forms"]:
    r_ = subprocess.run([sys.executable, f"{ROOT}/tools/day4_look_metrics.py", f"day4={R(ref[form])}:{form}:356.4",
                         f"d7={OUT}/still_{form}_peak_1080.png:{form}:{QC[form + '_torso_px']}"], capture_output=True, text=True)
    try:
        QC[f"metrics_{form}_vs_day4"] = json.loads(r_.stdout[r_.stdout.index("{\n"):])
    except Exception:
        QC[f"metrics_{form}_vs_day4"] = {"error": r_.stderr[-400:]}
json.dump(QC, open(os.path.join(OUT, "stills_qc.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in QC.items() if not k.startswith("metrics")}, indent=1), flush=True)
if A.stills_only:
    sys.exit(0)

# ---------------- stage 2: motion ----------------
TL = C["timeline"]; FPS = TL["fps"]; DUR = TL["duration"]; Z = TL["push_in"]
NF = int(round(FPS * DUR))
ICONS = [b4._make_icon_pair(G[("wrong", "peak1")]["rgb"], G[("wrong", "peak1")]["kp"]),
         b4._make_icon_pair(G[("right", "peak1")]["rgb"], G[("right", "peak1")]["kp"])]
zc = {f: 0.5 * (np.array(G[(f, "peak1")]["kp"]["sh"]) + 0.5 * (np.array(G[(f, "peak1")]["kp"]["hip_r"]) + np.array(G[(f, "peak1")]["kp"]["hip_l"]))) for f in ("wrong", "right")}
ZC = 0.5 * (zc["wrong"] + zc["right"])
if TL.get("zoom_center_x") is not None:
    ZC = np.array([float(TL["zoom_center_x"]), ZC[1]])
ease = lambda x: 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, x)))
lin = lambda t, a, b: min(1.0, max(0.0, (t - a) / (b - a)))
pulse = lambda t: 0.84 + 0.16 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 1.6))
# keyframes (s): wrong hook 0-0.5 | ->start 0.5-1.0 | hold 1.0-1.6 | ->peak 1.6-2.2, glow 2.0-2.8 | hold ..5.4 |
# ->right start 5.4-6.0 | hold 6.0-6.6 | ->peak 6.6-7.2, glow 7.0-7.8 | hold ..11.4 | -> wrong peak 11.4-12.0 (loop)
SW = DUR / 12.0
K = {k: v * SW for k, v in dict(h=0.5, a0=0.5, a1=1.0, b0=1.6, b1=2.2, g0=2.0, g1=2.8, x0=5.4, x1=6.0, c0=6.6, c1=7.2,
                                   r0=7.0, r1=7.8, y0=11.4, y1=12.0).items()}


def state(t):
    """-> list of (form, pose_mix 0 start..1 peak, glow 0..1, weight)."""
    if t < K["a0"]:
        return [("wrong", 1.0, pulse(t), 1.0)]
    if t < K["a1"]:
        a = ease(lin(t, K["a0"], K["a1"]))
        return [("wrong", 1.0, pulse(t) * (1 - a), 1 - a), ("wrong", 0.0, 0.0, a)]
    if t < K["b0"]:
        return [("wrong", 0.0, 0.0, 1.0)]
    if t < K["x0"]:
        a = ease(lin(t, K["b0"], K["b1"])); gl = ease(lin(t, K["g0"], K["g1"])) * (pulse(t) if t > K["g1"] else 1.0)
        return [("wrong", 0.0, 0.0, 1 - a), ("wrong", 1.0, gl, a)] if a < 1 else [("wrong", 1.0, gl, 1.0)]
    if t < K["x1"]:
        a = ease(lin(t, K["x0"], K["x1"]))
        return [("wrong", 1.0, pulse(t), 1 - a), ("right", 0.0, 0.0, a)]
    if t < K["c0"]:
        return [("right", 0.0, 0.0, 1.0)]
    if t < K["y0"]:
        a = ease(lin(t, K["c0"], K["c1"])); gl = ease(lin(t, K["r0"], K["r1"])) * (pulse(t) if t > K["r1"] else 1.0)
        return [("right", 0.0, 0.0, 1 - a), ("right", 1.0, gl, a)] if a < 1 else [("right", 1.0, gl, 1.0)]
    a = ease(lin(t, K["y0"], K["y1"]))
    return [("right", 1.0, pulse(t), 1 - a), ("wrong", 1.0, pulse(t) * a + (1 - a) * pulse(t), a)]


def zoom(t):
    if t < K["x0"]:
        return 1.0 + (Z - 1.0) * ease(t / K["x0"])
    if t < K["x1"]:
        return Z - (Z - 1.0) * ease(lin(t, K["x0"], K["x1"]))
    if t < K["y0"]:
        return 1.0 + (Z - 1.0) * ease(lin(t, K["x1"], K["y0"]))
    return Z - (Z - 1.0) * ease(lin(t, K["y0"], K["y1"]))


def still(form, pose, glow):
    if pose < 0.5:
        return G[(form, "start")]["rgb"]
    p0, p1 = G[(form, "peak0")]["rgb"], G[(form, "peak1")]["rgb"]
    return p1 if glow >= 0.999 else (p0 if glow <= 0.001 else cv2.addWeighted(p0, 1 - glow, p1, glow, 0))


def badge(fr, slot, is_wrong):
    """X (wrong) / check (right) badge on the active thumbnail's top-right corner."""
    cx = [461, 619][slot] + 58; cy = 252; r = 17; ss = 4
    t = np.zeros((2 * r + 12) * ss, np.uint8); t = np.zeros(((2 * r + 12) * ss, (2 * r + 12) * ss), np.uint8)
    c = ((r + 6) * ss, (r + 6) * ss)
    disc = t.copy(); cv2.circle(disc, c, r * ss, 255, -1, cv2.LINE_AA)
    mk = t.copy()
    if is_wrong:
        o = int(0.42 * r * ss)
        cv2.line(mk, (c[0] - o, c[1] - o), (c[0] + o, c[1] + o), 255, int(3.2 * ss), cv2.LINE_AA)
        cv2.line(mk, (c[0] - o, c[1] + o), (c[0] + o, c[1] - o), 255, int(3.2 * ss), cv2.LINE_AA)
    else:
        q = r * ss
        pts = np.array([(c[0] - 0.45 * q, c[1] + 0.02 * q), (c[0] - 0.12 * q, c[1] + 0.34 * q), (c[0] + 0.48 * q, c[1] - 0.32 * q)], np.int32)
        cv2.polylines(mk, [pts], False, 255, int(3.2 * ss), cv2.LINE_AA)
    n = 2 * r + 12
    da = cv2.resize(disc, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    ma = cv2.resize(mk, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    col = np.array(jc.RED if is_wrong else jc.GREEN, np.float32)
    y0, x0 = cy - r - 6, cx - r - 6
    roi = fr[y0:y0 + n, x0:x0 + n].astype(np.float32)
    sh = cv2.GaussianBlur(da, (0, 0), 3.0)[..., None] * 0.45
    roi = roi * (1 - sh)
    roi = roi * (1 - da) + col * da
    roi = roi * (1 - ma) + 255.0 * ma
    fr[y0:y0 + n, x0:x0 + n] = np.clip(roi, 0, 255).astype(np.uint8)
    return fr


def frame_at(i):
    t = i / FPS
    st = state(t)
    s = zoom(t)
    M = np.float32([[s, 0, (1 - s) * ZC[0]], [0, s, (1 - s) * ZC[1]]])
    if len(st) == 1:
        img = still(*st[0][:3])
    else:
        (f1, p1, g1, w1), (f2, p2, g2, w2) = st
        img = cv2.addWeighted(still(f1, p1, g1), w1, still(f2, p2, g2), w2, 0)
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # callout: interpolate joint / box / number between the start and peak placements of the visible form(s)
    for form, pose, glow, w in st:
        cs, cp = G[(form, "start")]["callout"], G[(form, "peak1")]["callout"]
        if len(st) == 2 and st[0][0] == st[1][0]:   # same-form crossfade start<->peak: one moving callout
            pm = st[1][3] if st[1][1] > st[0][1] else st[0][3]
            if form != st[0][0] or (form, pose) != (st[0][0], st[0][1]):
                continue
            pose_m, wgt = pm if st[1][1] > st[0][1] else 1 - pm, 1.0
        else:
            pose_m, wgt = pose, w
        lerp = lambda a, b: (1 - pose_m) * np.array(a, np.float64) + pose_m * np.array(b, np.float64)
        j = lerp(cs["joint"], cp["joint"]); bxy = lerp(cs["text_box"][:2], cp["text_box"][:2])
        deg = (1 - pose_m) * float(cs["deg"][:-1]) + pose_m * float(cp["deg"][:-1])
        j = s * j + (1 - s) * ZC; bxy = s * bxy + (1 - s) * ZC
        img = jc.blend_tile(img, j, bxy, f"{int(round(deg))}\u00b0", C["forms"][form]["label"], form == "wrong", opacity=wgt)
    # UI: title + 2-thumb preview (active = dominant form) + badge
    dom = max(st, key=lambda z: z[3])[0]
    slot = 0 if dom == "wrong" else 1
    ctx.LAST_MASKS["eq_front"] = np.zeros((H, W), np.float32)
    fr = layout.composite_title_and_preview_v4(img, np.zeros((H, W), np.float32), ICONS, slot, dom == "wrong", cfg["exercise_name"])
    return badge(fr, slot, dom == "wrong")


tmp = ctx.TMP["dir"]
vid = os.path.join(tmp, "video.mp4")
t_mot = time.time()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                       "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", vid],
                      stdin=subprocess.PIPE)
SHEET_T = [0.3 * SW, 1.3 * SW, 6.3 * SW, 9.5 * SW]
sheet = {}
for i in range(NF):
    fr = frame_at(i)
    ff.stdin.write(fr.tobytes())
    for ts in SHEET_T:
        if i == int(round(ts * FPS)):
            sheet[ts] = fr.copy()
    if i == 0:
        f0 = fr.copy()
    if i == NF - 1:
        fl = fr.copy()
ff.stdin.close(); ff.wait()
motion_s = time.time() - t_mot
# audio: gymform.tts voice + Day 4 bed / chimes (220 Hz wrong, 528 Hz right) at each segment start
from gymform import tts
SR = tts.SR; n = int(SR * DUR); tt = np.arange(n, dtype=np.float32) / SR
beat = 0.45 + 0.55 * (np.sin(2.0 * math.pi * (100.0 / 60.0) * tt) > 0.72).astype(np.float32)
master = 0.040 * np.sin(2.0 * math.pi * 55.0 * tt) * beat
segs = []
for k, (form, ts, limit) in enumerate((("wrong", 0.2 * SW, K["x0"] + 0.2), ("right", K["x1"] + 0.1, K["y0"] + 0.4))):
    s0, dc = int((ts - 0.15) * SR), int(0.24 * SR); tc = np.linspace(0.0, 0.24, dc, endpoint=False)
    master[s0:s0 + dc] += 0.085 * np.sin(2.0 * math.pi * (220.0 if form == "wrong" else 528.0) * tc) * np.exp(-tc * 13.0)
    vo = asyncio.run(tts._say(C["forms"][form]["voice"], k))
    s_ = int(ts * SR); e_ = min(n, s_ + len(vo)); master[s_:e_] += vo[:e_ - s_] * 0.92
    segs.append({"form": form, "start": round(ts, 2), "end": round(ts + len(vo) / SR, 2), "limit": round(limit, 2),
                 "fits": ts + len(vo) / SR <= limit})
wav = os.path.join(tmp, "audio.wav")
with wave.open(wav, "wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(np.repeat((np.clip(master, -0.96, 0.96) * 32767).astype(np.int16), 2).tobytes())
MP4 = R(C["out_mp4"])
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "160k", "-shortest", "-movflags", "+faststart", MP4], check=True)
tiles = []
for ts in SHEET_T:
    tl = cv2.resize(cv2.cvtColor(sheet[ts], cv2.COLOR_RGB2BGR), (360, 640), interpolation=cv2.INTER_AREA)
    cv2.putText(tl, f"t={ts:.1f}s", (10, 628), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    tiles.append(tl)
SHEET = os.path.join(OUT, "reel_4frame_sheet.jpg")
cv2.imwrite(SHEET, np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 90])
QC.update({"mp4": MP4, "sheet": SHEET, "frames": NF, "fps": FPS, "duration_s": DUR, "voice": segs,
           "loop_seam_mean_abs": round(float(np.abs(f0.astype(int) - fl.astype(int)).mean()), 2),
           "time_s": {"grade_stills": round(grade_s, 1), "motion_frames_encode": round(motion_s, 1), "total": round(time.time() - T0, 1)},
           "generations": 0})
json.dump(QC, open(os.path.join(OUT, "reel_qc.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in QC.items() if not k.startswith("metrics") and not k.startswith(("wrong_", "right_"))}, indent=1))
