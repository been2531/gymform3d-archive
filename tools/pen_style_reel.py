#!/usr/bin/env python3
"""t672u pen-line single-segment reel (opt-in, 0 generations): D7 Seated Cable Row RIGHT form only.

Per frame of the existing clip (default output/assets/day7_right_raw_v3.mp4, first = last frame):
  pose (MediaPipe, interpolated + temporally smoothed) -> mannequin_relight.relight masks (skin / shorts / handle /
  person vs the EMPTY machine plate) -> temporal 3-tap smoothing of the line source, tone source and masks (no line
  flicker) -> gymform.pen_style line art (machine traced from the static empty plate = pixel-exact and flicker-free)
  -> green pen-line muscles (region_glow regions: lats, rhomboids/mid trap, rear delt, faint biceps; translucent wash,
  fibre hatch, re-inked creases, fibres anchored to the body) -> Day 4 template (1-thumb preview, Kaushan title,
  measured elbow angle + one label, hide_lines) -> 2 loops (first = last) -> voice (gymform.tts voice + bed) -> mp4.
usage: nice -n 19 python3 tools/pen_style_reel.py [--check] [--loops 2] [--src clip.mp4] [--out dir]"""
import sys, os, json, time, argparse, subprocess, shutil, asyncio, math, wave
import cv2, numpy as np
ROOT = "/workspace/daily_info_reels"
sys.path.insert(0, ROOT)
from gymform import pen_style as ps
from gymform import mannequin_relight as mr

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=f"{ROOT}/output/assets/day7_right_raw_v3.mp4")
ap.add_argument("--empty", default=f"{ROOT}/output/_consistency/d7m/pin_empty_machine_720x1280.png")
ap.add_argument("--out", default=f"{ROOT}/output/_consistency/pen_style")
ap.add_argument("--check", action="store_true", help="render ONE check frame (peak) only")
ap.add_argument("--loops", type=int, default=2)
ap.add_argument("--wash", type=float, default=0.55)
ap.add_argument("--line", default="Chest up, shoulders down. Drive your elbows back, and let your lats do the work.")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
W, H = 1080, 1920
T0 = time.time()
cfg = json.load(open(f"{ROOT}/configs/day07_muscle2.json"))
TARGETS = (cfg.get("target_regions") or {}).get("right") or {"lats": 1.0, "mid_trap_rhomboid": 1.0, "rear_delt": 1.0, "biceps": 0.35}

# ---------- load clip (mirror = D7 framing, tower right) ----------
cap = cv2.VideoCapture(a.src); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.flip(f, 1))
EM = cv2.flip(cv2.imread(a.empty), 1)
assert EM.shape == frames[0].shape, (EM.shape, frames[0].shape)
loop_same = float(np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean())
N = len(frames) - 1 if loop_same < 3.0 else len(frames)      # first = last: drop the duplicate, loop seamlessly
frames = frames[:N]
# ---------- pose: per frame, interpolated, circular temporal smoothing ----------
kp_cache = f"{a.out}/_pose_right_{N}.npy"
if os.path.exists(kp_cache):
    K = np.load(kp_cache)
else:
    raw = [mr.detect_pose(f) for f in frames]
    idx = [i for i, k in enumerate(raw) if k is not None]
    Ks = np.stack([raw[i] for i in idx])
    K = np.stack([np.stack([np.interp(np.arange(N), idx, Ks[:, j, c], period=N) for c in range(2)], -1) for j in range(33)], 1)
    np.save(kp_cache, K)
sig = 1.5; r_ = int(3 * sig + 0.5); kern = np.exp(-0.5 * (np.arange(-r_, r_ + 1) / sig) ** 2); kern /= kern.sum()
K = np.stack([np.tensordot(kern, K[[(t + d) % N for d in range(-r_, r_ + 1)]], axes=(0, 0)) for t in range(N)])
S = 1080.0 / 720.0


def kp_1080(t):
    P = K[t] * S
    A, B = (11, 13, 15, 23, 25), (12, 14, 16, 24, 26)
    r, l = (A, B) if P[11][0] <= P[12][0] else (B, A)
    k = {}
    for sd, ids in (("r", r), ("l", l)):
        for nm, i in zip(("sh", "elb", "wr", "hip", "kn"), ids):
            k[f"{nm}_{sd}"] = (float(P[i][0]), float(P[i][1]))
    k["sh"] = tuple(0.5 * (np.array(k["sh_r"]) + np.array(k["sh_l"])))
    k["head"] = tuple(0.5 * (P[7] + P[8]))
    return k


def visible_side(kp):
    tq = np.zeros((H, W), np.uint8)
    cv2.fillPoly(tq, [np.array([kp[k] for k in ("sh_r", "sh_l", "hip_l", "hip_r")], np.int32)], 1)
    best, sd_b = 2.0, "r"
    for sd in ("r", "l"):
        m = np.zeros((H, W), np.uint8)
        cv2.line(m, tuple(int(v) for v in kp[f"elb_{sd}"]), tuple(int(v) for v in kp[f"wr_{sd}"]), 1, 3)
        fr_ = (m & tq).sum() / max(m.sum(), 1)
        if fr_ < best:
            best, sd_b = fr_, sd
    return sd_b


def flexion(kp, sd):
    p, q, r = (np.array(kp[f"{j}_{sd}"]) for j in ("sh", "elb", "wr"))
    v1, v2 = p - q, r - q
    return 180.0 - float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))


# ---------- stage A (720): relight + masks, cached per frame ----------
_A = {}


def stage_a(t):
    t %= N
    if t not in _A:
        o, m = mr.relight(frames[t], EM, K[t])
        lab = np.stack([m["skin"], m["shorts"], m["handle"]], -1).astype(np.uint8) * 255
        _A[t] = (o, lab, m["person"].astype(np.uint8) * 255)
        for k in [k for k in _A if min((k - t) % N, (t - k) % N) > 2]:
            del _A[k]
    return _A[t]


up = lambda im, it=cv2.INTER_CUBIC: cv2.resize(im, (W, H), interpolation=it)
TW = (0.25, 0.5, 0.25)      # temporal 3-tap (t-1, t, t+1), circular


def smoothed(t):
    """temporally smoothed line source (raw), tone source (relit) and masks at 1080."""
    raw = sum(w_ * frames[(t + d) % N].astype(np.float32) for w_, d in zip(TW, (-1, 0, 1)))
    rel = sum(w_ * stage_a(t + d)[0].astype(np.float32) for w_, d in zip(TW, (-1, 0, 1)))
    lab = sum(w_ * stage_a(t + d)[1].astype(np.float32) for w_, d in zip(TW, (-1, 0, 1)))
    per = sum(w_ * stage_a(t + d)[2].astype(np.float32) for w_, d in zip(TW, (-1, 0, 1)))
    u8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)
    return up(u8(raw)), up(u8(rel)), up(u8(lab), cv2.INTER_LINEAR), up(u8(per), cv2.INTER_LINEAR)


E1080 = up(EM)
from gymform.region_glow import region_masks   # (imports gymform.ctx)
from gymform import ctx, layout
import build_4step_bone_muscle_master as b4
cfg1 = dict(cfg); cfg1["tmp_dir"] = f"{ROOT}/_tmp_pen"
ctx.configure(cfg1)
os.makedirs(ctx.TMP["dir"], exist_ok=True)


def pen(t):
    O, R, lab, per = smoothed(t)
    mach, body, fg, bg = ps.masks(E1080, R, per, lab)
    canvas, body_w, lay = ps.line_art(E1080, R, mach, body, fg, bg, O)
    kp = kp_1080(t)
    out, wgt, ms = ps.muscle_fill(canvas, O, body_w, kp, TARGETS, "right", ink=lay["ink_crease"], wash=a.wash)
    return np.clip(out, 0, 255).astype(np.uint8), body_w, wgt, ms, kp, lay


# measured angle: elbow flexion of the visible arm at the peak (max flexion over the clip)
fl = []
for t in range(N):
    kp = kp_1080(t); fl.append(flexion(kp, visible_side(kp)))
PEAK = int(np.argmax(fl)); DEG = f"{int(round(fl[PEAK]))}\u00b0"
L1 = cfg["motions"]["right"]["callout"]["l1"]
pk_rgb, *_r = pen(PEAK)
ICONS = [b4._make_icon_pair(pk_rgb, _r[3])]


def compose(rgb, body_w, kp):
    fr = layout.draw_callout_day4(rgb, np.zeros((H, W), np.float32), 2, kp, False, DEG, L1, "")
    ctx.LAST_MASKS["eq_front"] = np.zeros((H, W), np.float32)
    return layout.composite_title_and_preview_v4(fr, layout.fg_for_preview(body_w), ICONS, 0, False, cfg["exercise_name"])


def qc_frame(t, rgb_base, wgt, ms, kp, lay, fr):
    allowed = np.zeros((H, W), np.float32)
    for n_ in TARGETS:
        allowed = np.maximum(allowed, ms[n_])
    allowed_d = cv2.dilate(allowed, np.ones((41, 41), np.uint8))
    f32 = rgb_base.astype(np.float32)
    domg = f32[..., 1] - np.maximum(f32[..., 0], f32[..., 2])
    spill = int(((domg > 25) & (allowed_d == 0)).sum())
    # head tone: head disc vs upper torso (both drawn by the same pen fill) - must match within a few levels
    L = 0.299 * f32[..., 0] + 0.587 * f32[..., 1] + 0.114 * f32[..., 2]
    hd = np.zeros((H, W), np.uint8); cv2.circle(hd, tuple(int(v) for v in kp["head"]), 34, 1, -1)
    sh = np.array(kp["sh"]); hp = 0.5 * (np.array(kp["hip_r"]) + np.array(kp["hip_l"]))
    tr = np.zeros((H, W), np.uint8); cv2.circle(tr, tuple(int(v) for v in (0.6 * hp + 0.4 * sh)), 34, 1, -1)
    hd_b = (hd > 0) & (lay["ink_body"] < 0.2) & (wgt < 0.05)
    tr_b = (tr > 0) & (lay["ink_body"] < 0.2) & (wgt < 0.05)
    head_L = float(np.median(L[hd_b])) if hd_b.sum() > 50 else -1
    torso_L = float(np.median(L[tr_b])) if tr_b.sum() > 50 else -1
    return {"t": t, "spill_px_outside_regions": spill, "glow_px": int((domg > 25).sum()),
            "head_L": round(head_L, 1), "body_L": round(torso_L, 1)}


if a.check:
    t1 = time.time()
    rgb, body_w, wgt, ms, kp, lay = pen(PEAK)
    fr = compose(rgb, body_w, kp)
    p = f"{a.out}/pen_right_check_f{PEAK:03d}_1080.jpg"
    cv2.imwrite(p, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
    q = qc_frame(PEAK, rgb, wgt, ms, kp, lay, fr)
    print(json.dumps({"check": p, "N": N, "peak": PEAK, "deg": DEG, "label": L1, "frame_s": round(time.time() - t1, 2),
                      "loop_first_last_diff": round(loop_same, 2), "qc": q}, indent=1))
    sys.exit(0)

# ---------- full render ----------
tmpd = f"{ROOT}/_tmp_pen/frames"; shutil.rmtree(tmpd, ignore_errors=True); os.makedirs(tmpd)
t_render = time.time()
qcs, inks, prev_ink, ink_flick, mach_fid = [], None, None, [], []
from skimage.metrics import structural_similarity as ssim
for t in range(N):
    rgb, body_w, wgt, ms, kp, lay = pen(t)
    fr = compose(rgb, body_w, kp)
    cv2.imwrite(f"{tmpd}/f_{t:04d}.png", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if t % 6 == 0 or t == PEAK:
        qcs.append(qc_frame(t, rgb, wgt, ms, kp, lay, fr))
    ib = lay["ink_body"][::2, ::2]
    if prev_ink is not None:
        ink_flick.append(float(np.abs(ib - prev_ink).mean()))
    prev_ink = ib
    if t % 12 == 0:   # machine fidelity: raw clip vs empty plate outside the person (720, grey)
        per = stage_a(t)[2] > 0
        perd = cv2.dilate(per.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        g1 = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(EM, cv2.COLOR_BGR2GRAY)
        _, smap = ssim(g1, g2, full=True, data_range=255)
        mach_fid.append(round(float(smap[~perd].mean()), 4))
    if t % 20 == 0:
        print(f"frame {t}/{N} {time.time() - t_render:.0f}s", flush=True)
render_s = time.time() - t_render
# ---------- audio: gymform.tts voice + Day 4 bed (55 Hz pulse, right-form chime) ----------
from gymform import tts
DUR = a.loops * N / fps
SR = tts.SR
n = int(SR * DUR); tt = np.arange(n, dtype=np.float32) / SR
beat = 0.45 + 0.55 * (np.sin(2.0 * math.pi * (100.0 / 60.0) * tt) > 0.72).astype(np.float32)
master = 0.040 * np.sin(2.0 * math.pi * 55.0 * tt) * beat
for ts in [0.05] + [k * N / fps for k in range(1, a.loops)]:
    s0, dc = int(ts * SR), int(0.24 * SR); tc = np.linspace(0.0, 0.24, dc, endpoint=False)
    master[s0:s0 + dc] += 0.085 * np.sin(2.0 * math.pi * 528.0 * tc) * np.exp(-tc * 13.0)
vo = asyncio.run(tts._say(a.line, 0))
v0 = 0.35; s = int(v0 * SR); e = min(n, s + len(vo)); master[s:e] += vo[:e - s] * 0.92
vo_end = v0 + len(vo) / SR
wav = f"{ROOT}/_tmp_pen/pen_right.wav"
with wave.open(wav, "wb") as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(np.repeat((np.clip(master, -0.96, 0.96) * 32767).astype(np.int16), 2).tobytes())
mp4 = f"{ROOT}/output/day7_pen_right_seated_cable_row.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-stream_loop", str(a.loops - 1), "-framerate", str(fps),
                "-i", f"{tmpd}/f_%04d.png", "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-t", f"{DUR:.3f}", "-movflags", "+faststart", mp4], check=True)
# ---------- sheet (3 frames) ----------
pick = [0, PEAK, (PEAK + N // 2) % N]
tiles = [cv2.resize(cv2.imread(f"{tmpd}/f_{t:04d}.png"), (360, 640), interpolation=cv2.INTER_AREA) for t in pick]
for tl, t in zip(tiles, pick):
    cv2.putText(tl, f"t={t / fps:.2f}s", (10, 628), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
sheet = f"{a.out}/pen_right_reel_sheet.jpg"
cv2.imwrite(sheet, np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 90])
loop_seam = float(np.abs(cv2.imread(f"{tmpd}/f_0000.png").astype(int) - cv2.imread(f"{tmpd}/f_{N - 1:04d}.png").astype(int)).mean())
step_mean = float(np.mean([np.abs(cv2.imread(f"{tmpd}/f_{t:04d}.png").astype(int) - cv2.imread(f"{tmpd}/f_{t + 1:04d}.png").astype(int)).mean() for t in (10, 40, 70)]))
fl_arr = np.array(ink_flick)
rep = {"mp4": mp4, "sheet": sheet, "frames_unique": N, "fps": fps, "loops": a.loops, "duration_s": round(DUR, 2),
       "peak_frame": PEAK, "deg": DEG, "label": L1, "voice_line": a.line, "voice_end_s": round(vo_end, 2),
       "render_s_frames": round(render_s, 1), "total_s": round(time.time() - T0, 1), "s_per_frame": round(render_s / N, 2),
       "qc": {"spill_px_max": max(q["spill_px_outside_regions"] for q in qcs), "frames": qcs,
              "ink_flicker_mean": round(float(fl_arr.mean()), 4), "ink_flicker_p99": round(float(np.percentile(fl_arr, 99)), 4),
              "ink_flicker_max_over_median": round(float(fl_arr.max() / max(np.median(fl_arr), 1e-6)), 2),
              "machine_ssim_outside_person_min": min(mach_fid), "machine_lines_source": "static empty plate (identical every frame)",
              "loop_seam_mean_abs": round(loop_seam, 2), "frame_step_mean_abs": round(step_mean, 2),
              "head_minus_body_L_max": round(max(abs(q["head_L"] - q["body_L"]) for q in qcs if q["head_L"] >= 0 and q["body_L"] >= 0), 1)},
       "generations": 0}
json.dump(rep, open(f"{a.out}/pen_right_reel_qc.json", "w"), indent=1)
shutil.rmtree(tmpd, ignore_errors=True)
print(json.dumps({k: v for k, v in rep.items() if k != "qc"}, indent=1)); print(json.dumps({k: v for k, v in rep["qc"].items() if k != "frames"}, indent=1))
