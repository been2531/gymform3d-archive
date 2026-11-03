#!/usr/bin/env python3
"""Option-B motion test (t691s2456): evaluate a 2 s pinned start->peak clip and build the loop.
Loop = forward -> 0.3 s hold at peak -> reverse -> (repeat). Motion only, no grade.
Measures: machine SSIM vs the start pin outside a dilated body mask (+ cable/handle band excluded),
per-frame pose (MediaPipe heavy): torso-from-vertical, elbow flexion, wrist-wrist distance, wrist vs
shoulder x (elbows behind the back line), loop-seam deltas. usage: nice -n 19 python3 tools/optb_motion_test.py"""
import sys, json, cv2, numpy as np, mediapipe as mp
from skimage.metrics import structural_similarity as ssim
sys.path.insert(0, "/workspace/daily_info_reels")
from universal_anatomy_rig import get_pose_detector
R = "/workspace/daily_info_reels/"; O = R + "output/_consistency/optB/"
RAW = O + "d7_optB_right_pull_2s_720p_raw.mp4"
PIN_S = R + "output/_consistency/d7m/pin_right_fibre_head_t2_720x1280.png"
PIN_P = R + "output/_consistency/still_motion/reel/src_right_peak_stackfix_720.png"
c = cv2.VideoCapture(RAW); F = []
while True:
    ok, f = c.read()
    if not ok: break
    F.append(f)
F = np.stack(F); n = len(F); fps = 24
det = get_pose_detector(); kps = []; masks = []
for f in F:
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks: kps.append(None); masks.append(np.zeros(f.shape[:2], np.uint8)); continue
    L = r.pose_landmarks[0]; kps.append(np.array([[p.x * 720, p.y * 1280, p.visibility] for p in L]))
    masks.append((r.segmentation_masks[0].numpy_view()[..., 0] > 0.3).astype(np.uint8))
pin = cv2.imread(PIN_S); pg = cv2.cvtColor(pin, cv2.COLOR_BGR2GRAY)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
ROI = {"tower+stack": (430, 1100, 0, 190), "foot plate+rail": (1080, 1280, 0, 720), "seat post": (980, 1280, 560, 720)}
mach = {kk: [] for kk in ROI}; mach_all = []
def ang(a, b, c_):
    v1, v2 = a - b, c_ - b; return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))))
pose = []
for i, f in enumerate(F):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    excl = cv2.dilate(masks[i], k).astype(bool)
    _, smap = ssim(pg, g, full=True)
    for kk, (y0, y1, x0, x1) in ROI.items():
        m = ~excl[y0:y1, x0:x1]; mach[kk].append(float(smap[y0:y1, x0:x1][m].mean()) if m.sum() > 500 else None)
    band = np.zeros_like(excl); band[700:1000, 150:720] = True  # handle + cable travel band (moves by design)
    keep = ~(excl | band); mach_all.append(float(smap[keep].mean()))
    kp = kps[i]
    if kp is None: pose.append(None); continue
    sh = (kp[11, :2] + kp[12, :2]) / 2; hp = (kp[23, :2] + kp[24, :2]) / 2
    side = 11 if kp[11, 2] >= kp[12, 2] else 12  # visible-side arm
    e, w, s = kp[side + 2, :2], kp[side + 4, :2], kp[side, :2]
    tv = float(np.degrees(np.arctan2(sh[0] - hp[0], hp[1] - sh[1])))  # + = leaning toward +x (backward, away from tower)
    pose.append({"f": i, "torso_deg": round(tv, 1), "elbow_flex": round(180 - ang(s, e, w), 1),
                 "wrist_x": round(float(w[0]), 1), "elbow_minus_shoulder_x": round(float(e[0] - s[0]), 1),
                 "wrist_gap_px": round(float(np.linalg.norm(kp[15, :2] - kp[16, :2])), 1)})
good = [p for p in pose if p]
tor = [p["torso_deg"] for p in good]; ef = [p["elbow_flex"] for p in good]; wx = [p["wrist_x"] for p in good]
# reach-forward overshoot vs start pin: wrist x smaller than f0 (toward the tower)
over = round(float(good[0]["wrist_x"] - min(wx)), 1)
# loop seams: peak hold and reverse seam are identical frames by construction; the wrap seam is f0 -> f0 (identical)
d_end_pin = float(ssim(cv2.cvtColor(cv2.imread(PIN_P), cv2.COLOR_BGR2GRAY), cv2.cvtColor(F[-1], cv2.COLOR_BGR2GRAY)))
d_start_pin = float(ssim(pg, cv2.cvtColor(F[0], cv2.COLOR_BGR2GRAY)))
step = [float(np.abs(F[i + 1].astype(np.int16) - F[i]).mean()) for i in range(n - 1)]
qc = {"frames": n, "fps": fps, "machine_ssim_vs_start_pin_min": {kk: round(min(x for x in v if x is not None), 3) for kk, v in mach.items()},
      "machine_ssim_all_static_min": round(min(mach_all), 3), "machine_ssim_all_static_series": [round(x, 3) for x in mach_all],
      "start_pin_ssim_f0": round(d_start_pin, 3), "peak_pin_ssim_flast": round(d_end_pin, 3),
      "torso_deg_range": round(max(tor) - min(tor), 1), "torso_deg_minmax": [min(tor), max(tor)],
      "elbow_flex_minmax": [min(ef), max(ef)], "wrist_reach_overshoot_px": over,
      "frame_step_absdiff_mean_max": round(max(step), 2), "frame_step_absdiff_median": round(float(np.median(step)), 2),
      "pose_series": pose}
json.dump(qc, open(O + "optb_qc.json", "w"), indent=1)
# loop mp4: forward, hold 0.3 s (7 f), reverse (excluding the repeated end frames), 4 loops
hold = int(round(0.3 * fps)); cyc = list(range(n)) + [n - 1] * hold + list(range(n - 2, 0, -1))
seq = cyc * 4 + [0]
w = cv2.VideoWriter(O + "_loop_tmp.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (720, 1280))
for i in seq: w.write(F[i])
w.release()
import subprocess
subprocess.run(["nice", "-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-i", O + "_loop_tmp.avi", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", O + "d7_optB_right_loop_motion_only.mp4"], check=True)
import os; os.remove(O + "_loop_tmp.avi")
print(json.dumps({k: v for k, v in qc.items() if k not in ("pose_series", "machine_ssim_all_static_series")}, indent=1))
print("cycle frames", len(cyc), "loop s", round(len(seq) / fps, 2))
for p in good[::4]: print(p)
