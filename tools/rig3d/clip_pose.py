#!/usr/bin/env python3
"""Extract MediaPipe pose (2D image + 3D world landmarks) from a reference clip, and auto-check it as a motion source.
usage: python3 tools/rig3d/clip_pose.py CLIP.mp4 OUT.npz [--side auto|left|right]
Checks: full body visible (all key landmarks visibility>0.5 and inside the frame), torso hinge 35-60 deg (median),
>=2 clean reps of the working elbow angle, jitter (median |2nd difference| of wrist world position, metres)."""
import sys, json, cv2, numpy as np, mediapipe as mp
sys.path.insert(0, "/workspace/daily_info_reels")
from universal_anatomy_rig import get_pose_detector
from scipy.signal import find_peaks
clip, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(clip); fps = cap.get(cv2.CAP_PROP_FPS); det = get_pose_detector(); P2, P3 = [], []
while True:
    ok, f = cap.read()
    if not ok: break
    h, w = f.shape[:2]; r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks: P2.append(np.full((33, 3), np.nan)); P3.append(np.full((33, 3), np.nan)); continue
    P2.append([[p.x * w, p.y * h, p.visibility] for p in r.pose_landmarks[0]]); P3.append([[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]])
P2, P3 = np.array(P2, float), np.array(P3, float); n = len(P2)
key = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
vis = np.nanmean((P2[:, key, 2] > 0.5) & (P2[:, key, 0] > 2) & (P2[:, key, 0] < w - 2) & (P2[:, key, 1] > 2) & (P2[:, key, 1] < h - 2), axis=1)
full = float(np.mean(vis >= 0.99))
inside = np.all((P2[:, key, 0] > 2) & (P2[:, key, 0] < w - 2) & (P2[:, key, 1] > 2) & (P2[:, key, 1] < h - 2), axis=1)
inframe = float(np.mean(inside))
sh = (P2[:, 11, :2] + P2[:, 12, :2]) / 2; hp = (P2[:, 23, :2] + P2[:, 24, :2]) / 2
hinge = np.degrees(np.arctan2(np.abs(sh[:, 0] - hp[:, 0]), hp[:, 1] - sh[:, 1]))   # forward lean from vertical (image plane)
def elbow(s):
    a, b, c = P3[:, s], P3[:, s + 2], P3[:, s + 4]; v1, v2 = a - b, c - b
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9), -1, 1)))
e = {11: elbow(11), 12: elbow(12)}; side = max(e, key=lambda s: np.nanstd(e[s]))   # working arm = the one that moves most
flex = 180 - e[side]; fs = np.convolve(np.nan_to_num(flex, nan=np.nanmedian(flex)), np.ones(5) / 5, "same")
pk, _ = find_peaks(fs, prominence=35, distance=int(fps * 1.0)); tr, _ = find_peaks(-fs, prominence=35, distance=int(fps * 1.0))
wr = P3[:, side + 4]; jit = float(np.nanmedian(np.linalg.norm(wr[2:] - 2 * wr[1:-1] + wr[:-2], axis=1)))
tl = np.nanmedian(np.linalg.norm(sh - hp, axis=1)); w2 = P2[:, side + 4, :2] / tl
jit2 = float(np.nanmedian(np.linalg.norm(w2[2:] - 2 * w2[1:-1] + w2[:-2], axis=1)))   # 2D, in torso lengths
res = {"clip": clip, "frames": n, "fps": fps, "size": [w, h], "working_side": "left" if side == 11 else "right",
       "full_body": {"all_key_visible_frac": round(full, 3), "all_key_in_frame_frac": round(inframe, 3), "pass": inframe >= 0.95},
       "hinge": {"median_deg": round(float(np.nanmedian(hinge)), 1), "p10_p90": [round(float(np.nanpercentile(hinge, 10)), 1), round(float(np.nanpercentile(hinge, 90)), 1)], "pass": 35 <= np.nanmedian(hinge) <= 60},
       "reps": {"peaks_f": pk.tolist(), "bottoms_f": tr.tolist(), "flex_range": [round(float(np.nanmin(fs)), 1), round(float(np.nanmax(fs)), 1)], "pass": len(pk) >= 2},
       "jitter": {"wrist3d_m_per_f2": round(jit, 4), "wrist2d_torso_per_f2": round(jit2, 4), "pass": jit2 <= 0.02}}
res["pass"] = all(res[k]["pass"] for k in ("full_body", "hinge", "reps", "jitter"))
np.savez(out, P2=P2, P3=P3, flex=fs, hinge=hinge, side=side, fps=fps); s = json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(s); open(out.replace(".npz", "_check.json"), "w").write(s)
