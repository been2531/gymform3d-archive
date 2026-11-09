#!/usr/bin/env python3
"""t665u deterministic flat relight of the mannequin in a pinned D7 still (0 generations).
Machine pixels stay byte-identical: only pixels that differ from the EMPTY-machine pin (= the person) are touched,
and inside the person the matte-black shorts (MediaPipe hip->thigh zone, dark) and the chrome handle (bright, neutral)
are kept. On the mannequin skin the LOW-frequency shading (sigma 14 base) is compressed toward a flat level above
the backdrop (base' = T + k (base - T)); fine detail (fibres, edges) and the glow chroma (rgb - lum) are kept as is,
so the source glow the Day 4 shader reads stays where it was.
usage: python3 tools/d7_mannequin_relight.py <pin> <empty_pin> <out> [--T 82] [--k 0.30] [--floor 58]"""
import sys, argparse, cv2, numpy as np
sys.path.insert(0, "/workspace/daily_info_reels")
ap = argparse.ArgumentParser(); ap.add_argument("pin"); ap.add_argument("empty"); ap.add_argument("out")
ap.add_argument("--T", type=float, default=82.0); ap.add_argument("--k", type=float, default=0.30); ap.add_argument("--floor", type=float, default=58.0)
ap.add_argument("--mask", default="", help="person mask in pin space (the still_person_composite mask, warped): exact person region")
ap.add_argument("--deglow", action="store_true", help="t665u: matte neutral charcoal skin, no glow (glow comes from code: muscle_glow_source regions)")
a = ap.parse_args()
im = cv2.imread(a.pin).astype(np.float32); em = cv2.imread(a.empty).astype(np.float32)
d = np.abs(cv2.GaussianBlur(im, (0, 0), 1.5) - cv2.GaussianBlur(em, (0, 0), 1.5)).max(2)
raw_p = (d > 10).astype(np.uint8)
person = cv2.morphologyEx(raw_p, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(person)
person = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 3000]).astype(np.uint8)
person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) & cv2.dilate(raw_p, np.ones((9, 9), np.uint8))
person &= cv2.dilate(raw_p, np.ones((3, 3), np.uint8)) | (cv2.erode(person, np.ones((9, 9), np.uint8)))   # interior holes filled, rim never past the person
if a.mask:
    d0 = np.abs(im - em).max(2)
    ev = cv2.morphologyEx((d0 > 4).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    person = ((cv2.imread(a.mask, 0) > 127) & (cv2.dilate(ev, np.ones((5, 5), np.uint8)) > 0)).astype(np.uint8)   # pasted AND differs from the empty machine
b, g, r = im[..., 0], im[..., 1], im[..., 2]
L = 0.299 * r + 0.587 * g + 0.114 * b
sat = im.max(2) - im.min(2)
# shorts zone from the pose
import universal_anatomy_rig as uar
res = uar.get_pose_detector().detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))))
lm = res.pose_landmarks[0]; H, W = L.shape
P = lambda i: np.array([lm[i].x * W, lm[i].y * H])
hip = [P(23), P(24)]; kn = [P(25), P(26)]; sh = 0.5 * (P(11) + P(12))
tl = float(np.linalg.norm(sh - 0.5 * (hip[0] + hip[1])))
zone = np.zeros((H, W), np.uint8)
poly = np.array([hip[0], hip[1], hip[1] + 0.6 * (kn[1] - hip[1]), hip[0] + 0.6 * (kn[0] - hip[0])], np.int32)
cv2.fillConvexPoly(zone, cv2.convexHull(poly), 1)
zone = cv2.dilate(zone, np.ones((int(0.35 * tl) | 1, int(0.35 * tl) | 1), np.uint8))
shorts = (zone > 0) & (cv2.GaussianBlur(L, (0, 0), 3) < 34) & (sat < 14)
shorts = cv2.morphologyEx(shorts.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
shorts = cv2.morphologyEx(shorts, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
# fabric folds / rim are brighter than the core: grow the shorts geodesically over dark neutral pixels in the zone
grow = ((zone > 0) & (L < 58) & (sat < 16)).astype(np.uint8)
for _ in range(12):
    shorts = np.maximum(shorts, cv2.dilate(shorts, np.ones((3, 3), np.uint8)) & grow)
shorts = cv2.dilate(shorts, np.ones((3, 3), np.uint8)) > 0
handle = (L > 115) & (sat < 20)
handle = cv2.dilate(handle.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
cable = (L < 30) & (sat < 10) & ~(cv2.erode(person, np.ones((15, 15), np.uint8)) > 0)   # thin dark cable at the person edge
# keypoint body capsule (incl. shins/feet): pasted machine pixels around the person (footplate, pulley) are never skin
capm = np.zeros((H, W), np.uint8)
Pi = lambda i: tuple(int(v) for v in P(i))
cv2.fillConvexPoly(capm, np.array([P(11), P(12), P(24), P(23)], np.int32), 1)
for i, j in ((11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (11, 12), (23, 24), (15, 19), (16, 20)):
    cv2.line(capm, Pi(i), Pi(j), 1, max(8, int(0.30 * tl)))
cv2.circle(capm, tuple(int(v) for v in 0.5 * (P(7) + P(8))), int(0.36 * tl), 1, -1)   # head (ears)
cv2.line(capm, tuple(int(v) for v in 0.5 * (P(7) + P(8))), tuple(int(v) for v in sh), 1, int(0.30 * tl))
skin = (person > 0) & ~shorts & ~handle & ~cable & (capm > 0)
w = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 1.5)
# low-frequency shading on skin only (normalised blur so shorts / backdrop never bleed into the base)
sk = skin.astype(np.float32)
base = cv2.GaussianBlur(L * sk, (0, 0), 14) / np.maximum(cv2.GaussianBlur(sk, (0, 0), 14), 1e-3)
det = L - base
nb = a.T + a.k * (base - a.T)
Ln = np.maximum(nb + det, a.floor + 0.3 * (det.clip(-40, 0)))   # deep crevices keep a little depth, never black
chroma = im - L[..., None]
if a.deglow:
    chroma = chroma * 0.0
out = Ln[..., None] + chroma
res_ = im * (1 - w[..., None]) + out * w[..., None]
if a.deglow:   # any remaining glow-coloured person pixel (outside the capsule rim) -> neutral at its own luminance
    domg = np.maximum(r - np.maximum(g, b), g - np.maximum(r, b))
    gw = cv2.GaussianBlur(((domg > 8) & (person > 0) & ~shorts).astype(np.float32), (0, 0), 1.5)[..., None] * (1 - w[..., None])
    res_ = res_ * (1 - gw) + np.repeat(L[..., None], 3, -1) * gw
res_ = np.clip(res_, 0, 255).astype(np.uint8)
mach = (person == 0)
res_[mach] = im.astype(np.uint8)[mach]   # machine / backdrop byte-identical
cv2.imwrite(a.out, res_)
cv2.imwrite(a.out.replace(".png", "_mask.png"), (np.stack([skin, shorts, handle], -1) * 255).astype(np.uint8))
Ls = L[skin]; Lo = (0.299 * res_[..., 2] + 0.587 * res_[..., 1] + 0.114 * res_[..., 0]).astype(np.float32)[skin]
bg = float(np.median(L[:300, :150]))
print({"skin_px": int(skin.sum()), "backdrop": round(bg, 1), "before_p5_50_90": np.percentile(Ls, [5, 50, 90]).round(1).tolist(),
       "after_p5_50_90": np.percentile(Lo, [5, 50, 90]).round(1).tolist(), "machine_px_changed": int((np.abs(res_.astype(int) - im.astype(int)).max(2)[mach] > 0).sum())})
