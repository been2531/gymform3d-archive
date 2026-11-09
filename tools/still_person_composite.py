#!/usr/bin/env python3
"""STILL composite (t643u/t645 owner OK for stills, never for video frames): keep the EMPTY machine's pixels everywhere and
take only the person (+ handle + cable) from the edited frame, so the machine geometry stays pixel-identical to the frame
that was checked against the real photo (the edit model re-renders and shifts the rail/seat by ~20 px).
mask = GrabCut on the edited frame, seeded by |edited - empty| inside the person box; --add boxes take pixels with
|diff| > 8 (dark head / shorts on a dark backdrop); --cut boxes force the empty frame back (e.g. the seat pad under the
person). 1.2 px feather.
usage: python3 tools/still_person_composite.py <empty> <edited> <out> --box x0,y0,x1,y1 [--add x0,y0,x1,y1 ...] [--cut x0,y0,x1,y1 ...]"""
import sys, argparse, cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("empty"); ap.add_argument("edited"); ap.add_argument("out")
ap.add_argument("--box", required=True); ap.add_argument("--add", action="append", default=[]); ap.add_argument("--cut", action="append", default=[])
a = ap.parse_args()
B = lambda s: tuple(map(int, s.split(",")))
e = cv2.imread(a.empty); m = cv2.resize(cv2.imread(a.edited), (e.shape[1], e.shape[0]))
d = np.abs(cv2.GaussianBlur(m, (0, 0), 2).astype(int) - cv2.GaussianBlur(e, (0, 0), 2).astype(int)).max(2)
x0, y0, x1, y1 = B(a.box)
M = np.full(d.shape, cv2.GC_BGD, np.uint8); M[y0:y1, x0:x1] = cv2.GC_PR_BGD
sub = M[y0:y1, x0:x1]; sub[d[y0:y1, x0:x1] > 20] = cv2.GC_PR_FGD; sub[d[y0:y1, x0:x1] > 45] = cv2.GC_FGD
bgd = np.zeros((1, 65)); fgd = np.zeros((1, 65))
cv2.grabCut(m, M, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
fg = ((M == cv2.GC_FGD) | (M == cv2.GC_PR_FGD)).astype(np.uint8)
for s in a.add:
    p0, q0, p1, q1 = B(s); fg[q0:q1, p0:p1] |= (d[q0:q1, p0:p1] > 8).astype(np.uint8)
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
for s in a.cut:
    p0, q0, p1, q1 = B(s); fg[q0:q1, p0:p1] = 0
n, lab, st, _ = cv2.connectedComponentsWithStats(fg); keep = np.zeros_like(fg)
for i in range(1, n):
    if st[i, cv2.CC_STAT_AREA] > 300: keep[lab == i] = 1
al = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.2)[:, :, None]
o = e.astype(np.float32) * (1 - al) + m.astype(np.float32) * al
cv2.imwrite(a.out, np.clip(o, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(a.out.rsplit(".", 1)[0] + "_mask.png", keep * 255)
print("mask frac", round(float(keep.mean()), 4))
