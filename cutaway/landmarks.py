"""Automatic anatomical landmarks used by registration and the landmark gate.

knee (lateral view, front of the knee facing +x or -x):
  skin_bump   : apex of the anterior knee contour on the skin silhouette (most anterior
                point within the middle band of the image) -> where the kneecap shows.
  patella     : centroid of the most anterior compact ivory-bone blob in the bone layer
                (beige pixels: R > B + 22, bright; cartilage is blue-white and the ghost
                outline is blue-grey, both excluded).
  patella_tgt : skin_bump moved inward by (patella half-width + soft tissue 2 % of width);
                the gate compares patella vs patella_tgt.
"""
from __future__ import annotations

import cv2
import numpy as np

import register as R


def skin_bump(skin_rgb, front="+x", band=(0.25, 0.75)):
    m = R.fg_mask(skin_rgb)
    h, w = m.shape
    best = None
    for y in range(int(h * band[0]), int(h * band[1])):
        xs = np.where(m[y] > 0)[0]
        if len(xs) == 0:
            continue
        x = xs.max() if front == "+x" else xs.min()
        score = x if front == "+x" else -x
        if best is None or score > best[0]:
            best = (score, x, y)
    # average the rows within 4 px of the apex for a stable y
    _, xa, ya = best
    ys = []
    for y in range(int(h * band[0]), int(h * band[1])):
        xs = np.where(m[y] > 0)[0]
        if len(xs) and abs((xs.max() if front == "+x" else xs.min()) - xa) <= 4:
            ys.append(y)
    return float(xa), float(np.median(ys))


def bone_mask(rgb):
    f = rgb.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    m = ((r - b) > 22) & (r > 95) & (g > 75) & ((r - g) < 60)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def patella(bone_rgb, front="+x", min_area_frac=0.002, max_area_frac=0.05):
    m = bone_mask(bone_rgb)
    h, w = m.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    cands = []
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        if not (min_area_frac * h * w <= a <= max_area_frac * h * w):
            continue
        bw, bh = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        if max(bw, bh) / max(1, min(bw, bh)) > 2.6:      # compact blob, not a shaft
            continue
        cx, cy = cen[k]
        cands.append((cx if front == "+x" else -cx, float(cx), float(cy), int(bw), int(a)))
    if not cands:
        return None
    _, cx, cy, bw, a = max(cands)
    return {"xy": [cx, cy], "width": bw, "area": a}


def front_apex(mask, front="+x", band=(0.25, 0.75)):
    h, w = mask.shape
    pts = []
    for y in range(int(h * band[0]), int(h * band[1])):
        xs = np.where(mask[y] > 0)[0]
        if len(xs):
            pts.append((xs.max() if front == "+x" else -xs.min(), y))
    if not pts:
        return None
    best = max(p[0] for p in pts)
    ys = [y for x, y in pts if x >= best - 4]
    return float(abs(best)), float(np.median(ys))


def knee_report(skin_rgb, bone_rgb, front="+x"):
    """Patella landmark = anterior apex of the bone mask (the kneecap is the most
    anterior bone of a lateral knee; generated images often let it touch the femur,
    so a centroid is not separable).  Target = skin bump apex moved inward by a soft
    tissue allowance of 2 % of the canvas width.  Apex height ~= patella centre height."""
    sx, sy = skin_bump(skin_rgb, front)
    h, w = skin_rgb.shape[:2]
    ap = front_apex(bone_mask(bone_rgb), front)
    if ap is None:
        return {"skin_bump": [sx, sy], "patella_apex": None, "error_canvas_px": 9999.0}
    t = 0.02 * w
    tx = sx - t if front == "+x" else sx + t
    err = float(np.hypot(ap[0] - tx, ap[1] - sy))
    return {"skin_bump": [sx, sy], "patella_target": [tx, sy], "patella_apex": [ap[0], ap[1]],
            "dx": round(ap[0] - tx, 1), "dy": round(ap[1] - sy, 1), "error_canvas_px": round(err, 1)}
