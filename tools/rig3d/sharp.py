"""Shared sharpness metric (rig3d QC): Laplacian variance of the FIGURE's own neutral-grey surface only.
Excludes coloured pixels (glow, laser, callouts, cobalt dumbbell; dilated) so crisp UI lines / equipment edges never
count as body detail. Used for both the Day 4 reference frames and our graded frames, so the ratio is like-for-like."""
import cv2, numpy as np

def neutral_lapvar(rgb, body, exclude_rows=None):
    f = rgb.astype(np.float32); sat = f.max(-1) - f.min(-1)
    coloured = cv2.dilate((sat > 25).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    m = body & (sat < 14) & ~coloured
    if exclude_rows:
        for a, b in exclude_rows: m[a:b] = False
    m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if m.sum() < 2000: return 0.0, int(m.sum())
    lap = cv2.Laplacian(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    return float(lap[m].var()), int(m.sum())
