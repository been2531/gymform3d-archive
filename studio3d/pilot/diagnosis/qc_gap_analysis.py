#!/usr/bin/env python3
"""Demonstrate why pilot QC passes broken v3 frames; compute metrics QC should use."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
KF = ROOT / "output/pilot/lat/keyframes"
OUT = ROOT / "output/pilot/diagnosis/qc_gap_metrics.json"


def lum(img):
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


def glow_color_area(rgb, hue="green"):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if hue == "green":
        mask = ((h > 35) & (h < 95) & (s > 80) & (v > 60))
    else:
        mask = (((h < 15) | (h > 165)) & (s > 80) & (v > 60))
    return int(mask.sum()), mask


def high_freq_energy(gray, roi):
    """Proxy for fibre detail vs smooth skin."""
    if roi.sum() < 100:
        return 0.0
    crop = gray.copy()
    crop[~roi] = 0
    lap = cv2.Laplacian(crop, cv2.CV_32F)
    return float(np.var(lap[roi]))


def bar_horizontal_score(rgb):
    """Detect near-horizontal high-luminance line (bar) in upper 70% of frame."""
    g = lum(rgb)
    upper = g[: int(g.shape[0] * 0.72), :]
    edges = cv2.Canny(upper.astype(np.uint8), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=120, maxLineGap=20)
    horiz = 0
    if lines is not None:
        for ln in lines.reshape(-1, 4):
            x1, y1, x2, y2 = ln
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang < 25 or ang > 155:
                horiz += 1
    return horiz


def main():
    frames = [
        "right_contract_side",
        "right_contract_front",
        "wrong_contract_side",
    ]
    rows = []
    for name in frames:
        p = KF / f"{name}_beauty.png"
        if not p.exists():
            continue
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        g = lum(rgb)
        skin = g > 45
        green_px, gmask = glow_color_area(rgb, "green")
        red_px, _ = glow_color_area(rgb, "red")
        hf = high_freq_energy(g, skin)
        bar_h = bar_horizontal_score(rgb)
        rows.append(
            {
                "frame": name,
                "beauty_glow_green_px": green_px,
                "beauty_glow_red_px": red_px,
                "skin_laplacian_var": hf,
                "horizontal_bar_lines_upper": bar_h,
                "would_fail_glow_min_8000_green": green_px < 8000,
                "would_fail_fibre_var_min_200": hf < 200,
                "would_flag_horizontal_bar_side": name.endswith("side") and bar_h > 0,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
