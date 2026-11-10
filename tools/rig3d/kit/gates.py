"""Backstop gates for the welded-body kit: topology + stills.

These do not substitute for the construction (weld / attribute glow / closed
tube). They only report whether the built meshes and the rendered pixels
still have tears, clay-in-shorts, or holes in the glow.
"""
from __future__ import annotations
import json
import os
import numpy as np

from .weld import topology_stats


def mesh_report(ob):
    if ob is None or getattr(ob, "data", None) is None:
        return None
    st = topology_stats(ob.data)
    st["name"] = ob.name
    return st


def write_topo(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def _load_rgb(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return np.asarray(im)


def clay_in_shorts(beauty_rgb, mask_rgb=None, hips_box=(20, 620, 380, 1140)):
    """Fraction of the shorts silhouette that still shows porcelain clay."""
    import cv2
    h, w = beauty_rgb.shape[:2]
    hsv = cv2.cvtColor(beauty_rgb, cv2.COLOR_RGB2HSV)
    if mask_rgb is not None:
        # briefs mask pass: white emission on the garment
        sil = mask_rgb.min(axis=2) > 200
    else:
        x0, y0, x1, y1 = hips_box
        dark = np.zeros((h, w), dtype=bool)
        crop = hsv[y0:y1, x0:x1]
        dark[y0:y1, x0:x1] = (crop[:, :, 2] < 45) & (crop[:, :, 1] < 70)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        sil = cv2.morphologyEx(dark.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k) > 0
    clay = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 70) & (hsv[:, :, 2] < 190)
    inside = sil & clay
    n_sil = int(sil.sum())
    n_clay = int(inside.sum())
    frac = float(n_clay / max(n_sil, 1))
    return {"clay_px": n_clay, "shorts_px": n_sil, "clay_frac": frac,
            "pass": n_clay == 0}


def glow_holes(beauty_rgb):
    """Holes = pixels inside a morph-closed glow sheet that are not glow."""
    import cv2
    hsv = cv2.cvtColor(beauty_rgb, cv2.COLOR_RGB2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    glow = (s > 70) & (v > 70) & ((hh < 28) | ((hh > 120) & (hh < 178)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(glow.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k) > 0
    holes = closed & (~glow)
    n_sheet = int(closed.sum())
    n_holes = int(holes.sum())
    frac = float(n_holes / max(n_sheet, 1))
    return {"hole_px": n_holes, "sheet_px": n_sheet, "hole_frac": frac,
            "pass": n_holes == 0}


def check_stills(beauty_path, mask_path=None, log=print):
    out = {"beauty": beauty_path, "mask": mask_path}
    if not os.path.isfile(beauty_path):
        out["error"] = "missing beauty"
        return out
    rgb = _load_rgb(beauty_path)
    mask = _load_rgb(mask_path) if mask_path and os.path.isfile(mask_path) else None
    out["clay_in_shorts"] = clay_in_shorts(rgb, mask)
    out["glow_holes"] = glow_holes(rgb)
    log(f"kit gates clay_in_shorts={out['clay_in_shorts']} glow_holes={out['glow_holes']}")
    return out
