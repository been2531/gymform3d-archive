"""Shared state for the gymform modules (locked spec + the active day config)."""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import build_day5_lateral_raise_master as d5      # legacy loaders, kp extraction, callout laser chains, TTS
import build_4step_bone_muscle_master as b4       # 1080 transform, title/preview compositor, icons
import preview_v62_seamless_day2_master as p62    # laser primitives
import build_v30_pure_graphic_numbers as v30      # fonts

cv2.setNumThreads(2)
SPEC = json.load(open(os.path.join(ROOT, "gymform_spec.json")))
W, H = SPEC["format"]["width"], SPEC["format"]["height"]
FPS, DURATION = SPEC["format"]["fps"], SPEC["format"]["duration_s"]
_YY, _XX = np.mgrid[:H, :W].astype(np.float32)
CACHE_DIR = os.path.join(ROOT, "_cache_gymform")
os.makedirs(CACHE_DIR, exist_ok=True)

CFG = {}          # active day config (mutated in place by configure)
LAST_MASKS = {}   # per-frame masks for compositing + QC sidecars
CUR_PLATE = {}    # temporal backdrop plate of the current step
STEP_SCALE = {}
TMP = {"dir": None}

__all__ = ["W", "H", "FPS", "DURATION", "_XX", "_YY", "CFG", "LAST_MASKS", "CUR_PLATE", "STEP_SCALE",
           "d5", "b4", "p62", "v30", "SPEC", "ROOT"]


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def configure(day_cfg: dict, overrides: dict | None = None):
    """Locked spec defaults + day config (+ autofix overrides) -> CFG."""
    c = {
        "studio_bg": SPEC["background"]["plate"],
        "xray_style": "day1_v36",
        "mask_feather": 0.85,
        "equipment_hull": False,
        "bone_steps": [1, 3],
        "muscle_steps": [2, 4],
    }
    c.update(json.loads(json.dumps(day_cfg)))
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    for k in ("studio_bg", "kp_cache", "assets_dir", "out_mp4", "out_sheet", "out_cover", "qc_dir", "tmp_dir"):
        if k in c:
            c[k] = _abs(c[k])
    c["bone_steps"] = tuple(c["bone_steps"])
    c["muscle_steps"] = tuple(c["muscle_steps"])
    CFG.clear()
    CFG.update(c)
    STEP_SCALE.clear()
    STEP_SCALE.update({int(k): float(v) for k, v in c["step_scale"].items()})
    b4.STEP_DY = dict(b4.STEP_DY)
    b4.STEP_DY.update({int(k): int(v) for k, v in c["step_dy"].items()})
    TMP["dir"] = c.get("tmp_dir", os.path.join(ROOT, f"_tmp_gymform_day{c['day']:02d}"))
    os.makedirs(TMP["dir"], exist_ok=True)
    d5.TMP_DIR = TMP["dir"]
    return CFG


def step_spec(s):
    """(clip file, i0, i1, is_wrong, mode, y_shift)."""
    c = CFG["clips"][str(s)]
    return c["file"], int(c["i0"]), int(c["i1"]), bool(c["is_wrong"]), c.get("mode", ""), int(c.get("y_shift", 80))
