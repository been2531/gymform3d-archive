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


def skeletal_size(kp):
    """Pose-invariant mannequin size: torso + thigh + shin + upper arm (keypoint segment lengths). The head-top /
    vertical-extent metric is fooled by overhead arms and seated poses (Day 6: Step 3 rendered 20% larger while
    the extent metric read equal); bone lengths are the same in every pose."""
    import math
    D = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    M = lambda a, b: (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    tor = D(M(kp["sh_r"], kp["sh_l"]), M(kp["hip_r"], kp["hip_l"]))
    th = 0.5 * (D(kp["hip_r"], kp["kn_r"]) + D(kp["hip_l"], kp["kn_l"]))
    shin = 0.5 * (D(kp["kn_r"], kp["ank_r"]) + D(kp["kn_l"], kp["ank_l"]))
    ua = 0.5 * (D(kp["sh_r"], kp["elb_r"]) + D(kp["sh_l"], kp["elb_l"]))
    return tor + th + shin + ua


def normalize_scale(kps_seqs):
    """Per-step scale + floor normalisation from the cached keypoints (all days, standing or seated): every step's
    on-screen skeletal size = the median of the configured sizes, and every step's feet on one floor line.
    Overrides step_scale/step_dy (they become the starting point only). Deterministic -> identical in all workers."""
    if not CFG.get("auto_scale", True):
        return
    sizes, ank = {}, {}
    for s in (1, 2, 3, 4):
        y_sh = step_spec(s)[5]
        sz = [skeletal_size(k) for k in kps_seqs[s]]
        sizes[s] = float(np.median(sz))
        ank[s] = float(np.median([max(k["ank_r"][1], k["ank_l"][1]) - y_sh for k in kps_seqs[s]]))
    cur = {s: sizes[s] * float(CFG["step_scale"][str(s)]) for s in sizes}
    target = float(np.median(list(cur.values())))
    floor = float(np.median([(ank[s] - H * 0.5) * float(CFG["step_scale"][str(s)]) + H * 0.5 + int(CFG["step_dy"][str(s)]) for s in sizes]))
    for s in sizes:
        sc = target / max(sizes[s], 1.0)
        STEP_SCALE[s] = round(sc, 4)
        b4.STEP_DY[s] = int(round(floor - ((ank[s] - H * 0.5) * sc + H * 0.5)))
    CFG["_normalized"] = {"scale": dict(STEP_SCALE), "dy": dict(b4.STEP_DY), "target_size": round(target, 1), "floor_y": round(floor, 1)}
