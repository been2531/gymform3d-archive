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
STEP_DX = {}      # per-step horizontal offset of the clip (target zoom framing)
TMP = {"dir": None}

__all__ = ["W", "H", "FPS", "DURATION", "_XX", "_YY", "CFG", "LAST_MASKS", "CUR_PLATE", "STEP_SCALE", "STEP_DX",
           "d5", "b4", "p62", "v30", "SPEC", "ROOT"]


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


PAIRS = {1: 3, 2: 4}          # same MOTION, different rendering: 1/3 = wrong form, 2/4 = right form (Day 6+)
_KIND_STEP = {("wrong", "bone"): 1, ("right", "bone"): 2, ("wrong", "muscle"): 3, ("right", "muscle"): 4}


def expand_motions(cfg: dict) -> dict:
    """Day 6+ schema: `motions` = {"wrong": M, "right": M}; M = {"base": file, "i0", "i1", "y_shift",
    "renders": {"bone": {"file", "edit_of"?}, "muscle": {"file", "edit_of"?}}}. The 4 steps are DERIVED:
    1 = wrong/bone, 2 = right/bone, 3 = wrong/muscle, 4 = right/muscle. Both renders of a motion share its
    slice, y_shift and keypoints (kp_file = the motion base), so pair steps have the identical motion.
    The anatomy layer is NATIVE in each file (bone X-Ray or muscle fibre baked by the 3D source), so a motion
    needs one native file per rendering: the base clip and its motion-preserving edit (edit_of = base)."""
    if "motions" not in cfg:
        return cfg
    clips = {}
    for (form, kind), s in _KIND_STEP.items():
        m = cfg["motions"][form]
        r = m["renders"][kind]
        if r.get("pair_match"):   # t541u: a separate NATIVE clip matched to the base within joint tolerance:
            # its own DTW-aligned slice and its own keypoints (same motion phase at rest/peak, not the same frames)
            clips[str(s)] = {"file": r["file"], "i0": int(r["i0"]), "i1": int(r["i1"]), "y_shift": int(m.get("y_shift", 80)),
                             "is_wrong": form == "wrong", "mode": f"{kind}_blood", "motion": form, "kp_file": r["file"],
                             "edit_of": None, "mirror": bool(r.get("mirror", m.get("mirror", False))), "pair_match": r["pair_match"],
                             "native_tint": r.get("native_tint")}
            continue
        clips[str(s)] = {"file": r["file"], "i0": int(m["i0"]), "i1": int(m["i1"]), "y_shift": int(m.get("y_shift", 80)),
                         "is_wrong": form == "wrong", "mode": f"{kind}_blood", "motion": form, "kp_file": m["base"],
                         "edit_of": r.get("edit_of"), "mirror": bool(r.get("mirror", m.get("mirror", False)))}
    cfg["clips"] = clips
    cfg["bone_steps"], cfg["muscle_steps"] = [1, 2], [3, 4]
    return cfg


def pair_lint(cfg, spec=None):
    """Day >= spec.step_order.pair_from_day: pair steps (1/3 wrong, 2/4 right) must be the SAME motion:
    defined via `motions`, each rendering either the motion's base clip or a declared motion-preserving edit
    of it (edit_of == base). Any other clip pairing is rejected."""
    so = (spec or SPEC).get("step_order", {})
    if int(cfg.get("day", 0)) < int(so.get("pair_from_day", 6)):
        return []
    if "motions" not in cfg:
        return ["Day 6+ config must define `motions` {wrong, right} (pair steps derived from one clip each); "
                "per-step `clips` with different motions are rejected"]
    errs = []
    for form in ("wrong", "right"):
        m = cfg["motions"].get(form)
        if not m:
            errs.append(f"motions.{form} missing")
            continue
        for kind in ("bone", "muscle"):
            r = (m.get("renders") or {}).get(kind)
            if not r:
                errs.append(f"motions.{form}.renders.{kind} missing")
            elif r.get("pair_match"):   # t541u relaxed rule: same motion + angle within joint tolerance (measured)
                pm, tol = r["pair_match"], so.get("pair_tolerance", {"joint_dev_pct": 15.0, "angle_deg": 15.0, "view_dev_pct": 15.0})
                for k_, lim in (("joint_dev_pct", tol["joint_dev_pct"]), ("elbow_deg", tol["angle_deg"]),
                                ("shoulder_deg", tol["angle_deg"]), ("view_dev_pct", tol["view_dev_pct"])):
                    if float(pm.get(k_, 1e9)) > float(lim):
                        errs.append(f"motions.{form}.{kind} {r['file']}: {k_} {pm.get(k_)} > tolerance {lim}")
            elif r["file"] != m["base"] and r.get("edit_of") != m["base"]:
                errs.append(f"motions.{form}.{kind} file {r['file']} is not the motion base {m['base']} nor a declared edit of it")
    return errs


def configure(day_cfg: dict, overrides: dict | None = None):
    """Locked spec defaults + day config (+ autofix overrides) -> CFG."""
    c = {
        "studio_bg": SPEC["background"]["plate"],
        "xray_style": "day1_v36",
        "mask_feather": 0.85,
        "equipment_hull": False,
        "bone_steps": [1, 2],
        "muscle_steps": [3, 4],
    }
    c.update(expand_motions(json.loads(json.dumps(day_cfg))))
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
    STEP_DX.clear()
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
    """Per-step scale + framing from the cached keypoints (all days, standing or seated): every step's on-screen
    skeletal size = median configured size x target_zoom.zoom (identical in all 4 steps).
    Without target_zoom: feet on one floor line. With target_zoom: the focus region (e.g. upper chest/shoulders)
    is placed at the same screen point in every step, and the zoom is reduced automatically if the dumbbells
    (wrists + plate margin) would leave the frame. Deterministic -> identical in all workers."""
    if not CFG.get("auto_scale", True):
        return
    sizes, ank = {}, {}
    for s in (1, 2, 3, 4):
        y_sh = step_spec(s)[5]
        sizes[s] = float(np.median([skeletal_size(k) for k in kps_seqs[s]]))
        ank[s] = float(np.median([max(k["ank_r"][1], k["ank_l"][1]) - y_sh for k in kps_seqs[s]]))
    cur = {s: sizes[s] * float(CFG["step_scale"][str(s)]) for s in sizes}
    base = float(np.median(list(cur.values())))
    tz = CFG.get("target_zoom") or {}
    zoom = float(tz.get("zoom", 1.0))
    note = ""
    if not tz:
        floor = float(np.median([(ank[s] - H * 0.5) * float(CFG["step_scale"][str(s)]) + H * 0.5 + int(CFG["step_dy"][str(s)]) for s in sizes]))
        for s in sizes:
            sc = base / max(sizes[s], 1.0)
            STEP_SCALE[s] = round(sc, 4)
            b4.STEP_DY[s] = int(round(floor - ((ank[s] - H * 0.5) * sc + H * 0.5)))
            STEP_DX[s] = 0
    else:
        fx_t, fy_t = float(tz.get("focus_x", W * 0.5)), float(tz.get("focus_y", 1000))
        pad = float(tz.get("equipment_margin", 0.11))     # dumbbell half-size beyond the wrist, x skeletal size
        top_min, side_min = float(tz.get("top_min_y", 140)), float(tz.get("side_min", 24))

        def frame(zm):
            out, worst = {}, 1e9
            for s in sizes:
                y_sh = step_spec(s)[5]
                sc = base * zm / max(sizes[s], 1.0)
                fpts = [((k["sh_r"][0] + k["sh_l"][0]) * 0.5, (k["sh_r"][1] + k["sh_l"][1]) * 0.5 - y_sh) for k in kps_seqs[s]]
                fx, fy = float(np.median([p[0] for p in fpts])), float(np.median([p[1] for p in fpts]))
                dx = fx_t - ((fx - W * 0.5) * sc + W * 0.5)
                dy = fy_t - ((fy - H * 0.5) * sc + H * 0.5)
                m = pad * base * zm
                for k in kps_seqs[s]:
                    for w_ in ("wr_r", "wr_l"):
                        x = (k[w_][0] - W * 0.5) * sc + W * 0.5 + dx
                        y = (k[w_][1] - y_sh - H * 0.5) * sc + H * 0.5 + dy
                        worst = min(worst, y - m - top_min, x - m - side_min, (W - side_min) - (x + m))
                out[s] = (round(sc, 4), int(round(dx)), int(round(dy)))
            return out, worst
        zm = zoom
        out, worst = frame(zm)
        while worst < 0 and zm > 1.0:          # never cut the dumbbells: back the zoom off in small steps
            zm = round(zm - 0.02, 3)
            out, worst = frame(zm)
        if zm < zoom:
            note = f"zoom reduced {zoom:.2f}->{zm:.2f} to keep dumbbells in frame"
        for s, (sc, dx, dy) in out.items():
            STEP_SCALE[s], STEP_DX[s], b4.STEP_DY[s] = sc, dx, dy
        zoom = zm
    CFG["_normalized"] = {"scale": dict(STEP_SCALE), "dy": dict(b4.STEP_DY), "dx": dict(STEP_DX), "target_size": round(base * zoom, 1),
                          "zoom": zoom, "upscale_vs_720p": round(1.5 * float(np.median(list(STEP_SCALE.values()))), 3), "note": note}


# ---------------------------------------------------------------------------------------------------------------
# LOCKED STEP ORDER (from Day 6): bone vs bone, then muscle vs muscle.
#   1 = WRONG bone X-Ray (crimson) | 2 = RIGHT bone X-Ray (emerald) | 3 = WRONG muscle (red) | 4 = RIGHT muscle (green)
# Every step-specific branch in the modules goes through role()/legacy(), never a literal step number.
LOCKED_ORDER = {1: ("bone", True), 2: ("bone", False), 3: ("muscle", True), 4: ("muscle", False)}
_LEGACY = {("bone", True): 1, ("muscle", True): 2, ("bone", False): 3, ("muscle", False): 4}   # Day 1-5 index convention


def role(step):
    """(kind, is_wrong) of a step from the day config (bone_steps / clips.is_wrong)."""
    kind = "bone" if int(step) in CFG["bone_steps"] else "muscle"
    return kind, bool(CFG["clips"][str(int(step))]["is_wrong"])


def legacy(step):
    """Role-equivalent index in the Day 1-5 convention (1 bone-wrong, 2 muscle-wrong, 3 bone-right, 4 muscle-right),
    used by the shared look code so a step's look follows its ROLE, not its position."""
    return _LEGACY[role(step)]


def order_lint(cfg, spec=None):
    """Reject any step order other than the locked one (Day >= spec.step_order.from_day)."""
    so = (spec or SPEC).get("step_order", {})
    if int(cfg.get("day", 0)) < int(so.get("from_day", 6)):
        return []
    errs = []
    if list(cfg.get("bone_steps", [1, 2])) != [1, 2] or list(cfg.get("muscle_steps", [3, 4])) != [3, 4]:
        errs.append(f"bone_steps/muscle_steps must be [1,2]/[3,4], got {cfg.get('bone_steps')}/{cfg.get('muscle_steps')}")
    for s, (kind, wrong) in LOCKED_ORDER.items():
        c = cfg["clips"][str(s)]
        if bool(c["is_wrong"]) != wrong:
            errs.append(f"step{s} must be {'WRONG' if wrong else 'RIGHT'} form")
        if kind not in c.get("mode", kind):
            errs.append(f"step{s} clip mode {c.get('mode')!r} must be a {kind} clip")
        co = cfg.get("callouts", {}).get(str(s), {})
        if co and int(co.get("slot", s - 1)) != s - 1:
            errs.append(f"step{s} preview slot must be {s - 1}")
    xf = cfg.get("xray_full_steps")
    if xf is not None and not set(xf) <= {1, 2}:
        errs.append(f"xray_full_steps {xf} must be bone steps 1/2")
    return errs


def clip_rect(step):
    """(x0, y0, nw, nh) of the step's scaled 720p clip on the 1080x1920 canvas."""
    sc = STEP_SCALE.get(step, b4.SCALE)
    nw, nh = int(round(W * sc)), int(round(H * sc))
    return (W - nw) // 2 + int(STEP_DX.get(step, 0)), (H - nh) // 2 + int(b4.STEP_DY.get(step, 0)), nw, nh
