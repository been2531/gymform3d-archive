#!/usr/bin/env python3
"""t669u pen-line style mock (opt-in, 0 generations): ONE 1080x1920 frame of the D7 Seated Cable Row WRONG step drawn by
code (gymform/pen_style.py) + a 540x960 side-by-side against the current D7 Day 4-grade mock frame.

Sources (all existing, no generation): the empty machine plate, the pinned wrong / right stills, their relit versions
and relight masks, the still compositor paste masks. Framing = the D7 mock framing (whole-frame mirror, 720 -> 1080).
Regions = gymform/region_glow.py anatomy polygons on MediaPipe keypoints of the same frame. Template = gymform/layout
(Kaushan title, 2-slot preview, measured angle + one label, hide_lines from the D7 config).
usage: nice -n 19 python3 tools/pen_style_mock.py [out_dir]"""
import sys, os, json, time, subprocess
import cv2, numpy as np
ROOT = "/workspace/daily_info_reels"
sys.path.insert(0, ROOT)
t0 = time.time()
from gymform import pen_style as ps
from gymform.mannequin_relight import detect_pose
D = f"{ROOT}/output/_consistency/d7m/"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/output/_consistency/pen_style"
os.makedirs(OUT, exist_ok=True)
W, H = 1080, 1920


def prep(p, interp=cv2.INTER_CUBIC):
    im = cv2.imread(p, -1)
    return cv2.resize(cv2.flip(im, 1), (W, H), interpolation=interp)   # D7 concept: whole-frame mirror, tower right


def kp_dict(P):
    """MediaPipe 33 -> region_glow / layout names; 'r' = the screen-left shoulder chain (the extractor's sort)."""
    A, B = (11, 13, 15, 23, 25), (12, 14, 16, 24, 26)
    r, l = (A, B) if P[11][0] <= P[12][0] else (B, A)
    k = {}
    for sd, ids in (("r", r), ("l", l)):
        for nm, i in zip(("sh", "elb", "wr", "hip", "kn"), ids):
            k[f"{nm}_{sd}"] = (float(P[i][0]), float(P[i][1]))
    k["sh"] = tuple(0.5 * (np.array(k["sh_r"]) + np.array(k["sh_l"])))
    return k


def visible_arm(kp, body):
    """the arm whose shoulder->wrist line lies mostly OUTSIDE the torso quad (the far arm hides behind the back)."""
    tq = np.zeros((H, W), np.uint8)
    cv2.fillPoly(tq, [np.array([kp[k] for k in ("sh_r", "sh_l", "hip_l", "hip_r")], np.int32)], 1)
    best, sd_best = 2.0, "r"
    for sd in ("r", "l"):
        m = np.zeros((H, W), np.uint8)
        cv2.line(m, tuple(int(v) for v in kp[f"elb_{sd}"]), tuple(int(v) for v in kp[f"wr_{sd}"]), 1, 3)
        f = (m & tq).sum() / max(m.sum(), 1)
        if f < best:
            best, sd_best = f, sd
    return sd_best


def elbow_flexion(kp, sd):
    a, b, c = (np.array(kp[f"{j}_{sd}"]) for j in ("sh", "elb", "wr"))
    v1, v2 = a - b, c - b
    ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
    return 180.0 - ang


def pen_frame(form, pin, relit, relit_mask, paste, targets):
    E = prep(D + "pin_empty_machine_720x1280.png")
    O = prep(D + pin)                       # un-relit pinned still: crispest anatomy creases (line source)
    R = prep(D + relit)                     # relit still: body tone + person mask
    lab = prep(D + relit_mask, cv2.INTER_NEAREST)
    pa = prep(D + paste, cv2.INTER_NEAREST)
    pa = pa if pa.ndim == 2 else pa[..., 0]
    mach, body, fg, bg = ps.masks(E, R, pa, lab)
    canvas, body_w, lay = ps.line_art(E, R, mach, body, fg, bg, O)
    kp = kp_dict(detect_pose(O))
    out, wgt, ms = ps.muscle_fill(canvas, O, body_w, kp, targets, form, ink=lay["ink_crease"])
    return np.clip(out, 0, 255).astype(np.uint8), body_w, wgt, ms, kp


from gymform.region_glow import DEFAULT_TARGETS
cfg = json.load(open(f"{ROOT}/configs/day07_muscle2.json"))
t_gen = time.time()
wr_rgb, wr_body, wr_w, wr_ms, wr_kp = pen_frame("wrong", "pin_wrong_head_t1_720x1280.png", "relight/pin_wrong_head_t1_relit.png",
                                                 "relight/pin_wrong_head_t1_relit_mask.png", "relight/wrong_head_t1_mask720.png",
                                                 (cfg.get("target_regions") or {}).get("wrong") or DEFAULT_TARGETS["wrong"])
t_w = time.time() - t_gen
rt_rgb, rt_body, rt_w, rt_ms, rt_kp = pen_frame("right", "pin_right_fibre_head_t2_720x1280.png", "relight/pin_right_fibre_head_t2_relit.png",
                                                 "relight/pin_right_fibre_head_t2_relit_mask.png", "relight/right_fibre_head_t2_mask720.png",
                                                 (cfg.get("target_regions") or {}).get("right") or DEFAULT_TARGETS["right"])
cv2.imwrite(f"{OUT}/pen_wrong_base_1080.png", cv2.cvtColor(wr_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{OUT}/pen_right_base_1080.png", cv2.cvtColor(rt_rgb, cv2.COLOR_RGB2BGR))
# --- Day 4 template (unchanged production layout code) ---
from gymform import ctx, layout
import build_4step_bone_muscle_master as b4
ctx.configure(cfg)
icons = [b4._make_icon_pair(wr_rgb, wr_kp), b4._make_icon_pair(rt_rgb, rt_kp)]
sd = visible_arm(wr_kp, wr_body)
deg = f"{int(round(elbow_flexion(wr_kp, sd)))}\u00b0"
l1 = cfg["motions"]["wrong"]["callout"]["l1"]
fr = layout.draw_callout_day4(wr_rgb, np.zeros((H, W), np.float32), 1, wr_kp, True, deg, l1, "")
ctx.LAST_MASKS["eq_front"] = np.zeros((H, W), np.float32)
fr = layout.composite_title_and_preview_v4(fr, layout.fg_for_preview(wr_body), icons, 0, True, cfg["exercise_name"])
mock = f"{OUT}/pen_mock_d7_wrong_1080.jpg"
cv2.imwrite(mock, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
t_frame = time.time() - t0
# --- region QC: colour only inside the allowed regions ---
allowed = np.zeros((H, W), np.float32)
for n_ in ((cfg.get("target_regions") or {}).get("wrong") or DEFAULT_TARGETS["wrong"]):
    allowed = np.maximum(allowed, wr_ms[n_])
allowed_d = cv2.dilate(allowed, np.ones((41, 41), np.uint8))   # feather + falloff margin
f32 = fr.astype(np.float32)
dom = f32[..., 0] - np.maximum(f32[..., 1], f32[..., 2])
ui = np.zeros((H, W), bool); ui[:600] = True               # title / preview / callout band
spill = int(((dom > 40) & (allowed_d == 0) & ~ui).sum())
# --- Day 4 look metrics at equal on-screen scale (torso px from this frame's keypoints) ---
torso = float(np.linalg.norm(np.array(wr_kp["sh"]) - 0.5 * (np.array(wr_kp["hip_r"]) + np.array(wr_kp["hip_l"]))))
ref = f"{D}day4_master_t7.5_wrong.jpg"
r = subprocess.run([sys.executable, f"{ROOT}/tools/day4_look_metrics.py", f"d4w={ref}:wrong:356.4", f"pen_w={OUT}/pen_wrong_base_1080.png:wrong:{torso:.1f}"],
                   capture_output=True, text=True)
try:
    met = json.loads(r.stdout[r.stdout.index("{\n"):])
except Exception:
    met = {"error": r.stderr[-800:]}
info = {"mock": mock, "deg": deg, "visible_arm": sd, "label": l1, "spill_px_outside_regions": spill,
        "torso_px": round(torso, 1), "time_s": {"wrong_pen_frame": round(t_w, 2), "whole_mock_incl_imports": round(t_frame, 2)},
        "generations": 0, "metrics_vs_day4": met}
json.dump(info, open(f"{OUT}/pen_mock_d7_wrong.json", "w"), indent=1)
# --- side by side: current D7 Day 4-grade frame vs pen mock ---
cur = cv2.imread(f"{D}mock_final/mock_day4_wrong_1080.jpg")
pen = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
tiles = []
for im, lab_ in ((cur, "CURRENT D7 (3D clip + Day 4 grade)"), (pen, "PEN STYLE (code only)")):
    t = cv2.resize(im, (540, 960), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 918), (540, 960), (20, 22, 26), -1)
    cv2.putText(t, lab_, (14, 946), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    tiles.append(t)
sbs = np.concatenate([tiles[0], np.full((960, 8, 3), 12, np.uint8), tiles[1]], 1)
cv2.imwrite(f"{OUT}/pen_vs_current_d7_sbs.jpg", sbs, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(json.dumps(info, indent=1))
