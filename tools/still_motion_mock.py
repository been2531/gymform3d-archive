#!/usr/bin/env python3
"""t674u-t676u still-motion format: ONE static 1080x1920 peak frame, Day 4 grade applied ONCE to a still.

Runs the UNCHANGED production pipeline (gymform native_grade + muscle.day4_fibre via region_glow with
muscle_glow_clamp, the Day 4 calibrated knobs of configs/day07_muscle2.json) on a single still turned into a lossless
12-frame static clip (same as tools/day4_template_mock.py), renders only the requested step, then re-composites the
top preview with ONE thumbnail (1-segment reel; b4 1-icon branch). No video generation.
Still prep: optional mild unsharp mask at source resolution (the pipeline does the 720 -> 1080 resize).
usage: nice -n 19 python3 tools/still_motion_mock.py <peak_still_720.png> <out_dir> [--form right|wrong] [--other <still>] [--sharpen 0.6]"""
import sys, os, json, copy, gc, argparse, time
import cv2, numpy as np
ROOT = "/workspace/daily_info_reels"
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("still"); ap.add_argument("out")
ap.add_argument("--form", default="right")
ap.add_argument("--other", default=f"{ROOT}/output/_consistency/d7m/relight/pin_wrong_head_t1_relit.png",
                help="still for the unused step (the 2-step config needs both inputs; it is never shown)")
ap.add_argument("--sharpen", type=float, default=0.6)
ap.add_argument("--glow-detail", type=float, default=1.0,
                help="still prep inside the source's own glow only: detail gain around a sigma-2.5 base (<1 = shallower, "
                     "narrower fibre grooves; Day 4 shows fine dark striations on a solid core, the generator paints wide "
                     "black gaps). 1.0 = off")
ap.add_argument("--arm-limit", type=float, default=None,
                help="right form: cap the allowed glow on the VISIBLE upper arm (shoulder->elbow, past the rear delt) at this "
                     "weight. In a rear-3/4 view the near arm's triceps overlaps the lats / rear-delt polygons in 2D, so the "
                     "clamp alone lets it glow full; biceps are meant to be faint. Tool-local wrapper, region_glow unchanged")
ap.add_argument("--name", default="peak", help="output tag (start / peak)")
ap.add_argument("--callout", default="joint", choices=["joint", "corner"], help="joint = t678u angle at the elbow (default); corner = legacy top-left block")
ap.add_argument("--config", default=f"{ROOT}/configs/day07_muscle2.json")
ap.add_argument("--cfg-json", default="{}", help="calibration overrides, e.g. {\"glow_gain_override\": {\"2\": 12}}")
a = ap.parse_args()
T0 = time.time()
os.makedirs(a.out, exist_ok=True)
assets = os.path.join(a.out, f"_static_assets_{a.name}"); os.makedirs(assets, exist_ok=True)


def prep(p):
    im = cv2.resize(cv2.imread(p), (720, 1280), interpolation=cv2.INTER_LANCZOS4)
    if a.sharpen > 0:   # mild unsharp mask (sigma 1.0) - crisp edges without halos
        bl = cv2.GaussianBlur(im, (0, 0), 1.0)
        im = cv2.addWeighted(im, 1.0 + a.sharpen, bl, -a.sharpen, 0)
    if a.glow_detail != 1.0:
        f = im.astype(np.float32)
        dom = np.maximum(f[..., 1] - np.maximum(f[..., 0], f[..., 2]), f[..., 2] - np.maximum(f[..., 0], f[..., 1]))
        g = cv2.GaussianBlur(cv2.dilate((dom > 18).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3.0)
        base_ = cv2.GaussianBlur(f, (0, 0), 2.5)
        comp = base_ + a.glow_detail * (f - base_)
        im = np.clip(f * (1 - g[..., None]) + comp * g[..., None], 0, 255).astype(np.uint8)
    return im


srcs = {a.form: prep(a.still), ("wrong" if a.form == "right" else "right"): prep(a.other)}
for form, im in srcs.items():
    vw = cv2.VideoWriter(os.path.join(assets, f"{form}_static.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 24.0, (720, 1280))
    for _ in range(12):
        vw.write(im)
    vw.release()
from gymform import ctx, render, native, layout
import build_4step_bone_muscle_master as b4
cfg = copy.deepcopy(json.load(open(a.config)))
cfg["assets_dir"] = assets
cfg["kp_cache"] = os.path.join(a.out, f"_kps_still_{a.name}.pkl")
cfg["tmp_dir"] = os.path.join(ROOT, "_tmp_still_motion")
for form in ("wrong", "right"):
    m = cfg["motions"][form]; fn = f"{form}_static.avi"
    m["base"] = fn; m["i0"] = 0; m["i1"] = 11; m["renders"]["front"]["file"] = fn
for k_, v_ in json.loads(a.cfg_json).items():
    cfg[k_] = dict(cfg.get(k_) or {}, **v_) if isinstance(v_, dict) and isinstance(cfg.get(k_), dict) else v_
ctx.configure(cfg)
if a.arm_limit is not None:
    from gymform import region_glow as _rg
    _orig_cm = _rg.clamp_mask

    def _clamp_arm(kp, is_wrong, shape):
        m, ms = _orig_cm(kp, is_wrong, shape)
        if is_wrong:
            return m, ms
        tq = np.zeros(shape, np.uint8)
        cv2.fillPoly(tq, [np.array([kp[k] for k in ("sh_r", "sh_l", "hip_l", "hip_r")], np.int32)], 1)
        cap = np.zeros(shape, np.float32)
        for sd in ("r", "l"):
            sh, el = np.array(kp[f"sh_{sd}"], np.float32), np.array(kp[f"elb_{sd}"], np.float32)
            ln = np.zeros(shape, np.uint8); cv2.line(ln, tuple(int(v) for v in sh), tuple(int(v) for v in el), 1, 3)
            if (ln & tq).sum() > 0.5 * max(ln.sum(), 1):
                continue                                   # far arm hidden behind the back
            q0 = sh + 0.38 * (el - sh); q1 = el + 0.08 * (el - sh)
            c = np.zeros(shape, np.uint8)
            cv2.line(c, tuple(int(v) for v in q0), tuple(int(v) for v in q1), 1, max(4, int(0.55 * np.linalg.norm(el - sh))))
            cap = np.maximum(cap, c.astype(np.float32))
        cap = cv2.GaussianBlur(cap, (0, 0), 6.0)
        return m * (1.0 - cap * (1.0 - a.arm_limit)), ms
    _rg.clamp_mask = _clamp_arm
raw, kps = render.load_inputs()
icons = render.build_icons(raw, kps)
step = [s for s in ctx.steps() if (ctx.step_spec(s)[3] == (a.form == "wrong"))][0]
native._ST.clear(); gc.collect()
t1 = time.time()
fr2, base, sh_m, w_fg, kp = render.render_pose(step, raw[step], kps[step], 0, icons, qc=True)
grade_s = time.time() - t1
from gymform import region_glow as rg
cm, ms = rg.clamp_mask(kp, a.form == "wrong", w_fg.shape)
info_c = {}
if a.callout == "joint":
    from gymform import joint_callout as jc
    lay, info_c = jc.draw(base, kp, protect=cm, is_wrong=a.form == "wrong", label=cfg["motions"][a.form]["callout"]["l1"])
    pre = jc.composite(base, lay)
    cv2.imwrite(os.path.join(a.out, f"still_{a.form}_{a.name}_callout_rgba.png"),
                cv2.cvtColor((np.clip(lay, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA))
else:
    pre = ctx.LAST_MASKS["pre_ui"]
one = [icons[step - 1]]
fr = layout.composite_title_and_preview_v4(pre, layout.fg_for_preview(w_fg), one, 0, a.form == "wrong", cfg["exercise_name"])
p = os.path.join(a.out, f"still_{a.form}_{a.name}_1080.jpg")
cv2.imwrite(p, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
cv2.imwrite(os.path.join(a.out, f"still_{a.form}_{a.name}_base_1080.png"), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
np.savez_compressed(os.path.join(a.out, f"still_{a.form}_{a.name}_regions.npz"), allowed=(cm > 0.02).astype(np.uint8), w_fg=(w_fg * 255).astype(np.uint8))
json.dump({"frame": p, "step": step, "joint_callout": info_c, "callout": ctx.LAST_MASKS.get("callout"), "grade_s": round(grade_s, 2),
           "total_s_incl_imports": round(time.time() - T0, 2), "kp": {k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in kp.items() if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 2}},
          open(os.path.join(a.out, f"still_{a.form}_{a.name}.json"), "w"), indent=1)
print(json.dumps({"frame": p, "joint_callout": info_c, "grade_s": round(grade_s, 2), "total_s": round(time.time() - T0, 2), "callout": ctx.LAST_MASKS.get("callout")}))
