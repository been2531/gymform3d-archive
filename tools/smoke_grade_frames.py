#!/usr/bin/env python3
"""Apply the UNCHANGED production Day 4 look (gymform native_grade with configs/day07_muscle2.json, only the clip
files / windows swapped) to a few frames of a wrong/right clip pair. Writes base graded frames (no UI layer).
usage: nice -n 19 python3 tools/smoke_grade_frames.py <tag> <assets_dir> <wrong.mp4> <right.mp4> <out_dir> <f0,f1,f2>"""
import sys, os, json, copy, gc
import cv2, numpy as np
sys.path.insert(0, "/workspace/daily_info_reels")
from gymform import ctx, render, native
tag, assets, wf, rf, out, frames = sys.argv[1:7]
frames = [int(x) for x in frames.split(",")]
os.makedirs(out, exist_ok=True)
cfg = json.load(open("/workspace/daily_info_reels/configs/day07_muscle2.json"))
cfg = copy.deepcopy(cfg)
cfg["assets_dir"] = assets
cfg["kp_cache"] = os.path.join(out, f"_kps_{tag}.pkl")
cfg["tmp_dir"] = os.path.join(out, f"_tmp_{tag}")
for form, fn in (("wrong", wf), ("right", rf)):
    n = int(cv2.VideoCapture(os.path.join(assets, fn)).get(cv2.CAP_PROP_FRAME_COUNT))
    m = cfg["motions"][form]; m["base"] = fn; m["i0"] = 0; m["i1"] = n - 1
    m["renders"]["front"]["file"] = fn
ctx.configure(cfg)
raw, kps = render.load_inputs()
res = {}
for s in ctx.steps():
    fn, i0, i1, is_w, mode, y_sh = ctx.step_spec(s)
    native._ST.clear(); gc.collect()
    native.prepare(s, raw[s], kps[s], y_sh)
    for p in frames:
        p = min(p, len(raw[s]) - 1)
        base, *_ = native.grade(s, p, is_w)
        im = np.clip(base, 0, 255).astype(np.uint8)
        path = os.path.join(out, f"graded_{tag}_{'wrong' if is_w else 'right'}_f{p:03d}.jpg")
        cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        L = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
        res[path] = {"bg_rgb_top": np.median(im[60:180, 100:980].reshape(-1, 3), 0).tolist()}
    native._ST.clear(); gc.collect()
json.dump(res, open(os.path.join(out, f"graded_{tag}.json"), "w"), indent=1)
print("done", list(res))
