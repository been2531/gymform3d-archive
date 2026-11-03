#!/usr/bin/env python3
"""t651u: time the D7 grade on the pinned stills (same config/path as tools/day4_template_mock.py, re-using its
static assets + kp cache). Prints load / per-step prepare (12 frames) / per-frame render_pose (grade + Day 4 fibre +
clamp + UI). usage: nice -n 19 python3 tools/time_d7_grade.py <mock_dir>"""
import sys, os, json, copy, time, gc
sys.path.insert(0, "/workspace/daily_info_reels")
from gymform import ctx, render, native
out = sys.argv[1]
cfg = copy.deepcopy(json.load(open("/workspace/daily_info_reels/configs/day07_muscle2.json")))
cfg.update(assets_dir=os.path.join(out, "_static_assets"), kp_cache=os.path.join(out, "_kps_mock.pkl"), tmp_dir=os.path.join(out, "_tmp_mock"))
for form in ("wrong", "right"):
    m = cfg["motions"][form]; fn = f"{form}_static.mp4"
    m.update(base=fn, i0=0, i1=11, mirror=True); m["renders"]["front"].update(file=fn, mirror=True)
cfg.pop("target_zoom", None); cfg["step_scale"] = {"1": 1.0, "2": 1.0}; cfg["step_dy"] = {"1": 0, "2": 0}
for off in (False, True):
    c2 = copy.deepcopy(cfg)
    if off:
        c2.pop("native_muscle_shader", None); c2.pop("muscle_glow_clamp", None)
    ctx.configure(c2)
    t = time.time(); raw, kps = render.load_inputs(); icons = render.build_icons(raw, kps); tl = time.time() - t
    for s in ctx.steps():
        native._ST.clear(); gc.collect()
        _, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(s)
        t = time.time(); native.prepare(s, raw[s], kps[s], y_sh); tp = time.time() - t
        ts = []
        for p in (0, 5, 11):
            t = time.time(); render.render_pose(s, raw[s], kps[s], p, icons); ts.append(time.time() - t)
        print(json.dumps({"day4_fibre+clamp": not off, "step": s, "load+icons_s": round(tl, 2), "prepare_12f_s": round(tp, 2),
                          "render_pose_s_per_frame": [round(x, 3) for x in ts]}))
