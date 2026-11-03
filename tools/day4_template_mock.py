#!/usr/bin/env python3
"""Static Day 4-template mock (t637u/t638u): run the UNCHANGED production pipeline (gymform native_grade + layout:
Kaushan back-wall title, 2-slot top preview pill (t568u 2-step day), Day 4 callout/laser, see-through overlap
alpha 0.40 over mannequin/equipment) on START FRAMES only. Each start frame becomes a 12-frame static clip so the
pipeline can load it; no video generation. Writes 1080x1920 frames + a 540x960 side-by-side.
usage: nice -n 19 python3 tools/day4_template_mock.py <wrong_start_720.png> <right_start_720.png> <out_dir>"""
import sys, os, json, copy, gc
import cv2, numpy as np
sys.path.insert(0, "/workspace/daily_info_reels")
from gymform import ctx, render, native, layout
wpng, rpng, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
assets = os.path.join(out, "_static_assets"); os.makedirs(assets, exist_ok=True)
for name, p in (("wrong_static.mp4", wpng), ("right_static.mp4", rpng)):
    im = cv2.resize(cv2.imread(p), (720, 1280), interpolation=cv2.INTER_AREA)
    vw = cv2.VideoWriter(os.path.join(assets, name), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (720, 1280))
    for _ in range(12):
        vw.write(im)
    vw.release()
cfg = copy.deepcopy(json.load(open("/workspace/daily_info_reels/configs/day07_muscle2.json")))
cfg["assets_dir"] = assets
cfg["kp_cache"] = os.path.join(out, "_kps_mock.pkl")
cfg["tmp_dir"] = os.path.join(out, "_tmp_mock")
for form in ("wrong", "right"):
    m = cfg["motions"][form]; fn = f"{form}_static.mp4"
    m["base"] = fn; m["i0"] = 0; m["i1"] = 11; m["renders"]["front"]["file"] = fn
# t637u framing: the whole real machine (tower + stack + footplate) must stay in frame, so this mock uses the
# floor-line framing (no target_zoom crop) at scale MOCK_SCALE; head/shoulders then sit under the title band,
# which is the see-through overlap case of t638u.
MOCK_SCALE = float(os.environ.get("MOCK_SCALE", "1.0"))
cfg.pop("target_zoom", None)
cfg["step_scale"] = {"1": MOCK_SCALE, "2": MOCK_SCALE}
cfg["step_dy"] = {"1": int(os.environ.get("MOCK_DY", "0")), "2": int(os.environ.get("MOCK_DY", "0"))}
ctx.configure(cfg)
raw, kps = render.load_inputs()
# overlap stress case (t638u): optional extra shift AFTER the normal framing so the head runs under the title
SX, SY = int(os.environ.get("MOCK_SHIFT_X", "0")), int(os.environ.get("MOCK_SHIFT_Y", "0"))
SUF = os.environ.get("MOCK_SUFFIX", "")
for s_ in ctx.steps():
    ctx.STEP_DX[s_] = ctx.STEP_DX.get(s_, 0) + SX
    ctx.b4.STEP_DY[s_] = ctx.b4.STEP_DY.get(s_, 0) + SY
icons = render.build_icons(raw, kps)
outs = []
a_t, a_p = layout.overlap_alpha()
for s in ctx.steps():
    native._ST.clear(); gc.collect()
    fr, base, sh_m, w_fg, kp = render.render_pose(s, raw[s], kps[s], 0, icons, qc=True)
    fn, i0, i1, is_w, *_ = ctx.step_spec(s)
    tag = "wrong" if is_w else "right"
    path = os.path.join(out, f"mock_day4_{tag}{SUF}_1080.jpg")
    cv2.imwrite(path, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    occ = layout.fg_for_preview(w_fg)
    rx0, ry0, rx1, ry1 = layout.PREV_ROI; ty0, ty1 = layout.TITLE_ROI
    info = {"step": s, "path": path, "overlap_alpha": {"title": a_t, "preview": a_p},
            "occluder_frac_in_preview_roi": round(float((occ[ry0:ry1, rx0:rx1] > 0.5).mean()), 4),
            "occluder_frac_in_title_roi": round(float((occ[ty0:ty1] > 0.5).mean()), 4)}
    json.dump(info, open(os.path.join(out, f"mock_day4_{tag}{SUF}.json"), "w"), indent=1)
    print(info)
    outs.append(fr)
native._ST.clear(); gc.collect()
sbs = np.concatenate([cv2.resize(o, (540, 960), interpolation=cv2.INTER_AREA) for o in outs], axis=1)
cv2.imwrite(os.path.join(out, f"mock_day4_template{SUF}_540.jpg"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
