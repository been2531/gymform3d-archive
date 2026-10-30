"""Proof crops for docs/ROOTCAUSE_DAY6.md: raw aligned source frame (top row) vs our output (bottom row)."""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gymform import ctx, render, tempo
from gymform.ctx import b4
OUT = "output/qc/rootcause_day6"
cfg = "configs/day06_muscle2_v4.json"
ctx.configure(json.load(open(cfg)), json.load(open("_cache_gymform/overrides_day06_muscle2_v4.json")))
raw, kps = render.load_inputs()
cap = cv2.VideoCapture("output/day6_incline_dumbbell_press_muscle2_v4_master.mp4")
outs = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    outs.append(f)


def strip(s, frames, key, r=130, dy=0, name=""):
    y_sh = ctx.step_spec(s)[5]
    n = len(raw[s]); rows = [[], []]
    for fl in frames:
        p = tempo.pose_index(fl, n)
        rgb, kp = b4.transform_frame_and_kp_1080p(raw[s][p], kps[s][p], y_sh, s)
        if callable(key):
            cx, cy = key(kp)
        else:
            cx, cy = kp[key]
        cx, cy = int(cx), int(cy + dy)
        a = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        o = outs[(s - 1) * 150 + fl]
        for i, im in enumerate((a, o)):
            pad = cv2.copyMakeBorder(im, r, r, r, r, cv2.BORDER_REPLICATE)
            c = pad[cy:cy + 2 * r, cx:cx + 2 * r].copy()
            c = cv2.convertScaleAbs(c, alpha=2.2, beta=-30)   # brightened x2.2 so the dark shapes are visible
            cv2.putText(c, f"{'RAW' if i == 0 else 'OUT'} f{fl} p{p}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            rows[i].append(c)
    im = np.vstack([np.hstack(rows[0]), np.hstack(rows[1])])
    cv2.imwrite(f"{OUT}/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])


strip(1, [10, 12, 14, 16, 18, 20], "wr_l", name="s1_right_dumbbell_raw_vs_out")
strip(2, [10, 12, 14, 16, 18, 20], "wr_r", name="s2_left_dumbbell_raw_vs_out")
mid = lambda kp: (0.25 * (kp["hip_r"][0] + kp["hip_l"][0] + kp["kn_r"][0] + kp["kn_l"][0]), 0.25 * (kp["hip_r"][1] + kp["hip_l"][1] + kp["kn_r"][1] + kp["kn_l"][1]))
strip(2, [10, 12, 14, 16, 18, 20], mid, r=150, dy=40, name="s2_bench_between_legs_raw_vs_out")
strip(2, [10, 12, 14, 16, 18, 20], "elb_r", r=130, name="s2_elbow_forearm_raw_vs_out")
strip(1, [10, 12, 14, 16, 18, 20], lambda kp: (kp["ank_r"][0], kp["ank_r"][1]), r=150, name="s1_feet_bench_base_raw_vs_out")
cut = np.hstack([cv2.resize(outs[149], (540, 960)), cv2.resize(outs[150], (540, 960))])
cv2.imwrite(f"{OUT}/step_cut_f149_vs_f150.jpg", cut, [cv2.IMWRITE_JPEG_QUALITY, 88])
