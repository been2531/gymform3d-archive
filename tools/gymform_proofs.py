#!/usr/bin/env python3
"""Visual proofs from the FINAL mp4: Day 1 v36 vs Day N X-Ray comparison, and an equipment debug strip
(source | expected footprint (red) + backdrop covered (yellow) | w_fg | final mp4 frame) per QC frame."""
import json, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymform import qc as q

def frame(mp4, i):
    cap = cv2.VideoCapture(mp4); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read(); cap.release(); return f

def label(img, t):
    img = img.copy(); cv2.putText(img, t, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA); return img

def main(day):
    cfg = json.load(open(os.path.join(ROOT, "configs", f"day{day:02d}.json")))
    mp4, qd = os.path.join(ROOT, cfg["out_mp4"]), os.path.join(ROOT, cfg["qc_dir"])
    meta = json.load(open(os.path.join(qd, "meta.json")))
    tag = os.path.basename(qd)
    # 1) Day 1 v36 X-Ray vs this day's bone steps (peak frames, from the final mp4)
    ref = cv2.imread(os.path.join(ROOT, "output/qc/ref/day1_v36_xray_t1.5.jpg"))
    tiles = [label(cv2.resize(ref, (540, 960)), "Day 1 v36 (approved)")]
    for s in cfg.get("bone_steps", [1, 3]):
        fm = meta["frames"][f"step{s}_peak"]
        tiles.append(label(cv2.resize(frame(mp4, fm["frame_idx"]), (540, 960)), f"Day {day} Step {s} (final mp4)"))
    p1 = os.path.join(ROOT, "output", "qc", f"{tag}_day1_vs_day{day}_xray.jpg")
    cv2.imwrite(p1, np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 92])
    # 2) equipment strip: every QC frame of the requested step (default 2), source vs final
    s = int(os.environ.get("PROOF_STEP", "2"))
    rows = []
    for key, fm in sorted(meta["frames"].items(), key=lambda kv: kv[1]["frame_idx"]):
        if fm["step"] != s:
            continue
        pre = os.path.join(qd, key)
        src = cv2.imread(pre + "_src.png"); pl = cv2.imread(os.path.join(qd, f"step{s}_plate.png"))
        z, w, sh = q.load_mask(pre + "_eqzone.png"), q.load_mask(pre + "_wfg.png"), q.load_mask(pre + "_shorts.png")
        rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        f_, r_, e = q.check_equipment(key, rgb(src), rgb(pl), z, w, sh, fm["kp"])
        d = np.abs(src.astype(np.float32) - pl).max(-1)
        bk = q.LAST_BK[0] if q.LAST_BK[0] is not None else np.zeros(e.shape, bool)
        ov = src.copy(); ov[e] = (ov[e] * 0.5 + [0, 0, 120]).astype(np.uint8); ov[bk] = (0, 255, 255)
        fin = frame(mp4, fm["frame_idx"])
        ys, xs = np.where(e); y0, y1 = max(0, ys.min() - 30), min(1920, ys.max() + 30)
        g = cv2.cvtColor((np.clip(w, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        row = np.concatenate([label(src[y0:y1], f"{key} source"), ov[y0:y1], g[y0:y1], label(fin[y0:y1], "final mp4")], 1)
        rows.append(cv2.resize(row, (1600, int(row.shape[0] * 1600 / row.shape[1]))))
        print(r_[0][:200])
    p2 = os.path.join(ROOT, "output", "qc", f"{tag}_step{s}_equipment_strip.jpg")
    cv2.imwrite(p2, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(p1); print(p2)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
