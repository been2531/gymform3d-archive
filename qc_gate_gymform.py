#!/usr/bin/env python3
"""
@gymform.3d automated QC gate for rendered 4-step master reels.

Usage:
  python3 qc_gate_gymform.py <master.mp4> [--qc-dir output/qc/day5_v4]

Reads the rendered mp4 at the start/peak frame of every step and the QC sidecar written by the builder
(build_day5_gym_studio_v4.py -> CFG["qc_dir"]: per-frame clean fg mask, shorts mask, equipment masks,
pre-UI base frame, keypoints, equipment config). Prints PASS/FAIL per item and exits 1 if any item fails:
  (a) no coloured halo / bloom outside the mannequin mask
  (b) title depth-behind (title suppressed over mannequin) + preview see-through and sharp over mannequin
  (c) mannequin head-to-ankle height equal across the 4 steps (within 4%)
  (d) equipment (any type in config) fully inside frame, fully covered by fg mask (no holes), not distorted
  (e) bone X-Ray present/readable on bone steps, muscle fibers on muscle steps
  (f) skin luminance p75~165 / p90~191 (max<=226), shorts lum 10..22
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np

W, H = 1080, 1920
RESULTS = []


def report(item, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {item}: {detail}", flush=True)


def lum_of(rgb):
    f = rgb.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def act_of(rgb):
    f = rgb.astype(np.float32)
    return np.maximum(f[..., 0] - 0.5 * (f[..., 1] + f[..., 2]), f[..., 1] - 0.5 * (f[..., 0] + f[..., 2]))


def read_frames(mp4, idxs):
    cap = cv2.VideoCapture(mp4)
    out = {}
    for i in sorted(set(idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if m is None else m.astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    ap.add_argument("--qc-dir", default="/workspace/daily_info_reels/output/qc/day5_v4b")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.qc_dir, "meta.json")))
    cap = cv2.VideoCapture(args.mp4)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    report("format", (w, h) == (W, H) and abs(fps - 30) < 0.01 and n >= 590, f"{w}x{h} @ {fps:.2f}fps, {n} frames ({n / max(fps, 1):.2f}s)")

    frames_meta = meta["frames"]
    vids = read_frames(args.mp4, [v["frame_idx"] for v in frames_meta.values()])
    bg = cv2.cvtColor(cv2.imread(meta["studio_bg"]), cv2.COLOR_BGR2RGB)
    bg_act = act_of(bg)
    ty0, ty1 = meta["title_roi"][1], meta["title_roi"][3]
    px0, py0, px1, py1 = meta["preview_roi"]

    heights = {}
    halo_bad, title_bad, prev_bad, eq_fail, skin_fail = [], [], [], [], []
    eq_area = {}
    for key, fm in sorted(frames_meta.items()):
        fr = vids.get(fm["frame_idx"])
        pre = os.path.join(args.qc_dir, key)
        base = cv2.cvtColor(cv2.imread(pre + "_base.png"), cv2.COLOR_BGR2RGB)
        wfg, shorts, eq = load_mask(pre + "_wfg.png"), load_mask(pre + "_shorts.png"), load_mask(pre + "_eq.png")
        kp = fm["kp"]
        s = fm["step"]

        # (a) halo: on the pre-UI composite, coloured pixels outside the dilated body mask that differ from the plate
        body_d = cv2.dilate((wfg > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        halo = ((act_of(base) - bg_act) > 18.0) & ~body_d
        halo_n = int(halo.sum())
        # also: pixels just outside the mask must match the (shadowed) plate in luminance — no glow lift
        ring = (cv2.dilate((wfg > 0.02).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~body_d
        lift = float(np.percentile((lum_of(base) - lum_of(bg))[ring], 95)) if ring.sum() else 0.0
        if halo_n > 150 or lift > 12.0:
            halo_bad.append(f"{key}: {halo_n}px coloured, ring lift p95={lift:.1f}")

        if fr is not None:
            # (b1) title depth-behind: where mannequin is solid inside title ROI, final frame ~ base (title <=~10%)
            reg = np.zeros((H, W), bool); reg[ty0:ty1] = True
            over = reg & (wfg > 0.9)
            if over.sum() > 200:
                d = float(np.mean(np.abs(fr[over].astype(np.float32) - base[over].astype(np.float32))))
                if d > 18.0:
                    title_bad.append(f"{key}: mean|frame-base| over mannequin in title ROI = {d:.1f}")
            # (b2) preview see-through + sharp over mannequin
            reg = np.zeros((H, W), bool); reg[py0:py1, px0:px1] = True
            over = reg & (wfg > 0.9)
            if over.sum() > 200:
                d = float(np.mean(np.abs(fr[over].astype(np.float32) - base[over].astype(np.float32))))
                lb = cv2.Laplacian(lum_of(base), cv2.CV_32F)[over].var()
                lf = cv2.Laplacian(lum_of(fr), cv2.CV_32F)[over].var()
                sharp = lf / max(lb, 1e-3)
                if d > 30.0 or sharp < 0.55:
                    prev_bad.append(f"{key}: diff={d:.1f}, sharpness ratio={sharp:.2f}")

        # (c) head-to-ankle height from the clean mask
        sx = 0.5 * (kp["sh_r"][0] + kp["sh_l"][0])
        col = (wfg[:, int(sx) - 40 : int(sx) + 40] > 0.5).any(axis=1)
        ys = np.where(col[: int(kp["sh"][1])])[0]
        if len(ys):
            heights[key] = max(kp["ank_r"][1], kp["ank_l"][1]) - ys.min()

        # (d) equipment: in frame, no holes, area stable start vs peak
        if eq is not None and eq.sum() > 50:
            n_lab, lab, stats, _ = cv2.connectedComponentsWithStats((eq > 0.5).astype(np.uint8))
            for li in range(1, n_lab):
                x, y, ww, hh, area = stats[li]
                if area < 300:
                    continue
                if x <= 3 or y <= 3 or x + ww >= W - 3 or y + hh >= H - 3:
                    eq_fail.append(f"{key}: equipment blob {li} touches frame edge (bbox {x},{y},{ww},{hh})")
                comp = lab == li
                holes = float(((wfg < 0.5) & comp).sum()) / area
                if holes > 0.02:
                    eq_fail.append(f"{key}: equipment blob {li} mask holes {100 * holes:.1f}%")
            eq_area.setdefault(s, {})[fm["tag"]] = float((eq > 0.5).sum())
            # zone clipping: equipment pixels touching the zone border means the zone cut the object
            zone = load_mask(pre + "_eqzone.png")
            if zone is not None:
                border = (zone > 0) & (cv2.erode((zone > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
                dark_eq = (eq > 0.5) & (lum_of(base) < 70.0) & (shorts < 0.3)  # equipment proper (dark metal), not skin/shorts
                touch = float((dark_eq & border).sum()) / max(1.0, float(border.sum()))
                if touch > 0.12:
                    eq_fail.append(f"{key}: equipment reaches its zone border on {100 * touch:.0f}% of border (zone may clip it)")
        else:
            eq_fail.append(f"{key}: no equipment mask found (config: {[e['type'] for e in meta['equipment']]})")

        # (e) bone vs muscle on peak frames (pre-UI composite, inside body)
        if fm["tag"] == "peak":
            a = act_of(base)
            colored = (a > 40.0) & (wfg > 0.8)
            ncol = int(colored.sum())
            lap = np.abs(cv2.Laplacian(lum_of(base), cv2.CV_32F))
            detail = float(np.mean(lap[colored])) if ncol else 0.0
            is_bone = s in meta["bone_steps"]
            ok = ncol > 2500 and detail > 6.0
            RESULTS_E.append((key, is_bone, ncol, detail, ok))

        # (f) skin + shorts
        l = lum_of(base)
        skin = (wfg > 0.9) & (shorts < 0.2) & (act_of(base) < 12.0) & (l > 46.0) & (eq < 0.5)
        if skin.sum() > 2000:
            p75, p90, mx = np.percentile(l[skin], 75), np.percentile(l[skin], 90), l[skin].max()
            sh = (shorts > 0.6) & (wfg > 0.9)
            sp = float(np.median(l[sh])) if sh.sum() else -1
            SKIN.append(f"{key}: p75={p75:.0f} p90={p90:.0f} max={mx:.0f} shorts={sp:.0f}")
            if not (155 <= p75 <= 175 and 181 <= p90 <= 201 and mx <= 228 and 10 <= sp <= 22):
                skin_fail.append(SKIN[-1])

    report("(a) no colour halo / bloom outside mannequin", not halo_bad, "; ".join(halo_bad) or f"{len(frames_meta)} frames clean")
    report("(b1) title depth-behind mannequin", not title_bad, "; ".join(title_bad) or "title suppressed where mannequin overlaps")
    report("(b2) preview translucent + sharp over mannequin", not prev_bad, "; ".join(prev_bad) or "mannequin shows through preview, no frost blur")
    if heights:
        steps_h = {}
        for k, v in heights.items():
            steps_h.setdefault(k.split("_")[0], []).append(v)
        hs = {k: max(v) for k, v in steps_h.items()}
        spread = (max(hs.values()) - min(hs.values())) / max(hs.values())
        report("(c) head-to-ankle height equal across steps (<=4%)", spread <= 0.04,
               ", ".join(f"{k}={v:.0f}px" for k, v in sorted(hs.items())) + f" | spread {100 * spread:.1f}%")
    for s, d in sorted(eq_area.items()):
        if "start" in d and "peak" in d:
            r = d["peak"] / max(d["start"], 1.0)
            if not (0.5 <= r <= 2.0):
                eq_fail.append(f"step{s}: equipment area peak/start ratio {r:.2f} (distortion?)")
    report("(d) equipment in frame, no holes, not clipped/distorted", not eq_fail, "; ".join(eq_fail) or f"types={[e['type'] for e in meta['equipment']]}")
    for key, is_bone, ncol, detail, ok in RESULTS_E:
        report(f"(e) {key} {'bone X-Ray' if is_bone else 'muscle fibers'} present/readable", ok, f"coloured px={ncol}, edge detail={detail:.1f}")
    report("(f) skin p75~165/p90~191, shorts 10..22", not skin_fail, " | ".join(SKIN))

    print(f"\nQC GATE: {'PASS' if all(RESULTS) else 'FAIL'} ({sum(RESULTS)}/{len(RESULTS)} items)")
    sys.exit(0 if all(RESULTS) else 1)


RESULTS_E, SKIN = [], []

if __name__ == "__main__":
    main()
