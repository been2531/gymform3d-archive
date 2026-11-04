#!/usr/bin/env python3
"""Phone-friendly result sheet for a rig3d run (PNG, 1080 wide): verdict header, 4 graded frames (start / 1/3 / peak /
end), Day 4 reference + mask overlay + torso/elbow curves, then every QC check with PASS/FAIL. No third-party frames
(the YMove source clip is never shown or committed).  usage: python3 tools/rig3d/sheet.py RUN_DIR CFG.json OUT.png"""
import glob, json, os, subprocess, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WH = (255, 255, 255); GR = (90, 220, 120); RD = (90, 90, 240); MU = (170, 170, 170)   # BGR

def txt(img, s, x, y, sc=0.7, c=WH, th=1):
    cv2.putText(img, str(s), (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

def tile(p, w, h):
    im = cv2.imread(p); return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

def build(run_dir, cfg, out):
    qc = json.load(open(os.path.join(run_dir, "qc.json"))); meta = json.load(open(os.path.join(run_dir, "run.json")))
    G = sorted(glob.glob(os.path.join(run_dir, "grade/g_*.png"))); n = len(G)
    R = json.load(open(os.path.join(run_dir, "render/checks.json")))["frames"]
    el = np.array([f["elbow_deg"] for f in R]); peak = int(np.argmin(el))
    pick = [0, n // 3, min(peak, n - 1), n - 1]
    W = 1080; tw, th = 270, 480
    head = np.full((150, W, 3), 24, np.uint8)
    ok = qc["pass"]; txt(head, f"{cfg['name']}  rig3d v2  -  QC {'PASS' if ok else 'FAIL'} {qc['n_pass']}/{qc['n']}", 24, 52, 1.05, GR if ok else RD, 2)
    pr = meta.get("production", {})
    txt(head, f"run {meta['run_id']}  commit {meta.get('code_commit', '?')}  chain {pr.get('chain_s', meta.get('wall_s', '?'))} s (render {pr.get('render_s', '?')} s)  peak RAM {pr.get('peak_rss_mb', meta.get('peak_rss_mb', '?'))} MB", 24, 92, 0.58, MU)
    txt(head, f"0 generations - source pose: {cfg['source']['provider']} (pose only) - body Grab3D CC0 - HDRI Poly Haven CC0", 24, 124, 0.55, MU)
    row1 = np.hstack([tile(G[i], tw, th) for i in pick])
    lab = np.full((34, W, 3), 24, np.uint8)
    for k, (i, s) in enumerate(zip(pick, ("start", "1/3", "peak (elbow)", "end"))): txt(lab, f"{s}  f{i + 1}", k * tw + 10, 24, 0.6, MU)
    # row 2: Day 4 reference | mask overlay at peak | curves
    d4 = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "18.5", "-i", os.path.join(ROOT, "output/day4_romanian_deadlift_4step_master.mp4"),
                         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    d4 = cv2.resize(np.frombuffer(d4, np.uint8).reshape(1920, 1080, 3), (tw, th)) if d4 else np.zeros((th, tw, 3), np.uint8)
    fb = sorted(glob.glob(os.path.join(run_dir, "render/m_*.png")))[pick[2]]; mk = cv2.imread(fb, cv2.IMREAD_UNCHANGED)
    gp = cv2.resize(cv2.imread(G[pick[2]]), (mk.shape[1], mk.shape[0]))
    ov = gp.copy(); col = mk[..., :3].astype(np.float32); a = (col.max(-1, keepdims=True) > 40) * 0.65
    ov = (ov * (1 - a) + col * a).astype(np.uint8); ov = cv2.resize(ov, (tw, th))
    txt(ov, "mask: red lats, green rhomb, blue midtrap", 6, th - 10, 0.42, WH)
    pl = np.full((th, 2 * tw, 3), 30, np.uint8); t = np.array([f["torso_deg"] for f in R])
    def curve(v, lo, hi, c, y0, y1, label):
        xs = np.linspace(20, 2 * tw - 20, len(v)); ys = y1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0)
        cv2.polylines(pl, [np.stack([xs, ys], 1).astype(np.int32)], False, c, 2, cv2.LINE_AA); txt(pl, label, 20, y0 - 8, 0.55, c)
    curve(t, 30, 70, GR, 40, 210, f"torso deg (30-70): {t.min():.1f}-{t.max():.1f}  target {cfg['motion']['torso_target_deg']}")
    curve(el, 60, 180, (230, 180, 80), 270, 450, f"working elbow deg (60-180): ROM {el.max() - el.min():.0f}")
    row2 = np.hstack([d4, ov, pl]); lab2 = np.full((34, W, 3), 24, np.uint8)
    txt(lab2, "Day 4 reference", 10, 24, 0.6, MU); txt(lab2, "mask overlay (peak)", tw + 10, 24, 0.6, MU); txt(lab2, "per-frame rig curves", 2 * tw + 10, 24, 0.6, MU)
    rows = []
    for c in qc["checks"]:
        r = np.full((34, W, 3), 20, np.uint8); col = GR if c["pass"] else RD
        txt(r, "PASS" if c["pass"] else "FAIL", 16, 24, 0.62, col, 2); txt(r, f"[{c['group']}] {c['name']}", 96, 24, 0.58, WH)
        txt(r, f"{c['value']}  (limit {c['limit']})"[:46], 610, 24, 0.5, MU); rows.append(r)
    foot = np.full((70, W, 3), 24, np.uint8)
    txt(foot, f"mp4: {os.path.relpath(meta['reel'], ROOT)}", 16, 28, 0.5, MU)
    txt(foot, f"auto-fix rules used: {meta.get('fixes_applied') or 'none'}   AI needed: {qc['needs_ai'] or 'no'}", 16, 56, 0.5, MU)
    sheet = np.vstack([head, row1, lab, row2, lab2] + rows + [foot]); cv2.imwrite(out, sheet, [cv2.IMWRITE_PNG_COMPRESSION, 9]); return out

if __name__ == "__main__":
    print(build(sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]))
