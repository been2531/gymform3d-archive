"""Root-cause metrics (t595u-t597u): raw generated clip vs our rendered output, aligned frame by frame.
For every output frame we rebuild the exact raw source frame the renderer used (same pose index from the tempo
wave, same 1080 transform), so any instability present in the output but not in the aligned raw frame was created
by our processing. Temporal instability = optical-flow-compensated residual (Farneback flow on the RAW frames,
applied to both raw and output), so real motion cancels and only morph / boil / flicker remains.
Usage: python3 -m gymform.temporal_qc gate <cfg> <mp4> | python3 tools/rootcause_metrics.py day6 <cfg> <mp4> [tag]   |   python3 tools/rootcause_metrics.py day4"""
import json, os, sys, pickle
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("ROOTCAUSE_OUT", "output/qc/rootcause_day6")
HS = 0.5   # metrics at half resolution (540x960)


def lum(a):
    a = a.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def act(a):
    a = a.astype(np.float32)
    return np.maximum(a[..., 0] - 0.5 * (a[..., 1] + a[..., 2]), a[..., 1] - 0.5 * (a[..., 0] + a[..., 2]))


def border_bg(a):
    h = a.shape[0]
    b = np.concatenate([a[int(h * .365):int(h * .94), :12].reshape(-1, 3), a[int(h * .365):int(h * .94), -12:].reshape(-1, 3)])
    return np.median(b, axis=0)


def sil(a, thr):
    return (np.abs(a.astype(np.float32) - border_bg(a)[None, None]).max(-1) > thr).astype(np.uint8)


def box(shape, c, r):
    m = np.zeros(shape, np.uint8)
    x, y = int(c[0] * HS), int(c[1] * HS)
    m[max(0, y - r):y + r, max(0, x - r):x + r] = 1
    return m


def warp(img, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def analyze(raws, outs, kps, pidx, target_fn, wrists=True, label=""):
    """raws/outs: lists of 1080x1920 RGB aligned per output frame; kps: 1080 keypoints per output frame."""
    sm = lambda a: a if a.shape[1] == 540 else cv2.resize(a, (540, 960), interpolation=cv2.INTER_AREA)
    R = {k: [] for k in ("boil_raw", "boil_out", "edge_raw", "edge_out", "silflk_raw", "silflk_out",
                         "db_raw", "db_out", "bench_raw", "bench_out", "glow_raw", "glow_out")}
    dbA = {"raw": [], "out": []}; gl = {"raw": [], "out": []}; feet = {"raw": [], "out": []}
    form = {"raw": [], "out": []}; fib = {"raw": [], "out": []}
    for t in range(len(outs)):
        r1, o1 = sm(raws[t]), sm(outs[t])
        sr, so = sil(r1, 14), sil(o1, 8)
        so[:310] = 0; sr[:310] = 0           # UI band
        kp = kps[t]
        # wrist-box silhouette area (dumbbell + hand), glow centroid offset vs callout target, feet contrast
        if wrists:
            for side in ("wr_r", "wr_l"):
                b = box(sr.shape, kp[side], 70)
                dbA["raw"].append((side, t, int((sr & b).sum()))); dbA["out"].append((side, t, int((so & b).sum())))
        tg = np.array(target_fn(kp)) * HS
        for nm, im, s_ in (("raw", r1, sr), ("out", o1, so)):
            g = (act(im) > 25) & (s_ > 0)
            if g.sum() > 150:
                yy, xx = np.nonzero(g)
                gl[nm].append((xx.mean() - tg[0], yy.mean() - tg[1]))
            fb = np.zeros_like(s_)
            for a_ in ("ank_r", "ank_l"):
                fb |= box(s_.shape, kp[a_], 40)
            bgv = lum(border_bg(im)[None, None])[0, 0]
            fm = (fb > 0) & (s_ > 0)
            feet[nm].append(float(np.median(np.abs(lum(im)[fm] - bgv))) if fm.sum() > 50 else np.nan)
            inn = cv2.erode(sr, np.ones((15, 15), np.uint8)) > 0
            if wrists:
                for side in ("wr_r", "wr_l"):
                    inn &= box(sr.shape, kp[side], 80) == 0
            inn &= act(r1) < 12
            L = lum(im)
            if inn.sum() > 2000:
                form[nm].append(float(cv2.GaussianBlur(L, (0, 0), 8)[inn].std()))
                fib[nm].append(float((L - cv2.GaussianBlur(L, (0, 0), 2))[inn].std()))
        if t == 0 or pidx[t] == pidx[t - 1]:
            continue
        r0, o0 = sm(raws[t - 1]), sm(outs[t - 1])
        g0, g1 = lum(r0).astype(np.uint8), lum(r1).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 4, 21, 4, 7, 1.5, 0)
        res_r = np.abs(lum(r1) - warp(lum(r0), flow))
        res_o = np.abs(lum(o1) - warp(lum(o0), flow))
        band = (cv2.dilate(sr, np.ones((9, 9), np.uint8)) - cv2.erode(sr, np.ones((9, 9), np.uint8))) > 0
        inn = (cv2.erode(sr, np.ones((15, 15), np.uint8)) > 0)
        sr0, so0 = sil(r0, 14), sil(o0, 8); sr0[:310] = 0; so0[:310] = 0
        wsr = warp(sr0.astype(np.float32), flow) > 0.5; wso = warp(so0.astype(np.float32), flow) > 0.5
        R["silflk_raw"].append(float((wsr ^ (sr > 0)).sum())); R["silflk_out"].append(float((wso ^ (so > 0)).sum()))
        db = np.zeros_like(sr)
        if wrists:
            for side in ("wr_r", "wr_l"):
                db |= box(sr.shape, kp[side], 70)
        hip = (0.5 * (kp["hip_r"][0] + kp["hip_l"][0]), 0.5 * (kp["hip_r"][1] + kp["hip_l"][1]))
        knee = (0.5 * (kp["kn_r"][0] + kp["kn_l"][0]), 0.5 * (kp["kn_r"][1] + kp["kn_l"][1]))
        bench = box(sr.shape, (0.5 * (hip[0] + knee[0]), 0.5 * (hip[1] + knee[1]) + 40), 60)
        glow = cv2.dilate(((act(r1) > 25) & (sr > 0)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        inn_nodb = inn & (db == 0) & ~glow
        for k, m in (("boil", inn_nodb), ("edge", band & (db == 0)), ("db", db > 0), ("bench", bench > 0), ("glow", glow)):
            if m.sum() > 100:
                R[k + "_raw"].append(float(res_r[m].mean())); R[k + "_out"].append(float(res_o[m].mean()))
    f = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    res = {k: round(f(v), 2) for k, v in R.items()}
    for nm in ("raw", "out"):
        g = np.array(gl[nm]) if gl[nm] else np.zeros((0, 2))
        res[f"glow_offset_std_px_{nm}"] = round(float(np.hypot(*g.std(0)) / HS), 1) if len(g) > 3 else None
        res[f"glow_offset_range_px_{nm}"] = round(float(np.hypot(*(g.max(0) - g.min(0))) / HS), 1) if len(g) > 3 else None
        res[f"feet_contrast_{nm}"] = round(float(np.nanmedian(feet[nm])), 1)
        res[f"form_std_{nm}"] = round(f(form[nm]), 2); res[f"fibre_std_{nm}"] = round(f(fib[nm]), 2)
        if dbA[nm]:
            cov = []
            for side in ("wr_r", "wr_l"):
                a = np.array([v for s_, t, v in dbA[nm] if s_ == side], np.float32)
                d = np.abs(np.diff(a)) / np.maximum(a[1:], 1)
                cov.append(round(float(np.percentile(d[d > 0], 90)) * 100, 1) if (d > 0).any() else 0.0)
            res[f"dumbbell_area_jump_p90_pct_{nm}"] = cov
    res["unique_src_frames"] = len(set(pidx)); res["changes"] = int(sum(1 for i in range(1, len(pidx)) if pidx[i] != pidx[i - 1]))
    print(label, json.dumps(res), flush=True)
    return res


def read_mp4(p):
    cap, fr = cv2.VideoCapture(p), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (540, 960), interpolation=cv2.INTER_AREA))
    return fr


def run_day6(cfg_p, mp4, tag):
    from gymform import ctx, render, tempo
    from gymform.ctx import b4
    var = os.path.splitext(os.path.basename(cfg_p))[0]
    ovr_p = f"_cache_gymform/overrides_{var}.json"
    ctx.configure(json.load(open(cfg_p)), json.load(open(ovr_p)) if os.path.exists(ovr_p) else None)
    raw, kps = render.load_inputs()
    outs = read_mp4(mp4)
    allres = {}
    for s in ctx.steps():
        y_sh = ctx.step_spec(s)[5]
        n = len(raw[s]); cache = {}
        pidx = [tempo.pose_index(f, n, render.step_phase(s)) for f in range(tempo.FRAMES_PER_STEP)]
        R_, K_ = [], []
        for p in pidx:
            if p not in cache:
                rgb, kp = b4.transform_frame_and_kp_1080p(raw[s][p], kps[s][p], y_sh, s)
                cache[p] = (cv2.resize(np.clip(rgb, 0, 255).astype(np.uint8), (540, 960), interpolation=cv2.INTER_AREA), kp)
            R_.append(cache[p][0]); K_.append(cache[p][1])
        O_ = outs[(s - 1) * 150:s * 150]

        def tgt(kp):
            sd = "r" if kp["sh_r"][0] <= kp["sh_l"][0] else "l"
            v, b = kp[f"sh_{sd}"], kp[f"elb_{sd}"]
            return (0.48 * v[0] + 0.52 * b[0], 0.48 * v[1] + 0.52 * b[1])
        allres[s] = analyze(R_, O_, K_, pidx, tgt, True, f"{tag} step{s}")
        allres[s]["n_src_slice"] = n
        # kp jitter: high-frequency part of the anchor track over the native slice order
        tr = np.array([tgt(b4.transform_frame_and_kp_1080p(raw[s][i], kps[s][i], y_sh, s)[1]) for i in range(n)])
        allres[s]["anchor_step_px_p90"] = round(float(np.percentile(np.hypot(*np.diff(tr, axis=0).T), 90)), 1)
        # flow-tracked anchor (follows the actual muscle surface) vs keypoint anchor, native slice order
        p = tr[0].copy(); drift = []
        prev = cv2.cvtColor(np.clip(b4.transform_frame_and_kp_1080p(raw[s][0], kps[s][0], y_sh, s)[0], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        for i in range(1, n):
            cur = cv2.cvtColor(np.clip(b4.transform_frame_and_kp_1080p(raw[s][i], kps[s][i], y_sh, s)[0], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p.reshape(1, 1, 2).astype(np.float32), None, winSize=(41, 41), maxLevel=3)
            if st[0, 0]:
                p = nxt.reshape(2)
            drift.append(float(np.hypot(*(p - tr[i])))); prev = cur
        allres[s]["anchor_vs_flowtrack_max_px"] = round(max(drift), 1)
        if s == ctx.steps()[0]:
            last_raw, last_kp = R_[-1], K_[-1]
        else:
            fr0 = O_[0]; frl = outs[(s - 1) * 150 - 1]
            jump = {k: round(float(np.hypot(K_[0][k][0] - last_kp[k][0], K_[0][k][1] - last_kp[k][1])), 1) for k in ("sh_r", "sh_l", "hip_r", "hip_l", "kn_r", "ank_r", "ank_l", "wr_r", "wr_l")}
            body = lambda a: lum(a)[(sil(a, 8) > 0) & (np.arange(a.shape[0])[:, None] > 310)]
            allres["cut"] = {"kp_jump_px": jump, "body_lum_mean_before": round(float(body(frl).mean()), 1),
                             "body_lum_mean_after": round(float(body(fr0).mean()), 1),
                             "frame_diff_at_cut": round(float(np.abs(lum(fr0) - lum(frl)).mean()), 2),
                             "frame_diff_typical_p90": round(float(np.percentile([np.abs(lum(outs[i]) - lum(outs[i - 1])).mean() for i in range(1, 149)], 90)), 2)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(allres, open(f"{OUT}/metrics_{tag}.json", "w"), indent=1)
    print(json.dumps(allres.get("cut")))
    return allres


H_ = 1920


def run_day4():
    import build_day4_rdl_master as d4
    b4 = d4.b4
    kps = pickle.load(open(d4.CACHE_KP_PKL, "rb"))
    outs = read_mp4("output/day4_romanian_deadlift_4step_master.mp4")
    allres = {}
    for s, (fn, i0, i1, is_w, vm, y_sh) in d4.STEP_SPECS_DAY4.items():
        seq = d4.load_native_slice_720p(os.path.join(d4.OUT_DIR, fn), i0, i1)
        n = len(seq)
        import math
        pidx = [int(round((0.5 - 0.5 * math.cos(2 * math.pi * (f / 150.0) * 1.5)) * (n - 1))) for f in range(150)]
        cache, R_, K_ = {}, [], []
        for p in pidx:
            if p not in cache:
                rgb, kp = b4.transform_frame_and_kp_1080p(seq[p], kps[s][p], y_sh, s)
                cache[p] = (cv2.resize(np.clip(rgb, 0, 255).astype(np.uint8), (540, 960), interpolation=cv2.INTER_AREA), kp)
            R_.append(cache[p][0]); K_.append(cache[p][1])
        tgt = (lambda kp: (0.48 * kp["sh"][0] + 0.52 * kp["hip_r"][0], 0.48 * kp["sh"][1] + 0.52 * kp["hip_r"][1])) if s < 4 else \
              (lambda kp: (0.48 * kp["hip_r"][0] + 0.52 * kp["kn_r"][0], 0.48 * kp["hip_r"][1] + 0.52 * kp["kn_r"][1]))
        allres[s] = analyze(R_, outs[(s - 1) * 150:s * 150], K_, pidx, tgt, False, f"day4 step{s}")
        allres[s]["n_src_slice"] = n
    json.dump(allres, open(f"{OUT}/metrics_day4.json", "w"), indent=1)


# t600 enforced gates (thresholds = Day 4's BEST measured ratio or better; docs/DEFECT_PLAYBOOK.md)
GATES = {"boil": 1.20, "silflk": 1.56, "glow": 1.59}
CUT = {"body_px": 25.0, "sh_px": 40.0, "lum": 3.0, "diff_x_p90": 2.0}


def gate(cfg_p, mp4, tag):
    r = run_day6(cfg_p, mp4, tag)
    fails, rows = [], []
    for s, v in r.items():
        if s == "cut":
            continue
        for k, thr in GATES.items():
            ratio = v[f"{k}_out"] / max(v[f"{k}_raw"], 1e-6)
            rows.append(f"step{s} {k} {v[f'{k}_raw']}->{v[f'{k}_out']} x{ratio:.2f} (<= {thr})")
            if ratio > thr:
                fails.append(rows[-1])
    c = r.get("cut")
    if c:
        j = c["kp_jump_px"]
        body = max(j[k] for k in ("hip_r", "hip_l", "ank_r", "ank_l"))
        sh = max(j["sh_r"], j["sh_l"])
        dl = abs(c["body_lum_mean_after"] - c["body_lum_mean_before"])
        dx = c["frame_diff_at_cut"] / max(c["frame_diff_typical_p90"], 1e-6)
        rows.append(f"cut hips/ankles {body:.0f}px (<= {CUT['body_px']:.0f}) shoulders {sh:.0f}px (<= {CUT['sh_px']:.0f}) "
                    f"wrists {max(j['wr_r'], j['wr_l']):.0f}px (info: wrong/right form differ) body lum {dl:.1f} (<= {CUT['lum']}) "
                    f"cut diff x{dx:.1f} typical (<= {CUT['diff_x_p90']})")
        if body > CUT["body_px"] or sh > CUT["sh_px"] or dl > CUT["lum"] or dx > CUT["diff_x_p90"]:
            fails.append(rows[-1])
    return not fails, rows, fails


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        ok, rows, fails = gate(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "gate")
        print("TEMPORAL_QC " + json.dumps({"ok": ok, "rows": rows, "fails": fails}))
        sys.exit(0 if ok else 1)
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "day4":
        run_day4()
    else:
        run_day6(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "day6")
