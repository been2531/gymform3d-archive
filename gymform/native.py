"""NATIVE GRADE look (t600, fix plan of docs/ROOTCAUSE_DAY6.md, CFG look == "native_grade").

Root cause of Day 5/6 morph, ghosts, halo and flicker: the picture was re-synthesised from per-frame keypoint masks
(body key, shorts zone, plate-diff equipment, metal re-shade, static/bench layer, shin capsule) and composited over a
replaced backdrop. This look KEEPS THE CLIP'S OWN PIXELS everywhere (body, bench, dumbbells, shorts, feet, floor):

  1. temporal denoise in SOURCE order: each frame = 0.5 own + 0.25 x each flow-warped neighbour, per-pixel occlusion
     guard (no blending where the warp disagrees), so generation micro-noise (fibre boil) is reduced, never smeared
  2. no unsharp (bg.transform honours CFG no_sharpen)
  3. backdrop flattened by a SMOOTH low-frequency gain field B0/B(x,y), B = normalised convolution (sigma 100 px) of the
     step's temporal-median backdrop pixels; applied to every pixel, so there is no matte edge anywhere
  4. ONE monotonic tone curve per step (constant over the step, matched across steps): backdrop -> Day 4 38 lum,
     skin p5/p50/p75/p90/max -> 52/64/75/88/143 (SPEC skin_lock_all_steps), darker-than-backdrop (shorts, bench) linear
  5. glow = the clip's own colour dominance, recoloured LINEARLY (no threshold ramp, no seed blur), temporally smoothed
     by the same flow filter
  6. UI matte (title/preview depth, laser gating, QC) = Day 4 soft colour matte on the graded frame; shorts only where
     the pixel differs from the flattened backdrop (foreground evidence)
  7. callout joints locked to the skin: Lucas-Kanade tracked through the source sequence, kept within 12 px of the kp
No per-frame cut-out, no keypoint geometry touches a pixel."""
import cv2
import numpy as np
from gymform.ctx import CFG, SPEC, W, H, b4, LAST_MASKS
from gymform import mask

_ST = {}
BG_RGB = np.array([35.0, 39.0, 43.0], np.float32)
BG_L = float(0.299 * 35 + 0.587 * 39 + 0.114 * 43)
TRACK_J = ("sh_r", "sh_l", "elb_r", "elb_l", "hip_r", "hip_l", "kn_r", "kn_l", "wr_r", "wr_l")


def on():
    return CFG.get("look") == "native_grade"


def load_slice(path, i0, i1):
    """d5.load_native_slice_720p without the 36-frame cap (Day 4 loader had none; the cap skipped poses)."""
    cap, frs = cv2.VideoCapture(path), []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] != 720 or fr.shape[0] != 1280:
            fr = cv2.resize(fr, (720, 1280), interpolation=cv2.INTER_LANCZOS4)
        frs.append(fr)
    cap.release()
    i1 = min(i1, len(frs) - 1)
    return frs[i0:i1 + 1] if i0 <= i1 else frs[i1:i0 + 1][::-1]


def _lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _lumx(a):
    """luminance with a global channel mix: the studio backdrop is violet (b,r above g by ~4-6) while the plates,
    bench and shorts are neutral; lum - k * smoothed violet chroma separates neutral dark equipment from an
    equal-luminance violet backdrop (one global 3x3-style mix, no mask; chroma smoothed sigma 2 so it adds no noise)."""
    a = a.astype(np.float32)
    k = float(CFG.get("native_chroma_k", 0.0))
    L = _lum(a)
    if k <= 0:
        return L
    c = 0.5 * (a[..., 0] + a[..., 2]) - a[..., 1]
    s = 2.0 * a.shape[0] / 1920.0
    return L - k * cv2.GaussianBlur(c, (0, 0), max(0.6, s))


def _act(a):
    return np.maximum(a[..., 0] - 0.5 * (a[..., 1] + a[..., 2]), a[..., 1] - 0.5 * (a[..., 0] + a[..., 2]))


def _flow_full(g_t, g_n):
    """flow at 540x960 such that frame_n(x + f) ~ frame_t(x), returned at 1080x1920 (pixels)."""
    f = cv2.calcOpticalFlowFarneback(g_t, g_n, None, 0.5, 4, 21, 4, 7, 1.5, 0)
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) * 2.0


_GX, _GY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def _warp(img, f):
    return cv2.remap(img, _GX + f[..., 0], _GY + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def prepare(step, seq, kps, y_sh):
    """Per-step precompute (once per worker): denoised 1080 frames, flat field, tone curve, glow scale, tracked kp."""
    if step in _ST:
        return _ST[step]
    raw, kp_t = [], []
    for i in range(len(seq)):
        rgb, kp = b4.transform_frame_and_kp_1080p(seq[i], kps[i], y_sh, step)
        raw.append(np.clip(rgb, 0, 255).astype(np.uint8)); kp_t.append(kp)
    n = len(raw)
    small = [cv2.resize(_lum(r.astype(np.float32)), (540, 960), interpolation=cv2.INTER_AREA).astype(np.uint8) for r in raw]
    tw = float(CFG.get("native_temporal_w", 0.25))
    den = []
    for t in range(n):
        ft = raw[t].astype(np.float32)
        acc, ws = ft * (1.0 - 2 * tw), np.full((H, W), 1.0 - 2 * tw, np.float32)
        lt = _lum(ft)
        for nb in (t - 1, t + 1):
            if 0 <= nb < n:
                wp = _warp(raw[nb].astype(np.float32), _flow_full(small[t], small[nb]))
                ok = np.clip(1.0 - (np.abs(_lum(wp) - lt) - 6.0) / 10.0, 0.0, 1.0) * tw   # occlusion guard
                acc += wp * ok[..., None]; ws += ok
            else:
                acc += ft * tw; ws += tw
        den.append(np.clip(acc / ws[..., None], 0, 255).astype(np.uint8))
    # flat field from the temporal median (quarter res)
    q = np.median(np.stack([cv2.resize(d, (270, 480), interpolation=cv2.INTER_AREA) for d in den[::max(1, n // 15)]]), axis=0).astype(np.float32)
    h_ = q.shape[0]
    bgc = np.median(np.concatenate([q[int(h_ * .36):int(h_ * .94), :4].reshape(-1, 3), q[int(h_ * .36):int(h_ * .94), -4:].reshape(-1, 3)]), axis=0)
    lq = _lumx(q)
    tex = cv2.GaussianBlur(np.abs(lq - cv2.GaussianBlur(lq, (0, 0), 1.5)), (0, 0), 2.0)
    # the canvas outside the scaled clip is a synthetic border-median fill, not clip content: never a field sample
    from gymform import ctx as _ctx
    x0, y0, nw, nh = _ctx.clip_rect(step)
    ins = np.zeros((H, W), np.float32)
    ins[max(0, y0):max(0, y0 + nh), max(0, x0):max(0, x0 + nw)] = 1.0
    ins_q = cv2.resize(cv2.erode(ins, np.ones((81, 81), np.uint8)), (270, 480), interpolation=cv2.INTER_NEAREST) > 0.5
    # inside weight: 0 outside the clip, 1 from ~110 px inside its edge (smooth ramp; the fill becomes the flat backdrop)
    wm = cv2.GaussianBlur(cv2.erode(ins, np.ones((131, 131), np.uint8)), (0, 0), 28.0)
    # per-row backdrop reference from the clip's own left/right edge columns (vignette changes with height)
    cols = np.where(ins_q.any(axis=0))[0]
    if len(cols) > 20:
        edge = np.concatenate([lq[:, cols[:8]], lq[:, cols[-8:]]], axis=1)
        prof = np.array([np.median(edge[y][ins_q[y, np.concatenate([cols[:8], cols[-8:]])]]) if ins_q[y].any() else np.nan
                         for y in range(h_)])
        good = ~np.isnan(prof)
        prof = np.interp(np.arange(h_), np.where(good)[0], prof[good]) if good.any() else np.full(h_, _lum(bgc))
        prof = cv2.GaussianBlur(prof.astype(np.float32)[:, None], (0, 0), 6.0)[:, 0]
    else:
        prof = np.full(h_, float(_lum(bgc)), np.float32)
    cand = ((np.abs(lq - prof[:, None]) < 10.0) & (tex < 4.0) & (_act(q) < 6.0) & ins_q).astype(np.float32)
    s_ = 14.0
    B = cv2.GaussianBlur(lq * cand, (0, 0), s_) / np.maximum(cv2.GaussianBlur(cand, (0, 0), s_), 1e-3)
    B0 = float(np.median(lq[cand > 0])) if cand.sum() > 100 else float(_lum(bgc))
    G = cv2.resize(np.clip(B0 / np.maximum(B, 1.0), 0.5, 2.0).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    # STATIC backdrop weight: pixels that are backdrop in EVERY sampled frame of the step (never touched by the body,
    # bench or dumbbells anywhere in its swept volume) take the flat studio colour; one weight per step -> zero flicker,
    # never a per-frame cut-out. Everything the athlete or equipment ever covers keeps the clip's own pixels.
    allb = np.ones_like(lq, bool)
    for d in den[::max(1, n // 24)]:
        qd = cv2.resize(d, (270, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
        ld = _lumx(qd)
        td = cv2.GaussianBlur(np.abs(ld - cv2.GaussianBlur(ld, (0, 0), 1.5)), (0, 0), 2.0)
        # backdrop in every frame: at the local level, or a soft static contact shadow (<= 14 darker, no texture)
        allb &= (ld - B < 6.0) & (ld - B > -14.0) & (td < 4.0) & (_act(qd) < 6.0)
    # outside the scaled clip is synthetic fill; the clip's own vignetted rim follows the local level B
    ins_qr = cv2.resize(ins, (270, 480), interpolation=cv2.INTER_NEAREST) > 0.5
    allb |= ~ins_qr
    allb = cv2.erode(allb.astype(np.uint8), np.ones((7, 7), np.uint8))
    wb = cv2.resize(cv2.GaussianBlur(allb.astype(np.float32), (0, 0), 5.0), (W, H), interpolation=cv2.INTER_CUBIC)
    wb = np.clip((wb - 0.15) / 0.7, 0.0, 1.0)
    bench_q = shadow_q = None
    if CFG.get("native_bench_map"):
        # t609: STATIC bench map from clearly bench-coloured pixels (palette A warm grey: r - b > 4, brighter than the
        # backdrop by > 15, saturation < 30 = never glow) that are stable over the step (|frame - median| < 12 in >= 85% of
        # the sampled frames); large components only. Used for the QC static layer AND the flat field (bench pixels are
        # never flat-field samples and never take the flat backdrop colour).
        qq = q.astype(np.float32)
        warm = ((qq[..., 0] - qq[..., 2]) > 4.0) & (_lum(qq) > B0 + 15.0) & ((qq.max(-1) - qq.min(-1)) < 30.0)   # glow is sat > 30
        stab = np.zeros(lq.shape, np.float32); n_s = 0
        for d in den[::max(1, n // 24)]:
            qd = cv2.resize(d, (270, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
            stab += (np.abs(qd - qq).max(-1) < 12.0); n_s += 1
        bq = (warm & (stab >= 0.85 * n_s)).astype(np.uint8)
        bq = cv2.morphologyEx(bq, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        nc, lab_, st_, _ = cv2.connectedComponentsWithStats(bq)
        bq = np.isin(lab_, [i for i in range(1, nc) if st_[i, cv2.CC_STAT_AREA] >= 150]).astype(np.uint8)
        # the bench's own static floor/contact shadow (darker than the backdrop, neutral, stable, outside every
        # sampled skeleton capsule, touching the bench or on the floor below the hips) belongs to the static bench layer, never to the skin
        cap = np.zeros(lq.shape, np.uint8)
        bones_ = (("sh_r", "elb_r"), ("elb_r", "wr_r"), ("sh_l", "elb_l"), ("elb_l", "wr_l"), ("sh_r", "sh_l"), ("sh_r", "hip_r"),
                  ("sh_l", "hip_l"), ("hip_r", "hip_l"), ("hip_r", "kn_r"), ("hip_l", "kn_l"), ("kn_r", "ank_r"), ("kn_l", "ank_l"))
        for kq in kp_t[::max(1, n // 12)]:
            for j1, j2 in bones_:
                if j1 in kq and j2 in kq:
                    cv2.line(cap, tuple(int(c / 4) for c in kq[j1]), tuple(int(c / 4) for c in kq[j2]), 1, 36)   # ~144 px: a whole thigh
        # (no texture test: thin crisp bench shadows have edge texture at quarter res; the body is excluded by the
        # capsule, and inside it only very dark (< B0 - 12) static gaps that touch the bench count - skin creases
        # touching the bench are lit skin (> 40))
        dk_ = (lq < B0 - 6.0) & ((qq.max(-1) - qq.min(-1)) < 12.0) & (stab >= 0.85 * n_s)
        bnd = cv2.dilate(bq, np.ones((5, 5), np.uint8)) > 0
        # + static near-black occlusion gaps anywhere (source lum < B0 - 20 ~ 16, all step long: the gap between back and
        # pad, under the chin, the knee pit - no lit surface, so never skin; a shaded skin side is 25-45, not < 16)
        shd = (dk_ & ((cap == 0) | ((lq < B0 - 12.0) & bnd) | (lq < B0 - 20.0))).astype(np.uint8)
        hip_y = float(np.median([0.5 * (k_["hip_r"][1] + k_["hip_l"][1]) for k_ in kp_t if "hip_r" in k_ and "hip_l" in k_])) / 4.0
        nc, lab_, st_, _ = cv2.connectedComponentsWithStats(shd)
        touch = set(np.unique(lab_[((cv2.dilate(bq, np.ones((5, 5), np.uint8)) > 0) | (lq < B0 - 20.0)) & (shd > 0)]).tolist()) - {0}
        floor = {i for i in range(1, nc) if st_[i, cv2.CC_STAT_TOP] > hip_y and st_[i, cv2.CC_STAT_AREA] >= 60}   # on the floor
        sq = np.isin(lab_, list(touch | floor)).astype(np.uint8)
        # penumbra: the shadow's soft rim (still darker than the backdrop), connected, outside the body
        grow = ((lq < B0 - 2.0) & (cap == 0)).astype(np.uint8)
        for _ in range(8):   # geodesic: the penumbra connected to the shadow, <= 8 quarter-res px (32 px)
            sq = np.maximum(sq, cv2.dilate(sq, np.ones((3, 3), np.uint8)) & grow)
        bench_q = bq
        shadow_q = sq & (1 - bq)
        # flat field: bench + its shadow are never given the flat backdrop colour (the clip's own pixels stay)
        wb = wb * (1.0 - cv2.resize(cv2.dilate(np.maximum(bq, sq), np.ones((5, 5), np.uint8)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR))
    # tone curve from skin statistics on the flattened lum (sampled frames)
    vals, actv = [], []
    bones = (("sh_r", "elb_r"), ("elb_r", "wr_r"), ("sh_l", "elb_l"), ("elb_l", "wr_l"), ("sh_r", "sh_l"), ("sh_r", "hip_r"),
             ("sh_l", "hip_l"), ("hip_r", "hip_l"), ("hip_r", "kn_r"), ("hip_l", "kn_l"), ("kn_r", "ank_r"), ("kn_l", "ank_l"))
    for i_ in range(0, n, max(1, n // 8)):
        f = den[i_].astype(np.float32)
        L1, a = _lumx(f) * G, _act(f)
        # skin population = pixels ON the skeleton (bench / dumbbell highlights excluded, as QC (f) excludes them)
        skel = np.zeros((H, W), np.uint8)
        kq = kp_t[i_]
        for j1, j2 in bones:
            if j1 in kq and j2 in kq:
                cv2.line(skel, tuple(int(c) for c in kq[j1]), tuple(int(c) for c in kq[j2]), 1, 70)
        for wj in ("wr_r", "wr_l"):
            if wj in kq:
                cv2.circle(skel, tuple(int(c) for c in kq[wj]), 75, 0, -1)
        sk = (L1 > B0 + 10.0) & (a < 12.0) & (skel > 0)
        sk[:620] = False
        vals.append(L1[sk][::7]); actv.append(a[(a > 15.0)][::5])
    v = np.concatenate(vals)
    lk = SPEC["mannequin"]["skin_lock_all_steps"]

    def _curve(qs, ys_in):
        sdx = min(float(CFG.get("native_sep_dx", 0.0)), B0 - 2.0)
        xs_ = [0.0] + ([B0 - sdx] if sdx > 0 else []) + [B0, B0 + float(CFG.get("native_steel_dx", 0.0) or 0.5)] + qs + [max(255.0, qs[-1] + 2.0)]
        for i in range(1, len(xs_)):
            xs_[i] = max(xs_[i], xs_[i - 1] + 1.0)
        return xs_, ys_in
    # darker than the backdrop (shorts, bench, metal): one linear segment floor..backdrop (slope < 1, so backdrop
    # noise is never amplified - a steep knot at the backdrop turned sensor noise into boiling texture); above: skin
    # anchors (pass 1: all skin, pass 2: the QC/Day 4 skin population above lum 46, as (f) measures it)
    # t601 separation: neutral mid-dark equipment just above the backdrop level (dumbbell plates) is lifted to
    # brushed steel (BG + native_steel_lift at B0 + native_steel_dx), skin above p5 untouched (Day 4 tone)
    lo = [float(CFG.get("native_dark_floor", 24.0))] + ([BG_L - float(CFG.get("native_sep_dy", 0.0))] if float(CFG.get("native_sep_dx", 0.0)) > 0 else []) \
        + [BG_L, BG_L + float(CFG.get("native_steel_lift", 0.0))]
    q1 = [float(np.percentile(v, p)) for p in (5, 50, 75, 90, 99.9)]
    ys1 = lo + [float(CFG.get("native_skin_p5", 52.0)), float(lk["p50"]), float(lk["p75"]), float(CFG.get("native_skin_p90", lk["p90"])), float(lk["max"]), float(lk["max"]) + 6.0]
    xs, ys = _curve(q1, ys1)
    lv = np.interp(v, xs, ys)
    v2 = v[lv > 46.0]
    if v2.size > 2000:
        q2 = [q1[0], q1[1]] + [float(np.percentile(v2, p)) for p in (75, 90, 99.9)]
        q2[2] = max(q2[2], q2[1] + 2.0)
        xs, ys = _curve(q2, ys1)
    a_all = np.concatenate(actv) if actv else np.zeros(1)
    A95 = float(np.percentile(a_all, 95)) if a_all.size > 500 else 60.0
    # callout joints locked to the skin (LK through the source order, clamped to 12 px of the kp)
    g8 = [cv2.cvtColor(d, cv2.COLOR_RGB2GRAY) for d in den]
    trk = [dict(kp_t[0])]
    pts = np.array([kp_t[0][j] for j in TRACK_J], np.float32).reshape(-1, 1, 2)
    R = float(CFG.get("anchor_lock_px", 12.0))
    for t in range(1, n):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(g8[t - 1], g8[t], pts, None, winSize=(41, 41), maxLevel=3)
        out = dict(kp_t[t])
        newp = []
        for k_, j in enumerate(TRACK_J):
            kp_ = np.array(kp_t[t][j], np.float32)
            p_ = nxt[k_, 0] if st[k_, 0] else kp_
            d_ = p_ - kp_; dn = float(np.hypot(*d_))
            if dn > R:
                p_ = kp_ + d_ * (R / dn)
            out[j] = (float(p_[0]), float(p_[1])); newp.append(p_)
        out["sh"] = (0.5 * (out["sh_r"][0] + out["sh_l"][0]), 0.5 * (out["sh_r"][1] + out["sh_l"][1]))
        trk.append(out)
        pts = np.array(newp, np.float32).reshape(-1, 1, 2)
    sg_ = float(CFG.get("native_anchor_smooth", 0.0))
    if sg_ > 0 and n > 4:
        # t609: callout anchor jitter (boil hotspot under the anchor dot / laser): temporal Gaussian over the SOURCE
        # order (the tempo replays source frames, so the smoothed track is replayed identically), reflect at the ends
        r_ = int(3 * sg_ + 0.5)
        kern = np.exp(-0.5 * (np.arange(-r_, r_ + 1) / sg_) ** 2); kern /= kern.sum()
        for j in TRACK_J:
            arr = np.array([trk[t][j] for t in range(n)], np.float32)
            pad = np.pad(arr, ((r_, r_), (0, 0)), mode="reflect")
            sm_ = np.stack([np.convolve(pad[:, c], kern, mode="valid") for c in range(2)], -1)
            for t in range(n):
                trk[t][j] = (float(sm_[t, 0]), float(sm_[t, 1]))
        for t in range(n):
            trk[t]["sh"] = (0.5 * (trk[t]["sh_r"][0] + trk[t]["sh_l"][0]), 0.5 * (trk[t]["sh_r"][1] + trk[t]["sh_l"][1]))
    _ST[step] = dict(den=den, G=G, B0=B0, wm=wm * (1.0 - wb), xs=xs, ys=ys, A95=A95, kp=trk, raw_kp=kp_t)
    if bench_q is not None:
        _ST[step]["bench"] = cv2.resize(bench_q.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        _ST[step]["shadow"] = cv2.resize(shadow_q.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        _ST[step]["bench_med"] = cv2.resize(q, (W, H), interpolation=cv2.INTER_LINEAR)
    from gymform import plates as _pl   # t605 plate recolour (flagged): soft tracked matte per source frame
    _ST[step]["plate"] = _pl.build(_ST[step], _lumx, _act, _flow_full, _warp, small)
    return _ST[step]


def grade(step, p_i, is_wrong):
    """-> (base_rgb uint8, shorts_mask, w_fg, src_rgb, kp_locked)."""
    S = _ST[step]
    src = S["den"][p_i]
    f = src.astype(np.float32)
    L1 = _lumx(f) * S["G"]
    Lg = np.interp(L1, S["xs"], S["ys"]).astype(np.float32)
    if CFG.get("native_highlight_rolloff"):
        # t609: soft knee above the skin p99 (output ~lk max - 10): highlights compress smoothly to <= lk max,
        # p75 / p90 untouched (they sit far below the knee)
        M_ = float(SPEC["mannequin"]["skin_lock_all_steps"]["max"])
        K_ = M_ - 10.0
        hi = Lg > K_
        Lg[hi] = K_ + 10.0 * np.tanh((Lg[hi] - K_) / 10.0)
    lk = SPEC["mannequin"]["skin_lock_all_steps"]
    t_sk = np.asarray(lk.get("tint_rgb", [-1.5, 0.5, 3.0]), np.float32)
    t_bg = BG_RGB - BG_L
    tb = np.clip((Lg - 45.0) / 25.0, 0.0, 1.0)[..., None]
    grey = Lg[..., None] + t_bg * (1.0 - tb) + t_sk * tb
    wt = np.asarray(CFG.get("native_dark_warm_rgb", [0.0, 0.0, 0.0]), np.float32)
    if wt.any():   # t601: darker-than-backdrop neutrals (bench, frame) read warm grey, never black or backdrop-cool
        tw_ = np.clip((BG_L - 2.0 - Lg) / 8.0, 0.0, 1.0)[..., None]
        grey = grey + (wt - t_bg) * tw_
    a = _act(f)
    if CFG.get("native_glow_hue_strict"):
        # t609: the glow is red OR green only; gold/yellow (both r and g over b: the plate rim) is never glow
        r_, g_, b_ = f[..., 0], f[..., 1], f[..., 2]
        if CFG.get("native_glow_form_hue"):
            # t612 D19: the step's OWN glow hue only (wrong = red dominance, right = green dominance). With max(red,
            # green) a warm-grey machine highlight (r - max(g,b) ~ 15) was recoloured GREEN on the right step.
            dom = (r_ - np.maximum(g_, b_)) if is_wrong else (g_ - np.maximum(r_, b_))
        else:
            dom = np.maximum(r_ - np.maximum(g_, b_), g_ - np.maximum(r_, b_))
        a = np.maximum(dom - 0.8 * np.maximum(0.0, np.minimum(r_, g_) - b_ - 20.0), 0.0)
    w = (np.clip((a - 8.0) / max(S["A95"] - 8.0, 20.0), 0.0, 1.0) * float(CFG.get("native_glow_max", 0.9)))[..., None]
    if is_wrong:
        col = np.stack([40.0 + 1.55 * Lg, 6.0 + 0.38 * Lg, 8.0 + 0.40 * Lg], -1)
    else:
        col = np.stack([8.0 + 0.32 * Lg, 30.0 + 1.38 * Lg, 14.0 + 0.72 * Lg], -1)
    kc = float(CFG.get("native_keep_chroma", 0.0))
    if kc > 0:
        # t609 palette A in the source: saturated NON-glow colours (cobalt plates, gold rim) keep their own chroma,
        # scaled to the graded brightness (global colour function, no mask); near-neutral skin / backdrop untouched
        c_src = f - _lum(f)[..., None]
        sat = np.linalg.norm(c_src, axis=-1)
        kw = (np.clip((sat - float(CFG.get("native_keep_chroma_sat", 8.0))) / 6.0, 0.0, 1.0) * kc)[..., None] * (1.0 - w)
        grey = grey + c_src * (Lg / np.maximum(_lum(f), 4.0))[..., None].clip(0.6, 2.0) * kw
    base = grey * (1.0 - w) + np.clip(col, 0, 255) * w
    pm = S["plate"][p_i] if S.get("plate") is not None else None
    if pm is not None:   # t605: plate colour remap (soft matte, the glow never sits on plates: act-gated matte)
        from gymform import plates as _pl
        base = _pl.apply(base, Lg, pm)
    wm = S["wm"][..., None]
    ref = np.clip(grey * wm + BG_RGB * (1.0 - wm), 0, 255).astype(np.uint8)   # global tone map only (QC (m) reference)
    base = np.clip(base * wm + BG_RGB * (1.0 - wm), 0, 255).astype(np.uint8)
    bf = base.astype(np.float32)
    if CFG.get("native_wfg_colour"):
        # t609: palette A cobalt plates have NEGATIVE red/green dominance (act < 0) -> the lum+act matte read them as
        # backdrop; foreground = lum distance + max(act, chroma distance from the backdrop tint)
        # (act >= 0: identical to the legacy |dL| + act; only a NEGATIVE act (blue/cyan) is replaced by the chroma term)
        chd = np.linalg.norm((bf - _lum(bf)[..., None]) - (BG_RGB - BG_L), axis=-1)
        ac_ = _act(bf)
        e_ = np.abs(_lum(bf) - BG_L) + np.where(ac_ >= 0.0, 0.95 * ac_, np.maximum(0.95 * ac_, 0.8 * chd - 6.0))
    else:
        e_ = np.abs(_lum(bf) - BG_L) + 0.95 * _act(bf)
    w_fg = cv2.GaussianBlur(np.clip((e_ - 6.0) / 14.0, 0.0, 1.0).astype(np.float32), (0, 0), 2.0)
    kp = S["kp"][p_i]
    F = mask.features(f)
    fab, _zone = mask.native_shorts(F, kp)
    # foreground evidence on the smoothed flattened lum (fabric sits ~6 under the backdrop; pixel noise ~3)
    if CFG.get("shorts_evidence_strict"):   # t609: fabric evidence is DARKER than the backdrop (bright thigh skin never shorts)
        evid = ((S["B0"] - cv2.GaussianBlur(L1, (0, 0), 4.0)) > float(CFG.get("shorts_evidence_lum", 3.5))).astype(np.uint8)
    else:
        evid = (np.abs(cv2.GaussianBlur(L1, (0, 0), 4.0) - S["B0"]) > float(CFG.get("shorts_evidence_lum", 3.5))).astype(np.uint8)
    sm = cv2.morphologyEx((fab > 0.5).astype(np.uint8) & evid, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) & evid
    sm_ev = sm.copy()   # Day 4 definition (foreground-evidence mask): QC (f) shorts median
    if CFG.get("shorts_evidence_strict"):
        # t609 (Day 4 x w_fg idea): fabric must be darker than the local backdrop by evidence on the UNBLURRED lum and
        # carry no backdrop texture-free flatness at its rim -> 3 px inner erosion of the evidence before the AND
        ev2 = ((S["B0"] - L1) > float(CFG.get("shorts_evidence_lum", 3.5)) + 2.0).astype(np.uint8)
        ev2 = cv2.erode(cv2.morphologyEx(ev2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((5, 5), np.uint8))
        sm = sm & cv2.dilate(ev2, np.ones((3, 3), np.uint8))
        # no shorts treatment on pixels without foreground evidence: backdrop-level (4 px smoothed, within 6 of the
        # local level) AND texture-free pixels are backdrop, never shorts
        ls_ = cv2.GaussianBlur(L1, (0, 0), 4.0)
        tx_ = cv2.GaussianBlur(np.abs(L1 - cv2.GaussianBlur(L1, (0, 0), 2.0)), (0, 0), 2.0)
        sm = sm & ((np.abs(ls_ - S["B0"]) >= 6.0) | (tx_ >= 4.0)).astype(np.uint8)
    shorts = cv2.GaussianBlur(sm.astype(np.float32), (0, 0), 0.8)
    bench = benchc = None
    if S.get("bench") is not None:   # static bench where the frame currently shows it (an arm in front wins)
        shows = np.abs(f - S["bench_med"]).max(-1) < 12.0
        bench = np.maximum(S["bench"], S["shadow"]) * shows   # static layer = bench + its floor shadow
        benchc = S["bench"] * shows                            # bench body only (QC (sep) colour)
    LAST_MASKS["native"] = dict(L1=L1, B0=S["B0"], ref=ref, plate=pm if pm is not None else np.zeros((H, W), np.float32),
                                shorts_ev=cv2.GaussianBlur(sm_ev.astype(np.float32), (0, 0), 0.8), bench=bench, benchc=benchc)
    return base, shorts, w_fg, src, kp
