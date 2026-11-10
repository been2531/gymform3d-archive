"""Power-grip ("fist around the handle") for a rig with ONE hand bone and no finger bones (Grab3D CC0 body, 24 bones).
Pure numpy, used inside Blender (retarget_render_v2.py) and by QC. 0 generations: the fist is a geometric shape key.

1. hand frame (rest, from the mesh, not the bone roll): f = hand bone axis (wrist -> fingers); the four fingers are the
   vertices past the web line (where the thumb cluster ends); across-palm axis a = line through the 4 finger
   centroids (index -> little finger order known from the thumb side); palm normal n from handedness (thumb side t:
   right hand n = t x f, left hand n = f x t).
2. handle: axis along a, at the finger base (y_k), pushed out from the palm/finger palmar surface by r + clearance.
3. fingers: a bend about the handle axis, arc length preserved along each finger's midline (radius interpolated across
   the palm) -> the four fingers wrap the handle; capped at max_wrap_deg.
4. thumb: rigid flexion about an axis through its base (parallel to a), until its palmar surface touches the handle
   from the wrist side (opposes the fingers, which close from the far side).
Labels per hand vertex: 0..3 fingers (little -> index), 4 thumb, 5 palm."""
import numpy as np

def _u(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def build(co_w, idx, bone_M, is_left, r=0.016, clear=0.002, max_wrap_deg=250.0, web_gap=0.004, mcp_back=0.015, thumb_ramp=0.02, seam_mm=3.0, seam_iter=4):
    """co_w: (N,3) world rest coords of the whole mesh; idx: hand vertex ids (weight > 0.5); bone_M: 4x4 world rest
    matrix of the hand bone (orthonormal). Returns dict with labels, deformed world rest coords for idx, handle (local)."""
    R, o = bone_M[:3, :3], bone_M[:3, 3]; L = (co_w[idx] - o) @ R          # bone-local (Y = finger direction)
    y = L[:, 1]; c0 = np.median(L, 0)
    P = L - c0; w, V = np.linalg.eigh(np.cov(P[:, [0, 2]].T)); a0 = np.array([V[0, 1], 0.0, V[1, 1]])   # wide axis (x/z plane)
    x0 = L @ a0
    # thumb = the side cluster in mid-hand slices (gap > web_gap along a0)
    ts, gaps = [], []
    for yy in np.arange(np.percentile(y, 30), np.percentile(y, 75), 0.01):
        s = (y >= yy) & (y < yy + 0.01); xs = np.sort(x0[s])
        g = np.nonzero(np.diff(xs) > web_gap)[0]
        if len(g): gaps.append((yy, (xs[g[0]] + xs[g[0] + 1]) / 2, (xs[g[-1]] + xs[g[-1] + 1]) / 2, xs.min(), xs.max()))
    if not gaps: raise RuntimeError("hand_grip: no thumb gap found")
    # thumb is on the side whose cluster is narrower
    left_w = np.median([g[1] - g[3] for g in gaps]); right_w = np.median([g[4] - g[2] for g in gaps])
    side = 1.0 if right_w < left_w else -1.0; gx = np.median([g[2] if side > 0 else g[1] for g in gaps])
    y_k = max(g[0] for g in gaps) + 0.01                                       # web line: thumb cluster ends
    thumb = (side * (x0 - gx) > 0) & (y > np.percentile(y, 15)) & (y < y_k)
    fing = (y >= y_k) & ~thumb
    # 4 fingers: 1-D k-means along a0
    xf = x0[fing]; c = np.quantile(xf, [0.125, 0.375, 0.625, 0.875])
    for _ in range(30):
        lab = np.argmin(np.abs(xf[:, None] - c[None]), 1); c = np.array([xf[lab == k].mean() if (lab == k).any() else c[k] for k in range(4)])
    tc = L[thumb].mean(0)
    order = np.argsort(np.abs(c - float(tc @ a0)))[::-1]                      # little (far from the thumb) -> index (next to it)
    fc = np.array([L[fing][lab == k].mean(0) for k in order])
    a = _u(fc[-1] - fc[0]); a[1] = 0.0; a = _u(a)                              # little -> index, perpendicular to f
    f = np.array([0.0, 1.0, 0.0])
    n = _u(np.cross(a, f))
    # palm side = the side the relaxed thumb sits on (it rests in front of the palm, opposed); handedness-free
    if float((tc - fc.mean(0)) @ n) < 0: n = -n
    ua, un = L @ a, L @ n
    labels = np.full(len(idx), 5); fl = np.zeros(fing.sum(), int)
    for rank, k in enumerate(order): fl[lab == k] = rank
    labels[np.nonzero(fing)[0]] = fl; labels[thumb] = 4
    # the knuckles (MCP joints) sit ~1.5 cm proximal to the web: the strip between them bends with the fingers
    y_b = y_k - mcp_back; fu0 = np.array([np.median(ua[labels == k]) for k in range(4)]); hw = 0.5 * float(np.median(np.diff(np.sort(fu0))))
    strip = (labels == 5) & (y >= y_b) & (ua >= fu0.min() - hw) & (ua <= fu0.max() + hw)
    labels[strip] = np.argmin(np.abs(ua[strip, None] - fu0[None]), 1); y_k = y_b
    # handle axis (along a, at the web line y_h): as close to the palm as possible with every non-thumb vertex at least
    # r + clear from the axis (fingers measured before bending: bending keeps their distance to the axis)
    y_h = y_k; rc = r + clear; nt = labels != 4
    dy = np.abs(y[nt] - y_h); cand = un[nt] + np.sqrt(np.maximum(rc * rc - dy * dy, 0.0)); cand[dy >= rc] = -1.0
    fb = (labels <= 3); n_h = max(float(cand.max()), float(un[fb].max()) + rc)
    fu = np.array([np.median(ua[labels == k]) for k in range(4)]); fmid = np.array([np.median(un[labels == k]) for k in range(4)])
    R0 = np.interp(ua, fu, n_h - fmid)                                         # midline wrap radius across the palm (info)
    # per finger: hug the handle (palmar surface pulled to r + clear along the finger, ramped in over the first 12 mm),
    # wrap angle = integral of ds / local midline radius (arc length kept along the midline)
    newL = L.copy(); F = labels <= 3; s_all = y - y_k; rho_all = n_h - un; yy_all = np.zeros(len(L)); nn_all = np.zeros(len(L))
    for k in range(4):
        m = labels == k; sk, rk = s_all[m], rho_all[m]; bins = np.arange(0, sk.max() + 0.004, 0.004)
        bi = np.clip(np.digitize(sk, bins) - 1, 0, len(bins) - 1)
        pal = np.array([rk[bi == b].min() if (bi == b).any() else np.nan for b in range(len(bins))])
        mid = np.array([np.median(rk[bi == b]) if (bi == b).any() else np.nan for b in range(len(bins))])
        ok = ~np.isnan(pal); pal = np.interp(bins, bins[ok], pal[ok]); mid = np.interp(bins, bins[ok], mid[ok])
        w_ = np.clip(bins / 0.012, 0, 1); sh = np.maximum(pal - rc, 0.0) * w_; Rm = np.maximum(mid - sh, rc + 0.003)
        phi_b = np.concatenate([[0.0], np.cumsum(np.diff(bins) / Rm[:-1])]); phi_b = np.minimum(phi_b, np.radians(max_wrap_deg))
        phi = np.interp(sk, bins, phi_b); rr = rk - np.interp(sk, bins, sh)
        yy_all[m] = y_k + rr * np.sin(phi); nn_all[m] = n_h - rr * np.cos(phi)
    yy_, nn_ = yy_all[F], nn_all[F]
    base = L[F] - np.outer(y[F], f) - np.outer(un[F], n)                       # keep the across-palm component
    newL[F] = base + np.outer(yy_, f) + np.outer(nn_, n)
    # thumb: rigid flexion about an axis through its base, parallel to a, toward the palm side until it touches
    T = labels == 4; tb = L[T][np.argsort(y[T])[: max(5, T.sum() // 20)]].mean(0)
    # flexion ramps in from the thenar seam (0 at the nearest palm vertex -> full over thumb_ramp): no torn seam edges
    LT, LP = L[T], L[labels == 5]
    dmin = np.min(np.linalg.norm(LT[:, None, :] - LP[None, ::4, :], axis=2), axis=1) if len(LP) else np.full(len(LT), 1.0)
    wT = np.clip(dmin / thumb_ramp, 0.0, 1.0); wT = wT * wT * (3 - 2 * wT)
    def rot(Pp, ang):
        k = a; v = Pp - tb; ang = ang * wT[:, None]
        return tb + v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + np.outer(v @ k, k) * (1 - np.cos(ang))
    hc_l = lambda Pp: Pp - (np.outer(Pp @ a, a))                               # projection onto the plane perpendicular to a
    H0 = y_h * f + n_h * n
    def gap(ang):
        Q = rot(L[T], ang); d = np.linalg.norm(hc_l(Q) - hc_l(H0[None])[0], axis=1); return float(d.min() - r)
    # pick the flexion that touches the handle (gap in [clear, 6 mm]) with the best opposition to the fingertips
    hcen = y_h * f + n_h * n + a * float(np.mean(fu)); sF = np.where(labels <= 3, y - y_k, -1.0); best, bg, bo = 0.0, gap(0.0), -1.0
    for ag in np.radians(np.arange(-90, 91, 1.0)):
        g_ = gap(ag)
        if not (clear <= g_ <= 0.006): continue
        Ltry = newL.copy(); Ltry[T] = rot(L[T], ag); mt = metrics(Ltry, labels, sF, hcen, a, r)
        if mt.get("thumb_opposition_deg", 0) > bo + 1e-6: best, bg, bo = ag, g_, mt["thumb_opposition_deg"]
    newL[T] = rot(L[T], best)
    # seams (finger web, knuckle strip): adjacent fingers bend by different radii, so blend the displacement over a few
    # mm across label changes (Gaussian, rest-space neighbours, thumb excluded: it has its own ramp)
    if seam_mm > 0:
        nt_i = np.nonzero(labels != 4)[0]; Pn = L[nt_i]; ln = labels[nt_i]; Dn = (newL - L)[nt_i]; h2 = (seam_mm / 1000.0) ** 2
        nb = []
        for c0 in range(0, len(Pn), 1024):
            d2 = ((Pn[c0:c0 + 1024, None, :] - Pn[None, :, :]) ** 2).sum(-1); nb.append([np.nonzero(row < 4 * h2)[0] for row in d2])
        nb = [x for ch in nb for x in ch]
        zone = np.array([len(set(ln[j].tolist())) > 1 for j in nb]); zi = np.nonzero(zone)[0]
        for _ in range(seam_iter):
            Dz = np.array([(np.exp(-((Pn[nb[i]] - Pn[i]) ** 2).sum(1) / h2)[:, None] * Dn[nb[i]]).sum(0) / np.exp(-((Pn[nb[i]] - Pn[i]) ** 2).sum(1) / h2).sum() for i in zi])
            if len(zi): Dn[zi] = Dz
        newL[nt_i] = L[nt_i] + Dn
    # Fade wrap/flexion to 0 at the wrist so the forearm seam does not ring.
    y_lo = float(np.percentile(y, 6))
    y_hi = y_lo + 0.060
    ramp = np.clip((y - y_lo) / max(y_hi - y_lo, 1e-4), 0.0, 1.0)
    newL = L + (newL - L) * ramp[:, None]
    newW = newL @ R.T + o
    handle_local = {"center": (y_h * f + n_h * n + a * float(np.mean(fu))).tolist(), "axis": a.tolist(), "n": n.tolist(), "r": r}
    info = {"y_k": round(float(y_k), 4), "n_h": round(float(n_h), 4), "R0_mm": [round(float(v) * 1000, 1) for v in (n_h - fmid)],
            "thumb_flex_deg": round(float(np.degrees(best)), 1), "thumb_gap_mm": round(bg * 1000, 1), "n_thumb": int(T.sum()), "n_fingers": int(F.sum())}
    return {"labels": labels, "new_world": newW, "handle_local": handle_local, "info": info, "y_local": y, "s_finger": np.where(F_full(labels), y - y_k, -1.0)}

def F_full(labels): return labels <= 3

def metrics(Pw, labels, s_fing, h_center, h_axis, r, h_half_len=0.085):
    """Pw: (N,3) deformed WORLD hand verts (same order as labels); handle centre/axis in world. Per finger: wrap angle of
    the tip around the handle relative to the knuckle, fingertip gap to the handle surface; thumb tip gap + angular
    separation from the fingertips; hand verts inside the handle cylinder (intersection)."""
    a = _u(np.asarray(h_axis, float)); c = np.asarray(h_center, float); V = Pw - c; t = V @ a; Rr = V - np.outer(t, a); d = np.linalg.norm(Rr, axis=1)
    ref = None; out = {"finger_wrap_deg": [], "fingertip_gap_mm": []}
    e1 = None
    for k in range(4):
        m = labels == k
        if not m.any(): out["finger_wrap_deg"].append(0.0); out["fingertip_gap_mm"].append(99.0); continue
        s = s_fing[m]; kn = Rr[m][s <= np.percentile(s, 8)].mean(0); tipm = s >= np.percentile(s, 92)
        if e1 is None: e1 = _u(kn)
        e2 = np.cross(a, e1)
        qs = np.percentile(s, np.linspace(4, 96, 24)); pts = [Rr[m][np.abs(s - q) <= 0.004].mean(0) for q in qs if (np.abs(s - q) <= 0.004).any()]
        an = np.unwrap(np.array([np.arctan2(v @ e2, v @ e1) for v in pts]))
        out["finger_wrap_deg"].append(round(float(abs(np.degrees(an[-1] - an[0]))), 1))
        out["fingertip_gap_mm"].append(round(float((d[m][tipm].min() - r) * 1000), 1))
    ft = [Rr[labels == k][s_fing[labels == k] >= np.percentile(s_fing[labels == k], 92)].mean(0) for k in range(4) if (labels == k).any()]
    T = labels == 4
    if T.any() and e1 is not None:
        e2 = np.cross(a, e1); ang = lambda v: np.degrees(np.arctan2(v @ e2, v @ e1))
        th_d = d[T]; tip = Rr[T][np.argsort(th_d)[:20]].mean(0)
        out["thumb_gap_mm"] = round(float((th_d.min() - r) * 1000), 1)
        sep = abs(((ang(tip) - ang(np.mean(ft, 0))) + 180) % 360 - 180); out["thumb_opposition_deg"] = round(float(sep), 1)
    inside = (d < r - 0.001) & (np.abs(t) < h_half_len); out["handle_intersect_verts"] = int(inside.sum())
    return out


def _label_hand(L, web_gap=0.004, mcp_back=0.015):
    """Label bone-local hand verts (Y along fingers). Returns labels, a, f, n, y, y_k."""
    y = L[:, 1]
    c0 = np.median(L, 0)
    P = L - c0
    _w, V = np.linalg.eigh(np.cov(P[:, [0, 2]].T))
    a0 = np.array([V[0, 1], 0.0, V[1, 1]])
    x0 = L @ a0
    gaps = []
    for yy in np.arange(np.percentile(y, 30), np.percentile(y, 75), 0.01):
        s = (y >= yy) & (y < yy + 0.01)
        xs = np.sort(x0[s])
        g = np.nonzero(np.diff(xs) > web_gap)[0]
        if len(g):
            gaps.append((yy, (xs[g[0]] + xs[g[0] + 1]) / 2, (xs[g[-1]] + xs[g[-1] + 1]) / 2,
                         xs.min(), xs.max()))
    if not gaps:
        raise RuntimeError("hand_grip: no thumb gap found")
    left_w = np.median([g[1] - g[3] for g in gaps])
    right_w = np.median([g[4] - g[2] for g in gaps])
    side = 1.0 if right_w < left_w else -1.0
    gx = np.median([g[2] if side > 0 else g[1] for g in gaps])
    y_k = max(g[0] for g in gaps) + 0.01
    thumb = (side * (x0 - gx) > 0) & (y > np.percentile(y, 15)) & (y < y_k)
    fing = (y >= y_k) & ~thumb
    xf = x0[fing]
    c = np.quantile(xf, [0.125, 0.375, 0.625, 0.875])
    for _ in range(30):
        lab = np.argmin(np.abs(xf[:, None] - c[None]), 1)
        c = np.array([xf[lab == k].mean() if (lab == k).any() else c[k] for k in range(4)])
    tc = L[thumb].mean(0)
    order = np.argsort(np.abs(c - float(tc @ a0)))[::-1]
    fc = np.array([L[fing][lab == k].mean(0) for k in order])
    a = _u(fc[-1] - fc[0]); a[1] = 0.0; a = _u(a)
    f = np.array([0.0, 1.0, 0.0])
    n = _u(np.cross(a, f))
    if float((tc - fc.mean(0)) @ n) < 0:
        n = -n
    ua = L @ a
    labels = np.full(len(L), 5)
    fl = np.zeros(fing.sum(), int)
    for rank, k in enumerate(order):
        fl[lab == k] = rank
    labels[np.nonzero(fing)[0]] = fl
    labels[thumb] = 4
    fu0 = np.array([np.median(ua[labels == k]) for k in range(4)])
    hw = 0.5 * float(np.median(np.diff(np.sort(fu0))))
    y_b = y_k - mcp_back
    strip = (labels == 5) & (y >= y_b) & (ua >= fu0.min() - hw) & (ua <= fu0.max() + hw)
    labels[strip] = np.argmin(np.abs(ua[strip, None] - fu0[None]), 1)
    y_k = y_b
    return labels, a, f, n, y, y_k


def flatten_palm(co_w, idx, bone_M, is_left=True, adduct=0.62, thumb_adduct=0.40, flatten=0.50):
    """Bring splayed fingers together and flatten the palm onto a pad (no finger bones).

    adduct: fraction of each finger's across-palm offset removed toward the four-finger mean.
    flatten: fraction of palm-normal variation removed so the palm reads planar.
    Returns dict with new_world (len idx), labels, finger_spread_m, palm_normal_world, info.
    """
    R, o = bone_M[:3, :3], bone_M[:3, 3]
    L = (co_w[idx] - o) @ R
    labels, a, f, n, y, y_k = _label_hand(L)
    ua, un = L @ a, L @ n
    newL = L.copy()
    fing = labels <= 3
    ua_mid = float(np.median(ua[fing])) if fing.any() else float(np.median(ua))
    # Adduction: slide each finger toward the palm midline along the across-palm axis.
    for k in range(4):
        m = labels == k
        if m.any():
            newL[m] = newL[m] - np.outer(adduct * (ua[m] - ua_mid), a)
    m = labels == 4
    if m.any():
        newL[m] = newL[m] - np.outer(thumb_adduct * (ua[m] - ua_mid), a)
    # Flatten: pull palm and fingers toward a common palmar plane.
    palm_un = float(np.median(un[labels == 5])) if (labels == 5).any() else float(np.median(un))
    newL = newL - np.outer(flatten * (un - palm_un), n)
    # Fade the delta out at the wrist so the hand/forearm seam does not ring.
    y_lo = float(np.percentile(y, 8))
    ramp = np.clip((y - y_lo) / max(float(y_k) - y_lo, 1e-4), 0.0, 1.0)
    newL = L + (newL - L) * ramp[:, None]
    new_world = newL @ R.T + o
    meas = measure_palm_local(newL, labels, a, n, y)
    n_w = R @ n
    info = {
        "adduct": round(float(adduct), 3),
        "flatten": round(float(flatten), 3),
        "finger_spread_m": meas["finger_spread_m"],
        "n_verts": int(len(idx)),
        "is_left": bool(is_left),
    }
    return {
        "new_world": new_world,
        "labels": labels,
        "finger_spread_m": meas["finger_spread_m"],
        "palm_normal_world": n_w,
        "info": info,
    }


def measure_palm_local(L, labels, a, n, y):
    tips = []
    for k in range(4):
        m = labels == k
        if not m.any():
            continue
        sel = L[m]
        ys = y[m]
        hi = ys >= np.percentile(ys, 88)
        tips.append(sel[hi].mean(0) if hi.any() else sel[np.argmax(ys)])
    spread = float(np.linalg.norm(tips[0] - tips[-1])) if len(tips) >= 2 else 0.0
    return {"finger_spread_m": round(spread, 4)}


def measure_palm(co_w, idx, bone_M, pad_normal=None):
    """Measure posed-hand finger spread and how flat the palm sits on the pad."""
    if pad_normal is None:
        pad_normal = np.array([0.0, 0.0, 1.0])
    R, o = bone_M[:3, :3], bone_M[:3, 3]
    L = (co_w[idx] - o) @ R
    labels, a, f, n, y, y_k = _label_hand(L)
    meas = measure_palm_local(L, labels, a, n, y)
    n_w = R @ n
    pn = pad_normal / (np.linalg.norm(pad_normal) + 1e-12)
    # Palmar normal should face into the pad (−Z on a horizontal bench).
    into_pad = -pn
    ang = float(np.degrees(np.arccos(np.clip(float(n_w @ into_pad), -1.0, 1.0))))
    ang = min(ang, 180.0 - ang)
    return {
        "finger_spread_m": meas["finger_spread_m"],
        "palm_from_pad_deg": round(ang, 2),
        "palm_normal_world": n_w,
        "labels": labels,
    }

