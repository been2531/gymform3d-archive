"""t605 plate-only recolour (config "plate_recolour": {...}; off when absent -> older days unchanged).

Custom-coloured dumbbell plates that never melt into the backdrop. NOT a cut-out:
  * matte = soft (feathered) plate evidence near the LK-tracked wrists: darker than the local backdrop, no glow,
    not the step's static bench (temporal median), thin handle removed (opening), plate faces closed in;
  * temporally stable: flow-warped EMA forward AND backward through the source order, averaged (no lag, no pop);
  * applied as a colour/luminance REMAP of the graded clip pixel: out = colour * (graded lum / plate median lum),
    so the clip's own shading and texture stay; the remap weight is the soft matte (no holes, no hard edge);
  * hands / forearms are never recoloured: pixels at skin brightness carry zero weight (skin key on raw lum);
  * optional rim colour on the matte's outer band, optional accent on its brightest faces (parameterised).
Parameters (all optional): base_rgb, rim_rgb, rim_px, accent_rgb, accent_q, radius_px, ema, feather_px.
"""
import cv2
import numpy as np

from gymform.ctx import CFG, W, H

DEF = dict(base_rgb=[46, 88, 196], rim_rgb=[214, 200, 168], rim_px=5, accent_rgb=None, accent_q=0.92,
           radius_px=150, ema=0.45, feather_px=2.5, dark_dl=5.0, close_px=13, open_px=7)


def cfg():
    c = CFG.get("plate_recolour")
    if not c:
        return None
    out = dict(DEF)
    out.update(c if isinstance(c, dict) else {})
    return out


def _raw_cand(L1, act, B0, kp, med_l, skin_l, P):
    # plates sit on the handle axis (perpendicular to the forearm through the fist): one disc per plate
    near = np.zeros((H, W), np.uint8)
    for s_ in ("r", "l"):
        wr, el = kp.get("wr_" + s_), kp.get("elb_" + s_)
        if wr is None or el is None:
            continue
        f = np.array(wr, np.float32) - np.array(el, np.float32)
        f /= max(1.0, float(np.hypot(*f)))
        p = np.array([-f[1], f[0]], np.float32)
        c0 = np.array(wr, np.float32) + f * float(P.get("grip_dy", 12.0))
        for sg in (-1.0, 1.0):
            q = c0 + sg * p * float(P.get("plate_off", 72.0))
            cv2.circle(near, (int(q[0]), int(q[1])), int(P.get("plate_r", 64.0)), 1, -1)
    Lb = cv2.GaussianBlur(L1, (0, 0), 1.2)
    dark = Lb < (B0 - float(P["dark_dl"]))
    moving = np.abs(Lb - med_l) > 6.0            # the static bench / swept backdrop equal the step's temporal median
    mid = Lb < (skin_l - 2.0)                     # plate faces up to just under skin brightness
    c = (near > 0) & (act < 10.0) & mid & (moving | dark)
    c = cv2.morphologyEx(c.astype(np.uint8), cv2.MORPH_OPEN, np.ones((int(P["open_px"]),) * 2, np.uint8))
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, np.ones((int(P["close_px"]),) * 2, np.uint8))
    # each plate is a convex hex: hull per blob fills its lit faces; blobs of plate size only (handle / specks out)
    n, lab, st, _ = cv2.connectedComponentsWithStats(c, connectivity=8)
    out = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 900:
            continue
        ys, xs = np.where(lab == i)
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
        cv2.fillConvexPoly(out, hull, 1)
    out &= (near > 0).astype(np.uint8)
    return out.astype(np.float32)


def build(S, lum_fn, act_fn, flow_fn, warp_fn, small):
    """-> list of soft plate mattes (float32 HxW, 0..1), one per source frame, temporally smoothed."""
    P = cfg()
    if P is None:
        return None
    den, G, B0, kps = S["den"], S["G"], S["B0"], S["kp"]
    n = len(den)
    Ls = [lum_fn(d.astype(np.float32)) * G for d in den]
    med_l = np.median(np.stack(Ls[::max(1, n // 15)]), axis=0).astype(np.float32)
    skin_l = float(S["xs"][list(S["ys"]).index(next(y for y in S["ys"] if y >= 52.0))]) if len(S["xs"]) else 60.0
    raw = []
    for t in range(n):
        m = _raw_cand(Ls[t], act_fn(den[t].astype(np.float32)), B0, kps[t], med_l, skin_l, P)
        raw.append(cv2.GaussianBlur(m, (0, 0), float(P["feather_px"])))
    a = float(P["ema"])
    fwd, bwd = [None] * n, [None] * n
    for t in range(n):
        fwd[t] = raw[t] if t == 0 else a * raw[t] + (1 - a) * warp_fn(fwd[t - 1], flow_fn(small[t], small[t - 1]))
    for t in range(n - 1, -1, -1):
        bwd[t] = raw[t] if t == n - 1 else a * raw[t] + (1 - a) * warp_fn(bwd[t + 1], flow_fn(small[t], small[t + 1]))
    out = []
    for t in range(n):
        m = 0.5 * (fwd[t] + bwd[t])
        # never on skin (hands / forearms): zero weight at skin brightness, soft
        m *= np.clip((skin_l + 6.0 - cv2.GaussianBlur(Ls[t], (0, 0), 1.5)) / 12.0, 0.0, 1.0)
        out.append(np.clip((m - 0.12) / 0.6, 0.0, 1.0).astype(np.float32))
    return out


def apply(base_f, Lg, m):
    """colour remap of the graded frame under the soft matte; shading = graded lum relative to the plate median."""
    P = cfg()
    if P is None or m is None or m.max() <= 0.01:
        return base_f
    sel = m > 0.6
    ref = float(np.median(Lg[sel])) if sel.sum() > 200 else 30.0
    sh = np.clip(Lg / max(ref, 8.0), 0.45, 1.8)[..., None]
    col = np.asarray(P["base_rgb"], np.float32)[None, None, :] * sh
    if P.get("accent_rgb") is not None and sel.sum() > 200:
        q = float(np.quantile(Lg[sel], float(P["accent_q"])))
        wa = np.clip((Lg - q) / 6.0, 0.0, 1.0)[..., None]
        col = col * (1 - wa) + np.asarray(P["accent_rgb"], np.float32) * sh * wa
    if P.get("rim_rgb") is not None and int(P["rim_px"]) > 0:
        inner = cv2.GaussianBlur(cv2.erode((m > 0.5).astype(np.uint8), np.ones((2 * int(P["rim_px"]) + 1,) * 2, np.uint8)).astype(np.float32), (0, 0), 1.0)
        wr = np.clip(m - inner, 0.0, 1.0)[..., None] * (m[..., None] > 0.35)
        col = col * (1 - wr) + np.asarray(P["rim_rgb"], np.float32)[None, None, :] * np.clip(sh, 0.7, 1.2) * wr
    mm = m[..., None]
    return base_f * (1 - mm) + np.clip(col, 0, 255) * mm
