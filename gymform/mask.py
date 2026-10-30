"""LOCKED clean foreground mask: anatomical envelope, glow-robust key, opaque matte-black shorts."""
import math
import cv2
import numpy as np
from gymform.ctx import *  # shared state: W,H,CFG,LAST_MASKS,CUR_PLATE,STEP_SCALE,_XX,_YY,d5,b4,p62,v30
from gymform.equipment import equipment_soft_zone


def _seg_dist(p0, p1, x0, y0, x1, y1):
    """Distance field to segment p0-p1 restricted to ROI [x0:x1, y0:y1]."""
    xx = _XX[y0:y1, x0:x1]
    yy = _YY[y0:y1, x0:x1]
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay
    L2 = max(dx * dx + dy * dy, 1e-6)
    t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(xx - (ax + t * dx), yy - (ay + t * dy))


def _ext(p0, p1, t):
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def anatomical_envelope(kp: dict) -> np.ndarray:
    sh_r, sh_l = kp["sh_r"], kp["sh_l"]
    hip_r, hip_l = kp["hip_r"], kp["hip_l"]
    sh_m = (0.5 * (sh_r[0] + sh_l[0]), 0.5 * (sh_r[1] + sh_l[1]))
    hip_m = (0.5 * (hip_r[0] + hip_l[0]), 0.5 * (hip_r[1] + hip_l[1]))
    torso_len = math.hypot(sh_m[0] - hip_m[0], sh_m[1] - hip_m[1])
    head = (sh_m[0] + (sh_m[0] - hip_m[0]) * 0.42, sh_m[1] - 0.42 * torso_len)

    segs = [  # (p0, p1, normaliser)
        (head, hip_m, 185.0),
        (sh_r, sh_l, 185.0 * 0.62),
        (hip_r, hip_l, 185.0 * 0.62),
    ]
    for side in ("r", "l"):
        sh, el, wr = kp[f"sh_{side}"], kp[f"elb_{side}"], kp[f"wr_{side}"]
        hip, kn, ank = kp[f"hip_{side}"], kp[f"kn_{side}"], kp[f"ank_{side}"]
        segs.append((sh, el, 92.0))                    # upper arm
        segs.append((el, _ext(el, wr, 1.30), 120.0))   # forearm incl. hand + dumbbell
        segs.append((hip, kn, 108.0))                  # thigh
        segs.append((kn, _ext(kn, ank, 1.12), 95.0))   # shin + foot

    pts = np.array([p for s in segs for p in s[:2]], dtype=np.float32)
    pad = int(1.35 * 185.0) + 8
    x0 = int(max(0, pts[:, 0].min() - pad)); x1 = int(min(W, pts[:, 0].max() + pad))
    y0 = int(max(0, pts[:, 1].min() - pad)); y1 = int(min(H, pts[:, 1].max() + pad))
    dmin = np.full((y1 - y0, x1 - x0), 99.0, dtype=np.float32)
    for p0, p1, r in segs:
        np.minimum(dmin, _seg_dist(p0, p1, x0, y0, x1, y1) / r, out=dmin)
    env = np.zeros((H, W), dtype=np.float32)
    env[y0:y1, x0:x1] = np.clip((1.35 - dmin) / 0.30, 0.0, 1.0)
    return env


def _torso_len(kp):
    sh_m = (0.5 * (kp["sh_r"][0] + kp["sh_l"][0]), 0.5 * (kp["sh_r"][1] + kp["sh_l"][1]))
    hip_m = (0.5 * (kp["hip_r"][0] + kp["hip_l"][0]), 0.5 * (kp["hip_r"][1] + kp["hip_l"][1]))
    return max(60.0, math.hypot(sh_m[0] - hip_m[0], sh_m[1] - hip_m[1])), sh_m, hip_m


def native_shorts(F, kp):
    """NATIVE 3D shorts (no 2D painting): the matte-black shorts are part of the generated 3D mannequin, so the
    mask is the source's own fabric pixels (dark, neutral, un-lit) and nothing else. Pose-aware for standing AND
    seated/bench poses: the search zone is built in thigh-axis coordinates (hip -> knee, any orientation), and each
    leg's hem is measured along its own thigh axis as the last cross-section where native fabric spans >= 50% of
    the thigh width. Pixels past the hem are never shorts, so nothing dark can appear below the hem."""
    lum, act_dom, sat, min_c = F["lum"], F["act_dom"], F["sat"], F["min_c"]
    lum_eq = np.maximum(lum - 0.85 * act_dom, min_c)
    fabric = (lum_eq < 62.0) & (sat < 0.30) & (act_dom < 18.0)
    tl, sh_m, hip_m = _torso_len(kp)
    r = 0.24 * tl                                    # thigh half-width ~ 0.24 x torso length (measured Day 5/6)
    P = lambda p: (int(round(p[0])), int(round(p[1])))
    zone = np.zeros((H, W), np.uint8)
    # hip block: waistband (0.36 x torso above the hip joints) down to the hip joints, hip-to-hip + thigh width
    up = (sh_m[0] - hip_m[0], sh_m[1] - hip_m[1])
    un = math.hypot(*up) or 1.0
    up = (up[0] / un, up[1] / un)
    wb = 0.36 * tl
    hr, hl = kp["hip_r"], kp["hip_l"]
    side = (hl[0] - hr[0], hl[1] - hr[1]); sn = math.hypot(*side) or 1.0; side = (side[0] / sn, side[1] / sn)
    e = 0.85 * r
    quad = [(hr[0] - side[0] * e + up[0] * wb, hr[1] - side[1] * e + up[1] * wb), (hl[0] + side[0] * e + up[0] * wb, hl[1] + side[1] * e + up[1] * wb),
            (hl[0] + side[0] * e, hl[1] + side[1] * e), (hr[0] - side[0] * e, hr[1] - side[1] * e)]
    cv2.fillConvexPoly(zone, np.array([P(q) for q in quad], np.int32), 1)
    hems = {}
    for sd in ("r", "l"):
        hip, kn = kp[f"hip_{sd}"], kp[f"kn_{sd}"]
        L = math.hypot(kn[0] - hip[0], kn[1] - hip[1]) or 1.0
        ax = ((kn[0] - hip[0]) / L, (kn[1] - hip[1]) / L)
        nx = (-ax[1], ax[0])
        hem_t = 0.0
        for t in np.arange(0.0, 0.80, 0.02):
            c = (hip[0] + ax[0] * L * t, hip[1] + ax[1] * L * t)
            xs = np.clip((c[0] + nx[0] * np.linspace(-r, r, 25)).astype(int), 0, W - 1)
            ys = np.clip((c[1] + nx[1] * np.linspace(-r, r, 25)).astype(int), 0, H - 1)
            if fabric[ys, xs].mean() >= 0.5:
                hem_t = t
            elif t > hem_t + 0.07:
                break
        hems[sd] = hem_t
        end = (hip[0] + ax[0] * L * (hem_t + 0.03), hip[1] + ax[1] * L * (hem_t + 0.03))
        cv2.line(zone, P(hip), P(end), 1, int(2 * r))
    LAST_MASKS["shorts_hem_t"] = hems
    m = (fabric & (zone > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        for i in range(1, n):   # the fabric body + any sizeable native fabric patch (folds split it)
            if i == big or st[i, cv2.CC_STAT_AREA] >= 0.08 * st[big, cv2.CC_STAT_AREA]:
                keep[lab == i] = 1
    # small enclosed sheen spots inside native fabric are fabric (native highlights, not skin)
    ff = np.pad(cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), 1).copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    keep = ((ff[1:-1, 1:-1] != 2) & (zone > 0)).astype(np.uint8)
    return keep.astype(np.float32), zone.astype(np.float32)


def skin_segments(kp):
    """Keypoint body segments that are ALWAYS skin (arms to just before the hand, torso above the hip band,
    shins/calves below the shorts). Equipment and shorts can never paint these pixels dark."""
    m = np.zeros((H, W), np.uint8)
    hip_y = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1])
    sh_m = (0.5 * (kp["sh_r"][0] + kp["sh_l"][0]), 0.5 * (kp["sh_r"][1] + kp["sh_l"][1]))
    hip_m = (0.5 * (kp["hip_r"][0] + kp["hip_l"][0]), hip_y - 150.0)
    sw = abs(kp["sh_l"][0] - kp["sh_r"][0])
    P = lambda p: (int(round(p[0])), int(round(p[1])))
    for side in ("r", "l"):
        sh, el, wr = kp[f"sh_{side}"], kp[f"elb_{side}"], kp[f"wr_{side}"]
        cv2.line(m, P(sh), P(el), 1, 30)                       # upper arm core
        cv2.line(m, P(el), P(_ext(el, wr, 0.80)), 1, 22)        # forearm core, stops before hand/handle
        kn, an = kp[f"kn_{side}"], kp[f"ank_{side}"]
        hip = kp[f"hip_{side}"]
        cv2.line(m, P(_ext(hip, kn, 0.78)), P(an), 1, 30)      # knee..ankle (below the shorts)
    t = np.zeros((H, W), np.uint8)
    cv2.line(t, P(sh_m), P(hip_m), 1, int(max(40, 0.45 * sw)))  # torso core above the hip band
    t[max(0, int(hip_y - 135.0)):] = 0                             # never reaches into the shorts (no round cap on the waistband)
    m |= t
    return m.astype(np.float32)


def features(rgb_1080):
    """Per-pixel colour features shared by every module."""
    r_ch = rgb_1080[:, :, 0].astype(np.float32)
    g_ch = rgb_1080[:, :, 1].astype(np.float32)
    b_ch = rgb_1080[:, :, 2].astype(np.float32)
    lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    min_c = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    sat = (max_c - min_c) / (max_c + 1.0)
    red_dom = np.clip(r_ch - 0.50 * (g_ch + b_ch), 0.0, 255.0)
    grn_dom = np.clip(g_ch - 0.50 * (r_ch + b_ch), 0.0, 255.0)
    act_dom = np.maximum(red_dom, grn_dom)
    return dict(rgb=rgb_1080.astype(np.float32), r=r_ch, g=g_ch, b=b_ch, lum=lum, max_c=max_c, min_c=min_c, sat=sat, act_dom=act_dom)


def body_key(F, step_idx, kp):
    """Clean foreground key (no activation bloom): row-backdrop difference inside the keypoint envelope,
    glow-robust bright signal, below-ankle platform suppression, shorts band dark key, shorts polygon prior."""
    lum, act_dom, min_c = F["lum"], F["act_dom"], F["min_c"]
    yy, xx = _YY, _XX
    hip_y = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1])
    kn_y = 0.5 * (kp["kn_r"][1] + kp["kn_l"][1])
    ank_max = max(kp["ank_r"][1], kp["ank_l"][1])

    # --- clean fg mask (no act_dom bloom) ---
    from gymform import ctx as _ctx
    cx0, _, cnw, _ = _ctx.clip_rect(step_idx)             # clip edges on the canvas (zoom/DX aware)
    la, rb_ = max(0, cx0) + 16, min(W, cx0 + cnw) - 16
    strips = np.concatenate([lum[:, la: la + 75], lum[:, rb_ - 75: rb_]], axis=1)
    row_bg = cv2.GaussianBlur(np.percentile(strips, 35, axis=1).astype(np.float32)[:, None], (0, 0), 16.0)
    env = anatomical_envelope(kp)
    floor_gate = np.clip(((ank_max + 52.0) - yy) / 24.0, 0.0, 1.0)
    # Coloured glow over the backdrop has high act_dom but a backdrop-level min channel; tinted body pixels keep a
    # high min channel. Taking the max of both keeps the body solid while the glow fringe drops out.
    bright_sig = np.maximum(lum - 0.85 * act_dom, min_c - 6.0)
    bright_diff = np.maximum(0.0, bright_sig - row_bg) * env

    # Platform / floor spill below the ankles: only clearly bright foot pixels count there.
    ank_min = min(kp["ank_r"][1], kp["ank_l"][1])
    below = np.clip((yy - (ank_min - 45.0)) / 12.0, 0.0, 1.0)
    bright_diff = np.maximum(0.0, bright_diff - 18.0 * below)

    near_wrist = equipment_soft_zone(kp)  # equipment zones from CFG (Day 5: 140px discs at wrists, full inside 115px)
    shorts_band = np.clip((yy - (hip_y - 140.0)) / 24.0, 0.0, 1.0) * np.clip(((kn_y - 32.0) - yy) / 34.0, 0.0, 1.0)
    dark_gate = shorts_band * env  # equipment is handled by convex-hull fill below (dark_diff near wrists picked up backdrop shadows)
    dark_diff = np.maximum(0.0, row_bg - lum) * dark_gate

    fg_diff = (bright_diff + dark_diff) * floor_gate
    w_raw = np.clip((fg_diff - 8.5) / 10.5, 0.0, 1.0).astype(np.float32)
    # ROOT CAUSE of the black holes in saturated native muscle fibres (Day 6 Step 3 deltoids): the glow-robust
    # bright signal subtracts act_dom, so a fully saturated red/green fibre (min channel ~ backdrop) read as
    # "glow over backdrop". Native fibres are TEXTURED (striations) and bright in their max channel; a glow
    # fringe over the backdrop is smooth. Textured, bright-max-channel pixels inside the envelope are body.
    max_c = F["max_c"]
    tex_m = cv2.GaussianBlur(np.abs(max_c - cv2.GaussianBlur(max_c, (0, 0), 2.0)), (0, 0), 1.5)
    fib = ((max_c - row_bg) > 70.0) & (tex_m > 3.0) & (act_dom > 25.0) & (env > 0.5)
    fib = cv2.morphologyEx(fib.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    # (muscle-fibre clips only: in bone X-Ray clips the coloured glow is handled by the X-Ray window)
    fib = (fib > 0) & ((max_c - row_bg) > 45.0) & (floor_gate > 0.5) & (step_idx in CFG.get("muscle_steps", [3, 4]))
    w_raw = np.maximum(w_raw, fib.astype(np.float32))
    # small holes fully enclosed by the body (smooth fibre bellies between striations) are body
    solid_b = cv2.morphologyEx((w_raw > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    hf = np.pad(solid_b, 1).copy()
    cv2.floodFill(hf, None, (0, 0), 2)
    holes = ((hf[1:-1, 1:-1] != 2) & (solid_b == 0)).astype(np.uint8)
    n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(holes)
    if step_idx in CFG.get("muscle_steps", [3, 4]):
        for i in range(1, n_h):
            if st_h[i, cv2.CC_STAT_AREA] <= 1500:
                w_raw[(lab_h == i) & (act_dom > 25.0)] = 1.0
        # gaps the 11px closing bridged inside saturated fibre regions (not the backdrop)
        w_raw = np.where((solid_b > 0) & (w_raw < 0.5) & (act_dom > 40.0) & (env > 0.5), 1.0, w_raw).astype(np.float32)

    # Matte-black dumbbells and shorts are nearly the same luminance as the source backdrop, so a pure
    # luminance key leaves them see-through. Close them with shape priors (wrist zone hole-fill, shorts polygon).
    solid = np.zeros((H, W), dtype=np.float32)
    shorts_native, shorts_zone = native_shorts(F, kp)
    return dict(w_raw=w_raw, shorts_poly=shorts_native, shorts_zone=shorts_zone, near_wrist=near_wrist, shorts_band=shorts_band, env=env,
                skin_seg=cv2.GaussianBlur(skin_segments(kp), (0, 0), 1.5))


def finalize(F, K, foot):
    """Union of body key, equipment footprint and the NATIVE shorts -> w_fg; shorts mask = native fabric pixels only
    (no polygon, no fill painting: 2D shorts overpaint is 0 by construction)."""
    solid = foot.copy()
    eq_solid = solid.copy()
    native = K["shorts_poly"] * (1.0 - K["skin_seg"])
    solid = np.maximum(solid, native)
    w_raw = K["w_raw"]
    plate = CUR_PLATE.get("rgb")
    if plate is not None:
        d = np.abs(F["rgb"] - plate).max(axis=-1)
        w_raw = np.where((w_raw < 0.95) & (d < 6.0) & (K["near_wrist"] > 0) & (foot < 0.5), 0.0, w_raw).astype(np.float32)
    w_fg = cv2.GaussianBlur(np.maximum(w_raw, solid), (0, 0), float(CFG.get("mask_feather", 0.85)))
    shorts_mask = cv2.GaussianBlur(native * (1.0 - np.clip(eq_solid * 1.5, 0.0, 1.0)), (0, 0), 0.8)
    LAST_MASKS["shorts_native"] = native
    return w_fg, eq_solid, shorts_mask
