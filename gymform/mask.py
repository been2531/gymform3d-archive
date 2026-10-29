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


def shorts_polygon(kp: dict, bright_m: np.ndarray, hip_y: float, kn_y: float) -> np.ndarray:
    """Opaque boxer-shorts region from body extents at the waist (above) and thighs (below) + crotch notch."""
    hx = int(0.5 * (kp["hip_r"][0] + kp["hip_l"][0]))
    xa, xb = max(0, hx - 300), min(W, hx + 300)
    top = int(hip_y - 125.0)
    bot = int(hip_y + 0.55 * (kn_y - hip_y))
    crotch = int(hip_y + 0.36 * (kn_y - hip_y))

    # exclude hanging forearms / hands / dumbbells next to the hips from the body-width measurement
    y_a, y_b = max(0, int(hip_y) - 140), min(H, int(kn_y) + 10)
    arm_x = np.zeros((y_b - y_a, W), bool)
    for side in ("r", "l"):
        el, wr = kp[f"elb_{side}"], kp[f"wr_{side}"]
        d1 = _seg_dist(el, _ext(el, wr, 1.35), 0, y_a, W, y_b)
        d2 = np.hypot(_XX[y_a:y_b] - wr[0], _YY[y_a:y_b] - wr[1])
        arm_x |= (d1 < 55.0) | (d2 < 105.0)
    bright_m = bright_m.copy()
    bright_m[y_a:y_b] &= ~arm_x

    def cols(r):
        return np.where(bright_m[max(0, r - 4) : r + 5, xa:xb].any(axis=0))[0] + xa

    def occ_row(r):
        o = bright_m[max(0, r - 4) : r + 5, :].any(axis=0).astype(np.uint8)
        return cv2.morphologyEx(o[None, :], cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))[0] > 0

    def run_at(o, x):
        """Contiguous occupied run containing (or nearest to, within 40px) column x."""
        x = int(np.clip(x, 0, W - 1))
        if not o[x]:
            near = np.where(o[max(0, x - 40) : x + 41])[0]
            if len(near) == 0:
                return None
            x = max(0, x - 40) + int(near[np.argmin(np.abs(near - 40))])
        l = x
        while l > 0 and o[l - 1]:
            l -= 1
        r = x
        while r < W - 1 and o[r + 1]:
            r += 1
        return l, r

    # waist: the skin run through the body centre (arms hanging beside it are separate runs)
    rt_run = run_at(occ_row(top), hx)
    ob = occ_row(bot)
    rl = run_at(ob, kp["hip_r"][0])
    rr = run_at(ob, kp["hip_l"][0])
    out = np.zeros((H, W), dtype=np.float32)
    if rt_run is None or rl is None or rr is None:
        return out
    lt, rt = rt_run[0] - 4, rt_run[1] + 4
    lb, rb = rl[0] - 4, rr[1] + 4
    pts = [(lt, top), (rt, top), (rb, bot)]
    occ = bright_m[max(0, bot - 4) : bot + 5, :].any(axis=0)
    gl = gr = None
    if not occ[hx]:
        gl, gr = hx, hx
        while gl > lb and not occ[gl]:
            gl -= 1
        while gr < rb and not occ[gr]:
            gr += 1
    if gl is not None and gr - gl > 8:
        pts += [(gr - 2, bot), (hx, crotch), (gl + 2, bot)]
    pts += [(lb, bot)]
    cv2.fillPoly(out, [np.array(pts, dtype=np.int32)], 1.0, cv2.LINE_AA)
    return out


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
    sx0 = (W - int(round(W * STEP_SCALE.get(step_idx, b4.SCALE)))) // 2  # clip left edge (54 at scale 0.90)
    strips = np.concatenate([lum[:, sx0 + 16 : sx0 + 91], lum[:, W - sx0 - 91 : W - sx0 - 16]], axis=1)
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

    # Matte-black dumbbells and shorts are nearly the same luminance as the source backdrop, so a pure
    # luminance key leaves them see-through. Close them with shape priors (wrist zone hole-fill, shorts polygon).
    solid = np.zeros((H, W), dtype=np.float32)
    shorts_poly = shorts_polygon(kp, (bright_diff * floor_gate) > 14.0, hip_y, kn_y)
    return dict(w_raw=w_raw, shorts_poly=shorts_poly.astype(np.float32), near_wrist=near_wrist, shorts_band=shorts_band, env=env,
                skin_seg=cv2.GaussianBlur(skin_segments(kp), (0, 0), 1.5))


def finalize(F, K, foot):
    """Union of body key, equipment footprint and shorts prior -> w_fg; opaque matte-black shorts mask."""
    lum, sat, act_dom = F["lum"], F["sat"], F["act_dom"]
    shorts_band = K["shorts_band"]
    solid = foot.copy()
    eq_solid = solid.copy()
    solid = np.maximum(solid, K["shorts_poly"])
    w_raw = K["w_raw"]
    plate = CUR_PLATE.get("rgb")
    if plate is not None:
        # faint key pixels identical to the static backdrop plate inside the equipment zones are backdrop light
        # (arm ghosts held in the median plate), not body: solid body pixels are untouched
        d = np.abs(F["rgb"] - plate).max(axis=-1)
        w_raw = np.where((w_raw < 0.95) & (d < 6.0) & (K["near_wrist"] > 0) & (foot < 0.5), 0.0, w_raw).astype(np.float32)
    w_fg = cv2.GaussianBlur(np.maximum(w_raw, solid), (0, 0), float(CFG.get("mask_feather", 0.85)))
    dark_low_sat = np.clip((68.0 - lum) / 18.0, 0.0, 1.0) * np.clip((0.28 - sat) / 0.14, 0.0, 1.0) * np.clip((18.0 - act_dom) / 12.0, 0.0, 1.0)
    shorts_mask = cv2.GaussianBlur((shorts_band * dark_low_sat * (w_fg > 0.25)).astype(np.float32), (0, 0), 2.0)
    shorts_mask = np.clip(shorts_mask * 1.35, 0.0, 1.0)
    # equipment beside the hips is never "shorts"; inside the hip-band polygon (which already excludes the hanging
    # arms/hands/dumbbells) the fabric wins, so a footprint that picked up moving fabric cannot punch holes in it
    poly_in = cv2.erode((K["shorts_poly"] > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    shorts_mask = shorts_mask * (1.0 - np.clip(eq_solid * 1.5, 0.0, 1.0) * (1.0 - poly_in))
    # shorts only inside the hip-band polygon, never on a skin segment (correct by construction)
    shorts_mask = shorts_mask * np.clip(cv2.GaussianBlur(K["shorts_poly"], (0, 0), 2.0) * 1.5, 0.0, 1.0) * (1.0 - K["skin_seg"])
    # clean hem: shorts fabric rows are wide horizontal runs; a narrow vertical tongue below the hem is the
    # shadowed inner thigh, never fabric. A 61px-wide horizontal opening keeps only real fabric rows.
    fab = cv2.morphologyEx((shorts_mask > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 61), np.uint8))
    fab = cv2.morphologyEx(fab, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # re-join the fabric body
    shorts_mask = shorts_mask * cv2.GaussianBlur(cv2.dilate(fab, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.0)
    # fabric highlights (lum > 68) inside the fabric are still fabric: enclosed gaps in the fabric are filled
    fc = cv2.morphologyEx(fab, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    ff = np.pad(fc, 1).copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    fill = ((ff[1:-1, 1:-1] != 2) & (K["shorts_poly"] > 0.5)).astype(np.float32)
    # clean waistband top: the real band starts at the first row where the opened fabric spans >=70% of the
    # polygon width; anything above (dark fibre texture / shadows on the lower back) is skin, never shorts
    ys_p = np.where(K["shorts_poly"].max(axis=1) > 0.5)[0]
    top = np.ones((H, 1), np.float32)
    if len(ys_p):
        pw = np.maximum((K["shorts_poly"] > 0.5).sum(axis=1), 1)
        cov = ((fab > 0) & (K["shorts_poly"] > 0.5)).sum(axis=1) / pw
        rows = np.where((cov >= 0.70) & (np.arange(H) >= ys_p[0]))[0]
        if len(rows):
            band_top = int(rows[0])
            top[:max(0, band_top - 2)] = 0.0
            # waistband highlights (shiny elastic, lum < 125) in the top 60px of the band are fabric too, so the
            # band is never punched by a highlight notch
            hl = np.zeros((H, W), np.float32)
            hl[band_top: band_top + 60] = 1.0
            fill = np.maximum(fill, hl * poly_in * ((lum < 125.0) & (act_dom < 18.0)).astype(np.float32))
    topm = cv2.GaussianBlur(np.repeat(top, W, 1), (0, 0), 1.2)
    shorts_mask = np.maximum(shorts_mask, cv2.GaussianBlur(fill, (0, 0), 1.0) * (1.0 - K["skin_seg"]) * (1.0 - np.clip(eq_solid * 1.5, 0, 1) * (1.0 - poly_in)))
    shorts_mask = shorts_mask * topm
    return w_fg, eq_solid, shorts_mask
