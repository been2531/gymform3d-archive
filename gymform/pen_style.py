"""Pen-line style (t669u, opt-in, default off): a code-only look for @gymform.3d frames.

Charcoal Day 4 backdrop + clean white/light-grey pen lines traced from the source pixels (machine from the EMPTY
machine plate, mannequin from the person frame), a light 45-degree hatch in the shadow side of the mannequin, and the
target muscles only filled in the Day 4 red/emerald with code-made fibre striations and an edge falloff.
Nothing is generated or deformed: every line sits on an edge of the real photo pixels (Canny / mask contours),
so the machine geometry is the photo's, pixel for pixel.

Inputs (all 1080x1920 BGR, same framing): empty-machine plate, person frame, keypoints (region_glow names).
Regions come from gymform.region_glow.region_masks (the same anatomy polygons as the 3D pipeline).
"""
import cv2
import numpy as np

BG_RGB = (35, 39, 43)            # Day 4 master backdrop median (gymform_spec background.grade_to_rgb)
INK_MACHINE = 226                # line grey levels
INK_BODY = 242
# Day 4 core colour targets (tools/day4_look_metrics.py on day4_master_t7.5_wrong / t17.5_right)
D4_HSV = {"wrong": (359.1, 155.6, 175.1), "right": (147.6, 163.5, 179.0)}
FIBRE_PERIOD_PX = 14.0           # stripe period at 1080 on-screen scale (tuned against Day 4 density, see mock json)


def _lum(bgr):
    f = bgr.astype(np.float32)
    return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]


def bg_fit(L, iters=3, thr=6.0):
    """robust 2D quadratic fit of the studio backdrop luminance (machine/person excluded iteratively)."""
    h, w = L.shape
    ys, xs = np.mgrid[0:h:8, 0:w:8]
    l = L[::8, ::8]
    x = xs.ravel() / w; y = ys.ravel() / h; v = l.ravel()
    A = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
    keep = np.ones_like(v, bool)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        keep = np.abs(A @ c - v) < thr
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    X /= w; Y /= h
    return (c[0] + c[1] * X + c[2] * Y + c[3] * X * X + c[4] * Y * Y + c[5] * X * Y).astype(np.float32)


def _clean(m, open_px=3, close_px=7, min_area=400):
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def fill_holes(m, max_frac=0.01):
    """fill enclosed holes smaller than max_frac of the mask area (charcoal pixels at the backdrop level inside the
    body); larger enclosed gaps (backdrop seen under the thighs) stay open."""
    holes = (m == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, 4)
    out = m.copy().astype(np.uint8)
    lim = max_frac * max(int(m.sum()), 1)
    h, w = m.shape
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < lim and x > 0 and y > 0 and x + bw < w and y + bh < h:
            out[lab == i] = 1
    return out


def backdrop(L, sigma=48.0, thr=12.0, iters=2):
    """smooth studio backdrop estimate: normalised Gaussian convolution of the pixels that look like backdrop."""
    k = np.ones_like(L)
    for _ in range(iters):
        bg = cv2.GaussianBlur(L * k, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(k, (0, 0), sigma), 1e-3)
        k = (np.abs(L - bg) < thr).astype(np.float32)
    return bg


def masks(empty_bgr, person_bgr, paste, relight_mask_bgr=None, diff_thr=2.5):
    """machine mask (empty plate vs its own backdrop estimate), body mask and person-foreground mask.
    paste = the still compositing mask (tools/still_person_composite.py): the person silhouette, but it also holds
    machine bits copied around the person -> only pixels that differ from the empty plate count (the same rule as
    mannequin_relight.person_mask). relight_mask_bgr (optional, mannequin_relight debug mask, R = handle / bright
    neutral) removes the chrome handle from the body so it is drawn with machine-weight lines."""
    Le = _lum(empty_bgr)
    bg = backdrop(Le)
    mach = _clean(np.abs(Le - bg) > 10.0, 3, 5, 300)
    # the compositor keeps machine/backdrop pixels byte-identical, so ANY local change marks the person (the charcoal
    # head/shorts sit only 3-15 levels from the backdrop, a hard threshold loses them)
    d = np.abs(person_bgr.astype(np.int16) - empty_bgr.astype(np.int16)).max(2).astype(np.float32)
    ev = (cv2.GaussianBlur(d, (0, 0), 3.0) > diff_thr).astype(np.uint8)
    ev = cv2.morphologyEx(ev, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    fg = _clean((paste > 127) & (ev > 0), 5, 9, 400)
    fg = fill_holes(fg)
    body = fg.copy()
    if relight_mask_bgr is not None:
        # skin + shorts (+ bright rim on the head/shoulder next to skin); the black footplate / handle the compositor
        # re-drew around the feet and hands are NOT body -> drawn as machine from the frame pixels
        sk, sh, hd = (relight_mask_bgr[..., i] > 127 for i in range(3))
        near = cv2.dilate((sk | sh).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        hand_ = hd & ~(cv2.dilate((sk | sh).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
        reach = cv2.dilate((sk | sh).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
        b = cv2.morphologyEx(((fg > 0) & reach & ~hand_).astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        b = _clean(b, 3, 15, 4000)
        # close notches the dark (unlabelled) shadow side of the head/face leaves in the silhouette, inside fg only
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))) & fg
        body = fill_holes(b, 0.02) & fg
    return mach, body, fg, bg


def _canny_ink(L, sigma, lo, hi, width=1.6):
    g = cv2.GaussianBlur(L, (0, 0), sigma)
    e = cv2.Canny(np.clip(g, 0, 255).astype(np.uint8), lo, hi, L2gradient=True)
    e = (e > 0).astype(np.float32)
    if width > 1.2:
        e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    return np.clip(cv2.GaussianBlur(e, (0, 0), 0.7) * 1.6, 0, 1)


def _drop_short(ink, min_len=40):
    """remove tiny edge specks (texture noise) so only drawn strokes remain."""
    b = (ink > 0.35).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= min_len
    k = keep[lab].astype(np.float32)
    k = cv2.dilate(k, np.ones((3, 3), np.uint8))
    return ink * k


def contour_ink(m, width=3.0, smooth=2.0):
    ms = cv2.GaussianBlur(m.astype(np.float32), (0, 0), smooth) > 0.5
    cs, _ = cv2.findContours(ms.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = np.zeros(m.shape, np.uint8)
    cv2.drawContours(c, cs, -1, 255, int(round(width)), cv2.LINE_AA)
    return c.astype(np.float32) / 255.0


def hatch(shape, spacing=7, angle=45.0, width=1):
    h, w = shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.radians(angle)
    s = (X * np.cos(t) + Y * np.sin(t)) % spacing
    return np.clip(1.0 - np.abs(s - spacing / 2.0) / (0.5 * width + 0.6), 0, 1).astype(np.float32)


def crease_ink(L, sigma=1.4, k=1.6, lo=0.25, span=3.2, gamma=0.75):
    """pen strokes = luminance VALLEYS of a difference of Gaussians (XDoG-style, one-sided): every crease, seam and
    dark-side edge of the source becomes a thin line exactly where it is in the photo (no vectorising, no warp)."""
    D = cv2.GaussianBlur(L, (0, 0), sigma) - cv2.GaussianBlur(L, (0, 0), sigma * k)
    return np.clip((-D - lo) / span, 0, 1) ** gamma


def line_art(empty_bgr, person_bgr, mach, pers, fg, bg, line_src_bgr=None):
    """-> (rgb float32 canvas, body weight, layers). line_src_bgr = frame the body lines are traced from (default
    person_bgr; the un-relit pinned still gives the crispest anatomy creases)."""
    h, w = mach.shape
    out = np.empty((h, w, 3), np.float32); out[:] = BG_RGB
    # very low backdrop falloff (Day 4 studio), so the frame reads flat / premium
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    vig = 1.0 + 0.10 * np.exp(-(((X - 0.55 * w) / (0.75 * w)) ** 2 + ((Y - 0.55 * h) / (0.6 * h)) ** 2))
    out *= vig[..., None]
    Le, Lp = _lum(empty_bgr), _lum(person_bgr)
    Ll = _lum(line_src_bgr) if line_src_bgr is not None else Lp
    persf = cv2.GaussianBlur(pers.astype(np.float32), (0, 0), 1.2)
    fgf = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.2)
    machf = cv2.GaussianBlur(mach.astype(np.float32), (0, 0), 1.2) * (1 - fgf)
    otherf = np.clip(fgf - persf, 0, 1)
    # 1) faint tonal fills: solids read as solids (machine a touch above the wall, body a little more)
    out += (machf * (5.0 + 9.0 * np.clip((Le - bg) / 120.0, 0, 1)))[..., None]
    out += (otherf * (5.0 + 9.0 * np.clip((Lp - bg) / 120.0, 0, 1)))[..., None]
    tone_p = cv2.GaussianBlur(Lp, (0, 0), 2.0)
    pm = float(np.median(tone_p[pers > 0])) if pers.sum() else 60.0
    out += (persf * (12.0 + 10.0 * np.clip((tone_p - pm) / 50.0, -1, 1)))[..., None]
    # 2) ink: machine from the EMPTY plate (exact geometry), person + whatever it changed from the frame
    near = cv2.dilate(np.maximum(mach, fg).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    ink_m = crease_ink(Le) * (1 - fgf) * near
    ink_o = crease_ink(Lp) * otherf
    ink_b = crease_ink(Ll) * persf
    ink_c = contour_ink(pers, 2.4, 3.0) * 0.9
    a_m = np.clip(np.maximum(ink_m, ink_o) * 0.92, 0, 1)
    a_p = np.clip(np.maximum(ink_b, ink_c), 0, 1)
    out = out * (1 - a_m[..., None]) + INK_MACHINE * a_m[..., None]
    out = out * (1 - a_p[..., None]) + INK_BODY * a_p[..., None]
    return out, persf, {"ink_machine": a_m, "ink_body": a_p, "ink_crease": ink_b}


def _dir_field(kp, region_names, ms):
    """per pixel: unit fibre direction + the body anchor the stripes are measured from (so fibres ride with the body
    in a clip instead of swimming in screen space). Erectors along the spine, traps neck->acromion, lats/rhomboids
    toward the humerus, biceps/forearm along the limb; anchor = hip midpoint (torso) or the limb's proximal joint."""
    from gymform.region_glow import _frame, _P
    hip_m, U, V = _frame(kp)
    u = U / max(np.linalg.norm(U), 1e-3)
    vn = V / max(np.linalg.norm(V), 1e-3)
    shape = next(iter(ms.values())).shape
    dx = np.zeros(shape, np.float32); dy = np.zeros(shape, np.float32); best = np.zeros(shape, np.float32)
    ox = np.full(shape, hip_m[0], np.float32); oy = np.full(shape, hip_m[1], np.float32)
    sd = "r" if kp["sh_r"][0] <= kp["sh_l"][0] else "l"

    def put(name, d, o):
        nonlocal best
        m = ms[name]; sel = m > best
        d = d / max(np.linalg.norm(d), 1e-3)
        dx[sel] = d[0]; dy[sel] = d[1]; ox[sel] = o[0]; oy[sel] = o[1]; best = np.maximum(best, m)

    for name in region_names:
        if name == "erector_spinae":
            put(name, u, hip_m)
        elif name == "lats":
            put(name, u + 0.9 * vn, hip_m)
        elif name == "mid_trap_rhomboid":
            put(name, 0.35 * u + vn, hip_m)
        elif name == "upper_trap":
            put(name, -0.30 * U + 0.80 * V, hip_m)
        elif name == "rear_delt":
            put(name, vn - 0.4 * u, hip_m)
        elif name in ("biceps", "forearm"):
            j0, j1 = ("sh", "elb") if name == "biceps" else ("elb", "wr")
            put(name, _P(kp, f"{j1}_{sd}") - _P(kp, f"{j0}_{sd}"), _P(kp, f"{j0}_{sd}"))
    return dx, dy, ox, oy


_NOISE = {}


def _noise_tex(seed, n=2048):
    """one fixed smooth noise texture (sampled in body-anchored coordinates -> no temporal flicker)."""
    if seed not in _NOISE:
        rng = np.random.default_rng(seed)
        z = cv2.GaussianBlur(rng.standard_normal((n // 8, n // 8)).astype(np.float32), (0, 0), 2.0)
        _NOISE[seed] = cv2.resize(z / (z.std() + 1e-6), (n, n), interpolation=cv2.INTER_CUBIC)
    return _NOISE[seed]


def muscle_fill(canvas, src_bgr, pers_w, kp, targets, form, period=FIBRE_PERIOD_PX, depth=0.5, seed=7, ink=None,
                wash=None, glow_src=True):
    """Day 4-like colour + fibres inside the allowed anatomy regions only (x body matte). -> (rgb, weight, regions).

    Shape (Day 4 way, t648u): where the source already glows inside an allowed region, the glow's own shape is used
    (clamped to the region, as muscle_glow_clamp does in the 3D pipeline); a target region with no source glow falls
    back to its keypoint polygon. Colour = Day 4 core HSV, fibres = thin striations along the muscle's line of pull
    (code, not drawn by a generator), volume = the source's low-frequency luminance, anatomy creases (pen ink) darken
    the colour a little so muscle borders stay readable.
    wash = None: solid Day 4 colour (stills).  wash = a (0..1): t672u pen-line muscle - translucent colour wash at
    opacity a over the pen drawing, fibres as light hatch strokes and the anatomy creases re-inked in a light tint."""
    from gymform.region_glow import region_masks
    h, w = pers_w.shape
    ms = region_masks(kp, (h, w))
    f = src_bgr.astype(np.float32)
    dom = (f[..., 2] - np.maximum(f[..., 1], f[..., 0])) if form == "wrong" else (f[..., 1] - np.maximum(f[..., 2], f[..., 0]))
    gl = np.clip((cv2.GaussianBlur(dom, (0, 0), 1.5) - 12.0) / 45.0, 0, 1)
    reg = np.zeros((h, w), np.float32)
    for name, k in targets.items():
        r = ms[name]
        rd = cv2.dilate(r, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        g_in = gl * rd
        cov = float((g_in > 0.5).sum()) / max(float(r.sum()), 1.0)
        shape = g_in if (glow_src and cov > 0.12) else r      # source glow shape, else the region polygon
        reg = np.maximum(reg, float(k) * shape)
    core = cv2.GaussianBlur(np.clip(reg * 1.6, 0, 1), (0, 0), 2.0 if wash is None else 5.0)   # pen wash: softer edge
    wgt = np.clip(core, 0, 1) * cv2.erode(pers_w, np.ones((3, 3), np.uint8))
    dx, dy, ox, oy = _dir_field(kp, list(targets), ms)
    dx = cv2.GaussianBlur(dx, (0, 0), 6.0); dy = cv2.GaussianBlur(dy, (0, 0), 6.0)
    nn = np.sqrt(dx * dx + dy * dy) + 1e-6; dx /= nn; dy /= nn
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    Xr, Yr = X - ox, Y - oy                                      # body-anchored coordinates
    tex = _noise_tex(seed)
    nz = cv2.remap(tex, (Xr + 1024).astype(np.float32), (Yr + 1024).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    s = (-dy * Xr + dx * Yr) + 1.6 * nz
    t = (dx * Xr + dy * Yr)
    ph = (s / period) % 1.0
    line = np.clip(1.0 - np.abs(ph - 0.5) / 0.16, 0, 1) ** 1.5
    brk = 0.5 + 0.5 * np.sin(2 * np.pi * t / (period * 9.0) + 2.3 * nz)
    Lp = cv2.GaussianBlur(_lum(src_bgr), (0, 0), 4.0)
    inr = wgt > 0.5
    mu = float(Lp[inr].mean()) if inr.sum() > 100 else 80.0
    sd_ = float(Lp[inr].std()) + 1e-3 if inr.sum() > 100 else 30.0
    vol = np.clip(1.0 + 0.10 * (Lp - mu) / sd_, 0.80, 1.18)
    hue, sat, val = D4_HSV[form]
    hsv2rgb = lambda hh, ss, vv: cv2.cvtColor(np.stack([np.full_like(vv, (hh % 360) / 2.0), np.full_like(vv, ss), vv], -1)
                                              .astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if wash is None:
        fibre = 1.0 - depth * line * (0.55 + 0.45 * brk)
        if ink is not None:
            vol = vol * (1.0 - 0.35 * np.clip(ink, 0, 1))
        col = hsv2rgb(hue, sat, np.clip(val * 1.06 * fibre * vol, 0, 255))
        out = canvas * (1 - wgt[..., None]) + col * wgt[..., None]
        return out, wgt, ms
    # pen-line muscle: wash + light hatch strokes + re-inked creases (all inside wgt)
    col = hsv2rgb(hue, sat, np.clip(val * vol, 0, 255))
    a = (wash * wgt)[..., None]
    out = canvas * (1 - a) + col * a
    hat_col = hsv2rgb(hue, sat * 0.55, np.full((h, w), 238.0, np.float32))
    ha = (0.60 * line * (0.45 + 0.55 * brk) * wgt)[..., None]
    out = out * (1 - ha) + hat_col * ha
    if ink is not None:
        ia = (0.75 * np.clip(ink, 0, 1) * wgt)[..., None]
        out = out * (1 - ia) + hsv2rgb(hue, sat * 0.30, np.full((h, w), 248.0, np.float32)) * ia
    return out, wgt, ms
