"""Opt-in SMOKED-GLASS character look (t687u, owner-approved combo ② glass + Ⓑ 2-shot).
A smooth opaque mannequin frame becomes a translucent navy-smoke glass body with a fresnel rim. Behind it shows the REAL
machine plate (aligned empty plate), and the target muscles carry code-synthesised Day 4 neon fibres.
Body-anchored: regions and fibre texture live in a canonical TORSO space (origin = shoulder midpoint, axes = shoulder
vector u and shoulder->hip vector v) and are affine-warped per frame from the keypoints, so they track the body with no
sliding. Nothing here is used unless a tool calls it."""
import cv2, numpy as np

CW, CH = 480, 640            # canonical torso canvas; x = CW/2*(1+u/UM) (u = -1 sh_l .. +1 sh_r), y = CH/2*v (v 0 = sh, 1 = hip)
UM = 2.5            # canvas spans +-2.5 shoulder half-widths, so canonical px are ~isotropic vs the image (shoulder half-width ~ 0.3 torso)
DAY4_RIGHT_HSV = (74, 164, 179)      # OpenCV hue 74 = 148 deg
DAY4_WRONG_HSV = (0, 156, 175)


def torso_affine(kp):
    """2x3 affine canonical -> image. kp: dict sh_l, sh_r, hip_l, hip_r (image px)."""
    sl, sr = np.array(kp["sh_l"], np.float32), np.array(kp["sh_r"], np.float32)
    hm = 0.5 * (np.array(kp["hip_l"], np.float32) + np.array(kp["hip_r"], np.float32)); sm = 0.5 * (sl + sr)
    src = np.float32([C(0, 0), C(1, 0), C(0, 1)])
    return cv2.getAffineTransform(src, np.float32([sm, sr, hm]))


def C(u, v):   # canonical px
    return (CW * 0.5 * (1 + u / UM), CH * 0.5 * v)


def _poly(pts, soft=6.0):
    m = np.zeros((CH, CW), np.float32); cv2.fillPoly(m, [np.int32([C(u, v) for u, v in pts])], 1.0, cv2.LINE_AA)
    return np.clip(cv2.GaussianBlur(m, (0, 0), soft) * 1.6 - 0.3, 0, 1)


def _strokes(lines, rng, width=1.6, ss=2):
    """anti-aliased fibre strokes (canonical), random brightness per fibre; lines = list of [(u, v), ...] polylines"""
    t = np.zeros((CH * ss, CW * ss), np.float32)
    for ln in lines:
        b = float(rng.uniform(0.55, 1.0))
        pts = np.int32([[C(u, v)[0] * ss * 16, C(u, v)[1] * ss * 16] for u, v in ln])
        cv2.polylines(t, [pts], False, b, max(1, int(round(width * ss))), cv2.LINE_AA, shift=4)
    return cv2.resize(t, (CW, CH), interpolation=cv2.INTER_AREA)


def _fan(origin_a, origin_b, insert, n, rng, bend=0.0, jitter=0.012):
    """n fibres from points on the origin segment a->b converging on the insertion point, slightly curved"""
    out = []
    for i in range(n):
        f = (i + rng.uniform(0.2, 0.8)) / n
        o = (origin_a[0] + f * (origin_b[0] - origin_a[0]), origin_a[1] + f * (origin_b[1] - origin_a[1]))
        ln = []
        for k in range(9):
            q = k / 8.0; mx = o[0] + q * (insert[0] - o[0]); my = o[1] + q * (insert[1] - o[1])
            off = bend * 4 * q * (1 - q)
            ln.append((mx + off * (insert[1] - o[1]) + rng.normal(0, jitter), my - off * (insert[0] - o[0]) + rng.normal(0, jitter)))
        out.append(ln)
    return out


def regions_right(rng):
    """canonical (weight mask, fibre texture) per muscle for the RIGHT form (rear view, both sides):
    latissimus dorsi (fan: spine T7-L5 + iliac crest -> armpit), rhomboids + middle trapezius (spine -> medial scapula),
    posterior deltoid (shoulder cap back -> arm). Fibre spacing ~ 0.03 torso (Day 4 striation)."""
    R = {}
    for sg, nm in ((-1, "l"), (1, "r")):
        lat = _poly([(sg * 0.92, 0.30), (sg * 1.00, 0.46), (sg * 0.86, 0.74), (sg * 0.58, 0.96), (sg * 0.16, 1.00),
                     (sg * 0.10, 0.62), (sg * 0.42, 0.44), (sg * 0.74, 0.32)])
        fl = _fan((sg * 0.06, 0.52), (sg * 0.62, 1.04), (sg * 1.02, 0.30), 46, rng, bend=-0.10 * sg)
        R[f"lat_{nm}"] = (lat, _strokes(fl, rng))
        rh = _poly([(sg * 0.04, 0.10), (sg * 0.36, 0.13), (sg * 0.46, 0.24), (sg * 0.32, 0.44), (sg * 0.04, 0.50)])
        fr = [[(sg * 0.03, 0.08 + i * 0.016), (sg * 0.25, 0.13 + i * 0.016), (sg * 0.48, 0.18 + i * 0.016)] for i in range(26)]
        R[f"rhomb_{nm}"] = (rh, _strokes(fr, rng))
        mt = _poly([(sg * 0.04, 0.00), (sg * 0.34, 0.01), (sg * 0.66, 0.04), (sg * 0.36, 0.09), (sg * 0.04, 0.12)], soft=7) * 0.85
        fm = [[(sg * 0.03, -0.01 + i * 0.012), (sg * 0.68, 0.01 + i * 0.010)] for i in range(10)]
        R[f"midtrap_{nm}"] = (mt, _strokes(fm, rng))
        rd = _poly([(sg * 0.86, -0.06), (sg * 1.10, -0.04), (sg * 1.20, 0.10), (sg * 1.10, 0.24), (sg * 0.94, 0.16)], soft=4)
        fd = _fan((sg * 0.84, -0.04), (sg * 0.96, 0.18), (sg * 1.22, 0.24), 20, rng, bend=0.06 * sg)
        R[f"reardelt_{nm}"] = (rd, _strokes(fd, rng))
    return R


def build_canon(rng_seed=7, weights=None, **_):
    rng = np.random.default_rng(rng_seed)
    weights = weights or {}
    W = np.zeros((CH, CW), np.float32); T = np.zeros((CH, CW), np.float32)
    for k, (m, tex) in regions_right(rng).items():
        w = weights.get(k.split("_")[0], 1.0)
        W = np.maximum(W, m * w); T = np.maximum(T, tex * m)
    return W, T


def glass_body(bgr, body_a, behind=None, shorts=None, tint=(62, 50, 38), see=0.30, rim_px=6.0, rim_col=(235, 225, 205)):
    """body_a float 0..1 (soft silhouette); behind = what is behind the body (aligned empty plate) or None (inpaint)."""
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = (body_a > 0.5).astype(np.uint8)
    if behind is None:
        behind = cv2.inpaint(bgr, cv2.dilate(m, np.ones((7, 7), np.uint8)), 9, cv2.INPAINT_TELEA)
    Ls = cv2.GaussianBlur(L, (0, 0), 3.0); med = float(np.median(Ls[m > 0])) if m.any() else 128.0
    shade = np.clip(0.80 + 0.45 * (Ls - med) / 70.0, 0.45, 1.35)[..., None]
    inner = (see * cv2.GaussianBlur(behind.astype(np.float32), (0, 0), 1.2) + (1 - see) * np.array(tint, np.float32)) * shade
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    rim = (np.exp(-dist / rim_px) * m)[..., None]
    spec = np.clip((L - cv2.GaussianBlur(L, (0, 0), 9.0)) / 25.0, 0, 1)[..., None] * 0.35 * m[..., None]
    g = inner + rim * np.array(rim_col, np.float32) * 0.80 + spec * 255
    if shorts is not None:   # shorts stay matte black (opaque), with a faint rim
        s = shorts[..., None].astype(np.float32)
        g = g * (1 - s) + (bgr.astype(np.float32) * 0.85 + rim * 40) * s
    a = body_a[..., None]
    return np.clip(bgr * (1 - a) + g * a, 0, 255)


def neon(img, W, T, body_a, hsv=DAY4_RIGHT_HSV, gain=1.0, falloff_px=4.0):
    """Day 4 neon: solid core colour modulated by the fibre texture, edge ramp falloff_px, soft halo"""
    col = cv2.cvtColor(np.uint8([[list(hsv)]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    w = np.clip(W * body_a, 0, 1)
    core = (w > 0.5).astype(np.uint8)
    ramp = np.clip(cv2.distanceTransform(core, cv2.DIST_L2, 5) / falloff_px, 0, 1) * w
    f = 0.38 + 0.85 * cv2.GaussianBlur(T, (0, 0), 0.6)
    lit = col[None, None] * f[..., None]
    hot = np.clip((T - 0.80) / 0.20, 0, 1)[..., None] * ramp[..., None] * 70.0     # white-hot ridge tips
    a = (ramp * gain)[..., None]
    o = img * (1 - a) + lit * a + hot
    halo = cv2.GaussianBlur(ramp, (0, 0), 9.0)[..., None] * col * 0.45 * gain
    return np.clip(o + halo * (1 - a * 0.6), 0, 255)


# ---------------- t688u: Day 4 muscle look (volume + dark fibre grooves + per-muscle separation) ----------------
def _grooves(kind, param, rng, period=6.2, width=0.20, mask=None):
    """dark fibre-groove field 0..1 (1 = groove centre) in canonical space, plus per-bundle brightness variation.
    kind 'fan': fibres are rays converging on param (u, v) = insertion (spacing = period at the muscle centroid);
    'par': parallel fibres at angle param (deg)."""
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    if kind == "fan":
        px, py = C(*param); dx, dy = xx - px, yy - py
        th = np.arctan2(dy, dx)
        if mask is not None and mask.sum() > 0:
            my, mx = np.argwhere(mask > 0.5).mean(0); r_ref = float(np.hypot(mx - px, my - py))
            th_c = np.arctan2(my - py, mx - px); th = (th - th_c + np.pi) % (2 * np.pi) - np.pi   # no wrap inside the muscle
        else:
            r_ref = 200.0
        phase = th * r_ref / period
    else:
        a = np.radians(param); phase = (-xx * np.sin(a) + yy * np.cos(a)) / period
    phase = phase + cv2.GaussianBlur(rng.standard_normal((CH, CW)).astype(np.float32), (0, 0), 14) * 3.0   # wavy bundles
    q = np.abs(np.sin(np.pi * phase))
    wv = width * np.clip(1.0 + 0.6 * cv2.GaussianBlur(rng.standard_normal((CH, CW)).astype(np.float32), (0, 0), 3) * 3.0, 0.4, 1.8)
    g = np.exp(-(q / wv) ** 2) * np.clip(0.75 + 0.5 * cv2.GaussianBlur(rng.standard_normal((CH, CW)).astype(np.float32), (0, 0), 2.5) * 3.0, 0.2, 1.0)
    bid = np.floor(phase / 2.0)                                           # bundles of ~2 fibres
    bv = np.modf(np.abs(np.sin(bid * 12.9898 + 78.233)) * 43758.5453)[0]  # hash -> 0..1 per bundle
    # fine fibre sheen: noise stretched ALONG the fibre direction (per-muscle mean direction)
    if kind == "fan":
        if mask is not None and mask.sum() > 0:
            ang = np.degrees(np.arctan2(my - py, mx - px))
        else:
            ang = 0.0
    else:
        ang = param
    n = rng.standard_normal((CH * 2, CW * 2)).astype(np.float32)
    Mr = cv2.getRotationMatrix2D((CW, CH), ang, 1.0)
    r_ = cv2.warpAffine(n, Mr, (CW * 2, CH * 2), borderMode=cv2.BORDER_REFLECT)
    r_ = cv2.GaussianBlur(r_, (0, 0), sigmaX=9.0, sigmaY=1.25)
    r_ = cv2.warpAffine(r_, cv2.getRotationMatrix2D((CW, CH), -ang, 1.0), (CW * 2, CH * 2), borderMode=cv2.BORDER_REFLECT)[CH // 2:CH // 2 + CH, CW // 2:CW // 2 + CW]
    fine = r_ / (r_.std() + 1e-6)
    return np.dstack([np.clip(np.maximum(g, 0.35 * (1 - bv)), 0, 1), fine]).astype(np.float32)


def regions_right_v2(rng):
    """fuller anatomical shapes (rear view, canonical torso space), one entry per muscle: (name, mask, grooves, weight)"""
    out = []
    for sg in (-1, 1):
        lat = _poly([(sg * 0.96, 0.26), (sg * 1.04, 0.42), (sg * 0.98, 0.62), (sg * 0.84, 0.82), (sg * 0.60, 0.98),
                     (sg * 0.26, 1.00), (sg * 0.12, 0.92), (sg * 0.12, 0.62), (sg * 0.34, 0.50), (sg * 0.62, 0.40), (sg * 0.80, 0.30)], soft=12)
        out.append(("lat", lat, _grooves("fan", (sg * 1.10, 0.28), rng, mask=lat), 1.0))
        rh = _poly([(sg * 0.07, 0.13), (sg * 0.34, 0.15), (sg * 0.50, 0.24), (sg * 0.42, 0.46), (sg * 0.07, 0.52)], soft=9)
        out.append(("rhomb", rh, _grooves("par", -sg * 62.0, rng), 1.0))
        mt = _poly([(sg * 0.07, 0.00), (sg * 0.30, -0.01), (sg * 0.62, 0.01), (sg * 0.80, 0.04), (sg * 0.50, 0.10), (sg * 0.07, 0.12)], soft=9)
        out.append(("midtrap", mt, _grooves("fan", (sg * 0.95, 0.03), rng, mask=mt), 0.95))
        rd = _poly([(sg * 0.82, -0.08), (sg * 1.06, -0.08), (sg * 1.22, 0.06), (sg * 1.18, 0.24), (sg * 1.02, 0.28), (sg * 0.88, 0.12)], soft=8)
        out.append(("reardelt", rd, _grooves("fan", (sg * 1.24, 0.34), rng, mask=rd), 1.0))
    return out


def warp_muscles(kp, shape, rng_seed=7):
    H, W = shape; M = torso_affine(kp); rng = np.random.default_rng(rng_seed)
    return [(n, cv2.warpAffine(m, M, (W, H)), cv2.warpAffine(t, M, (W, H)), w) for n, m, t, w in regions_right_v2(rng)]


def day4_muscle(img, muscles, body_a, hsv=DAY4_RIGHT_HSV, gain=1.0, falloff_px=5.0, groove=0.60, halo=0.55, light=(-0.45, -0.9), fine_amp=0.12, neutral_amp=60.0):
    """Day 4 look per muscle: solid neon belly with a volume highlight (distance-to-edge profile, lit from upper left),
    dark fibre grooves (strokes), a falloff_px edge ramp, muscle-to-muscle separation (each muscle's own edge darkens),
    and a soft halo clipped to the body. muscles = [(name, mask, strokes, weight)] in image space."""
    h0, s0, v0 = hsv
    o = img.astype(np.float32).copy(); halo_acc = np.zeros(img.shape[:2], np.float32)
    col = cv2.cvtColor(np.uint8([[[h0, s0, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255.0
    hard = (body_a > 0.5).astype(np.uint8)
    for name, m, t, w in muscles:
        mm = (m > 0.5).astype(np.uint8) & hard
        if mm.sum() < 50:
            continue
        d = cv2.distanceTransform(mm, cv2.DIST_L2, 5)
        dmax = max(4.0, float(np.percentile(d[mm > 0], 97)))
        b = np.clip(d / dmax, 0, 1) ** 0.55                                   # belly profile 0 edge .. 1 centre
        gy, gx = np.gradient(cv2.GaussianBlur(b, (0, 0), 3))
        lit = np.clip(1.0 + 2.2 * (gx * light[0] + gy * light[1]) * dmax / 6.0, 0.7, 1.3)   # bulge facing the light
        shade = (0.76 + 0.50 * b) * lit
        if t.ndim == 3:
            tg, fine = t[..., 0], t[..., 1]
        else:
            tg = t; fine = cv2.GaussianBlur(np.random.default_rng(11).standard_normal(t.shape).astype(np.float32), (0, 0), 1.9) * 4.0
        ridge = np.clip(1.0 - tg * 1.6, 0, 1) ** 3                          # bright bundle crests between grooves
        val = v0 * shade * (1.0 - groove * np.clip(tg, 0, 1)) * np.clip(1.0 + fine_amp * fine, 0.40, 1.60) + ridge * 18.0 * (0.4 + 0.6 * b)
        val = val + np.clip(b - 0.75, 0, 1) * 50 * (1 - tg)                # belly specular
        a = np.clip(d / falloff_px, 0, 1) * np.clip(m, 0, 1) * w * gain
        lay = col[None, None] * np.clip(val, 0, 290)[..., None]
        lay = lay + (neutral_amp * fine * (0.5 + 0.5 * b) * (1 - 0.6 * tg))[..., None]   # neutral (white/grey) fibre sheen: keeps the hue dominance
        o = o * (1 - a[..., None]) + np.clip(lay, 0, 255) * a[..., None]
        halo_acc = np.maximum(halo_acc, a)
    hl = cv2.GaussianBlur(halo_acc, (0, 0), 7.0) * body_a * (1 - halo_acc)
    o = o + hl[..., None] * col * 255 * halo * gain
    return np.clip(o, 0, 255)


def glass_body_v2(bgr, body_a, behind, shorts=None, tint=(62, 50, 38), see=0.30, rim_px=5.0, rim_col=(235, 225, 205), edge_pad=4, light_edge=None):
    """v1 + no rim halo leak: the original (light) mannequin edge band just outside the silhouette is replaced with the
    behind plate, and the rim is drawn inside the hard edge only."""
    hard = (body_a > 0.5).astype(np.uint8)
    if light_edge is not None:   # the light mannequin edge the segmentation missed (within light_edge px of the mask)
        L0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        near = cv2.dilate(hard, np.ones((2 * light_edge + 1, 2 * light_edge + 1), np.uint8)) > 0
        Lb = cv2.cvtColor(behind, cv2.COLOR_BGR2GRAY).astype(np.int32)
        hard = (hard | (near & (L0 > 70) & (np.abs(L0.astype(np.int32) - Lb) > 22))).astype(np.uint8)
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    pad = cv2.dilate(hard, np.ones((2 * edge_pad + 1, 2 * edge_pad + 1), np.uint8))
    band = (pad > 0) & (hard == 0)
    base = bgr.copy(); base[band] = behind[band]
    pa = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 0.8)
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ls = cv2.GaussianBlur(L, (0, 0), 3.0); med = float(np.median(Ls[hard > 0]))
    shade = np.clip(0.80 + 0.45 * (Ls - med) / 70.0, 0.45, 1.35)[..., None]
    inner = (see * cv2.GaussianBlur(behind.astype(np.float32), (0, 0), 1.2) + (1 - see) * np.array(tint, np.float32)) * shade
    dist = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    rim = (np.exp(-dist / rim_px) * hard)[..., None]
    spec = np.clip((L - cv2.GaussianBlur(L, (0, 0), 9.0)) / 25.0, 0, 1)[..., None] * 0.35 * hard[..., None]
    g = inner + rim * np.array(rim_col, np.float32) * 0.70 + spec * 255
    if shorts is not None:
        s = shorts[..., None].astype(np.float32); g = g * (1 - s) + (bgr.astype(np.float32) * 0.85 + rim * 40) * s
    return np.clip(base * (1 - pa[..., None]) + g * pa[..., None], 0, 255), pa
