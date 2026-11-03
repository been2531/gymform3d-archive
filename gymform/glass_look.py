"""Opt-in SMOKED-GLASS character look (t687u, owner-approved combo ② glass + Ⓑ 2-shot).
A smooth opaque mannequin frame becomes a translucent navy-smoke glass body with a fresnel rim. Behind it shows the REAL
machine plate (aligned empty plate), and the target muscles carry code-synthesised Day 4 neon fibres.
Body-anchored: regions and fibre texture live in a canonical TORSO space (origin = shoulder midpoint, axes = shoulder
vector u and shoulder->hip vector v) and are affine-warped per frame from the keypoints, so they track the body with no
sliding. Nothing here is used unless a tool calls it."""
import cv2, numpy as np

CW, CH = 480, 640            # canonical torso canvas; x = CW/2*(1+u/UM) (u = -1 sh_l .. +1 sh_r), y = CH/2*v (v 0 = sh, 1 = hip)
UM = 1.25
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
