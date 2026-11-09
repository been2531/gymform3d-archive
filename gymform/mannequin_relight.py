"""t665u/t667u opt-in deterministic MANNEQUIN RELIGHT (0 generations), reusable for any day with a fixed camera and a
static set (machine / bench) for which an EMPTY plate exists (same framing, no person).

Per frame, in place, on the clip's OWN pixels (no cut-out, no paste, no layer - DEFECT_PLAYBOOK per-frame cut-out rule):
  person   = pixels that differ from the empty plate (|frame - empty| > diff_thr, blurred 1.5) - closed - large components -
             closed again (fills the charcoal pixels that coincide with the backdrop) - inside the keypoint body capsule
             (pasted machine bits around the person, footplate / pulley, are never person)
  kept     = matte-black shorts (pose hip->60% knee zone, dark neutral, geodesic grow), chrome handle (bright neutral),
             thin dark cable at the person rim
  skin     = person - kept; low-frequency shading (normalised blur sigma) compressed toward a flat level T above the
             backdrop: base' = T + k (base - T); fine detail (fibres, edges) kept; crevices floored (never black);
             glow chroma (rgb - lum) kept unless deglow
  outside the person mask every pixel is byte-identical (machine geometry and pixels untouched).
Opt-in: nothing imports this unless a tool / config asks for it. Days 4/5/6 never touch it."""
import cv2
import numpy as np

CAP_PAIRS = ((11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32),
             (11, 12), (23, 24), (15, 19), (16, 20))
DEFAULTS = dict(T=82.0, k=0.2, floor=70.0, diff_thr=5.0, cap_w=0.30, deglow=False, shorts_L=34.0, handle_L=115.0)


def lum(im):
    return 0.299 * im[..., 2] + 0.587 * im[..., 1] + 0.114 * im[..., 0]


def detect_pose(img_bgr):
    """MediaPipe 33 landmarks in pixels (array 33x2) or None."""
    import universal_anatomy_rig as uar
    res = uar.get_pose_detector().detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB,
                                                       data=np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))))
    if not res.pose_landmarks:
        return None
    H, W = img_bgr.shape[:2]
    return np.array([[l.x * W, l.y * H] for l in res.pose_landmarks[0]], np.float32)


def capsule(P, shape, cap_w=0.30):
    H, W = shape
    sh = 0.5 * (P[11] + P[12]); hp = 0.5 * (P[23] + P[24])
    tl = float(np.linalg.norm(sh - hp))
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.array([P[11], P[12], P[24], P[23]], np.int32)), 1)
    Pi = lambda i: tuple(int(v) for v in P[i])
    for i, j in CAP_PAIRS:
        cv2.line(m, Pi(i), Pi(j), 1, max(8, int(cap_w * tl)))
    hc = tuple(int(v) for v in 0.5 * (P[7] + P[8]))
    cv2.circle(m, hc, int(0.36 * tl), 1, -1)
    cv2.line(m, hc, tuple(int(v) for v in sh), 1, int(0.30 * tl))
    return m > 0, tl


def person_mask(im, em, P, o, given=None):
    d = np.abs(cv2.GaussianBlur(im, (0, 0), 1.5) - cv2.GaussianBlur(em, (0, 0), 1.5)).max(2)
    cap, tl = capsule(P, im.shape[:2], o["cap_w"])
    if given is not None:   # exact paste mask (still compositing) AND differs from the empty plate
        ev = cv2.morphologyEx((np.abs(im - em).max(2) > 4).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return (given & (cv2.dilate(ev, np.ones((5, 5), np.uint8)) > 0)), cap, tl
    raw = (d > o["diff_thr"]).astype(np.uint8)
    p = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(p)
    p = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 3000]).astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    # fill fully enclosed holes (charcoal = backdrop level inside the silhouette)
    ff = p.copy(); mk = np.zeros((p.shape[0] + 2, p.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 1)
    p = p | (1 - ff)
    return (p > 0) & cap, cap, tl


def relight(img_bgr, empty_bgr, P, opts=None, given_mask=None):
    """-> (out uint8, dict(skin, shorts, handle, person))."""
    o = dict(DEFAULTS, **(opts or {}))
    im = img_bgr.astype(np.float32); em = empty_bgr.astype(np.float32)
    H, W = im.shape[:2]
    person, cap, tl = person_mask(im, em, P, o, given_mask)
    L = lum(im); sat = im.max(2) - im.min(2)
    hip = [P[23], P[24]]; kn = [P[25], P[26]]
    zone = np.zeros((H, W), np.uint8)
    poly = np.array([hip[0], hip[1], hip[1] + 0.6 * (kn[1] - hip[1]), hip[0] + 0.6 * (kn[0] - hip[0])], np.int32)
    cv2.fillConvexPoly(zone, cv2.convexHull(poly), 1)
    zk = int(0.35 * tl) | 1
    zone = cv2.dilate(zone, np.ones((zk, zk), np.uint8))
    shorts = ((zone > 0) & (cv2.GaussianBlur(L, (0, 0), 3) < o["shorts_L"]) & (sat < 14)).astype(np.uint8)
    shorts = cv2.morphologyEx(shorts, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    shorts = cv2.morphologyEx(shorts, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    grow = ((zone > 0) & (L < 58) & (sat < 16)).astype(np.uint8)
    for _ in range(12):
        shorts = np.maximum(shorts, cv2.dilate(shorts, np.ones((3, 3), np.uint8)) & grow)
    shorts = cv2.dilate(shorts, np.ones((3, 3), np.uint8)) > 0
    handle = cv2.dilate(((L > o["handle_L"]) & (sat < 20)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    pu = person.astype(np.uint8)
    near_h = cv2.dilate(handle.astype(np.uint8), np.ones((121, 121), np.uint8)) > 0   # the cable only runs off the handle
    cable = (L < 30) & (sat < 10) & ~(cv2.erode(pu, np.ones((15, 15), np.uint8)) > 0) & near_h
    skin = person & ~shorts & ~handle & ~cable
    w = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 1.5) * person
    sk = skin.astype(np.float32)
    base = cv2.GaussianBlur(L * sk, (0, 0), 14) / np.maximum(cv2.GaussianBlur(sk, (0, 0), 14), 1e-3)
    det = L - base
    nb = o["T"] + o["k"] * (base - o["T"])
    Ln = np.maximum(nb + det, o["floor"] + 0.3 * det.clip(-40, 0))
    chroma = (im - L[..., None]) * (0.0 if o["deglow"] else 1.0)
    out = im * (1 - w[..., None]) + (Ln[..., None] + chroma) * w[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[~person] = img_bgr[~person]          # machine / backdrop byte-identical
    return out, dict(skin=skin, shorts=shorts, handle=handle, person=person)


def reframe(img, scale):
    """whole-frame zoom-out about the centre (same for every frame, no cut-out); the border is the replicated edge
    (pure backdrop on a fixed-camera plate - check edges_are_backdrop first)."""
    if abs(scale - 1.0) < 1e-6:
        return img
    H, W = img.shape[:2]
    M = np.float32([[scale, 0, (1 - scale) * W / 2], [0, scale, (1 - scale) * H / 2]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def edges_are_backdrop(empty_bgr, tol=16.0):
    L = lum(empty_bgr.astype(np.float32)); bg = float(np.median(L[:200, W_ // 3:2 * W_ // 3])) if (W_ := L.shape[1]) else 0
    e = np.concatenate([L[:, :3].ravel(), L[:, -3:].ravel(), L[:3].ravel(), L[-3:].ravel()])
    return float(np.abs(e - bg).max()) <= tol


def relight_clip(src, empty_png, dst, opts=None, kp_sigma=1.5, scale=1.0):
    """Relight every frame of a clip (lossless FFV1 .avi out). Keypoints: MediaPipe per frame, missing frames
    interpolated, temporally smoothed (Gaussian sigma kp_sigma frames) so the capsule never jitters."""
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    em = cv2.imread(empty_png)
    assert em.shape == frames[0].shape, (em.shape, frames[0].shape)
    if scale != 1.0:
        assert edges_are_backdrop(em), "reframe needs a pure-backdrop frame edge (machine touches the edge)"
    kps = [detect_pose(f) for f in frames]
    idx = [i for i, k in enumerate(kps) if k is not None]
    K = np.stack([kps[i] for i in idx])
    allk = np.stack([np.stack([np.interp(np.arange(len(frames)), idx, K[:, j, c]) for c in range(2)], -1) for j in range(33)], 1)
    if kp_sigma > 0:
        r = int(3 * kp_sigma + 0.5); kern = np.exp(-0.5 * (np.arange(-r, r + 1) / kp_sigma) ** 2); kern /= kern.sum()
        pad = np.pad(allk, ((r, r), (0, 0), (0, 0)), mode="edge")
        allk = np.stack([np.tensordot(kern, pad[t:t + 2 * r + 1], axes=(0, 0)) for t in range(len(frames))])
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"FFV1"), fps, (frames[0].shape[1], frames[0].shape[0]))
    stats = []
    for t, f in enumerate(frames):
        o_, m = relight(f, em, allk[t], opts)
        vw.write(reframe(o_, scale))
        if t % 10 == 0 or t == len(frames) - 1:
            stats.append(dict(t=t, skin=int(m["skin"].sum()), after_p5_50_90=np.percentile(lum(o_.astype(np.float32))[m["skin"]], [5, 50, 90]).round(1).tolist(),
                              outside_changed=int((np.abs(o_.astype(int) - f.astype(int)).max(2)[~m["person"]] > 0).sum())))
    vw.release()
    return dict(n=len(frames), missed_pose=len(frames) - len(idx), stats=stats)


if __name__ == "__main__":
    import argparse, json, sys
    sys.path.insert(0, "/workspace/daily_info_reels")
    ap = argparse.ArgumentParser(description="relight the mannequin in every frame of a clip (fixed camera, empty plate)")
    ap.add_argument("src"); ap.add_argument("empty"); ap.add_argument("dst"); ap.add_argument("--scale", type=float, default=1.0)
    for k_, v_ in DEFAULTS.items():
        if isinstance(v_, bool):
            ap.add_argument("--" + k_, action="store_true")
        else:
            ap.add_argument("--" + k_, type=float, default=v_)
    a = ap.parse_args()
    print(json.dumps(relight_clip(a.src, a.empty, a.dst, {k_: getattr(a, k_) for k_ in DEFAULTS}, scale=a.scale)))
