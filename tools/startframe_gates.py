"""t609 start-frame colour gates (D17/D18) for a generated still: figure-backdrop, plates (ALL faces), bench, flat backdrop.
usage: python3 tools/startframe_gates.py <img> [...]  -> prints JSON per image"""
import json, sys
import cv2
import numpy as np


def lum(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def gates(path):
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (1920, 1080):
        im = cv2.resize(im, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
    f = im.astype(np.float32)
    L = lum(f)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    # backdrop: border-connected flat region near the border median colour
    bc = np.median(np.concatenate([f[:, :30].reshape(-1, 3), f[:, -30:].reshape(-1, 3), f[:30].reshape(-1, 3)]), axis=0)
    near = (np.linalg.norm(cv2.GaussianBlur(f, (0, 0), 3) - bc, axis=-1) < 14).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(near, connectivity=4)
    ids = np.unique(np.concatenate([lab[:, :5].ravel(), lab[:, -5:].ravel(), lab[:5].ravel()]))
    bk = np.isin(lab, ids[ids > 0])
    bk = cv2.erode(bk.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    bgc = np.median(f[bk], axis=0)
    # backdrop flatness: 120 px block medians
    meds = [float(np.median(L[y:y + 120, x:x + 120][bk[y:y + 120, x:x + 120]])) for y in range(0, 1920, 120) for x in range(0, 1080, 120)
            if bk[y:y + 120, x:x + 120].sum() > 6000]
    span = max(meds) - min(meds)
    farb = cv2.erode(bk.astype(np.uint8), np.ones((121, 121), np.uint8)) > 0   # >= 60 px clear of any object
    medf = [float(np.median(L[y:y + 120, x:x + 120][farb[y:y + 120, x:x + 120]])) for y in range(0, 1920, 120) for x in range(0, 1080, 120)
            if farb[y:y + 120, x:x + 120].sum() > 6000]
    span_far = max(medf) - min(medf) if medf else 99.0
    obj = ~cv2.dilate(bk.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    # plates: blue blobs -> convex hull per blob = ALL faces (lit tops included); distance per pixel inside the hull
    blue = ((b - np.maximum(r, g)) > 25) & obj
    blue = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
    hull = np.zeros(L.shape, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 1500:
            ys, xs = np.where(lab == i)
            cv2.fillConvexPoly(hull, cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)), 1)
    hull = cv2.erode(hull, np.ones((7, 7), np.uint8)) > 0
    # steel handle / gold rim inside hull are not plate faces
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
    gold = (hsv[..., 0] > 10) & (hsv[..., 0] < 35) & (hsv[..., 1] > 60)
    steel = (hsv[..., 1] < 40) & (L > 110)
    face = hull & ~gold & ~steel
    pd = np.linalg.norm(f[face] - bgc, axis=-1) if face.sum() else np.zeros(1)
    # a FACE is an area: dark pixels that survive a 5x5 opening (thin seams between faces / inside the rim ring and
    # the handle collar are edges, not faces) -> the gate fraction; the raw per-pixel fraction is reported too
    badm = face & (np.linalg.norm(f - bgc, axis=-1) < 40)
    bad_area = cv2.morphologyEx(badm.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    # skin: neutral grey object, not plate, not shorts (black), not bench (warm)
    warm = (r - b) > 6
    glow = (np.maximum(r - 0.5 * (g + b), g - 0.5 * (r + b)) > 25)
    nh = ~cv2.dilate(hull.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    skin = obj & nh & ~warm & ~glow & (L > 45) & (np.abs(r - b) <= 6) & ((b - np.maximum(r, g)) < 15)
    bench = obj & nh & warm & ~glow & (L > 60) & (hsv[..., 1] < 90)
    skc = np.median(f[skin], axis=0)
    bnc = np.median(f[bench], axis=0) if bench.sum() > 2000 else np.array([np.nan] * 3)
    res = dict(img=path, backdrop_rgb=bgc.round(0).tolist(), backdrop_span=round(span, 1), backdrop_span_far=round(span_far, 1), blocks=len(meds),
               fig=round(float(np.percentile(L[skin], 75) - lum(bgc[None])[0]), 1), skin_rgb=skc.round(0).tolist(),
               skin_p50=round(float(np.median(L[skin])), 1), skin_p75=round(float(np.percentile(L[skin], 75)), 1),
               plate_px=int(face.sum()), plate_vs_bg_p05=round(float(np.percentile(pd, 5)), 1), plate_vs_bg_med=round(float(np.median(pd)), 1),
               plate_px_below40_raw=round(float((pd < 40).mean()) * 100, 2),
               plate_faces_below40=round(100.0 * float(bad_area.sum()) / max(1, int(face.sum())), 2),
               bench_px=int(bench.sum()), bench_rgb=bnc.round(0).tolist(),
               bench_vs_bg=round(float(np.linalg.norm(bnc - bgc)), 1), bench_vs_skin=round(float(np.linalg.norm(bnc - skc)), 1))
    res["pass"] = bool(res["fig"] >= 33 and res["backdrop_span_far"] <= 4 and res["plate_px"] > 3000 and res["plate_faces_below40"] <= 2.0
                       and res["bench_vs_bg"] >= 15 and res["bench_vs_skin"] >= 10)
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(gates(p)))
