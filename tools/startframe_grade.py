"""t609 global grade for generated start frames (no masks, no per-region paint):
1) smooth low-frequency flat field: the backdrop level (border-connected flat region, normalized-conv sigma 120 px)
   is divided out -> no vignette; one smooth gain image (0.6..1.6) applied to the whole frame;
2) one global tone curve on luminance (RGB scaled by the same ratio): backdrop -> BG_L, skin p50 -> Day 4 p50 (64),
   highlights kept; darks below the backdrop compressed toward 6.
usage: python3 tools/startframe_grade.py <in> <out.png> <bg_lum> <skin_p50> [bg_rgb e.g. 35,39,43] [skin_p75_lift]"""
import sys
import cv2
import numpy as np
sys.path.insert(0, "tools")
from startframe_gates import lum


def grade(src, dst, bg_t=20.0, sk_t=64.0, bg_rgb=None, sk_hi=None):
    im = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (1080, 1920), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    L = lum(im)
    bc = np.median(np.concatenate([im[:, :30].reshape(-1, 3), im[:, -30:].reshape(-1, 3), im[:30].reshape(-1, 3)]), axis=0)
    near = (np.abs(cv2.GaussianBlur(L, (0, 0), 3) - cv2.GaussianBlur(L, (0, 0), 60)) < 6) & \
           (np.linalg.norm(cv2.GaussianBlur(im, (0, 0), 3) - bc, axis=-1) < 30)
    n, lab, st, _ = cv2.connectedComponentsWithStats(near.astype(np.uint8), connectivity=4)
    ids = np.unique(np.concatenate([lab[:, :5].ravel(), lab[:, -5:].ravel(), lab[:5].ravel()]))
    bk = np.isin(lab, ids[ids > 0]).astype(np.float32)
    s = 60.0   # low-frequency (60 px) so soft contact shadows on the backdrop flatten too
    small = lambda a: cv2.resize(a, (270, 480), interpolation=cv2.INTER_AREA)
    B = cv2.GaussianBlur(small(L * bk), (0, 0), s / 4) / np.maximum(cv2.GaussianBlur(small(bk), (0, 0), s / 4), 1e-3)
    B0 = float(np.median(L[bk > 0]))
    Gs = cv2.resize(np.clip(B0 / np.maximum(B, 1.0), 0.6, 2.2).astype(np.float32), (1080, 1920), interpolation=cv2.INTER_CUBIC)
    # the gain only matters where the backdrop is: blend to 1 away from it with a very wide smooth weight
    wbk = cv2.resize(cv2.GaussianBlur(small(bk), (0, 0), 6), (1080, 1920), interpolation=cv2.INTER_CUBIC)
    G = 1.0 + (Gs - 1.0) * np.clip(wbk * 1.4, 0, 1)
    f = im * G[..., None]
    L1 = lum(f)
    obj = (bk < 0.5) & (L1 > B0 + 15)
    sk50 = float(np.median(L1[obj & (np.abs(f[..., 0] - f[..., 2]) < 6)]))
    xs = [0.0, B0 * 0.5, B0, sk50, 255.0]
    ys = [3.0, bg_t * 0.5, bg_t, sk_t, 245.0]
    if bg_rgb is not None:   # v2: darks keep a gentle slope under the studio level (shorts / plate shadows stay dark)
        ys = [4.0, bg_t * 0.42, bg_t, sk_t, 245.0]
    if sk_hi is not None:    # optional: lift ONLY the skin highlights (p75 region) - backdrop knot untouched
        sk75 = float(np.percentile(L1[obj & (np.abs(f[..., 0] - f[..., 2]) < 6)], 75))
        xs = [0.0, B0 * 0.5, B0, sk50, sk75, 255.0]
        ys = ys[:4] + [float(sk_hi), 245.0]
    L2 = np.interp(L1, xs, ys)
    out = f * (L2 / np.maximum(L1, 1.0))[..., None]
    if bg_rgb is None:
        # v1 (dark): global cool tint of the darks (a function of luminance only)
        wd = np.clip((bg_t + 12.0 - L2) / 12.0, 0.0, 1.0)[..., None]
        out = out + np.array([-1.5, 0.5, 4.0], np.float32) * wd
    else:
        # v2: Day 4 studio colour. A global 3D colour function (no mask): LOW-saturation pixels near the backdrop
        # luminance take the studio chroma (bg_rgb - its lum); saturated pixels (cobalt plates, glow) and pixels far
        # from the backdrop level (skin, bench, shorts) are untouched.
        bgr = np.asarray(bg_rgb, np.float32)
        tint = bgr - float(lum(bgr[None])[0])
        sat = np.linalg.norm(out - L2[..., None], axis=-1)
        w = np.exp(-((L2 - bg_t) / 10.0) ** 2) * np.clip((16.0 - sat) / 8.0, 0.0, 1.0)
        out = out * (1 - w[..., None]) + (L2[..., None] + tint) * w[..., None]
        # hue-qualified global colour function (a 3D LUT, no mask): saturated BLUE only (cobalt plates; the studio
        # tint is far below the threshold) keeps its hue, chroma x1.3-2.5 (more on the shaded faces) and a luminance floor 44, so the shaded plate
        # faces stay cobalt instead of sinking toward the charcoal backdrop
        bl = np.clip(((out[..., 2] - np.maximum(out[..., 0], out[..., 1])) - 9.0) / 10.0, 0.0, 1.0)[..., None]
        Lo = lum(out)[..., None]
        k_ = 1.3 + 1.2 * np.clip((54.0 - Lo) / 22.0, 0.0, 1.0)   # shaded (dark) cobalt faces get the most chroma
        boosted = np.maximum(Lo, 44.0) + (out - Lo) * k_
        out = out * (1 - bl) + boosted * bl
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return dict(B0=round(B0, 1), skin_p50_in=round(sk50, 1), gain_min=round(float(Gs.min()), 2), gain_max=round(float(Gs.max()), 2))


if __name__ == "__main__":
    a = sys.argv
    kw = {}
    if len(a) > 5:
        kw["bg_rgb"] = [float(x) for x in a[5].split(",")]
    if len(a) > 6:
        kw["sk_hi"] = float(a[6])
    print(grade(a[1], a[2], float(a[3]), float(a[4]), **kw))
