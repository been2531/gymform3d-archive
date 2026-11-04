#!/usr/bin/env python3
"""Day 4 grade for rig3d renders (Routinizer v2). Pure numpy/cv2 + the locked gymform modules, no network, no AI.
Per frame: beauty RGBA (transparent film) + muscle-mask RGBA (R lats, G rhomboids, B mid trap, dumbbell holdout) ->
  1. upscale to 1080x1920
  2. skin lock: body luminance percentile-mapped to the Day 4 lock (p50 64 / p75 75 / p90 88, gymform_spec.json),
     Day 4 cool tint; shorts held in the Day 4 shorts band (24-40)
  3. glow: gymform.muscle.day4_fibre (Day 4 GOLD #1 fibre shader) with its activation = the RENDERED muscle mask
     (exact regions, no per-frame cut-out), green (correct form)
  4. flat Day 4 backdrop (35,39,43) + soft floor contact shadow
  5. Day 4 Kaushan title (gymform.layout._title_layer), see-through alpha 0.40 over the figure (t638u rule)
Writes g_####.png + look.json (per-frame look stats used by qc3d.py).
usage: python3 tools/rig3d/grade3d.py RENDER_DIR OUT_DIR CFG.json"""
import glob, json, math, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from gymform.ctx import SPEC                      # noqa: E402  (locked spec)
from gymform import muscle as gm, layout as gl   # noqa: E402

def skin_lock_lut(lum_skin):
    """Monotone luminance map whose skin percentiles land on the Day 4 lock."""
    lk = SPEC["mannequin"]["skin_lock_all_steps"]
    src = np.percentile(lum_skin, [1, 10, 50, 75, 90, 99.5])
    dst = np.array([47.0, 54.0, lk["p50"], lk["p75"], lk["p90"], lk["max"] - 8.0])
    src = np.maximum.accumulate(src + np.arange(6) * 1e-3)
    xs = np.concatenate([[0.0], src, [255.0]]); ys = np.concatenate([[40.0], dst, [lk["max"]]])
    return lambda L: np.interp(L, xs, ys)

def grade_frame(beauty, mask, cfg, fi, n):
    G = cfg["grade"]; W, H = G["out_res"]
    b = cv2.resize(beauty, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rgb = b[..., [2, 1, 0]]; fg = np.clip(b[..., 3] / 255.0, 0, 1)
    body = np.clip(m[..., 3] / 255.0, 0, 1); body = np.minimum(body, fg)
    regs = m[..., [2, 1, 0]] / 255.0                                   # R lats, G rhomboids, B mid trap
    reg_any = np.clip(regs.max(-1), 0, 1) * body
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    shorts = ((lum < 28.0) & (body > 0.5) & (reg_any < 0.05)).astype(np.float32)
    shorts = cv2.GaussianBlur(cv2.morphologyEx(shorts, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), (0, 0), 1.0)
    skin = (body > 0.9) & (shorts < 0.2) & (reg_any < 0.05)
    lut = skin_lock_lut(lum[skin]) if skin.sum() > 2000 else (lambda L: L)
    tone = lut(lum)
    lo, hi = SPEC["mannequin"]["shorts_lum_range"]; tone = tone * (1 - shorts) + np.clip(lum * 0.3 + lo - 2.0, lo, hi) * shorts
    tr, tg, tb = SPEC["mannequin"]["skin_lock_all_steps"].get("tint_rgb", [-3, 1, 3])
    mono = np.stack([tone + tr, tone + tg, tone + tb], -1)
    # Day 4 fibre shader: activation from the rendered mask (Day 4 native glow level ~100)
    F = {"r": rgb[..., 0], "g": rgb[..., 1], "b": rgb[..., 2], "act_dom": cv2.GaussianBlur(reg_any, (0, 0), 1.2) * 105.0}
    graded, core_w = gm.day4_fibre(F, mono, body, shorts, G.get("glow", "right") == "wrong", fi / max(n, 1))
    # composite: body = graded, dumbbell = beauty (untouched), backdrop flat Day 4
    bw = np.clip(body / np.maximum(fg, 1e-4), 0, 1)[..., None]
    fg_rgb = graded * bw + rgb * (1 - bw)
    bg = np.empty((H, W, 3), np.float32); bg[:] = G.get("backdrop_rgb", SPEC["background"]["grade_to_rgb"])
    ys, xs = np.nonzero(fg > 0.5)
    if len(ys):                                                          # soft contact shadow under the feet
        fy = ys.max(); sel = ys > fy - 40; cx = xs[sel].mean(); sw = max(60.0, float(np.ptp(xs[sel])) * 0.9 + 60)
        YY, XX = np.mgrid[:H, :W].astype(np.float32)
        sh = np.exp(-(((XX - cx) / sw) ** 2 + ((YY - fy - 6) / 18.0) ** 2) * 2.2); bg *= (1 - 0.38 * sh)[..., None]
    out = bg * (1 - fg[..., None]) + fg_rgb * fg[..., None]
    # title (see-through over the figure)
    a_t, _ = gl.overlap_alpha(); ty0, ty1 = gl.TITLE_ROI; t_rgb, t_a = gl._title_layer(cfg["name"])
    eff = t_a * (1.0 - fg[ty0:ty1, :, None] * (1.0 - a_t))
    out[ty0:ty1] = out[ty0:ty1] * (1 - eff) + t_rgb * eff
    out8 = np.clip(out, 0, 255).astype(np.uint8)
    # look stats (on the graded frame, before the title rows are judged)
    o = out8.astype(np.float32); L = 0.299 * o[..., 0] + 0.587 * o[..., 1] + 0.114 * o[..., 2]
    gdom = o[..., 1] - np.maximum(o[..., 0], o[..., 2]); rdom = o[..., 0] - np.maximum(o[..., 1], o[..., 2])
    bodyb = body > 0.5; green = bodyb & (gdom > 30)
    regd = cv2.dilate((reg_any > 0.15).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    skin_q = bodyb & (shorts < 0.2) & (core_w < 0.03) & (L > 46)
    far = (cv2.dilate((fg > 0.02).astype(np.uint8), np.ones((61, 61), np.uint8)) == 0); far[ty0 - 10:ty1 + 10] = False
    er = cv2.erode(bodyb.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    lap = cv2.Laplacian(cv2.cvtColor(out8, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    st = {"f": fi, "body_px": int(bodyb.sum()),
          "skin_p50": round(float(np.percentile(L[skin_q], 50)), 1) if skin_q.sum() > 500 else None,
          "skin_p75": round(float(np.percentile(L[skin_q], 75)), 1) if skin_q.sum() > 500 else None,
          "skin_p90": round(float(np.percentile(L[skin_q], 90)), 1) if skin_q.sum() > 500 else None,
          "skin_max": round(float(np.percentile(L[skin_q], 99.9)), 1) if skin_q.sum() > 500 else None,
          "backdrop_rgb": [round(float(np.median(o[..., c][far])), 1) for c in range(3)] if far.sum() > 1000 else None,
          "glow_frac": round(float(green.sum() / max(bodyb.sum(), 1)), 4),
          "glow_in_mask": round(float((green & regd).sum() / max(green.sum(), 1)), 4),
          "red_px": int((bodyb & (rdom > 40)).sum()),
          "region_px": {k: int(((regs[..., i] > 0.5) & bodyb).sum()) for i, k in enumerate(("lats", "rhomboids", "mid_trap"))},
          "sharp_lapvar": round(float(lap[er].var()), 1) if er.sum() > 1000 else 0.0,
          "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) else None}
    return out8, st

def main():
    rdir, odir, cfgp = sys.argv[1:4]; cfg = json.load(open(cfgp)); os.makedirs(odir, exist_ok=True)
    cv2.setNumThreads(2)
    fr = sorted(glob.glob(os.path.join(rdir, "f_*.png"))); stats = []
    for i, fp in enumerate(fr):
        mp = fp.replace("/f_", "/m_")
        bty = cv2.imread(fp, cv2.IMREAD_UNCHANGED); msk = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        out, st = grade_frame(bty, msk, cfg, i, len(fr)); st["src"] = os.path.basename(fp)
        cv2.imwrite(os.path.join(odir, "g_" + os.path.basename(fp)[2:]), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)); stats.append(st)
    json.dump({"frames": stats}, open(os.path.join(odir, "look.json"), "w"), indent=0)
    print("graded", len(fr))

if __name__ == "__main__":
    main()
