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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp import neutral_lapvar   # noqa: E402

def skin_lock_lut(lum_skin):
    """Monotone luminance map whose skin percentiles land on the Day 4 lock."""
    lk = SPEC["mannequin"]["skin_lock_all_steps"]
    src = np.percentile(lum_skin, [1, 10, 50, 75, 90, 99.5])
    dst = np.array([47.0, 54.0, lk["p50"], lk["p75"], lk["p90"], lk["max"] - 8.0])
    src = np.maximum.accumulate(src + np.arange(6) * 1e-3)
    xs = np.concatenate([[0.0], src, [255.0]]); ys = np.concatenate([[40.0], dst, [lk["max"]]])
    return lambda L: np.interp(L, xs, ys)

def grade_frame(beauty, mask, cfg, fi, n, kp=None, icons=None, ui_on=True):
    G = cfg["grade"]; W, H = G["out_res"]
    b = cv2.resize(beauty, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rgb = b[..., [2, 1, 0]]; fg = np.clip(b[..., 3] / 255.0, 0, 1)
    body = np.clip(m[..., 3] / 255.0, 0, 1); body = np.minimum(body, fg)
    regs = m[..., [2, 1, 0]] / 255.0                                   # R lats, G rhomboids, B mid trap; white = briefs
    garment = np.clip((regs.min(-1) - 0.35) / 0.3, 0, 1) * body       # briefs mesh: emission white in the mask pass
    regs = regs * (1 - garment[..., None]) * (regs.max(-1, keepdims=True) - regs.min(-1, keepdims=True) > 0.25)
    reg_any = np.clip(regs.max(-1), 0, 1) * body
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    da = float(G.get("detail_amount", 0.0))
    if da > 0:   # detail pass: unsharp mask of the render's own surface (fibre / sculpt relief), body only
        hp = lum - cv2.GaussianBlur(lum, (0, 0), float(G.get("detail_sigma", 1.6)))
        bm = cv2.erode((body > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        d = da * hp * bm; lum = lum + d; rgb = np.clip(rgb + d[..., None], 0, 255)
    if cfg.get("garment", {}).get("enabled"):
        shorts = garment.astype(np.float32)                              # the briefs ARE the shorts (Day 4 matte band)
    else:
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
    out8 = np.clip(out, 0, 255).astype(np.uint8)
    ui = {}
    if kp is not None and cfg.get("layout", {}).get("callout"):
        out8, ui["callout"] = draw_torso_callout(out8, shorts, kp, cfg)
    if not ui_on:
        pass
    elif icons is not None:
        gl.LAST_MASKS.pop("eq_front", None)
        out8 = gl.composite_title_and_preview_v4(out8, fg.astype(np.float32), icons, 0, False, cfg["name"])
        ui["pill"] = [449, 248, 631, 410]                                 # 1-slot pill + active slot frame (b4 geometry)
    else:
        a_t, _ = gl.overlap_alpha(); ty0, ty1 = gl.TITLE_ROI; t_rgb, t_a = gl._title_layer(cfg["name"])
        eff = t_a * (1.0 - fg[ty0:ty1, :, None] * (1.0 - a_t)); o_ = out8.astype(np.float32)
        o_[ty0:ty1] = o_[ty0:ty1] * (1 - eff) + t_rgb * eff; out8 = np.clip(o_, 0, 255).astype(np.uint8)
    ty0, ty1 = gl.TITLE_ROI; t_rgb, t_a = gl._title_layer(cfg["name"]); ys_, xs_ = np.nonzero(t_a[..., 0] > 0.1)
    ui["title"] = [int(xs_.min()), int(ty0 + ys_.min()), int(xs_.max()), int(ty0 + ys_.max())]
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
          "sharp_lapvar": round(neutral_lapvar(out8, bodyb)[0], 1),
          "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) else None, "ui": ui}
    gb_ = garment > 0.5
    if gb_.any():
        cl = cv2.morphologyEx(gb_.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        inner = cv2.erode(cl, np.ones((9, 9), np.uint8)) > 0
        st.update({"briefs_px": int(gb_.sum()), "briefs_poke_px": int((inner & ~gb_ & (body > 0.5)).sum()), "green_on_briefs_px": int((gb_ & green).sum())})
    if "callout" in ui:
        c_ = ui["callout"]["text_box"]; st["callout_fig_overlap"] = round(float((fg[c_[1]:c_[3], c_[0]:c_[2]] > 0.5).mean()), 4)
    return out8, st

def draw_torso_callout(rgb8, shorts, kp, cfg):
    """Day 4 callout (gymform.layout.draw_callout_day4 look: crisp laser joint axis + vertex rings, precision leader,
    Pretendard 84/21/16 at x=86, y >= callout_ty_min) for the hinge: vertical reference ray -> hip vertex -> torso.
    The NUMBER is the rig-measured 3D torso angle (rep median), never the 2D projection."""
    from PIL import Image, ImageDraw
    from gymform.ctx import p62, v30, d5
    L = cfg["layout"]["callout"]; v = tuple(kp["hip"]); b = tuple(kp["sh"]); ln = math.hypot(b[0] - v[0], b[1] - v[1])
    a = (v[0], v[1] - 0.62 * ln)                                                 # vertical reference ray
    _, b_ext = p62._extend_segment(v, b, 0.0, 1.14)
    canvas = Image.fromarray(rgb8).convert("RGBA")
    gl.draw_crisp_laser_joint_axis(canvas, main_chains_1x=[[a, v, b_ext]], aux_ext_chains_1x=[], vertex_pts_1x=[v, b], is_wrong=False)
    laser = np.array(canvas.convert("RGB")).astype(np.float32); g = np.clip(shorts * 0.85, 0, 0.85)[..., None]
    base = Image.fromarray(np.clip(laser * (1 - g) + rgb8.astype(np.float32) * g, 0, 255).astype(np.uint8)).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0)); rd = ImageDraw.Draw(ov); col = (32, 245, 130)
    tgt = (int(0.48 * v[0] + 0.52 * b[0]), int(0.48 * v[1] + 0.52 * b[1]))
    big, tx = 84, 86; ty = int(np.clip(tgt[1] - 24 - 76, int(SPEC["ui"].get("callout_ty_min", 600)), 1460))
    ey = ty + 76; ex = max(286, tgt[0] - 42)
    d5.bcd._draw_precision_leader(rd, tgt, (ex, ey), (278, ey), col)
    deg = f"{int(kp['deg_shown'])}\u00b0"; fb, f1, f2 = v30.f_pretendard(big, "extrabold"), v30.f_pretendard(21, "extrabold"), v30.f_pretendard(16, "bold")
    rd.text((tx + 2, ty + 3), deg, font=fb, fill=(0, 0, 0, 210)); rd.text((tx, ty), deg, font=fb, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + big + 6), L["l1"], font=f1, fill=(*col, 255)); rd.text((tx + 2, ty + big + 33), L["l2"], font=f2, fill=(215, 222, 232, 235))
    boxes = [rd.textbbox((tx, ty), deg, font=fb), rd.textbbox((tx + 2, ty + big + 6), L["l1"], font=f1), rd.textbbox((tx + 2, ty + big + 33), L["l2"], font=f2)]
    tb = [min(q[0] for q in boxes), min(q[1] for q in boxes), max(q[2] for q in boxes), max(q[3] for q in boxes)]
    xs_ = [a[0], v[0], b_ext[0], tgt[0], ex, 278]; ys_ = [a[1], v[1], b_ext[1], tgt[1], ey]
    ang2d = math.degrees(math.atan2(abs(b[0] - v[0]), v[1] - b[1]))
    info = {"deg_shown": int(kp["deg_shown"]), "text_box": [int(q) for q in tb], "lines_box": [int(min(xs_)), int(min(ys_)), int(max(xs_)), int(max(ys_))],
            "angle_2d_deg": round(ang2d, 1), "target": list(tgt)}
    return np.array(Image.alpha_composite(base, ov).convert("RGB")), info

def make_icons(graded_rgb, bbox):
    """Preview-pill thumbnail (Day 4: each slot = that step's peak frame): the figure crop of the peak frame."""
    x0, y0, x1, y1 = bbox; y1 = y0 + 0.62 * (y1 - y0)                    # Day 4 slots show the working torso, not the feet
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h = max(y1 - y0, (x1 - x0) * 158 / 134) * 1.08; w = h * 134 / 158
    X0, Y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2)); X1, Y1 = int(min(graded_rgb.shape[1], cx + w / 2)), int(min(graded_rgb.shape[0], cy + h / 2))
    crop = cv2.resize(graded_rgb[Y0:Y1, X0:X1], (224, 264), interpolation=cv2.INTER_AREA)
    return [(crop, (crop * 0.55).astype(np.uint8))]

def main():
    rdir, odir, cfgp = sys.argv[1:4]; cfg = json.load(open(cfgp)); os.makedirs(odir, exist_ok=True)
    cv2.setNumThreads(2)
    fr = sorted(glob.glob(os.path.join(rdir, "f_*.png"))); stats = []
    CH = json.load(open(os.path.join(rdir, "checks.json"))); R = CH["frames"]; KPW = (CH.get("kp_res") or [None])[0]; KP = {r_["f"]: r_.get("kp2d") for r_ in R}
    tor = np.array([r_["torso_deg"] for r_ in R]); shown = int(round(float(np.median(tor))))
    pf = R[int(np.argmin([r_["elbow_deg"] for r_ in R]))]["f"]          # peak frame number (1-based)
    nums = [int(os.path.basename(q)[2:6]) for q in fr]; peak = int(np.argmin([abs(n_ - pf) for n_ in nums]))
    icons = None
    if cfg.get("layout", {}).get("preview_pill"):
        fp = fr[peak]; b0 = cv2.imread(fp, cv2.IMREAD_UNCHANGED); g0, st0 = grade_frame(b0, cv2.imread(fp.replace("/f_", "/m_"), cv2.IMREAD_UNCHANGED), cfg, peak, len(fr), ui_on=False)
        icons = make_icons(g0, st0["bbox"])
    for i, fp in enumerate(fr):
        mp = fp.replace("/f_", "/m_"); fnum = int(os.path.basename(fp)[2:6])
        kp = dict(KP.get(fnum) or {}); kp = (dict(kp, deg_shown=shown) if kp else None)
        bty = cv2.imread(fp, cv2.IMREAD_UNCHANGED); msk = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if kp:   # kp2d is in render pixels; the grade works at out_res
            k_ = cfg["grade"]["out_res"][0] / (KPW or bty.shape[1]); kp = {kk: ([vv[0] * k_, vv[1] * k_] if isinstance(vv, list) else vv) for kk, vv in kp.items()}
        out, st = grade_frame(bty, msk, cfg, i, len(fr), kp, icons); st["src"] = os.path.basename(fp)
        cv2.imwrite(os.path.join(odir, "g_" + os.path.basename(fp)[2:]), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)); stats.append(st)
    json.dump({"frames": stats, "callout_shown_deg": shown, "torso_median_deg": round(float(np.median(tor)), 2)}, open(os.path.join(odir, "look.json"), "w"), indent=0)
    print("graded", len(fr))

if __name__ == "__main__":
    main()
