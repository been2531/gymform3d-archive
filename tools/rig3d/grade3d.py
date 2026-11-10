#!/usr/bin/env python3
"""Day 4 grade for rig3d renders (Routinizer v2). Pure numpy/cv2 + the locked gymform modules, no network, no AI.
Per frame: beauty RGBA (transparent film) + muscle-mask RGBA (R lats, G rhomboids, B mid trap, dumbbell holdout) ->
  1. upscale to 1080x1920
  2. skin lock: look preset charcoal (soft Day 4) or clay (p50 ~150), high-pass keeps lighting form;
     shorts held in the Day 4 shorts band (24-40)
  3. glow: 3D emission from the beauty pass + day4_fibre at fibre_amount (fascicle direction) +
     muscle-clamped bloom (callouts drawn after, so they never bloom)
  4. flat Day 4 backdrop (35,39,43) + soft floor contact shadow
  5. Day 4 Kaushan title (gymform.layout._title_layer), see-through alpha 0.40 over the figure (t638u rule)
Writes g_####.png + look.json (per-frame look stats used by qc3d.py).
usage: python3 tools/rig3d/grade3d.py RENDER_DIR OUT_DIR CFG.json"""
import glob, json, math, os, sys, time
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from gymform.ctx import SPEC                      # noqa: E402  (locked spec)
from gymform import muscle as gm, layout as gl   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp import neutral_lapvar   # noqa: E402

# P0 look presets (t728u). charcoal = Day 4 identity with extra highlight headroom + high-pass so lighting survives.
# clay = lifted mannequin near luma p50 140-160 (Demic-like volume, gymform green glow kept).
LOOKS = {
    "charcoal": {"dst": [64.0, 74.0, 84.0, 102.0, 128.0, 178.0], "contrast": 0.58, "tint_rgb": [-3, 1, 3], "lo": 56.0, "hi": 255.0},
    "clay":     {"dst": [88.0, 122.0, 150.0, 172.0, 198.0, 230.0], "contrast": 0.55, "tint_rgb": [-1, 0, 2], "lo": 70.0, "hi": 248.0},
}

def skin_lock_lut(lum_skin, look):
    """Monotone luminance map whose skin percentiles land on the active look, plus room for rim highlights."""
    src = np.percentile(lum_skin, [1, 10, 50, 75, 90, 99.5])
    dst = np.array(look["dst"], dtype=np.float64)
    src = np.maximum.accumulate(src + np.arange(6) * 1e-3)
    xs = np.concatenate([[0.0], src, [255.0]]); ys = np.concatenate([[float(look["lo"])], dst, [255.0]])
    return lambda L: np.interp(L, xs, ys)

def muscle_bloom(rgb, reg_any, body, amount):
    """Fog-glow equivalent, clamped to a dilated muscle mask so UI drawn later never blooms."""
    if amount <= 0:
        return rgb
    w = np.clip(reg_any, 0, 1) * (body > 0.45).astype(np.float32)
    glow = rgb * w[..., None]
    bloom = np.zeros_like(rgb)
    for sig, wt in ((6.0, 0.42), (16.0, 0.33), (34.0, 0.22)):
        bloom += wt * cv2.GaussianBlur(glow, (0, 0), sig)
    dil = cv2.dilate((w > 0.12).astype(np.uint8), np.ones((21, 21), np.uint8)).astype(np.float32)
    clamp = cv2.GaussianBlur(dil, (0, 0), 5.0)
    inner = cv2.erode((w > 0.18).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    inner = cv2.GaussianBlur(inner, (0, 0), 2.0)
    wtmap = clamp * (0.20 + 0.80 * (1.0 - inner))   # keep interior fibres; bloom the soft muscle edge
    return np.clip(rgb + bloom * float(amount) * wtmap[..., None], 0, 255)

def studio_floor_mask(fg, body, rgb):
    """Opaque ground-plane pixels connected to the bottom edge. The receding studio floor spans
    the full width (and would otherwise become the figure bbox). Blue dumbbell heads are excluded
    so a 2D overlap with the floor cannot swallow the prop."""
    R, Gc, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = (B > R + 18.0) & (B > Gc + 12.0)
    seedable = (fg > 0.5) & (body < 0.2) & (~blue)
    if not np.any(seedable):
        return np.zeros(fg.shape, dtype=bool)
    _n, labels = cv2.connectedComponents(seedable.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[-16:][seedable[-16:]])
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros(fg.shape, dtype=bool)
    return np.isin(labels, keep)

def floor_quality_stats(L, floor_m, far, fy):
    """Horizon jump (luma step at the top of the floor silhouette) and floor-vs-backdrop delta.
    A bright studio plane with a hard far edge scores high on both; a dark melted cyclorama scores low."""
    H, W = L.shape
    bg_l = float(np.median(L[far])) if far.sum() > 1000 else 38.0
    yy = np.arange(H)[:, None]
    recede = floor_m & (yy < fy - 30) if fy is not None else floor_m
    if recede.sum() > 200:
        fl = float(np.median(L[recede]))
    elif floor_m.any():
        fl = float(np.median(L[floor_m]))
    else:
        fl = bg_l
    has = floor_m.any(0)
    yscan = np.argmax(floor_m, axis=0)
    xs = np.arange(W)
    above = np.clip(yscan - 5, 0, H - 1)
    below = np.clip(yscan + 5, 0, H - 1)
    jump = L[below, xs] - L[above, xs]
    valid = has & (yscan > 30)
    if fy is not None:
        valid = valid & (yscan < fy - 60)
    jmed = float(np.median(jump[valid])) if valid.any() else 0.0
    return {"floor_bg_delta": round(fl - bg_l, 2), "floor_horizon_jump": round(jmed, 2)}

def glow_quality_stats(o, green):
    """Fibre texture (high-pass std of the green channel inside an eroded glow core) and HSV band."""
    if not np.any(green):
        return {"glow_fibre_hp_std": 0.0, "glow_hue_p50": None, "glow_sat_p50": None}
    inner = cv2.erode(green.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    if inner.sum() < 200:
        inner = green
    gc = o[..., 1]
    hp = gc - cv2.GaussianBlur(gc, (0, 0), 2.5)
    bgr = np.clip(o[..., ::-1], 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return {
        "glow_fibre_hp_std": round(float(hp[inner].std()), 2),
        "glow_hue_p50": round(float(np.percentile(hsv[..., 0][green], 50)), 1),
        "glow_sat_p50": round(float(np.percentile(hsv[..., 1][green], 50)), 1),
    }


def wrap_glow_to_beauty(graded, base, src_l, hp_form, mus_w, reg_any, shorts):
    """Shade 2D fibres with the underlying 3D lighting so the glow wraps the muscle
    volume (darker at grazing/shadow edges) and specular highlights survive.
    Interior troughs are lifted just above the green-detect floor so fibre gaps
    cannot split the connected glow component (clay v3.1 peak cc 0.8996)."""
    w = np.clip(mus_w, 0, 1) * (1.0 - np.clip(shorts, 0, 1))
    if float(w.max()) <= 0:
        return graded
    mus = (reg_any > 0.15) & (shorts < 0.25)
    if int(mus.sum()) > 200:
        lo, hi = np.percentile(src_l[mus], [8, 92])
        wrap = np.clip((src_l - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)
    else:
        wrap = np.clip(src_l / 180.0, 0.0, 1.0)
    wrap = 0.58 + 0.42 * wrap
    glow = graded - base
    graded = base + glow * wrap[..., None]
    hp_pos = np.clip(hp_form, 0.0, None)
    spec = np.clip(hp_pos / 26.0, 0.0, 1.0) * w
    graded = np.clip(graded + (0.50 * hp_pos * spec)[..., None], 0, 255)
    interior = cv2.erode(((reg_any > 0.16) & (shorts < 0.25)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    g, r, b = graded[..., 1], graded[..., 0], graded[..., 2]
    gdom = g - np.maximum(r, b)
    need = interior & (gdom < 36.0)
    lift = (36.0 - gdom) * need.astype(np.float32)
    graded = graded.copy()
    graded[..., 1] = np.clip(g + lift, 0, 255)
    graded[..., 0] = np.clip(r + 0.14 * lift, 0, 255)
    graded[..., 2] = np.clip(b + 0.38 * lift, 0, 255)
    return graded

def snap_glow_hsv(rgb, w, target_h=74.0, target_s=196.0, hue_mix=0.90, sat_mix=0.82):
    """Push glow pixels toward gymform neon green while keeping value (fibre / form)."""
    ww = np.clip(w, 0, 1).astype(np.float32)
    if float(ww.max()) <= 0:
        return rgb
    bgr = np.clip(rgb[..., ::-1], 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = hsv[..., 0] * (1.0 - hue_mix * ww) + target_h * (hue_mix * ww)
    s = hsv[..., 1]
    hsv[..., 1] = np.clip(s + (target_s - s) * sat_mix * ww, 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)[..., ::-1].astype(np.float32)
    return rgb * (1.0 - ww[..., None]) + out * ww[..., None]

def studio_surface_mask(fg, body, rgb):
    """Any non-body, non-dumbbell studio pixel, including faint fade-zone ghosts."""
    R, Gc, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = (B > R + 18.0) & (B > Gc + 12.0)
    return (fg > 0.04) & (body < 0.25) & (~blue)

def melt_studio_floor(out, surface, backdrop, fy, xs_b):
    """Replace every studio-surface pixel (including semi-transparent ghosts) with a dark
    charcoal that is indistinguishable from the backdrop except in a tight patch under the feet."""
    if not np.any(surface) or fy is None:
        return out
    H, W = surface.shape
    bg = np.asarray(backdrop, np.float32).reshape(3)
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    cx = float(np.mean(xs_b)) if xs_b is not None and len(xs_b) else W * 0.5
    dist = np.sqrt(((xx - cx) / 150.0) ** 2 + ((yy - fy) / 95.0) ** 2)
    keep = np.clip(1.0 - dist, 0.0, 1.0) * surface.astype(np.float32)
    dark = bg + np.array([2.5, 2.5, 3.0], np.float32)
    target = dark * keep[..., None] + bg * (1.0 - keep)[..., None]
    mix = np.clip(surface.astype(np.float32), 0, 1)
    return out * (1.0 - mix[..., None]) + target * mix[..., None]

def add_grounding_reflection(out, body, floor_m, fy):
    """Faint flipped-leg reflection on the floor just under the feet (grounding, not a mirror)."""
    if fy is None:
        return out
    H, W = floor_m.shape
    ht = 72
    dest0 = min(H - 2, fy + 3); dest1 = min(H, dest0 + ht)
    src_h = dest1 - dest0
    if src_h < 12:
        return out
    y0 = max(0, fy - src_h)
    src = out[y0:fy][-src_h:].copy()
    if src.shape[0] != src_h:
        return out
    refl = src[::-1] * 0.10 + 4.0
    src_body = (body[y0:fy][-src_h:] > 0.5).astype(np.float32)[::-1]
    w = floor_m[dest0:dest1].astype(np.float32) * (body[dest0:dest1] < 0.5).astype(np.float32)
    w = w * src_body * np.linspace(0.16, 0.02, src_h, dtype=np.float32)[:, None]
    sl = out[dest0:dest1]
    sl[:] = sl * (1.0 - w[..., None]) + refl * w[..., None]
    return out

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
    look_name = str(G.get("look", "charcoal")).lower()
    look = LOOKS.get(look_name, LOOKS["charcoal"]); look_name = look_name if look_name in LOOKS else "charcoal"
    lut = skin_lock_lut(lum[skin], look) if skin.sum() > 2000 else (lambda L: L)
    locked = lut(lum)
    hp = lum - cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 2.5)    # keep key/rim form the lock would otherwise flatten
    tone = np.clip(locked + float(look["contrast"]) * hp, 0, float(look.get("hi", 255)))
    lo, hi = SPEC["mannequin"]["shorts_lum_range"]; tone = tone * (1 - shorts) + np.clip(lum * 0.3 + lo - 2.0, lo, hi) * shorts
    tr, tg, tb = look.get("tint_rgb", SPEC["mannequin"]["skin_lock_all_steps"].get("tint_rgb", [-3, 1, 3]))
    mono = np.stack([tone + tr, tone + tg, tone + tb], -1)
    # 3D form as luma high-pass from the beauty sculpt (not the pale-mint AgX emission color)
    rw = np.clip(reg_any, 0, 1)[..., None] * (1.0 - shorts[..., None])
    src_l = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    hp_form = src_l - cv2.GaussianBlur(src_l, (0, 0), 2.2)
    base = np.clip(mono + (float(G.get("form_hp", 0.90)) * hp_form * rw[..., 0])[..., None], 0, 255)
    emit_keep = float(G.get("emission_keep", 0.20))
    if emit_keep > 0:
        base = base * (1.0 - emit_keep * rw) + rgb * (emit_keep * rw)
    glow_src = str(G.get("glow_source", "post")).lower()
    if glow_src == "blender":
        # Beauty already has per-muscle fibre emission + compositor bloom. Keep it.
        fibre_amt = 0.0
        bloom_amt = 0.0
        emit_keep = float(G.get("emission_keep_blender", 0.90))
        base = base * (1.0 - emit_keep * rw) + rgb * (emit_keep * rw)
        graded = base
        core_w = cv2.GaussianBlur(reg_any, (0, 0), 1.2)
        graded = snap_glow_hsv(graded, (reg_any * (1.0 - shorts)),
                               target_h=52.0 if G.get("glow", "right") == "wrong" else 74.0,
                               target_s=196.0, sat_mix=0.55)
    else:
        fibre_amt = float(G.get("fibre_amount", 0.80))
        F = {"r": rgb[..., 0], "g": rgb[..., 1], "b": rgb[..., 2], "act_dom": cv2.GaussianBlur(reg_any, (0, 0), 1.2) * 105.0}
        graded_full, core_w = gm.day4_fibre(F, base, body, shorts, G.get("glow", "right") == "wrong", fi / max(n, 1))
        graded = base * (1.0 - fibre_amt) + graded_full * fibre_amt
        graded = wrap_glow_to_beauty(graded, base, src_l, hp_form, rw[..., 0], reg_any, shorts)
        graded = snap_glow_hsv(graded, (reg_any * (1.0 - shorts)),
                               target_h=52.0 if G.get("glow", "right") == "wrong" else 74.0,
                               target_s=196.0, sat_mix=0.82)
        graded = muscle_bloom(graded, reg_any * (1.0 - shorts), body, float(G.get("bloom_amount", 0.38)))
    sm = shorts > 0.45
    if np.any(sm):                                          # QC: 0 green pixels on the briefs
        graded[sm] = mono[sm]
    # composite: body = graded, dumbbell + studio floor = beauty (untouched), backdrop flat Day 4
    bw = np.clip(body / np.maximum(fg, 1e-4), 0, 1)[..., None]
    fg_rgb = graded * bw + rgb * (1 - bw)
    bg = np.empty((H, W, 3), np.float32); bg[:] = G.get("backdrop_rgb", SPEC["background"]["grade_to_rgb"])
    ys_b, xs_b = np.nonzero(body > 0.5)
    YY = np.arange(H, dtype=np.float32)[:, None]
    floor_m = studio_floor_mask(fg, body, rgb)
    surface = studio_surface_mask(fg, body, rgb)
    floor_m = floor_m | ((fg > 0.5) & surface)          # leftover disc / cove must not enter the figure bbox
    fig = (fg > 0.5) & ~floor_m & ~surface
    ys, xs = np.nonzero(fig) if fig.any() else (ys_b, xs_b)
    sh = None
    if len(ys_b):                                                          # contact shadow under the BODY feet (also darkens the studio floor)
        fy = ys_b.max(); sel = ys_b > fy - 40; cx = xs_b[sel].mean(); sw = max(60.0, float(np.ptp(xs_b[sel])) * 0.9 + 60)
        YY, XX = np.mgrid[:H, :W].astype(np.float32)
        sh = np.exp(-(((XX - cx) / sw) ** 2 + ((YY - fy - 6) / 18.0) ** 2) * 2.2); bg *= (1 - 0.38 * sh)[..., None]
    out = bg * (1 - fg[..., None]) + fg_rgb * fg[..., None]
    fy = int(ys_b.max()) if len(ys_b) else None
    out = melt_studio_floor(out, surface, bg[0, 0], fy, xs_b if len(ys_b) else None)
    if sh is not None:
        out = out * (1.0 - 0.58 * sh * (body < 0.5))[..., None]
    out = add_grounding_reflection(out, body, surface, fy)
    out8 = np.clip(out, 0, 255).astype(np.uint8)
    base8 = out8                     # look stats are measured on the figure BEFORE the UI (laser/leader/pill have their own QC)
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
    # look stats (graded figure without UI)
    o = base8.astype(np.float32); L = 0.299 * o[..., 0] + 0.587 * o[..., 1] + 0.114 * o[..., 2]
    gdom = o[..., 1] - np.maximum(o[..., 0], o[..., 2]); rdom = o[..., 0] - np.maximum(o[..., 1], o[..., 2])
    bodyb = body > 0.5; green = bodyb & (gdom > 30)
    regd = cv2.dilate((reg_any > 0.15).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    skin_q = bodyb & (shorts < 0.2) & (core_w < 0.03) & (L > 46) & (gdom < 25)
    far = (cv2.dilate((fg > 0.02).astype(np.uint8), np.ones((61, 61), np.uint8)) == 0); far[ty0 - 10:ty1 + 10] = False
    er = cv2.erode(bodyb.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    lap = cv2.Laplacian(cv2.cvtColor(base8, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
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
          "sharp_lapvar": round(neutral_lapvar(base8, bodyb)[0], 1),
          "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) else None, "ui": ui}
    gb_ = garment > 0.5
    if gb_.any():
        cl = cv2.morphologyEx(gb_.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        inner = cv2.erode(cl, np.ones((9, 9), np.uint8)) > 0
        st.update({"briefs_px": int(gb_.sum()), "briefs_poke_px": int((inner & ~gb_ & (body > 0.5)).sum()), "green_on_briefs_px": int((gb_ & green).sum())})
    st["look"] = look_name
    if len(ys_b):
        fy = int(ys_b.max()); x0s, x1s = int(xs_b.min()), int(xs_b.max())
        foot = ys_b > fy - 40
        x0f = int(xs_b[foot].min()) - 16; x1f = int(xs_b[foot].max()) + 16
        xx = np.arange(W)[None, :]; yy = np.arange(H)[:, None]
        under = (yy >= fy + 2) & (yy <= fy + 40) & (xx >= x0f) & (xx <= x1f) & (body < 0.4)
        if under.sum() > 80 and far.sum() > 80:
            st["floor_shadow_delta"] = round(float(np.median(L[far]) - np.median(L[under])), 2)
    fy_ = int(ys_b.max()) if len(ys_b) else None
    st.update(floor_quality_stats(L, floor_m, far, fy_))
    st.update(glow_quality_stats(o, green))
    nlab, lab = cv2.connectedComponents(green.astype(np.uint8), connectivity=8)
    sizes = np.bincount(lab.ravel())
    sizes = sizes[1:] if len(sizes) > 1 else np.array([0])
    st["glow_area_px"] = int(green.sum())
    st["glow_cc"] = int((sizes >= 12).sum())
    st["glow_largest_cc_frac"] = round(float(sizes.max() / max(int(green.sum()), 1)), 4)
    bg_l = float(np.median(L[far])) if far.sum() > 1000 else 38.0
    hip = (kp or {}).get("hip")
    if hip is not None and bodyb.any():
        yy = np.arange(H, dtype=np.float32)[:, None]
        legs = bodyb & (yy > float(hip[1]) + 24.0) & (~green) & (shorts < 0.45)
        if int(legs.sum()) > 200:
            lp = np.percentile(L[legs], [10, 50])
            st["legs_luma_p10"] = round(float(lp[0]), 1)
            st["legs_luma_p50"] = round(float(lp[1]), 1)
            st["legs_vs_backdrop"] = round(float(lp[0] - bg_l), 2)
            st["legs_vs_backdrop_p50"] = round(float(lp[1] - bg_l), 2)
    # v5: figure pixels under the title text box / preview pill (the figure must never sit under the title)
    st["overlay_fig_px"] = {k: int((fg[ui[k][1]:ui[k][3] + 1, ui[k][0]:ui[k][2] + 1] > 0.5).sum()) for k in ("title", "pill") if k in ui}
    if "callout" in ui:
        c_ = ui["callout"]["text_box"]; st["callout_fig_overlap"] = round(float((fig[c_[1]:c_[3], c_[0]:c_[2]]).mean()), 4)
    return out8, st

def draw_torso_callout(rgb8, shorts, kp, cfg):
    """Day 4 callout (gymform.layout.draw_callout_day4 look: crisp laser joint axis + vertex rings, precision leader,
    Pretendard 84/21/16 at x=86, y >= callout_ty_min) for the hinge: vertical reference ray -> hip vertex -> torso.
    The NUMBER is the rig-measured 3D torso angle (rep median), never the 2D projection."""
    from PIL import Image, ImageDraw
    from gymform.ctx import p62, v30, d5
    def _seg_dist(p, p0, p1):
        p = np.asarray(p, float); p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        ab = p1 - p0; t = float(np.clip(((p - p0) @ ab) / (float(ab @ ab) + 1e-12), 0, 1))
        return float(np.linalg.norm(p - (p0 + t * ab)))
    L = cfg["layout"]["callout"]; v = [float(kp["hip"][0]), float(kp["hip"][1])]; b = [float(kp["sh"][0]), float(kp["sh"][1])]
    ln = math.hypot(b[0] - v[0], b[1] - v[1])
    a = [v[0], v[1] - 0.62 * ln]                                                 # vertical reference ray
    _, b_ext = p62._extend_segment(tuple(v), tuple(b), 0.0, 1.14); b_ext = [float(b_ext[0]), float(b_ext[1])]
    tgt = [0.48 * v[0] + 0.52 * b[0], 0.48 * v[1] + 0.52 * b[1]]
    db = kp.get("db"); clear_min = float((cfg.get("grade") or {}).get("callout_clear_min_px", 180)); prop_clear = None
    if db:
        away = -1.0 if db[0] > v[0] else 1.0
        for _ in range(16):
            dmin = min(_seg_dist(db, a, v), _seg_dist(db, v, b_ext))
            if dmin >= clear_min: break
            step = 6.0 * away
            a[0] += step; v[0] += step; b[0] += step; b_ext[0] += step; tgt[0] += step
        prop_clear = min(_seg_dist(db, a, v), _seg_dist(db, v, b_ext))
    a, v, b, b_ext, tgt = tuple(a), tuple(v), tuple(b), tuple(b_ext), (int(tgt[0]), int(tgt[1]))
    canvas = Image.fromarray(rgb8).convert("RGBA")
    gl.draw_crisp_laser_joint_axis(canvas, main_chains_1x=[[a, v, b_ext]], aux_ext_chains_1x=[], vertex_pts_1x=[v, b], is_wrong=False)
    laser = np.array(canvas.convert("RGB")).astype(np.float32); g = np.clip(shorts * 0.85, 0, 0.85)[..., None]
    base = Image.fromarray(np.clip(laser * (1 - g) + rgb8.astype(np.float32) * g, 0, 255).astype(np.uint8)).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0)); rd = ImageDraw.Draw(ov); col = (32, 245, 130)
    big, tx = 84, 86; ty = int(np.clip(tgt[1] - 24 - 76, int(SPEC["ui"].get("callout_ty_min", 600)), 1460))
    ey = ty + 76; ex = max(286, tgt[0] - 42)
    d5.bcd._draw_precision_leader(rd, tgt, (ex, ey), (278, ey), col)
    deg = f"{int(kp['deg_shown'])}°"; fb, f1, f2 = v30.f_pretendard(big, "extrabold"), v30.f_pretendard(21, "extrabold"), v30.f_pretendard(16, "bold")
    rd.text((tx + 2, ty + 3), deg, font=fb, fill=(0, 0, 0, 210)); rd.text((tx, ty), deg, font=fb, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + big + 6), L["l1"], font=f1, fill=(*col, 255)); rd.text((tx + 2, ty + big + 33), L["l2"], font=f2, fill=(215, 222, 232, 235))
    boxes = [rd.textbbox((tx, ty), deg, font=fb), rd.textbbox((tx + 2, ty + big + 6), L["l1"], font=f1), rd.textbbox((tx + 2, ty + big + 33), L["l2"], font=f2)]
    tb = [min(q[0] for q in boxes), min(q[1] for q in boxes), max(q[2] for q in boxes), max(q[3] for q in boxes)]
    xs_ = [a[0], v[0], b_ext[0], tgt[0], ex, 278]; ys_ = [a[1], v[1], b_ext[1], tgt[1], ey]
    ang2d = math.degrees(math.atan2(abs(b[0] - v[0]), v[1] - b[1]))
    info = {"deg_shown": int(kp["deg_shown"]), "text_box": [int(q) for q in tb], "lines_box": [int(min(xs_)), int(min(ys_)), int(max(xs_)), int(max(ys_))],
            "angle_2d_deg": round(ang2d, 1), "target": list(tgt),
            "prop_clear_px": None if prop_clear is None else round(prop_clear, 1),
            "segments": [[list(map(float, a)), list(map(float, v))], [list(map(float, v)), list(map(float, b_ext))]]}
    return np.array(Image.alpha_composite(base, ov).convert("RGB")), info

def make_icons(graded_rgb, bbox):
    """Preview-pill thumbnail (Day 4: each slot = that step's peak frame): the figure crop of the peak frame."""
    x0, y0, x1, y1 = bbox; y1 = y0 + 0.62 * (y1 - y0)                    # Day 4 slots show the working torso, not the feet
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h = max(y1 - y0, (x1 - x0) * 158 / 134) * 1.08; w = h * 134 / 158
    X0, Y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2)); X1, Y1 = int(min(graded_rgb.shape[1], cx + w / 2)), int(min(graded_rgb.shape[0], cy + h / 2))
    crop = cv2.resize(graded_rgb[Y0:Y1, X0:X1], (224, 264), interpolation=cv2.INTER_AREA)
    return [(crop, (crop * 0.55).astype(np.uint8))]

def main():
    t0 = time.time()
    rdir, odir, cfgp = sys.argv[1:4]; prefix = sys.argv[4] if len(sys.argv) > 4 else "f"
    cfg = json.load(open(cfgp)); os.makedirs(odir, exist_ok=True)
    if prefix == "b":
        cfg = dict(cfg); cfg["grade"] = dict(cfg.get("grade") or {}); cfg["grade"]["glow_source"] = "blender"
    cv2.setNumThreads(2)
    fr = sorted(glob.glob(os.path.join(rdir, f"{prefix}_*.png"))); stats = []
    CH = json.load(open(os.path.join(rdir, "checks.json"))); R = CH["frames"]; KPW = (CH.get("kp_res") or [None])[0]; KP = {r_["f"]: r_.get("kp2d") for r_ in R}
    tor = np.array([r_["torso_deg"] for r_ in R]); shown = int(round(float(np.median(tor))))
    pf = R[int(np.argmin([r_["elbow_deg"] for r_ in R]))]["f"]          # peak frame number (1-based)
    nums = [int(os.path.basename(q)[2:6]) for q in fr]; peak = int(np.argmin([abs(n_ - pf) for n_ in nums]))
    icons = None
    if cfg.get("layout", {}).get("preview_pill"):
        fp = fr[peak]; b0 = cv2.imread(fp, cv2.IMREAD_UNCHANGED); g0, st0 = grade_frame(b0, cv2.imread(fp.replace("/f_", "/m_"), cv2.IMREAD_UNCHANGED), cfg, peak, len(fr), ui_on=False)
        icons = make_icons(g0, st0["bbox"])
    for i, fp in enumerate(fr):
        mp = fp.replace(f"/{prefix}_", "/m_"); fnum = int(os.path.basename(fp)[2:6])
        kp = dict(KP.get(fnum) or {}); kp = (dict(kp, deg_shown=shown) if kp else None)
        bty = cv2.imread(fp, cv2.IMREAD_UNCHANGED); msk = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if kp:   # kp2d is in render pixels; the grade works at out_res
            k_ = cfg["grade"]["out_res"][0] / (KPW or bty.shape[1]); kp = {kk: ([vv[0] * k_, vv[1] * k_] if isinstance(vv, list) else vv) for kk, vv in kp.items()}
        out, st = grade_frame(bty, msk, cfg, i, len(fr), kp, icons); st["src"] = os.path.basename(fp)
        cv2.imwrite(os.path.join(odir, "g_" + os.path.basename(fp)[2:]), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)); stats.append(st)
    json.dump({"frames": stats, "callout_shown_deg": shown, "torso_median_deg": round(float(np.median(tor)), 2), "grade_s": round(time.time() - t0, 1)}, open(os.path.join(odir, "look.json"), "w"), indent=0)
    print("graded", len(fr), "in", round(time.time() - t0, 1), "s")

if __name__ == "__main__":
    main()
