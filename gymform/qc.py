#!/usr/bin/env python3
"""
@gymform.3d automated QC gate for rendered 4-step master reels.

Usage:
  python3 qc_gate_gymform.py <master.mp4> [--qc-dir output/qc/day5_v4]

Reads the rendered mp4 at the start/peak frame of every step and the QC sidecar written by the builder
(build_day5_gym_studio_v4.py -> CFG["qc_dir"]: per-frame clean fg mask, shorts mask, equipment masks,
pre-UI base frame, keypoints, equipment config). Prints PASS/FAIL per item and exits 1 if any item fails:
  (a) no coloured halo / bloom outside the mannequin mask
  (b) title + preview VISIBLE see-through (ui.overlap_alpha, Day 4 rule) over mannequin; mannequin sharp under preview
  (c) mannequin head-to-ankle height equal across the 4 steps (within 4%)
  (d) equipment (any type in config) fully inside frame, fully covered by fg mask (no holes), not distorted
  (e) bone X-Ray present/readable on bone steps, muscle fibers on muscle steps
  (f) skin luminance p75~165 / p90~191 (max<=226), shorts lum 10..22
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np

W, H = 1080, 1920
RESULTS = []


def report(item, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {item}: {detail}", flush=True)


def lum_of(rgb):
    f = rgb.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def act_of(rgb):
    f = rgb.astype(np.float32)
    return np.maximum(f[..., 0] - 0.5 * (f[..., 1] + f[..., 2]), f[..., 1] - 0.5 * (f[..., 0] + f[..., 2]))


def xray_signature(rgb, mask, tint):
    """Colour/structure signature of luminous X-Ray bone pixels: primary channel = tint (R for crimson, G for
    emerald), hot channel = the secondary channel that rises toward the white-hot core in the Day 1 ramp."""
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    if tint == "crimson":
        prim, sec, oth = r, g, b
    else:
        prim, sec, oth = g, b, r
    bone = (prim - oth > 60.0) & (prim > 150.0) & (mask > 0.5)
    n = int(bone.sum())
    if n == 0:
        return {"n": 0, "hot_ratio": 0.0, "core_ratio": 0.0, "detail": 0.0}
    hot_ratio = float(np.median(sec[bone] / np.maximum(prim[bone], 1.0)))
    core_ratio = float(np.mean((sec[bone] > 0.75 * prim[bone])))
    lap = np.abs(cv2.Laplacian(0.299 * r + 0.587 * g + 0.114 * b, cv2.CV_32F))
    return {"n": n, "hot_ratio": hot_ratio, "core_ratio": max(core_ratio, 1e-3), "detail": float(np.mean(lap[bone]))}


# (u) BONE VISIBILITY (owner t530u: "뼈 색상이 너무 진하면 뼈가 안보이자나"). Bone pixels on the rendered X-Ray
# window = luminous structure (white top-hat > 22 over a 15px disc and lum > 120) or tinted glow (act > 40).
# Calibrated on approved Day 1 v36 (5 frames, arm box): L50 150.3-152.9, L90 200.3-206.0, contrast 27.0-42.6,
# dark-saturated share 0.079-0.142. A dark #1E9E5A fill scores L50 112, L90 112, dark share 0.91 (fails).
BONE_VIS = {"L50_min": 140.0, "L90_min": 190.0, "contrast_min": 25.0, "dark_sat_max": 0.20}


def bone_visibility(rgb, region):
    f = rgb.astype(np.float32)
    L = lum_of(f)
    act = act_of(f)
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bone = (((th > 22.0) & (L > 120.0)) | (act > 40.0)) & region
    bone = cv2.morphologyEx(bone.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    n = int(bone.sum())
    if n < 300:
        return {"n": n, "L50": 0.0, "L90": 0.0, "contrast": 0.0, "dark_sat": 1.0}
    ring = (cv2.dilate(bone.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & \
        ~(cv2.dilate(bone.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & region
    mx, mn = f.max(axis=-1), f.min(axis=-1)
    sat = (mx - mn) / (mx + 1.0)
    return {"n": n, "L50": float(np.median(L[bone])), "L90": float(np.percentile(L[bone], 90)),
            "contrast": float(np.median(L[bone]) - (np.median(L[ring]) if ring.any() else 0.0)),
            "dark_sat": float(((sat > 0.6) & (L < 150.0) & bone).sum()) / n}


def bone_visible_ok(m):
    return (m["L50"] >= BONE_VIS["L50_min"] and m["L90"] >= BONE_VIS["L90_min"]
            and m["contrast"] >= BONE_VIS["contrast_min"] and m["dark_sat"] <= BONE_VIS["dark_sat_max"])


def read_frames(mp4, idxs):
    cap = cv2.VideoCapture(mp4)
    out = {}
    for i in sorted(set(idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def expected_equipment(src, plate, zone, clean=None):
    """
    QC's OWN expected equipment footprint for one frame, independent of the renderer's fg mask: dark/neutral
    pixels of the raw source frame that differ from the step's temporal backdrop plate (the backdrop is static,
    the equipment moves), inside the configured equipment zone. Holes enclosed by metal (plate faces) are filled.
    Bright body pixels (hand/arm) and activation glow are excluded: this is the equipment itself.
    """
    f = src.astype(np.float32)
    p = plate.astype(np.float32)
    lum = lum_of(f)
    act = act_of(f)
    d = np.abs(f - p).max(axis=-1)
    tex = cv2.GaussianBlur(np.abs(lum - cv2.GaussianBlur(lum, (0, 0), 2.0)), (0, 0), 1.2)
    p_max = p.max(axis=-1)
    drop = np.minimum((p[:, :, 2] - p[:, :, 0]) - (f[:, :, 2] - f[:, :, 0]), (p[:, :, 2] - p[:, :, 1]) - (f[:, :, 2] - f[:, :, 1]))
    drop = cv2.GaussianBlur(drop, (0, 0), 1.5)
    darker = ((p - f) > 5.5).sum(axis=-1) >= 2
    lum_eq = np.maximum(lum - 0.85 * act, f.min(axis=-1))
    cand = (((d > 12.0) & (tex > 3.0)) | ((drop > 5.0) & (d > 3.0) & (act < 12.0)) | ((d > 6.0) & darker))
    cand &= (zone > 0.5) & (p_max < 52.0) & (lum_eq < 90.0)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    keep = np.zeros_like(cand)
    if clean is None:
        clean = clean_qc(p)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 300:
            c = lab == i
            # a candidate that is, as a whole, plain backdrop in this frame (p90 |lum - object-free backdrop| < 6,
            # no rim/texture) is a temporal-median GHOST of a held item, not equipment
            if np.percentile(np.abs(lum[c] - clean[c]), 90) < 6.0 and np.percentile(tex[c], 90) < 1.5:
                continue
            keep[c] = 1
    ff = np.pad(keep, 1).copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    holes = ((ff[1:-1, 1:-1] != 2) & (keep == 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes)
    out = keep > 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] <= 2500:
            out |= lab == i
    return out & (zone > 0.5)


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if m is None else m.astype(np.float32) / 255.0


def ui_see_through(fr, pre, uia, uirgb, occ, band):
    """Day 4 see-through rule. Over occluder pixels (mannequin / equipment) inside the UI layer, fit the effective
    UI alpha A in  frame = preUI + A * ui_a * (ui_rgb - preUI)  (least squares). Returns (A_hat, residual, n):
    A_hat ~ spec alpha => title/preview VISIBLE (non-zero contrast) but not opaque; small residual => the 3D layer
    under it is intact (not cut, hidden, blurred or distorted)."""
    sel = occ & band & (uia > 0.5)
    f, p_, u = (x[sel].astype(np.float32) for x in (fr, pre, uirgb))
    a = uia[sel][:, None]
    du = (u - p_) * a
    strong = np.abs(du).max(axis=1) > 20.0
    if strong.sum() < 60:
        return None, None, int(strong.sum())
    d, du = (f - p_)[strong], du[strong]
    a_hat = float((d * du).sum() / max((du * du).sum(), 1e-6))
    resid = float(np.abs(d - a_hat * du).max(axis=1).mean())
    return a_hat, resid, int(strong.sum())


def skeletal_size(kp):
    import math
    D = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    M = lambda a, b: (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    return (D(M(kp["sh_r"], kp["sh_l"]), M(kp["hip_r"], kp["hip_l"])) + 0.5 * (D(kp["hip_r"], kp["kn_r"]) + D(kp["hip_l"], kp["kn_l"]))
            + 0.5 * (D(kp["kn_r"], kp["ank_r"]) + D(kp["kn_l"], kp["ank_l"])) + 0.5 * (D(kp["sh_r"], kp["elb_r"]) + D(kp["sh_l"], kp["elb_l"])))


def native_fabric(src):
    """QC's own native fabric map from the raw source frame: dark, neutral, un-lit pixels, small enclosed sheen
    spots included (they are native highlights on the 3D fabric)."""
    f = src.astype(np.float32)
    lum, act = lum_of(f), act_of(f)
    mx, mn = f.max(axis=-1), f.min(axis=-1)
    sat = (mx - mn) / (mx + 1.0)
    lum_eq = np.maximum(lum - 0.85 * act, mn)
    fab = ((lum_eq < 64.0) & (sat < 0.32) & (act < 20.0)).astype(np.uint8)
    fc = cv2.morphologyEx(fab, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ff = np.pad(fc, 1).copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    return (ff[1:-1, 1:-1] != 2)


def overpaint_px(src, shorts):
    """(n) shorts pixels (core, >0.5, 1px in from the soft edge) that are NOT native fabric in the source."""
    core = cv2.erode((shorts > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    nat = cv2.dilate(native_fabric(src).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return int((core & ~nat).sum()), int(core.sum())


_SQC = {}


_CLEAN = {}


def clean_qc(plate):
    """Object-free backdrop luminance of a step plate (gymform.equipment.clean_backdrop), cached per plate."""
    k = (plate.shape, float(plate[::97, ::89].sum()), float(plate[::53, ::61].sum()))
    if k not in _CLEAN:
        from gymform.equipment import clean_backdrop
        _CLEAN[k] = clean_backdrop(plate.astype(np.float32), cache=False)[2]
    return _CLEAN[k]


def static_objects_qc(plate):
    """QC's own static-object map (independent of the renderer): textured or clearly (>16) darker than a coarse
    median backdrop of the plate; smooth faint shadows are not required to be kept."""
    k = (plate.shape, float(plate[::97, ::89].sum()))
    if k in _SQC:
        return _SQC[k]
    l = lum_of(plate.astype(np.float32))
    tex = cv2.GaussianBlur(np.abs(l - cv2.GaussianBlur(l, (0, 0), 2.0)), (0, 0), 1.5)
    # object-free backdrop: everything textured or far from the global backdrop level is masked and inpainted
    # (a plain median is biased by the large bright seated body and over-calls backdrop as "object")
    sm = cv2.resize(l, (W // 8, H // 8), interpolation=cv2.INTER_AREA).astype(np.uint8)
    lev = float(np.median(sm))
    ob = (np.abs(sm.astype(np.float32) - lev) > 10.0) | (cv2.resize(tex, (W // 8, H // 8), interpolation=cv2.INTER_AREA) > 2.0)
    ob = cv2.dilate(ob.astype(np.uint8), np.ones((5, 5), np.uint8))
    bgm = cv2.resize(cv2.GaussianBlur(cv2.inpaint(sm, ob, 8, cv2.INPAINT_NS), (0, 0), 3.0).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    _SQC[k] = (((bgm - l) > 16.0) | (tex > 3.0), tex)
    return _SQC[k]


def stray_components(wfg, src=None, plate=None, base=None, bg=None, min_area=30):
    """(q) Stray / unanchored overlay marks: a foreground component (wfg>0.3) not connected to the largest blob
    (5px bridge) that is either a THIN LINE (mean width < 5px, length >= 20px) or a FAINT GHOST (its source
    pixels are plain backdrop: median |lum - object-free backdrop| < 3.5 and no native texture). Detached but
    solid native objects (a bench foot or plate rim seen past a limb) are not strays. Returns (ok, detail)."""
    fg = (wfg > 0.3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(fg, np.ones((5, 5), np.uint8)), 8)
    if n <= 1:
        return False, "no foreground"
    main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    clean = tex = l = None
    if src is not None and plate is not None:
        clean = clean_qc(plate)
        l = lum_of(src.astype(np.float32))
        tex = cv2.GaussianBlur(np.abs(l - cv2.GaussianBlur(l, (0, 0), 2.0)), (0, 0), 1.2)
    bad = []
    for i in range(1, n):
        if i == main:
            continue
        c = (lab == i) & (fg > 0)
        area = int(c.sum())
        if area < min_area:
            continue
        x, y, w, h = st[i, :4]
        thin = max(w, h) >= 20 and area / float(max(w, h)) < 5.0
        if thin and base is not None and bg is not None:   # a thin mark counts when it is VISIBLE over the studio
            thin = float(np.median(np.abs(lum_of(base.astype(np.float32))[c] - lum_of(bg.astype(np.float32))[c]))) >= 6.0
        faint = clean is not None and np.median(np.abs(l[c] - clean[c])) < 3.5 and np.median(tex[c]) < 1.0
        if thin or faint or clean is None:
            bad.append(f"{area}px@({x},{y},{w}x{h}){' thin' if thin else ''}{' faint' if faint else ''}")
    return (not bad), (", ".join(bad[:4]) if bad else "clean")


def static_check(src, base, plate, wfg, shorts, kp, body_lum=70.0, ref=None):
    """(m) QC's own static-object map (plate vs smooth backdrop; dark, not skin, frame shows the plate) near the
    body: rendered pre-UI base must equal the native source there (not faded into the backdrop, not re-shaded)
    and be fully opaque in the fg mask."""
    f = src.astype(np.float32)
    stat, tex = static_objects_qc(plate)
    # t578: "dark = bench" stops at the render's skin/bench split (static_body_lum) when the mannequin itself is
    # dark charcoal (Day 4 tone v2 clips); 70 for the pale v1 mannequin (meta default 60 -> min(70, 60+10)).
    obj = stat & (np.abs(f - plate).max(axis=-1) < 8.0) & (lum_of(f) < body_lum) & (shorts < 0.2)
    hip = (0.5 * (kp["hip_r"][0] + kp["hip_l"][0]), 0.5 * (kp["hip_r"][1] + kp["hip_l"][1]))
    near = np.zeros((H, W), np.uint8)
    cv2.ellipse(near, (int(hip[0]), int(hip[1]) + 120), (330, 420), 0, 0, 360, 1, -1)
    obj = cv2.erode((obj & (near > 0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    n = int(obj.sum())
    if n < 1500:
        return False, f"static object not found ({n}px) - bench missing/faded or cut"
    if ref is not None:   # native grade: same pixels after the global tone map (the only allowed change)
        f = ref.astype(np.float32)
        lr = lum_of(f)
        tex = cv2.GaussianBlur(np.abs(lr - cv2.GaussianBlur(lr, (0, 0), 2.0)), (0, 0), 1.5)
    diff = np.abs(base[obj].astype(np.float32) - f[obj]).max(axis=-1)
    cov = float((wfg[obj] > 0.9).mean())
    t_src = float(tex[obj].mean())
    lb = lum_of(base.astype(np.float32))
    t_out = float(cv2.GaussianBlur(np.abs(lb - cv2.GaussianBlur(lb, (0, 0), 2.0)), (0, 0), 1.5)[obj].mean())
    ok = cov >= 0.95 and float(np.median(diff)) <= 6.0 and t_out >= 0.85 * t_src
    return ok, f"{n}px, opaque {100 * cov:.0f}%, median |out-src|={np.median(diff):.1f}, detail {t_out:.1f}/{t_src:.1f}"


ALLOWED_SOURCES = ("native_3d_clip", "native_3d_blender")
FORBIDDEN_2D = ("shorts_polygon", "shorts_fill", "drawn_bones", "drawn_shorts", "drawn_equipment", "warp", "rembg_cutout")


def native_source_ok(prov):
    """Hard rule: source must be native 3D (mannequin, 3D shorts, equipment and anatomy rendered in the 3D source,
    not drawn afterwards) and no 2D overpaint stage may be enabled."""
    if not prov:
        return False, "config has no 'source' provenance block"
    if prov.get("type") not in ALLOWED_SOURCES:
        return False, f"source.type={prov.get('type')!r} not in {ALLOWED_SOURCES}"
    need = {"mannequin", "shorts", "equipment"}
    missing = need - set(prov.get("native", []))
    if missing:
        return False, f"not native in source: {sorted(missing)}"
    bad = [k for k in FORBIDDEN_2D if prov.get("overpaint_2d", {}).get(k)]
    if bad:
        return False, f"2D overpaint stage enabled: {bad}"
    return True, f"type={prov['type']} native={sorted(prov.get('native', []))} overpaint_2d=none"


LAST_BK = [None]   # backdrop pixels covered by fg in the last checked frame (proof overlays)


def check_equipment(key, src, plate, zone, wfg, shorts, kp, vis=None):
    """(d) one frame: equipment vs the QC's own expected footprint. Returns (fail_list, rows, exp).
    Speed: all per-pixel work runs on the equipment zone's bounding box (+48px); plate-wide models (clean
    backdrop, static objects) are computed once per plate on the full frame and cropped, so results equal the
    full-frame computation."""
    ys_, xs_ = np.where(zone > 0)
    if len(xs_) == 0:
        return check_equipment_crop(key, src, plate, zone, wfg, shorts, kp, (0, 0), None, None, vis)
    x0, x1 = max(0, xs_.min() - 48), min(W, xs_.max() + 49)
    y0, y1 = max(0, ys_.min() - 48), min(H, ys_.max() + 49)
    sl = (slice(y0, y1), slice(x0, x1))
    clean = clean_qc(plate)[sl]
    stq = static_objects_qc(plate.astype(np.float32))[0][sl] if any(e.get("static") for e in CFG_EQ[0]) else None
    kp_c = {k_: (v_[0] - x0, v_[1] - y0) for k_, v_ in kp.items()}
    f_, r_, exp_c = check_equipment_crop(key, src[sl], plate[sl], zone[sl], wfg[sl], shorts[sl], kp_c, (x0, y0), clean, stq,
                                         None if vis is None else vis[sl])
    exp = np.zeros((H, W), bool)
    exp[sl] = exp_c
    return f_, r_, exp


def check_equipment_crop(key, src, plate, zone, wfg, shorts, kp, off, clean, stq, vis=None):
    fails, rows = [], []
    Hc, Wc = zone.shape[:2]
    exp = expected_equipment(src, plate, zone, clean)
    n_exp = int(exp.sum())
    if n_exp < 1500:
        return [f"{key}: expected equipment footprint too small ({n_exp}px) - detector/zone problem"], rows, exp
    core = cv2.erode(exp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    cov = float((wfg[core] > 0.5).mean())
    miss = (cv2.erode(exp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (wfg < 0.5)
    n_m, lab_m, st_m, _ = cv2.connectedComponentsWithStats(miss.astype(np.uint8))
    big_hole = max([st_m[i, cv2.CC_STAT_AREA] for i in range(1, n_m)] or [0])
    ys, xs = np.where(exp)
    edge = xs.min() + off[0] <= 3 or ys.min() + off[1] <= 3 or xs.max() + off[0] >= W - 4 or ys.max() + off[1] >= H - 4
    border = (zone > 0) & (cv2.erode((zone > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
    touch = float((exp & border).sum()) / max(1.0, float(border.sum()))
    fs, fp = src.astype(np.float32), plate.astype(np.float32)
    d = np.abs(fs - fp).max(axis=-1)
    shd = cv2.dilate((shorts > 0.1).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    lum_eq = np.maximum(lum_of(fs) - 0.85 * act_of(fs), fs.min(axis=-1))
    # neutral (the backdrop is cool blue-grey: B-R ~ +4..10, B-G ~ +5; metal/rubber ~ 0)
    neutral = \
        (cv2.blur((fs[:, :, 2] - fs[:, :, 0]), (3, 3)) < 3.0) & (cv2.blur((fs[:, :, 2] - fs[:, :, 1]), (3, 3)) < 3.5)
    metal = (lum_eq < 80.0) & (act_of(fs) < 12.0) & neutral & (zone > 0.5)
    # (matte plate faces can equal the backdrop in max-channel; the QC's own footprint says they are the item)
    backdrop = (d < 6.0) & (fp.max(axis=-1) < 52.0) & (zone > 0.5) & ~shd & ~(cv2.dilate(exp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    if any(e.get("static") for e in CFG_EQ[0]):   # a static bench/machine in the zone is equipment, not backdrop
        sq_ = stq if stq is not None else static_objects_qc(fp)[0]
        backdrop &= ~(cv2.dilate(sq_.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    backdrop &= ~metal   # backdrop = the cool blue-grey studio colour; neutral dark metal of a held-still item is not backdrop
    bk = ((wfg > 0.5) if vis is None else vis) & cv2.erode(backdrop.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    bloat = float(bk.sum()) / n_exp
    LAST_BK[0] = bk
    mid_x = int(0.5 * (kp["sh_r"][0] + kp["sh_l"][0]))
    side = np.zeros((Hc, Wc), bool); side[:, :max(0, mid_x)] = True
    # per-item size vs the source, METAL pixels only (neutral, dark: never hand/forearm skin or glow), on both
    # sides of the comparison: source metal object vs metal pixels the composite renders solid around it.
    # Held-still items (the temporal plate already contains them, so plate-difference cannot see them) are
    # excluded from the size test and reported as such; coverage/holes above still apply.
    sz, ratios = [], []
    for nm_, sd in (("L", side), ("R", ~side)):
        e_m = exp & metal & sd
        if e_m.sum() < 800:
            continue
        near = cv2.dilate(e_m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        still = float((metal & (d < 6.0) & near & ~e_m).sum()) / float(e_m.sum())
        if still > 0.5:
            sz.append(f"{nm_} held-still (plate holds {100 * still:.0f}%) skipped")
            continue
        # the composite shows source pixels wherever w_fg is solid, so the rendered item can only grow by
        # covering source BACKDROP pixels (bk) or shrink by dropping item pixels: rendered = kept item + covered backdrop
        r_m = (e_m & (wfg > 0.5)) | (bk & near & sd)
        ratios.append(float((bk & sd).sum()) / float((exp & sd).sum()))
        ey, ex = np.where(e_m); ry, rx = np.where(r_m)
        ew, eh = np.percentile(ex, 99) - np.percentile(ex, 1), np.percentile(ey, 99) - np.percentile(ey, 1)
        rw, rh = np.percentile(rx, 99) - np.percentile(rx, 1), np.percentile(ry, 99) - np.percentile(ry, 1)
        dw, dh, da = rw / max(ew, 1) - 1, rh / max(eh, 1) - 1, r_m.sum() / max(e_m.sum(), 1) - 1
        sz.append(f"{nm_} {int(ew)}x{int(eh)} w{100 * dw:+.0f}% h{100 * dh:+.0f}% a{100 * da:+.0f}%")
        if max(abs(dw), abs(dh), abs(da)) > 0.08:
            fails.append(f"{key}: {nm_} equipment size off vs source (w {100 * dw:+.0f}%, h {100 * dh:+.0f}%, area {100 * da:+.0f}%)")
    rows.append(f"{key}: exp={n_exp}px cov={100 * cov:.1f}% maxhole={big_hole}px bloat={100 * bloat:.1f}% "
                f"bloatLR={'/'.join(f'{100 * r:.1f}%' for r in ratios)} size[{', '.join(sz)}]")
    if bloat > 0.03:
        fails.append(f"{key}: equipment bloated beyond its source shape ({100 * bloat:.1f}% extra)")
    if any(r > 0.04 for r in ratios):
        fails.append(f"{key}: one item bloated beyond its source shape (L/R bloat {['%.1f%%' % (100 * r) for r in ratios]})")
    if cov < 0.985:
        fails.append(f"{key}: only {100 * cov:.1f}% of expected equipment is solid fg")
    if big_hole > 60:
        fails.append(f"{key}: equipment hole of {big_hole}px")
    if edge:
        fails.append(f"{key}: equipment touches frame edge")
    if touch > 0.10:
        fails.append(f"{key}: equipment reaches its zone border on {100 * touch:.0f}% (zone clips it)")
    return fails, rows, exp


def arm_skin_check(base, wfg, eq, xw, kp, src=None):
    """(j) both arms (keypoint segments, shoulder..80% forearm) outside the X-Ray window: silver skin, no dark fill.
    Returns {side: (p75, dark_fraction)}."""
    l, a = lum_of(base.astype(np.float32)), act_of(base.astype(np.float32))
    xw_d = (cv2.dilate((xw > 0.02).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) if xw is not None else np.zeros((H, W), bool)
    out = {}
    for side in ("r", "l"):
        m = np.zeros((H, W), np.uint8)
        sh, el, wr = kp[f"sh_{side}"], kp[f"elb_{side}"], kp[f"wr_{side}"]
        fw = (el[0] + 0.8 * (wr[0] - el[0]), el[1] + 0.8 * (wr[1] - el[1]))
        P = lambda p: (int(round(p[0])), int(round(p[1])))
        cv2.line(m, P(sh), P(el), 1, 20); cv2.line(m, P(el), P(fw), 1, 14)
        lit_src = (act_of(src.astype(np.float32)) > 12.0) if src is not None else False   # native bone/fibre glow in the source (X-Ray content, not skin)
        lit = cv2.dilate((a > 12.0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0   # muscle-fibre colour + its sulci
        glow = cv2.dilate(np.asarray(lit_src, np.uint8) * np.ones((H, W), np.uint8), np.ones((9, 9), np.uint8)) > 0
        sel = (m > 0) & (wfg > 0.9) & ~xw_d & ~lit & ((eq < 0.5) if eq is not None else True)
        if sel.sum() < 300:
            continue
        # a dark FILL is a solid dark blob (natural creases are thin lines): largest opened dark component
        dk = cv2.morphologyEx(((l < 60.0 * SK_S) & sel).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(dk)
        blob = int(max([st[i, cv2.CC_STAT_AREA] for i in range(1, n)] or [0]))
        # silver-range p50 only on non-glow skin (native X-Ray glow regraded to silver is X-Ray content); dark fill on all
        tone = sel & ~glow
        out[side] = (float(np.percentile(l[tone], 50)) if tone.sum() >= 300 else float(SK['p50']), blob, int(tone.sum()))
    return out


def leg_below_hem_check(base, wfg, shorts, kp, eqfoot=None, src=None):
    """(k) leg skin segments below the shorts hem: no dark (lum < 60) pixels. The hem of each leg is the last row
    where the shorts cover >= 60% of a band around the thigh line (a narrow tongue below the hem does not move it)."""
    l = lum_of(base.astype(np.float32))
    out = {}
    P = lambda p: (int(round(p[0])), int(round(p[1])))
    for side in ("r", "l"):
        hip, kn, an = kp[f"hip_{side}"], kp[f"kn_{side}"], kp[f"ank_{side}"]
        hem = int(hip[1])
        for y in range(int(hip[1]), int(kn[1])):
            t = (y - hip[1]) / max(1.0, kn[1] - hip[1])
            x = int(hip[0] + t * (kn[0] - hip[0]))
            band = shorts[y, max(0, x - 40):x + 41]
            if band.size and (band > 0.5).mean() >= 0.6:
                hem = y
        m = np.zeros((H, W), np.uint8)
        cv2.line(m, P(hip), P(kn), 1, 70); cv2.line(m, P(kn), P(an), 1, 50)
        m[:hem + 5] = 0
        sel = (m > 0) & (wfg > 0.9)
        # native shorts fabric is not leg skin ((n) proves every shorts pixel is native fabric); seated thighs
        # run diagonally, so the row-based hem alone left the lower shorts edge inside the "leg" band
        sel &= ~(cv2.dilate((shorts > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        if eqfoot is not None:   # hands/dumbbells resting beside the thigh are not leg skin
            sel &= ~(cv2.dilate((eqfoot > 0.3).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
        dark = (l < 60.0 * SK_S) & sel   # thresholds scale with the locked skin tone (Day 4 charcoal, t542u)
        if src is not None:   # a FILL = rendered dark where the source is not (held-still dumbbells/creases are dark in the source too)
            dark &= lum_of(src.astype(np.float32)) * SK_S - l > 12.0 * SK_S
        dk = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_, _, st_, _ = cv2.connectedComponentsWithStats(dk)
        blob = int(max([st_[i, cv2.CC_STAT_AREA] for i in range(1, n_)] or [0]))   # largest dark blob (creases are tiny)
        out[side] = (hem, blob, int(sel.sum()))
    return out


DAY4_MUSCLE_PX = {"wrong": (50145, 62472), "right": (21053, 35179)}   # Day 4 master act>40 px (peak / start), t567u


def _raw_backdrop(src):
    """pixels of the RAW source frame that are plain backdrop: within 6 lum of the per-row backdrop level (median of
    the frame's left/right 24 px, smoothed over rows) on the 4 px-smoothed lum, no texture, AND connected to the
    backdrop at the frame's left/right edge (or a large open area). A flat fabric highlight that happens to sit at
    the backdrop level but is enclosed by the shorts is NOT backdrop (t600: a lum-only test flagged shorts folds)."""
    l = lum_of(src.astype(np.float32))
    prof = np.median(np.concatenate([l[:, :24], l[:, -24:]], axis=1), axis=1).astype(np.float32)
    prof = cv2.GaussianBlur(prof[:, None], (0, 0), 25.0)[:, 0]
    ls = cv2.GaussianBlur(l, (0, 0), 4.0)
    tex = cv2.GaussianBlur(np.abs(l - cv2.GaussianBlur(l, (0, 0), 2.0)), (0, 0), 2.0)
    cand = ((np.abs(ls - prof[:, None]) < 6.0) & (tex < 4.0)).astype(np.uint8)
    cand[:620] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    keep = np.zeros(n, bool)
    edge_ids = np.unique(np.concatenate([lab[:, :30].ravel(), lab[:, -30:].ravel()]))
    keep[edge_ids] = True
    keep |= st[:, cv2.CC_STAT_AREA] > 30000
    keep[0] = False
    bk = keep[lab]
    obj = ~bk
    obj[:620] = False
    return bk, obj


def playbook_frame_checks(frames_meta, qc_dir):
    """t600 DEFECT_PLAYBOOK gates on every QC frame (preflight + final):
    (sb) D15 shorts on backdrop: shorts-mask pixels that are raw backdrop <= 2% of the shorts mask (v4: 38.6%)
    (gh) D13 ghosts: raw-backdrop pixels rendered as a visible object (|out - flat backdrop| > 12, 15 px clear of the
         raw object) <= 2% of the raw object area (v4: 11.9% painted opaque)"""
    sb_bad, sb_rows, gh_bad, gh_rows, sep_bad, sep_rows, pl_bad, pl_rows = [], [], [], [], [], [], [], []
    bg_l = 0.299 * 35 + 0.587 * 39 + 0.114 * 43
    for key, fm in sorted(frames_meta.items()):
        pre = os.path.join(qc_dir, key)
        src, base = cv2.imread(pre + "_src.png"), cv2.imread(pre + "_base.png")
        sh = load_mask(pre + "_shorts.png")
        if src is None or base is None or sh is None:
            continue
        bk, obj = _raw_backdrop(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        shm = sh > 0.5
        r = float((shm & bk).sum()) / max(1.0, float(shm.sum()))
        sb_rows.append(f"{key}: {100 * r:.1f}%")
        if r > 0.02:
            sb_bad.append(sb_rows[-1])
        # (sep) t601 separation (D17)
        b_rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype(np.float32)
        lb_ = lum_of(b_rgb)
        wfg_, eq_, st_ = (load_mask(pre + f"_{n_}.png") for n_ in ("wfg", "eq", "static"))
        if wfg_ is not None:
            bgz = cv2.dilate((wfg_ > 0.02).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
            bgz[:620] = False
            skz = (wfg_ > 0.5) & (lb_ > 46.0) & (act_of(b_rgb) < 12.0) & (sh < 0.2)
            for m_ in (eq_, st_):
                if m_ is not None:
                    skz &= m_ < 0.5
            skz[:620] = False
            if bgz.sum() > 5000 and skz.sum() > 2000:
                bgc = np.median(b_rgb[bgz], axis=0)
                skc = np.median(b_rgb[skz], axis=0)
                fig = float(np.percentile(lb_[skz], 75) - lum_of(bgc[None])[0])
                parts, ok_ = [f"fig {fig:.0f}"], fig >= 33.0
                bh2 = ((b_rgb[..., 2] - np.maximum(b_rgb[..., 0], b_rgb[..., 1])) > 25).astype(np.uint8)
                _pm0 = load_mask(pre + "_plate.png")
                if bh2[620:].sum() > 3000 and (_pm0 is None or _pm0.max() <= 0.05):
                    eq_ = cv2.morphologyEx(bh2, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(np.float32)   # source cobalt plates
                for nm, m_ in (("dumbbell", eq_), ("bench", st_)):
                    if m_ is None:
                        continue
                    z = cv2.erode((m_ > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    z[:620] = False
                    if z.sum() < 800:
                        continue
                    ec = np.median(b_rgb[z], axis=0)
                    d_b, d_s = float(np.linalg.norm(ec - bgc)), float(np.linalg.norm(ec - skc))
                    lost = float((np.abs(lb_[z] - lum_of(bgc[None])[0]) < 6.0).mean())   # pixels melting into the backdrop
                    parts.append(f"{nm} vs bg {d_b:.0f} vs skin {d_s:.0f} lost {100 * lost:.0f}%")
                    ok_ &= d_b >= 15.0 and d_s >= 10.0 and lost <= 0.25
                gl = float((bk & (act_of(b_rgb) > 15.0)).sum()) / max(1.0, float(bk.sum()))
                parts.append(f"glow on bg {100 * gl:.2f}%")
                ok_ &= gl <= 0.005
                sep_rows.append(f"{key}: " + ", ".join(parts))
                if not ok_:
                    sep_bad.append(sep_rows[-1])
        # (pl) t605 custom plates (D18): plate vs backdrop >= 40 RGB, recolour weight on raw skin / raw backdrop <= 0.5%
        pmk = load_mask(pre + "_plate.png")
        if (pmk is None or pmk.max() <= 0.05) and wfg_ is not None and bgz.sum() > 5000:
            # t609 source-coloured plates (palette A in the clip, no recolour): plate = cobalt-hue blobs, convex hull
            # per blob (all faces incl. lit tops / shaded sides); plate vs backdrop median >= 40 (D18) + p05 reported
            bh = ((b_rgb[..., 2] - np.maximum(b_rgb[..., 0], b_rgb[..., 1])) > 25).astype(np.uint8)
            bh[:620] = 0
            bh = cv2.morphologyEx(bh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n_, lab_, st__, _ = cv2.connectedComponentsWithStats(bh, connectivity=8)
            hull_ = np.zeros(bh.shape, np.uint8)
            for i_ in range(1, n_):
                if st__[i_, cv2.CC_STAT_AREA] > 800:
                    yy, xx = np.where(lab_ == i_)
                    cv2.fillConvexPoly(hull_, cv2.convexHull(np.stack([xx, yy], 1).astype(np.int32)), 1)
            hull_ = cv2.erode(hull_, np.ones((7, 7), np.uint8)) > 0
            if hull_.sum() > 1500:
                bgc_ = np.median(b_rgb[bgz], axis=0)
                dd = np.linalg.norm(b_rgb - bgc_, axis=-1)
                area40 = cv2.morphologyEx((hull_ & (dd < 40)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).sum() / float(hull_.sum())
                d_pb = float(np.median(dd[hull_]))
                pl_rows.append(f"{key}: source plates vs bg median {d_pb:.0f} p05 {np.percentile(dd[hull_], 5):.0f}, faces<40 {100 * area40:.1f}% ({int(hull_.sum())}px)")
                if d_pb < 40.0:
                    pl_bad.append(pl_rows[-1])
        if pmk is not None and pmk.max() > 0.05 and wfg_ is not None and bgz.sum() > 5000:
            ps = pmk > 0.7
            if ps.sum() > 300:
                d_pb = float(np.linalg.norm(np.median(b_rgb[ps], axis=0) - np.median(b_rgb[bgz], axis=0)))
                s_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32)
                ls_ = lum_of(s_rgb)
                pw = pmk > 0.2
                skin_raw = (cv2.GaussianBlur(ls_, (0, 0), 1.5) > 55.0) & (act_of(s_rgb) < 12.0)
                bleed = float((pw & (skin_raw | bk)).sum()) / max(1.0, float(pw.sum()))
                pl_rows.append(f"{key}: plate vs bg {d_pb:.0f}, bleed {100 * bleed:.2f}% ({int(ps.sum())}px)")
                if d_pb < 40.0 or bleed > 0.005:
                    pl_bad.append(pl_rows[-1])
        far = bk & (cv2.dilate(obj.astype(np.uint8), np.ones((31, 31), np.uint8)) == 0)
        lo = lum_of(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
        g = float((far & (np.abs(lo - bg_l) > 12.0)).sum()) / max(1.0, float(obj[620:].sum()))
        gh_rows.append(f"{key}: {100 * g:.2f}%")
        if g > 0.02:
            gh_bad.append(gh_rows[-1])
    report("(sep) figure / equipment / backdrop separated (playbook D17: skin p75 - backdrop >= 33; dumbbells and "
           "bench >= 15 RGB from backdrop, >= 10 from skin, <= 25% of their pixels within 6 lum of it; glow on backdrop <= 0.5%)", bool(sep_rows) and not sep_bad,
           "; ".join(sep_bad[:4]) or " | ".join(sep_rows[:4]))
    if pl_rows or pl_bad:
        report("(pl) custom plates (playbook D18): plate vs backdrop >= 40 RGB, recolour on raw skin/backdrop <= 0.5%",
               not pl_bad, "; ".join(pl_bad[:4]) or " | ".join(pl_rows[:4]))
    report("(sb) shorts never on raw backdrop (<= 2% of the shorts mask; playbook D15)", bool(sb_rows) and not sb_bad,
           "; ".join(sb_bad[:4]) or " | ".join(sb_rows[:4]))
    report("(gh) no ghosts: raw backdrop rendered as object <= 2% of object area (playbook D13)", bool(gh_rows) and not gh_bad,
           "; ".join(gh_bad[:4]) or " | ".join(gh_rows[:4]))


def playbook_temporal_check(meta, mp4):
    """t600 DEFECT_PLAYBOOK temporal gates (full render only): raw clip vs output, flow-compensated
    (gymform/temporal_qc.py): boil <= x1.20, edge (silhouette) flicker <= x1.56, glow flicker <= x1.59 of the raw
    clip (= Day 4's best ratios), step cut hips/ankles <= 25 px, shoulders <= 40 px, body lum <= 3, cut diff <= 2x."""
    cp = meta.get("config_path")
    if not cp or not os.path.exists(cp):
        report("(tq) temporal playbook gates (boil / edge flicker / glow flicker / step cut)", False, "meta has no config_path")
        return
    import subprocess as _sp
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = _sp.run([sys.executable, "-m", "gymform.temporal_qc", "gate", cp, mp4, "qc_" + os.path.basename(mp4)[:-4]],
                cwd=root, capture_output=True, text=True, env=dict(os.environ, ROOTCAUSE_OUT=os.path.join(root, "output", "qc", "temporal")))
    line = next((l for l in r.stdout.splitlines() if l.startswith("TEMPORAL_QC ")), None)
    if not line:
        report("(tq) temporal playbook gates (boil / edge flicker / glow flicker / step cut)", False, (r.stderr or "no output")[-300:])
        return
    res = json.loads(line[12:])
    report("(tq) temporal playbook gates: boil<=x1.20 edge flicker<=x1.56 glow<=x1.59 of raw; step cut matched", res["ok"],
           "; ".join(res["fails"]) or " | ".join(res["rows"]))


def flat_look_checks(meta, frames_meta, vids, qc_dir):
    """t592u (shading came back in Day 6 v2/v3): Day 4 = flat 35/39/43 backdrop and one even body tone.
    (bg) backdrop flat: 120x120 block medians of pure-backdrop pixels (below the UI, no body/equipment/callout)
         span <= 4 lum (Day 4: 0; the vignetted plate of v3: 23)
    (sh) body shading: skin (no shorts/equipment/static/glow) darker than lum 46 <= 6% and p5 >= 44
         (Day 4 floors its body at 42-48; v3 had a dark shaded side with lum 15-45)"""
    b_bad, b_rows, s_bad, s_rows = [], [], [], []
    for key, fm in sorted(frames_meta.items()):
        pre = os.path.join(qc_dir, key)
        fr = vids.get(fm["frame_idx"])
        wfg, st, sh, eq = (load_mask(pre + f"_{n}.png") for n in ("wfg", "static", "shorts", "eq"))
        if fr is None or wfg is None:
            continue
        f = fr.astype(np.float32)
        l = lum_of(f)
        occ = (wfg > 0.02) | ((st if st is not None else 0) > 0.02) | ((eq if eq is not None else 0) > 0.02)
        occ = cv2.dilate(occ.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        bgm = ~occ
        bgm[:620] = False                                     # title / preview capsule
        bgm[:1300, :320] = False                              # callout text block
        meds = []
        for y in range(620, H - 60, 120):
            for x in range(0, W, 120):
                blk = bgm[y:y + 120, x:x + 120]
                if blk.sum() > 3000:
                    meds.append(float(np.median(l[y:y + 120, x:x + 120][blk])))
        if len(meds) >= 6:
            span = max(meds) - min(meds)
            b_rows.append(f"{key}: span {span:.1f} ({min(meds):.0f}-{max(meds):.0f}, {len(meds)} blocks)")
            if span > 4.0:
                b_bad.append(b_rows[-1])
        skin = (wfg > 0.9) & (act_of(f) < 12.0)
        for m_ in (st, sh, eq):
            if m_ is not None:
                skin &= m_ < 0.2 if m_ is sh else m_ < 0.5
        skin[:620] = False
        if skin.sum() > 2000:
            dark = float((l[skin] < 46.0).mean())
            p5 = float(np.percentile(l[skin], 5))
            s_rows.append(f"{key}: dark<46 {100 * dark:.1f}% p5 {p5:.0f}")
            if dark > 0.06 or p5 < 44.0:
                s_bad.append(s_rows[-1])
    report("(bg) backdrop flat like Day 4 (block span <= 4 lum, no vignette / spotlight / floor shadow)", not b_bad,
           "; ".join(b_bad[:4]) or " | ".join(b_rows[:4]))
    report("(sh) body evenly lit like Day 4 (skin < lum 46 <= 6%, p5 >= 44: no dark shaded side)", not s_bad,
           "; ".join(s_bad[:4]) or " | ".join(s_rows[:4]))


def t567_checks(meta, frames_meta, vids, qc_dir):
    """t567u checks for the t558 diagnosis fixes (muscle-only day):
    (mf) muscle fibre colour = the Day 4 shader with the Day 4/5 look (t572u: no speckle, Day 4/5 colour strength)
    (o) whole figure centred (silhouette bbox centre 540 +-40 px, >= 24 px margin both sides)
    (p) the static (bench) layer never claims bright mannequin skin (pale feet / skin streaks <= 300 px)
    (r) callout axis follows the upper arm (target point on the shoulder->elbow segment, arm_axis)"""
    m_bad, m_rows, o_bad, o_rows, p_bad, p_rows, r_bad, r_rows = [], [], [], [], [], [], [], []
    for key, fm in sorted(frames_meta.items()):
        pre = os.path.join(qc_dir, key)
        fr = vids.get(fm["frame_idx"])
        src = cv2.imread(pre + "_src.png")
        wfg, st = load_mask(pre + "_wfg.png"), load_mask(pre + "_static.png")
        if fr is None or src is None or wfg is None:
            continue
        srcf = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32)
        a_src, a_out = act_of(srcf), act_of(fr.astype(np.float32))
        body = (wfg > 0.5) & ((st if st is not None else 0) < 0.5)
        ui = np.zeros_like(body); ui[:600] = True; ui[:, :300] = True
        if fm["tag"] in ("start", "peak"):
            # t572u: judged against the Day 4 (GOLD #1) / Day 5 look, measured on their posted masters:
            # coloured regions 3-11 (no speckle), colour strength act median 52-86, shader = day4
            m_ = ((a_out > 25) & body & ~ui).astype(np.uint8)
            n_, _, st_, _ = cv2.connectedComponentsWithStats(m_, 8)
            comps = sum(1 for i in range(1, n_) if st_[i, cv2.CC_STAT_AREA] >= 20)
            amed = float(np.median(a_out[m_ > 0])) if m_.sum() else 0.0
            form = "wrong" if meta.get("is_wrong", {}).get(str(fm["step"])) else "right"
            d4 = DAY4_MUSCLE_PX[form]
            m_rows.append(f"{key}: {int(m_.sum())}px, {comps} regions, strength {amed:.0f} (Day 4 {form}: {d4[0] // 1000}-{d4[1] // 1000}k px, 3-8 regions, 52-78)")
            if meta.get("muscle_shader") != "day4" or comps > 16 or not (45.0 <= amed <= 100.0) or m_.sum() < 4000:
                m_bad.append(m_rows[-1] + f" shader={meta.get('muscle_shader')}")
        ys, xs = np.where(body)
        if xs.size:
            x0, x1 = int(np.percentile(xs, 0.2)), int(np.percentile(xs, 99.8))
            cx = 0.5 * (x0 + x1)
            o_rows.append(f"{key}: x {x0}-{x1} centre {cx:.0f}")
            if abs(cx - W / 2) > 40 or x0 < 24 or x1 > W - 24:
                o_bad.append(o_rows[-1])
        if st is not None:
            L = srcf.mean(-1); sat = srcf.max(-1) - srcf.min(-1)
            n_sk = int(((st > 0.5) & (L > 110) & (sat < 30)).sum())
            p_rows.append(f"{key}: {n_sk}px")
            if n_sk > 300:
                p_bad.append(f"{key}: static layer on {n_sk}px bright skin")
        ca = fm.get("callout_axis")
        if ca:
            v, b_, t = np.array(ca["vertex"]), np.array(ca["limb"]), np.array(ca["target"], float)
            seg = b_ - v
            u = float(np.clip(np.dot(t - v, seg) / max(np.dot(seg, seg), 1.0), 0, 1))
            d = float(np.linalg.norm(v + u * seg - t))
            if fm["tag"] == "peak":
                r_rows.append(f"{key}: target {d:.1f}px off the upper arm (u={u:.2f}), arm_axis={ca.get('arm_axis')}")
            if not ca.get("arm_axis") or d > 8.0:
                r_bad.append(f"{key}: target {d:.1f}px off arm, arm_axis={ca.get('arm_axis')}")
        else:
            r_bad.append(f"{key}: no callout_axis in meta")
    report("(mf) muscle fibre colour = Day 4 shader + Day 4/5 look (<=16 regions, strength 45-100, >=4k px)", bool(m_rows) and not m_bad,
           "; ".join(m_bad) or " | ".join(m_rows))
    report("(oc) whole figure centred (bbox centre 540+-40, margins >=24px)", bool(o_rows) and not o_bad, "; ".join(o_bad[:4]) or " | ".join(o_rows[::3]))
    report("(pf) bench/static layer never on bright mannequin skin (<=300px)", not p_bad, "; ".join(p_bad[:4]) or "max " + max(p_rows, key=lambda r: int(r.split(": ")[1][:-2])) if p_rows else "no static")
    report("(ra) callout axis follows the upper arm (target on shoulder->elbow)", not r_bad, "; ".join(r_bad[:4]) or " | ".join(r_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4", help="master mp4, or '-' with --preflight")
    ap.add_argument("--preflight", action="store_true", help="QC sample frames rendered by run_day --preflight (no mp4)")
    ap.add_argument("--qc-dir", default="/workspace/daily_info_reels/output/qc/day5_v4d")
    ap.add_argument("--xray-ref", default="/workspace/daily_info_reels/output/qc/ref/day1_v36_xray_t1.5.jpg",
                    help="approved Day 1 v36 X-Ray reference frame (extracted from output/day1_v36_master_reel.mp4 @1.5s)")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.qc_dir, "meta.json")))
    frames_meta = meta["frames"]
    CFG_EQ[0] = meta.get("equipment", [])
    if args.preflight:   # container format is checked on the full render only (fixed x264 settings)
        n, fps = int(round(30 * float(meta.get("duration_s", 20.0)))), 30.0   # (l) timing: the reel is fixed at 20.0s (tempo spec); the wav timing is real
        vids = {}
        for k_, v_ in frames_meta.items():
            fp_ = os.path.join(args.qc_dir, k_ + "_final.png")
            if os.path.exists(fp_):
                vids[v_["frame_idx"]] = cv2.cvtColor(cv2.imread(fp_), cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(args.mp4)
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        report("format", (w, h) == (W, H) and abs(fps - 30) < 0.01 and n >= 30 * float(meta.get("duration_s", 20.0)) - 10, f"{w}x{h} @ {fps:.2f}fps, {n} frames ({n / max(fps, 1):.2f}s)")
        vids = read_frames(args.mp4, [v["frame_idx"] for v in frames_meta.values()])
    bg = cv2.cvtColor(cv2.imread(meta["studio_bg"]), cv2.COLOR_BGR2RGB)
    if bg.shape[:2] != (H, W):
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
    if _SPEC_Q["background"].get("grade_to_rgb"):   # same Day 4 backdrop grade as gymform.bg.studio_bg (t543u)
        bgf = bg.astype(np.float32)
        bg = np.clip(bgf * (np.asarray(_SPEC_Q["background"]["grade_to_rgb"], np.float32)
                            / np.maximum(np.median(bgf.reshape(-1, 3), axis=0), 1.0)), 0, 255).astype(np.uint8)
    bg_act = act_of(bg)
    ty0, ty1 = meta["title_roi"][1], meta["title_roi"][3]
    px0, py0, px1, py1 = meta["preview_roi"]

    heights = {}
    halo_bad, title_bad, prev_bad, eq_fail, skin_fail, ui_fail = [], [], [], [], [], []
    oa = meta.get("overlap_alpha", {"title": 0.40, "preview": 0.40, "qc_tolerance": 0.12})
    global OA_SPEC, OA_TOL
    OA_SPEC, OA_TOL = float(oa["title"]), float(oa.get("qc_tolerance", 0.12))
    B1_ROWS = []
    eq_area = {}
    for key, fm in sorted(frames_meta.items()):
        fr = vids.get(fm["frame_idx"])
        pre = os.path.join(args.qc_dir, key)
        base = cv2.cvtColor(cv2.imread(pre + "_base.png"), cv2.COLOR_BGR2RGB)
        wfg, shorts, eq = load_mask(pre + "_wfg.png"), load_mask(pre + "_shorts.png"), load_mask(pre + "_eq.png")
        # t600 native grade: the render keeps every clip pixel (alpha 1, nothing cut); opacity / coverage / holes are
        # judged on that alpha, bloat on what is VISIBLE over raw backdrop (|out - flat backdrop| > 12)
        alpha = load_mask(pre + "_alpha.png")
        wsol = alpha if alpha is not None else wfg
        vis = (np.abs(lum_of(base.astype(np.float32)) - (0.299 * 35 + 0.587 * 39 + 0.114 * 43)) > 12.0) if alpha is not None else None
        kp = fm["kp"]
        s = fm["step"]
        main_frame = fm["tag"] in ("start", "peak")

        # (d) equipment vs the QC's own per-frame expected footprint (every sidecar frame, whole movement)
        zone = load_mask(pre + "_eqzone.png")
        src_p, plate_p = pre + "_src.png", os.path.join(args.qc_dir, f"step{s}_plate.png")
        # motion-preserving X-Ray edit (edit_of = the pair's base clip): the edit renders arms/torso as dark
        # see-through glass that the colour model of the expected footprint reads as metal, so the expected
        # equipment GEOMETRY comes from the base frame of the same pose (_geo.png) and the base step's plate
        # (pair steps share keypoints, scale and offset, so its plate is in this step's transform)
        geo_p = pre + "_geo.png"
        if os.path.exists(geo_p):
            src_p, plate_p = geo_p, os.path.join(args.qc_dir, f"step{ {1: 3, 2: 4, 3: 1, 4: 2}[s]}_plate.png")
        if zone is not None and os.path.exists(src_p) and os.path.exists(plate_p):
            src = cv2.cvtColor(cv2.imread(src_p), cv2.COLOR_BGR2RGB)
            plate = cv2.cvtColor(cv2.imread(plate_p), cv2.COLOR_BGR2RGB)
            f_, r_, exp = check_equipment(key, src, plate, zone, wsol, shorts, kp, vis)
            eq_fail += f_
            EQ_ROWS.extend(r_)
            core = cv2.erode(exp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if exp.sum() >= 1500:
                # (h) Day 4 see-through over EQUIPMENT: title/preview visible at ui.overlap_alpha over the equipment
                # (not hidden behind it) while the equipment under it is reconstructed intact (never cut/distorted)
                pre_ui_p = pre + "_preui.png"
                ref_ui = cv2.cvtColor(cv2.imread(pre_ui_p), cv2.COLOR_BGR2RGB) if os.path.exists(pre_ui_p) else base
                uia = load_mask(pre + "_uia.png")
                if fr is not None and uia is not None:
                    uirgb = cv2.cvtColor(cv2.imread(pre + "_uirgb.png"), cv2.COLOR_BGR2RGB)
                    band = np.ones((H, W), bool); band[405:430] = False      # title/preview stacking rows
                    free = core & (uia < 0.02)
                    noise = float(np.abs(fr[free].astype(np.float32) - ref_ui[free].astype(np.float32)).max(axis=-1).mean()) if free.sum() > 200 else 3.0
                    a_hat, resid, n_px = ui_see_through(fr, ref_ui, uia, uirgb, core, band)
                    if a_hat is not None:
                        H_ROWS.append(f"{key}: {n_px}px UI over equipment, alpha={a_hat:.2f} (spec {OA_SPEC:.2f}), residual={resid:.1f} (codec {noise:.1f})")
                        if abs(a_hat - OA_SPEC) > OA_TOL or resid > noise + 6.0:
                            ui_fail.append(H_ROWS[-1])
        else:
            eq_fail.append(f"{key}: no equipment sidecar (config: {[e['type'] for e in meta['equipment']]})")
        plate_p = os.path.join(args.qc_dir, f"step{s}_plate.png")
        # (c) mannequin size: pose-invariant skeletal size (torso+thigh+shin+upper arm) on EVERY sidecar frame;
        # the old head-top extent was fooled by overhead arms / seated poses (Day 6 Step 3 rendered 20% larger)
        heights[key] = skeletal_size(kp)
        # (n) NATIVE shorts: every rendered shorts pixel must be a native fabric pixel of the source (0% 2D overpaint)
        src_rgb = cv2.cvtColor(cv2.imread(pre + "_src.png"), cv2.COLOR_BGR2RGB)
        OVP.append((key,) + overpaint_px(src_rgb, shorts))
        # (m) static equipment (bench) preserved as native pixels: not faded, not cut, not re-shaded
        if any(e.get("static") for e in meta["equipment"]) and os.path.exists(plate_p):
            _ref_p = pre + "_ref.png"   # native grade: bench judged against the global tone map of its own pixels
            STATIC.append((key,) + static_check(src_rgb, base, cv2.cvtColor(cv2.imread(plate_p), cv2.COLOR_BGR2RGB), wsol, shorts, kp,
                                                body_lum=min(70.0, float(meta.get("static_body_lum", 60.0)) + 10.0),
                                                ref=cv2.cvtColor(cv2.imread(_ref_p), cv2.COLOR_BGR2RGB) if alpha is not None and os.path.exists(_ref_p) else None))
        # (q) stray / unanchored overlay lines: detached fg components that are visible thin lines or plain-backdrop
        # ghosts (root cause of the Day 6 0:10 line: temporal-median ghosts accepted by the equipment/static layers)
        STRAY.append((key,) + stray_components(wsol, src_rgb, cv2.cvtColor(cv2.imread(plate_p), cv2.COLOR_BGR2RGB) if os.path.exists(plate_p) else None, base, bg))
        if not main_frame:
            continue

        # (a) halo: on the pre-UI composite, coloured pixels outside the dilated body mask that differ from the plate
        body_d = cv2.dilate((wfg > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        halo = ((act_of(base) - bg_act) > 18.0) & ~body_d
        halo_n = int(halo.sum())
        # also: pixels just outside the mask must match the (shadowed) plate in luminance — no glow lift
        ring = (cv2.dilate((wfg > 0.02).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~body_d
        lift = float(np.percentile((lum_of(base) - lum_of(bg))[ring], 95)) if ring.sum() else 0.0
        if halo_n > 150 or lift > 12.0:
            halo_bad.append(f"{key}: {halo_n}px coloured, ring lift p95={lift:.1f}")

        if fr is not None:
            # (b1) Day 4 see-through: title + preview VISIBLE at ui.overlap_alpha over the solid mannequin
            uia = load_mask(pre + "_uia.png")
            if uia is not None:
                uirgb = cv2.cvtColor(cv2.imread(pre + "_uirgb.png"), cv2.COLOR_BGR2RGB)
                pre_ui_p = pre + "_preui.png"
                ref_ui = cv2.cvtColor(cv2.imread(pre_ui_p), cv2.COLOR_BGR2RGB) if os.path.exists(pre_ui_p) else base
                solid = cv2.erode((wfg > 0.9).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                for nm, (y0, y1) in (("title", (430, ty1)), ("preview", (py0, 405))):
                    band = np.zeros((H, W), bool); band[y0:y1] = True
                    a_hat, resid, n_px = ui_see_through(fr, ref_ui, uia, uirgb, solid, band)
                    if a_hat is None:
                        continue
                    B1_ROWS.append(f"{key} {nm}: {n_px}px over body, alpha={a_hat:.2f}, residual={resid:.1f}")
                    if abs(a_hat - OA_SPEC) > OA_TOL or resid > 10.0:
                        title_bad.append(B1_ROWS[-1])
            else:
                title_bad.append(f"{key}: no UI alpha sidecar (_uia.png)")
            # (b2) preview see-through + sharp over mannequin
            reg = np.zeros((H, W), bool); reg[py0:py1, px0:px1] = True
            over = reg & (wfg > 0.9)
            if over.sum() > 200:
                d = float(np.mean(np.abs(fr[over].astype(np.float32) - base[over].astype(np.float32))))
                lb = cv2.Laplacian(lum_of(base), cv2.CV_32F)[over].var()
                lf = cv2.Laplacian(lum_of(fr), cv2.CV_32F)[over].var()
                sharp = lf / max(lb, 1e-3)
                if d > 60.0 or sharp < 0.55:
                    prev_bad.append(f"{key}: diff={d:.1f}, sharpness ratio={sharp:.2f}")


        # (e) bone vs muscle on peak frames (pre-UI composite, inside body)
        if fm["tag"] == "peak":
            a = act_of(base)
            colored = (a > 40.0) & (wfg > 0.8)
            ncol = int(colored.sum())
            lap = np.abs(cv2.Laplacian(lum_of(base), cv2.CV_32F))
            detail = float(np.mean(lap[colored])) if ncol else 0.0
            is_bone = s in meta["bone_steps"]
            ok = ncol > 2500 and detail > 6.0
            RESULTS_E.append((key, is_bone, ncol, detail, ok))

        # (i) colour lock: whole-body skin outside the X-Ray window, per step (compared across steps below)
        l = lum_of(base)
        xw = load_mask(pre + "_xwin.png")
        xw_d = (cv2.dilate((xw > 0.02).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) if xw is not None else np.zeros((H, W), bool)
        sk_o = (wsol > 0.9) & (shorts < 0.2) & (act_of(base) < 12.0) & (l > 46.0) & (eq < 0.5) & ~xw_d
        if alpha is not None:   # native: the warm-grey bench (r-b > 4) is never skin (skin lock tint is cool)
            sk_o &= (base[..., 0].astype(np.float32) - base[..., 2]) <= 4.0
        st_i = load_mask(pre + "_static.png")   # same skin as (f): native static bench highlights are not skin
        if st_i is not None:
            sk_o &= st_i < 0.5
        if sk_o.sum() > 2000:
            LOCK.setdefault(s, []).append((float(np.percentile(l[sk_o], 75)), float(np.percentile(l[sk_o], 90))))
        # (j) arm skin: silver on both arms, no dark fill
        for side, (p75a, blob, npx) in arm_skin_check(base, wfg, eq, xw, kp, cv2.cvtColor(cv2.imread(pre + "_src.png"), cv2.COLOR_BGR2RGB)).items():
            ARM_ROWS.append(f"{key} {side}: p50={p75a:.0f} darkfill={blob}px ({npx}px)")
            # silver range: shaded silver .. highlight (spec curve 46..226); a black fill sits far below it
            if blob > 250 or (npx > 800 and not (100.0 * SK_S <= p75a <= 215.0 * SK_S)):
                arm_fail.append(ARM_ROWS[-1])
        # (k) legs below the shorts hem: silver skin, no dark leak
        for side, (hem, ndk, npx) in leg_below_hem_check(base, wfg, shorts, kp, load_mask(pre + "_eqfoot.png"), cv2.cvtColor(cv2.imread(pre + "_src.png"), cv2.COLOR_BGR2RGB)).items():
            LEG_ROWS.append(f"{key} {side}: hem y={hem} dark={ndk}px/{npx}px")
            if ndk > 40:
                leg_fail.append(LEG_ROWS[-1])
        # (f) skin + shorts
        # (spec: the skin lock excludes only the local X-Ray window, same mask as (i))
        st_m = load_mask(pre + "_static.png")   # native static-equipment pixels (bench highlights) are not skin
        skin = (wsol > 0.9) & (shorts < 0.2) & (act_of(base) < 12.0) & (l > 46.0) & (eq < 0.5) & ~xw_d
        if alpha is not None:
            skin &= (base[..., 0].astype(np.float32) - base[..., 2]) <= 4.0
        if st_m is not None:
            skin &= st_m < 0.5
        if skin.sum() > 2000:
            # max = p99.9: the statistic the lock anchors and the Day 4 reference was measured with (t542u)
            p75, p90, mx = np.percentile(l[skin], 75), np.percentile(l[skin], 90), np.percentile(l[skin], 99.9)
            sh = (shorts > 0.6) & (wsol > 0.9)
            sp = float(np.median(l[sh])) if sh.sum() else -1
            SKIN.append(f"{key}: p75={p75:.0f} p90={p90:.0f} max={mx:.0f} shorts={sp:.0f}")
            if not (SK["qc_p75_range"][0] <= p75 <= SK["qc_p75_range"][1] and SK["qc_p90_range"][0] <= p90 <= SK["qc_p90_range"][1]
                    and mx <= SK["qc_max"] and SH_RANGE[0] <= sp <= SH_RANGE[1]):
                skin_fail.append(SKIN[-1])

    report("(a) no colour halo / bloom outside mannequin", not halo_bad, "; ".join(halo_bad) or f"{len(frames_meta)} frames clean")
    report("(b1) title/preview see-through over mannequin (Day 4 rule)", not title_bad, "; ".join(title_bad) or (" | ".join(B1_ROWS[:6]) if B1_ROWS else "no UI/body overlap in sampled frames"))
    report("(b2) preview translucent + sharp over mannequin", not prev_bad, "; ".join(prev_bad) or "mannequin shows through preview, no frost blur")
    if heights:
        steps_h = {}
        for k, v in heights.items():
            steps_h.setdefault(k.split("_")[0], []).append(v)
        hs = {k: float(np.median(v)) for k, v in steps_h.items()}
        spread = (max(hs.values()) - min(hs.values())) / max(hs.values())
        report("(c) mannequin size equal across steps (skeletal, <=4%)", spread <= 0.04,
               ", ".join(f"{k}={v:.0f}px" for k, v in sorted(hs.items())) + f" | spread {100 * spread:.1f}%")
    ov_bad = [f"{k}: {n}px painted outside native fabric (of {t}px shorts)" for k, n, t in OVP if n > 0]
    tot = sum(t for _, _, t in OVP)
    report("(n) 2d_overpaint_ratio == 0 (shorts = native 3D fabric pixels only)", not ov_bad,
           "; ".join(ov_bad) or f"2d_overpaint_ratio=0.000 over {len(OVP)} frames ({tot}px native shorts)")
    # (s) pair steps = SAME motion (Day 6+): 1/3 (wrong) and 2/4 (right) must share the clip motion. Per QC tag:
    # keypoint trajectory (mean joint distance / skeletal size) AND the rendered body silhouette (each step's
    # own pixels, static bench excluded) must match; only the anatomy rendering may differ.
    if int(meta.get("day") or 0) >= 6:
        pr_bad, pr_rows = [], []
        joints = ("sh_r", "sh_l", "elb_r", "elb_l", "wr_r", "wr_l", "hip_r", "hip_l", "kn_r", "kn_l")
        PAIRS_Q = () if int(meta.get("n_steps", 4)) == 2 else ((1, 3), (2, 4))   # t568u 2-step day: no front/side pair
        if not PAIRS_Q:
            pr_rows.append("2-step day (t568u): 1 wrong / 2 right, no paired front/side steps")
        for a_, b_ in PAIRS_Q:
            ca, cb = (meta.get("clips") or {}).get(str(a_), {}), (meta.get("clips") or {}).get(str(b_), {})
            if not ca or ca.get("motion") is None or ca.get("motion") != cb.get("motion"):
                pr_bad.append(f"steps {a_}/{b_}: not derived from one motion ({ca.get('file')} vs {cb.get('file')})")
                continue
            if ca.get("camera") in ("front", "side") and cb.get("camera") in ("front", "side") and ca["camera"] != cb["camera"]:
                pr_rows.append(f"{a_}/{b_}: {ca['camera']} vs {cb['camera']} camera (t556u) - 2D pose not comparable, gated by (y)")
                continue
            for tag in sorted({fm["tag"] for fm in frames_meta.values()}):
                ka, kb = frames_meta.get(f"step{a_}_{tag}"), frames_meta.get(f"step{b_}_{tag}")
                if not ka or not kb:
                    continue
                sz = skeletal_size(ka["kp"])
                dev = float(np.mean([np.hypot(ka["kp"][j][0] - kb["kp"][j][0], ka["kp"][j][1] - kb["kp"][j][1]) for j in joints])) / max(sz, 1.0)
                wa = load_mask(os.path.join(args.qc_dir, f"step{a_}_{tag}_wfg.png"))
                wb = load_mask(os.path.join(args.qc_dir, f"step{b_}_{tag}_wfg.png"))
                sa = load_mask(os.path.join(args.qc_dir, f"step{a_}_{tag}_static.png"))
                sb = load_mask(os.path.join(args.qc_dir, f"step{b_}_{tag}_static.png"))
                ma = (wa > 0.5) & ((sa if sa is not None else 0) < 0.5)
                mb = (wb > 0.5) & ((sb if sb is not None else 0) < 0.5)
                iou = float((ma & mb).sum()) / max(1.0, float((ma | mb).sum()))
                pm = ca.get("pair_match") or cb.get("pair_match")
                pr_rows.append(f"{a_}/{b_} {tag}: kp {100 * dev:.1f}% iou {iou:.2f}" + (" (joint-tolerance pair)" if pm else ""))
                # t541u relaxed rule for a measured native pair: same motion + angle within joint tolerance; the
                # silhouette of two different native clips is reported, not gated
                tol_j = float(_SPEC_Q["step_order"].get("pair_tolerance", {}).get("joint_dev_pct", 15.0)) / 100.0
                if (dev > tol_j) if pm else (dev > 0.03 or iou < 0.85):
                    pr_bad.append(pr_rows[-1])
        report("(s) pair steps share the same motion (1/3 wrong, 2/4 right): keypoints + silhouette", not pr_bad,
               "; ".join(pr_bad) or " | ".join(pr_rows[:8]))
        # (y) t556u: front (1/2) and side (3/4) views of a form come from the SAME motion. A yaw camera change keeps
        # vertical joint positions: compare wrist/elbow/shoulder/knee heights relative to the hip midpoint per
        # tempo tag (same phase), in units of the mean skeletal size; tolerance = pair tolerance for matched
        # native pairs, else 6 %.
        y_bad, y_rows = [], ([] if PAIRS_Q else ["2-step day: no front/side pair"])
        for a_, b_ in PAIRS_Q:
            ca, cb = (meta.get("clips") or {}).get(str(a_), {}), (meta.get("clips") or {}).get(str(b_), {})
            pm = ca.get("pair_match") or cb.get("pair_match")
            tol_y = float(_SPEC_Q["step_order"].get("pair_tolerance", {}).get("joint_dev_pct", 15.0)) / 100.0 if pm else 0.06
            devs = []
            for tag in sorted({fm["tag"] for fm in frames_meta.values()}):
                ka, kb = frames_meta.get(f"step{a_}_{tag}"), frames_meta.get(f"step{b_}_{tag}")
                if not ka or not kb:
                    continue
                def vprof(k):
                    hy = 0.5 * (k["hip_r"][1] + k["hip_l"][1])
                    return np.array([k[j_][1] - hy for j_ in ("wr_r", "wr_l", "elb_r", "elb_l", "sh_r", "sh_l", "kn_r", "kn_l")])
                sz = 0.5 * (skeletal_size(ka["kp"]) + skeletal_size(kb["kp"]))
                devs.append(float(np.mean(np.abs(vprof(ka["kp"]) - vprof(kb["kp"])))) / max(sz, 1.0))
            if devs:
                y_rows.append(f"{a_}/{b_} ({ca.get('camera')} vs {cb.get('camera')}): vertical-profile dev mean {100 * np.mean(devs):.1f}% max {100 * max(devs):.1f}% (tol {100 * tol_y:.0f}%)")
                if max(devs) > tol_y:
                    y_bad.append(y_rows[-1])
        report("(y) front and side views come from the same motion (vertical joint profile per tempo phase)", not y_bad,
               "; ".join(y_bad) or " | ".join(y_rows))
        # (z) t557u: paired steps show the same callout angle (+-5 deg) and label at every tempo phase
        z_bad, z_rows = [], []
        if not PAIRS_Q:   # 2-step day: each step keeps ONE angle + label at every tempo phase (+-5 deg)
            for s_ in (1, 2):
                cs_ = [fm_["callout"] for k_, fm_ in sorted(frames_meta.items()) if fm_["step"] == s_ and fm_.get("callout")]
                if not cs_:
                    z_bad.append(f"step{s_}: no callout"); continue
                dd_ = [int(str(c_["deg"]).rstrip("\u00b0")) for c_ in cs_]
                if max(dd_) - min(dd_) > 5 or len({c_["l1"] for c_ in cs_}) != 1:
                    z_bad.append(f"step{s_}: angles {sorted(set(dd_))} labels {sorted({c_['l1'] for c_ in cs_})}")
                z_rows.append(f"step{s_}: {cs_[0]['deg']} {cs_[0]['l1']} on {len(cs_)} phases")
        for a_, b_ in PAIRS_Q:
            for tag in sorted({fm["tag"] for fm in frames_meta.values()}):
                ka, kb = frames_meta.get(f"step{a_}_{tag}"), frames_meta.get(f"step{b_}_{tag}")
                if not ka or not kb or not ka.get("callout") or not kb.get("callout"):
                    continue
                da, db = (int(str(k["callout"]["deg"]).rstrip("\u00b0")) for k in (ka, kb))
                ok_z = abs(da - db) <= 5 and ka["callout"]["l1"] == kb["callout"]["l1"]
                if not ok_z:
                    z_bad.append(f"{a_}/{b_} {tag}: {da}\u00b0 {ka['callout']['l1']!r} vs {db}\u00b0 {kb['callout']['l1']!r}")
            pk_a, pk_b = frames_meta.get(f"step{a_}_peak", {}).get("callout"), frames_meta.get(f"step{b_}_peak", {}).get("callout")
            if pk_a and pk_b:
                z_rows.append(f"{a_}/{b_} peak: {pk_a['deg']} {pk_a['l1']} vs {pk_b['deg']} {pk_b['l1']}")
        report("(z) paired steps show the same callout angle (+-5 deg) and label", not z_bad, "; ".join(z_bad[:6]) or " | ".join(z_rows))
    if int(meta.get("n_steps", 4)) == 2:
        t567_checks(meta, frames_meta, vids, args.qc_dir)
    if int(meta.get("day") or 0) >= 6:
        flat_look_checks(meta, frames_meta, vids, args.qc_dir)
    sq_bad = [f"{k}: {msg}" for k, ok, msg in STRAY if not ok]
    report("(q) no stray / unanchored overlay lines or blobs", not sq_bad,
           "; ".join(sq_bad) or f"{len(STRAY)} frames: all fg components attached to mannequin/equipment")
    from gymform.ctx import order_lint
    o_bad = order_lint({"day": meta.get("day"), "n_steps": int(meta.get("n_steps", 4)), "clips": meta.get("clips") or {}, "bone_steps": meta["bone_steps"],
                        "muscle_steps": meta["muscle_steps"], "callouts": meta.get("callouts") or {},
                        "xray_full_steps": meta.get("xray_full_steps") or meta["bone_steps"]}) if meta.get("clips") else ["clips missing in meta"]
    report("(r) step order locked (1 wrong bone, 2 right bone, 3 wrong muscle, 4 right muscle)", not o_bad,
           "; ".join(o_bad) or f"bone {meta['bone_steps']} muscle {meta['muscle_steps']} wrong={meta.get('is_wrong')}")
    if STATIC:
        st_bad = [f"{k}: {msg}" for k, ok, msg in STATIC if not ok]
        report("(m) static equipment (bench) preserved: native, opaque, not faded/cut", not st_bad,
               "; ".join(st_bad) or " | ".join(f"{k}: {m}" for k, _, m in STATIC[:6]))
    # (p) on-screen angle numbers agree with label / config / TTS (Day 6: 121 deg and 74 deg under 'ELBOWS 45 deg')
    import re as _re
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gymform.layout import callout_lint
    p_bad = []
    if meta.get("callouts"):
        p_bad += callout_lint({"callouts": meta["callouts"], "tts_lines": meta.get("tts_lines", []), "n_steps": int(meta.get("n_steps", 4))})
        for key, fm in sorted(frames_meta.items()):
            co, c = fm.get("callout"), meta["callouts"][str(fm["step"])]
            if not co:
                p_bad.append(f"{key}: no drawn-callout sidecar"); continue
            v = int(co["deg"].rstrip("\u00b0"))
            d = c["deg"]
            lo, hi = (sorted((d["from"], d["to"])) if isinstance(d, dict) else (int(d[:-1]), int(d[:-1])))
            lab = [int(x) for x in _re.findall(r"(\d+)\u00b0", co["l1"])]
            if not (lo <= v <= hi) or (lab and lab[0] != v):
                p_bad.append(f"{key}: drew {co['deg']} under '{co['l1']}' (config {lo}-{hi})")
    else:
        p_bad.append("meta has no callout config")
    shown = sorted({(fm["step"], fm.get("callout", {}).get("deg")) for fm in frames_meta.values() if fm.get("tag") == "peak"})
    report("(p) on-screen angle numbers match label / config / TTS", not p_bad,
           "; ".join(p_bad) or " | ".join(f"step{s} peak {d}" for s, d in shown))
    prov = meta.get("source", {})
    report("(o) native_3d_source (config provenance, no 2D overpaint stage)", native_source_ok(prov)[0], native_source_ok(prov)[1])
    report("(d) equipment = per-frame expected footprint: solid, no holes, in frame, not clipped", not eq_fail,
           "; ".join(eq_fail) or f"types={[e['type'] for e in meta['equipment']]} | " + " | ".join(EQ_ROWS))
    report("(h) title/preview see-through over equipment, equipment intact under it", not ui_fail,
           "; ".join(ui_fail) or (" | ".join(H_ROWS) if H_ROWS else "no equipment under UI in sampled frames"))
    report("(j) both arms silver skin, no dark fill (keypoint segments)", not arm_fail,
           "; ".join(arm_fail) or " | ".join(ARM_ROWS))
    report("(k) no dark pixels in leg skin below the shorts hem", not leg_fail, "; ".join(leg_fail) or " | ".join(LEG_ROWS[:8]))
    for key, is_bone, ncol, detail, ok in RESULTS_E:
        report(f"(e) {key} {'bone X-Ray' if is_bone else 'muscle fibers'} present/readable", ok, f"coloured px={ncol}, edge detail={detail:.1f}")
    report(f"(f) skin = Day 4 tone p75~{SK['p75']}/p90~{SK['p90']}, shorts {SH_RANGE[0]}..{SH_RANGE[1]}", not skin_fail, " | ".join(SKIN))
    if LOCK:
        p75s = {k: float(np.mean([v[0] for v in vs])) for k, vs in LOCK.items()}
        p90s = {k: float(np.mean([v[1] for v in vs])) for k, vs in LOCK.items()}
        r75 = max(p75s.values()) / max(1.0, min(p75s.values()))
        r90 = max(p90s.values()) / max(1.0, min(p90s.values()))
        report("(i) mannequin colour locked across steps outside X-Ray window (p75/p90 within 5%)", r75 <= 1.05 and r90 <= 1.05,
               ", ".join(f"step{k} p75={p75s[k]:.0f}/p90={p90s[k]:.0f}" for k in sorted(p75s)) + f" | spread p75 {100 * (r75 - 1):.1f}%, p90 {100 * (r90 - 1):.1f}%")

    # t552u: bones like muscle fibre. (w) no bone pixels outside the exercise's target-bone region;
    # (x) bone-step skin = Day 4 tone everywhere (incl. right next to the bones): same (f) ranges + (i) 5% vs the
    # muscle steps + Day 4 tint, measured WITHOUT the old X-Ray-window exclusion (only the shown bone px +5 px).
    tb = meta.get("target_bones") or []
    if tb and meta.get("bone_steps"):   # muscle-mistake days have no bone steps (t554u)
        from gymform import bones as _bones
        w_bad, w_rows, x_bad, x_rows, x_stats = [], [], [], [], {}
        mus = {s_: LOCK[s_] for s_ in LOCK if s_ not in meta["bone_steps"]}
        m75 = float(np.mean([v[0] for vs in mus.values() for v in vs])) if mus else float(SK["p75"])
        m90 = float(np.mean([v[1] for vs in mus.values() for v in vs])) if mus else float(SK["p90"])
        for key, fm in sorted(frames_meta.items()):
            if fm["step"] not in meta["bone_steps"]:
                continue
            pre = os.path.join(args.qc_dir, key)
            base = cv2.cvtColor(cv2.imread(pre + "_base.png"), cv2.COLOR_BGR2RGB).astype(np.float32)
            wfg, shorts, eq = load_mask(pre + "_wfg.png"), load_mask(pre + "_shorts.png"), load_mask(pre + "_eq.png")
            st_m, xw = load_mask(pre + "_static.png"), load_mask(pre + "_xwin.png")
            reg = _bones.region(fm["kp"], tb, (H, W))
            l, a = lum_of(base), act_of(base)
            reg_d = cv2.dilate((reg > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            body = (wfg > 0.5) & (shorts < 0.3) & ((eq < 0.5) if eq is not None else True) & ((st_m < 0.5) if st_m is not None else True)
            stray = body & ~reg_d & (a > 25.0) & (l > 110.0)
            n_out = int(cv2.morphologyEx(stray.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).sum())
            w_rows.append(f"{key}: {n_out}px")
            if n_out > 40:
                w_bad.append(w_rows[-1])
            bone_d = cv2.dilate(((xw if xw is not None else np.zeros((H, W))) > 0.02).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
            skin = (wfg > 0.9) & (shorts < 0.2) & (l > 46.0) & (a < 12.0) & ~bone_d & ((eq < 0.5) if eq is not None else True) & ((st_m < 0.5) if st_m is not None else True)
            near = skin & reg_d          # skin right around / between the shown bones (old window area)
            if skin.sum() < 2000:
                continue
            p75, p90 = float(np.percentile(l[skin], 75)), float(np.percentile(l[skin], 90))
            n50 = float(np.percentile(l[near], 50)) if near.sum() > 500 else -1.0
            s50 = float(np.percentile(l[skin & ~reg_d], 50)) if (skin & ~reg_d).sum() > 500 else n50
            tint = [float(np.median(base[:, :, c][skin] - l[skin])) for c in range(3)]
            x_stats.setdefault(fm["step"], []).append((p75, p90))
            x_rows.append(f"{key}: p75={p75:.0f} p90={p90:.0f} near-bone p50={n50:.0f} vs rest {s50:.0f} tint={'/'.join(f'{t:+.0f}' for t in tint)}")
            ok = (SK["qc_p75_range"][0] <= p75 <= SK["qc_p75_range"][1] and SK["qc_p90_range"][0] <= p90 <= SK["qc_p90_range"][1]
                  and abs(p75 / m75 - 1) <= 0.05 and abs(p90 / m90 - 1) <= 0.05
                  and all(abs(t - t0) <= 3.0 for t, t0 in zip(tint, SK.get("tint_rgb", [0, 0, 0])))
                  and (n50 < 0 or abs(n50 / max(s50, 1) - 1) <= 0.08))
            if not ok:
                x_bad.append(x_rows[-1])
        report(f"(w) bones only inside the target region ({', '.join(tb)})", not w_bad, "; ".join(w_bad) or " | ".join(w_rows[:8]))
        report(f"(x) bone-step skin = Day 4 tone (f ranges, <=5% vs muscle steps p75 {m75:.0f}/p90 {m90:.0f}, tint +-3, near-bone <=8%)",
               not x_bad, "; ".join(x_bad) or " | ".join(x_rows[:8]))

    # (g) X-Ray look vs approved Day 1 v36 reference: luminous bone ramp (tint -> hot core), glassy darkened body
    if os.path.exists(args.xray_ref):
        ref = cv2.cvtColor(cv2.imread(args.xray_ref), cv2.COLOR_BGR2RGB)
        ref_m = np.zeros(ref.shape[:2], np.float32)
        ref_m[600:1500, 600:1000] = 1.0  # Day 1 X-Ray arm region (excludes red UI text/arc)
        ref_sig = xray_signature(ref, ref_m, "crimson")
        for key, fm in sorted(frames_meta.items()):
            if fm["tag"] != "peak" or fm["step"] not in meta["bone_steps"]:
                continue
            base = cv2.cvtColor(cv2.imread(os.path.join(args.qc_dir, key) + "_base.png"), cv2.COLOR_BGR2RGB)
            wfg = load_mask(os.path.join(args.qc_dir, key) + "_wfg.png")
            iw = meta.get("is_wrong", {}).get(str(fm["step"]))
            tint = ("crimson" if iw else "emerald") if iw is not None else ("crimson" if fm["step"] == meta["bone_steps"][0] else "emerald")
            sig = xray_signature(base, wfg, tint)
            ok = (sig["n"] > 1500 and abs(sig["hot_ratio"] - ref_sig["hot_ratio"]) < 0.30
                  and sig["core_ratio"] > 0.5 * ref_sig["core_ratio"] and sig["detail"] > 0.55 * ref_sig["detail"])
            report(f"(g) {key} X-Ray matches Day 1 v36 look ({tint})", ok,
                   f"bone px={sig['n']} hot/primary={sig['hot_ratio']:.2f} (ref {ref_sig['hot_ratio']:.2f}) "
                   f"hot-core share={sig['core_ratio']:.2f} (ref {ref_sig['core_ratio']:.2f}) "
                   f"detail={sig['detail']:.1f} (ref {ref_sig['detail']:.1f})")
    else:
        report("(g) X-Ray vs Day 1 reference", False, f"reference frame missing: {args.xray_ref}")

    # (u) bone visibility on every bone-step QC frame: luminous near-white bones, visible against body + backdrop
    bv_bad, bv_rows = [], []
    for key, fm in sorted(frames_meta.items()):
        if fm["step"] not in meta["bone_steps"]:
            continue
        pre = os.path.join(args.qc_dir, key)
        if not (os.path.exists(pre + "_base.png") and os.path.exists(pre + "_xwin.png")):
            continue
        base = cv2.cvtColor(cv2.imread(pre + "_base.png"), cv2.COLOR_BGR2RGB)
        region = (load_mask(pre + "_wfg.png") > 0.5) & (load_mask(pre + "_xwin.png") > 0.3)
        mv = bone_visibility(base, region)
        bv_rows.append(f"{key}: L50={mv['L50']:.0f} L90={mv['L90']:.0f} contrast={mv['contrast']:.0f} dark={mv['dark_sat']:.2f}")
        if not bone_visible_ok(mv):
            bv_bad.append(bv_rows[-1])
    if meta.get("bone_steps"):   # t554u/t568u: muscle-only days show no bones at all (nothing to check)
     report(f"(u) bones visible: L50>={BONE_VIS['L50_min']:.0f} L90>={BONE_VIS['L90_min']:.0f} contrast>={BONE_VIS['contrast_min']:.0f} "
           f"dark-sat<={BONE_VIS['dark_sat_max']:.2f} (Day 1 v36 calibrated)", bool(bv_rows) and not bv_bad,
           "; ".join(bv_bad) or " | ".join(bv_rows[:4]) or "no bone-step QC frames")

    # (l) spoken end question (Day 6+): present, short, after the Step 4 line, inside the 20.0s reel
    spec = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gymform_spec.json")))
    sq = spec["caption_rules"].get("spoken_question", {})
    tp = os.path.join(args.qc_dir, "tts_timing.json")
    if sq and int(meta.get("day", 0) or 0) >= int(sq.get("from_day", 6)) or os.path.exists(tp):
        segs = json.load(open(tp))["segments"] if os.path.exists(tp) else []
        q = [g for g in segs if g["role"] == "question"]
        dur_ = float(meta.get("duration_s", 20.0)); last_ = f"step{int(meta.get('n_steps', 4))}"
        s4 = [g for g in segs if g["role"] == last_]
        ok = bool(q and s4) and q[0]["text"].strip().endswith("?") and len(q[0]["text"].split()) <= int(sq.get("max_words", 6)) \
            and q[0]["start"] >= s4[0]["end"] + 0.15 and q[0]["end"] <= float(sq.get("latest_end_s", 19.85)) + dur_ - 20.0 and n / max(fps, 1) >= dur_ - 0.05
        report(f"(l) spoken end question after the last step line, ends inside {dur_:.1f}s", ok,
               (f"{q[0]['text']!r} {q[0]['start']:.2f}-{q[0]['end']:.2f}s (last step line ends {s4[0]['end']:.2f}s)" if q and s4
                else "no question segment in tts_timing.json"))

    # (v) tempo = Day 4 baseline (t544u): smooth cosine, turnarounds at frames 50/100 of every 5 s step, no flat holds
    from gymform import tempo as _tempo
    if args.preflight:
        rv = _tempo.reversal_frames()
        report("(v) tempo wave = Day 4 (turnarounds 50/100 per step)", rv == list(_tempo.TEMPO.get("reversal_frames", rv)),
               f"{_tempo.TEMPO.get('style')} reversals {rv}")
    else:
        ok_v, exp_v, bad_v = _tempo.tempo_ok(_tempo.measure_reversals(args.mp4))
        report("(v) tempo = Day 4 (motion turnarounds at frames 50/100 of each step, +-6)", ok_v,
               f"expected {exp_v}; off: {bad_v}" if bad_v else f"all 4 steps turn at {exp_v}")

    if int(meta.get("day") or 0) >= 6:
        playbook_frame_checks(frames_meta, args.qc_dir)
        if not args.preflight:
            playbook_temporal_check(meta, args.mp4)

    print(f"\nQC GATE: {'PASS' if all(RESULTS) else 'FAIL'} ({sum(RESULTS)}/{len(RESULTS)} items)")
    sys.exit(0 if all(RESULTS) else 1)


OVP, STATIC, STRAY = [], [], []
CFG_EQ = [[]]
_SPEC_Q = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gymform_spec.json")))
SK = _SPEC_Q["mannequin"]["skin_lock_all_steps"]          # t542u: Day 4 skin tone is the target + QC reference
SH_RANGE = _SPEC_Q["mannequin"]["shorts_lum_range"]
SK_S = SK["p75"] / 165.0                                    # scale of the silver-era absolute thresholds to the Day 4 tone
RESULTS_E, SKIN, EQ_ROWS, H_ROWS, LOCK, ARM_ROWS, arm_fail, LEG_ROWS, leg_fail = [], [], [], [], {}, [], [], [], []

if __name__ == "__main__":
    main()
