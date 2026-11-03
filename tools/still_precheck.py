#!/usr/bin/env python3
"""Reusable still pre-check before any i2v (t693u). One JSON config per exercise/camera (configs/precheck/*.json).
Checks, each PASS/FAIL with the number:
  machine   SSIM vs the empty plate per ROI, outside the dilated body mask and the moving handle/cable band
  cable     dark thin line from the pulley point to the hands (fraction of samples darker than both side neighbours)
  arm       START: elbow flexion <= max (arms extended);  PEAK: elbow behind the shoulder by >= frac x torso length
  torso     lean from vertical <= max per still; |start - peak| <= delta (pair mode)
  handle    rectangle-frame detector: bright metal pixels in a box ABOVE the hands (a V handle has none there)
  head      head vs body luminance |dL|
  light     shaded side (p10 of upper-body luminance) minus backdrop median >= min
  glow      green/red dominance pixels on the body
usage: python3 tools/still_precheck.py CONFIG --start S.png [--peak P.png] [--out json]
       python3 tools/still_precheck.py CONFIG --plate-handle IMG [--person] [--out json]   (machine-only plate + handle:
       machine SSIM outside the handle, cable pulley->eyelet, attached, unbroken, V/triangle-not-rectangle shape)"""
import sys, json, argparse, cv2, numpy as np, mediapipe as mp
from skimage.metrics import structural_similarity as ssim
sys.path.insert(0, "/workspace/daily_info_reels")
from universal_anatomy_rig import get_pose_detector
R = "/workspace/daily_info_reels/"
def pose(img):
    r = get_pose_detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks: return None, None
    h, w = img.shape[:2]; L = r.pose_landmarks[0]
    return np.array([[p.x * w, p.y * h, p.visibility] for p in L]), (r.segmentation_masks[0].numpy_view()[..., 0] > 0.3).astype(np.uint8)
def ang(a, b, c):
    v1, v2 = a - b, c - b; return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))))
def check(cfg, path, kind):
    img = cv2.imread(path); H, W = cfg["size"][1], cfg["size"][0]
    if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    plate = cv2.imread(R + cfg["plate"]); out = {"file": path, "kind": kind}
    kp, seg = pose(img)
    if kp is None: return {**out, "pass": False, "error": "no pose"}
    g, pg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    excl = cv2.dilate(seg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool)
    y0, y1, x0, x1 = cfg["moving_band"]; band = np.zeros_like(excl); band[y0:y1, x0:x1] = True
    _, sm = ssim(pg, g, full=True); res = {}
    for k, (a, b, c, d) in cfg["machine_rois"].items():
        m = ~(excl | band)[a:b, c:d]; res[k] = round(float(sm[a:b, c:d][m].mean()), 3) if m.sum() > 500 else None
    allm = ~(excl | band); res["all_static"] = round(float(sm[allm].mean()), 3)
    vals = [v for v in res.values() if v is not None]
    out["machine"] = {"ssim": res, "pass": min(vals) >= cfg["machine_ssim_min"]}
    side = 11 if kp[11, 2] >= kp[12, 2] else 12
    s, e, wr = kp[side, :2], kp[side + 2, :2], kp[side + 4, :2]
    sh = (kp[11, :2] + kp[12, :2]) / 2; hp = (kp[23, :2] + kp[24, :2]) / 2; tl = float(np.linalg.norm(sh - hp))
    sgn = 1 if cfg.get("tower_side", "left") == "left" else -1   # +x = away from the tower = backward
    lean = float(np.degrees(np.arctan2(sgn * (sh[0] - hp[0]), hp[1] - sh[1])))
    flex = 180 - ang(s, e, wr); past = sgn * float(e[0] - s[0]) / tl
    out["pose"] = {"torso_lean_deg": round(lean, 1), "elbow_flex_deg": round(flex, 1), "elbow_behind_shoulder_frac": round(past, 3), "torso_px": round(tl, 1)}
    out["torso"] = {"pass": abs(lean) <= cfg["torso_lean_max"]}
    if kind == "start": out["arm"] = {"rule": f"elbow flex <= {cfg['start']['elbow_flex_max']}", "pass": flex <= cfg["start"]["elbow_flex_max"]}
    else: out["arm"] = {"rule": f"elbow behind shoulder >= {cfg['peak']['elbow_past_shoulder_min_frac']} torso", "pass": past >= cfg["peak"]["elbow_past_shoulder_min_frac"]}
    # cable: samples from the pulley to the hand midpoint, skip the last 12% (hands/handle) and the first 4% (pulley)
    hand = (kp[15, :2] + kp[16, :2]) / 2 if min(kp[15, 2], kp[16, 2]) > 0.2 else wr
    gb = cv2.GaussianBlur(g, (3, 3), 0).astype(float)
    def cover(p0):
        v = hand - p0; L = np.linalg.norm(v); u = v / (L + 1e-6); nrm = np.array([-u[1], u[0]]); hits = tot = 0
        for t in np.linspace(0.06, 0.88, 60):
            c = p0 + v * t; best = 0
            for off in range(-3, 4):
                q = c + nrm * off; xi, yi = int(q[0]), int(q[1])
                if not (3 <= xi < W - 3 and 3 <= yi < H - 3): continue
                sa, sb = c + nrm * (off - 5), c + nrm * (off + 5)
                va = gb[int(np.clip(sa[1], 0, H - 1)), int(np.clip(sa[0], 0, W - 1))]; vb = gb[int(np.clip(sb[1], 0, H - 1)), int(np.clip(sb[0], 0, W - 1))]
                best = max(best, min(va, vb) - gb[yi, xi])
            tot += 1; hits += best >= 6
        return hits / max(tot, 1), L
    pr = cfg.get("pulley_search_px", 24); cands = []
    for dx in range(-pr, pr + 1, 4):
        for dy in range(-pr, pr + 1, 4):
            p0 = np.array(cfg["pulley_xy"], float) + [dx, dy]; cv_, L = cover(p0); cands.append((cv_, L, p0))
    cov, L, p0b = max(cands, key=lambda t: t[0]); hits = tot = 1
    out["cable"] = {"cover": round(cov, 2), "len_px": round(float(L), 1), "pulley_used": p0b.round(0).tolist(), "pass": cov >= cfg["cable_min_cover"]}
    # handle: bright low-sat metal pixels in a box above the hands (rectangle frame top bar); a V handle sits below/at the grip
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); metal = (hsv[..., 2] > 150) & (hsv[..., 1] < 45) & ~seg.astype(bool)
    hx, hy = int(hand[0]), int(hand[1]); bx = metal[max(0, hy - 75):max(0, hy - 22), max(0, hx - 70):hx + 70]
    out["handle"] = {"rect_top_metal_px": int(bx.sum()), "pass": int(bx.sum()) <= cfg["handle_rect_top_px_max"]}
    # head vs body tone, light, glow
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float) * 100 / 255
    head = np.zeros_like(seg); hc = kp[[0, 7, 8], :2].mean(0); hr = int(0.22 * tl)
    cv2.circle(head, (int(hc[0]), int(hc[1])), hr, 1, -1); head &= seg
    upper = seg.copy(); upper[int(hp[1]) - 10:, :] = 0; upper &= (1 - cv2.dilate(head, np.ones((9, 9), np.uint8)))
    hL, bL = float(np.median(lum[head > 0])), float(np.median(lum[upper > 0]))
    out["head"] = {"head_L": round(hL, 1), "body_L": round(bL, 1), "dL": round(abs(hL - bL), 1), "pass": abs(hL - bL) <= cfg["head_body_dL_max"]}
    bgm = ~cv2.dilate(seg, np.ones((61, 61), np.uint8)).astype(bool); bgm[:, :240] = False; bgm[1000:, :] = False
    bgL = float(np.median(lum[bgm]))
    # shaded SIDE, not crevices: low-pass luminance (sigma 8) inside the eroded upper body, then p10
    er = cv2.erode(upper, np.ones((11, 11), np.uint8)).astype(bool)
    lp = cv2.GaussianBlur(lum * upper, (0, 0), 8) / np.maximum(cv2.GaussianBlur(upper.astype(float), (0, 0), 8), 1e-3)
    p10 = float(np.percentile(lp[er], 10))
    out["light"] = {"body_p10_L": round(p10, 1), "backdrop_L": round(bgL, 1), "diff": round(p10 - bgL, 1), "pass": p10 - bgL >= cfg["shaded_vs_backdrop_min"]}
    b, gr, r = [img[..., i].astype(int) for i in range(3)]
    glow = ((gr - np.maximum(r, b) > 25) | (r - np.maximum(gr, b) > 35)) & seg.astype(bool)
    gf = float(glow.sum() / max(seg.sum(), 1)); out["glow"] = {"frac": round(gf, 4), "pass": gf <= cfg["glow_frac_max"]}
    out["kp"] = {"shoulder": s.round(1).tolist(), "elbow": e.round(1).tolist(), "wrist": wr.round(1).tolist(), "hand_mid": hand.round(1).tolist(), "sh_mid": sh.round(1).tolist(), "hip_mid": hp.round(1).tolist()}
    keys = ["machine", "torso", "arm", "cable", "handle", "head", "light", "glow"]
    out["fails"] = [k for k in keys if not out[k]["pass"]]; out["pass"] = not out["fails"]
    return out


# ---------------------------------------------------------------- handle checks (t694u) ----------------------------
def handle_mask(cfg, img, person=False):
    """Pixels the still ADDS vs the empty plate inside the moving band (handle + cable), minus the person and the
    resting-handle removal boxes. Returns (cable+handle mask, handle-only mask after opening away the thin cable)."""
    P = cv2.imread(R + cfg["plate"]); H, W = P.shape[:2]
    y0, y1, x0, x1 = cfg["moving_band"]
    d = np.abs(img.astype(int) - P.astype(int)).max(2) > 18
    m = np.zeros((H, W), np.uint8); m[y0:y1, x0:x1] = d[y0:y1, x0:x1]
    for (a, b, c, e) in cfg.get("remove_boxes", []): m[a:b, c:e] = 0
    if person:
        _, seg = pose(img)
        if seg is not None: m[cv2.dilate(seg, np.ones((9, 9), np.uint8)) > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    h = cv2.morphologyEx(h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(h)
    keep = np.zeros_like(h)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= cfg.get("handle_min_blob_px", 60): keep[lab == i] = 1
    return m, keep

def check_plate_handle(cfg, path, person=False):
    img = cv2.imread(path); P = cv2.imread(R + cfg["plate"]); H, W = P.shape[:2]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if img.shape[:2] != (H, W) else img
    out = {"file": path, "kind": "plate_handle"}
    allm, hm = handle_mask(cfg, img, person)
    g, pg = cv2.cvtColor(img, 6), cv2.cvtColor(P, 6); _, sm = ssim(pg, g, full=True)
    changed = cv2.dilate(allm, np.ones((15, 15), np.uint8)).astype(bool)
    for (a, b, c, e) in cfg.get("remove_boxes", []): changed[a:b, c:e] = True
    res = {}
    for k, (a, b, c, e) in cfg["machine_rois"].items():
        mm = ~changed[a:b, c:e]; res[k] = round(float(sm[a:b, c:e][mm].mean()), 4)
    res["all_outside_handle"] = round(float(sm[~changed].mean()), 4)
    out["machine"] = {"ssim": res, "pass": min(res.values()) >= cfg.get("plate_machine_ssim_min", 0.99)}
    if hm.sum() == 0:
        out.update({"handle_found": False, "fails": ["handle_found"], "pass": False}); return out
    # find the cable as a ray from the pulley (best thin dark line over 15..110 px), march along it to the handle -> E
    p0 = np.array(cfg["pulley_xy"], float); gb0 = cv2.GaussianBlur(g, (3, 3), 0).astype(float)
    def thin(q, nr_):
        xi, yi = int(q[0]), int(q[1])
        if not (6 <= xi < W - 6 and 6 <= yi < H - 6): return -99
        sa, sb = q + nr_ * 5, q - nr_ * 5
        return min(gb0[int(sa[1]), int(sa[0])], gb0[int(sb[1]), int(sb[0])]) - gb0[yi, xi]
    pr = cfg.get("pulley_search_px", 24); bestray = (-1, p0, 0.0)
    for dx in range(-pr, pr + 1, 4):
        for dy in range(-pr, pr + 1, 4):
            for th in np.radians(np.arange(-30, 31, 1.0)):
                u_ = np.array([np.cos(th), np.sin(th)]) * (1 if cfg.get("tower_side", "left") == "left" else -1); nr_ = np.array([-u_[1], u_[0]])
                c_ = np.mean([max(thin(p0 + [dx, dy] + u_ * t + nr_ * o, nr_) for o in (-1, 0, 1)) >= 6 for t in range(15, 111, 4)])
                if c_ > bestray[0]: bestray = (c_, p0 + [dx, dy], th)
    _, pr0, th = bestray; u_ = np.array([np.cos(th), np.sin(th)]) * (1 if cfg.get("tower_side", "left") == "left" else -1); nr_ = np.array([-u_[1], u_[0]])
    E = None
    for t in range(15, 700):
        q = pr0 + u_ * t
        if not (0 <= q[0] < W and 0 <= q[1] < H): break
        hit = [q + nr_ * o for o in range(-5, 6) if 0 <= int((q + nr_ * o)[1]) < H and 0 <= int((q + nr_ * o)[0]) < W and hm[int((q + nr_ * o)[1]), int((q + nr_ * o)[0])]]
        if hit: E = hit[len(hit) // 2]; break
    if E is None:  # no handle on the cable line: fall back to the handle pixel nearest the pulley
        ys, xs = np.nonzero(hm); dist = np.hypot(xs - p0[0], ys - p0[1]); E = np.array([xs[dist.argmin()], ys[dist.argmin()]], float); out["cable_ray_hits_handle"] = False
    else: out["cable_ray_hits_handle"] = True
    p0 = pr0
    n, lab, st, _ = cv2.connectedComponentsWithStats(hm)
    areas = st[1:, cv2.CC_STAT_AREA]; main = 1 + int(areas.argmax())
    out["unbroken"] = {"blobs": int(n - 1), "main_frac": round(float(areas.max() / areas.sum()), 3), "pass": bool(areas.max() / areas.sum() >= 0.9)}
    # cable: dark thin line pulley -> E (best pulley point within +-search), plus the last 15% (attachment)
    gb = cv2.GaussianBlur(g, (3, 3), 0).astype(float)
    def cov(a, b, t0, t1, ns=50):
        v = b - a; L = np.linalg.norm(v); u = v / (L + 1e-6); nr = np.array([-u[1], u[0]]); hits = 0
        for t in np.linspace(t0, t1, ns):
            c = a + v * t; best = -99
            for off in range(-3, 4):
                q = c + nr * off; xi, yi = int(q[0]), int(q[1])
                sa, sb = c + nr * (off - 5), c + nr * (off + 5)
                va = gb[int(np.clip(sa[1], 0, H - 1)), int(np.clip(sa[0], 0, W - 1))]; vb = gb[int(np.clip(sb[1], 0, H - 1)), int(np.clip(sb[0], 0, W - 1))]
                best = max(best, min(va, vb) - gb[int(np.clip(yi, 0, H - 1)), int(np.clip(xi, 0, W - 1))])
            hits += best >= 6
        return hits / ns, L
    cc, L = cov(p0, E, 0.06, 0.9); pb = p0
    out["cable"] = {"cover": round(cc, 2), "len_px": round(float(L), 1), "pulley_used": pb.round(0).tolist(), "ray_deg": round(float(np.degrees(th)), 1), "pass": bool(cc >= cfg["cable_min_cover"] and L >= 60 and out["cable_ray_hits_handle"])}
    # attached: every handle blob chains to the main blob (gap <= 14 px) and the cable reaches the nearest blob
    # (the hook / eyelet): the thin-line test is run on the 70-90 % stretch; E itself is the handle pixel nearest the pulley
    conn = cv2.morphologyEx(allm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)); _, clab = cv2.connectedComponents(conn)
    mlab = np.bincount(clab[lab == main]).argmax(); gaps = []
    for i in range(1, n):
        if i == main: continue
        same = np.bincount(clab[lab == i]).argmax() == mlab
        gaps.append(0.0 if same else 99.0)
    # cable -> handle: walk the cable line from 40 % to the eyelet in a +-3 px tube of the change mask; the longest unchanged run must be short
    Lc = float(np.hypot(*(E - p0)))
    def longest_gap(path):
        run = mx = 0
        for t in np.arange(0.4 * Lc, Lc + 1):
            q = path(t)
            on = any(0 <= int((q + nr_ * o)[1]) < H and 0 <= int((q + nr_ * o)[0]) < W and (conn[int((q + nr_ * o)[1]), int((q + nr_ * o)[0])] or thin(q + nr_ * o, nr_) >= 6) for o in range(-3, 4))
            run = 0 if on else run + 1; mx = max(mx, run)
        return mx
    # two candidate cable paths (fitted ray, straight pulley->eyelet); the better one counts
    mx = min(longest_gap(lambda t: p0 + (E - p0) * (t / max(Lc, 1))), longest_gap(lambda t: p0 + u_ * t) if out.get("cable_ray_hits_handle") else 999)
    gmax = cfg.get("attach_gap_px_max", 6)
    out["attached"] = {"max_gap_on_cable_px": int(mx), "blob_gaps_px": gaps,
                       "pass": bool(mx <= gmax and all(gp <= 14 for gp in gaps) and out.get("cable_ray_hits_handle", False))}
    out["unbroken"]["pass"] = bool(out["unbroken"]["main_frac"] >= 0.9 or all(gp <= 14 for gp in gaps))
    # shape: rectangle frame (hull fills its min-area rect, 4 corners, no taper) vs V / double-D (tapers to the eyelet)
    allpts = np.argwhere(hm > 0)[:, ::-1].astype(float)
    cnts, _ = cv2.findContours(hm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer = np.vstack(cnts); hull = cv2.convexHull(outer)
    rect = cv2.minAreaRect(outer); ra = rect[1][0] * rect[1][1] + 1e-6
    fill = float(cv2.contourArea(hull) / ra); approx = cv2.approxPolyDP(hull, 0.06 * cv2.arcLength(hull, True), True)
    u = (E - pb) / (np.linalg.norm(E - pb) + 1e-6); nr = np.array([-u[1], u[0]])
    along, perp = (allpts - E) @ u, (allpts - E) @ nr; Lh = along.max() - along.min() + 1e-6
    def ext(lo, hi):
        sel = (along >= along.min() + lo * Lh) & (along <= along.min() + hi * Lh)
        return float(np.ptp(perp[sel])) if sel.sum() > 3 else 0.0
    near, far = ext(0.0, 0.3), ext(0.6, 1.0); taper = near / (far + 1e-6)
    rect_frame = fill >= cfg.get("rect_fill_min", 0.88) and len(approx) == 4 and taper >= 0.75
    out["shape"] = {"hull_vs_minrect": round(fill, 3), "hull_corners": int(len(approx)), "taper_near_over_far": round(taper, 2),
                    "rect_frame_like": bool(rect_frame), "pass": bool((not rect_frame) and taper <= cfg.get("taper_max", 0.6))}
    outer = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(outer)
    out["handle_box"] = [int(x), int(y), int(w), int(h)]; out["eyelet_xy"] = E.round(0).tolist()
    keys = ["machine", "cable", "attached", "unbroken", "shape"]
    out["fails"] = [k for k in keys if not out[k]["pass"]]; out["pass"] = not out["fails"]
    return out

def _py(o):
    if isinstance(o, dict): return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_py(v) for v in o]
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.generic): return o.item()
    return o

if __name__ == "__main__" and "--plate-handle" in sys.argv:
    a_ = sys.argv; cfg_ = json.load(open(a_[1])); r_ = _py(check_plate_handle(cfg_, a_[a_.index("--plate-handle") + 1], person="--person" in a_))
    if "--out" in a_: json.dump(r_, open(a_[a_.index("--out") + 1], "w"), indent=1)
    print(json.dumps(r_))
elif __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--start"); ap.add_argument("--peak"); ap.add_argument("--out")
    a = ap.parse_args(); cfg = json.load(open(a.config)); res = {}
    if a.start: res["start"] = check(cfg, a.start, "start")
    if a.peak: res["peak"] = check(cfg, a.peak, "peak")
    if a.start and a.peak and "pose" in res["start"] and "pose" in res["peak"]:
        dlt = abs(res["start"]["pose"]["torso_lean_deg"] - res["peak"]["pose"]["torso_lean_deg"])
        res["pair"] = {"torso_delta_deg": round(dlt, 1), "pass": dlt <= cfg["torso_delta_max"]}
    def _py(o):
        if isinstance(o, dict): return {k: _py(v) for k, v in o.items()}
        if isinstance(o, list): return [_py(v) for v in o]
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.generic): return o.item()
        return o
    res = _py(res)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items(): print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("kp",)}))

