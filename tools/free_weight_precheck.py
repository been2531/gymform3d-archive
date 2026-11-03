#!/usr/bin/env python3
"""Reusable start-still pre-check for dumbbell / bodyweight days (t696u). No machine plate needed.
Config = configs/precheck/free_weight_base.json (shared thresholds) + one small per-exercise JSON with "extends".
Checks per still (each PASS/FAIL with its number):
  size      mannequin height / frame height >= min, and the body + equipment do not touch the frame border
  torso     forward lean from vertical (shoulder-mid vs hip-mid) inside the form's [min, max]
  arm       elbow flexion <= max (start = arms extended) or inside [min, max]
  dumbbell  colour-keyed heads (palette A cobalt): expected count, each with >= 2 blue parts (both ends), convex closed outline,
            end-face height ratio (perspective-safe), one next to a visible hand, the pair at the same height
  head      head vs body luminance |dL|
  light     shaded side (low-pass p10 of the upper body) minus backdrop median >= min
  sharp     Laplacian variance on the eroded body >= frac x the Day 4 reference frame (no blur)
  glow      the form's colour (wrong = red, right = green) covers >= min of the body, the other colour <= max,
            and its centroid lies in the target band (fraction of shoulder->hip)
usage: python3 tools/free_weight_precheck.py CONFIG --still IMG --form wrong|right [--out J]
       python3 tools/free_weight_precheck.py CONFIG --calib IMG     (prints sharpness/light of a reference frame)"""
import sys, json, argparse, cv2, numpy as np
sys.path.insert(0, "/workspace/daily_info_reels/tools"); sys.path.insert(0, "/workspace/daily_info_reels")
from still_precheck import pose, ang, _py
R = "/workspace/daily_info_reels/"
def load(p):
    c = json.load(open(p))
    if "extends" in c: b = load(R + c.pop("extends")); b.update({k: v for k, v in c.items() if not isinstance(v, dict) or k not in b}); [b[k].update(v) for k, v in c.items() if isinstance(v, dict) and k in b and isinstance(b[k], dict)]; return b
    return c
def prep(cfg, path):
    img = cv2.imread(path); W, H = cfg["size"]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if img.shape[:2] != (H, W) else img
def body_stats(img, kp, seg):
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float) * 100 / 255
    sh = (kp[11, :2] + kp[12, :2]) / 2; hp = (kp[23, :2] + kp[24, :2]) / 2; tl = float(np.linalg.norm(sh - hp))
    head = np.zeros_like(seg); hc = kp[[0, 7, 8], :2].mean(0); cv2.circle(head, (int(hc[0]), int(hc[1])), int(0.22 * tl), 1, -1); head &= seg
    upper = seg.copy(); upper[int(max(hp[1], sh[1])) - 10:, :] = 0; upper &= (1 - cv2.dilate(head, np.ones((9, 9), np.uint8)))
    bgm = ~cv2.dilate(seg, np.ones((61, 61), np.uint8)).astype(bool)
    er = cv2.erode(upper, np.ones((11, 11), np.uint8)).astype(bool)
    lp = cv2.GaussianBlur(lum * upper, (0, 0), 8) / np.maximum(cv2.GaussianBlur(upper.astype(float), (0, 0), 8), 1e-3)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); lv = cv2.Laplacian(g, cv2.CV_64F)
    bodyer = cv2.erode(seg, np.ones((9, 9), np.uint8)).astype(bool)
    return dict(lum=lum, sh=sh, hp=hp, tl=tl, head=head, upper=upper, hL=float(np.median(lum[head > 0])) if head.sum() else float("nan"),
                bL=float(np.median(lum[upper > 0])), bgL=float(np.median(lum[bgm])), p10=float(np.percentile(lp[er], 10)),
                sharp=float(lv[bodyer].var()))
def check(cfg, path, form):
    img = prep(cfg, path); H, W = img.shape[:2]; out = {"file": path, "form": form}
    kp, seg = pose(img)
    if kp is None: return {**out, "pass": False, "fails": ["pose"], "error": "no pose"}
    st = body_stats(img, kp, seg); F = cfg["forms"][form]; tl = st["tl"]
    # size + in frame
    ys, xs = np.nonzero(seg); hfrac = (ys.max() - ys.min()) / H
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); lo, hi = cfg["dumbbell_hsv"]
    blue = cv2.inRange(hsv, np.array(lo), np.array(hi)); blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    obj = (seg > 0) | (blue > 0); m = cfg["border_px"]
    touch = bool(obj[:m].any() or obj[-m:].any() or obj[:, :m].any() or obj[:, -m:].any())
    out["size"] = {"height_frac": round(float(hfrac), 3), "touches_border": touch, "pass": bool(hfrac >= cfg["height_frac_min"] and not touch)}
    # torso lean from vertical (absolute, either facing)
    sh, hp = st["sh"], st["hp"]; lean = abs(float(np.degrees(np.arctan2(sh[0] - hp[0], hp[1] - sh[1]))))
    out["torso"] = {"lean_deg": round(lean, 1), "range": F["torso_lean"], "pass": bool(F["torso_lean"][0] <= lean <= F["torso_lean"][1])}
    flexes = []
    for sd in (11, 12):
        if min(kp[sd, 2], kp[sd + 2, 2], kp[sd + 4, 2]) > 0.3: flexes.append(180 - ang(kp[sd, :2], kp[sd + 2, :2], kp[sd + 4, :2]))
    flex = max(flexes) if flexes else 999.0; ar = F["elbow_flex"]
    out["arm"] = {"elbow_flex_deg": round(flex, 1), "range": ar, "sides_seen": len(flexes), "pass": bool(ar[0] <= flex <= ar[1])}
    # dumbbells: cluster blue blobs, one cluster per dumbbell (merge blobs whose dilated masks touch)
    n, lab, sts, cen = cv2.connectedComponentsWithStats(blue)
    heads = [i for i in range(1, n) if sts[i, 4] >= cfg["dumbbell_head_min_px"]]
    hm = np.isin(lab, heads).astype(np.uint8); cl = cv2.dilate(hm, np.ones((cfg["dumbbell_merge_px"],) * 2, np.uint8))
    cn, clab, cst, ccen = cv2.connectedComponentsWithStats(cl)
    wrists = [kp[i, :2] for i in (15, 16) if kp[i, 2] > 0.2] or [kp[i, :2] for i in (15, 16)]
    vis_w = [kp[i, :2] for i in (15, 16) if kp[i, 2] > 0.5]
    dbs = []
    for j in range(1, cn):
        mem = [i for i in heads if clab[int(cen[i][1]), int(cen[i][0])] == j]
        # end faces are split by the gold ring -> judge the closed, hole-filled cluster, not single blobs
        cm = cv2.morphologyEx((np.isin(lab, mem)).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cs, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); c = max(cs, key=cv2.contourArea)
        solid = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
        dw = min(float(np.linalg.norm(ccen[j] - w)) for w in (vis_w or wrists)) / tl
        dbs.append({"area": int(sum(sts[i, 4] for i in mem)), "blue_parts": len(mem), "solidity": round(solid, 3), "near_visible_hand": round(dw, 2),
                    "bbox": cst[j, :4].tolist(), "cy": float(ccen[j][1])})
    dbs.sort(key=lambda d: -d["area"]); exp = cfg["dumbbell_expected"] if "dumbbell_expected" in cfg else cfg["dumbbells_expected"]; main = dbs[:exp]
    # perspective-safe size test: end-face HEIGHT ratio (width shrinks with yaw/occlusion; height does not)
    hr = (min(d["bbox"][3] for d in main) / max(d["bbox"][3] for d in main)) if len(main) == exp and exp > 1 else 1.0
    pair_dy = (abs(main[0]["cy"] - main[1]["cy"]) / tl) if len(main) == 2 else 0.0
    okdb = (len(main) == exp and hr >= cfg["dumbbell_height_ratio_min"] and all(d["solidity"] >= cfg["dumbbell_solidity_min"] for d in main)
            and all(d["blue_parts"] >= 2 for d in main) and min(d["near_visible_hand"] for d in main) <= cfg["dumbbell_near_hand_max"]
            and pair_dy <= cfg["dumbbell_pair_dy_max"])
    for d in main: d.pop("cy")
    out["dumbbell"] = {"found": len(dbs), "expected": exp, "height_ratio": round(hr, 2), "pair_dy_torso": round(pair_dy, 2), "items": main, "pass": bool(okdb)}
    out["head"] = {"dL": round(abs(st["hL"] - st["bL"]), 1), "pass": bool(abs(st["hL"] - st["bL"]) <= cfg["head_body_dL_max"])}
    out["light"] = {"diff": round(st["p10"] - st["bgL"], 1), "pass": bool(st["p10"] - st["bgL"] >= cfg["shaded_vs_backdrop_min"])}
    sref = cfg["sharp_ref_var"]; out["sharp"] = {"lap_var": round(st["sharp"], 1), "ref": sref, "frac": round(st["sharp"] / sref, 2), "pass": bool(st["sharp"] >= cfg["sharp_min_frac"] * sref)}
    b, gr, r = [img[..., i].astype(int) for i in range(3)]; sb = seg.astype(bool)
    green = (gr - np.maximum(r, b) > 25) & sb; red = (r - np.maximum(gr, b) > 35) & sb
    want, other = (red, green) if F["glow"] == "red" else (green, red)
    wf, of = float(want.sum() / sb.sum()), float(other.sum() / sb.sum())
    if want.sum(): cy = float(np.nonzero(want)[0].mean()); band = (cy - sh[1]) / max(hp[1] - sh[1], 1)
    else: band = float("nan")
    tb = F["glow_band"]
    out["glow"] = {"colour": F["glow"], "frac": round(wf, 4), "other_frac": round(of, 4), "centroid_band": round(band, 2), "band": tb,
                   "pass": bool(wf >= cfg["glow_frac_min"] and of <= cfg["glow_other_max"] and tb[0] <= band <= tb[1])}
    keys = ["size", "torso", "arm", "dumbbell", "head", "light", "sharp", "glow"]
    out["fails"] = [k for k in keys if not out[k]["pass"]]; out["pass"] = not out["fails"]
    out["kp"] = {"sh_mid": sh.round(1).tolist(), "hip_mid": hp.round(1).tolist()}
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--still"); ap.add_argument("--form"); ap.add_argument("--calib"); ap.add_argument("--out")
    a = ap.parse_args(); cfg = load(a.config)
    if a.calib:
        img = prep(cfg, a.calib); kp, seg = pose(img); st = body_stats(img, kp, seg)
        print(json.dumps({"sharp_lap_var": round(st["sharp"], 1), "light_diff": round(st["p10"] - st["bgL"], 1), "head_dL": round(abs(st["hL"] - st["bL"]), 1)})); sys.exit()
    res = check(cfg, a.still, a.form); s = json.dumps(_py(res), indent=1); print(s)
    if a.out: open(a.out, "w").write(s)
