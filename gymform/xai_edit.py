"""Grok Imagine video edit (xAI API) for pair renders: same motion, different anatomy rendering.

Day 6+ pair rule: steps 1/3 share one motion and steps 2/4 share one motion. When a motion's base clip
has its anatomy baked in (e.g. muscle), the other rendering (e.g. bone X-Ray) is made by a *video edit*
of the base, which keeps the duration, framing, camera and motion and changes only the anatomy overlay.

API (docs.x.ai > Video Editing, checked 2026-10-30):
  POST https://api.x.ai/v1/videos/edits  {model: grok-imagine-video, prompt, video: {url | data URI | file_id}}
  -> {request_id};  poll GET /v1/videos/{request_id} until status == done (video.url) | failed | expired.
  Source mp4 <= 8.7 s, output keeps duration/aspect/resolution (<= 720p). ~$0.07 per output second at 720p.

Safety:
  * Key only from env XAI_API_KEY (or /home/box/.kairo/xai_api_key, chmod 600). Never printed/logged/committed.
  * ONE TAKE per target: every request is recorded in _cache_gymform/xai_edits.jsonl; a target that already
    has a ledger entry is refused unless --retake is passed explicitly (needs owner approval: costs money).
  * Existing target files are never overwritten.

Usage:
  python3 -m gymform.xai_edit --day 6 [--dry-run] [--only wrong|right] [--retake]
Config (motions.<m>.renders.bone): {"file": ..., "edit_of": <base>, "edit": {"color": "crimson"}}
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "output", "assets")
LEDGER = os.path.join(ROOT, "_cache_gymform", "xai_edits.jsonl")
API = "https://api.x.ai/v1"
MODEL = "grok-imagine-video"
USD_PER_S_720P = 0.07
MAX_SRC_S = 8.7

# t529u root cause: the first prompt said "show a glowing X-ray skeleton", which the model read as a full-body X-ray:
# it darkened/thinned the limbs into see-through glass, added leg bones and drew cyan (not emerald). The approved
# bone clips (Day 1-6 i2v) are an OPAQUE silver mannequin with thin glowing UPPER-BODY bones only. Say exactly that.
# t530u: "뼈 색상이 너무 진하면 뼈가 안보이자나" -> bright luminous near-white bones (Day 1 v36), only a subtle tint
COLOR_WORDS = {
    "crimson": "bright luminous near-white bones with only a subtle soft red glow at the edges (not dark, not saturated red)",
    "emerald": "bright luminous near-white bones with only a subtle soft green glow at the edges (not dark, not saturated green, not cyan)",
}
PROMPTS = {
    "bone": ("Change only the anatomy overlay. Remove the coloured muscle fibres and instead draw thin glowing "
             "bones of the UPPER BODY ONLY (rib cage, clavicles, scapulae, upper-arm and forearm bones) as a luminous "
             "X-ray overlay on top of the skin: {cw}. The mannequin skin stays fully OPAQUE matte silver-grey exactly as in the "
             "original: same arm, shoulder, torso and leg thickness and shape, not transparent, not darkened, not glass. "
             "No bones and no glow on the hips, legs or feet; no glow outside the body. Keep everything else exactly as "
             "in the original: camera angle, framing, mannequin size, black shorts, dumbbells, bench, studio background, "
             "lighting, and the exact same motion and timing."),
    "muscle": ("Change only the anatomy overlay: remove the bone overlay and show glowing {cw} muscle fibres on the "
               "target muscles. The mannequin skin stays fully opaque matte silver with the same limb thickness and "
               "shape. Keep everything else exactly as in the original: camera angle, framing, mannequin, black shorts, "
               "equipment, studio background, lighting, and the exact same motion and timing."),
}
# ACCEPTANCE QC (before an edit may enter output/assets), measured against the base clip on 9 frames.
# Calibrated on Day 6: base vs itself = 1.00; t522 wrong edit = sil IoU 0.82 + leg bones (reject); t522 right
# edit = IoU 0.75, area 0.75, hue 86 (cyan) (reject).
ACCEPT = {"sil_iou_min": 0.90, "area_ratio": (0.93, 1.07), "flow_p90_max_px": 4.0, "leg_glow_max": 0.01,
          # bone colour (t530u): soft tint hue range + bright luminous bones. Calibrated on the approved-look native
          # i2v bone clips day6_idp_step1_bone (V50 219, S50 0.40, core 0.17, dark 0.075) / step3_bone (V50 199,
          # S50 0.33, core 0.22, dark 0.003); t522 right edit = V50 249 but S50 0.67, core 0.00, hue 86 (reject)
          "hue": {"emerald": (40, 80), "crimson": (168, 12)},
          "bone_V50_min": 190, "bone_S50_max": 0.50, "white_core_min": 0.10, "dark_fill_max": 0.12}


def bone_colour(bgr):
    """Raw-clip bone pixels: tinted glow (S>64, V>100) + their near-white cores (L>185, S<=64, within 3px)."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, S, V = (hsv[..., i].astype(np.float32) for i in range(3))
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tint = (S > 64) & (V > 100)
    core = (cv2.dilate(tint.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & (L > 185) & (S <= 64)
    b = tint | core
    n = int(b.sum())
    if n < 300 or not tint.any():
        return None
    return {"V50": float(np.median(V[b])), "S50": float(np.median(S[b]) / 255.0), "hue": float(np.median(Hh[tint])),
            "white_core": float(core.sum()) / n, "dark_fill": float(((S > 150) & (V < 200) & b).sum()) / n}
DEFAULT_COLOR = {"wrong": "crimson", "right": "emerald"}


def _key():
    k = os.environ.get("XAI_API_KEY", "").strip()
    if not k:
        p = "/home/box/.kairo/xai_api_key"
        if os.path.exists(p):
            k = open(p).read().strip()
    if not k:
        sys.exit("xai_edit: XAI_API_KEY not available")
    return k


def _probe_s(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def _ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def _log(rec):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def jobs_for(cfg, only=None):
    """(motion, kind, src, dst, prompt) for every render that is an edit of a base and does not exist yet."""
    out = []
    for m, mo in (cfg.get("motions") or {}).items():
        if only and m != only:
            continue
        for kind, r in (mo.get("renders") or {}).items():
            if not r.get("edit_of") or r.get("file") == r.get("edit_of"):
                continue
            color = (r.get("edit") or {}).get("color") or DEFAULT_COLOR.get(m, "crimson")
            prompt = (r.get("edit") or {}).get("prompt") or PROMPTS[kind].format(cw=COLOR_WORDS.get(color, color))
            out.append((m, kind, os.path.join(ASSETS, r["edit_of"]), os.path.join(ASSETS, r["file"]), prompt, color))
    return out


def _frames(path, idx):
    import cv2
    cap, out, i = cv2.VideoCapture(path), {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idx:
            out[i] = fr
        i += 1
    cap.release()
    return out, i


def acceptance(src, dst, color):
    """Edit acceptance QC vs its base clip (no spend): silhouette (limb thickness/opacity), motion, leg glow, hue."""
    import cv2
    import numpy as np
    B, nb = _frames(src, set(range(0, 10 ** 4, 15)))
    E, ne = _frames(dst, set(B))
    if nb != ne:
        return False, {"why": f"frame count {ne} != {nb}"}
    k9 = np.ones((9, 9), np.uint8)

    def sil(x):   # opaque silver/bone body (backdrop ~35, glass ~40-60): closed, holes filled
        m = cv2.morphologyEx((cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) > 70).astype(np.uint8), cv2.MORPH_CLOSE, k9)
        ff = np.pad(m, 1).copy()
        cv2.floodFill(ff, None, (0, 0), 2)
        return ff[1:-1, 1:-1] != 2

    ious, ratios, flows, legs, hues = [], [], [], [], []
    for i in sorted(B):
        b, e = B[i], E[i]
        mb, me = sil(b), sil(e)
        ious.append(float((mb & me).sum()) / max(1, (mb | me).sum()))
        ratios.append(float(me.sum()) / max(1, mb.sum()))
        gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (360, 640))
        ge = cv2.resize(cv2.cvtColor(e, cv2.COLOR_BGR2GRAY), (360, 640))
        fl = cv2.calcOpticalFlowFarneback(gb, ge, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        flows.append(float(np.percentile(np.hypot(fl[..., 0], fl[..., 1])[gb > 40], 90)) * 2.0)
        hb, he = cv2.cvtColor(b, cv2.COLOR_BGR2HSV), cv2.cvtColor(e, cv2.COLOR_BGR2HSV)
        glow = (he[..., 1] > 110) & (he[..., 2] > 120) & ~((hb[..., 1] > 110) & (hb[..., 2] > 120))
        ys = np.where(mb.any(axis=1))[0]
        if len(ys):   # lower 40% of the body = hips/legs/feet: must stay plain silver
            y0 = int(ys[0] + 0.6 * (ys[-1] - ys[0]))
            legs.append(float(glow[y0:].sum()) / max(1, mb[y0:].sum()))
        bc = bone_colour(e)
        if bc:
            hues.append(bc)
    pick = lambda k: float(np.median([h[k] for h in hues])) if hues else -1.0
    hue = pick("hue")
    lo, hi = ACCEPT["hue"].get(color, (0, 180))
    hue_ok = (lo <= hue <= hi) if lo <= hi else (hue >= lo or 0 <= hue <= hi)
    m = {"sil_iou_min": round(min(ious), 3), "area_ratio_min": round(min(ratios), 3), "area_ratio_max": round(max(ratios), 3),
         "flow_p90_px": round(max(flows), 1), "leg_glow": round(max(legs or [0]), 3), "hue": round(hue, 1), "color": color}
    fails = []
    if m["sil_iou_min"] < ACCEPT["sil_iou_min"]:
        fails.append(f"limb/body silhouette drift (IoU {m['sil_iou_min']} < {ACCEPT['sil_iou_min']}: thinner/see-through limbs)")
    if not (ACCEPT["area_ratio"][0] <= m["area_ratio_min"] and m["area_ratio_max"] <= ACCEPT["area_ratio"][1]):
        fails.append(f"body area ratio {m['area_ratio_min']}..{m['area_ratio_max']} outside {ACCEPT['area_ratio']}")
    if m["flow_p90_px"] > ACCEPT["flow_p90_max_px"]:
        fails.append(f"motion drift p90 {m['flow_p90_px']}px")
    if m["leg_glow"] > ACCEPT["leg_glow_max"]:
        fails.append(f"glow on hips/legs ({m['leg_glow']})")
    m.update({k: round(pick(k), 3) for k in ("V50", "S50", "white_core", "dark_fill")})
    if not hue_ok:
        fails.append(f"bone tint hue {hue} not {color} {ACCEPT['hue'].get(color)}")
    if m["V50"] < ACCEPT["bone_V50_min"] or m["white_core"] < ACCEPT["white_core_min"]:
        fails.append(f"bones not bright/near-white (V50 {m['V50']}, white core {m['white_core']})")
    if m["S50"] > ACCEPT["bone_S50_max"] or m["dark_fill"] > ACCEPT["dark_fill_max"]:
        fails.append(f"bone tint too dark/saturated (S50 {m['S50']}, dark fill {m['dark_fill']})")
    m["fails"] = fails
    return not fails, m


def edit_one(src, dst, prompt, key, color="", poll_s=6, timeout_s=900):
    dur = _probe_s(src)
    if dur > MAX_SRC_S:
        raise RuntimeError(f"source {dur:.2f}s > {MAX_SRC_S}s")
    data = "data:video/mp4;base64," + base64.b64encode(open(src, "rb").read()).decode()
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    t0 = time.time()
    r = requests.post(f"{API}/videos/edits", headers=h, timeout=120,
                      json={"model": MODEL, "prompt": prompt, "video": {"url": data}})
    if r.status_code >= 300:
        raise RuntimeError(f"edit submit HTTP {r.status_code}: {r.text[:300]}")
    rid = r.json()["request_id"]
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "src": os.path.basename(src), "dst": os.path.basename(dst),
           "request_id": rid, "model": MODEL, "prompt": prompt, "src_s": round(dur, 3),
           "est_usd": round(dur * USD_PER_S_720P, 3), "status": "submitted"}
    _log(rec)
    print(f"[xai_edit] submitted {rec['dst']} request_id={rid}", flush=True)
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        g = requests.get(f"{API}/videos/{rid}", headers={"Authorization": f"Bearer {key}"}, timeout=60)
        if g.status_code >= 300:
            continue
        j = g.json()
        st = j.get("status")
        if st == "done":
            url = (j.get("video") or {}).get("url")
            tmp = dst + ".part"
            with requests.get(url, stream=True, timeout=300) as d:
                d.raise_for_status()
                with open(tmp, "wb") as f:
                    for c in d.iter_content(1 << 20):
                        f.write(c)
            ok, acc = acceptance(src, tmp, color)
            final = dst if ok else dst[:-4] + "_REJECTED.mp4"
            os.replace(tmp, final)
            rec.update(status="done" if ok else "rejected", secs=round(time.time() - t0, 1), out_s=round(_probe_s(final), 3),
                       usage=j.get("usage"), acceptance=acc, file=os.path.basename(final))
            _log(rec)
            print(f"[xai_edit] {rec['status']} {os.path.basename(final)} in {rec['secs']}s {acc}", flush=True)
            if not ok:
                raise RuntimeError("edit rejected by acceptance QC (kept as _REJECTED.mp4; owner decides on a retake)")
            return rec
        if st in ("failed", "expired"):
            rec.update(status=st, secs=round(time.time() - t0, 1), detail=str(j)[:300])
            _log(rec)
            raise RuntimeError(f"edit {st}: {rec['detail']}")
    rec.update(status="timeout")
    _log(rec)
    raise RuntimeError("edit poll timeout (request_id in ledger; fetch later, do not resubmit)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--only", choices=["wrong", "right"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="acceptance QC of existing edits vs their base (no spend)")
    ap.add_argument("--retake", action="store_true", help="allow a 2nd paid take (owner approval required)")
    a = ap.parse_args(argv)
    cfg = json.load(open(os.path.join(ROOT, "configs", f"day{a.day:02d}.json")))
    jobs = jobs_for(cfg, a.only)
    done = {r["dst"] for r in _ledger()}
    todo = []
    if a.check:   # acceptance QC of existing edits only (no API call, no spend)
        bad = 0
        for m, kind, src, dst, prompt, color in jobs:
            if os.path.exists(dst):
                ok, acc = acceptance(src, dst, color)
                bad += not ok
                print(f"[xai_edit] {'ACCEPT' if ok else 'REJECT'} {os.path.basename(dst)}: {json.dumps(acc)}")
        return 1 if bad else 0
    for m, kind, src, dst, prompt, color in jobs:
        b = os.path.basename(dst)
        if os.path.exists(dst):
            print(f"[xai_edit] exists, skip: {b}")
        elif b in done and not a.retake:
            print(f"[xai_edit] REFUSED (one take; already requested, see ledger): {b}")
        elif not os.path.exists(src):
            print(f"[xai_edit] missing source: {src}")
        else:
            todo.append((m, kind, src, dst, prompt, color))
    est = sum(_probe_s(s) * USD_PER_S_720P for _, _, s, _, _, _ in todo)
    print(f"[xai_edit] {len(todo)} edit(s), est ${est:.2f}")
    if a.dry_run or not todo:
        for t in todo:
            print("  ", t[0], t[1], os.path.basename(t[2]), "->", os.path.basename(t[3]))
            print("   prompt:", t[4])
        return 0
    key = _key()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(edit_one, s, d, p, key, c) for _, _, s, d, p, c in todo]
        errs = 0
        for f in futs:
            try:
                f.result()
            except Exception as e:  # noqa
                errs += 1
                print(f"[xai_edit] ERROR {e}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
