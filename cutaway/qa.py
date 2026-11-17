"""Automatic QA gates for cutaway reels.  Pass criteria are fixed here, *before*
rendering, and written into every report.  A gate either passes or fails; the
make_reel retry loop may change render params at most twice, then stops."""
from __future__ import annotations

import json
import re
import subprocess

import cv2
import numpy as np

import cutaway2 as C

CRITERIA = {
    "motion_first_0_5s": "mean |frame(0) - frame(0.5s)| on 270x480 gray >= 3.0 (0-255)",
    "no_static_hold": "no run of consecutive frame diffs < 0.20 (135x240 blurred gray) longer than 1.0 s",
    "registration": "silhouette-edge chamfer median (layer vs skin) x max full-frame zoom <= 40 px",
    "label_words": "every label / stat caption <= 3 words",
    "bottom_clear": "overlay alpha in bottom 15 % (y >= 1632) < 0.02 on every sampled frame",
    "duration": "12.0 s <= duration <= 15.0 s",
    "window_holes": "<= 10 % of window pixels darker than L=0.04 (window open, before bloom)",
    "loudness": "-16 <= integrated LUFS <= -12 and true peak <= -1.0 dBTP",
    "text_size": "label/caption cap height >= 70 px, hook cap >= 110 px, every line fits 1080 width",
    "text_contrast": "WCAG ratio (white glyphs vs 90th-pct local background ring) >= 4.5 for every visible text",
    "text_clear_of_window": "no text box overlaps the open window circle (leader lines allowed)",
    "facts_sourced": "every number in on-screen text and caption maps to a topic fact with source + url",
    "view_in_content": "camera never shows padding/letterbox outside the real image (overscan <= 2 px)",
    "caption": "exactly 3 lines, 0 hashtags, English (ASCII + x/°)",
}


def rel_lum(rgb):
    c = np.asarray(rgb, np.float32)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def sample_times(reel, step=5):
    n = int(round(reel.dur * reel.fps))
    return [k / reel.fps for k in range(0, n, step)]


def prerender(reel: C.Reel, topic: dict, reg: dict) -> dict:
    """Gates that can be judged from the renderer itself (no encode needed)."""
    res = {}
    # registration: worst layer, scaled to output px at the largest full-frame zoom
    zmax = reel.cover * max(s.get("zoom", 1.0) for s in topic["shots"] if s.get("kind") == "bloom") \
        if any(s.get("kind") == "bloom" for s in topic["shots"]) else reel.cover
    worst = max(v["chamfer_median_px"] for v in reg["layers"].values())
    reg_px = worst * zmax * (1 + reel.p["push"])
    res["registration"] = {"pass": reg_px <= 40.0 or not reel.p["bloom"],
                           "value": round(reg_px, 1),
                           "detail": {k: {"median_px_out": round(v["chamfer_median_px"] * zmax, 1),
                                          "p90_px_out": round(v["chamfer_p90_px"] * zmax, 1)}
                                      for k, v in reg["layers"].items()},
                           "note": "bloom disabled -> no full-frame reveal" if not reel.p["bloom"] else ""}
    labels = [s["label"] for s in topic["shots"] if s.get("label")]
    if topic.get("counter"):
        labels.append(topic["counter"]["caption"])
    bad = [l for l in labels if len(l.split()) > 3]
    res["label_words"] = {"pass": not bad, "value": max(len(l.split()) for l in labels), "bad": bad}
    # text sizes
    caps = {"hook": reel.hook.cap}
    fits = reel.hook.fits
    for i, sp in reel.labels.items():
        caps[f"label:{sp.lines[0]}"] = sp.cap
        fits = fits and sp.fits
    if topic.get("counter"):
        caps["caption:" + reel.cap_sprite.lines[0]] = reel.cap_sprite.cap
    sz_ok = fits and all((v >= 110 if k == "hook" else v >= 70) for k, v in caps.items())
    res["text_size"] = {"pass": bool(sz_ok), "value": {k: round(v, 1) for k, v in caps.items()}, "fits": fits}
    # sampled frames: bottom clear, contrast, holes, text vs window
    bottom = 0.0
    overscan = 0.0
    worst_c = (99.0, "")
    holes = 0.0
    overlap = []
    y_safe = C.SAFE_BOTTOM
    bloom_t = next((s["t"] for s in topic["shots"] if s.get("kind") == "bloom"), 1e9)
    for t in sample_times(reel, 5):
        fr = reel.frame(t, want_info=True)
        info = reel.last_info
        bottom = max(bottom, float(info["overlay_alpha"][y_safe:].max()))
        overscan = max(overscan, info["view_overscan_px"])
        jx, jy, r = info["win"]
        if 60 < r < 1000 and t < bloom_t:
            x0, y0, x1, y1 = int(max(0, jx - r)), int(max(0, jy - r)), int(min(C.W, jx + r)), int(min(C.H, jy + r))
            ys, xs = np.mgrid[y0:y1, x0:x1]
            m = ((xs - jx) ** 2 + (ys - jy) ** 2) < (r - 8) ** 2
            V = fr[y0:y1, x0:x1].max(axis=2)[m]          # sRGB value: "black hole" = every channel < 0.04
            holes = max(holes, float((V < 0.04).mean()))
        for tx in info["texts"]:
            if tx["bbox"] is None or tx["alpha"] < 0.95:
                continue
            bx0, by0, bx1, by1 = tx["bbox"]
            fill = tx["fill"] > 0.6
            if fill.sum() < 50:
                continue
            k5 = np.ones((5, 5), np.uint8)
            ring = cv2.dilate(fill.astype(np.uint8), np.ones((15, 15), np.uint8)) & ~cv2.dilate(fill.astype(np.uint8), k5).astype(bool)
            patch = rel_lum(fr[by0:by1, bx0:bx1])
            core = cv2.erode(fill.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            lt = float(np.median(patch[core])) if core.sum() > 20 else float(np.median(patch[fill]))
            lb = float(np.percentile(patch[ring > 0], 90)) if (ring > 0).sum() > 20 else 0.0
            cr = (max(lt, lb) + 0.05) / (min(lt, lb) + 0.05)
            if cr < worst_c[0]:
                worst_c = (cr, f'{tx["text"]} @ {t:.2f}s')
            if 60 < r < 1000 and t < bloom_t:
                # nearest point of text box to window centre
                nx, ny = min(max(jx, bx0), bx1), min(max(jy, by0), by1)
                if (nx - jx) ** 2 + (ny - jy) ** 2 < (r + 6) ** 2:
                    overlap.append(f'{tx["text"]} @ {t:.2f}s')
    res["view_in_content"] = {"pass": overscan <= 2.0, "value": round(overscan, 1)}
    res["bottom_clear"] = {"pass": bottom < 0.02, "value": round(bottom, 3)}
    res["text_contrast"] = {"pass": worst_c[0] >= 4.5, "value": round(worst_c[0], 2), "worst": worst_c[1]}
    res["window_holes"] = {"pass": holes <= 0.10, "value": round(holes, 3)}
    res["text_clear_of_window"] = {"pass": not overlap, "value": len(overlap), "where": overlap[:5]}
    # facts
    facts = topic.get("facts", [])
    fvals = {f"{float(f['value']):g}" for f in facts if f.get("source") and f.get("url")}
    texts = list(topic["hook"]) + labels + list(topic["caption"])
    if topic.get("counter"):
        texts.append(topic["counter"]["fmt"].format(topic["counter"]["to"]))
    nums = set()
    for tx in texts:
        for m_ in re.findall(r"\d+(?:\.\d+)?", tx):
            nums.add(f"{float(m_):g}")
    missing = sorted(n for n in nums if n not in fvals)
    res["facts_sourced"] = {"pass": not missing, "numbers": sorted(nums), "missing_source": missing}
    cap = topic["caption"]
    cap_ok = len(cap) == 3 and all(l.strip() for l in cap) and not any("#" in l for l in cap) \
        and all(re.fullmatch(r"[\x20-\x7E×°’]+", l) for l in cap)
    res["caption"] = {"pass": bool(cap_ok), "lines": len(cap)}
    return res


def decode_gray(mp4, w, h, ffmpeg="/usr/bin/ffmpeg"):
    p = subprocess.run([ffmpeg, "-v", "error", "-i", mp4, "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w).astype(np.float32)


def postrender(mp4: str, fps: int = 30) -> dict:
    res = {}
    pr = subprocess.run(["/usr/bin/ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", mp4],
                        capture_output=True, text=True, check=True)
    d = float(json.loads(pr.stdout)["format"]["duration"])
    res["duration"] = {"pass": 12.0 <= d <= 15.05, "value": round(d, 3)}
    g = decode_gray(mp4, 270, 480)
    m0 = float(np.abs(g[0] - g[int(0.5 * fps)]).mean())
    res["motion_first_0_5s"] = {"pass": m0 >= 3.0, "value": round(m0, 2)}
    s = np.stack([cv2.GaussianBlur(cv2.resize(x, (135, 240), interpolation=cv2.INTER_AREA), (0, 0), 1.2) for x in g])
    diffs = np.abs(np.diff(s, axis=0)).mean(axis=(1, 2))
    run = best = 0
    for v in diffs:
        run = run + 1 if v < 0.20 else 0
        best = max(best, run)
    res["no_static_hold"] = {"pass": best / fps <= 1.0, "value_s": round(best / fps, 2),
                             "min_diff": round(float(diffs.min()), 3), "median_diff": round(float(np.median(diffs)), 3)}
    e = subprocess.run(["/usr/bin/ffmpeg", "-hide_banner", "-nostats", "-i", mp4, "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", e.stderr)[-1])
    tp = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", e.stderr)[-1])
    res["loudness"] = {"pass": -16 <= I <= -12 and tp <= -1.0, "lufs": I, "true_peak_dbtp": tp}
    return res
