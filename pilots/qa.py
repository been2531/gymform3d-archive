"""Automatic QA gates. No human review: a reel either passes all gates or the
daily runner skips it."""
from __future__ import annotations
import json, re, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .rig import bone_errors

H = 1920


def probe(mp4):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,r_frame_rate,codec_type",
                        "-of", "json", mp4], capture_output=True, text=True)
    return json.loads(p.stdout)


def loudness(mp4):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", mp4, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    i = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", p.stderr)[-1])
    tp = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", p.stderr)[-1])
    return i, tp


def run(style, ep, mp4, metas, small, bg, narr, body):
    gates = {}
    pr = probe(mp4)
    dur = float(pr["format"]["duration"])
    v = [s for s in pr["streams"] if s["codec_type"] == "video"][0]
    gates["format_1080x1920_30fps"] = (v["width"] == 1080 and v["height"] == 1920 and v["r_frame_rate"] == "30/1", f"{v['width']}x{v['height']}@{v['r_frame_rate']}")
    gates["length_10_15s"] = (10.0 <= dur <= 15.0, round(dur, 2))
    # motion
    diffs = [float((np.abs(small[i] - small[i - 1]).max(axis=2) > 6).mean()) for i in range(1, len(small))]
    gates["motion_frame0"] = (diffs[0] > 0.0005, round(diffs[0], 5))
    run_, longest = 0, 0
    for d in diffs:
        run_ = run_ + 1 if d < 0.0005 else 0
        longest = max(longest, run_)
    gates["no_still_over_1s"] = (longest < 30, f"longest still {longest / 30:.2f}s")
    # hook at frame 0
    hook0 = [x for x in metas[0]["text"] if x["role"] in ("hook", "title")]
    words = len(hook0[0]["text"].split()) if hook0 else 0
    gates["hook_frame0_2to4_words"] = (bool(hook0) and 2 <= words <= 4 and hook0[0]["top"] < 400, f"{hook0[0]['text'] if hook0 else None!r} ({words} words)")
    # text sizes, contrast, bottom 15%
    caps_h = [x["cap"] for m in metas for x in m["text"] if x["role"] in ("hook", "title")]
    caps_l = [x["cap"] for m in metas for x in m["text"] if x["role"] not in ("hook", "title")]
    gates["hook_cap_ge_110px"] = (bool(caps_h) and min(caps_h) >= 110, min(caps_h) if caps_h else None)
    gates["labels_cap_ge_70px"] = (not caps_l or min(caps_l) >= 70, min(caps_l) if caps_l else None)
    cons = [x["contrast"] for m in metas for x in m["text"]]
    gates["text_contrast_ge_4_5"] = (min(cons) >= 4.5, round(min(cons), 2))
    tb = max(x["bottom"] for m in metas for x in m["text"])
    y0 = int(H * 0.85) // 4
    bgs = bg[::4, ::4].astype(np.int16)[y0:]
    worst = max(float((np.abs(s[y0:] - bgs).max(axis=2) > 10).mean()) for s in small)
    gates["bottom_15pct_clear"] = (tb < H * 0.85 and worst < 0.001, f"lowest text {tb:.0f}px, max dirty {worst:.4%}")
    # title lines + text never over a figure (head/limbs)
    tl = [x.get("lines", 1) for m in metas for x in m["text"] if x["role"] in ("hook", "title")]
    gates["title_max_2_lines"] = (max(tl) <= 2, f"max {max(tl)} lines")
    hits = 0; first = None
    for m in metas:
        for x in m["text"]:
            if "left" not in x:
                continue
            for (fx0, fy0, fx1, fy1) in m.get("figs", []):
                if x["left"] < fx1 and x["right"] > fx0 and x["top"] < fy1 and x["bottom"] > fy0:
                    hits += 1
                    first = first or f"t={m['t']:.2f}s {x['text']!r}"
    outside = [x["text"] for m in metas for x in m["text"] if "left" in x and (x["left"] < 20 or x["right"] > 1060)]
    gates["text_inside_frame"] = (not outside, f"{len(outside)} frames" + (f", e.g. {outside[0]!r}" if outside else ""))
    gates["text_clear_of_figure"] = (hits == 0, f"{hits} overlapping text/figure frames" + (f", first {first}" if first else ""))
    # limb lengths
    errs = [bone_errors(body, J) for m in metas for J in m["joints"]]
    gates["limb_length_drift_lt_0_5px"] = (max(errs) < 0.5, f"max {max(errs):.4f}px over {len(errs)} poses")
    # audio
    li, tp = loudness(mp4)
    gates["loudness_-14pm1_LUFS"] = (abs(li + 14) <= 1.0, f"{li} LUFS, TP {tp} dBFS")
    gates["narration_fits_cues"] = (all(n["fits"] for n in narr), [f"{n['start']}-{n['end']}" for n in narr])
    gates["narration_present"] = (len(narr) >= 3, len(narr))
    if style == "B2":
        _b2_gates(gates, ep, metas)
    ok = all(g[0] for g in gates.values())
    return {"pass": ok, "passed": f"{sum(g[0] for g in gates.values())}/{len(gates)}",
            "gates": {k: {"ok": bool(g[0]), "value": g[1]} for k, g in gates.items()}}


def flow_sheet(frames, out):
    cw, ch = 360, 640
    sheet = Image.new("RGB", (cw * 4, ch * 3), (0, 0, 0))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    for i, (t, arr) in enumerate(frames):
        im = Image.fromarray(arr).resize((cw, ch), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, ch - 44, 120, ch], fill=(0, 0, 0)); d.text((8, ch - 40), f"{t:.1f}s", fill=(255, 255, 255), font=font)
        sheet.paste(im, ((i % 4) * cw, (i // 4) * ch))
    sheet.save(out, quality=90)


MUSCLES = ("quads", "glutes", "hamstrings", "chest", "triceps", "core", "shoulders", "calves")
FILL_MIN = 0.10          # solo side figure: swept bbox over each rep scene >= 10% of the frame
TALL_H_MIN = 0.50        # standing poses (squat/hinge/lunge): >= 50% of frame height in rep + close-up scenes
FILL_H_MIN = 0.17        # horizontal poses (push-up/plank/bridge): >= 17% of frame height tall
MUSCLE_MIN_PX2 = 1500    # every target muscle shape on screen, every solo frame
FAULT_MIN = 0.06         # wrong vs good form: >= 6% of body height apart at the least visible moment
HL_MIN_W = 18            # fault band width on screen (px)


def _b2_gates(gates, ep, metas):
    fps = 30.0
    # target muscles: right-form solo frames + ending show every listed muscle; wrong-form panels show none
    from .style_b2 import muscles_named
    want = ep.get("muscles", [])
    solo = [m for m in metas if m.get("solo")]
    worst = None; missing = 0; wrong_orange = 0
    for m in metas:
        got = {}
        for name, area in m.get("muscles", []):
            got[name] = max(got.get(name, 0), area)
        if m.get("solo") and not m.get("solo_good"):
            wrong_orange += bool(got)
            continue
        if m.get("solo") or m.get("scene") == "conclusion":
            for w in m.get("shown_muscles", []):
                a = got.get(w, 0)
                worst = a if worst is None else min(worst, a)
                missing += a < MUSCLE_MIN_PX2
    ok = bool(want) and all(w in MUSCLES for w in want) and missing == 0 and wrong_orange == 0 and worst is not None
    gates["target_muscle_visible"] = (ok, f"{'+'.join(want) or 'none listed'}; min {min(worst or 0, 1e5):.0f}px2, {missing} misses, {wrong_orange} wrong-form frames with orange")
    # any muscle named on screen must be highlighted in that frame
    bad = []
    for m in metas:
        drawn = {n for n, _ in m.get("muscles", [])}
        for x in m["text"]:
            miss = muscles_named(x["text"]) - drawn
            if miss:
                bad.append(f"t={m['t']:.2f}s {x['text']!r} names {sorted(miss)}")
    zh = [h for m in metas for h in m.get("front_zone_hits", [])]
    gates["no_highlight_in_groin_zone"] = (sum(zh) == 0, f"{sum(zh)} highlighted samples in the front-view groin zone over {len(zh)} figures")
    gates["named_muscle_is_highlighted"] = (not bad, f"{len(bad)} frames" + (f", first {bad[0]}" if bad else ""))
    # slow reps: period, measured period, and every rep scene shows a full rep
    P = float(ep.get("rep_period", 0))
    ups = [m["t"] for prev, m in zip(metas, metas[1:]) if m.get("solo") and prev.get("solo")
           and prev.get("phase", 0) < 0.5 <= m.get("phase", 0)]
    gaps = [b - a for a, b in zip(ups, ups[1:]) if b - a < 2 * P + 0.1]
    scenes = (ep.get("B2") or {}).get("scenes", [])
    short = [sc["type"] for sc in scenes if sc["type"] in ("reps", "closeup", "split") and sc["t1"] - sc["t0"] < P - 1e-6]
    meas = min(gaps) if gaps else P
    gates["min_rep_duration_ge_2_2s"] = (P >= 2.2 and meas >= 2.2 - 1 / fps and not short,
                                         f"period {P}s, measured {meas:.2f}s" + (f", scenes shorter than a rep: {short}" if short else ""))
    # framing: per solo scene, the figure's swept box over the rep fills the frame
    groups = {}
    for m in solo:
        key = m.get("scene_t0", m.get("scene"))
        for (x0, y0, x1, y1) in m.get("figs", []):
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, 1080), min(y1, 1920)
            g = groups.setdefault(key, [x0, y0, x1, y1])
            g[0], g[1], g[2], g[3] = min(g[0], x0), min(g[1], y0), max(g[2], x1), max(g[3], y1)
    fills = {k: ((g[2] - g[0]) * (g[3] - g[1]) / (1080 * 1920), (g[3] - g[1]) / 1920) for k, g in groups.items()}
    horiz = ep.get("exercise") in ("pushup", "plank", "bridge")
    tall = ep.get("exercise") in ("squat", "hinge", "lunge")
    okf = bool(fills) and all(a >= FILL_MIN and (not horiz or h >= FILL_H_MIN) and (not tall or h >= TALL_H_MIN)
                              for a, h in fills.values())
    gates["figure_fill"] = (okf, ", ".join(f"{k}: {a:.1%} area / {h:.1%} tall" for k, (a, h) in fills.items()))
    # faces: profile faces look where the body faces
    faces = [f for m in metas for f in m.get("faces", [])]
    gates["profile_face_matches_body"] = (bool(faces) and min(faces) > 0.5, f"min dot {min(faces) if faces else None} over {len(faces)} side faces")
    # fault highlight strength + readability of wrong forms
    hl = [h for m in metas for h in m.get("hl", [])]
    wrong_used = any(sc.get("form", "good") != "good" for sc in scenes if sc["type"] in ("reps", "closeup")) or \
        any(sc.get(k, {}).get("form", "good") != "good" for sc in scenes for k in ("top", "bottom"))
    gates["fault_highlight_strong"] = ((not wrong_used) or (bool(hl) and min(h[0] for h in hl) >= HL_MIN_W),
                                       f"min band {min(h[0] for h in hl) if hl else 0:.0f}px over {len(hl)} segments")
    sep = metas[0].get("fault_sep", {})
    gates["wrong_form_readable_all_rep"] = (all(v >= FAULT_MIN for v in sep.values()), sep)
    var = metas[0].get("variants", {})
    gates["think_end_variant_valid"] = (var.get("think", "scratch") in ("scratch", "shrug", "hold")
                                        and var.get("conclusion", "flex") in ("flex", "victory", "thumbs"), var)
