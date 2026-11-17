"""Batch validator for new episode JSONs (run when an AI/human writes a batch of ~10).

  /workspace/venv-pilots/bin/python -m autoreel.validate topics/episodes/*.json [--add] [--render]

Static gates (fast): schema, scene timing 1.7-3.0 s and contiguous, hook 2-4 words and <=2 lines,
cue/labels fit one line inside the frame, known exercise/forms/faces/pain spot/guide, caption lint,
every narration line (Kokoro, 1.12x) ends before the next cue.
--render: full render + all QA gates (into output/autoreel/_validate).  --add: append passing files to topics/queue.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import skia  # noqa: E402
from autoreel import caption, common  # noqa: E402
from autoreel.render import render  # noqa: E402
from pilots import audio, text as T  # noqa: E402
from pilots.style_b2 import EXERCISES, PAIN_SPOTS, MUSCLES, THINK_VARIANTS, END_VARIANTS, muscles_named, scene_shown_muscles, scene_texts  # noqa: E402

FACES = {"strain", "shocked", "worried", "happy", "proud"}
TYPES = {"reps", "think", "split", "closeup", "conclusion"}
GUIDES = {"line", "depth", "vertical", "torso"}


def _w(text, font, cap):
    return skia.Font(T.typeface(font), cap / T.cap_ratio(font)).measureText(text)


def _lines(text, font, cap, max_w):
    f = skia.Font(T.typeface(font), cap / T.cap_ratio(font))
    lines = T.wrap(f, text, max_w)
    if any(f.measureText(ln) > max_w for ln in lines):  # a single word too wide for the frame
        return 99
    return len(lines)


def check(path, tts=True):
    errs = []
    try:
        ep = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        return [f"not JSON: {e}"]
    for k in ("id", "exercise", "fps", "duration", "rep_period", "forms", "B2", "caption"):
        if k not in ep:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    if ep["exercise"] not in EXERCISES:
        errs.append(f"unknown exercise {ep['exercise']} (add a pose function in pilots/rig.py)")
    if ep["fps"] != 30 or not 10 <= ep["duration"] <= 15:
        errs.append("fps must be 30 and duration 10-15 s")
    if ep["rep_period"] < 2.2:
        errs.append(f"rep_period {ep['rep_period']} s < 2.2 s (too fast to follow)")
    mus = ep.get("muscles") or []
    if not mus or any(m not in MUSCLES for m in mus):
        errs.append(f"muscles must be a non-empty list from {MUSCLES}: {mus}")
    if "good" not in ep["forms"]:
        errs.append("forms needs 'good'")
    sc = ep["B2"]["scenes"]
    if sc[0]["t0"] != 0 or abs(sc[-1]["t1"] - ep["duration"]) > 1e-6:
        errs.append("scenes must cover 0..duration")
    for i, s in enumerate(sc):
        d = s["t1"] - s["t0"]
        if s["type"] in ("reps", "split", "closeup"):  # must show at least one full slow rep
            if not ep["rep_period"] - 1e-6 <= d <= 3.6 + 1e-6:
                errs.append(f"scene {i} ({s['type']}) lasts {d:.2f}s (need rep_period {ep['rep_period']}-3.6)")
        elif not 1.7 - 1e-6 <= d <= 3.0 + 1e-6:
            errs.append(f"scene {i} lasts {d:.2f}s (need 1.7-3.0)")
        shown = set(scene_shown_muscles(ep, s))
        for txt in scene_texts(s):
            miss = muscles_named(txt) - shown
            if miss:
                errs.append(f"scene {i} text {txt!r} names {sorted(miss)} but the scene highlights {sorted(shown) or 'none'}")
        if s.get("muscles") and any(m not in MUSCLES for m in s["muscles"]):
            errs.append(f"scene {i} muscles {s['muscles']}")
        if s["type"] == "think" and s.get("variant") and s["variant"] not in THINK_VARIANTS:
            errs.append(f"scene {i} think variant {s['variant']}")
        if s["type"] == "conclusion" and s.get("variant") and s["variant"] not in END_VARIANTS:
            errs.append(f"scene {i} ending variant {s['variant']}")
        if i and abs(s["t0"] - sc[i - 1]["t1"]) > 1e-6:
            errs.append(f"scene {i} not contiguous")
        if s["type"] not in TYPES:
            errs.append(f"scene {i} type {s['type']}")
        for key in ("form",):
            if key in s and s[key] not in ep["forms"]:
                errs.append(f"scene {i} form {s[key]} not in forms")
        for side in ("top", "bottom"):
            if side in s:
                if s[side]["form"] not in ep["forms"]:
                    errs.append(f"scene {i} {side} form missing")
                if _w(s[side]["label"], "marker", 80) > 470:
                    errs.append(f"scene {i} label too wide: {s[side]['label']}")
        if s.get("face") and s["face"] not in FACES:
            errs.append(f"scene {i} face {s['face']}")
        if s["type"] == "think" and s.get("pain", "lower_back") not in PAIN_SPOTS:
            errs.append(f"scene {i} pain spot {s.get('pain')}")
        g = s.get("guide")
        if g and (g if isinstance(g, str) else g.get("type")) not in GUIDES:
            errs.append(f"scene {i} guide {g}")
        if s.get("title"):
            if _lines(s["title"], "marker", 112, 1010) > 2 and _lines(s["title"], "kalam", 112, 1010) > 2:
                errs.append(f"scene {i} title needs >2 lines: {s['title']}")
        if s.get("cue") and _w(s["cue"]["text"], "marker", 70) + 100 > 1000:
            errs.append(f"scene {i} cue too long for one line: {s['cue']['text']}")
        if s.get("sub") and _w(s["sub"], "marker", 76) > 980:
            errs.append(f"scene {i} sub too wide")
    hook = sc[0].get("title", "")
    if not 2 <= len(hook.split()) <= 4:
        errs.append(f"hook must be 2-4 words: {hook!r}")
    cap = caption.build(ep)
    errs += [f"caption: {e}" for e in caption.lint(cap)]
    if any(x in json.dumps(ep) for x in common.NEVER_REPOST):
        errs.append("references a never-repost media id")
    if tts:
        nar = ep["B2"]["narration"]
        for i, ln in enumerate(nar):
            s = audio.tts(ln["text"], ep.get("voice", "am_michael"), ep.get("speed", 1.12), os.path.join(common.WORK, "_tts_cache"))
            end = ln["t"] + len(s) / audio.SR
            nxt = nar[i + 1]["t"] if i + 1 < len(nar) else ep["duration"]
            if end > nxt - 0.03:
                errs.append(f"narration {i} ends {end:.2f}s > next cue {nxt:.2f}s: {ln['text']!r}")
    return errs


def variant_of(ep, typ):
    sc = next((x for x in ep["B2"]["scenes"] if x["type"] == typ), {})
    if sc.get("variant"):
        return sc["variant"]
    opts = THINK_VARIANTS if typ == "think" else END_VARIANTS
    return opts[(sum(ord(ch) for ch in ep.get("id", "")) + (0 if typ == "think" else 1)) % len(opts)]


def queue_variety(paths):
    """Batch gate: neighbouring queue episodes never reuse the same think pose or the same ending pose."""
    errs = []
    eps = [json.load(open(os.path.join(common.REPO, p))) for p in paths]
    for a, b in zip(eps, eps[1:]):
        for typ in ("think", "conclusion"):
            if variant_of(a, typ) == variant_of(b, typ):
                errs.append(f"{a['id']} -> {b['id']}: same {typ} variant {variant_of(a, typ)}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    q = common.load_json(common.QUEUE, {"episodes": []})
    ok_all = True
    for p in a.files:
        rel = os.path.relpath(os.path.abspath(p), common.REPO)
        errs = check(p)
        if not errs and a.render:
            qa = render(os.path.abspath(p), os.path.join(common.WORK, "_validate"), tag="B")
            if not qa.get("pass"):
                errs.append("render QA: " + ", ".join(k for k, g in qa.get("gates", {}).items() if not g["ok"]) or qa.get("error", "?"))
            else:
                print(f"   render {qa['passed']} in {qa.get('total_s')} s")
        print(("OK   " if not errs else "FAIL ") + rel)
        for e in errs:
            print("     - " + e)
        ok_all &= not errs
        if not errs and a.add and rel not in q["episodes"]:
            q["episodes"].append(rel)
    ve = queue_variety(q["episodes"])
    print("queue variety: " + ("OK" if not ve else "FAIL"))
    for e in ve:
        print("     - " + e)
    ok_all &= not ve
    if a.add:
        common.write_json(common.QUEUE, q)
        print(f"queue: {len(q['episodes'])} episodes")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
