#!/usr/bin/env python3
"""One command from a topic entry to a publish-ready (but NOT published) package.

    nice -n 19 python3 cutaway/make_reel.py --topic knee

Inputs : cutaway/topics/<id>.json  +  output/cutaway/assets/<id>/{skin,outer,bone,macro}_raw.* + macro_crop.json
Outputs: output/cutaway/<id>/  reel.mp4, cover.jpg, carousel_1.jpg, carousel_2.jpg, caption.txt,
         contact_sheet.jpg, qa.json, report.md
Exit   : 0 = every gate passed, 2 = stopped after the retry budget (see report.md), 1 = crash.

Flow: register layers (if needed) -> pre-render gates -> render -> procedural audio -> mux ->
post-render gates.  On a failed gate we change render params from a fixed table and retry, at
most twice in total; gates that params cannot fix (wrong facts, caption, words, assets) stop the
run immediately.  Nothing is uploaded or posted here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cutaway2 as C  # noqa: E402
import qa  # noqa: E402
import register  # noqa: E402
import sound  # noqa: E402

MAX_RETRIES = 2
# gate -> how to change params (None = not fixable by params -> stop)
FIXES = {
    "motion_first_0_5s": lambda p: p.update(push=p["push"] + 0.05, open_at=0.05),
    "no_static_hold": lambda p: p.update(inner_drift=p["inner_drift"] * 1.6, breath=p["breath"] * 1.6),
    "window_holes": lambda p: p.update(win_scale=p.get("win_scale", 1.0) * 0.85),
    "text_contrast": lambda p: p.update(pill_alpha=248),
    "text_clear_of_window": lambda p: p.update(screen_y_shift=p.get("screen_y_shift", 0) + 60,
                                               win_scale=p.get("win_scale", 1.0) * 0.92),
    "registration": lambda p: p.update(bloom=False),
    "loudness": "audio",
    "bottom_clear": None, "label_words": None, "text_size": None, "facts_sourced": None,
    "caption": None, "duration": None,
    "view_in_content": lambda p: p.update(push=p["push"] + 0.04),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_registered(adir):
    rj = os.path.join(adir, "registered", "registration.json")
    raws = [os.path.join(adir, f) for f in os.listdir(adir) if "_raw." in f or f == "macro_crop.json"]
    if not os.path.exists(rj) or os.path.getmtime(rj) < max(os.path.getmtime(p) for p in raws):
        log("registering layers")
        register.run(adir)
    return json.load(open(rj))


def contact_sheet(mp4, out, dur, n=12):
    ts = [round(0.05 + k * (dur - 0.15) / (n - 1), 2) for k in range(n)]
    tiles = []
    for t in ts:
        p = subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-ss", str(t), "-i", mp4, "-frames:v", "1",
                            "-vf", "scale=360:640", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True, check=True)
        tiles.append(Image.frombytes("RGB", (360, 640), p.stdout[:360 * 640 * 3]))
    sheet = Image.new("RGB", (6 * 360, 2 * 640 + 2 * 40), (12, 12, 14))
    f = ImageFont.truetype(C.FONT, 30)
    d = ImageDraw.Draw(sheet)
    for k, (t, im) in enumerate(zip(ts, tiles)):
        x, y = (k % 6) * 360, (k // 6) * 680
        sheet.paste(im, (x, y + 40))
        d.text((x + 10, y + 4), f"{t:.2f}s", font=f, fill=(180, 255, 57))
    sheet.save(out, quality=88)
    return out


def cover_and_carousel(reel, topic, odir):
    t_cover = 1.1
    fr = reel.frame(t_cover)
    Image.fromarray((fr * 255 + 0.5).astype(np.uint8)).save(os.path.join(odir, "cover.jpg"), quality=92)
    # slide 1: hook frame cropped 4:5
    im = Image.fromarray((fr * 255 + 0.5).astype(np.uint8)).crop((0, 110, 1080, 1460))
    im.save(os.path.join(odir, "carousel_1.jpg"), quality=92)
    # slide 2: the three inner layers + the sourced stat
    s2 = Image.new("RGB", (1080, 1350), (8, 9, 10))
    d = ImageDraw.Draw(s2)
    big = ImageFont.truetype(C.FONT, 84)
    mid = ImageFont.truetype(C.FONT, 60)
    small = ImageFont.truetype(os.path.join(HERE, "assets", "fonts", "BarlowCondensed-SemiBold.ttf"), 34)
    shots = [s for s in topic["shots"] if s.get("label") and s.get("kind") in ("cut", "macro")][:3]
    reel.hide_text = True     # clean window crops for the carousel (no leader lines)
    for k, s in enumerate(shots):
        t = s["t"] + 1.2
        f = reel.frame(t, want_info=True)
        jx, jy, r = reel.last_info["win"]
        rr = int(min(r, 330))
        crop = Image.fromarray((f * 255 + 0.5).astype(np.uint8)).crop((int(jx - rr), int(jy - rr), int(jx + rr), int(jy + rr)))
        crop = crop.resize((300, 300))
        mask = Image.new("L", (300, 300), 0)
        ImageDraw.Draw(mask).ellipse((2, 2, 298, 298), fill=255)
        cx = 40 + k * 340
        s2.paste(crop, (cx, 120), mask)
        d.ellipse((cx, 120, cx + 300, 420), outline=C.LIME, width=4)
        words = s["label"].split()
        lines = [" ".join(words[:2]), " ".join(words[2:])] if mid.getlength(s["label"]) > 300 else [s["label"]]
        for j, ln in enumerate(lines):
            d.text((cx + 150 - mid.getlength(ln) / 2, 450 + j * 66), ln, font=mid, fill=(255, 255, 255))
    reel.hide_text = False
    d.text((60, 40), f"INSIDE YOUR {topic['joint_name'].upper()}", font=mid, fill=C.LIME)
    c = topic.get("counter")
    if c:
        fact = next(f for f in topic["facts"] if f["id"] == c["fact"])
        num = c["fmt"].format(c["to"])
        d.text((60, 640), num, font=ImageFont.truetype(C.FONT, 220), fill=(255, 255, 255))
        d.text((60, 900), c["caption"], font=big, fill=(255, 255, 255))
        lab = next((s["label"] for s in topic["shots"] if s.get("kind") == "counter"), "")
        d.text((60, 1000), lab, font=big, fill=C.LIME)
        src = fact["source"]
        # wrap source
        words, line, y = src.split(), "", 1150
        for w_ in words:
            if small.getlength(line + " " + w_) > 960:
                d.text((60, y), line.strip(), font=small, fill=(200, 205, 198)); y += 40; line = ""
            line += " " + w_
        d.text((60, y), line.strip(), font=small, fill=(200, 205, 198))
    s2.save(os.path.join(odir, "carousel_2.jpg"), quality=92)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t_start = time.time()
    topic = json.load(open(os.path.join(HERE, "topics", f"{args.topic}.json")))
    adir = os.path.join(REPO, topic["assets"])
    odir = args.out or os.path.join(REPO, "output", "cutaway", topic["id"])
    os.makedirs(odir, exist_ok=True)
    reg = ensure_registered(adir)
    params = {}
    attempts = []
    final = os.path.join(odir, "reel.mp4")
    audio_gain = 0.0
    status = "fail"
    stop_reason = ""
    for attempt in range(MAX_RETRIES + 1):
        a = {"attempt": attempt + 1, "params": copy.deepcopy(params)}
        reel = C.Reel(copy.deepcopy(topic), adir, params)
        a["params_full"] = dict(reel.p)
        a["autolayout"] = reel.layout_log
        log(f"attempt {attempt + 1}: pre-render gates")
        pre = qa.prerender(reel, topic, reg)
        a["gates"] = dict(pre)
        failed = [g for g, v in pre.items() if not v["pass"]]
        if not failed:
            t0 = time.time()
            vid = os.path.join(odir, "_video.mp4")
            log("rendering video")
            C.render_video(reel, vid)
            a["render_s"] = round(time.time() - t0, 1)
            wav = os.path.join(odir, "_audio.wav")
            a["audio"] = sound.build(topic, wav, gain_db=audio_gain)
            sound.mux(vid, wav, final)
            post = qa.postrender(final)
            if not post["loudness"]["pass"] and attempt < MAX_RETRIES + 1:
                # audio-only fix, no re-render needed (still counts in the report)
                audio_gain += -14.0 - post["loudness"]["lufs"]
                a["audio_fix"] = audio_gain
                sound.build(topic, wav, gain_db=audio_gain)
                sound.mux(vid, wav, final)
                post = qa.postrender(final)
            a["gates"].update(post)
            failed = [g for g, v in post.items() if not v["pass"]]
        a["failed"] = failed
        attempts.append(a)
        log("failed gates:", failed or "none")
        if not failed:
            status = "pass"
            break
        unfixable = [g for g in failed if FIXES.get(g) is None]
        if unfixable:
            stop_reason = f"not fixable by render params: {unfixable}"
            break
        if attempt == MAX_RETRIES:
            stop_reason = "retry budget used (2)"
            break
        for g in failed:
            fx = FIXES[g]
            if callable(fx):
                base = dict(C.Reel(copy.deepcopy(topic), adir, params).p)
                base.update(params)
                fx(base)
                params = {k: v for k, v in base.items()}
    if status == "pass":
        log("cover / carousel / caption / sheet")
        cover_and_carousel(reel, topic, odir)
        open(os.path.join(odir, "caption.txt"), "w").write("\n".join(topic["caption"]) + "\n")
        contact_sheet(final, os.path.join(odir, "contact_sheet.jpg"), topic["duration"])
        for f in ("_video.mp4", "_audio.wav"):
            p = os.path.join(odir, f)
            if os.path.exists(p):
                os.remove(p)
    rep = {"topic": topic["id"], "status": status, "stop_reason": stop_reason,
           "criteria": qa.CRITERIA, "attempts": attempts, "wall_s": round(time.time() - t_start, 1),
           "outputs": sorted(os.listdir(odir)), "published": False}
    json.dump(rep, open(os.path.join(odir, "qa.json"), "w"), indent=1, default=str)
    lines = [f"# {topic['id']} — {status.upper()}", "", f"wall time: {rep['wall_s']} s", ""]
    if stop_reason:
        lines += [f"STOPPED: {stop_reason}", ""]
    last = attempts[-1]["gates"]
    for g in qa.CRITERIA:
        v = last.get(g)
        mark = "—" if v is None else ("PASS" if v["pass"] else "FAIL")
        val = "" if v is None else {k: x for k, x in v.items() if k != "pass"}
        lines.append(f"- {mark} {g}: {qa.CRITERIA[g]}  -> {val}")
    open(os.path.join(odir, "report.md"), "w").write("\n".join(lines) + "\n")
    log(status, rep["wall_s"], "s")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
