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
    gates["length_10_12s"] = (10.0 <= dur <= 12.0, round(dur, 2))
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
    # limb lengths
    errs = [bone_errors(body, J) for m in metas for J in m["joints"]]
    gates["limb_length_drift_lt_0_5px"] = (max(errs) < 0.5, f"max {max(errs):.4f}px over {len(errs)} poses")
    # audio
    li, tp = loudness(mp4)
    gates["loudness_-14pm1_LUFS"] = (abs(li + 14) <= 1.0, f"{li} LUFS, TP {tp} dBFS")
    gates["narration_fits_cues"] = (all(n["fits"] for n in narr), [f"{n['start']}-{n['end']}" for n in narr])
    gates["narration_present"] = (len(narr) >= 3, len(narr))
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
