#!/usr/bin/env python3
"""gymform_reels renderer CLI.

    python render.py stills  --config data/gymform_ep01.json --times 0.5,3.8,8.4,12.0,14.2
    python render.py reel    --config data/gymform_ep01.json
    python render.py audio   --config data/gymform_ep01.json
    python render.py mux     --config data/gymform_ep01.json
    python render.py cards   --config data/gymform_ep01.json
    python render.py sheet   --config data/gymform_ep01.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from gfr import media as M
from gfr import theme as T

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(PKG))   # /workspace, where downloads/ lives
sys.path.insert(0, PKG)


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def builder(cfg, root):
    """Pick the renderer for this config: CUTAWAY template or the ep01 reel."""
    if cfg.get("format", {}).get("kind") == "CUTAWAY":
        from gfr.cutaway import Cutaway
        return Cutaway(cfg, root)
    from gfr.reel import Reel
    return Reel(cfg, root)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stills", "reel", "audio", "mux", "cards", "sheet", "all"])
    ap.add_argument("--config", default=os.path.join(PKG, "data", "gymform_ep01.json"))
    ap.add_argument("--times", default="0.4,2.0,3.7,4.6,7.8,9.2,12.2,14.4")
    ap.add_argument("--out", default=os.path.join(PKG, "build"))
    ap.add_argument("--tag", default=None, help="output file prefix (default: video/final/...)")
    ap.add_argument("--segments", type=int, default=0,
                    help="split the reel encode into N parts and concatenate (long renders)")
    ap.add_argument("--no-voice", action="store_true", help="SFX + bed only (F1 silent variant)")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    if args.no_voice:
        cfg.setdefault("audio", {})["lines"] = []
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag

    if args.mode == "stills":
        r = builder(cfg, ROOT)
        for ts in args.times.split(","):
            t = float(ts)
            arr = r.frame(t)
            p = os.path.join(args.out, f"still_{t:05.2f}.png")
            M.to_pil(arr).save(p)
            print("wrote", p)
        return 0

    video = os.path.join(PKG, "build", f"{tag + '_' if tag else ''}video.mp4")
    wav = os.path.join(PKG, "build", f"{tag + '_' if tag else ''}audio.wav")
    final = os.path.join(PKG, "build", f"{tag + '_' if tag else ''}final.mp4")

    if args.mode in ("reel", "all"):
        r = builder(cfg, ROOT)
        if args.segments and args.segments > 1:
            import subprocess
            import imageio_ffmpeg
            n = int(round(r.dur * r.fps))
            bounds = [round(n * k / args.segments) for k in range(args.segments + 1)]
            segs = []
            for k in range(args.segments):
                sp = os.path.join(PKG, "build", f"{tag + '_' if tag else ''}seg{k}.mp4")
                print(f"segment {k + 1}/{args.segments}  frames {bounds[k]}..{bounds[k + 1]}", flush=True)
                r.render_video(sp, bounds[k], bounds[k + 1])
                segs.append(sp)
            lst = os.path.join(PKG, "build", f"{tag + '_' if tag else ''}segs.txt")
            with open(lst, "w") as fh:
                for sp in segs:
                    fh.write(f"file '{sp}'\n")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                            "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                            "-movflags", "+faststart", video], check=True)
            for sp in segs:
                os.remove(sp)
            os.remove(lst)
        else:
            r.render_video(video)
        print("video:", video)

    if args.mode in ("audio", "all"):
        from gfr import audio
        audio.build(cfg, ROOT, wav)
        print("audio:", wav)

    if args.mode in ("mux", "all"):
        from gfr import audio
        audio.mux(video, wav, final, cfg["format"]["duration"])
        print("final:", final)

    if args.mode in ("cards", "all") and cfg.get("format", {}).get("kind") != "CUTAWAY":
        from gfr import carousel
        outdir = os.path.join(PKG, "build")
        for name, path in carousel.render_all(cfg, ROOT, outdir):
            print("card:", path)

    if args.mode in ("sheet", "all"):
        from gfr import sheet
        p = sheet.build(cfg, ROOT, os.path.join(PKG, "build", f"{tag + '_' if tag else ''}frames_check.png"),
                        mp4=final, times=cfg.get("sheet", {}).get("times"),
                        title=cfg.get("sheet", {}).get("title"))
        print("sheet:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
