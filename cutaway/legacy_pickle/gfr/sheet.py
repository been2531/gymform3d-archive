"""Verification artefacts: frame contact sheet + report for the encoded MP4."""
from __future__ import annotations

import json
import os
import subprocess

from PIL import Image, ImageDraw

from . import theme as T

FFMPEG = "/workspace/.local/lib/python3.12/site-packages/static_ffmpeg/bin/linux/ffmpeg"
FFPROBE = "/workspace/.local/lib/python3.12/site-packages/static_ffmpeg/bin/linux/ffprobe"

TIMES = [0.30, 1.60, 2.90, 3.90, 5.20, 6.90, 7.85, 8.60, 9.80, 11.20, 12.30, 14.70]


def probe(path: str) -> dict:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "format=duration,size:stream=codec_type,codec_name,width,height,"
                        "r_frame_rate,nb_frames,duration,pix_fmt,sample_rate,channels",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def build(cfg: dict, root: str, out_png: str, mp4: str | None = None,
          times=None, title: str | None = None) -> str:
    mp4 = mp4 or os.path.join(os.path.dirname(out_png), "final.mp4")
    times = times or TIMES
    title = title or "GYMFORM.3D  ·  15s REEL FRAME CHECK  (0.3s – 14.7s)"
    tmp = os.path.join(os.path.dirname(out_png), "_sheet_frames")
    os.makedirs(tmp, exist_ok=True)
    cw, ch, cols = 360, 640, 4
    rows = (len(times) + cols - 1) // cols
    labels = 34
    sheet = Image.new("RGB", (cols * cw, rows * (ch + labels) + 62), T.hex_rgb("#0B0C0D"))
    d = ImageDraw.Draw(sheet)
    T.draw_text(d, (24, 30), title,
                T.font("cond_bold", 30), T.hex_rgb("#E8EAE6"), anchor="lm", tracking=2.4)

    for i, t in enumerate(times):
        p = os.path.join(tmp, f"f{t:05.2f}.png")
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{t:.2f}", "-i", mp4, "-frames:v", "1", p], check=True)
        im = Image.open(p).convert("RGB").resize((cw, ch), Image.LANCZOS)
        x = (i % cols) * cw
        y = 62 + (i // cols) * (ch + labels)
        sheet.paste(im, (x, y))
        d.rectangle((x, y, x + cw - 1, y + ch - 1), outline=T.hex_rgb("#2A2D30"))
        T.draw_text(d, (x + 10, y + ch + 17), f"{t:.2f}s",
                    T.font("cond_bold", 24), T.hex_rgb("#B4FF39"), anchor="lm", tracking=1.6)
        d.line([(x, y + ch), (x + cw, y + ch)], fill=T.hex_rgb("#2A2D30"))

    sheet.save(out_png)
    info = probe(mp4)
    with open(out_png.replace(".png", "_probe.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    return out_png
