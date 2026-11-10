#!/usr/bin/env python3
"""Side-by-side demic reference + candidate porcelain stills."""
from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMIC_SHEET = os.path.join(ROOT, "output/_rig3d/v4_authored/peach34_air/ours_vs_demic_peak.png")
DEMIC_PANEL = (58, 84, 482, 804)  # left demic panel, 424x720
PANEL = (540, 960)
BG = (18, 18, 20)
CARD = (28, 28, 32)
FG = (230, 230, 232)
MUTED = (160, 160, 166)
FAIL = (220, 90, 80)
OK = (140, 200, 140)


def _font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _fit(im, size):
    canvas = Image.new("RGB", size, BG)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im = im.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - im.size[0]) // 2
    y = (size[1] - im.size[1]) // 2
    canvas.paste(im, (x, y))
    return canvas


def _wrap(draw, text, font, width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--demic", default=DEMIC_SHEET)
    ap.add_argument("--candidate", action="append", nargs=2, metavar=("TITLE", "PNG"),
                    dest="candidates", required=True)
    ap.add_argument("--facts", help="json {title: [lines...]}")
    args = ap.parse_args()
    facts = {}
    if args.facts and os.path.isfile(args.facts):
        facts = json.load(open(args.facts))

    demic = Image.open(args.demic).convert("RGB").crop(DEMIC_PANEL)
    demic = _fit(demic, PANEL)

    cols = [("demic reference", demic, facts.get("demic", [
        "Demic clay (left panel of the locked comparison sheet).",
        "Target look only. Do not copy the DEMIC wordmark.",
    ]))]
    for title, path in args.candidates:
        im = _fit(Image.open(path).convert("RGB"), PANEL)
        cols.append((title, im, facts.get(title, facts.get(path, []))))

    pad = 24
    header_h = 86
    caption_h = 220
    n = len(cols)
    W = pad + n * (PANEL[0] + pad)
    H = pad + header_h + PANEL[1] + pad + caption_h + pad
    sheet = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(sheet)
    title_f = _font(28, bold=True)
    sub_f = _font(16, bold=True)
    body_f = _font(15)
    draw.text((pad, 18), "Base-model search  |  porcelain clay, rear 3/4, vs demic", font=title_f, fill=FG)

    for i, (title, im, lines) in enumerate(cols):
        x = pad + i * (PANEL[0] + pad)
        y = pad + header_h
        # card
        draw.rounded_rectangle((x - 8, y - 36, x + PANEL[0] + 8, y + PANEL[1] + caption_h + 8), 10, fill=CARD)
        draw.text((x, y - 30), title, font=sub_f, fill=FG)
        sheet.paste(im, (x, y))
        ty = y + PANEL[1] + 10
        for line in lines[:10]:
            wrapped = _wrap(draw, line, body_f, PANEL[0] - 4)
            for wline in wrapped:
                color = FAIL if wline.lower().startswith(("fail", "fails", "missing", "not ")) or "FAIL" in wline else FG
                if wline.lower().startswith(("ok", "pass")):
                    color = OK
                draw.text((x, ty), wline, font=body_f, fill=color)
                ty += 18
                if ty > H - pad:
                    break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out, "PNG")
    print("wrote", args.out, sheet.size)


if __name__ == "__main__":
    main()
