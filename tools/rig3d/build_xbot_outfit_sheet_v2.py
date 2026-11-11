#!/usr/bin/env python3
"""Compose xbot_outfit_v2.png from rendered stills. System Python (PIL).

  python3 tools/rig3d/build_xbot_outfit_sheet_v2.py
  python3 tools/rig3d/build_xbot_outfit_sheet_v2.py --facts path/to/facts.json
"""
from __future__ import annotations

import json
import os
import sys

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(ROOT, "output/_basemodels/eval")
PANEL = (540, 960)
CLOSE = (540, 720)
BG = (16, 16, 18)
FG = (232, 232, 234)
MUTED = (158, 158, 164)
ACCENT = (220, 200, 140)
OK = (120, 210, 150)
PAD = 14
LABEL_H = 44
TITLE_H = 100
FOOT_H = 110

VARIANTS = (
    ("plain", "plain X Bot"),
    ("v1A", "v1 A  ·  crimson gloves + charcoal beanie"),
    ("v2A", "v2 A  ·  compact 10–12 oz + ribbed slouch beanie"),
)
POSES = (
    ("idle", "guard", "Fighting Idle — guard"),
    ("jab", "full", "Lead Jab — full extension"),
)
CLOSES = (
    ("close_glove", "v2 glove close  ·  jab, 3⁄4"),
    ("close_beanie", "v2 beanie close  ·  fold-up cuff + rib knit"),
    ("close_ref", "CC0 reference  ·  modern boxing glove (Wikimedia, Petey21, 2014)"),
)


def _font(size, bold=False):
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_panel(path, size):
    if path and os.path.isfile(path):
        im = Image.open(path).convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.Resampling.LANCZOS)
        return im
    im = Image.new("RGB", size, (40, 28, 28))
    ImageDraw.Draw(im).text((40, 80), f"missing\n{path}", font=_font(22, True),
                            fill=(220, 110, 96))
    return im


def load_ref_crop(path, size):
    """Crop one glove from the CC0 pair photo and fit it into the panel."""
    if not path or not os.path.isfile(path):
        return load_panel(None, size)
    im = Image.open(path).convert("RGB")
    w, h = im.size
    # Left glove occupies roughly the left half; keep a little margin.
    box = (int(w * 0.02), int(h * 0.08), int(w * 0.52), int(h * 0.98))
    crop = im.crop(box)
    # Lift shadows slightly so the black leather reads on a dark sheet.
    crop = ImageEnhance.Brightness(crop).enhance(1.18)
    crop = ImageEnhance.Contrast(crop).enhance(1.08)
    # letterbox into the panel
    fitted = ImageOps.contain(crop, (size[0] - 28, size[1] - 28), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (28, 28, 30))
    x = (size[0] - fitted.size[0]) // 2
    y = (size[1] - fitted.size[1]) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _wrap(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if font.getlength(trial) > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def compose(stills, notes, out_path):
    cols = len(VARIANTS)
    rows = len(POSES)
    col_w, row_body = PANEL
    close_h = CLOSE[1]
    w = PAD * (cols + 1) + col_w * cols
    h = TITLE_H + (LABEL_H + row_body) * rows + (LABEL_H + close_h) + FOOT_H + PAD
    sheet = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 8, 14),
            "X Bot outfit v2  —  plain  |  v1 A  |  v2 A   (crimson gloves + charcoal beanie)",
            font=_font(22, True), fill=FG)
    sub = ("Procedural / CC0 geometry (no third-party meshes).  v2 gloves follow 10–12 oz "
           "proportions: compact rounded fist, padded knuckle, attached thumb, 9 cm cuff.  "
           "Beanie: fold-up cuff, geometric rib knit, slight slouch.  Headgear dropped.  "
           "Same 50 mm camera per row, ≥5% margin.  Close-ups are tighter 50 mm crops.")
    for i, line in enumerate(_wrap(dr, sub, _font(13), w - 2 * PAD - 16)[:3]):
        dr.text((PAD + 8, 48 + i * 16), line, font=_font(13), fill=MUTED)

    for r, (clip, stage, pose_title) in enumerate(POSES):
        y0 = TITLE_H + r * (LABEL_H + row_body)
        for c, (key, title) in enumerate(VARIANTS):
            x = PAD + c * (col_w + PAD)
            still_key = f"{clip}_{stage}_{key}"
            path = stills.get(still_key)
            if not path:
                path = os.path.join(EVAL, f"xbot_outfit_v2_{clip}_{stage}_{key}.png")
            sheet.paste(load_panel(path, PANEL), (x, y0 + LABEL_H))
            cap = f"{title}    ·    {pose_title}"
            fill = MUTED if key == "plain" else (ACCENT if key == "v1A" else OK)
            dr.text((x + 8, y0 + 12), cap, font=_font(14, True), fill=fill)

    # Close-up row
    y0 = TITLE_H + rows * (LABEL_H + row_body)
    for c, (key, title) in enumerate(CLOSES):
        x = PAD + c * (col_w + PAD)
        path = stills.get(key)
        if key == "close_ref":
            panel = load_ref_crop(path, CLOSE)
        else:
            if not path:
                path = os.path.join(EVAL, f"xbot_outfit_v2_{key}.png")
            panel = load_panel(path, CLOSE)
        sheet.paste(panel, (x, y0 + LABEL_H))
        fill = MUTED if key == "close_ref" else OK
        dr.text((x + 8, y0 + 12), title, font=_font(13, True), fill=fill)

    foot = "  |  ".join(notes) if notes else "See facts JSON for fit notes."
    fw = _font(13)
    lines = _wrap(dr, foot, fw, w - 2 * PAD - 8)
    fy = TITLE_H + (LABEL_H + row_body) * rows + (LABEL_H + close_h) + 16
    for i, line in enumerate(lines[:4]):
        dr.text((PAD + 8, fy + i * 20), line, font=fw, fill=MUTED)

    sheet.save(out_path, "PNG", optimize=True)
    print("WROTE", out_path, sheet.size, os.path.getsize(out_path))
    return out_path


def main():
    argv = sys.argv[1:]
    facts_path = os.path.join(EVAL, "xbot_outfit_v2_facts.json")
    if "--facts" in argv:
        facts_path = argv[argv.index("--facts") + 1]
    stills, notes = {}, []
    if os.path.isfile(facts_path):
        with open(facts_path) as fh:
            facts = json.load(fh)
        stills = facts.get("stills") or {}
        notes = facts.get("honest_notes") or []
    out = os.path.join(EVAL, "xbot_outfit_v2.png")
    compose(stills, notes, out)


if __name__ == "__main__":
    main()
