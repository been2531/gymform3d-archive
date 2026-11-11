#!/usr/bin/env python3
"""Compose xbot_outfit_v1.png from rendered stills. System Python (PIL).

  python3 tools/rig3d/build_xbot_outfit_sheet.py
  python3 tools/rig3d/build_xbot_outfit_sheet.py --facts path/to/facts.json
"""
from __future__ import annotations

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(ROOT, "output/_basemodels/eval")
PANEL = (540, 960)
BG = (16, 16, 18)
FG = (232, 232, 234)
MUTED = (158, 158, 164)
ACCENT = (220, 200, 140)
OK = (120, 210, 150)
PAD = 14
LABEL_H = 44
TITLE_H = 96
FOOT_H = 88

VARIANTS = (
    ("plain", "plain X Bot"),
    ("A", "A  ·  crimson gloves + charcoal beanie"),
    ("B", "B  ·  black gloves + matte headgear"),
)
POSES = (
    ("idle", "guard", "Fighting Idle — guard"),
    ("jab", "full", "Lead Jab — full extension"),
)


def _font(size, bold=False):
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_panel(path):
    if path and os.path.isfile(path):
        im = Image.open(path).convert("RGB")
        if im.size != PANEL:
            im = im.resize(PANEL, Image.Resampling.LANCZOS)
        return im
    im = Image.new("RGB", PANEL, (40, 28, 28))
    ImageDraw.Draw(im).text((40, 80), f"missing\n{path}", font=_font(22, True),
                            fill=(220, 110, 96))
    return im


def compose(stills, notes, out_path):
    cols = len(VARIANTS)
    rows = len(POSES)
    col_w, row_body = PANEL
    w = PAD * (cols + 1) + col_w * cols
    h = TITLE_H + (LABEL_H + row_body) * rows + FOOT_H + PAD
    sheet = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 8, 16),
            "X Bot outfit v1  —  plain  |  gloves+beanie  |  gloves+headgear",
            font=_font(24, True), fill=FG)
    sub = ("Procedural meshes (no third-party assets).  Colour A: crimson matte leather + charcoal knit.  "
           "Colour B: lifted-black leather + matte amateur headgear.  Same 50 mm camera per row, ≥5% margin.")
    dr.text((PAD + 8, 52), sub, font=_font(13), fill=MUTED)

    for r, (clip, stage, pose_title) in enumerate(POSES):
        y0 = TITLE_H + r * (LABEL_H + row_body)
        for c, (key, title) in enumerate(VARIANTS):
            x = PAD + c * (col_w + PAD)
            still_key = f"{clip}_{stage}_{key}"
            path = stills.get(still_key)
            if not path:
                path = os.path.join(EVAL, f"xbot_outfit_{clip}_{stage}_{key}.png")
            sheet.paste(load_panel(path), (x, y0 + LABEL_H))
            cap = f"{title}    ·    {pose_title}"
            fill = MUTED if key == "plain" else (OK if key == "A" else ACCENT)
            dr.text((x + 8, y0 + 12), cap, font=_font(14, True), fill=fill)

    foot = "  |  ".join(notes) if notes else "See facts JSON for fit notes."
    fw = _font(13)
    words = foot.split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if fw.getlength(trial) > w - 2 * PAD - 8 and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    fy = TITLE_H + (LABEL_H + row_body) * rows + 16
    for i, line in enumerate(lines[:3]):
        dr.text((PAD + 8, fy + i * 20), line, font=fw, fill=MUTED)

    sheet.save(out_path, "PNG", optimize=True)
    print("WROTE", out_path, sheet.size, os.path.getsize(out_path))
    return out_path


def main():
    argv = sys.argv[1:]
    facts_path = os.path.join(EVAL, "xbot_outfit_v1_facts.json")
    if "--facts" in argv:
        facts_path = argv[argv.index("--facts") + 1]
    stills, notes = {}, []
    if os.path.isfile(facts_path):
        with open(facts_path) as fh:
            facts = json.load(fh)
        stills = facts.get("stills") or {}
        notes = facts.get("honest_notes") or []
    out = os.path.join(EVAL, "xbot_outfit_v1.png")
    compose(stills, notes, out)


if __name__ == "__main__":
    main()
