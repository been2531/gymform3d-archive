#!/usr/bin/env python3
"""Compose before/after lighting comparison: xbot_jab_lighting_v2.png."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(ROOT, "output/_basemodels/eval")
PANEL = (540, 960)
BG = (16, 16, 18)
FG = (232, 232, 234)
MUTED = (158, 158, 164)
ACCENT = (220, 200, 140)
OK = (120, 210, 150)
PAD = 12
LABEL_H = 46
TITLE_H = 88
FOOT_H = 70

ROWS = (
    ("Fighting Idle", "xbot_idle_guard_v1.png", "xbot_idle_guard.png"),
    ("Lead Jab — guard", "xbot_jab_guard_v1.png", "xbot_jab_guard.png"),
    ("Lead Jab — mid", "xbot_jab_mid_v1.png", "xbot_jab_mid.png"),
    ("Lead Jab — full", "xbot_jab_full_v1.png", "xbot_jab_full.png"),
)


def _font(size, bold=False):
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_panel(name):
    path = os.path.join(EVAL, name)
    if os.path.isfile(path):
        im = Image.open(path).convert("RGB")
        if im.size != PANEL:
            canvas = Image.new("RGB", PANEL, BG)
            im.thumbnail(PANEL, Image.Resampling.LANCZOS)
            x = (PANEL[0] - im.size[0]) // 2
            y = (PANEL[1] - im.size[1]) // 2
            canvas.paste(im, (x, y))
            return canvas
        return im
    im = Image.new("RGB", PANEL, (40, 28, 28))
    ImageDraw.Draw(im).text((40, 80), f"missing\n{name}", font=_font(22, True),
                            fill=(220, 110, 96))
    return im


def main():
    col_w = PANEL[0]
    row_h = PANEL[1] + LABEL_H
    w = PAD * 3 + col_w * 2
    h = TITLE_H + row_h * len(ROWS) + FOOT_H + PAD
    sheet = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(sheet)
    title = "X Bot lighting v2 — before  |  after"
    dr.text((PAD + 8, 18), title, font=_font(28, True), fill=FG)
    sub = ("Before: Standard, exposure -0.12, 16 spp, clay 0.46, no front fill/rim wrap, loose frame.  "
           "After: AgX, exposure 0, 96 spp + denoise, clay 0.65, key+front fill 36%+rim, 72% frame fill.")
    dr.text((PAD + 8, 54), sub, font=_font(13), fill=MUTED)

    headers = ("BEFORE  (v1, muddy/crushed)", "AFTER  (v2, studio fill + rim)")
    for r, (label, before, after) in enumerate(ROWS):
        y0 = TITLE_H + r * row_h
        for c, (head, fname) in enumerate(((headers[0], before), (headers[1], after))):
            x = PAD + c * (col_w + PAD)
            panel = load_panel(fname)
            sheet.paste(panel, (x, y0 + LABEL_H))
            cap = f"{head}   ·   {label}"
            fill = MUTED if c == 0 else OK
            dr.text((x + 8, y0 + 12), cap, font=_font(16, True), fill=fill)

    foot = ("Check: far-side limbs stay readable grey (not crushed black); clay highlights do not blow to white; "
            "floor contact shadow remains; glow still visible on punching-side delt / triceps.")
    dr.text((PAD + 8, TITLE_H + row_h * len(ROWS) + 18), foot, font=_font(13), fill=MUTED)

    out = os.path.join(EVAL, "xbot_jab_lighting_v2.png")
    sheet.save(out, "PNG", optimize=True)
    print("WROTE", out, sheet.size, os.path.getsize(out))
    return out


if __name__ == "__main__":
    main()
