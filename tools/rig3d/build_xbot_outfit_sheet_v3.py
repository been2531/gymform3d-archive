#!/usr/bin/env python3
"""Compose xbot_outfit_v3.png from rendered stills. System Python (PIL).

  python3 tools/rig3d/build_xbot_outfit_sheet_v3.py
  python3 tools/rig3d/build_xbot_outfit_sheet_v3.py --facts path/to/facts.json
"""
from __future__ import annotations

import json
import os
import sys

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps, ImageFilter, ImageChops

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
    ("v2A", "v2 A  ·  compact fist (too small) + ribbed beanie"),
    ("v3A", "v3 A  ·  10–12 oz, fist ~0.8× head, crown-rib beanie"),
)
POSES = (
    ("idle", "guard", "Fighting Idle — guard"),
    ("jab", "full", "Lead Jab — full extension"),
)
CLOSES = (
    ("close_glove", "v3 glove close  ·  jab, thumb side"),
    ("close_beanie", "v3 beanie close  ·  rib knit to crown + slouch fold"),
    ("close_ref", "CC0 reference  ·  same long-axis scale (Wikimedia, Petey21, 2014)"),
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


def _bbox_from_mask(mask, pad=4):
    m = mask.filter(ImageFilter.MaxFilter(3))
    box = m.getbbox()
    if not box:
        return None
    l, t, r, b = box
    return (max(0, l - pad), max(0, t - pad),
            min(mask.size[0], r + pad), min(mask.size[1], b + pad))


def _glove_bbox_render(im):
    """BBox of the crimson procedural glove against the dark studio."""
    r, g, b = im.split()
    # crimson: R clearly above G/B, not the clay body or charcoal beanie.
    cond = ImageChops.subtract(r, ImageChops.add(g, b, scale=2.0, offset=0))
    # threshold-ish: enhance and drop weak pixels
    cond = ImageEnhance.Contrast(cond).enhance(3.0)
    bw = cond.point(lambda p: 255 if p > 18 else 0)
    return _bbox_from_mask(bw, pad=6)


def _glove_bbox_photo(im):
    """BBox of the (typically dark) leather glove on a light / mixed photo."""
    # Work on a luma inverted mask: dark object on lighter ground.
    g = ImageOps.grayscale(im)
    # If the background is already dark, fall back to a deviation mask.
    stats = list(g.getdata())
    mean = sum(stats) / max(1, len(stats))
    if mean > 80:
        bw = g.point(lambda p: 255 if p < mean * 0.78 else 0)
    else:
        # already a dark field — use edges / mid values
        bw = g.point(lambda p: 255 if 18 < p < 200 else 0)
    # keep the largest blob via a coarse erode/dilate
    bw = bw.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5))
    return _bbox_from_mask(bw, pad=8)


def load_ref_same_scale(path, size, glove_close_path):
    """Crop one CC0 glove and scale it so its long axis matches the v3 close-up."""
    if not path or not os.path.isfile(path):
        return load_panel(None, size)
    im = Image.open(path).convert("RGB")
    w, h = im.size
    # Left glove occupies roughly the left half; keep a little margin.
    box = (int(w * 0.02), int(h * 0.08), int(w * 0.52), int(h * 0.98))
    crop = im.crop(box)
    crop = ImageEnhance.Brightness(crop).enhance(1.18)
    crop = ImageEnhance.Contrast(crop).enhance(1.08)

    target_axis = None
    if glove_close_path and os.path.isfile(glove_close_path):
        close = Image.open(glove_close_path).convert("RGB")
        if close.size != size:
            close = close.resize(size, Image.Resampling.LANCZOS)
        rb = _glove_bbox_render(close)
        if rb:
            target_axis = max(rb[2] - rb[0], rb[3] - rb[1])

    pb = _glove_bbox_photo(crop)
    if pb:
        crop = crop.crop(pb)
    src_axis = max(crop.size)
    inner = (size[0] - 28, size[1] - 28)
    if target_axis and src_axis > 8:
        scale = target_axis / src_axis
        nw = max(8, int(round(crop.size[0] * scale)))
        nh = max(8, int(round(crop.size[1] * scale)))
        # Don't overflow the panel; if it does, shrink both equally and note
        # the mismatch is preferrable to clipping the reference.
        fit = min(inner[0] / nw, inner[1] / nh, 1.0)
        if fit < 1.0:
            nw, nh = max(8, int(nw * fit)), max(8, int(nh * fit))
        fitted = crop.resize((nw, nh), Image.Resampling.LANCZOS)
    else:
        fitted = ImageOps.contain(crop, inner, Image.Resampling.LANCZOS)

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
            "X Bot outfit v3  —  plain  |  v2 A  |  v3 A   (crimson gloves + charcoal beanie)",
            font=_font(22, True), fill=FG)
    sub = ("Procedural / CC0 geometry (no third-party meshes).  v3 gloves: fist front ~0.8× head "
           "width, longer padded knuckle, visible attached thumb, same-red 9 cm cuff.  "
           "Beanie: fold-up cuff, rib knit continuing to the crown, slight slouch/fold.  "
           "Same 50 mm camera per row, ≥5% margin.  CC0 crop is scaled to the v3 glove long axis.")
    for i, line in enumerate(_wrap(dr, sub, _font(13), w - 2 * PAD - 16)[:3]):
        dr.text((PAD + 8, 48 + i * 16), line, font=_font(13), fill=MUTED)

    for r, (clip, stage, pose_title) in enumerate(POSES):
        y0 = TITLE_H + r * (LABEL_H + row_body)
        for c, (key, title) in enumerate(VARIANTS):
            x = PAD + c * (col_w + PAD)
            still_key = f"{clip}_{stage}_{key}"
            path = stills.get(still_key)
            if not path:
                path = os.path.join(EVAL, f"xbot_outfit_v3_{clip}_{stage}_{key}.png")
            sheet.paste(load_panel(path, PANEL), (x, y0 + LABEL_H))
            cap = f"{title}    ·    {pose_title}"
            fill = MUTED if key == "plain" else (ACCENT if key == "v2A" else OK)
            dr.text((x + 8, y0 + 12), cap, font=_font(14, True), fill=fill)

    y0 = TITLE_H + rows * (LABEL_H + row_body)
    glove_path = stills.get("close_glove") or os.path.join(EVAL, "xbot_outfit_v3_close_glove.png")
    for c, (key, title) in enumerate(CLOSES):
        x = PAD + c * (col_w + PAD)
        path = stills.get(key)
        if key == "close_ref":
            panel = load_ref_same_scale(path, CLOSE, glove_path)
        else:
            if not path:
                path = os.path.join(EVAL, f"xbot_outfit_v3_{key}.png")
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
    facts_path = os.path.join(EVAL, "xbot_outfit_v3_facts.json")
    if "--facts" in argv:
        facts_path = argv[argv.index("--facts") + 1]
    stills, notes = {}, []
    if os.path.isfile(facts_path):
        with open(facts_path) as fh:
            facts = json.load(fh)
        stills = facts.get("stills") or {}
        notes = facts.get("honest_notes") or []
    out = os.path.join(EVAL, "xbot_outfit_v3.png")
    compose(stills, notes, out)


if __name__ == "__main__":
    main()
