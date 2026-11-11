#!/usr/bin/env python3
"""Compose xbot_outfit_v5.png from rendered stills. System Python (PIL).

  python3 tools/rig3d/build_xbot_outfit_sheet_v5.py
  python3 tools/rig3d/build_xbot_outfit_sheet_v5.py --facts path/to/facts.json
"""
from __future__ import annotations

import json
import os
import sys

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps, ImageFilter

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
TITLE_H = 108
FOOT_H = 140

VARIANTS = (
    ("plain", "plain X Bot"),
    ("v4A", "v4 A  ·  ballooned trunks / crop tank"),
    ("v5A", "v5 A  ·  burgundy trunks, charcoal tank, kidney thumb"),
)
POSES = (
    ("idle", "guard", "Fighting Idle — guard"),
    ("jab", "full", "Lead Jab — full extension"),
    ("jab", "return", "Lead Jab — return"),
)
# Two close-up rows, three columns each.
CLOSE_ROWS = (
    (
        ("close_glove", "v5 glove close  ·  same studio lighting, no grey card"),
        ("close_ref", "CC0 glove ref  ·  same long-axis scale (Wikimedia, Petey21, 2014)"),
        ("close_garments", "v5 tank + trunks  ·  hem covers waistband, delts open"),
    ),
    (
        ("close_trunks", "v5 trunks close  ·  burgundy satin, snug navel band, mid-thigh flare"),
        ("close_trunks_ref", "CC0 trunks ref  ·  same scale (Nationaal Archief / Anefo, 1951)"),
        ("close_garments", "v5 garments (repeat)  ·  hip rotation still readable"),
    ),
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


def _glove_bbox_dark_on_light(im):
    g = ImageOps.grayscale(im)
    stats = list(g.getdata())
    mean = sum(stats) / max(1, len(stats))
    if mean > 70:
        bw = g.point(lambda p: 255 if p < mean * 0.78 else 0)
    else:
        bw = g.point(lambda p: 255 if 18 < p < 200 else 0)
    bw = bw.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5))
    return _bbox_from_mask(bw, pad=8)


def _glove_bbox_on_studio(im):
    """BBox of a dark glossy glove on the charcoal studio (no pale card)."""
    g = ImageOps.grayscale(im)
    # Studio is very dark; the glove is slightly lighter with highlights.
    # Use deviation from the mode-dark background.
    stats = list(g.getdata())
    # background = darkest 15% mean
    dark = sorted(stats)[: max(1, len(stats) // 8)]
    bg = sum(dark) / len(dark)
    bw = g.point(lambda p: 255 if p > bg + 14 else 0)
    bw = bw.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    return _bbox_from_mask(bw, pad=8) or _glove_bbox_dark_on_light(im)


def _fit_crop_to_axis(crop, size, target_axis):
    src_axis = max(crop.size)
    inner = (size[0] - 28, size[1] - 28)
    if target_axis and src_axis > 8:
        scale = target_axis / src_axis
        nw = max(8, int(round(crop.size[0] * scale)))
        nh = max(8, int(round(crop.size[1] * scale)))
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


def load_glove_ref_same_scale(path, size, glove_close_path):
    if not path or not os.path.isfile(path):
        return load_panel(None, size)
    im = Image.open(path).convert("RGB")
    w, h = im.size
    box = (int(w * 0.02), int(h * 0.08), int(w * 0.52), int(h * 0.98))
    crop = im.crop(box)
    crop = ImageEnhance.Brightness(crop).enhance(1.18)
    crop = ImageEnhance.Contrast(crop).enhance(1.08)
    target_axis = None
    if glove_close_path and os.path.isfile(glove_close_path):
        close = Image.open(glove_close_path).convert("RGB")
        if close.size != size:
            close = close.resize(size, Image.Resampling.LANCZOS)
        rb = _glove_bbox_on_studio(close) or _glove_bbox_dark_on_light(close)
        if rb:
            target_axis = max(rb[2] - rb[0], rb[3] - rb[1])
    pb = _glove_bbox_dark_on_light(crop)
    if pb:
        crop = crop.crop(pb)
    return _fit_crop_to_axis(crop, size, target_axis)


def load_trunks_ref_same_scale(path, size, trunks_close_path):
    """Crop the worn trunks from the CC0 Anefo photo and match the close-up long axis."""
    if not path or not os.path.isfile(path):
        return load_panel(None, size)
    im = Image.open(path).convert("RGB")
    w, h = im.size
    # Boxer stands on the right; trunks sit in the lower-right third.
    box = (int(w * 0.46), int(h * 0.50), int(w * 0.90), int(h * 0.96))
    crop = im.crop(box)
    crop = ImageEnhance.Contrast(crop).enhance(1.12)
    # Isolate the dark shorts against the tiled wall / skin.
    g = ImageOps.grayscale(crop)
    bw = g.point(lambda p: 255 if p < 70 else 0)
    bw = bw.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(7))
    pb = _bbox_from_mask(bw, pad=10)
    if pb:
        crop = crop.crop(pb)
    target_axis = None
    if trunks_close_path and os.path.isfile(trunks_close_path):
        close = Image.open(trunks_close_path).convert("RGB")
        if close.size != size:
            close = close.resize(size, Image.Resampling.LANCZOS)
        # Trunks are mid-value charcoal on a dark studio — use a mid-band mask.
        cg = ImageOps.grayscale(close)
        cbw = cg.point(lambda p: 255 if 22 < p < 140 else 0)
        cbw = cbw.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
        rb = _bbox_from_mask(cbw, pad=8)
        if rb:
            target_axis = max(rb[2] - rb[0], rb[3] - rb[1])
    return _fit_crop_to_axis(crop, size, target_axis)


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
    close_rows = len(CLOSE_ROWS)
    col_w, row_body = PANEL
    close_h = CLOSE[1]
    w = PAD * (cols + 1) + col_w * cols
    h = (TITLE_H + (LABEL_H + row_body) * rows
         + (LABEL_H + close_h) * close_rows + FOOT_H + PAD)
    sheet = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 8, 14),
            "X Bot outfit v5  —  plain  |  v4 A  |  v5 A   (burgundy trunks + charcoal tank + glossy black gloves)",
            font=_font(18, True), fill=FG)
    sub = ("Procedural / CC0 geometry (no third-party meshes, no PRO lettering).  "
           "v5 trunks: deep burgundy/wine satin (distinct from purple/gold jab glow), snug navel waistband, fabric follows hips/glutes, gentle flare, "
           "leg opening ~3.5 cm off the thigh, clean hem.  "
           "Tank hem covers the waistband top; armholes stay clear of the delts.  "
           "Glove close-up uses the same studio lighting (no pale card).  "
           "CC0 glove: Wikimedia / Petey21 2014.  CC0 trunks: Nationaal Archief / Anefo, Harry Pot, 1951.")
    for i, line in enumerate(_wrap(dr, sub, _font(13), w - 2 * PAD - 16)[:4]):
        dr.text((PAD + 8, 46 + i * 16), line, font=_font(13), fill=MUTED)

    for r, (clip, stage, pose_title) in enumerate(POSES):
        y0 = TITLE_H + r * (LABEL_H + row_body)
        for c, (key, title) in enumerate(VARIANTS):
            x = PAD + c * (col_w + PAD)
            still_key = f"{clip}_{stage}_{key}"
            path = stills.get(still_key)
            if not path:
                path = os.path.join(EVAL, f"xbot_outfit_v5_{clip}_{stage}_{key}.png")
            sheet.paste(load_panel(path, PANEL), (x, y0 + LABEL_H))
            cap = f"{title}    ·    {pose_title}"
            fill = MUTED if key == "plain" else (ACCENT if key == "v4A" else OK)
            dr.text((x + 8, y0 + 12), cap, font=_font(13, True), fill=fill)

    glove_path = stills.get("close_glove") or os.path.join(EVAL, "xbot_outfit_v5_close_glove.png")
    trunks_path = stills.get("close_trunks") or os.path.join(EVAL, "xbot_outfit_v5_close_trunks.png")
    for cr, entries in enumerate(CLOSE_ROWS):
        y0 = TITLE_H + rows * (LABEL_H + row_body) + cr * (LABEL_H + close_h)
        for c, (key, title) in enumerate(entries):
            x = PAD + c * (col_w + PAD)
            path = stills.get(key)
            if key == "close_ref":
                panel = load_glove_ref_same_scale(path, CLOSE, glove_path)
            elif key == "close_trunks_ref":
                panel = load_trunks_ref_same_scale(path, CLOSE, trunks_path)
            else:
                if not path:
                    path = os.path.join(EVAL, f"xbot_outfit_v5_{key}.png")
                panel = load_panel(path, CLOSE)
            sheet.paste(panel, (x, y0 + LABEL_H))
            fill = MUTED if key.endswith("ref") else OK
            dr.text((x + 8, y0 + 10), title, font=_font(11, True), fill=fill)

    foot = "  |  ".join(notes) if notes else "See facts JSON for fit notes."
    fw = _font(12)
    lines = _wrap(dr, foot, fw, w - 2 * PAD - 8)
    fy = TITLE_H + (LABEL_H + row_body) * rows + (LABEL_H + close_h) * close_rows + 16
    for i, line in enumerate(lines[:5]):
        dr.text((PAD + 8, fy + i * 20), line, font=fw, fill=MUTED)

    sheet.save(out_path, "PNG", optimize=True)
    print("WROTE", out_path, sheet.size, os.path.getsize(out_path))
    return out_path


def main():
    argv = sys.argv[1:]
    facts_path = os.path.join(EVAL, "xbot_outfit_v5_facts.json")
    if "--facts" in argv:
        facts_path = argv[argv.index("--facts") + 1]
    stills, notes = {}, []
    if os.path.isfile(facts_path):
        with open(facts_path) as fh:
            facts = json.load(fh)
        stills = facts.get("stills") or {}
        notes = facts.get("honest_notes") or []
    out = os.path.join(EVAL, "xbot_outfit_v5.png")
    compose(stills, notes, out)


if __name__ == "__main__":
    main()
