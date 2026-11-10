#!/usr/bin/env python3
"""Build a side-by-side demic vs ours (c026ff5) look-comparison sheet.

Does not render. It only composites stills and demic reference frames that
already live on the box. Demicstory frames are a quality bar, not a source
to copy; this script must not be used to commit those images.

  python3 tools/rig3d/compare_demic_sheet.py
  python3 tools/rig3d/compare_demic_sheet.py --out output/_rig3d/db_row/render/ours_vs_demic_c026ff5.png
"""
from __future__ import annotations

import argparse
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

OURS_PEAK = os.path.join(ROOT, "output/_rig3d/db_row/render/f_0025.png")
# Largest demic bench-row still already on the box (left panel of an earlier
# look sheet). Source of peach34_air/ours_vs_demic_peak.png.
DEMIC_SHEET = os.path.join(ROOT, "output/_rig3d/v4_authored/peach34_air/ours_vs_demic_peak.png")

# Pixel boxes in the original frames (left, top, right, bottom).
# ours f_0025 is 1080x1920. Demic panel extracted below is 424x720.
OURS_HIPS = (20, 620, 380, 1140)
OURS_BACK = (160, 430, 780, 880)
DEMIC_PANEL = (58, 84, 482, 804)
DEMIC_HIPS = (40, 280, 400, 700)   # coords inside the extracted demic panel
DEMIC_BACK = (40, 40, 390, 360)

BG = (18, 18, 20)
CARD = (28, 28, 32)
FG = (236, 236, 240)
MUTED = (168, 168, 174)
ACCENT = (232, 196, 120)
RULE = (52, 52, 58)


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = f"/usr/share/fonts/truetype/dejavu/{name}"
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit(im, w, h):
    im = im.convert("RGB")
    scale = min(w / im.size[0], h / im.size[1])
    nw, nh = max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _cover(im, w, h):
    im = im.convert("RGB")
    scale = max(w / im.size[0], h / im.size[1])
    nw, nh = max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    x = (nw - w) // 2
    y = (nh - h) // 2
    return im.crop((x, y, x + w, y + h))


def _panel(im, w, h, label, caption, fonts):
    """Image card with a caption bar under the picture."""
    cap_h = 110
    card = Image.new("RGB", (w, h + cap_h), CARD)
    fitted = _fit(im, w - 8, h - 8)
    x = (w - fitted.size[0]) // 2
    y = (h - fitted.size[1]) // 2
    card.paste(fitted, (x, y))
    d = ImageDraw.Draw(card)
    d.rectangle((0, h, w, h + cap_h), fill=(24, 24, 26))
    d.text((12, h + 8), label, font=fonts["label"], fill=ACCENT)
    # wrap caption
    y = h + 30
    for line in textwrap.wrap(caption, width=52):
        d.text((12, y), line, font=fonts["small"], fill=MUTED)
        y += 16
        if y > h + cap_h - 16:
            break
    return card


def _text_block(draw, xy, lines, fonts, width):
    x, y = xy
    for kind, text in lines:
        if kind == "h":
            draw.text((x, y), text, font=fonts["h"], fill=FG)
            y += 28
        elif kind == "sub":
            for line in textwrap.wrap(text, width=width):
                draw.text((x, y), line, font=fonts["small"], fill=MUTED)
                y += 18
            y += 6
        elif kind == "n":
            num, rest = text.split(" ", 1)
            draw.text((x, y), num, font=fonts["label"], fill=ACCENT)
            for i, line in enumerate(textwrap.wrap(rest, width=width - 4)):
                draw.text((x + 28, y), line, font=fonts["body"], fill=FG)
                y += 20
            y += 8
        else:
            for line in textwrap.wrap(text, width=width):
                draw.text((x, y), line, font=fonts["body"], fill=FG)
                y += 20
            y += 4
    return y


def extract_demic_panel(path):
    im = Image.open(path)
    return im.crop(DEMIC_PANEL)


def build(ours_path, demic_sheet, out_path):
    if not os.path.isfile(ours_path):
        raise SystemExit(f"missing ours still: {ours_path}")
    if not os.path.isfile(demic_sheet):
        raise SystemExit(f"missing demic sheet: {demic_sheet}")

    ours = Image.open(ours_path).convert("RGB")
    demic = extract_demic_panel(demic_sheet)

    fonts = {
        "title": _font(28, True),
        "h": _font(18, True),
        "label": _font(14, True),
        "body": _font(14, False),
        "small": _font(13, False),
    }

    ours_hips = ours.crop(OURS_HIPS)
    ours_back = ours.crop(OURS_BACK)
    demic_hips = demic.crop(DEMIC_HIPS)
    demic_back = demic.crop(DEMIC_BACK)

    # Magnify crops ~3x relative to how they appear in a 720px-tall full-body.
    hips_w, hips_h = 520, 640
    back_w, back_h = 520, 400
    full_w, full_h = 420, 720

    pad = 24
    gutter = 16
    header_h = 96
    row1_h = full_h + 110
    row2_h = hips_h + 110
    row3_h = back_h + 110
    list_h = 440
    footer_h = 36
    width = pad + full_w + gutter + full_w + pad
    # keep width at least wide enough for two 520 crops
    width = max(width, pad + hips_w + gutter + hips_w + pad)
    height = header_h + row1_h + gutter + row2_h + gutter + row3_h + gutter + list_h + footer_h + pad

    sheet = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 18), "ours (kit)  vs  demic   ·   bench one-arm row stills",
           font=fonts["title"], fill=FG)
    sub = ("Look comparison only — cameras differ (demic: elevated back / barbell; "
           "ours: fixed head-end 3/4 beauty). Demicstory is the quality bar, not a source to copy.")
    yy = 52
    for line in textwrap.wrap(sub, width=110):
        d.text((pad, yy), line, font=fonts["small"], fill=MUTED)
        yy += 16

    y = header_h
    x0 = pad
    x1 = pad + full_w + gutter
    # if width grew, center the pair
    pair_w = full_w + gutter + full_w
    x0 = (width - pair_w) // 2
    x1 = x0 + full_w + gutter

    p = _panel(demic, full_w, full_h,
               "demic  ·  full body",
               "Porcelain clay, luminous coral lat sheet with a pale halo, "
               "athletic shorts and a wide waistband, smooth cloth over the seat.",
               fonts)
    sheet.paste(p, (x0, y))
    p = _panel(ours, full_w, full_h,
               "ours kit  ·  full body  ·  f_0025",
               "Warmer grey clay, duller cracked glow, shrink-wrapped briefs "
               "with no real waistband, simpler bench and dumbbell.",
               fonts)
    sheet.paste(p, (x1, y))

    y += row1_h + gutter
    pair_w = hips_w + gutter + hips_w
    x0 = (width - pair_w) // 2
    x1 = x0 + hips_w + gutter
    p = _panel(demic_hips, hips_w, hips_h,
               "demic  ·  hips  3×",
               "Wide elastic waistband (do not copy the DEMIC mark), mid-thigh hem, "
               "smooth matte cloth that bridges the cleft. Slight fabric volume, no clay showing through.",
               fonts)
    sheet.paste(p, (x0, y))
    p = _panel(ours_hips, hips_w, hips_h,
               "ours kit  ·  hips  3×",
               "Offset of the Grab3D pelvis: every glute groove reads as cloth. "
               "No waistband, high-cut hem, pale clay spot at the seat/cleft, clay peeking at the waist.",
               fonts)
    sheet.paste(p, (x1, y))

    y += row2_h + gutter
    pair_w = back_w + gutter + back_w
    x0 = (width - pair_w) // 2
    x1 = x0 + back_w + gutter
    p = _panel(demic_back, back_w, back_h,
               "demic  ·  back glow  3×",
               "Saturated coral-orange, painted fibre strokes, soft airbrushed edge "
               "and a pale bloom. No crack lines. Body tone stays cool porcelain.",
               fonts)
    sheet.paste(p, (x0, y))
    p = _panel(ours_back, back_w, back_h,
               "ours kit  ·  back glow  3×",
               "Lower-value salmon (glow V ~0.74 vs demic ~0.95) because colour is "
               "multiplied onto sculpted clay. UV/groove valleys read as dark cracks.",
               fonts)
    sheet.paste(p, (x1, y))

    y += row3_h + gutter
    d.rectangle((pad, y, width - pad, y + list_h), fill=CARD)
    lines = [
        ("h", "Ranked improvements  ·  fix the base assets once, do not patch the current methods"),
        ("sub", "Look defects should not be caught by gates after the fact. They should not exist, because the kit starts from well-made meshes and a studio light rig. See docs/specs/ROOTCAUSE_LOOK.md."),
        ("n", "1  Drawers — author a real boxer-brief garment (waistband, crotch gusset, hem to upper-mid thigh) that bridges the gluteal cleft. This removes the white spot, the shrink-wrap grooves, the missing waistband and the short cut, by construction."),
        ("n", "2  Glow — stop projecting paint onto the Grab3D single mesh. Overlay separate seamless muscle meshes (BodyParts3D CC BY 4.0, or four shells we sculpt ourselves as CC0) with opaque emissive materials. This removes the cracks and the pale multiply-on-clay look together."),
        ("n", "3  Light — a studio HDRI (Poly Haven, CC0) plus an authored three-point area-light rig and a floor that holds contact shadows, so the clay reads porcelain and the glow can bloom like demic."),
        ("n", "4  Equipment — a padded commercial flat bench (black pad, gunmetal frame) and a dumbbell with a knurled handle. The current primitives read as placeholders next to demic."),
    ]
    _text_block(d, (pad + 18, y + 16), lines, fonts, width=84)

    d.text((pad, height - 28),
           "Refs stay on the box and are not committed.  Sheet code: tools/rig3d/compare_demic_sheet.py",
           font=fonts["small"], fill=MUTED)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path, optimize=True)
    print(out_path, sheet.size)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", default=OURS_PEAK)
    ap.add_argument("--demic-sheet", default=DEMIC_SHEET)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "output/_rig3d/db_row/render/ours_vs_demic_c026ff5.png"))
    args = ap.parse_args()
    build(args.ours, args.demic_sheet, args.out)


if __name__ == "__main__":
    main()
