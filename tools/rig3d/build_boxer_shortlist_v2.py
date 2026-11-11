#!/usr/bin/env python3
"""Compose boxer_shortlist_v2.png from already-downloaded public thumbnails."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

THUMB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output/_basemodels/boxer_thumbs",
)
OUT = os.path.join(THUMB_DIR, "boxer_shortlist_v2.png")

BG = (18, 18, 20)
FG = (230, 230, 232)
MUTED = (160, 160, 166)
ACCENT = (220, 200, 140)
FAIL = (220, 110, 96)
OK = (140, 196, 140)
CARD = (30, 30, 34)
CELL = (280, 210)
CAP_H = 78
PAD = 16

# (thumb filename, title, license, notes, status)
# status: lead | tested | review | exclude
ENTRIES = [
    ("01_tishkok_base.jpg",
     "Tishkok · Stylized Boxer Base",
     "CC BY 4.0",
     "Still the look leader. Unrigged A-pose, sculpted gloves/shorts/shoes, simplified face. ~83k tris.",
     "lead"),
    ("02_tishkok_paint.jpg",
     "Tishkok · Boxing Character Handpainted",
     "CC BY 4.0 · rigged",
     "Same family, already rigged, 9.7k faces. Painted low-poly does not beat the clay base look.",
     "review"),
    ("03_quaternius.jpg",
     "Quaternius · Superhero Male (Standard)",
     "CC0 · rigged",
     "Already jab-tested (v1). Throws a punch, not a boxer. Open-hand UAL. Motion guinea pig only.",
     "tested"),
    ("04_kickboxer.jpg",
     "mr.turkish34 · Kickboxer (Mixamo file)",
     "CC BY · rigged",
     "Low-poly kickboxer, already on v1 list. Not a cleaner boxing silhouette than Tishkok.",
     "review"),
    ("v2_ece056b61bc946a78c7a6f4490b3f8bb.jpg",
     "AhmedNasr55 · Fighter",
     "CC BY · unrigged",
     "Stylized A-pose athlete, wrist wraps not gloves, cartoon face. Not a boxer body.",
     "review"),
    ("v2_92e1999d04624ecf910d60a0659edee3.jpg",
     "Nautical10 · Male base mesh",
     "CC BY · unrigged",
     "T-pose underwear basemesh, faced. Not a boxer. No gloves/kit.",
     "review"),
    ("v2_7e7958f2e8c24f70a7ccb035336df201.jpg",
     "pavelsirucek · Male basemeshes",
     "CC BY · unrigged",
     "Four nude body types. Useful as anatomy refs, not a boxing character.",
     "review"),
    ("v2_d6e40d149e48497ea61b0b8dc262aeb9.jpg",
     "Nimmerschnell · LowPoly Person Boxing",
     "CC BY · rigged",
     "1.5k-face wooden mannequin. Too crude for production stills.",
     "review"),
    ("v2_3fc27d11568541ebab448c32efa4d8fa.jpg",
     "emelyarules · Boxer Cartoon 1",
     "CC BY · unrigged",
     "Cartoon kid boxer. Wrong style and age read.",
     "review"),
    ("v2_1b562d3662c64ab7a963639f2e465e19.jpg",
     "sihagames · Brawler",
     "CC BY · unrigged",
     "Tank top + jeans, Makehuman. Not a boxer.",
     "review"),
    ("v2_8ba28f9b40be4cee84f229624348568d.jpg",
     "MINDSFunGames · Boxer",
     "CC BY · EXCLUDE",
     "Tagged mannypacquiao — real-person likeness. Out.",
     "exclude"),
    ("v2_766c3ddd41b94cfba2f59b656bff357b.jpg",
     "johnny136 · boxer",
     "CC BY · EXCLUDE",
     "Arcade/street-fighter lookalike, 602k faces. IP risk. Out.",
     "exclude"),
    ("v2_a8c6d71658cc4f6692e5b0db7466102b.jpg",
     "FedeOde · MMA Figther",
     "CC BY · EXCLUDE",
     "Female MMA wearing UFC branding. Trademark/IP. Out.",
     "exclude"),
    ("v2_b16a527e908e4fc7b501be6aa7175564.jpg",
     "bertil · Boxer 1",
     "CC BY · EXCLUDE",
     "CrazyTalk head from one photo + Mixamo. Real-person likeness risk, nude. Out.",
     "exclude"),
    ("v2_127cc242790c422fb9599f774b1c9642.jpg",
     "jungle_jim · Mr Sexy Man",
     "CC BY · EXCLUDE",
     "NSFW-leaning fighter, heels, not a boxer. Taste/IP-adjacent. Out.",
     "exclude"),
    ("v2_51edfec934004084b0ec3ec13e0b87bc.jpg",
     "jerlymans · Boxer 3D Human Riged",
     "CC BY-SA · ref only",
     "Looks like a heavy boxer and is rigged, but BY-SA (share-alike) — internal ref only, not shipping.",
     "exclude"),
    ("v2_7fcc8f0bc8114a3585f46c785a54a1b0.jpg",
     "rhitemax · Man Body Standing Base",
     "CC BY · unrigged",
     "396k-face 3D-print figure. Faceless idea is right; mesh/pose/thumb are not a boxer.",
     "review"),
]


def _font(size, bold=False):
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


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


def fit_thumb(path, size):
    canvas = Image.new("RGB", size, (40, 40, 44))
    if not path or not os.path.isfile(path):
        return canvas
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - im.size[0]) // 2
    y = (size[1] - im.size[1]) // 2
    canvas.paste(im, (x, y))
    return canvas


def main():
    cols = 4
    rows = (len(ENTRIES) + cols - 1) // cols
    header_h = 150
    footer_h = 110
    cell_w, cell_h = CELL[0], CELL[1] + CAP_H
    w = PAD + cols * (cell_w + PAD)
    h = header_h + rows * (cell_h + PAD) + footer_h
    im = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(im)
    title_f = _font(26, True)
    sub_f = _font(14)
    name_f = _font(13, True)
    small = _font(11)

    dr.text((PAD, 18),
            "Boxer body shortlist v2   ·   free commercial only",
            font=title_f, fill=FG)
    sub = (
        "Sources: Sketchfab public API (CC Attribution / CC0), OpenGameArt, itch listings, "
        "Quaternius, Kenney (no human boxer), Blend Swap (search page blocked). "
        "Rules: no NC, no NoAI, no real-person / game / film IP, no paid assets."
    )
    y = 56
    for line in _wrap(dr, sub, sub_f, w - 2 * PAD):
        dr.text((PAD, y), line, font=sub_f, fill=MUTED)
        y += 20
    verdict = (
        "Verdict: nothing clearly beats Tishkok’s lean boxer silhouette under those rules. "
        "OGA “Boxer Game Character” / “Mini Boxing Character” are 2D sprite sheets (CC0), not 3D bodies. "
        "No extra full-body download this pass."
    )
    for line in _wrap(dr, verdict, sub_f, w - 2 * PAD):
        dr.text((PAD, y), line, font=sub_f, fill=ACCENT)
        y += 20

    status_col = {
        "lead": OK,
        "tested": (180, 180, 120),
        "review": MUTED,
        "exclude": FAIL,
    }

    for i, (fn, title, lic, notes, status) in enumerate(ENTRIES):
        r, c = divmod(i, cols)
        x = PAD + c * (cell_w + PAD)
        y0 = header_h + r * (cell_h + PAD)
        card = Image.new("RGB", (cell_w, cell_h), CARD)
        cd = ImageDraw.Draw(card)
        thumb = fit_thumb(os.path.join(THUMB_DIR, fn), CELL)
        card.paste(thumb, (0, 0))
        # status bar
        cd.rectangle((0, CELL[1] - 4, cell_w, CELL[1]), fill=status_col[status])
        cd.text((8, CELL[1] + 6), title, font=name_f, fill=FG)
        cd.text((8, CELL[1] + 24), f"{lic}   ·   {status}", font=small,
                fill=status_col[status])
        ty = CELL[1] + 40
        for line in _wrap(cd, notes, small, cell_w - 16)[:3]:
            cd.text((8, ty), line, font=small, fill=MUTED)
            ty += 13
        im.paste(card, (x, y0))

    fy = h - footer_h + 12
    footer = (
        "Legend: green = current lead   yellow = already motion-tested   grey = reviewed, does not beat Tishkok   "
        "red = excluded (IP / likeness / NSFW / BY-SA).\n"
        "itch “Sonny The Boxer! 3D Character” and pixel/Spine packs skipped (2D or not confirmed free-commercial 3D). "
        "Poly Pizza search API requires a key — not used.\n"
        "A faceless, already-rigged, lean athletic boxer on a single surface is still the missing piece. "
        "jerlymans is the closest look among rejects, blocked by CC BY-SA."
    )
    dr.multiline_text((PAD, fy), footer, font=small, fill=MUTED, spacing=4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    im.save(OUT, "PNG")
    print("wrote", OUT, im.size)


if __name__ == "__main__":
    main()
