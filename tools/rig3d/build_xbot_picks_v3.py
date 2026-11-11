#!/usr/bin/env python3
"""Compose xbot_boxing_picks_v3.png from winner stills + runner-up thumbs."""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(ROOT, "output/_basemodels/eval")
FACTS = os.path.join(EVAL, "xbot_punch_pick_facts_v3.json")
OUT = os.path.join(EVAL, "xbot_boxing_picks_v3.png")

PANEL = (540, 960)
THUMB = (168, 300)
BG = (16, 16, 18)
FG = (232, 232, 234)
MUTED = (156, 156, 162)
ACCENT = (220, 200, 140)
OK = (120, 210, 150)
PAD = 12
TITLE_H = 96
LABEL_H = 56
THUMB_H = 300 + 52
FOOT_H = 92
ORDER = ("jab", "cross", "hook", "uppercut")
STAGE = ("guard", "mid", "full")
TYPE_LABEL = {
    "jab": "JAB",
    "cross": "CROSS",
    "hook": "LEAD HOOK",
    "uppercut": "UPPERCUT",
}


def _font(size, bold=False):
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit(im, size, bg=BG):
    canvas = Image.new("RGB", size, bg)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im = im.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - im.size[0]) // 2
    y = (size[1] - im.size[1]) // 2
    canvas.paste(im, (x, y))
    return canvas


def load(path, size):
    if path and os.path.isfile(path):
        return _fit(Image.open(path).convert("RGB"), size)
    im = Image.new("RGB", size, (40, 28, 28))
    ImageDraw.Draw(im).text((16, 24), "missing", font=_font(16, True), fill=(220, 110, 96))
    return im


def punch_by_id(facts, pid):
    for p in facts.get("punches", []):
        if p.get("id") == pid:
            return p
    return {}


def metrics_line(p):
    if not p:
        return "—"
    bits = [
        f"{p.get('side','?')} hand",
        f"elbow {p.get('elbow_deg',0):.0f}°",
        f"fist vs head {p.get('fist_vs_head_m',0)*100:+.0f} cm",
        f"opp-guard {p.get('opp_hand_to_head_m',0)*100:.0f} cm",
        f"hip {p.get('hip_rot_deg',0):.0f}°",
        f"drift {p.get('root_drift_m',0)*100:.0f} cm",
        f"slide {max(p.get('foot_slide_l_m',0), p.get('foot_slide_r_m',0))*100:.0f} cm",
    ]
    if p.get("arm_elev_deg") is not None:
        bits.insert(2, f"arm elev {p['arm_elev_deg']:+.0f}°")
    return "  ·  ".join(bits)


def main():
    facts = json.load(open(FACTS))
    winners = facts["winners"]
    runners = facts.get("runner_ups") or {}
    col_w = PANEL[0]
    row_h = LABEL_H + PANEL[1] + 10 + THUMB_H
    w = PAD * 4 + col_w * 3
    h = TITLE_H + row_h * 4 + FOOT_H
    sheet = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 8, 16), "X Bot boxing picks v3  —  textbook form over muscle detail",
            font=_font(26, True), fill=FG)
    dr.text((PAD + 8, 54),
            "Winners: visual review + measurements (extension, chin/face-height fist, rear-hand guard, "
            "hip/shoulder rotation, rear-foot pivot, 90° horizontal hook, short uppercut).  "
            "Runner-up thumbs are EEVEE contact frames.",
            font=_font(13), fill=MUTED)

    for ri, ptype in enumerate(ORDER):
        pid = winners.get(ptype)
        p = punch_by_id(facts, pid)
        y0 = TITLE_H + ri * row_h
        title = f"{TYPE_LABEL[ptype]}   winner: {p.get('clip_title','?')}"
        dr.text((PAD + 8, y0 + 6), title, font=_font(18, True), fill=OK)
        dr.text((PAD + 8, y0 + 30), metrics_line(p), font=_font(13), fill=ACCENT)
        for ci, stage in enumerate(STAGE):
            x = PAD + ci * (col_w + PAD)
            path = os.path.join(EVAL, f"xbot_v3_{ptype}_{stage}.png")
            panel = load(path, PANEL)
            sheet.paste(panel, (x, y0 + LABEL_H))
            cap = f"{stage}   f{p.get(stage,'?')}"
            # overlay a small caption on the bottom of the panel area via extra text under? 
            # actually write just above panel bottom by a bar - skip, header covers it
            dr.text((x + 10, y0 + LABEL_H + PANEL[1] - 22), cap,
                    font=_font(14, True), fill=(210, 210, 214))

        # runner-up strip
        ty = y0 + LABEL_H + PANEL[1] + 8
        dr.text((PAD + 8, ty), "runner-ups", font=_font(13, True), fill=MUTED)
        rx = PAD + 110
        for rid in runners.get(ptype, []):
            rp = punch_by_id(facts, rid)
            thumb = rp.get("thumb") or os.path.join(EVAL, "xbot_picks_thumbs", rid + ".png")
            tim = load(thumb, THUMB)
            sheet.paste(tim, (rx, ty + 18))
            label = rp.get("clip_title") or rid
            # compact two-line caption
            fnt = _font(11)
            dr.text((rx, ty + 18 + THUMB[1] + 2), label[:28], font=fnt, fill=FG)
            sub = (f"el {rp.get('elbow_deg',0):.0f}°  "
                   f"h {rp.get('fist_vs_head_m',0)*100:+.0f}cm")
            dr.text((rx, ty + 18 + THUMB[1] + 16), sub, font=fnt, fill=MUTED)
            rx += THUMB[0] + 16

    foot = (
        "Still short of textbook: jab elbow 161° (not a locked 170–175°) and is extracted from a combo, not a dedicated single; "
        "cross rear foot slides ~19 cm while pivoting; hook lead foot slides ~27 cm and hip turn is only ~24°; "
        "uppercut finishes ~26 cm below the head bone (low chin / high chest) with a short 4-frame window. "
        "Current Lead Jab drops to body height (−38 cm). Current Uppercut over-rotates and drops the rear guard."
    )
    # wrap footer
    fnt = _font(13)
    words = foot.split()
    lines, cur = [], ""
    max_w = w - 2 * PAD - 16
    for word in words:
        t = (cur + " " + word).strip()
        if dr.textlength(t, font=fnt) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    fy = TITLE_H + row_h * 4 + 16
    for i, line in enumerate(lines):
        dr.text((PAD + 8, fy + i * 18), line, font=fnt, fill=MUTED)

    sheet.save(OUT, "PNG", optimize=True)
    print("WROTE", OUT, sheet.size, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
