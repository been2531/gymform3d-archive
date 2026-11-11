#!/usr/bin/env python3
"""Compose a 2-body x 3-shot jab comparison sheet plus a Wikimedia reference."""
from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

PANEL = (540, 960)
BG = (18, 18, 20)
FG = (230, 230, 232)
MUTED = (160, 160, 166)
ACCENT = (220, 200, 140)
FAIL = (220, 110, 96)
CARD = (32, 28, 30)


def _font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
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


def fail_card(size, title, body):
    im = Image.new("RGB", size, CARD)
    dr = ImageDraw.Draw(im)
    title_f = _font(22, True)
    body_f = _font(16)
    dr.text((36, 80), title, font=title_f, fill=FAIL)
    y = 140
    for line in _wrap(dr, body, body_f, size[0] - 72):
        dr.text((36, y), line, font=body_f, fill=MUTED)
        y += 28
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", default="boxer_jab")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tishkok-rest", default="")
    ap.add_argument("--facts", default="")
    args = ap.parse_args()
    facts = {}
    if args.facts and os.path.isfile(args.facts):
        facts = json.load(open(args.facts))

    cols = [("Guard", "guard"), ("Mid-extension", "mid"), ("Full extension", "full")]
    pad = 18
    header_h = 96
    label_h = 42
    row_h = 40
    ref_h = 560
    footer_h = 130
    w = pad + 3 * (PANEL[0] + pad)
    h = header_h + 2 * (row_h + label_h + PANEL[1]) + ref_h + footer_h + pad
    im = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(im)
    title_f = _font(26, True)
    sub_f = _font(15)
    lab_f = _font(18, True)
    small = _font(14)

    punch = facts.get("punch_side", "?")
    keys = facts.get("keyframes", {})
    title = "Boxer jab test   ·   UAL Punch_Jab (CC0)   ·   clay, fixed 3/4 front"
    dr.text((pad, 20), title, font=title_f, fill=FG)
    sub = (f"Punching side: {punch}    "
           f"frames guard/mid/full = {keys.get('guard')} / {keys.get('mid')} / {keys.get('full')}    "
           f"540×960 Cycles  ·  dark charcoal studio")
    dr.text((pad, 58), sub, font=sub_f, fill=MUTED)

    y = header_h

    # --- Tishkok row ---
    dr.text((pad, y + 8),
            "01  Tishkok  ·  Stylized Boxer Base Mesh  ·  CC BY 4.0  ·  unrigged A-pose",
            font=lab_f, fill=ACCENT)
    y += row_h
    tish_labels = ["Rest pose (A-pose)", "Jab bind", "Jab bind"]
    for i, lab in enumerate(tish_labels):
        x = pad + i * (PANEL[0] + pad)
        dr.text((x, y), lab, font=sub_f, fill=MUTED)
    y += label_h
    rest = args.tishkok_rest
    if rest and os.path.isfile(rest):
        p0 = _fit(Image.open(rest), PANEL)
    else:
        p0 = fail_card(PANEL, "No rest still", "Missing --tishkok-rest")
    im.paste(p0, (pad, y))
    reason = ("Weight transfer from the T-pose Quaternius Superhero tore this mesh. "
              "Tishkok is an A-pose base with sculpted gloves, shorts and shoes; "
              "the donor arms do not overlap. Automatic weights onto a T-pose "
              "armature would fail for the same reason. Rigify would need the "
              "metarig folded to this A-pose first — not done this pass. "
              "Do not read the exploded jab stills as the body.")
    fail = fail_card(PANEL, "FAILED", reason)
    im.paste(fail, (pad + (PANEL[0] + pad), y))
    im.paste(fail, (pad + 2 * (PANEL[0] + pad), y))
    y += PANEL[1]

    # --- Quaternius row ---
    dr.text((pad, y + 8),
            "03  Quaternius  ·  Superhero Male (Standard)  ·  CC0  ·  65-bone UE humanoid",
            font=lab_f, fill=ACCENT)
    y += row_h
    for i, (clab, ckey) in enumerate(cols):
        x = pad + i * (PANEL[0] + pad)
        dr.text((x, y), clab, font=sub_f, fill=MUTED)
    y += label_h
    for i, (clab, ckey) in enumerate(cols):
        x = pad + i * (PANEL[0] + pad)
        path = os.path.join(args.dir, f"{args.name}_quaternius_{ckey}.png")
        if os.path.isfile(path):
            panel = _fit(Image.open(path).convert("RGB"), PANEL)
        else:
            panel = fail_card(PANEL, "Missing", path)
        im.paste(panel, (x, y))
    y += PANEL[1]

    y += 18
    dr.text((pad, y),
            "Reference   ·   U.S. Army All-Army Boxing jab   ·   Public domain (Wikimedia Commons)",
            font=lab_f, fill=ACCENT)
    y += 34
    ref_box_w = w - 2 * pad
    ref_box_h = ref_h - 52
    if os.path.isfile(args.ref):
        fitted = _fit(Image.open(args.ref).convert("RGB"), (ref_box_w, ref_box_h), BG)
        im.paste(fitted, (pad, y))
    y += ref_box_h + 16

    t_weld = ((facts.get("tishkok") or {}).get("metrics") or {}).get("after_0.1mm_weld") or {}
    q_weld = ((facts.get("quaternius") or {}).get("metrics") or {}).get("after_0.1mm_weld") or {}
    q_g = ((facts.get("quaternius") or {}).get("glow") or {}).get("totals") or {}
    footer = (
        f"Tishkok weld parts/boundary/non-manifold = "
        f"{t_weld.get('loose_parts')} / {t_weld.get('boundary_edges')} / {t_weld.get('nonmanifold_edges')}"
        f"    bones=0    height ~1.79 m\n"
        f"Quaternius Superhero Male weld = "
        f"{q_weld.get('loose_parts')} / {q_weld.get('boundary_edges')} / {q_weld.get('nonmanifold_edges')}"
        f"    bones=65    body height ~1.82 m    "
        f"glow verts delt/tri/obl = {q_g.get('front_delt')}/{q_g.get('triceps')}/{q_g.get('obliques')}\n"
        f"Glow (punching-side): front delt purple, triceps yellow, obliques orange. "
        f"UAL Punch_Jab is CC0 (open hands, not fists). "
        f"Tishkok credit required (CC BY 4.0, Tishkok). "
        f"Reference: U.S. Army, public domain."
    )
    dr.multiline_text((pad, y), footer, font=small, fill=MUTED, spacing=4)
    im.save(args.out, "PNG")
    print("wrote", args.out, im.size)


if __name__ == "__main__":
    main()
