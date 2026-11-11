#!/usr/bin/env python3
"""Compose xbot_boxing_test.png from eval_xbot_boxing stills + US Army jab ref."""
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
LABEL_H = 52
TITLE_H = 92
LEGEND_H = 48
PAD = 10

COL_LEGEND = (
    ("front delt", (140, 41, 255)),
    ("triceps", (255, 184, 10)),
    ("biceps", (255, 224, 46)),
    ("chest", (255, 56, 133)),
    ("obliques", (255, 87, 20)),
    ("calf", (31, 209, 224)),
    ("hip", (56, 242, 97)),
    ("quads", (122, 255, 41)),
    ("glutes", (255, 82, 148)),
)


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


def load_panel(path):
    if path and os.path.isfile(path):
        return _fit(Image.open(path).convert("RGB"), PANEL)
    im = Image.new("RGB", PANEL, (32, 28, 30))
    dr = ImageDraw.Draw(im)
    dr.text((36, 80), "missing", font=_font(22, True), fill=(220, 110, 96))
    if path:
        for i, line in enumerate(_wrap(dr, path, _font(16), PANEL[0] - 72)):
            dr.text((36, 140 + i * 26), line, font=_font(16), fill=MUTED)
    return im


def caption_bar(text, sub, width):
    im = Image.new("RGB", (width, LABEL_H), (28, 28, 32))
    dr = ImageDraw.Draw(im)
    dr.text((12, 6), text, font=_font(17, True), fill=FG)
    dr.text((12, 28), sub, font=_font(13), fill=MUTED)
    return im


def clip_by_key(facts, key):
    for c in facts.get("clips", []):
        if c.get("key") == key:
            return c
    return {}


def key_sub(clip, stage):
    keys = clip.get("keys") or {}
    side = clip.get("punch_side") or "?"
    f = keys.get(stage)
    parts = [f"f{f}" if f is not None else "—", f"hand {side}"]
    dist = keys.get(f"{stage}_dist")
    if dist is None and stage == "guard":
        dl, dr = keys.get("guard_dist_l"), keys.get("guard_dist_r")
        if dl is not None and dr is not None:
            parts.append(f"L {dl*100:.0f} / R {dr*100:.0f} cm to head")
    elif dist is not None:
        parts.append(f"{dist*100:.0f} cm to head")
    hz = keys.get(f"{stage}_hand_z")
    if hz is not None:
        parts.append(f"hand z {hz:.2f} m")
    return "   ·   ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    facts = json.load(open(args.facts))

    punches = ["jab", "cross", "hook", "uppercut"]
    idle = clip_by_key(facts, "idle")

    cols = 3
    # header + idle/ref/legend row + 4 punch rows
    rows = 1 + 4
    W = cols * PANEL[0] + (cols + 1) * PAD
    H = (TITLE_H + LEGEND_H
         + rows * (LABEL_H + PANEL[1]) + (rows + 1) * PAD + 16)
    sheet = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(sheet)

    dr.text((PAD + 8, 14),
            "X Bot boxing test   ·   Mixamo built-in mannequin   ·   posture over muscle detail",
            font=_font(26, True), fill=FG)
    dr.text((PAD + 8, 50),
            "Fixed 3/4-front camera  ·  540×960 Cycles 16spp  ·  clay + bone-weight blob glow  ·  "
            "keyframes by punching-hand distance to head",
            font=_font(15), fill=MUTED)

    # colour legend
    lx = PAD + 8
    ly = TITLE_H - 4
    dr.text((lx, ly), "Glow:", font=_font(14, True), fill=MUTED)
    lx += 52
    for name, rgb in COL_LEGEND:
        dr.rectangle((lx, ly + 2, lx + 14, ly + 16), fill=rgb)
        dr.text((lx + 18, ly), name, font=_font(13), fill=FG)
        lx += 18 + int(dr.textlength(name, font=_font(13))) + 16

    y = TITLE_H + LEGEND_H

    def paste_cell(cx, cy, img, cap, sub):
        sheet.paste(caption_bar(cap, sub, PANEL[0]), (cx, cy))
        sheet.paste(img, (cx, cy + LABEL_H))

    # Row 0: idle | US Army ref | notes
    idle_img = load_panel((idle.get("stills") or {}).get("guard"))
    ref_img = load_panel(args.ref)
    notes = Image.new("RGB", PANEL, (26, 26, 30))
    nd = ImageDraw.Draw(notes)
    nd.text((28, 40), "How to read", font=_font(22, True), fill=ACCENT)
    body = (
        "Each punch row is guard / mid / full extension. "
        "Distance is punching-hand to head (small = guard, large = extension). "
        "Glow follows Mixamo skin weights along the live bones, not a static coat. "
        "Camera is locked (same 3/4 front) so root drift would walk out of frame. "
        "Idle is un-tinted clay. Compare the jab row to the U.S. Army public-domain photo."
    )
    yy = 90
    for line in _wrap(nd, body, _font(16), PANEL[0] - 56):
        nd.text((28, yy), line, font=_font(16), fill=FG)
        yy += 26
    extra = []
    if idle:
        extra.append(
            f"Idle root drift {idle.get('root_drift_m', 0):.3f} m"
            f"{' (locked)' if idle.get('root_locked') else ''}.")
        extra.append(
            f"Min foot z {idle.get('min_foot_z')} m · "
            f"both-feet-up frames {idle.get('frames_both_feet_ungrounded')}.")
    extra.append("Mixamo X Bot: commercial finished works OK; raw FBX not redistributed.")
    yy += 16
    for line in extra:
        for wline in _wrap(nd, line, _font(15), PANEL[0] - 56):
            nd.text((28, yy), wline, font=_font(15), fill=MUTED)
            yy += 24

    cells0 = [
        (idle_img, "Fighting Idle  ·  guard", key_sub(idle, "guard")),
        (ref_img, "U.S. Army jab  ·  public domain",
         "Reference: chin tucked, lead hand extends, rear hand high"),
        (notes, "Notes", "Locked camera · bone-weight glow · no raw Mixamo commit"),
    ]
    for i, (im, cap, sub) in enumerate(cells0):
        cx = PAD + i * (PANEL[0] + PAD)
        paste_cell(cx, y, im, cap, sub)
    y += LABEL_H + PANEL[1] + PAD

    stage_label = {"guard": "guard", "mid": "mid", "full": "full extension"}
    for key in punches:
        clip = clip_by_key(facts, key)
        title = clip.get("title") or key
        muscles = ", ".join(clip.get("muscles") or [])
        drift = clip.get("root_drift_m")
        drift_s = f"drift {drift:.3f} m" if drift is not None else ""
        if clip.get("root_locked"):
            drift_s += " locked"
        feet = ""
        if clip.get("min_foot_z") is not None:
            feet = f"min foot z {clip['min_foot_z']:.3f} m"
            up = clip.get("frames_both_feet_ungrounded")
            if up:
                feet += f" · {up} frames both feet up"
        meta = "  ·  ".join(p for p in (muscles, drift_s, feet) if p)
        for i, stage in enumerate(("guard", "mid", "full")):
            path = (clip.get("stills") or {}).get(stage)
            img = load_panel(path)
            cap = f"{title}  ·  {stage_label[stage]}"
            if i == 0 and meta:
                cap = f"{title}  ·  {stage_label[stage]}"
            sub = key_sub(clip, stage)
            if i == 0 and meta:
                sub = f"{sub}   ·   {meta}" if sub else meta
            cx = PAD + i * (PANEL[0] + PAD)
            paste_cell(cx, y, img, cap, sub)
        y += LABEL_H + PANEL[1] + PAD

    sheet.save(args.out, "PNG")
    print("WROTE", args.out, sheet.size, os.path.getsize(args.out))


if __name__ == "__main__":
    main()
