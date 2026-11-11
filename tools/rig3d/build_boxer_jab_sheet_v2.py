#!/usr/bin/env python3
"""Compose boxer_jab_test_v2.png: Tishkok rest+jab stills + US Army reference."""
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


def load_panel(path, fallback_title, fallback_body):
    if path and os.path.isfile(path):
        return _fit(Image.open(path).convert("RGB"), PANEL)
    return fail_card(PANEL, fallback_title, fallback_body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", default="boxer_jab_tishkok")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--facts", default="")
    args = ap.parse_args()
    facts = {}
    if args.facts and os.path.isfile(args.facts):
        facts = json.load(open(args.facts))

    pad = 18
    header_h = 110
    label_h = 36
    row_h = 36
    ref_h = 620
    footer_h = 160
    cols = 4
    w = pad + cols * (PANEL[0] + pad)
    h = header_h + row_h + label_h + PANEL[1] + 28 + ref_h + footer_h
    im = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(im)
    title_f = _font(26, True)
    sub_f = _font(15)
    lab_f = _font(18, True)
    small = _font(14)

    keys = facts.get("keyframes") or {}
    shots = facts.get("shots") or {}
    glow = ((facts.get("glow") or {}).get("totals")) or {}
    title = "Boxer jab test v2   ·   Tishkok A-pose Rigify   ·   ACCAD Male2 JabRight"
    dr.text((pad, 18), title, font=title_f, fill=FG)
    sub = (
        f"Punching side: right    frames guard/mid/full = "
        f"{keys.get('guard')} / {keys.get('mid')} / {keys.get('full')}    "
        f"reaches {keys.get('guard_reach')} / {keys.get('mid_reach')} / {keys.get('full_reach')} m    "
        f"540×960 Cycles  ·  charcoal studio  ·  yaw −40° (punch side 3/4)"
    )
    y = 58
    for line in _wrap(dr, sub, sub_f, w - 2 * pad):
        dr.text((pad, y), line, font=sub_f, fill=MUTED)
        y += 20

    ok = facts.get("ok")
    retarget = facts.get("retarget") or ""
    keys = facts.get("keyframes") or {}
    full_z = keys.get("full_hand_z")
    height = facts.get("height_m") or 0
    jab_ok = bool(ok) and retarget and (full_z is None or height == 0 or full_z < height * 0.95)
    if not ok:
        status = f"FAILED: {str(facts.get('error') or 'see facts')[:160]}"
        col = FAIL
    elif not jab_ok:
        status = (
            "Bind OK (standing A-pose, gloves/shoes held). "
            "Jab retarget FAILED as a boxing jab — arms go overhead, not forward "
            f"(full hand_z={full_z:.2f} m on a {height:.2f} m figure, {retarget}). "
            "Stopped rather than keep guessing ACCAD axes."
        )
        col = FAIL
    else:
        status = f"bind+retarget OK ({retarget})"
        col = (140, 196, 140)
    # wrapped status
    sy = y + 4
    for line in _wrap(dr, status, sub_f, w - 2 * pad):
        dr.text((pad, sy), line, font=sub_f, fill=col)
        sy += 18

    y = header_h
    dr.text((pad, y + 6),
            "Tishkok  ·  Stylized Boxer Base Mesh  ·  CC BY 4.0  ·  Rigify fitted to A-pose  ·  sculpted gloves = fists",
            font=lab_f, fill=ACCENT)
    y += row_h
    labels = [
        "Rest (bind check)",
        "Guard",
        "Mid-extension",
        "Full extension",
    ]
    paths = [
        os.path.join(args.dir, f"{args.name}_rest.png"),
        os.path.join(args.dir, f"{args.name}_guard.png"),
        os.path.join(args.dir, f"{args.name}_mid.png"),
        os.path.join(args.dir, f"{args.name}_full.png"),
    ]
    for i, lab in enumerate(labels):
        x = pad + i * (PANEL[0] + pad)
        extra = ""
        key = ("rest", "guard", "mid", "full")[i]
        if key in shots:
            sm = shots[key]
            punch = sm.get("punch") or {}
            extra = f"   reach {punch.get('reach')}"
            if sm.get("punch_vs_head") is not None:
                extra += f"   punch-head Δz {sm['punch_vs_head']:.2f}"
        dr.text((x, y), lab + extra, font=sub_f, fill=MUTED)
    y += label_h
    err = facts.get("error") or "Missing still — bind or retarget did not finish."
    for i, path in enumerate(paths):
        x = pad + i * (PANEL[0] + pad)
        panel = load_panel(path, "Missing", err)
        im.paste(panel, (x, y))
    y += PANEL[1] + 20

    dr.text((pad, y),
            "Reference   ·   U.S. Army All-Army Boxing jab   ·   Public domain (Wikimedia Commons)",
            font=lab_f, fill=ACCENT)
    y += 34
    ref_box_w = w - 2 * pad
    ref_box_h = ref_h - 50
    if os.path.isfile(args.ref):
        im.paste(_fit(Image.open(args.ref).convert("RGB"), (ref_box_w, ref_box_h), BG),
                 (pad, y))
    y += ref_box_h + 16

    shells = facts.get("shells") or {}
    parent = facts.get("parent_bones") or {}
    footer = (
        f"Shells after 0.1 mm weld: "
        f"body={((shells.get('body') or {}).get('tris'))} tris  "
        f"shorts={((shells.get('shorts') or {}).get('tris'))}  "
        f"glove L/R={((shells.get('glove_l') or {}).get('tris'))}/"
        f"{((shells.get('glove_r') or {}).get('tris'))}  "
        f"auto-weight body+shorts; rigid gloves→{parent.get('glove_r')}  "
        f"shoes→{parent.get('shoe_r')}\n"
        f"Glow verts (em {((facts.get('glow') or {}).get('em_str'))}): "
        f"front delt {glow.get('front_delt')}  triceps {glow.get('triceps')}  "
        f"obliques {glow.get('obliques')}   "
        f"(purple / yellow / orange, punching-side only)\n"
        f"Motion: {facts.get('motion') or 'ACCAD Male2_E2_JabRight.bvh'}   "
        f"retarget: {facts.get('retarget')}   "
        f"height ~{facts.get('height_m')} m\n"
        f"Credits: Tishkok (CC BY 4.0); ACCAD / Ohio State mocap Male2 JabRight (CC BY 3.0); "
        f"U.S. Army reference public domain. No Mixamo, no paid assets."
    )
    dr.multiline_text((pad, y), footer, font=small, fill=MUTED, spacing=4)
    im.save(args.out, "PNG")
    print("wrote", args.out, im.size)


if __name__ == "__main__":
    main()
