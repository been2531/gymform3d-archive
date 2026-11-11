#!/usr/bin/env python3
"""Compose xbot_jab_wrongform_v1.png from wrong_form.py panel renders.

  python3 tools/rig3d/build_wrongform_sheet.py
Columns: correct + wrong variants; rows: mid-punch / extension / return.
Each column header shows the boxing_qc mocap verdict (faults flagged).
"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "output", "_basemodels", "eval"))
WF = os.path.join(OUT, "wrongform")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def f(sz, bold=False):
    p = FONTB if bold else FONT
    return ImageFont.truetype(p, sz) if os.path.isfile(p) else ImageFont.load_default()


def main(out_name="xbot_jab_wrongform_v1.png"):
    meta = json.load(open(os.path.join(WF, "sheet_meta.json")))
    cols, stages = meta["cols"], meta["stages"]
    first = Image.open(meta["paths"][f"{cols[0]}|{stages[0]}"])
    pw, ph = first.size
    lab_w, head_h, foot_h, title_h = 120, 92, 70, 64
    W = lab_w + len(cols) * (pw + 8) + 8
    H = title_h + head_h + len(stages) * (ph + 8) + foot_h
    sheet = Image.new("RGB", (W, H), (14, 14, 16))
    d = ImageDraw.Draw(sheet)
    d.text((16, 14), f"Jab: correct vs procedural wrong forms  ({meta['punch']}, v2 lighting)",
           fill=(235, 235, 238), font=f(26, True))
    stage_names = {"mid": "mid-punch", "full": "extension", "ret": "return"}
    for ci, c in enumerate(cols):
        x = lab_w + 8 + ci * (pw + 8)
        if c == "correct":
            title, sub = "CORRECT (base)", f"mocap dist {meta['base']['dist']}  faults: {meta['base']['faults'] or 'none'}"
            col = (110, 220, 140)
        else:
            r = meta["report"].get(c, {})
            title = meta["labels"].get(c, c)
            sub = (f"flags: {', '.join(r.get('faults') or []) or 'none'}  "
                   f"{'exact' if r.get('exact') else 'NOT exact'}")
            col = (255, 120, 100)
        d.text((x + 4, title_h + 6), title, fill=col, font=f(20, True))
        d.text((x + 4, title_h + 40), sub, fill=(200, 200, 205), font=f(14))
        for si, st in enumerate(stages):
            y = title_h + head_h + si * (ph + 8)
            p = meta["paths"].get(f"{c}|{st}")
            if p and os.path.isfile(p):
                sheet.paste(Image.open(p).convert("RGB"), (x, y))
    frames = {}
    try:
        facts = json.load(open(os.path.join(OUT, "xbot_punch_pick_facts_v4.json")))
        p = next(x for x in facts["punches"] if x["id"] == meta["punch"])
        frames = {k: p[k] for k in ("mid", "full", "ret")}
    except Exception:
        pass
    for si, st in enumerate(stages):
        y = title_h + head_h + si * (ph + 8) + ph // 2 - 10
        d.text((14, y), stage_names.get(st, st), fill=(210, 210, 214), font=f(18, True))
        if st in frames:
            d.text((14, y + 26), f"f{frames[st]}", fill=(160, 160, 166), font=f(15))

    d.text((16, H - foot_h + 12),
           "Faults are measured, not labelled: boxing_qc.reference_distance vs CMU+ACCAD mocap "
           "(z>2.5 & abs gate). Same camera for every column.",
           fill=(170, 170, 176), font=f(15))
    d.text((16, H - foot_h + 36),
           "Variants = additive envelope-weighted bone rotations on the original Mixamo action "
           "(tools/rig3d/wrong_form.py).", fill=(170, 170, 176), font=f(15))
    out = os.path.join(OUT, out_name)
    sheet.save(out)
    print("WROTE", out, sheet.size)


if __name__ == "__main__":
    main()
