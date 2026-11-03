#!/usr/bin/env python3
"""A/B sheet: pilot key poses vs Day-4-look reference (Day 2 lat pulldown frame)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
KF = ROOT / "output/pilot/lat/keyframes"
REF = ROOT / "output/day2_scene1_wrong.jpg"
OUT = ROOT / "output/pilot/lat/ab_keypose_vs_day4.jpg"
ART = Path("/opt/cursor/artifacts/screenshots")


def cell(im, w, h):
    im = im.convert("RGB")
    return im.resize((w, h), Image.LANCZOS)


def main():
    pairs = [
        ("wrong_stretch_front", "WRONG · STRETCH · FRONT"),
        ("wrong_contract_front", "WRONG · CONTRACT · FRONT"),
        ("right_stretch_front", "RIGHT · STRETCH · FRONT"),
        ("right_contract_front", "RIGHT · CONTRACT · FRONT"),
        ("wrong_stretch_side", "WRONG · STRETCH · SIDE"),
        ("wrong_contract_side", "WRONG · CONTRACT · SIDE"),
        ("right_stretch_side", "RIGHT · STRETCH · SIDE"),
        ("right_contract_side", "RIGHT · CONTRACT · SIDE"),
    ]
    cw, ch = 540, 960
    cols = 4
    rows = len(pairs)
    sheet = Image.new("RGB", (cols * cw, (rows + 1) * ch // 2 + ch), (20, 22, 26))
    try:
        title_f = ImageFont.truetype(str(ROOT / "fonts/Yellowtail-Regular.ttf"), 44)
        label_f = ImageFont.truetype(str(ROOT / "fonts/Yellowtail-Regular.ttf"), 28)
    except OSError:
        title_f = label_f = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    draw.text((sheet.width // 2, 40), "Pilot Blender vs Day 4 look reference", fill=(255, 255, 255), font=title_f, anchor="mm")
    ref = cell(Image.open(REF), cw, ch) if REF.exists() else Image.new("RGB", (cw, ch), (35, 39, 43))
    y0 = 80
    for i, (name, label) in enumerate(pairs):
        y = y0 + i * ch
        pilot = cell(Image.open(KF / f"{name}_beauty.png"), cw, ch)
        sheet.paste(ref, (0, y))
        sheet.paste(pilot, (cw, y))
        draw.text((cw // 2, y + 24), "Day 4 ref (Day 2 clip)", fill=(255, 180, 180), font=label_f, anchor="mm")
        draw.text((cw + cw // 2, y + 24), "Pilot " + label, fill=(180, 255, 200), font=label_f, anchor="mm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=92)
    ART.mkdir(parents=True, exist_ok=True)
    sheet.save(ART / "ab_keypose_vs_day4.jpg", quality=92)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
