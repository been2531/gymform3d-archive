#!/usr/bin/env python3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
SM = ROOT / "output/pilot/demic/smoke"
OUT = ROOT / "output/pilot/demic/smoke_contact_2x2.jpg"

tiles = [
    ("wrong_front_stretch_beauty.png", "Wrong · front · stretch"),
    ("wrong_side_contract_beauty.png", "Wrong · side · contract"),
    ("right_front_stretch_beauty.png", "Right · front · stretch"),
    ("right_side_contract_beauty.png", "Right · side · contract"),
]

def main():
    imgs = [(Image.open(SM / n).convert("RGB"), lbl) for n, lbl in tiles]
    w, h = imgs[0][0].size
    sheet = Image.new("RGB", (w * 2, h * 2), (35, 39, 43))
    try:
        font = ImageFont.truetype(str(ROOT / "fonts/Yellowtail-Regular.ttf"), 36)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for i, (im, lbl) in enumerate(imgs):
        ox, oy = (i % 2) * w, (i // 2) * h
        sheet.paste(im, (ox, oy))
        draw.text((ox + w // 2, oy + 40), lbl, fill=(240, 240, 240), font=font, anchor="mm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=92)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
