#!/usr/bin/env python3
"""Side-by-side: pilot DEMIC keyframes vs owner reference frame sheets."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
KF = ROOT / "output/pilot/demic/keyframes"
REF = Path("/home/ubuntu/.cursor/projects/agent-repos-gymform3d-archive/assets")
OUT = ROOT / "output/pilot/demic/ab_pilot_vs_demic.jpg"

PILOT_TILES = [
    ("wrong_stretch_front", "Pilot wrong stretch"),
    ("wrong_contract_front", "Pilot wrong contract"),
    ("right_stretch_front", "Pilot right stretch"),
    ("right_contract_front", "Pilot right contract"),
    ("right_stretch_side", "Pilot right side stretch"),
    ("right_contract_side", "Pilot right side contract"),
]
REF_IMAGES = [
    (REF / "DePzKDbFB0y.jpg", "DEMIC STORY ref"),
    (REF / "DetDsIDj2BJ.jpg", "DEMIC ref"),
]


def load_pilot(name):
    p = KF / f"{name}_beauty.png"
    im = Image.open(p).convert("RGB")
    return im.resize((540, 960), Image.Resampling.LANCZOS)


def crop_ref_panel(im, col, row, cols=4, rows=3):
    w, h = im.size
    cw, ch = w // cols, h // rows
    return im.crop((col * cw, row * ch, (col + 1) * cw, (row + 1) * ch)).resize((540, 960), Image.Resampling.LANCZOS)


def main():
    try:
        font = ImageFont.truetype(str(ROOT / "fonts/Yellowtail-Regular.ttf"), 28)
        font_s = ImageFont.truetype(str(ROOT / "fonts/Yellowtail-Regular.ttf"), 22)
    except OSError:
        font = font_s = ImageFont.load_default()

    ref_panels = []
    demic_story = Image.open(REF_IMAGES[0][0]).convert("RGB")
    demic = Image.open(REF_IMAGES[1][0]).convert("RGB")
    ref_panels.append((crop_ref_panel(demic_story, 1, 1), "DEMIC STORY (wrong/right grid)"))
    ref_panels.append((crop_ref_panel(demic_story, 0, 2), "DEMIC STORY side"))
    ref_panels.append((crop_ref_panel(demic, 2, 0), "DEMIC lat glow"))
    ref_panels.append((crop_ref_panel(demic, 1, 1), "DEMIC muscle UI"))

    pilot_imgs = [(load_pilot(n), lbl) for n, lbl in PILOT_TILES]

    cols = 4
    rows = max(len(pilot_imgs), len(ref_panels)) // 2 + 1
    cell_w, cell_h = 540, 1000
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (35, 39, 43))
    draw = ImageDraw.Draw(sheet)

    y = 0
    draw.text((cell_w, 30), "Reference (DEMIC / M&M)", fill=(220, 220, 220), font=font, anchor="mm")
    x = 0
    for im, lbl in ref_panels:
        sheet.paste(im, (x, 50))
        draw.text((x + cell_w // 2, cell_h - 30), lbl, fill=(200, 200, 200), font=font_s, anchor="mm")
        x += cell_w
        if x >= cols * cell_w:
            x = 0
            y += cell_h

    y2 = cell_h
    draw.text((cell_w * 2, y2 + 30), "Pilot (pilot-demic-1)", fill=(220, 220, 220), font=font, anchor="mm")
    x = 0
    for im, lbl in pilot_imgs:
        sheet.paste(im, (x, y2 + 50))
        draw.text((x + cell_w // 2, y2 + cell_h - 30), lbl, fill=(200, 200, 200), font=font_s, anchor="mm")
        x += cell_w

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=92)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
