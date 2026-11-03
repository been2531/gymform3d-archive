#!/usr/bin/env python3
"""A/B sheet: pilot v2 key poses vs real Day 4 locked reference frames (RDL step 1 / 4)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
KF = ROOT / "output/pilot/lat/keyframes"
OUT = ROOT / "output/pilot/lat/ab_keypose_vs_day4_v2.jpg"
ART = Path("/opt/cursor/artifacts/screenshots")

DAY4_WRONG = Path(
    "/home/ubuntu/.cursor/projects/agent-repos-gymform3d-archive/assets/_check_day4_rdl_step1_locked_720p.mp4.jpg"
)
DAY4_RIGHT = Path(
    "/home/ubuntu/.cursor/projects/agent-repos-gymform3d-archive/assets/_check_day4_rdl_step4_locked_720p.mp4.jpg"
)


def cell(im, w, h):
    return im.convert("RGB").resize((w, h), Image.LANCZOS)


def main():
    pairs = [
        ("wrong_stretch_front", "wrong", "front", "STRETCH"),
        ("wrong_contract_front", "wrong", "front", "CONTRACT"),
        ("wrong_stretch_side", "wrong", "side", "STRETCH"),
        ("wrong_contract_side", "wrong", "side", "CONTRACT"),
        ("right_stretch_front", "right", "front", "STRETCH"),
        ("right_contract_front", "right", "front", "CONTRACT"),
        ("right_stretch_side", "right", "side", "STRETCH"),
        ("right_contract_side", "right", "side", "CONTRACT"),
    ]
    cw, ch = 540, 960
    sheet = Image.new("RGB", (cw * 2, ch * len(pairs)), (22, 24, 28))
    try:
        title_f = ImageFont.truetype(str(ROOT / "fonts/Yellowtail-Regular.ttf"), 40)
        label_f = ImageFont.truetype(str(ROOT / "fonts/Yellowtail-Regular.ttf"), 26)
    except OSError:
        title_f = label_f = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (cw, 36),
        "Pilot v2 (écorché) vs Day 4 locked reference (look)",
        fill=(255, 255, 255),
        font=title_f,
        anchor="mm",
    )
    ref_wrong = cell(Image.open(DAY4_WRONG), cw, ch) if DAY4_WRONG.exists() else None
    ref_right = cell(Image.open(DAY4_RIGHT), cw, ch) if DAY4_RIGHT.exists() else None
    y0 = 72
    for i, (name, form, _cam, phase) in enumerate(pairs):
        y = y0 + i * ch
        ref = ref_wrong if form == "wrong" else ref_right
        if ref:
            sheet.paste(ref, (0, y))
            draw.text((cw // 2, y + 20), f"Day 4 ref · {form.upper()} · {phase}", fill=(255, 200, 200), font=label_f, anchor="mm")
        pilot = cell(Image.open(KF / f"{name}_beauty.png"), cw, ch)
        sheet.paste(pilot, (cw, y))
        draw.text((cw + cw // 2, y + 20), f"Pilot v2 · {name.replace('_', ' ')}", fill=(180, 255, 210), font=label_f, anchor="mm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=92)
    ART.mkdir(parents=True, exist_ok=True)
    sheet.save(ART / "ab_keypose_vs_day4_v2.jpg", quality=92)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
