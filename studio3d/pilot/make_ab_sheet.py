#!/usr/bin/env python3
"""A/B sheet: pilot key poses vs real Day 4 locked reference (filmstrip JPG → single panel)."""
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
def day4_panel(path: Path, cw: int, ch: int, frame_index: int = 8) -> Image.Image | None:
    """Reference JPGs are 15-frame horizontal strips (2700×320); crop one panel then letterbox to 9:16."""
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    pw = 180 if w >= 180 * 2 else w  # Day 4 locked JPGs = 15×180px filmstrip @ 320px tall
    n_panels = max(1, w // pw)
    if n_panels > 1:
        idx = min(max(0, frame_index), n_panels - 1)
        im = im.crop((idx * pw, 0, (idx + 1) * pw, h))
    # Fit panel into target (center crop to aspect)
    pw, ph = im.size
    target_aspect = cw / ch
    src_aspect = pw / ph
    if src_aspect > target_aspect:
        new_w = int(ph * target_aspect)
        x0 = (pw - new_w) // 2
        im = im.crop((x0, 0, x0 + new_w, ph))
    else:
        new_h = int(pw / target_aspect)
        y0 = (ph - new_h) // 2
        im = im.crop((0, y0, pw, y0 + new_h))
    return im.resize((cw, ch), Image.LANCZOS)


def cell(im, w, h):
    return im.convert("RGB").resize((w, h), Image.LANCZOS)


def main():
    pairs = [
        ("wrong_stretch_front", "wrong", "front", "STRETCH", 4),
        ("wrong_contract_front", "wrong", "front", "CONTRACT", 10),
        ("wrong_stretch_side", "wrong", "side", "STRETCH", 4),
        ("wrong_contract_side", "wrong", "side", "CONTRACT", 10),
        ("right_stretch_front", "right", "front", "STRETCH", 4),
        ("right_contract_front", "right", "front", "CONTRACT", 10),
        ("right_stretch_side", "right", "side", "STRETCH", 4),
        ("right_contract_side", "right", "side", "CONTRACT", 10),
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
        "Pilot v3 vs Day 4 locked reference (look)",
        fill=(255, 255, 255),
        font=title_f,
        anchor="mm",
    )
    y0 = 72
    for i, (name, form, _cam, phase, fidx) in enumerate(pairs):
        y = y0 + i * ch
        ref_path = DAY4_WRONG if form == "wrong" else DAY4_RIGHT
        ref = day4_panel(ref_path, cw, ch, frame_index=fidx)
        if ref:
            sheet.paste(ref, (0, y))
            draw.text(
                (cw // 2, y + 20),
                f"Day 4 ref · {form.upper()} · {phase}",
                fill=(255, 200, 200),
                font=label_f,
                anchor="mm",
            )
        pilot = cell(Image.open(KF / f"{name}_beauty.png"), cw, ch)
        sheet.paste(pilot, (cw, y))
        draw.text(
            (cw + cw // 2, y + 20),
            f"Pilot v3 · {name.replace('_', ' ')}",
            fill=(180, 255, 210),
            font=label_f,
            anchor="mm",
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=92)
    ART.mkdir(parents=True, exist_ok=True)
    sheet.save(ART / "ab_keypose_vs_day4_v2.jpg", quality=92)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
