#!/usr/bin/env python3
"""4-step test reel: step1 wrong front, step2 right front, step3 wrong side, step4 right side (contract pose)."""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
KF = ROOT / "output/pilot/lat/keyframes"
OUT = ROOT / "output/pilot/lat/test_reel_4step.mp4"
TMP = ROOT / "output/pilot/lat/reel_stills"


def load(name):
    p = KF / f"{name}_beauty.png"
    im = Image.open(p).convert("RGB")
    return im


def main():
    steps = [
        ("wrong_contract_front", "STEP 1 · WRONG · FRONT"),
        ("right_contract_front", "STEP 2 · RIGHT · FRONT"),
        ("wrong_contract_side", "STEP 3 · WRONG · SIDE"),
        ("right_contract_side", "STEP 4 · RIGHT · SIDE"),
    ]
    TMP.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        font = ImageFont.truetype(str(ROOT / "fonts/Yellowtail-Regular.ttf"), 52)
    except OSError:
        font = ImageFont.load_default()
    for i, (name, label) in enumerate(steps):
        im = load(name)
        draw = ImageDraw.Draw(im)
        draw.text((540, 120), label, fill=(255, 255, 255), font=font, anchor="mm")
        p = TMP / f"step{i+1}.png"
        im.save(p)
        paths.append(p)
    list_file = TMP / "list.txt"
    dur = 2.5
    with open(list_file, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
            f.write(f"duration {dur}\n")
        f.write(f"file '{paths[-1]}'\n")
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-vf",
            "fps=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(OUT),
        ]
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
