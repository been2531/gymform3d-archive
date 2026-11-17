#!/usr/bin/env python3
"""Label stills, build a contact sheet, and mux a silent reel.

Typography (locked):
  one label, at most 3 words, white uppercase Bebas Neue
  on a semi-transparent black pill, top-centre.
  Bottom 15% of the frame stays clear of type.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/sand-box/google/Bebas Neue/BebasNeue-Regular.ttf"
PILL_ALPHA = 185
# Designed on 1080-wide frames (1.5x the 720 concept stills).
LABEL_PX_1080 = 120
TOP_Y_1080 = 72
PAD_X_1080 = 48
PAD_Y_1080 = 20
RADIUS_1080 = 36

EXPRESSIONS = (
    ("smug", "SQUAT DAY"),
    ("scream", "SEND HELP"),
    ("faint", "NEVER AGAIN"),
)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def _fit_font(lines: list[str], max_width: int, start_px: int, min_px: int) -> tuple[ImageFont.FreeTypeFont, int]:
    size = start_px
    while size >= min_px:
        font = _font(size)
        widths = [font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines]
        if max(widths) <= max_width:
            return font, size
        size -= 2
    return _font(min_px), min_px


def draw_label(im: Image.Image, text: str) -> Image.Image:
    """One uppercase label in a semi-transparent black pill, top-centre."""
    lines = text.upper().split()
    # Keep a short phrase as one line when it is two or three short words.
    if len(lines) <= 3:
        lines = [text.upper()]
    w, h = im.size
    scale = w / 1080
    label_px = max(70, int(round(LABEL_PX_1080 * scale)))
    top_y = int(round(TOP_Y_1080 * scale))
    pad_x = int(round(PAD_X_1080 * scale))
    pad_y = int(round(PAD_Y_1080 * scale))
    radius = int(round(RADIUS_1080 * scale))
    safe_bottom = int(h * 0.85)

    base = im.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font, _ = _fit_font(lines, w - 2 * pad_x - 40, label_px, min_px=max(70, int(70 * scale)))
    boxes = [font.getbbox(line) for line in lines]
    widths = [b[2] - b[0] for b in boxes]
    glyph_h = max(b[3] for b in boxes) - min(b[1] for b in boxes)
    gap = int(8 * scale) if len(lines) > 1 else 0
    tw = max(widths)
    th = glyph_h * len(lines) + gap * (len(lines) - 1)
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2
    x0 = (w - pill_w) // 2
    y0 = top_y
    x1, y1 = x0 + pill_w, y0 + pill_h
    if y1 >= safe_bottom:
        raise RuntimeError(f"label pill entered the bottom 15% ({y1} >= {safe_bottom})")
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=(0, 0, 0, PILL_ALPHA))
    y = y0 + pad_y
    for line, lw, b in zip(lines, widths, boxes):
        tx = (w - lw) // 2
        draw.text((tx, y - b[1]), line, font=font, fill=(255, 255, 255, 255))
        y += glyph_h + gap
    return Image.alpha_composite(base, overlay).convert("RGB")


def _load_raw(out: Path, name: str) -> Image.Image:
    for cand in (out / f"raw_{name}.png", out / f"raw_{name}.jpg"):
        if cand.exists():
            return Image.open(cand).convert("RGB")
    raise FileNotFoundError(f"missing raw still for {name} in {out}")


def compose_stills(out: Path) -> list[Path]:
    labeled_paths = []
    panels = []
    for name, label in EXPRESSIONS:
        raw = _load_raw(out, name)
        labeled = draw_label(raw, label)
        dest = out / f"{name}.jpg"
        labeled.save(dest, quality=95, subsampling=0)
        labeled_paths.append(dest)
        panels.append(labeled)
        print(f"labeled {dest} {labeled.size}")
    sheet_path = out / "sheet.jpg"
    build_sheet(panels, sheet_path)
    print(f"sheet {sheet_path} {Image.open(sheet_path).size}")
    return labeled_paths


def build_sheet(panels: list[Image.Image], dest: Path) -> None:
    gap, margin = 24, 24
    w, h = panels[0].size
    sw = margin * 2 + len(panels) * w + (len(panels) - 1) * gap
    sh = margin * 2 + h
    canvas = Image.new("RGB", (sw, sh), (12, 12, 14))
    x = margin
    for p in panels:
        if p.size != (w, h):
            p = p.resize((w, h), Image.Resampling.LANCZOS)
        canvas.paste(p, (x, margin))
        x += w + gap
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=95, subsampling=0)


def label_frames(frames_dir: Path, labeled_dir: Path, fps: int, segment_frames: list[tuple[int, int, str]]) -> int:
    """Apply the segment label to every frame. Returns frame count."""
    labeled_dir.mkdir(parents=True, exist_ok=True)
    raws = sorted(frames_dir.glob("frame_*.png"))
    if not raws:
        raws = sorted(frames_dir.glob("*.png"))
    if not raws:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    count = 0
    for path in raws:
        # frame_0001.png -> 1
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        idx = int(digits) if digits else count + 1
        label = segment_frames[-1][2]
        for start, end, text in segment_frames:
            if start <= idx <= end:
                label = text
                break
        im = draw_label(Image.open(path).convert("RGB"), label)
        im.save(labeled_dir / f"frame_{idx:04d}.jpg", quality=92, subsampling=0)
        count += 1
    return count


def mux_reel(labeled_dir: Path, dest: Path, fps: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(labeled_dir / "frame_%04d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-movflags", "+faststart",
        "-an",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    print(f"reel {dest}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("stills", "reel", "all"), default="stills")
    p.add_argument("--frames", type=Path, default=None)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--reel-seconds", type=float, default=13.0)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.mode in ("stills", "all"):
        compose_stills(out)

    if args.mode in ("reel", "all"):
        frames = args.frames or (out / "frames")
        fps = args.fps
        total = int(round(args.reel_seconds * fps))
        # 0-4.5s squat day, 4.5-9.0s send help, 9.0-end never again
        s1 = int(4.5 * fps)
        s2 = int(9.0 * fps)
        labeled_dir = out / "frames_labeled"
        n = label_frames(
            frames,
            labeled_dir,
            fps,
            [
                (1, s1, "SQUAT DAY"),
                (s1 + 1, s2, "SEND HELP"),
                (s2 + 1, max(total, s2 + 1), "NEVER AGAIN"),
            ],
        )
        print(f"labeled {n} frames")
        mux_reel(labeled_dir, out / "reel.mp4", fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
