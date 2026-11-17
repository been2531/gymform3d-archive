"""CLI: stills, comparison sheets, and silent squat test clips."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, Optional

from PIL import Image, ImageDraw, ImageFont

from . import neon as neon_mod
from . import paper as paper_mod
from .label import draw_label
from .skeleton import CANVAS, DEFAULT, Proportions, solve_squat, squat_depth

ROOT = Path(__file__).resolve().parents[1]
CONCEPT_DIR = ROOT / "output" / "concepts" / "2026-11-17"
EVOLVE = CONCEPT_DIR / "evolve"
FONT = "/usr/share/fonts/truetype/sand-box/google/Poppins/Poppins-Bold.ttf"

Style = Literal["neon", "paper"]


def _params(style: Style, round_n: int):
    if style == "neon":
        return neon_mod.V2 if round_n >= 2 else neon_mod.V1
    return paper_mod.V2 if round_n >= 2 else paper_mod.V1


def render_frame(
    style: Style,
    depth: float,
    pulse: Optional[float] = None,
    size=CANVAS,
    params=None,
    label: Optional[str] = None,
    prop: Proportions = DEFAULT,
) -> Image.Image:
    joints = solve_squat(depth, prop)
    if pulse is None:
        pulse = depth
    if style == "neon":
        im = neon_mod.render(joints, size=size, pulse=pulse, params=params)
        text = label if label is not None else "NEON STICKMAN"
    else:
        im = paper_mod.render(joints, size=size, pulse=pulse, params=params)
        text = label if label is not None else "PAPER CUTOUT"
    if text:
        im = draw_label(im, text)
    return im


def render_still(
    style: Style,
    out: Path,
    depth: float = 1.0,
    round_n: int = 1,
    label: Optional[str] = None,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    im = render_frame(style, depth, params=_params(style, round_n), label=label)
    im.save(out, "JPEG", quality=93, subsampling=0, optimize=True)
    return out


def _tag(im: Image.Image, text: str) -> Image.Image:
    """Small lower-left tag used only on comparison sheets, not on reels."""
    canvas = im.convert("RGB")
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 28)
    pad = 10
    x0, y0, x1, y1 = font.getbbox(text)
    tw, th = x1 - x0, y1 - y0
    w, h = canvas.size
    bx, by = 16, h - th - pad * 2 - 16
    d.rounded_rectangle(
        [bx, by, bx + tw + pad * 2, by + th + pad * 2],
        radius=12,
        fill=(0, 0, 0),
        outline=(255, 255, 255),
        width=2,
    )
    d.text((bx + pad - x0, by + pad - y0), text, font=font, fill=(255, 255, 255))
    return canvas


def make_sheet(
    lefts: list[tuple[Image.Image, str]],
    rights: list[tuple[Image.Image, str]],
    out: Path,
    width: int = 1080,
) -> Path:
    """2-column phone sheet. Each cell keeps a 9:16 panel."""
    rows = max(len(lefts), len(rights))
    gutter = 10
    col_w = (width - gutter * 3) // 2
    col_h = int(col_w * 16 / 9)
    sheet_h = gutter * (rows + 1) + col_h * rows
    sheet = Image.new("RGB", (width, sheet_h), (14, 14, 16))

    def place(im: Image.Image, tag: str, col: int, row: int) -> None:
        thumb = _tag(im, tag).resize((col_w, col_h), Image.Resampling.LANCZOS)
        x = gutter + col * (col_w + gutter)
        y = gutter + row * (col_h + gutter)
        sheet.paste(thumb, (x, y))

    for i, (im, tag) in enumerate(lefts):
        place(im, tag, 0, i)
    for i, (im, tag) in enumerate(rights):
        place(im, tag, 1, i)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=92, subsampling=0, optimize=True)
    return out


def concept_image(style: Style) -> Image.Image:
    name = (
        "style01_neon_stickman.jpg"
        if style == "neon"
        else "style05_paper_cutout.jpg"
    )
    path = CONCEPT_DIR / name
    return Image.open(path).convert("RGB")


def render_clip(
    style: Style,
    out: Path,
    duration: float = 7.0,
    fps: int = 30,
    reps: int = 2,
    round_n: int = 3,
) -> tuple[Path, float]:
    """Silent H.264 clip, streamed to ffmpeg (no frame dump). Returns (path, seconds)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = CANVAS
    params = _params(style, round_n)
    n = int(round(duration * fps))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        str(out),
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for i in range(n):
            t = i / fps
            depth = squat_depth(t, duration=duration, reps=reps)
            im = render_frame(style, depth, pulse=depth, params=params)
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        err = proc.stderr.read()
        rc = proc.wait(timeout=60)
    except Exception:
        proc.kill()
        raise
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError((err or b"").decode("utf-8", "replace")[-2000:])
    return out, elapsed


def cmd_round1() -> None:
    EVOLVE.mkdir(parents=True, exist_ok=True)
    neon = render_still("neon", EVOLVE / "round1_neon.jpg", depth=1.0, round_n=1)
    paper = render_still("paper", EVOLVE / "round1_paper.jpg", depth=1.0, round_n=1)
    make_sheet(
        [(concept_image("neon"), "CONCEPT"), (concept_image("paper"), "CONCEPT")],
        [(Image.open(neon), "CODE R1"), (Image.open(paper), "CODE R1")],
        EVOLVE / "round1_sheet.jpg",
    )
    print(f"wrote {neon}")
    print(f"wrote {paper}")
    print(f"wrote {EVOLVE / 'round1_sheet.jpg'}")


def cmd_round2() -> None:
    EVOLVE.mkdir(parents=True, exist_ok=True)
    neon = render_still("neon", EVOLVE / "round2_neon.jpg", depth=1.0, round_n=2)
    paper = render_still("paper", EVOLVE / "round2_paper.jpg", depth=1.0, round_n=2)
    r1n = Image.open(EVOLVE / "round1_neon.jpg")
    r1p = Image.open(EVOLVE / "round1_paper.jpg")
    make_sheet(
        [(r1n, "CODE R1"), (r1p, "CODE R1")],
        [(Image.open(neon), "CODE R2"), (Image.open(paper), "CODE R2")],
        EVOLVE / "round2_sheet.jpg",
    )
    print(f"wrote {neon}")
    print(f"wrote {paper}")
    print(f"wrote {EVOLVE / 'round2_sheet.jpg'}")


def cmd_round3(duration: float = 7.0, fps: int = 30) -> None:
    EVOLVE.mkdir(parents=True, exist_ok=True)
    clips = []
    for style in ("neon", "paper"):
        path, elapsed = render_clip(
            style,
            EVOLVE / f"round3_{style}.mp4",
            duration=duration,
            fps=fps,
            reps=2,
            round_n=3,
        )
        print(f"{style} clip {path}  render={elapsed:.2f}s  ({elapsed / duration:.2f}x realtime)")
        clips.append((style, path, elapsed))
    # Contact sheet: 4 frames x 2 styles.
    frames = []
    lefts = []
    rights = []
    for style in ("neon", "paper"):
        tagged = []
        for i, t in enumerate((0.0, 1.4, 2.0, 3.5)):
            depth = squat_depth(t, duration=duration, reps=2)
            im = render_frame(style, depth, params=_params(style, 3))
            tagged.append((im, f"{style.upper()} {t:.1f}s"))
        if style == "neon":
            lefts = tagged[:2]
            # put 4 frames as 2 rows on the left? better: 2x2 neon vs paper at mid and bottom
        else:
            rights = tagged[:2]
    # A clearer 2x2: both styles at the top and at the bottom of the squat.
    neon_top = render_frame("neon", 0.0, params=_params("neon", 3))
    neon_bot = render_frame("neon", 1.0, params=_params("neon", 3))
    paper_top = render_frame("paper", 0.0, params=_params("paper", 3))
    paper_bot = render_frame("paper", 1.0, params=_params("paper", 3))
    make_sheet(
        [(neon_top, "NEON TOP"), (neon_bot, "NEON BOTTOM")],
        [(paper_top, "PAPER TOP"), (paper_bot, "PAPER BOTTOM")],
        EVOLVE / "round3_sheet.jpg",
    )
    print(f"wrote {EVOLVE / 'round3_sheet.jpg'}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Procedural neon / paper stick-figure renderer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("round1")
    sub.add_parser("round2")
    p3 = sub.add_parser("round3")
    p3.add_argument("--duration", type=float, default=7.0)
    p3.add_argument("--fps", type=int, default=30)
    still = sub.add_parser("still")
    still.add_argument("--style", choices=("neon", "paper"), required=True)
    still.add_argument("--out", type=Path, required=True)
    still.add_argument("--depth", type=float, default=1.0)
    still.add_argument("--round", dest="round_n", type=int, default=2)
    args = parser.parse_args(argv)
    if args.cmd == "round1":
        cmd_round1()
    elif args.cmd == "round2":
        cmd_round2()
    elif args.cmd == "round3":
        cmd_round3(duration=args.duration, fps=args.fps)
    elif args.cmd == "still":
        render_still(args.style, args.out, depth=args.depth, round_n=args.round_n)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
