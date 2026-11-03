#!/usr/bin/env python3
"""Render a short seamless loop (phase 0→1→0) for one form + camera; ffmpeg → mp4."""
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "studio3d/pilot/blender_cli.sh"
BUILD = ROOT / "studio3d/pilot/build_scene.py"


def phase_wave(t, n):
    """Cosine ease 0→1→0 over n frames (inclusive endpoints match for loop)."""
    x = t / max(n - 1, 1)
    return 0.5 - 0.5 * math.cos(2 * math.pi * x)


def main():
    form = sys.argv[1] if len(sys.argv) > 1 else "right"
    camera = sys.argv[2] if len(sys.argv) > 2 else "front"
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    fps = 24
    tag = f"{form}_{camera}"
    out_dir = ROOT / "output/pilot/lat/loops" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        ph = phase_wave(i, frames)
        prefix = out_dir / f"frame_{i:04d}"
        subprocess.check_call(
            [
                str(CLI),
                str(BUILD),
                "--form",
                form,
                "--phase",
                f"{ph:.4f}",
                "--camera",
                camera,
                "--out",
                str(prefix),
                "--scale",
                "1.0",
                "--samples",
                "12",
                "--skip-id",
            ]
        )
    mp4 = ROOT / "output/pilot/lat" / f"loop_{tag}.mp4"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(out_dir / "frame_%04d_beauty.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(mp4),
        ]
    )
    print("wrote", mp4)


if __name__ == "__main__":
    main()
