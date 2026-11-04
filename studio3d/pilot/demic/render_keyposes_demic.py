#!/usr/bin/env python3
"""Keyframe stills for comparison sheet (stretch + contract, both forms, both cameras)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "studio3d/pilot/blender_cli.sh"
BUILD = ROOT / "studio3d/pilot/demic/build_scene_demic.py"
OUT = ROOT / "output/pilot/demic/keyframes"


def shot(form, phase, camera, name):
    prefix = OUT / name
    subprocess.check_call(
        [
            str(CLI),
            str(BUILD),
            "--form",
            form,
            "--phase",
            str(phase),
            "--camera",
            camera,
            "--out",
            str(prefix),
            "--scale",
            "1.0",
            "--samples",
            "16",
        ]
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for form in ("wrong", "right"):
        for cam in ("front", "side"):
            shot(form, 0.0, cam, f"{form}_stretch_{cam}")
            shot(form, 1.0, cam, f"{form}_contract_{cam}")
    print("keyframes in", OUT)


if __name__ == "__main__":
    main()
