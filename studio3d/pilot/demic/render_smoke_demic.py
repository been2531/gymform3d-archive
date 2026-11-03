#!/usr/bin/env python3
"""Four cheap smoke frames at 540x960 (scale 0.5), low samples."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "studio3d/pilot/blender_cli.sh"
BUILD = ROOT / "studio3d/pilot/demic/build_scene_r6.py"
OUT = ROOT / "output/pilot/demic/smoke"


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
            "0.5",
            "--samples",
            "8",
        ]
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shots = [
        ("wrong", 0.0, "front", "wrong_front_stretch"),
        ("wrong", 1.0, "side", "wrong_side_contract"),
        ("right", 0.0, "front", "right_front_stretch"),
        ("right", 1.0, "side", "right_side_contract"),
    ]
    for form, phase, cam, name in shots:
        shot(form, phase, cam, name)
    print("smoke frames in", OUT)


if __name__ == "__main__":
    main()
