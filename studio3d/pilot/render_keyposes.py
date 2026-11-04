#!/usr/bin/env python3
"""Render all pilot key-pose stills (stretch + contract × wrong/right × front/side)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "studio3d/pilot/blender_cli.sh"
BUILD = ROOT / "studio3d/pilot/build_scene.py"
OUT = ROOT / "output/pilot/lat/keyframes"


def run(form, phase, camera, name):
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / name
    cmd = [
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
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main():
    jobs = [
        ("wrong", 0, "front", "wrong_stretch_front"),
        ("wrong", 1, "front", "wrong_contract_front"),
        ("wrong", 0, "side", "wrong_stretch_side"),
        ("wrong", 1, "side", "wrong_contract_side"),
        ("right", 0, "front", "right_stretch_front"),
        ("right", 1, "front", "right_contract_front"),
        ("right", 0, "side", "right_stretch_side"),
        ("right", 1, "side", "right_contract_side"),
    ]
    for j in jobs:
        run(*j)
    print("done", len(jobs), "frames")


if __name__ == "__main__":
    main()
