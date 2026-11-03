#!/usr/bin/env python3
"""4-step test reel with full-ROM loops (~5 s each)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/pilot/demic/test_reel_4step.mp4"
TMP = ROOT / "output/pilot/demic/reel_concat"


def main():
    steps = [
        ("loop_wrong_front.mp4", "STEP 1 · WRONG · FRONT"),
        ("loop_right_front.mp4", "STEP 2 · RIGHT · FRONT"),
        ("loop_wrong_side.mp4", "STEP 3 · WRONG · SIDE"),
        ("loop_right_side.mp4", "STEP 4 · RIGHT · SIDE"),
    ]
    TMP.mkdir(parents=True, exist_ok=True)
    labeled = []
    for i, (name, label) in enumerate(steps):
        src = ROOT / "output/pilot/demic" / name
        if not src.is_file():
            raise SystemExit(f"missing {src} — run render_loop_demic for each step first")
        dst = TMP / f"step{i+1}.mp4"
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-vf",
                f"drawtext=text='{label}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=120",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(dst),
            ]
        )
        labeled.append(dst)
    list_file = TMP / "list.txt"
    with open(list_file, "w") as f:
        for p in labeled:
            f.write(f"file '{p}'\n")
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
            "-c",
            "copy",
            str(OUT),
        ]
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
