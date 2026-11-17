"""Cover frame: the hook shot with the wrong-form cue visible (grid-readable)."""
import subprocess


def make(mp4, out_jpg, ep):
    sc = (ep.get("B2") or ep["B"])["scenes"][0]
    t = min(sc["t1"] - 0.2, sc["t0"] + 1.4)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", mp4,
                    "-frames:v", "1", "-q:v", "2", out_jpg], check=True)
    return out_jpg
