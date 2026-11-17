"""make_reel.py --style A|B --episode episodes/x.json [--out DIR]

Pure code: one JSON in, MP4 + 12-frame flow sheet + qa.json out.
No generation per episode; TTS is local Kokoro (Apache-2.0)."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np
import skia

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilots import audio, qa  # noqa: E402

W, H = 1080, 1920


def load_style(name, ep):
    if name == "A":
        from pilots.style_a import StyleA
        return StyleA(ep)
    if name == "B2":
        from pilots.style_b2 import StyleB2
        return StyleB2(ep)
    from pilots.style_b import StyleB
    return StyleB(ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, choices=["A", "B", "B2"])
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=0, help="debug: render only N frames")
    a = ap.parse_args()
    ep = json.load(open(a.episode))
    fps, dur = ep["fps"], ep["duration"]
    n = int(round(fps * dur)) if not a.frames else a.frames
    out_dir = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "pilots", time.strftime("%Y-%m-%d"))
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{ep['id']}_{'B_v2' if a.style == 'B2' else a.style}"
    sec = ep.get(a.style) or ep["B"]
    silent = os.path.join(out_dir, f"_{stem}_silent.mp4")
    mp4 = os.path.join(out_dir, f"{stem}.mp4")
    t0 = time.time()
    style = load_style(a.style, ep)
    enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
                            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent], stdin=subprocess.PIPE)
    surf = skia.Surface(W, H)
    metas, small, sheet_frames = [], [], {}
    sheet_idx = {int((i + 0.5) * n / 12): i for i in range(12)}
    for f in range(n):
        t = f / fps
        cv = surf.getCanvas()
        meta = {"t": t, "joints": [], "text": []}
        style.render(cv, t, meta)
        img = surf.makeImageSnapshot()
        arr = img.toarray(colorType=skia.kRGBA_8888_ColorType)
        enc.stdin.write(arr.tobytes())
        metas.append(meta)
        g = arr[::4, ::4, :3].astype(np.int16)
        small.append(g)
        if f in sheet_idx:
            sheet_frames[sheet_idx[f]] = (t, arr[:, :, :3].copy())
        if f == 0:
            bottom0 = arr[int(H * 0.85):, :, :3].copy()
    enc.stdin.close(); enc.wait()
    render_s = time.time() - t0
    # audio
    wav = os.path.join(out_dir, f"_{stem}.wav")
    narr = audio.build(sec, ep, style.events(), dur, os.path.join(out_dir, "_tts_cache"), wav,
                       use_bed=sec.get("bed", True))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", mp4], check=True)
    total_s = time.time() - t0
    # flow sheet
    sheet = os.path.join(out_dir, f"{stem}_sheet.jpg")
    qa.flow_sheet([sheet_frames[i] for i in range(len(sheet_frames))], sheet)
    report = qa.run(a.style, ep, mp4, metas, small, style.bg_img.toarray(colorType=skia.kRGBA_8888_ColorType)[:, :, :3],
                    narr, style.body)
    report.update({"render_video_s": round(render_s, 1), "total_s": round(total_s, 1), "mp4": mp4, "sheet": sheet})
    json.dump(report, open(os.path.join(out_dir, f"{stem}_qa.json"), "w"), indent=2, default=float)
    for p in (silent, wav, wav.replace(".wav", "_raw.wav")):
        if os.path.exists(p): os.remove(p)
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=1, default=float))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
