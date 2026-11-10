#!/usr/bin/env python3
"""4-segment reel composer (shared, not per-exercise).

Order (Day 6+ / Demic-mapped): 0–S wrong bone (B x-ray) · S–2S right bone (B) ·
2S–3S wrong muscle (H red compensators) · 3S–4S right muscle (COMBO green).
Default S=5 s → 20 s master.

One retarget per take is reused; this script only grades (if needed) and muxes.
Callouts stay on the Day-4 1080×1920 canvas (grade3d out_res); the encode scales
to --res (default 720x1280). Caption lint + optional edge-tts use gymform rules.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, shutil
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def _run(cmd, logf=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if logf:
        open(logf, "a").write(p.stdout or "")
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-2000:]}")
    return p


def _grade(rdir, odir, cfgp, method):
    os.makedirs(odir, exist_ok=True)
    existing = sorted(f for f in os.listdir(odir) if f.startswith("g_") and f.endswith(".png"))
    src = sorted(f for f in os.listdir(rdir) if f.startswith("f_") and f.endswith(".png"))
    if existing and len(existing) >= len(src) > 0:
        print(f"grade skip {odir} ({len(existing)} g_ already)")
        return
    cmd = [sys.executable, os.path.join(HERE, "grade3d.py"), rdir, odir, cfgp, "f", method]
    print("grade", method, "<-", rdir, "->", odir)
    t0 = time.time()
    _run(cmd, os.path.join(odir, "grade.log"))
    print(f"  graded {method} in {time.time()-t0:.1f}s")


def _write_method_cfg(src_cfg, dest, method, glow):
    cfg = json.load(open(src_cfg))
    cfg = dict(cfg); cfg["grade"] = dict(cfg.get("grade") or {})
    cfg["grade"]["glow_method"] = method
    cfg["grade"]["glow"] = glow
    cfg["grade"]["look"] = cfg["grade"].get("look") or "clay"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(cfg, open(dest, "w"), indent=1)
    return dest


def _encode_segment(gdir, mp4, seg_s, W, H):
    frames = sorted(f for f in os.listdir(gdir) if f.startswith("g_") and f.endswith(".png"))
    if not frames:
        raise SystemExit(f"no graded frames in {gdir}")
    n = len(frames)
    fps_src = n / float(seg_s)
    # g_NNNN.png — use the first file's number as -start_number
    start = int(frames[0][2:6])
    pattern = os.path.join(gdir, "g_%04d.png")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-threads", "2",
           "-framerate", f"{fps_src:.6f}", "-start_number", str(start), "-i", pattern,
           "-vf", f"fps=30,scale={W}:{H}:flags=lanczos,format=yuv420p",
           "-t", f"{seg_s:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-an", "-movflags", "+faststart", mp4]
    _run(cmd, os.path.join(os.path.dirname(mp4), "ffmpeg_seg.log"))


def _tts_and_caption(out, cfg, total_s):
    from gymform import caption
    cap = caption.build(cfg.get("caption") or {
        "title": "One-arm dumbbell row: drive the elbow, not the hand.",
        "question": "Where do you feel your dumbbell rows most right now?",
        "save": "Save this for your next back day.",
    })
    errs = caption.lint(cap)
    reel = os.path.join(out, "reel")
    os.makedirs(reel, exist_ok=True)
    open(os.path.join(reel, "caption.txt"), "w").write(cap)
    json.dump({"errors": errs}, open(os.path.join(reel, "caption_lint.json"), "w"))
    tts_info = {"segments": [], "end_s": 0.0, "voice": None, "ok": False}
    wav_path = os.path.join(reel, "audio.wav")
    try:
        import asyncio, wave
        import edge_tts
        from gymform.tts import VOICE, CLEAN
        T = cfg.get("tts") or {}
        texts = [T.get("line") or "Pull the elbow back to your hip. Feel your lats and mid back squeeze.",
                 T.get("question_spoken") or "Where do you feel it most?"]
        cache = os.path.join(ROOT, "_cache_rig3d", "tts"); os.makedirs(cache, exist_ok=True)
        wavs = []; segs = []; t0 = 0.25
        for tx in texts:
            import hashlib
            k = hashlib.sha1((tx + json.dumps(VOICE, sort_keys=True)).encode()).hexdigest()[:16]
            wp = os.path.join(cache, k + ".wav")
            if not os.path.exists(wp):
                mp3 = wp[:-4] + ".mp3"
                asyncio.run(edge_tts.Communicate(tx, **VOICE).save(mp3))
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-threads", "2",
                                "-i", mp3, "-af", CLEAN, "-ac", "1", "-ar", "44100", wp], check=True)
            with wave.open(wp, "rb") as wf:
                w = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768.0
            wavs.append(w)
            segs.append({"text": tx, "start": round(t0, 3), "dur": round(len(w) / 44100, 3)})
            t0 += len(w) / 44100 + 0.3
        sr = 44100; n = int(sr * total_s); mix = np.zeros(n, np.float32)
        for s, w in zip(segs, wavs):
            a = int(s["start"] * sr); b = min(n, a + len(w)); mix[a:b] += w[:b - a] * 0.92
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(np.repeat((np.clip(mix, -0.96, 0.96) * 32767).astype(np.int16), 2).tobytes())
        tts_info = {"segments": segs, "end_s": round(segs[-1]["start"] + segs[-1]["dur"], 3),
                    "voice": "en-US-ChristopherNeural", "ok": True, "dur_s": total_s}
    except Exception as e:
        # silent stereo bed so the mux still has an audio track
        sr = 44100; n = int(sr * total_s)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", f"anullsrc=r={sr}:cl=stereo", "-t", f"{total_s:.3f}",
                        "-c:a", "pcm_s16le", wav_path], check=True)
        tts_info = {"segments": [], "end_s": 0.0, "voice": None, "ok": False, "error": str(e), "dur_s": total_s}
    json.dump(tts_info, open(os.path.join(reel, "tts.json"), "w"), indent=1)
    return cap, errs, wav_path, tts_info


def _contact_sheet(segments, sheet_path, start_i=0, peak_is=None):
    """2×4 start/peak contact sheet from the four graded directories."""
    tiles = []
    labels = []
    for i, (name, gdir, _method) in enumerate(segments):
        frames = sorted(f for f in os.listdir(gdir) if f.startswith("g_") and f.endswith(".png"))
        if not frames:
            continue
        peak_i = (peak_is or [len(frames)//2] * 4)[i]
        peak_i = max(0, min(len(frames) - 1, peak_i))
        for tag, idx in (("start", start_i), ("peak", peak_i)):
            im = cv2.imread(os.path.join(gdir, frames[idx]))
            tiles.append(im); labels.append(f"{name} · {tag}")
    if not tiles:
        raise SystemExit("no tiles for contact sheet")
    th, tw = 640, 360
    row_n = 2; col_n = 4
    canvas = np.full((th * row_n + 80, tw * col_n, 3), 28, np.uint8)
    # tiles order: seg0 start, seg0 peak, seg1 start, ... → rearrange to row0=starts, row1=peaks
    starts = tiles[0::2]; peaks = tiles[1::2]
    sl = labels[0::2]; pl = labels[1::2]
    def blit(img, r, c, lab):
        x, y = c * tw, r * th + 48
        rs = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        canvas[y:y+th, x:x+tw] = rs
        cv2.putText(canvas, lab, (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    for c, (im, lab) in enumerate(zip(starts, sl)):
        blit(im, 0, c, lab)
    for c, (im, lab) in enumerate(zip(peaks, pl)):
        blit(im, 1, c, lab)
    cv2.putText(canvas, "db_row 4-seg smoke  ·  wrong B  |  right B  |  wrong H  |  right COMBO",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(sheet_path) or ".", exist_ok=True)
    cv2.imwrite(sheet_path, canvas)
    return sheet_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrong", required=True, help="wrong-form render dir (f_/m_/id_ + cfg.json)")
    ap.add_argument("--right", required=True, help="right-form render dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg-s", type=float, default=5.0)
    ap.add_argument("--res", default="720x1280")
    ap.add_argument("--look", default="clay")
    args = ap.parse_args()
    W, H = map(int, args.res.lower().split("x"))
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    t_all = time.time()
    wrong_cfg = os.path.join(args.wrong, "cfg.json")
    right_cfg = os.path.join(args.right, "cfg.json")
    cfg = json.load(open(right_cfg))

    segspec = [
        ("wrong_bone", args.wrong, os.path.join(out, "wrong", "grade_B"), "B", "wrong", wrong_cfg),
        ("right_bone", args.right, os.path.join(out, "right", "grade_B"), "B", "right", right_cfg),
        ("wrong_muscle", args.wrong, os.path.join(out, "wrong", "grade_H"), "H", "wrong", wrong_cfg),
        ("right_muscle", args.right, os.path.join(out, "right", "grade_COMBO"), "COMBO", "right", right_cfg),
    ]
    grade_s = {}
    for name, rdir, gdir, method, glow, src_cfg in segspec:
        cfgp = os.path.join(gdir, "cfg.json")
        t0 = time.time()
        _write_method_cfg(src_cfg, cfgp, method, glow)
        # honour --look
        c = json.load(open(cfgp)); c["grade"]["look"] = args.look
        json.dump(c, open(cfgp, "w"), indent=1)
        _grade(rdir, gdir, cfgp, method)
        grade_s[name] = round(time.time() - t0, 1)

    reel_dir = os.path.join(out, "reel"); os.makedirs(reel_dir, exist_ok=True)
    clip_paths = []
    for name, _r, gdir, method, _g, _c in segspec:
        mp4 = os.path.join(reel_dir, f"{name}.mp4")
        _encode_segment(gdir, mp4, args.seg_s, W, H)
        clip_paths.append(mp4)

    total_s = args.seg_s * 4
    _cap, errs, wav, tts_info = _tts_and_caption(out, cfg, total_s)
    concat_list = os.path.join(reel_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    master = os.path.join(out, "db_row_seg4_smoke.mp4")
    tmpv = os.path.join(reel_dir, "video.mp4")
    _run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
          "-c", "copy", tmpv], os.path.join(reel_dir, "ffmpeg.log"))
    _run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmpv, "-i", wav,
          "-c:v", "libx264", "-preset", "medium", "-crf", "18",
          "-vf", f"fps=30,scale={W}:{H}:flags=lanczos,format=yuv420p",
          "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", master],
         os.path.join(reel_dir, "ffmpeg.log"))

    # peak index per take from checks.json (working elbow min)
    peak_is = []
    for name, rdir, gdir, method, _g, _c in segspec:
        frames = sorted(f for f in os.listdir(gdir) if f.startswith("g_") and f.endswith(".png"))
        nums = [int(f[2:6]) for f in frames]
        pf = nums[len(nums)//2]
        chp = os.path.join(rdir, "checks.json")
        if os.path.exists(chp):
            R = json.load(open(chp))["frames"]
            pf = int(min(R, key=lambda r: r["elbow_deg"])["f"])
        peak_is.append(int(np.argmin([abs(n - pf) for n in nums])))
    sheet = os.path.join(out, "sheet.png")
    _contact_sheet([(n, g, m) for n, _r, g, m, _gl, _c in segspec], sheet, 0, peak_is)

    meta = {
        "mp4": master, "sheet": sheet, "dur_s": total_s, "res": [W, H],
        "seg_s": args.seg_s, "grade_s": grade_s, "caption_errors": errs,
        "tts": tts_info, "wall_s": round(time.time() - t_all, 1),
        "order": [s[0] for s in segspec],
    }
    json.dump(meta, open(os.path.join(out, "compose.json"), "w"), indent=1)
    print("wrote", master, "sheet", sheet, "in", meta["wall_s"], "s")


if __name__ == "__main__":
    main()
