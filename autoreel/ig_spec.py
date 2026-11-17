"""Instagram Reels container and encode spec.

One source of truth for the renderer (pilots/make_reel.py), the QA gate `ig_spec`,
and publish_run prechecks. No per-episode overrides.

Encode: 1080x1920 30fps H.264 High yuv420p 7 Mbps CBR (VBV 14 Mbps),
AAC-LC 48 kHz stereo 192k, +faststart (moov before mdat). Loudness stays -14 LUFS
(handled in pilots/audio.py; the stereo pan is -3 dB/ch so mux does not raise it).
"""
from __future__ import annotations
import json
import os
import struct
import subprocess

W, H = 1080, 1920
FPS = 30
VIDEO_BITRATE = "7M"
VIDEO_MINRATE = "7M"
VIDEO_MAXRATE = "7M"
VIDEO_BUFSIZE = "14M"
AUDIO_BITRATE = "192k"
AUDIO_RATE = 48000
MIN_VIDEO_BPS = 4_000_000
MIN_DURATION_S = 3.0
MAX_DURATION_S = 90.0
MAX_BYTES = 300 * 1024 * 1024  # 300 MB
AUDIO_RATES = {44100, 48000}

# ffmpeg args after the raw-frame input: encode the silent video
VIDEO_ENCODE = [
    "-c:v", "libx264",
    "-preset", "medium",
    "-profile:v", "high",
    "-pix_fmt", "yuv420p",
    "-b:v", VIDEO_BITRATE,
    "-minrate", VIDEO_MINRATE,
    "-maxrate", VIDEO_MAXRATE,
    "-bufsize", VIDEO_BUFSIZE,
    "-x264-params", "nal-hrd=cbr:force-cfr=1",  # stickman scenes undershoot ABR; CBR hits 7 Mbps
    "-movflags", "+faststart",
]

# ffmpeg args to mux loudnormed wav onto the silent video (video is copied)
# stereo pan at -3 dB/ch keeps integrated loudness at -14 LUFS
AUDIO_MUX = [
    "-c:v", "copy",
    "-c:a", "aac",
    "-profile:a", "aac_low",
    "-af", "pan=stereo|c0=0.7071*c0|c1=0.7071*c0",
    "-ar", str(AUDIO_RATE),
    "-b:a", AUDIO_BITRATE,
    "-shortest",
    "-movflags", "+faststart",
]


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fps(rate):
    if not rate or rate in ("0/0", "N/A"):
        return None
    if "/" in str(rate):
        a, b = str(rate).split("/", 1)
        den = _num(b, 0)
        return (_num(a, 0) / den) if den else None
    return _num(rate)


def moov_before_mdat(path):
    """True when the moov atom appears before mdat (faststart)."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return False
            size, typ = struct.unpack(">I4s", hdr)
            if typ == b"moov":
                return True
            if typ == b"mdat":
                return False
            if size == 1:
                wide = f.read(8)
                if len(wide) < 8:
                    return False
                size = struct.unpack(">Q", wide)[0]
                if size < 16:
                    return False
                f.seek(size - 16, os.SEEK_CUR)
            elif size < 8:
                return False
            else:
                f.seek(size - 8, os.SEEK_CUR)


def probe(mp4):
    p = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            "format=duration,size,bit_rate"
            ":stream=codec_name,codec_type,profile,width,height,"
            "r_frame_rate,avg_frame_rate,pix_fmt,channels,channel_layout,"
            "sample_rate,bit_rate",
            "-of", "json", mp4,
        ],
        capture_output=True, text=True,
    )
    if p.returncode != 0 or not (p.stdout or "").strip():
        raise RuntimeError((p.stderr or "ffprobe failed").strip()[:300])
    return json.loads(p.stdout)


def check_ig_spec(mp4):
    """Return (ok, detail) for the final mp4. Detail lists every check."""
    if not os.path.isfile(mp4):
        return False, "mp4 missing"
    try:
        pr = probe(mp4)
    except Exception as e:  # noqa: BLE001
        return False, f"ffprobe failed: {e}"
    streams = pr.get("streams") or []
    fmt = pr.get("format") or {}
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    size = os.path.getsize(mp4)
    dur = _num(fmt.get("duration"))
    fails, notes = [], []

    def see(key, ok, got):
        (notes if ok else fails).append(f"{key}={got}")

    if v is None:
        see("video", False, "missing")
        w = h = fps = v_bps = None
    else:
        w, h = v.get("width"), v.get("height")
        fps = _fps(v.get("r_frame_rate")) or _fps(v.get("avg_frame_rate"))
        v_bps = _num(v.get("bit_rate"))
        see("1080x1920_9:16", w == W and h == H and h and abs(w / h - 9 / 16) < 0.001, f"{w}x{h}")
        see("fps30", fps is not None and abs(fps - FPS) < 0.05, fps)
        see("h264", (v.get("codec_name") or "").lower() in ("h264", "avc1", "avc"), v.get("codec_name"))
        see("yuv420p", v.get("pix_fmt") == "yuv420p", v.get("pix_fmt"))
    if a is None:
        see("audio", False, "missing")
        a_rate = a_ch = None
    else:
        a_rate = int(_num(a.get("sample_rate"), 0) or 0)
        a_ch = int(_num(a.get("channels"), 0) or 0)
        see("aac", (a.get("codec_name") or "").lower() == "aac", a.get("codec_name"))
        see("stereo", a_ch == 2, a_ch)
        see("sr_44_1_or_48k", a_rate in AUDIO_RATES, a_rate)
    if v_bps is None and dur and dur > 0:
        # fall back: container bitrate minus audio (or just size/duration)
        a_bps = _num((a or {}).get("bit_rate")) or 0
        fmt_bps = _num(fmt.get("bit_rate"))
        v_bps = (fmt_bps - a_bps) if fmt_bps else (size * 8 / dur - a_bps)
    see("video_bitrate>=4Mbps", v_bps is not None and v_bps >= MIN_VIDEO_BPS,
        f"{v_bps / 1e6:.2f}Mbps" if v_bps is not None else "unknown")
    see("duration_3_90s", dur is not None and MIN_DURATION_S <= dur <= MAX_DURATION_S,
        f"{dur:.3f}s" if dur is not None else "unknown")
    see("file<=300MB", size <= MAX_BYTES, f"{size / (1024 * 1024):.2f}MB")
    try:
        fast = moov_before_mdat(mp4)
    except OSError as e:
        fast = False
        see("faststart", False, str(e))
    else:
        see("faststart", fast, "moov_before_mdat" if fast else "moov_after_mdat")

    if fails:
        detail = "FAIL " + "; ".join(fails)
        if notes:
            detail += " | " + ", ".join(notes)
    else:
        detail = "OK " + ", ".join(notes)
    return not fails, detail


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: python -m autoreel.ig_spec <mp4>")
    ok, detail = check_ig_spec(sys.argv[1])
    print(detail)
    sys.exit(0 if ok else 1)
