"""Narration (local Kokoro TTS) + self-synthesised CC0 SFX, loudness-normalised."""
from __future__ import annotations
import hashlib, json, os, re, subprocess
import numpy as np
import soundfile as sf

SR = 24000
# Narration engines. Kokoro is the default; the Supertonic worker is used on hosts
# without the Kokoro venv (same 24 kHz mono contract, so nothing downstream changes).
KOKORO_PY = os.environ.get("GYMFORM_KOKORO_PY", "/workspace/venv-kokoro/bin/python")
SUPERTONIC_PY = os.environ.get("GYMFORM_SUPERTONIC_PY", "/workspace/venv-supertonic/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))


def engine():
    """(name, interpreter, worker) for the narration voice available on this host."""
    want = os.environ.get("GYMFORM_TTS", "auto").lower()
    if want == "kokoro" or (want == "auto" and os.path.exists(KOKORO_PY)):
        return "kokoro", KOKORO_PY, "tts_kokoro.py"
    if want == "supertonic" or (want == "auto" and os.path.exists(SUPERTONIC_PY)):
        return "supertonic", SUPERTONIC_PY, "tts_supertonic.py"
    raise RuntimeError(f"no narration engine: set GYMFORM_TTS and GYMFORM_KOKORO_PY (missing {KOKORO_PY})")


def tts(text, voice, speed, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    name, py, worker = engine()
    key = hashlib.sha1(f"{name}|{voice}|{speed}|{text}".encode()).hexdigest()[:16]
    out = os.path.join(cache_dir, f"{key}.wav")
    if not os.path.exists(out):
        subprocess.run(["nice", "-n", "19", py, os.path.join(HERE, worker), voice, str(speed), out, text],
                       check=True, capture_output=True)
    s, sr = sf.read(out, dtype="float32")
    assert sr == SR
    # trim leading/trailing silence
    nz = np.where(np.abs(s) > 0.01)[0]
    if len(nz):
        s = s[max(nz[0] - 240, 0): nz[-1] + 1200]
    return s


def _env(n, a, r):
    e = np.ones(n, np.float32)
    na, nr = int(a * SR), int(r * SR)
    if na: e[:na] = np.linspace(0, 1, na)
    if nr: e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def _lp(x, cut):
    """One-pole low-pass with per-sample cutoff array (Hz)."""
    y = np.zeros_like(x); s = 0.0
    a = 1 - np.exp(-2 * np.pi * np.asarray(cut) / SR)
    a = np.broadcast_to(a, x.shape)
    for i in range(len(x)):
        s += a[i] * (x[i] - s); y[i] = s
    return y


def sfx(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "whoosh":
        n = int(0.38 * SR); t = np.arange(n) / SR
        cut = 300 + 4200 * np.sin(np.pi * t / t[-1]) ** 2
        x = _lp(rng.standard_normal(n).astype(np.float32), cut) - _lp(rng.standard_normal(n).astype(np.float32), 120)
        x = x * np.sin(np.pi * t / t[-1]) ** 1.5
        return 0.9 * x / (np.abs(x).max() + 1e-9)
    if kind == "pop":
        n = int(0.11 * SR); t = np.arange(n) / SR
        f = 180 + 520 * np.exp(-t * 45)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.002, 0.08)
        return 0.8 * x.astype(np.float32)
    if kind == "tick":
        n = int(0.03 * SR); t = np.arange(n) / SR
        x = (np.sin(2 * np.pi * 2600 * t) + 0.3 * rng.standard_normal(n)) * np.exp(-t * 180)
        return 0.5 * x.astype(np.float32)
    if kind == "thump":
        n = int(0.16 * SR); t = np.arange(n) / SR
        f = 48 + 70 * np.exp(-t * 30)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.003, 0.12)
        return 0.9 * x.astype(np.float32)
    if kind == "kick":
        n = int(0.22 * SR); t = np.arange(n) / SR
        f = 45 + 110 * np.exp(-t * 35)
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14)).astype(np.float32)
    if kind == "hat":
        n = int(0.05 * SR); x = rng.standard_normal(n).astype(np.float32)
        x = x - _lp(x, 6000)
        return 0.35 * x * np.exp(-np.arange(n) / SR * 90)
    raise ValueError(kind)


def place(buf, clip, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(clip))
    buf[i:j] += gain * clip[: j - i]


def measure(path):
    """(integrated LUFS, true peak dBFS) of a file, measured on the decoded stream."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    i = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", p.stderr)[-1])
    tp = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", p.stderr)[-1])
    return i, tp


def final_trim(mp4, target=-14.0, tol=0.3, max_tp=-1.5, tries=2):
    """loudnorm in linear mode lands near -14 LUFS but not on it (the stereo pan and the
    codec both move it a little), so the muxed file is measured and trimmed to the target.
    Video is copied; only the audio is re-encoded."""
    i, tp = measure(mp4)
    for _ in range(tries):
        d = target - i
        if abs(d) <= tol:
            break
        gain = min(d, max(max_tp - tp, -12.0)) if tp + d > max_tp else d
        tmp = mp4 + ".trim.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", mp4, "-c:v", "copy",
                        "-af", f"volume={gain:.2f}dB", "-c:a", "aac", "-profile:a", "aac_low", "-ar", "48000",
                        "-b:a", "192k", "-movflags", "+faststart", tmp], check=True)
        os.replace(tmp, mp4)
        i, tp = measure(mp4)
    return i, tp


def bed(duration, bpm=104):
    """Thin CC0 rhythm loop, ducked later under speech. Kick on the beat, shaker on
    the offbeat, quiet 16th ticks, and a sustained low pad so the track never falls
    to digital silence in the gaps between narration lines (a silent hole reads as a
    broken file on a phone, and IG's own audio never has one). Snare-free and quiet
    enough to stay under the voice."""
    b = np.zeros(int(duration * SR), np.float32)
    beat = 60.0 / bpm
    k, h, tk = sfx("kick"), sfx("hat", 3), sfx("tick", 5)
    t = 0.0; i = 0
    while t < duration:
        if i % 2 == 0:
            place(b, k, t, 0.5)
        place(b, h, t + beat / 2, 0.4)
        place(b, tk, t + beat / 4, 0.09)
        place(b, tk, t + 3 * beat / 4, 0.07)
        t += beat; i += 1
    tt = np.arange(len(b)) / SR
    lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.13 * tt)
    pad = (np.sin(2 * np.pi * 110.0 * tt) + 0.8 * np.sin(2 * np.pi * 164.81 * tt)
           + 0.45 * np.sin(2 * np.pi * 220.0 * tt)) * lfo * 0.08
    b += pad.astype(np.float32)
    return b


def build(ep_style, ep, events, duration, cache_dir, out_wav, use_bed=True):
    """events: list of (t, kind). Returns dict with narration timing checks."""
    n = int(duration * SR)
    voice = np.zeros(n, np.float32); fx = np.zeros(n, np.float32)
    lines = ep_style["narration"]; report = []
    for idx, ln in enumerate(lines):
        s = tts(ln["text"], ep.get("voice", "am_michael"), ep.get("speed", 1.12), cache_dir)
        end = ln["t"] + len(s) / SR
        nxt = lines[idx + 1]["t"] if idx + 1 < len(lines) else duration
        report.append({"text": ln["text"], "start": ln["t"], "end": round(end, 2), "fits": end <= nxt - 0.03})
        place(voice, s, ln["t"])
    for i, (t, kind) in enumerate(events):
        place(fx, sfx(kind, i), t, {"whoosh": 0.35, "pop": 0.4, "tick": 0.25, "thump": 0.35}[kind])
    mix = voice * 1.0 + fx
    if use_bed:
        bd = bed(duration)
        env = np.convolve((np.abs(voice) > 0.02).astype(np.float32), np.ones(2400) / 2400, mode="same")
        duck = 1.0 - 0.6 * np.clip(env * 3, 0, 1)
        mix += 0.16 * bd * duck
    # fade edges
    f = int(0.02 * SR); mix[:f] *= np.linspace(0, 1, f); mix[-f:] *= np.linspace(1, 0, f)
    raw0 = out_wav.replace(".wav", "_pre.wav")
    sf.write(raw0, mix, SR)
    raw = out_wav.replace(".wav", "_raw.wav")
    # tame peaks first so linear loudnorm can reach -14 LUFS under a -1.5 dBTP ceiling
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", "-i", raw0, "-af",
                    "acompressor=threshold=-20dB:ratio=3:attack=4:release=90:makeup=4,alimiter=limit=0.5:level=false",
                    raw], check=True)
    os.remove(raw0)
    # two-pass loudnorm to -14 LUFS
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", raw, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S).group(0))
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", "-i", raw, "-af", af, "-ar", "48000", out_wav], check=True)
    return report
