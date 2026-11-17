"""Narration + synthesised transition sounds, mixed and loudness-normalised."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100

FFMPEG = "/workspace/.local/lib/python3.12/site-packages/static_ffmpeg/bin/linux/ffmpeg"
FFPROBE = "/workspace/.local/lib/python3.12/site-packages/static_ffmpeg/bin/linux/ffprobe"


# ------------------------------------------------------------------ utils ---

def _env(n: int, a: float, d: float, curve: float = 2.0) -> np.ndarray:
    """Attack/decay envelope over n samples."""
    na, nd = int(a * SR), int(d * SR)
    e = np.zeros(n, np.float32)
    if na:
        e[:na] = np.linspace(0, 1, na) ** 0.6
    hold = min(n, na + nd)
    if hold > na:
        e[na:hold] = np.exp(-np.linspace(0, curve * 3, hold - na))
    return e


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, n).astype(np.float32)


def _band(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    ny = SR / 2
    lo = max(20.0, min(lo, ny * 0.98))
    hi = max(lo * 1.2, min(hi, ny * 0.98))
    sos = butter(order, [lo / ny, hi / ny], btype="band", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def sfx_whoosh(dur: float = 0.6, seed: int = 3) -> np.ndarray:
    n = int(dur * SR)
    x = _noise(n, seed)
    blocks, curves = 10, []
    seg = n // blocks
    for i in range(blocks):
        c = 260 * (4200 / 260) ** (i / (blocks - 1))
        s = _band(x, c * 0.55, c * 1.85)
        curves.append(s)
    out = np.zeros(n, np.float32)
    win = np.hanning(seg * 2)
    for i, s in enumerate(curves):
        a = i * seg
        bnd = min(n, a + seg * 2)
        w = win[:bnd - a]
        out[a:bnd] += s[a:bnd] * w[:bnd - a]
    env = np.concatenate([np.linspace(0, 1, int(n * 0.55)) ** 2,
                          np.linspace(1, 0, n - int(n * 0.55)) ** 1.4])
    return out * env * 0.9


def sfx_impact(dur: float = 0.55, seed: int = 5) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * (58 * t + 26 * t * t)) * np.exp(-t * 7.0)
    click = _band(_noise(n, seed), 180, 2600) * np.exp(-t * 55) * 0.5
    return (body * 0.95 + click).astype(np.float32)


def sfx_pop(dur: float = 0.18, seed: int = 7) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 330 * np.exp(-t * 9) + 120
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 16)
    tick = _band(_noise(n, seed), 900, 5200) * np.exp(-t * 90) * 0.35
    return (body * 0.8 + tick).astype(np.float32)


def sfx_ding(dur: float = 1.0, seed: int = 9) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    partials = [(1568, 1.0, 3.2), (2349, 0.5, 4.4), (3136, 0.28, 6.0)]
    out = np.zeros(n, np.float32)
    for f, g, dcy in partials:
        out += (np.sin(2 * np.pi * f * t) * g * np.exp(-t * dcy)).astype(np.float32)
    return out / 1.75


def sfx_riser(dur: float = 0.9, seed: int = 11) -> np.ndarray:
    n = int(dur * SR)
    x = _noise(n, seed)
    out = _band(x, 400, 3000) * np.linspace(0, 1, n) ** 2.2
    t = np.arange(n) / SR
    out += (np.sin(2 * np.pi * (180 + 900 * (t / dur) ** 2) * t) *
            np.linspace(0, 0.35, n) ** 2).astype(np.float32)
    return out.astype(np.float32)


def sfx_tick(dur: float = 0.07, seed: int = 13) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (_band(_noise(n, seed), 2600, 9000) * np.exp(-t * 120) * 0.6).astype(np.float32)


SFX = {
    "whoosh": sfx_whoosh,
    "impact": sfx_impact,
    "pop": sfx_pop,
    "ding": sfx_ding,
    "riser": sfx_riser,
    "tick": sfx_tick,
}


# ------------------------------------------------------------------- tts ----

def _hash(text: str, voice: str, speed: float, steps: int) -> str:
    h = hashlib.sha1(f"{text}|{voice}|{speed}|{steps}".encode()).hexdigest()[:10]
    return h


def synth_line(cfg: dict, text: str, out_wav: str) -> str:
    a = cfg["audio"]
    h = _hash(text, a["voice"], a["speed"], a["steps"])
    path = os.path.join(os.path.dirname(out_wav), f"{h}.wav")
    manifest = out_wav + ".json"
    seen = json.load(open(manifest)) if os.path.exists(manifest) else {}
    if os.path.exists(path) and seen.get(path) == text:
        return path
    cmd = [a["python"], "tts", text, "-o", path, "--voice", a["voice"],
           "--speed", str(a["speed"]), "--steps", str(a["steps"]),
           "--model", "supertonic-3", "--lang", a["lang"]]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seen[path] = text
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    json.dump(seen, open(manifest, "w"), ensure_ascii=False, indent=1)
    return path


def read_wav(path: str) -> np.ndarray:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    else:
        x = data.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import scipy.signal as ss
        n = int(round(len(x) * SR / sr))
        x = ss.resample(x, n).astype(np.float32)
    return x


# ------------------------------------------------------------------ build ---

def trim_silence(x: np.ndarray, thresh: float = 0.006, pad: float = 0.02) -> np.ndarray:
    """Trim leading/trailing silence so a line lands exactly at its `at` time."""
    if x.size == 0:
        return x
    nz = np.flatnonzero(np.abs(x) > thresh)
    if nz.size == 0:
        return x
    a = max(0, nz[0] - int(pad * SR))
    b = min(len(x), nz[-1] + int(pad * SR))
    return x[a:b].copy()


def build(cfg: dict, root: str, out_wav: str) -> str:
    dur = cfg["format"]["duration"]
    n = int(dur * SR)
    voice = np.zeros(n, np.float32)
    fx = np.zeros(n, np.float32)
    do_trim = bool(cfg["audio"].get("trim_lead", False))

    ttsdir = os.path.join(os.path.dirname(out_wav), "tts")
    os.makedirs(ttsdir, exist_ok=True)

    lines = cfg["audio"]["lines"]
    for i, line in enumerate(lines):
        path = synth_line(cfg, line["text"], os.path.join(ttsdir, "line.wav"))
        x = read_wav(path)
        if do_trim:
            x = trim_silence(x)
        s = int(round(line["at"] * SR))
        limit = n - s
        if i + 1 < len(lines):
            limit = min(limit, int(round(lines[i + 1]["at"] * SR)) - s)
        x = x[:max(0, limit)].copy()
        fade = int(0.010 * SR)
        if len(x) > fade:
            x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        end = min(n, s + len(x))
        voice[s:end] += x[:end - s] * 1.0
        print(f"  voice {line['id']}: {len(x)/SR:.2f}s at {line['at']:.2f}s")

    for ev in cfg["audio"]["sfx"]:
        fn = SFX[ev["type"]]
        x = fn()
        s = int(round(ev["at"] * SR))
        end = min(n, s + len(x))
        fx[s:end] += x[:end - s] * ev.get("gain", 0.5)

    bed = _bed(dur, cfg["audio"].get("bed", {}))
    mix = voice * 1.0 + fx * 0.85 + bed * cfg["audio"].get("bed", {}).get("level", 0.3)

    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 0.98:
        mix *= 0.98 / peak

    tmp = out_wav.replace(".wav", "_raw.wav")
    wavfile.write(tmp, SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    _loudnorm(tmp, out_wav, cfg["audio"].get("target_lufs", -14.0), dur)
    return out_wav


def _bed(dur: float, cfg: dict) -> np.ndarray:
    """Half-time kick + shaker, deliberately quiet: it should be felt, not heard."""
    n = int(dur * SR)
    out = np.zeros(n, np.float32)
    tempo = cfg.get("tempo", 120.0)
    beat = 60.0 / tempo
    t = 0.0
    i = 0
    while t < dur - 0.05:
        if i % 4 in (0, 2):
            k = sfx_impact(0.30, seed=100 + i)
            s = int(t * SR)
            e = min(n, s + len(k))
            out[s:e] += k[:e - s] * 0.55
        t += beat * 0.5
        i += 1
    # shaker on every 8th
    t = beat * 0.5
    while t < dur - 0.05:
        sh = _band(_noise(int(0.05 * SR), 999), 4500, 12000) * np.exp(-np.arange(int(0.05 * SR)) / SR * 90)
        s = int(t * SR)
        e = min(n, s + len(sh))
        out[s:e] += sh[:e - s] * 0.10
        t += beat * 0.5
    return out


def _measure_i(path: str) -> float:
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path, "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    vals = [ln.split(":")[1].strip() for ln in r.stderr.splitlines()
            if ln.strip().startswith("I:") and "LUFS" in ln]
    return float(vals[-1].split()[0]) if vals else 0.0


def _loudnorm(src: str, dst: str, lufs: float, dur: float) -> None:
    """Measure, apply linear gain into a limiter, then trim/pad to an exact length."""
    i0 = _measure_i(src)
    gain = lufs - i0
    af = (f"volume={gain:.2f}dB,alimiter=limit=0.84:attack=3:release=80:level=false,"
          f"apad,atrim=0:{dur},asetpts=N/SR/TB")
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-af", af, "-ar", str(SR), "-ac", "2", dst], check=True)
    i1 = _measure_i(dst)
    d = lufs - i1
    if abs(d) > 0.3:
        tmp = dst + ".tmp.wav"
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", dst,
                        "-af", f"volume={d:.2f}dB,alimiter=limit=0.84:attack=3:release=80:level=false,"
                               f"apad,atrim=0:{dur},asetpts=N/SR/TB",
                        "-ar", str(SR), "-ac", "2", tmp], check=True)
        os.replace(tmp, dst)
        print(f"  loudness {i0:.1f} -> {i1:.1f} -> {_measure_i(dst):.1f} LUFS")


def mux(video: str, audio: str, out: str, dur: float) -> str:
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video, "-i", audio,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                    "-t", f"{dur}", "-movflags", "+faststart", out], check=True)
    return out
