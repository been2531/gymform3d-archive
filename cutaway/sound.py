"""Procedural audio bed + SFX for cutaway reels (self-synthesised, CC0).

No samples, no music library: everything is generated from sines and filtered
noise with a fixed seed, so a topic always gets the same sound and nothing has
licence strings attached.  (Never use /workspace/bgm: commercial music.)

  drone  : two detuned low partials + fifth + octave, slow LFO, opening filter
  air    : low-passed pink-ish noise, swells into the bloom
  whoosh : band-passed noise sweep on every cut (pre-roll 0.18 s)
  thump  : pitch-dropping sine + click on every cut
  shimmer: soft high partials on the macro shot
  ticks  : rising soft ticks under the counter
  riser + hit + tail on the bloom
Loudness: BS.1770 K-weighted integrated loudness normalised to the target
(default -14 LUFS), then a soft limiter keeps sample peaks under -1.5 dBFS.
"""
from __future__ import annotations

import subprocess
import wave

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

SR = 48000


def _env(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / max(d, 1e-4))


def _bp(x, lo, hi, order=2):
    sos = butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, x)


def _lp(x, f, order=2):
    return sosfilt(butter(order, f, btype="low", fs=SR, output="sos"), x)


def whoosh(rng, dur=0.42):
    n = int(dur * SR)
    noise = rng.normal(0, 1, n)
    out = np.zeros(n)
    seg = 960
    for k in range(0, n, seg):
        f = 300 + 4200 * (k / n) ** 1.4
        out[k:k + seg] = _bp(noise[k:k + seg + 0], f * 0.6, min(f * 1.6, 20000))[:len(out[k:k + seg])]
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.6
    return out * env * 0.5


def thump(dur=0.5, f0=95, f1=38):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 18)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * _env(n, 0.003, 0.16)
    click = np.random.default_rng(1).normal(0, 1, n) * _env(n, 0.0005, 0.006) * 0.3
    return x + _lp(click, 5000)


def tick(freq):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t) * _env(n, 0.001, 0.018) * 0.35


def shimmer(rng, dur=1.8):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f in (1318.5, 1760.0, 2217.5, 2637.0):
        x += np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))
    return x * _env(n, 0.25, 0.7) * 0.06


def riser(rng, dur=0.6):
    n = int(dur * SR)
    noise = rng.normal(0, 1, n)
    x = _bp(noise, 800, 6000) * np.linspace(0, 1, n) ** 2.2
    return x * 0.35


def tail(rng, dur=2.2):
    n = int(dur * SR)
    return _lp(rng.normal(0, 1, n), 1800) * _env(n, 0.01, 0.7) * 0.18


def k_weight(x):
    # BS.1770 pre-filter (high shelf) + RLB high-pass at 48 kHz
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)


def lufs(x):
    y = k_weight(x)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    ms = []
    for s in range(0, len(y) - blk, hop):
        ms.append(np.sum(np.mean(y[s:s + blk] ** 2, axis=0)))
    ms = np.array(ms)
    L = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[L > -70]
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    g2 = ms[(L > -70) & (L > rel)]
    return -0.691 + 10 * np.log10(np.mean(g2) + 1e-12)


def build(topic: dict, out_wav: str, target_lufs: float = -14.0, gain_db: float = 0.0):
    dur = float(topic["duration"])
    au = topic.get("audio", {})
    rng = np.random.default_rng(au.get("seed", 7))
    n = int(dur * SR)
    t = np.arange(n) / SR
    key = au.get("key_hz", 55.0)
    shots = topic["shots"]
    bloom_t = next((s["t"] for s in shots if s.get("kind") == "bloom"), dur - 2)
    # drone: present from sample 0 (no silent intro)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * t / 3.3)
    drone = (np.sin(2 * np.pi * key * t) + np.sin(2 * np.pi * key * 1.004 * t + 1.3)
             + 0.55 * np.sin(2 * np.pi * key * 1.5 * t + 0.4) * (0.6 + 0.4 * lfo)
             + 0.35 * np.sin(2 * np.pi * key * 2 * t + 2.1) * lfo
             + 0.18 * np.sin(2 * np.pi * key * 3 * t + 0.7) * (1 - lfo))
    swell = 0.75 + 0.25 * np.clip(t / bloom_t, 0, 1) + 0.25 * np.exp(-((t - bloom_t - 0.6) / 1.0) ** 2)
    drone = drone * swell * np.minimum(1, t / 0.03) * np.minimum(1, (dur - t) / 0.6) * 0.16
    air = _lp(rng.normal(0, 1, n), 900) * 0.05 * (0.6 + 0.4 * lfo) * swell * np.minimum(1, t / 0.05)
    mix = drone + air
    def add(x, at, g=1.0):
        i = int(max(0, at) * SR)
        j = min(n, i + len(x))
        if j > i:
            mix[i:j] += x[: j - i] * g
    add(thump(0.5, 120, 40), 0.0, 0.7)              # frame-0 hit
    add(whoosh(rng, 0.38), 0.05, 0.8)               # window opening
    for s in shots[1:]:
        if s.get("kind") == "bloom":
            continue
        add(whoosh(rng, 0.42), s["t"] - 0.2, 0.9)
        add(thump(0.45, 100, 42), s["t"], 0.6)
        if s.get("kind") == "macro":
            add(shimmer(rng, 2.2), s["t"] + 0.05, 1.0)
    c = topic.get("counter")
    if c:
        cnt = c.get("count", 1.3)
        for k in range(12):
            add(tick(900 + 90 * k), c["t0"] + 0.15 + cnt * (1 - (1 - k / 12) ** 2), 1.0)
        if c.get("action"):      # LOAD METER signature: heavy hit when the count lands
            add(thump(0.7, 75, 30), c["t0"] + 0.15 + cnt, 1.0)
    add(riser(rng, 0.6), bloom_t - 0.6, 1.0)
    add(thump(0.8, 90, 32), bloom_t, 0.95)
    add(tail(rng), bloom_t, 1.0)
    stereo = np.stack([mix, np.roll(mix, int(0.004 * SR))], axis=1)
    L = lufs(stereo)
    stereo *= 10 ** ((target_lufs + gain_db - L) / 20)
    # soft limiter
    ceil = 10 ** (-1.5 / 20)
    stereo = np.tanh(stereo / ceil) * ceil
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return {"lufs_pre": round(float(L), 2), "lufs_est": round(float(lufs(pcm / 32767.0)), 2)}


def mux(video, wav, out, ffmpeg="/usr/bin/ffmpeg"):
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    return out
