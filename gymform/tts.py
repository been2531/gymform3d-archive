"""LOCKED voice: edge_tts en-US-ChristopherNeural (+24%, -2Hz, atempo 1.08), one line per step at fixed offsets
(lines from the day config). Day >= caption_rules.spoken_question.from_day: the spoken experience question
(caption.spoken_question) is appended right after the Step 4 line, inside the final hold/bottom, ending <= latest_end_s.
Writes <wav>.timing.json (segments with real durations) for QC item (l)."""
import asyncio
import json
import math
import os
import subprocess
import threading
import wave

import numpy as np

from gymform.ctx import CFG, DURATION, SPEC, TMP
from gymform import caption

SR = 44100
VOICE = dict(voice="en-US-ChristopherNeural", rate="+24%", pitch="-2Hz")
CLEAN = ("silenceremove=start_periods=1:start_silence=0.03:start_threshold=-42dB:stop_periods=-1:stop_duration=0.12:"
         "stop_threshold=-42dB,areverse,silenceremove=start_periods=1:start_silence=0.04:start_threshold=-42dB,areverse,atempo=1.08")


def segments():
    """[(target_start or None, text, role)]; None = place after the previous segment (spoken question)."""
    segs = [(float(t), s, f"step{i + 1}") for i, (t, s) in enumerate(CFG["tts_lines"])]
    r = SPEC["caption_rules"].get("spoken_question", {})
    if r and int(CFG.get("day", 0)) >= int(r.get("from_day", 6)):
        segs.append((None, caption.spoken_question(CFG, SPEC), "question"))
    return segs


async def _say(text, i):
    import edge_tts
    mp3_p, wav_p = os.path.join(TMP["dir"], f"vo_{i}.mp3"), os.path.join(TMP["dir"], f"vo_{i}.wav")
    await edge_tts.Communicate(text, **VOICE).save(mp3_p)
    subprocess.run(["ffmpeg", "-y", "-threads", "2", "-i", mp3_p, "-af", CLEAN, "-ac", "1", "-ar", str(SR), wav_p],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with wave.open(wav_p, "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


async def _synth(wav_path):
    n = int(SR * DURATION)
    t = np.arange(n, dtype=np.float32) / SR
    beat = 0.45 + 0.55 * (np.sin(2.0 * math.pi * (100.0 / 60.0) * t) > 0.72).astype(np.float32)
    master = 0.040 * np.sin(2.0 * math.pi * 55.0 * t) * beat
    for k, ts in enumerate([0.05, 5.0, 10.0, 15.0]):
        s0, dc = int(ts * SR), int(0.24 * SR)
        tc = np.linspace(0.0, 0.24, dc, endpoint=False)
        master[s0:s0 + dc] += 0.085 * np.sin(2.0 * math.pi * (220.0 if k < 2 else 528.0) * tc) * np.exp(-tc * 13.0)
    r = SPEC["caption_rules"].get("spoken_question", {})
    out, prev_end = [], 0.0
    for i, (ts, text, role) in enumerate(segments()):
        raw = await _say(text, i)
        dur = len(raw) / SR
        if ts is None:
            ts = max(prev_end + float(r.get("gap_after_step4_s", 0.25)), float(r.get("min_start_s", 0.0)))
            limit = float(r.get("latest_end_s", DURATION - 0.1))
        else:
            limit = (int(role[4:])) * 5.0 - 0.85 if role != "step4" or not r else DURATION - 0.1
        end = ts + dur
        out.append({"role": role, "text": text, "start": round(ts, 3), "dur": round(dur, 3), "end": round(end, 3),
                    "gap_from_prev": round(ts - prev_end, 3)})
        assert end <= limit + 1e-6, f"{role} ends {end:.2f}s > {limit:.2f}s ({text!r})"
        s = int(ts * SR)
        e = min(n, s + len(raw))
        master[s:e] += raw[:e - s] * 0.92
        prev_end = end
    pcm = (np.clip(master, -0.96, 0.96) * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(np.repeat(pcm, 2).tobytes())
    json.dump({"day": CFG.get("day"), "duration_s": DURATION, "segments": out}, open(wav_path + ".timing.json", "w"), indent=1)
    return wav_path


def synth(wav_path):
    return asyncio.run(_synth(wav_path))


def synth_async(wav_path):
    th = threading.Thread(target=synth, args=(wav_path,))
    th.start()
    return th
