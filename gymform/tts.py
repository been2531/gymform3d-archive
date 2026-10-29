"""LOCKED voice: edge_tts en-US-ChristopherNeural, one line per step at fixed offsets (lines come from the day config)."""
import asyncio
import threading

from gymform.ctx import CFG, DURATION, d5


def synth(wav_path):
    lines = [(float(t), s) for t, s in CFG["tts_lines"]]
    asyncio.run(d5.synthesize_day5_voiceover_wav(wav_path, lines, DURATION))
    return wav_path


def synth_async(wav_path):
    t = threading.Thread(target=synth, args=(wav_path,))
    t.start()
    return t
