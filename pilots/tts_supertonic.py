"""Run inside /workspace/venv-supertonic: python tts_supertonic.py voice speed out.wav "text"
Narration voice for hosts without the Kokoro venv (audio.py picks the backend that exists).
Supertonic 3 (ONNX, local). Writes the same 24 kHz mono wav as tts_kokoro.py.
"""
import sys
import numpy as np
import soundfile as sf

SR = 24000
# Kokoro voice ids used by episodes -> Supertonic voice styles (male M*, female F*)
VOICE_MAP = {"am_michael": "M1", "am_adam": "M2", "am_puck": "M3",
             "af_heart": "F1", "af_bella": "F2", "af_nicole": "F3"}

voice, speed, out, text = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]

import supertonic  # noqa: E402

tts = supertonic.TTS(model="supertonic-3")
style = tts.get_voice_style(VOICE_MAP.get(voice, "M1"))
wav, _dur = tts.synthesize(text, voice_style=style, total_steps=12, speed=speed,
                          lang="en", silence_duration=0.0)
x = np.asarray(wav, dtype=np.float32)
if x.ndim > 1:  # supertonic returns (channels, samples)
    x = x.mean(0) if x.shape[0] <= 2 else x.mean(1)
src_sr = int(getattr(tts, "sample_rate", 0) or 44100)
if src_sr != SR:
    n = int(round(len(x) * SR / src_sr))
    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
sf.write(out, x, SR)
