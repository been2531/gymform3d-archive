"""Run inside /workspace/venv-kokoro: python tts_kokoro.py voice speed out.wav "text"
Kokoro-82M (Apache-2.0) via kokoro-onnx (MIT). Local only."""
import sys
from kokoro_onnx import Kokoro
import soundfile as sf
voice, speed, out, text = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
k = Kokoro("/workspace/venv-kokoro/models/kokoro-v1.0.int8.onnx", "/workspace/venv-kokoro/models/voices-v1.0.bin")
s, sr = k.create(text, voice=voice, speed=speed, lang="en-us")
sf.write(out, s, sr)
