# pilots — code-only reel engine (one JSON per episode)

```bash
nice -n 19 /workspace/venv-pilots/bin/python pilots/make_reel.py --style A --episode pilots/episodes/pushup_fix.json
nice -n 19 /workspace/venv-pilots/bin/python pilots/make_reel.py --style B --episode pilots/episodes/pushup_fix.json
```

- **A**: grey matte mannequin cut-out puppet (`puppet.py`, palettes incl. colour/feel/boxer variants), charcoal #2E3033, ghost wrong-pose overlay, continuous rep loop.
- **B**: pen-line stickman on pastel blue with line boil, a new shot every 2–3 s (`style_b.py` scene types: reps / think / split / closeup / conclusion).
- `rig.py`: fixed-length 2D rig, exact two-bone IK (limb drift is a QA gate).
- `audio.py`: Kokoro-82M (Apache-2.0) via kokoro-onnx in `/workspace/venv-kokoro` at speed 1.12; SFX and beat are synthesised here (CC0); two-pass loudnorm to -14 LUFS.
- `qa.py`: gates (format, 10–12 s, motion from frame 0, no still > 1 s, hook at frame 0, cap heights, contrast, bottom 15% clear, limb drift, loudness, narration fits cues). Output: MP4, 12-frame flow sheet, qa.json in `output/pilots/<date>/` (not committed).
- venv: `/workspace/venv-pilots` (skia-python, numpy, pillow, soundfile). Model weights are not committed.
- Only push-up poses exist so far; a squat needs a new pose function in `rig.py`.
