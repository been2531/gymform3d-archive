# pilots — code-only reel engine (one JSON per episode)

```bash
nice -n 19 /workspace/venv-pilots/bin/python pilots/make_reel.py --style A --episode pilots/episodes/pushup_fix.json
nice -n 19 /workspace/venv-pilots/bin/python pilots/make_reel.py --style B2 --episode topics/episodes/squat_fix.json --tag B_v6
```

- **A**: grey matte mannequin cut-out puppet (`puppet.py`, palettes incl. colour/feel/boxer variants), charcoal #2E3033, ghost wrong-pose overlay, continuous rep loop.
- **B**: pen-line stickman on pastel blue with line boil, a new shot every 2–3 s (`style_b.py` scene types: reps / think / split / closeup / conclusion).
- **B2** (`style_b2.py`, `--style B2`, default output `<id>_B_v2.mp4`, the daily pipeline renders it as `<id>_B.mp4`): expressive faces (strain, shocked, worried, happy, proud), flex ending, props, auto-framing from pose bounding boxes. Exercises: `pushup_pose`, `squat_pose` in `rig.py`; a new exercise = one pose function + one JSON (see `topics/episodes/squat_fix.json`).
- `rig.py`: fixed-length 2D rig, exact two-bone IK (limb drift is a QA gate). `squat_pose` takes the bottom-of-rep joint angles as form keys — `shin`, `thigh` (deg from horizontal), `lean` (extra torso lean, deg), `heel` (deg the heel lifts, pivoting on the ball) — so a wrong form is a different pose, not a different colour; the good squat needs none of them.
- `claims.py`: posture claims. A caption that names a posture (`HIPS BELOW KNEES`, `HEELS UP`, `HEELS DOWN`, `CHEST TO FLOOR`, `ONE LINE`, `SIT BACK`) is measured on the rig joints, and `style_b2` draws it only while the pose in frame shows it (`HIPS BELOW KNEES` therefore appears only in the lowered part of each rep, and a floor claim cannot sit over the standing ending).
- `audio.py`: narration from the first engine this host has — Kokoro-82M (Apache-2.0) via kokoro-onnx in `/workspace/venv-kokoro`, else Supertonic 3 (`tts_supertonic.py`, `/workspace/venv-supertonic`); override with `GYMFORM_TTS=kokoro|supertonic`. SFX and the thin rhythm bed are synthesised here (CC0); the bed keeps the track audible between lines (no digital-silence hole) and the muxed file is trimmed onto -14.0 LUFS.
- `text.py`: display faces resolve from `fonts/` in the repo first (Anton, Kalam, Permanent Marker), then the host font path, so a fresh checkout renders the same type.
- `qa.py`: gates — format 1080x1920/30, `ig_spec` (7 Mbps CBR H.264 High + AAC-LC 48 kHz stereo + faststart), 10–15 s, motion from frame 0, no still > 1 s, hook at frame 0, cap heights, contrast, bottom 15% clear, text inside the frame and clear of the figure, limb drift, `limbs_in_bounds` (every hand, foot and mat end inside an 8 px margin), `pose_caption_sync`, `no_dead_air`, loudness -14 LUFS ±1, narration fits cues, and the B2 gates (target muscle visible, named muscle highlighted, rep duration, figure fill, profile faces, fault highlight, wrong-form readability, think/ending variant, `fault_joint_delta`: wrong vs right differ by ≥15° of joint angle — or a clear posture difference for a depth-only fault — at the bottom of the rep and still differ at the top).
- Output: MP4, 12-frame flow sheet, qa.json in `output/pilots/<date>/` (not committed).
- venv: `/workspace/venv-pilots` (skia-python, numpy, pillow, soundfile). Model weights are not committed.
- Tests: `python3 -m unittest tests.test_pose_caption_gates -v` (claims, caption sync, limb bounds, fault delta).
