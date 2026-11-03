# Option-B motion test (t691s2456, approved by 중현 님) — D7 Seated Cable Row, RIGHT form

- 1 Grok Bot GenerateVideo (0 images, no grok.com, no paid API, no retry): 2 s, 720p, 9:16.
  - First frame: `output/_consistency/d7m/pin_right_fibre_head_t2_720x1280.png`.
  - Last frame: `output/_consistency/still_motion/reel/src_right_peak_stackfix_720.png`.
  - Prompt: locked camera, one smooth row pull. The raw clip is `d7_optB_right_pull_2s_720p_raw.mp4` (49 f).
- Loop: `d7_optB_right_loop_motion_only.mp4`, built by `tools/optb_motion_test.py`.
  - Structure: forward, 0.3 s hold at the peak, then reverse, 4 loops (17.2 s). No grade.
- Diagnostic only: `d7_optB_right_loop_trim_f22_codeonly.mp4` is the same frames trimmed to f22→f48. Code only, no new generation.
- Sheet: `OPTB_TEST_sheet.jpg`. Numbers: `optb_qc.json` (MediaPipe heavy pose per frame; SSIM vs the start pin outside the dilated body mask).

## Verdict: FAIL, 2 of 5 criteria pass
| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Machine unchanged (SSIM ≥ 0.98) | PASS | Static machine area min 0.99; looks identical by eye. The tower crop reads 0.977–0.983, but the two pins differ by 0.989 in that crop too. The weight stack never lifts during the pull (realism note) |
| 2 | Both hands on the handle, no split | PASS | Every frame. The handle is still the rectangular frame from the pins (not a V) |
| 3 | Cable is one line from the low pulley | FAIL | f0: no cable (the start pin has none). f1: a cable segment floating in mid-air. f2–f48: correct. In the loop the cable blinks at every start turnaround |
| 4 | One natural pull, torso ≤ 10°, elbows straight back, no cross/twist/fly/pulldown | FAIL | f0→f24 the model first REACHES forward: wrist moves 78 px toward the tower, elbow flexion 50→18°, torso 12→0°. Then f24→f48 is one clean pull to 19° lean-back. Torso range 20.1° (pins alone: 12° vs 19°). At the peak the elbow stays about 20 px in front of the shoulder line, so it never passes the back line. No crossing, twisting, fly or pulldown |
| 5 | Invisible loop seam | FAIL | No pixel jump (the turnarounds reuse the same frame). But the start turnaround reverses at speed (wrist 10–13 px/frame), which reads as a bounce, plus the cable blink. The peak turnaround is smooth (≤ 4 px/frame) |

## Root cause
The pins, not the generator.
- **Start pin:** the arms aren't fully extended (elbow flexion 50°) and no cable is drawn, so the model inserts a reach and pops the cable in.
- **Peak pin:** leans back 19° against the start pin's 12°, and the elbows don't pass the back line.
- **What held:** the machine, the handle grip and the absence of drift. That supports the method: short + pinned.

## Next step
No retry was made. Recommended next step: make 2 new pins, which needs approval for 2 GenerateImage, then 1 GenerateVideo with the same settings. The pins must share:
- the same torso angle (upright, ≤ 5° apart);
- the start with full arm extension and a visible cable from the low pulley;
- the peak with the elbows past the back line;
- a V handle;
- the sculpted Day 4 body with flat light.

Code-only check (trim f22→f48):
- What it fixes: the cable is present and the start turnaround is at ≈ 0 speed.
- What it doesn't fix: the torso range is still 19.3°, because that comes from the peak pin.
