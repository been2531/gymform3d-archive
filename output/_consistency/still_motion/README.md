# D7 still-motion mock (t674u-t676u) — RIGHT form peak, Day 4 grade applied ONCE

Source: frame 36 (peak, elbows back) of output/assets/day7_right_raw_v3.mp4 (NOT the relit clip: the relight lifted a
translucent ghost of the far arm and cut a notch into the head). 0 generations.

    nice -n 19 python3 tools/still_motion_mock.py output/_consistency/still_motion/src_right_peak_f36_720.png \
        output/_consistency/still_motion --glow-detail 0.45 --arm-limit 0.03

- Lanczos to 720x1280 + mild unsharp (0.6, sigma 1.0); the pipeline does the 720 -> 1080 transform.
- --glow-detail 0.45: inside the source glow only, fibre-groove depth x0.45 (generator paints wide black gaps).
- Production grade: native_grade + muscle.day4_fibre via region_glow (muscle_glow_clamp, target_regions.right,
  Day 4 knobs day4_input_scale 2 = 0.6, glow_gain_override 2 = 18 from configs/day07_muscle2.json).
- --arm-limit 0.03: the visible upper arm (triceps) overlaps the lats / rear-delt polygons in this rear-3/4 view; the
  clamp is capped there (tool-local wrapper, region_glow.py unchanged).
- Template: production layout (Kaushan title, 1-thumb preview with the green right badge, measured elbow flexion
  79 deg, one label ELBOWS BACK, hide_lines).
