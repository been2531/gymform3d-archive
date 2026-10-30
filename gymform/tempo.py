"""LOCKED tempo (t544u: Day 4 tempo is the baseline): 4 x 5 s steps, 150 frames each.
Day 4 wave: 0.5 - 0.5*cos(2*pi*1.5*u) -> rest -> top (1.67 s) -> rest (3.33 s) -> top (5.0 s), smooth cosine
reversals, no flat holds. The Day 5 hold wave (0.3/1.5/1.3/1.5/0.4) stays as legacy style "day5_holds"."""
import json
import math
import os

FRAMES_PER_STEP = 150
_SPEC = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gymform_spec.json")))
TEMPO = _SPEC["tempo_s"]
# QC sidecar frames per step (f_local -> tag); "peak" = frame 50 = first top reversal of the Day 4 wave
QC_TAGS = {4: "start", 20: "t20", 35: "t35", 50: "peak", 70: "t70", 100: "t100", 130: "t130"}


def _wave_day5(u):
    if u < 0.06:
        return 0.0
    if u < 0.36:
        return 0.5 - 0.5 * math.cos(math.pi * ((u - 0.06) / 0.30))
    if u < 0.62:
        return 1.0
    if u < 0.92:
        return 0.5 + 0.5 * math.cos(math.pi * ((u - 0.62) / 0.30))
    return 0.0


def tempo_wave(u, style=None, phase=0.0):
    """phase = pi continues the motion across a step cut (t600: step N ends at the top of the wave, so step N+1
    starts there instead of jumping back to pose 0); turnarounds stay at frames 50/100."""
    style = style or TEMPO.get("style", "day4_cosine")
    if style == "day5_holds":
        return _wave_day5(u)
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * u * float(TEMPO.get("cycles_per_step", 1.5)) + phase)


def pose_index(f_local, n_f, phase=0.0):
    return int(round(tempo_wave(f_local / float(FRAMES_PER_STEP), phase=phase) * (n_f - 1)))


def step_phase(step_order_idx, continuous):
    return (math.pi * (step_order_idx % 2)) if continuous else 0.0


def peak_frame(phase):
    """frame of the first deep (wave = 1) pose: 50 normally, 100 when the step runs in the continued phase."""
    return 100 if abs(math.cos(phase) + 1.0) < 1e-6 else 50


def reversal_frames(style=None):
    """Interior frames where the wave changes direction (top/bottom turnarounds)."""
    w = [tempo_wave(f / float(FRAMES_PER_STEP), style) for f in range(FRAMES_PER_STEP + 1)]
    return [f for f in range(1, FRAMES_PER_STEP) if (w[f] - w[f - 1]) * (w[f + 1] - w[f]) < 0
            or (w[f] - w[f - 1] != 0 and w[f + 1] == w[f])]


def measure_reversals(mp4, roi=(0.30, 0.95, 0.15, 0.85)):
    """(v) tempo QC on a rendered reel: per step, low-motion minima of the mannequin region (frame differences)."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(mp4)
    prev, d = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (270, 480)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = g[int(roi[0] * 480):int(roi[1] * 480), int(roi[2] * 270):int(roi[3] * 270)]
        d.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    d = np.array(d)
    out = {}
    for s in range(4):
        seg = d[s * FRAMES_PER_STEP + 3:(s + 1) * FRAMES_PER_STEP - 2]
        if len(seg) < 20:
            continue
        sm = np.convolve(seg, np.ones(5) / 5, "same")
        out[s + 1] = [i + 3 for i in range(5, len(sm) - 5) if sm[i] == sm[i - 5:i + 6].min() and sm[i] < 0.5 * sm.max()]
    return out


def tempo_ok(found, tol=None):
    tol = tol if tol is not None else int(TEMPO.get("qc_reversal_tol_frames", 6))
    exp = TEMPO.get("reversal_frames") or reversal_frames()
    bad = {s: f for s, f in found.items() if not all(any(abs(x - e) <= tol for x in f) for e in exp)}
    return not bad, exp, bad
