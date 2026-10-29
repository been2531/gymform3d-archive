"""LOCKED tempo: 4 x 5 s steps, 150 frames each, raise-hold-lower wave; QC sidecar frame tags."""
import math

FRAMES_PER_STEP = 150
# QC sidecar frames per step (f_local -> tag)
QC_TAGS = {4: "start", 20: "t20", 35: "t35", 50: "peak", 70: "t70", 100: "t100", 130: "t130"}


def tempo_wave(u):
    if u < 0.06:
        return 0.0
    if u < 0.36:
        return 0.5 - 0.5 * math.cos(math.pi * ((u - 0.06) / 0.30))
    if u < 0.62:
        return 1.0
    if u < 0.92:
        return 0.5 + 0.5 * math.cos(math.pi * ((u - 0.62) / 0.30))
    return 0.0


def pose_index(f_local, n_f):
    return int(round(tempo_wave(f_local / float(FRAMES_PER_STEP)) * (n_f - 1)))
