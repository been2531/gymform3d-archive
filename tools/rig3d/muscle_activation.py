"""Per-muscle contraction from rig bone-length change (origin → insertion).

Shared by every exercise: bone pairs are anatomical, not db_row-specific.
Working-side tokens `{S}` expand to Left/Right from bones.npz `working_hand`.
"""
from __future__ import annotations
import numpy as np

# origin, insertion — distance shortens when the muscle contracts
LINKS = {
    # Use spans that actually change with pose — never a parent-child bone length.
    "lats":       ("Spine02", "{S}Hand"),
    "rhomboids":  ("Spine", "{S}Hand"),
    "mid_trap":   ("neck", "{S}Hand"),
    "upper_trap": ("Head", "{S}Shoulder"),       # constant on a locked-neck row (compensation = shrug)
    "rear_delt":  ("Spine01", "{S}Arm"),
    "side_delt":  ("{S}Shoulder", "{S}Hand"),
    "front_delt": ("Spine", "{S}Hand"),
    "biceps":     ("{S}Arm", "{S}Hand"),
    "triceps":    ("{S}Shoulder", "{S}Hand"),
    "forearm":    ("{S}Arm", "{S}Hand"),
    "erectors":   ("Hips", "Head"),              # torso hinge (nearly locked on a good row)
    "pecs":       ("Spine", "{S}Hand"),
    "abs":        ("Hips", "Spine"),
    "obliques":   ("Hips", "{S}Shoulder"),
    "glutes":     ("Hips", "{S}Leg"),
    "quads":      ("{S}UpLeg", "{S}Foot"),
    "hams":       ("Hips", "{S}Foot"),
    "calves":     ("{S}Leg", "{S}ToeBase"),
}

# Fitness-content chain: effort travels from the working hand up into the back.
FLOW_CHAIN = ["forearm", "biceps", "rear_delt", "lats", "rhomboids", "mid_trap"]
COMPENSATORS = ["upper_trap", "biceps", "erectors"]


def _side(working_hand):
    s = str(working_hand or "RightHand")
    return "Left" if s.lower().startswith("left") else "Right"


def _pos(mat):
    return np.asarray(mat)[:3, 3]


def activations_for_clip(bones_path):
    """Return {1-based frame: {muscle: 0..1 contraction}} and rest lengths."""
    z = np.load(bones_path, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    pose = z["pose"]          # (F, J, 4, 4)
    rest = z["rest"]
    ni = {n: i for i, n in enumerate(names)}
    side = _side(z["working_hand"] if "working_hand" in z.files else "RightHand")

    def pair(link):
        a, b = link[0].replace("{S}", side), link[1].replace("{S}", side)
        # fallbacks for neck naming
        if a not in ni and a.lower() == "neck":
            a = next((k for k in ni if k.lower() == "neck"), a)
        if b not in ni and b.lower() == "neck":
            b = next((k for k in ni if k.lower() == "neck"), b)
        return ni.get(a), ni.get(b)

    F = pose.shape[0]
    out = {f + 1: {} for f in range(F)}
    for mus, link in LINKS.items():
        ia, ib = pair(link)
        if ia is None or ib is None:
            continue
        rest_len = float(np.linalg.norm(_pos(rest[ia]) - _pos(rest[ib]))) + 1e-6
        curr = np.linalg.norm(pose[:, ia, :3, 3] - pose[:, ib, :3, 3], axis=1)
        # shortening vs the longest pose in the clip (stretched = 0, shortest = 1)
        lo, hi = float(curr.min()), float(curr.max())
        span = max(hi - lo, 1e-4)
        act = (hi - curr) / span
        for f in range(F):
            out[f + 1][mus] = round(float(np.clip(act[f], 0, 1)), 4)
    return out, {"side": side, "n": F}


def flow_weights(progress, chain=None, width=0.22):
    """Muscle → 0..1 as a wavefront traveling along `chain` with progress in [0,1]."""
    chain = list(chain or FLOW_CHAIN)
    n = max(len(chain) - 1, 1)
    p = float(np.clip(progress, 0, 1))
    w = {}
    for i, name in enumerate(chain):
        onset = i / n * 0.82
        t = (p - onset) / max(width, 1e-4)
        t = float(np.clip(t, 0.0, 1.0))
        w[name] = float(t * t * (3.0 - 2.0 * t))
    return w
