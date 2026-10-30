"""Target bones (t552u): bones work like muscle fibre - only the MAIN bones of the exercise are shown, locally.
A bone name maps to a soft region built from the 13-point keypoints (no drawing: the region only GATES which
native bone pixels of the clip may be shown/regraded). Names without a side suffix mean both sides.
Pure numpy/cv2, no pipeline state, so QC can rebuild the exact same region from the frame keypoints."""
import math
import numpy as np

VOCAB = ("clavicle", "scapula", "humerus", "radius_ulna", "sternum", "upper_ribs", "ribs", "thoracic_spine",
         "lumbar_spine", "pelvis", "femur", "patella", "tibia_fibula")


def _seg_dist(xx, yy, a, b):
    ax, ay = a; bx, by = b
    vx, vy = bx - ax, by - ay
    t = np.clip(((xx - ax) * vx + (yy - ay) * vy) / max(vx * vx + vy * vy, 1e-6), 0.0, 1.0)
    return np.hypot(xx - (ax + t * vx), yy - (ay + t * vy))


def _mid(a, b):
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def region(kp, names, shape, soft=10.0):
    """Soft 0..1 mask (H, W) of the union of the named bones' regions."""
    H, W = shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    P = {k: (float(v[0]), float(v[1])) for k, v in kp.items()}
    ua = 0.5 * (math.dist(P["sh_r"], P["elb_r"]) + math.dist(P["sh_l"], P["elb_l"]))   # body scale
    sh_mid, hip_mid = _mid(P["sh_r"], P["sh_l"]), _mid(P["hip_r"], P["hip_l"])
    out = np.zeros((H, W), np.float32)

    def seg(a, b, r):
        nonlocal out
        out = np.maximum(out, np.clip((r - _seg_dist(xx, yy, a, b)) / soft + 0.5, 0.0, 1.0))

    def disc(c, r):
        nonlocal out
        out = np.maximum(out, np.clip((r - np.hypot(xx - c[0], yy - c[1])) / soft + 0.5, 0.0, 1.0))

    for n in names:
        base, side = (n[:-2], n[-1]) if n.endswith(("_r", "_l")) and n[:-2] in VOCAB else (n, "")
        if base not in VOCAB:
            raise ValueError(f"unknown target bone {n!r} (vocabulary: {VOCAB})")
        sides = [side] if side else ["r", "l"]
        for s in sides:
            sh, elb, wr = P[f"sh_{s}"], P[f"elb_{s}"], P[f"wr_{s}"]
            hip, kn, an = P[f"hip_{s}"], P[f"kn_{s}"], P[f"ank_{s}"]
            if base == "clavicle":
                seg(sh, _lerp(sh_mid, hip_mid, 0.05), 0.20 * ua)
            elif base == "scapula":   # acromion / glenoid / blade around the shoulder joint
                disc(_lerp(sh, hip_mid, 0.12), 0.45 * ua)
            elif base == "humerus":
                seg(sh, elb, 0.24 * ua)
            elif base == "radius_ulna":
                seg(elb, wr, 0.20 * ua)
            elif base == "femur":
                seg(hip, kn, 0.28 * ua)
            elif base == "patella":
                disc(kn, 0.22 * ua)
            elif base == "tibia_fibula":
                seg(kn, an, 0.22 * ua)
        if base == "sternum":
            seg(_lerp(sh_mid, hip_mid, 0.02), _lerp(sh_mid, hip_mid, 0.55), 0.22 * ua)
        elif base in ("upper_ribs", "ribs"):
            depth = 0.50 if base == "upper_ribs" else 0.80
            c = _lerp(sh_mid, hip_mid, depth / 2)
            half_w = 0.62 * math.dist(P["sh_r"], P["sh_l"]) + 0.20 * ua
            half_h = depth / 2 * math.dist(sh_mid, hip_mid) + 0.15 * ua
            ang = math.atan2(hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1])
            dx, dy = xx - c[0], yy - c[1]
            u = dx * math.cos(ang) - dy * math.sin(ang)
            v = dx * math.sin(ang) + dy * math.cos(ang)
            e = np.hypot(u / max(half_w, 1), v / max(half_h, 1))
            out = np.maximum(out, np.clip((1.0 - e) * min(half_w, half_h) / soft + 0.5, 0.0, 1.0))
        elif base == "thoracic_spine":
            seg(sh_mid, _lerp(sh_mid, hip_mid, 0.70), 0.16 * ua)
        elif base == "lumbar_spine":
            seg(_lerp(sh_mid, hip_mid, 0.65), hip_mid, 0.18 * ua)
        elif base == "pelvis":
            seg(P["hip_r"], P["hip_l"], 0.45 * ua)
    return out


def target_bones(cfg, root):
    """Day config -> target bone list: explicit `target_bones`, else configs/motion_standards.json[motion_standard]."""
    import json, os
    if cfg.get("target_bones"):
        return list(cfg["target_bones"])
    ms = cfg.get("motion_standard")
    if not ms:
        return []
    std = json.load(open(os.path.join(root, "configs", "motion_standards.json")))["exercises"][ms]
    return list(std.get("target_bones", []))
