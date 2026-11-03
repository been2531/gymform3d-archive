"""Day 4 fibre look on a GLOW-FREE source (t648u, D7 machine-first stills / pinned-frame clips).

Day 4 (build_day4_rdl_master.shade_day4_3d_frame_1080p, muscle.day4_fibre) had no anatomical gate: the clip's own
native glow (act_dom) decided where colour went, and day4_fibre turned it into fibres + red/emerald colour. A D7
machine-first frame carries NO glow (the image generator must not paint muscles). This module supplies the act_dom
field instead: per-frame keypoint region polygons from the anatomy rules (configs: target_regions), feathered and set
to Day 4's native core level (act ~100, the t572u calibration point), and then calls the UNCHANGED muscle.day4_fibre.
Fibre texture, colour ramps, core blur, cap and shorts exclusion are Day 4's, line for line.

Regions live in a torso frame built from the keypoints of THE SAME frame (hips -> shoulders axis U, half shoulder
width V), so they follow the body in a clip exactly as the Day 5 shoulder-relative gate did; no pixel cut-out.
Coordinates: a = 0 at the hip midpoint, 1 at the shoulder midpoint; b = -1 / +1 at the screen-left / right shoulder.
"""
import cv2
import numpy as np
from gymform.ctx import CFG, W, H

# anatomy rules (t645u / t648u). poly = list of (a, b) torso points, mirrored for b -> -b when "both": true.
# limb regions: capsule along shoulder->elbow (biceps) or elbow->wrist (forearm), t0..t1 along the segment.
REGIONS = {
    "erector_spinae": dict(poly=[(0.04, 0.10), (0.04, 0.30), (0.62, 0.26), (0.62, 0.09)], both=True),
    "upper_trap":     dict(poly=[(1.30, 0.10), (1.30, 0.26), (1.04, 0.86), (0.94, 0.70), (0.86, 0.16)], both=True),
    "mid_trap_rhomboid": dict(poly=[(0.95, 0.06), (0.95, 0.48), (0.55, 0.40), (0.50, 0.06)], both=True),
    "lats":           dict(poly=[(0.86, 0.62), (0.88, 0.98), (0.50, 0.92), (0.14, 0.64), (0.14, 0.34), (0.48, 0.42)], both=True),
    "rear_delt":      dict(ell=(0.98, 1.02, 0.17, 0.20), both=True),   # centre a, b, radius a, radius b (torso units)
    "biceps":         dict(limb=("sh", "elb"), t=(0.18, 0.88), r=0.17),
    "forearm":        dict(limb=("elb", "wr"), t=(0.08, 0.80), r=0.13),
}
DEFAULT_TARGETS = {
    "wrong": {"erector_spinae": 1.0, "biceps": 1.0, "forearm": 1.0, "upper_trap": 1.0},
    "right": {"lats": 1.0, "mid_trap_rhomboid": 1.0, "rear_delt": 1.0, "biceps": 0.35},
}
ACT_CORE = 112.0   # Day 4 native glow core level, measured: day4_rdl_step2_ecorche_720p f39, act median 112 (act > 30)
# Day 4's glowing source pixels (same frame, act > 30): R median 145 (IQR 112-181), G 26, B 31. day4_fibre reads the
# fibre texture and the colour volume from these channels, so a glow-free source is lifted to the same levels by its
# OWN luminance (fibre structure kept, only the level calibrated - the t572u input-calibration idea)
D4_GLOW_R, D4_GLOW_G, D4_GLOW_B = 145.0, 26.0, 31.0


def on():
    """opt-in (default off): native look runs the Day 4 fibre shader instead of the linear glow recolour."""
    return CFG.get("native_muscle_shader") == "day4"


def _P(kp, k):
    return np.asarray(kp[k], np.float32)


def _frame(kp):
    sh_m = 0.5 * (_P(kp, "sh_r") + _P(kp, "sh_l"))
    hip_m = 0.5 * (_P(kp, "hip_r") + _P(kp, "hip_l"))
    U = sh_m - hip_m
    V = 0.5 * (_P(kp, "sh_l") - _P(kp, "sh_r"))
    # side / rear-3/4 views foreshorten the shoulder line; keep a minimum width so regions never collapse
    shw = float(np.linalg.norm(V)); minw = 0.28 * float(np.linalg.norm(U))
    if shw < minw:
        n = np.array([-U[1], U[0]], np.float32) / max(float(np.linalg.norm(U)), 1.0)
        n = n if np.dot(n, V) >= 0 else -n
        V = V + n * (minw - shw)
    return hip_m, U, V


def _pt(hip_m, U, V, a, b):
    return hip_m + a * U + b * V


def region_masks(kp, shape=(H, W)):
    """-> {name: float32 mask 0..1} at 1080x1920, keypoints in the same 1080 coords."""
    hip_m, U, V = _frame(kp)
    unit = float(np.linalg.norm(V))
    out = {}
    for name, R in REGIONS.items():
        m = np.zeros(shape, np.uint8)
        sides = (1.0, -1.0) if R.get("both") else (1.0,)
        if "poly" in R:
            for s in sides:
                pts = np.array([_pt(hip_m, U, V, a, s * b) for a, b in R["poly"]], np.int32)
                cv2.fillPoly(m, [pts], 1)
        elif "ell" in R:
            ca, cb, ra, rb = R["ell"]
            for s in sides:
                c = _pt(hip_m, U, V, ca, s * cb)
                ang = float(np.degrees(np.arctan2(U[1], U[0])))
                cv2.ellipse(m, (int(c[0]), int(c[1])), (max(2, int(ra * np.linalg.norm(U))), max(2, int(rb * unit))), ang, 0, 360, 1, -1)
        else:
            j0, j1 = R["limb"]
            torso = np.zeros(shape, np.uint8)
            cv2.fillPoly(torso, [np.array([_P(kp, k) for k in ("sh_r", "sh_l", "hip_l", "hip_r")], np.int32)], 1)
            for sd in ("r", "l"):
                # same screen-sorted chain the extractor uses (sh_r-elb_r-wr_r); in a rear view the FAR arm is hidden
                # behind the back and projects onto the torso -> a limb capsule mostly inside the torso is skipped
                if _P(kp, f"sh_{sd}") is None:
                    continue
                p0, p1 = _P(kp, f"{j0}_{sd}"), _P(kp, f"{j1}_{sd}")
                q0 = p0 + R["t"][0] * (p1 - p0); q1 = p0 + R["t"][1] * (p1 - p0)
                cap = np.zeros(shape, np.uint8)
                cv2.line(cap, tuple(int(v) for v in q0), tuple(int(v) for v in q1), 1, max(2, int(2 * R["r"] * unit)))
                arm = np.zeros(shape, np.uint8)
                cv2.line(arm, tuple(int(v) for v in _P(kp, f"sh_{sd}")), tuple(int(v) for v in _P(kp, f"wr_{sd}")), 1, 3)
                if (arm & torso).sum() > 0.5 * max(arm.sum(), 1):
                    continue
                m |= cap
        out[name] = m.astype(np.float32)
    return out


def act_field(kp, is_wrong, w_fg):
    """Day 4-level act_dom map from the step's target regions (soft 9 px feather, inside the body matte only)."""
    tg = (CFG.get("target_regions") or {}).get("wrong" if is_wrong else "right") or DEFAULT_TARGETS["wrong" if is_wrong else "right"]
    ms = region_masks(kp, w_fg.shape)
    a = np.zeros(w_fg.shape, np.float32)
    for name, k in tg.items():
        a = np.maximum(a, float(k) * ms[name])
    a = cv2.GaussianBlur(a, (0, 0), float(CFG.get("region_feather_px", 9.0)))
    return ACT_CORE * a * w_fg, ms


def clamp_mask(kp, is_wrong, shape):
    """allowed-region mask (opt-in muscle_glow_clamp): 1 inside the listed anatomy regions, soft 9 px edge."""
    tg = (CFG.get("target_regions") or {}).get("wrong" if is_wrong else "right") or DEFAULT_TARGETS["wrong" if is_wrong else "right"]
    ms = region_masks(kp, shape)
    a = np.zeros(shape, np.float32)
    for name, k in tg.items():
        a = np.maximum(a, float(k) * ms[name])
    return cv2.GaussianBlur(a, (0, 0), float(CFG.get("region_feather_px", 9.0))), ms


def apply(base_rgb, src_rgb, w_fg, shorts, kp, is_wrong, pulse_phase=0.75):
    """Day 4 way (t648u correction): base_rgb = native graded frame with the linear glow switched off; src_rgb = the
    clip pixels. act_dom = the SOURCE's own native glow (x _glow_gain calibration, as Day 6 t572u), optionally
    multiplied by the allowed-region clamp (muscle_glow_clamp); then the UNCHANGED muscle.day4_fibre.
    muscle_glow_source == "regions" (experimental, glow-free source) synthesises act from the regions instead.
    Returns (rgb uint8, core_w, region masks)."""
    from gymform import mask, muscle
    s = src_rgb.astype(np.float32)
    ms = {}
    if CFG.get("muscle_glow_source") == "regions":
        act, ms = act_field(kp, is_wrong, w_fg)
        L = 0.299 * s[..., 0] + 0.587 * s[..., 1] + 0.114 * s[..., 2]
        core = act > 0.6 * ACT_CORE
        Lm = float(np.median(L[core])) if core.sum() > 200 else 80.0
        an = np.clip(act / ACT_CORE, 0.0, 1.0)[..., None]
        glow_src = np.stack([L * (D4_GLOW_R / Lm), L * (D4_GLOW_G / Lm), L * (D4_GLOW_B / Lm)], -1)
        F = mask.features(np.clip(s * (1.0 - an) + glow_src * an, 0, 255))
        F["act_dom"] = act
    else:
        # t654u day4_input_scale: source level calibration (Day 4 glowing source R ~145 median); scales the channels
        # day4_fibre reads its colour volume from. act_dom scales with it -> use glow_gain_override to re-level.
        k_in = CFG.get("day4_input_scale", 1.0)
        k_in = float(k_in.get(str(CFG.get("_cur_step")), 1.0)) if isinstance(k_in, dict) else float(k_in)
        F = mask.features(np.clip(s * k_in, 0, 255))
        if CFG.get("muscle_glow_clamp"):
            cm, ms = clamp_mask(kp, is_wrong, w_fg.shape)
            F["act_dom"] = F["act_dom"] * cm      # -> w_act and core_seed (hence core_w) are zero outside the regions
            LAST_MASKS_CLAMP["m"] = cm
    if CFG.get("muscle_glow_clamp") and CFG.get("muscle_glow_wfg_from_glow"):
        # t663u: on the charcoal mannequin the lum matte w_fg collapses where the body sits at the backdrop level
        # (lower erectors, back contour), so day4_fibre (colour x w_fg) left gaps and only ridges coloured. A pixel the
        # SOURCE itself glows on, inside the allowed region, is body by definition -> body weight 1 there. Backdrop and
        # everything outside the clamp keep the lum matte; day4_fibre itself is unchanged.
        a_src = F["act_dom"] / max(k_in, 1e-3) if CFG.get("muscle_glow_source") != "regions" else F["act_dom"]
        g_ = cv2.GaussianBlur(np.clip((a_src - 6.0) / 10.0, 0.0, 1.0).astype(np.float32), (0, 0), 1.5)
        w_fg = np.maximum(w_fg, g_)
    out, core_w = muscle.day4_fibre(F, base_rgb.astype(np.float32), w_fg, shorts, is_wrong, pulse_phase)
    return np.clip(out, 0, 255).astype(np.uint8), core_w, ms


LAST_MASKS_CLAMP = {}
