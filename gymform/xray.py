"""Reusable X-Ray module: approved Day 1 v36 look (glassy local window, luminous native 3D bones, zero blur)."""
import math
import cv2
import numpy as np
from gymform.ctx import *  # shared state: W,H,CFG,LAST_MASKS,CUR_PLATE,STEP_SCALE,_XX,_YY,d5,b4,p62,v30
from gymform.mask import _seg_dist, _ext


def xray_window(kp: dict, is_wrong: bool) -> np.ndarray:
    """
    Local X-Ray window (soft 0..1) around the target bones only: near-side shoulder girdle (clavicle to the
    midline, scapula/acromion, humeral head) + near upper arm (humerus). Step 1 (wrong posture, both arms
    overhead, rib cage in the native X-Ray) also opens both upper arms and the upper rib cage.
    """
    sh_r, sh_l = kp["sh_r"], kp["sh_l"]
    sw = abs(sh_l[0] - sh_r[0])
    sh_mid = (0.5 * (sh_r[0] + sh_l[0]), 0.5 * (sh_r[1] + sh_l[1]))
    win = np.zeros((H, W), np.float32)
    soft = 20.0
    win = np.maximum(win, np.clip((95.0 - np.hypot(_XX - sh_r[0], _YY - sh_r[1])) / soft, 0.0, 1.0))
    win = np.maximum(win, np.clip((48.0 - _seg_dist(sh_r, (sh_mid[0], sh_mid[1] + 10.0), 0, 0, W, H)) / soft, 0.0, 1.0))
    win = np.maximum(win, np.clip((62.0 - _seg_dist(sh_r, _ext(sh_r, kp["elb_r"], 1.05), 0, 0, W, H)) / soft, 0.0, 1.0))
    if is_wrong:
        win = np.maximum(win, np.clip((95.0 - np.hypot(_XX - sh_l[0], _YY - sh_l[1])) / soft, 0.0, 1.0))
        win = np.maximum(win, np.clip((48.0 - _seg_dist(sh_l, (sh_mid[0], sh_mid[1] + 10.0), 0, 0, W, H)) / soft, 0.0, 1.0))
        win = np.maximum(win, np.clip((62.0 - _seg_dist(sh_l, _ext(sh_l, kp["elb_l"], 1.05), 0, 0, W, H)) / soft, 0.0, 1.0))
        rib = np.hypot((_XX - sh_mid[0]) / max(110.0, 0.42 * sw + 30.0), (_YY - (sh_mid[1] + 150.0)) / 170.0)
        win = np.maximum(win, np.clip((1.0 - rib) / 0.18, 0.0, 1.0))
    return win


def xray_day1_v36(rgb: np.ndarray, mono: np.ndarray, gate: np.ndarray, shorts_mask: np.ndarray, tint: str = "crimson",
                  glass_gate: np.ndarray | None = None, glass_floor: float = 0.0, src_tint: str | None = None) -> np.ndarray:
    """
    Reusable X-Ray module = the APPROVED Day 1 v36 'Step 1 native 3D X-Ray' method
    (gymform_engine.prepare_xray_bone_frame + render_xray_skeletal_frame), generalised to any tint/gate:
      1. inside the tinted X-Ray zone the body luminance is replaced by a dark glassy cavity whose only bright
         structure is the native bone detail (xray_cavity_lum = 28 + 135 * bone_sharp^1.15) -> see-through body;
      2. bone pixels (strongly tinted + bright in the native render) are filled with the Day 1 luminous bone ramp
         (tint -> hot core) at alpha bone_mask^0.8 * 0.96, with the Day 1 0.48 X-Ray window darkening around them.
    Pure per-pixel shading of the native 3D frame: no warps, no drawn shapes. Window/bone-mask blur kept tiny.
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    # t541u: the native clip's own bone tint (detection) can differ from the step's role tint (output colour):
    # a matched native bone clip of the other colour is colour-GRADED (its own bone pixels), never repainted
    src_tint = src_tint or tint
    if src_tint == "crimson":
        t_any = np.clip((r - b - 12.0) / 95.0, 0.0, 1.0)
        detail = 0.38 * r + 0.62 * g
        bone_m = np.clip((r - b - 82.0) / 95.0, 0.0, 1.0) * np.clip((r - 130.0) / 75.0, 0.0, 1.0)
    else:
        # emerald source glow is ~2x weaker than the crimson one (g-r p50~52 vs r-b p50~74): thresholds rescaled
        t_any = np.clip((g - r - 8.0) / 50.0, 0.0, 1.0)
        detail = 0.38 * g + 0.62 * r
        bone_m = np.clip((g - r - 30.0) / 40.0, 0.0, 1.0) * np.clip((g - 120.0) / 70.0, 0.0, 1.0)
    bone_sharp = np.clip((detail + (detail - cv2.GaussianBlur(detail, (0, 0), 1.5)) * 2.1 - 20.0) / 205.0, 0.0, 1.0)
    # soft, structureless glow (e.g. armpit haze) is not bone: weight the colour-based bone mask by native
    # high-frequency structure so only crisp 3D bone edges/fills light up
    hf = np.abs(detail - cv2.GaussianBlur(detail, (0, 0), 2.5))
    structured = np.clip((hf - 1.5) / 5.0, 0.0, 1.0)
    bone_m = bone_m * (0.20 + 0.80 * structured)
    # native bright bone edges inside the tinted zone also count as bone (Day 5 sources tint bones less than Day 1)
    bone_m = np.maximum(bone_m, 0.9 * np.clip(t_any * 1.6, 0.0, 1.0) * np.clip((bone_sharp - 0.50) / 0.22, 0.0, 1.0))
    if glass_gate is not None and glass_floor > 0.0 and src_tint != "crimson":
        # Step 3 source tints only thin bone outlines: the native 3D bone/skeletal relief inside the glass
        # window (positive high-frequency ridges of the render: clavicle, scapular spine/acromion, humeral head,
        # humerus, rib arcs) is lit as luminous bone. Per-pixel on the native frame, no drawn shapes, no blur.
        lum_n = 0.299 * r + 0.587 * g + 0.114 * b
        ridge = lum_n - cv2.GaussianBlur(lum_n, (0, 0), 3.0)
        ridge_m = np.clip((ridge - 1.5) / 6.0, 0.0, 1.0)
        bone_m = np.maximum(bone_m, ridge_m * glass_gate)
        bone_m = np.maximum(bone_m, np.clip((g - r - 14.0) / 30.0, 0.0, 1.0) * glass_gate)
        bone_sharp = np.maximum(bone_sharp, np.clip(0.72 + ridge / 25.0, 0.0, 1.0) * ridge_m * glass_gate)
        # native emerald-tinted bone (clavicle/humeral head outlines) gets the same luminous core as Step 1
        bone_sharp = np.maximum(bone_sharp, np.clip((g - r - 14.0) / 30.0, 0.0, 1.0) * 0.85 * glass_gate)
    zone2 = t_any * gate
    if glass_gate is not None and glass_floor > 0.0:
        # see-through glass body over the whole shoulder girdle + upper arm (Day 1 look), even where the
        # native emerald tint is faint; bones stay native structure only
        zone2 = np.maximum(zone2, glass_floor * glass_gate)
    zone = (zone2 * (1.0 - shorts_mask))[:, :, None]

    if glass_gate is not None and glass_floor > 0.0:
        # outside the native tint the body detail is plain skin, not bone: the glass there keeps only the native
        # surface structure (high-pass edges of the 3D render), so the body reads dark and see-through
        t_w = np.clip(t_any * 1.6, 0.0, 1.0)
        surf = np.clip((hf - 1.0) / 16.0, 0.0, 1.0) * 0.55
        bone_sharp_g = t_w * bone_sharp + (1.0 - t_w) * surf
    else:
        bone_sharp_g = bone_sharp
    cav = 28.0 + 135.0 * (bone_sharp_g ** 1.15)
    norm = (cav / 255.0) ** 0.92
    glass = np.stack([30.0 + 202.0 * norm, 32.0 + 204.0 * norm, 36.0 + 206.0 * norm], axis=-1)
    out = mono * (1.0 - zone) + glass * zone

    bm = cv2.GaussianBlur(bone_m.astype(np.float32), (0, 0), 1.0) * gate * (1.0 - shorts_mask)
    window = np.clip(cv2.GaussianBlur(bm, (0, 0), 6.0) * 1.1, 0.0, 0.65)
    out = out * (1.0 - window[:, :, None] * 0.48)

    bv = np.clip((bone_sharp - 0.12) / 0.78, 0.0, 1.0)
    hot = 110.0 * (bv ** 1.25) + 120.0 * (bv ** 2.6)
    if tint == "crimson":   # Day 1 ramp: deep crimson -> orange-gold -> white-hot core
        col = np.stack([95.0 + 160.0 * (bv ** 0.55), 14.0 + hot, 10.0 + 45.0 * (bv ** 1.5) + 135.0 * (bv ** 3.0)], axis=-1)
    else:                   # same ramp mirrored to emerald: deep green -> mint -> white core
        col = np.stack([10.0 + 45.0 * (bv ** 1.5) + 135.0 * (bv ** 3.0), 95.0 + 160.0 * (bv ** 0.55), 14.0 + hot], axis=-1)
    # t530u/t543u: bones must read bright near-white with only a subtle tint (dark saturated bone = invisible):
    # the low end of the ramp is lifted toward a soft white, the hot core is unchanged
    col = col + (np.array([228.0, 230.0, 234.0], np.float32) - col) * (0.42 * (1.0 - bv))[:, :, None]
    a = np.clip((bm ** 0.80) * 0.96, 0.0, 0.96)[:, :, None]
    out = out * (1.0 - a) + np.clip(col, 0, 255) * a
    return out


def bone_signal(rgb, src_tint):
    """Native bone pixels of the clip (its own tinted, bright, structured bone render) -> (bone_m, bv, act)."""
    r, g, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    if src_tint == "crimson":
        t_any = np.clip((r - b - 12.0) / 95.0, 0.0, 1.0)
        detail = 0.38 * r + 0.62 * g
        bone_m = np.clip((r - b - 82.0) / 95.0, 0.0, 1.0) * np.clip((r - 130.0) / 75.0, 0.0, 1.0)
    else:
        t_any = np.clip((g - r - 8.0) / 50.0, 0.0, 1.0)
        detail = 0.38 * g + 0.62 * r
        bone_m = np.clip((g - r - 30.0) / 40.0, 0.0, 1.0) * np.clip((g - 120.0) / 70.0, 0.0, 1.0)
    bone_sharp = np.clip((detail + (detail - cv2.GaussianBlur(detail, (0, 0), 1.5)) * 2.1 - 20.0) / 205.0, 0.0, 1.0)
    hf = np.abs(detail - cv2.GaussianBlur(detail, (0, 0), 2.5))
    bone_m = bone_m * (0.20 + 0.80 * np.clip((hf - 1.5) / 5.0, 0.0, 1.0))
    bone_m = np.maximum(bone_m, 0.9 * np.clip(t_any * 1.6, 0.0, 1.0) * np.clip((bone_sharp - 0.50) / 0.22, 0.0, 1.0))
    act = np.maximum(r - 0.5 * (g + b), g - 0.5 * (r + b))
    return bone_m, np.clip((bone_sharp - 0.12) / 0.78, 0.0, 1.0), act


def bone_colour(bv, tint):
    """Day 1 v36 luminous ramp, lifted toward a soft white (t530u bright bones, subtle tint)."""
    hot = 110.0 * (bv ** 1.25) + 120.0 * (bv ** 2.6)
    if tint == "crimson":
        col = np.stack([95.0 + 160.0 * (bv ** 0.55), 14.0 + hot, 10.0 + 45.0 * (bv ** 1.5) + 135.0 * (bv ** 3.0)], axis=-1)
    else:
        col = np.stack([10.0 + 45.0 * (bv ** 1.5) + 135.0 * (bv ** 3.0), 95.0 + 160.0 * (bv ** 0.55), 14.0 + hot], axis=-1)
    return np.clip(col + (np.array([228.0, 230.0, 234.0], np.float32) - col) * (0.42 * (1.0 - bv))[:, :, None], 0, 255)


def apply_target_bones(rgb_1080, mono, w_fg, shorts_mask, eq_solid, kp, step_idx, is_wrong):
    """t552u: bones work like muscle fibre. Only the native bone pixels INSIDE the exercise's target-bone region
    are shown (regraded to the role tint); every other pixel is the locked Day 4 skin (`mono`) - no glass, no
    window darkening, no tint, no see-through on the surrounding body. Native bone glow outside the region is
    removed by the skin regrade (de-glowed luminance), never painted over with shapes."""
    from gymform import bones
    src_tint = CFG["clips"][str(step_idx)].get("native_tint") or ("crimson" if is_wrong else "emerald")
    reg = bones.region(kp, CFG["target_bones"], (H, W)) * w_fg * (1.0 - shorts_mask)
    reg = reg * (1.0 - np.clip(cv2.dilate(eq_solid.astype(np.float32), np.ones((9, 9), np.uint8)), 0, 1))
    bone_m, bv, _ = bone_signal(rgb_1080, src_tint)
    a = np.clip((cv2.GaussianBlur(bone_m.astype(np.float32), (0, 0), 0.8) ** 0.80) * 0.96, 0.0, 0.96) * reg
    LAST_MASKS["xwin"] = a                     # = the shown bone pixels only (skin lock + QC exclude just these)
    LAST_MASKS["bone_region"] = reg
    col = bone_colour(bv, "crimson" if is_wrong else "emerald")
    mono = flatten_offtarget_relief(mono, bone_m, w_fg, reg, eq_solid)
    return mono * (1.0 - a[:, :, None]) + col * a[:, :, None]


def flatten_offtarget_relief(mono, bone_m, w_fg, reg, eq_solid, strength=None):
    """t557u: the native clip bakes bone edges (forearm radius/ulna etc.) into the skin as relief lines. Outside
    the target-bone region, flatten that relief by blending the skin toward a foreground-normalised blur of its
    OWN pixels (a local tone grade - nothing is painted, no shape is added). Inside the target region and on the
    rest of the body the skin is untouched."""
    k = float(strength if strength is not None else CFG.get("xray_relief_flatten", 0.85))
    if k <= 0:
        return mono
    eqd = np.clip(cv2.dilate(eq_solid.astype(np.float32), np.ones((15, 15), np.uint8)), 0, 1)
    bm = cv2.GaussianBlur(cv2.dilate(bone_m.astype(np.float32), np.ones((7, 7), np.uint8)), (0, 0), 2.0)
    m = np.clip(bm * 1.2, 0, 1) * np.clip((w_fg - 0.5) / 0.4, 0, 1) * (1.0 - np.clip(reg * 3.0, 0, 1)) * (1.0 - eqd)
    wf = np.clip((w_fg - 0.5) / 0.4, 0, 1).astype(np.float32)
    num = cv2.GaussianBlur(mono * wf[:, :, None], (0, 0), 6.0)
    den = cv2.GaussianBlur(wf, (0, 0), 6.0)[:, :, None] + 1e-4
    flat = num / den
    LAST_MASKS["relief_flat"] = m
    mk = (k * m)[:, :, None]
    return mono * (1.0 - mk) + flat * mk


def apply(rgb_1080, mono, mannequin, anat, shorts_mask, w_fg, eq_solid, kp, step_idx, is_wrong):
    if step_idx in CFG["bone_steps"] and CFG.get("target_bones"):
        return apply_target_bones(rgb_1080, mono, w_fg, shorts_mask, eq_solid, kp, step_idx, is_wrong)
    return _apply_window(rgb_1080, mono, mannequin, anat, shorts_mask, w_fg, eq_solid, kp, step_idx, is_wrong)


def _apply_window(rgb_1080, mono, mannequin, anat, shorts_mask, w_fg, eq_solid, kp, step_idx, is_wrong):
    """Local Day-1 v36 X-Ray window on bone steps; outside the window the locked silver skin is untouched."""
    if step_idx in CFG["bone_steps"] and CFG.get("xray_style") == "day1_v36":
        win = xray_window(kp, step_idx in CFG.get('xray_full_steps', [CFG['bone_steps'][0]])) * w_fg
        win = win * (1.0 - cv2.GaussianBlur(cv2.dilate(eq_solid, np.ones((9, 9), np.uint8)), (0, 0), 2.0))
        LAST_MASKS["xwin"] = win
        src_tint = CFG["clips"][str(step_idx)].get("native_tint") if "clips" in CFG else None
        if is_wrong:
            mannequin_x = xray_day1_v36(rgb_1080, mono, (anat if src_tint in (None, "crimson") else np.maximum(anat, win)) * win,
                                        shorts_mask, tint="crimson", src_tint=src_tint)
        else:
            mannequin_x = xray_day1_v36(rgb_1080, mono, np.maximum(anat, win) * win, shorts_mask, tint="emerald",
                                        glass_gate=win, glass_floor=0.80, src_tint=src_tint)
        wv = win[:, :, None]
        return mono * (1.0 - wv) + mannequin_x * wv
    LAST_MASKS["xwin"] = np.zeros((H, W), np.float32)
    return mannequin
