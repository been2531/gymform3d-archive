"""Mannequin surface: locked silver skin + target-muscle fibre activation (crimson wrong / green correct)."""
import math
import cv2
import numpy as np
from gymform.ctx import *  # shared state: W,H,CFG,LAST_MASKS,CUR_PLATE,STEP_SCALE,_XX,_YY,d5,b4,p62,v30
from gymform.xray import xray_window
from gymform.equipment import metal_shade



def silver_skin(F, w_fg, shorts_mask, eq_solid, kp, step_idx, is_wrong):
    """LOCKED bright silver-grey skin (p75~165, p90~191, max 226), dual-anchor lock measured outside the X-Ray window."""
    lum, act_dom = F["lum"], F["act_dom"]
    skin_m = (w_fg > 0.6) & (shorts_mask < 0.2) & (act_dom < 15.0) & (lum > 42.0)
    p90_raw = float(np.percentile(lum[skin_m], 90)) if skin_m.sum() > 500 else 180.0
    hv_silver = np.clip((lum - 24.0) / max(p90_raw - 24.0, 95.0), 0.0, 1.28)
    sulc = np.clip((cv2.GaussianBlur(lum, (0, 0), 2.8) - lum) / 14.0, 0.0, 1.0)
    hf_ridge = lum - cv2.GaussianBlur(lum, (0, 0), 1.6)
    grey = np.clip(48.0 + 143.0 * (hv_silver ** 0.88) + hf_ridge * 0.70 - sulc * 11.0, 44.0, 226.0)
    # Lock skin tone to the Bright Silver-Grey standard (p75~165, p90~191, max 226) with a max-anchored remap.
    skin_q = (w_fg > 0.9) & (shorts_mask < 0.2) & (act_dom < 12.0) & (lum > 46.0)
    from gymform import ctx as _ctx
    if step_idx in CFG["bone_steps"]:   # colour lock is defined on the body OUTSIDE the X-Ray window
        skin_q &= cv2.dilate((xray_window(kp, step_idx in CFG.get('xray_full_steps', [CFG['bone_steps'][0]])) > 0.02).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
    if skin_q.sum() > 2000:
        p75_g = float(np.percentile(grey[skin_q], 75))
        k = float(np.clip((226.0 - 165.0) / max(226.0 - p75_g, 1.0), 0.90, 1.30))
        grey = np.clip(226.0 - (226.0 - grey) * k, 44.0, 226.0)
        # second anchor: highlights locked to p90~191 so every step has the same silver (colour lock)
        p90_g = float(np.percentile(grey[skin_q], 90))
        if p90_g > 170.0:
            k90 = float(np.clip((191.0 - 165.0) / (p90_g - 165.0), 0.75, 1.6))
            hi = grey > 165.0
            grey[hi] = np.clip(165.0 + (grey[hi] - 165.0) * k90, 165.0, 226.0)
    # shadow floor: the p75 anchor stretch (k up to 1.3 around 226) crushed native skin shadows below the source
    # (seated legs: lum 69 -> 49, read as dark patches below the hem). Skin never renders darker than its own
    # native shadow up to lum 62; highlights / p75 / p90 lock are unaffected.
    grey = np.maximum(grey, np.minimum(np.maximum(lum - 0.85 * act_dom, F["min_c"]), 62.0))
    shorts_lum = np.clip(lum * 0.30 + 8.0, 10.0, 22.0)
    grey = metal_shade(grey, F, eq_solid)
    matte = grey * (1.0 - shorts_mask) + shorts_lum * shorts_mask
    return np.stack([matte - 1.5, matte + 0.5, matte + 3.0], axis=-1)


def activation(F, mono, w_fg, shorts_mask, kp, step_idx, is_wrong, pulse_phase):
    """Target-muscle fibre colour (muscle steps) / bone tint seed (bone steps) inside the anatomical gate.
    Returns (mannequin_rgb, anat)."""
    r_ch, g_ch, b_ch, lum, act_dom = F["r"], F["g"], F["b"], F["lum"], F["act_dom"]
    xx, yy = _XX, _YY
    from gymform import ctx as _ctx
    step_idx = _ctx.legacy(step_idx)   # look follows the step's ROLE (bone/muscle x wrong/right), not its position
    sh_r, sh_l = kp["sh_r"], kp["sh_l"]
    sh_y = 0.5 * (sh_r[1] + sh_l[1])
    if step_idx in (1, 3):
        d_shr = np.hypot((xx - sh_r[0]) / 1.15, (yy - sh_r[1]) / 0.95)
        d_shl = np.hypot((xx - sh_l[0]) / 1.15, (yy - sh_l[1]) / 0.95)
        cx_clav = 0.5 * (sh_r[0] + sh_l[0])
        rx_clav = max(90.0, 0.55 * abs(sh_l[0] - sh_r[0]) + 42.0)
        d_clav = np.hypot((xx - cx_clav) / rx_clav, (yy - (sh_y + 12.0)) / 72.0)
        joint = np.maximum(np.clip((135.0 - np.minimum(d_shr, d_shl)) / 42.0, 0.0, 1.0), np.clip((1.18 - d_clav) / 0.35, 0.0, 1.0))
        anat = joint * np.clip((yy - (sh_y - 95.0)) / 28.0, 0.0, 1.0) * np.clip(((sh_y + 135.0) - yy) / 36.0, 0.0, 1.0)
    elif step_idx == 2:
        anat = np.clip((yy - (sh_y - 120.0)) / 28.0, 0.0, 1.0) * np.clip(((sh_y + 135.0) - yy) / 40.0, 0.0, 1.0)
    else:
        d_r = np.hypot((xx - (sh_r[0] - 10.0)) / 92.0, (yy - (sh_r[1] + 22.0)) / 108.0)
        d_l = np.hypot((xx - (sh_l[0] + 10.0)) / 92.0, (yy - (sh_l[1] + 22.0)) / 108.0)
        anat = np.clip((1.35 - np.minimum(d_r, d_l)) / 0.40, 0.0, 1.0)
    anat = anat * w_fg  # colour only inside the clean body mask

    w_act = cv2.GaussianBlur((np.clip((act_dom - 10.0) / 32.0, 0.0, 1.0) ** 0.85).astype(np.float32), (0, 0), 1.8) * (1.0 - shorts_mask) * anat
    core_seed = (np.clip((act_dom - 16.0) / 42.0, 0.0, 1.0) ** 1.08) * anat
    pulse = 1.0 + 0.06 * math.sin(2.0 * math.pi * pulse_phase)
    core_w = np.clip(cv2.GaussianBlur(core_seed.astype(np.float32), (0, 0), 1.0) * w_act * 0.90 * pulse, 0.0, 0.90)

    act_ch = r_ch if is_wrong else g_ch  # Step 4 source glow is green: read fibers from G (R left dark holes)
    gb_ch = 0.55 * g_ch + 0.45 * b_ch
    fiber_fine = (act_ch - cv2.GaussianBlur(act_ch, (0, 0), 4.8)) * 2.05 + (gb_ch - cv2.GaussianBlur(gb_ch, (0, 0), 4.2)) * 1.45
    fiber_bundle = (cv2.GaussianBlur(act_ch, (0, 0), 1.3) - cv2.GaussianBlur(act_ch, (0, 0), 13.0)) * 1.05
    sulc_3d = np.clip(-fiber_fine / 13.0, 0.0, 1.0)
    ridge_3d = np.clip(fiber_fine / 15.0, 0.0, 1.0)
    vol_3d = np.clip((0.58 * act_ch + 0.42 * gb_ch) / 170.0, 0.18, 1.20)
    fiber_mod = np.clip(vol_3d + 0.54 * ridge_3d - 0.58 * sulc_3d + 0.004 * fiber_bundle, 0.12, 1.42)

    bone_edge = np.clip(lum - cv2.GaussianBlur(lum, (0, 0), 5.0), -24.0, 38.0)
    bone_lum = np.clip(cv2.GaussianBlur(lum, (0, 0), 0.6) + 0.85 * bone_edge, 0.0, 255.0)
    L_bone = np.clip((bone_lum - 24.0) / 140.0, 0.0, 1.0)
    bone_core = (np.clip((bone_lum - 115.0) / 55.0, 0.0, 1.0) * np.clip((act_dom - 12.0) / 26.0, 0.0, 1.0) * anat) if step_idx in (1, 3) else 0.0

    if is_wrong:
        if step_idx == 1:
            col_r = np.clip(bone_lum * 1.45 + 38.0 * (L_bone ** 0.85), 22.0, 242.0)
            col_g = np.clip(bone_lum * 0.32, 10.0, 155.0)
            col_b = np.clip(bone_lum * 0.34, 12.0, 142.0)
        else:
            col_r = np.clip(36.0 + 174.0 * fiber_mod + 52.0 * (ridge_3d ** 1.25) - 46.0 * sulc_3d, 22.0, 245.0)
            col_g = np.clip(14.0 + 40.0 * fiber_mod + 28.0 * (ridge_3d ** 1.5) - 18.0 * sulc_3d, 10.0, 102.0)
            col_b = np.clip(16.0 + 42.0 * fiber_mod + 26.0 * (ridge_3d ** 1.5) - 18.0 * sulc_3d, 10.0, 102.0)
    else:
        if step_idx == 3:
            col_g = np.clip(bone_lum * 1.45 + 36.0 * (L_bone ** 0.85), 22.0, 242.0)
            col_r = np.clip(bone_lum * 0.32, 10.0, 138.0)
            col_b = np.clip(bone_lum * 0.76, 14.0, 185.0)
        else:
            col_r = np.clip(14.0 + 36.0 * fiber_mod + 30.0 * (ridge_3d ** 1.5) - 14.0 * sulc_3d, 10.0, 94.0)
            col_g = np.clip(34.0 + 180.0 * fiber_mod + 58.0 * (ridge_3d ** 1.25) - 48.0 * sulc_3d, 24.0, 248.0)
            col_b = np.clip(22.0 + 96.0 * fiber_mod + 42.0 * (ridge_3d ** 1.35) - 28.0 * sulc_3d, 16.0, 170.0)

    act_rgb = np.stack([col_r, col_g, col_b], axis=-1)
    if step_idx == 1:
        # Step 1 = WRONG-posture BONE X-Ray: keep only crisp bone structure (clavicle, acromion/scapula, humeral head),
        # suppress the broad red soft-tissue tint.
        bone_lines = np.clip((bone_edge - 4.0) / 10.0, 0.0, 1.0) * np.clip((bone_lum - 140.0) / 40.0, 0.0, 1.0) * np.clip((act_dom - 10.0) / 25.0, 0.0, 1.0) * anat * (1.0 - shorts_mask)
        core_w = np.clip(0.15 * core_w + 0.90 * bone_lines, 0.0, 0.92)
        # tight gate: clavicle line + acromion / humeral-head discs only (no pec / trap soft tissue)
        d_j = np.minimum(np.hypot(xx - sh_r[0], yy - sh_r[1]), np.hypot(xx - sh_l[0], yy - sh_l[1]))
        clav_d = np.hypot((xx - 0.5 * (sh_r[0] + sh_l[0])) / max(60.0, 0.55 * abs(sh_l[0] - sh_r[0])), (yy - (sh_y + 6.0)) / 34.0)
        tight = np.maximum(np.clip((92.0 - d_j) / 24.0, 0.0, 1.0), np.clip((1.05 - clav_d) / 0.25, 0.0, 1.0))
        core_w = core_w * tight
        bone_core = bone_core * tight
    if step_idx == 3:
        # Step 3 source glow is faint: lift the crisp green bone X-Ray so clavicle/scapula/humeral head read clearly.
        bone_w3 = np.clip((act_dom - 6.0) / 18.0, 0.0, 1.0) * anat * (1.0 - shorts_mask)
        core_w = np.maximum(core_w, 0.80 * cv2.GaussianBlur(bone_w3.astype(np.float32), (0, 0), 0.8))
        bone_core = np.clip(1.8 * bone_core, 0.0, 1.0)
    blend_w = np.clip(core_w + 0.45 * bone_core, 0.0, 0.92) * (1.0 - shorts_mask)
    mannequin = mono * (1.0 - blend_w[:, :, None]) + act_rgb * blend_w[:, :, None]
    return mannequin, anat
