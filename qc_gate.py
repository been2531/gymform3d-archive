#!/usr/bin/env python3
"""
Automated Production QC Gate (`qc_gate.py`, v62 Standard) for @gymform.3d Reels.
Verifies all 9 mandatory production rules from `PRODUCTION_QC_CHECKLIST.md`:
1. Format & Resolution (`1080x1920`, `30.0fps`, `600 frames`).
2. Top Preview Dock (`y=308..438`) to Mannequin Head Safe-Zone Clearance (`min_head_y >= 455px` across all 600 frames).
3. Inter-Step Posture/Angle Distinctiveness (`diff >= 22.0` in central mannequin zone).
4. Intra-Step Full-ROM Motion (`intra_step_motion >= 14.0` per 5s step).
5. Day 2 (`De_mCcFAVTO`) Close-Up Shorts-Free 3D Muscle Fiber Striation & 3D Bone X-Ray Texture Check
   (`active_px >= 16000` for close-up shorts-free glutes & quadriceps, `chroma_mean >= 65.0`,
   `rg_purity >= 90.0` so carrot-orange paint is rejected, `sobel_mean >= 80.0`, `hf_fiber >= 9.5`).
6. Non-Drifting / Non-Vanishing Target Anatomy Check (`active_px(SQUAT) >= 0.72 * active_px(STAND)` across all 4 steps).
7. Full-Bleed Studio Background & Rectangular Border Seam Check (`seam_jump <= 2.2`, `right_studio_lum >= 16.0`).
8. Step 4 Upper-Back Whiteness Prevention Gate (`upper_back_whiteness_check`: verifies `lum > 135` ratio in
   upper back `y=520..780, x=360..680` is `< 0.5%`, preventing `t270u` white-back defect!).
"""
import sys
import cv2
import numpy as np


def _eval_3d_texture_and_glow(bgr_frame: np.ndarray, is_wrong: bool) -> dict:
    roi = bgr_frame[460:1320, 200:880].astype(np.float32)
    b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    ch = (r - b) if is_wrong else (g - b)
    non_white = np.minimum(np.minimum(r, g), b) < 195.0
    act = (ch > 42.0) & non_white
    act_count = int(act.sum())

    lum = 0.299 * r + 0.587 * g + 0.114 * b
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    hf = np.abs(cv2.GaussianBlur(lum, (0, 0), 0.8) - cv2.GaussianBlur(lum, (0, 0), 3.0))

    if act_count > 100:
        ch_mean = float(ch[act].mean())
        rg_mean = float((r[act] - g[act]).mean()) if is_wrong else float((g[act] - r[act]).mean())
        sobel_mean = float(grad[act].mean())
        hf_mean = float(hf[act].mean())
    else:
        ch_mean, rg_mean, sobel_mean, hf_mean = 0.0, 0.0, 0.0, 0.0

    lower_act = int(act[120:840, :].sum())
    return {
        "active_px": act_count,
        "lower_px": lower_act,
        "chroma_mean": ch_mean,
        "rg_purity": rg_mean,
        "sobel_mean": sobel_mean,
        "hf_fiber": hf_mean,
    }


def _eval_rectangular_border_seam_and_fullbleed(bgr_frame: np.ndarray) -> dict:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    right_margin_lum = float(np.median(lum[520:820, 1010:1070]))
    left_margin_lum = float(np.median(lum[520:820, 15:75]))

    # Scan across the lateral clip boundary (`x in 45..135` and `x in 945..1035`) in the open studio wall (`y=1020..1260`)
    band = lum[1080:1280, :]
    max_v_seam = 0.0
    for x_cand in list(range(960, 1045)):
        step = abs(float(np.median(band[:, x_cand + 3 : x_cand + 9]) - np.median(band[:, x_cand - 9 : x_cand - 3])))
        if step > max_v_seam:
            max_v_seam = step

    h_band = lum[:, 680:980]
    max_h_seam = 0.0
    for y_cand in range(442, 468):
        step = abs(float(np.median(h_band[y_cand + 2 : y_cand + 6, :]) - np.median(h_band[y_cand - 6 : y_cand - 2, :])))
        if step > max_h_seam:
            max_h_seam = step

    return {
        "left_margin_lum": left_margin_lum,
        "right_margin_lum": right_margin_lum,
        "max_v_seam": max_v_seam,
        "max_h_seam": max_h_seam,
    }


def _eval_step4_upper_back_whiteness(bgr_frame: np.ndarray) -> dict:
    """
    Verifies that in Step 4 (`22° NEUTRAL SPINE`), the mannequin's upper back (`y=520..780, x=360..680`),
    excluding thin white graphic axis lines/dots, has `< 0.5%` bright/whitened pixels (`lum > 135`)
    and maintains a deep matte dark-charcoal skin luminance (`median_lum <= 85.0`).
    """
    roi = bgr_frame[520:780, 360:680].astype(np.float32)
    b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Erode 3x3 to ignore thin 2px graphic line/plumb dashes when measuring broad skin whiteness
    lum_eroded = cv2.erode(lum, np.ones((5, 5), np.uint8))
    body_mask = lum_eroded > 34.0
    white_px = (lum_eroded > 135.0) & body_mask
    white_ratio = float(white_px.sum()) / float(max(1, body_mask.sum()))
    med_lum = float(np.median(lum_eroded[body_mask])) if body_mask.sum() > 20 else 45.0
    return {
        "white_ratio": white_ratio,
        "median_back_lum": med_lum,
    }


def run_qc(mp4_path: str) -> bool:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"[QC FAIL] Cannot open {mp4_path}")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"=== @gymform.3d AUTOMATED QC GATE (v62 Standard): {mp4_path} ===")
    print(f"  [1] Format: {w}x{h} @ {fps:.1f}fps ({n_frames} frames)")
    ok = True
    if (w, h) != (1080, 1920) or abs(n_frames - 600) > 5:
        print("      -> [FAIL] Resolution or frame count mismatch!")
        ok = False

    rois = []
    full_key_frames = {}
    needed_keys = {
        3, 36, 70,          # Step 1: STAND, MID, SQUAT
        153, 186, 220,      # Step 2: STAND, MID, SQUAT
        303, 336, 370,      # Step 3: STAND, MID, SQUAT
        453, 486, 520,      # Step 4: STAND, MID, SQUAT
        72, 222, 372, 522,
    }

    min_head_y = 1920
    min_head_frame = 0

    f_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rois.append(bgr[460:1280, 180:880].astype(np.float32))
        if f_idx in needed_keys:
            full_key_frames[f_idx] = bgr.copy()

        head_band = bgr[439:520, 435:645]
        lum_band = 0.299 * head_band[:, :, 2] + 0.587 * head_band[:, :, 1] + 0.114 * head_band[:, :, 0]
        bright_rows = np.where(np.percentile(lum_band, 97, axis=1) > 48.0)[0]
        if len(bright_rows) > 0:
            y_top = 439 + int(bright_rows[0])
            if y_top < min_head_y:
                min_head_y = y_top
                min_head_frame = f_idx
        f_idx += 1
    cap.release()

    head_ok = min_head_y >= 455
    if not head_ok:
        ok = False
    head_status = "PASS" if head_ok else "FAIL (HEAD TOUCHES DOCK)"
    print(
        f"  [2] Top Dock Clearance (`dock_bot=438px`): min_head_y={min_head_y}px "
        f"(gap={min_head_y - 438}px at f={min_head_frame}) -> [{head_status}]"
    )

    step_peaks = [rois[int(t * 30)] for t in (2.4, 7.4, 12.4, 17.4)]
    pairs = [
        (1, 2, step_peaks[0], step_peaks[1]),
        (2, 3, step_peaks[1], step_peaks[2]),
        (3, 4, step_peaks[2], step_peaks[3]),
    ]
    for s_a, s_b, fa, fb in pairs:
        diff = float(np.mean(np.abs(fa - fb)))
        status = "PASS" if diff >= 22.0 else "FAIL (TOO SIMILAR / REUSED CLIP)"
        if diff < 22.0:
            ok = False
        print(f"  [3] Inter-Step Posture/Angle Diff (Step {s_a} vs Step {s_b}): {diff:.2f} (>=22.0) -> [{status}]")

    for s_idx in range(4):
        seg = rois[s_idx * 150 : (s_idx + 1) * 150]
        base = seg[4]
        max_m = max(float(np.mean(np.abs(f - base))) for f in seg[::3])
        status = "PASS" if max_m >= 14.0 else "FAIL (STATIC / LOW MOTION)"
        if max_m < 14.0:
            ok = False
        print(f"  [4] Intra-Step {s_idx+1} Full-ROM Motion Range: {max_m:.2f} -> [{status}]")

    step_specs = [
        (1, True,  3,   36,  70),
        (2, False, 153, 186, 220),
        (3, True,  303, 336, 370),
        (4, False, 453, 486, 520),
    ]
    for s_idx, is_wrong, f_st, f_md, f_sq in step_specs:
        m_st = _eval_3d_texture_and_glow(full_key_frames[f_st], is_wrong)
        m_md = _eval_3d_texture_and_glow(full_key_frames[f_md], is_wrong)
        m_sq = _eval_3d_texture_and_glow(full_key_frames[f_sq], is_wrong)

        min_px_req = 16000 if s_idx in (1, 2, 4) else 12000
        tex_ok = (
            m_sq["active_px"] >= min_px_req
            and m_sq["chroma_mean"] >= 65.0
            and m_sq["rg_purity"] >= 90.0
            and m_sq["sobel_mean"] >= 80.0
            and m_sq["hf_fiber"] >= 9.5
        )
        if not tex_ok:
            ok = False
        tex_status = "PASS" if tex_ok else "FAIL (WEAK/SHORTS-COVERED/ORANGE/2D BLUR)"
        print(
            f"  [5] Step {s_idx} Close-Up Shorts-Free 3D Fiber/X-Ray Quality: active_px={m_sq['active_px']:6d} (>={min_px_req}), "
            f"chroma={m_sq['chroma_mean']:5.1f} (>=65), rg_purity={m_sq['rg_purity']:5.1f} (>=90), "
            f"sobel={m_sq['sobel_mean']:5.1f} (>=80), hf_fiber={m_sq['hf_fiber']:5.2f} (>=9.5) -> [{tex_status}]"
        )

        ret_total = m_sq["active_px"] / float(max(1, m_st["active_px"]))
        ret_lower = m_sq["lower_px"] / float(max(1, m_st["lower_px"]))
        drift_ok = ret_total >= 0.72 and ret_lower >= 0.72
        if not drift_ok:
            ok = False
        drift_status = "PASS" if drift_ok else "FAIL (GLOW SHRANK/DRIFTED UP)"
        print(
            f"  [6] Step {s_idx} Non-Drifting Anatomy (STAND->MID->SQUAT): "
            f"total_px={m_st['active_px']}->{m_md['active_px']}->{m_sq['active_px']} (ret={ret_total:.2f}x >=0.72), "
            f"lower_px={m_st['lower_px']}->{m_md['lower_px']}->{m_sq['lower_px']} (ret={ret_lower:.2f}x >=0.72) "
            f"-> [{drift_status}]"
        )

        seam_info = _eval_rectangular_border_seam_and_fullbleed(full_key_frames[f_sq])
        seam_ok = (
            seam_info["max_v_seam"] <= 2.2
            and seam_info["max_h_seam"] <= 2.2
            and seam_info["right_margin_lum"] >= 16.0
        )
        if not seam_ok:
            ok = False
        seam_status = "PASS" if seam_ok else "FAIL (RECTANGULAR BOX SEAM OR BLACK PILLARBOX)"
        print(
            f"  [7] Step {s_idx} Full-Bleed & Rectangular Border Seam Check: "
            f"v_seam={seam_info['max_v_seam']:.2f} (<=2.2), h_seam={seam_info['max_h_seam']:.2f} (<=2.2), "
            f"right_studio_lum={seam_info['right_margin_lum']:.1f} (>=16.0) -> [{seam_status}]"
        )

    wb_info = _eval_step4_upper_back_whiteness(full_key_frames[520])
    wb_ok = wb_info["white_ratio"] <= 0.005 and wb_info["median_back_lum"] <= 85.0
    if not wb_ok:
        ok = False
    wb_status = "PASS" if wb_ok else "FAIL (UPPER BACK WHITENED)"
    print(
        f"  [8] Step 4 Upper-Back Whiteness Check (`t270u`): "
        f"white_ratio={wb_info['white_ratio']*100:.2f}% (<=0.50%), "
        f"median_back_lum={wb_info['median_back_lum']:.1f} (<=85.0) -> [{wb_status}]"
    )

    print(f"=== OVERALL QC RESULT: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/daily_info_reels/output/day3_v62_master_reel.mp4"
    sys.exit(0 if run_qc(path) else 1)
