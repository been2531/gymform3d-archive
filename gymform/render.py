"""Orchestrator: shade -> laser/callout -> title/preview per pose; parallel step workers; concat + TTS mux."""
import json
import os
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

from gymform import ctx
from gymform.ctx import CFG, LAST_MASKS, CUR_PLATE, W, H, FPS, d5, b4, SPEC
from gymform import bg, mask, equipment, xray, muscle, layout, tempo


def shade(rgb_1080, step_idx, is_wrong, kp, pulse_phase=0.0):
    """Returns (composited_rgb_uint8, shorts_mask, w_fg)."""
    F = mask.features(rgb_1080)
    K = mask.body_key(F, step_idx, kp)
    foot = equipment.footprint(rgb_1080, kp, F)
    w_fg, eq_solid, shorts_mask = mask.finalize(F, K, foot)
    metal = equipment.metal_mask(F, foot) * (1.0 - K["skin_seg"])   # metal only, never on a skin segment
    LAST_MASKS.update(eq_solid=metal, eq_foot=foot, eq_zone=K["near_wrist"], src=rgb_1080)
    eq_solid = metal
    mono = muscle.silver_skin(F, w_fg, shorts_mask, eq_solid, kp, step_idx, is_wrong)
    mannequin, anat = muscle.activation(F, mono, w_fg, shorts_mask, kp, step_idx, is_wrong, pulse_phase)
    mannequin = xray.apply(rgb_1080, mono, mannequin, anat, shorts_mask, w_fg, eq_solid, kp, step_idx, is_wrong)
    mannequin = skin_tone_lock(mannequin, w_fg, shorts_mask, metal, step_idx)
    back = bg.studio_bg() * (1.0 - 0.80 * bg.contact_shadow(kp)[:, :, None])
    wf = w_fg[:, :, None]
    return np.clip(back * (1.0 - wf) + mannequin * wf, 0, 255).astype(np.uint8), shorts_mask, w_fg


def skin_tone_lock(rgb, w_fg, shorts_mask, metal, step_idx):
    """Deterministic silver lock on the composited mannequin, every step and frame: skin luminance is mapped by a
    monotonic piecewise-linear curve so p75 -> 165, p90 -> 191, max -> 226 (spec), scaling RGB by the same
    factor per pixel so 3D shading ratios are kept. Activation colour, X-Ray window, shorts and metal are untouched."""
    lk = SPEC["mannequin"]["skin_lock_all_steps"]
    f = rgb.astype(np.float32)
    L = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    act = np.maximum(f[:, :, 0] - 0.5 * (f[:, :, 1] + f[:, :, 2]), f[:, :, 1] - 0.5 * (f[:, :, 0] + f[:, :, 2]))
    xw = LAST_MASKS.get("xwin")
    keep_out = np.maximum(np.maximum(shorts_mask, metal), xw if xw is not None else 0.0)
    skin = (w_fg > 0.9) & (keep_out < 0.05) & (act < 12.0) & (L > 46.0)
    if skin.sum() < 2000:
        return rgb
    p75, p90, pmx = (float(np.percentile(L[skin], q)) for q in (75, 90, 99.9))
    xs = [0.0, 46.0, p75, max(p90, p75 + 1), max(pmx, p90 + 2), 255.0]
    ys = [0.0, 46.0, float(lk["p75"]), float(lk["p90"]), float(lk["max"]), float(lk["max"])]
    gain = np.interp(L, xs, ys) / np.maximum(L, 1.0)
    w = np.clip(w_fg, 0, 1) * (1.0 - np.clip(keep_out * 4, 0, 1)) * np.clip((12.0 - act) / 6.0, 0.0, 1.0)
    g = 1.0 + (gain - 1.0) * w
    return np.clip(f * g[:, :, None], 0, 255)


def render_pose(step_idx, seq, kps, p_i, icons, qc=False):
    fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(step_idx)
    pose_frac = p_i / float(max(1, len(seq) - 1))
    if CUR_PLATE.get("step") != step_idx:
        bg.set_step_plate(seq, kps, step_idx)
    rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(seq[p_i], kps[p_i], y_sh, step_idx)
    base_rgb, sh_mask, w_fg = shade(rgb_1080, step_idx, is_w, kp_1080, pulse_phase=pose_frac * 0.75)
    slot_i, deg_s, l1_s, l2_s = layout.callout_for(step_idx, pose_frac)
    fr = layout.draw_laser_and_callout_v4(base_rgb, sh_mask, step_idx, kp_1080, is_w, deg_s, l1_s, l2_s)
    LAST_MASKS["pre_ui"] = fr
    LAST_MASKS["eq_front"] = cv2.GaussianBlur(cv2.dilate(LAST_MASKS["eq_foot"], np.ones((5, 5), np.uint8)), (0, 0), 1.0)
    fr = layout.composite_title_and_preview_v4(fr, layout.fg_for_preview(w_fg), icons, slot_i, is_w, CFG["exercise_name"])
    if qc:
        return fr, base_rgb, sh_mask, w_fg, kp_1080
    return fr


def load_inputs():
    """Clips + cached keypoints (extracted once per clip slice and stored in the day's kp cache)."""
    kps_seqs = pickle.load(open(CFG["kp_cache"], "rb")) if os.path.exists(CFG["kp_cache"]) else {}
    raw_seqs, dirty = {}, False
    for s in (1, 2, 3, 4):
        fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(s)
        seq = d5.load_native_slice_720p(os.path.join(CFG["assets_dir"], fn), i0, i1)
        raw_seqs[s] = seq
        sig = (fn, i0, i1, y_sh)
        if ("sig", s) not in kps_seqs and s in kps_seqs and tuple(d5.STEP_SPECS_DAY5.get(s, ())[:3]) == (fn, i0, i1):
            kps_seqs[("sig", s)] = sig      # legacy cache entry built from the same clip slice
            dirty = True
        if s not in kps_seqs or len(kps_seqs[s]) != len(seq) or kps_seqs.get(("sig", s)) != sig:
            kps_seqs[("sig", s)] = sig
            kps_seqs[s] = d5.extract_day5_kps_1080(seq, s, y_sh)
            dirty = True
    if dirty:
        pickle.dump(kps_seqs, open(CFG["kp_cache"], "wb"))
    return raw_seqs, kps_seqs


def build_icons(raw_seqs, kps_seqs):
    icons = []
    for s in (1, 2, 3, 4):
        fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(s)
        bg.set_step_plate(raw_seqs[s], kps_seqs[s], s)
        rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(raw_seqs[s][-1], kps_seqs[s][-1], y_sh, s)
        sh_rgb, sh_m, w_fg = shade(rgb_1080, s, is_w, kp_1080, pulse_phase=0.75)
        icons.append(b4._make_icon_pair(sh_rgb, kp_1080))
    return icons


def save_qc_sidecar(step_idx, tag, frame_idx, base, shorts_m, w_fg, kp):
    d = CFG["qc_dir"]
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(d, f"step{step_idx}_{tag}")
    z = np.zeros_like(w_fg)
    m8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(pre + "_base.png", cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
    cv2.imwrite(pre + "_wfg.png", m8(w_fg))
    cv2.imwrite(pre + "_shorts.png", m8(shorts_m))
    cv2.imwrite(pre + "_eq.png", m8(LAST_MASKS.get("eq_solid", z)))
    cv2.imwrite(pre + "_eqfoot.png", m8(LAST_MASKS.get("eq_foot", z)))
    cv2.imwrite(pre + "_src.png", cv2.cvtColor(LAST_MASKS["src"], cv2.COLOR_RGB2BGR))
    plate_p = os.path.join(d, f"step{step_idx}_plate.png")
    if CUR_PLATE.get("saved") != (d, step_idx):
        cv2.imwrite(plate_p, cv2.cvtColor(np.clip(CUR_PLATE["rgb"], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        CUR_PLATE["saved"] = (d, step_idx)
    cv2.imwrite(pre + "_preui.png", cv2.cvtColor(LAST_MASKS["pre_ui"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(pre + "_xwin.png", m8(LAST_MASKS.get("xwin", z)))
    cv2.imwrite(pre + "_eqzone.png", m8(LAST_MASKS.get("eq_zone", z)))
    meta_p = os.path.join(d, f"meta_step{step_idx}.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.setdefault("frames", {})[f"step{step_idx}_{tag}"] = {"frame_idx": frame_idx, "step": step_idx, "tag": tag, "kp": kp}
    meta.update({
        "exercise_name": CFG["exercise_name"], "studio_bg": CFG["studio_bg"], "equipment": CFG["equipment"],
        "bone_steps": list(CFG["bone_steps"]), "muscle_steps": list(CFG["muscle_steps"]),
        "title_roi": [0, 410, W, 565], "preview_roi": [190, 232, 890, 424], "fps": FPS, "qc_tags": tempo.QC_TAGS,
    })
    json.dump(meta, open(meta_p, "w"), indent=1)


def render_step_to_mp4(step_idx, raw_seqs, kps_seqs, icons, out, peak_png):
    """Composite + overlays for one step piped straight into x264 (unique poses rendered once)."""
    cv2.setNumThreads(1)
    seq, kps = raw_seqs[step_idx], kps_seqs[step_idx]
    n_f = len(seq)
    mp_ = os.path.join(CFG["qc_dir"], f"meta_step{step_idx}.json")
    if os.path.exists(mp_):
        os.remove(mp_)
    ff = subprocess.Popen(["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset",
                           os.environ.get("GYMFORM_X264_PRESET", "veryfast"), "-crf", "17", "-pix_fmt", "yuv420p",
                           "-threads", "1", out], stdin=subprocess.PIPE)
    cache = {}   # unique poses rendered once (holds / slow tempo reuse frames)
    for f_local in range(tempo.FRAMES_PER_STEP):
        p_i = tempo.pose_index(f_local, n_f)
        if f_local in tempo.QC_TAGS:
            fr_q, base_q, shm_q, wfg_q, kp_q = render_pose(step_idx, seq, kps, p_i, icons, qc=True)
            save_qc_sidecar(step_idx, tempo.QC_TAGS[f_local], (step_idx - 1) * tempo.FRAMES_PER_STEP + f_local,
                            base_q, shm_q, wfg_q, kp_q)
            cache[p_i] = fr_q
        if p_i not in cache:
            cache[p_i] = render_pose(step_idx, seq, kps, p_i, icons)
        if f_local == 50:
            Image.fromarray(cache[p_i]).save(peak_png)
        ff.stdin.write(np.ascontiguousarray(cache[p_i]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"   Step {step_idx}/4 done ({len(cache)} unique poses)", flush=True)


def merge_qc_meta():
    d, meta = CFG["qc_dir"], {}
    for s in (1, 2, 3, 4):
        mp_ = os.path.join(d, f"meta_step{s}.json")
        if os.path.exists(mp_):
            m = json.load(open(mp_))
            frames = meta.get("frames", {})
            frames.update(m.get("frames", {}))
            meta.update(m)
            meta["frames"] = frames
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)


def _load_cfg(argv):
    cfg_path = argv[argv.index("--config") + 1]
    ovr = json.load(open(argv[argv.index("--overrides") + 1])) if "--overrides" in argv else None
    ctx.configure(json.load(open(cfg_path)), ovr)
    return cfg_path


if __name__ == "__main__":
    a = sys.argv
    _load_cfg(a)
    if "--sheet" in a:          # stage: contact sheet + cover from the cached step peaks
        i = a.index("--sheet")
        pk = {s: Image.open(a[i + 1 + s]).convert("RGB") for s in (1, 2, 3, 4)}
        pk[4].save(CFG["out_cover"], quality=95)
        layout.make_sheet(pk, a[i + 1])
    elif "--icons-out" in a:      # stage: preview icons (all 4 steps), pickled for the step workers
        raw, kps = load_inputs()
        pickle.dump(build_icons(raw, kps), open(a[a.index("--icons-out") + 1], "wb"))
    elif "--render-step" in a:  # stage: composite + overlays + encode for one step
        s = int(a[a.index("--render-step") + 1])
        raw, kps = load_inputs()
        ic = pickle.load(open(a[a.index("--icons") + 1], "rb"))
        render_step_to_mp4(s, raw, kps, ic, a[a.index("--out") + 1], a[a.index("--peak") + 1])
