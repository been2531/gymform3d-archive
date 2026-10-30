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
from gymform import bg, mask, equipment, xray, muscle, layout, tempo, native


def step_phase(step_idx):
    return tempo.step_phase(ctx.steps().index(step_idx), bool(CFG.get("phase_continuous_cut")))


def edit_base(step_idx):
    """Pair step whose clip is the motion base of this step's motion-preserving edit (edit_of), else None."""
    c = CFG["clips"][str(step_idx)]
    partner = ctx.PAIRS.get(step_idx) or next((a for a, b in ctx.PAIRS.items() if b == step_idx), None)
    if not c.get("edit_of") or partner is None or CFG["clips"][str(partner)]["file"] != c["edit_of"]:
        return None
    return partner if partner in SEQS else None


def geo_frame(step_idx, p_i):
    """ROOT CAUSE fix for Day 6+ X-Ray video edits: the edit keeps geometry, camera and motion of its base clip but
    renders the body as dark see-through glass with glowing bones (and cyan/emerald glow spill), so keys computed
    on the edit read the X-Ray cavity as backdrop, the glass as bench and the glow as equipment. For an edit, ALL
    geometry (body key, equipment footprint, static bench, shorts) is taken from the base frame of the same pose;
    only the colour/anatomy pixels come from the edit."""
    pb = edit_base(step_idx)
    if pb is None or p_i >= len(SEQS[pb]):
        return None
    y_sh = ctx.step_spec(step_idx)[5]
    return b4.transform_frame_and_kp_1080p(SEQS[pb][p_i], KPS[step_idx][p_i], y_sh, step_idx)[0]


SEQS, KPS, GEO_PLATE = {}, {}, {}
_GEO_H = {}


def _geo_hash():
    """Cache key of the base-frame geometry: config + every module that changes the key/footprint."""
    if "h" not in _GEO_H:
        import hashlib
        hh = hashlib.sha1(json.dumps({k: v for k, v in CFG.items() if k != "qc_dir"}, sort_keys=True, default=str).encode())
        for m in ("mask", "equipment", "bg", "ctx", "render"):
            hh.update(open(os.path.join(ctx.ROOT, "gymform", f"{m}.py"), "rb").read())
        _GEO_H["h"] = hh.hexdigest()[:12]
    return _GEO_H["h"]


def geo_cache_path(base_step, p_i):
    d = os.path.join(ctx.CACHE_DIR, "geo", f"day{int(CFG['day']):02d}_s{base_step}_{_geo_hash()}")
    return os.path.join(d, f"f{p_i:03d}.npz")


def is_edit_base(step_idx):
    return any(edit_base(e) == step_idx for e in ctx.steps() if e != step_idx)


def save_geo(step_idx, p_i):
    """SPEED (t529u): the base step already computes the base frame's body key + footprint for the same pose and
    transform; store them (bit-packed) so the edit step reuses them instead of a second cutout per frame."""
    kb, fb = LAST_MASKS.get("geo_kb"), LAST_MASKS.get("geo_fb")
    if kb is None:
        return
    p = geo_cache_path(step_idx, p_i)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p[:-4] + ".part.npz"
    np.savez_compressed(tmp, kb=np.packbits(kb), fb=np.packbits(fb))
    os.replace(tmp, p)


def load_geo(step_idx, p_i):
    pb = edit_base(step_idx)
    p = geo_cache_path(pb, p_i) if pb else None
    if not p or not os.path.exists(p):
        return None
    z = np.load(p)
    return (np.unpackbits(z["kb"])[: H * W].reshape(H, W), np.unpackbits(z["fb"])[: H * W].reshape(H, W))


def _with_geo_plate(step_idx, fn):
    """Run fn with the BASE clip's backdrop plate (same transform) as the current plate, then restore."""
    own = dict(CUR_PLATE)
    if step_idx not in GEO_PLATE:
        pb = edit_base(step_idx)
        bg.set_step_plate(SEQS[pb], KPS[step_idx], step_idx, src=CFG["clips"][str(pb)]["file"])
        GEO_PLATE[step_idx] = dict(CUR_PLATE)
    CUR_PLATE.clear(); CUR_PLATE.update(GEO_PLATE[step_idx])
    try:
        return fn()
    finally:
        GEO_PLATE[step_idx] = dict(CUR_PLATE)
        CUR_PLATE.clear(); CUR_PLATE.update(own)


def shade(rgb_1080, step_idx, is_wrong, kp, pulse_phase=0.0, geo=None, geo_masks=None):
    """Returns (composited_rgb_uint8, shorts_mask, w_fg). geo = base frame of a motion-preserving edit (geometry)."""
    F = mask.features(rgb_1080)
    K = mask.body_key(F, step_idx, kp)
    foot = equipment.footprint(rgb_1080, kp, F)
    inner = None
    if geo is not None:
        # body silhouette: the base frame's key (eroded 3px) is body; the edit's own key only within 4px of it
        # (drops the X-Ray glow spill). Equipment: the edit's own footprint clamped to the base footprint +3px
        # (computed against the BASE plate), with the base core filled (bone glow over the handles neither
        # bloats nor holes the dumbbells). Bench/static: the edit's own, never on the base body core.
        if geo_masks is not None:      # cached by the base step (same pose, same transform): no second cutout
            kb, fb = geo_masks
        else:
            Fg = mask.features(geo)
            kb = (mask.body_key(Fg, edit_base(step_idx), kp)["w_raw"] > 0.5).astype(np.uint8)   # base = muscle key
            fb = (_with_geo_plate(step_idx, lambda: equipment.footprint(geo, kp, Fg)) > 0.5).astype(np.uint8)
        el = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        inner = cv2.erode(kb, el(3)).astype(np.float32)
        K["w_raw"] = np.maximum(K["w_raw"] * cv2.dilate(kb, el(4)), inner).astype(np.float32)
        foot = np.maximum(foot * cv2.dilate(fb, el(3)), cv2.erode(fb, el(2))).astype(np.float32)
    if geo is None:   # this frame's own geometry (stored by a base step for its edit twin)
        LAST_MASKS["geo_kb"], LAST_MASKS["geo_fb"] = (K["w_raw"] > 0.5), (foot > 0.5)
    w_fg, eq_solid, shorts_mask = mask.finalize(F, K, foot)
    metal = equipment.metal_mask(F, foot) * (1.0 - K["skin_seg"])   # metal only, never on a skin segment
    LAST_MASKS.update(eq_solid=metal, eq_foot=foot, eq_zone=K["near_wrist"], src=rgb_1080, geo=geo)
    eq_solid = metal
    mono = muscle.silver_skin(F, w_fg, shorts_mask, eq_solid, kp, step_idx, is_wrong)
    mannequin, anat = muscle.activation(F, mono, w_fg, shorts_mask, kp, step_idx, is_wrong, pulse_phase)
    stat = equipment.static_equipment(rgb_1080, F, kp, K, shorts_mask)
    if inner is not None:   # X-Ray glass of the edit is body, never bench
        stat = stat * (1.0 - inner)
    # t567u: motionless mannequin parts (feet, head) are in the temporal plate -> they are body, never bench
    sb_body, sb_rim = equipment.static_body(F, K, kp, stat)
    stat = stat * (1.0 - np.maximum(sb_body, sb_rim))
    if sb_rim.max() > 0:   # pieces of the static layer cut loose by the rim removal are ghosts, not bench
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats((stat > 0.5).astype(np.uint8))
        small = np.isin(lab_, [i for i in range(1, n_) if st_[i, cv2.CC_STAT_AREA] < 800])
        stat = np.where(small, 0.0, stat).astype(np.float32)
        sb_rim = np.maximum(sb_rim, small.astype(np.float32))
    w_fg = np.maximum(w_fg, sb_body) * (1.0 - sb_rim)
    LAST_MASKS["static_eq"] = stat
    # X-Ray glass never tints the bench (static equipment is excluded from the X-Ray window like the dumbbells)
    mannequin = xray.apply(rgb_1080, mono, mannequin, anat, shorts_mask, w_fg, np.maximum(eq_solid, stat), kp, step_idx, is_wrong)
    mannequin = skin_tone_lock(mannequin, w_fg, np.maximum(shorts_mask, stat), metal, step_idx)
    # static equipment (bench) = native source pixels, fully opaque, never faded into the backdrop
    sw = stat[:, :, None]
    mannequin = mannequin * (1.0 - sw) + rgb_1080.astype(np.float32) * sw
    w_fg = np.maximum(w_fg, stat)
    # t592u: Day 4 has a flat 35/39/43 backdrop and no floor shadow; configs with flat_look keep it that way
    shadow = 0.0 if CFG.get("flat_look") else 0.80 * bg.contact_shadow(kp)[:, :, None]
    back = bg.studio_bg() * (1.0 - shadow)
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
    if xw is not None:   # statistics on the same skin as QC (f)/(i): 25 px clear of the X-Ray window (glow spill)
        skin &= cv2.dilate((xw > 0.02).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
    if CFG.get("flat_look"):
        return _flat_skin_lock(f, L, act, w_fg, keep_out, lk)
    if skin.sum() < 2000:
        return rgb
    p75, p90, pmx = (float(np.percentile(L[skin], q)) for q in (75, 90, 99.9))
    xs = [0.0, 46.0, p75, max(p90, p75 + 1), max(pmx, p90 + 2), 255.0]
    ys = [0.0, 46.0, float(lk["p75"]), float(lk["p90"]), float(lk["max"]), float(lk["max"])]
    gain = np.interp(L, xs, ys) / np.maximum(L, 1.0)
    # full lock on every solid-body pixel (w_fg >= 0.9 is skin for QC (f)); only the soft matte edge blends
    w = np.clip((w_fg - 0.5) / 0.4, 0, 1) * (1.0 - np.clip(keep_out * 4, 0, 1)) * np.clip((12.0 - act) / 6.0, 0.0, 1.0)
    g = 1.0 + (gain - 1.0) * w
    return np.clip(f * g[:, :, None], 0, 255)


def _flat_skin_lock(f, L, act, w_fg, keep_out, lk):
    """t592u ROOT CAUSE (shading back in Day 6 v2/v3): the lock only remaps skin above lum 46, so the dark shaded
    side of the Day 4-tone charcoal clips (lum 15-45) passed through untouched. Day 4's own shader floors the body
    at 42-48 and lifts shadows with a 0.78 gamma (build_day4_rdl_master grey_back_exact). Same idea here: one
    monotonic curve over ALL skin (no lum gate) p5 -> 52, p50 -> spec p50, p75/p90/max -> spec, so the body reads
    as one even matte tone without a dark side, while 3D ridges keep their order."""
    skin = (w_fg > 0.9) & (keep_out < 0.05) & (act < 12.0) & (L > 6.0)
    if skin.sum() < 2000:
        return f
    q = [float(np.percentile(L[skin], x)) for x in (5, 50, 75, 90, 99.9)]
    for i in range(1, len(q)):
        q[i] = max(q[i], q[i - 1] + 1.0)
    xs = [0.0] + q + [max(255.0, q[-1] + 1.0)]
    ys = [0.0, 52.0, float(lk["p50"]), float(lk["p75"]), float(lk["p90"]), float(lk["max"]), float(lk["max"])]
    gain = np.interp(L, xs, ys) / np.maximum(L, 1.0)
    w = np.clip((w_fg - 0.5) / 0.4, 0, 1) * (1.0 - np.clip(keep_out * 4, 0, 1)) * np.clip((12.0 - act) / 6.0, 0.0, 1.0)
    g = 1.0 + (gain - 1.0) * w
    return np.clip(f * g[:, :, None], 0, 255)


def render_pose(step_idx, seq, kps, p_i, icons, qc=False):
    fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(step_idx)
    CFG["_cur_step"] = step_idx
    pose_frac = p_i / float(max(1, len(seq) - 1))
    if CUR_PLATE.get("step") != step_idx:
        bg.set_step_plate(seq, kps, step_idx)
    if native.on():
        return _render_pose_native(step_idx, seq, kps, p_i, icons, qc, is_w, y_sh, pose_frac)
    rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(seq[p_i], kps[p_i], y_sh, step_idx)
    gm = load_geo(step_idx, p_i) if edit_base(step_idx) else None
    base_rgb, sh_mask, w_fg = shade(rgb_1080, step_idx, is_w, kp_1080, pulse_phase=pose_frac * 0.75,
                                    geo=geo_frame(step_idx, p_i) if edit_base(step_idx) else None, geo_masks=gm)
    if is_edit_base(step_idx):
        save_geo(step_idx, p_i)
    slot_i, deg_s, l1_s, l2_s = layout.callout_for(step_idx, pose_frac)
    fr = layout.draw_laser_and_callout_v4(base_rgb, sh_mask, step_idx, kp_1080, is_w, deg_s, l1_s, l2_s)
    LAST_MASKS["pre_ui"] = fr
    LAST_MASKS["eq_front"] = cv2.GaussianBlur(cv2.dilate(LAST_MASKS["eq_foot"], np.ones((5, 5), np.uint8)), (0, 0), 1.0)
    fr = layout.composite_title_and_preview_v4(fr, layout.fg_for_preview(w_fg), icons, slot_i, is_w, CFG["exercise_name"])
    if qc:
        return fr, base_rgb, sh_mask, w_fg, kp_1080
    return fr


def _render_pose_native(step_idx, seq, kps, p_i, icons, qc, is_w, y_sh, pose_frac):
    """t600 native_grade: the clip's own pixels, graded globally (gymform/native.py). The legacy shade() runs ONLY on
    QC sample frames and only to give the QC its exclusion masks (equipment zone / bench / X-Ray) - never pixels."""
    native.prepare(step_idx, seq, kps, y_sh)
    base_rgb, sh_mask, w_fg, src, kp_1080 = native.grade(step_idx, p_i, is_w)
    if qc:
        shade(src.astype(np.float32), step_idx, is_w, native._ST[step_idx]["raw_kp"][p_i])
        LAST_MASKS["src"] = src
    slot_i, deg_s, l1_s, l2_s = layout.callout_for(step_idx, pose_frac)
    fr = layout.draw_laser_and_callout_v4(base_rgb, sh_mask, step_idx, kp_1080, is_w, deg_s, l1_s, l2_s)
    LAST_MASKS["pre_ui"] = fr
    LAST_MASKS["eq_front"] = np.zeros((H, W), np.float32)
    LAST_MASKS.setdefault("eq_foot", np.zeros((H, W), np.float32))
    fr = layout.composite_title_and_preview_v4(fr, layout.fg_for_preview(w_fg), icons, slot_i, is_w, CFG["exercise_name"])
    if qc:
        return fr, base_rgb, sh_mask, w_fg, kp_1080
    return fr


def load_inputs():
    """Clips + cached keypoints (extracted once per clip slice and stored in the day's kp cache)."""
    kps_seqs = pickle.load(open(CFG["kp_cache"], "rb")) if os.path.exists(CFG["kp_cache"]) else {}
    raw_seqs, dirty = {}, False
    for s in ctx.steps():
        fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(s)
        loader = native.load_slice if CFG.get("no_slice_cap") else d5.load_native_slice_720p
        seq = loader(os.path.join(CFG["assets_dir"], fn), i0, i1)
        mir = bool(CFG["clips"][str(s)].get("mirror"))
        if mir:   # t543u: Day 4 locked camera side for all 4 steps (whole-frame flip of the native clip, no repaint)
            seq = [np.ascontiguousarray(f[:, ::-1]) for f in seq]
        raw_seqs[s] = seq
        # pair steps (same motion, Day 6+) share ONE keypoint track extracted from the motion base clip
        kp_fn = CFG["clips"][str(s)].get("kp_file") or fn
        twin = next((t for t in ctx.steps() if t < s and (CFG["clips"][str(t)].get("kp_file") or "") == kp_fn
                     and kp_fn and (CFG["clips"][str(t)]["i0"], CFG["clips"][str(t)]["i1"]) == (i0, i1)), None)
        sig = (kp_fn, i0, i1, y_sh) + (("mirror",) if mir else ())
        if twin is not None and kps_seqs.get(("sig", twin)) == sig and len(kps_seqs[twin]) == len(seq):
            if kps_seqs.get(("sig", s)) != sig or kps_seqs.get(s) is not kps_seqs[twin]:
                kps_seqs[s], kps_seqs[("sig", s)] = kps_seqs[twin], sig
                dirty = True
            continue
        if kp_fn != fn:   # extract from the motion base clip (not the edited twin)
            if s not in kps_seqs or kps_seqs.get(("sig", s)) != sig:
                kps_seqs[("sig", s)] = sig
                base_seq = loader(os.path.join(CFG["assets_dir"], kp_fn), i0, i1)
                if mir:
                    base_seq = [np.ascontiguousarray(f[:, ::-1]) for f in base_seq]
                kps_seqs[s] = d5.extract_day5_kps_1080(base_seq, s, y_sh)
                dirty = True
            continue
        if ("sig", s) not in kps_seqs and s in kps_seqs and tuple(d5.STEP_SPECS_DAY5.get(s, ())[:3]) == (fn, i0, i1):
            kps_seqs[("sig", s)] = sig      # legacy cache entry built from the same clip slice
            dirty = True
        if s not in kps_seqs or len(kps_seqs[s]) != len(seq) or kps_seqs.get(("sig", s)) != sig:
            kps_seqs[("sig", s)] = sig
            kps_seqs[s] = d5.extract_day5_kps_1080(seq, s, y_sh)
            dirty = True
    if dirty:
        pickle.dump(kps_seqs, open(CFG["kp_cache"], "wb"))
    ctx.normalize_scale(kps_seqs)      # same mannequin size + floor line in all 4 steps (any pose)
    # t572u: native glow level per step (Day 4 fibre shader input calibration, see muscle.day4_fibre)
    ref = float(SPEC.get("muscle_shader", {}).get("day4_native_glow_p50", 100.0))
    CFG["_glow_gain"] = {}
    for s in ctx.steps():
        v = []
        for f in raw_seqs[s][:: max(1, len(raw_seqs[s]) // 6)]:
            ff = f.astype(np.float32)   # BGR
            d = np.maximum(ff[..., 2] - 0.5 * (ff[..., 1] + ff[..., 0]), ff[..., 1] - 0.5 * (ff[..., 2] + ff[..., 0]))
            v.append(d[d > 10])
        v = np.concatenate(v) if v else np.zeros(0)
        p50 = float(np.median(v)) if v.size > 2000 else ref
        CFG["_glow_gain"][str(s)] = round(float(np.clip(ref / max(p50, 1.0), 1.0, 2.5)), 3)
    SEQS.update(raw_seqs)
    KPS.update({s: kps_seqs[s] for s in ctx.steps()})
    return raw_seqs, kps_seqs


def build_icons(raw_seqs, kps_seqs):
    icons = []
    for s in ctx.steps():   # one preview slot per step (t568u: 2-step day = 2 thumbnails)
        fn, i0, i1, is_w, v_mode, y_sh = ctx.step_spec(s)
        CFG["_cur_step"] = s
        bg.set_step_plate(raw_seqs[s], kps_seqs[s], s)
        if native.on():
            native.prepare(s, raw_seqs[s], kps_seqs[s], y_sh)
            sh_rgb, _, _, _, kp_1080 = native.grade(s, len(raw_seqs[s]) - 1, is_w)
            icons.append(b4._make_icon_pair(sh_rgb, kp_1080))
            continue
        rgb_1080, kp_1080 = b4.transform_frame_and_kp_1080p(raw_seqs[s][-1], kps_seqs[s][-1], y_sh, s)
        sh_rgb, sh_m, w_fg = shade(rgb_1080, s, is_w, kp_1080, pulse_phase=0.75,
                                   geo=geo_frame(s, len(raw_seqs[s]) - 1))
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
    if native.on():   # native grade: every pixel is the clip's own (nothing cut or composited) -> alpha 1
        cv2.imwrite(pre + "_alpha.png", m8(np.ones_like(w_fg)))
        cv2.imwrite(pre + "_ref.png", cv2.cvtColor(native.LAST_MASKS["native"]["ref"], cv2.COLOR_RGB2BGR))
    else:
        for x_ in ("_alpha.png", "_ref.png"):
            if os.path.exists(pre + x_):
                os.remove(pre + x_)
    cv2.imwrite(pre + "_shorts.png", m8(shorts_m))
    cv2.imwrite(pre + "_eq.png", m8(LAST_MASKS.get("eq_solid", z)))
    cv2.imwrite(pre + "_eqfoot.png", m8(LAST_MASKS.get("eq_foot", z)))
    cv2.imwrite(pre + "_src.png", cv2.cvtColor(LAST_MASKS["src"], cv2.COLOR_RGB2BGR))
    if LAST_MASKS.get("geo") is not None:   # base frame of a motion-preserving edit (QC geometry reference)
        cv2.imwrite(pre + "_geo.png", cv2.cvtColor(LAST_MASKS["geo"], cv2.COLOR_RGB2BGR))
    elif os.path.exists(pre + "_geo.png"):
        os.remove(pre + "_geo.png")
    plate_p = os.path.join(d, f"step{step_idx}_plate.png")
    if CUR_PLATE.get("saved") != (d, step_idx):
        cv2.imwrite(plate_p, cv2.cvtColor(np.clip(CUR_PLATE["rgb"], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        CUR_PLATE["saved"] = (d, step_idx)
    cv2.imwrite(pre + "_preui.png", cv2.cvtColor(LAST_MASKS["pre_ui"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(pre + "_xwin.png", m8(LAST_MASKS.get("xwin", z)))
    ui_a, ui_rgb = np.zeros((H, W), np.float32), np.zeros((H, W, 3), np.float32)
    for (y0, y1, x0, x1, l_rgb, l_a) in LAST_MASKS.get("ui_layers", ()):   # title, then preview over it
        a_ = l_a[:, :, 0]
        ui_rgb[y0:y1, x0:x1] = ui_rgb[y0:y1, x0:x1] * (1 - a_[:, :, None]) + l_rgb * a_[:, :, None]
        ui_a[y0:y1, x0:x1] = a_ + ui_a[y0:y1, x0:x1] * (1 - a_)
    cv2.imwrite(pre + "_uia.png", m8(ui_a))
    cv2.imwrite(pre + "_uirgb.png", cv2.cvtColor(np.clip(ui_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(pre + "_eqzone.png", m8(LAST_MASKS.get("eq_zone", z)))
    cv2.imwrite(pre + "_static.png", m8(LAST_MASKS.get("static_eq", z)))
    meta_p = os.path.join(d, f"meta_step{step_idx}.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.setdefault("frames", {})[f"step{step_idx}_{tag}"] = {"frame_idx": frame_idx, "step": step_idx, "tag": tag, "kp": kp,
                                                              "callout": LAST_MASKS.get("callout"),
                                                              "callout_axis": LAST_MASKS.get("callout_axis")}
    meta.update({
        "exercise_name": CFG["exercise_name"], "studio_bg": CFG["studio_bg"], "equipment": CFG["equipment"],
        "bone_steps": list(CFG["bone_steps"]), "muscle_steps": list(CFG["muscle_steps"]),
        "clips": CFG.get("clips"), "xray_full_steps": CFG.get("xray_full_steps"),
        "is_wrong": {str(k): bool(v["is_wrong"]) for k, v in CFG.get("clips", {}).items()},
        "source": CFG.get("source", {}), "callouts": CFG.get("callouts"), "tts_lines": CFG.get("tts_lines"), "day": CFG.get("day"), "muscle_shader": muscle.muscle_shader(), "glow_gain": CFG.get("_glow_gain"), "framing_qc": (CFG.get("target_zoom") or {}).get("focus_mode"), "n_steps": ctx.n_steps(), "duration_s": ctx.duration(), "static_body_lum": float(CFG.get("static_body_lum", 60.0)), "config_path": CFG.get("_cfg_path"), "look": CFG.get("look"), "normalized": {k: {str(a): b for a, b in v.items()} if isinstance(v, dict) else v for k, v in CFG.get("_normalized", {}).items()},
        "title_roi": [0, 410, W, 565], "preview_roi": [190, 232, 890, 424], "overlap_alpha": SPEC["ui"]["overlap_alpha"], "fps": FPS, "qc_tags": tempo.QC_TAGS, "target_bones": CFG.get("target_bones", []),
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
        p_i = tempo.pose_index(f_local, n_f, step_phase(step_idx))
        if f_local in tempo.QC_TAGS:
            fr_q, base_q, shm_q, wfg_q, kp_q = render_pose(step_idx, seq, kps, p_i, icons, qc=True)
            save_qc_sidecar(step_idx, tempo.QC_TAGS[f_local], (step_idx - 1) * tempo.FRAMES_PER_STEP + f_local,
                            base_q, shm_q, wfg_q, kp_q)
            cache[p_i] = fr_q
        if p_i not in cache:
            cache[p_i] = render_pose(step_idx, seq, kps, p_i, icons)
        if f_local == tempo.peak_frame(step_phase(step_idx)):
            Image.fromarray(cache[p_i]).save(peak_png)
        ff.stdin.write(np.ascontiguousarray(cache[p_i]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"   Step {step_idx}/{ctx.n_steps()} done ({len(cache)} unique poses)", flush=True)


def preflight_step(step_idx, raw_seqs, kps_seqs, icons):
    """PREFLIGHT: render ONLY the QC sample frames (tempo.QC_TAGS: start/t20/t35/peak/t70/t100/t130) of one
    step through the exact production path (render_pose: plate, zoom transform, shade, callout, title/preview)
    and write the same QC sidecars + the final UI frame (<tag>_final.png) the full-render QC reads from the mp4."""
    cv2.setNumThreads(1)
    seq, kps = raw_seqs[step_idx], kps_seqs[step_idx]
    mp_ = os.path.join(CFG["qc_dir"], f"meta_step{step_idx}.json")
    if os.path.exists(mp_):
        os.remove(mp_)
    for f_local, tag in sorted(tempo.QC_TAGS.items()):
        p_i = tempo.pose_index(f_local, len(seq), step_phase(step_idx))
        fr_q, base_q, shm_q, wfg_q, kp_q = render_pose(step_idx, seq, kps, p_i, icons, qc=True)
        save_qc_sidecar(step_idx, tag, (step_idx - 1) * tempo.FRAMES_PER_STEP + f_local, base_q, shm_q, wfg_q, kp_q)
        cv2.imwrite(os.path.join(CFG["qc_dir"], f"step{step_idx}_{tag}_final.png"), cv2.cvtColor(fr_q, cv2.COLOR_RGB2BGR))
    print(f"   preflight step {step_idx}: {len(tempo.QC_TAGS)} QC frames", flush=True)


def merge_qc_meta():
    d, meta = CFG["qc_dir"], {}
    for s in ctx.steps():
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
    CFG["_cfg_path"] = os.path.abspath(cfg_path)
    return cfg_path


if __name__ == "__main__":
    a = sys.argv
    _load_cfg(a)
    if "--sheet" in a:          # stage: contact sheet + cover from the cached step peaks
        i = a.index("--sheet")
        pk = {s: Image.open(a[i + 1 + s]).convert("RGB") for s in ctx.steps()}
        pk[ctx.steps()[-1]].save(CFG["out_cover"], quality=95)
        layout.make_sheet(pk, a[i + 1])
    elif "--icons-out" in a:      # stage: preview icons (all 4 steps), pickled for the step workers
        raw, kps = load_inputs()
        pickle.dump(build_icons(raw, kps), open(a[a.index("--icons-out") + 1], "wb"))
    elif "--preflight-step" in a:   # preflight: QC sample frames only, real composite path
        CFG["qc_dir"] = a[a.index("--qc-dir") + 1]
        os.makedirs(CFG["qc_dir"], exist_ok=True)
        s = int(a[a.index("--preflight-step") + 1])
        raw, kps = load_inputs()
        preflight_step(s, raw, kps, pickle.load(open(a[a.index("--icons") + 1], "rb")))
    elif "--render-step" in a:  # stage: composite + overlays + encode for one step
        # HARD GATE (RULES.md): refuse without a passing preflight stamp for this exact config/code hash
        from gymform import preflight as _pf
        _pf.require(a[a.index("--config") + 1], a[a.index("--overrides") + 1] if "--overrides" in a else None)
        s = int(a[a.index("--render-step") + 1])
        raw, kps = load_inputs()
        ic = pickle.load(open(a[a.index("--icons") + 1], "rb"))
        render_step_to_mp4(s, raw, kps, ic, a[a.index("--out") + 1], a[a.index("--peak") + 1])
