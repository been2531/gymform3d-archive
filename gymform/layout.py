"""LOCKED layout: crisp laser, callouts (text from day config), see-through title + translucent 4-slot preview
(visible at ui.overlap_alpha over mannequin/equipment, Day 4 rule), sheet."""
import math
import cv2
import numpy as np
from gymform.ctx import *  # shared state: W,H,CFG,LAST_MASKS,CUR_PLATE,STEP_SCALE,_XX,_YY,d5,b4,p62,v30
from PIL import Image, ImageDraw
from gymform import ctx


def draw_crisp_laser_joint_axis(canvas, main_chains_1x, aux_ext_chains_1x, vertex_pts_1x, is_wrong):
    all_pts = [pt for ch in (main_chains_1x + aux_ext_chains_1x) for pt in ch] + list(vertex_pts_1x)
    if not all_pts:
        return
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    pad = 44
    rx0 = max(0, int(min(xs)) - pad); ry0 = max(0, int(min(ys)) - pad)
    rx1 = min(W, int(max(xs)) + pad); ry1 = min(H, int(max(ys)) + pad)
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw <= 4 or rh <= 4:
        return
    sh = lambda chains: [[(px - rx0, py - ry0) for (px, py) in ch] for ch in chains]  # noqa: E731
    m_loc, a_loc = sh(main_chains_1x), sh(aux_ext_chains_1x)
    v_loc = [(vx - rx0, vy - ry0) for (vx, vy) in vertex_pts_1x]
    col_halo = (255, 70, 72) if is_wrong else (52, 255, 140)
    col_core = (255, 236, 234) if is_wrong else (226, 255, 238)
    taper = p62._draw_tapered_laser_polyline_ss

    glow = Image.new("RGBA", (rw, rh), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    for ch in m_loc:
        taper(gd, ch, base_w_1x=9.0, col_rgba=(*col_halo, 45), ss=1, fade_start=0.15, fade_end=0.15)
    glow_arr = cv2.GaussianBlur(np.array(glow), (0, 0), 10.0)

    halo = Image.new("RGBA", (rw, rh), (0, 0, 0, 0)); hd = ImageDraw.Draw(halo)
    for ch in m_loc:
        taper(hd, ch, base_w_1x=6.0, col_rgba=(*col_halo, 150), ss=1, fade_start=0.15, fade_end=0.15)
    for ch in a_loc:
        taper(hd, ch, base_w_1x=4.0, col_rgba=(*col_halo, 110), ss=1, fade_start=0.20, fade_end=0.22)
    halo_arr = cv2.GaussianBlur(np.array(halo), (0, 0), 3.5)

    SS = 3
    core = Image.new("RGBA", (rw * SS, rh * SS), (0, 0, 0, 0)); cd = ImageDraw.Draw(core)
    for ch in a_loc:
        taper(cd, ch, base_w_1x=2.2, col_rgba=(*col_halo, 210), ss=SS, fade_start=0.20, fade_end=0.22)
        taper(cd, ch, base_w_1x=1.1, col_rgba=(*col_core, 230), ss=SS, fade_start=0.20, fade_end=0.22)
    for ch in m_loc:
        taper(cd, ch, base_w_1x=2.8, col_rgba=(*col_halo, 255), ss=SS, fade_start=0.15, fade_end=0.15)
        taper(cd, ch, base_w_1x=1.4, col_rgba=(*col_core, 255), ss=SS, fade_start=0.15, fade_end=0.15)
    for vx, vy in v_loc:  # small crisp vertex rings
        r_o = 6.5 * SS
        cd.ellipse([vx * SS - r_o, vy * SS - r_o, vx * SS + r_o, vy * SS + r_o], outline=(*col_halo, 255), width=int(1.8 * SS))
        r_i = 2.4 * SS
        cd.ellipse([vx * SS - r_i, vy * SS - r_i, vx * SS + r_i, vy * SS + r_i], fill=(*col_core, 255))
    core_arr = cv2.resize(np.array(core), (rw, rh), interpolation=cv2.INTER_AREA)

    roi = canvas.crop((rx0, ry0, rx1, ry1))
    for arr in (glow_arr, halo_arr, core_arr):
        roi.alpha_composite(Image.fromarray(arr))
    canvas.paste(roi, (rx0, ry0))


_DEMIC_LASER_DAY4 = p62.draw_demic_laser_joint_axis   # the original volumetric Day 4 laser (kept for callout_style day4)
# d5.draw_day5_laser_and_callout looks the laser function up on the p62 module at call time;
# patch it in-process only (files on disk are untouched).
p62.draw_demic_laser_joint_axis = draw_crisp_laser_joint_axis


_TITLE_A = {}


def _title_layer(exercise_name):
    """Title RGB + full alpha in the title ROI (cached per exercise name)."""
    if exercise_name not in _TITLE_A:
        ty0, ty1 = TITLE_ROI
        blank = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        b4.draw_single_line_kaushan_title(blank, exercise_name)
        t = np.array(blank.crop((0, ty0, W, ty1)), dtype=np.float32)
        _TITLE_A[exercise_name] = (t[:, :, :3], t[:, :, 3:4] / 255.0)
    return _TITLE_A[exercise_name]


TITLE_ROI = (410, 565)
PREV_ROI = (190, 232, 890, 424)   # x0, y0, x1, y1 (= b4 top-bar ROI)


def overlap_alpha():
    """Locked see-through alphas (gymform_spec.json ui.overlap_alpha): opacity of the title / preview where
    the mannequin or equipment overlaps them (Day 4 measured ~0.30 median title over body, preview >=0.35)."""
    oa = SPEC["ui"].get("overlap_alpha", {})
    return float(oa.get("title", 0.40)), float(oa.get("preview", 0.40))


def composite_title_and_preview_v4(base_rgb, fg_mask, torso_icons, active_slot, is_wrong, exercise_name):
    """Day 4 see-through rule: title + 4-slot preview stay VISIBLE (translucent, alpha A) over the mannequin AND
    the equipment; everywhere else they are drawn exactly as before. Under the overlap the 3D layer stays sharp
    (no frost) and keeps (1-A) of its own pixels, so equipment is never cut, hidden or distorted."""
    a_t, a_p = overlap_alpha()
    occ = fg_mask
    eqm = LAST_MASKS.get("eq_front")
    if eqm is not None:
        occ = np.maximum(occ, eqm)
    occ = np.clip(occ, 0.0, 1.0)
    out = base_rgb.copy()
    # 1. title (back wall): full alpha on the wall, a_t over mannequin/equipment
    ty0, ty1 = TITLE_ROI
    t_rgb, t_a = _title_layer(exercise_name)
    eff_t = t_a * (1.0 - occ[ty0:ty1, :, None] * (1.0 - a_t))
    out[ty0:ty1] = np.clip(out[ty0:ty1].astype(np.float32) * (1.0 - eff_t) + t_rgb * eff_t, 0, 255).astype(np.uint8)
    # 2. preview pill: frosted glass on the wall; over occluders unfrosted (sharp 3D) at alpha a_p
    rx0, ry0, rx1, ry1 = PREV_ROI
    under = out[ry0:ry1, rx0:rx1].astype(np.float32)
    frosted_full = b4.draw_compact_demic_top_preview(out, torso_icons, active_slot=active_slot, is_wrong=is_wrong)
    _, ov_rgb, ov_a = b4._TOP_BAR_CACHE[(id(torso_icons), active_slot, is_wrong)]
    frosted = frosted_full[ry0:ry1, rx0:rx1].astype(np.float32)
    sharp = under * (1.0 - ov_a) + ov_rgb * ov_a
    w = occ[ry0:ry1, rx0:rx1, None]
    prev = frosted * (1.0 - w) + sharp * w
    k = w * (1.0 - a_p)                       # share of the untouched 3D layer kept
    frosted_full[ry0:ry1, rx0:rx1] = np.clip(prev * (1.0 - k) + under * k, 0, 255).astype(np.uint8)
    # UI layers for the QC sidecar (full alpha, colour) - QC measures the effective alpha over occluders
    LAST_MASKS["ui_layers"] = ((ty0, ty1, 0, W, t_rgb, t_a), (ry0, ry1, rx0, rx1, ov_rgb, ov_a))
    return frosted_full


def fg_for_preview(w_fg: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(np.clip((w_fg - 0.05) / 0.6, 0.0, 1.0).astype(np.float32), (0, 0), 1.2)


def callout_for(step_idx, pose_frac):
    """Callout text from the day config.
    deg: "N\u00b0" (fixed coaching angle) or {"from": a, "to": b} (animated from a at the start pose to b at the peak).
    The number is the joint angle the label names, as coached for the shown pose (2D keypoint angles in a 3/4 view
    are projection artefacts, never displayed). A bare list [a, b] is rejected: it was read as start + b*frac while
    configs wrote it as a range (Day 6: [45, 78] rendered 121\u00b0, [30, 45] rendered 74\u00b0 under 'ELBOWS 45\u00b0').
    l2 [label, a, b] -> 'round(a + b*frac)% label'."""
    c = CFG["callouts"][str(step_idx)]
    deg = c["deg"]
    if isinstance(deg, dict):
        deg = f"{int(round(deg['from'] + (deg['to'] - deg['from']) * pose_frac))}\u00b0"
    elif not isinstance(deg, str):
        raise ValueError(f"callouts.{step_idx}.deg must be 'N\u00b0' or {{'from','to'}}, got {deg!r}")
    l2 = c["l2"] if isinstance(c["l2"], str) else f"{int(round(c['l2'][1] + c['l2'][2] * pose_frac))}% {c['l2'][0]}"
    LAST_MASKS["callout"] = {"deg": deg, "l1": c["l1"], "l2": l2}
    return int(c["slot"]), deg, c["l1"], l2


_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                   "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def spoken_degrees(text):
    """Angles spoken in a TTS line ('forty-five degrees', '45 degrees', 'ninety degree')."""
    import re
    out = []
    for m in re.finditer(r"([a-z0-9\- ]+?)\s+degrees?\b", text.lower()):
        words = m.group(1).replace("-", " ").split()[-2:]
        if words and words[-1].isdigit():
            out.append(int(words[-1])); continue
        v = 0
        for w in words:
            if w in _TENS: v = _TENS[w] if v == 0 else v
            elif w in _NUM: v += _NUM[w]
        if v or "zero" in words:
            out.append(v)
    return out


def callout_lint(cfg):
    """Displayed numbers, labels and TTS must agree. Returns a list of errors (empty = OK)."""
    import re
    errs = []
    tts = [t for _, t in cfg.get("tts_lines", [])]
    for s in ("1", "2", "3", "4"):
        c = cfg["callouts"][s]
        d = c["deg"]
        if isinstance(d, dict):
            lo, hi = sorted((float(d["from"]), float(d["to"])))
            peak = float(d["to"])
        elif isinstance(d, str) and re.fullmatch(r"\d+\u00b0", d):
            lo = hi = peak = float(d[:-1])
        else:
            errs.append(f"step{s}: deg {d!r} is ambiguous (use 'N\u00b0' or {{'from','to'}})"); continue
        lab = [int(x) for x in re.findall(r"(\d+)\u00b0", c["l1"])]
        if lab and (lo != hi or lab[0] != peak):
            errs.append(f"step{s}: label '{c['l1']}' says {lab[0]}\u00b0 but the number shows {lo:g}-{hi:g}\u00b0")
        if int(s) - 1 < len(tts):
            for v in spoken_degrees(tts[int(s) - 1]):
                if not (lo <= v <= hi):
                    errs.append(f"step{s}: TTS says {v} degrees but the number shows {lo:g}-{hi:g}\u00b0")
        if not (0 <= lo <= 180 and 0 <= hi <= 180):
            errs.append(f"step{s}: angle {lo:g}-{hi:g}\u00b0 outside 0-180")
    return errs


def draw_laser_and_callout_v4(base_rgb, shorts_mask_1080, step_idx, kp, is_wrong, deg_str, l1_str, l2_str):
    """d5.draw_day5_laser_and_callout with an optional fixed text block per step (CFG['callout_ty'])."""
    if SPEC["ui"].get("callout_style", "day5") in ("day4", "day4_refined") and int(CFG.get("day", 0)) >= 6:
        return draw_callout_day4(base_rgb, shorts_mask_1080, step_idx, kp, is_wrong, deg_str, l1_str, l2_str)
    ov = CFG.get("callout_ty", {}).get(str(step_idx))
    step_idx = ctx.legacy(step_idx)   # laser chains / callout target follow the step's ROLE, not its position
    if ov is None:
        return d5.draw_day5_laser_and_callout(base_rgb, shorts_mask_1080, step_idx, kp, is_wrong, deg_str, l1_str, l2_str)
    # laser only (empty callout strings), then our own text block + leader
    canvas = Image.fromarray(base_rgb).convert("RGBA")
    sh_r, sh_l, elb_r, wr_r = kp["sh_r"], kp["sh_l"], kp["elb_r"], kp["wr_r"]
    elb_l, wr_l = kp["elb_l"], kp["wr_l"]
    _, wr_r_ext = p62._extend_segment(elb_r, wr_r, 0.0, 1.12)
    _, wr_l_ext = p62._extend_segment(elb_l, wr_l, 0.0, 1.12)
    sh_r_ext, sh_l_ext = p62._extend_segment(sh_r, sh_l, -0.08, 1.08)
    clav_mid = (0.52 * sh_r[0] + 0.48 * sh_l[0], 0.52 * sh_r[1] + 0.48 * sh_l[1] + 4.0)
    if step_idx == 2:   # identical laser chains / targets to d5.draw_day5_laser_and_callout
        p62.draw_demic_laser_joint_axis(canvas, main_chains_1x=[[sh_r, elb_r, wr_r_ext], [sh_l, elb_l, wr_l_ext]],
                                        aux_ext_chains_1x=[[sh_r_ext, sh_l_ext]], vertex_pts_1x=[sh_r, elb_r, sh_l, elb_l],
                                        is_wrong=is_wrong)
        target_pt = (int(0.65 * sh_r[0] + 0.35 * sh_l[0]), int(sh_r[1] - 22))
    else:
        p62.draw_demic_laser_joint_axis(canvas, main_chains_1x=[[sh_r, elb_r, wr_r_ext]], aux_ext_chains_1x=[[sh_r_ext, clav_mid]],
                                        vertex_pts_1x=[sh_r, elb_r], is_wrong=is_wrong)
        if step_idx in (1, 3):
            target_pt = (int(sh_r[0] + 8), int(sh_r[1] + 6))
        else:
            target_pt = (int(0.76 * sh_r[0] + 0.24 * elb_r[0]), int(0.76 * sh_r[1] + 0.24 * elb_r[1] + 10))
    laser_rgb = np.array(canvas.convert("RGB")).astype(np.float32)
    g = np.clip(shorts_mask_1080 * 0.90, 0.0, 0.90)[:, :, None]
    blended = np.clip(laser_rgb * (1.0 - g) + base_rgb.astype(np.float32) * g, 0, 255).astype(np.uint8)
    base_pil = Image.fromarray(blended).convert("RGBA")
    ovl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ovl)
    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    tx, ty = 54, int(ov)
    elbow_y = ty + 76
    elbow_x = max(254, min(target_pt[0] - 40, 300))
    d5.bcd._draw_precision_leader(rd, target_pt, (elbow_x, elbow_y), (242, elbow_y), col)
    rd.text((tx + 2, ty + 3), deg_str, font=v30.f_pretendard(84, "extrabold"), fill=(0, 0, 0, 210))
    rd.text((tx, ty), deg_str, font=v30.f_pretendard(84, "extrabold"), fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + 90), l1_str, font=v30.f_pretendard(21, "extrabold"), fill=(*col, 255))
    rd.text((tx + 2, ty + 117), l2_str, font=v30.f_pretendard(16, "bold"), fill=(215, 222, 232, 235))
    return np.array(Image.alpha_composite(base_pil, ovl).convert("RGB"))


def draw_callout_day4(base_rgb, shorts_mask_1080, step_idx, kp, is_wrong, deg_str, l1_str, l2_str, variant=None):
    """t547u: Day 4 callout look and feel (build_day4_rdl_master.draw_day4_laser_and_callout), generalised:
    ONE straight joint axis through the coached joint (torso line -> joint vertex -> limb, bent only at the vertex)
    + two short cross ticks, vertex ring(s) (wrong: the joint only; right: joint + next joint), a horizontal
    precision leader from a target point on the segment to a left text block (number 86 px at x=86, label 21,
    sub-line 16, sitting level with the target).
    variant "day4" = Day 4 volumetric laser + 86 px number; "day4_refined" = same geometry/placement with the
    Day 5 crisp 2.8 px laser (no bloom) and the Day 5 84 px number."""
    if variant is None:   # t554u: muscle-mistake days use the crisper callout (Day 5 2.8 px line, 84 px number)
        variant = SPEC["ui"].get("callout_style_muscle_days", "day4_refined") if not CFG.get("bone_steps") else SPEC["ui"].get("callout_style", "day4")
    ax = SPEC["ui"].get("callout_axis", {}).get(str(CFG.get("day")), {})
    j = ax.get("joint", "shoulder")
    if j == "shoulder":   # angle between torso and upper arm (elbow flare / shoulder abduction)
        hip_mid = (0.5 * (kp["hip_r"][0] + kp["hip_l"][0]), 0.5 * (kp["hip_r"][1] + kp["hip_l"][1]))
        a, v, b = p62._extend_segment(kp["sh_r"], hip_mid, 0.0, 0.62)[1], kp["sh_r"], kp["elb_r"]
        nxt = kp["elb_r"]
        cross1 = p62._extend_segment(kp["sh_r"], kp["sh_l"], -0.10, 0.30)
    else:                 # hip hinge (Day 4 RDL): shoulder -> hip -> knee
        a, v, b = kp["sh"], kp["hip_r"], kp["kn_r"]
        nxt = kp["kn_r"]
        cross1 = p62._extend_segment(kp["sh"], kp["hip_r"], 0.66, 1.36)
    _, b_ext = p62._extend_segment(v, b, 0.0, 1.14)
    a_top, _ = p62._extend_segment(a, v, -0.12, 1.0)
    cross2 = p62._extend_segment(b, v, 0.66, 1.32)
    canvas = Image.fromarray(base_rgb).convert("RGBA")
    laser = _DEMIC_LASER_DAY4 if variant == "day4" else draw_crisp_laser_joint_axis
    laser(canvas, main_chains_1x=[[a_top, v, b_ext]], aux_ext_chains_1x=[list(cross1), list(cross2)],
          vertex_pts_1x=[v] if is_wrong else [v, nxt], is_wrong=is_wrong)
    target_pt = (int(0.48 * v[0] + 0.52 * b[0]), int(0.48 * v[1] + 0.52 * b[1]))
    laser_rgb = np.array(canvas.convert("RGB")).astype(np.float32)
    g = np.clip(shorts_mask_1080 * 0.85, 0.0, 0.85)[:, :, None]
    blended = np.clip(laser_rgb * (1.0 - g) + base_rgb.astype(np.float32) * g, 0, 255).astype(np.uint8)
    base_pil = Image.fromarray(blended).convert("RGBA")
    ovl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ovl)
    col = (255, 96, 36) if is_wrong else (32, 245, 130)
    big = 86 if variant == "day4" else 84
    tx = 86
    ty_min = int(SPEC["ui"].get("callout_ty_min", 600))   # below the title band
    ty = int(np.clip(target_pt[1] - 24 - 76, ty_min, 1460))
    elbow_y = ty + 76
    elbow_x = max(286, target_pt[0] - 42)
    d5.bcd._draw_precision_leader(rd, target_pt, (elbow_x, elbow_y), (278, elbow_y), col)
    f_big = v30.f_pretendard(big, "extrabold")
    rd.text((tx + 2, ty + 3), deg_str, font=f_big, fill=(0, 0, 0, 210))
    rd.text((tx, ty), deg_str, font=f_big, fill=(255, 255, 255, 255))
    rd.text((tx + 2, ty + big + 6), l1_str, font=v30.f_pretendard(21, "extrabold"), fill=(*col, 255))
    rd.text((tx + 2, ty + big + 33), l2_str, font=v30.f_pretendard(16, "bold"), fill=(215, 222, 232, 235))
    return np.array(Image.alpha_composite(base_pil, ovl).convert("RGB"))


def make_sheet(peaks: dict, out_path: str):
    pw, ph, pad, header_h = 520, 924, 22, 96
    sheet = Image.new("RGB", (pad + 4 * (pw + pad), header_h + ph + pad), (14, 16, 20))
    sd = ImageDraw.Draw(sheet)
    sd.text((28, 18), CFG.get("sheet_title", f"Day {CFG['day']} {CFG['exercise_name']}"), font=v30.f_pretendard(26, "extrabold"), fill=(242, 246, 252))
    sd.text((28, 54), " | ".join(CFG.get("sheet_labels", [])), font=v30.f_pretendard(18, "bold"), fill=(185, 225, 255))
    for i, s_i in enumerate([1, 2, 3, 4]):
        x0, y0 = pad + i * (pw + pad), header_h
        sheet.paste(peaks[s_i].resize((pw, ph), Image.Resampling.LANCZOS), (x0, y0))
        col = (245, 65, 65) if ctx.step_spec(s_i)[3] else (35, 235, 130)
        sd.rectangle([x0 - 2, y0 - 2, x0 + pw + 1, y0 + ph + 1], outline=col, width=3)
    sheet.save(out_path, quality=94)
    return out_path
