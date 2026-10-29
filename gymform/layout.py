"""LOCKED layout: crisp laser, callouts (text from day config), depth-behind title, translucent 4-slot preview, sheet."""
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


# d5.draw_day5_laser_and_callout looks the laser function up on the p62 module at call time;
# patch it in-process only (files on disk are untouched).
p62.draw_demic_laser_joint_axis = draw_crisp_laser_joint_axis


def composite_title_and_preview_v4(base_rgb, fg_mask, torso_icons, active_slot, is_wrong, exercise_name):
    # Title: identical to b4 (depth-behind with fg_front = 0.92 * fg)
    out = b4.composite_depth_behind_title_and_translucent_preview(
        base_rgb, fg_mask, torso_icons, active_slot=active_slot, is_wrong=is_wrong, exercise_name=exercise_name
    )
    # Preview ROI: wherever the mannequin/dumbbells touch the pill, show them sharp (pre-preview pixels with title).
    ty0, ty1 = 410, 565
    titled = base_rgb.copy()
    t_rgb, t_a = b4._TITLE_ROI_CACHE[exercise_name]
    eff_a = t_a * (1.0 - np.clip(fg_mask[ty0:ty1, :, None] * 0.92, 0.0, 0.92))
    titled[ty0:ty1] = np.clip(titled[ty0:ty1].astype(np.float32) * (1.0 - eff_a) + t_rgb * eff_a, 0, 255).astype(np.uint8)
    py0, py1 = 230, 425
    touch = np.clip(fg_mask[py0:py1, :, None] * 1.15, 0.0, 0.92)
    eqm = LAST_MASKS.get("eq_front")
    if eqm is not None:  # equipment renders fully in front of the preview pill (no preview pixels on it)
        touch = np.maximum(touch, eqm[py0:py1, :, None])
    out[py0:py1] = np.clip(out[py0:py1].astype(np.float32) * (1.0 - touch) + titled[py0:py1].astype(np.float32) * touch, 0, 255).astype(np.uint8)
    return out


def fg_for_preview(w_fg: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(np.clip((w_fg - 0.05) / 0.6, 0.0, 1.0).astype(np.float32), (0, 0), 1.2)


def callout_for(step_idx, pose_frac):
    """Callout text from the day config: deg [a, b] -> round(a + b*frac) deg (or fixed string);
    l2 [label, a, b] -> 'round(a + b*frac)% label'."""
    c = CFG["callouts"][str(step_idx)]
    deg = c["deg"] if isinstance(c["deg"], str) else f"{int(round(c['deg'][0] + c['deg'][1] * pose_frac))}\u00b0"
    l2 = c["l2"] if isinstance(c["l2"], str) else f"{int(round(c['l2'][1] + c['l2'][2] * pose_frac))}% {c['l2'][0]}"
    return int(c["slot"]), deg, c["l1"], l2


def draw_laser_and_callout_v4(base_rgb, shorts_mask_1080, step_idx, kp, is_wrong, deg_str, l1_str, l2_str):
    """d5.draw_day5_laser_and_callout with an optional fixed text block per step (CFG['callout_ty'])."""
    ov = CFG.get("callout_ty", {}).get(str(step_idx))
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
