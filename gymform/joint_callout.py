"""t678u joint-attached angle callout (opt-in, still-motion format): replaces the floating top-left angle block.

A small readout (measured number + one short label under it) placed near the body, connected to the joint it measures
by ONE thin neon leader line (green right / red wrong, soft glow) ending in a small dot at the joint (t679u/t680u: no
angle arc, nothing else). Placement: candidate boxes around the joint, scored so the text sits on clean
backdrop - never over the target-muscle regions or the machine (anything that is not backdrop), never in the title band.
Returns an RGBA layer in frame coordinates, so the motion stage warps it with the same crossfade / push-in transform
as the still underneath (the callout moves with the body)."""
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

GREEN = (42, 245, 135)
RED = (255, 96, 36)


def _font(size, weight):
    import build_v28_luxury_ab as v28
    return v28.f_pretendard(size, weight)


def visible_side(kp, shape):
    H, W = shape
    tq = np.zeros((H, W), np.uint8)
    cv2.fillPoly(tq, [np.array([kp[k] for k in ("sh_r", "sh_l", "hip_l", "hip_r")], np.int32)], 1)
    best, sd_b = 2.0, "r"
    for sd in ("r", "l"):
        m = np.zeros((H, W), np.uint8)
        cv2.line(m, tuple(int(v) for v in kp[f"elb_{sd}"]), tuple(int(v) for v in kp[f"wr_{sd}"]), 1, 3)
        f = (m & tq).sum() / max(m.sum(), 1)
        if f < best:
            best, sd_b = f, sd
    return sd_b


def elbow_flexion(kp, sd):
    S, E, Wr = (np.array(kp[f"{j}_{sd}"], np.float64) for j in ("sh", "elb", "wr"))
    d0 = (E - S) / max(np.linalg.norm(E - S), 1e-6)       # upper-arm extension
    d1 = (Wr - E) / max(np.linalg.norm(Wr - E), 1e-6)     # forearm
    a0, a1 = math.degrees(math.atan2(d0[1], d0[0])), math.degrees(math.atan2(d1[1], d1[0]))
    diff = (a1 - a0 + 540.0) % 360.0 - 180.0
    return abs(diff), E, a0, diff, float(np.linalg.norm(E - S))


def draw(frame_rgb, kp, protect=None, is_wrong=False, label="", ui_top=640, num_px=50, lab_px=16):
    """-> (rgba float32 HxWx4 0..1, info). protect = float mask of target-muscle regions (text never on it)."""
    H, W = frame_rgb.shape[:2]
    sd = visible_side(kp, (H, W))
    deg, E, a0, diff, ua = elbow_flexion(kp, sd)
    col = RED if is_wrong else GREEN
    cx, cy = int(round(E[0])), int(round(E[1]))
    layer = np.zeros((H, W, 4), np.float32)
    # --- text block placement on clean backdrop ---
    f_num, f_lab = _font(num_px, "extrabold"), _font(lab_px, "extrabold")
    num = f"{int(round(deg))}\u00b0"
    nb = f_num.getbbox(num); lb = f_lab.getbbox(label) if label else (0, 0, 0, 0)
    bw = max(nb[2] - nb[0], lb[2] - lb[0]) + 4
    bh = (nb[3] - nb[1]) + (lb[3] - lb[1] + 8 if label else 0) + 4
    L = frame_rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    bgL = float(np.median(L[ui_top:ui_top + 200, W // 2 - 150: W // 2 + 150]))
    obst = (np.abs(cv2.GaussianBlur(L, (0, 0), 1.5) - bgL) > 9.0).astype(np.float32)
    if protect is not None:
        obst = np.maximum(obst, 10.0 * (cv2.dilate((protect > 0.02).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0))
    arcm = np.zeros((H, W), np.float32); cv2.circle(arcm, (cx, cy), 34, 1.0, -1)
    obst = np.maximum(obst, 5.0 * arcm)
    prot = None
    if protect is not None:
        prot = (cv2.dilate((protect > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)

    def anchor(bx_, by_):   # leader end: the box edge point nearest the joint (left/right middle of the number)
        ax_ = bx_ - 8 if cx < bx_ else (bx_ + bw + 8 if cx > bx_ + bw else bx_ + bw // 2)
        ay_ = int(np.clip(cy, by_ + 6, by_ + bh - 6)) if ax_ != bx_ + bw // 2 else (by_ - 8 if cy < by_ else by_ + bh + 8)
        return ax_, ay_
    ii = cv2.integral(obst)
    box_sum = lambda x0_, y0_, x1_, y1_: ii[y1_, x1_] - ii[y0_, x1_] - ii[y1_, x0_] + ii[y0_, x0_]
    best = None
    m_ = 10
    for dist in range(80, 320, 10):
        for k in range(24):
            th = 2 * math.pi * k / 24
            bx = int(cx + dist * math.cos(th) - bw / 2); by = int(cy + dist * math.sin(th) - bh / 2)
            if bx - m_ < 8 or by - m_ < ui_top or bx + bw + m_ > W - 8 or by + bh + m_ > H - 120:
                continue
            o = box_sum(bx - m_, by - m_, bx + bw + m_, by + bh + m_) / ((bw + 2 * m_) * (bh + 2 * m_))
            ax_, ay_ = anchor(bx, by)
            n = 24
            pts = [(int(cx + (ax_ - cx) * i / n), int(cy + (ay_ - cy) * i / n)) for i in range(3, n + 1)]
            on_m = np.mean([prot[min(H - 1, max(0, y)), min(W - 1, max(0, x))] for x, y in pts]) if prot is not None else 0.0
            cost = 100.0 * o + 40.0 * on_m + dist / 60.0
            if best is None or cost < best[0]:
                best = (cost, bx, by, dist, o, on_m)
    _, bx, by, dist, ovl, on_m = best
    # --- neon leader + joint dot (supersampled thin core + soft glow) ---
    ax, ay = anchor(bx, by)
    ss = 4
    x0, y0 = max(0, min(cx, ax) - 30), max(0, min(cy, ay) - 30)
    x1, y1 = min(W, max(cx, ax) + 30), min(H, max(cy, ay) + 30)
    tile = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    P = lambda x, y: (int((x - x0) * ss), int((y - y0) * ss))
    cv2.line(tile, P(cx, cy), P(ax, ay), 255, int(2.0 * ss), cv2.LINE_AA)
    cv2.circle(tile, P(cx, cy), int(5.0 * ss), 255, -1, cv2.LINE_AA)
    core = cv2.resize(tile, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    glow = cv2.GaussianBlur(core, (0, 0), 4.0) * 0.85
    neon = np.clip(np.maximum(core, glow), 0, 1)
    c_rgb = np.array(col, np.float32) / 255.0
    hot = np.clip(core * 0.55, 0, 1)[..., None]            # white-hot core, coloured glow = neon
    layer[y0:y1, x0:x1, :3] = c_rgb * (1 - hot) + hot
    layer[y0:y1, x0:x1, 3] = neon
    pil = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(pil)
    tx, ty = bx - nb[0] + 2, by - nb[1] + 2
    d.text((tx + 1, ty + 2), num, font=f_num, fill=(0, 0, 0, 150))
    d.text((tx, ty), num, font=f_num, fill=(255, 255, 255, 255))
    if label:
        ly = by + (nb[3] - nb[1]) + 8 - lb[1] + 2
        d.text((bx - lb[0] + 3, ly + 1), label, font=f_lab, fill=(0, 0, 0, 140))
        d.text((bx - lb[0] + 2, ly), label, font=f_lab, fill=(*col, 255))
    t = np.asarray(pil, np.float32) / 255.0
    ta = t[..., 3:4]
    rgb = layer[..., :3] * layer[..., 3:4] * (1 - ta) + t[..., :3] * ta
    a = layer[..., 3:4] * (1 - ta) + ta
    out = np.concatenate([np.where(a > 1e-4, rgb / np.maximum(a, 1e-4), 0), a], -1)
    return out, {"deg": num, "side": sd, "joint": [cx, cy], "leader_to": [ax, ay], "text_box": [bx, by, bw, bh],
                 "text_dist_px": dist, "text_box_obstacle_frac": round(float(ovl), 4),
                 "leader_on_target_muscle_frac": round(float(on_m), 3)}


def composite(frame_rgb, layer):
    a = layer[..., 3:4]
    return np.clip(frame_rgb.astype(np.float32) * (1 - a) + layer[..., :3] * 255.0 * a, 0, 255).astype(np.uint8)
