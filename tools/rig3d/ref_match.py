"""Reference-silhouette match for a one-arm bench row.

Downloads a linked, freely licensed reference photo into _scratch (never
committed), splits the hang/peak panels, runs MediaPipe Pose + mask, then
fits 3D landmarks so their reprojection through a fixed camera matches the
2D targets. Also writes an overlay QC (silhouette IoU + joint error).

Reference (do not commit the image):
  https://commons.wikimedia.org/wiki/File:DumbbellBentOverRow.JPG
  Photo: GeorgeStepanek / Wikimedia Commons / CC BY-SA 3.0
  Model credited as Emily on the file page.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request

import numpy as np

REF_DEFAULT = {
    "page": "https://commons.wikimedia.org/wiki/File:DumbbellBentOverRow.JPG",
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/6/67/DumbbellBentOverRow.JPG",
    "license": "CC BY-SA 3.0",
    "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
    "author": "GeorgeStepanek",
    "credit": "GeorgeStepanek / Wikimedia Commons / CC BY-SA 3.0 (model: Emily)",
}

# MediaPipe / BlazePose indices we actually drive
TRACK = {
    "nose": 0, "lear": 7, "rear": 8,
    "lsh": 11, "rsh": 12, "lel": 13, "rel": 14, "lwr": 15, "rwr": 16,
    "lhip": 23, "rhip": 24, "lkn": 25, "rkn": 26, "lank": 27, "rank": 28,
    "lheel": 29, "rheel": 30, "ltoe": 31, "rtoe": 32,
}
# Heavier weights on the row silhouette: torso, working arm, standing leg.
# Support shin (25/27/29/31) is planted on the pad and often occluded in a
# right-side photo, so it is not a 2D target (MediaPipe guesses it poorly).
W_JOINT = {
    0: 0.8, 7: 0.3, 8: 0.3,
    11: 1.2, 12: 1.5, 14: 1.8, 16: 2.0,
    23: 0.9, 24: 1.1, 26: 1.2, 28: 1.3,
    30: 0.5, 32: 0.6,
}
MOVE = set(W_JOINT)


def _n(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def scratch_dir(root):
    d = os.path.join(root, "_scratch", "refs", "db_row")
    os.makedirs(d, exist_ok=True)
    return d


def download_reference(root, spec=None):
    spec = dict(REF_DEFAULT if spec is None else {**REF_DEFAULT, **spec})
    out_dir = scratch_dir(root)
    dest = os.path.join(out_dir, "source.jpg")
    if not os.path.isfile(dest) or os.path.getsize(dest) < 1000:
        url = spec["image_url"]
        req = urllib.request.Request(url, headers={"User-Agent": "gymform3d-ref-match/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
    spec["local"] = dest
    json.dump(spec, open(os.path.join(out_dir, "source_meta.json"), "w"), indent=2)
    return dest, spec


def split_panels(bgr):
    """The Commons file is a 2-up: hang (left) | peak (right)."""
    h, w = bgr.shape[:2]
    mid = w // 2
    return {"bottom": bgr[:, :mid], "peak": bgr[:, mid:]}


def detect_pose_and_mask(bgr, model_path):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=True,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_img)
    landmarker.close()
    if not res.pose_landmarks:
        raise RuntimeError("MediaPipe Pose found no person in the reference panel")
    lms = res.pose_landmarks[0]
    xy = np.zeros((33, 2), np.float64)
    vis = np.zeros(33, np.float64)
    for i, lm in enumerate(lms):
        xy[i] = (float(lm.x), float(lm.y))  # origin top-left, 0-1 of THIS panel
        vis[i] = float(getattr(lm, "visibility", 1.0) or 1.0)
    mask = None
    if res.segmentation_masks:
        mask = np.array(res.segmentation_masks[0].numpy_view())
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0.45).astype(np.uint8) * 255
    return xy, vis, mask


def letterbox_to(xy_panel, mask, panel_shape, canvas_wh):
    """Map panel-normalized (x right, y down) into a canvas, uniform scale."""
    cw, ch = canvas_wh
    ph, pw = panel_shape[:2]
    s = min(cw / pw, ch / ph)
    nw, nh = int(round(pw * s)), int(round(ph * s))
    x0, y0 = (cw - nw) // 2, (ch - nh) // 2
    # panel xy (0-1, y-down) -> canvas xy (0-1, y-down)
    xy_c = np.empty_like(xy_panel)
    xy_c[:, 0] = (x0 + xy_panel[:, 0] * nw) / cw
    xy_c[:, 1] = (y0 + xy_panel[:, 1] * nh) / ch
    canvas_mask = np.zeros((ch, cw), np.uint8)
    if mask is not None:
        import cv2
        rs = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas_mask[y0:y0 + nh, x0:x0 + nw] = rs
    meta = {"scale": s, "x0": x0, "y0": y0, "nw": nw, "nh": nh, "cw": cw, "ch": ch}
    return xy_c, canvas_mask, meta


def prepare(root, canvas_wh=(540, 960), spec=None):
    dest, spec = download_reference(root, spec)
    import cv2
    bgr = cv2.imread(dest)
    if bgr is None:
        raise RuntimeError(f"failed to read reference {dest}")
    panels = split_panels(bgr)
    model = os.path.join(root, "_scratch", "models", "pose_landmarker_full.task")
    if not os.path.isfile(model):
        raise RuntimeError(f"missing MediaPipe model {model}")
    out_dir = scratch_dir(root)
    packed = {"spec": spec, "canvas": list(canvas_wh), "panels": {}}
    for name, pan in panels.items():
        xy, vis, mask = detect_pose_and_mask(pan, model)
        xy_c, mask_c, box = letterbox_to(xy, mask, pan.shape, canvas_wh)
        np.save(os.path.join(out_dir, f"lm2d_{name}.npy"), xy_c)
        np.save(os.path.join(out_dir, f"vis_{name}.npy"), vis)
        if mask_c is not None:
            cv2.imwrite(os.path.join(out_dir, f"sil_{name}.png"), mask_c)
        cv2.imwrite(os.path.join(out_dir, f"panel_{name}.png"), pan)
        packed["panels"][name] = {
            "panel_wh": [int(pan.shape[1]), int(pan.shape[0])],
            "letterbox": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                          for k, v in box.items()},
            "n_visible": int((vis > 0.4).sum()),
        }
    json.dump(packed, open(os.path.join(out_dir, "prepared.json"), "w"), indent=2)
    return out_dir, packed


# ---- fixed camera (matches Blender VERTICAL sensor_fit, sensor_height=36) ----

def camera_from_cfg(camc):
    look = np.array(camc.get("look") or [0.18, 0.18, 0.78], np.float64)
    dist = float(camc.get("dist_m", 2.55))
    yaw = math.radians(float(camc.get("yaw_back_deg", 8)))
    el = math.radians(float(camc.get("elev_deg", 8)))
    # 0 yaw = pure working-side (+X); +yaw rotates toward the feet (−Y)
    dirv = _n([math.cos(yaw) * math.cos(el), -math.sin(yaw) * math.cos(el), math.sin(el)])
    loc = look + dirv * dist
    fwd = _n(look - loc)             # camera −Z
    right = _n(np.cross(fwd, [0.0, 0.0, 1.0]))
    up = np.cross(right, fwd)
    return {
        "look": look, "loc": loc, "fwd": fwd, "right": right, "up": up,
        "lens": float(camc.get("lens_mm", 50)),
        "shift_x": float(camc.get("shift_x", 0.0)),
        "shift_y": float(camc.get("shift_y", 0.02)),
        "dirv": dirv, "dist": dist,
        "yaw_back_deg": float(camc.get("yaw_back_deg", 8)),
        "elev_deg": float(camc.get("elev_deg", 8)),
    }


def project_ndc(X, cam, res_wh):
    """X (...,3) world -> NDC (...,2) with origin bottom-left (Blender)."""
    X = np.asarray(X, np.float64)
    rel = X - cam["loc"]
    z = rel @ cam["fwd"]
    x = rel @ cam["right"]
    y = rel @ cam["up"]
    cw, ch = res_wh
    asp = ch / float(cw)               # H/W
    kf = cam["lens"] / 18.0
    # VERTICAL fit: half-height at z is (18/lens)*z
    ndc_x = 0.5 + (x / np.maximum(z, 1e-4)) * kf * asp / 2.0 - cam["shift_x"] * asp
    ndc_y = 0.5 + (y / np.maximum(z, 1e-4)) * kf / 2.0 - cam["shift_y"]
    return np.stack([ndc_x, ndc_y], -1), z


def _restore_lengths(P, Lpairs):
    """Pull child toward parent to restore bone length (keeps parent fixed)."""
    P = P.copy()
    for a, b, length in Lpairs:
        d = P[b] - P[a]
        n = np.linalg.norm(d)
        if n < 1e-8:
            continue
        P[b] = P[a] + d * (length / n)
    return P


def _ik(sh, wt, la, lf, pole_hint):
    sh = np.asarray(sh, np.float64)
    wt = np.asarray(wt, np.float64)
    pole_hint = np.asarray(pole_hint, np.float64)
    dvec = wt - sh
    d = float(np.linalg.norm(dvec))
    reach = 0.98 * (la + lf)
    if d < 1e-4:
        dvec = np.array([0.0, 0.0, -1e-4]); d = 1e-4
    if d > reach:
        dvec = dvec * (reach / d); wt = sh + dvec; d = reach
    u = dvec / d
    x = (la * la - lf * lf + d * d) / (2.0 * d)
    h = math.sqrt(max(la * la - x * x, 0.0))
    pole = pole_hint - sh
    pole = pole - np.dot(pole, u) * u
    pn = np.linalg.norm(pole)
    if pn < 1e-6:
        pole = np.array([1.0, 0.0, 0.0]); pole = pole - np.dot(pole, u) * u
        pn = np.linalg.norm(pole)
    pole = pole / max(pn, 1e-9)
    return sh + u * x + pole * h, wt


def _flatten_torso(P, torso_len, max_deg=8.0):
    hip = 0.5 * (P[23] + P[24]); sh = 0.5 * (P[11] + P[12])
    d = sh - hip
    horiz = math.hypot(float(d[0]), float(d[1]))
    pitch = math.atan2(float(d[2]), max(horiz, 1e-6))
    lim = math.radians(max_deg)
    if abs(pitch) > lim:
        pitch = math.copysign(lim, pitch)
    nd = np.array([d[0], d[1], math.tan(pitch) * horiz], np.float64)
    n = np.linalg.norm(nd)
    if n < 1e-8:
        nd = np.array([0.0, torso_len, 0.0])
    else:
        nd = nd * (torso_len / n)
    delta = (hip + nd) - sh
    P[11] += delta; P[12] += delta
    for j in range(0, 11):
        P[j] += delta
    return P


def _clamp_rom_and_contacts(P, spec, L):
    """Plant pad/floor contacts, two-bone IK, shoulder/elbow ROM, no penetration."""
    pad_z = float(spec.get("pad_z", 0.49))
    hx, hy = spec.get("pad_half_xy", (0.15, 0.625))
    La = float(L["upper_arm"]); Lf = float(L["forearm"])
    Lt = float(L["thigh"]); Ls = float(L["shin"])
    torso_len = float(L["torso"])

    P = _flatten_torso(P, torso_len, max_deg=3.0)
    # keep the pelvis level
    mz = 0.5 * (P[23, 2] + P[24, 2])
    P[23, 2] = mz; P[24, 2] = mz

    # support knee + hand planted on the pad; hand 50-60 cm ahead of the knee
    P[25, 0] = float(np.clip(P[25, 0], -hx + 0.02, hx - 0.02))
    P[25, 1] = float(np.clip(P[25, 1], -hy + 0.08, hy - 0.20))
    P[25, 2] = pad_z + 0.04
    P[15, 0] = float(np.clip(P[15, 0], -hx + 0.02, hx - 0.02))
    P[15, 1] = float(np.clip(P[15, 1], P[25, 1] + 0.50, P[25, 1] + 0.60))
    P[15, 2] = pad_z + 0.02
    P[27] = np.array([P[25, 0], P[25, 1] - Ls, pad_z + 0.03])
    P[29] = P[27] + np.array([0.0, 0.04, -0.01])
    P[31] = P[27] + np.array([0.0, -0.08, -0.01])

    # thigh length from planted knee: solve hip z (keep hip xy)
    dx = P[23, 0] - P[25, 0]; dy = P[23, 1] - P[25, 1]
    horiz = math.hypot(float(dx), float(dy))
    if horiz < Lt * 0.99:
        P[23, 2] = P[25, 2] + math.sqrt(Lt * Lt - horiz * horiz)
    else:
        s = (Lt * 0.98) / horiz
        P[23, 0] = P[25, 0] + dx * s
        P[23, 1] = P[25, 1] + dy * s
        P[23, 2] = P[25, 2] + 0.08
    P[24, 2] = P[23, 2]

    # support arm IK from the (possibly moved) left shoulder onto the planted hand
    pole = P[11] + np.array([0.0, -0.18, 0.0])
    P[13], hand = _ik(P[11], P[15], La, Lf, pole)
    P[15] = hand
    P[15, 0] = float(np.clip(P[15, 0], -hx, hx))
    P[15, 2] = pad_z + 0.02

    # standing foot on the floor beside the bench
    P[28, 0] = max(float(P[28, 0]), hx + 0.03)
    P[28, 2] = min(float(P[28, 2]), 0.10)
    pole = P[24] + np.array([0.10, 0.12, 0.02])
    P[26], ank = _ik(P[24], P[28], Lt, Ls, pole)
    P[28] = ank
    if P[28, 2] > 0.14:
        tgt = np.array([max(P[24, 0], hx + 0.05), P[24, 1] - 0.08, 0.06])
        P[26], P[28] = _ik(P[24], tgt, Lt, Ls, pole)
    P[30] = np.array([P[28, 0], P[28, 1] - 0.06, 0.0])
    P[32] = np.array([P[28, 0] + 0.02, P[28, 1] + 0.14, 0.0])

    # working arm lengths (parent -> child)
    for a, b, length in ((12, 14, La), (14, 16, Lf)):
        d = P[b] - P[a]; n = np.linalg.norm(d)
        if n > 1e-8:
            P[b] = P[a] + d * (length / n)

    # elbow ROM 40–170 deg on the working (right) arm
    sh, el, wr = P[12], P[14], P[16]
    u = sh - el
    v = wr - el
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu > 1e-6 and nv > 1e-6:
        ang = math.degrees(math.acos(float(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))
        lo, hi = 40.0, 170.0
        if ang < lo or ang > hi:
            tgt = lo if ang < lo else hi
            axis = _n(np.cross(u, v))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.array([1.0, 0.0, 0.0])
            delta = math.radians(tgt - ang)
            k = axis
            v2 = (v * math.cos(delta) + np.cross(k, v) * math.sin(delta)
                  + k * np.dot(k, v) * (1 - math.cos(delta)))
            P[16] = el + v2 * (nv / max(np.linalg.norm(v2), 1e-8))

    # no arm-torso penetration: working wrist stays outside the ribcage
    hip_m = 0.5 * (P[23] + P[24])
    sh_m = 0.5 * (P[11] + P[12])
    torso = sh_m - hip_m
    tn = _n(torso)
    wrel = P[16] - hip_m
    tt = float(np.clip(np.dot(wrel, tn), 0.0, np.linalg.norm(torso)))
    near = hip_m + tn * tt
    away = P[16] - near
    dist = np.linalg.norm(away)
    min_clear = 0.14
    if dist < min_clear:
        side = away if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
        P[16] = near + _n(side) * min_clear
    mid_x = 0.5 * (P[11, 0] + P[12, 0])
    P[16, 0] = max(P[16, 0], mid_x + 0.06)

    # shoulder ROM: upper arm must not point toward the head and ceiling
    ua = P[14] - P[12]
    if np.dot(ua, tn) > 0.12 and ua[2] > 0.12:
        P[14] = P[12] + _n(np.array([0.10, -0.15, -0.05])) * np.linalg.norm(ua)
    return P


def fit_frame(P0, ref_xy_down, vis, cam, res_wh, spec, L, steps=40, phase="bottom"):
    """Move 3D landmarks so their reprojection matches ref 2D (y-down stored)."""
    P = np.asarray(P0, np.float64).copy()
    ref = np.asarray(ref_xy_down, np.float64).copy()
    # convert ref to Blender NDC (y up)
    ref_up = ref.copy()
    ref_up[:, 1] = 1.0 - ref_up[:, 1]
    La = float(L["upper_arm"]); Lf = float(L["forearm"])
    Lt = float(L["thigh"]); Ls = float(L["shin"])
    torso = float(L["torso"])
    # restore order: roots first
    pairs = [
        (23, 25, Lt), (25, 27, Ls),
        (24, 26, Lt), (26, 28, Ls),
        (11, 13, La), (13, 15, Lf),
        (12, 14, La), (14, 16, Lf),
    ]
    tracked = [j for j in MOVE if vis[j] > 0.55]
    last_err = 1e9
    it = 0
    for it in range(steps):
        ndc, z = project_ndc(P, cam, res_wh)
        err = 0.0
        wsum = 0.0
        cw, ch = res_wh
        for j in tracked:
            w = W_JOINT[j] * float(vis[j])
            dlt = ref_up[j] - ndc[j]
            err += w * float(dlt @ dlt)
            wsum += w
            depth = max(float(z[j]), 0.25)
            world_per_ndc_y = depth * (36.0 / cam["lens"])
            world_per_ndc_x = world_per_ndc_y * (cw / float(ch))
            step = 0.35 * (1.0 - 0.012 * it)
            P[j] = P[j] + cam["right"] * dlt[0] * world_per_ndc_x * step * min(w, 1.6) \
                        + cam["up"] * dlt[1] * world_per_ndc_y * step * min(w, 1.6)
        if phase == "peak":
            # Demic squeeze: drive the working elbow back and the wrist to the hip
            # line. The side-view photo is a shallower row; the peak prior wins.
            La = float(L["upper_arm"]); Lf = float(L["forearm"])
            tgt = P[24] + np.array([0.13, -0.03, -0.06])
            P[16] = 0.25 * P[16] + 0.75 * tgt
            el_hint = P[12] + np.array([0.06, -0.28, 0.00])
            P[14], P[16] = _ik(P[12], P[16], La, Lf, el_hint)
        P = _clamp_rom_and_contacts(P, spec, L)
        mean = err / max(wsum, 1e-6)
        last_err = mean
        if mean < 1.2e-4:
            break
    ndc, _ = project_ndc(P, cam, res_wh)
    mae = float(np.mean([np.linalg.norm(ref_up[j] - ndc[j]) for j in tracked]))
    return P, {"joint_mae_ndc": round(mae, 4), "iters": it + 1, "tracked": len(tracked)}


def fit_cycle(B, u, ref_dir, cam, res_wh, spec, L):
    """Fit hang (u=0) and peak (u=1) independently, lerp the cycle."""
    u = np.asarray(u, np.float64)
    bottom = np.load(os.path.join(ref_dir, "lm2d_bottom.npy"))
    peak = np.load(os.path.join(ref_dir, "lm2d_peak.npy"))
    vb = np.load(os.path.join(ref_dir, "vis_bottom.npy"))
    vp = np.load(os.path.join(ref_dir, "vis_peak.npy"))
    i0 = int(np.argmin(u)); i1 = int(np.argmax(u))
    Pb, info_b = fit_frame(B[i0], bottom, vb, cam, res_wh, spec, L, phase='bottom')
    Pp, info_p = fit_frame(B[i1], peak, vp, cam, res_wh, spec, L, phase='peak')
    out = np.empty_like(B)
    for i, uu in enumerate(u):
        out[i] = Pb * (1.0 - uu) + Pp * uu
    return out, {"bottom": info_b, "peak": info_p}


def beauty_silhouette(bgr, bg=(38, 38, 38), tol=28):
    d = np.linalg.norm(bgr.astype(np.int16) - np.array(bg, np.int16)[None, None, :], axis=2)
    return (d > tol).astype(np.uint8) * 255


def silhouette_iou(a, b):
    a = a > 127; b = b > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def overlay_qc(beauty_bgr, ref_dir, proj_xy_down, tag="peak",
               iou_min=0.35, joint_mae_max=0.08, out_path=None, ours_sil=None):
    """Build the overlay image and score IoU + joint MAE.

    proj_xy_down: (33,2) our joints in 0-1, origin top-left (image space).
    ours_sil: optional side-camera silhouette (used when the beauty is a
    different view, e.g. a 3/4 render).
    """
    import cv2
    ref_xy = np.load(os.path.join(ref_dir, f"lm2d_{tag}.npy"))
    vis = np.load(os.path.join(ref_dir, f"vis_{tag}.npy"))
    ref_sil = cv2.imread(os.path.join(ref_dir, f"sil_{tag}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = beauty_bgr.shape[:2]
    if ref_sil is None or ref_sil.shape[:2] != (h, w):
        ref_sil = cv2.resize(ref_sil if ref_sil is not None else np.zeros((h, w), np.uint8),
                             (w, h), interpolation=cv2.INTER_NEAREST)
    if ours_sil is None:
        ours_sil = beauty_silhouette(beauty_bgr)
        canvas = (beauty_bgr.astype(np.float32) * 0.45).astype(np.uint8)
    else:
        if ours_sil.ndim == 3:
            ours_sil = cv2.cvtColor(ours_sil, cv2.COLOR_BGR2GRAY)
        if ours_sil.shape[:2] != (h, w):
            ours_sil = cv2.resize(ours_sil, (w, h), interpolation=cv2.INTER_NEAREST)
        # Side-space QC on a dark canvas (beauty may be a different camera).
        canvas = np.full((h, w, 3), (18, 18, 20), np.uint8)
        panel = cv2.imread(os.path.join(ref_dir, f"panel_{tag}.png"))
        if panel is not None:
            ph, pw = panel.shape[:2]
            s = min(w / pw, h / ph)
            nw, nh = int(round(pw * s)), int(round(ph * s))
            x0, y0 = (w - nw) // 2, (h - nh) // 2
            rs = cv2.resize(panel, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas[y0:y0 + nh, x0:x0 + nw] = (rs.astype(np.float32) * 0.35).astype(np.uint8)
    iou = silhouette_iou(ours_sil, ref_sil)

    proj = np.asarray(proj_xy_down, np.float64)
    tracked = [j for j in W_JOINT
               if vis[j] > 0.55 and float(np.linalg.norm(proj[j])) > 1e-6]
    mae = float(np.mean([np.linalg.norm(ref_xy[j] - proj[j]) for j in tracked])) if tracked else 1.0
    diag = math.hypot(1.0, 1.0)
    mae_diag = mae / diag

    # overlay canvas: beauty darkened (or dark + ref panel) + silhouettes + joints
    tint = canvas.astype(np.float32)
    ref_m = ref_sil > 127
    our_m = ours_sil > 127
    both = np.logical_and(ref_m, our_m)
    only_r = np.logical_and(ref_m, np.logical_not(our_m))
    only_o = np.logical_and(our_m, np.logical_not(ref_m))
    tint[only_r] = tint[only_r] * 0.4 + np.array([40, 80, 220], np.float32) * 0.6   # BGR ref-only = red-ish
    tint[only_o] = tint[only_o] * 0.4 + np.array([80, 220, 80], np.float32) * 0.6    # ours-only = green
    tint[both] = tint[both] * 0.3 + np.array([40, 220, 220], np.float32) * 0.7       # overlap = yellow
    canvas = np.clip(tint, 0, 255).astype(np.uint8)

    def _pt(xy):
        return (int(np.clip(xy[0], 0, 1) * (w - 1)), int(np.clip(xy[1], 0, 1) * (h - 1)))

    for j in tracked:
        pr = _pt(ref_xy[j]); po = _pt(proj[j])
        cv2.circle(canvas, pr, 5, (40, 80, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, po, 5, (80, 255, 80), -1, cv2.LINE_AA)
        cv2.line(canvas, pr, po, (240, 240, 240), 1, cv2.LINE_AA)

    ok_iou = iou >= iou_min
    ok_j = mae_diag <= joint_mae_max
    passed = bool(ok_iou and ok_j)
    label = (f"sil IoU {iou:.3f} ({'PASS' if ok_iou else 'FAIL'}>={iou_min:.2f})   "
             f"joint MAE {mae_diag:.3f} diag ({'PASS' if ok_j else 'FAIL'}<={joint_mae_max:.2f})")
    cv2.rectangle(canvas, (0, 0), (w, 64), (16, 16, 18), -1)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (80, 230, 120) if passed else (80, 80, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "red=ref joints/sil  green=ours  yellow=overlap", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 184), 1, cv2.LINE_AA)
    if out_path:
        cv2.imwrite(out_path, canvas)
    return {
        "sil_iou": round(iou, 4),
        "joint_mae_ndc": round(mae, 4),
        "joint_mae_diag": round(mae_diag, 4),
        "sil_iou_min": iou_min,
        "joint_mae_max": joint_mae_max,
        "pass_iou": bool(ok_iou),
        "pass_joints": bool(ok_j),
        "pass": passed,
        "overlay": out_path,
        "n_tracked": len(tracked),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "overlay"])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--beauty")
    ap.add_argument("--proj")
    ap.add_argument("--tag", default="peak")
    ap.add_argument("--out")
    ap.add_argument("--ours-sil")
    ap.add_argument("--iou-min", type=float, default=0.35)
    ap.add_argument("--mae-max", type=float, default=0.08)
    a = ap.parse_args()
    if a.cmd == "prepare":
        d, info = prepare(a.root)
        print("prepared", d, json.dumps(info["panels"], indent=2))
    else:
        import cv2
        bgr = cv2.imread(a.beauty)
        proj = np.load(a.proj)
        osil = None
        if a.ours_sil:
            osil = cv2.imread(a.ours_sil, cv2.IMREAD_GRAYSCALE)
            if osil is None and a.ours_sil.endswith(".npy"):
                osil = np.load(a.ours_sil)
                if osil.max() <= 1.0:
                    osil = (osil * 255).astype(np.uint8)
        q = overlay_qc(bgr, scratch_dir(a.root), proj, a.tag, a.iou_min, a.mae_max, a.out, ours_sil=osil)
        print(json.dumps(q, indent=2))
