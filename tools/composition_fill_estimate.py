#!/usr/bin/env python3
"""t685u: body-fill estimate per camera for the Matrix Aura Low Row G3-MS53 (230 x 71 x 240 cm, L x W x H, spec sheet)
with a seated 178 cm user. Pinhole camera (35 mm-equiv ~ 50 mm, vertical FOV 39.6 deg for a 9:16 frame), the camera
distance is solved so the WHOLE machine + person fit the 9:16 frame with 4 % margins; reports person/machine height
fraction. Machine axis: x = 0 tower back .. 230 rail end, y = width, z = up. Wireframe points are approximate."""
import json, numpy as np
TOWER = [(x, y, z) for x in (0, 62) for y in (-35, 35) for z in (0, 240)]
RAIL = [(x, y, z) for x in (62, 230) for y in (-20, 20) for z in (0, 12)]
SEAT = [(x, y, z) for x in (165, 215) for y in (-18, 18) for z in (40, 52)]
FOOT = [(x, y, z) for x in (78, 98) for y in (-25, 25) for z in (5, 55)]
MACH = TOWER + RAIL + SEAT + FOOT
PERSON = {"head_top": (198, 0, 132), "head": (196, 0, 120), "neck": (194, 0, 106), "sh_l": (194, 21, 100), "sh_r": (194, -21, 100),
          "elb_l": (205, 24, 76), "elb_r": (205, -24, 76), "wr_l": (175, 6, 74), "wr_r": (175, -6, 74), "hip_l": (190, 12, 55),
          "hip_r": (190, -12, 55), "kn_l": (135, 14, 72), "kn_r": (135, -14, 72), "ank_l": (100, 14, 30), "ank_r": (100, -14, 30)}


def project(P, cam, tgt, f):
    c = np.array(cam, float); t = np.array(tgt, float); fw = t - c; fw /= np.linalg.norm(fw)
    rt = np.cross(fw, [0, 0, 1.0]); rt /= np.linalg.norm(rt); up = np.cross(rt, fw)
    P = np.array(P, float) - c; z = P @ fw
    return np.stack([P @ rt / z * f, -(P @ up) / z * f], 1)


def fit(yaw, pitch, tgt=(150, 0, 100), W=1080, H=1920, margin=0.04, frame_pts=None):
    """yaw: 0 = camera on the +x side (behind the seat, looking at the tower), 90 = side view (+y); pitch deg up."""
    f = (H / 2) / np.tan(np.radians(39.6 / 2)); best = None
    pts_fit = frame_pts if frame_pts is not None else MACH + list(PERSON.values())
    for d in np.linspace(120, 2500, 477):
        cam = (tgt[0] + d * np.cos(np.radians(pitch)) * np.cos(np.radians(yaw)), tgt[1] + d * np.cos(np.radians(pitch)) * np.sin(np.radians(yaw)),
               tgt[2] + d * np.sin(np.radians(pitch)))
        a = project(pts_fit, cam, tgt, f)
        span_x = a[:, 0].max() - a[:, 0].min(); span_y = a[:, 1].max() - a[:, 1].min()
        if span_x <= W * (1 - 2 * margin) and span_y <= H * (1 - 2 * margin):
            best = (d, cam); break
    d, cam = best
    am = project(MACH, cam, tgt, f); ap = project(list(PERSON.values()), cam, tgt, f)
    allp = np.vstack([am, ap]); off = np.array([W / 2, H / 2]) - 0.5 * (allp.max(0) + allp.min(0))
    return {"yaw": yaw, "pitch": pitch, "dist_cm": round(float(d)),
            "person_h_frac": round(float((ap[:, 1].max() - ap[:, 1].min()) / H), 3),
            "machine_h_frac": round(float((am[:, 1].max() - am[:, 1].min()) / H), 3),
            "all_h_frac": round(float((allp[:, 1].max() - allp[:, 1].min()) / H), 3),
            "all_w_frac": round(float((allp[:, 0].max() - allp[:, 0].min()) / W), 3),
            "_cam": cam, "_tgt": tgt, "_f": f, "_off": off.tolist()}


if __name__ == "__main__":
    out = {"side_now": fit(90, 5), "rear34_high": fit(35, 28), "front34_low": fit(150, -6), "front34_eye": fit(140, 8)}
    # 2-shot tight cut: frame only the person (machine part-visible), 6 % margin
    out["tight_body_rear34"] = fit(35, 22, tgt=(170, 0, 80), frame_pts=list(PERSON.values()), margin=0.07)
    out["stack_top_wide_rear34"] = fit(35, 28)  # same wide shot, shown in the top 42 % band of a vertical stack
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in out.items()}, open("docs/mission/COMPOSITION_fill_estimate.json", "w"), indent=1)
    for k, v in out.items(): print(k, {kk: vv for kk, vv in v.items() if not kk.startswith("_")})
