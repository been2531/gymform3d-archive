#!/usr/bin/env python3
"""Rebuild tools/rig3d/boxing_reference.json from free boxing mocap.

Sources (raw files are NOT committed; they live under the gitignored
output/_basemodels/boxer/{accad,cmu}/):
  * ACCAD Male2 E1-E8 (Jab/Cross/Hook/Uppercut L/R), CC BY 3.0,
    https://accad.osu.edu/sites/accad.osu.edu/files/Male2_bvh.zip
    (credit: "Motion capture data courtesy of ACCAD, The Ohio State University")
  * CMU Graphics Lab Motion Capture Database, subjects 13/14/15/17 boxing
    trials (13_17 13_18 14_01 14_02 14_03 15_13 17_10). Free for any use incl.
    commercial (http://mocap.cs.cmu.edu). BVH mirror:
    https://raw.githubusercontent.com/konyshevgmbh/cmu-mocap/master/data/0SS/SS_TT.bvh

Run (pure Python + numpy, no Blender):
  python3 tools/rig3d/extract_boxing_reference.py --download   # fetch raw files
  python3 tools/rig3d/extract_boxing_reference.py               # rebuild JSON

Punch windows are found with the same peak logic as pick_xbot_punches_v3
(reimplemented here bpy-free). ACCAD clips are labelled (type + hand from the
file name). CMU trials are unlabelled: only clearly classifiable punches are
kept (straight: elbow>=150 at extension; hook: elbow 70-120 with little rise;
uppercut: rise_z dominant).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boxing_qc as qc  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(REPO, "output", "_basemodels", "boxer")
OUT_JSON = os.path.join(HERE, "boxing_reference.json")
TARGET_FPS = 30.0
NORM_H = 1.75  # positions are rescaled so estimated body height = 1.75 m

ACCAD = {
    "Male2_E1_JabLeft.bvh": ("jab", "L"),
    "Male2_E2_JabRight.bvh": ("jab", "R"),
    "Male2_E3_CrossLeft.bvh": ("cross", "L"),
    "Male2_E4_CrossRight.bvh": ("cross", "R"),
    "Male2_E5_HookLeft.bvh": ("hook", "L"),
    "Male2_E6_HookRight.bvh": ("hook", "R"),
    "Male2_E7_UppercutLeft.bvh": ("uppercut", "L"),
    "Male2_E8_UppercutRight.bvh": ("uppercut", "R"),
}
CMU = ["13_17", "13_18", "14_01", "14_02", "14_03", "15_13", "17_10"]
CMU_URL = "https://raw.githubusercontent.com/konyshevgmbh/cmu-mocap/master/data/{s:03d}/{t}.bvh"
ACCAD_ZIP = "https://accad.osu.edu/sites/accad.osu.edu/files/Male2_bvh.zip"


# --------------------------------------------------------------------- BVH
class Joint:
    def __init__(self, name, parent):
        self.name, self.parent = name, parent
        self.offset = np.zeros(3)
        self.channels: list[str] = []
        self.children: list[Joint] = []
        self.end = None  # end-site offset


def parse_bvh(path):
    with open(path) as fh:
        tokens = fh.read().split()
    i = 0
    joints: list[Joint] = []
    stack: list[Joint] = []
    cur = None
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            cur = Joint(tokens[i + 1], stack[-1] if stack else None)
            if cur.parent:
                cur.parent.children.append(cur)
            joints.append(cur)
            i += 2
        elif t == "End":
            # End Site { OFFSET x y z }
            j = i + 2
            assert tokens[j] == "{"
            assert tokens[j + 1] == "OFFSET"
            stack[-1].end = np.array([float(x) for x in tokens[j + 2:j + 5]])
            i = j + 6
        elif t == "{":
            stack.append(cur)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            cur.offset = np.array([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            cur.channels = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    nf = int(tokens[i + 2])
    ft = float(tokens[i + 5])
    data = np.asarray(tokens[i + 6:], dtype=np.float64)
    nch = sum(len(j.channels) for j in joints)
    data = data[: nf * nch].reshape(nf, nch)
    return joints, data, ft


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    n = len(a)
    m = np.zeros((n, 3, 3))
    if axis == "X":
        m[:, 0, 0] = 1; m[:, 1, 1] = c; m[:, 1, 2] = -s; m[:, 2, 1] = s; m[:, 2, 2] = c
    elif axis == "Y":
        m[:, 1, 1] = 1; m[:, 0, 0] = c; m[:, 0, 2] = s; m[:, 2, 0] = -s; m[:, 2, 2] = c
    else:
        m[:, 2, 2] = 1; m[:, 0, 0] = c; m[:, 0, 1] = -s; m[:, 1, 0] = s; m[:, 1, 1] = c
    return m


def fk(joints, data):
    """World positions (Y-up, file units) for every joint + end sites."""
    nf = data.shape[0]
    pos, rot = {}, {}
    col = 0
    for j in joints:
        R = np.tile(np.eye(3), (nf, 1, 1))
        T = np.zeros((nf, 3))
        for ch in j.channels:
            v = data[:, col]
            col += 1
            if ch.endswith("position"):
                T[:, "XYZ".index(ch[0])] = v
            else:
                R = R @ _rot(ch[0], v)
        if j.parent is None:
            pos[j.name] = j.offset + T
            rot[j.name] = R
        else:
            pr, pp = rot[j.parent.name], pos[j.parent.name]
            pos[j.name] = pp + np.einsum("nij,j->ni", pr, j.offset) + np.einsum("nij,nj->ni", pr, T)
            rot[j.name] = pr @ R
        if j.end is not None:
            pos[j.name + "_End"] = pos[j.name] + np.einsum("nij,j->ni", rot[j.name], j.end)
    return pos


NAMES = {
    "hips": ["Hips"], "head": ["Head"], "headtop": ["Head_End"],
    "neck": ["Neck", "Neck1"],
    "hand_l": ["LeftHand"], "hand_r": ["RightHand"],
    "fore_l": ["LeftForeArm"], "fore_r": ["RightForeArm"],
    "arm_l": ["LeftArm"], "arm_r": ["RightArm"],
    "sh_l": ["LeftShoulder"], "sh_r": ["RightShoulder"],
    "upleg_l": ["LeftUpLeg"], "upleg_r": ["RightUpLeg"],
    "foot_l": ["LeftFoot"], "foot_r": ["RightFoot"],
    "toe_l": ["LeftToeBase"], "toe_r": ["RightToeBase"],
}


def bvh_to_rows(path):
    joints, data, ft = parse_bvh(path)
    pos = fk(joints, data)
    step = max(1, int(round((1.0 / ft) / TARGET_FPS)))
    P = {}
    for key, cands in NAMES.items():
        for c in cands:
            if c in pos:
                p = pos[c][::step]
                # Y-up → Z-up (x, y, z) → (x, -z, y)
                P[key] = np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)
                break
    # scale so estimated body height = NORM_H
    ua = np.linalg.norm(P["fore_l"] - P["arm_l"], axis=1) + np.linalg.norm(P["hand_l"] - P["fore_l"], axis=1)
    H = float(np.median(ua)) / qc.ARM_TO_HEIGHT
    s = NORM_H / H
    for k in P:
        P[k] = P[k] * s
    floor = float(min(P["toe_l"][:, 2].min(), P["toe_r"][:, 2].min(),
                      P["foot_l"][:, 2].min(), P["foot_r"][:, 2].min()))
    for k in P:
        P[k][:, 2] -= floor
    n = P["hips"].shape[0]
    return P, n, {"src_fps": round(1.0 / ft, 2), "step": step, "scale": s}


def orient_chunk(P, i0, i1):
    """Rotate/translate a frame chunk so the boxer faces -Y, hips at origin."""
    sl = slice(i0, i1)
    mid = 0.5 * (P["hand_l"][sl] + P["hand_r"][sl]) - P["hips"][sl]
    d = mid[:, :2].mean(0)
    if np.linalg.norm(d) < 1e-6:
        d = np.array([0.0, -1.0])
    ang = math.atan2(d[1], d[0])
    rot = -math.pi / 2 - ang  # rotate d onto -Y
    c, s = math.cos(rot), math.sin(rot)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    o = P["hips"][i0].copy()
    o[2] = 0
    return {k: (v[sl] - o) @ Rz.T for k, v in P.items()}


def chunk_rows(C, f_base=0):
    n = C["hips"].shape[0]
    rows = []
    for i in range(n):
        r = {"f": f_base + i}
        for k in NAMES:
            if k in C:
                r[k] = [float(x) for x in C[k][i]]
        for s in ("l", "r"):
            if r.get(f"hand_{s}") and r.get("hips"):
                r[f"reach_{s}"] = math.hypot(r[f"hand_{s}"][0] - r["hips"][0],
                                             r[f"hand_{s}"][1] - r["hips"][1])
            if r.get(f"hand_{s}") and r.get("head"):
                r[f"dist_{s}"] = float(np.linalg.norm(np.subtract(r[f"hand_{s}"], r["head"])))
            r[f"elbow_{s}"] = qc._angle(r[f"arm_{s}"], r[f"fore_{s}"], r[f"hand_{s}"])
            t = np.subtract(r[f"toe_{s}"], r[f"foot_{s}"])
            r[f"foot_{s}_yaw"] = math.degrees(math.atan2(t[1], t[0]))
            r[f"foot_{s}_z"] = r[f"foot_{s}"][2]
        ax = np.subtract(r["headtop"], r["head"])
        ax = ax / max(np.linalg.norm(ax), 1e-9)
        r["head_up_z"] = float(ax[2])
        r["head_axis"] = [float(x) for x in ax]
        r["head_tilt_fwd"] = float(np.dot(ax, [0.0, -1.0, 0.0]))
        rows.append(r)
    return rows


# ------------------------------------------------ punch windows (bpy-free)
def _local_peaks(values, min_dist=6, min_rise=0.10, win=14):
    n, peaks, i = len(values), [], 1
    while i < n - 1:
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            left = values[max(0, i - win): i]
            if left and values[i] - min(left) >= min_rise:
                if not peaks or i - peaks[-1] >= min_dist:
                    peaks.append(i)
                i += min_dist
                continue
        i += 1
    return peaks


def _argmin_win(values, lo, hi):
    lo, hi = max(0, lo), min(len(values) - 1, hi)
    w = values[lo:hi + 1]
    return lo + int(np.argmin(w)) if w else lo


def detect_lead(rows):
    sample = rows[: max(3, min(12, len(rows) // 5))] or rows
    votes = 0
    ly = np.mean([r["hand_l"][1] for r in sample])
    ry = np.mean([r["hand_r"][1] for r in sample])
    lf = np.mean([r["foot_l"][1] for r in sample])
    rf = np.mean([r["foot_r"][1] for r in sample])
    votes += -1 if ly < ry else 1
    votes += -1 if lf < rf else 1
    return "L" if votes <= 0 else "R"


def find_punches(rows, side, upper=False):
    s = side.lower()
    reach = [r[f"reach_{s}"] for r in rows]
    z = [r[f"hand_{s}"][2] for r in rows]
    series = z if upper else reach
    out = []
    for i in _local_peaks(series, min_dist=6, min_rise=0.09, win=14):
        g = _argmin_win(series, i - 22, i)
        rt = _argmin_win(series, i, i + 22)
        if g >= i:
            g = max(0, i - 4)
        if rt <= i:
            rt = min(len(rows) - 1, i + 4)
        out.append({"side": side, "guard": rows[g]["f"], "mid": rows[(g + i) // 2]["f"],
                    "full": rows[i]["f"], "ret": rows[rt]["f"],
                    "rise_xy": reach[i] - reach[g], "rise_z": z[i] - z[g],
                    "elbow": rows[i][f"elbow_{s}"],
                    "fvh": rows[i][f"hand_{s}"][2] - rows[i]["head"][2]})
    return out


def classify_cmu(p, lead):
    """Conservative labels for unlabelled CMU punches (None = skip).

    CMU amateurs often punch from a low hand, so straight punches can rise a
    lot in z; they are accepted when the arm ends nearly straight at head
    height and travels more forward than up."""
    is_lead = p["side"] == lead
    if (p["rise_z"] >= 0.16 and p["rise_z"] >= 1.2 * p["rise_xy"]
            and 60 <= p["elbow"] <= 125 and -0.22 <= p["fvh"] <= 0.06):
        return "uppercut"
    if (p["elbow"] >= 145 and p["rise_xy"] >= 0.25 and p["fvh"] > -0.16
            and p["rise_z"] <= 0.8 * p["rise_xy"]):
        return "jab" if is_lead else "cross"
    if (70 <= p["elbow"] <= 120 and p["rise_z"] < 0.10 and p["fvh"] > -0.18
            and p["rise_xy"] >= 0.12):
        return "hook"
    return None


# ------------------------------------------------------------- pipeline
def collect():
    samples = []
    for fn, (ptype, hand) in ACCAD.items():
        path = os.path.join(RAW, "accad", fn)
        if not os.path.isfile(path):
            print("missing", path)
            continue
        P, n, meta = bvh_to_rows(path)
        C = orient_chunk(P, 0, n)
        rows = chunk_rows(C)
        lead = detect_lead(rows)
        cands = find_punches(rows, hand, upper=(ptype == "uppercut"))
        if not cands:
            print("no punch", fn)
            continue
        # labelled file: take the strongest rep(s) on the labelled hand
        key = "rise_z" if ptype == "uppercut" else "rise_xy"
        cands.sort(key=lambda p: -p[key])
        best = cands[0]
        for p in cands:
            if p[key] < 0.6 * best[key]:
                continue
            p["lead_side"] = lead
            p["is_lead_hand"] = hand == lead
            feats = qc.form_features(rows, p, lead=lead)
            samples.append({"src": f"ACCAD/{fn}", "license": "CC BY 3.0", "type": ptype,
                            "side": hand, "lead": lead, "frames": [p["guard"], p["full"], p["ret"]],
                            "features": feats})
        print(f"ACCAD {fn}: lead={lead} kept {sum(1 for s in samples if s['src'].endswith(fn))}")
    for t in CMU:
        path = os.path.join(RAW, "cmu", f"{t}.bvh")
        if not os.path.isfile(path):
            print("missing", path)
            continue
        P, n, meta = bvh_to_rows(path)
        chunk = 90  # 3 s at 30 fps, re-oriented per chunk (subjects turn)
        kept = 0
        for c0 in range(0, n - 30, chunk // 2):
            c1 = min(n, c0 + chunk)
            C = orient_chunk(P, c0, c1)
            rows = chunk_rows(C, f_base=0)
            lead = detect_lead(rows)
            for side in ("L", "R"):
                for upper in (False, True):
                    for p in find_punches(rows, side, upper=upper):
                        # ignore windows touching the chunk edge
                        if p["guard"] < 3 or p["ret"] > len(rows) - 4:
                            continue
                        ptype = classify_cmu(p, lead)
                        if ptype is None or (ptype == "uppercut") != upper:
                            continue
                        gid = (t, c0 + p["full"], side)
                        if any(s.get("gid") == gid or (s.get("gid") and s["gid"][0] == t and
                               s["gid"][2] == side and abs(s["gid"][1] - gid[1]) <= 6)
                               for s in samples):
                            continue
                        p["lead_side"] = lead
                        p["is_lead_hand"] = side == lead
                        feats = qc.form_features(rows, p, lead=lead)
                        if "error" in feats:
                            continue
                        samples.append({"src": f"CMU/{t}", "license": "CMU free (incl. commercial)",
                                        "type": ptype, "side": side, "lead": lead, "gid": gid,
                                        "frames": [c0 + p["guard"], c0 + p["full"], c0 + p["ret"]],
                                        "features": feats})
                        kept += 1
        print(f"CMU {t}: frames@30={n} kept {kept}")
    return samples


def robust_stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) >= 6:  # trim 10% tails so odd CMU detections don't dominate
        lo, hi = np.percentile(a, [10, 90])
        a = a[(a >= lo) & (a <= hi)] if ((a >= lo) & (a <= hi)).sum() >= 3 else a
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "p10": round(float(np.percentile(vals, 10)), 4),
            "p90": round(float(np.percentile(vals, 90)), 4), "n": int(len(vals))}


def build(samples):
    types = {}
    for ptype in ("jab", "cross", "hook", "uppercut"):
        ss = [s for s in samples if s["type"] == ptype]
        if not ss:
            continue
        feats = {}
        for k in qc.FEATURE_SPEC:
            vals = [s["features"][k] for s in ss if s["features"].get(k) is not None]
            if vals:
                feats[k] = robust_stats(vals)
        curves = {}
        for ck in ("fist_rel_shoulder_H", "elbow_deg", "elbow_above_H"):
            arr = [s["features"]["curves"][ck] for s in ss
                   if len(s["features"]["curves"][ck]) == qc.CURVE_N]
            if arr:
                A = np.asarray(arr, dtype=np.float64)
                curves[ck] = {"mean": np.round(A.mean(0), 4).tolist(),
                              "std": np.round(A.std(0), 4).tolist(), "n": len(arr)}
        types[ptype] = {
            "n": len(ss),
            "n_by_source": {src: sum(1 for s in ss if s["src"].startswith(src))
                            for src in ("ACCAD", "CMU")},
            "features": feats, "curves": curves,
        }
    return {
        "version": 1,
        "units": "lengths in body heights (H); H = (upper arm + forearm) / 0.332; angles deg",
        "fps": TARGET_FPS,
        "sources": [
            {"name": "ACCAD Male2 E1-E8", "license": "CC BY 3.0",
             "credit": "Motion capture data courtesy of ACCAD, The Ohio State University",
             "url": ACCAD_ZIP},
            {"name": "CMU Graphics Lab Motion Capture Database",
             "license": "free for all uses incl. commercial (credit appreciated)",
             "trials": CMU, "url": "http://mocap.cs.cmu.edu"},
        ],
        "time_axis": "curves: 21 samples, guard->full = 0..0.5, full->return = 0.5..1",
        "types": types,
        "samples": [{k: (v if k != "features" else {kk: vv for kk, vv in v.items() if kk != "curves"})
                     for k, v in s.items() if k != "gid"} for s in samples],
    }


def download():
    import io
    import zipfile
    os.makedirs(os.path.join(RAW, "accad"), exist_ok=True)
    os.makedirs(os.path.join(RAW, "cmu"), exist_ok=True)
    zbytes = urllib.request.urlopen(ACCAD_ZIP, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        for fn in ACCAD:
            with open(os.path.join(RAW, "accad", fn), "wb") as fh:
                fh.write(z.read(fn))
    for t in CMU:
        s = int(t.split("_")[0])
        url = CMU_URL.format(s=s, t=t)
        with open(os.path.join(RAW, "cmu", f"{t}.bvh"), "wb") as fh:
            fh.write(urllib.request.urlopen(url, timeout=120).read())
    print("downloaded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    if a.download:
        download()
    samples = collect()
    ref = build(samples)
    with open(a.out, "w") as fh:
        json.dump(ref, fh, indent=1)
    for k, v in ref["types"].items():
        f = v["features"]
        print(f"{k:9s} n={v['n']:3d} {v['n_by_source']}  "
              + " ".join(f"{kk}={f[kk]['mean']:.3f}±{f[kk]['std']:.3f}"
                         for kk in qc.TYPE_FEATURES[k] if kk in f))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
