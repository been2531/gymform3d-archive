#!/usr/bin/env python3
"""Reusable procedural motion authoring: YAML keyposes + timing curves -> smooth landmark tracks.

Used by retarget_render_v2 when motion.authored.enabled. Replaces per-frame MediaPipe body
jitter with an eased anatomical cycle (first pose == last pose, matching C2 tangents).
Presets live in modules/authored/<preset>.yaml and are merged under motion.authored.
"""
from __future__ import annotations
import copy, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MOD = os.path.join(ROOT, "modules", "authored")


def deep_merge(base, over):
    out = copy.deepcopy(base) if base is not None else {}
    if not isinstance(over, dict):
        return copy.deepcopy(over)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_spec(cfg):
    """Merge motion.authored with modules/authored/<preset>.yaml if a preset is named."""
    raw = copy.deepcopy((cfg.get("motion") or {}).get("authored") or {})
    preset = raw.get("preset")
    base = {}
    # If the caller already inlined phases+profiles (Blender's bpyenv has no PyYAML), skip the file.
    already = bool(raw.get("phases") and (raw.get("right") or raw.get("wrong")))
    if preset and not already:
        path = os.path.join(MOD, f"{preset}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"authored preset missing: {path}")
        try:
            import yaml
            with open(path) as f:
                base = yaml.safe_load(f) or {}
        except ImportError:
            jpath = os.path.join(MOD, f"{preset}.json")
            if not os.path.exists(jpath):
                raise ImportError("PyYAML missing and no modules/authored/%s.json sidecar" % preset)
            import json
            base = json.load(open(jpath))
    return deep_merge(base, raw)


def smootherstep(x):
    """C2-flat at 0 and 1: f=f'=f''=0 at ends (loop-safe phase joins)."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def ease_time(x, kind):
    """Warp the phase clock; smootherstep is applied after so endpoints stay C2-flat."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    kind = (kind or "ease_in_out").lower()
    if kind in ("hold", "linear_hold"):
        return x
    if kind in ("ease_out", "out"):
        return 1.0 - (1.0 - x) ** 2
    if kind in ("ease_in", "in"):
        return x ** 2
    if kind in ("yank", "punch"):
        return 1.0 - (1.0 - x) ** 1.35
    return x  # ease_in_out: identity then smootherstep


def phase_u(spec, form="right"):
    """Return (u, fps, dur_s) where u[t] in [0,1] is pull progress (0=hang, 1=peak)."""
    form = (form or "right").lower()
    fps = float(spec.get("fps", 30))
    phases = list(spec.get("phases") or [])
    override = ((spec.get(form) or {}).get("phases")) or {}
    # allow form-specific duration overrides by phase name
    named = []
    for ph in phases:
        p = dict(ph)
        if p.get("name") in override:
            p = deep_merge(p, override[p["name"]])
        named.append(p)
    if form == "wrong" and spec.get("wrong", {}).get("concentric_dur_s"):
        for p in named:
            if p.get("name") == "concentric":
                p["dur_s"] = float(spec["wrong"]["concentric_dur_s"])
                p["ease"] = spec["wrong"].get("concentric_ease", p.get("ease", "yank"))
    if not named:
        named = [
            {"name": "concentric", "dur_s": 0.80, "ease": "ease_out", "u0": 0, "u1": 1},
            {"name": "squeeze", "dur_s": 0.30, "ease": "hold", "u0": 1, "u1": 1},
            {"name": "eccentric", "dur_s": 1.40, "ease": "ease_in_out", "u0": 1, "u1": 0},
            {"name": "rest", "dur_s": 0.50, "ease": "hold", "u0": 0, "u1": 0},
        ]
    counts = [max(1, int(round(float(p["dur_s"]) * fps))) for p in named]
    # drop one frame from the last phase so total frames = sum(dur)*fps exactly when possible
    n = int(sum(counts))
    u = np.zeros(n, dtype=np.float64)
    i = 0
    for p, c in zip(named, counts):
        xs = np.linspace(0.0, 1.0, c, endpoint=False)
        kind = p.get("ease", "ease_in_out")
        u0, u1 = float(p.get("u0", 0)), float(p.get("u1", 1))
        if kind == "hold" or abs(u1 - u0) < 1e-12:
            uu = np.full(c, u1)
        else:
            w = smootherstep(ease_time(xs, kind))
            uu = u0 + (u1 - u0) * w
        u[i:i + c] = uu
        i += c
    # force seamless endpoints (matching pose); C2-flat phases already have u'(end)=0
    u[0] = float(named[0].get("u0", 0.0))
    u[-1] = float(named[-1].get("u1", named[-1].get("u0", 0.0)))
    dur = n / fps
    return u, fps, dur


def _n(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _ik_elbow(sh, wt, la, lf, pole_hint):
    dvec = wt - sh
    d = float(np.linalg.norm(dvec))
    reach = 0.98 * (la + lf)
    if d < 1e-4:
        d = 1e-4
        dvec = np.array([0.0, 0.0, -1e-4])
    if d > reach:
        dvec = dvec * (reach / d)
        wt = sh + dvec
        d = reach
    u = dvec / d
    x = (la * la - lf * lf + d * d) / (2 * d)
    h = math.sqrt(max(la * la - x * x, 0.0))
    pole = pole_hint - sh
    pole = pole - np.dot(pole, u) * u
    pn = np.linalg.norm(pole)
    if pn < 1e-6:
        pole = np.array([0.0, 0.0, 1.0])
        pole = pole - np.dot(pole, u) * u
        pn = np.linalg.norm(pole)
        if pn < 1e-6:
            pole = np.array([1.0, 0.0, 0.0])
            pole = pole - np.dot(pole, u) * u
            pn = np.linalg.norm(pole)
    pole = pole / max(pn, 1e-9)
    return sh + u * x + pole * h, wt


def expand_landmarks(B, spec, side, form="right", arm_len=None):
    """Rebuild landmark tracks from a planted/scaled template B (any F>=1) into one seamless authored cycle.
    arm_len=(la, lf) uses rig bone lengths so DAMPED_TRACK lands on the authored elbow/wrist."""
    form = (form or "right").lower()
    u, fps, dur = phase_u(spec, form)
    n = int(u.shape[0])
    tmpl = np.median(B, axis=0) if B.ndim == 3 else np.asarray(B, dtype=np.float64)
    if tmpl.ndim == 3:
        tmpl = tmpl[0]
    prof = deep_merge(spec.get("right") or {}, spec.get(form) or {} if form != "right" else {})
    # stance frame from template
    hip = 0.5 * (tmpl[23] + tmpl[24])
    sh0 = 0.5 * (tmpl[11] + tmpl[12])
    tv = sh0 - hip
    fwd = _n(np.array([tv[0], tv[1], 0.0])) if np.hypot(tv[0], tv[1]) > 1e-6 else np.array([0.0, -1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    lat = _n(np.cross(up, fwd))  # +lat = toward the figure's left
    # working side: MediaPipe 11=L, 12=R; side==11 left working
    left = (int(side) == 11)
    w_sh_i, w_el_i, w_wr_i = (11, 13, 15) if left else (12, 14, 16)
    s_sh_i, s_el_i, s_wr_i = (12, 14, 16) if left else (11, 13, 15)
    lat_out = -lat if left else lat  # away from midline on the working side
    if arm_len is not None:
        la, lf = float(arm_len[0]), float(arm_len[1])
    else:
        la = float(np.linalg.norm(tmpl[w_sh_i] - tmpl[w_el_i]))
        lf = float(np.linalg.norm(tmpl[w_el_i] - tmpl[w_wr_i]))
    if la < 0.05: la = 0.28
    if lf < 0.05: lf = 0.26
    torso_len = float(np.linalg.norm(tv)) or 0.45
    half_w = 0.5 * float(np.linalg.norm(tmpl[11] - tmpl[12])) or 0.16
    head_off = {j: tmpl[j] - sh0 for j in list(range(0, 11)) + [17, 18, 19, 20, 21, 22]}
    sup_off = {j: tmpl[j] - tmpl[s_sh_i] for j in (s_el_i, s_wr_i)}
    for j in (17, 19, 21) if not left else (18, 20, 22):
        if j not in head_off:
            head_off[j] = tmpl[j] - sh0

    torso_start = math.radians(float(prof.get("torso_start_deg", prof.get("torso_deg", 49))))
    torso_peak = math.radians(float(prof.get("torso_peak_deg", prof.get("torso_deg", 49))))
    thor_amp = math.radians(float(prof.get("thoracic_rot_deg", 6.0)))
    flex0 = float(prof.get("elbow_flex_start_deg", 10.0))
    flex1 = float(prof.get("elbow_flex_peak_deg", 95.0))
    # aim the row plane at the waist / hip-pocket (iliac, not the low hip-joint landmark)
    tuck_deg = float(prof.get("elbow_tuck_deg", 8.0))
    flare = bool(prof.get("elbow_flare", False))
    if flare:
        tuck_deg = float(prof.get("elbow_flare_deg", -18.0))
    breath_m = float(spec.get("breathing_amp_m", 0.006))
    hip_w = tmpl[23] if left else tmpl[24]

    out = np.repeat(tmpl[None, ...], n, axis=0)
    # feet / ankles / heels stay exactly at the planted template (no slide)
    for j in (27, 28, 29, 30, 31, 32, 23, 24, 25, 26):
        out[:, j] = tmpl[j]

    for i, uu in enumerate(u):
        td = torso_start + (torso_peak - torso_start) * uu
        axis = _n(math.sin(td) * fwd + math.cos(td) * up)
        # thoracic rotation about the spine (working shoulder retracts)
        ang = thor_amp * uu
        ca, sa = math.cos(ang), math.sin(ang)
        # Rodrigues about `axis` for the shoulder-girdle lateral
        def rot(v):
            return v * ca + np.cross(axis, v) * sa + axis * np.dot(axis, v) * (1 - ca)
        lat_t, fwd_t = rot(lat), rot(fwd)
        lat_out_t = -lat_t if left else lat_t
        breath = breath_m * math.sin(math.pi * uu)     # exhale toward the peak, inhale toward hang
        shm = hip + torso_len * axis
        lsh = shm - (half_w + breath) * lat_t
        rsh = shm + (half_w + breath) * lat_t
        out[i, 11], out[i, 12] = lsh, rsh
        sh = out[i, w_sh_i]
        # Anatomical one-arm row in the sagittal plane through the working shoulder:
        # rotate world-down toward the back (shoulder extension) so the elbow drives
        # up-and-back beside the ribs — not medially through the torso toward the hip joint.
        # Forearm stays world-vertical (wrist under the elbow, no roll). Hang flex~10,
        # peak flex~95; the hand traces a parallel arc toward the hip pocket.
        down = -up
        flex = flex0 + (flex1 - flex0) * uu
        # axis that rotates `down` toward `-fwd` (backward extension)
        axis_el = np.cross(down, -fwd_t)
        if np.linalg.norm(axis_el) < 1e-6:
            axis_el = lat_out_t
        axis_el = _n(axis_el)
        def rodrig(v, ax, deg):
            a = math.radians(deg); c, s = math.cos(a), math.sin(a)
            return v * c + np.cross(ax, v) * s + ax * np.dot(ax, v) * (1 - c)
        hum = _n(rodrig(down, axis_el, flex))
        # small adduction (tuck) or abduction (flare) about the forward axis
        if abs(tuck_deg) > 1e-3:
            probe = _n(rodrig(hum, fwd_t, 6.0))
            # pick the sign that moves the elbow toward (+) or away (−) from the chest mid
            d0 = np.linalg.norm((sh + hum * la) - shm)
            d1 = np.linalg.norm((sh + probe * la) - shm)
            sign = 1.0 if d1 < d0 else -1.0
            if tuck_deg < 0:
                sign = -sign
            hum = _n(rodrig(hum, fwd_t, sign * abs(tuck_deg) * uu))
        el = sh + hum * la
        wt = el + down * lf
        out[i, w_el_i] = el
        out[i, w_wr_i] = wt
        # rigid head / face cluster from the chest (gaze fixed, no landmark jitter)
        for j, off in head_off.items():
            # re-express the rest offset in the hinged+rotated shoulder frame
            # rest offset was in world; rotate the world-horizontal part with thoracic rot
            out[i, j] = shm + rot(off)
        # support arm keeps rest shape relative to its (moving) shoulder; later pinned to thigh
        for j, off in sup_off.items():
            out[i, j] = out[i, s_sh_i] + off

    meta = {"u": u, "fps": fps, "dur_s": dur, "n": n, "form": form,
            "torso_start_deg": math.degrees(torso_start), "torso_peak_deg": math.degrees(torso_peak)}
    return out, meta
