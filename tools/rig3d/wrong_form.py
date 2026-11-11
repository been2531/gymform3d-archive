#!/usr/bin/env python3
"""Procedural wrong-form generator for boxing clips (Blender).

Takes a *correct* punch (a Mixamo clip + guard/full/ret frames) and layers
additive, envelope-weighted bone rotations on top of the original action,
so the rest of the motion (footwork, torso, timing) stays mocap-real.

Each registry entry names exactly one boxing fault (boxing_qc.FAULTS). After
generation the clip is re-measured with boxing_qc.form_features +
reference_distance (mocap ground truth); a variant is valid only if the
intended fault is newly flagged and no other fault appears.

  nice -n 19 blender -b --python tools/rig3d/wrong_form.py -- \
      --type jab --verify --render-sheet

Operations (all in world space, applied parent→child per frame, then keyed):
  toward(bone, end, dir, deg)   rotate bone subtree about its head so the
                                head→end vector turns toward dir
  roll(bone, a, b, deg, pref)   rotate bone subtree about the a→b line (e.g.
                                shoulder→wrist: elbow orbits, fist stays)
  open(bone, deg)               open (+) / close (−) a hinge toward its parent
  pitch(bone, end, deg)         raise the subtree about the body's lateral axis
                                (no sideways component)
  aim(bone, end)                rotate bone so `end` returns to its pre-edit
                                world position (keeps the fist on target)
  yaw_counter(bone, k)          remove k·(yaw change since guard) about Z,
                                children listed in `keep` keep world rotation
Envelopes (0..1 over the punch window): bump, hold, drive, after_full,
after_mid, pre, pre_full, const.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import boxing_qc as qc  # noqa: E402
import eval_xbot_boxing as xb  # noqa: E402
import pick_xbot_punches_v3 as v3  # noqa: E402

OUT = xb.OUT_DIR
WF_DIR = os.path.join(OUT, "wrongform")

# ------------------------------------------------------------------ registry
# side tokens: "s" = punching side, "o" = other side ("l"/"r" resolved per punch)
VARIANTS = {
    # ---- jab
    "jab_elbow_flare": {
        "type": "jab", "fault": "elbow_flare", "label": "Chicken wing (elbow flare)",
        "env": "bump",
        # v2: 92° roll, elbow allowed just above shoulder so the wing reads
        # instantly from a front 3/4 (v1 65° + cap z=-0.01 was almost invisible).
        "ops": [("roll", "arm_s", "arm_s", "hand_s", 92.0, "out", ["hand_s"])],
        "elbow_cap_z": 0.05,
    },
    "jab_dropped_guard": {
        "type": "jab", "fault": "rear_guard_drop", "label": "Dropped rear guard",
        "env": "hold",
        "ops": [("toward", "arm_o", "hand_o", "down", 62.0),
                ("open", "fore_o", 48.0),
                ("toward", "arm_o", "hand_o", "out", 16.0)],
    },
    "jab_chin_up": {
        "type": "jab", "fault": "chin_up", "label": "Chin up",
        "env": "const",
        "ops": [("toward", "head", None, "back", 30.0)],
    },
    "jab_overextension": {
        "type": "jab", "fault": "overextension", "label": "Overextension lean",
        "env": "drive",
        "ops": [("toward", "spine", "head", "fwd", 14.0),
                ("toward", "spine1", "head", "fwd", 10.0),
                ("aim", "arm_s", "hand_s")],
    },
    "jab_no_retraction": {
        "type": "jab", "fault": "no_retraction", "label": "No retraction (fist drops)",
        "env": "after_full",
        "ops": [("toward", "arm_s", "hand_s", "down", 58.0),
                ("open", "fore_s", 38.0)],
    },
    # ---- cross
    "cross_no_hip": {
        "type": "cross", "fault": "no_hip_rotation", "label": "No hip rotation / pivot",
        "env": "hold",
        "ops": [("yaw_counter", "hips", 0.85, ["upleg_l", "upleg_r"])],
    },
    "cross_overhand_loop": {
        "type": "cross", "fault": "looping_path", "label": "Looping overhand",
        "env": "pre_full",
        "ops": [("pitch", "arm_s", "fore_s", 34.0)],
    },
    # ---- hook
    "hook_elbow_low": {
        "type": "hook", "fault": "elbow_low", "label": "Elbow too low (wide swing)",
        "env": "bump",
        "ops": [("toward", "arm_s", "fore_s", "down", 26.0),
                ("aim", "fore_s", "hand_s")],
    },
    "hook_arm_open": {
        "type": "hook", "fault": "arm_too_open", "label": "Arm too open",
        "env": "bump",
        "ops": [("open", "fore_s", 45.0)],
    },
    # ---- uppercut
    "uppercut_telegraph": {
        "type": "uppercut", "fault": "telegraph", "label": "Winding down (telegraph)",
        "env": "dip",
        "ops": [("toward", "arm_s", "hand_s", "down", 50.0), ("open", "fore_s", 25.0)],
    },
    "uppercut_overswing": {
        "type": "uppercut", "fault": "overswing", "label": "Overswing",
        "env": "after_mid",
        "ops": [("toward", "arm_s", "hand_s", "up", 40.0), ("open", "fore_s", 62.0)],
    },
}
SHEET_JAB = ["jab_elbow_flare", "jab_dropped_guard", "jab_no_retraction"]


# ------------------------------------------------------------------ helpers
def _smooth(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def envelope(kind, f, g, mid, full, ret, pad=4):
    """0..1 weight at frame f. Every envelope starts/ends at 0 for blending."""
    if kind == "bump":  # rises guard→full, falls full→ret
        if f <= g - pad or f >= ret + pad:
            return 0.0
        if f <= mid:
            return _smooth((f - (g - pad)) / max(1, mid - (g - pad)))
        if f <= full:
            return 1.0
        return 1.0 - _smooth((f - full) / max(1, ret + pad - full))
    if kind == "hold":  # in by mid, out after ret
        if f <= g - pad or f >= ret + 2 * pad:
            return 0.0
        if f <= mid:
            return _smooth((f - (g - pad)) / max(1, mid - (g - pad)))
        if f <= ret:
            return 1.0
        return 1.0 - _smooth((f - ret) / (2 * pad))
    if kind == "after_full":  # 0 until full, 1 at ret, eases out late
        if f <= full or f >= ret + 3 * pad:
            return 0.0
        if f <= ret:
            return _smooth((f - full) / max(1, ret - full))
        return 1.0 - _smooth((f - ret) / (3 * pad))
    if kind == "after_mid":
        if f <= mid or f >= ret + pad:
            return 0.0
        if f <= full:
            return _smooth((f - mid) / max(1, full - mid))
        return 1.0 - _smooth((f - full) / max(1, ret + pad - full))
    if kind == "pre":  # dip before the drive, gone at full
        a = g - 2 * pad
        if f <= a or f >= full:
            return 0.0
        if f <= g:
            return _smooth((f - a) / max(1, g - a))
        return 1.0 - _smooth((f - g) / max(1, full - g))
    if kind == "pre_full":  # strongest mid-outbound, small at full
        if f <= g - pad or f >= ret + pad:
            return 0.0
        if f <= mid:
            return _smooth((f - (g - pad)) / max(1, mid - (g - pad)))
        if f <= full:
            return 1.0 - 0.6 * _smooth((f - mid) / max(1, full - mid))
        return 0.4 * (1.0 - _smooth((f - full) / max(1, ret + pad - full)))
    if kind == "dip":  # fist sinks early in the drive, gone by full
        pk = g + 0.3 * (full - g)
        if f <= g or f >= full:
            return 0.0
        if f <= pk:
            return _smooth((f - g) / max(1e-6, pk - g))
        return 1.0 - _smooth((f - pk) / max(1e-6, full - pk))
    if kind == "drive":  # 0 at guard, 1 at full, gone shortly after ret
        if f <= g or f >= ret + 2 * pad:
            return 0.0
        if f <= full:
            return _smooth((f - g) / max(1, full - g))
        return 1.0 - _smooth((f - full) / max(1, ret + 2 * pad - full))
    if kind == "const":
        if f <= g - 2 * pad or f >= ret + 2 * pad:
            return 0.0
        if f < g:
            return _smooth((f - (g - 2 * pad)) / (2 * pad))
        if f <= ret:
            return 1.0
        return 1.0 - _smooth((f - ret) / (2 * pad))
    raise ValueError(kind)


def _resolve(key, s, o):
    if key is None:
        return None
    return key.replace("_s", "_" + s).replace("_o", "_" + o) if key.endswith(("_s", "_o")) else key


def _pb(ctx, key):
    name = ctx["bones"].get(key)
    return ctx["arm"].pose.bones.get(name) if name else None


def _world(ctx, pb):
    return ctx["arm"].matrix_world @ pb.matrix


def _head(ctx, pb):
    return _world(ctx, pb).to_translation()


def _tail(ctx, pb):
    return ctx["arm"].matrix_world @ pb.tail


def _point(ctx, key):
    pb = _pb(ctx, key)
    return _head(ctx, pb) if pb else None


def _set_world(ctx, pb, M):
    pb.matrix = ctx["arm"].matrix_world.inverted() @ M
    bpy.context.view_layer.update()


def _rotate_about(ctx, pb, axis, angle, pivot):
    if axis.length < 1e-9 or abs(angle) < 1e-6:
        return
    R = Matrix.Rotation(angle, 4, axis.normalized())
    T = Matrix.Translation(pivot) @ R @ Matrix.Translation(-pivot)
    _set_world(ctx, pb, T @ _world(ctx, pb))


def _dirs(ctx, s, o):
    a_s, a_o = _point(ctx, f"arm_{s}"), _point(ctx, f"arm_{o}")
    out = (a_s - a_o)
    out.z = 0
    out = out.normalized() if out.length > 1e-6 else Vector((1, 0, 0))
    hl, hr, hp = _point(ctx, "hand_l"), _point(ctx, "hand_r"), _point(ctx, "hips")
    fwd = (hl + hr) * 0.5 - hp
    fwd.z = 0
    fwd = fwd.normalized() if fwd.length > 1e-6 else Vector((0, -1, 0))
    up = Vector((0, 0, 1))
    return {"up": up, "down": -up, "fwd": fwd, "back": -fwd, "out": out,
            "out_up": (out + 0.7 * up).normalized()}


def _op_toward(ctx, pb, end_key, dirv, deg):
    head = _head(ctx, pb)
    end = _point(ctx, end_key) if end_key else _tail(ctx, pb)
    v = end - head
    if v.length < 1e-6:
        return
    ang = min(math.radians(deg), v.angle(dirv) if v.angle(dirv) > 1e-6 else 0.0)
    _rotate_about(ctx, pb, v.cross(dirv), ang, head)


def _op_roll(ctx, pb, a_key, b_key, deg, pref, elbow_key, to_up_w=None, elbow_cap_z=None):
    a, b = _point(ctx, a_key), _point(ctx, b_key)
    axis = b - a
    if axis.length < 1e-6:
        return
    e = _point(ctx, elbow_key)
    if to_up_w is not None:  # align the elbow offset with "up" (⊥ axis)
        n = axis.normalized()
        off = (e - a) - n * (e - a).dot(n)
        up = Vector((0, 0, 1))
        upp = up - n * up.dot(n)
        if off.length < 1e-6 or upp.length < 1e-6:
            return
        ang = off.angle(upp)
        sgn = 1.0 if off.cross(upp).dot(n) >= 0 else -1.0
        _rotate_about(ctx, pb, n, sgn * ang * to_up_w, a)
        return
    best = None
    for sgn in (1, -1):
        R = Matrix.Rotation(sgn * math.radians(deg), 3, axis.normalized())
        e2 = a + R @ (e - a)
        score = (e2 - e).dot(pref)
        if best is None or score > best[0]:
            best = (score, sgn)
    ang = best[1] * math.radians(deg)
    if elbow_cap_z is not None:  # never lift the elbow above shoulder + cap
        for _ in range(12):
            R = Matrix.Rotation(ang, 3, axis.normalized())
            if (a + R @ (e - a)).z <= a.z + elbow_cap_z or abs(ang) < 1e-3:
                break
            ang *= 0.85
    _rotate_about(ctx, pb, axis, ang, a)


def _op_open(ctx, pb, deg):
    par = pb.parent
    if par is None:
        return
    j = _head(ctx, pb)
    p0 = _head(ctx, par)
    child = pb.children[0] if pb.children else None
    end = _head(ctx, child) if child else _tail(ctx, pb)
    upper = (j - p0)
    lower = (end - j)
    if upper.length < 1e-6 or lower.length < 1e-6:
        return
    # opening = turning `lower` toward the continuation of `upper`
    target = upper.normalized()
    max_ang = lower.angle(target) if lower.angle(target) > 1e-6 else 0.0
    ang = math.radians(deg)
    ang = min(ang, max_ang) if deg > 0 else max(ang, -math.radians(150) + max_ang)
    _rotate_about(ctx, pb, lower.cross(target), ang, j)


def _yaw_of(ctx, key_l="upleg_l", key_r="upleg_r"):
    pl, pr = _point(ctx, key_l), _point(ctx, key_r)
    return math.atan2(pr.y - pl.y, pr.x - pl.x)


# ------------------------------------------------------------------ apply
def keyframe_bone(pb, f):
    pb.keyframe_insert("location", frame=f)
    if pb.rotation_mode == "QUATERNION":
        pb.keyframe_insert("rotation_quaternion", frame=f)
    else:
        pb.keyframe_insert("rotation_euler", frame=f)


def apply_variant(ctx, punch, name, strength=1.0):
    """Edit ctx's action in place. Returns per-frame weights for the log."""
    spec = VARIANTS[name]
    s = "l" if punch["side"] == "L" else "r"
    o = "r" if s == "l" else "l"
    g, mid, full, ret = (int(punch[k]) for k in ("guard", "mid", "full", "ret"))
    f0 = max(ctx["f0"], g - 10)
    f1 = min(ctx["f1"], ret + 14)
    # guard yaw for yaw_counter
    bpy.context.scene.frame_set(g)
    bpy.context.view_layer.update()
    yaw_g = _yaw_of(ctx)
    weights = {}
    for f in range(f0, f1 + 1):
        w = envelope(spec["env"], f, g, mid, full, ret) * strength
        weights[f] = round(w, 3)
        if w <= 1e-4:
            continue
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        dirs = _dirs(ctx, s, o)
        orig_pts = {k: _point(ctx, k).copy() for k in (f"hand_{s}", f"hand_{o}")}
        touched = []
        for op in spec["ops"]:
            kind = op[0]
            if kind == "toward":
                _, bk, ek, dk, deg = op
                pb = _pb(ctx, _resolve(bk, s, o))
                _op_toward(ctx, pb, _resolve(ek, s, o), dirs[dk], deg * w)
                touched.append(pb)
            elif kind == "roll":
                _, bk, ak, bk2, deg, pk, keep = op
                pb = _pb(ctx, _resolve(bk, s, o))
                keep_pbs = [_pb(ctx, _resolve(k, s, o)) for k in keep]
                saved = [(kp, _world(ctx, kp).copy()) for kp in keep_pbs]
                cap_z = spec.get("elbow_cap_z", -0.01 if pk == "out" else None)
                _op_roll(ctx, pb, _resolve(ak, s, o), _resolve(bk2, s, o), deg * w,
                         dirs.get(pk, dirs["up"]), f"fore_{s}" if "_s" in bk else f"fore_{o}",
                         to_up_w=(w * deg if pk == "to_up" else None),
                         elbow_cap_z=cap_z)
                for kp, M in saved:  # keep fist orientation (wrist is on the axis)
                    loc = _world(ctx, kp).to_translation()
                    M2 = M.copy()
                    M2.translation = loc
                    _set_world(ctx, kp, M2)
                    touched.append(kp)
                touched.append(pb)
            elif kind == "pitch":  # raise (+) the subtree about the lateral axis only
                _, bk, ek, deg = op
                pb = _pb(ctx, _resolve(bk, s, o))
                head = _head(ctx, pb)
                end = _point(ctx, _resolve(ek, s, o))
                ax = dirs["out"]
                best = None
                for sgn in (1, -1):
                    R = Matrix.Rotation(sgn * math.radians(deg * w), 3, ax)
                    dz = (head + R @ (end - head)).z - end.z
                    if best is None or dz > best[0]:
                        best = (dz, sgn)
                _rotate_about(ctx, pb, ax, best[1] * math.radians(deg * w), head)
                touched.append(pb)
            elif kind == "aim":  # keep the fist on its original target point
                _, bk, ek = op
                pb = _pb(ctx, _resolve(bk, s, o))
                ekr = _resolve(ek, s, o)
                head = _head(ctx, pb)
                v_now = _point(ctx, ekr) - head
                v_want = orig_pts[ekr] - head
                if v_now.length > 1e-6 and v_want.length > 1e-6:
                    _rotate_about(ctx, pb, v_now.cross(v_want), v_now.angle(v_want), head)
                touched.append(pb)
            elif kind == "open":
                _, bk, deg = op
                pb = _pb(ctx, _resolve(bk, s, o))
                _op_open(ctx, pb, deg * w)
                touched.append(pb)
            elif kind == "yaw_counter":
                _, bk, k, keep = op
                pb = _pb(ctx, bk)
                keep_pbs = [_pb(ctx, kk) for kk in keep]
                saved = [(kp, _world(ctx, kp).copy()) for kp in keep_pbs]
                d = _yaw_of(ctx) - yaw_g
                d = (d + math.pi) % (2 * math.pi) - math.pi
                _rotate_about(ctx, pb, Vector((0, 0, 1)), -k * d * w, _head(ctx, pb))
                for kp, M in saved:
                    loc = _world(ctx, kp).to_translation()
                    M2 = M.copy()
                    M2.translation = loc
                    _set_world(ctx, kp, M2)
                    touched.append(kp)
                touched.append(pb)
        seen = set()
        for pb in touched:
            if pb and pb.name not in seen:
                seen.add(pb.name)
                keyframe_bone(pb, f)
    bpy.context.view_layer.update()
    return weights


# ------------------------------------------------------------------ measure
def measure(ctx, punch, ptype, ref=None):
    rows = v3.analyze_rich(ctx["arm"], ctx["bones"], ctx["f0"], ctx["f1"])
    feats = qc.form_features(rows, punch, lead=punch.get("lead_side"))
    rd = qc.reference_distance(feats, ptype, ref)
    return {"features": {k: v for k, v in feats.items() if k != "curves"}, "ref": rd,
            "rows": rows}


def verdict(base_faults, var_faults, intended):
    new = sorted(set(var_faults) - set(base_faults))
    return {
        "intended": intended,
        "new_faults": new,
        "intended_flagged": intended in var_faults,
        "exact": new == [intended],
        "lost_base_faults": sorted(set(base_faults) - set(var_faults)),
    }


# ------------------------------------------------------------------ CLI
def _cli():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = {"type": "jab", "verify": False, "sheet": False, "variants": None,
         "punch": None, "facts": None, "yaw": None, "out_name": None}
    i = 0
    while i < len(argv):
        t = argv[i]
        if t == "--type":
            a["type"] = argv[i + 1]; i += 2; continue
        if t == "--variants":
            a["variants"] = argv[i + 1].split(","); i += 2; continue
        if t == "--punch":
            a["punch"] = argv[i + 1]; i += 2; continue
        if t == "--facts":
            a["facts"] = argv[i + 1]; i += 2; continue
        if t == "--yaw":
            a["yaw"] = float(argv[i + 1]); i += 2; continue
        if t == "--out-name":
            a["out_name"] = argv[i + 1]; i += 2; continue
        if t == "--verify":
            a["verify"] = True
        if t == "--render-sheet":
            a["sheet"] = True
        i += 1
    return a


def load_base_punch(ptype, punch_id=None, facts_path=None):
    facts_path = facts_path or os.path.join(OUT, "xbot_punch_pick_facts_v4.json")
    facts = json.load(open(facts_path))
    pid = punch_id or facts["winners"][ptype]
    p = next(x for x in facts["punches"] if x["id"] == pid)
    spec = next(c for c in v3.CATALOG if c["id"] == p["clip_id"])
    return p, spec


def run(a):
    os.makedirs(WF_DIR, exist_ok=True)
    ref = qc.load_reference()
    p, spec = load_base_punch(a["type"], a["punch"], a["facts"])
    names = a["variants"] or [k for k, v in VARIANTS.items() if v["type"] == a["type"]]
    print(f"BASE {a['type']} {p['id']} ({spec['file']})", flush=True)
    report = {"type": a["type"], "base_punch": p["id"], "clip": spec["file"], "variants": {}}
    ctx = v3.load_clip(spec)
    base = measure(ctx, p, a["type"], ref)
    report["base"] = {"features": base["features"], "ref": base["ref"]}
    bf = base["ref"].get("faults") or []
    print(f"  base dist={base['ref'].get('distance')} faults={bf}", flush=True)
    for n in names:
        ctx = v3.load_clip(spec)
        t0 = time.time()
        apply_variant(ctx, p, n)
        m = measure(ctx, p, a["type"], ref)
        vf = m["ref"].get("faults") or []
        vd = verdict(bf, vf, VARIANTS[n]["fault"])
        report["variants"][n] = {"label": VARIANTS[n]["label"], "features": m["features"],
                                 "ref": m["ref"], "verdict": vd,
                                 "elapsed_s": round(time.time() - t0, 1)}
        print(f"  {n}: dist={m['ref'].get('distance')} faults={vf} exact={vd['exact']} "
              f"new={vd['new_faults']} worst={m['ref'].get('worst')}", flush=True)
    out = os.path.join(WF_DIR, f"wrongform_{a['type']}_{p['id']}_verify.json")
    qc.write_qc_json(out, report)
    print("WROTE", out, flush=True)
    if a["sheet"]:
        render_sheet(p, spec, [n for n in (a["variants"] or SHEET_JAB)], report,
                     yaw=a.get("yaw"), out_name=a.get("out_name"))
    return report


# ------------------------------------------------------------------ sheet
def _chest_forward(arm, bones, frame):
    """Horizontal chest-forward vector at `frame` from the shoulder line."""
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()
    sh_l = xb.bone_world(arm, bones["sh_l"]) if bones.get("sh_l") else None
    sh_r = xb.bone_world(arm, bones["sh_r"]) if bones.get("sh_r") else None
    if sh_l is None or sh_r is None:
        return Vector((0.0, -1.0, 0.0))
    across = sh_r - sh_l
    across.z = 0.0
    if across.length < 1e-6:
        return Vector((0.0, -1.0, 0.0))
    # (sh_r - sh_l) × world-up points roughly chest-forward for a Mixamo T-pose
    # that was turned to face -Y (rest across ≈ +X, +X × +Z = +Y = back), so
    # flip to get forward.
    fwd = across.normalized().cross(Vector((0.0, 0.0, 1.0)))
    if fwd.length < 1e-6:
        return Vector((0.0, -1.0, 0.0))
    fwd.normalize()
    # If this points backward of -Y more than forward, leave it; the sign is
    # chosen so a rest-facing -Y character has fwd ≈ -Y.
    rest = Vector((0.0, -1.0, 0.0))
    if fwd.dot(rest) < 0:
        fwd = -fwd
    return fwd


def front_three_quarter_yaw(arm, bones, frame, offset_deg=-38.0):
    """Yaw (degrees) for a front 3/4 of the chest, not a side profile.

    setup_fitted_camera places the camera at (sin yaw, -cos yaw); yaw=0 is
    world -Y (true front after face_minus_y). A bladed boxing stance turns
    the chest toward -X, so YAW=35 reads as a side view. Orbit ~38° off the
    chest-forward so both gloves and the elbow line are readable.
    """
    fwd = _chest_forward(arm, bones, frame)
    # Camera sits in front of the chest: direction look→camera = -fwd, then
    # orbit `offset_deg` about Z for a 3/4 (negative = toward the lead/left).
    cam = -fwd
    ang = math.radians(offset_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    x, y = cam.x * ca - cam.y * sa, cam.x * sa + cam.y * ca
    # direction = (sin yaw, -cos yaw)  =>  yaw = atan2(x, -y)
    return math.degrees(math.atan2(x, -y))


def render_sheet(p, spec, names, report, res=(432, 768), yaw=None, out_name=None):
    stages = [("mid", int(p["mid"])), ("full", int(p["full"])), ("ret", int(p["ret"]))]
    cols = [None] + list(names)
    cam_saved = None
    paths = {}
    old_res = xb.RES
    xb.RES = res
    used_yaw = yaw
    try:
        for col in cols:
            ctx = v3.load_clip(spec)
            if col:
                apply_variant(ctx, p, col)
            xb.assign_clay(ctx["meshes"])
            frames = [int(p["guard"]), int(p["mid"]), int(p["full"]), int(p["ret"])]
            if used_yaw is None:
                used_yaw = front_three_quarter_yaw(ctx["arm"], ctx["bones"], int(p["full"]))
                # A bladed jab at yaw≈70 still reads as a side profile. Fall
                # back to a known front 3/4 (both gloves + chest visible).
                if abs(used_yaw) > 50:
                    used_yaw = -25.0
                print(f"front 3/4 yaw={used_yaw:.1f}", flush=True)
            v3.setup_fitted_camera(ctx["meshes"], frames, ctx["arm"], ctx["bones"],
                                   ctx["start_xy"], ctx["lock"], yaw=used_yaw)
            cam = bpy.context.scene.camera
            if cam_saved is None:
                # widen a touch so every variant fits the correct clip's camera
                cam.data.lens *= 0.94
                cam_saved = (cam.location.copy(), cam.rotation_euler.copy(), cam.data.lens)
            else:  # identical camera for every column (fair comparison)
                cam.location = cam_saved[0]
                cam.rotation_euler = cam_saved[1]
                cam.data.lens = cam_saved[2]
            bpy.context.view_layer.update()
            for stage, f in stages:
                bpy.context.scene.frame_set(f)
                bpy.context.view_layer.update()
                if ctx["lock"]:
                    xb.lock_root_xy(ctx["arm"], ctx["bones"], ctx["start_xy"])
                fp = os.path.join(WF_DIR, f"wf_{col or 'correct'}_{stage}.png")
                xb.render_png(fp)
                paths[(col or "correct", stage)] = fp
                print(f"  RENDER {fp}", flush=True)
    finally:
        xb.RES = old_res
    meta = {"paths": {f"{k[0]}|{k[1]}": v for k, v in paths.items()},
            "cols": [c or "correct" for c in cols], "stages": [s for s, _ in stages],
            "labels": {c: VARIANTS[c]["label"] for c in names},
            "report": {k: {"faults": v["ref"].get("faults"), "dist": v["ref"].get("distance"),
                           "exact": v["verdict"]["exact"]} for k, v in report["variants"].items()},
            "base": {"faults": report["base"]["ref"].get("faults"),
                     "dist": report["base"]["ref"].get("distance")},
            "punch": p["id"],
            "yaw": used_yaw,
            "view": "front_3_4"}
    mp = os.path.join(WF_DIR, "sheet_meta.json")
    qc.write_qc_json(mp, meta)
    print("SHEET_META", mp, flush=True)
    # Compose the v2 sheet by default when a name is given; leave v1 as
    # the fallback so an unnamed call still builds the historic filename.
    try:
        import subprocess
        cmd = [
            "/usr/bin/python3",
            os.path.join(HERE, "build_wrongform_sheet.py"),
        ]
        # build_wrongform_sheet.main() takes out_name; pass via env
        env = dict(os.environ)
        env["WRONGFORM_OUT"] = out_name or "xbot_jab_wrongform_v2.png"
        subprocess.check_call(cmd, env=env)
    except Exception as exc:
        print(f"SHEET_COMPOSE_FAILED {exc}", flush=True)


if __name__ == "__main__":
    run(_cli())
