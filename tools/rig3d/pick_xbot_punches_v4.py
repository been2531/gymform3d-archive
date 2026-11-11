#!/usr/bin/env python3
"""v4: re-score Mixamo X Bot punches with whole-path QC metrics.

  nice -n 19 blender -b --python tools/rig3d/pick_xbot_punches_v4.py -- --phase analyze
  nice -n 19 blender -b --python tools/rig3d/pick_xbot_punches_v4.py -- --phase render
  nice -n 19 blender -b --python tools/rig3d/pick_xbot_punches_v4.py -- --phase iktest

Path metrics live in boxing_qc.py (reusable JSON scores). This script only
drives Blender I/O, contact strips, optional IK straighten test, and renders.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import boxing_qc as qc  # noqa: E402
import eval_xbot_boxing as xb  # noqa: E402
import pick_xbot_punches_v3 as v3  # noqa: E402

OUT = xb.OUT_DIR
FACTS_PATH = os.path.join(OUT, "xbot_punch_pick_facts_v4.json")
STRIP_DIR = os.path.join(OUT, "xbot_v4_strips")
QC_DIR = os.path.join(OUT, "qc")
IK_DIR = os.path.join(OUT, "xbot_v4_iktest")

# Clips whose punches get a frame-by-frame contact strip.
NO_STRIPS = False
STRIP_CLIPS = {
    "orig_cross", "alt18", "alt15", "alt04", "alt05", "alt06", "alt12",
    "alt16", "alt17", "orig_jab", "orig_hook", "orig_upper", "alt14", "alt01",
}
V3_IDS = {
    "jab": "alt16_jab_L_f20",
    "cross": "orig_cross_cross_R_f35",
    "hook": "orig_hook_uppercut_L_f13",
    "uppercut": "alt14_uppercut_L_f22",
}


def _cli():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    phase = "analyze"
    winners = {}
    global NO_STRIPS
    NO_STRIPS = "--no-strips" in argv
    i = 0
    while i < len(argv):
        if argv[i] == "--phase" and i + 1 < len(argv):
            phase = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--winners" and i + 1 < len(argv):
            for part in argv[i + 1].split(","):
                if ":" in part:
                    k, v = part.split(":", 1)
                    winners[k.strip()] = v.strip()
            i += 2
            continue
        i += 1
    return phase, winners


REF = qc.load_reference()


def rescore(p, want):
    base, parts = v3.score_punch(p, want)
    if parts.get("reject"):
        return 0.0, parts
    extra = qc.path_score_parts(p.get("path_qc") or {}, want)
    parts.update(extra)
    rd = (p.get("ref") or {}).get(want)
    if rd:
        parts.update(qc.reference_score_part(rd))
    return float(sum(v for v in parts.values() if isinstance(v, (int, float)))), parts


def attach_path_qc(p, rows):
    pq = qc.punch_path_metrics(rows, p, side=p["side"])
    p["path_qc"] = {k: v for k, v in pq.items() if k != "traj"}
    p["traj"] = pq.get("traj") or []
    p["is_overhand"] = bool(pq.get("is_overhand"))
    p["is_looping"] = bool(pq.get("is_looping"))
    # ground-truth mocap distance (boxing_reference.json)
    try:
        feats = qc.form_features(rows, p, lead=p.get("lead_side"))
        p["form_features"] = {k: v for k, v in feats.items() if k != "curves"}
        p["ref"] = {w: qc.reference_distance(feats, w, REF)
                    for w in ("jab", "cross", "hook", "uppercut")} if REF else {}
    except Exception as e:  # never block the pick on reference issues
        p["ref"] = {}
        p["ref_error"] = str(e)
    scores, parts = {}, {}
    for want in ("jab", "cross", "hook", "uppercut"):
        sc, br = rescore(p, want)
        scores[want] = round(sc, 2)
        parts[want] = {k: (round(v, 2) if isinstance(v, (int, float)) else v)
                       for k, v in br.items()}
    p["scores"] = scores
    p["score_parts"] = parts
    p["score"] = scores.get(p.get("ptype") or "jab", 0.0)
    return p


def should_strip(spec, p):
    if spec["id"] not in STRIP_CLIPS:
        return False
    if p.get("id") in V3_IDS.values():
        return True
    if not p.get("is_lead_hand") and (p.get("elbow_deg") or 0) >= 125:
        return True
    if spec["hint"] in ("cross", "combo") and not p.get("is_lead_hand"):
        return (p.get("rise_xy_m") or 0) > 0.12
    if spec["id"] in ("orig_jab", "alt01", "alt16") and p.get("ptype") == "jab":
        return True
    if spec["id"] in ("orig_hook",) and p.get("side") == "L":
        return (p.get("elbow_deg") or 99) < 130
    if spec["id"] in ("orig_upper", "alt14") and p.get("ptype") == "uppercut":
        return True
    return False


def render_strip(ctx, p, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    g, full, ret = int(p["guard"]), int(p["full"]), int(p["ret"])
    frames = list(range(g, ret + 1))
    step = max(1, int(math.ceil(len(frames) / 12)))
    pick = frames[::step]
    if pick[-1] != frames[-1]:
        pick.append(frames[-1])
    v3.setup_fitted_camera(ctx["meshes"], [g, full, ret],
                           ctx["arm"], ctx["bones"], ctx["start_xy"], ctx["lock"])
    xb.assign_clay(ctx["meshes"])
    written, labels = [], []
    for f in pick:
        bpy.context.scene.frame_set(int(f))
        bpy.context.view_layer.update()
        if ctx["lock"]:
            xb.lock_root_xy(ctx["arm"], ctx["bones"], ctx["start_xy"])
        tag = "G" if f == g else ("F" if f == full else ("R" if f == ret else ""))
        fp = os.path.join(dest_dir, f"f_{f:04d}.png")
        v3.render_eevee(fp, res=(150, 266), samples=6)
        written.append(fp)
        labels.append(f"{f}{tag}")
    out = os.path.join(STRIP_DIR, f"{p['id']}.png")
    os.makedirs(STRIP_DIR, exist_ok=True)
    try:
        strip = qc.compose_contact_strip(written, labels, thumb=(150, 266))
        strip.save(out, "PNG")
    except ImportError:
        # Blender's Python has no PIL: frames stay in dest_dir; compose later
        # with system python (build_xbot_picks_v4.py --compose-strips).
        print(f"  strip frames in {dest_dir} (compose with system PIL)", flush=True)
    p["strip"] = out
    print(f"  STRIP {out} n={len(written)}", flush=True)
    return out


def type_gate(p, want):
    """Is this punch geometrically the wanted type at all? (v3 scores are
    type-agnostic, so a 75-degree uppercut could top the jab list.)"""
    el = p.get("elbow_deg") or 0.0
    rz = p.get("rise_z_m") or 0.0
    rxy = p.get("rise_xy_m") or 0.0
    fvh = p.get("fist_vs_head_m") or 0.0
    if want == "jab":
        return p.get("is_lead_hand") and el >= 140 and fvh > -0.28
    if want == "cross":
        return (not p.get("is_lead_hand")) and el >= 140 and fvh > -0.28
    if want == "hook":
        return p.get("is_lead_hand") and 55 <= el <= 130 and fvh > -0.25 and rz < 0.15
    if want == "uppercut":
        return el <= 135 and rz >= 0.12 and rz >= 0.6 * rxy
    return True


def rank_all(punches):
    rankings = {}
    for want in ("jab", "cross", "hook", "uppercut"):
        ranked = sorted(punches, key=lambda x: x["scores"].get(want, 0.0), reverse=True)
        ranked = [p for p in ranked if p["scores"].get(want, 0.0) > 1 and type_gate(p, want)]
        rankings[want] = [
            {
                "id": p["id"], "clip_id": p["clip_id"], "title": p["clip_title"],
                "score": p["scores"].get(want, 0.0),
                "side": p["side"], "is_lead_hand": p["is_lead_hand"],
                "elbow_deg": p["elbow_deg"],
                "fist_vs_head_m": p["fist_vs_head_m"],
                "is_overhand": p.get("is_overhand"),
                "is_looping": p.get("is_looping"),
                "fist_max_dev_m": (p.get("path_qc") or {}).get("fist_max_dev_m"),
                "fist_arc_height_m": (p.get("path_qc") or {}).get("fist_arc_height_m"),
                "elbow_above_shoulder_max_m": (p.get("path_qc") or {}).get(
                    "elbow_above_shoulder_max_m"),
                "retract_max_dev_m": (p.get("path_qc") or {}).get("retract_max_dev_m"),
                "guard": p["guard"], "mid": p["mid"], "full": p["full"],
                "ret": p["ret"], "strip": p.get("strip"),
                "target": p.get("target"),
                "ref_distance": ((p.get("ref") or {}).get(want) or {}).get("distance"),
                "ref_closeness": ((p.get("ref") or {}).get(want) or {}).get("closeness"),
                "ref_faults": ((p.get("ref") or {}).get(want) or {}).get("faults"),
            }
            for p in ranked[:12]
        ]
    return rankings


def auto_pick(rankings, window=15.0):
    """Path-clean first, then ground truth: among candidates within `window`
    points of the best score that have no mocap-reference faults (and, for
    straight punches, no overhand/looping path), take the one closest to the
    mocap reference (lowest RMS z-distance)."""
    winners, notes = {}, {}
    for want in ("jab", "cross", "hook", "uppercut"):
        rows = rankings.get(want) or []
        if not rows:
            winners[want], notes[want] = None, "no candidates"
            continue
        if want == "hook":  # hooks curve by design; only an overhead elbow is wrong
            clean = [r for r in rows if not r.get("is_overhand")]
        else:
            clean = [r for r in rows if not r.get("is_overhand") and not r.get("is_looping")]
        if want in ("jab", "cross"):  # owner priority: elbow never above shoulder
            th = qc.load_thresholds()[want]
            clean = [r for r in clean
                     if (r.get("elbow_above_shoulder_max_m") or 0) <= th["elbow_above_shoulder_max_m"]
                     and (r.get("fist_max_dev_m") or 0) <= th["fist_max_dev_m"]]
        if want in ("jab", "cross") and not clean:
            winners[want] = rows[0]["id"]
            notes[want] = "NO clean path; least-bad by score (still looping/overhand)"
            continue
        pool = clean or rows
        top = pool[0]["score"]
        near = [r for r in pool if r["score"] >= top - window and not r.get("ref_faults")]
        if not near:
            near = [pool[0]]
        best = min(near, key=lambda r: (r.get("ref_distance") if r.get("ref_distance")
                                        is not None else 9.0))
        winners[want] = best["id"]
        if best.get("ref_faults"):
            notes[want] = (f"path-clean but mocap flags {best['ref_faults']}; no fault-free "
                           f"candidate within {window:.0f} pts; ref distance {best.get('ref_distance')}")
        else:
            notes[want] = (f"path-clean, no mocap faults, ref distance {best.get('ref_distance')} "
                           f"(closest of {len(near)} within {window:.0f} pts of top score)")
    return winners, notes


def phase_analyze():
    os.makedirs(STRIP_DIR, exist_ok=True)
    facts = {"ok": False, "clips": [], "punches": [], "rankings": {},
             "qc_version": qc.VERSION}
    t0 = time.time()
    for spec in v3.CATALOG:
        print(f"ANALYZE {spec['id']} {spec['file']}", flush=True)
        try:
            ctx = v3.load_clip(spec)
        except Exception as e:
            print(f"FAIL {spec['id']}: {e}", flush=True)
            facts["clips"].append({"id": spec["id"], "error": str(e), **spec})
            continue
        punches = v3.extract_punches(ctx["rows"], ctx["lead"])
        facts["clips"].append({
            **spec,
            "frames": [ctx["f0"], ctx["f1"]],
            "lead": ctx["lead"],
            "clip_root_drift_m": ctx["clip_drift"],
            "n_punches": len(punches),
        })
        # sort strip candidates by relevance, cap 3 per clip
        tagged = []
        for idx, p in enumerate(punches):
            p["id"] = f"{spec['id']}_{p['ptype']}_{p['side']}_f{p['full']}"
            p["clip_id"] = spec["id"]
            p["clip_title"] = spec["title"]
            p["file"] = spec["file"]
            p["lead_side"] = ctx["lead"]
            attach_path_qc(p, ctx["rows"])
            tagged.append(p)
            print(
                f"  {p['id']} type={p['ptype']} el={p['elbow_deg']:.1f} "
                f"dev={p['path_qc']['fist_max_dev_m']:.3f} "
                f"arc={p['path_qc']['fist_arc_height_m']:.3f} "
                f"elb+={p['path_qc']['elbow_above_shoulder_max_m']:.3f} "
                f"overhand={p['is_overhand']} loop={p['is_looping']} "
                f"score_c={p['scores']['cross']:.1f} j={p['scores']['jab']:.1f} "
                f"ref[{p['ptype']}]={((p.get('ref') or {}).get(p['ptype']) or {}).get('distance')} "
                f"faults={((p.get('ref') or {}).get(p['ptype']) or {}).get('faults')}",
                flush=True,
            )

        want_strips = [] if NO_STRIPS else [p for p in tagged if should_strip(spec, p)]
        want_strips.sort(key=lambda x: (
            0 if x.get("id") in V3_IDS.values() else 1,
            -(x.get("rise_xy_m") or 0),
        ))
        for p in want_strips[:3]:
            try:
                dest = os.path.join(STRIP_DIR, p["id"])
                render_strip(ctx, p, dest)
            except Exception as e:
                print(f"  STRIP FAIL {p['id']}: {e}", flush=True)
        for p in tagged:
            prev = os.path.join(STRIP_DIR, f"{p['id']}.png")
            if not p.get("strip") and os.path.isdir(os.path.join(STRIP_DIR, p["id"])):
                p["strip_frames_dir"] = os.path.join(STRIP_DIR, p["id"])
            if not p.get("strip") and os.path.isfile(prev):
                p["strip"] = prev
        facts["punches"].extend(tagged)

    facts["rankings"] = rank_all(facts["punches"])
    winners, notes = auto_pick(facts["rankings"])
    facts["winners"] = winners
    facts["pick_notes"] = notes
    facts["elapsed_s"] = time.time() - t0
    facts["ok"] = True
    os.makedirs(OUT, exist_ok=True)
    with open(FACTS_PATH, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("WINNERS", winners, notes, flush=True)
    for want in ("jab", "cross", "hook", "uppercut"):
        print(f"RANK {want}:", flush=True)
        for i, r in enumerate(facts["rankings"].get(want, [])[:8], 1):
            print(
                f"  {i:2d}. {r['score']:5.1f}  oh={r['is_overhand']} loop={r['is_looping']} "
                f"dev={r['fist_max_dev_m']} arc={r['fist_arc_height_m']} "
                f"elb+={r['elbow_above_shoulder_max_m']} refd={r.get('ref_distance')} "
                f"faults={r.get('ref_faults')}  {r['title']}  {r['id']}",
                flush=True,
            )
    print("FACTS", FACTS_PATH, "punches", len(facts["punches"]),
          f"in {facts['elapsed_s']:.1f}s", flush=True)
    return facts


def _find(facts, pid):
    for p in facts.get("punches", []):
        if p.get("id") == pid:
            return p
    return None


def _ik_target_on_line(orig, a, b):
    t = qc.project_t(orig, a, b)
    a = qc._v(a)
    b = qc._v(b)
    return (a + t * (b - a)).tolist()


def run_ik_test(facts, punch_id):
    """Straighten the fist along guard→full→guard line; keep torso/hips."""
    p = _find(facts, punch_id)
    if not p:
        return {"ok": False, "reason": f"missing {punch_id}"}
    spec = next(c for c in v3.CATALOG if c["id"] == p["clip_id"])
    print(f"IKTEST {punch_id}", flush=True)
    ctx = v3.load_clip(spec)
    side = p["side"]
    sl = "l" if side == "L" else "r"
    hkey = f"hand_{sl}"
    g, full, ret = int(p["guard"]), int(p["full"]), int(p["ret"])
    rows = [r for r in ctx["rows"] if g <= r["f"] <= ret and r.get(hkey)]
    if len(rows) < 3:
        return {"ok": False, "reason": "too few frames"}
    start = rows[0][hkey]
    end = next(r[hkey] for r in rows if r["f"] == full)
    # return to start (same line)
    hand_name = ctx["bones"][f"hand_{sl}"]
    # create target empty
    tgt = bpy.data.objects.new("IKFistTarget", None)
    bpy.context.scene.collection.objects.link(tgt)
    tgt.empty_display_size = 0.05
    pb = ctx["arm"].pose.bones[hand_name]
    con = pb.constraints.new("IK")
    con.target = tgt
    con.chain_count = 3
    con.use_tail = True
    chain_names = []
    bone = ctx["arm"].pose.bones[hand_name]
    for _ in range(3):
        chain_names.append(bone.name)
        if bone.parent is None:
            break
        bone = bone.parent

    bpy.ops.object.select_all(action="DESELECT")
    ctx["arm"].select_set(True)
    bpy.context.view_layer.objects.active = ctx["arm"]
    bpy.ops.object.mode_set(mode="POSE")
    for b in ctx["arm"].pose.bones:
        b.bone.select = b.name in chain_names

    for r in rows:
        f = int(r["f"])
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        orig = r[hkey]
        if f <= full:
            loc = _ik_target_on_line(orig, start, end)
        else:
            loc = _ik_target_on_line(orig, end, start)
        tgt.location = Vector(loc)
        bpy.context.view_layer.update()
        for bname in chain_names:
            b = ctx["arm"].pose.bones[bname]
            b.keyframe_insert("location", frame=f)
            if b.rotation_mode == "QUATERNION":
                b.keyframe_insert("rotation_quaternion", frame=f)
            else:
                b.keyframe_insert("rotation_euler", frame=f)
        # visual key so IK result is stored
    try:
        bpy.ops.nla.bake(
            frame_start=g, frame_end=ret, only_selected=True,
            visual_keying=True, clear_constraints=True, bake_types={"POSE"},
        )
    except Exception as e:
        print(f"  bake warning: {e}", flush=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    # clear leftover IK if bake didn't
    try:
        pb.constraints.remove(con)
    except Exception:
        pass

    new_rows = v3.analyze_rich(ctx["arm"], ctx["bones"], ctx["f0"], ctx["f1"])
    p2 = dict(p)
    attach_path_qc(p2, new_rows)
    os.makedirs(IK_DIR, exist_ok=True)
    before = p.get("strip")
    try:
        after_strip = render_strip(ctx, p2, os.path.join(IK_DIR, p["id"] + "_ik"))
    except Exception as e:
        after_strip = None
        print(f"  IK strip fail: {e}", flush=True)
    report = {
        "ok": True,
        "src": punch_id,
        "before": p.get("path_qc"),
        "after": p2.get("path_qc"),
        "before_strip": before,
        "after_strip": after_strip,
        "after_overhand": p2.get("is_overhand"),
        "after_looping": p2.get("is_looping"),
        "after_scores": p2.get("scores"),
    }
    out = os.path.join(IK_DIR, "iktest.json")
    qc.write_qc_json(out, report)
    print("IKTEST", json.dumps({k: report[k] for k in report if k not in ("before", "after")},
                               default=str), flush=True)
    print("  before arc/elb+/dev",
          (p.get("path_qc") or {}).get("fist_arc_height_m"),
          (p.get("path_qc") or {}).get("elbow_above_shoulder_max_m"),
          (p.get("path_qc") or {}).get("fist_max_dev_m"), flush=True)
    print("  after  arc/elb+/dev",
          (p2.get("path_qc") or {}).get("fist_arc_height_m"),
          (p2.get("path_qc") or {}).get("elbow_above_shoulder_max_m"),
          (p2.get("path_qc") or {}).get("fist_max_dev_m"), flush=True)
    return report


def phase_iktest(facts=None, punch_id=None):
    if facts is None:
        facts = json.load(open(FACTS_PATH))
    pid = punch_id or facts.get("winners", {}).get("cross")
    report = run_ik_test(facts, pid)
    facts["iktest"] = report
    with open(FACTS_PATH, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    return report


def phase_render(facts=None, winners_override=None):
    if facts is None:
        facts = json.load(open(FACTS_PATH))
    if winners_override:
        facts["winners"].update(winners_override)
    winners = facts.get("winners") or {}
    os.makedirs(QC_DIR, exist_ok=True)
    t0 = time.time()
    rendered = {}
    for ptype, pid in winners.items():
        p = _find(facts, pid)
        if not p:
            print(f"missing winner {ptype} {pid}", flush=True)
            continue
        spec = next(c for c in v3.CATALOG if c["id"] == p["clip_id"])
        print(f"RENDER {ptype} {p['clip_title']} {pid}", flush=True)
        ctx = v3.load_clip(spec)
        xb.assign_clay(ctx["meshes"])
        frames = [p["guard"], p["mid"], p["full"]]
        v3.setup_fitted_camera(ctx["meshes"], frames + [p.get("ret") or p["full"]],
                               ctx["arm"], ctx["bones"], ctx["start_xy"], ctx["lock"])
        stills = {}
        for stage in ("guard", "mid", "full"):
            f = int(p[stage])
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            if ctx["lock"]:
                xb.lock_root_xy(ctx["arm"], ctx["bones"], ctx["start_xy"])
            xb.paint_glow(ctx["meshes"], ctx["arm"], ctx["bones"],
                          p["side"], v3.MUSCLES.get(ptype) or ())
            out = os.path.join(OUT, f"xbot_v4_{ptype}_{stage}.png")
            xb.render_png(out)
            stills[stage] = out
            extra = {
                "foot_slide_m": max(p.get("foot_slide_l_m") or 0,
                                    p.get("foot_slide_r_m") or 0),
                "root_drift_m": p.get("root_drift_m"),
            }
            report = qc.evaluate_render(
                out, path_qc=p.get("path_qc"), punch_type=ptype, extra=extra,
            )
            report["punch_id"] = pid
            report["stage"] = stage
            report["reference"] = (p.get("ref") or {}).get(ptype)
            report["form_features"] = p.get("form_features")
            qpath = os.path.join(QC_DIR, f"xbot_v4_{ptype}_{stage}.json")
            qc.write_qc_json(qpath, report)
            print(f"  WROTE {out} qc_pass={report.get('all_pass')} "
                  f"overhand={report.get('is_overhand')}", flush=True)
        rendered[ptype] = {"stills": stills, "punch": p}
    facts["winner_stills"] = {k: v["stills"] for k, v in rendered.items()}
    facts["preview"] = render_preview(facts, winners)
    facts["render_elapsed_s"] = time.time() - t0
    with open(FACTS_PATH, "w") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print("RENDER done", facts.get("preview"), flush=True)
    return facts


def render_preview(facts, winners):
    seq_dir = os.path.join(OUT, "_xbot_v4_seq")
    os.makedirs(seq_dir, exist_ok=True)
    written = []
    t0 = time.time()
    fps = 15
    for ptype in ("jab", "cross", "hook", "uppercut"):
        p = _find(facts, winners.get(ptype))
        if not p:
            continue
        if time.time() - t0 > 900:
            break
        spec = next(c for c in v3.CATALOG if c["id"] == p["clip_id"])
        ctx = v3.load_clip(spec)
        xb.assign_clay(ctx["meshes"])
        raw = list(range(int(p["guard"]), int(p["full"]) + 2))
        v3.setup_fitted_camera(ctx["meshes"], [p["guard"], p["full"]],
                               ctx["arm"], ctx["bones"], ctx["start_xy"], ctx["lock"])
        step = max(1, int(math.ceil(len(raw) / 10)))
        pick = raw[::step][:10]
        if pick[-1] != raw[-1]:
            pick.append(raw[-1])
        for f in pick:
            bpy.context.scene.frame_set(int(f))
            bpy.context.view_layer.update()
            if ctx["lock"]:
                xb.lock_root_xy(ctx["arm"], ctx["bones"], ctx["start_xy"])
            xb.paint_glow(ctx["meshes"], ctx["arm"], ctx["bones"],
                          p["side"], v3.MUSCLES.get(ptype) or ())
            fp = os.path.join(seq_dir, f"f_{len(written):04d}.png")
            v3.render_eevee(fp, res=(360, 640), samples=8)
            written.append(fp)
        if written:
            hold = written[-1]
            for _ in range(2):
                fp = os.path.join(seq_dir, f"f_{len(written):04d}.png")
                shutil.copy2(hold, fp)
                written.append(fp)
    if not written:
        return {"skipped": True, "reason": "no frames"}
    mp4 = os.path.join(OUT, "xbot_boxing_picks_v4_preview.mp4")
    import subprocess
    n = len(written)
    duration = n / float(fps)
    fps_use = fps
    if duration < 2.0:
        fps_use = max(8, int(round(n / 2.2)))
    elif duration > 4.0:
        fps_use = int(math.ceil(n / 3.8))
    pattern = os.path.join(seq_dir, "f_%04d.png")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-framerate", str(fps_use), "-i", pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4]
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        return {"skipped": True, "reason": str(e), "n": n}
    return {"skipped": False, "path": mp4, "bytes": os.path.getsize(mp4),
            "n_src": n, "fps": fps_use, "duration_s": n / float(fps_use),
            "elapsed_s": time.time() - t0}


def main():
    phase, winners = _cli()
    if phase == "analyze":
        phase_analyze()
    elif phase == "iktest":
        phase_iktest()
    elif phase == "preview":
        facts = json.load(open(FACTS_PATH))
        facts["preview"] = render_preview(facts, facts.get("winners") or {})
        with open(FACTS_PATH, "w") as fh:
            json.dump(facts, fh, indent=2, default=str)
        print("PREVIEW", facts["preview"], flush=True)
    elif phase == "strips":  # frame-by-frame strips for the current winners
        facts = json.load(open(FACTS_PATH))
        for ptype, pid in (facts.get("winners") or {}).items():
            p = _find(facts, pid)
            if not p:
                continue
            spec = next(c for c in v3.CATALOG if c["id"] == p["clip_id"])
            ctx = v3.load_clip(spec)
            render_strip(ctx, p, os.path.join(STRIP_DIR, p["id"]))
    elif phase == "render":
        phase_render(winners_override=winners or None)
    elif phase == "all":
        facts = phase_analyze()
        # If the auto-picked cross still loops, run the IK straighten test.
        cr = None
        for p in facts["punches"]:
            if p["id"] == facts.get("winners", {}).get("cross"):
                cr = p
                break
        if cr and (cr.get("is_overhand") or cr.get("is_looping")):
            print("cross failed path QC — running IK straighten test", flush=True)
            phase_iktest(facts, cr["id"])
            facts = json.load(open(FACTS_PATH))
        if winners:
            facts["winners"].update(winners)
        phase_render(facts)
    else:
        raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    main()
