#!/usr/bin/env python3
"""X Bot painted outfit v2 — units unified in world metres. One attempt.

Root-cause fix for v1 (bone-local cm mixed with world-metre GLBs). All
placement is in world metres from normalized bone world axes. Sane sizes
are asserted before any render (glove length 0.25–0.35 m, beanie width
0.18–0.24 m, head width from the X Bot mesh in world metres).

Pass criteria (written before any render; a miss is a fail, do not iterate):
  (1) Gloves enclose the fist on every frame of all 4 v4 punches,
      with no floating and no forearm placement.
  (2) The beanie stays on the crown on every frame.
  (3) Paint zones are clean and symmetric, with no tearing
      (by construction: body-surface attributes, no garment meshes).
  (4) boxing_qc motion numbers are identical to bare X Bot.

  nice -n 19 blender -b --python tools/rig3d/build_xbot_painted_v2.py
  python3 tools/rig3d/build_xbot_painted_v2.py --compose
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "tools/rig3d")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

EVAL = os.path.join(ROOT, "output/_basemodels/eval")
SRC_DIR = os.path.join(ROOT, "output/_basemodels/boxer/mixamo_xbot")
FACTS_PATH = os.path.join(EVAL, "xbot_painted_v2_facts.json")
CRITERIA_PATH = os.path.join(EVAL, "xbot_painted_v2_criteria.json")
SHEET_PATH = os.path.join(EVAL, "xbot_painted_v2.png")
PREVIEW_PATH = os.path.join(EVAL, "xbot_painted_v2_jab_preview.mp4")

# v4 winners (jab alt04, cross alt18, hook alt06, uppercut alt05).
PICKS = {
    "jab": {
        "file": "alts/alt04_Jab Cross_BoxingJabCrossMedium.fbx",
        "side": "L", "guard": 9, "mid": 11, "full": 14, "ret": 22,
        "muscles": ("front_delt", "triceps"),
        "title": "JAB  ·  alt04",
        "stages": ("guard", "full", "ret"),
    },
    "cross": {
        "file": "alts/alt18_Punching_CrossPunch.fbx",
        "side": "R", "guard": 4, "mid": 8, "full": 12, "ret": 24,
        "muscles": ("chest", "front_delt", "obliques", "calf", "hip"),
        "title": "CROSS  ·  alt18",
        "stages": ("full",),
    },
    "hook": {
        "file": "alts/alt06_Jab Cross_ShortJabCross.fbx",
        "side": "L", "guard": 21, "mid": 24, "full": 28, "ret": 50,
        "muscles": ("front_delt", "obliques"),
        "title": "HOOK  ·  alt06",
        "stages": ("full",),
    },
    "uppercut": {
        "file": "alts/alt05_Body Jab Cross_BoxingBodyJabCrossMedium.fbx",
        "side": "R", "guard": 21, "mid": 32, "full": 43, "ret": 52,
        "muscles": ("biceps", "front_delt", "quads", "glutes"),
        "title": "UPPERCUT  ·  alt05",
        "stages": ("full",),
    },
}

QC_KEYS = (
    "fist_max_dev_m", "fist_arc_height_m", "fist_arc_depth_m",
    "retract_max_dev_m", "start_to_chin_m", "chin_line_dev_m",
    "elbow_above_shoulder_max_m", "elbow_at_impact_vs_shoulder_m",
    "is_overhand", "is_looping",
)

PANEL = (540, 960)
CLOSE = (540, 720)


def write_criteria():
    os.makedirs(EVAL, exist_ok=True)
    import xbot_painted_outfit_v2 as painted
    payload = {
        **painted.CRITERIA,
        "written_before_render": True,
        "picks": {k: {kk: vv for kk, vv in v.items() if kk != "muscles"}
                  for k, v in PICKS.items()},
        "note": (
            "This file is written at the start of the run, before any outfit "
            "is applied or any frame is rendered. A missed criterion is a "
            "fail. Do not iterate."
        ),
    }
    with open(CRITERIA_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print("CRITERIA written", CRITERIA_PATH, flush=True)
    return payload


def _path_qc_compact(pq):
    if not pq:
        return {}
    out = {}
    for k in QC_KEYS:
        if k in pq:
            v = pq[k]
            if isinstance(v, float):
                out[k] = round(v, 6)
            else:
                out[k] = v
    return out


def compare_qc(bare, dressed, tol=1e-9):
    """Criterion 4: dressed path metrics must match bare X Bot."""
    report = {"ok": True, "per_punch": {}}
    for key in PICKS:
        b = (bare.get(key) or {}).get("path_qc") or {}
        d = (dressed.get(key) or {}).get("path_qc") or {}
        diffs = {}
        for k in QC_KEYS:
            if k not in b and k not in d:
                continue
            bv, dv = b.get(k), d.get(k)
            if isinstance(bv, float) or isinstance(dv, float):
                if abs(float(bv or 0) - float(dv or 0)) > tol:
                    diffs[k] = {"bare": bv, "dressed": dv,
                                "delta": float(dv or 0) - float(bv or 0)}
            elif bv != dv:
                diffs[k] = {"bare": bv, "dressed": dv}
        report["per_punch"][key] = {"ok": not diffs, "diffs": diffs,
                                    "bare": _path_qc_compact(b),
                                    "dressed": _path_qc_compact(d)}
        if diffs:
            report["ok"] = False
    return report


def compose(facts_path=FACTS_PATH, out_path=SHEET_PATH):
    from PIL import Image, ImageDraw, ImageFont

    facts = json.load(open(facts_path))
    stills = facts.get("stills") or {}
    BG, FG, MUTED = (16, 16, 18), (232, 232, 234), (158, 158, 164)
    OK, BAD, ACCENT = (120, 210, 150), (220, 110, 96), (220, 200, 140)
    PAD, TITLE_H, LABEL_H, FOOT_H = 14, 120, 44, 160

    def font(size, bold=False):
        path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def load(path, size):
        if path and os.path.isfile(path):
            im = Image.open(path).convert("RGB")
            if im.size != size:
                im = im.resize(size, Image.Resampling.LANCZOS)
            return im
        im = Image.new("RGB", size, (40, 28, 28))
        ImageDraw.Draw(im).text((24, 40), f"missing\n{path}", font=font(18, True),
                                fill=BAD)
        return im

    def wrap(dr, text, fn, width):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if dr.textlength(trial, font=fn) <= width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    rows = (
        (
            ("jab_guard", "JAB  ·  guard"),
            ("jab_full", "JAB  ·  extension"),
            ("jab_ret", "JAB  ·  return"),
        ),
        (
            ("cross_full", "CROSS  ·  extension"),
            ("hook_full", "HOOK  ·  extension"),
            ("uppercut_full", "UPPERCUT  ·  extension"),
        ),
    )
    close_row = (
        ("close_glove", "glove close  ·  rigid on the hand bone"),
        ("close_beanie", "beanie close  ·  crown, hem above brows"),
        ("close_paint", "paint edges  ·  waistband, armhole, mid-thigh hem"),
    )
    cols = 3
    w = PAD * (cols + 1) + PANEL[0] * cols
    h = (TITLE_H
         + (LABEL_H + PANEL[1]) * len(rows)
         + (LABEL_H + CLOSE[1])
         + FOOT_H + PAD)
    sheet = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 8, 12), "X Bot  ·  painted outfit v2", font=font(28, True), fill=FG)
    sub = ("Rigid ARTEL_3D gloves (glossy black, CC BY) and oluwaphemi rib beanie "
           "(charcoal, CC BY). Trunks and tank are paint on the X Bot surface — "
           "burgundy (0.18, 0.025, 0.04) with a charcoal waistband, charcoal sleeveless "
           "tank, deltoids bare. Units unified in world metres. One attempt, no patching. v2 lighting.")
    for i, line in enumerate(wrap(dr, sub, font(13), w - 2 * PAD - 16)[:4]):
        dr.text((PAD + 8, 48 + i * 16), line, font=font(13), fill=MUTED)

    def paste_row(items, y0, size):
        for c, (key, label) in enumerate(items):
            x = PAD + c * (size[0] + PAD)
            dr.text((x + 8, y0 + 10), label, font=font(15, True), fill=ACCENT)
            sheet.paste(load(stills.get(key), size), (x, y0 + LABEL_H))

    y = TITLE_H
    for items in rows:
        paste_row(items, y, PANEL)
        y += LABEL_H + PANEL[1]
    paste_row(close_row, y, CLOSE)
    y += LABEL_H + CLOSE[1] + 12

    verdicts = facts.get("verdicts") or {}
    bits = []
    for key, label in (
        ("1_gloves", "1 gloves"),
        ("2_beanie", "2 beanie"),
        ("3_paint", "3 paint (by construction)"),
        ("4_motion", "4 boxing_qc vs bare"),
    ):
        rec = verdicts.get(key) or {}
        ok = rec.get("ok")
        tag = "PASS" if ok else "FAIL"
        bits.append(f"{label} {tag}")
    overall = all((verdicts.get(k) or {}).get("ok") for k in
                  ("1_gloves", "2_beanie", "3_paint", "4_motion"))
    head = "OVERALL PASS" if overall else "OVERALL FAIL"
    dr.text((PAD + 8, y), head + "   ·   " + "   |   ".join(bits),
            font=font(16, True), fill=OK if overall else BAD)
    foot = facts.get("footer") or (
        "Credits: Classic Red Boxing Glove (Low Poly) by ARTEL_3D, CC BY 4.0.  "
        "Beanie by oluwaphemi, CC BY 4.0."
    )
    for i, line in enumerate(wrap(dr, foot, font(13), w - 2 * PAD - 16)[:5]):
        dr.text((PAD + 8, y + 28 + i * 18), line, font=font(13), fill=MUTED)
    sheet.save(out_path)
    print("SHEET", out_path, os.path.getsize(out_path), "bytes", sheet.size, flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Blender path
# ---------------------------------------------------------------------------
def run_blender():
    import bpy
    from mathutils import Vector

    from eval_base_model import _reset  # noqa: F401
    import eval_xbot_boxing as xb
    import pick_xbot_punches_v3 as v3
    import boxing_qc as qc
    import xbot_painted_outfit_v2 as painted

    os.makedirs(EVAL, exist_ok=True)
    criteria = write_criteria()
    t_all = time.time()
    facts = {
        "criteria": criteria,
        "stills": {},
        "qc_bare": {},
        "qc_dressed": {},
        "checks": {},
        "clips": {},
        "verdicts": {},
    }

    def measure_qc(rows, pick):
        punch = {"side": pick["side"], "guard": pick["guard"],
                 "full": pick["full"], "ret": pick["ret"]}
        pq = qc.punch_path_metrics(rows, punch, side=pick["side"])
        compact = {k: v for k, v in pq.items() if k != "traj"}
        return compact

    def still_path(key):
        p = os.path.join(EVAL, f"xbot_painted_v2_{key}.png")
        return p

    def close_camera(look, direction, fill_h, res=CLOSE):
        """Tight 50 mm camera, vertical fit, figure-relative fill_h metres."""
        for o in list(bpy.data.objects):
            if o.type in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(o, do_unlink=True)
        look = Vector(look)
        direction = Vector(direction).normalized()
        cam_data = bpy.data.cameras.new("Cam")
        cam_data.lens = 50
        cam_data.sensor_fit = "VERTICAL"
        cam_data.sensor_height = 24.0
        cam_data.clip_start = 0.02
        cam_data.clip_end = 50
        v_ang = 2 * math.atan((cam_data.sensor_height * 0.5) / cam_data.lens)
        dist = (fill_h * 0.5) / math.tan(v_ang * 0.5)
        loc = look + direction * dist
        cam = bpy.data.objects.new("Cam", cam_data)
        bpy.context.scene.collection.objects.link(cam)
        cam.location = loc
        cam.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.camera = cam
        xb.setup_studio(cam, look)
        return cam

    def render_still(path, res=PANEL, samples=None):
        sc = bpy.context.scene
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = samples if samples is not None else xb.SAMPLES
        sc.cycles.use_denoising = True
        try:
            sc.cycles.denoiser = "OPENIMAGEDENOISE"
        except Exception:
            pass
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGB"
        sc.render.image_settings.color_depth = "8"
        sc.render.filepath = path
        xb.apply_color_mgmt(sc)
        sc.cycles.filter_width = 1.5
        bpy.ops.render.render(write_still=True)
        print(f"WROTE {path} ({os.path.getsize(path)} bytes)", flush=True)
        return path

    glove_check_all = {"ok": True, "per_clip": {}}
    beanie_check_all = {"ok": True, "per_clip": {}}
    jab_state = None

    for key, pick in PICKS.items():
        path = os.path.join(SRC_DIR, pick["file"])
        print(f"\n==== CLIP {key} {pick['file']} ====", flush=True)
        meshes, arm = xb.import_clip(path)
        bones = v3.enrich_bones(arm, xb.bone_map(arm))
        f0, f1 = xb.frame_range(arm)
        bpy.context.scene.frame_set(f0)
        bpy.context.view_layer.update()
        scale = xb.scale_ground_center(arm, meshes, bones)
        xb.face_minus_y(arm, bones)
        bpy.context.scene.frame_set(f0)
        bpy.context.view_layer.update()
        start_xy = xb.hips_xy(arm, bones)
        print(f"  scale={scale} frames={f0}-{f1} height={scale.get('height_m')}",
              flush=True)

        rows_bare = v3.analyze_rich(arm, bones, f0, f1)
        qc_bare = measure_qc(rows_bare, pick)
        facts["qc_bare"][key] = {"path_qc": qc_bare, "frames": [f0, f1],
                                 "scale": scale}

        handle = painted.apply_painted_outfit(arm, meshes, f0, f1)
        print("  outfit",
              {s: handle["gloves"][s]["scale"] for s in ("L", "R")},
              "beanie_scale", handle["beanie"]["scale"],
              "glove_len", {s: handle["gloves"][s].get("length_m") for s in ("L", "R")},
              "beanie_w", handle["beanie"].get("width_m"),
              "head_w", handle.get("head_width_m"),
              "zones", {k: handle["zones"].get(k) for k in
                        ("tank", "trunks", "band", "skin")},
              flush=True)
        size_assert = handle.get("size_assert") or painted.assert_sane_sizes(handle)
        facts.setdefault("size_asserts", {})[key] = size_assert
        if not size_assert.get("ok"):
            print("  SIZE ASSERT FAILED — no render for this clip:",
                  size_assert.get("failures"), flush=True)
            facts["clips"][key] = {
                "file": pick["file"], "frames": [f0, f1],
                "size_assert": size_assert,
                "handle_glove_scale": {
                    s: handle["gloves"][s]["scale"] for s in ("L", "R")},
                "glove_length_m": {
                    s: handle["gloves"][s].get("length_m") for s in ("L", "R")},
                "beanie": {k: handle["beanie"][k] for k in
                           ("scale", "width_m", "hem", "crown", "hem_r", "height")
                           if k in handle["beanie"]},
                "head_width_m": handle.get("head_width_m"),
                "zones": {k: handle["zones"].get(k) for k in
                          ("tank", "trunks", "band", "skin", "landmarks")},
                "skipped_render": True,
            }
            # Still run motion QC and containment checks — they do not need a PNG.
            rows_dressed = v3.analyze_rich(arm, bones, f0, f1)
            qc_dressed = measure_qc(rows_dressed, pick)
            facts["qc_dressed"][key] = {"path_qc": qc_dressed}
            gchk = painted.check_gloves(arm, meshes, handle, f0, f1, step=1)
            bchk = painted.check_beanie(arm, meshes, handle, f0, f1, step=1)
            glove_check_all["per_clip"][key] = gchk
            beanie_check_all["per_clip"][key] = bchk
            if not gchk.get("ok"):
                glove_check_all["ok"] = False
            if not bchk.get("ok"):
                beanie_check_all["ok"] = False
            continue

        rows_dressed = v3.analyze_rich(arm, bones, f0, f1)
        qc_dressed = measure_qc(rows_dressed, pick)
        facts["qc_dressed"][key] = {"path_qc": qc_dressed}

        gchk = painted.check_gloves(arm, meshes, handle, f0, f1, step=1)
        bchk = painted.check_beanie(arm, meshes, handle, f0, f1, step=1)
        glove_check_all["per_clip"][key] = gchk
        beanie_check_all["per_clip"][key] = bchk
        if not gchk.get("ok"):
            glove_check_all["ok"] = False
        if not bchk.get("ok"):
            beanie_check_all["ok"] = False
        print(f"  glove_check ok={gchk.get('ok')} beanie_check ok={bchk.get('ok')}",
              flush=True)

        # Root lock only for rendering, after QC.
        xs = [r["hips"][0] for r in rows_bare if r.get("hips")]
        ys = [r["hips"][1] for r in rows_bare if r.get("hips")]
        drift = float(math.hypot(max(xs) - min(xs), max(ys) - min(ys))) if xs else 0.0
        lock = drift > xb.ROOT_DRIFT_LOCK_M

        # Camera from the body at guard (stable framing for the clip).
        bpy.context.scene.frame_set(int(pick["guard"]))
        bpy.context.view_layer.update()
        if lock:
            xb.lock_root_xy(arm, bones, start_xy)
        xb.setup_fixed_camera(meshes)

        for stage in pick["stages"]:
            f = int(pick[stage])
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            if lock:
                xb.lock_root_xy(arm, bones, start_xy)
            if pick["muscles"]:
                xb.paint_glow(meshes, arm, bones, pick["side"], pick["muscles"])
            else:
                xb.clear_glow(meshes)
            out = still_path(f"{key}_{stage}")
            render_still(out, res=PANEL)
            facts["stills"][f"{key}_{stage}"] = out

        facts["clips"][key] = {
            "file": pick["file"], "frames": [f0, f1], "drift_m": drift,
            "lock": lock, "handle_glove_scale": {
                s: handle["gloves"][s]["scale"] for s in ("L", "R")},
            "glove_length_m": {
                s: handle["gloves"][s].get("length_m") for s in ("L", "R")},
            "beanie": {k: handle["beanie"][k] for k in
                       ("scale", "width_m", "hem", "crown", "hem_r", "height")
                       if k in handle["beanie"]},
            "head_width_m": handle.get("head_width_m"),
            "size_assert": size_assert,
            "zones": {k: handle["zones"].get(k) for k in
                      ("tank", "trunks", "band", "skin", "landmarks")},
        }

        if key == "jab":
            # Close-ups on this clip, then keep enough state for the preview.
            jab_state = {
                "file": path, "f0": f0, "f1": f1, "pick": pick,
                "lock": lock, "start_xy": list(start_xy),
            }
            # Glove close: punching-side glove at full extension.
            bpy.context.scene.frame_set(int(pick["full"]))
            bpy.context.view_layer.update()
            if lock:
                xb.lock_root_xy(arm, bones, start_xy)
            xb.paint_glow(meshes, arm, bones, pick["side"], pick["muscles"])
            gname = handle["gloves"][pick["side"]]["object"]
            glove = bpy.data.objects[gname]
            gpts = painted.world_pts(glove)
            glook = sum(gpts, Vector()) / len(gpts)
            close_camera(glook, Vector((0.55, -0.80, 0.22)), fill_h=0.22, res=CLOSE)
            out = still_path("close_glove")
            render_still(out, res=CLOSE)
            facts["stills"]["close_glove"] = out

            # Beanie close at guard (head is easier to read before the punch).
            bpy.context.scene.frame_set(int(pick["guard"]))
            bpy.context.view_layer.update()
            if lock:
                xb.lock_root_xy(arm, bones, start_xy)
            hat = bpy.data.objects[handle["beanie"]["object"]]
            hinfo = painted.beanie_anatomy(hat)
            hlook = (Vector(hinfo["hem"]) + Vector(hinfo["crown"])) * 0.5
            close_camera(hlook, Vector((0.40, -0.85, 0.18)), fill_h=0.28, res=CLOSE)
            out = still_path("close_beanie")
            render_still(out, res=CLOSE)
            facts["stills"]["close_beanie"] = out

            # Paint-edge close: waistband + one armhole, 3/4, at guard.
            land = handle["zones"].get("landmarks") or {}
            mid_x = land.get("mid_x", 0.05)
            band_z = 0.5 * (land.get("band_bot_z", 1.05) + land.get("band_top_z", 1.09))
            armhole_z = 0.5 * (land.get("band_top_z", 1.09) + land.get("tank_top_z", 1.50))
            look = Vector((mid_x + 0.04, -0.02, 0.55 * band_z + 0.45 * armhole_z))
            close_camera(look, Vector((0.65, -0.75, 0.08)), fill_h=0.55, res=CLOSE)
            out = still_path("close_paint")
            render_still(out, res=CLOSE)
            facts["stills"]["close_paint"] = out

    facts["checks"]["gloves"] = glove_check_all
    facts["checks"]["beanie"] = beanie_check_all
    facts["checks"]["motion"] = compare_qc(facts["qc_bare"], facts["qc_dressed"])
    # Paint is by construction (no garment mesh to tear). Still require that
    # the zone counts are non-zero and both sides of the body were painted.
    zones_ok = True
    zone_notes = []
    for key, rec in facts["clips"].items():
        z = rec.get("zones") or {}
        if (z.get("tank") or 0) < 80 or (z.get("trunks") or 0) < 80 or (z.get("band") or 0) < 20:
            zones_ok = False
            zone_notes.append(f"{key}: thin zones {z}")
    facts["checks"]["paint"] = {
        "ok": zones_ok,
        "by_construction": (
            "No garment meshes; zones are rest-pose vertex attributes "
            "that deform with the skin. Symmetry comes from Mixamo "
            "weights and Z thresholds, not from a left/right bake."
        ),
        "notes": zone_notes,
    }

    facts["verdicts"] = {
        "1_gloves": {"ok": bool(glove_check_all.get("ok")),
                     "detail": {k: {"ok": v.get("ok"),
                                    "n_failures": (v.get("per_side") or {}).get("L", {}).get("n_failures"),
                                    "R_n_failures": (v.get("per_side") or {}).get("R", {}).get("n_failures"),
                                    "worst": v.get("worst")}
                                for k, v in (glove_check_all.get("per_clip") or {}).items()}},
        "2_beanie": {"ok": bool(beanie_check_all.get("ok")),
                     "detail": {k: {"ok": v.get("ok"), "n_failures": v.get("n_failures"),
                                    "failures": v.get("failures")}
                                for k, v in (beanie_check_all.get("per_clip") or {}).items()}},
        "3_paint": {"ok": bool(zones_ok), "notes": zone_notes},
        "4_motion": {"ok": bool(facts["checks"]["motion"]["ok"]),
                     "detail": facts["checks"]["motion"]["per_punch"]},
    }
    size_ok = all(v.get("ok") for v in (facts.get("size_asserts") or {}).values())
    if facts.get("size_asserts") and not size_ok:
        # A failed pre-render assert is a fail of the attempt, even if a
        # later criterion was not computed for a skipped clip.
        pass
    overall = all(v.get("ok") for v in facts["verdicts"].values()) and size_ok
    facts["overall"] = overall
    facts["size_assert_ok"] = size_ok
    facts["footer"] = (
        "Credits: Classic Red Boxing Glove (Low Poly) "
        "(https://sketchfab.com/3d-models/classic-red-boxing-glove-low-poly-7faefb99c91d484995899b0e4aa6921c) "
        "by ARTEL_3D, CC BY 4.0.  Beanie "
        "(https://sketchfab.com/3d-models/beanie-ff6dc83c818c4e2aa722cb89e6ada577) "
        "by oluwaphemi, CC BY 4.0.  Trunks and tank are paint, not third-party meshes."
    )

    # Short jab preview (EEVEE), using the same alt04 clip. Skip if the
    # jab size assert failed — we must not render an unsized asset.
    jab_assert = (facts.get("size_asserts") or {}).get("jab") or {}
    if jab_state is None or not jab_assert.get("ok", True):
        facts["preview"] = {"skipped": True,
                            "reason": "size assert failed or no jab state",
                            "size_assert": jab_assert}
    else:
        try:
            facts["preview"] = render_jab_preview(xb, painted, v3, jab_state)
        except Exception as e:
            facts["preview"] = {"skipped": True, "reason": repr(e)}
            print("PREVIEW failed", e, flush=True)

    facts["elapsed_s"] = time.time() - t_all
    with open(FACTS_PATH, "w") as f:
        json.dump(facts, f, indent=2, default=str)
    print("FACTS", FACTS_PATH, flush=True)

    # Compose the contact sheet with system Python (PIL).
    cmd = [sys.executable]
    # Blender's Python often has no PIL. Prefer the box system interpreter.
    py = "/usr/bin/python3" if os.path.isfile("/usr/bin/python3") else sys.executable
    try:
        subprocess.check_call(
            [py, os.path.join(HERE, "build_xbot_painted_v2.py"), "--compose",
             "--facts", FACTS_PATH, "--out", SHEET_PATH]
        )
    except Exception as e:
        print("compose via subprocess failed, trying in-process:", e, flush=True)
        try:
            compose(FACTS_PATH, SHEET_PATH)
        except Exception as e2:
            print("compose failed", e2, flush=True)
            facts["sheet_error"] = repr(e2)
            with open(FACTS_PATH, "w") as f:
                json.dump(facts, f, indent=2, default=str)

    print("OVERALL", "PASS" if overall else "FAIL",
          {k: v.get("ok") for k, v in facts["verdicts"].items()},
          f"elapsed={facts['elapsed_s']:.1f}s", flush=True)
    return facts


def render_jab_preview(xb, painted, v3, jab_state):
    """Low-res EEVEE loop of the dressed jab (tiny preview, not a final reel)."""
    import bpy
    from mathutils import Vector
    if not jab_state:
        return {"skipped": True, "reason": "no jab state"}
    t0 = time.time()
    pick = jab_state["pick"]
    meshes, arm = xb.import_clip(jab_state["file"])
    bones = v3.enrich_bones(arm, xb.bone_map(arm))
    f0, f1 = xb.frame_range(arm)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    xb.scale_ground_center(arm, meshes, bones)
    xb.face_minus_y(arm, bones)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    start_xy = Vector(jab_state["start_xy"])
    painted.apply_painted_outfit(arm, meshes, f0, f1)
    lock = jab_state["lock"]
    bpy.context.scene.frame_set(int(pick["guard"]))
    bpy.context.view_layer.update()
    if lock:
        xb.lock_root_xy(arm, bones, start_xy)
    xb.setup_fixed_camera(meshes)

    a, b = int(pick["guard"]), int(pick["ret"])
    frames = list(range(a, b + 1))
    fps = 15
    if len(frames) > 40:
        step = max(1, int(math.ceil(len(frames) / 30)))
        frames = frames[::step]

    seq_dir = os.path.join(EVAL, "_xbot_painted_v2_jab_seq")
    os.makedirs(seq_dir, exist_ok=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.render.resolution_x, sc.render.resolution_y = 360, 640
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    xb.apply_color_mgmt(sc)
    if hasattr(sc, "eevee"):
        try:
            sc.eevee.taa_render_samples = 8
        except Exception:
            pass

    written = []
    for i, f in enumerate(frames):
        if time.time() - t0 > 150 and i > 4:
            return {"skipped": True, "reason": "pacing would exceed 3 min",
                    "elapsed_s": time.time() - t0, "done": i}
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        if lock:
            xb.lock_root_xy(arm, bones, start_xy)
        xb.paint_glow(meshes, arm, bones, pick["side"], pick["muscles"])
        fp = os.path.join(seq_dir, f"f_{i:04d}.png")
        sc.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        written.append(fp)
    if not written:
        return {"skipped": True, "reason": "no frames"}
    pattern = os.path.join(seq_dir, "f_%04d.png")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-i", pattern,
        "-vf", "loop=loop=1:size=" + str(len(written)) + ":start=0",
        "-frames:v", str(max(len(written), int(2 * fps))),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        PREVIEW_PATH,
    ]
    try:
        subprocess.check_call(cmd)
    except Exception:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps), "-i", pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            PREVIEW_PATH,
        ]
        subprocess.check_call(cmd)
    return {
        "skipped": False, "path": PREVIEW_PATH,
        "bytes": os.path.getsize(PREVIEW_PATH) if os.path.isfile(PREVIEW_PATH) else 0,
        "n_src": len(written), "fps": fps,
        "elapsed_s": time.time() - t0,
    }


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if "--compose" in argv:
        facts = FACTS_PATH
        out = SHEET_PATH
        if "--facts" in argv:
            facts = argv[argv.index("--facts") + 1]
        if "--out" in argv:
            out = argv[argv.index("--out") + 1]
        compose(facts, out)
        return
    run_blender()


if __name__ == "__main__":
    main()
