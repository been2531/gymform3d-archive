#!/usr/bin/env python3
"""Compose xbot_boxing_picks_v4.png with winner rows, path traces, and strips."""
from __future__ import annotations

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import boxing_qc as qc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
EVAL = os.path.join(ROOT, "output/_basemodels/eval")
FACTS = os.path.join(EVAL, "xbot_punch_pick_facts_v4.json")
OUT = os.path.join(EVAL, "xbot_boxing_picks_v4.png")

PANEL = (500, 888)
BG = (16, 16, 18)
FG = (232, 232, 234)
MUTED = (156, 156, 162)
ACCENT = (220, 200, 140)
OK = (120, 210, 150)
BAD = (220, 110, 96)
PAD = 10
TITLE_H = 100
LABEL_H = 86
TRACE_H = 200
STRIP_H = 290
FOOT_H = 110
ORDER = ("jab", "cross", "hook", "uppercut")
STAGE = ("guard", "mid", "full")
TYPE_LABEL = {
    "jab": "JAB",
    "cross": "CROSS",
    "hook": "LEAD HOOK",
    "uppercut": "UPPERCUT",
}


def _font(size, bold=False):
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit(im, size, bg=BG):
    canvas = Image.new("RGB", size, bg)
    im = im.convert("RGB")
    im = im.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(im, ((size[0] - im.size[0]) // 2, (size[1] - im.size[1]) // 2))
    return canvas


def load(path, size):
    if path and os.path.isfile(path):
        return _fit(Image.open(path), size)
    im = Image.new("RGB", size, (40, 28, 28))
    ImageDraw.Draw(im).text((16, 24), "missing", font=_font(16, True), fill=BAD)
    return im


def punch_by_id(facts, pid):
    for p in facts.get("punches", []):
        if p.get("id") == pid:
            return p
    return {}


def path_line(p):
    q = p.get("path_qc") or {}
    flags = []
    if p.get("is_overhand"):
        flags.append("OVERHAND")
    if p.get("is_looping"):
        flags.append("LOOP")
    flag = ("  [" + ", ".join(flags) + "]") if flags else ""
    return (
        f"path dev {q.get('fist_max_dev_m', 0)*100:.0f} cm   "
        f"arc {q.get('fist_arc_height_m', 0)*100:+.0f} cm   "
        f"elbow-above {q.get('elbow_above_shoulder_max_m', 0)*100:+.0f} cm   "
        f"retract {q.get('retract_max_dev_m', 0)*100:.0f} cm   "
        f"start→chin {((q.get('start_to_chin_m') or 0)*100):.0f} cm"
        f"{flag}"
    )


def ref_line(p, ptype):
    r = (p.get("ref") or {}).get(ptype) or {}
    if not r.get("available"):
        return "mocap reference: n/a"
    faults = ", ".join(r.get("faults") or []) or "none"
    worst = ", ".join(f"{k} {v:+.1f}σ" for k, v in (r.get("worst") or [])[:3])
    return (f"vs CMU+ACCAD mocap (n={r.get('n_ref')}): distance {r.get('distance')}  "
            f"closeness {r.get('closeness')}%  ·  faults: {faults}  ·  biggest gaps: {worst}")


def metrics_line(p):
    return (
        f"{p.get('side','?')} hand  ·  elbow {p.get('elbow_deg',0):.0f}°  ·  "
        f"fist vs head {p.get('fist_vs_head_m',0)*100:+.0f} cm  ·  "
        f"opp-guard {p.get('opp_hand_to_head_m',0)*100:.0f} cm  ·  "
        f"hip {p.get('hip_rot_deg',0):.0f}°  ·  "
        f"drift {p.get('root_drift_m',0)*100:.0f} cm"
    )


def main():
    facts = json.load(open(FACTS))
    winners = facts.get("winners") or {}
    notes = facts.get("pick_notes") or {}
    col_w = PANEL[0]
    # 3 stills + path trace column
    row_h = LABEL_H + PANEL[1] + 8 + TRACE_H + 8 + STRIP_H
    w = PAD * 5 + col_w * 3
    h = TITLE_H + row_h * 4 + FOOT_H
    sheet = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD + 8, 14),
            "X Bot boxing picks v4  —  whole-path QC (not just the end frame)",
            font=_font(24, True), fill=FG)
    dr.text((PAD + 8, 50),
            "Path: fist max deviation from guard→extension chord, overhead arc height, "
            "elbow vs shoulder over the whole punch, retract along the same line.  "
            "Blue = outbound, orange = return.  Cross Punch.fbx was an overhand (swimming stroke).",
            font=_font(13), fill=MUTED)

    for ri, ptype in enumerate(ORDER):
        pid = winners.get(ptype)
        p = punch_by_id(facts, pid)
        y0 = TITLE_H + ri * row_h
        failed = bool(p.get("is_overhand") or p.get("is_looping")) and ptype in ("jab", "cross")
        color = BAD if failed else OK
        note = notes.get(ptype) or ""
        title = f"{TYPE_LABEL[ptype]}   winner: {p.get('clip_title') or pid or 'NONE'}"
        if failed:
            title += "   —  FAILS textbook path"
        dr.text((PAD + 8, y0 + 4), title, font=_font(17, True), fill=color)
        dr.text((PAD + 8, y0 + 28), metrics_line(p), font=_font(12), fill=ACCENT)
        dr.text((PAD + 8, y0 + 46), path_line(p), font=_font(12), fill=BAD if failed else MUTED)
        rr = (p.get("ref") or {}).get(ptype) or {}
        dr.text((PAD + 8, y0 + 64), ref_line(p, ptype), font=_font(12),
                fill=BAD if rr.get("faults") else OK)

        for ci, stage in enumerate(STAGE):
            x = PAD + ci * (col_w + PAD)
            path = os.path.join(EVAL, f"xbot_v4_{ptype}_{stage}.png")
            sheet.paste(load(path, PANEL), (x, y0 + LABEL_H))
            dr.text((x + 8, y0 + LABEL_H + PANEL[1] - 20),
                    f"{stage}  f{p.get(stage,'?')}", font=_font(13, True), fill=(210, 210, 214))

        # path traces under the three panels
        ty = y0 + LABEL_H + PANEL[1] + 8
        traj = p.get("traj") or []
        trace = qc.draw_path_trace(traj, size=(col_w * 2 + PAD, TRACE_H),
                                   title=f"{TYPE_LABEL[ptype]} fist path")
        sheet.paste(trace, (PAD, ty))
        # small runner-up / note panel
        rx = PAD + col_w * 2 + PAD * 2
        dr.rectangle((rx, ty, rx + col_w, ty + TRACE_H), outline=(50, 50, 56))
        runners = (facts.get("rankings") or {}).get(ptype) or []
        yy = ty + 8
        dr.text((rx + 8, yy), "top path-aware scores", font=_font(13, True), fill=MUTED)
        yy += 22
        for r in runners[:5]:
            mark = "←" if r["id"] == pid else " "
            flag = " OH" if r.get("is_overhand") else (" LP" if r.get("is_looping") else "")
            line = (f"{mark} {r['score']:.0f}  {r['title'][:26]:<26}  "
                    f"dev {((r.get('fist_max_dev_m') or 0)*100):.0f} "
                    f"arc {((r.get('fist_arc_height_m') or 0)*100):.0f} "
                    f"elb+ {((r.get('elbow_above_shoulder_max_m') or 0)*100):.0f} "
                    f"ref {r.get('ref_distance')}{flag}"
                    f"{'!' if r.get('ref_faults') else ''}")
            fill = OK if r["id"] == pid else (BAD if flag else FG)
            dr.text((rx + 8, yy), line[:66], font=_font(10), fill=fill)
            yy += 16

        # contact strip
        sy = ty + TRACE_H + 8
        strip_path = p.get("strip") or os.path.join(EVAL, "xbot_v4_strips", f"{pid}.png")
        if strip_path and os.path.isfile(strip_path):
            sim = Image.open(strip_path).convert("RGB")
            max_w = w - 2 * PAD
            sim.thumbnail((max_w, STRIP_H))
            sheet.paste(sim, (PAD, sy))
        else:
            dr.text((PAD + 8, sy + 8), "no contact strip", font=_font(14), fill=MUTED)

    # footer / IK note
    ik = facts.get("iktest") or {}
    fy = TITLE_H + row_h * 4 + 12
    lines = [
        "v3 cross (Cross Punch.fbx) rejected: rear elbow rises 16 cm above the shoulder mid-path (overhand / swimming stroke). "
        "v3 jab (alt16) also fails the new path check (elbow +8 cm). Pick rule: type gate -> path gates (elbow <= 3 cm above shoulder, "
        "fist deviation <= 6 cm for jab/cross) -> no mocap faults -> lowest distance to CMU+ACCAD reference. "
        "ref = RMS z-distance over height-normalised features (lower is closer; ~1 = typical real boxer spread).",
    ]
    if ik.get("ok"):
        b, a = ik.get("before") or {}, ik.get("after") or {}
        lines.append(
            f"IK straighten test on {ik.get('src')}: "
            f"arc {b.get('fist_arc_height_m')}→{a.get('fist_arc_height_m')} m, "
            f"elbow-above {b.get('elbow_above_shoulder_max_m')}→{a.get('elbow_above_shoulder_max_m')} m, "
            f"dev {b.get('fist_max_dev_m')}→{a.get('fist_max_dev_m')} m. "
            f"After looping={ik.get('after_looping')} overhand={ik.get('after_overhand')}."
        )
    fnt = _font(13)
    y = fy
    for paragraph in lines:
        words = paragraph.split()
        cur = ""
        for word in words:
            t = (cur + " " + word).strip()
            if dr.textlength(t, font=fnt) <= w - 2 * PAD - 16:
                cur = t
            else:
                dr.text((PAD + 8, y), cur, font=fnt, fill=MUTED)
                y += 18
                cur = word
        if cur:
            dr.text((PAD + 8, y), cur, font=fnt, fill=MUTED)
            y += 18

    sheet.save(OUT, "PNG", optimize=True)
    print("WROTE", OUT, sheet.size, os.path.getsize(OUT))


def compose_strips():
    """Compose contact strips from Blender-rendered frames (Blender has no PIL)."""
    import glob
    d = os.path.join(EVAL, "xbot_v4_strips")
    for sub in sorted(glob.glob(os.path.join(d, "*", ""))):
        pid = os.path.basename(sub.rstrip("/"))
        fr = sorted(glob.glob(os.path.join(sub, "f_*.png")))
        if fr:
            labels = [os.path.basename(x)[2:6].lstrip("0") for x in fr]
            qc.compose_contact_strip(fr, labels, thumb=(150, 266)).save(os.path.join(d, pid + ".png"))


if __name__ == "__main__":
    import sys as _s
    if "--compose-strips" in _s.argv:
        compose_strips()
    main()
