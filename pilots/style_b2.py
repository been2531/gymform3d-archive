"""Style B v2: pen-line stickman, big expressive faces, props, auto-framing.
Exercise-agnostic: side poses come from rig.<exercise>_pose via POSERS, so a
new episode is a JSON file (plus one pose function per new exercise)."""
from __future__ import annotations
import math
import skia
from .rig import (Body, pushup_pose, squat_pose, plank_pose, lunge_pose, bridge_pose, hinge_pose,
                  hand_x_for, rep_phase, rep_concentric, REP_DONE, add, d, two_bone)
from . import text as T
from .style_b import Pen, ease, back_out, BG, INK, RED, GREEN

W, H = 1080, 1920
FLOOR = 1250.0
SAFE_BOTTOM = int(H * 0.85) - 40
MAT_FILL = 0xFFFBEBD0
PAPER = 0xFFFFFFFF
GOLD = 0xFFD79A00
MUSCLE = 0xFFFF7A00       # target-muscle accent (orange; red stays reserved for faults)
MUSCLE_INK = 0xFFC2410C
HEAD_SCALE = {"bridge": 0.8}  # lying poses: smaller head so shoulders/torso read
MUSCLES = ("quads", "glutes", "hamstrings", "chest", "triceps", "core", "shoulders", "calves")
THINK_VARIANTS = ("scratch", "shrug", "hold")
TALL = ("squat", "hinge", "lunge")
FRONT_DRAWN = ("quads", "hamstrings", "calves", "chest", "core", "triceps", "shoulders")  # glutes: not visible from the front   # standing exercises: tall layout (figure >= 50% of frame height)
import re as _re
MUSCLE_WORDS = {"GLUTE": "glutes", "GLUTES": "glutes", "CORE": "core", "ABS": "core", "QUAD": "quads", "QUADS": "quads",
                "HAMSTRING": "hamstrings", "HAMSTRINGS": "hamstrings", "CHEST": "chest", "PEC": "chest", "PECS": "chest",
                "TRICEP": "triceps", "TRICEPS": "triceps", "SHOULDER": "shoulders", "SHOULDERS": "shoulders",
                "DELT": "shoulders", "DELTS": "shoulders", "CALF": "calves", "CALVES": "calves"}


def muscles_named(text):
    """Muscles a piece of on-screen text names (GLUTES, CORE, QUADS ...)."""
    return {MUSCLE_WORDS[w] for w in _re.findall(r"[A-Z]+", (text or "").upper()) if w in MUSCLE_WORDS}


def scene_shown_muscles(ep, sc):
    """Muscles drawn in a scene: wrong-form panels show none (orange only on right form / ending)."""
    base = sc.get("muscles", ep.get("muscles", []))
    t = sc["type"]
    if t in ("reps", "closeup"):
        return list(base) if sc.get("form") == "good" else []
    if t == "think":
        return [m for m in sc.get("muscles", []) if m in FRONT_DRAWN]
    if t == "conclusion":
        return [m for m in base if m in FRONT_DRAWN]
    return list(base)  # split (good panel)


def scene_texts(sc):
    out = [sc.get("title"), sc.get("sub"), (sc.get("cue") or {}).get("text")]
    out += [(sc.get(k) or {}).get("label") for k in ("top", "bottom")]
    return [x for x in out if x]
END_VARIANTS = ("flex", "victory", "thumbs")


def _unit(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / n, v[1] / n)


def body_front(J):
    """Screen-space unit vector the chest faces (perpendicular to the torso, toward the face)."""
    v = _unit((J["shoulder"][0] - J["hip"][0], J["shoulder"][1] - J["hip"][1]))
    return (-v[1], v[0])


class Poser:
    def __init__(self, body, ep):
        self.body = body; self.ep = ep; self.ex = ep.get("exercise", "pushup")
        if self.ex == "pushup":
            self.hand_x = hand_x_for(body, FLOOR, 130.0)

    def __call__(self, ph, form):
        b = self.body
        if self.ex == "pushup":
            return pushup_pose(b, ph, form, FLOOR, 130.0, self.hand_x)
        if self.ex == "squat":
            return squat_pose(b, ph, form, FLOOR, 520.0)
        if self.ex == "plank":
            return plank_pose(b, ph, form, FLOOR, 130.0)
        if self.ex == "lunge":
            return lunge_pose(b, ph, form, FLOOR, 520.0)
        if self.ex == "bridge":
            return bridge_pose(b, ph, form, FLOOR, 200.0)
        if self.ex == "hinge":
            return hinge_pose(b, ph, form, FLOOR, 520.0)
        raise ValueError(f"no pose function for exercise {self.ex!r}")


EXERCISES = ("pushup", "squat", "plank", "lunge", "bridge", "hinge")
PAIN_SPOTS = ("lower_back", "knee", "shoulder", "wrist", "hamstring", "neck", "hip")


class StyleB2:
    def __init__(self, ep):
        self.ep = ep; self.B = ep.get("B2") or ep["B"]; self.scenes = self.B["scenes"]
        self.body = Body(H=760)
        self.R = 0.105 * self.body.H * HEAD_SCALE.get(ep.get("exercise"), 1.0)  # head radius (world)
        self.muscles = [m for m in ep.get("muscles", []) if m in MUSCLES]
        self.poser = Poser(self.body, ep)
        bgs = skia.Surface(W, H); c = bgs.getCanvas(); c.clear(BG)
        self.bg_img = bgs.makeImageSnapshot()
        self._fit_cache = {}

    # ---------------- framing ----------------
    def head_c(self, J):
        a = math.radians(J["_head_ang"]); u = (math.cos(a), -math.sin(a))
        return (J["neck"][0] + u[0] * self.R * 1.0, J["neck"][1] + u[1] * self.R * 1.0)

    def bbox_side(self, forms):
        xs, ys = [], []
        for f in forms:
            for k in range(13):
                J = self.poser(k / 12, self.ep["forms"][f])
                for key in ("ball", "ankle", "knee", "hip", "shoulder", "elbow", "wrist", "fingers", "knee2", "ankle2", "ball2"):
                    if key in J:
                        xs.append(J[key][0]); ys.append(J[key][1])
                c = self.head_c(J); xs += [c[0] - self.R, c[0] + self.R]; ys += [c[1] - self.R, c[1] + self.R]
        if self.ep.get("exercise") in TALL:  # tall layout: box = head top .. floor (mat/floor may sit just below)
            return min(xs) - 40, min(ys) - 8, max(xs) + 40, FLOOR
        ys.append(FLOOR + 30)
        return min(xs) - 40, min(ys) - 20, max(xs) + 40, max(ys)

    def keep_box(self, form_key):
        """World bbox over the whole rep of everything a closeup must never crop
        (head + body joints; feet may crop)."""
        k = ("keep", form_key)
        if k not in self._fit_cache:
            xs, ys = [], []
            for i in range(13):
                J = self.poser(i / 12, self.ep["forms"][form_key])
                keys = ("knee", "hip", "shoulder", "neck", "elbow", "wrist", "knee2")
                if self.ep.get("exercise") in ("squat", "hinge", "lunge"):  # standing: keep the planted foot + floor
                    keys += ("ankle", "ball")
                for key in keys:
                    if key in J:
                        xs.append(J[key][0]); ys.append(J[key][1])
                c = self.head_c(J); xs += [c[0] - self.R * 1.15, c[0] + self.R * 1.15]; ys += [c[1] - self.R * 1.15, c[1] + self.R * 1.15]
            y1 = max(ys) + (6 if self.ep.get("exercise") in TALL else 0)
            self._fit_cache[k] = (min(xs), min(ys), max(xs), y1)
        return self._fit_cache[k]

    @staticmethod
    def clamp_view(z, f, a, kb, area):
        """Shrink zoom / shift focus so world box kb lands inside screen area."""
        ax0, ay0, ax1, ay1 = area
        z = min(z, (ax1 - ax0) / max(kb[2] - kb[0], 1), (ay1 - ay0) / max(kb[3] - kb[1], 1))
        fx, fy = f
        sx0 = a[0] + (kb[0] - fx) * z; sx1 = a[0] + (kb[2] - fx) * z
        if sx0 < ax0: fx -= (ax0 - sx0) / z
        elif sx1 > ax1: fx += (sx1 - ax1) / z
        sy0 = a[1] + (kb[1] - fy) * z; sy1 = a[1] + (kb[3] - fy) * z
        if sy0 < ay0: fy -= (ay0 - sy0) / z
        elif sy1 > ay1: fy += (sy1 - ay1) / z
        return z, (fx, fy)

    def fit(self, key, box, rect):
        """Zoom/focus so world box fits screen rect (x0,y0,x1,y1), bottom-aligned."""
        if key in self._fit_cache:
            return self._fit_cache[key]
        bx0, by0, bx1, by1 = box; rx0, ry0, rx1, ry1 = rect
        z = min((rx1 - rx0) / (bx1 - bx0), (ry1 - ry0) / (by1 - by0))
        focus = ((bx0 + bx1) / 2, by1); anchor = ((rx0 + rx1) / 2, ry1)
        self._fit_cache[key] = (z, focus, anchor)
        return self._fit_cache[key]

    def world(self, cv, z, focus, anchor, fn, clip=None):
        cv.save()
        if clip:
            cv.clipRect(skia.Rect.MakeLTRB(*clip))
        cv.translate(*anchor); cv.scale(z, z); cv.translate(-focus[0], -focus[1])
        self._xf = (z, focus, anchor, clip)
        fn(cv); cv.restore()

    def log_fig(self, meta, pts):
        """Screen-space bbox of a figure (joints + head), clipped, for the text-overlap gate."""
        if meta is None:
            return
        z, f, a, clip = self._xf
        sp = [(a[0] + (p[0] - f[0]) * z, a[1] + (p[1] - f[1]) * z) for p in pts]
        x0, y0 = min(p[0] for p in sp), min(p[1] for p in sp); x1, y1 = max(p[0] for p in sp), max(p[1] for p in sp)
        if clip:
            x0, y0, x1, y1 = max(x0, clip[0]), max(y0, clip[1]), min(x1, clip[2]), min(y1, clip[3])
        if x1 > x0 and y1 > y0:
            meta.setdefault("figs", []).append((x0, y0, x1, y1))

    # ---------------- face ----------------
    def face(self, cv, c, r, mood, t, look=(0.0, 0.0), z=1.0):
        lw = max(r * 0.075, 3.0)
        pen = Pen(t, lw, INK, 0.6)
        cx, cy = c[0] + look[0] * r * 0.18, c[1] + look[1] * r * 0.12
        ex, ey = r * 0.34, -r * 0.12
        er = {"shocked": 0.24, "worried": 0.19, "happy": 0.19, "strain": 0.17, "proud": 0.18}[mood]
        blink = (t % 2.7) < 0.09 and mood in ("worried", "happy")
        for s in (-1, 1):
            e = (cx + s * ex, cy + ey)
            if mood == "happy" or blink:
                p = skia.Path(); p.moveTo(e[0] - r * er, e[1] + r * 0.05); p.quadTo(e[0], e[1] - r * er * 1.3, e[0] + r * er, e[1] + r * 0.05)
                cv.drawPath(p, pen.paint())
            elif mood == "strain":
                p = skia.Path(); p.moveTo(e[0] - r * er, e[1] - s * r * 0.0 - r * 0.06); p.lineTo(e[0] + r * er * 0.2 * s, e[1]); p.lineTo(e[0] - r * er, e[1] + r * 0.06)
                if s == 1:
                    p = skia.Path(); p.moveTo(e[0] + r * er, e[1] - r * 0.06); p.lineTo(e[0] - r * er * 0.2, e[1]); p.lineTo(e[0] + r * er, e[1] + r * 0.06)
                cv.drawPath(p, pen.paint())
            elif mood == "proud":
                p = skia.Path(); p.addOval(skia.Rect.MakeXYWH(e[0] - r * er, e[1] - r * er * 0.9, 2 * r * er, 1.8 * r * er))
                cv.drawPath(p, pen.paint(fill=True, color=PAPER)); cv.drawPath(p, pen.paint())
                cv.drawCircle(e[0] + r * 0.03, e[1] + r * 0.03, r * er * 0.45, pen.paint(fill=True))
                lid = skia.Path(); lid.moveTo(e[0] - r * er * 1.1, e[1] - r * er * 0.1); lid.lineTo(e[0] + r * er * 1.1, e[1] - r * er * 0.1)
                cv.drawPath(lid, pen.paint())
            else:
                p = skia.Path(); p.addOval(skia.Rect.MakeXYWH(e[0] - r * er, e[1] - r * er * 1.15, 2 * r * er, 2.3 * r * er))
                cv.drawPath(p, pen.paint(fill=True, color=PAPER)); cv.drawPath(p, pen.paint())
                pr = r * er * (0.32 if mood == "shocked" else 0.5)
                cv.drawCircle(e[0] + look[0] * r * 0.05, e[1] + r * 0.03, pr, pen.paint(fill=True))
            # brows
            by = e[1] - r * er * 1.15 - r * (0.16 if mood == "shocked" else 0.1)
            b = skia.Path()
            if mood == "worried":
                b.moveTo(e[0] - s * r * 0.17, by - r * 0.08); b.lineTo(e[0] + s * r * 0.17, by + r * 0.04)
                b = skia.Path(); b.moveTo(e[0] + s * r * 0.17, by + r * 0.05); b.lineTo(e[0] - s * r * 0.15, by - r * 0.07)
            elif mood == "strain":
                b.moveTo(e[0] - s * r * 0.18, by + r * 0.08); b.lineTo(e[0] + s * r * 0.18, by - r * 0.04)
                b = skia.Path(); b.moveTo(e[0] + s * r * 0.18, by - r * 0.05); b.lineTo(e[0] - s * r * 0.16, by + r * 0.09)
            elif mood == "shocked":
                b.moveTo(e[0] - r * 0.16, by + r * 0.02); b.quadTo(e[0], by - r * 0.12, e[0] + r * 0.16, by + r * 0.02)
            elif mood == "proud":
                b.moveTo(e[0] - r * 0.16, by + (r * 0.04 if s == 1 else -r * 0.05)); b.lineTo(e[0] + r * 0.16, by + (-r * 0.05 if s == 1 else r * 0.04))
            else:
                b.moveTo(e[0] - r * 0.15, by); b.quadTo(e[0], by - r * 0.07, e[0] + r * 0.15, by)
            cv.drawPath(b, pen.paint())
        my = cy + r * 0.42
        m = skia.Path()
        if mood == "happy":
            m.moveTo(cx - r * 0.32, my - r * 0.06); m.quadTo(cx, my + r * 0.38, cx + r * 0.32, my - r * 0.06); m.close()
            cv.drawPath(m, pen.paint(fill=True, color=0xFFE86A6A)); cv.drawPath(m, pen.paint())
        elif mood == "shocked":
            m.addOval(skia.Rect.MakeXYWH(cx - r * 0.13, my - r * 0.14, r * 0.26, r * 0.32))
            cv.drawPath(m, pen.paint(fill=True)); 
        elif mood == "worried":
            m.moveTo(cx - r * 0.26, my + r * 0.05); m.cubicTo(cx - r * 0.12, my - r * 0.08, cx + r * 0.0, my + r * 0.12, cx + r * 0.12, my - r * 0.02)
            m.lineTo(cx + r * 0.26, my + r * 0.06)
            cv.drawPath(m, pen.paint())
        elif mood == "strain":
            m.addRect(skia.Rect.MakeXYWH(cx - r * 0.27, my - r * 0.08, r * 0.54, r * 0.2))
            cv.drawPath(m, pen.paint(fill=True, color=PAPER)); cv.drawPath(m, pen.paint())
            for k in (-1, 0, 1):
                cv.drawLine(cx + k * r * 0.12, my - r * 0.08, cx + k * r * 0.12, my + r * 0.12, pen.paint())
            cv.drawLine(cx - r * 0.27, my + r * 0.02, cx + r * 0.27, my + r * 0.02, pen.paint())
            # sweat drop
            dp = skia.Path(); sx, sy = c[0] + r * 0.95, c[1] - r * 0.35 + (t * 60 % 30)
            dp.moveTo(sx, sy - r * 0.18); dp.quadTo(sx + r * 0.12, sy, sx, sy + r * 0.06); dp.quadTo(sx - r * 0.12, sy, sx, sy - r * 0.18)
            cv.drawPath(dp, pen.paint(fill=True, color=0xFF8FD0FF)); cv.drawPath(dp, Pen(t, lw * 0.7, INK, 0.3).paint())
        elif mood == "proud":
            m.moveTo(cx - r * 0.24, my - r * 0.02); m.quadTo(cx + r * 0.02, my + r * 0.2, cx + r * 0.3, my - r * 0.12)
            cv.drawPath(m, pen.paint())

    def head(self, cv, pen, c, mood, t, seed, look=(0.0, 0.0)):
        cv.drawPath(pen.circle(c, self.R, seed), pen.paint(fill=True, color=PAPER))
        cv.drawPath(pen.circle(c, self.R, seed), pen.paint())
        self.face(cv, c, self.R, mood, t, look)

    # ---------------- figures ----------------
    def stick_side(self, cv, pen, J, mood, t, seed=0, hl=(), meta=None, muscles=None, pulse=0.0):
        far = Pen(t, pen.w, 0xFF8A93A0, pen.amp)
        off = (9, -6)
        F = {k: (v[0] + off[0], v[1] + off[1]) if isinstance(v, tuple) else v for k, v in J.items()}
        z = self._xf[0] if getattr(self, "_xf", None) else 1.0
        if "knee2" in J:
            cv.drawPath(far.poly([J["hip"], J["knee2"], J["ankle2"], J["ball2"]], seed + 1), far.paint())
        else:
            cv.drawPath(far.poly([F["hip"], F["knee"], F["ankle"], F["ball"]], seed + 1), far.paint())
        cv.drawPath(far.poly([F["shoulder"], F["elbow"], F["wrist"], F["fingers"]], seed + 2), far.paint())
        cv.drawPath(pen.poly([J["hip"], J["knee"], J["ankle"], J["ball"]], seed + 3), pen.paint())
        cv.drawPath(pen.poly([J["hip"], J["shoulder"], J["neck"]], seed + 4), pen.paint())
        cv.drawPath(pen.poly([J["shoulder"], J["elbow"], J["wrist"], J["fingers"]], seed + 5), pen.paint())
        if muscles:
            self.draw_muscles_side(cv, pen, J, muscles, t, pulse, meta, z)
        rp = Pen(t, pen.w * 1.5, RED, pen.amp)
        for k_, (a_, b_) in enumerate(hl):  # fault: the faulty segment itself re-inked red (simple stroke, no band)
            if a_ in J and b_ in J:
                cv.drawPath(rp.poly([J[a_], J[b_]], seed + 30 + k_), rp.paint())
                if meta is not None:
                    ln = math.hypot(J[b_][0] - J[a_][0], J[b_][1] - J[a_][1])
                    meta.setdefault("hl", []).append((pen.w * 1.5 * z, ln * z))
        hc = self.head_c(J)
        self.head_profile(cv, pen, hc, J, mood, t, seed + 6, meta)
        pts = [v for k, v in J.items() if not k.startswith("_") and isinstance(v, tuple)]
        pts += [(hc[0] - self.R, hc[1] - self.R), (hc[0] + self.R, hc[1] + self.R)]
        self.log_fig(meta, pts)

    # ---------------- target muscles (simple: recolour the worked segments) ----------------
    # muscle -> (from joint, to joint, start fraction, end fraction) on the side figure
    SIDE_SEGS = {"quads": [("hip", "knee", 0.0, 1.0)], "hamstrings": [("hip", "knee", 0.0, 1.0)],
                 "glutes": [("hip", "knee", 0.0, 0.45)], "calves": [("knee", "ankle", 0.0, 0.7)],
                 "chest": [("shoulder", "hip", 0.0, 0.45)], "core": [("shoulder", "hip", 0.25, 1.0)],
                 "triceps": [("shoulder", "elbow", 0.0, 1.0)], "shoulders": [("shoulder", "elbow", 0.0, 0.4)]}

    def _seg_stroke(self, cv, pen, A, B, f0, f1, color, wmul, t, seed, alpha=1.0):
        p0 = (A[0] + (B[0] - A[0]) * f0, A[1] + (B[1] - A[1]) * f0)
        p1 = (A[0] + (B[0] - A[0]) * f1, A[1] + (B[1] - A[1]) * f1)
        sp = Pen(t, pen.w * wmul, color, pen.amp)
        cv.drawPath(sp.poly([p0, p1], seed), sp.paint(alpha=alpha))
        return p0, p1, math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    def draw_muscles_side(self, cv, pen, J, muscles, t, pulse, meta, z):
        """Worked segments re-inked in orange, slightly thicker, gentle pulse on the lift. No extra shapes."""
        wmul = 1.45 + 0.25 * pulse
        for k, m in enumerate(muscles):
            area = 0.0
            for (a_, b_, f0, f1) in self.SIDE_SEGS.get(m, []):
                if a_ in J and b_ in J:
                    _, _, ln = self._seg_stroke(cv, pen, J[a_], J[b_], f0, f1, MUSCLE, wmul, t, 60 + k)
                    area += ln * pen.w * wmul * z * z
            if meta is not None:
                meta.setdefault("muscles", []).append((m, area))

    def draw_muscles_front(self, cv, pen, P, muscles, t, pulse, meta=None):
        """Front view: only limb/torso lines, never near the groin. Thighs are coloured from mid-thigh down,
        glutes are not drawn (not visible from the front), chest/core stay on the torso line above the hips."""
        H_ = self.body.H
        hip, sh = P["hip"], P["shoulder"]
        wmul = 1.45 + 0.25 * pulse
        segs = []
        for m in muscles:
            if m in ("quads", "hamstrings"):
                segs += [(hip, kn, 0.72, 1.0) for (kn, an, bl) in P["legs"]]
            elif m == "calves":
                segs += [(kn, an, 0.0, 0.7) for (kn, an, bl) in P["legs"]]
            elif m == "chest":
                segs.append((sh, hip, 0.0, 0.4))
            elif m == "core":
                segs.append((sh, hip, 0.35, 0.78))
            elif m == "triceps":
                segs += [(sh, e, 0.0, 1.0) for (e, w) in P["arms"]]
            elif m == "shoulders":
                segs += [(sh, e, 0.0, 0.4) for (e, w) in P["arms"]]
        thigh = math.hypot(P["legs"][0][0][0] - hip[0], P["legs"][0][0][1] - hip[1])
        zone = (hip[0] - 0.07 * H_, hip[1] - 0.04 * H_, hip[0] + 0.07 * H_, hip[1] + 0.55 * thigh)  # crotch exclusion
        hits = 0
        for k, (A, B, f0, f1) in enumerate(segs):
            p0, p1, _ = self._seg_stroke(cv, pen, A, B, f0, f1, MUSCLE, wmul, t, 120 + k)
            for i in range(21):
                q = (p0[0] + (p1[0] - p0[0]) * i / 20, p0[1] + (p1[1] - p0[1]) * i / 20)
                r = pen.w * wmul / 2
                if zone[0] - r < q[0] < zone[2] + r and zone[1] - r < q[1] < zone[3] + r:
                    hits += 1
        if meta is not None:
            meta.setdefault("muscles", []).extend((m, 1e6) for m in muscles if m in FRONT_DRAWN)  # only what is really drawn
            meta.setdefault("front_zone_hits", []).append(hits)

    # ---------------- side-profile face ----------------
    def head_profile(self, cv, pen, c, J, mood, t, seed, meta=None):
        """Profile face looking the way the body faces (eye/nose/mouth on the front half)."""
        R = self.R
        cv.drawPath(pen.circle(c, R, seed), pen.paint(fill=True, color=PAPER))
        cv.drawPath(pen.circle(c, R, seed), pen.paint())
        u = _unit((c[0] - J["neck"][0], c[1] - J["neck"][1]))  # crown direction
        f = (-u[1], u[0])                                           # face direction
        nb = body_front(J)
        if f[0] * nb[0] + f[1] * nb[1] < 0:  # neck bent past the torso: keep the face with the chest
            f = (-f[0], -f[1])
        f = _unit((f[0] + 0.45 * u[0] * abs(u[0]), f[1] + 0.45 * u[1] * abs(u[0])))  # head pointing forward: look ahead-down
        if meta is not None:
            meta.setdefault("faces", []).append(round(f[0] * nb[0] + f[1] * nb[1], 3))
        P = lambda a, b: (c[0] + f[0] * a * R + u[0] * b * R, c[1] + f[1] * a * R + u[1] * b * R)  # noqa: E731
        lw = max(R * 0.09, 3.0)
        fp = Pen(t, lw, INK, 0.5)
        def poly(pts, close=False):
            p = skia.Path(); p.moveTo(*pts[0])
            for q in pts[1:]: p.lineTo(*q)
            if close: p.close()
            return p
        def quad(a, m, b):
            p = skia.Path(); p.moveTo(*a); p.quadTo(*m, *b); return p
        # nose: one clear bump on the front edge
        cv.drawPath(quad(P(0.96, 0.10), P(1.24, -0.02), P(0.97, -0.16)), fp.paint())
        # eye (one, on the front half)
        e = P(0.46, 0.16)
        blink = (t % 2.7) < 0.09 and mood in ("worried", "happy")
        if mood == "happy" or blink:
            cv.drawPath(quad(P(0.30, 0.12), P(0.46, 0.34), P(0.62, 0.12)), fp.paint())
        else:
            er = R * (0.23 if mood == "shocked" else 0.2)
            cv.drawCircle(e[0], e[1], er, fp.paint(fill=True, color=PAPER)); cv.drawCircle(e[0], e[1], er, fp.paint())
            pe = P(0.46 + (0.07 if mood != "shocked" else 0.02), 0.15 - (0.04 if mood == "strain" else 0))
            cv.drawCircle(pe[0], pe[1], er * (0.38 if mood == "shocked" else 0.5), fp.paint(fill=True))
            if mood == "strain":  # heavy lid
                cv.drawPath(poly([P(0.24, 0.30), P(0.70, 0.20)]), Pen(t, lw * 1.2, INK, 0.4).paint())
            if mood == "proud":
                cv.drawPath(poly([P(0.27, 0.24), P(0.66, 0.24)]), fp.paint())
        # brow
        brow = {"strain": (P(0.24, 0.52), P(0.70, 0.36)), "worried": (P(0.28, 0.36), P(0.64, 0.46)),
                "shocked": (P(0.28, 0.50), P(0.64, 0.54)), "proud": (P(0.28, 0.40), P(0.64, 0.38)),
                "happy": (P(0.30, 0.42), P(0.62, 0.46))}.get(mood, (P(0.3, 0.42), P(0.64, 0.42)))
        cv.drawPath(poly(list(brow)), Pen(t, lw * 1.3, INK, 0.4).paint())
        # mouth
        if mood == "happy":
            m = skia.Path(); m.moveTo(*P(0.42, -0.36)); m.quadTo(*P(0.66, -0.66), *P(0.90, -0.30)); m.close()
            cv.drawPath(m, fp.paint(fill=True, color=0xFFE86A6A)); cv.drawPath(m, fp.paint())
        elif mood == "shocked":
            mc = P(0.68, -0.44); cv.drawCircle(mc[0], mc[1], R * 0.13, fp.paint(fill=True))
        elif mood == "strain":  # clenched grimace: zig-zag lips
            m = poly([P(0.40, -0.34), P(0.92, -0.30), P(0.88, -0.56), P(0.44, -0.52)], close=True)
            cv.drawPath(m, fp.paint(fill=True, color=PAPER)); cv.drawPath(m, fp.paint())
            cv.drawPath(poly([P(0.42, -0.43), P(0.90, -0.43)]), Pen(t, lw * 0.8, INK, 0.3).paint())
            dp = skia.Path(); sx_, sy_ = P(-0.45, 0.80); sy_ += (t * 60 % 30)
            dp.moveTo(sx_, sy_ - R * 0.18); dp.quadTo(sx_ + R * 0.12, sy_, sx_, sy_ + R * 0.06); dp.quadTo(sx_ - R * 0.12, sy_, sx_, sy_ - R * 0.18)
            cv.drawPath(dp, fp.paint(fill=True, color=0xFF8FD0FF)); cv.drawPath(dp, Pen(t, lw * 0.6, INK, 0.3).paint())
        elif mood == "worried":
            cv.drawPath(quad(P(0.48, -0.48), P(0.68, -0.34), P(0.88, -0.48)), fp.paint())
        else:  # proud smirk
            cv.drawPath(quad(P(0.46, -0.42), P(0.70, -0.52), P(0.90, -0.30)), fp.paint())

    def front_pose(self, base, t, kind, variant=None):
        """Front view, FK/IK with rig lengths; returns joints + draw list."""
        L = self.body.L; Hh = self.body.H
        bob = math.sin(t * 2 * math.pi * 1.3)
        hip = (base[0], base[1] - L["thigh"] - L["shin"] - 0.03 * Hh + 3 * bob)
        legs = []
        for s in (-1, 1):
            knee = add(hip, d(-90 + s * 11), L["thigh"])
            ankle = add(knee, d(-90 + s * 4), L["shin"])
            legs.append((knee, ankle, add(ankle, d(90 - s * 90), L["foot"] * 0.55)))
        sh = add(hip, d(90 + 1.5 * bob), L["torso"])
        neck = add(sh, d(90), L["neck"])
        hc = (neck[0], neck[1] - self.R * 1.0)
        arms = []
        for s in (-1, 1):
            if kind == "flex" and variant == "victory":  # both arms up in a V
                pump = math.sin(t * 2 * math.pi * 1.4)
                el = add(sh, d(90 - s * (48 + 4 * pump)), L["upper_arm"])
                wr = add(el, d(90 - s * (28 + 6 * pump)), L["forearm"])
            elif kind == "flex" and variant == "thumbs":  # right: thumbs up at chest height, left: hand on hip
                if s == 1:
                    pump = math.sin(t * 2 * math.pi * 1.6)
                    el = add(sh, d(-38 + 4 * pump), L["upper_arm"]); wr = add(el, d(62 + 6 * pump), L["forearm"])
                else:
                    el, wr = two_bone(sh, (hip[0] - 0.05 * Hh, hip[1] - 0.02 * Hh), L["upper_arm"], L["forearm"], pick=lambda c: -c[0])
            elif kind == "think" and variant == "shrug":  # palms up, elbows in
                sh_up = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.1)
                el = add(sh, d(-90 + s * 28), L["upper_arm"])
                wr = add(el, d(90 - s * (72 + 10 * sh_up)), L["forearm"])
            elif kind == "think" and variant == "hold":  # hands on hips / lower back, wincing
                el, wr = two_bone(sh, (hip[0] + s * 0.05 * Hh, hip[1] - 0.03 * Hh), L["upper_arm"], L["forearm"], pick=lambda c, s=s: s * c[0])
            elif kind == "flex":
                pump = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.6)
                el = add(sh, d(90 - s * (82 - 6 * pump)), L["upper_arm"])
                wr = add(el, d(90 - s * (8 + 14 * (1 - pump))), L["forearm"])
            elif kind == "think" and s == 1:
                wig = math.sin(t * 2 * math.pi * 3.2)
                target = (hc[0] + self.R * 1.25, hc[1] - self.R * 0.25 + 6 * wig)
                el, wr = two_bone(sh, target, L["upper_arm"], L["forearm"], pick=lambda c: c[0])  # elbow outward
            elif kind == "think":
                el = add(sh, d(-118), L["upper_arm"]); wr = add(el, d(-55), L["forearm"])  # hand on lower back/hip
            else:
                el = add(sh, d(-90 - s * 22), L["upper_arm"]); wr = add(el, d(-90 - s * 12), L["forearm"])
            arms.append((el, wr))
        return {"hip": hip, "legs": legs, "shoulder": sh, "neck": neck, "head_c": hc, "arms": arms, "variant": variant}

    def stick_front(self, cv, pen, P, mood, t, meta, kind, seed=0, muscles=None, pulse=0.0):
        hip, sh, neck = P["hip"], P["shoulder"], P["neck"]
        var = P.get("variant") or ("flex" if kind == "flex" else "scratch")
        for i, (k, a, b) in enumerate(P["legs"]):
            cv.drawPath(pen.poly([hip, k, a, b], seed + i), pen.paint())
        cv.drawPath(pen.poly([hip, sh, neck], seed + 3), pen.paint())
        for i, (e, w) in enumerate(P["arms"]):
            cv.drawPath(pen.poly([sh, e, w], seed + 4 + i), pen.paint())
            cv.drawCircle(w[0], w[1], pen.w * (1.9 if kind == "flex" else 1.0), pen.paint(fill=True, color=PAPER))
            cv.drawCircle(w[0], w[1], pen.w * (1.9 if kind == "flex" else 1.0), pen.paint())
            if kind == "flex" and var == "thumbs" and w[0] > sh[0]:
                th = skia.Path(); th.moveTo(w[0], w[1] - pen.w * 1.6); th.lineTo(w[0] + 4, w[1] - pen.w * 1.6 - 46)
                cv.drawPath(th, Pen(t, pen.w * 1.1, INK, 0.5).paint())
            if kind == "flex" and var == "flex":
                # bicep bump on top of the upper arm + motion lines by the fist
                mx, my = (sh[0] + e[0]) / 2, (sh[1] + e[1]) / 2
                bump = skia.Path(); bump.moveTo(mx - 34, my - 6); bump.quadTo(mx, my - 62, mx + 34, my - 6)
                cv.drawPath(bump, pen.paint())
                s = -1 if e[0] < sh[0] else 1
                for k in range(3):
                    ph = (t * 3 + k * 0.33) % 1.0
                    x0 = w[0] + s * (40 + 14 * ph); y0 = w[1] - 30 + k * 30
                    lp = Pen(t, pen.w * 0.6, INK, 0.5).paint(alpha=1 - ph * 0.6)
                    cv.drawLine(x0, y0, x0 + s * 34, y0 - 8, lp)
        if muscles:  # worked segments re-inked orange on top of the body lines
            self.draw_muscles_front(cv, pen, P, muscles, t, pulse, meta)
        mood_look = (0.0, 0.0)
        self.head(cv, pen, P["head_c"], mood, t, seed + 9, mood_look)
        if kind == "think" and var == "scratch":  # scratching hand stays in front of the head
            e, w = P["arms"][1]
            cv.drawPath(pen.poly([sh, e, w], seed + 5), pen.paint())
            cv.drawCircle(w[0], w[1], pen.w * 1.3, pen.paint(fill=True, color=PAPER)); cv.drawCircle(w[0], w[1], pen.w * 1.3, pen.paint())
        hc = P["head_c"]
        pts = [hip, sh, neck, (hc[0] - self.R, hc[1] - self.R), (hc[0] + self.R, hc[1] + self.R)]
        pts += [p for leg in P["legs"] for p in leg] + [p for arm in P["arms"] for p in arm]
        self.log_fig(meta, pts)
        if meta is not None:
            for (k, a, b), (e, w) in zip(P["legs"], P["arms"]):
                meta["joints"].append({"hip": hip, "knee": k, "ankle": a, "shoulder": sh, "elbow": e, "wrist": w, "neck": neck})

    # ---------------- props ----------------
    def mat(self, cv, pen, x0, x1):
        r = skia.Rect.MakeLTRB(x0, FLOOR - 4, x1, FLOOR + 22)
        p = skia.Path(); p.addRoundRect(r, 12, 12)
        cv.drawPath(p, pen.paint(fill=True, color=MAT_FILL)); cv.drawPath(p, Pen(0, pen.w * 0.7, INK, 1.2).paint())

    def floor(self, cv, pen, x0=-200, x1=1300):
        cv.drawPath(pen.poly([(x0, FLOOR + 24), (x1, FLOOR + 24)], 91), Pen(0, pen.w * 0.55, 0xFF7C8796).paint())

    def bubble(self, cv, c, rx, ry, t, content="?"):
        pen = Pen(t, 9, INK, 1.4)
        lobes = 9
        path = skia.Path()
        for k in range(lobes):
            a = 2 * math.pi * k / lobes
            path.addCircle(c[0] + rx * 0.82 * math.cos(a), c[1] + ry * 0.75 * math.sin(a), min(rx, ry) * 0.42)
        path.addOval(skia.Rect.MakeXYWH(c[0] - rx * 0.85, c[1] - ry * 0.8, rx * 1.7, ry * 1.6))
        u = skia.Path(); 
        u = path
        cv.drawPath(u, pen.paint(fill=True, color=PAPER))
        o = skia.Op(path, skia.Path(), skia.PathOp.kUnion_PathOp) or path
        cv.drawPath(o, pen.paint())
        f = skia.Font(T.typeface("marker"), ry * 1.1)
        w = f.measureText(content)
        cv.drawString(content, c[0] - w / 2, c[1] + ry * 0.38, f, pen.paint(fill=True))
        return pen

    def sparkle(self, cv, c, s, t, k):
        tw = 0.6 + 0.4 * math.sin(t * 7 + k * 1.7)
        s *= tw
        p = skia.Path(); p.moveTo(c[0], c[1] - s)
        for a, b in (((c[0] + s * 0.2, c[1] - s * 0.2), (c[0] + s, c[1])), ((c[0] + s * 0.2, c[1] + s * 0.2), (c[0], c[1] + s)),
                     ((c[0] - s * 0.2, c[1] + s * 0.2), (c[0] - s, c[1])), ((c[0] - s * 0.2, c[1] - s * 0.2), (c[0], c[1] - s))):
            p.quadTo(*a, *b)
        cv.drawPath(p, Pen(t, 6, GOLD).paint(fill=True, color=0xFFFFD34D)); cv.drawPath(p, Pen(t, 6, INK).paint())

    def mark(self, cv, kind, c, s, t_since, seed):
        if t_since < 0: return
        k = back_out(t_since / 0.25) if t_since < 0.25 else 1.0
        pen = Pen(t_since + seed, 20, RED if kind == "bad" else GREEN, 2.5)
        cv.save(); cv.translate(*c); cv.scale(k, k)
        if kind == "bad":
            cv.drawPath(pen.poly([(-s, -s), (s, s)], seed), pen.paint()); cv.drawPath(pen.poly([(s, -s), (-s, s)], seed + 1), pen.paint())
        else:
            cv.drawPath(pen.poly([(-s, 0), (-s * 0.3, s * 0.7), (s, -s * 0.8)], seed), pen.paint())
        cv.restore()

    def guide(self, cv, spec, J, t, phase):
        g = {"type": spec} if isinstance(spec, str) else dict(spec)
        typ = g.get("type", "line")
        alpha = 1.0
        if typ == "depth":  # only meaningful at the bottom of the rep
            alpha = min(max((phase - 0.72) / 0.18, 0.0), 1.0)
            if alpha <= 0:
                return
        p = skia.Paint(AntiAlias=True, Color=GREEN, Style=skia.Paint.kStroke_Style, StrokeWidth=8,
                       PathEffect=skia.DashPathEffect.Make([28, 18], -t * 140))
        p.setAlphaf(alpha)
        if typ == "line":
            a, b = J[g.get("a", "ankle")], J[g.get("b", "shoulder")]
            dx, dy = b[0] - a[0], b[1] - a[1]; n = math.hypot(dx, dy)
            off = g.get("offset", 70)
            ox, oy = dy / n * off, -dx / n * off
            if oy > 0: ox, oy = -ox, -oy  # always above the body
            cv.drawLine(a[0] - dx / n * 60 + ox, a[1] - dy / n * 60 + oy, b[0] + dx / n * 120 + ox, b[1] + dy / n * 120 + oy, p)
        elif typ == "depth":
            y = J["knee"][1]
            cv.drawLine(J["hip"][0] - 140, y, J["knee"][0] + 90, y, p)
        elif typ == "vertical":
            a = J[g.get("a", "ankle")]
            cv.drawLine(a[0], a[1] + 20, a[0], a[1] - g.get("len", 420), p)
        elif typ == "torso":
            hx, hy = J["hip"]; sx, sy = J["shoulder"]; dx, dy = sx - hx, sy - hy; n = math.hypot(dx, dy)
            cv.drawLine(hx - dx / n * 30, hy - dy / n * 30, sx + dx / n * 160, sy + dy / n * 160, p)

    def arrow(self, cv, target, t, frm=(-20, -230)):
        pen = Pen(t, 9, RED)
        tx, ty = target
        cv.drawPath(pen.poly([(tx + frm[0], ty + frm[1]), (tx - 5, ty - 30)], 4), pen.paint())
        cv.drawPath(pen.poly([(tx - 40, ty - 65), (tx - 5, ty - 25), (tx + 28, ty - 68)], 6), pen.paint())

    # ---------------- timeline ----------------
    def scene_at(self, t):
        for i, s in enumerate(self.scenes):
            if s["t0"] <= t < s["t1"]:
                return i, s
        return len(self.scenes) - 1, self.scenes[-1]

    def events(self):
        ev = []
        for s in self.scenes:
            if s["t0"] > 0: ev.append((s["t0"] - 0.1, "whoosh"))
            ev.append((s["t0"] + 0.05, "pop"))
            if s.get("cue"): ev.append((s["t0"] + 0.45, "pop"))
            if s["type"] in ("reps", "split", "closeup"):
                P = self.ep["rep_period"]; t = REP_DONE * P
                while t < s["t1"]:
                    if t > s["t0"]: ev.append((t, "tick"))
                    t += P
        return ev

    def pose(self, t, form, offset=0.0):
        return self.poser(rep_phase(t, self.ep["rep_period"], offset), self.ep["forms"][form])

    def variant(self, typ):
        """Think / ending pose variant: JSON 'variant' wins, else a stable pick from the episode id."""
        opts = THINK_VARIANTS if typ == "think" else END_VARIANTS
        h = sum(ord(ch) for ch in self.ep.get("id", "")) + (0 if typ == "think" else 1)
        return opts[h % len(opts)]

    def fault_separation(self):
        """Per wrong form: how far (in body heights) it sits from 'good' at its least visible moment.
        Range-only faults (depth/lift) are judged at the bottom of the rep."""
        out = {}
        good = self.ep["forms"].get("good", {})
        for name, fm in self.ep["forms"].items():
            if name == "good":
                continue
            keys = {k for k in set(fm) | set(good) if k != "highlight" and fm.get(k) != good.get(k)}
            phases = [1.0] if keys <= {"depth", "lift"} else [i / 12 for i in range(13)]
            worst = 9.0
            for ph in phases:
                Jw, Jg = self.poser(ph, fm), self.poser(ph, good)
                dd = max(math.hypot(Jw[k][0] - Jg[k][0], Jw[k][1] - Jg[k][1]) for k in Jw
                         if not k.startswith("_") and k in Jg and isinstance(Jw[k], tuple))
                worst = min(worst, dd / self.body.H)
            out[name] = round(worst, 3)
        return out

    def render(self, cv, t, meta):
        if t == 0:
            meta["fault_sep"] = self.fault_separation()
            meta["variants"] = {sc["type"]: sc.get("variant") or self.variant(sc["type"]) for sc in self.scenes
                                if sc["type"] in ("think", "conclusion")}
        sc_ = self.scene_at(t)[1]; meta["scene"] = sc_["type"]; meta["scene_t0"] = f"{sc_['type']}@{sc_['t0']}"
        meta["shown_muscles"] = scene_shown_muscles(self.ep, sc_)
        cv.drawImage(self.bg_img, 0, 0)
        i, s = self.scene_at(t); ts = t - s["t0"]
        punch = 1.0 + 0.08 * (1 - ease(ts / 0.3))
        typ = s["type"]; logt = meta["text"]
        title_bottom = 150
        if s.get("title"):
            title_bottom = T.block(cv, logt, "title", s["title"], self.B.get("title_font", "marker"), 112, 540, 150, INK, BG,
                                   max_w=1010, max_lines=2, fallback_font="kalam")
        if typ in ("reps", "closeup"):
            form = self.ep["forms"][s["form"]]
            ph = rep_phase(t, self.ep["rep_period"])
            J = self.poser(ph, form); meta["joints"].append(J)
            meta["phase"] = ph; meta["solo"] = True
            box = self.bbox_side([s["form"]])
            tall = self.ep.get("exercise") in TALL
            fig_top = title_bottom + 30 if tall else max(title_bottom + 50, 470)
            if tall:  # gentler scene-start punch so the tall figure never grows into the title
                punch = 1.0 + 0.025 * (1 - ease(ts / 0.3))
            rb = 1430 if tall else 1190
            meta["solo_good"] = s["form"] == "good"
            fbox = box
            if (box[2] - box[0]) > 1.6 * (box[3] - box[1]):  # horizontal pose: frame hips-to-head, feet may bleed off
                kb = self.keep_box(s["form"])
                fbox = (kb[0] - 40, box[1], kb[2] + 40, box[3])
            z, f, a = self.fit(("solo", s["form"], fig_top), fbox, (40, fig_top, 1040, rb))
            clip = None
            if typ == "closeup" and not tall:  # tall poses already fill the frame top-to-floor: no crop-free zoom left
                fj = J[s.get("focus", "hip")]
                z = z * (s.get("zoom", 1.35) + 0.1 * ease(ts / 2.0))
                f = (fj[0] + s.get("dx", 0), fj[1] + s.get("dy", 0)); a = (540, (fig_top + rb + 20) / 2)
                clip = (0, fig_top - 30, 1080, rb + (45 if tall else 25))
                # never crop the head/body: account for the punch zoom too
                z, f = self.clamp_view(z * 1.08, f, a, self.keep_box(s["form"]), (40, fig_top - 10, 1040, rb + (22 if tall else 15)))
                z /= 1.08
            pen = Pen(t, 13 / z * 1.15, INK, 2.0 / z * 1.1)
            hl = [tuple(x) for x in form.get("highlight", [])]
            def draw(c):
                self.floor(c, pen); self.mat(c, pen, box[0] + 10, box[2] - 10)
                self.stick_side(c, pen, J, s.get("face", "strain"), t, hl=hl, meta=meta,
                                muscles=scene_shown_muscles(self.ep, s), pulse=rep_concentric(t, self.ep["rep_period"]))
                if typ == "closeup" and s.get("guide"):
                    self.guide(c, s["guide"], J, t, ph)
                if typ == "closeup" and s.get("arrow"):
                    self.arrow(c, J[s["arrow"]], t)
            self.world(cv, z * punch, f, a, draw, clip=clip)
            self.cue_block(cv, logt, s, ts, t, tall=tall)
        elif typ in ("think", "conclusion"):
            kind = "think" if typ == "think" else "flex"
            var = s.get("variant") or self.variant(typ)
            P0 = self.front_pose((540, FLOOR), 0.0, kind, var)
            ys = [P0["head_c"][1] - self.R] + [w[1] for _, w in P0["arms"]]
            top = min(ys) - 30
            z = (1560 - (680 if typ == "conclusion" else 600)) / (FLOOR + 24 - top)  # character ~46-50% of frame height
            if typ == "think":  # bubble top (head - ~365px on screen) must clear the title block
                hc0 = P0["head_c"][1]
                z = min(z, (1560 - (title_bottom + 30 + 365 * 1.08)) / (FLOOR + 24 - hc0))
            P = self.front_pose((540, FLOOR), t, kind, var)
            pen = Pen(t, 13 / z * 1.15, INK, 2.0 / z * 1.1)
            def draw(c):
                if typ == "conclusion":
                    self.mat(c, pen, 300, 780)
                else:
                    self.floor(c, pen, 250, 830)
                self.stick_front(c, pen, P, s.get("face", "worried"), t, meta, kind,
                                 muscles=scene_shown_muscles(self.ep, s) or None,
                                 pulse=0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.2))
                if typ == "think":
                    self.pain(c, P, s.get("pain", "lower_back"), t)
            self.world(cv, z * punch, (540, FLOOR + 24), (540, 1560), draw)
            if typ == "conclusion" and s.get("sub"):
                pass
            if typ == "think":
                hc = P["head_c"]
                # bubble in screen space, up-left of the head
                hs = (540 + (hc[0] - 540) * z, 1560 + (hc[1] - FLOOR - 24) * z)
                bob = 8 * math.sin(t * 3)
                for k, (dx, dy, rr) in enumerate(((-150, -40, 14), (-205, -95, 22))):
                    if ts > 0.1 + k * 0.1:
                        cv.drawCircle(hs[0] + dx, hs[1] + dy + bob, rr, Pen(t, 8).paint(fill=True, color=PAPER))
                        cv.drawCircle(hs[0] + dx, hs[1] + dy + bob, rr, Pen(t, 8).paint())
                if ts > 0.3:
                    sc = back_out((ts - 0.3) / 0.25) if ts < 0.55 else 1.0
                    cv.save(); bc = (hs[0] - 290, hs[1] - 240 + bob); cv.translate(*bc); cv.scale(sc, sc); cv.translate(-bc[0], -bc[1])
                    self.bubble(cv, bc, 150, 115, t, s.get("bubble", "?"))
                    cv.restore()
                    meta.setdefault("figs", []).append((bc[0] - 155, bc[1] - 120, bc[0] + 155, bc[1] + 120))
            else:
                for k, (x, y, sz) in enumerate(((170, 820, 34), (900, 760, 42), (130, 1250, 26), (950, 1180, 30), (985, 720, 22))):
                    self.sparkle(cv, (x, y), sz, t, k)
                if s.get("sub"):
                    T.block(cv, logt, "label", s["sub"], "marker", 76, 540, title_bottom + 40, GREEN, BG)
        elif typ == "split":
            box = self.bbox_side([s["top"]["form"], s["bottom"]["form"]])
            cols = (box[3] - box[1]) > 0.8 * (box[2] - box[0])  # tall poses (squat): side by side
            for k, side in enumerate(("top", "bottom")):
                sd = s[side]; J = self.pose(t, sd["form"], offset=k * 0.3); meta["joints"].append(J)
                y0 = 300 if k == 0 else 975
                bad = sd["form"] != "good"
                if cols:
                    x0 = 20 if k == 0 else 550
                    z, f, a = self.fit(("splitc", k), box, (x0 + 10, 600, x0 + 500, 1560))
                    lab_c, lab_y, mk = x0 + 230, 330, (x0 + 255, 470)
                else:
                    z, f, a = self.fit(("split", k), box, (50, y0 + 130, 1030, y0 + 600))
                    lab_c, lab_y, mk = 540, y0 + 20, (900, y0 + 60)
                pen = Pen(t, 12 / z * 1.1, INK, 2.0 / z)
                hl = [tuple(x) for x in self.ep["forms"][sd["form"]].get("highlight", [])]
                def draw(c, J=J, bad=bad, pen=pen, hl=hl):
                    self.floor(c, pen); self.mat(c, pen, box[0] + 10, box[2] - 10)
                    self.stick_side(c, pen, J, "worried" if bad else "happy", t, seed=k * 20, hl=hl, meta=meta,
                                    muscles=None if bad else scene_shown_muscles(self.ep, s),
                                    pulse=rep_concentric(t, self.ep["rep_period"], k * 0.3))
                self.world(cv, z * punch, f, a, draw)
                T.block(cv, logt, "label", sd["label"], "marker", 80, lab_c, lab_y, RED if bad else GREEN, BG, max_w=500)
                self.mark(cv, "bad" if bad else "good", mk, 46, ts - 0.15 - k * 0.35, 9 + k)
            dv = [(540, 330), (540, 1560)] if cols else [(60, 955), (1020, 955)]
            cv.drawPath(Pen(t, 6, 0xFF7C8796).poly(dv, 55), Pen(t, 6, 0xFF7C8796).paint())

    def pain(self, c, P, where, t):
        """Red zigzag at the JSON-given pain spot (front pose)."""
        hip, sh, neck = P["hip"], P["shoulder"], P["neck"]
        knee = P["legs"][1][0]
        spot = {"lower_back": (hip[0] + 30, hip[1] - 70), "hip": (hip[0] + 30, hip[1] - 20),
                "knee": (knee[0] + 28, knee[1] - 30), "shoulder": (sh[0] + 30, sh[1] - 10),
                "neck": (neck[0] + 26, neck[1] - 6), "hamstring": ((hip[0] + knee[0]) / 2 + 26, (hip[1] + knee[1]) / 2 - 20),
                "wrist": (P["arms"][0][1][0] - 70, P["arms"][0][1][1] - 30)}[where]
        zp = Pen(t, 9, RED)
        x, y = spot
        c.drawPath(zp.poly([(x, y), (x + 40, y + 22), (x + 15, y + 44), (x + 62, y + 66)], 5), zp.paint())
        c.drawPath(zp.poly([(x + 70, y - 6), (x + 92, y + 8)], 7), zp.paint())

    def cue_block(self, cv, logt, s, ts, t, tall=False):
        """Lower-third block under horizontal figures: big wrong/right cue + rep counter.
        Tall (standing) layout: the figure fills the lower third, cue sits under the floor, no counter."""
        y = 1478 if tall else 1250
        cue = s.get("cue")
        if cue and ts > 0.45:
            k = (ts - 0.45) / 0.25
            sc = back_out(k) if k < 1 else 1.0
            col = RED if cue.get("kind") == "bad" else GREEN
            cap = 84; iw = 70
            while True:  # one line only: shrink (not below 70 px cap) until it fits
                f_ = skia.Font(T.typeface("marker"), cap / T.cap_ratio("marker"))
                tw = f_.measureText(cue["text"])
                if tw + iw + 30 <= 1000 or cap <= 70:
                    break
                cap -= 2
            x0 = 540 - (tw + iw + 30) / 2
            cv.save(); cv.translate(540, y + cap / 2); cv.scale(sc, sc); cv.translate(-540, -(y + cap / 2))
            self.mark(cv, cue.get("kind", "bad"), (x0 + iw / 2, y + cap / 2), 30, 1.0, 21)
            cv.restore()
            T.block(cv, logt, "label", cue["text"], "marker", cap, x0 + iw + 30 + tw / 2, y, col, BG, scale=sc, max_w=tw + 2)
        if s.get("counter") and not tall:
            P = self.ep["rep_period"]
            n = int(math.floor((t - REP_DONE * P) / P) - math.floor((s["t0"] - REP_DONE * P) / P))
            if n >= 1:
                T.block(cv, logt, "label", f"REP {n}", "marker", 90, 540, 1410, INK, BG)
