"""Style B: pen-line stickman on pastel blue, a new shot every 2-3 s.
Scene types are code; order, timing, text and forms come from the JSON."""
from __future__ import annotations
import math
import skia
from .rig import Body, pushup_pose, hand_x_for, rep_phase, add, d
from . import text as T

W, H = 1080, 1920
BG = 0xFFDCEEFA
INK = 0xFF15171A
RED = 0xFFB81B30
GREEN = 0xFF0B6A2C
FLOOR, BALL = 1250.0, 130.0


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def back_out(x):
    x = min(max(x, 0.0), 1.0); c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def _noise(i, seed):
    return math.sin(i * 12.9898 + seed * 78.233) * 43758.5453 % 1.0 - 0.5


class Pen:
    """Hand-drawn wobble: points jitter slightly, re-drawn 8x/s ("line boil")."""
    def __init__(self, t, width=11.0, color=INK, amp=2.0):
        self.boil = int(t * 8); self.w = width; self.c = color; self.amp = amp

    def paint(self, alpha=1.0, fill=False, color=None):
        p = skia.Paint(AntiAlias=True, Color=color or self.c)
        if not fill:
            p.setStyle(skia.Paint.kStroke_Style); p.setStrokeWidth(self.w)
            p.setStrokeCap(skia.Paint.kRound_Cap); p.setStrokeJoin(skia.Paint.kRound_Join)
        p.setAlphaf(alpha)
        return p

    def poly(self, pts, seed=0, closed=False):
        out = []
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            L = math.hypot(b[0] - a[0], b[1] - a[1]); n = max(int(L / 28), 1)
            nx, ny = (-(b[1] - a[1]) / (L + 1e-6), (b[0] - a[0]) / (L + 1e-6))
            for j in range(n):
                u = j / n
                o = self.amp * _noise(k * 31 + j, seed + self.boil * 7) * (1 if 0 < j else 0.3)
                out.append((a[0] + (b[0] - a[0]) * u + nx * o, a[1] + (b[1] - a[1]) * u + ny * o))
        out.append(pts[-1])
        p = skia.Path(); p.moveTo(*out[0])
        for q in out[1:]:
            p.lineTo(*q)
        if closed: p.close()
        return p

    def circle(self, c, r, seed=0):
        pts = []
        for i in range(25):
            a = 2 * math.pi * i / 24
            rr = r + self.amp * 0.8 * _noise(i % 24, seed + self.boil * 5)
            pts.append((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)))
        p = skia.Path(); p.moveTo(*pts[0])
        for q in pts[1:]:
            p.lineTo(*q)
        p.close()
        return p


class StyleB:
    def __init__(self, ep):
        self.ep = ep; self.B = ep["B"]; self.scenes = self.B["scenes"]
        self.body = Body(H=760)
        self.hand_x = hand_x_for(self.body, FLOOR, BALL)
        bgs = skia.Surface(W, H); c = bgs.getCanvas(); c.clear(BG)
        self.bg_img = bgs.makeImageSnapshot()

    # ---------- figures ----------
    def head_r(self):
        return 0.085 * self.body.H

    def face(self, cv, pen, c, r, facing, mood, seed):
        """facing: unit vector the face looks toward (side) or None (front)."""
        if facing is None:
            ex = [(c[0] - r * 0.32, c[1] - r * 0.12), (c[0] + r * 0.32, c[1] - r * 0.12)]; mc = (c[0], c[1] + r * 0.38)
            mw = r * 0.42
        else:
            fx, fy = facing
            ex = [(c[0] + fx * r * 0.55 - fy * r * 0.15, c[1] + fy * r * 0.55 + fx * r * 0.15)]
            mc = (c[0] + fx * r * 0.45 + fy * r * 0.45 * 0 - fy * -r * 0.3, c[1] + fy * r * 0.45 + r * 0.35)
            mw = r * 0.3
        for e in ex:
            cv.drawCircle(e[0], e[1], r * 0.11, pen.paint(fill=True))
        mp = pen.paint(); mp.setStrokeWidth(pen.w * 0.6)
        path = skia.Path()
        if mood == "happy":
            path.moveTo(mc[0] - mw / 2, mc[1] - 4); path.quadTo(mc[0], mc[1] + mw * 0.6, mc[0] + mw / 2, mc[1] - 4)
        elif mood == "strain":
            path.moveTo(mc[0] - mw / 2, mc[1]); path.lineTo(mc[0] - mw / 6, mc[1] - 6); path.lineTo(mc[0] + mw / 6, mc[1] + 6); path.lineTo(mc[0] + mw / 2, mc[1])
        else:  # worried
            path.moveTo(mc[0] - mw / 2, mc[1] + 6); path.quadTo(mc[0], mc[1] - mw * 0.35, mc[0] + mw / 2, mc[1] + 6)
        cv.drawPath(path, mp)

    def stick_side(self, cv, pen, J, mood, seed=0):
        """Push-up side view: one line per limb (far limbs lighter)."""
        far = Pen(0, pen.w, 0xFF8A93A0, pen.amp); far.boil = pen.boil
        off = (10, -6)
        F = {k: (v[0] + off[0], v[1] + off[1]) if isinstance(v, tuple) else v for k, v in J.items()}
        cv.drawPath(far.poly([F["hip"], F["knee"], F["ankle"], F["ball"]], seed + 1), far.paint())
        cv.drawPath(far.poly([F["shoulder"], F["elbow"], F["wrist"], F["fingers"]], seed + 2), far.paint())
        cv.drawPath(pen.poly([J["hip"], J["knee"], J["ankle"], J["ball"]], seed + 3), pen.paint())
        cv.drawPath(pen.poly([J["hip"], J["shoulder"], J["neck"]], seed + 4), pen.paint())
        cv.drawPath(pen.poly([J["shoulder"], J["elbow"], J["wrist"], J["fingers"]], seed + 5), pen.paint())
        r = self.head_r()
        a = math.radians(J["_head_ang"]); u = (math.cos(a), -math.sin(a))
        c = (J["neck"][0] + u[0] * r * 1.05, J["neck"][1] + u[1] * r * 1.05)
        cv.drawPath(pen.circle(c, r, seed + 6), pen.paint(fill=True, color=0xFFFFFFFF))
        cv.drawPath(pen.circle(c, r, seed + 6), pen.paint())
        # looks toward the floor/forward: rotate bone dir 90 deg clockwise
        self.face(cv, pen, c, r, (u[1] * -1 * -1 if False else -u[1] * -1, u[0]) if False else (math.sin(a) * 0 + u[0] * 0.6, 0.8), mood, seed)

    def front_pose(self, base, t, kind):
        L = self.body.L
        bob = math.sin(t * 2 * math.pi * 1.4)
        hip = (base[0], base[1] - L["thigh"] - L["shin"] - 0.02 * self.body.H + 4 * bob)
        legs = []
        for s in (-1, 1):
            knee = add(hip, d(-90 + s * 12), L["thigh"])
            ankle = add(knee, d(-90 + s * 6), L["shin"])
            ball = add(ankle, d(90 - s * 90), L["foot"] * 0.6)
            legs.append((knee, ankle, ball))
        sh = add(hip, d(90 + 2 * bob), L["torso"])
        neck = add(sh, d(90), L["neck"])
        head = add(neck, d(90 + (8 * math.sin(t * 5) if kind == "think" else 0)), L["head"])
        arms = []
        for s in (-1, 1):
            if kind == "think" and s == 1:
                wig = math.sin(t * 2 * math.pi * 3)
                el = add(sh, d(20 + 5 * wig), L["upper_arm"]); wr = add(el, d(120 + 12 * wig), L["forearm"])
            elif kind == "think":
                el = add(sh, d(-115), L["upper_arm"]); wr = add(el, d(-100), L["forearm"])
            elif kind == "flex":
                pump = math.sin(t * 2 * math.pi * 1.8)
                el = add(sh, d(90 - s * 85 + s * 4 * pump), L["upper_arm"]); wr = add(el, d(90 - s * 0 + s * 25 * pump * 0 - s * 10 + s * 12 * pump), L["forearm"])
            else:
                el = add(sh, d(-90 - s * 25), L["upper_arm"]); wr = add(el, d(-90 - s * 15), L["forearm"])
            fi = add(wr, d(math.degrees(math.atan2(-(wr[1] - el[1]), wr[0] - el[0]))), L["hand"] * 0.35)
            arms.append((el, wr, fi))
        return hip, legs, sh, neck, head, arms

    def stick_front(self, cv, pen, P, mood, seed=0, meta=None):
        hip, legs, sh, neck, head, arms = P
        L = self.body.L
        for i, (k, a, b) in enumerate(legs):
            cv.drawPath(pen.poly([hip, k, a, b], seed + i), pen.paint())
        cv.drawPath(pen.poly([hip, sh, neck], seed + 3), pen.paint())
        for i, (e, w, f) in enumerate(arms):
            cv.drawPath(pen.poly([sh, e, w], seed + 4 + i), pen.paint())
            cv.drawCircle(w[0], w[1], pen.w * 0.9, pen.paint(fill=True))
        r = self.head_r(); c = (neck[0], neck[1] - r * 1.05)
        cv.drawPath(pen.circle(c, r, seed + 9), pen.paint(fill=True, color=0xFFFFFFFF))
        cv.drawPath(pen.circle(c, r, seed + 9), pen.paint())
        self.face(cv, pen, c, r, None, mood, seed)
        if meta is not None:
            for (k, a, b), (e, w, f) in zip(legs, arms):
                meta["joints"].append({"hip": hip, "knee": k, "ankle": a, "shoulder": sh, "elbow": e, "wrist": w,
                                       "neck": neck, "head": head, "_partial": True})

    # ---------- marks ----------
    def mark(self, cv, kind, c, s, t_since, seed):
        k = back_out(t_since / 0.25) if t_since < 0.25 else 1.0
        if t_since < 0: return
        pen = Pen(t_since + seed, 18, RED if kind == "bad" else GREEN, 2.5)
        cv.save(); cv.translate(*c); cv.scale(k, k)
        if kind == "bad":
            cv.drawPath(pen.poly([(-s, -s), (s, s)], seed), pen.paint())
            cv.drawPath(pen.poly([(s, -s), (-s, s)], seed + 1), pen.paint())
        else:
            cv.drawPath(pen.poly([(-s, 0), (-s * 0.3, s * 0.7), (s, -s * 0.8)], seed), pen.paint())
        cv.restore()

    # ---------- scenes ----------
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
            if s.get("stamp"): ev.append((s["t0"] + 1.0, "pop"))
            if s["type"] in ("reps", "split", "closeup"):
                P = self.ep["rep_period"]; t = s["t0"] + 0.55 * P - (s["t0"] % P)
                while t < s["t1"]:
                    if t > s["t0"]: ev.append((t, "tick"))
                    t += P
        return ev

    def pose(self, t, form_name):
        return pushup_pose(self.body, rep_phase(t, self.ep["rep_period"]), self.ep["forms"][form_name], FLOOR, BALL, self.hand_x)

    def render(self, cv, t, meta):
        cv.drawImage(self.bg_img, 0, 0)
        i, s = self.scene_at(t); ts = t - s["t0"]
        punch = 1.0 + 0.10 * (1 - ease(ts / 0.3))
        pen = Pen(t)
        typ = s["type"]
        logt = meta["text"]
        if typ == "reps":
            J = self.pose(t, s["form"]); meta["joints"].append(J)
            self.world(cv, 1.22 * punch, (510, 1130), (540, 1000), lambda c: (self.floor(c, pen), self.stick_side(c, pen, J, s.get("face", "strain"))))
            if s.get("stamp"):
                self.mark(cv, s["stamp"], (540, 1330), 95, ts - 1.0, 3)
            T.block(cv, logt, "title", s["title"], "marker", 112, 540, 150, INK, BG, max_w=1000, scale=1.0)
        elif typ == "think":
            P = self.front_pose((540, 1450), t, "think")
            def draw(c):
                c.drawPath(pen.poly([(240, 1450), (840, 1450)], 77), pen.paint())
                self.stick_front(c, pen, P, s.get("face", "worried"), meta=meta)
                # pain zigzag at lower back + floating ?s
                hip = P[0]
                c.drawPath(Pen(t, 9, RED).poly([(hip[0] + 40, hip[1] - 60), (hip[0] + 75, hip[1] - 40), (hip[0] + 55, hip[1] - 20), (hip[0] + 95, hip[1])], 5), Pen(t, 9, RED).paint())
            self.world(cv, 1.2 * punch, (540, 1180), (540, 1150), draw)
            f = skia.Font(T.typeface("marker"), 120)
            for k, (x, y0) in enumerate([(300, 700), (760, 640), (840, 820)]):
                yy = y0 - 40 * ((ts * 0.8 + k * 0.33) % 1.0)
                cv.drawString("?", x, yy, f, Pen(t).paint(fill=True, alpha=min(1, ts * 4)))
            T.block(cv, logt, "title", s["title"], "marker", 112, 540, 200, INK, BG)
        elif typ == "split":
            for k, side in enumerate(("top", "bottom")):
                sd = s[side]; J = self.pose(t + k * 0.3, sd["form"]); meta["joints"].append(J)
                cy = 700 if k == 0 else 1300
                bad = sd["form"] != "good"
                def draw(c, J=J, bad=bad):
                    self.floor(c, pen); self.stick_side(c, pen, J, "strain" if bad else "happy", seed=k * 20)
                self.world(cv, 0.82 * punch, (500, 1120), (560, cy + 40), draw)
                col = RED if bad else GREEN
                T.block(cv, logt, "label", sd["label"], "marker", 76, 540, cy - 300, col, BG)
                self.mark(cv, "bad" if bad else "good", (930, cy - 255), 46, ts - 0.15 - k * 0.35, 9 + k)
            cv.drawPath(pen.poly([(60, 990), (1020, 990)], 55), Pen(t, 6, 0xFF7C8796).paint())
        elif typ == "closeup":
            J = self.pose(t, s["form"]); meta["joints"].append(J)
            f = {"hip": (J["hip"][0] + 40, 1140)}[s.get("focus", "hip")]
            def draw(c):
                self.floor(c, pen); self.stick_side(c, pen, J, "happy")
                ax, ay = J["ankle"]; sx, sy = J["shoulder"]; dx, dy = sx - ax, sy - ay; n = math.hypot(dx, dy)
                p = skia.Paint(AntiAlias=True, Color=GREEN, Style=skia.Paint.kStroke_Style, StrokeWidth=8,
                               PathEffect=skia.DashPathEffect.Make([28, 18], -t * 140))
                ox, oy = dy / n * 70, -dx / n * 70  # parallel, just above the back
                c.drawLine(ax - dx / n * 60 + ox, ay - dy / n * 60 + oy, sx + dx / n * 120 + ox, sy + dy / n * 120 + oy, p)
                # arrow to glutes
                hp = J["hip"]
                c.drawPath(Pen(t, 9, RED).poly([(hp[0] - 20, hp[1] - 230), (hp[0] - 5, hp[1] - 30)], 4), Pen(t, 9, RED).paint())
                c.drawPath(Pen(t, 9, RED).poly([(hp[0] - 40, hp[1] - 65), (hp[0] - 5, hp[1] - 25), (hp[0] + 28, hp[1] - 68)], 6), Pen(t, 9, RED).paint())
            self.world(cv, (1.45 + 0.12 * ease(ts / 2.0)) * punch, f, (540, 1060), draw)
            T.block(cv, logt, "title", s["title"], "marker", 112, 540, 150, INK, BG)
            if s.get("note"):
                T.block(cv, logt, "label", s["note"], "marker", 76, 540, 1390, GREEN, BG, scale=back_out((ts - 0.5) / 0.25) if ts < 0.75 else 1.0, alpha=1.0 if ts > 0.5 else 0.0)
        elif typ == "conclusion":
            P = self.front_pose((540, 1480), t, "flex")
            def draw(c):
                c.drawPath(pen.poly([(240, 1480), (840, 1480)], 78), pen.paint())
                self.stick_front(c, pen, P, s.get("face", "happy"), meta=meta)
                for k in range(4):  # sparkles
                    a = k * math.pi / 2 + t * 2; r = 300 + 20 * math.sin(t * 6 + k)
                    cx, cy = 540 + r * math.cos(a), 1080 + 0.6 * r * math.sin(a)
                    sp = Pen(t, 7, 0xFFE0A100)
                    c.drawPath(sp.poly([(cx - 18, cy), (cx + 18, cy)], k), sp.paint()); c.drawPath(sp.poly([(cx, cy - 18), (cx, cy + 18)], k + 9), sp.paint())
            self.world(cv, 1.15 * punch, (540, 1180), (540, 1180), draw)
            T.block(cv, logt, "title", s["title"], "marker", 112, 540, 150, INK, BG)
            if s.get("sub"):
                T.block(cv, logt, "label", s["sub"], "marker", 76, 540, 560, GREEN, BG)

    def world(self, cv, z, focus, anchor, fn):
        cv.save(); cv.translate(*anchor); cv.scale(z, z); cv.translate(-focus[0], -focus[1])
        fn(cv); cv.restore()

    def floor(self, cv, pen):
        cv.drawPath(pen.poly([(40, FLOOR + 4), (1000, FLOOR + 4)], 91), Pen(0, 6, 0xFF7C8796).paint())
