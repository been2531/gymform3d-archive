"""Style B v2: pen-line stickman, big expressive faces, props, auto-framing.
Exercise-agnostic: side poses come from rig.<exercise>_pose via POSERS, so a
new episode is a JSON file (plus one pose function per new exercise)."""
from __future__ import annotations
import math
import skia
from .rig import Body, pushup_pose, squat_pose, hand_x_for, rep_phase, add, d, two_bone
from . import text as T
from .style_b import Pen, ease, back_out, BG, INK, RED, GREEN

W, H = 1080, 1920
FLOOR = 1250.0
SAFE_BOTTOM = int(H * 0.85) - 40
MAT_FILL = 0xFFFBEBD0
PAPER = 0xFFFFFFFF
GOLD = 0xFFD79A00


class Poser:
    def __init__(self, body, ep):
        self.body = body; self.ep = ep; self.ex = ep.get("exercise", "pushup")
        if self.ex == "pushup":
            self.hand_x = hand_x_for(body, FLOOR, 130.0)

    def __call__(self, ph, form):
        if self.ex == "pushup":
            return pushup_pose(self.body, ph, form, FLOOR, 130.0, self.hand_x)
        if self.ex == "squat":
            return squat_pose(self.body, ph, form, FLOOR, 520.0)
        raise ValueError(f"no pose function for exercise {self.ex!r}")


class StyleB2:
    def __init__(self, ep):
        self.ep = ep; self.B = ep.get("B2") or ep["B"]; self.scenes = self.B["scenes"]
        self.body = Body(H=760)
        self.R = 0.105 * self.body.H  # head radius (world)
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
                for key in ("ball", "ankle", "knee", "hip", "shoulder", "elbow", "wrist", "fingers"):
                    xs.append(J[key][0]); ys.append(J[key][1])
                c = self.head_c(J); xs += [c[0] - self.R, c[0] + self.R]; ys += [c[1] - self.R, c[1] + self.R]
        ys.append(FLOOR + 30)
        return min(xs) - 40, min(ys) - 20, max(xs) + 40, max(ys)

    def fit(self, key, box, rect):
        """Zoom/focus so world box fits screen rect (x0,y0,x1,y1), bottom-aligned."""
        if key in self._fit_cache:
            return self._fit_cache[key]
        bx0, by0, bx1, by1 = box; rx0, ry0, rx1, ry1 = rect
        z = min((rx1 - rx0) / (bx1 - bx0), (ry1 - ry0) / (by1 - by0))
        focus = ((bx0 + bx1) / 2, by1); anchor = ((rx0 + rx1) / 2, ry1)
        self._fit_cache[key] = (z, focus, anchor)
        return self._fit_cache[key]

    def world(self, cv, z, focus, anchor, fn):
        cv.save(); cv.translate(*anchor); cv.scale(z, z); cv.translate(-focus[0], -focus[1])
        fn(cv); cv.restore()

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
    def stick_side(self, cv, pen, J, mood, t, seed=0):
        far = Pen(t, pen.w, 0xFF8A93A0, pen.amp)
        off = (9, -6)
        F = {k: (v[0] + off[0], v[1] + off[1]) if isinstance(v, tuple) else v for k, v in J.items()}
        cv.drawPath(far.poly([F["hip"], F["knee"], F["ankle"], F["ball"]], seed + 1), far.paint())
        cv.drawPath(far.poly([F["shoulder"], F["elbow"], F["wrist"], F["fingers"]], seed + 2), far.paint())
        cv.drawPath(pen.poly([J["hip"], J["knee"], J["ankle"], J["ball"]], seed + 3), pen.paint())
        cv.drawPath(pen.poly([J["hip"], J["shoulder"], J["neck"]], seed + 4), pen.paint())
        cv.drawPath(pen.poly([J["shoulder"], J["elbow"], J["wrist"], J["fingers"]], seed + 5), pen.paint())
        self.head(cv, pen, self.head_c(J), mood, t, seed + 6, look=(1.0, 0.3))

    def front_pose(self, base, t, kind):
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
            if kind == "flex":
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
        return {"hip": hip, "legs": legs, "shoulder": sh, "neck": neck, "head_c": hc, "arms": arms}

    def stick_front(self, cv, pen, P, mood, t, meta, kind, seed=0):
        hip, sh, neck = P["hip"], P["shoulder"], P["neck"]
        for i, (k, a, b) in enumerate(P["legs"]):
            cv.drawPath(pen.poly([hip, k, a, b], seed + i), pen.paint())
        cv.drawPath(pen.poly([hip, sh, neck], seed + 3), pen.paint())
        for i, (e, w) in enumerate(P["arms"]):
            cv.drawPath(pen.poly([sh, e, w], seed + 4 + i), pen.paint())
            cv.drawCircle(w[0], w[1], pen.w * (1.9 if kind == "flex" else 1.0), pen.paint(fill=True, color=PAPER))
            cv.drawCircle(w[0], w[1], pen.w * (1.9 if kind == "flex" else 1.0), pen.paint())
            if kind == "flex":
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
        mood_look = (0.0, 0.0)
        self.head(cv, pen, P["head_c"], mood, t, seed + 9, mood_look)
        if kind == "think":  # scratching hand stays in front of the head
            e, w = P["arms"][1]
            cv.drawPath(pen.poly([sh, e, w], seed + 5), pen.paint())
            cv.drawCircle(w[0], w[1], pen.w * 1.3, pen.paint(fill=True, color=PAPER)); cv.drawCircle(w[0], w[1], pen.w * 1.3, pen.paint())
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

    def guide(self, cv, kind, J, t):
        p = skia.Paint(AntiAlias=True, Color=GREEN, Style=skia.Paint.kStroke_Style, StrokeWidth=8,
                       PathEffect=skia.DashPathEffect.Make([28, 18], -t * 140))
        if kind == "line":
            ax, ay = J["ankle"]; sx, sy = J["shoulder"]; dx, dy = sx - ax, sy - ay; n = math.hypot(dx, dy)
            ox, oy = dy / n * 70, -dx / n * 70
            cv.drawLine(ax - dx / n * 60 + ox, ay - dy / n * 60 + oy, sx + dx / n * 120 + ox, sy + dy / n * 120 + oy, p)
        elif kind == "depth":
            y = J["knee"][1]
            cv.drawLine(J["hip"][0] - 120, y, J["knee"][0] + 80, y, p)
        elif kind == "torso":
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
            if s.get("stamp"): ev.append((s["t0"] + 0.9, "pop"))
            if s["type"] in ("reps", "split", "closeup"):
                P = self.ep["rep_period"]; t = 0.55 * P
                while t < s["t1"]:
                    if t > s["t0"]: ev.append((t, "tick"))
                    t += P
        return ev

    def pose(self, t, form, offset=0.0):
        return self.poser(rep_phase(t, self.ep["rep_period"], offset), self.ep["forms"][form])

    def render(self, cv, t, meta):
        cv.drawImage(self.bg_img, 0, 0)
        i, s = self.scene_at(t); ts = t - s["t0"]
        punch = 1.0 + 0.08 * (1 - ease(ts / 0.3))
        typ = s["type"]; logt = meta["text"]
        title_bottom = 150
        if s.get("title"):
            title_bottom = T.block(cv, logt, "title", s["title"], "marker", 112, 540, 150, INK, BG, max_w=1000)
        if typ in ("reps", "closeup"):
            J = self.pose(t, s["form"]); meta["joints"].append(J)
            box = self.bbox_side([s["form"]])
            z, f, a = self.fit(("solo", s["form"]), box, (40, 640, 1040, 1380))
            if typ == "closeup":
                fj = J[s.get("focus", "hip")]
                zz = z * (1.35 + 0.1 * ease(ts / 2.0))
                f = (fj[0] + s.get("dx", 0), fj[1] + s.get("dy", 0)); a = (540, 1080)
                z = zz
            pen = Pen(t, 13 / z * 1.15, INK, 2.0 / z * 1.1)
            def draw(c):
                self.floor(c, pen); self.mat(c, pen, box[0] + 10, box[2] - 10)
                self.stick_side(c, pen, J, s.get("face", "strain"), t)
                if typ == "closeup":
                    self.guide(c, s.get("guide", "line"), J, t)
                    if s.get("arrow"): self.arrow(c, J[s["arrow"]], t)
            self.world(cv, z * punch, f, a, draw)
            if s.get("stamp"):
                self.mark(cv, s["stamp"], (540, 1500), 85, ts - 0.9, 3)
            if s.get("note"):
                k = (ts - 0.5) / 0.25
                T.block(cv, logt, "label", s["note"], "marker", 76, 540, title_bottom + 40, GREEN, BG,
                        scale=back_out(k) if k < 1 else 1.0, alpha=1.0 if ts > 0.5 else 0.0)
        elif typ in ("think", "conclusion"):
            kind = "think" if typ == "think" else "flex"
            P0 = self.front_pose((540, FLOOR), 0.0, kind)
            ys = [P0["head_c"][1] - self.R] + [w[1] for _, w in P0["arms"]]
            top = min(ys) - 30
            z = (1560 - 600) / (FLOOR + 24 - top)  # character ~50% of frame height
            P = self.front_pose((540, FLOOR), t, kind)
            pen = Pen(t, 13 / z * 1.15, INK, 2.0 / z * 1.1)
            def draw(c):
                if typ == "conclusion":
                    self.mat(c, pen, 300, 780)
                else:
                    self.floor(c, pen, 250, 830)
                self.stick_front(c, pen, P, s.get("face", "worried"), t, meta, kind)
                if typ == "think":
                    hip = P["hip"]; zp = Pen(t, 9, RED)
                    c.drawPath(zp.poly([(hip[0] + 30, hip[1] - 70), (hip[0] + 70, hip[1] - 48), (hip[0] + 45, hip[1] - 26), (hip[0] + 92, hip[1] - 4)], 5), zp.paint())
            self.world(cv, z * punch, (540, FLOOR + 24), (540, 1560), draw)
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
                    z, f, a = self.fit(("splitc", k), box, (x0, 560, x0 + 510, 1560))
                    lab_c, lab_y, mk = x0 + 230, 330, (x0 + 255, 470)
                else:
                    z, f, a = self.fit(("split", k), box, (50, y0 + 130, 1030, y0 + 600))
                    lab_c, lab_y, mk = 540, y0 + 20, (900, y0 + 60)
                pen = Pen(t, 12 / z * 1.1, INK, 2.0 / z)
                def draw(c, J=J, bad=bad, pen=pen):
                    self.floor(c, pen); self.mat(c, pen, box[0] + 10, box[2] - 10)
                    self.stick_side(c, pen, J, "worried" if bad else "happy", t, seed=k * 20)
                self.world(cv, z * punch, f, a, draw)
                T.block(cv, logt, "label", sd["label"], "marker", 80, lab_c, lab_y, RED if bad else GREEN, BG, max_w=500)
                self.mark(cv, "bad" if bad else "good", mk, 46, ts - 0.15 - k * 0.35, 9 + k)
            dv = [(540, 330), (540, 1560)] if cols else [(60, 955), (1020, 955)]
            cv.drawPath(Pen(t, 6, 0xFF7C8796).poly(dv, 55), Pen(t, 6, 0xFF7C8796).paint())
