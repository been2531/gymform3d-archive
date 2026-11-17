"""Style A: grey matte mannequin cut-out puppet on charcoal, ghost wrong pose,
continuous rep loop. Everything per episode comes from the JSON."""
from __future__ import annotations
import math
import skia
from .rig import Body, pushup_pose, hand_x_for, rep_phase
from .puppet import Puppet
from . import text as T

W, H = 1080, 1920
BG = 0xFF2E3033
KIND = {"bad": (0xFFFF6B6B, "x"), "good": (0xFF4ADE80, "check"), "tip": (0xFFFFC94D, "bolt")}
FLOOR_W, BALL_W = 1250.0, 130.0


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def back_out(x):
    x = min(max(x, 0.0), 1.0); c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


class StyleA:
    def __init__(self, ep):
        self.ep = ep; self.A = ep["A"]
        self.body = Body(H=820)
        self.pup = Puppet(self.body, self.A.get("palette", "grey"))
        self.hand_x = hand_x_for(self.body, FLOOR_W, BALL_W)
        self.segs = self.A["segments"]
        mid = self.pose(0.5, ep["forms"]["good"])
        self.focus = {
            "body": ((mid["ball"][0] + mid["head"][0]) / 2 + 20, (mid["hip"][1] + FLOOR_W) / 2 - 40),
            "hip": (mid["hip"][0] + 60, mid["hip"][1]),
            "shoulder": (mid["shoulder"][0] - 40, mid["shoulder"][1] + 10),
        }
        bgs = skia.Surface(W, H); c = bgs.getCanvas(); c.clear(BG)
        sh = skia.GradientShader.MakeRadial((540, 900), 1100, [0x00000000, 0x55000000], [0.55, 1.0])
        c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Shader=sh))
        self.bg_img = bgs.makeImageSnapshot()

    def pose(self, ph, form):
        return pushup_pose(self.body, ph, form, FLOOR_W, BALL_W, self.hand_x)

    def seg_at(self, t):
        for i, s in enumerate(self.segs):
            if s["t0"] <= t < s["t1"]:
                return i, s
        return len(self.segs) - 1, self.segs[-1]

    def form_at(self, t, key):
        i, s = self.seg_at(t)
        cur = self.ep["forms"][s[key]] if s.get(key) else None
        if i == 0 or not self.segs[i - 1].get(key) or cur is None:
            return cur, 1.0 if cur else 0.0
        prev = self.ep["forms"][self.segs[i - 1][key]]
        k = ease((t - s["t0"]) / 0.35)
        return {p: prev.get(p, 0) * (1 - k) + cur.get(p, 0) * k for p in set(prev) | set(cur)}, 1.0

    def camera(self, t):
        i, s = self.seg_at(t)
        def cam(seg):
            return seg["camera"]["zoom"], self.focus[seg["camera"]["focus"]]
        z1, f1 = cam(s)
        if i == 0:
            z, f = z1, f1
        else:
            z0, f0 = cam(self.segs[i - 1]); k = ease((t - s["t0"]) / 0.45)
            z = z0 + (z1 - z0) * k; f = (f0[0] + (f1[0] - f0[0]) * k, f0[1] + (f1[1] - f0[1]) * k)
        drift = 1.0 + 0.012 * math.sin(t * 0.9)  # never fully static
        return z * drift, f

    def events(self):
        ev = []
        P = self.ep["rep_period"]; t = 0.55 * P
        while t < self.ep["duration"]:
            ev.append((t, "thump")); t += P
        for s in self.segs:
            if s["t0"] > 0: ev.append((max(s["t0"] - 0.12, 0), "whoosh"))
            ev.append((s["t0"] + 0.05, "pop"))
        return ev

    def render(self, cv, t, meta):
        ep = self.ep
        cv.drawImage(self.bg_img, 0, 0)
        ph = rep_phase(t, ep["rep_period"])
        solid_form, _ = self.form_at(t, "solid")
        ghost_form, _ = self.form_at(t, "ghost")
        i, s = self.seg_at(t)
        J = self.pose(ph, solid_form)
        meta["joints"].append(J)
        z, f = self.camera(t)
        cv.save()
        cv.translate(540, 980); cv.scale(z, z); cv.translate(-f[0], -f[1])
        # floor + contact shadow
        cv.drawLine(-400, FLOOR_W + 2, 1600, FLOOR_W + 2, skia.Paint(AntiAlias=True, Color=0x40FFFFFF, StrokeWidth=3))
        shp = skia.Paint(AntiAlias=True, Color=0x66000000, MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 18))
        cv.drawOval(skia.Rect.MakeLTRB(J["ball"][0] - 40, FLOOR_W - 14, J["wrist"][0] + 90, FLOOR_W + 16), shp)
        if ghost_form is not None:
            Jg = self.pose(ph, ghost_form); meta["joints"].append(Jg)
            a = ease((t - s["t0"]) / 0.25) if s.get("ghost") and (i == 0 or self.segs[i-1].get("ghost") != s["ghost"]) else 1.0
            cv.saveLayerAlpha(None, int(255 * a))
            self.pup.draw(cv, Jg, "ghost", ghost=0xFFFF4D4D)
            cv.restore()
        self.pup.draw(cv, J)
        self.guide(cv, s, J, t)
        cv.restore()
        # text: hook (always, frame 0) + current label
        T.block(cv, meta["text"], "hook", self.A["hook"], "anton", 116, 540, 150, 0xFFFFFFFF, BG, max_w=1000)
        lab = s["label"]; col, icon = KIND[lab["kind"]]
        k = (t - s["t0"]) / 0.28
        sc = back_out(k) if k < 1 else 1.0
        sc = max(sc, 0.01)
        cap = 76
        size = cap / T.cap_ratio("anton")
        f_ = skia.Font(T.typeface("anton"), size)
        tw = f_.measureText(lab["text"]); iw = cap * 1.05
        x0 = 540 - (tw + iw + 26) / 2
        top = 380
        cv.save(); cv.translate(540, top + cap / 2); cv.scale(sc, sc); cv.translate(-540, -(top + cap / 2))
        self.icon(cv, icon, x0, top, cap, col)
        cv.restore()
        T.block(cv, meta["text"], "label", lab["text"], "anton", cap, x0 + iw + 26 + tw / 2, top, col, BG, scale=sc)

    def icon(self, cv, kind, x, y, s, col):
        p = skia.Paint(AntiAlias=True, Color=col, Style=skia.Paint.kStroke_Style, StrokeWidth=s * 0.2,
                       StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)
        path = skia.Path()
        if kind == "x":
            path.moveTo(x + s * .15, y + s * .1); path.lineTo(x + s * .9, y + s * .9)
            path.moveTo(x + s * .9, y + s * .1); path.lineTo(x + s * .15, y + s * .9)
        elif kind == "check":
            path.moveTo(x + s * .05, y + s * .55); path.lineTo(x + s * .38, y + s * .88); path.lineTo(x + s * 1.0, y + s * .08)
        else:
            path.moveTo(x + s * .6, y); path.lineTo(x + s * .2, y + s * .55); path.lineTo(x + s * .55, y + s * .55); path.lineTo(x + s * .35, y + s)
        cv.drawPath(path, p)

    def guide(self, cv, s, J, t):
        g = s.get("guide")
        if not g:
            return
        a = ease((t - s["t0"]) / 0.3)
        if g == "bent":
            p = skia.Paint(AntiAlias=True, Color=0xFFFF6B6B, Style=skia.Paint.kStroke_Style, StrokeWidth=7, StrokeCap=skia.Paint.kRound_Cap)
            p.setAlphaf(0.9 * a)
            path = skia.Path(); path.moveTo(*J["ankle"]); path.lineTo(*J["hip"]); path.lineTo(*J["shoulder"])
            cv.drawPath(path, p)
            for k in ("ankle", "hip", "shoulder"):
                cv.drawCircle(*J[k], 11, skia.Paint(AntiAlias=True, Color=0xFFFF6B6B, Alphaf=a))
        elif g == "line":
            ax, ay = J["ankle"]; sx, sy = J["shoulder"]
            dx, dy = sx - ax, sy - ay; n = math.hypot(dx, dy)
            e = (sx + dx / n * 140, sy + dy / n * 140)
            p = skia.Paint(AntiAlias=True, Color=0xFF4ADE80, Style=skia.Paint.kStroke_Style, StrokeWidth=7,
                           PathEffect=skia.DashPathEffect.Make([26, 16], -t * 120))
            p.setAlphaf(0.95 * a)
            cv.drawLine(ax - dx / n * 40, ay - dy / n * 40, e[0], e[1], p)
        elif g == "depth":
            y = FLOOR_W - 0.065 * self.body.H
            p = skia.Paint(AntiAlias=True, Color=0xFF4ADE80, Style=skia.Paint.kStroke_Style, StrokeWidth=6,
                           PathEffect=skia.DashPathEffect.Make([22, 14], -t * 100))
            p.setAlphaf(0.9 * a)
            cv.drawLine(J["hip"][0] + 40, y, J["elbow"][0] - 10, y, p)
