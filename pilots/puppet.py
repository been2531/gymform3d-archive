"""Code-drawn cut-out puppet (style A): per-part vector shapes with pivots
at the rig joints, soft matte shading, outline, black shorts. One palette
dict switches the look; geometry never changes."""
from __future__ import annotations
import math
import skia
from .rig import dist, ang

PALETTES = {
    "grey":      dict(base=0xFFC9CCD1, shadow=0xFF8E939B, hi=0xFFEDEFF2, line=0xFF16171A, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033),
    "porcelain": dict(base=0xFFF3EFE8, shadow=0xFFBDB4A8, hi=0xFFFFFFFF, line=0xFF1A1918, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033),
    "coral":     dict(base=0xFFF28B74, shadow=0xFFC0564A, hi=0xFFFFC2B0, line=0xFF2A1512, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033),
    "mint":      dict(base=0xFF6FD3C1, shadow=0xFF2F9387, hi=0xFFB9F2E6, line=0xFF0F2622, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033),
    "cobalt":    dict(base=0xFF4C7DF0, shadow=0xFF2647A8, hi=0xFF9DB9FF, line=0xFF0B1330, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFFEDEBE6),
    "wood":      dict(base=0xFFD9AE78, shadow=0xFF9C6B3C, hi=0xFFF2D3A6, line=0xFF2A1A0C, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033, feel="wood"),
    "chrome":    dict(base=0xFFB8BEC8, shadow=0xFF5D636E, hi=0xFFE9EEF5, line=0xFF101114, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033, feel="chrome"),
    "lowpoly":   dict(base=0xFFBFC4CB, shadow=0xFF7D838C, hi=0xFFE6E9ED, line=0xFF16171A, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033, feel="lowpoly"),
    "ecorche":   dict(base=0xFFB8463C, shadow=0xFF7A231D, hi=0xFFE07A68, line=0xFF2A0D0A, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033, feel="ecorche"),
    "vinyl":     dict(base=0xFFF6F1E7, shadow=0xFFCFC3B0, hi=0xFFFFFFFF, line=0xFF1A1918, shorts=0xFF1E1F22, shorts_hi=0xFF45484E, bg=0xFF8FC6E8, feel="vinyl"),
    "silhouette": dict(base=0xFF0E0F11, shadow=0xFF0E0F11, hi=0xFF7FE3FF, line=0xFF050506, shorts=0xFF0E0F11, shorts_hi=0xFF2A2C30, bg=0xFF2E3033, feel="silhouette"),
    "boxer_grey": dict(base=0xFFC9CCD1, shadow=0xFF8E939B, hi=0xFFEDEFF2, line=0xFF16171A, shorts=0xFF6B1E2A, shorts_hi=0xFF9C3442, top=0xFF2B2D31, top_hi=0xFF4D5056, hat=0xFF2B2D31, hat_hi=0xFF4D5056, bg=0xFF2E3033, outfit="boxer"),
    "boxer_tan": dict(base=0xFFD6A27C, shadow=0xFF9C6A4A, hi=0xFFF0C8A6, line=0xFF20140E, shorts=0xFF6B1E2A, shorts_hi=0xFF9C3442, top=0xFF2B2D31, top_hi=0xFF4D5056, hat=0xFF2B2D31, hat_hi=0xFF4D5056, bg=0xFF2E3033, outfit="boxer"),
    "boxer_deep": dict(base=0xFF8A5A3E, shadow=0xFF5A3624, hi=0xFFB98463, line=0xFF140C08, shorts=0xFF6B1E2A, shorts_hi=0xFF9C3442, top=0xFF2B2D31, top_hi=0xFF4D5056, hat=0xFF2B2D31, hat_hi=0xFF4D5056, bg=0xFFE9E6E1, outfit="boxer"),
    "lilac":     dict(base=0xFFB9A3F0, shadow=0xFF7C64C2, hi=0xFFE3D9FF, line=0xFF1E1530, shorts=0xFF141518, shorts_hi=0xFF3A3D43, bg=0xFF2E3033),
}


def _cr(p0, p1, p2, p3, s):
    return 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3)


def _profile(ctrl, t):
    """Catmull-Rom through (t, w) control points (t sorted)."""
    if t <= ctrl[0][0]:
        return ctrl[0][1]
    if t >= ctrl[-1][0]:
        return ctrl[-1][1]
    for i in range(len(ctrl) - 1):
        if ctrl[i][0] <= t <= ctrl[i + 1][0]:
            p0 = ctrl[max(i - 1, 0)][1]; p1 = ctrl[i][1]; p2 = ctrl[i + 1][1]; p3 = ctrl[min(i + 2, len(ctrl) - 1)][1]
            s = (t - ctrl[i][0]) / (ctrl[i + 1][0] - ctrl[i][0])
            return max(_cr(p0, p1, p2, p3, s), 0.0)
    return ctrl[-1][1]


def part_path(A, B, top, bot, cap0=0.5, cap1=0.5, n=40):
    """Closed outline around bone A->B. top/bot: [(t, w)] half-widths in px
    (top = left of travel direction on screen). Elliptical caps beyond ends."""
    Lb = max(dist(A, B), 1e-6)
    ux, uy = (B[0] - A[0]) / Lb, (B[1] - A[1]) / Lb
    nx, ny = uy, -ux
    t0 = top[0][0]; t1 = top[-1][0]
    w_s = (_profile(top, t0) + _profile(bot, t0)) / 2
    w_e = (_profile(top, t1) + _profile(bot, t1)) / 2
    c0 = cap0 * w_s / Lb; c1 = cap1 * w_e / Lb
    ts = [t0 - c0 + (t1 + c1 - (t0 - c0)) * i / (n - 1) for i in range(n)]

    def w_at(prof, t):
        if t < t0:
            k = (t0 - t) / c0 if c0 > 0 else 1
            return _profile(prof, t0) * math.sqrt(max(1 - k * k, 0))
        if t > t1:
            k = (t - t1) / c1 if c1 > 0 else 1
            return _profile(prof, t1) * math.sqrt(max(1 - k * k, 0))
        return _profile(prof, t)

    p = skia.Path()
    pts_top = [(A[0] + ux * Lb * t + nx * w_at(top, t), A[1] + uy * Lb * t + ny * w_at(top, t)) for t in ts]
    pts_bot = [(A[0] + ux * Lb * t - nx * w_at(bot, t), A[1] + uy * Lb * t - ny * w_at(bot, t)) for t in ts]
    pts = pts_top + pts_bot[::-1]
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    p.close()
    return p


def local(A, B, t, off):
    Lb = max(dist(A, B), 1e-6)
    ux, uy = (B[0] - A[0]) / Lb, (B[1] - A[1]) / Lb
    return (A[0] + ux * Lb * t + uy * off, A[1] + uy * Lb * t - ux * off)


def curve(A, B, pts):
    p = skia.Path()
    q = [local(A, B, t, o) for t, o in pts]
    p.moveTo(*q[0])
    if len(q) == 3:
        p.quadTo(*q[1], *q[2])
    else:
        for r in q[1:]:
            p.lineTo(*r)
    return p


def _darken(c, k):
    a = (c >> 24) & 255; r = (c >> 16) & 255; g = (c >> 8) & 255; b = c & 255
    return (a << 24) | (int(r * k) << 16) | (int(g * k) << 8) | int(b * k)


class Puppet:
    def __init__(self, body, palette="grey"):
        self.body = body
        self.pal = PALETTES[palette] if isinstance(palette, str) else palette
        self.H = body.H

    # ---- shapes (geometry only; widths are fractions of height H) ----
    def shapes(self, J, far=False):
        H = self.H
        limb = lambda v: v * H * 1.28
        w = limb
        S = {}
        S["upper_arm"] = (part_path(J["shoulder"], J["elbow"],
                                    [(0, w(.048)), (.25, w(.050)), (.55, w(.046)), (1, w(.030))],
                                    [(0, w(.044)), (.3, w(.040)), (.6, w(.042)), (1, w(.030))], .8, .9),
                          [((J["shoulder"], J["elbow"]), [(.05, -w(.02)), (.30, w(.0)), (.42, w(.03))]),
                           ((J["shoulder"], J["elbow"]), [(.45, w(.03)), (.7, w(.005)), (.95, w(.012))])])
        S["forearm"] = (part_path(J["elbow"], J["wrist"],
                                  [(0, w(.032)), (.25, w(.038)), (1, w(.021))],
                                  [(0, w(.030)), (.3, w(.034)), (1, w(.021))], .8, .6),
                        [((J["elbow"], J["wrist"]), [(.1, w(.012)), (.4, w(.018)), (.8, w(.006))])])
        S["hand"] = (part_path(J["wrist"], J["fingers"],
                               [(0, w(.020)), (.4, w(.022)), (1, w(.012))],
                               [(0, w(.016)), (.5, w(.012)), (1, w(.006))], .5, 1.0), [])
        S["thigh"] = (part_path(J["hip"], J["knee"],
                                [(0, w(.062)), (.35, w(.060)), (.75, w(.045)), (1, w(.034))],
                                [(0, w(.058)), (.4, w(.056)), (.8, w(.040)), (1, w(.034))], .6, .8),
                      [((J["hip"], J["knee"]), [(.45, w(.02)), (.7, w(.005)), (.92, -w(.01))])])
        S["shin"] = (part_path(J["knee"], J["ankle"],
                               [(0, w(.034)), (.3, w(.040)), (.55, w(.034)), (1, w(.020))],
                               [(0, w(.032)), (.4, w(.028)), (1, w(.019))], .8, .6),
                     [((J["knee"], J["ankle"]), [(.12, w(.03)), (.32, w(.034)), (.6, w(.012))])])
        S["foot"] = (part_path(J["ankle"], J["ball"],
                               [(-.15, w(.022)), (.3, w(.024)), (1, w(.016))],
                               [(-.15, w(.030)), (.4, w(.022)), (1, w(.014))], 1.0, 1.0), [])
        tw = lambda v: v * H * 1.12
        S["torso"] = (part_path(J["hip"], J["shoulder"],
                                [(-.08, tw(.060)), (.2, tw(.060)), (.5, tw(.058)), (.82, tw(.074)), (1.02, tw(.060))],
                                [(-.08, tw(.062)), (.2, tw(.066)), (.45, tw(.060)), (.75, tw(.080)), (.95, tw(.074)), (1.02, tw(.050))],
                                .5, .6),
                      [((J["hip"], J["shoulder"]), [(.62, -tw(.074)), (.78, -tw(.054)), (.95, -tw(.050))]),  # pec line
                       ((J["hip"], J["shoulder"]), [(.62, tw(.030)), (.80, tw(.050)), (.98, tw(.040))]),   # lat / scapula
                       ((J["hip"], J["shoulder"]), [(.25, -tw(.05)), (.38, -tw(.054)), (.5, -tw(.052))])])  # abs edge
        toe = (J["ball"][0] + 0.05 * H, J["ball"][1] + 0.004 * H)
        S["toes"] = (part_path(J["ball"], toe, [(0, w(.012)), (1, w(.008))], [(0, w(.010)), (1, w(.007))], 1.0, 1.0), [])
        S["neck"] = (part_path(J["shoulder"], J["head"],
                               [(0, w(.035)), (.6, w(.030)), (1, w(.028))],
                               [(0, w(.035)), (.6, w(.028)), (1, w(.026))], .3, .3), [])
        return S

    def head_path(self, J):
        H = self.H
        c = J["head"]
        a = J["_head_ang"] if "_head_ang" in J else ang(J["neck"], J["head"])
        p = skia.Path()
        p.addOval(skia.Rect.MakeXYWH(-0.058 * H, -0.068 * H, 0.122 * H, 0.130 * H))
        jaw = skia.Path()
        jaw.addOval(skia.Rect.MakeXYWH(-0.02 * H, -0.035 * H, 0.08 * H, 0.08 * H))
        p = skia.Op(p, jaw, skia.PathOp.kUnion_PathOp) or p
        m = skia.Matrix()
        m.setRotate(-(a - 90.0))
        m.postTranslate(c[0], c[1])
        # head local "up" = bone direction; local +y toward chin side
        p.transform(m)
        return p

    # ---- painting ----
    def _paint(self, color, style="fill", width=0.0, blur=0.0, alpha=None):
        p = skia.Paint(AntiAlias=True, Color=color)
        if style == "stroke":
            p.setStyle(skia.Paint.kStroke_Style); p.setStrokeWidth(width)
            p.setStrokeCap(skia.Paint.kRound_Cap); p.setStrokeJoin(skia.Paint.kRound_Join)
        if blur:
            p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
        if alpha is not None:
            p.setAlphaf(alpha)
        return p

    def _shade(self, cv, path, base, shadow, hi, lines=(), clip_extra=None):
        H = self.H
        cv.drawPath(path, self._paint(shadow))
        cv.save(); cv.clipPath(path, doAntiAlias=True)
        lit = skia.Path(path); lit.offset(-0.010 * H, -0.016 * H)
        cv.drawPath(lit, self._paint(base, blur=0.010 * H))
        sp = skia.Path(path); sp.offset(-0.022 * H, -0.030 * H)
        cv.drawPath(sp, self._paint(hi, blur=0.016 * H, alpha=0.55))
        for A_B, pts in lines:
            cv.drawPath(curve(A_B[0], A_B[1], pts), self._paint(shadow, "stroke", 0.008 * H, blur=0.004 * H, alpha=0.75))
        if clip_extra:
            clip_extra(cv)
        cv.restore()

    def draw(self, cv, J, mode="solid", ghost=0xFFFF5A5A, far_offset=(0.020, -0.012)):
        pal = self.pal; H = self.H
        feel = pal.get("feel", "matte")
        line_w = 0.0065 * H * (1.8 if feel == "vinyl" else 1.0)
        if mode == "ghost":
            cv.saveLayerAlpha(None, int(255 * 0.55))
        Jf = {k: ((v[0] + far_offset[0] * H, v[1] + far_offset[1] * H) if isinstance(v, tuple) else v) for k, v in J.items()}
        Sf = self.shapes(Jf); S = self.shapes(J)
        order = [
            ("far", ["toes", "foot", "shin", "thigh"]), ("far", ["hand", "forearm", "upper_arm"]),
            ("near", ["torso"]), ("near", ["toes", "foot", "shin", "thigh"]), ("near", ["neck"]),
            ("head", ["head"]), ("near", ["hand", "forearm", "upper_arm"]),
        ]
        boxer = pal.get("outfit") == "boxer" and mode != "ghost"
        if boxer:
            order = [(sd, (["forearm", "upper_arm", "hand"] if p == ["hand", "forearm", "upper_arm"] else p)) for sd, p in order]
        for side, parts in order:
            src = Sf if side == "far" else S
            k = 0.80 if side == "far" else 1.0
            JJ = Jf if side == "far" else J
            for name in parts:
                if name == "head":
                    path, lines = self.head_path(J), []
                elif name == "hand" and boxer:
                    c = local(JJ["wrist"], JJ["fingers"], 0.45, 0.012 * H)
                    path = skia.Path(); path.addOval(skia.Rect.MakeXYWH(c[0] - 0.062 * H, c[1] - 0.05 * H, 0.124 * H, 0.10 * H)); lines = []
                else:
                    path, lines = src[name]
                cloth = None
                if name == "torso":
                    cloth = [(self._shorts_clip(JJ["hip"], JJ["shoulder"], 0.26, extend_back=True), "shorts")]
                    if boxer:
                        cloth.append((self._band(JJ["hip"], JJ["shoulder"], 0.26, 0.86), "top"))
                elif name == "thigh":
                    cloth = [(self._shorts_clip(JJ["hip"], JJ["knee"], 0.42 if not boxer else 0.5), "shorts")]
                elif name == "head" and boxer:
                    cap = self._cap(J)
                    r_ = math.radians(J.get("_head_ang", 0)); ux_, uy_ = math.cos(r_), -math.sin(r_)
                    cuff = self._band(J["head"], (J["head"][0] + ux_ * H, J["head"][1] + uy_ * H), -0.012, 0.016)
                    cloth = [(cap, "hat"), (cuff, "cuff")]
                elif name == "hand" and boxer:
                    cloth = [(path, "glove")]
                self._fill_part(cv, path, lines, mode, ghost, k, line_w, cloth, name, JJ)
                if feel == "wood" and mode != "ghost" and name in ("upper_arm", "forearm", "shin"):
                    jn = {"upper_arm": "shoulder", "forearm": "elbow", "shin": "knee"}[name]
                    self._ball(cv, JJ[jn], 0.034 * H, k, line_w)
        if mode == "ghost":
            cv.restore()

    def _ball(self, cv, c, r, k, line_w):
        pal = self.pal
        p = skia.Path(); p.addCircle(c[0], c[1], r)
        self._shade(cv, p, _darken(pal["base"], k * 0.9), _darken(pal["shadow"], k), _darken(pal["hi"], k), [])
        cv.drawPath(p, self._paint(pal["line"], "stroke", line_w))

    def _band(self, A, B, t0, t1):
        Lb = dist(A, B); ux, uy = (B[0] - A[0]) / Lb, (B[1] - A[1]) / Lb; nx, ny = uy, -ux; R = 0.3 * self.H
        pa = (A[0] + ux * Lb * t0, A[1] + uy * Lb * t0); pb = (A[0] + ux * Lb * t1, A[1] + uy * Lb * t1)
        p = skia.Path()
        p.moveTo(pa[0] + nx * R, pa[1] + ny * R); p.lineTo(pb[0] + nx * R, pb[1] + ny * R)
        p.lineTo(pb[0] - nx * R, pb[1] - ny * R); p.lineTo(pa[0] - nx * R, pa[1] - ny * R); p.close()
        return p

    def _cap(self, J):
        """Beanie: upper part of the head (away from the face direction)."""
        H = self.H; c = J["head"]
        a = J.get("_head_ang", ang(J["neck"], J["head"]))
        r = math.radians(a)
        ux, uy = math.cos(r), -math.sin(r)
        # beanie covers the half of the head on the crown side (along bone dir) plus back
        base = (c[0] - ux * 0.012 * H, c[1] - uy * 0.012 * H)
        nx, ny = uy, -ux
        R = 0.2 * H
        p = skia.Path()
        p.moveTo(base[0] + nx * R, base[1] + ny * R); p.lineTo(base[0] + nx * R + ux * R, base[1] + ny * R + uy * R)
        p.lineTo(base[0] - nx * R + ux * R, base[1] - ny * R + uy * R); p.lineTo(base[0] - nx * R, base[1] - ny * R); p.close()
        return p

    def _shorts_clip(self, A, B, tmax, extend_back=False):
        return self._band(A, B, -0.6 if extend_back else -0.3, tmax)

    # ---- per-feel surface treatments ----
    def _surface(self, cv, path, lines, k, name, J):
        pal = self.pal; H = self.H; feel = pal.get("feel", "matte")
        base, shadow, hi = _darken(pal["base"], k), _darken(pal["shadow"], k), _darken(pal["hi"], k)
        if feel in ("matte", "vinyl", "wood"):
            self._shade(cv, path, base, shadow, hi, lines if feel != "vinyl" else [])
            cv.save(); cv.clipPath(path, doAntiAlias=True)
            if feel == "vinyl":
                g = skia.Path(path); g.offset(-0.03 * H, -0.04 * H)
                cv.drawPath(g, self._paint(0xFFFFFFFF, "stroke", 0.012 * H, blur=0.006 * H, alpha=0.7))
            if feel == "wood":
                b = path.getBounds()
                for j in range(int(b.width() / 9) + 2):
                    x = b.left() + j * 9
                    cv.drawLine(x, b.top(), x + 0.4 * b.height(), b.bottom(), self._paint(shadow, "stroke", 1.6, alpha=0.25))
            cv.restore()
        elif feel == "chrome":
            cv.drawPath(path, self._paint(shadow))
            cv.save(); cv.clipPath(path, doAntiAlias=True)
            b = path.getBounds()
            shader = skia.GradientShader.MakeLinear([(b.left(), b.top()), (b.left(), b.bottom())],
                                                    [hi, 0xFFFFFFFF, base, 0xFF2A2D33, shadow, hi],
                                                    [0.0, 0.18, 0.4, 0.55, 0.8, 1.0])
            cv.drawPath(path, skia.Paint(AntiAlias=True, Shader=shader))
            g = skia.Path(path); g.offset(-0.012 * H, -0.02 * H)
            cv.drawPath(g, self._paint(0xFFFFFFFF, "stroke", 0.006 * H, blur=0.002 * H, alpha=0.9))
            cv.restore()
        elif feel == "lowpoly":
            n = path.countPoints(); pts = path.getPoints(n)
            half = n // 2; top = pts[:half]; bot = pts[half:][::-1]
            step = 4
            cv.save(); cv.clipPath(path, doAntiAlias=True)
            cv.drawPath(path, self._paint(base))
            for j in range(0, half - step, step):
                a, b2 = top[j], top[min(j + step, half - 1)]
                c, dd = bot[j], bot[min(j + step, half - 1)]
                mid = ((a.x() + c.x()) / 2, (a.y() + c.y()) / 2)
                mid2 = ((b2.x() + dd.x()) / 2, (b2.y() + dd.y()) / 2)
                for tri, col in (((a, b2, mid2), hi), ((a, mid, mid2), base), ((c, dd, mid2), shadow), ((c, mid, mid2), _darken(base, 0.9))):
                    p = skia.Path(); q = [(t.x(), t.y()) if hasattr(t, "x") else t for t in tri]
                    p.moveTo(*q[0]); p.lineTo(*q[1]); p.lineTo(*q[2]); p.close()
                    cv.drawPath(p, self._paint(col))
                    cv.drawPath(p, self._paint(_darken(col, 0.85), "stroke", 1.2))
            cv.restore()
        elif feel == "ecorche":
            cv.drawPath(path, self._paint(base))
            cv.save(); cv.clipPath(path, doAntiAlias=True)
            n = path.countPoints(); pts = path.getPoints(n); half = n // 2
            top = pts[:half]; bot = pts[half:][::-1]
            for f in [i / 11 for i in range(1, 11)]:
                fp = skia.Path()
                for j in range(half):
                    x = top[j].x() * (1 - f) + bot[j].x() * f; y = top[j].y() * (1 - f) + bot[j].y() * f
                    (fp.moveTo if j == 0 else fp.lineTo)(x, y)
                cv.drawPath(fp, self._paint(hi if int(f * 11) % 2 else shadow, "stroke", 2.2, alpha=0.8))
            sp = skia.Path(path); sp.offset(0.012 * H, 0.018 * H)
            cv.drawPath(sp, self._paint(0xFF2A0F0C, blur=0.012 * H, alpha=0.0))
            cv.restore()
            # tendon caps (pale) near distal ends
            for A_B, pts2 in lines:
                cv.drawPath(curve(A_B[0], A_B[1], pts2), self._paint(0xFFF2E6D8, "stroke", 0.006 * H, alpha=0.8))
        elif feel == "silhouette":
            cv.drawPath(path, self._paint(base))
            cv.save(); cv.clipPath(path, doAntiAlias=True)
            cv.drawPath(path, self._paint(hi, "stroke", 0.03 * H, blur=0.004 * H))
            g = skia.Path(path); g.offset(0.012 * H, 0.016 * H)
            cv.drawPath(g, self._paint(base, blur=0.003 * H))
            cv.restore()

    def _fill_part(self, cv, path, lines, mode, ghost, k, line_w, cloth, name="", J=None):
        pal = self.pal
        if mode == "ghost":
            cv.drawPath(path, self._paint(ghost))
            cv.drawPath(path, self._paint(_darken(ghost, 0.55), "stroke", line_w))
            return
        self._surface(cv, path, lines, k, name, J)
        for clip, kind in (cloth or []):
            sp = skia.Op(path, clip, skia.PathOp.kIntersect_PathOp)
            if sp is None:
                continue
            col = {"shorts": (pal["shorts"], pal["shorts_hi"]), "top": (pal.get("top", 0xFF2B2D31), pal.get("top_hi", 0xFF4A4D53)),
                   "hat": (pal.get("hat", 0xFF2B2D31), pal.get("hat_hi", 0xFF4A4D53)), "glove": (0xFF0B0B0D, 0xFF5A5D66), "cuff": (pal.get("hat_hi", 0xFF4A4D53), 0xFF5E6168)}[kind]
            self._shade(cv, sp, _darken(col[1], k * 0.85), _darken(col[0], k), _darken(col[1], k), [])
            if kind == "glove":
                cv.save(); cv.clipPath(sp, doAntiAlias=True)
                g = skia.Path(sp); g.offset(-0.01 * self.H, -0.016 * self.H)
                cv.drawPath(g, self._paint(0xFFFFFFFF, "stroke", 0.008 * self.H, blur=0.003 * self.H, alpha=0.8))
                cv.restore()
            cv.drawPath(sp, self._paint(pal["line"], "stroke", line_w * 0.8))
        cv.drawPath(path, self._paint(pal["line"], "stroke", line_w))
