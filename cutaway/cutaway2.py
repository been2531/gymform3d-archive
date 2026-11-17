"""Cutaway reel renderer v2 (GymForm 3D).

Rewrite of Pickle's F1 cutaway (gfr/cutaway.py, kept for reference) around
*registered* layers so the window and the final full-frame reveal never jump.

Timeline per topic JSON (see topics/*.json):
  0.00  skin plate already pushing in + breathing; hook type on screen
  0.12  window starts opening on the first inner layer (fully open by ~0.45 s)
  shots iris-wipe to the next layer from the label target (punch zoom + flash)
  macro shot: camera dives into the macro box, macro crossfades over the bone
  counter shot: big sourced number + caption (lower middle)
  bloom: window grows past the frame edges -> registered bone layer full screen

Text rules (phone legibility): one label at a time, <=3 words, Anton, white on a
dark pill or thick dark stroke, labels at the top band (~9 %), stats in the
lower middle, nothing in the bottom 15 %.  Label cap height >= 70 px, hook cap
height >= 110 px (Anton cap height = 0.86 x font size).
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "assets", "fonts", "Anton-Regular.ttf")
CAP_RATIO = 0.86          # Anton cap height / font size (measured on "H")
W, H = 1080, 1920
LIME = (180, 255, 57)
SAFE_BOTTOM = int(H * 0.85)
C_SAFE_WIN_BOTTOM = SAFE_BOTTOM - 20


# ------------------------------------------------------------------ easing ---
def clamp(v, a=0.0, b=1.0):
    return a if v < a else b if v > b else v


def inv(a, b, v):
    return 1.0 if b == a else clamp((v - a) / (b - a))


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_io(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def back_out(t, s=1.6):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def lerp(a, b, t):
    return a + (b - a) * t


# ------------------------------------------------------------------- text ----
def _font(size):
    return ImageFont.truetype(FONT, int(size))


class TextSprite:
    """Pre-rendered text block. `fill` is the glyph mask (for QA), `rgba` the
    composite (pill or stroke + white glyphs)."""

    def __init__(self, lines, size, style="pill", align="left", max_w=940, pad=(34, 22), min_size=None, pill_alpha=222):
        self.lines = [l for l in lines if l]
        size = int(size)
        f = _font(size)
        while min_size and size > min_size and max(f.getlength(l) for l in self.lines) > max_w:
            size -= 2
            f = _font(size)
        self.size = size
        self.cap = CAP_RATIO * size
        self.fits = max(f.getlength(l) for l in self.lines) <= max_w
        stroke = max(6, int(size * 0.07)) if style == "stroke" else 0
        lead = int(size * 1.02)
        tw = int(max(f.getlength(l) for l in self.lines)) + 2 * stroke
        asc = f.getbbox("H")[1]
        th = int(lead * (len(self.lines) - 1) + self.cap) + 2 * stroke
        px, py = pad if style == "pill" else (stroke + 6, stroke + 6)
        bw, bh = tw + 2 * px, th + 2 * py
        self.w, self.h = bw, bh
        img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        fillm = Image.new("L", (bw, bh), 0)
        d = ImageDraw.Draw(img)
        dm = ImageDraw.Draw(fillm)
        if style == "pill":
            d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=min(22, bh // 4), fill=(7, 9, 11, int(pill_alpha)))
            d.rounded_rectangle([0, 0, 9, bh - 1], radius=4, fill=LIME + (255,))
        for i, l in enumerate(self.lines):
            lw = f.getlength(l)
            x = px + stroke + ((tw - 2 * stroke - lw) / 2 if align == "center" else 0) + (8 if style == "pill" else 0)
            y = py + stroke + i * lead - asc
            if stroke:
                d.text((x, y), l, font=f, fill=(255, 255, 255, 255), stroke_width=stroke, stroke_fill=(6, 7, 9, 255))
            else:
                d.text((x, y), l, font=f, fill=(255, 255, 255, 255))
            dm.text((x, y), l, font=f, fill=255)
        self.rgba = np.asarray(img, np.float32) / 255.0
        self.fill = np.asarray(fillm, np.float32) / 255.0
        self.w, self.h = img.size
        self.words = sum(len(l.split()) for l in self.lines)


# ----------------------------------------------------------------- render ----
class Reel:
    def __init__(self, topic: dict, asset_dir: str, params: dict | None = None):
        self.t = topic
        self.p = {"push": 0.075, "breath": 0.006, "open_at": 0.12, "open_dur": 0.34,
                  "inner_drift": 0.035, "win_r": 300, "pill_alpha": 222, "bloom": True,
                  "parallax": 0.012, "sweep": 0.10}
        self.p.update(topic.get("render", {}))
        self.p.update(params or {})
        self.fps = 30
        self.dur = float(topic["duration"])
        reg = os.path.join(asset_dir, "registered")
        self.reg = json.load(open(os.path.join(reg, "registration.json")))
        rd = lambda n: cv2.cvtColor(cv2.imread(os.path.join(reg, n + ".png")), cv2.COLOR_BGR2RGB)
        self.L = {n: rd(n) for n in ("skin", "outer", "bone")}
        self.macro = rd("macro")
        mx, my, ms = self.reg["macro"]["box_canvas"]
        k = ms / self.macro.shape[1]
        self.A_macro = np.array([[k, 0, mx], [0, k, my], [0, 0, 1]], np.float64)   # macro px -> canvas
        self.cw, self.ch = self.reg["canvas"]
        self.content = self._content_rect()
        self.shots = topic["shots"]
        self.joint = np.array(topic["joint"], np.float64)
        self.cover = max(W / (self.content[2] - self.content[0]), H / (self.content[3] - self.content[1]))
        self.grain = [np.random.default_rng(s).normal(0, 1, (H // 2, W // 2)).astype(np.float32) for s in range(4)]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        self.vig = (1 - 0.38 * np.clip(d, 0, 1) ** 2.0)[..., None].astype(np.float32)
        # text sprites
        hk = topic["hook"]
        self.hook = TextSprite(hk, self.p.get("hook_size", 146), style="stroke", align="left", max_w=950, min_size=128)
        self.labels = {}
        for i, s in enumerate(self.shots):
            if s.get("label"):
                self.labels[i] = TextSprite([s["label"]], self.p.get("label_size", 92), style="pill",
                                            max_w=930, min_size=82, pill_alpha=self.p["pill_alpha"])
        c = topic.get("counter")
        if c:
            self.num_sprites = {}
            self.cap_sprite = TextSprite([c["caption"]], self.p.get("label_size", 92), style="pill", max_w=930, min_size=82,
                                         pill_alpha=self.p["pill_alpha"])
        self.last_info = None
        self.layout_log = self._autolayout() if self.p.get("autolayout", True) else []

    def _settle(self, sh, zoom):
        """Where the window lands on screen once a shot has settled (ignores the tiny push/breath)."""
        z = self.cover * zoom
        w = np.array(sh.get("win", self.t["joint"]), np.float64)
        sy = sh.get("screen_y", self.p.get("screen_y", 840)) + self.p.get("screen_y_shift", 0)
        foc = w + np.array([0.0, (H / 2 - sy) / z])
        hw, hh = W / (2 * z) * 1.02, H / (2 * z) * 1.02
        x0, y0, x1, y1 = self.content
        if x1 - x0 > 2 * hw:
            foc[0] = clamp(foc[0], x0 + hw, x1 - hw)
        if y1 - y0 > 2 * hh:
            foc[1] = clamp(foc[1], y0 + hh, y1 - hh)
        return (w - foc) * z + np.array([W / 2, H / 2])

    def _autolayout(self):
        """Keep every window clear of the text bands: raise the shot zoom (the camera
        clamp is what usually pushes a joint near the frame edge up into the hook) in
        4 % steps up to 2.6x.  Logged into the QA report; never moves the anatomy."""
        log = []
        hook_bottom = 166 + self.hook.h + 24
        lab_bottom = 170 + (max((l.h for l in self.labels.values()), default=0)) + 24
        cnt_top = 1170
        for i, sh in enumerate(self.shots):
            if sh.get("kind") == "bloom":
                continue
            z0 = z = sh.get("zoom", 1.0)
            r = sh.get("win_r", self.p["win_r"]) * self.p.get("win_scale", 1.0) * 1.05
            top_lim = hook_bottom if i == 0 else lab_bottom
            bot_lim = cnt_top if sh.get("kind") == "counter" else C_SAFE_WIN_BOTTOM
            for _ in range(40):
                cx, cy = self._settle(sh, z)
                if cy - r >= top_lim and cy + r <= bot_lim:
                    break
                z *= 1.04
                if z > 2.6:
                    break
            if z != z0:
                sh["zoom"] = round(z, 3)
                log.append(f"shot {i} zoom {z0} -> {sh['zoom']}")
        return log

    def _content_rect(self):
        """Canvas rows that hold real pixels (drops generator/padding letterbox rows:
        near-uniform and near-black).  Columns: full width (dark studio background
        is real background).  Topic may override with render.content_rect."""
        if self.p.get("content_rect"):
            return list(self.p["content_rect"])
        y0, y1 = 0, self.ch
        for im in self.L.values():
            g = im.astype(np.int16)
            bar = (g.std(axis=(1, 2)) < 1.2) & (g.max(axis=(1, 2)) < 22)
            r = np.where(~bar)[0]
            y0, y1 = max(y0, r[0] + 3), min(y1, r[-1] - 3)
        return [0, int(y0), int(self.cw), int(y1)]

    # ---------------------------------------------------------------- shots --
    def shot_at(self, t):
        i = 0
        for k, s in enumerate(self.shots):
            if t >= s["t"] - 1e-6:
                i = k
        return i

    def camera(self, t):
        """Returns (zoom, focus_canvas_xy, rot_deg). Continuous: global push-in +
        breathing + eased moves between per-shot framings + punch on cuts."""
        i = self.shot_at(t)
        s = self.shots[i]
        prev = self.shots[i - 1] if i > 0 else s
        mv = s.get("move", 0.55)
        q = ease_io((t - s["t"]) / mv) if i > 0 else 1.0
        z_s = lerp(prev.get("zoom", 1.0), s.get("zoom", 1.0), q)
        zz = self.cover * z_s
        def fo(sh):
            w = np.array(sh.get("win", self.t["joint"]), np.float64)
            return w + np.array([0.0, (H / 2 - (sh.get("screen_y", self.p.get("screen_y", 840)) + self.p.get("screen_y_shift", 0))) / (self.cover * sh.get("zoom", 1.0))])
        f0, f1 = fo(prev), fo(s)
        foc = f0 + (f1 - f0) * q
        push = 1 + self.p["push"] * (t / self.dur) + 0.03 * ease_out(t / 1.2)
        breath = 1 + self.p["breath"] * math.sin(t * 2 * math.pi / 3.3)
        punch = 0.0
        if i > 0 and s.get("kind") != "bloom":
            dt = t - s["t"]
            if 0 <= dt < 0.45:
                punch = 0.035 * math.sin(math.pi * dt / 0.45) * (1 - dt / 0.45)
        z = self.cover * max(1.0, z_s * push * breath * (1 + punch)) * 1.025   # never wider than real pixels (+ room for the 0.35 deg roll)
        rot = 0.35 * math.sin(t * 2 * math.pi / 7.0)
        foc = foc + np.array([6 * math.sin(t * 0.9), 4 * math.sin(t * 0.7 + 1)])
        # clamp so the view stays inside real pixels
        hw, hh = W / (2 * z) * 1.008, H / (2 * z) * 1.008
        x0, y0, x1, y1 = self.content
        foc[0] = clamp(foc[0], x0 + hw, x1 - hw) if x1 - x0 > 2 * hw else (x0 + x1) / 2
        foc[1] = clamp(foc[1], y0 + hh, y1 - hh) if y1 - y0 > 2 * hh else (y0 + y1) / 2
        return z, foc, rot

    @staticmethod
    def view_matrix(z, foc, rot, extra_scale=1.0, about=None):
        a = math.radians(rot)
        c, s = math.cos(a) * z, math.sin(a) * z
        M = np.array([[c, -s, W / 2 - (c * foc[0] - s * foc[1])],
                      [s, c, H / 2 - (s * foc[0] + c * foc[1])],
                      [0, 0, 1]], np.float64)
        if extra_scale != 1.0 and about is not None:
            ax, ay = about
            S = np.array([[extra_scale, 0, ax * (1 - extra_scale)], [0, extra_scale, ay * (1 - extra_scale)], [0, 0, 1]])
            M = S @ M
        return M

    def to_out(self, M, pt):
        v = M @ np.array([pt[0], pt[1], 1.0])
        return float(v[0]), float(v[1])

    # ------------------------------------------------------------- windows --
    def win_radius(self, t, i):
        s = self.shots[i]
        r_target = s.get("win_r", self.p["win_r"])
        prev = self.shots[i - 1] if i > 0 else s
        r_prev = prev.get("win_r", self.p["win_r"])
        r = lerp(r_prev, r_target, ease_io((t - s["t"]) / s.get("move", 0.55))) if i > 0 else r_target
        r *= self.p.get("win_scale", 1.0)
        o = (t - self.p["open_at"]) / self.p["open_dur"]
        if o <= 0:
            return 0.0
        r *= back_out(o, 1.4) if o < 1 else 1.0
        r *= 1 + 0.012 * math.sin(t * 2 * math.pi / 1.7)
        if s.get("kind") == "bloom" and self.p["bloom"]:
            r = lerp(r, 1500, ease_io((t - s["t"]) / 0.7) ** 1.3)
        return r

    def layer_patch(self, name, M, bbox, inner):
        """Warp a layer (with its backdrop fill) into output bbox. inner = extra
        content transform about window centre (depth drift)."""
        x0, y0, x1, y1 = bbox
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ inner @ M
        size = (x1 - x0, y1 - y0)
        if name == "macro":
            Tm = T @ self.A_macro
            mac = cv2.warpAffine(self.macro, Tm[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            under = cv2.warpAffine(self.L["bone"], T[:2], size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # feather the macro square into the bone plate so its edge never shows
            mk = cv2.warpAffine(np.ones(self.macro.shape[:2], np.float32), Tm[:2], size, flags=cv2.INTER_LINEAR)
            mk = cv2.GaussianBlur(cv2.erode(mk, np.ones((25, 25), np.uint8)), (0, 0), 14)[..., None]
            out = mac.astype(np.float32) * mk + under.astype(np.float32) * (1 - mk)
            return out / 255.0
        im = cv2.warpAffine(self.L[name], T[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        out = im.astype(np.float32) / 255.0
        if name == "bone":   # fill empty dark space around the bones with dimmed muscle (no holes)
            back = cv2.warpAffine(self.L["outer"], T[:2], size, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
            lum = out.max(axis=2, keepdims=True)
            a = np.clip((lum - 0.05) / 0.12, 0, 1)
            out = out * a + back * 0.42 * (1 - a)
        return out

    # --------------------------------------------------------------- frame --
    def frame(self, t, want_info=False):
        i = self.shot_at(t)
        s = self.shots[i]
        z, foc, rot = self.camera(t)
        M = self.view_matrix(z, foc, rot)
        base = cv2.warpAffine(self.L["skin"], M[:2], (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        c = base.astype(np.float32) / 255.0
        overlay_alpha = np.zeros((H, W), np.float32)
        texts = []
        jx, jy = self.to_out(M, s.get("win", self.t["joint"]))
        if i > 0:   # window centre glides between shots
            pw = self.shots[i - 1].get("win", self.t["joint"])
            px, py = self.to_out(M, pw)
            q = ease_io((t - s["t"]) / s.get("move", 0.55))
            jx, jy = lerp(px, jx, q), lerp(py, jy, q)
        r = self.win_radius(t, i)
        win = (jx, jy, r)
        if r > 2:
            pad = 4
            x0, y0 = int(max(0, jx - r - pad)), int(max(0, jy - r - pad))
            x1, y1 = int(min(W, jx + r + pad)), int(min(H, jy + r + pad))
            if x1 > x0 and y1 > y0:
                bbox = (x0, y0, x1, y1)
                dt = t - s["t"]
                nxt = self.shots[i + 1]["t"] if i + 1 < len(self.shots) else self.dur
                depth = 1 + self.p["parallax"] + self.p["inner_drift"] * ease_io(inv(s["t"], nxt, t))
                inner = np.array([[depth, 0, jx * (1 - depth)], [0, depth, jy * (1 - depth)], [0, 0, 1]])
                cur = self.layer_patch(s["layer"], M, bbox, inner)
                # iris wipe from the label target into the new layer
                prev_layer = self.shots[i - 1]["layer"] if i > 0 else None
                ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                dist = np.sqrt((xs - jx) ** 2 + (ys - jy) ** 2)
                if prev_layer and prev_layer != s["layer"] and dt < 0.34:
                    pdepth = 1 + self.p["parallax"] + self.p["inner_drift"]
                    pinner = np.array([[pdepth, 0, jx * (1 - pdepth)], [0, pdepth, jy * (1 - pdepth)], [0, 0, 1]])
                    old = self.layer_patch(prev_layer, M, bbox, pinner)
                    tx_, ty_ = self.to_out(M, s.get("target", self.t["joint"]))
                    rr = ease_out(dt / 0.34) * (r * 2.2)
                    d2 = np.sqrt((xs - tx_) ** 2 + (ys - ty_) ** 2)
                    m2 = np.clip((rr - d2) / 6.0 + 0.5, 0, 1)[..., None]
                    ring = np.exp(-((d2 - rr) / 7.0) ** 2)[..., None] * (1 - dt / 0.34)
                    cur = cur * m2 + old * (1 - m2) + ring * np.array(LIME, np.float32) / 255 * 0.85
                if 0 <= dt < 0.1 and i > 0:
                    cur = cur + (1 - dt / 0.1) * 0.22
                # light sweep (continuous motion inside the window)
                ph = (t * 0.62) % 1.0
                ang = math.radians(32)
                u = ((xs - jx) * math.cos(ang) + (ys - jy) * math.sin(ang)) / max(r, 1)
                band = np.exp(-((u - (ph * 3.2 - 1.6)) / 0.22) ** 2)[..., None]
                cur = cur + band * self.p["sweep"] * np.array([1.0, 0.97, 0.9], np.float32)
                # inner shadow near rim -> depth
                rim_d = np.clip((r - dist) / 46.0, 0, 1)[..., None]
                cur = cur * (0.55 + 0.45 * rim_d)
                m = np.clip(r - dist + 0.5, 0, 1)[..., None]
                c[y0:y1, x0:x1] = c[y0:y1, x0:x1] * (1 - m) + np.clip(cur, 0, 1) * m
                # rim: dark outer halo + lime ring + white hairline + scanner arc
                if r < 1400:
                    halo = np.clip(1 - (dist - r) / 26.0, 0, 1) * (dist > r) * 0.55
                    c[y0:y1, x0:x1] *= (1 - halo[..., None])
                    ring = np.clip(1.6 - np.abs(dist - r) / 1.6, 0, 1)
                    hair = np.clip(1 - np.abs(dist - (r - 6)) / 0.9, 0, 1) * 0.6
                    a = np.arctan2(ys - jy, xs - jx)
                    scan = ((np.mod(a - t * 2.2, 2 * np.pi) < 0.9).astype(np.float32)
                            * np.clip(1.6 - np.abs(dist - (r + 11)) / 1.8, 0, 1))
                    alpha = np.clip(ring + scan, 0, 1)
                    # UI graphics stay out of the bottom 15 % (IG caption/UI zone)
                    fade = np.clip((SAFE_BOTTOM - 8 - ys) / 70.0, 0, 1)
                    alpha *= fade
                    hair *= fade
                    col = np.array(LIME, np.float32) / 255
                    c[y0:y1, x0:x1] = c[y0:y1, x0:x1] * (1 - alpha[..., None]) + col * alpha[..., None]
                    c[y0:y1, x0:x1] = c[y0:y1, x0:x1] * (1 - hair[..., None]) + hair[..., None]
                    overlay_alpha[y0:y1, x0:x1] = np.maximum(overlay_alpha[y0:y1, x0:x1], alpha)
        # grade
        c *= self.vig
        g = cv2.resize(self.grain[int(t * self.fps) % 4], (W, H), interpolation=cv2.INTER_LINEAR)
        c += (g * 0.012)[..., None]
        if getattr(self, "hide_text", False):
            out = np.clip(c, 0, 1)
            if want_info:
                self.last_info = {"t": t, "shot": i, "win": win, "texts": [], "overlay_alpha": overlay_alpha,
                                  "layer": s["layer"], "view_overscan_px": 0.0}
            return out
        # hook
        if t < self.t.get("hook_out", 2.25):
            a = (1 - inv(self.t.get("hook_out", 2.25) - 0.18, self.t.get("hook_out", 2.25), t))
            yo = int(166 + 30 * (1 - ease_out(t / 0.25)))
            texts.append(self._blit(c, overlay_alpha, self.hook, 66, yo, a, "hook"))
        # label + leader
        lab = self.labels.get(i)
        if lab is not None:
            dt = t - s["t"] - s.get("label_delay", 0.12)
            a = ease_out(dt / 0.16)
            if a > 0:
                ly = int(170 - 26 * (1 - ease_out(dt / 0.22)))
                lx = 66
                tgt = self.to_out(M, s.get("target", self.t["joint"]))
                self._leader(c, overlay_alpha, (lx + 40, ly + lab.h), tgt, ease_out((dt - 0.05) / 0.26) * a)
                texts.append(self._blit(c, overlay_alpha, lab, lx, ly, a, "label"))
        # counter
        cn = self.t.get("counter")
        if cn and cn["t0"] <= t < cn["t1"]:
            dt = t - cn["t0"]
            a = ease_out(dt / 0.15) * (1 - inv(cn["t1"] - 0.15, cn["t1"], t))
            v = cn["to"] * ease_out(inv(0.1, 0.1 + cn.get("count", 1.3), dt))
            txt = cn["fmt"].format(v)
            sp = self.num_sprites.get(txt)
            if sp is None:
                sp = self.num_sprites[txt] = TextSprite([txt], self.p.get("num_size", 236), style="stroke", max_w=940)
            yb = 1180
            texts.append(self._blit(c, overlay_alpha, sp, 66, yb, a, "number"))
            texts.append(self._blit(c, overlay_alpha, self.cap_sprite, 66, yb + sp.h + 12, a, "caption"))
        out = np.clip(c, 0, 1)
        if want_info:
            Mi = np.linalg.inv(M)
            cs = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], np.float64) @ Mi.T
            x0, y0, x1, y1 = self.content
            out_px = max(0.0, x0 - cs[:, 0].min(), cs[:, 0].max() - x1, y0 - cs[:, 1].min(), cs[:, 1].max() - y1)
            self.last_info = {"t": t, "shot": i, "win": win, "texts": texts, "overlay_alpha": overlay_alpha,
                              "layer": s["layer"], "view_overscan_px": float(out_px * M[0, 0])}
        return out

    def _blit(self, c, oa, sp: TextSprite, x, y, a, kind):
        h, w = sp.h, sp.w
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0 or a <= 0.001:
            return {"kind": kind, "text": " / ".join(sp.lines), "bbox": None, "cap": sp.cap, "alpha": a,
                    "words": sp.words, "fits": sp.fits}
        src = sp.rgba[y0 - y:y1 - y, x0 - x:x1 - x]
        al = src[..., 3:4] * a
        c[y0:y1, x0:x1] = c[y0:y1, x0:x1] * (1 - al) + src[..., :3] * al
        oa[y0:y1, x0:x1] = np.maximum(oa[y0:y1, x0:x1], al[..., 0])
        return {"kind": kind, "text": " / ".join(sp.lines), "bbox": (x0, y0, x1, y1), "cap": sp.cap, "alpha": a,
                "fill": sp.fill[y0 - y:y1 - y, x0 - x:x1 - x], "words": sp.words, "fits": sp.fits}

    def _leader(self, c, oa, p0, p1, prog):
        if prog <= 0.01:
            return
        x0, y0 = p0
        x1, y1 = p1
        # elbow: straight down from the pill, then diagonal into the target
        ex, ey = x0, min(y1 - 60, y0 + 0.45 * (y1 - y0))
        pts = [(x0, y0), (ex, ey), (x1, y1)]
        segs = [math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(2)]
        L = sum(segs) * clamp(prog)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        drawn = []
        rem = L
        for k in range(2):
            if rem <= 0:
                break
            f = min(1.0, rem / max(segs[k], 1e-6))
            a, b = pts[k], pts[k + 1]
            e = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            drawn.append((a, e))
            rem -= segs[k]
        for a_, e in drawn:
            d.line([a_, e], fill=(5, 6, 8, 200), width=13)
        for a_, e in drawn:
            d.line([a_, e], fill=LIME + (255,), width=5)
        if prog >= 0.98:
            d.ellipse([x1 - 15, y1 - 15, x1 + 15, y1 + 15], fill=(5, 6, 8, 210))
            d.ellipse([x1 - 10, y1 - 10, x1 + 10, y1 + 10], fill=LIME + (255,))
            d.ellipse([x1 - 4, y1 - 4, x1 + 4, y1 + 4], fill=(255, 255, 255, 255))
        bb = img.getbbox()
        if not bb:
            return
        bx0, by0, bx1, by1 = bb
        arr = np.asarray(img.crop(bb), np.float32) / 255.0
        al = arr[..., 3:4]
        c[by0:by1, bx0:bx1] = c[by0:by1, bx0:bx1] * (1 - al) + arr[..., :3] * al
        oa[by0:by1, bx0:bx1] = np.maximum(oa[by0:by1, bx0:bx1], al[..., 0])


def render_video(reel: Reel, out_path: str, ffmpeg="/usr/bin/ffmpeg", workers=1):
    import subprocess
    n = int(round(reel.dur * reel.fps))
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(reel.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-g", "60", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for k in range(n):
            fr = reel.frame(k / reel.fps)
            proc.stdin.write((fr * 255 + 0.5).astype(np.uint8).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode:
        raise RuntimeError("ffmpeg encode failed")
    return out_path
