"""F1 CUTAWAY reel builder — "what's inside" anatomy window.

One JSON config + four plate images (living skin, muscle/fat layer, deep
bone/cartilage, macro joint gap) produce a 1080x1920 / 30fps vertical reel:

  hook (0-frame editorial type)  ->  circular surgical window opens  ->
  deeper plate inside the same window  ->  macro synovial-fluid cut  ->
  load count-up on a rim gauge  ->  window blooms open onto the bone plate.

The window content is registered to the base camera when `content.match_base`
is set, so the hole reads as a real cross-section instead of a floating inset.

Type & graphics follow one editorial / premium-medical language, all tunable
from the JSON through `style`:

  * hook      — thin eyebrow run + lime rule, big display title filled with a
                vertical paper gradient, metadata line; no hard stroke — the
                type stays legible from a soft multi-layer shadow plus a
                feathered scrim, so it works over both skin and black.
  * label     — dark glass pill (blurred backdrop, 1px metallic/lime rim,
                accent tab) and a technical callout leader (elbow + end node)
                that ties the pill to the exact spot inside the window.
  * counter   — caption / huge figure / separated unit / fine tick scale.
  * rim       — double bevel, metallic angular shading, hairline, scanner sweep.
  * arrows    — thick shaft, big head, dark halo, bright accent, white core.

Reused from the package: `theme` (type, palette, easings, tool glyphs),
`media` (canvas, blending, vignette, grain), `audio` (TTS + SFX + bed + LUFS).
Nothing here writes to flipcut-archive/ggt_template.
"""
from __future__ import annotations

import copy
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import media as M
from . import theme as T

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SS = 2  # supersample factor for line / callout / type layers


def _asset(root: str, p: str) -> str:
    """Template assets may be workspace-relative (like ep01) or package-relative."""
    if os.path.isabs(p):
        return p
    for base in (root, PKG):
        cand = os.path.join(base, p)
        if os.path.exists(cand):
            return cand
    return os.path.join(root, p)


def _merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------- defaults ---

STYLE_DEFAULTS = {
    "hook": {
        "x": 88,
        "size": 132,
        "track": 0.4,
        "lead": 1.02,            # leading as a multiple of the type size
        "grad_top": "#FFFFFF",
        "grad_bot": "#C6CDC1",
        "eyebrow": None,         # "BIOMECHANICS · DEEP SQUAT"
        "eyebrow_size": 26,
        "eyebrow_track": 5.6,
        "eyebrow_color": "#DCE1D7",
        "meta": [],
        "meta_size": 21,
        "meta_track": 3.6,
        "meta_color": "#AEB5A9",
        "scrim": 0.50,           # feathered darkening behind the block
        "dim": 0.10,             # whole-frame dim while the hook is up
    },
    "label": {
        "x": 88,
        "y": 300,
        "size": 60,
        "track": 3.4,
        "pad_x": 46,
        "pad_y": 26,
        "radius": 15,
        "glass": 0.34,           # 0 = clear glass, 1 = black
        "blur": 13.0,
        "rim_alpha": 0.50,
        "leader": True,
        "leader_w": 3.4,
        "leader_color": "#B4FF39",
    },
    "counter": {
        "x": 88,
        "y": 1488,               # figure baseline
        "size": 178,
        "unit_size": 70,
        "cap_size": 28,
        "cap_track": 7.5,
        "meter_w": 452,
        "meter_ticks": 41,
    },
    "rim": {
        "hairline": "#EAF2D8",
        "accent": "#B4FF39",
        "bevel": 10.0,
        "scanner": True,
    },
}


# ------------------------------------------------------------------ plate ---

class Plate:
    """A source image plus the src->screen mapping used by every shot.

    out = (src - focus) * scale + out_pt
    """

    def __init__(self, path: str, pad: int = 900):
        im = Image.open(path).convert("RGB")
        self.raw = im
        self.size = im.size
        self.pad = pad
        self.img = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad), (4, 4, 5))
        self.img.paste(im, (pad, pad))

    def render(self, W: int, H: int, scale: float, focus, out_pt) -> Image.Image:
        p = self.pad
        x0 = p + focus[0] - out_pt[0] / scale
        y0 = p + focus[1] - out_pt[1] / scale
        cw, ch = W / scale, H / scale
        iw, ih = self.img.size
        x0 = min(max(x0, 0.0), iw - cw) if cw < iw else 0.0
        y0 = min(max(y0, 0.0), ih - ch) if ch < ih else 0.0
        return self.img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.BICUBIC)


# ---------------------------------------------------------------- helpers ---

def circle_mask(bbox, cx, cy, r, soft=1.4):
    x0, y0, x1, y1 = bbox
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) - r
    return np.clip(0.5 - d / soft, 0.0, 1.0).astype(np.float32)


def ring_mask(bbox, cx, cy, r0, r1, soft=1.2):
    x0, y0, x1, y1 = bbox
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return (np.clip(0.5 - (d - r1) / soft, 0, 1) * np.clip(0.5 + (d - r0) / soft, 0, 1)).astype(np.float32)


def radial(bbox, cx, cy, r, power=2.0):
    x0, y0, x1, y1 = bbox
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.clip(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max(1.0, r), 0, 1)
    return (1.0 - d) ** power


def grade(rgb: np.ndarray, tint: str, mix: float) -> np.ndarray:
    if mix <= 0:
        return rgb
    t = np.asarray(T.rgb_f(tint), np.float32) * 1.5
    return rgb * (1 - mix) + (rgb * t) * mix


def bbox_of(cx, cy, r, W, H, pad=0):
    x0 = int(max(0, math.floor(cx - r - pad)))
    y0 = int(max(0, math.floor(cy - r - pad)))
    x1 = int(min(W, math.ceil(cx + r + pad)))
    y1 = int(min(H, math.ceil(cy + r + pad)))
    return x0, y0, x1, y1


def blur2d(a: np.ndarray, radius: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    if a.ndim == 3:
        return gaussian_filter(a.astype(np.float32), (radius, radius, 0))
    return gaussian_filter(a.astype(np.float32), radius)


def rr_mask(size, box, radius: float) -> np.ndarray:
    """Antialiased rounded-rect alpha mask (H, W) float32, drawn at SS and reduced."""
    w, h = size
    m = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([box[0] * SS, box[1] * SS, box[2] * SS, box[3] * SS],
                        radius=radius * SS, fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    return np.asarray(m, np.float32) / 255.0


def _layer(W, H):
    return Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))


def _down(layer: Image.Image, W: int, H: int) -> Image.Image:
    return layer.resize((W, H), Image.LANCZOS)


def _sprite_from_mask(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    out = np.dstack([np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)])
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


def _shadow_of(sprite: Image.Image, radius: float, gain: float) -> Image.Image:
    a = sprite.getchannel("A").filter(ImageFilter.GaussianBlur(radius))
    a = a.point(lambda v: int(min(255, v * gain)))
    sh = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    sh.putalpha(a)
    return sh


# ------------------------------------------------------------------ tools ---

def draw_tool(d: ImageDraw.ImageDraw, name: str, cx: float, cy: float, s: float, col) -> None:
    """Line-art surgical tool glyph inside a badge of size ~s."""
    w = max(2, int(round(s * 0.075)))
    if name == "saw":
        d.line([(cx - s * 0.30, cy - s * 0.02), (cx + s * 0.30, cy - s * 0.02)], fill=col, width=w)
        d.line([(cx - s * 0.30, cy - s * 0.02), (cx - s * 0.40, cy - s * 0.20)], fill=col, width=w)
        d.line([(cx - s * 0.30, cy - s * 0.02), (cx - s * 0.40, cy + s * 0.14)], fill=col, width=w)
        for i in range(6):
            x = cx - s * 0.26 + i * (s * 0.52 / 5)
            d.line([(x, cy - s * 0.02), (x + s * 0.045, cy + s * 0.13)], fill=col, width=w)
    elif name == "scalpel":
        d.line([(cx - s * 0.34, cy + s * 0.22), (cx + s * 0.16, cy - s * 0.22)], fill=col, width=w)
        d.polygon([(cx + s * 0.08, cy - s * 0.08), (cx + s * 0.34, cy - s * 0.30),
                   (cx + s * 0.18, cy + s * 0.08)], outline=col)
    elif name == "syringe":
        d.line([(cx - s * 0.34, cy + s * 0.06), (cx + s * 0.10, cy - s * 0.24)], fill=col, width=w)
        d.line([(cx + s * 0.10, cy - s * 0.24), (cx + s * 0.34, cy - s * 0.34)], fill=col, width=w)
        d.line([(cx - s * 0.30, cy + 0), (cx - s * 0.16, cy - s * 0.18)], fill=col, width=w)
        d.line([(cx - s * 0.24, cy + s * 0.14), (cx - s * 0.10, cy - s * 0.04)], fill=col, width=w)
    elif name == "drill":
        d.line([(cx - s * 0.34, cy + s * 0.10), (cx + s * 0.02, cy - s * 0.02)], fill=col, width=w)
        d.line([(cx + s * 0.02, cy - s * 0.02), (cx + s * 0.34, cy - s * 0.16)], fill=col, width=w)
        for i in range(3):
            x = cx + s * (0.02 + i * 0.11)
            d.line([(x, cy - s * 0.02 - i * s * 0.04), (x + s * 0.05, cy - s * 0.14 - i * s * 0.04)],
                   fill=col, width=max(1, w - 1))
    else:
        d.ellipse((cx - s * 0.2, cy - s * 0.2, cx + s * 0.2, cy + s * 0.2), outline=col, width=w)


# ------------------------------------------------------------------- reel ---

class Cutaway:
    def __init__(self, cfg: dict, root: str):
        self.cfg = cfg
        self.root = root
        f = cfg["format"]
        self.W, self.H = f["width"], f["height"]
        self.fps = f["fps"]
        self.dur = f["duration"]
        self.pal = cfg["palette"]
        self.cam = cfg["camera"]
        self.win = cfg["window"]
        self.shots = cfg["shots"]
        self.events = cfg.get("events", [])
        self.assets = {k: Plate(_asset(root, p)) for k, p in cfg["assets"].items()}
        self.bloom = cfg.get("bloom", {})
        self.style = _merge(STYLE_DEFAULTS, cfg.get("style"))
        self.grain = M.grain_bank(self.W, self.H, count=6, seed=21)
        self.vig = M.vignette(self.W, self.H, strength=0.40, power=1.9)[..., None]
        self.scrim_top = self._scrim(top=True)
        self.scrim_bottom = self._scrim(top=False)
        self._hook_cache = None

    # ------------------------------------------------------------- scrims --
    def _scrim(self, top: bool) -> np.ndarray:
        H, W = self.H, self.W
        arr = np.zeros((H, W, 4), np.float32)
        ys = np.arange(H, dtype=np.float32)[:, None]
        if top:
            a = np.clip((400 - ys) / 400.0, 0, 1) ** 1.4 * 0.66
        else:
            a = np.clip((ys - 1240) / 560.0, 0, 1) ** 1.5 * 0.70
        arr[:, :, 3] = a
        arr[:, :, 0:3] = np.array(T.rgb_f("#04050A"), np.float32)
        return arr

    # -------------------------------------------------------------- shots --
    def _shot_at(self, t: float) -> tuple[dict, dict | None]:
        cur, prev = self.shots[0], None
        for s in self.shots:
            if t >= s["t"] - 1e-6:
                prev, cur = cur, s
        return cur, prev

    def _next_shot_t(self, t: float) -> float:
        for s in self.shots[1:]:
            if t < s["t"] - 1e-6:
                return s["t"]
        return 1e9

    # -------------------------------------------------------------- camera --
    def _base_cam(self, t: float):
        c = self.cam
        p = T.ease_in_out(t / self.dur)
        return T.lerp(c["scale"][0], c["scale"][1], p), tuple(c["focus_src"]), tuple(c["focus_out"])

    # -------------------------------------------------------------- window --
    def _win(self, t: float, shot: dict, prev: dict | None):
        """Window centre + radius at t, easing from the previous shot's window."""
        base = shot.get("window", {})
        cx = base.get("cx", self.win["cx"])
        cy = base.get("cy", self.win["cy"])
        r = base.get("r", self.win["r"])
        if prev is not None:
            pw = prev.get("window", {})
            px = pw.get("cx", self.win["cx"])
            py = pw.get("cy", self.win["cy"])
            pr = pw.get("r", self.win["r"])
            mv = shot.get("move", 0.52)
            q = T.ease_in_out(T.inv(prev["t"] + prev.get("hold", 1e9), shot["t"] + mv, t)) \
                if prev["t"] + prev.get("hold", 1e9) < shot["t"] else T.ease_in_out((t - shot["t"]) / mv)
            cx, cy, r = T.lerp(px, cx, q), T.lerp(py, cy, q), T.lerp(pr, r, q)
        open_at, open_dur = self.win["open"]["at"], self.win["open"]["dur"]
        if t < open_at:
            return cx, cy, 0.0
        p = T.ease_out_cubic((t - open_at) / open_dur)
        r = r * (0.62 + 0.38 * p) + r * 0.045 * math.sin(math.pi * min(1.0, p * 1.15))
        if p >= 1.0:
            r = r + 1.1 * math.sin(t * 3.1)
        if self.bloom and t >= self.bloom.get("at", 1e9):
            q = T.ease_in_out((t - self.bloom["at"]) / self.bloom.get("dur", 0.6))
            r = r + (self.bloom["r"] - r) * q
        return cx, cy, r

    def _content(self, t: float, shot: dict, cam):
        c = shot.get("content")
        if not c:
            return None
        sc, fs, fo = cam
        if c.get("match_base"):
            scale, focus, out = sc, fs, fo
        else:
            scale, focus, out = c.get("scale", 1.0), tuple(c["focus"]), None
        drift = c.get("scale_drift", 0.0)
        if drift:
            nxt = self._next_shot_t(shot["t"] + 1e-3)
            end = min(nxt, shot["t"] + c.get("drift_dur", 2.4))
            scale *= 1.0 + drift * T.ease_in_out((t - shot["t"]) / max(0.01, end - shot["t"]))
        return (self.assets[c["asset"]], scale, focus, out, c.get("tint", "#FFFFFF"),
                c.get("mix", 0.0), c.get("backdrop"), cam, c.get("backdrop_scale", 1.0),
                c.get("exposure", 1.0))

    # --------------------------------------------------------------- frame --
    def frame(self, t: float) -> np.ndarray:
        W, H = self.W, self.H
        shot, prev = self._shot_at(t)
        cam = self._base_cam(t)
        sc, fs, fo = cam
        c = M.from_pil(self.assets[self.cfg["base_asset"]].render(W, H, sc, fs, fo))

        cx, cy, r = self._win(t, shot, prev)
        content = self._content(t, shot, cam)
        if content is not None and r > 1.0:
            plate, csc, cfocus, cout, tint, mix, backdrop, bcam, bscale, exposure = content
            cout = cout if cout is not None else (cx, cy)
            cont = M.from_pil(plate.render(W, H, csc, cfocus, cout))
            if backdrop:
                bsc, bfs, bfo = bcam
                bk = M.from_pil(self.assets[backdrop].render(W, H, bsc * bscale, bfs, bfo))
                lum = cont.max(axis=2, keepdims=True)
                a = np.clip((lum - 0.035) / 0.075, 0, 1)
                cont = cont * a + bk * (1 - a)
            cont = grade(cont, tint, mix)
            if exposure != 1.0:
                cont = cont * exposure
            con = shot.get("content", {}).get("contrast", 1.0)
            if con != 1.0:
                cont = np.clip((cont - 0.42) * con + 0.42, 0, 1)
            lift = shot.get("content", {}).get("lift", 0.0)
            if lift:
                cont = cont * (1.0 - lift) + lift * 0.5
            bbox = bbox_of(cx, cy, min(r, 1500) + 40, W, H)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                x0, y0, x1, y1 = bbox
                m = circle_mask(bbox, cx, cy, r)
                rad = radial(bbox, cx, cy, r * 1.1, power=2.4)[..., None]
                cg = float(shot.get("content", {}).get("center_gain", 0.30))
                cont[y0:y1, x0:x1] *= (0.84 + cg * rad)
                sh = circle_mask(bbox, cx, cy + 10, r + 7, soft=26)
                sh = np.clip(blur2d(sh * m, 16) * 1.6, 0, 1) * (1 - m)
                base_r = c[y0:y1, x0:x1]
                base_r *= (1.0 - 0.42 * sh[..., None])
                c[y0:y1, x0:x1] = base_r * (1 - m[..., None]) + cont[y0:y1, x0:x1] * m[..., None]
                if r < 1150:
                    self._rim(c, cx, cy, r, bbox, t, shot)

        win_mask = None
        if r > 1.0:
            wb = bbox_of(cx, cy, r, W, H)
            if wb[2] > wb[0] and wb[3] > wb[1]:
                win_mask = np.zeros((H, W), np.float32)
                win_mask[wb[1]:wb[3], wb[0]:wb[2]] = circle_mask(wb, cx, cy, r)

        self._overlay(c, t, shot, (cx, cy, r), win_mask)

        # grade the picture first, then lay type on top so it stays pure paper-white
        c += self.grain[int(t * self.fps) % len(self.grain)] * 0.013
        c = self._blend_scrim(c)
        c *= self.vig

        hk = self.cfg.get("hook")
        if hk and t < hk.get("out", 2.3) + 0.6:
            dim = 1.0 - self.style["hook"]["dim"] * (1 - T.inv(hk.get("out", 2.3) - 0.1,
                                                              hk.get("out", 2.3) + 0.5, t))
            c *= dim

        self._hook(c, t)
        self._label(c, t, shot, (cx, cy, r))
        self._counter(c, t, shot, (cx, cy, r))
        self._watermark(c)
        return np.clip(c, 0, 1)

    # ---------------------------------------------------------------- rim ---
    def _rim(self, c, cx, cy, r, bbox, t=0.0, shot=None) -> None:
        """Double bevel + metallic angular shading + hairline + scanner sweep."""
        x0, y0, x1, y1 = bbox
        st = self.style["rim"]
        bev = float(st["bevel"])
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        th = np.arctan2(ys - cy, xs - cx)
        light = 0.5 + 0.5 * np.cos(th + 2.25)   # bright upper-left, dark lower-right

        reg = c[y0:y1, x0:x1]
        # 1) outer wall: dark metal band just outside the cut
        wall = np.clip(0.5 - (d - (r + bev)) / 1.6, 0, 1) * np.clip(0.5 + (d - (r + 1.0)) / 1.6, 0, 1)
        reg *= (1.0 - 0.52 * wall[..., None])
        # 2) bevel face: metal gradient turning with the light
        face = np.clip(0.5 - (d - (r + bev * 0.62)) / 1.4, 0, 1) * \
            np.clip(0.5 + (d - (r + 1.2)) / 1.4, 0, 1)
        metal = np.asarray(T.rgb_f("#A8B0B4"), np.float32)[None, None, :] * (0.14 + 0.62 * light)[..., None]
        reg += metal * (face[..., None] * 0.55)
        # 3) crisp hairlines on the cut edge (metal) and at the bevel foot (lime)
        hair = np.exp(-((d - (r + 0.4)) / 1.15) ** 2)
        reg += np.asarray(T.rgb_f(st["hairline"]), np.float32) * (hair[..., None] * 0.62)
        accent = np.exp(-((d - (r + bev + 1.0)) / 1.0) ** 2)
        reg += np.asarray(T.rgb_f(st["accent"]), np.float32) * (accent[..., None] * 0.10)
        # 4) inner lip: falloff inside the window so the cut has depth
        lip = np.clip(1.0 - (r - d) / 26.0, 0, 1) * (d < r)
        reg *= (1.0 - 0.34 * lip[..., None] ** 1.6)
        # 5) specular arcs
        band = np.exp(-((d - (r + 2.4)) / 3.4) ** 2)
        arc = band * np.clip((th + 2.75) / 0.55, 0, 1) * np.clip((0.05 - th) / 0.55, 0, 1)
        reg += np.asarray(T.rgb_f("#FFFFFF"), np.float32) * (arc[..., None] * 0.34)
        arc2 = band * np.clip((th - 0.55) / 0.55, 0, 1) * np.clip((2.05 - th) / 0.55, 0, 1)
        reg += np.asarray(T.rgb_f("#7FD8FF"), np.float32) * (arc2[..., None] * 0.10)
        # 6) four registration ticks around the ring
        if r > 120:
            for k in range(4):
                a = math.radians(45 + 90 * k)
                aa = a if a <= math.pi else a - 2 * math.pi
                ang_band = np.exp(-((th - aa) / 0.014) ** 2)
                tick = ang_band * np.clip(1.0 - np.abs(d - (r + bev + 12)) / 10.0, 0, 1)
                reg += np.asarray(T.rgb_f("#D9E0DA"), np.float32) * (tick[..., None] * 0.30)
        c[y0:y1, x0:x1] = reg

        # 7) scanner sweep: one clean pass around the ring at the start of the cut
        if st.get("scanner") and shot is not None:
            el = t - shot["t"]
            if 0.0 <= el <= 1.35:
                q = T.ease_in_out(el / 1.35)
                ang = math.radians(-150 + 360 * q)
                fade = 1.0 - T.inv(0.95, 1.35, el)
                dth = np.abs(((th - ang + math.pi) % (2 * math.pi)) - math.pi)
                sweep = np.exp(-((dth / 0.10) ** 2)) * np.exp(-((d - (r + 1.0)) / 3.0) ** 2)
                tail = np.exp(-(dth / 0.85) ** 2) * (th < ang + 0.1) * \
                    np.exp(-((d - (r + 2.0)) / 5.0) ** 2)
                add = (sweep * 0.85 + tail * 0.20) * fade
                reg2 = c[y0:y1, x0:x1]
                reg2 += np.asarray(T.rgb_f("#EAF6FF"), np.float32) * (add[..., None])
                hx = cx + math.cos(ang) * (r + 1.0) - x0
                hy = cy + math.sin(ang) * (r + 1.0) - y0
                if 0 <= hy < reg2.shape[0] and 0 <= hx < reg2.shape[1]:
                    yy, xx = np.mgrid[0:reg2.shape[0], 0:reg2.shape[1]]
                    head = np.exp(-(((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * 5.0 ** 2))) * fade
                    reg2 += np.asarray(T.rgb_f("#FFFFFF"), np.float32) * (head[..., None] * 0.55)
                c[y0:y1, x0:x1] = reg2

    # ------------------------------------------------------------ overlays ---
    def _overlay(self, c, t, shot, win, win_mask):
        cx, cy, r = win
        flash = T.inv(shot["t"] - 0.01, shot["t"] + 0.02, t) * (1 - T.inv(shot["t"] + 0.05, shot["t"] + 0.34, t))
        if flash > 0.001 and r > 4:
            bbox = bbox_of(cx, cy, r, self.W, self.H)
            if bbox[2] > bbox[0]:
                x0, y0, x1, y1 = bbox
                m = circle_mask(bbox, cx, cy, r)
                c[y0:y1, x0:x1] += np.asarray(T.rgb_f("#CFEEFF"), np.float32) * (m[..., None] * flash * 0.14)
        for ev in self.events:
            if not (ev["t0"] <= t <= ev["t1"]):
                continue
            p = T.inv(ev["t0"], ev["t1"], t)
            kind = ev["type"]
            if kind == "glow":
                self._region_glow(c, ev, p, win, win_mask)
            elif kind == "angle":
                self._angle_line(c, ev, p, win, win_mask)
            elif kind == "arrow":
                self._arrow(c, ev, p, win, win_mask)
            elif kind == "gauge":
                self._gauge(c, ev, p, win)
            elif kind == "sheen":
                self._sheen(c, ev, p, win, win_mask)

    @staticmethod
    def _ev_pos(ev, win):
        dx, dy = ev.get("rel", (0.0, 0.0))
        if ev.get("rel") is not None:
            return win[0] + dx, win[1] + dy
        return tuple(ev["at"])

    def _region_glow(self, c, ev, p, win, win_mask):
        gx, gy = self._ev_pos(ev, win)
        rad = ev["r"]
        amount = ev["amount"] * (0.62 + 0.38 * math.sin(math.pi * p))
        bbox = bbox_of(gx, gy, rad * 1.7, self.W, self.H)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return
        x0, y0, x1, y1 = bbox
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt(((xs - gx) / rad) ** 2 + ((ys - gy) / (rad * ev.get("squash", 1.0))) ** 2)
        a = np.clip(1 - d, 0, 1) ** 2.3
        if ev.get("clip", True) and win_mask is not None:
            a = a * win_mask[y0:y1, x0:x1]
        c[y0:y1, x0:x1] += np.asarray(T.rgb_f(ev.get("color", "#FFF3DA")), np.float32) * (a[..., None] * amount)

    # --------------------------------------------------------- bold strokes --
    def _stroke_layer(self, draw_fn, mask=None):
        """Draw into a supersampled RGBA layer, reduce, and composite over `c`."""
        layer = _layer(self.W, self.H)
        draw_fn(ImageDraw.Draw(layer))
        return _down(layer, self.W, self.H), mask

    def _composite(self, c, layer: Image.Image, mask=None):
        a = np.asarray(layer.getchannel("A"), np.float32) / 255.0
        if mask is not None:
            a = a * mask
        rgb = np.asarray(layer.convert("RGB"), np.float32) / 255.0
        c *= (1 - a[..., None])
        c += rgb * a[..., None]

    def _arrow(self, c, ev, p, win, win_mask):
        """Thick high-contrast arrow: dark halo, accent shaft, white core, big head."""
        x0, y0 = ev["from"]
        x1, y1 = ev["to"]
        grow = T.ease_out_cubic(T.inv(0.0, 0.45, p))
        fade = 1 - T.inv(0.74, 1.0, p)
        if grow <= 0.01 or fade <= 0.01:
            return
        col = T.hex_rgb(ev.get("color", self.pal["lime"]))
        shaft = float(ev.get("width", 8))
        hs = float(ev.get("head", 36))
        ang = math.atan2(y1 - y0, x1 - x0)
        ex, ey = x0 + (x1 - x0) * grow, y0 + (y1 - y0) * grow
        L = math.hypot(x1 - x0, y1 - y0)
        stop = max(0.0, L - hs * 0.9) * grow / max(1e-6, L)
        mx, my = x0 + (x1 - x0) * stop, y0 + (y1 - y0) * stop

        def draw(d: ImageDraw.ImageDraw):
            d.line([(x0 * SS, y0 * SS), (mx * SS, my * SS)],
                   fill=(4, 6, 8, int(205 * fade)), width=int(shaft * 2.7 * SS))
            d.line([(x0 * SS, y0 * SS), (mx * SS, my * SS)],
                   fill=col + (int(250 * fade),), width=int(shaft * SS))
            d.line([(x0 * SS, y0 * SS), (mx * SS, my * SS)],
                   fill=(255, 255, 255, int(150 * fade)), width=max(1, int(shaft * 0.34 * SS)))
            if grow > 0.86:
                a2 = T.inv(0.86, 1.0, grow)
                hx, hy = ex * SS, ey * SS
                tip = [(hx, hy),
                       (hx - math.cos(ang - 0.45) * hs * 1.05 * SS, hy - math.sin(ang - 0.45) * hs * 1.05 * SS),
                       (hx - math.cos(ang + 0.45) * hs * 1.05 * SS, hy - math.sin(ang + 0.45) * hs * 1.05 * SS)]
                d.polygon([(px + 3 * SS, py + 3 * SS) for px, py in tip],
                          fill=(4, 6, 8, int(210 * fade * a2)))
                d.polygon(tip, fill=col + (int(255 * fade * a2),))
                inner = [(hx, hy),
                         (hx - math.cos(ang - 0.45) * hs * 0.50 * SS, hy - math.sin(ang - 0.45) * hs * 0.50 * SS),
                         (hx - math.cos(ang + 0.45) * hs * 0.50 * SS, hy - math.sin(ang + 0.45) * hs * 0.50 * SS)]
                d.polygon(inner, fill=(255, 255, 255, int(110 * fade * a2)))
        layer, _ = self._stroke_layer(draw)
        self._composite(c, layer, win_mask if ev.get("clip", True) else None)

    def _angle_line(self, c, ev, p, win, win_mask):
        """Two converging load rays with a span arc — same bold treatment."""
        a0, a1 = ev["angle"]
        ax, ay = self._ev_pos(ev, win) if ev.get("rel") is not None else tuple(
            ev.get("apex", (win[0], win[1])))
        L = ev.get("len", win[2] * 0.7)
        grow = T.ease_out_cubic(T.inv(0.0, 0.55, p))
        fade = 1 - T.inv(0.80, 1.0, p)
        if grow <= 0.01 or fade <= 0.01:
            return
        col = T.hex_rgb(ev.get("color", "#EAF6FF"))
        w = float(ev.get("width", 7))
        hs = 34.0

        def draw(d: ImageDraw.ImageDraw):
            for ang in (a0, a1):
                ex = ax + math.cos(math.radians(ang)) * L * grow
                ey = ay + math.sin(math.radians(ang)) * L * grow
                d.line([(ax * SS, ay * SS), (ex * SS, ey * SS)],
                       fill=(4, 6, 8, int(200 * fade)), width=int(w * 2.6 * SS))
                d.line([(ax * SS, ay * SS), (ex * SS, ey * SS)],
                       fill=col + (int(245 * fade),), width=int(w * SS))
                d.line([(ax * SS, ay * SS), (ex * SS, ey * SS)],
                       fill=(255, 255, 255, int(130 * fade)), width=max(1, int(w * 0.34 * SS)))
                if grow > 0.82:
                    a2 = T.inv(0.82, 1.0, grow)
                    tip = [(ex * SS, ey * SS),
                           ((ex - math.cos(math.radians(ang) - 0.42) * hs) * SS,
                            (ey - math.sin(math.radians(ang) - 0.42) * hs) * SS),
                           ((ex - math.cos(math.radians(ang) + 0.42) * hs) * SS,
                            (ey - math.sin(math.radians(ang) + 0.42) * hs) * SS)]
                    d.polygon([(px + 3 * SS, py + 3 * SS) for px, py in tip],
                              fill=(4, 6, 8, int(200 * fade * a2)))
                    d.polygon(tip, fill=col + (int(250 * fade * a2),))
            rr = L * 0.40
            steps = 40
            span = a1 - a0
            pts = [((ax + math.cos(math.radians(a0 + span * i / (steps - 1) * grow)) * rr) * SS,
                    (ay + math.sin(math.radians(a0 + span * i / (steps - 1) * grow)) * rr) * SS)
                   for i in range(steps)]
            n = max(2, int(steps * grow))
            d.line(pts[:n], fill=(4, 6, 8, int(170 * fade)), width=int(w * 2.2 * SS))
            d.line(pts[:n], fill=col + (int(210 * fade),), width=max(1, int(w * 0.7 * SS)))
        layer, _ = self._stroke_layer(draw)
        self._composite(c, layer, win_mask if ev.get("clip", True) else None)

    def _sheen(self, c, ev, p, win, win_mask):
        """A wet specular sweep across the window: reads as liquid, not as a still."""
        cx, cy, r = win
        bbox = bbox_of(cx, cy, r, self.W, self.H)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return
        x0, y0, x1, y1 = bbox
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        a_ang = math.radians(ev.get("angle", 28.0))
        proj = ((xs - cx) * math.cos(a_ang) + (ys - cy) * math.sin(a_ang)) / (2 * r) + 0.5
        pos = T.ease_in_out(p) * 1.25 - 0.12
        band = np.exp(-((proj - pos) / ev.get("width", 0.10)) ** 2) * ev.get("amount", 0.5)
        if win_mask is not None:
            band = band * win_mask[y0:y1, x0:x1]
        rgb = np.asarray(T.rgb_f(ev.get("color", "#DFF6FF")), np.float32)
        reg = c[y0:y1, x0:x1]
        c[y0:y1, x0:x1] = reg + rgb * (band[..., None] * np.clip(1.0 - reg, 0, 1))

    def _gauge(self, c, ev, p, win):
        """Fine rim dial: 60 ticks, thin arc, glowing head."""
        cx, cy, r = win
        if r < 10 or r > 900:
            return
        col = T.hex_rgb(ev.get("color", self.pal["lime"]))
        rr = r + 30
        q = min(1.0, p / max(1e-6, ev.get("sweep", 0.86)))
        fade = 1 - T.inv(0.93, 1.0, p)

        def draw(d: ImageDraw.ImageDraw):
            box = ((cx - rr) * SS, (cy - rr) * SS, (cx + rr) * SS, (cy + rr) * SS)
            d.arc(box, 0, 360, fill=(255, 255, 255, int(30 * fade)), width=int(3 * SS))
            d.arc(box, -90, -90 + 360 * q, fill=col + (int(235 * fade),), width=int(5 * SS))
            d.arc(box, -90, -90 + 360 * q, fill=(255, 255, 255, int(85 * fade)), width=int(1.6 * SS))
            for i in range(60):
                a = math.radians(-90 + 6 * i)
                on = (i / 60.0) <= q
                long = (i % 5 == 0)
                r0 = rr + (13 if long else 8)
                r1 = rr + (26 if long else 17)
                d.line([((cx + math.cos(a) * r0) * SS, (cy + math.sin(a) * r0) * SS),
                        ((cx + math.cos(a) * r1) * SS, (cy + math.sin(a) * r1) * SS)],
                       fill=(col + (int(220 * fade),)) if on else ((255, 255, 255, int(40 * fade))),
                       width=int((2.4 if long else 1.6) * SS))
            if q > 0.005:
                head = math.radians(-90 + 360 * q)
                hx, hy = cx + math.cos(head) * rr, cy + math.sin(head) * rr
                d.ellipse(((hx - 8) * SS, (hy - 8) * SS, (hx + 8) * SS, (hy + 8) * SS),
                          fill=(240, 255, 210, int(245 * fade)))
                d.ellipse(((hx - 15) * SS, (hy - 15) * SS, (hx + 15) * SS, (hy + 15) * SS),
                          outline=col + (int(120 * fade),), width=int(2 * SS))
        layer, _ = self._stroke_layer(draw)
        self._composite(c, layer, None)

    # ---------------------------------------------------------------- type ---
    def _build_hook(self):
        """Bake the hook block once: ink sprite, shadow sprites, scrim mask."""
        W, H = self.W, self.H
        hk = self.cfg["hook"]
        st = self.style["hook"]
        pal = self.pal
        x = float(st["x"])
        maxw = W - 2 * x

        lines = hk["text"]
        size = int(hk.get("size", st["size"]))
        tr = float(hk.get("track", st["track"]))
        f = T.font("display", size)
        while size > 60 and max(T.text_len(l, f, tr) for l in lines) > maxw:
            size -= 2
            f = T.font("display", size)
        lead = int(round(size * float(hk.get("lead", st["lead"]))))

        y_t = float(hk.get("y", 470))                 # title ascent top
        e_size = int(st["eyebrow_size"])
        y_e = y_t - int(e_size * 2.15)                 # eyebrow ascent top
        meta = st.get("meta") or []

        # --- title: glyph mask filled with a per-line vertical paper gradient --
        mask = Image.new("L", (W, H), 0)
        dm = ImageDraw.Draw(mask)
        for i, line in enumerate(lines):
            T.draw_text(dm, (x, y_t + i * lead), line, f, 255, anchor="la", tracking=tr)
        m = np.asarray(mask, np.float32) / 255.0
        bb = mask.getbbox() or (0, 0, W, H)
        col_top = np.asarray(T.rgb_f(st["grad_top"]), np.float32)[None, None, :]
        col_bot = np.asarray(T.rgb_f(st["grad_bot"]), np.float32)[None, None, :]
        ys = np.arange(H, dtype=np.float32)[:, None, None]
        grad = np.zeros((H, 1, 3), np.float32)
        asc, desc = f.getmetrics()
        for i in range(len(lines)):
            top = y_t + i * lead
            bot = top + asc + desc * 0.35
            gp = np.clip((ys[:, 0, 0] - top) / max(1.0, bot - top), 0, 1)[:, None, None]
            line_grad = col_top * (1 - gp) + col_bot * gp
            sel = (ys[:, 0, 0] >= top) & (ys[:, 0, 0] < bot)
            grad = np.where(sel[:, None, None], line_grad, grad)
        grad = np.where(grad == 0.0, col_top, grad)
        rgb = np.broadcast_to(grad, (H, W, 3)).astype(np.float32)
        ink = _sprite_from_mask(rgb, m)
        sh_far = _shadow_of(ink, 24, 1.30)
        sh_near = _shadow_of(ink, 6, 1.10)

        y_after = float(bb[3])                          # ink bottom of the title
        y_m = y_after + 104                            # metadata ascent top

        # --- eyebrow, rule, meta (small type, drawn flat) -------------------
        small = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ds = ImageDraw.Draw(small)
        eyebrow = st.get("eyebrow")
        if eyebrow:
            fe = T.font("cond_semi", e_size)
            etr = float(st["eyebrow_track"])
            ecol = T.hex_rgb(st["eyebrow_color"])
            runs = [p.strip() for p in eyebrow.split("·")]
            cx_ = x
            for i, run in enumerate(runs):
                if i:
                    dotw = T.text_len("·", fe, etr)
                    T.draw_text(ds, (cx_, y_e), "·", fe, T.hex_rgb(pal["lime"]) + (255,),
                                anchor="la", tracking=etr)
                    cx_ += dotw + etr * 2.4
                T.draw_text(ds, (cx_, y_e), run.upper(), fe, ecol + (255,), anchor="la", tracking=etr)
                cx_ += T.text_len(run.upper(), fe, etr) + etr * 2.4
        if meta:
            fm = T.font("cond_semi", int(st["meta_size"]))
            mt = float(st["meta_track"])
            mcol = T.hex_rgb(st["meta_color"])
            cx_ = x
            for i, item in enumerate(meta):
                if i:
                    ds.line([(cx_ + 10, y_m + 7), (cx_ + 10, y_m + 19)],
                            fill=(255, 255, 255, 96), width=1)
                    cx_ += 24
                ds.text((cx_, y_m), item.upper(), font=fm, fill=mcol + (255,), anchor="la")
                cx_ += T.text_len(item.upper(), fm, mt) + mt

        # feathered scrim behind the block (no visible edge)
        txt_w = max(T.text_len(l, f, tr) for l in lines)
        bx0, by0 = x - 110, y_e - 70
        bx1, by1 = x + txt_w + 110, (y_m + 60) if meta else (y_after + 40)
        yy = np.arange(H, dtype=np.float32)[:, None]
        xx = np.arange(W, dtype=np.float32)[None, :]
        fx = np.clip(np.minimum(xx - bx0, bx1 - xx) / 300.0, 0, 1)
        fy = np.clip(np.minimum(yy - by0, by1 - yy) / 260.0, 0, 1)
        scrim = (fx * fy) ** 1.25
        sc = np.zeros((H, W, 3), np.float32)
        sc[:] = np.asarray(T.rgb_f("#05060A"), np.float32)

        rw = float(hk.get("rule", {}).get("w", 300))
        self._hook_cache = {
            "ink": ink, "sh_far": sh_far, "sh_near": sh_near,
            "small": small, "scrim": scrim * float(st["scrim"]), "sc": sc,
            "rule_y": y_after + 40, "x": x, "rw": rw, "size": size,
        }

    def _hook(self, c, t) -> None:
        hk = self.cfg.get("hook")
        if not hk:
            return
        if self._hook_cache is None:
            self._build_hook()
        k = self._hook_cache
        t_in, t_out = hk["t"], hk.get("out", hk["t"] + 2.2)
        if t > t_out + 0.5:
            return
        p_in = T.ease_out_cubic(T.inv(t_in, t_in + 0.24, t)) if t_in > 0 else 1.0
        p_out = T.ease_in_out(T.inv(t_out, t_out + 0.34, t))
        a = (1.0 if t_in <= 0 else p_in) * (1 - p_out)
        if a <= 0.01:
            return
        slide = -30 * p_out + 10 * (1 - p_in)

        sm = k["scrim"] * a
        c *= (1 - sm[..., None])
        c += k["sc"][...] * sm[..., None]

        M.blend_sprite(c, k["sh_far"], 0, int(16 + slide), 0.60 * a)
        M.blend_sprite(c, k["sh_near"], 0, int(3 + slide), 0.55 * a)
        M.blend_sprite(c, k["small"], 0, int(slide * 0.6), a)
        M.blend_sprite(c, k["ink"], 0, int(slide), a)

        # lime rule under the block; grows from the left on the way in
        rp = T.ease_out_cubic(T.inv(t_in, t_in + 0.42, t)) if t_in > 0 else 1.0
        rw = max(1.0, k["rw"] * rp)
        x0 = k["x"]
        y0 = k["rule_y"] + slide
        lime = np.asarray(T.rgb_f(self.pal["lime"]), np.float32)
        iy0, iy1 = int(max(0, y0)), int(min(self.H, y0 + 4))
        ix0, ix1 = int(max(0, x0)), int(min(self.W, x0 + rw))
        if iy1 > iy0 and ix1 > ix0:
            c[iy0:iy1, ix0:ix1] = c[iy0:iy1, ix0:ix1] * (1 - a) + lime * a
        # thin trailing hairline after the rule
        tx0 = int(max(0, x0 + rw + 14))
        tx1 = int(min(self.W, x0 + rw + 14 + k["rw"] * 0.20))
        if iy1 > iy0 and tx1 > tx0:
            c[iy0 + 1:iy1 - 1, tx0:tx1] = c[iy0 + 1:iy1 - 1, tx0:tx1] * (1 - a * 0.35) + \
                np.asarray(T.rgb_f(self.pal["paper"]), np.float32) * (a * 0.35)

    # --------------------------------------------------------------- label ---
    def _label(self, c, t, shot, win) -> None:
        label = shot.get("label")
        if not label:
            return
        cx, cy, r = win
        st = self.style["label"]
        t0 = shot["t"] + shot.get("label_delay", 0.16)
        p_in = T.ease_out_cubic(T.inv(t0, t0 + 0.30, t))
        nxt = self._next_shot_t(shot["t"] + 1e-3)
        p_out = T.ease_in_out(T.inv(nxt - 0.18, nxt + 0.02, t)) if nxt < 1e9 else 0.0
        a = p_in * (1 - p_out)
        if a <= 0.01:
            return

        size = int(shot.get("label_size", st["size"]))
        tr = float(shot.get("label_track", st["track"]))
        f = T.font("cond_bold", size)
        txt = label.upper()
        tw = T.text_len(txt, f, tr)
        pad_x, pad_y = float(st["pad_x"]), float(st["pad_y"])
        maxw = self.W - 2 * st["x"] - 30
        while tw + pad_x * 2 > maxw and size > 32:
            size -= 2
            f = T.font("cond_bold", size)
            tw = T.text_len(txt, f, tr)
        ph = size * 0.98 + pad_y * 2
        pw = tw + pad_x * 2 + 14
        x0 = float(st["x"])
        yc = float(shot.get("label_y", st["y"])) + 20 * (1 - p_in)
        y0 = yc - ph / 2
        x1, y1 = x0 + pw, y0 + ph

        # --- glass: blurred, darkened backdrop inside a rounded rect -------
        gx0, gy0 = int(max(0, x0 - 6)), int(max(0, y0 - 8))
        gx1, gy1 = int(min(self.W, x1 + 6)), int(min(self.H, y1 + 10))
        reg = c[gy0:gy1, gx0:gx1]
        bl = blur2d(reg, float(st["blur"]))
        g = float(st["glass"])
        # desaturate then darken the backdrop so the glass never picks up the
        # warm bone behind it — it stays a clean, cool dark pane
        lum = bl @ np.array([0.30, 0.59, 0.11], np.float32)
        sober = bl * 0.46 + lum[..., None] * 0.54
        glass = sober * (1.0 - g) * 0.80 + np.asarray(T.rgb_f("#0A0F12"), np.float32) * g
        hh = glass.shape[0]
        sheen = (1.0 - np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]) * 0.05
        glass = glass + sheen * np.asarray(T.rgb_f("#DCE8EA"), np.float32)
        m = rr_mask((gx1 - gx0, gy1 - gy0),
                    (x0 - gx0, y0 - gy0, x1 - gx0, y1 - gy0), float(st["radius"]))
        mix = (m * a)[..., None]
        c[gy0:gy1, gx0:gx1] = reg * (1 - mix) + glass * mix

        # --- rim + type at 2x ---------------------------------------------
        rad = float(st["radius"])
        rim_a = float(st["rim_alpha"])
        lime = T.hex_rgb(self.pal["lime"])
        paper = T.hex_rgb(self.pal["paper"])

        def draw(d: ImageDraw.ImageDraw):
            box = (x0 * SS, y0 * SS, x1 * SS, y1 * SS)
            d.rounded_rectangle(box, radius=rad * SS,
                                outline=(216, 226, 220, int(96 * a)), width=int(2 * SS))
            d.line([((x0 + rad) * SS, (y1 - 1.2) * SS), ((x1 - rad) * SS, (y1 - 1.2) * SS)],
                   fill=lime + (int(rim_a * 200 * a),), width=int(2 * SS))
            d.line([((x0 + rad) * SS, (y0 + 1.6) * SS), ((x1 - rad) * SS, (y0 + 1.6) * SS)],
                   fill=(255, 255, 255, int(80 * a)), width=int(1.6 * SS))
            tab_h = ph * 0.44
            d.rounded_rectangle(((x0 + 15) * SS, (yc - tab_h / 2) * SS,
                                 (x0 + 20.5) * SS, (yc + tab_h / 2) * SS),
                                radius=2 * SS, fill=lime + (int(235 * a),))
            T.draw_text(d, ((x0 + pad_x + 12) * SS, (yc - 1) * SS), txt,
                        T.font("cond_bold", size * SS), paper + (int(255 * a),),
                        anchor="lm", tracking=tr * SS)
            # tool chip travelling with the callout label
            tool = shot.get("tool")
            if tool:
                bp = T.ease_out_back(T.inv(t0 + 0.10, t0 + 0.40, t), 1.7)
                if bp > 0.02:
                    br = 27.0 * bp
                    bxp, byp = x1 + 16 + 27, yc
                    d.ellipse(((bxp - br) * SS, (byp - br) * SS, (bxp + br) * SS, (byp + br) * SS),
                              fill=(6, 9, 11, int(225 * a)))
                    d.ellipse(((bxp - br) * SS, (byp - br) * SS, (bxp + br) * SS, (byp + br) * SS),
                              outline=lime + (int(150 * a),), width=int(1.8 * SS))
                    draw_tool(d, tool, bxp * SS, byp * SS, br * 1.5 * SS, paper + (int(240 * a),))
        layer, _ = self._stroke_layer(draw)
        sh = _shadow_of(_down(layer, self.W, self.H), 9, 1.0)
        M.blend_sprite(c, sh, 0, 4, 0.45 * a)
        self._composite(c, layer, None)

        if st.get("leader", True) and 24 < r < 900:
            self._callout(c, t, t0, p_in, (x0, y0, x1, y1), win, a)

    def _callout(self, c, t, t0, p_in, pill, win, alpha) -> None:
        cx, cy, r = win
        x0, y0, x1, y1 = pill
        st = self.style["label"]
        ax = x1 - 24 if cx > (x0 + x1) / 2 else x0 + 24
        ay = y1
        vx, vy = ax - cx, ay - cy
        n = math.hypot(vx, vy) or 1.0
        ex, ey = cx + vx / n * (r + 2.5), cy + vy / n * (r + 2.5)
        pts = [(ax, ay)]
        if ey < ay - 40:
            pts.append((ax, ay - 26))
        elif ey > ay + 40:
            pts.append((ax, ay + 26))
        pts.append((ex, ey))
        segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        total = sum(math.hypot(b[0] - a_[0], b[1] - a_[1]) for a_, b in segs)
        if total <= 1:
            return
        grow = T.ease_out_cubic(T.inv(t0 + 0.02, t0 + 0.34, t))
        lw = float(st["leader_w"])
        col = T.hex_rgb(st["leader_color"])
        if grow <= 0.01:
            return

        def draw(d: ImageDraw.ImageDraw):
            budget = total * grow
            acc = 0.0
            tip = pts[0]
            for a_, b in segs:
                if acc >= budget:
                    break
                sl = math.hypot(b[0] - a_[0], b[1] - a_[1])
                take = min(1.0, (budget - acc) / max(1e-6, sl))
                nb = (a_[0] + (b[0] - a_[0]) * take, a_[1] + (b[1] - a_[1]) * take)
                d.line([(a_[0] * SS, a_[1] * SS), (nb[0] * SS, nb[1] * SS)],
                       fill=(6, 8, 10, int(200 * alpha)), width=int(lw * 2.8 * SS))
                d.line([(a_[0] * SS, a_[1] * SS), (nb[0] * SS, nb[1] * SS)],
                       fill=col + (int(250 * alpha),), width=int(lw * SS))
                acc += sl * take
                tip = nb
            # the growing end always carries a dot, so the mark never stops in mid-air
            if grow < 0.999:
                d.ellipse(((tip[0] - 5.5) * SS, (tip[1] - 5.5) * SS,
                           (tip[0] + 5.5) * SS, (tip[1] + 5.5) * SS),
                          fill=(6, 8, 10, int(180 * alpha)))
                d.ellipse(((tip[0] - 3.4) * SS, (tip[1] - 3.4) * SS,
                           (tip[0] + 3.4) * SS, (tip[1] + 3.4) * SS),
                          fill=(255, 255, 255, int(235 * alpha)))
            d.ellipse(((pts[0][0] - 4.5) * SS, (pts[0][1] - 4.5) * SS,
                       (pts[0][0] + 4.5) * SS, (pts[0][1] + 4.5) * SS),
                      fill=col + (int(240 * alpha),))
            if grow > 0.7:
                na = T.inv(0.7, 1.0, grow) * alpha
                d.ellipse(((ex - 16) * SS, (ey - 16) * SS, (ex + 16) * SS, (ey + 16) * SS),
                          fill=(6, 8, 10, int(110 * na)))
                d.ellipse(((ex - 11.5) * SS, (ey - 11.5) * SS, (ex + 11.5) * SS, (ey + 11.5) * SS),
                          outline=col + (int(250 * na),), width=int(3.2 * SS))
                d.ellipse(((ex - 4.2) * SS, (ey - 4.2) * SS, (ex + 4.2) * SS, (ey + 4.2) * SS),
                          fill=(255, 255, 255, int(245 * na)))
        layer, _ = self._stroke_layer(draw)
        self._composite(c, layer, None)

    # ------------------------------------------------------------- counter ---
    def _counter(self, c, t, shot, win) -> None:
        cn = self.cfg.get("counter")
        if not cn or t < cn["t"] or t > cn["out"] + 0.4:
            return
        st = self.style["counter"]
        t0 = cn["t"]
        p_in = T.ease_out_cubic(T.inv(t0, t0 + 0.26, t))
        p_out = T.ease_in_out(T.inv(cn["out"], cn["out"] + 0.30, t))
        a = p_in * (1 - p_out)
        if a <= 0.01:
            return
        q = T.ease_out_cubic(T.inv(cn["start"], cn["start"] + cn["dur"], t))
        val = cn["from"] + (cn["to"] - cn["from"]) * q
        frac = max(0.0, min(1.0, (val - cn["from"]) / max(1e-6, cn["to"] - cn["from"])))

        x = float(cn.get("x", st["x"]))
        yb = float(cn.get("y", st["y"]))
        size = float(cn.get("size", st["size"]))
        lime = T.hex_rgb(self.pal["lime"])
        paper = T.hex_rgb(self.pal["paper"])
        muted = T.hex_rgb(self.pal["muted"])
        num = f"{val:.1f}"
        fnum = T.font("display", int(size * SS))
        funit = T.font("cond_bold", int(cn.get("unit_size", st["unit_size"]) * SS))
        fcap = T.font("cond_semi", int(cn.get("cap_size", st["cap_size"]) * SS))
        mw = float(cn.get("meter_w", st["meter_w"]))
        ticks = int(cn.get("meter_ticks", st["meter_ticks"]))
        cap_track = float(cn.get("cap_track", st["cap_track"]))
        rise = 24 * (1 - p_in)
        wnum = T.text_len(num, fnum, 0.6 * SS)

        def draw(d: ImageDraw.ImageDraw):
            cy_ = (yb - size * 0.94 + rise) * SS
            d.rounded_rectangle((x * SS, cy_ - 3 * SS, (x + 3.4) * SS, cy_ + 16 * SS),
                                radius=1.5 * SS, fill=lime + (int(240 * a),))
            T.draw_text(d, ((x + 22) * SS, cy_ + 6 * SS), cn["caption"].upper(), fcap,
                        muted + (int(255 * a),), anchor="lm", tracking=cap_track * SS)
            base = (yb + rise) * SS
            T.draw_text(d, (x * SS, base), num, fnum, paper + (int(255 * a),),
                        anchor="ls", tracking=0.6 * SS)
            T.draw_text(d, (x * SS + wnum + 8 * SS, base), "×", funit,
                        lime + (int(245 * a),), anchor="ls", tracking=0)
            ty = (yb + rise + 38) * SS
            span = mw * SS
            for i in range(ticks):
                px = x * SS + span * i / (ticks - 1)
                on = (i / (ticks - 1)) <= frac
                long = (i % 5 == 0)
                hgt = (13 if long else 6) * SS
                d.line([(px, ty), (px, ty + hgt)],
                       fill=(lime + (int(235 * a),)) if on else (255, 255, 255, int(46 * a)),
                       width=int((2.0 if long else 1.4) * SS))
            if frac > 0.005:
                hx = x * SS + span * frac
                d.ellipse((hx - 4.5 * SS, ty - 4.5 * SS, hx + 4.5 * SS, ty + 4.5 * SS),
                          fill=(240, 255, 220, int(235 * a)))
            fend = T.font("cond_semi", 18 * SS)
            T.draw_text(d, (x * SS, ty + 18 * SS), "0", fend, muted + (int(150 * a),),
                        anchor="la", tracking=1.6 * SS)
            T.draw_text(d, ((x + mw) * SS, ty + 18 * SS), f"{cn['to']:.1f}×", fend,
                        muted + (int(150 * a),), anchor="ra", tracking=1.6 * SS)
        layer, _ = self._stroke_layer(draw)
        sh = _shadow_of(_down(layer, self.W, self.H), 10, 1.0)
        M.blend_sprite(c, sh, 0, 6, 0.5 * a)
        self._composite(c, layer, None)

    # --------------------------------------------------------------- chrome --
    def _blend_scrim(self, c) -> np.ndarray:
        for s in (self.scrim_top, self.scrim_bottom):
            a = s[..., 3:4]
            c = c * (1 - a) + s[..., 0:3] * a
        return c

    def _watermark(self, c) -> None:
        if getattr(self, "_wm_cache", None) is None:
            W = self.W
            f = T.font("cond_semi", 27 * SS)
            txt = "@" + self.cfg["brand"]["handle"]
            tw = T.text_len(txt, f, 2.4 * SS) / SS

            def draw(d: ImageDraw.ImageDraw):
                T.draw_text(d, ((W - 44) * SS, 52 * SS), txt, f,
                            T.hex_rgb(self.pal["paper"]) + (120,), anchor="rm", tracking=2.4 * SS)
                cx = W - 44 - tw - 15
                d.ellipse(((cx - 3.2) * SS, (52 - 3.2) * SS, (cx + 3.2) * SS, (52 + 3.2) * SS),
                          fill=T.hex_rgb(self.pal["lime"]) + (160,))
            layer, _ = self._stroke_layer(draw)
            self._wm_cache = layer
        self._composite(c, self._wm_cache, None)

    # --------------------------------------------------------------- render --
    def render_video(self, out_path: str, i0: int = 0, i1: int | None = None):
        """Encode frames [i0, i1) — segments let a long reel render in parts."""
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        n_total = int(round(self.dur * self.fps))
        i1 = n_total if i1 is None else min(int(i1), n_total)
        n = max(0, i1 - int(i0))
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}",
               "-r", str(self.fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
               "-pix_fmt", "yuv420p", "-g", str(self.fps * 2), "-movflags", "+faststart",
               out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for k in range(n):
                i = int(i0) + k
                t = i / self.fps
                fr = self.frame(t)
                proc.stdin.write(np.clip(fr * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes())
                if k % 60 == 0:
                    print(f"  frame {i}/{n_total}  t={t:5.2f}s", flush=True)
        finally:
            proc.stdin.close()
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg encode failed")
        return out_path
