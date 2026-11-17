"""gymform_reels — shared typography, palette and drawing helpers.

Everything visual in this template funnels through this module so a new
episode only needs a new JSON file (data/*.json) and two base visuals.
"""
from __future__ import annotations

import functools
import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
FONT_DIR = os.path.join(PKG, "assets", "fonts")

FONT_FILES = {
    "display": os.path.join(FONT_DIR, "Anton-Regular.ttf"),
    "archivo": os.path.join(FONT_DIR, "ArchivoBlack-Regular.ttf"),
    "cond_bold": os.path.join(FONT_DIR, "BarlowCondensed-Bold.ttf"),
    "cond_semi": os.path.join(FONT_DIR, "BarlowCondensed-SemiBold.ttf"),
    "body": os.path.join(FONT_DIR, "Inter[opsz,wght].ttf"),
    "sans_bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
}

# Inter is a variable font: axes are (opsz, wght) in that file's fvar order.
_VAR_AXES = {"body": (14.0, 700.0)}


@functools.lru_cache(maxsize=None)
def font(name: str, size: int = 40, weight: float | None = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(FONT_FILES[name], size)
    if name in _VAR_AXES:
        try:
            opsz, wght = _VAR_AXES[name]
            f.set_variation_by_axes([opsz, weight if weight is not None else wght])
        except Exception:
            pass
    return f


# ---------------------------------------------------------------- maths -----

def clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def inv(a: float, b: float, v: float) -> float:
    """Normalise v into 0..1 across [a, b]."""
    if b == a:
        return 1.0
    return clamp01((v - a) / (b - a))


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_in_cubic(t: float) -> float:
    t = clamp01(t)
    return t ** 3


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_expo(t: float) -> float:
    t = clamp01(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def ease_out_back(t: float, s: float = 1.9) -> float:
    t = clamp01(t)
    return 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2


def pulse(t: float, period: float = 1.1) -> float:
    """0..1 soft pulse, phase-locked to absolute time t."""
    return 0.5 + 0.5 * math.sin(2 * math.pi * t / period - math.pi / 2)


# ---------------------------------------------------------------- colour ----

def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_f(h: str) -> tuple[float, float, float]:
    r, g, b = hex_rgb(h)
    return (r / 255.0, g / 255.0, b / 255.0)


# ------------------------------------------------------------------ text ----

def text_len(s: str, f: ImageFont.FreeTypeFont, tracking: float = 0.0) -> float:
    if tracking == 0:
        return f.getlength(s)
    return sum(f.getlength(ch) for ch in s) + tracking * max(0, len(s) - 1)


def draw_text(draw: ImageDraw.ImageDraw, xy, s: str, f, fill, anchor: str = "lm",
              tracking: float = 0.0, stroke: int = 0, stroke_fill=None) -> None:
    """Anchor follows Pillow's scheme plus explicit tracking and baseline control.

    anchor[0]: l / m / r (horizontal), anchor[1]: a (ascent top) / m (vert centre) / s (baseline).
    """
    x, y = xy
    w = text_len(s, f, tracking)
    asc, desc = f.getmetrics()
    if anchor[0] == "l":
        pass
    elif anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    if anchor[1] == "a":
        y -= 0
    elif anchor[1] == "m":
        y -= (asc + desc) / 2
    elif anchor[1] == "s":
        y -= asc
    if tracking == 0:
        draw.text((x, y), s, font=f, fill=fill, anchor="la",
                  stroke_width=stroke, stroke_fill=stroke_fill)
        return
    for ch in s:
        draw.text((x, y), ch, font=f, fill=fill, anchor="la",
                  stroke_width=stroke, stroke_fill=stroke_fill)
        x += f.getlength(ch) + tracking


def fit_size(s: str, font_name: str, max_w: float, size: int, min_size: int = 12,
             tracking: float = 0.0, weight: float | None = None) -> int:
    """Largest size <= `size` whose rendered string fits max_w."""
    for sz in range(size, min_size - 1, -2):
        if text_len(s, font(font_name, sz, weight), tracking) <= max_w:
            return sz
    return min_size


def wrap(s: str, f, max_w: float) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for wd in words:
        cand = (cur + " " + wd).strip()
        if f.getlength(cand) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


# ----------------------------------------------------------------- shapes ---

def rrect(draw: ImageDraw.ImageDraw, box, r: int, fill=None, outline=None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def vgrad(size, top, bottom) -> Image.Image:
    """Vertical gradient RGB image."""
    w, h = size
    col = Image.new("RGB", (1, h))
    px = col.load()
    for y in range(h):
        t = y / max(1, h - 1)
        px[0, y] = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
    return col.resize((w, h), Image.BILINEAR)


def radial_mask(size, center, radius, softness=0.55) -> Image.Image:
    """L-mode ellipse mask with a soft falloff, as a numpy float array later."""
    import numpy as np
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype("float32")
    cx, cy = center
    d = np.sqrt(((xs - cx) / radius) ** 2 + ((ys - cy) / (radius * (h / w))) ** 2) if False else \
        np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / radius
    a = np.clip((1.0 - d) / softness, 0.0, 1.0)
    a = a * a * (3 - 2 * a)
    return Image.fromarray((a * 255).astype("uint8"), "L")


def soft_mask_from_box(size, box, feather: int = 40) -> Image.Image:
    """Rectangular mask with feathered edges (for patching source art)."""
    import numpy as np
    w, h = size
    x0, y0, x1, y1 = box
    ys, xs = np.mgrid[0:h, 0:w].astype("float32")
    a = np.ones((h, w), "float32")
    a *= np.clip((xs - x0) / feather, 0, 1) * np.clip((x1 - xs) / feather, 0, 1)
    a *= np.clip((ys - y0) / feather, 0, 1) * np.clip((y1 - ys) / feather, 0, 1)
    return Image.fromarray((a * 255).astype("uint8"), "L")


# ------------------------------------------------------------------ icons ---

def icon_bookmark(d: ImageDraw.ImageDraw, x: float, y: float, s: float, fill, cut=None) -> None:
    """Bookmark glyph drawn from a polygon (no emoji font needed)."""
    pts = [(x, y), (x + s, y), (x + s, y + s), (x + s / 2, y + s * 0.74), (x, y + s)]
    d.polygon(pts, fill=fill)


def icon_check(d: ImageDraw.ImageDraw, x: float, y: float, s: float, fill, w: int) -> None:
    d.line([(x, y + s * 0.55), (x + s * 0.36, y + s * 0.92), (x + s, y + s * 0.08)],
           fill=fill, width=w, joint="curve")


def icon_bolt(d: ImageDraw.ImageDraw, x: float, y: float, s: float, fill) -> None:
    d.polygon([(x + s * 0.55, y), (x + s * 0.12, y + s * 0.56), (x + s * 0.44, y + s * 0.56),
               (x + s * 0.36, y + s), (x + s * 0.88, y + s * 0.40), (x + s * 0.54, y + s * 0.40)],
              fill=fill)
