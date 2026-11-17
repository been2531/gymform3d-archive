"""Image plumbing: numpy canvas, Ken Burns crop, sprite extraction, glows."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFilter

from . import theme as T


# --------------------------------------------------------------- canvas -----

def canvas(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), np.float32)


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")


def from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), np.float32) / 255.0


def blend_rgb(dst: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, opacity: float) -> None:
    """dst = dst*(1-a) + rgb*a, alpha is (H,W) in 0..1."""
    if opacity <= 0.002:
        return
    a = alpha * opacity
    a = a[..., None]
    np.multiply(dst, 1.0 - a, out=dst)
    dst += rgb * a


def blend_sprite(dst: np.ndarray, sprite: Image.Image, x: int, y: int, opacity: float = 1.0) -> None:
    """Paste an RGBA sprite (PIL) onto the float canvas at (x, y)."""
    if opacity <= 0.002:
        return
    H, W, _ = dst.shape
    sw, sh = sprite.size
    x0, y0 = int(round(x)), int(round(y))
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    sx1 = min(sw, W - x0)
    sy1 = min(sh, H - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    box = (sx0, sy0, sx1, sy1)
    reg = sprite.crop(box)
    a = np.asarray(reg.getchannel("A"), np.float32) / 255.0 * opacity
    rgb = np.asarray(reg.convert("RGB"), np.float32) / 255.0
    dy0, dx0 = y0 + sy0, x0 + sx0
    dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
    sub = dst[dy0:dy1, dx0:dx1]
    a3 = a[..., None]
    sub *= (1.0 - a3)
    sub += rgb * a3


def darken(dst: np.ndarray, mask: np.ndarray, strength: float) -> None:
    dst *= (1.0 - mask[..., None] * strength)


def add_light(dst: np.ndarray, rgb, mask: np.ndarray, strength: float) -> None:
    dst += np.asarray(rgb, np.float32) * (mask[..., None] * strength)


# ---------------------------------------------------------------- source ----

class Source:
    """A base visual plus cached derivative layers."""

    def __init__(self, path: str, w: int, h: int):
        self.path = path
        self.img = Image.open(path).convert("RGB")
        self.w, self.h = w, h
        self._cache = {}

    def ken_burns(self, scale: float = 1.0, cx: float = 0.5, cy: float = 0.5) -> Image.Image:
        """Crop a `1/scale` window centred on (cx, cy) (relative), resize to output."""
        iw, ih = self.img.size
        cw = iw / max(1e-3, scale)
        ch = cw * self.h / self.w
        if ch > ih / max(1e-3, scale):
            ch = ih / max(1e-3, scale)
            cw = ch * self.w / self.h
        x0 = min(max(0.0, cx * iw - cw / 2), iw - cw)
        y0 = min(max(0.0, cy * ih - ch / 2), ih - ch)
        if cw >= iw - 0.5:
            x0, cw = 0.0, iw
            ch = cw * self.h / self.w
            y0 = min(max(0.0, cy * ih - ch / 2), ih - ch)
        return self.img.resize((self.w, self.h), Image.BICUBIC,
                               box=(x0, y0, x0 + cw, y0 + ch))

    def sprite(self, box, gain: float = 1.0, keep_color: bool = True) -> Image.Image:
        """Extract a region as RGBA with alpha from luminance (art sits on black)."""
        key = ("sprite", tuple(box), gain, keep_color)
        if key not in self._cache:
            crop = self.img.crop(box).convert("RGB")
            arr = np.asarray(crop, np.float32) / 255.0
            lum = arr.max(axis=2)
            a = np.clip(lum * gain, 0, 1)
            rgb = arr
            if not keep_color:
                rgb = np.ones_like(arr)
            out = np.dstack([rgb, a])
            self._cache[key] = Image.fromarray((out * 255).astype(np.uint8), "RGBA")
        return self._cache[key]


def glow(size, shapes, color, radius: float, intensity: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Build an additive glow layer from a list of draw callables on an L mask."""
    mask = Image.new("L", size, 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(mask)
    for fn in shapes:
        fn(d)
    if radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius))
    a = np.asarray(mask, np.float32) / 255.0 * intensity
    rgb = np.zeros((size[1], size[0], 3), np.float32)
    rgb[:] = np.asarray(T.rgb_f(color), np.float32)
    return rgb, a


def glow_from_mask(mask: Image.Image, color: str, radius: float, intensity: float = 1.0):
    if radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius))
    a = np.asarray(mask, np.float32) / 255.0 * intensity
    rgb = np.zeros((mask.size[1], mask.size[0], 3), np.float32)
    rgb[:] = np.asarray(T.rgb_f(color), np.float32)
    return rgb, a


def vignette(w: int, h: int, strength: float = 0.55, power: float = 2.2) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs - w / 2) / (w / 2)
    ny = (ys - h / 2) / (h / 2)
    r = np.sqrt(nx ** 2 + (ny * 0.92) ** 2)
    v = 1.0 - strength * np.clip(r ** power, 0, 1)
    return v.astype(np.float32)


def grain_bank(w: int = 0, h: int = 0, count: int = 6, seed: int = 7):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 1, (h, w, 1)).astype(np.float32) for _ in range(count)]


def linear_fade_mask(size, direction: str = "left", feather: float = 0.25) -> np.ndarray:
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    if direction == "left":
        a = xs / (w * feather)
    elif direction == "right":
        a = (w - xs) / (w * feather)
    elif direction == "top":
        a = ys / (h * feather)
    else:
        a = (h - ys) / (h * feather)
    return np.clip(a, 0, 1) ** 1.0


def circular_mask(size, feather: float = 0.06) -> np.ndarray:
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / (min(w, h) / 2)
    a = np.clip((1 - r) / feather, 0, 1)
    return (a * a * (3 - 2 * a)).astype(np.float32)
