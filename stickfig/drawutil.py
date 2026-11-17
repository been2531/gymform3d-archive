"""Shared drawing helpers: capsules, tapered muscles, glow, paper grain."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Point = Tuple[float, float]
Color = Tuple[int, int, int]
ColorA = Tuple[int, int, int, int]


def _r(pt: Point) -> Tuple[float, float]:
    return (float(pt[0]), float(pt[1]))


def capsule_points(p1: Point, p2: Point, radius: float, n: int = 16) -> List[Point]:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    px, py = -uy * radius, ux * radius
    pts: List[Point] = []
    # Semicircle around p1, from +perp through the back to -perp.
    for i in range(n + 1):
        a = math.pi / 2 + math.pi * i / n
        ca, sa = math.cos(a), math.sin(a)
        pts.append((x1 + ca * px - sa * ux * radius, y1 + ca * py - sa * uy * radius))
    # Semicircle around p2, from -perp through the front to +perp.
    for i in range(n + 1):
        a = math.pi * 3 / 2 + math.pi * i / n
        ca, sa = math.cos(a), math.sin(a)
        pts.append((x2 + ca * px - sa * ux * radius, y2 + ca * py - sa * uy * radius))
    return pts


def draw_capsule(
    draw: ImageDraw.ImageDraw,
    p1: Point,
    p2: Point,
    radius: float,
    fill: ColorA | Color,
    outline=None,
    width: int = 0,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    px, py = -dy / L * radius, dx / L * radius
    draw.polygon(
        [(x1 + px, y1 + py), (x2 + px, y2 + py), (x2 - px, y2 - py), (x1 - px, y1 - py)],
        fill=fill,
        outline=outline,
        width=width or 0,
    )
    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=fill)
    draw.ellipse((x2 - radius, y2 - radius, x2 + radius, y2 + radius), fill=fill)


def draw_circle(
    draw: ImageDraw.ImageDraw,
    c: Point,
    r: float,
    fill: ColorA | Color,
    outline=None,
    width: int = 0,
) -> None:
    x, y = c
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=width)


def tapered_muscle(
    hip: Point, knee: Point, width: float, along0: float, along1: float, out: float
) -> List[Point]:
    """Teardrop / vastus shape on the front of the thigh, stopping short of the knee."""
    from .skeleton import anterior, dist, norm, _sub, _add, _mul

    u = norm(_sub(knee, hip))
    n = anterior(hip, knee)
    length = dist(hip, knee)
    a = _add(hip, _mul(u, length * along0))
    b = _add(hip, _mul(u, length * along1))
    # Offset the whole shape onto the anterior side, and taper: fat in the
    # middle, narrow at the hip and especially at the knee.
    mid = 0.5
    pts_l: List[Point] = []
    pts_r: List[Point] = []
    steps = 14
    for i in range(steps + 1):
        t = i / steps
        p = _add(a, _mul(_sub(b, a), t))
        # bulge peaks around t=0.45, falls to ~0.25 at the knee end
        envelope = math.sin(math.pi * t) ** 0.7
        envelope = 0.30 + 0.70 * envelope
        w = width * envelope
        # shift centre onto the front of the bone
        c = _add(p, _mul(n, out))
        pts_l.append(_add(c, _mul(n, w)))
        pts_r.append(_add(c, _mul(n, -w * 0.75)))  # wrap onto the thigh, not past the knee
    return pts_l + pts_r[::-1]


def glute_shape(hip_mid: Point, hip_far: Point, hip_near: Point, scale: float) -> List[Point]:
    """Small posterior oval pair behind the hips."""
    from .skeleton import _add, _mul, _sub, norm, perp

    # Back is +x (character faces left).
    back = (1.0, 0.15)
    c = _add(hip_mid, _mul(back, 22.0 * scale))
    rx, ry = 36 * scale, 28 * scale
    pts = []
    for i in range(20):
        a = 2 * math.pi * i / 20
        pts.append((c[0] + math.cos(a) * rx + 8 * scale, c[1] + math.sin(a) * ry))
    return pts


def fast_glow(layer: Image.Image, radius: float) -> Image.Image:
    """Downscale-blur-upscale glow; much cheaper than a full-res Gaussian."""
    if radius <= 0:
        return layer
    w, h = layer.size
    factor = 4 if radius >= 8 else 2
    sw, sh = max(1, w // factor), max(1, h // factor)
    small = layer.resize((sw, sh), Image.Resampling.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(max(0.6, radius / factor)))
    return small.resize((w, h), Image.Resampling.BILINEAR)


def add_glow(base: Image.Image, glow: Image.Image, gain: float = 1.0) -> Image.Image:
    """Additive composite of an RGBA glow onto an RGB/RGBA base."""
    canvas = base.convert("RGB")
    arr = np.asarray(canvas, dtype=np.float32)
    g = np.asarray(glow.convert("RGBA"), dtype=np.float32)
    a = (g[:, :, 3:4] / 255.0) * gain
    arr = np.clip(arr + g[:, :, :3] * a, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def line_layer(
    size: Tuple[int, int],
    segments: Iterable[Tuple[Point, Point]],
    color: ColorA,
    width: float,
    joints: Iterable[Tuple[Point, float]] | None = None,
) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = max(1.0, width / 2.0)
    rgba = color
    for a, b in segments:
        draw_capsule(d, a, b, r, rgba)
    if joints:
        for c, jr in joints:
            draw_circle(d, c, jr, rgba)
    return layer


def paper_background(size: Tuple[int, int], color: Color, seed: int = 17) -> Image.Image:
    w, h = size
    rng = np.random.RandomState(seed)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :] = np.array(color, dtype=np.float32)
    # Fine fibre grain, deterministic.
    grain = rng.normal(0, 2.4, (h, w)).astype(np.float32)
    # Long faint fibres in a couple of directions.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fibres = (
        0.45 * np.sin((xx * 0.37 + yy * 0.09) * 0.2)
        + 0.30 * np.sin((xx * -0.11 + yy * 0.41) * 0.13)
        + 0.22 * np.sin((xx * 0.02 + yy * 0.8) * 0.07)
    ).astype(np.float32)
    # Sparse darker flecks.
    flecks = (rng.rand(h, w) > 0.996).astype(np.float32) * rng.uniform(-10, -4, (h, w))
    acc = grain + fibres + flecks
    out = np.clip(base + acc[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def textured_mask(
    mask: Image.Image, color: Color, seed: int, grain: float = 5.0
) -> Image.Image:
    """Fill a mask with construction-paper colour + a locked grain."""
    m = np.asarray(mask, dtype=np.uint8)
    if m.ndim == 3:
        alpha = m[:, :, 3] if m.shape[2] == 4 else m[:, :, 0]
    else:
        alpha = m
    h, w = alpha.shape
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, grain, (h, w)).astype(np.float32)
    # Stretch a little so it reads as fibre, not TV static.
    if h > 8 and w > 8:
        tiny = rng.normal(0, 1.0, (h // 6 + 1, w // 2 + 1)).astype(np.float32)
        stretched = np.array(
            Image.fromarray(((tiny + 3) * 30).clip(0, 255).astype(np.uint8), "L").resize(
                (w, h), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        noise = noise + (stretched - 90) * 0.08
    rgb = np.clip(np.array(color, dtype=np.float32) + noise[:, :, None], 0, 255)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb.astype(np.uint8)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def drop_shadow(
    mask: Image.Image,
    offset: Tuple[int, int] = (10, 14),
    blur: float = 7.0,
    opacity: int = 70,
) -> Image.Image:
    if mask.mode != "L":
        a = mask.split()[-1] if mask.mode == "RGBA" else mask.convert("L")
    else:
        a = mask
    shadow_a = Image.new("L", a.size, 0)
    shadow_a.paste(a, offset)
    # Darken and blur.
    arr = np.asarray(shadow_a, dtype=np.float32)
    arr = arr * (opacity / 255.0)
    shadow_a = Image.fromarray(arr.astype(np.uint8), "L")
    shadow_a = shadow_a.filter(ImageFilter.GaussianBlur(blur))
    layer = Image.new("RGBA", a.size, (0, 0, 0, 0))
    layer.putalpha(shadow_a)
    return layer
