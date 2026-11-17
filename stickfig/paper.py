"""Flat paper-cut character. Cute, layered, fixed shapes, no drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw

from . import drawutil as du
from .skeleton import Joints, _add, _mul, _sub, dist, lerp, norm

Point = Tuple[float, float]
Color = Tuple[int, int, int]

_BG_CACHE: dict = {}


def _background(size: Tuple[int, int], color: Color, seed: int = 17) -> Image.Image:
    key = (size, color, seed)
    if key not in _BG_CACHE:
        _BG_CACHE[key] = du.paper_background(size, color, seed=seed)
    return _BG_CACHE[key]


@dataclass
class PaperParams:
    bg: Color = (243, 237, 226)
    head: Color = (230, 198, 162)
    tank: Color = (176, 150, 196)
    arm: Color = (92, 94, 102)
    thigh: Color = (220, 192, 154)
    quad: Color = (214, 64, 48)
    glute: Color = (196, 78, 62)
    shin: Color = (92, 94, 102)
    foot: Color = (220, 192, 154)
    bar: Color = (118, 126, 138)
    plate: Color = (130, 144, 164)
    eye: Color = (45, 40, 38)
    limb_r: float = 34.0
    thigh_r: float = 42.0
    shin_r: float = 30.0
    head_r: float = 96.0
    hand_r: float = 24.0
    knee_r: float = 24.0
    plate_rx: float = 88.0
    plate_ry: float = 108.0
    bar_h: float = 20.0
    muscle_w: float = 36.0
    muscle_out: float = 26.0
    shadow_op: int = 64
    shadow_blur: float = 6.5
    shadow_off: Tuple[int, int] = (9, 13)


V1 = PaperParams()
V2 = PaperParams(
    limb_r=36.0,
    thigh_r=46.0,
    shin_r=31.0,
    head_r=100.0,
    hand_r=26.0,
    knee_r=26.0,
    plate_rx=92.0,
    plate_ry=112.0,
    bar_h=22.0,
    muscle_w=50.0,
    muscle_out=30.0,
    shadow_op=78,
    shadow_blur=8.0,
    shadow_off=(10, 16),
)


class _Stack:
    """Draw cutouts back-to-front, each with its own drop shadow and grain."""

    def __init__(self, size: Tuple[int, int], params: PaperParams):
        self.size = size
        self.p = params
        self.canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        self._n = 0

    def _mask_from(self, painter) -> Image.Image:
        mask = Image.new("L", self.size, 0)
        d = ImageDraw.Draw(mask)
        painter(d)
        return mask

    def add_mask(self, mask: Image.Image, color: Color, seed: int) -> None:
        self._n += 1
        alpha = mask if mask.mode == "L" else mask.split()[-1]
        bbox = alpha.getbbox()
        if not bbox:
            return
        pad = int(max(abs(self.p.shadow_off[0]), abs(self.p.shadow_off[1])) + self.p.shadow_blur * 3 + 4)
        x0 = max(0, bbox[0] - pad)
        y0 = max(0, bbox[1] - pad)
        x1 = min(self.size[0], bbox[2] + pad)
        y1 = min(self.size[1], bbox[3] + pad)
        crop = alpha.crop((x0, y0, x1, y1))
        piece = du.textured_mask(crop, color, seed=seed, grain=5.5)
        shadow = du.drop_shadow(
            crop,
            offset=self.p.shadow_off,
            blur=self.p.shadow_blur,
            opacity=self.p.shadow_op,
        )
        full_s = Image.new("RGBA", self.size, (0, 0, 0, 0))
        full_p = Image.new("RGBA", self.size, (0, 0, 0, 0))
        full_s.paste(shadow, (x0, y0), shadow)
        full_p.paste(piece, (x0, y0), piece)
        self.canvas = Image.alpha_composite(self.canvas, full_s)
        self.canvas = Image.alpha_composite(self.canvas, full_p)

    def capsule(self, a: Point, b: Point, r: float, color: Color, seed: int) -> None:
        def paint(d):
            du.draw_capsule(d, a, b, r, 255)

        self.add_mask(self._mask_from(paint), color, seed)

    def circle(self, c: Point, r: float, color: Color, seed: int) -> None:
        def paint(d):
            du.draw_circle(d, c, r, 255)

        self.add_mask(self._mask_from(paint), color, seed)

    def ellipse(self, c: Point, rx: float, ry: float, color: Color, seed: int) -> None:
        def paint(d):
            d.ellipse((c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry), fill=255)

        self.add_mask(self._mask_from(paint), color, seed)

    def poly(self, pts, color: Color, seed: int) -> None:
        def paint(d):
            d.polygon(pts, fill=255)

        self.add_mask(self._mask_from(paint), color, seed)

    def rounded_bar(self, a: Point, b: Point, height: float, color: Color, seed: int) -> None:
        self.capsule(a, b, height / 2.0, color, seed)


def _torso_poly(j: Joints, width_s: float, width_h: float) -> list:
    """Rounded tank top that follows the spine, wide at the shoulders."""
    from . import drawutil as _du
    from .skeleton import lerp as _lerp

    top = (j["shoulder_mid"][0], j["shoulder_mid"][1] - 12)
    bot = (j["hip_mid"][0], j["hip_mid"][1] + 16)
    # A long stadium, then we also add a shoulder cap in render().
    return _du.capsule_points(top, bot, 78.0 * width_s, n=18)


def _neck_cap(j: Joints, head_r: float) -> Tuple[Point, Point]:
    return j["neck"], j["head"]


def render(
    joints: Joints,
    size: Tuple[int, int] = (1080, 1920),
    pulse: float = 1.0,
    params: PaperParams | None = None,
) -> Image.Image:
    p = params or V1
    bg = _background(size, p.bg)
    st = _Stack(size, p)
    j = joints

    # --- back to front ---
    # Far plate (left), slightly behind everything.
    st.ellipse(j["plate_l"], p.plate_rx, p.plate_ry, p.plate, seed=21)

    # Far arm, then far hand (hand later with the bar too).
    st.capsule(j["shoulder_far"], j["elbow_far"], p.limb_r * 0.92, p.arm, seed=31)
    st.capsule(j["elbow_far"], j["hand_far"], p.limb_r * 0.82, p.arm, seed=32)

    # Far leg: shin + foot first (behind the thigh slightly is fine if we do thigh after).
    st.capsule(j["knee_far"], j["ankle_far"], p.shin_r, p.shin, seed=41)
    st.capsule(j["ankle_far"], j["toe_far"], p.shin_r * 0.85, p.foot, seed=42)
    st.circle(j["ankle_far"], p.shin_r * 0.9, p.foot, seed=43)

    # Far thigh, then the red quad on top of it, knee cap last so the quad
    # cannot read as a knee pad.
    st.capsule(j["hip_far"], j["knee_far"], p.thigh_r, p.thigh, seed=44)
    far_quad = du.tapered_muscle(
        j["hip_far"],
        j["knee_far"],
        width=p.muscle_w * (0.9 + 0.18 * pulse),
        along0=0.05,
        along1=0.64,
        out=p.muscle_out,
    )
    st.poly(far_quad, p.quad, seed=45)
    st.circle(j["knee_far"], p.knee_r, p.thigh, seed=46)

    # Glute, then torso tank, then near thigh / quad / knee.
    st.poly(du.glute_shape(j["hip_mid"], j["hip_far"], j["hip_near"], 1.0 + 0.12 * pulse), p.glute, seed=50)
    top = (j["shoulder_mid"][0], j["shoulder_mid"][1] - 8)
    bot = (j["hip_mid"][0], j["hip_mid"][1] + 24)
    st.capsule(top, bot, 90.0, p.tank, seed=51)
    st.capsule(j["hip_far"], j["hip_near"], 40.0, p.tank, seed=511)
    # Short neck that stops at the chin, so it never cuts through the face.
    chin = (j["head"][0], j["head"][1] + p.head_r * 0.78)
    st.capsule(j["shoulder_mid"], chin, 16.0, p.head, seed=52)
    # Shoulder cap so the tank reads as a shirt.
    st.capsule(j["shoulder_far"], j["shoulder_near"], 36.0, p.tank, seed=53)

    st.capsule(j["hip_near"], j["knee_near"], p.thigh_r, p.thigh, seed=54)
    near_quad = du.tapered_muscle(
        j["hip_near"],
        j["knee_near"],
        width=p.muscle_w * (0.95 + 0.2 * pulse),
        along0=0.05,
        along1=0.64,
        out=p.muscle_out,
    )
    st.poly(near_quad, p.quad, seed=55)
    st.circle(j["knee_near"], p.knee_r, p.thigh, seed=56)

    st.capsule(j["knee_near"], j["ankle_near"], p.shin_r, p.shin, seed=57)
    st.capsule(j["ankle_near"], j["toe_near"], p.shin_r * 0.85, p.foot, seed=58)
    st.circle(j["ankle_near"], p.shin_r * 0.9, p.foot, seed=59)

    # Head + eyes.
    st.circle(j["head"], p.head_r, p.head, seed=61)
    # Tiny punched-dot eyes, close together, slightly above centre, looking forward.
    hx, hy = j["head"]
    eye_y = hy - p.head_r * 0.08
    eye_d = p.head_r * 0.16
    # Face looks left (forward).
    face_x = hx - p.head_r * 0.12
    st.circle((face_x - eye_d, eye_y), 5.0, p.eye, seed=62)
    st.circle((face_x + eye_d, eye_y), 5.0, p.eye, seed=63)

    # Near arm behind the bar, then the bar through both plates, then hands on top.
    st.capsule(j["shoulder_near"], j["elbow_near"], p.limb_r * 0.95, p.arm, seed=71)
    st.capsule(j["elbow_near"], j["hand_near"], p.limb_r * 0.84, p.arm, seed=72)

    # Right plate, then the bar laid across both plate centres so it is
    # visibly attached, then a short collar past each plate.
    st.ellipse(j["plate_r"], p.plate_rx, p.plate_ry, p.plate, seed=81)
    st.rounded_bar(j["bar_l"], j["bar_r"], p.bar_h, p.bar, seed=82)
    # Small inner "holes" are implied by the bar crossing the plates.

    # Hands grip the bar: paper discs in front of the bar.
    st.circle(j["hand_far"], p.hand_r, p.arm, seed=91)
    st.circle(j["hand_near"], p.hand_r, p.arm, seed=92)

    return Image.alpha_composite(bg.convert("RGBA"), st.canvas).convert("RGB")
