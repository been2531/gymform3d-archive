"""Glowing neon stickman on black. Cool, minimal, fixed line set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from PIL import Image, ImageDraw

from . import drawutil as du
from .skeleton import Joints, Proportions, anterior, dist, muscle_center, norm, _sub, _add, _mul

Point = Tuple[float, float]


@dataclass
class NeonParams:
    line_w: float = 11.0
    joint_r: float = 10.0
    head_w: float = 8.0
    bar_w: float = 12.0
    plate_r: float = 62.0
    plate_ring: float = 8.0
    glow: float = 26.0
    glow_gain: float = 1.15
    muscle_w: float = 28.0
    muscle_out: float = 16.0
    cyan: Tuple[int, int, int] = (40, 210, 255)
    core: Tuple[int, int, int] = (220, 250, 255)
    magenta: Tuple[int, int, int] = (255, 50, 170)
    magenta_core: Tuple[int, int, int] = (255, 210, 240)
    far_dim: float = 0.72


V1 = NeonParams()
# Round 2 tunables are filled in after the first stills are judged.
V2 = NeonParams(
    line_w=10.0,
    joint_r=9.5,
    head_w=7.5,
    bar_w=11.0,
    plate_r=50.0,
    plate_ring=7.0,
    glow=32.0,
    glow_gain=1.35,
    muscle_w=30.0,
    muscle_out=17.0,
)


def _rgba(rgb: Tuple[int, int, int], a: int) -> Tuple[int, int, int, int]:
    return (rgb[0], rgb[1], rgb[2], a)


def _segments(j: Joints, far: bool) -> list:
    if far:
        return [
            (j["shoulder_far"], j["elbow_far"]),
            (j["elbow_far"], j["hand_far"]),
            (j["hip_far"], j["knee_far"]),
            (j["knee_far"], j["ankle_far"]),
            (j["ankle_far"], j["toe_far"]),
        ]
    return [
        (j["shoulder_near"], j["elbow_near"]),
        (j["elbow_near"], j["hand_near"]),
        (j["hip_near"], j["knee_near"]),
        (j["knee_near"], j["ankle_near"]),
        (j["ankle_near"], j["toe_near"]),
    ]


def _torso(j: Joints) -> list:
    # Minimal trunk: short neck, one spine, one shoulder line. No rib cage.
    chin = (j["head"][0], j["head"][1] + 78.0 * 0.92)
    return [
        (chin, j["shoulder_mid"]),
        (j["shoulder_mid"], j["hip_mid"]),
        (j["shoulder_far"], j["shoulder_near"]),
    ]


def _joints(j: Joints, far: bool, r: float) -> list:
    if far:
        keys = ("shoulder_far", "elbow_far", "hand_far", "hip_far", "knee_far", "ankle_far")
    else:
        keys = ("shoulder_near", "elbow_near", "hand_near", "hip_near", "knee_near", "ankle_near")
    return [(j[k], r) for k in keys]


def _muscle_poly(j: Joints, side: str, width: float, out: float, pulse: float):
    hip = j[f"hip_{side}"]
    knee = j[f"knee_{side}"]
    w = width * (0.88 + 0.22 * pulse)
    o = out * (0.90 + 0.20 * pulse)
    return du.tapered_muscle(hip, knee, width=w, along0=0.06, along1=0.68, out=o)


def _glute_poly(j: Joints, pulse: float):
    return du.glute_shape(j["hip_mid"], j["hip_far"], j["hip_near"], 0.85 + 0.25 * pulse)


def _draw_head_ring(size, center, radius, width, color, alpha) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = center
    # Outer ring minus inner hole, as a thick ellipse.
    d.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=_rgba(color, alpha),
        width=max(2, int(width)),
    )
    return layer


def _plates_and_bar(size, j: Joints, p: NeonParams, alpha: int) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _rgba(p.cyan, alpha)
    # Bar first as a capsule so it runs through both plate centres.
    du.draw_capsule(d, j["bar_l"], j["bar_r"], p.bar_w / 2.0, col)
    for key in ("plate_l", "plate_r"):
        c = j[key]
        # Two concentric rings so the bar is visibly threaded through.
        du.draw_circle(d, c, p.plate_r, (0, 0, 0, 0))
        d.ellipse(
            (c[0] - p.plate_r, c[1] - p.plate_r, c[0] + p.plate_r, c[1] + p.plate_r),
            outline=col,
            width=max(2, int(p.plate_ring)),
        )
        inner = p.plate_r * 0.55
        d.ellipse(
            (c[0] - inner, c[1] - inner, c[0] + inner, c[1] + inner),
            outline=col,
            width=max(2, int(p.plate_ring * 0.7)),
        )
    # Redraw the bar over the rings so it clearly crosses the plate holes.
    du.draw_capsule(d, j["bar_l"], j["bar_r"], p.bar_w / 2.0, col)
    # Hands as grip dots on the bar.
    du.draw_circle(d, j["hand_far"], p.joint_r + 2, col)
    du.draw_circle(d, j["hand_near"], p.joint_r + 2, col)
    return layer


def render(
    joints: Joints,
    size: Tuple[int, int] = (1080, 1920),
    pulse: float = 1.0,
    params: NeonParams | None = None,
) -> Image.Image:
    p = params or V1
    canvas = Image.new("RGB", size, (0, 0, 0))

    # One cyan layer (far + torso + near + bar + head) and one muscle layer.
    # Glowing each once is dramatically cheaper than a pass per limb.
    far_a = int(255 * p.far_dim)
    cyan = du.line_layer(
        size,
        _segments(joints, True),
        _rgba(p.cyan, far_a),
        p.line_w,
        _joints(joints, True, p.joint_r),
    )
    cyan = Image.alpha_composite(
        cyan,
        du.line_layer(
            size,
            _torso(joints) + _segments(joints, False),
            _rgba(p.cyan, 255),
            p.line_w,
            _joints(joints, False, p.joint_r)
            + [(joints["neck"], p.joint_r), (joints["hip_mid"], p.joint_r)],
        ),
    )
    bar = _plates_and_bar(size, joints, p, 255)
    cyan = Image.alpha_composite(cyan, bar)
    head_r = 78.0
    head = _draw_head_ring(size, joints["head"], head_r, p.head_w, p.cyan, 255)
    cyan = Image.alpha_composite(cyan, head)

    m_alpha = int(255 * (0.55 + 0.45 * pulse))
    m_gain = p.glow_gain * (0.85 + 0.55 * pulse)
    ml = Image.new("RGBA", size, (0, 0, 0, 0))
    md = ImageDraw.Draw(ml)
    for side in ("far", "near"):
        md.polygon(
            _muscle_poly(joints, side, p.muscle_w, p.muscle_out, pulse),
            fill=_rgba(p.magenta, m_alpha),
        )

    canvas = du.add_glow(canvas, du.fast_glow(cyan, p.glow), gain=p.glow_gain)
    canvas = du.add_glow(canvas, du.fast_glow(cyan, max(4.0, p.glow * 0.35)), gain=p.glow_gain * 0.7)
    canvas = du.add_glow(canvas, du.fast_glow(ml, p.glow * 1.15), gain=m_gain)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), cyan).convert("RGB")
    canvas = Image.alpha_composite(canvas.convert("RGBA"), ml).convert("RGB")

    core_m = Image.new("RGBA", size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(core_m)
    for side in ("far", "near"):
        poly = _muscle_poly(joints, side, p.muscle_w * 0.45, p.muscle_out, pulse)
        cd.polygon(poly, fill=_rgba(p.magenta_core, int(200 + 55 * pulse)))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), core_m).convert("RGB")

    white = du.line_layer(
        size,
        _torso(joints) + _segments(joints, True) + _segments(joints, False),
        (230, 250, 255, 220),
        max(2.5, p.line_w * 0.35),
        _joints(joints, True, p.joint_r * 0.55) + _joints(joints, False, p.joint_r * 0.55),
    )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), white).convert("RGB")
    return canvas
