"""Top-centre black pill label, white bold uppercase, bottom 15% left clear."""

from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/sand-box/google/Poppins/Poppins-Bold.ttf"

# 1080x1920 defaults
FONT_SIZE = 54
PAD_X, PAD_Y = 34, 14
RADIUS = 30
TOP = 72
STROKE = 4


def _font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, FONT_SIZE)


def draw_label(im: Image.Image, text: str) -> Image.Image:
    text = " ".join(text.strip().upper().split())
    words = text.split()
    if len(words) > 3:
        raise ValueError(f"label must be <=3 words, got {len(words)}: {text!r}")
    canvas = im.convert("RGBA")
    w, h = canvas.size
    # Scale the pill if the canvas is not 1080 wide.
    s = w / 1080.0
    font = ImageFont.truetype(FONT_PATH, max(18, int(FONT_SIZE * s)))
    pad_x, pad_y = int(PAD_X * s), int(PAD_Y * s)
    radius = int(RADIUS * s)
    top = int(TOP * s)
    stroke = max(2, int(STROKE * s))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    x0, y0, x1, y1 = font.getbbox(text)
    tw, th = x1 - x0, y1 - y0
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    bx = (w - box_w) // 2
    by = top

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        [bx - 1, by + 3, bx + box_w + 1, by + box_h + 3],
        radius=radius + 1,
        fill=(0, 0, 0, 90),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(1, int(3 * s))))
    overlay = Image.alpha_composite(shadow, overlay)
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle(
        [bx - stroke, by - stroke, bx + box_w + stroke, by + box_h + stroke],
        radius=radius + stroke,
        fill=(255, 255, 255, 230),
    )
    d.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=radius,
        fill=(0, 0, 0, 245),
    )
    tx = bx + (box_w - tw) // 2 - x0
    ty = by + (box_h - th) // 2 - y0
    d.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))
    return Image.alpha_composite(canvas, overlay).convert("RGB")
