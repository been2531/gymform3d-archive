"""Two 1080x1350 (4:5) carousel cards built from the same JSON data as the reel."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import media as M
from . import theme as T

W, H = 1080, 1350
MARGIN = 72


def _grid(size, step, color, alpha, width=1):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for x in range(0, size[0], step):
        d.line([(x, 0), (x, size[1])], fill=color + (alpha,), width=width)
    for y in range(0, size[1], step):
        d.line([(0, y), (size[0], y)], fill=color + (alpha,), width=width)
    return im


def _background(pal) -> Image.Image:
    bg = Image.new("RGBA", (W, H), T.hex_rgb(pal["bg"]) + (255,))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    d.ellipse((-260, -520, W + 260, 700), fill=T.hex_rgb("#16281A") + (120,))
    glow = glow.filter(ImageFilter.GaussianBlur(160))
    bg = Image.alpha_composite(bg, glow)
    bg = Image.alpha_composite(bg, _grid((W, H), 54, T.hex_rgb("#FFFFFF"), 7))
    arr = np.asarray(bg.convert("RGB"), np.float32) / 255.0
    arr *= M.vignette(W, H, strength=0.42, power=1.8)[..., None]
    arr += np.random.default_rng(4).normal(0, 1, (H, W, 1)).astype(np.float32) * 0.010
    return M.to_pil(np.clip(arr, 0, 1)).convert("RGBA")


def display_block(d, x, y_top, lines, size, max_w, fill, gap_ratio=0.10, tracking=0.5):
    """Draw display lines top-aligned by ink and return the next free y."""
    y = y_top
    for line in lines:
        fs = T.fit_size(line, "display", max_w, size, 40, tracking)
        f = T.font("display", fs)
        cap = -f.getbbox("H", anchor="ls")[1]
        T.draw_text(d, (x, y + cap), line, f, fill, anchor="ls", tracking=tracking)
        y += cap + int(fs * gap_ratio)
    return y


def _topbar(d, cfg, idx: str):
    pal = cfg["palette"]
    d.ellipse((MARGIN, 74, MARGIN + 16, 90), fill=T.hex_rgb(pal["lime"]) + (255,))
    T.draw_text(d, (MARGIN + 30, 82), cfg["brand"]["handle"].upper(), T.font("cond_bold", 30),
                T.hex_rgb(pal["paper"]) + (235,), anchor="lm", tracking=3.2)
    T.draw_text(d, (W - MARGIN, 82), idx, T.font("cond_semi", 28),
                T.hex_rgb(pal["muted"]) + (255,), anchor="rm", tracking=2.0)


def _footer(d, cfg, text: str):
    pal = cfg["palette"]
    y = H - 74
    d.line([(MARGIN, y - 44), (W - MARGIN, y - 44)], fill=T.hex_rgb("#2A2D30") + (255,), width=2)
    T.draw_text(d, (MARGIN, y), text, T.font("cond_bold", 31),
                T.hex_rgb("#D6D9D3") + (255,), anchor="lm", tracking=2.2)
    T.draw_text(d, (W - MARGIN, y), "@" + cfg["brand"]["handle"], T.font("cond_semi", 28),
                T.hex_rgb(pal["lime"]) + (255,), anchor="rm", tracking=1.8)


def _panel(img: Image.Image, size, radius: int = 22, border: str = "#2A2D30") -> Image.Image:
    pw, ph = size
    iw, ih = img.size
    s = max(pw / iw, ph / ih)
    rs = img.resize((int(iw * s + 0.5), int(ih * s + 0.5)), Image.LANCZOS)
    x0 = (rs.width - pw) // 2
    y0 = (rs.height - ph) // 2
    crop = rs.crop((x0, y0, x0 + pw, y0 + ph)).convert("RGBA")
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=radius, fill=255)
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.paste(crop, (0, 0), mask)
    ImageDraw.Draw(out).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=radius,
                                          outline=T.hex_rgb(border) + (255,), width=2)
    return out


def _circle(img: Image.Image, dia: int, border: str = "#3A3E42") -> Image.Image:
    iw, ih = img.size
    s = dia / min(iw, ih)
    rs = img.resize((max(dia, int(iw * s)), max(dia, int(ih * s))), Image.LANCZOS)
    x0 = (rs.width - dia) // 2
    y0 = (rs.height - dia) // 2
    crop = rs.crop((x0, y0, x0 + dia, y0 + dia)).convert("RGBA")
    mask = Image.new("L", (dia, dia), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, dia - 1, dia - 1), fill=255)
    out = Image.new("RGBA", (dia, dia), (0, 0, 0, 0))
    out.paste(crop, (0, 0), mask)
    ImageDraw.Draw(out).ellipse((0, 0, dia - 1, dia - 1),
                                outline=T.hex_rgb(border) + (255,), width=3)
    return out


# ------------------------------------------------------------------ card 1 --

def card_form(cfg: dict, root: str) -> Image.Image:
    c = cfg["carousel"]["card1"]
    pal = cfg["palette"]
    base = _background(pal)
    img = Image.open(os.path.join(root, cfg["assets"]["form"])).convert("RGB")
    d = ImageDraw.Draw(base)

    _topbar(d, cfg, "01 / 02")
    T.draw_text(d, (MARGIN, 166), c["eyebrow"], T.font("cond_bold", 34),
                T.hex_rgb(pal["lime"]) + (255,), anchor="lm", tracking=5.0)

    y = display_block(d, MARGIN, 202, c["headline"], 98, W - 2 * MARGIN,
                      T.hex_rgb(pal["paper"]) + (255,), gap_ratio=0.17)
    T.draw_text(d, (MARGIN, y + 30), c["sub"], T.font("cond_semi", 30),
                T.hex_rgb(pal["muted"]) + (255,), anchor="lm", tracking=2.2)

    pw, ph, py = 440, 462, y + 74
    for i, (side, key) in enumerate((("left", "wrong"), ("right", "safe"))):
        crop = img.crop(tuple(c["crops"][side]))
        panel = _panel(crop, (pw, ph), 24)
        px = MARGIN + i * (pw + 56)
        base.alpha_composite(panel, (px, py))
        spec = c[key]
        col = T.hex_rgb(pal["red"] if key == "wrong" else pal["green"])
        bw, bh = 158, 64
        badge = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        bd = ImageDraw.Draw(badge)
        bd.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=32, fill=T.hex_rgb("#050706") + (220,),
                             outline=col + (255,), width=2)
        T.draw_text(bd, (bw / 2, bh / 2 - 1), f"{spec['label']} {spec['angle']}",
                    T.font("display", 31), col + (255,), anchor="mm", tracking=0.8)
        base.alpha_composite(badge, (px + 18, py + 16))

    sy = py + ph + 40
    for i, (side, key) in enumerate((("left", "wrong"), ("right", "safe"))):
        spec = c[key]
        col = T.hex_rgb(pal["red"] if key == "wrong" else pal["green"])
        px = MARGIN + i * (pw + 56)
        T.draw_text(d, (px, sy), f"{spec['label']}  {spec['angle']}", T.font("display", 48),
                    col + (255,), anchor="lm", tracking=0.6)
        ty = sy + 52
        for line in spec["cue"]:
            T.draw_text(d, (px, ty), line, T.font("cond_semi", 30),
                        T.hex_rgb("#C9CDC7") + (255,), anchor="lm", tracking=2.0)
            ty += 35

    # the fix strip
    fy0, fy1 = 1160, 1226
    d.rounded_rectangle((MARGIN, fy0, W - MARGIN, fy1), radius=16,
                        fill=T.hex_rgb("#101512") + (255,),
                        outline=T.hex_rgb(pal["lime"]) + (140,), width=2)
    T.draw_text(d, (MARGIN + 26, (fy0 + fy1) / 2), c["fix_label"], T.font("cond_bold", 28),
                T.hex_rgb(pal["lime"]) + (255,), anchor="lm", tracking=3.0)
    T.draw_text(d, (W - MARGIN - 26, (fy0 + fy1) / 2), c["fix"], T.font("cond_bold", 30),
                T.hex_rgb(pal["paper"]) + (255,), anchor="rm", tracking=1.8)

    _footer(d, cfg, c["foot"])
    print(f"  card1: headline_bottom={y} sub={y+30} panels={py}-{py+ph} "
          f"stats={sy} cues_end={ty} fix=1160-1226")
    return base.convert("RGB")


# ------------------------------------------------------------------ card 2 --

def card_protein(cfg: dict, root: str) -> Image.Image:
    c = cfg["carousel"]["card2"]
    pal = cfg["palette"]
    base = _background(pal)
    img = Image.open(os.path.join(root, cfg["assets"]["protein"])).convert("RGB")
    d = ImageDraw.Draw(base)

    _topbar(d, cfg, "02 / 02")
    T.draw_text(d, (MARGIN, 166), c["eyebrow"], T.font("cond_bold", 34),
                T.hex_rgb(pal["lime"]) + (255,), anchor="lm", tracking=5.0)

    y = display_block(d, MARGIN, 202, c["headline"], 96, W - 2 * MARGIN,
                      T.hex_rgb(pal["paper"]) + (255,), gap_ratio=0.17)
    T.draw_text(d, (MARGIN, y + 30), c["sub"], T.font("cond_semi", 30),
                T.hex_rgb(pal["muted"]) + (255,), anchor="lm", tracking=2.2)

    dia, py = 344, y + 72
    xs = {0: MARGIN + dia // 2 + 10, 1: W - MARGIN - dia // 2 - 10}
    for i, (side, key) in enumerate((("left", "left"), ("right", "right"))):
        crop = img.crop(tuple(c["crops"][side]))
        circ = _circle(crop, dia)
        cx = xs[i]
        base.alpha_composite(circ, (cx - dia // 2, py))
        spec = c[key]
        tag = spec["tag"]
        tw = T.text_len(tag, T.font("cond_bold", 26), 3.0) + 34
        tx, ty = cx - tw / 2, py + dia + 6
        d.rounded_rectangle((tx, ty, tx + tw, ty + 40), radius=20,
                            fill=T.hex_rgb(pal["lime"]) + (255,))
        T.draw_text(d, (cx, ty + 20), tag, T.font("cond_bold", 26),
                    T.hex_rgb(pal["ink"]) + (255,), anchor="mm", tracking=3.0)
        ny = ty + 62
        for line in spec["title"]:
            T.draw_text(d, (cx, ny), line, T.font("cond_bold", 33),
                        T.hex_rgb(pal["paper"]) + (255,), anchor="mm", tracking=1.4)
            ny += 38

    # macro table
    lv = cfg["reel"]["macro"]["left"]["values"]
    rv = cfg["reel"]["macro"]["right"]["values"]
    rows = [("CALORIES", lv["calories"], rv["calories"], "kcal", "amber"),
            ("PROTEIN", lv["protein"], rv["protein"], "g", "green"),
            ("CARBS", lv["carbs"], rv["carbs"], "g", "amber"),
            ("FIBER", lv["fiber"], rv["fiber"], "g", "green")]
    hx, vx1, vx2 = MARGIN, 560, W - MARGIN
    ty = py + dia + 170
    d.line([(hx, ty), (W - MARGIN, ty)], fill=T.hex_rgb("#2A2D30"), width=2)
    T.draw_text(d, (hx, ty + 26), "PER SERVING", T.font("cond_semi", 25),
                T.hex_rgb(pal["muted"]) + (255,), anchor="lm", tracking=2.6)
    T.draw_text(d, (vx1, ty + 26), "PLANT", T.font("cond_semi", 25),
                T.hex_rgb(pal["muted"]) + (255,), anchor="rm", tracking=2.6)
    T.draw_text(d, (vx2, ty + 26), "ANIMAL", T.font("cond_semi", 25),
                T.hex_rgb(pal["muted"]) + (255,), anchor="rm", tracking=2.6)
    for i, (label, a, b, unit, colkey) in enumerate(rows):
        ry = ty + 70 + i * 48
        T.draw_text(d, (hx, ry), label, T.font("cond_semi", 29),
                    T.hex_rgb("#C9CDC7") + (255,), anchor="lm", tracking=2.4)
        accent = T.hex_rgb(pal[colkey])
        d.rounded_rectangle((hx + 172, ry - 5, hx + 194, ry + 7), radius=6, fill=accent + (255,))
        T.draw_text(d, (vx1, ry), f"{a}", T.font("display", 38),
                    T.hex_rgb(pal["paper"]) + (255,), anchor="rm")
        T.draw_text(d, (vx2, ry), f"{b}", T.font("display", 38),
                    T.hex_rgb(pal["paper"]) + (255,), anchor="rm")
        if unit:
            T.draw_text(d, (vx1 + 6, ry + 2), unit, T.font("cond_semi", 22),
                        T.hex_rgb(pal["muted"]) + (255,), anchor="lm", tracking=1.0)
            T.draw_text(d, (vx2 + 6, ry + 2), unit, T.font("cond_semi", 22),
                        T.hex_rgb(pal["muted"]) + (255,), anchor="lm", tracking=1.0)

    _footer(d, cfg, c["foot"])
    print(f"  card2: headline_bottom={y} sub={y+30} plates={py}-{py+dia} "
          f"names_end={ny} table={ty}-{ry}")
    return base.convert("RGB")


def render_all(cfg: dict, root: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    p1 = os.path.join(outdir, "gymform_carousel_1.png")
    p2 = os.path.join(outdir, "gymform_carousel_2.png")
    card_form(cfg, root).save(p1)
    card_protein(cfg, root).save(p2)
    return [("card1", p1), ("card2", p2)]
