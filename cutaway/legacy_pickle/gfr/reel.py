"""Reel frame builder: 15s / 1080x1920 / 30fps, driven entirely by the JSON config."""
from __future__ import annotations

import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from . import media as M
from . import theme as T

SW = 941          # source art width
SH = 1672         # source art height


class Map:
    """Maps source-art pixels to output pixels for one Ken Burns crop."""

    def __init__(self, box, out_w: int, out_h: int):
        self.x0, self.y0, self.x1, self.y1 = box
        self.kx = out_w / (self.x1 - self.x0)
        self.ky = out_h / (self.y1 - self.y0)

    def pt(self, sx: float, sy: float):
        return ((sx - self.x0) * self.kx, (sy - self.y0) * self.ky)

    def sz(self, sw: float, sh: float):
        return (sw * self.kx, sh * self.ky)


class Reel:
    def __init__(self, cfg: dict, root: str):
        self.cfg = cfg
        self.root = root
        f = cfg["format"]
        self.W, self.H, self.fps, self.dur = f["width"], f["height"], f["fps"], f["duration"]
        self.pal = cfg["palette"]
        self.r = cfg["reel"]
        self.form = M.Source(os.path.join(root, cfg["assets"]["form"]), self.W, self.H)
        self.prot = M.Source(os.path.join(root, cfg["assets"]["protein"]), self.W, self.H)
        self._prepare()

    # ------------------------------------------------------------- prepare --
    def _prepare(self):
        W, H = self.W, self.H
        fb = self.r["form"]["sprite_boxes"]
        lb, rb = tuple(fb["left"]), tuple(fb["right"])
        self.sp_l = self.form.sprite(lb, gain=1.06)
        self.sp_r = self.form.sprite(rb, gain=1.06)
        self.form_clean = self.form.img.copy()

        pimg = self.prot.img.copy()
        pimg = self._patch(pimg, (36, 1104, 906, 1470), feather=18)    # baked stat cards
        pimg = self._patch(pimg, (44, 1436, 898, 1600), feather=16)    # baked CTA pill
        pimg = self._patch(pimg, (96, 1568, 852, 1650), feather=14)    # baked save/share/like row
        self.prot_clean = pimg

        self.vig = M.vignette(W, H, strength=0.30, power=2.0)[..., None]
        self.grain = M.grain_bank(W, H, count=6, seed=11)
        self.band = self._build_band_base()
        self.scrim = self._build_scrim()

    @staticmethod
    def _patch(img: Image.Image, box, feather: int = 14) -> Image.Image:
        """Fill a region with the row-wise background colour either side of it."""
        x0, y0, x1, y1 = [int(v) for v in box]
        iw, ih = img.size
        arr = np.asarray(img, np.float32)
        out = arr.copy()
        left = arr[y0:y1, max(0, x0 - 12):x0]
        right = arr[y0:y1, x1:min(iw, x1 + 12)]
        rowcol = np.median(np.concatenate([left, right], axis=1), axis=1)
        k = np.ones(31, np.float32) / 31
        for ch in range(3):
            rowcol[:, ch] = np.convolve(rowcol[:, ch], k, mode="same")
        fill = rowcol[:, None, :]
        mask = np.asarray(T.soft_mask_from_box((iw, ih), (x0, y0, x1, y1), feather),
                          np.float32)[..., None] / 255.0
        m = mask[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * (1 - m) + fill * m
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def _patch_smooth(img: Image.Image, box, ring: int = 44, feather: int = 14) -> Image.Image:
        """Fill a region with a quadratic surface fitted to the surrounding ring."""
        x0, y0, x1, y1 = [int(v) for v in box]
        iw, ih = img.size
        arr = np.asarray(img, np.float32)
        out = arr.copy()
        rx0, ry0 = max(0, x0 - ring), max(0, y0 - ring)
        rx1, ry1 = min(iw, x1 + ring), min(ih, y1 + ring)
        ys, xs = np.mgrid[ry0:ry1:2, rx0:rx1:2]
        band = arr[ry0:ry1:2, rx0:rx1:2]
        inside = (xs >= x0 - 2) & (xs < x1 + 2) & (ys >= y0 - 2) & (ys < y1 + 2)
        px, py = xs[~inside].astype(np.float32), ys[~inside].astype(np.float32)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        u, v = (px - cx) / max(1, x1 - x0), (py - cy) / max(1, y1 - y0)
        A = np.stack([np.ones_like(u), u, v, u * u, v * v, u * v], axis=1)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        gu, gv = (gx - cx) / max(1, x1 - x0), (gy - cy) / max(1, y1 - y0)
        Ag = np.stack([np.ones_like(gu), gu, gv, gu * gu, gv * gv, gu * gv], axis=2)
        fill = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        for ch in range(3):
            coef, *_ = np.linalg.lstsq(A, band[~inside][:, ch], rcond=None)
            fill[:, :, ch] = np.clip(np.einsum("hwc,c->hw", Ag, coef), 0, 255)
        mask = np.asarray(T.soft_mask_from_box((iw, ih), (x0, y0, x1, y1), feather),
                          np.float32)[..., None] / 255.0
        m = mask[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * (1 - m) + fill * m
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    # --------------------------------------------------------------- band ---
    def _band_geom(self):
        return dict(x0=44, y0=1268, x1=1036, y1=1672,
                    lcol=(78, 508), rcol=(572, 1002), div=540,
                    header_y=1292, rule_y=1326, row_top=1342, row_h=80,
                    pad_x=34)

    def _build_band_base(self) -> Image.Image:
        g = self._band_geom()
        w, h = g["x1"] - g["x0"], g["y1"] - g["y0"]
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=24,
                            fill=T.hex_rgb("#0B0D0F") + (250,),
                            outline=T.hex_rgb("#2A2E32") + (255,), width=2)
        d.rounded_rectangle((26, 1, w - 26, 5), radius=2, fill=T.hex_rgb(self.pal["lime"]) + (215,))
        d.line([(24, g["rule_y"] - g["y0"]), (w - 24, g["rule_y"] - g["y0"])],
               fill=T.hex_rgb("#1F2327") + (255,), width=2)
        d.line([(g["div"] - g["x0"], g["rule_y"] - g["y0"] + 6), (g["div"] - g["x0"], h - 22)],
               fill=T.hex_rgb("#202428") + (255,), width=2)
        return im

    def _build_scrim(self) -> Image.Image:
        W, H = self.W, self.H
        arr = np.zeros((H, W, 4), np.float32)
        ys = np.arange(H, dtype=np.float32)[:, None]
        arr[:, :, 3] = np.clip((ys - 1790) / 80.0, 0, 1) * 0.90
        arr[:, :, 0:3] = np.array(T.rgb_f("#060708"), np.float32)
        return Image.fromarray((arr * 255).astype(np.uint8), "RGBA")

    # -------------------------------------------------------------- scenes --
    def frame(self, t: float) -> np.ndarray:
        sc = self.r["scenes"]
        if t < sc["form"][1]:
            c = self.scene_form(t)
        elif t < sc["cut"][1]:
            c = self.scene_cut(t)
        else:
            c = self.scene_macro(t)
        gi = int(t * self.fps) % len(self.grain)
        c = (c + self.grain[gi] * 0.012) * self.vig
        self._watermark(c)
        return np.clip(c, 0, 1)

    # --- scene 1: form check ------------------------------------------------
    def scene_form(self, t: float) -> np.ndarray:
        f = self.r["form"]
        sw = f["switch"]
        a, b = self.r["scenes"]["form"]
        p = T.inv(a, b, t)
        zoom = T.lerp(f["zoom"][0], f["zoom"][1], T.ease_in_out(p))
        punch = 1 + 0.05 * (1 - T.ease_out_cubic(min(1.0, t / 0.40)))
        punch += 0.012 * np.sin(np.pi * T.inv(sw - 0.02, sw + 0.30, t))
        cy = T.lerp(0.492, 0.500, T.ease_in_out(p))
        img, box = self._ken_burns(self.form_clean, zoom * punch, 0.5, cy)
        c = M.from_pil(img)
        mp = Map(box, self.W, self.H)

        red_out = 1 - T.inv(sw, sw + 0.22, t)
        grn_in = T.inv(sw + 0.03, sw + 0.42, t)

        if red_out > 0.01:
            amt = (0.20 + 0.26 * T.pulse(t, 0.82)) * red_out
            self._numeral_glow(c, mp, "left", amt, self.pal["red"], 1.0)
            self._tick_frame(c, mp, "left", (0.35 + 0.45 * T.pulse(t, 0.82)) * red_out, self.pal["red"])
        if grn_in > 0.01:
            settle = 1 - 0.30 * T.inv(sw + 0.4, sw + 1.8, t)
            amt = (0.18 + 0.24 * T.pulse(t + 0.4, 1.0)) * grn_in * settle
            self._numeral_glow(c, mp, "right", amt, self.pal["green"], 1.06)
            self._tick_frame(c, mp, "right", (0.4 + 0.45 * T.pulse(t + 0.4, 1.0)) * grn_in,
                             self.pal["green"])
            self._sweep(c, T.inv(sw, sw + 0.55, t))

        self._numeral_flash(c, mp, "left", self.pal["red"], 0.9 * np.sin(np.pi * T.inv(0.05, 0.55, t)))
        self._numeral_flash(c, mp, "right", self.pal["green"], 1.5 * np.sin(np.pi * T.inv(sw, sw + 0.42, t)))

        fl = T.inv(sw, sw + 0.06, t) * (1 - T.inv(sw + 0.06, sw + 0.26, t))
        if fl > 0.001:
            c = c + fl * 0.08

        cb = T.inv(sw + 0.26, sw + 0.52, t)
        if cb > 0.001:
            self._check_badge(c, mp, cb)
        return c

    def _ken_burns(self, img: Image.Image, scale: float, cx: float, cy: float):
        iw, ih = img.size
        cw = iw / max(1e-3, scale)
        ch = cw * self.H / self.W
        if ch > ih:
            ch = ih
            cw = ch * self.W / self.H
        x0 = min(max(0.0, cx * iw - cw / 2), max(0.0, iw - cw))
        y0 = min(max(0.0, cy * ih - ch / 2), max(0.0, ih - ch))
        out = img.resize((self.W, self.H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
        return out, (x0, y0, x0 + cw, y0 + ch)

    def _numeral_flash(self, c, mp, side: str, color: str, amount: float):
        """Light up the baked numerals through their own alpha mask (no re-typing)."""
        if amount <= 0.004:
            return
        box = self.r["form"]["sprite_boxes"][side]
        sp = self.sp_l if side == "left" else self.sp_r
        x, y = mp.pt(box[0], box[1])
        w, h = mp.sz(box[2] - box[0], box[3] - box[1])
        w, h = max(1, int(w)), max(1, int(h))
        a = np.asarray(sp.getchannel("A").resize((w, h), Image.BILINEAR), np.float32) / 255.0
        a = np.clip((a - 0.07) * 3.0, 0, 1)      # keep the art's near-black off the mask
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = min(self.W, x0 + w), min(self.H, y0 + h)
        if x1 <= x0 or y1 <= y0:
            return
        am = a[:y1 - y0, :x1 - x0][..., None]
        c[y0:y1, x0:x1] += np.asarray(T.rgb_f(color), np.float32) * (am * amount)

    def _numeral_glow(self, c, mp, side: str, amt: float, color: str, spread: float):
        box = self.r["form"]["sprite_boxes"][side]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        r = 190 * spread
        rgb, alpha = self._glow_patch(r, color, 0.38)
        x, y = mp.pt(cx - r, cy - r)
        w, h = mp.sz(2 * r, 2 * r)
        w, h = max(1, int(w)), max(1, int(h))
        sp = rgb.resize((w, h), Image.BILINEAR)
        sp.putalpha(alpha.resize((w, h), Image.BILINEAR))
        M.blend_sprite(c, sp, x, y, amt)

    @staticmethod
    def _glow_patch(r, color, strength):
        size = (int(2 * r), int(2 * r))
        rgb = Image.new("RGB", size, T.hex_rgb(color))
        yy, xx = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
        d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / r
        a = np.clip(1 - d, 0, 1) ** 2.0 * strength
        return rgb, Image.fromarray((a * 255).astype(np.uint8), "L")

    def _tick_frame(self, c, mp, side: str, amt: float, color: str):
        box = self.r["form"]["frame_boxes"][side]
        x0, y0 = mp.pt(box[0], box[1])
        x1, y1 = mp.pt(box[2], box[3])
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        col = T.hex_rgb(color) + (int(205 * min(1.0, amt)),)
        arm, wd = 46, 4
        for (ax, ay, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            d.line([(ax, ay), (ax + arm * dx, ay)], fill=col, width=wd)
            d.line([(ax, ay), (ax, ay + arm * dy)], fill=col, width=wd)
        M.blend_sprite(c, layer, 0, 0, 1.0)

    def _sweep(self, c, p: float):
        if p <= 0.001 or p >= 0.999:
            return
        W, H = self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        pos = T.ease_in_out(p) * 1.30 - 0.15
        band = np.abs((xx / W) * 0.8 + (yy / H) * 0.55 - pos)
        a = np.clip(1 - band / 0.055, 0, 1) ** 2 * (1 - p) ** 0.6
        rgb = np.asarray(T.rgb_f(self.pal["lime"]), np.float32)
        c += rgb * (a[..., None] * 0.38)

    def _check_badge(self, c, mp, t01: float):
        s = T.ease_out_back(t01, 2.6)
        if s <= 0.01:
            return
        size = 86 * s
        x, y = mp.pt(872, 1206)
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        cx, cy = x, y
        d.ellipse((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
                  fill=T.hex_rgb(self.pal["green"]) + (238,))
        T.icon_check(d, cx - size * 0.24, cy - size * 0.28, size * 0.48,
                     T.hex_rgb("#06110A") + (255,), max(3, int(size * 0.11)))
        M.blend_sprite(c, layer, 0, 0, min(1.0, t01 * 3))

    # --- transition ---------------------------------------------------------
    def scene_cut(self, t: float) -> np.ndarray:
        a, b = self.r["scenes"]["cut"]
        p = T.inv(a, b, t)
        W, H = self.W, self.H
        c = np.zeros((H, W, 3), np.float32)
        mv = T.ease_in_out(min(1.0, p / 0.60))
        off1 = int(-W * 1.06 * mv)
        off2 = int(W * (1 - mv))
        i1, _ = self._ken_burns(self.form_clean, 1.07, 0.5, 0.55)
        i2, _ = self._ken_burns(self.prot_clean, 1.0, 0.5, 0.5)
        a1 = M.from_pil(i1)
        a2 = M.from_pil(i2)
        for im, ox in ((a1, off1), (a2, off2)):
            x0, x1 = max(0, ox), min(W, ox + W)
            if x1 > x0:
                c[:, x0:x1] += im[:, x0 - ox:x1 - ox]

        # stats panel lands during the wipe, so the block is never blank
        self._place_band(c, t)

        cfg = self.r["cut"]
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        bh = 176
        by = int(836 - 34 * (1 - T.ease_out_cubic(min(1.0, p / 0.35))))
        alpha = int(255 * min(1.0, p * 7))
        d.rectangle((0, by, W, by + bh), fill=T.hex_rgb(self.pal["lime"]) + (alpha,))
        f1 = T.font("display", 90)
        T.draw_text(d, (W / 2, by + bh / 2 + 2), f"{cfg['line1']} {cfg['line2']}", f1,
                    T.hex_rgb(self.pal["ink"]) + (alpha,), anchor="mm", tracking=1.5)
        M.blend_sprite(c, layer, 0, 0, 1.0)
        fl = max(0.0, 1 - p / 0.30)
        c = c + fl * 0.15
        return c

    # --- scene 3: macros + CTA ---------------------------------------------
    def scene_macro(self, t: float) -> np.ndarray:
        a, _ = self.r["scenes"]["macro"]
        cta_a = self.r["scenes"]["cta"][0]
        img, _ = self._ken_burns(self.prot_clean, 1.0, 0.5, 0.5)
        c = M.from_pil(img)

        self._place_band(c, t)

        M.blend_sprite(c, self.scrim, 0, 0, 1.0)
        if t >= cta_a - 0.14:
            self._cta(c, t, cta_a)
        return c

    def _place_band(self, c, t: float):
        """Panel lands once, during the transition, then stays put."""
        g = self._band_geom()
        in_p = T.inv(self.r["band_entry"], self.r["band_entry"] + 0.26, t)
        M.blend_sprite(c, self._render_band(t), g["x0"],
                       g["y0"] + int(46 * (1 - T.ease_out_cubic(in_p))),
                       min(1.0, in_p * 1.3))

    def _render_band(self, t: float) -> Image.Image:
        m = self.r["macro"]
        g = self._band_geom()
        im = self.band.copy()
        d = ImageDraw.Draw(im)
        ox, oy = g["x0"], g["y0"]
        rev_p = T.inv(m["reveal_at"], m["reveal_at"] + 0.35, t)
        reveal = t >= m["reveal_at"]

        f_hdr = T.font("cond_semi", 31)
        if not reveal:
            T.draw_text(d, (g["pad_x"], g["header_y"] - oy), m["header"], f_hdr,
                        T.hex_rgb(self.pal["muted"]) + (255,), anchor="lm", tracking=3.0)
        else:
            col = T.hex_rgb(self.pal["lime"]) + (int(255 * min(1.0, rev_p * 2.5)),)
            T.draw_text(d, (g["pad_x"], g["header_y"] - oy), m["header_reveal"], f_hdr,
                        col, anchor="lm", tracking=3.0)

        f_label = T.font("cond_semi", 32)
        f_num = T.font("display", 46)
        f_unit = T.font("cond_semi", 26)
        for i, row in enumerate(m["rows"]):
            ry = g["row_top"] + i * g["row_h"]
            t0 = m["count_start"] + i * m["count_stagger"]
            q = T.ease_out_cubic(T.inv(t0, t0 + m["count_dur"], t))
            accent = T.hex_rgb(self.pal[row["color"]])
            for side, colbox in (("left", g["lcol"]), ("right", g["rcol"])):
                val = m[side]["values"][row["key"]]
                shown = int(round(val * q))
                cx0, cx1 = colbox[0] - ox, colbox[1] - ox
                T.draw_text(d, (cx0 + 24, ry - oy + 21), row["label"], f_label,
                            T.hex_rgb(self.pal["muted"]) + (255,), anchor="lm", tracking=2.6)
                vcol = T.hex_rgb(self.pal["paper"]) + (255,)
                if reveal and side == "right" and val != m["left"]["values"][row["key"]]:
                    flash = 1 - T.inv(m["reveal_at"] + 0.25, m["reveal_at"] + 1.0, t)
                    vcol = tuple(int(a + (b - a) * flash) for a, b in
                                 zip(vcol[:3], T.hex_rgb(self.pal["green"]))) + (255,)
                unit = row["unit"]
                uw = T.text_len(unit, f_unit, 1.0) if unit else 0
                if unit:
                    T.draw_text(d, (cx1, ry - oy + 27), unit, f_unit,
                                T.hex_rgb(self.pal["muted"]) + (255,), anchor="rm", tracking=1.0)
                T.draw_text(d, (cx1 - uw - 7, ry - oy + 20), str(shown), f_num, vcol, anchor="rm")
                bw = cx1 - cx0 - 24
                by = ry - oy + 54
                d.rounded_rectangle((cx0 + 24, by, cx0 + 24 + bw, by + 12), radius=6,
                                    fill=T.hex_rgb("#23272B") + (255,))
                frac = min(1.0, (val / row["max"]) if row["max"] else 0.0)
                fw = bw * frac * q
                if fw > 3:
                    d.rounded_rectangle((cx0 + 24, by, cx0 + 24 + fw, by + 12), radius=6,
                                        fill=accent + (255,))

        if rev_p > 0:
            i = next(k for k, r in enumerate(m["rows"]) if r["key"] == "protein")
            ry = g["row_top"] + i * g["row_h"] - oy
            col = T.hex_rgb(self.pal["lime"]) + (int(235 * min(1.0, rev_p * 2)),)
            T.icon_check(d, g["div"] - ox - 10, ry + 10, 21, col, 4)
        return im

    def _cta(self, c, t: float, a: float):
        p = T.inv(a - 0.14, a + 0.24, t)
        s = T.ease_out_back(p, 2.2)
        W, H = self.W, self.H
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pw, ph = 872, 108
        cx, cy = W / 2, 1744
        w2, h2 = pw * s, ph * s
        d.rounded_rectangle((cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2),
                            radius=int(ph * s / 2), fill=T.hex_rgb(self.pal["lime"]) + (255,))
        if s > 0.35:
            txt = self.r["cta"]["text"]
            f = T.font("display", int(56 * s))
            icon = 34 * s
            tw = T.text_len(txt, f, 1.5)
            total = tw + icon + 20 * s
            x = cx - total / 2
            T.draw_text(d, (x, cy - 3 * s), txt, f, T.hex_rgb(self.pal["ink"]) + (255,),
                        anchor="lm", tracking=1.5)
            bob = 4 * s * np.sin(t * 8.0)
            T.icon_bookmark(d, x + tw + 20 * s, cy - icon / 2 + bob, icon,
                            T.hex_rgb(self.pal["ink"]) + (255,))
        M.blend_sprite(c, layer, 0, 0, min(1.0, p * 3))

    def _watermark(self, c):
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        T.draw_text(d, (self.W - 40, 46), "@" + self.cfg["brand"]["handle"],
                    T.font("cond_semi", 25), T.hex_rgb(self.pal["paper"]) + (85,),
                    anchor="rm", tracking=2.2)
        M.blend_sprite(c, layer, 0, 0, 1.0)

    # --------------------------------------------------------------- render --
    def render_video(self, out_path: str):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        n = int(round(self.dur * self.fps))
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}",
               "-r", str(self.fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-g", str(self.fps * 2), "-movflags", "+faststart",
               out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(n):
                t = i / self.fps
                fr = self.frame(t)
                proc.stdin.write(np.clip(fr * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes())
                if i % 60 == 0:
                    print(f"  frame {i}/{n}  t={t:5.2f}s", flush=True)
        finally:
            proc.stdin.close()
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg encode failed")
        return out_path
