"""Text blocks with measured cap height, logged per frame for QA."""
import skia

FONTS = {
    "anton": "/usr/share/fonts/truetype/sand-box/google/Anton/Anton-Regular.ttf",
    "kalam": "/usr/share/fonts/truetype/sand-box/google/Kalam/Kalam-Bold.ttf",
    "marker": "/usr/share/fonts/truetype/sand-box/google/Permanent Marker/PermanentMarker-Regular.ttf",
}
_tf = {}


def typeface(name):
    if name not in _tf:
        _tf[name] = skia.Typeface.MakeFromFile(FONTS[name])
    return _tf[name]


def cap_ratio(name):
    f = skia.Font(typeface(name), 100)
    return -f.getMetrics().fCapHeight / 100 if f.getMetrics().fCapHeight < 0 else f.getMetrics().fCapHeight / 100


def _rel_lum(c):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(c1, c2):
    a, b = _rel_lum(c1), _rel_lum(c2)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def wrap(font, text, max_w):
    words = text.split(); lines = []; cur = ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.measureText(t) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines


def block(cv, log, role, text, font_name, cap_px, cx, top, color, bg, max_w=980, scale=1.0, alpha=1.0,
          outline=None, align="center", line_gap=0.35, max_lines=None, fallback_font=None):
    """Draw centred text whose CAP HEIGHT is cap_px (before pop scale). Returns bottom y.
    max_lines: if the wrap needs more lines, retry with fallback_font (narrower) at the same cap height."""
    size = cap_px / cap_ratio(font_name)
    font = skia.Font(typeface(font_name), size)
    font.setEdging(skia.Font.Edging.kAntiAlias)
    lines = wrap(font, text, max_w)
    if max_lines and len(lines) > max_lines and fallback_font:
        f2 = skia.Font(typeface(fallback_font), cap_px / cap_ratio(fallback_font)); f2.setEdging(skia.Font.Edging.kAntiAlias)
        l2 = wrap(f2, text, max_w)
        if len(l2) < len(lines):
            font, lines, font_name = f2, l2, fallback_font
    widest = max(font.measureText(ln) for ln in lines)
    paint = skia.Paint(AntiAlias=True, Color=color); paint.setAlphaf(alpha)
    y = top
    cv.save()
    if scale != 1.0:
        cv.translate(cx, top + cap_px / 2); cv.scale(scale, scale); cv.translate(-cx, -(top + cap_px / 2))
    for ln in lines:
        w = font.measureText(ln)
        x = cx - w / 2 if align == "center" else cx
        base = y + cap_px
        if outline:
            op = skia.Paint(AntiAlias=True, Color=outline[0], Style=skia.Paint.kStroke_Style, StrokeWidth=outline[1],
                            StrokeJoin=skia.Paint.kRound_Join)
            op.setAlphaf(alpha)
            cv.drawString(ln, x, base, font, op)
        cv.drawString(ln, x, base, font, paint)
        y = base + cap_px * line_gap + cap_px * 0.15
    cv.restore()
    bottom = y - cap_px * line_gap
    if log is not None and alpha > 0.5 and scale >= 0.98:
        log.append({"role": role, "text": text, "cap": cap_px, "lines": len(lines),
                    "left": cx - widest / 2 if align == "center" else cx, "right": cx + widest / 2 if align == "center" else cx + widest,
                    "top": top, "bottom": bottom, "contrast": contrast(color, outline[0] if outline else bg)})
    return bottom
