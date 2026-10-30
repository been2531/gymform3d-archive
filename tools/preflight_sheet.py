#!/usr/bin/env python3
"""Mobile-readable preflight sheet (t567u): representative QC frames of every step (start / peak / t100, callouts
and preview included, straight from the preflight sidecars) + the QC result list from the run log.

    python3 tools/preflight_sheet.py --config configs/day06_muscle2.json --log output/logs/<run>.log --out output/x.jpg
"""
import argparse, json, os, re, textwrap
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = "/usr/share/fonts/truetype/sand-box/custom/Pretendard/"


def font(sz, w="Bold"):
    p = os.path.join(FD, f"Pretendard-{w}.otf")
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default="", help="text file: one remaining-issue line per row")
    ap.add_argument("--tags", default="start,peak,t100")
    ap.add_argument("--compare", default="", help="gold mp4 for a side-by-side muscle-colour crop, e.g. "
                    "output/day4_romanian_deadlift_4step_master.mp4:200,500 (wrong frame, right frame)")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, a.config)))
    n = int(cfg.get("n_steps", 4))
    pdir = os.path.join(ROOT, cfg["qc_dir"].rstrip("/") + "_preflight")
    tags = a.tags.split(",")
    Wd, pad = 1080, 18
    tw = (Wd - pad * (len(tags) + 1)) // len(tags)
    th = int(tw * 1920 / 1080)
    rows = []
    for line in open(os.path.join(ROOT, a.log)):
        m = re.match(r"\[(PASS|FAIL)\] (.*)", line.strip())
        if m:
            rows.append((m.group(1), m.group(2)))
    notes = [l.rstrip() for l in open(a.notes)] if a.notes and os.path.exists(a.notes) else []
    f_h, f_s, f_q, f_t = font(38, "Bold"), font(30, "Bold"), font(25, "SemiBold"), font(24, "Medium")
    wrap = lambda t, w: textwrap.wrap(t, w) or [""]
    q_lines = []
    for st, t in rows:
        q_lines.append((st, wrap(t[:230], 70)))
    hq = sum(34 * len(l) + 8 for _, l in q_lines)
    hn = sum(32 * len(wrap(t, 72)) + 6 for t in notes)
    cmp_rows = []
    if a.compare:
        import cv2, numpy as np
        mp4, frs = a.compare.split(":")
        cap = cv2.VideoCapture(os.path.join(ROOT, mp4))

        def crop(bgr, sz=440):
            f = bgr.astype(np.float32)
            act = np.maximum(f[..., 2] - 0.5 * (f[..., 1] + f[..., 0]), f[..., 1] - 0.5 * (f[..., 2] + f[..., 0]))
            act[:600] = 0; act[:, :300] = 0
            ys, xs = np.where(act > 40)
            cy, cx = (int(np.median(ys)), int(np.median(xs))) if ys.size else (960, 540)
            y0, x0 = max(0, min(1920 - sz, cy - sz // 2)), max(0, min(1080 - sz, cx - sz // 2))
            return Image.fromarray(cv2.cvtColor(bgr[y0:y0 + sz, x0:x0 + sz], cv2.COLOR_BGR2RGB))
        gold_day = re.search(r"day(\d+)", os.path.basename(mp4)).group(1)
        for s_, fi in zip((1, 2), [int(x) for x in frs.split(",")]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, g = cap.read()
            d6 = cv2.imread(os.path.join(pdir, f"step{s_}_peak_final.png"))
            if ok and d6 is not None:
                cmp_rows.append((("wrong / red" if s_ == 1 else "right / green"), f"Day {gold_day} (gold)", crop(g), f"Day {cfg['day']} now", crop(d6)))
    cw = (Wd - 3 * pad) // 2
    H = 150 + n * (th + 70) + (80 + len(cmp_rows) * (cw + 60) if cmp_rows else 0) + 90 + hq + (90 + hn if notes else 0) + 40
    img = Image.new("RGB", (Wd, H), (14, 16, 20))
    d = ImageDraw.Draw(img)
    npass = sum(1 for s, _ in rows if s == "PASS")
    d.text((pad, 20), f"Day {cfg['day']} {cfg['exercise_name']} — {n}-step preflight", font=f_h, fill=(242, 246, 252))
    d.text((pad, 78), f"QC {npass}/{len(rows)} PASS · sample frames only (no full render) · muscle fibre only, no bones",
           font=f_t, fill=(185, 225, 255))
    y = 150
    labels = cfg.get("sheet_labels", [])
    for s in range(1, n + 1):
        wrong = (s == 1) if n == 2 else s in (1, 3)
        col = (245, 80, 60) if wrong else (40, 230, 130)
        d.text((pad, y), labels[s - 1] if s - 1 < len(labels) else f"Step {s}", font=f_s, fill=col)
        y += 50
        for i, tg in enumerate(tags):
            p = os.path.join(pdir, f"step{s}_{tg}_final.png")
            x = pad + i * (tw + pad)
            if os.path.exists(p):
                img.paste(Image.open(p).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS), (x, y))
            d.rectangle([x - 2, y - 2, x + tw + 1, y + th + 1], outline=col, width=3)
            d.text((x + 8, y + th - 34), tg, font=f_q, fill=(235, 240, 248))
        y += th + 20
    if cmp_rows:
        d.text((pad, y + 10), "Muscle colour: gold vs now (same crop size, 1:1 pixels)", font=f_s, fill=(255, 200, 90))
        y += 70
        for lab, l1, im1, l2, im2 in cmp_rows:
            d.text((pad, y), lab, font=f_q, fill=(215, 222, 232))
            y += 40
            for k, (l_, im_) in enumerate(((l1, im1), (l2, im2))):
                x = pad + k * (cw + pad)
                img.paste(im_.resize((cw, cw), Image.Resampling.LANCZOS), (x, y))
                d.rectangle([x, y, x + 210, y + 40], fill=(14, 16, 20))
                d.text((x + 8, y + 6), l_, font=f_q, fill=(242, 246, 252))
            y += cw + 20
    d.text((pad, y + 10), "QC (preflight sample frames)", font=f_s, fill=(242, 246, 252))
    y += 60
    for st, ls in q_lines:
        c = (60, 220, 120) if st == "PASS" else (255, 90, 70)
        d.text((pad, y), st, font=f_q, fill=c)
        for j, l in enumerate(ls):
            d.text((pad + 90, y + 34 * j), l, font=f_t, fill=(215, 222, 232))
        y += 34 * len(ls) + 8
    if notes:
        d.text((pad, y + 20), "Remaining honest issues vs Day 4", font=f_s, fill=(255, 200, 90))
        y += 80
        for t in notes:
            for j, l in enumerate(wrap(t, 72)):
                d.text((pad, y + 32 * j), l, font=f_t, fill=(225, 230, 238))
            y += 32 * len(wrap(t, 72)) + 6
    img.save(os.path.join(ROOT, a.out), quality=88)
    print(a.out, img.size)


if __name__ == "__main__":
    main()
