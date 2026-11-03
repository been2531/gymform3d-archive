#!/usr/bin/env python3
"""t688u mobile sheet: muscle look BEFORE (t687u) / AFTER (gymform.glass_look.day4_muscle) / Day 4, back crops at the same
on-screen muscle scale (Day 4 resized by our torso / Day 4 torso), plus the full frames and the metrics table.
usage: python3 tools/glass_muscle_sheet.py   (after tools/glass_2shot_stills.py)"""
import sys, json, cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from build_v28_luxury_ab import f_pretendard
R = "/workspace/daily_info_reels/"; O = R + "output/_consistency/glass2shot/"; B = O + "before_t688/"
q = json.load(open(O + "g2s_stills_qc.json")); qb = json.load(open(B + "g2s_stills_qc.json"))
mt, mb = q["metrics_vs_day4"], qb["metrics_vs_day4"]
tor = q["shots"]["tight_peak"]["torso_px_1080"]
aft = cv2.imread(O + "g2s_tight_peak_graded_1080.png"); bef = cv2.imread(B + "g2s_tight_peak_graded_1080.png")
d4 = cv2.imread(R + "output/_consistency/d7m/day4_master_t17.5_right.jpg")
s = tor / 356.4; d4s = cv2.resize(d4, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
cr = (560, 820, 1020, 1340); w_, h_ = cr[2] - cr[0], cr[3] - cr[1]
c4 = d4s[int(1000 * s) - h_ // 2:int(1000 * s) + h_ // 2, int(560 * s) - w_ // 2:int(560 * s) + w_ // 2]
F = lambda z, w="extrabold": f_pretendard(z, w); G = (120, 230, 160); WH = (240, 240, 240); GR = (170, 175, 185)
def tile(im, w, h):
    k = min(w / im.shape[1], h / im.shape[0]); r = cv2.resize(im, (int(im.shape[1] * k), int(im.shape[0] * k)), interpolation=cv2.INTER_AREA)
    t = np.full((h, w, 3), 16, np.uint8); y0 = (h - r.shape[0]) // 2; x0 = (w - r.shape[1]) // 2; t[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
W = 1080; sh = Image.new("RGB", (W, 4200), (14, 15, 18)); d = ImageDraw.Draw(sh); y = 36
d.text((40, y), "근육 표현 개선 (코드만)", font=F(62), fill=WH); y += 84
d.text((40, y), "생성 0 · 같은 근육 크기로 비교", font=F(30, "medium"), fill=GR); y += 70
tw, th = 326, 370
for i, (im, t) in enumerate([(bef[cr[1]:cr[3], cr[0]:cr[2]], "전"), (aft[cr[1]:cr[3], cr[0]:cr[2]], "후"), (c4, "Day 4")]):
    x = 40 + i * (tw + 11); sh.paste(tile(im, tw, th), (x, y))
    if i == 1: d.rounded_rectangle((x - 4, y - 4, x + tw + 4, y + th + 4), 14, outline=G, width=6)
    d.text((x + 6, y + th + 10), t, font=F(42), fill=G if i == 1 else WH)
y += th + 80
fw, fh = 494, 878
for i, (nm, t) in enumerate([("wide_peak", "와이드 (후)"), ("tight_peak", "타이트 (후)")]):
    x = 40 + i * (fw + 12); sh.paste(tile(cv2.imread(O + f"g2s_{nm}_graded_1080.png"), fw, fh), (x, y)); d.text((x, y + fh + 10), t, font=F(40), fill=G)
y += fh + 80
d.text((40, y), "Day 4 대비 (타이트)", font=F(46), fill=G); y += 66
d.text((330, y), "전", font=F(32), fill=GR); d.text((560, y), "후", font=F(32), fill=G); d.text((790, y), "Day4", font=F(32), fill=GR); y += 50
rows = [("색상", "core_hue_deg", 0), ("채도", "core_sat", 0), ("밝기", "core_val", 0), ("섬유 대비", "fibre_contrast", 1),
        ("섬유 밀도", "fibre_density_per100px", 1), ("가장자리 px", "falloff_px", 1), ("면적", "cover_glow_per_torso2", 3)]
for lab, k, r in rows:
    vb, va, v4 = mb["tight_peak"][k], mt["tight_peak"][k], mt["day4"][k]
    ok = abs(va - v4) / max(1e-6, abs(v4)) <= 0.10
    d.text((40, y), lab, font=F(34), fill=WH); d.text((330, y), f"{vb:.{r}f}", font=F(34, "bold"), fill=GR)
    d.text((560, y), f"{va:.{r}f}", font=F(34), fill=G if ok else (255, 150, 120)); d.text((790, y), f"{v4:.{r}f}", font=F(34, "bold"), fill=WH); y += 52
y += 20
for t in ["± 10% 이내 = 초록", "남은 과제: V 손잡이 · 피크 자세 (다음 단계)"]:
    d.text((40, y), t, font=F(32, "bold"), fill=WH); y += 52
sh = sh.crop((0, 0, W, y + 20)); sh.save(O + "G2S_MUSCLE_sheet.jpg", quality=92); print(O + "G2S_MUSCLE_sheet.jpg")
