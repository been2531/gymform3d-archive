#!/usr/bin/env python3
"""Contact sheet: Day 4 vs Day 5/6/7 source clips with measured drift values (reads output/_consistency/*.json)."""
import json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = "/workspace/daily_info_reels"; OUT = f"{ROOT}/output/_consistency"
M = json.load(open(f"{OUT}/metrics.json")); C, B = M["clips"], M["between"]
SF = json.load(open(f"{OUT}/startframe_vs_clip.json"))
F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)
FB = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
TW, TH = 150, 267; CW = 2 * TW + 14; LW = 250; ROWH = TH + 110
def fname(day, n): return f"{OUT}/f_{day}_{n.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '-')}.jpg"
def thumbs(day, n):
    im = cv2.imread(fname(day, n)); w = im.shape[1] // 3
    a = cv2.resize(im[:, :w], (TW, TH)); b = cv2.resize(im[:, 2 * w:], (TW, TH))
    return Image.fromarray(cv2.cvtColor(np.hstack([a, np.full((TH, 4, 3), 30, np.uint8), b]), cv2.COLOR_BGR2RGB))
rows = [("D4", "Day 4 RDL (GOLD)", ["s1 locked", "s2 ecorche", "s3=s1 ecorche", "s4 locked"]),
        ("D5", "Day 5 Lat Raise", ["s1 bone", "s2 muscle", "s3 bone v3", "s4 muscle"]),
        ("D6", "Day 6 IDP v5", ["v5 wrong", "v5 right"]),
        ("D7", "Day 7 Row v3", ["v3 wrong", "v3 right"])]
Wd = LW + 4 * CW + 20; Ht = 110 + len(rows) * ROWH + 330
img = Image.new("RGB", (Wd, Ht), (18, 20, 23)); d = ImageDraw.Draw(img)
d.text((20, 18), "2D source consistency — raw clips (first | last frame), measured at 360x640, 9 frames/clip", font=FB(26), fill=(235, 235, 235))
d.text((20, 58), "skin = torso L p50 (glow-free) mean [min-max] · view = shoulder width / torso length · glow cv = within-clip variation of glow area · bgΔ = RGB distance to 35/39/43",
       font=F(15), fill=(170, 175, 180))
y = 100
for day, title, names in rows:
    b = B[day]; ok = day == "D4"
    d.text((20, y + 6), title, font=FB(19), fill=(120, 220, 140) if ok else (240, 170, 90))
    lines = [f"f0 SSIM between clips", f"  min {b['f0_ssim_min']}",
             f"skin spread {b['skin_L_p50_spread']:.0f} L", f"view spread {b['view_ratio_spread']:.3f}",
             f"body-h spread {b['body_h_frac_spread']:.3f}", f"bgΔ spread {b['bg_dist_spread']}"]
    for i, t in enumerate(lines): d.text((20, y + 40 + i * 24), t, font=F(16), fill=(210, 210, 210))
    for j, n in enumerate(names):
        x = LW + j * CW; m = C[day][n]
        img.paste(thumbs(day, n), (x, y))
        g = m["glow_red_frac"]["mean"] + m["glow_green_frac"]["mean"]
        gcv = max(m["glow_red_frac"]["cv_pct"] or 0, m["glow_green_frac"]["cv_pct"] or 0)
        s = m["skin_L_p50"]
        t = [f"{n}  ({m['dur_s']} s)",
             f"skin {s['mean']:.0f} [{s['min']:.0f}-{s['max']:.0f}]  view {m['view_ratio']['mean']:.2f}",
             f"glow {g*100:.1f}% body, cv {gcv:.0f}%",
             f"bgΔ {m['bg_dist_to_35_39_43']}  SSIM f0→ {m['ssim_vs_f0']['min']:.2f}"]
        for i, tt in enumerate(t):
            col = (235, 235, 235) if i == 0 else (195, 200, 205)
            if i == 2 and gcv > 30: col = (255, 120, 110)
            d.text((x, y + TH + 6 + i * 22), tt, font=F(15), fill=col)
    y += ROWH
# start frame -> clip drift (GenerateVideo v3 vs grok.com v2)
d.line((20, y, Wd - 20, y), fill=(70, 70, 70), width=2); y += 12
d.text((20, y), "Start frame held by the generator?  start → clip f0 → clip last", font=FB(20), fill=(235, 235, 235)); y += 34
for i, k in enumerate(["D6 v5 right", "D7 v2 right", "D7 v3 wrong", "D7 v3 right"]):
    s = SF[k]; gs = s["glow_series"]
    t = (f"{k}: SSIM start→f0 {s['ssim_start_f0']}  start→last {s['ssim_start_last']} | "
         f"skin {s['skin_start']}→{s['skin_f0']}→{s['skin_last']} | bg {s['bg_start']}→{s['bg_f0']} | "
         f"glow {s['glow_start']*100:.1f}% → min {min(gs)*100:.1f}% / max {max(gs)*100:.1f}%")
    bad = s["ssim_start_f0"] < 0.97 or max(gs) > 2 * max(s["glow_start"], 0.01) or min(gs) < 0.6 * s["glow_start"]
    d.text((20, y + i * 28), t, font=F(16), fill=(255, 130, 110) if bad else (150, 225, 160))
y += 4 * 28 + 14
d.text((20, y), "Day 4: one base generation; steps are motion-preserving edits of it (s3 re-uses s1) → same body, barbell, rear-3/4 camera (view 0.52-0.56).",
       font=F(16), fill=(150, 225, 160)); y += 26
d.text((20, y), "Day 5: 4 independent generations → 2 different mannequins (skin 82-138), front vs back view, floor mat in s2, s4 glow spreads to arm (cv 78%).",
       font=F(16), fill=(255, 150, 120)); y += 26
d.text((20, y), "Day 7 v3 (GenerateVideo, start frame only): frame 0 re-rendered (SSIM 0.94 vs 0.99), backdrop to 27/31/39, skin darkens 30%, right glow halves.",
       font=F(16), fill=(255, 150, 120))
img = img.crop((0, 0, Wd, y + 40))
img.save(f"{OUT}/consistency_sheet_d4_vs_d5_d6_d7.jpg", quality=88)
print(f"{OUT}/consistency_sheet_d4_vs_d5_d6_d7.jpg", img.size)
