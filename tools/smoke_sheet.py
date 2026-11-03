#!/usr/bin/env python3
"""Day 7 smoke sheet: new pinned clips (raw first/mid/last + graded) vs v3, gate metrics labelled."""
import json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
R = "/workspace/daily_info_reels"; D = f"{R}/output/_consistency/smoke_d7"
G1 = json.load(open(f"{D}/gate_t1.json")); G2 = json.load(open(f"{D}/gate_t2.json")); GV = json.load(open(f"{D}/gate_v3_baseline.json"))
F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)
FB = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
TW, TH = 150, 267
def raw_frames(path, idx):
    cap = cv2.VideoCapture(path); out = []
    for i in idx:
        cap.set(1, i); ok, f = cap.read(); out.append(cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA))
    return out
def graded(tag, form, idx):
    out = []
    for i in idx:
        g = cv2.imread(f"{D}/graded/graded_{tag}_{form}_f{i:03d}.jpg")
        g = g[300:1900, 90:990]   # crop the canvas around the figure (same crop for all)
        out.append(cv2.resize(g, (TW, TH), interpolation=cv2.INTER_AREA))
    return out
rows = [("NEW wrong (GenerateVideo 720p, first=last=start, key 2.5 s)", f"{D}/d7_smoke_wrong_720p_t2.mp4", [0, 60, 120], "t2", "wrong", G2["wrong"]),
        ("NEW right (GenerateVideo 720p, first=last=start, key 2.5 s)", f"{D}/d7_smoke_right_720p_t2.mp4", [0, 60, 120], "t2", "right", G2["right"]),
        ("v3 wrong (GenerateVideo, first frame only, 6 s)", f"{R}/output/assets/day7_wrong_raw.mp4", [0, 72, 144], "v3", "wrong", GV["v3wrong"]),
        ("v3 right (GenerateVideo, first frame only, 6 s)", f"{R}/output/assets/day7_right_raw.mp4", [0, 72, 144], "v3", "right", GV["v3right"])]
LW = 470; Wd = max(LW + 6 * (TW + 6) + 30, 1560); RH = TH + 40
Ht = 120 + len(rows) * RH + 260
img = Image.new("RGB", (Wd, Ht), (18, 20, 23)); d = ImageDraw.Draw(img)
d.text((20, 16), "Day 7 smoke (t633s2233): pinned GenerateVideo vs v3 — raw first / mid / last | Day 4 grade (native, unchanged) on the same 3 frames", font=FB(19), fill=(235, 235, 235))
d.text((20, 50), "Gate G1–G8 at 360x640 (docs/mission/2D_CONSISTENCY_PLAN.md §3d). Green = pass, red = fail. Raw cells left, graded cells right.", font=F(15), fill=(170, 175, 180))
x0 = LW
for j, t in enumerate(["raw f0", "raw mid", "raw last", "graded f0", "graded mid", "graded last"]):
    d.text((x0 + j * (TW + 6) + 4, 86), t, font=F(14), fill=(200, 200, 200))
y = 110
def ok(b): return (120, 220, 140) if b else (255, 120, 110)
for title, clip, idx, tag, form, g in rows:
    cells = raw_frames(clip, idx) + graded(tag, form, idx)
    for j, c in enumerate(cells):
        img.paste(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)), (x0 + j * (TW + 6), y))
    d.text((20, y), title, font=FB(14), fill=(235, 235, 235))
    lines = [
        (f"G1 start hold  SSIM {g['G1']['ssim']}  edge {g['G1']['edge_iou']}", g["G1"]["pass"]),
        (f"G2 end {g['G2']['end_ssim']}/{g['G2']['end_edge_iou']}  key±6 {g['G2']['key_best_pm6'][0]}/{g['G2']['key_best_pm6'][1]}", g["G2"]["pass"]),
        (f"G3 glow {g['G3']['start_glow_pct']}% start → {g['G3']['clip_min_pct']}–{g['G3']['clip_max_pct']}% ({g['G3']['ratio_min']}–{g['G3']['ratio_max']}×) cv {g['G3']['cv_pct']}%", g["G3"]["pass"]),
        (f"G4 off-body {g['G4']['max_off_body_pct']}%   G5 bg SSIM {g['G5']['bg_ssim_min']}", g["G4"]["pass"] and g["G5"]["pass"]),
        (f"G6 bg drift {g['G6']['bg_lum_drift']}  f0 bg {g['G6']['bg_rgb_f0']}", g["G6"]["pass"]),
        (f"G7 view {g['G7']['view_mean']} (start {g['G7']['view_start']}) spread {g['G7']['view_spread_between_clips']}", g["G7"]["pass"]),
        (f"G8 cobalt cv {g['G8']['cobalt_cv_pct']}%  ΔRGB {g['G8']['cobalt_rgb_dist_vs_start']}  between {g['G8']['between_rel_pct']}%", g["G8"]["pass"]),
        (f"skin {min(g['skin_series']):.0f}–{max(g['skin_series']):.0f} (start {g['skin_start']})  elbow max {g['motion']['elbow_flex_max']}°", True),
    ]
    for i, (t, b) in enumerate(lines):
        d.text((20, y + 22 + i * 28), t, font=F(13), fill=ok(b) if i < 7 else (190, 195, 200))
    y += RH
d.line((20, y, Wd - 20, y), fill=(70, 70, 70), width=2); y += 10
t1w, t1r = G1["wrong"], G1["right"]
notes = [
    f"Try 1 (480p, same pins): G1 {t1w['G1']['ssim']}/{t1w['G1']['edge_iou']} & {t1r['G1']['ssim']}/{t1r['G1']['edge_iou']} FAIL — a perfect copy only reaches edge 0.885–0.894 at 480p, so 480p can't pass G1/G2 → retry at 720p.",
    "720p retry: G1 passes on both (0.993/0.897, 0.992/0.894; v3 0.955/0.83, 0.959/0.80). End hold 0.978/0.879 and 0.987/0.875 vs v3 0.87/0.60, 0.88/0.64 — misses the edge bar by 0.001–0.005.",
    "Right G3: clip glow follows the pinned keyframe (5.2%, lats partly hidden by the arms at the contraction), so 0.50× the start frame; vs the keyframe it is 0.84×. A glow-matched keyframe edit came out at 5.1%.",
    "G9 (motion_check): the standard's only measurable wrong signature is lean-back (not this mistake), so it doesn't fire; right items pass except elbows_past_sides (warn): 2D elbow max 40° (t1 50°, v3 61°).",
    "Wrong t2 does ~2 reps despite the 'one rep' prompt; the right rep is shallower than v3. Tone, backdrop, camera and equipment now hold (G4–G8 pass on both).",
]
for i, t in enumerate(notes):
    d.text((20, y + i * 26), t, font=F(14), fill=(220, 220, 220))
img = img.crop((0, 0, Wd, y + len(notes) * 26 + 20))
img.save(f"{D}/smoke_d7_sheet.jpg", quality=88); print(f"{D}/smoke_d7_sheet.jpg", img.size)
