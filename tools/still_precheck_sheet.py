#!/usr/bin/env python3
"""Mobile sheet (Korean) for still pre-check runs: one column per attempt, full still + hand/cable zoom,
PASS/FAIL per criterion from tools/still_precheck.py JSONs, plus optional by-eye overrides.
usage: python3 tools/still_precheck_sheet.py SPEC.json   (SPEC: {"title","verdict","out","attempts":[{"label","img","json","kind","eye":{crit:[PASS|FAIL, note]}}],"notes":[...]})"""
import sys, json, cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from build_v28_luxury_ab import f_pretendard
R = "/workspace/daily_info_reels/"
sp = json.load(open(sys.argv[1])); F = lambda z, w="extrabold": f_pretendard(z, w)
G, WH, GR, RD, YL = (120, 230, 160), (240, 240, 240), (175, 180, 190), (255, 130, 110), (250, 210, 110)
W = 1080; sh = Image.new("RGB", (W, 7000), (14, 15, 18)); d = ImageDraw.Draw(sh); y = 36
def paste(im, x, y, w, h):
    sh.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)), (x, y))
d.text((40, y), sp["title"], font=F(54), fill=WH); y += 80
for s in sp.get("sub", []): d.text((40, y), s, font=F(28, "bold"), fill=GR); y += 44
y += 10; ok = sp["verdict"].startswith("PASS")
d.rounded_rectangle((40, y, W - 40, y + 110), 22, fill=(20, 36, 28) if ok else (36, 24, 24))
d.text((70, y + 26), sp["verdict"], font=F(42), fill=G if ok else RD); y += 140
A = sp["attempts"]; n = len(A); cw = (W - 80 - (n - 1) * 12) // n
for k, a in enumerate(A):
    im = cv2.imread(R + a["img"]); x = 40 + k * (cw + 12)
    d.text((x, y), a["label"], font=F(32), fill=WH)
    paste(im[380:1280, 0:720], x, y + 50, cw, int(cw * 900 / 720))
    paste(im[680:1020, 150:700], x, y + 60 + int(cw * 900 / 720), cw, int(cw * 340 / 550))
y += 70 + int(cw * 900 / 720) + int(cw * 340 / 550) + 20
names = [("machine", "기구 = 원판"), ("torso", "몸통 수직 ≤10°"), ("arm", "팔 쭉 (시작)"), ("cable", "케이블 한 줄"), ("handle", "V 손잡이"),
         ("head", "머리=몸 톤"), ("light", "그늘면 ≥ 배경"), ("glow", "발광 없음")]
for key, lab in names:
    d.text((40, y), lab, font=F(28, "bold"), fill=WH); yy = y + 40
    for k, a in enumerate(A):
        j = json.load(open(R + a["json"]))[a.get("kind", "start")]; x = 40 + k * (cw + 12)
        v = "PASS" if j[key]["pass"] else "FAIL"; note = ""
        if key == "machine": note = f"SSIM {min(x_ for x_ in j['machine']['ssim'].values() if x_ is not None)}"
        if key == "torso": note = f"{j['pose']['torso_lean_deg']}°"
        if key == "arm": note = f"굽힘 {j['pose']['elbow_flex_deg']}°"
        if key == "cable": note = f"{int(j['cable']['cover'] * 100)}%"
        if key == "handle": note = f"위쪽 금속 {j['handle']['rect_top_metal_px']}px"
        if key == "head": note = f"ΔL {j['head']['dL']}"
        if key == "light": note = f"{j['light']['diff']:+.1f}"
        if key == "glow": note = f"{j['glow']['frac'] * 100:.1f}%"
        eye = a.get("eye", {}).get(key)
        if eye: v, note = eye[0], eye[1]
        d.text((x, yy), v, font=F(30), fill=G if v == "PASS" else RD); d.text((x + 100, yy + 4), note, font=F(22, "bold"), fill=GR)
    y = yy + 52
y += 10
d.rounded_rectangle((40, y, W - 40, y + 60 + 50 * len(sp["notes"])), 24, fill=(24, 26, 30), outline=YL, width=4)
d.text((70, y + 18), "다음 단계", font=F(36), fill=YL)
for i, t in enumerate(sp["notes"]): d.text((70, y + 70 + i * 50), t, font=F(27, "bold"), fill=WH)
y += 90 + 50 * len(sp["notes"])
sh = sh.crop((0, 0, W, y)); sh.save(R + sp["out"], quality=90); print(sp["out"], sh.size)
