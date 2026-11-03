#!/usr/bin/env python3
"""Mobile sheet (Korean) for the empty-plate handle test: plate vs each on-plate handle result (full + zoom),
detector PASS/FAIL per check from `still_precheck.py --plate-handle` JSONs, by-eye overrides/notes, fixture calibration.
usage: python3 tools/plate_handle_sheet.py SPEC.json
SPEC: {"title","sub":[..],"verdict","out","plate","attempts":[{"label","img","json","eye":{check:[PASS|FAIL|WARN, note]}}],
       "fixtures":[{"label","img","json"}],"notes":[..]}"""
import sys, json, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from build_v28_luxury_ab import f_pretendard
R = "/workspace/daily_info_reels/"
sp = json.load(open(sys.argv[1])); F = lambda z, w="extrabold": f_pretendard(z, w)
G, WH, GR, RD, YL = (120, 230, 160), (240, 240, 240), (175, 180, 190), (255, 130, 110), (250, 210, 110)
COL = {"PASS": G, "FAIL": RD, "WARN": YL}
W = 1080; sh = Image.new("RGB", (W, 6000), (14, 15, 18)); d = ImageDraw.Draw(sh); y = 36
def paste(im, x, y, w, h):
    sh.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)), (x, y))
d.text((40, y), sp["title"], font=F(54), fill=WH); y += 80
for s in sp.get("sub", []): d.text((40, y), s, font=F(28, "bold"), fill=GR); y += 44
y += 10; ok = sp["verdict"].startswith("PASS")
d.rounded_rectangle((40, y, W - 40, y + 110), 22, fill=(20, 36, 28) if ok else (36, 24, 24))
d.text((70, y + 26), sp["verdict"], font=F(40), fill=G if ok else RD); y += 140
cols = [{"label": "빈 기구 (원판)", "img": sp["plate"]}] + sp["attempts"]; n = len(cols); cw = (W - 80 - (n - 1) * 12) // n
fh, zh = int(cw * 900 / 720), int(cw * 220 / 570)
for k, a in enumerate(cols):
    im = cv2.imread(R + a["img"]); x = 40 + k * (cw + 12)
    d.text((x, y), a["label"], font=F(30), fill=WH)
    paste(im[380:1280, 0:720], x, y + 48, cw, fh)
    paste(im[840:1060, 150:720], x, y + 58 + fh, cw, zh)
y += 80 + fh + zh
d.text((40, y), "검사 (자동 / 눈)", font=F(34), fill=WH); y += 54
names = [("machine", "기구 = 원판 (SSIM≥0.99)"), ("cable", "케이블 있음·팽팽"), ("attached", "케이블에 연결"),
         ("unbroken", "끊김 없음"), ("shape", "V/삼각 (사각 아님)")]
A = sp["attempts"]; cw2 = (W - 80 - (len(A) - 1) * 12) // len(A)
for key, lab in names:
    d.text((40, y), lab, font=F(28, "bold"), fill=WH); yy = y + 40; hmax = 0
    for k, a in enumerate(A):
        j = json.load(open(R + a["json"])); x = 40 + k * (cw2 + 12)
        v = "PASS" if j.get(key, {}).get("pass") else "FAIL"
        if key == "machine": note = f"최저 {min(j['machine']['ssim'].values()):.3f}"
        elif key == "cable": note = f"{int(j['cable']['cover'] * 100)}% · {int(j['cable']['len_px'])}px"
        elif key == "attached": note = f"틈 {j['attached']['max_gap_on_cable_px']}px"
        elif key == "unbroken": note = f"조각 {j['unbroken']['blobs']}"
        else: note = f"끝/앞 폭비 {j['shape']['taper_near_over_far']}"
        d.text((x, yy), f"자동 {v}", font=F(28), fill=COL[v]); d.text((x + 150, yy + 2), note, font=F(24, "bold"), fill=GR)
        e = a.get("eye", {}).get(key)
        if e: d.text((x, yy + 38), f"눈 {e[0]}", font=F(28), fill=COL[e[0]]); d.text((x + 150, yy + 40), e[1], font=F(24, "bold"), fill=GR); hmax = max(hmax, 40)
    y = yy + 46 + hmax
for key, lab in sp.get("eye_only", []):
    d.text((40, y), lab, font=F(28, "bold"), fill=WH); yy = y + 40
    for k, a in enumerate(A):
        e = a["eye"][key]; x = 40 + k * (cw2 + 12)
        d.text((x, yy), f"눈 {e[0]}", font=F(28), fill=COL[e[0]]); d.text((x + 150, yy + 2), e[1], font=F(24, "bold"), fill=GR)
    y = yy + 50
if sp.get("fixtures"):
    y += 10; d.text((40, y), "검사기 보정 (가짜 그림)", font=F(30), fill=WH); y += 50
    fw = 240
    for k, fx in enumerate(sp["fixtures"]):
        im = cv2.imread(R + fx["img"]); j = json.load(open(R + fx["json"])); x = 40 + k * (W - 80) // 2
        paste(im[860:1040, 200:640], x, y, fw, int(fw * 180 / 440))
        v = "PASS" if j["shape"]["pass"] else "FAIL"
        d.text((x + fw + 14, y), fx["label"], font=F(26, "bold"), fill=WH)
        d.text((x + fw + 14, y + 40), f"모양 {v}", font=F(28), fill=COL[v])
    y += int(fw * 180 / 440) + 30
for s in sp.get("notes", []): d.text((40, y), s, font=F(27, "bold"), fill=GR); y += 42
sh.crop((0, 0, W, y + 30)).save(R + sp["out"], quality=90); print(R + sp["out"], W, y + 30)
