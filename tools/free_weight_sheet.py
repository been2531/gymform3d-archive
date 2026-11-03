#!/usr/bin/env python3
"""Mobile sheet (Korean) for free-weight start stills: Day 4 reference vs master vs wrong/right stills,
fibre zoom vs Day 4, auto PASS/FAIL per check (tools/free_weight_precheck.py JSONs) + by-eye notes.
usage: python3 tools/free_weight_sheet.py SPEC.json
SPEC: {"title","sub":[..],"verdict","out","ref","master","stills":[{"label","img","json","eye":{check:[PASS|FAIL|WARN,note]}}],"zoom":[y0,y1,x0,x1],"notes":[..]}"""
import sys, json, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from build_v28_luxury_ab import f_pretendard
R = "/workspace/daily_info_reels/"
sp = json.load(open(sys.argv[1])); F = lambda z, w="extrabold": f_pretendard(z, w)
G, WH, GR, RD, YL = (120, 230, 160), (240, 240, 240), (175, 180, 190), (255, 130, 110), (250, 210, 110)
COL = {"PASS": G, "FAIL": RD, "WARN": YL}
W = 1080; sh = Image.new("RGB", (W, 7000), (14, 15, 18)); d = ImageDraw.Draw(sh); y = 36
rd = lambda p: cv2.resize(cv2.imread(R + p), (720, 1280), interpolation=cv2.INTER_AREA)
def paste(im, x, y, w, h):
    sh.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)), (x, y))
d.text((40, y), sp["title"], font=F(52), fill=WH); y += 78
for s in sp.get("sub", []): d.text((40, y), s, font=F(28, "bold"), fill=GR); y += 44
y += 10; ok = sp["verdict"].startswith("PASS")
d.rounded_rectangle((40, y, W - 40, y + 110), 22, fill=(20, 36, 28) if ok else (36, 24, 24))
d.text((70, y + 28), sp["verdict"], font=F(38), fill=G if ok else RD); y += 140
cw = (W - 80 - 16) // 2; ch = int(cw * 1280 / 720)
def pair(items, y):
    for k, (lab, im) in enumerate(items):
        x = 40 + k * (cw + 16); d.text((x, y), lab, font=F(30), fill=WH); paste(im, x, y + 46, cw, ch)
    return y + 60 + ch
y = pair([("Day 4 원본 (기준)", rd(sp["ref"])), ("마스터 마네킹", rd(sp["master"]))], y)
y = pair([(s["label"], rd(s["img"])) for s in sp["stills"]], y)
z0, z1, x0, x1 = sp["zoom"]; zh = int(cw * (z1 - z0) / (x1 - x0))
d.text((40, y), "근육결 확대: Day 4 vs 바른 자세", font=F(30), fill=WH); y += 46
paste(rd(sp["ref"])[sp["ref_zoom"][0]:sp["ref_zoom"][1], sp["ref_zoom"][2]:sp["ref_zoom"][3]], 40, y, cw, zh)
paste(rd(sp["stills"][1]["img"])[z0:z1, x0:x1], 40 + cw + 16, y, cw, zh); y += zh + 30
names = [("size", "마네킹 크기·화면 안"), ("torso", "몸통 각도"), ("arm", "팔 펴짐 (시작)"), ("dumbbell", "덤벨 모양·개수"),
         ("head", "머리=몸 톤"), ("light", "그늘면 ≥ 배경"), ("sharp", "선명 (흐림 없음)"), ("glow", "근육 색 (빨강/초록)")]
for key, lab in names:
    d.text((40, y), lab, font=F(28, "bold"), fill=WH); yy = y + 40; extra = 0
    for k, s in enumerate(sp["stills"]):
        j = json.load(open(R + s["json"])); x = 40 + k * (cw + 16); v = "PASS" if j[key]["pass"] else "FAIL"
        note = {"size": lambda: f"키 {int(j['size']['height_frac'] * 100)}%", "torso": lambda: f"{j['torso']['lean_deg']}° (기준 {j['torso']['range'][0]}~{j['torso']['range'][1]})",
                "arm": lambda: f"굽힘 {j['arm']['elbow_flex_deg']}°", "dumbbell": lambda: f"{j['dumbbell']['found']}개 · 높이비 {j['dumbbell']['height_ratio']}",
                "head": lambda: f"ΔL {j['head']['dL']}", "light": lambda: f"{j['light']['diff']:+.1f}", "sharp": lambda: f"Day4 대비 {int(j['sharp']['frac'] * 100)}%",
                "glow": lambda: f"{'빨강' if j['glow']['colour'] == 'red' else '초록'} {j['glow']['frac'] * 100:.1f}%"}[key]()
        d.text((x, yy), f"자동 {v}", font=F(27), fill=COL[v]); d.text((x + 140, yy + 2), note, font=F(24, "bold"), fill=GR)
        e = s.get("eye", {}).get(key)
        if e: d.text((x, yy + 38), f"눈 {e[0]}", font=F(27), fill=COL[e[0]]); d.text((x + 140, yy + 40), e[1], font=F(24, "bold"), fill=GR); extra = 40
    y = yy + 46 + extra
d.text((40, y), "Day 4 느낌 (눈)", font=F(28, "bold"), fill=WH); yy = y + 40
for k, s in enumerate(sp["stills"]):
    e = s["eye"]["look"]; x = 40 + k * (cw + 16); d.text((x, yy), f"눈 {e[0]}", font=F(27), fill=COL[e[0]]); d.text((x + 140, yy + 2), e[1], font=F(24, "bold"), fill=GR)
y = yy + 60
for s in sp.get("notes", []): d.text((40, y), s, font=F(27, "bold"), fill=GR); y += 42
sh.crop((0, 0, W, y + 30)).save(R + sp["out"], quality=90); print(R + sp["out"], W, y + 30)
