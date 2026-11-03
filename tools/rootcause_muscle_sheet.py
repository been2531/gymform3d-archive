#!/usr/bin/env python3
"""t689u root-cause sheet (no generation, no look code): same-scale back crops of Day 4 / D7 sculpted body + Day 4
grade / glass + code fibres (before, after), a by-eye cue checklist, and the 4 fundamental options. Mobile, Korean.
usage: python3 tools/rootcause_muscle_sheet.py"""
import sys, cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from build_v28_luxury_ab import f_pretendard
R = "/workspace/daily_info_reels/"; OUT = R + "docs/mission/ROOTCAUSE_MUSCLE_LOOK_sheet.jpg"
def sc(p, torso, cx, cy, w=470, h=560):
    im = cv2.imread(p); s = 356.4 / torso; im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    x0, y0 = int(cx * s - w / 2), int(cy * s - h / 2); x0 = max(0, min(im.shape[1] - w, x0)); y0 = max(0, min(im.shape[0] - h, y0))
    return im[y0:y0 + h, x0:x0 + w]
crops = [("Day 4", sc(R + "output/_consistency/d7m/day4_master_t17.5_right.jpg", 356.4, 560, 900)),
         ("D7 조각체+Day4", sc(R + "output/_consistency/still_motion/still_right_peak_1080.jpg", 290.5, 200, 1390)),
         ("글라스 (전)", sc(R + "output/_consistency/glass2shot/before_t688/g2s_tight_peak_graded_1080.png", 309.7, 860, 1110)),
         ("글라스 (후)", sc(R + "output/_consistency/glass2shot/g2s_tight_peak_graded_1080.png", 309.7, 860, 1110))]
F = lambda z, w="extrabold": f_pretendard(z, w); G = (120, 230, 160); WH = (240, 240, 240); GR = (170, 175, 185); RD = (255, 130, 110); YL = (250, 210, 110)
W = 1080; sh = Image.new("RGB", (W, 5200), (14, 15, 18)); d = ImageDraw.Draw(sh); y = 36
d.text((40, y), "근육 표현 · 근본 원인", font=F(64), fill=WH); y += 88
d.rounded_rectangle((40, y, W - 40, y + 236), 22, fill=(36, 24, 24))
d.text((70, y + 22), "원인", font=F(36), fill=RD)
for i, t in enumerate(["Day 4 = 몸 자체가 근육 조각 (빛이 굴곡을 탐)", "우리 = 매끈한 몸 위에 2D 결을 그림", "→ 스티커/섬처럼 보임. 수치로는 못 잡음"]):
    d.text((70, y + 72 + i * 50), t, font=F(34, "bold"), fill=WH)
y += 270
d.text((40, y), "같은 크기 비교", font=F(48), fill=G); y += 70
tw, th = 494, 588
for i, (t, im) in enumerate(crops):
    x = 40 + (i % 2) * (tw + 12); yy = y + (i // 2) * (th + 70)
    sh.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)), (x, yy))
    d.text((x + 6, yy + th + 8), t, font=F(38), fill=G if i < 2 else RD)
y += 2 * (th + 70) + 20
d.text((40, y), "진짜 근육으로 보이는 단서", font=F(48), fill=G); y += 70
cues = ["몸의 근육 굴곡이 빛을 받음", "근육 경계가 몸 전체로 이어짐", "결이 표면 곡률 따라 휨", "하이라이트·그림자·가림", "글로우와 몸이 한 덩어리"]
marks = [["O", "O", "X", "X"], ["O", "O", "X", "X"], ["O", "△", "X", "X"], ["O", "O", "X", "△"], ["O", "O", "X", "X"]]
cols = ["Day4", "D7조각", "글라스전", "글라스후"]
for j, c in enumerate(cols): d.text((560 + j * 125, y), c, font=F(24, "bold"), fill=GR)
y += 40
for c, mk in zip(cues, marks):
    d.text((40, y), c, font=F(30, "bold"), fill=WH)
    for j, m_ in enumerate(mk): d.text((585 + j * 125, y - 4), m_, font=F(40), fill=G if m_ == "O" else (YL if m_ == "△" else RD))
    y += 58
y += 20
d.rounded_rectangle((40, y, W - 40, y + 150), 22, fill=(28, 28, 36))
d.text((70, y + 20), "수치가 틀린 이유", font=F(34), fill=YL)
d.text((70, y + 70), "글로우 픽셀의 색·결 통계만 봄 → 몸 형태·연속성은 안 봄", font=F(28, "bold"), fill=WH); y += 190
d.text((40, y), "근본 해결안 4개", font=F(48), fill=G); y += 70
opts = [("A  근육 조각 마네킹 생성 + Day4 그레이드", "형태 = 원본에서 · Day4 검증됨 · 생성 2~3 · 렌더 1~2분", G),
        ("B  Blender + Z-Anatomy 3D 근육", "진짜 형상 · 리깅 수일 · GPU 없음 → 릴당 수 시간", WH),
        ("C  하이브리드 (근육 노멀맵만 Blender)", "몸에 맞추기 어려움 · B 준비 필요 · 위험 큼", WH),
        ("D  데믹식 전문 파이프라인", "리깅 캐릭터+근육 메쉬 2패스 · 유료 에셋/외주", WH)]
for t, s, c in opts:
    if c == G: d.rounded_rectangle((32, y - 10, W - 32, y + 100), 16, outline=G, width=5)
    d.text((50, y), t, font=F(34), fill=c); d.text((50, y + 50), s, font=F(27, "bold"), fill=(205, 210, 215)); y += 128
y += 10
d.rounded_rectangle((40, y, W - 40, y + 300), 24, fill=(20, 36, 28), outline=G, width=5)
d.text((70, y + 24), "추천: A  ·  검증 스틸 1장", font=F(44), fill=G)
for i, t in enumerate(["와이드 스틸의 몸만 Day4 조각 마네킹으로 교체", "기존 Day4 그레이드 그대로 적용", "Day4와 나란히 눈으로 판정: 단서 5개 전부 O"]):
    d.text((70, y + 96 + i * 62), t, font=F(32, "bold"), fill=WH)
y += 330
sh = sh.crop((0, 0, W, y)); sh.save(OUT, quality=92); print(OUT, sh.size)
