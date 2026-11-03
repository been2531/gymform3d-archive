#!/usr/bin/env python3
"""t691u D7 motion root-cause sheet (no generation, no render): frames read from the existing D7 clips
(grok.com v1/v2, GenerateVideo smoke/v2/v3, relight) + Days 1-4 for contrast, the failure catalogue and the
options. Mobile, big Korean labels. usage: nice -n 19 python3 tools/rootcause_d7_motion_sheet.py"""
import sys, cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from build_v28_luxury_ab import f_pretendard
R = "/workspace/daily_info_reels/"; A = R + "output/assets/"; OUT = R + "docs/mission/ROOTCAUSE_D7_MOTION_sheet.jpg"
def fr(p, i, crop=None):
    c = cv2.VideoCapture(R + p if not p.startswith("/") else p); c.set(1, i); ok, f = c.read()
    if crop: y0, y1, x0, x1 = crop; f = f[y0:y1, x0:x1]
    return f
F = lambda z, w="extrabold": f_pretendard(z, w)
G, WH, GR, RD, YL = (120, 230, 160), (240, 240, 240), (175, 180, 190), (255, 130, 110), (250, 210, 110)
W = 1080; sh = Image.new("RGB", (W, 9000), (14, 15, 18)); d = ImageDraw.Draw(sh); y = 36
def paste(im, x, y, w, h, tag=None):
    t = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    sh.paste(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)), (x, y))
    if tag: d.rounded_rectangle((x + 6, y + 6, x + 22 + 17 * len(tag), y + 46), 8, fill=(0, 0, 0)); d.text((x + 14, y + 10), tag, font=F(26), fill=WH)
d.text((40, y), "D7 동작 붕괴 · 근본 원인", font=F(62), fill=WH); y += 90
d.rounded_rectangle((40, y, W - 40, y + 290), 22, fill=(36, 24, 24)); d.text((70, y + 20), "원인", font=F(36), fill=RD)
for i, t in enumerate(["영상 모델은 매 프레임 전체를 새로 그림", "케이블=도르래→손, 손=손잡이, 발=발판 규칙을 모름", "고정 안 된 시간이 길수록(5~6초) 동작이 무너짐", "실제 기구+첫/끝 고정(v3)은 기구를 지켰음"]):
    d.text((70, y + 72 + i * 52), t, font=F(32, "bold"), fill=WH if i < 3 else G)
y += 320
# --- failure catalogue: (title, sub-lines, frames)
cat = [("① grok.com v1 · 고정 없음 6초", ["기구 옆에 앉음 · 케이블 방향 불명", "→ 플라이 · 머리 위 풀다운으로 변함, 손잡이 분리"],
        [("day7_wrong_raw_v1.mp4", 41, (0, 900, 0, 720), "f41"), ("day7_wrong_raw_v1.mp4", 102, (0, 900, 0, 720), "f102"), ("day7_right_raw_v1.mp4", 61, (0, 900, 0, 720), "f61")], RD),
       ("② grok.com v2 · 정면 카메라", ["당기는 방향이 화면 깊이 → 동작이 안 보임", "가짜 기구 (실제 Matrix와 다름)"],
        [("day7_wrong_raw_v2.mp4", 0, (200, 1100, 0, 720), "f0"), ("day7_wrong_raw_v2.mp4", 82, (200, 1100, 0, 720), "f82"), ("day7_right_raw_v2.mp4", 102, (200, 1100, 0, 720), "f102")], RD),
       ("③ 고정 스모크 (첫=끝 핀) · 가짜 기구", ["화면은 안정 · 하지만 동작 얕음 (팔꿈치 43~59°)", "\"1회\" 요청에도 2회 반복"],
        [(R + "output/_consistency/smoke_d7v2/d7v2_right_720p_t2.mp4", 0, (100, 900, 0, 720), "f0"), (R + "output/_consistency/smoke_d7v2/d7v2_right_720p_t2.mp4", 60, (100, 900, 0, 720), "f60"), (R + "output/_consistency/smoke_d7/d7_smoke_right_720p_t2.mp4", 40, (200, 1100, 0, 720), "f40")], YL),
       ("④ v3 · 실제 기구 + 첫=끝 핀 · 5초", ["기구 SSIM 0.99 유지 ✓ · 케이블 도르래에서 ✓", "가운데 자유 구간: 몸 흔들림 19~23°, 반 회 추가,", "팔이 시작보다 더 뻗음 · 사각 손잡이(핀) · 그늘 어두움"],
        [("day7_wrong_raw_v3.mp4", 0, (380, 1280, 0, 720), "f0"), ("day7_wrong_raw_v3.mp4", 57, (380, 1280, 0, 720), "f57"), ("day7_right_raw_v3.mp4", 80, (380, 1280, 0, 720), "f80")], YL)]
tw, th = 322, 400
for title, subs, frames, col in cat:
    d.text((40, y), title, font=F(40), fill=col); y += 58
    for s in subs: d.text((40, y), s, font=F(29, "bold"), fill=WH); y += 44
    y += 6
    for k, (p, i, cr, tag) in enumerate(frames):
        paste(fr(p if p.startswith("/") else "output/assets/" + p, i, cr), 40 + k * (tw + 12), y, tw, th, tag)
    y += th + 40
d.text((40, y), "⑤ Blender 1~6차", font=F(40), fill=RD); y += 58
for s in ["팔 X자 꼬임 · 어깨 틈 · 바 위치 오류 (6차)", "상자형 기구 · 매끈한 몸 (Day 4 룩 불가)"]: d.text((40, y), s, font=F(29, "bold"), fill=WH); y += 44
d.text((40, y), "⑥ 조명 보정(relight): 그늘만 밝힘, 동작은 그대로", font=F(32), fill=GR); y += 80
# --- Days 1-6 contrast
d.text((40, y), "Day 1~6은 왜 됐나", font=F(48), fill=G); y += 70
dd = [("output/day4_rdl_step1_locked_720p.mp4", 82, (250, 1150, 0, 720), "D4"), ("output/day2_v38_master_reel.mp4", 427, (250, 1500, 0, 1080), "D2"), ("output/day1_v38_master_reel.mp4", 342, (250, 1500, 0, 1080), "D1")]
for k, (p, i, cr, tag) in enumerate(dd): paste(fr(p, i, cr), 40 + k * (tw + 12), y, tw, th, tag)
y += th + 24
for s in ["기구가 손과 함께 움직임 (바벨·덤벨) 또는 배경일 뿐", "동작이 화면 평면 안에서 크고 흔함 (힌지·내리기)", "실제 기구와 똑같을 필요 없었음", "D7: 고정 기구 + 움직이는 손 + 케이블을 한 장면에서 맞춰야 함"]:
    d.text((40, y), s, font=F(30, "bold"), fill=WH if not s.startswith("D7") else YL); y += 48
y += 30
# --- options
d.text((40, y), "해결안 5개", font=F(48), fill=G); y += 70
opts = [("A  사람만 생성 → 실제 기구 위 합성", "기구 100% · 손잡이·발판 접촉 어긋남 · 가장자리 깜빡임(D4)", WH),
        ("B  2초 짧은 고정 클립 + 복귀는 역재생", "시작→수축만 생성 · 반복·흔들림 시간 없음 · 영상 1개/폼", G),
        ("C  스틸 2장 코드 퍼펫", "기구 100% · 영상 0 · 고무 같은 움직임 위험", WH),
        ("D  Blender 뼈대로 C 구동", "6차 팔 꼬임 · 2D 관절 계산이면 충분 · 가치 낮음", WH),
        ("E  모션 전이 (Kling, Higgsfield)", "실제 사람 동작 · 유료 크레딧 · 참고 영상 필요", WH)]
for t, s, c in opts:
    if c == G: d.rounded_rectangle((32, y - 10, W - 32, y + 100), 16, outline=G, width=5)
    d.text((50, y), t, font=F(34), fill=c); d.text((50, y + 50), s, font=F(26, "bold"), fill=(205, 210, 215)); y += 128
y += 10
d.rounded_rectangle((40, y, W - 40, y + 420), 24, fill=(20, 36, 28), outline=G, width=5)
d.text((70, y + 24), "추천: B · 검증 영상 1개", font=F(44), fill=G)
for i, t in enumerate(["기존 정자세 시작·수축 스틸 2장 (같은 카메라)", "2초 · 첫=시작, 끝=수축 · 720p", "복귀는 코드로 역재생 → 한 번만, 루프", "판정: 기구 안 변함 · 손=손잡이 · 케이블=도르래", "      자연스러운 로우 · 팔 꼬임 없음"]):
    d.text((70, y + 96 + i * 60), t, font=F(31, "bold"), fill=WH)
y += 450
sh = sh.crop((0, 0, W, y)); sh.save(OUT, quality=90); print(OUT, sh.size)
