#!/usr/bin/env python3
"""Option-B test sheet (mobile, Korean): frame strip of the 2 s pinned clip + PASS/FAIL per criterion, from
output/_consistency/optB/optb_qc.json. Also writes a code-only trimmed loop (f22->f48, no new generation) as a
diagnostic variant. usage: nice -n 19 python3 tools/optb_motion_sheet.py"""
import sys, json, os, subprocess, cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from build_v28_luxury_ab import f_pretendard
R = "/workspace/daily_info_reels/"; O = R + "output/_consistency/optB/"
q = json.load(open(O + "optb_qc.json"))
c = cv2.VideoCapture(O + "d7_optB_right_pull_2s_720p_raw.mp4"); F = []
while True:
    ok, f = c.read()
    if not ok: break
    F.append(f)
n = len(F); fps = 24; T0 = 22
# trimmed diagnostic loop: f22 (arms extended, cable visible, speed ~0) -> f48, hold 0.3 s, reverse, x4
cyc = list(range(T0, n)) + [n - 1] * 7 + list(range(n - 2, T0, -1)); seq = cyc * 4 + [T0]
w = cv2.VideoWriter(O + "_t.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (720, 1280))
for i in seq: w.write(F[i])
w.release()
subprocess.run(["nice", "-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-i", O + "_t.avi", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", O + "d7_optB_right_loop_trim_f22_codeonly.mp4"], check=True)
os.remove(O + "_t.avi")
ps = [p for p in q["pose_series"] if p]; trim = [p for p in ps if p["f"] >= T0]
trim_torso = round(max(p["torso_deg"] for p in trim) - min(p["torso_deg"] for p in trim), 1)
q["trim_f22"] = {"torso_deg_range": trim_torso, "elbow_flex_minmax": [min(p["elbow_flex"] for p in trim), max(p["elbow_flex"] for p in trim)],
                 "cycle_s": round(len(cyc) / fps, 2)}
json.dump(q, open(O + "optb_qc.json", "w"), indent=1)
F_ = lambda z, w="extrabold": f_pretendard(z, w)
G, WH, GR, RD, YL = (120, 230, 160), (240, 240, 240), (175, 180, 190), (255, 130, 110), (250, 210, 110)
W = 1080; sh = Image.new("RGB", (W, 6000), (14, 15, 18)); d = ImageDraw.Draw(sh); y = 36
def paste(im, x, y, w, h, tag=None):
    t = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA); sh.paste(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)), (x, y))
    if tag: d.rounded_rectangle((x + 4, y + 4, x + 20 + 16 * len(tag), y + 42), 8, fill=(0, 0, 0)); d.text((x + 11, y + 8), tag, font=F_(24), fill=WH)
d.text((40, y), "B안 테스트 · 2초 고정 클립", font=F_(58), fill=WH); y += 80
d.text((40, y), "영상 1개 · 이미지 0 · 그레이딩 없음 (동작만)", font=F_(30, "bold"), fill=GR); y += 70
d.rounded_rectangle((40, y, W - 40, y + 130), 22, fill=(36, 24, 24))
d.text((70, y + 18), "결과: FAIL (2/5 통과)", font=F_(44), fill=RD)
d.text((70, y + 78), "기구·손잡이는 OK · 시작 핀 때문에 동작이 엇나감", font=F_(29, "bold"), fill=WH); y += 160
tw, th = 322, 402
for row in ([0, 12, 24], [32, 40, 48]):
    for k, i in enumerate(row): paste(F[i][380:1280, 0:720], 40 + k * (tw + 12), y, tw, th, f"f{i}")
    y += th + 12
y += 10
p = {pp["f"]: pp for pp in ps}
d.text((40, y), "f0→f24: 팔을 더 뻗고 몸이 앞으로 (당기기 전 '뻗기')", font=F_(28, "bold"), fill=YL); y += 44
d.text((40, y), "f24→f48: 한 번 당김 · 몸이 뒤로 19°", font=F_(28, "bold"), fill=YL); y += 64
mi = q["machine_ssim_vs_start_pin_min"]
crit = [("① 기구 안 변함", "PASS", f"정지 영역 SSIM {q['machine_ssim_all_static_min']} · 눈으로 동일", f"(타워 크롭 {mi['tower+stack']}: 두 핀끼리도 0.989)"),
        ("② 두 손 = 손잡이 · 분리 없음", "PASS", "전 프레임 두 손이 손잡이 위", "손잡이는 사각형 (핀에서 옴)"),
        ("③ 케이블 = 도르래에서 한 줄", "FAIL", "f0 케이블 없음 · f1 공중에 뜸 · f2부터 OK", "→ 루프마다 케이블이 깜빡임"),
        ("④ 자연스러운 한 번 당김", "FAIL", f"몸통 {q['torso_deg_range']}° (≤10°) · 먼저 78px 앞으로 뻗음", "팔꿈치가 등선 뒤로 안 감 · 꼬임/플라이/풀다운 없음"),
        ("⑤ 루프 이음새 안 보임", "FAIL", "프레임 점프는 없음 · 시작점에서 빠르게 튕김", "(손목 10~13px/프레임) + 케이블 깜빡임")]
for t, v, a, b in crit:
    col = G if v == "PASS" else RD
    d.rounded_rectangle((40, y, W - 40, y + 150), 18, fill=(24, 26, 30), outline=col, width=4)
    d.text((64, y + 14), t, font=F_(34), fill=WH); d.text((W - 190, y + 14), v, font=F_(38), fill=col)
    d.text((64, y + 66), a, font=F_(26, "bold"), fill=(215, 220, 225)); d.text((64, y + 106), b, font=F_(24, "bold"), fill=GR); y += 168
y += 10
# cable evidence
d.text((40, y), "케이블 증거 (f0 · f1 · f2)", font=F_(36), fill=YL); y += 56
for k, i in enumerate([0, 1, 2]): paste(F[i][800:940, 180:470], 40 + k * (tw + 12), y, tw, 156, f"f{i}")
y += 180
d.rounded_rectangle((40, y, W - 40, y + 330), 24, fill=(20, 36, 28), outline=G, width=5)
d.text((70, y + 22), "원인 & 다음 단계", font=F_(42), fill=G)
for i, t in enumerate(["시작 핀: 팔이 덜 뻗고 케이블 없음 → 모델이 '뻗기' 추가", "수축 핀: 몸이 뒤로 19° 기울어짐 → 몸통 기준 초과", "다음: 핀 2장 다시 (몸통 같게 · 팔 쭉 · 케이블 보임 · V 손잡이)", f"코드만 자른 루프(f22→48): 몸통 {trim_torso}° · 케이블 OK"]):
    d.text((70, y + 92 + i * 56), t, font=F_(28, "bold"), fill=WH)
y += 360
sh = sh.crop((0, 0, W, y)); sh.save(O + "OPTB_TEST_sheet.jpg", quality=90); print(O + "OPTB_TEST_sheet.jpg", sh.size, q["trim_f22"])
