#!/usr/bin/env python3
"""t685u CHARACTER x COMPOSITION concept v2 (no generation): 3 character directions mocked in code on the existing D7
right-peak frame (real machine pixels; the see-through glass uses the real empty plate behind the body), 3 composition
wireframes projected from the Matrix Aura Low Row geometry (tools/composition_fill_estimate.py), mobile-readable Korean
labels. usage: nice -n 19 python3 tools/concept_v2_sheet.py"""
import sys, json, cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels"); sys.path.insert(0, "/workspace/daily_info_reels/tools")
from build_v28_luxury_ab import f_pretendard
import composition_fill_estimate as CF
R = "/workspace/daily_info_reels/"; SM = R + "output/_consistency/still_motion/"; SC = R + "_scratch/cr/"; OUT = R + "docs/mission/"
base = cv2.imread(SM + "still_right_peak_base_1080.png"); glow = cv2.imread(SM + "still_right_peak_1080.jpg")
kp = json.load(open(SM + "still_right_peak.json"))["kp"]
e = cv2.resize(cv2.flip(cv2.imread(R + "output/_consistency/d7m/pin_empty_machine_720x1280.png"), 1), (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
E = np.zeros_like(e); E[:, 14:] = e[:, :-14]; E[:, :14] = e[:, :1]
# silhouette: MediaPipe pose segmentation (pose_landmarker_heavy) on the graded frame, minus the seat post
import universal_anatomy_rig as uar
_r = uar.get_pose_detector().detect(uar.mp.Image(image_format=uar.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))))
pm = (_r.segmentation_masks[0].numpy_view()[..., 0] > 0.5).astype(np.uint8)
hy = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1]); pm[int(hy) + 60:, :int(kp["kn_r"][0]) - 60] = 0
n, lab, st, _ = cv2.connectedComponentsWithStats(pm); pm = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
pa = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 1.2)[..., None]
gb = glow.astype(np.int32); gdom = gb[..., 1] - np.maximum(gb[..., 0], gb[..., 2])
gw = np.clip((gdom - 15) / 60.0, 0, 1).astype(np.float32) ** 0.7; gw[:1150] = 0; gw[:, 330:] = 0; gw *= pm
L = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
sm = cv2.GaussianBlur(cv2.bilateralFilter(base, 0, 40, 9), (0, 0), 5); Ls = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32)
med = float(np.median(Ls[pm > 0]))
glow_rgb = glow.astype(np.float32)

def put_glow(img, k=1.0):
    o = img.astype(np.float32) * (1 - gw[..., None] * k) + glow_rgb * gw[..., None] * k
    return np.clip(o + cv2.GaussianBlur(glow_rgb * gw[..., None], (0, 0), 10) * 0.6 * k, 0, 255).astype(np.uint8)

# 1 ecorche: the Day 4 graded body as is (generator muscle relief) + code glow, head toned to the body
ecor = glow.copy()
# 2 smoked glass: 30 % see-through to the REAL empty plate behind, cool tint, fresnel rim from the silhouette distance
dist = cv2.distanceTransform(pm, cv2.DIST_L2, 5); rim = np.exp(-dist / 7.0) * pm
inner = 0.30 * E.astype(np.float32) + 0.70 * np.array([34, 26, 18], np.float32)          # BGR deep navy smoke
shade = (0.85 + 0.30 * (Ls - med) / 60.0)[..., None]
glass = inner * shade + rim[..., None] * np.array([235, 225, 200], np.float32) * 0.85
glass = np.clip(base * (1 - pa) + glass * pa, 0, 255).astype(np.uint8); glass = put_glow(glass, 1.15)
# 3 light clay: bright matte grey from the smoothed shading (relief kept soft), head = body
lc = 150 + 0.9 * (Ls - med); clay = np.stack([lc * 1.0, lc * 0.99, lc * 0.97], -1)
clay = np.clip(base * (1 - pa) + clay * pa, 0, 255).astype(np.uint8)
cg = gw[..., None]; col = np.array([110, 220, 60], np.float32)                            # green on light needs a darker core
clay = np.clip(clay * (1 - 0.85 * cg) + (glow_rgb * 0.75) * 0.85 * cg, 0, 255).astype(np.uint8)
for nm, im in [("ecor", ecor), ("glass", glass), ("clay", clay)]: cv2.imwrite(SC + f"v2_{nm}.png", im)

def stats(im):
    l = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32); body = (cv2.erode(pm, np.ones((9, 9), np.uint8)) > 0) & (gw < 0.05)
    head = body & (np.arange(1920)[:, None] < kp["sh"][1] - 110)
    lap = cv2.Laplacian(cv2.GaussianBlur(l, (0, 0), 1.0), cv2.CV_32F)
    return {"body_L": round(float(np.median(l[body])), 1), "head_L": round(float(np.median(l[head])), 1) if head.any() else None,
            "backdrop_L": round(float(np.median(l[200:400, 50:300])), 1), "body_texture_lap_std": round(float(lap[body].std()), 2)}
S = {k: stats(v) for k, v in [("ecorche", ecor), ("glass", glass), ("clay", clay)]}

# ---- composition wireframes
def wire(yaw, pitch, tgt=(150, 0, 100), pts=None, margin=0.04, W=1080, H=1920, crop_note=None):
    r = CF.fit(yaw, pitch, tgt=tgt, frame_pts=pts, margin=margin, W=W, H=H)
    img = np.full((H, W, 3), (30, 26, 23), np.uint8)
    def P(ps): return CF.project(ps, r["_cam"], r["_tgt"], r["_f"]) + np.array(r["_off"]) if pts is None else CF.project(ps, r["_cam"], r["_tgt"], r["_f"]) + np.array([W / 2, H / 2]) - 0.5 * (lambda a: a.max(0) + a.min(0))(CF.project(pts, r["_cam"], r["_tgt"], r["_f"]))
    for box in [CF.TOWER, CF.RAIL, CF.SEAT, CF.FOOT]:
        a = P(box).astype(int)
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count("1") == 1: cv2.line(img, tuple(a[i]), tuple(a[j]), (225, 225, 225), 5, cv2.LINE_AA)
    k = {n: tuple(P([v])[0].astype(int)) for n, v in CF.PERSON.items()}
    for a_, b_, w_ in [("hip_l", "kn_l", 46), ("kn_l", "ank_l", 36), ("hip_r", "kn_r", 46), ("kn_r", "ank_r", 36), ("neck", "hip_l", 70), ("neck", "hip_r", 70),
                       ("sh_l", "elb_l", 30), ("elb_l", "wr_l", 26), ("sh_r", "elb_r", 30), ("elb_r", "wr_r", 26), ("sh_l", "sh_r", 40)]:
        s_ = max(0.4, r["_f"] / max(200.0, r["dist_cm"]) / 4.5)
        cv2.line(img, k[a_], k[b_], (150, 220, 120), max(6, int(w_ * s_)), cv2.LINE_AA)
    cv2.circle(img, k["head"], max(14, int(42 * max(0.4, r["_f"] / max(200.0, r["dist_cm"]) / 4.5))), (150, 220, 120), -1, cv2.LINE_AA)
    return img, r

wA, rA = wire(35, 28)                                                           # rear-3/4 high, whole machine
wT, rT = wire(35, 22, tgt=(170, 0, 80), pts=list(CF.PERSON.values()), margin=0.07)   # tight cut, same angle
wF, rF = wire(140, 8)                                                           # front-3/4 eye
S["composition"] = {"A_rear34_high_one_shot": {"person_h": rA["person_h_frac"], "machine_h": rA["machine_h_frac"]},
                    "B_two_shot_wide_then_tight": {"wide_person_h": rA["person_h_frac"], "tight_person_h": rT["person_h_frac"]},
                    "C_front34_eye_one_shot": {"person_h": rF["person_h_frac"], "machine_h": rF["machine_h_frac"]}}
json.dump(S, open(OUT + "CONCEPT_V2_metrics.json", "w"), indent=1)

# ---- mobile sheet 1080 wide
W = 1080; G = (120, 230, 160); WH = (240, 240, 240); GR = (170, 175, 185)
crop = (0, 960, 700, 1760)
def tile(im, w, h, c=None):
    if c: im = im[c[1]:c[3], c[0]:c[2]]
    s = min(w / im.shape[1], h / im.shape[0]); im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    t = np.full((h, w, 3), 17, np.uint8); y0 = (h - im.shape[0]) // 2; x0 = (w - im.shape[1]) // 2; t[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
H = 3900; sh = Image.new("RGB", (W, H), (14, 15, 18)); d = ImageDraw.Draw(sh); F = lambda s, w="extrabold": f_pretendard(s, w)
y = 40
d.text((40, y), "D7 캐릭터 · 구도 재구상", font=F(64), fill=WH); y += 90
d.text((40, y), "생성 0건 · 코드 목업 + 실제 기구 치수", font=F(30, "medium"), fill=GR); y += 70
d.rounded_rectangle((40, y, W - 40, y + 190), 22, fill=(32, 26, 26))
d.text((70, y + 24), "핵심 문제", font=F(34), fill=(255, 140, 120))
d.text((70, y + 74), "기구 높이 2.4m · 앉은 사람 1.3m", font=F(38), fill=WH)
d.text((70, y + 124), f"기구 전체 1컷이면 사람은 화면의 {rA['person_h_frac']*100:.0f}~{rF['person_h_frac']*100:.0f}%가 한계", font=F(34, "bold"), fill=WH); y += 230
d.text((40, y), "캐릭터 3안", font=F(52), fill=G); y += 76
cw, ch = 326, 420
chars = [("① 해부학 근육체", v, t) for v, t in []]
chars = [("①  근육체", ecor, ["움직임 ●○○", "근섬유 ●●○", "머리톤 ●○○"]),
         ("②  스모크 글라스", glass, ["움직임 ●●●", "근섬유 ●●●", "머리톤 ●●●"]),
         ("③  라이트 클레이", clay, ["움직임 ●●○", "근섬유 ●○○", "머리톤 ●●●"])]
for i, (t, im, tags) in enumerate(chars):
    x = 40 + i * (cw + 11); sh.paste(tile(im, cw, ch, crop), (x, y))
    if i == 1: d.rounded_rectangle((x - 4, y - 4, x + cw + 4, y + ch + 4), 14, outline=G, width=6)
    d.text((x + 4, y + ch + 14), t, font=F(36), fill=G if i == 1 else WH)
    for j, tg in enumerate(tags): d.text((x + 4, y + ch + 64 + j * 44), tg, font=F(31, "bold"), fill=WH)
y += ch + 64 + 3 * 44 + 30
d.text((40, y), "② = 몸은 반투명, 타깃 근육만 네온 · 뒤 기구가 비쳐 보임", font=F(30, "bold"), fill=GR); y += 70
d.text((40, y), "구도 3안 (기구 변형 0, 카메라만)", font=F(52), fill=G); y += 76
cw2, ch2 = 326, 580
comps = [("Ⓐ 뒤3/4 하이", [wA], [f"사람 {rA['person_h_frac']*100:.0f}%", f"기구 {rA['machine_h_frac']*100:.0f}%"]),
         ("Ⓑ 2컷 와이드→타이트", [wA, wT], [f"사람 {rA['person_h_frac']*100:.0f}% → {rT['person_h_frac']*100:.0f}%", "매 루프 기구 전체"]),
         ("Ⓒ 앞3/4 눈높이", [wF], [f"사람 {rF['person_h_frac']*100:.0f}%", "등 근육 안 보임"])]
for i, (t, ims, tags) in enumerate(comps):
    x = 40 + i * (cw2 + 11)
    if len(ims) == 1: sh.paste(tile(ims[0], cw2, ch2), (x, y))
    else:
        sh.paste(tile(ims[0], cw2 // 2 - 4, ch2), (x, y)); sh.paste(tile(ims[1], cw2 // 2 - 4, ch2), (x + cw2 // 2 + 4, y))
        d.text((x + cw2 // 2 - 18, y + ch2 // 2 - 30), "→", font=F(48), fill=G)
    if i == 1: d.rounded_rectangle((x - 4, y - 4, x + cw2 + 4, y + ch2 + 4), 14, outline=G, width=6)
    d.text((x + 4, y + ch2 + 14), t, font=F(32), fill=G if i == 1 else WH)
    for j, tg in enumerate(tags): d.text((x + 4, y + ch2 + 60 + j * 42), tg, font=F(29, "bold"), fill=WH)
y += ch2 + 60 + 2 * 42 + 50
d.rounded_rectangle((40, y, W - 40, y + 700), 26, fill=(20, 36, 28), outline=G, width=5)
d.text((70, y + 30), "추천: ② 스모크 글라스 + Ⓑ 2컷", font=F(50), fill=G)
reasons = [("움직임", "몸 표면 디테일 0 → 일렁임 없음"), ("근섬유", "Day 4 네온을 코드로 100% 그림"), ("머리톤", "몸과 같은 재질, 자동 일치"),
           ("기구", "와이드 컷에서 전체 + 몸 뒤로 비침"), ("고급감", "어두운 글라스 + 네온 = 프리미엄"), ("속도", "판 2장 + 클립 2개, 나머지 코드")]
for j, (a, b) in enumerate(reasons):
    yy = y + 120 + j * 92; d.text((70, yy), a, font=F(36), fill=WH); d.text((260, yy + 2), b, font=F(34, "bold"), fill=(215, 220, 225))
y += 740
d.text((40, y), "목업은 코드 근사치 (실제 품질 아님) · 기구 수치: Matrix 스펙 230×71×240cm", font=F(26, "medium"), fill=GR); y += 60
sh = sh.crop((0, 0, W, y)); sh.save(OUT + "CONCEPT_V2_sheet.jpg", quality=92)
print(json.dumps(S, indent=1))
