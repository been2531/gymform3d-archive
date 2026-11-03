#!/usr/bin/env python3
"""t687u STILLS for the approved combo: smoked-glass character + 2-shot (wide whole machine -> tight, rear-3/4 high).
Input: 3 Grok Bot GenerateImage stills (720x1280 native) in output/_consistency/glass2shot/gen/ + kp.json/_seg.png
(MediaPipe). Output (1080x1920, Lanczos x1.5 of the native still; the glass body is smooth by design, and the fibres, rim
and callout are drawn at 1080): graded stills, machine-fidelity json, Day 4 metrics, a mobile comparison sheet.
usage: nice -n 19 python3 tools/glass_2shot_stills.py"""
import sys, json, time, subprocess, cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from gymform import glass_look as gl, joint_callout as jc
from build_v28_luxury_ab import f_pretendard
T0 = time.time(); R = "/workspace/daily_info_reels/"; G = R + "output/_consistency/glass2shot/gen/"; O = R + "output/_consistency/glass2shot/"
SHOTS = {"wide_peak": "wide_peak_gen1", "tight_peak": "tight_peak_gen2", "tight_start": "tight_start_gen3"}
KP = json.load(open(G + "kp.json")); S = 1.5
MP = {"sh_l": "11", "sh_r": "12", "elb_l": "13", "elb_r": "14", "wr_l": "15", "wr_r": "16", "hip_l": "23", "hip_r": "24",
      "kn_l": "25", "kn_r": "26", "ank_l": "27", "ank_r": "28", "head": "0"}
plate = cv2.imread(G + "plate_empty_1024.jpg"); real = cv2.imread(G + "ref_real_photo.jpg")
pl_g = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
pl_mach = (np.abs(plate.astype(int) - np.median(plate[20:200, 600:1000].reshape(-1, 3), 0)).max(2) > 18).astype(np.uint8)
PARTS = {"tower_frame": (232, 90, 380, 810), "stack_pin_horns": (250, 495, 352, 700), "low_pulley": (325, 585, 450, 660),
         "footplate": (392, 655, 592, 855), "rail_seat": (505, 725, 800, 905)}


def ssim(a, b, m):
    a = a.astype(np.float32); b = b.astype(np.float32); g = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)
    ma, mb = g(a), g(b); va = g(a * a) - ma ** 2; vb = g(b * b) - mb ** 2; cab = g(a * b) - ma * mb
    s = ((2 * ma * mb + 6.5) * (2 * cab + 58.5)) / ((ma ** 2 + mb ** 2 + 6.5) * (va + vb + 58.5))
    return float(s[m > 0].mean()) if m.any() else None


def align_to_plate(img, person):
    """similarity transform img -> plate (ORB on the machine, person masked out)"""
    orb = cv2.ORB_create(6000); gi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(gi, (1 - cv2.dilate(person, np.ones((15, 15), np.uint8))) * 255)
    k2, d2 = orb.detectAndCompute(pl_g, pl_mach * 255)
    mt = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda m: m.distance)[:800]
    A, inl = cv2.estimateAffinePartial2D(np.float32([k1[m.queryIdx].pt for m in mt]), np.float32([k2[m.trainIdx].pt for m in mt]),
                                         method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return A, int(inl.sum())


fid, out, info = {}, {}, {}
t_fid = time.time()
for nm, fn in SHOTS.items():
    img = cv2.imread(G + fn + ".jpg"); seg = (cv2.imread(G + fn + "_seg.png", 0) > 127).astype(np.uint8)
    A, ninl = align_to_plate(img, seg)
    w = cv2.warpAffine(img, A, (1024, 1024), flags=cv2.INTER_LINEAR); ws = cv2.warpAffine(seg, A, (1024, 1024), flags=cv2.INTER_NEAREST)
    valid = cv2.warpAffine(np.ones(img.shape[:2], np.uint8), A, (1024, 1024), flags=cv2.INTER_NEAREST)
    mm = pl_mach & (1 - cv2.dilate(ws, np.ones((21, 21), np.uint8))) & valid
    wg = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
    sc = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    parts = {}
    for p, (x0, y0, x1, y1) in PARTS.items():
        mk = np.zeros_like(mm); mk[y0:y1, x0:x1] = mm[y0:y1, x0:x1]
        parts[p] = {"ssim_vs_verified_plate": round(ssim(wg, pl_g, mk), 3) if mk.sum() > 200 else None,
                    "visible_frac": round(float(mk.sum() / max(1, (pl_mach[y0:y1, x0:x1]).sum())), 2)}
    ed_g = cv2.Canny(wg, 40, 120) & (mm * 255); ed_p = cv2.Canny(pl_g, 40, 120) & (mm * 255)
    dt = cv2.distanceTransform((ed_p == 0).astype(np.uint8), cv2.DIST_L2, 3)
    fid[nm] = {"orb_inliers": ninl, "scale_to_plate": round(sc, 3), "machine_ssim_all": round(ssim(wg, pl_g, mm), 3),
               "edge_chamfer_px_median": round(float(np.median(dt[ed_g > 0])), 2) if (ed_g > 0).any() else None, "parts": parts}
    cv2.imwrite(O + f"_align_{nm}_vs_plate.jpg", np.hstack([w, plate]))
    out[nm] = (img, seg, A)
fid["_note"] = ("verified plate = output/_consistency/d7m/empty_machine_t1_gen_raw.jpg (an edit of the real Matrix photo, "
                "PASS part-by-part in docs/mission/D7_MACHINE_CONCEPT.md §9). SSIM on machine pixels not covered by the person.")
t_fid = time.time() - t_fid

t_grade = time.time(); graded = {}; mets_in = []
for nm, (img, seg, A) in out.items():
    H, W = 1920, 1080
    up = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
    up = np.clip(up.astype(np.float32) * 1.4 - cv2.GaussianBlur(up, (0, 0), 1.0).astype(np.float32) * 0.4, 0, 255).astype(np.uint8)
    kp = {k: (KP[SHOTS[nm]][v][0] * S, KP[SHOTS[nm]][v][1] * S) for k, v in MP.items()}
    sega = cv2.resize(cv2.imread(G + SHOTS[nm] + "_seg.png", 0), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    sega = np.clip((sega - 0.35) / 0.3, 0, 1); body = (sega > 0.5).astype(np.uint8)
    # behind the body = the verified empty plate warped into this frame
    Ai = cv2.invertAffineTransform(A); Ai_up = Ai.copy(); Ai_up[:, :] *= S
    behind = cv2.warpAffine(plate, Ai_up, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    L = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY).astype(np.float32)
    wy = 0.5 * (kp["hip_l"][1] + kp["hip_r"][1]) - 40
    shorts = ((L < 62) & (body > 0) & (np.arange(H)[:, None] > wy)).astype(np.uint8)
    shorts = cv2.morphologyEx(shorts, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); shorts = cv2.GaussianBlur(shorts.astype(np.float32), (0, 0), 1.5)
    g = gl.glass_body(up, sega, behind=behind, shorts=shorts)
    # regions: canonical torso -> image; lats/rhomboids exclude the arm capsule, faint biceps on the visible upper arm
    Wc, Tc = gl.build_canon()
    M = gl.torso_affine(kp)
    Wm = cv2.warpAffine(Wc, M, (W, H)); Tm = cv2.warpAffine(Tc, M, (W, H))
    arm = np.zeros((H, W), np.float32); uw = np.linalg.norm(np.subtract(kp["sh_r"], kp["sh_l"]))
    vis = {sd: KP[SHOTS[nm]][MP[f"elb_{sd}"]][2] for sd in ("l", "r")}
    for a_, b_ in [p_ for sd in ("l", "r") if vis[sd] > 0.5 for p_ in ((f"sh_{sd}", f"elb_{sd}"), (f"elb_{sd}", f"wr_{sd}"))]:   # only arms in front of the back
        cv2.line(arm, tuple(int(v) for v in kp[a_]), tuple(int(v) for v in kp[b_]), 1.0, max(6, int(uw * 0.30)))
    arm = cv2.GaussianBlur(arm, (0, 0), 3)
    Wm = Wm * (1 - 0.9 * arm)
    bic = np.zeros((H, W), np.float32)
    sh_, el_ = np.array(kp["sh_l"]), np.array(kp["elb_l"])
    cv2.line(bic, tuple((sh_ + 0.3 * (el_ - sh_)).astype(int)), tuple((sh_ + 0.85 * (el_ - sh_)).astype(int)), 0.22, max(4, int(uw * 0.16)))
    Wm = np.maximum(Wm, cv2.GaussianBlur(bic, (0, 0), 3)) * cv2.erode(body, np.ones((5, 5), np.uint8)) * (1 - shorts)
    gain = 0.22 if "start" in nm else 1.0
    gr = gl.neon(g, Wm, Tm, sega, gain=gain).astype(np.uint8)
    rgb = cv2.cvtColor(gr, cv2.COLOR_BGR2RGB)
    jkp = {k: kp[k] for k in ("sh_l", "sh_r", "elb_l", "elb_r", "wr_l", "wr_r", "hip_l", "hip_r")}
    layer, ci = jc.draw(rgb, jkp, protect=Wm, is_wrong=False, label="ELBOWS BACK", ui_top=200, num_px=84, lab_px=26, max_dist=520,
                         prefer_xy=(kp["elb_l"][0] - 230, kp["elb_l"][1] - 330))
    fin = cv2.cvtColor(jc.composite(rgb, layer), cv2.COLOR_RGB2BGR)
    cv2.imwrite(O + f"g2s_{nm}_graded_1080.png", fin); cv2.imwrite(O + f"g2s_{nm}_nocallout_1080.png", gr)
    torso = float(np.linalg.norm(0.5 * (np.array(kp["sh_l"]) + np.array(kp["sh_r"])) - 0.5 * (np.array(kp["hip_l"]) + np.array(kp["hip_r"]))))
    lb = cv2.cvtColor(fin, cv2.COLOR_BGR2GRAY).astype(np.float32); bm = cv2.erode(body, np.ones((9, 9), np.uint8)) > 0
    head = bm & (np.arange(H)[:, None] < min(kp["sh_l"][1], kp["sh_r"][1]) - 0.35 * torso) & (Wm < 0.05)
    tor = bm & (Wm < 0.05) & (shorts < 0.5) & (np.arange(H)[:, None] > min(kp["sh_l"][1], kp["sh_r"][1])) & (np.arange(H)[:, None] < wy)
    ys, xs = np.where(body > 0)
    info[nm] = {"deg": ci.get("deg"), "callout_text_obstacle": ci.get("text_box_obstacle_frac"), "leader_on_target": ci.get("leader_on_target_muscle_frac"),
                "torso_px_1080": round(torso, 1), "body_h_frac_incl_seat_seg": round(float((ys.max() - ys.min()) / H), 3),
                "head_L": round(float(np.median(lb[head])), 1) if head.any() else None, "torso_L": round(float(np.median(lb[tor])), 1) if tor.any() else None,
                "backdrop_L": round(float(np.median(lb[150:400, 700:1000])), 1),
                "glow_px": int((Wm * sega > 0.3).sum()), "glow_outside_body_px": int(((Wm > 0.3) & (body == 0)).sum())}
    graded[nm] = fin
    if "peak" in nm: mets_in.append(f"{nm}={O}g2s_{nm}_nocallout_1080.png:right:{torso:.1f}")
t_grade = time.time() - t_grade
m = subprocess.run([sys.executable, R + "tools/day4_look_metrics.py", "day4=output/_consistency/d7m/day4_master_t17.5_right.jpg:right:356.4"] + mets_in,
                   capture_output=True, text=True, cwd=R)
try: mets = json.loads(m.stdout[m.stdout.index("{"):])
except Exception: mets = {"error": (m.stdout + m.stderr)[-800:]}

# ---------- mobile sheet (1080 wide, big Korean labels)
Wd = 1080; F = lambda s, w="extrabold": f_pretendard(s, w); G_ = (120, 230, 160); WH = (240, 240, 240); GR = (170, 175, 185)
def tile(im, w, h):
    s = min(w / im.shape[1], h / im.shape[0]); r = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    t = np.full((h, w, 3), 16, np.uint8); y0 = (h - r.shape[0]) // 2; x0 = (w - r.shape[1]) // 2; t[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
sh = Image.new("RGB", (Wd, 5200), (14, 15, 18)); d = ImageDraw.Draw(sh); y = 36
d.text((40, y), "D7 글라스 + 2컷 · 스틸 시트", font=F(62), fill=WH); y += 86
d.text((40, y), "이미지 생성 3장 · 영상/렌더 0", font=F(30, "medium"), fill=GR); y += 66
w2, h2 = 494, 878
for i, (nm, t) in enumerate([("wide_peak", "와이드 (기구 전체)"), ("tight_peak", "타이트 피크")]):
    x = 40 + i * (w2 + 12); sh.paste(tile(graded[nm], w2, h2), (x, y)); d.text((x, y + h2 + 10), t, font=F(40), fill=G_)
y += h2 + 76
w3, h3 = 326, 580
raws = [(out["wide_peak"][0], "원본 와이드"), (out["tight_start"][0], "원본 타이트 시작"), (out["tight_peak"][0], "원본 타이트 피크")]
for i, (im, t) in enumerate(raws):
    x = 40 + i * (w3 + 11); sh.paste(tile(im, w3, h3), (x, y)); d.text((x, y + h3 + 8), t, font=F(30), fill=WH)
y += h3 + 64
day4 = cv2.imread(R + "output/_consistency/d7m/day4_master_t17.5_right.jpg")
sh.paste(tile(graded["tight_start"], w3, h3), (40, y)); d.text((40, y + h3 + 8), "타이트 시작 (글로우 약)", font=F(28), fill=WH)
sh.paste(tile(day4, w3, h3), (40 + w3 + 11, y)); d.text((40 + w3 + 11, y + h3 + 8), "Day 4 기준", font=F(30), fill=WH)
cr = graded["tight_peak"][700:1500, 380:1000]; sh.paste(tile(cr, w3, h3), (40 + 2 * (w3 + 11), y)); d.text((40 + 2 * (w3 + 11), y + h3 + 8), "등 확대", font=F(30), fill=WH)
y += h3 + 70
mk = lambda k, nm: mets.get(nm, {}).get(k) if isinstance(mets, dict) else None
rows = [("색상", "core_hue_deg", 1), ("채도", "core_sat", 0), ("밝기", "core_val", 0), ("섬유 대비", "fibre_contrast", 1), ("섬유 밀도", "fibre_density_per100px", 1), ("번짐 px", "falloff_px", 1)]
d.text((40, y), "Day 4 대비 수치", font=F(44), fill=G_); y += 62
d.text((300, y), "Day4", font=F(30), fill=GR); d.text((520, y), "와이드", font=F(30), fill=GR); d.text((760, y), "타이트", font=F(30), fill=GR); y += 46
for lab, k, r in rows:
    vals = [mk(k, "day4"), mk(k, "wide_peak"), mk(k, "tight_peak")]
    d.text((40, y), lab, font=F(32), fill=WH)
    for j, v in enumerate(vals): d.text((300 + j * 230, y), "-" if v is None else (f"{v:.{r}f}"), font=F(32, "bold"), fill=WH)
    y += 48
y += 20
d.text((40, y), "기구 일치 (검증된 판 대비 SSIM)", font=F(44), fill=G_); y += 62
for nm, t in [("wide_peak", "와이드"), ("tight_peak", "타이트")]:
    p = fid[nm]["parts"]; s_ = " · ".join(f"{a}{'-' if p[k]['ssim_vs_verified_plate'] is None else p[k]['ssim_vs_verified_plate']:}" for k, a in
                                       [("tower_frame", "타워 "), ("stack_pin_horns", "스택 "), ("footplate", "발판 "), ("rail_seat", "레일·시트 ")])
    d.text((40, y), f"{t}: 전체 {fid[nm]['machine_ssim_all']}", font=F(32), fill=WH); y += 44
    d.text((40, y), s_, font=F(28, "bold"), fill=(210, 215, 220)); y += 50
y += 10
d.text((40, y), "머리톤 = 몸 (재질 동일)", font=F(32), fill=WH); y += 46
d.text((40, y), f"머리 {info['tight_peak']['head_L']} · 몸 {info['tight_peak']['torso_L']} · 배경 {info['tight_peak']['backdrop_L']}", font=F(30, "bold"), fill=(210, 215, 220)); y += 70
sh = sh.crop((0, 0, Wd, y)); sh.save(O + "G2S_STILLS_sheet.jpg", quality=92)
res = {"time_s": {"fidelity": round(t_fid, 1), "grade_3_stills": round(t_grade, 1), "total_script": round(time.time() - T0, 1)},
       "generations": {"GenerateImage": 3, "GenerateVideo": 0}, "shots": info, "machine_fidelity": fid, "metrics_vs_day4": mets}
json.dump(res, open(O + "g2s_stills_qc.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "metrics_vs_day4"}, indent=1)[:4000]); print(json.dumps(mets)[:1500])
