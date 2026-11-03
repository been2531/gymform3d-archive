#!/usr/bin/env python3
"""t684u CHARACTER CONCEPT sheet (no generation): 3 simplified-mannequin options mocked in CODE on the existing D7 right
peak frame (same pose, same real machine pixels), next to the Day 1 / Day 2 / Day 4 references and the current D7 reel
frame. The fibre look on A and B is code-synthesised (oriented stripes inside the glow mask, Day 4 right HSV targets),
which is what the production path would have to do on a smooth mannequin. Writes docs/mission/CHARACTER_CONCEPT_*.
usage: nice -n 19 python3 tools/character_concept_sheet.py"""
import sys, json, subprocess, cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/workspace/daily_info_reels")
from build_v28_luxury_ab import f_pretendard
R = "/workspace/daily_info_reels/"; SM = R + "output/_consistency/still_motion/"; OUT = R + "docs/mission/"
SC = R + "_scratch/cr/"
base = cv2.imread(SM + "still_right_peak_base_1080.png")          # Day 4 graded, no glow, no callout
glow = cv2.imread(SM + "still_right_peak_1080.jpg")               # Day 4 graded with glow + callout (current)
kp = json.load(open(SM + "still_right_peak.json"))["kp"]
# empty plate in the same 1080 space (found by template match: mirror, x1.5, +14 px x)
e = cv2.resize(cv2.flip(cv2.imread(R + "output/_consistency/d7m/pin_empty_machine_720x1280.png"), 1), (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
E = np.zeros_like(e); E[:, 14:] = e[:, :-14]; E[:, :14] = e[:, :1]
# person mask = Day 4 foreground weight inside keypoint capsules (head disc, torso quad, arms, legs); the capsules keep
# the machine out (w_fg also covers the machine) and catch the dark head that a plate diff misses
wfg = np.load(SM + "still_right_peak_regions.npz")["w_fg"]
P = lambda k: tuple(int(v) for v in kp[k])
cap = np.zeros((1920, 1080), np.uint8); sh = np.array(kp["sh"])
cv2.circle(cap, (int(sh[0]), int(sh[1] - 185)), 92, 1, -1); cv2.line(cap, (int(sh[0]), int(sh[1] - 185)), P("sh"), 1, 90)
cv2.fillConvexPoly(cap, np.array([P("sh_r"), P("sh_l"), P("hip_l"), P("hip_r")]), 1)
cv2.dilate(cap, np.ones((61, 61), np.uint8), dst=cap)
for a_, b_, w_ in [("sh_r", "elb_r", 70), ("elb_r", "wr_r", 60), ("sh_l", "elb_l", 70), ("elb_l", "wr_l", 60),
                   ("hip_r", "kn_r", 110), ("hip_l", "kn_l", 110), ("kn_r", "ank_r", 80), ("kn_l", "ank_l", 80)]:
    cv2.line(cap, P(a_), P(b_), 1, w_)
pm = ((wfg > 70) & (cap > 0)).astype(np.uint8)
# the dark head only has edge evidence in w_fg (head ~ backdrop level): fill the convex hull of it inside the head disc
hd = np.zeros_like(pm); cv2.circle(hd, (int(sh[0]), int(sh[1] - 185)), 110, 1, -1); hd[int(sh[1]) - 60:] = 0
ys_, xs_ = np.where((wfg > 120) & (hd > 0))
if len(xs_) > 50:
    el = cv2.fitEllipse(np.stack([xs_, ys_], 1).astype(np.int32)); el = (el[0], (el[1][0] * 0.78, el[1][1] * 0.8), el[2])
    cv2.ellipse(pm, el, 1, -1)
    nb = np.zeros_like(pm); cv2.line(nb, (int(sh[0]), int(sh[1] - 120)), P("sh"), 1, 70); pm |= nb & ((wfg > 30) | (cv2.dilate(pm, np.ones((25, 25), np.uint8)) > 0)).astype(np.uint8)
hy = 0.5 * (kp["hip_r"][1] + kp["hip_l"][1]); pm[int(hy) + 50:, :int(kp["kn_r"][0]) - 60] = 0   # seat pad underside / post
pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(pm); pm = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
ff = pm.copy() * 255; cv2.floodFill(ff, np.zeros((1922, 1082), np.uint8), (5, 5), 128); pm = (ff != 128).astype(np.uint8)   # fill holes
pm = (cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 6) > 0.5).astype(np.uint8)                                          # smooth contour
cv2.imwrite(SC + "person_mask.png", pm * 255)
pa = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 1.2)[..., None]
gb = glow.astype(np.int32); gdom = gb[..., 1] - np.maximum(gb[..., 0], gb[..., 2])
gm = ((gdom > 30) & (cv2.erode(pm, np.ones((5, 5), np.uint8)) > 0)).astype(np.uint8); gm[:1150] = 0
tb = json.load(open(SM + "still_right_peak.json"))["joint_callout"].get("text_box")
gm[:, 330:] = 0   # callout / leader live right of the back; the glow regions (lats, rhomboids, rear delt) are left of x 330
gm = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
L = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
bgL = float(np.median(L[200:400, 50:300]))
head = (pm > 0) & (np.arange(1920)[:, None] < kp["sh"][1] - 110)
torso = (pm > 0) & (np.arange(1920)[:, None] > kp["sh"][1]) & (np.arange(1920)[:, None] < kp["hip_r"][1])


def fibres(img, strength=1.0):
    """code fibres: stripes along the lat line (armpit -> opposite hip) + rear-delt line, 10/100 px, 4 px falloff"""
    H, W = gm.shape; yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    def field(p0, p1):
        a = np.array(kp[p1], np.float32) - np.array(kp[p0], np.float32); a /= np.linalg.norm(a); nrm = np.array([-a[1], a[0]])
        return (xx * nrm[0] + yy * nrm[1]), (xx * a[0] + yy * a[1])
    n1, t1 = field("sh_l", "hip_r")            # lats: armpit -> opposite hip / spine
    n2, t2 = field("sh_r", "sh_l")             # rhomboids / mid trap / rear delt: across the scapula
    wy = np.clip((yy - (kp["sh"][1] + 40)) / 90.0, 0, 1)
    def st_(nn, tt): return 0.5 + 0.5 * np.cos(nn / 10.0 * 2 * np.pi + 0.8 * np.sin(tt / 45.0))
    stripe = wy * st_(n1, t1) + (1 - wy) * st_(n2, t2)
    dist = cv2.distanceTransform(gm, cv2.DIST_L2, 5); w = np.clip(dist / 4.0, 0, 1) * strength
    col = cv2.cvtColor(np.uint8([[[74, 164, 179]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    f = col[None, None] * (0.45 + 0.55 * stripe[..., None])
    o = img.astype(np.float32) * (1 - w[..., None]) + f * w[..., None]
    halo = cv2.GaussianBlur(w, (0, 0), 6)[..., None] * 0.35
    return np.clip(o + col * halo, 0, 255).astype(np.uint8)


def matte(levels=None):
    s = cv2.bilateralFilter(base, 0, 40, 9); s = cv2.GaussianBlur(s, (0, 0), 7)
    l = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
    l = 78 + 0.55 * (l - np.median(l[pm > 0]))                    # flat even light, charcoal above the backdrop
    if levels:
        q = np.round((l - 50) / 14) * 14 + 50; q = cv2.medianBlur(q.astype(np.uint8), 7).astype(np.float32)
        edge = (cv2.morphologyEx(q.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & (pm > 0)
        l = q; l[edge] -= 16
    c = np.stack([l * 1.03, l * 1.0, l * 0.97], -1)               # neutral cool charcoal (BGR)
    return np.clip(base * (1 - pa) + c * pa, 0, 255).astype(np.uint8)


A = fibres(matte()); B = fibres(matte(levels=True))
C = glow.copy(); hm = head.astype(np.float32); hm = cv2.GaussianBlur(hm, (0, 0), 3)[..., None]
gain = (np.median(L[torso]) + 6) / max(1.0, np.median(L[head]))
C = np.clip(C * (1 - hm) + cv2.GaussianBlur(C, (0, 0), 3) * gain * hm, 0, 255).astype(np.uint8)   # head = body tone, no face
for nm, im in [("A", A), ("B", B), ("C", C), ("cur", glow)]:
    cv2.imwrite(SC + f"opt_{nm}.png", im)


def stats(im):
    l = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(cv2.GaussianBlur(l, (0, 0), 1.0), cv2.CV_32F)
    body = (pm > 0) & (gm == 0) & (cv2.erode(pm, np.ones((9, 9), np.uint8)) > 0)
    return {"head_L": round(float(np.median(l[head])), 1), "torso_L": round(float(np.median(l[torso & (gm == 0)])), 1),
            "backdrop_L": round(bgL, 1), "body_texture_lap_std": round(float(lap[body].std()), 2)}


S = {k: stats(v) for k, v in [("current_D7", glow), ("A", A), ("B", B), ("C", C)]}
ys, xs = np.where(pm > 0); S["person_height_frac_of_1920"] = round((ys.max() - ys.min()) / 1920, 3)
mb = (np.abs(E.astype(int) - E[300, 60].astype(int)).max(2) > 18); my, mx = np.where(mb[300:1900] > 0)
S["machine_plus_person_height_frac"] = round((max(my.max() + 300, ys.max()) - min(my.min() + 300, ys.min())) / 1920, 3)
m = subprocess.run([sys.executable, R + "tools/day4_look_metrics.py",
                    f"day4=output/_consistency/d7m/day4_master_t17.5_right.jpg:right:356.4",
                    f"A={SC}opt_A.png:right:290.5", f"B={SC}opt_B.png:right:290.5", f"C={SC}opt_C.png:right:290.5"],
                   capture_output=True, text=True, cwd=R)
try: S["fibre_metrics_vs_day4"] = json.loads(m.stdout[m.stdout.index("{"):])
except Exception: S["fibre_metrics_vs_day4"] = m.stdout[-1500:] + m.stderr[-500:]
json.dump(S, open(OUT + "CHARACTER_CONCEPT_metrics.json", "w"), indent=1)

# ---- sheet
PW, PH = 405, 720
def panel(img, title, sub, crop=None):
    im = img if crop is None else img[crop[1]:crop[3], crop[0]:crop[2]]
    im = cv2.resize(im, (PW, PH) if crop is None else (PW, int(PW * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
    p = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); return p, title, sub
def ref(path, t):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read(); return f
cr = (0, 980, 640, 1780)   # torso close-up crop (same for all options)
day1 = ref(R + "output/day1_v38_master_reel.mp4", 3); day2 = ref(R + "output/day2_v39_master_reel.mp4", 3)
day4 = cv2.imread(R + "output/_consistency/d7m/day4_master_t17.5_right.jpg")
reel = ref(R + "output/day7_still_motion_seated_cable_row.mp4", 9.5)
rows = [[panel(day1, "Day 1 ref", "figure fills the frame"), panel(day2, "Day 2 ref", "close camera, big body"),
         panel(day4, "Day 4 ref (look target)", "fibres / colour target"), panel(reel, "D7 now (rejected)", f"person {S['person_height_frac_of_1920']*100:.0f}% of height")],
        [panel(A, "A  smooth matte charcoal", "code fibres"), panel(B, "B  low-poly facets", "code fibres"),
         panel(C, "C  Day 4 body, plain head", "generator fibres"), panel(glow, "D7 now", "for comparison")],
        [panel(A, "A close-up", "", cr), panel(B, "B close-up", "", cr), panel(C, "C close-up", "", cr), panel(glow, "now close-up", "", cr)]]
GAP, TOP = 14, 92
rh = [PH, PH, rows[2][0][0].height]
sheet = Image.new("RGB", (4 * PW + 5 * GAP, TOP + sum(rh) + 3 * 58 + 4 * GAP + 300), (17, 19, 23)); dr = ImageDraw.Draw(sheet)
dr.text((GAP, 20), "D7 Seated Cable Row: simplified mannequin concept (rough CODE mock-ups on the real frame, 0 generations)", font=f_pretendard(28, "extrabold"), fill=(240, 240, 240))
y = TOP
for r, h in zip(rows, rh):
    for i, (p, t, s) in enumerate(r):
        x = GAP + i * (PW + GAP); dr.text((x, y), t, font=f_pretendard(24, "extrabold"), fill=(120, 230, 160) if t.startswith(("A", "B", "C")) else (230, 230, 230))
        dr.text((x, y + 30), s, font=f_pretendard(17, "medium"), fill=(160, 165, 175)); sheet.paste(p, (x, y + 58))
    y += h + 58 + GAP
notes = [("A  smooth matte (RECOMMENDED)", "+ stablest motion (no surface detail to boil); + code fibres fully in our control and tracked by kp; + head = body tone by construction; + reads cleanly against the white machine.  - must synthesise ALL muscle structure in code (no anatomy cue left)."),
         ("B  low-poly facets", "+ more 3D cue than A; + still low boil.  - facet edges shimmer when the video model re-renders them per frame; - fights the Day 4 fibre look (two textures); - reads 'game asset', less premium."),
         ("C  Day 4 body, plain head", "+ closest to Day 4 today; + generator gives real muscle cues.  - generator muscle detail boils / slides in motion (D5 fibre boil); - head tone needs a fix every frame; - glow sits on generated ridges we don't control."),
         ("Scale (all options)", f"In this side view the full machine is wider than tall, so at full width the person is only ~{S['person_height_frac_of_1920']*100:.0f}% and machine+person ~{S['machine_plus_person_height_frac']*100:.0f}% of the 9:16 height. To get Day 1/2 size with the whole machine, the next plate needs a closer rear-3/4, higher camera (rail foreshortened) at native 1080x1920.")]
for t, s in notes:
    dr.text((GAP, y), t, font=f_pretendard(21, "extrabold"), fill=(120, 230, 160)); 
    words, line, yy = s.split(" "), "", y + 2
    for w_ in words:
        if dr.textlength(line + w_ + " ", font=f_pretendard(16, "medium")) > sheet.width - GAP - 380 - GAP:
            dr.text((GAP + 380, yy), line, font=f_pretendard(16, "medium"), fill=(205, 208, 214)); line, yy = "", yy + 22
        line += w_ + " "
    dr.text((GAP + 380, yy), line, font=f_pretendard(16, "medium"), fill=(205, 208, 214)); y = max(y + 40, yy + 34)
sheet = sheet.crop((0, 0, sheet.width, y + 10))
sheet.save(OUT + "CHARACTER_CONCEPT_sheet.jpg", quality=92)
print(json.dumps(S, indent=1)[:3000])
