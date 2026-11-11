#!/usr/bin/env python3
"""Compose output/_basemodels/eval/boxer_options_v1.png from boxer_candidate_smoke stills.

  python3 tools/rig3d/build_boxer_options_sheet.py
"""
import json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SM = os.path.join(ROOT, "output/_basemodels/boxer_v3/smoke")
OUT = os.path.join(ROOT, "output/_basemodels/eval/boxer_options_v1.png")
COLS = [
    ("xbot", "X Bot (current)", "Mixamo | alt04 source | no outfit"),
    ("tishkok_handpainted", "Tishkok Handpainted", "CC BY 4.0 | Mixamo 26 b | gloves+trunks"),
    ("kickboxer_mixamo", "OldTURK34 Kickboxer", "CC BY 4.0 | Mixamo 66 b | shorts only"),
    ("muscle_man", "YuxiangCG Muscle Man", "CC BY 4.0 | Mixamo 26 b | no textures"),
    ("chempunk_boxer", "finnvleuten chempunk", "CC BY 4.0 | custom 35-bone rig"),
]
VERDICT = {
    "xbot": "reference",
    "tishkok_handpainted": "PASS smoke (best)",
    "kickboxer_mixamo": "PASS smoke (no gloves)",
    "muscle_man": "FAIL (blob hands/feet, 1,581 shells)",
    "chempunk_boxer": "FAIL (rest pose explodes on import)",
}
W, H = 360, 640
HEAD, ROWLAB, FOOT = 120, 0, 150
try:
    F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    FT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except OSError:
    F = FB = FT = ImageFont.load_default()
sheet = Image.new("RGB", (W * len(COLS), 60 + HEAD + H * 3 + FOOT), (24, 24, 26))
d = ImageDraw.Draw(sheet)
d.text((12, 14), "Boxer options v1 | rest / guard f9 / jab f14 | X Bot alt04 jab retargeted by bone direction", font=FT, fill=(235, 235, 235))
for c, (key, title, sub) in enumerate(COLS):
    x = c * W
    col = (120, 220, 120) if "PASS" in VERDICT[key] else ((230, 110, 100) if "FAIL" in VERDICT[key] else (200, 200, 200))
    d.text((x + 8, 66), title, font=FB, fill=(240, 240, 240))
    d.text((x + 8, 92), sub, font=F, fill=(190, 190, 190))
    d.text((x + 8, 116), VERDICT[key], font=FB, fill=col)
    for r, tag in enumerate(("rest", "guard", "full")):
        p = os.path.join(SM, f"{key}_{tag}.png")
        if os.path.exists(p):
            sheet.paste(Image.open(p).convert("RGB").resize((W, H)), (x, 60 + HEAD + r * H))
        d.text((x + 8, 60 + HEAD + r * H + 6), tag, font=FB, fill=(255, 230, 120))
    js = os.path.join(SM, f"{key}_smoke.json")
    if os.path.exists(js):
        q = json.load(open(js))["qc"]
        st = json.load(open(js)).get("stretch", {}).get("14", {})
        y = 60 + HEAD + 3 * H + 8
        lines = [
            f"fist dev {q['fist_max_dev_m']*100:.1f} cm | elbow-sh {q['elbow_above_shoulder_max_m']*100:+.1f} cm",
            f"guard lead/rear hand-head {q['guard_lead_hand_to_head_rel']:.2f}/{q['guard_rear_hand_to_head_rel']:.2f} H",
            f"foot slide L {q['foot_slide_l_m']:.2f} m | drift {q['root_drift_m']:.2f} m",
            f"edge stretch f14 p99 {st.get('p99','-')} max {st.get('max','-')}",
        ]
        for i, ln in enumerate(lines):
            d.text((x + 8, y + i * 22), ln, font=F, fill=(215, 215, 215))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
sheet.save(OUT)
print(OUT, sheet.size)
