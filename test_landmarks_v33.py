#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v25_fast_preview as v25
import build_v30_pure_graphic_numbers as v30

raw_w45 = v25.read_specific_frame(eng.WRONG_I2V, 45)
raw_c14 = v25.read_specific_frame(eng.CORRECT_I2V, 14)
raw_c42 = v25.read_specific_frame(eng.CORRECT_I2V, 42)
raw_c64 = v25.read_specific_frame(eng.CORRECT_I2V, 64)

ANATOMY_POSE = {
    1: {
        # Step 1: Hunched forward (f=45 of WRONG_I2V)
        "occiput": (595.0, 505.0),
        "spine": [
            (606.0, 522.0),  # C1/C2
            (632.0, 585.0),  # C5
            (668.0, 642.0),  # C7/T1
            (748.0, 682.0),  # T3
            (832.0, 735.0),  # T6 (kyphotic apex)
            (886.0, 800.0),  # T9
            (906.0, 875.0),  # T12
            (908.0, 955.0),  # L3
            (902.0, 1035.0), # L5
            (908.0, 1095.0), # S2 Sacrum
        ],
        "acromion": (786.0, 792.0),
        "sh_glenoid": (794.0, 816.0),
        "el_center": (834.0, 962.0),
        "el_olecranon": (856.0, 958.0),
        "wr_center": (826.0, 1168.0),
        "hand_tip": (818.0, 1215.0),
        "pronation": 0.45,
        # Pelvis & Right (near) Leg
        "iliac_crest": (868.0, 1065.0),
        "hip_R": (862.0, 1148.0),
        "trochanter_R": (892.0, 1162.0),
        "knee_R": (822.0, 1482.0),
        "patella_R": (788.0, 1478.0),
        "ankle_R": (858.0, 1705.0),
        # Left (far/lead) Leg
        "hip_L": (785.0, 1155.0),
        "knee_L": (642.0, 1495.0),
        "patella_L": (610.0, 1490.0),
        "ankle_L": (638.0, 1705.0),
    },
    2: {
        # Step 2: Upright/neutral, 90° forearm stop short (f=14 of CORRECT_I2V)
        "occiput": (690.0, 488.0),
        "spine": [
            (694.0, 512.0),  # C1/C2
            (692.0, 558.0),  # C4
            (708.0, 602.0),  # C7
            (736.0, 642.0),  # T2
            (762.0, 695.0),  # T5
            (772.0, 765.0),  # T8
            (764.0, 845.0),  # T11
            (750.0, 920.0),  # L2
            (740.0, 985.0),  # L5
            (746.0, 1045.0), # S2 Sacrum
        ],
        "acromion": (748.0, 678.0),
        "sh_glenoid": (754.0, 704.0),
        "el_center": (744.0, 892.0),
        "el_olecranon": (766.0, 902.0),
        "wr_center": (542.0, 852.0),
        "hand_tip": (488.0, 840.0),
        "pronation": 0.15,
        # Pelvis & Right (near) Leg
        "iliac_crest": (712.0, 1020.0),
        "hip_R": (708.0, 1105.0),
        "trochanter_R": (738.0, 1122.0),
        "knee_R": (722.0, 1402.0),
        "patella_R": (690.0, 1398.0),
        "ankle_R": (725.0, 1675.0),
        # Left (far/lead) Leg
        "hip_L": (642.0, 1105.0),
        "knee_L": (604.0, 1380.0),
        "patella_L": (576.0, 1376.0),
        "ankle_L": (592.0, 1585.0),
    },
    3: {
        # Step 3: 15° Hinge, 135° Forearm extension (f=42 of CORRECT_I2V)
        "occiput": (692.0, 488.0),
        "spine": [
            (696.0, 512.0),  # C1/C2
            (694.0, 558.0),  # C4
            (710.0, 602.0),  # C7
            (738.0, 642.0),  # T2
            (766.0, 695.0),  # T5
            (776.0, 765.0),  # T8
            (768.0, 845.0),  # T11
            (752.0, 920.0),  # L2
            (742.0, 985.0),  # L5
            (748.0, 1045.0), # S2 Sacrum
        ],
        "acromion": (754.0, 682.0),
        "sh_glenoid": (762.0, 708.0),
        "el_center": (768.0, 934.0),
        "el_olecranon": (792.0, 940.0),
        "wr_center": (678.0, 1128.0),
        "hand_tip": (648.0, 1172.0),
        "pronation": 0.45,
        # Pelvis & Right (near) Leg
        "iliac_crest": (714.0, 1020.0),
        "hip_R": (710.0, 1105.0),
        "trochanter_R": (740.0, 1122.0),
        "knee_R": (723.0, 1402.0),
        "patella_R": (691.0, 1398.0),
        "ankle_R": (725.0, 1675.0),
        # Left (far/lead) Leg
        "hip_L": (644.0, 1105.0),
        "knee_L": (605.0, 1380.0),
        "patella_L": (577.0, 1376.0),
        "ankle_L": (592.0, 1585.0),
    },
    4: {
        # Step 4: 180° Full Lockout + 25° Rope Pronation (f=64 of CORRECT_I2V)
        "occiput": (693.0, 488.0),
        "spine": [
            (697.0, 512.0),  # C1/C2
            (695.0, 558.0),  # C4
            (712.0, 602.0),  # C7
            (742.0, 642.0),  # T2
            (772.0, 695.0),  # T5
            (784.0, 765.0),  # T8
            (774.0, 845.0),  # T11
            (756.0, 920.0),  # L2
            (744.0, 985.0),  # L5
            (750.0, 1045.0), # S2 Sacrum
        ],
        "acromion": (756.0, 680.0),
        "sh_glenoid": (765.0, 706.0),
        "el_center": (812.0, 938.0),
        "el_olecranon": (834.0, 936.0),
        "wr_center": (820.0, 1142.0),
        "hand_tip": (818.0, 1192.0),
        "pronation": 0.90,
        # Pelvis & Right (near) Leg
        "iliac_crest": (715.0, 1020.0),
        "hip_R": (710.0, 1105.0),
        "trochanter_R": (740.0, 1122.0),
        "knee_R": (723.0, 1402.0),
        "patella_R": (691.0, 1398.0),
        "ankle_R": (725.0, 1675.0),
        # Left (far/lead) Leg
        "hip_L": (644.0, 1105.0),
        "knee_L": (605.0, 1380.0),
        "patella_L": (577.0, 1376.0),
        "ankle_L": (592.0, 1585.0),
    },
}

panels = []
for idx, bgr in [(1, raw_w45), (2, raw_c14), (3, raw_c42), (4, raw_c64)]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    tw, th = int(eng.W * 0.84), int(eng.H * 0.84)
    rgb_r = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    canvas, _ = eng.place_in_studio_canvas_clean(rgb_r, {}, scale=0.84, x_shift=34, y_shift=50)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    p = ANATOMY_POSE[idx]

    # Draw spine curve in cyan
    sp = p["spine"]
    for i in range(len(sp) - 1):
        d.line([sp[i], sp[i+1]], fill=(0, 255, 255), width=4)
    for pt in sp:
        d.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(255, 255, 0))

    # Draw Upper Limb in magenta
    d.line([p["sh_glenoid"], p["el_center"], p["wr_center"], p["hand_tip"]], fill=(255, 0, 255), width=4)
    d.ellipse([p["el_olecranon"][0]-5, p["el_olecranon"][1]-5, p["el_olecranon"][0]+5, p["el_olecranon"][1]+5], fill=(255, 120, 0))

    # Draw Right Leg in lime green
    d.line([p["iliac_crest"], p["hip_R"], p["knee_R"], p["ankle_R"]], fill=(50, 255, 50), width=4)
    d.ellipse([p["patella_R"][0]-5, p["patella_R"][1]-5, p["patella_R"][0]+5, p["patella_R"][1]+5], fill=(255, 255, 255))

    # Draw Left Leg in blue-cyan
    d.line([p["hip_L"], p["knee_L"], p["ankle_L"]], fill=(80, 180, 255), width=3)

    crop = im.crop((420, 420, 980, 1740))
    panels.append(crop)

w_p, h_p = panels[0].size
out_im = Image.new("RGB", (w_p * 4, h_p))
for i, cp in enumerate(panels):
    out_im.paste(cp, (i * w_p, 0))
out_im.save("/workspace/daily_info_reels/output/v33_landmark_grid.jpg", quality=95)
print("Saved /workspace/daily_info_reels/output/v33_landmark_grid.jpg")
