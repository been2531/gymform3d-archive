#!/usr/bin/env python3
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = "/workspace/daily_info_reels/output"
CORRECT_JPG = os.path.join(OUT_DIR, "gym_day1_correct_plate.jpg")
AVATAR_PNG = os.path.join(OUT_DIR, "gymform_lab_avatar.png")

# Create a 1080x1080 circular-safe profile avatar from the 3D muscular gym mannequin
im = cv2.imread(CORRECT_JPG)  # 1280x720
# Crop square around the upper torso, head, and glowing triceps/deltoid
h, w = im.shape[:2]
# y=180..700 (520px), x=140..660 (520px)
sq = im[160:700, 140:680]
sq = cv2.resize(sq, (1080, 1080), interpolation=cv2.INTER_LANCZOS4)

# Enhance contrast and add subtle circular neon emerald/cyan rim aura for Instagram profile circle
pil_im = Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)).convert("RGBA")
glow = Image.new("RGBA", (1080, 1080), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse([28, 28, 1052, 1052], outline=(40, 255, 120, 180), width=18)
glow = glow.filter(ImageFilter.GaussianBlur(radius=14))
pil_im.alpha_composite(glow)

ring = Image.new("RGBA", (1080, 1080), (0, 0, 0, 0))
rd = ImageDraw.Draw(ring)
rd.ellipse([34, 34, 1046, 1046], outline=(220, 255, 235, 210), width=5)
pil_im.alpha_composite(ring)

pil_im.convert("RGB").save(AVATAR_PNG, quality=95)
print("Saved avatar:", AVATAR_PNG)
