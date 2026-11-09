#!/usr/bin/env python3
import os, math, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = "/workspace/daily_info_reels/output"
W, H = 1080, 1920

def place_on_stage(fr_720x1280: np.ndarray, scale: float, y_foot_src: int, y_foot_dst: int, x_shift: int = 0) -> np.ndarray:
    sh, sw = fr_720x1280.shape[:2]
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    resized = cv2.resize(fr_720x1280, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((H, W, 3), (14, 11, 10), dtype=np.uint8)
    x0 = (W - nw) // 2 + x_shift
    y0 = y_foot_dst - int(round(y_foot_src * scale))
    dst_x1, dst_x2 = max(0, x0), min(W, x0 + nw)
    dst_y1, dst_y2 = max(315, y0), min(1275, y0 + nh)
    src_x1, src_x2 = dst_x1 - x0, (dst_x1 - x0) + (dst_x2 - dst_x1)
    src_y1, src_y2 = dst_y1 - y0, (dst_y1 - y0) + (dst_y2 - dst_y1)
    crop = resized[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
    yy = np.arange(dst_y1, dst_y2, dtype=np.float32)[:, None]
    xx = np.arange(dst_x1, dst_x2, dtype=np.float32)[None, :]
    ay_top = np.clip((yy - 315.0) / 50.0, 0.0, 1.0)
    ay_bot = np.clip((1275.0 - yy) / 45.0, 0.0, 1.0)
    ax_left = np.clip((xx - dst_x1) / 50.0, 0.0, 1.0) if dst_x1 > 0 else np.ones_like(xx)
    ax_right = np.clip((dst_x2 - xx) / 50.0, 0.0, 1.0) if dst_x2 < W else np.ones_like(xx)
    alpha = (ay_top * ay_bot * ax_left * ax_right)[..., None]
    bg = canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(crop * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)
    return canvas

# Let's inspect the exact pixel locations of the joints in the 4 base plates when placed on stage!
plates = [
    ("P1_front_wrong", "/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/attachments/1f31291aff1f3b8fcf439621627072b92cf7c13769570767b062d75ddbb2e2a0.jpg", 0.96, 1085, 1225),
    ("P2_front_fix",   "/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/attachments/ce9487fd707364467e0d050840047bba0423208b526080fff73d2fdc1c51ebcf.jpg", 1.04, 1110, 1228),
    ("P3_side_wrong",  "/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/attachments/128766dd47d4a0a4a2e8935c0854a5a0277a3356a21f2b955de56826308d4e11.jpg", 0.94, 1175, 1230),
    ("P4_side_fix",    "/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/attachments/332c6cba31404e0993de09b2881b572d3b301fc7fe7669619a53761f9ed017cd.jpg", 1.02, 1175, 1230),
]

for name, p, scale, yf_src, yf_dst in plates:
    im = cv2.imread(p)
    im = cv2.resize(im, (720, 1280), interpolation=cv2.INTER_AREA)
    st = place_on_stage(im, scale, yf_src, yf_dst, 0)
    b, g, r = st[:,:,0].astype(float), st[:,:,1].astype(float), st[:,:,2].astype(float)
    print(f"=== {name} ===")
    if "front" in name:
        score = np.clip(r - 0.5*(g+b), 0, None) if "wrong" in name else np.clip(g - 0.4*(r+b), 0, None)
        # Left knee glow
        yl, xl = np.where(score[840:1040, 280:530] > 45)
        yr, xr = np.where(score[840:1040, 550:800] > 45)
        print(f"  L_knee_glow=({np.mean(xl)+280:.1f}, {np.mean(yl)+840:.1f}), R_knee_glow=({np.mean(xr)+550:.1f}, {np.mean(yr)+840:.1f})")
    else:
        score = np.clip(r - 0.45*(g+b), 0, None) if "wrong" in name else np.clip(g - 0.35*(r+b), 0, None)
        # Print connected components of glow!
        mask = (score > 45).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 80:
                cx, cy = centroids[i]
                bx, by, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                print(f"  comp {i}: area={area:5d} center=({cx:.1f},{cy:.1f}) box=[x={bx}..{bx+bw}, y={by}..{by+bh}]")
