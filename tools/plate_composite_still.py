#!/usr/bin/env python3
"""Put a generated still back on the verified empty-machine plate ONCE (still only, not per frame), so the machine
pixels are the plate's: (1) similarity-align the generated image to the plate from machine features (camera-only
transform: scale/shift/tiny rotation, reported), (2) alpha = person (MediaPipe segmentation, dilated) U pixels in
the moving handle/cable band that differ from the plate, feathered, (3) composite.
usage: python3 tools/plate_composite_still.py CONFIG GEN.jpg OUT.png"""
import sys, json, cv2, numpy as np, mediapipe as mp
sys.path.insert(0, "/workspace/daily_info_reels")
from universal_anatomy_rig import get_pose_detector
R = "/workspace/daily_info_reels/"
def run(cfg, gen_path, out_path):
    P = cv2.imread(R + cfg["plate"]); H, W = P.shape[:2]
    G = cv2.imread(gen_path); G = cv2.resize(G, (W, H), interpolation=cv2.INTER_AREA) if G.shape[:2] != (H, W) else G
    pg, gg = cv2.cvtColor(P, 6), cv2.cvtColor(G, 6)
    m = np.zeros_like(pg); a, b, c, d = cfg.get("align_roi", [300, 1280, 0, 330]); m[a:b, c:d] = 255
    orb = cv2.ORB_create(4000); k1, d1 = orb.detectAndCompute(pg, m); k2, d2 = orb.detectAndCompute(gg, None)
    mt = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda x: x.distance)[:400]
    A, inl = cv2.estimateAffinePartial2D(np.float32([k2[x.trainIdx].pt for x in mt]), np.float32([k1[x.queryIdx].pt for x in mt]), method=cv2.RANSAC, ransacReprojThreshold=3)
    Ga = cv2.warpAffine(G, A, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    r = get_pose_detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(Ga, cv2.COLOR_BGR2RGB)))
    seg = (r.segmentation_masks[0].numpy_view()[..., 0] > 0.25).astype(np.uint8) if r.segmentation_masks else np.zeros((H, W), np.uint8)
    seg = cv2.dilate(seg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    y0, y1, x0, x1 = cfg["moving_band"]; diff = np.abs(Ga.astype(int) - P.astype(int)).max(2) > 18
    band = np.zeros((H, W), np.uint8); band[y0:y1, x0:x1] = diff[y0:y1, x0:x1]
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)); band = cv2.dilate(band, np.ones((5, 5), np.uint8))
    alpha = np.clip(cv2.GaussianBlur(np.maximum(seg, band).astype(np.float32), (0, 0), 2.0), 0, 1)[..., None]
    out = (Ga * alpha + P * (1 - alpha)).astype(np.uint8); cv2.imwrite(out_path, out)
    cv2.imwrite(out_path.replace(".png", "_alpha.png"), (alpha[..., 0] * 255).astype(np.uint8))
    info = {"gen": gen_path, "out": out_path, "align_scale": round(float(np.hypot(A[0, 0], A[1, 0])), 4),
            "align_rot_deg": round(float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), 2), "align_shift_px": [round(float(A[0, 2]), 1), round(float(A[1, 2]), 1)],
            "inliers": int(inl.sum()), "alpha_frac": round(float(alpha.mean()), 3)}
    json.dump(info, open(out_path.replace(".png", "_composite.json"), "w"), indent=1); return info
if __name__ == "__main__":
    print(run(json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]))
