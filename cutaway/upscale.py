#!/usr/bin/env python3
"""Real-ESRGAN x4plus (BSD-3, ONNX, CPU) super-resolution for library layers.

    python3 cutaway/upscale.py in.jpg out.png --scale 2

Tiled 128 px input with 24 px overlap and feathered blending, then Lanczos down to the
requested scale (x4 then down gives cleaner edges than a native x2).  Used once per layer
when a joint enters the library, so the renderer never upscales a 720 px source ~2x."""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "real-esrgan-x4plus-128.onnx")
T, OV = 128, 24


def sr4(rgb: np.ndarray) -> np.ndarray:
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    h, w = rgb.shape[:2]
    x = rgb.astype(np.float32) / 255.0
    pad = cv2.copyMakeBorder(x, OV, OV + T, OV, OV + T, cv2.BORDER_REFLECT)
    out = np.zeros(((h + 2 * OV + T) * 4, (w + 2 * OV + T) * 4, 3), np.float32)
    wsum = np.zeros(out.shape[:2] + (1,), np.float32)
    ramp = np.minimum(np.arange(T * 4) + 1, np.arange(T * 4)[::-1] + 1).astype(np.float32)
    ramp = np.minimum(ramp / (OV * 4), 1.0)
    win = (ramp[:, None] * ramp[None, :])[..., None]
    step = T - OV
    for y in range(0, h + OV, step):
        for x0 in range(0, w + OV, step):
            tile = pad[y:y + T, x0:x0 + T]
            r = sess.run(None, {"image": tile.transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
            out[y * 4:(y + T) * 4, x0 * 4:(x0 + T) * 4] += r * win
            wsum[y * 4:(y + T) * 4, x0 * 4:(x0 + T) * 4] += win
    out = out / np.maximum(wsum, 1e-6)
    out = out[OV * 4:(OV + h) * 4, OV * 4:(OV + w) * 4]
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def upscale_file(src: str, dst: str, scale: float = 2.0) -> tuple[int, int]:
    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    up = sr4(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = bgr.shape[:2]
    up = cv2.resize(up, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
    return up.shape[1], up.shape[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--scale", type=float, default=2.0)
    a = ap.parse_args()
    print(upscale_file(a.src, a.dst, a.scale))
