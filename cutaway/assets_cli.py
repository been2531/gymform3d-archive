#!/usr/bin/env python3
"""Asset step for a new joint (4 registered layers: skin, outer muscle, bone/cartilage, macro).

Image generation itself is a GenerateImage tool call made by the agent/routine (no API keys in
code, no xAI, no paid assets).  This CLI writes the exact prompts, prepares the macro crop and
the coordinate grids, and checks that files are where make_reel.py expects them.

  python3 cutaway/assets_cli.py prompts --joint hip     # -> output/cutaway/assets/hip/prompts.json
  # agent: GenerateImage(prompts.skin, 9:16)                  -> save as skin_raw.jpg
  # agent: GenerateImage(prompts.outer, ref=skin_raw, 9:16)   -> outer_raw.jpg   (parallel with bone)
  # agent: GenerateImage(prompts.bone,  ref=skin_raw, 9:16)   -> bone_raw.jpg
  python3 cutaway/assets_cli.py crop --joint hip        # -> macro_src_crop.png + macro_crop.json
  # agent: GenerateImage(prompts.macro, ref=macro_src_crop.png, 1:1) -> macro_raw.jpg
  python3 cutaway/assets_cli.py register --joint hip    # -> registered/*.png + registration_debug.jpg
  python3 cutaway/assets_cli.py grids --joint hip       # -> grid_*.jpg to pick label targets (canvas px)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import register  # noqa: E402

STYLE = ("Premium medical-visualisation lighting, soft rim light, photoreal, high detail, pure black "
         "background, no text, no labels, no arrows.")
KEEP = ("Keep the EXACT same framing, camera angle, pose, scale, limb outline and black background as the "
        "reference image, pixel-aligned.")


def prompts(joint):
    j = json.load(open(os.path.join(HERE, "joints.json")))[joint]
    return {
        "skin": {"aspect_ratio": "9:16", "reference": None,
                 "description": f"Photoreal studio photo of {j['skin']}. Dramatic low-key lighting with a soft rim "
                                f"light, the joint in the upper-middle of the frame, tight crop on the joint only, "
                                f"no face, no clothing, no text, pure black background."},
        "outer": {"aspect_ratio": "9:16", "reference": "skin_raw",
                  "description": f"Edit the reference photo into a medical anatomy cutaway layer. {KEEP} Replace the "
                                 f"skin with the layer just beneath it: {j['outer']}. No skin. {STYLE}"},
        "bone": {"aspect_ratio": "9:16", "reference": "skin_raw",
                 "description": f"Edit the reference photo into the deepest anatomy cutaway layer, textbook-accurate "
                                f"anatomy. {KEEP} Inside a faint translucent ghost of the skin outline show: {j['bone']}. "
                                f"Ivory bone with subtle texture, no muscles. {STYLE}"},
        "macro": {"aspect_ratio": "1:1", "reference": "macro_src_crop",
                  "description": f"Edit the reference image: keep the EXACT same framing, camera and bone positions, "
                                 f"pixel-aligned. Macro close-up: {j['macro']}. Any fluid must look clear and pale "
                                 f"straw-yellow, never blue, metallic or opaque. Shallow depth of field. {STYLE}"},
        "macro_box_frac": j["macro_box_frac"],
    }


def find(adir, stem):
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(adir, stem + ext)
        if os.path.exists(p):
            return p
    raise SystemExit(f"missing {stem}.* in {adir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "crop", "register", "grids"])
    ap.add_argument("--joint", required=True)
    a = ap.parse_args()
    adir = os.path.join(REPO, "output", "cutaway", "assets", a.joint)
    os.makedirs(adir, exist_ok=True)
    if a.cmd == "prompts":
        p = prompts(a.joint)
        json.dump(p, open(os.path.join(adir, "prompts.json"), "w"), indent=1)
        print(os.path.join(adir, "prompts.json"))
    elif a.cmd == "crop":
        box = prompts(a.joint)["macro_box_frac"]
        im = cv2.imread(find(adir, "bone_raw"))
        h, w = im.shape[:2]
        x0, y0, s = int(box[0] * w), int(box[1] * h), int(box[2] * w)
        cv2.imwrite(os.path.join(adir, "macro_src_crop.png"),
                    cv2.resize(im[y0:y0 + s, x0:x0 + s], (1024, 1024), interpolation=cv2.INTER_LANCZOS4))
        json.dump({"box_frac": box}, open(os.path.join(adir, "macro_crop.json"), "w"))
        print(os.path.join(adir, "macro_src_crop.png"))
    elif a.cmd == "register":
        rep = register.run(adir)
        print(json.dumps({k: {x: v[x] for x in ("chamfer_median_px", "chamfer_p90_px", "iou_best")}
                          for k, v in rep["layers"].items()}, indent=1))
    elif a.cmd == "grids":
        for n in ("outer", "bone"):
            im = cv2.imread(os.path.join(adir, "registered", n + ".png"))
            h, w = im.shape[:2]
            st = 100 if w > 800 else 50
            for x in range(0, w, st):
                cv2.line(im, (x, 0), (x, h), (0, 200, 0), 1)
                cv2.putText(im, str(x), (x + 2, 20), 0, 0.6, (0, 255, 255), 1)
            for y in range(0, h, st):
                cv2.line(im, (0, y), (w, y), (0, 200, 0), 1)
                cv2.putText(im, str(y), (2, y - 2), 0, 0.6, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(adir, f"grid_{n}.jpg"), im)
        print(adir)


if __name__ == "__main__":
    main()
