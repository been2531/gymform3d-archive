#!/usr/bin/env python3
"""Automated QC for the Blender lat-pulldown pilot."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "studio3d/pilot"
KF = ROOT / "output/pilot/lat/keyframes"
LOOPS = ROOT / "output/pilot/lat/loops"
REPORT = ROOT / "output/pilot/lat/qc_report.json"


def lum(img):
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


def id_masks(idp):
    r, g, b = idp[..., 0], idp[..., 1], idp[..., 2]
    hi, lo = 160, 90
    return {
        "skin": (r > hi) & (g < lo) & (b < lo),
        "machine": (b > hi) & (r < lo) & (g < lo),
        "bar": (r > hi) & (g > hi) & (b < lo),
        "glow": (r > hi) & (g < lo) & (b > hi),
    }


def check_backdrop_flat(rgb, tol=4.0):
    h, w = rgb.shape[:2]
    block = rgb[8:128, 8:128].astype(np.float32)
    span = block.max(axis=(0, 1)) - block.min(axis=(0, 1))
    return float(span.max()) <= tol, {"lum_span": float(span.max())}


def check_separation(rgb, masks):
    L = lum(rgb)
    bg = ~(
        masks.get("skin", np.zeros_like(L, bool))
        | masks.get("machine", np.zeros_like(L, bool))
        | masks.get("bar", np.zeros_like(L, bool))
        | masks.get("glow", np.zeros_like(L, bool))
    )
    if bg.sum() < 100:
        bg = L < 50
    skin_p75 = float(np.percentile(L[masks["skin"]], 75)) if masks["skin"].any() else 0
    bg_med = float(np.median(L[bg]))
    mach_med = float(np.median(L[masks["machine"]])) if masks["machine"].any() else 0
    ok = skin_p75 - bg_med >= 8 and abs(mach_med - bg_med) >= 12
    return ok, {"skin_p75": skin_p75, "bg_med": bg_med, "machine_med": mach_med}


def check_glow(masks, form):
    area = int(masks["glow"].sum())
    ok = area > 800
    return ok, {"glow_px": area, "form": form}


def check_angles(meta_path, std):
    m = json.loads(meta_path.read_text())
    ex = std["exercises"]["wide_grip_lat_pulldown"]
    trunk = m.get("trunk_from_vertical_deg", 0)
    ok = True
    notes = []
    if m["form"] == "wrong" and m["phase"] >= 0.9:
        if trunk < 32:
            ok = False
        notes.append(f"wrong contract trunk {trunk} (want >=32)")
    if m["form"] == "right":
        if trunk > 30:
            ok = False
        notes.append(f"right trunk {trunk} (want <=30)")
    return ok, {"trunk": trunk, "angles": m["angles_3d"], "notes": notes}


def deterministic_rerender(prefix: Path):
    """Two renders of same settings should match within tiny tolerance."""
    cli = ROOT / "studio3d/pilot/blender_cli.sh"
    build = ROOT / "studio3d/pilot/build_scene.py"
    a = prefix.parent / "_det_a"
    b = prefix.parent / "_det_b"
    for out in (a, b):
        subprocess.check_call(
            [
                str(cli),
                str(build),
                "--form",
                "right",
                "--phase",
                "0.5",
                "--camera",
                "front",
                "--out",
                str(out),
                "--scale",
                "0.5",
                "--samples",
                "16",
                "--skip-id",
            ]
        )
    ia = cv2.imread(str(a.with_name(a.name + "_beauty.png")))
    ib = cv2.imread(str(b.with_name(b.name + "_beauty.png")))
    diff = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
    mx = int(diff.max())
    ok = mx <= 1
    return ok, {"max_abs_diff": mx}


def temporal_stability(loop_dir: Path):
    frames = sorted(loop_dir.glob("frame_*_beauty.png"))
    if len(frames) < 4:
        return False, {"reason": "no frames"}
    prev = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    res = []
    for fp in frames[1:]:
        cur = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        res.append(float(np.mean(np.abs(cur - prev))))
        prev = cur
    mean_delta = float(np.mean(res))
    ok = mean_delta < 6.0
    return ok, {"mean_frame_delta_lum": mean_delta, "n": len(frames)}


def check_grip(meta_paths):
    ok_all = True
    info = []
    for mp in [p for p in meta_paths if not p.name.startswith("_det_")]:
        m = json.loads(mp.read_text())
        ok = m.get("grip_ok", False) and m.get("grip_distance_m", 9) < 0.08
        ok_all = ok_all and ok
        info.append({"file": mp.name, "grip_m": m.get("grip_distance_m"), "ok": ok})
    return ok_all, info


def main():
    std = json.loads((ROOT / "configs/motion_standards.json").read_text())
    results = []
    # backdrop / separation on one frame with ID pass re-rendered if missing
    sample = KF / "right_contract_front"
    rgb = cv2.imread(str(sample.with_name(sample.name + "_beauty.png")))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    id_path = sample.with_name(sample.name + "_id.png")
    if not id_path.exists():
        subprocess.check_call(
            [
                str(ROOT / "studio3d/pilot/blender_cli.sh"),
                str(PILOT / "build_scene.py"),
                "--form",
                "right",
                "--phase",
                "1",
                "--camera",
                "front",
                "--out",
                str(sample),
                "--scale",
                "1.0",
                "--samples",
                "16",
            ]
        )
    idp = cv2.imread(str(id_path))
    masks = id_masks(idp)
    for name, fn in (
        ("backdrop_flat", lambda: check_backdrop_flat(rgb)),
        ("separation", lambda: check_separation(rgb, masks)),
        ("glow_area", lambda: check_glow(masks, "right")),
        ("angles", lambda: check_angles(sample.with_name(sample.name + "_meta.json"), std)),
        ("deterministic", lambda: deterministic_rerender(sample)),
        ("temporal", lambda: temporal_stability(LOOPS / "right_front") if (LOOPS / "right_front").exists() else (True, {"skipped": True})),
        ("hand_bar_clearance", lambda: check_grip(list(KF.glob("*_meta.json")))),
    ):
        ok, info = fn()
        results.append({"check": name, "pass": ok, "info": info})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}", info)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))
    return 0 if all(r["pass"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
