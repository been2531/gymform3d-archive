#!/usr/bin/env python3
"""
Generates v9 Master Template Preview Sheets for Day 1 & Day 2 + Instagram Reels Safe-Zone Simulator Proof.
Overrides the old top header, separate top-right badge, and low bottom card with:
1) Unified Top Floating Pill (x=96..884, y=246..322)
2) Proximity Floating Glass Scanner HUD (x=96..884, y=1158..1374) attached right below the active anatomy!
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

import build_3scene_ox_preview as d1
import build_day2_ox_preview as d2
import build_v9_master_preview as v9

OUT_DIR = "/workspace/daily_info_reels/output"


def patch_modules_to_v9():
    def d1_v9_header(canvas_rgba, active_idx, is_correct=True):
        v9.draw_v9_unified_top_pill(canvas_rgba, "CABLE ROPE PUSHDOWN", active_idx, is_correct)

    def d2_v9_header(canvas_rgba, active_idx, is_correct=True):
        v9.draw_v9_unified_top_pill(canvas_rgba, "WIDE-GRIP LAT PULLDOWN", active_idx, is_correct)

    def v9_noop_badge(canvas_rgba, is_correct=True, pulse=0.5):
        # Integrated into the unified top pill in v9 so the eye never splits to the top-right corner!
        pass

    def v9_scanner_callout(
        canvas_rgba,
        target_pt,
        card_center,
        is_correct,
        badge_text,
        anatomy_tag,
        title_text,
        metric_label,
        metric_pct,
        prog=1.0,
    ):
        v9.draw_v9_proximity_scanner_hud(
            canvas_rgba=canvas_rgba,
            target_pt=target_pt,
            is_correct=is_correct,
            badge_text=badge_text,
            anatomy_tag=anatomy_tag,
            title_text=title_text,
            metric_label=metric_label,
            metric_pct=metric_pct,
            prog=prog,
        )

    d1.draw_top_header_and_step_bar = d1_v9_header
    d1.draw_ox_status_badge = v9_noop_badge
    d1.draw_anatomical_scanner_callout = v9_scanner_callout

    d2.draw_day2_header = d2_v9_header
    d2.draw_ox_status_badge = v9_noop_badge
    d2.draw_anatomical_scanner_callout = v9_scanner_callout


def build_sheet(frames_wrong, frames_correct, step_titles, out_path):
    scale = 0.5
    tw, th = int(v9.W * scale), int(v9.H * scale)
    pad = 24
    header_h = 68
    sheet_w = pad * 4 + tw * 3
    sheet_h = pad * 3 + (th + header_h) * 2
    sheet = Image.new("RGB", (sheet_w, sheet_h), (8, 11, 16))
    sd = ImageDraw.Draw(sheet)

    for col in range(3):
        x0 = pad + col * (tw + pad)
        # Row 0: WRONG
        y0_w = pad
        sd.rounded_rectangle([x0, y0_w, x0 + tw, y0_w + header_h - 10], radius=12, fill=(48, 16, 18), outline=(255, 58, 48), width=2)
        v9.draw_tracked_text(sd, (x0 + 18, y0_w + 13), f"✕  {step_titles[col]} (WRONG)", v9.FONT_SHEET_TITLE, fill=(255, 115, 105), tracking=1.0)
        im_w = Image.fromarray(cv2.cvtColor(frames_wrong[col], cv2.COLOR_BGR2RGB)).resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(im_w, (x0, y0_w + header_h))

        # Row 1: CORRECT
        y0_c = pad * 2 + th + header_h
        sd.rounded_rectangle([x0, y0_c, x0 + tw, y0_c + header_h - 10], radius=12, fill=(14, 44, 28), outline=(40, 250, 125), width=2)
        v9.draw_tracked_text(sd, (x0 + 18, y0_c + 13), f"✓  {step_titles[col]} (CORRECT)", v9.FONT_SHEET_TITLE, fill=(95, 255, 160), tracking=1.0)
        im_c = Image.fromarray(cv2.cvtColor(frames_correct[col], cv2.COLOR_BGR2RGB)).resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(im_c, (x0, y0_c + header_h))

    sheet.save(out_path, quality=94)
    print(f"Saved: {out_path}")


def main():
    patch_modules_to_v9()

    d1_wrong = [d1.build_scene_frame(i, is_correct=False) for i in range(3)]
    d1_correct = [d1.build_scene_frame(i, is_correct=True) for i in range(3)]
    build_sheet(
        d1_wrong,
        d1_correct,
        ["STEP 1 · TORSO & SCAPULA", "STEP 2 · ELBOW HINGE", "STEP 3 · ROPE SPREAD"],
        os.path.join(OUT_DIR, "v9_day1_preview_sheet.jpg"),
    )

    d2_wrong = [d2.build_day2_scene_frame(i, is_correct=False) for i in range(3)]
    d2_correct = [d2.build_day2_scene_frame(i, is_correct=True) for i in range(3)]
    build_sheet(
        d2_wrong,
        d2_correct,
        ["STEP 1 · SCAPULA SET", "STEP 2 · TORSO & PATH", "STEP 3 · ELBOW DRIVE"],
        os.path.join(OUT_DIR, "v9_day2_preview_sheet.jpg"),
    )

    # Build Instagram Reels UI Safe-Zone Simulator Comparison (4 panels side-by-side)
    p1 = d1_wrong[0]
    p2 = v9.overlay_ig_reels_ui_simulator(d1_wrong[0])
    p3 = d2_correct[2]
    p4 = v9.overlay_ig_reels_ui_simulator(d2_correct[2])

    scale = 0.55
    pw, ph = int(v9.W * scale), int(v9.H * scale)
    pad = 24
    hdr = 68
    comp = Image.new("RGB", (pad * 5 + pw * 4, pad * 2 + ph + hdr), (8, 11, 16))
    cd = ImageDraw.Draw(comp)
    panels = [
        (p1, "1. DAY 1 (v9 CLEAN)", (255, 115, 105), (48, 16, 18), (255, 58, 48)),
        (p2, "2. DAY 1 + IG REELS UI CHECK", (100, 235, 255), (12, 36, 48), (0, 220, 255)),
        (p3, "3. DAY 2 (v9 CLEAN)", (95, 255, 160), (14, 44, 28), (40, 250, 125)),
        (p4, "4. DAY 2 + IG REELS UI CHECK", (100, 235, 255), (12, 36, 48), (0, 220, 255)),
    ]
    for idx, (frm, lbl, tcol, bgcol, bcol) in enumerate(panels):
        x0 = pad + idx * (pw + pad)
        cd.rounded_rectangle([x0, pad, x0 + pw, pad + hdr - 10], radius=12, fill=bgcol, outline=bcol, width=2)
        v9.draw_tracked_text(cd, (x0 + 16, pad + 13), lbl, v9.FONT_SHEET_TITLE, fill=tcol, tracking=0.8)
        im = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)).resize((pw, ph), Image.Resampling.LANCZOS)
        comp.paste(im, (x0, pad + hdr))

    comp_path = os.path.join(OUT_DIR, "v9_ig_safe_zone_comparison.jpg")
    comp.save(comp_path, quality=94)
    print(f"Saved: {comp_path}")


if __name__ == "__main__":
    main()
