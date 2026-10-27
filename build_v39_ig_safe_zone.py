#!/usr/bin/env python3
"""
v39 Instagram Reels Safe-Zone Master (`build_v39_ig_safe_zone.py`)
==================================================================
Fixes the exact iPhone Instagram Reels UI overlap and 19.5:9 horizontal edge cropping
shown in the user's screenshot (`t176u`: "릴스 규격 최적화 안되어있는것 같은데?"):

Exact Pixel Measurements from User's iPhone Screenshot (`1179 × 2556`):
1. Horizontal 19.5:9 Aspect-Fill Crop:
   - iPhone Reels scales the `1080 × 1920` video by `1.202x` vertically (`0..1920` -> `0..2308`),
     cropping `x = 0..50` on the left and `x = 1030..1080` on the right!
   - In `v38`, the bottom HUD started at `bx = 68` (only `18px` from the physical iPhone screen edge!),
     and right-side exterior badges (`LUMBAR LOAD`, `ELBOW PLANE`, `THORACIC LOCK`, `BACK HUNCH`) at `x = 855..965`
     extended to `x = 975..1065`, getting clipped on the right edge!
   - **v39 Fix**:
     * All left-aligned HUD elements (Top Exercise Header, 3-Thumbnail Dock, Bottom-Left HUD) start at **`x = 96`**
       (`56px` of clean studio breathing room inside the iPhone screen's left edge!).
     * All right-side exterior callout badges are clamped to **`x <= 810`** (so their right edge never exceeds `x = 920`,
       leaving `110px+` of safe margin before `x = 1030` and staying completely above the right-side Heart icon `y = 1055`).
2. Top Instagram Reels Header (`+  /  릴스  친구`) Overlap:
   - On iPhone Reels, the top bar (`+` on the left, `릴스 친구` in the center) occupies `y = 165..242` in video coordinates.
   - In `v38`, the Top Exercise Title Header (`WIDE-GRIP LAT PULLDOWN [BACK · LATS]`) sat at `y = 164..214`,
     colliding directly with `+` and `릴스 친구`!
   - **v39 Fix**:
     * Moves the Top Exercise Title Header down to **`y = 258..306`** (`x = 96`) and the 3-Thumbnail Progression Dock
       to **`y = 318..448`** (`x = 96`), placing them 100% cleanly BELOW Instagram's `+ / 릴스 친구` header while staying
       well above the mannequin's head and cable bar!
3. Bottom Instagram Reels Profile (`@gymform.3d 팔로우`) & Right Action Bar Safe Zone:
   - `@gymform.3d 팔로우` starts at `y = 1585` in video coordinates; right-side action icons (`♥ 💬 🔁 ✈️ 🔖`) occupy `x = 915..1030, y = 1055..1700`.
   - **v39 Fix**:
     * Places the Bottom-Left HUD at **`x = 96, y = 1300..1444`**, leaving `140px+` of clean studio space above `@gymform.3d`
       and ending at `x <= 635` (nearly `300px` clear of the right-side action icons!).
"""

import os
import sys
import math
import subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/workspace/daily_info_reels")
import gymform_engine as eng
import build_v28_luxury_ab as v28
import build_v30_pure_graphic_numbers as v30
import build_v35_repeatable_master as v35
import build_v36_clean_master as v36
import build_v36_publish_reels as v36p
import build_v37_minimal_luxury as v37
import build_v38_high_contrast_hud as v38

OUT_DIR = "/workspace/daily_info_reels/output"
TMP_AUDIO = "/workspace/daily_info_reels/_tmp_v38_audio"
USER_SCREENSHOT_PATH = "/cursor/stores/store-14bb1fde-19a5-4770-86aa-e1790c2ff7ea/grok-bot/agents/7c203a21-2233-41e0-92f1-189a01945890/attachments/01M5YD1CAPPE9NE5AFM74HFYXB.png"

W, H = eng.W, eng.H
FPS = 30
DURATION = 20.0
TOTAL_FRAMES = int(FPS * DURATION)

# Safe-zone calibrated fonts
F_EX_TITLE = v30.f_pretendard(24, "extrabold")
F_EX_SUB = v30.f_pretendard(16, "extrabold")

F_TAG_PILL = v30.f_pretendard(19, "extrabold")
F_TAG_CTX = v30.f_pretendard(19, "bold")
F_HERO_PCT = v30.f_barlow(84, extrabold=True)
F_HERO_SYM = v30.f_barlow(44, extrabold=True)
F_SINGLE_LINE = v30.f_pretendard(42, "extrabold")

# Safe-Zone Constants (verified against 1179x2556 iPhone Reels viewport)
SAFE_X_LEFT = 96       # Screen left edge is at x=50; x=96 gives 55px screen margin
SAFE_Y_TOP = 258       # IG "+ / 릴스 친구" ends at y=242; y=258 gives clean clearance below it
SAFE_X_RIGHT_BADGE = 805  # Keeps right-aligned/left-aligned exterior callout text inside x <= 920 (screen right edge is x=1030)


def draw_v39_safezone_top_dock(
    canvas: Image.Image,
    torso_icons: list[tuple[Image.Image, Image.Image]],
    active_idx: int,
    is_wrong: bool,
    exercise_name: str,
    muscle_group: str,
):
    """
    v39 Instagram Reels Safe-Zone Top HUD (`x=96, y=258..446`):
    - Positioned 100% BELOW the Instagram Reels `+ / 릴스 친구` top UI bar (`y=165..242`).
    - Starts at `x=96` so it has generous left margin on 19.5:9 iPhone screens (`x=50..1030` visible).
    - Zero checkmark (`✓`) or cross (`✕`) icons.
    """
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    accent = v30.COL_RED if is_wrong else v30.COL_GREEN
    pill_bg = (218, 44, 38, 245) if is_wrong else (18, 168, 88, 245)

    bar_x = SAFE_X_LEFT
    # --- 1. Top Exercise Title Header (`y=258..304`) ---
    hdr_y = SAFE_Y_TOP
    hdr_h = 46
    tb_ex = od.textbbox((0, 0), exercise_name, font=F_EX_TITLE)
    ex_w = tb_ex[2] - tb_ex[0]
    tb_mg = od.textbbox((0, 0), muscle_group, font=F_EX_SUB)
    mg_w = tb_mg[2] - tb_mg[0]

    pill_w = mg_w + 24
    hdr_w = 22 + 12 + ex_w + 16 + pill_w + 16

    od.rounded_rectangle([bar_x + 2, hdr_y + 4, bar_x + hdr_w + 2, hdr_y + hdr_h + 4], radius=14, fill=(0, 0, 0, 150))
    od.rounded_rectangle(
        [bar_x, hdr_y, bar_x + hdr_w, hdr_y + hdr_h],
        radius=14,
        fill=(14, 17, 23, 232),
        outline=(255, 255, 255, 44),
        width=1,
    )

    dot_cx = bar_x + 20
    dot_cy = hdr_y + hdr_h // 2
    od.ellipse([dot_cx - 8, dot_cy - 8, dot_cx + 8, dot_cy + 8], fill=(*accent, 95))
    od.ellipse([dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5], fill=(*accent, 255))

    ex_x = dot_cx + 15
    od.text((ex_x, hdr_y + 9), exercise_name, font=F_EX_TITLE, fill=(255, 255, 255, 255))

    p_x0 = ex_x + ex_w + 14
    p_y0 = hdr_y + 8
    p_h = hdr_h - 16
    od.rounded_rectangle([p_x0, p_y0, p_x0 + pill_w, p_y0 + p_h], radius=8, fill=pill_bg)
    od.text((p_x0 + 12, p_y0 + 5), muscle_group, font=F_EX_SUB, fill=(255, 255, 255, 255))

    # --- 2. 3-Stage Thumbnail Dock (`y=316..446`) ---
    tw, th = 136, 108
    gap = 10
    pad = 11
    dock_y = hdr_y + hdr_h + 12  # y = 316
    dock_w = 3 * tw + 2 * gap + 2 * pad
    dock_h = th + 2 * pad        # 130px -> ends at y = 446

    od.rounded_rectangle([bar_x + 2, dock_y + 6, bar_x + dock_w + 2, dock_y + dock_h + 6], radius=22, fill=(0, 0, 0, 135))
    od.rounded_rectangle(
        [bar_x, dock_y, bar_x + dock_w, dock_y + dock_h],
        radius=22,
        fill=(16, 19, 25, 222),
        outline=(255, 255, 255, 38),
        width=1,
    )

    for i in range(3):
        tx = bar_x + pad + i * (tw + gap)
        ty = dock_y + pad
        act_im, inact_im = torso_icons[i]
        icon_im = act_im.copy().resize((tw, th), Image.Resampling.LANCZOS) if i == active_idx else inact_im.copy().resize((tw, th), Image.Resampling.LANCZOS)

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], radius=13, fill=255)
        ov.paste(icon_im, (tx, ty), mask)

        if i == active_idx:
            i_col = v30.COL_RED if i < 2 else v30.COL_GREEN
            od.rounded_rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], radius=14, outline=(*i_col, 245), width=2)
            od.rounded_rectangle([tx + tw // 2 - 16, ty + th - 6, tx + tw // 2 + 16, ty + th - 2], radius=2, fill=(*i_col, 255))
        else:
            od.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=13, outline=(255, 255, 255, 22), width=1)

    canvas.alpha_composite(ov)


def draw_v39_safezone_bottom_hud(canvas: Image.Image, sp: dict):
    """
    v39 Instagram Reels Safe-Zone Bottom-Left HUD (`x=96..630, y=1300..1444`):
    - Starts at `bx = 96` (instead of `68`), giving `55px` of clean left margin on iPhone screens (`x=50` crop edge).
    - Ends at `y = 1444`, leaving `145px+` of clean studio space above `@gymform.3d 팔로우` (`y=1590`).
    - Ends horizontally at `x <= 630`, leaving `285px+` clear of the right-side Instagram action icons (`x=915`).
    """
    is_wrong = sp["is_wrong"]
    accent = v30.COL_RED if is_wrong else v30.COL_GREEN
    pill_fill = (224, 44, 38, 250) if is_wrong else (16, 168, 86, 250)
    pct = int(np.clip(sp["pct"], 0, 100))

    tag_pill_str = sp["tag_pill"]
    tag_ctx_str = sp["tag_ctx"]
    cue_1line = sp["cue_1line"]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    bx = SAFE_X_LEFT  # 96px

    # --- 1. Top Full-Width High-Contrast Solid Accent Capsule + Context Label (`y=1300`) ---
    py = 1300
    tb_p = od.textbbox((0, 0), tag_pill_str, font=F_TAG_PILL)
    pw = (tb_p[2] - tb_p[0]) + 26
    ph = 34

    sd.rounded_rectangle([bx, py + 3, bx + pw, py + ph + 3], radius=8, fill=(0, 0, 0, 225))
    od.rounded_rectangle([bx, py, bx + pw, py + ph], radius=8, fill=pill_fill, outline=(255, 255, 255, 55), width=1)
    od.text((bx + 13, py + 6), tag_pill_str, font=F_TAG_PILL, fill=(255, 255, 255, 255))

    ctx_x = bx + pw + 13
    ctx_y = py + 6
    sd.text((ctx_x, ctx_y + 3), tag_ctx_str, font=F_TAG_CTX, fill=(0, 0, 0, 245))
    od.text((ctx_x, ctx_y), tag_ctx_str, font=F_TAG_CTX, fill=(228, 236, 248, 255))

    # --- 2. Main Hero 1-Line Row (`y=1344`) ---
    by = 1344
    num_str = f"{pct}"
    nb = od.textbbox((0, 0), num_str, font=F_HERO_PCT)
    nw = nb[2] - nb[0]

    sd.text((bx, by + 4), num_str, font=F_HERO_PCT, fill=(0, 0, 0, 245))
    od.text((bx, by), num_str, font=F_HERO_PCT, fill=(255, 255, 255, 255))

    sym_x = bx + nw + 3
    sym_y = by + 12
    sd.text((sym_x, sym_y + 3), "%", font=F_HERO_SYM, fill=(0, 0, 0, 240))
    od.text((sym_x, sym_y), "%", font=F_HERO_SYM, fill=(*accent, 255))
    sb = od.textbbox((0, 0), "%", font=F_HERO_SYM)
    sw = sb[2] - sb[0]

    div_x = sym_x + sw + 18
    div_y0 = by + 18
    div_y1 = by + 68
    sd.rounded_rectangle([div_x, div_y0 + 2, div_x + 4, div_y1 + 2], radius=2, fill=(0, 0, 0, 225))
    od.rounded_rectangle([div_x, div_y0, div_x + 4, div_y1], radius=2, fill=(*accent, 255))

    tx = div_x + 18
    ty = by + 19
    tb_c = od.textbbox((0, 0), cue_1line, font=F_SINGLE_LINE)
    cw = tb_c[2] - tb_c[0]

    sd.text((tx, ty + 4), cue_1line, font=F_SINGLE_LINE, fill=(0, 0, 0, 245))
    od.text((tx, ty), cue_1line, font=F_SINGLE_LINE, fill=(255, 255, 255, 255))

    # --- 3. Bottom Sleek Continuous Progress Hairline (`y=1436`) ---
    bar_x0 = bx + 2
    bar_y0 = by + 90
    total_row_w = (tx + cw) - bar_x0
    bar_w = max(360, min(510, total_row_w))
    bar_h = 6
    fill_w = max(14, int(round(bar_w * (pct / 100.0))))

    sd.rounded_rectangle([bar_x0, bar_y0 + 2, bar_x0 + bar_w, bar_y0 + bar_h + 2], radius=3, fill=(0, 0, 0, 215))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], radius=3, fill=(255, 255, 255, 42))
    od.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y0 + bar_h], radius=3, fill=(*accent, 248))

    dot_x = bar_x0 + fill_w
    dot_y = bar_y0 + bar_h * 0.5
    sd.ellipse([dot_x - 8, dot_y - 8, dot_x + 8, dot_y + 8], fill=(*accent, 175))
    od.ellipse([dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5], fill=(255, 255, 255, 255))

    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
    canvas.alpha_composite(ov)


def render_day2_v39_safezone_callouts(
    canvas: Image.Image,
    step_idx: int,
    map_pt: callable,
    el_pt: tuple[float, float],
    is_wrong: bool,
    swing_amount: float = 1.0,
):
    """
    Day 2 exterior callouts with strict horizontal (`x in 135..910`) & vertical (`y in 520..980`) safe-zone clamping
    so no callout ever touches the iPhone left/right crop edges (`x=50, x=1030`) or right-side IG icons (`y>=1055`).
    """
    SS = 2
    rig_layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rig_layer)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    def S(pt):
        return (pt[0] * SS, pt[1] * SS)

    col_main = v30.COL_RED if is_wrong else v30.COL_GREEN
    col_soft = v30.COL_RED_SOFT if is_wrong else v30.COL_GREEN_SOFT
    el_s = S(map_pt(el_pt[0], el_pt[1]))

    if step_idx == 1:
        cur_deg = int(round(18 + 24 * swing_amount))
        cur_lumbar = int(round(58 + 26 * swing_amount))

        c_arc = S(map_pt(470, 940))
        r_a = 172 * SS
        end_ang = 246
        start_ang = int(round(230 - 38 * swing_amount))
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=start_ang, end=end_ang, fill=(*col_main, 248), width=5 * SS)
        ang_t = math.radians(start_ang)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, start_ang - 90, 13 * SS, (*col_soft, 255))

        c_back = S(map_pt(745, 920))
        r_b = 165 * SS
        b_start = 285
        b_end = int(round(302 + 34 * swing_amount))
        rd.arc([c_back[0] - r_b, c_back[1] - r_b, c_back[0] + r_b, c_back[1] + r_b], start=b_start, end=b_end, fill=(*col_main, 240), width=4 * SS)
        ang_b = math.radians(b_end)
        tip_b = (c_back[0] + r_b * math.cos(ang_b), c_back[1] + r_b * math.sin(ang_b))
        v30.draw_arrow_head(rd, tip_b, b_end + 90, 12 * SS, (*col_soft, 255))

        p_b = map_pt(405, 755)
        v30.draw_floating_number_badge(od, sd, (max(240, int(p_b[0])), int(p_b[1])), f"{cur_deg}°", "BACK SWING", col_main, col_soft, align="right")

        # Keep LUMBAR LOAD well above the right-side Heart icon (y=960 instead of 1085) and inside x <= 805!
        p_l = map_pt(810, 955)
        v30.draw_floating_number_badge(od, sd, (min(SAFE_X_RIGHT_BADGE, int(p_l[0])), min(965, int(p_l[1]))), f"{cur_lumbar}%", "LUMBAR LOAD", col_main, col_soft, align="left")

    elif step_idx == 2:
        v36.draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

        c_arc = S(map_pt(315, 795))
        r_a = 128 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=205, end=268, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(268)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 268 + 90, 12 * SS, (*col_soft, 255))

        p_shrug = map_pt(425, 595)
        v30.draw_floating_number_badge(od, sd, (max(240, int(p_shrug[0])), int(p_shrug[1])), "+22°", "TRAP SHRUG", col_main, col_soft, align="right")

        p_bi = map_pt(385, 935)
        v30.draw_floating_number_badge(od, sd, (max(240, int(p_bi[0])), int(p_bi[1])), "68%", "BICEPS STEAL", col_main, col_soft, align="right")

    elif step_idx == 3:
        v36.draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

        c_arc = S(map_pt(435, 835))
        r_a = 148 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=118, end=170, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(118)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 118 - 90, 12 * SS, (*col_soft, 255))

        p_sc = map_pt(395, 745)
        v30.draw_floating_number_badge(od, sd, (max(240, int(p_sc[0])), int(p_sc[1])), "15°", "SCAPULA LOCK", col_main, col_soft, align="right")

        p_pl = map_pt(815, 760)
        v30.draw_floating_number_badge(od, sd, (min(SAFE_X_RIGHT_BADGE, int(p_pl[0])), int(p_pl[1])), "30°", "ELBOW PLANE", col_main, col_soft, align="left")

    else:
        v36.draw_single_elbow_pivot_ring(rd, el_s, 9.5, col_main, SS)

        c_arc = S(map_pt(435, 865))
        r_a = 158 * SS
        rd.arc([c_arc[0] - r_a, c_arc[1] - r_a, c_arc[0] + r_a, c_arc[1] + r_a], start=116, end=172, fill=(*col_main, 245), width=4 * SS)
        ang_t = math.radians(116)
        tip_p = (c_arc[0] + r_a * math.cos(ang_t), c_arc[1] + r_a * math.sin(ang_t))
        v30.draw_arrow_head(rd, tip_p, 116 - 90, 13 * SS, (*col_soft, 255))

        p_dr = map_pt(405, 825)
        v30.draw_floating_number_badge(od, sd, (max(240, int(p_dr[0])), int(p_dr[1])), "100%", "ELBOW TO HIP", col_main, col_soft, align="right")

        p_arch = map_pt(815, 790)
        v30.draw_floating_number_badge(od, sd, (min(SAFE_X_RIGHT_BADGE, int(p_arch[0])), int(p_arch[1])), "15°", "THORACIC LOCK", col_main, col_soft, align="left")

    canvas.alpha_composite(rig_layer.resize((W, H), Image.Resampling.LANCZOS))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
    canvas.alpha_composite(ov)


def build_cached_v39_top_docks(torso_icons, exercise_name: str, muscle_group: str) -> dict:
    cache = {}
    for active_slot, is_wrong in [(0, True), (1, True), (2, False)]:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_v39_safezone_top_dock(layer, torso_icons, active_slot, is_wrong, exercise_name, muscle_group)
        cache[(active_slot, is_wrong)] = layer
    return cache


def simulate_iphone_ig_reels_view(frame_1080x1920_rgb: Image.Image, real_shot_bgr: np.ndarray, old_v38_frame_bgr: np.ndarray) -> Image.Image:
    """
    Overlays the exact iPhone Instagram Reels UI (`+ / 릴스 친구` top header, status bar, right-side action icons,
    bottom `@gymform.3d 팔로우` & nav bar) extracted from the user's real screenshot onto any `1080×1920` frame,
    using the exact `1.20208x` scale and `x_crop = 49.5px` measured from the user's iPhone!
    """
    scale = 1.20208
    M = np.float32([[scale, 0, -49.5 * scale], [0, scale, 0]])
    new_bgr = cv2.cvtColor(np.array(frame_1080x1920_rgb), cv2.COLOR_RGB2BGR)
    new_warped = cv2.warpAffine(new_bgr, M, (1179, 2556), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    old_warped = cv2.warpAffine(old_v38_frame_bgr, M, (1179, 2556), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Extract pure IG UI pixels where real_shot_bgr differs from old_warped (in UI zones: top y<310, right x>980 & y>1200, bottom y>1880)
    ui_mask = np.zeros((2556, 1179), dtype=np.float32)
    diff = np.max(np.abs(real_shot_bgr.astype(np.float32) - old_warped.astype(np.float32)), axis=2)

    # Top status bar + "+ 릴스 친구" UI zone (y=0..305):
    # Wait: in old_warped, our old v38 header was at y=195..260, x=35..650!
    # So for x=35..650, y=195..260, let's only keep bright white pixels of "+" and "릴스 친구" from real_shot_bgr!
    top_zone = np.zeros((2556, 1179), dtype=np.float32)
    top_zone[:185, :] = np.clip((diff[:185, :] - 12.0) / 35.0, 0.0, 1.0)
    # "+" icon on left (x=50..120, y=215..285) and "릴스 친구" in center (x=430..750, y=205..285)
    luma = 0.299 * real_shot_bgr[:, :, 2] + 0.587 * real_shot_bgr[:, :, 1] + 0.114 * real_shot_bgr[:, :, 0]
    plus_mask = np.zeros((2556, 1179), dtype=np.float32)
    plus_mask[220:282, 50:115] = np.clip((luma[220:282, 50:115] - 150.0) / 70.0, 0.0, 1.0)
    reels_tabs_mask = np.zeros((2556, 1179), dtype=np.float32)
    reels_tabs_mask[210:282, 435:745] = np.clip((luma[210:282, 435:745] - 110.0) / 70.0, 0.0, 1.0)

    # Right-side action icons (x=1010..1160, y=1320..2220)
    right_mask = np.zeros((2556, 1179), dtype=np.float32)
    right_mask[1320:2220, 1010:1170] = np.clip((diff[1320:2220, 1010:1170] - 15.0) / 40.0, 0.0, 1.0)

    # Bottom profile + caption + progress bar + bottom tab bar (y=1920..2556)
    bot_mask = np.zeros((2556, 1179), dtype=np.float32)
    bot_mask[1920:2280, :] = np.clip((diff[1920:2280, :] - 12.0) / 35.0, 0.0, 1.0)
    bot_mask[2280:, :] = 1.0  # Bottom black navigation bar is 100% IG UI

    ui_alpha = np.maximum.reduce([top_zone, plus_mask, reels_tabs_mask, right_mask, bot_mask])[:, :, None]
    comp_bgr = new_warped.astype(np.float32) * (1.0 - ui_alpha) + real_shot_bgr.astype(np.float32) * ui_alpha
    comp_rgb = cv2.cvtColor(np.clip(comp_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(comp_rgb)


def build_day2_v39_master_reel():
    print("Building Day 2 (Wide-Grip Lat Pulldown) v39 Instagram Reels Safe-Zone Master...")
    cap_w = cv2.VideoCapture(v36p.DAY2_WRONG_I2V)
    raw_w = []
    while True:
        ret, fr = cap_w.read()
        if not ret:
            break
        raw_w.append(fr)
    cap_w.release()

    cap_c = cv2.VideoCapture(v36p.DAY2_CORRECT_I2V)
    raw_c = []
    while True:
        ret, fr = cap_c.read()
        if not ret:
            break
        raw_c.append(fr)
    cap_c.release()

    raw_w_stroke = raw_w[22:56]
    pool_s1, pool_s2, raw_el_w = [], [], []
    for fr in raw_w_stroke:
        rgb1, el1 = v36p.render_day2_i2v_frame_v36(fr, 1, True)
        rgb2, _   = v36p.render_day2_i2v_frame_v36(fr, 2, True)
        pool_s1.append(rgb1)
        pool_s2.append(rgb2)
        raw_el_w.append(el1)

    el_w_arr = np.array(raw_el_w, dtype=np.float32)
    el_w_x = gaussian_filter1d(el_w_arr[:, 0], sigma=2.0, mode="nearest")
    el_w_y = gaussian_filter1d(el_w_arr[:, 1], sigma=2.0, mode="nearest")
    smooth_el_w = [(float(el_w_x[i]), float(el_w_y[i])) for i in range(len(raw_w_stroke))]

    raw_c_stroke = raw_c[40:121][::-1]
    pool_s3, pool_s4, raw_el_c = [], [], []
    for fr in raw_c_stroke:
        rgb3, el3 = v36p.render_day2_i2v_frame_v36(fr, 3, False)
        rgb4, _   = v36p.render_day2_i2v_frame_v36(fr, 4, False)
        pool_s3.append(rgb3)
        pool_s4.append(rgb4)
        raw_el_c.append(el3)

    el_c_arr = np.array(raw_el_c, dtype=np.float32)
    el_c_x = gaussian_filter1d(el_c_arr[:, 0], sigma=2.2, mode="nearest")
    el_c_y = gaussian_filter1d(el_c_arr[:, 1], sigma=2.2, mode="nearest")
    smooth_el_c = [(float(el_c_x[i]), float(el_c_y[i])) for i in range(len(raw_c_stroke))]

    size = (136, 108)
    crop_box = (460, 620, 880, 1080)
    inact_base = Image.fromarray(pool_s3[55]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    inact_base.alpha_composite(Image.new("RGBA", size, (10, 12, 16, 135)))

    im0 = Image.fromarray(pool_s1[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im1 = Image.fromarray(pool_s2[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    im2 = Image.fromarray(pool_s4[-1]).convert("RGBA").crop(crop_box).resize(size, Image.Resampling.LANCZOS)
    torso_icons = [(im0, inact_base.copy()), (im1, inact_base.copy()), (im2, inact_base.copy())]
    dock_cache = build_cached_v39_top_docks(torso_icons, "WIDE-GRIP LAT PULLDOWN", "BACK · LATS")
    hud_cache = {}

    wav_day2 = os.path.join(TMP_AUDIO, "day2_v38_audio.wav")
    tmp_mp4 = os.path.join(TMP_AUDIO, "_day2_v39_video.mp4")
    final_mp4 = os.path.join(OUT_DIR, "day2_v39_master_reel.mp4")
    vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W, H))

    left_w = 430
    curve_l = (np.linspace(0.65, 1.0, left_w, dtype=np.float32) ** 0.75)[None, :, None]
    scrim_2d = v38.make_bottom_scrim_2d()

    step_static_frames = {}
    cover_pil = None
    f210_rgb = None

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        if t < 5.0:
            u = t / 5.0
            swing_w = v37.back_swing_momentum_wave(u)
            pos_f = swing_w * (len(pool_s1) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s1, pos_f)
            base_rgb = v37.amplify_back_swing_posture(base_rgb, swing_w)
            el_pt = v37.sample_pt_interpolated(smooth_el_w, pos_f)
            step_idx = 1
            is_wrong = True
            active_slot = 0
            cam_zoom = 1.02 + 0.025 * swing_w
            cam_focus, cam_dest = (650.0, 940.0), (665.0, 940.0)
            pct = int(round(74 + 10 * swing_w))
            tag_pill = "LUMBAR OVERLOAD"
            tag_ctx = "MOMENTUM SWING"
            cue_1line = "42° BACK SWING"
        elif t < 10.0:
            u = (t - 5.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s2) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s2, pos_f)
            el_pt = v37.sample_pt_interpolated(smooth_el_w, pos_f)
            step_idx = 2
            is_wrong = True
            active_slot = 1
            swing_w = rep_w
            cam_zoom = 1.04
            cam_focus, cam_dest = (635.0, 900.0), (655.0, 900.0)
            pct = int(round(66 + 8 * rep_w))
            tag_pill = "BICEPS & TRAP STEAL"
            tag_ctx = "ARM PULL ERROR"
            cue_1line = "+22° TRAP SHRUG"
        elif t < 15.0:
            u = (t - 10.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s3) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s3, pos_f)
            el_pt = v37.sample_pt_interpolated(smooth_el_c, pos_f)
            step_idx = 3
            is_wrong = False
            active_slot = 2
            swing_w = rep_w
            cam_zoom = 1.03
            cam_focus, cam_dest = (645.0, 910.0), (660.0, 910.0)
            pct = int(round(88 + 8 * rep_w))
            tag_pill = "UPPER LAT & TERES"
            tag_ctx = "SCAPULA DEPRESSION"
            cue_1line = "15° SCAPULA LOCK"
        else:
            u = (t - 15.0) / 5.0
            rep_w = v37.controlled_single_rep_wave(u)
            pos_f = rep_w * (len(pool_s4) - 1)
            base_rgb = v37.sample_pool_interpolated(pool_s4, pos_f)
            el_pt = v37.sample_pt_interpolated(smooth_el_c, pos_f)
            step_idx = 4
            is_wrong = False
            active_slot = 2
            swing_w = rep_w
            cam_zoom = 1.04
            cam_focus, cam_dest = (650.0, 915.0), (665.0, 915.0)
            pct = int(round(91 + 7 * rep_w))
            tag_pill = "FULL LAT ISOLATION"
            tag_ctx = "4 × 12 REPS  ·  1s HOLD"
            cue_1line = "ELBOWS TO HIPS"

        cam_rgb, map_pt = v28.apply_cinema_camera(base_rgb, zoom=cam_zoom, focus_xy=cam_focus, dest_xy=cam_dest)
        clean_rgb = v38.apply_bottom_left_scrim(cam_rgb, curve_l, scrim_2d)
        im = Image.fromarray(clean_rgb).convert("RGBA")

        render_day2_v39_safezone_callouts(im, step_idx, map_pt, el_pt, is_wrong, swing_amount=swing_w)
        im.alpha_composite(dock_cache[(active_slot, is_wrong)])

        hud_key = (is_wrong, pct, tag_pill, tag_ctx, cue_1line)
        if hud_key not in hud_cache:
            h_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_v39_safezone_bottom_hud(
                h_layer,
                {
                    "is_wrong": is_wrong,
                    "pct": pct,
                    "tag_pill": tag_pill,
                    "tag_ctx": tag_ctx,
                    "cue_1line": cue_1line,
                },
            )
            hud_cache[hud_key] = h_layer
        im.alpha_composite(hud_cache[hud_key])

        rgb_frame = im.convert("RGB")
        if f_idx == 210:
            f210_rgb = rgb_frame.copy()
        if f_idx in (int(1.35 * FPS), int(7.4 * FPS), int(12.4 * FPS), int(17.4 * FPS)):
            step_static_frames[step_idx] = rgb_frame.copy()
        if f_idx == int(17.4 * FPS):
            cover_pil = rgb_frame.copy()

        vw.write(cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR))

    vw.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-threads", "4",
            "-i", tmp_mp4, "-i", wav_day2,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_mp4)

    cover_path = os.path.join(OUT_DIR, "day2_v39_cover.jpg")
    cover_pil.save(cover_path, quality=95)

    # Build Side-by-Side iPhone Reels Safe-Zone Verification Sheet (`v38 Old Overlap` vs `v39 Safe-Zone Optimized`)
    real_shot_bgr = cv2.imread(USER_SCREENSHOT_PATH)
    cap_old = cv2.VideoCapture(os.path.join(OUT_DIR, "day2_v38_master_reel.mp4"))
    cap_old.set(cv2.CAP_PROP_POS_FRAMES, 210)
    _, old_f210_bgr = cap_old.read()
    cap_old.release()

    v38_iphone_im = Image.fromarray(cv2.cvtColor(real_shot_bgr, cv2.COLOR_BGR2RGB))
    v39_iphone_im = simulate_iphone_ig_reels_view(f210_rgb, real_shot_bgr, old_f210_bgr)
    v39_s4_iphone_im = simulate_iphone_ig_reels_view(step_static_frames[4], real_shot_bgr, old_f210_bgr)

    cw, ch = 460, 997
    pad = 24
    comp = Image.new("RGB", (cw * 3 + pad * 4, ch + 110), (10, 12, 16))
    cd = ImageDraw.Draw(comp)
    f_title = v30.f_pretendard(21, "extrabold")
    f_sub = v30.f_pretendard(15, "bold")
    cd.text(
        (24, 18),
        "iPHONE INSTAGRAM REELS SAFE-ZONE VERIFICATION  ·  v38 (상단 '+ 릴스 친구' 겹침 & 좌측 쏠림)  vs  v39 (릴스 세이프존 100% 최적화)",
        font=f_title,
        fill=v30.COL_GREEN,
    )
    cols = [
        (v38_iphone_im, "1. 기존 v38 실제 화면 (상단 '+ 릴스 친구' 겹침 & 좌측 끝 쏠림)", v30.COL_RED),
        (v39_iphone_im, "2. v39 세이프존 적용 (Step 2: 상단 UI 아래 배치 & 좌우 안전 여백 확보)", v30.COL_GREEN),
        (v39_s4_iphone_im, "3. v39 세이프존 적용 (Step 4: 하단 프로필·우측 아이콘 간섭 0%)", v30.COL_GREEN),
    ]
    for i, (pim, lbl, col) in enumerate(cols):
        x = pad + i * (cw + pad)
        cd.text((x + 4, 56), lbl, font=f_sub, fill=col)
        rc = pim.resize((cw, ch), Image.Resampling.LANCZOS)
        ImageDraw.Draw(rc).rectangle([0, 0, cw - 1, ch - 1], outline=col, width=2)
        comp.paste(rc, (x, 84))

    comp_path = os.path.join(OUT_DIR, "v39_iphone_safezone_comparison.jpg")
    comp.save(comp_path, quality=95)
    print("Saved iPhone Safe-Zone Comparison:", comp_path)
    return step_static_frames


if __name__ == "__main__":
    build_day2_v39_master_reel()
