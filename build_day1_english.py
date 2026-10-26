#!/usr/bin/env python3
"""
Day 1 Flagship 100% English Clinical Anatomy Reel & Infographic Builder
Combines @demicstory 5 viral mechanics (Macro X-Ray Hook, Top 4-Slot Mini-Map Pill,
3-Color Bio-Luminescence, Circular Neon X/Check Badges, o-o-o Joint Nodes)
with Precision English Clinical HUD Callouts & Degree Protractors.
"""
import os
import math
import time
import wave
import base64
import subprocess
import cv2
import numpy as np
from PIL import Image

W, H = 1080, 1920
FPS = 30
DURATION = 8.0
TOTAL_FRAMES = int(FPS * DURATION)

RAW_PATH = "/workspace/info_concepts_r1/c1_glass_sleep.png"
SHEET_PREVIEW = "/workspace/daily_info_reels/sheet_r1/s1_master_sheet_preview.jpg"
OUT_DIR = "/workspace/daily_info_reels/output"
LAYER_DIR = "/workspace/daily_info_reels/output/layers_day1"
OUT_PNG = os.path.join(OUT_DIR, "clinical_english_day1_sleep.png")
OUT_MP4 = os.path.join(OUT_DIR, "clinical_english_day1_sleep.mp4")


def prepare_clean_50_50_base(raw_path: str) -> np.ndarray:
    """
    Loads the 4K raw image, removes outer white borders and re-frames into a symmetric
    50/50 split (Top half y=0..945, Bottom half y=945..1920) with zero stray green pixels
    in the top half. Returns BGR uint8 image (1920, 1080, 3).
    """
    raw = cv2.imread(raw_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read {raw_path}")
    rh, rw = raw.shape[:2]
    sx = rw / 1080.0
    sy = rh / 1920.0

    # Top half: crop y=236..1181 (945px tall in 1080p space), x=14..1066
    x0, x1 = int(14 * sx), int(1066 * sx)
    ty0, ty1 = int(236 * sy), int(1181 * sy)
    top_h = 945
    bot_h = H - top_h  # 975
    top_crop = cv2.resize(raw[ty0:ty1, x0:x1], (W, top_h), interpolation=cv2.INTER_AREA)

    # Clean/inpaint any stray green pixels in the top half
    hsv_top = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
    stray_green = cv2.inRange(hsv_top, (35, 70, 60), (95, 255, 255))
    if np.count_nonzero(stray_green) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        stray_green = cv2.dilate(stray_green, kernel, iterations=1)
        top_crop = cv2.inpaint(top_crop, stray_green, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Bottom half: raw bottom panel is y=1216..1904 (688px tall in 1080p space)
    by0, by1 = int(1216 * sy), int(1904 * sy)
    bot_H_body = 700
    bot_pad_top = bot_h - bot_H_body  # 275px of clean dark studio headroom for top-left card
    bot_body = cv2.resize(raw[by0:by1, x0:x1], (W, bot_H_body), interpolation=cv2.INTER_AREA)

    bg_strip = cv2.resize(bot_body[16:72, :], (W, bot_pad_top), interpolation=cv2.INTER_LINEAR)
    grad = np.linspace(0.80, 1.0, bot_pad_top)[:, None, None]
    bg_strip = np.clip(bg_strip.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    bot_crop = np.vstack([bg_strip, bot_body])

    for dy in range(-24, 24):
        y = bot_pad_top + dy
        alpha = (dy + 24) / 48.0
        bot_crop[y] = (
            bg_strip[-1].astype(np.float32) * (1.0 - alpha)
            + bot_body[max(0, 16 + dy)].astype(np.float32) * alpha
        ).astype(np.uint8)

    base = np.vstack([top_crop, bot_crop])
    return base


def build_glow_layers(base_bgr: np.ndarray):
    """
    Precompute emissive glow layers for Top (red-orange C4-C7 compression, y=0..945)
    and Bottom (emerald C1-T4 neutral alignment, y=945..1920).
    """
    split_y = 945
    top_roi = base_bgr[:split_y].astype(np.float32)
    r_excess = np.clip(top_roi[:, :, 2] - top_roi[:, :, 0] * 0.72, 0, 255)
    top_glow = np.zeros_like(top_roi)
    top_glow[:, :, 2] = r_excess * 1.25  # Red
    top_glow[:, :, 1] = r_excess * 0.42  # Orange tint
    top_glow[:, :, 0] = r_excess * 0.08
    top_glow = cv2.GaussianBlur(top_glow, (0, 0), sigmaX=20, sigmaY=20)

    bot_roi = base_bgr[split_y:].astype(np.float32)
    g_excess = np.clip(bot_roi[:, :, 1] - bot_roi[:, :, 2] * 0.62, 0, 255)
    bot_glow = np.zeros_like(bot_roi)
    bot_glow[:, :, 1] = g_excess * 1.18  # Emerald green
    bot_glow[:, :, 0] = g_excess * 0.68  # Cyan-teal tint
    bot_glow[:, :, 2] = g_excess * 0.12
    bot_glow = cv2.GaussianBlur(bot_glow, (0, 0), sigmaX=20, sigmaY=20)

    return top_glow, bot_glow


def encode_crop_b64(img_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int, size: int = 128) -> str:
    h, w = img_bgr.shape[:2]
    x0, x1 = max(0, min(w - 1, x0)), max(1, min(w, x1))
    y0, y1 = max(0, min(h - 1, y0)), max(1, min(h, y1))
    crop = cv2.resize(img_bgr[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", crop)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def render_hud_layers_playwright(base_bgr: np.ndarray, layer_dir: str):
    """
    Uses Playwright headless Chromium to render 4 ultra-crisp 1080x1920 transparent SVG/CSS HUD layers:
    - layer_chrome.png
    - layer_top_hud.png
    - layer_bottom_hud.png
    - layer_hook_hud.png
    """
    from playwright.sync_api import sync_playwright

    os.makedirs(layer_dir, exist_ok=True)
    sheet_img = cv2.imread(SHEET_PREVIEW) if os.path.exists(SHEET_PREVIEW) else base_bgr

    # 4 anatomical square crops for the @demicstory top mini-map pill:
    # [01 CERVICAL]: glowing red cervical spine from base_bgr
    thumb_01 = encode_crop_b64(base_bgr, 370, 395, 650, 675)
    # [02 THORACIC]: thoracic ribcage/spine from base_bgr bottom/top
    thumb_02 = encode_crop_b64(base_bgr, 140, 1460, 420, 1740)
    # [03 LUMBAR] & [04 PELVIC]: crops from master sheet or base anatomy
    if sheet_img is not None and sheet_img.shape[0] > 400:
        sh, sw = sheet_img.shape[:2]
        thumb_03 = encode_crop_b64(sheet_img, int(sw * 0.12), int(sh * 0.36), int(sw * 0.38), int(sh * 0.60))
        thumb_04 = encode_crop_b64(sheet_img, int(sw * 0.60), int(sh * 0.44), int(sw * 0.88), int(sh * 0.72))
    else:
        thumb_03 = encode_crop_b64(base_bgr, 80, 460, 340, 720)
        thumb_04 = encode_crop_b64(base_bgr, 220, 1490, 500, 1770)

    common_css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        width: 1080px; height: 1920px;
        background: transparent;
        font-family: 'Inter', 'Liberation Sans', 'DejaVu Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #FFFFFF;
        overflow: hidden;
        -webkit-font-smoothing: antialiased;
    }
    .layer-root {
        position: relative;
        width: 1080px; height: 1920px;
        background: transparent;
    }
    .hud-card {
        position: absolute;
        background: rgba(10, 14, 22, 0.86);
        border-radius: 16px;
        padding: 16px 20px 16px 22px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    }
    .hud-card::before {
        content: '';
        position: absolute;
        left: 0; top: 10px; bottom: 10px;
        width: 6px;
        border-radius: 4px;
    }
    .card-red {
        border: 1.5px solid rgba(255, 59, 48, 0.62);
        box-shadow: 0 10px 28px rgba(0,0,0,0.65), 0 0 22px rgba(255, 59, 48, 0.22);
    }
    .card-red::before { background: #FF3B30; box-shadow: 0 0 12px #FF3B30; }
    .card-orange {
        border: 1.5px solid rgba(255, 107, 44, 0.62);
        box-shadow: 0 10px 28px rgba(0,0,0,0.65), 0 0 22px rgba(255, 107, 44, 0.20);
    }
    .card-orange::before { background: #FF6B2C; box-shadow: 0 0 12px #FF6B2C; }
    .card-emerald {
        border: 1.5px solid rgba(16, 185, 129, 0.65);
        box-shadow: 0 10px 28px rgba(0,0,0,0.65), 0 0 22px rgba(16, 185, 129, 0.22);
    }
    .card-emerald::before { background: #10B981; box-shadow: 0 0 12px #10B981; }
    .card-cyan {
        border: 1.5px solid rgba(52, 211, 153, 0.65);
        box-shadow: 0 10px 28px rgba(0,0,0,0.65), 0 0 22px rgba(52, 211, 153, 0.20);
    }
    .card-cyan::before { background: #34D399; box-shadow: 0 0 12px #34D399; }

    .card-kicker {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-title {
        font-size: 23px;
        font-weight: 900;
        letter-spacing: 0.4px;
        color: #FFFFFF;
        line-height: 1.15;
        margin-bottom: 5px;
    }
    .card-sub {
        font-size: 16.5px;
        font-weight: 600;
        color: rgba(230, 238, 250, 0.88);
        letter-spacing: 0.2px;
        line-height: 1.25;
    }
    """

    # 1. LAYER CHROME: Top Mini-Map Pill, Status Badges (x=980,y=250 & x=980,y=1030), Divider y=945, Bottom Save Bar
    html_chrome = f"""<!DOCTYPE html><html><head><style>{common_css}
    .minimap-pill {{
        position: absolute;
        top: 22px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(12, 16, 24, 0.86);
        border: 1.5px solid rgba(255, 255, 255, 0.18);
        border-radius: 22px;
        padding: 10px 18px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.65), 0 0 24px rgba(255, 77, 42, 0.14);
    }}
    .mm-item {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px;
    }}
    .mm-thumb-wrap {{
        position: relative;
        width: 64px;
        height: 64px;
        border-radius: 14px;
        overflow: hidden;
        border: 1.5px solid rgba(255, 255, 255, 0.18);
        background: #090D14;
    }}
    .mm-item.active .mm-thumb-wrap {{
        border: 2.5px solid #FF4D2A;
        box-shadow: 0 0 16px rgba(255, 77, 42, 0.85), inset 0 0 10px rgba(255, 77, 42, 0.35);
    }}
    .mm-thumb-wrap img {{
        width: 100%; height: 100%; object-fit: cover;
        filter: contrast(1.12) brightness(1.05);
    }}
    .mm-item:not(.active) .mm-thumb-wrap img {{
        opacity: 0.58;
        filter: grayscale(35%) contrast(1.05);
    }}
    .mm-dot {{
        position: absolute;
        top: 5px; right: 5px;
        width: 9px; height: 9px;
        border-radius: 50%;
        background: #FF4D2A;
        box-shadow: 0 0 8px #FF4D2A;
    }}
    .mm-label {{
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.9px;
        color: rgba(200, 212, 230, 0.68);
    }}
    .mm-item.active .mm-label {{
        color: #FF6B2C;
        text-shadow: 0 0 10px rgba(255, 77, 42, 0.55);
    }}
    .status-badge {{
        position: absolute;
        width: 68px; height: 68px;
        transform: translate(-50%, -50%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 3px solid rgba(255, 255, 255, 0.92);
    }}
    .badge-wrong {{
        left: 980px; top: 250px;
        background: radial-gradient(circle at 35% 30%, #FF6259, #FF3B30 65%, #C81E14);
        box-shadow: 0 0 28px rgba(255, 59, 48, 0.92), 0 0 56px rgba(255, 59, 48, 0.45), inset 0 2px 4px rgba(255,255,255,0.4);
    }}
    .badge-right {{
        left: 980px; top: 1030px;
        background: radial-gradient(circle at 35% 30%, #34D399, #10B981 65%, #057A55);
        box-shadow: 0 0 28px rgba(16, 185, 129, 0.92), 0 0 56px rgba(16, 185, 129, 0.45), inset 0 2px 4px rgba(255,255,255,0.4);
    }}
    .divider-line {{
        position: absolute;
        left: 0; top: 945px; width: 1080px; height: 2px;
        background: linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(210,230,255,0.65) 20%, rgba(255,255,255,0.9) 50%, rgba(210,230,255,0.65) 80%, rgba(255,255,255,0.05) 100%);
        box-shadow: 0 0 14px rgba(180, 220, 255, 0.45);
    }}
    .divider-pill {{
        position: absolute;
        left: 50%; top: 945px;
        transform: translate(-50%, -50%);
        background: rgba(10, 14, 22, 0.94);
        border: 1.5px solid rgba(200, 220, 255, 0.35);
        border-radius: 999px;
        padding: 6px 22px;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2.2px;
        color: #E2ECF8;
        box-shadow: 0 4px 16px rgba(0,0,0,0.75);
        white-space: nowrap;
    }}
    .bottom-bar {{
        position: absolute;
        left: 50%; bottom: 22px;
        transform: translateX(-50%);
        background: rgba(10, 14, 22, 0.88);
        border: 1.5px solid rgba(255, 255, 255, 0.20);
        border-radius: 999px;
        padding: 10px 28px;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1.8px;
        color: rgba(240, 246, 255, 0.94);
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.70);
        white-space: nowrap;
    }}
    .bottom-bar .sep {{ color: rgba(255,255,255,0.32); }}
    .bottom-bar .cta {{ color: #34D399; }}
    </style></head><body><div class="layer-root">
        <div class="minimap-pill">
            <div class="mm-item active">
                <div class="mm-thumb-wrap">
                    <img src="{thumb_01}" />
                    <div class="mm-dot"></div>
                </div>
                <div class="mm-label">[01 CERVICAL]</div>
            </div>
            <div class="mm-item">
                <div class="mm-thumb-wrap"><img src="{thumb_02}" /></div>
                <div class="mm-label">[02 THORACIC]</div>
            </div>
            <div class="mm-item">
                <div class="mm-thumb-wrap"><img src="{thumb_03}" /></div>
                <div class="mm-label">[03 LUMBAR]</div>
            </div>
            <div class="mm-item">
                <div class="mm-thumb-wrap"><img src="{thumb_04}" /></div>
                <div class="mm-label">[04 PELVIC]</div>
            </div>
        </div>

        <div class="status-badge badge-wrong">
            <svg width="34" height="34" viewBox="0 0 34 34">
                <path d="M8 8 L26 26 M26 8 L8 26" stroke="#FFFFFF" stroke-width="5.5" stroke-linecap="round"/>
            </svg>
        </div>

        <div class="divider-line"></div>
        <div class="divider-pill">SIDE-SLEEPING PILLOW HEIGHT BIOMECHANICS</div>

        <div class="status-badge badge-right">
            <svg width="36" height="36" viewBox="0 0 36 36">
                <path d="M7 18.5 L15 26.5 L29 10.5" fill="none" stroke="#FFFFFF" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>

        <div class="bottom-bar">
            <span>DAILY BODY ENCYCLOPEDIA &bull; EP.01</span>
            <span class="sep">|</span>
            <span class="cta">&#9733; SAVE FOR LATER</span>
        </div>
    </div></body></html>"""

    # 2. LAYER TOP HUD: Wrong Form (High Pillow)
    # Spine nodes: T4(195,556) - T1(305,556) - C7(395,556) - C5(485,526) - C2(575,496) - Skull(685,459)
    # Card 1 in upper-left negative space: x=36, y=148, w=545, h=124 (well clear of skull at x>600, y>330 and spine at y=526)
    # Card 2 in lower-left negative space: x=36, y=775, w=555, h=124 (below ribcage/shoulder, above divider y=945)
    html_top_hud = f"""<!DOCTYPE html><html><head><style>{common_css}
    .angle-pill {{
        position: absolute;
        left: 618px; top: 504px;
        background: rgba(24, 10, 12, 0.90);
        border: 2px solid #FF4D2A;
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 19px;
        font-weight: 900;
        letter-spacing: 0.6px;
        color: #FF8A70;
        box-shadow: 0 0 20px rgba(255, 59, 48, 0.45);
        white-space: nowrap;
    }}
    .panel-tag-wrong {{
        position: absolute;
        left: 620px; top: 226px;
        background: rgba(255, 59, 48, 0.16);
        border: 1.5px solid rgba(255, 59, 48, 0.65);
        border-radius: 999px;
        padding: 7px 18px;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #FF8075;
    }}
    </style></head><body><div class="layer-root">
        <svg width="1080" height="1920" viewBox="0 0 1080 1920" style="position:absolute;left:0;top:0;">
            <defs>
                <filter id="glowRed" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="6" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <linearGradient id="wedgeRed" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#FF3B30" stop-opacity="0.10"/>
                    <stop offset="100%" stop-color="#FF3B30" stop-opacity="0.38"/>
                </linearGradient>
            </defs>

            <!-- Horizontal 0° Neutral Reference Dashed Line -->
            <line x1="175" y1="556" x2="775" y2="556" stroke="rgba(220,235,255,0.80)" stroke-width="3" stroke-dasharray="10,8"/>
            <text x="785" y="561" fill="rgba(210,228,255,0.80)" font-size="14" font-weight="800" letter-spacing="1">0.0&deg; REF</text>

            <!-- +18.4° Protractor Wedge & Arc around pivot C7(395,556) radius 205 -->
            <!-- At r=205, -18.49 deg endpoint is x = 395 + 205*cos(-18.49°) = 589.4, y = 556 + 205*sin(-18.49°) = 491.0 -->
            <path d="M 395 556 L 600 556 A 205 205 0 0 0 589.4 491.0 Z" fill="url(#wedgeRed)"/>
            <path d="M 600 556 A 205 205 0 0 0 589.4 491.0" fill="none" stroke="#FF5E3A" stroke-width="4.5" filter="url(#glowRed)"/>

            <!-- Degree Protractor Tick Marks along arc r=205..216 -->
            <line x1="600" y1="556" x2="612" y2="556" stroke="#FF8A70" stroke-width="2.5"/>
            <line x1="597.8" y1="534.5" x2="609.7" y2="533.3" stroke="#FF8A70" stroke-width="2"/>
            <line x1="594.4" y1="513.1" x2="606.1" y2="510.6" stroke="#FF8A70" stroke-width="2"/>
            <line x1="589.4" y1="491.0" x2="600.8" y2="487.2" stroke="#FF8A70" stroke-width="2.5"/>

            <!-- Spine o-o-o segments: T4(195,556)->T1(305,556)->C7(395,556)->C5(485,526)->C2(575,496)->Skull(685,459) -->
            <polyline points="195,556 305,556 395,556" fill="none" stroke="rgba(235,245,255,0.9)" stroke-width="5"/>
            <polyline points="395,556 485,526 575,496 685,459" fill="none" stroke="#FF4528" stroke-width="6" filter="url(#glowRed)"/>

            <!-- Glowing Target Reticle on C4-C7 (485, 526) -->
            <circle cx="485" cy="526" r="34" fill="rgba(255,59,48,0.12)" stroke="#FF4D2A" stroke-width="2" stroke-dasharray="6,4" filter="url(#glowRed)"/>
            <circle cx="485" cy="526" r="21" fill="none" stroke="rgba(255,140,115,0.9)" stroke-width="2"/>

            <!-- Pinching Compression Arrows at C5 Foramen (485,526) -->
            <path d="M 467 472 L 478 502" stroke="#FF6B2C" stroke-width="4" stroke-linecap="round"/>
            <polygon points="481,509 470,500 483,495" fill="#FF6B2C"/>
            <path d="M 503 580 L 492 550" stroke="#FF6B2C" stroke-width="4" stroke-linecap="round"/>
            <polygon points="489,543 500,552 487,557" fill="#FF6B2C"/>

            <!-- Smooth Cubic-Bezier Leader Line 1: C4-C7 Reticle (485,505) -> Card 1 bottom-right (410, 272) -->
            <path d="M 485 502 C 485 385, 360 365, 360 272" fill="none" stroke="#FF4D2A" stroke-width="3" filter="url(#glowRed)"/>
            <circle cx="360" cy="272" r="5" fill="#FF4D2A"/>

            <!-- Smooth Cubic-Bezier Leader Line 2: Levator Scapulae / Upper Trap strain zone (415,572) -> Card 2 top-right (360, 775) -->
            <circle cx="415" cy="572" r="14" fill="rgba(255,107,44,0.18)" stroke="#FF6B2C" stroke-width="2"/>
            <circle cx="415" cy="572" r="5" fill="#FF6B2C"/>
            <path d="M 415 586 C 415 675, 340 685, 340 775" fill="none" stroke="#FF6B2C" stroke-width="3" filter="url(#glowRed)"/>
            <circle cx="340" cy="775" r="5" fill="#FF6B2C"/>

            <!-- Vertebral Joint Nodes (o-o-o) -->
            <g>
                <circle cx="195" cy="556" r="7" fill="#F0F6FF" stroke="#0B0F19" stroke-width="2.5"/>
                <text x="182" y="582" fill="#D8E6F8" font-size="16" font-weight="800">T4</text>

                <circle cx="305" cy="556" r="7" fill="#F0F6FF" stroke="#0B0F19" stroke-width="2.5"/>
                <text x="292" y="582" fill="#D8E6F8" font-size="16" font-weight="800">T1</text>

                <circle cx="395" cy="556" r="9" fill="#FF3B30" stroke="#FFFFFF" stroke-width="3"/>
                <text x="382" y="584" fill="#FFB4A4" font-size="17" font-weight="900">C7</text>

                <circle cx="485" cy="526" r="9.5" fill="#FF3B30" stroke="#FFFFFF" stroke-width="3"/>
                <text x="472" y="556" fill="#FFB4A4" font-size="17" font-weight="900">C5</text>

                <circle cx="575" cy="496" r="9" fill="#FF3B30" stroke="#FFFFFF" stroke-width="3"/>
                <text x="562" y="526" fill="#FFB4A4" font-size="17" font-weight="900">C2</text>

                <circle cx="685" cy="459" r="7.5" fill="#FFD2C8" stroke="#FFFFFF" stroke-width="2.5"/>
                <text x="665" y="440" fill="#FFD2C8" font-size="15" font-weight="800">C1 / SKULL</text>
            </g>
        </svg>

        <div class="panel-tag-wrong">HIGH PILLOW &bull; ELEVATED HEAD</div>
        <div class="angle-pill">+18.4&deg; LATERAL FLEXION</div>

        <!-- Callout Card 1: Upper-Left Negative Space -->
        <div class="hud-card card-red" style="left: 36px; top: 144px; width: 548px;">
            <div class="card-kicker" style="color: #FF6E63;">
                <span>&#9679; PATHOLOGY 01</span>
                <span>&bull;</span>
                <span>INTERVERTEBRAL FORAMEN</span>
            </div>
            <div class="card-title">C4&ndash;C7 FORAMINAL STENOSIS</div>
            <div class="card-sub">Nerve Root Compression &amp; Disc Shear (+240% Load)</div>
        </div>

        <!-- Callout Card 2: Lower-Left Negative Space -->
        <div class="hud-card card-orange" style="left: 36px; top: 775px; width: 558px;">
            <div class="card-kicker" style="color: #FF9466;">
                <span>&#9679; PATHOLOGY 02</span>
                <span>&bull;</span>
                <span>MYOFASCIAL TENSION</span>
            </div>
            <div class="card-title">ASYMMETRIC MUSCLE OVERLOAD</div>
            <div class="card-sub">Levator Scapulae &amp; Upper Trapezius Strain</div>
        </div>
    </div></body></html>"""

    # 3. LAYER BOTTOM HUD: Correct Form (Orthopaedic Contour Pillow)
    # Spine nodes along y=1627: T4(195,1627) - T1(305,1627) - C7(395,1627) - C5(485,1627) - C2(575,1627) - Skull(685,1627)
    # Card 1 in upper-left negative space of bottom panel: left=36, top=990, width=558
    # Card 2 in lower-left negative space of bottom panel: left=36, top=1720, width=568
    html_bottom_hud = f"""<!DOCTYPE html><html><head><style>{common_css}
    .neutral-pill {{
        position: absolute;
        left: 615px; top: 1542px;
        background: rgba(8, 24, 18, 0.92);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 8px 18px;
        font-size: 19px;
        font-weight: 900;
        letter-spacing: 0.8px;
        color: #6EE7B7;
        box-shadow: 0 0 22px rgba(16, 185, 129, 0.45);
        white-space: nowrap;
    }}
    .panel-tag-right {{
        position: absolute;
        left: 615px; top: 1006px;
        background: rgba(16, 185, 129, 0.16);
        border: 1.5px solid rgba(16, 185, 129, 0.68);
        border-radius: 999px;
        padding: 7px 18px;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #6EE7B7;
    }}
    </style></head><body><div class="layer-root">
        <svg width="1080" height="1920" viewBox="0 0 1080 1920" style="position:absolute;left:0;top:0;">
            <defs>
                <filter id="glowEmerald" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="6" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>

            <!-- Extended 0.0° Neutral Reference Dashed Line across entire spine -->
            <line x1="145" y1="1627" x2="825" y2="1627" stroke="rgba(167,243,208,0.85)" stroke-width="3" stroke-dasharray="12,8"/>
            <text x="835" y="1632" fill="#6EE7B7" font-size="14" font-weight="800" letter-spacing="1">0.0&deg; AXIS</text>

            <!-- Aligned Spine o-o-o segments along y=1627 -->
            <line x1="195" y1="1627" x2="685" y2="1627" stroke="#10B981" stroke-width="6" filter="url(#glowEmerald)"/>

            <!-- Glowing Target Reticle on C5 Neutral Zone (485, 1627) -->
            <circle cx="485" cy="1627" r="32" fill="rgba(16,185,129,0.12)" stroke="#10B981" stroke-width="2" stroke-dasharray="6,4" filter="url(#glowEmerald)"/>
            <circle cx="485" cy="1627" r="19" fill="none" stroke="#6EE7B7" stroke-width="2"/>

            <!-- Ergonomic Contour Support Upward Vectors under Cervical Curve -->
            <path d="M 485 1692 L 485 1660" stroke="#34D399" stroke-width="4" stroke-linecap="round"/>
            <polygon points="485,1652 478,1664 492,1664" fill="#34D399"/>
            <path d="M 545 1692 L 545 1660" stroke="#34D399" stroke-width="4" stroke-linecap="round"/>
            <polygon points="545,1652 538,1664 552,1664" fill="#34D399"/>

            <!-- Smooth Cubic-Bezier Leader Line 1: C5 Neutral Axis (485, 1595) -> Card 1 bottom-right (360, 1116) -->
            <path d="M 485 1595 C 485 1360, 355 1310, 355 1116" fill="none" stroke="#10B981" stroke-width="3" filter="url(#glowEmerald)"/>
            <circle cx="355" cy="1116" r="5" fill="#10B981"/>

            <!-- Smooth Cubic-Bezier Leader Line 2: T1-C7 Brachial / Tonus zone (395, 1642) -> Card 2 top-right (465, 1732) -->
            <path d="M 395 1642 C 395 1695, 465 1685, 465 1732" fill="none" stroke="#34D399" stroke-width="3" filter="url(#glowEmerald)"/>
            <circle cx="465" cy="1732" r="5" fill="#34D399"/>

            <!-- Vertebral Joint Nodes (o-o-o) along C1-T4 -->
            <g>
                <circle cx="195" cy="1627" r="8" fill="#10B981" stroke="#FFFFFF" stroke-width="3"/>
                <text x="182" y="1655" fill="#A7F3D0" font-size="16" font-weight="800">T4</text>

                <circle cx="305" cy="1627" r="8" fill="#10B981" stroke="#FFFFFF" stroke-width="3"/>
                <text x="292" y="1655" fill="#A7F3D0" font-size="16" font-weight="800">T1</text>

                <circle cx="395" cy="1627" r="9" fill="#10B981" stroke="#FFFFFF" stroke-width="3"/>
                <text x="382" y="1655" fill="#A7F3D0" font-size="17" font-weight="900">C7</text>

                <circle cx="485" cy="1627" r="9.5" fill="#10B981" stroke="#FFFFFF" stroke-width="3"/>
                <text x="472" y="1655" fill="#A7F3D0" font-size="17" font-weight="900">C5</text>

                <circle cx="575" cy="1627" r="9" fill="#10B981" stroke="#FFFFFF" stroke-width="3"/>
                <text x="562" y="1655" fill="#A7F3D0" font-size="17" font-weight="900">C2</text>

                <circle cx="685" cy="1627" r="8" fill="#34D399" stroke="#FFFFFF" stroke-width="3"/>
                <text x="665" y="1605" fill="#A7F3D0" font-size="15" font-weight="800">C1 / SKULL</text>
            </g>
        </svg>

        <div class="panel-tag-right">ORTHOPAEDIC CONTOUR PILLOW</div>
        <div class="neutral-pill">0.0&deg; NEUTRAL AXIS</div>

        <!-- Callout Card 1: Upper-Left of Bottom Panel -->
        <div class="hud-card card-emerald" style="left: 36px; top: 990px; width: 556px;">
            <div class="card-kicker" style="color: #34D399;">
                <span>&#10003; ALIGNMENT 01</span>
                <span>&bull;</span>
                <span>AXIAL DECOMPRESSION</span>
            </div>
            <div class="card-title">0.0&deg; CERVICAL&ndash;THORACIC AXIS</div>
            <div class="card-sub">Zero Intervertebral Disc Shear Stress</div>
        </div>

        <!-- Callout Card 2: Lower-Left of Bottom Panel -->
        <div class="hud-card card-cyan" style="left: 36px; top: 1732px; width: 566px;">
            <div class="card-kicker" style="color: #6EE7B7;">
                <span>&#10003; ALIGNMENT 02</span>
                <span>&bull;</span>
                <span>NEUROMUSCULAR BALANCE</span>
            </div>
            <div class="card-title">BILATERAL TONUS EQUILIBRIUM</div>
            <div class="card-sub">Unobstructed Brachial Plexus &amp; Airway</div>
        </div>
    </div></body></html>"""

    # 4. LAYER HOOK HUD (0.0s - 1.5s @demicstory Macro X-Ray Hook):
    # Minimal high-impact HUD reticle centered around screen center (where C4-C7 is zoomed) & top-left clinical warning pill
    html_hook_hud = f"""<!DOCTYPE html><html><head><style>{common_css}
    .hook-warning-pill {{
        position: absolute;
        left: 42px; top: 52px;
        background: rgba(20, 8, 10, 0.90);
        border: 2px solid #FF3B30;
        border-radius: 16px;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.75), 0 0 28px rgba(255, 59, 48, 0.45);
    }}
    .hook-warning-text {{
        font-size: 20px;
        font-weight: 900;
        letter-spacing: 0.8px;
        color: #FFFFFF;
    }}
    .hook-warning-text .accent {{
        color: #FF6B2C;
    }}
    .macro-tag {{
        position: absolute;
        left: 42px; top: 132px;
        background: rgba(255, 59, 48, 0.20);
        border: 1.5px solid rgba(255, 77, 42, 0.70);
        border-radius: 999px;
        padding: 6px 18px;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1.8px;
        color: #FF9E8A;
    }}
    </style></head><body><div class="layer-root">
        <div class="hook-warning-pill">
            <span style="font-size:26px;color:#FF3B30;">&#9888;</span>
            <span class="hook-warning-text">C4&ndash;C7 CERVICAL COMPRESSION DETECTED &nbsp;|&nbsp; <span class="accent">+18.4&deg; ELEVATION</span></span>
        </div>
        <div class="macro-tag">MACRO X-RAY SCAN &bull; FORAMINAL IMPINGEMENT</div>

        <svg width="1080" height="1920" viewBox="0 0 1080 1920" style="position:absolute;left:0;top:0;">
            <!-- Precision Macro Target Reticle centered at (540, 960) -->
            <circle cx="540" cy="960" r="210" fill="none" stroke="rgba(255, 77, 42, 0.45)" stroke-width="2.5" stroke-dasharray="14,10"/>
            <circle cx="540" cy="960" r="145" fill="none" stroke="rgba(255, 59, 48, 0.85)" stroke-width="3"/>
            <line x1="350" y1="960" x2="415" y2="960" stroke="#FF4D2A" stroke-width="3.5"/>
            <line x1="665" y1="960" x2="730" y2="960" stroke="#FF4D2A" stroke-width="3.5"/>
            <line x1="540" y1="770" x2="540" y2="835" stroke="#FF4D2A" stroke-width="3.5"/>
            <line x1="540" y1="1085" x2="540" y2="1150" stroke="#FF4D2A" stroke-width="3.5"/>
            <!-- Corner Brackets -->
            <path d="M 310 760 L 310 710 L 360 710" fill="none" stroke="#FF6B2C" stroke-width="4"/>
            <path d="M 770 760 L 770 710 L 720 710" fill="none" stroke="#FF6B2C" stroke-width="4"/>
            <path d="M 310 1160 L 310 1210 L 360 1210" fill="none" stroke="#FF6B2C" stroke-width="4"/>
            <path d="M 770 1160 L 770 1210 L 720 1210" fill="none" stroke="#FF6B2C" stroke-width="4"/>
            <text x="540" y="1202" text-anchor="middle" fill="#FF8A70" font-size="18" font-weight="900" letter-spacing="2">TARGET: C4-C7 NERVE ROOT SHEAR</text>
        </svg>
    </div></body></html>"""

    layers = {
        "layer_chrome.png": html_chrome,
        "layer_top_hud.png": html_top_hud,
        "layer_bottom_hud.png": html_bottom_hud,
        "layer_hook_hud.png": html_hook_hud,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path="/usr/bin/google-chrome", args=["--no-sandbox", "--disable-gpu"])
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        for fname, html_content in layers.items():
            page.set_content(html_content, wait_until="load")
            out_path = os.path.join(layer_dir, fname)
            page.screenshot(path=out_path, omit_background=True)
            # Also save in /workspace/daily_info_reels/ as requested
            page.screenshot(path=os.path.join("/workspace/daily_info_reels", fname), omit_background=True)
        browser.close()


def alpha_blend(dst_bgr: np.ndarray, src_rgba: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    if opacity <= 0.005:
        return dst_bgr
    alpha = (src_rgba[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    src_bgr = src_rgba[:, :, :3].astype(np.float32)
    return np.clip(dst_bgr.astype(np.float32) * (1.0 - alpha) + src_bgr * alpha, 0, 255).astype(np.uint8)


def load_rgba_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        raise RuntimeError(f"Failed to load RGBA layer {path}")
    return img


def generate_medical_audio(wav_path: str):
    sr = 44100
    n_samples = int(sr * DURATION)
    t = np.linspace(0, DURATION, n_samples, endpoint=False)
    audio = np.zeros_like(t)

    # Subtle ambient sub-bass drone
    audio += 0.04 * np.sin(2 * np.pi * 65.0 * t)

    # 0.0s - 1.5s: Macro X-Ray Hook scanner lock-on ping
    for pt in [0.12, 0.65, 1.20]:
        mask = (t >= pt) & (t < pt + 0.35)
        dt = t[mask] - pt
        env = np.exp(-dt * 11.0) * np.sin(np.pi * dt / 0.35)
        audio[mask] += 0.16 * env * (np.sin(2 * np.pi * 310.0 * dt) + 0.6 * np.sin(2 * np.pi * 620.0 * dt))

    # 1.7s - 4.2s: Top half red compression pulse
    for pt in [1.85, 2.75, 3.60]:
        mask = (t >= pt) & (t < pt + 0.45)
        dt = t[mask] - pt
        env = np.exp(-dt * 9.5) * np.sin(np.pi * dt / 0.45)
        audio[mask] += 0.14 * env * (np.sin(2 * np.pi * 220.0 * dt) + 0.5 * np.sin(2 * np.pi * 440.0 * dt))

    # 4.3s: Emerald alignment lock-in chime
    chime_t = 4.30
    mask_c = (t >= chime_t) & (t < DURATION)
    dtc = t[mask_c] - chime_t
    env_c = np.exp(-dtc * 2.1) * (1.0 - np.exp(-dtc * 45.0))
    audio[mask_c] += 0.17 * env_c * (
        0.50 * np.sin(2 * np.pi * 523.25 * dtc)
        + 0.35 * np.sin(2 * np.pi * 783.99 * dtc)
        + 0.25 * np.sin(2 * np.pi * 987.77 * dtc)
    )

    fade_len = int(0.15 * sr)
    audio[:fade_len] *= np.linspace(0, 1, fade_len)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    audio_int16 = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def crop_zoom(frame_bgr: np.ndarray, cx: float, cy: float, zoom: float) -> np.ndarray:
    if zoom <= 1.001:
        return frame_bgr
    crop_w = W / zoom
    crop_h = H / zoom
    x0 = max(0.0, min(W - crop_w, cx - crop_w / 2.0))
    y0 = max(0.0, min(H - crop_h, cy - crop_h / 2.0))
    M = np.array([
        [zoom, 0.0, -x0 * zoom],
        [0.0, zoom, -y0 * zoom],
    ], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LAYER_DIR, exist_ok=True)

    print("[1/4] Preparing clean 50/50 glass anatomy base & glow layers...")
    base_bgr = prepare_clean_50_50_base(RAW_PATH)
    top_glow, bot_glow = build_glow_layers(base_bgr)

    print("[2/4] Rendering 4 transparent HUD overlay PNG layers via Playwright Chromium...")
    render_hud_layers_playwright(base_bgr, LAYER_DIR)

    layer_chrome = load_rgba_bgr(os.path.join(LAYER_DIR, "layer_chrome.png"))
    layer_top = load_rgba_bgr(os.path.join(LAYER_DIR, "layer_top_hud.png"))
    layer_bot = load_rgba_bgr(os.path.join(LAYER_DIR, "layer_bottom_hud.png"))
    layer_hook = load_rgba_bgr(os.path.join(LAYER_DIR, "layer_hook_hud.png"))

    print("[3/4] Saving full composite static flagship PNG...")
    split_y = 945
    static_f = base_bgr.astype(np.float32).copy()
    static_f[:split_y] = np.clip(static_f[:split_y] + top_glow * 0.42, 0, 255)
    static_f[split_y:] = np.clip(static_f[split_y:] + bot_glow * 0.42, 0, 255)
    static_bgr = static_f.astype(np.uint8)
    static_bgr = alpha_blend(static_bgr, layer_top, 1.0)
    static_bgr = alpha_blend(static_bgr, layer_bot, 1.0)
    static_bgr = alpha_blend(static_bgr, layer_chrome, 1.0)
    cv2.imwrite(OUT_PNG, static_bgr)
    print(f"      Saved: {OUT_PNG}")

    print("[4/4] Synthesizing 8.0s 30fps MP4 reel with @demicstory 1.5s Macro X-Ray Hook...")
    wav_path = "/tmp/clinical_english_day1_audio.wav"
    generate_medical_audio(wav_path)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-i", wav_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        OUT_MP4,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / FPS
        frame_f = base_bgr.astype(np.float32).copy()

        # Glow pulses & upper-panel dimming
        if t < 1.7:
            # 0.0s - 1.7s: Macro X-Ray Hook on glowing red C4-C7 cervical spine
            top_pulse = 0.55 + 0.45 * math.sin(2 * math.pi * 2.2 * t) ** 2
            bot_pulse = 0.10
            top_dim = 1.0
        elif t < 4.3:
            # 1.7s - 4.3s: Full frame with pulsing red C4-C7 glow & progressive top HUD
            top_pulse = 0.35 + 0.40 * math.sin(2 * math.pi * 1.4 * (t - 1.7)) ** 2
            bot_pulse = 0.12
            top_dim = 1.0
        else:
            # 4.3s - 8.0s: Upper panel slightly dims (85% brightness) while lower panel pulses emerald
            dim_prog = min(1.0, (t - 4.3) / 0.5)
            top_dim = 1.0 - 0.15 * dim_prog  # 0.85 brightness
            top_pulse = 0.15
            bot_pulse = 0.38 + 0.42 * math.sin(2 * math.pi * 1.25 * (t - 4.3)) ** 2

        frame_f[:split_y] = np.clip(frame_f[:split_y] * top_dim + top_glow * top_pulse, 0, 255)
        frame_f[split_y:] = np.clip(frame_f[split_y:] + bot_glow * bot_pulse, 0, 255)
        frame_bgr = frame_f.astype(np.uint8)

        # Camera Zoom for 0.0s - 1.7s (@demicstory Macro X-Ray Hook)
        # Starts at 2.15x zoom tightly centered on C4-C7 (cx=485, cy=526), eases out 1.0s->1.7s
        if t < 1.0:
            zoom = 2.15
        elif t < 1.7:
            u = (t - 1.0) / 0.7
            ease_out = 1.0 - (1.0 - u) ** 3
            zoom = 2.15 - (2.15 - 1.0) * ease_out
        else:
            zoom = 1.0

        if zoom > 1.001:
            frame_bgr = crop_zoom(frame_bgr, cx=485.0, cy=526.0, zoom=zoom)

        # Hook HUD opacity (visible during 0.0s - 1.5s, fades out 1.15s - 1.50s)
        if t < 1.15:
            hook_op = min(1.0, t / 0.20)
        elif t < 1.50:
            hook_op = max(0.0, 1.0 - (t - 1.15) / 0.35)
        else:
            hook_op = 0.0

        # Chrome opacity (fades in smoothly as camera eases out 1.25s - 1.75s)
        chrome_op = max(0.0, min(1.0, (t - 1.25) / 0.45))

        # Top HUD opacity (fades in progressively 1.70s - 2.45s)
        top_hud_op = max(0.0, min(1.0, (t - 1.70) / 0.75))
        if t >= 4.3:
            top_hud_op = 0.88  # keep crisp and readable while upper panel dims slightly

        # Bottom HUD opacity (fades in smoothly 4.30s - 5.10s)
        bot_hud_op = max(0.0, min(1.0, (t - 4.30) / 0.80))

        if hook_op > 0.005:
            frame_bgr = alpha_blend(frame_bgr, layer_hook, hook_op)
        if top_hud_op > 0.005:
            frame_bgr = alpha_blend(frame_bgr, layer_top, top_hud_op)
        if bot_hud_op > 0.005:
            frame_bgr = alpha_blend(frame_bgr, layer_bot, bot_hud_op)
        if chrome_op > 0.005:
            frame_bgr = alpha_blend(frame_bgr, layer_chrome, chrome_op)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        proc.stdin.write(frame_rgb.tobytes())

    proc.stdin.close()
    proc.wait()
    elapsed = time.time() - t0
    print(f"      Saved: {OUT_MP4} in {elapsed:.2f}s total!")


if __name__ == "__main__":
    main()
