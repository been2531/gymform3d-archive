import os
import math
import base64
import wave
import struct
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT_DIR = Path("/workspace/daily_info_reels/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = Path("/workspace/daily_info_reels/frames_tmp")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

def img_to_data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"

KAIRO_HOOK_URI = img_to_data_uri("/workspace/kairo_pack/test_r1/k2_heavybag_hook.png")
KAIRO_AURA_URI = img_to_data_uri("/workspace/kairo_pack/test_r1/k1_gym_aura.png")
KAIRO_MASTER_URI = img_to_data_uri("/workspace/kairo_pack/refs/kairo_master.jpg")

SPLIT_AND_REEL_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  @font-face {
    font-family: 'NotoSansKR';
    src: local('Noto Sans KR'), local('NotoSansKR'), url('file:///home/box/.fonts/NotoSansKR[wght].ttf');
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1080px;
    height: 1920px;
    background: #070A10;
    color: #F4F7FB;
    font-family: 'Noto Sans KR', 'NanumGothic', sans-serif;
    overflow: hidden;
    position: relative;
  }
  /* Dark medical studio grid */
  .bg-grid {
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(0, 210, 255, 0.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0, 210, 255, 0.045) 1px, transparent 1px),
      radial-gradient(circle at 50% 38%, rgba(0, 210, 255, 0.10), transparent 65%);
    background-size: 40px 40px, 40px 40px, 100% 100%;
    z-index: 0;
  }
  .scanline {
    position: absolute;
    left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, transparent, rgba(0, 245, 160, 0.5), rgba(0, 210, 255, 0.8), transparent);
    box-shadow: 0 0 24px rgba(0, 210, 255, 0.7);
    z-index: 5;
    top: 400px;
    opacity: 0.65;
  }

  /* Header HUD */
  .header-hud {
    position: absolute;
    top: 56px;
    left: 48px;
    right: 48px;
    z-index: 10;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .manual-badge {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: rgba(14, 22, 36, 0.92);
    border: 1.5px solid rgba(0, 245, 160, 0.55);
    padding: 10px 22px;
    border-radius: 999px;
    box-shadow: 0 0 24px rgba(0, 245, 160, 0.18);
  }
  .pulse-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #00F5A0;
    box-shadow: 0 0 12px #00F5A0;
  }
  .manual-code {
    font-size: 23px;
    font-weight: 900;
    letter-spacing: 1.5px;
    color: #00F5A0;
  }
  .manual-sub {
    font-size: 21px;
    font-weight: 700;
    color: #D6E4F0;
    border-left: 1.5px solid rgba(255,255,255,0.2);
    padding-left: 12px;
  }
  .track-pill {
    font-size: 20px;
    font-weight: 800;
    color: #00D2FF;
    background: rgba(0, 210, 255, 0.12);
    border: 1px solid rgba(0, 210, 255, 0.4);
    padding: 10px 20px;
    border-radius: 12px;
    letter-spacing: 0.5px;
  }
  .title-box {
    background: linear-gradient(90deg, rgba(14, 21, 34, 0.95), rgba(14, 21, 34, 0.6));
    border-left: 6px solid #00F5A0;
    padding: 18px 26px;
    border-radius: 0 16px 16px 0;
  }
  .main-title {
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.18;
    color: #FFFFFF;
  }
  .main-title .hl-red { color: #FF3B5C; text-shadow: 0 0 18px rgba(255,59,92,0.45); }
  .main-title .hl-mint { color: #00F5A0; text-shadow: 0 0 18px rgba(0,245,160,0.45); }
  .subtitle-metrics {
    margin-top: 8px;
    font-size: 25px;
    font-weight: 700;
    color: #94A8C4;
    display: flex;
    align-items: center;
    gap: 18px;
  }

  /* Split Comparison Stage */
  .stage-wrap {
    position: absolute;
    top: 300px;
    left: 44px;
    right: 44px;
    height: 1120px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    z-index: 6;
  }
  .panel {
    position: relative;
    border-radius: 28px;
    overflow: hidden;
    background: radial-gradient(circle at 50% 42%, #131D31 0%, #090D16 100%);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 28px;
  }
  .panel-bad {
    border: 2.5px solid rgba(255, 46, 76, 0.65);
    box-shadow: 0 0 40px rgba(255, 46, 76, 0.15), inset 0 0 60px rgba(255, 46, 76, 0.08);
  }
  .panel-good {
    border: 2.5px solid rgba(0, 245, 160, 0.75);
    box-shadow: 0 0 45px rgba(0, 245, 160, 0.18), inset 0 0 60px rgba(0, 245, 160, 0.08);
  }
  .status-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    z-index: 5;
  }
  .status-tag {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    border-radius: 14px;
    font-size: 28px;
    font-weight: 900;
  }
  .tag-bad {
    background: rgba(255, 46, 76, 0.18);
    border: 2px solid #FF2E4C;
    color: #FF4D6A;
  }
  .tag-good {
    background: rgba(0, 245, 160, 0.16);
    border: 2px solid #00F5A0;
    color: #00F5A0;
  }
  .load-readout {
    text-align: right;
  }
  .load-num {
    font-size: 54px;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -1px;
  }
  .num-bad { color: #FF3B5C; text-shadow: 0 0 20px rgba(255, 59, 92, 0.6); }
  .num-good { color: #00F5A0; text-shadow: 0 0 20px rgba(0, 245, 160, 0.6); }
  .load-label {
    font-size: 20px;
    font-weight: 700;
    color: #9FB2CC;
    margin-top: 4px;
  }

  .svg-container {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2;
  }

  .muscle-callout-list {
    position: relative;
    z-index: 5;
    display: flex;
    flex-direction: column;
    gap: 12px;
    background: rgba(8, 12, 20, 0.86);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 18px 20px;
    backdrop-filter: blur(8px);
  }
  .m-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 22px;
    font-weight: 700;
  }
  .m-name { color: #E2ECF7; }
  .m-state-bad { color: #FF5C77; font-weight: 800; }
  .m-state-good { color: #00F5A0; font-weight: 800; }

  /* Bottom 3-Step Action Protocol Bar */
  .protocol-bar {
    position: absolute;
    bottom: 54px;
    left: 44px;
    right: 44px;
    height: 410px;
    background: linear-gradient(180deg, rgba(13, 20, 34, 0.96), rgba(8, 12, 22, 0.98));
    border: 2px solid rgba(0, 210, 255, 0.4);
    border-radius: 28px;
    padding: 26px 32px;
    z-index: 10;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 16px 50px rgba(0,0,0,0.65);
  }
  .proto-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1.5px solid rgba(255,255,255,0.1);
    padding-bottom: 14px;
  }
  .proto-title {
    font-size: 28px;
    font-weight: 900;
    color: #00D2FF;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .save-cta {
    font-size: 22px;
    font-weight: 800;
    color: #070A10;
    background: linear-gradient(90deg, #00F5A0, #00D2FF);
    padding: 8px 20px;
    border-radius: 999px;
  }
  .steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-top: 6px;
  }
  .step-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1.5px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 18px 20px;
    transition: all 0.2s ease;
  }
  .step-card.active {
    background: rgba(0, 245, 160, 0.12);
    border-color: #00F5A0;
    box-shadow: 0 0 24px rgba(0, 245, 160, 0.22);
  }
  .step-num {
    font-size: 20px;
    font-weight: 900;
    color: #00F5A0;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }
  .step-head {
    font-size: 26px;
    font-weight: 900;
    color: #FFFFFF;
    margin-bottom: 6px;
  }
  .step-desc {
    font-size: 20px;
    font-weight: 600;
    color: #A6B9D2;
    line-height: 1.35;
  }
  .progress-track {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
  }
  .progress-fill {
    height: 100%;
    width: 100%;
    background: linear-gradient(90deg, #FF2E4C 0%, #00D2FF 45%, #00F5A0 100%);
    border-radius: 999px;
  }
</style>
</head>
<body>
  <div class="bg-grid"></div>
  <div class="scanline" id="scanline"></div>

  <!-- Top Header -->
  <div class="header-hud">
    <div class="top-bar">
      <div class="manual-badge">
        <div class="pulse-dot" id="pulse-dot"></div>
        <span class="manual-code">BODY MANUAL #001</span>
        <span class="manual-sub">언젠가 쓸모있는 신체 도감</span>
      </div>
      <div class="track-pill">TRACK 01 · 통증·재활 도감</div>
    </div>
    <div class="title-box">
      <div class="main-title">거북목 10초 해부학 리셋 : <span class="hl-red">27kg 과부하</span> → <span class="hl-mint">5kg 정렬</span></div>
      <div class="subtitle-metrics">
        <span>● 경추 C1–C7 분절 압력 비교</span>
        <span>● 상부승모근 억제 &amp; 심부경추굴곡근 활성화</span>
      </div>
    </div>
  </div>

  <!-- Main Split Comparison Stage -->
  <div class="stage-wrap">
    <!-- Left Panel: ❌ 45 deg Overload -->
    <div class="panel panel-bad" id="panel-bad">
      <div class="status-header">
        <div class="status-tag tag-bad">❌ 45° 전방 돌출</div>
        <div class="load-readout">
          <div class="load-num num-bad" id="bad-kg">27.2 kg</div>
          <div class="load-label">C5–C7 디스크 압축 하중</div>
        </div>
      </div>

      <div class="svg-container">
        <svg width="460" height="760" viewBox="0 0 460 760" fill="none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="glowRed" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="8" result="blur"/>
              <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
            <linearGradient id="bodySilBad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#1E2B45" stop-opacity="0.88"/>
              <stop offset="100%" stop-color="#0F1726" stop-opacity="0.95"/>
            </linearGradient>
          </defs>

          <!-- Plumb reference line -->
          <line x1="175" y1="70" x2="175" y2="690" stroke="rgba(255,255,255,0.22)" stroke-width="2.5" stroke-dasharray="8 8"/>
          <text x="165" y="60" fill="#8FA5C2" font-size="16" font-weight="700" text-anchor="end">중력 수직축 (0°)</text>

          <!-- Forward tilted axis line (45 deg) -->
          <line x1="175" y1="490" x2="345" y2="180" stroke="#FF2E4C" stroke-width="3.5" stroke-dasharray="10 6" filter="url(#glowRed)"/>

          <!-- Angle Arc 45 deg -->
          <path d="M 175 330 A 160 160 0 0 1 255 352" stroke="#FF7A00" stroke-width="5" fill="rgba(255, 46, 76, 0.18)"/>
          <text x="232" y="310" fill="#FF9E2C" font-size="30" font-weight="900">45°</text>

          <!-- Anatomical Head & Torso Silhouette (Forward Head Posture) -->
          <path d="M 85 680 L 95 490 Q 110 410 185 375 L 255 250 Q 260 130 335 135 Q 405 140 412 215 Q 418 265 395 305 L 372 330 Q 345 348 310 342 L 265 425 Q 290 485 280 680 Z"
                fill="url(#bodySilBad)" stroke="#384D70" stroke-width="2.5"/>

          <!-- Overloaded Upper Trapezius & Levator Scapulae (Red Glowing Muscle Fibers) -->
          <path d="M 245 240 Q 175 340 120 465 Q 165 475 215 430 Q 245 340 265 260 Z"
                fill="rgba(255, 46, 76, 0.48)" stroke="#FF2E4C" stroke-width="3" filter="url(#glowRed)"/>
          <!-- Muscle striations -->
          <path d="M 240 260 Q 185 350 140 450 M 250 275 Q 198 360 165 448 M 255 295 Q 212 375 190 438"
                stroke="#FF8FA0" stroke-width="2" opacity="0.85"/>

          <!-- Overactive SCM (Sternocleidomastoid) -->
          <path d="M 285 255 Q 260 350 238 440" stroke="#FF5500" stroke-width="12" stroke-linecap="round" opacity="0.75" filter="url(#glowRed)"/>

          <!-- Cervical Spine Vertebrae C1-C7 (Hyper-extended / Forward Shear) -->
          <g filter="url(#glowRed)">
            <rect x="292" y="220" width="26" height="15" rx="5" transform="rotate(28 292 220)" fill="#FF3B5C"/>
            <rect x="278" y="246" width="28" height="16" rx="5" transform="rotate(32 278 246)" fill="#FF3B5C"/>
            <rect x="262" y="274" width="29" height="16" rx="5" transform="rotate(35 262 274)" fill="#FF3B5C"/>
            <rect x="245" y="302" width="30" height="17" rx="5" transform="rotate(37 245 302)" fill="#FF2E4C"/>
            <rect x="226" y="330" width="32" height="18" rx="5" transform="rotate(38 226 330)" fill="#FF1E3A"/>
            <rect x="206" y="360" width="34" height="18" rx="5" transform="rotate(34 206 360)" fill="#FF1E3A"/>
            <rect x="188" y="392" width="36" height="19" rx="5" transform="rotate(25 188 392)" fill="#FF002B"/>
          </g>

          <!-- C5-C7 Disc Compression Warning Rings -->
          <circle cx="205" cy="395" r="28" stroke="#FF2E4C" stroke-width="3" fill="rgba(255,46,76,0.2)"/>
          <circle cx="205" cy="395" r="44" stroke="#FF2E4C" stroke-width="1.5" stroke-dasharray="6 4"/>
          <text x="80" y="388" fill="#FF4D6A" font-size="20" font-weight="900">C5–C7 압박</text>

          <!-- Downward Gravity Overload Vector -->
          <path d="M 365 225 L 365 365" stroke="#FF2E4C" stroke-width="6" stroke-linecap="round"/>
          <polygon points="365,382 351,356 379,356" fill="#FF2E4C"/>
          <text x="350" y="415" fill="#FF4D6A" font-size="22" font-weight="900" text-anchor="middle">하중 ×5.4배</text>
        </svg>
      </div>

      <div class="muscle-callout-list">
        <div class="m-item">
          <span class="m-name">상부승모근 · 견갑거근</span>
          <span class="m-state-bad">▲ 과긴장 · 섬유 경직</span>
        </div>
        <div class="m-item">
          <span class="m-name">경추 C5–C7 디스크</span>
          <span class="m-state-bad">▲ 전단 압력 27.2kg</span>
        </div>
      </div>
    </div>

    <!-- Right Panel: ⭕ 0 deg Biomechanical Alignment (Animated in video mode) -->
    <div class="panel panel-good" id="panel-good">
      <div class="status-header">
        <div class="status-tag tag-good" id="good-tag">⭕ 0° 해부학 정렬</div>
        <div class="load-readout">
          <div class="load-num num-good" id="good-kg">5.0 kg</div>
          <div class="load-label">정상 경추 수직 하중</div>
        </div>
      </div>

      <div class="svg-container">
        <svg width="460" height="760" viewBox="0 0 460 760" fill="none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="glowMint" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="8" result="blur"/>
              <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
            <linearGradient id="bodySilGood" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#18313E" stop-opacity="0.9"/>
              <stop offset="100%" stop-color="#0D1924" stop-opacity="0.95"/>
            </linearGradient>
          </defs>

          <!-- Vertical Plumb Alignment Line (Glowing Cyan/Mint) -->
          <line x1="215" y1="65" x2="215" y2="690" stroke="#00F5A0" stroke-width="3.5" stroke-dasharray="10 6" filter="url(#glowMint)"/>
          <text x="230" y="92" fill="#00F5A0" font-size="20" font-weight="900">귀걸이선 = 어깨중심 수직 일치 (0°)</text>

          <!-- Animated Group for Head/Spine Correction -->
          <g id="good-posture-group">
            <!-- Aligned Head & Torso Silhouette -->
            <path d="M 115 680 L 125 470 Q 135 395 185 360 L 192 235 Q 185 115 258 115 Q 328 115 335 190 Q 340 245 312 282 L 285 302 Q 260 315 235 310 L 245 415 Q 285 475 280 680 Z"
                  fill="url(#bodySilGood)" stroke="#00D2FF" stroke-width="2.5"/>

            <!-- Activated Deep Cervical Flexors (Mint Glowing Band on Anterior Neck) -->
            <path d="M 225 225 Q 228 295 222 365" stroke="#00F5A0" stroke-width="14" stroke-linecap="round" opacity="0.8" filter="url(#glowMint)"/>

            <!-- Activated Lower Trapezius & Scapular Depression (Cyan/Mint Highlight) -->
            <path d="M 185 375 Q 155 455 145 555 Q 190 545 215 475 Z"
                  fill="rgba(0, 245, 160, 0.42)" stroke="#00F5A0" stroke-width="3" filter="url(#glowMint)"/>

            <!-- Aligned Cervical Spine C1-C7 (Natural Lordotic Curve, Mint/Cyan) -->
            <g filter="url(#glowMint)">
              <rect x="202" y="198" width="26" height="15" rx="5" transform="rotate(-4 202 198)" fill="#00F5A0"/>
              <rect x="198" y="225" width="28" height="16" rx="5" transform="rotate(-6 198 225)" fill="#00F5A0"/>
              <rect x="195" y="253" width="29" height="16" rx="5" transform="rotate(-5 195 253)" fill="#00F5A0"/>
              <rect x="194" y="281" width="30" height="17" rx="5" transform="rotate(-2 194 281)" fill="#00D2FF"/>
              <rect x="195" y="310" width="32" height="18" rx="5" transform="rotate(3 195 310)" fill="#00D2FF"/>
              <rect x="198" y="340" width="34" height="18" rx="5" transform="rotate(7 198 340)" fill="#00F5A0"/>
              <rect x="202" y="372" width="36" height="19" rx="5" transform="rotate(10 202 372)" fill="#00F5A0"/>
            </g>

            <!-- Chin Tuck Vector Arrow (Horizontal Posterior Translation) -->
            <path d="M 385 255 L 320 255" stroke="#00F5A0" stroke-width="6" stroke-linecap="round" filter="url(#glowMint)"/>
            <polygon points="305,255 328,241 328,269" fill="#00F5A0"/>
            <text x="335" y="232" fill="#00F5A0" font-size="21" font-weight="900">① 턱 수평 당김</text>

            <!-- Scapular Depression Arrow -->
            <path d="M 175 435 L 175 520" stroke="#00D2FF" stroke-width="6" stroke-linecap="round" filter="url(#glowMint)"/>
            <polygon points="175,536 161,512 189,512" fill="#00D2FF"/>
            <text x="62" y="505" fill="#00D2FF" font-size="20" font-weight="900">② 견갑 하강</text>
          </g>

          <!-- Ear-to-Shoulder Alignment Nodes -->
          <circle cx="215" cy="195" r="10" fill="#00F5A0" filter="url(#glowMint)"/>
          <circle cx="215" cy="410" r="10" fill="#00F5A0" filter="url(#glowMint)"/>
        </svg>
      </div>

      <div class="muscle-callout-list">
        <div class="m-item">
          <span class="m-name">심부경추굴곡근 · 하부승모근</span>
          <span class="m-state-good">● 코어 정렬 · 활성 ON</span>
        </div>
        <div class="m-item">
          <span class="m-name">경추 C1–C7 전만 커브</span>
          <span class="m-state-good">● 압력 81% 즉시 감압</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom 3-Step Action Checklist -->
  <div class="protocol-bar">
    <div class="proto-top">
      <div class="proto-title">
        <span>⚡ 10초 신경근 리셋 프로토콜 (3-STEP CHECKLIST)</span>
      </div>
      <div class="save-cta" id="save-cta">📌 저장해두고 거북목 통증 시 즉시 실행</div>
    </div>

    <div class="steps-grid">
      <div class="step-card active" id="step-1">
        <div class="step-num">STEP 01 · CHIN GLIDE</div>
        <div class="step-head">턱 끝 15° 수평 후인</div>
        <div class="step-desc">고개를 숙이지 않고 뒤통수를 벽 쪽으로 수평 이동</div>
      </div>
      <div class="step-card active" id="step-2">
        <div class="step-num">STEP 02 · SCAPULA LOCK</div>
        <div class="step-head">날개뼈 하강 고정</div>
        <div class="step-desc">어깨를 귀에서 멀리 내리며 하부승모근 스위치 ON</div>
      </div>
      <div class="step-card active" id="step-3">
        <div class="step-num">STEP 03 · 10s ISOMETRIC</div>
        <div class="step-head">심부굴곡근 10초 유지</div>
        <div class="step-desc">목 앞쪽 깊은 코어 수축감 유지하며 호흡 3회</div>
      </div>
    </div>

    <div class="progress-track">
      <div class="progress-fill" id="progress-fill"></div>
    </div>
  </div>

  <script>
    window.setFrameProgress = function(t) {
      // t in [0.0, 1.0] across 8 seconds
      const scan = document.getElementById('scanline');
      scan.style.top = (260 + ((t * 2.2) % 1.0) * 1160) + 'px';

      const pFill = document.getElementById('progress-fill');
      pFill.style.width = Math.min(100, Math.max(5, t * 100)) + '%';

      // Animate right panel from 45 deg -> 0 deg between t=0.20 and t=0.65
      let corr = 1.0;
      if (t < 0.20) {
        corr = 0.0;
      } else if (t < 0.65) {
        const u = (t - 0.20) / 0.45;
        corr = u * u * (3 - 2 * u); // smoothstep
      } else {
        corr = 1.0;
      }

      const angleDeg = Math.round(45 * (1 - corr));
      const kgVal = (27.2 - (22.2 * corr)).toFixed(1);

      const goodTag = document.getElementById('good-tag');
      const goodKg = document.getElementById('good-kg');
      const grp = document.getElementById('good-posture-group');

      goodTag.textContent = (angleDeg <= 5 ? '⭕ ' : '🔄 ') + angleDeg + '° 해부학 정렬';
      goodKg.textContent = kgVal + ' kg';

      // Slight biomechanical translation/rotation of the posture group during correction
      const shiftX = (1 - corr) * 42;
      const rotDeg = (1 - corr) * 14;
      grp.setAttribute('transform', `translate(${shiftX}, 0) rotate(${rotDeg} 215 410)`);

      // Step cards sequential highlight
      const s1 = document.getElementById('step-1');
      const s2 = document.getElementById('step-2');
      const s3 = document.getElementById('step-3');
      s1.className = 'step-card' + (t >= 0.15 ? ' active' : '');
      s2.className = 'step-card' + (t >= 0.42 ? ' active' : '');
      s3.className = 'step-card' + (t >= 0.68 ? ' active' : '');
    };
  </script>
</body>
</html>
"""

FOUR_TRACKS_HTML = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  @font-face {{
    font-family: 'NotoSansKR';
    src: local('Noto Sans KR'), local('NotoSansKR'), url('file:///home/box/.fonts/NotoSansKR[wght].ttf');
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1080px;
    height: 1350px;
    background: #06090F;
    color: #F4F7FB;
    font-family: 'Noto Sans KR', 'NanumGothic', sans-serif;
    overflow: hidden;
    padding: 36px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background-image:
      linear-gradient(rgba(0, 210, 255, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0, 210, 255, 0.04) 1px, transparent 1px);
    background-size: 36px 36px;
  }}
  .top-banner {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(14, 21, 34, 0.94);
    border: 1.5px solid rgba(0, 245, 160, 0.45);
    border-radius: 20px;
    padding: 20px 28px;
  }}
  .b-left {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .b-tag {{
    font-size: 18px;
    font-weight: 900;
    color: #00F5A0;
    letter-spacing: 1.2px;
  }}
  .b-title {{
    font-size: 34px;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: -0.5px;
  }}
  .b-right {{
    text-align: right;
    font-size: 18px;
    font-weight: 700;
    color: #9BB0C9;
    line-height: 1.4;
  }}
  .grid-2x2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 22px;
    height: 1050px;
  }}
  .track-card {{
    background: radial-gradient(circle at 50% 35%, #142038 0%, #0A0F1A 100%);
    border: 2px solid rgba(0, 210, 255, 0.35);
    border-radius: 24px;
    padding: 22px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
  }}
  .tc-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 4;
  }}
  .tc-code {{
    font-size: 17px;
    font-weight: 900;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(0, 245, 160, 0.15);
    border: 1.5px solid #00F5A0;
    color: #00F5A0;
  }}
  .tc-code.combat {{
    background: rgba(255, 59, 92, 0.18);
    border-color: #FF3B5C;
    color: #FF5C77;
  }}
  .tc-code.neuro {{
    background: rgba(0, 210, 255, 0.18);
    border-color: #00D2FF;
    color: #00D2FF;
  }}
  .tc-code.posture {{
    background: rgba(255, 170, 0, 0.18);
    border-color: #FFAA00;
    color: #FFBE3D;
  }}
  .tc-num {{
    font-size: 17px;
    font-weight: 800;
    color: #8DA2C0;
  }}
  .tc-title {{
    margin-top: 10px;
    font-size: 27px;
    font-weight: 900;
    color: #FFFFFF;
    z-index: 4;
    line-height: 1.22;
  }}
  .tc-visual {{
    position: relative;
    height: 295px;
    border-radius: 16px;
    background: rgba(6, 10, 18, 0.75);
    border: 1px solid rgba(255,255,255,0.1);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 10px 0;
  }}
  .tc-visual img.bg-photo {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: 0.58;
    filter: contrast(1.15) saturate(1.1);
  }}
  .tc-visual svg {{
    position: relative;
    z-index: 3;
    width: 100%;
    height: 100%;
  }}
  .tc-foot {{
    background: rgba(8, 12, 22, 0.9);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 18px;
    font-weight: 700;
    z-index: 4;
  }}
  .bottom-strip {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(14, 21, 34, 0.9);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 14px 24px;
    font-size: 18px;
    font-weight: 700;
    color: #B4C6DF;
  }}
</style>
</head>
<body>
  <div class="top-banner">
    <div class="b-left">
      <div class="b-tag">DAILY BODY MANUAL · 4 CANDIDATE TRACKS</div>
      <div class="b-title">@demicstory 벤치마크 — 4대 후보 트랙 비주얼 도감</div>
    </div>
    <div class="b-right">
      <div>무음 1.5초 직관 이해 (❌ vs ⭕)</div>
      <div style="color:#00F5A0;">해부학 하이라이트 + 각도·하중 HUD</div>
    </div>
  </div>

  <div class="grid-2x2">
    <!-- Track 1: Pain & Rehab -->
    <div class="track-card">
      <div>
        <div class="tc-head">
          <span class="tc-code">TRACK 01 · 통증·재활 스트레칭</span>
          <span class="tc-num">MANUAL #001</span>
        </div>
        <div class="tc-title">거북목 10초 리셋 (45°→0° 감압)</div>
      </div>
      <div class="tc-visual">
        <svg viewBox="0 0 440 290" fill="none">
          <line x1="220" y1="15" x2="220" y2="275" stroke="rgba(255,255,255,0.15)" stroke-width="2" stroke-dasharray="6 6"/>
          <!-- Left Bad -->
          <text x="35" y="42" fill="#FF4D6A" font-size="22" font-weight="900">❌ 45° (27kg)</text>
          <path d="M 75 250 Q 95 160 155 95" stroke="#FF2E4C" stroke-width="14" stroke-linecap="round"/>
          <circle cx="165" cy="75" r="28" fill="rgba(255,46,76,0.25)" stroke="#FF2E4C" stroke-width="3"/>
          <text x="45" y="215" fill="#FF8FA0" font-size="18" font-weight="800">C5-C7 과부하</text>
          <!-- Right Good -->
          <text x="255" y="42" fill="#00F5A0" font-size="22" font-weight="900">⭕ 0° (5kg)</text>
          <path d="M 335 250 Q 332 165 335 95" stroke="#00F5A0" stroke-width="14" stroke-linecap="round"/>
          <circle cx="335" cy="70" r="28" fill="rgba(0,245,160,0.22)" stroke="#00F5A0" stroke-width="3"/>
          <path d="M 395 75 L 368 75" stroke="#00D2FF" stroke-width="5" stroke-linecap="round"/>
          <text x="275" y="215" fill="#00F5A0" font-size="18" font-weight="800">심부굴곡근 활성</text>
        </svg>
      </div>
      <div class="tc-foot">
        <span style="color:#FF4D6A;">❌ 상부승모근 경직</span>
        <span style="color:#00F5A0;">⭕ 경추 전만 81% 감압</span>
      </div>
    </div>

    <!-- Track 2: UFC / Boxing 1-Frame Anatomy with KAIRO -->
    <div class="track-card">
      <div>
        <div class="tc-head">
          <span class="tc-code combat">TRACK 02 · 격투 1프레임 해부</span>
          <span class="tc-num">KAIRO × HUD</span>
        </div>
        <div class="tc-title">리드훅 카운터 : 어깨 90° 잠금 &amp; 코어 회전</div>
      </div>
      <div class="tc-visual">
        <img class="bg-photo" src="{KAIRO_HOOK_URI}" alt="KAIRO Hook"/>
        <svg viewBox="0 0 440 290" fill="none">
          <!-- Biomechanical Kinetic Chain Overlay on KAIRO -->
          <path d="M 140 165 L 225 145 L 295 165" stroke="#00F5A0" stroke-width="6" stroke-linecap="round"/>
          <circle cx="225" cy="145" r="10" fill="#00F5A0"/>
          <path d="M 190 155 A 45 45 0 0 1 255 155" stroke="#00D2FF" stroke-width="4" fill="rgba(0,210,255,0.2)"/>
          <text x="195" y="125" fill="#00F5A0" font-size="22" font-weight="900">90° 수평 잠금</text>
          <!-- Torque Arc at Hip/Core -->
          <path d="M 155 245 Q 235 275 315 235" stroke="#FF3B5C" stroke-width="6" stroke-linecap="round"/>
          <polygon points="325,230 302,230 314,250" fill="#FF3B5C"/>
          <text x="220" y="278" fill="#FF8FA0" font-size="18" font-weight="900" text-anchor="middle">지면반력 → 광배근·복사근 토크 420N</text>
        </svg>
      </div>
      <div class="tc-foot">
        <span style="color:#FF5C77;">❌ 팔만 휘두르는 스윙</span>
        <span style="color:#00F5A0;">⭕ 견관절 90° 프레임 고정</span>
      </div>
    </div>

    <!-- Track 3: Daily Body Bio-Hacking -->
    <div class="track-card">
      <div>
        <div class="tc-head">
          <span class="tc-code neuro">TRACK 03 · 일상 신체 해킹</span>
          <span class="tc-num">3초 신경 스위치</span>
        </div>
        <div class="tc-title">막힌 코 3초 개통 : 상악동·영향혈 자율신경 스위치</div>
      </div>
      <div class="tc-visual">
        <svg viewBox="0 0 440 290" fill="none">
          <!-- Stylized Cranial / Sinus Nerve HUD -->
          <ellipse cx="220" cy="145" rx="95" ry="115" fill="rgba(0,210,255,0.08)" stroke="#00D2FF" stroke-width="2.5"/>
          <!-- Nasal passage & maxillary sinus -->
          <path d="M 220 75 L 202 155 L 238 155 Z" fill="rgba(0,245,160,0.22)" stroke="#00F5A0" stroke-width="2.5"/>
          <!-- Acupressure Trigger Nodes (LI20 + Glabella) -->
          <circle cx="182" cy="160" r="14" fill="rgba(255,46,76,0.35)" stroke="#FF2E4C" stroke-width="3"/>
          <circle cx="258" cy="160" r="14" fill="rgba(255,46,76,0.35)" stroke="#FF2E4C" stroke-width="3"/>
          <circle cx="220" cy="95" r="12" fill="rgba(0,245,160,0.35)" stroke="#00F5A0" stroke-width="3"/>
          <!-- Glowing Nerve Pathways -->
          <path d="M 120 85 Q 175 110 182 160 M 320 85 Q 265 110 258 160" stroke="#00D2FF" stroke-width="3.5" stroke-dasharray="6 4"/>
          <text x="32" y="75" fill="#00D2FF" font-size="19" font-weight="900">① 미간(인당) 3초 지압</text>
          <text x="242" y="245" fill="#FF5C77" font-size="19" font-weight="900">② 콧망울 옆(영향혈) 확장</text>
        </svg>
      </div>
      <div class="tc-foot">
        <span style="color:#00D2FF;">● 삼차신경 반사 자극</span>
        <span style="color:#00F5A0;">⭕ 비강 혈관 수축 · 즉시 개통</span>
      </div>
    </div>

    <!-- Track 4: Posture & Jawline Correction with KAIRO -->
    <div class="track-card">
      <div>
        <div class="tc-head">
          <span class="tc-code posture">TRACK 04 · 체형·인상 교정</span>
          <span class="tc-num">턱선 · 어깨선</span>
        </div>
        <div class="tc-title">숨겨진 턱선 복구 : 설골근 &amp; 혀 루프 밀착 ❌ vs ⭕</div>
      </div>
      <div class="tc-visual">
        <img class="bg-photo" src="{KAIRO_MASTER_URI}" alt="KAIRO Jawline"/>
        <svg viewBox="0 0 440 290" fill="none">
          <!-- Left Bad Submental droop vs Right Sharp Mandibular Angle -->
          <path d="M 75 195 Q 155 235 215 190" stroke="#FF2E4C" stroke-width="5" stroke-dasharray="8 5"/>
          <text x="35" y="245" fill="#FF5C77" font-size="19" font-weight="900">❌ 구호흡 · 설골 하강 (이중턱)</text>

          <path d="M 225 175 L 315 195 L 355 115" stroke="#00F5A0" stroke-width="6" stroke-linecap="round"/>
          <circle cx="315" cy="195" r="9" fill="#00F5A0"/>
          <text x="235" y="75" fill="#00F5A0" font-size="21" font-weight="900">⭕ 하악각 115° 리프팅</text>
          <path d="M 275 160 L 275 120" stroke="#00D2FF" stroke-width="5" stroke-linecap="round"/>
          <polygon points="275,106 265,124 285,124" fill="#00D2FF"/>
        </svg>
      </div>
      <div class="tc-foot">
        <span style="color:#FFBE3D;">❌ 혀 바닥 방치</span>
        <span style="color:#00F5A0;">⭕ 입천장 전체 밀착 (설골 상승)</span>
      </div>
    </div>
  </div>

  <div class="bottom-strip">
    <span>📌 전 트랙 공통 포맷 : 8초 무음 루프 · 상단 도감 넘버링(#001~#365) · 하단 3단계 실행 체크리스트</span>
    <span style="color:#00F5A0;">DAILY INFO REELS ENGINE v1.0</span>
  </div>
</body>
</html>
"""

def generate_procedural_audio(wav_path: Path, duration_sec: float = 8.0, sr: int = 44100):
    """Generates subtle dark medical HUD sound design: ambient pulse + UI clicks + lock chime."""
    total_samples = int(duration_sec * sr)
    samples = []
    # Key event timestamps (in seconds)
    click_times = [0.1, 1.2, 1.6, 2.5, 3.4, 4.3, 5.2, 6.0, 6.8]
    for i in range(total_samples):
        t = i / sr
        # Subtle low sub-bass medical studio hum (55Hz + 110Hz)
        val = 0.04 * math.sin(2 * math.pi * 55.0 * t) + 0.02 * math.sin(2 * math.pi * 110.0 * t)

        # Smooth biomechanical scan sweep (2.0s to 5.2s)
        if 1.6 <= t <= 5.2:
            u = (t - 1.6) / 3.6
            freq = 180.0 + 320.0 * u
            env = math.sin(math.pi * u) * 0.05
            val += env * math.sin(2 * math.pi * freq * t)

        # Crisp HUD clicks
        for ct in click_times:
            dt = t - ct
            if 0.0 <= dt < 0.045:
                c_env = math.exp(-dt * 95.0)
                val += 0.22 * c_env * math.sin(2 * math.pi * 1250.0 * dt)

        # Alignment lock-in chime at t = 5.4s (Mint harmonic chord)
        if 5.35 <= t < 6.6:
            dt = t - 5.35
            c_env = math.exp(-dt * 3.5) * 0.12
            val += c_env * (
                math.sin(2 * math.pi * 523.25 * dt) +
                0.7 * math.sin(2 * math.pi * 659.25 * dt) +
                0.5 * math.sin(2 * math.pi * 783.99 * dt)
            )

        val = max(-0.95, min(0.95, val))
        samples.append(int(val * 32767))

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))

def main():
    split_html_path = Path("/workspace/daily_info_reels/templates/day1_split.html")
    tracks_html_path = Path("/workspace/daily_info_reels/templates/four_tracks.html")
    split_html_path.write_text(SPLIT_AND_REEL_HTML, encoding="utf-8")
    tracks_html_path.write_text(FOUR_TRACKS_HTML, encoding="utf-8")

    split_png = OUT_DIR / "day1_split_preview.png"
    tracks_png = OUT_DIR / "four_tracks_showcase.png"
    reel_mp4 = OUT_DIR / "day1_prototype_reel.mp4"
    wav_path = FRAMES_DIR / "hud_audio.wav"

    generate_procedural_audio(wav_path, duration_sec=8.0)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox", "--disable-gpu"])

        # 1. Render 4-Tracks Showcase (1080x1350)
        page_tracks = browser.new_page(viewport={"width": 1080, "height": 1350, "device_scale_factor": 1})
        page_tracks.goto(f"file://{tracks_html_path}")
        page_tracks.wait_for_timeout(150)
        page_tracks.screenshot(path=str(tracks_png))
        page_tracks.close()

        # 2. Render Day 1 Split Preview (1080x1920) & 8s Reel Frames
        page_reel = browser.new_page(viewport={"width": 1080, "height": 1920, "device_scale_factor": 1})
        page_reel.goto(f"file://{split_html_path}")
        page_reel.wait_for_timeout(150)

        # Save the static key comparison preview at full aligned state (t=0.85)
        page_reel.evaluate("window.setFrameProgress(0.85)")
        page_reel.screenshot(path=str(split_png))

        # Render 80 frames (10 fps over 8s) and duplicate smoothly to 30fps via ffmpeg for <4s render speed
        num_keyframes = 80
        for idx in range(num_keyframes):
            t_norm = idx / (num_keyframes - 1)
            page_reel.evaluate(f"window.setFrameProgress({t_norm:.4f})")
            frame_file = FRAMES_DIR / f"f_{idx:04d}.jpg"
            page_reel.screenshot(path=str(frame_file), type="jpeg", quality=94)

        browser.close()

    # Encode 8.0s 30fps H.264 MP4 with yuv420p + procedural audio
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "10",
        "-i", str(FRAMES_DIR / "f_%04d.jpg"),
        "-i", str(wav_path),
        "-vf", "fps=30,scale=1080:1920:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", "8.0",
        "-movflags", "+faststart",
        str(reel_mp4)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("DONE:", split_png, tracks_png, reel_mp4)

if __name__ == "__main__":
    main()
