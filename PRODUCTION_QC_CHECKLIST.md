# @gymform.3d — 일일 3D 헬스·웨이트 도감 릴스 필수 제작·검수 체크리스트 (v61 Standard)

모든 릴스는 렌더링 후 중현 님께 보고하기 전에 반드시 아래 **8대 필수 체크리스트**와 자동 검증 스크립트(`python3 /workspace/daily_info_reels/qc_gate.py <reel.mp4>`)를 100% 통과해야 합니다.

---

## 1. [최우선] 풀스크린(Full-Bleed `1080×1920`) 스튜디오 일체화 & 네모 박스 테두리·검은 여백 0% (`rectangular_border_seam_check`)
- [ ] **네모난 회색 박스 경계선(액자 테두리) 및 좌우 검은 여백(Pillarbox) 영구 금지 (`t260u`, `t262u`)**:
  - `720×1280` 클립을 축소해 검은 캔버스(`10, 11, 14`) 가운데에 붙여서 좌우(`x=0..220`, `x=860..1080`)가 시커멓게 비거나 회색 사각형 테두리가 뜨는 현상을 영구 금지한다.
  - Day 2 (`day2_v38_master_reel.mp4`)와 동일하게 `1080×1920` 화면 좌우 끝(`x=0..1080`)과 상하 끝까지 매트 다크 차콜 스튜디오 벽면·바닥이 끊김 없이 풀스크린으로 이어져야 하며, `qc_gate.py`에서 경계선 단차 `v_seam < 2.0`, `h_seam < 2.0` 및 우측 외곽 스튜디오 밝기 `right_studio_lum >= 16.0`을 자동 검증한다.

## 2. 헬스 기구·바벨·원판의 물리적 정합성 검증 (`equipment_physical_integrity_check`)
- [ ] **유령 원판(Ghost Plate)·추가 바벨·기구 방향과 신체 방향 불일치 0% (`t261u`, `t264u`)**:
  - 파워랙 기둥은 정면을 보는데 마네킹은 옆을 보고 서 있거나, 마네킹이 멘 바벨 양끝 원판 2개 외에 기둥 옆에 3번째 유령 원판이 떠 있는 AI 기구 왜곡 클립은 절대 사용하지 않는다.
  - 또한 `rembg`로 마네킹 누끼를 따서 다리/팔/바벨이 잘려 나가게 하거나 OpenCV 2D 도형으로 가짜 바벨을 덧그리는 행위를 절대 금지하며, 오직 **처음부터 기구 왜곡이 없는 순수 3D 클립(어깨 위 바벨 양끝 원판 딱 2개)**만 사용한다.

## 3. Day 2 (`De_mCcFAVTO`) 3D 근육결(Fiber Striation) & 내부 3D 뼈대 X-Ray 질감 강제 (`muscle_fiber_grain_check`)
- [ ] **외계인 등뼈 돌출 및 당근 주황색 통째 페인트칠 영구 금지 (`t260u`)**:
  - 척추뼈가 등 피부 밖으로 하얗게 튀어나오거나, 요추 과부하 단계(Step 3)에서 엉덩이·허벅지 전체를 탁한 주황색으로 통째로 칠하는 것을 금지한다.
  - **Wrong 구간 (Step 1, Step 3)**: 매끄러운 다크 차콜 회색 근육 몸체 안쪽에서 과부하가 걸리는 **해당 관절·척추(Step 1 무릎 관절, Step 3 요추 L1~L5 및 하부 척추기립근)에만 깊고 선명한 크림슨 레드(`#F52830`, `R - G >= 90`) 내부 X-Ray & 3D 근섬유**가 비쳐야 하며, 비타겟 부위(정강이·발·둔근 하단)는 깨끗한 매트 다크 차콜 회색 피부를 유지한다.
  - **Right 구간 (Step 2, Step 4)**: 뼈 노출 0%, 오직 타겟 근육(대퇴사두근·대둔근)에만 조각된 에메랄드 그린(`#14E478`) 3D 근육결(`fs`·`sulc`)이 선명하게 살아나야 한다.
- [ ] **자동 수치 게이트 (`qc_gate.py`)**:
  - 각 Step 피크 프레임에서 `active_px >= 9,000px`, `chroma_mean >= 65.0`, 색상 순도 `rg_purity >= 90.0`, `sobel_mean >= 80.0`, `hf_fiber >= 9.5` 이상이어야 PASS.

## 4. 좌·우 플로팅 콜아웃 숫자와 하단 HUD 숫자 100% 실시간 동기화 (`callout_hud_sync_check`)
- [ ] **한 화면 내 각도(`°`) 및 퍼센트(`%`) 숫자 불일치 0% (`t260u`)**:
  - 좌측 콜아웃(`36°`)과 하단 HUD(`41°`)가 서로 다른 숫자를 표시하는 버그를 원천 차단하기 위해, 플로팅 콜아웃과 하단 HUD는 반드시 동일한 프레임 루프에서 동일한 `deg_val`, `pct_val` 변수를 받아 렌더링한다 (`좌측 각도 == 하단 각도`, `우측 % == 하단 %` 100% 일치).

## 5. 다크 언더레이(Dark Halo) 대구경 솔리드 각도 게이지 (`high_contrast_angle_gauge_check`)
- [ ] **밝은 근육/X-Ray 발광 위에서도 100% 선명한 각도 아크 (`t260u`)**:
  - 수직 기준 점선, 기울기 실선, 부채꼴 웨지, 솔리드 아크(`r = 126~144px`) 뒤에 다크 언더레이(검은 그림자 후광)를 깔아 밝은 발광 부위와 겹쳐도 각도 아크가 100% 또렷하게 보이도록 한다.
  - 측면 뷰(Step 3, Step 4)에서 의미 없는 무릎-발목 보조선은 제거하고, 고관절(`hip_r`)을 꼭짓점으로 한 상체 기울기 게이지(`41° FORWARD LEAN` vs `22° NEUTRAL SPINE`) 딱 1개에 시선을 집중시킨다.

## 6. 현실적인 헬스장 실수 자세 & 양발 바닥 완전 접지 (`realistic_posture_and_flat_foot_check`)
- [ ] 전 구간(Step 1~4)에서 양발바닥과 뒤꿈치가 스튜디오 바닥에 100% 평평하게 밀착(`0%` 까치발)되어야 하며, 무릎 아래 정강이·발가락 해골 노출은 `0%`여야 한다.

## 7. 하강·상승 동작 중 타겟 발광 부위 고정 및 4-Step 모션 차별화 (`non_drifting_anatomy_check`)
- [ ] 서 있을 때(`STAND`)부터 최저점 스쿼트(`SQUAT`)까지 타겟 발광 부위가 사라지지 않고 유지(`retention >= 0.72`)되어야 하며, 인접 Step 간 차이 `diff >= 28.0`, 구간 내 모션 `intra_step_motion >= 14.0`을 충족해야 한다.

## 8. 아이폰 릴스 세이프존 & 무왜곡 `136×108` 3-썸네일 도크-머리 분리 (`head_clearance_check`)
- [ ] 상단 운동명 헤더(`x=96, y=252..298`) + 무왜곡 `136×108` 3-썸네일 도크(`x=96, y=308..438`)와 마네킹 머리 최상단(`min_head_y >= 455px`) 사이 최소 `17px` 이상의 여유를 600프레임 전 구간에서 유지한다.


## 9. Shorts-Free 100% Exposed Anatomy, Close-Up Scale, Zero Upper-Back Whitening & Landmark-Locked Arrows (`v62 Standard` — `t267u`~`t272u`)
1. **Shorts-Free 100% Exposed Écorché Musculature (`t268u`)**:
   - Mannequins in lower-body reels MUST NEVER wear black shorts, trunks, or pants that obscure the hip crease, gluteus maximus, gluteus medius, or upper quadriceps.
   - Use medical/sculptural écorché prompts (`"seamless matte dark-charcoal 3D medical écorché anatomical muscle statue with exposed striated gluteus maximus, gluteus medius, and full quadriceps femoris muscle fiber bundles from pelvis to knee, with no fabric covering the hip or thigh muscles"`).
2. **True Day-2 Close-Up Full-Body Framing (`t269u`)**:
   - Scale 720p clips at `scale = 1.24 .. 1.32` with smooth vertical squat head tracking (`y_head_stand = 464px -> y_head_squat = 505..525px`) so there is ZERO empty black gap band below the Top Preview Dock (`y = 438px`) and `active_px >= 49,000 .. 85,000px` (`>= 16,000px` gate).
3. **Zero Upper-Back Whitening in Step 4 (`t270u`, `t272u`)**:
   - Non-target matte dark-charcoal skin luminance MUST be computed strictly from `cool_base = np.minimum(g_ch, b_ch)` clamped to `[18, 76]` with `over_bright_body` glare suppression across all 4 steps.
   - Verified by `qc_gate.py` `[8] Step 4 Upper-Back Whiteness Check`: `white_ratio <= 0.50%` (`0.01%` in `v62`) and `median_back_lum <= 85.0` (`49.1` in `v62`).
4. **MediaPipe Landmark-Locked Joint Pivot Rings (`◎`) & Directional Arrow Arcs (`t271u`, `t272u`)**:
   - Joint pivot rings (`◎`) and alignment axes are 100% locked to MediaPipe joint landmarks (`kn_r`, `kn_l`, `hip_r`, `sh`, `ank_r/l`).
   - Step 1 & Step 2 include high-contrast directional curved arrow arcs (`→ ←` inward knee valgus collapse in Step 1, `← →` outward knee tracking in Step 2), and Step 3 & Step 4 include hip-anchored solid angle arcs with directional arrowheads.

## 10. [필수] 양손 바벨 그립 락온 검증 (`hand_grip_lock_check` — `t273u`, `t277u`, `t278u`)
- [ ] **근본 원인 분석 (`v62` 노핸드 바벨 오류)**:
  - `v62`에서 반바지를 없애기 위해 `"medical écorché anatomical muscle statue"`(해부학 조각상) 프롬프트를 단독으로 넣자, 이미지 생성 AI가 해부학 교본의 차렷 자세 조각상을 우선시해 양팔을 허벅지 옆으로 내린 채 바벨만 어깨에 얹는 치명적 오류를 범함.
  - 또한 `qc_gate.py`에서 MediaPipe 관절 좌표를 추출하면서도 `left_wrist(15)`, `right_wrist(16)`가 어깨/바벨 샤프트 높이에 올라와 있는지 검증하는 게이트가 빠져 있었음.
- [ ] **재발 방지 규칙 & 자동 게이트 (`qc_gate.py` `[9] Hand-to-Barbell Grip Lock Check`)**:
  - 프롬프트에 `statue`, `mannequin`, `sculpture` 단어를 쓰지 않고, 양손이 바벨을 꽉 쥔 참조 포즈(`day3_v61_p1..p4_raw.jpg`)와 `"BOTH HANDS RAISED UP WITH FINGERS TIGHTLY WRAPPED AROUND THE STEEL BARBELL SHAFT BESIDE THE SHOULDERS (wide overhand grip locked onto the bar, elbows bent under the bar)"`를 명시한다.
  - `qc_gate.py` `[9]`에서 Step 1~4 전 구간(`STAND`, `MID`, `SQUAT`) 프레임에 대해 MediaPipe Pose로 `wrist_l(15)`, `wrist_r(16)`, `elbow_l(13)`, `elbow_r(14)`, `shoulder(11, 12)`를 추출하여:
    1. **양 손목이 양 팔꿈치보다 위에 위치 (`wrist_l_y < elbow_l_y` 및 `wrist_r_y < elbow_r_y`)**,
    2. **양 손목이 어깨/바벨 샤프트 높이(`wrist_y <= shoulder_y + 70px`)에 고정**되어 있는지 자동 검사하며, 한 프레임이라도 손이 내려가 있으면 즉시 `FAIL` 처리한다.
