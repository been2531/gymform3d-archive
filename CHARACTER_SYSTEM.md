# @gymform.3d — A/B 마스터 캐릭터 시스템 & 근거 기반 스포팅/큐잉 가이드라인 (`CHARACTER_SYSTEM.md`)

중현 님의 지시(`t282u`–`t296u`)에 따라 **Day 2 (`day2_v38_master_reel.mp4`) 처음 마네킹(Step 1·2)의 실측 색상·무광 매트 질감(`RGB 84, 87, 91`, `median lum ~ 86`, `lap_median <= 1.8`)**, **`@demicstory` 24편 전수 분석 결과**, 그리고 **공인 스트렝스 가이드라인(NSCA Spotting Guidelines, Starting Strength Tactile Cueing, Muscle & Motion 3D Biomechanics, ACE Scapular Cueing)**을 결합하여 수립한 `@gymform.3d` 공식 A/B 캐릭터 표준 문서입니다.

---

## 1. Day 2 처음 마네킹(Step 1·2) 실측 & `@demicstory` 24편 전수 조사 결과 (`t296u`)

| 분석 항목 | Day 2 처음 마네킹(Step 1·2) & `@demicstory` 실측 결과 | `@gymform.3d` 반영 및 업그레이드 표준 |
| :--- | :--- | :--- |
| **1) 마네킹 피부 색상 & 무광 매트 질감 (`Character A`)** | - **Day 2 처음 마네킹(Step 1·2) 실측**: 뒤쪽 Step 3(`lum 135`, 허옇게 뜬 실버)와 달리, 처음 마네킹(Step 1·2)은 **`RGB = (84, 87, 91)` (`p10=62, p25=72, p50=86, p75=106, p90=122`, `max <= 136`)**의 **차분하고 묵직한 무광 매트 슬레이트-그레이(`Smooth Matte Cool Slate-Grey`)**임.<br>- **표면 매트 질감(`Surface Matte Smoothness`)**: 비타겟 피부에 자글자글한 에코르셰 근육 잔주름이나 번들거리는 금속 하이라이트가 없고, `lap_median = 1.0~2.6`, `lap_p90 = 11~18`의 **부드럽고 매끈한 무광 클레이/마네킹(`Smooth Matte Clay Mannequin`)** 질감임. | - **Bilateral + Gaussian Dome Matte Clay Smoothing**을 비타겟 피부에 적용해 자글자글한 잔주름을 완전히 정돈(`lap_median = 0.25~0.55 <= 3.5`, `lap_p90 <= 4.5`).<br>- **Step 1~4 전 구간 `RGB = (84, 87, 91)` (`median lum ~ 86.0`) 분위수 정규화**로 Day 2 처음 마네킹과 100% 동일한 색상·매트 질감 통일 (`qc_gate.py` Gate `[10]` 자동 검증). |
| **2) 수록 운동 종목 & 기구 그립** | - `demic_bench/` 24편 전수 확인 결과:<br>  * **하체/둔근**: 스쿼트(`DHj07hdPpMV`), 불가리안 스플릿 스쿼트 & 바벨 힙쓰러스트(`De77WaXAiXd`), 덤벨 둔근 루틴(`DH8BOjPvTNS`)<br>  * **등/후면**: 와이드그립 랫풀다운(`demic_gym_day2.mp4`), 바벨 로우 3변형(`DetDsIDj2BJ`), 페이스풀(`De1ZjBfDzns`), 덤벨 등(`De9qPwyGFvK`)<br>  * **어깨/가슴/팔**: 덤벨 3D 숄더(`DUlzpgekQRy`), 숄더 6동작(`DHo3VoDtbIa`), 케이블 푸시다운(`De5iN7hCmke`), 팔꿈치 각도별 프레스/푸시업(`DePzKDbFB0y`)<br>- 모든 기구 동작에서 **양손 손가락이 바벨/덤벨/케이블 로프를 물리적으로 꽉 쥐고 있음**. | - 앞으로 제작할 운동 자세·기구·카메라 앵글은 박스 내 `@demicstory` 24편 라이브러리를 1순위 레퍼런스로 카피.<br>- `qc_gate.py` Gate `[9] Hand-to-Barbell Grip Lock Check`로 양손 그립 100% 강제. |
| **3) 화살표 & 그래픽 오버레이** | - 다리나 몸통 위에 부채꼴 웨지(색칠), 수직 점선, 뼈대 직선, 여러 개의 마커 점을 겹쳐 그리지 않음.<br>- **신체 실루엣 바깥 스튜디오 여백(`exterior margin`)에 얇고 깔끔한 `4px` 단일 곡선 화살표(`rd.arc` + 화살촉) 딱 1개**만 배치해 운동하는 마네킹의 근육과 동작을 전혀 가리지 않음 (`t280u`, `t289u`). | - 부채꼴·점선·뼈대 직선·끝점 핀 마커 **100% 영구 제거**.<br>- 오직 관절 피벗 링(`◎`) + **몸 바깥 여백(`exterior margin`)의 단일 `4px` 곡선 화살표**만 사용. |
| **4) 2인 보조/코칭 등장 여부** | - `@demicstory` 24편은 **전부 1인 마네킹(`Solo Lifter`) 단독 구성**이며, 2인 보조자(Spotter)가 등장하는 릴스는 없음 (`0편`). | - 1인 운동/기구/화살표는 `@demicstory` & Day 2 처음 마네킹을 100% 카피하고, **2인 보조(`Character B`)는 임의 창작 없이 공인 NSCA Spotting Guidelines 및 임상 촉지 큐잉(Tactile Cueing) 정석 위치**를 따름 (`t290u`–`t294u`). |

---

## 2. CHARACTER A vs CHARACTER B 공식 비주얼 스펙 (`t286u`–`t296u`)

| 구분 | **CHARACTER A — `PRIMARY LIFTER` (메인 운동 수행자)** | **CHARACTER B — `SPOTTER / BIOMECHANICS COACH` (보조자 · 코치)** |
| :--- | :--- | :--- |
| **역할** | 모든 릴스의 주연 운동 수행자. 타겟 근육의 수축과 잘못된 자세의 관절 과부하를 보여주는 주체. | 고중량 프리웨이트 안전 보조(Spotting) 및 자세 교정 핸즈온 촉지 큐잉(Tactile Cueing) 파트너. |
| **표면 재질 & 밝기** | **Smooth Matte Cool Slate-Grey (`RGB = 84, 87, 91`, `#54575B`)**<br>- Day 2 처음 마네킹(Step 1·2) 1:1 매칭: `p10=62, p25=72, p50=86, p75=106, p90=122` (`max <= 136`)<br>- **Bilateral Matte Clay Finish (`lap_median <= 1.8`)**: 잔주름·번들거림 없는 매끈한 무광 마네킹 표면 (`R:G:B = 0.97 : 1.00 : 1.05`) | **Translucent Deep Gunmetal-Charcoal (`RGB = 46, 52, 62`, `#2E343E`)**<br>- 중앙 밝기 `median lum = 50` (`p10=34, p50=50, p90=72`, Character A보다 한 단계 더 어둡고 차분한 딥 건메탈)<br>- `65%` 반투명 고스트 + 시안(`#38E8FF`) 림라이트 처리로 **A의 근육 시야 방해 0% (`Zero Occlusion`)** |
| **해부학 발광 색상** | - **Wrong (과부하/보상)**: Crimson Red (`#F82830`) 3D 근섬유 / 관절 X-Ray<br>- **Right (타겟 고립)**: Emerald Green (`#14E478`) 3D 근섬유 결 | - 자체 근육 발광 **0%** (시선 분산 방지)<br>- 오직 **보조자의 손끝이 A 또는 바벨에 닿는 접점(Contact Zone)**에만 **Cyan-Blue (`#38E8FF`) 촉지 링/가이드 글로우** 점등 |
| **시야 방해 금지 원칙 (`t289u`)** | 화면 중앙(`Center Stage`)에서 타겟 근육(대둔근·대퇴사두근·광배근·삼각근)이 100% 가림 없이 노출됨. | **절대 카메라와 Character A의 타겟 근육 사이(전면)를 가로막지 않음.** 항상 A의 등 뒤쪽 대각선 바깥(`Rear-Exterior`) 또는 머리 위쪽(`Head-of-Bench`) 외곽 프레임에 위치. |

---

## 3. 종목별 정석 보조자(Character B) 위치 및 손 위치 (근거 문헌 기반 — 임의 창작 0%)

### [근거 문헌 출처]
1. **NSCA (National Strength and Conditioning Association)**: *Basics of Strength and Conditioning Manual* (Chapter 3: Spotting Techniques, Figures 3-3 to 3-16) & *Essentials of Strength Training and Conditioning (4th Ed.)*.
2. **Starting Strength (Mark Rippetoe)**: *Tactile Cues and Coaching* (Platform Biomechanical & Sacrum/Thoracic Cueing Standard).
3. **NIH PMC4725067 (Myer et al.)**: *The Back Squat Part 2: Targeted Training Techniques to Correct Functional Deficits and Technical Factors that Limit Performance* (Lateral Knee RNT / Tactile Valgus Correction).
4. **ACE (American Council on Exercise) & Muscle & Motion**: *Correct Cues for Scapular Motion* (Inferior Angle of Scapula Depression/Retraction Tactile Cueing).

| 운동 종목 | 보조자(Character B) 정석 스탠스 위치 | 보조자(Character B) 정석 손 위치 & 그립 | 3D 릴스 카메라 구도 & 시야 방해 방지(`Zero Occlusion`) 원리 |
| :--- | :--- | :--- | :--- |
| **1. 바벨 백스쿼트**<br>(`Barbell Back Squat`)<br>*[NSCA Fig 3-13~3-16]* | **운동자(A) 바로 뒤쪽 (`Direct Rear Stance`)**<br>- A보다 발 간격을 살짝 넓게 벌리고, A의 하강·상승 템포에 맞춰 무릎을 함께 굽히며 동행(Descend/Ascend in unison). | **양팔을 A의 겨드랑이 아래(`Under Armpits / Around Upper Torso`)로 넣어 흉곽 옆에 비접촉 대기**<br>- 바벨을 직접 잡으면 균형이 무너지므로 금지.<br>- 실패/앞쏠림(`Good-Morning Squat`) 시 **흉곽(가슴 아래 몸통)을 감싸 들어 올리며 상체 각도 유지**. | - **3/4 전면 또는 3/4 측면 카메라**: B는 A의 등 뒤쪽 바깥(`Rear-Left Exterior`)에 반투명 건메탈(`65% opacity`)로 서므로 **A의 대둔근·대퇴사두근·무릎과 바벨 그립을 0%도 가리지 않음**.<br>- A의 상체 앞쏠림 교정 시 B의 손끝이 흉추/가슴 옆에 시안(`#38E8FF`) 큐잉 포인트로 표시됨. |
| **2. 바벨 벤치프레스**<br>(`Barbell Bench Press`)<br>*[NSCA Fig 3-3~3-6]* | **벤치 머리맡 정중앙 직립 (`Stand Erect at Head of Bench`)**<br>- 양발 어깨너비, 무릎 미세 굴곡, 허리 중립 유지. | **운동자(A) 양손 안쪽 바벨 샤프트에 얼터네이트 그립 (`Closed Alternated Grip Inside Lifter's Hands`)**<br>- 리프트오프(Lift-off)와 랙인(Rack-in) 시에는 바를 꽉 잡고, 수행 중에는 바 궤적을 따라가되 비접촉 근접 대기. | - **3/4 측면-상단 카메라**: B는 벤치 머리 뒤쪽 외곽(`Top-Rear Margin`)에 위치하므로 **A의 대흉근·전면삼각근·삼두근과 바벨 하강 궤적을 전혀 가리지 않음**. |
| **3. 덤벨 인클라인 / 숄더 프레스**<br>(`Dumbbell Press`)<br>*[NSCA Fig 3-7~3-9]* | **벤치 머리 뒤쪽 직립 (`Behind Head of Bench`)**<br>- A의 시야를 방해하지 않는 머리 뒤 밀착 스탠스. | **운동자(A)의 팔꿈치가 아닌 양쪽 손목/전완 원위부 (`Grasp/Hover Near Wrists, NEVER Elbows`)**<br>- *근거*: 팔꿈치를 받치면 손목이 꺾이며 덤벨이 얼굴/가슴으로 떨어질 위험이 있으므로 반드시 **손목(Wrists)**을 보조함. | - **3/4 전면 카메라**: B는 벤치 등받이 뒤쪽에서 팔만 뻗어 A의 **손목 외곽(`Wrist Contact Ring`)**을 가이드하므로 **A의 측면/전면 삼각근과 상부 흉근이 100% 노출됨**. |
| **4. 랫풀다운 / 시티드 로우**<br>(`Lat Pulldown / Row`)<br>*[ACE & Muscle & Motion]* | **운동자(A)의 등 뒤쪽 대각선 측면 (`Posterolateral Coaching Stance`)** | **견갑골 하각(`Inferior Angle of Scapula`) 및 팔꿈치 하방 궤적 가벼운 촉지(`Tactile Cue`)**<br>- 상부 승모근 으쓱거림(`Trap Shrug`) 방지 및 견갑골 하강(`Scapula Depression`) 유도. | - **3/4 후면 카메라**: B는 케이블 머신 반대편 바깥 여백에 반투명 건메탈 실루엣으로 위치하거나 **시안(`#38E8FF`) 핸즈온 터치 마커만 견갑골 하각에 표시**되어 광배근(`Lats`) 근육결을 가리지 않음. |
