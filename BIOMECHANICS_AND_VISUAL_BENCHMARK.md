# @gymform.3d — 해부학 교보재 & 생체역학(EMG) 논문 리서치 및 v21 렌더링 개선안

## 1. 왜 기존 버전(v20)이 봇 프로필(`avatar.png`)보다 못해 보였는가? (핵심 원인 3가지)

1. **흑백 변환(`Luminance`) 과정에서 원본 근섬유 결(Striations) 증발**
   - 봇 프로필(`gym_day1_correct_plate.jpg` 크롭)의 빨간 근육 부위는 `R 채널`은 210~255로 이미 포화되어 있고, 실제 촘촘한 **백색·은색 근섬유 결(Muscle Grain / Fascicle Striations)은 `G 채널`과 `B 채널`의 미세 명암차(`std = 21.0`)**에 들어 있습니다.
   - 기존 `v16~v20` 파이프라인은 영상을 먼저 회색 마네킹(`0.299R + 0.587G + 0.114B`)으로 평탄화한 뒤 단색(`#FF3B30` / `#18E269`)을 덮어씌웠기 때문에, 프로필 사진에서 느껴지던 **근섬유 가닥의 입체적인 고랑(Sulcus)과 능선(Fascicle Ridge)**이 뭉개져 플라스틱처럼 보였습니다.
2. **`✕ WRONG` 원본 클립의 한계 (근육이 아니라 뼈/스켈레톤만 발광)**
   - `gym_day1_wrong_i2v.mp4`는 애초에 근육결이 없는 **반투명 유리 팔 + 뼈(상완골·요골) X-Ray 주황빛**으로 생성된 클립입니다. 여기에 빨간색을 입히니 근육이 타겟팅되는 게 아니라 팔뼈 전체가 붉게 번져 보였습니다.
3. **단순 원형(Radial Gaussian) 마스킹으로 인한 타겟 근육 오조준**
   - 삼두근 외측두(말굽 모양 외측 갈래), 장두(후내측 긴 갈래), 전면삼각근(어깨 앞쪽 갈래)은 서로 뚜렷한 **해부학적 경계고랑(Intermuscular Septum)**으로 나뉘어 있는데, 기존에는 동그란 원형 마스크를 씌워 어깨와 삼두가 한 덩어리로 뭉뚱그려졌습니다.

---

## 2. 글로벌 3대 운동 해부학 교보재 벤치마킹 포인트

| 교보재 / 레퍼런스 | 핵심 시각화 기법 (벤치마킹 포인트) | `@gymform.3d` v21 적용 방식 |
| :--- | :--- | :--- |
| **1. Muscle & Motion** ([Strength Training App](https://www.muscleandmotion.com/strength-training-app/)) | - **Primary(주동근) vs Secondary(협응근)/보상근**을 명확한 해부학 경계선(Sulcus)으로 분리<br>- 근육 전체를 뭉뚱그리지 않고 **개별 갈래(Head)** 단위로 점등 | - 상완을 통째로 칠하지 않고 **① 회전근개 관절낭 → ② 전면삼각근(보상근) → ③ 삼두근 외측두(Lateral Head) + 장두(Long Head) + 내측두(Medial Head)** 폴리곤을 개별 분리 |
| **2. Frédéric Delavier** (*Strength Training Anatomy*) | - 피부 아래 근육의 **다발(Fascicle) 주행 방향(Pennation Angle)**을 따라 촘촘한 흑백/컬러 스트라이프 결을 그림<br>- 근육이 뼈에 붙는 **건(Tendon, 팔꿈치 주두/Olecranon)** 부위는 은백색으로 남겨 근육 벨리(Muscle Belly)를 돋보이게 함 | - 근육이 끝나는 팔꿈치 건(Olecranon Tendon)과 근육 사이 고랑(Septum)은 어두운 차콜/실버로 잡아주고, 근육 벨리 내부에만 **우상(羽狀) 근섬유 결**을 고대비로 조각 |
| **3. Kenhub / Anatomy Atlases** ([Kenhub Triceps Brachii](https://www.kenhub.com/en/library/anatomy/triceps-brachii-muscle), [NCBI StatPearls](https://ncbi.nlm.nih.gov/books/NBK536996/)) | - **외측두(Lateral Head)**: 상완골 후외측 요골신경구 위쪽에서 시작해 비스듬히 아래로 주행(가장 강하고 표층에서 말굽 외곽 형성)<br>- **장두(Long Head)**: 견갑골 관절하결절(Infraglenoid tubercle)에서 시작해 상완 후내측을 수직으로 내려감<br>- **내측두(Medial Head)**: 팔꿈치 바로 위 하부 심층에서 삼각형 모양으로 주두(Olecranon)에 부착 | - 3개 갈래(Lateral / Long / Medial Head)의 실제 기시·정지점 위치와 섬유 각도를 정확히 반영한 **해부학 폴리곤 세그멘테이션** 적용 |

---

## 3. 생체역학·근전도(EMG) 학술 논문 근거 (Day 1: Cable Rope Pushdown)

1. **[Kholinne et al., 2018 (PMC6136322 / PubMed 29503079)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6136322/) — *The different role of each head of the triceps brachii muscle in elbow extension***
   - **핵심 발견**: 어깨 거상 각도 `0°`(팔꿈치를 몸통 옆에 고정한 푸시다운 자세)에서는 **삼두근 장두(Long Head)와 외측두(Lateral Head)**가 가장 큰 장력과 근활성도를 발휘함 (`p < 0.05`). 반대로 팔꿈치가 앞으로 들리면(`90°+`) 장두의 길이-장력 관계(Length-Tension Relationship)가 깨져 장두 개입이 급감함.
   - **시각화 반영**: `Mistake 02 (팔꿈치 앞뒤 스윙)`에서 왜 삼두 장두가 죽고 전면삼각근/광배근이 개입하는지, 반대로 `Correct (팔꿈치 0° 고정)`에서 왜 **장두 + 외측두**가 폭발적으로 수축하는지 정확히 설명.
2. **[MSSE 2023 (Medicine & Science in Sports & Exercise)](https://doi.org/10.1249/01.mss.0000984912.08186.1d) — *Electromyographic Activity Of The Heads Of The Triceps Brachii During Selected Exercises***
   - **핵심 발견**: 트라이셉스 푸시다운(70% 1RM) 시 하단 완전 신전 구간에서 외측두·장두뿐 아니라 팔꿈치 바로 위 **내측두(Medial Head)**의 RMS EMG 활성도가 유의미하게 높게 나타남 (`p = 0.041, d = 0.62`).
   - **시각화 반영**: 하단 락아웃(Phase 4)에서 상완 상·중부의 **외측두·장두**와 함께 팔꿈치 바로 위 **내측두(Medial Head)**까지 3갈래가 입체적으로 수축하도록 타겟팅 정밀화.
3. **[Villalba et al., 2024 (IJSC)](https://doi.org/10.47206/ijsc.v4i1.250) — *Forearm Position Influences Triceps Brachii Activation During Triceps Push-Down Exercise***
   - **핵심 발견**: 로프/핸들 그립과 전완(Forearm)의 회전 제어가 삼두근 장두(`TLO`) 및 외측두(`TLA`)의 근전도 활성에 유의미한 차이를 만듦 (`p < 0.001`).
   - **시각화 반영**: Phase 3(`PIN ELBOWS & SPREAD ROPE`)에서 하단 로프를 벌릴 때 전완에서 삼두근 외측두로 이어지는 수축 경로를 명확히 연결.
4. **[Ando et al., 2020 (MSSE)](https://doi.org/10.1249/01.mss.0000685840.32359.18) & [CNKI Cadaver Architecture Study](https://en.cnki.com.cn/Article_en/CJFDTotal-YYCY200917011.htm)**
   - **핵심 발견**: 초음파(EFOV B-mode) 및 해부 실측 결과, 삼두근 **장두의 우상각(Pennation Angle)은 약 `12.9°`**, **내측두는 `9.3°`**, **외측두는 사선(`15°~22°`)**으로 근섬유 다발(Fascicle)이 배열되어 있으며 수축 시 근섬유 다발이 `8.7cm → 7.2cm`로 짧아지며 두꺼워짐.
   - **시각화 반영**: 단순 단색 칠하기가 아니라, 각 근육 갈래의 **실제 우상각(Pennation Angle: 외측두 `-18°` 사선, 장두 `+8°` 수직사선, 전면삼각근 `-24°` 부채꼴)** 방향을 따라 **미세 근섬유 결(Anisotropic Fascicle Striations)**을 생성·강조.
