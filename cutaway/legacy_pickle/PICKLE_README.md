# gymform_reels — gymform.3d 15초 릴스 + 캐러셀 템플릿

JSON 데이터 하나로 **15초 세로 릴스 MP4 + 4:5 캐러셀 카드 2장**을 뽑는 독립 파이프라인.
`flipcut-archive/`는 건드리지 않는다. 이 폴더만으로 동작한다.

두 가지 포맷을 같은 CLI로 돌린다.

| config | 포맷 | 렌더러 | 산출물 |
|---|---|---|---|
| `data/gymform_ep01.json` | FORM-CHECK + MACRO (나레이션) | `gfr/reel.py` | `downloads/gymform_squat_15s.mp4` 외 |
| `data/gymform_f1_knee_cutaway.json` | **F1 CUTAWAY** (해부 컷어웨이) | `gfr/cutaway.py` | `downloads/gymform_anatomy_15s.mp4` 외 |

## F1 CUTAWAY — "몸속을 잘라 보여준다"

`work/viral_fitness_reels_audit.md` §5 F1 설계안의 구현. 15.0초 / 450프레임.

```
0.00  훅 문장 2줄 (0프레임부터 표시)      WHAT'S INSIDE / YOUR KNEE?
2.60  살아있는 피부에 원형 창이 열림       PATELLA      (톱 아이콘, 부위 발광 + 화살표)
5.20  같은 창이 더 깊은 판으로 컷           CARTILAGE    (메스, 하중 화살표 2개 수렴)
7.80  활액 macro 판으로 컷                 SYNOVIAL FLUID (주사기, 젖은 광택 스윕)
10.40 하중 카운트업 0.0× → 7.6× + 링 게이지  BODY WEIGHT
12.86 창이 화면 전체로 확장(bloom)          FIVE MOVING PARTS
```

### 실행

```bash
cd work/gymform_reels
python3 render.py stills --config data/gymform_f1_knee_cutaway.json --times 0.0,3.2,6.9,12.4
python3 render.py all    --config data/gymform_f1_knee_cutaway.json --tag f1
python3 render.py audio  --config data/gymform_f1_knee_cutaway.json --tag f1nv --no-voice  # SFX만
```

`--tag`를 주면 `build/<tag>_video.mp4` · `<tag>_audio.wav` · `<tag>_final.mp4` · `<tag>_frames_check.png`로 나온다.
ep01(태그 없음)은 예전 경로(`build/video.mp4` …) 그대로다.

### 필요한 에셋 (같은 카메라에 등록되는 4장)

| 키 | 내용 | 비고 |
|---|---|---|
| `skin` | 피부가 덮인 살아있는 다리 (베이스) | 이 이미지 위에 구멍을 뚫는다 |
| `outer` | 같은 자세의 1차 단면(근육·인대·지방패드) | 창을 열 때 첫 번째로 보이는 층 |
| `bone` | 근육을 걷어낸 뼈·연골·반월판 | 두 번째 깊이 |
| `macro` | 관절강 활액 macro (`knee_fluid_v3`: 청백색 활액 방울) | 세 번째 깊이 |

네 장은 **같은 포즈·같은 프레이밍**이어야 한다. `skin`은 `outer`를 `images --edit`로 "피부를 덮어라"로 만들면
포즈가 유지된 채 등록된다(이번 화는 `assets/anatomy/knee_skin_v1.png`). 카메라는 `camera.focus_src`(원본 좌표) →
`camera.focus_out`(화면 좌표) 한 쌍만 쓰므로, 네 장의 관절 위치가 ±40px 안에 있으면 창 안팎이 이어진다.

### JSON 필드

- `camera` — `focus_src`/`focus_out`/`scale[2]`. 15초 동안 `scale`만 천천히 밀어 미세 클로즈업(팬 없음).
- `window` — 창 중심 `cx/cy`, 반지름 `r`, 여는 시각 `open.at`·`open.dur`.
- `bloom` — 창이 화면을 덮는 시각·길이·최종 반지름.
- `shots[]` — `t`마다 `window`(중심·반지름), `label`(≤3단어), `tool`(saw/scalpel/syringe/drill), `content`:
  - `match_base: true` → 베이스 카메라에 그대로 등록(구멍 뚫린 단면처럼 보임)
  - `scale`+`focus` → 그 판을 창 중심에 맞춰 확대(macro 컷)
  - `backdrop` → 콘텐츠의 검은 영역을 다른 판(예: `outer`)으로 채워 "빈 구멍"이 안 보이게
  - `tint`/`mix`/`lift`/`scale_drift` — 색보정과 미세 푸시
- `events[]` — `glow`(부위 발광), `arrow`(지시/하중 화살표), `sheen`(젖은 광택 스윕), `gauge`(링 게이지).
  `rel:[dx,dy]`는 창 중심 기준 상대 좌표, `clip:true`면 창 밖으로 안 나간다.
- `counter` — `from`→`to`, `start`/`dur`, `caption`, `x/y/size`.
- `sheet.times` — 검수 시트 컷. `audio.lines[].at`은 **합성 실측 길이로 잡아야** 다음 줄에서 잘리지 않는다
  (`trim_lead: true`면 앞뒤 무음을 잘라 `at`이 실제 발성 시작이 된다).

## 타이포·그래픽 시스템 (`style` 블록, 2026-11-16 고급화)

모든 타이포/그래픽은 `gfr/cutaway.py`의 `STYLE_DEFAULTS` 하나로 묶여 있고, JSON `style`에서 덮어쓴다.

| 키 | 하는 일 |
|---|---|
| `style.hook` | `eyebrow`(·로 구분하면 가운뎃점만 라임), `meta`(얇은 세로 구분선으로 이어 붙는 메타 문자열 배열), `grad_top/bot`(줄마다 다시 시작하는 세로 그라디언트), `scrim`(글자 뒤 페더 스크림 — 스트로크 대신 이걸로 가독성 확보), `dim` |
| `style.label` | 다크 글래스 알약: `glass`(0=투명,1=검정)·`blur`·`radius`·`pad_x/pad_y`·`rim_alpha`, 그리고 `leader`/`leader_w`/`leader_color`(캘아웃 엘보 + 끝점 노드) |
| `style.counter` | 캡션/숫자/단위 위계와 눈금 스케일(`meter_w`, `meter_ticks`), `y`는 숫자 베이스라인 |
| `style.rim` | 컷 창 테두리: `hairline`(메탈 실선), `accent`(라임 헤어라인), `bevel`(이중 베벨 폭), `scanner`(컷 진입 시 1회 스윕) |

원칙: **굵은 검은 스트로크를 쓰지 않는다.** 다층 그림자 + 페더 스크림으로 대비를 만들고,
글자는 줄 단위 그라디언트로 종이 질감을 준다. 화살표/캘아웃은 모바일 가독을 위해
`어두운 헤일로 → 포인트 컬러 → 흰색 코어` 3겹으로 그리고, 화살촉은 44px, 선은 9px가 기본이다.

`render_video(out, i0, i1)`로 프레임 구간만 인코딩할 수 있어(긴 렌더를 조각내 실행),
`render.py reel --segments N`이 N조각으로 나눠 인코딩한 뒤 `-c copy`로 이어 붙인다.

### 이번 화 실측 (2026-11-16)

| 항목 | 값 |
|---|---|
| 컷 간격 | 2.6 / 2.6 / 2.6 / 2.6 / 2.46초 (F1 게이트 ≥2.0) |
| 라벨 | 화면당 1개, 전부 ≤3단어 |
| 훅 | 0프레임 표시(0.00s) |
| 카메라 | 팬 0회, 5% 푸시만 |
| 나레이션 | Supertonic M1 · 1.13x · 6줄 13.6초, 줄 간 최대 공백 0.18초 |
| 음량 | -14.0 LUFS (2패스 보정), h264 crf17 + aac 192k |
| 검증 | ffprobe 15.000s·450프레임, PyAV 전 프레임 디코드 0오류 |

### 2026-11-16 타이포·그래픽 고급화 라운드

| 항목 | 이전 | 이후 |
|---|---|---|
| 훅 | 7px 검정 스트로크 + 흰 자막 | eyebrow(`BIOMECHANICS · DEEP SQUAT`) + 132px 디스플레이(줄별 그라디언트) + 라임 룰 + 메타 3종, 스트로크 없음 |
| 라벨 | 불투명 검정 사각 박스 | 블러 유리 알약 + 메탈 아웃라인 + 라임 밑변 + 좌측 탭 + 도구 칩 + 엘보 캘아웃(끝점 노드) |
| 화살표 | 7px, 흰색, 창 안쪽만 | 9px + 44px 화살촉, 검정 헤일로 + 라임 + 흰 코어, `clip: false`로 창 밖에서 관절로 들어오게 |
| 카운트업 | 가운데 정렬 196px + 밑줄 바 | 좌측 정렬: 캡션(라임 불릿) / 178px 숫자 + 라임 `×` / 41칸 눈금 스케일, 창 테두리 60칸 다이얼 |
| 창 테두리 | 단일 링 + 호 | 이중 베벨 + 각도별 메탈 셰이딩 + 헤어라인 + 라임 실선 + 45° 정합 틱 + 진입 시 스캐너 스윕 |
| 활액 컷 | `knee_fluid_v2`(섬유 가닥 = 거미줄) | **`knee_fluid_v3`**(청백색 활액 방울 + 연골 광택), scale 0.74 → 0.79 미세 푸시 |
| 검증 | — | 15.000s·450f·1080×1920·30fps, -14.0 LUFS, PyAV 450/450 디코드, 시트 12컷 육안 검수 |

## 산출물 (1화 = gymform_ep01)

| 파일 | 내용 |
|---|---|
| `downloads/gymform_squat_15s.mp4` | 1080×1920 · 30fps · 450프레임 · h264 + aac 15.000s, -14 LUFS |
| `downloads/gymform_carousel_1.png` | 1080×1350 스쿼트 폼체크 요약 카드 |
| `downloads/gymform_carousel_2.png` | 1080×1350 30g 단백질 비교 카드 |
| `downloads/gymform_15s_frames_check.png` | 12컷 프레임 검수 시트 (0.3s~14.7s) |

## 실행

```bash
cd work/gymform_reels
python3 render.py stills --times 0.6,3.9,8.6,14.4   # 프레임 미리보기 → build/still_*.png
python3 render.py reel      # 영상만 (build/video.mp4)
python3 render.py audio     # 나레이션+효과음 (build/audio.wav)
python3 render.py mux       # 합치기 (build/final.mp4)
python3 render.py cards     # 캐러셀 2장
python3 render.py sheet     # 12컷 검수 시트
python3 render.py all       # 전부
```

## 구조

```
data/gymform_ep01.json   # 대본·수치·타이밍·문구·팔레트 (다음 화는 이 파일만 복제)
gfr/theme.py             # 폰트·색·텍스트(자간/정렬)·도형·이징
gfr/media.py             # numpy 캔버스, Ken Burns 크롭, 스프라이트 추출, 글로우·비네트
gfr/reel.py              # 씬 3개(폼체크 / 전환 / 매크로+CTA) 프레임 합성 → ffmpeg 파이프
gfr/audio.py             # Supertonic TTS(1.12x) + 합성 SFX( whoosh/pop/ding/impact/riser/tick) + 베드 → -14 LUFS
gfr/carousel.py          # 1080×1350 카드 2장
gfr/sheet.py             # ffmpeg로 최종 MP4에서 12프레임 추출 → 검수 시트
assets/fonts/            # Anton, Barlow Condensed, Archivo Black, Inter (Google Fonts, OFL)
```

## 다음 화 만드는 법

1. `data/gymform_ep01.json`을 `data/gymform_ep02.json`으로 복제.
2. `assets.form` / `assets.protein` 경로만 새 기준 이미지로 교체. 두 이미지는 `941×1672` 기준으로
   좌표(`sprite_boxes`, `crops`)를 잡았으므로, 다른 크기면 `reel.py`의 `SW/SH`와 좌표를 맞춰야 한다.
3. `reel.scenes`의 초 단위 컷, `reel.macro.rows`의 수치/최대값, `audio.lines`의 대사와 `at` 시각,
   `carousel.card1|card2`의 문구를 바꾼다.
4. `python3 render.py all` → `build/final.mp4` 확인 후 `downloads/`로 복사.

## 설계 메모

- 기준 이미지의 **박힌 숫자("78°", "95°")는 지우지 않는다.** 스프라이트 알파를 마스크로 써서
  같은 픽셀을 밝히기만 하므로 타이포가 100% 원본 그대로다. (초기에는 패치 후 재합성했으나
  배경 그라디언트 차이로 사각 흔적이 보여 폐기.)
- 매크로 패널은 기준 이미지의 카드 블록을 **행별 배경색으로 덮고** 그 위에 새 HUD를 그린다.
  덮는 영역은 카드(1104~1470)·기존 CTA(1436~1600)·가짜 SAVE/SHARE/LIKE(1568~1650) 세 곳.
- 패널은 화면 고정 좌표라서 매크로 씬 베이스에는 줌을 넣지 않았다(넣으면 라벨/패널이 겹친다).
  대신 전환 씬의 1.05 크롭에서 1.0으로 풀리는 낙차로 "줌" 느낌을 만든다.
- 나레이션은 구간별로 따로 합성해 `at` 시각에 배치하고, 다음 대사 시작에서 자른 뒤 10ms 페이드아웃.
- 최종 음량: `ebur128` 측정 → 선형 게인 → `alimiter`(-1.5dB) → 2차 보정으로 -14.0 LUFS 고정.
