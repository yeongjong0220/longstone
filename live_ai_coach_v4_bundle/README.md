# OnPose Live Quality Coach v4 — Bundle

웹캠으로 필라테스 동작을 실시간 인식하고, 3D 자세 복원 → 품질 점수화 → Gemini LLM 한국어 코칭 피드백까지 이어지는 통합 파이프라인입니다.

이 폴더는 [`live_ai_coach_v4_quality.py`](min/min_dev_park/live_ai_coach_v4_quality.py)를 실행하는 데 **실제로 필요한 파일만** 모아둔 최소 번들입니다.

---

## 🧠 파이프라인 개요

```
웹캠 → MediaPipe Pose Landmarker (2D 33점)
     → TemporalLifterWithPhaseHead (3D 15관절 복원, 81프레임 윈도우)
     → 국면별 Mahalanobis 거리 점수화 (PASS / WARN / FAIL)
     → Gemini 2.5 Flash → 한국어 코칭 피드백 3문장
```

### 🔄 전체 데이터 흐름

```mermaid
flowchart TD
    Cam([📷 웹캠 frame]) --> Flip[좌우 반전 + RGB 변환]
    Flip --> MP[MediaPipe PoseLandmarker<br/>33개 2D 관절 + visibility]
    MP --> StateRouter{현재 상태?}

    StateRouter -- SELECTION --> Fist[update_fist_click<br/>주먹→펴기 = 클릭]
    Fist --> SelectBox[3개 자세 박스 hit-test]
    SelectBox --> WAIT[STATE: WAIT_START]

    StateRouter -- WAIT_START / ACTIVE / REPORT --> Hold[HoldTracker<br/>1.4초 정지 = 발동]
    Hold --> CtrlBtn{버튼?}
    CtrlBtn -- START --> ACTIVE[STATE: ACTIVE 녹화]
    CtrlBtn -- FINISH --> Finish[finish_session]
    CtrlBtn -- RESELECT --> SEL[STATE: SELECTION]
    CtrlBtn -- QUIT --> End([종료])

    ACTIVE --> Map2D[33점 → 15관절 매핑<br/>mediapipe_landmarks_to_lifter_2d]
    Map2D --> Buffer[81프레임 슬라이딩 윈도우]
    Buffer --> Lifter[🧠 TemporalLifterWithPhaseHead<br/>2D → 3D + 국면 추정]
    Lifter --> Frame3D[(latest_3d 누적<br/>session_3d)]

    Finish --> CheckScorer{스코어러<br/>존재 & frames≥20?}
    CheckScorer -- Yes --> Score[score_sequence_3d<br/>국면별 Mahalanobis d²]
    CheckScorer -- No --> Fallback[fallback_summary<br/>각도 기반 단순 비교]
    Score --> Summary[summarize_score_rows<br/>PASS/WARN/FAIL 통계]
    Fallback --> Summary
    Summary --> Gemini[🤖 Gemini 2.5 Flash<br/>한국어 코칭 3문장]
    Gemini --> Report[STATE: REPORT 화면 표시]

    style Cam fill:#FFE4B5
    style Lifter fill:#B5E7FF
    style Gemini fill:#D4FFB5
    style End fill:#FFB5B5
```

### 🎮 상태 머신

```mermaid
stateDiagram-v2
    [*] --> SELECTION
    SELECTION --> WAIT_START : 자세 박스 클릭<br/>(주먹→펴기)
    WAIT_START --> ACTIVE : START 1.4초 홀드
    WAIT_START --> SELECTION : RESELECT 홀드
    WAIT_START --> [*] : QUIT 홀드 / q
    ACTIVE --> REPORT : FINISH 홀드 / f<br/>(점수화 + LLM)
    ACTIVE --> SELECTION : RESELECT 홀드
    ACTIVE --> [*] : QUIT 홀드 / q
    REPORT --> ACTIVE : START 홀드<br/>(재시도)
    REPORT --> SELECTION : RESELECT 홀드
    REPORT --> [*] : QUIT 홀드 / q

    note right of ACTIVE
        깨끗한 프레임만 session_3d에
        누적 (컨트롤 호버 시 일시정지)
    end note
```

### 🧠 모델 아키텍처 — `TemporalLifterWithPhaseHead`

[pilates_temporal_lifter/model.py](pilates_temporal_lifter/model.py)에 정의된 멀티태스크 시간 합성곱 모델로, **2D 관절 시퀀스 → 3D 관절 시퀀스 + 운동 국면(phase)** 을 동시에 예측합니다.

```mermaid
flowchart TB
    Input["입력 텐서<br/>(B, T=81, J=15, C=3)<br/>C = (x, y, observed_mask)"]:::io
    Input --> Reshape["Reshape + Transpose<br/>(B, T, J·C) → (B, 45, T)"]:::reshape
    Reshape --> Proj["Input Projection<br/>Conv1d 45 → 256, k=1"]:::proj

    Proj --> B1
    subgraph TCN["🧱 Dilated Residual TCN (4 blocks)"]
        direction TB
        B1["ResidualTemporalBlock #1<br/>causal Conv1d, k=3, dilation=1<br/>hidden=256"]:::block
        B2["ResidualTemporalBlock #2<br/>causal Conv1d, k=3, dilation=2"]:::block
        B3["ResidualTemporalBlock #3<br/>causal Conv1d, k=3, dilation=4"]:::block
        B4["ResidualTemporalBlock #4<br/>causal Conv1d, k=3, dilation=8"]:::block
        B1 --> B2 --> B3 --> B4
    end

    B4 --> Feat[("공유 시계열 피처<br/>(B, 256, T)")]:::feat

    Feat --> PoseHead["Pose Head<br/>Conv1d 256 → 45, k=1"]:::head
    Feat --> PhaseHead["Phase Head<br/>Conv1d 256 → 3, k=1"]:::head

    PoseHead --> Pose3D["pred_3d<br/>(B, T, 15, 3)<br/>3D 관절 좌표"]:::out
    PhaseHead --> Phase["phase_logits<br/>(B, T, 3)<br/>국면 분류 logits"]:::out

    classDef io fill:#FFE4B5,stroke:#333,stroke-width:2px
    classDef reshape fill:#FFFACD,stroke:#999
    classDef proj fill:#E0F4FF,stroke:#3399cc
    classDef block fill:#B5E7FF,stroke:#0066aa,stroke-width:1.5px
    classDef feat fill:#D4FFB5,stroke:#339933,stroke-width:2px
    classDef head fill:#FFD6F5,stroke:#cc3399
    classDef out fill:#FFB5B5,stroke:#cc3333,stroke-width:2px
```

#### 🔬 ResidualTemporalBlock 내부

```mermaid
flowchart LR
    X["입력 x<br/>(B, 256, T)"]:::io --> C1["CausalConv1d<br/>k=3, dilation=d"]:::conv
    C1 --> N1[BatchNorm1d]:::norm
    N1 --> G1[GELU]:::act
    G1 --> D1[Dropout 0.10]:::drop
    D1 --> C2["CausalConv1d<br/>k=3, dilation=d"]:::conv
    C2 --> N2[BatchNorm1d]:::norm
    N2 --> D2[Dropout 0.10]:::drop
    D2 --> Add(("+"))
    X -. residual .-> Add
    Add --> G2[GELU]:::act
    G2 --> Y["출력<br/>(B, 256, T)"]:::io

    classDef io fill:#FFE4B5,stroke:#333
    classDef conv fill:#B5E7FF,stroke:#0066aa
    classDef norm fill:#E0E0FF,stroke:#6666cc
    classDef act fill:#D4FFB5,stroke:#339933
    classDef drop fill:#FFFACD,stroke:#999
```

> 💡 **Causal 1D Convolution**: 미래 프레임을 절대 보지 않도록 좌측에만 `(k-1)·dilation` 만큼 zero-pad 후 일반 Conv1d 적용 → 실시간 추론 가능.

#### 📐 텐서 차원 변화

| 단계 | 텐서 모양 | 비고 |
|---|---|---|
| 입력 | `(B, T=81, J=15, C=3)` | (x, y, observed_mask) |
| Reshape + Transpose | `(B, 45, T)` | Conv1d용 채널-시간 배치 |
| Input Projection | `(B, 256, T)` | 1×1 conv |
| TCN 4개 block 통과 | `(B, 256, T)` | 동일 차원 유지 (residual) |
| Pose Head + reshape | `(B, T, 15, 3)` | **3D 좌표 출력** |
| Phase Head + transpose | `(B, T, 3)` | **국면 logits 출력** |

#### 🕒 시간적 수용 영역 (Receptive Field)

| Block | dilation | block당 추가 | 누적 RF |
|---|---|---|---|
| #1 | 1 | 4 | 5 |
| #2 | 2 | 8 | 13 |
| #3 | 4 | 16 | 29 |
| #4 | 8 | 32 | **61** |

> 시간 t의 출력은 과거 60프레임 + 현재 1프레임을 종합 — 윈도우 크기 81보다 작아 **causal**하게 안전 동작합니다.

#### 🎯 멀티태스크 학습 의도

- **Pose Head** — 각 시간 t에서 15개 관절의 3D 좌표 회귀 (MPJPE 등 손실)
- **Phase Head** — 같은 시간 t의 운동 국면 3-way 분류 (예: prepare / hold / release)
- **공유 백본** — 두 head가 동일한 TCN 피처를 공유하여 자세와 국면이 상호 정규화됨
- 추론 시 `predict_latest()`는 **윈도우 마지막 프레임 t=T-1** 의 출력만 취합니다.

---

### 🧩 모듈 의존성

```mermaid
graph LR
    Main[live_ai_coach_v4_quality.py]:::main
    Main --> RL[runtime_lifting.py]
    Main --> RQ[runtime_quality.py]
    Main --> DS[dataset.py]
    Main --> Ext1{{cv2 / mediapipe<br/>numpy / dotenv}}:::ext
    Main --> Gem{{Gemini REST API}}:::ext

    RL --> DS
    RL --> Model[model.py]
    RL --> Ckpt[(best.pt)]:::file
    RL --> Torch{{torch}}:::ext

    RQ --> KS[kinematic_scoring.py]
    RQ --> Json[(model.json<br/>Mahalanobis 스코어러)]:::file

    KS --> DS

    Model --> Torch

    classDef main fill:#FFE4B5,stroke:#333,stroke-width:2px
    classDef ext fill:#E8E8E8,stroke:#999,stroke-dasharray: 4 4
    classDef file fill:#D4FFB5,stroke:#333
```

지원 자세:
| 자세 | 한글명 | 학습된 스코어러 |
|---|---|---|
| The Seal | 더 씰 | ✅ `the_seal_mahalanobis_hip_knee_all_v2` |
| Spine Stretch | 스파인 스트레치 | ❌ (각도 fallback) |
| Bridging | 브릿징 | ✅ `bridging_mahalanobis_v1` |

---

## 📁 폴더 구조

```
live_ai_coach_v4_bundle/
├── .env                                    # GOOGLE_API_KEY (직접 채워야 함)
├── .env.example                            # 템플릿
├── pose_landmarker_heavy.task              # MediaPipe 모델 (~30MB)
├── README.md
├── min/
│   └── min_dev_park/
│       └── live_ai_coach_v4_quality.py     # 🎯 메인 실행 스크립트
└── pilates_temporal_lifter/
    ├── __init__.py
    ├── dataset.py                          # JOINT_ORDER, normalize_skeleton 등
    ├── model.py                            # TemporalLifterWithPhaseHead
    ├── kinematic_scoring.py                # mahalanobis_d2, phase_labels 등
    ├── runtime_lifting.py                  # OnlineTemporalLifter (실시간 3D)
    ├── runtime_quality.py                  # load_scorer, score_sequence_3d
    ├── requirements.txt
    └── runs/
        ├── the_seal_mahalanobis_hip_knee_all_v2/model.json
        ├── bridging_mahalanobis_v1/model.json
        ├── the_seal_progress3_angle_causal_v1/best.pt    # 우선 사용 lifter
        └── the_seal_progress3_lift_only_v1/best.pt       # fallback lifter
```

---

## ⚙️ 설치

### 1. Python 환경

Python 3.10+ 권장. 가상환경 사용을 권장합니다.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux
```

### 2. 의존 패키지

```bash
pip install opencv-python mediapipe numpy pandas torch python-dotenv
```

또는 `pilates_temporal_lifter/requirements.txt` 참고:

```bash
pip install -r pilates_temporal_lifter/requirements.txt
```

> 💡 GPU 사용 시 [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/)의 CUDA 빌드 명령으로 설치하세요. CPU만으로도 동작합니다.

### 3. API 키 설정

`.env` 파일을 열고 Gemini API 키를 입력합니다:

```
GOOGLE_API_KEY=여기에_본인_키_입력
```

키 발급: <https://aistudio.google.com/app/apikey>

---

## ▶️ 실행

번들 루트(`live_ai_coach_v4_bundle/`)에서:

```bash
python min/min_dev_park/live_ai_coach_v4_quality.py
```

웹캠 창이 열리면 손 제스처로 조작합니다.

---

## ✋ 사용법 (제스처 UI)

### 1️⃣ 자세 선택 화면
- 화면 중앙의 **3개 박스** 중 하나 위에 손을 위치
- **주먹을 쥐었다 펴면** 클릭 (마우스처럼 동작)

### 2️⃣ 운동 화면
오른쪽 컨트롤 박스 위에 손목을 **1.4초간 정지**시켜 발동:

| 상태 | 사용 가능한 컨트롤 |
|---|---|
| `WAIT_START` | START / RESELECT / QUIT |
| `ACTIVE` (녹화 중) | FINISH / RESELECT / QUIT |
| `REPORT` | START / RESELECT / QUIT |

`ACTIVE` 상태에서 컨트롤 박스 위에 손이 닿아있으면 녹화가 일시정지되어 깨끗한 프레임만 점수화됩니다.

### 3️⃣ 결과 리포트
FINISH 후 콘솔에 다음이 출력됩니다:
- PASS/WARN/FAIL 카운트, 국면별 분포
- 가장 문제된 feature와 worst frames
- **Gemini가 생성한 한국어 코칭 피드백 3문장**

### ⌨️ 키보드 단축키
- `q` — 종료
- `r` — 자세 선택 화면으로 리셋
- `f` — 강제 FINISH 또는 마지막 리포트로 LLM 피드백 재요청

---

## 🧩 의존성 트리 (텍스트)

```
live_ai_coach_v4_quality.py
├── dataset.JOINT_ORDER
├── runtime_lifting.OnlineTemporalLifter
│   ├── dataset.{JOINT_ORDER, build_observation_mask, normalize_skeleton}
│   └── model.{TemporalLifterConfig, TemporalLifterWithPhaseHead}
└── runtime_quality.{load_scorer, score_sequence_3d, summarize_score_rows, format_summary_for_prompt}
    └── kinematic_scoring.{extract_kinematic_features, mahalanobis_d2, phase_labels_for_sequence, rolling_median, status_from_score}
        └── dataset.{PHASE_NAMES_3, PHASE_NAMES_4, coarse_progress_phase_labels, cyclic_anchor_phase_labels, normalize_skeleton, read_keypoint_csv}
```

런타임에 읽는 외부 파일:
- `.env` — GOOGLE_API_KEY
- `pose_landmarker_heavy.task` — MediaPipe 자세 검출 모델 (없으면 자동 다운로드)
- `pilates_temporal_lifter/runs/.../model.json` × 2 — Mahalanobis 스코어러
- `pilates_temporal_lifter/runs/.../best.pt` × 1 — 3D lifter 체크포인트 (causal 우선, 없으면 lift_only)

---

## ⚠️ 주의 사항

- **카메라 권한**이 필요합니다 (Windows: 설정 → 개인정보 보호 → 카메라).
- 프레임이 81개 미만이면 lifter가 첫 프레임을 패딩해 추론합니다 — 초반 몇 초는 정확도가 낮을 수 있습니다.
- 점수화 리포트는 최소 `MIN_REPORT_FRAMES = 20`개의 깨끗한 프레임이 모여야 생성됩니다. 그보다 적으면 fallback 각도 리포트로 대체됩니다.
- Spine Stretch는 학습된 스코어러가 없어 항상 fallback 모드로 동작합니다.

---

## 🔗 원본 위치

이 번들의 원본은 다음 경로에 있습니다:
- 메인 스크립트: `longstone/min/min_dev_park/live_ai_coach_v4_quality.py`
- 라이브러리: `longstone/pilates_temporal_lifter/`
