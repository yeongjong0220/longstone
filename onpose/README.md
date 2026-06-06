# OnPose v10

필라테스 자세 코칭 시스템. 카메라 영상에서 2D pose 추출, TCN 기반 lifter 로 3D 좌표 추정, 전문가 분포 기반 채점, 한국어 코칭 메시지 생성.

## 구성

```
onpose_v10/
├── README.md
├── core/                            v6 부터 유지되는 채점/피드백/후처리 모듈
│   ├── angle_scorer.py
│   ├── distribution_scorer.py
│   ├── feedback_engine.py           3-tier LLM fallback (Gemini → ollama → template)
│   ├── occlusion_robust.py
│   ├── pose_pipeline.py
│   └── ...
├── app/
│   ├── backend/                     FastAPI + WebSocket
│   │   └── app/ai_bridge.py         landmarks → angles → score → feedback
│   └── frontend/                    React + Vite + TypeScript (PWA)
│       └── src/lib/poseLandmarker.ts  MediaPipe Pose Landmarker 통합
├── lifter/                          학습 코드 + 가중치
│   ├── dataset.py                   Neck-Hip 정규화 + GT 시간축 스무딩
│   ├── train_pilates_temporal_lifter.py
│   ├── manifests/all_v10.jsonl      AI Hub 216 — 3 자세, 6,288 clips
│   └── runs/3poses_v10_perpose_gpu/
│       ├── best.pt                  PyTorch 체크포인트 (6.45 MB)
│       ├── history.json             30 epoch 학습 metric
│       └── split_actors.json        per-pose actor split 기록
└── reports/
    ├── lifter_causal_int8.onnx      INT8 양자화 ONNX (1.59 MB)
    ├── lifter_causal.onnx           FP32 ONNX (6.13 MB)
    ├── pose_stats.json              전문가 44 actor 각도/각속도 분포
    └── rubric_calibrated.json       자세별 정답 범위
```

## 파이프라인

```
Webcam (browser)
   ↓
MediaPipe Pose Landmarker (Full, 33 lm + visibility)        21 fps @ Pixel 5
   ↓ WebSocket
Landmark2DSmoother  (EMA α=0.55, hold 8 frame, use_mirror=False)
   ↓
MediaPipe 33 → Lifter 15 인덱스 매핑
   ↓
TemporalLifter (causal TCN, sliding window 81)             1.59 MB INT8 ONNX
   ├ Pose Head  → (T, 15, 3) 3D 좌표
   └ Phase Head → (T, 4) 단계 logits
   ↓
Neck-Hip 정규화 + visibility 가중 좌/우 각도 (hip / knee / trunk)
   ↓
Sanity check (|3D - 2D| < 40°)
   ↓
Distribution scorer (z-score + 각속도 likelihood)
   ↓
Session aggregator → generate_feedback (3-tier)            → Report
```

## 학습 모델

| 항목 | 값 |
|---|---|
| 아키텍처 | TemporalLifterWithPhaseHead (Causal 1D-TCN, dilations [1,2,4,8]) |
| 입력 | (B, 81, 15, 3)  — batch × time × joints × (x, y, observation_mask) |
| 출력 | pose3d (B, 81, 15, 3), phase_logits (B, 81, 4) |
| 파라미터 | 1.60 M |
| FLOPs / window | ~130 M |
| 학습 시간 | RTX 4080, 30 epoch, 약 1시간 43분 |
| 정규화 | Hip → origin / scale = ‖Neck − Hip‖ / Neck-Hip → +y 회전 정렬 |
| GT 전처리 | Savitzky-Golay window=7, polyorder=2 |
| 출력 검증 | PyTorch ↔ ONNX max abs diff = 1e-6 |

학습 데이터 (AI Hub 216 — Pilates Mat):

| Pose | clips | train actors | val actor | test actor |
|---|---|---|---|---|
| Bridging | 1,648 | 59 | actorP061 | actorP085 |
| Spine Stretch | 4,360 | 80 | actorP108 | actorP061 |
| The Seal | 280 | 10 | actorP064 | actorP085 |
| total | 6,288 | 149 (unique) | 3 | 2 |

학습 결과 (epoch 30):

| | Train | Val | Test |
|---|---|---|---|
| MPJPE (정규화 단위) | 0.176 | 0.249 | 0.148 |
| Phase accuracy | — | 89.6% | 90.5% |

best.pt 는 val 최저 시점(epoch 4)에서 저장된 체크포인트.

## 채점

전문가 44 actor 의 동작별 각도 분포 통계 (`reports/pose_stats.json`) 와 비교한다.

```
z = |측정 각도 − μ| / σ
각도 점수 = 100 × exp(−z² / 4.5)
각속도 점수 = exp(−(Δθ − μ_v)² / (2 σ_v²)) × 100
프레임 점수 = 0.8 × 각도 점수 + 0.2 × 각속도 점수
세션 점수 = mean(프레임 점수)
O/X accuracy = (점수 ≥ 80인 프레임) / (전체 프레임)
```

## LLM (코칭 메시지 생성)

3-tier fallback. 위에서부터 시도하고 실패 시 다음으로 내려간다.

| Tier | mode 문자열 | 조건 | latency |
|---|---|---|---|
| 1 | `online_gemini` | 인터넷 + `GOOGLE_API_KEY` | 1–3 s |
| 2 | `offline_local_<model>` | ollama 데몬 + 사전 pull 된 모델 (기본 `gemma2:2b`) | 2–3 s |
| 3 | `offline_rule` | 항상 동작 | <10 ms |

오프라인 환경에서도 자연어 코칭이 동작하도록 ollama 통합. 노트북 backend 위에서 Gemma 2B Q4 (1.6 GB) 가 CPU 추론한다.

## 사전 준비

| 항목 | 버전 |
|---|---|
| Python | 3.11–3.13 |
| Node.js | 20+ |
| Ollama (옵션, 오프라인 LLM) | 0.5+ |
| Gemma 2B Q4 (옵션) | `ollama pull gemma2:2b` |
| MediaPipe Pose Landmarker Full | `app/frontend/public/models/pose_landmarker_full.task` (~9 MB) |

## 실행

### Backend

```powershell
cd app\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

성공 시 로그:

```
[ai_bridge] calibrated rubrics applied from .../reports/rubric_calibrated.json
[ai_bridge] distribution rubrics loaded: ['Spine_Stretch', 'The_Seal', 'Bridging']
INFO:     Application startup complete.
```

첫 frame 수신 시 추가 로그:

```
[ai_bridge] Landmark2DSmoother 활성화 (alpha=0.55, hold=8)
[ai_bridge] lifter loaded from lifter_causal_int8.onnx
```

### Frontend

```powershell
cd app\frontend
npm install
npm run dev
```

http://localhost:5173 (또는 다른 포트) 에서 동작.

### Ollama (오프라인 LLM)

```powershell
ollama serve            # Windows 는 설치 시 자동 startup 등록
ollama pull gemma2:2b   # 1.6 GB
```

## 환경변수

| 변수 | 기본 | 설명 |
|---|---|---|
| `GOOGLE_API_KEY` | (없음) | Gemini 사용 시 |
| `USE_LOCAL_LLM` | `1` | `0` 이면 ollama 시도 안 함 |
| `LOCAL_LLM_MODEL` | `gemma2:2b` | ollama 모델 이름 |
| `LOCAL_LLM_HOST` | `http://localhost:11434` | ollama 서버 주소 |
| `ONPOSE_USE_LIFTER` | `1` | `0` 이면 lifter 무시, 2D direct 만 사용 |
| `ONPOSE_LIFTER_SANITY_DEG` | `40` | lifter 출력이 2D 와 이 각도 이상 차이나면 reject |
| `VITE_MP_MODEL` | `full` | `lite` 로 바꾸면 가벼운 MediaPipe 모델 |

## 학습 재현

```powershell
cd lifter

# 1) manifest
python build_manifest.py `
  --data-root "D:\dataset\216.필라테스 동작 데이터\01-1.정식개방데이터" `
  --out manifests\all_v10.jsonl

# 2) 학습 (RTX 4080 기준 30 epoch / ~1.7 시간)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python train_pilates_temporal_lifter.py `
  --manifest manifests\all_v10.jsonl `
  --outdir runs\3poses_v10_perpose_gpu `
  --pose-filter "Bridging,Spine Stretch,The Seal" `
  --per-pose-split --test-per-pose 1 --val-per-pose 1 `
  --epochs 30 --batch-size 32 --window-size 81 --stride 27 `
  --device cuda --num-workers 4
```

## ONNX export

```powershell
python ..\..\onpose_v7\eval\export_lifter_onnx.py `
  --ckpt lifter\runs\3poses_v10_perpose_gpu\best.pt `
  --out reports\lifter_causal.onnx `
  --verify --quantize --benchmark
```

결과: FP32 6.13 MB, INT8 1.59 MB (3.87× 축소), CPU 추론 1.47 ms/window.

## 변경 사항 (v8 → v10)

| | v8 | v10 |
|---|---|---|
| Lifter 학습 자세 | The Seal 만 (v6 위 상속) | Bridging + Spine Stretch + The Seal |
| 좌표 정규화 | root=Hip, scale=median(shoulder/torso/hip) | Hip + torso scale + Neck-Hip 축 회전 정렬 |
| GT 전처리 | 없음 | Savitzky-Golay window=7 |
| Frame 모델 (frontend) | MediaPipe Pose Lite | MediaPipe Pose Full |
| LLM | Gemini → template (2-tier) | Gemini → ollama → template (3-tier) |
| Lifter sanity threshold | 25° (자주 reject) | 40° |

## 라이선스

학부 캡스톤 디자인 (2026-1) 산출물. 데이터셋은 AI Hub "필라테스 동작 인식" 216 — 검증된 actor subset 사용.
