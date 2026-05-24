# OnPose v6 — 실시간 자세 코칭 AI

> **온디바이스 AI 기반 실시간 필라테스 자세 교정 시스템**
>
> MediaPipe Pose → TemporalLifter(2D→3D) → 분포 기반 채점 → 친근체 LLM 피드백
> 노트북 데스크탑 UI + 모바일/PWA 풀스택을 한 저장소에 통합.

---

## 📦 GitHub 에 올릴 폴더

**이 `onpose_v6/` 폴더만 올리면 됩니다.**

- 상위 `longstone/` 에는 실험/탐색용 폴더가 섞여 있어서, 캡스톤 최종 결과물은 이 `onpose_v6/` 하나에 모두 정리되어 있어요.
- `.gitignore` 가 `__pycache__`, `node_modules`, `.env`, 빌드 산출물 등을 알아서 제외해 줍니다.
- 큰 자산(전문가 영상 `assets/*.mp4`, MediaPipe `pose_landmarker_lite.task` 3MB)은 명시적으로 **추적합니다** — 시연에 바로 필요해서요.

### 새 저장소로 push 하는 방법

```powershell
# 1) 이 폴더로 이동
cd D:\수업\4-1학기\인공지능캡스톤디자인\longstone\onpose_v6

# 2) 단독 git 저장소로 초기화 (longstone 의 git 과 분리)
git init
git add .
git status   # 추적될 파일 목록 점검 (.env / node_modules 가 없는지 확인)
git commit -m "OnPose v6: 데스크탑 + 모바일 풀스택 통합"

# 3) GitHub 에서 새 repo 만들고 URL 복사 → push
git branch -M main
git remote add origin https://github.com/<your-id>/onpose_v6.git
git push -u origin main
```

> ⚠️ `app/backend/.env` 에 Gemini API 키가 들어 있다면 **절대 commit 되지 않게** 했는지 `git status` 로 한 번 더 확인하세요. `.gitignore` 가 막아 두긴 합니다.

---

## 📁 폴더 구조

```
onpose_v6/
├── core/                            # 채점/lifter/feedback 코어 (공용)
│   ├── pose_pipeline.py             #   MediaPipe + TemporalLifter
│   ├── angle_scorer.py              #   rubric 가중치 채점
│   ├── distribution_scorer.py       #   분포(평균±표준편차) 기반 z-score 채점
│   ├── feedback_engine.py           #   친근체 LLM (온/오프라인)
│   ├── occlusion_robust.py          #   smoother / 가려짐 처리
│   └── light_ui.py                  #   NCCOSS 라이트 테마 OpenCV 컴포넌트
├── onpose_v6_coach.py               # 🖥️  데스크탑 OpenCV UI
├── app/                             # 📱 모바일/PWA 풀스택
│   ├── backend/                     #   FastAPI WebSocket (core 직접 import)
│   └── frontend/                    #   React + Vite + TypeScript + PWA
├── assets/                          # 전문가 시범 영상 (3종)
├── reports/                         # 정량 평가 결과 / 자동 산출물
│   ├── pose_stats.json              #   AI Hub 216 데이터셋 44 actor 통계
│   ├── rubric_calibrated.json       #   분포 기반 rubric
│   ├── QUANTITATIVE_REPORT.md       #   FLOPs/Params/Latency/PCK 표
│   └── architecture_paper.png       #   논문용 아키텍처 다이어그램
├── eval/                            # 평가 스크립트 (lifting_accuracy 등)
├── android_skeleton/                # 네이티브 Android 포팅 가이드
├── web_pwa/                         # 정적 PWA 데모 (간단 시연용)
├── requirements.txt                 # 데스크탑 + backend 공용 의존성
├── DEMO_GUIDE.md
├── MOBILE_DEPLOYMENT.md
└── INTEGRATION_V6.md (= app/INTEGRATION_V6.md)
```

---

## ⚡ 빠른 시작

본 시스템은 두 가지 방식으로 동작합니다.

| 방식 | 어디서 보나요? | 추천 시점 |
|---|---|---|
| **A. 모바일 / PWA** (FastAPI + React) | 핸드폰 브라우저 | 시연 / 모바일 데모 |
| **B. 데스크탑 OpenCV** (단일 Python) | 노트북 창 | 빠른 디버깅 / 발표 영상 |

---

## A. 모바일 / PWA 실행

### 사전 준비

| 도구 | 권장 버전 | 비고 |
|---|---|---|
| Python | 3.10–3.12 | anaconda 권장 |
| Node.js | ≥ 20 | npm 포함 |
| 핸드폰 | iOS/Android | 노트북과 **같은 WiFi** 에 연결 |

> Windows 에서 `uvloop` 는 설치되지 않습니다 — `requirements.txt` 에 `sys_platform != "win32"` 마커가 있어서 자동 건너뜁니다.

### 1) 백엔드 (FastAPI)

```powershell
cd D:\...\onpose_v6\app\backend

# 의존성 설치
pip install -r requirements.txt
pip install numpy pandas scipy python-dotenv   # core/* 가 쓰는 추가 패키지

# (선택) Gemini API 키 — 없어도 친근체 오프라인 템플릿으로 동작
#   app/backend/.env  파일 생성:
#   GOOGLE_API_KEY=your_key_here

# 서버 시작
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

성공 로그 예시:
```
[ai_bridge] calibrated rubrics applied from .../rubric_calibrated.json
[ai_bridge] distribution rubrics loaded: ['Spine_Stretch', 'The_Seal', 'Bridging']
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 2) 프런트엔드 (React + Vite)

```powershell
cd D:\...\onpose_v6\app\frontend

npm install         # postinstall 이 MediaPipe WASM 자동 복사
npm run dev -- --host
```

성공하면 여러 IP 가 출력됩니다:
```
➜  Local:   http://localhost:5173/
➜  Network: http://168.131.39.110:5173/   ← 핸드폰에서 접속할 주소
```

### 3) 핸드폰에서 사용

1. 핸드폰을 **노트북과 같은 WiFi** 에 연결
2. Chrome/Safari 에서 `http://<노트북IP>:5173` 접속
3. 카메라 권한 허용
4. **클릭이 필요 없습니다** — 손목을 카드에 1.2초 머무르면 자동 진행:
   - Home: 운동 카드 위에 손 → 선택 + 세션 생성
   - Setup: 33관절 다 잡힌 뒤 "시작" 카드에 손 → `/coaching` 진입
5. 코칭 화면 좌상단에 전문가 영상 PIP 자동 재생
6. 운동 종료 후 Report 화면에서 **내 영상이 재생**되며, 옆/아래로 AI 코치 메시지 + 잘한 점 + 개선점 함께 표시

> 💡 핸드폰 접속이 안 되면 Windows 방화벽에서 5173/8000 인바운드 허용 필요.

### Windows 방화벽 한 줄 (관리자 PowerShell)

```powershell
New-NetFirewallRule -DisplayName "OnPose Dev" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 5173,8000
```

---

## B. 데스크탑 OpenCV 실행

핸드폰 없이 노트북 한 대로 빠르게 데모하고 싶을 때.

```powershell
cd D:\...\onpose_v6
pip install -r requirements.txt

# 표준 시연 (Gemini 친근체)
python onpose_v6_coach.py

# 완전 오프라인 (네트워크/.env 불필요)
python onpose_v6_coach.py --offline

# 라이트 모드 (3D lifter 끄기 — Jetson Nano 등 저사양)
python onpose_v6_coach.py --no-lifter
```

사용 흐름:
1. 자세 선택 (손목으로 카드 호버 1.4초)
2. 3초 안내 → 5초 자동 측정 (손/클릭 동작 없음)
3. 결과: 점수 + 각도 막대 + 친근체 코치 피드백 3문장

---

## 🎯 핵심 기능

| 기능 | 위치 | 비고 |
|---|---|---|
| **2D→3D 리프팅** | `core/pose_pipeline.py` + `pilates_temporal_lifter/` | TCN (dilation [1,2,4,8]), 1.6M params, 130M FLOPs, INT8 ONNX 1.59MB |
| **분포 기반 채점** | `core/distribution_scorer.py` | z-score Gaussian + 각속도 likelihood 가중 평균 |
| **친근체 LLM** | `core/feedback_engine.py` | Gemini 2.5 Flash + 오프라인 템플릿 fallback |
| **호버 트리거 UX** | `app/frontend/src/hooks/useHoverTrigger.ts` | 손목 1.2초 dwell → 자동 선택 (클릭 X) |
| **본인 영상 리플레이** | `app/frontend/src/hooks/useRecorder.ts` | MediaRecorder → Report 화면에서 재생 |
| **NCCOSS 라이트 테마** | `core/light_ui.py` | 흰 카드 + 초록 강조 + 한글 폰트 |

---

## 📊 정량 평가 (`reports/QUANTITATIVE_REPORT.md` 발췌)

| 지표 | 값 | 비고 |
|---|---|---|
| Hip PCK @15° | **89.86%** | 목표 80% 초과 달성 |
| Hip PCK @20° + bone-lock | **94.20%** | 목표 90% 초과 |
| End-phase Verdict Agreement | **100.00%** | 자세 유지 구간에서 GT-Pred 완전 일치 |
| Lifter Params | 1.60 M | INT8 ONNX 1.59 MB |
| Lifter FLOPs / frame | ~130 M | causal, 9-frame window |
| Backend `analyze_frame` | **< 2 ms** | numpy only |
| Total round-trip (LAN) | **50–80 ms** | ≈ 13–20 fps |

상세 평가는 [`reports/QUANTITATIVE_REPORT.md`](reports/QUANTITATIVE_REPORT.md) 및 [`reports/architecture_paper.png`](reports/architecture_paper.png) 참고.

---

## 🏗️ 통합 아키텍처

```
┌─────────────────────┐  WebSocket  ┌──────────────────────────┐
│  Mobile/PWA Browser │ ─────────→  │  FastAPI backend         │
│  React + TS         │  landmarks  │  app/ai_bridge.py        │
│  MediaPipe Web      │  10 fps     │   ↓ imports              │
│  (33 lm × x,y,z,vis)│  ←──────────│  onpose_v6/core/*        │
│                     │  Coaching   │  - distribution_scorer   │
└─────────────────────┘  Frame      │  - angle_scorer          │
                                    │  - feedback_engine       │
                                    └──────────────────────────┘
```

모바일 frontend 는 **클라이언트에서 MediaPipe Web 으로 자세 추출**, 백엔드는 **landmarks 만 받아서 채점**. 영상은 서버로 보내지 않음 → **개인정보 보호** + **저대역폭**.

---

## 🔐 오프라인 동작

`.env` 비워두면 자동으로 친근체 템플릿 모드로 fallback:

```
[ai_bridge] LLM offline mode (no GOOGLE_API_KEY)
```

→ **노트북(backend) + 핸드폰(frontend)이 같은 WiFi 에 연결**되어 있으면 인터넷 없어도 100% 작동.

---

## 📚 추가 문서

- [`INTEGRATION_V6.md`](app/INTEGRATION_V6.md) — backend ↔ core 통합 상세
- [`MOBILE_DEPLOYMENT.md`](MOBILE_DEPLOYMENT.md) — 네이티브 모바일 포팅 로드맵
- [`DEMO_GUIDE.md`](DEMO_GUIDE.md) — 시연 명령어 모음
- [`android_skeleton/`](android_skeleton/) — Kotlin 포팅 가이드

---

## 🧪 정량 평가 재현

```powershell
# The Seal causal lifter (Hip PCK@15° = 89.86%)
python eval/lifting_accuracy.py `
  --pred-csv ../pilates_temporal_lifter/predicted_eval_progress3_angle_causal_v1.csv `
  --gt-csv  ../pilates_temporal_lifter/the_seal_gt3d_trim.csv `
  --pose the_seal --tol-deg 15 `
  --out reports/lifting_accuracy_the_seal_causal.json
```

---

## 📜 라이선스 / 저자

학부 캡스톤 디자인(2026-1) 프로젝트 산출물. 데이터셋은 AI Hub "필라테스 동작 인식" (216 actors, 44 verified) 사용.
