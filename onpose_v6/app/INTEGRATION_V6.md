# onpose_v6/app — Mobile React + FastAPI 통합

OnPose v6 채점/코칭 엔진을 활용하는 모바일/PWA 풀스택. 기존 `longstone/ui_ux/`에서 `onpose_v6/app/`로 이동 (2026-05-24).

**폴더 구조:**
```
onpose_v6/
├── core/                  # 채점/lifter/feedback 코어
├── reports/               # pose_stats.json, rubric_calibrated.json
├── assets/                # 전문가 영상 mp4
├── onpose_v6_coach.py     # OpenCV 데스크탑 UI
└── app/                   # 🆕 모바일 풀스택
    ├── backend/           # FastAPI WebSocket
    │   └── app/ai_bridge.py  ← onpose_v6/core/* 직접 import
    └── frontend/          # React + Vite + TS + PWA
```

```
┌─────────────────────┐  WebSocket  ┌──────────────────────────┐
│  Mobile/PWA Browser │ ────────→  │  FastAPI backend         │
│  React + TS         │  landmarks │  app/ai_bridge.py        │
│  MediaPipe Web      │            │   ↓ imports              │
│  (33 lm × x,y,z,vis)│  ←────────│  onpose_v6/core/*        │
│  매 100ms = 10fps   │  Coaching  │  - distribution_scorer   │
└─────────────────────┘  Frame     │  - feedback_engine       │
                                    │  - angle_scorer (rubrics)│
                                    │  - pose_classifier       │
                                    └──────────────────────────┘
```

## 통합 지점

| 파일 | 역할 | 변경 |
|---|---|---|
| `backend/app/ai_bridge.py` | **mock → v6 호출** | `analyze_frame` / `generate_coaching` 전면 교체. v6 `core/*` 모듈 lazy import (MediaPipe stub 주입해서 백엔드는 가벼움) |
| `backend/app/main.py` | `_build_report`에 `angle_history` 전달 | v6 분포 채점이 시퀀스 필요 |
| `backend/app/sessions.py` | 기존 그대로 — `frame_idx`, `exercise_id` 등 자동 누적 | (변경 없음) |
| `frontend/*` | 기존 그대로 — MediaPipe Web으로 landmarks 추출 후 WebSocket 전송 | (변경 없음) |

## v6 통합 효과

| 항목 | Before (mock) | After (v6) |
|---|---|---|
| 채점 방식 | 항상 score=85 고정 | **분포 기반 z-score + 각속도 likelihood** (전문가 mu±sigma 비교) |
| 자세별 정답 각도 | 하드코딩 (90°) | **AI Hub 216 데이터셋 44 actor 통계** (`pose_stats.json`) |
| 친근체 피드백 | "허리를 잘 유지하고 있어요" 고정 | **Gemini 2.5 Flash** (또는 오프라인 친근체 템플릿) |
| 각도 계산 | 단일 좌측 관절 | **좌/우 visibility 가중 평균** (옆모습 robust) |
| Exercise 종류 | 3종 (the_seal/spine_stretch/bridging) | **동일 — v6 RUBRICS와 1:1 매핑** |

## Exercise ↔ Pose Key 매핑

```python
_EXERCISE_TO_POSE = {
    "the_seal":     "The_Seal",
    "spine_stretch": "Spine_Stretch",
    "bridging":     "Bridging",
}
```

`backend/app/main.py`의 `_EXERCISES`와 `onpose_v6/core/angle_scorer.RUBRICS`가 자동 일치.

## 통신 스키마 (변경 없음)

WebSocket 송신 (frontend → backend) — `LandmarkFrame`:
```json
{
  "t": 1716000000.123,
  "landmarks": [
    [0.5, 0.5, 0.0, 1.0],
    ...  // 33 keypoints × [x, y, z, visibility]
  ]
}
```

WebSocket 수신 (backend → frontend) — `CoachingFrame`:
```json
{
  "t": 1716000000.123,
  "phase": "core",
  "rep_count": 3,
  "set_count": 0,
  "angles": {"hip": 172.1, "knee": 179.8, "trunk": 90.5},
  "score": 87,
  "status": "good",
  "feedback": [
    {"level": "ok",   "msg": "핵심 각도 모두 정답 범위 안에 있어요"},
    {"level": "warn", "msg": "무릎 신전 5° 차이 — 더 펴 주세요"}
  ],
  "_dbg": {"analyze_ms": 1.85}
}
```

세션 종료 후 코칭 리포트 (`GET /api/sessions/{id}/report`) — `Report`:
```json
{
  "score_avg": 83,
  "good_points": ["고관절 굴곡이(가) 정답 범위 안에 잘 들어왔어요 (95%)"],
  "improvements": ["상체 굴곡을(를) 평균 8° 더 조절해 보세요"],
  "llm_msg": "스파인 스트레치 정말 잘 따라오셨어요! 무릎을 한 끗만 ..."
}
```

## 실행 (개발 환경)

### 1. backend
```powershell
cd D:\수업\4-1학기\인공지능캡스톤디자인\longstone\onpose_v6\app\backend
pip install -r requirements.txt
# OnPose v6 의존성 (v6/requirements.txt 와 동일)
pip install numpy pandas scipy python-dotenv
# Gemini 사용 시 .env에 GOOGLE_API_KEY 설정 (없으면 오프라인 친근체 자동)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. frontend
```powershell
cd D:\수업\4-1학기\인공지능캡스톤디자인\longstone\onpose_v6\app\frontend
pnpm install
pnpm dev --host
# 같은 네트워크의 핸드폰 브라우저에서: http://<노트북IP>:5173
```

### 3. 핸드폰에서 사용 (PWA)
1. 핸드폰 Chrome/Safari에서 `http://<노트북IP>:5173` 접속
2. "홈 화면에 추가" → PWA 설치 (offline 동작은 ServiceWorker)
3. 첫 사용 시 카메라 권한 허용
4. Home → 자세 선택 → Coaching 화면에서 5초 측정

## 부하/지연

| Per-frame (frontend → backend → frontend) | 측정값 |
|---|---|
| MediaPipe Web 추론 (frontend) | 30–60 ms (디바이스별) |
| WebSocket round-trip | 5–15 ms (LAN) |
| backend `analyze_frame` | **<2 ms** (v6 거의 numpy) |
| **총 사용자 체감** | **~50–80 ms (≈ 13–20 fps)** |

세션 종료 코칭 리포트:
- offline 친근체 템플릿: **<10 ms**
- Gemini API: ~1.5 s (네트워크 의존)

## 완전 오프라인 시연

`.env` 비워두면 백엔드가 자동으로 친근체 템플릿 모드로 fallback:
```
[ai_bridge] calibrated rubrics applied from .../rubric_calibrated.json
[ai_bridge] distribution rubrics loaded: ['Spine_Stretch', 'The_Seal', 'Bridging']
```

→ **노트북(backend) + 핸드폰(frontend)이 같은 WiFi에 연결**되면 인터넷 없어도 100% 작동.

## 모바일 풀 네이티브 포팅 (장기)

현재는 frontend(웹) + backend(노트북) 모드. 완전 온디바이스로 가려면:

1. `core/angle_scorer.py`의 가중치/임계값 → JSON (`onpose_metadata.json`) 이미 export 완료
2. Friendly LLM offline 템플릿 → JSON 리소스 번들
3. ONNX lifter(`lifter_causal_int8.onnx` 1.59MB) 모바일 추론

→ `onpose_v6/android_skeleton/` 가이드 참고. 동일 채점 로직을 Kotlin/Swift로 직역하면 됨.

## Smoke test 결과

```
frame 0: score= 22  status=err   phase=ready  fb=[warn, warn]
frame 1: score= 22  status=err   phase=ready
...
--- generate_coaching ---
score_avg : 33
good      : ['무릎 신전이(가) 정답 범위 안에 잘 들어왔어요 (100%)']
improve   : ['고관절 굴곡을(를) 평균 111° 더 조절해 보세요', ...]
llm_msg   : 스파인 스트레치 동작 정말 잘 따라오셨어요! ...
```

서있는 자세에 spine_stretch rubric을 매겨봤더니 22점 — 정상 동작 (rubric이 앉아서 굽힌 자세를 기대).

## 향후 개선 후보

| 항목 | 상태 |
|---|---|
| frontend에서 자세 자동 인식 (v6 `classify_pose` 그대로 호출 가능) | 인터페이스 준비됨, frontend wiring만 추가 |
| 결과 화면 영상 리플레이 | frontend는 `MediaRecorder API`로 5초 캡처 가능 |
| 전문가 영상 PIP | `onpose_v6/assets/*.mp4`를 `frontend/public/videos/`로 복사 |
| ServiceWorker로 frontend 자체를 PWA 오프라인 | `frontend/public/manifest.webmanifest` 이미 있음 |
