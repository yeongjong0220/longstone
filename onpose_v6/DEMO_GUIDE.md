# 🎬 OnPose v6 — 시연 가이드 (직접 돌려보기)

> 이 문서는 **개발자가 아닌 시연 담당자**가 그대로 복사-붙여넣기 해서 돌릴 수 있도록 명령어를 정리한 가이드입니다.

PowerShell에서 줄바꿈은 백틱(`` ` ``)이고, `\`는 사용 불가입니다. 헷갈리면 **한 줄로** 입력하세요.

---

## 0. 사전 준비 (한 번만)

```powershell
cd D:\수업\4-1학기\인공지능캡스톤디자인\longstone\onpose_v6

# protobuf 충돌 해결 + 의존성 일괄 설치
pip install --upgrade "protobuf>=4.25,<6" "mediapipe>=0.10.14"
pip install -r requirements.txt
```

Gemini를 쓰려면 (오프라인 시연은 안 써도 됨):
```
# longstone/.env 파일에
GOOGLE_API_KEY=your_key_here
```

---

## 1. 라이브 시연 시나리오 (7가지)

### 시연 ① — 표준 시연 (최고 품질, 인터넷 필요)
```powershell
python onpose_v6_coach.py
```
MediaPipe Heavy + TemporalLifter + Gemini 친근체 코치. 발표 시연에 가장 권장.

### 시연 ② — **완전 오프라인 시연** (네트워크 끄고 데모)
```powershell
python onpose_v6_coach.py --offline
```
Gemini 호출 없이 친근체 템플릿만으로 동작. 네트워크 차단 환경에서 그대로 시연 가능.

### 시연 ③ — **모바일/보드 시뮬레이션** (Lite 모드)
```powershell
python onpose_v6_coach.py --lite
```
MediaPipe Lite (3MB) + Lifter off + Offline.  
**여기서 측정된 `Estimated UI FPS`가 모바일 예상 성능의 상한**입니다.

### 시연 ④ — Lifter 검증 (raw vs bone-lock 비교)
```powershell
# Bone-length 후처리 끄고 보기
python onpose_v6_coach.py --no-bone-lock

# 다시 켜고 비교
python onpose_v6_coach.py
```

### 시연 ⑤ — 자동 순환 시연 (전시용)
```powershell
python onpose_v6_coach.py --demo
```
20초마다 자세를 자동으로 바꿔서 무한 반복. 전시 부스에 유용.

### 시연 ⑥ — **TTS 음성 안내** (들으면서 자세 잡기)
```powershell
python onpose_v6_coach.py --voice
```
Windows SAPI / macOS `say` / Linux `espeak`로 자세 안내, 카운트다운, 점수 발표를 한국어 음성으로.

### 시연 ⑦ — **풀세트 (오프라인+음성+녹화 동시)**
```powershell
python onpose_v6_coach.py --offline --voice --record reports/demo_full.mp4
```
네트워크 없이도, 음성 안내 들으면서, 시연 영상까지 한 번에 저장.

---

## 2. 영상 파일로 시연 (웹캠 없이)

### 미리 찍어둔 영상으로 채점/피드백
```powershell
python onpose_v6_coach.py --video "..\your_video.mp4"
```

### 시연 자체를 mp4로 녹화
```powershell
python onpose_v6_coach.py --record "reports\demo_session.mp4"
```
화면 캔버스(카메라+패널+피드백 전체)가 mp4로 저장됩니다. 발표용 영상 자료에 그대로 삽입 가능.

### 둘 다 동시에 (입력은 영상, 출력도 녹화)
```powershell
python onpose_v6_coach.py --video "input.mp4" --record "reports\out.mp4" --offline
```

---

## 3. 단축키 (UI 안에서 사용)

| 키 | 기능 |
|---|---|
| `q` | 종료 |
| `r` | 자세 선택 화면으로 복귀 |
| `h` | 키보드 단축키 도움말 토글 |
| `1` | 더 씰 자세 빠른 선택 |
| `2` | 스파인 스트레치 빠른 선택 |
| `3` | 브릿징 빠른 선택 |
| `s` | 현재 화면 스크린샷 PNG 저장 |

**CLI 옵션 요약:**
| 옵션 | 효과 |
|---|---|
| `--offline` | LLM 호출 안 함 (오프라인 친근체 템플릿) |
| `--lite` | 모바일 시뮬레이션 (Lite + lifter off + offline) |
| `--no-lifter` | 3D lifter 비활성 |
| `--no-bone-lock` | bone-length 후처리 끄기 |
| `--variant {lite,full,heavy}` | MediaPipe 모델 변형 |
| `--video FILE` | 영상 파일을 입력으로 |
| `--record OUT.mp4` | 화면 캔버스를 mp4로 저장 |
| `--demo` | 자동 순환 시연 (20초 간격) |
| `--voice` | TTS 음성 안내 |
| `--no-splash` | 스플래시 화면 건너뛰기 |
| `--camera N` | 카메라 인덱스 |

시연 중 카메라 앞에서 손을 흔들기 어려우면 키보드 단축키로 빠르게 전환하세요.

---

## 4. 정량 평가 (숫자로 보여주기)

### 4-1) Lifter 정확도 평가 (한 줄)
```powershell
python eval/lifting_accuracy.py --pred-csv "..\pilates_temporal_lifter\predicted_eval_progress3_angle_causal_v1.csv" --gt-csv "..\pilates_temporal_lifter\the_seal_gt3d_trim.csv" --pose the_seal --tol-deg 15 --out reports\lifting_accuracy_the_seal_causal.json
```

**기대 출력 (실측):**
```
[Hip  PCK @15deg]                  89.86%   PASS80
[Hip  PCK @20deg]                  91.79%   PASS80 PASS90
[Knee PCK @20deg]                  81.64%   PASS80
[End-phase Verdict Agreement @15] 100.00%   PASS80 PASS90
```

### 4-2) 후처리 효과 비교 (Before vs After)
```powershell
python eval/compare_postprocess.py --pred-csv "..\pilates_temporal_lifter\predicted_eval_progress3_angle_causal_v1.csv" --gt-csv "..\pilates_temporal_lifter\the_seal_gt3d_trim.csv" --extra-pred-csv "..\pilates_temporal_lifter\predicted_eval_progress3_lift_only_v1.csv" --pose the_seal --out reports\postprocess_comparison.json
```

**기대 출력 (실측):**
- Bone-length enforce 단독: `Hip PCK @20° 91.79 → 94.20%`, `MPJPE 0.3394 → 0.3339`

### 4-3) GT vs Pred 골격 비교 영상 생성 (발표 자료)
```powershell
python eval/visualize_lifting.py --pred-csv "..\pilates_temporal_lifter\predicted_eval_progress3_angle_causal_v1.csv" --gt-csv "..\pilates_temporal_lifter\the_seal_gt3d_trim.csv" --out "reports\compare_the_seal.mp4" --fps 20 --apply-postproc
```
출력: `reports/compare_the_seal.mp4` (좌 GT 노란색, 우 Pred 빨간색, 하단 MPJPE 실시간).

### 4-4) ONNX 모바일 변환 + 양자화 + 벤치마크
```powershell
python eval/export_lifter_onnx.py --ckpt "..\pilates_temporal_lifter\runs\the_seal_progress3_angle_causal_v1\best.pt" --out reports\lifter_causal.onnx --verify --quantize --benchmark
```
**기대 출력 (실측):**
```
verify : pose3d max-abs-diff 1.07e-06   match: OK
total  : 6.17 MB (graph 0.07 + weights 6.10)
INT8   : 1.59 MB  (74.2% smaller, 3.87x)
PyTorch CPU      :   5.98 ms
ONNX Runtime CPU :   3.36 ms  (1.78x speedup)
```
결과물:
- `reports/lifter_causal.onnx` + `.onnx.data` (FP32, 6.17 MB)
- `reports/lifter_causal_int8.onnx` (INT8, 1.59 MB) — **모바일 앱에 그대로 번들 가능**

### 4-5) 모바일 이식용 메타데이터 JSON 생성
```powershell
python eval/export_metadata.py
```
출력: `reports/onpose_metadata.json`. 자세 임계값, 가중치, 친근체 템플릿, MediaPipe→Lifter 매핑까지 한 파일에. 모바일 앱은 이 JSON 한 개만 번들하면 같은 채점 로직 재사용 가능.

---

## 5. 결과물이 저장되는 곳

| 폴더/파일 | 내용 |
|---|---|
| `reports/session_YYYYMMDD_HHMMSS_*.json` | 세션별 점수/피드백/latency 스냅샷 |
| `reports/latency_*.json` | 컴포넌트별 성능 측정 |
| `reports/session_history.json` | 모든 세션의 점수 히스토리 (UI 좌측 차트에 표시됨) |
| `reports/screenshot_*.png` | `s` 키로 저장한 화면 |
| `reports/compare_*.mp4` | GT vs Pred 비교 영상 |
| `reports/lifting_accuracy_*.json` | 정량 평가 결과 |
| `reports/postprocess_comparison.json` | 후처리 효과 비교 |

---

## 6. 가장 빠르게 발표 시연하는 한 줄 명령어

```powershell
python onpose_v6_coach.py --offline
```

이 명령어 하나로:
- 네트워크 없어도 됨
- 카메라 자동 인식
- 손 흔들기로 자세 선택
- 5초 캡처
- 친근체 한국어 코치 피드백
- 화면 좌측: 사용자 영상 / 우측 위: 자세 가이드 / 우측 아래: 점수/막대/그래프/코치 멘트

전부 자동으로 흘러갑니다.

---

## 7. 자주 묻는 트러블슈팅

| 증상 | 해결 |
|---|---|
| `ImportError: cannot import name 'runtime_version' from 'google.protobuf'` | `pip install --upgrade "protobuf>=4.25,<6" "mediapipe>=0.10.14"` |
| `--pred-csv` 파싱 에러 (PowerShell) | 백슬래시 `\`를 백틱 `` ` ``로 바꾸거나 한 줄에 |
| 카메라 안 켜짐 | `--camera 1` 또는 `--camera 2`로 다른 인덱스 시도 |
| 한글 깨짐 | `python -X utf8 …`로 실행 (Windows 한국어 cp949 회피) |
| 자세가 인식 안 됨 | 전신이 화면 안에 들어오게 카메라에서 2~3m 떨어지기, 측면이 보이게 |
| 점수가 너무 낮음 | tolerance를 늘려서 시도, `eval/lifting_accuracy.py --tol-deg 20` |

---

## 8. 점수 의미 한눈에

| 가중 점수 | O/X 정확도 | 평가 |
|---|---|---|
| ≥85 | ≥80% | 훌륭해요 — 거의 정자세 |
| 70~85 | 60~80% | 좋아요 — 한 끗만 다듬으면 완벽 |
| 55~70 | 40~60% | 조금 더 — 핵심 각도가 어긋남 |
| <55 | <40% | 다시 도전 — 자세 자체를 다시 잡기 |

각 자세별 채점 기준은 [README.md](README.md)의 채점 방식 섹션 참고.
