# 📱 OnPose 모바일/경량화 포팅 가이드

> **최종 목표**: 핸드폰에서 완전 오프라인으로 동작하는 자세 코칭 앱

---

## 1. 현재 구성요소별 모바일 호환성

| 컴포넌트 | 현재 (Python) | 모바일 이식성 | 비고 |
|---|---|---|---|
| **MediaPipe Pose** | mediapipe (TF Lite backend) | ✅ 매우 좋음 | Google이 모바일용으로 만든 모델 — `.task` 파일을 Android/iOS SDK에서 그대로 사용 |
| **TemporalLifter** | PyTorch (`best.pt`) | ⚠️ ONNX → TFLite 변환 필요 | 모델 자체는 작아서 변환 후 모바일에서 충분히 동작 |
| **각도 채점** (`angle_scorer`) | numpy | ✅ 완벽 | Java/Kotlin/Swift 산술로 직접 포팅 가능 (의존성 0) |
| **친근체 LLM (Gemini)** | HTTP REST | ❌ 오프라인 불가 | `--offline` 모드로 룰베이스 템플릿 사용 → 그대로 모바일 이식 가능 |
| **친근체 템플릿 (offline)** | Python 문자열 | ✅ 완벽 | JSON resource로 모바일 앱에 번들 |
| **OpenCV UI** | OpenCV-Python | ⚠️ Android: OpenCV4Android | UI는 모바일 네이티브로 다시 그리는 게 일반적 |

**결론**: 모든 코어 알고리즘이 모바일 친화. UI만 네이티브로 다시 그리면 됩니다.

---

## 2. 단계별 모바일 포팅 로드맵

### Phase 1 — Lite 모드 (현재 가능, 검증 완료)
```bash
python onpose_v6_coach.py --lite
```
- MediaPipe **Lite** (3MB 모델, Heavy의 1/10)
- TemporalLifter OFF → 2D 각도만 (loss는 있지만 모바일 RAM 절약)
- Offline 친근체 (네트워크 불필요)
- **현재 상태로 Android 폰에 mediapipe-tasks 앱 + 각도 계산 식을 포팅하면 동작**

### Phase 2 — TemporalLifter ONNX 변환 (반나절 작업)
```python
# 변환 예시 스크립트 (별도 작성)
import torch
from pilates_temporal_lifter.model import TemporalLifterWithPhaseHead, TemporalLifterConfig

ckpt = torch.load("pilates_temporal_lifter/runs/the_seal_progress3_angle_causal_v1/best.pt",
                  map_location="cpu")
cfg = TemporalLifterConfig(**ckpt["config"])
model = TemporalLifterWithPhaseHead(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()

dummy = torch.randn(1, 81, 15, 3)
torch.onnx.export(model, dummy, "lifter.onnx",
                  input_names=["x"], output_names=["pose3d", "phase_logits"],
                  dynamic_axes={"x": {0: "batch"}})
```
→ `onnxruntime-mobile` 또는 `tf2onnx`로 TFLite 변환 → Android assets 폴더에 번들.

### Phase 3 — INT8 양자화 (성능 최적화)
```bash
# ONNX dynamic quantization
python -m onnxruntime.quantization.preprocess --input lifter.onnx --output lifter_pre.onnx
python -m onnxruntime.quantization.quantize_dynamic lifter_pre.onnx lifter_int8.onnx
```
**기대 효과**: 모델 크기 6MB → 1.5MB, 추론 속도 2~4배↑ (정확도 손실 1~2%p 내)

### Phase 4 — 모바일 네이티브 앱 (Android/iOS)
**구조 권장:**
```
app/
├── assets/
│   ├── pose_landmarker_lite.task    (3MB)
│   └── lifter_int8.onnx              (1.5MB)
├── src/main/cpp/   (Android NDK) 또는 swift/  (iOS)
│   ├── pose_pipeline.cpp     (MediaPipe + ONNX Runtime 호출)
│   ├── angle_scorer.cpp       (angle_scorer.py의 직역)
│   └── feedback_templates.json  (친근체 템플릿 그대로)
└── ui/   (Jetpack Compose / SwiftUI)
    └── ...
```

---

## 3. FPS / RAM 예측치

| 디바이스 | 모드 | 예상 FPS | RAM | 비고 |
|---|---|---|---|---|
| Galaxy S24 (Snapdragon 8 Gen 3) | Lite + Lifter ONNX INT8 | 25~30 fps | ~120 MB | 풀 기능 |
| Galaxy A시리즈 (중급) | Lite + 2D 각도 only | 15~20 fps | ~60 MB | TemporalLifter off |
| iPhone 13~ | Lite + Lifter Core ML | 25~30 fps | ~100 MB | Apple Neural Engine 활용 |
| Jetson Nano | Heavy + Lifter | 8~12 fps | ~250 MB | 보드 데모용 |
| Raspberry Pi 4 | Lite + 2D only | 5~8 fps | ~80 MB | 최저 사양 |

> 노트북(MX150 등 약 GPU)으로 측정한 `frame_total` mean ms를 `reports/latency_*.json`에서 확인할 수 있습니다.

---

## 4. 오프라인 보장 매트릭스

| 요소 | 오프라인 | 비고 |
|---|---|---|
| MediaPipe 추론 | ✅ | 모델 파일을 앱에 번들 |
| TemporalLifter 추론 | ✅ | 체크포인트 번들 |
| 각도 채점 / O/X | ✅ | 순수 계산 |
| 친근체 코칭 (template) | ✅ | JSON 리소스 |
| Gemini 호출 | ❌ | 옵션 — 네트워크 있을 때만 |

→ `--offline` 또는 `--lite` 플래그로 **노트북에서도 완전 오프라인 시연 가능**. 같은 코어 로직이 모바일에서도 그대로 동작.

---

## 5. 모바일 UI 와이어프레임 (제안)

```
┌──────────────────────┐
│  📷 카메라 프리뷰     │
│                      │
│   (전신 보이게)       │
│                      │
├──────────────────────┤
│ ① 더씰  ② 스트레치    │
│      ③ 브릿징         │
├──────────────────────┤
│  안내: 자세를 취하세요 │
│      ⏱ 3.2s          │
├──────────────────────┤
│  점수 ⬤ 87           │
│  무릎 ✓  허리 ✓  엉덩이 △ │
│  💬 "거의 다 왔어요!"  │
└──────────────────────┘
```

핵심: **세로 단일 화면, 카메라가 가장 큼, 채점 결과는 하단 카드로 슬라이드.**

---

## 6. 즉시 시도 가능한 가벼운 시연

```bash
# 노트북에서 핸드폰처럼 가볍게 시연
python onpose_v6_coach.py --lite

# 다 끄고 정말 최소 사양으로
python onpose_v6_coach.py --lite --no-bone-lock
```

이 두 모드에서 측정된 `frame_total` 시간이 **모바일에서 예상되는 latency의 상한**입니다 (모바일 GPU는 보통 비슷하거나 빠름).
