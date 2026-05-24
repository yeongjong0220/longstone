# 🌐 OnPose Web PWA 스켈레톤

브라우저만으로 동작하는 자세 코칭 데모. 핸드폰 카메라로 그대로 시연 가능.

## 1. 즉시 실행 (로컬 정적 서버)

```powershell
# Python 내장 서버
cd web_pwa
python -m http.server 8000

# 그다음 핸드폰/노트북 브라우저에서
# http://<노트북 IP>:8000  접속
```

또는 더 간단히:
```powershell
npx -y serve web_pwa
```

## 2. PWA로 설치 (홈 화면에)

- iOS Safari: 공유 메뉴 → "홈 화면에 추가"
- Android Chrome: 메뉴 → "앱 설치"
- 설치 후 네트워크 없이도 ServiceWorker가 캐시한 자원으로 동작

## 3. 현재 상태

이 스켈레톤은 **카메라 + UI 흐름 데모**입니다. 실제 추론 통합을 위해 추가해야 할 것:

| 컴포넌트 | 라이브러리 | 가이드 |
|---|---|---|
| MediaPipe Pose | `@mediapipe/tasks-vision` | [공식 문서](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/web_js) |
| ONNX Lifter | `onnxruntime-web` | `import * as ort from 'onnxruntime-web'` |
| 채점 로직 | (직역) | [core/angle_scorer.py](../core/angle_scorer.py) → JS |
| 메타데이터 | fetch | `fetch('../reports/onpose_metadata.json')` |

### 통합 가이드 스니펫

```js
// MediaPipe Pose Landmarker
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm");
const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath: 'pose_landmarker_lite.task'   // 같은 폴더에 둠
  },
  runningMode: 'VIDEO',
});

// 매 frame
const result = poseLandmarker.detectForVideo(video, performance.now());
const angles = computeAngles(result.landmarks[0]);

// ONNX Lifter
import * as ort from 'onnxruntime-web';
const session = await ort.InferenceSession.create('lifter_causal_int8.onnx',
                                                  { executionProviders: ['wasm'] });
const x = new ort.Tensor('float32', flatInput, [1, 81, 15, 3]);
const out = await session.run({ x });
const pose3d = out.pose3d.data;   // (1, T, 15, 3)
```

## 4. 폴더 구조

```
web_pwa/
├── index.html          # 카메라 + UI 흐름 (현재 스켈레톤)
├── manifest.json       # PWA 매니페스트
├── sw.js               # ServiceWorker (오프라인 캐시)
├── README.md           # 이 문서
└── (추가 필요)
    ├── pose_landmarker_lite.task    ← reports/ 에서 복사
    ├── lifter_causal_int8.onnx       ← reports/ 에서 복사
    └── onpose_metadata.json          ← reports/ 에서 복사
```

## 5. 성능 예상치

| 기기 | FPS | 추론 위치 |
|---|---|---|
| iPhone 13~ | 20-30 | Safari WebAssembly |
| Galaxy S22~ | 18-25 | Chrome WebAssembly |
| 데스크탑 Chrome | 30+ | WebAssembly / WebGL |
| 저사양 폰 (4년 이상) | 10-15 | WASM 단일 스레드 |

## 6. Android 네이티브 vs PWA

| 항목 | PWA | Android 네이티브 |
|---|---|---|
| 개발 비용 | 낮음 (HTML+JS) | 중간 (Kotlin) |
| 성능 | WASM 한계 | 네이티브 풀파워 |
| 설치 마찰 | 매우 낮음 (URL만) | Play Store 거쳐야 |
| 오프라인 | ServiceWorker로 가능 | 완전 |
| 카메라 권한 | 매번 (HTTPS 필요) | 한 번 |
| 배포 속도 | 즉시 | 심사 1-3일 |

**권장**: PWA로 빠른 검증 → 좋은 반응이면 Android 네이티브로 가는 2단계 전략.
