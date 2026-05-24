# 📱 OnPose Android 앱 스켈레톤

> Python 백엔드의 `core/angle_scorer.py`, `core/feedback_engine.py`를 Kotlin으로 직역 + ONNX/MediaPipe를 결합한 최소 동작 구조.

이 디렉터리는 **빌드 가능한 Android Studio 프로젝트가 아닌, 핵심 파일 템플릿 모음**입니다. 사용자가 Android Studio 프로젝트를 만든 뒤 해당 파일들을 참고하여 채우면 됩니다.

---

## 1. 의존성 (app/build.gradle.kts)
```kotlin
dependencies {
    // MediaPipe Tasks (Vision)
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
    // ONNX Runtime Mobile
    implementation("com.microsoft.onnxruntime:onnxruntime-mobile:1.18.0")
    // OkHttp (선택: 오프라인이면 불필요)
    // implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")
}
```

## 2. Assets 폴더 구성
`app/src/main/assets/`:
```
pose_landmarker_lite.task    (3 MB)
lifter_causal_int8.onnx      (1.59 MB)
onpose_metadata.json         (~10 KB)
```
모두 [reports/](../reports/) 에서 복사. 총 약 5MB의 추가 용량으로 풀 기능 동작.

## 3. 디렉토리 트리 (제안)
```
app/src/main/java/com/onpose/v6/
├── MainActivity.kt              // 카메라 프리뷰 + 상태 머신
├── pose/
│   ├── PoseDetector.kt          // MediaPipe wrapper
│   ├── TemporalLifter.kt        // ONNX Runtime wrapper (81프레임 슬라이딩)
│   └── AngleCalculator.kt       // hip/knee/trunk 각도
├── scoring/
│   ├── AngleScorer.kt           // angle_scorer.py 직역 (PoseRubric/AngleSpec)
│   ├── PoseRubric.kt            // data class
│   └── PostProcess.kt           // bone-length lock 직역
├── feedback/
│   └── FriendlyCoach.kt         // feedback_engine.py 오프라인 부분 직역
├── ui/
│   ├── ScoreCircleView.kt       // 원형 점수 게이지
│   ├── AngleBarsView.kt         // 가중치 막대
│   └── TimelineView.kt          // 각도 시계열
└── data/
    └── Metadata.kt              // onpose_metadata.json 로더
```

## 4. 핵심 클래스 스니펫

`AngleScorer.kt` (angle_scorer.py 직역):
```kotlin
data class AngleSpec(val nameKr: String, val targetDeg: Float,
                     val toleranceDeg: Float, val weight: Float, val importance: String)

data class PoseRubric(val poseKey: String, val poseNameKr: String,
                      val angles: Map<String, AngleSpec>, val passThreshold: Float = 80f)

object AngleScorer {
    fun scoreFrame(angles: Map<String, Float>, rubric: PoseRubric): FrameScore {
        var weightedSum = 0f; var totalW = 0f
        val details = mutableMapOf<String, AngleDetail>()
        for ((k, spec) in rubric.angles) {
            val v = angles[k] ?: 0f
            val diff = kotlin.math.abs(v - spec.targetDeg)
            val sub = if (diff <= spec.toleranceDeg)
                100f - (diff / spec.toleranceDeg) * 30f
            else
                kotlin.math.max(0f, 70f - ((diff - spec.toleranceDeg) / spec.toleranceDeg) * 70f)
            details[k] = AngleDetail(v, spec.targetDeg, diff, sub, diff <= spec.toleranceDeg)
            weightedSum += sub * spec.weight; totalW += spec.weight
        }
        val score = weightedSum / totalW
        return FrameScore(score, score >= rubric.passThreshold, details)
    }
}
```

`Metadata.kt` — JSON에서 룰 자동 로드 (Python 백엔드와 같은 임계값/가중치 보장):
```kotlin
class Metadata(context: Context) {
    val rubrics: Map<String, PoseRubric>
    val feedbackTemplates: FeedbackTemplates

    init {
        val json = context.assets.open("onpose_metadata.json")
            .bufferedReader().use { it.readText() }
        val root = JSONObject(json)
        // poses 파싱 -> PoseRubric Map
        // feedback_templates 파싱 -> FeedbackTemplates
        ...
    }
}
```

`TemporalLifter.kt` — 81프레임 윈도우 onnx 추론:
```kotlin
class TemporalLifter(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val window = 81; private val joints = 15
    private val frames2D = ArrayDeque<FloatArray>(window)

    init {
        val bytes = context.assets.open("lifter_causal_int8.onnx").readBytes()
        session = env.createSession(bytes, OrtSession.SessionOptions())
    }

    fun appendAndPredict(frame2D: FloatArray): FloatArray? {
        frames2D.addLast(frame2D)
        while (frames2D.size > window) frames2D.removeFirst()
        // 패딩 + normalize_skeleton + observation_mask 계산은 별도 helper
        val inputTensor = buildInputTensor(frames2D.toList())
        val out = session.run(mapOf("x" to inputTensor))
        val pose3d = out[0].value as Array<*>      // (1, T, 15, 3)
        return extractLastFrame(pose3d)
    }
}
```

`FriendlyCoach.kt` — feedback_engine.py 오프라인 부분 직역:
```kotlin
class FriendlyCoach(private val templates: FeedbackTemplates) {
    fun generate(poseKr: String, score: SessionScore): String {
        val s1 = pickPraise(score.verdict, poseKr)
        val s2 = improvementTip(score)
        val s3 = pickCue(score.topIssue) + " " + pickEnding()
        return "$s1 $s2 $s3"
    }
}
```

## 5. UI 와이어프레임

```
┌──────────────────────────┐
│       카메라 프리뷰         │
│                          │
│      (전신이 보이게)        │
│                          │
├──────────────────────────┤
│  [더 씰]  [스트레치]       │  <- 자세 선택 (자동 인식 배지 우측)
│   [브릿징]                 │
├──────────────────────────┤
│  ⏱ 5초 측정 중              │
│  ◯ 87점     ✓✓✗            │
├──────────────────────────┤
│  💬 "거의 다 왔어요!"        │
└──────────────────────────┘
```

## 6. 빌드 + 배포 흐름

1. Android Studio Hedgehog 이상에서 `Empty Activity` 프로젝트 생성
2. `app/src/main/assets/`에 `reports/`의 3개 파일 복사
3. 위 디렉터리 트리대로 Kotlin 파일 생성, 본 README의 스니펫 시작점으로 사용
4. `MainActivity.kt`에서 CameraX로 프리뷰 + `PoseDetector` 호출 → `AngleScorer` → 화면 표시
5. APK 빌드 (`./gradlew assembleRelease`) → 핸드폰에 설치
6. **완전 오프라인 동작 확인 후** Play Store / 직접 배포

## 7. 예상 APK 크기

| 구성 요소 | 크기 |
|---|---|
| 기본 APK (코드 + UI) | ~3 MB |
| pose_landmarker_lite.task | 3 MB |
| lifter_causal_int8.onnx | 1.6 MB |
| onpose_metadata.json | ~10 KB |
| **총 APK** | **~7-8 MB** |

가벼운 자세 코칭 앱으로 일반적인 수준.

## 8. 다음 단계 (Phase 4 완성 단계)

- ✅ Phase 1: Lite 모드 + 오프라인 (이미 노트북에서 동작 확인)
- ✅ Phase 2: ONNX 변환 + INT8 (1.59MB)
- ✅ Phase 3: 메타데이터 JSON 분리 (모바일 친화)
- ⏳ Phase 4: 이 디렉터리 가이드로 실제 Android 앱 빌드 (Kotlin 직역)

Phase 4를 위해 필요한 것:
- Android Studio + Kotlin 경험자 1인
- 1~2주의 개발 기간
- 테스트용 안드로이드 기기

논리는 모두 [core/](../core/)에 있고, 직역 가이드가 본 README에 있으므로 매우 직선적인 작업입니다.
