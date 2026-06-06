import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision"

let landmarkerPromise: Promise<PoseLandmarker> | null = null

export function createPoseLandmarker(): Promise<PoseLandmarker> {
  if (landmarkerPromise) return landmarkerPromise
  landmarkerPromise = (async () => {
    // 오프라인 컨셉을 위해 WASM도 로컬 호스팅 (public/mediapipe-wasm/).
    const vision = await FilesetResolver.forVisionTasks("/mediapipe-wasm")
    // 모델 선택 — Lite 는 jitter 가 심해서 Full 기본 (멘토 자료: 21 fps, 33 lm, 안정성 ↑).
    // 폰 성능이 부족하면 .env 에 VITE_MP_MODEL=lite 으로 가벼운 모델 강제 가능.
    const modelChoice = (import.meta.env.VITE_MP_MODEL ?? "full").toLowerCase()
    const modelAssetPath =
      modelChoice === "lite"
        ? "/models/pose_landmarker_lite.task"
        : "/models/pose_landmarker_full.task"

    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.6,
      minPosePresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    })
  })()
  return landmarkerPromise
}
