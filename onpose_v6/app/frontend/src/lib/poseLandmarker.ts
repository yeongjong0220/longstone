import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision"

let landmarkerPromise: Promise<PoseLandmarker> | null = null

export function createPoseLandmarker(): Promise<PoseLandmarker> {
  if (landmarkerPromise) return landmarkerPromise
  landmarkerPromise = (async () => {
    // 오프라인 컨셉을 위해 WASM도 로컬 호스팅 (public/mediapipe-wasm/).
    const vision = await FilesetResolver.forVisionTasks("/mediapipe-wasm")
    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/models/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })
  })()
  return landmarkerPromise
}
