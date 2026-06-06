/**
 * MoveNet Thunder (TF.js) detector — MediaPipe 대체.
 *
 * 핸드폰 단독 동작 + 가려짐 강건 + ~30fps WebGL.
 * 출력 17 COCO keypoint → backend 호환 MediaPipe 33 Landmark 로 매핑.
 *
 * 사용:
 *   const det = await createMoveNetDetector()
 *   const poses = await det.estimatePoses(videoEl)
 *   const lms = cocoToMpLandmarks(poses[0].keypoints, video.videoWidth, video.videoHeight)
 */
import "@tensorflow/tfjs-backend-webgl"
import * as tf from "@tensorflow/tfjs-core"
import * as poseDetection from "@tensorflow-models/pose-detection"

let detector: poseDetection.PoseDetector | null = null
let pending: Promise<poseDetection.PoseDetector> | null = null

export async function createMoveNetDetector(): Promise<poseDetection.PoseDetector> {
  if (detector) return detector
  if (pending) return pending
  pending = (async () => {
    await tf.setBackend("webgl")
    await tf.ready()
    // Pixel5 기준 fps 비교 (멘토 자료 참고):
    //   THUNDER     ~12 fps  (정확도 ↑, 폰 빠듯)
    //   LIGHTNING   ~34 fps  (가장 가벼움, 정확도는 lifter 후처리로 보완)
    // 모바일 단독 동작이 목표라 LIGHTNING 기본. desktop 에선 Thunder 도 OK.
    const useThunder = (import.meta.env.VITE_MOVENET_THUNDER ?? "0") === "1"
    const d = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: useThunder
          ? poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
          : poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true,
        minPoseScore: 0.25,
      },
    )
    detector = d
    return d
  })()
  return pending
}

export type Landmark = [number, number, number, number] // [x_norm, y_norm, z, visibility]

/** COCO 17 → MediaPipe 33 (관심 있는 키만 매핑, 나머지는 0,0,0,0). */
const COCO_TO_MP: Record<number, number> = {
  0: 0,    // nose
  5: 11,   // left_shoulder
  6: 12,   // right_shoulder
  7: 13,   // left_elbow
  8: 14,   // right_elbow
  9: 15,   // left_wrist
  10: 16,  // right_wrist
  11: 23,  // left_hip
  12: 24,  // right_hip
  13: 25,  // left_knee
  14: 26,  // right_knee
  15: 27,  // left_ankle
  16: 28,  // right_ankle
}

/**
 * MoveNet COCO 17 keypoint → MediaPipe 33 Landmark.
 * 매핑되지 않은 33 슬롯은 visibility 0 → backend Landmark2DSmoother 가 hold/제외.
 *
 * 좌표는 입력 영상 픽셀 단위 → [0,1] 정규화.
 */
export function cocoToMpLandmarks(
  keypoints: { x: number; y: number; score?: number }[],
  videoWidth: number,
  videoHeight: number,
): Landmark[] {
  const out: Landmark[] = Array.from({ length: 33 }, () => [0, 0, 0, 0])
  const w = Math.max(1, videoWidth)
  const h = Math.max(1, videoHeight)
  for (let i = 0; i < keypoints.length; i++) {
    const mp = COCO_TO_MP[i]
    if (mp === undefined) continue
    const kp = keypoints[i]
    if (!kp) continue
    out[mp] = [kp.x / w, kp.y / h, 0, kp.score ?? 0]
  }
  return out
}

export function resetMoveNetDetector(): void {
  if (detector) {
    detector.dispose?.()
    detector = null
  }
  pending = null
}
