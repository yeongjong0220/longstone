/**
 * Stub for @mediapipe/pose.
 *
 * @tensorflow-models/pose-detection 패키지가 BlazePose detector 를 위해
 * @mediapipe/pose 를 자동 import 하는데, 그 패키지의 CommonJS export 가
 * Vite (rolldown) 의 분석에서 깨짐.
 *
 * 우리는 MoveNet 만 사용 → BlazePose 코드 path 는 절대 실행 안 됨 →
 * 이 stub 으로 alias 해서 import 만 만족시키면 충분.
 */
export class Pose {
  constructor(..._args: unknown[]) {
    throw new Error("BlazePose is stubbed out. Use MoveNet instead.")
  }
}
export const VERSION = "stub-0.0.0"
export default { Pose, VERSION }
