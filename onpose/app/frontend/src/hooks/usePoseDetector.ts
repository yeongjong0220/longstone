import type { PoseLandmarker, PoseLandmarkerResult } from "@mediapipe/tasks-vision"
import { useEffect, useRef, useState } from "react"

import { perf } from "@/lib/perf"
import { createPoseLandmarker } from "@/lib/poseLandmarker"

// MediaPipe NormalizedLandmark: { x, y, z, visibility? }
export type Landmark = [number, number, number, number]

/**
 * Pose detector hook — MediaPipe Pose Landmarker (Full 기본).
 *
 * 멘토 자료의 Pixel5 fps 비교 결과:
 *   - MediaPipe Lite     ~25 fps  / 흔들림 큼 (정확도 ↓)
 *   - MediaPipe Full     ~21 fps  / 안정성 ↑ ← 현 기본
 *   - MediaPipe Heavy    ~8  fps  / 폰 실시간 X
 *   - MoveNet Lightning  ~34 fps  / 17 lm only, 매핑 wrapper 필요
 *   - MoveNet Thunder    ~12 fps  / 17 lm only
 *
 * Full + backend Smoother + new lifter (3 동작 학습) 조합이 최적.
 * 모델 종류는 .env 의 VITE_MP_MODEL=lite|full 로 override 가능.
 */
export function usePoseDetector(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  enabled: boolean,
  onResult?: (landmarks: Landmark[]) => void,
) {
  const latestRef = useRef<Landmark[]>([])
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const onResultRef = useRef(onResult)
  onResultRef.current = onResult

  useEffect(() => {
    if (!enabled) return
    let landmarker: PoseLandmarker | null = null
    let raf = 0
    let cancelled = false
    let lastTs = -1

    createPoseLandmarker()
      .then((l) => {
        if (cancelled) return
        landmarker = l
        setReady(true)
        const loop = () => {
          if (cancelled) return
          const video = videoRef.current
          if (video && video.readyState >= 2 && landmarker) {
            const ts = performance.now()
            // MediaPipe 는 monotonically increasing timestamp 요구
            if (ts > lastTs) {
              lastTs = ts
              try {
                const t0 = performance.now()
                const result: PoseLandmarkerResult = landmarker.detectForVideo(video, ts)
                perf.inference.push(performance.now() - t0)
                if (result.landmarks && result.landmarks.length > 0) {
                  const lms: Landmark[] = result.landmarks[0].map((p) => [
                    p.x,
                    p.y,
                    p.z,
                    p.visibility ?? 0,
                  ])
                  latestRef.current = lms
                  onResultRef.current?.(lms)
                } else {
                  latestRef.current = []
                  onResultRef.current?.([])
                }
              } catch (e) {
                setError((e as Error).message)
              }
            }
          }
          raf = requestAnimationFrame(loop)
        }
        raf = requestAnimationFrame(loop)
      })
      .catch((e: Error) => {
        setError(e.message)
      })

    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
      // landmarker 는 싱글톤 (재사용 위해 cache) — close 안 함
    }
  }, [videoRef, enabled])

  return { latestRef, ready, error }
}
