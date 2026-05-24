import type { PoseLandmarker, PoseLandmarkerResult } from "@mediapipe/tasks-vision"
import { useEffect, useRef, useState } from "react"

import { perf } from "@/lib/perf"
import { createPoseLandmarker } from "@/lib/poseLandmarker"

// MediaPipe NormalizedLandmark: { x, y, z, visibility? }
export type Landmark = [number, number, number, number]

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
            // MediaPipe requires monotonically increasing timestamps.
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
      // Don't close landmarker — it's cached as a singleton for reuse.
    }
  }, [videoRef, enabled])

  return { latestRef, ready, error }
}
