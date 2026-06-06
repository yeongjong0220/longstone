import { useCallback, useEffect, useRef, useState } from "react"

export type CameraStatus = "idle" | "requesting" | "ready" | "denied" | "error"

export function useCamera(enabled: boolean = true) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [status, setStatus] = useState<CameraStatus>("idle")
  const [error, setError] = useState<string | null>(null)
  const [retryToken, setRetryToken] = useState(0)

  useEffect(() => {
    if (!enabled) return
    let stream: MediaStream | null = null
    let cancelled = false

    setStatus("requesting")
    setError(null)

    navigator.mediaDevices
      .getUserMedia({
        // 폰 카메라의 native FOV/aspect 그대로 사용하도록 dim 제약 X.
        video: { facingMode: "user" },
        audio: false,
      })
      .then(async (s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop())
          return
        }
        stream = s
        const video = videoRef.current
        if (!video) return
        video.srcObject = s
        video.muted = true
        video.playsInline = true
        try {
          await video.play()
          setStatus("ready")
        } catch (e) {
          setStatus("error")
          setError((e as Error).message)
        }
      })
      .catch((e: Error) => {
        if (cancelled) return
        if (e.name === "NotAllowedError" || e.name === "SecurityError") {
          setStatus("denied")
        } else {
          setStatus("error")
        }
        setError(e.message)
      })

    return () => {
      cancelled = true
      stream?.getTracks().forEach((t) => t.stop())
      if (videoRef.current) videoRef.current.srcObject = null
    }
  }, [enabled, retryToken])

  const retry = useCallback(() => setRetryToken((t) => t + 1), [])

  return { videoRef, status, error, retry }
}
