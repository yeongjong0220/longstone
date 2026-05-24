import { useCallback, useEffect, useRef, useState } from "react"

type RecorderState = "idle" | "recording" | "stopped"

const pickMimeType = (): string => {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ]
  for (const m of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) return m
  }
  return ""
}

export function useRecorder(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const [state, setState] = useState<RecorderState>("idle")
  const [lastUrl, setLastUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const start = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state === "recording") return
    const video = videoRef.current
    const stream = (video?.srcObject as MediaStream | null) ?? null
    if (!stream) {
      setError("녹화할 카메라 스트림이 없어요")
      return
    }
    if (typeof MediaRecorder === "undefined") {
      setError("이 브라우저는 녹화를 지원하지 않아요")
      return
    }
    try {
      const mimeType = pickMimeType()
      const rec = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      chunksRef.current = []
      rec.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data)
      }
      rec.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "video/webm" })
        chunksRef.current = []
        if (blob.size === 0) {
          setState("stopped")
          return
        }
        const url = URL.createObjectURL(blob)
        setLastUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev)
          return url
        })
        setState("stopped")
      }
      rec.onerror = (ev) => setError(String((ev as ErrorEvent).message || "녹화 오류"))
      recorderRef.current = rec
      rec.start(1000) // 1초 단위로 chunk
      setState("recording")
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [videoRef])

  const stop = useCallback(() => {
    const rec = recorderRef.current
    if (rec && rec.state === "recording") {
      rec.stop()
    }
  }, [])

  useEffect(() => {
    return () => {
      const rec = recorderRef.current
      if (rec && rec.state === "recording") {
        try {
          rec.stop()
        } catch {
          /* ignore */
        }
      }
    }
  }, [])

  return { state, lastUrl, error, start, stop }
}
