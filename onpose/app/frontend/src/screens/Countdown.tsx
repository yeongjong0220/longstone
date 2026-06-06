import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"

import RotateHint from "@/components/RotateHint"
import { useCamera } from "@/hooks/useCamera"
import { useOrientationLock } from "@/hooks/useOrientationLock"
import { primeSpeech } from "@/hooks/useSpeechFeedback"
import { cn } from "@/lib/utils"
import { useSessionStore } from "@/stores/sessionStore"

const COUNTDOWN_START = 3

export default function Countdown() {
  const navigate = useNavigate()
  const sessionId = useSessionStore((s) => s.sessionId)
  useOrientationLock("landscape")
  const { videoRef } = useCamera(true)
  const [n, setN] = useState<number>(COUNTDOWN_START)

  // 카운트다운 동안 TTS 엔진 워밍업 — Coaching 도착 시 첫 발화 지연 회피
  useEffect(() => {
    primeSpeech()
  }, [])

  useEffect(() => {
    if (!sessionId) {
      navigate("/home", { replace: true })
    }
  }, [sessionId, navigate])

  useEffect(() => {
    if (n <= 0) {
      const t = window.setTimeout(() => navigate("/coaching", { replace: true }), 500)
      return () => window.clearTimeout(t)
    }
    const t = window.setTimeout(() => setN((c) => c - 1), 1000)
    return () => window.clearTimeout(t)
  }, [n, navigate])

  return (
    <main className="fixed inset-0 bg-black overflow-hidden">
      <RotateHint />
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]"
      />
      <div className="absolute inset-0 grid place-items-center bg-[#0A0815]/55">
        <div className="flex flex-col items-center gap-3">
          <p className="text-white/85 text-base font-medium">3초 후 시작..</p>
          <span
            className={cn(
              "text-[14rem] font-extrabold leading-none tabular-nums drop-shadow-[0_8px_30px_rgba(0,0,0,0.45)]",
              n === 0 ? "text-[#E6FB4D]" : "text-white",
            )}
          >
            {n === 0 ? "GO" : n}
          </span>
        </div>
      </div>
    </main>
  )
}
