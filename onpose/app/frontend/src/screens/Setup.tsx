import { useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"

import PoseOverlay from "@/components/PoseOverlay"
import RotateHint from "@/components/RotateHint"
import { Glass } from "@/components/aura"
import { Button } from "@/components/ui/button"
import { useCamera } from "@/hooks/useCamera"
import { useOrientationLock } from "@/hooks/useOrientationLock"
import { usePoseDetector } from "@/hooks/usePoseDetector"
import { useSessionStore } from "@/stores/sessionStore"

const VISIBILITY_THRESHOLD = 0.5
const REQUIRED_LANDMARKS = 33
const HOLD_MS = 1000

export default function Setup() {
  const navigate = useNavigate()
  const sessionId = useSessionStore((s) => s.sessionId)
  const exerciseName = useSessionStore((s) => s.exerciseName)
  useOrientationLock("landscape")
  const { videoRef, status, error, retry } = useCamera(true)
  const [visibleCount, setVisibleCount] = useState(0)
  const [holdStart, setHoldStart] = useState<number | null>(null)
  const [holdProgress, setHoldProgress] = useState(0)
  const navigatedRef = useRef(false)

  const { latestRef, ready: detectorReady } = usePoseDetector(
    videoRef,
    status === "ready",
    (lms) => {
      if (lms.length === 0) {
        setVisibleCount(0)
        return
      }
      let n = 0
      for (const lm of lms) {
        if (lm[3] >= VISIBILITY_THRESHOLD) n += 1
      }
      setVisibleCount(n)
    },
  )

  const readyToStart = visibleCount === REQUIRED_LANDMARKS && !!sessionId

  // 33관절 인식 후 1초 holding → 자동 카운트다운
  useEffect(() => {
    if (!readyToStart) {
      setHoldStart(null)
      setHoldProgress(0)
      return
    }
    if (holdStart === null) setHoldStart(performance.now())
  }, [readyToStart, holdStart])

  useEffect(() => {
    if (holdStart === null || navigatedRef.current) return
    let raf = 0
    const tick = () => {
      const elapsed = performance.now() - holdStart
      const p = Math.min(1, elapsed / HOLD_MS)
      setHoldProgress(p)
      if (p >= 1 && !navigatedRef.current) {
        navigatedRef.current = true
        navigate("/countdown", { replace: true })
        return
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [holdStart, navigate])

  useEffect(() => {
    if (!sessionId) {
      navigate("/home", { replace: true })
    }
  }, [sessionId, navigate])

  const totalProgress = Math.min(
    1,
    (visibleCount / REQUIRED_LANDMARKS) * 0.85 + holdProgress * 0.15,
  )

  return (
    <main className="fixed inset-0 bg-black overflow-hidden">
      <RotateHint />
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]"
      />
      <PoseOverlay landmarksRef={latestRef} className="absolute inset-0 h-full w-full" />

      {/* 중앙 대형 안내 (자세 잡힘 또는 모델 로딩 시 숨김) */}
      {!readyToStart && (
        <div className="absolute inset-0 grid place-items-center pointer-events-none px-8">
          <p className="text-white text-3xl md:text-4xl font-bold text-center drop-shadow-lg break-keep">
            {detectorReady ? "몸 전신이 화면에 담기도록 해주세요" : "준비 중…"}
          </p>
        </div>
      )}

      {/* 운동 이름 */}
      {exerciseName && (
        <div className="absolute top-4 right-4 rounded-full bg-white border border-[#E5E8EB] px-3 py-1.5 shadow-[0_1px_3px_rgba(0,0,0,0.04)]">
          <p className="text-aura-ink text-xs font-semibold">{exerciseName}</p>
        </div>
      )}

      {/* 하단 진행 바 */}
      <div className="absolute left-0 right-0 bottom-0">
        <div className="mx-auto max-w-md px-4 pb-6">
          <div className="rounded-full h-2.5 bg-white/30 overflow-hidden">
            <div
              className="h-full bg-[#E6FB4D] transition-all"
              style={{ width: `${totalProgress * 100}%` }}
            />
          </div>
          <p className="text-white text-center text-xs mt-2 font-medium drop-shadow">
            {readyToStart
              ? holdProgress >= 1
                ? "시작합니다…"
                : "자세 유지 중…"
              : "전신을 화면에 맞춰주세요"}
          </p>
        </div>
      </div>

      {(status === "denied" || status === "error") && (
        <div className="absolute inset-0 flex items-center justify-center p-4" role="alert">
          <Glass className="flex flex-col gap-3 p-5 text-center max-w-xs">
            <p className="text-aura-ink font-semibold">
              {status === "denied" ? "카메라 권한이 거부되었어요" : "카메라를 열 수 없어요"}
            </p>
            <p className="text-aura-ter text-sm leading-relaxed">
              {status === "denied"
                ? "브라우저 설정에서 카메라를 허용한 뒤 다시 시도해주세요."
                : (error ?? "알 수 없는 오류")}
            </p>
            <Button size="sm" variant="outline" onClick={retry}>
              다시 시도
            </Button>
          </Glass>
        </div>
      )}
    </main>
  )
}
