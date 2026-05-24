import { useCallback, useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"

import PoseOverlay from "@/components/PoseOverlay"
import { BackHeader, Glass } from "@/components/aura"
import { Button } from "@/components/ui/button"
import { useCamera } from "@/hooks/useCamera"
import { useHoverTrigger, type HoverTarget } from "@/hooks/useHoverTrigger"
import { usePoseDetector } from "@/hooks/usePoseDetector"
import { cn } from "@/lib/utils"
import { useSessionStore } from "@/stores/sessionStore"

const VISIBILITY_THRESHOLD = 0.5
const REQUIRED_LANDMARKS = 33
const DWELL_MS = 1200

export default function Setup() {
  const navigate = useNavigate()
  const sessionId = useSessionStore((s) => s.sessionId)
  const exerciseName = useSessionStore((s) => s.exerciseName)
  const { videoRef, status, error, retry } = useCamera(true)
  const [visibleCount, setVisibleCount] = useState(0)

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

  const cameraDivRef = useRef<HTMLDivElement | null>(null)
  const cameraBoundsRef = useRef<DOMRect | null>(null)
  const startCardRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const update = () => {
      const el = cameraDivRef.current
      cameraBoundsRef.current = el ? el.getBoundingClientRect() : null
    }
    update()
    window.addEventListener("resize", update)
    window.addEventListener("scroll", update, true)
    const id = window.setInterval(update, 500)
    return () => {
      window.removeEventListener("resize", update)
      window.removeEventListener("scroll", update, true)
      window.clearInterval(id)
    }
  }, [])

  const getTargets = useCallback((): HoverTarget[] => {
    if (!readyToStart || !startCardRef.current) return []
    return [{ key: "start", rect: startCardRef.current.getBoundingClientRect() }]
  }, [readyToStart])

  const onSelect = useCallback(() => {
    navigate("/coaching")
  }, [navigate])

  const hover = useHoverTrigger({
    landmarksRef: latestRef,
    cameraBoundsRef,
    getTargets,
    onSelect,
    dwellMs: DWELL_MS,
    enabled: readyToStart,
    mirror: true,
  })

  const startActive = hover.activeKey === "start"
  const progress = startActive ? hover.progress : 0

  const subtitle = exerciseName
    ? `${exerciseName} — 33관절이 모두 잡히면 손을 시작 버튼에 가져가세요.`
    : "33관절이 모두 잡히면 손을 시작 버튼에 가져가세요."

  return (
    <main className="min-h-svh flex flex-col gap-3 pb-6">
      <BackHeader title="자세 준비" onBack={() => navigate("/")} action={<span className="text-aura-ter text-xs">{exerciseName ?? ""}</span>} />
      <p className="text-aura-ter mx-auto w-full max-w-md px-4 text-[13px]">{subtitle}</p>

      <div className="mx-auto w-full max-w-md px-4">
        <div
          ref={cameraDivRef}
          className="border-aura-glass-border relative overflow-hidden rounded-2xl border bg-black aspect-[3/4]"
        >
          <video
            ref={videoRef}
            className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]"
          />
          <PoseOverlay landmarksRef={latestRef} className="absolute inset-0 h-full w-full" />

          {/* 관절 진행 표시 - 좌상단 */}
          <div className="absolute left-3 top-3 rounded-full bg-black/70 px-3 py-1.5 text-white text-[11px] font-medium backdrop-blur-md">
            관절 {visibleCount}/{REQUIRED_LANDMARKS}
            {!detectorReady && " · 모델 로딩"}
          </div>

          {/* 진행률 바 - 상단 */}
          <div className="absolute left-3 right-3 top-12 h-1 rounded-full bg-white/20 overflow-hidden">
            <div
              className={cn(
                "h-full transition-all",
                readyToStart ? "bg-emerald-400" : "bg-aura-primary/80",
              )}
              style={{ width: `${(visibleCount / REQUIRED_LANDMARKS) * 100}%` }}
            />
          </div>

          {/* 시작 hover 카드 - 카메라 하단 중앙 */}
          <div
            ref={startCardRef}
            className={cn(
              "absolute left-1/2 -translate-x-1/2 bottom-6 rounded-full px-8 py-4 shadow-lg transition-all backdrop-blur-md",
              readyToStart
                ? "bg-emerald-500/95 text-white"
                : "bg-white/60 text-aura-ter",
              startActive && "scale-110 ring-4 ring-emerald-200",
            )}
          >
            <span className="text-base font-bold tracking-tight">
              {readyToStart
                ? startActive
                  ? "시작합니다…"
                  : "손을 여기에 1초 멈춰주세요"
                : "준비 중…"}
            </span>
            {progress > 0 && (
              <div className="absolute inset-0 grid place-items-center">
                <svg width="78" height="78" viewBox="0 0 78 78">
                  <circle cx="39" cy="39" r="35" fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="4" />
                  <circle
                    cx="39"
                    cy="39"
                    r="35"
                    fill="none"
                    stroke="white"
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeDasharray={`${2 * Math.PI * 35}`}
                    strokeDashoffset={`${2 * Math.PI * 35 * (1 - progress)}`}
                    transform="rotate(-90 39 39)"
                  />
                </svg>
              </div>
            )}
          </div>

          {/* 손 커서 */}
          {hover.cursor && cameraBoundsRef.current && (
            <div
              className="pointer-events-none absolute size-6 rounded-full bg-emerald-500/30 ring-2 ring-emerald-500"
              style={{
                left: hover.cursor.x - (cameraBoundsRef.current.left ?? 0) - 12,
                top: hover.cursor.y - (cameraBoundsRef.current.top ?? 0) - 12,
              }}
              aria-hidden
            />
          )}

          {(status === "denied" || status === "error") && (
            <div className="absolute inset-0 flex items-center justify-center p-4" role="alert">
              <Glass strong className="flex flex-col gap-3 p-5 text-center">
                <p className="text-aura-ink font-semibold">
                  {status === "denied"
                    ? "카메라 권한이 거부되었어요"
                    : "카메라를 열 수 없어요"}
                </p>
                {status === "denied" ? (
                  <p className="text-aura-ter text-sm leading-relaxed">
                    설정 → 브라우저 → 이 사이트 →<br />
                    카메라를 &quot;허용&quot;으로 바꾼 뒤<br />
                    아래 버튼을 눌러주세요.
                  </p>
                ) : (
                  <p className="text-aura-ter text-sm leading-relaxed">{error ?? "알 수 없는 오류"}</p>
                )}
                <Button size="sm" variant="outline" onClick={retry}>
                  다시 시도
                </Button>
              </Glass>
            </div>
          )}
        </div>
      </div>

      <div className="mx-auto flex w-full max-w-md flex-col gap-2 px-4">
        <Button
          variant="ghost"
          onClick={() => navigate("/coaching")}
          disabled={!readyToStart}
          className="text-aura-ter text-xs"
        >
          {readyToStart ? "(또는 클릭으로 시작)" : ""}
        </Button>
        <Button variant="ghost" onClick={() => navigate("/")}>
          홈으로
        </Button>
      </div>
    </main>
  )
}
