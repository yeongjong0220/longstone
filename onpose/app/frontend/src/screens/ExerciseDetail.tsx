import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useLocation, useNavigate } from "react-router-dom"

import RotateHint from "@/components/RotateHint"
import { useCamera } from "@/hooks/useCamera"
import { useHoverTrigger, type HoverTarget } from "@/hooks/useHoverTrigger"
import { useOrientationLock } from "@/hooks/useOrientationLock"
import { usePoseDetector } from "@/hooks/usePoseDetector"
import { createSession, listExercises, type Exercise } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useSessionStore } from "@/stores/sessionStore"

const DWELL_MS = 1200

type LocationState = { exerciseId?: string } | null

export default function ExerciseDetail() {
  const navigate = useNavigate()
  const location = useLocation()
  const stateExerciseId = (location.state as LocationState)?.exerciseId
  useOrientationLock("landscape")
  const setSession = useSessionStore((s) => s.setSession)

  const [exercise, setExercise] = useState<Exercise | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [busy, setBusy] = useState<null | "start" | "expert" | "other">(null)

  useEffect(() => {
    if (!stateExerciseId) {
      navigate("/home", { replace: true })
      return
    }
    let cancelled = false
    listExercises()
      .then((list) => {
        if (cancelled) return
        const found = list.find((e) => e.id === stateExerciseId) ?? null
        if (!found) {
          navigate("/home", { replace: true })
          return
        }
        setExercise(found)
      })
      .catch((e: Error) => !cancelled && setLoadError(e.message))
    return () => {
      cancelled = true
    }
  }, [stateExerciseId, navigate])

  const { videoRef, status: camStatus } = useCamera(true)
  const { latestRef } = usePoseDetector(videoRef, camStatus === "ready")

  const cameraDivRef = useRef<HTMLDivElement | null>(null)
  const cameraBoundsRef = useRef<DOMRect | null>(null)
  const startBtnRef = useRef<HTMLDivElement | null>(null)
  const expertBtnRef = useRef<HTMLDivElement | null>(null)
  const otherBtnRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const update = () => {
      const el = cameraDivRef.current
      cameraBoundsRef.current = el ? el.getBoundingClientRect() : null
    }
    update()
    window.addEventListener("resize", update)
    const id = window.setInterval(update, 500)
    return () => {
      window.removeEventListener("resize", update)
      window.clearInterval(id)
    }
  }, [])

  const onStart = useCallback(async () => {
    if (!exercise || busy) return
    setBusy("start")
    try {
      const { session_id } = await createSession(exercise.id, exercise.reps_default, exercise.sets_default)
      setSession({
        sessionId: session_id,
        exerciseId: exercise.id,
        exerciseName: exercise.name,
        reps: exercise.reps_default,
        sets: exercise.sets_default,
      })
      navigate("/setup")
    } catch (e) {
      setLoadError((e as Error).message)
      setBusy(null)
    }
  }, [exercise, busy, setSession, navigate])

  const onExpert = useCallback(() => {
    if (!exercise || busy) return
    setBusy("expert")
    navigate(`/expert/${exercise.id}`, { state: { exerciseName: exercise.name } })
  }, [exercise, busy, navigate])

  const onOther = useCallback(() => {
    if (busy) return
    setBusy("other")
    navigate("/home")
  }, [busy, navigate])

  const onHoverSelect = useCallback(
    (key: string) => {
      if (key === "start") void onStart()
      else if (key === "expert") onExpert()
      else if (key === "other") onOther()
    },
    [onStart, onExpert, onOther],
  )

  const getTargets = useCallback((): HoverTarget[] => {
    const targets: HoverTarget[] = []
    if (startBtnRef.current) targets.push({ key: "start", rect: startBtnRef.current.getBoundingClientRect() })
    if (expertBtnRef.current) targets.push({ key: "expert", rect: expertBtnRef.current.getBoundingClientRect() })
    if (otherBtnRef.current) targets.push({ key: "other", rect: otherBtnRef.current.getBoundingClientRect() })
    return targets
  }, [])

  const hover = useHoverTrigger({
    landmarksRef: latestRef,
    cameraBoundsRef,
    getTargets,
    onSelect: onHoverSelect,
    dwellMs: DWELL_MS,
    enabled: !!exercise && !busy,
    mirror: true,
  })

  const buttons = useMemo(
    () =>
      [
        { key: "start" as const, label: "시작", primary: true, ref: startBtnRef },
        { key: "expert" as const, label: "전문가 영상", primary: false, ref: expertBtnRef },
        { key: "other" as const, label: "다른 운동", primary: false, ref: otherBtnRef },
      ],
    [],
  )

  return (
    <main className="fixed inset-0 bg-black overflow-hidden">
      <RotateHint />
      <div ref={cameraDivRef} className="absolute inset-0" style={{ isolation: "isolate" }}>
        <video
          ref={videoRef}
          className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]"
        />
        {/* 블러 + 화이트 오버레이 (옅은 톤)
            key={camStatus} 로 카메라 ready 시점에 강제 리마운트해
            backdrop-filter 합성 레이어가 누락되는 첫진입 버그 회피. */}
        <div
          key={camStatus}
          className="absolute inset-0 bg-white/25 backdrop-blur-sm"
          style={{ transform: "translateZ(0)", willChange: "backdrop-filter" }}
        />

        {/* 운동 이름 */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 rounded-full bg-white border border-[#E5E8EB] px-4 py-1.5 shadow-[0_1px_3px_rgba(0,0,0,0.04)]">
          <span className="text-aura-ink text-sm font-bold">{exercise?.name ?? "..."}</span>
        </div>

        {/* 중앙 3 버튼 세로 정렬 */}
        <div className="absolute inset-0 grid place-items-center">
          <div className="flex flex-col gap-3 w-[280px]">
            {buttons.map((b) => {
              const isActive = hover.activeKey === b.key
              const progress = isActive ? hover.progress : 0
              return (
                <div
                  key={b.key}
                  ref={b.ref}
                  className={cn(
                    "relative rounded-full border h-14 grid place-items-center transition-all overflow-hidden shadow-[0_10px_24px_-12px_rgba(0,0,0,0.5)]",
                    b.primary
                      ? "bg-[#E6FB4D] text-[#15110A] border-transparent"
                      : "bg-white text-[#191F28] border-[#E5E8EB]",
                    isActive && "scale-[1.03] ring-4 ring-[#E6FB4D]/70",
                  )}
                >
                  <span className="text-base font-bold tracking-tight">{b.label}</span>
                  {progress > 0 && (
                    <div
                      className={cn(
                        "absolute bottom-0 left-0 h-1 transition-none",
                        b.primary ? "bg-[#15110A]/70" : "bg-[#4E45E6]",
                      )}
                      style={{ width: `${progress * 100}%` }}
                    />
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* 손 커서 (블러 위) */}
        {hover.cursor && (
          <div
            className="pointer-events-none fixed size-6 rounded-full bg-aura-primary/30 ring-2 ring-aura-primary z-10"
            style={{ left: hover.cursor.x - 12, top: hover.cursor.y - 12 }}
            aria-hidden
          />
        )}

        {loadError && (
          <div role="alert" className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-2xl bg-white border border-[#E5E8EB] px-4 py-2 shadow-md">
            <p className="text-destructive text-xs">{loadError}</p>
          </div>
        )}
      </div>
    </main>
  )
}
