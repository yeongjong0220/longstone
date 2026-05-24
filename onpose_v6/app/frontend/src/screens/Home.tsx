import { useCallback, useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"

import PoseOverlay from "@/components/PoseOverlay"
import { Glass, PageHeader } from "@/components/aura"
import { Button } from "@/components/ui/button"
import { useCamera } from "@/hooks/useCamera"
import { useHoverTrigger, type HoverTarget } from "@/hooks/useHoverTrigger"
import { usePoseDetector } from "@/hooks/usePoseDetector"
import { createSession, listExercises, type Exercise } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useSessionStore } from "@/stores/sessionStore"

const DWELL_MS = 1200

export default function Home() {
  const [exercises, setExercises] = useState<Exercise[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const navigate = useNavigate()
  const setSession = useSessionStore((s) => s.setSession)

  const { videoRef, status: camStatus } = useCamera(true)
  const { latestRef } = usePoseDetector(videoRef, camStatus === "ready")

  const cameraDivRef = useRef<HTMLDivElement | null>(null)
  const cameraBoundsRef = useRef<DOMRect | null>(null)
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map())

  useEffect(() => {
    let cancelled = false
    listExercises()
      .then((list) => {
        if (!cancelled) setExercises(list)
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

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

  const startWith = useCallback(
    async (ex: Exercise) => {
      if (pending) return
      setPending(true)
      setError(null)
      try {
        const { session_id } = await createSession(ex.id, ex.reps_default, ex.sets_default)
        setSession({
          sessionId: session_id,
          exerciseId: ex.id,
          exerciseName: ex.name,
          reps: ex.reps_default,
          sets: ex.sets_default,
        })
        navigate("/setup")
      } catch (e) {
        setError((e as Error).message)
        setPending(false)
      }
    },
    [pending, setSession, navigate],
  )

  const getTargets = useCallback((): HoverTarget[] => {
    const out: HoverTarget[] = []
    cardRefs.current.forEach((el, key) => {
      out.push({ key, rect: el.getBoundingClientRect() })
    })
    return out
  }, [])

  const onSelect = useCallback(
    (key: string) => {
      const ex = exercises?.find((e) => e.id === key)
      if (ex) void startWith(ex)
    },
    [exercises, startWith],
  )

  const hover = useHoverTrigger({
    landmarksRef: latestRef,
    cameraBoundsRef,
    getTargets,
    onSelect,
    dwellMs: DWELL_MS,
    enabled: camStatus === "ready" && !pending && exercises !== null,
    mirror: true,
  })

  const setCardRef = (id: string) => (el: HTMLDivElement | null) => {
    if (el) cardRefs.current.set(id, el)
    else cardRefs.current.delete(id)
  }

  return (
    <main className="min-h-svh flex flex-col gap-4 px-4 py-4">
      <PageHeader title="Longstone" subtitle="손을 카드에 1.2초 머무르면 시작돼요." className="px-1" />

      {error && (
        <p className="text-destructive text-sm text-center whitespace-pre-line break-keep" role="alert">
          {error}
        </p>
      )}

      <div
        ref={cameraDivRef}
        className="relative mx-auto w-full max-w-md overflow-hidden rounded-2xl border border-aura-glass-border bg-black aspect-[3/4]"
      >
        <video ref={videoRef} className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]" />
        <PoseOverlay landmarksRef={latestRef} className="absolute inset-0 h-full w-full" />

        {/* 카메라 상태 메시지 */}
        {camStatus !== "ready" && (
          <div className="absolute inset-0 grid place-items-center bg-black/50 text-white text-sm p-4 text-center">
            {camStatus === "denied"
              ? "카메라 권한을 허용해 주세요"
              : camStatus === "error"
              ? "카메라를 열 수 없어요"
              : "카메라 준비 중…"}
          </div>
        )}

        {/* LIVE chip */}
        <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full bg-black/70 px-2.5 py-1 text-white text-[10px] font-medium backdrop-blur-md">
          <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" aria-hidden />
          LIVE · On-device
        </div>

        {/* 운동 카드 - 우측 세로 스택 */}
        <div className="absolute right-3 top-3 bottom-3 flex flex-col gap-2 w-[44%] max-w-[180px]">
          {exercises === null && !error && (
            <div className="rounded-xl bg-white/90 px-3 py-2 text-aura-ter text-xs">불러오는 중…</div>
          )}
          {exercises?.map((ex) => {
            const isActive = hover.activeKey === ex.id
            const progress = isActive ? hover.progress : 0
            return (
              <div
                key={ex.id}
                ref={setCardRef(ex.id)}
                className={cn(
                  "relative flex-1 min-h-0 rounded-xl overflow-hidden bg-white/95 shadow-md transition-all",
                  isActive ? "ring-4 ring-aura-primary scale-[1.02]" : "ring-1 ring-aura-glass-border",
                )}
              >
                <video
                  src={`/videos/${ex.id}.mp4`}
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="absolute inset-0 h-full w-full object-cover"
                  onError={(e) => {
                    (e.currentTarget as HTMLVideoElement).style.display = "none"
                  }}
                />
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent px-2.5 py-2">
                  <div className="text-white text-[12px] font-bold leading-tight">{ex.name}</div>
                  <div className="text-white/80 text-[9px]">
                    {ex.reps_default}회 × {ex.sets_default}세트
                  </div>
                </div>
                {progress > 0 && (
                  <div className="absolute inset-0 grid place-items-center bg-aura-primary/20">
                    <svg width="48" height="48" viewBox="0 0 48 48">
                      <circle cx="24" cy="24" r="20" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="4" />
                      <circle
                        cx="24"
                        cy="24"
                        r="20"
                        fill="none"
                        stroke="#4F46E5"
                        strokeWidth="4"
                        strokeLinecap="round"
                        strokeDasharray={`${2 * Math.PI * 20}`}
                        strokeDashoffset={`${2 * Math.PI * 20 * (1 - progress)}`}
                        transform="rotate(-90 24 24)"
                      />
                    </svg>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* 손 커서 (사용자에게 어디를 가리키는지 시각화) */}
        {hover.cursor && cameraBoundsRef.current && (
          <div
            className="pointer-events-none absolute size-6 rounded-full bg-aura-primary/30 ring-2 ring-aura-primary"
            style={{
              left: hover.cursor.x - (cameraBoundsRef.current.left ?? 0) - 12,
              top: hover.cursor.y - (cameraBoundsRef.current.top ?? 0) - 12,
            }}
            aria-hidden
          />
        )}
      </div>

      <Glass className="mx-auto w-full max-w-md p-3 text-center text-aura-sec text-xs">
        손목을 원하는 카드 위에 잠시 멈춰주세요 · 클릭 없이 시작됩니다
      </Glass>

      {/* 클릭 fallback: 카메라가 없는 환경/PC 데모용 */}
      {exercises && (
        <div className="mx-auto w-full max-w-md grid grid-cols-3 gap-2">
          {exercises.map((ex) => (
            <Button
              key={ex.id}
              variant="ghost"
              size="sm"
              className="text-[11px]"
              onClick={() => startWith(ex)}
              disabled={pending}
            >
              {ex.name} 클릭 시작
            </Button>
          ))}
        </div>
      )}
    </main>
  )
}
