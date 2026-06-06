import { useCallback, useEffect, useRef, useState } from "react"
import { useLocation, useNavigate, useParams } from "react-router-dom"

import { CloseIcon } from "@/components/aura"
import { useCamera } from "@/hooks/useCamera"
import { useHoverTrigger, type HoverTarget } from "@/hooks/useHoverTrigger"
import { useOrientationLock } from "@/hooks/useOrientationLock"
import { usePoseDetector } from "@/hooks/usePoseDetector"
import { cn } from "@/lib/utils"
import { useSessionStore } from "@/stores/sessionStore"

type LocationState = { exerciseName?: string } | null

const BACK_DWELL_MS = 1200

export default function ExpertVideo() {
  const navigate = useNavigate()
  const location = useLocation()
  const { exerciseId = "" } = useParams<{ exerciseId: string }>()
  const storeName = useSessionStore((s) => s.exerciseName)
  const stateName = (location.state as LocationState)?.exerciseName
  const exerciseName = stateName ?? storeName ?? "전문가 시범"
  useOrientationLock("landscape")
  const [errored, setErrored] = useState(false)

  // 카메라 + 포즈 — 영상은 화면에 안 보이지만 손목 좌표를 얻기 위해 가동
  const { videoRef, status: camStatus } = useCamera(true)
  const { latestRef } = usePoseDetector(videoRef, camStatus === "ready")

  const stageRef = useRef<HTMLDivElement | null>(null)
  const stageBoundsRef = useRef<DOMRect | null>(null)
  const backBtnRef = useRef<HTMLDivElement | null>(null)

  // 뷰포트(스테이지) bounds 추적 — 호버 좌표 매핑용
  useEffect(() => {
    const update = () => {
      const el = stageRef.current
      stageBoundsRef.current = el ? el.getBoundingClientRect() : null
    }
    update()
    window.addEventListener("resize", update)
    const id = window.setInterval(update, 500)
    return () => {
      window.removeEventListener("resize", update)
      window.clearInterval(id)
    }
  }, [])

  const goBack = useCallback(() => navigate(-1), [navigate])

  const getTargets = useCallback((): HoverTarget[] => {
    if (!backBtnRef.current) return []
    return [{ key: "back", rect: backBtnRef.current.getBoundingClientRect() }]
  }, [])

  const hover = useHoverTrigger({
    landmarksRef: latestRef,
    cameraBoundsRef: stageBoundsRef,
    getTargets,
    onSelect: () => goBack(),
    dwellMs: BACK_DWELL_MS,
    enabled: true,
    mirror: true,
  })

  const src = `/videos/${exerciseId}.mp4`

  return (
    <div
      ref={stageRef}
      role="dialog"
      aria-modal="true"
      aria-label={`${exerciseName} 영상`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black"
    >
      {/* 사용자 카메라 — 화면엔 안 보이지만 MediaPipe가 detect할 수 있도록 DOM에 살려둠 */}
      <video
        ref={videoRef}
        playsInline
        muted
        style={{
          position: "absolute",
          width: "2px",
          height: "2px",
          opacity: 0,
          pointerEvents: "none",
          left: 0,
          top: 0,
        }}
        aria-hidden
      />

      {/* 운동 이름 */}
      <div className="absolute top-3 left-3 z-10 text-white text-sm font-semibold drop-shadow">
        {exerciseName}
      </div>

      {/* 전문가 영상 */}
      {errored ? (
        <div className="px-6 text-center">
          <p className="text-base font-semibold text-white">전문가 영상 준비 중이에요</p>
          <p className="mt-2 text-sm text-white/70">{exerciseName} 영상은 곧 추가될 예정입니다.</p>
        </div>
      ) : (
        <video
          key={src}
          src={src}
          autoPlay
          playsInline
          controls={false}
          onEnded={goBack}
          onError={() => setErrored(true)}
          className="max-h-full max-w-full"
        />
      )}

      {/* 뒤로 — 호버 트리거 */}
      <div
        ref={backBtnRef}
        className="absolute top-[max(env(safe-area-inset-top),12px)] right-3"
      >
        <button
          type="button"
          onClick={goBack}
          aria-label="뒤로"
          className={cn(
            "relative flex items-center gap-1.5 rounded-full bg-[#191F28]/85 px-4 py-2.5 text-sm font-medium text-white overflow-hidden transition-all",
            hover.activeKey === "back" && "scale-105 ring-4 ring-[#E6FB4D]/70",
          )}
        >
          <span className="relative z-10 inline-flex items-center gap-1.5">
            <span>
              {hover.activeKey === "back" ? "돌아갑니다…" : "뒤로"}
            </span>
            <CloseIcon color="currentColor" size={16} />
          </span>
          {hover.activeKey === "back" && hover.progress > 0 && (
            <span
              className="absolute inset-y-0 left-0 bg-[#E6FB4D]/35"
              style={{ width: `${hover.progress * 100}%` }}
              aria-hidden
            />
          )}
        </button>
      </div>

      {/* 손목 동그라미 커서 — 카메라는 숨겼지만 손 위치는 이걸로 인지 */}
      {hover.cursor && (
        <div
          className="pointer-events-none fixed size-8 rounded-full bg-aura-primary/40 ring-2 ring-aura-primary z-[60] shadow-lg"
          style={{
            left: hover.cursor.x - 16,
            top: hover.cursor.y - 16,
          }}
          aria-hidden
        />
      )}
    </div>
  )
}
