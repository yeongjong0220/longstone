import { useCallback, useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"

import PoseOverlay from "@/components/PoseOverlay"
import RotateHint from "@/components/RotateHint"
import { useCamera } from "@/hooks/useCamera"
import { useHoverTrigger, type HoverTarget } from "@/hooks/useHoverTrigger"
import { useOrientationLock } from "@/hooks/useOrientationLock"
import { usePoseDetector } from "@/hooks/usePoseDetector"
import { useRecorder } from "@/hooks/useRecorder"
import { useSpeechFeedback } from "@/hooks/useSpeechFeedback"
import { useWebSocket, type WsStatus } from "@/hooks/useWebSocket"
import { endSession } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useSessionStore, type CoachingFrame, type FeedbackItem } from "@/stores/sessionStore"

const END_DWELL_MS = 1200

const STATUS_TONE: Record<CoachingFrame["status"], string> = {
  good: "text-aura-primary",
  warn: "text-amber-500",
  err: "text-rose-500",
}

const STATUS_LABEL: Record<CoachingFrame["status"], string> = {
  good: "좋아요!",
  warn: "주의",
  err: "교정 필요",
}

const WS_LABEL: Record<WsStatus, { text: string; dot: string }> = {
  idle: { text: "대기", dot: "bg-aura-mute" },
  connecting: { text: "연결 중", dot: "bg-amber-400 animate-pulse" },
  open: { text: "연결됨", dot: "bg-emerald-500" },
  closed: { text: "끊김", dot: "bg-aura-ter" },
  failed: { text: "연결 실패", dot: "bg-rose-500" },
}

const formatTimer = (ms: number) => {
  const total = Math.max(0, Math.floor(ms / 1000))
  const m = String(Math.floor(total / 60)).padStart(2, "0")
  const s = String(total % 60).padStart(2, "0")
  return `${m}:${s}`
}

const feedbackIcon = (level: FeedbackItem["level"]) => {
  if (level === "ok") {
    return (
      <span className="bg-emerald-100 text-emerald-700 grid size-5 flex-shrink-0 place-items-center rounded-full">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="4 12 10 18 20 6" />
        </svg>
      </span>
    )
  }
  if (level === "warn") {
    return (
      <span className="bg-amber-100 text-amber-700 grid size-5 flex-shrink-0 place-items-center rounded-full">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3 2 21h20L12 3z" />
          <line x1="12" y1="10" x2="12" y2="14" />
          <circle cx="12" cy="17.5" r="0.7" fill="currentColor" stroke="none" />
        </svg>
      </span>
    )
  }
  // err = 빨간 주의 아이콘
  return (
    <span className="bg-red-100 text-red-600 grid size-5 flex-shrink-0 place-items-center rounded-full">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 3 2 21h20L12 3z" />
        <line x1="12" y1="10" x2="12" y2="14" />
        <circle cx="12" cy="17.5" r="0.7" fill="currentColor" stroke="none" />
      </svg>
    </span>
  )
}

// 관절 상태 → 색 (ok 초록 / warn 노랑 / err 빨강)
const ANGLE_COLOR: Record<string, string> = {
  ok: "#22C55E",
  warn: "#F59E0B",
  err: "#EF4444",
}
const angleColor = (s?: string) => ANGLE_COLOR[s ?? ""] ?? "#8B95A1"

function BodyAngleMini({
  knee,
  hip,
  kneeStatus,
  hipStatus,
}: {
  knee: number | null
  hip: number | null
  kneeStatus?: string
  hipStatus?: string
}) {
  const hipColor = angleColor(hipStatus)
  const kneeColor = angleColor(kneeStatus)
  return (
    <div className="flex items-center gap-2">
      <svg width="42" height="60" viewBox="0 0 56 80" aria-hidden>
        {/* 뼈대(라인) 먼저 그림 */}
        <circle cx="22" cy="10" r="6" fill="#E5E7EB" />
        <line x1="22" y1="16" x2="22" y2="36" stroke="#374151" strokeWidth="3" strokeLinecap="round" />
        <line x1="22" y1="22" x2="32" y2="36" stroke="#9CA3AF" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="22" y1="36" x2="42" y2="48" stroke="#374151" strokeWidth="3" strokeLinecap="round" />
        <line x1="42" y1="48" x2="36" y2="72" stroke="#374151" strokeWidth="3" strokeLinecap="round" />
        {/* 관절 원: 맨 위에, 지름 2배(r6), 상태색 + 흰 테두리 */}
        <circle cx="22" cy="36" r="6" fill={hipColor} stroke="#fff" strokeWidth="1.5" />
        <circle cx="42" cy="48" r="6" fill={kneeColor} stroke="#fff" strokeWidth="1.5" />
      </svg>
      <div className="flex flex-col gap-2 text-xs">
        {/* 고관절 먼저 */}
        <div>
          <div className="text-aura-ter text-[10px]">고관절 각도</div>
          <div
            className="text-base font-bold tabular-nums leading-none"
            style={{ color: hip !== null ? hipColor : "#8B95A1" }}
          >
            {hip !== null ? `${Math.round(hip)}°` : "—"}
          </div>
        </div>
        {/* 무릎 나중 */}
        <div>
          <div className="text-aura-ter text-[10px]">무릎 각도</div>
          <div
            className="text-base font-bold tabular-nums leading-none"
            style={{ color: knee !== null ? kneeColor : "#8B95A1" }}
          >
            {knee !== null ? `${Math.round(knee)}°` : "—"}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Coaching() {
  const navigate = useNavigate()
  const sessionId = useSessionStore((s) => s.sessionId)
  const exerciseName = useSessionStore((s) => s.exerciseName)
  const setCoaching = useSessionStore((s) => s.setCoaching)
  const lastCoaching = useSessionStore((s) => s.lastCoaching)
  const setRecordedUrl = useSessionStore((s) => s.setRecordedUrl)

  useOrientationLock("landscape")
  const [ending, setEnding] = useState(false)
  const [elapsedMs, setElapsedMs] = useState(0)
  const endedRef = useRef(false)
  const startedAtRef = useRef<number | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const stageBoundsRef = useRef<DOMRect | null>(null)
  const endBtnRef = useRef<HTMLDivElement | null>(null)

  const { videoRef, status: camStatus } = useCamera(true)
  const { latestRef } = usePoseDetector(videoRef, camStatus === "ready")
  const wsEnabled = camStatus === "ready" && !ending
  const { status: wsStatus } = useWebSocket({
    sessionId,
    enabled: wsEnabled,
    landmarksRef: latestRef,
    onMessage: (frame: CoachingFrame) => setCoaching(frame),
  })
  const recorder = useRecorder(videoRef)

  // 녹화 + 타이머
  useEffect(() => {
    if (startedAtRef.current === null) startedAtRef.current = performance.now()
    if (recorder.state === "idle" && camStatus === "ready") recorder.start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camStatus])

  useEffect(() => {
    if (ending) return
    const id = window.setInterval(() => {
      if (startedAtRef.current !== null) {
        setElapsedMs(performance.now() - startedAtRef.current)
      }
    }, 250)
    return () => window.clearInterval(id)
  }, [ending])

  useEffect(() => {
    if (recorder.lastUrl) setRecordedUrl(recorder.lastUrl)
  }, [recorder.lastUrl, setRecordedUrl])

  const onEnd = useCallback(async () => {
    if (endedRef.current) return
    endedRef.current = true
    if (recorder.state === "recording") recorder.stop()
    if (!sessionId) {
      navigate("/home")
      return
    }
    setEnding(true)
    try {
      await endSession(sessionId)
    } catch {
      /* Feedback에서 에러 표시 */
    }
    navigate("/feedback")
  }, [recorder, sessionId, navigate])

  // 카메라 영역 bounds (호버 좌표 매핑)
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

  const getEndTargets = useCallback((): HoverTarget[] => {
    if (!endBtnRef.current || ending) return []
    return [{ key: "end", rect: endBtnRef.current.getBoundingClientRect() }]
  }, [ending])

  const endHover = useHoverTrigger({
    landmarksRef: latestRef,
    cameraBoundsRef: stageBoundsRef,
    getTargets: getEndTargets,
    onSelect: () => void onEnd(),
    dwellMs: END_DWELL_MS,
    enabled: !ending,
    mirror: true,
  })

  const feedbackList: FeedbackItem[] = lastCoaching?.feedback?.slice(0, 3) ?? []
  const angles = lastCoaching?.angles ?? {}
  const kneeAngle = angles.knee ?? null
  const hipAngle = angles.hip ?? null
  const angleStatus = lastCoaching?.angle_status ?? {}
  const hipStatus = angleStatus.hip
  const kneeStatus = angleStatus.knee

  // 실시간 피드백 TTS — 잘하고 있을 때(유지 멘트) 포함 모두 안내.
  // (useSpeechFeedback 이 동일 메시지 중복발화 방지 + 10초 간격 재발화로 제어)
  const ttsMessages = feedbackList.slice(0, 2).map((f) => f.msg)
  useSpeechFeedback(ttsMessages, !ending)

  return (
    <main ref={stageRef} className="fixed inset-0 bg-black overflow-hidden">
      <RotateHint />
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]"
      />
      <PoseOverlay landmarksRef={latestRef} className="absolute inset-0 h-full w-full" />

      {/* 좌상단: 운동 이름만 (실시간/TTS, 정확도% 칩 제거) */}
      <div className="absolute top-3 left-3 max-w-[30%]">
        <h1 className="text-white text-xl font-bold tracking-tight drop-shadow-lg truncate">
          {exerciseName ?? "운동"}
        </h1>
      </div>

      {/* 상단 중앙: 실시간 피드백 패널 (v7 패턴) */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 w-[min(46vw,430px)] z-10">
        <div className="rounded-2xl bg-white/95 shadow-md border border-[#E5E8EB] p-3">
          <div className="flex items-stretch gap-2">
            <div className="flex-1 min-w-0">
              <h3 className="text-aura-ink text-sm font-bold mb-2">실시간 피드백</h3>
              <ul className="flex flex-col gap-1.5">
                {feedbackList.length > 0 ? (
                  feedbackList.map((f, i) => (
                    <li key={i} className="flex items-start gap-2">
                      {feedbackIcon(f.level)}
                      <span className="text-aura-ink text-xs leading-snug break-keep">{f.msg}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-aura-ter text-xs">분석 중…</li>
                )}
              </ul>
            </div>
            <div className="flex-shrink-0 border-l border-[#E5E8EB] pl-2">
              <BodyAngleMini knee={kneeAngle} hip={hipAngle} kneeStatus={kneeStatus} hipStatus={hipStatus} />
            </div>
          </div>
        </div>
      </div>

      {/* 우상단: 타이머 */}
      <div className="absolute top-3 right-3">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-white/90 px-3 py-1 text-aura-ink text-xs font-semibold shadow-sm">
          <span className={cn("size-1.5 rounded-full", WS_LABEL[wsStatus].dot)} aria-hidden />
          <span className="tabular-nums">{formatTimer(elapsedMs)}</span>
        </span>
      </div>

      {/* 우하단: 종료 호버 버튼 */}
      <div ref={endBtnRef} className="absolute bottom-3 right-3">
        <button
          type="button"
          onClick={onEnd}
          disabled={ending}
          className={cn(
            "relative inline-flex items-center gap-2 rounded-full bg-[#191F28] text-white text-sm font-bold px-6 py-3 shadow-[0_10px_28px_-10px_rgba(0,0,0,0.6)] overflow-hidden transition-all",
            endHover.activeKey === "end" && "scale-105 ring-4 ring-[#E6FB4D]/70",
          )}
        >
          <span className="relative z-10 inline-flex items-center gap-1.5">
            <span className="size-2.5 rounded-[2px] bg-[#FF5A1F]" aria-hidden />
            {ending
              ? "종료 중…"
              : endHover.activeKey === "end"
                ? "종료합니다…"
                : "종료"}
          </span>
          {endHover.activeKey === "end" && endHover.progress > 0 && (
            <span
              className="absolute inset-y-0 left-0 bg-white/20"
              style={{ width: `${endHover.progress * 100}%` }}
              aria-hidden
            />
          )}
        </button>
      </div>

      {/* WS 끊김 알림 (가운데) */}
      {wsStatus === "failed" && !ending && (
        <div
          role="alert"
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white border border-[#E5E8EB] px-5 py-4 shadow-lg max-w-xs"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="size-2 rounded-full bg-[#FF5A1F]" aria-hidden />
            <p className="text-aura-ink text-sm font-bold">노트북과 연결이 끊어졌어요</p>
          </div>
          <button
            type="button"
            onClick={onEnd}
            disabled={ending}
            className="w-full rounded-full bg-[#191F28] text-white text-sm font-semibold py-2.5"
          >
            종료
          </button>
        </div>
      )}

      {/* 손 커서 (호버 활성 시) */}
      {endHover.cursor && (
        <div
          className="pointer-events-none fixed size-5 rounded-full bg-aura-primary/30 ring-2 ring-aura-primary z-40"
          style={{ left: endHover.cursor.x - 10, top: endHover.cursor.y - 10 }}
          aria-hidden
        />
      )}
    </main>
  )
}
