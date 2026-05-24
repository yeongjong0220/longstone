import { useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"

import PerfHud from "@/components/PerfHud"
import PoseOverlay from "@/components/PoseOverlay"
import { BackHeader, Glass } from "@/components/aura"
import { Button } from "@/components/ui/button"
import { useCamera } from "@/hooks/useCamera"
import { usePoseDetector } from "@/hooks/usePoseDetector"
import { useRecorder } from "@/hooks/useRecorder"
import { useWebSocket, type WsStatus } from "@/hooks/useWebSocket"
import { endSession } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useSessionStore, type CoachingFrame, type FeedbackItem } from "@/stores/sessionStore"

const COUNTDOWN_START = 3

const STATUS_TONE: Record<CoachingFrame["status"], string> = {
  good: "text-emerald-600",
  warn: "text-amber-600",
  err: "text-rose-600",
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
      <span className="bg-emerald-100 text-emerald-700 grid size-6 flex-shrink-0 place-items-center rounded-full">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="4 12 10 18 20 6" />
        </svg>
      </span>
    )
  }
  if (level === "warn") {
    return (
      <span className="bg-amber-100 text-amber-700 grid size-6 flex-shrink-0 place-items-center rounded-full">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3 2 21h20L12 3z" />
          <line x1="12" y1="10" x2="12" y2="14" />
          <circle cx="12" cy="17.5" r="0.7" fill="currentColor" stroke="none" />
        </svg>
      </span>
    )
  }
  return (
    <span className="bg-sky-100 text-sky-700 grid size-6 flex-shrink-0 place-items-center rounded-full">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="9" />
        <line x1="12" y1="8" x2="12" y2="8.01" />
        <path d="M11 12h1v5h1" />
      </svg>
    </span>
  )
}

function BodyAngleMini({
  knee,
  hip,
  status,
}: {
  knee: number | null
  hip: number | null
  status: CoachingFrame["status"]
}) {
  const label = STATUS_LABEL[status]
  const tone = STATUS_TONE[status]
  return (
    <div className="flex items-center gap-3">
      <svg width="56" height="80" viewBox="0 0 56 80" aria-hidden>
        <circle cx="22" cy="10" r="6" fill="#E5E7EB" />
        <line x1="22" y1="16" x2="22" y2="36" stroke="#374151" strokeWidth="3" strokeLinecap="round" />
        <line x1="22" y1="22" x2="32" y2="36" stroke="#9CA3AF" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="22" cy="36" r="3" fill="#F97316" />
        <line x1="22" y1="36" x2="42" y2="48" stroke="#374151" strokeWidth="3" strokeLinecap="round" />
        <circle cx="42" cy="48" r="3" fill="#22C55E" />
        <line x1="42" y1="48" x2="36" y2="72" stroke="#374151" strokeWidth="3" strokeLinecap="round" />
      </svg>
      <div className="flex flex-col gap-2 text-xs">
        <div>
          <div className="text-aura-ter">무릎 각도</div>
          <div className="text-aura-ink text-lg font-bold tabular-nums leading-none">
            {knee !== null ? `${Math.round(knee)}°` : "—"}
          </div>
          <div className={cn("text-[10px] font-medium", knee !== null ? tone : "text-aura-ter")}>
            {knee !== null ? label : "—"}
          </div>
        </div>
        <div>
          <div className="text-aura-ter">엉덩이 각도</div>
          <div className="text-aura-ink text-lg font-bold tabular-nums leading-none">
            {hip !== null ? `${Math.round(hip)}°` : "—"}
          </div>
          <div className={cn("text-[10px] font-medium", hip !== null ? tone : "text-aura-ter")}>
            {hip !== null ? label : "—"}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Coaching() {
  const navigate = useNavigate()
  const sessionId = useSessionStore((s) => s.sessionId)
  const exerciseId = useSessionStore((s) => s.exerciseId)
  const exerciseName = useSessionStore((s) => s.exerciseName)
  const reps = useSessionStore((s) => s.reps)
  const sets = useSessionStore((s) => s.sets)
  const setCoaching = useSessionStore((s) => s.setCoaching)
  const lastCoaching = useSessionStore((s) => s.lastCoaching)
  const setRecordedUrl = useSessionStore((s) => s.setRecordedUrl)

  const [countdown, setCountdown] = useState<number | null>(COUNTDOWN_START)
  const [paused, setPaused] = useState(false)
  const [ending, setEnding] = useState(false)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [expertReady, setExpertReady] = useState(true)
  const endedRef = useRef(false)
  const startedAtRef = useRef<number | null>(null)

  const { videoRef, status: camStatus } = useCamera(true)
  const { latestRef } = usePoseDetector(videoRef, camStatus === "ready")
  const wsEnabled = camStatus === "ready" && countdown === null && !paused
  const { status: wsStatus } = useWebSocket({
    sessionId,
    enabled: wsEnabled,
    landmarksRef: latestRef,
    onMessage: (frame: CoachingFrame) => setCoaching(frame),
  })
  const recorder = useRecorder(videoRef)

  // 카운트다운
  useEffect(() => {
    if (countdown === null || countdown <= 0) {
      if (countdown !== null) setCountdown(null)
      return
    }
    const t = window.setTimeout(() => setCountdown((c) => (c === null ? null : c - 1)), 1000)
    return () => window.clearTimeout(t)
  }, [countdown])

  // 카운트다운 끝나면 녹화 시작 + 타이머 시작
  useEffect(() => {
    if (countdown !== null) return
    if (startedAtRef.current === null) startedAtRef.current = performance.now()
    if (recorder.state === "idle" && camStatus === "ready") recorder.start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [countdown, camStatus])

  // 타이머
  useEffect(() => {
    if (countdown !== null || paused || ending) return
    const id = window.setInterval(() => {
      if (startedAtRef.current !== null) {
        setElapsedMs(performance.now() - startedAtRef.current)
      }
    }, 250)
    return () => window.clearInterval(id)
  }, [countdown, paused, ending])

  // 녹화 결과 URL을 store에 반영 (Report에서 사용)
  useEffect(() => {
    if (recorder.lastUrl) setRecordedUrl(recorder.lastUrl)
  }, [recorder.lastUrl, setRecordedUrl])

  const onEnd = async () => {
    if (endedRef.current) return
    endedRef.current = true
    if (recorder.state === "recording") recorder.stop()
    if (!sessionId) {
      navigate("/")
      return
    }
    setEnding(true)
    try {
      await endSession(sessionId)
    } catch {
      /* Analyzing/Report에서 에러 표시 담당 */
    }
    navigate("/analyzing")
  }

  // rep 목표 도달 시 자동 종료
  const totalTarget = reps * sets
  const rawRepCount = lastCoaching?.rep_count ?? 0
  useEffect(() => {
    if (totalTarget > 0 && rawRepCount >= totalTarget) void onEnd()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rawRepCount, totalTarget])

  const currentSetIdx = reps > 0 ? Math.min(Math.floor(rawRepCount / reps), sets - 1) : 0
  const repInSet = reps > 0 ? rawRepCount % reps : 0
  const score = lastCoaching?.score ?? 0
  const status: CoachingFrame["status"] = lastCoaching?.status ?? "good"
  const feedbackList: FeedbackItem[] = lastCoaching?.feedback?.slice(0, 3) ?? []
  const angles = lastCoaching?.angles ?? {}
  const kneeAngle = angles.knee ?? null
  const hipAngle = angles.hip ?? null
  const phase = lastCoaching?.phase ?? "ready"

  return (
    <main className="min-h-svh flex flex-col gap-3 pb-4">
      <BackHeader
        title="실시간 자세 코칭"
        onBack={onEnd}
        action={
          <span className="text-aura-ter inline-flex items-center gap-1.5 text-xs">
            <span className={cn("size-1.5 rounded-full", WS_LABEL[wsStatus].dot)} aria-hidden />
            {WS_LABEL[wsStatus].text}
          </span>
        }
      />

      <div className="mx-auto w-full max-w-md px-4">
        <Glass strong className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="grid size-7 place-items-center rounded-full bg-aura-primary/10 text-aura-primary flex-shrink-0">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="6.5" y1="6.5" x2="6.5" y2="17.5" />
                <line x1="17.5" y1="6.5" x2="17.5" y2="17.5" />
                <line x1="4" y1="9" x2="4" y2="15" />
                <line x1="20" y1="9" x2="20" y2="15" />
                <line x1="6.5" y1="12" x2="17.5" y2="12" />
              </svg>
            </span>
            <span className="text-aura-ink text-sm font-semibold truncate">
              {exerciseName ?? "운동"}
            </span>
          </div>
          <span className="text-aura-ink text-sm font-semibold tabular-nums flex-shrink-0">
            {formatTimer(elapsedMs)}
          </span>
        </Glass>
      </div>

      {wsStatus === "failed" && !ending && (
        <div role="alert" className="mx-auto w-full max-w-md px-4">
          <Glass strong className="flex items-center gap-3 p-3">
            <span className="size-2 flex-shrink-0 rounded-full bg-rose-500" aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="text-aura-ink text-sm font-semibold">노트북과 연결이 끊어졌어요</p>
            </div>
            <Button size="sm" variant="destructive" onClick={onEnd} disabled={ending}>
              종료
            </Button>
          </Glass>
        </div>
      )}

      <div className="mx-auto w-full max-w-md px-4">
        <div className="border-aura-glass-border relative overflow-hidden rounded-2xl border bg-black aspect-[3/4]">
          <video ref={videoRef} className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]" />
          <PoseOverlay landmarksRef={latestRef} className="absolute inset-0 h-full w-full" />

          {countdown !== null && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/45">
              <span className="text-9xl font-semibold text-white drop-shadow-lg tabular-nums">
                {countdown === 0 ? "GO" : countdown}
              </span>
            </div>
          )}

          {/* LIVE chip - 좌하단 */}
          <div className="absolute left-3 bottom-3 flex items-center gap-2 rounded-full bg-black/70 px-3 py-1.5 text-white text-[11px] font-medium backdrop-blur-md">
            <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" aria-hidden />
            <div className="leading-tight">
              <div>LIVE</div>
              <div className="text-[9px] text-white/70">On-device AI</div>
            </div>
          </div>

          {/* 3D 자세 - 우하단 */}
          <div className="absolute right-3 bottom-3 flex items-center gap-1.5 rounded-full bg-black/70 px-3 py-1.5 text-white text-[11px] font-medium backdrop-blur-md">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
              <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
              <line x1="12" y1="22.08" x2="12" y2="12" />
            </svg>
            <span>3D 자세</span>
          </div>

          {/* 우측 메트릭 사이드 패널 */}
          <div className="absolute right-2 top-2 w-[36%] max-w-[150px]">
            <div className="rounded-2xl bg-white/95 backdrop-blur-md shadow-lg overflow-hidden divide-y divide-aura-glass-border">
              <div className="px-3 py-2 text-center">
                <div className="text-aura-ter text-[10px]">자세 정확도</div>
                <div className={cn("text-2xl font-bold tabular-nums leading-tight", STATUS_TONE[status])}>
                  {score}%
                </div>
                <div className={cn("text-[10px] font-medium", STATUS_TONE[status])}>{STATUS_LABEL[status]}</div>
              </div>
              <div className="px-3 py-2 text-center">
                <div className="text-aura-ter text-[10px]">반복</div>
                <div className="text-aura-ink text-lg font-bold tabular-nums leading-tight">
                  {repInSet}<span className="text-aura-ter text-xs font-normal"> /{reps}</span>
                </div>
              </div>
              <div className="px-3 py-2 text-center">
                <div className="text-aura-ter text-[10px]">세트</div>
                <div className="text-aura-ink text-lg font-bold tabular-nums leading-tight">
                  {Math.min(currentSetIdx + 1, sets)}<span className="text-aura-ter text-xs font-normal"> /{sets}</span>
                </div>
              </div>
              <div className="px-3 py-2 text-center">
                <div className="text-aura-ter text-[10px]">단계</div>
                <div className="text-aura-ink text-sm font-bold leading-tight capitalize">
                  {phase}
                </div>
              </div>
            </div>
          </div>

          {/* 전문가 영상 PIP - 우측 상단 사이드 패널 아래 */}
          {exerciseId && expertReady && (
            <div className="absolute left-3 top-3 w-[28%] max-w-[110px] rounded-xl overflow-hidden border border-white/30 shadow-md bg-black">
              <div className="absolute left-0 right-0 top-0 bg-black/60 text-white text-[9px] font-medium px-1.5 py-0.5 z-10 backdrop-blur-sm">
                전문가 시범
              </div>
              <video
                src={`/videos/${exerciseId}.mp4`}
                autoPlay
                loop
                muted
                playsInline
                onError={() => setExpertReady(false)}
                className="block w-full"
              />
            </div>
          )}
        </div>
      </div>

      {/* 실시간 피드백 + 인체 다이어그램 */}
      <div className="mx-auto w-full max-w-md px-4">
        <Glass strong className="p-4">
          <div className="flex items-start gap-3">
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
            <div className="flex-shrink-0 border-l border-aura-glass-border pl-3">
              <BodyAngleMini knee={kneeAngle} hip={hipAngle} status={status} />
            </div>
          </div>
        </Glass>
      </div>

      {/* On-device / Offline / Privacy 뱃지 */}
      <div className="mx-auto w-full max-w-md px-4">
        <div className="grid grid-cols-3 gap-2 rounded-2xl bg-emerald-50 px-3 py-2.5">
          <Badge
            iconColor="text-emerald-700"
            icon={
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="4" y="4" width="16" height="16" rx="2" />
                <rect x="9" y="9" width="6" height="6" />
                <line x1="9" y1="2" x2="9" y2="4" />
                <line x1="15" y1="2" x2="15" y2="4" />
                <line x1="9" y1="20" x2="9" y2="22" />
                <line x1="15" y1="20" x2="15" y2="22" />
                <line x1="2" y1="9" x2="4" y2="9" />
                <line x1="2" y1="15" x2="4" y2="15" />
                <line x1="20" y1="9" x2="22" y2="9" />
                <line x1="20" y1="15" x2="22" y2="15" />
              </svg>
            }
            title="On-device"
            sub="기기 내 처리"
          />
          <Badge
            iconColor="text-emerald-700"
            icon={
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="2" y1="2" x2="22" y2="22" />
                <path d="M8.5 16.5a5 5 0 017 0" />
                <path d="M5 12.55a11 11 0 015.17-2.39" />
                <path d="M19 12.55a11 11 0 00-3-2.05" />
                <path d="M1.42 9a16 16 0 014.66-2.93" />
                <path d="M22.58 9a16 16 0 00-9.95-3.81" />
                <line x1="12" y1="20" x2="12" y2="20" />
              </svg>
            }
            title="오프라인 모드"
            sub="인터넷 불필요"
          />
          <Badge
            iconColor="text-emerald-700"
            icon={
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="11" width="18" height="11" rx="2" />
                <path d="M7 11V7a5 5 0 0110 0v4" />
              </svg>
            }
            title="개인정보 보호"
            sub="영상 저장 없음"
          />
        </div>
      </div>

      {/* 하단 액션 */}
      <div className="mx-auto grid w-full max-w-md grid-cols-2 gap-2 px-4">
        <Button
          onClick={() => setPaused((p) => !p)}
          disabled={countdown !== null || ending}
          className="rounded-full bg-aura-ink text-white hover:bg-aura-ink/90 h-12"
        >
          <span className="inline-flex items-center gap-2">
            {paused ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4" /></svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="4" width="5" height="16" /><rect x="14" y="4" width="5" height="16" /></svg>
            )}
            {paused ? "이어하기" : "일시정지"}
          </span>
        </Button>
        <Button
          variant="outline"
          onClick={onEnd}
          disabled={ending}
          className="rounded-full bg-white text-aura-ink h-12 border-aura-glass-border"
        >
          <span className="inline-flex items-center gap-2">
            <span className="size-3 rounded-sm bg-rose-500" aria-hidden />
            {ending ? "종료 중…" : "운동 종료"}
          </span>
        </Button>
      </div>

      {import.meta.env.DEV && <PerfHud />}
    </main>
  )
}

type BadgeProps = {
  icon: React.ReactNode
  iconColor: string
  title: string
  sub: string
}

function Badge({ icon, iconColor, title, sub }: BadgeProps) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className={cn("flex-shrink-0", iconColor)}>{icon}</span>
      <div className="min-w-0">
        <div className="text-aura-ink text-[11px] font-semibold leading-tight truncate">{title}</div>
        <div className="text-aura-ter text-[9px] leading-tight truncate">{sub}</div>
      </div>
    </div>
  )
}
