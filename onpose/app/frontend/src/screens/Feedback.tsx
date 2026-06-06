import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"

import { Glass } from "@/components/aura"
import { Button } from "@/components/ui/button"
import { useOrientationLock } from "@/hooks/useOrientationLock"
import { createSession, getReport, type Report as ReportData } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useSessionStore } from "@/stores/sessionStore"

const POLL_INTERVAL_MS = 1000
const TOTAL_TIMEOUT_MS = 60_000

export default function Feedback() {
  const navigate = useNavigate()
  useOrientationLock("portrait")
  const sessionId = useSessionStore((s) => s.sessionId)
  const exerciseId = useSessionStore((s) => s.exerciseId)
  const exerciseName = useSessionStore((s) => s.exerciseName)
  const reps = useSessionStore((s) => s.reps)
  const sets = useSessionStore((s) => s.sets)
  const setSession = useSessionStore((s) => s.setSession)
  const reset = useSessionStore((s) => s.reset)

  const [report, setReport] = useState<ReportData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!sessionId) {
      navigate("/home", { replace: true })
      return
    }
    let cancelled = false
    let timer: number | null = null
    const startedAt = Date.now()

    const poll = async () => {
      try {
        const res = await getReport(sessionId)
        if (cancelled) return
        if (res.status === "ready") {
          setReport(res.report)
          return
        }
        if (Date.now() - startedAt > TOTAL_TIMEOUT_MS) {
          setError("분석이 너무 오래 걸려요. 잠시 후 다시 시도해 주세요.")
          return
        }
        timer = window.setTimeout(poll, POLL_INTERVAL_MS)
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    }
    void poll()
    return () => {
      cancelled = true
      if (timer !== null) window.clearTimeout(timer)
    }
  }, [sessionId, navigate])

  const onRestart = async () => {
    if (!exerciseId || !exerciseName) {
      navigate("/home")
      return
    }
    setBusy(true)
    try {
      const { session_id } = await createSession(exerciseId, reps, sets)
      setSession({ sessionId: session_id, exerciseId, exerciseName, reps, sets })
      navigate("/setup")
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }

  const onHome = () => {
    reset()
    navigate("/home")
  }

  // 1. 에러
  if (error) {
    return (
      <main className="min-h-svh flex flex-col items-center justify-center gap-6 px-6 text-center">
        <Glass className="flex w-full max-w-xs flex-col items-center gap-4 p-8">
          <h1 className="text-aura-ink text-2xl font-semibold tracking-tight">분석 실패</h1>
          <p className="text-destructive text-sm whitespace-pre-line break-keep" role="alert">{error}</p>
          <Button variant="outline" onClick={onHome}>홈으로</Button>
        </Glass>
      </main>
    )
  }

  // 2. 분석 중 (로딩)
  if (!report) {
    return (
      <main className="min-h-svh flex flex-col items-center justify-center gap-6 px-6 text-center">
        <Glass className="flex w-full max-w-xs flex-col items-center gap-4 p-8">
          <div
            aria-hidden
            className="border-aura-mute border-t-aura-primary size-12 animate-spin rounded-full border-4"
          />
          <div>
            <h1 className="text-aura-ink text-2xl font-semibold tracking-tight">분석 중</h1>
            <p className="text-aura-ter mt-1 text-sm">코칭 메시지를 생성하고 있어요.</p>
          </div>
        </Glass>
      </main>
    )
  }

  // 3. 결과 표시
  const scoreTone =
    report.score_avg >= 80
      ? "text-[#E6FB4D]"
      : report.score_avg >= 60
        ? "text-[#FFD64D]"
        : "text-[#FF8A5C]"

  return (
    <main className="min-h-svh flex flex-col gap-4 px-4 pb-6 pt-6">
      <header className="flex flex-col items-center gap-1 text-center">
        <p className="text-white/55 text-xs">{exerciseName ?? "운동"}</p>
        <h1 className="text-white text-2xl font-bold">분석 피드백</h1>
        <div className={cn("mt-1 text-5xl font-extrabold tabular-nums", scoreTone)}>
          {report.score_avg}
          <span className="text-white/45 text-base font-normal"> /100</span>
        </div>
      </header>

      <div className="mx-auto w-full max-w-md flex flex-col gap-3">
        {/* 잘 / 못 그리드 */}
        <div className="grid grid-cols-2 gap-3">
          <Glass className="p-4">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="grid size-5 place-items-center rounded-full bg-aura-primary/10">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#4E45E6" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="4 12 10 18 20 6" />
                </svg>
              </span>
              <h2 className="text-aura-primary text-sm font-bold">잘한점</h2>
            </div>
            <ul className="text-aura-ink space-y-1.5 text-[13px] leading-snug">
              {report.good_points.length > 0 ? (
                report.good_points.map((p, i) => (
                  <li key={i} className="break-keep">· {p}</li>
                ))
              ) : (
                <li className="text-aura-ter">기록 없음</li>
              )}
            </ul>
          </Glass>

          <Glass className="p-4">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="grid size-5 place-items-center rounded-full bg-amber-100">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#D97706" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 3 2 21h20L12 3z" />
                  <line x1="12" y1="10" x2="12" y2="14" />
                </svg>
              </span>
              <h2 className="text-amber-700 text-sm font-bold">개선점</h2>
            </div>
            <ul className="text-aura-ink space-y-1.5 text-[13px] leading-snug">
              {report.improvements.length > 0 ? (
                report.improvements.map((p, i) => (
                  <li key={i} className="break-keep">· {p}</li>
                ))
              ) : (
                <li className="text-aura-ter">기록 없음</li>
              )}
            </ul>
          </Glass>
        </div>

        {/* AI 코치 한마디 */}
        <Glass className="p-4">
          <h2 className="text-aura-ter text-xs font-medium mb-1.5">AI 코치 한마디</h2>
          <p className="text-aura-ink text-sm leading-relaxed break-keep">{report.llm_msg}</p>
        </Glass>
      </div>

      <div className="mx-auto mt-auto flex w-full max-w-md flex-col gap-2">
        <Button size="lg" onClick={onRestart} disabled={busy || !exerciseId}>
          {busy ? "준비 중…" : "재시작"}
        </Button>
        <Button size="lg" variant="outline" onClick={onHome}>홈으로</Button>
      </div>
    </main>
  )
}
