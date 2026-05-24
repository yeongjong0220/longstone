import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"

import { BackHeader, Glass } from "@/components/aura"
import { Button } from "@/components/ui/button"
import { createSession, getReport, type Report as ReportData } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useSessionStore } from "@/stores/sessionStore"

export default function Report() {
  const navigate = useNavigate()
  const sessionId = useSessionStore((s) => s.sessionId)
  const exerciseId = useSessionStore((s) => s.exerciseId)
  const exerciseName = useSessionStore((s) => s.exerciseName)
  const reps = useSessionStore((s) => s.reps)
  const sets = useSessionStore((s) => s.sets)
  const recordedUrl = useSessionStore((s) => s.recordedUrl)
  const setSession = useSessionStore((s) => s.setSession)
  const reset = useSessionStore((s) => s.reset)

  const [report, setReport] = useState<ReportData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!sessionId) {
      navigate("/")
      return
    }
    let cancelled = false
    let timer: number | null = null

    const poll = async () => {
      try {
        const res = await getReport(sessionId)
        if (cancelled) return
        if (res.status === "ready") {
          setReport(res.report)
          return
        }
        timer = window.setTimeout(poll, 1000)
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

  const onAnotherSet = async () => {
    if (!exerciseId || !exerciseName) {
      navigate("/")
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
    navigate("/")
  }

  const scoreTone =
    report === null
      ? "text-aura-ter"
      : report.score_avg >= 80
        ? "text-emerald-600"
        : report.score_avg >= 60
          ? "text-amber-600"
          : "text-rose-600"

  return (
    <main className="min-h-svh flex flex-col gap-3 pb-6">
      <BackHeader title="운동 리포트" onBack={onHome} action={<span className="text-aura-ter text-xs">{exerciseName ?? ""}</span>} />

      {error && (
        <p className="text-destructive text-center text-sm whitespace-pre-line break-keep px-4" role="alert">
          {error}
        </p>
      )}

      {!error && !report && (
        <div className="px-4">
          <Glass strong className="p-4 text-center">
            <p className="text-aura-ter text-sm">코칭 결과 분석 중…</p>
          </Glass>
        </div>
      )}

      {report && (
        <div className="mx-auto flex w-full max-w-md flex-col gap-3 px-4">
          {/* 영상 + 점수 오버레이 */}
          <div className="border-aura-glass-border relative overflow-hidden rounded-2xl border bg-black aspect-[3/4]">
            {recordedUrl ? (
              <video
                src={recordedUrl}
                autoPlay
                loop
                muted
                playsInline
                controls
                className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]"
              />
            ) : (
              <div className="absolute inset-0 grid place-items-center text-white/70 text-sm">
                녹화 영상 없음
              </div>
            )}
            <div className="absolute left-3 top-3 rounded-2xl bg-white/95 px-4 py-2 shadow-md backdrop-blur-md">
              <div className="text-aura-ter text-[10px]">평균 정확도</div>
              <div className={cn("text-3xl font-bold tabular-nums leading-tight", scoreTone)}>
                {report.score_avg}
                <span className="text-aura-ter text-sm font-normal"> /100</span>
              </div>
            </div>
            <div className="absolute right-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-[10px] text-white backdrop-blur-md">
              나의 영상 · 좌우 반전
            </div>
          </div>

          {/* LLM 코칭 메시지 (영상 바로 아래 — 같이 보이도록) */}
          <Glass strong className="p-4">
            <h2 className="text-aura-ter text-xs font-medium mb-1">AI 코치 한마디</h2>
            <p className="text-aura-ink text-sm leading-relaxed break-keep">{report.llm_msg}</p>
          </Glass>

          {/* 잘한 점 / 개선점 */}
          <div className="grid grid-cols-2 gap-2">
            <Glass strong className="p-3">
              <h3 className="text-emerald-700 text-xs font-bold mb-2 inline-flex items-center gap-1">
                <span className="grid size-4 place-items-center rounded-full bg-emerald-100">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="4 12 10 18 20 6" />
                  </svg>
                </span>
                잘한 점
              </h3>
              <ul className="text-aura-ink space-y-1 text-[12px] leading-snug">
                {report.good_points.length > 0 ? (
                  report.good_points.map((p, i) => (
                    <li key={i} className="break-keep">· {p}</li>
                  ))
                ) : (
                  <li className="text-aura-ter">기록 없음</li>
                )}
              </ul>
            </Glass>

            <Glass strong className="p-3">
              <h3 className="text-amber-700 text-xs font-bold mb-2 inline-flex items-center gap-1">
                <span className="grid size-4 place-items-center rounded-full bg-amber-100">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 3 2 21h20L12 3z" />
                    <line x1="12" y1="10" x2="12" y2="14" />
                    <circle cx="12" cy="17.5" r="0.7" fill="currentColor" stroke="none" />
                  </svg>
                </span>
                개선점
              </h3>
              <ul className="text-aura-ink space-y-1 text-[12px] leading-snug">
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
        </div>
      )}

      <div className="mx-auto flex w-full max-w-md flex-col gap-2 px-4">
        <Button
          size="lg"
          onClick={onAnotherSet}
          disabled={busy || !exerciseId}
          className="text-white"
          style={{ background: "var(--gradient-cta)" }}
        >
          {busy ? "준비 중…" : "한 세트 더"}
        </Button>
        <Button size="lg" variant="ghost" onClick={onHome}>
          홈으로
        </Button>
      </div>
    </main>
  )
}
