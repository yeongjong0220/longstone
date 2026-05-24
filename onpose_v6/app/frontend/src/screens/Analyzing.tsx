import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"

import { Glass } from "@/components/aura"
import { Button } from "@/components/ui/button"
import { getReport } from "@/lib/api"
import { useSessionStore } from "@/stores/sessionStore"

const POLL_INTERVAL_MS = 1000
const TOTAL_TIMEOUT_MS = 60_000

export default function Analyzing() {
  const navigate = useNavigate()
  const sessionId = useSessionStore((s) => s.sessionId)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!sessionId) {
      navigate("/")
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
          navigate("/report")
          return
        }
        if (Date.now() - startedAt > TOTAL_TIMEOUT_MS) {
          setError("분석이 너무 오래 걸려요. 잠시 후 다시 시도해 주세요.")
          return
        }
        timer = window.setTimeout(poll, POLL_INTERVAL_MS)
      } catch (e) {
        if (cancelled) return
        setError((e as Error).message)
      }
    }
    void poll()
    return () => {
      cancelled = true
      if (timer !== null) window.clearTimeout(timer)
    }
  }, [sessionId, navigate])

  if (error) {
    return (
      <main className="min-h-svh flex flex-col items-center justify-center gap-6 px-6 text-center">
        <Glass strong className="flex w-full max-w-xs flex-col items-center gap-4 p-8">
          <h1 className="text-aura-ink text-2xl font-semibold tracking-tight">분석 실패</h1>
          <p
            className="text-destructive text-sm whitespace-pre-line break-keep"
            role="alert"
          >
            {error}
          </p>
          <Button variant="ghost" onClick={() => navigate("/")}>
            홈으로
          </Button>
        </Glass>
      </main>
    )
  }

  return (
    <main className="min-h-svh flex flex-col items-center justify-center gap-6 px-6 text-center">
      <Glass strong className="flex w-full max-w-xs flex-col items-center gap-4 p-8">
        <div
          aria-hidden
          className="border-aura-mute border-t-aura-primary size-12 animate-spin rounded-full border-4"
        />
        <div>
          <h1 className="text-aura-ink text-2xl font-semibold tracking-tight">분석 중</h1>
          <p className="text-aura-ter mt-1 text-sm">
            코칭 메시지를 생성하고 있어요.
          </p>
        </div>
      </Glass>
    </main>
  )
}
