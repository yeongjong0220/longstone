import { useEffect, useRef, useState } from "react"

import { wsUrl } from "@/lib/api"
import { perf } from "@/lib/perf"
import type { CoachingFrame } from "@/stores/sessionStore"
import type { Landmark } from "./usePoseDetector"

export type WsStatus = "idle" | "connecting" | "open" | "closed" | "failed"

interface UseWebSocketArgs {
  sessionId: string | null
  enabled: boolean
  /** ms between outbound frames. 100 = 10fps. */
  sendIntervalMs?: number
  /** Source of latest landmarks. Read on each send tick. */
  landmarksRef: React.RefObject<Landmark[]>
  onMessage?: (frame: CoachingFrame) => void
}

const MAX_RETRIES = 5

export function useWebSocket({
  sessionId,
  enabled,
  sendIntervalMs = 100,
  landmarksRef,
  onMessage,
}: UseWebSocketArgs) {
  const [status, setStatus] = useState<WsStatus>("idle")
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  useEffect(() => {
    if (!enabled || !sessionId) return
    let ws: WebSocket | null = null
    let sendTimer: number | null = null
    let cancelled = false
    let retries = 0
    // t(=Date.now()/1000) → performance.now() at send. 응답 도착 시 매칭.
    const sentAt = new Map<number, number>()

    const connect = () => {
      if (cancelled) return
      setStatus("connecting")
      ws = new WebSocket(wsUrl(sessionId))

      ws.onopen = () => {
        if (cancelled) return
        retries = 0
        setStatus("open")
        sendTimer = window.setInterval(() => {
          if (!ws || ws.readyState !== WebSocket.OPEN) return
          const lms = landmarksRef.current
          if (!lms || lms.length === 0) return
          const t = Date.now() / 1000
          sentAt.set(t, performance.now())
          // Map이 누수되지 않도록 오래된 엔트리 정리 (응답 누락 대비).
          if (sentAt.size > 50) {
            const oldest = sentAt.keys().next().value
            if (oldest !== undefined) sentAt.delete(oldest)
          }
          ws.send(JSON.stringify({ t, landmarks: lms }))
        }, sendIntervalMs)
      }

      ws.onmessage = (ev) => {
        try {
          const frame = JSON.parse(ev.data) as CoachingFrame & {
            _dbg?: { analyze_ms?: number }
          }
          const sent = sentAt.get(frame.t)
          if (sent !== undefined) {
            sentAt.delete(frame.t)
            const rt = performance.now() - sent
            perf.wsRoundtrip.push(rt)
            const analyze = frame._dbg?.analyze_ms
            if (typeof analyze === "number") {
              perf.analyze.push(analyze)
              perf.network.push(Math.max(0, rt - analyze))
            }
          }
          onMessageRef.current?.(frame)
        } catch {
          // ignore non-JSON frames
        }
      }

      ws.onclose = () => {
        if (sendTimer != null) {
          clearInterval(sendTimer)
          sendTimer = null
        }
        if (cancelled) return
        if (retries < MAX_RETRIES) {
          retries += 1
          setStatus("connecting")
          setTimeout(connect, 500 * retries)
        } else {
          setStatus("failed")
        }
      }

      ws.onerror = () => {
        // onclose will fire next and handle retry / failure state.
      }
    }

    connect()

    return () => {
      cancelled = true
      if (sendTimer != null) clearInterval(sendTimer)
      if (ws && ws.readyState <= WebSocket.OPEN) ws.close()
      setStatus("closed")
    }
  }, [sessionId, enabled, sendIntervalMs, landmarksRef])

  return { status }
}
