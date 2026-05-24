// FastAPI 백엔드(8000)는 Vite dev(5173)와 다른 포트라 절대 URL이 필요하다.
// VITE_API_BASE가 있으면 우선, 없으면 현재 호스트의 8000 포트를 가정.
const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  `${window.location.protocol}//${window.location.hostname}:8000`

export interface Exercise {
  id: string
  name: string
  reps_default: number
  sets_default: number
}

export interface SessionCreateResp {
  session_id: string
}

export interface Report {
  score_avg: number
  good_points: string[]
  improvements: string[]
  llm_msg: string
}

const DEFAULT_TIMEOUT_MS = 5000
const OFFLINE_MSG = "노트북에 연결되지 않았습니다.\n같은 네트워크인지 확인해 주세요."

async function request<T>(
  path: string,
  init?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...rest } = init ?? {}
  const ctrl = new AbortController()
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...rest,
      signal: ctrl.signal,
    })
    if (!res.ok) {
      throw new Error(`${rest.method ?? "GET"} ${path} → ${res.status}`)
    }
    return (await res.json()) as T
  } catch (e) {
    const err = e as Error
    if (err.name === "AbortError" || err instanceof TypeError) {
      throw new Error(OFFLINE_MSG)
    }
    throw err
  } finally {
    window.clearTimeout(timer)
  }
}

export function listExercises(): Promise<Exercise[]> {
  return request<Exercise[]>("/api/exercises")
}

export function createSession(
  exercise_id: string,
  reps: number,
  sets: number,
): Promise<SessionCreateResp> {
  return request<SessionCreateResp>("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ exercise_id, reps, sets }),
  })
}

export function endSession(session_id: string): Promise<{ session_id: string; ended_at: number }> {
  return request(`/api/sessions/${session_id}/end`, { method: "POST" })
}

export type ReportStatus =
  | { status: "pending" }
  | { status: "ready"; report: Report }

export function getReport(session_id: string): Promise<ReportStatus> {
  return request<ReportStatus>(`/api/sessions/${session_id}/report`)
}

export function wsUrl(session_id: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:"
  return `${proto}//${window.location.hostname}:8000/ws/pose/${session_id}`
}
