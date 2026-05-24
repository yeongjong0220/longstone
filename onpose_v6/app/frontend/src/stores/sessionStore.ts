import { create } from "zustand"

export interface FeedbackItem {
  level: "ok" | "warn" | "err"
  msg: string
}

export interface CoachingFrame {
  t: number
  phase: string
  rep_count: number
  set_count: number
  angles: Record<string, number>
  score: number
  status: "good" | "warn" | "err"
  feedback: FeedbackItem[]
}

interface SessionState {
  sessionId: string | null
  exerciseId: string | null
  exerciseName: string | null
  reps: number
  sets: number
  lastCoaching: CoachingFrame | null
  recordedUrl: string | null
  setSession: (args: {
    sessionId: string
    exerciseId: string
    exerciseName: string
    reps: number
    sets: number
  }) => void
  setCoaching: (frame: CoachingFrame) => void
  setRecordedUrl: (url: string | null) => void
  reset: () => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  exerciseId: null,
  exerciseName: null,
  reps: 0,
  sets: 0,
  lastCoaching: null,
  recordedUrl: null,
  setSession: ({ sessionId, exerciseId, exerciseName, reps, sets }) => {
    const prev = get().recordedUrl
    if (prev) URL.revokeObjectURL(prev)
    set({ sessionId, exerciseId, exerciseName, reps, sets, lastCoaching: null, recordedUrl: null })
  },
  setCoaching: (frame) => set({ lastCoaching: frame }),
  setRecordedUrl: (url) => {
    const prev = get().recordedUrl
    if (prev && prev !== url) URL.revokeObjectURL(prev)
    set({ recordedUrl: url })
  },
  reset: () => {
    const prev = get().recordedUrl
    if (prev) URL.revokeObjectURL(prev)
    set({
      sessionId: null,
      exerciseId: null,
      exerciseName: null,
      reps: 0,
      sets: 0,
      lastCoaching: null,
      recordedUrl: null,
    })
  },
}))
