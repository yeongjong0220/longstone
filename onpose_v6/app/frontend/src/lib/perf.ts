// PLAN.md §검증 W11: 컴포넌트별 ms 측정 인프라.
// 멘토 영종 요청 — e2e 사이클 < 200ms / LLM < 5s 검증용.

const WINDOW_SIZE = 60 // ≈6초 @ 10fps

export interface Stats {
  count: number
  p50: number
  p95: number
  avg: number
}

const EMPTY: Stats = { count: 0, p50: 0, p95: 0, avg: 0 }

/** 고정 크기 링버퍼. 최근 N 샘플의 p50/p95/avg 계산. */
export class RollingStats {
  private readonly size: number
  private readonly buf: number[]
  private idx = 0
  private filled = false

  constructor(size = WINDOW_SIZE) {
    this.size = size
    this.buf = new Array<number>(size)
  }

  push(v: number): void {
    this.buf[this.idx] = v
    this.idx = (this.idx + 1) % this.size
    if (this.idx === 0) this.filled = true
  }

  /** Pure read — 정렬은 매 호출 시 (HUD가 1Hz 갱신이라 OK). */
  snapshot(): Stats {
    const n = this.filled ? this.size : this.idx
    if (n === 0) return EMPTY
    const sorted = this.buf.slice(0, n).sort((a, b) => a - b)
    let sum = 0
    for (const v of sorted) sum += v
    return {
      count: n,
      p50: sorted[Math.floor(n * 0.5)],
      p95: sorted[Math.min(n - 1, Math.floor(n * 0.95))],
      avg: sum / n,
    }
  }

  reset(): void {
    this.idx = 0
    this.filled = false
  }
}

/** 측정 채널. 코칭 세션 동안 공유되는 4개 stage 인스턴스. */
export const perf = {
  inference: new RollingStats(),
  wsRoundtrip: new RollingStats(),
  analyze: new RollingStats(),
  network: new RollingStats(),
} as const

export type PerfChannel = keyof typeof perf

export function resetPerf(): void {
  for (const k of Object.keys(perf) as PerfChannel[]) perf[k].reset()
}
