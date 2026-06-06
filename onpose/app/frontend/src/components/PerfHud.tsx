import { useEffect, useState } from "react"

import { Glass } from "@/components/aura"
import { perf, type PerfChannel, type Stats } from "@/lib/perf"

const REFRESH_MS = 1000
const E2E_BUDGET_MS = 200 // PLAN.md §검증 W11

const ROWS: { key: PerfChannel; label: string }[] = [
  { key: "inference", label: "추론" },
  { key: "wsRoundtrip", label: "WS 왕복" },
  { key: "analyze", label: "분석" },
  { key: "network", label: "네트" },
]

export default function PerfHud() {
  const [snap, setSnap] = useState<Record<PerfChannel, Stats>>(() => ({
    inference: perf.inference.snapshot(),
    wsRoundtrip: perf.wsRoundtrip.snapshot(),
    analyze: perf.analyze.snapshot(),
    network: perf.network.snapshot(),
  }))

  useEffect(() => {
    const id = window.setInterval(() => {
      setSnap({
        inference: perf.inference.snapshot(),
        wsRoundtrip: perf.wsRoundtrip.snapshot(),
        analyze: perf.analyze.snapshot(),
        network: perf.network.snapshot(),
      })
    }, REFRESH_MS)
    return () => window.clearInterval(id)
  }, [])

  const e2e = snap.inference.p50 + snap.wsRoundtrip.p50
  const overBudget = e2e > E2E_BUDGET_MS

  return (
    <div className="pointer-events-none fixed right-3 bottom-3 z-30 max-w-[180px]">
      <Glass strong className="pointer-events-auto p-3 font-mono text-[10px] leading-tight">
        <div className="text-aura-ter mb-1 flex items-center justify-between text-[9px] uppercase tracking-wider">
          <span>perf · p50/p95</span>
          <span className={overBudget ? "text-rose-500" : "text-emerald-600"}>
            e2e {e2e.toFixed(0)}
          </span>
        </div>
        <table className="text-aura-ink w-full tabular-nums">
          <tbody>
            {ROWS.map(({ key, label }) => {
              const s = snap[key]
              return (
                <tr key={key}>
                  <td className="text-aura-sec pr-1">{label}</td>
                  <td className="pr-1 text-right">{s.count ? s.p50.toFixed(0) : "·"}</td>
                  <td className="text-aura-ter text-right">
                    {s.count ? s.p95.toFixed(0) : "·"}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Glass>
    </div>
  )
}
