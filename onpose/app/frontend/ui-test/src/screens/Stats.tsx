import { GoLogo } from "@/components/GoLogo"
import {
  ArrowDownLeftIcon,
  ArrowUpRightIcon,
  BottleIcon,
  ChevronDownIcon,
  FootprintsIcon,
  WaveformIcon,
} from "@/components/icons"
import { StatusBar } from "@/components/StatusBar"
import { HighlightChip, RoundButton } from "@/components/ui"
import { cn } from "@/lib/cn"

const Y_LABELS = [
  "3K",
  "2.7K",
  "2.4K",
  "2.1K",
  "1.8K",
  "1.5K",
  "1.2K",
  "900",
  "600",
  "300",
]

type Bar = { day: string; h: number; tone: "low" | "mid" | "high"; marker?: boolean }
const BARS: Bar[] = [
  { day: "Wed", h: 22, tone: "low" },
  { day: "Thu", h: 10, tone: "low" },
  { day: "Fri", h: 30, tone: "mid", marker: true },
  { day: "Mon", h: 86, tone: "high" },
  { day: "Tue", h: 12, tone: "low" },
  { day: "Tue", h: 16, tone: "low" },
]

const TONE: Record<Bar["tone"], string> = {
  low: "bg-white/25",
  mid: "bg-white/55",
  high: "bg-white/90",
}

/** 화면 4 — 걸음 통계. 상단 바차트 + 하단 흰 시트. */
export function Stats() {
  return (
    <div className="relative h-full w-full overflow-hidden bg-go-primary text-white">
      <StatusBar />
      <HighlightChip className="absolute left-4 top-2 z-20" />

      {/* 헤더 지표 */}
      <header className="flex items-start justify-between px-6 pt-3">
        <div className="flex gap-6">
          <div>
            <div className="flex items-center gap-1 text-[19px] font-extrabold">
              <ArrowDownLeftIcon className="h-4 w-4 text-emerald-300" />
              2,212
            </div>
            <div className="text-[11px] text-white/60">Steps behind</div>
          </div>
          <div>
            <div className="flex items-center gap-0.5 text-[19px] font-extrabold text-go-orange">
              <ChevronDownIcon className="h-4 w-4" />
              96%
            </div>
            <div className="text-[11px] text-white/60">Lower than yesterday</div>
          </div>
        </div>
        <RoundButton className="bg-white/15" aria-label="물">
          <BottleIcon className="h-4 w-4" />
        </RoundButton>
      </header>

      {/* 바차트 */}
      <div className="mt-5 px-6">
        <div className="relative h-[240px]">
          <div className="absolute inset-0 flex flex-col justify-between">
            {Y_LABELS.map((y) => (
              <div
                key={y}
                className="flex items-center gap-2 text-[10px] text-white/40"
              >
                <span className="w-7 text-right">{y}</span>
                <span className="h-px flex-1 bg-white/10" />
              </div>
            ))}
          </div>
          <div className="absolute inset-y-0 left-9 right-0 flex items-end justify-around">
            {BARS.map((b, i) => (
              <div key={i} className="relative flex h-full w-[18px] items-end">
                <div
                  style={{ height: `${b.h}%` }}
                  className={cn("w-full rounded-full", TONE[b.tone])}
                />
                {b.marker ? (
                  <span
                    style={{ bottom: `${b.h}%` }}
                    className="absolute left-1/2 mb-1 h-2 w-2 -translate-x-1/2 rounded-full bg-white"
                  />
                ) : null}
              </div>
            ))}
          </div>
        </div>
        <div className="ml-9 mt-2 flex justify-around text-[10px] text-white/55">
          {BARS.map((b, i) => (
            <span key={i} className="w-[18px] text-center">
              {b.day}
            </span>
          ))}
        </div>
      </div>

      {/* 흰 시트 */}
      <div className="absolute inset-x-0 bottom-0 rounded-t-[34px] bg-white px-6 pb-7 pt-5 text-go-ink shadow-[0_-18px_40px_rgba(0,0,0,0.22)]">
        <div className="flex items-center justify-between">
          <GoLogo
            variant="solid"
            color="#14141c"
            letterColor="#ffffff"
            className="w-9"
          />
          <div className="flex items-center gap-1 rounded-full bg-go-ink/[0.06] p-1 text-[12px] font-semibold">
            <span className="rounded-full bg-go-ink px-3 py-1 text-white">D</span>
            <span className="px-3 py-1 text-go-muted">W</span>
            <span className="px-3 py-1 text-go-muted">M</span>
          </div>
        </div>

        <div className="mt-4">
          <div className="text-[44px] font-extrabold leading-none">84</div>
          <div className="mt-1 text-[13px] text-go-muted">Steps</div>
        </div>

        <div className="mt-5 flex justify-between">
          <Metric value="0.04" unit="mi" label="Distance" />
          <Metric value="2" unit="kcal" label="Calories" />
          <Metric value="0" label="Floors" />
        </div>

        <div className="mt-6 flex items-center gap-2 rounded-full bg-go-ink/[0.06] p-1.5">
          <button
            type="button"
            className="flex flex-1 items-center justify-center gap-2 rounded-full bg-white px-4 py-2.5 shadow-sm"
          >
            <FootprintsIcon className="h-4 w-4 text-go-primary" />
            <span className="text-[13px] font-semibold">Steps</span>
          </button>
          <button
            type="button"
            className="flex h-10 w-10 items-center justify-center rounded-full text-go-ink/60"
            aria-label="소리"
          >
            <WaveformIcon className="h-4 w-4" />
          </button>
          <button
            type="button"
            className="flex h-10 w-10 items-center justify-center rounded-full bg-go-ink text-white"
            aria-label="더보기"
          >
            <ArrowUpRightIcon className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

function Metric({
  value,
  unit,
  label,
}: {
  value: string
  unit?: string
  label: string
}) {
  return (
    <div>
      <div className="flex items-baseline gap-0.5">
        <span className="text-[20px] font-extrabold">{value}</span>
        {unit ? (
          <span className="text-[12px] font-semibold text-go-muted">{unit}</span>
        ) : null}
      </div>
      <div className="text-[12px] text-go-muted">{label}</div>
    </div>
  )
}
