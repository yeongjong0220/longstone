import { GoLogo } from "@/components/GoLogo"
import {
  ArrowUpRightIcon,
  BarsIcon,
  ChevronRightIcon,
  ClockIcon,
  FootprintsIcon,
  RouteIcon,
  SunIcon,
  WaveformIcon,
} from "@/components/icons"
import { StatusBar } from "@/components/StatusBar"
import { DashedDivider, HighlightChip, RoundButton } from "@/components/ui"
import { cn } from "@/lib/cn"

/** 화면 3 — 홈 대시보드. */
export function Home() {
  return (
    <div className="relative h-full w-full overflow-hidden bg-go-primary text-white">
      <StatusBar />
      <HighlightChip className="absolute left-4 top-2 z-20" />

      <div className="no-scrollbar h-[calc(100%-48px)] overflow-y-auto px-6 pb-32">
        {/* 인사 + 차트 버튼 */}
        <header className="flex items-start justify-between pt-3">
          <div>
            <p className="text-[17px] text-white/85">Good afternoon.</p>
            <h1 className="mt-1 flex items-center gap-1.5 text-[22px] font-extrabold">
              <FootprintsIcon className="h-5 w-5" />
              Let&rsquo;s step it up!
            </h1>
          </div>
          <RoundButton className="bg-white/15" aria-label="통계">
            <BarsIcon className="h-4 w-4" />
          </RoundButton>
        </header>

        <DashedDivider className="my-4" />

        {/* 프로모 카드 */}
        <div className="flex items-center gap-3 rounded-3xl bg-white/[0.12] p-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-white">
            <GoLogo
              variant="solid"
              color="#5a4be6"
              letterColor="#ffffff"
              className="w-8"
            />
          </div>
          <p className="text-[12.5px] leading-snug text-white/85">
            Start your journey today!{" "}
            <span className="font-bold text-white">first week&rsquo;s on us.</span>{" "}
            After that? Just{" "}
            <span className="font-bold text-white">$19.99 a year.</span> Yep, less
            than your monthly coffee habit.
          </p>
        </div>

        <DashedDivider className="my-4" />

        {/* 오늘의 목표 */}
        <div className="flex items-center justify-between">
          <h2 className="text-[17px] font-bold">Today&rsquo;s Goal</h2>
          <button
            type="button"
            className="flex items-center gap-0.5 text-[13px] font-medium text-white/80"
          >
            View Plan
            <ChevronRightIcon className="h-4 w-4" />
          </button>
        </div>

        <p className="mt-3 text-[14px] font-semibold text-white/90">Moderate Walk</p>
        <div className="mt-2 flex items-center gap-4 text-[13px] text-white/80">
          <span className="flex items-center gap-1.5">
            <RouteIcon className="h-4 w-4" /> 0mi
          </span>
          <span className="flex items-center gap-1.5">
            <ClockIcon className="h-4 w-4" /> 60min
          </span>
          <span className="flex items-center gap-1.5">
            <FootprintsIcon className="h-4 w-4" /> 7,500
          </span>
        </div>

        <ProgressTicks />

        {/* 골든아워 카드 */}
        <div className="relative mt-5 overflow-hidden rounded-[28px] bg-go-yellow p-5 text-go-ink">
          <SunIcon className="absolute right-5 top-5 h-5 w-5 text-go-ink/70" />
          <h3 className="text-[22px] font-extrabold leading-[1.1]">
            Walk during the
            <br />
            Golden hour
          </h3>
          <p className="mt-2 max-w-[180px] text-[12.5px] leading-snug text-go-ink/70">
            Turn on location to know the best time for your walk.
          </p>
          <button
            type="button"
            className="mt-4 rounded-full bg-go-ink px-6 py-2.5 text-[13px] font-semibold text-white transition active:scale-95"
          >
            Continue
          </button>
          <GoldenHourGauge className="pointer-events-none absolute -bottom-3 right-1 w-32 opacity-90" />
        </div>
      </div>

      {/* 하단 플로팅 네비 */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center pb-5">
        <nav className="pointer-events-auto flex items-center gap-2 rounded-full bg-white/85 p-2 shadow-2xl shadow-black/30 ring-1 ring-black/5 backdrop-blur-md">
          <button
            type="button"
            className="flex h-11 w-11 items-center justify-center rounded-full text-go-ink/70"
            aria-label="걸음"
          >
            <FootprintsIcon className="h-5 w-5" />
          </button>
          <button
            type="button"
            className="flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-go-ink shadow-md"
          >
            <WaveformIcon className="h-4 w-4 text-go-primary" />
            <span className="text-sm font-semibold">Plan</span>
          </button>
          <button
            type="button"
            className="flex h-11 w-11 items-center justify-center rounded-full bg-go-primary text-white"
            aria-label="더보기"
          >
            <ArrowUpRightIcon className="h-5 w-5" />
          </button>
        </nav>
      </div>
    </div>
  )
}

function ProgressTicks() {
  return (
    <div className="mt-3 flex items-center justify-between">
      {Array.from({ length: 42 }).map((_, i) => (
        <span
          key={i}
          className={cn(
            "w-[2px] rounded-full",
            i < 2 ? "h-5 bg-white" : "h-4 bg-white/30",
          )}
        />
      ))}
    </div>
  )
}

function GoldenHourGauge({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 140 92" className={className} aria-hidden>
      <defs>
        <linearGradient id="gh" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#FF8A3D" />
          <stop offset="1" stopColor="#FFC93D" />
        </linearGradient>
      </defs>
      <g fill="none" stroke="url(#gh)" strokeLinecap="round">
        <path d="M10 86 A60 60 0 0 1 130 86" strokeWidth="5" opacity="0.3" />
        <path d="M26 86 A44 44 0 0 1 114 86" strokeWidth="5" opacity="0.55" />
        <path d="M42 86 A28 28 0 0 1 98 86" strokeWidth="5" opacity="0.85" />
      </g>
      <circle cx="70" cy="86" r="14" fill="url(#gh)" />
    </svg>
  )
}
