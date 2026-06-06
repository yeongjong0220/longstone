import { GoLogo } from "@/components/GoLogo"
import { StatusBar } from "@/components/StatusBar"
import { HighlightChip } from "@/components/ui"

/** 화면 1 — 스플래시. 풀블리드 바이올렛 + 중앙 GO 로고. */
export function Splash() {
  return (
    <div className="relative flex h-full w-full flex-col bg-go-primary">
      <StatusBar />
      <HighlightChip className="absolute left-4 top-2 z-10" />

      <div className="flex flex-1 items-center justify-center">
        <GoLogo variant="outline" className="w-[150px]" />
      </div>

      <div className="flex flex-col items-center gap-1.5 pb-12 text-white/55">
        <span className="text-[15px] font-extrabold tracking-[0.25em]">
          GO Club
        </span>
        <span className="flex items-center gap-1 text-[9px] font-semibold tracking-[0.32em]">
          POWERED BY
          <span className="inline-flex h-3 w-3 items-center justify-center rounded-full bg-white/55 text-[7px] font-black leading-none text-go-primary">
            f
          </span>
        </span>
      </div>
    </div>
  )
}
