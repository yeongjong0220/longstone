import { useState } from "react"

import { ArrowLeftIcon, LayersIcon } from "@/components/icons"
import { StatusBar } from "@/components/StatusBar"
import { HighlightChip, RoundButton } from "@/components/ui"
import { cn } from "@/lib/cn"

const LEVELS = ["Beginner", "Intermediate", "Advanced", "Athletic"] as const

/** 화면 2 — 피트니스 레벨 선택. 다크 그라디언트 + 2x2 옵션 + Next. */
export function FitnessLevel() {
  const [selected, setSelected] = useState<string>("Beginner")

  return (
    <div
      className="relative flex h-full w-full flex-col px-6 pb-7 text-white"
      style={{
        background:
          "radial-gradient(135% 95% at 50% 122%, #5a4bb6 0%, #221c42 40%, #0b0a11 76%)",
      }}
    >
      <StatusBar />
      <HighlightChip className="absolute left-4 top-2 z-10" />
      <RoundButton
        className="absolute left-4 top-[46px] z-10"
        aria-label="뒤로"
      >
        <ArrowLeftIcon className="h-[18px] w-[18px]" />
      </RoundButton>

      <div className="mt-[78px] flex justify-center">
        <LayersIcon className="h-[72px] w-[72px] text-go-primary-soft drop-shadow-[0_14px_30px_rgba(111,99,236,0.55)]" />
      </div>

      <h1 className="mt-9 text-[34px] font-extrabold leading-[1.04] tracking-tight">
        Current
        <br />
        fitness level?
      </h1>

      <div className="mt-7 grid grid-cols-2 gap-3">
        {LEVELS.map((level) => {
          const active = level === selected
          return (
            <button
              key={level}
              type="button"
              onClick={() => setSelected(level)}
              className={cn(
                "rounded-[18px] py-4 text-[15px] font-semibold transition active:scale-[0.97]",
                active
                  ? "bg-white text-go-primary shadow-lg shadow-black/20"
                  : "border border-white/15 bg-white/[0.06] text-white/65",
              )}
            >
              {level}
            </button>
          )
        })}
      </div>

      <div className="flex-1" />

      <button
        type="button"
        className="w-full rounded-full bg-white py-4 text-[16px] font-bold text-go-primary shadow-xl shadow-black/25 transition active:scale-[0.98]"
      >
        Next
      </button>
    </div>
  )
}
