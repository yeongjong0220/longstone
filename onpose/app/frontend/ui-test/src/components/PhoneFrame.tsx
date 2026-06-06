import type { ReactNode } from "react"

import { cn } from "@/lib/cn"

/** 레퍼런스처럼 모서리가 둥근 풀블리드 폰 화면 프레임. */
export function PhoneFrame({
  children,
  label,
  className,
}: {
  children: ReactNode
  label?: string
  className?: string
}) {
  return (
    <div className="flex shrink-0 flex-col items-center gap-3">
      <div
        className={cn(
          "relative h-[640px] w-[296px] overflow-hidden rounded-[42px]",
          "shadow-[0_30px_80px_-24px_rgba(0,0,0,0.7)] ring-1 ring-white/5",
          className,
        )}
      >
        {children}
      </div>
      {label ? (
        <span className="text-xs font-medium tracking-wide text-white/40">
          {label}
        </span>
      ) : null}
    </div>
  )
}
