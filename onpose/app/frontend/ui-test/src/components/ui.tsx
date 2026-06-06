import type { ButtonHTMLAttributes, ReactNode } from "react"

import { cn } from "@/lib/cn"

/** 상단 좌측의 "Highlight" 주석 칩 (목업 캡쳐에 공통으로 보이는 요소). */
export function HighlightChip({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full bg-black/35 px-3.5 py-1.5",
        "text-[13px] font-semibold text-white backdrop-blur-sm",
        className,
      )}
    >
      Highlight
    </span>
  )
}

/** 원형 아이콘 버튼 (반투명 글래스). */
export function RoundButton({
  children,
  className,
  ...props
}: { children: ReactNode } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex h-9 w-9 items-center justify-center rounded-full",
        "bg-white/15 text-white backdrop-blur-sm transition active:scale-95",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}

/** 점선 구분선. */
export function DashedDivider({ className }: { className?: string }) {
  return (
    <div
      className={cn("h-px w-full border-t border-dashed border-white/30", className)}
    />
  )
}
