import { type HTMLAttributes } from "react"

import { cn } from "@/lib/utils"

type GlassProps = HTMLAttributes<HTMLDivElement> & {
  /** true면 보라 배경 위 반투명 글래스(흰 글자용), 기본은 흰색 카드(어두운 글자용) */
  strong?: boolean
}

export function Glass({ strong, className, children, ...rest }: GlassProps) {
  return (
    <div
      className={cn(
        strong
          ? "rounded-3xl border border-white/15 bg-white/12 backdrop-blur-xl"
          : "rounded-3xl border border-black/[0.04] bg-white",
        "shadow-[0_18px_40px_-18px_rgba(20,16,60,0.45)]",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}
