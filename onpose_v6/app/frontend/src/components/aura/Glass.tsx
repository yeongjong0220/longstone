import { type HTMLAttributes } from "react"

import { cn } from "@/lib/utils"

type GlassProps = HTMLAttributes<HTMLDivElement> & {
  strong?: boolean
}

export function Glass({ strong, className, children, ...rest }: GlassProps) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-aura-glass-border backdrop-blur-xl",
        "shadow-[0_4px_24px_rgba(0,0,0,0.04)]",
        strong ? "bg-aura-glass-strong" : "bg-aura-glass",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}
