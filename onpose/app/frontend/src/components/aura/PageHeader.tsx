import { type ReactNode } from "react"

import { cn } from "@/lib/utils"

type PageHeaderProps = {
  title: string
  subtitle?: string
  action?: ReactNode
  className?: string
}

export function PageHeader({ title, subtitle, action, className }: PageHeaderProps) {
  return (
    <header className={cn("px-5 pt-2", className)}>
      <div className="flex items-center justify-between">
        <span className="font-logo text-aura-ink text-[32px] leading-none">{title}</span>
        {action}
      </div>
      {subtitle && <p className="text-aura-ter mt-1 text-[13px]">{subtitle}</p>}
    </header>
  )
}
