import { type ReactNode } from "react"

import { cn } from "@/lib/utils"

import { BackIcon } from "./icons"

type BackHeaderProps = {
  title: string
  onBack?: () => void
  action?: ReactNode
  className?: string
}

export function BackHeader({ title, onBack, action, className }: BackHeaderProps) {
  return (
    <header className={cn("flex items-center justify-between px-5 pt-2", className)}>
      <button
        type="button"
        onClick={onBack}
        aria-label="뒤로가기"
        className="flex h-8 w-8 items-center justify-center"
      >
        <BackIcon color="var(--color-aura-ink)" />
      </button>
      <span className="text-aura-ink text-base font-semibold">{title}</span>
      <div className="flex h-8 w-8 items-center justify-center">{action}</div>
    </header>
  )
}
