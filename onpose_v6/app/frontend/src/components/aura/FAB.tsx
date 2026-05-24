import { type ButtonHTMLAttributes, type ReactNode } from "react"

import { cn } from "@/lib/utils"

import { PlusIcon } from "./icons"

type FABProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ReactNode
}

export function FAB({ icon, className, ...rest }: FABProps) {
  return (
    <button
      type="button"
      className={cn(
        "bg-aura-primary absolute bottom-24 right-[18px] z-40",
        "flex h-12 w-12 items-center justify-center rounded-full",
        "shadow-[0_6px_20px_rgba(79,70,229,0.35)]",
        "transition hover:brightness-110 active:scale-95",
        className,
      )}
      {...rest}
    >
      {icon ?? <PlusIcon color="#fff" size={20} />}
    </button>
  )
}
