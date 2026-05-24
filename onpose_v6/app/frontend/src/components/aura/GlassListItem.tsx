import { type ReactNode } from "react"

import { cn } from "@/lib/utils"

type GlassListItemProps = {
  thumb?: ReactNode
  title: string
  description?: string
  value?: ReactNode
  onClick?: () => void
  divider?: boolean
  className?: string
}

/**
 * Glass 컨테이너 안에서 사용하는 리스트 행.
 * AURA `Glass strong` 카드 안에 여러 개 쌓아서 사용.
 */
export function GlassListItem({
  thumb,
  title,
  description,
  value,
  onClick,
  divider = true,
  className,
}: GlassListItemProps) {
  const Tag = onClick ? "button" : "div"
  return (
    <Tag
      type={onClick ? "button" : undefined}
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-3 px-4 py-[13px] text-left",
        divider && "border-b-[0.5px] border-black/[0.06] last:border-b-0",
        onClick && "hover:bg-black/[0.02] active:bg-black/[0.04]",
        className,
      )}
    >
      {thumb && <div className="h-11 w-11 flex-shrink-0 overflow-hidden rounded-xl">{thumb}</div>}
      <div className="min-w-0 flex-1">
        <p className="text-aura-ink truncate text-sm font-semibold">{title}</p>
        {description && <p className="text-aura-ter mt-px truncate text-xs">{description}</p>}
      </div>
      {value !== undefined && (
        <span className="text-aura-primary flex-shrink-0 text-sm font-bold">{value}</span>
      )}
    </Tag>
  )
}
