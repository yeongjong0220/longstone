import { type ReactNode } from "react"

import { cn } from "@/lib/utils"

type BottomSheetProps = {
  open: boolean
  onClose: () => void
  children: ReactNode
  className?: string
}

export function BottomSheet({ open, onClose, children, className }: BottomSheetProps) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex flex-col justify-end bg-black/65">
      <button
        type="button"
        aria-label="닫기"
        onClick={onClose}
        className="flex-1 cursor-default"
      />
      <div
        role="dialog"
        aria-modal="true"
        className={cn("rounded-t-3xl bg-white px-5 pb-10 pt-6", className)}
      >
        {children}
      </div>
    </div>
  )
}
