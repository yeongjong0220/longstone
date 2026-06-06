import { cn } from "@/lib/cn"

import { BatteryIcon, SignalIcon, WifiIcon } from "./icons"

/** iOS 스타일 상태바 (목업용). tone 으로 글리프 색 전환. */
export function StatusBar({
  tone = "light",
  className,
}: {
  tone?: "light" | "dark"
  className?: string
}) {
  const color = tone === "light" ? "text-white" : "text-go-ink"
  return (
    <div
      className={cn(
        "flex h-12 items-center justify-between px-7 pt-2 select-none",
        color,
        className,
      )}
    >
      <span className="text-[15px] font-semibold tracking-tight">9:41</span>
      <div className="flex items-center gap-1.5">
        <SignalIcon className="h-3 w-[18px]" />
        <WifiIcon className="h-3 w-[16px]" />
        <BatteryIcon className="h-3.5 w-[25px]" />
      </div>
    </div>
  )
}
