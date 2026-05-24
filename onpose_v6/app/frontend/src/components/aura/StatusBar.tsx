import { cn } from "@/lib/utils"

type StatusBarProps = {
  dark?: boolean
  className?: string
  time?: string
}

/**
 * 디바이스 mockup용 가짜 상태바 (9:41 + 신호/배터리).
 * 실제 PWA에선 OS 상태바가 표시되므로 디자인 갤러리/스크린샷 용도.
 */
export function StatusBar({ dark, className, time = "9:41" }: StatusBarProps) {
  const fg = dark ? "#fff" : "var(--color-aura-ink)"
  return (
    <div
      className={cn(
        "flex items-center justify-between px-7 pt-3.5 text-[15px] font-semibold",
        className,
      )}
      style={{ color: fg }}
    >
      <span>{time}</span>
      <div className="flex items-center gap-[5px]">
        <svg width="16" height="11" viewBox="0 0 16 11" aria-hidden>
          <rect x="0" y="1" width="3" height="10" rx=".5" fill={fg} opacity=".4" />
          <rect x="4.5" y="3" width="3" height="8" rx=".5" fill={fg} opacity=".6" />
          <rect x="9" y=".5" width="3" height="10.5" rx=".5" fill={fg} opacity=".8" />
          <rect x="13" y="3" width="3" height="8" rx=".5" fill={fg} />
        </svg>
        <div
          className="relative ml-0.5 h-[11px] w-6 rounded-[3px]"
          style={{ border: `1.2px solid ${dark ? "rgba(255,255,255,.5)" : "#aaa"}` }}
        >
          <div
            className="absolute left-0.5 top-0.5 h-[5.5px] w-[14px] rounded-[1.5px]"
            style={{ background: fg }}
          />
        </div>
      </div>
    </div>
  )
}
