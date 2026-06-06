// 브랜드/UI 글리프를 전부 인라인 SVG 로 구현 (외부 아이콘 라이브러리 의존 0).
// 본 프로젝트의 components/aura/icons.tsx 패턴과 동일한 접근.
import type { SVGProps } from "react"

type IconProps = SVGProps<SVGSVGElement>

const stroke = (props: IconProps): IconProps => ({
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  ...props,
})

/* ── 상태바 ───────────────────────────── */
export function SignalIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 20 14" fill="currentColor" {...props}>
      <rect x="0" y="9" width="3.4" height="5" rx="1" />
      <rect x="5.5" y="6" width="3.4" height="8" rx="1" />
      <rect x="11" y="3" width="3.4" height="11" rx="1" />
      <rect x="16.5" y="0" width="3.4" height="14" rx="1" />
    </svg>
  )
}

export function WifiIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 18 14" fill="currentColor" {...props}>
      <path d="M9 14a2 2 0 100-4 2 2 0 000 4z" />
      <path d="M2.4 6.1a9.4 9.4 0 0113.2 0l-1.7 1.7a7 7 0 00-9.8 0L2.4 6.1z" />
      <path d="M0 3.7a12.8 12.8 0 0118 0l-1.7 1.7a10.4 10.4 0 00-14.6 0L0 3.7z" />
    </svg>
  )
}

export function BatteryIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 27 14" fill="none" {...props}>
      <rect
        x="0.6"
        y="0.6"
        width="22"
        height="12.8"
        rx="3.6"
        stroke="currentColor"
        strokeOpacity="0.45"
        strokeWidth="1.1"
      />
      <rect x="2.3" y="2.3" width="18" height="9.4" rx="2.2" fill="currentColor" />
      <path
        d="M24.6 4.6c1.3.5 1.3 4.3 0 4.8V4.6z"
        fill="currentColor"
        fillOpacity="0.5"
      />
    </svg>
  )
}

/* ── 네비/방향 ─────────────────────────── */
export function ArrowLeftIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <path d="M19 12H5" />
      <path d="M12 19l-7-7 7-7" />
    </svg>
  )
}

export function ChevronRightIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <path d="M9 18l6-6-6-6" />
    </svg>
  )
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <path d="M6 9l6 6 6-6" />
    </svg>
  )
}

export function ArrowDownLeftIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <path d="M17 7L7 17" />
      <path d="M17 17H7V7" />
    </svg>
  )
}

export function ArrowUpRightIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <path d="M7 17L17 7" />
      <path d="M7 7h10v10" />
    </svg>
  )
}

/* ── 컨텐츠 글리프 ───────────────────────── */
export function BarsIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <rect x="3" y="13" width="4" height="8" rx="1.4" />
      <rect x="10" y="8" width="4" height="13" rx="1.4" />
      <rect x="17" y="4" width="4" height="17" rx="1.4" />
    </svg>
  )
}

export function FootprintsIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M6.5 2.3c1.6 0 2.5 1.7 2.5 3.9 0 1.7-.4 3.3-.4 4.6 0 1.1-.9 1.7-2.1 1.7s-2.1-.7-2.1-1.9c0-1.3.1-2 .1-4 0-2.5.8-4.3 2-4.3z" />
      <path d="M4.6 14.6c.3-.9 1.3-1.3 2.7-1.1 1.6.2 2.3 1 2.1 2.4-.2 1.3-.3 2-.6 3.1-.3 1.2-1.2 1.8-2.4 1.5-1.1-.3-1.7-1.1-1.5-2.3.1-.7-.5-2.3-.3-3.6z" />
      <path d="M17.5 5c-1.6 0-2.5 1.7-2.5 3.9 0 1.7.4 3.3.4 4.6 0 1.1.9 1.7 2.1 1.7s2.1-.7 2.1-1.9c0-1.3-.1-2-.1-4 0-2.5-.8-4.3-2-4.3z" />
      <path d="M19.4 17.3c-.3-.9-1.3-1.3-2.7-1.1-1.6.2-2.3 1-2.1 2.4.2 1.3.3 1.7.6 2.8.1.5 1.2.8 2.4.5 1.1-.3 1.7-1.1 1.5-2.3-.1-.7.5-1 .3-2.3z" />
    </svg>
  )
}

export function ClockIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </svg>
  )
}

export function RouteIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <circle cx="6" cy="19" r="2.5" />
      <circle cx="18" cy="5" r="2.5" />
      <path d="M8.5 19H14a3.5 3.5 0 000-7H10a3.5 3.5 0 010-7h5.5" />
    </svg>
  )
}

export function WaveformIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <rect x="2" y="10" width="2.4" height="4" rx="1.2" />
      <rect x="6.4" y="7" width="2.4" height="10" rx="1.2" />
      <rect x="10.8" y="3.5" width="2.4" height="17" rx="1.2" />
      <rect x="15.2" y="7" width="2.4" height="10" rx="1.2" />
      <rect x="19.6" y="10" width="2.4" height="4" rx="1.2" />
    </svg>
  )
}

export function SunIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <circle cx="12" cy="12" r="4.5" />
      <path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" />
    </svg>
  )
}

export function BottleIcon(props: IconProps) {
  return (
    <svg {...stroke(props)}>
      <path d="M9 2h6v2.2a3 3 0 00.8 2L17 9.5a4 4 0 01.9 2.6V19a3 3 0 01-3 3H9a3 3 0 01-3-3v-6.9a4 4 0 01.9-2.6L8.2 6.2A3 3 0 009 4.2V2z" />
      <path d="M6 13h12" />
    </svg>
  )
}

export function LayersIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <rect x="3" y="3" width="13" height="13" rx="4" opacity="0.45" />
      <rect x="6" y="6" width="13" height="13" rx="4" opacity="0.7" />
      <rect x="9" y="9" width="12" height="12" rx="4" />
    </svg>
  )
}
