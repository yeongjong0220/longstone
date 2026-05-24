type IconProps = {
  color?: string
  size?: number
  className?: string
}

const make = (path: React.ReactNode, defaultSize = 22) =>
  function Icon({ color = "currentColor", size = defaultSize, className }: IconProps) {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
      >
        {path}
      </svg>
    )
  }

export const HomeIcon = make(<path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z" />)
export const HomeFilledIcon = ({ color = "currentColor", size = 22, className }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color} stroke={color} strokeWidth="1.5" strokeLinejoin="round" className={className}>
    <path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z" />
  </svg>
)

export const GridIcon = make(
  <>
    <rect x="3" y="3" width="7.5" height="7.5" rx="1.5" />
    <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5" />
    <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5" />
    <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5" />
  </>,
)

export const ScanIcon = make(
  <>
    <path d="M7 3H5a2 2 0 00-2 2v2" />
    <path d="M17 3h2a2 2 0 012 2v2" />
    <path d="M7 21H5a2 2 0 01-2-2v-2" />
    <path d="M17 21h2a2 2 0 002-2v-2" />
    <circle cx="12" cy="12" r="3.5" />
  </>,
)

export const UserIcon = make(
  <>
    <circle cx="12" cy="8" r="3.5" />
    <path d="M6 21v-1a6 6 0 0112 0v1" />
  </>,
)

export const CameraIcon = make(
  <>
    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
    <circle cx="12" cy="13" r="4" />
  </>,
  24,
)

export const CalendarIcon = make(
  <>
    <rect x="3" y="4" width="18" height="18" rx="2" />
    <path d="M3 10h18" />
    <path d="M8 2v4" />
    <path d="M16 2v4" />
  </>,
)

export const PlusIcon = ({ color = "#fff", size = 18, className }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" className={className}>
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
)

export const MinusIcon = ({ color = "currentColor", size = 18, className }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" className={className}>
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
)

export const CheckIcon = ({ color = "currentColor", size = 18, className }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="4 12 10 18 20 6" />
  </svg>
)

export const BackIcon = ({ color = "currentColor", size = 20, className }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" className={className}>
    <path d="M15 18l-6-6 6-6" />
  </svg>
)

export const ChevronRightIcon = ({ color = "currentColor", size = 14, className }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" className={className}>
    <path d="M9 6l6 6-6 6" />
  </svg>
)

export const CloseIcon = ({ color = "currentColor", size = 20, className }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" className={className}>
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
)

export const ShareIcon = make(
  <>
    <path d="M4 12v8a2 2 0 002 2h12a2 2 0 002-2v-8" />
    <polyline points="16 6 12 2 8 6" />
    <line x1="12" y1="2" x2="12" y2="15" />
  </>,
  18,
)

export const LockIcon = make(
  <>
    <rect x="3" y="11" width="18" height="11" rx="2" />
    <path d="M7 11V7a5 5 0 0110 0v4" />
  </>,
  16,
)

export const ExternalIcon = make(
  <>
    <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" />
    <polyline points="15 3 21 3 21 9" />
    <line x1="10" y1="14" x2="21" y2="3" />
  </>,
  12,
)
