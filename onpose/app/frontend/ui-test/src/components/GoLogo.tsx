import { cn } from "@/lib/cn"

/** GO Club 워드마크 — 겹친 두 둥근 사각형 + GO 글자.
 *  outline: 속 빈 흰 로고(스플래시) / solid: 채운 배지(카드·시트). */
export function GoLogo({
  variant = "outline",
  color = "#ffffff",
  letterColor = "#5a4be6",
  className,
}: {
  variant?: "outline" | "solid"
  color?: string
  letterColor?: string
  className?: string
}) {
  const isOutline = variant === "outline"
  return (
    <svg
      viewBox="0 0 156 88"
      className={cn(className)}
      role="img"
      aria-label="GO"
    >
      <g
        fill={isOutline ? "none" : color}
        stroke={isOutline ? color : "none"}
        strokeWidth={isOutline ? 7 : 0}
      >
        <rect x="4" y="6" width="86" height="76" rx="30" />
        <rect x="66" y="6" width="86" height="76" rx="36" />
      </g>
      <g
        fontFamily='"Geist Variable", system-ui, sans-serif'
        fontWeight={800}
        fontSize={44}
        textAnchor="middle"
        fill={isOutline ? color : letterColor}
      >
        <text x="45" y="60">
          G
        </text>
        <text x="111" y="60">
          O
        </text>
      </g>
    </svg>
  )
}
