import { useEffect, useState } from "react"

/**
 * 가로 전용 화면(코칭/카메라)에서 기기가 세로일 때 "가로로 돌려주세요" 안내를 띄운다.
 * 방향 기준이라 실사용상 폰을 세로로 들었을 때만 노출되고, 가로로 돌리면 사라진다.
 * 데스크탑(가로 창)에서는 표시되지 않는다.
 */
export default function RotateHint() {
  const [portrait, setPortrait] = useState<boolean>(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(orientation: portrait)").matches,
  )

  useEffect(() => {
    const mql = window.matchMedia("(orientation: portrait)")
    const onChange = () => setPortrait(mql.matches)
    onChange()
    mql.addEventListener("change", onChange)
    return () => mql.removeEventListener("change", onChange)
  }, [])

  if (!portrait) return null

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-white px-8">
      <div className="flex flex-col items-center gap-5 text-center">
        <svg
          width="72"
          height="72"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#3182F6"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="animate-pulse"
          aria-hidden
        >
          <rect x="7" y="2" width="10" height="20" rx="2.5" />
          <path d="M3 9a9 9 0 0 1 9-7" />
          <polyline points="2 5 3 9 7 8" />
        </svg>
        <div>
          <p className="text-aura-ink text-xl font-bold">가로로 돌려주세요</p>
          <p className="text-aura-ter mt-1.5 text-sm break-keep">
            폰을 가로로 돌리면 코칭 화면이 시작돼요.
          </p>
        </div>
      </div>
    </div>
  )
}
