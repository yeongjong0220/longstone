import { useEffect } from "react"
import { useNavigate } from "react-router-dom"

import LongstoneWordmark from "@/components/LongstoneWordmark"
import { useOrientationLock } from "@/hooks/useOrientationLock"

declare global {
  interface Window {
    __splashStart?: number
  }
}

const SPLASH_MS = 2000

export default function Splash() {
  const navigate = useNavigate()
  useOrientationLock("portrait")

  useEffect(() => {
    // 앱 시작(아이콘 탭)부터 총 ~1.5초. 정적 스플래시가 JS 로드 동안 이미 떠 있었으므로
    // 경과 시간을 빼서 남은 시간만 더 기다린다(로드가 길었으면 즉시 홈으로).
    const start = window.__splashStart ?? Date.now()
    const remaining = Math.max(0, SPLASH_MS - (Date.now() - start))
    const t = window.setTimeout(() => navigate("/home", { replace: true }), remaining)
    return () => window.clearTimeout(t)
  }, [navigate])

  return (
    <main className="relative min-h-svh grid place-items-center overflow-hidden bg-[#4E45E6]">
      {/* 라임/보라 글로우 데코 */}
      <div className="pointer-events-none absolute -top-24 left-1/2 h-72 w-72 -translate-x-1/2 rounded-full bg-[#E6FB4D]/20 blur-[80px]" />
      <div className="pointer-events-none absolute -bottom-28 -right-16 h-72 w-72 rounded-full bg-[#7A6FF7]/40 blur-[90px]" />

      <div className="relative flex flex-col items-center gap-3">
        <LongstoneWordmark color="#FFFFFF" />
        <p className="text-sm tracking-wide text-white/65">on-device pose coach</p>
      </div>

      <footer className="absolute bottom-10 flex select-none items-center gap-2">
        <span className="flex items-center gap-2 text-[10px] font-bold tracking-[0.28em] text-white/40">
          POWERED BY
          <span className="grid h-4 w-4 place-items-center rounded-[5px] bg-[#E6FB4D]/80">
            <span className="h-1.5 w-1.5 rounded-[1px] bg-[#15110A]" />
          </span>
          LONGSTONE
        </span>
      </footer>
    </main>
  )
}
