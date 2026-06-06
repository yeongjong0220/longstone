import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"

import { useOrientationLock } from "@/hooks/useOrientationLock"
import { primeSpeech } from "@/hooks/useSpeechFeedback"
import { listExercises, type Exercise } from "@/lib/api"

export default function Home() {
  const navigate = useNavigate()
  useOrientationLock("portrait")
  const [exercises, setExercises] = useState<Exercise[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listExercises()
      .then((list) => !cancelled && setExercises(list))
      .catch((e: Error) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [])

  const onSelect = (exerciseId: string) => {
    // 사용자 제스처(탭) 시점에 TTS 엔진 워밍업 — Chrome 첫 발화 지연 회피
    primeSpeech()
    navigate("/exercise", { state: { exerciseId } })
  }

  return (
    <main className="min-h-svh flex flex-col gap-6 bg-white px-5 pt-10 pb-6">
      <header>
        <h1 className="text-aura-ink text-[30px] font-extrabold tracking-tight">운동 선택</h1>
        <p className="text-aura-sec mt-1.5 text-sm">오늘 어떤 운동을 할까요?</p>
      </header>

      <div className="flex flex-col gap-3">
        {error && (
          <div role="alert" className="rounded-[22px] border border-[#FFD9CC] bg-[#FFF1EC] px-4 py-3">
            <p className="text-[#D2451E] text-sm whitespace-pre-line break-keep">{error}</p>
          </div>
        )}

        {!error && exercises === null && (
          <div className="rounded-[22px] border border-[#E5E8EB] bg-[#F2F4F6] px-4 py-3">
            <p className="text-aura-ter text-sm">불러오는 중…</p>
          </div>
        )}

        {exercises?.map((ex) => (
          <button
            key={ex.id}
            type="button"
            onClick={() => onSelect(ex.id)}
            className="group/card flex items-center gap-4 rounded-[22px] border border-[#E5E8EB] bg-white px-4 py-3.5 text-left shadow-[0_10px_28px_-18px_rgba(0,0,0,0.45)] transition-all active:scale-[0.98] active:bg-[#F7F8FF]"
          >
            {/* 운동 자세 일러스트 (public/images/{id}.png) */}
            <div className="relative h-20 w-28 flex-shrink-0 overflow-hidden rounded-2xl bg-[#F2F4F6] ring-1 ring-[#E5E8EB]">
              <img
                src={`/images/${ex.id}.png`}
                alt={ex.name}
                loading="lazy"
                className="absolute inset-0 h-full w-full object-cover"
                onError={(e) => {
                  const img = e.currentTarget as HTMLImageElement
                  img.style.display = "none"
                }}
              />
            </div>

            <div className="flex-1 min-w-0 text-left">
              <p className="text-aura-ink text-base font-bold truncate">{ex.name}</p>
              <p className="mt-0.5 text-[12px] font-medium text-aura-ter">자세 코칭 시작하기</p>
            </div>

            <span className="grid h-8 w-8 flex-shrink-0 place-items-center rounded-full bg-[#E6FB4D] transition-transform group-active/card:scale-90">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#15110A"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
              >
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </span>
          </button>
        ))}

        {/* 필라테스 운동 추가 — 자리표시 버튼 (기능 없음) */}
        <button
          type="button"
          aria-label="필라테스 운동 추가"
          className="flex min-h-[108px] items-center justify-center gap-2.5 rounded-[22px] border-2 border-dashed border-[#C9CDF5] bg-[#F7F8FF] px-4 py-3.5 text-aura-primary transition-all active:scale-[0.98] active:bg-[#EFF0FE]"
        >
          <span className="grid h-8 w-8 flex-shrink-0 place-items-center rounded-full bg-[#E6FB4D]">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#15110A"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
          </span>
          <span className="text-base font-bold">필라테스 운동 추가</span>
        </button>
      </div>
    </main>
  )
}
