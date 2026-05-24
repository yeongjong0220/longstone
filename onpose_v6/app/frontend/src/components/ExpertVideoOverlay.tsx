import { useEffect, useState } from "react"

import { CloseIcon } from "@/components/aura"

type ExpertVideoOverlayProps = {
  exerciseId: string
  exerciseName: string
  onClose: () => void
}

export default function ExpertVideoOverlay({
  exerciseId,
  exerciseName,
  onClose,
}: ExpertVideoOverlayProps) {
  const [errored, setErrored] = useState(false)
  const src = `/videos/${exerciseId}.mp4`

  // 오버레이가 떠 있는 동안 뒤 페이지 스크롤 잠금
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      document.body.style.overflow = prev
    }
  }, [])

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`${exerciseName} 전문가 영상`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black"
    >
      {errored ? (
        <div className="px-6 text-center">
          <p className="text-base font-semibold text-white">전문가 영상 준비 중이에요</p>
          <p className="mt-2 text-sm text-white/70">{exerciseName} 영상은 곧 추가될 예정입니다.</p>
        </div>
      ) : (
        <video
          key={src}
          src={src}
          autoPlay
          playsInline
          controls={false}
          onEnded={onClose}
          onError={() => setErrored(true)}
          className="max-h-full max-w-full"
        />
      )}

      <button
        type="button"
        onClick={onClose}
        aria-label="영상 닫기"
        className="absolute top-[max(env(safe-area-inset-top),12px)] right-3 flex items-center gap-1.5 rounded-full bg-white/15 px-3 py-2 text-sm font-medium text-white backdrop-blur-md hover:bg-white/25 active:bg-white/35"
      >
        <span>건너뛰기</span>
        <CloseIcon color="currentColor" size={16} />
      </button>
    </div>
  )
}
