import { useEffect, useRef } from "react"

import type { Landmark } from "@/hooks/usePoseDetector"
import { POSE_CONNECTIONS } from "@/lib/poseConnections"

interface PoseOverlayProps {
  landmarksRef: React.RefObject<Landmark[]>
  /** Mirror horizontally to match front-camera preview. */
  mirror?: boolean
  className?: string
}

/**
 * 가려짐 강건 PoseOverlay
 *  - VISIBILITY_THR=0.60 — 가려진 관절은 본 스켈레톤에서 제외 (phantom 억제)
 *  - 화면(0,1) 밖 좌표는 무시 — MediaPipe outlier 좌표 제거
 *  - 직전 frame 대비 큰 점프 관절은 한 프레임 그리지 않음 (jitter 억제)
 *  - 가려진 부분은 회색 점선 으로 표시 → 사용자에게 "추적 손실" 시각화 ("팔 옆 phantom" 대신)
 */

const VISIBILITY_THR = 0.6
const MAX_JUMP_NORM = 0.18
const OUT_OF_FRAME_MARGIN = 0.04

export default function PoseOverlay({
  landmarksRef,
  mirror = true,
  className,
}: PoseOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const prevLmsRef = useRef<Landmark[] | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let raf = 0
    let cancelled = false

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      const w = Math.max(1, Math.floor(rect.width * dpr))
      const h = Math.max(1, Math.floor(rect.height * dpr))
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
    }

    const draw = () => {
      if (cancelled) return
      resize()
      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)
      const lms = landmarksRef.current
      const prev = prevLmsRef.current

      if (lms && lms.length > 0) {
        const xOf = (lm: Landmark) => (mirror ? 1 - lm[0] : lm[0]) * w
        const yOf = (lm: Landmark) => lm[1] * h

        const confident: boolean[] = new Array(lms.length).fill(false)
        for (let i = 0; i < lms.length; i++) {
          const lm = lms[i]
          if (!lm) continue
          if (lm[3] < VISIBILITY_THR) continue
          if (lm[0] < -OUT_OF_FRAME_MARGIN || lm[0] > 1 + OUT_OF_FRAME_MARGIN) continue
          if (lm[1] < -OUT_OF_FRAME_MARGIN || lm[1] > 1 + OUT_OF_FRAME_MARGIN) continue
          if (prev && prev[i]) {
            const dx = lm[0] - prev[i][0]
            const dy = lm[1] - prev[i][1]
            const jump = Math.sqrt(dx * dx + dy * dy)
            if (jump > MAX_JUMP_NORM) continue
          }
          confident[i] = true
        }

        // 1) 가려진 bone hint — 회색 점선 (한쪽만 보이는 bone)
        ctx.save()
        ctx.strokeStyle = "rgba(160, 160, 160, 0.45)"
        ctx.lineWidth = Math.max(1.2, w / 540)
        ctx.setLineDash([4, 6])
        ctx.beginPath()
        for (const [a, b] of POSE_CONNECTIONS) {
          const la = lms[a]
          const lb = lms[b]
          if (!la || !lb) continue
          if (confident[a] && confident[b]) continue
          if (!(confident[a] || confident[b])) continue
          ctx.moveTo(xOf(la), yOf(la))
          ctx.lineTo(xOf(lb), yOf(lb))
        }
        ctx.stroke()
        ctx.restore()

        // 2) 본 스켈레톤 — 양쪽 confident 한 bone 만 초록 굵게
        ctx.strokeStyle = "rgba(110, 231, 183, 0.92)"
        ctx.lineWidth = Math.max(2.4, w / 320)
        ctx.beginPath()
        for (const [a, b] of POSE_CONNECTIONS) {
          if (!confident[a] || !confident[b]) continue
          ctx.moveTo(xOf(lms[a]), yOf(lms[a]))
          ctx.lineTo(xOf(lms[b]), yOf(lms[b]))
        }
        ctx.stroke()

        // 3) 관절 점 — confident 만 (가려진 관절은 표시 안 함 → phantom 없음)
        ctx.fillStyle = "rgba(56, 189, 248, 0.95)"
        const r = Math.max(3, w / 220)
        for (let i = 0; i < lms.length; i++) {
          if (!confident[i]) continue
          ctx.beginPath()
          ctx.arc(xOf(lms[i]), yOf(lms[i]), r, 0, Math.PI * 2)
          ctx.fill()
        }

        prevLmsRef.current = lms.map((l) => [l[0], l[1], l[2], l[3]]) as Landmark[]
      } else {
        prevLmsRef.current = null
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)

    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
    }
  }, [landmarksRef, mirror])

  return <canvas ref={canvasRef} className={className} />
}
